Modern neural network architectures use structured linear transformations, such as low-rank matrices, sparse matrices, permutations, and the Fourier transform, to improve inference speed and reduce memory usage compared to general linear maps.

However, choosing which of the myriad structured transformations to use (and its associated parameterization) is a laborious task that requires trading off speed, space, and accuracy.

We consider a different approach: we introduce a family of matrices called kaleidoscope matrices (K-matrices) that provably capture any structured matrix with near-optimal space (parameter) and time (arithmetic operation) complexity.

We empirically validate that K-matrices can be automatically learned within end-to-end pipelines to replace hand-crafted procedures, in order to improve model quality.

For example, replacing channel shuffles in ShuffleNet improves classification accuracy on ImageNet by up to 5%.

Learnable K-matrices can also simplify hand-engineered pipelines---we replace filter bank feature computation in speech data preprocessing with a kaleidoscope layer, resulting in only 0.4% loss in accuracy on the TIMIT speech recognition task.

K-matrices can also capture latent structure in models: for a challenging permuted image classification task, adding a K-matrix to a standard convolutional architecture can enable learning the latent permutation and improve accuracy by over 8 points.

We provide a practically efficient implementation of our approach, and use K-matrices in a Transformer network to attain 36% faster end-to-end inference speed on a language translation task.

Structured linear maps are fundamental and ubiquitous in modern machine learning.

Their efficiency in speed (fast algorithms) and space (few parameters) can reduce computation and memory usage.

They include fixed specialized transforms such as the discrete Fourier transform (DFT) and Hadamard transform used in signal processing (Cooley et al., 1969) , convolutions for image, language, and speech modeling (Gu et al., 2018) , and low-rank and sparse matrices for efficient storage and inference on edge devices (Yu et al., 2017) .

Forms of structure such as sparsity have been at the forefront of recent advances in ML (Frankle & Carbin, 2019) , and are critical for on-device and energy-efficient models, two application areas of tremendous recent interest (Tsidulko, 2019; Schwartz et al., 2019) .

There are a plethora of classes of structured linear maps, each with a significantly different representation, algorithm, and implementation.

They have different tradeoffs in terms of inference speed, training speed, and accuracy, and the conventional wisdom is that no one class works uniformly well across all applications.

As a result, ML practitioners currently hand-pick specific classes of structured linear maps for each of their applications.

This is a difficult and labor-intensive task.

Ideally, these problems should be addressed with a universal representation for structured linear maps: (i) Such a parameterization should be expressive enough to capture important classes of structure, with a nearly tight parameter count and runtime: the space required to represent the linear map should be close to optimal, and the resulting algorithm for matrix vector multiplication should be close to the fastest possible algorithm.

(ii) The parameterization should be differentiable in order to be learned as a component of end-to-end ML pipelines, enabling it to easily be used as a drop-in replacement for manually engineered structured components. (iii) The parameterization should admit practically efficient algorithms for training and inference, in terms of both speed and memory.

Currently, no class of structured linear maps satisfies all of these criteria.

Most existing classes of structured matrices-such as the class of low-rank matrices-fail to tightly capture other important types of structure.

For example, the DFT has an efficient structured representation of size O(n log n), yet cannot be well-approximated by a low-rank transform of size n 2 .

Sparsity is another important type of structure; lots of exciting recent work has focused on the design of sparse neural networks.

For instance, sparse networks of comparable quality to their dense counterparts-yet an order of magnitude fewer parameters-may be created via pruning (Han et al., 2016) or by identifying "winning lottery tickets" (Frankle & Carbin, 2019) .

In parallel, recent theoretical results by De Sa et al. (2018) show that sparsity and the notion of structure in linear maps are fundamentally linked: any given matrix can be factored into a product of sparse matrices with total parameter count equal to the efficiency (i.e. minimum arithmetic circuit complexity) of the matrix.

In other words, the representation of linear maps as products of sparse matrices tightly captures all forms of structure.

Unfortunately, actually learning sparse representations is difficult, because it requires finding the matrices' sparsity patterns-a discrete, nondifferentiable search problem.

So, current methods for training sparse neural networks are either expensive (Frankle & Carbin, 2019) , or rely on highly handtuned heuristics for evolving the sparsity patterns throughout training (Dettmers & Zettlemoyer, 2019) .

By contrast, we propose a representation of linear maps as products of sparse matrices with specific predefined sparsity patterns (Section 2), and show that it does satisfy our desiderata: it retains the expressiveness of unstructured sparsity, while being differentiably learnable and efficient like other structured representations.

Concretely, our representation is based on products of a particular building block known as a butterfly matrix (Parker, 1995; Dao et al., 2019) ; we term such products kaleidoscope matrices (K-matrices for short).

1 (i) Our main theoretical contribution (Section 2.3) concerns the expressiveness of this representation: we show that any structured linear map (i.e. one that can be applied using s n 2 arithmetic operations) can be represented as a K-matrix, with a nearly tight number of parameters and algorithmic complexity (both on the order of s up to logarithmic factors).

(ii) The kaleidoscope representation is fully differentiable; thus, all the parameters of a K-matrix can be learned using standard optimization algorithms such as SGD. (iii) Because of their simple, regular structure, K-matrices are practical and easy to use.

We provide memory-and runtime-efficient implementations of K-matrix multiplication on CPU and GPU for training and inference, with a simple PyTorch interface.

We empirically validate that, due to their expressiveness, learnability, and efficiency, we can use K-matrices as a drop-in replacement for linear components in deep learning models.

In Section 3.1, we use K-matrices to replace hand-crafted structure in two different settings.

We simplify the six steps of filter bank computation in speech preprocessing into a single learnable K-matrix step, with only an 0.4% accuracy drop on the TIMIT speech recognition task.

We use K-matrices to replace channel shuffles in ShuffleNet, improving ImageNet classification accuracy by up to 5%.

In Section 3.2, we show that K-matrices can successfully recover latent structure; a K-matrix is used to learn latent permutations in a permuted image dataset (Permuted CIFAR), resulting in 9 points higher accuracy in a downstream CNN model.

In Section 3.3, we show that our efficient K-matrix multiplication implementation can be applied to speed up real-world tasks: we replace linear layers with K-matrices in a DynamicConv-Transformer network to attain 36% faster end-to-end inference speed with a 1.0 drop in BLEU score on the IWSLT14 German→English translation task.

We first present some background on the characterization of all structured matrices (i.e. those with subquadratic multiplication algorithms) as products of sparse factors, along with the definition of butterfly matrices.

We then propose a differentiable family of kaleidoscope matrices, composed of products of butterfly matrices, and prove their expressivity: all structured matrices can be represented in this form, with almost optimal parameter count and runtime.

Sparse factorization One method of constructing matrices with theoretically fast matrix-vector multiplication algorithms is as a product of sparse matrices, so that multiplication by an arbitrary vector has cost proportional to the total number of nonzeros (NNZ) of the matrices in the product.

Surprisingly, the converse is also true.

De Sa et al. (2018) introduce the concept of sparse product width (SPW), which roughly corresponds to the total NNZ in a factorization of a matrix, and show that it is an asymptotically optimal descriptor of the algorithmic complexity of matrix-vector multiplication (Bürgisser et al., 2013) .

We use a similar argument in the proof of our main theorem (Section 2.3).

However, attempting to learn such a factorization of a given matrix is difficult, as the sparsity constraint is non-continuous.

Moreover, because of the possibly irregular sparsity patterns, it is difficult to realize the theoretical speedups in practice (Gray et al., 2017; Gahvari et al., 2007) .

Butterfly matrices Butterfly matrices, encoding the recursive divide-and-conquer structure of the fast Fourier transform (FFT) algorithm, have long been used in numerical linear algebra (Parker, 1995; Li et al., 2015) and machine learning (Mathieu & LeCun, 2014; Jing et al., 2017; Munkhoeva et al., 2018; Dao et al., 2019; Choromanski et al., 2019) .

Here we define butterfly matrices, which we use as a building block for our hierarchy of kaleidoscope matrices.

Definition 2.1.

A butterfly factor of size k ≥ 2 (denoted as B k ) is a matrix of the form

where each D i is a k 2 × k 2 diagonal matrix.

We restrict k to be a power of 2.

Definition 2.2.

A butterfly factor matrix of size n with block size k (denoted as B (n) k ) is a block diagonal matrix of n k (possibly different) butterfly factors of size k:

Definition 2.3.

A butterfly matrix of size n (denoted as B (n) ) is a matrix that can be expressed as a product of butterfly factor matrices:

2 .

Equivalently, we may define B (n) recursively as a matrix that can be expressed in the following form:

(Note that [B

Using the building block of butterfly matrices, we formally define the kaleidoscope (BB * ) hierarchy and prove its expressiveness.

This serves as a fully differentiable alternative to products of sparse matrices (Section 2.1), with similar expressivity.

In Appendix J, we show where various common structured matrix classes are located within this hierarchy.

The building block for this hierarchy is the product of a butterfly matrix and the (conjugate) transpose of another butterfly matrix (which is simply a product of butterfly factors taken in the opposite order).

Figure 1 visualizes the sparsity patterns of the butterfly factors in BB * , where the red and blue dots represent the allowed locations of nonzero entries.

Definition 2.4 (Kaleidoscope hierarchy, kaleidoscope matrices).

• Define B as the set of all matrices that can be expressed as in the form B (n) (for some n).

• Define BB * as the set of matrices M of the form M = M 1 M

We now present our main theoretical result: the fact that general linear transformations, expressed as low-depth linear arithmetic circuits, are captured in the BB * hierarchy with low width.

Arithmetic circuits are commonly used to formalize algebraic algorithmic complexity (Bürgisser et al., 2013) ; we include a primer on this in Appendix M. The quantities of interest are the total number of gates in the circuit, representing the total number of steps required to perform the algorithm for a serial processor, and the depth, representing the minimum number of steps required for a parallel processor.

Theorem 1.

Let M be an n × n matrix such that multiplication of M times an arbitrary vector v can be represented as a linear arithmetic circuit with s total gates and depth

The representation of such a matrix M in the BB * hierarchy has O(ds log s) parameters and yields a O(ds log s) multiplication algorithm, compared to the O(s) parameters and runtime of the circuit representation.

To the best of our knowledge, the most general classes of efficient matrices that have been studied (De Sa et al., 2018) have depth d on the order of log n or poly log n. In these cases, the representation with K-matrices matches the best known bounds up to polylogarithmic factors.

The crux of the proof of Theorem 1 (shown in Appendix F) is an almost tight representation of any sparse matrix as a K-matrix (i.e. a product of butterfly matrices): any n × n sparse matrix with s

(Theorem 3, Appendix I).

We then leverage the expressivity result of products of sparse matrices to represent all arithmetic circuits (similar to the sparse product width result of De Sa et al. (2018) in Section 2.1) to complete the proof of Theorem 1.

This intermediate result is also a novel characterization of sparse matrices, to the best of our knowledge.

For a matrix with s NNZ, the kaleidoscope representation has O(s log n) parameters and runtime, instead of the optimal O(s) parameters and runtime.

We trade off an extra logarithmic factor in space and time for full differentiability (thanks to the fixed sparsity patterns in the representation).

The intuition behind this result is as follows: a sparse matrix with s NNZ can be written as a sum of s/n matrices each with at most n NNZ.

Any n × n matrix with at most n NNZ, up to permuting the rows and columns, is a product of two butterfly matrices (Lemma I.1).

Sorting networks (Knuth, 1997) imply that permutation matrices are in (BB * ) O(log n) , but we tighten the result to show that they are in fact in BB * (Theorem 2, Appendix G).

We thus obtain a kaleidoscope representation for each summand matrix with O(n log n) parameters.

By the addition closure property of the BB * hierarchy (Lemma H.5), each sparse matrix with s NNZ then has a kaleidoscope representation with O(s log n) parameters.

Tight representation for structured linear maps common in ML Even though Theorem 1 suggests that the kaleidoscope representation can be loose by logarithmic factors, many structured linear maps common in ML can be represented in this hierarchy with an optimal number of parameters and runtime compared to best known parameterizations, up to constant factors.

Appendix J includes several examples such as discrete transforms (the DFT, discrete cosine transform (DCT), discrete sine transform (DST), Hadamard transform), convolution (i.e. circulant matrix), Toeplitz matrices (Gray et al., 2006) , structured matrices for kernel approximation ((HD) 3 (Yu et al., 2016) ) and compact neural network design (Fastfood (Le et al., 2013) , ACDC (Moczulski et al., 2016) ).

There have been other large classes structured matrices proposed in the machine learning literature, such as Toeplitz-like (Sindhwani et al., 2015) or low displacement rank (LDR) (Thomas et al., 2018) , but to the best of our knowledge, they are not able to capture these common structures as tightly as K-matrices.

More detailed discussions are in Appendix A.

ReLU networks with low-depth structured weight matrices In Appendix L, we prove that finding an efficient circuit for a ReLU network can be reduced to finding efficient circuits for each of its weight matrices, with at most a constant factor greater size and run-time (i.e. number of gates).

We also show that ReLU networks with kaleidoscope weight matrices have VC dimension near-linear in the number of parameters, matching the bound for networks with unconstrained weight matrices (Bartlett et al., 1999; Harvey et al., 2017) and LDR (Thomas et al., 2018) .

This yields a corresponding sample complexity bound.

Orthogonal kaleidoscope hierarchy Orthogonal butterfly matrices are one commonly used variant due to their improved stability (Parker, 1995) , where each butterfly factor is constrained to be orthogonal:

C S −S C with C, S being diagonal and C 2 + S 2 = I. Similar to the BB * hierarchy, in Appendix K, we define the OBB hierarchy consisting of products of orthogonal butterfly matrices and diagonal matrices, and show that this hierarchy has the same expressiveness as the BB * hierarchy.

We validate three claims that suggest that kaleidoscopes are a promising technique to learn different types of structure in modern architectures.

1.

Section 3.1: for applications in speech and lightweight computer vision relying on highly hand-crafted structured transformations, we show that we can recover (and even improve) the quality of such architectures by simply replacing existing hand-structured components with K-matrices, with only a small overhead in memory and computation.

2.

In Section 3.2, for a challenging task with latent structure (Permuted CIFAR-10), a K-matrixbased relaxation of permutations is able to learn the right latent permutation, yielding 9 points better accuracy in a downstream CNN compared to standard RNN and CNN baselines used on such permuted image classification tasks.

3.

In Section 3.3, we show that, although not yet highly optimized, our current implementation of K-matrices can improve the inference throughput of DynamicConv Transformer, a state-ofthe-art fast machine translation model, by 36%, with only a relatively small drop in translation quality.

In all of the above applications, as K-matrices are fully differentiable, we simply train them jointly with the rest of the model using standard learning algorithms (such as SGD).

Full details for all of the experiments (precise architectures, hyperparameters, etc.) are in Appendix B.

We validate that kaleidoscope matrices can recover or improve on the performance of hand-crafted structure in ML models.

For example, a single learnable kaleidoscope layer can be used to replace the hand-engineered filter bank speech preprocessing pipeline with only 0.4% loss in accuracy on the TIMIT speech recognition task (Section 3.1.1).

Replacing channel shuffles in ShuffleNet with learnable K-matrices improves classification accuracy on ImageNet by up to 5.0% (Section 3.1.2).

We show that K-matrices can remove the need for hand-tuning by significantly simplifying speech recognition data preprocessing pipelines.

In particular, we can entirely replace the complex handcrafted MFSC featurization commonly used in speech recognition tasks with a fully learnable Figure 2 : Comparison of the standard MFSC featurization pipeline with our "kaleidoscope" pipeline.

kaleidoscope layer, with only 0.4% drop in accuracy on the TIMIT speech recognition benchmark.

Results are presented in Table 1 .

Our approach is competitive with the accuracy of standard models that use hand-crafted features, and significantly outperforms current approaches for learning from raw audio input.

Modern speech recognition models currently rely on carefully hand-crafted features extracted from the audio, which are then fed into an acoustic model.

By contrast, learning directly from the raw audio-i.e.

end-to-end learning from the audio waveform without any manual featurization-obviates the need for this complicated and often expensive preprocessing step.

There have been recent attempts to learn directly from raw audio, such as SincNet (Ravanelli & Bengio, 2018) ; however, they often rely on specialized architectures designed by domain experts.

Instead, we use a standard RNN speech recognition architecture, but use a learnable kaleidoscope layer to replace the featurization steps.

The baseline architecture takes as input filter bank (MFSC) features, which are a popular standard featurization for speech recognition (Paliwal, 1999) and involve several steps hand-crafted specifically for this domain.

These features are extracted from the raw audio waveform, and fed as the input into a Bi-LSTM model.

We significantly simplify this pipeline by replacing the featurization step with a trainable kaleidoscope layer that is trained end-to-end together with the Bi-LSTM.

The original pipeline and our modified kaleidoscope version are depicted in Figure 2 .

The computation of MFSC features involves a series of painstakingly hand-designed steps (further described in Appendix B.1), each involving their own hyperparameters: (i) the waveform is framed (split into chunks), (ii) dithering, (iii) pre-emphasis, (iv) the Hamming window is applied, (v) the FFT is applied and the power spectrum is computed, (vi) the result is mapped to the mel scale (which involves applying a linear transformation and then taking the logarithm), (vii) cepstral mean and variance normalization is applied.

We replace the last six steps (ii-vii) of this featurization with a kaleidoscope layer; specifically, after windowing, we multiply the input by a K-matrix, and then compute the logarithm of the power spectrum; the output is fed into the Bi-LSTM model.

We evaluate how K-matrices can improve the quality of hand-crafted, lightweight architectures for computer vision tasks, without the need for hand-tuning.

We select ShuffleNet (Zhang et al., 2018) , which is a state-of-the-art lightweight CNN architecture that uses a manually designed "channel shuffle" permutation matrix to improve performance.

By replacing this fixed permutation with a learnable K-matrix, we achieve up to 5% further improvement in classification accuracy, without hand-tuned components and with a modest space penalty of up to 10%.

Results are given in Table 2 .

Grouped convolution (Krizhevsky et al., 2012) is often used to reduce parameter count and speed up inference compared to standard convolution, but by default, channels in different groups cannot exchange information.

To remedy this, ShuffleNet uses a permutation matrix to shuffle the channels after each grouped convolution.

Zhao et al. (2019) propose to instead use the Hadamard transform before and after each grouped convolution to mix the channels.

In place of these hand-engineered solutions, we use a K-matrix before and after each grouped convolution, and learn these end-to-end together with the rest of the network.

As shown in Table 2 , across a range of sizes, replacing the channel shuffles with K-matrices results in improved performance at comparable parameter counts.

We show that K-matrices can be used in a challenging task for which existing classes of structured linear maps have not been found suitable.

We investigate the problem of image classification on a permuted image dataset (Permuted CIFAR-10).

This problem is challenging due to the discrete nature of learning the latent permutation of the dataset; we present a differentiable relaxation for this using a K-matrix as a key component.

Results are presented in Table 3 ; compared to methods that do not have a permutation learning step, our approach gets 9 points higher accuracy (84.4% to 93.6%), coming within 2 points of the accuracy on the un-permuted dataset (94.9%).

In this task, we use a permuted image classification dataset (Permuted CIFAR-10), wherein a fixed global permutation is applied to the pixels of every image in the original input set.

Typically, only fully-connected (FC) and recurrent models are applied to such datasets (Le et al., 2015) , because the permutation destroys locality in the image, presenting a difficulty for CNNs.

However, CNNs are much better-suited for standard image tasks.

We thus expect that learning the permutation and then applying a standard CNN should outperform these baselines.

As mentioned in Section 2, the kaleidoscope hierarchy provides a nearly tight parameterization of permutations; this makes them a natural fit for the permutation learning step.

Experimentally, a K-matrix is used to represent a distribution over permutations, which converges to a single permutation at the end of training.

The correct latent structure is learned by applying samples from this distribution to the permuted training images, and minimizing an auxiliary smoothness-based loss that encourages the reconstructed images to be more "natural" (i.e. vary smoothly pixel-to-pixel).

The learned permutation is evaluated by training a ResNet18 with the K-matrix permutation layer inserted at the beginning.

Full details of our approach are provided in Appendix B.3.

In Table 3 , we compare our approach to a ResNet18 without this extra K-matrix layer, a ResNet18 with an extra dense matrix at the beginning instead of a K-matrix, and other baselines.

As generic representations such as unstructured matrices do not have the requisite properties to fit in the pipeline, these baselines fail to effectively learn the latent permutation.

We emphasize that a K-matrix provides this ability to recover latent structure despite not being specialized for permutations.

Figure 3 describes the pipeline and displays examples of permuted and unpermuted images.

Figure 3: (a) (Left) Schematic describing permutation learning approach.

The inputs are multiplied by a K-matrix and then fed into a CNN, from which the classification loss is computed.

Separately, the input is permuted by a permutation matrix sampled from the distribution described by the Kmatrix, and a "smoothness" loss (Rudin et al., 1992) is computed from the result, as described in Appendix B.3.

(b) (Right) Left panel: original (unpermuted) images.

Center panel: the permuted versions.

Right panel: these images after then applying the permutation recovered by the K-matrix.

The K-matrix is able to nearly unscramble the images into their unpermuted versions.

We evaluate the inference speed benefit of using K-matrices on a real language translation model.

We choose the state-of-the-art DynamicConv Transformer translation model (Wu et al., 2019) , which has shown 20% inference speedup over the standard Transformer model, and replace dense matrices in the decoder's linear layers with K-matrices, which leads to a further 36% inference speedup (Table 4 ).

As outlined in Section 2.3, K-matrices admit a simple and fast O(n log n) matrix-vector multiplication algorithm.

We provide fast implementations of this algorithm in C++ and CUDA, with an interface to PyTorch (Paszke et al., 2017) , and use this implementation in our experiments.

We use K-matrices to replace all the linear layers in the decoder of DynamicConv (since 90% of inference time is spent in the decoder).

As shown in Table 4 , on the IWSLT-14 German-English translation task, this yields a 25% smaller model with 36% faster inference time on CPU, at the cost of 1.0 drop in BLEU score 4 (nearly matching SOTA performance of 2 years ago (Vaswani et al., 2017) ).

The majority (55%) of inference time is spent in matrix-vector multiplication; our implementation of K-matrix-vector multiplication is about 2x faster than the optimized implementation of dense matrix-vector multiplication in the Intel MKL library.

Direct comparisons of K-matrix multiplication with this and other highly-optimized routines such as the FFT are further detailed in Appendix C.

We address the problem of having to manually choose among the numerous classes of structured linear maps by proposing the universal (expressive, efficient, and learnable) family of kaleidoscope matrices.

We prove that K-matrices can represent any structured linear maps with near-optimal space and time complexity.

Empirical validations suggest that K-matrices are a promising way to employ structure in modern ML; they can be used to reduce the need for hand-engineering, capture challenging latent structure, and improve efficiency in models.

We are excited about future work on further hardware-optimized implementations of K-matrices, to fully realize the size and speed benefits of structured matrices on a broad array of real-world applications.

Structured linear maps such as the DFT, the Hadamard transform and convolution are a workhorse of machine learning, with diverse applications ranging from data preprocessing, random projection, featurization, to model compression.

For example, the DFT is a crucial step in the standard filter bank speech preprocessing pipeline (Jurafsky & Martin, 2014) .

Fast random projection and kernel approximation methods rely on the fast Hadamard transform (Le et al., 2013; Yu et al., 2016) and convolution (Yu et al., 2015) .

Large learnable classes of structured matrices such as Toeplitz-like matrices (Sindhwani et al., 2015) and low-displacement rank (LDR) matrices (Thomas et al., 2018) have been used for model compression.

However, despite their theoretical speedup, they lack efficient implementations, especially on GPUs.

Therefore their use has been confined to small models (e.g. single hidden layer neural nets) and small datasets (e.g. CIFAR-10).

Several classes of structured linear transforms are ubiquitous in modern deep learning architectures; particularly widespread examples include convolution and multiheaded attention.

Recently, attempts to impose sparsity on the neural network weights have been gaining traction.

State-of-the art approaches of this type typically accomplish this by pruning dense weights (either gradually during training (Zhu & Gupta, 2017) , or post-training (Han et al., 2016) ) or by training a dense network and then identifying "winning lottery tickets" -sparse subnetworks which may then be retrained from scratch with appropriate initialization (Frankle & Carbin, 2019) .

Importantly, these approaches start from a dense network, and therefore training is expensive.

There is also a more nascent line of work that aims to train unstructured sparse neural networks directly (Mocanu et al., 2018; Mostafa & Wang, 2019a; Dettmers & Zettlemoyer, 2019) .

These approaches maintain a constant network sparsity level throughout training, and use heuristics to evolve the sparsity pattern during training.

One drawback is that the indices of the nonzero entries need to be stored in addition to the entry values themselves, which increases the memory required to store the sparse weight tensors.

Another drawback is that these approaches to learn the sparsity pattern are based on intricate heuristics, which can be brittle.

We note that these heuristic sparsification techniques could potentially be combined with our approach, to further sparsify the K-matrix factors.

Numerous works focus on the problem of speech recognition from raw audio input, i.e. without manual featurization.

SincNet (Ravanelli & Bengio, 2018 ) is a CNN-based architecture parameterized with sinc functions such that the first convolutional layer imitates a band-pass filter.

Zeghidour et al. (2018) formulate a learnable version of a filter bank featurization; their filters are initialized as an approximation of MFSC features and then fine-tuned jointly with the rest of the model.

Sainath et al. (2015) proposed a powerful combined convolutional LSTM (CLDNN)-based model for learning from raw audio, using a large amount of training data.

The WaveNet generative architecture (van den Oord et al., 2016) , based on dilated convolutions, has been adapted to speech recognition and can be trained on raw audio.

Some other approaches that can learn from raw audio can be found in (Palaz et al., 2013; Collobert et al., 2016; Ghahremani et al., 2016) .

To our knowledge, the 14.6% PER achieved by our kaleidoscope + LSTM model on the TIMIT test set is the lowest error rate obtained by a model trained directly on the raw audio.

Permutation matrices find use in tasks such as matching and sorting.

Techniques to obtain posterior distribution over permutations have been developed, such as the exponential weights algorithm (Helmbold & Warmuth, 2009 ) and the Gumbel-Sinkhorn network (Mena et al., 2018) .

Classifying images with permuted pixels has been a standard task to benchmark the ability of RNNs to learn long range dependency.

Le et al. (2015) propose Permuted MNIST task where the model has to classify digit images with all the pixels permuted.

Many new RNN architectures, with unitary or orthogonal weight matrices to avoid gradient explosion or vanishing, have been proposed and tested on this task (Le et al., 2015; Arjovsky et al., 2016; Wisdom et al., 2016; Mhammedi et al., 2017; Trinh et al., 2018) .

Standard gated RNN architectures such as LSTM and GRU have also been found to be competitive with these new RNN architectures (Bai et al., 2018) Our baseline Bi-LSTM architecture is taken from the PyTorch-Kaldi repository.

5 This is a strong baseline model that, to the best of our knowledge, matches state-of-the-art performance for models that use a single type of input featurization (Ravanelli et al., 2019) .

The original Bi-LSTM model takes as input filter bank features.

These are computed as follows: (i) the waveform is framed (split into chunks of 25 ms each that overlap by 10 ms each), (ii) the waveform is dithered (zero-mean Gaussian random noise is added), (iii) pre-emphasis is applied to amplify high frequencies, (iv) the Hamming window function (Harris, 1978) is applied, (v) the FFT is applied, and the power spectrum of the resulting (complex-valued) output is computed, (vi) the power spectrum (which has dimension 512) is mapped to the "mel scale" (which is a scale intended to mimic human auditory perception (Stevens et al., 1937) ) by multiplication with a specific banded matrix of dimension 512 × 23, and the entrywise logarithm of the output is taken (the 23 outputs are called the filters), and (vii) cepstral mean and variance normalization (Liu et al., 1993 ) is applied.

Numerical hyperparameters of this procedure include the dither noise scale, the pre-emphasis coefficient, the Hamming window size, the number of mel filters, and more; we kept all these the same as the Kaldi/PyTorch-Kaldi defaults.

In contrast, our version of the model takes as input the raw waveform, split into chunks the same way as before but with no normalization, dithering, or other preprocessing, which is then fed into a complex-valued kaleidoscope [(BB * ) 2 ] matrix.

Similarly to the nonlinear steps in computing filter bank features, the logarithm of the power spectrum of the output (which has dimension 512) is then computed.

This output is fed into the Bi-LSTM; the Bi-LSTM and kaleidoscope layer are trained together in standard end-to-end fashion.

The Bi-LSTM architecture is not modified aside from changing the input dimension from 23 to 512; this (along with the ≈ 75K parameters in the kaleidoscope layer itself) results in approximately a 1.1M increase in the total number of parameters compared to the model that takes in MFSC features (a modest 8% relative increase).

Total training time for our kaleidoscope-based architecture is 7% greater than that required for the model that uses MFSC features, not counting the time required to precompute the MFSC features; the FLOPs for inference-time are approximately 15% greater (mostly due to the larger dimension of the input to the Bi-LSTM; the kaleidoscope layer accounts for less than 0.5% of the total FLOPs).

As baselines, we also compare to inserting other types of linear transformations before the Bi-LSTM: fixed linear transformations (such as the fixed FFT, or no transform at all [the identity]), trainable structured layers (low-rank, sparse, and circulant) and a trainable unstructured (dense) linear layer.

The kaleidoscope layer performs the best out of all such approaches.

Full results are given in Table 5 .

In our experiments, we grid search the initial learning rate for the "preprocessing layer" (if applicable) in {5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3}, and fix all other hyperparameters (including the initial learning rates for the other parts of the network) to their default values in the PyTorch-Kaldi repository.

The model and any preprocessing layers are trained end-to-end with the RMSProp optimizer for 24 epochs (as per the defaults in PyTorch-Kaldi).

For each model, we use the validation set to select the best preprocessing learning rate; while the final error rates are reported on the separate held-out test set.

For all structured matrix baselines except circulant (which always has n parameters for an n × n matrix), the number of parameters in the structured matrices is set to equal the number of parameters in the butterfly layer, while the unconstrained matrix is simply a standard dense complexvalued square matrix.

For all experiments with a trainable "preprocessing layer," we initialize the preprocessing matrix to represent the FFT (or approximate it as closely as possible [i.e. minimize the Frobenius error to the true FFT matrix], in the case of low-rank, sparse, and circulant), which we found to outperform random initialization.

As an additional experiment, we sought to investigate whether combining the hand-engineered MFSC featurization pipeline and a learnable kaleidoscope layer (instead of replacing the former with the latter) could lead to accuracy gains.

Specifically, in this experiment we first used the standard filter bank featurization pipeline described above, and trained end-to-end as usual.

Then, we replaced the FFT step with a K-matrix initialized to the FFT, and made the weights of the Hamming window function and the mel filter bank matrix learnable as well (similarly to (Zeghidour et al., 2018) ).

We fine-tuned the resulting architecture for an additional 10 epochs.

The final test PER% attained by this "hybrid" model is 13.9 ± 0.2; the model has 14.4M parameters-a negligible increase over the 14.3M in the original architecture.

Thus, by combining the manually encoded domain knowledge in the filter bank featurization and allowing this structure to be learnable rather than fixed, we are able to nearly match the state-of-the-art 13.8% accuracy on TIMIT.

(While this "hybrid" model certainly involves hand-engineering, the state-of-the-art results use a concatenation of three different speech audio featurizations-MFSC, MFCC, and fMLLR-as the neural network input, along with a customized RNN architecture (LiGRU) specifically designed for speech recognition, and thus require a more complicated pipeline that is arguably even more hand-crafted.)

ShuffleNet uses a permutation matrix to shuffle the channels after each grouped 1x1 convolution, sending the i-th channel to the (i mod g)-th group, where g is the total number of groups.

The architecture for each blocks is: 1x1 group conv → Batch norm, ReLU →

Permutation → 3x3 depthwise conv →

Batch norm → 1x1 group conv.

The permutation is fixed.

Zhao et al. (2019) propose to use the Hadamard transform before and after each grouped 1x1 convolution to mix the channels.

Note that the Hadamard transforms are placed before the batch norm and ReLU layer (unlike the permutation matrix in the original ShuffleNet design).

In particular, the architecture for each block is: Hadamard → 1x1 group conv → Hadamard → Batch norm, ReLU → 3x3 depthwise conv →

Batch norm → 1x1 group conv.

The Hadamard transform is fixed.

In our architecture, we use a kaleidoscope matrix in OBB (product of an orthogonal butterfly matrix, a diagonal matrix, and the transpose of another butterfly matrix) before and after each grouped 1x1 convolution.

We place the second K-matrix after batch norm and ReLU to more closely mimic the original ShuffleNet design.

The structure for each block is: K-matrix → 1x1 group conv → Batch norm, ReLU → K-matrix → 3x3 depthwise conv → Batch norm → 1x1 group conv.

The K-matrices are learned along with the rest of the network.

We evaluate the CNN architectures on the image classification task of the standard ImageNet dataset (Russakovsky et al., 2015) .

We use the standard data augmentation, training, and evaluation pipeline as in (Xie et al., 2017) .

We train with SGD on 8 GPUs for 90 epochs, with a total batch size of 2048 and initial learning rate 0.8.

For the 1.0 ShuffleNet g8 architecture, we reduce total batch size to 1792 to fit into GPU memory, and linear scale initial learning rate to 0.7.

Other hyperparameters (e.g. learning rate schedule, weight decay, etc.) are the same as the ShuffleNet paper (Zhang et al., 2018) .

We use the training script from Nvidia's deep learning examples repository.

In Table 6 , we report top-5 classification accuracy on ImageNet, to complement the Top-1 accuracy in Table 2 .

In each setting, the total training time of our K-matrix approach is within 20% of the total training time of vanilla ShuffleNet.

In Figure 4 , we plot the loss and accuracy on the training set and validation set when we train 1.0 ShuffleNet g8, with either a fixed permutation (Shuffle) or a K-matrix for channel shuffling.

Even though each K-matrix is a product of multiple (sparse) matrices, K-matrices take about the same number of training steps to converge as the baseline model.

One reason is that K-matrices can be easily initialized or constrained to be orthogonal (Section 2.4), thus avoiding vanishing or exploding gradients.

The permuted CIFAR-10 dataset is constructed by applying a fixed permutation to every input.

We choose to use the 2-D bit-reversal permutation, 7 i.e., the bit reversal permutation on 32 elements is applied to the rows and to the columns.

This permutation was chosen because it is locality-destroying: if two indices i, j are close, they must differ in a lower-order bit, so that the bit-reversed indices i , j are far.

This makes it a particularly difficult test case for architectures that rely on spatial locality such as "vanilla" CNNs.

We describe the model architectures used in Section 3.1 (those reported in Table 3 ).

The model represents a fixed permutation P , parametrized as a K-matrix, to learn to recover the true permutation, followed by a standard ResNet18 architecture (He et al., 2016) .

Because of the simple decomposable nature of the butterfly factors (Section 2.1), our parameterization is easily extensible with additional techniques:

(i) We constrain each butterfly factor matrix in the K-matrix to be doubly-stochastic.

For example, each 2 × 2 block in the butterfly factor matrix of block size 2 has the form a 1 − a 1 − a a , where a ∈ [0, 1].

We treat this block as a distribution over permutations, generating the identity 1 0 0 1 with probability a and the swap 0 1 1 0 with probability 1−a.

Butterfly factor matrices with larger block sizes are constrained to be doubly-stochastic in a similar manner.

In this way, a permutation is sampled for each butterfly factor matrix, and these permutations are composed to get the final permutation that is applied to the image. (ii) For each minibatch, the examples P x by applying permutation samples on the (permuted) inputs are fed into an additional unsupervised reconstruction loss 0≤i,j<n

measuring total variation smoothness of the de-noised inputs.

Such loss functions are often used in image denoising (Rudin et al., 1992) .

A final regularization loss was placed on the entropy of P , which was annealed over time to encourage P to converge toward a sharper doubly-stochastic matrix (in other words, a permutation).

The model is trained with just the reconstruction loss to convergence before a standard ResNet is trained on top.

These techniques are applicable to the K-matrix as well as specialized methods for representing permutations such as Gumbel-Sinkhorn (Mena et al., 2018) and are important for recovering the true permutation.

However, they are not applicable to a general linear layer, which shows the flexibility of K-matrices for representing generic structure despite not being a specially tailored method for this task.

We also remark that other classes of structured linear maps such as low-rank, circulant, and so on, are even less suited to this task, as they are incapable of representing all permutations.

1.

Fully connected (FC): This is a 3-layer MLP, with hidden size 1024 and ReLU nonlinearity in-between the fully connected layers.

We use a gated recurrent unit (GRU) model (Cho et al., 2014) , with hidden size 1024.

Many RNN architectures have been proposed to capture long-range dependency on permuted image dataset such as Permuted MNIST (Arjovsky et al., 2016) .

Standard gated architectures such as LSTM and GRU have shown competitive performance on the Permuted MNIST dataset, and we choose GRU as a baseline since it has been reported to slightly outperform LSTM (Bai et al., 2018) .

We use the standard ResNet18 architecture, adapted to smaller image size of the CIFAR-10 dataset (changing stride from 2 to 1 of the first convolutional layer, and removing max-pooling layer that follows).

4. Dense + CNN: We add an additional linear layer (i.e. a dense matrix) of size 1024 × 1024 before the ResNet18 architecture.

This dense layer can in theory represent a permutation, but cannot benefit from the additional techniques described above.

We use the standard ResNet18 architecture applied to the unpermuted CIFAR-10 dataset.

All models are trained for 200 total epochs, with the Adam optimizer.

We use the standard learning rate schedule and weight decay from Mostafa & Wang (2019b) .

We use Hyperband (Li et al., 2017) to tune other hyperparameters such as the initial learning rate and annealing temperature.

The architecture of each layer of the decoder is:

For every layer of the decoder, we replace all four dense weight matrices in the four Linear layers with four K-matrices from the B class (i.e. butterfly matrices).

The models are trained from scratch using the training script from the Fairseq repository, with the same hyperparameters (optimizer, learning rate, number of updates) used in the DynamicConv paper (Wu et al., 2019) .

We note that the DynamicConv model with K-matrices in the decoder trains slightly faster than the default DynamicConv model (both models are trained for 50,000 updates, which requires approximately 7% less time for the K-matrix model than for the default model).

To evaluate inference speed, we run the decoding script on the IWSLT-14 De-En test set in singlethreaded mode on a server Intel Xeon CPU E5-2690 v4 at 2.60GHz, and measure wall-clock time.

The test set contains 6750 sentences, with 149241 tokens.

Following Wu et al. (2019), we set the batch size to 1 and beam size to 1.

We additionally compare the speed-quality tradeoff of K-matrices with other classes of structured matrices, when used to replace the fully-connected layers of DynamicConv's decoder.

We consider the following classes of structured matrices, in addition to K-matrices: low-rank, circulant, Toeplitzlike (Sindhwani et al., 2015) , ACDC (Moczulski et al., 2016 ), Fastfood (Le et al., 2013 , and sparse.

For classes with variable number of parameters (e.g. low-rank, sparse), we set the number of parameters to match that of K-matrices.

For sparse matrices, besides the result for an ensemble of 10 models (the default setting in the Fairseq repository), we also report the result for a single model, as that could have faster inference time (ensembling/averaging sparse matrices produces less a less sparse matrix).

In Figure 5 , we plot the tradeoff between translation quality (measured by BLEU score) and inference speed (sentences per second).

Most classes of structured matrices produce similar translation quality (between 34.1 and 34.4 BLEU score).

K-matrices have the second fastest inference time, only 7% slower than low-rank matrices.

We note that low-rank matrices benefit from very well-tuned BLAS routines (matrix-matrix multiplication).

Even though our implementation of K-matrix multiplication is not yet highly optimized, it is already quite close to the speed of low-rank matrices.

Each K-matrix (for fixed width and expansion), has an O(n log n) matrix-vector multiplication algorithm by sequentially multiply the input vector with each of the sparse factor.

Our implementation of this simple algorithm is surprisingly competitive with optimized subroutines both on GPU (e.g. for training) and on CPU (e.g. for inference).

In Figure 6 , we compare the speed of multiplying by a K-matrix in class B (i.e. a butterfly matrix) against specialized implementation of the FFT.

We normalize the speed by the speed of dense matrix-matrix multiply (on GPU) or dense matrix-vector multiply (on CPU).

On GPU, with input sizes n = 1024 and batch size 256, the training time (forward and backward) of K-matrices matrix is 23% faster than dense matrix multiply (GEMM from cuBLAS).

For inference on CPU, the kaleidoscope fast multiplication can be one or two orders of magnitude faster than GEMV.

Over a range of matrix sizes, our implementation is within a factor of 4x of specialized implementation of the FFT, a highly optimized kernel.

Our implementation is also memory efficient.

In the forward pass through the O(log n) sparse factors, we do not store the intermediate results, but recompute them during the backward pass.

Therefore the activation memory required is O(bn) for input batch size b.

We directly validate Theorem 1 on well-known types of structured matrices used in machine learning.

Given a structured matrix M, we attempt to represent M as closely as possible using K-matrices as well as the standard classes of structured matrices: sparsity and low-rank.

In Table 7 , we quantify the expressivity of each of the three methods, as measured by their ability to approximate a range of different structures.

Results for "global minimum" of kaleidoscope matrices are obtained from the theoretical expressiveness results in Section I and Section J. Low-rank and sparse approximation have closed form solutions: truncating the SVD and keeping the largest-magnitude entries, respectively.

We also report the results using SGD for kaleidoscope matrices to validate that good approximation with K-matrices can be obtained even from standard first-order optimization algorithms.

Even with imperfect optimization, kaleidoscope matrices can still capture out-of-class target matrices better than low-rank and sparse matrices.

Table 7 : Expressiveness of different classes of structured matrices: Frobenius error of representing common structured matrices (columns) of dimension 256 using three structured representations of matrices with adjustable numbers of parameters.

(Left group: Target matrices in the same class as the methods.

Middle group: Target matrices with fixed number of parameters.

Right: Random matrix to show typical scale of error.)

Each method is allotted the same number of parameters, equal to a log n factor more than that of the target matrix.

Low-rank and sparse matrices are unable to capture any structure outside their own class, while the minima for kaleidoscope matrices found via optimization better capture the actual structure for out-of-class targets better than the baselines.

The target matrices are kaleidoscope, low-rank, sparse, convolution (i.e. circulant matrices), Fastfood (Le et al., 2013) , and entrywise random iid Gaussian matrix (to show the typical magnitude of the error).

All target matrices M were randomly initialized such that

To find a kaleidoscope approximation with SGD, we Hyperband to tune its learning rate (from 0.001 to 0.5).

E PROPERTIES OF THE BB * HIERARCHY Here, we justify why the definitions in Section 2.2 give rise to a hierarchy.

We first make some basic observations about the parameterization.

Observation E.1.

An n × n matrix M ∈ BB * has 4n log n parameters.

Proof.

M can be expressed as a product of 2 log n butterfly factor matrices of size n × n. Each of these factor matrices has 2 parameters per row, for a total of 2n parameters each.

Hence, the total number of parameters is 4n log n.

Observation E.2.

Let M be an n × n matrix in (BB * ) w e .

Then, given an arbitrary vector v of length n, we can compute Mv with O(wne log(ne)) field operations.

Proof.

Since M ∈ (BB * ) w e , we can decompose it as SE 1 E 2 . . .

E w S T , where S is as given in Definition 2.4, and each E i is an en × en matrix in BB * .

Therefore, to compute Mv, we can use associativity of matrix multiplication to multiply the vector by one of these matrices at a time.

Since all of these factors are sparse, we use the naïve sparse matrix-vector multiplication algorithm (begin with a 0-vector and perform the corresponding multiplication and addition for each nonzero matrix entry).

S (and thus S T ) have n NNZ.

Therefore, matrix-vector multiplication by S or S T requires O(n) operations, which is dominated by the butterfly matrix-vector multiplication.

Each E i can be further decomposed into 2 log(ne) matrices with at most 2ne non-zero entries each (by Observation E.1).

Therefore, matrix vector multiplication by each E i requires O(ne log(ne)).

Since there are w such E i , we require a total of O(wne log(ne)) operations.

Now, we are ready to show that our definition of classes (BB * ) w e forms a natural hierarchy.

First, we must argue that all matrices are contained within the hierarchy.

Lemma E.3.

Let M be an arbitrary n × n matrix.

Then M ∈ (BB * ) (2n−2) .

Proof.

Corollary E.3 in Appendix K shows that any n × n matrix can be written in the form

, where M i , M i are orthogonal butterfly matrices and M is a diagonal matrix.

We can combine D with M n to form another (possibly not orthogonal) butterfly matrix.

This yields a decomposition of M as products of (possibly not orthogonal) butterfly matrices and their (conjugate) transposes, completing the proof.

Next, we argue that, up to a certain point, this hierarchy is strict.

Lemma E.4.

For every fixed c ≥ 1, there is an n × n matrix M n (with n sufficiently large) such that

Proof.

Given c, fix n to be a power of 2 such that c < n 4 log 2 n .

For sake of contradiction, assume that every n × n matrix in (BB * ) c+1 is also in (BB * ) c .

Let A be an arbitrary n × n matrix.

From Lemma E.3, A ∈ (BB * ) (2n−2) .

From our assumption, we can replace the first c + 1 BB * factors of A with c (potentially different) BB * factors and still recover A. We can repeat this process until we are left with c BB * factors, implying that A ∈ (BB * ) c .

From Observation E.1, we require 4cn log n < n 2 (by our choice of n) parameters to completely describe A. This is a contradiction since A is an arbitrary n × n matrix, and therefore has n 2 arbitrary parameters.

Hence, there must be some n × n matrix in (BB * ) c+1 that is not in (BB * ) c .

In this appendix, we prove our main theoretical result, namely, our ability to capture general transformations, expressed as low-depth linear arithmetic circuits, in the BB * hierarchy.

This result is recorded in Theorem 1.

Theorem 1.

Let M be an n×n matrix such that matrix-vector multiplication of M times an arbitrary vector v can be represented as a be a linear arithmetic circuit C comprised of s gates (including inputs) and having depth d. Then, M ∈ (BB * )

To prove Theorem 1, we make use of the following two theorems.

Theorem 2.

Let P be an n × n permutation matrix (with n a power of 2).

Then P ∈ BB * .

Theorem 3.

Let S be an n × n matrix of s NNZ.

Then S ∈ (BB * )

Theorem 2 is proven in Appendix G, and Theorem 3 is proven in Appendix I.

Proof of Theorem 1.

We will represent C as a product of d matrices, each of size s × s , where s is the smallest power of 2 that is greater than or equal to s.

To introduce some notation, define w 1 , . . .

w d such that w k represents the number of gates in the k'th layer of C (note that s = n + d k=1 w k ).

Also, define z 1 , . . .

z d such that z 1 = n and z k = w k−1 + z k−1 (z k is the number of gates that have already been used by the time we get to layer k).

Let g i denote the i'th gate (and its output) of C (0 ≤ i < s), defined such that:

where i 1 , i 2 are indices of gates in earlier layers.

For the k'th layer of C, we define the s × s matrix M k such that it performs the computations of the gates in that layer.

Define the i'th row of M k to be:

We'd like to argue that v d contains the outputs of all gates in C (i.e, the n values that make up Mv).

To do this we argue, by induction on k, that v k is the vector whose first z k+1 entries are g 0 , g 1 , . . . , g (z k −1) , and whose remaining entries are 0.

The base case, k = 0 is trivial.

Assuming this holds for the case k − 1, and consider multiplying v k−1 by M k .

The first z k rows of M k duplicate the first z k entries of v k−1 The next w k rows perform the computation of gates g z k , . . .

, g (z k+1 −1) .

Finally, the remaining rows pad the output vector with zeros.

Therefore, v k is exactly as desired.

The final matrix product will contain all n elements of the output.

By left multiplying by some permutation matrix P, we can reorder this vector such that the first n entries are exactly Mv.

Hence, we are left to argue the position of PM d . . .

M 2 M 1 within the BB * hierarchy.

Each M k is a matrix with total 2w k + z k < 2s NNZ.

From Theorem 3, we can, therefore, represent M k as a product of O(1) matrices (of size 2s ) in BB * .

From Theorem 2, P ∈ BB * .

Note that s ≤ s < 2s, so s = Θ(s).

* factors, and requires an expansion from size n to size 2s , or an expansion factor of O(

, as desired.

Remark F.1.

By applying Observation E.2, we see that Theorem 1 gives an O(sd log s) matrix vector multiplication algorithm for M.

In this appendix, we prove Theorem 2.

To do this, we decompose permutation matrix P into P = LR, with L ∈ B and R ∈ B * .

Throughout the proof, we make use of the following definition.

Definition G.1.

Let L be an n × n permutation matrix (n a power of 2).

We say that L meets the 2 j balance condition if L can be divided into chunks of 2 j (with each chunk having all columns i such that i 2 j has the same value) such that for every 0 ≤ m < 2 j , each chunk has exactly one L[:, k] = e π k with π k ≡ m ( mod 2 j ).

We say that L is modular-balanced if it meets the 2 j balance condition for each 2 ≤ 2 j ≤ n.

First step of decomposition of modular-balanced matrix L. Here, the red entries must be permuted into the main diagonal blocks.

Proof.

We proceed by induction on n. The base case n = 2 is trivial.

As our inductive hypothesis, we assume that all modular-balanced matrices of size n 2 × n 2 are butterfly matrices of size n 2 .

From Definition 2.3, it is sufficient to show that L can be decomposed as:

where B n is a butterfly factor of size n and each L j is an n 2 × n 2 modular-balanced matrix.

Define L 1 and L 2 such that:

Note that since L is a permutation matrix (and thus has exactly one non-zero entry per column), at most one term of each of these sums can be non-zero.

For sake of contradiction, assume L 1 is not modular-balanced.

Then, for some 2 j ≤ n 2 , there are two columns c 1 , c 2 such that c1 2 j = c2 2 j and such that indices of the non-zero entries of L 1 in columns c 1 and c 2 are the same modulo 2 j .

However, from the definition of L 1 , this implies that the indices of the non-zero entries of L in columns c 1 and c 2 are also the same modulo 2 j , contradicting L being modular-balanced.

Hence, L 1 is modular-balanced.

An analogous argument (that instead considers columns c 1 + n 2 , c 2 + n 2 of L) shows that L 2 is also modular-balanced.

To complete the proof, we must argue that B n is a butterfly factor of size n. Since each L i is modular-balanced, it is a permutation matrix.

Therefore, L has exactly 1 non-zero entry in each of the first n 2 rows and columns from L 1 and exactly 1 non-zero entry in each of the second n 2 rows and columns from L 2 .

Hence, L is a permutation matrix.

Since both L and L are permutation matrices, B = L (L ) −1 must also be a permutation matrix.

Therefore, we can view B as performing a permutation of the rows of L to get L.

Consider the i'th row of L , with 0 ≤ i < In both cases, the non-zero entries of B fall into the correct diagonal bands (the main diagonal, and the bands n 2 away).

Hence, B is a butterfly factor of size n. Now, we consider the process of transforming P into a modular-balanced matrix.

We make use of the following lemma.

If M met the k 2 balance condition, then each node would additionally have in-degree exactly 1 and out-degree exactly 1.

By reversing edges of G such that each (undirected) cycle becomes a directed cycle, we can achieve this.

However, reversing edges corresponds to swapping columns of M that are k 2 apart.

Let B k be the permutation matrix that performs all such swaps.

B k has non-zero entries only along the main diagonal and the diagonal bands k 2 away, and thus is a butterfly factor of size k.

We are ready to present the decomposition of P. Lemma G.3.

Let P be an n × n permutation matrix.

Then we can decompose P into P = LR, where L is modular-balanced and R ∈ B * .

Proof.

We repeatedly apply Lemma G.2.

First, we conclude that there is a butterfly factor B n such that PB n = P , where P meets the n 2 balance condition.

Now, we consider the first and last n 2 columns of P independently.

We can again apply Lemma G.2 (twice) to conclude that there are butterfly factors B n 2 1 , B n 2 2 such that

where P meets the n 2 and n 4 balance conditions.

We continue this process until we obtain a matrix that meets all of the balance conditions.

Our final equation is of the form:

where B is a butterfly matrix and L is a modular-balanced matrix.

Let R = B −1 = B * (since B is a permutation matrix, and thus is orthogonal) and hence R ∈ B * .

Then P = LR, as desired.

Theorem 2 follows immediately from Lemmas G.3 and G.1.

Here, we present some basic facts of the BB * hierarchy that will be useful for later constructions.

For simplicity, we assume (WLOG via 0-padding) that all matrices are square matrices with size that is a power of 2.

Lemma H.1.

If M ∈ B (or M ∈ B * ), then DM, MD ∈ B (B * resp.) for any diagonal matrix D.

Proof.

Left multiplication by a diagonal matrix scales the rows of M by the corresponding diagonal entries.

The same can be achieved by scaling all entries the leftmost butterfly factor matrix.

Similarly, right multiplication by a diagonal matrix scales the columns of M, which can be achieved by scaling all entries in the columns of the rightmost butterfly factor matrix.

w2 by Lemma H.1.

Hence, AB ∈ (BB * ) w1+w2 e by Definition 2.4.

where P is a permutation that that moves the first k rows of each E Ai (in order) into the top mk rows.

From Theorem 2, P ∈ BB * , (and so is P T , also a permutation).

Within the RHS block matrix, the decompositions of each E Ai can be done in parallel, requiring total width w. Hence,

w+2 e , as desired.

Remark H.4.

If e = 1 in Lemma H.3, then P is unnecessary.

Hence,

Proof.

For each 1 ≤ i ≤ m, let E Ai ∈ F ek×ek be defined such that A i = SE Ai S T (with S as in Definition 2.4).

Note that E Ai ∈ (BB * ) w .

Consider matrices of the form:

Here, L and R compute the sum of the 2ek × 2ek matrices on the diagonal of SP 1 , where P 1 is a permutation swapping E Ai to the 4 th ek-block column.

Note that S is the diagonalization of four matrices in (BB * ) w , so S ∈ (BB * ) w by Remark H.4.

In addition, since each block in S is a butterfly matrix of size ek, S only uses butterfly factors up to size ek, so the outer factor matrices of sizes 4ek and 2ek in S are unused.

Also note that L and R are butterfly factor matrices of size 4ek (or B (4ek) 4ek ), and P 1 is a butterfly factor matrix of size 2ek (or B (4ek) 2ek ).

This allows us to fold the surrounding matrices L,

Through repeated application (m times) of the identity

From Lemma H.2, M ∈ (BB * ) mw .

Finally, note that

where P 2 is a permutation that moves the first k columns of the second block-column of M to the left.

P 2 can be folded into the final summation factor M m as follows:

Lemma H.6.

Let M be an invertible n × n matrix such that M ∈ B. Then M −1 ∈ B * .

Proof.

We prove this in a series of steps. .

By the form of B, non-zero entries within a row or column are always exactly k 2 positions apart.

Therefore, the only row operations needed for this Gaussian elimination are:

• Scaling a row by a constant factor c = 0

• Addition of a row to another row exactly k 2 rows apart Performing these operations on I k will only allow non-zeros on the main diagonal and k 2 diagonals away from the main diagonal.

Hence, B −1 k is also a butterfly factor of size k.

k be an invertible butterfly factor matrix of size n and block size k. Its inverse is the block diagonal matrix formed by the inverses of each of its constituent butterfly factors.

From above,

is also a butterfly factor matrix of size n and block size k.

Finally, consider M ∈ B.

Finally, we include a closure result for the Kronecker product, another common matrix composition operation.

Although Lemma H.7 is not directly used in the subsequent proofs, it allows for examples the results for the DFT to be lifted to higher-dimensional Fourier transforms.

We also note that the closure bound in Lemma H.7 can be tightened in such cases (cfṘemark H.4).

Proof.

Note that

for some permutation P. In this appendix, we prove Theorem 3.

First, we consider matrices with at most n NNZ.

Lemma I.1. let S be an n × n matrix with at most n NNZ.

Then, S ∈ (BB * ) 5 .

We use this lemma and the addition closure lemma to prove Theorem 3.

Proof of Theorem 3.

We note that any s sparse matrix is the sum of s n matrices of at most n NNZ, and we appeal to Lemma H.5.

In the rest of the section we will prove Lemma I.1.

We begin by defining two classes of matrices that will be used in our decomposition.

Definition I.1.

An n × n matrix H is a horizontal step matrix if for every 0 ≤ i, i < n and

An n × n matrix V is a vertical step matrix if V * is a horizontal step matrix.

With this definition, the horizontal step matrix obeys a "Lipschitz-like" condition.

Each column of a horizontal step matrix can have at most one non-zero entry, and given two non-zero columns k apart, the non-zero entry in the right column must be between 0 and k rows below the non-zero entry in the left column.

Note that to show that a matrix is a horizontal step matrix, it is sufficient to argue that this condition holds for each pair of neighboring non-zero columns.

Similarly, each row of a vertical step matrix can have at most one non-zero entry, and given two non-zero rows k apart, the non-zero entry in the lower row must be between 0 and k columns to the right of the non-zero entry in the upper row.

Lemma I.2.

Let H be an n × n horizontal step matrix.

Then H ∈ B.

Proof.

We proceed by induction on n. The base case n = 2 is trivial.

As our inductive hypothesis, we assume that all horizontal step matrices of size n 2 × n 2 are butterfly matrices of size n 2 .

From Definition 2.3, it is sufficient to show that H can be decomposed as:

where H 1 , H 2 are n 2 × n 2 horizontal step matrices and each D k is a n 2 × n 2 diagonal matrix.

Denote the four, n 2 × n 2 corner submatrices of H by:

Then, define H 1 and H 2 by:

For sake of contradiction, assume that H 1 is not a horizontal step matrix.

Then, there are 0

From our definition of H 1 , the non-zero entries in columns j and j of H are either (i − i) mod n 2 or n 2 + (i − i) mod n 2 , both of which are greater than j − j, rows apart.

This contradicts H being a horizontal step matrix.

Hence, H 1 must be a horizontal step matrix, as must H 2 from an analogous argument.

To finish the proof, we argue the correctness of the decomposition by equating arbitrary entries of each of the 4 corner submatrices.

We begin with the upper left submatrix.

Here, we consider two cases:

Since H is a horizontal step matrix (and hence may have at most one non-zero entry per column), it follows that H 11 [i, j] = 0.

In this case, the indicator function evaluates to 0, so

Otherwise, for sake of contradiction, suppose that H 21 [i, :] = 0.

Then, two of the first n 2 columns of H would have non-zero entries n 2 rows apart, contradicting H being a horizontal step matrix.

Hence,

In all cases,

, so our decomposition correctly recovers the upper left corner of H. Analogous arguments show that the other three corners are also correctly recovered.

Hence, our decomposition is correct, and by induction, H ∈ B.

Corollary I.1.

Let V be a vertical step matrix.

Then V ∈ B * .

Now, we use step matrices to prove Lemma I.1.

Proof of Lemma I.1.

Given S, we decompose it as S = P 1 HP 2 VP 3 , where each P is a permutation matrix, H is a horizontal step matrix, and V is a vertical step matrix.

For an example of this, see Figure 9 .

We first decompose S as S = P 1 S P 3 , where P 1 is the permutation that moves all 0 rows of S to the bottom and P 3 is the permutation that moves all 0 columns of S to the right.

Next, we further decompose S into S = HV as follows.

Since S has s ≤ n NNZ, we can parameterize

with the non-zero entries indexed in row-major order.

Define matrix H by:

Define matrix V by:

To show that S = HV , we consider an arbitrary entry:

by definition of matrix multiplication

by definition of H and V Here, we note that (i, j) can equal (i k , j k ) for at most one value of k since the locations in θ are unique.

Hence, HV [i, j] = c k only if (i, j) = (i k , j k ) for some k, which is exactly the definition of S .

Hence, S = HV .

We argue that H is a horizontal step matrix through a series of assertions.

First, note that H has exactly one non-zero entry in each of its first s columns.

Also, note that since θ is in row-major order, these non-zero entries are sorted (any column to the right cannot have a non-zero entry in a higher row).

Hence, to show that H is a horizontal step matrix, it is sufficient to argue that adjacent columns of H have non-zero entries at most one row apart.

This is equivalent to S having no zero rows between two non-zero rows, which is guaranteed by P 1 .

Hence, H is a horizontal step matrix.

Since V has at most one non-zero entry per row, we may permute the rows of V to obtain a matrix V, where the non-zero entries of V are sorted (any lower row below cannot have a non-zero entry in an earlier column).

Hence, for some permutation matrix (P 2 ) −1 , V = (P 2 ) −1 V , which implies that V = P 2 V. It has exactly one non-zero entry in each of its first s columns.

From the action of P 2 , these non-zero entries are sorted.

Therefore, by the same argument as for H above, V T is a horizontal step matrix.

Hence, V is a vertical step matrix.

In all, we have found a decomposition S = P 1 HP 2 VP 3 , where each P is a permutation matrix (∈ BB * by Theorem 2), H is a horizontal step matrix (∈ BB * by Lemma I.2), and V is a vertical step matrix (∈ BB * by Corollary I.1).

By Lemma H.2, S ∈ (BB * ) 5 .

Corollary I.2.

Let R be an n × n matrix of rank r. Then R ∈ (BB * ) 10r 4 .

Proof.

We can decompose R as R = GH * where G, H are n × r matrices.

With appropriate zero-padding, both of these can be made into n × n matrices with at most rn NNZ.

The proof follows immediately from Theorem 3 and Lemma H.2.

In this appendix, we draw comparisons between the BB * hierarchy and the BP hierarchy introduced by Dao et al. (2019) .

Lemma J.1.

Let F n be the Discrete Fourier Transform of size n.

Then F n ∈ (BB * ) 2 .

Proof.

From Parker (1995), we can express F n as F n = B P, where B ∈ B and P is a permutation (the bit reversal permutation).

From Theorem 2, P ∈ BB * .

Hence, by Lemma H.2, F n ∈ (BB * ) 2 .

Lemma J.2.

Let H n be the Hadamard Transform of size n.

Then H n ∈ BB * .

Proof.

H n ∈ B, so trivially H n ∈ BB * .

Lemma J.3.

Let S n be the Discrete Sine Transform of size n.

Then S n ∈ (BB * ) 2 .

Proof.

As described in Makhoul (1980), S n can be performed as a scaled permutation (separating the even and odd indices of the input, and reversing and negating the odd indices) composed with F n .

Therefore, we may decompose S n as S n = B P 2 D P 1 , where P 1 , P 2 are permutations, B ∈ B, and D is a diagonal matrix.

P 2 D P 1 is simply a permutation matrix with scaled entries, which can be equivalently expressed as D P for some diagonal matrix D and permutation P .

By Lemma H.1, B D ∈ BB * .

By Theorem 2, P ∈ BB * .

Hence, by Lemma H.2, S n ∈ (BB * ) 2 .

Remark J.

4.

An analogous argument shows that the Discrete Cosine Transform is also in (BB * ) 2 .

Lemma J.5.

Let C n be an n × n circulant (convolution) matrix.

Then C n ∈ BB * .

Proof.

Using Theorem 2.6.4 of Pan (2001), we can express C n as C n = (F n ) −1 DF n where F n is the Discrete Fourier Transform and D is a diagonal matrix. (F n ) −1 = B P (with B ∈ B, P a permutation), which implies that F n = (P) −1 (B) −1 .

Therefore

The middle three factors have the effect of performing a permutation, scaling each element, and undoing the permutation, which is equivalent to simply scaling by some diagonal matrix D .

Hence, we are left with

By Lemma H.1, B D ∈ B. By Lemma H.6, (B) −1 ∈ B * .

Hence, C n ∈ BB * .

Remark J.6.

We can expand any n × n Toeplitz matrix T n into a 2n × 2n circulant matrix (with upper left n × n submatrix equal to T n ).

Hence, T n ∈ (BB * ) 1 2 by Lemma J.5.

The Fastfood matrix class (Le et al., 2013) can be tightly captured in the BB * hierarchy:

Lemma J.7.

The product SHDPHB where S, D, B are diagonal matrices, H is the Hadamard transform, and P is a permutation matrix, is in (BB * ) 3 .

Proof.

We have shown in Lemma J.2 that H ∈ BB * , and in Theorem 2 that P ∈ BB * .

Since BB * is closed under diagonal multiplication (Lemma H.1), we conclude that SHDPHB ∈ (BB * ) 3 .

The two classes of matrices introduced in Moczulski et al. (2016) , called AFDF and ACDC, are also tightly captured in the BB * hierarchy:

Lemma J.8.

Let AF −1 DF be a product of a diagonal matrix A, the inverse Fourier transform F −1 , another diagonal matrix D, and the Fourier transform F. Then AF −1 DF ∈ BB * .

Let AC −1 DC be a product of a diagonal matrix A, the inverse cosine transform C −1 , another diagonal matrix D, and the cosine transform C. Then AC −1 DC ∈ (BB * ) 4 .

Proof.

We have argued in Lemma J.5 that F −1 DF ∈ BB * .

Since BB * is closed under diagonal multiplication (Lemma H.1), we conclude that AF −1 DF ∈ BB * .

We have shown that C ∈ (BB * ) 2 , so C −1 ∈ (BBS) 2 as well.

Since BB * is closed under diagonal multiplication (Lemma H.1), we conclude that AC −1 DC ∈ (BB * ) 4 .

Remark J.9.

Within each butterfly factor matrix of the DFT (excluding the bit reversal permutation) and the Hadamard transform, the columns are pairwise orthogonal and have norm 2.

Hence, we can divide all factors by √ 2 to make orthogonal factor matrices.

To counteract this scaling, we can add a diagonal matrix with √ 2 log 2 (n) = √ n in all entries to the factorization.

By doing this we can place all of the above transforms in the OBB hierarchy (defined in Appendix K) with the same width and expansion factor.

Here, we show that, using larger matrices, we are able to similarly capture multi-dimensional versions of the above transforms.

Lemma J.10.

Let F 2 n be the 2-dimensional Discrete Fourier Transform (represented as an

Proof.

The separation property of the 2-D DFT allows us to express its action on an n × n matrix as the composition of a 1-D DFT on each of its rows and a 1-D DFT on each of its columns.

If we view the 2-D DFT as an n 2 × n 2 matrix, its input and outputs will both be column vectors of size n 2 .

As our convention, we list the entries of the input vector in the row-major order corresponding to the n × n input matrix.

Then, we consider the 2-D DFT in four steps, where the first two steps perform the 1-D DFT row-wise, and the second two steps perform the 1-D DFT column-wise:

Step 1: Permute the columns:

We permute the columns (with a bit reversal permutation), which performs a bit reversal permutation on each row.

Viewing the input as a vector, this step corresponds to left multiplication by a permutation matrix P c that permutes the entries of each chunk of size n of the input vector.

Step 2:

Multiply each row by a butterfly matrix Since the entries of the input were listed in row major order, this step is achieved through multiplication by a block diagonal matrix of n butterfly matrices of size n, which can be viewed as a product of butterfly factor matrices B

Step 3: Permute the rows:

We permute the rows (with a bit reversal permutation), which performs a bit reversal permutation on each column.

This corresponds to left multiplication by a permutation matrix P r .

Since we are permuting the rows, P r permutes the entries at the granularity of each n-chunk.

Since Steps 1 and 2 each performed an identical computation to each n-chunk we can move this row permutation before

Step 2, combining P c and P r into a single permutation P.

Step 4:

Multiply each column by a butterfly matrix Consider multiplication by the first factor matrix.

In each row, this matrix is taking linear combinations of adjacent column entries.

In our length-n 2 vector, these entries will be exactly n indices apart.

Therefore this multiplication can be handled by a butterfly factor matrix B (n 2 )

2n .

Similarly, we find that this butterfly multiplication can be expressed as multiplication by a product of butterfly factor matrices B (n 2 )

2n .

Combined with the factor matrices from Step 2, these form a butterfly matrix B of size n 2 .

In all, we see that the 2-D DFT may be realized as multiplication by a permutation matrix P followed by multiplication by a butterfly matrix B. The same argument as Lemma J.1 shows that F 2 n ∈ (BB * ) 2 .

Remark J.11.

An analogous argument (using the separation property of the respective transforms) can be used to argue that 2-D Discrete Sine and Discrete Cosine transforms are in (BB * ) 2 , and that 2-D Hadamard Transforms are in BB * .

Lemma J.12.

Let C 2 n be a 2-dimensional convolution matrix.

Then C 2 n ∈ BB * .

Proof.

We can express a 2-D convolution matrix as C −1 ) as the product of a butterfly matrix and a permutation matrix.

The rest of the argument is analogous to the proof of Lemma J.5.

Remark J.13.

Using an inductive argument, we can show that all k-dimensional (k ∈ Z) variants of the above transforms, expressed as n k × n k matrices are contained in BB * or (BB * ) 2 .

To do this, we use the separation property of the transforms to break them into a k − 1-dimensional transform (the inductive hypothesis) followed by a 1-dimensional transform.

Through practical application of the butterfly matrices, it has been found useful to constrain them in orthogonality.

In Section K.1 we will modify the existing kaleidoscope hierarchy to create the orthogonal kaleidoscope hierarchy OBB.

Then, in Section K.2, we will argue that all orthogonal matrices, and as a result all matrices, can also be expressed in this hierarchy in O(n) width.

Lastly, in Section K.3, we will argue that permutation matrices and sparse matrices also exist in this hierarchy in O(1) width, which in turn implies a corresponding result for matrices with low-depth arithmetic circuits.

The definition of the orthogonal butterfly is identical to the original butterfly, with the constraint that all butterfly factors are orthogonal.

We specify this definition below: Definition K.1 (Analog of Definition 2.1).

An orthogonal butterfly factor of size k ≥ 2 (denoted as B k ) is a butterfly factor that is also orthogonal.

Definition K.2 (Analog of Definition 2.3).

An orthogonal butterfly matrix of size n (denoted as B (n) ) is a butterfly matrix with all butterfly factor matrices being orthogonal.

It is easily checked that

We choose

, which are orthogonal by construction (via (5)).

Hence, L ∈ B (n)

where • denotes the Hadamard product.

From (6)

Denoting the first half of this vector by w 0 ∈ C n/2 , we have

where w 0 2 = u 2 = 1.

The result follows inductively.

As an immediate corollary, we can use Singular Value Decomposition to obtain a factorization for an arbitrary n × n matrix.

Corollary K.1.

Let M be an arbitrary n × n matrix.

Then, M ∈ (OBB) 2n−1 , where all but one matrix in the decomposition is orthogonal (unitary).

Proof.

By employing Singular Value Decomposition, we can decompose M as M = UΣV * , where U, V * are orthogonal and Σ is diagonal.

By Lemma K.2, U, V * ∈ (OBB) n−1 , and trivially Σ ∈ OBB.

Hence, M ∈ (OBB) 2n−1 .

Note that Σ is the only matrix in the decomposition that is not orthogonal (unitary).

We show that we can construct s-sparse matrices in the OBB hierarchy with the same width as the BB * hierarchy.

The proof follows a structure to that of Theorem 3.

We begin by arguing about permutation and step matrices, then using the same factorization to argue that matrices with at most n NNZ are contained in (BB * ) 5 .

Then, we will appeal to a modified sum closure lemma to extend the argument to matrices of general s NNZ.

Similar to Appendix F, we can use these results to place all matrices with low-depth circuits for matrix vector multiplication in the OBB hierarchy.

We begin by presenting the argument that permutations are included in OBB as a corollary to Theorem 2.

Corollary K.2.

Let P be a permutation matrix.

Then P ∈ B B * .

Proof.

We appeal to the decomposition from Theorem 2, noting that all butterfly factor matrices constructed in the proofs of Lemmas G.3 and G.1 are permutation matrices, and thus are orthogonal.

Hence, P ∈ OBB where the inner diagonal matrix is I.

To prove the containment of sparse matrices within the OBB hierarchy, we make use of the following lemma.

Lemma K.3.

Let P be a permutation matrix and D a diagonal matrix.

Then there exist diagonal matrices D and D such that:

Proof.

Let σ be the permutation such that

An analogous argument to above shows that DP = PD .

In the BB * hierarchy (Lemma I.2), we were able to show that horizontal step matrices are butterfly matrices.

Here, we present a similar result for the OBB hierarchy.

Lemma K.4.

Let H be an n × n horizontal step matrix.

Then we can decompose H = DO, where D is a diagonal matrix and O ∈ B.

Proof.

Throughout the proof, we make reference to the original horizontal step matrix construction given in Lemma I.2 and its proof.

To begin, we show that an arbitrary 2 k × 2 k butterfly factor H 2 k in the decomposition of H can be expressed as the product of a diagonal matrix and an orthogonal butterfly factor.

Since a butterfly factor is direct sum of 2 × 2 matrices, there is a permutation matrix P 2 k such that conjugation of H 2 k by P 2 k gives a block diagonal matrix H 2 k of n 2 2 × 2 matrices, i.e. Figure 10 for an illustration.)

Specifically, P 2 k is the permutation where: We argue that each of these 2×2 blocks can be decomposed into a diagonal matrix times an orthogonal matrix.

Note that the butterfly factor matrices constructed in the proof of Lemma I.2 each have at most one non-zero entry per column.

Hence, there are 4 cases to consider.

Note that matrices with at most one non-zero entry are exhausted by Cases 1 and 2.

Case 1:

In the last two cases, O is a 2 × 2 rotation matrix, which is commonly known to be orthogonal.

Assume that we perform the above decomposition on all of the blocks of H 2 k in parallel, therefore expressing H 2 k = D O .

We now have

is the product of three orthogonal matrices, and thus orthogonal.

Additionally, the construction of P 2 k ensures that P * 2 k O P 2 k is butterfly factor.

10 Hence, H 2 k can be expressed as the product of a diagonal matrix and an orthogonal butterfly factor, as desired.

Now, we show that this decomposition of butterfly factors implies Lemma K.4.

By performing this decomposition in parallel on each butterfly factor, we conclude that any butterfly factor matrix H

We complete the argument by induction on n. The base case n = 2 holds by the observation about butterfly factor matrices above.

Assume that any horizontal step matrix of size n 2 × n 2 can be expressed as a diagonal matrix times an orthogonal butterfly matrix.

Now, consider the n × n horizontal step matrix H. From Lemma I.2, H can be expressed as

where H 1 , H 2 are n 2 × n 2 horizontal step matrices.

By our inductive hypothesis,

n D 1 is a butterfly factor, and therefore can be expressed as

with O ∈ B, as desired.

10 Conjugation by P 2 k is an isomorphism from 2 k × 2 k butterfly factors onto block diagonal matrices with 2 k−1 , 2 × 2 blocks.

Therefore, conjugation by P −1 2 k = P * 2 k maps a block diagonal matrix to a butterfly factor.

11 Note that a block diagonal matrix composed of orthogonal matrices is, itself, orthogonal.

Just as with the BB * hierarchy, the decomposition of vertical step matrices falls out as an immediate corollary to the horizontal step matrix proof.

Corollary K.3.

Let V be a vertical step matrix.

Then we can decompose V = O * D, where D is a diagonal matrix and O * ∈ B * .

Now that we have argued about the decomposition of permutation and step matrices in the OBB hierarchy, we can leverage the construction from Lemma I.1 to argue about matrices with at most n NNZ.

Corollary K.4.

Let S be an n × n matrix with at most n NNZ.

Then, S ∈ (OBB) 5 .

Proof.

We use the construction from Lemma I.1, along with Lemma K.4 and Corollary K.3, to express S as:

with each O i ∈ B, each O j ∈ B * , and each D k diagonal.

Noting that O 1 and O 5 are permutations, we make use of Lemma K.3 to re-express S as:

Note that each M ∈ OBB.

Hence, S ∈ (OBB) 5 , as desired.

Just as in Appendix I, we would like to extend this orthogonal-based construction to capture matrices of general sparsity.

To accomplish this, we introduce an addition closure lemma analogous to Lemma K.5 for the OBB hierarchy.

With Lemma K.5, we arrive at the following Corollary on general orthogonal sparsity.

Corollary K.5.

Let S be an n × n matrix with s NNZ.

Then, S ∈ (OBB)

Proof.

Just as in the proof of Theorem 3, we accomplish this using a sum of s n matrices of at most n NNZ.

For handling the sum of matrices, we need to appeal to Lemma K.5.

To conclude the argument, we give the proof of Lemma K.5.

Proof of Lemma K.5.

For each 1 ≤ i ≤ m, let E Ai ∈ F ek×ek be defined such that A i = SE Ai S * (with S as in Definition 2.4).

Note that E Ai ∈ (OBB) w .

Consider matrices of the form:

Note that K, a block diagonal matrix composed of matrices in (OBB) w , is itself in (OBB) w since was not yet used in L w ) to conclude that OL w ∈ B. Similarly, since no btterfly factor from B (4ek) 2ek has been used in R 1 , we may fold P into R 1 to conclude that R 1 P ∈ B * .

Finally, we address the scalar multiple of √ 2 by multiplying all entries of any diagonal matrix in the decomposition of K by √ 2.

Hence, we may conclude that M i ∈ (OBB) w .

Through repeated application (m times) of the identity

we see that

Therefore, M ∈ (OBB) mw .

Next, we note that

We would like to show that we can fold Q into the rightmost OBB factor of M. The rightmost matrix in the decomposition of M is P. Note that

.

Just as earlier, the factor of √ 2 can be multiplied through any diagonal matrix.

Also, these two orthogonal butterfly factor matrices can be folded into the the rightmost R matrix (the decomposition of K above does not use these two, rightmost butterfly factors).

Hence,

Just as in Theorem 1, we can use the sparsity result in Lemma K.5 to place matrices with low-depth (linear) arithmetic circuits for matrix vector multiplication in the OBB hierarchy.

Corollary K.6.

Let M be an n × n matrix such that matrix-vector multiplication of M times an arbitrary vector v can be represented as a be a linear arithmetic circuit C comprised of s gates (including inputs) and having depth d. Then, M ∈ (OBB)

Proof.

We use the construction given in the proof of Theorem 1.

Corollaries K.4 and K.2 allow us to recover the same width and expansion factor with the OBB hierarchy.

We show that for any neural network with ReLU nonlinearities and whose weight matrices have arithmetic circuits with few gates, its linear network counterpart (obtained by removing all the ReLU's) also has an arithmetic circuit with not too many more gates.

This implies that in trying to find the smallest arithmetic circuit augmented with ReLU gates to represent a ReLU network, one might as well try to find the smallest arithmetic circuits that represent the matrix-vector multiplication of each weight matrix.

Proposition 2.

Consider a neural network architecture consisting of L layers with weight matrices W 1 , . . .

, W L ∈ F n×n and ReLU nonlinearity in between.

Suppose that matrix-vector multiplication of W i times an arbitrary vector v can be represented as a linear arithmetic circuit with s i gates (including inputs Proof of Proposition 2.

To compute the output of the network ReLU(W L (. . .

ReLU(W 1 v))), we first compute the matrix-vector product W 1 v with an arithmetic circuit of s 1 gates by assumption, and use n other ReLU gates to compute the pointwise ReLU.

Then we repeat the process for layer 2, 3, . . .

, L, using the arithmetic circuits of W 1 , . . .

, W L and Ln additional gates for ReLU.

In total we obtain an arithmetic circuit augmented with ReLU gates with L i=1 s i + Ln total gates.

Conversely, to build an arithmetic circuit augmented with ReLU gates to compute W 1 v, . . .

, W L . . .

W 1 v, we pass v and then −v through the circuit that computes ReLU(W 1 x) for an arbitrary x to get ReLU(W 1 v) and ReLU(−W 1 v).

Noting that x = ReLU(x) − ReLU(−x), we can use n additional gates to compute W 1 v from ReLU(W 1 v) and ReLU(−W 1 v).

Repeat the process for layer 2, 3, . . .

, L (for example, pass W 1 v and −W 1 v to the circuit that computes W 2 x for an arbitrary x on layer 2).

Overall we need to double the circuits that computes all the activations of the network ReLU (W 1 v) , . . .

, ReLU(W L . . .

ReLU(W 1 v)), requiring 2s gates.

We also need n additional gates per layer to compute the negation of the input to that layer (e.g. computing −v from v), and n additional gates per layer to subtract the output of the ReLU circuit (e.g. computing W 1 v from ReLU(W 1 v) and ReLU(−W 1 v).)

Therefore we can construct an arithmetic circuit augmented with ReLU gates with 2s + 2L total gates that computes the activations of the network without ReLU W 1 v, . . .

, W L . . .

W 1 v.

We now prove an asymptotic bound on the VC dimension of a ReLU network whose weight matrices are kaleidoscope matrices with bounded width and expansion.

Proposition 3.

Let F be the class of ReLU neural networks consisting of L layers, where each layer is a K-matrix with width and expansion bounded by some constant C. Suppose that the network has W total parameters.

Let sign F denote the corresponding classification functions: {x → sign f (x) : f ∈ F}. Then this class has VC dimension:

VCdim(sign F) = O(LW log W ).

We leverage the result from Thomas et al. (2018) for the case where the entries of the weight matrices interact multiplicatively, but with polynomially bounded degrees.

This proof is similar to the VC bound for ReLU networks whose weight matrices are butterfly matrices (Dao et al., 2019) .

Proof.

To use Theorem 3 of Thomas et al. (2018) , we simply need to check that the entries of the linear layer, as polynomials of the parameters, has degree at most c 1 m c2 l for some universal constant c 1 , c 2 > 0, where m l is the size of output of the l-th layer.

If the network weight matrices are K-matrices with bounded width and expansion, each weight matrix is a product of at most c 3 log m l sparse factors, for some universal constant c 3 > 0.

This means that the degree is polynomially bounded, which satisfies the condition of the theorem.

Therefore the VC dimension is bounded to be almost linear in the number of parameters:

VCdim(sign F) = O(LW log W ).

We give a quick overview of arithmetic circuits.

This is a model of computation that has been studied for numerous computational problems (and is the basic model for algebraic complexity theory).

For our purposes, we will exclusively focus on arithmetic circuits for the matrix-vector multiplication problem.

For a more detailed exposition, the reader is referred to the standard book on this topic (Bürgisser et al., 2013) .

Definition M.1 (Arithmetic Circuits).

An arithmetic circuit that computes y = Ax (for A ∈ F m×n ) has n input gates (corresponding to x[0], . . .

, x[n − 1]) and m output gates (corresponding to y[0], . . . , y[m − 1]).

All the internal gates correspond to addition, subtraction, multiplication and division 12 over the underlying field F. The circuit is also allowed to use constants from F for 'free.'

The definition of the internal gates can depend on A (as well as x of course).

In other words, one can 'bake' the knowledge about A into the circuit.

The size s of a circuit is n plus the number of addition, multiplication, subtraction and division gates used in the circuit.

The depth d of a circuit is the minimum number of layers such that all gates in a given layer take as its input gates from previous layers.

One drawback of arithmetic circuits (especially for infinite fields e.g. F = R, which is our preferred choice in this work) is that they assume operations over F can be performed exactly.

In particular, it ignores precision issues involved with real arithmetic.

Nonetheless, this model turns out to be a very useful model in reasoning about the complexity of doing matrix-vector multplication for any family of matrices.

Perhaps the strongest argument in support of arithmetic circuits is that a large (if not an overwhelming) majority of matrix-vector multplication algorithm also imply an arithmetic circuit of size comparable to the runtime of the algorithm (and the depth of the circuit roughly correponds to the time taken to compute it by a parallel algorithm).

For example consider the obvious algorithm to compute Ax 14 One thing to note about the arithmetic circuit above is that all the multplications involve at least one input that is a constant from F (recall that we can assume that the entries of A are constants that can be used to build the circuit).

This leads to the following important sub-class of arithmetic circuits: Definition M.2 (Linear Arithmetic Circuits).

An arithmetic circuit is called a linear arithmetic circuit if it only uses addition, subtraction and multiplication.

Further, every multiplcation has a fixed constant from F as at least one of its two inputs.

In other words, all gates in the circuit are linear functions of their inputs (i.e. of the form ax + by for fixed constants a, b ∈ F).

Intuitively for the matrix-vector multiplication, it makes sense to consider linear arithmetic circuits since the final function we want to compute Ax is indeed a linear function of its inputs.

For inifinite fields (e.g. F = R or F = C), it turns out that this is essentially without loss of generality:

Theorem 4 ((Bürgisser et al., 2013)).

Let F be an infinite field.

Any (general) arithmetic circuit to compute Ax over F of size s and depth d can be converted into a linear arithmetic circuit of size O(s) and depth O(d).

12 Here we assume all the gates have two inputs.

13 The input layer corresponding to the input gates does not contriubte to the depth.

14 The claim on the depth follow from the fact that each of the sums The above result implies that for asymptotic considerations, linear arithmetic circuits for matrix-vector multiplication are equivalent to general arithmetic circuits.

One important property of linear arithmetic circuits of depth d, which we will use in our arguments, is that such a circuit can be equivalently represented as product of d sparse matrices (see the proof of Theorem 1 for the precise derivation 16 ).

As mentioned earlier, a vast majority of efficient matrix vector multiplication algorithms are equivalent to small (both in size and depth) linear arithmetic circuit.

For example the FFT can be thought of as an efficient arithmetic circuit to compute the Discrete Fourier Transform (indeed when one converts the linear arithmetic circuit for FFT into a matrix decomposition, 17 then each matrix in the decomposition is a butterfly factor, with each block matrix in each factor being the same).

For an illustration of this consider the DFT with n = 4 as illustrated in Figure 11 .

Finally, Figure 13 is representation of the arithmetic circuit of Figure 12 as a product of a butterfly matrix and (the bit-reversal) permutation.

We note that our generic arithmetic circuit to decomposition into BB * is not as tight as in Figure 13 .

One reason for the vast majority of existing efficient matrix vector algorithms leading to (linear) arithmetic circuits is that they generally are divide and conquer algorithms that use polynomial operations such as polynomial multiplication or evaluation (both of which themselves are divide and conquer algorithms that use FFT as a blackbox) or polynomial addition.

Each of these pieces are well known to have small (depth and size) linear arithmetic circuits (since FFT has these properties).

Finally, the divide and conquer structure of the algorithms leads to the circuit being of low depth.

See the book of Pan (Pan, 2001) for a more elaborate description of this connection.

In fact, the recent work of De Sa et al. (De Sa et al., 2018) makes this fact explicit and presents the most general known structure on matrices that imply near-linear size linear arithmetic circuits for the corresponding matrix vector multiplication.

Their work combines two separate classes of structures matrices-orthogonal polynomial transforms (Driscoll et al., 1997; Szegö, 1967) as well as matrices with low displacement rank (Kailath et al., 1979; Olshevsky & Shokrollahi, 2000)

-and presents a linear class of linear arithmetic circuits to solve their matrix vector multiplication problem.

We note that structured matrices with low displacement rank have been used to replace fully connected layers in some neural network architectures (Sainath et al., 2013; Thomas et al., 2018) .

@highlight

We propose a differentiable family of "kaleidoscope matrices," prove that all structured matrices can be represented in this form, and use them to replace hand-crafted linear maps in deep learning models.