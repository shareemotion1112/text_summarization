Modern neural networks often require deep compositions of high-dimensional nonlinear functions (wide architecture) to achieve high test accuracy, and thus can have overwhelming number of parameters.

Repeated high cost in prediction at test-time makes neural networks ill-suited for devices with constrained memory or computational power.

We introduce an efficient mechanism, reshaped tensor decomposition, to compress neural networks by exploiting three types of invariant structures: periodicity, modulation and low rank.

Our reshaped tensor decomposition method exploits such invariance structures using a technique called tensorization (reshaping the layers into higher-order tensors) combined with higher order tensor decompositions on top of the tensorized layers.

Our compression method improves low rank approximation methods and can be incorporated to (is complementary to) most of the existing compression methods for neural networks to achieve better compression.

Experiments on LeNet-5 (MNIST), ResNet-32 (CI- FAR10) and ResNet-50 (ImageNet) demonstrate that our reshaped tensor decomposition outperforms (5% test accuracy improvement universally on CIFAR10) the state-of-the-art low-rank approximation techniques under same compression rate, besides achieving orders of magnitude faster convergence rates.

Modern neural networks achieve unprecedented accuracy over many difficult learning problems at the cost of deeper and wider architectures with overwhelming number of model parameters.

The large number of model parameters causes repeated high cost in test-time as predictions require loading the network into the memory and repeatedly passing the unseen examples through the large network.

Therefore, the model size becomes a practical bottleneck when neural networks are deployed on constrained devices, such as smartphones and IoT cameras.

Compressing a successful large network (i.e., reducing the number of parameters), while maintaining its performance, is non-trivial.

Many approaches have been employed, including pruning, quantization, encoding and knowledge distillation (see appendix A for a detailed survey).

A complementary compression technique, on top of which the aforementioned approaches can be used, is low rank approximation.

For instance, singular value decomposition (SVD) can be performed on fully connected layers (weights matrices) and tensor decomposition on convolutional layers (convolutional kernels).

Low rank approximation methods can work well and reduce the number of parameters by a factor polynomial in the dimension only when the weight matrices or convolutional kernels have low rank structures, which might not always hold in practice.

We propose to exploit additional invariant structures in the neural network for compression.

A set of experiments on several benchmark datasets justified our conjecture (Section 4): large neural networks have some invariant structures, namely periodicity, modulation and low rank, which make part of the parameters redundant.

Consider this toy example of a vector with periodic structure [1, 2, 3, 1, 2, 3, 1, 2, 3] or modulated structure [1, 1, 1, 2, 2, 2, 3, 3, 3] in FIG23 .

The number of parameters needed to represent this vector, naively, is 9.

However if we map or reshape the vector into a higher order object, for instance, a matrix [1,1,1;2,2,2;3,3,3] where the columns of the matrix are repeated, then apparently this reshaped matrix can be decomposed into rank one without losing information.

Therefore only 6 parameters are needed to represent the original length-9 vector. [1, 2, 3, 1, 2, 3, 1, 2, 3] Periodic structure [1, 1, 1, 2, 2, 2, 3, 3, 3] [ ] FIG23 : A toy example of invariant structures.

The periodic and modulated structures are picked out by exploiting the low rank structure in the reshaped matrix.

Although the invariant structures in large neural networks allow compression of redundant parameters, designing a sophisticated way of storing a minimal representation of the parameters (while maintaining the expressive power of the network) is nontrivial.

To solve this problem, we proposed a new framework called reshaped tensor decomposition (RTD) which has three phases:1.

Tensorization.

We reshape the neural network layers into higher-order tensors.• For instance, consider a special square tensor convolutional kernel T ∈ R D×D×D×D , we reshape T into a higher m-order tensor 2.

Higher-order tensor decomposition.

We deploy tensor decomposition (a low rank approximation technique detailed in section 3) on the tensorized layers to exploit the periodic, modulated as well as low rank structures in the original layers.• A rank-R tensor decomposition of the above 4-order tensor T will result in R number of components (each contains 4D parameters), and thus 4DR number of parameters in totalsmaller than the original D 4 number of parameters if R is small.• A rank-R tensor decomposition of the above reshaped m-order kernel tensor T ′ maps the layer into m + 1 narrower layers.

The decomposition will result in R number of components with mD 4 m parameters and thus mD 4 m R in total -better than the 4DR number of parameters required by doing tensor decomposition on the original tensor T (D is usually large).

Now the weights of the tensorized neural networks are the components of the tensor, i.e., result of the tensor decomposition.

However, decomposing higher order tensors is challenging and known methods are not guaranteed to converge to the minimum error decomposition (Hillar & Lim, 2013) .

Therefore fine tuning is needed to achieve high performance.3.

Data reconstruction-based sequential tuning.

We fine-tune the parameters using a data reconstruction-based sequential tuning (Seq) method which minimizes the difference between training output of the uncompressed and compressed, layer by layer.

Our Seq tuning is a novel approach inspired by a sequential training method proved to converge faster and achieve guaranteed accuracy using a boosting framework (Huang et al., 2017) .

Unlike traditional end-to-end (E2E) backpropagation through the entire network, Seq tunes individual compressed "blocks" one at a time, reducing the memory and complexity required during compression.

• Novel compression schemes.

We propose new reshaped tensor decomposition methods to exploit invariant structures for compressing the parameters in neural networks.

By first tensorizing the kernel/weights into a higher-order tensor, our reshaped tensor decomposition discovers extra invariant structures and therefore outperform existing "low rank approximation methods".• Efficient computational framework.

We introduce a system of tensor algebra that enables efficient training and inference for our compressed models.

We show that a tensor decomposition on the parameters is equivalent to transforming one layer into multiple narrower sublayers in the compressed model.

Therefore, other compression techniques (e.g. pruning) can be applied on top of our method by further compressing the sublayers returned by our method.• Sequential knowledge distillation.

We introduce Seq tuning to transfer knowledge to a compressed network from its uncompressed counterpart by minimizing the data reconstruction error block by block.

With our strategy, only one block of the network is loaded into the GPU "at each time", therefore allowing compression of large networks on moderate devices.

Furthermore, we show empirically that our strategy converges much faster than normal end to end tuning.• Comprehensive experiments.

We perform extensive experiments to demonstrate that our reshaped tensor decomposition outperforms state-of-the-art low-rank approximation techniques (obtains 5% higher accuracy on CIFAR10 under same compression rates).

Our experiments also show that our method scales to deep residual neural networks on large benchmark dataset, ImageNet.

Organization of the paper Section 2 introduces tensor operations, tensor decompositions and their representations in tensor diagrams.

In Section 3, we introduce convolutional layer diagram, review existing low-rank approximation techniques, and propose three new schemes to exploit additional invariant structures.

In Section 4 and Appendix B, we demonstrate by extensive experiments that our compression obtains higher accuracy than existing low-rank approximation techniques.

Appendix A surveys compression techniques and discuss how our method is related or complementary to existing techniques.

For simplicity, we will use tensor diagrams throughout the text.

However we provide a detailed appendix where the tensor operations are mathematically defined.

Notations An m-dimensional array T is defined as an m-order tensor T ∈ R I0×···×Im−1 .

Its (i 0 , · · · , i n−1 , i n+1 , · · · , i m−1 ) th mode-n fiber, a vector along the n th axis, is denoted as T i0,··· ,in−1,:,in+1,··· ,im−1 .

Tensor Diagrams.

Following the convention in quantum physics Cichocki et al. (2016) , FIG2 introduces graphical representations for multi-dimensional objects.

In tensor diagrams, an array (scalar/vector/matrix/tensor) is represented as a node in the graph, and its order is denoted by the number of edges extending from the node, where each edge corresponds to one mode (whose dimension is denoted by the number associated to the edge) of the multi-dimensional array.

Figure 3: Tensor operation illustration.

Examples of tensor operations in which M ∈ R J0×J1 , X ∈ R I0×I1×I2 and Y ∈ R J0×J1×J2 are input matrix/tensors, and T 1 ∈ R I1×I2×J0×J2 , T 2 ∈ R J0×I1×I2 , T 3 ∈

R I ′ 0 ×I1×I2×J0×J2 and T 4 ∈ R I0×I1×I2×J0×J2 are output tensors of corresponding operations.

Similar definitions apply to general mode-(i, j) tensor operations.

Tensor Operations.

In Figure 3 , we use some simple examples to introduce four types of tensor operations, which are higher-order generalization of their matrix/vector counterparts, on input tensors X and Y and input matrix M. In tensor diagram, an operation is represented by linking edges from the input tensors, where the type of operation is denoted by the shape of line that connects the nodes: solid line stands for tensor contraction / tensor multiplication, dashed line represents tensor convolution, and curved line is for tensor partial outer product.

The rigorous definitions of high-order general tensor operations are defined in Appendix D.Tensor Decompositions.

We introduce generalized tensor decomposition as the reverse mapping of the general tensor operations (detailed in Appendix F): given a set of operations and a tensor, the generalized tensor decomposition recovers the factors/components such that the operations on these factors result in a tensor approximately equal to the original one.

Several classical types of tensor decompositions (such as CANDECOMP/PARAFAC (CP), Tucker (TK) and Tensor-train (TT) decompositions) are introduced in Appendix F, and their applications on the convolutional kernel in FIG4 (defined in Section 3) are illustrated as tensor diagrams in FIG4 .

DISPLAYFORM0 (c) Tucker (TK) Figure ( f) (g) and (h) are three types of reshaped tensor decomposition for our tensorized kernel K ′ in (e) where the reshaping order m ∈ Z is chosen to be 3 for illustrative simplicity.

DISPLAYFORM1

A standard convolutional layer in neural networks is parameterized by a 4-order kernel K ∈ R H×W ×S×T where H, W are height/width of the filters, and S, T are the numbers of input/output channels.

The layer maps a 3-order input tensor U ∈ R X×Y ×S (with S number of feature maps of height X and width Y ) to another 3-order output tensor V ∈ R X ′ ×Y ′ ×T (with T number of feature maps of height X ′ and width Y ′ ) according to the following equation: DISPLAYFORM0 where d is the stride of the convolution.

With HW ST parameters, it takes O(HW ST XY ) operations (FLOPs) to compute the output V. The diagram of the convolutional layer is in Figure 5a .Plain Tensor Decomposition (PD) Traditional techniques compress a convolutional layer by directly factorizing the kernel K using tensor decompositions Jaderberg et al. (2014); Lebedev et al. (2014) ; Kim et al. (2015) , such as CANDECOMP/PARAFAC (CP), Tucker (TK) and Tensor-train (TT) decompositions.

For example, consider a Tensor-train decomposition on K, the kernel can be factorized and stored as DISPLAYFORM1 Rt×T , which only requires (SR s + HR s R + W R t R + T R t ) parameters as illustrated in FIG4 .

The decomposition is rigorously defined element-wisely as DISPLAYFORM2 We defer the details of using CP and TK to Appendix G, although their tensor diagrams are illustrated in Figures 4b, 4c and 4d and their complexities are summarized in TAB2 DISPLAYFORM3 Figure 5: Convolutional layer diagram.

Input U is passed through the layer kernel K. The forward propogation operation of an uncompressed layer, a plain tensor decomposition compressed layer and our reshaped tensor decomposition compressed layer are illustrated in (a), (b) and (c) respectively.

Obviously, M as a rank-1 matrix can be represented by two length-L 2 vectors a and b, resulting in a total of 2L 2 parameters.

However, if we reshape the matrix M into a 4-order tensor T ∈ R L×L×L×L , it can be factorized by CP decomposition as DISPLAYFORM0 , and represented by four length-L vectors, requiring only 4L parameters.

We refer the process of reshaping an array into a higher-order tensor as tensorization, and the use of tensor decomposition following tensorization as reshaped tensor decomposition (RTD).

Therefore, the example above demonstrates that RTD discovers additional invariant structures that baseline plain tensor decomposition (PD) fails to identify.

Reshaped Tensor Decomposition (RTD) Inspired by this intuition, we tensorize the convolutional kernel K into a higher-order tensor K ′ ∈ H×W ×S0×···×Sm−1×T0×···×Tm−1 .

Correspondingly, we define an equivalent tensorized convolutional layer to Equation 3.1, by further reshaping input U and output V into higher-order tensors DISPLAYFORM1 Now we can compress the convolutional layer by factorizing the tensorized kernel K ′ by tensor decompositions, and name the schemes using CP, Tucker and Tensor-train as reshaped CP (r-CP), reshaped Tucker (r-TK) and reshaped Tensor-train (r-TT) respectively.

For example, consider a r-TT decomposition on K ′ , the tensorized kernel can now be stored in m+1 FIG4 ).

The decomposition scheme is rigorously defined element-wisely as DISPLAYFORM2 DISPLAYFORM3 We defer the detailed descriptions of r-CP and r-TK to Appendix I, but we illustrate their tensor diagrams in Figures 4f and 4g and summarize their complexities in TAB2 .Sequential Tuning Tensor decompositions provide weight estimates in the tensorized convolutional layers.

However, decomposing higher order tensors is challenging and known methods are not guaranteed to converge to the minimum error decompositions (Hillar & Lim, 2013) .

Therefore fine tuning is needed to restore high performance.

Analogous to Huang et al. (2017) , our strategy of data reconstruction-based sequential tuning (Seq) sequentially fine-tunes the parameters layer by layer, using backpropagation to minimize the difference between the outputs from uncompressed layer V ′ and the tensorized compressed layer V ′

.Computational Complexity As shown in Figure 5c , reshaped tensor decomposition maps a layer into multiple (and thus deeper) narrower layers, each of which has width R l that is usually smaller than the original width T .

We design efficient algorithms for forward/backward propagations for prediction/fine-tuning on these modified layers using tensor algebra.

A naive forward and backward propagation mechanism is to explicitly reconstruct the original kernel K ′ using the factors {K l } m−1 l=0 , which however makes propagations highly inefficient as shown in Appendix F, G and I. Alternatively, we propose a framework where both propagations are evaluated efficiently without explicitly forming or computing the original kernel.

The key idea is to interact the input U ′ with each of the factors K l individually.

Taking r-TT as an example, we plug the decomposition 3.4 into 3.3, then the computation of V ′ is reduced into m + 1 steps: DISPLAYFORM4 where U l is the intermediate result after interacting with K l−1 , and U 0 = U. Each step in 3.5 takes O(max(S, T ) 1+ 1 m RXY ) operations, while the last step in 3.6 requires O(HW T RXY )1 .

Therefore, the time complexity for the forward pass is O((m max(S, T ) DISPLAYFORM5 Backpropagation is derived and analyzed in Appendix I, and the analyses of other decomposition are in Appendix G and I, but we summarize their computational complexities in TAB2 , 10 and 12.Parallel Computational Complexity Tensor algebra allows us to implement the propagations 3.5 and 3.6 in parallel given enough computational resources, further speeding up prediction.

The parallel tine complexities of prediction 2 with our RTD implementation is displayed in Table 2 .

The prediction time complexity of RTD outperforms the baseline PD, whereas the PD outperforms the original convolutional layers as R ≪ N and m ≥ 3.

O TAB2 .

Table 2 : Parallel time complexity of forward pass using various types of tensor decompositions on convolutional layers.

The uncompressed parallel complexity of forward pass is O(k 2 N ).

DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 1 The optimal complexity of tensor algebra is NP complete in general Lam et al. (1997) , therefore the complexity presented in this paper is the complexity of our implementation.2 Assuming adding n terms takes n time in parallel for memory efficiency, although it could be O(log n).

We evaluate our reshaped tensor decomposition method on the state-of-art networks for a set of benchmark datasets: we evaluate convolutional layer compression on ResNet-32 He et al. (2016) for CIFAR-10; we evaluate fully-connected layer compression on MNIST; and we evaluate the scalability of our compression method on ResNet-50 He et al. (2016) for ImageNet (2012) dataset.

The baseline we compare against is the state-of-the-art low-rank approximation methods called plain tensor decomposition (PD), as other compression methods are complementary and can be used on top of our reshaped tensor decomposition (RTD) method.

All types of tensor decomposition (CP, TK, and TT) in baseline PD will be evaluated and compared with corresponding types of tensor decomposition (r-CP, r-TK and r-TT) in our RTD method.

Our primary contribution is to introduce a new framework, reshaped tensor decomposition, that picks out additional invariance structure such as periodicity and modulation, which the low rank approximation baseline, plain tensor decomposition, fails to find.

Now we demonstrate that our RTD maintains high accuracy even when the networks are highly compressed on CIFAR-10.

We refer to traditional backpropogation-based tuning of the network as end-to-end (E2E) tuning, and to our proposed approach that trains each block individually as data reconstruction-based sequential (Seq) tunning.

Our algorithm achieves 5% higher accuracy than baseline on ResNet-34 CIFAR10.

As in Table 3, using baseline CP decomposition with end-to-end tuning, ResNet-34 is compressed to 10% of its original size, reducing the accuracy from 93.2% to 86.93%.

Our reshaped tensor decomposition using r-CP, paired with Seq tuning, increases the accuracy to 91.28% with the same 10% compression rate -a performance loss of 2% with only 10% of the number of parameters.

It achieves further aggressive compression -a performance loss of 6% with only 2% of the number of parameters.

We observe similar trends (higher compression and higher accuracy) for Tensor-train decomposition.

The structure of the Tucker decomposition (see section I) makes it less effective with very high compression, since the "internal structure" of the network reduces to very low rank, which may lose necessary information.

Increasing the network size to 20% of the original provides reasonable performance on CIFAR-10 for Tucker as well.

Seq tuning, reshaped tensor decomposition, or both?(1) We present the effect of different tuning methods on accuracy in TAB7 .

Other than at very high compression rate (5% column in TAB7 ), Seq tuning (Seq) consistently outperforms end-to-end (E2E) tuning.

In addition, Seq tuning is also much faster and leads to more stable convergence compared to end-to-end tuning.

FIG6 plots the compression error over the number of gradient updates for various tuning methods.

(2) We present the effect of different compression methods on accuracy in Table 5 .

Interestingly, if our RTD is used, the test accuracy is restored for even very high compression ratios 3 .

These results confirm the existence of extra invariant structure in the parameter space of deep neural networks.

Such invariant structure is picked up by our proposed aproach, tensorization combined with low rank approximation (i.e., our RTD), but not by low rank approximation itself (i.e., baseline PD).

Therefore, our results show that RTD and Seq tuning are symbiotic, and both are necessary to simultaneously obtain a high accuracy and a high compression rate.

Decomp.

Table 5 : Percentage accuracy of our RTD vs. baseline PD using Seq tuning on CIFAR10.

Scalability Finally, we show that our methods scale to state-of-the-art large networks, by evaluating performance on the ImageNet 2012 dataset on a 50-layer ResNet (uncompressed with 76.05% accuracy).

TAB9 shows the accuracy of RTD (TT decomposition) with Seq tuning compared to plain tensor decomposition with E2E tuning and the uncompressed network, on ResNet-50 with 10% compression rate.

TAB9 shows that Seq tuning of RTD is faster than the alternative.

This is an important result because it empirically validates our hypotheses that (1) our RTD compression captures the invariance structure of the ResNet (with few redundancies) better and faster than the baseline PD compression, (2) data reconstruction Seq tuning is effective even on the largest networks and datasets, and (3) our proposed efficient RTD compression methods scale to the state-of-the-art neural networks.

We describe an efficient mechanism for compressing neural networks by tensorizing network layers.

We implement tensorized decompositions to find approximations of the tensorized kernel, potentially preserving invariance structures missed by implementing decompositions on the original kernels.

We extend vector/matrix operations to their higher order tensor counterparts, providing systematic notations and libraries for tensorization of neural networks and higher order tensor decompositions.

As a future step, we will explore optimizing the parallel implementations of the tensor algebra.

Recognition, pp.

1984 Recognition, pp.

-1992 Recognition, pp. , 2015 .

A RELATED WORKS A recent survey Cheng et al. (2017) reviews state-of-the-art techniques for compressing neural networks, in which they group the methods into four categories: (1) low-rank factorization; (2) design of compact filters; (3) knowledge distillation; and 4) parameters pruning, quantization and encoding.

Generally, our decomposition schemes fall into the category of low-rank factorization, but these schemes also naturally lead to novel designs of compact filters in the sublayers of the compressed network.

On the other hand, our strategy of sequential tuning is an advanced scheme of knowledge distillation that transfers information from pre-trained teacher network to compressed student network block by block.

Furthermore, our method is complementary to the techniques of parameters pruning, quantization and encoding, which can be applied on top of our method by further compressing the parameters in the sublayers returned by tensor decomposition.• Low-rank Factorization.

Low-rank approximation techniques have been used for a long time to reduce the number of parameters in both fully connected and convolutional layers.

Pioneering papers propose to flatten/unfold the parameters in convolutional layer into matrices (a.k.a matricization), followed by (sparse) dictionary learning or matrix decomposition (Jaderberg et al., 2014; Denton et al., 2014; Zhang et al., 2015) .

Subsequently in Lebedev et al. (2014) ; Kim et al. (2015) , the authors show that it is possible to compress the tensor of parameters directly by standard tensor decomposition (in particular CP or Tucker decomposition).

The groundbreaking work (Novikov et al., 2015) demonstrates that the parameters in fully connected layer can be efficiently compressed by tensor decomposition by first reshaping the matrix of parameters into higher-order tensor, and the idea is later extended to compress LSTM and GRU layers in recurrent neural networks (Yang et al., 2017) .

Concurrent to Wang et al. (2018) , our paper extends this basic idea to convolutional layer by exploiting the invariant structures among the filters.

Different from Wang et al. (2018) that only focuses Tensor-ring decomposition, we investigates, analyzes and implements a boarder range of decomposition schemes, besides other benefits discussed below.• Design of Compact Filters.

These techniques reduce the number of parameters by imposing additional constraints on linear layers (fully connected or convolutional).

For example, the matrix of parameters in fully connected layer is restricted to circular BID1 , Toeplitz/Vandermonde/Cauchy (Sindhwani et al., 2015) , or multiplication of special matrices (Yang et al., 2015) .

Historically, convolutional layer is proposed as a compact design of fully connected layer, where spatial connections are local (thus sparse) with repeated weights.

Recent research further suggests to use more compact convolutional layers, such as 1 × 1 convolutional layer (Szegedy et al., 2017; Wu et al., 2017) (where each filter is simply a scalar), and depthwise convolutional layer (Chollet, 2016) (where connections between the feature maps are also sparse).

In our paper, we show that the sublayers returned by our decomposition schemes are in fact 1 × 1 depthwise convolutional layers, combing advantages from both designs above.• Knowledge Distillation.

The algorithms of knowledge distillation aim to transfer information from a pre-trained teacher network to a smaller student network.

In BID0 Hinton et al. (2015) , the authors propose to train the student network supervised by the logits (the vector before softmax layer) of the teacher network.

Romero et al. (2014) extends the idea to matching the outputs from both networks at each layer, up to an affine transformation.

Our Seq tuning strategy is therefore similar to Romero et al. FORMULA70 , but we use identical mapping instead of affine transformation, and train the compressed network block by block.• Pruning, Quantization and Encoding.

Han et al. FORMULA70 proposes a three-step pipeline to compress a pre-trained network, by (1) pruning uninformative connections, (2) quantizing the remaining weights and (3) encoding the discretized parameters.

Since our decomposition schemes effectively transform one layer in the original network into the multiple sublayers, this pipeline can be applied by further compressing all sublayers.

Therefore, our method is complementary (and can be used independently) to the techniques in this pipeline.

Convergence Rate Compared to end-to-end, an ancillary benefit of Seq tuning is much faster and leads to more stable convergence.

FIG6 plots compression error over number of gradient updates for various methods.

(This experiment is for PD with 10% compression rate.)

There are three salient points: first, Seq tuning has very high error in the beginning while the "early" blocks of the network are being tuned (and the rest of the network is left unchanged to tensor decomposition values).

However, as the final block is tuned (around 2 × 10 11 gradient updates) in the figure, the errors drop to nearly minimum immediately.

In comparison, end-to-end tuning requires 50-100% more gradient updates to achieve stable performance.

Finally, the result also shows that for each block, Seq tuning achieves convergence very quickly (and nearly monotonically), which results in the stair-step pattern since extra tuning of a block does not improve (or appreciably reduce) performance.

Performance on Fully-Connected Layers An extra advantage of reshaped tensor decomposition compression is that it can apply flexibly to fully-connected as well as convolutional layers of a neural network.

Table 7 shows the results of applying reshaped tensor decomposition compression to various tensor decompositions on a variant of LeNet-5 network LeCun et al. (1998) .

The convolutional layers of the LeNet-5 network were not compressed, trained or updated in these experiments.

The uncompressed network achieves 99.31% accuracy.

Table 7 shows the fully-connected layers can be compressed to 0.2% losing only about 2% accuracy.

In fact, compressing the dense layers to 1% of their original size reduce accuracy by less then 1%, demonstrating the extreme efficacy of reshaped tensor decomposition compression when applied to fully-connected neural network layers.

Table 7 : Reshaped tensor decomposition combined with sequential for fully-connected layers on MNIST.

The uncompressed network achieves 99.31% accuracy.

Symbols: Lower case letters (e.g. v) are used to denote column vectors, while upper case letters (e.g. M) are used for matrices, and curled letters (e.g. T ) for multi-dimensional arrays (tensors).

For a tensor T ∈ R I0×···×Im−1 , we will refer to the number of indices as order, each individual index as mode and the length at one mode as dimension.

Therefore, we will say that T ∈ R I0×···×Im−1 is an m-order tensor which has dimension I k at mode-k.

Tensor operations are extensively used in this paper: The tensor (partial) outer product is denoted as ⊗, tensor convolution as * , and finally × denotes either tensor contraction or tensor multiplication.

Each of these operators will be equipped with subscript and superscript when used in practice, for example × m n denotes mode-(m, n) tensor contraction (defined in Appendix D).

Furthermore, the symbol • is used to construct compound operations.

For example, ( * • ⊗) is a compound operator simultaneously performing tensor convolution and tensor partial outer product between two tensors.

Indexing: In this paragraph, we explain the usages of subscripts/superscripts for both multidimensional arrays and operators, and further introduce several functions that are used to alter the layout of multi-dimensional arrays.• Nature indices start from 0, but reversed indices are used occasionally, which start from −1.Therefore the first entry of a vector v is v 0 , while the last one is v −1 .•

For multi-dimensional arrays, the subscript is used to denote an entry or a subarray within an object, while superscript is to index among a sequence of arrays.

For example, M i,j denotes the entry at i th row and j th column of a matrix M, and M (k) is the k th matrix in a set of N matrices DISPLAYFORM0 For operators, as we have seen, both subscript and superscript are used to denote the modes involved in the operation.• The symbol colon ':' is used to slice a multi-dimensional array.

For example, M :,k denotes the k th column of M, and T :,:,k denotes the k th frontal slice of a 3-order tensor T .• Big-endian notation is adopted in conversion between multi-dimensional array and vectors.

Specifically, the function vec(·) flattens (a.k.a.

vectorize) a tensor T ∈ R I0×···×Im−1 into a vector DISPLAYFORM1 • The function swapaxes(·) is used to permute ordering of the modes of a tensor as needed.

For example, given two tensors U ∈ R I×J×K and V ∈ R K×J×I , the operation V = swapaxes(U) convert the tensor U into V such that V k,j,i = U i,j,k .•

The function flipaxis(·, ·) flips a tensor along a given mode.

For example, given a tensor U ∈ R I×J×K and V = flipaxis(U, 0), the entries in V is defined as DISPLAYFORM2 inner product of mode-k fiber of X and mode-l fiber of Y mode-k Tensor Multiplication DISPLAYFORM3 inner product of mode-k fiber of X and r th column of M mode-(k, l) Tensor Convolution DISPLAYFORM4 Hadamard product of mode-k fiber of X and mode-l fiber of Y DISPLAYFORM5 In this section, we introduce a number of tensor operations that serve as building blocks of tensorial neural networks.

To begin with, we describe several basic tensor operations that are natural generalization to their vector/matrix counterparts.

Despite their simplicity, these basic operations can be combined among themselves to construct complicated compound operators that are actually used in all designs.

We will analyze their theoretical sequential time complexities in details, and point out the implementational concerns along the way.

Although all these operations can in principle be implemented by parallel programs, the degree of parallelism depends on their particular software and hardware realizations.

Therefore, we will use the sequential time complexity as a rough estimate of the computational expense in this paper.

Tensor contraction Given a m-order tensor T (0) ∈ R I0×···×Im−1 and another n-order tensor T(1) ∈ R J0×···×Jn−1 , which share the same dimension at mode-k of T (0) and mode-l of T (1) ( i.e. DISPLAYFORM6 entries are computed as DISPLAYFORM7 Notice that tensor contraction is a direct generalization of matrix multiplication to higher-order tensor, and it reduces to matrix multiplication if both tensors are 2-order (and therefore matrices).

As each entry in T can be computed as inner product of two vectors, which requires I k = J l multiplications, the total number of operations to evaluate a tensor contraction is therefore O(( DISPLAYFORM8 , taking additions into account.

Notice that the analysis of time complexity only serves as a rough estimate for actual execution time, because we do not consider the factors of parallel computing and computer systems.

In practice, (1) the modes that are not contracted over can be computed in parallel, and summations can be computed in logarithmic instead of linear time; FORMULA91 The spatial locality of the memory layout plays a key role in speeding up the computation of tensor operations.

These arguments equally apply to all tensor operations in this paper, but we will not repeat them in the analyses for simplicity.

Tensor multiplication (Tensor product) Tensor multiplication (a.k.a.

tensor product) is a special case of tensor contraction where the second operant is a matrix.

Given a m-order tensor U ∈ R I0×···×Im−1 and a matrix M ∈ R I k ×J , where the dimension of U at mode-k agrees with the number of the rows in M, the mode-k tensor multiplication of U and M, denoted as V U × k M, yields another m-order tensor V ∈ R I0×···×I k−1 ×J×I k+1 ×···Im−1 , whose entries are computed as DISPLAYFORM9 Following the convention of multi-linear algebra, the mode for J now substitutes the location originally for I k (which is different from the definition of tensor contraction).

Regardlessly, the number of operations for tensor multiplication follows tensor contraction exactly, that is O(( DISPLAYFORM10 Tensor convolution Given a m-order tensor T (0) ∈ R I0×I1×···×Im−1 and another n-order tensor DISPLAYFORM11 The entries of T can be computed using any convolution operation * that is defined for two vectors.

DISPLAYFORM12 Here we deliberately omit the exact definition of vector convolution * , as it can be defined in multiple forms depending on the user case (Interestingly, the "convolution" in convolutional layer indeed computes correlation instead of convolution).

Correspondingly, the resulted dimension I ′ k at modek is determined by the chosen type of convolution.

For example, the "convolution" in convolutional layer typically yields I ′ k = I k (with same padding) or I ′ k = I k − J l + 1 (with valid padding).

Notice that vector convolution itself is generally asymmetric , i.e. u * v = v * u (except for the case of circular convolution).

For convenience, we can define its conjugate as * such that u * v = v * u. With this notations, Equation D.3a can also be written as D.3b.

Generally speaking, Fast Fourier Transform (FFT) plays a critical role to lower the computational complexities for all types of convolution.

In the case of tensor convolution, the number of required operations without FFT is O(( DISPLAYFORM13 .

That being said, FFT is not always necessary: if min(I k , J l ) < log (max(I k , J l )) (which is typical in convolutional layers, where I k is the height/width of the feature maps and J l is the side length of the square filters), computing the convolution without FFT is actually faster.

Furthermore, FFT can be difficult to implement (thus not supported by popular software libraries) if convolution is fancily defined in neural networks (e.g. dilated, atrous).

Therefore, we will assume that tensor convolutions are computed without FFT in subsequent sections unless otherwise noted.

Tensor outer product Given a m-order tensor T (0) ∈ R I0×I1×···×Im−1 and another n-order tensor T(1) ∈ R J0×J1×···×Jn−1 , the outer product of T (0) and DISPLAYFORM14 , concatenates all the indices of T (0) and T (1) , and returns a (m + n)-order tensor T ∈ R I0×···×Im−1×J0×···×Jn−1 whose entries are computed as DISPLAYFORM15 It is not difficult to see that tensor outer product is a direct generalization for outer product for two DISPLAYFORM16 Obviously, the number of operations to compute a tensor outer product explicitly is O(( DISPLAYFORM17 Tensor outer product is rarely calculated alone in practice because it requires significant amounts of computational and memory resources.

Tensor partial outer product Tensor partial outer product is a variant of tensor outer product defined above, which is widely used in conjunction with other operations.

Given a m-order tensor T (0) ∈ R I0×I1×···×Im−1 and another n-order tensor T (1) ∈ R J0×J1×···×Jn−1 , which share the same dimension at mode-k of T (0) and mode-l of DISPLAYFORM18 , whose entries are computed as DISPLAYFORM19 The operation bears the name "partial outer product" because it reduces to outer product once we fix the indices at mode-k of T (0) and mode-l of T (1) .

Referring to the computational complexity of tensor outer product, the number of operations for each fixed index is O(( DISPLAYFORM20 , therefore the total time complexity for the tensor partial outer product is O(( DISPLAYFORM21 , the same as tensor contraction.

• Similar to matrix multiplication, the operants in tensor operations are not commutative in general.

For example, neither DISPLAYFORM0 holds even if the dimensions at the specified modes happen to match.• Different from matrix multiplication, the law of associative also fails in general.

For example, DISPLAYFORM1 , mainly because tensor operations can change the locations of modes in a tensor.• However, both problems are not fundamental, and can be fixed by adjusting the superscripts and subscripts of the operators carefully (and further permute ordering of the modes in the result accordingly).

For example, DISPLAYFORM2 is properly performed.

Due to space limits, we can not develop general rules in this paper, and will derive such identities as needed.

In general, the take away message is a simple statement: Given an expression that contains multiple tensor operations, these operations need to be evaluated from left to right unless a bracket is explicitly supplied.

Compound operations: As building blocks, the basic tensor operations defined above can further combined to construct compound operations that perform multiple operations on multiple tensors simultaneously.

For simplicity, we illustrate their usage using two representative examples in this section.

More examples will arise naturally when we discuss the derivatives and backpropagation rules for compound operations in Appendix E.• Simultaneous multi-operations between two tensors.

For example, given two 3-order tensors DISPLAYFORM3 , where mode-(0, 0) partial outer product, mode-(1, 1) convolution and mode-(2, 2) contraction are performed simultaneously, which results in a 2-order tensor T of R For commonly used vector convolution, it is not difficult to show that number of operations required to compute the result T is O (R max(X, H) log(max(X, H))S) with FFT and O(RXHS) without FFT, as each of the R vectors in T is computed with a sum of S vector convolutions.• Simultaneous operations between a tensor and a set of multiple tensors.

For example, given a 3-order tensor U ∈ R R×X×S and a set of three tensors DISPLAYFORM4 , which performs mode-(0, 0) partial outer product with T (0) , mode-(1, 0) convolution with T (1) and mode-(2, 0) contraction with T (2) simultaneously.

In this case, a 5-order tensor V ∈ R R×X ′ ×P ×Q×T is returned, with entries calculated as DISPLAYFORM5 The analysis of time complexity of a compound operation with multiple tensors turns out to be a non-trivial problem.

To see this, let us first follow the naive way to evaluate the output according to the expression above: (1) each vector in the result V r,:,p,q,t can be computed with a sum of S vector convolutions, which requires O(XHS) operations; (2) and with RP QT such vectors in the result V, the time complexity for the whole compound operation is therefore O(RXHP RST ).

However, it is obviously not the best strategy to perform these operations.

In fact, the equations can be equivalently rewritten as DISPLAYFORM6 If we follows the supplied brackets and break the evaluation into three steps, it is not difficult to verify that these steps take O(RXST ), O(RXHP T ) and O(RX ′ HP T ) operations respectively, and result in a total time complexity of O(RXST + RXHP T + RX ′ P QT ) for the compound operation, which is far lower than the one with the naive way.

Unfortunately, it is an NP-hard problem to determine the best order (with minimal number of operations) to evaluate a compound operation over multiple tensors, therefore in practice the order is either determined by exhaustive search (if there are only a few tensors) or follows a heuristic strategy (if the number of tensors is large).The examples provided above are by no mean comprehensive, and in fact more complicated compound operations simultaneously perform multiple operations on multiple tensors can be defined, and we will see examples of them in the next section when we derive the backpropagation equations for the compound operations above.

Generally, compound operations over multiple tensors are difficult to flatten into mathematical expressions without introducing tedious notations.

Therefore, these operations are usually described by graphical representations, which are usually called tensor network in the physics literature (not to confuse with tensorial network in this paper).

Interested readers are referred to the monograph Cichocki et al. (2016) , which serves a comprehensive introduction to the application of tensor network in the field of machine learning.

All operations introduced in the last section, both basic and compound, are linear in their operants.

Therefore, the derivatives of the result with respect to its inputs are in principle easy to calculate.

In this section, we will explicitly derive the derivatives for all operations we have seen in Appendix D.These derivatives can be further combined with classic chain rule to obtain the corresponding backpropagation equations (i.e. how gradient of the loss function propagates backward through tensor operations), which are the cornerstones of modern feed-forward neural networks.

In the section, we show that these backpropagation equations can also be characterized by (compound) tensor operations, therefore their computational complexities can be analyzed similarly as in Appendix D.Interestingly, the backpropagation equations associated with a tensor operation, though typically appear to be more involved, share the same asymptotic complexities as in the forward pass (with tensor convolution as an exception).

This observation is extremely useful in the analyses of tensorial neural networks in Appendix G, H and I, which allows us to reuse the same number in the forward pass in the analysis of backward propagation.

In this section, we will assume for simplicity the loss function L is differentiable.

However, all derivatives and backpropagation equations equally apply when L is only sub-differentiable (piecewise smooth).

Also, we will focus on one step of backpropagation, therefore we assume the gradient of the loss function is known to us in prior.

Tensor contraction Recall the definition of tensor contraction in Equation D.1a, the partial derivatives of the result T with respect to its operants T (0) , T (1) can be computed at the entries level: DISPLAYFORM0 With classic chain rule, the derivatives of L with respect to T (0) and T (1) can be obtained through the derivative of Lwith respect to T .

DISPLAYFORM1 Though tedious at entries level, it can be simplified with tensor notations in this paper.

DISPLAYFORM2 where swapaxes(·) is used to align the modes of outputs.

Notice that the backpropagation equations are compound operations, even if the original operation is a basic one.

It is not difficult to show that the number of operations required for both backpropagation equations are O(( DISPLAYFORM3 , which are exactly the same as in the forward pass in Equation D.1a.

The result should not surprise us however, since the tensor contraction is a direct generalization to matrix multiplication (where backward propagation has exactly the same time complexity as the matrix multiplication itself).Tensor multiplication (Tensor product) As a special case of tensor contraction, the derivatives and backpropagation equations for tensor multiplication can be obtained in the same manner.

To begin with, the derivatives of V with respect to U and M can be computed from the definition in Equation D.2a.

DISPLAYFORM4 Subsequently, the derivatives of L with respect to U and M can be computed as with chain rule, DISPLAYFORM5 Again, the backpropagation equations above can be succinctly written in tensor notations.

DISPLAYFORM6 where the time complexities for both equations are O(( m−1 u=0 I u )J), which is identical to the forward pass in Equation D.2a (obviously since tensor multiplication is a special of tensor contraction).Tensor convolution Recall in the definition of tensor convolution in Equation D.3a, we deliberately omit the exact definition of vector convolution for generality.

For simplicity, we temporarily limit ourselves to the special case of circular convolution.

In this case, tensor convolution can be concretely defined by either equation below: DISPLAYFORM7 where Cir(·) returns a circular matrix of the input vector.

Concretely, given a vector v ∈ R I , the circular matrix Cir(v) is defined as Cir(v) i,j = v i−j(modI) .

Now, the derivatives of the result tensor T with respect to T (0) and T (1) can be obtained by matrix calculus.∂T i0,··· ,i k−1 ,:,i k+1 ,··· ,im−1,j0,··· ,j l−1 ,j l+1 ,··· ,jn−1 DISPLAYFORM8 Applying chain rule to the equations above, we arrive at two lengthy equations: DISPLAYFORM9 Cir T(1) j0,··· ,j l−1 ,:,j l+1 ,··· ,jn−1 ⊤ ∂L ∂T i0,··· ,i k−1 ,:,i k+1 ,··· ,im−1,j0,··· ,j l−1 ,j l+1 ,··· ,jn−1 (E.9a)

(1) j0,··· ,j l−1 ,r,j l+1 ,··· ,jn−1 DISPLAYFORM0 With notations of tensor operations, they can be greatly simplified as DISPLAYFORM1 Although these backpropagation equations are derived for the special case of circular convolution, they hold for general convolution if we replace * k l by its corresponding adjoint operator ( * DISPLAYFORM2 where the exact form of the adjoint operator ( * k l ) ⊤ depends on the original definition of vector convolution.

Generally, the trick to start with circular convolution and generalize to general cases is very useful to derive backpropagation equations for operations that convolution plays a part.

Despite varieties in the definitions of tensor convolution, the analyses of their time complexities of backpropagation equations are identical, since the numbers of operations only differ by a constant for different definitions (therefore asymptotically the same).

With FFT, the number of operations for these two backpropagation equations are O(( DISPLAYFORM3 Different from other operations, the time complexities for forward and backward passes are different (with circular convolution as an exception).

This asymmetry can be utilized in neural networks (whereI which can then be converted to tensor notations in this paper: DISPLAYFORM4 The number of operations required for both equations are O(( DISPLAYFORM5 , which are again identical to one in the forward pass in Equation D.4.Tensor partial outer product Finally, the derivatives of T with respect to T (0) and T (1) can be obtained from the definition of tensor partial outer product in Equation D.5a.∂T i0,··· ,i k−1 ,r,i k+1 ,··· ,im−1,j0,··· ,j l−1 ,j l+1 ,··· ,jn−1 DISPLAYFORM6 Again with chain rule, the backpropagation equations for T (0) and T (1) at the entries level are DISPLAYFORM7 ∂L ∂T i0,··· ,i k−1 ,r,i k+1 ,··· ,im−1,j0,··· ,j l−1 ,j l+1 ,··· ,jn−1 T(1) j0,··· ,j l−1 ,r,j l+1 ,··· ,jn−1 DISPLAYFORM8 Though the backpropagation equations above appear very similar to the ones for tensor contraction in Equations E.1a and E.1b, written in tensor notations, they are almost the same as the ones for tensor convolution in Equations E.10a and E.10b, except that ( * DISPLAYFORM9 It is not difficult to recognize the time complexity for the two equations above are O(( Compound operations Up to this point, we have developed the derivatives and backpropagation equations for all basic operations.

In this part, we will continue to show similar techniques above equally apply to compound operations, though slightly more involved, and derive the backpropagation equations for the examples we used in Appendix D. Though these equations are not immediately useful in later sections, the techniques to derive them are useful for all other compound operations.

Furthermore, these induced equations, which are more complicated than their original definitions, serve as complementary examples of compound operations to the ones in the last section.• Simultaneous multi-operations between two tensors.

In Appendix D, we introduced a compound operation (⊗ DISPLAYFORM10 2 ) on two tensors T (0) ∈ R R×X×S and T (1) ∈ R R×H×S , which returns a tensor T ∈ R R×X ′ .

Here, we recap its definitions as follows: DISPLAYFORM11 T r,: DISPLAYFORM12 r,:,s * T(1) r,:,s (E.18b)To ease the derivation, we use the trick to start with circular convolution: directly apply the chain rule, the backpropagation equations at entries level are obtained as follows: DISPLAYFORM13 Now we convert the equations above to tensor notations, and replace the circular convolutions with their adjoints to obtain general backpropagation rules: DISPLAYFORM14 For simplicity, we assume FFT is not used to accelerate the backpropagation equations.

In this case, the derivatives with respect to T (0) and T (1) can be computed in O(RHX ′ ST ) and O(RXX ′ ST ) operations respectively.

Again, the time complexities of forward and backward passes are not the same when a (compound) tensor operation contains convolution.• Simultaneous operations between a tensor and a set of multiple tensors.

Another compound operation presented in Appendix D is defined between a tensor U ∈ R R×X×S and a set of tensors DISPLAYFORM15 ∈ R H×Q and T (2) ∈ R S×T , which returns a tensor V ∈ R R×X ′ ×P ×Q×T .

Again, we recap its definitions in the following: DISPLAYFORM16 In order to derive the backpropagation rule for the core tensor U, we follow the standard procedure to (1) first obtain its entries level representation, and (2) explicitly convert it to tensor notations subsequently.

Concretely, the backpropagation equation in both formats are displayed as follows: DISPLAYFORM17 Notice that the equation above is indeed simultaneous multi-operations between a tensor and a set of multiple tensors, which combines two types of "basic" compound operations introduced in Appendix D. In principle, we can obtain backpropagation equations for {T (0) , T (1) , T (2) } in the same manner.

However, there is a simpler way to derive them by rewriting the definition as: DISPLAYFORM18 where U (0) , U (1) and U (2) are short-hand notations for U( * DISPLAYFORM19 .

With these notations, we are able to reduce these complex expressions to basic ones, by which we can reuse the backpropagation rules derived in this section: DISPLAYFORM20 The complicity of tensor operation culminates at this point: the equations above are examples of simultaneous multi-operations on multiple tensors, which we omitted in the discussion in Appendix D due to their complexity.

Although the expressions themselves suggest particular orderings to evaluate the compound operations, they are merely the traces of the techniques used in deriving them.

It is completely reasonable to reorganize the equations such that they can be computed with more efficient strategies: for instance, one can verify that the following set of equations is actually equivalent to the one above: DISPLAYFORM21 As discussed in Appendix D, the problem to find the optimal order to evaluate a compound operation over multiple tensors is NP-hard in general and usually we need to resort to heuristics to obtain a reasonably efficient algorithm.

Indeed, one can verify that the second set of equations is more efficient than the first one.

For this example, interested readers are encouraged to find the most efficient way by combinatoric search.

Tensor decompositions are natural extensions of matrix factorizations for multi-dimensional arrays.

In this section, we will review three commonly used tensor decompositions, namely For each of these decompositions, we will present their forms both at the entries level and in tensor notations introduced in Appendix D. When tensor decompositions are used in neural networks, a natural question to ask is how the backpropagation algorithm adapts to the decomposition schemes, i.e. how the gradient of the original tensor backpropagates to its factors.

In this section, we will follow the standard procedure in Appendix E to derive the corresponding backpropagation equation for each tensor decomposition.

Different from previous works (Novikov et al., 2015; Kossaifi et al., 2017b ) that use matrix calculus following matricization, we present the backpropagation equations directly in tensor notations, which makes our presentation concise and easy to analyze.

As we will see in the analyses, backpropagation equations through the original tensor to its factors are computationally expensive for all decomposition schemes, therefore it is preferable to avoid explicit computation of these equations in practice.

CP decomposition CP decomposition is a direct generalization of singular value decomposition (SVD) which decomposes a tensor into additions of rank-1 tensors (outer product of multiple vectors).

Specifically, given an m-order tensor T ∈ R I0×I1×···×Im−1 , CP decomposition factorizes it into m factor matrices {M (0) } m−1 l=0 , where DISPLAYFORM0 , where R is called the canonical rank of the CP decomposition, which is allowed to be larger than the I l 's. DISPLAYFORM1 where 1 ∈ R R is an all-ones vector of length R. With CP decomposition, T can be represented with only ( m−1 l=0 I l )R entries instead of ( m−1 l=0 I l ) as in the original tensor.

Now we proceed to derive the backpropagation rules for CP decomposition, i.e. the equations relating ∂L/∂T to {∂L/∂M (l) } m−1 l=0 .

In order to avoid deriving these equations from the entries level, we first isolate the factor of interest and rewrite the definition of CP decomposition as: DISPLAYFORM2 where we treat the first term as a constant tensor A (l) .

Once we reduce the compound operation to a basic one, we can simply refer to the rule derived in Appendix E, which gives us DISPLAYFORM3 The number of operations to compute one such equation both is O(( DISPLAYFORM4 is required for all m equations.

Therefore, evaluating these equations are computationally expensive (which takes O(mR) order as many operations as the size ( m−1 l=0 I l ) of the original tensor T ), and should be avoided whenever possible.

Tucker decomposition Tucker decomposition provides more general factorization than CP decomposition.

Given an m-order tensor T ∈ R I0×I1×···×Im−1 , Tucker decomposition factors it into m factor matrices DISPLAYFORM5 and an additional m-order core tensor C ∈ R R0×R1×···×Rm−1 , where the Tucker ranks R l 's are required to be smaller or equal than the dimensions at their corresponding modes, i.e. R l ≤ I l , ∀l ∈ [m].

DISPLAYFORM6 Notice that when R 0 = · · · = R m−1 = R and C is a super-diagonal tensor with all super-diagonal entries to be ones (a.k.a.

identity tensor), Tucker decomposition reduces to CP decomposition, and therefore CP decomposition is a special case of Tucker decomposition.

With Tucker decomposition, a tensor is approximately by ( DISPLAYFORM7 The backpropagation equations relating ∂L/∂T to ∂L/∂C and {∂L/M (l) } m−1 l=0 can be derived similarly as in CP decomposition.

First, we derive the equation for C at the entries level: DISPLAYFORM8 The equation above, written in tensor notations, reveals an expression in "reversed" Tucker form: DISPLAYFORM9 Although the number of operations to evaluate the equation depends on the particular order of tensor multiplications between ∂L/∂T and {M DISPLAYFORM10 where the first term is abbreviated as a tensor A (l) .

Subsequently, we apply the standard backpropagation rule of tensor multiplication in Appendix E and obtain the following equation: DISPLAYFORM11 where the second expression is equivalent to the first one, but requires fewer operations.

Though the exact number of operations depends on the order of tensor multiplications, it can be (again) bounded by O(( l=0 I l )), which is also highly inefficient and should be avoided in practice.

Tensor-train decomposition Tensor-train decomposition factorizes a m-order tensor into m interconnected low-order core tensors DISPLAYFORM12 where the R l 's are known as Tensor-train ranks, which controls the tradeoff between the number of parameters and accuracy of representation.

With Tensor-train decomposition, a tensor is represented by (R 0 I 0 + m−2 l=1 R l I l R l+1 + R m−1 I m−1 ) entries.

The backpropagation equations are derived following the paper Novikov et al. FORMULA70 , although we reformat them in tensor notations.

To begin with, we introduce two sets of auxiliary tensors {P (l) } m−1 l=0 and {Q (l) } m−1 l=0 as follows: DISPLAYFORM13 with corner cases as P l=0 can be computed using dynamic programming (DP) using the recursive definitions above.

With these auxiliary tensors, the definition of Tensor-train decomposition in Equation F.9b can be rewritten as: DISPLAYFORM14 Applying the backpropagation rule for tensor contraction twice, the backpropagation equations can be obtained in tensor notations as: Variants of standard decompositions In this paper, tensor decompositions are usually used in flexible ways, i.e. we will not stick to the standard formats defined in the previous paragraphs.

Indeed, we consider tensor decomposition as a reverse mapping of tensor operations: given a tensor T and a set of operations, the corresponding tensor decomposition aims to recover the input factors DISPLAYFORM15 DISPLAYFORM16 l=0 such that the operations on these factors return a tensor approximately equal to the given one.

In the following, we demonstrate some possibilities using examples:• The ordering of the modes can be arbitrary.

Therefore, CP decomposition of 3-order tensor T ∈ R I0×I1×I2 can be factorized as DISPLAYFORM17 i2,r .

It is easy to observe these decompositions are equivalent to each other if factor matrices are properly transposed.• A tensor may be partially factorized over a subset of modes.

For example, we can define a partial Tucker decomposition which factors only the last two modes of a 4-order tensor T ∈ R I0×I1×I2×I3 into a core tensor C ∈ R I0×I1×R2×R3 and two factor matrices DISPLAYFORM18 if written in our tensor notations.• Multiple modes can be grouped into supermode and decomposed like a single mode.

For example, given a 6-order tensor T ∈ R I0×I1×I2×J0×J1×J2 can be factorized into three factors DISPLAYFORM19 r1,i2,j2 , or more succinctly as DISPLAYFORM20 .

where I 0 and J 0 are grouped into a supermode (I 0 , J 0 ) and similarly for (I 1 , J 1 ) and (I 2 , J 2 ).

Notations Table 9 : Summary of tensor decompositions.

In this table, we summarize three types of tensor decompositions in tensor notations, and list their numbers of parameters and time complexities to backpropagate the gradient of a tensor T ∈ R I0×I1×···Im−1 to its m factors (and an additional core tensor C for Tucker decomposition).

For simplicity, we assume all dimensions I l 's of T are equal, and denote the size of T as the product of all dimensions I = m−1 l=0 I l .

Furthermore, we assume all ranks R l 's (in Tucker and Tensor-train decompositions) share the same number R. DISPLAYFORM0

In this section, we will show how tensor decomposition is able to compress (and accelerate) the standard convolutional layer in neural networks.

In order to achieve this, we first represent the operation of a standard convolutional layer in tensor notations.

By factorizing the tensor of parameters (a.k.a.

kernel) into multiple smaller factors, compression is achieved immediately.

As we discussed in Appendix F, learning the factors through the gradient of the original tensor of parameters is highly inefficient.

In this section, we provide an alternative strategy that interacts the input with the factors individually, in which explicit reference to the original kernel is avoided.

Therefore, our strategy also reduces the computational complexity along with compression.

For simplicity, we assume FFT is not used in computing convolutions, although we show in Appendix E that FFT can possibly speed up the backward pass Mathieu et al. (2013) .Standard convolutional layer In modern convolutional neural network (CNN), a standard convolutional layer is parameterized by a 4-order kernel K ∈ R H×W ×S×T , where H and W are height and width of the filters (which are typically equal), S and T are the number of input and output channels respectively.

A convolutional layer maps a 3-order tensor U ∈ R X×Y ×S to another 3-order tensor V ∈ R X ′ ×Y ′ ×T , where X and Y are the height and width for the input feature map, while X ′ and Y ′ are the ones for the output feature map, with the following equation: DISPLAYFORM0 where d is the stride of the convolution layer and the scopes of summations over i and j are determined by the boundary conditions.

Notice that the number of parameters in a standard convolutional layer is HW ST and the number of operations needed to evaluate the output V is O(HW ST XY ).With tensor notations in this paper, a standard convolutional layer can be defined abstractly as DISPLAYFORM1 which states that the standard convolutional layer in fact performs a compound operation of two tensor convolutions and one tensor contraction simultaneously between the input tensor U and the kernel of parameters K. Following the standard procedure in Appendix E, we obtain both backpropagation equations in tensor notations as follows: DISPLAYFORM2 It is not difficult to verify that the numbers of operations to compute these two backpropagation DISPLAYFORM3 In the next few paragraphs, we will apply various decompositions in Appendix F as well as singular value decomposition (SVD) on the kernel K, and derive the steps to evaluate Equation G.1 that interact the input with the factors individually.

Interestingly, these steps are themselves (non-standard) convolutional layers, therefore tensor decomposition on the parameters is equivalent to decoupling a layer in the original model into several sublayers in the compressed network, which can be implemented efficiently using modern deep learning libraries.

For simplicity, we assume in the analyses of these decomposition schemes that the output feature maps have approximately the same size as the input ones, i.e. DISPLAYFORM4 SVD-convolutional layer Many researchers propose to compress a convolutional layer using singular value decomposition, under the name of dictionary learning (Jaderberg et al., 2014; Denton et al., 2014; Zhang et al., 2015) .

These methods differ in their matricization of the tensor of parameters K, i.e. how to group the four modes into two and flatten the kernel K into a matrix.

By simple combinatorics, it is not difficult to show there are seven different types of matricization in total.

Here, we only pick to present the one by Jaderberg et al. FORMULA70 , which groups filter height and input channels as a supermode (H, S) and filter width and output channels (W, T ) as another.

DISPLAYFORM5 where K (0) ∈ R H×S×R and K (1) ∈ R W ×R×T are the two factor tensors.

It is easy to see an SVDconvolutional layer has (HS + W T )R parameters in total (HSR in K (0) and W T R in K (1) ).

Now we plug the Equation G.4a into G.1, and break the evaluation of V into two steps such that only one factor is involved at each step.

DISPLAYFORM6 where DISPLAYFORM7 DISPLAYFORM8 After decomposition, each operation is still a compound operation of tensor convolution and tensor contraction, and therefore itself a convolutional layer whose filters have size either H × 1 and 1 × W .

Effectively, SVD-convolutional layer is in fact a concatenation of two convolutional layers without nonlinearity in between.

Now we proceed to derive the corresponding backpropagation equations for these two steps following the procedure in Appendix E, which are presented in the following: DISPLAYFORM9 It is not hard to show the number of operations required to obtain the derivatives with respect to U and DISPLAYFORM10

CP-convolutional layer Both Lebedev et al. FORMULA70 ; Denton et al. (2014) propose to decompose the kernel K using CP decomposition, differing at whether the height H and width W of the filters are grouped into a supermode.

For simplicity, we follow the scheme in Denton et al. FORMULA70 : DISPLAYFORM0 where DISPLAYFORM1 ∈ R H×W ×R and K (2) ∈ R R×T are three factor tensors, which contain (HW + S + T )R parameters in total.

Again, plugging Equation G.8a into G.1 yields a three-steps procedure to evaluate V: DISPLAYFORM2 where DISPLAYFORM3 ′ ×R are two intermediate tensors.

Written in tensor notations, these equations are represented as: DISPLAYFORM4 (G.10c) After CP decomposition, the first and third steps are basic tensor multiplications on the input/intermediate tensor, which are usually named (weirdly) as 1 × 1 convolutional layers despite that no convolution is involved at all, while the second step is a compound operation of two tensor convolutions and one partial outer product, which is known as depth-wise convolutional layer (Chollet, 2016) .

The number of operations for these three steps are O(SRXY ), O(HW RXY ) and O(T RX ′ Y ′ ) respectively, resulting in a time complexity of O((SXY + HW XY + T X ′ Y ′ )R) for the forward pass, which is faster than the standard convolutional layer, since (HW + S + T )R ≤ HW ST implies (SXY + HW XY + T X ′ Y ′ )R ≤ HW ST XY .

Now we proceed to obtain their backpropagation equations following the procedure in Appendix E: DISPLAYFORM5 The number of operations in all three steps to calculate the derivatives with respect to input/intermediate tensors can be counted as DISPLAYFORM6 Tucker-convolutional layer The use of Tucker decomposition to compress and accelerate convolutional layers is proposed in Kim et al. (2015) .

Despite the name of Tucker decomposition, they in fact suggest a partial Tucker decomposition, which only factorizes the modes over the numbers of input/output filters and keeps the other two modes for filter height/width untouched.

DISPLAYFORM7 where DISPLAYFORM8 Rt×T are three factor tensors, with a total of (SR s + HW R s R t + R t T ) parameters.

All that follow are identical to the ones for SVD and CP layers.

A three-steps forward pass procedure is obtained by plugging Equation G.12a in G.1.

DISPLAYFORM9 where DISPLAYFORM10 DISPLAYFORM11 for the forward pass.

Like CP and SVD convolutional layers, Tucker-convolutional layer is faster than the standard convolutional layer, since SR s + HW R s R t + R t T ≤ HW ST implies SR s XY + HW R s R t XY + R t T X ′ Y ′ ≤ HW ST XY .

These equations, again, can be concisely written in tensor notations: DISPLAYFORM12 where the first and the third steps are two 1 × 1 convolutional layers, and the second step is itself a standard convolutional layer, which only differs from CP-convolutional layer at the second step.

For completeness, we summarize all backpropagation equations in the following: DISPLAYFORM13 Referring to the CP-convolutional layer, the time complexity for the backward pass is obtained with slight modification: the number of operations for the input/intermediate tensors is DISPLAYFORM14 , and the one for factors is DISPLAYFORM15 Tensor-train-convolutional layer Lastly, we propose to apply Tensor-train decomposition to compress a convolutional layer.

However, naive Tensor-train decomposition on the kernel K may give inferior results (Garipov et al., 2016) , and careful reordering of the modes is necessary.

In this paper, we propose to reorder the modes as (input channels S, filter height H, filter width W , output channels T ), and decompose the kernel as DISPLAYFORM16 where DISPLAYFORM17 Rt×T are factors, which require (SR s +HR s R+W RR t +R t T ).

Once we plug the decomposition scheme in Equation G.16a into G.1, the evaluation of V is decoupled into four steps , with number of operations as DISPLAYFORM18 x,j+dy,r = Rs−1 DISPLAYFORM19 where DISPLAYFORM20 ′ ×Rt are three intermediate tensors.

In tensor notations, these equations can be rewritten as as follows: DISPLAYFORM21 Tensor-train-convolutional layer is concatenation of four sub-layers, where the first and the last ones are 1 × 1 convolutional layers, while the other two in between are convolutional layers with rectangular kernels.

In fact, Tensor-train-convolutional layer can either be interpreted as (1) a Tucker-convolutional layer where the second sublayer is further compressed by a SVD, or (2) a SVD-convolutional layer where both factors are further decomposed again by SVD.

Referring to the previous results, the corresponding backpropagation equations are easily derived as DISPLAYFORM22 Similar to all previous layers, the time complexities for input/intermediate tensors and factors can be calculated as DISPLAYFORM23 DISPLAYFORM24 Table 10: Summary of plain tensor decomposition on convolutional layer.

We list the number of parameters and the number of operations required by forward/backward passes for various plain tensor decomposition on convolutional layer.

For reference, a standard convolutional layer maps a set of S feature maps with height X and width Y , to another set of T feature maps with height X ′ and width Y ′ .

All filters in the convolutional layer share the same height H and width W .

The operation of dense layer (a.k.a. fully connected layer) in neural network can be simply characterized by a matrix-vector multiplication, which maps a vector u ∈ R S to another vector v ∈ R T , where S and T are the number of units for the input and output respectively.

DISPLAYFORM0 It is easy to see that a dense layer is parameterized by a matrix K with ST parameters, and evaluating the output v requires O(ST ) operations.

With a matrix at hand, the simplest compression is via singular value decomposition (SVD), which decomposes K into multiplication of two matrices K = P Q, where P ∈ R S×R , Q ∈ R R×T with R ≤ min(S, T ).

With SVD decomposition, the number of parameters is reduced from ST to ((S +T )R) and time complexity from O(ST ) to O((S +T )R).Inspired by the intuition that invariant structures can be exploited by tensor decompositions in Section 3, we tensorize the matrix K into a tensor K ∈ R S0×···×Sm−1×T0×···×Tm−1 such that DISPLAYFORM1 l=0 T l and vec(K) = vec(K).

Correspondingly, we reshape the input/output u, v into U ∈ R S0×···×Sm−1 , V ∈ R T0×···×Tm−1 such that vec(U) = u, vec(V) = v and present an (uncompressed) tensorized dense layer as follows: DISPLAYFORM2 Therefore, a tensorized dense layer, parameterized by a 2m-order tensor K, maps an m-order tensor U to another m-order tensor V. It is straightforward to observe that the tensorized dense layer is mathematically equivalent is to the dense layer in Equation H.1a.

Correspondingly, its backpropagation equations can be obtained by simply reshaping the ones for standard dense layer: DISPLAYFORM3 In the section, we will compress the tensorized dense layer by decomposing the kernel K into multiple smaller factors.

As we will see, the schemes of tensor decompositions used in this section are not as straightforward as in Appendix F and G, and our principles in their designs are analyzed at the end of this section.

Again, learning the the factors through the gradient of the original tensor is extremely costly, therefore a multi-steps procedure to compute the output by interacting the input with the factors individually is desirable.

For simplicity of analyses, we will assume for the rest of the paper that S and T are factored evenly, that is DISPLAYFORM4 and all ranks are equal to a single number R.r-CP-dense layer Obviously, the simplest way to factorize K is to perform naive CP decomposition over all 2m modes without grouping any supermode.

However, such naive decomposition leads to significant loss of information, as we discuss at the end of this section.

In this paper, we instead propose to factor the kernel K by grouping (S l , T l )'s as supermodes.

Concretely, the tensor of parameters K is decomposed as: DISPLAYFORM5 where DISPLAYFORM6 are m factors and R is the canonical rank that controls the tradeoff between the number of parameters and the fidelity of representation.

Therefore, the total number of parameters of a r-CP-dense layer is approximately DISPLAYFORM7 m R, which is significantly smaller than ST given that R is reasonably small.

The next step to derive the sequential procedure mirrors the ones in all schemes in Appendix G, by plugging the Equation H.5a into H.2a, we arrive at a multi-steps procedure for the forward pass.

DISPLAYFORM8 DISPLAYFORM9 (H.7c) Following the procedure in Appendix E, their backpropagation equations are obtained as: DISPLAYFORM10 ∂L ∂U (0) As we discussed in Appendix E, if a compound operation does not contain convolution, the time complexity of backpropagation is identical to the forward pass.

Therefore, we claim the number of operations required for backward pass is also bounded by O(m max(S, T ) DISPLAYFORM11 r-Tucker-dense layer The application of TK decomposition is rather straightforward, which factors the tensor of parameters K exactly the same as in Appendix F. DISPLAYFORM12 where DISPLAYFORM13 DISPLAYFORM14 where the first and last steps are compound operations between a tensor and a set of multiple tensors, while the middle step is a multi-operations between two tensors.

Under the assumptions that DISPLAYFORM15 , the order to contract the factors in the first and last steps makes no difference, therefore we assume the order follows the indices without loss of generality.

With this strategy, the contraction with the l th factor takes O(( by the definition of Tucker decomposition: therefore, the time complexity to contract all m factors is at most O(mSR).

Likewise, the number of operations for the last step can also be bounded by O(mT R).

Lastly, it is easy to see the middle step needs O(R 2m ) operations, therefore leads to a total time complexity of O(m(S + T )R + R 2m ) for the three-step procedure.

In tensor notations, these equations can be concisely written as DISPLAYFORM16 DISPLAYFORM17 Though compound in nature, the procedure to derive their backpropagation rules are pretty straightforward: notice the equations for the first and last steps have the exactly the same form as standard Tucker decomposition in Appendix F. Therefore, we can simply modify the variable names therein to obtain the backpropagation equations for these two steps.

DISPLAYFORM18 The step in the middle is itself a tensorized layer defined in Equation H.2b, therefore its backpropagation rules can be obtained by renaming the variable in Equations H.4.

DISPLAYFORM19 Despite their technical complicity, we can resort to conclusion that the complexities for forward and backward passes are the same for operations without convolution, and claim that the number of operations required for the backpropagation equations above is bounded by O(m(S + T )R + R 2m ).r-Tensor-train-dense layer The layer presented in this part follows closely the pioneering work in compressing network using tensor decompositions (Novikov et al., 2015) , except that we replace the backpropagation algorithm in the original paper (as discussed in Appendix F) with a multistep procedure similar to all other layers in this paper.

With the replacement, the efficiency of the backward is greatly improved compared to original design.

Similar to r-CP-dense layer, we will group (S l , T l )'s as supermodes and decomposed the kernel K by Tensor-train decomposition following the order of their indices: DISPLAYFORM20 where the factor tensors are DISPLAYFORM21 DISPLAYFORM22 , and rename U as U (0) .

Now insert the Equation H.14a into H.2a and expand accordingly, we obtain an (m + 1)-steps procedure to evaluate the output V: DISPLAYFORM23 DISPLAYFORM24 .

These steps very simple in tensor notations.

DISPLAYFORM25 Observe that the operations in the forward pass are entirely tensor contractions, therefore their backpropagation equations are easily derived following the procedure in Appendix E. DISPLAYFORM26 In the analysis of the backward pass, we can again take advantage of the argument that the forward and backward passes share the same number of operations.

Therefore, we claim the time complexity for backpropagation is bounded by O(m max(S, T ) DISPLAYFORM27 Relation to tensor contraction layer: In Kossaifi et al. (2017a) , the authors propose a novel tensor contraction layer, which takes a tensor of arbitrary order as input and return a tensor of the same order.

Formally, a tensor contraction layer, parameterized by a set of m matrices {M (l) }) DISPLAYFORM28 , maps a m-order tensor U ∈ R S0×···×Sm−1 to another m-order tensor V ∈ R T0×···Tm−1 such that DISPLAYFORM29 It is not difficult to observe that the tensor contraction layer is in fact special case of r-CP-dense layer where the kernel is restricted to rank-1, that is K s0,··· ,sm−1,t0,··· ,tm DISPLAYFORM30 Relation to tensor regression layer: Along with the tensor contraction layer, tensor regression layer is also proposed in Kossaifi et al. (2017b) , which takes a tensor of arbitrary order as input and maps it to a scalar.

Formally, given an m-order tensor U ∈ R S0×···×Sm−1 , it is reduced to a scalar v by contracting of all the modes with another tensor of the same size K ∈ R S0×···×Sm−1 .

H.19b) where the tensor K is stored in Tucker-format as in Equation F.4a.

Therefore, the tensor regression layer is effectively parameterized by a set of matrices DISPLAYFORM31 DISPLAYFORM32 , with an additional core tensor C ∈ R R0×···×Rm−1 .

Therefore the definition of tensor regression layer in Equation H.19a can also be rephrased as DISPLAYFORM33 Now we are able to observe that the tensor regression layer is indeed a special case of r-Tuckerdense layer where the input factors DISPLAYFORM34 , while the output factors Q (l) 's are simply scalar 1's with DISPLAYFORM35 Comments on the designs: As we can observe, the design of r-Tucker-dense is different from the other two layers using CP and Tensor-train decompositions, in which (S l , T l )'s are first grouped into supermodes before factorization.

Indeed, it is a major drawback in the design of r-Tucker-dense layer: notice that the first intermediate result U , which becomes very tiny if the kernel K is aggressively compressed.

Therefore, the size of the intermediate tensor poses an "information bottleneck", causing significant loss during the forward pass, which is verified by our experimental results in Section app:experiments.

Therefore, the use of r-Tucker-dense layer is not recommended when we expect excessive compression rate.

On the other hand, by grouping (S l , T l )'s as supermodes in r-CP-dense layer and r-Tensor-traindense layer, all intermediate tensors U (l) 's have similar size as the input, therefore the bottleneck in r-Tucker-dense layer is completely avoided.

But why do we not group (S l , T l )'s together in the design of Tucker-dense layer at the beginning?

In theory, we are for sure able to factorize the kernel as DISPLAYFORM36 However, the contractions among the input and the factors become problematic: (1) interacting the input with the factors K (l) 's yields an intermediate tensor of size T 0 × · · · × T m−1 × R 0 × · · · × R m−1 , which is too large to fit into memory; (2) while reconstructing the kernel K from K (l) 's and subsequently invoking the Equation H.2a will make the time complexity for backward pass intractable as we discussed in Appendix F. Therefore, we have to abandon this attempting design, in order to maintain a reasonable time complexity.

As a compensation for the possible loss, the current design of Tucker-dense layer actually has one benefit over the other two layers: the numbers of operations for the backward pass remains at the same order as the number of parameters, while the number of operations required by r-CP-dense and r-Tensor-train-dense layers are orders higher.

As a result, r-Tucker-dense layer is much faster than r-CP-dense and r-Tensor-train-dense layers at the same compression rate.

Therefore, r-Tucker-dense layer is more desirable if we value speed over accuracy and the compression rate is not too high.

O(# of params.)

DISPLAYFORM0

In Appendix H, we tensorize the parameters into higher-order tensor in order to exploit their invariance structures.

It is tempting to extend the same idea to convolutional layer such that similar structures can be discovered.

In the section, we propose several additional convolutional layers based on the same technique as in Appendix H: the input and output tensors are folded as U ∈ R X×Y ×S0×···×Sm−1 and V ∈ R X ′ ×Y ′ ×T0×···×Tm−1 , while the tensor of parameters are reshaped into K ∈ R H×W ×S0×···×Sm−1×T0×···×Tm−1 .

Similar to the tensorized dense layer in Equation H.2a, we define a (uncompressed) tensorized convolutional layer equivalent to Equation G.1: DISPLAYFORM0 Their corresponding backpropagation equations are then easily obtained by reshaping the ones for standard convolutional layer in Equation G.3.

DISPLAYFORM1 What follows are almost replications of Appendix G and H: applying tensor decompositions in Appendix F to the kernel K and derive the corresponding multi-steps procedures.

As we shall expect, all layers in this section mimic their counterparts in Appendix H (in fact they will reduce to counterpart layers when original layer is 1 × 1 convolutional layer).

Therefore in this section, we will borrow results from last section whenever possible and only emphasize their differences.r-CP-convolutional layer In this part, CP decomposition is used in a similar way as in r-CPdense layer, which decomposes the kernel K grouping (S l , T l )'s as supermodes, and (H, W ) as an additional supermode.

Specifically, the tensor K takes the form as DISPLAYFORM2 where DISPLAYFORM3 Notice that Equation I.3a only differs from Equation H.5a in r-CP-dense layer by an additional factor K (m) , therefore has HW R more parameters and reaches O(m(ST ) 1 m R + HW R) in total.

Accordingly, the multisteps procedure to evaluate the output V now has one extra step at the end, and the (m + 2)-steps algorithm is presented at the entries level as follows: DISPLAYFORM4 r,i+dx,j+dy,s0,··· ,sm−1 = U i+dx,j+dy,s0,··· ,sm−1 (I.4a) where U (l) ∈ R R×S l ×···×Sm+1×T0×···×T l−1 , ∀l ∈ [m] are m intermediate tensors.

Notice that the order to interact the (m + 1) factors is arbitrary, that is the convolutional factor U (m) can be convoluted over at any step during the forward pass.

In this paper, we place the convolutional factor U (m) to the end simply for implementational convenience: it is not difficult to recognize that the last step is a 3D-convolutional layer with R input feature volumes and one output feature volume, if we treat the number of feature maps T as depth of the feature volumes.

The time complexity in the forward pass are easily obtained through the results of r-CP-dense layer: compared to r-CP-dense layer, each of the existing m steps will be scaled by a factor of XY , while the additional last step requires O(HW T RXY ) operations.

Therefore, the total number of operations for r-CP-convolutional layer is O(m max(S, T ) r-Tucker-convolutional layer Incorporating the features from both Tucker-convolutional layer in Appendix G and r-Tucker-dense layer in Appendix H, we propose to apply partial Tucker decomposition on the tensorized kernel K over all modes except the filter height H and width W .

Concretely, the tensorized kernel K is factorized as: Compared to r-Tensor-train-dense layer, the only difference is that the core tensor now has two extra modes for filter height and width, and therefore the number of parameters is magnified by a factor of HW .

Similar to Tucker-convolutional and r-Tucker-dense layers, the procedure to evaluate V can be sequentialized into three steps: In principle, the backpropagation rules for the sequential steps can be derived almost identically as in the r-Tucker-dense layer.

For reference, we list all equations for the first and last steps as follows: The analyses for the backpropagation equations mimic the ones in the forward pass, again by comparison against the ones in r-Tucker-dense layer: the time complexity to obtain the derivatives in the first step is magnified by XY , while the ones for the middle step and the last step are scaled by HW X ′ Y ′ and X ′ Y ′ respectively.

Therefore, the total number of operations for the derivatives with respect to input/intermediate results is O(mS r-Tensor-train-convolutional layer Following the r-CP-convolutional layer, we propose to apply Tensor-train decomposition to the tensorized kernel K by grouping (S l , T l )'s and filter height/width (H, W ) as supermodes.

In Tensor-train decomposition, these supermodes are ordered by their indices, with the extra supermode (H, W ) appended to the end.

Concretely, the tensorized kernel K is decomposed as: where K (0) ∈ R S0×T0×R0 , K (l) ∈ R R l−1 ×S l ×T l ×R l and K (m) ∈ R Rm−1×H×W are (m + 1) factor tensors.

Compared to r-Tensor-train-dense layer, the r-Tensor-train-convolutional layer has an additional factor K (m) that contains RHW parameters, which leads to a total number of O((m(ST ) 1 m + HW )R).

For conciseness, we follow the preprocessing steps to add singleton mode R −1 = 1 to U and K (0) such that U ∈ R X×Y ×S0×···×Sm−1×R−1 and K (0) ∈ R R−1×S0×T0×R0 and rename U as U (0) .

As we shall expect, the multi-steps procedure to evaluate V now has (m + 1) steps, with the last step as a 3D-convolutional layer: TAB2 , we denote the numbers for the dense layers as: DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 DISPLAYFORM10

@highlight

Compression of neural networks which improves the state-of-the-art low rank approximation techniques and is complementary to most of other compression techniques. 