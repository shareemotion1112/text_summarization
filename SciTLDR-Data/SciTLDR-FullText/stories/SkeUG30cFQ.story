Recent results from linear algebra stating that any matrix can be decomposed into products of diagonal and circulant matrices has lead to the design of compact deep neural network architectures that perform well in practice.

In this paper, we bridge the gap between these good empirical results  and the theoretical approximation capabilities of Deep diagonal-circulant ReLU networks.

More precisely, we first demonstrate  that a Deep diagonal-circulant ReLU networks of bounded width and small depth can approximate a deep ReLU network in which the dense matrices are of low rank.

Based on this result, we provide new bounds on the expressive power and universal approximativeness of this type of networks.

We support our experimental results with thorough experiments on a large, real world video classification problem.

Recent progress in deep neural networks came at the cost of an important increase of model sizes.

Nowadays, state-of-the-art architectures for common tasks such as object recognition typically have tens of millions of parameters BID13 ) and up to a billion parameters in some cases BID10 .

Best performing (ensemble) models typically combine dozens of such models, and their size can quickly add up to ten or twenty gigabytes.

Large models are often more accurate, but training them requires time and large amounts of computational resources.

Even when they are trained, they remain difficult to deploy, especially on mobile devices where memory or computational power is limited.

In linear algebra, it is common to exploit structural properties of matrices to speedup computations, or reduce memory usage.

BID7 have applied this principle in the context of deep neural networks, and proposed a network architecture in which large unstructured weight matrices have been replaced with more compact matrices with a circulant structure.

Since any n-by-n circulant matrix can be represented in memory using only a vector of dimension n, the change resulted in a drastic reduction of the model size (from 230MB to 21MB).

Furthermore, Cheng et al. have shown empirically that their network architecture can be almost as accurate as the original network.

BID23 have proposed a more principled approach leveraging a result by BID16 stating that any matrix A ??? C n??n can be decomposed into 2n ??? 1 diagonal and circulant matrices.

They use this result to design Deep diagonal-circulant ReLU networks.

However their experiments show good results even with a small number of factors (down to 2 factors), suggesting that Deep diagonal-circulant ReLU networks can achieve good approximation error, even with few factors.

In this paper, we bridge the gap between the good empirical results observed by BID23 , and the theoretical approximation capabilities of Deep diagonal-circulant ReLU networks.

We prove that Deep diagonal-circulant ReLU networks with bounded width and small depth can approximate any dense neural network.

We obtain this result by showing that any matrix A can be decomposed into 4k + 1 diagonal and circulant matrices where k is the rank of the matrix A. In practice, this result is more useful than the one by Huhtanen & Per??m??ki since one can rely on a low rank SVD decomposition of A while controlling the approximation error.

In addition to this theoretical contribution, we also conduct thorough experiments on synthetic and real datasets.

In accordance with the theory, our experiments demonstrate that we can easily tradeoff accuracy for model size by adjusting the number of factors in the matrix decomposition.

Finally we evaluate the applicability of this approach on state-of-the-art neural network architectures trained for video classification on the Youtube-8m Video dataset (over 1TB of training data).

This experiment demonstrates that Deep diagonal-circulant ReLU networks can be used to train more compact neural networks on large scale, real world scenarios.

A variety of techniques have been proposed to build more compact deep learning models.

A first category of techniques aims at compressing a trained network into a smaller model, without compromising the accuracy.

For example model distillation BID15 is two step training procedure: first a large model is trained to be as accurate as possible, second a more compact model is trained to approximate the first one.

Other approaches have focused on compressing the network by reducing the memory, at the level of individual weights, (for example, using weight quantization BID11 or parameter pruning) or at the level of weight matrices, using low rank decomposition of the original weight matrix BID28 or using sparse representations BID8 BID9 BID19 .Instead of compressing the network a posteriori, several researchers have focused on designing models that are compact by design.

This approach has several benefits, but most importantly, it reduces memory footprint, both required during training and inference.

have proposed to compress weight matrices by using hashing functions to map several matrix coefficients into the same memory cell.

The techniques works well in theory, but suffers from poor performance on modern GPU devices due to irregular access patterns.

In their paper, BID7 observed that fully connected layers (which typically occupy 90%1 .

of the total number of weights) are often used to perform simple dimensionality reduction operation between layers of different dimension.

The idea of replacing large weight matrices from fully connected layers with more compact circulant layers comes from a result by BID14 that have demonstrated that circulant matrices can be use to approximate the Johson-Lindenstrauss transform, often used to perform dimensionality reduction.

Building on this result Cheng et al. proposed to replace the weight matrix of a fully connected layer by a circulant matrix initialized with random weights.

The resulting models achieve good accuracy, with the random circulant matrix, but even better when the weights of the circulant matrix are trained with the rest of the network using a gradient based optimization algorithm.

This suggests that such layers, often perform more than simple random projections, and that more expressive fully connected layers are beneficial to the overall accuracy of the model.

Fortunately, more general linear transforms can also be described using circulant matrices or other structured matrices, at the cost of using more of them.

BID24 and Schmid et al. (2000) have demonstrated this formally by showing that any matrix can be decomposed into the product of diagonal and circulant matrices, and BID23 have proposed a compact neural network architecture based on this decomposition that exhibit good accuracy in practice.

Other researchers have investigated using alternative structures such as Toeplitz (Sindhwani et al., 2015) , Vandermonde (Sindhwani et al., 2015) or Fastfood transforms (Yang et al., 2015) .

Despite demonstrating good empirical results, there have been little theoretical insight to explain the good approximation capabilities of deep neural networks based on structured matrices.

BID5 presented the universal approximation theorem which states that any neural network with at least 1 hidden layer and sigmoid non linearity can approximate any function.

However, the theorem by BID5 does not bound the width of the neural network and does not consider the training procedure.

Since then, substantial theoretical work has been done to evaluate the expressiveness of a neural network as a function of the width (i.e. the number of neurons) and the depth of the network Arora et al. (2018)

A n-by-n circulant matrix C is a special kind of Toeplitz matrix where each row is a cyclic right shift of the previous one as illustrated below.

DISPLAYFORM0 Despite their rigorous structure, circulant matrices are expressive enough to model a variety of linear transforms such as random projections BID14 ) and when they are combined together with diagonal matrices, they can be used to represent an arbitrary transform (Schmid et al., 2000) .Circulant matrices also exhibit several properties that are interesting from a computational perspective.

First, a circulant n-by-n matrix C can be represented using only n coefficients.

Thus, it is far more compact that a full matrix that requires n 2 coefficient.

Second, the product between a circulant matrix C and a vector x can be simplified to a simple element-wise product between the vector c and x in the Fourier domain (which is generally performed efficiently on GPU devices).

This results in a complexity reduced from O(n 2 ) to O(nlog(n)).In their paper, BID16 have demonstrated that any matrix A ??? C n??n can be approximated with an arbitrary precision by a product of circulant and diagonal matrices: Theorem 1.

BID16 For any given matrix A ??? C n??n , let p be the smallest integer DISPLAYFORM1 Then for any > 0, for any matrix norm ?? , there exists a sequence of matrices B 1 . . .

B 2n???1 where B i is a circulant matrix if i is odd, and a diagonal matrix otherwise, such that B 1 B 2 . . .

B 2n???1 ??? A < , and where S = circ(0, 1, 0, . . . , 0)Because of their interesting properties, several researchers have considered circulant matrices as a replacement from full weight matrices inside neural networks.

There has already been some recent theoretical work on Deep diagonal-circulant ReLU networks, in which 2-layer networks of unbounded width where shown to be universal approximators.

These results are of limited interest, because the networks used in practice are of bounded width.

Unfortunately, nothing is known about the theoretical properties of Deep diagonal-circulant ReLU networks in this case.

In particular, the following questions remained unanswered up to now: Are Deep diagonal-circulant ReLU networks with bounded width universal approximators?

What kind of functions can Deep diagonal-circulant ReLU networks with bounded-width and small depth approximate?In this section, we first define formally diagonal-circulant ReLU networks, and then provide a theoretical analysis of their approximation capabilities.

DISPLAYFORM0 , where A 1 . . .

A l are arbitrary n ?? n matrices and b 1 . . .

b l ??? C n and where l and n are the depth and the width of the network respectively.

As in BID23 , Deep diagonal-circulant ReLU networks can be defined as follows: DISPLAYFORM1 are circulant matrices, and where l and n are the depth and the width of the network respectively.

To show that bounded-width Deep diagonal-circulant ReLU networks are universal approximators, we first need a proposition relating standard deep neural networks to Deep diagonal-circulant ReLU networks.

Proposition 2.

Let N : R n ??? R n be a deep ReLU networks of width n and depth l, and let X ??? R n be a compact set.

For any > 0, there exists a deep diagonal-circulant ReLU network N of width n and of DISPLAYFORM2 We can now state the universal approximation corrolary: Corrolary 1.

Bounded depth Deep diagonal-circulant ReLU networks are universal approximators on any compact set X .Proof.

Proposition 2 shows that bounded-width Deep diagonal-circulant ReLU networks can approximate any Deep ReLU network.

It has been shown recently in BID12 that bounded-width deep ReLU networks are universal approximators.

Together, these two results concludes the proof.

It is important to remark that Deep diagonal-circulant ReLU networks are not necessarily more compact than Deep ReLU networks.

Indeed, consider a n-wide Deep ReLU network with l layers having ln 2 weights.

The previous corollary tells us that this network can be decomposed in a Deep ReLU networks involving l(2n???1) matrices, i.e. 2ln(2n ??? 1) weights.

Despite the lack of theoretical guarantees a number of work provided empirical evidence that bounded width and small depth Deep diagonal-circulant ReLU networks result in good performance (e.g. BID23 ; BID3 ; BID7 ).

The following theorem studies the approximation properties of these small depth networks.

DISPLAYFORM3 ??? f A1,b1 be a deep ReLU network of width n and depth l, such that each matrix A i is of rank k i , where k i divides n. Let X ??? R n be a compact set.

For any > 0, there exists a deep diagonal-circulant ReLU network N of width n and of depth ( DISPLAYFORM4 This result generalizes Proposition 2, showing that a Deep diagonal-circulant ReLU networks of bounded width and small depth can approximate a deep ReLU network in which the dense matrices are of low rank.

Note in the proposition, we require that k i divides n. We conjecture that the proposition holds even without this condition, but we were not able to prove it.

Finally, what if we choose to use small depth network to approximate deep ReLU networks where matrices are not of low rank ?

To answer this question, we first need to show the negative impact of replacing matrices by their low rank approximators in neural networks: DISPLAYFORM5 .

Let?? i be the matrix obtained by a SVD approximation of rank k of matrix A i .

Let ?? i,j is the j th singular value of DISPLAYFORM6 where R is an upper bound on norm of the output of any layer in N , and ?? max,j = max i ?? i,j .Basically, this proposition shows that we can approximate matrices in a neural network by low rank matrices, and control the approximation error.

In general, the term ?? l max,1 could seem large, but in practice, it is likely that most singular values in deep neural network are small in order to avoid divergent behaviors.

We can now prove the result on Deep diagonal-circulant ReLU networks: DISPLAYFORM7 of depth l and width n. Let ?? max,j = max i ?? i,j where ?? i,j is the j th singular value of A i .

Let X ??? R n be a compact set.

For any k dividing n, there exists a deep diagonal-circulant ReLU network N = f DmCm,b l ??? . . .

??? f D1C1,b 1 of width n and of depth m = 4(k + 1)n, such that for any DISPLAYFORM8 , where R is an upper bound on the norm of the outputs of each layer in N .

DISPLAYFORM9 , where each?? i is the matrix obtained by a SVD approximation of rank k of matrix A i .With proposition 4, we have an error bound on N (x) ???N (x) .

Now each matrix?? i can be replaced by a product of k diagonal-circulant matrices.

By lemma 1, this product yields a Deep diagonal-circulant ReLU networks of depth m = 4(k + 1)n, strictly equivalent toN on X .

The result follows.

The experiments that we present in this section aim at answering the following questions.

First question: what is the impact of increasing the number of diagonal-circulant factors on the accuracy of the network?

To answer this question, we conduct a series of experiments on a synthetic classification dataset with an increasing number of factors.

As we will show, the results match our theoretical analysis from Section 3.

Second question: can this approach be useful to build more compact models in the context of large scale realworld machine learning applications.

To answer this second question, we build a deep diagonal-circulant neural network architecture for video classification.

The architecture is based on state-of-the-art architecture initially proposed by Abu-El-Haija et al. (2016b) and later improved by BID22 in involve several large layers that can be made more compact using circulant matrices as done in BID3 .

As we will show, the approach demonstrate good accuracy and can be used to build a more compact network than the original one.

Experimental setup The dataset is generated using the make classification 2 function from ScikitLearn BID25 .

It is made of 10000 examples, 5 variables, 2 classes and 2 clusters for each class.

We train a neural network with 3 hidden layers of 1024 neurons each.

We used a batch size of 50, a learning rate of 5 ?? 10 ???2 , a learning rate decay of 0.9 every 10 000 examples.

We compare the dense neural network with a Deep diagonal-circulant ReLU networks with several factors.

We use the initialization proposed in BID23 .

Results Table 4 .1 shows the loss of the dense architecture versus the Deep diagonal-circulant ReLU networks with different factors.

The table also shows the compression rate obtain with the Deep diagonalcirculant ReLU networks.

We notice that the Deep diagonal-circulant ReLU networks manage to achieve more than 90% compression rate with a substantial loss in accuracy with factor up to 16.

Adding factors improve the accuracy but make the convergence difficult.

We were note able to train a model with more than 32 layers.

A solution would be to use the circulant-diagonal ReLU decomposition only on certain layer in order to trade-off compression with accuracy more precisely.

In this section, we demonstrate the applicability of diagonal-circulant ReLU networks in the context of a large scale video classification architecture trained on the Youtube-8M dataset.

Our architecture is based on a state-of-the-art architecture that was initially proposed by BID2 and later improved by BID22 .

Dataset The dataset is composed of an embedding (each video and audio frames are represented by a vector of respectively 1024 and 128) of video and audio frames extracted every 1 seconds with up to 300 frames per video.

Model Architecture This architecture can be decomposed into three blocks of layers, as illustrated in Experiment We want to compare the effect on the circulant-diagonal ReLU decomposition only on certain layer to evaluate the trade-off between compression rate and accuracy.

First, we train the architecture presented in Figure 4 .1 without any circulant matrices to serve as a baseline.

Then, we used the circulantdiagonal decomposition on each layer independently.

Hyper-parameters All our experiments are developed with TensorFlow Framework BID0 .

We trained our models with the CrossEntropy loss and used Adam optimizer with a 0.0002 learning rate and a 0.8 exponential decay every 4 million examples.

We used a fully connected layer of size 8192 for the video DBoF and 4096 for the audio.

The fully connected layers used for dimensionality reduction have a size of 512 neurons.

We used 4 mixtures for the MoE Layer.

Evaluation Metric We used the GAP (Global Average Precision), as used in Abu- BID2 , to compare our experiments.

This series of experiments aims at understanding the effect of circulant-diagonal ReLU decomposition over different layers with 1 factors.

Table 2 shows the result in terms of number of weights, size of the model (MB) and GAP.

We also compute the compression ratio with respect to the dense model.

The compact fully connected layer achieves a compression rate of 9.5 while having a very similar performance, whereas the compact DBoF and MoE achieve a higher compression rate at the expense of accuracy.

Figure 4 .2 shows that the model with a compact FC converges faster than the dense model.

The model with a compact DBoF shows a big variance over the validation GAP which can be associated with a difficulty to train.

The model with a compact MoE is more stable but at the expense of its performance.

In this paper we provided a theoretical study of the properties of Deep diagonal-circulant ReLU networks and demonstrated that they are bounded width universal approximators.

The bound on this decomposition allowed us to calculate the error bound on any Deep diagonal-circulant ReLU networks given the depth on the network and the singular values associated with the weight matrices.

Our empirical study demonstrate that we can trade-off model size for accuracy in accordance with the theory, and that we can use Deep diagonal-circulant ReLU networks in large scale machine learning applications.

Lemma 1.

Let A l , . . .

A 1 ??? C n??n , b ??? C n and let X ??? R n be a compact set.

There exists ?? l . . .

DISPLAYFORM0 where 1 n is the n-vector of ones.

Clearly, for all x ??? X we have h 1 (x) ??? 0, so ReLU ??? h 1 (x) = h 1 (x).More generally, for all j < n ??? 1 define ?? j+1 = 1 n ??? ??? A j+1 ?? j .

It is easy to see that for all j < n we have h j ??? . . .

DISPLAYFORM1 .

By theorem 1, for any > 0, any matrix A i , there exists a sequence of 2n???1 matrices DISPLAYFORM2 is the identity matrix.

By lemma 1, we know that there exists DISPLAYFORM3 Now if tends to zero, f DinCin,??in ??? . . .

??? f Di1Ci1,??i1 ??? ReLU (A i x + b i ) will also tend to zero for any x ??? X , because the ReLU function is continuous and X is compact.

Let N = f D1nC1n,??1n ???. .

.???f Di1Ci1,??i1 .

Again, because all functions are continuous, for all x ??? X , N (x) ??? N (x) tends to zero as tends to zero.

Proposition 5.

Let A ??? C n??n a matrix of rank k. Assume that n can be divided by k. For any > 0, there exists a sequence of 4k + 1 matrices B 1 , . . .

, B 4k+1 , where B i is a circulant matrix if i is odd, and a diagonal matrix otherwise, such that DISPLAYFORM4 T be the SVD decomposition of M where U, V and ?? are n ?? n matrices.

Because M is of rank k, the last n???k columns of U and V are null.

In the following, we will first decompose U into a product of matrices W RO, where R and O are respectively circulant and diagonal matrices, and W is a matrix which will be further decomposed into a product of diagonal and circulant matrices.

Then, we will apply the same decomposition technique to V .

Ultimately, we will get a product of 4k + 2 matrices alternatively diagonal and circulant.

Let R = circ(r 1 . . .

r n ).

Let O be a n ?? n diagonal matrix where O i,i = 1 if i ??? k and 0 otherwise.

The k first columns of the product RO will be equal to that of R, and the n ??? k last colomns of RO will be zeros.

For example, if k = 2, we have: DISPLAYFORM5 .

For now, the values of d ij are unknown, but we will show how to compute them.

DISPLAYFORM6 Note that the n ??? k last columns of the product W RO will be zeros.

For example, with k = 2, we have: DISPLAYFORM7 We want to find the values of d ij such that W RO = U .

We can formulate this as linear equation system.

In case k = 2, we get: DISPLAYFORM8 r n r 1 r n???1 r n r 1 r 2 r n r 1 r 2 r 3 r 1 r 2 . . .

DISPLAYFORM9 . . .

DISPLAYFORM10 The i th bloc of the bloc-diagonal matrix is a Toeplitz matrix induced by a subsequence of length k of (r 1 , . . .

r n , r 1 . . .

r n ).

Set r j = 1 for all j ??? {k, 2k, 3k, . . .

n} and set r j = 0 for all other values of j. Then it is easy to see that each bloc is a permutation of the identity matrix.

Thus, all blocs are invertible.

This entails that the block diagonal matrix above is also invertible.

So by solving this set of linear equations, we find d 1,1 . . .

d k,n such that W RO = U .

We can apply the same idea to factorize V = W .R.O for some matrix W .

Finally, we get DISPLAYFORM11 Thanks to Theorem 1, W and W can both be factorized in a product of 2k ??? 1 circulant and diagonal matrices.

Note that O??O T is diagonal, because all three are diagonal.

Overall, A can be represented with a product of 4k + 2 matrices, alternatively diagonal and circulant.

Proof.

of proposition 3 By proposition 5, each low rank matrix of the neural net can be decomposed in a small number of diagonal and circulant matrices.

By lemma 1, the matrices can be connected to form a neural net.

Observe that for any sequence a 0 , a 1 . . .

defined reccurently by a 0 = 0 and a i = ra i???1 + s, the reccurence relation can be unfold as follows: a i = s(r i ???1) r???1 .

We can apply this formula to bound our error as follows x l ???x l ??? (?? l max,1 ???1)?? max,k maxi xi ??max,1???1 .Lemma 2.

Let A ??? C n??n with singular values ?? 1 . . .

?? n , and let x,x ??? C n .

Let?? be the matrix obtained by a SVD approximation of rank k of matrix A. Then we have: Finally, it is easy to see that for any pair of vectors a, b ??? C n , we have ReLU (a) ??? ReLU (b) ??? a ??? b .

This concludes the proof.

DISPLAYFORM12

@highlight

We provide a theoretical study of the properties of Deep circulant-diagonal ReLU Networks and demonstrate that they are bounded width universal approximators.

@highlight

The paper proposes using circulant and diagonal matrices to speed up computation and reduce memory requirements in eural networks.

@highlight

This paper proves that bounded width diagonal-circulent ReLU networks (DC-ReLU) are universal approximators.