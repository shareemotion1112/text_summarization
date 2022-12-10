In this paper, we study deep diagonal circulant neural networks, that is deep neural networks in which weight matrices are the product of diagonal and circulant ones.

Besides making a theoretical analysis of their expressivity, we introduced principled techniques for training these models: we devise an initialization scheme and proposed a smart use of non-linearity functions in order to train deep diagonal circulant networks.

Furthermore, we show that these networks outperform recently introduced deep networks with other types of structured layers.

We conduct a thorough experimental study to compare the performance of deep diagonal circulant networks with state of the art models based on structured matrices and with dense models.

We show that our models achieve better accuracy than other structured approaches while required 2x fewer weights as the next best approach.

Finally we train deep diagonal circulant networks to build a compact and accurate models on a real world video classification dataset with over 3.8 million training examples.

The deep learning revolution has yielded models of increasingly large size.

In recent years, designing compact and accurate neural networks with a small number of trainable parameters has been an active research topic, motivated by practical applications in embedded systems (to reduce memory footprint (Sainath & Parada, 2015) ), federated and distributed learning (to reduce communication (Konečný et al., 2016) ), derivative-free optimization in reinforcement learning (to simplify the computation of the approximated gradient (Choromanski et al., 2018) ).

Besides a number of practical applications, it is also an important research question whether or not models really need to be this big or if smaller results can achieve similar accuracy (Ba & Caruana, 2014) .

Structured matrices are at the very core of most of the work on compact networks.

In these models, dense weight matrices are replaced by matrices with a prescribed structure (e.g. low rank matrices, Toeplitz matrices, circulant matrices, LDR, etc.).

Despite substantial efforts (e.g. Cheng et al. (2015) ; ), the performance of compact models is still far from achieving an acceptable accuracy motivating their use in real-world scenarios.

This raises several questions about the effectiveness of such models and about our ability to train them.

In particular two main questions call for investigation: Q1 How to efficiently train deep neural networks with a large number of structured layers?

Q2 What is the expressive power of structured layers compared to dense layers?

In this paper, we provide principled answers to these questions for the particular case of deep neural networks based on diagonal and circulant matrices (a.k.a.

Diagonal-circulant networks or DCNNs).

The idea of using diagonal and circulant matrices together comes from a series of results in linear algebra by Müller-Quade et al. (1998) and .

The most recent result from Huhtanen & Perämäki demonstrates that any matrix A in C n⇥n can be decomposed into the product of 2n 1 alternating diagonal and circulant matrices.

The diagonal-circulant decomposition inspired to design the AFDF structured layer, which is the building block of DCNNs.

However, were not able to train deep neural networks based on AFDF.

To answer Q1, we first describe a theoretically sound initialization procedure for DCNN which allows the signal to propagate through the network without vanishing or exploding.

Furthermore, we provide a number of empirical insights to explain the behaviour of DCNNs, and show the impact of the number of the non-linearities in the network on the convergence rate and the accuracy of the network.

By combining all these insights, we are able (for the first time) to train large and deep DCNNs.

We demonstrate the good performance of DCNNs on a large scale application (the YouTube-8M video classification problem) and obtain very competitive accuracy.

To answer Q2, we propose an analysis of the expressivity of DCNNs by extending the results by .

We introduce a new bound on the number of diagonal-circulant required to approximate a matrix that depends on its rank.

Building on this result, we demonstrate that a DCNN with bounded width and small depth can approximate any dense networks with ReLU activations.

Outline of the paper: We present in Section 2 the related work on structured neural networks and several compression techniques.

Section 3 introduces circulant matrices, our new result extending the one from .

Section 4 proposes an theoretical analysis on the expressivity on DCNNs.

Section 5 describes two efficient techniques for training deep diagonal circulant neural networks.

Finally, Section 6 presents extensive experiments to compare the performance of deep diagonal circulant neural networks in different settings w.r.t.

other state of the art approaches.

Section 7 provides a discussion and concluding remarks.

Structured matrices exhibit a number of good properties which have been exploited by deep learning practitioners, mainly to compress large neural networks architectures into smaller ones.

For example Hinrichs & Vybíral (2011) have demonstrated that a single circulant matrix can be used to approximate the Johson-Lindenstrauss transform, often used in machine learning to perform dimensionality reduction.

Building upon this result, Cheng et al. (2015) proposed to replace the weight matrix of a fully connected layer by a circulant matrix effectively replacing the complex transform modeled by the fully connected layer by a simple dimensionality reduction.

Despite the reduction of expressivity, the resulting network demonstrated good accuracy using only a fraction of its original size (90% reduction).

Comparison with ACDC.

have introduced two Structured Efficient Linear Layers (SELL) called AFDF and ACDC.

The AFDF structured layer benefits from the theoretical results introduced by Huhtanen & Perämäki and can be seen the building block of DCNNs.

However, only experiment using ACDC, a different type of layer that does not involve circulant matrices.

As far as we can tell, the theoretical guarantees available for the AFDF layer do not apply on the ACDC layer since the cosine transform does not diagonalize circulant matrices (Sanchez et al., 1995) .

Another possible limit of the ACDC paper is that they only train large neural networks involving ACDC layers combined with many other expressive layers.

Although the resulting network demonstrates good accuracy, it is difficult the characterize the true contribution of the ACDC layers in this setting.

Comparison with Low displacement rank structures.

More recently, Thomas et al. (2018) have generalized these works by proposing neural networks with low-displacement rank matrices (LDR), that are structured matrices encompassing a large family of structured matrices, including Toeplitzlike, Vandermonde-like, Cauchy-like and more notably DCNNs.

To obtain this result, LDR represents a structured matrix using two displacement operators and a low-rank residual.

Despite being elegant and general, we found that the LDR framework suffers from several limits which are inherent to its generality, and makes it difficult to use in the context of large and deep neural networks.

First, the training procedure for learning LDR matrices is highly involved and implies many complex mathematical objects such as Krylov matrices.

Then, as acknowledged by the authors, the number of parameters required to represent a given structured matrix (e.g. a Toeplitz matrix) in practice is unnecessarily high (higher than required in theory).

Other compression techniques.

Besides structured matrices, a variety of techniques have been proposed to build more compact deep learning models.

These include model distillation (Hinton et al., 2015) , Tensor Train (Novikov et al., 2015) , Low-rank decomposition (Denil et al., 2013) , to mention a few.

However, Circulant networks show good performances in several contexts (the interested reader can refer to the results reported by and Thomas et al. (2018) ).

An n-by-n circulant matrix C is a matrix where each row is a cyclic right shift of the previous one as illustrated below.

Circulant matrices exhibit several interesting properties from the perspective of numerical computations.

Most importantly, any n-by-n circulant matrix C can be represented using only n coefficients instead of the n 2 coefficients required to represent classical unstructured matrices.

In addition, the matrix-vector product is simplified from O(n 2 ) to O(n log(n)) using the convolution theorem.

As we will show in this paper, circulant matrices also have a strong expressive power.

So far, we know that a single circulant matrix can be used to represent a variety of important linear transforms such as random projections (Hinrichs & Vybíral, 2011) .

When they are combined with diagonal matrices, they can also be used as building blocks to represent any linear transform (Schmid et al., 2000; with an arbitrary precision.

Huhtanen & Perämäki were able to bound the number of factors that is required to approximate any matrix A with arbitrary precision.

Relation between diagonal circulant matrices and low rank matrices We recall this result in Theorem 1 as it is the starting point of our theoretical analysis (note that in the rest of the paper, k·k denotes the`2 norm when applied to vectors, and the operator norm when applied to matrices).

Unfortunately, this theorem is of little use to understand the expressive power of diagonal-circulant matrices when they are used in deep neural networks.

This is because: 1) the bound only depends on the dimension of the matrix A, not on the matrix itself, 2) the theorem does not provide any insights regarding the expressive power of m diagonal-circulant factors when m is much lower than 2n 1 as it is the case in most practical scenarios we consider in this paper.

In the following theorem, we enhance the result by Huhtanen & Perämäki by expressing the number of factors required to approximate A, as a function of the rank of A. This is useful when one deals with low-rank matrices, which is common in machine learning problems.

Theorem 2. (Rank-based circulant decomposition) Let A 2 C n⇥n be a matrix of rank at most k. Assume that n can be divided by k. For any ✏ > 0, there exists a sequence of 4k + 1 matrices B 1 , . . .

, B 4k+1 , where B i is a circulant matrix if i is odd, and a diagonal matrix otherwise, such that kB 1 B 2 . . .

B 4k+1 Ak < ✏ A direct consequence of Theorem 2, is that if the number of diagonal-circulant factors is set to a value K, we can represent all linear transform A whose rank is , this result shows that structured matrices with fewer than 2n diagonal-circulant matrices (as it is the case in practice) can still represent a large class of matrices.

As we will show in the following section, this result will be useful to analyze the expressivity of neural networks based on diagonal and circulant matrices.

Zhao et al. (2017) have shown that circulant networks with 2 layers and unbounded width are universal approximators.

However, results on unbounded networks offer weak guarantees and two important questions have remained open until now: 1) Can we approximate any function with a bounded-width circulant networks?

2) What function can we approximate with a circulant network that has a bounded width and a small depth?

We answer these two questions in this section.

First, we introduce some necessary definitions regarding neural networks and we provide a theoretical analysis of their approximation capabilities.

In the rest of this paper, we call L and n respectively the depth and the width of the network.

Moreover, we call total rank k, the sum of the ranks of the matrices

We also need to introduce DCNNs, similarly to .

and where i (.) is a ReLU non-linearity or the identity function.

We can now show that bounded-width DCNNs can approximate any Deep ReLU Network, and as a corollary, that they are universal approximators.

Lemma 1.

Let N be a deep ReLU network of width n and depth L, and let X ⇢ C n be a bounded set.

For any ✏ > 0, there exists a DCNN N 0 of width n and of depth

The proof is in the supplemental material.

We can now state the universal approximation corollary:

Corollary 1.

Bounded width DCNNs are universal approximators in the following sense: for any continuous function f :

This is a first result, however (2n + 5)L is not a small depth (in our experiments, n can be over 300 000), and a number of work provided empirical evidences that DCNN with small depth can offer good performances (e.g. Araujo et al. (2018); Cheng et al. (2015) ).

To improve our result, we introduce our main theorem which studies the approximation properties of these small depth networks.

Theorem 3. (Rank-based expressive power of DCNNs) Let N be a deep ReLU network of width n, depth L and a total rank k and assume n is a power of 2.

Let X ⇢ C n be a bounded set.

Then, for any ✏ > 0, there exists a DCNN with ReLU activation N 0 of width n such that kN (x) N 0 (x)k < ✏ for all x 2 X and the depth of N 0 is bounded by 9k.

Remark that in the theorem, we require that n is a power of 2.

We conjecture that the result still holds even without this condition.

This result refines Lemma 1, and answer our second question: a DCNN of bounded width and small depth can approximate a Deep ReLU network of low total rank.

Note that the converse is not true: because n-by-n circulant matrix can be of rank n, approximating a DCNN of depth 1 can require a deep ReLU network of total rank equals to n.

Expressivity of DCNNs For the sake of clarity, we highlight the significance of these results with the two following properties.

Properties.

Given an arbitrary fixed integer n, let R k be the set of all functions f : R n !

R n representable by a deep ReLU network of total rank at most k and let C l the set of all functions f : R n !

R n representable by deep diagonal-circulant networks of depth at most l, then:

We illustrate the meaning of this properties using Figure 1 .

As we can see, the set R k of all the functions representable by a deep ReLU network of total rank k is strictly included in the set C 9k of all DCNN of depth 9k (as by Theorem 3).

Figure 1: Illustration of Properties (1) and (2).

These properties are interesting for many reasons.

First, Property (2) shows that diagonal-circulant networks are strictly more expressive than networks with low total rank.

Second and most importantly, in standard deep neural networks, it is known that the most of the singular values are close to zero (see e.g. Sedghi et al. (2018) ; Arora et al. (2019)).

Property (1) shows that these networks can efficiently be approximated by diagonal-circulant networks.

Finally, several publications have shown that neural networks can be trained explicitly to have low-rank weight matrices (Li & Shi, 2018; Goyal et al., 2019) .

This opens the possibility of learning compact and accurate diagonal-circulant networks.

Training DCNNs has revealed to be a challenging problem.

We devise two techniques to facilitate the training of deep DCNNs.

First, we propose an initialization procedure which guarantee the signal is propagated across the network without vanishing nor exploding.

Secondly, we study the behavior of DCNNs with different non-linearity functions and determine the best parameters for different settings.

Initialization scheme The following initialization procedure which is a variant of Xavier initialization.

First, for each circulant matrix C = circ(c 1 . . .

c n ), each c i is randomly drawn from N 0, 2 ,

, each d i is drawn randomly and uniformly from { 1, 1} for all i. Finally, all biases in the network are randomly drawn from N 0, 02 , for some small value of 0 .

The following proposition states that the covariance matrix at the output of any layer in a DCNN, independent of the depth, is constant.

Proposition 4.

Let N be a DCNN of depth L initialized according to our procedure, with 0 = 0.

Assume that all layers 1 to L 1 have ReLU activation functions, and that the last layer has the identity activation function.

Then, for any x 2 R n , the covariance matrix of N (x) is 2.Id n kxk 2 2 .

Moreover, note that this covariance does not depend on the depth of the network.

Non-linearity function We empirically found that reducing the number of non-linearities in the networks simplifies the training of deep neural networks.

To support this claim, we conduct a series of experiments on various DCNNs with a varying number of ReLU activations (to reduce the number of non-linearities, we replace some ReLU activations with the identity function).

In a second experiment, we replace the ReLU activations with Leaky-ReLU activations and vary the slope of the Leaky ReLU (a higher slope means an activation function that is closer to a linear function).

The results of this experiment are presented in Figure 2 (a) and 2(b).

In 2(a), "ReLU(DC)" means that we interleave on ReLU activation functions between every diagonal-circulant matrix, whereas ReLU(DCDC) means we interleave a ReLU activation every other block etc.

In both Figure 2 (a) and Figure 2 (b), we observe that reducing the non-linearity of the networks can be used to train deeper networks.

This is an interesting result, since we can use this technique to adjust the number of parameters in the network, without facing training difficulties.

We obtain a maximum accuracy of 0.56 with one ReLU every three layers and leaky-ReLUs with a slope of 0.5.

We hence rely on this setting in the experimental section.

This experimental section aims at answering the following questions: Comparison with ACDC Moczulski et al. (2015) .

In Section 2, we have discussed the differences between the ACDC framework and our approach from a theoretical perspective.

In this section, we conduct experiments to compare the performance of DCNNs with neural networks based on ACDC layers.

We first reproduce the experimental setting from , and compare both approaches using only linear networks (i.e. networks without any ReLU activations).

The results are presented in Figure 3 (a).

On this simple setting, both architectures demonstrate good performance, however, DCNNs offer better convergence rate.

In Figure 3 (b), we compare neural networks with ReLU activations on CIFAR-10.

The synthetic dataset has been created in order to reproduce the experiment on the regression linear problem proposed by .

We draw X, Y and W from a uniform distribution between [-1, +1] and ✏ from a normal distribution with mean 0 and variance 0.01.

The relationship between X and Y is define by Y = XW + ✏.

We found that networks which are based only on ACDC layers are difficult to train and offer poor accuracy on CIFAR.

(We have tried different initialization schemes including the one from the original paper, and the one we propose in this paper.) manage to train a large VGG network however these networks are generally highly redundant, the contribution of the structured layer is difficult to quantify.

We also observe that adding a single dense layer improves the convergence rate of ACDC in the linear case networks, which explain the good results of .

However, it is difficult to characterize the true contribution of the ACDC layers when the network involved a large number of other expressive layers.

In contrast, deep DCNNs can be trained and offer good performance without additional dense layers (these results are in line with our experiments on the YouTube-8M dataset).

We can conclude that DCNNs are able to model complex relations at a low cost.

Comparison with Dense networks, Toeplitz networks and Low Rank networks.

We now compare DCNNs with other state-of-the-art structured networks by measuring the accuracy on a flattened version of the CIFAR-10 dataset.

Our baseline is a dense feed-forward network with a fixed number of weights (9 million weights).

We compare with DCNNs and with DTNNs (see below), Toeplitz networks, and Low-Rank networks Yu et al. (2017) .

We first consider Toeplitz networks which are stacked Toeplitz matrices interleaved with ReLU activations since Toeplitz matrices are closely related to circulant matrices.

Since Toeplitz networks have a different structure (they do not include diagonal matrices), we also experiment using DTNNs, a variant of DCNNs where all the circulant matrices have been replaced by Toeplitz matrices.

Finally we conduct experiments using networks based on low-rank matrices as they are also closely related to our work.

For each approach, we report the accuracy of several networks with a varying depth ranging from 1 to 40 (DCNNs, Toeplitz networks) and from 1 to 30 (from DTNNs).

For low-rank networks, we used a fixed depth network and increased the rank of each matrix from 7 to 40.

We also tried to increase the depth of low rank matrices, but we found that deep low-rank networks are difficult to train so we do not report the results here.

We compare all the networks based on the number of weights from 21K (0.2% of the dense network) to 370K weights (4% of the dense network) and we report the results in Figure 4 (a).

First we can see that the size of the networks correlates positively with their accuracy which demonstrate successful training in all cases.

We can also see that the DCNNs achieves the maximum accuracy of 56% with 20 layers (⇠ 200K weights) which as as good as the dense networks with only 2% of the number of weights.

Other approaches also offer good performance but they are not able to reach the accuracy of a dense network.

Comparison with LDR networks Thomas et al. (2018) .

We now compare DCNNs with the LDR framework using the network configuration experimented in the original paper: a single LDR structured layer followed by a dense layer.

In the LDR framework, we can change the size of a network by adjusting the rank of the residual matrix, effectively capturing matrices with a structure that is close to a known structure but not exactly (e.g. in the LDR framework, Toeplitz matrices can be encoded with a residual matrix with rank=2, so a matrix that can be encoded with a residual of rank=3 can be seen as Toeplitz-like.).

The results are presented in Table 1 and demonstrate that DCNNs outperforms all LDR networks both in terms in size and accuracy.

Exploiting image features.

Dense layers and DCNNs are not designed to capture task-specific features such as the translation invariance inherently useful in image classification.

We can further improve the accuracy of such general purpose architectures on image classification without dramatically increasing the number of trained parameters by stacking them on top of fixed (i.e. non-trained) transforms such as the scattering transform (Mallat, 2010) .

In this section we compare the accuracy of various structured networks, enhanced with the scattering transform, on an image classification task, and run comparative experiments on CIFAR-10.

Our test architecture consists of 2 depth scattering on the RGB images followed by a batch norm and LDR or DC layer.

To vary the number of parameters of Scattering+LDR architecture, we increase the rank of the matrix (stacking several LDR matrices quickly exhausted the memory).

The Figure 4 (b) and 2 shows the accuracy of these architectures given the number of trainable parameters.

First, we can see that the DCNN architecture very much benefits from the scattering transform and is able to reach a competitive accuracy over 78%.

We can also see that scattering followed by a DC layer systematically outperforms scattering + LDR or scattering + Toeplitz-like with less parameters.

We provide a comparison with other compression based approaches such as HashNet Chen et al. (2015) , Dark Knowledge Hinton et al. (2015) and Fast Food Transform (FF) Yang et al. (2015) .

Table 3 shows the test error of DCNN against other know compression techniques on the MNIST datasets.

We can observe that DCNN outperform easily HashNet Chen et al. (2015) and Dark Knowledge Hinton et al. (2015) with fewer number of parameters.

The architecture with Fast Food (FF) Yang et al. (2015) achieves better performance but with convolutional layers and only 1 Fast Food Layer as the last Softmax layer.

Experiments on the aggregated dataset with DCNNs: We compared DCNNs with a dense baseline with 5.7 millions weights.

The goal of this experiment is to discover a good trade-off between depth and model accuracy.

To compare the models we use the GAP metric (Global Average Precision) following the experimental protocol in Abu- El-Haija et al. (2016b) , to compare our experiments.

Table 4 shows the results of our experiments on the aggrgated YouTube-8M dataset in terms of number of weights, compression rate and GAP.

We can see that the compression ratio offered by the circulant architectures is high.

This comes at the cost of a little decrease of GAP measure.

The 32 layers DCNN is 46 times smaller than the original model in terms of number of parameters while having a close performance.

This paper deals with the training of diagonal circulant neural networks.

To the best of our knowledge, training such networks with a large number of layers had not been done before.

We also endowed this kind of models with theoretical guarantees, hence enriching and refining previous theoretical work from the literature.

More importantly, we showed that DCNNs outperform their competing structured alternatives, including the very recent general approach based on LDR networks.

Our results suggest that stacking diagonal circulant layers with non linearities improves the convergence rate and the final accuracy of the network.

Formally proving these statements constitutes the future directions of this work.

As future work, we would like to generalize the good results of DCNNs to convolutions neural networks.

We also believe that circulant matrices deserve a particular attention in deep learning because of their strong ties with convolutions: a circulant matrix operator is equivalent to the convolution operator with circular paddings (as shown in [5]).

This fact makes any contribution to the area of circulant matrices particularly relevant to the field of deep learning with impacts beyond the problem of designing compact models.

As future work, we would like to generalize our results to deep convolutional neural networks.

Paper under double-blind review

We note R(z) and I(z) the real and imaginary parts the complex number z. We note (·) t is the t th component of a vector.

Let i be the imaginary number defined by i 2 = 1.

Define 1 n as the n-vector of ones.

Also, we note [n] = {1, . . . , n}. The rectified linear unit on the complex domain is defined by ReLU (z) = max (0, R(z)) + i max (0, I(z)).

The notation |·| refers to the complex modulus.

Finally, define the cyclic shift matrix S 2 R n⇥n as follows: We introduce some necessary definitions regarding neural networks.

In the rest of this paper, we call L and n respectively the depth and the width of the network.

Moreover, we call total rank k, the sum of the ranks of the matrices

In the rest of this paper, we call L and n respectively the depth and the width of the network.

Moreover, we call total rank k, the sum of the ranks of the matrices

Theorem 1. (Reformulation ) For any given matrix A 2 C n⇥n , for any ✏ > 0, there exists a sequence of matrices B 1 . . .

B 2n 1 where B i is a circulant matrix if i is odd, and a diagonal matrix otherwise, such that kB 1 B 2 . . .

B 2n 1 Ak < ✏.

Moreover, if A can be decomposed as A =

where S is the cyclic-shift matrix and D 1 . . .

D k are diagonal matrices, then A can be written as a product B 1 B 2 . . .

B 2k 1 where B i is a circulant matrix if i is odd, and a diagonal matrix otherwise.

Theorem 2. (Rank-based circulant decomposition) Let A 2 C n⇥n be a matrix of rank at most k. Assume that n can be divided by k. For any ✏ > 0, there exists a sequence of 4k + 1 matrices B 1 , . . .

, B 4k+1 , where B i is a circulant matrix if i is odd, and a diagonal matrix otherwise, such that kB 1 B 2 . . .

B 4k+1 Ak < ✏ Proof. (Theorem 2) Let U ⌃V T be the SVD decomposition of M where U, V and ⌃ are n ⇥ n matrices.

Because M is of rank k, the last n k columns of U and V are null.

In the following, we will first decompose U into a product of matrices W RO, where R and O are respectively circulant and diagonal matrices, and W is a matrix which will be further decomposed into a product of diagonal and circulant matrices.

Then, we will apply the same decomposition technique to V .

Ultimately, we will get a product of 4k + 2 matrices alternatively diagonal and circulant.

Let R = circ(r 1 . . .

r n ).

Let O be a n ⇥ n diagonal matrix where O i,i = 1 if i  k and 0 otherwise.

The k first columns of the product RO will be equal to that of R, and the n k last colomns of RO will be zeros.

For example, if k = 2, we have:

.

For now, the values of d ij are unknown, but we will show how to compute them.

Note that the n k last columns of the product W RO will be zeros.

For example, with k = 2, we have:

. . .

We want to find the values of d ij such that W RO = U .

We can formulate this as linear equation system.

In case k = 2, we get:

r n r 1 r n 1 r n r 1 r 2 r n r 1 r 2 r 3 r 1 r 2 . . .

The i th bloc of the bloc-diagonal matrix is a Toeplitz matrix induced by a subsequence of length k of (r 1 , . . .

r n , r 1 . . .

r n ).

Set r j = 1 for all j 2 {k, 2k, 3k, . . .

n} and set r j = 0 for all other values of j. Then it is easy to see that each bloc is a permutation of the identity matrix.

Thus, all blocs are invertible.

This entails that the block diagonal matrix above is also invertible.

So by solving this set of linear equations, we find d 1,1 . . .

d k,n such that W RO = U .

We can apply the same idea to factorize V = W 0 .R.O for some matrix W 0 .

Finally, we get

Thanks to Theorem 1, W and W 0 can both be factorized in a product of 2k 1 circulant and diagonal matrices.

Note that O⌃O T is diagonal, because all three are diagonal.

Overall, A can be represented with a product of 4k + 2 matrices, alternatively diagonal and circulant.

Intuitively, the real and imaginary parts of ⌦ are the largest any activation in the network can have.

Define h j (x) = W j x + j .

Let 1 = ⌦1 n .

Clearly, for all x 2 X we have h 1 (x) 0, so ReLU h 1 (x) = h 1 (x).

More generally, for all j < n 1 define j+1 = 1 n ⌦ W j+1 j .

It is easy to see that for all j < n we have h j . . .

h 1 (x) = W j W j 1 . . .

W 1 x + 1 n ⌦.

This guarantees that for all j < n,

Lemma 2.

Let N be a deep ReLU network of width n and depth L, and let X ⇢ C n be a bounded set.

For any ✏ > 0, there exists a DCNN N 0 of width n and of depth (2n 1)L such that kN (x)

.

By theorem 1, for any ✏ 0 > 0, any matrix W i , there exists a sequence of 2n 1 matrices

, where D i,1 is the identity matrix.

By lemma 1, we know

ReLU (W i x + b i )k will also tend to zero for any x 2 X , because the ReLU function is continuous and X is bounded.

Let

. . .

f Di1Ci1, i1 .

Again, because all functions are continuous, for all x 2 X , kN (x) N 0 (x)k tends to zero as ✏ 0 tends to zero.

n !

R + of bounded supremum norm, for any ✏ > 0, there exists a dense neural network N with an input layer of width n, an output layer of width 1, hidden layers of width n + 3 and ReLU activations such that 8x 2 [0, 1] n , |f (x) N (x)| < ✏.

From N , we can easily build a deep ReLU network N 0 of width exactly n + 3, such that 8x

Thanks to lemma 2, this last network can be approximated arbitrarily well by a DCNN of width n + 3.

Proof. (Theorem 3) Let k 1 . . .

k L be the ranks of matrices W 1 . . .

W L , which are n-by-n matrices.

For all i, there exists k 0 i 2 {k i . . .

2k i } such that k 0 i is a power of 2.

Due to the fact that n is also a power of 2, k 0 i divides n. By theorem 2, for all i each matrix W i can be decomposed as an alternating product of diagonal-circulant matrices B i,1 . .

Using the exact same technique as in lemma 2, we can build a DCNN N 0 using matrices

The total number of layers is P i (4k

Finally, what if we choose to use small depth networks to approximate deep ReLU networks where matrices are not of low rank?

To answer this question, we first need to show the negative impact of replacing matrices by their low rank approximators in neural networks:

where R is an upper bound on norm of the output of any layer in N , and max,j = max i i,j .

By lemma 3, we have

Observe that for any sequence a 0 , a 1 . . .

defined recurrently by a 0 = 0 and a i = ra i 1 + s, the recurrence relation can be unfold as follows: a i = s(r i 1) r 1 .

We can apply this formula to bound our error as follows:

with singular values 1 . . .

n , and let x,x 2 C n .

LetW be the matrix obtained by a SVD approximation of rank k of matrix W .

Then we have:

, because 1 is the greatest singular value of both W andW .

Also, note that W W 2 = k+1 .

Let us bound the formula without ReLUs:

Finally, it is easy to see that for any pair of vectors a, b 2 C n , we have kReLU (a) ReLU (b)k  ka bk.

This concludes the proof.

where R is an upper bound on the norm of the outputs of each layer in N .

, where eachW i is the matrix obtained by an SVD approximation of rank k of matrix W i .

With Proposition 4, we have an error bound on kN (x) Ñ (x) k. Now each matrixW i can be replaced by a product of k diagonal-circulant matrices.

By theorem 3, this product yields a DCNN of depth m = L(4k + 1), strictly equivalent toÑ on X .

The result follows.

Proposition 5.

Let N be a DCNN of depth L initialized according to our procedure, with 0 = 0.

Assume that all layers 1 to L 1 have ReLU activation functions, and that the last layer has the identity activation function.

Then, for any x 2 R n , the covariance matrix of N (x) is To derive cov(z i , z i 0 ) and cov(z i ,z i 0 ) , the required calculus is nearly identical.

We let the reader check by himself/herself.

Architectures & Hyper-Parameters:

For the first set of our experiments (e.g. experiments on CIFAR-10), we train all networks for 200 epochs, a batch size of 200, Leaky ReLU activation with a different slope.

We minimize the Cross Entropy Loss with Adam optimizer and use a piecewise constant learning rate of 5 ⇥ 10e 5, 2.5 ⇥ 10e 5, 5 ⇥ 10e 6 and 1 ⇥ 10e 6 after respectively 40000, 60000 and 80000 steps.

For the YouTube-8M dataset experiments, we built a neural network based on the state-of-the-art architecture initially proposed by Abu-El-Haija et al. (2016) and later improved by .

Remark that no convolution layer is involved in this application since the input vectors are embeddings of video frames processed using state-of-the-art convolutional neural networks trained on ImageNet.

We trained our models with the CrossEntropy loss and used Adam optimizer with a 0.0002 learning rate and a 0.8 exponential decay every 4 million examples.

All fully connected layers are composed of 512 units.

DBoF, NetVLAD and NetFV are respectively 8192, 64 and 64 of cluster size for video frames and 4096, 32, 32 for audio frames.

We used 4 mixtures for the MoE Layer.

We used all the available 300 frames for the DBoF embedding.

In order to stabilize and accelerate the training, we used batch normalization before each non linear activation and gradient clipping.

@highlight

We train deep neural networks based on diagonal and circulant matrices, and show that this type of networks are both compact and accurate on real world applications.

@highlight

The authors provide a theoretical analysis of the expressive power of diagonal circulant neural networks (DCNN) and propose an initialization scheme for deep DCNNs.