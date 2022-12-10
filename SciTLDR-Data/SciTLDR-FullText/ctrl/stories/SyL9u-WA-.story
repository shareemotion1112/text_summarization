Vanishing and exploding gradients are two of the main obstacles in training deep neural networks, especially in capturing long range dependencies in recurrent neural networks (RNNs).

In this paper, we present an efficient parametrization of the transition matrix of an RNN that allows us to stabilize the gradients that arise in its training.

Specifically, we parameterize the transition matrix by its singular value decomposition (SVD), which allows us to explicitly track and control its singular values.

We attain efficiency by using tools that are common in numerical linear algebra, namely Householder reflectors for representing the orthogonal matrices that arise in the SVD.

By explicitly controlling the singular values, our proposed svdRNN method allows us to easily solve the exploding gradient problem and we observe that it empirically solves the vanishing gradient issue to a large extent.

We note that the SVD parameterization can be used for any rectangular weight matrix, hence it can be easily extended to any deep neural network, such as a multi-layer perceptron.

Theoretically, we demonstrate that our parameterization does not lose any expressive power, and show how it potentially makes the optimization process easier.

Our extensive  experimental results also demonstrate that the proposed framework converges faster, and has good generalization, especially when the depth is large.

Deep neural networks have achieved great success in various fields, including computer vision, speech recognition, natural language processing, etc.

Despite their tremendous capacity to fit complex functions, optimizing deep neural networks remains a contemporary challenge.

Two main obstacles are vanishing and exploding gradients, that become particularly problematic in Recurrent Neural Networks (RNNs) since the transition matrix is identical at each layer, and any slight change to it is amplified through recurrent layers BID3 ).Several methods have been proposed to solve the issue, for example, Long Short Term Memory (LSTM) BID8 ) and residual networks BID7 ).

Another recently proposed class of methods is designed to enforce orthogonality of the square transition matrices, such as unitary and orthogonal RNNs (oRNN) BID1 ; BID13 ).

However, while these methods solve the exploding gradient problem, they limit the expressivity of the network.

In this paper, we present an efficient parametrization of weight matrices that arise in a deep neural network, thus allowing us to stabilize the gradients that arise in its training, while retaining the desired expressive power of the network.

In more detail we make the following contributions:• We propose a method to parameterize weight matrices through their singular value decomposition (SVD).

Inspired by BID13 ), we attain efficiency by using tools that are common in numerical linear algebra, namely Householder reflectors for representing the orthogonal matrices that arise in the SVD.

The SVD parametrization allows us to retain the desired expressive power of the network, while enabling us to explicitly track and control singular values.• We apply our SVD parameterization to recurrent neural networks to exert spectral constraints on the RNN transition matrix.

Our proposed svdRNN method enjoys similar space and time complexity as the vanilla RNN.

We empirically verify the superiority of svdRNN over RNN/oRNN, in some case even LSTMs, over an exhaustive collection of time series classification tasks and the synthetic addition and copying tasks, especially when the network depth is large.• Theoretically, we show how our proposed SVD parametrization can make the optimization process easier.

Specifically, under a simple setting, we show that there are no spurious local minimum for the linear svdRNN in the population risk.• Our parameterization is general enough to eliminate the gradient vanishing/exploding problem not only in RNNs, but also in various deep networks.

We illustrate this by applying SVD parametrization to problems with non-square weight matrices, specifically multi-layer perceptrons (MLPs) and residual networks.

We now present the outline of our paper.

In Section 2, we discuss related work, while in Section 3 we introduce our SVD parametrization and demonstrate how it spans the whole parameter space and does not limit expressivity.

In Section 4 we propose the svdRNN model that is able to efficiently control and track the singular values of the transition matrices, and we extend our parameterization to non-square weight matrices and apply it to MLPs in Section 5.

Section 6 provides the optimization landscape of svdRNN by showing that linear svdRNN has no spurious local minimum.

Experimental results on MNIST and a popular time series archive are present in Section 7.

Finally, we present our conclusions and future work in Section 8.

Numerous approaches have been proposed to address the vanishing and exploding gradient problem.

Long short-term memory (LSTM) BID8 ) attempts to address the vanishing gradient problem by adding additional memory gates.

Residual networks BID7 ) pass the original input directly to the next layer in addition to the original layer output.

BID14 performs gradient clipping, while BID15 applies spectral regularization to the weight matrices.

Other approaches include introducing L 1 or L 2 penalization on successive gradient norm pairs in back propagation BID15 ).Recently the idea of restricting transition matrices to be orthogonal has drawn some attention.

BID12 proposed initializing recurrent transition matrices to be identity or orthogonal (IRNN).

This strategy shows better performance when compared to vanilla RNN and LSTM.

However, there is no guarantee that the transition matrix is close to orthogonal after a few iterations.

The unitary RNN (uRNN) algorithm proposed in BID1 parameterizes the transition matrix with reflection, diagonal and Fourier transform matrices.

By construction, uRNN ensures that the transition matrix is unitary at all times.

Although this algorithm performs well on several small tasks, BID19 showed that uRNN only covers a subset of possible unitary matrices and thus detracts from the expressive power of RNN.

An improvement over uRNN, the orthogonal RNN (oRNN), was proposed by BID13 .

oRNN uses products of Householder reflectors to represent an orthogonal transition matrix, which is rich enough to span the entire space of orthogonal matrices.

Meanwhile, BID18 empirically demonstrate that the strong constraint of orthogonality limits the model's expressivity, thereby hindering its performance.

Therefore, they parameterize the transition matrix by its SVD, W = U ΣV (factorized RNN) and restrict Σ to be in a range close to 1; however, the orthogonal matrices U and V are updated by geodesic gradient descent using the Cayley transform, thereby resulting in time complexity cubic in the number of hidden nodes which is prohibitive for large scale problems.

Motivated by the shortcomings of the above methods, our work in this paper attempts to answer the following questions: Is there an efficient way to solve the gradient vanishing/exploding problem without hurting expressive power?As brought to wide notice in BID7 , deep neural networks should be able to preserve features that are already good.

BID6 consolidate this point by showing that deep linear residual networks have no spurious local optima.

In our work, we broaden this concept and bring it to the area of recurrent neural networks, showing that each layer is not necessarily near identity, but being close to orthogonality suffices to get a similar result.

Generalization is a major concern in training deep neural networks.

BID2 provide a generalization bound for neural networks by a spectral Lipschitz constant, namely the product of spectral norm of each layer.

Thus, our scheme of restricting the spectral norm of weight matrices reduces generalization error in the setting of BID2 .

As supported by the analysis in BID5 , since our SVD parametrization allows us to develop an efficient way to constrain the weight matrix to be a tight frame BID17 ), we consequently are able to reduce the sensitivity of the network to adversarial examples.

The SVD of the transition matrix W ∈ R n×n of an RNN is given by W = U ΣV T , where Σ is the diagonal matrix of singular values, and U, V ∈ R n×n are orthogonal matrices, i.e., BID16 ).

During the training of an RNN, our proposal is to maintain the transition matrix in its SVD form.

However, in order to do so efficiently, we need to maintain the orthogonal matrices U and V in compact form, so that they can be easily updated by forward and backward propagation.

In order to do so, as in BID13 , we use a tool that is commonly used in numerical linear algebra, namely Householder reflectors (which, for example, are used in computing the QR decomposition of a matrix).

DISPLAYFORM0 Given a vector u ∈ R k , k ≤ n, the n × n Householder reflector H n k (u) is defined as: DISPLAYFORM1 (1)The Householder reflector is clearly a symmetric matrix, and it can be shown that it is orthogonal, i.e., H 2 = I (Householder (1958)).

Further, when u = 0, it has n−1 eigenvalues that are 1, and one eigenvalue which is −1 (hence the name that it is a reflector) .

In practice, to store a Householder reflector, we only need to store u ∈ R k rather than the full matrix.

Given a series of vectors {u i } n i=k where u k ∈ R k , we define the map: DISPLAYFORM2 where the right hand side is a product of Householder reflectors, yielding an orthogonal matrix (to make the notation less cumbersome, we remove the superscript from H n k for the rest of this section).

Theorem 1.

The image of M 1 is the set of all n × n orthogonal matrices.

The proof of Theorem 1 is an easy extension of the Householder QR factorization Theorem, and is presented in Appendix A. Although we cannot express all n × n matrices with M k , any W ∈ R n×n can be expressed as the product of two orthogonal matrices U, V and a diagonal matrix Σ, i.e. by its SVD: DISPLAYFORM3 , we finally define our proposed SVD parametrization: DISPLAYFORM4 Theorem 2.

The image of M 1,1 is the set of n × n real matrices.

i.e. DISPLAYFORM5 The proof of Theorem 2 is based on the singular value decomposition and Theorem 1, and is presented in Appendix A. The astute reader might note that M 1,1 seemingly maps an input space of n 2 + 2n dimensions to a space of n 2 dimensions; however, since H n k (u k ) is invariant to the norm of u k , the input space also has exactly n 2 dimensions.

Although Theorems 1 and 2 are simple extensions of well known linear algebra results, they ensure that our parameterization has the ability to represent any matrix and so the full expressive power of the RNN is retained.

Theorem 3.

The image of M k1,k2 includes the set of all orthogonal n×n matrices if k 1 +k 2 ≤ n+2.Theorem 3 indicates that if the total number of reflectors is greater than n: (n − k 1 + 1) + (n − k 2 + 1) ≥ n, then the parameterization covers all orthogonal matrices.

Note that when fixing σ = 1, DISPLAYFORM6

In this section, we apply our SVD parameterization to RNNs and describe the resulting svdRNN algorithm in detail.

Given a hidden state vector from the previous step h (t−1) ∈ R n and input x (t−1) ∈ R ni , RNN computes the next hidden state h (t) and output vector o (t) ∈ R no as: DISPLAYFORM0 DISPLAYFORM1 In svdRNN we parametrize the transition matrix W ∈ R n×n using m 1 + m 2 Householder reflectors as: DISPLAYFORM2 This parameterization gives us several advantages over the regular RNN.

First, we can select the number of reflectors m 1 and m 2 to balance expressive power versus time and space complexity.

By Theorem 2, the choice m 1 = m 2 = n gives us the same expressive power as vanilla RNN.

Notice oRNN could be considered a special case of our parametrization, since when we set m 1 + m 2 ≥ n and σ = 1, we can represent all orthogonal matrices, as proven by Theorem 3.

Most importantly, we are able to explicitly control the singular values of the transition matrix.

In most cases, we want to constrain the singular values to be within a small interval near 1.

The most intuitive method is to clip the singular values that are out of range.

Another approach would be to initialize all singular values to 1, and add a penalty term σ − 1 2 to the objective function.

Here, we have applied another parameterization of σ proposed in BID18 : DISPLAYFORM3 (8) where f is the sigmoid function andσ i is updated from u i , v i via stochastic gradient descent.

The above allows us to constrain σ i to be within [σ * − r, σ * + r].

In practice, σ * is usually set to 1 and r 1.

Note that we are not incurring more computation cost or memory for the parameterization.

For regular RNN, the number of parameters is (n o + n i + n + 1)n, while for svdRNN it is (n o + DISPLAYFORM4 .

In the extreme case where m 1 = m 2 = n, it becomes (n o + n i + n + 3)n.

Later we will show that the computational cost of svdRNN is also of the same order as RNN in the worst case.

In forward propagation, we need to iteratively evaluate h (t) from t = 0 to L using (4).

The only different aspect from a regular RNN in the forward propagation is the computation of W h (t−1) .

Note that in svdRNN, W is expressed as product of m 1 + m 2 Householder matrices and a diagonal matrix.

Thus W h (t−1) can be computed iteratively using (m 1 + m 2 ) inner products and vector additions.

Denotingû k = 0 n−k u k , we have: DISPLAYFORM0 Thus, the total cost of computing W h (t−1) is O((m 1 + m 2 )n) floating point operations (flops).

Detailed analysis can be found in Section 4.2.

Let L({u i }, {v i }, σ, M, Y, b) be the loss or objective function, DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Back propagation for svdRNN requires DISPLAYFORM4 ∂Σ (t) and DISPLAYFORM5 ∂h (t−1) .

These partial gradients can also be computed iteratively by computing the gradient of each Householder matrix at a time.

We drop the superscript (t) now for ease of exposition.

Givenĥ = H k (u k )h and g = ∂L ∂ĥ, we have DISPLAYFORM6 Details of forward and backward propagation can be found in Appendix (B).

One thing worth noticing is that the oRNN method in BID13 actually omitted the last term in (15) by assuming that u k are fixed.

Although the scaling of u k in the Householder transform does not affect the transform itself, it does produce different gradient update for u k even if it is scaled to norm 1 afterwards.

TAB0 gives the time complexity of various algorithms.

Hprod and Hgrad are defined in Algorithm 2 3 (see Appendix (B)).

Algorithm 2 needs 6k flops, while Algorithm 3 uses (3n + 10k) flops.

Since u k 2 only needs to be computed once per iteration, we can further decrease the flops to 4k and (3n + 8k).

Also, in back propagation we can reuse α in forward propagation to save 2k flops.

flops DISPLAYFORM0

In this section, we extend the parameterization to non-square matrices and use Multi-Layer Perceptrons(MLP) as an example to illustrate its application to general deep networks.

For any weight matrix W ∈ R m×n (without loss of generality m ≤ n), its reduced SVD can be written as: DISPLAYFORM0 DISPLAYFORM1 .

Thus we can extend the SVD parameterization for any non-square matrix: DISPLAYFORM2 (17) whereΣ = (diag(σ)|0) if m < n and (diag(σ)|0) otherwise.

Next we show that we only need 2 min(m, n) reflectors (rather than m + n) to parametrize any m × n matrix.

By the definition of H n k , we have the following lemma: DISPLAYFORM3 Here V * ,i indicates the ith column of matrix V .

According to Lemma 1, we only need at most first m Householder vectors to express V L , which results in the following Theorem: Theorem 4.

If m ≤ n, the image of M m,n 1,n−m+1 is the set of all m × n matrices; else the image of M m,n n−m+1,1 is the set of all m × n matrices.

Similarly if we constrain u i , v i to have unit length, the input space dimensions of M m,n 1,n−m+1 and M m,n m−n+1,1 are both mn, which matches the output dimension.

Thus we extend Theorem 2 to the non-square case, which enables us to apply SVD parameterization to not only the RNN transition matrix, but also to general weight matrices in various deep learning models.

For example, the Multilayer perceptron (MLP) model is a class of feedforward neural network with fully connected layers: DISPLAYFORM4 say n t < n t−1 , we have: DISPLAYFORM5 nt−1 (v nt−1 ).

We can use the same forward/backward propagation algorithm as described in Algorithm 1.

Besides RNN and MLP, SVD parameterization method also applies to more advanced frameworks, such as Residual networks and LSTM, which we will not describe in detail here.

Since we can control and upper bound the singular values of the transition matrix in svdRNN, we can clearly eliminate the exploding gradient problem.

In this section, we now analytically illustrate the advantages of svdRNN with lower-bounded singular values from the optimization perspective.

For the theoretical analysis in this section, we will limit ourselves to a linear recurrent neural network, i.e., an RNN without any activation.

Linear recurrent neural network.

For simplicity, we follow a setting similar to BID6 .

For compact presentation, we stack the input data as X ∈ R n×t , where X = x (0) |x (1) | · · · |x (t−1) , and transition weights as W ∈ R n×nt where W = W |W 2 | · · · |W t .

Then we can simplify the output as: DISPLAYFORM0 By absorbing M and b in each data x (t) and assuming h (0) = 0, we further simplify the output as: DISPLAYFORM1 Suppose the input data X ∼ D, and assume its underlying relation to the output is y = Avec(X )+η, where A ∈ R n×nt and residue η ∈ R n satisfies E X ∼D [η|X ] = 0.

We consider the individual loss: DISPLAYFORM2 2 .

Claim 1.

With linear recurrent neural networks, the population risk DISPLAYFORM3 DISPLAYFORM4 Therefore when ∇ W R[W ] = 0 suffices R(W ) = R * , meaning W reaches the global minimum.

Theorem 5 potentially explains why our system is easier to optimize, since with our scheme of SVD parametrization, we have the following corollary.

Corollary 1.

With the update rule in (8), linear svdRNNs have no spurious local minimum.

While the above analysis lends further credence to our observed experimental results, we leave it to future work to perform a similar analysis in the presence of non-linear activation functions.

In this section, we provide empirical evidence that shows the advantages of SVD parameterization in both RNNs and MLPs.

For RNN models, we compare our svdRNN algorithm with (vanilla) RNN, IRNN(Le et al. FORMULA2 ), oRNN BID13 ) and LSTM BID8 ).

The transition matrix in IRNN is initialized to be orthogonal while other matrices are initialized by sampling from a Gaussian distribution.

For MLP models, we implemented vanilla MLP, Residual Network (ResNet) BID7 ) and used SVD parameterization for both of them.

We used a residual block of two layers in ResNet.

In most cases leaky Relu is used as activation function, except for LSTM, where leaky Relu will drastically harm the performance.

To train these models, we applied Adam optimizer with stochastic gradient descent BID11 ).

These models are implemented with Theano (Al-Rfou et al. FORMULA2 ).

In this experiment, we focus on the time series classification problem, where time series are fed into RNN sequentially, which then tries to predict the right class upon receiving the sequence end BID10 ).

The dataset we choose is the largest public collection of classlabeled time-series with widely varying length, namely, the UCR time-series collection from BID4 2 .

We present the test accuracy on 20 datasets with RNN, LSTM, oRNN and svdRNN in TAB3 (Appendix C) and Figure 1 .

In all experiments, we used hidden dimension n h = 32, and chose total number of reflectors for oRNN and svdRNN to be m = 16 (for svdRNN m 1 = m 2 = 8).We choose proper depth t as well as input size n i .

Given sequence length L, since tn i = L, we choose n i to be the maximum divisor of L that satisfies depth ≤ √ L. To have a fair comparison (a) (b) (c) Figure 1 : Performance comparisons of the RNN based models on three UCR datasets.of how the proposed principle itself influences the training procedure, we did not use dropout in any of these models.

As illustrated in the optimization process in Figure 1 , this resulted in some overfitting (see (a) CBF), but on the other hand it shows that svdRNN is able to prevent overfitting.

This supports our claim that since generalization is bounded by the spectral norm of the weights BID2 , svdRNN will potentially generalize better than other schemes.

This phenomenon is more drastic when the depth is large (e.g. ArrowHead(251 layers) and FaceAll(131 layers)), since regular RNN, and even LSTM, have no control over the spectral norms.

Also note that there are substantially fewer parameters in oRNN and svdRNN as compared to LSTM.

In this experiment, we compare different models on the MNIST image dataset.

The dataset was split into a training set of 60000 instances and a test set of 10000 instances.

The 28 × 28 MNIST pixels are flattened into a vector and then traversed by the RNN models.

Table 2 shows accuracy scores across multiple We tested different models with different network depth as well as width.

Figure 2(a)(b) shows the test accuracy on networks with 28 and 112 layers (20 and 128 hidden dimensions) respectively.

It can be seen that the svdRNN algorithms have the best performance and the choice of r (the amount that singular values are allowed to deviate from 1) does not have much influence on the final precision.

Also we explored the effect of different spectral constraints and explicitly tracked the spectral margin (max i |σ i − 1|) of the transition matrix.

Intuitively, the influence of large spectral margin should increase as the network becomes deeper.

Figure 2(d) shows the spectral margin of different RNN models.

Although IRNN has small spectral margin at first few iterations, it quickly deviates from orthogonal and cannot match the performance of oRNN and svdRNN.

Figure 2(e) shows the magnitude of first layer gradient ∂L ∂h (0) 2 .

RNN suffers from vanishing gradient at first 50k iterations while oRNN and svdRNN are much more stable.

Note that LSTM can perform relatively well even though it has exploding gradient in the first layer.

We also tested RNN and svdRNN with different amount of non-linearity, as shown in Figure 2 (c).

This is achieved by adjusting the leak parameter in leaky Relu: f (x) = max(leak · x, x).

With leak = 1.0, it reduces to the identity map and when leak = 0 we are at the original Relu function.

From the figures, we show that svdRNN is resistant to different amount of non-linearity, namely converge faster and achieve higher accuracy invariant to the amount of the leak factor.

To explore the reason underneath, we illustrate the gradient in Figure 2 (f), and find out svdRNN could eliminate the gradient vanishing problem on all circumstances, while RNN suffers from gradient vanishing when non-linearity is higher.

FORMULA2 256(m = 32) ≈ 11k 97.2 RNN BID18 128 ≈ 35k 94.1 uRNN BID1 ) 512 ≈ 16k 95.1 RC uRNN BID19 ) 512 ≈ 16k 97.5 FC uRNN BID19 ) 116 ≈ 16k 92.8 factorized RNN BID18 ) 128 ≈ 32k 94.6 LSTM BID18 ) 128 ≈ 64k 97.3 Table 2 : Results for the pixel MNIST dataset across multiple algorithms.

For the MLP models, each instance is flattened to a vector of length 784 and fed to the input layer.

After the input layer there are 40 layers with hidden dimension 32 (Figure 3(a) ) or 30 to 100 layers with hidden dimension 128 (Figure 3(b) ).

On a 40-layer network, svdMLP and svdResNet achieve similar performance as ResNet while MLP's convergence is slower.

However, when the network is deeper, both MLP and ResNet start to fail.

With n h = 128, MLP is not able to function with L > 35 and ResNet with L > 70.

On the other hand, the SVD based methods are resilient to increasing depth and thus achieve higher precision.(a) (b) Figure 3 : MLP models on MNIST with L layers n h hidden dimension

In this paper, we have proposed an efficient SVD parametrization of various weight matrices in deep neural networks, which allows us to explicitly track and control their singular values.

This parameterization does not restrict the network's expressive power, while simultaneously allowing fast forward as well as backward propagation.

The method is easy to implement and has the same time and space complexity as compared to original methods like RNN and MLP.

The ability to control singular values helps in avoiding the gradient vanishing and exploding problems, and as we have empirically shown, gives good performance.

Although we only showed examples in the RNN and MLP framework, our method is applicable to many more deep networks, such as Convolutional Networks etc.

However, further experimentation is required to fully understand the influence of using different number of reflectors in our SVD parameterization.

Also, the underlying structures of the image of M k1,k2 when k 1 , k 2 = 1 is a subject worth investigating.

DISPLAYFORM0 Proof of Proposition 1.

For n = 1, note that H 1 1 (u 1 ) = ±1.

By setting u 1 = 0 if B 1,1 > 0 and u 1 = 0 otherwise, we have the factorization desired.

Assume that the result holds for n = k, then for n = k + 1 set u k+1 = B 1 − B 1 e 1 .

Here B 1 is the first column of B and e 1 = (1, 0, ..., 0) .

Thus we have DISPLAYFORM1 , whereB ∈ R k×k .

Note that H k+1 k+1 (u k+1 ) = I k+1 when u k+1 = 0 and the above still holds.

By DISPLAYFORM2 is an upper triangular matrix with positive diagonal elements.

Thus the result holds for any n by the theory of mathematical induction.

A.2 PROOF OF THEOREM 1 Proof.

Observe that the image of M 1 is a subset of O(n), and we now show that the converse is also true.

Given A ∈ O(n), by Proposition 1, there exists an upper triangular matrix R with positive diagonal elements, and an orthogonal matrix Q expressed as DISPLAYFORM3 , such that A = QR.

Since A is orthogonal, we have A A = AA = I n , thus:A A = R Q QR = R R = I n ; Q AA Q = Q QRR Q Q = RR = I n Thus R is orthogonal and upper triangular matrix with positive diagonal elements.

So R = I n and DISPLAYFORM4

Proof.

It is easy to see that the image of M 1,1 is a subset of R n×n .

For any W ∈ R n×n , we have its SVD, W = U ΣV , where Σ = diag(σ).

By Theorem 1, for any orthogonal matrix U, V ∈ R n×n , there exists DISPLAYFORM0 Proof.

Let A ∈ R n×n be an orthogonal matrix.

By Theorem 1, there exist DISPLAYFORM1 , such that A = M 1 (a 1 , ..., a n ).

Since A is also orthogonal, for the same reason, there exist DISPLAYFORM2 v t = 0, t = k 2 + k 1 − 2, ..., n, and then we have: DISPLAYFORM3 Else, assign: DISPLAYFORM4 .., n, and then we have: DISPLAYFORM5 A.5 PROOF OF THEOREM 4Proof.

It is easy to see that the image of M m,n * , * is a subset of R m×n .

For any W ∈ R m×n , we have its SVD, W = U ΣV , where Σ is an m × n diagonal matrix.

By Theorem 1, for any orthogonal DISPLAYFORM6 Similarly, for n < m, we have: DISPLAYFORM7 Remark: here when W and ∆W are not commutative, each W i ∆W should instead be written as DISPLAYFORM8 Since the change of order doesn't impact the analysis, we informally simplify the expressions here.

DISPLAYFORM9 C.1 DETAILS ON THE TIME SERIES CLASSIFICATION TASK For the time series classification task, we use the training and testing sets directly from the UCR time series archive http://www.cs.ucr.edu/˜eamonn/time_series_data/, and randomly choose 20% of the training set as validation data.

We provide the statistical descriptions of the datasets and experimental results in TAB3 BID1 .

The Adding task requires the network to remember two marked numbers in a long sequence and add them.

Each input data includes two sequences: top sequence whose values are sampled uniformly from [0, 1] and bottom sequence which is a binary sequence with only two 1's.

The network is asked to output the sum of the two values.

From the empirical results in Figure 4 , we can see that when the network is not deep (number of layers L=30 in (a)(d)), every model outperforms the baseline of 0.167 (always output 1 regardless of the input).

Also, the first layer gradients do not vanish for all models.

However, on longer sequences (L=100 in (b)(e)), IRNN failed and LSTM converges much slower than svdRNN and oRNN.

If we further increase the sequence length (L=300 in (c)(f)), only svdRNN and oRNN are able to beat the baseline within reasonable number of iterations.

We can also observe that the first layer gradient of oRNN/svdRNN does not vanish regardless of the depth, while IRNN/LSTM's gradient vanish as L becomes lager.(a) (b) (c) (d) (e) (f) Figure 4 : RNN models on the adding task with L layers and n h hidden dimension.

The top plots show the test MSE, while the bottom plots show the magnitude of the gradient at the first layer.

Let A = {a i } 9 i=0 be the alphabet.

The input data sequence x ∈ A T +20 where T is the time lag.

x 1:10 are sampled uniformly from i{a i } 7 i=0 and x T +10 is set to a 9 .

Rest of x i is set to a 8 .

The network is asked to output x 1:10 after seeing a 9 .

That is to copy x 1:10 from the beginning to the end with time lag T .A baseline strategy is to predict a 8 for T +10 entrees and randomly sample from {a i } 7 i=1 for the last 10 digits.

From the empirical results in FIG3 , svdRNN consistently outperforms all other models.

IRNN and LSTM models are not able to beat the baseline with large time lag.

In fact, the loss of RNN/LSTM is very close to the baseline (memoryless strategy) indicates that they do not memorize any useful information throughout the time lag.

<|TLDR|>

@highlight

To solve the gradient vanishing/exploding problems, we proprose an efficient parametrization of the transition matrix of RNN that loses no expressive power, converges faster and has good generalization.