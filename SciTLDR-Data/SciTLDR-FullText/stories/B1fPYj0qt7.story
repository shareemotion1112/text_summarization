The Tensor-Train factorization (TTF) is an efficient way to compress large weight matrices of fully-connected layers and recurrent layers in recurrent neural networks (RNNs).

However, high Tensor-Train ranks for all the core tensors of parameters need to be element-wise fixed, which results in an unnecessary redundancy of model parameters.

This work applies Riemannian stochastic gradient descent (RSGD) to train core tensors of parameters in the Riemannian Manifold before finding vectors of lower Tensor-Train ranks for parameters.

The paper first presents the RSGD algorithm with a convergence analysis and then tests it on more advanced Tensor-Train RNNs such as bi-directional GRU/LSTM and Encoder-Decoder RNNs with a Tensor-Train attention model.

The experiments on digit recognition and machine translation tasks suggest the effectiveness of the RSGD algorithm for Tensor-Train RNNs.

Recurrent Neural Networks (RNNs) are typically composed of large weight matrices of fullyconnected and recurrent layers, thus massive training data as well as exhaustive computational resources are required.

The Tensor-Train factorization (TTF) aims to reduce the redundancy of RNN parameters by reshaping large weight matrices into high-dimensional tensors before factorizing them in a Tensor-Train format BID10 .

The notation of Tensor-Train usually suggests that TTF is applied for the tensor representation of model parameters.

Tensor-Train was initially applied to fully-connected layers BID8 , and it has been recently generalized to recurrent layers in RNNs such as LSTM and GRU BID13 .

Compared with other tensor decomposition techniques like the CANDECOMP/PARAFAC decomposition BID5 and Tucker decomposition BID4 , Tensor-Train can be easily scaled to arbitrarily high dimensions and have the advantage of computational tractability to significantly large weight matrices.

Given a vector of Tensor-Train ranks r = (r 1 , r 2 , · · ·, r d+1 ), TTF decomposes a d-dimensional tensor W ∈ R (m1·n1)×(m2·n2)×···×(m d ·n d ) into a multiplication of core tensors according to (1),where the k-th core tensor C [k] ∈ R r k ×m k ×n k ×r k+1 , and any index pair (i k , j k ) satisfies 1 ≤ i k ≤ m k , 1 ≤ j k ≤ n k .

Additionally, the ranks r 1 and r d+1 are fixed to 1.

1, j 1 ), (i 2 , j 2 ), ..., (i d , j d )) = C [1] (r 1 , i 1 , j 1 , r 2 )C [2] (r 2 , i 2 , j 2 , r 3 )···C DISPLAYFORM0 Thus, when the TTF technique is applied to the fully-connected (FC) layer with feed-forward weight matrix W , a tensor W is firstly converted from W and is then decomposed to a multiplication of the core tensors as shown in (2), where C [t] is the t-th core tensor, X denotes a tensor of input, B refers to a tensor of bias, the tensor of outputsŶ ∈ R n1×n2×···×n d , and σ is a sigmoid function.

For clarity, the notation T T L(W , X) is used to simplify the representation of a Tensor-Train fully-connected layer, which is shown in (3).

DISPLAYFORM1 Likely, we use (4) to represent an RNN with a feed-forward weight matrix W and a recurrent weight matrix U .

In (4), X t is an input matrix at time t, h t−1 and h t separately denote the hidden vectors of time t − 1 and t, and σ refers to the sigmoid function.

DISPLAYFORM2 The largest benefit from Tensor-Train models is the capability to reduce the model parameters tremendously.

For example, a Tensor-Train FC layer needs i m i · n i ·

r i · r i+1 parameters in total.

In comparison, the total number of parameters of an FC layer is about O( i m i · i n i ), which is much larger than the associated Tensor-Train FC one.

The Tensor-Train models are found widespread.

For example, BID12 Contributions.

We summarize the key contributions of this paper as follows:• This work applies RSGD to iteratively find vectors of lower Tensor-Train ranks with the update of parameters in the training process.

The RSGD algorithm and the related theoretical analysis are also presented.• We design Bi-directional Tensor-Train GRU/LSTM BID2 , and EncoderDecoder Tensor-Train RNNs with a Tensor-Train Attention mechanism BID7 .

We apply the RSGD algorithm to the Tensor-Train RNN models on the digit recognition and machine translation tasks.

To the best of our knowledge, this is the first work that applies RSGD to train Tensor-Train RNNs to find the optimal core tensors of parameters with vectors of lower Tensor-Train ranks.

Moreover, this is the first work that builds Tensor-Train RNNs with complex architectures for natural language processing (NLP) tasks.

The optimization problem of Tensor-Train RNNs can be formulated as a Riemannian optimization problem as shown in FORMULA3 , where {X, Y } is a data sequence with length T , W represents a tensor of parameters which lies in a d-dimensional Riemannian Manifold (M, µ) with a Riemannian measure µ , and the Tensor-Train ranks for core tensors of W must be element-wise no higher than the vector r = (r 1 , r 2 , ..., r d+1 ) as shown in FORMULA3 .

DISPLAYFORM0 Besides, the Riemannian measure µ induces an inner product structure in each tangent space T x M associated with a tensor x ∈ M .

Specifically, ∀u, v ∈ T x M , the inner product < u, v >= µ x (u, v).Algorithm 1 Riemannian Stochastic Gradient Descent 1.

Given the labeled input data (X, Y ) with sequence length T , and the learning rate η.

DISPLAYFORM1 Choose a gradient g DISPLAYFORM2 ).9.

DISPLAYFORM3 , · · ·Ĉ DISPLAYFORM4 Reshape the core tensors DISPLAYFORM5 Similarly, µ induces the norm of u ∈ T x M as ||u|| = µ x (u, u) ≥ 0.

In addition, the µ induced inner product and the norm preserve the basic properties like definiteness, homogeneity and triangle inequality.

Algorithm 1 presents the RSGD Algorithm.

The algorithm mainly consists of two main procedures: one is the update of parameters in the tangent space and conducting an exponential mapping, and the second one is the rounding to lower Tensor-Train ranks.

As illustrated in FIG0 , step 5 firstly obtains a gradient g C [i] on a tangent space T Ci M at the core tensor C [i] in Riemannian Manifold (M, µ), and step 6 conducts a gradient descent on the tangent space to generate a new tensorÂ i .Step 7 projectsÂ i back toĈ [i] in Riemannian Manifold (M, µ) by an exponential mapping.

Finally, as shown in FIG1 , the rounding function in step 8 transformsĈ [i] in the submanifold S r to the core tensorĈ [i] with a vector of lower Tensor-Train ranksr in a new submanifold Sr.

Note that the vectors of Tensor-Train ranks r andr span two submanifolds S r ⊂ M and Sr ⊂ M respectively.

After that, the next iteration of the the parameter update is conducted in Sr.

The exponential mapping in Algorithm 1 is formulated in (6).

Unfortunately, it is not easy to solve the problem because we have to deal with the calculus of variations, or we have to know the Christoffel symbols BID6 .

Therefore, a fast and straightforward retraction method is applied as a first-order approximation to the exponential mapping as shown in Algorithm 2.

min DISPLAYFORM6 1.

Given the core tensors DISPLAYFORM7 Algorithm 3 The Rounding Algorithm DISPLAYFORM8 8.r i+1 = min(r max , col num( DISPLAYFORM9 Algorithm 2 presents the retraction algorithm.

The main idea of the retraction algorithm is to orthogonalize the core tensors DISPLAYFORM10 } in a left-to-right order by the QR decomposition.

The rounding algorithm is shown in Algorithm 3.

Similar to the retraction algorithm, a left-to-right tensor matricization is firstly initialized.

Then, the Tensor-Train rank is updated before conducting an SVD computation.

The returned core tensors DISPLAYFORM11 } are based on the updated vector of lower Tensor-Train ranks.

Furthermore, the rounding procedure has the property presented in Proposition 1.

Proposition 1.

The rounding procedure of Algorithm 3 does not change values of the objective function f for the Tensor-Train RNNs.

That is, for a tensor x ∈ S r , the rounding tensorx ∈ Sr, we have f (x) = f (x).Proof.

Given the weight matrix W with core tensors DISPLAYFORM12 for an input tensor X ∈ M , we obtain (7) and (8) according to (1).

DISPLAYFORM13 DISPLAYFORM14 which suggests that a vector of Tensor-Train ranks determines a submanifold for generating core tensors of the tensor W, but the values of the objective functions are invariant to the change of the vector of Tensor-Train ranks, obtaining Proposition 1.

This section analyzes the convergence of the RSGD algorithm.

The necessary definitions and theorems are firstly introduced, and the analysis is then provided.

Since the objective functions of Tensor-Train RNNs are always geodesically non-convex BID14 , we only consider the convergence of RSGD for non-convex cases.

Definition 2.

A different function f : M → R is geodesically L-smooth if its first-order gradient is geodesically L-Lipchitz continuous.

Specifically, ∀x, y ∈ M we have 9.

DISPLAYFORM0 −1 DISPLAYFORM1 where g x is the sub-gradient of f (x) at x in the tangent space T x M , and Exp −1x (y) is the inverse exponential mapping which projects the curve line in M d from x to y back to the gradient x in the tangent space T x M .From the RSGD algorithm (Algorithm 1), it is not hard to find the sub-gradient g x = ∇f (x) and Exp −1x (y) = −η∇ x f (x), and thus Theorem 3 can be derived.

Theorem 3.

For a differentiable and geodesically L-smooth function f , the Riemannian stochastic gradient descent algorithm ensures (10).

DISPLAYFORM2 where T refers the total iterations, x 0 and x * denote the initial and the optimal points respectively, η is the learning rate, and ||Exp DISPLAYFORM3 Proof.

Assume x * and x 0 separately refer to the optimal and initial points.

For all x t and x t+1 at two consecutive times, we can derive (11) based on Definition 2.

DISPLAYFORM4 By applying Exp DISPLAYFORM5 , we obtain (12), wherex t+1 is the rounding tensor ofx t+1 .

The Proposition 1 ensures that f (x t+1 ) = f (x t+1 ).

DISPLAYFORM6 By summing the two sides of equation 12 from 0 to T − 1, we derive (13).

DISPLAYFORM7 After rounding x * tox * , we finally obtain the result 15 of Theorem 3.

DISPLAYFORM8 Furthermore, Theorem 3 suggests that the number of iterations T satisfies (16) before reaching the convergence.

DISPLAYFORM9

In addition, we also design a Bi-directional Tensor-Train LSTM which involves more operational gates than the Bi-directional Tensor-Train GRU.

The Encoder-Decoder RNNs are commonly used in sequence-to-sequence deep learning applications.

Moreover, the attention mechanism significantly improves the performance of EncoderDecoder RNNs BID11 .The Bi-directional Tensor-Train RNN like GRU or LSTM is used to construct the Encoder-Decoder architecture.

Moreover, we set up the Tensor-Train Attention model in addition to the Tensor-Train Encoder-Decoder RNNs.

Thus, the entire model is built on Tensor-Train layers.

To build a Tensor-Train Attention model, it is necessary to add the Tensor-Train layer to generate an Attention vector as shown in FORMULA3 , where c t is a context vector (26) with attention weights α ts (27), a t is the output of the Attention model at time t,h s denotes a vector built from the outputs of the forward and backward stages, and h t refers to the output of the hidden layers at time t. DISPLAYFORM0 DISPLAYFORM1

This section first introduces the implementation of the Tensor-Tensor RNNs.

Then, we present two applications where the RSGD algorithms were tested.

One application is the digit recognition task on the sequential MNIST dataset; the other is the task of machine translation on the Multi30K dataset BID3 .

We employed PyTorch to implement our Tensor-Train RNNs.

The data structures of our implementations were partly built on the free Tensor-Train toolkit implemented by the tool Tensorflow BID9 .

However, we employed the tool PyTorch to take the advantage of dynamic graph generation, which is much more useful for NLP tasks.

The first application is the digit recognition task on the MNIST dataset.

The dataset consists of 60000 data with 28 * 28 pixels for each digital image.

Instead of vectorizing the image pixels into a long vector as an input for a static deep neural network, image pixels are taken as data sequence where the time step is set to 28, and the input dimension is set to 28.

In our experiments, the training and testing sets were separately composed of 50000 and 10000 data.

2000 data were selected from the training set for building a validation set, and they were not included in the training set.

As for the experimental setup, we applied both Bi-directional Tensor-Train GRU and Bi-directional Tensor-Train LSTM to the task.

The dimension of the Manifold (M, µ) was set to d = 3, and the vector of Tensor-Train ranks was initialized with high and shared values r = (1, 10, 10, 1) for the Tensor-Train layers.

The weight matrix of the input-hidden Tensor-Train layer was converted to the tensor with the shape (2 × 7 × 2) by (6 × 6 × 6), and the weight matrix of the hidden-hidden Tensor-Train layer was converted to the tensor with the shape (6 × 6 × 6) by (6 × 6 × 6).

The RSGD algorithm with a learning rate 0.01 was applied to both Bi-directional Tensor-Train GRU/LSTM.The results are shown in FIG2 , where we compared the Bi-directional Tensor-Train GRU/LSTM with the traditional Bi-directional GRU/LSTM regarding recognition error rates and number of parameters.

Inspecting the recognition error rate, the Bi-directional Tensor-Train GRU obtains a result that is close to that of the traditional Bi-directional GRU/LSTM, and the performance of the Bidirectional Tensor-Train LSTM becomes a bit worse.

Regarding comparison with the number of parameters, both Bi-directional Tensor-Train GRU/LSTM can significantly reduce the number of parameters by taking only 1% parameters of the Bi-directional GRU/LSTM at final.

Notably, the RSGD algorithm further reduces the number of parameters of Bi-directional GRU/LSTM by lowering the Tensor-Train ranks.

The next application is a machine translation task from Dutch to English on the Multi30K dataset.

In the dataset, there are separately 29000, 1014, and 1000 sentence pairs for training data, validation data, and test data, respectively.

The Bi-directional Tensor-Train GRU was used to build the Encoder-Decoder architecture, and a Tensor-Train Attention model was added to the architecture.

The RSGD algorithm with a learning rate 0.01 was applied to the Tensor-Train Encoder-Decoder RNNs with the Tensor-Train Attention model.

An initial vector of Tensor-Train ranks was set as r = (1, 6, 6, 6, 1), and the weight matrix of hidden layers was converted to the tensor with the shape of (4 × 4 × 4 × 4) by (4 × 4 × 4 × 4).

Besides, the baseline was based on the traditional Bi-directional GRU and LSTM-based EncoderDecoder architecture with the Attention model, where the Stochastic Gradient Descent algorithm with learning rate 0.001 was used to update parameters.

The experimental results are shown in FIG3 , and the statistics of parameters of hidden layers are shown in TAB1 .

The results in FIG3 suggest that the Tensor-Train Encoder-Decoder RNN performs closer or even better than the Encoder-Decoder one, although the convergence speed of the Tensor-Train model is relatively slower in the first several iterations.

On the other hand, TAB1 shows that RSGD leads to a further decrease in the number of parameters.

This paper presents the RSGD algorithm for training Tensor-Train RNNs including the related properties, implementations, and convergence analysis.

Our experiments on digit recognition and machine translation tasks suggest that RSGD can work effectively on the Tensor-Train RNNs regarding performance and model complexity, although the convergence speed is relatively slower in the beginning stages.

Our future work will consider two directions: one is to apply the RSGD algorithm to more Tensor-Train models and test it on larger datasets of other fields; and the second one is to generalize Riemannian optimization to the variants of the SGD algorithms and study how to speed up the convergence rate.

@highlight

Applying the Riemannian SGD (RSGD) algorithm for training Tensor-Train RNNs to further reduce model parameters.

@highlight

The paper proposes to use Riemannian stochastic gradient algorithm for low-rank tensor train learning in deep networks.

@highlight

Proposes an algorithm for optimizing neural networks parametrized by Tensor Train decomposition based on the Riemannian optimization and rank adaptation, and designs a bidirectional TT LSTM architecture.