We characterize the singular values of the linear transformation associated with a standard 2D multi-channel convolutional layer, enabling their efficient computation.

This characterization also leads to an algorithm for projecting a convolutional layer onto an operator-norm ball.

We show that this is an effective regularizer;  for example, it improves the test error of a deep residual network using batch normalization on CIFAR-10 from 6.2% to 5.3%.

Convolutional layers BID18 are key components of modern deep networks.

They compute linear transformations of their inputs.

The Jacobian of a linear transformation is always equal to the linear transformation itself.

Because of the central importance of convolutional layers to the practice of deep learning, and the fact that the singular values of the linear transformation computed by a convolutional layer are the key to its contribution to exploding and vanishing gradients, we study these singular values.

Up until now, authors seeking to control the operator norm of convolutional layers have resorted to approximations BID23 BID20 BID9 .

In this paper, we provide an efficient way to compute the singular values exactly -this opens the door to various regularizers.

We consider the convolutional layers commonly applied to image analysis tasks.

The input to a typical layer is a feature map, with multiple channels for each position in an n × n field.

If there are m channels, then the input as a whole is a m × n × n tensor.

The output is also an n × n field with multiple channels per position1.

Each channel of the output is obtained by taking a linear combination of the values of the features in all channels in a local neighborhood centered at the corresponding position in the input feature map.

Crucially, the same linear combination is used for all positions in the feature map.

The coefficients are compiled in the kernel of the convolution.

If the neighborhood is a k × k region, a kernel K is a k × k × m × m tensor.

The projection K :,:,c,: gives the coefficients that determine the cth channel of the output, in terms of the values found in all of the channels of all positions in it neighborhood; K :,:,c,d gives the coefficients to apply to the dth input channel, and K p,q,c,d is the coefficient to apply to this input at in the position in the field offset horizontally by p and vertically by q. For ease of exposition, we assume that feature maps and local neighborhoods are square and that the number of channels in the output is equal to the number of channels in the inputthe extension to the general case is completely straightforward.

To handle edge cases in which the offsets call for inputs that are off the feature maps, practical convolutional layers either do not compute outputs (reducing the size of the feature map), or pad the input with zeros.

The behavior of these layers can be approximated by a layer that treats the input as if it were a torus; when the offset calls for a pixel that is off the right end of the image, the layer "wraps around" to take it from the left edge, and similarly for the other edges.

The quality of this approximation has been heavily analyzed in the case of one-dimensional signals BID11 .

Consequently, theoretical analysis of convolutions that wrap around has been become standard.

This is the case analyzed in this paper.

Our main result is a characterization of the singular values of a convolutional layer in terms of the kernel tensor K. Our characterization enables these singular values to be computed exactly in a simple and practically fast way, using O(n 2 m 2 (m + log n)) time.

For comparison, the brute force solution that performs SVD on the matrix that encodes the convolutional layer's linear transformation would take O((n 2 m) 3 ) = O(n 6 m 3 ) time, and is impractical for commonly used network sizes.

As another point of comparison, simply to compute the convolution takes O(n 2 m 2 k 2 ) time.

We prove that the following two lines of NumPy correctly compute the singular values.

Here kernel is any k × k × m × m tensor2 and input_shape is the shape of the feature map to be convolved.

A TensorFlow implementation is similarly simple.

Timing tests, reported in Section 4.1, confirm that this characterization speeds up the computation of singular values by multiple orders of magnitude -making it usable in practice.

The algorithm first performs m 2 FFTs, and then it performs n 2 SVDs.

The FFTs, and then the SVDs, may be executed in parallel.

Our TensorFlow implementation runs a lot faster than the NumPy implementation (see Figure 1) ; we think that this parallelism is the cause.

We used our code to compute the singular values of the convolutional layers of the official ResNet-v2 model released with TensorFlow BID12 ).

The results are described in Appendix C.Exposing the singular values of a convolutional layer opens the door to a variety of regularizers for these layers, including operator-norm regularizers.

In Section 4.2, we evaluate an algorithm that periodically projects each convolutional layer onto a operator-norm ball.

Using the projections improves the test error from 6.2% to 5.3% on CIFAR-10.

We evaluate bounding the operator norm with and without batchnorm and we see that regularizing the operator norm helps, even in the presence of batch normalization.

Moreover, operator-norm regularization and batch normalization are not redundant, and neither dominates the other.

They complement each other.

Related work: Prior to our work, authors have responded to the difficulty of computing the singular values of convolutional layers in various ways.

BID6 constrained the matrix to have orthogonal rows and scale the output of each layer by a factor of (2k + 1) BID9 proposed regularizing using a per-mini-batch approximation to the operator norm.

They find the largest ratio between the input and output of a layer in the minibatch, and then scale down the transformation (thereby scaling down all of the singular values, not just the largest ones) so that the new value of this ratio obeys a constraint.

BID23 used an approximation of the operator norm of a reshaping of K in place of the operator norm for the linear transformation associated with K in their experiments.

They reshape the given k × k × m × m into a mk 2 × m matrix, and compute its largest singular value using a power iteration method, and use this as a substitute for the operator norm.

While this provides 2The same code also works if the filter height is different from the filter width, and if the number of channels in the input is different from the number of channels of the output.

a useful heuristic for regularization, the largest singular value of the reshaped matrix is often quite different from the operator norm of the linear transform associated with K. Furthermore if we want to regularize using projection onto an operator-norm ball, we need the whole spectrum of the linear transformation (see Section 3).

The reshaped K has only m singular values, whereas the linear transformation has mn 2 singular values of which mn 2 /2 are distinct except in rare degenerate cases.

It is possible to project the reshaped K onto an operator-norm ball by taking its SVD and clipping its singular values -we conducted experiments with this projection and report the results in Section 4.4.

DISPLAYFORM0 A close relative of our main result was independently discovered by Bibi et al. (2019, Lemma 2) .

If the signal is 1D and there is a single input and output channel, then the linear transformation associated with a convolution is encoded by a circulant matrix, i.e., a matrix whose rows are circular shifts of a single row BID11 .

For example, for a row a = (a 1 , a 2 , a 3 ), the circulant matrix circ(a) generated by a is a 0 a 1 a 2 a 2 a 0 a 1 a 1 a 2 a 0 .

In the special case of a 2D signal with a single input channel and single output channel, the linear transformation is doubly block circulant (see BID8 ).

Such a matrix is made up of a circulant matrix of blocks, each of which in turn is itself circulant.

Finally, when there are m input channels and m output channels, there are three levels to the hierarchy: there is a m × m matrix of blocks, each of which is doubly block circulant.

Our analysis extends tools from the literature built for circulant BID16 and doubly circulant BID4 matrices to analyze the matrices with a third level in the hierarchy arising from the convolutional layers used in deep learning.

One key point is that the eigenvectors of a circulant matrix are Fourier basis vectors: in the 2D, one-channel case, the matrix whose columns are the eigenvectors is F ⊗ F , for the matrix F whose columns form the Fourier basis.

Multiplying by this matrix is a 2D Fourier transform.

In the multi-channel case, we show that the singular values can be computed by (a) finding the eigenvalues of each of the m 2 doubly circulant matrices (of dimensions n 2 × n 2 ) using a 2D Fourier transform, (b) by forming multiple m × m matrices, one for each eigenvalue, by picking out the i-th eigenvalue of each of the n 2 × n 2 blocks, DISPLAYFORM0 The union of all of the singular values of all of those m × m matrices is the multiset of singular values of the layer.

We use upper case letters for matrices, lower case for vectors.

For matrix M , M i,: represents the i-th row and M :,j represents the j-th column; we will also use the analogous notation for higher-order tensors.

The operator norm of M is denoted by ||M || 2 .

For n ∈ N, we use [n] to denote the set {0, 1, . . .

, n − 1} (instead of usual {1, . . . , n}).

We will index the rows and columns of matrices using elements of [n], i.e. numbering from 0.

Addition of row and column indices will be done mod n unless otherwise indicated.

(Tensors will be treated analogously.) Let σ(·) be the mapping from a matrix to (the multiset of) its singular values.

3Let ω = exp(2πi/n), where i = √ −1. (Because we need a lot of indices in this paper, our use of i to define ω is the only place in the paper where we will use i to denote √ −1.)Let F be the n × n matrix that represents the discrete Fourier transform: DISPLAYFORM0 We use I n to denote the identity matrix of size n × n. For i ∈ [n], we use e i to represent the ith basis vector in R n .

We use ⊗ to represent the Kronecker product between two matrices (which also refers to the outer product of two vectors).

As a warmup, we focus on the case that the number m of input channels and output channels is 1.

In this case, the filter coefficients are simply a k × k matrix.

It will simplify notation, however, if we embed this k × k matrix in an n × n matrix, by padding with zeroes (which corresponds to the fact that the offsets with those indices are not used).

Let us refer to this n × n matrix also as K.An n 2 × n 2 matrix A is doubly block circulant if A is a circulant matrix of n × n blocks that are in turn circulant.

DISPLAYFORM0 That is, if X is an n × n matrix, and Y is the result of a 2-d convolution of X with K, i.e. DISPLAYFORM1 then vec(Y ) = A vec(X).So now we want to determine the singular values of a doubly block circulant matrix.

We will make use of the characterization of the eigenvalues and eigenvectors of doubly block circulant matrices, which uses the following definition: DISPLAYFORM2 Theorem 2 (Jain (1989) Section 5.5) For any n 2 × n 2 doubly block circulant matrix A, the eigenvectors of A are the columns of Q.To get singular values in addition to eigenvalues, we need the following two lemmas.

Using Theorem 2 and Lemma 3, we can get the eigenvalues as the diagonal elements of Q * AQ.

Proof: DISPLAYFORM0 The following theorem characterizes the singular values of A as a simple function of K. As we will see, a characterization of the eigenvalues plays a major role.

BID4 provided a more technical characterization of the eigenvalues which may be regarded as making partial progress toward Theorem 5.

However, we provide a proof from first principles, since it is the cleanest way we know to prove the theorem.

Proof:

By Theorems 2 and Lemma 3, the eigenvalues of A are the diagonal elements of DISPLAYFORM0 as a compound n × n matrix of n × n blocks, for u, v ∈ [n], the (un + v)th diagonal element is the vth element of the uth diagonal block.

Let us first evaluate the uth diagonal block.

Using i, j to index blocks, we have DISPLAYFORM1 To get the vth element of the diagonal of (4), we may sum the vth elements of the diagonals of each of its terms.

Toward this end, we have DISPLAYFORM2 Substituting into (4), we get DISPLAYFORM3 DISPLAYFORM4 s−r .

Collecting terms where j − i = p and sSince the singular values of any normal matrix are the magnitudes of its eigenvalues BID16 page 158), applying Lemma 4 completes the proof.

Note that F T KF is the 2D Fourier transform of K, and recall that ||A|| 2 is the largest singular value of A.

Now, we consider case where the number m of channels may be more than one.

Assume we have a 4D kernel tensor K with element K p,q,c,d giving the connection strength between a unit in channel d of the input and a unit in channel c of the output, with an offset of p rows and q columns between the input unit and the output unit.

The input X ∈ R m×n×n ; element X d,i,j is the value of the input unit within channel d at row i and column j.

The output Y ∈ R m×n×n has the same format as X, and is produced by The following is our main result.

DISPLAYFORM0

For any K ∈ R n×n×m×m , let M is the matrix encoding the linear transformation computed by a convolutional layer parameterized by K, defined as in (6) DISPLAYFORM0 The rest of this section is devoted to proving Theorem 6 through a series of lemmas.

The analysis of Section 2.1 implies that for all c, d DISPLAYFORM1 Lemma FORMULA15 DISPLAYFORM2 this implies that M and L have the same singular values.

So now we have as a subproblem characterizing the singular values of a block matrix whose blocks are diagonal.

To express the the characterization, it helps to reshape the nonzero elements of L into a m × m × n 2 tensor G as follows: DISPLAYFORM3 σ (G :,:,w ) .

Choose an arbitrary w ∈ [n 2 ], and a (scalar) singular value σ of G :,:,w whose left singular vector is x and whose right singular vector is y, so that G :,:,w y = σx.

Recall that e w ∈ R n 2 is a standard basis vector.

Ifσ is another singular value of G :,:,w with a left singular vectorx and a right singular vector y, then (x ⊗ e w ), (x ⊗ e w ) = x,x = 0 and, similarly (y ⊗ e w ), (ỹ ⊗ e w ) = 0.

Also, (x ⊗ e w ), (x ⊗ e w ) = 1 and (y ⊗ e w ), (y ⊗ e w ) = 1.

DISPLAYFORM0 For any x andx, whether they are equal or not, if w =w, then (x ⊗ e w ), (x ⊗ ew) = 0, simply because their non-zero components do not overlap.

Thus, by taking the Kronecker product of each singular vector of G :,:,w with e w and assembling the results for various w, we may form a singular value decomposition of L whose singular values are ∪ w∈[n 2 ] σ(G :,:,w ).

This completes the proof.

Using Lemmas 7 and Theorem 8, we are now ready to prove Theorem 6.

Applying Lemmas 7 and 8 completes the proof.

We now show how to use the spectrum computed above to project a convolution onto the set of convolutions with bounded operator norm.

We exploit the following key fact.

This implies that the desired projection can be obtained by clipping the singular values of linear transformation associated with a convolutional layer to the interval [0, c] .

Note that the eigenvectors remained the same in the proposition, hence the projected matrix is still generated by a convolution.

However, after the projection, the resulting convolution neighborhood may become as large as n × n. On the other hand, we can project this convolution onto the set of convolutions with k × k neighborhoods, by zeroing out all other coefficients.

NumPy code for this is in Appendix A.Repeatedly alternating the two projections would give a point in the intersection of the two sets, i.e., a k × k convolution with bounded operator norm BID5 Theorem 4, BID2 Section 2), and the projection onto that intersection could be found using the more complicated Dykstra's projection algorithm BID3 ).When we wish to control the operator norm during an iterative optimization process, however, repeating the alternating projections does not seem to be worth it -we found that the first two projections already often produced a convolutional layer with an operator norm close to the desired value.

Furthermore, because SGD does not change the parameters very fast, we can think of a given pair of projections as providing a warm start for the next pair.

In practice, we run the two projections once every few steps, thus letting the projection alternate with the training.

First, we validated Theorem 6 with unit tests in which the output of the code given in the introduction is compared with evaluating the singular values by constructing the full matrix encoding the linear transformation corresponding to the convolutional layer and computing its SVD.

We generated 4D tensors of various shapes with random standard normal values, and computed their singular values using the full matrix method, the NumPy code given above and the equivalent TensorFlow code.

For small tensors, the NumPy code was faster than TensorFlow, but for larger tensors, the TensorFlow code was able to exploit the parallelism in the algorithm and run much faster on a GPU.

The timing results are shown in Figure 1.

We next explored the effect of regularizing the convolutional layers by clipping their operator norms as described in Section 3.

We ran the CIFAR-10 benchmark with a standard 32 layer residual network with 2.4M training parameters; BID12 .

This network reached a test error rate of 6.2% after 250 epochs, using a learning rate schedule determined by a grid search (shown by the gray plot in Figure 2) .

We then evaluated an algorithm that, every 100 steps, clipped the norms of the convolutional layers to various different values between 0.1 and 3.0.

As expected, clipping to 2.5 and 3.0 had little impact on the performance, since the norms of the convolutional layers were between 2.5 and 2.8.

Clipping to 0.1 yielded a surprising 6.7% test error, whereas clipping to 0.5 and 1.0 yielded test errors of 5.3% and 5.5% respectively (shown in Figure 2) .

A plot of test error against training time is provided in Figure 4 in Appendix B, showing that the projections did not slow down the training very much.

BID12 for CIFAR-10.

The baseline algorithm studied in the previous subsection used batch normalization.

Batch normalization tends to make the network less sensitive to linear transformations with large operator norms.

However, batch normalization includes trainable scaling parameters (called γ in the original paper) that are applied after the normalization step.

The existence of these parameters lead to a complicated interaction between batch normalization and methods like ours that act to control the norm of the linear transformation applied before batch normalization.

Because the effect of regularizing the operator norm is more easily understood in the absence of batch normalization, we also performed experiments with a baseline that did not use batch normalization.

Another possibility that we wanted to study was that using a regularizer may make the process overall more stable, enabling a larger learning rate.

We were generally interested in whether operator-norm regularization made the training process more robust to the choice of hyperparameters.

In one experiment, we started with the same baseline as the previous subsection, but disabled batch normalization.

This baseline started with a learning rate of 0.1, which was multiplied by a factor 0.95 after every epoch.

We tried all combinations of the following hyperparameters: (a) the norm of the ball projected onto (no projection, 0. improved the best result, and also made the process more robust to the choice of hyperparameters.

We conducted a similar experiment in the presence of batch normalization, except using learning rates 0.01, 0.03, 0.1, 0.2, and 0.3.

Those results are shown in Figure 3b .

Regularizing the operator norm helps, even in the presence of batch normalization.

It appears that operator-norm regularization and batch normalization are not redundant, and neither dominates the other.

We were surprised by this.

In Section 1 we mentioned that BID23 approximated the linear transformation induced by K by reshaping K. This leads to an alternate regularization method -we compute the spectrum of the reshaped K, and project it onto a ball using clipping, as above.

We implemented this an experimented with it using the same network and hyperparameters as in Section 4.2 and found the following.• We clipped the singular values of the reshaped K every 100 steps.

We tried various constants for the clipped value (0.05, 0.1, 0.2, 0.5, 1.0), and found that the best accuracy we achieved, using 0.2, was the same as the accuracy we achieved in Section 4.2.•

We clipped the singular values of the reshaped K to these same values every step, and found that the best accuracy achieved was slightly worse than the accuracy achieved in the previous step.

We observed similar behavior when we clipped norms using our method.• Most surprisingly, we found that clipping norms by our method on a GPU was about 25% faster than clipping the singular values of the reshaped K -when we clipped after every step, on the same machine, 10000 batches of CIFAR10 took 14490 seconds when we clipped the reshaped K, whereas they took 11004 seconds with our exact method!

One possible explanation is parallelization -clipping reshaped K takes O(m 3 k 2 ) flops, whereas our method does m 2 FFTs, followed by n 2 m × m SVDs, which takes O(m 3 n 2 ) flops, but these can be parallelized and completed in as little as O(n 2 log n + m 3 ) time.

Clearly this is only one dataset, and the results may not generalize to other sets.

However it does suggest that finding the full spectrum of the convolutional layer may be no worse than computing heuristic approximations, both in classification accuracy and speed.

B T .

Figure 4 shows the plots of test error vs. training time in our CIFAR-10 experiment.

The singular values of the convolutional layers from the official "Resnet V2" pre-trained model BID12 are plotted in Figure 5 .

The singular values are ordered by value.

Only layers with kernels larger than 1 × 1 are plotted.

The curves are plotted with a mixture of red and green; layers closer to the input are plotted with colors with a greater share of red.

The transformations with the largest operator norms are closest to the inputs.

As the data has undergone more rounds of processing, as we proceed through the layers, the number of non-negligible singular values increases for a while, but at the end, it tapers off.

In Figure 5 , we plotted the singular values ordered by value.

It can be observed that while singular values in the first layer are much larger than the rest, many layers have a lot of singular values that are pretty big.

For example, most of the layers have at least 10000 singular values that are at least 1.

To give a complementary view, FIG9 presents a plot of the ratios of the singular values in each layer with the largest singular value in that layer.

We see that the effective rank of the convolutional layers is larger closer to the inputs.

FIG9 shows that different convolutional layers have significantly different numbers of non-negligible singular values.

A question that may arise is to what extent this was due to the fact that different layers simply are of different sizes, so that the total number of their singular values, tiny or not, was different.

To look into this, instead of plotting the singular value ratios as a function of the rank of the singular values, as in the FIG9 , we normalized the values on the horizontal axis by dividing by the total number of singular values.

The result is shown in FIG10 .

<|TLDR|>

@highlight

We characterize the singular values of the linear transformation associated with a standard 2D multi-channel convolutional layer, enabling their efficient computation. 

@highlight

The paper is dedicated to computation of singular values of convolutional layers

@highlight

Derives exact formulas for computing singular values of convolutional layers of deep neural networks and show that computing the singular values can be done much faster than computing the full SVD of the convolution matrix by appealing to fast FFT transformations.