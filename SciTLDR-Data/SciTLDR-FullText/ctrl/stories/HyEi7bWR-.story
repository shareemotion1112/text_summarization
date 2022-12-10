Recurrent Neural Networks (RNNs) are designed to handle sequential data but suffer from vanishing or exploding gradients.

Recent work on Unitary Recurrent Neural Networks (uRNNs) have been used to address this issue and in some cases, exceed the capabilities of Long Short-Term Memory networks (LSTMs).

We propose a simpler and novel update scheme to maintain orthogonal recurrent weight matrices without using complex valued matrices.

This is done by parametrizing with a skew-symmetric matrix using the Cayley transform.

Such a parametrization is unable to represent matrices with negative one eigenvalues, but this limitation is overcome by scaling the recurrent weight matrix by a diagonal matrix consisting of ones and negative ones.

The proposed training scheme involves a straightforward gradient calculation and update step.

In several experiments, the proposed scaled Cayley orthogonal recurrent neural network (scoRNN) achieves superior results with fewer trainable parameters than other unitary RNNs.

Deep neural networks have been used to solve numerical problems of varying complexity.

RNNs have parameters that are reused at each time step of a sequential data point and have achieved state of the art performance on many sequential learning tasks.

Nearly all optimization algorithms for neural networks involve some variant of gradient descent.

One major obstacle to training RNNs with gradient descent is due to vanishing or exploding gradients, as described in BID1 and BID14 .

This problem refers to the tendency of gradients to grow or decay exponentially in size, resulting in gradient descent steps that are too small to be effective or so large that the network oversteps the local minimum.

This issue significantly diminishes RNNs' ability to learn time-based dependencies, particularly in problems with long input sequences.

A variety of architectures have been introduced to overcome this difficulty.

The current preferred RNN architectures are those that introduce gating mechanisms to control when information is retained or discarded, such as LSTMs BID6 and GRUs BID3 , at the cost of additional trainable parameters.

More recently, the unitary evolution RNN (uRNN) BID0 ) uses a parametrization that forces the recurrent weight matrix to remain unitary throughout training, and exhibits superior performance to LSTMs on a variety of synthetic and real-world tasks.

For clarity, we follow the convention of BID21 and refer to this network as the restricted-capacity uRNN.Since the introduction of uRNNs, orthogonal and unitary RNN schemes have increased in both popularity and complexity.

BID21 use a multiplicative update method detailed in Tagare (2011) and BID20 to expand uRNNs' capacity to include all unitary matrices.

These networks are referred to as full-capacity uRNNs.

BID7 's EURNN parametrizes this same space with Givens rotations, while BID8 's GORU introduces a gating mechanism for unitary RNNs to enable short term memory.

BID19 introduced modified optimization and regularization methods that restrict singular values of the recurrent matrix to an interval around 1.

Each of these methods involve complex valued recurrent weights.

For other work in addressing the vanishing and exploding gradient problem, see BID5 and BID11 .In this paper, we consider RNNs with a recurrent weight matrix taken from the set of all orthogonal matrices.

To construct the orthognal weight matrix, we parametrize it with a skew-symmetric matrix through a scaled Cayley transform.

This scaling allows us to avoid the singularity issue occuring for −1 eigenvalues that may arise in the standard Cayley transform.

With the parameterization, the network optimization involves a relatively simple gradient descent update.

The resulting method achieves superior performance on sequential data tasks with a smaller number of trainable parameters and hidden sizes than other unitary RNNs and LSTMs.

The method we present in this paper works entirely with real matrices, and as such, our results deal only with orthogonal and skew-symmetric matrices.

However, the method and all related theory remain valid for unitary and skew-Hermitian matrices in the complex case.

The experimental results in this paper indicate that state of the art performance can be achieved without the increased complexity of optimization along the Stiefel manifold and using complex matrices.

A recurrent neural network (RNN) is a function with input parameters U ∈ R n×m , recurrent parameters W ∈ R n×n , recurrent bias b ∈ R n , output parameters V ∈ R p×n , and output bias c ∈ R p where m is the data input size, n is the number of hidden units, and p is the output data size.

From an input sequence x = (x 1 , x 2 , ..., x T ) where x i ∈ R m , the RNN returns an output sequence y = (y 1 , y 2 , ..., y T ) where each y i ∈ R p is given recursively by DISPLAYFORM0 where h = (h 0 , . . .

, h T −1 ), h i ∈ R n is the hidden layer state at time i and σ(·) is the activation function, which is often a pointwise nonlinearity such as a hyperbolic tangent function or rectified linear unit BID13 .

A real matrix W is orthogonal if it satisfies W T W = I. The complex analog of orthogonal matrices are unitary matrices, which satisfy W * W = I, where * denotes the conjugate transpose.

Orthogonal and unitary matrices have the desirable property that W x 2 = x 2 for any vector x. This property motivates the use of orthogonal or unitary matrices in RNNs to avoid vanishing and exploding gradients, as detailed in BID0 .

BID0 follow the framework of the previous section for their restricted-capacity uRNN, but introduce a parametrization of the recurrent matrix W using a product of simpler matrices.

This parameterization is given by a product consisting of diagonal matrices with complex norm 1, complex Householder reflection matrices, discrete Fourier transform matrices, and a fixed permutation matrix with the resulting product being unitary.

BID21 note that this representation has only 7n parameters, which is insufficient to represent all unitary matrices for n > 7.

In response, they present the full-capacity uRNN, which uses a multiplicative update step that is able to reach all unitary matrices of order n.

The full-capacity uRNN aims to construct a unitary matrix W (k+1) from W (k) by moving along a curve on the Stiefel manifold {W ∈ C n×n | W * W = I}. For the network optimization, it is necessary to use a curve that is in a descent direction of the cost function L := L(W ).

In Tagare (2011), BID20 , and BID21 , a descent direction is constructed as DISPLAYFORM0 , which is a representation of the derivative operator DL(W (k) ) in the tangent space of the Stiefel manifold at W (k) .

Then, with B (k) W (k) defining the direction of a descent curve, an update along the Stiefel manifold is obtained as DISPLAYFORM1 where λ is the learning rate.3 SCALED CAYLEY ORTHOGONAL RNN

The Cayley transform gives a bijection between the set of orthogonal matrices without −1 eigenvalues and the set of skew-symmetric matrices (i.e., matrices where A T = −A): DISPLAYFORM0 We can use this bijection to parametrize the set of orthogonal matrices without −1 eigenvalues using skew-symmetric matrices.

This parametrization is attractive from a machine learning perspective because it is closed under addition: the sum or difference of two skew-symmetric matrices is also skew-symmetric, so we can use gradient descent algorithms like RMSprop BID18 or Adam BID10 to train parameters.

However, this parametrization cannot represent orthogonal matrices with −1 eigenvalues, since in this case I + W , is not invertible.

Theoretically, we can still represent matrices with eigenvalues that are arbitrarily close to −1; however, it can require large entries of A. For example, a 2x2 orthogonal matrix W with eigenvalues ≈ −0.99999 ± 0.00447i and its parametrization A by the Cayley transform is given below.

DISPLAYFORM1 Gradient descent algorithms will learn this A matrix very slowly, if at all.

This difficulty can be overcome through a suitable diagonal scaling according to results from BID9 .Theorem 3.1 Every orthogonal matrix W can be expressed as DISPLAYFORM2 where A = [a ij ] is real-valued, skew-symmetric with |a ij | ≤ 1, and D is diagonal with all nonzero entries equal to ±1.We call the transform in Theorem 3.1 the scaled Cayley transform.

Then, with an appropriate choice of D, the scaled Cayley transform can reach any orthogonal matrix including those with −1 eigenvalues.

Further, it ensures that the skew-symmetric matrix A that generates the orthogonal matrix will be bounded.

Our proposed network, the scaled Cayley orthogonal recurrent neural network (scoRNN), is based on this theorem.

We parametrize the recurrent weight matrix W through a skew-symmetric matrix A, which results in n(n−1) 2 trainable weights.

The recurrent matrix W is formed by the scaled Cayley transform: DISPLAYFORM3 The scoRNN then operates identically to the set of equations given in Section 2.1, but during training we update the skew-symmetric matrix A using gradient descent, while D is fixed throughout the training process.

The number of −1s on the diagonal of D, which we call ρ, is considered a hyperparameter in this work and is manually chosen based on the task.

To update the recurrent parameter matrix A as described in Section 3.1, we must find the gradients of A by backpropagating through the Cayley transform.

The following theorem describes these gradients.

A proof is given in Appendix A. DISPLAYFORM0 where V := (I + A) DISPLAYFORM1 At each training step of scoRNN, we first use the standard backpropagation algorithm to compute ∂L ∂W and then use Theorem 3.2 to compute ∂L ∂A .

We then update A with gradient descent (or a related optimization method), and reconstruct W as follows: DISPLAYFORM2 The skew-symmetry of ∂L ∂A ensures that A (k+1) will be skew-symmetric and, in turn, W (k+1) will be orthogonal.

The scoRNN and the full-capacity uRNN from Section 2.2 both have the capacity to optimize an orthogonal or unitary recurrent matrix W , but they use different update schemes.

The full-capacity uRNN performs a multiplicative update that moves W along the tangent space of the Stiefel manifold, which can be shown to be a descent direction, but not necessarily the steepest one.

In contrast, scoRNN performs an additive update in the direction of steepest descent with respect to its parametrization.

The scoRNN update proves to be much more resistant to loss of orthogonality during training; see Appendix B.

It also maintains stable hidden state gradients in the sense that the gradient norm does not change significantly in time; see Appendix C for experimental results.

This is achieved with very little overhead computational costs over the standard RNN; see Appendix D for experiments comparing computational speeds.

The basic architecture of scoRNN is very similar to the standard RNN as presented in Section 2.1.

From a network layer perspective, one can think of the application of the recurrent weight in a three layer process.

Let h t ∈ R n be the current state of the scoRNN at a particular time step, t. We then pass h t through the following layers:• Layer 1: DISPLAYFORM0 Note that the above scheme is the same as taking h t → W h t as discussed previously.

The modReLU function was first implemented by BID0 to handle complex valued functions and weights.

Unlike previous methods, our method only uses real-valued functions and weights.

Nevertheless, we have found that the modReLU function in the real case also performed better than other activation functions.

The function is defined as DISPLAYFORM0 where b is a trainable bias.

In the real case, this simplifies to sign(z)σ ReLU (|z| + b).

To implement this activation function in scoRNN, we replace the computation of h t in Section 2.1 with DISPLAYFORM1 We believe that the improved performance of the modReLU over other activation functions, such as ReLU, is because it admits both positive and negative activation values, which appears to be important for the state transition in orthogonal RNNs.

This is similar to the hyperbolic tangent function but does not have vanishing gradient issues.

Modifying the initialization of our parameter matrices, in particular our recurrent parameter matrix A, had a significant effect on performance.

The most effective initialization method we found uses a technique inspired by BID5 .

We initialize all of the entries of A to be 0 except for 2x2 blocks along the diagonal, which are given as DISPLAYFORM0 1+cos (tj ) and t j is sampled uniformly from 0, π 2 .

The Cayley transform of this A will have eigenvalues equal to ±e itj for each j, which will be distributed uniformly along the right unit half-circle.

Multiplication by the scaling matrix D will reflect ρ of these eigenvalues across the imaginary axis.

We use this method to initialize scoRNN's A matrix in all of the experiments listed in section 5.

For each experiment, we found optimal hyperparameters for scoRNN using a grid search.

For other models, we used the best hyperparameters settings as reported in BID21 and BID0 .

If not available, we performed a grid search to find the best hyperparameters.

This experiment follows descriptions found in BID0 and BID21 , and tests an RNN's ability to reproduce a sequence seen many timesteps earlier.

In the problem setup, there are 10 input classes, which we denote using the digits 0-9, with 0 being used as a 'blank' class and 9 being used as a 'marker' class.

The RNN receives an input sequence of length T + 20.

This sequence consists of entirely zeros, except for the first ten elements, which are uniformly sampled from classes 1-8, and a 9 placed ten timesteps from the end.

The goal for the machine is to output zeros until it sees a 9, at which point it should output the ten elements from the beginning of the input sequence.

Thus, information must propagate from the beginning to the end of the sequence for a machine to successfully learn this task, making it critical to avoid vanishing/exploding gradients.

A baseline strategy with which to compare machine performance is that of outputting 0 until the machine sees a 9, and then outputting 10 elements randomly sampled from classes 1-8.

The expected cross-entropy for such a strategy is 10 log (8) T +20 .

In practice, it is common to see gated RNNs such as LSTMs converge to this local minimum.

We vary the number of hidden units of the machines to match the number of parameters, approximately 22k each.

This results in an LSTM with n = 68, a restricted-capacity uRNN with n = 470, a full-capacity uRNN with n = 128, and a scoRNN with n = 190.

We found the best performance with the scoRNN came from ρ = n/2, which gives an initial W with eigenvalues distributed uniformly on the entire unit circle.

FIG1 compares each model's performance for T = 1000 and T = 2000, with the baseline cross-entropy given as a dashed line.

In both cases, cross-entropy for the restricted-capacity uRNN and LSTM never drop below the baseline.

For the T = 1000 test, the full-capacity uRNN and scoRNN converge immediately to zero entropy solutions, with the full-capacity uRNN converging slightly faster.

For T = 2000, the full-capacity uRNN remains at the baseline for several thousand iterations, but is eventually able to find a correct solution.

In contrast, the scoRNN error has a smooth convergence that bypasses the baseline, but does so more slowly than the full-capacity uRNN.

We examined a variation of the adding problem as proposed by BID0 which is based on the work of BID6 .

This variation involves passing two sequences concurrently into the RNN, each of length T .

The first sequence is a sequence of digits sampled uniformly with values ranging in a half-open interval, U[0, 1).

The second sequence is a marker sequence consisting of all zeros except for two entries that are marked by one.

The first 1 is located uniformly within the interval [1, T 2 ) of the sequence and the second 1 is located uniformly within the interval [ T 2 , T ) of the sequence.

The label for each pair of sequences is the sum of the two entries that are marked by one, which forces the machine to identify relevant information in the first sequence among noise.

As the sequence length increases, it becomes more crucial to avoid vanishing/exploding gradients.

Naively predicting one regardless of the sequence gives an expected mean squared error (MSE) of approximately 0.167.

This will be considered as the baseline.

The number of hidden units for each network was adjusted so that each had approximately 14k trainable parameters.

This results in n = 170 for the scoRNN, n = 60 for the LSTM, n = 120 for the Full-Capacity uRNN, and n = 950 hidden units for the restricted-capacity uRNN.

The test set MSE results for sequence lengths T = 200, T = 400, and T = 750 can be found in FIG0 .

A training set size of 100,000 and a testing set size of 10,000 were used for each sequence length.

For each case, the networks start at or near the baseline MSE and drop towards zero after a few epochs.

As the sequence length increases, the number of epochs before the drop increases.

We found the best settings for the scoRNN were ρ = n/2 for T = 200 and ρ = 7n/10 for T = 400 and T = 750.

As can be seen, the LSTM error drops precipitously across the board before all other models, while the unitary and orthogonal RNNs descend more gradually.

Although in some cases the full-capacity uRNN begins to drop below the baseline before scoRNN, the full-capacity uRNN does not drop as quickly and has a more irregular descent curve.

We ran two experiments based around classifying samples from the well-known MNIST dataset (LeCun et al.) .

Following the implementation of BID11 , each pixel of the image is fed into the RNN sequentially, resulting in a single pixel sequence length of 784.

In the first experiment, which we refer to as unpermuted MNIST, pixels are arranged in the sequence row-by-row.

In the second, which we call permuted MNIST, a fixed permutation is applied to training and testing sequences.

All scoRNN machines were trained with the RMSProp optimization algorithm.

Input and output weights used a learning rate of 10 −3 , while the recurrent parameters used a learning rate of 10 −4 (for n = 170) or 10 −5 (for n = 360 and n = 512).

For unpermuted MNIST, we found ρ to be optimal at n/10, while the best value of ρ for permuted MNIST was n/2.

We suspect that the difference of these two values comes from the different types of dependencies in each: unpermuted MNIST has mostly local dependencies, while permuted MNIST requires learning many long-term dependencies, which appear to be more easily modeled when the diagonal of D has a higher proportion of −1s.

Each experiment used a training set of 55,000 images and a test set of 10,000 testing images.

Each machine was trained for 70 epochs, and test set accuracy, the percentage of test images classified correctly, was evaluated at the conclusion of each epoch.

FIG3 shows test set accuracy over time for each machine, and the best performance over all epochs by each machine is given in TAB0 .

In both experiments, the 170 hidden unit scoRNN gives similar performance to both of the 512 hidden unit uRNNs using a much smaller hidden dimension and, in the case of the full-capacity uRNN, an order of magnitude fewer parameters.

Matching the number of parameters (≈ 69k), the 2170 restricted-capacity uRNN performance was comparable to the 360 hidden unit scoRNN for unpermuted MNIST, but performed worse for permuted MNIST, and required a much larger hidden size and a significantly longer run time, see Appendix D. As in experiments presented in BID0 and BID21 , orthogonal and unitary RNNs are unable to outperform the LSTM in the unpermuted case.

However, the 360 and 512 hidden unit scoRNNs outperform the unitary RNNs.

On permuted MNIST, the 512 hidden unit scoRNN achieves a test-set accuracy of 96.6%, outperforming all of the uRNNs and LSTMs.

We believe this is a state of the art result.

To see how the models performed on audio data, speech prediction was performed on the TIMIT dataset BID4 , a collection of real-world speech recordings.

Excluding the dialect SA sentences and using only the core test set, the dataset consisted of 3,696 training and 192 testing audio files.

Similar to experiments in BID21 , audio files were downsampled to 8kHz and a short-time Fourier transform (STFT) was applied with a Hann window of 256 samples and a window hop of 128 samples (16 milliseconds).

The result is a set of frames, each with 129 complexvalued Fourier amplitudes.

The log-magnitude of these amplitudes is used as the input data for the machines.

Each frame was fed into the machine sequentially, and at each time step t, the machine's target output is to predict the t + 1 frame.

For each model, the hidden layer sizes were adjusted such that each model had approximately equal numbers of trainable parameters.

For scoRNN, we used the Adam optimizer with learning rate 10 −3 to train the input and output parameters, and RMSprop with a learning rate of 10 −3 (for n = 224) or 10 −4 (for n = 322 and n = 425) to train the recurrent weight matrix.

The number of negative eigenvalues used was ρ = n/10.The loss function used for training was the mean squared error (MSE) between the predicted and actual log-magnitudes of the next time frame over the entire sequence.

TAB1 contains the MSE on validation and testing sets, which shows that all scoRNN models achieve a smaller MSE than all LSTM and unitary RNN models.

Similar to BID21 , we reconstructed audio files using the predicted log-magnitudes from each machine and evaluated them on several audio metrics.

We found that the scoRNN predictions achieved better scores on the signal-to-noise ratio metric SegSNR BID2 , but performed slightly worse than the full-capacity uRNN predictions on the human intelligibility and perception metrics STOI BID16 PESQ (Rix et al., 2001 ).

DISPLAYFORM0 Using the identity (I + A) Z = I − A and taking the derivative with respect to A i,j to both sides we obtain: DISPLAYFORM1 and rearranging we get: DISPLAYFORM2 Let E i,j denote the matrix whose (i, j) entry is 1 with all others being 0.

Since A is skew-symmetric, we have ∂A ∂Ai,j = E i,j − E j,i .

Combining everything, we have: DISPLAYFORM3 Using the above formulation,

In the scoRNN architecture, the recurrent weight matrix is parameterized with a skew-symmetric matrix through the Cayley transform.

This ensures the computed recurrent weight matrix in floating point arithmetic is orthogonal to the order of machine precision after each update step.

Unlike scoRNN, the full-capacity uRNN maintains a unitary recurrent weight matrix by a multiplicative update scheme.

Due to the accumulation of rounding errors over a large number of repeated matrix multiplications, the recurrent weight may not remain unitary throughout training.

To investigate this, we ran the scoRNN and full-capacity uRNN with equal hidden unit sizes of n = 512 on the unpermuted MNIST experiment and checked for loss of orthogonality at each epoch.

The results of this experiment are shown in FIG6 .

As can be seen, the recurrent weight matrix for the fullcapacity uRNN becomes less unitary over time, but the orthogonality recurrent weight matrix for scoRNN is not affected by roundoff errors.

As discussed in BID0 , the vanishing/exploding gradient problem is caused by rapid growth or decay of the gradient of the hidden state ∂L ∂ht as we move earlier in the sequence (that is, as t decreases).

To see if vanishing/exploding gradients affect the scoRNN model, we examined hidden state gradients in the scoRNN and LSTM models on the adding problem experiment (see section 5.2) with sequence length T = 500.The norms of these gradients are shown at two different points in time during training in FIG7 .

As can be seen, LSTM gradient norms decrease steadily as we move away from the end of the sequence.

The right half of FIG7 shows that this vanishing effect is exacerbated by training after 300 iterations.

In contrast, scoRNN gradients decay by less than an order of magnitude at the beginning of training, remaining near 10 −2 for all timesteps.

Even after 300 iterations of training, scoRNN hidden state gradients decay only slightly, from 10 −3 at t = 500 to 10 −4 at t = 0.

This allows information to easily propagate from the beginning of the sequence to the end.

The scoRNN architecture is similar in complexity and memory usage to a standard RNN except for the additional memory requirement of storing the n(n − 1)/2 entries of the skew-symmetric matrix A and the additional complexity of forming the recurrent weight matrix W from A with the scaled Cayley transform.

We note that the recurrent weight matrix is generated from the skew-symmetric A matrix only once per training iteration; this computational cost is small compared to the cost of the forward and backward propagations through all time steps in a training batch for a standard RNN.

To experimentally quantify potential differences between scoRNN and the other models, the real run-time for the models on the unpermuted MNIST experiment were recorded and are included in TAB2 .

All models were run on the same machine, which has an Intel Core i5-7400 processor and an nVidia GeForce GTX 1080 GPU.

The scoRNN and LSTM models were run in Tensorflow, while the full and restricted capacity uRNNs were run using code provided in BID21 .The LSTM model was fastest, and hidden sizes largely did not affect time taken per epoch; we suspect this is because the LSTM model we used was built in to Tensorflow.

The LSTMs are of simialr speed to the n = 170 scoRNN, while they are approximately twice as fast as the n = 512 scoRNN.

Matching the number of hidden parameters, the scoRNN model with n = 170 is approximately 1.5 times faster than the restricted-capacity uRNN with n = 512, and twice as fast as the full-capacity uRNN with n = 116.

This relationship can also be seen in the scoRNN and full-capacity uRNN models with ≈ 137k parameters, where the scoRNN takes 11.2 minutes per epoch as compared to 25.8 minutes for the scoRNN.

<|TLDR|>

@highlight

A novel approach to maintain orthogonal recurrent weight matrices in a RNN.

@highlight

Introduces a scheme for learning the recurrent parameter matrix in a neural network that uses the Cayley transform and a scaling weight matrix. 

@highlight

This paper suggests an RNN reparametrization of the recurrent weights with a skew-symmetric matrix using Cayley transform to keep the recurrent weight matrix orthogonal.

@highlight

Novel parametrization of RNNs allows representing orthogonal weight matrices relatively easily.