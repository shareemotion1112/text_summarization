Our work addresses two important issues with recurrent neural networks: (1) they are over-parameterized, and (2) the recurrent weight matrix is ill-conditioned.

The former increases the sample complexity of learning and the training time.

The latter causes the vanishing and exploding gradient problem.

We present a flexible recurrent neural network model called Kronecker Recurrent Units (KRU).

KRU achieves parameter efficiency in RNNs through a Kronecker factored recurrent matrix.

It overcomes the ill-conditioning of the recurrent matrix by enforcing soft unitary constraints on the factors.

Thanks to the small dimensionality of the factors, maintaining these constraints is computationally efficient.

Our experimental results on seven standard data-sets reveal that KRU can reduce the number of parameters by three orders of magnitude in the recurrent weight matrix compared to the existing recurrent models, without trading the statistical performance.

These results in particular show that while there are advantages in having a high dimensional recurrent space, the capacity of the recurrent part of the model can be dramatically reduced.

Deep neural networks have defined the state-of-the-art in a wide range of problems in computer vision, speech analysis, and natural language processing BID28 BID36 .

However, these models suffer from two key issues.

(1) They are over-parametrized; thus it takes a very long time for training and inference.

(2) Learning deep models is difficult because of the poor conditioning of the matrices that parameterize the model.

These difficulties are especially relevant to recurrent neural networks.

Indeed, the number of distinct parameters in RNNs grows as the square of the size of the hidden state conversely to convolutional networks which enjoy weight sharing.

Moreover, poor conditioning of the recurrent matrices results in the gradients to explode or vanish exponentially fast along the time horizon.

This problem prevents RNN from capturing long-term dependencies BID22 BID5 .There exists an extensive body of literature addressing over-parametrization in neural networks.

BID31 first studied the problem and proposed to remove unimportant weights in neural networks by exploiting the second order information.

Several techniques which followed include low-rank decomposition BID13 , training a small network on the soft-targets predicted by a big pre-trained network BID2 , low bit precision training BID12 , hashing BID8 , etc.

A notable exception is the deep fried convnets BID44 which explicitly parameterizes the fully connected layers in a convnet with a computationally cheap and parameter-efficient structured linear operator, the Fastfood transform BID29 .

These techniques are primarily aimed at feed-forward fully connected networks and very few studies have focused on the particular case of recurrent networks BID1 .The problem of vanishing and exploding gradients has also received significant attention.

BID23 proposed an effective gating mechanism in their seminal work on LSTMs.

Later, this technique was adopted by other models such as the Gated Recurrent Units (GRU) BID10 and the Highway networks BID39 for recurrent and feed-forward neural networks respectively.

Other popular strategies include gradient clipping BID37 , and orthogonal initialization of the recurrent weights .

More recently BID1 proposed to use a unitary recurrent weight matrix.

The use of norm preserving unitary maps prevent the gradients from exploding or vanishing, and thus help to capture long-term dependencies.

The resulting model called unitary RNN (uRNN) is computationally efficient since it only explores a small subset of general unitary matrices.

Unfortunately, since uRNNs can only span a reduced subset of unitary matrices their expressive power is limited BID42 .

We denote this restricted capacity unitary RNN as RC uRNN.

Full capacity unitary RNN (FC uRNN) BID42 proposed to overcome this issue by parameterizing the recurrent matrix with a full dimensional unitary matrix, hence sacrificing computational efficiency.

Indeed, FC uRNN requires a computationally expensive projection step which takes O(N 3 ) time (N being the size of the hidden state) at each step of the stochastic optimization to maintain the unitary constraint on the recurrent matrix.

BID35 in their orthogonal RNN (oRNN) avoided the expensive projection step in FC uRNN by parametrizing the orthogonal matrices using Householder reflection vectors, it allows a fine-grained control over the number of parameters by choosing the number of Householder reflection vectors.

When the number of Householder reflection vector approaches N this parametrization spans the full reflection set, which is one of the disconnected subset of the full orthogonal set.

BID25 also presented a way of parametrizing unitary matrices which allows fine-grained control on the number of parameters.

This work called as Efficient Unitary RNN (EURNN), exploits the continuity of unitary set to have a tunable parametrization ranging from a subset to the full unitary set.

Although the idea of parametrizing recurrent weight matrices with strict unitary linear operator is appealing, it suffers from several issues: (1) Strict unitary constraints severely restrict the search space of the model, thus making the learning process unstable.

(2) Strict unitary constraints make forgetting irrelevant information difficult.

While this may not be an issue for problems with non-vanishing long term influence, it causes failure when dealing with real world problems that have vanishing long term influence 4.7.

BID20 have previously pointed out that the good performance of strict unitary models on certain synthetic problems is because it exploits the biases in these data-sets which favors a unitary recurrent map and these models may not generalize well to real world data-sets.

More recently BID41 have also studied this problem of unitary RNNs and the authors found out that relaxing the strict unitary constraint on the recurrent matrix to a soft unitary constraint improved the convergence speed as well as the generalization performance.

Our motivation is to address the problems of existing recurrent networks mentioned above.

We present a new model called Kronecker Recurrent Units (KRU).

At the heart of KRU is the use of Kronecker factored recurrent matrix which provide an elegant way to adjust the number of parameters to the problem at hand.

This factorization allows us to finely modulate the number of parameters required to encode N ?? N matrices, from O(log(N )) when using factors of size 2 ?? 2, to O(N 2 ) parameters when using a single factor of the size of the matrix itself.

We tackle the vanishing and exploding gradient problem through a soft unitary constraint BID26 BID20 BID11 BID41 .

Thanks to the properties of Kronecker matrices BID40 , this constraint can be enforced efficiently.

Please note that KRU can readily be plugged into vanilla real space RNN, LSTM and other variants in place of standard recurrent matrices.

However in case of LSTMs we do not need to explicitly enforce the approximate orthogonality constraints as the gating mechanism is designed to prevent vanishing and exploding gradients.

Our experimental results on seven standard data-sets reveal that KRU and KRU variants of real space RNN and LSTM can reduce the number of parameters drastically (hence the training and inference time) without trading the statistical performance.

Our core contribution in this work is a flexible, parameter efficient and expressive recurrent neural network model which is robust to vanishing and exploding gradient problem.

The paper is organized as follows, in section 2 we restate the formalism of RNN and detail the core motivations for KRU.

In section 3 we present the Kronecker recurrent units (KRU).

We present our experimental findings in section 4 and section 5 concludes our work.

DISPLAYFORM0

Hidden and output bias ??(.), L(??, y) Point-wise non-linear activation function and the loss function 2 RECURRENT NEURAL NETWORK FORMALISM TAB0 summarizes some notations that we use in the paper.

We consider the field to be complex rather than real numbers.

We will motivate the choice of complex numbers later in this section.

Consider a standard recurrent neural network BID14 .

Given a sequence of T input vectors: x 0 , x 1 , . . .

, x T ???1 , at a time step t RNN performs the following: DISPLAYFORM0 (1) DISPLAYFORM1 where?? t is the predicted value at time step t.

The total number of parameters in a RNN is c(DN + N 2 + N + M + M N ), where c is 1 for real and 2 for complex parametrization.

As we can see, the number of parameters grows quadratically with the hidden dimension, i.e., O(N 2 ).

We show in the experiments that this quadratic growth is an over parametrization for many real world problems.

Moreover, it has a direct impact on the computational efficiency of RNNs because the evaluation of Wh t???1 takes O(N 2 ) time and it recursively depends on previous hidden states.

However, other components Ux t and Vh t can usually be computed efficiently by a single matrix-matrix multiplication for each of the components.

That is, we can perform U[x 0 , . . .

, x T ] and V[h 0 , . . . , h T ???1 ], this is efficient using modern BLAS libraries.

So to summarize, if we can control the number of parameters in the recurrent matrix W, then we can control the computational efficiency.

The vanishing and exploding gradient problem refers to the decay or growth of the partial derivative of the loss L(.) with respect to the hidden state h t i.e. ???L ???ht as the number of time steps T grows BID1 .

By the application of the chain rule, the following can be shown BID1 : DISPLAYFORM0 From Equation 3, it is clear that if the absolute value of the eigenvalues of W deviates from 1 then ???L ???ht may explode or vanish exponentially fast with respect to T ??? t. So a strategy to prevent vanishing and exploding gradient is to control the spectrum of W.

Although BID1 and BID42 use complex valued networks with unitary constraints on the recurrent matrix, the motivations for such models are not clear.

We give a simple but compelling reason for complex-valued recurrent networks.

The absolute value of the determinant of a unitary matrix is 1.

Hence in the real space, the set of all unitary (orthogonal) matrices have a determinant of 1 or ???1, i.e., the set of all rotations and reflections respectively.

Since the determinant is a continuous function, the unitary set in real space is disconnected.

Consequently, with the real-valued networks we cannot span the full unitary set using the standard continuous optimization procedures.

On the contrary, the unitary set is connected in the complex space as its determinants are the points on the unit circle and we do not have this issue.

As we mentioned in the introduction BID25 uses this continuity of unitary space to have a tunable continuous parametrization ranging from subspace to full unitary space.

Any continuous parametrization in real space can only span a subset of the full orthogonal set.

For example, the Householder parametrization BID35 suffers from this issue.

We consider parameterizing the recurrent matrix W as a Kronecker product of F matrices DISPLAYFORM0 Where each W f ??? C P f ??Q f and DISPLAYFORM1 To illustrate the Kronecker product of matrices, let us consider the simple case when ??? f {P f = Q f = 2}. This implies F = log 2 N .

And W is recursevly defined as follows: DISPLAYFORM2 DISPLAYFORM3 When ??? f {p f = q f = 2} the number of parameters is 8 log 2 N and the time complexity of hidden state computation is O(N log 2 N ).

When ??? f {p f = q f = N } then F = 1 and we will recover standard complex valued recurrent neural network.

We can span every Kronecker representations in between by choosing the number of factors and the size of each factor.

In other words, the number of Kronecker factors and the size of each factor give us fine-grained control over the number of parameters and hence over the computational efficiency.

This strategy allows us to design models with the appropriate trade-off between computational budget and statistical performance.

All the existing models lack this flexibility.

The idea of using Kronecker factorization for approximating Fisher matrix in the context of natutal gradient methods have recently recieved much attention.

The algorithm was originally presented in BID33 and was later extended to convolutional layers , distributed second order optimization BID3 and for deep reinforcement learning BID43 .

However Kronecker matrices have not been well explored as learnable parameters except BID45 ) used it's spectral property for fast orthogonal projection and BID46 used it as a layer in convolutional neural networks.

Poor conditioning results in vanishing or exploding gradients.

Unfortunately, the standard solution which consists of optimization on the strict unitary set suffers from the retention of noise over time.

Indeed, the small eigenvalues of the recurrent matrix can represent a truly vanishing long-term influence on the particular problem and in that sense, there can be good or bad vanishing gradients.

Consequently, enforcing strict unitary constraint (forcing the network to never forget) can be a bad strategy.

A simple solution to get the best of both worlds is to enforce unitary constraint approximately by using the following regularization: DISPLAYFORM0 Please note that these constraints are enforced on each factor of the Kronecker factored recurrent matrix.

This procedure is computationally very efficient since the size of each factor is typically small.

It suffices to do so because if each of the Kronecker factors {W 0 , . . .

, W F ???1 } are unitary then the full matrix W is unitary BID40 and if each of the factors are approximately unitary then the full matrix is approximately unitary.

We apply soft unitary constraints as a regularizer whose strength is cross-validated on the validation set.

This type of regularizer has recently been exploited for real-valued models.

BID11 showed that enforcing approximate orthogonality constraint on the weight matrices make the network robust to adversarial samples as well as improve the learning speed.

In metric learning BID26 have shown that it better conditions the projection matrix thereby improving the robustness of stochastic gradient over a wide range of step sizes as well asthe generalization performance.

BID20 and BID41 have also used this soft unitary contraints on standard RNN after identifying the problems with the strict unitary RNN models.

However the computational complexity of naively applying this soft constraint is O(N 3 ).

This is prohibitive for RNNs with large hidden state unless one considers a Kronecker factorization.

Existing deep learning libraries such as Theano BID6 , Tensorflow BID0 and Pytorch BID38 do not support fast primitives for Kronecker products with arbitrary number of factors.

So we wrote custom CUDA kernels for Kronecker forward and backward operations.

All our models are implemented in C++.

We will release our library to reproduce all the results which we report in this paper.

We use tanh as activation function for RNN, LSTM and our model KRU-LSTM.

Whereas RC uRNN, FC uRNN and KRU uses complex rectified linear units BID1 .

Copy memory problem BID23 tests the model's ability to recall a sequence after a long time gap.

In this problem each sequence is of length T + 20 and each element in the sequence come from 10 classes {0, . . .

, 9}. The first 10 elements are sampled uniformly with replacement from {1, . . .

, 8}. The next T ??? 1 elements are filled with 0, the 'blank' class followed by 9, the 'delimiter' and the remaining 10 elements are 'blank' category.

The goal of the model is to output a sequence of T + 10 blank categories followed by the 10 element sequence from the beginning of the input sequence.

The expected average cross entropy for a memory-less strategy is FORMULA3 , we choose the training and test set size to be 100K and 10K respectively.

All the models were trained using RMSprop with a learning rate of 1e???3, decay of 0.9 and a batch size of 20.

For both the settings T = 1000 and T = 2000, KRU converges to zero average cross entropy faster than FC uRNN.

All the other baselines are stuck at the memory-less cross entropy.

The results are shown in figure 1.

For this problem we do not learn the recurrent matrix of KRU, We initialize it by random unitary matrix and just learn the input to hidden, hidden to output matrices and the bias.

We found out that this strategy already solves the problem faster than all other methods.

Our model in this case is similar to a parametrized echo state networks (ESN).

ESNs are known to be able to learn long-term dependencies if they are properly initialized BID24 .

We argue that this data-set is not an ideal benchmark for evaluating RNNs in capturing long term dependencies.

Just a unitary initialization of the recurrent matrix would solve the problem.

Following BID1 we describe the adding problem BID23 .

Each input vector is composed of two sequences of length T .

The first sequence is sampled from U[0, 1].

In the second sequence exactly two of the entries is 1, the 'marker' and the remaining is 0.

The first 1 is located uniformly at random in the first half of the sequence and the other 1 is located again uniformly at random in the other half of the sequence.

The network's goal is to predict the sum of the numbers from the first sequence corresponding to the marked locations in the second sequence.

We evaluate four settings as in BID1 with T =100, T =200, T =400, and T =750.

For all four settings, KRU uses a hidden dimension N of 512 with 2x2 Kronecker factors which corresponds to ???3K parameters in total.

We use a RNN of N = 128 (??? 17K parameters) , LSTM of N = 128 ( ??? 67K parameters), RC uRNN of N = 512 ( ??? 7K parameters) , FC uRNN of N = 128 ( ??? 33K parameters).

The train and test set sizes are chosen to be 100K and 10K respectively.

All the models were trained using RMSprop with a learning rate of 1e???3 and a batch size of 20 or 50 with the best results are being reported here.

The results are presented in figure 2.

KRU converges faster than all other baselines even though it has much fewer parameters.

This shows the effectiveness of soft unitary constraint which controls the flow of gradients through very long time steps and thus deciding what to forget and remember in an adaptive way.

LSTM also converges to the solution and this is achieved through its gating mechanism which controls the flow of the gradients and thus the long term influence.

However LSTM has 10 times more parameters than KRU.

Both RC uRNN and FC uRNN converges for T = 100 but as we can observe, the learning is not stable.

The reason for this is that RC uRNN and FC uRNN retains noise since they are strict unitary models.

Please note that we do not evaluate RC uRNN for T = 400 and T = 750 because we found out that the learning is unstable for this model and is often diverging.

Results on adding problem for T =100, T =200, T =400 and T =750.

KRU consistently outperforms the baselines on all the settings with fewer parameters.

As outlined by , we evaluate the Pixel by pixel MNIST task.

MNIST digits are shown to the network pixel by pixel and the goal is to predict the class of the digit after seeing all the pixels one by one.

We consider two tasks: (1) Pixels are read from left to right from top or bottom and (2) Pixels are randomly permuted before being shown to the network.

The sequence length for these tasks is T = 28 ?? 28 = 784.

The size of the MNIST training set is 60K among which we choose 5K as the validation set.

The models are trained on the remaining 55K points.

The model which gave the best validation accuracy is chosen for test set evaluation.

All the models are trained using RMSprop with a learning rate of 1e???3 and a decay of 0.9.The results are summarized in FIG3 and table 2.

On the unpermuted task LSTM achieve the state of the art performance even though the convergence speed is slow.

Recently a low rank plus diagonal gated recurrent unit (LRD GRU) (Barone, 2016) have shown to achieves 94.7 accuracy on permuted MNIST with 41.2K parameters whereas KRU achieves 94.5 with just 12K parameters i.e KRU has 3x parameters less than LRD GRU.

Please also note that KRU is a simple model without a gating mechanism.

KRU can be straightforwardly plugged into LSTM and GRU to exploit the additional benefits of the gating mechanism which we will show in the next experiments with a KRU-LSTM.

We now consider character level language modeling on Penn TreeBank data-set BID32 .

Penn TreeBank is composed of 5017K characters in the training set, 393K characters in the validation set and 442K characters in the test set.

The size of the vocabulary was limited to 10K most frequently occurring words and the rest of the words are replaced by a special <UNK> character BID36 .

The total number of unique characters in the data-set is 50, including the special <UNK> character.

All our models were trained for 50 epochs with a batch size of 50 and using ADAM BID27 .

We use a learning rate of 1e???3 which was found through cross-validation with default beta parameters BID27 ).

If we do not see an improvement in the validation bits per character (BPC) after each epoch then the learning rate is decreased by 0.30.

Back-propagation through time (BPTT) is unrolled for 30 time frames on this task.

We did two sets of experiments to have fair evaluation with the models whose results were available for a particular parameter setting BID35 and also to see how the performance evolves as the number of parameters are increased.

We present our results in table 3.

We observe that the strict orthogonal model, oRNN fails to generalize as well as other models even with a high capacity recurrent matrix.

KRU and KRU-LSTM performs very close to RNN and LSTM with fewer parameters in the recurrent matrix.

Please recall that the computational bottleneck in RNN is the computation of hidden states 2.1 and thus having fewer parameters in the recurrent matrix can significantly reduce the training and inference time.

Recently HyperNetworks BID18 have shown to achieve the state of the art performance of 1.265 and 1.219 BPC on the PTB test set with 4.91 and 14.41 million parameters respectively.

This is respectively 13 and 38 times more parameters than the KRU-LSTM model which achieves 1.47 test BPC.

Also Recurrent Highway Networks (RHN) BID47 proved to be a promising model for learning very deep recurrent neural networks.

Running experiments, and in particular exploring meta-parameters with models of that size, requires unfortunately computational means beyond what was at our disposal for this work.

However, there is no reason that the consistent behavior and improvement observed on the other reference baselines would not generalize to that type of large-scale models.

BID9 our main objective here is to have a fair evaluation of different recurrent neural networks.

We took the baseline RNN and LSTM models of BID9 whose model sizes were chosen to be small enough to avoid overfitting.

We choose the model size of KRU and KRU-LSTM in such way that it has fewer parameters compared to the baselines.

As we can in the table 4 both our models (KRU and KRU-LSTM) overfit less and generalizes better.

We also present the wall-clock running time of different methods in the figure 4.

BID9 100 ???20K 10K 8.82 9.10 5.64 9.03 LSTM BID9

Framewise phoneme classification BID16 is the problem of classifying the phoneme corresponding to a sound frame.

We evaluate the models for this task on the real world TIMIT data-set BID15 .

TIMIT contains a training set of 3696 utterances among which we use 184 as the validation set.

The test set is composed of 1344 utterances.

We extract 12 Mel-Frequency Cepstrum Coefficients (MFCC) BID34 ) from 26 filter banks and also the log energy per frame.

We also concatenate the first derivative, resulting in a feature descriptor of dimension 26 per frame.

The frame size is chosen to be 10ms and the window size is 25ms.

The number of time steps to which back-propagation through time (BPTT) is unrolled corresponds to the length of each sequence.

Since each sequence is of different length this implies that for each sample BPTT steps are different.

All the models are trained for 20 epochs with a batch size of 1 using ADAM with default beta parameters BID27 .

The learning rate was cross-validated for each of the models from ?? ??? {1e???2, 1e???3, 1e???4} and the best results are reported here.

The best learning rate for all the models was found out to be 1e???3 for all the models.

Again if we do not observe a decrease in the validation error after each epoch, we decrease the learning rate by a factor of ?? ??? {1e???1, 2e???1, 3e???1} which is again cross-validated.

Figure 5 summarizes Figure 5: KRU and KRU-LSTM performs better than the baseline models with far less parameters in the recurrent weight matrix on the challenging TIMIT data-set BID15 .

This significantly bring down the training and inference time of RNNs.

Both LSTM and KRU-LSTM converged within 5 epochs whereas RNN and KRU took 20 epochs.

A similar result was obtained by BID16 using RNN and LSTM with 4 times less parameters respectively than our models.

However in their work the LSTM took 20 epochs to converge and the RNN took 70 epochs.

We have also experimented with the same model size as that of BID16 and have obtained very similar results as in the table but at the expense of longer training times.

Here we study the properties of soft unitary constraints on KRU.

We use Polyphonic music modeling data-sets BID7 : JSB Chorales and Piano-midi, as well as TIMIT data-set for this set of experiments.

We varied the amplitude of soft unitary constraints from 1e ??? 7 to 1e ??? 1, the higher the amplitude the closer the recurrent matrix will be to the unitary set.

All other hyper-parameters, such as the learning rate and the model size are fixed.

We present our studies in the figure 6.

As we increase the amplitude we can see that the recurrent matrix is getting better conditioned and the spectral norm or the spectral radius is approaching towards 1.

As we can see that the validation performance can be improved using this simple soft unitary constraints.

For JSB Chorales the best validation performance is achieved at an amplitude of 1e ??? 2, whereas for Piano-midi it is at 1e ??? 1.For TIMIT phoneme recognition problem, the best validation error is achieved at 1e ??? 5 but as we increase the amplitude further, the performance drops.

This might be explained by a vanishing long-term influence that has to be forgotten.

Our model achieve this by cross-validating the amplitude of soft unitary constraints.

These experiments also reveals the problems of strict unitary models such as RC uRNN BID1 , FC uRNN BID42 , oRNN BID35 and EURNN BID25 ) that they suffer from the retention of noise from a vanishing long term influence and thus fails to generalize.

A popular heuristic strategy to avoid exploding gradients in RNNs and thereby making their training robust and stable is gradient clipping.

Most of the state of the art RNN models use gradient clipping for training.

Please note that we are not using gradient clipping with KRU.

Our soft unitary constraints offer a principled alternative to gradient clipping.

Moreover BID19 recently showed that gradient descent converges to the global optimizer of linear recurrent neural networks even though the learning problem is non-convex.

The necessary condition for the global convergence guarantee requires that the spectral norm of recurrent matrix is bounded by 1.

This seminal theoretical result also inspires to use regularizers which control the spectral norm of the recurrent matrix, such as the soft unitary constraints.

We have presented a new recurrent neural network model based on its core a Kronecker factored recurrent matrix.

Our core reason for using a Kronecker factored recurrent matrix stems from it's elegant algebraic and spectral properties.

Kronecker matrices are neither low-rank nor block-diagonal but it is multi-scale like the FFT matrix.

Kronecker factorization provides a fine control over the model capacity and it's algebraic properties enable us to design fast matrix multiplication algorithms.

It's spectral properties allow us to efficiently enforce constraints like positive semi-definitivity, unitarity and stochasticity.

As we have shown, we used the spectral properties to efficiently enforce a soft unitary constraint.

Experimental results show that our approach out-perform classical methods which uses O(N 2 ) parameters in the recurrent matrix.

Maybe as important, these experiments show that both on toy problems ( ?? 4.1 and 4.2), and on real ones ( ?? 4.3, 4.4, , and ?? 4.6) , while existing methods require tens of thousands of parameters in the recurrent matrix, competitive or better than state-of-the-art performance can be achieved with far less parameters in the recurrent weight matrix.

These surprising results provide a new and counter-intuitive perspective on desirable memory-capable architectures: the state should remain of high dimension to allow the use of high-capacity networks to encode the input into the internal state, and to extract the predicted value, but the recurrent dynamic itself can, and should, be implemented with a low-capacity model.

From a practical standpoint, the core idea in our method is applicable not only to vanilla recurrent neural networks and LSTMS as we showed, but also to a variety of machine learning models such as feed-forward networks BID46 , random projections and boosting weak learners.

Our future work encompasses exploring other machine learning models and on dynamically increasing the capacity of the models on the fly during training to have a perfect balance between computational efficiency and sample complexity.

Given a sequence of T input vectors: x 0 , x 1 , . . .

, x T ???1 , let us consider the operation at the hidden layer t of a recurrent neural network: DISPLAYFORM0 By the chain rule, DISPLAYFORM1 where ?? is the non-linear activation function and J k+1 = diag(?? (z k+1 )) is the Jacobian matrix of the non-linear activation function.

DISPLAYFORM2 From equation 14 it is clear the norm of the gradient is exponentially dependent upon two factors along the time horizon:??? The norm of the Jacobian matrix of the non-linear activation function J k+1 .???

The norm of the hidden to hidden weight matrix W .These two factors are causing the vanishing and exploding gradient problem.

Since the gradient of the standard non-linear activation functions such as tanh and ReLU are bounded between [0, 1], J k+1 does not contribute to the exploding gradient problem but it can still cause vanishing gradient problem.

LSTM networks presented an elegant solution to the vanishing and exploding gradients through the introduction of gating mechanism.

Apart from the standard hidden state in RNN, LSTM introduced one more state called cell state c t .

LSTM has three different gates whose functionality is described as follows: DISPLAYFORM0 Decides what information to keep and erase from the previous cell state.

DISPLAYFORM1 Decides what new information should be added to the cell state.??? Output gate (W o , U o , b o ):Decides which information from the cell state is going to the output.

In addition to the gates, LSTM prepares candidates for the information from the input gate that might get added to the cell state through the action of input gate.

Let's denote the parameters describing the function that prepares this candidate information as W c , U c , b c .Given a sequence of T input vectors: x 0 , x 1 , . . .

, x T ???1 , at a time step t LSTM performs the following: DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 DISPLAYFORM6 where ??(.) and ?? (.) are the point-wise sigmoid and tanh functions.

indicates element-wise multiplication.

The first three are gating operations and the 4th one prepares the candidate information.

The 5th operation updates the cell-state and finally in the 6th operation the output gate decided what to go into the current hidden state.

Unitary evolution RNN (uRNN) proposed to solve the vanishing and exploding gradients through a unitary recurrent matrix, which is for the form: DISPLAYFORM0 Where: DISPLAYFORM1 Diagonal matrices whose diagonal entries are of the from D kk = e i?? k , implies each matrix have N parameters, (?? 0 , . . .

, ?? N ???1 ).??? F and F ???1 : Fast Fourier operator and inverse fast Fourier operator respectively.??? R 1 , R 2 : Householder reflections.

DISPLAYFORM2 The total number of parameters for this uRNN operator is 7N and the matrix vector can be done N log(N ) time.

It is parameter efficient and fast but not flexible and suffers from the retention of noise and difficulty in optimization due its unitarity.

Orthogonal RNN (oRNN) parametrizes the recurrent matrices using Householder reflections.

DISPLAYFORM3 where DISPLAYFORM4 and DISPLAYFORM5 where DISPLAYFORM6 The number of parameters in this parametrization is O(N K).

When N = K = 1 and v = 1, it spans the rotation subset and when v = ???1, it spans the full reflection subset.

Consider a matrix W ??? C N ??N factorized as a Kronecker product of F matrices W 0 , . . .

, W F ???1 , DISPLAYFORM0 Where each W i ??? C Pi??Qi respectively and DISPLAYFORM1 DISPLAYFORM2 Proof.

DISPLAYFORM3

For simplicity here we use real number notations.

Consider a dense matrix X ??? R M ??K and a Kronecker factored matrix DISPLAYFORM0 The computational complexity first expanding the Kronecker factored matrix and then computing the matrix product is O(M N K).

This can be reduced by exploiting the recursive definition of Kronecker matrices.

For examples when N = K and ??? f {P f = Q f = 2}, the matrix product can be computed DISPLAYFORM1 The matrix product in 29 can be recursively defined as DISPLAYFORM2 Please note that the binary operator is not the standard matrix multiplication operator but instead it denotes a strided matrix multiplication.

The stride is computed according to the algebra of Kronecker matrices.

Let us define Y recursively: DISPLAYFORM3 Combining equation 34 and 32 DISPLAYFORM4 We use the above notation for Y in the algorithm.

That is the algorithm illustrated here will cache all the intermediate outputs (Y 0 , . . .

, Y F ???1 ) instead of just Y F ???1 .

These intermediate outputs are then later to compute the gradients during the back-propagation.

This cache will save some computation during the back-propagation.

If the model is just being used for inference then the algorithm can the organized in such a way that we do not need to cache the intermediate outputs and thus save memory.

Algorithm for computing the product between a dense matrix and a Kronecker factored matrix34 is given below 1.

All the matrices are assumed to be stored in row major order.

For simplicity the algorithm is illustrated in a serial fashion.

Please note the lines 4 to 15 except lines 9-11 can be trivially parallelized as it writes to independent memory locations.

The GPU implementation exploits this fact.

Algorithm 1 That is, the Kronecker layer is parametrized by a Kronecker factored matrix W = ??? F ???1 f =0 W f stored as it factors {W 0 , . . .

, W F ???1 } and it takes an input X and produces output Y = Y F ???1 using the algorithm 1.The following algorithm 2 computes the Gradient of the Kronecker factors: {gW 0 , . . .

, gW F ???1 } and the Jacobian of the input matrix gX given the Jacobian of the output matrix: gY = gY F ???1 .Algorithm 2 Gradient computation in a Kronecker layer.

Input: Input matrix X ??? R M ??K , Kronecker factors {W 0 , . . .

, W F ???1 } : W f ??? R p f ??q f , Size of each Kronecker factors {(P 0 , Q 0 ), . . . , (P F ???1 , Q F ???1 )} : DISPLAYFORM5

@highlight

Out work presents a Kronecker factorization of recurrent weight matrices for parameter efficient and well conditioned recurrent neural networks.