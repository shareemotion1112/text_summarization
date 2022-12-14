Bayesian methods have been successfully applied to sparsify weights of neural networks and to remove structure units from the networks, e. g. neurons.

We apply and further develop this approach for gated recurrent architectures.

Specifically, in addition to sparsification of individual weights and neurons, we propose to sparsify preactivations of gates and information flow in LSTM.

It makes some gates and information flow components constant, speeds up forward pass and improves compression.

Moreover, the resulting structure of gate sparsity is interpretable and depends on the task.

Recurrent neural networks (RNNs) yield high-quality results in many applications BID0 BID3 BID17 BID20 but often overfit due to overparametrization.

In many practical problems, RNNs can be compressed orders of times with only slight quality drop or even with quality improvement BID1 BID14 BID19 .

Methods for RNN compression can be divided into three groups: based on matrix factorization BID5 BID18 , quantization BID6 or sparsification BID1 BID14 BID19 .We focus on RNNs sparsification.

Two main groups of approaches for sparsification are pruning and Bayesian sparsification.

In pruning BID14 BID19 , weights with absolute values less than a predefined threshold are set to zero.

Such methods imply a lot of hyperparameters (thresholds, pruning schedule etc).

Bayesian sparsification techniques BID13 BID15 BID7 BID8 BID1 treat weights of an RNN as random variables and approximate posterior distribution over them given sparsity-inducing prior distribution.

After training weights with low signal-to-noise ratio are set to zero.

This allows eliminating the majority of weights from the model without time-consuming hyperparameters tuning.

Also, Bayesian sparsification techniques can be easily extended to permanently set to zero intermediate variables in the network's computational graph BID15 BID7 (e.g. neurons in fully-connected networks or filters in convolutional networks).

It is achieved by multiplying such a variable on a learnable weight, finding posterior over it and setting the weight to zero if the corresponding signal-to-noise ratio is small.

In this work, we investigate the last mentioned property for gated architectures, particularly for LSTM.

Following BID1 BID13 , we sparsify individual weights of the RNN.

Following BID7 , we eliminate neurons from the RNN by introducing multiplicative variables on activations of neurons.

Our main contribution is the introduction of multiplicative variables on preactivations of the gates and information flow in LSTM.

This leads to several positive effects.

Firstly, when some component of preactivations is permanently set to zero, the corresponding gate becomes constant.

It simplifies LSTM structure and speeds up computations.

Secondly, we obtain a three-level hierarchy of sparsification: sparsification of individual weights helps to sparsify gates and information flow (make their components constant), and sparsification of gates and information flow helps to sparsify neurons (remove them from the model).

As a result, the overall compression of the model is higher.

Consider a dataset of N objects (x i , y i ) and a model p(y|x, W ) parametrized by a neural network with weights W .

In BID13 , the authors propose a Bayesian technique called Sparse variational dropout (SparseVD) for neural networks sparsification.

This model comprises log-uniform prior over weights: p(|w ij |) ??? 1 |wij | and fully factorized normal approximate posterior: q(w ij ) = N (w ij |m ij , ?? 2 ij ).

To find parameters of the approximate posterior distribution, evidence lower bound (ELBO) is optimized: DISPLAYFORM0 Because of the log-uniform prior, for the majority of weights signal-to-noise ratio m 2 ij /?? 2 ij ??? 0 and these weights do not affect network's output.

In BID1 SparseVD is adapted to RNNs.

In BID7 the authors propose to multiply activations of neurons on group variables z and to learn and sparsify group variables along with W .

They put standard normal prior on W and log-uniform prior on z. The first prior moves mean values of W to 0, and it helps to set to zero z and to remove neurons from the model.

This model is equivalent to multiplying rows of weight matrices on group variables.

To sparsify individual weights, we apply SparseVD BID13 to all weights of the RNN, taking into account recurrent specifics underlined in BID1 .

To compress layers and remove neurons, we follow BID7 and introduce group variables for the neurons of all layers (excluding output predictions), and specifically, zx and z h for input and hidden neurons of LSTM.The key component of our model is introducing groups variables z i , z f , z g , z o on preactivations of gates and information flow.

The resulting LSTM layer looks as follows: DISPLAYFORM0 Described model is equivalent to multiplying rows and columns of weight matrices on group variables: DISPLAYFORM1 {same for i, o and g}We learn group variables z in the same way as weights W : approximate posterior with fully factorized normal distribution given fully factorized log-uniform prior distribution 2 .

To find approximate posterior distribution, we maximize ELBO (1).

After learning, we set all weights and group variables with signal-to-noise ratio less than 0.05 to 0.

g is set to 0, the corresponding gate or information flow component becomes constant (equal to activation function of bias).

It means that we don't need to compute this component, and the forward pass through LSTM is accelerated.

Related work.

In BID19 the authors propose a pruning-based method that removes neurons from LSTM and argue that independent removing of i, f, g, o components may lead to invalid LSTM units.

In our model, we do not remove these components but make them constant, gaining compression and acceleration with correct LSTM structure.

We perform experiments with LSTM architecture on two types of problems: text classification (datasets IMDb BID9 and AGNews BID21 ) and language modeling (dataset PTB BID10 , character and word level tasks).

For text classification, we use networks with an embedding layer, one recurrent layer and an output dense layer at the last timestep.

For language modeling, we use networks with one We compare four models in terms of quality and sparsity: baseline model without any regularization, standard SparseVD model for weights sparsification only (W), SparseVD model with group variables for neurons sparsification (W+N) and SparseVD model with group variables for gates and neurons sparsification (W+G+N).

In all SparseVD models, we sparsify weights matrices of all layers.

Since in text classification tasks usually only a small number of input words are important, we use additional multiplicative weights to sparsify the input vocabulary in case of group sparsification (W+N, W+G+N) following BID1 .

On the contrary, in language modeling tasks all input characters or words are usually important, therefore we do not use z x for this task.

Additional sparsification of input neurons in this case noticeably damage models quality and sparsity level of hidden neurons.

To measure the sparsity level of our models we calculate the compression rate of individual weights as follows: |w|/|w = 0|.

To compute the number of remaining neurons or non-constant gates we use corresponding rows/columns of W and corresponding weights z if applicable.

Quantitative results are shown in Table 1 .

Multiplicative variables for neurons boost group sparsity level without a significant quality drop.

Additional variables for gates and information flow not only make some gates constant but also increase group sparsity level even further.

Moreover, for a lot of constant gates bias values tend to be very large or small making corresponding gates either always open or close.

Proposed gate sparsification technique also reveals an interesting work-flow structure of LSTM networks for different tasks.

FIG0 shows typical examples of gates of remaining hidden neurons.

For language modeling tasks output gates are very important because models need both store all the information about the input in the memory and output only the current prediction at each timestep.

On the contrary, for text classification tasks models need to output the answer only once at the end of the sequence, hence they do not really use output gates.

Also, the character level language modeling task is more challenging than the word level one: the model uses the whole gating mechanism to solve it.

We think this is the main reason why gate sparsification does not help here.

Datasets.

To evaluate our approach on text classification task we use two standard datasets: IMDb dataset BID9 for binary classification and AGNews dataset BID21 for four-class classification.

We set aside 15% and 5% of training data for validation purposes respectively.

For both datasets, we use the vocabulary of 20,000 most frequent words.

To evaluate our approach on language modeling task we use the Penn Treebank corpus BID10 with the train/valid/test partition of Mikolov et al. mikolov11 .

The dataset has a vocabulary of 50 characters or 10,000 words.

Architectures for text classification.

We use networks with one embedding layer of 300 units, one LSTM layer of 128 / 512 hidden units for IMDb / AGNews, and finally, a fully connected layer applied to the last output of the LSTM.

Embedding layer is initialized with word2vec BID12 / GloVe BID16 and SparseVD models are trained for 800 / 150 epochs on IMDb / AGNews.

Hidden-to-hidden weight matrices W h are initialized orthogonally and all other matrices are initialized uniformly using the method by Glorot and Bengio pmlr-v9-glorot10a.

We train our networks using Adam BID4 with batches of size 128 and a learning rate of 0.0005.

Baseline networks overfit for all our tasks, therefore, we present results for them with early stopping.

Architectures for language modeling.

To solve character / word-level tasks we use networks with one LSTM layer of 1000 / 256 hidden units and fully-connected layer with softmax activation to predict next character or word.

We train SparseVD models for 250 / 150 epochs on character-level / word-level tasks.

All weight matrices of the networks are initialized orthogonally and all biases are initialized with zeros.

Initial values of hidden and cell elements are not trainable and equal to zero.

For the character-level task, we train our networks on non-overlapping sequences of 100 characters in mini-batches of 64 using a learning rate of 0.002 and clip gradients with threshold 1.

For the word-level task, networks are unrolled for 35 steps.

We use the final hidden states of the current mini-batch as the initial hidden state of the subsequent mini-batch (successive mini batches sequentially traverse the training set).

The size of each mini-batch is 32.

We train models using a learning rate of 0.002 and clip gradients with threshold 10.

Baseline networks overfit for all our tasks, therefore, we present results for them with early stopping.

<|TLDR|>

@highlight

We propose to sparsify preactivations of gates and information flow in LSTM to make them constant and boost the neuron sparsity level

@highlight

This paper proposed a sparsification method for recurrent neural networks by eliminating neurons with zero preactivations to obtain compact networks.