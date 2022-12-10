Backprop is the primary learning algorithm used in many machine learning algorithms.

In practice, however, Backprop in deep neural networks is a highly sensitive learning algorithm and successful learning depends on numerous conditions and constraints.

One set of constraints is to avoid weights that lead to saturated units.

The motivation for avoiding unit saturation is that gradients vanish and as a result learning comes to a halt.

Careful weight initialization and re-scaling schemes such as batch normalization ensure that input activity to the neuron is within the linear regime where gradients are not vanished and can flow.

Here we investigate backpropagating error terms only linearly.

That is, we ignore the saturation that arise by ensuring gradients always flow.

We refer to this learning rule as Linear Backprop since in the backward pass the network appears to be linear.

In addition to ensuring persistent gradient flow, Linear Backprop is also favorable when computation is expensive since gradients are never computed.

Our early results suggest that learning with Linear Backprop is competitive with Backprop and saves expensive gradient computations.

It is has been long known that deep neural networks with non-polynomial and non-linear units are universal function approximators BID13 .

In the early days of neural network research, however, it was not clear what learning algorithms would find the optimal set of synaptic weights for effective learning.

Rosenblatt's pioneering work essentially only learned the weights at the output layer of the Multi-Layer Perceptron and keeping the input-layer weights fixed at random while Fukushima used Hebbian learning.

Backprop as introduced into neural network research [Werbos, 1974; Rumelhart et al., 1986] has been enormously successful at learning diverse sets of tasks by various deep neural architectures and as a result is by far the most used learning algorithm.

Although enormously successful, Backprop is a highly sensitive learning algorithm and numerous tricks have been collected to make it work in practice BID2 BID4 .

Some of these issues are: dead or saturated units, appropriate learning rates, batch sizes, number of epochs in addition to many other issues.

In particular, considerable effort has been placed into avoiding saturated units.

The primary problem with saturated neurons is that gradients vanish in these regions and hence learning comes to a halt.

As a result, considerable effort has been placed into ensuring that the input activity to neurons are in the linear region.

Some of these efforts are the introduction of regularization such as l 2 penalty (also referred to as weight decay) BID12 Srivastava et al., 2014] , batch normalization BID10 , and careful weight initialization schemes BID6 BID15 .

Other solutions is to consider activation functions that have limited saturating regions to ensure gradient flows BID16 BID11 .

BID8 extensively study activation functions with non-saturating regions to provide gradients.

Since gradient flow is essential for learning, we investigate learning algorithms that ensure linear gradient flow.

The Linear Backprop algorithm (see Algorithm 2 below) ensures gradients flows for all regions and can be used as an alternative for learning.

Compared with Backprop as shown in Algorithm 1 BID7 , the forward pass in Linear Backprop is identical.

The network architecture is still highly non-linear with non-linear activation functions, but when we compute the loss function we only consider linearly backpropagating errors.

Since in Linear Backprop the derivatives of the activation functions are not computed (highlighted in red), the savings in computation in Linear Backprop compared to Backprop is O(ml).Another way to think of our proposed learning rule is that we introduce a regularization term such that when gradients are computed, all the non-linear components are cancelled and only linear gradients persist.

In other words, during inference we use a deep non-linear network however during training the loss function essentially reduces to a deep linear neural network.

The forward pass is computed as usual but the backward pass only uses linear feedback terms.

Several recent investigations have considered variants to Backprop for biological plausibility BID0 BID14 BID1 .

In particular, BID14 showed that learning is also possible with random weights.

BID0 exhaustively considers many Hebbian and error back propagation learning algorithms.

Linear Backprop also shares many similarity with the Straight Through Estimator BID3 , however we propose applying the estimator to any activation function.

Instead we suggest that Linear Backprop saves considerable computational costs as gradients are never computed.

While further research is needed to understand random and alternative learning rules, the Linear Backprop learning rule that we consider here is especially favorable for cases with limited computing resources.

Our empirical results suggest that Linear Backprop in certain conditions can be competitive with Backprop.

DISPLAYFORM0 . .

, l} Require Inputs x and targets y, loss function L Computeŷ, hidden activation's a (i) and h DISPLAYFORM1 DISPLAYFORM2

We begin by considering a synthetic data example to investigate the overfitting and generalization capabilities of Linear Backprop.

There has been considerable recent interest in the generalization capabilities of deep neural networks [Zhang et al., 2016; Poggio et al., 2018] .

The gist of the research is how networks that have far more parameters than training data (often by many orders of magnitude) generalize well to unseen data.

Many experiments have shown that even when training loss is 0 (severe overfitting), there is still generalization [Zhang et al., 2016; Wu et al., 2017] .Here we reproduce the motivating example from [Wu et al., 2017] to understand if Linear Backprop shares similar generalization properties.

We consider learning the third-order polynomial y = x 3 − 3x 2 − x + 1 + N (0, 0.1).

In this experiment, the training set consists of only 5 points and the neural network is trained until the training error is small (for example, ≤ 1 × 10 −6 ).

The neural network is a feed forward Multi-Layer Perceptron (MLP) with 4 hidden layers each with width of 50 ReLU units (over 7,000 parameters).

For a problem with 5 points, this is a highly overparameterized network.

We reproduce the result from [Wu et al., 2017] that such an overparameterized network overfits the training data yet also generalizes gracefully to unseen data (see FIG0 .

We also observe similar generalization capability with the same architecture when trained with Linear Backprop.

The Linear Backprop also overfits on the training data and has similar generalization when trained with Backprop.

BID9 .

For VGG architecture, we use tanh activation functions with vanilla SGD and sweep the learning rate in {10 −3 , 10 −4 , 10 −5 } and l 2 penalty in {10 −1 , 10 −2 , 10 −3 , 10 −4 , 10 −5 , 10 −6 }.

In all experiments we use a batch size of 128 points, train with 100 epochs, and evaluate on the same test set and select the best leaning rate and l 2 for Backprop and Linear Backprop respectively.

FIG1 shows the learning curves on the test set for the best learning rate and l 2 penalty for each learning rule.

The tanh activation function is in particular prone to vanishing gradients since it has many saturating regions.

By using Linear Backprop, however, we are able to ensure that gradients flow to continue learning.

We also consider how learning changes when the architecture is changed.

For the ResNet Binarized Neural Network, we use the hard tanh activation function and the same ResNet architecture in BID5 .

We similarly use only vanilla SGD with sweeping the learning learning rate in {10 −3 , 10 −4 , 10 −5 } and l 2 in {10 −1 , 10 −2 , 10 −3 , 10 −4 , 10 −5 , 10 −6 }.

Again, in all experiments we use a batch size of 128 points, train with 100 epochs, and evaluate on the same test set.

We train this architecture with both Backprop and Linear Backprop.

At 100 epochs, the best Precision@5 validation error we find using Backprop is 28.11% whereas using Linear Backprop the best best Precision@5 validation error is 18.43%.

These networks are significantly slower to train and in our investigation are are interested in knowing which learning algorithm is more favorable at the start of training.

Although these Precision levels are far from state-of-the-art, at only 100 epochs we find that Linear Backprop is highly competitive with traditional Backprop and that further fine-tuning (longer epochs and better batch size) will yield much better results.

The ReLU activation function is designed to ensure that gradients flow maximally.

For this reason we compare how learning with ReLU activation functions compare with Linear Backprop.

We again sweep the learning rate in {10 −3 , 10 −4 , 10 −5 } and l 2 penalty in {10 −1 , 10 −2 , 10 −3 , 10 −4 , 10 −5 , 10 −6 } for 100 epochs and select the best learning rate and l2 for a all activation function and learning algorithm combinations.

B Sweeping different learning rates and weight decays for VGG19 on CIFAR-10In all our VGG19 experiments, we sweep the learning rate in {10 −3 , 10 −4 , 10 −5 } and l 2 penalty in {10 −1 , 10 −2 , 10 −3 , 10 −4 , 10 −5 , 10 −6 } for 100 epochs.

It is instructive to observe how learning progresses for these different parameter settings.

In FIG4 we show the learning curves using the tanh activation function compared with Backprop (lighter color) and Linear Backprop (darker color).

In the early stages of learning, Linear Backprop in particular performs favorably compared with Backprop learning.

<|TLDR|>

@highlight

We ignore non-linearities and do not compute gradients in the backward pass to save computation and to ensure gradients always flow. 

@highlight

The author proposed linear backprop algorithms to ensure gradients flow for all parts during backpropagation.