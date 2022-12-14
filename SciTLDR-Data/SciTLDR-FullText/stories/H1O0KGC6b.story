One of the main challenges of deep learning methods is the choice of an appropriate training strategy.

In particular, additional steps, such as unsupervised pre-training, have been shown to greatly improve the performances of deep structures.

In this article, we propose an extra training step, called post-training, which only optimizes the last layer of the network.

We show that this procedure can be analyzed in the context of kernel theory, with the first layers computing an embedding of the data and the last layer a statistical model to solve the task based on this embedding.

This step makes sure that the embedding, or representation, of the data is used in the best possible way for the considered task.

This idea is then tested on multiple architectures with various data sets, showing that it consistently provides a boost in performance.

One of the main challenges of the deep learning methods is to efficiently solve the highly complex and non-convex optimization problem involved in the training step.

Many parameters influence the performances of trained networks, and small mistakes can drive the algorithm into a sub-optimal local minimum, resulting into poor performances BID0 .

Consequently, the choice of an appropriate training strategy is critical to the usage of deep learning models.

The most common approach to train deep networks is to use the stochastic gradient descent (SGD) algorithm.

This method selects a few points in the training set, called a batch, and compute the gradient of a cost function relatively to all the layers parameter.

The gradient is then used to update the weights of all layers.

Empirically, this method converges most of the time to a local minimum of the cost function which have good generalization properties.

The stochastic updates estimate the gradient of the error on the input distribution, and several works proposed to use variance reduction technique such as Adagrap BID2 , RMSprop BID8 or Adam (Kingma & Ba, 2015) , to achieve faster convergence.

While these algorithms converge to a local minima, this minima is often influenced by the properties of the initialization used for the network weights.

A frequently used approach to find a good starting point is to use pre-training BID12 BID5 .

This method iteratively constructs each layer using unsupervised learning to capture the information from the data.

The network is then fine-tuned using SGD to solve the task at hand.

Pretraining strategies have been applied successfully to many applications, such as classification tasks BID0 BID17 , regression BID6 , robotics BID4 or information retrieval BID19 .

The influence of different pre-training strategies over the different layers has been thoroughly studied in BID13 .

In addition to improving the training strategies, these works also shed light onto the role of the different layers BID3 BID15 .

The first layers of a deep neural network, qualified as general, tend to learn feature extractors which can be reused in other architectures, independently of the solved task.

Meanwhile, the last layers of the network are much more dependent of the task and data set, and are said to be specific.

Deep Learning generally achieves better results than shallow structures, but the later are generally easier to train and more stable.

For convex models such as logistic regression, the training problem is also convex when the data representation is fixed.

The separation between the representation and the model learning is a key ingredient for the model stability.

When the representation is learned simultaneously, for instance with dictionary learning or with EM algorithms, the problem often become non-convex.

But this coupling between the representation and the model is critical for end-to-end models.

For instance, showed that for networks trained using pretraining, the fine-tuning step -where all the layers are trained together -improves the performances of the network.

This shows the importance of the adaptation of the representation to the task in end-to-end models.

Our contribution in this chapter is an additional training step which improves the use of the representation learned by the network to solve the considered task.

This new step is called post-training.

It is based on the idea of separating representation learning and statistical analysis and it should be used after the training of the network.

In this step, only the specific layers are trained.

Since the general layers -which encode the data representation -are fixed, this step focuses on finding the best usage of the learned representation to solve the desired task.

In particular, we chose to study the case where only the last layer is trained during the post-training, as this layer is the most specific one BID22 .

In this setting, learning the weights of the last layer corresponds to learning the weights for the kernel associated to the feature map given by the previous layers.

The post-training scheme can thus be interpreted in light of different results from kernel theory.

To summarize our contributions:??? We introduce a post-training step, where all layers except the last one are frozen.

This method can be applied after any traditional training scheme for deep networks.

Note that this step does not replace the end-to-end training, which co-adapts the last layer representation with the solver weights, but it makes sure that this representation is used in the most efficient way for the given task.??? We show that this post-training step is easy to use, that it can be effortlessly added to most learning strategies, and that it is computationally inexpensive.??? We highlight the link existing between this method and the kernel techniques.

We also show numerically that the previous layers can be used as a kernel map when the problem is small enough.??? We experimentally show that the post-training does not overfit and often produces improvement for various architectures and data sets.

The rest of this article is organized as follows: Section 2 introduces the post-training step and discusses its relation with kernel methods.

Section 4 presents our numerical experiments with multiple neural network architectures and data sets and Section 5 discusses these results.

In this section, we consider a feedforward neural network with L layers, where X 1 , . . .

, X L denote the input space of the different layers, typically R d l with d l > 0 and Y = X L+1 the output space of our network.

Let ?? l : X l ??? X l+1 be the applications which respectively compute the output of the l-th layer of the network, for 1 ??? l ??? L, using the output of the l???1-th layer and ?? L = ?? L ??? ?? ?? ?? ??? ?? 1 be the mapping of the full network from X 1 to Y .

Also, for each layer l, we denote W W W l its weights matrix and ?? l its activation function.

The training of our network is done using a convex and continuous loss function : Y ?? Y ??? R + .

The objective of the neural network training is to find weights parametrizing ?? L that solves the following problem: DISPLAYFORM0 , drawn from this input distribution.

Using these notations, the training objective (1) can then be rewritten DISPLAYFORM1 This reformulation highlights the special role of the last layer in our network compared to the others.

When ?? L???1 is fixed, the problem of finding W W W L is simple for several popular choices of activation function ?? L and loss .

For instance, when the activation function ?? L is the softmax function and the loss is the cross entropy, (2) is a multinomial logistic regression.

In this case, training the last layer is equivalent to a regression of the labels y using the embedding of the data x in X L by the mapping ?? L???1 .

Since the problem is convex in W W W L (see Appendix A), classical optimization techniques can efficiently produce an accurate approximation of the optimal weights W W W L -and this optimization given the mapping ?? L???1 is the idea behind post-training.

Indeed, during the regular training, the network tries to simultaneously learn suitable representation for the data in the space X L through its L ??? 1 first layer and the best use of this representation with W W W L .

This joint minimization is a strongly non-convex problem, therefore resulting in a potentially sub-optimal usage of the learned data representation.

The post-training is an additional step of learning which takes place after the regular training and proceeds as follows :1.

Regular training: This step aims to obtain interesting features to solve the initial problem, as in any usual deep learning training.

Any training strategy can be applied to the network, optimizing the empirical loss DISPLAYFORM2 The stochastic gradient descent explores the parameter space and provides a solution for ?? L???1 and W W W L .

This step is non restrictive: any type of training strategy can be used here, including gradient bias reduction techniques, such as Adagrad BID2 , or regularization strategies, for instance using Dropout BID1 .

Similarly, any type of stopping criterion can be used here.

The training might last for a fixed number of epochs, or can stop after using early stopping BID16 .

Different combinations of training strategies and stopping criterion are tested in Section 4.2.

Post-training: During this step, the first L ??? 1 layers are fixed and only the last layer of the network, ?? L , is trained by minimizing over W W W L the following problem DISPLAYFORM3 where?? (x, y) := (?? L (x), y) .

This extra learning step uses the mapping ?? L???1 as an embedding of the data in X L and learn the best linear predictor in this space.

This optimization problem takes place in a significantly lower dimensional space and since there is no need for back propagation, this step is computationally faster.

To reduce the risk of overfitting with this step, a 2 -regularization is added.

FIG0 illustrates the post-training step.

We would like to emphasize the importance of the 2 -regularization used during the post-training (4).

This regularization is added regardless of the one used in the regular training, and for all the network architectures.

The extra term improves the strong convexity of the minimization problem, making post-training more efficient, and promotes the generalization of the model.

The choice of the 2 -regularization is motivated from the comparison with the kernel framework discussed in Section 3 and from our experimental results.

Remark 1 (Dropout.).

It is important to note that Dropout should not be applied on the previous layers of the network during the post-training, as it would lead to changes in the feature function ?? L???1 .

In this section, we show that for the case where DISPLAYFORM0 can be approximated using kernel methods.

We define the kernel k as follows, DISPLAYFORM1 Then k is the kernel associated with the feature function ?? L???1 .

It is easy to see that this kernel is continuous positive definite and that for DISPLAYFORM2 belongs by construction to the Reproducing Kernel Hilbert Space (RKHS) H k generated by k. The post-training problem (4) is therefore related to the problem posed in the RKHS space H k , defined by DISPLAYFORM3 This problem is classic for the kernel methods.

With mild hypothesis on?? , the generalized representer theorem can be applied BID20 .

As a consequence, there exists ?? * ??? R N such that DISPLAYFORM4 Rewriting FORMULA8 with g * of the form (5), we have that g * = g W W W * , with DISPLAYFORM5 We emphasize that W W W * gives the optimal solution for the problem (6) and should not be confused with W W W * L , the optimum of (4).

However, the two problems differ only in their regularization, which are closely related (see the next paragraph).

Thus W W W * can thus be seen as an approximation of the optimal value W W W * L .

It is worth noting that in our experiments, W W W * appears to be a nearly optimal estimator of W W W * L (see Subsection 4.3).Relation between ?? H and ?? 2 .

The problems (6) and (4) only differ in the choice of the regularization norm.

By definition of the RKHS norm, we have DISPLAYFORM6 Consequently, we have that g W H ??? W 2 , with equality when Vect(?? L???1 (X 1 )) spans the entire space X L .

In this case, the norm induced by the RKHS is equal to the 2 -norm.

This is generally the case, as the input space is usually in a far higher dimensional space than the embedding space, and since the neural network structure generally enforces the independence of the features.

Therefore, while both norms can be used in (4), we chose to use the 2 -norm for all our experiments as it is easier to compute than the RKHS norm.

Close-form Solution.

In the particular case where (y 1 , y 2 ) = y 1 ??? y 2 2 and f (x) = x, (6) can be reduced to a classical Kernel Ridge Regression problem.

In this setting, W * can be computed by combining FORMULA9 and DISPLAYFORM7 where DISPLAYFORM8 Y is the matrix of the output data y 1 , . . .

, y N and I I I N is the identity matrix in R N .

This result is experimentally illustrated in Subsection 4.3.

Although data sets are generally too large for (8) to be computed in practice, it is worth noting that some kernel methods, such as Random Features BID18 , can be applied to compute approximations of the optimal weights during the post-training.

Multidimensional Output.

Most of the previously discussed results related to kernel theory hold for multidimensional output spaces, i.e. dim(X L+1 ) = d > 1, using multitask or operator valued kernels BID9 .

Hence the previous remarks can be easily extended to multidimensional outputs, encouraging the use of post-training in most settings.

This section provides numerical arguments to study post-training and its influence on performances, over different data sets and network architectures.

All the experiments were run using python and Tensorflow.

The code to reproduce the figures is available online 1 .

The results of all the experiments are discussed in depth in Section 5.

The post-training method can be applied easily to feedforward convolutional neural network, used to solve a wide class of real world problems.

To assert its performance, we apply it to three classic benchmark datsets: CIFAR10 BID11 , MNIST and FACES BID5 .CIFAR10.

This data set is composed of 60, 000 images 32 ?? 32, representing objects from 10 classes.

We use the default architecture proposed by Tensorflow for CIFAR10 in our experiments, based on the original architecture proposed by BID11 .

It is composed of 5 layers described in FIG1 .

The first layers use various common tools such as local response normalization (lrn), max pooling and RELU activation.

The last layer have a softmax activation function Test Error (-0.1) Figure 3 : Evolution of the performances of the neural network on the CIFAR10 data set, (dashed) with the usual training and (solid) with the post-training phase.

For the post-training, the value of the curve at iteration q is the error for a network trained for q ??? 100 iterations with the regular training strategy and then trained for 100 iterations with post-training.

The top figure presents the classification error on the training set and the bottom figure displays the loss cost on the test set.

The curves have been smoothed to increase readability.

and the chosen training loss was the cross entropy function.

The network is trained for 90k iterations, with batches of size 128, using stochastic gradient descent (SGD), dropout and an exponential weight decay for the learning rate.

Figure 3 presents the performance of the network on the training and test sets for 2 different training strategies.

The dashed line present the classic training with SGD, with performance evaluated every 100 iterations and the solid line present the performance of the same network where the last 100 iterations are done using post-training instead of regular training.

To be clearer, the value of this curve at iteration q is the error of the network, trained for q ??? 100 iterations with the regular training strategy, and then trained for 100 iterations with post-training.

The regularization parameter ?? for post-training is set to 1 ?? 10 ???3 .The results show that while the training cost of the network mildly increases due to the use of post-training, this extra step improves the generalization of the solution.

The gain is smaller at the end of the training as the network converges to a local minimum, but it is consistent.

Also, it is interesting to note that the post-training iterations are 4?? faster than the classic iterations, due to their inexpensiveness.

Additional Data Sets.

We also evaluate post-training on the MNIST data set (65000 images 27 ?? 27, with 55000 for train and 10000 for test; 10 classes) and the pre-processed FACES data set (400 images 64 ?? 64, from which 102400 sub-images, 32 ?? 32, are extracted, with 92160 for training and 10240 for testing; 40 classes).

For each data set, we train two different convolutional neural networks -to assert the influence of the complexity of the network over post-training:??? a small network, with one convolutional layer (5 ?? 5 patches, 32 channels), one 2 ?? 2 max pooling layer, and one fully connected hidden layer with 512 neurons,??? a large network, with one convolutional layer (5 ?? 5 patches, 32 channels), one 2 ?? 2 max pooling layer, one convolutional layer (5 ?? 5 patches, 64 channels), one 2 ?? 2 max pooling layer and one fully connected hidden layer with 1024 neurons.

We use dropout for the regularization, and set ?? = 1 ?? 10 ???2 .

We compare the performance gain resulting of the application of post-training (100 iterations) at different epochs of each of these networks.

The results are reported in TAB0 .As seen in TAB0 , post-training improves the test performance of the networks with as little as 100 iterations -which is negligible compared to the time required to train the network.

While the improvement varies depending on the complexity of the network, of the data set, and of the time spent training the network, it is important to remark that it always provides an improvement.

While the kernel framework developed in Section 2 does not apply directly to Recurrent Neural Network, the idea of post-training can still be applied.

In this experiment, we test the performances of post-training on Long Short-Term Memory-based networks (LSTM), using PTB data set BID14 .Penn Tree Bank (PTB).

This data set is composed of 929k training words and 82k test word, with a 10000 words vocabulary.

We train a recurrent neural network to predict the next word given the word history.

We use the architecture proposed by Zaremba et al. (2014) , composed of 2 layers of 1500 LSTM units with tanh activation, followed by a fully connected softmax layer.

The network is trained to minimize the average per-word perplexity for 100 epochs, with batches of size 20, using gradient descent, an exponential weight decay for the learning rate, and dropout for regularization.

The performances of the network after each epoch are compared to the results obtained if the 100 last steps (i.e. 100 batches) are done using post-training.

The regularization parameter for posttraining, ??, is set to 1 ?? 10 ???3 .

The results are reported in Figure 4 , which presents the evolution of the training and testing perplexity.

Similarly to the previous experiments, post-training improves the test performance of the networks, even after the network has converged.

In this subsection we aim to empirically evaluate the close-form solution discussed in Section 2 for regression tasks.

We set the activation function of the last layer to be the identity f L (x) = x, and consider the loss function to be the least-squared error (x, y) = x ??? y 2 2 in (1).

In in each experiment, (8) and FORMULA9 are used to compute W * for the kernel learned after the regular training of Test Perplexity (-80.5) Figure 4 : Evolution of the performances of the Recurrent network on the PTB data set.

The top figure presents the train perplexity and the bottom figure displays the test perplexity.

For the posttraining, the value of the curve at iteration q is the error for a network trained for q ??? 100 iterations with the regular training strategy and then trained for 100 iterations with post-training.the neural network, which learn the embedding ?? L???1 and an estimate W L .

In order to illustrate this result, and to compare the performances of the weights W * with respect to the weights W L , learned either with usual learning strategies or with post-training, we train a neural network on two regression problems using a real and a synthetic data set.

70% of the data are used for training, and 30% for testing.

Real Data Set Regression.

For this experiment, we use the Parkinson Telemonitoring data set BID21 .

The input consists in 5, 875 instances of 17 dimensional data, and the output are one dimensional real number.

For this data set, a neural network made of two fully connected hidden layers of size 17 and 10 with respectively tanh and RELU activation, is trained for 250, 500 and 750 iterations, with batches of size 50.

The layer weights are all regularized with the 2 -norm and a fixed regularization parameter ?? = 10 ???3 .

Then, starting from each of the trained networks, 200 iterations of post-training are used with the same regularization parameter ?? and the performances are compared to the closed-form solutions computed using (8) for each saved network.

The results are presented in TAB1 .Simulated Data Set Regression.

For this experiment, we use a synthetic data set.

The inputs were generated using a uniform distribution on 0, 1 10 .

The outputs are computed as follows: DISPLAYFORM0 where W 1 ??? ???1, 1 10??5 and W 2 ??? ???1, 1 5 are randomly generated using a uniform law.

In total, the data set is composed of 10, 000 pairs (x i , y j ).

For this data set, a neural network with two fully connected hidden layers of size 10 with activation tanh for the first layer and RELU for the second layer is trained for 250, 500 and 750 iterations, with batches of size 50.

We use the same protocol with 200 extra post-training iterations.

The results are presented in TAB1 .For these two experiments, the post-training improves the performances toward these of the optimal solution, for several choices of stopping times.

It is worth noting that the performance of the optimal solution is better when the first layers are not fully optimized with Parkinson Telemonitoring data set.

This effect denotes an overfitting tendency with the full training, where the first layers become overly specified for the training set.

The experiments presented in Section 4 show that post-training improves the performances of all the networks considered -including recurrent, convolutional and fully connected networks.

The gain is significant, regardless of the time at which the regular training is stopped and the posttraining is done.

In both the CIFAR10 and the PTB experiment, the gap between the losses with and without post-training is more pronounced if the training is stopped early, and tends to be smaller as the network converges to a better solution (see Figure 4 and Figure 3 ).

The reduction of the gap between the test performances with and without post-training is made clear in TAB0 .

For the MNIST data set, with a small-size convolutional neural network, while the error rate drops by 1.5% when post-training is applied after 5000 iterations, this same error rate only drops by 0.2% when it is applied after 20000 iterations.

This same observation can be done for the other results reported in TAB0 .

However, while the improvement is larger when the network did not fully converge prior to the post-training, it is still significant when the network has reached its minimum: for example in PTB the final test perplexity is 81.7 with post-training and 82.4 without; in CIFAR10 the errors are respectively 0.147 and 0.154.If the networks are allowed to moderately overfit, for instance by training them with regular algorithm for a very large number of iterations, the advantage provided by post-training vanishes: for example in PTB the test perplexity after 2000 iterations (instead of 400) is 83.2 regardless of posttraining.

This is coherent with the intuition behind the post-training: after overfitting, the features learned by the network become less appropriate to the general problem, therefore their optimal usage obtained by post-training no longer provide an advantage.

It is important to note that the post-training computational cost is very low compared to the full training computations.

For instance, in the CIFAR10 experiment, each iteration for post-training is 4?? faster on the same GPU than an iteration using the full gradient.

Also, in the different experiments, post-training produces a performance gap after using as little as 100 batches.

There are multiple reasons behind this efficiency: first, the system reaches a local minimum relatively rapidly for post-training as the problem FORMULA3 has a small number of parameters compared to the dimensionality of the original optimization problem.

Second, the iterations used for the resolution of (4) are computationally cheaper, as there is no need to chain high dimensional linear operations, contrarily to regular backpropagation used during the training phase.

Finally, since the post-training optimization problem is generally convex, the optimization is guaranteed to converge rapidly to the optimal weights for the last layer.

Another interesting point is that there is no evidence that the post-training step leads to overfitting.

In CIFAR10, the test error is improved by the use of post-training, although the training loss is similar.

The other experiments do not show signs of overfitting either as the test error is mostly improved by our additional step.

This stems from the fact that the post-training optimization is much simpler than the original problem as it lies in a small-dimensional space -which, combined with the added 2 -regularization, efficiently prevents overfitting.

The regularization parameter ?? plays an important role in post-training.

Setting ?? to be very large reduces the explanatory capacity of the networks whereas if ?? is too small, the capacity can become too large and lead to overfitting.

Overall, our experiments highlighted that the post-training produces significant results for any choice of ?? reasonably small (i.e 10 ???5 ??? ?? ??? 10 ???2 ).

This parameter is linked to the regularization parameter of the kernel methods, as stated in Section 3.Overall, these results show that the post-training step can be applied to most trained networks, without prerequisites about how optimized they are since post-training does not degrade their performances, providing a consistent gain in performances for a very low additional computational cost.

In Subsection 4.3, numerical experiments highlight the link between post-training and kernel methods.

As illustrated in TAB1 , using the optimal weights derived from kernel theory immediately a performance boost for the considered network.

The post-training step estimate numerically this optimal layer with the gradient descent optimizer.

However, computing the optimal weights for the last layer is only achievable for small data set due to the required matrix inversion.

Moreover, the closed form solution is known only for specific problems, such as kernelized least square regression.

But post-training approaches the same performance in these cases solving (4) with gradient-based methods.

The post-training can be linked very naturally to the idea of pre-training, developed notably by BID12 , and BID5 .

The unsupervised pre-training of a layer is designed to find a representation that captures enough information from the data to be able to reconstruct it from its embedding.

The goal is thus to find suitable parametrization of the general layers to extract good features, summarizing the data.

Conversely, the goal of the posttraining is, given a representation, to find the best parametrization of the last layer to discriminate the data.

These two steps, in contrast with the usual training, focus on respectively the general or specific layers.

In this work, we studied the concept of post-training, an additional step performed after the regular training, where only the last layer is trained.

This step is intended to take fully advantage of the data representation learned by the network.

We empirically shown that post-training is computationally inexpensive and provide a non negligible increase of performance on most neural network structures.

While we chose to focus on post-training solely the last layer -as it is the most specific layer in the network and the resulting problem is strongly convex under reasonable prerequisites -the relationship between the number of layers frozen in the post-training and the resulting improvements might be an interesting direction for future works.

Wojciech Zaremba, Ilya Sutskever, and Oriol Vinyals.

Recurrent neural network regularization.arXiv preprint, arXiv:1409(2329), 2014.

We show here, for the sake of completeness, that the post-training problem is convex for the softmax activation in the last layer and the cross entropy loss.

This result is proved showing that the hessian of the function is positive semidefinite, as it is a diagonally dominant matrix.

Proposition 2 (convexity).

???N, M ??? N, ???X ??? R N , ???j ??? 1, M , the following function F is convex: DISPLAYFORM0 ?? ij log exp(XW i ) .where ?? is the Dirac function, and W i denotes the i-th row of a W .

Proof 1.

Let DISPLAYFORM1 .

DISPLAYFORM2 we have DISPLAYFORM3 ???W m,n ???W p,q = x n ???P m ???W p,q , = x n x q P m (W ) ?? m,p ??? P p (W ) .

where ??? is the Kronecker product, and the matrix P(W ) is defined by P m,p = P m (W ) ?? m,p ??? P p (W ) .

Now since ???1 ??? m ??? M , and thus P(W ) is positive semidefinite.

Since XX T is positive semidefinite too, their Kronecker product is also positive semidefinite, hence the conclusion.

@highlight

We propose an additional training step, called post-training, which computes optimal weights for the last layer of the network.