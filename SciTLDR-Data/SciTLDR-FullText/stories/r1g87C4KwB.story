Understanding the optimization trajectory is critical to understand training of deep neural networks.

We show how the hyperparameters of stochastic gradient descent influence the covariance of the gradients (K) and the Hessian of the training loss (H) along this trajectory.

Based on a theoretical model, we predict that using a high learning rate or a small batch size in the early phase of training leads SGD to regions of the parameter space with (1) reduced spectral norm of K, and (2) improved conditioning of K and H. We show that the point on the trajectory after which these effects hold, which we refer to as the break-even point, is reached early during training.

We demonstrate these effects empirically for a range of deep neural networks applied to multiple different tasks.

Finally, we apply our analysis to networks with batch normalization (BN) layers and find that it is necessary to use a high learning rate to achieve loss smoothing effects attributed previously to BN alone.

The choice of the optimization method implicitly regularizes deep neural networks (DNNs) by influencing the optimization trajectory in the loss surface (Neyshabur, 2017; Arora, 2019) .

In this work, we theoretically and empirically investigate how the learning rate and the batch size used at the beginning of training determine properties of the entire optimization trajectory.

Figure 1: Visualization of the early part of the training trajectory on CIFAR-10 (before reaching 65% training accuracy) of a simple CNN model (see Sec. 4 for details) optimized using SGD with learning rate η = 0.1 (red) and η = 0.01 (blue).

Each model, shown as a point on the trajectory, is represented by its test predictions embedded into a two-dimensional space using UMAP.

The background color indicates the spectral norm of K (left) and the training accuracy (right).

Depending on η, after reaching what we call the break-even point, trajectories are steered towards regions characterized by different K (left) for the same training accuracy (right).

See Sec. 4.1 for details.

We focus our analysis on two objects that quantify different properties of the optimization trajectory: the covariance of gradients (K) 1 , and the Hessian of the training loss (H).

The matrix K quantifies noise induced by noisy estimate of the full-batch gradient, and has been linked to the generalization error (Roux et al., 2008; .

The matrix H describes the curvature of the loss surface.

Better conditioning of H has been attributed as the main reason behind the efficacy of batch normalization (Bjorck et al., 2018; Ghorbani et al., 2019) .

Our first and main contribution is predicting, and empirically demonstrating two effects in the early phase of training influenced by the choice of the hyperparameters in stochastic gradient descent (SGD): (1) reduced spectral norms of K and H and (2) improved conditioning of K and H. These effects manifest themselves after a certain point on the optimization trajectory, to which we refer to as the break-even point.

See Fig. 1 for an illustration of this phenomenon.

We make our predictions based on a theoretical model of the initial phase of training, which incorporates recent observations on the instability and oscillations in the parameter space that characterize the learning dynamics of neural networks (Masters & Luschi, 2018; Xing et al., 2018; Lan et al., 2019) .

As our second contribution, we apply our analysis to a network with batch normalization (BN) layers and find that our predictions are valid in this case too.

Delving deeper in this direction of investigation, we show that using a large learning rate is necessary to reach better-conditioned, relatively to a network without BN layers, regions of the loss surface, which was previously attributed to BN alone (Bjorck et al., 2018; Ghorbani et al., 2019; Page, 2019) .

Learning dynamics and the early phase of training.

Our theoretical model is motivated by recent work on the learning dynamics of neural networks (Goodfellow et al., 2014; Masters & Luschi, 2018; Yao et al., 2018; Xing et al., 2018; Jastrzebski et al., 2018; Lan et al., 2019) .

We are directly inspired by Xing et al. (2018) ; Jastrzebski et al. (2018) ; Lan et al. (2019) who show that training oscillates, which can be characterized by a negative cosine between consecutive steps.

Our theoretical approach is closely related to .

While they study how SGD selects a minimum from a stability perspective, we apply their methodology to the early phase of training.

Based on their theoretical results and other recent studies, we make predictions about the whole training trajectory.

In our work we argue that the initial phase of training determines the rest of the trajectory.

This is directly related to Erhan et al. (2010) ; Achille et al. (2017) who show the existence of the critical period of learning.

Erhan et al. (2010) argue that training, unless pre-training is used, is sensitive to shuffling of examples in the first epochs of training.

Achille et al. (2017) ; Golatkar et al. (2019) ; Sagun et al. (2017) ; Keskar et al. (2017) demonstrate that adding regularization in the beginning of training affects the final generalization disproportionately more compared to doing so later.

The covariance of the gradients and the Hessian.

The covariance of the gradients, which we denote by K, encapsulates the geometry and magnitude of variation in gradients across different samples (Thomas et al., 2019) .

The matrix K was related to the generalization error in Roux et al. (2008) .

A similar quantity, cosine alignment between gradients computed on different examples, was recently shown to explain some aspects of deep networks generalization .

The second object that we study is the Hessian that quantifies the loss surface shape (LeCun et al., 2012) .

Recent work has shown that the largest eigenvalues of H grow quickly in the early phase of training, and then stabilize at a value dependent on the learning rate and the batch size (Keskar et al., 2017; Sagun et al., 2017; Fort & Scherlis, 2019; Jastrzebski et al., 2018) .

The Hessian can be decomposed into a sum of two terms, where the dominant term (at least at the end of training) is the uncentered covariance of gradients G (Sagun et al., 2017; Papyan, 2019) .

While we study K, the centered version of G, K and G are typically similar due to the dominance of noise in training (Zhu et al., 2018; Thomas et al., 2019) .

is the gradient of the training loss L with respect to θ on xi, N is the number of training examples, and g is the full-batch gradient.

Implicit regularization induced by optimization method.

Multiple prior works study the regularization effects that are attributed only to the optimization method (Neyshabur, 2017) .

A popular research direction is to bound the generalization error based on the properties of the final minimum such as the norm of the parameter vector or the Hessian (Bartlett et al., 2017; Keskar et al., 2017) .

Perhaps the most related work to ours is Arora, 2019) .

They suggest it is necessary to study the optimization trajectory to understand generalization of deep networks.

In this vein, but in contrast to most of the previous work, we focus (1) on the implicit regularization effects that can be attributed to the GD dynamics at the beginning of training, and (2) on the covariance of gradients.

In this section we make two conjectures about the optimization trajectory induced by SGD based on a theoretical model of the learning dynamics in the early stage of training.

Definitions.

Let us denote the loss on an example (x, y) by L(x, y; θ), where θ is a D-dimensional parameter vector.

A key object we study is the Hessian of the training loss (H).

The second key object we study is the covariance of the gradients

is the gradient of L with respect to θ calculated on i-th example, N is the number of training examples, and g is the full-batch gradient.

We denote the i-th normalized eigenvector and eigenvalue of a matrix A by e i A and λ i A .

Both H and K are computed at a given θ, but we omit this dependence in the notation.

Let t index steps of optimization, and let θ(t) denote the parameter vector at optimization step t.

are the smallest nonzero eigenvalues of H and K, respectively.

Furthermore, the maximum attained values of Tr(K), Tr(H) are smaller for a larger learning rate or a smaller batch size.

We consider non-zero eigenvalues in Con.

2, because K has N − D non-zero eigenvalues, where N is the number of training points.

Both conjectures are only valid for learning rates and batch sizes in the range that guarantee that training is stable after reaching the break-even point.

It is also important to stress that while our predictions are related to the curvature of the final minimum Jastrzebski et al., 2017) , in contrast to prior work we are interested in the properties of the whole training trajectory.

Finally, from the optimization perspective, the effects discussed above are desirable; many papers in the optimization literature focused on either reducing variance of the mini-batch gradients (Johnson & Zhang, 2013) , or on improving conditioning (Roux et al., 2008) .

2 We include in App.

A a similar argument for the opposite case.

In this section we first analyse learning dynamics in the early phase of training.

Next, we empirically validate the two conjectures.

In the final part we extend our analysis to a neural network with batch normalization layers.

Due to the space constraint we take the following approach to reporting results.

In the main body of the paper, we focus on the CIFAR-10 dataset (Krizhevsky, 2009) and the IMDB dataset (Maas et al., 2011) , to which we apply three architectures: a vanilla CNN (SimpleCNN) following Keras example (Chollet et al., 2015) , ResNet-32 (He et al., 2015) , and LSTM (Hochreiter & Schmidhuber, 1997) .

We also validate the two conjectures for DenseNet (Huang et al., 2016) on the ImageNet (Deng et al., 2009 ) dataset, BERT (Devlin et al., 2018) fine-tuned on the MNLI dataset (Williams et al., 2017) , and a multi-layer perceptron on the FashionMNIST dataset (Xiao et al., 2017) .

These results and all experimental details are in App.

C. (2019), we estimate the top eigenvalues and eigenvectors of H using the Lanczos algorithm on a random subset of 5% of the training set on CIFAR-10.

We estimate the top eigenvalues and eigenvectors of K using (in most cases) batch size of 128 and approximately 5% of the training set on CIFAR-10.

Furthermore, under the assumption that noise due to sampling data is normally distributed, our approximation biases the smallest eigenvalue towards 0.

To counteract this, we use the 5 th smallest non-zero eigenvalue in place of λ * H and λ * K .

We describe the procedure in more details, as well as examine the impact of using mini-batching in computing the eigenspectrum of K, in App.

B.

In this section we examine the learning dynamics in the early phase of training.

Our goal is to verify some of the assumptions made in Sec. 3.

We analyse the evolution of λ H .

This is expected: λ 1 H decays to 0 when the mean loss decays to 0 for cross entropy loss (Martens, 2016) .

Does training become increasingly unstable in the early phase of training?

A key aspect of our model is that an increase of λ 1 K and λ 1 H translates into a decrease in stability, which we formalized as stability along e 1 H .

Computing stability directly along e 1 H is computationally expensive.

Instead, we measure a more tractable proxy.

At each iteration we measure the loss on the training set before, and after taking the step, which we denote as ∆L (a positive value indicates reduction of the training loss).

In Fig. 2 we observe that training becomes unstable when λ 1 K reaches the maximum value.

Visualizing the break-even point.

Finally, to understand the break-even point phenomenon better, we visualize the learning dynamics leading to reaching the break-even point in our model in Fig. 1 .

Following Erhan et al. (2010), we embed the test set predictions at each step of training of SimpleCNN, using UMAP (McInnes et al., 2018) .

Background color indicates λ 1 K (left) and the training accuracy at the closest iteration (right).

We observe that early in training the trajectory corresponding to a lower learning rate of η = 0.01 reaches regions of the loss surface corresponding to the same accuracy as η = 0.1, but characterized by larger λ 1 K .

Additionally, in Fig. 3 we plot the spectrum of K (left) and H at the iteration when λ K and λ H , respectively, reach the highest value.

We can observe that for the lower learning rate the distribution has more outliers.

Summary.

We have shown that the dynamics of the early phase of training is consistent with the assumptions made in our theoretical model.

That is, λ 1 K and λ 1 H increase approximately proportionally to each other, which is also correlated with a decrease of a proxy of stability.

Finally, we have shown qualitatively reaching the break-even point.

In this section we validate empirically Con.

1 and Con.

2 in three settings.

For each model we pick manually a suitable range of learning rates and batch sizes to ensure that the properties of K and H that we converged in a reasonable computational budget and training is stable.

We use 200 epochs on CIFAR-10 and 50 epochs on IMDB.

In ResNet-32 we remove batch normalization.

We summarize the results for SimpleCNN, ResNet-32, and LSTM in Fig. 4 and Fig. 5 .

Curves are smoothed with a moving average.

The training curves, as well as experiments for other architectures and datasets (including a DenseNet on ImageNet and BERT on MNLI) can be found in App.

D.

Null hypothesis.

A natural assumption is that the choice of η or S does not influence K and H along the optimization trajectory.

In particular, it is not self-evident that using a high η, or a small S, would steer optimization towards better conditioned regions of the loss surface.

Other experiments.

We report how λ 1 H depends on η and S for ResNet-32 and SimpleCNN in Fig. 5 .

We observe that the conclusions also apply to λ 1 H , which is consistent with experiments in Jastrzebski et al. (2018) .

We found the effect of η and S on λ * H /λ 1 H of η and S to be weaker.

This might be because, in contrast to K, we approximate the spectrum of H using only the top five eigenvalues (see App.

B for details).

In this section we have demonstrated the variance reduction (Conjecture 1) and the preconditioning effect (Conjecture 2) of SGD.

Furthermore, we have shown these effects occur early in training.

We also found that conclusions also apply to other settings including BERT on MNLI and DenseNet on ImageNet (see App.

D).

The loss surface of deep networks has been widely reported to be ill-conditioned, which is the key motivation behind using second order optimization methods in deep learning (LeCun et al., 2012; Martens & Grosse, 2015) .

Recently, Ghorbani et al. (2019) ; Page (2019) have argued that the key reason behind the efficacy of batch normalization (Ioffe & Szegedy, 2015) is improving conditioning of the loss surface.

Our Conjecture 2 is that using a high η or a low S results as well in improved conditioning.

A natural question that we investigate in this section is how the two phenomena are related.

Are the two conjectures valid in batch-normalized networks?

First, to investigate whether our conjectures hold in batch-normalized network, we run similar experiments as in Sec. 4.2 on a SimpleCNN model with batch normalization layers inserted after each convolutional layer (SimpleCNN-BN), using the CIFAR-10 dataset.

We test η ∈ {0.001, 0.01, 0.1, 1.0} (η = 1.0 leads to divergence of SimpleCNN without BN).

3 We summarize the results in Fig. 6 .

The evolution of λ BN requires using a high learning rate.

As our conjectures hold for BN network, a natural question is if learning can be ill-conditioned with a low learning rate even when BN is used.

Ghorbani et al. (2019) show that without BN, mini-batch gradients are largely contained in the subspace spanned by the top eigenvectors of noncentered K. To answer this question we track g / g 5 , where g denotes the mini-batch gradient, and g 5 denotes the mini-batch gradient projected onto the top 5 eigenvectors of K. A value of g / g 5 close to 1 implies that the mini-batch gradient is mostly contained in the subspace spanned by the top 5 eigenvectors of K.

We compare two settings: (1) SimpleCNN-BN optimized with η = 0.001, and (2) SimpleCNN optimized with η = 0.01.

We make three observations.

First, the maximum (minimum) value of g / g 5 is 1.90 (1.37) and 1.88 (1.12), respectively.

Second, the maximum value of λ 1 K is 10.3 and 16, respectively.

Finally, λ * K /λ 1 K reaches 0.368 in the first setting, and 0.295 in the second setting.

Comparing these differences to differences that are induced by using the highest η = 1.0 in SimpleCNN-BN, we can conclude that using a large learning rate is necessary to observe the effect of loss smoothing which was previously attributed to BN alone (Ghorbani et al., 2019; Page, 2019; Bjorck et al., 2018) .

This might be directly related to the result that a high learning rate is necessary to achieve good generalization when using BN (Bjorck et al., 2018) .

Summary.

We have shown that the effects of changing the learning rate described in Con.

1 and Con.

2 also hold for a network with batch normalization layers, and that using a high learning rate is necessary in a batch-normalized network to improve conditioning of the loss surface, compared to conditioning of the loss surface in the same network without batch-normalization.

Based on a theoretical model, we conjectured and empirically argued for the existence of the breakeven point on the optimization trajectory induced by SGD.

Next, we demonstrated that using a high learning rate or a small batch size in SGD has two effects on K and H along the trajectory that we referred to as (1) variance reduction and (2) pre-conditioning.

There are many potential implications of the existence of the break-even point.

We investigated one in particular, and demonstrated that using a high learning rate is necessary to achieve the loss smoothing effects previously attributed to batch normalization alone.

Additionally, the break-even occurs typically early during training, which might be related to the recently discovered phenomenon of the critical learning period in training of deep networks (Achille et al., 2017; Golatkar et al., 2019) .

We plan to investigate this connection in the future.

In this section we will formally state and prove the theorem used informally in Sec. 3.

With the definitions introduced in Sec. 3 in mind, we state the following: Theorem 1.

Assuming that training is stable along e 1 H (t) at t = 0, then λ 1 H (t) and λ 1 K (t) at which SGD becomes unstable along e 1 H (t) are smaller for a larger η or a smaller S.

Proof.

This theorem is a straightforward application of Theorem 1 from to the early phase of training.

First, let us consider two optimization trajectories corresponding to using two different learning rates η 1 and η 2 (η 1 > η 2 ).

For both trajectories, Theorem 1 of states that SGD is stable at an iteration t if the following inequality is satisfied:

where

2 .

Using this we can rewrite inequality (2) as:

At the iteration t * at which training becomes unstable along e

where α denotes the proportionality constant from Assumption 2.

Note that if n = S the right hand side degenerates to

2 ) the value of λ 1 H at which training becomes unstable along e 1 H for η 1 and η 2 , respectively.

Similarly, let us denote by ψ(t * 1 ) and ψ(t * 2 ) the value of ψ at which training becomes unstable.

2 ).

A necessary condition for this inequality to hold is that ψ(t * 1 ) > ψ(t * 2 ).

However, ψ(t) by Assumption 4 decreases with increasing λ 1 H , which corresponds to reducing the distance to the minimizer along e 1 H .

Hence, that would imply λ 1

2 ).

Repeating the same argument for batch size completes the proof.

It is also straightforward to extend the argument to the case when training is initialized at an unstable region along e 1 H (0), which we formalize as follows.

Theorem 2.

If training is unstable along e 1 H (t) at t = 0, then λ 1 H (t) and λ 1 K (t) at which SGD becomes for the first time stable along e 1 H (t) are smaller for a larger η or a smaller S.

Proof.

We will use a similar argument as in the proof of Th.

1. Let us consider two optimization trajectories corresponding to using two different learning rates η 1 and η 2 (η 1 > η 2 ).

2 ) denote the spectral norm of H at the iteration at which training is for the first time stable along e 1 H for η 1 and η 2 , respectively.

Following the steps in the proof of Th.

1 we get that

1

2 ) has reached a smaller value, it means ψ(t * 2 ) has increased to a larger value, i.e. ψ(t * 2 ) > ψ(t * 1 ).

However, a necessary condition for λ 1

2 ) inequality to hold is that ψ(t * 1 ) > ψ(t * 2 ), which leads to a contradiction.

Repeating the same argument for batch size completes the proof.

A small subset of the training set approximates well the largest eigenvalues of the true Hessian on the CIFAR-10 dataset (Alain et al., 2019) .

This might be due to the hierarchical structure of the Hessian (Papyan, 2019; Fort & Ganguli, 2019) ; the largest eigenvalues are connected to the class structure in the data.

Following Alain et al. (2019) we use approximately the same 5% fraction of the dataset in CIFAR-10 experiments, and SCIPY Python package.

Evaluating the eigenspace of K is perhaps less common in deep learning.

We sample L mini-batch gradient of size M .

Then following Papyan (2019) we compute the corresponding Gram matrixK that has entriesK ij = g i −g, g j −g , where g is the full batch gradient.

Note thatK ij is only L×L dimensional.

It can be shown that in expectation of mini-batchesK has the same eigenspectrum as K.

In all experiments we fix L = 25.

When comparing different batch sizes we use M = 128, otherwise we use the same batch size as the one used to train the model.

For instance on the CIFAR-10 dataset this amounts to using approximately 5% of the training set.

To compute Tr(K) we compute the trace ofK. Due to large computational cost of estimating top eigenvalues using the Lanczos algorithm, we approximate in most experiments the Tr(H) using only the top 5 eigenvalues of H.

A natural question is whether using M = 128 approximates well the underlying K. To investigate this we compare λ

In this section we describe all the details for experiments in the main text and in the Appendix.

ResNet-32 on CIFAR The model trained is the ResNet-32 (He et al., 2015) .

The network is trained for 200 epochs with a batch size equal to 128.

The dataset used is CIFAR-10.

Standard data augmentation and preprocessing is applied.

The default learning rate is 0.05, and the default batch size is 128.

Weight decay 0.0001 is applied to all convolutional layers.

SimpleCNN on CIFAR The network used is a simple convolutional network based on example from Keras repository (Chollet et al., 2015) .

First, the data is passed through two convolutional layers with 32 filters, both using same padding, ReLU activations, and the kernel size of 3x3.

Next, the second pair of layers is used with 64 filters (the rest of parameters are the same).

Each pair ends with a max-pooling layer with a window of size 2.

Before the classification layer, a densely connected layer with 128 units is used.

The data follows the same scheme as in the ResNet-32 experiment.

The default learning rate is 0.05, and the default batch size is 128.

BERT on MNLI The model used in this experiment is BERT-base (Devlin et al., 2018) , pretrained on multilingual data 4 .

The model is trained on MultiNLI dataset (Williams et al., 2018) with the maximum sentence length equal to 40.

The network is trained for 20 epochs with a batch size of 32.

MLP on FMNIST This experiment is using a Multi Layer Perceptron network with two hidden layers of size 300 and 100, both with ReLU activations.

The data is normalized to the [0, 1] range and no augmentation is being used.

The network is trained with a batch size of 64 for 200 epochs.

LSTM on IMDB The network used in this experiment consists of an embedding layer followed by an LSTM with 100 hidden units.

We use vocabulary size of 20000 words and the maximum length of the sequence equal to 80.

The model is trained for 100 epochs with a batch size of 128.

The network used is the DenseNet-121 (Huang et al., 2016) .

The dataset used is the ILSVRC 2012 (Russakovsky et al., 2015) .

The images are centered, but no augmentation is being used.

Due to large computational cost, the network is trained for 10 epochs with a batch size of 32.

Neither dropout nor weight decay is used for training.

In this section we repeat experiments from Sec. 4.2 in various other settings, as well as include additional data from settings already included in the main text.

ResNet-32 on CIFAR-10.

In Fig. 8 and Fig. 9 we report accuracy on the training set and the validation set, λ 1 H , and Tr(K) for all runs on the ResNet-32 model and the CIFAR-10 dataset .

SimpleCNN on CIFAR-10.

In Fig. 8 and Fig. 9 we report accuracy on the training set and the validation set, λ LSTM on IMDB.

In Fig. 12 and Fig. 13 we report accuracy on the training set and the validation set, λ 1 H , and Tr(K) for all runs on the LSTM model and the IMDB dataset.

BERT on MNLI.

In Fig. 14 and Fig. 15 we report results for the BERT model on the MNLI dataset.

MLP on FMNIST.

In Fig. 16 and Fig. 17 we report results for the MLP model on the FMNIST dataset.

DenseNet on ImageNet.

In Fig. 18 and Fig. 19 we report results for the DenseNet-121 model on the ImageNet dataset.

@highlight

In the early phase of training of deep neural networks there exists a "break-even point" which determines properties of the entire optimization trajectory.

@highlight

This work analyzes the optimization of deep neural networks by considering how the batch size and step-size hyper-parameters modify learning trajectories.