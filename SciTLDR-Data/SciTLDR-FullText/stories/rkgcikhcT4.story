We extend the recent results of (Arora et al., 2019) by a spectral analysis of representations corresponding to kernel and neural embeddings.

They showed that in a simple single layer network, the alignment of the labels to the eigenvectors of the corresponding Gram matrix determines both the convergence of the optimization during training as well as the generalization properties.

We generalize their result to kernel and neural representations and show that these extensions improve both optimization and generalization of the basic setup studied in (Arora et al., 2019).

The well-known work of BID8 highlighted intriguing experimental phenomena about deep net trainingspecifically, optimization and generalization -and called for a rethinking of generalization in statistical learning theory.

In particular, two fundamental questions that need understanding are: Optimization.

Why do true labels give faster convergence rate than random labels for gradient descent?

Generalization.

What property of properly labeled data controls generalization?

BID0 have recently tried to answer this question in a simple model by conducting a spectral analysis of the associated Gram matrix.

They show that both training and generalization are better if the label vector aligns with the top eigenvectors.

However, their analysis applies only to a simple two layer network.

How could their insights be extended to deeper networks?A widely held intuitive view is that deep layers generate expressive representations of the raw input data.

Adopting this view, one may consider a model where a representation generated by successive neural network layers is viewed as a kernel embedding which is then fed into the two-layer model of BID0 .

The connection between neural networks and kernel machines has long been studied; BID2 ) introduced kernels that mimic deep networks and BID6 showed kernels equivalent to certain feed-forward neural networks.

Recently, BID1 ) also make the case that progress on understanding deep learning is unlikely to move forward until similar phenomena in classical kernel machines are recognized and understood.

Very recently, BID4 showed that the evolution of a neural network during training can be related to a new kernel, the Neural Tangent Kernel (NTK) which is central to describe the generalization properties of the network.

Here we pursue this approach by studying the effect of incorporating embeddings in the simple two layer model and we perform a spectral analysis of these embeddings along the lines of BID0 .

We can obtain embeddings in several ways: i. We can use an unbiased kernel such as Gaussian kernel.

This choice is consistent with the maximum entropy principle and makes no prior assumption about the data.

Or use a kernel which mimics or approximates deep networks ii.

We could use data driven embeddings explicitly produced by the hidden layers in neural networks: either use a subset of the same training data to compute such an embedding, or transfer the inferred embedding from a different (but similar) domain.

While a general transformation g(x) of the input data may have arbitrary effects, one would expect kernel and neural representations to improve performance.

The interplay of kernels and data labellings has been addressed before, for example in the work of kernel-target alignment BID3 )

.We do indeed observe a significant beneficial effect: Optimization.

Using kernel methods such as random Fourier features (RFF) to approximate the Gaussian kernel embedding BID5 and neural embeddings, we obtain substantially better convergence in training.

Generalization.

We also achieve significantly lower test error and we confirm that the data dependent spectral measure introduced in BID0 significantly improves with kernel and neural embeddings.

Thus this work shows empirically that kernel and neural embeddings improve the alignment of target labels to the eigenvectors of the Gram matrix and thus help training and generalization.

This suggests a way to extend the insights of BID0 ) to deeper networks, and possible theoretical results in this direction.

Network model.

In BID0 , the authors consider a simple two layer network model: DISPLAYFORM0 DISPLAYFORM1 These can be written jointly as a = (a 1 , .., a m ) T and W = (w 1 , .., w m ).

This network is trained on dataset of datapoints {x i } and their targets {y i }.They provide a fine grained analysis of training and generalization error by a spectral analysis of the Gram matrix: BID0 show that both training and generalization are better if the label vector y aligns with the eigenvectors corresponding to the top eigvalues of H ∞ .

DISPLAYFORM2 The two-layer ReLU network in this work follows the general structure as in BID0 with the difference being the addition of an embedding φ at the input layer corresponding to a kernel K. The corresponding model is: DISPLAYFORM3 and let its eigenvalues be ordered as DISPLAYFORM4 A kernel K such that the corresponding eigenvectors align well with the labels would be expected to perform well both for training optimization as well as generalization.

This is related to kernel target alignment BID3 .

Optimization.

For the simple two layer network, BID0 show that the convergence of gradient descent is controlled by DISPLAYFORM5 For our kernelized network, the corresponding convergence is controlled by DISPLAYFORM6 Generalization.

For the simple two layer network, BID0 show that the generalization performance is controlled by DISPLAYFORM7 For our kernelized two layer network, the corresponding data and representation dependent measure is: DISPLAYFORM8

We perform our experiments on two commonly-used datasets for validating deep neural models, i.e., MNIST and CIFAIR-10.

These datasets are used for the experiments in BID0 .

As in their work we only look at the first two classes and set the label y i = +1 if image i belongs to the first class and y i = −1 if it belongs to the second class.

The images are normalized such that ||x i || 2 = 1.

This is also done for kernel embeddings such that ||φ(x i )|| 2 = 1.

The weights in equation (2) are initialized as follows: DISPLAYFORM0 We then use the following loss function to train the model to predict the image labels.

DISPLAYFORM1 For optimization, we use (full batch) gradient descent with the learning rate η.

In our experiments we set k = 10 −2 , η = 2 · 10 −4 similar to BID0 .

We first use the Gaussian kernel K(x i , x j ) := exp −γ x i − x j 2 .

The corresponding embedding is infinite dimensional, hence we consider the fast approximations to the kernel given by random Fourier features (RFF) BID5 .

The idea of random Fourier features is to construct an explicit feature map which is of a dimension much lower than the number of observations, but the resulting inner product approximates the desired kernel function.

We use γ = 1 in all our experiments.

Optimization.

We first investigate the use of Gaussian kernel for a more efficient optimization of the loss function on the training data.

FIG0 show the training loss at different steps respectively on MNIST and CIFAR-10 datasets.

We consistently observe that the different Gaussian kernels (specified by various dimensions of the kernel) yields faster convergence of the optimization procedure on both datasets.

MNIST is a simple dataset which gives incredibly high score almost immediately, as shown by the train loss FIG0 ) and by the accuracy on the test data (the table in FIG3 (c)) thus we will focus our analysis on the CIFAR-10 dataset.

Similar to the setup in BID0 , in FIG0 , for different methods, we plot the eigenvalues of H(K)∞ and the projections of the true class labels on the eigenvectors (i.e., the projections {(v 2 's for top eigenvalues.

Generalization.

We next investigate the generalization performance of the Gaussian kernel method by analyzing the values of equations FORMULA7 and (6).

TAB0 shows this quantity for different settings and kernels respectively on MNIST and CIFAR-10 datasets.

We observe that in both datasets with several kernels we obtain a lower theoretical upper bound on the generalization error.

It is clear that the bound improves as the dimension of the representations increases but also that the generalization bound seems quite sensitive to values of γ.

In addition to the theoretical upper bound, we measure the test error for the studied datasets.

FIG3 show respectively the test error and the test accuracy at different steps of the optimization by Gradient Descent for CIFAR-10.

We observe that the kernel methods yield significant improvements of both the test error and the accuracy on the test dataset.

We observe that the larger the kernel, the larger the improvement.

Additionally, we can see a sharper reduction in test error compared to the no-kernel case.

This sharp transition (after a small number of steps) is particularly interesting.

Because, along such a transition, we observe a significant improvement in the accuracy on test dataset.

Thus early-stopping that is commonly used in deep learning can be even more efficient when using kernel methods.

Finally, similar to the no-kernel case in BID0 , by comparing the plots in FIG0 , 1(c) and 2(a) we find tight connections between, i) (training) optimization, ii) projection on the top eigenvalues, and iii) generalization.

We can therefore improve both training and generalization with kernels since we can get better alignment of the eigenvectors belonging the largest eigenvalues and the target labels.

Choosing a proper kernel and its parameters can be challenging BID7 , as also seen in TAB0 .

Thus, we investigate a data-dependent neural kernel and embedding.

For this purpose, we add a second hidden layer to the neural network with m = 10000 hidden units and ReLU activation.

We pre-train this embedding using two different approaches.

The first layer is then kept fix as an embedding where the rest of the network is reinitialized and trained.

The first approach is to split the training data in half.

We use the first subset to pre-train this three-layer network and the second subset to use for our optimization experiments.

In this approach we double η to keep the step length the same.

The other approach is to use data from a different domain for pre-training.

For instance, we use the last two classes of the CIFAR-10 dataset for pre-training the embedding.

We compare our results with not using any kernel and with using a RFF kernel with embedding of size 10000.

Optimization.

FIG4 shows the training loss for the CIFAR-10 dataset.

We observe the neural embeddings achieve faster convergence compared to the previous methods.

We report the training loss for neural embedding (same label) on the second (unused) subset of the data, whereas in the other cases we report the results on the full training data.

If we use only the second subset for the other methods, we observe very consistent results to FIG4 .

FIG4 (c) demonstrates the top eigenvalues as well as their eigenvector projections on the target labels.

This shows that both variants of neural embeddings improve alignment of the labels to eigenvectors corresponding to larger eigenvalues (compared to the best RFF kernel).

While the effect is unsurprisingly larger when pre-training on the same labels, it is still significantly better when pre-trained on other labels.

Generalization.

In FIG4 (b) we report the test error on the CIFAR-10.

This shows that the neural embeddings perform at least comparable with the best studied RFF kernel.

If the pre-training is done on the same labels we obtain a clear improvement, even if the actual training is only done on a dataset with half the size.

We extended the recent results of BID0 by a spectral analysis of the representations corresponding to kernel and neural embeddings and showed that such representations benefit both optimization and generalization.

By combining recent results connecting kernel embeddings to neural networks such as BID6 BID4 , one may be able to extend the fine-grained theoretical results of BID0 for two layer networks to deeper networks.

@highlight

Spectral analysis for understanding how different representations can improve optimization and generalization.