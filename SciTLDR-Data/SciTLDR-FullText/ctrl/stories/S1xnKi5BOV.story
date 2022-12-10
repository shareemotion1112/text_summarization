Recently deep neural networks have shown their capacity to memorize training data, even with noisy labels, which hurts generalization performance.

To mitigate this issue, we propose a simple but effective method that is robust to noisy labels, even with severe noise.

Our objective involves a variance regularization term that implicitly penalizes the Jacobian norm of the neural network on the whole training set (including the noisy-labeled data), which encourages generalization and prevents overfitting to the corrupted labels.

Experiments on noisy benchmarks demonstrate that our approach achieves state-of-the-art performance with a high tolerance to severe noise.

Recently deep neural networks (DNNs) have achieved remarkable performance on many tasks, such as speech recognition Amodei et al. (2016) , image classification He et al. (2016) , object detection Ren et al. (2015) .

However, DNNs usually need a large-scale training dataset to generalize well.

Such large-scale datasets can be collected by crowd-sourcing, web crawling and machine generation with a relative low price, but the labeling may contain errors.

Recent studies Zhang et al. (2016) ; Arpit et al. (2017) reveal that mislabeled examples hurt generalization.

Even worse, DNNs can memorize the training data with completely randomly-flipped labels, which indicates that DNNs are prone to overfit noisy training data.

Therefore, it is crucial to develop algorithms robust to various amounts of label noise that still obtain good generalization.

To address the degraded generalization of training with noisy labels, one direct approach is to reweigh training examples Ren et al. (2018); Jiang et al. (2017) ; Han et al. (2018) ; Ma et al. (2018) , which is related to curriculum learning.

The general idea is to assign important weights to examples with a high chance of being correct.

However, there are two major limitations of existing methods.

First, imagine an ideal weighting mechanism.

It will only focus on the selected clean examples.

For those incorrectly labeled data samples, the weights should be near zero.

If a dataset is under 80% noise corruption, an ideal weighting mechanism assigns nonzero weights to only 20% examples and abandons the information in a large amount of 80% examples.

This leads to an insufficient usage of training data.

Second, previous methods usually need some prior knowledge on the noise ratio or the availability of an additional clean unbiased validation dataset.

But it is usually impractical to get this extra information in real applications.

Another approach is correction-based, estimating the noisy corruption matrix and correcting the labels Patrini et al. (2017) ; Reed et al. (2014) ; Goldberger & Ben-Reuven (2017) .

But it is often difficult to estimate the underlying noise corruption matrix when the number of classes is large.

Further, there may not be an underlying ground truth corruption process but an open set of noisy labels in the real world.

Although many complex approaches Jiang et al. (2017); Ren et al. (2018); Han et al. (2018) have been proposed to deal with label noise, we find that a simple yet effective baseline can achieve surprisingly good performance compared to the strong competing methods.

In this paper, we first analyze the conditions for good generalization.

A model with simpler hypothesis and smoother decision boundaries can generalize better.

Then we propose a new algorithm which can satisfy the conditions and take advantage of the whole dataset including the noisy examples to improve the generalization.

Our main contributions are:• We build a connection between the generalization of models trained with noisy labels and the smoothness of solutions, which is related to the subspace dimensionality.• We propose a novel approach for training with noisy labels, which greatly mitigates overfitting.

Our method is simple yet effective and can be applied to any neural network architecture.

Additional knowledge on the clean validation dataset is not required.• A thorough empirical evaluation on various datasets (CIFAR-10, CIFAR-100) is conducted and demonstrates a significant improvement over the competing strong baselines.

In this section, we briefly introduce some notations and settings in learning with noisy labels.

The target is to learn a robust K-class classifier f from a training dataset of images with noisy supervision.

Let D = {(x 1 ,ỹ 1 ), ..., (x N ,ỹ N )} denote a training dataset, where x n ∈ X is the n-th image in sample space X (e.g., R d ) with its corresponding noisy labelỹ n ∈ {1, 2, ..., K}.

The label noise is often assumed to be class-conditional noise in previous work Natarajan et al. (2013); Patrini et al. (2017) , where the label y is flipped toỹ ∈ Y with some probability p(ỹ|y).

It means that p(ỹ|x, y) = p(ỹ|y), , the corruption of labels is independent of the input x. This kind of assumption is an abstract approximation to the real-world corruption process.

For example, non-expert labelers may fail to distinguish some specific species.

The probability p(ỹ|y) is represented by a noise transition matrix DISPLAYFORM0

In this section, we present a new robust training algorithm to deal with noisy labels.

We argue that a model with lower complexity is more robust to label noise and generalizes well.

The dimensionality of the learned subspace and the smoothness of decision boundaries can both indicate how complex the model is.

Therefore, we propose a method to regularize the predictive variance to achieve low subspace dimensionality and smoothness, respectively.

In order to alleviate over-fitting to the label noise, we propose a regularizer that is not dependent on the labels.

We induce the smoothness of decision boundaries along the data manifold, which is shown to improve the generalization and robustness.

If an example x is incorrectly labeled withỹ, it has a high probability to lie near the decision boundary or in the wrong cluster not belonging to y. Therefore, the prediction variance can be high on the noisy examples.

We propose to regularize the variance term.

The mapping function is smoothed and thus also the decision boundaries.

Concretely, the variance is estimated by the difference of predictions under perturbations ξ and ξ including the input noise like Gaussian noise and stochastic data augmentation, as well as the network noise like dropout: DISPLAYFORM0 We can show that R V (θ) is an unbiased estimation of the predictive variance if the perturbations are treated as a part of the model uncertainty.

Relation to the generalization of DNNs.

We show that this regularization helps to learn a lowdimensional feature space that captures the underlying data distribution.

The variance term implicitly estimates the Jacobian norm, , the Frobenius norm of the Jacobian of the network output w.r.t.

the inputs: J(x) F .

A simplified version is to assume ξ is sampled from a Gaussian distribution, i.e., ξ, ξ ∼ N (0, σ 2 I) and the perturbation is small and additive, ,x = x + ξ where σ is near zero.

DISPLAYFORM1 By first-order Taylor expansion, and let J(x) = ∂f ∂x DISPLAYFORM2 and omitting the high-order terms, we have DISPLAYFORM3 If we further take expectation over N samples of x i , we get DISPLAYFORM4 It can be proved that this is an unbiased estimator.

For perturbations of natural images, similar analysis applies.

It was shown in Sokolić et al.; Novak et al. (2018) that the Jacobian norm is related to the generalization performance both theoretically and empirically.

Perturbations on the data manifold can be approximated by stochastic data augmentation.

Similar objectives have been explored in semi-supervised learning Laine & Aila FORMULA1 (2018) .

It restricts the solution to be of some specific form, which is equivalent to imposing some prior knowledge of the model structure.

The regularizer serves as an inductive bias on the structure of the feature space.

By reducing the variance of predictions, the neural network is encouraged to learn a low-dimensional feature space where the training examples are far from the decision boundaries and tend to cluster together.

This alleviates the possibility of the model to increase its complexity to fit the noisy labels.

Therefore, the learning objective is simply DISPLAYFORM5 where the first term is any loss function including the cross-entropy loss or previously proposed noise-robust losses.

In Section 4, we show empirically that the objective can learn a model with low subspace dimensionality and low hypothesis complexity.

In this section, we present both quantitative and qualitative results to demonstrate the effectiveness of our method.

Our method is independent of both the architecture and the dataset.

We first provide results on the widely adopted benchmarks, CIFAR-10 and CIFAR-100.

Results on ImageNet and WebVision will be provided in Sec. 5.5 and Sec. , where performance relative to the standard clean settings can be observed.

We fix the hyper-parameter λ = 300 in all the experiments for CIFAR-10 and λ = 3000 for CIFAR-100.In all the experiments, our method achieves significantly better resistance to label noise from moderate to severe levels.

In particular, our approach attains a 13.31% error rate on CIFAR-10 with a noise fraction of 80%, down from the previous best 32.08%.

Using the same network architecture WRN-28-10 as ours and 1000 clean validation images, learning to reweight Ren et al. FORMULA1 achieves 38.66% test error on CIFAR-100 with 40% noise while our method achieves a better 25.73% even without any knowledge on the clean validation images.

FIG5 plot the test accuracy against the number of epochs on the two datasets.

We provide a simple baseline -CCE, standing for categorical cross-entropy loss that treats all the noisy training examples as clean and trains a WRN-28-10.

We can see that the baseline tends to over-fit the label noise at the later stage of training while our method does not suffer from the incorrect training signal.

We propose a simple but effective algorithm for robust deep learning with noisy labels.

Our method builds upon a variance regularizer that prevents the model from overfitting to the corrupted labels.

Extensive experiments given in the paper show that the generalization performance of DNNs trained with corrupted labels can be improved significantly using our method, which can serve as a strong baseline for deep learning with noisy labels.

Learning with noisy labels has been broadly studied in previous work, both theoretically Natarajan et al. FORMULA1 , that is to minimize D f θ (w(x, y)p(x, y), q(x, y)) where D is some distance measure implicitly learned by f θ and w(x, y) is the density ratio, the learned weights for each example (x, y).Using additional clean validation dataset.

Azadi et al. (2015) proposed a regularization term to encourage the model to select reliable examples.

Hendrycks et al. (2018) proposed Golden Loss Correction to use a set of trusted clean data to mitigate the effects of label noise.

They estimate the corruption matrix using the trained network with noisy labels and then re-train the network corrected by the corruption matrix.

Ren et al. (2018) also used a small clean validation dataset to determine the weights of training examples.

The success of these methods is based on the assumption that clean data is from the same distribution as the corrupted data as well as the test data.

However, more realistic scenario are ones where (1) p(x) varies between the clean data and the noisy data, e.g., imbalanced datasets.

2) There is class mismatch: p(y|x) differs.

Similar problems exist in semi-supervised learning.

All these methods require a clean validation dataset to work well while the proposed method does not require it.

We also plot label precision against number of epochs in Figure 4 .

Here we treat the 1 − η ratio of the training examples with minimal training losses as the clean examples.

The label precision is computed as the portion of true clean examples among them.

The ideal algorithm without over-fitting will have 100% label precision.

The higher the label precision is, the better robustness the model achieves.

Figure 4 demonstrates that our method obtains a higher label precision.

A more realistic and more challenging noise type than the uniform noise is to corrupt between the semantically similar classes.

For CIFAR-10, the class-dependent asymmetric noise is simulated by mapping TRUCK → AUTOMOBILE, BIRD → AIRPLANE, DEER → For CIFAR-100, class dependent noise is simulated by flipping each class into the next class with probability η.

The last class is flipped to the first class circularly, , the transition matrix has 1 − η on the diagonal and η off the diagonal: DISPLAYFORM0 DISPLAYFORM1 Results are presented in TAB2 .

We compare to a range of competing loss-correction methods whose results are taken from Zhang & Sabuncu (2018) and our baseline trained with only CCE.

We use the same hyper-parameter λ = 300 among all the experiments for CIFAR-10 and λ = 3000 for CIFAR-100.

Note that Forward T is the forward correction Patrini et al. (2017) using the ground-truth noise transition matrix, whose results are almost perfect.

Our method does not use any ground-truth knowledge of the noise corruption process.

We can see that our method is robust to all the settings and is less influenced by the variations of noise types.

The test accuracy along the training process on CIFAR-100 is also plotted in Figure 2 .

We assess the sensitivity of our algorithm with respect to the hyper-parameter λ and the results are plotted in Figure 5 .

We can see that the performance of our method remains stable across a wide range of hyper-parameter choices.

We visualize the embeddings of our algorithm on test data.

FIG2 shows the representations h(x) ∈ R 128 projected to 2 dimension using t-SNE Maaten & Hinton (2008) .

Figure 6: t-SNE 2D embeddings of the test dataset on CIFAR-10 trained with 60% uniform label noise.

Each color represents a class.

Our method learns a more separable feature space.

<|TLDR|>

@highlight

The paper proposed a simple yet effective baseline for learning with noisy labels.