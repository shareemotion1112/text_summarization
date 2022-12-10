This paper proposes and demonstrates a surprising pattern in the training of neural networks: there is a one to one relation between the values of any pair of losses (such as cross entropy, mean squared error, 0/1 error etc.) evaluated for a model arising at (any point of) a training run.

This pattern is universal in the sense that this one to one relationship is identical across architectures (such as VGG, Resnet, Densenet etc.), algorithms (SGD and SGD with momentum) and training loss functions (cross entropy and mean squared error).

Neural networks are state of the art models for various tasks in machine learning, especially those in computer vision and natural language processing.

While there has been significant progress in designing and applying neural networks for various tasks, our understanding of most aspects of their behavior has not yet caught up with these advances.

One significant challenge in furthering our understanding is the huge variation in deep learning models.

In the context of image classification (which will be the context of the current paper), there are several well known models that have been developed by the machine learning community: VGG BID5 , Resnet BID2 , Densenet BID3 to name a few; all of them having their own unique structure.

Is it possible to understand the behavior of all of these models through a common lens?The main contribution of the current paper is to propose and demonstrate that, despite the diversity in the structure of these different models, there are some striking resemblances in the behavior of all of these models.

More concretely, the training curves across any two loss functions (such as cross entropy vs 0/1 error or cross entropy vs mean squared error) essentially overlap across all of the above models.

See FIG1 for a pictorial description and Section 3 for a rigorous description.

This observation suggests that training of most (if not all) deep neural networks follows the same pattern.

The existence of such universal patterns is quite exciting as it points to the possibility of understanding the behavior (in this instance training behavior) of different neural networks through a single approach.

We also note that, while this similarity in behavior extends, to some extent, also to the test data, there are limitations (see Section 4.2).Paper organization: In Section 2, we will present the setup and required definitions.

Section 3 formally describes the phenomenon identified in this paper.

Section 4 presents the main experimental results.

We conclude in Section 5.

More experimental results are presented in the appendix.

Notation: Vectors are written in bold small case letters (such as p and x).

DISPLAYFORM0 The canonical basis for R K is represented as {e i , . . .

, e K } where e ij = 1 if i = j else e ij = 0.

For the task of learning a classifier, the goal is to learn a mapping from the input space X ⊆ R d to the set of class labels [K] := {1, . . . , K}, where K is the total number of DISPLAYFORM0 D , the classifier C can be obtained by choosing the class label for which the prediction is the maximum.

The classifier C θ (x) := arg max k∈ [K] [F θ (x)] k thus maps the input x ∈ X to a class label contained in [K] .

The training of the network DISPLAYFORM1 is a minimization problem over its parameters θ.

One minimizes the empirical loss L(θ; F, l) : DISPLAYFORM2 In practice, l is usually chosen to satisfy l(e y , y) = 0 ∀ y ∈ [K].

In this paper, we consider the following commonly used training loss functions: DISPLAYFORM3 It is to note that 1.

l 0/1 is not continuous in its first variable, therefore one cannot use gradient based optimization algorithms to directly minimize it, and 2.

l MSE & l CE are fundamentally different loss functions.l CE depends only on p y , i.e., prediction on true class, whereas l MSE depends on the full prediction vector p.

In the paper, we will denote l as the loss function and the corresponding L(θ; F, l) as the loss of the network F θ when trained using the loss function l.

We will also denote this loss by L (l) when we are arguing about multiple networks.

In this section, we will rigorously describe the phenomenon that we demonstrate in this paper.

Fix any two loss functions l 1 and l 2 (e.g., each of these could be l CE or l MSE or l 0/1 etc.).

As a training run progresses, the value of these losses on the training data L (θ t ; F, l 1 ) and L (θ t ; F, l 2 ), in general, decrease.

We posit that• Uniqueness: For any two loss functions l 1 and l 2 , there exists a unique function f (·) such that for any (intermediate) model F θt in the training run, its loss L (θ t ; F, l 2 ) on the training data is given by f (L (θ t ; F, l 1 )), where L (θ t ; F, l i ) is the loss evaluated on the training data using loss function l i .

In practice however, there are several sources of randomness (stochastic gradient descent, random initialization etc.) which mean that we expect that L (θ t ; F, l 2 ) ∼ f (L (θ t ; F, l 1 )) rather than exact equality.• Universality: The function f (·) depends only on l 1 and l 2 and is identical for all network architectures (such as Resnet, VGG, Densenet etc.), training loss functions (such as l CE , l MSE etc.) and training algorithms (such as SGD and SGDm etc.).While the universality of f (·) is clearly surprising, we note that the existence of a unique function f (·), as well as our empirical observation that it is monotonic, from a theory point of view, are quite surprising.

Before describing the observations, we first describe all the possible combinations of datasets, architectures, training losses and training algorithms we have used.• Architectures: We consider the below architectures 1.

Multi Layer Perceptrons (MLP) -1 and 2 hidden layered with 1024 neurons in each (mlp1024x1 and mlp1024x2 resp.) with batch normalization and ReLU activation, 2.

VGG (Simonyan & Zisserman, 2014)-VGG-11 and VGG-16 with and without batch normalization layers (vgg## and vgg##wobn resp.), 3.

Resnet BID2 ) -Resnet-18 and Resnet-50 (resnet18 and resnet50 resp.), 4.

Densenet (Huang et al., 2017) -Densenet-121 (densenet) and, 5.

Fully Connected (FC1) -No hidden layers.

The output is computed by softmax.• Datasets: We consider 4 datasets tabulated in TAB0 .•

Loss functions: We consider 2 training loss functions 1.

l MSE and, 2.

l CE• Algorithms: We consider 2 training algorithms: 1.

Stochastic Gradient Descent (SGD) 2.

SGD with 0.9 momentum (SGDm) Both the algorithms use a batch-size of 512 and a constant learning rate from 2 {−5,...,−10} such that there is no divergence in training.

The models are trained for 150 − 250 epochs.

Fixing a dataset, one can learn the classifier by training the network on a suitable training surrogate loss function (like l MSE and l CE ) using a variety of algorithms (like SGD and SGDm) and hope to minimize the empirical classification error L l 0/1 .

Across the training profile, we can as well see how their corresponding training surrogate losses behave with respect to each other.

From FIG1 we observe that the trends of training loss TAB0 Universality Patterns in the Training of Neural Networks DISPLAYFORM0 DISPLAYFORM1

Interestingly, the overlapping behavior observed between training surrogate loss and training classification error does not quite extend to the test data for CIFAR-10.

We show this in FIG3 where we do not observe a significant overlap across all architectures for both the training algorithms SGD and SGDm on CIFAR-10.

On the other hand, similar figure ( FIG8 ) for IMAGENET-20 exhibits a universal behavior.

The plot for IMAGENET-20 is presented in Appendix A.2.

These results motivate further exploration of possible universality patterns in the test data.

While a network F θ may be trained using a particular training loss function ( DISPLAYFORM0

We also verify if the observed trends hold across different random initialization of the parameters of the networks.

Similar trends are observed in FIG5 for CIFAR-10 where we keep the initialization and the training loss function fixed, but look at different training algorithms SGD and SGDm for all architectures jointly.

A similar figure ( FIG1 ) for IMAGENET-20 is presented in Appendix A.5.

Coming to universality across loss functions, we again observe similar overlapping trends in Figure 4 for CIFAR-10 where we keep the initialization and the training algorithm fixed, and vary the training loss on the two loss functions (l MSE and l MSE ) for all architectures jointly.

The corresponding figure FIG1 for IMAGENET-20 is presented in Appendix A.5.

As long as the datasets have same number of classes, we also check if this phenomenon is also observed across different datasets.

We compare CIFAR-10 with CIFAR-10 Random and Random datasets mentioned in TAB0 .

We do not observe similar significant overlapping behavior when we switch datasets as we present our observations in FIG1 and FIG1 in Appendix A.6.

The universality phenomenon seems to arise only for large models but not for small models.

For a shallow fully connected network with no hidden layers, the curves do not overlap in most of the cases as presented in Figure 5 .

We propose and demonstrate that there is a training dynamics which is universal for various neural networks.

While similar behavior does not seem to hold for test data, preliminary results call for further investigation.

One possible explanation is that distribution of predictions through the training process follows a universal law.

However, measuring standard distances (e.g., Wasserstein) requires a large number of samples.

One potential solution is to measure weaker distances such as neural net distances BID0 .

We believe that our observations could lead to a new way of understanding neural networks in a unified manner.

A. Appendix

This section is an extension to Section 4.1 where we describe our observations on CIFAR-10 dataset.

Here in Figure 6 we show similar results on IMAGENET-20 where we see that the overlapping trend of training loss and training 0/1 error also holds on Imagenet-20.

Figure 6 .

Training Imagenet-20 on train loss functions lMSE and lCE using SGD and SGDm for all architectures.

DISPLAYFORM0 We observe similar behavior when we vary the number of hidden neurons in a 1-hidden layer MLP.

Figure 7 shows that there is a significant overlap in the curves of training loss and classification error for MLPs with varying number of neurons in 2 {4,...,13} .(a) Trained L(lMSE) (b) Trained L(lCE) Figure 7 .

Training CIFAR-10 on train loss functions lMSE and lCE using SGD for MLPs with different number of neurons.

This section is an extension to Section 4.2 where we describe our observations on CIFAR-10 dataset.

Here in FIG8 we show the results on IMAGENET-20 where we observe a mild overlapping trend of training loss and 0/1 error on the test data.

This overlap observed is similar to that of CIFAR-10 if the over-fitting stage (non-monotonic part of the curves) is ignored.

DISPLAYFORM0

This section is an extension of Section 4.3 where we describe the universality phenomenon holding across all the architectures and all the pairs of surrogate losses.

We show these observations in Figure 9 and FIG1 for CIFAR-10 and IMAGENET-20 respectively.

This section is an extension to Section 4.4 where we claim that our observations are robust to initializations from Figure 11 and Figure 12.

277 TAB0 Universality Patterns in the Training of Neural Networks TAB0 Universality Patterns in the Training of Neural Networks TAB0 Universality Patterns in the Training of Neural Networks DISPLAYFORM0 DISPLAYFORM1

This section is in continuation to Section 4.6 where we describe our observations on the phenomenon across different datasets with same number of classes.

As we see in FIG1 and FIG1 , we do not observe the overlapping phenomenon when we switch datasets.(a) Trained using SGD (b) Trained using SGDm FIG1 .

All pairs of loss functions for IMAGENET-20 using SGD and SGDm across loss functions lMSE and lCE for all architectures.

TAB0 Universality Patterns in the Training of Neural Networks TAB0

<|TLDR|>

@highlight

We identify some universal patterns (i.e., holding across architectures) in the behavior of different surrogate losses (CE, MSE, 0-1 loss) while training neural networks and present supporting empirical evidence.