Recent work has focused on combining kernel methods and deep learning.

With this in mind, we introduce Deepström networks -- a new architecture of neural networks which we use to replace top dense layers of standard convolutional architectures with an approximation of a kernel function by relying on the Nyström approximation.

Our approach is easy highly flexible.

It is compatible with any kernel function and it allows exploiting multiple kernels.

We show that Deepström networks reach state-of-the-art performance on standard datasets like SVHN and CIFAR100.

One benefit of the method lies in its limited number of learnable parameters which make it particularly suited for small training set sizes, e.g. from 5 to 20 samples per class.

Finally we illustrate two ways of using multiple kernels, including a multiple Deepström  setting, that exploits a kernel on each feature map output by the convolutional part of the model.

Kernel machines and deep learning have mostly been investigated separately.

Both have strengths and weaknesses and appear as complementary family of methods with respect to the settings where they are most relevant.

Deep learning methods may learn from scratch relevant features from data and may work with huge quantities of data.

Yet they actually require large amount of data to fully exploit their potential and may not perform well with limited training datasets.

Moreover deep networks are complex and difficult to design and require lots of computing and memory resources both for training and for inference.

Kernel machines are powerful tools for learning nonlinear relations in data and are well suited for problems with limited training sets.

Their power comes from their ability to extend linear methods to nonlinear ones with theoretical guarantees.

However, they do not scale well to the size of the training datasets and do not learn features from the data.

They usually require a prior choice of a relevant kernel amongst the well known ones, or even require defining an appropriate kernel for the data at hand.

Although most research in the field of deep learning seems to have evolved as a "parallel learning strategy" to the field of kernel methods, there are a number of studies at the interface of the two domains which investigated how some concepts can be transferred from one field to another.

Mainly, there are two types of approaches that have been investigated to mix deep learning and kernels.

Few works explored the design of deep kernels that would allow working with a hierarchy of representations as the one that has been popularized with deep learning (2; 14; 7; 6; 20; 23) .

Other studies focused on various ways to plug kernels into deep networks (13; 24; 5; 12; 25) .

This paper follows this latter line of research, it focuses on convolutional networks.

Specifically, we propose Deepström networks which are built by replacing dense layers of a convolutional neural network by an adaptive approximation of a kernel function.

Our work is inspired from Deep Fried Convnets (24) which brings together convolutional neural networks and kernels via Fastfood (9), a kernel approximation technique based on random feature maps.

We revisit this concept in the context of Nyström kernel approximation BID21 .

One key advantage of our method is its flexibility that enables the use of any kernel function.

Indeed, since the Nyström approximation uses an explicit feature map from the data kernel matrix, it is not restricted to a specific kernel function and not limited only to RBF kernels, as in Fastfood approximation.

This is particularly useful when one wants to use or learn multiple different kernels instead of a single kernel function, as we demonstrate here.

In particular we investigate two different ways of using multiple kernels, one is a straightforward extension to using multiple kernels while the second is a multiple Deepström variant that exploits a Nyström kernel approximation for each of the feature map output by the convolutional part of the neural network.

Furthermore the specific nature of our architecture makes it use only a limited number of parameters, which favours learning with small training sets as we demonstrate on targeted experiments.

Our experiments on four datasets (MNIST, SVHN, CIFAR10 and CIFAR100) highlight three important features of our method.

First our approach compares well to standard approaches in standard settings (using ful training sets) while requiring a reduced number of parameters compared to full deep networks and of the same order of magnitude as Deep Fried Convnets.

This specific feature of our proposal makes it suitable for dealing with limited training set sizes as we show by considering experiments with tens or even fewer training samples per class.

Finally the method may exploit multiple kernels, providing a new tool with which to approach the problem of multiple kernel learning (MKL) (4), and enabling taking into account the rich information in multiple feature maps of convolution networks through multiple Deepström layers.

The rest of the paper is organized as follows.

We provide background on kernel approximation via the Nyström and the random Fourier features methods and describe the Deep Fried Convnet architecture in Section 2.

The detailed configuration of the proposed Deepström network is described in Section 3.

We also show in Section 3 how Deepström networks can be used with multiple kernels.

Section 4 reports experimental results on MNIST, SVHN, CIFAR10 and CIFAR100 datasets to first provide a deeper understanding of the behaviour of our method with respect to the choice of the kernels and the combination of these, and second to compare it to state of the art baselines on classification tasks with respect to accuracy and to complexity issues, in particular in the small training set size setting.

Kernel approximation methods have been proposed to make kernel methods scalable.

Two popular methods are Nyström approximation BID21 and random features approximation BID15 .

The former approximates the kernel matrix by an efficient low-rank decomposition, while the latter is based on mapping input features into a low-dimensional feature space where dot products between features approximate well the kernel function.

Nyström approximation BID21 It computes a low-rank approximation of the kernel matrix by randomly subsampling a subset of instances.

Let consider a training set of n training samples, DISPLAYFORM0 which is selected from the training set.

Assuming the subset includes the first samples, or rearranging the training samples this way, K may be rewritten as: DISPLAYFORM1 where K 11 is the Gram matrix on subset L. Nyström approximation is obtained as follows DISPLAYFORM2 From this approximation the Nyström nonlinear representation of a single example x is given by DISPLAYFORM3 where DISPLAYFORM4 Random features approximation (16) It computes a low-dimensional feature mapφ of dimension q such that φ (·),φ(·) =k(·, ·) k(·, ·).

Two instances of this method are Random Kitchen Sinks (RKS) and Fastfood (17; 9).

RKS approximates a Radial Basis Function (RBF) kernel using a random feature map defined as DISPLAYFORM5 where z ∈ R p , Q ∈ R q×p and Q i,j are drawn randomly.

If Q i,j are drawn according to a Gaussian distribution then the method is shown to approximate the Gaussian kernel, i.e. φ rks ( DISPLAYFORM6 where σ is the hyper-parameter of the kernel.

Note that σ is related to the parameters of the Gaussian distribution that generate the random features.

The Fastfood method (9) is a variant of RKS with reduced computational cost for the Gaussian kernel.

It is based on approximating the matrix Q in Eq. 2, when q = p, by a product of diagonal and hadamard matrices according to DISPLAYFORM7 where S,G and B are diagonal matrices of size p × p, Π ∈ {0, 1} p×p is a random permutation matrix, H is a Hadamard matrix which does not requite to be stored, and σ is an hyperparameter.

Matrix V may be used in place of Q in Eq. 2 to define the Fastfood nonlinear representation map DISPLAYFORM8 Note that this definition requires p to be a power of 2 to take advantage of the recursive structure of the Hadamard matrix.

Note also that to reach a representation dimension q > p one may compute multiple V and concatenate the corresponding φ f f .Deep Fried Convnets (24) Our attention in this work is especially focused on combining kernel approximation with deep learning architecture.

Deep Fried Convnets is a deep learning architecture that replaces dense layers of a convolutional neural architecture by a Fastfood approximation of a kernel.

This allows to take advantage of the low complexity cost in terms of computation and memory of Fastfood to reduce significantly the computation cost and the number of parameters of the fully-connected layers of the deep convolutional neural network.

More formally, let conv(x) be the representation of the data sample x learned by a convolutional neural network.

It may include a number of convolution blocks, each including convolution and pooling layers, batch normalization and nonlinear activation.

In Deep Fried Convnets, an input x ∈ R d is mapped to the representation spaces conv(x) ∈ R p and then the Fastfood feature map φ f f is applied to the convolutional representation conv(x) instead of the fully-connected layers.

The feature representation of x with Deep Fried Convnets is then (φ f f •conv)(x) ∈ R q .

It is of note that this method is dedicated to RBF kernels.

In (24), wo architectures have been proposed.

The first one relies on the Fastfood kernel approximation method as described above.

The second one is a variant of Fastfood called Adaptive-Fastfood.

It involves learning the weights of matrices S, G and B through gradient descent rather than setting them randomly, while matrices Π and H are kept unchanged.

In the next section we introduce Deepström Networks as an alternative to Deep Fried Convnets.

They are based on Nyström approximation and are not limited to RBF kernels.

They also allow the use of multiple different kernels and find an appropriate kernel function.3 Deepström NETWORKS In this section, we describe our new Deepström model which combines the desirable characteristics of Nyström approximation and convolutional neural networks.

First, we start by revisiting the concept of Nyström kernel approximation from a feature map perspective.

Nyström approximation from an empirical kernel map perspective The empirical kernel map is an explicit n-dimensional feature map that is obtained by applying the kernel function on the training data x i (18).

It is defined as DISPLAYFORM9 An interesting feature of the empirical kernel map is that if we consider the inner product in R n ·, · M = ·, M · with a positive semi-definite (psd) matrix M , we can recover the kernel matrix K using the empirical kernel map by setting M equals to the inverse of the kernel matrix.

In other words, DISPLAYFORM10 = K. Since K is a psd matrix, one can consider the feature φ emp : DISPLAYFORM11 as an explicit feature map that allows to reconstruct the DISPLAYFORM12 The Deepström network architecture involves a usual convolutional part, conv , including multiple convolutional blocks, and a Deepström layer which is then fed to (eventually multiple) standard dense layers up to the classification layer.

The Deepström layer computes the kernel between the output of the conv block for a given input and the corrsponding representations of the trains samples in the subsample L.kernel matrix.

This feature map is of dimension n and then is not interesting when the number of example is large.

The feature map of the Nyström approximation is given by DISPLAYFORM13 T with x i ∈ L. From an empirical kernel map point of view,φ nys (x) can be seen as an "empirical kernel map" (18) and K DISPLAYFORM14 11 as a metric in the "empirical feature space".

From this viewpoint, we think that it could be useful to learn a metric W in the empirical feature space instead of assuming it to be equal to K 11 .

In a sense, this should allow to learn a kernel by learning its Nyström feature representation.

In the following, we call the setting where W is learned by the network Adapative Deepström Network.

Principle Deepström networks we propose are an alternative to Deep Fried Convnets.

They are based on using the Nyström approximation rather than the Fastfood one to integrate any kernel function on top of convolutional layers of a deep net.

Indeed, although Deep Fried Convnets yield state-of-the-art results with a significant gain with respect to memory resource and to inference complexity, it is restricted to the Gaussian kernel Fastfood, which may not be always the best choice in practice.

In addition, our method can deal with multiple kernels.

Deepström nets are Neural Networks that make use of a nonlinear representation function computed with the Nyström approximation (see Figure 1) .

Starting from Deep Fried Convnets we replace φ f f with φ nys so that a Deepström net implements a function f (x) = (lc • φ nys • conv)(x).

In order to compute the above Nyström representation of a sample x one must consider a subsample L of training instances.

Since the kernel k is computed on the representations given by convolutional layers, the samples in L must be represented in the same space, and hence must be processed by the convolutional layers as well.

Once convolutional representations are calculated, the kernel function may be computed with an input sample and each instance in L in order to get the k x,L , which is then linearly transformed by W before the linear classification layer (see Figure 1) .Two main structural differences between Deep Fried Convnets and our Deepström nets: (I) Nyström has the flexibility to use different kernel functions and to combine multiple kernels, and (II) in contrast to Fastfood the Nyström approximation is data dependent.

However, one problem arises with the computation of K

We present a series of experimental results that explore the potential of Deepström networks with respect to various classification settings.

First we consider a rather standard setting and compare our approach with standard models on image classification tasks.

We explore in particular the behaviour of Deepström networks with various kernels and stress the very limited subsample size needed to reach state-of-the-art accuracy.

Next we investigate the use of Deepström networks in a small training set setting, which shows that our approach may allow to learn new classes with only very few training samples, taking advantage of the reduced number of parameters learned by our model.

Before describing all these results we detail the datasets used.

Finally we investigate first the multiple kernel architecture and illustrate its interest when learning with RBF kernel to overcome the hyperparameter selection, and second we demonstrate the benefit of a multiple Deepström approach, combining kernels computed from individual feature maps.

We conducted experiments on four well known image classification datasets: MNIST (11), SVHN (15), CIFAR10 and CIFAR100 BID7 , details on these datasets are provided in Table 1 .

We pretrained the convolutional layers using standard architectures on both datasets: Lenet (10) for MNIST and VGG19 BID18 for SVHN, CIFAR10 and CIFAR100.

We slightly modified the filters' sizes in Lenet network to ensure that the dimension of data after the convolution blocks is a power of 2 (needed for the Deep Fried Convnets architecture).We compare three convolutional architectures in all conducted experiments.

Pretrained convolutional parts are shared by the three architectures, which differ from the layers on top of it: (1) Dense architectures use dense hidden layers, i.e. these are classical convnets architectures ; (2) Deep Fried implements the Fastfood approximation (Equation 3) ; (3) Deepstrom stands for our proposal.

For Dense architectures, we considered one hidden layer with relu activation function, and varied the output dimension as {2, 4, 8, 16, 32, 64, 128, 1024} in order to highlight accuracies as a function of the number of parameters.

For the Fastfood approximation in Deep Fried Convnets we consider that φ f f is gained with one stack of random features to form V in equation 3, except in the experiments of section 4.3 which yields a representation dimension up to 5 times larger.

Regarding our approach φ nys , we varied the subset size L ∈ {2, 4, 8, 16, 32, 64, 128}, we tested with the linear, the RBF, and the Chi2 kernels, and we chose as output dimension the same size as the subset sample size.

Finally we explored the adaptive as well as non-adaptive variants.

Models were learned to optimize the cross entropy criterion with Adam optimizer and a gradient step fixed to 1e −4 .

Dropout was used on representation layers with probability equal to 0.5.

By default the RBF bandwidth was set to the inverse of the mean distance between the representations, after the convolutional part, of pairs of training samples.

All experiments were performed with Keras (3) and Tensorflow (1).

Note that the aim of all the experiments below is to investigate the potential of out architecture, not to reach or beat state of the art results on the datasets considered.

We then compare results gained with our architecture and with state-of-the-art models, given a shared convolutional model.

Consequently, we did not use tricks such as data augmentation and extensive tuning and, in particular, we did not use the best known convolutional architecture for each of the dataset, we rather used a reasonable deep architecture, VGG19, for the three datasets CIFAR10, CIFAR100 and SVHN.

We compare now Deepström networks to two similar architectures, Deep Fried Convnets and classical convolutional networks (inspired from VGG19 and Lenet depending on the dataset).

We vary the number of parameters of each architecture in order to highlight classification accuracy with respect to needed memory space.

FIG2 shows the compared networks accuracy with respect to the number of parameters, and ignore parameters for convolutions layers to ease the readability.

We repeated each experiments 10 times and plot average scores with standard deviations.

Deepström models of increasing complexity (number of parameters) correspond to the use of subsample of increasing size from 2 (leftmost point) to 128 (rightmost point).

One may see that there is no need of a large subsample here.

This may be explained since the convolutional part of the network has been learned to yield quite robust and stable representations of input images.

We provide a figure in the Appendix that illustrates this.

The Deepström network is able to reach state-of-the-art performance using much fewer parameters than both classical networks and Deep Fried Convnets.

Moreover, we also observe smaller variations that points out the robustness of our model.

The flexibility in the choice of the kernel function is a clear advantage of out method.

The best kernel is clearly dependent on the dataset (linear on MNIST, Chi2 on SVHN and CIFAR100, RBF on CIFAR10).

While Random Features in DeepFried are restricted to RBF kernels, we show for instance a gain by using the Chi2 Kernel (k(x 1 , x 2 ) = ||x 1 −x 2 || 2 /(x 1 +x 2 )) that had been used for image classification BID20 .

We also notice the benefit of adaptive variants of Deepström model, suggesting that our model is able to learn and adapt useful Kernel function.

Finally, note that we obtained very similar results with neural architectures exploiting two hidden layers instead of one after the convolution module conv.

Here we explore the ability of our model to work with few training samples, from very few to tens of samples per class.

It is an expected benefit of the method since the use of kernels could take advantage of small training samples.

Note that we do not exactly deal with a real small training set setting.

These preliminary experiments aim to show how the final layers of a convolutional model may be learned from very few samples, given a frozen convolutional model.

We actually performed the following experiments by exploiting a trained convolution model conv that has been learned on the full CIFAR100 training set and investigate the performance of Deepström architectures as a function of the training set used to learn the classification layers.

One perspective of this work is to exploit such a strategy for domain adaptation settings where the convolutional model is trained on a training set within a different domain as the classes to be recognized.

Having at our disposal such a trained convolution model conv, we leverage on the additional information that one may easily include in our models, which is brought by the subsample set.

Notice that this subsample may include unlabeled samples since their labels are not used for optimizing the model.

Table 2 reports the comparison of network architectures on four datasets.

We consider Adaptive Deepström using Linear, RBF or Chi2 kernels and compare with Dense and Adaptive Deepfried for training set sizes of 5, 10 and 20 samples per class.

We only consider here adaptive variants since they brought better results than their non adaptive counterparts.

We obtain models with different complexities: by increasing the hidden layer size in standard convolutional models, or by stacking the number of matrices V in DeepFried (up to 8 times, more was untractable on our machines), and by increasing the subset size in Deepström.

Reported results are averaged over 30 runs.

One may see first that Deepstrom architectures outperfom baselines on every setting except for 5 training samples per class on MNIST.

The linear kernel performs well on MNIST but is significantly worse than baselines on harder datasets.

At the opposite, both ADSR and ADSC significantly outperfom Adaptive DeepFried for any dataset and perform on par or significantly better than Dense architectures on the hardest CIFAR100 dataset.

Moreover one sees that no single kernel based Deepstrom architecture dominate on all settings, showing the potential interest of combining multiple kernels as following experiments will show.

We report here results gained using multiple kernels in two different ways.

First we exploited the Multiple Kernels strategy that we described in section 3 for exploiting multiple kernels in the output of the convolutional blocks, conv.

FIG4 reports results gained when using a combination of RBF kernels with various bandwidths and for different subsample sizes.

Our multiple kernel strategy, exploiting kernels defined with various values of the hyperparameter allows automatically handling this hyper-parameter which usually requires to be tuned either through cross validation or to be manually chosen.

The plots show the accuracy on the CIFAR10 dataset as a function of σ value, where the performance of the multiple kernel Deepström is shown as a horizontal line.

Plots report results for various subsample size equal to 2 (left), 4 (middle) and 8 (right), averaged over 10 runs.

As may be seen, using our Multiple kernel strategy allows adapting the kernel combination optimally from the data without requiring any prior choice on the RBF bandwith hyper-parameter.

Second, we investigated another architecture that exploits Multiple Deepström approximations as presented in section 3.

Here we use in parallel multiple Nyström approximations where kernels are dedicated to deal each with the output of a single feature map of the conv part.

TAB4 reports results on CIFAR100.

We show the best performances obtained for each method by grid-searching on various hyper-parameters depending on the models, within a similar range of number of parameters.

For Dense model, we considered one or two hidden layers of 16, 64, 128, 1024, 2048 or 4096 neurons.

Deepfried is the adaptive variant where we varied the number of stacks in 1, 3, 5, 7.

Deepström is also the adaptive variant where the subsample size is in 16, 64, 128, 256, 512.

We observe that both Deepström models outperform the considered baselines, demonstrating the interest in combining Deepström approximations.

We proposed Deepström, a new hybrid architecture that mixes deep networks and kernel methods.

It is based on the Nyström approximation that allow considering any kind of kernel function in contrast to Deep Fried Convnets.

Our proposal allows reaching state of the art results while significantly reducing the number of parameters on various datasets, enabling in particular learning from few samples.

Moreover the method allows to easily deal with multiple kernels and with multiple Deepström architectures.

FIG5 plots the 2-dimensional φ nys representations of some CIFAR10 test samples obtained with a subsample of size equal to 2 (while the number of classes is 10) and two different kernels.

One may see here that the 10 classes are already significantly well separated in this low dimensional representation space, illustrating that a very small sized subsammple is already powerfull.

Beside, we experienced that designing Deepström Convnets on lower level features output by lower level convolution blocks may yield state-of-the-art performance as well while requiring larger subsamples.

@highlight

A new neural architecture where top dense layers of standard convolutional architectures are replaced with an approximation of a kernel function by relying on the Nyström approximation.