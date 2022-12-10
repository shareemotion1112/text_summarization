This paper explores the simplicity of learned neural networks under various settings: learned on real vs random data, varying size/architecture and using large minibatch size vs small minibatch size.

The notion of simplicity used here is that of learnability i.e., how accurately can the prediction function of a neural network be learned from labeled samples from it.

While learnability is different from (in fact often higher than) test accuracy, the results herein suggest that there is a strong correlation between small generalization errors and high learnability.

This work also shows that there exist significant qualitative differences in shallow networks as compared to popular deep networks.

More broadly, this paper extends in a new direction, previous work on understanding the properties of learned neural networks.

Our hope is that such an empirical study of understanding learned neural networks might shed light on the right assumptions that can be made for a theoretical study of deep learning.

Over the last few years neural networks have significantly advanced state of the art on several tasks such as image classification BID23 ), machine translation BID32 ), structured prediction BID2 ) and so on, and have transformed the areas of computer vision and natural language processing.

Despite the success of neural networks in making these advances, the reasons for their success are not well understood.

Understanding the performance of neural networks and reasons for their success are major open problems at the moment.

Questions about the performance of neural networks can be broadly classified into two groups: i) optimization i.e., how are we able to train large neural networks well even though it is NP-hard to do so in the worst case, and ii) generalization i.e., how is it that the training error and test error are close to each other for large neural networks where the number of parameters in the network is much larger than the number of training examples (highly overparametrized).

This paper explores three aspects of generalization in neural networks.

The first aspect is the performance of neural networks on random training labels.

While neural networks generalize well (i.e., training and test error are close to each other) on real datasets even in highly overparametrized settings, BID33 shows that neural networks are nevertheless capable of achieving zero training error on random training labels.

Since any given network will have large error on random test labels, BID33 concludes that neural networks are indeed capable of poor generalization.

However since the labels of the test set are random and completely independent of the training data, this leaves open the question of whether neural networks learn simple patterns even on random training data.

Indeed the results of BID22 establish that even in the presence of massive label noise in the training data, neural networks obtain good test accuracy on real data.

This suggests that neural networks might learn some simple patterns even with random training labels.

The first question this paper asks is (Q1): Do neural networks learn simple patterns on random training data?A second, very curious, aspect about the generalization of neural networks is the observation that increasing the size of a neural network helps in achieving better test error even if a training error of zero has already been achieved (see, e.g., BID21 ) i.e., larger neural networks have better generalization error.

This is contrary to traditional wisdom in statistical learning theory which holds that larger models give better training error but at the cost of higher generalization error.

A recent line of work proposes that the reason for better generalization of larger neural networks is implicit regularization, or in other words larger learned models are simpler than smaller learned models.

See Neyshabur (2017) for references.

The second question this paper asks is (Q2): Do larger neural networks learn simpler patterns compared to smaller neural networks when trained on real data?The third aspect about generalization that this paper considers is the widely observed phenomenon that using large minibatches for stochastic gradient descent (SGD) leads to poor generalization LeCun et al..(Q3):

Are neural networks learned with small minibatch sizes simpler compared to those learned with large minibatch sizes?All the above questions have been looked at from the point of view of flat/sharp minimizers BID11 .

Here flat/sharp corresponds to the curvature of the loss function around the learned neural network.

BID18 for true vs random data, BID24 for large vs small neural networks and BID16 for small vs large minibatch training, all look at the sharpness of minimizers in various settings and connect it to the generalization performance of neural networks.

While there certainly seems to be a connection between the sharpness of the learned neural network, there is as yet no unambiguous notion of this sharpness to quantify it.

See BID4 for more details.

This paper takes a complementary approach: it looks at the above questions through the lens of learnability.

Let us say we are considering a multi-class classification problem with c classes and let D denote a distribution over the inputs x ∼ R d .

Given a neural network N , draw n independent samples x tr 1 , · · · , x tr n from D and train a neural network N on training data DISPLAYFORM0 The learnability of a neural network N is defined to be DISPLAYFORM1 Note that L(N ) implicitly depends on D, the architecture and learning algorithm used to learn N as well as n. This dependence is suppressed in the notation but will be clear from context.

Intuitively, larger the L(N ), easier it is to learn N from data.

This notion of learnability is not new and is very closely related to probably approximately correct (PAC) learnability Valiant (1984); BID15 .

In the context of neural networks, learnability has been well studied from a theoretical point as we discuss briefly in Sec.2.

There we also discuss some related empirical results but to the best of our knowledge there has been no work investigating the learnability of neural networks that are encountered in practice.

This paper empirically investigates the learnability of neural networks of varying sizes/architectures and minibatch sizes, learned on real/random data in order to answer (Q1) and (Q2) and (Q3).

The main contributions of this paper are as follows: DISPLAYFORM2 The results in this paper suggest that there is a strong correlation between generalizability and learnability of neural networks i.e., neural networks that generalize well are more learnable compared to those that do not generalize well.

Our experiments suggest that• Neural networks do not learn simple patterns on random data.• Learned neural networks of large size/architectures that achieve higher accuracies are more learnable.• Neural networks learned using small minibatch sizes are more learnable compared to those learned using large minibatch sizes.

Experiments also suggest that there are qualitative differences between learned shallow networks and deep networks and further investigation is warranted.

Paper organization: The paper is organized as follows.

Section 2 gives an overview of related work.

Section 3 presents the experimental setup and results.

Section 5 concludes the paper with some discussion of results and future directions.

Learnability of the concept class of neural networks has been addressed from a theoretical point of view in two recent lines of work.

The first line of work shows hardness of learning by exhibiting a distribution and neural net that is hard to learn by certain type of algorithms.

We will mention one of the recent results, further information can be obtained from references therein. (see also BID26 BID25 ) show that there exist families of single hidden layer neural networks of small size that is hard to learn for statistical query algorithms (statistical query algorithms BID14 capture a large class of learning algorithms, in particular, many deep learning algorithms such as SGD).

The result holds for log-concave distributions on the input and for a wide class of activation functions.

If each sample is used only ones, then the hardness in their result means that the number of samples required is exponentially large.

These results do not seem directly applicable to input distributions and networks encountered in practice.

The second line of work shows, under various assumptions on D and/or N , that the learnability of neural networks is close to 1 Arora et al. FORMULA1 ; Janzamin et al. FORMULA1 ; BID5 BID34 .

Recently, BID6 give a provably efficient algorithm for learning one hidden layer neural networks consisting of sigmoids.

However, their algorithm, which uses the kernel method, is different from the ones used in practice and the output hypothesis is not in the form of a neural network.

Using one neural net to train another has also been used in practice, e.g. Ba & Caurana FORMULA1 ; BID10 ; BID30 .

The goal in these works is to train a small neural net to the data with high accuracy by a process often called distillation.

To this end, first a large network is trained to high accuracy.

Then a smaller network is trained on the original data, but instead of class labels, the training now uses the classification probabilities or related quantities of the large network.

Thus the goal of this line of research, while related, is different from our goal.

In this section, we will describe our experiments and present results.

All our experiments were performed on CIFAR-10 BID17 .

The 60, 000 training examples were divided into three subsets D 1 , D 2 and D 3 with D 1 and D 2 having 25000 samples each and D 3 having 10000 samples.

We overload the term D i to denote both the unlabeled as well as labeled data points in the i th split; usage will be clear from context.

For all the experiments, we use vanilla stochastic gradient descent (SGD) i.e., no momentum parameter, with an initial learning rate of 0.01.

We decrease the learning rate by a factor of 3 4 if there is no decrease in train error for the last 10 epochs.

Learning proceeds for 500 epochs or when the training zero-one error becomes smaller than 1%, whichever is earlier.

Unless mentioned otherwise, minibatch size of 64 is used and the final training zero-one error is smaller than 1%.

For training, we minimize logloss and do not use weight decay.

The experimental setup is as follows.

Step 1 Train a network N 1 on (labeled) D 1 .Step 2 Use N 1 to predict labels for (unlabeled) D 2 , denoted by N 1 (D 2 ).Step 3 Train another network N 2 on the data (D 2 , N 1 (D 2 )).Learnability of a network is computed as DISPLAYFORM0 All the numbers reported here were averaged over 5 independent runs.

We now present experimental results aimed at answering (Q1), (Q2) and (Q3) we raised in Section 1.

The first set of experiments are aimed at understanding the effect of data on the simplicity of learned neural networks.

We work with three different kinds of data.

In this section we vary the data in three ways• True data: Use labeled images from CIFAR-10 for D 1 in Step 1.• Random labels: Use unlabeled images from CIFAR-10 for D 1 in Step 1 and assign them random labels uniformly between 1 and 10.• Random images: Use random images and labels in Step 1, where each pixel in the image is drawn uniformly from [−1, 1].For this set of experiments architecture of N 1 was the same as that of N 2 .

The networks N 1 and N 2 were varied over different architectures: VGG Simonyan & Zisserman (2014) , GoogleNet Szegedy et al. (2015) , ResNet He et al. (2016a) , PreActResnet BID9 , DPN Chen et al. (2017) and DenseNet Huang et al. (2016) .

We also do the same experiment on shallow convolutional neural networks with one convolutional layer and one fully connected layer.

For the shallow networks, we vary the number of filters in N 1 and N 2 from {16, 32, 64, 128, 256, 512, 1024}. We start with 16 filters since that is the minimum number of filters where the training zero one error goes below 1%.The learnability values for various networks for true data, random labels and random images are presented in Table 1 for shallow convolutional networks, TAB1 for popular deep convolutional networks and TAB2 clearly demonstrates that the complexity of a learned neural network heavily depends on the training data.

Given that complexity of the learned model is closely related to its generalizability, this further supports the view that generalization in neural networks heavily depends on training data.

Similar results can be observed for shallow convolutional networks on CIFAR-100 in Table 4 .It is perhaps surprising that the learnability of networks trained on random data is substantially higher than 10% for shallow networks, on the other hand it's close to 10% for deeper networks.

Some of this is due to class imbalance: in the case of true data, class imbalance is minimal for all architectures.

While, when trained on random labels or random images output of N 1 on D 2 was skewed.

This happened both for shallow networks and deeper networks but was slightly higher for shallow networks.

TAB3 presents the percentage of each class in the labels of N 1 on D2.

However, we do not have a quantification of how much of learnability in the case of shallow networks arises due to class imbalance and a compelling reason for high learnability of shallow networks.

TAB6 present these results for VGG-11 and GoogleNet.

The key point we would like to point out from these tables is that if we focus on those examples where N 1 does not predict the true label correctly i.e., TLP = 0 or the first row in the tables, we see that approximately half of these examples are still learned correctly by N 2 .

Contrast this with the learnability values of N 1 learned with random data which are all less than 20%.

This suggests that networks learned on true data make simpler predictions even on examples which they misclassify.

The second set of experiments are aimed at understanding the effect of network size and architecture on the learnability of the learned neural network.

First, we work with shallow convolutional neural networks (CNN) with 1 convolutional layer and 1 fully connected layer.

The results are presented in Table 10 .

Even though training accuracy is always greater than 99%, test accuracy increases with increase in the size of N 1 -Neyshabur et al. FORMULA1 reports similar results for 2-layer multilevel perceptrons (MLP).

It is clear that for any fixed N 2 , the learnability of the learned network increases as the number of filters in N 1 increases.

This suggests that the larger learned networks are indeed simpler than the smaller learned networks.

Note also that for every N 1 , its learnability values are always larger than its test accuracy when N 2 has 16 or more filters.

This suggests that N 2 learns information about N 1 that is not contained in the data.

We performed the same experiment for some popular architectures as in Section 3.2.

The results are presented in TAB1 .

Note that the accuracies reported here are significantly lower than those reported in published literature for the corresponding models; the reason for this is that our data size is essentially cut by half (see Section 3.1).

Except for the case where N 2 is ResNet18 and N 1 is either a VGG or ResNet, there is a positive correlation between test accuracy and learnability i.e., a network with higher test accuracy is more learnable.

We do not know the reason for the exception mentioned above.

Furthermore, the pattern observed for shallow networks, that learnability is larger than accuracy, does not seem to always hold for these larger networks.

The third set of experiments are aimed at understanding the effect of minibatch size on the learned model.

For this set of experiments, N 1 and N 2 are again varied over different architectures while keeping the architectures of N 1 and N 2 same.

The minibatch size for training of N 2 (Step 3) is fixed to 64 while the minibatch size for training of N 1 (Step 1) is varied over {32, 64, 128, 256}. TAB2 presents these results.

It is clear from these results that for any architecture, increasing the minibatch size leads to a reduction in learnability.

This suggests that using a larger minibatch size in SGD leads to a more complex neural network as compared to using a smaller minibatch size.

In this section, we will explore a slightly orthogonal question of whether neural networks learned with different random initializations converge to the same neural network, as functions.

While there are some existing works e.g., BID7 , which explore linear interpolation between the parameters of two learned neural networks with different initializations, we are interested here in understanding if different SGD solutions still correspond to the same function.

In order to do this, we compute the confusion matrix for different SGD solutions.

If SGD is run k times (k = 5 in this case), recall that the (i, j) entry of the confusion matrix, where 1 ≤ i, j ≤ k gives the fraction of examples on which the i th and j th SGD solutions agree.

The following are the confusion matrices For both the networks, we see that the off-diagonal entries are quite close to each other.

This seems to suggest that while the different SGD solutions are not same as functions, they agree on a common subset (93% for shallow network and 73% for VGG-11) of examples.

Furthermore, for VGG-11, the off-diagonal entries are very close to the test accuracy -this behavior of VGG-11 seems common to other popular architectures as well.

This seems to suggest that different SGD solutions agree on precisely those examples which they predict correctly, which in turn means that the subset of examples on which different SGD solutions agree with each other are precisely the correctly predicted examples.

However this does not seem to be the case.

Figures 1 and 2 show the histograms of the number of distinct predictions for shallow network and VGG-11 respectively.

For each number i ∈ [k], it shows the fraction of examples for which the k SGD solutions make exactly i distinct predictions.

The number of examples for which there is exactly 1 prediction, or equivalently all the SGD solutions agree is significantly smaller than the test accuracies reported above.

The experimental results so far show a clear correlation between learnability and generalizability of learned neural networks.

This naturally leads to the question of why this is the case.

We hypothesize that learnability captures the inductive bias of SGD training of neural networks.

More precisely, when we start training, intuitively, the initial random network generalizes well (i.e., both train and test errors are high) and is also simple (learnability is high).

As SGD changes the network to reduce the training error, it becomes more complex (learnability decreases) and generalization error increases.

FIG2 which shows the plots of learnability and generalizability of shallow 2-layer CNNs supports this hypothesis.

DISPLAYFORM0

This paper explores the learnability of learned neural networks under various scenarios.

The results herein suggest that while learnability is often higher than test accuracy, there is a strong correlation between low generalization error and high learnability of the learned neural networks.

This paper also shows that there are some qualitative differences between shallow and popular deep neural networks.

Some questions that this paper raises are the effect of optimization algorithms, hyperparameter selection and initialization schemes on learnability.

On the theoretical front, it would be interesting to characterize neural networks that can be learned efficiently via backprop.

Given the strong correlation between learnability and generalization, driving the network to converge to learnable networks might help achieve better generalization.

<|TLDR|>

@highlight

Exploring the Learnability of Learned Neural Networks