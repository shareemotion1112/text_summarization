We prove a multiclass boosting theory for the ResNet architectures which simultaneously creates a new technique for multiclass boosting and provides a new algorithm for ResNet-style architectures.

Our proposed training algorithm, BoostResNet, is particularly suitable in non-differentiable architectures.

Our method only requires the relatively inexpensive sequential training of T "shallow ResNets".

We prove that the training error decays exponentially with the depth T if the weak module classifiers that we train perform slightly better than some weak baseline.

In other words, we propose a weak learning condition and prove a boosting theory for ResNet under the weak learning condition.

A generalization error bound based on margin theory is proved and suggests that ResNet could be resistant to overfitting using a network with l_1 norm bounded weights.

Why do residual neural networks (ResNets) BID14 and the related highway networks BID35 work?

And if we study closely why they work, can we come up with new understandings of how to train them and how to define working algorithms?Deep neural networks have elicited breakthrough successes in machine learning, especially in image classification and object recognition BID19 BID32 BID34 Zeiler & Fergus, 2014) in recent years.

As the number of layers increases, the nonlinear network becomes more powerful, deriving richer features from input data.

Empirical studies suggest that challenging tasks in image classification BID15 BID34 and object recognition BID7 BID8 BID12 BID23 often require "deep" networks, consisting of tens or hundreds of layers.

Theoretical analyses have further justified the power of deep networks BID24 compared to shallow networks.

However, deep neural networks are difficult to train despite their intrinsic representational power.

Stochastic gradient descent with back-propagation (BP) BID20 and its variants are commonly used to solve the non-convex optimization problems.

A major challenge that exists for training both shallow and deep networks is vanishing or exploding gradients BID1 BID9 .

Recent works have proposed normalization techniques BID9 BID22 BID15 BID31 to effectively ease the problem and achieve convergence.

In training deep networks, however, a surprising training performance degradation is observed BID35 BID14 : the training performance degrades rapidly with increased network depth after some saturation point.

This training performance degradation is representationally surprising as one can easily construct a deep network identical to a shallow network by forcing any part of the deep network to be the same as the shallow network with the remaining layers functioning as identity maps.

He et al. BID14 presented a residual network (ResNet) learning framework to ease the training of networks that are substantially deeper than those used previously.

And they explicitly reformulate the layers as learning residual functions with reference to the layer inputs by adding identity loops to the layers.

It is shown in BID10 that identity loops ease the problem of spurious local optima in shallow networks.

BID35 introduce a novel architecture that enables the optimization of networks with virtually arbitrary depth through the use of a learned gating mechanism for regulating information flow.

Empirical evidence overwhelmingly shows that these deep residual networks are easier to optimize than non-residual ones.

Can we develop a theoretical justification for this observation?

And does that justification point us towards new algorithms with better characteristics?

We propose a new framework, multi-channel telescoping sum boosting (defined in Section 4), to characterize a feed forward ResNet in Section 3.

We show that the top level (final) output of a ResNet can be thought of as a layer-by-layer boosting method (defined in Section 2).

Error bounds for telescoping sum boosting are provided.

We introduce a learning algorithm (BoostResNet) guaranteed to reduce error exponentially as depth increases so long as a weak learning assumption is obeyed.

BoostResNet adaptively selects training samples or changes the cost function (Section 4 Theorem 4.2).

In Section 4.4, we analyze the generalization error of BoostResNet and provide advice to avoid overfitting.

The procedure trains each residual block sequentially, only requiring that each provides a better-than-a-weak-baseline in predicting labels.

BoostResNet requires radically lower computational complexity for training than end-to-end back propagation (e2eBP).

Memorywise, BoostResNet requires only individual layers of the network to be in the graphics processing unit (GPU) while e2eBP inevitably keeps all layers in the GPU.

For example, in a state-of-the-art deep ResNet, this might reduce the RAM requirements for GPU by a factor of the depth of the network.

Similar improvements in computation are observed since each e2eBP step involves back propagating through the entire deep network.

Experimentally, we compare BoostResNet with e2eBP over two types of feed-forward ResNets, multilayer perceptron residual network (MLP-ResNet) and convolutional neural network residual network (CNN-ResNet), on multiple datasets.

BoostResNet shows substantial computational performance improvements and accuracy improvement under the MLP-ResNet architecture.

Under CNN-ResNet, a faster convergence for BoostResNet is observed.

One of the hallmarks of our approach is to make an explicit distinction between the classes of the multiclass learning problem and channels that are constructed by the learning procedure.

A channel here is essentially a scalar value modified by the rounds of boosting so as to implicitly minimize the multiclass error rate.

Our multi-channel telescoping sum boosting learning framework is not limited to ResNet and can be extended to other, even non-differentiable, nonlinear hypothesis units, such as decision trees or tensor decompositions.

1.2 RELATED WORKS Training deep neural networks has been an active research area in the past few years.

The main optimization challenge lies in the highly non-convex nature of the loss function.

There are two main ways to address this optimization problem: one is to select a loss function and network architecture that have better geometric properties (details refer to appendix A.1), and the other is to improve the network's learning procedure (details refer to appendix A.2).Many authors have previously looked into neural networks and boosting, each in a different way.

BID2 introduce single hidden layer convex neural networks, and propose a gradient boosting algorithm to learn the weights of the linear classifier.

The approach has not been generalized to deep networks with more than one hidden layer.

Shalev-Shwartz (2014) proposes a selfieBoost algorithm which boosts the accuracy of an entire network.

Our algorithm is different as we instead construct ensembles of classifiers.

Veit et al. (2016) interpret residual networks as a collection of many paths of differing length.

Their empirical study shows that residual networks avoid the vanishing gradient problem by introducing short paths which can carry gradient throughout the extent of very deep networks.

The authors of AdaNet BID4 consider ensembles of neural layers with a boosting-style algorithm and provide a method for structural learning of neural networks by optimizing over the generalization bound, which consists of the training error and the complexity of the AdaNet architecture.

AdaNet uses the traditional boosting framework where weak classifiers are being boosted.

Therefore, to obtain low training error guarantee, AdaNet maps the feature vectors (hidden layer representations) to a classifier space and boosts the weak classifiers.

Our BoostResNet, instead, boosts representations (feature vectors) over multiple channels, and therefore produces a less "bushy" architecture.

BoostResNet focuses on a ResNet architecture, provides a new training algorithm for ResNet, and proves a training error guarantee for deep ResNet architecture.

A ResNet-style architecture is a special case of AdaNet, so AdaNet generalization guarantee applies here and our generalization analysis is built upon their work.

gT (x) DISPLAYFORM0 . . .

DISPLAYFORM1 n×k ,Ṽ t ∈ R k×n and σ is a nonlinear operator such as sigmoidal function or relu function.

Similarly, in convolutional neural network residual network (CNN-ResNet), f t (·) represents the t-th convolutional module.

Then the t-th residual block outputs DISPLAYFORM2 where x is the input fed to the ResNet.

See FIG0 for an illustration of a ResNet, which consists of stacked residual blocks (each residual block contains a nonlinear module and an identity loop).Output of ResNet Due to the recursive relation specified in Equation FORMULA3 , the output of the T -th residual block is equal to the summation over lower module outputs, i.e., DISPLAYFORM3 , where g 1 (x) = x. For binary classification tasks, the final output of a ResNet given input x is rendered after a linear classifier w ∈ R n on representation g T +1 (x) (In the multiclass setting, let C be the number of classes; the linear classifier W ∈ R n×C is a matrix instead of a vector.): DISPLAYFORM4 where F (x) = w ⊤ g T +1 (x) andσ(·) denotes a map from classifier outputs (scores) to labels.

For instanceσ(z) = sign(z) for binary classification (σ(z) = arg max i z i for multiclass classification).

The parameters of a depth-T ResNet are {w, {f t (·), ∀t ∈ T }}.

A ResNet training involves training the classifier w and the weights of modules f t (·) ∀t ∈ [T ] when training examples (x 1 , y 1 ), (x 2 , y 2 ), . . .

, (x m , y m ) are available.

Boosting Boosting BID6 assumes the availability of a weak learning algorithm which, given labeled training examples, produces a weak classifier (a.k.a.

base classifier).

The goal of boosting is to improve the performance of the weak learning algorithm.

The key idea behind boosting is to choose training sets for the weak classifier in such a fashion as to force it to infer something new about the data each time it is called.

The weak learning algorithm will finally combine many weak classifiers into a single strong classifier whose prediction power is strong.

From empirical experience, ResNet remedies the problem of training error degradation (instability of solving non-convex optimization problem using SGD) in deeper neural networks.

We are curious about whether there is a theoretical justification that identity loops help in training.

More importantly, we are interested in proposing a new algorithm that avoids end-to-end back-propagation (e2eBP) through the deep network and thus is immune to the instability of SGD for non-convex optimization of deep neural networks.

As we recall from Equation (2), ResNet indeed has a similar form as the strong classifier in boosting.

The key difference is that boosting is an ensemble of estimated hypotheses whereas ResNet is an ensemble of estimated feature representations T t=1 f t (g t (x)).

To solve this problem, we introduce an auxiliary linear classifier w t on top of each residual block to construct a hypothesis module.

Formally, a hypothesis module is defined as DISPLAYFORM0 in the binary classification setting.

Therefore DISPLAYFORM1 .

We emphasize that given g t (x), we only need to train f t and w t+1 to train o t+1 (x).

In other words, we feed the output of previous residual block (g t (x)) to the current module and train the weights of current module f t (·) and the auxiliary classifier w t+1 .

Now the input, g t+1 (x), of the t + 1-th residual block is the output, f t (g t (x)) + g t (x), of the t-th residual block.

As a result, o t (x) = t−1 DISPLAYFORM2 .

In other words, the auxiliary linear classifier is common for all modules underneath.

It would not be realistic to assume a common auxiliary linear classifier, as such an assumption prevents us from training the T hypothesis module sequentially.

We design a weak module classifier using the idea of telescoping sum as follows.

Definition 3.1.

A weak module classifier is defined as DISPLAYFORM3 where DISPLAYFORM4 is a hypothesis module, and α t is a scalar.

We call it a "telescoping sum boosting" framework if the weak learners are restricted to the form of the weak module classifier.

Recall that the T -th residual block of a ResNet outputs g T +1 (x), which is fed to the top/final linear classifier for the final classification.

We show that an ensemble of the weak module classifiers is equivalent to a ResNet's final output.

We state it formally in Lemma 3.2.

For purposes of exposition, we will call F (x) the output of ResNet although aσ function is applied on top of F (x), mapping the output to the label space Y. Lemma 3.2.

Let the input g t (x) of the t-th module be the output of the previous module, i.e., g t+1 (x) = f t (g t (x)) + g t (x).

Then the summation of T weak module classifiers divided by α T +1 is identical to the output, F (x), of the depth-T ResNet, DISPLAYFORM0 where the weak module classifier h t (x) is defined in Equation (4).See Appendix B for the proof.

Overall, our proposed ensemble of weak module classifiers is a new framework that allows for sequential training of ResNet.

Note that traditional boosting algorithm results do not apply here.

We now analyze our telescoping sum boosting framework in Section 4.

Our analysis applies to both binary and multiclass, but we will focus on the binary class for simplicity in the main text and defer the multiclass analysis to the Appendix F.

Below, we propose a learning algorithm whose training error decays exponentially with the number of weak module classifiers T under a weak learning condition.

We restrict to bounded hypothesis modules, i.e., |o t (x)| ≤ 1.

The weak module classifier involves the difference between (scaled version of) o t+1 (x) and o t (x).

DISPLAYFORM0 , where D t−1 is the weight of the examples.

As the hypothesis module o t (x) is bounded by 1, we obtain |γ t | ≤ 1.

Soγ t characterizes the performance of the hypothesis module o t (x).

A natural requirement would be that o t+1 (x) improves slightly upon o t (x), and thusγ t+1 −γ t ≥ γ ′ > 0 could serve as a weak learning condition.

However this weak learning condition is too strong: even when current hypothesis module is performing almost ideally (γ t is close to 1), we still seek a hypothesis module which performs consistently better than the previous one by γ ′ .

Instead, we consider a much weaker learning condition, inspired by training error analysis, as follows.

The weak learning condition is motivated by the learning theory and it is met in practice (refer to FIG5 ).

characterizes the normalized improvement of the correlation between the true labels y and the hypothesis modules o t+1 (x) over the correlation between the true labels y and the hypothesis modules o t (x).

The condition specified in Definition 4.1 is mild as it requires the hypothesis module o t+1 (x) to perform only slightly better than the previous hypothesis module o t (x).

In residual network, since o t+1 (x) represents a depth-(t + 1) residual network which is a deeper counterpart of the depth-t residual network o t (x), it is natural to assume that the deeper residual network improves slightly upon the shallower residual network.

Whenγ t is close to 1,γ 2 t+1 only needs to be slightly better thanγ 2 t as the denominator 1−γ 2 t is small.

The assumption of the covariance between exp(−yo t+1 (x)) and exp(yo t (x)) being non-positive is suggesting that the weak module classifiers should not be adversarial, which may be a reasonable assumption for ResNet.

We now propose a novel training algorithm for telescoping sum boosting under binary-class classification as in Algorithm 1.

In particular, we introduce a training procedure for deep ResNet in Algorithm 1 & 2, BoostResNet, which only requires sequential training of shallow ResNets.

The training algorithm is a module-by-module procedure following a bottom-up fashion as the outputs of the t-th module g t+1 (x) are fed as the training examples to the next t + 1-th module.

Each of the shallow ResNet f t (g t (x)) + g t (x) is combined with an auxiliary linear classifier w t+1 to form a hypothesis module o t+1 (x).

The weights of the ResNet are trained on these shallow ResNets.

The telescoping sum construction is the key for successful interpretation of ResNet as ensembles of weak module classifiers.

The innovative introduction of the auxiliary linear classifiers (w t+1 ) is the key solution for successful multi-channel representation boosting with theoretical guarantees.

Auxiliary linear classifiers are only used to guide training, and they are not included in the model (proved in Lemma 3.2).

This is the fundamental difference between BoostResNet and AdaNet.

AdaNet BID4 maps the feature vectors (hidden layer representations) to a classifier space and boosts the weak classifiers.

Our framework is a multi-channel representation (or information) boosting rather than a traditional classifier boosting.

Traditional boosting theory does not apply in our setting.

Algorithm 1 BoostResNet: telescoping sum boosting for binary-class classification Input: m labeled samples [(x i , y i )] m where y i ∈ {−1, +1} and a threshold γ Output: {f t (·), ∀t} and DISPLAYFORM0 t ← t + 1 8: end while 9: T ← t − 1 Theorem 4.2. [ Training error bound ]

The training error of a T -module telescoping sum boosting framework using Algorithms 1 and 2 decays exponentially with the number of modules T , DISPLAYFORM1 if ∀t ∈ [T ] the weak module classifier h t (x) satisfies the γ-weak learning condition defined in Definition 4.1.The training error of Algorithms 1 and 2 is guaranteed to decay exponentially with the ResNet depth even when each hypothesis module o t+1 (x) performs slightly better than its previous hypothesis module o t (x) (i.e., γ > 0).

Refer to Appendix F for the algorithm and theoretical guarantees for multiclass classification.

In Algorithm 2, the implementation of the oracle at line 1 is equivalent to DISPLAYFORM0 The minimization problem over f corresponds to finding the weights of the t-th nonlinear module of the residual network.

Auxiliary classifier w t+1 is used to help solve this minimization problem with the guidance of training labels y i .

However, the final neural network model includes none of the auxiliary classifiers, and still follows a standard ResNet structure (proved in Lemma 3.2).

In practice, there are various ways to implement Equation (6).

For instance, Janzamin et.

al. BID16 propose a tensor decomposition technique which decomposes a tensor formed by some transformation of the features x combined with labels y and recovers the weights of a one-hidden layer neural network with guarantees.

One can also use back-propagation as numerous works have shown that gradient based training are relatively stable on shallow networks with identity loops BID10 BID14 .

It is worth noting that BoostResNet training is memory efficient as the training process only requires parameters of two consecutive residual blocks to be in memory.

Given that the limited GPU memory being one of the main bottlenecks for computational efficiency, BoostResNet requires significantly less training time than e2eBP in deep networks as a result of reduced communication overhead and the speed-up in shallow gradient forwarding and back-propagation.

Let M 1 be the memory required for one module, and M 2 be the memory required for one linear classifier, the memory consumption is M 1 + M 2 by BoostResNet and M 1 T + M 2 by e2eBP.

Let the flops needed for gradient update over one module and one linear classifier be C 1 and C 2 respectively, the computation cost is C 1 + C 2 by BoostResNet and C 1 T + C 2 by e2eBP.

In this section, we analyze the generalization error to understand the possibility of overfitting under Algorithm 1.

The strong classifier or the ResNet is F (x) =.

Now we define the margin for example (x, y) as yF (x).

For simplicity, we consider MLP-ResNet with n multiple channels and assume that the weight vector connecting a neuron at layer t with its preceding layer neurons is l 1 norm bounded by Λ t,t−1 .

Recall that there exists a linear classifier w on top, and we restrict to l 1 norm bounded classifiers, i.e., w 1 ≤ C 0 < ∞. The expected training examples are l ∞ norm bounded r ∞ def = E S∼D max i∈[m] x i ∞ < ∞. We introduce Corollary 4.3 which follows directly from Lemma 2 of BID4 .

DISPLAYFORM0 where This corollary suggests that stronger weak module classifiers which produce higher accuracy predictions and larger edges, will yield larger margins and suffer less from overfitting.

The larger the value of θ, the smaller the term 4C0r∞ θ log(2n) 2m DISPLAYFORM1 DISPLAYFORM2 With larger edges on the training set and whenγ T +1 < 1, we are able to choose larger values of θ while keeping the error term zero or close to zero.

We compare our proposed BoostResNet algorithm with e2eBP training a ResNet on the MNIST (LeCun et al., 1998), street view house numbers (SVHN) BID27 , and CIFAR-10 (Krizhevsky & Hinton, 2009) benchmark datasets.

Two different types of architectures are tested: a ResNet where each module is a fully-connected multi-layer perceptron (MLP-ResNet) and a more common, convolutional neural network residual network (CNN-ResNet).

In each experiment the architecture of both algorithms is identical, and they are both initialized with the same random seed.

As a baseline, we also experiment with standard boosting (AdaBoost.

MM Mukherjee & Schapire (2013) ) of convolutional modules in appendix H.2 for SVHN and CIFAR-10 datasets.

Our experiments are programmed in the Torch deep learning framework for Lua and executed on NVIDIA Tesla P100 GPUs.

All models are trained using the Adam variant of SGD BID17 .MLP-ResNet on MNIST The MNIST database BID21 of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.

The data contains ten classes.

We test the performance of BoostResNet on MLP-ResNet using MNIST dataset, and compare it with e2eBP baseline.

Each residual block is composed of an MLP with a single, 1024-dimensional hidden layer.

The training and test error between BoostResNet and e2eBP is in FIG3 as a function of depth.

Surprisingly, we find that training error degrades for e2eBP, although the ResNet's identity loop is supposed to alleviate this problem.

Our proposed sequential training procedure, BoostResNet, relieves gradient instability issues, and continues to perform well as depth increases.

BID27 ) is a real-world image dataset, obtained from house numbers in Google Street View images.

The dataset contains over 600,000 training images, and about 20,000 test images.

We fit a 50-layer, 25-residual-block CNN-ResNet using both BoostResNet and e2eBP (figure 3a).

Each residual block is composed of a CNN using 15 3 × 3 filters.

We refine the result of BoostResNet by initializing the weights using the result of BoostResNet and run end-to-end back propagation (e2eBP).

From figure 3a, our BoostResNet converges much faster (requires much fewer gradient updates) than e2eBP.

The test accuracy of BoostResNet is comparable with e2eBP.

10 7 10 8 10 9Number of Gradient Updates

The CIFAR-10 dataset is a benchmark dataset composed of 10 classes of small images, such as animals and vehicles.

It consists of 50,000 training images and 10,000 test images.

We again fit a 50-layer, 25-residual-block CNN-ResNet using both BoostResNet and e2eBP (figure 3b).

BoostResNet training converges to the optimal solution faster than e2eBP.

Unlike in the previous two datasets, the efficiency of BoostResNet comes at a cost when training with CIFAR-10.

We find that the test accuracy of the e2eBP refined BoostResNet to be slightly lower than that produced by e2eBP.

The weak learning condition (Definition 4.1) inspired by learning theory is checked in FIG5 .

The required better than random guessing edge γ t is depicted in FIG5 , it is always greater than 0 and our weak learning condition is thus non-vacuous.

In FIG5 , the representations we learned using BoostResNet is increasingly better (for this classification task) as the depth increases.

Our proposed BoostResNet algorithm achieves exponentially decaying (with the depth T ) training error under the weak learning condition.

BoostResNet is much more computationally efficient compared to end-to-end back-propagation in deep ResNet.

More importantly, the memory required by BoostResNet is trivial compared to end-to-end back-propagation.

It is particularly beneficial given the limited GPU memory and large network depth.

Our learning framework is natural for nondifferentiable data.

For instance, our learning framework is amenable to take weak learning oracles using tensor decomposition techniques.

Tensor decomposition, a spectral learning framework with theoretical guarantees, is applied to learning one layer MLP in BID16 .

We plan to extend our learning framework to non-differentiable data using general weak learning oracles.

In neural network optimization, there are many commonly-used loss functions and criteria, e.g., mean squared error, negative log likelihood, margin criterion, etc.

There are extensive works BID7 BID30 BID37 on selecting or modifying loss functions to prevent empirical difficulties such as exploding/vanishing gradients or slow learning BID0 .

However, there are no rigorous principles for selecting a loss function in general.

Other works consider variations of the multilayer perceptron (MLP) or convolutional neural network (CNN) by adding identity skip connections BID14 , allowing information to bypass particular layers.

However, no theoretical guarantees on the training error are provided despite breakthrough empirical successes.

Hardt et al. BID10 have shown the advantage of identity loops in linear neural networks with theoretical justifications; however the linear setting is unrealistic in practice.

There have been extensive works on improving BP BID20 .

For instance, momentum (Qian, 1999), Nesterov accelerated gradient BID26 , Adagrad BID5 and its extension Adadelta (Zeiler, 2012) .

Most recently, Adaptive Moment Estimation (Adam) BID17 , a combination of momentum and Adagrad, has received substantial success in practice.

All these methods are modifications of stochastic gradient descent (SGD), but our method only requires an arbitrary oracle, which does not necessarily need to be an SGD solver, that solves a relatively simple shallow neural network.

Proof.

In our algorithm, the input of the next module is the output of the current module DISPLAYFORM0 we thus obtain that each weak learning module is DISPLAYFORM1 and similarly DISPLAYFORM2 Therefore the sum over h t (x) and h t+1 (x) is DISPLAYFORM3 And we further see that the weighted summation over all h t (x) is a telescoping sum DISPLAYFORM4 C PROOF FOR THEOREM 4.2: BINARY CLASS TELESCOPING SUM BOOSTING

Proof.

We will use a 0-1 loss to measure the training error.

In our analysis, the 0-1 loss is bounded by exponential loss.

The training error is therefore bounded by DISPLAYFORM0 where DISPLAYFORM1 We choose α t+1 to minimize Z t .

DISPLAYFORM2 Furthermore each learning module is bounded as we see in the following analysis.

We obtain DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 Equation FORMULA5 is due to the non-positive correlation between exp(−yo t+1 (x)) and exp(yo t (x)).

Jensen's inequality in Equation (27) holds only when |y i o t+1 (x i )| ≤ 1 which is satisfied by the definition of the weak learning module.

The algorithm chooses α t+1 to minimize Z t .

We achieve an upper bound on Z t , DISPLAYFORM6 by minimizing the bound in Equation (30) DISPLAYFORM7 DISPLAYFORM8 Therefore over the T modules, the training error is upper bounded as follows DISPLAYFORM9 Overall, Algorithm 1 leads us to consistent learning of ResNet.

Rademacher complexity technique is powerful for measuring the complexity of H any family of functions h : X → R, based on easiness of fitting any dataset using classifiers in H (where X is any space).

Let S =< x 1 , . . .

, x m > be a sample of m points in X .

The empirical Rademacher complexity of H with respect to S is defined to be DISPLAYFORM0 where σ is the Rademacher variable.

The Rademacher complexity on m data points drawn from distribution D is defined by BID3 ) Let H be a hypothesis set admitting a decomposition H = ∪ l i=1 H i for some l > 1.

H i are distinct hypothesis sets.

Let S be a random sequence of m points chosen independently from X according to some distribution D. For θ > 0 and any H = T t=1 h t , with probability at least 1 − δ, DISPLAYFORM1 DISPLAYFORM2 Lemma D.2.

Leth =w ⊤f , wherew ∈ R n ,f ∈ R n .

LetH andF be two hypothesis sets, and h ∈H ,f j ∈F , ∀j ∈ [n].

The Rademacher complexity ofH andF with respect to m points from D are related as follows DISPLAYFORM3

Let n be the number of channels in ResNet, i.e., the number of input or output neurons in a module f t (g t (x)).

We have proved that ResNet is equivalent as DISPLAYFORM0 We define the family of functions that each neuron f t,j , ∀j ∈ [n] belong to as DISPLAYFORM1 where u t−1,j denotes the vector of weights for connections from unit j to a lower layer t − 1, σ • f t−1 denotes element-wise nonlinear transformation on f t−1 .

The output layer of each module is connected to the output layer of previous module.

We consider 1-layer modules for convenience of analysis.

Therefore in ResNet with probability at least 1 − δ, DISPLAYFORM2 for all f t ∈ F t .Define the maximum infinity norm over samples as r ∞ def = E S∼D max i∈ [m] x i ∞ and the product of l 1 norm bound on weights as DISPLAYFORM3 According to lemma 2 of BID4 , the empirical Rademacher complexity is bounded as a function of r ∞ , Λ t and n: DISPLAYFORM4 Overall, with probability at least 1 − δ, DISPLAYFORM5 for all f t ∈ F t .

with probability at least 1 − δ for β(θ, m, T, δ) DISPLAYFORM0 Now the proof for Theorem E is the following.

Proof.

The fraction of examples in sample set S being smaller than θ is bounded DISPLAYFORM1 To bound exp(θα T +1 ) = ( DISPLAYFORM2 ) θ , we first boundγ T +1 : We know that DISPLAYFORM3 = (1 + 2 DISPLAYFORM4 DISPLAYFORM5 As DISPLAYFORM6

Recall that the weak module classifier is defined as DISPLAYFORM0 where o t (x) ∈ ∆ C−1 .The weak learning condition for multi-class classification is different from the binary classification stated in the previous section, although minimal demands placed on the weak module classifier require prediction better than random on any distribution over the training set intuitively.

We now define the weak learning condition.

It is again inspired by the slightly better than random idea, but requires a more sophisticated analysis in the multi-class setting.

In order to characterize the training error, we introduce the cost matrix C ∈ R m×C where each row denote the cost incurred by classifying that example into one of the C categories.

We will bound the training error using exponential loss, and under the exponential loss function defined as in Definition G.1, the optimal cost function used for best possible training error is therefore determined.

where DISPLAYFORM0 .

A multi-class DISPLAYFORM1 We propose a novel learning algorithm using the optimal edge-over-random cost function for training ResNet under multi-class classification task as in Algorithm 3.

DISPLAYFORM2 if the weak module classifier h t (x) satisfies the γ-weak learning condition ∀t ∈ [T ].The exponential loss function defined as in Definition G.1 DISPLAYFORM3 Update cost function DISPLAYFORM4 t ← t + 1 9: end while 10: T ← t − 1 Algorithm 4 BoostResNet: oracle implementation for training a ResNet module (multi-class) Input: g t (x),s t ,o t (x) and α t Output: DISPLAYFORM5

We implement an oracle to minimize DISPLAYFORM0 e st(xi,l)−st(xi,yi) e ht(xi,l)−ht(xi,yi) given current state s t and hypothesis module o t (x).

Therefore minimizing Z t is equivalent to the following.

DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Define the accumulated weak learner DISPLAYFORM4 , the loss for a T -module multiclass ResNet is thus DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 Note that Z 0 = 1 m as the initial accumulated weak learners s 0 (x i , l) = 0.

The loss fraction between module t and t − 1, DISPLAYFORM8 The Z t is bounded DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11 DISPLAYFORM12 Therefore DISPLAYFORM13 The algorithm chooses α t+1 to minimize Z t .

We achieve an upper bound on Z t , Therefore over the T modules, the training error is upper bounded as follows DISPLAYFORM14 Overall, Algorithm 3 and 4 leads us to consistent learning of ResNet.

We investigate e2eBP training performance on various depth ResNet.

Surprisingly, we observe a training error degradation for e2eBP although the ResNet's identity loop is supposed to alleviate this problem.

Despite the presence of identity loops, the e2eBP eventually is susceptible to spurious local optima.

This phenomenon is explored further in FIG8 and 5b, which respectively show how training and test accuracies vary throughout the fitting process.

Our proposed sequential training procedure, BoostResNet, relieves gradient instability issues, and continues to perform well as depth increases.

Besides e2eBP, we also experiment with standard boosting (AdaBoost.

MM Mukherjee & Schapire (2013) ), as another baseline, of convolutional modules.

In this experiment, each weak learner is a residual block of the ResNet, paired with a classification layer.

We do 25 rounds of AdaBoost.

MM and train each weak learner to convergence.

TAB7 exhibit a comparison of BoostResNet, e2eBP and AdaBoost performance on SVHN and CIFAR-10 dataset respectively.

On SVHN dataset, the advantage of BoostResNet over e2eBP is obvious.

Using 3 × 10 8 number of gradient updates, BoostResNet achieves 93.8% test accuracy whereas e2eBP obtains a test accuracy of 83%.

The training and test accuracies of SVHN are listed in TAB7 .

BoostResNet training allows the model to train much faster than end-to-end training, and still achieves the same test accuracy when refined with e2eBP.

To list the hyperparameters we use in our BoostResNet training after searching over candidate hyperparamters, we choose learning rate to be 0.004 with a 9 × 10 −5 learning rate decay.

The gamma threshold is set to be 0.001 and the initial gamma value on SVHN is 0.75.On CIFAR-10 dataset, the main advantage of BoostResNet over e2eBP is the speed of training.

BoostResNet refined with e2eBP obtains comparable results with e2eBP.

This is because we are using a suboptimal architecture of ResNet which overfits the CIFAR-10 dataset.

AdaBoost, on the other hand, is known to be resistant to overfitting.

Therefore, AdaBoost achieves the highest test accuracy on CIFAR-10.

To list the hyperparameters we use in our BoostResNet training after searching over candidate hyperparamters, we choose learning rate to be 0.014 with a 3.46 × 10 −5 learning rate decay.

The gamma threshold is set to be 0.007 and the initial gamma value on CIFAR-10 is 0.93.

Table 2 : Accuracies of CIFAR-10 task.

NGU is the number of gradient updates taken by the algorithm in training.

@highlight

We prove a multiclass boosting theory for the ResNet architectures which simultaneously creates a new technique for multiclass boosting and provides a new algorithm for ResNet-style architectures.

@highlight

Presents a boosting-style algorithm for training deep residual networks, a convergence analysis for training error, and a analysis of generalization ability.

@highlight

A learning method for ResNet using the boosting framework that decomposes the learning of complex networks and uses less computational costs.

@highlight

Authors propose the deep ResNet as a boosting algorithm, and they claim this is more efficient than standard end-to-end backropagation.