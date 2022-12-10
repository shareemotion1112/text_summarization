The main goal of network pruning is imposing sparsity on the neural network by increasing the number of parameters with zero value in order to reduce the architecture size and the computational speedup.

Recent advances in deep neural networks came with ideas to train deep architectures that have led to near-human accuracy for image recognition, object categorization and a wide variety of other applications LeCun et al. (2015) ; Maturana & Scherer (2015) ; Schmidhuber (2015) ; Mnih et al. (2013) ; .

One possible issue is that an over-parameterized network may make the architecture overcomplicated for the task at hand and it might be prone to over-fitting as well.

In addition to the model complexity, a huge amount of computational power is required to train such deep models due to having billions of weights.

Moreover, even if a huge model is trained, it cannot be effectively employed for model evaluation on low-power devices mainly due to having exhaustive matrix multiplications Courbariaux et al. (2015) .

So far, a wide variety of approaches have been proposed for creating more compact models.

Traditional methods include model compression Ba & Caruana (2014) ; , network pruning Han et al. (2015b) , sparsity-inducing regularizer Collins & Kohli (2014) , and low-rank approximation Jaderberg et al. (2014) ; Denton et al. (2014) ; Ioannou et al. (2015) ; Tai et al. (2015) .

The aforementioned methods usually induce random connection pruning which yields to few or no improvement in the computational cost.

On the other hand, structured pruning methods proposed to compress the architecture with significant computational efficiency Wen et al. (2016) ; Neklyudov et al. (2017) .

One of the critical subjects of interest in sparsity learning is to maintain the accuracy level.

In this paper, we discuss the intuitive reasons behind the accuracy drop and propose a method to prevent it.

The important step is to determine how the sparsity and accuracy are connected together in order to be able to propose a mechanism for controlling the sparsity to prevent severe accuracy drop.

In order to connect the sparsity to accuracy, intuitively, the accuracy drop is caused by imposing too much sparsity on the network in a way that the remaining elements cannot transfer enough information for optimal feature extraction for the desired task.

Another intuitive reasoning is to argue that the sparsity is not supervised with any attention towards the model performance during optimization.

For effective network pruning and feature selection, different approaches such as employing the group lasso for sparse structure learning Yuan & Lin (2006) , structure scale constraining Liu et al. (2015) , and structured regularizing deep architectures known as Structured Sparsity Learning (SSL) Wen et al. (2016) have previously been proposed.

For most of the previous research efforts, there is lack of addressing the direct effect of the proposed method on the combination of the sparsity and accuracy drop.

One may claim that successful sparsity imposing with negligible accuracy drop might be due to the initial over-parameterizing the network.

Moreover, there is no control mechanism to supervise the sparsity operation connected to the model performance which limits the available methods to intensive hyper-parameter tuning and multiple stages of training.

Our contribution.

We designed and employed a supervised attention mechanism for sparsity learning which: (1) performs model compression for having less number of parameters (2) prevents the accuracy drop by sparsity supervision by paying an attention towards the network using variance regularization and (3) is a generic mechanism that is not restricted by the sparsity penalty or any other limiting assumption regarding the network architecture.

To the best of our knowledge, this is the first research effort which proposes a supervised attention mechanism for sparsity learning.

Paper Organization.

At first, we provide a review of the related research efforts (Section 2).

Then, we introduce the attention mechanism which is aimed at forcing some sections of the network to be active (Section 3).

Later in Section 4, we propose an algorithm only for the attention supervision.

We complement our proposed method in Section 5, by providing experimental results for which we target the sparsity level, accuracy drop and robustness of the model to hyper-parameter tuning.

As will be observed, the proposed mechanism prevents the severe accuracy drop in higher levels of sparsity.

We will empirically show the robustness to exhaustive hyper-parameter tuning in addition to performance superiority of the proposed method in higher sparsity levels.

Network weights pruning.

Network compression for parameter reduction has been of great interest for a long time and a large number of research efforts are dedicated to it.

In Han et al. (2015b; a) ; Ullrich et al. (2017) ; , network pruning has been performed with a significant reduction in parameters, although they suffer from computational inefficiency due to the mere weight pruning rather than the structure pruning.

Network structure pruning.

In Louizos et al. (2017a) ; Wen et al. (2016); Neklyudov et al. (2017) , pruning the unimportant parts of the structure 1 rather than simple weight pruning has been proposed and significant computational speedup has been achieved.

However, for the aforementioned methods, the architecture must be fully trained at first and the potential training speedup regarding the sparsity enforcement cannot be attained.

A solution for training speedup has been proposed by 0 -regularization technique by using online sparsification Louizos et al. (2017b) .

Training speedup is of great importance but adding a regularization for solely speeding up the training (because of the concurrent network pruning) is not efficient due to adding a computational cost for imposing 0 -regularization itself.

Instead, we will use an adaptive gradient clipping Pascanu et al. (2013) for training speedup.

Attention.

In this paper, the goal is to impose the sparsity in an accurate and interpretable way using the attention mechanism.

So far, attention-based deep architectures has been proposed for image Fu et al. (2017); Jia et al. (2015) ; Mnih et al. (2014) ; Xu et al. (2015) and speech domains Bahdanau et al. (2016); Chorowski et al. (2015) ; Toro et al. (2005) , as well as machine translation Bahdanau et al. (2014); Luong et al. (2015) ; Vaswani et al. (2017) .

Recently, the supervision of the attention mechanism became a subject of interest as well Liu et al. (2016; ; ; Mi et al. (2016) for which they proposed to supervise the attention using some external guidance.

We propose the use of guided attention for enforcing the sparsity to map the sparsity distribution of the targeted elements 2 to the desired target distribution.

The main objective of the attention mechanism is to control and supervise the sparsity operation.

For this aim, it is necessary to propose a method which is neither dependent on the architecture of the model nor to any layer type while maintaining the model accuracy and enforcing the compression objective.

Considering the aforementioned goals, we propose the variance loss as an auxiliary cost term to force the distribution of the weights 3 to be skewed.

A skewed distribution with a high variance and a concentration on zero (to satisfy the sparsity objective) is desired.

Our proposed scheme supervises the sparsity operation to keep a portion of the targeted elements (such as weights) to be dominant (with respect to their magnitude) as opposed to the other majority of the weights to simultaneously impose and control sparsity.

Intuitively, this mechanism may force a portion of weights to survive for sufficient information transmission through the architecture.

Assume enforcing the sparsity is desired on a parametric model; let's have the training samples with {x i , y i } as pairs.

We propose the following objective function which is aimed to create a sparse structure in addition to the variance regularization:

in which Γ(.) corresponds to the cross-entropy loss function and θ can be any combination of the target parameters.

Model function is defined as F (.), R(.) is some regularization function, G(.) and H(.) are some arbitrary functions 4 on parameters (such as grouping parameters), N is the number of samples, λ parameters are the weighting coefficients for the associated losses and (.) and Ψ(.) are the sparsity and variance functions 5 , respectively.

The variance function is the utilized regularization term for any set of θ parameters 6 .

The inverse operation on top of the Ψ(.) in Eq. 1 is necessary due to the fact that the higher variance is the desired objective.

The power of the variance as a regularizer has been investigated in Namkoong & Duchi (2017) .

In this work, we expand the scope of variance regularization to the sparsity supervision as well.

Adding a new term to the loss function can increase the model complexity due to adding a new hyper-parameter (the coefficient of the variance loss).

For preventing this issue, we propose to have a dependent parameter as the variance loss coefficient.

If the new hyperparameter is defined in terms of a variable dependent upon another hyperparameter, then it does not increase the model complexity.

Considering the suggested approach, a dependency function must be defined over the hyperparameters definition.

The dependency is defined as λ v = f (λ s ) = α × λ s in which α is a scalar multiplier.

Group Sparsity.

Group sparsity has widely been utilized mostly for its feature selection ability by deactivating neurons 7 via imposing sparsity on the whole cluster of weights in a group Yuan & Lin (2006) ; Meier et al. (2008) .

Regarding the Eq. 1, the group sparsity objective function can be defined by following expression:

in which w (j) is the j th group of weights in w and |w (j) | is the number of weights in the associated group in the case of having M groups.

The l indicates the layer index, |G(W l )| is a normalizer factor which is in essence the number of groups for the l th layer and (.) l demonstrates the elements (weights) belonging to the the l th layer.

Structured attention.

We propose a Structured Attention (SA) regularization, which adds the attention mechanism on group sparsity learning (the sparsity imposition operation is similar to SSL Wen et al. (2016) ).

The attention is on the predefined groups.

Under our general framework, it can be expressed by the following substitutions in Eq. 1:

which is simply the variance of the group values for each layer, normalized by a factor and aggregated for all the layers.

Generalizability.

It is worth noting that our proposed mechanism is not limited to the suggested structured method.

It can operate on any (.) function as sparsity objective because the definition of the attention is independent of the type of the sparsity.

As an example, one can utilize an unstructured attention which is simply putting the attention function Ψ(.) on all the network weights without considering any special groups of weights or prior objectives such as pruning unimportant channels or filters.

The attention mechanism observes the areas of structure 8 on which the sparsity is supposed to be enforced.

we propose the Guided Attention in Sparsity Learning (GASL) mechanism, which aims at the attention supervision toward mapping the distribution of the elements' values to a certain target distribution.

The target distribution characteristics must be aligned with the attention objective function with the goal of increasing the variance of the elements for sparsity imposition.

Assume we have the vector

T that is the values of the elements in the group [θ] = {θ 1 , θ 2 , ..., θ |θ| } and for which we want to maximize the variance.

In Paisley et al. (2012) , variational Bayesian inference has been employed for the gradient computation of variational lower bound.

Inspired by Wang et al. (2013) , in which random vectors are used for stochastic gradient optimization, we utilize the additive random vectors for variance regularization.

The random vector is defined as

The formulation is as below:

where M is a |θ| × |θ| matrix.

The resulted vectorV (θ) does not make any changes in the mean of the parameters distribution since it has the same mean as the initial V (θ) vector.

Proof.

For that aim, the task breaks to the subtask of finding the optimal M for which the trace of theV (θ) is maximized.

For the mini-batch optimization problem, we prove that the proposed model is robust to the selection of M .

The maximizer M can be obtained when the trace of the variance matrix is maximized and it can be demonstrated as follows:

As can be observed from Eq. 5, as long as M is a positive definite matrix, the additive random can add to the value of the matrix trace without any upper bound.

The detailed mathematical proof is available in the Appendix.

Considering the mathematical proof, one can infer that the mere utilization of the variance loss term in Eq. 1, as the attention mechanism and without the additive random vector, can supervise the sparsity operation.

However, we will empirically show that the additive random vectors can improve the accuracy due to the supervision of the attention.

The supervision of the variance is important regarding the fact that the high variance of the parameters may decrease the algorithm speed and performance for sparsity enforcement.

This is due to the large number of operations that is necessary for gradient descent to find the trade-off between the sparsity and variance regularization.

From now one, without the loss of generality, we assume M to be identity matrix in our experiments.

In practice, the choice of V r should be highly correlated with V .

Furthermore, Eq. 5 shows that without being correlated, the terms associated with Cov[V r (θ), V (θ)] may go to zero which affects the high variance objective, negatively.

The algorithm for random vector selection is declared in Algorithm.

1.

The distribution pdf (.) should be specified regarding the desired output distribution.

We chose log-normal distribution due to its special characteristics which create a concentration around zero and a skewed tail Limpert et al. (2001) .

If the variance of the random vector V r (θ) is less than the main vector V (θ), no additive operation will be performed.

In case the [θ] parameters variance is high-enough compared to the V (θ) vector, then there is no need for additive random samples.

This preventing mechanism has been added due to the practical speedup.

Replacement operation: ReplaceV (θ) with V (θ); Return:V (θ); else Return: V (θ); Computation: Update gradient; 4.3 COMBINATION OF GASL AND SA GASL algorithm can operate on the top of the structured attention for attention supervision.

The schematic is depicted in Fig. 1 .

Furthermore, a visualized example of the output channels from the second convolutional layer in the MNIST experiments has also demonstrated in Fig. 1 .

The structured attention is dedicated to the output channels of a convolutional layer.

Figure 1: The combination of GASL and structured attention.

The cube demonstrates the output feature map of a convolutional layer.

The weights associated with each channel, form a group.

For visualization, the right column is the activation visualization of the attention-based sparsity enforcement on output channels and the left one is the results of imposing sparsity without attention.

As can be observed, some of the channels are turned off and the remaining ones are intensified.

We use three databases for the evaluation of our proposed method: MNIST LeCun et al. (1998) , CIFAR-10 and CIFAR-100 Krizhevsky & Hinton (2009) .

For increasing the convergence speed without degrading the overall performance, we used gradient clipping Pascanu et al. (2013) .

A common approach is to clip individual gradients to some fixed predefined range [−ζ, ζ].

As the learning rate becomes smaller continuously, the effective gradient 9 will approach zero and training convergence may become extremely slow.

For tackling this issue, we used the method proposed in Kim et al. (2016) for gradient clipping which defined the range dynamically as [−ζ/γ, ζ/γ] for which γ is the current learning rate.

We chose ζ = 0.1 in our experiments.

Hyper-parameters are selected by cross-validation.

For all our experiments, the output channels of convolutional layers and neurons in fully connected layers are considered as groups.

For experiments on MNIST dataset, we use 2 -regularization with the default hyperparameters.

Two network architectures have been employed: LeNet-5-Caffe 10 and a multilayer perceptron (MLP).

For the MLP network, the group sparsity is enforced on each neuron's outputs for feature selection; Same can be applied on the neurons' inputs as well.

The results are shown in Table.

1.

The percentage of sparsity is reported layer-wise.

One important observation is the superiority of the SA method to the SSL in terms of accuracy, while the sparsity objective function for both is identical and the only difference is the addition of structural attention (Eq. 3).

As a comparison to Louizos et al. (2017a) , we achieved closed sparsity level with better accuracy for the MLP network.

For the LeNet network, we obtained the highest sparsity level with competitive accuracy compared to the best method proposed in .

9 Which is gradient × learning rate 10 https://github.com/BVLC/caffe/blob/master/examples/mnist Table 1 : Experiments on LeNet-5-Caffe architecture with 20-50-800-500 number of output filters and hidden layers and MLP with the architecture of 784-500-300 as the number of hidden units for each layer.

The sparsity level is reported layer-wise.

The sparsity and error are both reported as %.

For experiments in this section, we used VGG-16 Simonyan & Zisserman (2014) as the baseline architecture.

Random cropping, horizontal flipping, and per-image standardization have been performed for data augmentation in the training phase and in the evaluation stage, only center cropping has been used Krizhevsky et al. (2012) .

Batch-normalization has also been utilized after each convolutional layer and before the activation Ioffe & Szegedy (2015) .

The initial learning rate of 0.1 has been chosen and the learning rate is dropped by a factor of 10 when the error plateaus for 5 consecutive epochs.

As can be observed from Table.

2, the combination of the GASL algorithm and SA dominates regarding the achieved sparsity level and demonstrates competitive results in terms of accuracy for Cifar-100.

We terminate the training after 300 epochs or if the averaged error is not improving for 20 consecutive epochs, whichever comes earlier.

For Cifar-10 Krizhevsky et al. (2010), we obtained the second best results for both accuracy and sparsity level.

The advantage of the proposed method for higher sparsity levels.

For Cifar-100 experiments, we continued the process of enforcing sparsity for achieving the desired level of compression 11 .

We chose three discrete level of sparsity and for any of which, the accuracy drop for different methods is reported.

Table.

3 demonstrates the comparison of different methods with regard to their accuracy drops at different levels of sparsity.

For some levels of sparsity, it was observed that some methods performed better than the baseline.

We deliberately selected higher levels of sparsity for having some performance drop as opposed to the baseline for all the implemented methods.

As can be observed, our method shows its performance superiority in accuracy for the higher levels of sparsity.

In another word, the proposed method outperforms in preventing the accuracy drop in the situation of having high sparsity level.

Robustness to the hyperparameter tuning.

Regarding the discussion in Section 3.1, it is worth to investigate the effect of λ v on the accuracy drop.

In another word, we investigate the relative importance of tuning the variance loss coefficient.

The accuracy drop is reported for Cifar-100 experiments using different α values and sparsity levels.

The results depicted in Table.

4, empirically shows the robustness of the proposed method to the selection of α, as the dependent factor, for which in the dynamic range of [0.1, 10], the accuracy drop is not changing drastically.

This clearly demonstrates the robustness of the proposed method to the selection of the new hyperparameter associated with the attention mechanism as it is only a dependent factor to the sparsity penalty coefficient.

In this paper, we proposed a guided attention mechanism for controlled sparsity enforcement by keeping a portion of the targeted elements to be alive.

The GASL algorithm has been utilized on top of the structured attention for attention supervision to prune unimportant channels and neurons of the convolutional and fully-connected layers.

We demonstrated the superiority of the method for preventing the accuracy drop in high levels of sparsity.

Moreover, it has been shown that regardless of adding a new term to the loss function objective, the model complexity remains the same and the proposed approach is relatively robust to exhaustive hyper-parameter selection.

Without the loss of generality, the method can be adapted to any layer type and different sparsity objectives such as weight pruning for unstructured sparsity or channel, neuron or filter cancellation for structured sparsity.

@highlight

Proposing a novel method based on the guided attention to enforce the sparisty in deep neural networks.