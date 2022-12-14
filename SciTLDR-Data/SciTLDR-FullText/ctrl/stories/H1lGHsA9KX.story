Determining the appropriate batch size for mini-batch gradient descent is always time consuming as it often relies on grid search.

This paper considers a resizable mini-batch gradient descent (RMGD) algorithm based on a multi-armed bandit that achieves performance equivalent to that of best fixed batch-size.

At each epoch, the RMGD samples a batch size according to a certain probability distribution proportional to a batch being successful in reducing the loss function.

Sampling from this probability provides a mechanism for exploring different batch size and exploiting batch sizes with history of success.

After obtaining the validation loss at each epoch with the sampled batch size, the probability distribution is updated to incorporate the effectiveness of the sampled batch size.

Experimental results show that the RMGD achieves performance better than the best performing single batch size.

It is surprising that the RMGD achieves better performance than grid search.

Furthermore, it attains this performance in a shorter amount of time than grid search.

Gradient descent (GD) is a common optimization algorithm for finding the minimum of the expected loss.

It takes iterative steps proportional to the negative gradient of the loss function at each iteration.

It is based on the observation that if the multi-variable loss functions f (w) is differentiable at point w, then f (w) decreases fastest in the direction of the negative gradient of f at w, i.e., −∇f (w).

The model parameters are updated iteratively in GD as follows: DISPLAYFORM0 where w t , g t , and η t are the model parameters, gradients of f with respect to w, and learning rate at time t respectively.

For small enough η t , f (w t ) ≥ f (w t+1 ) and ultimately the sequence of w t will move down toward a local minimum.

For a convex loss function, GD is guaranteed to converge to a global minimum with an appropriate learning rate.

There are various issues to consider in gradient-based optimization.

First, GD can be extremely slow and impractical for large dataset: gradients of all the data have to be evaluated for each iteration.

With larger data size, the convergence rate, the computational cost and memory become critical, and special care is required to minimize these factors.

Second, for non-convex function which is often encountered in deep learning, GD can get stuck in a local minimum without the hope of escaping.

Third, stochastic gradient descent (SGD), which is based on the gradient of a single training sample, has large gradient variance, and it requires a large number of iterations.

This ultimately translates to slow convergence.

Mini-batch gradient descent (MGD), which is based on the gradient over a small batch of training data, trades off between the robustness of SGD and the stability of GD.

There are three advantages for using MGD over GD and SGD: 1) The batching allows both the efficiency of memory usage and implementations; 2) The model update frequency is higher than GD which allows for a more robust convergence avoiding local minimum; 3) MGD requires less iteration per epoch and provides a more stable update than SGD.

For these reasons, MGD has been a popular algorithm for machine learning.

However, selecting an appropriate batch size is difficult.

Various studies suggest that there is a close link between performance and batch size used in MGD Breuel (2015) ; Keskar et al. (2016) ; Wilson & Martinez (2003) .There are various guidelines for selecting a batch size but have not been completely practical BID1 .

Grid search is a popular method but it comes at the expense of search time.

There are a small number of adaptive MGD algorithms to replace grid search BID3 ; BID4 Friedlander & Schmidt (2012) .

These algorithms increase the batch size gradually according to their own criterion.

However, these algorithms are based on convex loss function and hard to be applied to deep learning.

For non-convex optimization, it is difficult to determine the optimal batch size for best performance.

This paper considers a resizable mini-batch gradient descent (RMGD) algorithm based on a multi-armed bandit for achieving best performance in grid search by selecting an appropriate batch size at each epoch with a probability defined as a function of its previous success/failure.

At each epoch, RMGD samples a batch size from its probability distribution, then uses the selected batch size for mini-batch gradient descent.

After obtaining the validation loss at each epoch, the probability distribution is updated to incorporate the effectiveness of the sampled batch size.

The benefit of RMGD is that it avoids the need for cumbersome grid search to achieve best performance and that it is simple enough to apply to any optimization algorithm using MGD.

The detailed algorithm of RMGD are described in Section 4, and experimental results are presented in Section 5.

There are only a few published results on the topic of batch size.

It was empirically shown that SGD converged faster than GD on a large speech recognition database Wilson & Martinez (2003) .

It was determined that the range of learning rate resulting in low test errors was considerably getting smaller as the batch size increased on convolutional neural networks and that small batch size yielded the best test error, while large batch size could not yield comparable low error rate BID2 .

It was observed that larger batch size are more liable to converge to a sharp local minimum thus leading to poor generalization Keskar et al. (2016) .

It was found that the learning rate and the batch size controlled the trade-off between the depth and width of the minima in MGD Jastrzkebski et al..

A small number of adaptive MGD algorithms have been proposed.

BID3 introduced a methodology for using varying sample size in MGD.

A relatively small batch size is chosen at the start, then the algorithm chooses a larger batch size when the optimization step does not produce improvement in the target objective function.

They assumed that using a small batch size allowed rapid progress in the early stages, while a larger batch size yielded high accuracy.

However, this assumption did not corresponded with later researches that reported the degradation of performance with large batch size Breuel (2015); Keskar et al. (2016); Mishkin et al. (2017) .

Another similar adaptive algorithm, which increases the batch size gradually as the iteration proceeded, was done by Friedlander & Schmidt (2012) .

The algorithm uses relatively few samples to approximate the gradient, and gradually increase the number of samples with a constant learning rate.

It was observed that increasing the batch size is more effective than decaying the learning rate for reducing the number of iterations Smith et al. (2017) .

However, these increasing batch size algorithms lack flexibility since it is unidirectional.

BID0 proposed a dynamic batch size adaptation algorithm.

It estimates the variance of the stochastic gradients and adapts the batch size to decrease the variance.

However, this algorithm needs to find the gradient variance and its computation depends on the number of model parameters.

Batch size can also be considered as a hyperparameter, and there have been some proposals based on bandit-based hyperparameter (but not batch size) optimization which maybe applicable for determining the best fixed batch size.

Jamieson & Talwalkar (2016) introduced a successive halving algorithm.

This algorithm uniformly allocates a budget to a set of hyperparameter configurations, evaluates the performance of all configurations, and throws out the worst half until one configuration remains.

Li et al. (2017) introduced a novel bandit-based hyperparameter optimization algorithm referred as HYPERBAND.

This algorithm considers the optimization problem as a resource allocation problem.

The two algorithms mentioned above are not adaptive, and for searching a small hyperparameter space, the two algorithms will not be very effective.

The experimental results in this paper show that adaptive MGD tends to perform better than fixed MGD.Figure 1: An overall framework of considered resizable mini-batch gradient descent algorithm (RMGD).

The RMGD samples a batch size from a probability distribution, and parameters are updated by mini-batch gradient using the selected batch size.

Then the probability distribution is updated by checking the validation loss.3 SETUP DISPLAYFORM0 be the set of possible batch size and π = {π k } K k=1 be the probability distribution of batch size where b k , π k , and K are the k th batch size, the probability of b k to be selected, and number of batch sizes respectively.

This paper considers algorithm for multi-armed bandit over B according to Algorithm 1.

Let w τ ∈ W be the model parameters at epoch τ , andw t be the temporal parameters at sub iteration t. Let J : W → R be the training loss function and let g = ∇J(w) be the gradients of training loss function with respect to the model parameters.

η τ is the learning rate at epoch τ .

Let : W → R be the validation loss function, and y k ∈ {0, 1} be the cost of choosing the batch size b k .

In here, y k = 0 if the validation loss decreases by the selected batch size b k (well-updating) and y k = 1 otherwise (misupdating).

The aim of the algorithm is to have low misupdating.

For the cost function y k , graduated losses such as hinge loss and percentage of nonnegative changes in validation loss can be variations of 0-1 loss.

However, there are no differences in regret bound among them in this setting and it is experimentally confirmed that there are little performance gaps among them.

Therefore, this paper introduces the 0-1 loss, which is simple and basic.

The resizable mini-batch gradient descent (RMGD) sets the batch sizes as multi arms, and at each epoch it samples one of the batch sizes from probability distribution.

Then, it suffers a cost of selecting this batch size.

Using the cost, probability distribution is updated.

The overall framework of the RMGD algorithm is shown in Figure 1 .

The RMGD consists of two components: batch size selector and parameter optimizer.

The selector samples a batch size from probability distribution and updates the distribution.

The optimizer is usual mini-batch gradient.

Selector samples a batch size b kτ ∈ B from the probability distribution π τ at each epoch τ where k τ is selected index.

Here b k is associated with probability π k .

The selected batch size b kτ is applied to optimizer for MGD at each epoch, and the selector gets cost y kτ from optimizer.

Then, the selector

Input: DISPLAYFORM0 : Set of batch sizes π 0 = {1/K, . . .

, 1/K} : Prior probability distribution

1: Initialize model parameters w 0 2: for epoch τ = 0, 1, 2, . . .

Select batch size b kτ ∈ B from π τ

Set temporal parametersw 0 = w τ 5: DISPLAYFORM0 Compute gradient g t = ∇J(w t )7: DISPLAYFORM1 end for DISPLAYFORM2 Update w τ +1 =w T

Observe validation loss (w τ +1 )

: DISPLAYFORM0 Get cost y kτ = 0 13: DISPLAYFORM1 Get cost y kτ = 1 15:

16: DISPLAYFORM0 Set temporal probabilityπ i = π DISPLAYFORM1 updates probabilities by randomized weighted majority, DISPLAYFORM2 Optimizer updates the model parameters w. For each epoch, temporal parametersw 0 is set to w τ , and MGD iterates T = m/b kτ 1 times using the selected batch size b kτ where m is the total number of training samples:w DISPLAYFORM3 After T iterations at epoch τ , the model parameters is updated as w τ +1 =w T .

Then, the optimizer obtains validation loss , and outputs cost as follows: DISPLAYFORM4 The RMGD samples an appropriate batch size from a probability distribution at each epoch.

This probability distribution encourages exploration of different batch size and then later exploits batch size with history of success, which means decreasing validation loss.

FIG1 shows an example of training progress of RMGD.

The figure represents the probability distribution with respect to epoch.

The white dot represents the selected batch size at each epoch.

In the early stage of training, 1 x is the least integer that is greater than or equal to x commonly, all batch sizes tend to decrease validation loss: π is uniform.

Thus, all batch size have equal probability of being sampled (exploration).

In the later stages of training, the probability distribution varies based on success and failure.

Thus, better performing batch size gets higher probability to be sampled (exploitation).

In this case, 256 is the best performing batch size.

The regret bound of the RMGD follows the regret bound derived in Shalev-Shwartz et al. (2012) .

The goal of this algorithm is to have low regret for not selecting the best performing batch size such that DISPLAYFORM0 where the expectation is over the algorithm's randomness of batch size selection and the second term on the right-hand side is the cumulative sum of the cost by the best fixed batch size which minimizes the cumulative sum of the cost.

The regret of the RMGD is bounded, DISPLAYFORM1 In particular, setting β = log(K)/(KT ), the regret is bounded by 2 K log(K)T , which is sublinear with T .

The detailed derivation of regret bound is described in the appendix A.

This section describes various experimental results on MNIST, CIFAR10, and CIFAR100 dataset.

In the experiments, simple convolutional neural networks (CNN) is used for MNIST and 'All-CNN-C' Springenberg et al. FORMULA14 is used for CIFAR10 and CIFAR100.

The details of the dataset and experimental settings are presented in the appendix B.

The validity of the RMGD was assessed by performing image classification on the MNIST dataset using AdamOptimizer and AdagradOptimizer as optimizer.

The experiments were repeated 100 times for each algorithm and each optimizer, then the results were analyzed for significance.

FIG2 shows the probability distribution and the selected batch size with respect to epoch during training for the RMGD.

The white dot represents the batch size selected at each epoch.

The top figure is the case that small batch size (32) performs better.

After epoch 50, batch size 32 gets high probability and is selected more than others.

It means that batch size 32 has less misupdating in this case.

The gradually increasing batch size algorithm may not perform well in this case.

The middle figure is the case that large batch size (512) performs better.

After epoch 60, batch size 512 gets high probability and selected more than others.

The bottom figure shows that the best performing batch size varies with epoch.

During epoch from 40 to 55, batch size of 256 performs best, and best performing batch size switches to 128 during epoch from 60 to 70, then better performing batch size backs to 256 after epoch 80.

In the results, any batch size can be a successful batch size in the later stages without any particular order.

The RMGD is more flexible for such situation than the MGD or directional adaptive MGD such as gradually increasing batch size algorithm.

FIG3 shows the test accuracy of each algorithm.

The error bar is standard error.

The number in parenthesis next to MGD represents the batch size used in the MGD. '

Basic', 'sub', 'super', 'hinge', and 'ratio' in parenthesis next to RMGD represent RMGD settings 'batch size set equal to grid search, 0-1 loss', 'subset of basic, 0-1 loss', 'superset of basic, 0-1 loss', 'basic set, hinge loss', and 'basic set, percentage of non-negative changes in validation loss', respectively.

The left figure is the test accuracy with AdamOptimizer.

The right figure is the test accuracy with AdagradOptimizer.

Among the MGD algorithms, relatively small batch sizes (16 -64) lead to higher performance than large batch sizes (128 -512) and batch size 64 achieves the best performance in grid search.

These Most RMGD settings outperform all fixed MGD algorithms in both case.

Although the performance of RMGD is not significantly increased compared to the best MGD, the purpose of this algorithm is not to improve performance, but to ensure that the best performance is achieved without performing a grid search on the batch size.

Rather, the improved performance of the RMGD is a surprising result.

Therefore, the RMGD is said to be valid.

There are little performance gap among RMGD settings.

The 'sub' setting outperforms the 'basic' setting in left figure, but the opposite result is shown in right figure.

Therefore, there is no clear tendency of performance change depending on the size of the batch size set.

TAB0 and 2 present iterations and real time for training, mean, maximum, and minimum of test accuracies for each algorithm with AdamOptimizer and AdagradOptimizer respectively.

The MGD (total) is the summation of the iterations and real time of whole MGDs for grid search.

The RMGD (basic) outperforms best performing MGD and is, also, faster than best performing MGD.

Furthermore, it is 8 times faster than grid search in both cases.

In the results, the RMGD is effective regardless of the optimizer.

The CIFAR10 and CIFAR100 dataset were, also, used to assess effectiveness of the RMGD.

The experiments were repeated 25 times and 10 times, respectively.

In these experiments, all images are whitened and contrast normalized before being input to the network.

FIG4 shows the test accuracy for each algorithm.

The left figure represents the test accuracy on CIFAR10.

In contrast to the MNIST results, relatively large batch sizes (128 -256) lead to higher performance than small batch sizes (16 -64) and batch size 256 achieves the best performance in grid search.

The right figure represents the test accuracy on CIFAR100 and batch size 128 achieves the best performance in grid search.

The results on MNIST, CIFAR10 and CIFAR100 indicate that it is difficult to know which batch size is optimal before performing a grid search.

Meanwhile, all RMGD settings have again exceeded the best performance of fixed MGD.

There are no significant performance gaps among RMGD settings, so there is no need to worry about choosing appropriate batch size set or selecting cost function.

Table 3 and 4 present the detailed results on CIFAR10 and CIFAR100 dataset.

The RMGD (basic) is a little slower than single best performing MGD (256 for CIFAR10 and 128 for CIFAR100), however, it was much faster than grid search -about 4.6 times on CIFAR10 and 5.0 times on CIFAR100 faster.

Therefore, this results, also, show the effectiveness of the RMGD.It is difficult to compare the RMGD with other adaptive batch size algorithm, e.g. coupling adaptive batch sizes (CABS) BID0 , directly since the underlying goals are different.

While the goal of the RMGD is to reduce the validation loss in terms of generalization performance, the CABS determines the batch size to balance between the gradient variance and computation.

However, it is obvious that the RMGD is simpler and easier to implement than any other adaptive algorithm cited in this paper, and comparing the test accuracy between the RMGD and the CABS on the CIFAR10 and CIFAR100 using the same experimental settings with 'All-CNN-C' shows that the performance of the RMGD is higher than that of the CABS (CIFAR10: 87.862 ± 0.142, CIFAR100: 60.782 ± 0.421).

And again, the purpose of this algorithm is not to outperform other algorithms, but to guarantee that the best performance is reached without grid search.

Selecting batch size affects the model quality and training efficiency, and determining the appropriate batch size is time consuming and requires considerable resources as it often relies on grid search.

The focus of this paper is to design a simple robust algorithm that is theoretically sound and applicable in many situations.

This paper considers a resizable mini-batch gradient descent (RMGD) algorithm based on a multiarmed bandit that achieves equivalent performance to that of best fixed batch-size.

At each epoch, the RMGD samples a batch size according to certain probability distribution of a batch being successful in reducing the loss function.

Sampling from this probability provides a mechanism for exploring different batch size and exploiting batch sizes with history of success.

After obtaining the validation loss at each epoch with the sampled batch size, the probability distribution is updated to incorporate the effectiveness of the sampled batch size.

The goal of this algorithm is not to achieve state-of-the-art accuracy but rather to select appropriate batch size which leads low misupdating and performs better.

The RMGD essentially assists the learning process to explore the possible domain of the batch size and exploit successful batch size.

The benefit of RMGD is that it avoids the need for cumbersome grid search to achieve best performance and that it is simple enough to apply to various field of machine learning including deep learning using MGD.

Experimental results show that the RMGD achieves the best grid search performance on various dataset, networks, and optimizers.

Furthermore, it, obviously, attains this performance in a shorter amount of time than the grid search.

Also, there is no need to worry about which batch size set or cost function to choose when setting RMGD.

In conclusion, the RMGD is effective and flexible mini-batch gradient descent algorithm.

In the RMGD algorithm, there are K batch sizes as multi arms with the probability distribution π ∈ S, and at each epoch the algorithm should select one of the batch sizes b kτ .

Then it receives a cost of selecting this arm, y kτ τ ∈ {0, 1} by testing the validation loss .

The vector y τ ∈ {0, 1} K represents the selecting cost for each batch size.

The goal of this algorithm is to have low regret for not selecting the best performing batch size.

DISPLAYFORM0 where the expectation is over the algorithm's randomness of batch size selection.

Let S be the probability simplex, the selecting loss functions be f τ (π) = π, y τ 2 and R : S → R be a regularization function that is often chosen to be strongly convex with respect to some norm || · ||.

The algorithm select a batch size with probability P[b kτ ] = π kτ τ and therefore f τ (π τ ) is the expected cost of the selected batch size at epoch τ .

The gradient of the selecting loss function is y τ .

However, only one element y kτ τ is known at each epoch.

To estimate gradient, random vector z τ is defined as follows: DISPLAYFORM1 and expectation of z τ satisfies, DISPLAYFORM2 The most natural learning rule is to set the probability distribution which has minimal cost on all past epochs.

It is referred to as Follow-the-Regularized-Leader (FTRL) in online learning: DISPLAYFORM3 where β is positive hyperparameter.

The FTRL has a problem that it requires solving an optimization problem at each epoch.

To solve this problem, Online Mirror Descent (OMD) is applied.

The OMD computes the current probability distribution iteratively based on a gradient update rule and the previous probability distribution and lies in the update being carried out in a 'dual' space, defined by regularizer.

This follows from considering ∇R as a mapping from R K onto itself.

The OMD relies on Bregman divergence.

The Bregman divergence between π andπ with respect to the regularizer R is given as: DISPLAYFORM4 and a Bregman projection ofπ onto simplex S: DISPLAYFORM5 Then the probability distribution is updated by the OMD as follows: DISPLAYFORM6 In general, if R is strongly convex, then ∇R becomes a bijective mapping, thusπ τ +1 can be recovered by the inverse gradient mapping (∇R) −1 .

Given that R is strongly convex, the OMD and FTRL produce equivalent predictions: DISPLAYFORM7 2 π, y is the inner product between vectors π and y DISPLAYFORM8 Therefore, the regret of the RMGD is bounded, DISPLAYFORM9 In particular, setting β = log(K)/(KT ), the regret is bounded by 2 K log(K)T , which is sublinear with T .

MNIST is a dataset of handwritten digits that is commonly used for image classification.

Each sample is a black and white image and 28 × 28 in size.

The MNIST is split into three parts: 55,000 samples for training, 5,000 samples for validation, and 10,000 samples for test.

CIFAR10 consists of 60,000 32 × 32 color images in 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck), with 6,000 images per class.

The CIFAR10 is split into three parts: 45,000 samples for training, 5,000 samples for validation, and 10,000 samples for test.

CIFAR100 consists of 60,000 32 × 32 color images in 100 classes.

The CIFAR100 is split into three parts: 45,000 samples for training, 5,000 samples for validation, and 10,000 samples for test.

The simple CNN consists of two convolution layers with 5 × 5 filter and 1 × 1 stride, two max pooling layers with 2 × 2 kernel and 2 × 2 stride, single fully-connected layer, and softmax classifier.

Description of the 'All-CNN-C' is provided in TAB3 .

For MNIST, AdamOptimizer with η = 10 −4 and AdagradOptimizer with η = 0.1 are used as optimizer.

The basic batch size set B = {16, 32, 64, 128, 256, 512}, subset of basic B − = {16, 64, 256}, and superset of basic , 24, 32, 48, 64, 96, 128, 192, 256, 384 , 512}. The model is trained for a total of 100 epochs.

For CIFAR10 and CIFAR100, MomentumOptimizer with fixed momentum of 0.9 is used as optimizer.

The learning rate η k is scaled up proportionately to the batch size (η k = 0.05 * b k /256) and decayed by a schedule S = [200, 250, 300] in which η k is multiplied by a fixed multiplier of 0.1 after 200, 250, and 300 epochs respectively.

The model is trained for a total of 350 epochs.

Dropout is applied to the input image as well as after each convolution layer with stride 2.

The dropout probabilities are 20% for dropping out inputs and 50% otherwise.

The model is regularized with weight decay λ = 0.001.

The basic batch size set B = {16, 32, 62, 128, 256}, subset of basic B − = {16, 64, 256}, and superset of basic B + = {16, 24, 32, 48, 64, 96, 128, 192, 256} .

For all experiments, rectified linear unit (ReLU) is used as activation function.

For RMGD, β is set to log(6)/(6 * 100) ≈ 0.055 for MNIST and log(5)/(5 * 350) ≈ 0.030 for CIFAR10 and CIFAR100.

The basic batch size selecting cost is 0-1 loss, hinge loss is max{0, τ − τ −1 }, and ratio loss is max{0, ( τ − τ −1 )/ τ −1 }.

1 × 1 conv.

10 or 100 ReLU, stride 1 pool averaging over 6 × 6 spatial dimensions softmax 10-way or 100-way softmax DISPLAYFORM0

<|TLDR|>

@highlight

An optimization algorithm that explores various batch sizes based on probability and automatically exploits successful batch size which minimizes validation loss.