We propose an efficient online hyperparameter optimization method which uses a joint dynamical system to evaluate the gradient with respect to the hyperparameters.

While similar methods are usually limited to hyperparameters with a smooth impact on the model, we show how to apply it to the probability of dropout in neural networks.

Finally, we show its effectiveness on two distinct tasks.

With the growing size and complexity of both datasets and models, training times keep increasing and it is not uncommon to train a model for several days or weeks.

This effect is compounded by the number of hyperparameters a practitioner has to search through.

Even though search through hyperparameter space has improved beyond grid search, this task is still often computationally intensive, mainly because these techniques are offline, in that they need to perform a full learning before trying a new value.

Recently, several authors proposed online hyperparameter optimization techniques, where the hyperparameters are tuned alongside the parameters of the model themselves by running short runs of training and updating the hyperparameter after each such run.

By casting the joint learning of parameters and hyperparameters as a dynamical system, we show that these approaches are unstable and need to be stopped using an external process, like early stopping, to achieve a good performance.

We then modify these techniques such that the joint optimization procedure is stable as well as efficient by changing the hyperparameter at every time step.

Further, while existing techniques are limited in the type of hyperparameters they can optimize, we extend the process to dropout probability optimization, a popular regularization technique in deep learning.

Historically, optimizing hyperparameters was done by selecting a few values for each hyperparameter, computing the Cartesian product of all these values, then by running a full training for each set of values.

BID0 showed that performing a random search rather than a grid search was vastly more efficient, in particular by avoiding spending too much time training models with one of the hyperparameters set to a poor value.

This technique was later refined in BID1 by using quasi random search.

However, the parameters and their ranges have to be selected in advance, and potentially many trainings have to be performed to find good parameters.

To remedy this issue, BID9 used Gaussian processes to model the validation error as a function of the hyperparameters.

Each training further refines this function to minimize the number of sets of hyperparameters to try.

All these methods are "black-box" methods, in that they assume no knowledge about the internal training process to optimize hyperparameters.

In particular, they are gradient-free since they do not have access to the gradient of the validation loss with respect to the hyperparameters.

To solve this issue, BID5 and BID7 explicitly used the parameter learning process to obtain such a gradient.

These techniques, however, still need to complete a full optimization between each update of the hyperparameters.

This can be an issue for very long trainings as poor choices of hyperparameters are not discarded right away, leading to unnecessary computations.

The work most closely related to ours is that of BID2 where the hyperparameters are changed after a certain number of parameter updates.

However, not only must that number of updates be chosen manually, the proposed algorithm is not stable and moves away from the optimum after some time.

While this issue can be solved using other techniques, e.g., early stopping, we propose a stable algorithm by casting the joint learning of parameters and hyperparameters as a dynamical system.

We also show how convergence can be obtained by changing the hyperparameters after each parameter update, thus further simplifying the algorithm.

We then propose modifications to improve the speed and robustness of the optimization.

Finally, we demonstrate the performance of our method on several problems.

The goal of learning is to find parameters which minimize the true expected risk.

As we do not have access to that risk, we rely instead on the minimization of the empirical risk obtained using a training set.

However, it is well-known that this can lead to overfitting, which can be prevented by regularization.

A common method to choose which and how much regularization to use is to hold out part of the training set and to find which regularization yields the best performance on that held-out, or validation, set.

We denote by parameters the parameters of the function being learnt and by hyperparameters the parameters of the regularization being used.

Hyperparameter optimization looks for the hyperparameters λ such that the minimization of the regularized training loss over model parameters θ leads to the best performance on the validation set.

Using this nomenclature, the best hyperparameters are selected according to: DISPLAYFORM0 where L V is the unregularized validation loss and L T the regularized training loss.

It is important to note that this work focuses exclusively on regularization hyperparameters.

In particular, we do not attempt to optimize optimization parameters such as the learning rate.

Eq.

(1) shows that, to determine an optimization strategy for λ, one may compute the gradient of L V (θ * (λ)) with respect to λ, which we call the hypergradient, and perform gradient descent.

By the chain rule, we have DISPLAYFORM0 where, to simplify notations, we denoted by g V the gradient of L V with respect to θ, i.e. g V := ∂L V ∂θ .

Similarly, we denote g T := ∂L T ∂θ the gradient of the regularized training loss with respect to θ.

Since, by definition of the optimum, we have g T (θ * (λ), λ) = 0, we can use the implicit function theorem to get: DISPLAYFORM1 Several algorithms propose to compute the hypergradient exactly BID5 compute this derivative by backpropagating through the whole training procedure.

Unfortunately this is very costly both in memory footprint and in wall time as several training procedures need to be serialized, hence is not easily scalable to large models.

BID7 computes an approximate derivative when the model parameters are close to the optimal ones.

In both cases, one needs to perform a full, or almost full, optimization to compute the gradient, leading to expensive updates.

We shall now see how we can compute approximate updates using far fewer optimization steps.

The core idea is that the convergence of iterates θ t to θ * should allow us to use these iterates to update λ rather than wait until convergence.

In doing so, we could optimization the hyperparam-eters simultaneously with the optimization of the model parameters.

This idea has been explored by BID2 who proposed to optimize the validation error obtained when running exactly K steps of gradient descent with fixed hyperparameters, i.e. DISPLAYFORM0 subject to: DISPLAYFORM1 where the constraint corresponds to the updates of θ using gradient descent with a learning rate η and where θ t (t > 0) implicitly depends on λ.

The K-iterate hypergradient is then given by: DISPLAYFORM2 Computing ∂θ K ∂λ can be done recursively by differentiating the gradient update recurrence in Eq. (4) with respect to λ: DISPLAYFORM3 Defining y t = ∂θt ∂λ , we have two dynamical systems: DISPLAYFORM4 DISPLAYFORM5 starting from θ 0 = 0, y 0 = 0.

It is important to emphasize that, although we care about the convergence of the second system to compute the hypergradient, its trajectory is completely determined by that of θ t and thus by the first system.

In other words, the value of y t does not affect the optimization process over θ.

Even though the system defined in Eq. (7) converges to the right solution, it can do so very slowly.

For instance, assume that we are at hyperparameter λ 0 and that θ 0 is initialized to the optimal value, i.e. θ 0 = θ * (λ 0 ).

In that case, the system defined in Eq. (6) is already at convergence and θ t = θ * (λ 0 ) for all t. The second system, however, will take some time to converge to the final value ∂θ * (λ,θ0) ∂λ.

We now study the behavior of y t = ∂θt ∂λ (λ 0 ) whose recurrence is: DISPLAYFORM6 The fixed point of this recurrence is y * = −A −1 B which is equal to the true hypergradient ∂θ * ∂λ (λ 0 ) according to Eq. (2).

The convergence rate of y t depends on the spectrum of I − ηA. If η is too small, convergence will be slow and using a fixed number of steps K can lead to a poor estimation of ∂θ * ∂λ .

This poor estimation is mitigated by the fact that, if η is small and y 0 = 0, y K will be in O(Kη) and thus the steps taken in hyperparameter space will also be small.

We thus believe the overall effect of a small η on the hyperparameter optimization will be limited to a smaller convergence.

We described how to perform one hyperparameter update using gradient descent with the hypergradient DISPLAYFORM7 .

BID2 repeat this process in an outer loop, each time setting θ 0 to the previous θ K and initializing y 0 to 0.

This reinitialization of y at the beginning of each inner loop prevents the optimization from capturing any long term dependency of λ on θ and y t from converging to the true hypergradient.

We now propose another formulation which maintains a growing history of the dependency of λ on θ, yielding increased stability.

Using the method of BID2 with K = 1 updates the hyperparameters at every gradient step, as proposed by BID4 .

They compute y(λ t ) = −η ∂g T ∂λ (θ t , λ t ), so that the hypergradient is estimated by DISPLAYFORM0 Under this formulation, minimizing the validation loss over λ is equivalent to maximizing < g V (θ t ), g T (θ t , λ t ) > using a specific scaling η for the learning rate.

Assuming we are at a θ lying on the manifold {θ * (λ) : λ}, then the hypergradient defined by Eq. (8) is proportional to DISPLAYFORM1 which is in general not equal to the true hypergradient DISPLAYFORM2 The two hypergradients are only proportional when the Hessian DISPLAYFORM3 ∂θ is a multiple of the identity.

This suggests these first order methods can fail to converge to a local optimum.

However, their simplicity makes them good candidates for the early stages of the optimization.

Instead of reinitializing y 0 (λ t ) to 0 after every hyperparameter update, another possibility is to initialize y 0 (λ t ) to the last value y K (λ t−1 ) obtained using the previous value of λ.

While this is beneficial when λ t is close to λ t−1 , issues might arise earlier in the optimization.

Indeed, stopping the system before convergence could yield a value of y t much larger than y ∞ , overstimating the norm of the gradient and leading to a large change in λ.

Although reinitializing y 0 to 0 every time is crude, it favors smaller values of y t and thus smaller changes in λ, increasing stability.

To keep this stability while maintaining as much information about y as possible, we propose to modify recurrence y t by constraining y t to lie within a ball: DISPLAYFORM0 where P B(r) is the projection on the ball of radius r and is formally defined by P B(r) (x) = r max( x , r)x.

Every time the norm is clipped, this is equivalent to changing the stepsize for λ but not the direction of the gradient.

However, due to the dynamical nature of the system, it also affects future updates.

As the learning rate decreases, so does the probability of clipping since: DISPLAYFORM1 r is a hyperparameter which was chosen in the experiments so that clipping occurs almost at every step at the beginning of optimization, behaving like the method of BID4 .

Our proposed method is summarized in Algorithm 1.

DISPLAYFORM2 for t < num steps do 6: DISPLAYFORM3 θ t+1 ← θ t − ηg T 8: DISPLAYFORM4 if t ≥ warmup time then

λ t+1 ← λ t − cηg P y t with: c constant scaling 11: DISPLAYFORM0 λ t+1 ← λ t

We now describe in more details how we optimized two different regularizers: 2 penalty, which has been optimized previously using hyperparameter optimization techniques, and dropout probability, for which, to the best of our knowledge, no existing techniques can be applied.

The training loss, in case of 2 regularization, is given by DISPLAYFORM0 In neural networks, we can differentiate two types of linear layers depending on whether L(θ) is sensitive to the norm of the weights or not.

For example, the unregularized loss does not depend on the norm when a linear layer is followed by a normalization layer like batch norm: any change of the norm is compensated by the normalization layer.

In those cases, 2 regularization does not prevent overfitting as the norm can be decreased arbitrarily close to 0 without changing the function represented by the neural network.

However, it has an impact on the dynamic of the training.

van BID11 showed that for such a layer represented by weights θ, the effective learning rate is η eff = η θ 2 .

Since the gradient is orthogonal to the vector of weights BID8 , the norm of the weights after a gradient update DISPLAYFORM1 and keeps increasing if there is no 2 regularization (i.e. λ = 0).

The norm remains stable after a gradient update only when DISPLAYFORM2

2 Assuming a learning rate small enough such that ηλ << 1, we have DISPLAYFORM0

2 .

In terms of effective learning rate, the norm remains stable when η eff = 2λ ∂L ∂θ 2 , i.e. when the effective learning rate does not depend on the initial learning rate.

This short analysis shows that 2 regularization can have a significant impact on the optimization without having a proper regularization effect.

In this paper, we do not intend to address the problem of optimizing hyperparameters that have only an impact on the dynamic of the training and focus on the original intent of 2 regularization as a way to prevent overfitting.

Introduced in BID3 and further studied in BID10 , dropout is a way to regularize by preventing co-adaptation of output units of a neural network.

Regularization is achieved by considering an ensemble of network architectures which differ only by their connections between the output units and the input units of the next layer, the weights being shared.

Each output unit can be either kept with probability p or dropped, meaning there is no connection to the next layer.

The keep/drop decision can be represented by a vector mask m which indicates for each output unit whether this one is kept or dropped.

The probability to keep an output unit is often considered as an hyperparameter of the model, denoted as λ in this section.

The training loss can be computed as an expectation over dropout masks m: DISPLAYFORM0 where B(p = λ) denotes the Bernoulli distribution.

To compute the dependencies between the state θ and the hyperparameter λ (see Eq. FORMULA7 and FORMULA8 ), we need to have access to ∂L T ∂λ .

This cannot be formally computed, but can be approximated with finite differences: DISPLAYFORM1 In order to minimize the complexity, we just sample one dropout mask for p = λ + and one for p = λ − and compute the approximate derivative of the loss.

The variance of the derivative computed this way can be quite large though.

Instead, we use the fact that: DISPLAYFORM2 to sample a mask for p 1 = λ − which is not independent from the mask sampled using p 2 = λ + .

We compare several methods to train the hyperparameters in an online way:(a) Unroll1-gTgV, a first order method directly maximizing g T g V (Section 3.3, no η), (b) UnrollK, the version from BID2 (Section 3.2) which optimizes the hyperparameters over a fixed training window of size K, (c) ClipR, described in Algorithm 1 with a clipping threshold equal to R.Note that Unroll1 is a special case of UnrollK which differs from Unroll1-gTgV by the factor η (which can have an influence when η is depends on t).We evaluate these methods on models of increasing complexity, starting with a toy problem and ending on a typical deep learning setup.

As a baseline, we use the typical one-shot hyperparameter optimization where the model is learnt N times, once for each value of the hyperparameters, keeping the hyperparameters achieving the best validation loss.

While conducting the evaluation, we should be aware of possible overfitting on the validation set: information leaked from that set should be the same as with the typical one shot optimization algorithm.

In particular, tested methods contain hyper-hyperparameters, like the clipping threshold in method ClipR. Since our original goal is to simplify hyperparameter optimization, we also test how sensitive to the particular values of these hyper-hyperparameters the final result is.

Finally, we evaluate the intrisic stability of the various online algorithms.

To do so, we shall compare the performance of each online hyperparameter optimization method with and without using early stopping.

A large gap between these two values indicates the best hyperparameter value is not a stable point for this particular method.

Again, we use as baseline the same gap computed when the training is done with fixed hyperparameters.

While we mostly report final results, more detailed reports of all these experiments are available in the appendix.

We consider a strongly convex optimization problem where the training loss and the validation loss are given by DISPLAYFORM0 θ,θ are of dimension 20 and H is a diagonal matrix.

The optimal λ can be computed analytically and is equal to DISPLAYFORM1 For the first 20K steps, the learning rate η is set to 10 −3 and the parameters at the end of this first phase are denoted by (θ 1 , λ 1 ).

η is then set to 10 −4 for the following 20K steps and the parameters Table 2 : Performance on MNIST with a 4-hidden layer network (lower is better).

Clipping leads to the best results, with or without early stopping.

In that case, small unrolls are the most stable.

DISPLAYFORM2 at the end of this phase are (θ 2 , λ 2 ).

The learning rate for the hyperparameters is always set to: 0.1η.

Constantsθ,θ and H are sampled so that λ † lie in [0.2, 0.4].We repeat each training 50 times using differentθ,θ, H, and we evaluate the performance of each method using two metrics: (a) the average distance between λ i (i ∈ 1, 2) and λ † , (b) the increase in validation loss compared to optimal value θ * (λ † ).

Table 4 .1 shows that, in line with the theoretical observations, methods UnrollK methods are unable to estimate the hypergradient when the learning rate decreases, leading to a slight increase of the loss.

Additionally, while ∂θ * ∂λ is between 3.0 and 5.0, using 2.0 as a clipping threshold leads to only a slight increase of the validation loss while Clip5 converges to the correct value.

We consider here a feedforward network with 4 fully connected layers, of size 100 for the 3 hidden layers and of size 10 for the last layer.

It is trained on MNIST using 2 regularization.

We also use a decaying schedule for the learning rate as it has been shown to possibly impact the online hyperparameter optimization algorithms.

The dataset is split between a training set (75% of the samples) and a validation set (25% of the samples).

Every online algorithm is run 6 times with a different initialization of the weight decay, equally spaced in the log domain between 10 −6 and 10 −1 .

The results are averaged over those 6 runs.

The metric we optimize for is the cross-entropy on the validation dataset.

Table 4 .2 shows that clipping outperforms the other methods, especially when early stopping is not used, displaying higher stability.

We also note that Clip20 performs better than a fixed λ found through grid search, showing the efficiency of the method on more realistic problems.

Cross Table 3 : Performances on PTB with an LSTM (lower is better).

Finally, we consider a language modeling task using the PTB dataset BID6 ) and a typical LSTM-based network architecture 1 .

In this architecture, the use of dropout is critical in order to prevent overfitting on the training dataset.

Training is done using a decaying learning rate with a multiplicative decay of 0.95 every 5K mini-batches.

The hyperparameter learning rate is chosen as a constant scaling of the parameter learning rate.

As described previously, the performance metric (the perplexity on the validation dataset in this case) in Table 3 are given with and without early stopping so as to derive a measure of intrisic instability of the online algorithms.

Combined with early stopping, all the hyperparameter optimization methods achieve good performance with slightly worse results for Unroll1-gTgV and Unroll1.However, the algorithms are not equivalent in terms of stability.

Unrolling methods are the most unstable when close to convergence as they drift to higher probabilities of keeping the weights (under-regularization).

As explained in Section 3.3, method Unroll1 attenuates this effect compared to Unroll1-gTgV by using a lower effective learning rate when this drift occurs.

Contraty to what we observed on MNIST, longer rollouts seem here to increase the stability.

On the other side, none of the flavors of the gradient clipping algorithm is subject to this instability: the instability metric is low and of the same order as the one derived using a fixed keep probability of λ = 0.35.Last, online optimization of the hyperparameter does not increase overfitting on the validation dataset compared to a typical one shot algorithm: the minimum of the validation loss for those algorithms being close to the one obtained with one shot hyperparameter optimization.

A CONVERGENCE OF THE DERIVATIVE FIG1 shows that, even if we start at the optimal value θ * (λ) for the parameters, the dynamical system {y t } will take some time to converge to the true solution.

We now show how the value of the hyperparameter and the training and validation loss vary during optimization.

In particular, this will help determine when there are instabilities, i.e. when the best validation loss is not obtained at the end of the optimization.

@highlight

An algorithm for optimizing regularization hyper-parameters during training

@highlight

The paper proposes a way to re-initialize y at each update of lambda and a clipping procedure of y to maintain the stability of the dynamical system.

@highlight

Proposes an algorithm for hyperparameter optimization that can be seen as an extension of Franceschi 2017 were some estimates are warm restarted to increase the stability of the method.

@highlight

Proposes an extension to an existing method to optimize regularization hyperparameters.