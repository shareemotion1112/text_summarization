Stochastic gradient descent (SGD), which  trades off noisy gradient updates for computational efficiency, is the de-facto optimization algorithm to solve large-scale machine learning problems.

SGD can make rapid learning progress by performing updates using subsampled training data, but the noisy updates also lead to slow asymptotic convergence.

Several variance reduction algorithms, such as SVRG, introduce control variates to obtain a lower variance gradient estimate and faster convergence.

Despite their appealing asymptotic guarantees, SVRG-like algorithms have not been widely adopted in deep learning.

The traditional asymptotic analysis in stochastic optimization provides limited insight into training deep learning models under a fixed number of epochs.

In this paper, we present a non-asymptotic analysis of SVRG under a noisy least squares regression problem.

Our primary focus is to compare the exact loss of SVRG to that of SGD at each iteration t.

We show that the learning dynamics of our regression model closely matches with that of neural networks on MNIST and CIFAR-10 for both the underparameterized and the overparameterized models.

Our analysis and experimental results suggest there is a trade-off between the computational cost and the convergence speed in underparametrized neural networks.

SVRG outperforms SGD after a few epochs in this regime.

However, SGD is shown to always outperform SVRG in the overparameterized regime.

Many large-scale machine learning problems, especially in deep learning, are formulated as minimizing the sum of loss functions on millions of training examples (Krizhevsky et al., 2012; Devlin et al., 2018) .

Computing exact gradient over the entire training set is intractable for these problems.

Instead of using full batch gradients, the variants of stochastic gradient descent (SGD) (Robbins & Monro, 1951; Zhang, 2004; Bottou, 2010; Sutskever et al., 2013; Duchi et al., 2011; Kingma & Ba, 2014) evaluate noisy gradient estimates from small mini-batches of randomly sampled training points at each iteration.

The mini-batch size is often independent of the training set size, which allows SGD to immediately adapt the model parameters before going through the entire training set.

Despite its simplicity, SGD works very well, even in the non-convex non-smooth deep learning problems (He et al., 2016; Vaswani et al., 2017) .

However, the optimization performance of the stochastic algorithm near local optima is significantly limited by the mini-batch sampling noise, controlled by the learning rate and the mini-batch size.

The sampling variance and the slow convergence of SGD have been studied extensively in the past (Chen et al., 2016; Li et al., 2017; Toulis & Airoldi, 2017) .

To ensure convergence, machine learning practitioners have to either increase the mini-batch size or decrease the learning rate toward the end of the training (Smith et al., 2017; Ge et al., 2019) .

The minimum loss achieved in real dataset MNIST (a logistic regression model).

Our theoretical prediction (a) matched with the training dynamics for real datasets, demonstrating tradeoffs between computational cost and convergence speed.

The curves in red are SVRG and curves in blue are SGD.

Different markers refer to different per-iteration computational cost, i.e., the number of backpropagation used per iteration on average.

their strong theoretical guarantees, SVRG-like algorithms have seen limited success in training deep learning models (Defazio & Bottou, 2018) .

Traditional results from stochastic optimization focus on the asymptotic analysis, but in practice, most of deep neural networks are only trained for hundreds of epochs due to the high computational cost.

To address the gap between the asymptotic benefit of SVRG and the practical computational budget of training deep learning models, we provide a non-asymptotic study on the SVRG algorithms under a noisy least squares regression model.

Although optimizing least squares regression is a basic problem, it has been shown to characterize the learning dynamics of many realistic deep learning models (Zhang et al., 2019; Lee et al., 2019) .

Recent works suggest that neural network learning behaves very differently in the underparameterized regime vs the overparameterized regime Vaswani et al., 2019) , characterized by whether the learnt model can achieve zero expected loss.

We account for both training regimes in the analysis by assuming a linear target function and noisy labels.

In the presence of label noise, the loss is lower bounded by the label variance.

In the absence of the noise, the linear predictor can fit each training example perfectly.

We summarize the main contributions as follows:

??? We show the exact expected loss of SVRG and SGD along an optimization trajectory as a function of iterations and computational cost.

??? Our non-asymptotic analysis provides an insightful comparison of SGD and SVRG by considering their computational cost and learning rate schedule.

We discuss the trade-offs between the total computational cost, i.e. the total number of back-propagations performed, and convergence performance.

??? We consider two different training regimes with and without label noise.

Under noisy labels, the analysis suggests SGD only outperforms SVRG under a mild total computational cost.

However, SGD always exhibits a faster convergence compared to SVRG when there is no label noise.

??? Numerical experiments validate our theoretical predictions on both MNIST and CIFAR-10 using various neural network architectures.

In particular, we found the comparison of the convergence speed of SGD to that of SVRG in underparameterized neural networks closely matches with our noisy least squares model prediction.

Whereas, the effect of overparameterization is captured by the regression model without label noise.

Stochastic variance reduction methods consider minimizing a finite-sum of a collection of functions using SGD.

In case we use SGD to minimize these objective functions, the stochasticity comes from the randomness in sampling a function in each optimization step.

Due to the induced noise, SGD can only converge using decaying step sizes with sub-linear convergence rate.

Methods such as SAG (Roux et al., 2012) , SVRG (Johnson & Zhang, 2013) , and SAGA (Defazio et al., 2014) , are able to recover linear convergence rate of full-batch gradient descent with the asymptotic cost comparable to SGD.

SAG and SAGA achieve this improvement at the substantial cost of storing the most recent gradient of each individual function.

In contrast, SVRG spends extra computation at snapshot intervals by evaluating the full-batch gradient.

Theoretical results such as show that under certain smoothness conditions, we can use larger step sizes with stochastic variance reduction methods than is allowed for SGD and hence achieve even faster convergence.

In situations where we know the smoothness constant of functions, there are results on the optimal mini-batch size and the optimal step size given the inner loop size (Sebbouh et al., 2019) .

Applying variance reduction methods in deep learning has been studied recently (Defazio & Bottou, 2018) .

The authors conjectured the ineffectiveness is caused by various elements commonly used in deep learning such as data augmentation, batch normalization and dropout.

Such elements can potentially decrease the smoothness and make the stored gradients become stale quickly.

The proposed solution is to either remove these elements or update the gradients more frequently than is practical.

Dynamics of SGD and quadratic models Our main analysis tool is very closely related to recent work studying the dynamics of gradient-based stochastic methods.

Wu et al. (2018) derived the dynamics of stochastic gradient descent with momentum on a noisy quadratic model (Schaul et al., 2013) , showing the problem of short horizon bias.

In (Zhang et al., 2019) , the authors showed the same noisy quadratic model captures many of the essential characteristic of realistic neural networks training.

Their noisy quadratic model successfully predicts the effectiveness of momentum, preconditioning and learning rate choices in training ResNets and Transformers.

However, these previous quadratic models assume a constant variance in the gradient that is independent of the current parameters and the loss function.

It makes them inadequate for analyzing the stochastic variance reduction methods, as SVRG can trivially achieve zero variance under the constant gradient noise.

Instead, we adopted a noisy least-squares regression formulation by considering both the mini-batch sampling noise and the label noise.

There are also recent works that derived the risk of SGD, for least-squares regression models using the bias-variance decomposition of the risk Hastie et al., 2019) .

We use a similar decomposition in our analysis.

In contrast to the asymptotic analysis in these works, we compare SGD to SVRG along the optimization trajectory for any finite-time horizon under limited computation cost, not just the convergence points of those algorithms.

Underparameterization vs overparameterization.

Many of the state-of-the-art deep learning models are overparameterized deep neural networks with more parameters than the number of training examples.

Even though these models are able to overfit to the data, when trained using SGD, they generalize well (Zhang et al., 2017) .

As suggested in recent work, underparameterized and overparameterized regimes have different behaviours Vaswani et al., 2019; Schmidt & Roux, 2013) .

Given the infinite width and a proper weight initialization, the learning dynamics of a neural network can be well-approximated by a linear model via the neural tangent kernel (NTK) (Jacot et al., 2018; Chizat & Bach, 2018) .

In NTK regime, neural networks are known to achieve global convergence by memorizing every training example.

On the other hand, previous convergence results for SVRG have been obtained in stochastic convex optimization problems that are similar to that of an underparameterized model (Roux et al., 2012; Johnson & Zhang, 2013) .

Our proposed noisy least-squares regression analysis captures both the underparameterization and overparameterization behavior by considering the presence or the absence of the label noise.

We will primarily focus on comparing the minibatch version of two methods, SGD and SVRG (Johnson & Zhang, 2013) .

Denote L i as the loss on i th data point.

The SGD update is written as,

is the minibatch gradient, t is the training iteration, and ?? (t) is the learning rate.

The SVRG algorithm is an inner-outer loop algorithm proposed to reduce the variance of the gradient caused by the minibatch sampling.

In the outer loop, for every T steps, we evaluate a large batch gradient??? = 1 N N i ??? ?? (mT ) L i , where N b, and m is the outer loop index, and we store the parameters ?? (mT ) .

In the inner loop, the update rule of the parameters is given by,

where??

Note that in our analysis, the reference point is chosen to be the last iterate of previous outer loop ?? (mT ) , recommended as a practical implementation of the algorithm by the original SVRG paper Johnson & Zhang (2013) .

We now define the noisy least squares regression model (Schaul et al., 2013; Wu et al., 2018) .

In this setting, the input data is d-dimensional, and the output label is generated by a linear teacher model with additive noise,

y .

We assume WLOG ?? * = 0.

We also assume the data covariance matrix ?? is diagonal.

This is an assumption adopted in many previous analysis and it is also a practical assumption as we often apply whitening to pre-process the training data.

We would like to train a student model ?? that minimizes the squared loss over the data distribution:

At each iteration, the optimizer can query an arbitrary number of data points {x i , y i } i sampled from data distribution.

The SGD method uses b data points to form a minibatch gradient:

where

SVRG on the other hand, queries for N data points every T steps to form a large batch gradient

where X N and N are defined similarly.

At each inner loop step, it further queries for another b data points, to form the update in Eq. 2.

Lastly, note that the expected loss can be written as a function of the second moment of the iterate,

Hence for the following analysis we mainly focus on deriving the dynamics of the second mo-

When ?? is diagonal, the loss can further be reduced to

Definition 1 (Formula for dynamics).

We define the following functions and identities,

The SGD update (Eq. 1) with the mini-batch gradient of of the noisy least squares model (Eq. 4) is,

We substitute the update rule to derive the following dynamics for the second moment of the iterate:

This dynamics equation can be understood intuitively as follows.

The term 1 leads to an exponential shrinkage of the loss due to the gradient descent update.

Since we are using a noisy gradient, the second term 2 represents the variance of stochastic gradient caused by the random input X b .

The term 3 comes from the label noise.

We show in the next theorem that when the second moment of the iterate approaches zero, 2 will also approach zero.

However due to the presence of the label noise, the expected loss is lower bounded by 3 .

When ?? is diagonal, we further analyze and decompose C(M(??)) as a function of m(??) so as to derive the following dynamics and decay rate for SGD.

Theorem 2 (SGD Dynamics and Decay Rate).

Given the noisy linear regression objective function (Eq. 3), under the assumption that x ??? N (0, ??) with ?? diagonal and ?? * = 0, we can express C(??) as a function of m(??):

Then we derive following dynamics of expected second moment of ??:

Under the update rule of SGD, R is the decay rate of the second moment of parameters between two iterations.

And based on Theorem 2 the expected loss can be calculated by

3 A DILEMMA FOR SVRG By querying a large batch of datapoints X N every T steps, and a small minibatch X b at every step, the SVRG method forms the following update rule:

To derive the dynamics of the second moment of the parameters following the SVRG update, we look at the dynamics of one round of inner loop updates, i.e., from ?? (mT ) to ?? ((m+1)T ) : Lemma 3.

The dynamics of the second moment of the iterate following SVRG update rule is given by,

The dynamics equation above is very illuminating as it explicitly manifests the weakness of SVRG.

First notice that terms 1 , 2 , 3 reappear, contributed by the SGD update.

The additional terms, 4 and 5 , are due to the control variate.

Observe that the variance reduction term 5 decays exponentially throughout the inner loop, with decay rate I ??? ????, i.e. P .

We immediately notice that this is the same term that governs the decay rate of the term 1 , and hence resulting in a conflict between the two.

Specifically, if we want to reduce the term 1 as fast as possible, we would prefer a small decay rate and a large learning rate, i.e. ?? ??? 1 ??max(??) .

But this will also make the boosts provided by the control variate diminish rapidly, leading to a poor variance reduction.

The term 4 makes things even worse as it will maintain as a constant throughout the inner loop, contributing to an extra variance on top of the variance from standard SGD.

On the other hand, if one chooses a small learning rate for the variance reduction to take effect, this inevitably will make the decay rate for term 1 smaller, resulting in a slower convergence.

Nevertheless, a good news for SVRG is that the label noise (term 3 ) is scaled by b N , which lets SVRG converge to a lower loss value than SGD -a strict advantage of SVRG compared to SGD.

To summarize, the variance reduction from SVRG comes at a price of slower gradient descent shrinkage.

In contrast, SVRG is able to converge to a lower loss value.

This motivates the question, which algorithm to use given a certain computational cost?

We hence performed a thorough investigation through numerical simulation as well as experiments on real datasets in Sec. 4.

Similarly done for SGD, we decompose C(??) as a function of m(??) and derive the following decay rate for SVRG.

Theorem 4 (SVRG Dynamics and Decay rate).

Given the noisy linear regression objective function (Eq. 3), under the assumption that x ??? N (0, ??) with ?? diagonal and ?? * = 0, the dynamics for SVRG in m(??) is given by:

In the absence of the label noise (i.e., ?? y = 0), we observe that both SGD and SVRG enjoy linear convergence as a corollary of Theorem 2 and Theorem 4: Corollary 5.

Without the label noise, the dynamics of the second moment following SGD is given by,

and the dynamics of SVRG is given by,

where ?? is defined in Eq.( 11).

Note that similar results have been shown in the past Vaswani et al., 2019; Schmidt & Roux, 2013) , where a general condition known as "interpolation regime" is used to show linear convergence of SGD.

Specifically they assume that ???L i (?? * ) = 0 for all i, and our setting without label noise clearly also belongs to this regime.

This setting also has practical implications, as one can treat training overparameterized neural networks as in interpolation regime.

This motivates the investigation of the convergence rate of SGD and SVRG without label noise, and was also extensively studied in the experiments detailed as follows.

In Sec. 3 we discussed a critical dilemma for SVRG that is facing a choice between effective variance reduction and faster gradient descent shrinkage.

At the same time, it enjoys a strict advantage over SGD as it converges to a lower loss.

We define the total computational cost as the total number of back-propagations performed.

Similarly, per-iteration computational cost refers to the number of back-propagations performed per iteration.

In this section, we study the question, which algorithm converges faster given certain total computational cost?

We study this question for both the underparameterized and the overparameterized regimes.

Our investigation consists of two parts.

First, numerical simulations of the theoretical convergence rates (Sec. 4.1).

Second, experiments on real datasets (Sec. 4.2).

In both parts, we first fix the per-iteration computational cost.

For SGD, the per-iteration computational budge is equal to the minibatch size.

We picked three batch size {64, 128, 256}. Denote the batchsize of SGD as b, the equivalent batch size for SVRG is b = we also ran over a set of snapshot intervals T .

After running over all sets of hyperparameters, we gather all training curves of all hyperparameters.

We then summarize the performance for each algorithm by plotting the lower bound of all training curves, i.e. each point (l, t) on the curve showed the minimum loss l at time step t over all hyperparameters.

We compared the two methods under different computational cost.

Remarkably, we found in many cases phenomenon predicted by our theory matches with observations in practice.

Our experiments suggested there is a trade-off between the computational cost and the convergence speed for underparameterized neural networks.

SVRG outperformed SGD after a few epochs in this regime.

Interestingly, in the case of overparameterized model, a setting that matches modern day neural networks training, SGD strictly dominated SVRG by showing a faster convergence throughout the entire training.

We first performed numerical simulations of the dynamics derived in Theorem 2 for SGD and Theorem 4 for SVRG.

We picked a data distribution, with data dimension d = 100, and the spectrum of ?? is given by an exponential decay schedule from 1 to 0.01.

For both methods, we picked 50 learning rate from 1.5 to 0.01 using a exponential decay schedule.

For SVRG, we further picked a set of snapshot intervals for each learning rate: {256, 128, 64}. We performed simulations in both underparameterized and overparameterized setting (namely with and without label noise), and plotted the lower bound curves over all hyperparameters at Figure 2 .

The x-axis represents the normalized total computational cost, denoting tbN ???1 , which is equivalent to the notion of an epoch in finite dataset setting.

And the loss in Figure 2 does not contain bayes error (i.e. We have the following observations from our simulations.

In the case with label noise, the plot demonstrated an explicit trade-off between computational cost and convergence speed.

We observed a crossing point of between SGD and SVRG appear, indicating SGD achieved a faster convergence speed in the first phase of the training, but converged to a higher loss, for all per-iteration compute cost.

Hence it shows that one can trade more compute cost for convergence speed by choosing SGD than SVRG, and vice versa.

Interestingly, we found that the the per-iteration computational cost does not seem to affect the time crossing point takes place.

For all these three costs, the crossing points in the plot are at around the same time: 5.5 epochs.

In the case of no label noise, we observed both methods achieved linear convergence, while SGD achieved a much faster rate than SVRG, showing absolute dominance in this regime.

In this section, we performed a similar investigation as in the last section, on two standard machine learning benchmark datasets: MNIST (LeCun et al., 1998) and CIFAR-10 (Krizhevsky, 2009 ).

We present the results from underparameterized setting first, followed by the overparameterized setting.

We performed experiments with three batchsizes for SGD: {64, 128, 256}, and an equivalent batchsize for SVRG.

For each batch size, we pick 8 learning rates varying from 0.3 to 0.001 following an exponential schedule.

Additionally, we chose three snapshot intervals for every computational budget, searching over the best snapshot interval given the data.

Hence for each per-iteration computational cost {64, 128, 256}, there are 24 groups of experiments for SVRG and 8 groups of experiments for SGD.

for training on MNIST and CIFAR-10 with overparameterized models.

In this setting we observed strict dominance of SGD over SVRG in convergence speed for all computational cost, matching our previous theoretical prediction.

For MNIST, we trained two underparameterized model: 1.

logistic regression 784 ??? 10 2.

a underparameterized two layer MLP 784 ??? 10 ??? 10 where the hidden layer has 10 neurons.

For CIFAR-10, we chose a underparameterized convolutional neural network model, which has only two 8-channel convolutional layers, and one 16-channel convolutional layer with one additional fully-connected layer.

Filter size is 5.

The lowest loss achieved over all hyperparameters for these models for each per-iteration computational cost are shown in Figure 3 .

From these experiments, we observe that on MNIST, the results with underparameterized model were consistent with the dynamics simulation of noisy least squares regression model with label noise.

First of all, SGD converged faster in the early phase, resulting in a crossing point between SGD and SVRG.

It showed a trade-offs between computational cost and convergence speed: before the crossing point, SGD converged faster than SVRG; after crossing point, SVRG attained a lower loss.

In addition, in Fig 3a, all the crossing points of three costs matched at the same epoch (around 5), which was also consistent with the our findings with noisy least squares regression model.

On CIFAR-10, SGD achieved slightly faster convergence in the early phase, but was surpassed by SVRG around 17 ??? 25 epochs, again showing a trade-off between compute and speed.

Lastly, we compared SGD and SVRG on MNIST and CIFAR-10 using overparameterized models.

For MNIST, we used a MLP with two hidden layers, each layer having 1024 neurons.

For CIFAR-10, we chose a large convolutional network, which has one 64-channel convolutional layer, one 128-channel convolutional layer followed by one 3200 to 1000 fully connected layer and one 1000 to 10 fully connected layer.

The lowest loss achieved over all hyperparameters for these models for each per-iteration computational cost are shown in Figure 4 .

For training on MNIST, both SGD and SVRG attained close to zero training loss.

The results were again consistent with our dynamics analysis on the noisy linear regression model without label noise.

SGD has a strict advantage over SVRG, and achieved a much faster convergence rate than SVRG throughout the entire training.

As for CIFAR-10, we stopped the training before either of the two got close to zero training loss due to lack of computing time.

But we clearly see a trend of approaching to zero loss.

Similarly, we also had the same observations as before, where SGD outperforms SVRG, confirms the limitation of variance reduction in the overparameterized regime.

In this paper, we studied the convergence properties of SGD and SVRG in the underparameterized and overparameterized settings.

We provided a non-asymptotic analysis of both algorithms.

We then investigated the question about which algorithm to use under certain total computational cost.

We performed numerical simulations of dynamics equations for both methods, as well as extensive experiments on the standard machine learning datasets, MNIST and CIFAR-10.

Remarkably, we found in many cases phenomenon predicted by our theory matched with observations in practice.

Our experiments suggested there is a trade-off between the computational cost and the convergence speed for underparameterized neural networks.

SVRG outperformed SGD after the first few epochs in this regime.

In the case of overparameterized model, a setting that matches with modern day neural networks training, SGD strictly dominated SVRG by showing a faster convergence for all computational cost.

Lemma 6 (Gradient Covariance).

Given the noisy linear regression objective function (Eq. 3), under the assumption that x ??? N (0, ??) with ?? diagonal and ?? * = 0, we have

Proof.

In the following proof, we define the entry-wise p power on vector x as x ???p .

Under our assumption ?? = 0, ??

Eq. 12 is a conclusion from The Matrix Cookbook (See section 8.2.3 in Petersen & Pedersen (2012) ).

Then, for its main diagonal term, we have:

Hence, for C M(?? (t) ) , we have:

which is the first conclusion of Theorem 2.

Notice, this conclusion can be generalized to any square matrix A not only for E[?? (t) ?? (t) ], i.e. for any square matrix A ??? R d??d , with x ??? N (0, ??) and ?? diagonal, since

we have

For batch gradient

where [N ] b is the index set of X b .

Theorem 2.

Given the noisy linear regression objective function (Eq. 3), under the assumption that x ??? N (0, ??) with ?? diagonal and ?? * = 0, we can express C(M(??)) as a function of m(??):

Then we derive following dynamics of expected second moment of ??:

Proof.

Since, E[ b ] = 0, and b is independent with ?? (t) , X b , we have:

and,

Since X b is independent with ?? (t) , we have:

Thus,

For its diagonal term, we have:

This formula can be written as :

where

D THE PROOF OF LEMMA 3

Lemma 3.

The dynamics of the second moment of the iterate following SVRG update rule is given by,

4 variance due tog

5 Variance reduction from control variate

Proof.

For SVRG update rule Eq. 8, we have:

Using the update rule of SVRG, we can get the outer product of parameters as:

Likewise, since E[ N ] = 0 and N is independent with X b , X N and ?? (t) , we have the expectation of equation 46, equation 47 equal to 0.

And same as SGD, we also have

Then, we give a significant formula about the expectation of ?? (mT +t) ?? (mT ) , utilized to derive the expected term related to variance reduction amount.

and N is independent with X N and ?? (mT ) , the expectation of Eq. 53 is equal to 0.

Therefore,

which suggests the covariance between?? (mT +t) andg (mT +t) is exponentially decayed.

For every other term appearing in Eq. 41, we have the following conclusions.

First, similar with SGD, we have the formula about gradient descent shrinkage as:

Using Eq. 54, we have following conclusion for variance reduction term from control variate.

We first take expectation over ?? (mT +t) ?? (mT ) with Eq. 54 due to the independence among X b , X N and ??.

For the forth term, which represents the variance ofg (mT +t) , we consider the independence between X b and X N and get

Thus,

Under our definition, it can be expressed as:

4 variance due tog

5 Variance reduction from control variate E THE PROOF OF THEOREM 4

Theorem 4.

Given the noisy linear regression objective function (Eq. 3), under the assumption that x ??? N (0, ??) with ?? diagonal and ?? * = 0, the dynamics for SVRG in m(??) is given by:

Proof.

Form lemma 3 and lemma 6, we can get:

where

Recursively expending the above formula from m(?? ((m+1)T ) ) to m(?? (mT ) ), we can get the following result:

= R Rm(?? (mT +T ???2) ) ??? QP T ???2 m(?? (mT ) ) + F m(?? (mT ) ) + N ???1 n (80)

In other word, Eq. 79 describe the dynamic of expected second moment of iterate between two nearby snapshots, m(?? ((m+1)T ) ) = ??(??, b, T, N, ??)m(?? (mT ) ) + (I ??? R T )(I ??? R)

In our theoretical analysis (Section 3), we evaluate a large batch gradient??? to control variance.

That is because any number of data points can be directly sampled form the true distribution.

But in order to compare the computational cost between SVRG and SGD, we set the number of data points used to calculate??? as N , which is slightly different with the original SVRG's setup of full-batch gradient.

Therefore, we evaluate the sensitivity of N to illustrate when N is beyond a threshold, it will cause little difference in convergence speed for SVRG.

From figure 5a, we can tell N has little effect on the convergence speed of SVRG under the noisy least square model, but it determines the constant term of label noise in Eq. 9 which determines the level of final loss.

Besides, we also compare large batch SGD to SVRG in Figure 5b

@highlight

Non-asymptotic analysis of SGD and SVRG, showing the strength of each algorithm in convergence speed and computational cost, in both under-parametrized and over-parametrized settings.