Model-agnostic meta-learning (MAML) is known as a powerful meta-learning method.

However, MAML is notorious for being hard to train because of the existence of two learning rates.

Therefore, in this paper, we derive the conditions that inner learning rate $\alpha$ and meta-learning rate $\beta$ must satisfy for MAML to converge to minima with some simplifications.

We find that the upper bound of $\beta$ depends on $ \alpha$, in contrast to the case of using the normal gradient descent method.

Moreover, we show that the threshold of $\beta$ increases as $\alpha$ approaches its own upper bound.

This result is verified by experiments on various few-shot tasks and architectures; specifically, we perform sinusoid regression and classification of Omniglot and MiniImagenet datasets with a multilayer perceptron and a convolutional neural network.

Based on this outcome, we present a guideline for determining the learning rates: first, search for the largest possible $\alpha$; next, tune $\beta$ based on the chosen value of $\alpha$.

A pillar of human intelligence is the ability to learn and adapt to unseen tasks quickly and based on only a limited quantity of data.

Although machine learning has achieved remarkable results, many recent models require massive quantities of data and are designed for solving particular tasks.

Meta-learning, one of the ways of tackling this problem, tries to develop a model that can adapt to new tasks quickly by learning to learn new concepts from few data points (Schmidhuber, 1987; Thrun & Pratt, 1998) .

Among meta-learning algorithms, model-agnostic meta-learning (MAML), a gradient-based metalearning method proposed by Finn et al. (2017) , has recently been extensively studied.

For example, MAML is used for continual learning (Finn et al., 2019; Jerfel et al., 2019; Spigler, 2019; Al-Shedivat et al., 2018) , reinforcement learning (Finn et al., 2017; Al-Shedivat et al., 2018; Gupta et al., 2018; Deleu & Bengio, 2018; Liu & Theodorou, 2019) and probablistic inference Yoon et al., 2018; Grant et al., 2018) .

The reason why MAML is widely used is because MAML is simple but efficient and applicable to a wide range of tasks independent of model architecture and the loss function.

However, MAML is notorious for being hard to train (Antoniou et al., 2019) .

One of the reasons why training MAML is hard is the existence of two learning rates in MAML: the inner learning rate ?? and meta-learning rate ??.

A learning rate is known to be one of the most important parameters, and tuning this parameter may be challenging even if the simple gradient descent (GD) method is used.

Nevertheless, we do not yet know the relationship between these two learning rates and have little guidance on how to tune them.

Hence, guidelines for choosing these parameters are urgently needed.

In this paper, we investigate the MAML algorithm and propose a guideline for selecting the learning rates.

First, in Section 2 we briefly explain by using an approximation how MAML can be regarded as optimization with the negative gradient penalty.

Because the gradient norm is related to the shape of the loss surface, a bias towards a larger gradient norm can make training unstable.

Next, based on the approximation explained in Section 2, in Section 3, we derive a sufficinent condition of ?? and ?? for a simplified MAML to locally converge to local minima from any point in the neighborhood of the local minima.

Furthermore, by removing a constraint, we derive a sufficient condition for local convergence with fewer simplifications as well.

We find that the upper bound ?? c of meta-learning rate depends on inner learning rate ??.

In particular, ?? c of ?? ??? ?? c is larger than that of ?? = 0, where ?? c is the upper bound of ??.

This is verified by experiments in Section 5.

These results imply a guideline for selecting the learning rates: first, search for the largest possible ??; next, tune ??.

The goal of MAML is to find a representation that can rapidly adapt to new tasks with a small quantity of data.

In other words, MAML performs optimization for parameters ?? ??? R d that the optimizer can use to quickly reach the optimal parameter ?? * ?? for task ?? with few data points.

To this end, MAML takes the following steps to update ??.

First, it samples a batch of tasks from task distribution P (?? ) and updates ?? for each task ?? with stochastic gradient descent (SGD).

Although MAML allows multiple-step being taken to update ??, we will consider the case only one step being taken for simplicity.

The update equation is as follows:

where ?? is a step size referred to as the inner learning rate, L ?? (??) is the loss of ?? , and

where ?? is the learning rate called the meta-learning rate and

is an estimate of the true gradient by using the test data.

Though learning rates ?? and ?? can be tuned during training or different for each task in practice, we will think them as fixed scalar hyperparameters.

Unless otherwise noted, we will consider the case of only one step being made per update, and the data are not resampled to compute the loss for updating ??.

The case of multiple steps and that of training data and test data being separated are considered in Appendix A. The gradient of the loss at ??

is the gradient of L(??) with respect to ??.

If ?? is small, we can assume that I ???L?? ????? ??? ?? = g ?? (??); this seems to hold since ?? is usually small (Finn et al., 2017) .

Then,

The result but not the procedure with the approximation is the same as that with the well-known firstorder approximation as long as data are not resampled and only one step being taken.

The first order approximation has been mentioned by Finn et al. (2017) and extensively studied by Nichol et al. (2018) and Fallah et al. (2019) .

It is known that the error induced by the first-order approximation does not degrade the performance so much in practice (Finn et al., 2017) .

Also, Fallah et al. (2019) theoretically proved that the first-order approximation of MAML does not affect convergence result when each task is similar to each other or ?? is small enough.

For simplicity, we will assume that only one task is considered during training, omitting task index ?? .

Therefore, instead of

, we will consider L(?? ??? ) as the loss of the simplified MAML.

Since the MAML loss is just a sum of task-specific loss, extension to the case of multiple tasks being considered is straightforward, as provided in Appendix A.1.

The above means that the simplified MAML can be regarded as optimization with the negative gradient penalty.

We will analyze this simplified MAML loss in Section 3.

It can also be interpreted as a Taylor series expansion of the simplified MAML loss for the first-order term, up to scale:

The fact that the simplified MAML is optimization with the negative gradient penalty is worth keeping in mind.

Because the goal of gradient-based optimization is to find a point where the gradient is zero, a bias that favors a larger gradient is highly likely to make training unstable; this can be a cause of instability of MAML (Antoniou et al., 2019) .

In fact, as shown in Fig. 1 , the gradient norm becomes larger during training, as do the gradient inner products, as Guiroy et al. (2019) .

Gradient norm during training.

We compute the norm per task and subsequently compute their average.

Joint training shows when ?? = 0, and MAML is when ?? = 1e-2.

These results are computed using training data, but those determined using test data behave similarly.

The total number of iterations is 50000, ?? = 1e-3 and the Adam optimizer is used.

Other settings are the same as those in Section 5.2.

In this section, we will derive a sufficient condition of learning rate ?? and ?? for local convergence from any point in the vicinity of the local minima.

To this end, we will assume that only one step is taken for update and training data and test data are not distinguished as we did in Section 2.

Also, we do not consider SGD but steepest GD to derive the condition.

In other words, same training data are assumed to be used continuously for updating parameters during training.

First, we will consider the case of only single task being considered.

Next, we will consider the case of multiple tasks being considered.

3.1 SINGLE TASK

First, we derive the sufficient condition of learning rate ??.

To this end, we will consider the sufficinet condition that a fixed point is a local minimum.

Taking the Taylor series for the second-order term at a fixed point ?? * , the simplified MAML loss is

The calculation ofH is presented in Appendix B. We calculated the magnitudes of Tg and H 2 numerically and observed that Tg was much smaller than H 2 in practice.

Hence, we will ignore Tg while deriving the condition and will thus assume thatH = H ??? ??H 2 .

Further details are provided in Appendix C, and the case of Tg being considered is provided in Appendix D. Since

is a diagonal matrix with entries that are eigenvalues ofH and P is a matrix with rows that are eigenvectors ofH, the sufficinet condition of ?? for ?? * to be a local minimum is

Note that ??(A) i represents the ith eigenvalue of matrix A. Hence, the sufficient condition of ?? for ?? * to be a local minimum is

Therefore, ?? c is the inverse of the largest eigenvalue of H.

Next, we derive the sufficient condition of meta-learning rate ?? for the simplified MAML to locally converge to the local minima discussed above from any point in the vicinity of the local minimum.

This is an extension of research of LeCun et al. (1998) .

Since P P ??? = I, the simplified MAML loss can be written as

By using the simplified loss defined in Eq. 8, the update equation of the parameter ?? with GD is

where t (t = 0, ..., M ) is iteration and M is the total number of iterations.

Hence, ??(t

where v(t) is the value of v during iteration t. Assuming that Eq. 11 holds, the sufficinet condition of ?? is as follows: for all i,

Consequently, the sufficient condition for the simplified MAML to locally converge to local minima from any point in the vicinity of the local minima is as follows:

Vanilla GD with learning rate ?? corresponds to MAML if ?? = 0.

In this case, ?? < 2 ??max is the condition of ??, where ?? max is the largest eigenvalue of H, because 2/?? max is smaller than any other 2/?? i (LeCun et al., 1998) .

Though this holds for the simplified MAML as well, this is not the case if ?? is close to ?? c .

The reason is that ?? c diverges as ?? approaches 1 ??(H)i , or ?? c as Eq. 18 indicates.

Hence, for the simplified MAML we must consider not only the largest but also other eigenvalues and in particular, the second-largest eigenvalue.

In short, unlike when vanilla GD is employed, ?? c depends on ?? in the case of the simplified MAML, and ?? c is expected to be larger if ?? is close to ?? c , as shown in Fig. 2 .

This finding is validated by experiments presented in Section 5.

We discussed the case of only one task being available during training in Section 3.1.

In this section, we derive upper bounds of ?? and ?? that apply if multiple tasks ?? ??? P (?? ) are considered.

Assumptions, except that multiple tasks are considered, are the same as those in Section 3.1.

Now, we define the simplified meta-objective as a sum of task-specific objectives:

where?? is the Hessian matrix ofL(??) at ??.

Note that we ignore T ?? g ?? as we did in Section 3.1.1.

?? is not necessarily simultaneously diagonalizable for each task ?? , we cannot exactly express eigenvalues of?? as function of ??(H ?? ) and ??.

Therefore, instead of the exact value of ?? c , we will derive an upper bound of ?? c .

If ?? * is a local minimum for allL ?? (?? * ), ?? * is a local minimum forL(??) as well.

Hence, if all eigenvalues ofH ?? = H ?? ??? 2??H

holds, it guarantees that the condition that ?? * is a local minimum ofL(??) is satisfied.

Note that this is a sufficient condition if ?? * is a local minimum for allL ?? (?? * ).

Since ?? * can be a local minimum ofL(??) even when it is not a local minimum for allL ?? (?? * ), the upper bound of ?? seems to be larger than that in Eq. 23 in practice.

The analysis for ?? is also basically the same as that performed in Section 3.1.2.

Denoting P

SinceH ?? is not always simultaneously diagonalizable for each task, as mentioned in Section 3.2.1, P ?? differs from task to task in general.

Hence, we will consider an upper bound of ?? c as we did for ?? c in Section 3.2.1.

Accordingly, if both Eq. 23 and

hold for any eigenvalue ?? i of any task ?? , it guarantees that ?? satisfies the condition for local convergence.

Therefore, a sufficient for the simplified MAML to locally converge to local minima from any point in the neighborhood of the local minima in the case of multiple tasks is as follows:

4 RELATED WORKS Sevral papers have investigated MAML and proposed various algorithms (Nichol et al., 2018; Guiroy et al., 2019; Eshratifar et al., 2018; Antoniou et al., 2019; Fallah et al., 2019; Khodak et al., 2019; Vuorio et al., 2018; Finn et al., 2019; Deleu & Bengio, 2018; Liu & Theodorou, 2019; Deleu & Bengio, 2018; Grant et al., 2018) .

Nichol et al. (2018) studied the first-order MAML family in detail and showed that the MAML gradient could be decomposed into two terms: a term related to joint training and a term responsible for increasing the inner product between gradients for different tasks.

Guiroy et al. (2019) investigated the generalization ability of MAML.

The researchers observed that generalization was correlated with the average gradient inner product and that flatness of the loss surface, often thought to be an indicator of strong generalizability in normal neural network training, was not necessarily related to generalizability in the case of MAML.

Eshratifar et al. (2018) also noted that the average gradient inner product was important.

Hence, the authors proposed an algorithm that considered the relative importance of each parameter based on the magnitude of the inner product between the task-specific gradient and the average gradient.

Although the above studies were cognizant of the importance of the inner product of the gradients, they did not explicitly insert the negative gradient inner product, which is the negative squared gradient norm with simplifications, as a regularization term.

To consider the simplified MAML as optimization with a regularization term is a contribution of our study.

Antoniou et al. (2019) enumerated five factors that could cause training MAML to be difficult.

Then, they authors proposed an algorithm to address all of these problems and make training MAML easier and more stable.

Behl et al. (2019) , like us, pointed out that tuning the inner learning rate ?? and meta-learning rate ?? was troublesome.

The authors approached this problem by proposing an algorithm that tuned learning rates automatically during training.

Fallah et al. (2019) studied the convergence theory of MAML.

They proposed a method for selecting meta-learning rate by approximating smoothness of the loss.

Based on this result, they proved that MAML can find an ??-first-order stationary point after sufficient number of iterations.

On the other hand, we studied the relationship between the sufficient conditions of inner learning rate ?? and meta-learning rate ?? and showed that how the largest possible ?? is affected by the value of ??.

In this section, we will present the results of experiments to confirm our expectation that MAML allows larger ?? if ?? is close to its upper bound.

First, we will show the result of linear regression with simplifications used in Section 3.1.

Because linear regression is convex optimization, the result is expected to exactly match the theory presented in Section 3.1.

Second, to check if our expectation is confirmed in practice as well, we will present results of the practical case without any simplification.

In particular, we conducted sinusoid regression and classification of Omniglot and MiniImagenet datasets with a multilayer perceptron and a convolutional neural network (CNN).

Note that the meta-objective used for experiments is not a sum of task-specific objectives but a mean of them. .

The true function has the same architecture as that of the model.

We employed the steepest gradient descent method to minimize the mean squared loss, where 1 step was taken during update.

Only one task was considered during training and the same data was used to update the task-specific parameter and the meta parameter as we did in Section 3.1.

Using these settings, we computed the training loss after 500 iterations with ?? in the range of [1e-5, 9e-2] and ?? in the range of [5e-3, 9e+0].

The eigenvalues are those of the Hessian matrix of the training loss at the end of the training, where ?? = 5e-2 and ?? = 7e-1.

We chose this training loss because it was thought to be the closest to minima.

Fig. 3 shows the training losses at various values of ?? and ??.

Horizontal axis indicates ?? and vertical axis indicates ??.

The curves are ?? c of two eigenvalues and the dashed line shows ?? c .

In the case of linear regression with simplifications used in Section 3.1, the result of numerical experiment shows good agreement with upper bounds of ?? c and ?? c that we derived in Section 3.1, as shown in Fig. 3 .

We conducted a sinusoid regression, where each task is to regress a sine wave with amplitude in the range of [0.1, 5.0] and phase in the range of [0, ??] based on data points in the range of [-5.0, 5.0] .

A multilayer perceptron with two hidden units of size 40 and ReLU was trained with SGD.

The batch size of data was 10, the number of tasks was 100, and 1 step was taken for update.

Using these settings, we computed the training loss after 500 iterations with ?? in the range of [1e-4, 9e-1] and ?? in the range of [1e-2, 9e-0].

Fig. 4 (a) shows the training losses with various values of ?? and ??.

The dashed line indicates ?? of ?? = 0 over which training loss diverges.

According to Fig. 4 (a) , if ?? is close to the value above which the losses diverge, a larger ?? can be used.

As explained above, we did not put any simplification as we did in Section 3 and 5.1 in the experiment, meaning that we used different data for updating task-specific parameter and meta parameter, considered multiple tasks, and employed not steepest GD but SGD as optimizer.

Despite simplifications, surprisingly, this result confirms the expectation that MAML allows larger ?? if ?? is close to ?? c .

We performed classification of the Omniglot and MiniImagenet datasets (Lake et al., 2011; ravi & Larochelle, 2017) , which are benchmark datasets for few-shot learning.

The model used was essentially the same as that Finn et al. (2017) , and hence, Vinyals et al. (2016) used.

The task is a five-way one-shot classification, where the query size is 15, the number of update steps is two, and the task batch size is 32 for Omniglot and four for MiniImagenet.

In this setup, we computed the training losses after 100 iterations for the Omniglot dataset and one epoch for the MiniImagenet dataset with various values of ?? and ??; for Omniglot, ?? was in the range of [1e-3, 9e-0] and ?? was in the range of [1e-1, 9e+1], and for MiniImagenet, ?? was in the range of [1e-4, 9e-1] and ?? was in the range of [1e-2, 9e-0].

Fig. 4 (b) and (c) show the training losses of classification task at various values of ?? and ??.

The dashed line indicates ?? of ?? = 0 above which training loss diverges.

As shown in Fig. 4 (b) and (c), the maximum ?? is larger at large ??.

Even though the model architecture is composed of convolutional layer, max-pooling, and batch normalization (Ioffe & Szegedy, 2015) and practical dataset is used for training, our expectation is confirmed in this case as well.

This result confirms that our theory is applicable in practice.

Our experimental result confirms that larger ?? is good for stabilizing MAML training.

According to Fig. 4 (a) , (b) and (c), moreover, while large ?? does not necessarily make training loss smaller, employing large ?? leads to smaller training loss comparatively.

These result has a practical implication for tuning the learning rates: first, the largest possible ?? should be identified, and ?? may be subsequently tuned based on the value of ??.

Once you identify ?? c , MAML is likely to work well even if meta-learning rate is roughly chosen.

Taking large ?? is desirable for the goal of MAML as well.

The aim of MAML is to find a good initial parameter which quickly adapt to new tasks and the quickness is determined by inner learning rate ?? because ?? determines the step size from initial parameter to task-specific parameter when model is fine-tuned.

Therefore, identifying ?? c is not only good for robustifying the model against divergence but also good for finding good initial parameter.

We regard a simplified MAML as training with the negative gradient penalty.

Based on this formulation, we derived the sufficient condition of the inner learning rate ?? and the meta-learning rate ?? for the simplified MAML to locally converge to local minima from any point in the vicinity of the local minima.

We showed that the upper bound of ?? required for the simplified MAML to locally converge to local minima depends on ??.

Moreover, we found that if ?? is close to its upper bound ?? c , the maximum possible meta-learning rate ?? c is larger than that used while training with ordinary SGD.

This finding is validated by experiments, confirming that our theory is applicable in practice.

According to this result, we propose a guideline for determining ?? and ??; first, search for ?? close to ?? c ; next, tune ?? based on the selected value of ??.

In Section 2, we explained the simplest case of training data and test data being the same, only one task being considered and only one step being taken for update.

In this section, we analyze the cases of waiving one of these simplifications.

Here, we will interpret the simplified MAML loss as a Taylor series expansion of the MAML loss for the first-order term.

A.1 WHEN THE TRAINING DATA AND TEST DATA ARE DIFFERENT When using different data to compute the losses for updating the task-specific parameters and the meta parameters as done in practical applications, the simplified loss is as follows:

A.2 WHEN k STEPS ARE TAKEN DURING THE UPDATE If task-specific parameters are updated with k-step SGD, the loss can be written as follows;

Note that L i (??) is the loss computed with the data at the ith step, and g i (??) is the gradient of the loss.

BecauseH is the Hessian matrix ofL at ?? * , we derive the Hessian of Eq. 8.

Then,

C MAGNITUDE OF T g AND H

We conducted a sinusoid regression with essentially the same condition that we explain in Section 5.2 except that the total number of iterations is 50000 and learnig rates are fixed.

Parameters ?? and ?? are 1e-2 and 1e-3 respectively.

We calculated Tg numerically with the training error at the end of the training.

As we showed in Section 3, especially large eigenvalues ofH are important for the upper bounds of learning rates.

Therefore, if ??(Tg + H 2 ) max ??? ??(H 2 ) max , we can ignore Tg when deriving the condition.

We calculate the maximum and the second-largest eigenvalues of Tg, H 2 and Tg + H 2 of the trained model.

As shown in Fig. 5 (a

We assumed that Tg was negligible in Section 3.

In this section, we derive a sufficient condition of ?? and ?? for the simplified MAML to locally converge to local minima from any point in the vicinity of the local minima under some assumptions when Tg is considered.

When Tg is considered, the Hessian matrix of the simplified MAML lossL isH = H ??? ??

is a real symmetric matrix, it can be diagonalized.

Then, the sufficient condition that a fixed point ?? * is a local minimum is

Note that Tg and H are not simultaneously diagonalizable, so we cannot decompose ??(H) i into eigenvalues of each matrix as we did in Section 3.

Therefore, we have to consider the relationship among ??(H) i , ??(H) i and ??(Tg) i .

In general, it is known that n ?? n Hermitian matrices A and B satisfy the following equation (Bhatia, 2001) :

where ??(A) represents a vector with elements that are eigenvalues of A, ??? indicates the operation of sorting a vector in the ascending order, and ??? indicates that in the descending order.

If two real vectors x, y ??? R d are related in the following way, x is said to be majorized by y and the relationship is written as x ??? y:

1e-4 3e-4 5e-4 7e-4 9e-4 2e-3 4e-3 6e-3 8e-3 1e-2 3e-2 5e-2 7e-2 9e-2 2e-1 4e-1 6e-1 8e-1 9e-0 7e-0 5e-0 3e-0 1e-0 8e-1 6e-1 4e-1 2e-1 9e-2 7e-2 5e-2 3e-2 1e-2 0.8

@highlight

We analyzed the role of two learning rates in model-agnostic meta-learning in convergence.

@highlight

The authors tackled the optimization instability problem in MAML by investigating the two learning rates.

@highlight

This paper studies a method to help tune the two learning rates used in the MAML training algorithm.