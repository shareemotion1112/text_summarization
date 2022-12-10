There is significant recent evidence in supervised learning that, in the over-parametrized setting, wider networks achieve better test error.

In other words, the bias-variance tradeoff is not directly observable when increasing network width arbitrarily.

We investigate whether a corresponding phenomenon is present in reinforcement learning.

We experiment on four OpenAI Gym environments, increasing the width of the value and policy networks beyond their prescribed values.

Our empirical results lend support to this hypothesis.

However, tuning the hyperparameters of each network width separately remains as important future work in environments/algorithms where the optimal hyperparameters vary noticably across widths, confounding the results when the same hyperparameters are used for all widths.

A longstanding notion in supervised learning is that, as model complexity increases, test error decreases initially and, eventually, increases again.

Intuitively, the idea is that as the size of your hypothesis class grows, the closer you can approximate the ground-truth function with some function in your hypothesis class.

At the same time, the larger amount of functions to choose from in your hypothesis class leads to higher estimation error (overfitting) from fitting the finite data sample too closely.

This is the essential bias-variance tradeoff in supervised learning.

We discuss these tradeoffs in more depth in Section 2.2.However, BID20 found that increasing the width of a single hidden layer neural network leads to decreasing test error on MNIST and CIFAR-10.

Since then, there has been a large amount of evidence that wider networks generalize better in a variety of different architectures and hyperparameter settings BID27 BID21 BID15 BID19 BID0 BID24 BID17 , once in the over-parametrized setting BID24 BID0 .

In other words, the biasvariance tradeoff is not observed in this over-parametrized setting, as network width grows BID19 .How far can we inductively infer from this?

Is this phenomenon also present in deep reinforcement learning or do we eventually see a degradation in performance as we increase network width?

In this paper, we present preliminary evidence that this phenomenon is also present in reinforcement learning.

For example, using default hyperparameters, we can already see performance increase well past the default width that is commonly used (64) in FIG0 .

We test the hypothesis that wider networks (both policy and value) perform monotonically better than their smaller counterparts in policy gradients methods.

Of course, we will hit diminishing returns as the network width gets very large, but this is very different from the competing hypothesis that larger networks will overfit more.

We are given a training set S = {(x 1 , y 1 ), (x 2 , y 2 ), . . .

, (x m , y m )} of m training examples, where x i ∈ X and y i ∈ Y. Furthermore, Z = X × Y, so S ∈ Z m .

D denotes a distribution over Z, so we have (x i , y i ) ∼ D and S ∼ D m .

We use lowercase x and y to denote random variables because of convention in this field.

We learn a hypothesis h ∈ H via a learning algorithm A :

Z m → H. We denote a hypothesis learned from training set S as h S = A(S).

Given a loss function, : Y ×Y → R, the goal is to minimize the expected risk: DISPLAYFORM0

We present a discussion on tradeoffs in model complexity because it does not appear to be much of a focus in the reinforcement learning community.

A common way of thinking about the generalization performance of a learner is through the lens of a tradeoff.

For example, when h S is chosen from a hypothesis class H, R(h S ) can be decomposed into approximation error and estimation error DISPLAYFORM0 where E app = min h∈H R(h) and E est = R(h S ) − E app .

Shalev-Shwartz & Ben-David (2014, Section 5.2) present this decomposition and frame it as a tradeoff.

BID2 describe this as the "well known tradeoff between approximation error and estimation error" and present it in a slightly more lucid way as a decomposition of the excess risk: DISPLAYFORM1 where R(h * ) is the Bayes error and h * H = arg min h∈H R(h) is the best hypothesis in H. The approximation error can then be interpreted as the distance of the best hypothesis in H from the Bayes classifier, and the estimation error can be interpreted as the average distance of the learned hypothesis from the best hypothesis in H. It is common to associate larger H with smaller approximation error and larger estimation error.

The commonly cited universal approximation property of neural networks BID4 BID13 BID16 means that the approximation error goes to 0 as the network width increases; these results do not say anything about estimation error.

A similar tradeoff in model complexity is known as the bias-variance tradeoff BID7 .

Bias is analogous to the approximation error while variance is analogous to the estimation error.

This tradeoff is probably even more pervasive (Bishop (2006, Chapter 3 .2), BID7 , Hastie et al. (2001, Chapter 2.9) , Goodfellow et al. (2016, Chapter 5.4.4) ).

It is common to view the problem of designing a good learning algorithm as choosing a good H that optimizes this tradeoff.

Statistical learning theory for supervised learning is given in the i.i.d.

setting.

That is, examples are independent and identically distributed.

This also means the training distribution is the same as the test distribution.

In reinforcement learning, training examples are not independent because examples within the same episode depend on each other through the current behavior policy and through the environment's transition dynamics.

Training examples are not identically distributed because the policy produces training examples, and the policy changes over time.

For the same reason, the training distribution and the test distribution are not completely the same.

These differences make it nonobvious that the phenomenon seen in supervised learning would extend to reinforcement learning.

We run experiments, with a variety of combinations of environments and learning algorithms, where we vary the width of the shared policy and value network.

We use four different environments from OpenAI Gym BID3 : CartPole, Acrobot, MountainCar, and Pendulum.

We use four different learning algorithms: PPO , A2C BID18 , ACER BID25 , and ACKTR .

We make use of the existing implementations of these algorithms in the Stable Baselines library BID12 , an improved fork of OpenAI Baselines .

We were only able to train ACKTR up to width 512 because it is an approximate second-order method.

Experiments with ACKTR are in Appendix B.We get hyperparameters that were tuned on networks of width 64 from the RL Baselines Zoo that was built on top of Stable Baselines.

One hyperparameter is how many time steps the learners are trained for.

It is different for different environment/learner pairs, but always on the order of 1 million.

It is always the same across widths within an environment/learner pair.

In some of the plots, learners see fewer episodes because their episodes are, on average, longer.

We choose relatively simple tasks for these experiments partially because they are faster to train on, but more importantly, we choose them because their simplicity lends itself to more ease of overfitting.

In other words, on these tasks, we will see diminishing returns with much smaller networks, so we can test the "very wide networks will not see degraded performance" hypothesis with a much smaller range of networks.

We run each experiment with 5 different random seeds.

The policy and value networks are shared.

The architecture consists of 2 hidden fully connected layers followed by separate linear transformations: one to yield the policy and one to yield the value.

We use 2 hidden layers, rather than just 1, because 2 hidden layers are more common in reinforcement learning.

In CartPole (Fig. 2) , we see a lot of evidence for the hypothesis.

In the both the PPO and A2C experiments, peak performance is reached by width 64, and that level of performance is maintained through width 2048.

In the ACER experiment, near peak performance is reached by width 128, and through width 2048, we see peak performance.

Similarly, in Acrobot (Fig. 3) , we see even more evidence for the hypothesis.

We see peak performance as early as width 16 in PPO, ACER, and A2C.

This means that Acrobot is simple enough to only require a network of width 16 (compared to 64 for CartPole).

Still, we see peak performance through width 2048 in all 3 learners.

In Pendulum (Appendix A), we see more evidence for the hypothesis.

The default width (64) network, performs distinctly worse than the wider networks.

We do not see any degradation of performance through width 2048.

We only run PPO with the Pendulum environment because RL Baselines Zoo did not have tuned hyperparameters for the other algorithms.

In the MountainCar environment, we see the first hint of what looks like could be evidence against the hypothesis (Fig. 4) .

PPO (left) performance begins to degrade at width 2048, ACER (center) performance begins to degrade at width 512, and we see a sharp drop in performance from width 1024 to width 2048 in A2C (right).RL algorithms are known to be highly sensitive to hyperparameter settings BID10 BID14 , especially learning rate BID11 .

We believe this performance degradation is due to more variability across widths of the optimal hyperparameters on MountainCar (compared CartPole, Acrobot, and Pendulum).

In order to fairly compare all the widths, we would like the hyperparameters for each of them to be optimal.

BID6 ) study test error when scaling network width in supervised learning, and they scale the learning rate as h −1.5 , where h is the network width.

This scaling is motivated by making the number of steps to convergence independent of width, but it does not necessarily make the learning rate for each network optimal.

Because learning rate is such an important and sensitive hyperparameter in reinforcement learning BID11 , we try scaling the learning rate α with both of the following schemes: α ← min(α * 64 , (h/64) −1 ) and α ← min(α * 64 , (h/64) −1.5 ), where α * 64 is the learning rate that was tuned to network width 64 (pulled from RL Baselines Zoo).

We see that scaling the learning rate as h −1 (Fig. 5 ) and h −1.5 (Appendix C, Fig. 8 ) actually make the largest networks perform worse, indicating that this scaling is not useful for comparing networks with optimal hyperparameters.

We present these scalings on MountainCar because it was the environment that did not look like the others, but the scalings on CartPole and Acrobot are in Appendix D.

The phenomenon in supervised learning that motivated this work is that, in the over-parametrized setting, increasing network width leads to monotonically lower test error (no U curve).

We find a fair amount of evidence of this phenomenon extending to reinforcement learning in our preliminary experiments (namely CartPole, Acrobot, and Pendulum).However, we also saw that performance did consistently degrade in the MountainCar experiments.

We believe this to be because that environment is more sensitive to hyperparameters; since the hyperparameters were chosen using width 64 and then used for all of the other widths, the hyperparameters are likely not optimal for the other widths like they are for width 64.

The MountainCar environment exaggerates this lack suboptimality more than the other 3 environments.

The main next experiments we plan to run will use an automated tuning procedure that chooses the hyperparameters for each width individually.

We believe this protocol will yield MountainCar results that look much more like the CartPole and Acrobot results.

We then plan to replicate these findings across more learning algorithms and more environments.

A. Pendulum

<|TLDR|>

@highlight

Over-parametrization in width seems to help in deep reinforcement learning, just as it does in supervised learning.