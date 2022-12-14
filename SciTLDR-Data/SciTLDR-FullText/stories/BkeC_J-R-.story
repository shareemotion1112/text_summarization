Reinforcement learning methods have recently achieved impressive results on a wide range of control problems.

However, especially with complex inputs, they still require an extensive amount of training data in order to converge to a meaningful solution.

This limitation largely prohibits  their usage for complex input spaces such as video signals, and it is still impossible to use it for a number of complex problems in a real world environments, including many of those for video based control.

Supervised learning, on the contrary, is capable of learning on a relatively small number of samples, however it does not take into account reward-based control policies and is not capable to provide independent control policies.

In this article we propose a model-free control method, which uses a combination of reinforcement and supervised learning for autonomous control and paves the way towards policy based control in real world environments.

We use SpeedDreams/TORCS video game to demonstrate that our approach requires much less samples (hundreds of thousands against millions or tens of millions) comparing to the state-of-the-art reinforcement learning techniques on similar data, and at the same time overcomes both supervised and reinforcement learning approaches in terms of quality.

Additionally, we demonstrate the applicability of the method to MuJoCo control problems.

The problem becomes even more challenging if the results are dependent on the sequence of previous observations ), e.g. because of dynamic nature of the problem involving speed or acceleration, or the difference between the current and the previous control signal.

In many real-world problems, it is possible to combine the reinforcement and the supervised learning.

For the problem of autonomous driving, it is often possible to provide parallel signals of the autopilot in order to use this information to restrict the reinforcement learning solutions towards the sensible subsets of control actions.

Similar things can also be done for robotic control.

Such real world models can be analytical, or trained by machine learning techniques, and may use some other sensors, which are capable to provide alternative information (e.g., the model trained on LiDAR data can be used to train the vision based model).

However, although there were some works using partially labelled datasets within the reinforcement learning framework BID23 ), as far as we believe, the proposed problem statement, injecting supervised data into reinforcement learning using regularisation of Q-functions, is different from the ones published before.

In BID23 , the authors consider the problem of robotic control which does not involve video data, and their approach considers sharing the replay buffer between the reinforcement learning and demonstrator data.

The novelty of the approach, presented in this paper, is given as follows:1.

the regularised optimisation problem statement, combining reinforcement and supervised learning, is proposed;2.

the training algorithm for the control method is proposed, based on this problem statement, and assessed on the control problems;3.

the novel greedy actor-critic reinforcement learning algorithm is proposed as a part of the training algorithm, containing interlaced data collection, critic and actor update stages.

The proposed method reduces the number of samples from millions or tens of millions, required to train the reinforcement learning model on visual data, to just hundreds of thousands, and also improves the quality against the supervised and reinforcement learning.

The overall idea of the method is shown in figure 1 and can be described as follows: to perform an initial approximation by supervised learning and then, using both explicit labels and rewards, to fine-tune it.

For supervised pre-training, the annotated dataset should contain recorded examples of control by some existing model.

The aim of this stage is to mimic the existing control methods in order to avoid nonsense control behaviour of the trained model.

For supervised and reinforcement learning based fine-tuning, a pretrained model is used as an initial approximation.

Contrary to the standard reinforcement learning approaches, the pre-trained model helps it to avoid nonsense control values right from the beginning.

Also, for the control it is assumed that there is access to labels, which helps to divert the reinforcement learning model from spending its time on exploring those combinations of inputs and control signals, which are most likely not to provide meaningful solutions.

Hereinafter we consider the following problem statement for supervised pretraining.

Let Z be the space of all possible input signals.

To simplify the formalisation, we restrict ourselves to the image signals z ??? Z = R m??n , where m and n are the height and the width of the image, respectively.

Such signals can form sequences of finite length l > 0 : z 1 , z 2 , . . .

, z l ??? Z l .

We define an operator from the subsequences of real valued quantities to d-dimensional control signals ??(z i , z i???1 , . . .

, z i???p+1 |?? ?? ) : DISPLAYFORM0 , p ??? N is the number of frames used to produce the control signal, and ?? ?? are the parameters of the operator ??(??).

We denote The problem is stated as follows: given the set of N sequences ??? j ??? Z lj N j=1 DISPLAYFORM1 and the set of corresponding signal sequences ?? DISPLAYFORM2 , produced by some external control method called a 'reference actor', find the parameters ?? ?? of the actor ??(??|?? ?? ), which minimise a loss function (the used loss function is defined in formula FORMULA9 ).

The reinforcement learning method is inspired by the DDPG algorithm ), however, it is substantially reworked in order to meet the needs of the proposed combination of supervised and reinforcement learning working in real time.

First, the problem statement and basic definitions, necessary for formalisation of the method, need to be given.

In line with the standard terminology for reinforcement learning, we refer to a model, generating control signals, as an agent, and to control signals as actions.

Also, as it is usually done for reinforcement learning problem statements, we assume the states to be equivalent to the observations.

Initially, the model receives an initial state of the environment x 1 .

Then it applies the action c 1 , which results in transition into the state x 2 .

The procedure repeats recurrently, resulting in sequences of states X n = x 1 , x 2 , . . .

, x n and actions C n = c 1 , c 2 , . . . , c n .

Every time the agents performs an action it receives a reward r(x i , c i ) ??? R ).The emitted actions are defined by the policy ??, mapping states into actions.

In general BID18 ), it can be stochastic, defining a probability distribution over the action space, however here the deterministic policy is assumed.

For such a deterministic policy, one can express the discounted future reward using the following form of the Bellman equation for the expectation of discounted future reward Q(??, ??) ): DISPLAYFORM0 where ??(??) is an operator from states to actions, ?? ??? [0, 1] is a discount factor.

Q(??, ??) is approximated by the critic neural network in line with the actor-critic approach.

Similarly to , the proposed method uses a replay buffer to provide a training set.

In many of the state-of-the-art works on reinforcement learning, the actor's parameters are trained by maximisation of the critic's approximation of Q(??, ??) , using gradient descent with gradients, obtained using the chain rule , BID18 ): DISPLAYFORM1 where ?? Q and ?? ?? are the trainable parameters of the actor and the critic, respectively.

In , BID18 , the greedy optimisation algorithm is not used as it might be impossible to collect diverse training sample for the critic as the actor would not be able to advance through the task.

Instead, small steps in the gradient direction are applied to slightly modify the policy.

In the proposed method, contrary to many state-of-the-art methods, the optimisation is carried out in a greedy way, so that the steps of testing the current policy, updating the critic and the actor are interlaced.

This is done in order to meet the requirements of real world scenarios, namely minimising the difference between the measurements per second rate in testing and training scenarios and providing the reasonable performance of the actor after the smallest possible number of epochs.

In order to avoid deterioration of performance in the pretrained model (which would essentially lead to the number of steps comparable with the state-of-the-art reinforcement learning models), the regularisation is used to bring the parameters closer towards some pre-defined (reference) policy.

In the following derivations we use a restriction that Q(??, ??) ??? 0, which, as one can easily see from Equation (1), will be true if the rewards are non-negative.

We also assume (due to the practical reasons, it doesn't affect further derivations) that the control signal is bounded between values (t 1 , t 2 ), and t 1 and t 2 are not included into the appropriate operational values.

Based on these practical assumptions, the training sample for the critic is augmented with the values Q(x, t 1 ) = Q(x, t 2 ) = 0 for every value x.

The optimisation problem is based on the regularised Q-function f (x, ??(x|?? ?? )) : DISPLAYFORM2 DISPLAYFORM3 where ?? ??? 0 is some coefficient,??(??) is the reference actor, and ??(??, ??) is a differentiable distancelike function.

One can easily see that in the case ?? = 0 it completely coincides with the reinforcement problem, and with ?? ??? ??? the problem becomes a standard supervised training one.

The weight Q(x,??(x)) is given to encourage the actor to follow the reference policy if the expected reward is high, and not to encourage much otherwise.

After differentiating this expression with respect to ?? ?? one can see that DISPLAYFORM4 As in , the update of the critic is carried out by solving the following optimisation problem: DISPLAYFORM5 whereX and?? are taken from the replay buffer,Q ?? is the discounted reward function approximation.

One can see that this equation is recurrent, and therefore the current target for training is dependent on the values of the previous training stage.

The training procedure is carried out in the way described in Algorithm 1.

In this algorithm, NUM EPOCHS denotes the maximum number of epochs during the training procedure.

In a contrast to , the proposed model detaches the procedures of filling the replay buffer, training critic and actor into three subsequent steps, making the algorithm greedy rather than making steps every time control happens.

As one can see from the algorithm, the 0-th epoch's testing episodes reflect the performance of the model with supervised pretraining.

This was done to assess the performance of the same model parameterisation during the epoch, as well as to exclude the problem when the control signal frequency (and hence performance) is affected by the optimisation time.

To demonstrate the ability of the method to learn control signals, TORCS (Espi et al.) / SpeedDreams (SpeedDreams) driving simulator was used.

As a reference actor and for the baseline, the Simplix bot, which is a part of the standard package, was used.

The outputs of the bot steering angles were restricted between ???0.25 and 0.25.

The car name was Lynx 220.

In the case if the car has stuck (average speed is less than one unit as measured by the simulator), the recovery procedure is provided by the bot.

The time and rewards for the recovery are excluded from consideration.

The reward is defined as the current car speed measured by the simulator.

The assessment has been carried out on a single computer, using NVIDIA GeForce GTX 980 Ti graphical processor on Ubuntu Linux operating system.

The model parameters are described in the Appendix A.

Initialise and train the parameters ?? ?? for the actor ??(x|?? ?? ) on a supervised dataset with observations and labels {X S , C S } Initialise the parameters of the critic ?? Q , k = 1 Initialise the empty replay buffer B while k ??? NUM EPOCHS do Perform N TESTING EPISODES testing episodes of the length L TESTING EPISODES with the current parameters ?? ?? , where the states x i , actions c i , reference actor actions c GT i , rewards r i = r(x i , c i ) and subsequent states x i+1 are collected and put into the replay buffer B Perform N GD UPDATE CRITIC iterations of gradient descent, according to the optimisation problem (6), in order to update the parameters of the critic ?? Q ; values x i , x i+1 , c i , r i are taken from the replay buffer B Perform N GD UPDATE ACTOR iterations of gradient descent for the optimisation problem The network architecture, used for the proposed actor-critic approach, is shown in Figure 2 .

For supervised pretraining, the dataset has been collected from the SpeedDreams simulator, combining the sequences of images and the corresponding control values produced by the bot.

Using this dataset, the InceptionV1 network BID21 ) is first fine-tuned from the Tensorflow Slim implementation BID17 ) on the collected dataset, mapping each single image directly into the control signals.

The last (classification) layer of the network is replaced by a fully connected layer of the same size (1001) as the preceding logits layer; this layer is referred to as a feature extraction layer as it serves as an input space for the actor-critic model.

Such an approach is used in order to avoid expensive storage of images in the replay buffer, as well as circumvent the challenge of training deep architectures within the reinforcement learning setting.

Each of the actor and critic layers except the last contains ReLU nonlinearity BID14 ).

The actor's last layer with dimensionality d of the control signal is followed by the tanh nonlinearity, which restricts the output between the boundary values (t 1 , t 2 ) = (???1, 1); the last layer of the critic has no nonlinearities.

For supervised finetuning and pretraining stages, we use the following loss function: DISPLAYFORM0 where is a reasonably small constant (with respect to c, we use = 10 ???2 ), ??(x) is an operator, transforming input vectors x to the control signals, c is the reference actor control signal.

This was done because some of the control signals, supplied by the bot, were large enough to devaluate the impact of the smaller values within the loss.

Similarly, the distance-like function ?? in formula (4) is defined as: DISPLAYFORM1 where is a reasonably small constant (we use = 10 ???4 ).To implement the formalisation, described in section 2.1, we use a syamese network architecture (see BID1 ), given in Figure 2 .

The features, calculated by the fine-tuned network, are submitted for the current and the previous frame (p = 2 in terminology of section 2.1).

It is done in order to take into account the dynamic state of the environment, including speed.

Also we need to mention that the previous control signal is submitted as an input for the critic (for the initial video frame in each sequence, we assume that the frames and control signals are the same for the current and the previous frames).

TAB1 and Figure 3 contain the comparison of the results given different values of the parameter ??.

When ?? = 0, we rely solely on reinforcement learning; the higher is the parameter ??, the more restriction is put to stick to the supervised labels rather than improving the policy by reinforcement learning.

In Figure 3 , the scatter points depict total rewards during each of the testing episodes, and the curves are depicting the mean total rewards for each epoch over all testing episode rewards.

The total reward is calculated as an arithmetic sum of the rewards during one training episode.

The left figure shows the results for all tested parameters, while the right one shows the comparison between the best performing one and the one with the largest value of ?? (i.e. the closest to supervised active learning).

The performance of the pretrained model corresponds to the first epoch in the graph.

The shaded area in the right figure shows the standard deviation of the testing episodes performance during the epoch.

One can see from the figure that the smaller values of the coefficient ?? tend to yield worse performance; however, it may be attributed to longer convergence time as it implies more reliance on reinforcement learning exploration.

At the same time, the unlimited increase of the coefficient ?? does not help gaining further performance improvements.

Even more, one can see from the right figure that after certain point the curve for ?? = 100 slowly declines; we suggest that it could be caused by overfitting to the bot values.

We also see that for ?? = 0, when the supervised data is used only during the pre-training stage, the performance is much lower than for the rest of the graphs.

TAB1 shows the total rewards for different values of parameter ??.

The value max R shows the maximum total reward over one testing episode during all the training time (corresponds to the highest scatter point in Figure 3 for a given parameter ??), the value max R shows the maximum mean total reward, where the mean is calculated for each epoch over all testing episode total rewards (corresponds to the highest point of the curve in Figure 3 for a given parameter ??).In order to compare the rewards shown in these graphs and tables, we have also measured them the Simplix bot in the same conditions (frame per second rate) as the proposed algorithm.

The total rewards for the Simplix bot, available in the SpeedDreams/TORCS simulator, are given in TAB0 .

The mean value of these rewards is R bot = 81214.07.

The maximum value achieved by the bot is max R bot = 81931.15.

The percentage of the proposed algorithm's rewards with respect to the bot one, in average and for the best performance, is also given in TAB1 .

FORMULA3 ) or tens of millions BID12 ) measurements for standard reinforcement learning techniques ) to just hundreds of thousands (15000 per training epoch, several tens of training epochs).

For some of the reinforcement learning algorithms, trained on a problem of driving in a simulator, only some realisations were able to finish the task ), and for those methods, which report solving similar problems by reinforcement learning only, the reported performance constitutes 75 ??? 90% BID13 ), while we achieve up to 92.63% of the bot's performance as reported in TAB1 Another interesting aspect to be considered is how the proposed method would behave for other well-known control scenarios.

Appendix B discusses the performance of the method for MuJoCo environments BID22 ) using OpenAI Baselines package BID2 ).

The proposed method shows dramatic improvement in the number of samples for video data (down to just several hundred thousand) comparing to the reinforcement learning methods, as well as improves the performance comparing to both supervised and reinforcement learning.

We believe that such approach, combining reinforcement and supervised learning, could help to succeed in the areas of complex spaces where the state-of-the-art reinforcement learning methods are not working yet, as well as towards practical usage for real world models such as autonomous cars or robots.

However, there are still a few limitations of the proposed method.

First, it still requires label data through all the course of training.

We believe that in the future work it should be possible to reduce usage of training data to a limited number of labelled episodes.

Such decrease of the training data could benefit to the range of practical tasks solvable by the proposed approach.

The parameterisation for the experiments is given in TAB2 ; the parameters' verbal description is augmented with the names referencing to Algorithm 1.For supervised-only pretraining of the actor network, the Momentum algorithm is used BID15 ); for the rest of the stages, the Adam algorithm is used BID7 ).

The proposed algorithm has been implemented in Python using TensorFlow framework BID0 ).

For the stage of supervised pretraining, in order to improve convergence of the model at the initial stage, the additional soft update coefficient was introduced for exponential smoothing of the parameters of the network during gradient descent optimisation.

The experiments with MuJoCo environment BID22 ) demonstrate the applicability of the proposed idea of regularising Q-values with supervised learning to other well known control problems.

For this purpose, we have added the proposed Q-function regularisation to the standard DDPG algorithm with the implementation, network architecture and parameters, taken from OpenAI Baselines BID2 ).

L 2 distance has been used as ?? for the equation (4).

Also, in order to maintain the correctness of regularisation assumptions, as the condition Q ??? 0 is not met for some low reward values in MuJoCo, Q(x,??(x)) was substituted by max(0, Q(x,??(x)).

The reference actors were obtained by pretraining actors by standard DDPG algorithm.

The results of these experiments are given in FIG5 .

In every graph, the black lines show the performance of the pretrained reference actor.

These experiments are aimed to compare the following three cases:1.

the original DDPG algorithm (referenced as ?? = 0) 2.

the DDPG algorithm with fixed regularisation coefficient, ?? = 0.1 3.

the DDPG algorithm with exponential decay, initial value of ?? is 1, the value is decayed with the coefficient 0.01 every 20, 000 timesteps.

For the HalfCheetah scenario, one can see that the model with the fixed regularisation coefficient can easily reach the pretrained value but then lags behind the algorithm with exponential decay.

The exponential decay algorithm, in contrary, takes advantage of the reference actor performance and then gradual decay of the regularisation enables it to explore values further from the reference actor.

These results could suggest that in certain cases the regularisation can prevent the model of further exploration beyond the reference actor performance.

For the Hopper scenario, the peak in the performance of original DDPG algorithm beyond the reference actor baseline near step 270000 suggests that the original DDPG algorithm may be unstable for this task, which also holds for DDPG with exponential decay as the regularisation coefficient becomes negligibly small.

At the same time one can see that the model with the fixed regularisation coefficient can reach performance beyond the reference actor.

It could be concluded from the graphs for the InvertedDoublePendulum task that convergence depends in this case on the initial value of parameter ??.

The larger initial value for the DDPG with exponential decay appears to give better results due to heavier reliance on the supervised part.

What is important is that all versions of the method are able to maintain stable performance after the initial training episode.

For the Swimmer scenario, the exponential ?? setting allowed to go for some period of time beyond the reference actor baseline; at the same time, the version with ?? = 0.1, while not beating the reference actor, shows smaller variance than the original algorithm, and most of the time exposes better average performance than the original DDPG method.

@highlight

The new combination of reinforcement and supervised learning, dramatically decreasing the number of required samples for training on video

@highlight

This paper proposes leveraging labelled controlled data to accelerate reinforcement-based learning of a control policy