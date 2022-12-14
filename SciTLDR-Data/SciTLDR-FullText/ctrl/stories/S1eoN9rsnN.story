Reinforcement learning (RL) methods achieved major advances in multiple tasks surpassing human performance.

However, most of RL strategies show a certain degree of weakness and may become computationally intractable when dealing with high-dimensional and non-stationary environments.

In this paper, we build a meta-reinforcement learning (MRL) method embedding an adaptive neural network (NN) controller for efficient policy iteration in changing task conditions.

Our main goal is to extend RL application to the challenging task of urban autonomous driving in CARLA simulator.

"

Every living organism interacts with its environment and uses those interactions to improve its own actions in order to survive and increase" BID13 .

Inspired from animal behaviorist psychology, reinforcement learning (RL) is widely used in artificial intelligence research and refers to goal-oriented optimization driven by an impact response or signal BID30 .

Properly formalized and converted into practical approaches BID9 , RL algorithms have recently achieved major progress in many fields as games BID18 BID28 and advanced robotic manipulations BID12 BID17 beating human performance.

However, and despite several years of research and evolution, most of RL strategies show a certain degree of weakness and may become computationally intractable when dealing with high-dimensional and non-stationary environments BID34 .

More specifically, the industrial application of autonomous driving in which we are interested in this work, remains a highly challenging "unsolved problem" more than one decade after the promising 2007 DARPA Urban Challenge BID2 ).

The origin of its complexity lies in the large variability inherent to driving task arising from the uncertainty of human behavior, diversity of driving styles and complexity of scene perception.

An interpretation of the observed vulnerability due to learning environment changes has been provided in contextaware (dependence) research assuming that "concepts in the real world are not eternally fixed entities or structures, but can have a different appearance or definition or meaning in different contexts" BID36 .

There are several tasks that require context-aware adaptation like weather forecast with season or geography, speech recognition with speaker origins and control processes of industrial installations with climate conditions.

One solution to cope with this variability is to imitate the behavior of human who are more comfortable with learning from little experience and adapting to unexpected perturbations.

These natural differences compared to machine learning and specifically RL methods are shaping the current research intending to eschew the problem of data inefficiency and improve artificial agents generalization capabilities BID10 .

Tackling this issue as a multi-task learning problem BID3 , meta-learning has shown promising results and stands as one of the preferred frames to design fast adapting strategies BID25 BID23 .

It refers to learn-to-learn approaches that aim at training a model on a set of different but linked tasks and subsequently generalize to new cases using few additional examples BID7 .In this paper we aim at extending RL application to the challenging task of urban autonomous driving in CARLA simulator.

We build a meta-reinforcement learning (MRL) method where agent policies behave efficiently and flexibly in changing task conditions.

We consolidate the approach robustness by integrating a neural network (NN) controller that performs a continuous iteration of policy evaluation and improvement.

The latter allows reducing the variance of the policy-based RL and accelerating its convergence.

Before embarking with a theoretical modeling of the proposed approach in section 3, we introduce in the next section metalearning background and related work in order to better understand the current issues accompanying its application to RL settings.

In the last section, we evaluate our method using CARLA simulator and discuss experimental results.

Generally, in order to acquire new skills, it is more useful to rely on previous experience than starting from scratch.

Indeed, we learn how to learn across tasks requiring, each time, less data and trial-and-error effort to conquer further skills BID10 .

The term meta-learning that refers to learning awareness on the basis of prior experience was first cited by BID0 in the field of educational psychology.

It consists in taking control of a learning process and guiding it in accordance with the context of a specific task.

In machine learning research, meta-learning is not a new concept and displays many similarities with the above definition BID31 BID26 BID19 .

It assumes that rather than building a learning strategy on the basis of a single task, it will be more effective to train over a series of tasks sharing a set of similarities then generalize to new situations.

By acquiring prior biases, meta-learning addresses models inaccuracies achieving fast adaptation from few additional data BID4 .

At an architectural level, the learning is operated at two scales: a base-level system is assigned to rapid learning within each task, and a meta (higher) level system uses previous one feedback for gradual learning across tasks BID35 .One of the first contribution to meta-learning is the classical Algorithm Selection Problem (ASP) proposed by BID24 considering the relationship between problem characteristics and the algorithm suitable to solve it.

Then based on the concept of ASP, the No Free Lunch (NFL) theorem BID38 demonstrated that the generalization performance of any learner across all tasks is equal to 0.

The universal learner is consequently a myth and each algorithm performs well only on a set of tasks delimiting its area of expertise.

ASP and NFL theorem triggered a large amount of research assigned to parameter and algorithm recommendation BID8 BID1 BID29 BID22 .

In this type of meta-learning, a meta-learner apprehend the relationship between data characteristics called meta-features and base-learners performance in order to predict the best model to solve a specific task.

Various meta-learners have been used and generally consist of shallow algorithms like decision trees, k-Nearest Neighbors and Support Vector Machines BID32 .

Regarding meta-features, the most commonly used ones included statistical and information-theoretic parameters as well as land-marking and model-based extractors BID33 ).The recent regain of interest in neural network models and more specifically deep learning resulting from the advent of large training datasets and computational resources allowed the resurgence of neural network Meta-learning BID14 .

Instead of requiring explicit task characteristics, the meta-level learns from the structure of base-models themselves.

Neural networks are particularly suitable to this kind of transfer learning given their inner capabilities of data features abstraction and rule inductions reflected in their connection weights and biases.

The typology of meta-learners developed so far includes recurrent models, metrics and optimizers with several areas of application in classification, regression and RL BID15 .Meta-learning algorithms extended recently to the context of RL can be classified in two broad categories.

A first set of methods implement a recurrent neural network (RNN) or its memory-augmented variant (LSTM) as the meta-learner.

BID6 study RL optimization in the frame of a reinforcement learning problem (RL2) where policies are represented with RNNs that receive past rewards and actions, in addition to the usual inputs.

The approach is evaluated on multi-armed bandits (MAB) and tabular Markov Decision Processes (MDPs).

In BID35 , Advantage Actor-Critic (A2C) algorithms with recurrence are trained using different architectures of LSTM (simple, convolutional and stacked).

The experiments are conducted on bandits problems with increasing level of complexity (dependent/independent arms and restless).In the second category, the learner gradients are used for meta-learning.

Such methods are task-agnostic and adaptable to any model trained with gradient-descent.

The gradient-based strategy has been originally introduced by BID7 with their Model-Agnostic Meta-Learning (MAML) algorithm.

It has been demonstrated efficient for different problem settings including gradient RL with neural network policies.

MAML mainly aims at generating a model initialization sensitive to changes and reaching optimal results on a new scenario after just few gradient updates.

Meta-SGD BID15 uses stochastic gradient descent to meta-learn, besides a model initialization, the inner loop learning rate and the direction of weights update.

In Reptile BID20 , the authors design a first order approximation of MAML computationally less expensive than the original method which includes second order derivative of gradient.

BID27 propose a probabilistic view of MAML for continuous adaptation in RL settings.

A competitive multi-agent environment (RoboSumo) was designed to run iterated adaptation games for the approach testing.

A major part of MRL papers have been evaluated either at a preliminary level of experimentation or on elementary tasks (2D navigation, simulated muJoCo robots and bandit problems).

In this work we consider an application of gradient-based MRL in a more challenging dynamic environment involving realistic and complex sides of real world tasks, which is CARLA simulator for autonomous driving BID5 .

The proposed model consists of a MRL framework embedding an adaptive NN controller to tackle both the nonstationarity and high dimensionality issues inherent to autonomous driving environments in CARLA simulator.

The RL task considered in this work is a Markov Decision Process (MDP) defined according to the tuple (S, A, p, r, ??, ?? 0 , H) where S is the set of states, A is the set of actions, p(s t+1 |s t , a t ) is the state transition distribution predicting the probability to reach a state s t+1 in the next time step given current state and action, r is a reward function, ?? is the discount factor, ?? 0 is the initial state distribution and H the horizon.

Consider the sum of expected rewards (return) from a trajectory ?? (0,H???1) = (s 0 , a 0 , ..., s H???1 , a H???1 , s H ).

A RL setting aims at learning a policy ?? of parameters ?? (either deterministic or stochastic) that maps each state s to an optimal action a maximizing the return R of the trajectory.

DISPLAYFORM0 Following the discounted return expressed above, we can define a state value function V (s) : S ??? R to measure the current state return estimated under policy ??: DISPLAYFORM1 In order to optimize the parameterized policy ?? ?? , we use gradient descents like in the family of REINFORCE algorithms BID37 updating the policy parameters ?? in the direction: DISPLAYFORM2

We build an approach of NN meta-learning compatible with RL setting.

Our contribution consists in combining (1) a gradient-based meta-learner like in MAML BID7 to learn a generalizable model initialization and (2) a NN controller for more robust and continuous adaptation.

The agent policy ?? ?? approximated by a convolutional neural network (CNN) is trained to quickly adapt to a new task through few standard gradient descents.

Explicitly, this consists in finding an optimal initialization of parameters ?? * allowing a few-shot generalization of the learned model.

Given a batch of tasks T i sampled from p(T ), the metaobjective is formulated as follows: DISPLAYFORM0 The MRL approach includes two levels of processing: the inner and the outer loops associated respectively to the base and meta-learning.

In the inner loop, we start by reducing the disturbances characterizing policy based methods and induced by the score function R t .

Indeed, complex domains with conflicting dynamics and high dimensional observations like autonomous driving yield a large amount of uncertainty.

One flexible solution to reduce disturbances and accelerate learning convergence is policy iteration.

Subsequently, we modify the RL scheme by integrating a step of policy evaluation and improvement that generates added bonuses to guide the agent towards new states.

The policy evaluation is performed with temporal difference (TD) learning combining Monte Carlo method and dynamic programming BID30 to learn, with step size ??, the value function approximated by a CNN: DISPLAYFORM1 Where ?? t is the multi-step TD error that consists in bootstrapping the sampled returns from the value function estimate: DISPLAYFORM2 Multi-step returns allow the agent to gather more information on the environment before calculating the error in the value function estimates.

Subsequently, the improvement of the policy is performed through the replacement of the score function R t by the TD error ?? t in the policy gradient: DISPLAYFORM3 For each sampled task T i , the policy parameters ?? i are computed using the updated gradient descent: DISPLAYFORM4 Once the models and related evaluations are generated for all batch tasks, the outer loop is activated.

It consists in operating a meta-gradient update of the initial model parameters with a meta-step size ?? on the basis of the previous level rewards R Ti (?? i ): DISPLAYFORM5 The steps detailed above are iterated until an accepted performance is reached.

The resulting model initialization ?? * should be able to achieve fast driving adaptation after only a few gradient steps.

In this section we evaluate the performance of the continuous-adapting MRL model on the challenging task of urban autonomous driving.

The goal of our experiment is to demonstrate the effectiveness of meta-level learning combined with a NN controller to optimize the RL policy and achieve a more robust learning of high-dimensional and complex environments.

At this stage of work, we present the preliminary results of our study assessing 2 basic assumptions.

The MRL agent is (1) adapting faster at training time and (2) displaying better generalization capabilities in unseen environments.

Environment settings.

We conduct our experiments using CARLA simulator for autonomous driving BID5 BID21 designed as a server-client system.

Carla 3D environment consists of static objects and dynamic characters.

As we consider the problem of autonomous driving in changing conditions, we induce nonstationary environments across training episodes by varying several server settings.

(1) The task complexity: select one of the available towns as well as different start and end positions for the vehicle tasks (straight or with-turn driving).

FORMULA1 The traffic density: control the number of dynamic objects such as pedestrians and vehicles.

(3) Weather and lightening: select a combination of weather and illumination conditions to diversify visual effects controlling sun position, radiation intensity, cloudiness and precipitation.

Hence we can exclusively use a subset of environments for meta-training ("seen") and a second subset for test-time adaptation ("unseen").

The reward is shaped as a weighted sum of the distance traveled to target, speed in km/h, collisions damage and overlaps with sidewalk and opposite lane.

Results.

Given the preliminary level of experiments and the absence of various state-of-the-art work on the recent CARLA simulator, we adopt BID7 methodology consisting in comparing the continuous-adapting MRL initialization with conventionally pre-trained and randomly initialized RL algorithms.

In all experiments the average episodic reward is used to describe the methods global performance.

An episode is terminated when the target destination is reached or after a collision with a dynamic character.

FIG1 depicts the test-time adaptation performance of the 3 models.

During this phase, the RL agent initialized with meta-learning still uses the NN controller for continuous adaptation.

The results confirm that our approach generates models adapting faster in "unseen" environments comparatively to the standard RL strategies.

Zooming the initial driving steps (figure 1), we notice that our method has distinctly surpassed the standard RL versions only after 10000 steps (500 gradient descents).

Subsequently we should lead further tests to identify a specific threshold for few shot learning when evolving from low to high-dimensional settings like autonomous driving task.

In order to evaluate the generalization assumption, we compare the models behavior on "seen" and "unseen" environments.

FIG2 does not reveal a significant "shortfall" of our approach performance between the 2 scenarios reflecting its robustness in non-stationary conditions.

In the contrary, the performance of the pre-trained standard RL decreased notably in "unseen" environments due to the lack of generalization capabilities.

Although all results indicate a certain robustness of the continuous-adapting MRL, it is too early to draw firm conclusions at this preliminary stage of evaluation.

First, the episodic reward indicator should be completed with the percentage of successfully ended episodes in order to demonstrate the effective learning of the agent and allow the comparison with state-of-the-art work BID5 BID16 .

Second, further consideration should be addressed to the pertinence of few shot learning regimes in very complex and high dimensional environments like autonomous driving since the meta-learned strategy may acquires a particular bias at training time "that allows it to perform better from limited experience but also limits its capacity of utilizing more data" BID27 .

In this paper we addressed the limits of RL algorithms in solving high-dimensional and complex tasks.

Built on gradient-based meta-learning, the proposed approach implements a continuous process of policy assessment and improvement using a NN controller.

Evaluated on the challenging problem of autonomous driving using CARLA simulator, our approach showed higher performance and faster learning capabilities than conventionally pre-trained and randomly initialized RL algorithms.

Considering this paper as a preliminary attempt to scale up RL approaches to high-dimensional real world applications like autonomous driving, we plan in future work to bring deeper focus on several sides of the approach such as the reward function, CNN architecture and including vehicle characteristics in the tasks complexity setup.

<|TLDR|>

@highlight

A meta-reinforcement learning approach embedding a neural network controller applied to autonomous driving with Carla simulator.