The information bottleneck principle is an elegant and useful approach to representation learning.

In this paper, we investigate the problem of representation learning in the context of reinforcement learning using the information bottleneck framework, aiming at improving the sample efficiency of the learning algorithms.

We analytically derive the optimal conditional distribution of the representation, and provide a variational lower bound.

Then, we maximize this lower bound with the Stein variational (SV) gradient method.

We incorporate this framework in the advantageous actor critic algorithm (A2C) and the proximal policy optimization algorithm (PPO).

Our experimental results show that our framework can improve the sample efficiency of vanilla A2C and PPO significantly.

Finally, we study the information-bottleneck (IB) perspective in deep RL with the algorithm called mutual information neural estimation(MINE).

We experimentally verify that the information extraction-compression process also exists in deep RL and our framework is capable of accelerating this process.

We also analyze the relationship between MINE and our method, through this relationship, we theoretically derive an algorithm to optimize our IB framework without constructing the lower bound.

In training a reinforcement learning algorithm, an agent interacts with the environment, explores the (possibly unknown) state space, and learns a policy from the exploration sample data.

In many cases, such samples are quite expensive to obtain (e.g., requires interactions with the physical environment).

Hence, improving the sample efficiency of the learning algorithm is a key problem in RL and has been studied extensively in the literature.

Popular techniques include experience reuse/replay, which leads to powerful off-policy algorithms (e.g., (Mnih et al., 2013; Silver et al., 2014; Van Hasselt et al., 2015; Nachum et al., 2018a; Espeholt et al., 2018 )), and model-based algorithms (e.g., (Hafner et al., 2018; Kaiser et al., 2019) ).

Moreover, it is known that effective representations can greatly reduce the sample complexity in RL.

This can be seen from the following motivating example: In the environment of a classical Atari game: Seaquest, it may take dozens of millions samples to converge to an optimal policy when the input states are raw images (more than 28,000 dimensions), while it takes less samples when the inputs are 128-dimension pre-defined RAM data (Sygnowski & Michalewski, 2016) .

Clearly, the RAM data contain much less redundant information irrelevant to the learning process than the raw images.

Thus, we argue that an efficient representation is extremely crucial to the sample efficiency.

In this paper, we try to improve the sample efficiency in RL from the perspective of representation learning using the celebrated information bottleneck framework (Tishby et al., 2000) .

In standard deep learning, the experiments in (Shwartz-Ziv & Tishby, 2017) show that during the training process, the neural network first "remembers" the inputs by increasing the mutual information between the inputs and the representation variables, then compresses the inputs to efficient representation related to the learning task by discarding redundant information from inputs (decreasing the mutual information between inputs and representation variables).

We call this phenomena "information extraction-compression process" "information extraction-compression process" "information extraction-compression process"(information E-C process).

Our experiments shows that, similar to the results shown in (Shwartz-Ziv & Tishby, 2017) , we first (to the best of our knowledge) observe the information extraction-compression phenomena in the context of deep RL (we need to use MINE (Belghazi et al., 2018) for estimating the mutual information).

This observation motivates us to adopt the information bottleneck (IB) framework in reinforcement learning, in order to accelerate the extraction-compression process.

The IB framework is intended to explicitly enforce RL agents to learn an efficient representation, hence improving the sample efficiency, by discarding irrelevant information from raw input data.

Our technical contributions can be summarized as follows:

1.

We observe that the "information extraction-compression process" also exists in the context of deep RL (using MINE (Belghazi et al., 2018) to estimate the mutual information).

2.

We derive the optimization problem of our information bottleneck framework in RL.

In order to solve the optimization problem, we construct a lower bound and use the Stein variational gradient method developed in (Liu et al., 2017) to optimize the lower bound.

3.

We show that our framework can accelerate the information extraction-compression process.

Our experimental results also show that combining actor-critic algorithms (such as A2C, PPO) with our framework is more sample-efficient than their original versions.

4.

We analyze the relationship between our framework and MINE, through this relationship, we theoretically derive an algorithm to optimize our IB framework without constructing the lower bound.

Finally, we note that our IB method is orthogonal to other methods for improving the sample efficiency, and it is an interesting future work to incorporate it in other off-policy and model-based algorithms.

Information bottleneck framework was first introduced in (Tishby et al., 2000) .

They solve the framework by iterative Blahut Arimoto algorithm, which is infeasible to apply to deep neural networks. (Shwartz-Ziv & Tishby, 2017) tries to open the black box of deep learning from the perspective of information bottleneck, though the method they use to compute the mutual information is not precise. (Alemi et al., 2016 ) derives a variational information bottleneck framework, yet apart from adding prior target distribution of the representation distribution P (Z|X), they also assume that P (Z|X) itself must be a Gaussian distribution, which limits the capabilities of the representation function. (Peng et al., 2018) extends this framework to variational discriminator bottleneck to improve GANs (Goodfellow et al., 2014) , imitation learning and inverse RL.

As for improving sample-efficiency, (Mnih et al., 2013; Van Hasselt et al., 2015; Nachum et al., 2018a) mainly utilize the experience-reuse.

Besides experience-reuse, (Silver et al., 2014; Fujimoto et al., 2018) tries to learn a deterministic policy, (Espeholt et al., 2018) seeks to mitigate the delay of off-policy. (Hafner et al., 2018; Kaiser et al., 2019 ) learn the environment model.

Some other powerful techniques can be found in (Botvinick et al., 2019) .

State representation learning has been studied extensively, readers can find some classic works in the overview (Lesort et al., 2018) .

Apart from this overview, (Nachum et al., 2018b) shows a theoretical foundation of maintaining the optimality of representation space. (Bellemare et al., 2019) proposes a new perspective on representation learning in RL based on geometric properties of the space of value function. (Abel et al., 2019) learns representation via information bottleneck(IB) in imitation/apprenticeship learning.

To the best of our knowledge, there is no work that intends to directly use IB in basic RL algorithms.

A Markov decision process(MDP) is a tuple, (X , A, R, P, ??), where X is the set of states, A is the set of actions, R : X ?? A ?? X ??? R is the reward function, P :

is the transition probability function(where P (X ??? |X, a) is the probability of transitioning to state X ??? given that the previous state is X and the agent took action a in X), and ?? : X ???[0, 1] is the starting state distribution.

A policy ?? : X ??? P(A) is a map from states to probability distributions over actions, with ??(a|X) denoting the probability of choosing action a in state X.

In reinforcement learning, we aim to select a policy ?? which maximizes

is the expected return by policy ?? after taking action a in state X.

Actor-critic algorithms take the advantage of both policy gradient methods and valuefunction-based methods such as the well-known A2C (Mnih et al., 2016) .

Specifically, in the case that policy ??(a|X; ??) is parameterized by ??, A2C uses the following equation to approximate the real policy gradient

where R t = ??? ??? i=0 ?? i r t+i is the accumulated return from time step t, H(p) is the entropy of distribution p and b(X t ) is a baseline function, which is commonly replaced by V ?? (X t ).

A2C also includes the minimization of the mean square error between R t and value function V ?? (X t ).

Thus in practice, the total objective function in A2C can be written as:

where ?? 1 , ?? 2 are two coefficients.

In the context of representation learning in RL,

The information bottleneck framework is an information theoretical framework for extracting relevant information, or yielding a representation, that an input X ??? X contains about an output Y ??? Y. An optimal representation of X would capture the relevant factors and compress X by diminishing the irrelevant parts which do not contribute to the prediction of Y .

In a Markovian structure X ??? Z ??? Y where X is the input, Z is representation of X and Y is the label of X, IB seeks an embedding distribution P ??? (Z|X) such that:

Under review as a conference paper at ICLR 2020 for every X ??? X , which appears as the standard cross-entropy loss 1 in supervised learning with a MI-regularizer, ?? is a coefficient that controls the magnitude of the regularizer.

Next we derive an information bottleneck framework in reinforcement learning.

Just like the label Y in the context of supervised learning as showed in (3), we assume the supervising signal Y in RL to be the accurate value R t of a specific state X t for a fixed policy ??, which can be approximated by an n-step bootstrapping function

be the following distribution:

.This assumption is heuristic but reasonable: If we have an input X t and its relative label Y t = R t , we now have X t 's representation Z t , naturally we want to train our decision function V ?? (Z t ) to approximate the true label Y t .

If we set our target distribution to be

For simplicity, we just write P (R|Z) instead of P (Y t |Z t ) in the following context.

With this assumption, equation (3) can be written as:

The first term looks familiar with classic mean squared error in supervisd learning.

In a network with representation parameter ?? and policy-value parameter ??, policy loss??(Z; ??) in equation (1) and IB loss in (5) can be jointly written as:

where I(X, Z; ??) denotes the MI between X and Z ??? P ?? (??|X).

Notice that J(Z; ??) itself is a standard loss function in RL as showed in (2).

Finally we get the ultimate formalization of IB framework in reinforcement learning:

The following theorem shows that if the mutual information I(X, Z) of our framework and common RL framework are close, then our framework is near-optimality.

Theorem Theorem Theorem 1 (Near-optimality theorem).

Policy ?? r = ?? ?? r , parameter ?? r , optimal policy ?? ??? = ?? ?? ??? and its relevant representation parameter ?? ??? are defined as following:

Define J

Assume that for any

In this section we first derive the target distribution in (7) and then seek to optimize it by constructing a variational lower bound.

We would like to solve the optimization problem in (7):

Combining the derivative of L 1 and L 2 and setting their summation to 0, we can get that

We provide a rigorous derivation of (11) in the appendix(A.2).

We note that though our derivation is over the representation space instead of the whole network parameter space, the optimization problem (10) and the resulting distribution (11) are quite similar to the one studied in (Liu et al., 2017) in the context of Bayesian inference.

However, we stress that our formulation follows from the information bottleneck framework, and is mathematically different from that in (Liu et al., 2017) .

In particular, the difference lies in the term L 2 , which depends on the the distribution P ?? (Z | X) we want to optimize (while in (Liu et al., 2017) , the corresponding term is a fixed prior).

The following theorem shows that the distribution in (11) is an optimal target distribution (with respect to the IB objective L).

The proof can be found in the appendix(A.3).

Theorem Theorem Theorem 2. (Representation Improvement Theorem) Consider the objective function

, given a fixed policy-value parameter ??, representation distribution P ?? (Z|X) and state distribution P (X).

Define a new representation distribution:

Though we have derived the optimal target distribution, it is still difficult to compute P ?? (Z).

In order to resolve this problem, we construct a variational lower bound with a distribution U (Z) which is independent of ??. Notice that

.

Now, we can derive a lower bound of L(??, ??) in (6) as follows:

Naturally the target distribution of maximizing the lower bound is:

Next we utilize the method in (Liu & Wang, 2016) (Liu et al., 2017) (Haarnoja et al., 2017) to optimize the lower bound.

Stein variational gradient descent(SVGD) is a non-parametric variational inference algorithm that leverages efficient deterministic dynamics to transport a set of particles

to approximate given target distributions Q(Z).

We choose SVGD to optimize the lower bound because of its ability to handle unnormalized target distributions such as (13).

Briefly, SVGD iteratively updates the "particles"

via a direction function ?? ??? (??) in the unit ball of a reproducing kernel Hilbert space (RKHS) H:

where ?? * (??) is chosen as a direction to maximally decrease 2 the KL divergence between the particles' distribution P (Z) and the target distribution Q(Z) =Q

In fact, ?? * is chosen to maximize the directional derivative of F (P ) = ???DKL(P ||Q), which appears to be the "gradient" of F distribution, C is normalized coefficient) in the sense that

where P [????] is the distribution of Z + ????(Z) and P is the distribution of Z. (Liu & Wang, 2016) showed a closed form of this direction:

where K is a kernel function(typically an RBF kernel function).

Notice that C has been omitted.

In our case, we seek to minimize

, which is equivalent to maximizeL(??, ??), the greedy direction yields:

In practice we replace log U (???) with ?? log U (???) where ?? is a coefficient that controls the magnitude of ?????? log U (???).

Notice that ??(Z i ) is the greedy direction that Z i moves toward?? L(??, ??)'s target distribution as showed in (13)(distribution that maximizesL(??, ??)).

This means ??(Z i ) is the gradient ofL(Z i , ??, ??):

Since our ultimate purpose is to update ??, by the chain rule,

??(Z i ) is given in equation(17).

In practice we update the policy-value parameter ?? by common policy gradient algorithm since:

and update representation parameter ?? by (18).

This section we verify that the information E-C process exists in deep RL with MINE and our framework accelerates this process.

Mutual information neural estimation(MINE) is an algorithm that can compute mutual information(MI) between two high dimensional random variables more accurately and efficiently.

Specifically, for random variables X and Z, assume T to be a function of X and Z, the calculation of I(X, Z) can be transformed to the following optimization problem:

The optimal function T ??? (X, Z) can be approximated by updating a neural network T (X, Z; ??).

With the aid of this powerful tool, we would like to visualize the mutual information between input state X and its relative representation Z: Every a few update steps, we sample a batch of inputs and their relevant representations

and compute their MI with MINE, every time we train MINE(update ??) we just shuffle {Z i } n i=1 and roughly assume the shuffled representations {Z

Figure (1) is the tensorboard graph of mutual information estimation between X and Z in Atari game Pong, x-axis is update steps and y-axis is MI estimation.

More details and results can be found in appendix(A.6) and (A.7).

As we can see, in both A2C with our framework and common A2C, the MI first increases to encode more information from inputs("remember" the inputs), then decreases to drop irrelevant information from inputs("forget" the useless information).

And clearly, our framework extracts faster and compresses faster than common A2C as showed in figure(1)(b) .

(a) MI in A2C (b) MI in A2C with our framework

After completing the visualization of MI with MINE, we analyze the relationship between our framework and MINE.

According to (Belghazi et al., 2018) , the optimal function T * in (20) goes as follows:

Combining the result with Theorem(2), we get:

Through this relationship, we theoretically derive an algorithm that can directly optimize our framework without constructing the lower bound, we put this derivation in the appendix(A.5).

In the experiments we show that our framework can improve the sample efficiency of basic RL algorithms(typically A2C and PPO).

Our anonymous code can be found in https://github.

com/AnonymousSubmittedCode/SVIB.

Other results can be found in last two appendices.

In A2C with our framework, we sample Z by a network ??(X, ??) where ?? ??? N (??; 0, 0.1) and the number of samples from each state X is 32, readers are encouraged to take more samples if the computation resources are sufficient.

We set the IB coefficient as ?? = 0.001.

We choose two prior distributions U (Z) of our framework, the first one is uniform distribution, apparently when U (Z) is the uniform distribution, ?????? log U (???) |??? =Z can be omitted.

The second one is a Gaussian distribution, which is defined as follows: for a given state X i , sample a batch of {Z

We also set ?? as 0.005????????? 1 ?? J(???; ??)/?????? log U (???)??? |??? =Z to control the magnitude of ?????? log U (???) |??? =Z .

Following (Liu et al., 2017) , the kernel function in (17) we used is the Gaussian RBF kernel K(Z i , Z j ) = exp(??????Z i ??? Z j ??? 2 /h) where h = med 2 /2 log(n + 1), med denotes the median of pairwise distances between the particles {Z

.

As for the hyper-parameters in RL, we simply choose the default parameters in A2C of Openaibaselines(https://github.com/openai/baselines/tree/master/baselines/a2c).

In summary, we implement the following four algorithms: A2C with uniform SVIB A2C with uniform SVIB A2C with uniform SVIB: Use ??(X, ??) as the embedding function, optimize by our framework(algorithm(A.4)) with U (Z) being uniform distribution.

A2C with Gaussian SVIB A2C with Gaussian SVIB A2C with Gaussian SVIB: Use ??(X, ??) as the embedding function, optimize by our framework(algorithm(A.4)) with U (Z) being Gaussian distribution.

A2C A2C A2C:Regular A2C in Openai-baselines with ??(X) as the embedding function.

A2C with noise A2C with noise A2C with noise(For fairness):A2C with the same embedding function ??(X, ??) as A2C with our framework.

Figure(2)(a)-(e) show the performance of four A2C-based algorithms in 5 gym Atari games.

We can see that A2C with our framework is more sample-efficient than both A2C and A2C with noise in nearly all 5 games.

Figure 2: (a)-(e) show the performance of four A2C-based algorithms, x-axis is time steps(2000 update steps for each time step) and y-axis is the average reward over 10 episodes, (f)-(h) show the performance of four PPO-based algorithms, x-axis is time steps(300 update steps for each time step).

We make exponential moving average of each game to smooth the curve(In PPO-Pong, we add 21 to all four curves in order to make exponential moving average).

We can see that our framework improves sample efficiency of basic A2C and PPO.

Notice that in SpaceInvaders, A2C with Gaussian SVIB is worse.

We suspect that this is because the agent excessively drops information from inputs that it misses some information related to the learning process.

There is a more detailed experimental discussion about this phenomena in appendix(A.7) .

We also implement four PPO-based algorithms whose experimental settings are same as A2C except that we set the number of samples as 26 for the sake of computation efficiency.

Results can be found in the in figure( 2)(f)-(h).

We study the information bottleneck principle in RL: We propose an optimization problem for learning the representation in RL based on the information-bottleneck framework and derive the optimal form of the target distribution.

We construct a lower bound and utilize Stein Variational gradient method to optimize it.

Finally, we verify that the information extraction and compression process also exists in deep RL, and our framework can accelerate this process.

We also theoretically derive an algorithm based on MINE that can directly optimize our framework and we plan to study it experimentally in the future work.

According to the assumption, naturally we have:

Notice that if we use our IB framework in value-based algorithm, then the objective function J ?? can be defined as:

where

and d ?? is the discounted future state distribution, readers can find detailed definition of d ?? in the appendix of (Chen et al., 2018) .

We can get:

We show the rigorous derivation of the target distribution in (11).

Denote P as the distribution of X, P Z ?? (Z) = P ?? (Z) as the distribution of Z. We use P ?? as the short hand notation for the conditional distribution P ?? (Z|X).

Moreover, we write

Take the functional derivative with respect to P ?? of the first term L 1 :

Hence, we can see that

Then we consider the second term.

By the chain rule of functional derivative, we have that

Combining the derivative of L 1 and L 2 and setting their summation to 0, we can get that

A.3 Proof of Theorem 2 (X, Z; ??) , given a fixed policy-value parameter ??, representation distribution P ?? (Z|X) and state distribution P (X), define a new representation distribution:

Proof Proof Proof.

Define I(X)

as:

According to the positivity of the KL-divergence, we have L(??,??) ??? L(??, ??).

Algorithm 1 Information-bottleneck-based state abstraction in RL ??, ?? ??? initialize network parameters ??, ?? ??? initialize hyper-parameters in (17) ?? ??? learning rate M ??? number of samples from

A.5 Integrate MINE to our framework MINE can also be applied to the problem of minimizing the MI between Z and X where Z is generated by a neural network P ?? (??|X):

A.6 Study the information-bottleneck perspective in RL Now we introduce the experimental settings of MI visualization.

And we show that the agent in RL usually tends to follow the information E-C process.

We compare the MI(I(X, Z)) between A2C and A2C with our framework.

Every 2000 update steps(2560 frames each step), we re-initialize the parameter ??, then sample a batch of inputs and their relevant representations

, n = 64, and compute the MI with MINE.

The learning rate of updating ?? is same as openai-baselines' A2C: 0.0007, training steps is 256 and the network architecture can be found in our code file "policy.py".

Figure( 3) is the MI visualization in game Qbert.

Note that there is a certain degree of fluctuations in the curve.

This is because that unlike supervised learning, the distribution of datasets and learning signals R ?? (X) keep changing in reinforcement learning: R ?? (X) changes with policy ?? and when ?? gets better, the agent might find new states, in this case, I(X, Z) might increase again because the agent needs to encode information from new states in order to learn a better policy.

Yet finally, the MI always tends to decrease.

Thus we can say that the agent in RL usually tends to follow the information E-C process.

(a) MI in A2C (b) MI in A2C with our framework Figure 3 : Mutual information visualization in Qbert.

As policy ?? gets better, the agent might find new states, in this case, I(X, Z) might increase again because the agent needs to encode information from new states in order to learn a better policy.

Yet finally, the MI always tends to decrease.

Thus it follows the information E-C process.

We argue that it's unnecessary to compute I(Z, Y ) like (Shwartz-Ziv & Tishby, 2017): According to (3), if the training loss continually decreases in supervised learning(Reward continually increases as showed in figure(2)(a) in reinforcement learning), I(Z, Y ) must increase gradually.

We also add some additional experimental results of MI visualization in the appendix(A.7).

This section we add some additional experimental results about our framework.

Notice that in game MsPacman, performance of A2C with our framework is worse than regular A2C.

According to the MI visualization of MsPacman in figure(5)(b), we suspect that this is because A2C with our framework drops the information from inputs so excessively that it misses some information relative to the learning process.

To see it accurately, in figure(5)(b), the orange curve, which denotes A2C with our framework, from step(x-axis) 80 to 100, suddenly drops plenty of information.

Meanwhile, in figure(4)(b), from step(xaxis) 80 to 100, the rewards of orange curve start to decrease.

As showed in figure(6), unlike Pong, Breakout, Qbert and some other shooting games, the frame of MsPacman contains much more information related to the reward: The walls, the ghosts and the tiny beans everywhere.

Thus if the agent drops information too fast, it may hurt the performance.

<|TLDR|>

@highlight

Derive an information bottleneck framework in reinforcement learning and some simple relevant theories and tools.