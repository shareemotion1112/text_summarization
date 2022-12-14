Temporal Difference learning with function approximation has been widely used recently and has led to several successful results.

However, compared with the original tabular-based methods, one major drawback of temporal difference learning with neural networks and other function approximators is that they tend to over-generalize across temporally successive states, resulting in slow convergence and even instability.

In this work, we propose a novel TD learning method, Hadamard product Regularized TD (HR-TD), that reduces over-generalization and thus leads to faster convergence.

This approach can be easily applied to both linear and nonlinear function approximators.

HR-TD is evaluated on several linear and nonlinear benchmark domains, where we show improvement in learning behavior and performance.

Temporal Difference (TD) learning is one of the most important paradigms in Reinforcement Learning BID15 .

Techniques based on combining TD learning with nonlinear function approximators and stochastic gradient descent, such as deep networks, have led to significant breakthroughs in large-scale problems to which these methods can be applied BID8 BID12 .At its heart, the TD learning update is straightforward.

v(s) estimates the value of being in a state s.

After an action a that transitions the agent from s to next state s , v(s) is altered to be closer to the (discounted) estimated value of s , v(s ) (plus any received reward, r).

The difference between these estimated values is called the temporal difference error (TD error) and is typically denoted as δ.

Formally, δ = r + γv(s ) − v(s), where γ is the discount factor, and r + γv(s ) is known as the TD target.

When states are represented individually (the tabular case), v(s) can be altered independently from v(s ) using the update rule v(s) ← v(s) + αδ, where α is the learning rate.

In fully deterministic environments, α can be set to 1, thus causing v(s) to change all the way to the TD target.

Otherwise, in a stochastic environment, α is set less than 1 so that v(s) only moves part of the way towards the TD target, thus avoiding over-generalization from a single example.

When, on the other hand, states are represented with a function approximator, as is necessary in large or continuous environments, v(s) can no longer be updated independently from v(s ).

That is because s and s are likely to be similar (assuming actions have local effects), any change to v(s) is likely to also alter v(s ).

While such generalization is desirable in principle, it also has the unintended consequence of changing the TD target, which in turn can cause the TD update to lead to an increase in the TD error between s and s .

This unintended consequence can be seen as a second form of over-generalization: one that can be much more difficult to avoid.

Past work has identified this form of over-generalization in RL, has observed that it is particularly relevant in methods that use neural network function approximators such as DQN BID8 , and has proposed initial solutions BID4 BID10 .

In this paper, we present a deeper analysis of the reasons for this form of over-generalization and introduce a novel learning algorithm termed HR-TD, based on the recursive proximal mapping formulation of TD learning BID1 , which offers a mathematical framework for parameter regularization that allows one to control for this form of over-generalization.

Empirical results across multiple domains demonstrate that our novel algorithm learns more efficiently (from fewer samples) than prior approaches.

The rest of the paper is organized as follows.

Section 2 offers a brief background on TD learning, the over-generalization problem, and optimization techniques used in the derivation of our algorithm.

In Section 3, we discuss the state-of-the-art research in this direction.

The motivation and the design of our algorithm are presented in Section 4.

Finally, the experimental results of Section 5 validate the effectiveness of the proposed algorithm.

This section builds on the notation introduced in Section 1 to specify the problem of interest in full detail.

We introduce the background for TD learning, over-generalization, and proximal mapping, which are instrumental in the problem formulation and algorithm design.

Reinforcement Learning problems are generally defined as Markov Decision Processes (MDPs).

We use the definition and notation as used in BID16 , unless otherwise specified.

In this paper, we focus on domains with large or continuous state spaces such that function approximation is needed.

We define the value estimate of state s with parameter θ when following policy π as, DISPLAYFORM0 Here R t is the random variable associated with a reward at time t, and r t is used as an instantiation of this random variable.

The optimal (true) value function v * π satisfies the Bellman equation given as v * DISPLAYFORM1 .

During TD learning, the estimated value function is altered to try to make this property hold.

In effect, state values are updated by bootstrapping off of the estimated value of the predicted next states.

We focus on 1-step TD methods, i.e., TD(0), that bootstrap from the value of the immediate next state or states in the MDP to learn the value of the current state.

The TD error δ t (s t , s t+1 |θ) to be minimized is as follows:

DISPLAYFORM2 In the following, δ t (s t , s t+1 |θ) is written as δ t for short.

When using function approximation and gradient descent to optimize the parameters, the loss to be minimized is the squared TD error.

At the t-th time-step, the objective function used in TD learning is DISPLAYFORM3 Similarly, the optimal action value function Q satisfies the Bellman optimality equation DISPLAYFORM4 The partial derivative of v(s t |θ) or Q(s t , a t |θ) with respect to θ is the direction in which TD learning methods update the parameters.

We use g t (s t |θ) and g t (s t , a t |θ) to refer to these vectors.

In the linear case, v(s t |θ) = θ t φ(s t ), where φ(s t ) are the features of state s t .

In this case, g t (s t , a t |θ) is the feature vector φ(s t , a t ), and in general, g t (s t , a t |θ) = ∂ θ Q(s t , a t |θ).

It is computed as: DISPLAYFORM5 We have already briefly alluded to the issue of over-generalization in Section 1.

One of the reasons we use function approximation is that we want the values we learn to generalize to similar states.

But one of these similar states is likely to be the target of our Bellman equation v(s t+1 |θ).

If the weights that correspond to large or important features in φ(s t+1 ) are strengthened, then the TD error might not decrease as much as it should, or it might even increase.

We refer to parameter updates that work against the objective of reducing the TD error as over-generalization.

In this section, we introduce the basics of proximal mapping, which provide the mathematical formulation of our algorithm design.

A proximal mapping BID9 prox f (w) associated with a convex function f is defined as prox f (w) = arg min DISPLAYFORM0 (1) Such a proximal mapping is typically used after a parameter update step to incorporate constraints on the parameters.

Intuitively, the first term f (x) provides incentive to move x in the direction that minimizes f , whereas the second term DISPLAYFORM1 2 provides pressure to keep x close to w. If f (x) = 0, then prox f (w) = w, the identity function.

f can often be a regularization term to help incorporate prior knowledge.

For example, for learning sparse representations, the case of f (x) = β x 1 is particularly important.

In this case, the entry-wise proximal operator is: DISPLAYFORM2 Proximal methods have been shown to be useful for various reinforcement learning problems, e.g., proximal gradient TD learning BID6 integrates the proximal method with gradient TD learning BID14 ) using the Legendre-Fenchel convex conjugate function BID2 , and projected natural actor-critic BID17 interprets natural gradient as a special case of proximal mapping.

We now introduce the recursive proximal mapping formulation of TD learning algorithm BID1 .

At the t-th iteration, the TD update law solves a recursive proximal mapping, i.e., θ t+1 = θ t + α t δ t g t (s t ), which is equivalent to DISPLAYFORM3 It should be noted that Eq. FORMULA9 is different from Eq. (1) in that Eq. (1) has an explicit objective function f to optimize.

Eq. FORMULA9 does not have an explicit objective function, but rather corresponds to a fixed-point equation.

In fact, it has been proven that the TD update term δ t g t (s t ) does not optimize any objective function BID7 .

Discussing this in details goes beyond the scope of the paper, and we refer interested readers to BID7 BID1 for a comprehensive discussion of this topic.

To the best of our knowledge, the closest work to ours to address the over-generalization problem is the Temporal Consistency loss (TC-loss) method BID10 and the constrained TD approach BID4 .The TC-loss BID10 aims to minimize the change to the target state by minimizing explicitly a separate loss that measures change in the value of DISPLAYFORM0 When used in conjunction with a TD loss, it guarantees that the updated estimates adhere to the Bellman operator and thus are temporally consistent.

However, there are some drawbacks to this method.

Firstly, the asymptotic solution of the TC-loss method is different from the TD solution due to the two separate losses, and the solution property remains unclear.

Secondly, each parameter component plays a different role in changing v(s ).

For instance, if the component of θ is or close to 0, then this component does not have any impact on changing v(s ).

Different parameter components, therefore, should be treated differently according to their impact on the value function changes (or action-value change in case of DQN).Another recent work in this direction is the constrained TD (CTD) algorithm BID4 .

To avoid the over-generalization among similar sates, CTD tends to alleviate overgeneralization by using the vector rejection technique to diminish the update along the direction of the gradient of the action-value function of the successive state.

In other words, the real update is made to be orthogonal to the gradient of the next state.

However, the CTD method suffers from the double-sampling problem, which is explained in detail in Appendix A. Moreover, since it mainly uses vector rejection, this method is not straightforward to extend to nonlinear function approximation, such as the DQN network, where over-generalization can be severe.

Lastly, if the state representation of s t and s t+1 are highly similar, as in case of visual environments like Atari games, then the vector rejection causes the update to be almost orthogonal to the computed gradient.

In this section, we analyze the reason for over-generalization and propose a novel algorithm to mitigate it.

Consider the update to the parameter θ t as follows, with TD error δ t , learning rate α and a linear function approximation v(s t |θ t ) with features φ(s t ) and gradient g(s t |θ t ) = φ(s t ).

DISPLAYFORM0 If we substitute the above value for θ t+1 , the TD error for the same transition after the update is DISPLAYFORM1 and thus DISPLAYFORM2 We see above that the decrease in the TD error at t depends on two factors, the inner product of the gradient with features of s t , and its inner product with the features of s t+1 .

This decrease will be reduced if φ(s t ) and φ(s t+1 ) have a large inner product.

If this inner product exceeds 1 γ φ(s t ) φ(s t ), then in fact the error increases.

Thus over-generalization is an effect of a large positive correlation between the update and the features of s t+1 , especially when contrasted with the correlation of this same update with the features of s t .We are then left with the following question: what kind of weight update can maximize the reduction in δ t ?

Merely minimizing the correlation of the update with φ(s t+1 ) is insufficient, as it might lead to minimizing the correlation with φ(s t ).

This is the issue that Constrained TD BID4 faces with its gradient projection approach.

Hence, we must also maximize its correlation with φ(s t ).To examine this effect, we consider the properties of parameters that we should avoid changing, to the extent possible.

Consider the linear value function approximation case: v θ (s) = φ(s) θ.

For example, consider s t and s t+1 with the features φ(s t ) = [0, 2, 1], and φ(s t+1 ) = [2, 0, 1].

Then for two different weights, θ 1 = [0, 0, 2] and θ 2 = [1, 1, 0], we have the same value for both these parameter vectors at both s t and s t+1 , i.e. φ(s t ) θ 1 = φ(s t+1 ) θ 1 = φ(s t ) θ 2 = φ(s t+1 ) θ 2 = 2.

However, the results of the Hadamard product (•) of these parameters with the feature vectors are different, i.e. DISPLAYFORM3 where the Hadamard products of θ 1 with φ(s t ) and φ(s t+1 ) are more correlated than those of θ 2 .

An update to the last weight of θ 1 will cause the values of both s t and s t+1 to change, but an update to the second weight of θ 2 will affect only s t .

In fact, unless both the first and the second weights change, s t and s t+1 do not change simultaneously.

In this sense, θ 1 tends to cause aggressive generalization across the values of s t and s t+1 , and thus the TD update to θ 1 should be regularized more heavily.

The Hadamard product of the weights and the features allows us to distinguish between θ 1 and θ 2 in this way.

Motivated by this observation, we aim to reduce the over-generalization by controlling the weighted feature correlation between the current state g(s)•θ and the successive state g(s )•θ, i.e., Corr(g(s)• θ, g(s ) • θ).

Given the constraint as shown above, the constrained Mean-Squares Error (MSE) is formulated as DISPLAYFORM0

Require: T , α t (learning rate), γ(discount factor), η(initial regularization parameter).

Ensure: Initialize θ 0 . for t = 1, 2, 3, · · · , T do η t = η/t Update θ t+1 according to Eq. (5).

end for where V is the true value function.

Using the recursive proximal mapping with respect to the constrained objective function, the parameter update per-step of Eq. (3) can be written as DISPLAYFORM0 Using Lagrangian duality, it can be reformulated as DISPLAYFORM1 where η is the factor that weights the constraint against the objective.

The closed-form solution to the weight update is DISPLAYFORM2 Using sample-based estimation, i.e., using g t (s) (resp.

g t (s )) to estimate g(s) (resp.

g(s )) , and using δ t to estimate E[δ t ], the Eq. (4) becomes DISPLAYFORM3 In the proposed algorithm, if the component of the weights helps decrease the Hadamard product correlation, then it is not penalized.

Now the algorithm for value function approximation is formulated as in Algorithm 1, and the algorithm for control is formulated in Algorithm 2.

In DQN, the value function is learned by minimizing the following squared Bellman error using SGD and backpropagating the gradients through the parameter θ DISPLAYFORM0 Here, θ are the parameter of the target network that is periodically updated to match the parameters being trained.

The action a t+1 is chosen as arg max a Q(s t+1 , a|θ ) if we use DQN, and arg max a Q(s t+1 , a|θ) if we use Double DQN (DDQN) BID18 .

We use DDQN in experiments as DQN has been shown to over-estimate the target value.

Let φ(s t |θ) be the activations of the last hidden layer before the Q-value calculation and θ −1 be the corresponding weights for this layer.

The Correlation can be written as DISPLAYFORM1 We do not use the target network when calculating this loss.

The loss used in Hadamard regularized DDQN is then an η-weighted mixture of Eq. (6) and this loss DISPLAYFORM2 4.4 THEORETICAL ANALYSISIn this section, we conduct some initial analysis of Algorithm 1 with linear function approximation.

For simplicity, we only discuss the linear case, i.e., ∂v θ (s t ) = φ(s t ), ∂v θ (s t+1 ) = φ(s t+1 ).

If Algorithm 1 converges, the update of the weights according to Eq. (5) should satisfy the following condition DISPLAYFORM3 DISPLAYFORM4 If we set η → γ, we observe that the second and third terms in the RHS above cancel out in the diagonal element.

Consider the scheme where we initialize η = γ and then reduce it as over the training process.

It is equivalent to slowly introducing the discount factor into the error computation.

It has been shown BID11 ) that instead of the discount factor γ provided by the MDP, a user-defined time-varying γ t can help accelerate the learning process of the original MDP w.r.t γ.

This previous work suggests using a small discount factor γ t < γ in the beginning, and then increasing γ t gradually to γ.

HR-TD results in a similar effect without defining a separate γ t and its schedule.

We evaluate HR-TD on two classical control problems: Mountain Car and Acrobot using both linear function approximation with Fourier basis features and nonlinear function approximation using Deep Neural Networks.

We verify that this algorithm scales to complex domains such as the Atari Learning Environment BID0 , by evaluating our approach on the game of Pong.

We utilize OpenAI gym BID3 to interface our agent with the environments.

We compare HR-TD to the baselines by using the following metrics: 1) Accumulated reward per episode.2) Average change in the target Q value at s after every parameter update.

For comparison, we consider Q learning and Q learning with TC loss (and DDQN for neural networks).Based on our analysis, we expect HR-Q learning to begin improving the policy earlier in the learning process, and we expect HR-TD to be able to evaluate the value of a policy just as well as TD.

We evaluate the change of the value of the next state as well, and consider whether HR-TD is able to reduce this change as a consequence of the regularization.

We note, however, that this quantity is diagnostic in nature, rather than being the true objective.

It would definitely be possible to minimize this quantity by making no learning updates whatsoever, but then we would also observe no learning.

Before we consider the effect of HR-Q on control tasks, we compare the purely evaluative property of HR-TD.

Here, we evaluate a trained policy on the Mountain Car domain.

We run this experiment

Require: T , α t (learning rate), γ(discount factor), η(initial regularization parameter).

Ensure: Initialize θ 0 .

repeat η t = η/t Choose a t using policy derived from Q (e.g., -greedy) Take a t , observe r t , s t+1 Add s t , a t , r t , s t+1 to Replay Buffer Sample batch from Buffer and Update θ t+1 using backpropagation to minimize Eq. (7).

t ← t + 1 until training done FIG0 shows the cumulative score in an episode on the y-axis, with the episode number depicted on the x-axis.

1b compares how much the value of the TD target changed after an update.

The x-axis is number of iterations 10 times for each method.

For each experiment, the policy is executed in the environment for 10000 steps, resetting the agent to one of the start states if it terminates.

We learn the value function using TD by sampling a batch of transitions from this dataset and take 10,000 learning steps per run.

The metric we compare is the MSE with the Monte Carlo estimate of the same run, taken over 300,000 transitions.

The MSE value for each experiment is calculated by averaging the MSE of the last 10 training steps, to reduce sampling error.

Finally, we take the mean of these errors across the 10 runs and present them in Table 1 .

TD and HR-TD reach roughly the same value for all the runs.

TC, however, converges to a different minimum that leads to a very high MSE.

This may be because the competing TD and TC objectives in this method cause the learning to destabilize.

If we lower the learning rate for TC, then we avoid this behavior but also do not converge in the given max number of training steps.

We now consider the performance of HR-Q learning when using Neural Networks for function approximation.

We consider two domains, Mountain Car and Acrobot, but we do not perform any basis expansion and feed the state values directly into a neural network with a single hidden layer of 64 units.

We compare the performance of HR-Q in FIG0 and 2, with Q-Learning and Q-learning with TC loss.

We use DDQN Van Hasselt et al. (2016) as the underlying algorithm for Q-learning.

Details of We take 20 independent runs, with a different seed in each run used to initialize Tensorflow, NumPy, and the OpenAI Gym environment.

Each run is taken over 1000 episodes.

In both these experiments, we see HR-TD starts to learn a useful policy behavior before either of the other techniques.

Interesting to note is that in FIG0 , HR-TD learns a state representation that causes the target value to change less than DQN but does not restrict it as much as TC.

But in FIG1 we see that HR-TD is able to find a representation that is better at keeping the target value separate than TC is.

However, in both these cases, the value function that is learned seems to be quickly useful for learning a better policy.

We also validate the applicability of this technique to a more complex domain and a more complex network.

We apply the HR-Q to DDQN on the Atari domain to verify that the technique is scalable and that the findings and trends we see in the first two experiments carry forward to this challenging task.

We use the network architecture specified in BID8 , and the hyper-parameters for TC as specified in BID10 .

Experimental details are specified in Appendix B. From the results, we see that HR-TD does not interfere with learning on the complex network, and does about as well as DDQN.

Finally, we study HR-TD with the linear function approximation, we look at the Mountain Car domain.

We expand the feature space using Fourier basis functions BID5 .

All methods are trained with an order 6 Fourier basis expansion for Mountain Car BID5 , which leads to 36 features for Mountain Car.

We use a constant learning rate α = 0.01 for all three methods.

For HR-TD we initialize the regularization factor η = 0.3.

Each episode is run until we receive an episode termination signal from the Gym wrapper, which is a maximum of 200 steps if the goal is not reached.

We show the learning curves for 1000 episodes, averaged over 20 independent runs.

In FIG3 , we see that HR-Q and TC perform better than Q-learning.

HR-Q also shows a more stable updates (changes value of next state less) than Q learning, and comparable to Q-learning with the added TC loss over the course of training.

In this paper, we analyze the problem of over-generalization in TD learning with function approximation.

This analysis points to the potential pitfalls of over-generalization in TD-learning.

Based on the analysis, we propose a novel regularization scheme based on the Hadamard product.

We also show that with the right weight on the regularization, the solution of this method is the same as that of TD.

Finally, we experimentally validate the effectiveness of our algorithm on benchmarks of varying complexity.

A PROBLEM WITH CTD: DOUBLE SAMPLING PROBLEM Double sampling comes into effect whenever we need the product of two expectations.

If an expression contains 3 expectations we will need three independent samples.

Below we will first write out why residual gradients have a double sampling problem and why TD doesn't.

Then we shall show why CTD has this problem, and might actually suffer from a triple sampling problem.

Note that the double-sampling problem only exists in stochastic MDP problems.

In a Deterministic MDP, double sampling will not be an issue.

In the constrained TD update, the first term is the regular TD update, which has no double-sampling issues.

However, the second term, − Target update 500 500 500 η --0.03 Target update 500 500 500 η --0.01

<|TLDR|>

@highlight

A regularization technique for TD learning that avoids temporal over-generalization, especially in Deep Networks

@highlight

A variation on temporal difference learning for the function approximation case that attempts to resolve the issue of over-generalization across temporally-successive states.

@highlight

The paper introduces HR-TD, a variation of the TD(0) algorithm, meant to improve the over-generalization problem in conventional TD