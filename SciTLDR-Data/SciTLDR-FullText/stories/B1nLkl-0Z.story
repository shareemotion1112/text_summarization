State-action value functions (i.e., Q-values) are ubiquitous in reinforcement learning (RL), giving rise to popular algorithms such as SARSA and Q-learning.

We propose a new notion of action value defined by a Gaussian smoothed version of the expected Q-value used in SARSA.

We show that such smoothed Q-values still satisfy a Bellman equation, making them naturally learnable from experience sampled from an environment.

Moreover, the gradients of expected reward with respect to the mean and covariance of a parameterized Gaussian policy can be recovered from the gradient and Hessian of the smoothed Q-value function.

Based on these relationships we develop new algorithms for training a Gaussian policy directly from a learned Q-value approximator.

The approach is also amenable to proximal optimization techniques by augmenting the objective with a penalty on KL-divergence from a previous policy.

We find that the ability to learn both a mean and covariance during training allows this approach to achieve strong results on standard continuous control benchmarks.

Model-free reinforcement learning algorithms often alternate between two concurrent but interacting processes: (1) policy evaluation, where an action value function (i.e., a Q-value) is updated to obtain a better estimate of the return associated with taking a specific action, and (2) policy improvement, where the policy is updated aiming to maximize the current value function.

In the past, different notions of Q-value have led to distinct but important families of RL methods.

For example, SARSA BID18 BID22 BID26 ) uses the expected Q-value, defined as the expected return of following the current policy.

Q-learning BID28 ) exploits a hard-max notion of Q-value, defined as the expected return of following an optimal policy.

Soft Q-learning BID7 and PCL BID14 both use a soft-max form of Q-value, defined as the future return of following an optimal entropy regularized policy.

Clearly, the choice of Q-value function has a considerable effect on the resulting algorithm; for example, restricting the types of policies that can be expressed, and determining the type of exploration that can be naturally applied.

In this work we introduce a new notion of action value: the smoothed action value functionQ ?? .

Unlike previous notions, which associate a value with a specific action at each state, the smoothed Qvalue associates a value with a specific distribution over actions.

In particular, the smoothed Q-value of a state-action pair (s, a) is defined as the expected return of first taking an action sampled from a normal distribution N (a, ??(s)), centered at a, then following actions sampled from the current policy thereafter.

In this way, the smoothed Q-value can also be interpreted as a Gaussian-smoothed or noisy version of the expected Q-value.

We show that smoothed Q-values possess a number of interesting properties that make them attractive for use in RL algorithms.

For one, the smoothed Q-values satisfy a single-step Bellman consistency, which allows bootstrapping to be used to train a function approximator.

Secondly, for Gaussian policies, the standard optimization objective (expected return) can be expressed in terms of smoothed Q-values.

Moreover, the gradient of this objective with respect to the mean and covariance of the Gaussian policy is equivalent to the gradient and the Hessian of the smoothed Q-value function, which allows one to derive updates to the policy parameters by having access to the derivatives of a sufficiently accurate smoothed Q-value function.

This observation leads us to propose an algorithm called Smoothie, which in the spirit of (Deep) Deterministic Policy Gradient (DDPG) BID21 BID11 , trains a policy using the derivatives of a trained (smoothed) Q-value function, thus avoiding the high-variance of stochastic updates used in standard policy gradient algorithms BID29 BID10 .

Unlike DDPG, which is well-known to have poor exploratory behavior BID7 , the approach we develop is able to utilize a non-deterministic Gaussian policy parameterized by both a mean and a covariance, thus allowing the policy to be exploratory by default and alleviating the need for excessive hyperparameter tuning.

Furthermore, we show that Smoothie can be easily adapted to incorporate proximal policy optimization techniques by augmenting the objective with a penalty on KL-divergence from a previous version of the policy.

The inclusion of a KL-penalty is not feasible in the standard DDPG algorithm, but we show that it is possible with our formulation, and it significantly improves stability and overall performance.

On standard continuous control benchmarks, our results are competitive with or exceed state-of-the-art, especially for more difficult tasks in the low-data regime.

We consider the standard model-free RL framework, where an agent interacts with a stochastic black-box environment by sequentially observing the state of the environment, emitting an action, and receiving a reward feedback; the goal is to find an agent that achieves maximal cumulative discounted reward.

This problem can be expressed in terms of a Markov decision process (MDP) that consists of a state space S and an action space A, where at iteration t the agent encounters a state s t ??? S and emits an action a t ??? A, after which the environment returns a scalar reward r t ??? R(s t , a t ) and places the agent in a new state s t+1 ??? P (s t , a t ).We model the behavior of the agent using a stochastic policy ?? that produces a distribution over feasible actions at each state s as ??(a | s).

The optimization objective (expected discounted return), as a function of the policy, can then be expressed in terms of the expected action value function Q ?? (s, a) by, DISPLAYFORM0 where ?? ?? (s) is the stationary distribution of the states under ??, and Q ?? (s, a) is recursively defined using the Bellman equation, DISPLAYFORM1 where ?? ??? [0, 1] is the discount factor.

For brevity, we will often suppress explicit denotation of the sampling distribution R over immediate rewards and the distribution P over state transitions.

The policy gradient theorem BID23 expresses the gradient of O ER (?? ?? ) w.r.t.

??, the tunable parameters of a policy ?? ?? , as, DISPLAYFORM2 Many reinforcement learning algorithms, including policy gradient and actor-critic variants, trade off variance and bias when estimating the random variable inside the expectation in (4); for example, by attempting to estimate Q ?? (s, a) accurately using function approximation.

In the simplest scenario, an unbiased estimate of Q ?? (s, a) is formed by accumulating discounted rewards from each state forward using a single Monte Carlo sample.

In this paper, we focus on multivariate Gaussian policies over continuous action spaces, A ??? R da .

We represent the observed state of the MDP as a d s -dimensional feature vector ??(s) ??? R ds , and parametrize the Gaussian policy by a mean and covariance function, respectively ??(s) : DISPLAYFORM3 These map the observed state of the environment to a Gaussian distribution, DISPLAYFORM4 where DISPLAYFORM5 Below we develop new RL training methods for this family of parametric policies, but some of the ideas presented may generalize to other families of policies as well.

We begin the formulation by reviewing some prior work on learning Gaussian policies.

BID21 present a new formulation of the policy gradient, called the deterministic policy gradient, for the family of Gaussian policies in the limit where the policy covariance approaches zero.

In such a scenario, the policy becomes deterministic because sampling from the policy always returns the Gaussian mean.

The key observation of BID21 is that under a deterministic policy ?? ??? (??, ?? ??? 0), one can estimate the expected future return from a state s as,

Then, one can express the gradient of the optimization objective (expected discounted return) for a parameterized ?? ?? ??? ?? ?? as, DISPLAYFORM0 This can be thought of as a characterization of the policy gradient theorem for deterministic policies.

In the limit of ?? ??? 0, one can also re-express the Bellman equation FORMULA1 as, DISPLAYFORM1 Therefore, a value function approximator Q ?? w can be optimized by minimizing the Bellman error, DISPLAYFORM2 for transitions (s, a, r, s ) sampled from a dataset D of interactions of the agent with the environment.

Algorithms like DDPG BID11 ) alternate between improving the value function by gradient descent on (9), and improving the policy based on (7).In practice, to gain better sample efficiency, BID5 and BID21 replace the on-policy state distribution ?? ?? (s) in (7) with an off-policy distribution ?? ?? (s) based on a replay buffer.

After this substitution, the policy gradient identity in (7) does not hold exactly, however, prior work finds that this works well in practice and improves sample efficiency.

We also adopt a similar approximation in our method to make use of off-policy data.

In this paper, we introduce smoothed action value functions, the gradients of which provide an effective signal for optimizing the parameters of a Gaussian policy.

Our notion of smoothed Qvalues, denotedQ ?? (s, a), differs from ordinary Q-values Q ?? (s, a) in that smoothed Q-values do not assume the first action of the agent is fully specified, but rather they assume that only the mean of the distribution of the first action is known.

Hence, to computeQ ?? (s, a), one has to perform an expectation of Q ?? (s,??) for actions?? drawn in the vicinity of a. More formally, smoothed action values are defined as, DISPLAYFORM0 With this definition ofQ ?? , one can re-express the expected reward objective for a Gaussian policy ?? ??? (??, ??) as, DISPLAYFORM1 The insight that differentiates this approach from prior work including BID8 ; BID4 is that instead of learning a function approximator for Q ?? (s, a) and then drawing samples to approximate the expectation in (10) and its derivative, we directly learn a function approximator forQ ?? (s, a).The key observation that enables direct bootstrapping of smoothed Q-values,Q ?? (s, a), is that their form allows a notion of Bellman consistency.

First, note that for Gaussian policies ?? ??? (??, ??) we have DISPLAYFORM2 Then, combining FORMULA0 and FORMULA0 , one can derive the following one-step Bellman equation for smoothed Q-values, DISPLAYFORM3 wherer ands are sampled from R(s,??) and P (s,??).

Below, we elaborate on how one can make use of the derivatives ofQ ?? to learn ?? and ??, and how the Bellman equation in (13) enables direct optimization ofQ ?? .

We parameterize a Gaussian policy ?? ??,?? ??? (?? ?? , ?? ?? ) in terms of two sets of parameters ?? and ?? for the mean and the covariance.

The gradient of the objective w.r.t.

mean parameters follows from the policy gradient theorem and is almost identical to (7) , DISPLAYFORM0 Estimating the derivative of the objective w.r.t.

covariance parameters is not as straightforward, sinc??? Q ?? is not a direct function of ??. However, a key observation of this work is that the second derivative ofQ ?? w.r.t.

actions is sufficient to exactly compute the derivative ofQ ?? w.r.t.

??, DISPLAYFORM1 A proof of this identity is provided in the Appendix.

The proof may be easily derived by expressing both sides of the equation using standard matrix calculus like DISPLAYFORM2 Then, the full derivative w.r.t.

?? takes the form, DISPLAYFORM3

We can think of two ways to optimizeQ 2 where?? ??? N (a, ??(s)), using several samples.

When the target values in these residuals are treated as fixed (i.e., using a target network), such a training procedure will achieve a fixed point whenQ ?? w (s, a) satisfies the recursion in the Bellman equation (10).The second approach requires a single function approximator forQ ?? w (s, a), resulting in a simpler implementation, and thus we use this approach in our experimental evaluation.

Suppose one has access to a tuple (s,??,r,s ) sampled from a replay buffer with knowledge of the sampling probability q(?? | s) (possibly unnormalized).

Then assuming that this sampling distribution has a full support, we draw a phantom action a ??? N (??, ??(s)) and optimizeQ ?? w (s, a) by minimizing a weighted Bellman error DISPLAYFORM0 2 .

For a specific pair of state and action (s, a) the expected value of the objective is, DISPLAYFORM1 Note that N (a|??, ??(s)) = N (??|a, ??(s)).

Therefore, when the target valuer + ??Q ?? w (s , ??(s )) is treated as fixed (e.g., when using target networks) this training procedure reaches an optimum whe?? Q ?? w (s, a) satisfies the recursion in the Bellman equation FORMULA0 .

In practice, we find that it is unnecessary to keep track of the probabilities q(?? | s), and assume the replay buffer provides a near-uniform distribution of actions conditioned on states.

Other recent work has also benefited from ignoring or heavily damping importance weights BID13 BID27 BID20 .

However, it is possible when interacting with the environment to save the probability of sampled actions along with their transitions, and thus have access to q(?? | s) ??? N (?? | ?? old (s), ?? old (s)).

Policy gradient algorithms are notoriously unstable, particularly in continuous control problems.

Such instability has motivated the development of trust region methods that attempt to mitigate the issue by constraining each gradient step to lie within a trust region BID19 , or augmenting the expected reward objective with a penalty on KL-divergence from a previous policy BID15 BID20 BID0 .

These stabilizing techniques have thus far not been applicable to algorithms like DDPG, since the policy is deterministic.

The formulation we propose in this paper, however, is easily amenable to trust region optimization.

Specifically, we may augment the objective (11) with a penalty DISPLAYFORM0 where ?? old ??? (?? old , ?? old ) is a previous parameterization of the policy.

The optimization is straightforward, since the KL-divergence of two Gaussians can be expressed analytically.

This paper follows a long line of work that uses Q-value functions to stably learn a policy, which in the past has been used to either approximate expected BID18 BID26 BID6 or optimal BID28 BID21 BID14 BID7 BID12 future value.

Work that is most similar to what we present are methods that exploit gradient information from the Q-value function to train a policy.

Deterministic policy gradient BID21 is perhaps the best known of these.

The method we propose can be interpreted as a generalization of the deterministic policy gradient.

Indeed, if one takes the limit of the policy covariance ??(s) as it goes to 0, the proposed Q-value function becomes the deterministic value function of DDPG, and the updates for training the Q-value approximator and the policy mean are identical.

Stochastic Value Gradient (SVG) BID8 ) also trains stochastic policies using an update that is similar to DDPG (i.e., SVG(0) with replay).

The key differences with our approach are that SVG does not provide an update for the covariance, and the mean update in SVG estimates the gradient with a noisy Monte Carlo sample, which we avoid by estimating the smoothed Q-value function.

Although a covariance update could be derived using the same reparameterization trick as in the mean update, that would also require a noisy Monte Carlo estimate.

Methods for updating the covariance along the gradient of expected reward are essential for applying the subsequent trust region and proximal policy techniques.

More recently, BID4 introduced expected policy gradients (EPG), a generalization of DDPG that provides updates for the mean and covariance of a stochastic Gaussian policy using gradients of an estimated Q-value function.

In that work, the expected Q-value used in standard policy gradient algorithms such as SARSA BID22 BID18 BID26 ) is estimated.

The updates in EPG therefore require approximating an integral of the expected Q-value function.

Our analogous process directly estimates an integral (via the smoothed Q-value function) and avoids approximate integrals, thereby making the updates simpler.

Moreover, while BID4 rely on a quadratic Taylor expansion of the estimated Q-value function, we instead rely on the strength of neural network function approximators to directly estimate the smoothed Q-value function.

The novel training scheme we propose for learning the covariance of a Gaussian policy relies on properties of Gaussian integrals BID2 BID16 .

Similar identities have been used in the past to derive updates for variational auto-encoders BID9 and Gaussian back-propagation BID17 .Finally, the perspective presented in this paper, where Q-values represent the averaged return of a distribution of actions rather than a single action, is distinct from recent advances in distributional RL BID1 .

Those approaches focus on the distribution of returns of a single action, whereas we consider the single average return of a distribution of actions.

Although we restrict our attention in this paper to Gaussian policies, an interesting topic for further investigation is to study the applicability of this new perspective to a wider class of policy distributions.

We utilize the insights from Section 3 to introduce a new RL algorithm, Smoothie.

Smoothie maintains a parameterizedQ ?? w trained via the procedure described in Section 3.2.

It then uses the gradient and Hessian of this approximation to train a Gaussian policy ?? ?? , ?? ?? using the updates stated in FORMULA0 and FORMULA0 .

See Algorithm 1 for a simplified pseudocode of our algorithm.

Input: Environment EN V , learning rates ?? ?? , ?? Q , discount factor ??, KL-penalty ??, batch size B, number of training steps N , target network lag ?? .Initialize ??, ??, w, set ?? = ??, ?? = ??, w = w. for i = 0 to N ??? 1 do // Collect experience Sample action a ??? N (?? ?? (s), ?? ?? (s)) and apply to EN V to yield r and s .

Insert transition (s, a, r, s ) to replay buffer.

DISPLAYFORM0 We perform a number of evaluations of Smoothie compared to DDPG.

We choose DDPG as a baseline because it (1) utilizes gradient information of a Q-value approximator, much like our algorithm; and (2) is a standard algorithm well-known to have achieve good, sample-efficient performance on continuous control benchmarks.

To evaluate Smoothie we begin with a simple synthetic task which allows us to study its behavior in a restricted setting.

We devised a simple single-action one-shot environment in which the reward function is a mixture of two Gaussians, one better than the other (see FIG1 ).

We initialize the policy mean to be centered on the worse of the two Gaussians.

We plot the learnable policy mean and standard deviation during training for Smoothie and DDPG in FIG1 (Left).

Smoothie learns both the mean and variance, while DDPG learns only the mean and the variance plotted is the exploratory noise, whose scale is kept fixed during training.

As expected we observe that DDPG cannot escape the local optimum.

At the beginning of training it exhibits some movement away from the local optimum (likely due to the initial noisy approximation given by Q ?? w ), it is unable to progress very far from the initial mean.

Note that this is not an issue of exploration.

The exploration scale is high enough that Q ?? w is aware of the better Gaussian.

The issue is in the update for ?? ?? , which is only with regard to the derivative of Q ?? w at the current mean.

On the other hand, we find Smoothie is successfully able to solve the task.

This is because the smoothed reward function approximated byQ ?? w has a derivative which clearly points ?? ?? towards the better Gaussian.

We also observe that Smoothie is able to suitably adjust the covariance ?? ?? during training.

Initially, ?? ?? decreases due to the concavity of the smoothed reward function.

As a region of convexity is entered, it begins to increase, before again decreasing to near-zero as ?? ?? approaches the global optimum.

The learnable policy mean and standard deviation during training for Smoothie and DDPG on a simple one-shot synthetic task.

The standard deviation for DDPG is the exploratory noise kept constant during training.

Right: The reward function for the synthetic task along with its Gaussian-smoothed version.

We find that Smoothie can successfully escape the lower-reward local optimum.

We also notice Smoothie increases and decreases its policy variance as the convexity/concavity of the smoothed reward function changes.

We now turn our attention to standard continuous control benchmarks available on OpenAI Gym BID3 utilizing the MuJoCo environment BID24 .Our implementations utilize feed forward neural networks for policy and Q-values.

We parameterize the covariance ?? ?? as a diagonal given by e ?? .

The exploration for DDPG is determined by an Ornstein-Uhlenbeck process BID25 BID11 .

Additional implementation details are provided in the Appendix.

Each plot shows the average reward and standard deviation clipped at the min and max of six randomly seeded runs after choosing best hyperparameters.

We see that Smoothie is competitive with DDPG even when DDPG uses a hyperparameter-tuned noise scale, and Smoothie learns the optimal noise scale (the covariance) during training.

Moreoever, we observe significant advantages in terms of final reward performance, especially in the more difficult tasks like Hopper, Walker2d, and Humanoid.

Across all tasks, TRPO is not sufficiently sampleefficient to provide a competitive baseline.

We compare the results of Smoothie and DDPG in FIG2 .

For each task we performed a hyperparameter search over actor learning rate, critic learning rate and reward scale, and plot the average of six runs for the best hyperparameters.

For DDPG we extended the hyperparameter search to also consider the scale and damping of exploratory noise provided by the Ornstein-Uhlenbeck process.

Smoothie, on the other hand, contains an additional hyperparameter to determine the weight on KL-penalty.

Despite DDPG having the advantage of its exploration decided by a hyperparameter search while Smoothie must learn its exploration without supervision, we find that Smoothie performs competitively or better across all tasks, exhibiting a slight advantage in Swimmer and Ant, while showing more dramatic improvements in Hopper, Walker2d, and Humanoid.

The improvement is especially dramatic for Hopper, where the average reward is doubled.

We also highlight the results for Humanoid, which as far as we know, are the best published results for a method that only trains on the order of millions of environment steps.

In contrast, TRPO, which to the best of our knowledge is the only other algorithm which can achieve better performance, requires on the order of tens of millions of environment steps to achieve comparable reward.

This gives added evidence to the benefits of using a learnable covariance and not restricting a policy to be deterministic.

Empirically, we found the introduction of a KL-penalty to improve performance of Smoothie, especially on harder tasks.

We present a comparison of results of Smoothie with and without the KL-penalty on the four harder tasks in FIG3 .

A KL-penalty to encourage stability is not possible in DDPG.

Thus, our algorithm provides a much needed solution to the inherent instability in DDPG training.

We observe benefits of using a proximal policy optimization method, especially in Hopper and Humanoid, where the performance improvement is significant without sacrificing sample efficiency.

We have presented a new Q-value function,Q ?? , that is a Gaussian-smoothed version of the standard expected Q-value, Q ?? .

The advantage of usingQ ?? over Q ?? is that its gradient and Hessian possess an intimate relationship with the gradient of expected reward with respect to mean and covariance of a Gaussian policy.

The resulting algorithm, Smoothie, is able to successfully learn both mean and covariance during training, leading to performance that can match or surpass that of DDPG, especially when incorporating a penalty on divergence from a previous policy.

The success ofQ ?? is encouraging.

Intuitively it may be argued that learningQ ?? is more sensible than learning Q ?? .

The smoothed Q-values by definition make the true reward surface smoother, thus possibly easier to learn; moreover the smoothed Q-values have a more direct relationship with the expected discounted return objective.

We encourage future work to further investigate these claims as well as techniques to apply the underlying motivations forQ ?? to other types of policies.

A PROOF OF EQUATION FORMULA0 We note that similar identities for Gaussian integrals exist in the literature BID16 BID17 and point the reader to these works for further information.

The specific identity we state may be derived using standard matrix calculus.

We make use of the fact that DISPLAYFORM0 and for symmetric A, ??? ???A ||v|| DISPLAYFORM1 We omit s from ??(s) in the following equations for succinctness.

The LHS of FORMULA0 Meanwhile, towards tackling the RHS of FORMULA0 we note that DISPLAYFORM2 Thus we have DISPLAYFORM3

@highlight

We propose a new Q-value function that enables better learning of Gaussian policies.