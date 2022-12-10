Recent studies have shown the vulnerability of reinforcement learning (RL) models in noisy settings.

The sources of noises differ across scenarios.

For instance, in practice, the observed reward channel is often subject to noise (e.g., when observed rewards are collected through sensors), and thus observed rewards may not be credible as a result.

Also, in applications such as robotics, a deep reinforcement learning (DRL) algorithm can be manipulated to produce arbitrary errors.

In this paper, we consider noisy RL problems where observed rewards by RL agents are generated with a reward confusion matrix.

We call such observed rewards as perturbed rewards.

We develop an unbiased reward estimator aided robust RL framework that enables RL agents to learn in noisy environments while observing only perturbed rewards.

Our framework draws upon approaches for supervised learning with noisy data.

The core ideas of our solution include estimating a reward confusion matrix and defining a set of unbiased surrogate rewards.

We prove the convergence and sample complexity of our approach.

Extensive experiments on different DRL platforms show that policies based on our estimated surrogate reward can achieve higher expected rewards, and converge faster than existing baselines.

For instance, the state-of-the-art PPO algorithm is able to obtain 67.5% and 46.7% improvements in average on five Atari games, when the error rates are 10% and 30% respectively.

Designing a suitable reward function plays a critical role in building reinforcement learning models for real-world applications.

Ideally, one would want to customize reward functions to achieve application-specific goals (Hadfield-Menell et al., 2017) .

In practice, however, it is difficult to design a function that produces credible rewards in the presence of noise.

This is because the output from any reward function is subject to multiple kinds of randomness:• Inherent Noise.

For instance, sensors on a robot will be affected by physical conditions such as temperature and lighting, and therefore will report back noisy observed rewards.• Application-Specific Noise.

In machine teaching tasks BID13 Loftin et al., 2014) , when an RL agent receives feedback/instructions from people, different human instructors might provide drastically different feedback due to their personal styles and capabilities.

This way the RL agent (machine) will obtain reward with bias.• Adversarial Noise.

Adversarial perturbation has been widely explored in different learning tasks and shows strong attack power against different machine learning models.

For instance, Huang et al. (2017) has shown that by adding adversarial perturbation to each frame of the game, they can mislead RL policies arbitrarily.

Assuming an arbitrary noise model makes solving this noisy RL problem extremely challenging.

Instead, we focus on a specific noisy reward model which we call perturbed rewards, where the observed rewards by RL agents are generated according to a reward confusion matrix.

This is not a very restrictive setting to start with, even considering that the noise could be adversarial: Given that arbitrary pixel value manipulation attack in RL is not very practical, adversaries in the real-world have high incentives to inject adversarial perturbation to the reward value by slightly modifying it.

For instance, adversaries can manipulate sensors via reversing the reward value.

In this paper, we develop an unbiased reward estimator aided robust framework that enables an RL agent to learn in a noisy environment with observing only perturbed rewards.

Our solution framework builds on existing reinforcement learning algorithms, including the recently developed DRL ones (Q-Learning BID19 BID18 , Cross-Entropy Method (CEM) BID11 , Deep SARSA BID10 , Deep Q-Network (DQN) (Mnih et al., 2013; BID6 , Dueling DQN (DDQN) BID17 , Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2015) , Continuous DQN (NAF) (Gu et al., 2016) and Proximal Policy Optimization (PPO) BID4

The main challenge is that the observed rewards are likely to be biased, and in RL or DRL the accumulated errors could amplify the reward estimation error over time.

We do not require any assumption on knowing the true distribution of reward or adversarial strategies, other than the fact that the generation of noises follow an unknown reward confusion matrix.

Instead, we address the issue of estimating the reward confusion matrices by proposing an efficient and flexible estimation module.

Everitt et al. (2017) provided preliminary studies for the noisy reward problem and gave some general negative results.

The authors proved a No Free Lunch theorem, which is, without any assumption about what the reward corruption is, all agents can be misled.

Our results do not contradict with the results therein, as we consider a specific noise generation model (that leads to a set of perturbed rewards).

We analyze the convergence and sample complexity for the policy trained based on our proposed method using surrogate rewards in RL, using Q-Learning as an example.

We conduct extensive experiments on OpenAI Gym (Brockman et al., 2016 ) (AirRaid, Alien, Carnival, MsPacman, Pong, Phoenix, Seaquest) and show that the proposed reward robust RL method achieves comparable performance with the policy trained using the true rewards.

In some cases, our method even achieves higher cumulative reward -this is surprising to us at first, but we conjecture that the inserted noise together with our noisy-removal unbiased estimator adds another layer of exploration, which proves to be beneficial in some settings.

This merits a future study.

Our contributions are summarized as follows: FORMULA2 We adapt and generalize the idea of defining a simple but effective unbiased estimator for true rewards using observed and perturbed rewards to the reinforcement learning setting.

The proposed estimator helps guarantee the convergence to the optimal policy even when the RL agents only have noisy observations of the rewards.

(2) We analyze the convergence to the optimal policy and finite sample complexity of our reward robust RL methods, using Q-Learning as the running example.

(3) Extensive experiments on OpenAI Gym show that our proposed algorithms perform robustly even at high noise rates.

Robust Reinforcement Learning It is known that RL algorithms are vulnerable to noisy environments (Irpan, 2018) .

Recent studies (Huang et al., 2017; Kos & Song, 2017; Lin et al., 2017) show that learned RL policies can be easily misled with small perturbations in observations.

The presence of noise is very common in real-world environments, especially in robotics-relevant applications.

Consequently, robust (adversarial) reinforcement learning (RRL/RARL) algorithms have been widely studied, aiming to train a robust policy which is capable of withstanding perturbed observations BID12 Pinto et al., 2017; Gu et al., 2018) or transferring to unseen environments BID1 Fu et al., 2017) .

However, these robust RL algorithms mainly focus on noisy vision observations, instead of the observed rewards.

A couple of recent works (Lim et al., 2016; BID3 have also looked into a rather parallel question of training robust RL algorithms with uncertainty in models.

Learning with Noisy Data Learning appropriately with biased data has received quite a bit of attention in recent machine learning studies Natarajan et al. (2013) ; BID7 BID6 ; BID9 BID16 ; Menon et al. (2015) .

The idea of above line of works is to define unbiased surrogate loss function to recover the true loss using the knowledge of the noises.

We adapt these approaches to reinforcement learning.

Though intuitively the idea should apply in our RL settings, our work is the first one to formally establish this extension both theoretically and empirically.

Our quantitative understandings will provide practical insights when implementing reinforcement learning algorithms in noisy environments.

In this section, we define our problem of learning from perturbed rewards in reinforcement learning.

Throughout this paper, we will use perturbed reward and noisy reward interchangeably, as each time step of our sequential decision making setting is similar to the "learning with noisy data" setting in supervised learning (Natarajan et al., 2013; BID7 BID6 BID9 .

In what follows, we formulate our Markov Decision Process (MDP) problem and the reinforcement learning (RL) problem with perturbed (noisy) rewards.

Our RL agent interacts with an unknown environment and attempts to maximize the total of his collected reward.

The environment is formalized as a Markov Decision Process (MDP), denoting as M = S, A, R, P, γ .

At each time t, the agent in state s t ∈ S takes an action a t ∈ A, which returns a reward r(s t , a t , s t+1 ) ∈ R (which we will also shorthand as r t ), and leads to the next state s t+1 ∈ S according to a transition probability kernel P, which encodes the probability Pa(st, st+1).

Commonly P is unknown to the agent.

The agent's goal is to learn the optimal policy, a conditional distribution π(a|s) that maximizes the state's value function.

The value function calculates the cumulative reward the agent is expected to receive given he would follow the current policy π after observing the current state DISPLAYFORM0 where 0 ≤ γ ≤ 1 1 is a discount factor.

Intuitively, the agent evaluates how preferable each state is given the current policy.

From the Bellman Equation, the optimal value function is given by V * (s) = maxa∈A s t+1 ∈S Pa(st, st+1) [rt + γV * (st+1)] .

It is a standard practice for RL algorithms to learn a state-action value function, also called the Q-function.

Q-function denotes the expected cumulative reward if agent chooses a in the current state and follows π thereafter: DISPLAYFORM1

In many practical settings, our RL agent does not observe the reward feedback perfectly.

We consider the following MDP with perturbed reward, denoting asM = S, A, R, C, P, γ 2 : instead of observing r t ∈ R at each time t directly (following his action), our RL agent only observes a perturbed version of r t , denoting asr t ∈R. For most of our presentations, we focus on the cases where R,R are finite sets; but our results generalize to the continuous reward settings.

The generation ofr follows a certain function C : S × R →R. To let our presentation stay focused, we consider the following simple state-independent 3 flipping error rates model: if the rewards are binary (consider r + and r − ),r(s t , a t , s t+1 ) (r t ) can be characterized by the following noise rate parameters e + , e − : e+ = P(r(st, at, st+1) = r−|r(st, at, st+1) = r+), e− = P(r(st, at, st+1) = r+|r(st, at, st+1) = r−).

When the signal levels are beyond binary, suppose there are M outcomes in total, denoting as [R 0 , R 1 , · · · , R M −1 ].r t will be generated according to the following confusion matrix C M ×M where each entry c j,k indicates the flipping probability for generating a perturbed outcome: c j,k = P(r t = R k |r t = R j ).

Again we'd like to note that we focus on settings with finite reward levels for most of our paper, but we provide discussions in Section 3.1 on how to handle continuous rewards with discretizations.

In the paper, we do not assume knowing the noise rates (i.e., the reward confusion matrices), which is different from the assumption of knowing them as adopted in many supervised learning works Natarajan et al. (2013) .

Instead we will estimate the confusion matrices (Section 3.3).

In this section, we first introduce an unbiased estimator for binary rewards in our reinforcement learning setting when the error rates are known.

This idea is inspired by Natarajan et al. (2013) , but we will extend the method to the multi-outcome, as well as the continuous reward settings.

With the knowledge of noise rates (reward confusion matrices), we are able to establish an unbiased approximation of the true reward in a similar way as done in Natarajan et al. (2013) .

We will call such a constructed unbiased reward as a surrogate reward.

To give an intuition, we start with replicating the results for binary reward R = {r − , r + } in our RL setting: Lemma 1.

Let r be bounded.

Then, if we define, r(st, at, st+1) :=(1−e − )·r + −e + ·r − 1−e + −e − (r(st, at, st+1) = r+) DISPLAYFORM0 we have for any r(s t , a t , s t+1 ), Er |r [r(s t , a t , s t+1 )] = r(s t , a t , s t+1 ).In the standard supervised learning setting, the above property guarantees convergence -as more training data are collected, the empirical surrogate risk converges to its expectation, which is the same as the expectation of the true risk (due to unbiased estimators).

This is also the intuition why we would like to replace the reward terms with surrogate rewards in our RL algorithms.

The above idea can be generalized to the multi-outcome setting in a fairly straight-forward way.

DefineR := [r(r = R 0 ),r(r = R 1 ), ...,r(r = R M −1 )], wherer(r = R m ) denotes the value of the surrogate reward when the observed reward is DISPLAYFORM1 DISPLAYFORM2 we have for any r(s t , a t , s t+1 ), Er |r [r(s t , a t , s t+1 )] = r(s t , a t , s t+1 ).Continuous reward When the reward signal is continuous, we discretize it into M intervals and view each interval as a reward level, with its value approximated by its middle point.

With increasing M , this quantization error can be made arbitrarily small.

Our method is then the same as the solution for the multi-outcome setting, except for replacing rewards with discretized ones.

Note that the finerdegree quantization we take, the smaller the quantization error -but we would suffer from learning a bigger reward confusion matrix.

This is a trade-off question that can be addressed empirically.

So far we have assumed knowing the confusion matrices, but we will address this additional estimation issue in Section 3.3, and present our complete algorithm therein.

We now analyze the convergence and sample complexity of our surrogate reward based RL algorithms (with assuming knowing C), taking Q-Learning as an example.

Convergence guarantee First, the convergence guarantee is stated in the following theorem:Theorem 1.

Given a finite MDP, denoting asM = S, A,R, P, γ , the Q-learning algorithm with surrogate rewards, given by the update rule, DISPLAYFORM0 converges w.p.1 to the optimal Q-function as long as t α t = ∞ and t α 2 t < ∞.Note that the term on the right hand of Eqn.

(3) includes surrogate rewardr estimated using Eqn.

FORMULA2 and Eqn.

(2).

Theorem 1 states that that agents will converge to the optimal policy w.p.1 with replacing the rewards with surrogate rewards, despite of the noises in observing rewards.

This result is not surprising -though the surrogate rewards introduce larger variance, we are grateful of their unbiasedness, which grants us the convergence.

In other words, the addition of the perturbed reward does not destroy the convergence guarantees of Q-Learning.

Sample complexity To establish our sample complexity results, we first introduce a generative model following previous literature (Kearns & Singh, 1998; 2000; Kearns et al., 1999) .

This is a practical MDP setting to simplify the analysis.

Definition 1.

A generative model G(M) for an MDP M is a sampling model which takes a stateaction pair (s t , a t ) as input, and outputs the corresponding reward r(s t , a t ) and the next state s t+1 randomly with the probability of P a (s t , s t+1 ), i.e., s t+1 ∼ P(·|s, a).Exact value iteration is impractical if the agents follow the generative models above exactly (Kakade, 2003) .

Consequently, we introduce a phased Q-Learning which is similar to the ones presented in Kakade (2003) ; Kearns & Singh (1998) for the convenience of proving our sample complexity results.

We briefly outline phased Q-Learning as follows -the complete description (Algorithm 2) can be found in Appendix A. Definition 2.

Phased Q-Learning algorithm takes m samples per phase by calling generative model G(M).

It uses the collected m samples to estimate the transition probability P and update the estimated value function per phase.

Calling generative model G(M) means that surrogate rewards are returned and used to update value function per phase.

The sample complexity of Phased Q-Learning is given as follows: DISPLAYFORM1 be bounded reward, C be an invertible reward confusion matrix with det(C) denoting its determinant.

For an appropriate choice of m, the Phased Q-Learning algorithm calls the generative model DISPLAYFORM2 times in T epochs, and returns a policy such that for all state s ∈ S, DISPLAYFORM3 Theorem 2 states that, to guarantee the convergence to the optimal policy, the number of samples needed is no more than O(1/det(C)2 ) times of the one needed when the RL agent observes true rewards perfectly.

This additional constant is the price we pay for the noise presented in our learning environment.

When the noise level is high, we expect to see a much higher 1/det(C)2 ; otherwise when we are in a low-noise regime , Q-Learning can be very efficient with surrogate reward (Kearns & Singh, 2000) .

Note that Theorem 2 gives the upper bound in discounted MDP setting; for undiscounted setting (γ = 1), the upper bound is at the order of O .

Lower bound result is omitted due to the lack of space.

The idea of constructing MDP in which learning is difficult and the algorithm must make |S||A|T log 1 δ calls to G(M), is similar to Kakade (2003) .While the surrogate reward guarantees the unbiasedness, we sacrifice the variance at each of our learning steps, and this in turn delays the convergence (as also evidenced in the sample complexity bound).

It can be verified that the variance of surrogate reward is bounded when C is invertible, and it is always higher than the variance of true reward.

This is summarized in the following theorem: Theorem 3.

Let r ∈ [0, R max ] be bounded reward and confusion matrix C is invertible.

Then, the variance of surrogate rewardr is bounded as follows: DISPLAYFORM4 To give an intuition of the bound, when we have binary reward, the variance for surrogate reward bounds as follows: DISPLAYFORM5 (1−e+−e−) 2 .

As e − + e + → 1, the variance becomes unbounded and the proposed estimator is no longer effective, nor will it be well-defined.

In practice, there is a trade-off question between bias and variance by tuning a linear combination of R andR, i.e., R proxy = ηR + (1 − η)R, and choosing an appropriate η ∈ [0, 1].

In Section 3.1 we have assumed the knowledge of reward confusion matrices, in order to compute the surrogate reward.

This knowledge is often not available in practice.

Estimating these confusion matrices is challenging without knowing any ground truth reward information; but we'd like to note that efficient algorithms have been developed to estimate the confusion matrices in supervised learning settings BID0 Liu & Liu, 2017; Khetan et al., 2017; Hendrycks et al., 2018) .

The idea in these algorithms is to dynamically refine the error rates based on aggregated rewards.

Note this approach is not different from the inference methods in aggregating crowdsourcing labels, as referred in the literature (Dawid & Skene, 1979; Karger et al., 2011; Liu et al., 2012) .

We adapt this idea to our reinforcement learning setting, which is detailed as follows.

At each training step, the RL agent collects the noisy reward and the current state-action pair.

Then, for each pair in S × A, the agent predicts the true reward based on accumulated historical observations of reward for the corresponding state-action pair via, e.g., averaging (majority voting).

Finally, with the predicted true reward and the accuracy (error rate) for each state-action pair, the estimated reward confusion matricesC are given bỹ DISPLAYFORM0 where in above # [·] denotes the number of state-action pair that satisfies the condition [·] in the set of observed rewardsR(s, a) (see Algorithm 1 and 3);r(s, a) andr(s, a) denote predicted true rewards (using majority voting) and observed rewards when the state-action pair is (s, a).

The above procedure of updatingc i,j continues indefinitely as more observation arrives.

Algorithm 1 Reward Robust RL (sketch) DISPLAYFORM1 Initialize value function Q(s, a) arbitrarily.

while Q is not converged do Initialize state s ∈ S while s is not terminal do Choose a from s using policy derived from Q Take action a, observe s and noisy rewardr if collecting enoughr for every S × A pair then Get predicted true rewardr using majority voting Estimate confusion matrixC based onr andr (Eqn.

4) Obtain surrogate rewardṙ DISPLAYFORM2 Our final definition of surrogate reward replaces a known reward confusion C in Eqn.

FORMULA4 with our estimated oneC. We denote this estimated surrogate reward asṙ.

We present (Reward Robust RL) in Algorithm 1 4 .

Note that the algorithm is rather generic, and we can plug in any exisitng RL algorithm into our reward robust one, with only changes in replacing the rewards with our estimated surrogate rewards.

In this section, reward robust RL is tested in different games, with different noise settings.

Due to space limit, more experimental results can be found in Appendix D.

Environments and RL Algorithms To fully test the performance under different environments, we evaluate the proposed robust reward RL method on two classic control games (CartPole, Pendulum) and seven Atari 2600 games (AirRaid, Alien, Carnival, MsPacman, Pong, Phoenix, Seaquest), which encompass a large variety of environments, as well as rewards.

Specifically, the rewards could be unary (CartPole), binary (most of Atari games), multivariate (Pong) and even continuous (Pendulum).

A set of state-of-the-art reinforcement learning algorithms are experimented with while training under different amounts of noise (See TAB4 5 .

For each game and algorithm, three policies are trained based on different random initialization to decrease the variance.

Reward Post-Processing For each game and RL algorithm, we test the performances for learning with true rewards, learning with noisy rewards and learning with surrogate rewards.

Both symmetric and asymmetric noise settings with different noise levels are tested.

For symmetric noise, the confusion matrices are symmetric.

As for asymmetric noise, two types of random noise are tested: 1) rand-one, each reward level can only be perturbed into another reward; 2) rand-all, each reward could be perturbed to any other reward, via adding a random noise matrix.

To measure the amount of noise w.r.t confusion matrices, we define the weight of noise ω in Appendix B.2.

The larger ω is, the higher the noise rates are.

CartPole The goal in CartPole is to prevent the pole from falling by controlling the cart's direction and velocity.

The reward is +1 for every step taken, including the termination step.

When the cart or pole deviates too much or the episode length is longer than 200, the episode terminates.

Due to the unary reward {+1} in CartPole, a corrupted reward −1 is added as the unexpected error (e − = 0).

As a result, the reward space R is extended to {+1, −1}. Five algorithms Q-Learning (1992), CEM (2006) , SARSA (1998) , DQN (2016) and DDQN (2016) are evaluated.

In FIG1 , we show that our estimator successfully produces meaningful surrogate rewards that adapt the underlying RL algorithms to the noisy settings, without any assumption of the true distribution of rewards.

With the noise rate increasing (from 0.1 to 0.9), the models with noisy rewards converge slower due to larger biases.

However, we observe that the models always converge to the best score 200 with the help of surrogate rewards.

DISPLAYFORM0 In some circumstances (slight noise -see FIG8 , 6b, 6c, 6d), the surrogate rewards even lead to faster convergence.

This points out an interesting observation: learning with surrogate reward even outperforms the case with observing the true reward.

We conjecture that the way of adding noise and then removing the bias introduces implicit exploration.

This implies that for settings even with true reward, we might consider manually adding noise and then remove it in expectation.

Pendulum The goal in Pendulum is to keep a frictionless pendulum standing up.

Different from the CartPole setting, the rewards in pendulum are continuous: r ∈ (−16.28, 0.0].

The closer the reward is to zero, the better performance the model achieves.

Following our extension (see Section 3.1), the (−17, 0] is firstly discretized into 17 intervals: (−17, −16], (−16, −15], · · · , (−1, 0], with its value approximated using its maximum point.

After the quantization step, the surrogate rewards can be estimated using multi-outcome extensions presented in Section 3.1.

We experiment two popular algorithms, DDPG (2015) and NAF (2016) in this game.

In FIG2 , both algorithms perform well with surrogate rewards under different amounts of noise.

In most cases, the biases were corrected in the long-term, even when the amount of noise is extensive (e.g., ω = 0.7).

The quantitative scores on CartPole and Pendulum are given in TAB1 , where the TAB2 .

Our reward robust RL method is able to achieve consistently good scores.

Atari We validate our algorithm on seven Atari 2600 games using the state-of-the-art algorithm PPO BID4 .

The games are chosen to cover a variety of environments.

The rewards in the Atari games are clipped into {−1, 0, 1}. We leave the detailed settings to Appendix B. Results for PPO on Pong-v4 in symmetric noise setting are presented in FIG3 .

Due to limited space, more results on other Atari games and noise settings are given in Appendix D.3.

Similar to previous results, our surrogate estimator performs consistently well and helps PPO converge to the optimal policy.

TAB2 shows the average scores of PPO on five selected Atari games with different amounts of noise (symmetric & asymmetric).

In particular, when the noise rates e + = e − > 0.3, agents with surrogate rewards obtain significant amounts of improvements in average scores.

We do not present the results for the case with unknown C because the state-space (image-input) is very large for Atari games, which is difficult to handle with the solution given in Section 3.3.

DISPLAYFORM1

Only an underwhelming amount of reinforcement learning studies have focused on the settings with perturbed and noisy rewards, despite the fact that such noises are common when exploring a realworld scenario, that faces sensor errors or adversarial examples.

We adapt the ideas from supervised Er |r (r) = Pr |r (r =r − )r − + Pr |r (r =r + )r + .When r = r + , from the definition in Lemma 1:Pr |r (r =r − ) = e + , Pr |r (r =r + ) = 1 − e + .

Taking the definition of surrogate rewards Eqn.

FORMULA2 DISPLAYFORM0 Similarly, when r = r − , it also verifies Er |r [r(s t , a t , s t+1 )] = r(s t , a t , s t+1 ).Proof of Lemma 2.

The idea of constructing unbiased estimator is easily adapted to multi-outcome reward settings via writing out the conditions for the unbiasedness property (s.t.

Er |r [r] = r.).

For simplicity, we shorthandr(r = R i ) asR i in the following proofs.

Similar to Lemma 1, we need to solve the following set of functions to obtainr: DISPLAYFORM1 whereR i denotes the value of the surrogate reward when the observed reward is R i .

Define R := [R 0 ; R 1 ; · · · ; R M −1 ], andR := [R 0 ,R 1 , ...,R M −1 ], then the above equations are equivalent to: R = C ·R. If the confusion matrix C is invertible, we obtain the surrogate reward: DISPLAYFORM2 According to above definition, for any true reward level R i , i = 0, 1, · · · , M − 1, we have DISPLAYFORM3 Furthermore, the probabilities for observing surrogate rewards can be written as follows: DISPLAYFORM4 wherep i = j p j c j,i , andp i , p i represent the probabilities of occurrence for surrogate rewardR i and true reward R i respectively.

Corollary 1.

Letp i and p i denote the probabilities of occurrence for surrogate rewardr(r = R i ) and true reward R i .

Then the surrogate reward satisfies, DISPLAYFORM5 Proof of Corollary 1.

From Lemma 2, we have, DISPLAYFORM6 Consequently, DISPLAYFORM7 To establish Theorem 1, we need an auxiliary result (Lemma 3) from stochastic process approximation, which is widely adopted for the convergence proof for Q-Learning (Jaakkola et al., 1993; BID14 .

Lemma 3.

The random process {∆ t } taking values in R n and defined as DISPLAYFORM8 converges to zero w.p.1 under the following assumptions: DISPLAYFORM9 Here F t = {∆ t , ∆ t−1 , · · · , F t−1 · · · , α t , · · · } stands for the past at step t, α t (x) is allowed to depend on the past insofar as the above conditions remain valid.

The notation || · || W refers to some weighted maximum norm.

Proof of Lemma 3.

See previous literature (Jaakkola et al., 1993; BID14 .Proof of Theorem 1.

For simplicity, we abbreviate s t , s t+1 , Q t , Q t+1 , r t ,r t and α t as s, s , Q, Q , r,r, and α, respectively.

Subtracting from both sides the quantity Q * (s, a) in Eqn.

(3): DISPLAYFORM10 In consequence, DISPLAYFORM11 Finally, DISPLAYFORM12 Becauser is bounded, it can be clearly verified that DISPLAYFORM13 for some constant C. Then, due to the Lemma 3, ∆ t converges to zero w.p.1, i.e., Q (s, a) converges to Q * (s, a).The procedure of Phased Q-Learning is described as Algorithm 2: DISPLAYFORM14 DISPLAYFORM15 Note thatP here is the estimated transition probability, which is different from P in Eqn.

FORMULA22 .To obtain the sample complexity results, the range of our surrogate reward needs to be known.

Assuming reward r is bounded in [0, R max ], Lemma 4 below states that the surrogate reward is also bounded, when the confusion matrices are invertible:Lemma 4.

Let r ∈ [0, R max ] be bounded, where R max is a constant; suppose C M ×M , the confusion matrix, is invertible with its determinant denoting as det(C).

Then the surrogate reward satisfies DISPLAYFORM16 Proof of Lemma 4.

From Eqn.

FORMULA4 , we have, DISPLAYFORM17 where adj(C) is the adjugate matrix of C; det(C) is the determinant of C. It is known from linear algebra that, DISPLAYFORM18 where M ji is the determinant of the (M − 1) × (M − 1) matrix that results from deleting row j and column i of C. Therefore, M ji is also bounded: DISPLAYFORM19 where the sum is computed over all permutations σ of the set {0, 1, · · · , M − 2}; c is the element of M ji ; sgn(σ) returns a value that is +1 whenever the reordering given by σ can be achieved by successively interchanging two entries an even number of times, and −1 whenever it can not.

Consequently, DISPLAYFORM20 Proof of Theorem 2.

From Hoeffding's inequality, we obtain: DISPLAYFORM21 In the same way,r t is bounded by M det(C) · R max from Lemma 4.

We then have, DISPLAYFORM22 Further, due to the unbiasedness of surrogate rewards, we have st+1∈S P a (s t , s t+1 )r t = st+1∈S;rt∈R P a (s t , s t+1 ,r t )r t .As a result, DISPLAYFORM23 In the same way, DISPLAYFORM24 Recursing the two equations in two directions (0 → T ), we get DISPLAYFORM25 Combining these two inequalities above we have: DISPLAYFORM26 For arbitrarily small , by choosing m appropriately, there always exists 1 = 2 =(1−γ) 2(1+γ) such that the policy error is bounded within .

That is to say, the Phased Q-Learning algorithm can converge to the near optimal policy within finite steps using our proposed surrogate rewards.

Finally, there are |S||A|T transitions under which these conditions must hold, where | · | represent the number of elements in a specific set.

Using a union bound, the probability of failure in any condition is smaller than DISPLAYFORM27 We set the error rate less than δ, and m should satisfy that DISPLAYFORM28 In consequence, after m|S||A|T calls, which is, O DISPLAYFORM29 , the value function converges to the optimal one for every state s, with probability greater than 1 − δ.

The above bound is for discounted MDP setting with 0 ≤ γ < 1.

For undiscounted setting γ = 1, since the total error (for entire trajectory of T time-steps) has to be bounded by , therefore, the error for each time step has to be bounded by T .

Repeating our anayslis, we obtain the following upper bound: DISPLAYFORM30 Proof of Theorem 3.

DISPLAYFORM31 Using the CauchySchwarz inequality, DISPLAYFORM32 So we get, Var(r) − Var(r) ≥ 0.

In addition, DISPLAYFORM33

We set up our experiments within the popular OpenAI baselines BID4 and kerasrl (Plappert, 2016) framework.

Specifically, we integrate the algorithms and interact with OpenAI Gym (Brockman et al., 2016) environments TAB4 .

A set of state-of-the-art reinforcement learning algorithms are experimented with while training under different amounts of noise, including Q-Learning BID19 BID18 , Cross-Entropy Method (CEM) BID11 , Deep SARSA BID10 , Deep Q-Network (DQN) (Mnih et al., 2013; BID6 , Dueling DQN (DDQN) BID17 , Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2015) , Continuous DQN (NAF) (Gu et al., 2016) and Proximal Policy Optimization (PPO) BID4 algorithms.

For each game and algorithm, three policies are trained based on different random initialization to decrease the variance in experiments.

We explore both symmetric and asymmetric noise of different noise levels.

For symmetric noise, the confusion matrices are symmetric, which means the probabilities of corruption for each reward choice are equivalent.

For instance, a confusion matrix DISPLAYFORM0 says that r 1 could be corrupted into r 2 with a probability of 0.2 and so does r 2 (weight = 0.2).As for asymmetric noise, two types of random noise are tested: 1) rand-one, each reward level can only be perturbed into another reward; 2) rand-all, each reward could be perturbed to any other reward.

To measure the amount of noise w.r.t confusion matrices, we define the weight of noise as follows: DISPLAYFORM1 , where ω controls the weight of noise; I and N denote the identity and noise matrix respectively.

Suppose there are M outcomes for true rewards, N writes as: DISPLAYFORM2 where for each row i, 1) rand-one: randomly choose j, s.t n i,j = 1 and n i,k = 0 if k = j; 2) randall: generate M random numbers that sum to 1, i.e., j n i,j = 1.

For the simplicity, for symmetric noise, we choose N as an anti-identity matrix.

As a result, c i,j = 0, if i = j or i + j = M .

To obtain an intuitive view of the reward perturbation model, where the observed rewards are generated based on a reward confusion matrix, we constructed a simple MDP and evaluated the performance of robust reward Q-Learning (Algorithm 1) on different noise ratios (both symmetric and asymmetric).

The finite MDP is formulated as FIG6 : when the agent reaches state 5, it gets an instant reward of r + = 1, otherwise a zero reward r − = 0.

During the explorations, the rewards are perturbed according to the confusion matrix C 2×2 = [1 − e − , e − ; e + , 1 − e + ].

There are two experiments conducted in this setting: 1) performance of Q-Learning under different noise rates TAB5 ; 2) robustness of estimation module in time-variant noise ( FIG6 ).

As shown in TAB5 , Q-Learning achieved better results consistently with the guidance of surrogate rewards and the confusion matrix estimation algorithm.

For time-variant noise, we generated varying amount of noise at different training stages: 1) e − = 0.1, e + = 0.3 (0 to 1e 4 steps); 2) e − = 0.2, e + = 0.1 (1e 4 to 3e 4 steps); 3) e − = 0.3, e + = 0.2 (3e 4 to 5e 4 steps); 4) e − = 0.1, e + = 0.2 (5e 4 to 7e 4 steps).

In FIG6 , we show that Algorithm 1 is robust against time-variant noise, which dynamically adjusts the estimatedC after the noise distribution changes.

Note that we set a maximum memory size for collected noisy rewards to let the agents only learn with recent observations.

CartPole and Pendulum The policies use the default network from keras-rl framework.

which is a five-layer fully connected network 6 .

There are three hidden layers, each of which has 16 units and followed by a rectified nonlinearity.

The last output layer is activated by the linear function.

For CartPole, We trained the models using Adam optimizer with the learning rate of 1e −3 for 10,000 steps.

The exploration strategy is Boltzmann policy.

For DQN and Dueling-DQN, the update rate of target model and the memory size are 1e −2 and 50, 000.

For Pendulum, We trained DDPG and NAF using Adam optimizer with the learning rate of 5e −4 for 150, 000 steps.

the update rate of target model and the memory size are 1e −3 and 100, 000.Atari Games We adopt the pre-processing steps as well as the network architecture from Mnih et al. (2015) .

Specifically, the input to the network is 84×84×4, which is a concatenation of the last 4 frames and converted into 84 × 84 gray-scale.

The network comprises three convolutional layers and two fully connected layers 7 .

The kernel size of three convolutional layer are 8 × 8 with stride 4 (32 filters), 4 × 4 with stride 2 (64 filters) and 3 × 3 with stride 1 (64 filters), respectively.

Each hidden layer is followed by a rectified nonlinearity.

Except for Pong where we train the policies for 3e 7 steps, all the games are trained for 5e 7 steps with the learning rate of 3e −4 .

Note that the rewards in the Atari games are discrete and clipped into {−1, 0, 1}. Except for Pong game, in which r = −1 means missing the ball hit by the adversary, the agents in other games attempt to get higher scores in the episode with binary rewards 0 and 1.

C.1 REWARD ROBUST RL ALGORITHMS As stated in Section 3.3, the confusion matrix can be estimated dynamically based on the aggregated answers, similar to previous literature in supervised learning (Khetan et al., 2017) .

To get a concrete view, we take Q-Learning for an example, and the algorithm is called Reward Robust Q-Learning (Algorithm 3).

Note that is can be extended to other RL algorithms by plugging confusion matrix estimation steps and the computed surrogate rewards, as shown in the experiments FIG8 ).

Get predicted true rewardr(s, a) using majority voting in everyR(s, a) Estimate confusion matrixC based onr(s, a) andr(s, a) (Eqn.

FORMULA11 ) Empty all the sets of observed rewardsR(s, a) Obtain surrogate rewardṙ(s, a) using DISPLAYFORM0

In Algorithm 3, the predicted true rewardr(s, a) is derived from majority voting in collected noisy setsR(s, a) for every state-action pair (s, a) ∈ S × A, which is a simple but efficient way of leveraging the expectation of aggregated rewards without assumptions on prior distribution of noise.

In the following, we adopt standard Expectation-Maximization (EM) idea in the our estimation framework (arguably a simple version of it), inspired by previous works BID20 .Assuming the observed noisy rewards are independent conditional on the true reward, we can compute the posterior probability of true reward from the Bayes' theorem: DISPLAYFORM0 where P(r = R j ) is the prior of true rewards, and P(r = R k |r = R j ) is estimated by current estimated confusion matrixC: P(r = R k |r = R j ) =c j,i .

Note that the inference should be conducted for each state-action pair (s, a) ∈ S × A in every iteration, i.e., P(r(s, a) = R i |r(s, a, 1) = R 1 , · · · ,r(s, a, n) = R n ), abbreviated as P(r(s, a) = R i ), which requires relatively greater computation costs compared to the majority voting policy.

It also points out an interesting direction to check online EM algorithms for our perturbed-RL problem.

After the inference steps in Eqn. (8), the confusion matrixC is then updated based on the posterior probabilities:c DISPLAYFORM1 where P(r(s, a) = R i ) denotes the inference probabilities of true rewards based on collected noisy rewards setsR(s, a).

To utilize EM algorithms in the robust reward algorithms (e.g., Algorithm 3), we need to replace Eqn.

(4) by Eqn.

(9) for the estimation of reward confusion matrix.

In previous sections, to let our presentation stay focused, we consider the state-independent perturbed reward environments, which share the same confusion matrix for all states.

In other words, the noise for different states is generated within the same distribution.

More generally, the generation ofr follows a certain function C : S × R →R, where different states may correspond to varied noise distributions (also varied confusion matrices).

However, our algorithm is still applicable except for maintaining different confusion matrices C s for different states.

It is worthy to notice that Theorem 1 holds because the surrogate rewards produce an unbiased estimation of true rewards for each state, i.e., Er |r,st [r(s t , a t , s t+1 )] = r(s t , a t , s t+1 ).

Furthermore, Theorem 2 and 3 can be revised as:Theorem 4. (Upper bound) Let r ∈ [0, R max ] be bounded reward, C s be invertible reward confusion matrices with det(C s ) denoting its determinant.

For an appropriate choice of m, the Phased Q-Learning algorithm calls the generative model DISPLAYFORM0 times in T epochs, and returns a policy such that for all state s ∈ S, |V π (s) − V * (s)| ≤ , > 0, w.p.

≥ 1 − δ, 0 < δ < 1.Theorem 5.

Let r ∈ [0, R max ] be bounded reward and all confusion matrices C s are invertible.

Then, the variance of surrogate rewardr is bounded as follows: DISPLAYFORM1

As illustrated in Theorem 3, our surrogate rewards introduce larger variance while conducting unbiased estimation which are likely to decrease the stability of RL algorithms.

Apart from the linear combination idea (appropriate trade-off), some variance reduction techniques in statistics (e.g., correlated sampling) can also be applied into our surrogate rewards.

Specially, BID2 proposed to a reward estimator to compensate for stochastic corrupted reward signals.

It is worthy to notice that their method is designed for variance reduction under stochastic (zero-mean) noise, which is no longer efficacious in more general perturbed-reward setting.

However, it is potential to integrate their method with our robust-reward RL framework because surrogate rewards guarantee unbiasedness in reward expectation.

To verify this idea, we repeated the experiments of Cartpole in Section 4.2 but included variance reduction step for estimated surrogate rewards.

Following BID2 , we adopted sample mean as a simple approximator during the training and set sequence length as 100.

As shown in Figure 5, the models with only variance reduction technique (red lines) suffer from huge biases when the noise is large, and cannot converge to the optimal policies like those under noisy rewards.

Nevertheless, they benefits from variance reduction for surrogate rewards (purple lines), which achieve faster convergence or better performance in many cases (e.g., Figure 5a (ω = 0.7), 5b (ω = 0.3)).It is also not surprising that the integrated algorithm (purple lines) outperforms better as the noise rate increases (indicating larger variance from Theorem 3, e.g., ω = 0.9).

Similarly, TAB6 provides quantitative results which show that our surrogate benefits from variance reduction techniques ("ours + VRT"), especially when the noise rate is large.

DISPLAYFORM0 Figure 5: Learning curves from five reward robust RL algorithms (see Algorithm 3) on CartPole game with true rewards (r) , noisy rewards (r) (η = 1) , sample-mean noisy rewards (η = 1) , estimated surrogate rewards (ṙ) and sample-mean estimated surrogate rewards .

Note that confusion matrices C are unknown to the agents here.

From top to the bottom, the noise rates are 0.1, 0.3, 0.7 and 0.9.

Here we repeated each experiment 10 times with different random seeds and plotted 10% to 90% percentile area with its mean highlighted.

To validate the effectiveness of robust reward algorithms (like Algorithm 3), where the noise rates are unknown to the agents, we conduct extensive experiments in CartPole.

It is worthwhile to notice that the noisy rates are unknown in the explorations of RL agents.

Besides, we discretize the Figure 6 provides learning curves from five algorithms with different kinds of rewards.

The proposed estimation algorithms successfully obtain the approximate confusion matrices, and are robust in the unknown noise environments.

From FIG9 , we can observe that the estimation of confusion matrices converges very fast.

The results are inspiring because we don't assume any additional knowledge about noise or true reward distribution in the implementation.

and estimated surrogate rewards (ṙ) .

Note that confusion matrices C are unknown to the agents here.

From top to the bottom, the noise rates are 0.1, 0.3, 0.7 and 0.9.

Here we repeated each experiment 10 times with different random seeds and plotted 10% to 90% percentile area with its mean highlighted.

@highlight

A new approach for learning with noisy rewards in reinforcement learning