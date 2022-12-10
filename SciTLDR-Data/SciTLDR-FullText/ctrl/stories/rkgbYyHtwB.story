We present a simple and effective algorithm designed to address the covariate shift problem in imitation learning.

It operates by training an ensemble of policies on the expert demonstration data, and using the variance of their predictions as a cost which is minimized with RL together with a supervised behavioral cloning cost.

Unlike adversarial imitation methods, it uses a fixed reward function which is easy to optimize.

We prove a regret bound for the algorithm in the tabular setting which is linear in the time horizon multiplied by a coefficient which we show to be low for certain problems in which behavioral cloning fails.

We evaluate our algorithm empirically across multiple pixel-based Atari environments and continuous control tasks, and show that it matches or significantly outperforms behavioral cloning and generative adversarial imitation learning.

Training artificial agents to perform complex tasks is essential for many applications in robotics, video games and dialogue.

If success on the task can be accurately described using a reward or cost function, reinforcement learning (RL) methods offer an approach to learning policies which has been shown to be successful in a wide variety of applications (Mnih et al., 2015; Hessel et al., 2018) However, in other cases the desired behavior may only be roughly specified and it is unclear how to design a reward function to characterize it.

For example, training a video game agent to adopt more human-like behavior using RL would require designing a reward function which characterizes behaviors as more or less human-like, which is difficult.

Imitation learning (IL) offers an elegant approach whereby agents are trained to mimic the demonstrations of an expert rather than optimizing a reward function.

Its simplest form consists of training a policy to predict the expert's actions from states in the demonstration data using supervised learning.

While appealingly simple, this approach suffers from the fact that the distribution over states observed at execution time can differ from the distribution observed during training.

Minor errors which initially produce small deviations from the expert trajectories become magnified as the policy encounters states further and further from its training distribution.

This phenomenon, initially noted in the early work of (Pomerleau, 1989) , was formalized in the work of (Ross & Bagnell, 2010) who proved a quadratic O( T 2 ) bound on the regret and showed that this bound is tight.

The subsequent work of (Ross et al., 2011) showed that if the policy is allowed to further interact with the environment and make queries to the expert policy, it is possible to obtain a linear bound on the regret.

However, the ability to query an expert can often be a strong assumption.

In this work, we propose a new and simple algorithm called DRIL (Disagreement-Regularized Imitation Learning) to address the covariate shift problem in imitation learning, in the setting where the agent is allowed to interact with its environment.

Importantly, the algorithm does not require any additional interaction with the expert.

It operates by training an ensemble of policies on the demonstration data, and using the disagreement in their predictions as a cost which is optimized through RL together with a supervised behavioral cloning cost.

The motivation is that the policies in the ensemble will tend to agree on the set of states covered by the expert, leading to low cost, but are more likely to disagree on states not covered by the expert, leading to high cost.

The RL cost thus pushes the agent back towards the distribution of the expert, while the supervised cost ensures that it mimics the expert within the expert's distribution.

Our theoretical results show that, subject to realizability and optimization oracle assumptions, our algorithm obtains a O( κ T ) regret bound for tabular MDPs, where κ is a measure which quantifies a tradeoff between the concentration of the demonstration data and the diversity of the ensemble outside the demonstration data.

We evaluate DRIL empirically across multiple pixel-based Atari environments and continuous control tasks, and show that it matches or significantly outperforms behavioral cloning and generative adversarial imitation learning, often recovering expert performance with only a few trajectories.

Denote by S the state space, A the action space, and Π the class of policies the learner is considering.

Let T denote the task horizon and π the expert policy whose behavior the learner is trying to mimic.

For any policy π, let d π denote the distribution over states induced by following π.

Denote C(s, a) the expected immediate cost of performing action a in state s, which we assume is bounded in [0, 1] .

In the imitation learning setting, we do not necessarily know the true costs C(s, a), instead we observe expert demonstrations.

Our goal is to find a policy π which minimizes an observed surrogate loss between its actions and the actions of the expert under the induced distribution of states, i.e.π

For the following, we will assume is the total variation distance (denoted by · ), which is an upper bound on the 0 − 1 loss.

Our goal is thus to minimize the following quantity, which represents the distance between the actions taken by our policy π and the expert policy π :

The following result shows that if represents an upper bound on the 0 − 1 loss and C satisfies certain smoothness conditions, then minimizing this loss within translates into an O( T ) regret bound on the task cost

Theorem 1. (Ross et al., 2011) Let π be such that J exp (π) = , and

Unfortunately, it is often not possible to optimize J exp directly, since it requires evaluating the expert policy on the states induced by following the current policy.

The supervised behavioral cloning cost J BC , which is computed on states induced by the expert, is often used instead:

Minimizing this loss within yields a quadratic regret bound on regret:

Furthermore, this bound is tight: as we will discuss later, there exist simple problems which match the worst-case lower bound.

Our algorithm is motivated by two criteria: i) the policy should perform similarly to the expert on the expert's data distribution, and ii) the policy should move towards the expert's data distribution if it is away from it.

These two criteria are addressed by combining two losses: a standard behavior cloning loss, and an additional loss which represents the variance over actions induced by sampling different policies from the posterior given the demonstration data D. We call this the uncertainty cost, which is defined as:

2: Initialize policy π and policy ensemble E = {π e } 3: for e = 1, E do 4:

Sample D e ∼ D with replacement, with |D e | = |D|.

Train π e to minimize J BC (π e ) on D e to convergence.

6: end for 7: for i = 1, ... do

Perform one gradient update to minimize J BC (π) using a minibatch from D.

Perform one step of policy gradient to minimize E s∼dπ,a∼π(·|s) [C clip U (s, a)].

10: end for

The motivation is that the variance over plausible policies is high outside the expert's distribution, since the data is sparse, but it is low inside the expert's distribution, since the data there is dense.

Minimizing this cost encourages the policy to return to regions of dense coverage by the expert.

Intuitively, this is what we would expect the expert policy π to do as well.

The total cost which the algorithm optimizes is given by:

The first term is a behavior cloning loss and is computed over states generated by the expert policy, of which the demonstration data D is a representative sample.

The second term is computed over the distribution of states generated by the current policy and can be optimized using policy gradient.

A number of methods can be used to approximate the posterior p(π|D).

In our experiments, we use an ensemble E = {π e } |E| e=1 of models with different initializations which are trained on different bootstrap samples of the demonstration data.

Note that the demonstration data is fixed, and this ensemble can be trained once offline.

We then interleave the supervised behavioral cloning updates and the policy gradient updates which minimize the variance of the posterior.

The full algorithm is shown in Algorithm 1.

Other methods for approximating the posterior include Bayesian neural networks (MacKay, 1992) using diagonal Gaussian approximations (Blundell et al., 2015) or MCdropout (Gal & Ghahramani, 2016) , which we also found to work well (see Appendix D.2).

In practice, for the supervised loss we optimize the KL divergence between the actions predicted by the policy and the expert actions, which is an upper bound on the total variation distance due to Pinsker's inequality.

We also found it helpful to use a clipped uncertainty cost:

where the threshold q is a top quantile of the raw uncertainty costs computed over the demonstration data.

The threshold q defines a normal range of uncertainty based on the demonstration data, and values above this range incur a positive cost (or negative reward).

The RL cost can be optimized using any policy gradient method.

In our experiments we used advantage actor-critic (A2C) (Mnih et al., 2016) , which estimates the expected cost using rollouts from multiple parallel actors all sharing the same policy (see Appendix C for details).

We note that model-based RL methods could in principle be used as well if sample efficiency is a constraint.

We now analyze DRIL for tabular MDPs.

We will show that, subject to assumptions that the policy class contains an optimal policy and that we are able to optimize costs within of their global minimum, our algorithm obtains a regret bound which is linear in κ T , where κ is a quantity specific to the environment and d π .

Intuitively, κ represents a tradeoff between how concentrated the demonstration data is and how high the variance of the posterior is outside the expert distribution.

Assumption 1. (Realizability) π ∈ Π. Assumption 2. (Optimization Oracle) For any given cost function J, our minimization procedure returns a policyπ ∈ Π such that J(π) ≤ arg min π∈Π J(π) + .

The motivation behind our algorithm is that the policies in the ensemble agree inside the expert's distribution and disagree outside of it.

This defines a reward function which pushes the learner back towards the expert's distribution if it strays away.

However, what constitutes inside and outside the distribution, or sufficient agreement or disagreement, is ambiguous.

Below we define quantities which makes these ideas precise.

Definition 1.

For any set U ⊆ S, define the maximum probability ratio between the state distributions induced by the expert policy and by policies in the policy class inside of U as α(U) = max π∈Π s∈U

.

For a set U, α(U) will be low if the expert distribution has high mass inside of U, and the states in U is reachable by policies in the policy class.

Definition 2.

Define the minimum variance of the posterior outside of U as

We now define the κ coefficient as the minimum ratio of these two quantities over all possible subsets of S. Definition 3.

We define κ(U) = α(U ) β(U ) , and κ = min U ⊆S κ(U).

We can view minimizing κ(U) over different U ⊆ S as minimizing a tradeoff between coverage by the expert policy inside of U, and variance of the posterior outside of U. Note that by making U very small, it may be easy to make α(U) small, but doing so may also make β(U) small and κ(U) large.

Conversely, making U large may make β(U) large but may also make α(U) large as a result.

We now establish a relationship between the κ coefficient just defined, the cost our algorithm optimizes, and J exp defined in Equation (2) which we would ideally like to minimize and which translates into a regret bound.

All proofs can be found in Appendix A.

This result shows that if κ is not too large, and we are able to make our cost function J alg (π) small, then we can ensure J exp (π) is also be small.

This result is only useful if our cost function can indeed achieve a small minimum.

The next lemma shows that this is the case.

Here is the threshold specified in Assumption 2.

Combining these two lemmas with the previous result of Ross et al. (2011) , we get a regret bound which is linear in κ T .

Theorem 3.

Letπ be the result of minimizing J alg using our optimization oracle, and assume that

Our bound is an improvement over that of behavior cloning if κ is less than O(T ).

Note that DRIL does not require knowledge of κ .

The quantity κ is problem-dependent and depends on the environment dynamics, the expert policy and the class of policies available to the learner.

We next compute κ exactly for a problem for which behavior cloning is known to perform poorly, and show that it is independent of T .

Example 1.

Consider the tabular MDP given in (Ross & Bagnell, 2010) as an example of a problem where behavioral cloning incurs quadratic regret, shown in Figure 1 .

There are 3 states S = (s 0 , s 1 , s 2 ) and two actions (a 1 , a 2 ).

Each policy π can be represented as a set of probabilities π(a 1 |s) for each state s ∈ S 1 .

The posterior p(π(a 1 |s)|D) is given by a Beta distribution with parameters Beta(n 1 + 1, n 2 + 1) where n 1 , n 2 are the number of times the pairs (s, a 1 ) and (s, a 2 ) occur, respectively, in the demonstration data D. The agent always starts in s 0 and the expert's policy is given by π (a 1 |s 0 ) = 1, π (a 1 |s 1 ) = 0, π (a 1 |s 2 ) = 1.

due to the dynamics of the MDP, so dπ(s) d π (s) ≤ 1 for s ∈ {s 0 , s 1 }.

Furthermore, since s 2 is never visited in the demonstration data, p(π(a 1 |s 2 )|D) = Beta(1, 1) = U nif orm(0, 1), which also implies that p(π(a 1 |s 2 )|D) = U nif orm(0, 1).

It follows that Var π∼p(π|D) (π(a|s 2 )) is equal to the variance of a uniform distribution over [0, 1], i.e. 1 12 .

Therefore:

Applying our result from Theorem 3, we see that our algorithm obtains an O( T ) regret bound on this problem, in contrast to the O( T 2 ) regret of behavioral cloning 2 .

The idea of learning through imitation dates back at least to the work of (Pomerleau, 1989) , who trained a neural network to imitate the steering actions of a human driver using images as input.

The problem of covariate shift was already observed, as the author notes: "when driving for itself, the network may occasionally stray from the center of the road and so must be prepared to recover by steering the vehicle back to the center of the road".

This issue was formalized in the work of (Ross & Bagnell, 2010) , who on one hand proved an O( T 2 ) regret bound, and on the other hand provided an example showing this bound is tight.

The subsequent work (Ross et al., 2011) proposed the DAGGER algorithm which obtains linear regret, provided the agent can both interact with the environment, and query the expert policy.

Our approach also requires environment interaction, but importantly does not require the ability to query the expert.

Also of note is the work of (Venkatraman et al., 2015) , which extended DAGGER to time series prediction problems by using the true targets as expert corrections.

Imitation learning has been used within the context of modern RL to help improve sample efficiency (Hester et al., 2018) or overcome exploration (Nair et al., 2017) .

These settings assume the reward is known and that the policies can then be fine-tuned with reinforcement learning.

In this case, covariate shift is less of an issue since it can be corrected using the reinforcement signal.

The work of (Luo et al., 2019 ) also proposed a method to address the covariate shift problem when learning from demonstrations when the reward is known, by conservatively extrapolating the value function outside the training distribution using negative sampling.

This addresses a different setting from ours, and requires generating plausible states which are off the manifold of training data, which may be challenging when the states are high dimensional such as images.

The work of Reddy et al. (2019) proposed to treat imitation learning within the Q-learning framework, setting a positive reward for all transitions inside the demonstration data and zero reward for all other transitions in the replay buffer.

This rewards the agent for repeating (or returning to) the expert's transitions.

The work of (Sasaki et al., 2019 ) also incorporates a mechanism for reducing covariate shift by fitting a Q-function that classifies whether the demonstration states are reachable from the current state.

Random Expert Distillation (Wang et al., 2019) uses Random Network Distillation (RND) Burda et al. (2019) to estimate the support of the expert's distribution in state-action space, and minimizes an RL cost designed to guide the agent towards the expert's support.

This is related to our method, but differs in that it minimizes the RND prediction error rather than the posterior variance and does not include a behavior cloning cost.

The behavior cloning cost is essential to our theoretical results and avoids certain failure modes, see Appendix B for more discusion.

Generative Adversarial Imitation Learning (GAIL) (Ho & Ermon, 2016 ) is a state-of-the-art algorithm which addresses the same setting as ours.

It operates by training a discriminator network to distinguish expert states from states generated by the current policy, and the negative output of the discriminator is used as a reward signal to train the policy.

The motivation is that states which are outside the training distribution will be assigned a low reward while states which are close to it will be assigned a high reward.

This encourages the policy to return to the expert distribution if is strays away from it.

However, the adversarial training procedure means that the reward function is changing over time, which can make the algorithm unstable or difficult to tune.

In contrast, our approach uses a simple fixed reward function.

We include comparisons to GAIL in our experiments.

Using disagreement between models in an ensemble to represent uncertainty has recently been explored in several contexts.

The works of (Shyam et al., 2018; Pathak et al., 2019) used disagreement between different dynamics models to drive exploration in the context of model-based RL.

Conversely, (Henaff et al., 2019) used variance across different dropout masks to prevent policies from exploiting error in dynamics models.

Ensembles have also been used to represent uncertainty over Q-values in model-free RL in order to encourage exploration (Osband et al., 2016) .

Within the context of imitation learning, the work of (Menda et al., 2018) used the variance of the ensemble together with the DAGGER algorithm to decide when to query the expert demonstrator to minimize unsafe situations.

Here, we use disagreement between different policies sampled from the posterior to address covariate shift in the context of imitation learning.

As a first experiment, we applied DRIL to the tabular MDP of (Ross & Bagnell, 2010) shown in Figure 1 .

We computed the posterior over the policy parameters given the demonstration data using (Ross & Bagnell, 2010) .

Shaded region represents range between 5 th and 95 th quantiles, computed across 500 trials.

Behavior cloning exhibits poor worstcase regret, whereas DRIL has low regret across all trials.

a separate Beta distribution for each state s with parameters determined by the number of times each action was performed in s. For behavior cloning, we sampled a single policy from this posterior.

For our method, we sampled 5 policies and used their negative variance to define an additional reward function.

We combined this with a reward which was the probability density function of a given state-action pair under the posterior distribution, which corresponds to the supervised learning loss, and used tabular Q-learning to optimize the sum of these two reward functions.

This experiment was repeated 500 times for time horizon lengths up to 500 and N = 1, 5, 10 expert demonstration trajectories.

Figure 2 shows plots of the regret over the 500 different trials across different time horizons.

Although the average performance of BC improves with more expert demonstrations, it exhibits poor worst-case performance with some trials incurring very high regret, especially when using fewer demonstrations.

Our method has low regret across all trials, which stays close to constant independantly of the time horizon, even with a single demonstration.

This performance is better than that suggested by our analysis, which showed a worst-case linear bound with respect to time horizon.

We next evaluated our approach on six different Atari environments.

We used pretrained PPO (Schulman et al., 2017) agents from the stable baselines repository (Hill et al., 2018) to generate N = {1, 3, 5, 10, 15, 20} expert trajectories.

We compared against two other methods: standard behavioral cloning (BC) and Generative Adversarial Imitation Learning (GAIL).

Results are shown in Figure 3a .

DRIL outperforms behavioral cloning across most environments and numbers of demonstrations, often by a substantial margin.

In the worst case its performance matches that of behavior cloning.

In many cases, our method is able to match the expert's performance using a small number of trajectories.

Figure 3b shows the evolution of the uncertainty cost and the policy reward throughout training.

In all cases, the test reward improves while the uncertainty cost decreases.

Interestingly, there is correspondence between the change in the uncertainty cost during training and the gap in performance between behavior cloning and DRIL.

For example, in MsPacman there is both a small improvement in uncertainty cost over time and a small gap between behavior cloning and our method, whereas in Breakout there is a large improvement in uncertainty cost and a large gap between behavior cloning and our method.

This suggests that the gains from our method comes from redirecting the policy back towards the expert manifold, which is manifested as a decrease in the uncertainty cost.

We were not able to obtain meaningful performance for GAIL on these domains, despite performing a hyperparameter search across learning rates for the policy and discriminator, and across different numbers of discriminator updates.

We additionally experimented with clipping rewards in an effort to stabilize performance.

These results are consistent with those of (Reddy et al., 2019) , who also reported negative results when running GAIL on images.

While improved performance might be possible with more sophisticated adversarial training techniques, we note that this contrasts with our method which uses a fixed reward function obtained through simple supervised learning.

In Appendix D we provide ablation experiments examining the effects of different cost function choices and the role of the BC loss.

We also compare the ensemble approach to a dropout-based approach for posterior approximation and show that DRIL works well in both cases.

We next report results of running our method on a 6 different continuous control tasks from the PyBullet 3 and OpenAI Gym (Brockman et al., 2016) environments.

We again used pretrained agents to generate expert demonstrations.

Results are shown in Figure 4 .

In these environments we found behavior cloning to be a much stronger baseline than for the Atari environments, and in many tasks it was able to match expert performance using as little as 3 trajectories.

Our method exhibits a modest improvement on Walker2D and BipedalWalkerHardcore when a single trajectory is used, and otherwise has similar performance to behavior cloning.

The fact that our method does not perform worse than behavior cloning on tasks where covariate shift is likely less of an issue provides evidence of its robustness.

Addressing covariate shift has been a long-standing challenge in imitation learning.

In this work, we have proposed a new method to address this problem by penalizing the disagreement between an ensemble of different policies sampled from the posterior.

Importantly, our method requires no additional labeling by an expert.

Our experimental results demonstrate that DRIL can often match expert performance while using only a small number of trajectories across a wide array of tasks, ranging from tabular MDPs to pixel-based Atari games and continuous control tasks.

On the theoretical side, we have shown that our algorithm can provably obtain a low regret bound for tabular problems in which the κ parameter is low.

There are multiple directions for future work.

On the theoretical side, extending our analysis to continuous state spaces and characterizing the κ parameter on a larger array of problems would help to better understand the settings where our method can expect to do well.

Empirically, there are many other settings in structured prediction (Daumé et al., 2009 ) where covariate shift is an issue and where our method could be applied.

For example, in dialogue and language modeling it is common for generated text to become progressively less coherent as errors push the model off the manifold it was trained on.

Our method could potentially be used to fine-tune language or translation models (Cho et al., 2014; Welleck et al., 2019) after training by applying our uncertainty-based cost function to the generated text.

A PROOFS

Proof.

We will first show that for any π ∈ Π and U ⊆ S, we have

.

We can rewrite this as:

We begin by bounding the first term:

We next bound the second term:

Now observe we can decompose the RL cost as follows:

Putting these together, we get the following:

Here we have used the fact that β(U) ≤ 1 since 0 ≤ π(a|s) ≤ 1 and α(U) ≥ s∈U

Taking the minimum over subsets U ⊆ S, we get J exp (π) ≤ κJ alg (π).

Proof.

Plugging the optimal policy into J alg , we get:

We will first bound Term 1:

We will next bound Term 2:

The last step follows from our optimization oracle assumption:

Combining the bounds on the two terms, we get J alg (π ) ≤ 2 .

Since π ∈ Π, the result follows.

Theorem 1.

Letπ be the result of minimizing J alg using our optimization oracle, and assume that

Proof.

By our optimization oracle and Lemma 2, we have

Combining with Lemma 1, we get:

Applying Theorem 1 from (Ross et al., 2011) , we get J(π) ≤ J(π ) + 3uκ T .

The following example shows how minimizing the uncertainty cost alone without the BC cost can lead to highly sub-optimal policies if the demonstration data is generated by a stochastic policy which is only slightly suboptimal.

Consider the following deterministic chain MDP:

The agent always starts in s 1 , and gets a reward of 1 in s 3 and 0 elsewhere.

The optimal policy is given by:

Assume the demonstration data is generated by the following policy, which is only slightly suboptimal:

Let us assume realizability and perfect optimization for simplicity.

If both transitions (s 2 , a 0 ) and (s 2 , a 1 ) appear in the demonstration data, then Random Expert Distillation (RED) will assign zero cost to both transitions.

If we do not use bootstrapped samples to train the ensemble, then DRIL without the BC cost (we will call this UO-DRIL for Uncertainty-Only DRIL) will also assign zero cost to both transitions since all models in the ensemble would recover the Bayes optimal solution given the demonstration data.

If we are using bootstrapped samples, then the Bayes optimal solution for each bootstrapped sample may differ and thus the different policies in the ensemble might disagree in their predictions, although given enough demonstration data we would expect these differences (and thus the uncertainty cost) to be small.

Note also that since no samples at the state s 0 occur in the demonstration data, both RED and UO-DRIL will likely assign high uncertainty costs to state-action pairs at (s 0 , a 0 ), (s 0 , a 1 ) and thus avoid highly suboptimal policies which get stuck at s 0 .

Now consider policiesπ 1 ,π 2 given by:π

Both of these policies only visit state-action pairs which are visited by the demonstration policy.

In the case described above, both RED and UO-DRIL will assignπ 1 andπ 2 similarly low costs.

However,π 1 will cycle forever between s 1 and s 2 , never collecting reward, whileπ 2 will with high probability reach s 3 and stay there, thus achieving high reward.

This shows that minimizing the uncertainty cost alone does not necessarily distinguish between good and bad policies.

However,π 1 will incur a higher BC cost thanπ 2 , sinceπ 2 more closely matches the demonstration data at s 2 .

This shows that including the BC cost can be important for further disambiguating between policies which all stay within the distribution of the demonstration data, but have different behavior within that distribution.

C EXPERIMENTAL DETAILS C.1 ATARI ENVIRONMENTS All behavior cloning and ensemble models were trained to minimize the negative log-likelihood classification loss on the demonstration data for 500 epochs using Adam (Kingma & Ba, 2014 ) and a learning rate of 2.5 · 10 −4 .

For our method, we initially performed a hyperparameter search on Space Invaders over the following values:

We then chose the best values and kept those hyperparameters fixed for all other environments.

All other A2C hyperparameters follow the default values in the repo (Kostrikov, 2018) : policy networks consisted of 3-layer convolutional networks with 8−32−64 feature maps followed by a single-layer MLP with 512 hidden units.

For GAIL, we used the implementation in (Kostrikov, 2018) and replaced the MLP discriminator by a CNN discriminator with the same architecture as the policy network.

We initially performed a hyperparameter search on Breakout with 10 demonstrations over the values shown in Table 2 .

However, we did not find any hyperparameter configuration which performed better than behavioral cloning.

All behavior cloning and ensemble models were trained to minimize the mean-squared error regression loss on the demonstration data for 500 epochs using Adam (Kingma & Ba, 2014 ) and a learning rate of 2.5 · 10 −4 .

Policy networks were 2-layer fully-connected MLPs with tanh activations and 64 hidden units.

• DRIL This is the regular DRIL agent, which optimizes both the BC cost and the clipped cost:

• DRIL (clipped cost 0/1) This is the same as the regular DRIL agent, except that we use the following clipped cost function: • DRIL (raw cost) This is the same as the regular DRIL agent, except that we use the raw cost C U (s, a) rather than the clipped cost.

• DRIL (no BC cost) This is the same as the regular DRIL agent, except that we remove the BC updates and only optimize C clip U (s, a).

Results are shown in Figure 5 .

First, switching from the clipped cost in {−1, +1} to the clipped cost in {0, 1} or the raw cost causes a large drop in performance across most environments.

One explanation may be that since the costs are always positive for both variants (which corresponds to a reward which is always negative), the agent may learn to terminate the episode early in order to minimize the total cost incurred.

Using a cost/reward which has both positive and negative values avoids this behavior.

Second, optimizing the pure BC cost performs better than the pure uncertainty cost for some environments (MsPacman, SpaceInvaders, BeamRider) while optimizing the pure uncertainty cost performs better than BC for others (Breakout, Qbert).

DRIL, which optimizes both, has robust performance and performs the best out of the different variants over most environments and numbers of trajectories.

We provide additional results comparing the ensembling and MC-dropout (Gal & Ghahramani, 2016) approaches to posterior estimation.

For MC-Dropout we trained a single policy network with a dropout rate of 0.1 applied to all layers except the last, and estimated the variance for each state-action pair using 10 different dropout masks.

Similarly to the ensemble approach, we computed the 98 th quantile of the variance on the demonstration data and used this value in our clipped cost.

Results are shown for three environments in Figure 6 .

MC-dropout performs similarly to the ensembling approach, which shows that our method can be paired with different approaches to posterior estimation.

<|TLDR|>

@highlight

Method for addressing covariate shift in imitation learning using ensemble uncertainty