Optimistic initialisation is an effective strategy for efficient exploration in reinforcement learning (RL).

In the tabular case, all provably efficient model-free algorithms rely on it.

However, model-free deep RL algorithms do not use optimistic initialisation despite taking inspiration from these provably efficient tabular algorithms.

In particular, in scenarios with only positive rewards, Q-values are initialised at their lowest possible values due to commonly used network initialisation schemes, a pessimistic initialisation.

Merely initialising the network to output optimistic Q-values is not enough, since we cannot ensure that they remain optimistic for novel state-action pairs, which is crucial for exploration.

We propose a simple count-based augmentation to pessimistically initialised Q-values that separates the source of optimism from the neural network.

We show that this scheme is provably efficient in the tabular setting and extend it to the deep RL setting.

Our algorithm, Optimistic Pessimistically Initialised Q-Learning (OPIQ), augments the Q-value estimates of a DQN-based agent with count-derived bonuses to ensure optimism during both action selection and bootstrapping.

We show that OPIQ outperforms non-optimistic DQN variants that utilise a pseudocount-based intrinsic motivation in hard exploration tasks, and that it predicts optimistic estimates for novel state-action pairs.

In reinforcement learning (RL), exploration is crucial for gathering sufficient data to infer a good control policy.

As environment complexity grows, exploration becomes more challenging and simple randomisation strategies become inefficient.

While most provably efficient methods for tabular RL are model-based (Brafman and Tennenholtz, 2002; Strehl and Littman, 2008; Azar et al., 2017) , in deep RL, learning models that are useful for planning is notoriously difficult and often more complex (Hafner et al., 2019) than modelfree methods.

Consequently, model-free approaches have shown the best final performance on large complex tasks (Mnih et al., 2015; 2016; Hessel et al., 2018) , especially those requiring hard exploration (Bellemare et al., 2016; Ostrovski et al., 2017) .

Therefore, in this paper, we focus on how to devise model-free RL algorithms for efficient exploration that scale to large complex state spaces and have strong theoretical underpinnings.

Despite taking inspiration from tabular algorithms, current model-free approaches to exploration in deep RL do not employ optimistic initialisation, which is crucial to provably efficient exploration in all model-free tabular algorithms.

This is because deep RL algorithms do not pay special attention to the initialisation of the neural networks and instead use common initialisation schemes that yield initial Q-values around zero.

In the common case of non-negative rewards, this means Q-values are initialised to their lowest possible values, i.e., a pessimistic initialisation.

While initialising a neural network optimistically would be trivial, e.g., by setting the bias of the final layer of the network, the uncontrolled generalisation in neural networks changes this initialisation quickly.

Instead, to benefit exploration, we require the Q-values for novel state-action pairs must remain high until they are explored.

An empirically successful approach to exploration in deep RL, especially when reward is sparse, is intrinsic motivation (Oudeyer and Kaplan, 2009) .

A popular variant is based on pseudocounts (Bellemare et al., 2016) , which derive an intrinsic bonus from approximate visitation counts over states and is inspired by the tabular MBIE-EB algorithm (Strehl and Littman, 2008) .

However, adding a positive intrinsic bonus to the reward yields optimistic Q-values only for state-action pairs that have already been chosen sufficiently often.

Incentives to explore unvisited states rely therefore on the generalisation of the neural network.

Exactly how the network generalises to those novel state-action pairs is unknown, and thus it is unclear whether those estimates are optimistic when compared to nearby visited state-action pairs.

Figure 1 Consider the simple example with a single state and two actions shown in Figure  1 .

The left action yields +0.1 reward and the right action yields +1 reward.

An agent whose Q-value estimates have been zero-initialised must at the first time step select an action randomly.

As both actions are underestimated, this will increase the estimate of the chosen action.

Greedy agents always pick the action with the largest Q-value estimate and will select the same action forever, failing to explore the alternative.

Whether the agent learns the optimal policy or not is thus decided purely at random based on the initial Q-value estimates.

This effect will only be amplified by intrinsic reward.

To ensure optimism in unvisited, novel state-action pairs, we introduce Optimistic Pessimistically Initialised Q-Learning (OPIQ).

OPIQ does not rely on an optimistic initialisation to ensure efficient exploration, but instead augments the Q-value estimates with count-based bonuses in the following manner:

where N (s, a) is the number of times a state-action pair has been visited and M, C > 0 are hyperparameters.

These Q + -values are then used for both action selection and during bootstrapping, unlike the above methods which only utilise Q-values during these steps.

This allows OPIQ to maintain optimism when selecting actions and bootstrapping, since the Q + -values can be optimistic even when the Q-values are not.

In the tabular domain, we base OPIQ on UCB-H (Jin et al., 2018) , a simple online Q-learning algorithm that uses count-based intrinsic rewards and optimistic initialisation.

Instead of optimistically initialising the Q-values, we pessimistically initialise them and use Q + -values during action selection and bootstrapping.

Pessimistic initialisation is used to enable a worst case analysis where all of our Q-value estimates underestimate Q * and is not a requirement for OPIQ.

We show that these modifications retain the theoretical guarantees of UCB-H. Furthermore, our algorithm easily extends to the Deep RL setting.

The primary difficulty lies in obtaining appropriate state-action counts in high-dimensional and/or continuous state spaces, which has been tackled by a variety of approaches (Bellemare et al., 2016; Ostrovski et al., 2017; Tang et al., 2017; Machado et al., 2018a) and is orthogonal to our contributions.

We demonstrate clear performance improvements in sparse reward tasks over 1) a baseline DQN that just uses intrinsic motivation derived from the approximate counts, 2) simpler schemes that aim for an optimistic initialisation when using neural networks, and 3) strong exploration baselines.

We show the importance of optimism during action selection for ensuring efficient exploration.

Visualising the predicted Q + -values shows that they are indeed optimistic for novel state-action pairs.

We consider a Markov Decision Process (MDP) defined as a tuple (S, A, P, R), where S is the state space, A is the discrete action space, P (??|s, a) is the state-transition distribution, R(??|s, a) is the distribution over rewards and ?? ??? [0, 1) is the discount factor.

The goal of the agent is then to maximise the expected discounted sum of rewards: E[ ??? t=0 ?? t r t |r t ??? R(??|s t , a t )], in the discounted episodic setting.

A policy ??(??|s) is a mapping from states to actions such that it is a valid probability distribution.

Deep Q-Network (DQN) (Mnih et al., 2015) uses a nonlinear function approximator (a deep neural network) to estimate the action-value function, Q(s, a; ??) ??? Q * (s, a), where ?? are the parameters of the network.

Exploration based on intrinsic rewards (e.g., Bellemare et al., 2016) , which uses a DQN agent, additionally augments the observed rewards r t with a bonus ??/ N (s t , a t ) based on pseudo-visitation-counts N (s t , a t ).

The DQN parameters ?? are trained by gradient descent on the mean squared regression loss L with bootstrapped 'target' y t :

Figure 2: A simple regression task to illustrate the effect of an optimistic initialisation in neural networks.

Left: 10 different networks whose final layer biases are initialised at 3 (shown in green), and the same networks after training on the blue data points (shown in red).

Right: One of the trained networks whose output has been augmented with an optimistic bias as in equation 1.

The counts were obtained by computing a histogram over the input space [???2, 2] with 50 bins.

The expectation is estimated with uniform samples from a replay buffer D (Lin, 1992) .

D stores past transitions (s t , a t , r t , s t+1 ), where the state s t+1 is observed after taking the action a t in state s t and receiving reward r t .

To improve stability, DQN uses a target network, parameterised by ?? ??? , which is periodically copied from the regular network and kept fixed for a number of iterations.

Our method Optimistic Pessimistically Initialised Q-Learning (OPIQ) ensures optimism in the Qvalue estimates of unvisited, novel state-action pairs in order to drive exploration.

This is achieved by augmenting the Q-value estimates in the following manner:

and using these Q + -values during action selection and bootstrapping.

In this section, we motivate OPIQ, analyse it in the tabular setting, and describe a deep RL implementation.

Optimistic initialisation does not work with neural networks.

For an optimistic initialisation to benefit exploration, the Q-values must start sufficiently high.

More importantly, the values for unseen state-action pairs must remain high, until they are updated.

When using a deep neural network to approximate the Q-values, we can initialise the network to output optimistic values, for example, by adjusting the final bias.

However, after a small amount of training, the values for novel state-action pairs may not remain high.

Furthermore, due to the generalisation of neural networks we cannot know how the values for these unseen state-action pairs compare to the trained state-action pairs.

Figure 2 (left), which illustrates this effect for a simple regression task, shows that different initialisations can lead to dramatically different generalisations.

It is therefore prohibitively difficult to use optimistic initialisation of a deep neural network to drive exploration.

Instead, we augment our Q-value estimates with an optimistic bonus.

Our motivation for the form of the bonus in equation 1, C (N (s,a)+1) M , stems from UCB-H (Jin et al., 2018) , where all tabular Q-values are initialised with H and the first update for a state-action pair completely overwrites that value because the learning rate for the update (?? 1 ) is 1.

One can alternatively view these Q-values as zero-initialised with the additional term Q(s, a) + H ?? 1{N(s, a) < 1}, where N (s, a) is the visitation count for the state-action pair (s, a).

Our approach approximates the discrete indicator function 1 as (N (s, a) + 1) ???M for sufficiently large M .

However, since gradient descent cannot completely overwrite the Q-value estimate for a state-action pair after a single update, it is beneficial to have a smaller hyperparameter M that governs how quickly the optimism decays.

for each timestep t = 1, ..., H do Take action a t ??? arg max a Q + t (s t , a).

Receive r(s t , a t , t) and s t+1 .

Increment N (s t , a t , t).

For a worst case analysis we assume all Q-value estimates are pessimistic.

In the common scenario where all rewards are nonnegative, the lowest possible return for an episode is zero.

If we then zero-initialise our Q-value estimates, as is common for neural networks, we are starting with a pessimistic initialisation.

As shown in Figure 2 (left), we cannot predict how a neural network will generalise, and thus we cannot predict if the Q-value estimates for unvisited state-action pairs will be optimistic or pessimistic.

We thus assume they are pessimistic in order to perform a worst case analysis.

However, this is not a requirement: our method works with any initialisation and rewards.

In order to then approximate an optimistic initialisation, the scaling parameter C in equation 1 can be chosen to guarantee unseen Q + -values are overestimated, for example, C := H in the undiscounted finite-horizon tabular setting and C := 1/(1 ??? ??) in the discounted episodic setting (assuming 1 is the maximum reward obtainable at each timestep).

However, in some environments it may be beneficial to use a smaller parameter C for faster convergence.

These Q + -values are then used both during action selection and during bootstrapping.

Note that in the finite horizon setting the counts N , and thus Q + , would depend on the timestep t.

Hence, we split the optimistic Q + -values into two parts: a pessimistic Q-value component and an optimistic component based solely on the counts for a state-action pair.

This separates our source of optimism from the neural network function approximator, yielding Q + -values that remain high for unvisited state-action pairs, assuming a suitable counting scheme.

Figure 2 (right) shows the effects of adding this optimistic component to a network's outputs.

+ -values provide an increased incentive to explore.

By using optimistic Q + estimates, especially during action selection and bootstrapping, the agent is incentivised to try and visit novel state-action pairs.

Being optimistic during action selection in particular encourages the agent to try novel actions that have not yet been visitied.

Without an optimistic estimate for novel state-action pairs the agent would have no incentive to try an action it has never taken before at a given state.

Being optimistic during bootstrapping ensures the agent is incentivised to return to states in which it has not yet tried every action.

This is because the maximum Q + -value will be large due to the optimism bonus.

Both of these effects lead to a strong incentive to explore novel state-action pairs.

In order to ensure that OPIQ has a strong theoretical foundation, we must ensure it is provably efficient in the tabular domain.

We restrict our analysis to the finite horizon tabular setting and only consider building upon UCB-H (Jin et al., 2018) for simplicity.

Achieving a better regret bound using UCB-B (Jin et al., 2018) and extending the analysis to the infinite horizon discounted setting (Dong et al., 2019) are steps for future work.

Our algorithm removes the optimistic initialisation of UCB-H, instead using a pessimistic initialisation (all Q-values start at 0).

We then use our Q + -values during action selection and bootstrapping.

Pseudocode is presented in Algorithm 1.

Theorem 1.

For any p ??? (0, 1) , with probability at least 1 ??? p the total regret of Q + is at most

The proof is based on that of Theorem 1 from (Jin et al., 2018) .

Our Q + -values are always greater than or equal to the Q-values that UCB-H would estimate, thus ensuring that our estimates are also greater than or equal to Q * .

Our overestimation relative to UCB-H is then governed by the quantity H/(N (s, a) + 1) M , which when summed over all timesteps does not depend on T for M > 1.

As M ??? ??? we exactly recover UCB-H, and match the asymptotic performance of UCB-H for M ??? 1.

Smaller values of M result in our optimism decaying more slowly, which results in more exploration.

The full proof is included in Appendix I.

We also show that OPIQ without optimistic action selection or the count-based intrinsic motivation term b T N is not provably efficient by showing it can incur linear regret with high probability on simple MDPs (see Appendices G and H).

Our primary motivation for considering a tabular algorithm that pessimistically initialises its Q-values, is to provide a firm theoretical foundation on which to base a deep RL algorithm, which we describe in the next section.

For deep RL, we base OPIQ on DQN (Mnih et al., 2015) , which uses a deep neural network with parameters ?? as a function approximator Q ?? .

During action selection, we use our Q + -values to determine the greedy action:

where C action is a hyperparameter governing the scale of the optimistic bias during action selection.

In practice, we use an -greedy policy.

After every timestep, we sample a batch of experiences from our experience replay buffer, and use n-step Q-learning (Mnih et al., 2016) .

We recompute the counts for each relevant state-action pair, to avoid using stale pseudo-rewards.

The network is trained by gradient decent on the loss in equation 2 with the target:

where C bootstrap is a hyperparameter that governs the scale of the optimistic bias during bootstrapping.

For our final experiments on Montezuma's Revenge we additionally use the Mixed Monte Carlo (MMC) target (Bellemare et al., 2016; Ostrovski et al., 2017) , which mixes the target with the environmental monte carlo return for that episode.

Further details are included in Appendix D.4.

We use the method of static hashing (Tang et al., 2017 ) to obtain our pseudocounts on the first 2 of 3 environments we test on.

For our experiments on Montezuma's Revenge we count over a downsampled image of the current game frame.

More details can be found in Appendix B.

A DQN with pseudocount derived intrinsic reward (DQN + PC) (Bellemare et al., 2016) can be seen as a naive extension of UCB-H to the deep RL setting.

However, it does not attempt to ensure optimism in the Q-values used during action selection and bootstrapping, which is a crucial component of UCB-H. Furthermore, even if the Q-values were initialised optimistically at the start of training they would not remain optimistic long enough to drive exploration, due to the use of neural networks.

OPIQ, on the other hand, is designed with these limitations of neural networks in mind.

By augmenting the neural network's Q-value estimates with optimistic bonuses of the form C (N (s,a)+1) M , OPIQ ensures that the Q + -values used during action selection and bootstrapping are optimistic.

We can thus consider OPIQ as a deep version of UCB-H. Our results show that optimism during action selection and bootstrapping is extremely important for ensuring efficient exploration.

Tabular Domain:

There is a wealth of literature related to provably efficient exploration in the tabular domain.

Popular model-based algorithms such as R-MAX (Brafman and Tennenholtz, 2002) , MBIE (and MBIE-EB) (Strehl and Littman, 2008) , UCRL2 (Jaksch et al., 2010) and UCBVI (Azar et al., 2017) are all based on the principle of optimism in the face of uncertainty.

Osband and Van Roy (2017) adopt a Bayesian viewpoint and argue that posterior sampling (PSRL) (Strens, 2000) is more practically efficient than approaches that are optimistic in the face of uncertainty, and prove that in Bayesian expectation PSRL matches the performance of any optimistic algorithm up to constant factors.

Agrawal and Jia (2017) prove that an optimistic variant of PSRL is provably efficient under a frequentist regret bound.

The only provably efficient model-free algorithms to date are delayed Q-learning (Strehl et al., 2006) and UCB-H (and UCB-B) (Jin et al., 2018) .

Delayed Q-learning optimistically initialises the Q-values that are carefully controlled when they are updated.

UCB-H and UCB-B also optimistically initialise the Q-values, but also utilise a count-based intrinsic motivation term and a special learning rate to achieve a favourable regret bound compared to model-based algorithms.

In contrast, OPIQ pessimistically initialises the Q-values.

Whilst we base our current analysis on UCB-H, the idea of augmenting pessimistically initialised Q-values can be applied to any model-free algorithm.

Deep RL Setting: A popular approach to improving exploration in deep RL is to utilise intrinsic motivation (Oudeyer and Kaplan, 2009), which computes a quantity to add to the environmental reward.

Most relevant to our work is that of Bellemare et al. (2016) , which takes inspiration from MBIE-EB (Strehl and Littman, 2008) .

Bellemare et al. (2016) utilise the number of times a state has been visited to compute the intrinsic reward.

They outline a framework for obtaining approximate counts, dubbed pseudocounts, through a learned density model over the state space.

Ostrovski et al. (2017) show that RLSVI achieves provably efficient Bayesian expected regret, which requires a prior distribution over MDPs, whereas OPIQ achieves provably efficient worse case regret.

Bootstrapped DQN with a prior is thus a model-free algorithm that has strong theoretical support in the tabular setting.

Empirically, however, its performance on sparse reward tasks is worse than DQN with pseudocounts.

Machado et al. (2015) shift and scale the rewards so that a zero-initialisation is optimistic.

When applied to neural networks this approach does not result in optimistic Q-values due to the generalisation of the networks.

Bellemare et al. (2016) empirically show that using a pseudocount intrinsic motivation term performs much better empirically on hard exploration tasks.

Choshen et al. (2018) attempt to generalise the notion of a count to include information about the counts of future state-actions pairs in a trajectory, which they use to provide bonuses during action selection.

Oh and Iyengar (2018) extend delayed Q-learning to utilise these generalised counts and prove the scheme is PAC-MDP.

The generalised counts are obtained through E-values which are learnt using SARSA with a constant 0 reward and E-value estimates initialised at 1.

When scaling to the deep RL setting, these E-values are estimated using neural networks that cannot maintain their initialisation for unvisited state-action pairs, which is crucial for providing an incentive to explore.

By contrast, OPIQ uses a separate source to generate the optimism necessary to explore the environment.

We compare OPIQ against baselines and ablations on three sparse reward environments.

The first is a randomized version of the Chain environment proposed by Osband et al. (2016) and used in (Shyam et al., 2019 ) with a chain of length 100, which we call Randomised Chain.

The second is a two-dimensional maze in which the agent starts in the top left corner (white dot) and is only rewarded upon finding the goal (light grey dot).

We use an image of the maze as input and randomise the actions similarly to the chain.

The third is Montezuma's Revenge from Arcade Learning environment (Bellemare et al., 2013 ), a notoriously difficult sparse reward environment commonly used as a benchmark to evaluate the performance and scaling of Deep RL exploration algorithms.

See Appendix D for further details on the environments, baselines and hyperparameters used.

We compare OPIQ against a variety of DQN-based approaches that use pseudocount intrinsic rewards, the DORA agent (Choshen et al., 2018) (which generates count-like optimism bonuses using a neural network), and two strong exploration baselines:

-greedy DQN: a standard DQN that uses an -greedy policy to encourage exploration.

We anneal linearly over a fixed number of timesteps from 1 to 0.01.

DQN + PC: we add an intrinsic reward of ??/ N (s, a) to the environmental reward based on (Bellemare et al., 2016; Tang et al., 2017) .

DQN R-Subtract (+PC): we subtract a constant from all environmental rewards received when training, so that a zero-initialisation is optimistic, as described for a DQN in (Bellemare et al., 2016) and based on Machado et al. (2015) .

DQN Bias (+PC): we initialise the bias of the final layer of the DQN to a positive value at the start of training as a simple method for optimistic initialisation with neural networks.

DQN + DORA: we use the generalised counts from (Choshen et al., 2018) as an intrinsic reward.

DQN + DORA OA: we additionally use the generalised counts to provide an optimistic bonus during action selection.

DQN + RND: we add the RND bonus from (Burda et al., 2018) as an intrinsic reward.

BSP: we use Bootstrapped DQN with randomised prior functions (Osband et al., 2018) .

In order to better understand the importance of each component of our method, we also evaluate the following ablations:

Optimistic Action Selection (OPIQ w/o OB): we only use our Q + -values during action selection, and use Q during bootstrapping (without Optimistic Bootstrapping).

The intrinsic motivation term remains.

Optimistic Action Selection and Bootstrapping (OPIQ w/o PC): we use our Q + -values during action selection and bootstrapping, but do not include an intrinsic motivation term (without Pseudo Counts).

We first consider the visually simple domain of the randomised chain and compare the count-based methods.

Figure 3 shows the performance of OPIQ compared to the baselines and ablations.

OPIQ significantly outperforms the baselines, which do not have any explicit mechanism for optimism during action selection.

A DQN with pseudocount derived intrinsic rewards is unable to reliably find the goal state, but setting the final layer's bias to one produces much better performance.

For the DQN variant in which a constant is subtracted from all rewards, all of the configurations (including those with pseudocount derived intrinsic bonuses) were unable to find the goal on the right and thus the agents learn quickly to latch on the inferior reward of moving left.

Compared to its ablations, OPIQ is more stable in this task.

OPIQ without pseudocounts performs similarly to OPIQ but is more varied across seeds, whereas the lack of optimistic bootstrapping results in worse performance and significantly more variance across seeds.

We next consider the harder and more visually complex task of the Maze and compare against all baselines.

Figure 4 shows that only OPIQ is able to find the goal in the sparse reward maze.

This indicates that explicitly ensuring optimism during action selection and bootstrapping can have a significant positive impact in sparse reward tasks, and that a naive extension of UCB-H to the deep RL setting (DQN + PC) results in insufficient exploration.

Figure 4 (right) shows that attempting to ensure optimistic Q-values by adjusting the bias of the final layer (DQN Bias + PC), or by subtracting a constant from the reward (DQN R-Subtract + PC) has very little effect.

As expected DQN + RND performs poorly on this domain compared to the pseudocount based methods.

The visual input does not vary much across the state space, resulting in the RND bonus failing to provide enough intrinsic motivation to ensure efficient exploration.

Additionally it does not feature any explicit mechanism for optimism during action selection, and thus Figure 4 (right) shows it explores the environment relatively slowly.

Both DQN+DORA and DQN+DORA OA also perform poorly in this domain since their source of intrinsic motivation disappears quickly.

As noted in Figure 2 , neural networks do not maintain their starting initialisations after training.

Thus, the intrinsic reward DORA produces goes to 0 quickly since the network producing its bonuses learns to generalise quickly.

BSP is the only exploration baseline we test that does not add an intrinsic reward to the environmental reward, and thus it performs poorly compared to the other baselines on this environment.

Figure 5 shows that OPIQ and all its ablations manage to find the goal in the maze.

OPIQ also explores slightly faster than its ablations (right), which shows the benefits of optimism during both action selection and bootstrapping.

In addition, the episodic reward for the the ablation without optimistic bootstrapping is noticeably more unstable (Figure 5, left) .

Interestingly, OPIQ without pseudocounts performs significantly worse than the other ablations.

This is surprising since the theory suggests that the count-based intrinsic motivation is only required when the reward or transitions of the MDP are stochastic (Jin et al., 2018) , which is not the case here.

We hypothesise that adding PC-derived intrinsic bonuses to the reward provides an easier learning problem, especially when using n-step Q-Learning, which yields the performance gap.

However, our results show that the PC-derived intrinsic bonuses are not enough on their own to ensure sufficient exploration.

The large difference in performance between DQN + PC and OPIQ w/o OB is important, since they only differ in the use of optimistic action selection.

The results in Figures 4 and 5 show that optimism during action selection is extremely important in exploring the environment efficiently.

Intuitively, this makes sense, since this provides an incentive for the agent to try actions it has never tried before, which is crucial in exploration.

Finally, we consider Montezuma's Revenge, one of the hardest sparse reward games from the ALE (Bellemare et al., 2013) .

Note that we only train up to 12.5mil timesteps (50mil frames), a 1/4 of the usual training time (50mil timesteps, 200mil frames).

Figure 7 shows that OPIQ significantly outperforms the baselines in terms of the episodic reward and the maximum episodic reward achieved during training.

The higher episode reward and much higher maximum episode reward of OPIQ compared to DQN + PC once again demonstrates the importance of optimism during action selection and bootstrapping.

In this environment BSP performs much better than in the Maze, but achieves significantly lower episodic rewards than OPIQ.

Figure 8 shows the distinct number of rooms visited across the training period.

We can see that OPIQ manages to reliably explore 12 rooms during the 12.5mil timesteps, significantly more than the other methods, thus demonstrating its improved exploration in this complex environment.

Our results on this challenging environment show that OPIQ can scale to high dimensional complex environments and continue to provide significant performance improvements over an agent only using pseudocount based intrinsic rewards.

This paper presented OPIQ, a model-free algorithm that does not rely on an optimistic initialisation to ensure efficient exploration.

Instead, OPIQ augments the Q-values estimates with a count-based optimism bonus.

We showed that this is provably efficient in the tabular setting by modifying UCB-H to use a pessimistic initialisation and our augmented Q + -values for action selection and bootstrapping.

Since our method does not rely on a specific initialisation scheme, it easily scales to deep RL when paired with an appropriate counting scheme.

Our results showed the benefits of maintaining optimism both during action selection and bootstrapping for exploration on a number of hard sparse reward environments including Montezuma's Revenge.

In future work, we aim to extend OPIQ by integrating it with more expressive counting schemes.

For the tabular setting, we consider a discrete finite-horizon Markov Decision Process (MDP), which can be defined as a tuple (S, A, {P t }, {R t }, H, ??), where S is the finite state space, A is the finite action space, P t (??|s, a) is the state-transition distribution for timestep t = 1, ..., H, R t (??|s, a) is the distribution over rewards after taking action a in state s, H is the horizon, and ?? is the distribution over starting states.

Without loss of generality we assume that R t (??|s, a) ??? [0, 1].

We use S and A to denote the number of states and the number of actions, respectively, and N (s, a, t) as the number of times a state-action pair (s, a) has been visited at timestep t.

Our goal is to find a set of policies ?? t : S ??? A, ?? := {?? t }, that chooses the agent's actions at time t such that the expected sum of future rewards is maximised.

To this end we define the Q-value at time t of a given policy ?? as Q ?? t (s, a) := E r + Q ?? t+1 (s , ?? t+1 (s )) | r???Rt(??|s,a), s ???Pt(??|s,a) , where Q ?? t (s, a) = 0, ???t > H. The agent interacts with the environment for K episodes, T := KH, yielding a total regret:

refers to the starting state and ?? k to the policy at the beginning of episode k. We are interested in bounding the worst case total regret with probability 1 ??? p, 0 < p < 1.

) is an online Q-learning algorithm for the finite-horizon setting outlined above where the worse case total regret is bounded with a probability of 1???p by O( H 4 SAT log(SAT /p).

All Q-values for timesteps t ??? H are optimistically initialised at H. The learning rate is defined as ?? N = H+1 H+N , where N := N (s t , a t , t) is the number of times state-action pair (s t , a t ) has been observed at step t and ?? 1 = 1 at the first encounter of any state-action pair.

The update rule for a transition at step t from state s t to s t+1 , after executing action a t and receiving reward r t , is:

where b

, is the count-based intrinsic motivation term.

In deep RL, the primary difficulty for exploration based on count-based intrinsic rewards is obtaining appropriate state-action counts.

In this paper we utilise approximate counting schemes (Bellemare et al., 2016; Ostrovski et al., 2017; Tang et al., 2017) in order to cope with continuous and/or highdimensional state spaces.

In particular, for the chain and maze environments we use static hashing (Tang et al., 2017) , which projects a state s to a low-dimensional feature vector ??(s) = sign(Af (s)), where f flattens the state s into a single dimension of length D; A is a k ?? D matrix whose entries are initialised i.i.d.

from a unit Gaussian: N (0, 1); and k is a hyperparameter controling the granularity of counting: higher k leads to more distinguishable states at the expense of generalisation.

Given the vector ??(s), we use a counting bloom filter (Fan et al., 2000) to update and retrieve its counts efficiently.

To obtain counts N (s, a) for state-action pairs, we maintain a separate data structure of counts for each action (the same vector ??(s) is used for all actions).

This counting scheme is tabular and hence the counts for sufficiently different states do not interfere with one another.

This ensures Q + -values for unseen state-action pairs in equation 1 are large.

For our experiments on Montezuma's Revenge we use the same method of downsampling as in (Ecoffet et al., 2019) , in which the greyscale state representation is resized from (42x42) to (11x8) and then binned from {0, ..., 255} into 8 categories.

We then maintain tabular counts over the new representation.

The granularity of the counting scheme is an important modelling consideration.

If it is too granular, then it will assign an optimistic bias in regions of the state space where the network should be trusted

A 2-dimensional gridworld maze with a sparse reward in which the agent can move Up, Down, Left or Right.

The agent starts each episode at a fixed location and must traverse through the maze in order to find the goal which provides +10 reward and terminates the episode, all other rewards are 0.

The agent interacts with the maze for 250 timesteps before being reset.

Empty space is represented by a 0, walls are 1, the goal is 2 and the player is 3.

The state representation is a greyscaled image of the entire grid where each entry is divided by 3 to lie in [0, 1].

The shape of the state representation is: (24, 24, 1).

Once again the effect of each action is randomised at each state at the beginning of training.

Figure 11 shows the structure of the maze environment.

We follow the suggestions in (Machado et al., 2018b) and use the same environmental setup as used in (Burda et al., 2018) .

Specifically, we use stick actions with a probability of p = 0.25, a frame skip of 4 and do not show a terminal state on the loss of life.

In all experiments we set ?? = 0.99, use RMSProp with a learning rate of 0.0005 and scale the gradient norms during training to be at most 5.

The network used is a MLP with 2 hidden layers of 256 units and ReLU non-linearities.

We use 1 step Q-Learning.

Training lasts for 100k timesteps. is fixed at 0.01 for all methods except for -greedy DQN in which it is linearly decayed from 1 to 0.01 over {100, 50k, 100k} timesteps.

We train on a batch size of 64 after every timestep with a replay buffer of size 10k.

The target network is updated every 200 timesteps.

The embedding size used for the counts is 32.

We set ?? = 0.1 for the scale of the count-based intrinsic motivation.

For reward subtraction we consider subtracting {0.1, 1, 10} from the reward.

For an optimistic initialisation bias, we consider setting the final layer's bias to {0.1, 1, 10}. We consider both of the methods with and without count-based intrinsic motivation.

For OPIQ and its ablations we consider: M ??? {0.1, 0.5, 2, 10}, C action ??? {0.1, 1, 10}, C bootstrap ??? {0.01, 0.1, 1, 10}.

For all methods we run 20 independent runs across the cross-product of all relevant parameters considered.

We then sort them by the median test reward (largest area underneath the line) and report the median, lower and upper quartiles.

The best hyperparameters we found were:

DQN -greedy: Decay rate: 100 timesteps.

Optimistic Initialisation Bias: Bias: 1, Pseudocount intrinsic motivation: True.

Reward Subtraction: Constant to subtract: 1, Pseudocount intrinsic motivation:

False.

OPIQ: M: 0.5, C action : 1, C bootstrap : 1.

OPIQ without Optimistic Bootstrapping: M: 2, C action : 10.

OPIQ without Pseudocounts: M: 2, C action : 10, C bootstrap : 10.

For We use 3 step Q-Learning.

Training lasts for 1mil timesteps. is decayed linearly from 1 to 0.01 over 50k timesteps for all methods except for -greedy DQN in which it is linearly decayed from 1 to 0.01 over {100, 50k, 100k} timesteps.

We train on a batch of 64 after every timestep with a replay buffer of size 250k.

The target network is updated every 1000 timesteps.

The embedding dimension for the counts is 128.

For DQN + PC we consider ?? ??? {0.01, 0.1, 1, 10, 100}. For all other methods we set ?? = 0.1 as it performed best.

For reward subtraction we consider subtracting {0.1, 1, 10} from the reward.

For an optimistic initialisation bias, we consider setting the final layer's bias to {0.1, 1, 10}. Both methods utilise a count-based intrinsic motivation.

For OPIQ and its ablations we set M = 2 since it worked best in preliminary experiments.

We consider: C action ??? {0.1, 1, 10, 100}, C bootstrap ??? {0.01, 0.1, 1, 10}.

For the RND bonus we use the same architecture as the DQN for both the target and predictor networks, except the output is of size 128 instead of |A|.

We scale the squared error by ?? rnd ??? {0.001, 0.01, 0.1, 1, 10, 100}:

For DQN + DORA we use the same architecture for the E-network as the DQN.

We add a sigmoid non-linearity to the output and initialise the final layer's weights and bias to 0 as described in (Choshen et al., 2018) .

We sweep across the scale of the intrinsic reward ?? dora ??? {}.

For DQN + DORA OA we use ?? dora = and sweep across ?? dora_action ??? {}.

For BSP we use the following architecture: We use K = 10 different bootstrapped DQN heads, and sweep over ?? bsp ??? {0.1, 1, 3, 10, 30, 100}.

For all methods we run 8 independent runs across the cross-product of all relevant parameters considered.

We then sort them by the median episodic reward (largest area underneath the line) and report the median, lower and upper quartiles.

The best hyperparameters we found were: OPIQ without Optimistic Bootstrapping: M: 2, C action : 100.

OPIQ without Pseudocounts: M: 2, C action : 100, C bootstrap : 0.1.

The network used is the standard DQN used for Atari (Mnih et al., 2015; Bellemare et al., 2016) .

We use 3 step Q-Learning.

Training lasts for 12.5mil timesteps (50mil frames in Atari). is decayed linearly from 1 to 0.01 over 1mil timesteps.

We train on a batch of 32 after every 4th timestep with a replay buffer of size 1mil.

The target network is updated every 8000 timesteps.

For all methods we consider ?? mmc ??? {0.005, 0.01, 0.025}.

For DQN + PC we consider ?? ??? {0.01, 0.1, 1}.

For OPIQ and its ablations we set M = 2.

We consider: C action ??? {0.1, 1}, C bootstrap ??? {0.01, 0.1}, ?? ??? {0.01, 0.1}.

For the RND bonus we use the same architectures as in (Burda et al., 2018 ) (target network is smaller than the learned predictor network) except we use ReLU non-linearity.

The output is the same of size 512.

We scale the squared error by ?? rnd ??? {0.001, 0.01, 0.1, 1}:

For BSP we use the same architecture as in (Osband et al., 2018) .

We use K = 10 different bootstrapped DQN heads, and sweep over ?? bsp ??? {0.1, 1, 10, 100}.

For all methods we run 4 independent runs across the cross-product of all relevant parameters considered.

We then sort them by the median maximum episodic reward (largest area underneath the line) and report the median, lower and upper quartiles.

The best hyperparameters we found were: OPIQ without Optimistic Bootstrapping: M= 2, C action = 0.1, ?? mmc = 0.005.

We do a single gradient descent step on a minibatch of the 32 most recently visited states.

We also recompute the intrinsic rewards when sampling minibatches to train the DQN.

The intrinsic reward used for a state s, is the squared error between the predictor network and the target network ?? rnd ||predictor(s) ??? target(s)|| 2 2 .

DQN + DORA:

We train the E-values network using n-step SARSA (same n as the DQN) with ?? E = 0.99.

We maintain a replay buffer of size (batch size * 4) and sample batch size elements to train every timestep.

The intrinsic reward we use is s,a) .

DQN + DORA OA: We train the DQN + DORA agent described above and additionally augment the Q-values used for action selection with

.

We train each Bootstrapped DQN head on all of the data from the replay buffer (as is done in (Osband et al., 2016; 2018) .

We normalise the gradients of the shared part of the network by 1/K, where K is the number of heads.

The output of each head is Q k + ?? bsp p k , where p k is a randomly initialised network (of the same architecture as Q k ) which is kept fixed throughout training.

?? bsp is a hyperparameter governing the scale of the prior regularisation.

For our experiments on Montezuma's Revenge we additionally mixed the 3 step Q-Learning target with the environmental rewards monte carlo return for the episode.

That is, the 3 step targets y t become:

If the episode hasn't finished yet, we used 0 for the monte carlo return.

Our implementation differs from (Bellemare et al., 2016; Ostrovski et al., 2017) in that we do not use the intrinsic rewards as part of the monte carlo return.

This is because we recompute the intrinsic rewards whenever we are using them as part of the targets for training, and recomputing all the intrinsic rewards for an entire episode (which can be over 1000 timesteps) is computationally prohibitive.

E FURTHER RESULTS E.1 RANDOMISED CHAIN Figure 12 : The number of distinct states visited over training for the chain environment.

The median across 20 seeds is plotted and the 25%-75% quartile is shown shaded.

We can see that OPIQ and ablations explore the environment much more quickly than the count-based baselines.

The ablation without optimistic bootstrapping exhibits significantly more variance than the other ablations, showing the importance of optimism during bootstrapping.

On this simple task the ablation without count-based intrinsic motivation performs on par with the full OPIQ.

This is most likely due to the simpler nature of the environment that makes propagating rewards much easier than the Maze.

The importance of directed exploration is made abundantly clear by the -greedy baseline that fails to explore much of the environment.

Figure 13 compares OPIQ with differing values of M .

We can clearly see that a small value of 0.1 results in insufficient exploration, due to the over-exploration of already visited state-action pairs.

Additionally if M is too large then the rate of exploration suffers due to the decreased optimism.

On this task we found that M = 0.5 performed best, but on the harder Maze environment we found that M = 2 was better in preliminary experiments.

E.2 MAZE Figure 14 : The Q + -values OPIQ used during bootstrapping with C bootstrap = 0.01.

Figure 14 shows the values used during bootstrapping for OPIQ.

These Q-values show optimism near the novel state-action pairs which provides an incentive for the agent to return to this area of the state space.

E.3 MONTEZUMA'S REVENGE

To emphasise the necessity of optimistic Q-value estimates during exploration, we analyse the simple failure case for pessimistically initialised greedy Q-learning provided in the introduction.

We use Algorithm 1, but use Q instead of Q + for action selection.

We will assume the agent will act greedily with respect to its Q-value estimates and break ties uniformly:

Consider the single state MDP in Figure 17 with H = 1.

We use this MDP to show that with 0.5 probability pessimistically initialised greedy Q-learning never finds the optimal policy.

The agent receives a reward of +1 for selecting the right action and 0.1 otherwise.

Therefore the optimal policy is to select the right action.

Now consider the first episode: ???a, Q 1 (s, a) = 0.

Thus, the agent selects an action at random with uniform probability.

If it selects the left action, it updates:

Thus, in the second episode it selects the left action again, since Q 1 (s, L) > 0 = Q 1 (s, R).

Our estimate of Q 1 (s, L) never drops below 0.1, and so the right action is never taken.

Thus, with probability of 1 2 it never selects the correct action (also a linear regret of 0.9T ).

This counterexample applies for any non-negative intrinsic motivation (including no intrinsic motivation), and is unaffected if we utilise optimistic bootstrapping or not.

Despite introducing an extra optimism term with a tunable hyperparameter M , OPIQ still requires the intrinsic motivation term b T i to ensure it does not under-explore in stochastic environments.

We will prove that OPIQ without the intrinsic motivation term b T i does not satisfy Theorem 1.

Specifically we will show that there exists a 1 state, 2 action MDP with stochastic reward function such that for all M > 0 the probability of incurring linear regret is greater than the allowed failure probability p.

We choose to use stochastic rewards as opposed to stochastic transitions for a simpler proof.

Figure 18 : The parametrised MDP.

The MDP we will use is shown in Figure 18 , where ?? > 1 and a ??? (0, 1) s.t p < 1 ??? a. H = 1, S = 1 and A = 2.

The reward function for the left action is stochastic, and will return +1 reward with probability a and 0 otherwise.

The reward for the right action is always a/??.

Let p > 0, the probability with which we are allowed to incur a total regret not bounded by the theorem.

OPIQ cannot depend on the value of ?? or a as they are unknown.

) .

OPIQ will recover the sub-optimal policy of taking the right action if every time we take the left action we receive a 0 reward.

This will happen since our Q + -value estimate for the left action will eventually drop below the Q-value estimate for the right action which is a/?? > 0.

The sup-optimal policy will incur a linear regret, which is not bounded by the theorem.

Our probability of failure is at least (1 ??? a) R , where R is the number of times we select the left action, which decreases as R increases.

This corresponds to receiving a 0 reward for every one of the R left transitions we take.

Note that (1 ??? a)

R is an underestimate of the probability of failure.

For the first 2 episodes we will select both actions, and with probability (1 ??? a) the left action will return 0 reward.

Our Q-values will then be:

It is possible to take the left action as long as

since the optimistic bonus for the right action decays to 0.

This then provides a very loose upper bound for R as ( ?? a ) 1/M , which then leads to a further underestimation of the probability of failure.

Assume for a contradiction that (1 ??? a) R < p:

This provides our contradiction as we choose ?? s.t M > log(??/a)/ log(log(p)/ log(1 ??? a)).

We can always pick such a ?? because log(??/a) can get arbitrarily close to 0.

So our probability of failure (of which (1 ??? a) R is a severe underestimate) is greater than the allowed probability of failure p.

Theorem 1.

For any p ??? (0, 1) , with probability at least 1 ??? p the total regret of Q + is at most

OPIQ is heavily based on UCB-H (Jin et al., 2018) , and as such the proof very closely mirrors its proof except for a few minor differences.

For completeness, we reproduce the entirety of the proof with the minor adjustments required for our scheme.

The proof is concerned with bounding the regret of the algorithm after K episodes.

The algorithm we follow takes as input the value of K, and changes the magnitudes of b T N based on it.

We will make use of a corollary to Azuma's inequality multiple times during the proof.

Theorem 2. (Azuma's Inequality).

Let Z 0 , ..., Z n be a martingale sequence of random variables such that ???i???c i : |Z i ??? Z i???1 | < c i almost surely, then:

Corollary 1.

Let Z 0 , ..., Z n be a martingale sequence of random variables such that ???i???c i : |Z i ??? Z i???1 | < c i almost surely, then with probability at least 1 ??? ??:

(1 ??? ?? j ) The following properties hold for ?? i N :

Lemma 2.

Adapted slightly from (Jin et al., 2018) Define V For any (s, a, t) ??? S ?? A ?? [H], episode k ??? K and N = N (s, a, t).

Suppose (s, a) was previously taken at step t of episodes k 1 , ..., k N < k.

Then:

Proof.

We have the following recursive formula for Q + at episode k and timestep t:

We can produce a similar formula for Q * :

From the Bellman Optimality Equation

By definition of P

and, with probability at least 1 ??? ??, the following holds simultaneously for all (s, a, t, k) where N = N (s, a, t) and k 1 , ..., k N < k are the episodes where (s, a) was taken at step t.

Proof.

For each fixed (s, a, t) ??? S ?? A ?? [H], let k 0 = 0 and k i = min({k ??? [K]|k > k i???1 , (s k t , a k t ) = (s, a)} ??? {K + 1}) k i is then the episode at which (s, a) was taken at step t for the ith time, or k i = K + 1 if it has been taken fewer than i times.

Then the random variable k i is a stopping time.

Let F i be the ??-field generated by all the random variables until episode k i step t.

Then by Azuma's Inequality we have that with probability at least 1 ??? 2??/(SAHK)

Then by a Union bound over all ?? ??? [K], we have that with probability at least 1 ??? 2??/(SAH):

Since From Lemma 1 we have that

Since inequality equation 9 holds for all fixed ?? ??? [K] uniformly, it also holds for a random variable ?? ?? = N = N k (s, a, t) ??? K. Also note that 1[k i ??? K] = 1 for all i ??? N .

We can then additionally apply a union bound over all s ??? S, a ??? A, t ??? [H] to give:

which holds with probability 1 ??? 2?? for all (s, a, t, k) ??? S ?? A ?? [H] ?? [K].

We then rescale ?? to ??/2.

By Lemma 1 we have that b

Note on stochastic rewards: We have assumed so far that the reward function is deterministic for a simpler presentation of the proof.

If we allowed for a stochastic reward function, the previous lemmas can be easily adapted to allow for it.

Lemma 2 would give us: We will now prove that our algorithm is provably efficient.

We will do this by showing it has sub-linear regret, following closely the proof of Theorem 1 from (Jin et al., 2018) .

Theorem 1.

For any p ??? (0, 1) , with probability at least 1 ??? p the total regret of Q + is at most O( H 4 SAT log(SAT /p)) for M ??? 1 and at most O(H 1+M SAT 1???M + H 4 SAT log(SAT /p)) for 0 < M < 1.

@highlight

We augment the Q-value estimates with a count-based bonus that ensures optimism during action selection and bootstrapping, even if the Q-value estimates are pessimistic.