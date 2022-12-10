Exploration is a fundamental aspect of Reinforcement Learning, typically implemented using stochastic action-selection.

Exploration, however, can be more efficient if directed toward gaining new world knowledge.

Visit-counters have been proven useful both in practice and in theory for directed exploration.

However, a major limitation of counters is their locality.

While there are a few model-based solutions to this shortcoming, a model-free approach is still missing.

We propose $E$-values, a generalization of counters that can be used to evaluate the propagating exploratory value over state-action trajectories.

We compare our approach to commonly used RL techniques, and show that using $E$-values improves learning and performance over traditional counters.

We also show how our method can be implemented with function approximation to efficiently learn continuous MDPs.

We demonstrate this by showing that our approach surpasses state of the art performance in the Freeway Atari 2600 game.

"

If there's a place you gotta go -I'm the one you need to know."(Map, Dora The Explorer)We consider Reinforcement Learning in a Markov Decision Process (MDP).

An MDP is a fivetuple M = (S, A, P, R, γ) where S is a set of states and A is a set of actions.

The dynamics of the process is given by P (s |s, a) which denotes the transition probability from state s to state s following action a. Each such transition also has a distribution R (r|s, a) from which the reward for such transitions is sampled.

Given a policy π : S → A, a function -possibly stochastic -deciding which actions to take in each of the states, the state-action value function Q π : S × A → R satisfies: r,s ∼R×P (·|s,a) [r + γQ π (s , π (s ))] DISPLAYFORM0 where γ is the discount factor.

The agent's goal is to find an optimal policy π * that maximizes Q π (s, π (s)).

For brevity, Q π * Q * .

There are two main approaches for learning π * .

The first is a model-based approach, where the agent learns an internal model of the MDP (namely P and R).

Given a model, the optimal policy could be found using dynamic programming methods such as Value Iteration BID19 .

The alternative is a model-free approach, where the agent learns only the value function of states or state-action pairs, without learning a model BID5 1 .The ideas put forward in this paper are relevant to any model-free learning of MDPs.

For concreteness, we focus on a particular example, Q-Learning BID23 BID19 .

Q-Learning is a common method for learning Q * , where the agent iteratively updates its values of Q (s, a) by performing actions and observing their outcomes.

At each step the agent takes action a t then it is transferred from s t to s t+1 and observe reward r.

Then it applies the update rule regulated by a learning rate α: Q (s t , a t ) ← (1 − α) Q (s t , a t ) + α r + γ max a Q (s t+1 , a) .

Balancing between Exploration and Exploitation is a major challenge in Reinforcement Learning.

Seemingly, the agent may want to choose the alternative associated with the highest expected reward, a behavior known as exploitation.

However, in that case it may fail to learn that there are better options.

Therefore exploration, namely the taking of new actions and the visit of new states, may also be beneficial.

It is important to note that exploitation is also inherently relevant for learning, as we want the agent to have better estimations of the values of valuable state-actions and we care less about the exact values of actions that the agent already knows to be clearly inferior.

Formally, to guarantee convergence to Q * , the Q-Learning algorithm must visit each state-action pair infinitely many times.

A naive random walk exploration is sufficient for converging asymptotically.

However, such random exploration has two major limitations when the learning process is finite.

First, the agent would not utilize its current knowledge about the world to guide its exploration.

For example, an action with a known disastrous outcome will be explored over and over again.

Second, the agent would not be biased in favor of exploring unvisited trajectories more than the visited ones -hence "wasting" exploration resources on actions and trajectories which are already well known to it.

A widely used method for dealing with the first problem is the -greedy schema BID19 , in which with probability 1 − the agent greedily chooses the best action (according to current estimation), and with probability it chooses a random action.

Another popular alternative, emphasizing the preference to learn about actions associated with higher rewards, is to draw actions from a Boltzmann Distribution (Softmax) over the learned Q values, regulated by a Temperature parameter.

While such approaches lead to more informed exploration that is based on learning experience, they still fail to address the second issue, namely they are not directed BID20 towards gaining more knowledge, not biasing actions in the direction of unexplored trajectories.

Another important approach in the study of efficient exploration is based on Sample Complexity of Exploration as defined in the PAC-MDP literature BID6 .

Relevant to our work is Delayed Q Learning BID17 , a model-free algorithm that has theoretical PAC-MDP guarantees.

However, to ensure these theoretical guarantees this algorithm uses a conservative exploration which might be impractical (see also BID7 and Appendix B).

In order to achieve directed exploration, the estimation of an exploration value of the different stateactions (often termed exploration bonus) is needed.

The most commonly used exploration bonus is based on counting (Thrun, 1992) -for each pair (s, a), store a counter C (s, a) that indicates how many times the agent performed action a at state s so far.

Counter-based methods are widely used both in practice and in theory BID7 BID16 BID3 BID2 .

Other options for evaluating exploration include recency and value difference (or error) measures BID20 BID21 .

While all of these exploration measures can be used for directed exploration, their major limitation in a model-free settings is that the exploratory value of a state-action pair is evaluated with respect only to its immediate outcome, one step ahead.

It seems desirable to determine the exploratory value of an action not only by how much new immediate knowledge the agent gains from it, but also by how much more new knowledge could be gained from a trajectory starting with it.

The goal of this work is to develop a measure for such exploratory values of state-action pairs, in a model-free settings.

The challenge discussed in 1.2 is in fact similar to that of learning the value functions.

The value of a state-action represents not only the immediate reward, but also the temporally discounted sum of expected rewards over a trajectory starting from this state and action.

Similarly, the "exploration-value" of a state-action should represent not only the immediate knowledge gained but also the expected future gained knowledge.

This suggests that a similar approach to that used for value-learning might be appropriate for learning the exploration values as well, using exploration bonus as the immediate reward.

However, because it is reasonable to require exploration bonus to decrease over repetitions of the same trajectories, a naive implementation would violate the Markovian property.

This challenge has been addressed in a model-based setting: The idea is to use at every step the current estimate of the parameters of the MDP in order to compute, using dynamic programming, the future exploration bonus BID8 .

However, this solution cannot be implemented in a model-free setting.

Therefore, a satisfying approach for propagating directed exploration in model-free reinforcement learning is still missing.

In this section, we propose such an approach.

We propose a novel approach for directed exploration, based on two parallel MDPs.

One MDP is the original MDP, which is used to estimate the value function.

The second MDP is identical except for one important difference.

We posit that there are no rewards associated with any of the state-actions.

Thus, the true value of all state-action pairs is 0.

We will use an RL algorithm to "learn" the "actionvalues" in this new MDP which we denote as E-values.

We will show that these E-values represent the missing knowledge and thus can be used for propagating directed exploration.

This will be done by initializing E-values to 1.

These positive initial conditions will subsequently result in an optimistic bias that will lead to directed exploration, by giving high estimations only to state-action pairs from which an optimistic outcome has not yet been excluded by the agent's experience.

Formally, given an MDP M = (S, A, P, R, γ) we construct a new MDP M = (S, A, P, 0, γ E ) with 0 denoting the identically zero function, and 0 ≤ γ E < 1 is a discount parameter.

The agent now learns both Q and E values concurrently, while initially E (s, a) = 1 for all s, a. Clearly, E * = 0.

However intuitively, the value of E (s, a) at a given timestep during training stands for the knowledge, or uncertainty, that the agent has regarding this state-action pair.

Eventually, after enough exploration, there is no additional knowledge left to discover which corresponds to E (s, a) → E * (s, a) = 0.For learning E, we use the SARSA algorithm BID13 BID19 which differs from Watkin's Q-Learning by being on-policy, following the update rule: DISPLAYFORM0 Where α E is the learning rate.

For simplicity, we will assume throughout the paper that α E = α.

Note that this learning rule updates the E-values based on E (s t+1 , a t+1 ) rather than max a E (s t+1 , a), thus not considering potentially highly informative actions which are never selected.

This is important for guaranteeing that exploration values will decrease when repeating the same trajectory (as we will show below).

Maintaining these additional updates doesn't affect the asymptotic space/time complexity of the learning algorithm, since it is simply performing the same updates of a standard Q-Learning process twice.

The logarithm of E-Values can be thought of as a generalization of visit counters, with propagation of the values along state-action pairs.

To see this, let us examine the case of γ E = 0 in which there is no propagation from future states.

In this case, the update rule is given by: DISPLAYFORM0 So after being visited n times, the value of the state-action pair is (1 − α) n , where α is the learning rate.

By taking a logarithm transformation, we can see that log 1−α (E) = n. In addition, when s is a terminal state with one action, log 1−α (E) = n for any value of γ E .

When γ E > 0 and for non-terminal states, E will decrease more slowly and therefore log 1−α E will increase more slowly than a counter.

The exact rate will depend on the MDP, the policy and the specific value of γ E .

Crucially, for state-actions which lead to many potential states, each visit contributes less to the generalized counter, because more visits are required to exhaust the potential outcomes of the action.

To gain more insight, consider the MDP depicted in FIG0 left, a tree with the root as initial state and the leaves as terminal states.

If actions are chosen sequentially, one leaf after the other, we expect that each complete round of choices (which will result with k actual visits of the (s, start) pair) will be roughly equivalent to one generalized counter.

Simulation of this and other simple MDPs show that E-values behave in accordance with such intuitions (see FIG0 right).An important property of E-values is that they decrease over repetitions.

Formally, by completing a trajectory of the form s 0 , a 0 , . . . , s n , a n , s 0 , a 0 in the MDP, the maximal value of E (s i , a i ) will decrease.

To see this, assume that E (s i , a i ) was maximal, and consider its value after the update: DISPLAYFORM1 , we get that after the update, the value of E (s i , a i ) decreased.

For any non-maximal (s j , a j ), its value after the update is a convex combination of its previous value and γ E E (s k , a k ) which is not larger than its composing terms, which in turn are smaller than the maximal E-value.

The logarithm of E-values can be considered as a generalization of counters.

As such, algorithms that utilize counters can be generalized to incorporate E-values.

Here we consider two such generalizations.

In model-based RL, counters have been used to create an augmented reward function.

Motivated by this result, augmenting the reward with a counter-based exploration bonus has also been used in model-free RL BID15 BID1 .

E-Values can naturally generalize this approach, by replacing the standard counter with its corresponding generalized counter (log 1−α E).To demonstrate the advantage of using E-values over standard counters, we tested an -greedy agent with an exploration bonus of 1 log 1−α E added to the observed reward on the bridge MDP ( Figure 2 ).

To measure the learning progress and its convergence, we calculated the mean square error * on optimal policy per episode.

Convergence of -greedy on the short bridge environment (k = 5) with and without exploration bonuses added to the reward.

Note the logarithmic scale of the abscissa.

DISPLAYFORM0 , where the average is over the probability of state-action pairs when following the optimal policy π * .

We varied the value of γ E from 0 -resulting effectively in standard counters -to γ E = 0.9.

Our results (Figure 3) show that adding the exploration bonus to the reward leads to faster learning.

Moreover, the larger the value of γ E in this example the faster the learning, demonstrating that generalized counters significantly outperforming standard counters.

Another way in which counters can be used to assist exploration is by adding them to the estimated Q-values.

In this framework, action-selection is a function not only of the Q-values but also of the counters.

Several such action-selection rules have been proposed BID20 BID9 BID7 ).

These usually take the form of a deterministic policy that maximizes some combination of the estimated Q-value with a counter-based exploration bonus.

It is easy to generalize such rules using E-values -simply replace the counters C by the generalized counters log 1−α (E).

Here, we consider a special family of action-selection rules that are derived as deterministic equivalents of standard stochastic rules.

Stochastic action-selection rules are commonly used in RL.

In their simple form they include rules such as the -greedy or Softmax exploration described above.

In this framework, exploratory behavior is achieved by stochastic action selection, independent of past choices.

At first glance, it might be unclear how E-values can contribute or improve such rules.

We now turn to show that, by using counters, for every stochastic rule there exist equivalent deterministic rules.

Once turned to deterministic counter-based rules, it is again possible improve them using E-values.

The stochastic action-selection rules determine the frequency of choosing the different actions in the limit of a large number of repetitions, while abstracting away the specific order of choices.

This fact is a key to understanding the relation between deterministic and stochastic rules.

An equivalence of two such rules can only be an in-the-limit equivalence, and can be seen as choosing a specific realization of sample from the distribution.

Therefore, in order to derive a deterministic equivalent of a given stochastic rule, we only have to make sure that the frequencies of actions selected under both rules are equal in the limit of infinitely many steps.

As the probability for each action is likely to depend on the current Q-values, we have to consider fixed Q-values to define this equivalence.

We prove that given a stochastic action-selection rule f (a|s), every deterministic policy that does not choose an action that was visited too many times until now (with respect to the expected number according to the probability distribution) is a determinization of f .

Formally, lets assume that given a certain Q function and state s we wish a certain ratio between different choices of actions a ∈ A to hold.

We denote the frequency of this ratio f Q (a|s).

For brevity we assume s and Q are constants and denote f Q (a|s) = f (a).

We also assume a counter C (s, a) is kept denoting the number of choices of a in s. For brevity we denote C (s, a) = C (a) and a C (s, a) = C. When we look at the counters after T steps we use subscript C T (a).

Following this notation, note that C T = T .

Theorem 3.1.

For any sub-linear function b (t) and for any deterministic policy which chooses at step T an action a such that DISPLAYFORM0 Proof.

For a full proof of the theorem see Appendix A in the supplementary materialsThe result above is not a vacuous truth -we now provide two possible determinization rules that achieves it.

One rule is straightforward from the theorem, using b = 0, choosing arg min a C(a)C − f (a).

Another rule follows the probability ratio between the stochastic policy and the empirical distribution: arg max a f (a) C(a) .

We denote this determinization LLL, because when generalized counters are used instead of counters it becomes arg max a logf (s, a) − loglog 1−α E (s, a).

Now we can replace the visit counters C (s, a) with the generalized counters log 1−α (E (s, a)) to create Directed Outreaching Reinforcement Action-Selection -DORA the explorer.

By this, we can transform any stochastic or counter-based action-selection rule into a deterministic rule in which exploration propagates over the states and the expected trajectories to follow.

Input: Stochastic action-selection rule f , learning rate α, Exploration discount factor γ E initialize Q (s, a) = 0, E (s, a) = 1; foreach episode do init s; while not terminated do Choose a = arg max x log f Q (x|s) − log log 1−α E (s, x); Observe transitions (s, a, r, s , a ); DISPLAYFORM1 Algorithm 1: DORA algorithm using LLL determinization for stochastic policy f

To test this algorithm, the first set of experiments were done on Bridge environments of various lengths k (Figure 2) .

We considered the following agents: -greedy, Softmax and their respective LLL determinizations (as described in 3.2.1) using both counters and E-values.

In addition, we compared a more standard counter-based agent in the form of a UCB-like algorithm BID0 following an action-selection rule with exploration bonus of log t C .

We tested two variants of this algorithm, using ordinary visit counters and E-values.

Each agent's hyperparameters ( and temperature) were fitted separately to optimize learning.

For stochastic agents, we averaged the results over 50 trials for each execution.

Unless stated otherwise, γ E = 0.9.We also used a normalized version of the bridge environment, where all rewards are between 0 and 1, to compare DORA with the Delayed Q-Learning algorithm BID17 .Our results FIG2 demonstrate that E-value based agents outperform both their counter-based and their stochastic equivalents on the bridge problem.

As shown in FIG2 , Stochastic and counter-based -greedy agents, as well as the standard UCB fail to converge.

E-value agents are the first to reach low error values, indicating that they learn faster.

Similar results were achieved The success of E-values based learning relative to counter based learning implies that the use of E-values lead to more efficient exploration.

If this is indeed the case, we expect E-values to better represent the agent's missing knowledge than visit counters during learning.

To test this hypothesis we studied the behavior of an E-value LLL Softmax on a shorter bridge environment (k = 5).

For a given state-action pair, a measure of the missing knowledge is the normalized distance between its estimated value (Q) and its optimal-policy value (Q * ).

We recorded C, log 1−α (E) and Q−Q * Q * for each s, a at the end of each episode.

Generally, this measure of missing knowledge is expected to be a monotonously-decreasing function of the number of visits (C).

This is indeed true, as depicted in FIG3 (left).

However, considering all state-action pairs, visit counters do not capture well the amount of missing knowledge, as the convergence level depends not only on the counter but also on the identity of the state-action it counts.

By contrast, considering the convergence level as a function of the generalized counter ( FIG3 , right) reveals a strikingly different pattern.

Independently of the state-action identity, the convergence level is a unique function of the generalized counter.

These results demonstrate that generalized counters are a useful measure of the amount of missing knowledge.

So far we discussed E-values in the tabular case, relying on finite (and small) state and action spaces.

However, a main motivation for using model-free approach is that it can be successfully applied in large MDPs where tabular methods are intractable.

In this case (in particular for continuous MDPs), achieving directed exploration is a non-trivial task.

Because revisiting a state or a state-action pair is unlikely, and because it is intractable to store individual values for all state-action pairs, counterbased methods cannot be directly applied.

In fact, most implementations in these cases adopt simple exploration strategies such as -greedy or softmax BID1 .There are standard model-free techniques to estimate value function in function-approximation scenarios.

Because learning E-values is simply learning another value-function, the same techniques can be applied for learning E-values in these scenarios.

In this case, the concept of visit-countor a generalized visit-count -will depend on the representation of states used by the approximating function.

To test whether E-values can serve as generalized visit-counters in the function-approximation case, we used a linear approximation architecture on the MountainCar problem (Moore, 1990) (Appendix C).

To dissociate Q and E-values, actions were chosen by an -greedy agent independently of Evalues.

As shown in Appendix C, E-values are an effective way for counting both visits and generalized visits in continuous MDPs.

For completeness, we also compared the performance of LLL agents to stochastic agents on a sparse-reward MountainCar problem, and found that LLL agents learns substantially faster than the stochastic agents (Appendix D).

To show our approach scales to complex problems, we used the Freeway Atari 2600 game, which is known as a hard exploration problem BID1 .

We trained a neural network with two streams to predict the Q and E-values.

First, we trained the network using standard DQN technique BID10 , which ignores the E-values.

Second, we trained the network while adding an exploration bonus of β √ − log E to the reward (In all reported simulations, β = 0.05).

In both cases, action-selection was performed by an -greedy rule, as in BID1 .Note that the exploration bonus requires 0 < E < 1.

To satisfy this requirement, we applied a logistic activation fucntion on the output of the last layer of the E-value stream, and initialized the weights of this layer to 0.

As a result, the E-values were initialized at 0.5 and satisfied 0 < E < 1 throughout the training.

In comparison, no non-linearity was applied in the last layer of the Q-value stream and the weights were randmoly initialized.

We compared our approach to a DQN baseline, as well as to the density model counters suggested by BID1 .

The baseline used here does not utilize additional enhancements (such as Double DQN and Monte-Carlo return) which were used in BID1 .

Our results, depicted in FIG4 , demonstrate that the use of E-values outperform both DQN and density model counters baselines.

In addition, our approach results in better performance than in BID1 (with the mentioned enhancements), converging in approximately 2 · 10 6 steps, instead of 10 · 10 6 steps 2 .

The idea of using reinforcement-learning techniques to estimate exploration can be traced back to BID15 and BID9 who also analyzed propagation of uncertainties and exploration values.

These works followed a model-based approach, and did not fully deal with the problem of non-Markovity arising from using exploration bonus as the immediate reward.

A related approach was used by BID8 , where exploration was investigated by information-theoretic measures.

Such interpretation of exploration can also be found in other works BID14 ; BID18 BID4 ).Efficient exploration in model-free RL was also analyzed in PAC-MDP framework, most notably the Delayed Q Learning algorithm by BID17 .

For further discussion and comparison of our approach with Delayed Q Learning, see 1.1 and Appendix B.In terms of generalizing Counter-based methods, there has been some works on using counter-like notions for exploration in continuous MDPs BID12 .

A more direct attempt was recently proposed by BID1 .

This generalization provides a way to implement visit counters in large, continuous state and action spaces by using density models.

Our generalization is different, as it aims first on generalizing the notion of visit counts themselves, from actual counters to "propagating counters".

In addition, our approach does not depend on any estimated model -which might be an advantage in domains for which good density models are not available.

Nevertheless, we believe that an interesting future work will be comparing between the approach suggested by BID1 and our approach, in particular for the case of γ E = 0.

The proof for the determinization mentioned in the paper is achieved based on the following lemmata.

Lemma A.1.

The absolute sum of positive and negative differences between the empiric distribution (deterministic frequency) and goal distribution (non-deterministic frequency) is equal.

DISPLAYFORM0 Figure 7: Normalized MSE between Q and Q * on optimal policy per episode.

Convergence of E-value LLL and Delayed Q-Learning on, normalized bridge environment (k = 15).

MSE was noramlized for each agent to enable comparison.

Because Delayed Q learning initializes its values optimistically, which result in a high MSE, we normalized the MSE of the two agents (separately) to enable comparison.

Notably, to achieve this performance by the Delayed Q Learning, we had to manually choose a low value for m (in Figure 7 , m = 10), the hyperparameter regulating the number of visits required before any update.

This is an order of magnitude smaller than the theoretical value required for even moderate PAC-requirements in the usual notion of , δ, such m also implies learning in orders of magnitudes slower.

In fact, for this limit of m → 1 the algorithm is effectively quite similar to a "Vanilla" Q-Learning with an optimistic initialization, which is possible due to the assumption made by the algorithm that all rewards are between 0 and 1.

In fact, several exploration schemes relying on optimism in the face of uncertainty were proposed BID22 ).

However, because our approach separate reward values and exploratory values, we are able to use optimism for the latter without assuming any prior knowledge about the first -while still achieving competitive results to an optimistic initialization based on prior knowledge.

To gain insight into the relation between E-values and number of visits, we used the linearapproximation architecture on the MountainCar problem.

Note that when using E-values, they are generally correlated with visit counts both because visits result in update of the E-values through learning and because E-values affect visits through the exploration bonus (or action-selection rule).

To dissociate the two, Q-values and E-values were learned in parallel in these simulation, but actionselection was independent of the E-values.

Rather, actions were chosen by an -greedy agent.

To estimate visit-counts, we recorded the entire set of visited states, and computed the empirical visits histogram by binning the two-dimensional state-space.

For each state, its visit counter estimator C (s) is the value of the matching bin in the histogram for this state.

In addition, we recorded the learned model (weights vector for E-values) and computed the E-values map by sampling a state for each bin, and calculating its E-values using the model.

For simplicity, we consider here the resolution of states alone, summing over all 3 actions for each state.

That is, we compareC (s) to a log 1−α E (s, a) = C E (s).

FIG5 depicts the empirical visits histogram (left) and the estimated E-values for the case of γ E = 0 after the complete training.

The results of the analysis show that, roughly speaking, those regions in the state space that were more often visited, were also associated with a higher C E (s).

To better understand these results, we considered smaller time-windows in the learning process.

Specifically, FIG6 depicts the empirical visit histogram (left), and the corresponding C E (s) (right) in the first 10 episodes, in which visits were more centrally distributed.

FIG0 depicts the change in the empirical visit histogram (left), and change in the corresponding C E (s) (right) in the last 10 episodes of the training, in which visits were distributed along a spiral (forming an nearoptimal behavior).

These results demonstrate high similarity between visit-counts and the E-value representation of them, indicating that E-values are good proxies of visit counters.

The results depicted in Figures 9 and 10 were achieved with γ E = 0.

For γ E > 0, we expect the generalized counters (represented by E-values) to account not for standard visits but for "generalized visits", weighting the trajectories starting in each state.

We repeated the analysis of FIG0 for the case of γ E = 0.99.

Results, depicted in FIG0 , shows that indeed for terminal or nearterminal states (where position> 0.5) generalized visits, measured by difference in their generalized counters, are higher -comparing to far-from terminal states -than the empirical visits of these states (comparing to far-from terminal states).

To quantify the relation between visits and E-values, we densely sampled the (achievable) statespace to generate many examples of states.

For each sampled state, we computed the correlation coefficient between C E (s) andC (s) throughout the learning process (snapshots taken each 10 episodes).

The valuesC (s) were estimated by the empirical visits histogram (value of the bin corresponding to the sampled state) calculated based on visits history up to each snapshot.

FIG0 , depicting the histogram of correlation coefficients between the two measures, demonstrating strong positive correlations between empirical visit-counters and generalized counters represented by E-values.

These results indicate that E-values are an effective way for counting effective visits in continuous MDPs.

Note that the number of model parameters used to estimate E (s, a) in this case is much smaller than the size of the table we would have to use in order to track state-action counters in such binning resolution.

To test the performance of E-values based agents, simulations were performed using the MountainCar environment.

The version of the problem considered here is with sparse and delayed reward, meaning that there is a constant reward of 0 unless reaching a goal state which provides a reward of magnitude 1.

Episode length was limited to 1000 steps.

We used linear approximation with tilecoding features BID19 , learning the weights vectors for Q and E in parallel.

To guarantee that E-values are uniformly initialized and are kept between 0 and 1 throughout learning, we initialized the weights vector for E-values to 0 and added a logistic non-linearity to the results of the standard linear approximation.

In contrast, the Q-values weights vector was initialized at random, and there was no non-linearity.

We compared the performance of several agents.

The first two used only Q-values, with a softmax or an -greedy action-selection rules.

The other two agents are the DORA variants using both Q and E values, following the LLL determinization for softmax either with γ E = 0 or with γ E = 0.99.

Parameters for each agent (temperature and ) were fitted separately to maximize performance.

The results depicted in FIG0 demonstrate that using E-values with γ E > 0 lead to better performance in the MountainCar problem In addition we tested our approach using (relatively simple) neural networks.

We trained two neural networks in parallel (unlike the two-streams single network used for Atari simulations), for predicting Q and E values.

In this architecture, the same technique of 0 initializing and a logistic non-linearity was applied to the last linear of the E-network.

Similarly to the linear approximation approach, E-values based agents outperform their -greedy and softmax counterparts (not shown).Figure 13: Probability of reaching goal on MountainCar (computed by averaging over 50 simulations of each agent), as a function of training episodes.

While Softmax exploration fails to solve the problem within 1000 episodes, LLL E-values agents with generalized counters (γ E > 0) quickly reach high success rates.

<|TLDR|>

@highlight

We propose a generalization of visit-counters that evaluate the propagating exploratory value over trajectories, enabling efficient exploration for model-free RL