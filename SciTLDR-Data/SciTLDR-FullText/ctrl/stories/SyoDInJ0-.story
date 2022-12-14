This paper formalises the problem of online algorithm selection in the context of Reinforcement Learning (RL).

The setup is as follows: given an episodic task and a finite number of off-policy RL algorithms, a meta-algorithm has to decide which RL algorithm is in control during the next episode so as to maximize the expected return.

The article presents a novel meta-algorithm, called Epochal Stochastic Bandit Algorithm Selection (ESBAS).

Its principle is to freeze the policy updates at each epoch, and to leave a rebooted stochastic bandit in charge of the algorithm selection.

Under some assumptions, a thorough theoretical analysis demonstrates its near-optimality considering the structural sampling budget limitations.

ESBAS is first empirically evaluated on a dialogue task where it is shown to outperform each individual algorithm in most configurations.

ESBAS is then adapted to a true online setting where algorithms update their policies after each transition, which we call SSBAS.

SSBAS is evaluated on a fruit collection task where it is shown to adapt the stepsize parameter more efficiently than the classical hyperbolic decay, and on an Atari game, where it improves the performance by a wide margin.

Reinforcement Learning (RL, BID18 ) is a machine learning framework for optimising the behaviour of an agent interacting with an unknown environment.

For the most practical problems, such as dialogue or robotics, trajectory collection is costly and sample efficiency is the main key performance indicator.

Consequently, when applying RL to a new problem, one must carefully choose in advance a model, a representation, an optimisation technique and their parameters.

Facing the complexity of choice, RL and domain expertise is not sufficient.

Confronted to the cost of data, the popular trial and error approach shows its limits.

We develop an online learning version (Gagliolo & Schmidhuber, 2006; BID1 of Algorithm Selection (AS, BID15 ; BID17 BID5 ).

It consists in testing several algorithms on the task and in selecting the best one at a given time.

For clarity, throughout the whole article, the algorithm selector is called a meta-algorithm, and the set of algorithms available to the meta-algorithm is called a portfolio.

The meta-algorithm maximises an objective function such as the RL return.

Beyond the sample efficiency objective, the online AS approach besides addresses four practical problems for online RL-based systems.

First, it improves robustness: if an algorithm fails to terminate, or outputs to an aberrant policy, it will be dismissed and others will be selected instead.

Second, convergence guarantees and empirical efficiency may be united by covering the empirically efficient algorithms with slower algorithms that have convergence guarantees.

Third, it enables curriculum learning: shallow models control the policy in the early stages, while deep models discover the best solution in late stages.

And four, it allows to define an objective function that is not an RL return.

A fair algorithm selection implies a fair budget allocation between the algorithms, so that they can be equitably evaluated and compared.

In order to comply with this requirement, the reinforcement algorithms in the portfolio are assumed to be off-policy, and are trained on every trajectory, regardless which algorithm controls it.

Section 2 provides a unifying view of RL algorithms, that allows information sharing between algorithms, whatever their state representations and optimisation techniques.

It also formalises the problem of online selection of off-policy RL algorithms.

Next, Section 3 presents the Epochal Stochastic Bandit AS (ESBAS), a novel meta-algorithm addressing the online off-policy RL AS problem.

Its principle relies on a doubling trick: it divides the time-scale into epochs of exponential length inside which the algorithms are not allowed to update their policies.

During each epoch, the algorithms have therefore a constant policy and a stochastic multi-armed bandit can be in charge of the AS with strong pseudo-regret theoretical guaranties.

A thorough theoretical analysis provides for ESBAS upper bounds.

Then, Section 4 evaluates ESBAS on a dialogue task where it outperforms each individual algorithm in most configurations.

Afterwards, in Section 5, ESBAS, which is initially designed for a growing batch RL setting, is adapted to a true online setting where algorithms update their policies after each transition, which we call SSBAS.

It is evaluated on a fruit collection task where it is shown to adapt the stepsize parameter more efficiently than the classical hyperbolic decay, and on Q*bert, where running several DQN with different network size and depth in parallel allows to improve the final performance by a wide margin.

Finally, Section 6 concludes the paper with prospective ideas of improvement.

Stochastic environment a(t) o(t + 1) r(t + 1)Figure 1: RL framework: after performing action a(t), the agent perceives observation o(t + 1) and receives reward r(t + 1).The goal of this section is to enable information sharing between algorithms, even though they are considered as black boxes.

We propose to share their trajectories expressed in a universal format: the interaction process.

Reinforcement Learning (RL) consists in learning through trial and error to control an agent behaviour in a stochastic environment: at each time step t ??? N, the agent performs an action a(t) ??? A, and then perceives from its environment a signal o(t) ??? ??? called observation, and receives a reward r(t) ??? R, bounded between R min and R max .

Figure 1 illustrates the RL framework.

This interaction process is not Markovian: the agent may have an internal memory.

In this article, the RL problem is assumed to be episodic.

Let us introduce two time scales with different notations.

First, let us define meta-time as the time scale for AS: at one meta-time ?? corresponds a meta-algorithm decision, i.e. the choice of an algorithm and the generation of a full episode controlled with the policy determined by the chosen algorithm.

Its realisation is called a trajectory.

Second, RL-time is defined as the time scale inside a trajectory, at one RL-time t corresponds one triplet composed of an observation, an action, and a reward.

Let E denote the space of trajectories.

A trajectory ?? ?? ??? E collected at meta-time ?? is formalised as a sequence of (observation, action, reward) triplets: ?? ?? = o ?? (t), a ?? (t), r ?? (t) t??? 1,|???? | ??? E, where |?? ?? | is the length of trajectory ?? ?? .

The objective is, given a discount factor 0 ??? ?? < 1, to generate trajectories with high discounted cumulative reward, also called return, and noted ??(?? ?? ) = |???? | t=1 ?? t???1 r ?? (t).

Since ?? < 1 and R is bounded, the return is also bounded.

The trajectory set at meta-time T is denoted by D T = {?? ?? } ?? ??? 1,T ??? E T .

A sub-trajectory of ?? ?? until RL-time t is called the history at RL-time t and written ?? ?? (t) with t ??? |?? ?? |.

The history records what happened in episode ?? ?? until RL-time t: ?? ?? (t) = o ?? (t ), a ?? (t ), r ?? (t ) t ??? 1,t ??? E.The goal of each RL algorithm ?? is to find a policy ?? * : E ??? A which yields optimal expected returns.

Such an algorithm ?? is viewed as a black box that takes as an input a trajectory set D ??? E + , where E + is the ensemble of trajectory sets of undetermined size: DISPLAYFORM0 , and that outputs a policy ?? ?? D .

Consequently, a RL algorithm is formalised as follows: ?? : DISPLAYFORM1 Such a high level definition of the RL algorithms allows to share trajectories between algorithms: a trajectory as a sequence of observations, actions, and rewards can be interpreted by any algorithm in its own decision process and state representation.

For instance, RL algorithms classically rely on an MDP defined on a explicit or implicit state space representation DISPLAYFORM2 on the trajectories projected on its state space representation.

Off-policy RL optimisation techniques compatible with this approach are numerous in the literature BID20 Ernst et al., 2005; BID10 .

As well, any post-treatment of the state set, any alternative decision process BID9 , and any off-policy algorithm may be used.

The algorithms are defined here as black boxes and the considered meta-algorithms will be indifferent to how the algorithms compute their policies, granted they satisfy the off-policy assumption.

Pseudo-code 1: Online RL AS setting DISPLAYFORM0 Generate trajectory ?? ?? with policy ?? DISPLAYFORM1 The online learning approach is tackled in this article: different algorithms are experienced and evaluated during the data collection.

Since it boils down to a classical exploration/exploitation trade-off, multi-armed bandit (Bubeck & Cesa-Bianchi, 2012) have been used for combinatorial search AS (Gagliolo & Schmidhuber, 2006; BID1 and evolutionary algorithm meta-learning (Fialho et al., 2010) .

The online AS problem for off-policy RL is novel and we define it as follows: DISPLAYFORM2 is the current trajectory set;??? P = {?? k } k??? 1,K is the portfolio of off-policy RL algorithms;??? ?? : E ??? R is the objective function, generally set as the RL return.

Pseudo-code 1 formalises the online RL AS setting.

A meta-algorithm is defined as a function from a trajectory set to the selection of an algorithm: ?? : E + ??? P. The meta-algorithm is queried at each meta-time ?? = |D ?? ???1 |+1, with input D ?? ???1 , and it ouputs algorithm ?? (D ?? ???1 ) = ??(?? ) ??? P controlling with its policy ?? ??(?? ) D?????1 the generation of the trajectory ?? ?? in the stochastic environment.

Figure 2 illustrates the algorithm with a diagram flow.

The final goal is to optimise the cumulative expected return.

It is the expectation of the sum of rewards obtained after a run of T trajectories: DISPLAYFORM3 DISPLAYFORM4 Return ??(?? ?? ): DISPLAYFORM5 Figure 2: Algorithm selection for reinforcement learning flow diagram expectations.

The outside expectation E ?? assumes the meta-algorithm ?? fixed and averages over the trajectory set generation and the corresponding algorithms policies.

The inside expectation E?? assumes the policy fixed and averages over its possible trajectories in the stochastic environment.

Nota bene: there are three levels of decision: meta-algorithm ?? selects algorithm ?? that computes policy ?? that is in control.

In this paper, the focus is at the meta-algorithm level.

In this paper, we focus on sample efficiency, where a sample is meant to be a trajectory.

This is motivated by the following reasons.

First, in most real-world systems, the major regret is on the task failure.

The time expenditure is only a secondary concern that is already assessed by the discount factor dependency in the return.

Second, it would be inconsistent to consider regret on a different time scale as the algorithm selection.

Also, policy selection on non-episodic RL is known as a very difficult task where state-of-the-art algorithms only obtain regrets of the order of O( ??? T log(T )) on stationary policies Azar et al. (2013) .

Third, the regret on the decision steps cannot be assessed, since the rewards are discounted in the RL objective function.

And finally, the bandit rewards (defined as the objective function in Section 2.2) may account for the length of the episode.

In order to evaluate the meta-algorithms, let us formulate two additional notations.

First, the optimal expected return E?? * ??? is defined as the highest expected return achievable by a policy of an algorithm in portfolio P. Second, for every algorithm ?? in the portfolio, let us define ?? ?? as its canonical metaalgorithm, i.e. the meta-algorithm that always selects algorithm ??: ????? , ?? ?? (?? ) = ??.

The absolute pseudo-regret ?? ?? abs (T ) defines the regret as the loss for not having controlled the trajectory with an optimal policy: DISPLAYFORM0 It is worth noting that an optimal meta-algorithm will unlikely yield a null regret because a large part of the absolute pseudo-regret is caused by the sub-optimality of the algorithm policies when the trajectory set is still of limited size.

Indeed, the absolute pseudo-regret considers the regret for not selecting an optimal policy: it takes into account both the pseudo-regret of not selecting the best algorithm and the pseudo-regret of the algorithms for not finding an optimal policy.

Since the metaalgorithm does not interfere with the training of policies, it ought not account for the pseudo-regret related to the latter.

Related to AS for RL, BID16 use meta-learning to tune a fixed RL algorithm in order to fit observed animal behaviour, which is a very different problem to ours.

In Cauwet et al. FORMULA9 ; BID8 , the RL AS problem is solved with a portfolio composed of online RL algorithms.

The main limitation from these works relies on the fact that on-policy algorithms were used, which prevents them from sharing trajectories among algorithms (Cauwet et al., 2015) .

Meta-learning specifically for the eligibility trace parameter has also been studied in BID21 .

BID19 study the learning process of RL algorithms and selects the best one for learning faster on a new task.

This work is related to batch AS.An intuitive way to solve the AS problem is to consider algorithms as arms in a multi-armed bandit setting.

The bandit meta-algorithm selects the algorithm controlling the next trajectory ?? and the objective function ??(??) constitutes the reward of the bandit.

The aim of prediction with expert advice is to minimise the regret against the best expert of a set of predefined experts.

When the experts learn during time, their performances evolve and hence the sequence of expert rewards is non-stationary.

The exponential weight algorithms BID3 Cesa-Bianchi & Lugosi, 2006) are designed for prediction with expert advice when the sequence of rewards of experts is generated by an oblivious adversary.

This approach has been extended for competing against the best sequence of experts by adding in the update of weights a forgetting factor proportional to the mean reward (see Exp3.S in BID3 ), or by combining Exp3 with a concept drift detector BID0 .The exponential weight algorithms have been extended to the case where the rewards are generated by any sequence of stochastic processes of unknown means (Besbes et al., 2014) .A recent article Graves et al. (2017) uses Exp3.S algorithm BID3 for curriculum learning Bengio et al. (2009) .

The drawback of adversarial approaches is that they lead to very conservative algorithms which has to work against an adversary.

For handeling non-stationarity of rewards, another way is to assume that the rewards generated by each arm are not i.i.d., but are governed by some more complex stochastic processes.

The stochastic bandit algorithm such as UCB can be extended to the case of switching bandits using a discount factor or a window to forget the past Garivier & Moulines (2011) .

Restless bandits BID22 ; BID13 assume that a Markov chain governs the reward of arms independently of whether the learner is played or not the arm.

These classes of bandit algorithms are not designed for experts that learn and hence evolve at each time step.

Our approach takes the opposite view of adversarial bandits: we design a stochastic algorithm specifically for curriculum learning based on the doubling trick.

This reduction of the algorithm selection problem into several stochastic bandit problems with doubling time horizon begins to favour fast algorithms, and then more efficient algorithms.

ESBAS description -To solve the off-policy RL AS problem, we propose a novel meta-algorithm called Epochal Stochastic Bandit AS (ESBAS).

Because of the non-stationarity induced by the algorithm learning, the stochastic bandit cannot directly select algorithms.

Instead, the stochastic bandit can choose fixed policies.

To comply with this constraint, the meta-time scale is divided into epochs inside which the algorithms policies cannot be updated: the algorithms optimise their policies only when epochs start, in such a way that the policies are constant inside each epoch.

This can be seen as a doubling trick.

As a consequence and since the returns are bounded, at each new epoch, the problem can rigorously be cast into an independent stochastic K-armed bandit ??, with K = |P|.

Data: D 0 , P, ??: the online RL AS setting DISPLAYFORM0 n kmax + 1 n kmax ??? n kmax + 1 and n ??? n + 1 end endThe ESBAS meta-algorithm is formally sketched in Pseudo-code 2 embedding UCB1 Auer et al. (2002a) as the stochastic Karmed bandit ??. The meta-algorithm takes as an input the set of algorithms in the portfolio.

Meta-time scale is fragmented into epochs of exponential size.

The ?? th epoch lasts 2 ?? metatime steps, so that, at meta-time ?? = 2 ?? , epoch ?? starts.

At the beginning of each epoch, the ESBAS meta-algorithm asks each algorithm in the portfolio to update their current policy.

Inside an epoch, the policy is never updated anymore.

At the beginning of each epoch, a new ?? instance is reset and run.

During the whole epoch, ?? selects at each meta-time step the algorithm in control of the next trajectory.

Theoretical analysis -ESBAS intends to minimise the regret for not choosing the algorithm yielding the maximal return at a given meta-time ?? .

It is short-sighted: it does not intend to optimise the algorithms learning.

We define the short-sighted pseudo-regret as follows: DISPLAYFORM1 The short-sighted pseudo-regret depends on the gaps ??? ?? ?? : the difference of expected return between the best algorithm during epoch ?? and algorithm ??.

The smallest non null gap at epoch ?? is noted ??? ??? ?? .

We write its limit when ?? tends to infinity with ??? ??? ??? .

Analysis relies on three assumptions that are formalised in Section B of the supplementary material.

First, more data is better data states that algorithms improve on average from having additional data.

Second, order compatibility assumes that, if a dataset enables to generate a better policy than another dataset, then, on average, adding new samples to both datasets should not change the dataset ordering.

Third and last, let us introduce and discuss more in depth the learning is fair assumption.

The fairness of budget distribution has been formalised in Cauwet et al. (2015) .

It is the property stating that every algorithm in the portfolio has as much resources as the others, in terms of computational time and data.

It is an issue in most online AS problems, since the algorithm that has been the most selected has the most data, and therefore must be the most advanced one.

A way to circumvent this issue is to select them equally, but, in an online setting, the goal of AS is precisely to select the best algorithm as often as possible.

Our answer is to require that all algorithms in the portfolio are learning off-policy, i.e. without bias induced by the behavioural policy used in the learning dataset.

By assuming that all algorithms learn off-policy, we allow information sharing Cauwet et al. FORMULA9 between algorithms.

They share the trajectories they generate.

As a consequence, we can assume that every algorithm, the least or the most selected ones, will learn from the same trajectory set.

Therefore, the control unbalance does not directly lead to unfairness in algorithms performances: all algorithms learn equally from all trajectories.

However, unbalance might still remain in the exploration strategy if, for instance, an algorithm takes more benefit from the exploration it has chosen than the one chosen by another algorithm.

For analysis purposes, we assumes the complete fairness of AS.Based on those assumptions, three theorems show that ESBAS absolute pseudo-regret can be expressed in function of the absolute pseudo-regret of the best canonical algorithm and ESBAS shortsighted pseudo-regret.

They also provide upper bounds on the ESBAS short-sighted pseudo-regret as a function of the order of magnitude of the gap ??? ??? ?? .

Indeed, the stochastic multi-armed bandit algorithms have bounds that are, counter-intuitively, inversely proportional to the gaps between the best arm and the other ones.

In particular if ??? ??? ?? tends to 0, the algorithm selection might prove to be difficult, depending on the order of magnitude of it tending to 0.

The full theoretical analysis can be found in the supplementary material, Section B. We provide here an intuitive overlook of its results.

TAB0 numerically reports those bounds for a two-fold portfolio, depending on the nature of the algorithms.

It must be read by line.

According to the first column: the order of magnitude of ??? ??? ?? , the ESBAS short-sighted pseudo-regret bounds are displayed in the second column, and the third and fourth columns display the ESBAS absolute pseudo-regret bounds also depending on the order of magnitude of the best canonical algorithm absolute pseudo-regret: ?? ?? *

Regarding the short-sighted upper bounds, the main result appears in the last line, when the algorithms converge to policies with different performance: ESBAS converges with a regret in O log 2 (T )/??? ??? ??? .

Also, one should notice that the bounds of the first two lines are obtained by summing the gaps: this means that the algorithms are perceived equally good and that their gap goes beyond the threshold of distinguishability.

This threshold is structurally at ??? ??? DISPLAYFORM0 The impossibility to determine which is the better algorithm is interpreted in Cauwet et al. FORMULA9 as a budget issue.

The meta-time necessary to distinguish through evaluation arms that are ??? ??? ?? apart takes ??(1/??? ???2 ?? ) meta-time steps.

If the budget is inferior, then we are under the distinguishability threshold and the best bounds are obtained by summing up the gaps.

As a consequence, if ??? ??? DISPLAYFORM1 .

However, the budget, i.e. the length of epoch ?? starting at meta-time T = 2 ?? , equals DISPLAYFORM2 can therefore be considered as the structural limit of distinguishability between the algorithms.

Additionally, the absolute upper bounds are logarithmic in the best case and still inferior to O( ??? T ) in the worst case, which compares favorably with those of discounted UCB and Exp3.S in O( T log(T )) and Rexp3 in O(T 2/3 ), or the RL with Policy Advice's regret bounds of O( ??? T log(T )) on stationary policies Azar et al. FORMULA9 (on non-episodic RL tasks).

DISPLAYFORM3 , and c DISPLAYFORM4

ESBAS is particularly designed for RL tasks when it is impossible to update the policy after every transition or episode.

Policy update is very costly in most real-world applications, such as dialogue systems (Khouzaimi et al., 2016) for which a growing batch setting is preferred BID6 .

ESBAS practical efficiency is therefore illustrated on a dialogue negotiation game BID7 ) that involves two players: the system p s and a user p u .

Their goal is to find an agreement among 4 alternative options.

At each dialogue, for each option ??, players have a private uniformly drawn cost ?? Table 2 of the supplementary material.

Figures 3a and 3b plot the typical curves obtained with ESBAS selecting from a portfolio of two learning algorithms.

On Figure 3a , the ESBAS curve tends to reach more or less the best algorithm in each point as expected.

Surprisingly, Figure 3b reveals that the algorithm selection ratios are not very strong in favour of one or another at any time.

Indeed, the variance in trajectory set collection makes simple better on some runs until the end.

ESBAS proves to be efficient at selecting the best algorithm for each run and unexpectedly obtains a negative relative pseudo-regret of -90.

Figures 3c and 3d plot the typical curves obtained with ESBAS selecting from a portfolio constituted of a learning algorithm and an algorithm with a deterministic and stationary policy.

ESBAS succeeds in remaining close to the best algorithm at each epoch and saves 5361 return value for not selecting the constant algorithm, but overall yields a regret for not using only the best algorithm.

ESBAS also performs well on larger portfolios of 8 learners (see Figure 3e ) with negative relative pseudo-regrets: ???10, even if the algorithms are, on average, almost selected uniformly as Figure 3f reveals.

Each individual run may present different ratios, depending on the quality of the trained policies.

ESBAS also offers some curriculum learning, but more importantly, early bad policies are avoided.

Algorithms with a constant policy do not improve over time and the full reset of the K-multi armed bandit urges ESBAS to unnecessarily explore again and again the same underachieving algorithm.

One easy way to circumvent this drawback is to use this knowledge and to not reset their arms.

By operating this way, when the learning algorithm(s) start(s) outperforming the constant one, ESBAS simply neither exploits nor explores the constant algorithm anymore.

Without arm reset for constant algorithms, ESBAS's learning curve follows perfectly the learning algorithm's learning curve when this one outperforms the constant algorithm and achieves strong negative relative pseudo-regrets.

Again, the interested reader may refer to Table 2 in supplementary material for the numerical results.

Still, another harmful phenomenon may happen: the constant algorithm overrides the natural exploration of the learning algorithm in the early stages, and when the learning algorithm finally outperforms the constant algorithm, its exploration parameter is already low.

This can be observed in experiments with constant algorithm of expected return inferior to 1, as reported in Table 2 .

We propose to adapt ESBAS to a true online setting where algorithms update their policies after each transition.

The stochastic bandit is now trained on a sliding window containing the last ?? /2 selections.

Even though the arms are not stationary over this window, the bandit eventually forgets the oldest arm pulls.

This algorithm is called SSBAS for Sliding Stochastic Bandit AS.

Despite the lack of theoretical convergence bounds, we demonstrate on two domains and two different meta-optimisation tasks that SSBAS impressively outperforming all algorithms in the portfolio by a wide margin.

The goal here is to demonstrate that SSBAS can perform efficient hyper-parameter optimisation on a simple tabular domain: a 5x5 gridworld problem (see Figure 4) , where the goal is to collect the fruits placed at each corner as fast as possible.

The episodes terminate when all fruits have been collected or after 100 transitions.

The objective function ?? used to optimise the stochastic bandit ?? is no longer the RL return, but the time spent to collect all the fruits (200 in case of it did not).

The agent has 18 possible positions and there are 2 4 ??? 1 = 15 non-terminal fruits configurations, resulting in 270 states.

The action set is A = {N, E, S, W }.

The reward function mean is 1 when eating a fruit, 0 otherwise.

The reward function is corrupted with a strong Gaussian white noise of variance ?? 2 = 1.

The portfolio is composed of 4 Q-learning algorithms varying from each other by their learning rates: {0.001, 0.01, 0.1, 0.5}. They all have the same linearly annealing ?? -greedy exploration.

The selection ratios displayed in 5 show that SSBAS selected the algorithm with the highest (0.5) learning rate in the first stages, enabling to propagate efficiently the reward signal through the visited states, then, over time preferentially chooses the algorithm with a learning rate of 0.01, which is less sensible to the reward noise, finally, SSBAS favours the algorithm with the finest learning rate (0.001).

After 1 million episodes, SSBAS enables to save half a transition per episode on average as compared to the best fixed learning rate value (0.1), and two transitions against the worst fixed learning rate in the portfolio (0.001).We compare SSBAS to the efficiency of a linearly annealing learning rate: 1/(1 + 0.0001?? ): SSBAS performs under 21 steps on average after 10 5 , while the linearly annealing learning rate algorithm still performs a bit over 21 steps after 10 6 steps.

This is because SSBAS can adapt the best performing learning rate over time.

We also compare SSBAS performance to Exp3.S's performance BID3 .

The analysis of the algorithm selection history shows that Exp3.S is too conservative and fails at efficiently select the shallowest algorithms in the beginning of the learning (number of steps at the 10000 th episode: 28.3 for SSBAS vs 39.1 for Exp3.S), producing trajectories of lesser quality and therefore critically delaying the general training of all algorithms (number of steps at the 100000 th episode: 20.9 for SSBAS vs 22.5 for Exp3.S).

Overall, SSBAS outperforms Exp3.S by a significant and wide margin: number of steps averaged over all the training 10^5 episodes: 28.7 for SSBAS vs 33.6 for Exp3.S. FORMULA9 ) and more precisely the game Q*bert (see a frame on Figure 6 ), where the goal is to step once on each block.

Then a new similar level starts.

In later levels, one needs to step twice on each block, and even later stepping again on the same blocks will cancel the colour change.

We used three different settings of DQN instances: small uses the setting described in BID10 , large uses the setting in BID11 , and finally huge uses an even larger network (see the supplementary material, Section C.2 for details).

DQN is known to reach a near-human level performance at Q*bert.

Our SSBAS instance runs 6 algorithms with 2 different random initialisations of each DQN setting.

Disclaimer: contrarily to other experiments, each curve is the result of a single run, and the improvement might be aleatory.

Indeed, the DQN training is very long and SSBAS needs to train all the models in parallel.

A more computationally-efficient solution might be to use the same architecture as BID14 .7 reveals that SSBAS experiences a slight delay keeping in touch with the best setting performance during the initial learning phase, but, surprisingly, finds a better policy than the single algorithms in its portfolio and than the ones reported in the previous DQN articles.

We observe that the large setting is surprisingly by far the worst one on the Q*bert task, implying the difficulty to predict which model is the most efficient for a new task.

SSBAS allows to select online the best one.

In this article, we tackle the problem of selecting online off-policy RL algorithms.

The problem is formalised as follows: from a fixed portfolio of algorithms, a meta-algorithm learns which one performs the best on the task at hand.

Fairness of algorithm evaluation implies that the RL algorithms learn off-policy.

ESBAS, a novel meta-algorithm, is proposed.

Its principle is to divide the meta-time scale into epochs.

Algorithms are allowed to update their policies only at the start each epoch.

As the policies are constant inside each epoch, the problem can be cast into a stochastic multi-armed bandit.

An implementation is detailed and a theoretical analysis leads to upper bounds on the regrets.

ESBAS is designed for the growing batch RL setting.

This limited online setting is required in many real-world applications where updating the policy requires a lot of resources.

Experiments are first led on a negotiation dialogue game, interacting with a human data-built simulated user.

In most settings, not only ESBAS demonstrates its efficiency to select the best algorithm, but it also outperforms the best algorithm in the portfolio thanks to curriculum learning, and variance reduction similar to that of Ensemble Learning.

Then, ESBAS is adapted to a full online setting, where algorithms are allowed to update after each transition.

This meta-algorithm, called SSBAS, is empirically validated on a fruit collection task where it performs efficient hyper-parameter optimisation.

SSBAS is also evaluated on the Q*bert Atari game, where it achieves a substantial improvement over the single algorithm counterparts.

We interpret ESBAS/SSBAS's success at reliably outperforming the best algorithm in the portfolio as the result of the four following potential added values.

First, curriculum learning: ESBAS/SSBAS selects the algorithm that is the most fitted with the data size.

This property allows for instance to use shallow algorithms when having only a few data and deep algorithms once collected a lot.

Second, diversified policies: ESBAS/SSBAS computes and experiments several policies.

Those diversified policies generate trajectories that are less redundant, and therefore more informational.

As a result, the policies trained on these trajectories should be more efficient.

Third, robustness: if one algorithm fails at finding good policies, it will soon be discarded.

This property prevents the agent from repeating again and again the same obvious mistakes.

Four and last, run adaptation: of course, there has to be an algorithm that is the best on average for one given task at one given meta-time.

But depending on the variance in the trajectory collection, it did not necessarily train the best policy for each run.

The ESBAS/SSBAS meta-algorithm tries and selects the algorithm that is the best at each run.

Some of those properties are inherited by algorithm selection similarity with ensemble learning (Dietterich, 2002) .

BID23 uses a vote amongst the algorithms to decide the control of the next transition.

Instead, ESBAS/SSBAS selects the best performing algorithm.

Regarding the portfolio design, it mostly depends on the available computational power per sample ratio.

For practical implementations, we recommend to limit the use of two highly demanding algorithms, paired with several faster algorithms that can take care of first learning stages, and to use algorithms that are diverse regarding models, hypotheses, etc.

Adding two algorithms that are too similar adds inertia, while they are likely to not be distinguishable by ESBAS/SSBAS.

More detailed recommendations for building an efficient RL portfolio are left for future work.

Speech recognition score Section C.1.1

Normal distribution of centre x and variance v 2 Section C.1.1 REFINSIST REFPROP(??), with ?? being the last proposed option Section C.1.1 REFNEWPROP REFPROP(??), with ?? being the best option that has not been proposed yet Section C.1.1 ACCEPT ACCEPT(??), with ?? being the last understood option proposition Section C. DISPLAYFORM0 Non-learning algorithm with average performance ?? Section C.1.2 ?? Number of noisy features added to the feature set Section C.1.2

Probability that X = x conditionally to Y = y Equation FORMULA11 B THEORETICAL ANALYSISThe theoretical aspects of algorithm selection for reinforcement learning in general, and Epochal Stochastic Bandit Algorithm Selection in particular, are thoroughly detailed in this section.

The proofs of the Theorems are provided in Sections E, F, and G. We recall and formalise the absolute pseudo-regret definition provided in Section 2.3.Definition 1 (Absolute pseudo-regret).

The absolute pseudo-regret ?? ?? abs (T ) compares the metaalgorithm's expected return with the optimal expected return: DISPLAYFORM0

The theoretical analysis is hindered by the fact that AS, not only directly influences the return distribution, but also the trajectory set distribution and therefore the policies learnt by algorithms for next trajectories, which will indirectly affect the future expected returns.

In order to allow policy comparison, based on relation on trajectory sets they are derived from, our analysis relies on two assumptions.

Assumption 1 (More data is better data).

The algorithms train better policies with a larger trajectory set on average, whatever the algorithm that controlled the additional trajectory: DISPLAYFORM0 Assumption 1 states that algorithms are off-policy learners and that additional data cannot lead to performance degradation on average.

An algorithm that is not off-policy could be biased by a specific behavioural policy and would therefore transgress this assumption.

If an algorithm trains a better policy with one trajectory set than with another, then it remains the same, on average, after collecting an additional trajectory from any algorithm: DISPLAYFORM0 Assumption 2 states that a performance relation between two policies trained on two trajectory sets is preserved on average after adding another trajectory, whatever the behavioural policy used to generate it.

From these two assumptions, Theorem 1 provides an upper bound in order of magnitude in function of the worst algorithm in the portfolio.

It is verified for any meta-algorithm ??.

Theorem 1 (Not worse than the worst).

The absolute pseudo-regret is bounded by the worst algorithm absolute pseudo-regret in order of magnitude: DISPLAYFORM1 Contrarily to what the name of Theorem 1 suggests, a meta-algorithm might be worse than the worst algorithm (similarly, it can be better than the best algorithm), but not in order of magnitude.

Its proof is rather complex for such an intuitive result because, in order to control all the possible outcomes, one needs to translate the selections of algorithm ?? with meta-algorithm ?? into the canonical meta-algorithm ?? ?? 's view.

ESBAS intends to minimise the regret for not choosing the best algorithm at a given meta-time ?? .

It is short-sighted: it does not intend to optimise the algorithms learning.

Definition 2 (Short-sighted pseudo-regret).

The short-sighted pseudo-regret ?? ?? ss (T ) is the difference between the immediate best expected return algorithm and the one selected: DISPLAYFORM0 Theorem 2 (ESBAS short-sighted pseudo-regret).

If the stochastic multi-armed bandit ?? guarantees a regret of order of magnitude O(log(T )/??? ??? ?? ), then: DISPLAYFORM1 Theorem 2 expresses in order of magnitude an upper bound for the short-sighted pseudo-regret of ESBAS.

But first, let define the gaps: DISPLAYFORM2 .

It is the difference of expected return between the best algorithm during epoch ?? and algorithm ??.

The smallest non null gap at epoch ?? is noted: DISPLAYFORM3 if there is no non-null gap, the regret is null.

Several upper bounds in order of magnitude on ?? ss (T ) can be easily deduced from Theorem 2, depending on an order of magnitude of ??? ??? ?? .

See the corollaries in Section F.1, TAB0 and more generally Section 3 for a discussion.

The short-sighted pseudo-regret optimality depends on the meta-algorithm itself.

For instance, a poor deterministic algorithm might be optimal at meta-time ?? but yield no new information, implying the same situation at meta-time ?? + 1, and so on.

Thus, a meta-algorithm that exclusively selects the deterministic algorithm would achieve a short-sighted pseudo-regret equal to 0, but selecting other algorithms are, in the long run, more efficient.

Theorem 2 is a necessary step towards the absolute pseudo-regret analysis.

The absolute pseudo-regret can be decomposed between the absolute pseudo-regret of the best canonical meta-algorithm (i.e. the algorithm that finds the best policy), the regret for not always selecting the best algorithm, and potentially not learning as fast, and the short-sighted regret: the regret for not gaining the returns granted by the best algorithm.

This decomposition leads to Theorem 3 that provides an upper bound of the absolute pseudo-regret in function of the best canonical meta-algorithm, and the short-sighted pseudo-regret.

The fairness of budget distribution is the property stating that every algorithm in the portfolio has as much resources as the others, in terms of computational time and data.

Section 3 discusses it at length.

For analysis purposes, Theorem 3 assumes the fairness of AS:Assumption 3 (Learning is fair).

If one trajectory set is better than another for training one given algorithm, it is the same for other algorithms.

DISPLAYFORM0 Theorem 3 (ESBAS absolute pseudo-regret upper bound).

Under assumption 3, if the stochastic multi-armed bandit ?? guarantees that the best arm has been selected in the T first episodes at least T /K times, with high probability 1 ??? ?? T , with ?? T ??? O(1/T ), then: DISPLAYFORM1 where meta-algorithm ?? * selects exclusively algorithm ?? * = argmin ?????P ?? ?? ?? abs (T ).

Successive and Median Elimination (Even-Dar et al., 2002) and Upper Confidence Bound BID2 under some conditions BID1 are examples of appropriate ?? satisfying both conditions stated in Theorems 2 and 3.

Again, see TAB0 and more generally Section 3 for a discussion of those bounds.

C EXPERIMENTAL DETAILS C.1 DIALOGUE EXPERIMENTS DETAILS C.1.1 THE NEGOTIATION DIALOGUE GAME ESBAS practical efficiency is illustrated on a dialogue negotiation game BID7 that involves two players: the system p s and a user p u .

Their goal is to find an agreement among 4 alternative options.

At each dialogue, for each option ??, players have a private uniformly drawn cost ?? p ?? ??? U[0, 1] to agree on it.

Each player is considered fully empathetic to the other one.

As a result, if the players come to an agreement, the system's immediate reward at the end of the dialogue is R ps (s f ) = 2 ??? ?? ps ?? ??? ?? pu ?? , where s f is the state reached by player p s at the end of the dialogue, and ?? is the agreed option; if the players fail to agree, the final immediate reward is R ps (s f ) = 0, and finally, if one player misunderstands and agrees on a wrong option, the system gets the cost of selecting option ?? without the reward of successfully reaching an agreement: R ps (s f ) = ????? ps ?? ??? ?? pu ?? .

Players act each one in turn, starting randomly by one or the other.

They have four possible actions.

First, REFPROP(??): the player makes a proposition: option ??.

If there was any option previously proposed by the other player, the player refuses it.

Second, ASKREPEAT: the player asks the other player to repeat its proposition.

Third, ACCEPT(??): the player accepts option ?? that was understood to be proposed by the other player.

This act ends the dialogue either way: whether the understood proposition was the right one or not.

Four, ENDDIAL: the player does not want to negotiate anymore and ends the dialogue with a null reward.

Understanding through speech recognition of system p s is assumed to be noisy: with a sentence error rate of probability SER u s = 0.3, an error is made, and the system understands a random option instead of the one that was actually pronounced.

In order to reflect human-machine dialogue asymmetry, the simulated user always understands what the system says: SER The system, and therefore the portfolio algorithms, have their action set restrained to five non parametric actions: REFINSIST ??? REFPROP(?? t???1 ), ?? t???1 being the option lastly proposed by the system; REFNEWPROP ??? REFPROP(??), ?? being the preferred one after ?? t???1 , ASKREPEAT, ACCEPT??? ACCEPT(??), ?? being the last understood option proposition and ENDDIAL.

All learning algorithms are using Fitted-Q Iteration (Ernst et al., 2005) , with a linear parametrisation and an ?? -greedy exploration : ?? = 0.6 ?? , ?? being the epoch number.

Six algorithms differing by their state space representation ?? ?? are considered:??? simple: state space representation of four features: the constant feature ?? 0 = 1, the last recognition score feature ?? asr , the difference between the cost of the proposed option and the next best option ?? dif , and finally an RL-time feature ?? t = 0.1t DISPLAYFORM0 ??? fast: ?? ?? = {?? 0 , ?? asr , ?? dif }.???

simple-2: state space representation of ten second order polynomials of simple features.

DISPLAYFORM1 ??? fast-2: state space representation of six second order polynomials of fast features.

DISPLAYFORM2 ??? n-??-{simple/fast/simple-2/fast-2}: Versions of previous algorithms with ?? additional features of noise, randomly drawn from the uniform distribution in [0, 1].??? constant-??: the algorithm follows a deterministic policy of average performance ?? without exploration nor learning.

Those constant policies are generated with simple-2 learning from a predefined batch of limited size.

In all our experiments, ESBAS has been run with UCB parameter ?? = 1/4.

We consider 12 epochs.

The first and second epochs last 20 meta-time steps, then their lengths double at each new epoch, for a total of 40,920 meta-time steps and as many trajectories.

?? is set to 0.9.

The algorithms and ESBAS are playing with a stationary user simulator built through Imitation Learning from realhuman data.

All the results are averaged over 1000 runs.

The performance figures plot the curves of algorithms individual performance ?? ?? against the ESBAS portfolio control ?? ESBAS in function of the epoch (the scale is therefore logarithmic in meta-time).

The performance is the average return of the reinforcement learning return: it equals ?? | | R ps (s f ) in the negotiation game.

The ratio figures plot the average algorithm selection proportions of ESBAS at each epoch.

We define the relative pseudo regret as the difference between the ESBAS absolute pseudo-regret and the absolute pseudo-regret of the best canonical meta-algorithm.

All relative pseudo-regrets, as well as the gain for not having chosen the worst algorithm in the portfolio, are provided in Table 2 .

Relative pseudo-regrets have a 95% confidence interval about ??6 ??? ??1.5 ?? 10 ???4per trajectory.

Several results show that, in practice, the assumptions are transgressed.

Firstly, we observe that Assumption 3 is transgressed.

Indeed, it states that if a trajectory set is better than another for a given algorithm, then it's the same for the other algorithms.

Still, this assumption infringement does not seem to harm the experimental results.

It even seems to help in general: while this assumption is consistent curriculum learning, it is inconsistent with the run adaptation property advanced in Subsection 6 that states that an algorithm might be the best on some run and another one on other runs.

And secondly, off-policy reinforcement learning algorithms exist, but in practice, we use state space representations that distort their off-policy property .

However, experiments do not reveal any obvious bias related to the off/on-policiness of the trajectory set the algorithms train on.

The three DQN networks (small, large, and huge) are built in a similar fashion, with relu activations at each layer except for the output layer which is linear, with RMSprop optimizer (?? = 0.95 and = 10 no-op max 30??? small has a first convolution layer with a 4x4 kernel and a 2x2 stride, and a second convolution layer with a 4x4 kernel and a 2x2 stride, followed by a dense layer of size 128, and finally the output layer is also dense.??? large has a first convolution layer with a 8x8 kernel and a 4x4 stride, and a second convolution layer with a 4x4 kernel and a 2x2 stride, followed by a dense layer of size 256, and finally the output layer is also dense.??? huge has a first convolution layer with a 8x8 kernel and a 4x4 stride, a second convolution layer with a 4x4 kernel and a 2x2 stride, and a third convolution layer with a 3x3 kernel and a 1x1 stride, followed by a dense layer of size 512, and finally the output layer is also dense.

Portfolio w. best w. worst simple-2 + fast-2 35 -181 simple + n-1-simple-2 -73 -131 simple + n-1-simple 3 -2 simple-2 + n-1-simple-2 -12 -38 all-4 + constant-1.10 21 -2032 all-4 + constant-1.11-21 -1414 all-4 + constant-1.13-10 -561 all-4-28 -275 all-2-simple + constant-1.08-41 -2734 all-2-simple +

constant-1.11-40 -2013 all-2-simple + constant-1.13-123 -799 all-2-simple -90 -121 fast + simple-2 -39 -256 simple-2 + constant-1.01 169 -5361 simple-2 + constant-1.11 53 -1380 simple-2 + constant-1.11 57 -1288 simple + constant-1.08 54 -2622 simple + constant-1.10 88 -1565 simple + constant-1.14 -6 -297 all-4 + all-4-n-1 + constant-1.09 25 -2308 all-4 + all-4-n-1 + constant-1.11 20 -1324 all-4 + all-4-n-1 + constant-1.14 -16 -348 all-4 + all-4-n-1 -10 -142 all-2-simple + all-2-n-1-simple -80 -181 4*n-2-simple -20 -20 4*n-3-simple -13 -13 8*n-1-simple-2 -22 -22 simple-2 + constant-0.97 (no reset) 113 -7131 simple-2 + constant-1.05 (no reset) 23 -3756 simple-2 + constant-1.09 (no reset) -19 -2170 simple-2 + constant-1.13 (no reset) -16 -703 simple-2 + constant-1.14 (no reset) -125 -319 Table 2 : ESBAS pseudo-regret after 12 epochs (i.e. 40,920 trajectories) compared with the best and the worst algorithms in the portfolio, in function of the algorithms in the portfolio (described in the first column).

The '+' character is used to separate the algorithms.

all-4 means all the four learning algorithms described in Section C.1.2 simple + fast + simple-2 + fast-2.

all-4-n-1 means the same four algorithms with one additional feature of noise.

Finally, all-2-simple means simple + simple-2 and all-2-n-1-simple means n-1-simple + n-1-simple-2.

On the second column, the redder the colour, the worse ESBAS is achieving in comparison with the best algorithm.

Inversely, the greener the colour of the number, the better ESBAS is achieving in comparison with the best algorithm.

If the number is neither red nor green, it means that the difference between the portfolio and the best algorithm is insignificant and that they are performing as good.

This is already an achievement for ESBAS to be as good as the best.

On the third column, the bluer the cell, the weaker is the worst algorithm in the portfolio.

One can notice that positive regrets are always triggered by a very weak worst algorithm in the portfolio.

In these cases, ESBAS did not allow to outperform the best algorithm in the portfolio, but it can still be credited with the fact it dismissed efficiently the very weak algorithms in the portfolio.

Theorem 1 (Not worse than the worst).

The absolute pseudo-regret is bounded by the worst algorithm absolute pseudo-regret in order of magnitude: DISPLAYFORM0 Proof.

From Definition 1: DISPLAYFORM1 where sub ?? (D) is the subset of D with all the trajectories generated with algorithm ??, where ?? ?? i is the index of the i th trajectory generated with algorithm ??, and where |S| is the cardinality of finite set S. By convention, let us state that E?? DISPLAYFORM2 To conclude, let us prove by mathematical induction the following inequality: DISPLAYFORM3 is true by vacuity for i = 0: both left and right terms equal E?? ?? ??? .

Now let us assume the property true for i and prove it for i + 1: DISPLAYFORM4 .If |sub ?? (D ?? T )|??? i + 1, by applying mathematical induction assumption, then by applying Assumption 2 and finally by applying Assumption 1 recursively, we infer that: The mathematical induction proof is complete.

This result leads to the following inequalities: DISPLAYFORM5 DISPLAYFORM6 which leads directly to the result: DISPLAYFORM7 This proof may seem to the reader rather complex for such an intuitive and loose result but algorithm selection ?? and the algorithms it selects may act tricky.

For instance selecting algorithm ?? only when the collected trajectory sets contains misleading examples (i.e. with worse expected return than with an empty trajectory set) implies that the following unintuitive inequality is always true: E?? DISPLAYFORM8 .

In order to control all the possible outcomes, one needs to translate the selections of algorithm ?? into ?? ?? 's view.

Theorem 2 (ESBAS short-sighted pseudo-regret).

If the stochastic multi-armed bandit ?? guarantees a regret of order of magnitude O(log(T )/??? ??? ?? ), then: DISPLAYFORM0 Proof.

By simplification of notation, E?? DISPLAYFORM1 .

From Definition 2: Since we are interested in the order of magnitude, we can once again only consider the upper bound of DISPLAYFORM2 DISPLAYFORM3

Theorem 3 (ESBAS absolute pseudo-regret upper bound).

Under assumption 3, if the stochastic multi-armed bandit ?? guarantees that the best arm has been selected in the T first episodes at least T /K times, with high probability 1 ??? ?? T , with ?? T ??? O(1/T ), then: DISPLAYFORM0 where meta-algorithm ?? * selects exclusively algorithm ?? * = argmin ?????P ?? ?? ?? abs (T ).Proof.

The ESBAS absolute pseudo-regret is written with the following notation simplifications : DISPLAYFORM1 and k ?? = ?? ESBAS (?? ): Note that ?? * is the optimal constant algorithm selection at horizon T , but it is not necessarily the optimal algorithm selection: there might exist, and there probably exists a non constant algorithm selection yielding a smaller pseudo-regret.

DISPLAYFORM2 The ESBAS absolute pseudo-regret ?? ?? ESBAS abs (T ) can be decomposed into the pseudo-regret for not having followed the optimal constant algorithm selection ?? * and the pseudo-regret for not having selected the algorithm with the highest return, i.e. between the pseudo-regret on the trajectory and the pseudo-regret on the immediate optimal return: .

is to evaluate the size of sub * (D ?? ???1 ).On the one side, Assumption 3 of fairness states that one algorithm learns as fast as any another over any history.

The asymptotically optimal algorithm(s) when ?? ??? ??? is(are) therefore the same one(s) whatever the the algorithm selection is.

On the other side, let 1 ??? ?? ?? denote the probability, that at time ?? , the following inequality is true: DISPLAYFORM0 With probability ?? ?? , inequality 34 is not guaranteed and nothing can be inferred about E?? * sub * (D?????1) , except it is bounded under by R min /(1 ??? ??).

Let E DISPLAYFORM1 Let consider E * (??, N ) the set of all sets D such that |sub ?? (D)|= N and such that last trajectory in D was generated by ??.

Since ESBAS, with ??, a stochastic bandit with regret in O(log(T )/???), guarantees that all algorithms will eventually be selected an infinity of times, we know that :(37) ????? ??? P, ???N ??? N, D ???E + (??,N ) P(D|?? ESBAS ) = 1.By applying recursively Assumption 2, one demonstrates that: DISPLAYFORM2 D ???E + (??,N ) DISPLAYFORM3 One also notices the following piece-wisely domination from applying recursively Assumption 1: DISPLAYFORM4 (1 ??? ?? ?? )E?? * ??? ??? D ???E ; the meta-time spent on epoch ?? ?? ??? 1 is equal to 2 ???? ???1; the meta-time spent on epoch ?? is either below 2 ???? ???1 , in which case, the meta-time spent on epoch ?? ?? ??? 1 is higher than ?? 3 , or the meta-time spent on epoch ?? is over 2 ???? ???1 and therefore higher than ?? 3 .

Thus, ESBAS is guaranteed to try the best algorithm ?? * at least ?? /3K times with high probability 1 ??? ?? ?? and ?? ?? ??? O(?? ???1 ).

As a result: DISPLAYFORM5 with ?? = ?? 3 E?? * ??? ??? R min 1 ??? ?? , which proves the theorem.

<|TLDR|>

@highlight

This paper formalises the problem of online algorithm selection in the context of Reinforcement Learning.