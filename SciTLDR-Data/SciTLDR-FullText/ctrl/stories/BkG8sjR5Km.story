We study the emergence of cooperative behaviors in reinforcement learning agents by introducing a challenging competitive multi-agent soccer environment with continuous simulated physics.

We demonstrate that decentralized, population-based training with co-play can lead to a progression in agents' behaviors: from random, to simple ball chasing, and finally showing evidence of cooperation.

Our study highlights several of the challenges encountered in large scale multi-agent training in continuous control.

In particular, we demonstrate that the automatic optimization of simple shaping rewards, not themselves conducive to co-operative behavior, can lead to long-horizon team behavior.

We further apply an evaluation scheme, grounded by game theoretic principals, that can assess agent performance in the absence of pre-defined evaluation tasks or human baselines.

Competitive games have been grand challenges for artificial intelligence research since at least the 1950s BID38 BID45 BID6 .

In recent years, a number of breakthroughs in AI have been made in these domains by combining deep reinforcement learning (RL) with self-play, achieving superhuman performance at Go and Poker Moravk et al., 2017) .

In continuous control domains, competitive games possess a natural curriculum property, as observed in , where complex behaviors have the potential to emerge in simple environments as a result of competition between agents, rather than due to increasing difficulty of manually designed tasks.

Challenging collaborative-competitive multi-agent environments have only recently been addressed using end-to-end RL by BID21 , which learns visually complex first-person 2v2 video games to human level.

One longstanding challenge in AI has been robot soccer BID23 , including simulated leagues, which has been tackled with machine learning techniques BID37 BID29 but not yet mastered by end-to-end reinforcement learning.

We investigate the emergence of co-operative behaviors through multi-agent competitive games.

We design a simple research environment with simulated physics in which complexity arises primarily through competition between teams of learning agents.

We introduce a challenging multi-agent soccer environment, using MuJoCo BID46 which embeds soccer in a wider universe of possible environments with consistent simulated physics, already used extensively in the machine learning research community BID16 BID5 BID44 .

We focus here on multi-agent interaction by using relatively simple bodies with a 3-dimensional action space (though the environment is scalable to more agents and more complex bodies).

1 We use this environment to examine continuous multiagent reinforcement learning and some of its challenges including coordination, use of shaping rewards, exploitability and evaluation.

We study a framework for continuous multi-agent RL based on decentralized population-based training (PBT) of independent RL learners BID20 , where individual agents learn off-policy with recurrent memory and decomposed shaping reward channels.

In contrast to some recent work where some degree of centralized learning was essential for multi-agent coordinated behaviors (e.g. BID28 BID9 , we demonstrate that end-to-end PBT can lead to emergent cooperative behaviors in our soccer domain.

While designing shaping rewards that induce desired cooperative behavior is difficult, PBT provides a mechanism for automatically evolving simple shaping rewards over time, driven directly by competitive match results.

We further suggest to decompose reward into separate weighted channels, with individual discount factors and automatically optimize reward weights and corresponding discounts online.

We demonstrate that PBT is able to evolve agents' shaping rewards from myopically optimizing dense individual shaping rewards through to focusing relatively more on long-horizon game rewards, i.e. individual agent's rewards automatically align more with the team objective over time.

Their behavior correspondingly evolves from random, through simple ball chasing early in the learning process, to more co-operative and strategic behaviors showing awareness of other agents.

These behaviors are demonstrated visually and we provide quantitative evidence for coordination using game statistics, analysis of value functions and a new method of analyzing agents' counterfactual policy divergence.

Finally, evaluation in competitive multi-agent domains remains largely an open question.

Traditionally, multi-agent research in competitive domains relies on handcrafted bots or established human baselines BID21 , but these are often unavailable and difficult to design.

In this paper, we highlight that diversity and exploitability of evaluators is an issue, by observing non-transitivities in the agents pairwise rankings using tournaments between trained teams.

We apply an evaluation scheme based on Nash averaging BID2 and evaluate our agents based on performance against pre-trained agents in the support set of the Nash average.

We treat our soccer domain as a multi-agent reinforcement learning problem (MARL) which models a collection of agents interacting with an environment and learning, from these interactions, to optimize individual cumulative reward.

MARL can be cooperative, competitive or some mixture of the two (as is the case in soccer), depending upon the alignment of agents' rewards.

MARL is typically modelled as a Markov game BID39 Littman, 1994) , which comprises: a state space S, n agents with observation and action sets O 1 , ..., O n and A 1 , ..., A n ; a (possibly stochastic) reward function R i : S × A i → R for each agent; observation functions φ i : S → O i ; a transition function P which defines the conditional distribution over successor states given previous state-actions: P (S t+1 |S t , A 1 t , ..., A n t ), which satisfies the Markov property P (S t+1 |S τ , A 1 τ , ..., A n τ , ∀τ ≤ t) = P (S t+1 |S t , A Algorithm 1 Population-based Training for Multi-Agent RL.1: procedure PBT-MARL 2:{Ai} i∈ [1,..,N ] N independent agents forming a population.

for agent Ai in {Ai} i∈[1,..,N ] do 4:Initialize agent network parameters θi and agent rating ri to fixed initial rating Rinit.

Sample initial hyper-parameter θ h i from the initial hyper-parameter distribution.

6: end for 7:while true do 8:Agents play TrainingMatches and update network parameters by Retrace-SVG0.

for match result (si, sj) ∈ TrainingMatches do 10:UpdateRating ( agents, according to some fitness function, inherit network parameters and some hyperparameters from stronger agents, with additional mutation.

Hyperparameters can continue to evolve during training, rather than committing to a single fixed value (we show that this is indeed the case in Section 5.1).

PBT was extended to incorporate co-play BID21 as a method of optimizing agents for MARL: subsets of agents are selected from the population to play together in multi-agent games.

In any such game each agent in the population effectively treats the other agents as part of their environment and learns a policy π θ to optimize their expected return, averaged over such games.

In any game in which π θ controls player i in the game, if we denote by π \i := {π j } j∈{1,2,...,n},j =i the policies of the other agents j = i, we can write the expected cumulative return over a game as DISPLAYFORM0 where the expectation is w.r.t.

the environment dynamics and conditioned on the actions being drawn from policies π θ and π \i .

Each agent in the population attempts to optimize (1) averaged over the draw of all agents from the population P, leading to the PBT objective J(π θ ) : DISPLAYFORM1 , where the outer expectation is w.r.t.

the probability that the agent with policy π θ controls player i in the environment, and the inner expectation is the expectation over the draw of other agents, conditioned on π θ controlling player i in the game.

PBT achieves some robustness to exploitability by training a population of learning agents against each other.

Algorithm 1 describes PBT-MARL for a population of N agents {A i } i∈ [1,..,N ] , employed in this work.

Throughout our experiments we use Stochastic Value Gradients (SVG0) BID15 as our reinforcement learning algorithm for continuous control.

This is an actor-critic policy gradient algorithm, which in our setting is used to estimate gradients network for bootstrapping, as is also described in BID13 ; .

The identity of other agents π \i in a game are not explicitly revealed but are potentially vital for accurate action-value estimation (value will differ when playing against weak rather than strong opponents).

Thus, we use a recurrent critic to enable the Q-function to implicitly condition on other players observed behavior, better estimate the correct value for the current game, and generalize over the diversity of players in the population of PBT, and, to some extent, the diversity of behaviors in replay.

We find in practice that a recurrent Q-function, learned from partial unrolls, performs very well.

Details of our Q-critic updates, including how memory states are incorporated into replay, are given in Appendix A.2.

Reinforcement learning agents learning in environments with sparse rewards often require additional reward signal to provide more feedback to the optimizer.

Reward can be provided to encourage agents to explore novel states for instance (e.g. BID4 , or some other form of intrinsic motivation.

Reward shaping is particularly challenging in continuous control (e.g. BID35 where obtaining sparse rewards is often highly unlikely with random exploration, but shaping can perturb objectives (e.g. BID1 resulting in degenerate behaviors.

Reward shaping is yet more complicated in the cooperative multi-agent setting in which independent agents must optimize a joint objective.

Team rewards can be difficult to co-optimize due to complex credit assignment, and can result in degenerate behavior where one agent learns a reasonable policy before its teammate, discouraging exploration which could interfere with the first agent's behavior as observed by BID12 .

On the other hand, it is challenging to design shaping rewards which induce desired co-operative behavior.

We design n r shaping reward functions {r j : S × A → R} j=1,...,nr , weighted so that r(·) := nr j=1 α j r j (·) is the agent's internal reward and, as in BID21 , we use populationbased training to optimize the relative weighting {α j } j=1,...,nr .

Our shaping rewards are simple individual rewards to help with exploration, but which would induce degenerate behaviors if badly scaled.

Since the fitness function used in PBT will typically be the true environment reward (in our case win/loss signal in soccer), the weighting of shaping rewards can in principle be automatically optimized online using the environment reward signal.

One enhancement we introduce is to optimize separate discount factors {γ j } j=1,...,nr for each individual reward channel.

The objective optimized is then (recalling Equation 1) DISPLAYFORM0 This separation of discount factors enables agents to learn to optimize the sparse environment reward far in the future with a high discount factor, but optimize dense shaping rewards myopically, which would also make value-learning easier.

This would be impossible if discounts were confounded.

The specific shaping rewards used for soccer are detailed in Section 5.1.

We simulate 2v2 soccer using the MuJoCo physics engine BID46 .

The 4 players in the game are a single sphere (the body) with 2 fixed arms, and a box head, and have a 3-dimensional action space: accelerate the body forwards/backwards, torque can be applied around the vertical axis to rotate, and apply downwards force to "jump".

Applying torque makes the player spin, gently for steering, or with more force in order to "kick" the football with its arms.

At each timestep, proprioception (position, velocity, accelerometer information), task (egocentric ball position, velocity and angular velocity, goal and corner positions) and teammate and opponent (orientation, position and velocity) features are observed making a 93-dimensional input observation vector.

Each soccer match lasts upto 45 seconds, and is terminated when the first team scores.

We disable contacts between the players, but enable contacts between the players, the pitch and the ball.

This makes it impossible for players to foul and avoids the need for a complicated contact rules, and led to more dynamic matches.

There is a small border around the pitch which players can enter, but when the ball is kicked out-of-bounds it is reset by automatic "throw in" a small random distance towards the center of the pitch, and no penalty is incurred.

The players choose a new action every 0.05 seconds.

At the start of an episode the players and ball are positioned uniformly at random on the pitch.

We train agents on a field whose dimensions are randomized in the range 20m × 15m to 28m × 21m, with fixed aspect ratio, and are tested on a field of fixed size 24m × 18m.

We show an example frame of the game in FIG0 .

We use population-based training with 32 agents in the population, an agent is chosen for evolution if its expected win rate against another chosen agent drops below 0.47.

The k-factor learning rate for Elo is 0.1 (this is low, due to the high stochasticity in the game results).

Following evolution there is a grace period where the agent does not learn while its replay buffer refills with fresh data, and a further "burn-in" period before the agent can evolve again or before its weights can be copied into another agent, in order to limit the frequency of evolution and maintain diversity in the population.

For each 2v2 training match 4 agents were selected uniformly at random from the population of 32 agents, so that agents are paired with diverse teammates and opponents.

Unlike multi-agent domains where we possess hand-crafted bots or human baselines, evaluating agent performance in novel domains where we do not possess such knowledge remains an open question.

A number of solutions have been proposed: for competitive board games, there exits evaluation metrics such as Elo BID8 where ratings of two players should translate to their relative win-rates; in professional team sports, head-to-head tournaments are typically used to measure team performance; in BID0 , survival-of-the-fittest is directly translated to multiagent learning as a proxy to relative agent performance.

Unfortunately, as shown in BID2 , in a simple game of rock-paper-scissors, a rock-playing agent will attain high Elo score if we simply introduce more scissor-play agents into a tournament.

Survival-of-the-fittest analysis as shown in BID0 would lead to a cycle, and agent ranking would depend on when measurements are taken .

Nash-Averaging Evaluators: One desirable property for multi-agent evaluation is invariance to redundant agents: i.e. the presence of multiple agents with similar strategies should not bias the ranking.

In this work, we apply Nash-averaging which possesses this property.

Nash-Averaging consists of a meta-game played using a pair-wise win-rate matrix between N agents.

A row player and a column player simultaneously pick distributions over agents for a mixed strategy, aiming for a non-exploitable strategy (see BID2 .In order to meaningfully evaluate our learned agents, we need to bootstrap our evaluation process.

Concretely, we choose a set of fixed evaluation teams by Nash-averaging from a population of 10 teams previously produced by diverse training schemes, with 25B frames of learning experience each.

We collected 1M tournament matches between the set of 10 agents.

FIG1 shows the pairwise expected goal difference among the 3 agents in the support set.

Nash Averaging assigned nonzero weights to 3 teams that exhibit diverse policies with non-transitive performance which would not have been apparent under alternative evaluation schemes: agent A wins or draws against agent B on 59.7% of the games; agent B wins or draws against agent C on 71.1% of the games and agent C wins or draws against agent A on 65.3% of the matches.

We show recordings of example tournament matches between agent A, B and C to demonstrate qualitatively the diversity in their policies (video 3 on the website 2 ).

Elo rating alone would yield a different picture: agent B is the best agent in the tournament with an Elo rating of 1084.27, followed by C at 1068.85; Agent A ranks 5th at 1016.48 and we would have incorrectly concluded that agent B ought to beat agent A with a win-rate of 62%.

All variants of agents presented in the experimental section are evaluated against the set of 3 agents in terms of their pair-wise expected difference in score, weighted by support weights.

We describe in this section a set of experimental results.

We first present the incremental effect of various algorithmic components.

We further show that population-based training with co-play and reward shaping induces a progression from random to simple ball chasing and finally coordinated behaviors.

A tournament between all trained agents is provided in Appendix D.

We incrementally introduce algorithmic components and show the effect of each by evaluating them against the set of 3 evaluation agents.

We compare agent performance using expected goal difference weighted according to the Nash averaging procedure.

We annotate a number of algorithmic components as follows: ff: feedforward policy and action-value estimator; evo: population-based training with agents evolving within the population; rwd shp: providing dense shaping rewards on top of sparse environment scoring/conceding rewards; lstm: recurrent policy with recurrent action-value estimator; lstm q: feedforward policy with recurrent action-value estimator; channels: decomposed action-value estimation for each reward component; each with its own, individually evolving discount factor.

Population-based Training with Evolution: We first introduce PBT with evolution.

FIG2 (ff vs ff + evo) shows that Evolution kicks in at 2B steps, which quickly improves agent performance at the population level.

We show in FIG3 that Population-based training coupled with evolution yields a natural progression of learning rates, entropy costs as well as the discount factor.

Critic learning rate gradually decreases as training progresses, while discount factor increases over time, focusing increasingly on long-term return.

Entropy costs slowly decreases which reflects a shift from exploration to exploitation over the course training.

Reward Shaping: We introduced two simple dense shaping rewards in addition to the sparse scoring and conceding environment rewards: vel-to-ball: player's linear velocity projected onto its unit direction vector towards the ball, thresholded at zero; vel-ball-to-goal: ball's linear velocity projected onto its unit direction vector towards the center of opponent's goal.

Furthermore the sparse goal reward and concede penalty are separately evolved, and so can receive separate weight that trades off between the importance of scoring versus conceding.

Dense shaping rewards make learning significantly easier early in training.

This is reflected by agents' performance against the dummy evaluator where agents with dense shaping rewards quickly start to win games from the start FIG2 , ff + evo vs ff + evo + rwd shp).

On the other hand, shaping rewards tend to induce sub-optimal policies BID34 BID35 ; We show in FIG4 however that this is mitigated by coupling training with hyper-parameter evolution which adaptively adjusts the importance of shaping rewards.

Early on in the training, the population as a whole decreases the penalty of conceding a goal which evolves towards zero, assigning this reward relatively lower weight than scoring.

This trend is subsequently reversed towards the end of training, where the agents evolved to pay more attention to conceding goals: i.e. agents first learn to optimize scoring and then incorporate defending.

The dense shaping reward vel-to-ball however quickly decreases in relative importance which is mirrored in their changing behavior, see Section 5.2.Recurrence: The introduction of recurrence in the action-value function has a significant impact on agents' performance as shown in FIG2 (ff + evo + rwd shp vs lstm + evo + rwd shp reaching weighted expected goal difference of 0 at 22B vs 35B steps).

A recurrent policy seems to underperform its feedforward counterpart in the presence of a recurrent action-value function.

This could be due to out-of-sample evaluators which suggests that recurrent policy might overfit to the behaviors of agents from its own population while feedforward policy cannot.

Decomposed Action-Value Function: While we observed empirically that the discount factor increases over time during the evolution process, we hypothesize that different reward components require different discount factor.

We show in FIG5 that this is indeed the case, for sparse environment rewards and vel-ball-to-goal, the agents focus on increasingly long planning horizon.

In contrast, agents quickly evolve to pay attention to short-term returns on vel-to-ball, once they learned the basic movements.

Note that although this agent underperforms lstm + evo + rwd shp asymptot- ically, it achieved faster learning in comparison (reaching 0.2 at 15B vs 35B).

This agent also attains the highest Elo in a tournament between all of our trained agents, see Appendix D. This indicates that the training population is less diverse than the Nash-averaging evaluation set, motivating future work on introducing diversity as part of training regime.

Assessing cooperative behavior in soccer is difficult.

We present several indicators ranging from behavior statistics, policy analysis to behavior probing and qualitative game play in order to demonstrate the level of cooperation between agents.

We provide birds-eye view videos on the website 2 (video 1), where each agent's value-function is also plotted, along with a bar plot showing the value-functions for each weighted shaping reward component.

Early in the matches the 2 dense shaping rewards (rightmost channels) dominate the value, until it becomes apparent that one team has an advantage at which point all agent's value functions become dominated by the sparse conceding/scoring reward (first and second channels) indicating that PBT has learned a balance between sparse environment and dense shaping rewards so that positions with a clear advantage to score will be preferred.

There are recurring motifs in the videos: for example, evidence that agents have learned a "cross" pass from the sideline to a teammate in the centre (see Appendix F for example traces), and frequently appear to anticipate this and change direction to receive.

Another camera angle is provided on the website 2 (video 2) showing representative, consecutive games played between two fixed teams.

These particular agents generally kick the ball upfield, avoiding opponents and towards teammates.

Statistics collected during matches are shown in FIG6 .

The vel-to-ball plot shows the agents average velocity towards the ball as training progresses: early in the learning process agents quickly maximize their velocity towards the ball (optimizing their shaping reward) but gradually fixate less on simple ball chasing as they learn more useful behaviors, such as kicking the ball upfield.

The teammate-spread-out shows the evolution of the spread of teammates position on the pitch.

This shows the percentage of timesteps where the teammates are spread at least 5m apart: both agents quickly learn to hog the ball, driving this lower, but over time learn more useful behaviors which result in diverse player distributions.

pass/interception shows that pass, where players from the same team consecutively kicked the ball and interception, where players from the opposing teams kicked the ball in sequence, both remain flat throughout training.

To pass is the more difficult behavior as it requires two teammates to coordinate whereas interception only requires one of the two opponents to position correctly.

pass/interception-10m logs pass/interception events over more than 10m, and here we see a dramatic increase in pass-10m while interception-10m remains flat, i.e. long range passes become increasingly common over the course of training, reaching equal frequency as long-range interception.

In addition to analyzing behavior statistics, we could ask the following: "had a subset of the observation been different, how much would I have changed my policy?".

This reveals the extent to which an agent's policy is dependent on this subset of the observation space.

To quantify this, we analyze counterfactual policy divergence: at each step, we replace a subset of the observation with 10 valid alternatives, drawn from a fixed distribution, and we measure the KL divergence incurred in agents' policy distributions.

This cannot be measured for a recurrent policy due to recurrent states and we investigate ff + evo + rwd shp instead FIG2 , where the policy network is feedforward.

We study the effect of five types of counterfactual information over the course of training.ball-position has a strong impact on agent's policy distribution, more so than player and opponent positions.

Interestingly, ball-position initially reaches its peak quickly while divergence incurred by counterfactual player/opponent positions plateau until reaching 5B training steps.

This phase coincides with agent's greedy optimization of shaping rewards, as reflected in FIG7 .

Counterfactual teammate/opponent position increasingly affect agents' policies from 5B steps, as they spread out more and run less directly towards the ball.

Opponent-0/1-position incur less divergence than teammate position individually, suggesting that teammate position has relatively large impact than any single opponent, and increasingly so during 5B-20B steps.

This suggests that comparatively players learn to leverage a coordinating teammate first, before paying attention to competing opponents.

The gap between teammate-position and opponents-position eventually widens, as opponents become increasingly relevant to the game dynamics.

The progression observed in counterfactual policy divergence provides evidence for emergent cooperative behaviors among the players.

Qualitatively, we could ask the following question: would agents coordinate in scenarios where it's clearly advantageous to do so?

To this end, we designed a probe task, to test our trained agents for coordination, where blue0 possesses the ball, while the two opponents are centered on the pitch in front.

A teammate blue1 is introduced to either left or right side.

In Figure 9 we show typical traces of agents' behaviors (additional probe task video shown at Video 4 on our website 2 ): at 5B steps, pass intercept 5B left 0 100 5B right 31 90 80B left 76 24 80B right 56 27Figure 9: L1: Comparison between two snapshots (5B vs 80B) of the same agent.

L2: number of successful passes and interception occurred in the first 100 timesteps, aggregated over 100 episodes.when agents play more individualistically, we observe that blue0 always tries to dribble the ball by itself, regardless of the position of blue1.

Later on in the training, blue0 actively seeks to pass and its behavior is driven by the configuration of its teammate, showing a high-level of coordination.

In "8e10 left" in particular, we observe two consecutive pass (blue0 to blue1 and back), in the spirit of 2-on-1 passes that emerge frequently in human soccer games.

The population-based training we use here was introduced by BID21 for the capturethe-flag domain, whereas our implementation is for continuous control in simulated physics which is less visually rich but arguably more open-ended, with potential for sophisticated behaviors generally and allows us to focus on complex multi-agent interactions, which may often be physically observable and interpretable (as is the case with passing in soccer).

Other recent related approaches to multi-agent training include PSRO BID24 and NFSP BID18 , which are motivated by game-theoretic methods (fictitious play and double oracle) for solving matrix games, aiming for some robustness by playing previous best response policies, rather than the (more data efficient and parallelizable) approach of playing against simultaneous learning agents in a population.

The RoboCup competition is a grand challenge in AI and some top-performing teams have used elements of reinforcement learning BID37 BID29 , but are not end-to-end RL.

Our environment is intended as a research platform, and easily extendable along several lines of complexity: complex bodies; more agents; multi-task, transfer and continual learning.

Coordination and cooperation has been studied recently in deepRL in, for example, BID28 BID11 BID12 BID42 ; BID32 , but all of these require some degree of centralization.

Agents in our framework perform fully independent asynchronous learning yet demonstrate evidence of complex coordinated behaviors.

BID0 introduce a MuJoCo Sumo domain with similar motivation to ours, and observe emergent complexity from competition, in a 1v1 domain.

We are explicitly interested in cooperation within teams as well as competition.

Other attempts at optimizing rewards for multi-agent teams include BID27 .

We have introduced a new 2v2 soccer domain with simulated physics for continuous multi-agent reinforcement learning research, and used competition between agents in this simple domain to train teams of independent RL agents, demonstrating coordinated behavior, including repeated passing motifs.

We demonstrated that a framework of distributed population-based-training with continuous control, combined with automatic optimization of shaping reward channels, can learn in this environment end-to-end.

We introduced the idea of automatically optimizing separate discount factors for the shaping rewards, to facilitate the transition from myopically optimizing shaping rewards towards alignment with the sparse long-horizon team rewards and corresponding cooperative behavior.

We have introduced novel method of counterfactual policy divergence to analyze agent behavior.

Our evaluation has highlighted non-transitivities in pairwise match results and the practical need for robustness, which is a topic for future work.

Our environment can serve as a platform for multiagent research with continuous physical worlds, and can be easily scaled to more agents and more complex bodies, which we leave for future research.

In our soccer environment the reward is invariant over player and we can drop the dependence on i.

SVG requires the critic to learn a differentiable Q-function.

The true state of the game s and the identity of other agents π \i , are not revealed during a game and so identities must be inferred from their behavior, for example.

Further, as noted in BID10 , off-policy replay is not always fully sound in multi-agent environments since the effective dynamics from any single agent's perspective changes as the other agent's policies change.

Because of this, we generally model Q as a function of an agents history of observations -typically keeping a low dimensional summary in the internal state of an LSTM: Q π θ (·, ·; ψ) : X × A → R, where X denotes the space of possible histories or internal memory state, parameterized by a neural network with weights ψ.

This enables the Q-function to implicitly condition on other players observed behavior and generalize over the diversity of players in the population and diversity of behaviors in replay, Q is learned using trajectory data stored in an experience replay buffer B, by minimizing the k-step return TD-error with off-policy retrace correction BID33 , using a separate target network for bootstrapping, as is also described in BID13 ; .

Specifically we minimize: DISPLAYFORM0 where ξ := ((s t , a t , r t )) i+k t=i is a k-step trajectory snippet, where i denotes the timestep of the first state in the snippet, sampled uniformly from the replay buffer B of prior experience, and Q retrace is the off-policy corrected retrace target: DISPLAYFORM1 where, for stability,Q(·, ·;ψ) : X ×A → R andπ are target network and policies BID30 periodically synced with the online action-value critic and policy (in our experiments we sync after every 100 gradient steps), and c s := min(1, π(as|xs) β(as|xs) ), where β denotes the behavior policy which generated the trajectory snippet ξ sampled from B, and i s=i+1 c s := 1.

In our soccer experiments k = 40.

Though we use off-policy corrections, the replay buffer has a threshold, to ensure that data is relatively recent.

When modelling Q using an LSTM the agent's internal memory state at the first timestep of the snippet is stored in replay, along with the trajectory data.

When replaying the experience the LSTM is primed with this stored internal state but then updates its own state during replay of the snippet.

LSTMs are optimized using backpropagation through time with unrolls truncated to length 40 in our experiments.

We use Elo rating BID8 ), introduced to evaluate the strength of human chess players, to measure an agent's performance within the population of learning agents and determine eligibility for evolution.

Elo is updated from pairwise match results and can be used to predict expected win rates against the other members of the population.

For a given pair of agents i, j (or a pair of agent teams), s elo estimates the expected win rate of agent i playing against agent j.

We show in Algorithm 3 the update rule for a two player competitive game for simplicity, for a team of multiple players, we use their average Elo score instead.

By using Elo as the fitness function, driving the evolution of the population's hyperparamters, the agents' internal hyperparameters (see Section 3.3) can be automatically optimized for the objective we are ultimately interested in -the win rate against other agents.

Individual shaping rewards would otherwise be difficult to handcraft without biasing this objective.

We parametrize each agent's policy and critic using neural networks.

Observation preprocessing is first applied to each raw teammate and opponent feature using a shared 2-layer network with 32 and 16 neurons and Elu activations BID7 to embed each individual player's data into a consistent, learned 16 dimensional embedding space.

The maximum, minimum and mean of each dimension is then passed as input to the remainder of the network, where it is concatenated with the ball and pitch features.

This preprocessing makes the network architecture invariant to the order of teammates and opponents features.

Both critic and actor then apply 2 feed-forward, elu-activated, layers of size 512 and 256, followed by a final layer of 256 neurons which is either feed-forward or made recurrent using an LSTM BID19 .

Weights are not shared between critic and actor networks.

We learn the parametrized gaussian policies using SVG0 as detailed in Appendix A, and the critic as described in Section A.2, with the Adam optimizer BID22 used to apply gradient updates.

We also ran a round robin tournament with 50,000 matches between the best teams from 5 populations of agents (selected by Elo within their population), all trained for 5e10 agent steps -i.e.

each learner had processed at least 5e10 frames from the replay buffer, though the number of raw environment steps would be much lower than that) and computed the Elo score.

This shows the advantage of including shaping rewards, adding a recurrent critic and separate reward and discount channels, and the further (marginal) contribution of a recurrent actor.

The full win rate matrix for this tournament is given in FIG0 .

Note that the agent with full recurrence and separate reward channels attains the highest Elo in this tournament, though performance against our Nash evaluators in Section 5.1 is more mixed.

This highlights the possibility for non-transitivities in this domain and the practical need for robustness to opponents.

To assess the relative importance of hyperparameters we replicated a single experiment (using a feed-forward policy and critic network) with 3 different seeds, see FIG0 .

Critic learning rate and entropy regularizer evolve consistently over the three training runs.

In particular the critic learning rate tends to be reduced over time.

If a certain hyperparameter was not important to agent performance we would expect less consistency in its evolution across seeds, as selection would be driven by other hyperparameters: thus indicating performance is more sensitive to critic learning rate than actor learning rate.

Elo lstm + evo + rwd shp + channels 1071 lstm q + evo + rwd shp + channels 1069 lstm q + evo + rwd shp 1006 ff + evo + rwd shp 956 ff + evo 898Figure 10: Win rate matrix for the Tournament between teams: from top to bottom, ordered by Elo, ascending: ff + evo; ff + evo + rwd shp; lstm q + evo + rwd shp; lstm q + evo + rwd shp + channels; lstm + evo + rwd shp + channels.

ELo derived from the tournament is given in the table.

Figure 11: Hyperparameter evolution for three separate seeds, displayed over three separate rows.

As well as the videos at the website 3 , we provide visualizations of traces of the agent behavior, in the repeated "cross pass" motif, see FIG0 .

<|TLDR|>

@highlight

We introduce a new MuJoCo soccer environment for continuous multi-agent reinforcement learning research, and show that population-based training of independent reinforcement learners can learn cooperative behaviors