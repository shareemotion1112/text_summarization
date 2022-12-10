Counterfactual Regret Minimization (CFR) is the most successful algorithm for finding approximate Nash equilibria in imperfect information games.

However, CFR's reliance on full game-tree traversals limits its scalability and generality.

Therefore, the game's state- and action-space is often abstracted (i.e. simplified) for CFR, and the resulting strategy is then mapped back to the full game.

This requires extensive expert-knowledge, is not practical in many games outside of poker, and often converges to highly exploitable policies.

A recently proposed method, Deep CFR, applies deep learning directly to CFR, allowing the agent to intrinsically abstract and generalize over the state-space from samples, without requiring expert knowledge.

In this paper, we introduce Single Deep CFR (SD-CFR), a variant of Deep CFR that has a lower overall approximation error by avoiding the training of an average strategy network.

We show that SD-CFR is more attractive from a theoretical perspective and empirically outperforms Deep CFR with respect to exploitability and one-on-one play in poker.

In perfect information games, players usually seek to play an optimal deterministic strategy.

In contrast, sound policy optimization algorithms for imperfect information games converge towards a Nash equilibrium, a distributional strategy characterized by minimizing the losses against a worst-case opponent.

The most popular family of algorithms for finding such equilibria is Counterfactual Regret Minimization (CFR) (Zinkevich et al., 2008) .

Conventional CFR methods iteratively traverse the game-tree to improve the strategy played in each state.

For instance, CFR + (Tammelin, 2014), a fast variant of CFR, was used to solve two-player Limit Texas Hold'em Poker (Bowling et al., 2015; Tammelin et al., 2015) , a variant of poker frequently played by humans.

However, the scalability of such tabular CFR methods is limited since they need to visit a given state to update the policy played in it.

In games too large to fully traverse, practitioners hence often employ domain-specific abstraction schemes (Ganzfried & Sandholm, 2014; Brown et al., 2015) that can be mapped back to the full game after training has finished.

Unfortunately, these techniques have been shown to lead to highly exploitable policies in the large benchmark game Heads-Up No-Limit Texas Hold'em Poker (HUNL) (Lisy & Bowling, 2016) and typically require extensive expert knowledge.

To address these two problems, researchers started to augment CFR with neural network function approximation, first resulting in DeepStack (Moravčík et al., 2017) .

Concurrently with Libratus , DeepStack was one of the first algorithms to defeat professional poker players in HUNL, a game consisting of 10 160 states and thus being far too large to fully traverse.

While tabular CFR has to visit a state of the game to update its policy in it, a parameterized policy may be able to play an educated strategy in states it has never seen before.

Purely parameterized (i.e. non-tabular) policies have led to great breakthroughs in AI for perfect information games (Mnih et al., 2015; Schulman et al., 2017; Silver et al., 2017) and were recently also applied to large imperfect information games by Deep CFR to mimic a variant of tabular CFR from samples.

Deep CFR's strategy relies on a series of two independent neural approximations.

In this paper, we introduce Single Deep CFR (SD-CFR), a simplified variant of Deep CFR that obtains its final strategy after just one neural approximation by using what Deep CFR calls value networks directly instead of training an additional network to approximate the weighted average strategy.

This reduces the overall sampling-and approximation error and makes training more efficient.

We show experimentally that SD-CFR improves upon the convergence of Deep CFR in poker games and outperforms Deep CFR in one-one-one matches.

This section introduces extensive-form games and the notation we will use throughout this work.

Formally, a finite two-player extensive-form game with imperfect information is a set of histories H, where each history is a path from the root φ ∈ H to any particular state.

The subset Z ⊂ H contains all terminal histories.

A(h) is the set of actions available to the acting player at history h, who is chosen from the set {1, 2, chance} by the player function P (h).

In any h ∈ H where P (h) = chance, the action is chosen by the dynamics of the game itself.

Let N = {1, 2} be the set of both players.

When referring to a player i ∈ N , we refer to his opponent by −i.

All nodes z ∈ Z have an associated utility u(z) for each player.

This work focuses on zero-sum games, defined by the property

Imperfect information is represented by partitioning H into information sets.

An information set I i is a subset of H, where histories h, h ∈ H are in the same information set if and only if player i cannot distinguish between h and h given his private and all available public information.

For each player i ∈ N , an information partition I i is a set of all such information sets.

Let A(I) = A(h) and P (I) = P (h) for all h ∈ I and each I ∈

I i .

Each player i chooses actions according to a behavioural strategy σ i , with σ i (I, a) being the probability of choosing action a when in I. We refer to a tuple (σ 1 , σ 2 ) as a strategy profile σ.

Let π σ (h) be the probability of reaching history h if both players follow σ and let π σ i (h) be the probability of reaching h if player i acts according to σ i and player −i always acts deterministically to get to h. It follows that the probability of reaching an information set I if both players follow σ is π σ (I) = h∈I π σ (h) and is π σ i (I) = h∈I π σ i (h) if −i plays to get to I. Player i's expected utility from any history h assuming both players follow strategy profile σ from h onward is denoted by u σ i (h).

Thus, their expected utility over the whole game given a strategy profile σ can be written as u

Finally, a strategy profile σ = (σ 1 , σ 2 ) is a Nash equilibrium if no player i could increase their expected utility by deviating from σ i while −i plays according to σ −i .

We measure the exploitability e(σ) of a strategy profile by how much its optimal counter strategy profile (also called best response) can beat it by.

Let us denote a function that returns the best response to σ i by BR(σ i ).

Formally,

Counterfactual Regret Minimization (CFR) (Zinkevich et al., 2008) is an iterative algorithm.

It can run either simultaneous or alternating updates.

If the former is chosen, CFR produces a new iteration-strategy σ t i for all players i ∈ N on each iteration t. In contrast, alternating updates produce a new strategy for only one player per iteration, with player t mod 2 updating his on iteration t.

To understand how CFR converges to a Nash equilibrium, let us first define the instantaneous regret for player i of action a ∈ A(I) in any I ∈

I i as

where v

.

Intuitively, r t i (I, a) quantifies how much more player i would have won (in expectation), had he always chosen a in I and played to get to I but according to σ t thereafter.

The overall regret on iteration T is R (I, a) .

Now, the iteration-strategy for player i can be derived by

where

|A(I)| .

The iteration-strategy profile σ t does not converge to an equilibrium as t → ∞ in most variants of CFR 1 .

The policy that has been shown to converge to an equilibrium profile is the average strategȳ σ T i .

For all I ∈

I i and each a ∈ A(I) it is defined as

Aiming to solve ever bigger games, researchers have proposed many improvements upon vanilla CFR over the years (Tammelin et al., 2015; Moravčík et al., 2017) .

These improvements include alternative methods for regret updates (Tammelin, 2014; , automated schemes for abstraction design (Ganzfried & Sandholm, 2014) , and sampling variants of CFR (Lanctot et al., 2009 ).

Many of the most successful algorithms of the recent past also employ real-time solving or re-solving Moravčík et al., 2017) .

Discounted CFR (DCFR) slightly modifies the equations for R T i (I, a) andσ T i .

A special case of DCFR is linear CFR (LCFR), where the contribution of the instantaneous regret of iteration t as well as the contribution of σ t toσ T is weighted by t. This change alone suffices to let LCFR converge up to two orders of magnitude faster than vanilla CFR does in some large games.

Monte-Carlo CFR (MC-CFR) (Lanctot et al., 2009 ) proposes a family of tabular methods that visit only a subset of information sets on each iteration.

Different variants of MC-CFR can be constructed by choosing different sampling policies.

One such variant is External Sampling (ES), which executes all actions for player i, the traverser, in every I ∈

I i but draws only one sample for actions not controlled by i (i.e. those of −i and chance).

In games with many player-actions Average Strategy Sampling , Robust Sampling (Hui et al., 2018) are very useful.

They, in different ways, sample only a sub-set of actions for i. Both LCFR and a similarly fast variant called CFR + (Tammelin, 2014) are compatible with forms of MC sampling, although CFR + was regarded as to sensitive to variance until recently (Schmid et al., 2018) .

CFR methods either need to run on the full game tree or employ domain-specific abstractions.

The former is infeasible in large games and the latter not easily possible in all domains.

Deep CFR computes an approximation of linear CFR with alternating player updates.

It is sample-based and does not need to store regret tables, making it generally applicable to any two-player zero-sum game.

On each iteration, Deep CFR fits a value networkD i for one player i to approximate what we call advantage, which is defined as D

, where

In large games, reach-probabilities naturally are (on average) very small after many tree-branchings.

Considering that it is hard for neural networks to learn values across many orders of magnitude (van Hasselt et al., 2016) , Deep CFR divides R We can derive the iteration-strategy for t + 1 from D t similarly to CFR in equation 2 by

However, Deep CFR modifies this to heuristically choose the action with the highest advantage whenever ã∈A(I) D t i (I,ã) + ≤ 0.

Deep CFR obtains the training data forD via batched external sampling (Lanctot et al., 2009; .

All instantaneous regret values collected over the N traversals are stored in a memory buffer B v i .

After its maximum capacity is reached, B v i is updated via reservoir sampling (Vitter, 1985) .

To mimic the behaviour of linear CFR, we need to weight the training losses between the predictionsD makes and the sampled regret vectors in B v i with the iteration-number on which a given datapoint was added to the buffer.

At the end of its training procedure (i.e. after the last iteration), Deep CFR fits another neural network S i (I, a) to approximate the linear average strategȳ

Data to trainŜ i is collected in a separate reservoir buffer B .

Like before, we also need to weight the training loss for each datapoint by the iteration-number on which the datapoint was created.

Notice that tabular CFR achieves importance weighting between iterations through multiplying with some form of the reach probability (see equations 1 and 3).

In contrast, Deep CFR does so by controlling the expected frequency of datapoints from different iterations occurring in its buffers and by weighting the neural network losses differently for data from each iteration.

Notice that storing all iteration-strategies would allow one to compute the average strategy on the fly during play both in tabular and approximate CFR variants.

In tabular methods, the gain of not needing to keepσ in memory during training would come at the cost of storing t equally large tables (though potentially on disk) during training and during play.

However, this is very different with Deep CFR.

Not aggregating intoŜ removes the sampling-and approximation error that B s andŜ introduce, respectively.

Moreover, the computational work needed to trainŜ is no longer required.

Like in the tabular case, we do need to keep all iteration strategies, but this is much cheaper with Deep CFR as strategies are compressed within small neural networks.

We will now look at two methods for queryingσ from a buffer of past value networks B M .

Often (e.g. in one-one-one evaluations and during rollouts), a trajectory is played from the root of the game-tree and the agent is only required to return action-samples of the average strategy on each step forward.

In this case, SD-CFR chooses a value networkD

i at the start of the game, where eachD t i is assigned sampling weight t. The policy σ i , which this network gives by equation 4, is now going to be used for the whole game trajectory.

We call this method trajectory-sampling.

By applying the sampling weights when selecting aD i ∈ B M i , we satisfy the linear averaging constraint of equation 5, and by using the same σ i for the whole trajectory starting at the root, we ensure that the iteration-strategies are also weighted proportionally to each of their reach-probabilities in any given state along that trajectory.

The latter happens naturally, sinceD t i of any t produces σ t i , which reaches each information set I with a likelihood directly proportional to π σ t i (I) when playing from the root.

The query cost of this method is constant with the number of iterations (and equal to the cost of querying Deep CFR).

Let us now consider querying the complete action probability distributionσ

Here, I ∈ I means that I is on the trajectory leading to I and a :

I →

I is the action selected in I leading to I.

This computation can be done with at most 2 as many feedforward passes through each network in B M i as player i had decisions along the trajectory to I, typically taking a few seconds in poker when done on a CPU.

If a trajectory is played forward from the root, as is the case in e.g. exploitability evaluation, we can cache the step-wise reach-probabilities on each step I k along the trajectory and compute π

, where a is the action that leads from I k to I k+1 .

This reduces the number of queries per step to at most |B M i |.

SD-CFR always mimicsσ T i correctly from the iteration-strategies it is given.

Thus, if these iterationstrategies were perfect approximations of the real iteration-strategies, SD-CFR is equivalent to linear CFR (see Theorem 2), which is not necessarily true for Deep CFR (see Theorem 1).

As we later show in an experiment, SD-CFR's performance degrades if reservoir sampling is performed on B M after the number of iterations trained exceeds the buffer's capacity.

Thankfully, the neural network proposed to be used for Deep CFR in large poker games has under 100,000 parameters and thus requires under 400KB of disk space.

Deep CFR is usually trained for just a few hundred iterations , but storing even 25,000 such networks on disk would need only 10GB of disk space.

At no point during any computation do we need all networks in memory.

Thus, keeping all value networks will not represent a problem in practise.

Observing that Deep CFR and SD-CFR depend upon the accuracy of the value networks in exactly the same way, we can conclude that SD-CFR is a better or equally good approximation of linear CFR as long as all value networks are stored.

Though this shows that SD-CFR is largely superior in theory, it is not implicit that SD-CFR will always produce stronger strategies empirically.

We will investigate this next.

We empirically evaluate SD-CFR by comparing to Deep CFR and by analyzing the effect of sampling on B M .

Recall that Deep CFR and SD-CFR are equivalent in how they train their value networks.

This allows both algorithms to share the same value networks in our experiments, which makes comparisons far less susceptible to variance over algorithm runs and conveniently guarantees that both algorithms tend to the same Nash equilibrium.

Where not otherwise noted, we use hyperparamters as .

Our environment observations include additional features such as the size of the pot and represent cards as concatenated one-hot vectors without any higher level features, but are otherwise as .

In Leduc Poker, players start with an infinite number of chips.

The deck has six cards of two suits {a, b} and three ranks {J, Q, K}. There are two betting rounds: preflop and flop.

After the preflop, a card is publicly revealed.

At the start of the game, each player adds 1 chip, called the ante, to the pot and is dealt one private card.

There are at most two bets/raises per round, where the bet-size is fixed at 2 chips in the preflop, and 4 chips after the flop is revealed.

If no player folded, the winner is determined via hand strength.

If a player made a pair with the public card, they win.

Otherwise K > Q > J. If both players hold a card of the same rank, the pot is split.

Hyperparameters are chosen to favour Deep CFR as the neural networks and buffers are very large in relation to the size of the game.

Yet, we find that SD-CFR minimizes exploitability better than Deep CFR.

Exact hyperparameters can be found in the supplementary material.

Although we concluded that storing all value networks is feasible, we analyze the effect of reservoir sampling on B M in Figure 1b and find it leads to plateauing and oscillation, at least up to |B M | = 1000.

Figure 2 shows the results of one-one-one matches between SD-CFR and Deep CFR in 5-Flop Hold'em Poker (5-FHP).

5-FHP is a large poker game similar to regular FHP , which was used to evaluate Deep CFR Table 1 : Disagreement between SD-CFR's and Deep CFR's average strategies.

"DEPTH": number of player actions up until the measurement, "ROUND": PF=Preflop, FL=Flop, "DIF MEAN": mean and 95% confidence interval of the absolute differences between the strategies over the "N" occurrences.

"DIF STD": approximate standard deviation of agreement across information sets.

FHP, please refer to .

The neural architecture is as .

Both algorithms again share the same value networks during each training run.

Like , The y-axis plots SD-CFR's average winnings against Deep CFR in milli-big blinds per game (mbb/g) measured every 30 iterations.

For reference, 10 mbb/g is considered a good margin between humans in Heads-Up Limit Hold'em (HULH), a game with longer action sequences, but similar minimum and maximum winnings per game as 5-FHP.

Measuring the performance on iteration t compares how well the SD-CFR averaging procedure would do against the one of Deep CFR if the algorithm stopped training after t iterations B s reached its maximum capacity of 40 million for both players by iteration 120 in all runs.

Before this point, SD-CFR defeats Deep CFR by a sizable margin, but even after that, SD-CFR clearly defeats Deep CFR.

We analyze how far the average strategies of SD-CFR and Deep CFR are apart at different depths of the tree of 5-FHP.

In particular, we measure

We ran 200,000 trajectory rollouts for each player, where player i plays according to SD-CFR's average strategyσ

and −i plays uniformly random.

Hence, we only evaluate on trajectories on which the agent should feel comfortable.

The two agents again share the same value networks and thus approximate the same equilibrium.

We trained for 180 iterations, a little more than it takes for B s and B v to be full for both players.

Table 1 shows that Deep CFR's approximation is good on early levels of the tree but has a larger error in information sets reached only after multiple decision points.

Regression CFR (R-CFR) (Waugh et al., 2015) applies regression trees to estimate regret values in CFR and CFR + .

Unfortunately, despite promising expectations, recent work failed to apply R-CFR in combination with sampling (Srinivasan et al., 2018) .

Advantage Regret Minimization (ARM) (Jin et al., 2017 ) is similar to R-CFR but was only applied to single-player environments.

Nevertheless, ARM did show that regret-based methods can be of interest in imperfect information games much bigger, less structured, and more chaotic than poker.

DeepStack (Moravčík et al., 2017) was the first algorithm to defeat professional poker players in one-on-one gameplay of Heads-Up No-Limit Hold'em Poker (HUNL) requiring just a single GPU and CPU for real-time play.

It accomplished this through combining real-time solving with counterfactual value approximation with deep networks.

Unfortunately, DeepStack relies on tabular CFR methods without card abstraction to generate data for its counterfactual value networks, which could make applications to domains with many more private information states than HUNL has difficult.

Neural Fictitious Self-Play (NFSP) (Heinrich & Silver, 2016) was the first algorithm to soundly apply deep reinforcement learning from single trajectory samples to large extensive-form games.

While not showing record-breaking results in terms of exploitability, NFSP was able to learn a competitive strategy in Limit Texas Hold'em Poker over just 14 GPU/days.

Recent literature elaborates on the convergence properties of multi-agent deep reinforcement learning (Lanctot et al., 2017) and introduces novel actor-critic algorithms (Srinivasan et al., 2018) that have similar convergence properties as NFSP and SD-CFR.

So far, Deep CFR was only evaluated in games with three player actions.

Since external sampling would likely be intractable in games with tens or more actions, one could employ outcome sampling (Lanctot et al., 2009 ), robust sampling (Hui et al., 2018) , Targeted CFR (Jackson, 2017) , or average-strategy-sampling in such settings.

To avoid action translation after training in an action-abstracted game, continuous approximations of large discrete action-spaces where actions are closely related (e.g. bet-size selection in No-Limit Poker games, auctions, settlements, etc.) could be of interest.

This might be achieved by having the value networks predict parameters to a continuous function whose integral can be evaluated efficiently.

The iteration-strategy could be derived by normalizing the advantage clipped below 0.

The probability of action a could be calculated as the integral of the strategy on the interval corresponding to a in the discrete space.

Given a few modifications to its neural architecture and sampling procedure, SD-CFR could potentially be applied to much less structured domains than poker such as those that deep reinforcement learning methods like PPO (Schulman et al., 2017) are usually applied to.

A first step on this line of research could be to evaluate whether SD-CFR is preferred over approaches such as (Srinivasan et al., 2018) in these settings.

We introduced Single Deep CFR (SD-CFR), a new variant of CFR that uses function approximation and partial tree traversals to generalize over the game's state space.

In contrast to previous work, SD-CFR extracts the average strategy directly from a buffer of value networks from past iterations.

We show that SD-CFR is more attractive in theory and performs much better in practise than Deep CFR.

B v and B s have a capacity of 1 million for each player.

On each iteration, data is collected over 1,500 external sampling traversals and a new value network is trained to convergence (750 updates of batch size 2048), initialized randomly at t < 2 and with the weights of the value net from iteration t − 2 afterwards.

Average-strategy networks are trained to convergence (5000 updates of batch size 2048) always from a random initialization.

All networks used for this evaluation have 3 fully-connected layers of 64 units each, which adds up to more parameters than Leduc Hold'em has states.

All other hyperparameters were chosen as in .

Leduc Hold'em Poker is a two-player game, were players alternate seats after each round.

At the start of the game, both players add 1 chip, the ante, to the pot and are dealt a private card (unknown to the opponent) from a deck consisting of 6 cards: {A, A, B, B, C, C}. There are two rounds: pre-flop and flop.

The game starts at the pre-flop and transitions to the flop after both players have acted and wagered the same number of chips.

At each decision point, players can choose an action from a subset of {fold,call, raise}. When a player folds, the game ends and all chips in the pot are awarded to the opponent.

Calling means matching the opponent's raise.

The first player to act in a round has the option of checking, which is essentially a call of zero chips.

Their opponent can then bet or also check.

When a player raises, he adds more chips to the pot than his opponent wagered so far.

In Leduc Hold'em, the number of raises per round is capped at 2.

Each raise adds 2 additional chips in the pre-flop round and 4 in the flop round.

On the transition from pre-flop to flop, one card from the remaining deck is revealed publicly.

If no player folded and the game ends with a player calling, they show their hands and determine the winner by the rule that if a player's private card matches the flop card, they win.

Otherwise the player with the higher card according to A B C wins.

t i is the acting policy, this result also shows that an opponent cannot tell whether the agent is using this sampling method or following an explicitly computedσ T i

We conducted experiments searching to investigate the harm caused by the function approximation ofŜ. We found that in variants of Leduc Hold'em (Southey et al., 2005) with more that 3 ranks and multiple bets, the performance between Deep CFR and SD-CFR was closer.

Below we plot the exploitability curves of the early iterations in a variant of Leduc that uses a deck of 12 ranks and allows a maximum of 6 instead of 2 bets per round.

We believe the smaller difference in performance is due to the equilibrium in this game being less sensitive to small differences in action probabilities, while the game is still small enough to see every state often during training.

In vanilla Leduc, slight deviations from optimal play give away a lot about one's private information as there are just three distinguishable cards.

In contrast, this variant of Leduc, despite having more states, might be less susceptible to approximation error as it has 12 distinguishable cards but similarly simple rules.

For the plot below, we ran Deep CFR and SD-CFR with shared value networks, where all buffers have a capacity of 4 million.

On each iteration, data is collected over 8,800 external sampling traversals and a new value network is trained to convergence (1200 updates of batch size 2816), initialized randomly at t < 2 and with the weights of the value net from iteration t − 2 afterwards.

Average-strategy networks are trained to convergence (10000 updates of batch size 5632) from a random initialization.

The network architecture used is as , differing only by the card-branch having 64 units per layer instead of 192.

<|TLDR|>

@highlight

Better Deep Reinforcement Learning algorithm to approximate Counterfactual Regret Minimization