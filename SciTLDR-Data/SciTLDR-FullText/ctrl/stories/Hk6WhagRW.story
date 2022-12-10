Multi-agent reinforcement learning offers a way to study how communication could emerge in communities of agents needing to solve specific problems.

In this paper, we study the emergence of communication in the negotiation environment, a semi-cooperative model of agent interaction.

We introduce two communication protocols - one grounded in the semantics of the game, and one which is a priori ungrounded.

We show that self-interested agents can use the pre-grounded communication channel to negotiate fairly, but are unable to effectively use the ungrounded, cheap talk channel to do the same.

However, prosocial agents do learn to use cheap talk to find an optimal negotiating strategy, suggesting that cooperation is necessary for language to emerge.

We also study communication behaviour in a setting where one agent interacts with agents in a community with different levels of prosociality and show how agent identifiability can aid negotiation.

How can communication emerge?

A necessary prerequisite is a task that requires coordination between multiple agents to solve, and some communication protocol for the agents to exchange messages through (see a review by BID41 on earlier work on emergent communication as well as recent deep reinforcement learning methods by BID8 and BID39 ).

Given these basic requirements, an interesting question to ask is what task structures aid the emergence of communication and how different communication protocols affect task success.

In the context of linguistic communication, previous work on this subject has mainly studied the emergence of communication in co-operative games like referential games, variants of the Lewis signaling game BID19 , where messages are used to disambiguate between different possible referents BID11 BID17 BID6 .

Human language, though, is not merely a referential tool.

Amongst other things, we communicate private information and thoughts, discuss plans, ask questions and tell jokes.

Moreover, many human interactions are not fully cooperative, yet we can still successfully use language to communicate in these situations.

In this paper, we study communication in the negotiation game (see Figure 1 ), an established model of non-cooperative games in classical game theory BID25 BID26 BID24 BID23 BID35 BID1 BID29 .

In this game, agents are asked to establish a mutually acceptable division of a common pool of items while having their own hidden utilities for each of them.

Effective communication is crucial in this game, as the agents need to exchange strategic information about their desires, infer their opponent's desires from communication, and balance between the two.

Work in classical game theory on negotiation typically uses simple forms of offer / counter-offer bargaining games that do not explicitly address the question of emergent communication (Rubin- Figure 1 : High-level overview of the negotiation environment that we implement.

Agent A consistently refers to the agent who goes first.

stein, 1982) .

Recent work on deep multi-agent reinforcement learning (MARL) has shown great success in teaching agents complex behaviour without a complex environment simulator or demonstration data BID28 BID2 BID40 BID18 .

By repeatedly interacting with other agents learning at the same time, agents can gradually bootstrap complex behaviour, including motor skills BID0 and linguistic communication BID17 BID13 .We apply techniques from the MARL literature and train agents to negotiate using task success as the only supervision signal.

1 We show that, when communicating via a task-specific communication channel with inherent semantics, selfish agents can learn to negotiate fairly and divide up the item pool to the agents' mutual satisfaction.

However, when communicating via cheap talk BID4 BID7 , a task-independent communication channel consisting of sequences of arbitrary symbols similar to language, selfish agents fail to exhibit negotiating behaviour at all.

On the other hand, we show that cheap talk can facilitate effective negotiation in prosocial agents, which take the other agent's reward into consideration, providing experimental evidence that cooperation is necessary for language emergence BID27 .The above results are obtained from paired agents interacting exclusively with each other.

In more realistic multi-agent scenarios, agents may interact with many other agents within a society.

In these cases, cheap talk can have a significant effect on the evolutionary dynamics of the population BID32 ) as well as the equilibria, stability, and basins of attractions BID38 .

Furthermore, it is well-known that, unless trained in a diverse environment, agents overfit to the their specific opponent or teammate .

Inspired by these considerations, we perform experiments where agents interact with many agents having different prosociality levels, and find that being able to identify and model other agents' beliefs aids the negotiation success.

This is consistent with experiments using models based on Theory of Mind: boundedly rational agents can collectively benefit by making inferences about the sophistication levels and beliefs of their opponents, and there is evidence that this occurs in human behavior BID44 2 GAME SETTING

The negotiation task is based on the set-up of BID20 , itself based on BID5 .

Agents are presented with three types of items: peppers, cherries and strawberries.

At each round (i) an item pool is sampled uniformly, instantiating a quantity (between 0 and 5) for each of the types and represented as a vector i ∈ {0...5} 3 and (ii) each agent j receives a utility function sampled uniformly, which specifies how rewarding one unit of each item is (with item rewards between 0 and 10, and with the constraint that there is at least one item with non-zero utility), represented as a vector u j ∈ {0...10}3 .

These utilities are hidden, i.e., each agent only has access to its own utilities.

Following these assignments, the agents start "negotiating" for N timesteps by exchanging messages m j t and proposals p j t ∈ {0...

5}3 for each timestep 1 ≤ t ≤ N .

Agents alternate between timesteps, meaning that agent A always acts on the odd turns while agent B on the even turns.

Either agent can terminate the negotiation at any timestep t with a special action, which signifies agreement with the most recent proposal made by the other agent.

For example, if agent B terminates the negotiation at t = 4, agent A gets rewarded by DISPLAYFORM0 , where · denotes the dot product of vectors.

Note that, if an agent makes an invalid proposal, such as asking for more items than exist in the item pool, and this is accepted, then both agents receive R A = R B = 0.We impose an upper limit, N , on the number of negotiation turns allowed.

If the agents get to the limit without agreement, then both agents receive no reward.

We experimented with having a fixed upper limit of 10 turns.

However, this led to a "first-mover" advantage: if the agents know in advance how many turns the negotiation lasts, there is a degenerate policy in which agent A can wait until the last timestep N − 1 before giving a lopsided offer, which agent B is compelled to accept, a situation similar to the ultimatum game BID12 .

To eliminate this effect, we sample N between 4 and 10 at each round, according to a truncated Poisson distribution with mean 7.

To achieve negotiation, the agents need to communicate.

One obvious communication protocol is to directly transmit the proposed division of items p j .

We refer to this as the proposal channel.

This communication channel is task-specific and pertains to the action space of the agents used to derive their pay-offs.

Consequently, the information bandwidth is restricted and grounded in the action space of the game.

Motivated by work on emergent communication, we also give our agents the ability of transmitting strings of arbitrary symbols with no a priori grounding.

We call this the linguistic channel, a specific instantiation of the more general concept of cheap talk.

This channel differs from the proposal channel in two key properties:• Non-bindingness: Messages sent via this channel do not commit the sender to any course of action, unlike directly transmitting a proposal which binds the sender to the proposal.• Unverifiability: There is no inherent link between the linguistic utterance and the private proposal made, meaning that the agents could potentially lie.

We are interested in whether and under what circumstances the agents can make use of this channel to facilitate negotiation and establish a common ground for the symbols.

In our experiments, we consider 4 separate communication configurations: only the proposal channel, only the linguistic channel, both channels open, and no communication at all.

2 If a channel is closed, we replace any messages in that channel with a fixed dummy symbol to ensure no information is transmitted.

If the proposal channel is closed, we still use the decoded proposal to calculate the item division, but this information is not revealed to the opponent agent.

While purely self-interested agents may learn to divide up items so that only each individual is satisfied, this may not lead to an optimal joint allocation of items, where each item goes to the agent with the highest utility for that item.

We introduce a "prosocial" reward R that is shared across A and B, which is the sum of the agents' individual (selfish) rewards, i.e., R = R A + R B .

Prosocial agents are incentivised to communicate, as finding the optimal joint allocation requires communicating the hidden values between the agents.

These different reward schemes can be seen as specific instantiations of a more general reward formula of the form R = αR A +βR B BID30 .

For the selfish agent, α = 1 and β = 0, while for the prosocial agent α = β = 1.

At each timestep t, the proposer receives three inputs:• The item context c j = [i; u j ], a concatenation of the item pool and the proposer's utilities.• The utterance m t−1 produced by the other agent in the previous timestep t − 1.

If the linguistic channel is closed, this is simply a dummy message.• The proposal p t−1 made by the other agent in the previous timestep t − 1.

If the proposal channel is closed, this is again a dummy proposal.

First, the discrete inputs are turned into dense vectors through an embedding table.

We use two embedding tables, one for the item context and the previous proposal, and a separate one for the previous utterance.

This is essential as the item context and the previous proposal have predefined numeric semantics, distinct from the linguistic semantics of the utterances' symbols.

Then, each of the input sequences is encoded using an LSTM BID14 , one for each input.

This results in 3 fixed-size vectors: h c t , h m t and h p t .

These vectors are concatenated and fed through a feedforward layer, followed by a ReLU non-linearity BID22 , to give h t , the hidden state of the agent at timestep t. This hidden state is then used to initialise the policies for the actions the agent can take:• π term is the policy for the termination action.

If this action is taken by an agent, both agents receive reward according to the last proposal made by the other agent.

This is a binary decision, and we parametrise π term as a single feedforward layer, with the hidden state as input, followed by a sigmoid function, to represent the probability of termination.• π utt is the policy for the linguistic utterances.

This is parametrised by an LSTM, which takes the hidden state of the agent as the initial hidden state.

For the first timestep, a dummy symbol is fed in as input; subsequently, the model prediction from the previous timestep is fed in as input at the next timestep, in order to predict the next symbol.• π prop is the policy for the proposals the agent generates.

This is parametrised by 3 separate feedforward neural networks, one for each item type, which each take as input h t and output a distribution over {0...5} indicating the proposal for that item.

The overall policy for the agent, π * , is a combination of the separate policies.

The action that the agent takes at turn t can be summarised by a triple (e t , m t , p t ), where e t is a binary variable indicating whether the agent took the termination action, m t is a sequence of symbols produced from π utt , and p t is a proposal produced by π prop .

A negotiation can then be thought of as a sequence of triples τ = (e t , u t , p t )|1 ≤ t ≤ N , with the agents alternating turns.

During training, each agent i tries to independently find the policy π * i that maximises the following objective function: DISPLAYFORM0 where R i (τ ) is the reward agent i receives from the trajectory τ and H(π i ) is an entropy regularisation term to encourage the agent to explore during training.

Note that each agent has its own objective function, and the objectives of both agents are coupled by the trajectory sampled from both agents' policies.

At test time, instead of sampling from the policy π, we take the action with the highest probability.

Parameters are updated using the REINFORCE BID42 update rule with an exponentially smoothed mean baseline.

Full hyperparameter details are in the appendix.

In our first experiment, we test whether purely self-interested agents can learn to negotiate and divide up items fairly, and investigate the effect of the various communication channels on negotiation success.

We train self-interested agents to negotiate for 500k episodes.

Each episode corresponds to a batch of 128 games, each with item pools and hidden utilities generated as described in Section 2.1.

We also hold out 5 batches worth of games as a test set, and test the agents every 50 episodes, in order to measure the robustness of the negotiating behaviour to unseen environments.

Self-interested agents can learn to negotiate fairly FIG1 shows that self-interested agents learn to divide up the items fairly when they exchange proposals directly.

We see that the proportion of total utility each agent receives is roughly equal, and above 50%, suggesting that the agents have learnt to keep items with higher utility, while giving away items with low utility.

4 In addition, the agents seem to show evidence of compromise: there is a drop of 0.24 between the value of the proposal that agent 1 makes at the beginning of negotiation, and the final reward it receives; for agent 2, the corresponding drop is 0.18.

Furthermore, FIG1 shows the mean optimality of the proposal made at each turn of the negotiation averaged over 1280 games, calculated as the sum of the reward of both agents divided by the maximum possible reward.

This value increases with more rounds of negotiation, suggesting that the agents are adjusting their proposals based on information from the other agent.

Table 1 presents an illustrative example of two self-interested agents negotiating using the proposal channel.

While the initial proposals made by both agents are over-optimistic, over the course of the negotiation, they compromise to less items.

This allows them to achieve higher joint reward, as seen in FIG1 , a clear example of "negotiation".Self-interested agents do not appear to ground cheap talk When using the linguistic channel, agents do not negotiate optimally.

Instead, the agents randomly alternate taking all the items, which is borne out by the oscillations in FIG1 .

Indeed, examination of the messages exchanged by self-interested agents show they predominantly consist of a single symbol -see FIG4 and 5b in the appendix for more details.

This result suggests that the self-interested agents we implement Table 2 : Joint reward success and average number of turns taken for paired agents negotiating with random game termination, varying the agent reward scheme and communication channel.

The results are averaged across 20 seeds, with 128 games per seed.

We also report the standard deviation as the ± number and the interquartile range.find it difficult to cheap talk to exchange meaningful information about how to divide up the items, which is in line with the theoretical analysis of cheap talk in BID4 , who find that, once agent interests diverge by a finite amount, no communication is to be expected.

Our findings also corroborate the hypothesis found in the language evolution literature that cooperation is a prerequisite for the emergence of language BID27 .We conjecture that this finding is partially due to the fact that the agents operate in a non-iterated environment (i.e., they have no explicit memory of previous interactions).

In the classic prisoner's dilemma game, rational agents playing a single-shot version of the game learn to always defect, whereas rational agents playing iterated prisoner's dilemma can learn more mutually beneficial cooperative strategies under the right conditions BID34 BID43 .

More sophisticated players will often have an incentive to take a short-term loss in reward to teach or punish another player for long-term gain in the repeated setting BID10 .

We suspect that, by iterating the game, there is an incentive for the cheap talk to be more verifiable, as there is a more credible threat of punishment if the accepted proposal and the linguistic utterance do not correspond.

In the previous experiment, we showed that self-interested agents can divide up items fairly such that each agent gets equal reward.

However, this does not correspond to an optimal joint allocation where items go to the agent with the higher utility for that item (equivalently, maximising the sum of both agents' rewards).

The prosocial reward scheme aligns exactly with this optimal outcome, as learning to negotiate equates to finding the optimal joint item allocation.

In this setting, there is an in-built bias towards communication, as solving the game optimally requires pooling hidden utilities.

Table 2 reports the turns taken as well as the joint reward optimality, i.e., the sum of both agents' rewards, divided by the maximum possible reward given the item context, where the latter corresponds to each agent getting all the items for which they have a higher utility.

Table 3 : Actual transcript of two prosocial agents negotiating using the linguistic channel.

At each turn, the agents decode a proposal but this is hidden and not revealed to the other agent.

Cheap talk helps agents coordinate For prosocial agents with aligned interests, theoretical results suggest that communication using cheap talk is a Nash equilibrium.

This is indeed what we observe, as the linguistic channel results in much better task success than any other communication scheme.

The information bandwidth of the proposal channel is limited by the type and number of items, whereas the linguistic channel's bandwidth is unconstrained and can (in theory) be arbitrarily large, helping the effective exchange of task-specific information.

In addition, the variance in joint optimality is much lower for the linguistic channel, suggesting it is more robust.

We suspect that difficulties in optimisation cause the results for both channels open to be similar to those for only the proposal channel open, as the proposal channel is pre-grounded, which poses a strong local maxima.

With better optimisation, we believe that the performance of both channels open will more closely match those of just the linguistic channel open.

Table 3 illustrates an example of negotiation between the two prosocial agents.

Interestingly, we observe that agent A over the course of the negotiation game adjusts its proposal (which is however hidden and not revealed) based on the utterances of agent B.Prosocial agents can still use the proposal channel to co-ordinate With random termination, agents communicating with the proposal channel do even worse than the no-communication baseline.

However, when given the full 10 turns to exchange information (see Figure 7 in the appendix), prosocial agents manage to outperform the no-communication baseline.

This suggests that the agents are slowly learning to repurpose the proposal channel to transmit information, i.e., they communicate directly on the action space.

Reusing task-specific actions to transmit information has precedent, such as in bridge bidding BID37 , and it is interesting to observe that agents learn to develop a codebook without the need for prior agreement.

The symbol unigram and bigram distributions of messages exchanged by prosocial agents (see FIG2 show that agent A, the agent who initiates the negotiation, is not transmitting any information using the linguistic channel.

On the other hand, agent B uses a diversity of symbols resulting in a long-tailed bigram symbol distribution, reminiscent of the Zipfian distribution found in natural languages.

This suggests that, even though the task is symmetric, the agents differentiate into speaker and listener roles, similar to other co-ordination games of communication, such as the reference game.

Thus, they have adopted a simple strategy for solving the game: an agent shares its utilities (the speaker), while the other (the listener) compares the shared utilities with their own and generates the optimal proposal.

In contrast, the selfish agents do not show evidence of grounded symbol usage.

The unigram and bigram statistics show that most messages consist solely of strings of a single fixed symbol, regardless of the item context, and hence no information is being exchanged.

Table 4 : Accuracy of predicting individual elements of the agents' hidden utilities, as well as the final proposal that was accepted.

Numbers in brackets indicate accuracy predicting the proposal from just the item pool.

Random baseline numbers are obtained by predicting from a message transcript and item pool of all 0's.

To interpret the information that is transmitted through the linguistic channel, we use the message transcript and the item pool to create probe classifiers that predict the hidden utilities of each agent and the (hidden) accepted proposal.

We use an LSTM to encode the sequence of symbols belonging to the message transcript into a vector, and another LSTM to encode the item pool into a vector.

The resulting two vectors are then concatenated and used to predict the hidden utility functions and the accepted proposal using 9 linear classifiers, one for each item (3 each for agent A and agent B's utilities and 3 for the proposal).

We conduct 10-fold cross-validation and report averaged per-item accuracy (see Table 4 ).We also include two baselines.

The first, reported as the "random baseline" in Table 4 , attempts to predict the accepted proposal and agent utilities from a message transcript and item pool of all 0's (indicating chance performance at the task).

The other predicts the accepted proposal from the item pool and a message transcript of all 0's.

This shows how much additional information about the proposal is contained in the message transcript.

We can see from Table 4 that not only is there variety in the symbols used, but the messages also have semantic content: they contain information about the hidden utilities of agent B, and about the proposal that was made.

This shows that our agents have learned to give meaning to the symbols, and can use them to transmit information.

However, the purely self-interested agents do not seem to transmit meaningful information using the linguistic channel, and thus do not appear to ground the symbols.

In more realistic scenarios, we learn and practice negotiation in environments with diverse agent populations and levels of prosociality.

In this setting, maximising one's reward requires identifying which agents are most useful in achieving one's aims.

In practice, this involves identifying which agents are the most prosocial, as they are both the most exploitable for self-interested agents, and the most cooperative for other prosocial agents.

We trained a fixed agent against a community of 10 agents, with varying proportions of selfish and prosocial agents.

For each training episode, we randomly sample one agent from the community, play a batch of negotiation games, and then update both agents.

We do this for both fixed agent A (the agent who goes first) and for fixed agent B (the agent who goes second).

We experiment with either 1 or 5 prosocial agents, which communication channels are open, and also whether the agents are identifiable.

At test time, we care primarily about whether self-interested agents can exploit prosocial agents and whether prosocial agents can cooperate with other prosocial agents.

Hence, we only test the fixed agent against the prosocial agents in the community.

We generate 10 batches of 128 games each batch, and play each game with every pair of fixed agent and community prosocial agent.

We average the results across all pairs and all games, and show the mean and standard deviations of the rewards obtained in this experiment in Table 5 .If opposing agents are identifiable (i.e. each agent has a name-tag), the fixed agent receives this information as a one-hot vector, which is then used to look up an embedding table with trainable parameters with one embedding per opposing agent.

This opponent embedding is then concatenated with the encoded inputs, before being passed through the feedforward layer to get the agent's hidden state.

Intuitively, the induced agent embedding, that takes the form of machine theory of mind BID31 , should capture any information about the opponent's behaviour (e.g., negotiating strategy) that, when taken into account, can help an agent maximize their reward.

Identifying agents Our results show that providing agent ID information does help the fixed agent achieve its aims.

This effect is particularly pronounced for a selfish fixed agent; here, providing ID information uniformly improves performance.

For cooperative agents, the results mixed; for a fixed agent A communicating via the linguistic channel, having agent ID information harms performance.

Indeed, the only prosocial agents who beat the no communication baseline (which achieved an average joint optimal reward of 0.95) was when agent A was fixed and ID information was not provided.

We also computed the 2D PCA projections of the learnt opponent embeddings for a fixed agent A under a variety of channels and prosocial levels (see FIG3 .

Even in cases where the agent ID does not aid negotiation, we find that the embeddings cluster according to the reward scheme of the other agent.

This shows that the agents can distinguish different agents by their reward schemes from purely observing negotiating behaviour.

Community linguistic phenomena In one of our experimental settings, a community of prosocial agents developed a language and were able to use this to achieve better negotiation success.

This is the outcome in the starred cell in Table 5 .

For a visualisation of the bigram usage statistics in Table 5 : Fixed agent return vs opposing prosocial agents, for different numbers of opposing prosocial agents and whether IDs are revealed across the communication channels.

For the self-interested fixed agent, the numbers report personal reward, and for the prosocial fixed agent, the numbers report joint reward.

The starred result indicates when a community has made use of the linguistic channel to transmit information.each interaction pair in this community, see FIG5 .

Interestingly, the only situation when agents developed a language was when ID information was not provided.

We suspect this result is due to a poor local optimum; since the prosocial agents can learn to use IDs to distinguish prosocial agents from the selfish agents they can ignore the linguistic utterance.

Moreover, we find that when prosocial agents make use of the linguistic channel, the communication protocol differs within the community.

We calculate the Spearman correlation ρ between bigram ranks of different pairs of agents, all of them with the same fixed agent A, and show the results in TAB3 .

Even though all pairs of agents share the same fixed agent A, different pairs of agents learn to use bigrams in a different way, resulting in relatively low correlations between -0.22 and 0.27.

We showed that by communicating through a verifiable and binding communication channel, selfinterested agents can learn to negotiate fairly by reinforcement learning, using only task success as the reward signal.

Moreover, cheap talk facilitated negotiation in prosocial but not in self-interested agents, corroborating theoretical results from the game theory literature BID4 ).

An interesting future direction of research would be to investigate whether cheap talk can be made to emerge out of self-interested agents interacting.

Recent encouraging results by BID3 show that communication can help agents cooperate.

However, their signalling mechanism is heavily engineered: the speech acts are predefined and the consequences of the speech acts on the observed behaviour are deterministically hard-coded.

It would be interesting to see whether a learning algorithm, such as BID9 , can discover the same result.

A related paper from BID20 takes a top-down approach to learning to negotiate by leveraging dialogue data.

We demonstrated a bottom up alternative towards learning communicative behaviours directly from interaction with peers.

This opens up the exciting possibility of learning domain-specific reasoning capabilities from interaction, while having a general-purpose language layer at the top producing natural language.

A ADDITIONAL FIGURES AND TABLES Table 7 : Joint reward success and average number of turns taken for paired agents negotiating when allowed the full 10 turns, varying the agent reward scheme and communication channel.

The results are averaged across 20 seeds, with 128 games per seed.

We also report the standard deviation as the ± number and the quartiles.

Embedding sizes, and all neural network hidden states, had dimension 100.

We used the ADAM optimizer BID15 , with default parameter settings, to optimize the parameters of each agent.

Each agent had a separate optimizer.

We used a separate value of λ, the entropy regularisation weight hyperparameter, for each policy.

For π term and π prop , λ = 0.05; for π utt , λ = 0.001.

The symbol vocabulary size was 11, and the agents were allowed to generate utterances of up to length 6.

The smoothing constant for the exponential moving average baseline was 0.7 (i.e. if the old baseline value was b old , and the current reward is R, then the new estimate of the baseline is b new = 0.7b old + 0.3R).

<|TLDR|>

@highlight

We teach agents to negotiate using only reinforcement learning; selfish agents can do so, but only using a trustworthy communication channel, and prosocial agents can negotiate using cheap talk.

@highlight

The authors describe a variant of the negotation game with the consideration of a secondary communication channel for cheap talk, finding that the secondary channel improves negotation outcomes.

@highlight

This paper explores how agents can learn to communicate to solve a negotiation task and find that prosocial agents are able to learn to ground symbols using RL, but self-interested agents are not.

@highlight

Examines problems of how agents can use communication to maximise their rewards in a simple negotiation game.