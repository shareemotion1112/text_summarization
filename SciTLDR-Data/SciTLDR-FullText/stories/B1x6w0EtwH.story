Interactive Fiction games are text-based simulations in which an agent interacts with the world purely through natural language.

They are ideal environments for studying how to extend reinforcement learning agents to meet the challenges of natural language understanding, partial observability, and action generation in combinatorially-large text-based action spaces.

We present KG-A2C, an agent that builds a dynamic knowledge graph while exploring and generates actions using a template-based action space.

We contend that the dual uses of the knowledge graph to reason about game state and to constrain natural language generation are the keys to scalable exploration of combinatorially large natural language actions.

Results across a wide variety of IF games show that KG-A2C outperforms current IF agents despite the exponential increase in action space size.

Natural language communication has long been considered a defining characteristic of human intelligence.

We are motivated by the question of how learning agents can understand and generate contextually relevant natural language in service of achieving a goal.

In pursuit of this objective we study Interactive Fiction (IF) games, or text-adventures: simulations in which an agent interacts with the world purely through natural language-"seeing" and "talking" to the world using textual descriptions and commands.

To progress in these games, an agent must generate natural language actions that are coherent, contextually relevant, and able to effect the desired change in the world.

Complicating the problem of generating contextually relevant language in these games is the issue of partial observability: the fact that the agent never has access to the true underlying world state.

IF games are structured as puzzles and often consist of an complex, interconnected web of distinct locations, objects, and characters.

The agent needs to thus reason about the complexities of such a world solely through the textual descriptions that it receives, descriptions that are often incomplete.

Further, an agent must be able to perform commonsense reasoning-IF games assume that human players possess prior commonsense and thematic knowledge (e.g. knowing that swords can kill trolls or that trolls live in dark places).

Knowledge graphs provide us with an intuitive way of representing these partially observable worlds.

Prior works have shown how using knowledge graphs aids in the twin issues of partial observability (Ammanabrolu & Riedl, 2019a) and commonsense reasoning (Ammanabrolu & Riedl, 2019b ), but do not use them in the context of generating natural language.

To gain a sense for the challenges surrounding natural language generation, we need to first understand how large this space really is.

In order to solve solve a popular IF game such as Zork1 it's necessary to generate actions consisting of up to five-words from a relatively modest vocabulary of 697 words recognized by Zork's parser.

Even this modestly sized vocabulary leads to O(697 5 ) = 1.64 × 10 the structure required to further constrain our action space via our knowledge graph-and make the argument that the combination of these approaches allows us to generate meaningful natural language commands.

Our contributions are as follows: We introduce an novel agent that utilizes both a knowledge graph based state space and template based action space and show how to train such an agent.

We then conduct an empirical study evaluating our agent across a diverse set of IF games followed by an ablation analysis studying the effectiveness of various components of our algorithm as well as its overall generalizability.

Remarkably we show that our agent achieves state-of-the-art performance on a large proportion of the games despite the exponential increase in action space size.

We examine prior work in three broad categories: text-based game playing agents and frameworks as well as knowledge graphs used for natural language generation and game playing agents.

LSTM-DQN (Narasimhan et al., 2015) , considers verb-noun actions up to two-words in length.

Separate Q-Value estimates are produced for each possible verb and object, and the action consists of pairing the maximally valued verb combined with the maximally valued object.

The DRRN algorithm for choice-based games (He et al., 2016; Zelinka, 2018) estimates Q-Values for a particular action from a particular state.

Fulda et al. (2017 ) use Word2Vec (Mikolov et al., 2013 to aid in extracting affordances for items in these games and use this information to produce relevant action verbs.

Zahavy et al. (2018) reduce the combinatorially-sized action space into a discrete form using a walkthrough of the game and introduce the Action Elimination DQN, which learns to eliminate actions unlikely to cause a world change.

Côté et al. (2018) introduce TextWorld, a framework for procedurally generating parser-based games, allowing a user to control the difficulty of a generated game.

Yuan et al. (2019) , an optimized interface for playing human-made IF games-formalizing this task.

They further provide a comparative study of various types of agents on their set of games, testing the performance of heuristic based agents such as NAIL (Hausknecht et al., 2019b) and various reinforcement learning agents are benchmarked.

We use Jericho and the tools that it provides to develop our agents.

Knowledge graphs have been shown to be useful representations for a variety of tasks surrounding natural language generation and interactive fiction.

Ghazvininejad et al. (2017) and Guan et al. (2018) effectively use knowledge graph representations to improve neural conversational and story ending prediction models respectively.

Ammanabrolu et al. (2019) explore procedural content generation in text-adventure games-looking at constructing a quest for a given game world, and use knowledge graphs to ground generative systems trained to produce quest content.

From the perspective of text-game playing agent and most in line with the spirit of our work, Ammanabrolu & Riedl (2019a) present the Knowledge Graph DQN or KG-DQN, an approach where a knowledge graph built during exploration is used as a state representation for a deep reinforcement learning based agent.

Ammanabrolu & Riedl (2019b) further expand on this work, exploring methods of transferring control policies in text-games, using knowledge graphs to seed an agent with useful commonsense knowledge and to transfer knowledge between different games within a domain.

Both of these works, however, identify a discrete set of actions required to play the game beforehand and so do not fully tackle the issue of the combinatorial action space.

Formally, IF games are partially observable Markov decision processes (POMDP), represented as a 7-tuple of S, T, A, Ω, O, R, γ representing the set of environment states, mostly deterministic conditional transition probabilities between states, the vocabulary or words used to compose text commands, observations returned by the game, observation conditional probabilities, reward function, and the discount factor respectively (Côté et al., 2018; Hausknecht et al., 2019a) .

To deal with the resulting twin challenges of partial observability and combinatorial actions, we use a knowledge graph based state space and a template-based action space-each described in detail below.

Knowledge Graph State Space.

Building on Ammanabrolu & Riedl (2019a), we use a knowledge graph as a state representation that is learnt during exploration.

The knowledge graph is stored as a set of 3-tuples of subject, relation, object .

These triples are extracted from the observations using Stanford's Open Information Extraction (OpenIE) (Angeli et al., 2015) .

Human-made IF games often contain relatively complex semi-structured information that OpenIE is not designed to parse and so we add additional rules to ensure that we are parsing the relevant information.

Updated after every action, the knowledge graph helps the agent form a map of the world that it is exploring, in addition to retaining information that it has learned such as the affordances associated with an object, the properties of a character, current inventory, etc.

Nodes relating to such information are shown on the basis of their relation to the agent which is presented on the graph using a "you" node (see example in Fig. 2a ).

Ammanabrolu & Riedl (2019a) build a knowledge graph in a similar manner but restrict themselves to a single domain.

In contrast, we test our methods on a much more diverse set of games defined in the Jericho framework (Hausknecht et al., 2019a) .

These games are each structured differentlycovering a wider variety of genres-and so to be able to extract the same information from all of them in a general manner, we relax many of the rules found in Ammanabrolu & Riedl (2019a) .

To aid in the generalizability of graph building, we introduce the concept of interactive objects-items that an agent is able to directly interact with in the surrounding environment.

These items are directly linked to the "you" node, indicating that the agent can interact with them, and the node for the current room, showing their relative position.

All other triples built from the graph are extracted by OpenIE.

Further details regarding knowledge graph updates are found in Appendix A.1 An example of a graph built using these rules is seen in Fig. 2a .

Template Action Space.

Templates are subroutines used by the game's parser to interpret the player's action.

They consist of interchangeable verbs phrases (V P ) optionally followed by prepositional phrases (V P P P ), e.g. Figure 2b , actions may be constructed from templates by filling in the template's blanks using words in the game's vocabulary.

Templates and vocabulary words are programmatically accessible through the Jericho framework and are thus available for every IF game.

Further details about how we prioritize interchangeable verbs and prepositions are available in Appendix A.2.

Combining the knowledge-graph state space with the template action space, Knowledge Graph Advantage Actor Critic or KG-A2C, is an on-policy reinforcement learning agent that collects experience from many parallel environments.

We first discuss the architecture of KG-A2C, then detail the training algorithm.

As seen in Fig. 1 , KG-A2C's architecture can broadly be described in terms of encoding a state representation and then using this encoded representation to decode an action.

We describe each of these processes below.

Input Representation.

The input representation network is broadly divided into three parts: an observation encoder, a score encoder, and the knowledge graph.

At every step an observation consisting of several components is received: o t = (o t desc , o tgame , o tinv , a t−1 ) corresponding to the room description, game feedback, inventory, and previous action, and total score R t .

The room description o t desc is a textual description of the agent's location, obtained by executing the command "look."

The game feedback o tgame is the simulators response to the agent's previous action and consists of narrative and flavor text.

The inventory o tinv and previous action a t−1 components inform the agent about the contents of its inventory and the last action taken respectively.

The observation encoder processes each component of o t using a separate GRU encoder.

As we are not given the vocabulary that o t is comprised of, we use subword tokenization-specifically using the unigram subword tokenization method described in Kudo & Richardson (2018) .

This method predicts the most likely sequence of subword tokens for a given input using a unigram language and contains a total vocabulary of size 8000.

For each of the GRUs, we pass in the final hidden state of the GRU at step t − 1 to initialize the hidden state at step t. We concatenate each of the encoded components and use a linear layer to combine them into the final encoded observation o t .

At each step, we update our knowledge graph G t using o t as described in Sec. 3 and it is then embedded into a single vector g t .

Following Ammanabrolu & Riedl (2019a) we use Graph Attention networks or GATs (Veličković et al., 2018) with an attention mechanism similar to that described in Bahdanau et al. (2014) .

Node features are computed as

, where N is the number of nodes and F the number of features in each node, consist of the average subword embeddings of the entity and of the relations for all incoming edges using our unigram language model.

Self-attention is then used after a learnable linear transformation W ∈ IR 2F×F applied to all the node features.

Attention coefficients α ij are then computed by softmaxing k ∈ N with N being the neighborhood in which we compute the attention coefficients and consists of all edges in G t .

where p ∈ IR

is a learnable parameter.

The final knowledge graph embedding vector g t is computed as:

where k refers to the parameters of the k th independent attention mechanism, W g and b g the weights and biases of the output linear layer, and represents concatenation.

The final component of state embedding vector is a binary encoding c t of the total score obtained so far in the game-giving the agent a sense for how far it has progressed in the game even when it is not collecting reward.

The state embedding vector is then calculated as s t = g t ⊕ o t ⊕ c t .

Action Decoder.

The state embedding vector s t is then used to sequentially construct an action by first predicting a template and then picking the objects to fill into the template using a series of Decoder GRUs.

This gives rise to a template policy π T and a policy for each object π Oi .

Architecture

You are in the living room.

There is a doorway to the east, a wooden door with strange gothic lettering to the west, which appears to be nailed shut, a trophy case, and a large oriental rug in the center of the room.

Above the trophy case hangs an elvish sword of great antiquity.

A batterypowered brass lantern is on the trophy case.

You are carrying: A glass bottle The glass bottle contains: A quantity of water.

Figure 2: An overall example of the knowledge graph building and subsequent action decoding process for a given state in Zork1, illustrating the use of interactive objects and the graph mask.

wise, at every decoding step all previously predicted parts of the action are encoded and passed along with s t through an attention layer which learns to attend over these representations-conditioning every predicted object on all the previously predicted objects and template.

All the object decoder GRUs share parameters while the template decoder GRU T remains separate.

To effectively constrain the space of template-actions, we introduce the concept of a graph mask, leveraging our knowledge graph at that timestep G t to streamline the object decoding process.

Formally, the graph mask m t = {o : o ∈ G t ∧ o ∈ V }, consists of all the entities found within the knowledge graph G t and vocabulary V and is applied to the outputs of the object decoder GRUsrestricting them to predict objects in the mask.

Generally, in an IF game, it is impossible to interact with an object that you never seen or that are not in your inventory and so the mask lets us explore the action space more efficiently.

To account for cases where this assumption does not hold, i.e. when an object that the agent has never interacted with before must be referenced in order to progress in the game, we randomly add objects o ∈ V to m t with a probability p m .

An example of the graph-constrained action decoding process is illustrated in Fig. 2b .

We adapt the Advantage Actor Critic (A2C) method (Mnih et al., 2016) to train our network, using multiple workers to gather experiences from the simulator, making several significant changes along the way-as described below.

Valid Actions.

Using a template-action space there are millions of possible actions at each step.

Most of these actions do not make sense, are ungrammatical, etc.

and an even fewer number of them actually cause the agent effect change in the world.

Without any sense for which actions present valid interactions with the world, the combinatorial action space becomes prohibitively large for effective exploration.

We thus use the concept of valid actions, actions that can change the world in a particular state.

These actions can usually be recognized through the game feedback, with responses like "Nothing happens" or "That phrase is not recognized." In practice, we follow Hausknecht et al. (2019a) and use the valid action detection algorithm provided by Jericho.

Formally, V alid(s t ) = a 0 , a 1 ...a N and from this we can construct the corresponding set of valid templates

We further define a set of valid objects O valid (s t ) = o 0 , o 1 ...o M which consists of all objects in the graph mask as defined in Sec. 4.

This lets us introduce two cross-entropy loss terms to aid the action decoding process.

The template loss given a particular state and current network parameters is applied to the decoder GRU T .

Similarly, the object loss is applied across the decoder GRU O and is calculated by summing cross-entropy loss from all the object decoding steps.

Updates.

A2C training starts with calculating the advantage of taking an action in a state A(s t , a t ), defined as the value of taking an action Q(s t , a t ) compared to the average value of taking all possible valid actions in that state V (s t ):

V (s t ) is predicted by the critic as shown in Fig. 1 and r t is the reward received at step t.

The action decoder or actor is then updated according to the gradient:

updating the template policy π T and object policies π Oi based on the fact that each step in the action decoding process is conditioned on all the previously decoded portions.

The critic is updated with respect to the gradient:

bringing the critic's prediction of the value of being in a state closer to its true underlying value.

We further add an entropy loss over the valid actions, designed to prevent the agent from prematurely converging on a trajectory.

Our experiments are structured into two parts: We first present a comprehensive set of ablations designed to test the relative effectiveness of the various parts of our algorithm.

The full KG-A2C is then tested on a suite of Jericho supported games and is compared to strong, established baselines.

Additionally, as encouraged by Hausknecht et al. (2019a), we present the set of handicaps used by our agents: (1) Jericho's ability to identify valid actions and (2) the Load, Save handicap in order to acquire o t desc and o tinv using the look and inventory commands without changing the game state.

Hyperparameters are provided in Appendix B.

Ablation Study.

Our ablation study is performed on Zork1, identified by Hausknecht et al. (2019a) to be one of the most difficult games in their suite and the subject of much prior work (Zahavy et al., 2018; Yin & May, 2019) .

Zork1 is one of the earliest IF games and is a dungeon-crawler-a player must explore a vast labyrinth while fighting off enemies and complete puzzles in order to collect treasures.

It features a relatively sparse reward for collecting a treasure or moving along the right path to one, and stochasticity in terms of random enemy movements.

In order to understand the contributions of different components of KG-A2C's architecture, ablate KG-A2C's knowledge graph, template-action space, and valid-action loss.

In order to understand the effects of using a knowledge graph, LSTM-A2C removes all components of KG-A2C's knowledge graph.

In particular, the state embedding vector is now computed as s t = o t ⊕ c t and the graph mask is not used to constrain action decoding.

LSTM-A2C-masked The same as LSTM-A2C but we use interactive objects to provide an object mask that is used in the same manner as the graph mask of KG-A2C.

KG-A2C-seq discards the template action space and instead decodes actions word by word up to a maximum of four words.

A supervised cross-entropy-based valid action loss L V alid is now calculated by selecting a random valid action a t valid ∈ V alid(s t ) and using each token in it as a target label.

As this action space is orders of magnitude larger than template actions, we use teacher-forcing to enable more effective exploration while training the agent-executing a t valid with a probability p valid = 0.5 and the decoded action otherwise.

In order to understand the importance of training with valid-actions, KG-A2C-unsupervised is not allowed to access the list of valid actions-the valid-action-losses L T and L O are disabled and L E now based on the full action set.

Thus, the agent must explore the template action space manually.

Template DQN Baseline.

TDQN (Hausknecht et al., 2019a ) is an extension of LSTM-DQN (Narasimhan et al., 2015) to template-based action spaces.

This is accomplished using three output heads: one for estimating the Q-Values over templates Q(s t , u)∀u ∈ T and two for estimating Q-Values Q(s t , o 1 ), Q(s t , o 2 )∀o i ∈ O over vocabulary to fill in the blanks of the template.

The final executed action is constructed by greedily sampling from the predicted Q-values.

To understand how humans progress in Zork1, a group of 10 human players-familiar with IF games-were asked to play Zork1 for the first time (with no access to walkthroughs).

Half of the players reached a game score of around 40 before dying to the first beatable NPC, a troll, mostly due to neglecting to collect a weapon to fight it with beforehand.

Three of the remaining players died to hidden traps even before reaching this point, achieving scores between 5 and 15.

The final two players made it significantly past the troll gaining scores of around 70.

A map of Zork1 with annotated rewards can be found in Appendix C and additional learning curves can be found in Appendix B.

With this in mind, we first discuss the results of the ablation study and then KG-A2C's performance over a much wider set of games found in Jericho.

On Zork1, the full KG-A2C significantly outperforms all baselines and ablations as seen in Table 3b -indicating that all components of the full algorithm previously introduced are crucial for its performance.

The first two possible rewards that can be received in this game are of magnitude 5 and 10, both requiring 4 steps to reach from the starting point when following an optimal policy.

This is the extent of the progress of both the LSTM-A2C and the KG-A2C-seq.

The LSTM-A2C more often than not collects both of these rewards while the KG-A2C-seq usually only collects one or the other in the span of an episode.

KG-A2C-seq, using a action space consisting of the full vocabulary, performs significantly worse than the rest of the agents even when given the handicaps of teacher forcing and being allowed to train for significantly longer-indicating that the template based action space is necessary for effective exploration.

The LSTM-A2C-masked progresses significantly further in the game due to the object mask cutting down the action space, allowing for more efficient exploration.

It does not, however, perform as well as KG-A2C likely due to the lack of the graph component g t in the state embedding.

LSTM-A2C and TDQN, which use the template-based action space without a knowledge graph, also struggle to progress beyond the initial rewards.

Without a knowledge graph to maintain a belief over the world state and constrain the action generation, the agent is unable to produce contextually relevant commands.

Thus both the templates and the knowledge graph are critical for the agent to attain state-of-the-art performance.

The final component being tested, the valid action supervised loss does not appear to be a important as our choice of state and action spaces: KG-A2C-unsupervised achieves nearly comparable performance to the full algorithm, also achieving state-of-the-art when compared to prior agents,

Tabula rasa reinforcement learning offers an intuitive paradigm for exploring goal driven, contextually aware natural language generation.

The sheer size of the natural language action space, however, has proven to be out of the reach of existing algorithms.

In this paper we introduced KG-A2C, a novel learning agent that demonstrates the feasibility of scaling reinforcement learning towards natural language actions spaces with hundreds of millions of actions.

The key insight to being able to efficiently explore such large spaces is the combination of a knowledge-graph-based state space and a template-based action space.

The knowledge graph serves as a means for the agent to understand its surroundings, accumulate information about the game, and disambiguate similar textual observations while the templates lend a measure of structure that enables us to exploit that same knowledge graph for language generation.

Together they constrain the vast space of possible actions into the compact space of sensible ones.

An ablation study on Zork1 shows state-of-the-art performance with respect to any currently existing general reinforcement learning agent, including those with action spaces six orders of magnitude smaller than what we consider-indicating the overall efficacy of our combined state-action space.

Further, a suite of experiments shows wide improvement over TDQN, the current state-of-the-art template based agent, across a diverse set of 26 human-made IF games covering multiple genres and game structures demonstrate that our agent is able to generalize effectively.

A IMPLEMENTATION DETAILS

Candidate interactive objects are identified by performing part-of-speech tagging on the current observation, identifying singular and proper nouns as well as adjectives, and are then filtered by checking if they can be examined using the command examine OBJ.

Only the interactive objects not found in the inventory are linked to the node corresponding to the current room and the inventory items are linked to the "you" node.

The only other rule applied uses the navigational actions performed by the agent to infer the relative positions of rooms, e.g. kitchen, down, cellar when the agent performs go down when in the kitchen to move to the cellar.

Templates are processed by selecting a single verb and preposition from the aliases.

For the sake of agent explainability, we pick the verb and preposition that are most likely to be used by humans when playing IF games.

This is done by assessing token frequencies from a dataset of human playthroughs such as those given in ClubFloyd [at/against/on/onto] ), would then be converted to take and put on .

Episodes are terminated after 100 valid steps or game over/victory.

Agents that decode invalid actions often wouldn't make it very far into the game, and so we only count valid-actions against the hundred step limit.

All agents are trained individually on each game and then evaluated on that game.

All A2C based agents are trained using data collected from 32 parallel environments.

TDQN was trained using a single environment.

Hyperparameters for all agents were tuned on the game of Zork1 and held constant across all other games.

Final reported scores are an average over 5 runs of each algorithm.

Interactive objects: tree, path, branches, forest, large, all Action: up Score: 0 ---Obs: Desc: Up a Tree You are about 10 feet above the ground nestled among some large branches.

The nearest branch above you is above your reach.

Beside you on the branch is a small birds nest.

In the birds nest is a large egg encrusted with precious jewels, apparently scavenged by a childless songbird.

The egg is covered with fine gold inlay, and ornamented in lapis lazuli and motherofpearl.

Unlike most eggs, this one is hinged and closed with a delicate looking clasp.

The egg appears extremely fragile.

Inv: You are emptyhanded.

Feedback:

Up a Tree You are about 10 feet above the ground nestled among some large branches.

The nearest branch above you is above your reach.

Beside you on the branch is a small birds nest.

In the birds nest is a large egg encrusted with precious jewels, apparently scavenged by a childless songbird.

The egg is covered with fine gold inlay, and ornamented in lapis lazuli and motherofpearl.

Unlike most eggs, this one is hinged and closed with a delicate looking clasp.

The egg appears extremely fragile.

@highlight

We present KG-A2C, a reinforcement learning agent that builds a dynamic knowledge graph while exploring and generates natural language using a template-based action space - outperforming all current agents on a wide set of text-based games.