Referential games offer a grounded learning environment for neural agents which accounts for the fact that language is functionally used to communicate.

However, they do not take into account a second constraint considered to be fundamental for the shape of human language: that it must be learnable by new language learners and thus has to overcome a transmission bottleneck.

In this work, we insert such a bottleneck in a referential game, by introducing a changing population of agents in which new agents learn by playing with more experienced agents.

We show that mere cultural transmission results in a substantial improvement in language efficiency and communicative success, measured in convergence speed, degree of structure in the emerged languages and within-population consistency of the language.

However, as our core contribution, we show that the optimal situation is to co-evolve language and agents.

When we allow the agent population to evolve through genotypical evolution, we achieve across the board improvements on all considered metrics.

These results stress that for language emergence studies cultural evolution is important, but also the suitability of the architecture itself should be considered.

Human languages show a remarkable degree of structure and complexity, and how such a complex system can have emerged is still an open question.

One concept frequently named in the context of language evolution is cultural evolution.

Unlike animal languages, which are taken to be mostly innate, human languages must be re-acquired by each individual BID29 BID10 .

This pressures them to fit two constraints that govern their cross-generational transmission: They must be learnable by new language users, and they must allow effective communication between proficient language users (see, e.g. BID31 .In the recent past, computational studies of language emergence using referential games (see Section 2.1 for a review) has received a new wave of attention.

These studies are motivated by the second constraint, that language is used to communicate.

The first constraint, on the other hand, is in this framework not considered: language is not transmitted from agent to agent and there is thus no need for agents to develop languages that would survive a transmission bottleneck.

1 In this work, we introduce a transmission bottleneck in a population of agents playing referential games, implicitly modelling cultural evolution.

However, merely adding a transmission bottleneck is not enough.

Since the types of language that may emerge through passing this bottleneck are not just dependent on the existence of a bottleneck, but also on the shape of the bottleneck, which is determined by the biases of the architecture of the agents playing the game (their genotypical design).

If the genotypical design of those agents is not suitable to solve this task through communication, they will -at best -converge to a language that doesn't allow for effective communication or is difficult to learn for every new agent or -at worst -not converge to an appropriate culturally transmittable language at all.

In this work, we therefore study the co-evolution of language and architecture in a referential games.

To this end, we introduce the Language Transmission Engine that allows to model both cultural and genetic evolution in a population of agents.

We demonstrate that the emerging languages ben-efit from including cultural transmission as well as genetic evolution, but the best results are achieved when both types of evolution are included and languages and agents can co-evolve.2 Related Work

Much work has been done on the emergence of language in artificial agents and investigating its subsequent structure, compositionality and morphosyntax BID15 BID17 .

The original computer simulations dealt with logic and symbolic representations BID15 BID3 , but with the advent of modern deep learning methods and sequence-to-sequence models BID33 , there has been a renewed interest in simulating the emergence of language through neural network agents (i.a.

BID21 BID8 .

In the exploration of language emergence, different training approaches and tasks have been proposed to encourage agents to learn and develop communication.

These tasks are commonly set up in an end-to-end setting where reinforcement learning can be applied.

This is often a two-player referential game where one agent must communicate the information it has access to (typically an image), while the other must guess it out of a lineup BID6 BID21 .

BID25 and BID2 find that structure and compositionalility can arise in emerged languages in such setups; BID19 show that 'natural' language does not arise naturally and has to be incentivised by imposing specific restrictions on games and agents.

The evolution of human language is a well-studied but still poorly understood topic.

One particular open question concerns the relation between two different evolutionary processes: genetic evolution of the agents in the population and cultural evolution of the language itself BID7 .

BID3 assert that the question of genetic versus cultural evolution ultimately arises from three distinct but interacting adaptive systems: individual learning, cultural transmission, and genetic evolution.

Cultural transmission is thought to enforce structure and compression to languages, since a language must be used and learned by all individuals of the culture in which it resides and at the same time be suitable for a variety of tasks.

BID18 define those two pressures as compressibility and expressivity and find that structure arises from the trade-off between these pressures in generated languages.

The importance of cultural evolution for the emergence of structure is supported by a number of artificial language learning studies (e.g. BID30 and computational studies using the Iterated Learning paradigm, in which agents learn a language by observing the output produced by another agent from the previous 'generation' (e.g. BID13 BID16 BID18 .

An alternative way of imposing cultural pressures on agents, is by simulating a large population of them and pairing agents randomly to solve a communicative game BID4 .

This approach is more naturally aligned with cultural pressures in humans (see e.g. BID34 and is the one we use in this paper.

While there is much controversy about the selection pressures under which the fundamental traits underlying the human ability to learn and use language evolved in other humans, that genetic evolution played an essential role in endowing humans with the capabilities to learn and use language is generally undebated.

Pre-modern humans, for instance, did not have the ability to speak or understand complex structures BID7 .There are several approaches to simulate genetic evolution of neural network agents.

Neural Architectural Search (NAS) focuses on searching the architecture space of the networks, unlike many traditional evolutionary techniques which often include parameter weights in their search space.

Some of the earlier techniques such as NEAT gained considerable traction as a sound way of doing topology search using biologically inspired concepts BID32 .

NAS methods however have mostly reverted to optimising solely the neural architecture and using gradient based methods such as SGD for weight optimisation due to the large parameter space of modern architectures (see, e.g., BID5 for a survey).More recently, state-of-the-art one-shot search techniques such as ENAS (Efficient Neural Architecture Search) and DARTS (Differentiable Architecture Search) have allowed to bring a gradientbased approach to NAS through the use of intelligent weight-sharing schemes BID24 Pham et al., 2018) .In this work, we use the DARTS search space, which is constrained but still obtained state-of-the-art performance on benchmark natural language tasks BID23 .

We study language emergence in a referential game inspired by the signalling games proposed by BID22 .

In this game, one agent (called the sender) observes an image and generates a discrete message.

The other agent, the receiver of the message, uses the message to select the right image from a set of images containing both the sender image and several distractor images.

Since the information shown to the sender agent is crucial to the receivers success, this setup urges the two agents to come up with a communication protocol that conveys the right information.

Formally, our referential game is similar to BID8 :We use z = 512, and n = 3 and train agents with Gumbel-Softmax BID11 ) based on task-success.

We introduce both cultural and genetic evolution to this game through a process that we call the Language Transmission Engine (LTE), which is depicted in FIG0 .

2 Similar to Cogswell et al.(2019), we create a population of communicating agents.

In every training iteration, two random agents are sampled to play the game.

This forces the agents to adopt a simpler language naturally: to succeed they must be able to communicate or understand all opposing agents.

In our setup, agents are either sender or receiver, they do not switch roles during their lifetime.

To model cultural evolution in the LTE, we periodically replace agents in the population with newly initialised agents.

Cultural evolution is implicitly modelled in this setup, as new agents have to learn to communicate with agents that already master the task.

Following BID4 , we experiment with three different methods to select the agents that are replaced: randomly (no selection pressure), replacing the oldest agents or replacing the agents with the lowest fitness (as defined in Section 3.3).

We call these setups cu-random, cu-age and cu-best, respectively.

To model genetic evolution, rather than periodically replacing agents with randomly initialised new agents, we instead mutate the most successful agents and replace the worst agents with variations of the best agents, as outlined in Section 3.2.2.

Note that cultural evolution is still implicitly modelled in this setup, as new agents still have to learn to communicate with older agents.

Therefore, we call this setup with the term co-evolution.

Culling We refer to the selection process and subsequent mutation or re-initialisation step as culling.

In biology, culling is the process of artificially removing organisms from a group to promote certain characteristics, so, in this case, culling consists of removing a subset of the worst agents and replacing them with variations of the best architecture.

The proportion of agents from each population selected to be mutated is determined by the culling rate ???, where ??? 2 [0, 1).

The culling interval l defines the number of iterations between culling steps.

A formalisation of the LTE can be found in appendix A.1.

We base potential mutations on the RNN cell search space DARTS, defined by BID24 .

This space includes recurrent cells with up to N nodes, where each node n 1 , n 2 , ..., n N can take the output of any preceding nodes including n 0 , which represents the cell's input.

All potential connections are modulated by an activation function, which can be the identity function, Tanh, Sigmoid or ReLU.

Following BID24 and Pham et al. (2018) , we enhance each operation with a highway bypass BID35 and the average of all intermediate nodes is treated as the cell output.

To sample the initial model, we sample a random cell with a single node (N = 1).

As this node must necessarily be connected to the input, the only variation stems from the possible activation functions applied to the output of n 1 , resulting in four possible starting configurations.

We set a node cap of N = 8.

We mutate cells by randomly sampling an architecture which is one edit step away from the previous architecture.

Edit steps are uniformly sampled from i) changing an incoming connection, ii) changing an output operation or iii) adding a new node; the mutation location is uniformly sample from all possible mutations.

3 Note that while we use the DARTS search space to define potential mutations, contrary to BID24 , we do not use differentation to sample new architectures based on a selection criterion.

The fitness criterion that we use in both the cu-best and co-evolution setup is based on task performance.

However, rather than considering agents' performance right before the culling step, we consider the age of the youngest agent in the population (defined in terms of number of batches that it was trained) and for every agent compute their performance up until when they had DISPLAYFORM0 where T A = min a2A T (a) is the age T (a) of the youngest agent in the population, and L(a t j ) is the loss of agent a j at time step t. This fitness criterion is not biased towards older agents, that have seem already more data and have simply converged more.

It is thus not only considering task performance but also the speed at which this performance is reached.

We test the LTE framework on a compositionally defined image dataset, using a range of different selection mechanisms.

In all our experiments, we use a modified version of the Shapes dataset BID0 , which consists of 30 by 30 pixel images of 2D objects, characterised by shape (circle, square, triangle), colour (red, green, blue), and size (small, big).

While every image has a unique symbolic description -consisting of the shape, colour and size of the object and its horizontal and vertical position in a 3x3 grid -one symbolic representation maps to multiple images, that differ in terms of exact pixels and object location.

We use 80k, 8k, 40k images for train, validation and test sets, respectively.

Some example images are depicted in FIG1 .We pre-train a CNN feature extractor for the images in a two-agent setting of the task (see Appendix A.4 for more details).

For our co-evolution experiments, we use the DARTS search space as described above.

For all cultural evolution approaches, we use an LSTM BID9 for both the sender and receiver architectures (see Appendix A.3 for more details).

Unless otherwise specified, we use the same sizes and hyper-parameters for all models.

The sender and receiver models have a hidden size of 64 for the recurrent layer and an embedding layer of size 64.

Further, we use a vocabulary size V of 4, with an additional bound token serving as the indicator for beginning and end- of-sequence.

We limit the maximum length of a sentence L to 5.We back-propagate gradients through the discrete step outputs (message) of the sender by using the Straight-Through (ST) Gumbel-Softmax Estimator BID12 .

We run all experiments with a fixed temperature ??? = 1.2.

We use the default Pytorch BID26 Adam (Kingma and Ba, 2015) optimiser with a learning rate of 0.001 and a batch-size of 1024.

Note that the optimiser is reset for every batch.

For all multi-agent experiments we use a population size of 16 senders and 16 receivers.

The culling rate ??? is set to 0.25 or four agents, and we cull (re-initialise or mutate) every l = 5k iterations.

We run the experiments for a total of I = 500k iterations, and evaluate the populations before each culling step.

We use an range of metrics to evaluate both the population of agents and the emerging languages.

Jaccard Similarity We measure the consistency of the emerged languages throughout the population using Jaccard Similarity, which is defined as the ratio between the size of the intersection and the union of two sets.

We sample 200 messages per input image for each possible sender-receiver pair and average the Jaccard Similarity of the samples over the population.

A high Jaccard Similarity between two messages is an indication that the same tokens are used in both messages.

We compute how similar the messages that different agents emit for the same inputs by looking at all possible (sender, message) pairs for one input and assess whether they are the same.

This metric is 1 when all agents always emit the same messages for the same inputs.

We compute the average number of unique messages generated by each sender in the population.

An intuitive reference point for this metric is the number of images with distinct symbolic representations.

If agents generate more messages than expected by this reference point, this demonstrates that they use multiple messages for the images that are -from a task perspective -identical.

A smaller number of unique messages, on the other hand, indicates that the agent is using a simpler language which is underspecified compared to the symbolic description of the image.

Topographic Similarity Topographic similarity, used in a similar context by , represents the similarity between the meaning space (defined by the symbolic representations) and the signal space (the messages send by an agent).

It is defined as the correlation between the distances between pairs in meaning space and the distances between the corresponding messages in the signal space.

We compute the topographic similarity for an agent by sampling 5,000 pairs of symbolic inputs and corresponding messages and compute the Pearson's ??? correlation between the cosine similarity of the one-hot encoded symbolic input pairs and the cosine similarity of the one-hot encoded message pairs.

Average Population Convergence To estimate the speed of learning of the agents in the population, estimate the average population convergence.

For each agent, at each point in time, this is defined as the agents average performance from the time it was born until it had the age of the current youngest agent in the population (analogous to the fitness criterion defined in Section 3.3).

To get the average population convergence, we average those values for all agents in the population.

Average Agent Entropy We compute the average certainty of sender agents in their generation process by computing and averaging their entropy during generation.

We now present a detailed comparison of our cultural and co-evolution setups.

For each approach, we averaged over four random seeds, the error bars in all plots represent the standard deviation across these four runs.

To analyse the evolution of both agents and languages, we consider the development of all previously outlined metrics over time.

We then test the best converged languages and architectures in a single sender-receiver setup, to assess the impact of cultural and genetic evolution more independently.

In these experiments, we compare also directly to a single sender-receiver baseline, which is impossible for most of the metrics we consider in this paper.

Finally, we briefly consider the emerged architectures from a qualitative perspective.

We first confirm that all setups in fact converge to a solution to the task.

As can be seen in FIG2 , all populations converge to a (close to perfect) solution to the game.

The cu-age approach slightly outperforms the other approaches, with a accuracy that surpasses the 95% accuracy mark.

Note that, due to the ever changing population, the accuracy at any point in time is an average of both 'children' and 'adults', that communicate with different members of the population.

To assess the behaviour of the agents over time, we monitor their average message entropy convergence speed.

As can be seen in FIG3 , the co-evolution setup results in the lowest average entropy scores, the messages that they assign to one particular image will thus have lower variation than in the other setups.

Of the cultural evolution setups, the lowest entropy score is achieved in the cu-best setup.

FIG4 shows the average population convergence over time.

Also in this case, we observe a clear difference between cultural evolution only and co-evolution, with an immediately much lower convergence time for co-evolution and a slightly downward trending curve.

To check the consistencies of languages within a population, we compare the Jaccard Similarity and the Average Proportion of Unique Matches, which we plot in Figure 6 .

This shows that, compared to cultural evolution only, not only are the messages in co-evolution more similar across agents (higher Jaccard Similarity), but also that agents are considerably more aligned with respect to the same inputs (less unique matches).To assess the level of structure of the emerged languages, we plot the average Topographic Similarity and the Average Number of Unique Messages generated by all senders (Figure 7) .

The co-evolution condition again outperforms all cultural only conditions, with a simpler language (the number of the unique messages closer to the symbolic reference point) that is structurally more sim- Figure 6 : Average Jaccard Similarity and proportion of message matches for all cultural transmission modes and evolution ilar to the symbolic representation of the input (higher Topographical Similarity).

In Figure 8 we show the co-evolution of an agent and a sample of its language during three selected iterations in the co-evolution setup.

Strikingly, the best sender architecture does not evolve from its original form, which could point towards the limitations of of our search strategy and space.

On the contrary, the receiver goes through quite some evolution steps and converges into a significantly more complex architecture than its original form.

We observe a unification of language throughout evolution in Figure 8 , which is also supported by Figure 7 .

The population of senders starts out 11 different unique messages and ends with only two to describe the same input image.

We will leave more detailed analysis of the evolved architectures for future work.

With a series of experiments we test the a priori suitability of the evolved languages and agents for the task at hand, by monitoring the accuracy of new agents that are paired with converged agents and train them from scratch.

We focus, in particular, on training receivers with a frozen sender from different setups, which allows us to assess 1) whether cultural evolution made languages evolve to be more easily picked up by new agents 2) whether the genetic evolution made architectures converge more quickly when faced with this task.

We compare the accuracy development of: Figure 7 : Average Number of Unique Messages and Topographic Similarity for all cultural evolution modes and co-evolution.

For comparison, we also plot the number of unique messages for a symbolic solution that fully encodes all relevant features of the image (since we have three possible shapes and colours, two possible sizes, and a 3 ??? 3 grid of possible positions, this symbolic reference solution has 3 ??? 3 ??? 2 ??? 9 = 162 distinct messages.??? An LSTM receiver trained with a frozen sender taken from cu-best;??? An evolved receiver trained with a frozen evolved sender.

For both these experiments, we compare with two baselines:??? The performance of a receiver agent trained from scratch along with a receiver agent that has either the cu architecture or the evolved co architecture (cu-baseline and co-baseline, respectively);??? The performance of an agent trained with an agent that is pretrained in the single agent setup, with either the cu architecture or an evolved architecture (cu-baseline-pretrained and co-baseline-pretrained).Each experiment is run 10 times, keeping the same frozen agent.

The results confirm cultural evolution contributes to the learnability and suitability of emerging languages: the cu-best accuracy (green line) converges substantially quicker and is substantially higher than the cu-baseline-pretrained accuracy (orange line).

Selective pressure on the Figure 8 : Evolution of the best sender and receiver architecture according to convergence, and the evolution of the population's message description of the same input through iterations.

The bold messages represent the message outputted by the best sender whose architecture is pictured above.

The count of each message represents the number of agents in the population which uttered this exact sequence.

language appears to be important: the resulting languages are only easier to learn in the cu-best setup.

4 In addition, they show that the agents benefit also from the genetic evolution: the best accuracies are achieved in the co-evolution setup (red line).

The difference between the cu-baseline (blue) and the co-baseline (brown) further shows that even if the evolved architectures are trained from scratch, they perform much better than a baseline model trained from scratch.

The difference between the co-baseline-pretrained (only genetic evolution, purple line) and the co-evolution of agents and language line (red line) illustrates that genetic evolution alone is not enough: while a new evolved receiver certainly benefits from learning from a (from scratch) pretrained evolved sender, without the cultural transmission pressure, it's performance is still substantially below a receiver that learns from an evolved sender whose language was evolved as well.

In this paper, we introduced a language transmission bottleneck in a referential game, where new agents have to learn the language by playing with more experienced agents.

To overcome such bottleneck, we enabled both the cultural evolution of language and the genetic evolution of agents, using a new Language Transmission Engine.

Us- ing a battery of metrics, we monitored their respective impact on communication efficiency, degree of linguistic structure and intra-population language homogeneity.

While we could find important differences in between cultural evolution strategies, it is when we included genetic evolution that agents scored best.

In a second experiment, we paired new agents with evolved languages and agents and again confirmed that, while cultural evolution makes a language easier to learn, coevolution leads to the best communication.

In future research, we would like to apply the Language Transmission Engine on new, more complex tasks and further increase our understanding of the properties of the emerged languages and architectures.

Additionally, we would like to investigate other neuro-evolution techniques and apply them on different search spaces.

<|TLDR|>

@highlight

We enable both the cultural evolution of language and the genetic evolution of agents in a referential game, using a new Language Transmission Engine.