How can we teach artificial agents to use human language flexibly to solve problems in a real-world environment?

We have one example in nature of agents being able to solve this problem: human babies eventually learn to use human language to solve problems, and they are taught with an adult human-in-the-loop.

Unfortunately, current machine learning methods (e.g. from deep reinforcement learning) are too data inefficient to learn a language in this way (3).

An outstanding goal is finding an algorithm with a suitable ‘language learning prior’ that allows it to learn human language, while minimizing the number of required human interactions.



In this paper, we propose to learn such a prior in simulation, leveraging the increasing amount of available compute for machine learning experiments (1).

We call our approach Learning to Learn to Communicate (L2C).

Specifically, in L2C we train a meta-learning agent in simulation to interact with populations of pre-trained agents, each with their own distinct communication protocol.

Once the meta-learning agent is able to quickly adapt to each population of agents, it can be deployed in new populations unseen during training, including populations of humans.

To show the promise of the L2C framework, we conduct some preliminary experiments in a Lewis signaling game (4), where we show that agents trained with L2C are able to learn a simple form of human language (represented by a hand-coded compositional language) in fewer iterations than randomly initialized agents.

In this paper, we propose to learn such a prior in simulation, leveraging the increasing amount of available compute for machine learning experiments BID0 .

We call our approach Learning to Learn to Communicate (L2C).

Specifically, in L2C we train a meta-learning agent in simulation to interact with populations of pre-trained agents, each with their own distinct communication protocol.

Once the meta-learning agent is able to quickly adapt to each population of agents, it can be deployed in new populations, including populations speaking human language.

Our key insight is that such populations can be obtained via self-play, after pre-training agents with imitation learning on a small amount of off-policy human language data.

We call this latter technique Seeded Self-Play (S2P).

Our preliminary experiments show that agents trained with L2C and S2P need fewer on-policy samples to learn a compositional language in a Lewis signaling game (4).

Language is one of the most important aspects of human intelligence; it allows humans to coordinate and share knowledge with each other.

We will want artificial agents to understand language as it is a natural means for us to specify their goals.

So how can we train agents to understand language?

We adopt the functional view of language BID16 that has recently gained popularity (8; 14) : agents understand language when they can use language to carry out tasks in the real world.

One approach to training agents that can use language in their environment is via emergent communication, where researchers train randomly initialized agents to solve tasks requiring communication (7; 16 ).

An open question in emergent communication is how the resulting communication protocols can be transferred to learning human language.

Existing approaches attempt to do this using auxiliary tasks, for example having agents predict the label of an image in English while simultaneously playing an image-based referential game BID11 .

While this works for learning the names of objects, it's unclear if simply using an auxiliary loss will scale to learning the English names of complex concepts, or learning to use English to interact in an grounded environment.

One approach that we know will work (eventually) for training language learning agents is using a human-in-the-loop, as this is how human babies acquire language.

In other words, if we had a good enough model architecture and learning algorithm, the human-in-the-loop approach should work.

However, recent work in this direction has concluded that current algorithms are too sample inefficient to effectively learn a language with compositional properties from humans (3).

Human guidance is expensive, and thus we would want such an algorithm to be as sample efficient as possible.

An open problem is thus to create an algorithm or training procedure that results in increased sampleefficiency for language learning with a human-in-the-loop.

In this paper, we present the Learning to Learn to Communicate (L2C) framework, with the goal of training agents to quickly learn new (human) languages.

The core idea behind L2C is to leverage the increasing amount of available compute for machine learning experiments (1) to learn a 'language learning prior' by training agents via meta-learning in Figure 1 .

Diagram of the L2C framework.

An advantage of L2C is that agents can be trained in an external environment (which grounds the language), where agents interact with the environment via actions and language.

Thus, (in theory) L2C could be scaled to learn complicated grounded tasks involving language.simulation.

Specifically, we train a meta-learning agent in simulation to interact with populations of pre-trained agents, each with their own distinct communication protocol.

Once the meta-learning agent is able to quickly adapt to each population of agents, it can be deployed in new populations unseen during training, including populations of humans.

The L2C framework has two main advantages: (1) permits for agents to learn language that is grounded in an environment with which the agents can interact (i.e. it is not limited to referential games); and (2) in contrast with work from the instruction following literature (2), agents can be trained via L2C to both speak (output language to help accomplish their goal) and listen (map from the language to a goal or sequence of actions).To show the promise of the L2C framework, we provide some preliminary experiments in a Lewis signaling game BID3 .

Specifically, we show that agents trained with L2C are able to learn a simple form of human language (represented by a hand-coded compositional language) in fewer iterations than randomly initialized agents.

These preliminary results suggest that L2C is a promising framework for training agents to learn human language from few human interactions.

L2C is a training procedure that is composed of three main phases: (1) Training agent populations: Training populations of agents to solve some task (or set of tasks) in an environment.

(2) Train meta-learner on agent populations: We train a meta-learning agent to 'perform well' (i.e. achieve a high reward) on tasks in each of the training populations, after a small number of updates.

(3) Testing the meta-learner: testing the meta-learning agent's ability to learn new languages, which could be both artificial (emerged languages unseen during training) or human.

A diagram giving an overview of the L2C framework is shown in Figure 1 .

Phase 1 can be achieved in any number of ways, either through supervised learning (using approximate backpropogation) or via reinforcement learning (RL).

Phases 2 and 3 follow the typical meta-learning set-up: to conserve space, we do not replicate a formal description of the meta-learning framework, but we direct interested readers to Section 2.1 of (6).

In our case, each 'task' involves a separate population of agents with its own emergent language.

While meta-training can also be performed via supervised learning or RL, Phase 3 must be done using RL, as it involves interacting with humans which cannot be differentiated through.

See Section 6 for a discussion of each of these phases in more detail.

Seeded self-play (S2P) is a straightforward technique for training agents in simulation to develop complex behaviours.

The idea is to 'seed' a population of agents with complexWe collect some data which is sampled from a fixed seed population.

This corresponds to the actual number of samples that we care about i.e. which is basically the number of human demonstrations.

We first train each agent (a listener and a speaker) that performs well on these human samples.

We call this step as the imitation-learning step.

Then we take each of these trained agents (a pair of speaker and a listener) and deploy them against each other to solve the task via emergent communication.

We call this step as the fine-tuning step.

While these agents are exchanging messages in their emergent language, we make sure that the language does not diverge too much form the original language (i.e. the language of the fixed seed population).

We enforce this by having a schedule over the fine-tuning and the imitation-learning steps such that both the agents are able to solve the task while also keeping a perfect accuracy over the seed data.

We call this process of generating populations as seeded self-play (S2P).

A speaker-listener game We construct a referential game similar to the Task & Talk game from (11), except with a single turn.

The game is cooperative and consists of 2 agents, a speaker and a listener.

The speaker agent observes an object with a certain set of properties, and must describe the object to the listener using a sequence of words (represented by one-hot vectors).

The listener then attempts to reconstruct the object.

More specifically, the input space consists of p properties (e.g. shape, color) and t types per property (e.g. triangle, square).

The speaker observes a symbolic repre-sentation of the input x, consisting of the concatenation of p one-hot vectors, each of length t. The number of possible inputs scales as t p .

We define the vocabulary size (length of each one-hot vector sent from the speaker) as |V |, and fix the number of words sent to be w.

Developing a compositional language To simulate a simplified form of human language on this task, we programatically generate a perfectly compositional language, by assigning each 'concept' (each type of each property) a unique symbol.

In other words, to describe a blue shaded triangle, we create a language where the output description would be "blue, triangle, shaded", in some arbitrary order and without prepositions.

By 'unique symbol', we mean that no two concepts are assigned the same word.

By generating this language programmatically, we avoid the need to have humans in the loop for testing, which allows us to iterate much more quickly.

This is feasible because of the simplicity of our speaker-listener environment; we do not expect that generating these programmatic languages is practical when scaling to more complex environments.

We call these agents compositional bots (CBs).

The message produced by the speaker is a sequence of p categorical random variables which are discretized using Gumbel-Softmax with an initial temperature τ = 1.

We set the vocabulary size to be equal to the total number of concepts p ·

t. In our initial experiments, we train our agents using Cross Entropy which is summed over each property p.

We use the Adam optimizer (10) with a learning rate of 0.001.

We first demonstrate the results with a meta-learning listener (a meta-listener), that learns from the different speakers of each training population.

In Section 5 below, we perform L2C experiments varying the type and number of training populations, speaker and listener parameterizations, and listener meta-learning algorithms.

Unless otherwise specified, we use an overparameterized linear policy (i.e. an MLP with linear activations and 500 'hidden units') for the speaker and listener, and the meta-listener is trained with a first-order version of MAML (6) on 50 purely compositional training populations (CBs).

Here, we describe our initial results into the factors affecting the performance of L2C.

Since our ultimate goal is to transfer to learning a human language in as few human interactions as possible, we measure success based on the number of samples required for the meta-learner to reach a certain performance level (95%) on a held-out test population, and we permit ourselves as much computation during pre-training as necessary.

Varying meta-learning algorithms We experimented with several algorithms to train our meta-listener: a randomly initialized agent, an agent pre-trained on all populations that performs n = 1 update per population, the Reptile algorithm (15) that performs n > 1 updates per population, and a first-order variant of MAML (FOMAML) BID5 .

In our initial experiments, we found that the Reptile and the pre-training agent didn't improve significantly over the random initialization baseline.

However, when we have enough training populations, the FOMAML algorithm improved significantly over the randomly initialized baseline, and even approaches the minimum number of examples a human would need to solve the task (60, one for each word in the vocabulary) -see FIG0 .Varying listener parameterizations We tried various models for the meta-listener, including an LSTM, an MLP, and a linear model.

While L2C with FOMAML worked in all of these cases, we found the best results with an over-parameterized linear model (a 1-layer MLP with linear activations).

Strangely, the performance improved even further when adding a We suspect the over-parameterization helped (over a regular linear model) due to improved gradient descent dynamics.

Varying the number of training populations As can be inferred from FIG0 , having more training populations improves performance.

Having too few training populations (eg: 5 train encoders) results in overfitting to the set of training populations and as the meta-learning progresses, the model performs worse on the test populations.

For more than 10 training encoders, models trained with L2C require fewer samples to generalize to a held-out test population than a model not trained with L2C.

We wanted to see if we can further reduce the number of samples required after L2C.

So instead of doing L2C on a population of compositional bots, we train the population of agents using Seeded self-play (S2P).

We collect some seed data from a single compositional bot which we call as seed dataset.

Now we partition this data into train and test sets where the train set is used to train the agents via S2P.

This set of trained populations is now used as the set of populations for meta-training (L2C).We were able to get the number of samples from 60 in the best performing vanilla L2C case to 20 in the L2C with S2P case showing that the seeded self-play indeed helps in improving sample efficiency.

There are several immediate directions for future work: training the meta-agent via RL rather than supervised learning, and training the meta-agent as a joint speaker-listener (i.e. taking turns speaking and listening), as opposed to only listening.

We also want to scale L2C training to more complicated tasks involving grounded language learning, such as the Talk the Walk dataset (5), which involves two agents learning to navigate New York using language.

More broadly, there are still many challenges that remain for the L2C framework.

In fact, there are unique problems that face each of the phases described in Section 2.

In Phase 1, how do we know we can train agents to solve the tasks we want?

Recent work has shown that learning emergent communication protocols is very difficult, even for simple tasks BID12 .

This is particularly true in the multiagent reinforcement learning (RL) setting, where deciding on a communication protocol is difficult due to the nonstationary and high variance of the gradients BID12 .

This could be addressed in at least two ways: (1) by assuming the environment is differentiable, and backpropagating gradients using a stochastic differentiation procedure (9; 14), or (2) by 'seeding' each population with a small number of human demonstrations.

Point (1) is feasible because we are training in simulation, and we have control over how we build that simulation -in short, it doesn't really matter how we get our trained agent populations, so long as they are useful for the meta-learner in Phase 2.In Phase 2, the most pertinent question is: how can we be sure that a small number of updates is sufficient for a meta-agent to learn a language it has never seen before?The short answer is that it doesn't need to completely learn the language in only a few updates; rather it just needs to perform better on the language-task in the host population after a few updates, in order to provide a useful training signal to the meta-learner.

For instance, it has been shown that the model agnostic meta-learning (MAML) algorithm can perform well when multiple gradient steps are taken at test time, even if it is only trained with a single inner gradient step.

Another way to improve the meta-learner performance is to provide a dataset of agent interactions for each population.

In other words, rather than needing to metalearner perform well after interacting with a population a few times, we'd like it to perform well after doing some supervised learning on this dataset of language, and after a few interactions.

After all, we do have lots of available datasets of human language, and not using this seems like a waste.

Finally, in Phase 3, how do we know that the meta-learner will be able to generalize to learn a human language?

This comes down to the similarity between the training populations and human language, and the diversity of the training populations.

For instance, we expect certain properties like compositionality to be important in the training populations for transferring to human languages, which are inherently compositional.

But the training languages may not need to be very close to human language; as a comparison, significant progress has been made in transferring robots from simulation the real world using domain randomization, i.e. by adding random image textures during training.

Even though these image textures look nothing like the real world, it helps improve robustness of the learner, which allows it to generalize.

Providing a detailed examination of the required properties of the trained population languages, such that a meta-learner is able to generalize to human language, is an important direction for future work.

<|TLDR|>

@highlight

We propose to use meta-learning for more efficient language learning, via a kind of 'domain randomization'. 