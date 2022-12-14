Allowing humans to interactively train artificial agents to understand language instructions is desirable for both practical and scientific reasons.

Though, given  the lack of sample efficiency in current learning methods, reaching this goal may require substantial research efforts.

We introduce the BabyAI research platform, with the goal of supporting investigations towards including humans in the loop for grounded language learning.

The BabyAI platform comprises an extensible suite of 19 levels of increasing difficulty.

Each level gradually leads the agent towards acquiring a combinatorially rich synthetic language, which is a proper subset of English.

The platform also provides a hand-crafted bot agent, which simulates a human teacher.

We report estimated amount of supervision required for training neural reinforcement and behavioral-cloning agents on some BabyAI levels.

We put forward strong evidence that current deep learning methods are not yet sufficiently sample-efficient in the context of learning a language with compositional properties.

How can a human train an intelligent agent to understand natural language instructions?

We believe that this research question is important from both technological and scientific perspectives.

No matter how advanced AI technology becomes, human users will likely want to customize their intelligent helpers to better understand their desires and needs.

On the other hand, developmental psychology, cognitive science and linguistics study similar questions but applied to human children, and a synergy is possible between research in grounded language learning by computers and research in human language acquisition.

In this work, we present the BabyAI research platform, whose purpose is to facilitate research on grounded language learning.

In our platform we substitute a simulated human expert for a real human; yet our aspiration is that BabyAI-based studies enable substantial progress towards putting an actual human in the loop.

The current domain of BabyAI is a 2D gridworld in which synthetic natural-looking instructions (e.g. "put the red ball next to the box on your left") require the agent to navigate the world (including unlocking doors) and move objects to specified locations.

BabyAI improves upon similar prior setups BID16 BID7 BID41 by supporting simulation of certain essential aspects of the future human in the loop agent training: curriculum learning and interactive teaching.

The usefulness of curriculum learning for training machine learning models has been demonstrated numerous times in the literature BID6 BID23 BID42 BID15 , and we believe that gradually increasing the difficulty of the task will likely be essential for achieving efficient humanmachine teaching, as in the case of human-human teaching.

To facilitate curriculum learning studies, BabyAI currently features 19 levels in which the difficulty of the environment configuration and the complexity of the instruction language are gradually increased.

Interactive teaching, i.e. teaching differently based on what the learner can currently achieve, is another key capability of human teachers.

Many advanced agent training methods, including DAGGER BID28 , TAMER BID35 and learning from human preferences BID38 BID11 , assume that interaction between the learner and the teacher is possible.

To support interactive experiments, BabyAI provides a bot agent that can be used to generate new demonstrations on the fly and advise the learner on how to continue acting.

Arguably, the main obstacle to language learning with a human in the loop is the amount of data (and thus human-machine interactions) that would be required.

Deep learning methods that are used in the context of imitation learning or reinforcement learning paradigms have been shown to be very effective in both simulated language learning settings BID25 BID16 and applications BID32 BID3 BID39 .

These methods, however, require enormous amounts of data, either in terms of millions of reward function queries or hundreds of thousands of demonstrations.

To show how our BabyAI platform can be used for sample efficiency research, we perform several case studies.

In particular, we estimate the number of demonstrations/episodes required to solve several levels with imitation and reinforcement learning baselines.

As a first step towards improving sample efficiency, we additionally investigate to which extent pretraining and interactive imitation learning can improve sample efficiency.

The concrete contributions of this paper are two-fold.

First, we contribute the BabyAI research platform for learning to perform language instructions with a simulated human in the loop.

The platform already contains 19 levels and can easily be extended.

Second, we establish baseline results for all levels and report sample efficiency results for a number of learning approaches.

The platform and pretrained models are available online.

We hope that BabyAI will spur further research towards improving sample efficiency of grounded language learning, ultimately allowing human-in-the-loop training.

There are numerous 2D and 3D environments for studying synthetic language acquistion.

BID16 BID7 BID41 BID40 .

Inspired by these efforts, BabyAI augments them by uniquely combining three desirable features.

First, BabyAI supports world state manipulation, missing in the visually appealing 3D environments of BID16 , BID7 and BID40 .

In these environments, an agent can navigate around, but cannot alter its state by, for instance, moving objects.

Secondly, BabyAI introduces partial observability (unlike the gridworld of BID4 ).

Thirdly, BabyAI provides a systematic definition of the synthetic language.

As opposed to using instruction templates, the Baby Language we introduce defines the semantics of all utterances generated by a context-free grammar (Section 3.2).

This makes our language richer and more complete than prior work.

Most importantly, BabyAI provides a simulated human expert, which can be used to investigate human-in-the-loop training, the aspiration of this paper.

Currently, most general-purpose simulation frameworks do not feature language, such as PycoLab BID13 , MazeBase BID31 , Gazebo BID21 , VizDoom BID19 , DM-30 BID14 , and AI2-Thor BID22 .

Using a more realistic simulated environment such as a 3D rather than 2D world comes at a high computational cost.

Therefore, BabyAI uses a gridworld rather than 3D environments.

As we found that available gridworld platforms were insufficient for defining a compositional language, we built a MiniGrid environment for BabyAI.General-purpose RL testbeds such as the Arcade Learning Environment BID5 , DM-30 BID14 , and MazeBase BID31 do not assume a humanin-the-loop setting.

In order to simulate this, we have to assume that all rewards (except intrinsic (a) GoToObj: "go to the blue ball" (b)PutNextLocal: "put the blue key next to the green ball" (c) BossLevel: "pick up the grey box behind you, then go to the grey key and open a door".

Note that the green door near the bottom left needs to be unlocked with a green key, but this is not explicitly stated in the instruction.

rewards) would have to be given by a human, and are therefore rather expensive to get.

Under this assumption, imitation learning methods such as behavioral cloning, Searn BID12 , DAGGER BID28 or maximum-entropy RL BID43 are more appealing, as more learning can be achieved per human-input unit.

Similar to BabyAI, studying sample efficiency of deep learning methods was a goal of the bAbI tasks BID36 , which tested reasoning capabilities of a learning agent.

Our work differs in both of the object of the study (grounded language with a simulated human in the loop) and in the method: instead of generating a fixed-size dataset and measuring the performance, we measure how much data a general-purpose model would require to get close-to-perfect performance.

There has been much research on instruction following with natural language BID33 BID9 BID2 BID25 BID37 as well as several datasets including SAIL BID24 BID9 and Room-to-Room BID0 .

Instead of using natural language, BabyAI utilises a synthetic Baby language, in order to fully control the semantics of an instruction and easily generate as much data as needed.

Finally, BID34 presented a system that interactively learned language from a human.

We note that their system relied on substantial amounts of prior knowledge about the task, most importantly a task-specific executable formal language.

The BabyAI platform that we present in this work comprises an efficiently simulated gridworld environment (MiniGrid) and a number of instruction-following tasks that we call levels, all formulated using subsets of a synthetic language (Baby Language).

The platform also includes a bot that can generate successful demonstrations for all BabyAI levels.

All the code is available online at https://github.com/mila-iqia/babyai/tree/iclr19.

Studies of sample efficiency are very computationally expensive given that multiple runs are required for different amounts of data.

Hence, in our design of the environment, we have aimed for a minimalistic and efficient environment which still poses a considerable challenge for current general-purpose agent learning methods.

We have implemented MiniGrid, a partially observable 2D gridworld environment.

The environment is populated with entities of different colors, such as the agent, balls, boxes, doors and keys (see FIG0 .

Objects can be picked up, dropped and moved around by the agent.

Doors can be unlocked with keys matching their color.

At each step, the agent receives a 7x7 representation of its field of view (the grid cells in front of it) as well as a Baby Language instruction (textual string).The MiniGrid environment is fast and lightweight.

Throughput of over 3000 frames per second is possible on a modern multi-core laptop, which makes experimentation quicker and more accessible.

The environment is open source, available online, and supports integration with OpenAI Gym.

For more details, see Appendix A.

We have developed a synthetic Baby Language to give instructions to the agent as well as to automatically verify their execution.

Although Baby Language utterances are a comparatively small subset of English, they are combinatorially rich and unambigously understood by humans.

The language is intentionally kept simple, but still exhibits interesting combinatorial properties, and contains 2.48 ?? 10 19 possible instructions.

In this language, the agent can be instructed to go to objects, pick up objects, open doors, and put objects next to other objects.

The language also expresses the conjunction of several such tasks, for example "put a red ball next to the green box after you open the door".

The Backus-Naur Form (BNF) grammar for the language is presented in FIG1 and some example instructions drawn from this language are shown in FIG2 .

In order to keep the resulting instructions readable by humans, we have imposed some structural restrictions on this language: the and connector can only appear inside the then and after forms, and instructions can contain no more than one then or after word.

The BabyAI platform includes a verifier which checks if an agent's sequence of actions successfully achieves the goal of a given instruction within an environment.

Descriptors in the language refer to one or to multiple objects.

For instance, if an agent is instructed to "go to a red door", it can successfully execute this instruction by going to any of the red doors in the environment.

The then and after connectors can be used to sequence subgoals.

The and form implies that both subgoals must be completed, without ordering constraints.

Importantly, Baby Language instructions leave An agent may have to find a key and unlock a door, or move obstacles out of the way to complete instructions, without this being stated explicitly.

DISPLAYFORM0 DISPLAYFORM1

There is abundant evidence in prior literature which shows that a curriculum may greatly facilitate learning of complex tasks for neural architectures BID6 BID23 BID42 BID15 .

To investigate how a curriculum improves sample efficiency, we created 19 levels which require understanding only a limited subset of Baby Language within environments of varying complexity.

Formally, a level is a distribution of missions, where a mission combines an instruction within an initial environment state.

We built levels by selecting competencies necessary for each level and implementing a generator to generate missions solvable by an agent possessing only these competencies.

Each competency is informally defined by specifying what an agent should be able to do:??? Room Navigation (ROOM): navigate a 6x6 room.??? Ignoring Distracting Boxes (DISTR-BOX): navigate the environment even when there are multiple distracting grey box objects in it.??? Ignoring Distractors (DISTR): same as DISTR-BOX, but distractor objects can be boxes, keys or balls of any color.??? Maze Navigation (MAZE): navigate a 3x3 maze of 6x6 rooms, randomly inter-connected by doors.??? Unblocking the Way (UNBLOCK): navigate the environment even when it requires moving objects out of the way.??? Unlocking Doors (UNLOCK): to be able to find the key and unlock the door if the instruction requires this explicitly.??? Guessing to Unlock Doors (IMP-UNLOCK): to solve levels that require unlocking a door, even if this is not explicitly stated in the instruction.??? Go To Instructions (GOTO): understand "go to" instructions, e.g. "go to the red ball".??? Open Instructions (OPEN): understand "open" instructions, e.g. "open the door on your left".??? Pickup Instructions (PICKUP): understand "pick up" instructions, e.g. "pick up a box".??? Put Instructions (PUT): understand "put" instructions, e.g. "put a ball next to the blue key".??? Location Language (LOC): understand instructions where objects are referred to by relative location as well as their shape and color, e.g. "go to the red ball in front of you".??? Sequences of Commands (SEQ): understand composite instructions requiring an agent to execute a sequence of instruction clauses, e.g. "put red ball next to the green box after you open the door".

TAB1 lists all current BabyAI levels together with the competencies required to solve them.

These levels form a progression in terms of the competencies required to solve them, culminating with DISPLAYFORM0 the BossLevel, which requires mastering all competencies.

The definitions of competencies are informal and should be understood in the minimalistic sense, i.e. to test the ROOM competency we have built the GoToObj level where the agent needs to reach the only object in an empty room.

Note that the GoToObj level does not require the GOTO competency, as this level can be solved without any language understanding, since there is only a single object in the room.

However, solving the GoToLocal level, which instructs the agent to go to a specific object in the presence of multiple distractors, requires understanding GOTO instructions.

The bot is a key ingredient intended to perform the role of a simulated human teacher.

For any of the BabyAI levels, it can generate demonstrations or suggest actions for a given environment state.

Whereas the BabyAI learner is meant to be generic and should scale to new and more complex tasks, the bot is engineered using knowledge of the tasks.

This makes sense since the bot stands for the human in the loop, who is supposed to understand the environment, how to solve missions, and how to teach the baby learner.

The bot has direct access to a tree representation of instructions, and so does not need to parse the Baby Language.

Internally, it executes a stack machine in which instructions and subgoals are represented (more details can be found in Appendix B).

The stackbased design allows the bot to interrupt what it is currently doing to achieve a new subgoal, and then resume the original task.

For example, going to a given object will require exploring the environment to find that object.

The subgoals which the bot implements are:??? Open: Open a door that is in front of the agent.??? Close: Close a door that is in front of the agent.??? Pickup: Execute the pickup action (pick up an object).??? Drop: Execute the drop action (drop an object being carried).??? GoNextTo: Go next to an object matching a given (type, color) description or next to a cell at a given position.??? Explore: Uncover previously unseen parts of the environment.

All of the Baby Language instructions are decomposed into these internal subgoals which the bot knows how to solve.

Many of these subgoals, during their execution, can also push new subgoals on the stack.

A central part of the design of the bot is that it keeps track of the grid cells of the environment which it has and has not seen.

This is crucial to ensure that the bot can only use information which it could realistically have access to by exploring the environment.

Exploration is implemented as part of the Explore subgoal, which is recursive.

For instance, exploring the environment may require opening doors, or moving objects that are in the way.

Opening locked doors may in turn require finding a key, which may itself require exploration and moving obstructing objects.

Another key component of the bot's design is a shortest path search routine.

This is used to navigate to objects, to locate the closest door, or to navigate to the closest unexplored cell.

We assess the difficulty of BabyAI levels by training a behavioral cloning baseline for each level.

Furthermore, we estimate how much data is required to solve some of the simpler levels and study to which extent the data demands can be reduced by using basic curriculum learning and interactive teaching methods.

All the code that we use for the experiments, as well as containerized pretrained models, is available online.

The BabyAI platform provides by default a 7x7x3 symbolic observation x t (a partial and local egocentric view of the state of the environment) and a variable length instruction c as inputs at each step.

We use a basic model consisting of standard components to predict the next action a based on x and c. In particular, we use a GRU BID10 ) to encode the instruction and a convolutional network with two batch-normalized BID18 FiLM BID26 layers to jointly process the observation and the instruction.

An LSTM BID17 memory is used to integrate representations produced by the FiLM module at each step.

Our model is thus similar to the gated-attention model used by BID7 , inasmuch as gated attention is equivalent to using FiLM without biases and only at the output layer.

Previous work has shown that attention can significantly improve the performance of agents in grounded language learning experiments BID7 .

We have chosen to use FiLM over a gated attention mechanism because the FiLM mechanism seems like a more flexible alternative, which can not only rescale convolutional feature maps, but also conditionally re-normalize them.

We have used two versions of our model, to which we will refer as the Large model and the Small model.

In the Large model, the memory LSTM has 2048 units and the instruction GRU is bidirectional and has 256 units.

Furthermore, an attention mechanism BID3 is used to focus on the relevant states of the GRU.

The Small model uses a smaller memory of 128 units and encodes the instruction with a unidirectional GRU and no attention mechanism.

In all our experiments, we used the Adam optimizer BID20 with the hyperparameters ?? = 10 ???4 , ?? 1 = 0.9, ?? 2 = 0.999 and = 10 ???5 .

In our imitation learning (IL) experiments, we truncated the backpropagation through time at 20 steps for the Small model and at 80 steps for the Large model.

For our reinforcement learning experiments, we used the Proximal Policy Optimization (PPO, BID30 algorithm with parallelized data collection.

Namely, we performed 4 epochs of PPO using 64 rollouts of length 40 collected with multiple processes.

We gave a non-zero reward to the agent only when it fully completed the mission, and the magnitude of the reward was 1 ??? 0.9n/n max , where n is the length of the successful episode and n max is the maximum number of steps that we allowed for completing the episode, different for each mission.

The reward future returns were discounted without a factor ?? = 0.99.

For generalized advantage estimation BID29 in PPO we used ?? = 0.99.In all our experiments we reported the success rate, defined as the ratio of missions of the level that the agent was able to accomplish within n max steps.

Running the experiments outlined in this section required between 20 and 50 GPUs during two weeks.

At least as much computing was required for preliminary investigations.

To obtain baseline results for all BabyAI levels, we have trained the Large model (see Section 4.1) with imitation learning using one million demonstration episodes for each level.

The demonstrations were generated using the bot described in Section 3.4.

The models were trained for 80 epochs on levels with a single room and for 20 epochs on levels with a 3x3 maze of rooms.

TAB2 reports the maximum success rate on a validation set of 512 episodes.

All of the single-room levels are solved with a success rate of 100.0%.

As a general rule, levels for which demonstrations are longer tend to be more difficult to solve.

Using 1M demonstrations for levels as simple as GoToRedBall is very inefficient and hardly ever compatible with the long-term goal of enabling human teaching.

The BabyAI platform is meant to support studies of how neural agents can learn with less data.

To bootstrap such studies we have computed baseline sample efficiencies for imitation learning and reinforcement learning approaches to solving BabyAI levels.

We say an agent solves a level if it reaches a success rate of at least 99%.

We define the sample efficiency as the minimum number of demonstrations or RL episodes required to train an agent solve a given level.

To estimate the sample efficiency for imitation learning, we have tried training models with different numbers of demonstrations starting from one million and dividing each time by ??? 2.

Each model was trained for 2 ?? T L min parameter updates, where T L min is the number of parameter updates that was required for getting the target 99% performance with 1M demonstrations for the level L. For each level we find the minimum number of demonstrations k from our ??? 2 grid for which the 99% threshold was crossed in at least in 1 run out of 3.

We can then be sure that the minimum number of demonstrations lies somewhere in the k/ ??? 2; k bracket.

The results for a subset of levels are reported in TAB3 (see "IL from Bot" column).

In the same table (column "RL") we report the number of episodes that were required for reinforcement learning to solve each of these levels, and as expected, the sample efficiency of RL is substantially worse than that of IL (anywhere between 2 to 10 times in these experiments).To analyze how much the sample efficiency of IL depends on the source of demonstrations, we experimented with generating demonstrations with agents that were trained with RL for the previous experiments.

The results are reported in the "IL from RL Expert" column in TAB5 .

Interestingly, we found that the demonstrations produced by such an agent are sometimes easier for the learner to imitate (e.g. 1.4K vs 5.66K for GoToRedBallGrey).

This can be explained by the fact that the RL expert has the same neural network architecture as the learner.

Table 4 : The sample efficiency results for pretraining experiments.

For each pair of base levels and target levels that we have tried, we report how many demonstrations (in thousands) were required, as well as the baseline number of demonstrations required for training from scratch.

In both cases we report a k/ ??? 2; k range, see Section 4 for details.

Note how choosing the right base levels (e.g. GoToLocal and GoToObjMaze) is crucial for pretraining to be helpful.

Target To demonstrate how curriculum learning research can be done using the BabyAI platform, we perform a number of basic pretraining experiments.

In particular, we select 5 combinations of base levels and a target level and study if pretraining on base levels can help the agent master the target level with fewer demonstrations.

The results are reported in Table 4 .

In four cases, using GoToLocal as one of the base levels reduced the number of demonstrations required to solve the target level.

However, when only GoToObjMaze was used as the base level, we have not found pretraining to be beneficial.

We find this counter-intuitive result interesting, as it shows how current deep learning methods often can not take the full advantage of available curriculums.

Lastly, we perform an example case study of how sample efficiency can be improved by interactively providing more informative examples to the agent based on what it has already learned.

We experiment with an iterative algorithm for adaptively growing the agent's training set.

In particular, we start with 1000 base demonstrations, and at each iteration we increase the dataset size by a factor of 1.2 by providing bot demonstrations for missions on which the agent failed.

After each dataset increase we train a new agent from scratch.

We then report the size of the training set for which the agent's performance has surpassed the 99% threshold.

We repeat such an experiment 4 times for levels GoToRedBallGrey, GoToRedBall and GoToLocal and report the maximum and the minimum sample efficiency for this approach, which we call interactive imitation learning, in TAB5 .

We have observed substantial improvement on the vanilla IL in some runs (e.g. 2.98K vs 5.66K for GoToRedBallGrey), but it should be noted that the variance of interactive imitation learning results was rather high.

We present the BabyAI research platform to study language learning with a human in the loop.

The platform includes 19 levels of increasing difficulty, based on a decomposition of tasks into a set of basic competencies.

Solving the levels requires understanding the Baby Language, a subset of English with a formally defined grammar which exhibits compositional properties.

The language is minimalistic and the levels seem simple, but empirically we have found them quite challenging to solve.

The platform is open source and extensible, meaning new levels and language concepts can be integrated easily.

The results in Section 4 suggest that current imitation learning and reinforcement learning methods scale and generalize poorly when it comes to learning tasks with a compositional structure.

Hundreds of thousands of demonstrations are needed to learn tasks which seem trivial by human standards.

Methods such as curriculum learning and interactive learning can provide measurable improvements in terms of sample efficiency, but, in order for learning with an actual human in the loop to become realistic, an improvement of at least three orders of magnitude is required.

An obvious direction of future research to find strategies to improve sample efficiency of language learning.

Tackling this challenge will likely require new models and new teaching methods.

Approaches that involve an explicit notion of modularity and subroutines, such as Neural Module Networks BID1 or Neural Programmer-Interpreters BID27 , seem like a promising direction.

It is our hope that the BabyAI platform can serve as a challenge and a benchmark for the sample efficiency of language learning for years to come.

The environments used for this research are built on top of MiniGrid, which is an open source gridworld package.

This package includes a family of reinforcement learning environments compatible with the OpenAI Gym framework.

Many of these environments are parameterizable so that the difficulty of tasks can be adjusted (e.g. the size of rooms is often adjustable).

In MiniGrid, the world is a grid of size NxN.

Each tile in the grid contains exactly zero or one object, and the agent can only be on an empty tile or on a tile containing an open door.

The possible object types are wall, door, key, ball, box and goal.

Each object has an associated discrete color, which can be one of red, green, blue, purple, yellow and grey.

By default, walls are always grey and goal squares are always green.

Rewards are sparse for all MiniGrid environments.

Each environment has an associated time step limit.

The agent receives a positive reward if it succeeds in satisfying an environment's success criterion within the time step limit, otherwise zero.

The formula for calculating positive sparse rewards is 1 ??? 0.9 * (step_count/max_steps).

That is, rewards are always between zero and one, and the quicker the agent can successfully complete an episode, the closer to 1 the reward will be.

The max_steps parameter is different for each mission, and varies depending on the size of the environment (larger environments having a higher time step limit) and the length of the instruction (more time steps are allowed for longer instructions).

There are seven actions in MiniGrid: turn left, turn right, move forward, pick up an object, drop an object, toggle and done.

The agent can use the turn left and turn right action to rotate and face one of 4 possible directions (north, south, east, west).

The move forward action makes the agent move from its current tile onto the tile in the direction it is currently facing, provided there is nothing on that tile, or that the tile contains an open door.

The agent can open doors if they are right in front of it by using the toggle action.

Observations in MiniGrid are partial and egocentric.

By default, the agent sees a square of 7x7 tiles in the direction it is facing.

These include the tile the agent is standing on.

The agent cannot see through walls or closed doors.

The observations are provided as a tensor of shape 7x7x3.

However, note that these are not RGB images.

Each tile is encoded using 3 integer values: one describing the type of object contained in the cell, one describing its color, and a state indicating whether doors are open, closed or locked.

This compact encoding was chosen for space efficiency and to enable faster training.

The fully observable RGB image view of the environments shown in this paper is provided for human viewing.

The bot has access to a representation of the instructions for each environment.

These instructions are decomposed into subgoals that are added to a stack.

In FIG4 we show the stacks corresponding to the examples in FIG0 .

The stacks are illustrated in bottom to top order, that is, the lowest subgoal in the illustration is to be executed first.

Once instructions for a task are translated into the initial stack of subgoals, the bot starts by processing the first subgoal.

Each subgoal is processed independently, and can either lead to more subgoals being added to the stack, or to an action being taken.

When an action is taken, the state of the bot in the environment changes, and its visibility mask is populated with all the new observed cells and objects, if any.

The visibility mask is essential when looking for objects and paths towards cells, because it keeps track of what the bot has seen so far.

Once a subgoal is marked as completed, it is removed from the stack, and the bot starts processing the next subgoal in the stack.

Note that the same subgoal can remain on top of the stack for multiple time steps, and result in multiple actions being taken.

The Close, Drop and Pickup subgoals are trivial, that is, they result in the execution of the corresponding action and then immediately remove themselves from the stack.

Diagrams depicting how the Open, GoNextTo and Explore subgoals are handled are depicted in Figures 5, 6 , and 7 respectively.

In the diagrams, we use the term "forward cell" to refer to the grid cell that the agent is facing.

We say that a path from X to Y contains blockers if there are objects that need to be moved in order for the agent to be able to navigate from X to Y. A "clear path" is a path without blockers.

@highlight

We present the BabyAI platform for studying data efficiency of language learning with a human in the loop

@highlight

Presents a research platform with a bot in the loop for learning to execute language instructions in which language has compositional structures

@highlight

Introduces a platform for grounded language learning that replaces any human in the loop with a heuristic teacher and uses a synthetic language mapped to a 2D grid world