Prefrontal cortex (PFC) is a part of the brain which is responsible for behavior repertoire.

Inspired by PFC functionality and connectivity,  as well as human behavior formation process, we propose a novel modular architecture of neural networks with a Behavioral Module (BM) and corresponding end-to-end training strategy.

This approach allows the efficient learning of behaviors and preferences representation.

This property is particularly useful for user modeling (as for dialog agents) and recommendation tasks, as allows learning personalized representations of different user states.

In the experiment with video games playing, the resultsshow that the proposed method allows separation of main task’s objectives andbehaviors between different BMs.

The experiments also show network extendability through independent learning of new behavior patterns.

Moreover, we demonstrate a strategy for an efficient transfer of newly learned BMs to unseen tasks.

Humans are highly intelligent species and are capable of solving a large variety of compound and open-ended tasks.

The performance on those tasks often varies depending on a number of factors.

In this work, we group them into two main categories: Strategy and Behaviour.

The first group contains all the factors leading to the achievement of a defined set of goals.

On the other hand, Behaviour is responsible for all the factors not directly linked to the goals and having no significant effect on them.

Examples of such factors can be current sentiment status or the unique personality and preferences that affect the way an individual makes decisions.

Existing Deep Networks have been focused on learning of a Strategy component.

This was achieved by optimization of a model for defined sets of goals, also the goal might be decomposed into sub-goals first, as in FeUdal Networks BID29 or Policy Sketches approach BID1 .

Behavior component, in turn, obtained much less attention from the DL community.

Although some works have been conducted on the identification of Behavior component in the input, such as works in emotion recognition BID15 BID11 BID17 .

To the best of our knowledge, there was no previous research on incorporation on Behavior Component or Behavior Representation in Deep Networks before.

Modeling Behaviour along with Strategy component is an important step to mimicking a real human behavior and creation of robust Human-Computer Interaction systems, such as a dialog agent, social robot or recommendation system.

The early work of artificial neural networks was inspired by brain structure BID9 BID16 , and the convolution operation and hierarchical layer design found in the network designed for visual analytic are inspired by visual cortex BID9 BID16 .

In this work, we again seek inspiration from the human brain architecture.

In the neuroscience studies, the prefrontal cortex (PFC) is the region of the brain responsible for the behavioral repertoire of animals BID18 ).

Similar to the connectivity of the brain cortex (as shown in Figure 1 ), we hypothesize that a behavior can be modeled as a standalone module within the deep network architecture.

Thus, in this work, we introduce a general purpose modular architecture of deep networks with a Behavioural Module (BM) focusing on impersonating the functionality of PFC.Apart from mimicking the PFC connectivity in our model, we also borrow the model training strategy from human behavior formation process.

As we are trying to mimic the functionality of a human brain we approached the problem from the perspective of Reinforcement Learning.

This approach also aligns with the process of unique personality development.

According to BID6 and BID5 unique personality can be explained by different dopamine functions caused by genetic influence.

These differences are also a reason for different Positive Emotionality (PE) Sensory Cortex (Conv Layers) PFC (Behavior Module) Motor Cortex (FC layers) Figure 1: Abstract illustration of the prefrontal cortex (PFC) connections of the brain BID18 and corresponding parts of the proposed model.

patterns (sensitivity to reward stimuli), which are in turn a significant factor in behavior formation process BID5 .

Inspired by named biological processes we introduce extra positive rewards (referring to positive-stimuli or dopamine release, higher the reward referring to higher sensitivity) to encourage specific actions and provoke the development of specific behavioral patterns in the trained agent.

To validate our method, we selected the challenging domain of classic Atari 2600 games BID2 , where the simulated environment allows an AI algorithm to learn game playing by repeatedly seek to understand the input space, objectives and solution.

Based on this environment and an established agent (i.e. Deep Q-Network (DQN) BID20 ), the behavior of the agent can be represented by preferences over different sets of actions.

In other words, in the given setting, each behaviour is represented by a probability distribution over given action space.

In real-world tasks, the extra-reward can be represented by the human satisfaction by taken action along with the correctness of the output (main reward).Importantly, the effect of human behavior is not restricted to a single task and can be observed in various similar situations.

Although it is difficult to correlate the effect of human behavior on completely different tasks, it is often easier to observe akin patterns in similar domains and problems.

To verify this, we study two BM transfer strategies to transfer a set of newly learned BMs across different tasks.

As a human PFC is responsible for behavior patterns in a variety of tasks, we also aim to achieve a zero-shot transfer of learned modules across different tasks.

The contributions of our work are as follow:• We propose a novel modular architecture with behavior module and a learning method for the separation of behavior from a strategy component.• We provide a 0-shot transfer strategy for newly learned behaviors to previously unseen tasks.

The proposed approach ensures easy extendability of the model to new behaviors and transferability of learned BMs.• We demonstrate the effectiveness of our approach on video games domain.

The experimental results show good separation of behavior with different BMs, as well as promising results when transfer learned BMs to new tasks.

Along with that, we study the effects of different hyper-parameters on the behavior separation process.

Task separation is an important yet relatively unexplored topic in deep learning.

BID22 BID22 explored this idea by simulating a simplified primate visual cortex by separation a network into two parts, responsible for shape classification task and shape localization on a binary image, respectively The topic was further studied in BID12 BID13 b) , however, due to the limitations in computational resources at that time, it has not gotten much advancement.

Recently, number researchers have revisited the idea of task separation and modular networks with evolutionary algorithms.

So in BID28 BID28 and BID21 applied neuroevolution algorithms to evolve predefined modules responsible for the problem subtasks, where improved performance was reported when compared against monolithic architectures.

BID23 BID24 proposed a neuroevolution approach to develop a multi-modular network capable of learning different agent behaviors.

The module structure and the number of modules in the network were evolved in the training process.

Although the multi-module architecture achieved better performance, it has not achieved separation of the tasks among the modules.

A number of evolved modules appeared redundant and not used in the test phase, while others have used shared neurons.

Moreover, the architecture was fixed once learned and did not assume changes in the structure.

The same approach with modifications in mutation strategy' BID25 , genome encoding BID27 and task complexity BID25 , but has not achieved significant performance.

In 2016, BID4 proposed to use a coevolutionary algorithm for domain transfer problem to avoid training from the scratch.

It first independently learns a pool of networks on different Atari2600 games, During the transfer phase, the networks were frozen and used as a 'building blocks' in a new network while combined with newly evolved neurons.

In 2017, BID8 introduced PathNet to address the task-transfer module on the example of Atari2600 games.

PathNet has a fixed size architecture (L layers by N modules), where each module was represented by either convolutional or fully-connected block.

During the training phase, authors applied the tournament genetic algorithm to learn active paths between modules along with the weights.

Once the task was learned, active modules and paths were frozen and the new task could start learning a new path.

Recently proposed FeUdal Networks architecture BID29 , also proposed a Modular design for Reinforcement Learning problems with sub-goals.

In this work authors use Manager and Worker modules for learning abstract goals and primitive actions respectively.

FeUdal networks are designed to tackle environments with long-term credit assignment and sparse reward signals.

The modules in the named architecture are not transferable and designed to learn different time-span goal embeddings.

BID0 proposed the Neural Module Network for Visual Question Answering (VQA) task.

It consists of separate modules responsible for different tasks (e.g. Find, Transform, Combine, Describe and Measure modules), which could be combined in different ways depending on the network input.

A similar dynamic architecture was proposed and applied to robot manipulator task BID7 .

The model was end-to-end trained and consisted of two modules (i.e. robotspecific and task-specific) and achieved good performance on a zero-shot learning task.

The Modular Neural Network was also applied in Reinforcement Learning task in a robotics environment BID1 .

In this work, each module was responsible for a separate sub-task of the main task.

However, the modules could be combined only in a sequential manner.

Most of the previous works focused on multi-task problems or problems with sub-goals where the modules were responsible for learning explicit sub-functionality directly affecting the model performance.

Our approach is different in a sense, we learn a behavior module responsible for representation of user sentiment states or preferences not affecting the main goals.

This approach leads to high adaptability of the network performance to new preferences or states of an enduser without retraining of the whole network, expandability of the network to future variations, removability of BMs in case of unknown preferences, as well as high-potential to transfer of the learned representations to unseen tasks.

To the best of our knowledge, there are no similar approaches.

The goal of our modular network is to introduce Behavior component into Deep Networks, ensure separation of the behavior and main task functionalities into different components of the network and provide a strategy for an efficient transfer of learned behaviors.

Our model has three main parts (1) The Main Network is responsible for the main task (strategy component) functionality, (2) a replaceable/removable Behavior Module (BM) encodes the agent behavior and separate it from the main network, and (3) the Discriminator is used during the transfer stage and helps to learn similar feature representations among different tasks.

An overview of the proposed network architecture is shown in FIG0 .

In the given architecture Convolutional layers correspond to (Visual) Sensory cortex, Fully-Connected layers of the Main Network to the Motor Cortex and Behaviour Module to PFC from Figure 1.

In this work, we adopt the deep Q-Network (DQN) with target network and memory replay BID20 to solve the main task (denoted as main network).

DQN has reported good performance The DQN has a fairly simple network structure, which consists of 3 consecutive convolutional layers followed by two fully-connected (fc) layers (see FIG0 .

All the layers, except the last one, use ReLU activation functions.

The network output is represented by the set of expected future discounted rewards for each of the available actions.

The output obeys Bellman-equation and Root-Mean-Square Error is applied as a loss function (L m ) during the training phase.

In this work, we are interested in the separation of the behavioral component from the main task functionality.

Specifically, we design a network where the behavior is modeled with a replaceable and removable module, which is denoted as Behavioral Module (BM).

The BM is supposed to have no significant effect on the performance on the main task.

The BM is modeled as two fc layers with the first layer having ReLU activation function and the second layer having linear activation.

The proposed BM input is the output from the last convolutional layer of the main network, while its output is directly fed to the first fc layer of the main network.

This architecture follows the PFC connectivity pathways described in Figure 1 .

The forward pass of the network is represented by the following equations: DISPLAYFORM0 where I is the network input, j is the index of the current behaviour, l i is the output of the i-th layer, l bi is the output of i-th layer of a BM, f i is the activation function at the i-th layer, and denotes 2d convolution operation.

The Main Network contains layers from l 1 to l 5 .

Note that l b becomes zero vector if no behavior is required or enforced.

The summation operator at the layer 4 ensures the influence of BM can be easily removed from the main network (l b is zeros in this case).

It also minimizes the effects of BM on the gradients flow during the backpropagation stage.

The training is conducted in an end-to-end manner as presented in Algorithm 1.In our approach, the introduction of BM does not require additional loss function and the loss is directly incorporated into the main network loss (L m ).

To do this we introduce additional rewards for desired behaviors of the agent, similar to PE effect on human behavior formation process.

In our i ← random(1,size(B)) setting behavior is defined by agent's preference to play specific actions.

Thus, each preferred action played was rewarded with an extra action reward.

The action reward is subject to the game of interest and its designing process will be described in the Experiment section.

One of the advantages of network modularization is to allow the learned BMs to be transferred to a different main network with minimal or no network re-training.

This property is useful for knowledge sharing among a group of models in problems with a variety of possible implementations, changing environments and open-ended tasks.

Once task objectives have changed or new behaviors were developed in another model, the target model can just apply a new module without any updates or training of the main network.

This property allows easy extension of a learned model and knowledge share across different models without any additional training.

The learned BMs from the previous section is used during the transfer phase.

In this work, we consider two approaches, namely fine-tuning and adversarial transfer.

The first approach uses a source task model and fine-tunes it for a new target task, where BMs are kept unchanged.

In the adversarial transfer approach, we introduce a discriminator network (as shown in FIG0 , which enforces the convolutional layers to learn features similar to features of the source task.

To do so, we adopt the domain-adversarial training BID10 .

In this case, the discriminator network has 2 fully-connected layers with Relu and Softmax non-linearity functions, which tries to classify output of the last convolutional layers as being from the source or target task.

Different from the original paper, we minimize the softmax loss at the discriminator output and flip the gradient sign at the convolutional layers.

The weights update can be formulated as follows: DISPLAYFORM0 where θ dj t are the parameters of the discriminator at timestep t, θ aj t are the parameters of the convolutional layers at timestep t, β is the parameter as described by BID10 , L a is the classification loss of the descriminator.

In this section, we delineate the experiments that focus on two main aspects of this work: (1) the separation of agent's behavior from the main task, and (2) cross-task transfer of learned behaviors.

In order to demonstrate the flexibility and extendability of the proposed network, we also considered zero-shot setting so that an end-user will not require additional training for the case of behavior module transfer.

We evaluate the proposed novel modular architecture on the classic Atari 2600 games BID2 .

The main reason is that video games provide a controlled environment, where it is easier to control agent behavior by representing it with distribution over available actions.

In addition, Atari 2600 emulator does not require data collection and labeling, yet it provides a wide range of tasks in terms of different games.

The loss function used to encourage the learning of a specific behavior is described in the next section.

In this work, we evaluate our architecture on four games, namely Pong, Space Invaders, Demon Attack and Breakout, which consist of four available actions, namely No action, Fire, Up/Right, and Down/Left.

Data pre-processing: The input of the network is a stack of 4 game-frames.

Each frame was converted into a gray-scale image and resized to 84 × 84 pixels.

As the consecutive game frame contain similar information, we sample one game frame on every fourth frame.

and an additional action.

Additionally, we tested the effect of zero-behavior (i.e. BM0) presence during the training stage.

In other words, no actions were stimulated and BM is not applied.

Training: To train the proposed network (i.e. main network and BMs), we used the standard Qlearning algorithm and the corresponding loss function using Bellman equation BID19 .

The training used a combined reward represented by the sum of game score and individual action rewards.

The magnitude of the additional reward was represented by an average reward per frame of the game.

All the rewards obtained during the game were clipped at 1 and -1 BID20 .Evaluation Metrics: We evaluate the proposed models using two metrics.

The first metric focus on the game play performance.

As each game has different reward scales, we compute the mean of game scores ratios achieved by the proposed modular network and the Vanilla DQN model TAB0 .

We refer to this metric as Average Main Score Ratio (AMSR).

If AMSR is equal to 1, it means the trained model with BM performs equally well as the Vanilla DQN model.

Similarly, AMSR higher than 1 indicate our proposed model perform better than the Vanilla DQN, or worst if it is lower than 1.

Thus, AMSR that is close to or more than 1 would indicate our modular network is comparable to baseline.

The second metric reflects the capability of the proposed modular network in term of modeling the desired behavior.

To do this, we define the Behavior Distance (BD) by calculating the Bhattacharyya distance BID3 between the BMs' action distribution to an ideal target distribution.

The target distribution is computed by divide 1 with the rewarded actions (i.e. BM5's target distribution is [0.0, 0.0, 0.5, 0.5]).

In the ideal case, the BD of the learned network should be close to 1 as our training only encourages over certain actions set.

This experiment aims to show that the behavior functionality can be separated from the main task and learned independently.

To demonstrate that we conducted the training in two stages.

During the Stage 1, we first trained the main network with five behaviors (i.e., BM0 -BM2, BM4 -BM5 and BM8) using Algorithm 1.

Given the trained network from Stage 1, Stage 2 focused on training of the remaining BMs (i.e. BM3, BM6, and BM7) while the main network was frozen.

This includes behaviors stimulating 1, 2 and 3 actions, respectively.

Effect of key parameters: First of all, we studied the effect of the action reward magnitude on the training process.

We started with estimation of average reward per frame value (r) for each game (without additional action rewards) and observed performance of our model with various action rewards (i.e. 0.25r, 0.5r, r, 2r, and 5r).

TAB2 show that action reward magnitude directly affects the quality of learned behavior in both stages, where increasing the action reward above r value leads to degradation of the main task's performance.

Although additional reward magnitude selection depends on a desired main score and BD, we recommend the value equal to r as it leads to the highest BD score during Stage 2, and as a result better functionality separation.

Next, to see the effect of other parameters we set the value of the action reward to 0.5r, so that we can observe an effect of the changes on the main reward, as well as the behavior pattern.

As the next step, we have studied the effect of the complexity of the BMs on the quality of the learned behaviors by trying a different number of layers.

Also, we looked for a better separation of the Behavior component by studying the effects of dropout, BM0 and different learning rate for the Behavior module.

According to the results TAB3 use of 2 fully-connected layers resulted in a significant improvement on the main task score compared to 1-layer version.

However, adding more layers did not result in a better performance.

Similar effect demonstrated a higher BM learning rate compared to the main network TAB4 , while lower value leads to lower main scores.

Finally implementing a Dropout layer for the BM and using BM0 resulted in a higher BD score during Stage 2 and main score during Stage 1.Results: Taking into account hyper-parameter effect we have trained a final model with 2 layer BMs and 0.5 dropout layers, applying 2 times higher BM learning rate, BM0 and action reward equal to r. The trained model showed high main task scores compared to the vanilla DQN model, as well as high similarity of learned behaviors to ideal target distributions at Stage 1, as well as after separate training of BMs at Stage 2 TAB0 .

Experiments also showed that removing the BMs does not lead to a performance degradation of the model on the main task.

Importantly, the effect of the action reward magnitude directly correlated with agents preferences to play rewarded actions, which aligns with the PE effect in human behavior formation process.

Thus, the development process of exact behavior pattern can be controlled through variations in action reward magnitude.

Therefore, we conclude that the proposed model and training method allows a successful separation of the strategy (main task) and behavior functionalities and further network expansion.

Implementation details: To achieve a zero-shot performance of the transferred modules, we aimed to achieve a similar feature representation of the target model to the source model.

To achieve that we tested two approaches: fine-tuning and adversarial transfer.

In the first case, we have fine-tuned the main network obtained in Stage 1 of Section 4.1 on a new game with frozen Stage 1 BMs, applied to every pair of games and following Algorithm 1.

After that, we tested the performance on previously unseen Stage 2 BMs.

In adversarial setting we follow the same procedure, but with the use of the Discriminator part FIG0 ).

The performance was compared to the results of transferring Stage 2 BMs to the best model configuration after Stage 1 from Section 4.1.Results: As it can be seen from the Table 5 even a simple zero-shot transfer of learned BMs based on fine-tuning results in a good performance of the model on unseen BMs.

BM0 and Stage 1 behaviors achieved close performance to an original network.

Although the BD score of zero-shot adversarial transfer is approximately 9% lower, the main task performance of transferred modules on an unseen task is close to a separately trained network.

This fact shows that zero-shot transfer of separately learned BMs to unseen tasks results in slightly worse performance compared to the separately trained model.

This leads to a conclusion that target performance of transferred BMs can be achieved through much less training compared to complete network retraining.

In this work, we have proposed a novel Modular Network architecture with Behavior Module, inspired by human brain Pre-Frontal Cortex connectivity.

This approach demonstrated the successful separation of the Strategy and Behavior functionalities among different network components.

This is particularly useful for network expandability through independent learning of new Behavior Modules.

Adversarial 0-shot transfer approach showed high potential of the learned BMs to be transferred to unseen tasks.

Experiments showed that learned behaviors are removable and do not degrade the performance of the network on the main task.

This property allows the model to work in a general setting, when user preferences are unknown.

The results also align with human behavior formation process.

We also conducted an exhaustive study on the effect of hyper-parameters on behavior learning process.

As a future work, we are planning to extend the work to other domains, such as style transfer, chat bots, and recommendation systems.

Also, we will work on improving module transfer quality.

In this appendix, we show the details of our preliminary study on various key parameters.

The experiments were conducted on the Behavior Separation task.

@highlight

Extendable Modular Architecture is proposed for developing of variety of Agent Behaviors in DQN.