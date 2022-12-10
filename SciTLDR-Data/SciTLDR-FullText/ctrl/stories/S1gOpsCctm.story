Recurrent neural networks (RNNs) are an effective representation of control policies for a wide range of reinforcement and imitation learning problems.

RNN policies, however, are particularly difficult to explain, understand, and analyze due to their use of continuous-valued memory vectors and observation features.

In this paper, we introduce a new technique, Quantized Bottleneck Insertion, to learn finite representations of these vectors and features.

The result is a quantized representation of the RNN that can be analyzed to improve our understanding of memory use and general behavior.

We present results of this approach on synthetic environments and six Atari games.

The resulting finite representations are surprisingly small in some cases, using as few as 3 discrete memory states and 10 observations for a perfect Pong policy.

We also show that these finite policy representations lead to improved interpretability.

Deep reinforcement learning (RL) and imitation learning (IL) have demonstrated impressive performance across a wide range of applications.

Unfortunately, the learned policies are difficult to understand and explain, which limits the degree that they can be trusted and used in high-stakes applications.

Such explanations are particularly problematic for policies represented as recurrent neural networks (RNNs) BID16 BID14 , which are increasingly used to achieve state-of-the-art performance BID15 BID21 .

This is because RNN policies use internal memory to encode features of the observation history, which are critical to their decision making, but extremely difficult to interpret.

In this paper, we take a step towards comprehending and explaining RNN policies by learning more compact memory representations.

Explaining RNN memory is challenging due to the typical use of high-dimensional continuous memory vectors that are updated through complex gating networks (e.g. LSTMs, GRUs BID10 BID5 ).

We hypothesize that, in many cases, the continuous memory is capturing and updating one or more discrete concepts.

If exposed, such concepts could significantly aid explainability.

This motivates attempting to quantize the memory and observation representation used by an RNN to more directly capture those concepts.

In this case, understanding the memory use can be approached by manipulating and analyzing the quantized system.

Of course, not all RNN policies will have compact quantized representations, but many powerful forms of memory usage can be captured in this way.

Our main contribution is to introduce an approach for transforming an RNN policy with continuous memory and continuous observations to a finite-state representation known as a Moore Machine.

To accomplish this we introduce the idea of Quantized Bottleneck Network (QBN) insertion.

QBNs are simply auto-encoders, where the latent representation is quantized.

Given a trained RNN, we train QBNs to encode the memory states and observation vectors that are encountered during the RNN operation.

We then insert the QBNs into the trained RNN policy in place of the "wires" that propagated the memory and observation vectors.

The combination of the RNN and QBN results in a policy represented as a Moore Machine Network (MMN) with quantized memory and observations that is nearly equivalent to the original RNN.

The MMN can be used directly or fine-tuned to improve on inaccuracies introduced by QBN insertion.

While training quantized networks is often considered to be quite challenging, we show that a simple approach works well in the case of QBNs.

In particular, we demonstrate that "straight through" gradient estimators as in BID1 BID6 are quite effective.

We present experiments in synthetic domains designed to exercise different types of memory use as well as benchmark grammar learning problems.

Our approach is able to accurately extract the ground-truth MMNs, providing insight into the RNN memory use.

We also did experiments on 6 Atari games using RNNs that achieve state-of-the-art performance.

We show that in most cases it is possible to extract near-equivalent MMNs and that the MMNs can be surprisingly small.

Further, the extracted MMNs give insights into the memory usage that are not obvious based on just observing the RNN policy in action.

For example, we identify games where the RNNs do not use memory in a meaningful way, indicating the RNN is implementing purely reactive control.

In contrast, in other games, the RNN does not use observations in a meaningful way, which indicates that the RNN is implementing an open-loop controller.

There have been efforts made in the past to understand the internals of Recurrent Networks BID12 BID0 BID22 BID17 BID11 BID18 .

However, to the best of our knowledge there is no prior work on learning finite-memory representations of continuous RNN policies.

Our work, however, is related to a large body of work on learning finite-state representations of recurrent neural networks.

Below we summarize the branches of that work and the relationship to our own.

There has been a significant history of work on extracting Finite State Machines (FSMs) from recurrent networks trained to recognize languages BID28 BID23 BID3 .

Typical approaches include discretizing the continuous memory space via gridding or clustering followed by minimization.

A more recent approach is to use classic query-based learning algorithms to extract FSMs by asking membership and equivalence queries BID26 .

However, none of these approaches directly apply to learning policies, which require extending to Moore Machines.

In addition, all of these approaches produce an FSM approximation that is separated from the RNN and thus serve as only a proxy of the RNN behavior.

Rather, our approach directly inserts discrete elements into the RNN that preserves its behavior, but allows for a finite state characterization.

This insertion approach has the advantage of allowing fine-tuning and visualization using standard learning frameworks.

The work most similar to ours also focused on learning FSMs BID28 .

However, the approach is based on directly learning recurrent networks with finite memory, which are qualitatively similar to the memory representation of our MMNs.

That work, however, focused on learning from scratch rather than aiming to describe the behavior of a continuous RNN.

Our work extends that approach to learn MMNs and more importantly introduces the method of QBN insertion as a way of learning via guidance from a continuous RNN.This transforms any pre-trained recurrent policy into a finite representation.

We note that there has been prior work on learning fully binary networks, where the activation functions and/or weights are binary (e.g. BID1 BID6 BID8 ).

The goal of that line of work is typically to learn more time and space efficient networks.

Rather, we focus on learning only discrete representations of memory and observations, while allowing the rest of the network to use arbitrary activations and weights.

This is due to our alternative goal of supporting interpretability rather than efficiency.

Recurrent neural networks (RNNs) are commonly used in reinforcement learning to represent policies that require or can benefit from internal memory.

At each time step, an RNN is given an observation o t (e.g. image) and must output an action a t to be taken in the environment.

During execution an RNN maintains a continuous-valued hidden state h t , which is updated on each transition and influences the action choice.

In particular, given the current observation o t and current state h t , an RNN performs the following operations: 1) Extract a set of observation features f t from o t , for example, using a CNN when observations are images, 2) Outputting an action a t = π(h t ) according to policy π, which is often a linear softmax function of h t , 3) transition to a new state h t+1 = δ(f t , h t ) where δ is the transition function, which is often implemented via different types of gating networks such as LSTMs or GRUs.

The continuous and high dimensional nature of h t and f t can make interpreting the role of memory difficult.

This motivates our goal of extracting compact quantized representations of h t and f t .

Such representations have the potential to allow investigating a finite system that captures the key features of the memory and observations.

For this purpose we introduce Moore Machines and their deep network counterparts.

Moore Machines.

A classical Moore Machine (MM) is a standard finite state machine where all states are labeled by output values, which in our case will correspond to actions.

In particular, a Moore Machine is described by a finite set of (hidden) statesĤ, an initial hidden stateĥ 0 , a finite set of observationsÔ, a finite set of actions A, a transition functionδ, and a policyπ that maps hidden states to actions.

The transition functionδ :Ĥ ×Ô →Ĥ returns the next hidden statê h t+1 =δ(ĥ t ,ô t ) given the current stateĥ t and observationô t .

By convention we will use h t andĥ t to denote continuous and discrete states respectively and similarly for other quantities and functions.

Moore Machine Networks.

A Moore Machine Network (MMN) is a Moore Machine where the transition functionδ and policyπ are represented via deep networks.

In addition, since the raw observations given to an MMN are often continuous, or from an effectively unbounded set (e.g. images), an MMN will also provide a mappingĝ from the continuous observations to a finite discrete observation spaceÔ. Hereĝ will also be represented via a deep network.

In this work, we consider quantized state and observation representations where eachĥ ∈Ĥ is a discrete vector and each discrete observation inÔ is a discrete vector that describes the raw observation.

We will denote the quantization level as k and the dimensions ofĥ andf by B h and B f respectively.

Based on the above discussion, an MMN can be viewed as a traditional RNN, where: 1) The memory is restricted to be composed of k-level activation units, and 2) The environmental observations are intermediately transformed to a k-level representationf before being fed to the recurrent module.

Given an approach for incorporating quantized units into the backpropagation process, it is straightforward, in concept, to learn MMNs from scratch via standard RNN learning algorithms.

However, we have found that learning MMNs from scratch can be quite difficult for non-trivial problems, even when an RNN can be learned with relative ease.

For example, we have not been able to train highperforming MMNs from scratch for Atari games.

Below we introduce a new approach for learning MMNs that is able to leverage the ability to learn RNNs.

Given a trained RNN, our key idea is to first learn quantized bottleneck networks (QBNs) for embedding the continuous observation features and hidden state into a k-level quantized representation.

We will then insert the QBNs into the original recurrent net in such a way that its behavior is minimally changed with the option of fine-tuning after insertion.

The resulting network can be viewed as consuming quantized features and maintaining quantized state, which is effectively an MMN.

Below we describe the steps in further detail, which are illustrated in FIG0 .

A QBN is simply an autoencoder where the latent representation between the encoder and decoder (i.e. the bottleneck) is constrained to be composed of k-level activation units.

While, traditional autoencoders are generally used for the purpose of dimensionality reduction in continuous space BID9 , QBNs are motivated by the goal of discretizing a continuous space.

Conceptually, this can be done by quantizing the activations of units in the encoding layer.

We represent a QBN via a continuous multilayer encoder E, which maps inputs x to a latent encoding E(x), and a corresponding multilayer decoder D. To quantize the encoding, the QBN output is given by DISPLAYFORM0 In our case, we use 3-level quantization in the form of +1, 0 and −1 using the quantize function, which assumes the outputs of E(x) are in the range [−1, 1].

1 One choice for the output nodes of E(x) would be the tanh activation.

However, since the gradient of tanh is close to 1 near 0, it can be difficult to produce quantization level 0 during learning.

Thus, as suggested in Pitis (2017), to support 3-valued quantization we use the following activation function, which is flatter in the region around zero input.φ(x) = 1.5 tanh(x) + 0.5 tanh(−3x)Of course introducing the quantize function in the QBN results in b(x) being non-differentiable, making it apparently incompatible with backpropagation, since the gradients between the decoder and encoder will almost always be zero.

While there are a variety of ways to deal with this issue, we have found that the straight-through estimator, as suggested and used in prior work BID8 BID1 BID6 ) is quite effective.

In particular, the standard straightthrough estimator of the gradient simply treats the quantize function as the identity function during back-propagation.

Overall, the inclusion of the quantize function in the QBN effectively allows us to view the last layer of E as producing a k-level encoding.

We train a QBN as an autoencoder using the standard L 2 reconstruction error x − b(x) 2 for a given input x.

Given a recurrent policy we can run the policy in the target environment in order to produce an arbitrarily large set of training sequences of triples (o t , f t , h t ), giving the observation, corresponding observation feature, and hidden state at time t. Let F and H be the sets of all observed features and states respectively.

The first step of our approach is to train two QBNs, b f and b h , on F and H respectively.

If the QBNs are able to achieve low reconstruction error then we can view latent "bottlenecks" of the QBNs as a high-quality k-level encodings of the original hidden states and features.

We now view b f and b h as "wires" that propagate the input to the output, with some noise due to imperfect reconstruction.

We insert these wires into the original RNN in the natural way (stage-3 in FIG0 ).

The b f QBN is inserted between the RNN units that compute the features f and the nodes those units are connected to.

The b h QBN is inserted between the output of the recurrent network block and the input to the recurrent block.

If b f and b h always produced perfect reconstructions, then the result of inserting them as described would not change the behavior of the RNN.

Yet, the RNN can now be viewed as an MMN since the bottlenecks of b f and b h provide a quantized representation of the features f t and states h t .Fine Tuning.

In practice, the QBNs will not achieve perfect reconstruction and thus, the resulting MMN may not behave identically to the original RNN.

Empirically, we have found that the performance of the resulting MMN is often very close to the RNN directly after insertion.

However, when there is non-trivial performance degradation, we can fine-tune the MMN by training on the original rollout data of the RNN.

Importantly, since our primary goal is to learn a representation of the original RNN, during fine-tuning our objective is to have the MMN match the softmax distribution over actions produced by the RNN.

We found that training in this way was significantly more stable than training the MMN to simply output the same action as the RNN.

After obtaining the MMN, one could use visualization and other analysis tools to investigate the memory and it's feature bits in order to gain a semantic understanding of their roles.

Solving the full interpretation problem in a primitive way is beyond the scope of this work.

Extraction.

Another way to gain insight is to use the MMN to produce an equivalent Moore Machine over atomic state and observation spaces, where each state and observation is a discrete symbol.

This machine can be analyzed to understand the role of different machine states and how they are related.

In order to create the Moore Machine we run the learned MMN to produce a dataset of <ĥ t−1 ,f t ,ĥ t , a t >, giving the consecutive pairs of quantized states, the quantized features that led to the transition, and the action selected after the transition.

The state-space of the Moore Machine will correspond to the p distinct quantized states in the data and the observation-space of the machine will be the q unique quantized feature vectors in the data.

The transition function of the machineδ is constructed from the data by producing a p × q transaction table that captures the transitions seen in the data.

Minimization.

In general, the number of states p in the resulting Moore Machine will be larger than necessary in the sense that there is a much smaller, but equivalent, minimal machine.

Thus, we apply standard Moore Machine minimization techniques to arrive at the minimal 2 equivalent Moore Machine BID19 .

This often dramatically reduces the number of distinct states and observations.

Our experiments address the following questions: 1) Is it possible to extract MMNs from RNNs without significant loss in Performance?

2) What is the general magnitude of the number of states and observations in the minimal machines, especially for complex domains such as Atari?

3) Do the learned MMNs help with interpretability of the recurrent policies?

In this section, we begin addressing these questions by considering two domains where ground truth Moore Machines are known.

The first is a parameterized synthetic environment, Mode Counter, which can capture multiple types of memory use.

Second, we consider benchmark grammar learning problems.

The class of Mode Counter Environments (MCEs) allows us to vary the amount of memory required by a policy (including no memory) and the required type of memory usage.

In particular, MCEs can require varying amounts of memory for remembering past events and implementing internal counters.

An MCE is a restricted type of Partially Observable Markov Decision Process, which transitions between one of M modes over time according to a transition distribution, which can depend on the current mode and amount of time spent in the current mode.

There are M actions, one for each mode, and the agent receives a reward of +1 at the end of the episode if it takes the correct action associated with the active mode at each time step.

The agent does not observe the mode directly, but rather must infer the mode via a combination of observations and memory use.

Different parameterizations place different requirements on how (and if) memory needs to be used to infer the mode and achieve optimal performance.

Below we give an intuitive description of the MCEs 3 in our experiments.

We conduct experiments in three MCE instances, which use memory and observations in fundamentally different ways.

This tests our ability to use our approach for determining the type of memory use.

1) Amnesia.

This MCE is designed so that the optimal policy does not need memory to track past information and can select optimal actions based on just the current observation.

2) Blind.

Here we consider the opposite extreme, where the MCE observations provide no information about optimal actions.

Rather, memory must be used to implement counters that keep track of a deterministic mode sequence for determining the optimal actions.

3) Tracker.

This MCE is designed so that the optimal policy must both use observations and memory in order to select optimal actions.

Intuitively the memory must implement counters that keep track of key time steps where the observations provide information about the mode.

In all above instances, we used M = 4 modes.

For each MCE instance we use the following recurrent architecture: the input feeds into 1 feed-forward layer with 4 Relu6 nodes BID13 ) (f t ), followed by a 1-layer GRU with 8 hidden units (h t ), followed by a fully connected softmax layer giving a distribution over the M actions (one per mode).

Since we know the optimal policy in the MCEs we use imitation learning for training.

For all of the MCEs in our experiments, the trained RNNs achieve 100% accuracy on the imitation dataset and appeared to produce optimal policies.

MMN Training.

The observation QBN b f and hidden-state QBN b h have the same architecture, except that the number of quantized bottleneck units B f and B h are varied in our experiments.

The encoders consist of 1 feed-forward layer of tanh nodes, where the number of nodes is 4 times the size of the bottleneck.

This layer feeds into 1 feedforward layer of quantized bottleneck nodes (see Section 4).

The decoder for both b f and b h has a symmetric architecture to the encoder.

Training of b f and b h in the MCE environments was extremely fast compared to the RNN training, since QBNs do not need to learn temporal dependencies.

We trained QBNs with bottleneck sizes of B f ∈ {4, 8} and B h ∈ {4, 8}. For each combination of B f and B h we embedded the QBNs into the RNN to give a discrete MMN and then measured performance of the MMN before and after fine tuning.

TAB0 gives the average test score over 50 test episodes.

Score of 1 indicates the agent performed optimally for all episodes.

In most of the cases no fine tuning was required (marked as '-') since the agent achieved perfect performance immediately after bottleneck insertion due to low reconstruction error.

In all other cases, except for Tracker (B h = 4, B f = 4) fine-tuning resulted in perfect MMN performance.

The exception yielded 98% accuracy.

Interestingly in that case, we see that if we only insert one of the bottlenecks at a time, we yield perfect performance, which indicates that the combined error accumulation of the two bottlenecks is responsible for the reduced performance.

Moore Machine Extraction.

TAB0 also gives the number of states and observations of the MMs extracted from the MMNs both before and after minimization.

Recall that the number of states and obsevations before minimization is the number of distinct combinations of values observed for the bottleneck nodes during long executions of the MMN.

We see that there are typically significantly more states and observations before minimization than after.

This indicates that the MMN learning does not necessarily learn minimal discrete state and observation representations, though the representations accurately describe the RNN.

After minimization (Section 4.3), however, in all but one case we get exact minimal machines for each MCE domain.

The ground truth minimal machines that are found are shown in the Appendix (Figure 3 ).

This shows that the MMNs learned via QBN insertions were equivalent to the true minimal machines and hence indeed optimal in most cases.

The exception matches the case where the MMN did not achieve perfect accuracy.

Examining these machines allows one to understand the memory use.

For example, the machine for Blind has just a single observation symbol and hence its transitions cannot depend on the input observations.

In contrast, the machine for Amnesia shows that each distinct observation symbol leads to the same state (and hence action choice) regardless of the source state.

Thus, the policies action is completely determined by the current observation. , 2017) ).

Here we evaluate our approach over the 7 Tomita Grammars 4 , where each grammar defines the set of binary strings that should be accepted or rejected.

Since, our focus is on policy learning problems, we treat the grammars as environments with two actions 'accept' and 'reject'.

Each episode corresponds to a random string that is either part of the particular grammar or not.

The agent receives a reward of 1 if the correct action accept/reject is chosen on the last symbol of a string.

RNN Training.

The RNN for each grammar is comprised of a one-layer GRU with 10 hidden units, followed by a fully connected softmax layer with 2 nodes (accept/reject).

Since we know the optimal policy, we again use imitation learning to train each RNN using the Adam optimizer and learning rate of 0.001.

The training dataset is comprised of an equal number of accept/reject strings with lengths uniformly sampled in the range [1, 50] .

TAB1 presents the test results for the trained RNNs giving the accuracy over a test set of 100 strings drawn from the same distribution as used for training.

Other than grammar #6 5 , the RNNs were 100% accurate.

MMN Training.

Since the raw observations for this problem are from a finite alphabet, we don't need to employ a bottleneck encoder to discretize the observation features.

Thus the only bottleneck learned here is b h for the hidden memory state.

We use the same architecture for b h as used for the MCE experiments and conduct experiments with B h ∈ {8, 16}. These bottlenecks were then inserted in the RNNs to give MMNs.

The performance of the MMNs before and after fine-tuning are shared in TAB1 .

In almost all cases, the MMN is able to maintain the performance of the RNN without fine-tuning.

Fine tuning provides only minor improvements in other cases, which already are achieving high accuracy.

Moore Machine Extraction.

Our results for MM extraction and minimization are in TAB1 .

In each case, we see a considerable reduction in the MM's state-space after minimization while accurately maintaining the MMN's performance.

Again, this shows that the MMN learning does not directly result in minimal machines, yet are equivalent to the minimal machines and hence are exact 4 The Tomita grammars are the following 7 languages over the alphabet {0, 1}: solutions.

In all cases, except for grammar 6 the minimized machines are identical to the minimal machines that are known for these grammars BID24 .

DISPLAYFORM0

In this section, we consider applying our technique to RNNs learned for six Atari 6 games using the OpenAI gym BID2 .

Unlike the above experiments, where we knew the ground truth MMs, for Atari we did not have any preconception of what the MMs might look and how large they might be.

The fact that the input observations for Atari (i.e. images) are much more complex than the previous experiments inputs makes it completely unclear if we can expect similar types of results.

There have been other recent efforts towards understanding Atari agents BID27 BID7 .

However, we are not aware of any other work which aims to extract finite state representations for Atari policies.

RNN Training.

All the Atari agents have the same recurrent architecture.

The input observation is an image frame, preprocessed by gray-scaling, 2x down-sampling, cropping to an 80 × 80 square and normalizing the values to [0, 1] .

The network has 4 convolutional layers (kernel size 3, strides 2, padding 1, and 32,32,16,4 filters respectively).

We used Relu as the intermediate activation and Relu6 over the last convolutional layer.

This is followed by a GRU layer with 32 hidden units and a fully connected layer with n+1 units, where n is the dimension of the Atari action space.

We applied a softmax to first n neurons to obtain the policy and used the last neuron to predict the value function.

We used the A3C RL algorithm BID16 ) (learning rate 10 −4 , discount factor 0.99) and computed loss on the policy using Generalized Advantage Estimation (λ = 1.0) BID20 .

We report the trained RNN performance on our six games in the second column of TAB2 .MMN Training.

We used the same general architecture for the QBN b f as used for the MCE experiments, but adjusted the encoder input and decoder output sizes to match the dimension of the continuous observation features f t .

For b h , the encoder has 3 feed-forward layers with (8 × B h ), (4 × B h ) and B h nodes.

The decoder is symmetric to the encoder.

For the Atari domains, the training data for b f and b h was generated using noisy rollouts.

In particular, each training episode was generated by executing the learned RNN for a random number of steps and then executing an -greedy (with = 0.3) version of the RNN policy.

This is intended to increase the diversity of the training data and we found that it helps to more robustly learn the QBNs.

We trained bottlenecks for B h ∈ {64, 128} and B f ∈ {100, 400} noting that these values are significantly larger than for our earlier experiments due to the complexity of Atari.

Note that while there are an enormous number of potential discrete states for these values of B h the actual number of states observed and hence the number of MMN states can be substantially smaller.

Each bottleneck was trained to the point of saturation of training loss and then inserted into the RNN to give an MMN for each Atari game.

MMN Performance.

TAB2 gives the performance of the trained MMNs before and after finetuning for different combinations of B h and B f .

We see that for 4 games, Pong, Freeway, Bowling, and Boxing, the MMNs after fine tuning either achieve identical scores to the RNN or very close (in the case of boxing).

This demonstrates the ability to learn a discrete representation of the input and memory for these complex games with no impact on performance.

We see that for Boxing and Pong fine tuning was required to match the RNN performance.

In the case of Freeway and Bowling, fine-tuning was not required.

In the remaining two games, Breakout and Space Invaders, we see that the MMNs learned after fine tuning achieve lower scores than the original RNNs, though the scores are still relatively good.

On further investigation, we found that this drop in performance was due to poor reconstruction on some rare parts of the game.

For example, in Breakout, after the first board is cleared, the policy needs to press the fire-button to continue, but the learned MMN does not do this and instead times out, which results in less score.

This motivates the investigation into more intelligent approaches for training QBNs to capture critical information in such rare, but critically important, states.

MM Minimization.

We see from TAB2 that before minimization, the MMs often have relatively large numbers of discrete states and observations.

This is unsurprising given that we are using relatively large values of B h and B f .

However, we see that after minimizing the MMNs the number of states and observations reduces by orders of magnitude, sometimes to just a single state and/or single observation.

The number of states and observations in many cases are small enough to write out and analyze by hand, making them amenable to careful analysis.

However, this analysis is likely to be non-trivial for moderately complex policies due to the need to understand the "meaning" of the observations and in turn of the states.

Understanding Memory Use.

We were surprised to observe in Atari the same three types of memory use considered for the MCE domains above.

First, we see that the MM for Pong has just three states (one per action) and 10 discrete observation symbols (see Figure 2a) .

Most importantly we see that each observation transitions to the same state (and hence action) regardless of the current state.

So we can view this MM as defining a set of rule that maps individual observations to actions with no memory necessary.

In this sense, the Pong policy is analogous to the Amnesia MCE [5.1].In contrast, we see that in both Bowling and Freeway there is only one observation symbol in the minimal MM.

This means that the MM actually ignores the input image when selecting actions.

Rather the policies are open-loop controllers whose action just depends on the time-step rather than the observations.

Thus, these policies are analogous to the Blind MCE [5.1] .

Freeway has a particularly trivial policy that always takes the Up action at each time step.

While this policy behavior could have been determined by looking at the action sequence of the rollouts, it is encouraging that our MM extraction approach also discovered this.

As shown in Figure 2b , Bowling has a more interesting open-loop policy structure where it has an initial sequence of actions and then a loop is entered where the action sequence is repeated.

It is not immediately obvious that this policy has such an open-loop structure by just watching the policy.

Thus, we can see that the MM extraction approach we use here can provide significant additional insight.

Breakout, Space Invaders and Boxing use both memory and observations based on our analysis of the MM transition structures.

We have not yet attempted a full semantic analysis of the discrete observations and states for any of the Atari policies.

This will require additional visualization and interaction tools and is an important direction of future work that is enabled by our approach.

Motivated by the goal of better understanding memory use in RNN polices, we introduced an approach for extracting finite state Moore Machines from those policies.

The key idea, bottleneck insertion, is to train Quantized Bottleneck Networks to produce binary encodings of continuous RNN memory and input features, and then insert those bottlenecks into the RNN.

This yields a BID15 , A3C BID16 From the MMN we then extract a discrete Moore machine that can then be transformed into an equivalent minimal machine for analysis and usage.

Our results on two environments where the ground truth machines are known show that our approach is able to accurately extract the ground truth.

We also show experiments in six Atari games, where we have no prior insight into the ground truth machines.

We show that, in most cases, the learned MMNs maintain similar performance to the original RNN policies.

Further, the extracted machines provide insight into the memory usage of the policies.

First, we see that the number of required memory states and observations is surprisingly small.

Second, we can identify cases where the policy did not use memory in a significant way (e.g. Pong) and policies that relied only on memory and ignored the observations (e.g. Bowling and Freeway).

To our knowledge, this is the first work where this type of insight was reported for policies in such complex domains.

A key direction for future work is to develop tools and visualizations for attaching meaning to the discrete observations and in turn states, which will allow for an additional level of insight into the policies.

It is also worth considering the use of tools for analyzing finite-state machine structure to gain further insight and analyze formal properties of the policies.

An MCE is parameterized by the mode number M , a mode transition function P , a mode life span mapping ∆(m) that assigns a positive integer to each mode, and a count set C containing zero or more natural numbers.

At time t the MCE hidden state is a tuple (m t , c t ), where m t ∈ {1, 2, . . .

, M } is the current mode and c t is the count of time-steps that the system has been consecutively in mode m t .

The mode only changes when the lifespan is reached, i.e. c t = ∆(m t ) − 1, upon which the next mode m t+1 is generated according to the transition distribution P (m t+1 | m t ).

The transition distribution also specifies the distribution over initial modes.

The agent does not directly observe the state, but rather, the agent only receives a continuous-valued observations o t ∈ [0, 1] at each step, based on the current state (m t , c t ).

If c t ∈ C then o t is drawn uniformly at random from [m t /M, (m t + 1)/M )] and otherwise o t is drawn uniformly at random from [0, 1].

Thus, observations determine the mode when the mode count is in C and otherwise the observations are uninformative.

Note that the agent does not observe the counter.

This means that to keep track of the mode for optimal performance the agent must remember the current mode and use memory to keep track of how long the mode has been active, in order to determine when it needs to "pay attention" to the current observation.

We conduct experiments with the following three MCE instances 7 : 1) Amnesia.

This MCE uses ∆(m) = 1 for all modes, C = {0}, and uniformly samples random initial mode and transition distributions.

Thus, an optimal policy will not use memory to track information from the past, since the current observation alone determines the current mode.

This tests our ability to use MMN extraction to determine that a policy is purely reactive, i.e. not using memory.

2) Blind.

Here we use deterministic initial mode and transition distributions, mode life spans that can be larger than 1, and C = {}.

Thus, the observations provide no information about the mode and optimal performance can only be achieved by using memory to keep track of the deterministic mode sequence.

This allows us to test whether the extraction of an MMN could infer that the recurrent policy is ignoring observations and only using memory.

3) Tracker.

This MCE is identical to Amnesia, except that the ∆(m) values can be larger than 1.

This requires an optimal policy to pay attention to observations when c t = 0 and use memory to keep track of the current mode and mode count.

This is the most general instance of the environment and can result in difficult problems when the number of modes and their life-spans grow.

In all above instances, we used M = 4.

We use 'A' and 'R' to denote accept and reject states, respectively.

Other than Grammar 6, all machines are 100% accurate.

<|TLDR|>

@highlight

Extracting a finite state machine from a recurrent neural network via quantization for the purpose of interpretability with experiments on Atari.