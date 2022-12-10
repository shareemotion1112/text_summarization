Exploration in environments with sparse rewards is a key challenge for reinforcement learning.

How do we design agents with generic inductive biases so that they can explore in a consistent manner instead of just using local exploration schemes like epsilon-greedy?

We propose an unsupervised reinforcement learning agent which learns a discrete pixel grouping model that preserves spatial geometry of the sensors and implicitly of the environment as well.

We use this representation to derive geometric intrinsic reward functions, like centroid coordinates and area, and learn policies to control each one of them with off-policy learning.

These policies form a basis set of behaviors (options) which allows us explore in a consistent way and use them in a hierarchical reinforcement learning setup to solve for extrinsically defined rewards.

We show that our approach can scale to a variety of domains with competitive performance, including navigation in 3D environments and Atari games with sparse rewards.

Exploration in environments with sparse feedback is a key challenge for deep reinforcement learning (DRL) research.

In DRL, agents typically explore with local exploration strategies like epsilon greedy or entropy based schemes.

We are interested in learning structured exploration algorithms, grounded in spatio-temporal visual abstractions given raw pixels.

In human perception and its developmental trajectory, spatio-temporal pixel groupings is one of the first visual abstractions to emerge, which is also used for intrinsically motivated goal-driven behaviors BID16 .

Inspired by this insight, we develop a new agent architecture and loss functions to autonomously learn visual abstractions and ground temporally extended behaviors in them.

Our approach and key contributions can be broken down into two parts: (1) an information theoretic loss function and a neural network architecture to learn visual groupings (abstractions) given raw pixels and actions, (2) a hierarchical action-value function agent which explores in the space of options grounded in the learned visual abstractions, instead of low level actions.

In the first step, we pass images through an encoder which outputs spatial discrete vector-quantized (VQ') grids, with 1 of E discrete components.

We train this encoder to maximize the mutual information between VQ layers at different time steps, in order to obtain a temporally consistent representation, that preserves controllability and appearance information.

We extract segmentation masks from the VQ layers for the second step, referred to as visual entities.

We compute affine geometric measurements for each entity, namely centroid and area of the corresponding segment.

We use off-policy learning to train action-value function to minimize or maximize these measurements, referred collectively as the options bank.

Controlling these measurements enable higher levels of behaviors such as approaching an object (maximizing area), avoiding objects (minimize area or minimize/maximize centroid coordinates), moving an object away towards the left (minimize centroid x coordinate), controlling the avatars position on the screen etc.

Finally, given a task reward, we use off-policy learning to train a meta action-value function that takes actions at fixed intervals and selects either one of the policies in the options bank or low-level actions.

So effectively, this hierarchical action-value function setup solves a semi markov decision process as in BID18 BID9 .We demonstrate that our approach can scale to two different domains -navigation in a 3D environment and challenging Atari games -given raw pixels.

Although much work remains in improving the visual and temporal abstraction discovery models, our results indicate that it is possible to learn bottom-up structured exploration schemes with simple spatial inductive biases and loss functions.

Learning visual abstractions has a long history in computer vision with some of the earlier successes relying on clustering either for inference BID15 or learning BID12 .

More recently some of these intuitions were adapted to neural networks as in BID24 .

Instance segmentation algorithms have been developed to output spatio-temporal groupings of pixels from raw videos BID13 .

However, most of the existing deep learning based approaches require supervised data.

Structured deep generative models BID20 is another approach to learning disentangled representations from raw videos.

Very recently segmentation has been cast as a mutual information maximization problem in BID8 where the mutual information is computed between the original image segmentation and the output of its transformation (additive color changes in HSV space, horizontal flips and spatial shifts).

That approach uses the discrete mutual information estimate which makes it only applicable to enforce pixel-label constraints.

For continuous variables recent papers have proposed very promising techniques.

Most relevant to our work are BID21 and BID2 .In reinforcement learning research, semi-MDPs and the options framework BID18 have been proposed as a model for temporal abstractions of behaviors.

Our work is most similar to hierarchical-DQN BID9 .

However, this approach required hand-crafted instance segmentation and the agent architecture is not distributed to learn about many intrinsic rewards learners at the same time.

Object-Oriented-MDPs BID3 uses object oriented representations for structured exploration but requires prebuilt symbolic representations.

A recent paper also demonstrates the importance of object based exploration when humans learn to play video games BID4 .

HRA BID22 is an agent that used prebuilt object representations to obtain state of the art policies on Pacman using object based structured exploration.

Another interesting line of work is BID6 which formalizes a notion of empowerment which ends up as a mutual information between options in the same MDP.

Count-based exploration algorithms have yielded impressive results on hard exploration Atari games BID11 .The Horde architecture BID19 proposed learning many value functions, termed as Generalized Value Functions (GVFs), using off-policy learning.

This work was later extended with neural networks by BID14 .

Our approach automatically constructs GVFs or a UVFA using the abstract entity based representations.

Our work is also related to pixel control BID7 as an auxiliary task.

However, we learn to control and compose abstract discrete representations.

Notation.

Unless explicitly stated otherwise, we will use calligraphic upper-case letters for sets and MDPs, upper-case letters for random variables, bold capitals for constants and lower-case letters for realizations/measurements.

Consider a Markov Decision Process M = (S, A, P, r) (MDP) represented by states s ∈ S, actions a ∈ A, transition distribution function P : S × A × S → [0, 1] and an extrinsic reward function defined as r : S × A → R. In a discrete MDP, an agent observes a state s t at time t, produces and action a t then the agent can observe s t+1 ∼ P(S |S = s t , A = a t ) and a reward r t+1 = r(s t , a t ).

The agent's objective is to maximize the expected sum of rewards over time.

In this work we are focusing on visual inputs thus we assume S ⊂ R H×W , where H, W are the height and width of the image.

This is a very important special case in current applications but many of the intuitions and machinery we develop should carry over to different domains.

DISPLAYFORM0 forces the VQ to distinguish frames in the same unroll or large temporal segment with frames outside of this window.

I((G t , G t+1 ), A t ) encourages the VQ to encode action controllable elements in the frame.

I(V t , C t ) forces the VQ to represent color information.

Our agent also learns an separate abstract representation we call visual entities 1 and denote with v ∈ V, where V ⊂ {1 . . .

E} H×W i.e. it assigns an entity id to each location.

These are meant to capture useful information about the state s and form the basis for computing the intrinsic rewards r e,m (see Sec. 3.1).

Here e ∈ {1 . . .

E} denotes a discrete entity id, where E is the maximum number of possible entities, and m ∈ {1 . . .

M} denotes a geometric feature of e's derived segmentation mask, from M possible measurements.

In this work the measurements are fixed to what we consider a sufficient set that captures the essential information for natural 3D navigation and Atari game play, and settled on centroid cartesian coordinates and entity mask area, which we can both minimize and maximize, thus in all our experiments M = 6.

Temporal changes in these measurements constitute our intrinsic reward functions 2 which induce E × M additional MDPs O e,m = (S, A, P, r e,m ).

These intrisic rewards will induce behaviors which should hopefully provide structured exploration in the original MDP i.e. picking a random Q e,m is more likely to lead to higher reward than epsilon greedy exploration.

Our agent architecture tries to leverage this additional structure.

The top level MDP M is represented by Q meta which outputs action a meta t at time t (switched every T time steps), where the action space is discrete 1 of (E × M) + 1 possible actions.

In our implementation this is modelled by composite actions E + 1 and M i.e. a meta t = (e t , m t ) with e t ∈ {1 . . .

E + 1} and m t ∈ {1 . . .

M}. We also learn (E × M) + 1 separate Q functions: (E × M) that each solve one of the O e,m and one for the original MDP Q task .

These Q functions are defined over the environment action set and differ only in the reward function (see Figure 2 for schematic representation).

Our agent relies on an abstraction model that assigns each pixel in the image to one of E separate abstractions or entities.

To obtain this representation, the image is passed through a convolutional network (CNN) encoder to output a spatial grid of the same resolution as the original image.

Then a vector quantized layer V (van den BID20 assigns the planar activations to 1 of E entities.

From the agents perspective this means all the pixels that are being grouped together become indistinguishable thus providing the visual abstraction we desire.

Let us define f : R H×W → {1 . . .

E} H×W the function that takes the observation s t at time t and computes an abstract representation corresponding to it v t = f (s t ).

The key question is how to train f .

One way is to make it representative of the current state.

In most of the current literature, this is Figure 2: Agent Architecture: (a) The input s t is used to compute the visual entities v t .

Their one-hot encoding is a set of E {0, 1} masks.

These are used to compute geometric measurements such as: area and centroid positions (Sec. 3).

Temporal differences in these constitute intrinsic rewards r int for an option bank.

(b) The input is separately passed through different a CNN and LSTM network, whose output is then fed to: Q task , Q meta and options bank with E × M Q functions.

Q meta and Q task are both trained with external task reward but the options bank is trained with the previously computed measurements r int .

To act, Q meta outputs a new action every T steps.

Its actions correspond to selecting and executing either: (1) one of the Q functions in the options bank or (2) the Q task policy.

The selected Q function is then used to produce the actions returned to the environment.

All Q functions are trained simultaneously, off-policy, from a shared replay buffer D.achieved by training a function g : {1 . . .

E} H×W → R H×W such that g(f (s t )) s t .

This ensures that the representation preserves all the information about the state which is a sufficient condition for the purpose.

It is not hard to see however that it potentially wastes model capacity on unimportant factors of variation.

In this work we aim instead for a necessary condition for constraining the representation.

To this end we train f to be an injective function i.e. such that a decoding function g can distinguish between states by looking at the representation ∀x = y =⇒ g(f (x)) = g(f (y)).Our approach is to formulate classification losses to promote distinguishability between different states at different levels of the representation.

This was shown, in works like BID0 BID2 BID21 , to be equivalent to maximizing a lower-bound mutual information between random variables of the representation 3 .

By choosing the appropriate random variables and sampling strategies from the joint distribution and marginals we can specify the right representation invariances as follows (see FIG0 for an overview):Preserving global information.

The main term driving the representation learning is a global information term.

To estimate it, the VQ layer output at time t, v t , is further processed by another CNN encoder to output a frame level embedding vector G t .

We train a non-parametric classifier to distinguish pairs of frames from the same trajectory from pairs of frames from different trajectories.

For that we form pairs (g t , g t+∆ ), where ∆ is sampled randomly, from pairs (g t , g ), where g is sampled randomly from a different trajectory.

Training this classifier lets us indirectly maximize a lower-bound on I(G t , G t+∆ ).

This forces V to preserve enough information that distinguishes this particular trajectory from other ones which tends to remove all irrelevant information like textures and unchanging "background" elements and preserve useful moving elements.

Note that as long as predictions are stable across time there is no pressure exerted by this cost to simplify the representation.

Preserving controllable information.

Secondly, we want the controllable information to be preserved in the abstract representation.

That is, we want to know what aspects of the input were changed as a result of the action.

We achieve this by training to predict which action was taken in a particular transition based on our representation.

For that we add another loss, denoted by I((G t , G t+1 ), A t ), that maximizes a lower-bound on mutual information between the a pair of consecutive frames and the action that was taken in the transition.

Preserving local appearance information.

Finally, for hard exploration Atari games, where inputs change very little e.g. Montezuma's Revenge much of the initial experience is from the first room, we add an appearance term 4 .

This term is meant to align the abstract representation with appearance changes i.e. promote abstractions have consistent colors.

For that we feed the input image into a shallow CNN encoder which outputs an embedding C of the local color and texture structure of the image.

We would like for the abstract representation to follow the appearance changes thus we maximize a lower bound on mutual information I(V t , C t ) in the same way as before.

The positive pairs spatially aligned V and C embeddings, and the negative pairs are obtained by sampling appearance embeddings from other spatial locations and pairing them with non-sampled V .

This promotes V representations that represent well appearance under the representational constraints of the VQ representation.

To learn the agents' abstract representation we minimize a weighted sum of these classification losses DISPLAYFORM0 where we denote by lower case letter the samples from the corresponding uppercase random variables.

P(G) is the time independent marginal distribution over G t 's. We model q g as DISPLAYFORM1 with φ(., .) being the cosine similarity over the embeddings and K is a hyper parameter denoting the number of samples.

We can similarly derive the third term.

This is possible because the embeddings have the same dimensionality.

The second term is a cross entropy based action classifier.

The agent is represented primarily by three sets of Q functions: Q meta , Q task and {Q 1,1 , . . .

Q e,m } from the options bank.

Each Q function has a corresponding policy we denote by π, in our experiments these are epsilon greedy policies with respect to the corresponding Q function.

We denote T to be the fixed temporal commitment window for Q meta , which means that it acts every T steps.

Note that Q task and Q e,m act at each environment time step.

We can express all three Q functions as: DISPLAYFORM0 We represent each Q function with a deep Q network BID10 , for instance Q task (s, a) ≈ Q task (s, a; θ task ).

Each Q ∈ {Q task , Q meta , Q 1,1 , ..., Q e,m } can be trained by minimizing corresponding loss functions -L task (θ task ), L meta (θ meta ) and {L 1,1 DISPLAYFORM1 bank (θ e,m )}.

We store experiences (s t , (e t , m t , a t ), r t , s t+1 ) in D, a shared buffer from which all the different Q functions can sample to perform their updates.

The transitions are stored in such a way as to be able to sample trajectories.

The Q-learning BID17 objective can be written as minimizing the loss: where R τ = τ +U−1 t=τ γ t+τ r t + γ t+U max a Q task (s τ +U , a ; θ task ), where U is the length of the unroll.

The loss function L meta and all the L e,m bank can be written in a similar fashion.

In our experiments we use Q(λ) and learn all parameters with stochastic gradient descent by sampling experience trajectories from the shared replay buffer (see Algorithm 1 for details).

DISPLAYFORM2

The implementation has a learning setup inspired by the batched actor critic agent BID5 , with a central GPU learner and multiple actors (64 in most experiments).

See 6 for details regarding visual abstraction architecture and agent network architecture.

The exploration setup.

For the baseline we have 3 types of actors differing only in the exploration parameter corresponding to high exploitation, high exploration and medium exploration.

From the total number of 64 actors that corresponds to 20, 10 and 34 respectively with epsilon values of .001, .5, .01.

Both the meta agent and base policies require exploration so we keep the same split but the high exploration is considered independently for the base and meta policies i.e. half the actors have meta = .5 and base = .01 and half have meta = .1 and base = .5.

Furthermore we split equally the medium exploration actors in 3 groups with ( meta = .01, base = .1), ( meta = .33, base = .001) and ( meta = .01, base = .1) respectively.

This provides much more stable learning at both the higher and lower levels of the behavior hierarchy.

We tested our approach on a 3D navigation domain with sparse rewards and hard exploration requirements.

The domain, whose top down view (not visible to the agent) is shown in FIG2 , contains four rooms each having a textured number at the entrance.

The agent receives the image observation as well as hint, a number from 1 to 4, which indicates the number of the room which contains the target object, a green sphere.

The goal is to reach the target as often as possible in the Algorithm 1 Learning algorithm 1: Inputs: N number of episodes, base and meta exploration parameters, T the commitment length, λ task , λ meta , λ bank cost function parameters 2: Initialize experience replay buffer D and parameters {θ meta , θ task , θ 1,1 , ..., θ e,m } for the metacontrol agent, task agent and options models respectively.

3: for i = 1 to N do 4:Initialize environment and get start state s 5: DISPLAYFORM0 while s t is not terminal do

if t ≡ 0 mod T then 8: DISPLAYFORM0 end if 10:Compute abstract features v t from s t (Section 3.1).

Compute intrinsic rewards r int = (r e,m | ∀e ≤ E, m ≤ M) from v t and v t−1

: DISPLAYFORM0 else 15: DISPLAYFORM1 Execute a t and obtain next state s t+1 and extrinsic reward r t from environment 18: DISPLAYFORM2 Use RMSProp to optimize L abs (θ abs ) %see eq. FORMULA1 20: On these domains we achieve better or comparable results to a strong baseline.

DISPLAYFORM3 limited time budget (250 steps).

Once the goal is reached the target is relocated in another randomly chosen room and the hint is updated accordingly.

To solve the task the agent needs to associate the hint with its position in the environment, which in turn means exploring far away regions in the maze because the target may move in a far away place.

The policy bank provides exactly this type of exploration basis set.

A baseline agent with the same architecture but without the policy bank cannot solve this task in 100M steps see FIG2 .

In this domain we can also query the environment representation to determine if a given pixel comes from either a wall, the skyline, floor, one of the textured numbers or the target sphere.

An agent with our proposed architecture but computing intrinsic rewards based on this priviledged information solves the tasks easily.

Most interestingly, an agent that uses the abstraction inference method we propose can also solve it too, though with slightly worse performance.

To gain an insight into the evolution of the agent policy, we plot in FIG2 (c), a histogram of the Q meta policy actions during training (time is on the vertical axes and flows from top to bottom).

The leftmost 20 actions represent the policy bank and the rightmost one represents Q task .

We can see that the agent uses most of the policies in the bank but as training progresses learns to rely more and more on Q task which is the optimal behavior.

We have run our agent with the same architecture and parameters on hard exploration Atari games where we can show that our agent has better performance than the baseline with a comparable architecture and losses (see FIG3 ).

In order to see if our model scales to more visually challenging environments we have run our agent on 3 varied "DMLab-30" levels BID1 .

We show the training curves as a function of time in FIG4 .

In the "non-match" task the agent teleports first to a room with one object then a room with two, one of which is the same as before.

To get a positive reward the agent has to move on top of the other object, otherwise it receives a negative reward.

The sequence continues for a fixed number of steps.

Humans achieve 66 points and a state of the art agent gets 26.

Our Q learner baseline achieves only 9 points whereas our proposed agent achieves 33.

Though this is meant to be a memory task structured exploration seems to help achieve much better scores than the baselines.

We have also considered an experiment on the "keys doors" task.

In this task the agent has to successively pickup keys to unlock doors which seems like a task structure were our representation could be more effective than the baseline.

We found that, though both methods are competitive, our representation was not sufficent to learn a better policy.

We think that this may be due to noise in the abstraction inference as well as a planning aspect that is not well enough handled in our agent in its current form.

Finally, our agent is outperformed by the baseline on the challenging watermaze task.

This is most likely due to the policy bank exploring mostly straight trajectories rather than circular ones which are more appropriate on this task.

We have shown that it is possible to design unsupervised structured exploration schemes for modelfree DRL agents, with competitive performance on a range of environments given just raw pixels.

One of the biggest open question moving forward is to find strategies to balance structure or inductive biases and performance.

Our current solution was to augment the meta-controller with Q task along with the options bank as sub-behaviors.

The typical strategy that agents follow is to rely on the options bank early in training and then use this experience to train the Q task policy for optimality as training progresses.

This is reasonable given that the options models may not cover the optimal policy but could serve as a good exploration algorithm throughout training.

As new unsupervised architectures and losses are discovered, we expect to narrow the gap between the optimal desired behaviors and the options bank.

Learning visual entities from pixels is still a challenging open problem in unsupervised learning and computer vision.

We expect novel sampling schemes in our proposed architecture to improve the entity discovery results.

Other unsupervised video segmentation algorithms and discrete latent variable models could also be used to boost the discovery process.

Visual Abstraction Architecture.

The encoder for the abstractions is a set of 3 convolutional layers 3 × 3 kernels with 64 features each followed by ReLU nonlinearities.

After a 1 × 1 convolutional layer with 8 outputs the features are l 2 normalized.

We found that to work better in practice with the VQ layer sitting on top.

The VQ layer has 8 elements unless otherwise stated.

Since the strides are always 1 the VQ output has the same spatial dimensions as the input image.

The global loss is a stack of convolutions 3 × 3, followed by a 2 × 2 max pooling with stride two to reduce the resolution of the output and then a ReLU non-linearity.

The output is then flattened to give the global embedding of the abstract representation.

The image embedding is a two-layer 8 filter convnet with a ReLU non-linearity inbetween.

Agent architecture.

The agent architecture is a standard 3 layer convolutional stack similar to BID10 with 512 hidden unit output and an LSTM on top.

The output of the LSTM is fed into an visual abstraction selection Q function, a measurement selection Q function, a regular task Q function layer and the policy bank i.e. a layer with M × E × num actions outputs.

The Q function layers are all dueling heads as in BID23 .Setup and baseline.

Our setup is very similar to the one in BID5 .

Multiple actors (64 in most examples) in the acting loop that send trajectories to a shared learner which processes them in batched fashion (batch size of 32).

The main difference is the use of value based Q(λ) loss (with a λ value of .85) instead of actor-critic with off-policy correction.

The baseline agent has the same exact architecture and loss as our agent.

In fact if we ignored the meta control Q function and the options bank we get our baseline exactly.

@highlight

structured exploration in deep reinforcement learning via unsupervised visual abstraction discovery and control

@highlight

The paper introduces visual abstractions that are used for reinforcement learning, where an algorithm learns to "control" each abstraction as well as select the options to achieve the overall task.