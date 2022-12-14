Deep reinforcement learning has succeeded in sophisticated games such as Atari, Go, etc.

Real-world decision making, however, often requires reasoning with partial information extracted from complex visual observations.

This paper presents  Discriminative Particle Filter Reinforcement Learning (DPFRL), a new reinforcement learning framework for partial and complex observations.

DPFRL encodes a differentiable particle filter with learned transition and observation models in a neural network, which allows for reasoning with partial observations over multiple time steps.

While a standard particle filter relies on a generative observation model, DPFRL learns a discriminatively parameterized model that is training directly for decision making.

We show that the discriminative parameterization results in significantly improved performance, especially for tasks with complex visual observations, because it circumvents the difficulty of modelling observations explicitly.

In most cases, DPFRL outperforms state-of-the-art POMDP RL models in Flickering Atari Games, an existing POMDP RL benchmark, and in Natural Flickering Atari Games, a new, more challenging POMDP RL benchmark that we introduce.

We further show that DPFRL performs well for visual navigation with real-world data.

Deep Reinforcement Learning (DRL) has attracted significant interest with applications ranging from game playing (Mnih et al., 2013; Silver et al., 2017) to robot control and visual navigation Kahn et al., 2018; Savva et al., 2019) .

However, more natural or real-world environments pose significant challenges for current DRL methods (Arulkumaran et al., 2017) , in part because they require (1) reasoning in a partially observable environment (2) reasoning with complex observations, e.g. visually rich images.

For example, a robot navigating in a new environment has to (1) localize and plan a path having only partial information of the environment (2) extract the traversable space from image pixels, where the relevant geometric features are tightly coupled with irrelevant visual features, such as wall textures and lighting.

Decision making under partial observability can be formulated as a partially observable Markov decision process (POMDP).

Solving POMDPs requires tracking the posterior distribution of the states, called the belief.

Most POMDP RL methods track the belief, represented as a vector, using a recurrent neural network (RNN) (Hausknecht & Stone, 2015; Zhu et al., 2018) .

RNNs are model-free generic function approximators, and without appropriate structural priors they may need large amounts of data to learn to track a complex belief.

Model-based DRL methods aim to reduce the sample complexity by learning an environment model simultaneously with the policy.

In particular, to deal with partial observability, recently proposed DVRL that learns a generative observation model incorporated into the policy through a Bayes filter.

Because the Bayes filter tracks the belief explicitly, DVRL performs much better than generic RNNs under partial observability.

However, a Bayes filter normally assumes a generative observation model, that defines the probability p(o | h t ) of receiving an observation o = o t given the history h t of past observations and actions (Fig. 1b ).

Learning this model can be very challenging since the strong generative assumption requires modeling the whole observation space, including features irrelevant for RL.

When o t is an image, p(o | h t ) is a distribution over all possible images, e.g., parameterized by independent pixel-wise Gaussians with learned mean and variance.

This means, e.g., to navigate in a previously unseen environment, we need to learn the < l a t e x i t s h a 1 _ b a s e 6 4 = " r o 4 2 6 D H M J G U u G 8 x G G s K s G W z i 0 / s = " > A A A B 7 X i c b Z B N S w M x E I Z n 6 1 e t X 1 W P X o J F q J e y K 4 I e i 1 4 8 V r D b Q r u U b J p t Y 7 P Z J Z k V S u l / 8 O J B E a / + H 2 / + G 9 N 2 D 9 r 6 Q u D h n R k y 8 4 a p F A Z d 9 9 s p r K 1 v b G 4 V t 0 s 7 u 3 v 7 B + X D I 9 8 k m W a 8 y R K Z 6 H Z I D Z d C 8 S Y K l L y d a k 7 j U P J W O L q d 1 V t P X B u R q A c c p z y I 6 U C J S D C K 1 v L 9 a t j D 8 1 6 5 4 t b c u c g q e D l U I F e j V / 7 q 9 h O W x V w h k 9 S Y j u e m G E y o R s E k n 5 a 6 m e E p Z S M 6 4 B 2 L i s b c B J P 5 t l N y Z p 0 + i R J t n 0 I y d 3 9 P T G h s z D g O b W d M c W i W a z P z v 1 o n w + g 6 m A i V Z s g V W 3 w U Z Z J g Q m a n k 7 7 Q n K E c W 6 B M C 7 s r Y U O q K U M b U M m G 4 C 2 f v A r + R c 2 z f H 9 Z q d / k c R T h B E 6 h C h 5 c Q R 3 u o A F N Y P A I z / A K b 0 7 i v D j v z s e i t e D k M 8 f w R 8 7 n D 7 8 m j p I = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " r o 4 2 6 D H M J G U u G 8 x G G s K s G W z i 0 / s = " > A A A B 7 X i c b Z B N S w M x E I Z n 6 1 e t X 1 W P X o J F q J e y K 4 I e i 1 4 8 V r D b Q r u U b J p t Y 7 P Z J Z k V S u l / 8 O J B E a / + H 2 / + G 9 N 2 D 9 r 6 Q u D h n R k y 8 4 a p F A Z d 9 9 s p r K 1 v b G 4 V t 0 s 7 u 3 v 7 B + X D I 9 8 k m W a 8 y R K Z 6 H Z I D Z d C 8 S Y K l L y d a k 7 j U P J W O L q d 1 V t P X B u R q A c c p z y I 6 U C J S D C K 1 v L 9 a t j D 8 1 6 5 4 t b c u c g q e D l U I F e j V / 7 q 9 h O W x V w h k 9 S Y j u e m G E y o R s E k n 5 a 6 m e E p Z S M 6 4 B 2 L i s b c B J P 5 t l N y Z p 0 + i R J t n 0 I y d 3 9 P T G h s z D g O b W d M c W i W a z P z v 1 o n w + g 6 m A i V Z s g V W 3 w U Z Z J g Q m a n k 7 7 Q n K E c W 6 B M C 7 s r Y U O q K U M b U M m G 4 C 2 f v A r + R c 2 z f H 9 Z q d / k c R T h B E 6 h C h 5 c Q R 3 u o A F N Y P A I z / A K b 0 7 i v D j v z s e i t e D k M 8 f w R 8 7 n D 7 8 m j p I = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " r o 4 2 6 D H M J G U u G 8 x G G s K s G W z i 0 / s = " > A A A B 7 X i c b Z B N S w M x E I Z n 6 1 e t X 1 W P X o J F q J e y K 4 I e i 1 4 8 V r D b Q r u U b J p t Y 7 P Z J Z k V S u l / 8 O J B E a / + H 2 / + G 9 N 2 D 9 r 6 Q u D h n R k y 8 4 a p F A Z d 9 9 s p r K 1 v b G 4 V t 0 s 7 u 3 v 7 B + X D I 9 8 k m W a 8 y R K Z 6 H Z I D Z d C 8 S Y K l L y d a k 7 j U P J W O L q d 1 V t P X B u R q A c c p z y I 6 U C J S D C K 1 v L 9 a t j D 8 1 6 5 4 t b c u c g q e D l U I F e j V / 7 q 9 h O W x V w h k 9 S Y j u e m G E y o R s E k n 5 a 6 m e E p Z S M 6 4 B 2 L i s b c B J P 5 t l N y Z p 0 + i R J t n 0 I y d 3 9 P T G h s z D g O b W d M c W i W a z P z v 1 o n w + g 6 m A i V Z s g V W 3 w U Z Z J g Q m a n k 7 7 Q n K E c W 6 B M C 7 s r Y U O q K U M b U M m G 4 C 2 f v A r + R c 2 z f H 9 Z q d / k c R T h B E 6 h C h 5 c Q R 3 u o A F N Y P A I z / A K b 0 7 i v D j v z s e i t e D k M 8 f w R 8 7 n D 7 8 m j p I = < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " r o 4 2 6 D H M J G U u G 8 x G G s K s G W z i 0 / s = " > A A A B 7 X i c b Z B N S w M x E I Z n 6 1 e t X 1 W P X o J F q J e y K 4 I e i 1 4 8 V r D b Q r u U b J p t Y 7 P Z J Z k V S u l / 8 O J B E a / + H 2 / + G 9 N 2 D 9 r 6 Q u D h n R k y 8 4 a p F A Z d 9 9 s p r K 1 v b G 4 V t 0 s 7 u 3 v 7 B + X D I 9 8 k m W a 8 y R K Z 6 H Z I D Z d C 8 S Y K l L y d a k 7 j U P J W O L q d 1 V t P X B u R q A c c p z y I 6 U C J S D C K 1 v L 9 a t j D 8 1 6 5 4 t b c u c g q e D l U I F e j V / 7 q 9 h O W x V w h k 9 S Y j u e m G E y o R s E k n 5 a 6 m e E p Z S M 6 4 B 2 L i s b c B J P 5 t l N y Z p 0 + i R J t n 0 I y d 3 9 P T G h s z D g O b W d M c W i W a z P z v 1 o n w + g 6 m A i V Z s g V W 3 w U Z Z J g Q m a n k 7 7 Q n K E c W 6 B M C 7 s r Y U O q K U M b U M m G 4 C 2 f v A r + R c 2 z f H 9 Z q d / k c R T h B E 6 h C h 5 c Q R 3 u o A F N Y P A I z / A K b 0 7 i v D j v z s e i t e D k M 8 f w R 8 7 n D 7 8 m j p I = < / l a t e x i t > ???(b t ) < l a t e x i t s h a 1 _ b a s e 6 4 = " d M D G r E g / u U t R q i s D I A Z M / + 5 R W i g = " > A A A B 7 3 i c b Z B N S 8 N A E I Y n 9 a v W r 6 p H L 4 t F q J e S i K D H o h e P F e w H t K F s t p t 2 6 W Y T d y d C C f 0 T X j w o 4 t W / 4 8 1 / 4 7 b N Q V t f W H h 4 Z 4 a d e Y N E C o O u + + 0 U 1 t Y 3 N r e K 2 6 W d 3 b 3 9 g / L h U c v E q W a 8 y W I Z 6 0 5 A D Z d C 8 S Y K l L y T a E 6 j Q P J 2 M L 6 d 1 d t P X B s R q w e c J N y P 6 F C J U D C K 1 u r 0 E l E N + n j e L 1 f c m j s X W Q U v h w r k a v T L X 7 1 B z N K I K 2 S S G t P 1 3 A T 9 j G o U T P J p q Z c a n l A 2 p k P e t a h o x I 2 f z f e d k j P r D E g Y a / s U k r n 7 e y K j

0 X l x 3 p 2 P R W v B y W e O 4 Y + c z x 9 g o Y + F < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " d M D G r E g / u U t R q i s D I A Z M / + 5 R W i g = " > A A A B 7 3 i c b Z B N S 8 N A E I Y n 9 a v W r 6 p H L 4 t F q J e S i K D H o h e P F e w H t K F s t p t 2 6 W Y T d y d C C f 0 T X j w o 4 t W / 4 8 1 / 4 7 b N Q V t f W H h 4 Z 4 a d e Y N E C o O u + + 0 U 1 t Y 3 N r e K 2 6 W d 3 b 3 9 g / L h U c v E q W a 8 y W I Z 6 0 5 A D Z d C 8 S Y K l L y T a E 6 j Q P J 2 M L 6 d 1 d t P X B s R q w e c J N y P 6 F C J U D C K 1 u r 0 E l E N + n j e L 1 f c m j s X W Q U v h w r k a v T L X 7 1 B z N K I K 2 S S G t P 1 3 A T 9 j G o U T P J p q Z c a n l A 2 p k P e t a h o x I 2 f z f e d k j P r D E g Y a / s U k r n 7 e y K j

0 X l x 3 p 2 P R W v B y W e O 4 Y + c z x 9 g o Y + F < / l a t e x i t > < l a t e x i t s h a 1 _ b a s e 6 4 = " d M D G r E g / u U t R q i s D I A Z M / + 5 R W i g = " > A A A B 7 3 i c b Z B N S 8 N A E I Y n 9 a v W r 6 p H L 4 t F q J e S i K D H o h e P F e w H t K F s t p t 2 6 W Y T d y d C C f 0 T X j w o 4 t W / 4 8 1 / 4 7 b N Q V t f W H h 4 Z 4 a d e Y N E C o O u + + 0 U 1 t Y 3 N r e K 2 6 W d 3 b 3 9 g / L h U c v E q W a 8 y W I Z 6 0 5 A D Z d C 8 S Y K l L y T a E 6 j Q P J 2 M L 6 d 1 d t P X B s R q w e c J N y P 6 F C J U D C K 1 u r 0 E l E N + n j e L 1 f c m j s X W Q U v h w r k a v T L X 7 1 B z N K I K 2 S S G t P 1 3 A T 9 j G o U T P J p q Z c a n l A 2 p k P e t a h o x I 2 f z f e d k j P r D E g Y a / s U k r n 7 e y K j l L y T a E 6 j Q P J 2 M L 6 d 1 d t P X B s R q w e c J N y P 6 F C J U D C K 1 u r 0 E l E N + n j e L 1 f c m j s X W Q U v h w r k a v T L X 7 1 B z N K I K 2 S S G t P 1 3 A T 9 j G o U T P J p q Z c a n l A 2 p k P e t a h o x I 2 f z f e d k j P r D E g Y a / s U k r n 7 e y K j distribution of all possible environments with their visual appearance, lighting condition, etc.

-a much harder task than learning to extract features relevant to navigation, e.g. the traversable space.

We introduce the Discriminative Particle Filter Reinforcement Learning (DPFRL), a POMDP RL method that learns to explicitly track a latent belief, while circumventing the difficulty of generative observation modeling, and learns to make decisions based on features of the latent belief (Fig. 1a) .

DPFRL approximates the belief by a set of weighted learnable latent particles {(h

, and it tracks the particle belief by a non-parametric Bayes filter algorithm, a particle filter, encoded as a differentiable computational graph in the neural network architecture.

Transition and observation models for the particle filter are neural networks learned jointly end-to-end, optimized for the overall policy.

Importantly, we use a discriminatively parameterized observation model, f obs (o t , h t ), a neural network that takes in o t and h t and outputs a single value, a direct estimate of the log-likelihood as shown in Fig. 1c .

The discriminative parameterization relaxes the generative assumption and avoids explicitly modeling the entire complex observation space when computing observation likelihood.

The intuition is similar to that of, e.g., energy-based models (LeCun et al., 2006) and contrastive predictive coding (Oord et al., 2018) , but here the learning signal comes directly from the RL objective, backpropagating through the differentiable particle filter, thus f obs (o t , h t ) only needs to model the observation features relevant to decision making.

In addition, to summarize the particle belief, we introduce novel learnable features based on Moment-Generating Functions (MGFs) (Bulmer, 1979) .

MGF features are computationally efficient and permutation invariant, and they can be directly optimized to provide useful higher-order moment information for learning the policy.

MGF features could be also used as learned features of any empirical distribution in application beyond RL.

We evaluate DPFRL on a range of POMDP RL domains: a continuous control task from , Flickering Atari Games (Hausknecht & Stone, 2015) , Natural Flickering Atari Games, a new domain with more complex observations that we introduce, and the Habitat visual navigation domain using real-world data (Savva et al., 2019) .

DPFRL outperforms state-of-the-art POMDP RL methods in most cases.

Results show that the particle filter structure is effective for handling partial observations, and the discriminative parameterization allows for complex observations.

We summarize our contributions as follows: 1) a differentiable particle filter based method with a discriminatively parameterized observation model for RL with partial and complex observations.

2) effective MGF features for empirical distributions, e.g., particle distributions 3) a new RL benchmark, Natural Flickering Atari Games, that introduces both partial observability and complex visual observations to the popular Atari domain.

We will open source the code to enable future work.

Real-world decision-making problems are often formulated as POMDPs given the partial observations.

POMDPs are notoriously hard to solve; in the worst case, they are computationally intractable (Papadimitriou & Tsitsiklis, 1987) .

Approximate POMDP solvers have made dramatic progress in solving large-scale POMDPs (Kurniawati et al., 2008) .

Particle filters have been widely adopted as belief tracker for POMDP solvers (Silver & Veness, 2010; Somani et al., 2013) with the flexibility to model complex and multi-modal distributions, compared to Gaussian and Kalman filters.

However, a predefined model and state representations are required for these methods (see e.g. Bai et al. (2015) ).

Given the advance in generative neural network models, various neural models (Chung et al., 2015; Maddison et al., 2017; Naesseth et al., 2018) have been proposed for belief tracking.

DVRL uses Variational Sequential Monte-Carlo method (Naesseth et al., 2018) ,

are maintained by a differentiable discriminative particle filter algorithm, which includes transition function ftrans, discriminative observation function f obs , and a differentiable soft-resampling.

The policy and value function is conditioned on the belief, which is summarized by the mean particleht, and m moment generating function features, M 1:m t .

which is similar to the particle filters that we use, for belief tracking in RL.

This gives better belief tracking capabilities, but as we demonstrate in our experiments, generative modeling is not robust in complex observation spaces with high-dimensional irrelevant features.

More powerful generative models, e.g., DRAW (Gregor et al., 2015) , could be considered to improve generative observation modeling; however, evaluating a complex generative model for each particle would significantly increase the computational cost and optimization difficulty.

Learning a robust latent representation and avoiding reconstructing observations are of great interest for RL (Oord et al., 2018; Guo et al., 2018; Hung et al., 2018; Gregor et al., 2019; Gelada et al., 2019) .

Discriminative RNNs have also been widely used for belief approximation in partially observable domains (Bakker, 2002; Wierstra et al., 2007; Foerster et al., 2016) .

The latent representation is directly optimized for the policy p(a|h t ) that skips observation modeling.

Hausknecht & Stone (2015) ; Zhu et al. (2018) tackle the partially observable Flickering Atari Games by extending DQN (Mnih et al., 2013) with an LSTM memory.

Our experiments demonstrate that the additional structure provided by using a particle filter to track beliefs can give improved performance in RL.

Embedding algorithms into neural networks to allow end-to-end discriminative training has gained attention recently.

For belief tracking, the idea has been used in the differentiable histogram filter (Jonschkowski & Brock, 2016) , Kalman filter (Haarnoja et al., 2016) and particle filter (Karkus et al., 2018) .

Karkus et al. (2017) combined a learnable histogram filter with the Value Iteration Network (Tamar et al., 2016) and introduced the learnable POMDP planner, QMDP-net.

However, these methods require a predefined state representation and are limited to relatively small state spaces.

Ma et al. (2019) integrated the particle filter with standard RNNs, e.g., LSTM, and introduced a discriminative PF-RNNs for sequence prediction.

We build on the work in Ma et al. (2019) demonstrating its advantages for RL with complex partial observations and introducing MGF features to the method for improved decision making from particle beliefs.

The proposed discriminative reinforcement framework and algorithms can be applied to most of the discriminative particle filters.

We introduce DPFRL, a framework for reinforcement learning under partial and complex observations.

The DPFRL architecture is shown in Fig. 2 .

It has two main components, a discriminative particle filter that tracks a latent belief b t , and an actor network that learns a policy p(a | b t ) given belief b t .

A pseudocode for the DPFRL algorithm is given by Alg.

1 in the Appendix.

State Representation.

In DPFRL, we use a fully differentiable particle filter algorithm to maintain a belief state b t .

More specifically, we approximate the belief state with a set of weighted latent particles

are K latent states learned by policy-oriented training, and {w

represents the corresponding weights.

Each latent state h i t stands for a hypothesis in the belief; the set of latent particles approximates the statistics of the belief.

Belief Update.

We update the latent particles according to particle filter algorithm

Under review as a conference paper at ICLR 2020

Transition update.

Eqn.

1 implements the transition update step.

We sample the next state h

, where a t is the agent action and u t (o t ) parameterized function that can learn the environment dynamics.

This formulation assumes a fully controlled system where a learned latent state that captures the dynamics of both the agent and the environment.

Similar formulation has been used for sequence prediction (Ma et al., 2019) .

In our experiments, f trans is implemented by a gated function following PF-GRU (Ma et al., 2019) ) for all i = 1 : K and normalize using ??.

Gradients for training f obs are obtained by backpropagating the standard RL loss through the belief-conditional policy p(a | b t ), the update steps of the particle filter algorithm, and the particle weight update in Eqn.

2.

We expect the discriminatively parameterized f obs to be more effective than a standard generative model for the following reasons.

A generative model aims to approximate p(o | h) by learning a function that takes h as input and outputs parameterized distribution over o, for example, pixel-wise Gaussians with learned mean and variance.

However, the generative assumption forces the model to consider the entire observation space equally, including features irrelevant for RL.

In contrast, f obs has a much simpler functional form that takes o and h as inputs.

It is also more flexible and does not need to form a parametric distribution over o, which relaxes the strong generative assumption.

When trained end-to-end for RL, f obs learns to model only the features relevant to RL.

Since f obs is only used for particle filtering, it only needs to learn an unnormalized likelihood value that is proportionate to p(o t | h i t ) when evaluated for different h i t , i = 1 : K. This can be a much simpler function to learn than the full generative distribution, especially for complex observations, as we avoid modeling features that are irrelevant for belief tracking and decision making.

In implementation we compute weight updates in log-space for better numerical stability.

We use a simple fully connected layer for f obs that takes in the concatenated o t and h i t , and outputs a single value interpreted as the observation log-likelihood.

Note that more complex network architectures could be considered to further improve the capability of f obs , which we leave to future work.

Differentiable particle resampling.

To avoid particle degeneracy, i.e., most of the particles having near-zero weight, we adopt the differentiable soft-resampling strategy (Karkus et al., 2018; Ma et al., 2019) .

Instead of sampling from p t (i) = w i t , we sample particles {h .

We can have the final particle belief

).

As a result, f obs can be optimized with global belief information and model shared useful features across multiple time steps.

Another concern is that the particle distribution may collapse to particles with the same latent state.

This can be avoided by ensuring that the stochastic transition function f trans has a non-zero variance.

Conditioning a policy directly onto a particle belief distribution is non-trivial.

To feed it to the networks, we need to summarize it into a single vector.

We introduce a novel feature extraction method for empirical distributions based on MomentGenerating Functions (MGFs).

The MGF of an n-dimensional random variable X is given by

n .

In statistics, MGF is an alternative specification of its probability distribution (Bulmer, 1979) .

Since particle belief distribution b t is an empirical distribution, the moment generating function of b t can be denoted as M bt (v) = We summarize the belief distribution with

i t is the mean particle.

The mean particleh t , as the first-order moment, and m additional MGF features, give a summary of the belief distribution characteristics.

The number of MGF features, m, controls how much additional information we extract from the belief.

We empirically study the influence of MGF features in ablation studies.

Compared to Ma et al. (2019) that uses the mean as the belief estimate, MGF features provide additional features from the empirical distribution.

Compared to DVRL ) that treats the Monte-Carlo samples as a sequence and merges them by an RNN, MGF features are permutationinvariant, computationally efficient and easy to optimize, especially when the particle set is large.

, we compute the policy p(a | b t ) with a policy network ??(b t ).

In actor-critic setup, a value network V (b t ) is introduced to assist learning.

In our experiment, we evaluated on the A2C algorithm (Mnih et al., 2016) .

We evaluate DPFRL in a range of POMDP RL domains with increasing belief tracking and observation modeling complexity.

We first use benchmark domains from the literature, Mountain Hike, and 10 different Flickering Atari Games.

We then introduce a new, more challenging domain, Natural Flickering Atari Games, that uses a random video stream as the background.

Finally we apply DPFRL to a challenging visual navigation domain with RGB-D observations rendered from real-world data.

We compare DPFRL with a GRU network, a state-of-the-art POMDP RL method, DVRL, and ablations of the DPFRL architecture.

As a brief conclusion, we show that: 1) DPFRL significantly outperforms GRU in most cases because of its explicit structure for belief tracking; 2) DPFRL outperforms the state-of-the-art DVRL in most cases even with simple observations, and its benefit increases dramatically with more complex observations because of DPFRL's discriminative observation model; 3) MGF features are more effective for summarizing the latent particle belief than alternatives.

An agent navigates on the map from the start position (white dot) to the goal (green dot with the shaded area as the threshold).

Partial observation is introduced by a Gaussian noise and appended with a long noise vector of length l.

The reward r(x, y) for position (x, y) is given by the heat map.

We plot the accumulated rewards and all reported results are averaged over 3 different random seeds.

The curves are smoothed over time and averaged over parallel environment executions.

To be comparable with the GRU and DVRL baselines we train DPFRL with the same A2C algorithm and use a similar network architecture and hyperparameters as the original DVRL implementation.

DPFRL and DVRL differ in the particle belief update structure, but they use the same latent particle size dim(h) and the same number of particles K as in the DVRL paper (dim(h) = 128 and K = 30 for Mountain Hike, dim(h) = 256 and K = 15 for Atari games and visual navigation).

The effect of the number of particles is discussed in Sect.

4.5.

We train all models for the same number of iterations using the RMSProp optimizer (Tieleman & Hinton, 2012) .

Learning rates and gradient clipping values are chosen based on a search in the BeamRider Atari game independently for each model.

Further details are in the Appendix.

We have not performed additional searches for the network architecture and other hyper-parameters, nor tried other RL algorithm, such as PPO (Schulman et al., 2017) , which may all improve our results.

Mountain Hike has been introduced by to demonstrate the benefit of belief tracking for POMDP RL.

It is a continuous control problem where an agent navigates on a fixed 20 ?? 20 map.

In the original observation, partial observability is introduced by disturbing the agent observation with an additive Gaussian noise.

To illustrate the effect of observation complexity in natural environments, we concatenate the original observation vector with a random noise vector.

The complexity of the optimal policy remains unchanged, but the relevant information is now coupled with irrelevant observation features.

More specifically, the state space and action space in Mountain Hike are defined l is sampled from a uniform distribution U(???10, 10).

The reward for each step is given by r t = r(x t , y t ) ??? 0.01||a t || where r(x t , y t ) is shown in Fig. 3 .

Episodes end after 75 steps.

We train models for different settings of the noise vector length l, from l = 0 to l = 100.

Results are shown in Fig. 4 .

We observe that DPFRL learns faster than the DVRL and GRU in all cases, including the original setting l = 0.

Importantly, as the noise vector length increases, the performance of DVRL and GRU degrades, while DPFRL is unaffected.

This demonstrates the ability of DPFRL to track a latent belief without having to explicitly model complex observations.

Atari games are one of the most popular benchmark domains for RL methods (Mnih et al., 2013) .

Their partially observable variants, Flickering Atari Games, have been used to benchmark POMDP RL methods (Hausknecht & Stone, 2015; Zhu et al., 2018; .

Here image observations are single frames randomly replaced by a blank frame with a probability of 0.5.

The flickering observations introduce a simple form of partial observability.

Another variant, Natural Atari Games (Zhang et al., 2018) , replaces the simple black background of the frames of an Atari game with a randomly sampled video stream.

This modification brings the Atari domain one step closer to the visually rich real-world, in that the relevant information is now encoded in complex observations.

As shown by Zhang et al. (2018) , this poses a significant challenge for RL.

We propose a new RL domain, Natural Flickering Atari Games, that involves both challenges: partial observability simulated by flickering frames, and complex observations simulated by random background videos.

The background videos increase observation complexity without affecting the decision making complexity, making this a suitable domain for evaluating RL methods with complex observations.

We sample the background video from the ILSVRC dataset (Russakovsky et al., 2015) .

Examples for the BeamRider game are shown in Fig. 5 .

Details are in the Appendix.

We evaluate DPFRL for both Flickering Atari Games and Natural Flickering Atari Games.

We use the same set of games as .

To ensure a fair comparison, we take the GRU and DVRL results from the paper for Flickering Atari Games, use the same training iterations as in , and we use the official DVRL open source code to train for Natural Flickering Atari Games.

Results are summarized in Table 1 .

We highlight the best performance in bold where the difference is statistically significant (p = 0.05).

Detailed training curves are in the Appendix.

GRU We observe that DPFRL significantly outperforms GRU in almost all games, which indicates the importance of explicit belief tracking, and shows that DPFRL can learn a useful latent belief representation.

Despite the simpler observations, DPFRL significantly outperforms DVRL and achieves state-of-the-art results on 5 out of 10 standard Flickering Atari Games (ChopperCommand, MsPacman, BeamRider, Bowling, Asteroids), and it performs comparably in 3 other games (Centipede, Frostbite, IceHockey) .

The strength of DFPRL shows even more clearly in the Natural Flickering Atari Games, where it significantly outperforms DVRL on 7 out of 10 games and performs similarly in the rest.

In some games, e.g. in Pong, DPFRL performs similarly with and without videos in the background (15.65 vs. 15.40), while the DVRL performance degrades substantially (-19.78 vs. 18.17) .

These results show that while the architecture of DPFRL and DVRL are similar, the policy-oriented discriminative observation model of DPFRL is much more effective for handling complex observations, and the MGF features provide a more powerful summary of the particle belief for decision making.

However, on some games, e.g. ChooperCommand, even DPFRL performance drops significantly when adding background videos.

This shows that irrelevant features can make the task much harder, even for a discriminative approach, as also observed in in Zhang et al. (2018) . (Savva et al., 2019) 0.70 0.80 -Visual navigation poses a great challenge for deep RL Mirowski et al. (2016); Zhu et al. (2017); Lample & Chaplot (2017) .

We evaluate DPFRL for visual navigation in the Habitat Environment (Savva et al., 2019) , using the real-world Gibson dataset (Xia et al., 2018) .

In this domain, a robot needs to navigate to goals in previously unseen environments.

In each time step, it receives a first-person RGB-D camera image, and its distance and relative orientation to the goal.

The main challenge lies in the partial and complex observations: first-person view images only provide partial information about the unknown environment; and the relevant information for navigation, traversability, is encoded in rich RGB-D observations along with many irrelevant features, e.g., the texture of the wall.

We use the Gibson dataset with the training and validation split provided by the Habitat challenge.

We train models with the same architecture as for the Atari games, except for the observation encoder that accounts for the different observation formats.

We evaluate models in unseen environments from the validation split and compute the same set of metrics as in the literature, SPL and Success Rate, as well as average rewards.

Results are shown in Table 2 .

Further details and results are in the Appendix.

DPFRL significantly outperforms both DVRL and GRU in this challenging domain.

DVRL performs especially poorly, demonstrating the difficulty of learning a generative observation model in realistic, visually rich domains.

DPFRL also outperforms the PPO baseline from Savva et al. (2019) .

We note that submissions to the recently organized Habitat Challenge 2019 (Savva et al., 2019) , such as (Chaplot et al., 2019) , have demonstrated better performance than the PPO baseline (while our results are not directly comparable because of the closed test set of the competition).

However, these approaches rely on highly specialized structures, such as 2D mapping and 2D path planning, while we use the same generic network as for Atari games.

Future work may further improve our results by adding a task-specific structure to DPFRL or training with PPO instead of A2C.

We conduct an extensive ablation study on the Natural Flickering Atari Games to understand the influence of each DPFRL component.

The results are presented in Table 3 .

Discriminative parameterization is more effective than generative parameterization.

DPFRLgenerative replaces the discriminative observation function of DPFRL with a generative observation function, where grayscale image observations are modeled by pixel-wise Gaussian distributions with learned mean and variance.

Unlike DVRL, DPFRL-generative only differs from DPFRL in the parameterization of the observation function, the rest of the architecture and training loss remains the same.

In most cases, the performance for DPFRL-generative degrades significantly compared to DPFRL.

These results are aligned with our earlier observations and indicate that the discriminative parameterization is indeed important to extract the relevant information from complex observations without having to learn a more complex generative model.

More particles perform better.

DPFRL with 1 particle performs poorly on most of the tasks (DPFRLL-P1).

This indicates that a single latent state is insufficient to represent a complex latent distribution that is required for the task, and that more particles can be expected to improve performance.

MGF features are useful.

We compare DPFRL using MGF features with DPFRL-mean that only uses the mean particle, and with DPFRL-GRUmerge that uses a separate RNN to summarize the belief similar to DVRL.

Results show that DPFRL-mean does not work as well as the standard DPFRL, especially for tasks that may need complex belief tracking, e.g., Pong.

This can be attributed to the more rich belief statistics provided by MGF features, and that they do not constrain the learned belief representation to be always meaningful when averaged.

Comparing to DPFRL-GRUmerge shows that MGF features generally perform better.

While an RNN may learn to extract useful features from the latent belief, optimizing the RNN parameters is harder, because they are not permutation invariant to the set of particles and they result in a long backpropagation chain.

We have introduced DPFRL, a principled framework for POMDP RL in natural environments.

DPFRL combines the strength of Bayesian filtering and end-to-end RL: it performs explicit belief tracking with discriminative learnable particle filters optimized directly for the RL policy.

DPFRL achieved state-of-the-art results on POMDP RL benchmarks from prior work, Mountain Hike and a number of Flickering Atari Games, and it significantly outperformed alternative methods in a new, more challenging domain, Natural Flickering Atari Games, as well as for visual navigation using real-world data.

We have proposed a novel MGF feature for extracting statistics from an empirical distribution.

MGF feature extraction could be applied beyond RL, e.g., for general sequence prediction.

DPFRL does not perform well in some particular cases, e.g., DoubleDunk.

While our discriminatively parameterized observation function is less susceptible to observation noise, it does not allow for additional learning signals that improve sample efficiency, e.g., through a reconstruction loss.

Future work may combine generative and discriminative modeling with the principled DPFRL framework.

Particle filter is an approximate Bayes filter algorithm for belief tracking.

Bayes filters estimate the belief b t , i.e., a posterior distribution of the state s t , given the history of actions a 1:t and observations o 1:t .

Instead of explicitly modeling the posterior distribution, particle filter approximates the posterior with a set of weighted particles,

, and update the particles in a Bayesian manner.

Importantly, the particle set could approximate arbitrary distributions, e.g., Gaussians, continuous

where ?? is a normalization factor and p(o t | s Resampling.

The particle filter algorithm can suffer from particle degeneracy, where after some update steps only a few particles have non-zero weights.

This would prevent particle filter to approximate the posterior distribution effectively.

Particle degeneracy is typically addressed by performing resampling, where new particles are sampled with repetition proportional to its weight.

Specifically, we sample particles from a categorical distribution p parameterized by the particle weights {w

where p(i) is the probability for the i-th category, i.e., the i-th particle.

The new particles approximate the same distribution, but they assign more representation capacity to the relevant regions of the state space.

In probability theory, the moment-generating function (MGF) is an alternative specification of the probability distribution of a real-valued random variable (Bulmer, 1979) .

As its name suggests, MGF of a random variable could be used to generate any order of its moments, which characterize its probability distribution.

Mathematically, the MGF of a random variable X with dimension m is defined by M X (v) = E e vX (7) where v ??? R m and we could consider the MGF of random variable X is the expectation of the random variable e vX .

Consider the series expansion of e

Thus we have

To compute the j-th order moment M j , we can compute the j-th order derivative and set v to 0.

In DPFRL, we use MGFs as additional features to provide moment information of the particle distribution.

DPFRL learns to extract useful moment features for decision making by directly optimizing for policy p(a | b t ).

Observation Encoders:

For the observation encoders, we used the same structure with DVRL for a fair comparison.

For Mountain Hike, we use two fully connected layers with batch normalization and ReLU activation as the encoder.

The dimension for both layers is 64.

For the rest of the domains, we first down-sample the image size to 84??84, then we process images with 3 2D-convolution layers with channel number (32, 64, 32) , kernel sizes (8, 4, 3) and stride (4, 2, 1), without padding.

The compass and goal information are a vector of length 2; they are appended after the image encoding as the input.

Observation Decoders: Both DVRL and PFGRU-generative need observation decoders.

For the Mountain Hike, we use the same structure as the encoder with a reversed order.

The transposed 2D-convolutional network of the decoder has a reversed structure.

The decoder is processed by an additional fully connected layer which outputs the required dimension (1568 for Atari and Habitat Navigation, both of which have 84 ?? 84 observations).

We directly use the transition function in PF-GRU (Ma et al., 2019) for

, which is a stochastic function with GRU gated structure.

Action a t is first encoded by a fully connected layer with batch normalization and ReLU activation.

The encoding dimension for Mountain Hike is 64 and 128 for all the rest tasks.

The mean and variance of the normal distribution are learned again by two additional fully connected layers; for the variance, we use Softplus as the activation function.

Observation Function: f obs is implemented by a single fully connected layer without activation.

In DVRL, the observation function is parameterized over the full observation space o and p(o | h i t???1 , a i t ) is assumed as a multivariate independent Bernoulli distribution whose parameters are again determined by a neural network .

For numerical stability, all the probabilities are stored and computed in the log space and the particle weights are always normalized after each weight update.

The soft-resampling hyperparameter ?? is set to be 0.9 for Mountain Hike and 0.5 for the rest of the domains.

Note that the soft-resampling is used only for DPFRL, not including DVRL.

DVRL averages the particle weights to 1/K after each resampling step, which makes the resampling step cannot be trained by the RL.

Belief Summary: The GRU used in DVRL and DPFRL-GRUmerge is a single layer GRU with input dimension equals the dimension of the latent vector plus 1, which is the corresponding particle weight.

The dimension of this GRU is exactly the dimension of the latent vector.

For the MGF features, we use fully connected layers with feature dimensions as the number of MGF features.

The activation function used is the exponential function.

We could potentially explore the other activation functions to test the generalized-MGF features, e.g., ReLU.

Model Learning: For RL, we use an A2C algorithm with 16 parallel environments for both Mountain Hike and Atari games; for Habitat Navigation, we only use 6 parallel environments due to the GPU memory constraints.

The loss function for DPFRL and GRU-based policy is just the standard A2C loss,

We follow the default setting provided by and set ?? E = 0.1.

The rest of the hyperparameters, including learning rate, gradient clipping value and ?? in soft-resampling are tuned according to the BeamRider and directly applied to all domains due to the highly expensive experiment setups.

The learning rate for all the networks are searched among the following values: (3 ?? 10 ???5 , 5 ?? 10 ???5 , 1 ?? 10 ???4 , 2 ?? 10 ???4 , 3 ?? 10 ???4 ); the gradient clipping value are searched among {0.5, 1.0}; the soft-resampling ?? is searched among {0.5, 0.9}. The best performing learning rates were 1 ???4 for DPFRL and GRU, and 2 ???4 for DVRL; the gradient clipping value for all models was 0.5; the soft-resampling ?? is set to be 0.9 for Mountain Hike and 0.5 for Atari games.

Natural Flickering Atari games We follow the setting of the prior works (Zhu et al., 2018; : 1)

50% of the frames are randomly dropped 2) a frameskip of 4 is used 3) there is a 0.25 chance of repeating an action twice.

In our experiments, we sample background videos from the ILSVRC dataset (Russakovsky et al., 2015) .

Only the videos with the length longer than 500 frames are sampled to make sure the video length is long enough to introduce variability.

For each new episode, we first sample a new video from the dataset, and a random starting pointer is sampled in this video.

Once the video finishes, the pointer is reset to the first frame (not the starting pointer we sampled) and continues from there.

We implement all the models using PyTorch (Paszke et al., 2017) with CUDA 9.2 and CuDNN 7.1.2.

Flickering Atari environments are modified based on OpenAI Gym (Brockman et al., 2016) and we directly use Habitat APIs for visual navigation.

For Mountain Hike and Atari games, we run our experiments on servers with 4 NVidia GTX1080Ti GPUs on each server.

For Habitat visual navigation, we run the experiment on servers with 4 NVidia RTX2080Ti GPUs on each server.

We implement DPFRL with gated transition and observation functions for particle filtering similar to PF-GRU (Ma et al., 2019) .

In standard GRU, the memory update is implemented by a gated function:

where W n and b n are the corresponding weights and biases, and z t r t are the learned gates.

PF-GRU introduces stochastic cell update by assuming the update to the memory, n We present the reward curve for the Habitat visual navigation task below.

DPFRL outperforms both GRU-based policy and DVRL given the same training time.

DVRL struggles with training the observation model and fails during the first half of the training time.

GRU based policy learns fast; given only the model-free belief tracker, it struggles to achieve higher reward after a certain point.

We only provide the reward curve here as SPL and success rate are only evaluated after the training is finished.

We further visualize the latent particles by principal component analysis (PCA) and choose the first 2 components.

We choose a trajectory in the Habitat Visual Navigation experiment, where 15 particles are used.

We observe that particles initially spread across the space (t = 0).

As the robot only receive partial information in the visual navigation task, particles gradually form a distribution with two clusters (t = 56), which represent two major hypotheses of its current state.

After more information is incorporated into the belief, they begin to converge and finally become a single cluster (t = 81).

We did not observe particle depletion and posterior collapse in our experiment.

This could be better avoided by adding an entropy loss to the learned variance of f trans and we will leave it for future study.

@highlight

We introduce DPFRL, a framework for reinforcement learning under partial and complex observations with a fully differentiable discriminative particle filter

@highlight

Introduces ideas for training DLR agents with latent state variables, modeled as a belief distribution, so they can handle partially observed environments.

@highlight

This paper introduces a principled method for POMDP RL: Discriminative Particle Filter Reinforcement Learning that allows for reasoning with partial observations over multiple time steps, achieving state-of-the-art on benchmarks.