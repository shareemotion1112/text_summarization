Model-based reinforcement learning has been empirically demonstrated as a successful strategy to improve sample efficiency.

Particularly, Dyna architecture, as an elegant model-based architecture integrating learning and planning, provides huge flexibility of using a model.

One of the most important components in Dyna is called search-control, which refers to the process of generating state or state-action pairs from which we query the model to acquire simulated experiences.

Search-control is critical to improve learning efficiency.

In this work, we propose a simple and novel search-control strategy by searching high frequency region on value function.

Our main intuition is built on Shannon sampling theorem from signal processing, which indicates that a high frequency signal requires more samples to reconstruct.

We empirically show that a high frequency function is more difficult to approximate.

This suggests a search-control strategy: we should use states in high frequency region of the value function to query the model to acquire more samples.

We develop a simple strategy to locally measure the frequency of a function by gradient norm, and provide theoretical justification for this approach.

We then apply our strategy to search-control in Dyna, and conduct experiments to show its property and effectiveness on benchmark domains.

Model-based reinforcement learning (MBRL) (Lin, 1992; Sutton, 1991b; Daw, 2012; Sutton & Barto, 2018) methods have successfully been applied to many benchmark domains (Gu et al., 2016; Ha & Schmidhuber, 2018; Kaiser et al., 2019) .

The Dyna architecture, introduced by Sutton (1991a) , is one of the classical MBRL architectures, which integrates model-free and model-based policy updates in an online RL setting (Algorithm 2 in Appendix A.3).

At each time step, a Dyna agent uses the real experience to learn a model and to perform model-free policy update, and during the planning stage, simulated experiences are acquired from the model to further improve the policy.

A closely related method in model-free learning setting is experience replay (ER) (Lin, 1992; Adam et al., 2012) , which utilizes a buffer to store experiences.

An agent using the ER buffer randomly samples the recorded experiences at each time step to update the policy.

Though ER can be thought of as a simplified form of MBRL (van Seijen & Sutton, 2015) , a model provides more flexibility in acquiring simulated experiences.

A crucial aspect of the Dyna architecture is the search-control mechanism.

It is the mechanism for selecting states or state-action pairs to query the model in order to generate simulated experiences (cf.

Section 8.2 of Sutton & Barto 2018) .

We call the corresponding data structure for storing those states or state-action pairs the search-control queue.

Search-control is of vital importance in Dyna, as it can significantly affect the model-based agent's sample efficiency.

A simple approach to searchcontrol is to sample visited states or state-action pairs, i.e., use the initial state-action pairs stored in the ER buffer as the search-control queue.

This approach, however, does not lead to an agent that outperforms a model-free agent that uses ER.

To see this, consider a deterministic environment, and assume that we have the exact model.

If we simply sample visited state-action pairs for searchcontrol, the next-state and reward would be the same as those in the ER buffer.

In practice, we have model errors too, which causes some performance deterioration (Talvitie, 2014; 2017) .

Without an elegant search-control mechanism, we are not likely to benefit from the flexibility given by a model.

Several search-control mechanisms have already been explored.

Prioritized sweeping (Moore & Atkeson, 1993 ) is one such method that is designed to speed up the value iteration process: the simulated transitions are updated based on the absolute temporal difference error.

It has been adopted to continuous domains with function approximation too (Sutton et al., 2008; Pan et al., 2018; Corneil et al., 2018) .

Gu et al. (2016) utilizes local linear models to generate optimal trajectories through iLQR (Li & Todorov, 2004) .

Pan et al. (2019) suggest a method to generate states for the searchcontrol queue by hill climbing on the value function estimate.

This paper proposes an alternative perspective to design search-control strategy: we can sample more frequently from the regions of the state space where the value function is more difficult to estimate.

In order to quantify the difficulty of estimation, we borrow a crucial idea from the signal processing literature: the Shannon sampling theorem, which establishes the connection between a signal's bandwidth and the number of samples required for its reconstruction.

A signal with higher frequency terms requires more samples for accurate reconstruction (Shannon sampling theorem has been studied in learning theory too, e.g., by Smale & Zhou 2004; 2005; Jiang 2019) .

A parallel of this idea in the learning setting is the heuristic that we would like to have more samples from regions of the state space where the value function has higher local frequency.

We establish a connection between a function's local frequency and its gradient and Hessian norm.

In order to sample from those regions, we use the hill climbing approach of Pan et al. (2019) .

These allow us to propose a search-control mechanism that focuses on sampling from regions where learning the value function is likely to be more difficult.

In this paper, we first review some basic background in MBRL (Section 2).

Afterwards, we review some concepts in signal processing and conduct experiments in the supervised learning setting to show that a high frequency function is more difficult to approximate (Section 3).

We observe that providing more samples in the high frequency region of a function can improve the efficiency of learning.

We then propose a method to locally measure the frequency of a point in a function's domain and provide a theoretical justification for our method (Theorem 1 in Section 3.2).

We use the hill climbing approach of Pan et al. (2019) to adapt our method to design a search-control mechanism for the Dyna architecture (Section 4).

We conduct experiments on benchmark and challenging domains to illustrate the properties and utilities of our method (Section 5).

Reinforcement learning (RL) problems are typically formulated as Markov Decision Processes (MDPs) (Sutton & Barto, 2018; Szepesvári, 2010 ).

An MDP (S, A, P, R, γ) is determined by state space S, action space A, transition function P, reward function R : S × A × S → R, and discount factor γ ∈ [0, 1].

At each step t, an agent observes a state s t ∈ S, and takes an action a t ∈ A. The environment receives a t , and transits to the next state s t+1 ∼ P(·|s t , a t ).

The agent receives a reward scalar r t+1 = R(s t , a t , s t+1 ).

The agent maintains a policy π : S × A → [0, 1] that determines the probability of choosing an action at a given state.

For a given state-action pair (s, a), the action-value function of policy π is defined as

) is the return of s 0 , a 0 , s 1 , a 1 , ... following the policy π and transition P. Value-based RL methods learn the action-value function (Watkins & Dayan, 1992) , and act greedily w.r.t.

the action-value function.

Policy-based RL methods perform gradient update of parameters to learn policies with high expected rewards (Sutton et al., 1999) .

Both value and policy-based RL methods can be easily adopted in the Dyna framework.

Model-based RL.

A model is a mapping that takes a state-action pair as its input and outputs some projection of the future state.

A model can be local Gu et al., 2016) or global (Ha & Schmidhuber, 2018; Pan et al., 2018 ), deterministic (Sutton et al., 2008) or stochastic (Deisenroth & Rasmussen, 2011; Ha & Schmidhuber, 2018) , feature-to-feature (Corneil et al., 2018; Ha & Schmidhuber, 2018) or observation-to-observation (Gu et al., 2016; Pan et al., 2018; Kaiser et al., 2019) , one-step (Gu et al., 2016; Pan et al., 2018) , or multi-step (Sorg & Singh, 2010; Oh et al., 2017) , or decision-aware (Joseph et al., 2013; Farahmand et al., 2017; Silver et al., 2017) .

Modelling the environment dynamics through a reproducing kernel Hilbert space (RKHS) embedding has been also studied (Grunewalder et al., 2012) , where the Bellman operator is approximated in an RKHS.

The model we consider in this work is a one-step environment dynamics model, which takes a state-action pair as its input and returns the next-state and reward.

Our proposed search-control approach, however, can be used for different types of models.

The most relevant work to ours are the classical Dyna architecture (Sutton, 1991a; b) , and its hill climbing variant (Pan et al., 2019) .

Pan et al. (2019) proposes a search-control mechanism based on hill climbing on the value estimates (see Algorithm 3 in Appendix A.3).

We briefly review the key steps of their algorithm, which is called HC-Dyna, as it helps to understand ours.

HC-Dyna maintains an ER buffer.

At each step, a state is randomly sampled from the ER buffer and is used as the initial state to perform hill climbing on the learned value function.

The hill climbing is performed by following the natural gradient, and the states along the trajectory are stored in the search-control queue.

1 The hill climbing procedure generates states to populate the search-control queue.

By making the connection to the Langevin dynamics, it can be shown that the distribution of samples in the search-control queue of HC-Dyna is asymptotically ∝ exp(V (s)), with V being the value function estimate (see the original paper or Appendix A.4 for more detail).

During the planning stage, states are sampled from the search-control queue and are paired with their corresponding onpolicy actions (i.e., actions selected by the current Q network at the sampled states).

Afterwards, the model is queried for each of the state-action pairs to get the next-state and reward.

These constitute the simulated transitions.

The simulated transitions are then mixed with samples from the ER buffer, which are observed by the agent during its interaction with the real environment, to train the value function estimator, e.g., a deep neural network.

The heuristic idea behind the search-control mechanism of HC-Dyna is that the magnitude of the value function provides useful information for guiding where to query the model.

This heuristic can intuitively be understood by noticing that an RL agent tends to move towards high-value regions of the state space, so by performing gradient ascent on the (estimated) value function, we provide the agent with more samples from regions where it may move towards in the future.

Even if the estimated value function is incorrect and the samples are indeed from the low-value regions of the state space, these extra samples lead to the fast correction of the estimated value in those regions.

Nevertheless, the magnitude of the value function is only one source of extra information from which we can design a search-control mechanism.

This work suggests a different perspective: We have to sample more from the regions of the state space where learning the value function is more difficult.

And the difficulty of learning, as we show next, is related to the frequency representation of the function in that region.

Using the supervised setting of regression, we illustrate that high frequency regions of a function is difficult to approximate.

We show that by assigning more training data to those regions, the learning performance considerably improves.

To make this insight practically useful, we propose the norm of the gradient as a measure of the local frequency of a function.

We establish a theoretical connection between our proposed criterion and the local frequency of a function.

This would be the foundation of our frequency-based search-control method in Section 4.

Consider the standard regression problem with the 2 loss.

Given a training set D = {(x i , y i )} i=1:n , our goal is to learn an unknown target function f

by empirical risk minimization.

Formally, we aim to solve

where H is a hypothesis space.

Suppose that we can choose the distributions of samples {x i }.

How should we select them in order to improve the quality of the learned function?

One intuitive heuristic is that if we know the regions in the domain of f * that are more difficult to approximate, we can assign more training data there in order to help the learning process.

The important question is how to quantify the difficulty of approximating a function.

We borrow an idea from the field of signal processing to suggest a method.

Figure 1 : Learning curves are testing error as functions of mini-batch update number.

Bias level 60% means 60% of the training data are from the high frequency region [−2, 0) and is labeled as Biased-high.

Similarly, Biased-low means 60% of the training data are from the low frequency region [0, 2].

We include unbiased training dataset as a reference (Unbiased).

The total numbers of training data are the same across all experiments.

The testing set is unbiased and the results are averaged over 50 random seeds with the shade as standard error.

The Nyquist-Shannon sampling theorem in signal processing states that given a band-limited function (or signal) f : R → R with the highest frequency (in the Fourier domain) of ω bandwidth , we can perfectly reconstruct it based on regular samples (in the time domain) obtained at the sampling rate of 2ω bandwidth (Zayed, 1993) .

2 Therefore, if the Fourier transform of a function has high frequency terms, more samples are required to reconstruct it accurately.

We note that the sampling theory has been applied in the sample complexity analysis of machine learning algorithms (Smale & Zhou, 2004; 2005; Jiang, 2019) .

Although the problem setting in machine learning is somewhat different from this result in signal processing, it still provides a high-level intuition for us: regions with more high frequency signals require more learning data.

To make this high-level intuition concrete, we consider the following function:

It is easy to check that the regions [−2, 0) and [0, 2] contain signals with frequency ratio 8 : 1.

Based on the intuition from the sampling theorem, the [−2, 0) interval requires more training data than the [0, 2] interval.

Given the same amount of training data, and the same machine learning algorithm, we would expect that assigning more fraction of the training data on [−2, 0) to perform better than distributing them uniformly or if we assign more samples to the [0, 2] interval.

An illustrative experiment.

To empirically verify the intuition, we conduct a simple regression task, with f sin as the target function.

The training set D = {(x i , y i )} i=1:n is generated by sampling x ∈ [−2, 2], and adding Gaussian noise N (0, σ 2 ) on Eq. (1), where the standard deviation is set to be σ = 0.1.

We present 2 regression learning curves of training datasets with different biased sampling ratios p b ∈ {60%, 70%, 80%}, as shown in Fig. 1 (a) -(c).

We observe that biased training data sampling ratios towards high frequency region (more samples in high frequency region) clearly speeds up learning.

This is consistent with the intuitive insight and suggests that our heuristic to assign more data to high frequency regions leads to better learning results.

Identifying the high frequency region of f sin in the previous toy problem was easy, as each region contained a signal with a constant known frequency.

In practice, we face two main difficulties to identify the high frequency regions of a function.

The first is that we do not have access to the underlying target function, but only to data or possibly an approximate function that is estimated using data, e.g., a trained neural network.

The second is that frequency is a global property rather than a local one.

The value of the function at each (non-zero measure) region of the domain has impact on its global frequency representation.

To make the high frequency heuristic practically useful, we need a simple criterion that (a) uses function approximation, (b) characterizes local frequency information, and (c) can be efficiently calculated.

Inspired by the function f sin in Eq. (1), a natural idea is to calculate the first order f (x)

dx 2 because they both satisfy (a) and (c).

To do "sanity check" for property (b), consider the following illustrative examples.

Example 1.

For f sin defined in Eq. (1), calculate the integrals of squared first order derivative f sin on high frequency region [−2, 0) and low frequency region [0, 2], respectively:

Here a n , b n ∈ R, n = 1, 2, . . .

are Fourier coefficients of frequency n 2π , defined as

Example 1 shows that the integral of squared first order derivative ratio is 64 : 1 (the frequency ratio is 8 : 1), and the region with large gradient norm is indeed the high frequency region.

Moreover, Example 2 indicates that for one dimensional real-valued functions over a bounded domain, integrals of squared gradient norm and Hessian norm are closely related to the frequency information, and reflect their high frequency behaviour.

For the squared gradient norm, the integral is the same as weighting the frequency terms a n and b n proportional to n 2 , and for the squared Hessian norm, the integral is the same as weighting the frequency terms proportional to n 4 .

The weighting schemes n 2 or n 4 emphasize the higher frequency terms.

Empirical demonstration.

Our calculation in the above examples implies that regions with large gradient and Hessian norm correspond to high frequency regions.

Based on the same spirit of the l 2 regression task in Section 3.2, we empirically verify this insight.

Our expectation is that biasing training dataset towards high gradient norm and Hessian norm would achieve better learning results.

In Fig. 2 (a), Biased-GradientNorm corresponds to uniformly sampling x ∈ [−2, 2] for 60% of training data and sampling proportional to gradient norm (i.e., p(x) ∝ |f sin (x)|) for the remaining 40%; while Biased-HessianNorm corresponds to sampling proportional to Hessian norm (i.e., p(x) ∝ |f sin (x)|) for the remaining 40% of training data.

In Fig (c) is that, sampling according to Hessian norm leads to denser points around spikes: there are 18.17% points fall in the yellow area in (b) and 27.45% such points in (c).

Those areas around spikes should be more difficult to approximate as the underlying function changes sharply, which explains the superior performance on the data set biased by Hessian norm.

Fig. 2 (a) shows that such biased training datasets provide fast learning, similar to the high frequency biased training datasets in Fig. 1 .

We also observe that Biased-HessianNorm learns faster than Biased-GradientNorm.

As a result of passing "sanity check" of calculations and experiments, given a function f : X → Y and a point x ∈ X , we propose to measure frequency of f around a small neighborhood of x (we call this local frequency) using the following function:

where ∇ x f (x) is the 2 -norm of the gradient at x, and H f (x) F is the Frobenius norm of the Hessian matrix of f at x. We claim that local frequency of f around x is proportional to g(x).We theoretically justify this claim.

For real-valued functions in Euclidean spaces, our theory establishes a connection between local gradient norm, local Hessian norm, local function energy 3 , and local frequency distribution.

The proof can be found in Appendix A.2.

Theorem 1.

Given any function f : R n → R, for any frequency vector k ∈ R n , define its local Fourier transform around x ∈ R n ,

for local function around x, i.e., {y : y − x ≤ 1}. Assume that the local function "energy" is finite,

Define "local frequency distribution" of f (x) as:

Then, for any x ∈ R n , we have: 1) The first order connection:

2) The second order connection:

Remark 1.

Note that πf defined in Eq. (4) is a probability distribution over R n as:

We use such a distribution to characterize local frequency behaviour for reasons.

First, comparing frequencies of regions is more naturally captured by a distribution than one single scalar, since signals usually are within a range of frequencies.

Second, to eliminate the impact of the function energy Eq. (3), we normalize the Fourier coefficientf to get πf .

Remark 2.

For a frequency vector k ∈ R n , the larger its norm k is, the higher its frequency is.

Given any x and its local function (i.e., f (·) around x), πf (k) is the proportion/percentage that frequency k occupies.

Therefore, the integral of πf (k) · k 2 reflects the contribution of high frequency terms in the local frequency distribution of a function.

Remark 3.

Consider f as a value function in reinforcement learning setting.

Theorem 1 indicates that regions with large gradient norm can either have large absolute value function, or high local frequency, or both.

To prevent finding regions that only have large negative value function, our theory implies that it is reasonable to take both gradient norm and value function into account, as our proposed method does in next section.

Discussion with Uncertainty Principle.

The Uncertainty Principle says that a function cannot be too concentrated in both spatial and frequency space, i.e.,

Combing Eq. (7) with our Eq. (5),

which means the more concentrated f is locally around x, the larger local gradient norm must be.

On the other hand, if local gradient norm is small, then f cannot be too concentrated around x.

We present the Dyna architecture with the frequency-based search-control (Algorithm 1) in this section.

It combines the idea that samples from high-frequency regions of the state space is important, as discussed in the previous section, and the hill climbing process to effectively draw samples from those regions, as introduced by Pan et al. (2019) .

We omit detail such as preconditioning, noisy gradient for the hill climbing process, and other implementation detail in this section, and refer the reader to Appendices A.6 and A.7 instead.

Our goal is to query the model more often from the regions of the state space where the local frequency of the value function is higher.

The intuition behind this search-control mechanism, as discussed in the previous section in the context of supervised learning, is that those regions correspond to where learning the (value) function is more difficult, hence more samples from the model might be helpful.

To populate the search-control queue with states from those regions, we can do hill climbing on g(s) = ∇ s V (s) + H v (s) F .

Theorem 1, however, suggests that states with large gradient norm can either have large absolute value, or high local frequency, or both.

We want to avoid many samples from regions with large negative value states, as those states may be rarely visited under the optimal policy anyway.

A sensible strategy to get around this problem is to combine the proposed hill climbing method with the previous hill climbing on the value function (Pan et al., 2019) , as the latter tends to generate samples from high value states (and not states with large negative values).

We propose the following method for combining those approaches.

At each time step, with certain probability, we perform hill climbing by either

with probability of 1 − p

and store states along the gradient trajectory in the search-control queue.

When hill climbing on the value function (8b), we sample the initial state from the ER buffer as suggested by the previous work (Pan et al., 2019) .

This populates the search-control queue with states from the high value regions of the state space.

When hill climbing on g(s) (8a), however, we sample the initial state from the search-control queue itself (instead of the ER buffer).

This way ensures that the initial state for searching high frequency region has relatively high value.

Hill climbing based on ∇g(s) from an initial state with a high value populates the search-control queue with high frequency samples around high value regions of the state space.

We discuss some other intuitive mechanisms that we have tested in Appendix A.4.

Similar to the previous work (Pan et al., 2019) , we obtain the state-value function in both (8a) and (8b) by taking the maximum of the estimated action-value, i.e. V (s) = max a Q(s, a) ≈ max a Q θ (s, a) where θ is the parameter of the Q-network.

Similar to the Dyna architecture (Algorithm 2), during planning stage, we sample multiple mixed mini-batches to update the parameters draw βb sample states from the search-control queue B s , pair them with their corresponding on-policy action, and query M to get their next-states and rewards draw (1 − β)b sample transitions from the ER buffer B and add them to the simulated transitions use the mixed mini-batch for parameter update of the estimator, e.g., DQN n τ ← n τ + 1 if mod (n τ , τ ) == 0 then: Q ← Q t ← t + 1 (i.e. we call multiple planning steps/updates).

The mixed mini-batch was also used in the work by Gu et al. (2016) and can alleviate off-policy sampling issue as studied by Pan et al. (2019) .

Similar to the previous work (Pan et al., 2019) , we obtain the state-value function in both (8a) and (8b) by taking the maximum action-value, i.e. V (s) = max a Q(s, a) ≈ max a Q θ (s, a) where θ is the parameter in the Q-network.

As an analogy to the Dyna architecture (Algorithm 2), during planning stage, we sampled multiple mixed mini-batches to update the parameters (i.e. we call multiple planning steps/updates).

The mixed mini-batch was also used in the work by Gu et al. (2016) and has the effect of alleviating off-policy sampling issue as studied by Pan et al. (2019) .

In the experiments, we carefully study the properties of our algorithm on the MountainCar benchmark domain.

Then we illustrate the utility of our algorithm on a challenging self-designed MazeGridWorld domain, by which we illustrate the practical implication of having samples from the high frequency regions.

Though we mainly focuses on search-control instead of how to learn a model, we include the result of using an online learned model for our algorithm.

We refer readers to Appendix A.5 for additional experiments and Appendix A.6 for the reproducibility detail.

The MountainCar (Brockman et al., 2016) domain is well-studied, and it is known that the value function under the optimal value function has sharp changing regions (Sutton & Barto, 2018) , which should be beneficial for our algorithm.

The agent needs to learn to reach the goal state within as few steps as possible since the reward −1 per time step.

The purposes of experimenting on this domain are: 1) verify that our search-control can outperform several natural competitors under different number of planning updates; 2) show that our search-control is robust to environment noise.

We use the following competitors.

Dyna-Frequency is the Dyna architecture using the proposed search-control strategy (Algorithm 1); Dyna-Value is Algorithm 3 from the previous work (Pan et al., 2019) ; PrioritizedER is DQN with prioritized experience replay (Schaul et al., 2016) ; ER is simply DQN with experience replay (ER) (Mnih et al., 2015) .

Figure 3 shows the learning curves of all those algorithms using 10 planning updates (a)(b) and 30 planning updates (c)(d) under different stochasticity.

In Figure 3 (b)(d), we add zero mean Gaussion noise to the original reward, i.e., −1 + η, η ∼ N (0, σ 2 ).

We make several important observations: 1) With increased number of planning updates, these algorithms do not necessarily perform better, as shown in Figure 3 (c).

The proposed algorithm, however, appears to gain more through more number of updates since the difference between DynaFrequency and Dyna-Value seems to be clearer in Figure 3 (c) than in Figure 3 (a).

2) Since both Dyna-Value and our algorithm fetch the same number of states (i.e. m = 20) by hill climbing, the superior performance of our algorithm indicates the advantage of using samples from the high frequency regions.

3) PrioritizedER clearly performs worse than our algorithm and Dyna-Value, which probably implies the utility of the generalization power of the value function to acquire additional samples.

4) Our algorithm maintains superior performance in the presence of noise.

One reason is that, noisy perturbation leads to more "energy" in all frequencies.

When we take derivative, those high frequency terms are amplified.

Hence, even with perturbation, high frequency region remains while the value estimate itself may get affected in an unpredictable manner.

We now illustrate the utility of our method on a challenging MazeGridWorld domain as shown in Figure 4 (a).

The domain has continuous state space S = [0, 1] 2 and four discrete actions {up, down, left, right}. There are three walls in the middle, each of which has a hole for the agent to go through.

Each episode starts from the bottom left and ends at top right and the agent receives a reward of −1 at each time step, hence the agent should learn to use as few steps as possible to reach the goal area.

Model-free methods completely fail on this domain, and we mainly study our algorithm and the Dyna-Value algorithm.

Figure 4(b) shows the evaluation curves of the two algorithms.

An important difference between our algorithm and the previous work is in the variance of the evaluation curve, which implies a robust policy learned by our method.

In Figure 5 , we further investigate the state distribution in searchcontrol queues of the two algorithms by uniformly sampling 1000 states from the two queues.

Notice that a very important difference between the two distributions is that our search-control queue has a clearly high density around the bottleneck area, i.e., the hole areas where the agent can go across the walls.

Learning a stable policy around such areas is extremely important: the agent simply fails to reach the goal state if they cannot pass any one of the holes.

This distinguishes our algorithm with the previous work, which appears to acquire states near the goal area.

The state distribution in the search-control queue of our algorithm Dyna-Frequency (a) and Dyna-Value (b) at 50k environment time step.

Each blue shadow area is a 0.1 × 0.1 square indicating the hole where the agent can go through the wall.

Our search-control queue has a state distribution with a high density around those squares.

In (a), there are 25.3% points fall inside a 0.1 radius ball centered at each square in total; in (b), there are 11.7% such points.

The black box on the top right is the goal area.

We motivated and studied a new category of methods for search-control by considering the approximation difficulty of a function.

We provided a method for identifying the high frequency regions of a function, and justified it theoretically.

We conducted experiments to illustrate our theory.

We incorporated the proposed method into the Dyna architecture and empirically investigated its benefits.

The method achieved competitive learning performances on a difficult domain.

There are several promising future research directions.

First, it is worth exploring the combination of different search-control strategies.

Second, exploring the use of active learning methods (Settles, 2010; Hanneke, 2014) , which try to learn a function with as few samples as possible, to design search-control mechanism in MBRL algorithms might be a fruitful direction.

Proof.

Taking derivative and integral,

Example 2.

Let f : [−π, π] → R be a real valued function.

We have

a n , b n ∈ R, n = 1, 2, . . .

are Fourier coefficients of frequency n 2π , defined as

Proof.

The Fourier series of f (x) is

where a 0

Taking square of f ,

Using similar arguments, taking derivative of f (x),

Notations.

For any vector norm || · ||, we mean l 2 norm and we ignore the subscript unless clarification is needed.

We use Frobenius norm || · || F for matrix.

We use subscript y l to denote the lth element in vector y. Let H f (y) be the Hessian matrix of f (y).

We write H for short unless clarification is needed.

Let H l,: be the lth row of the Hessian matrix.

Proof description.

We establish the connection between local gradient norm, Hessian norm and local frequency.

To build such connection, we introduce a definition of πf as shown below and we call it "local frequency distribution" of f (x).

πf is a probability distribution over R n , i.e., k∈R n πf (k)dk = 1, and πf (k) ≥ 0, ∀k ∈ R n .

Within a subset of domain (an unit ball), this distribution characterizes the proportion of a particular frequency component occupies.

The proof can be described by three key steps:1) We use a local Fourier transform to express a function locally (i.e. within an unit ball).

2) we calculate the gradient/Hessian norm based on this local Fourier transform; 3) we take integration over the unit ball of the gradient/Hessian norm to build the connection with the local frequency distribution πf and function energy.

Theorem 1.

Given any function f : R n → R, for any frequency vector k ∈ R n , define its local Fourier transformation of x ∈ R n ,

for local function around x, i.e., {y : y − x ≤ 1}. Assume the local function "energy" is finite,

Define "local frequency distribution" of f (x) as:

Then, ∀x ∈ R n , we have: 1) the first order connection:

2) the second order connection:

Proof.

1) We first prove the first order connection.

Consider the following function defined locally around x,

The coefficient of frequency vector k in the Fourier series of

And the Fourier series of f x (y), ∀y, such that y − x ≤ 1, is,

The gradient is

To calculate gradient norm, we use complex conjugate,

is the complex conjugate off (k).

Therefore,

Taking integral of ∇f (y) 2 within the unit ball centered at x,

Recall the definition of local function "energy" around x,

The local gradient information is related to local energy and frequency distribution,

where the last equality follows by

2) Now we prove the second order connection.

To show the second order connection, we start from Eq. (9).

Unless otherwise specified, all notations are exactly the same with those defined in the proof for first-order connection.

Then the lth row of the Hessian matrix H l,: can be written as:

where we use the notation ∂∇f (y) ∂y l to denote the vector formed by taking partial derivative of each element in the gradient vector ∇f (y) w.r.t.

y l (i.e the lth element in the y vector).

Then, ∂∇f (y)

where e l is a one-hot vector where the lth element is one.

To calculate the norm of the vector H l,: = ∂∇f (y) ∂y l , we use complex conjugate again and follow the similar derivation as done in Eq. (10):

Note that the square of Frobenius norm of the Hessian matrix can be written as ||H|| Taking the integration of ||H|| 2 F over y variable within a ball with center x and unit radius, we acquire:

where the derivation process for the first equation is a simple modification from the derivation (11) and the second equation follows the same derivation (13).

In this section, we provide the vanilla Dyna (Sutton, 1991a; Sutton & Barto, 2018) in Algorithm 2, and Hill Climbing Dyna by Pan et al. (2019) in Algorithm 3.

Dyna is a classic model-based reinforcement learning architecture.

As described in Algorithm 2, at each time step, the real experience is used to directly improve policy/value estimates, and is also used to learn the environment dynamics model.

During planning stage, simulated experiences are acquired from the learned model and are used to further improve the policy.

(s) , and the effect is unclear.

It may lead to states with neither high value or high frequency.

Last, and probably the most important, hill climbing on V (s) and on g(s) have fundamentally different insights.

The former is based on the intuition that the value information should be propagated from the high value region to low value region; as a result, it requires to store states along the whole trajectory, including those in low value region.

This is empirically verified by Pan et al. (2019) .

However, the latter is based on the insight that the function value in high frequency region is more difficult to approximate and needs more samples, while there is no obvious reason to propagate those information back to low frequency region.

As a result, this approach does not emphasize on recording states throughout the whole hill climbing trajectory.

A theoretical interpretation of the search-control queue distribution.

Given our search-control queue, it is natural to ask that what would be the state distribution in the queue, as this may be helpful to develop other search-control strategies.

Pan et al. (2019) establishes the connection between the state distribution in the search-control queue filled by value-based hill climbing through Langevin dynamics.

The basic idea is to show that the hill climbing process is tracking a discretized version of a stochastic differential equation, whose limiting distribution is Gibbs (Roberts, 1996; Chiang et al., 1987; Welling & Teh, 2011) .

As a result, the state distribution in search-control queue is approximately p(s) ∝ exp (V (s)).

Following a similar reasoning line, our search-control queue should approximately be a state distribution: p(s) ∝ exp (V (s)) + exp (g(s)).

Notice that, removing the first term leads to a sampling distribution resembles to the one in our supervised learning experiment, where the training data distribution is biased towards gradient norm Fig. 2 .

As a future direction, it is interesting to investigate the underlying theoretical connection between sampling distribution p(x) ∝ exp (g(x)) and sample complexity.

In this section, we briefly study the effect of doing hill climbing on only gradient norm or Hessian norm.

Then we demonstrate that our search-control strategy can be also used for continuous control algorithms.

Hill climbing on only gradient norm or Hessian norm.

Throughout our paper, we use the form of g(s) = ∇ s V (s) + H v (s) F to search states from high (local) frequency region of the value function.

Besides the theoretical reason, there is a practical demand of such design.

On value function surface, regions which have low (or even zero) gradient magnitude may have high Hessian magnitude, and vice versa.

Hence, it can help move along the gradient trajectory in case that one of the term vanished at some point.

Such cases can be a result of function approximation (smoothness/differentiability), or of the nature of the task, or both.

In Fig. 6 , we show the results of using only either gradient norm or Hessian norm.

The reason we choose MountainCar and GridWorld (the same domain as described by Pan et al. (2019) ) is that, the former has a value function surface with lots of variations; while the latter's value function increases smoothly from the initial state to the goal state, which indicates a small magnitude second-order derivative.

Indeed, we empirically observe that the term ∇ s H v (s) F frequently gives a zero vector.

This explains the bad performance of Dyna-HessNorm in Fig. 6(b) .

In contrast, Fig. 6(a) shows slightly better performance of Dyna-HessNorm and Dyna-GradNorm.

Notice that, an intuitive and more general form of g(x) can be g(s) = η 1 ∇ s V (s) + η 2 H v (s) F , at the cost that additional meta-parameters are introduced.

Continuous Control.

In this section, we show a simple demonstration where our method is adapted to continuous control setting.

We now demonstrate the application of our search-control strategy on two challenging continuous control domains Hopper-v2 and Walker2d-v2 from Mujoco by using a continuous Q learning algorithm called NAF (Normalized Advantage Function) (Gu et al., 2016) .

The algorithm parameterizes the action value function as found: arg max a Q(s, a) = µ(s).

Our search-control strategy naturally applies here by utilizing the value function V (s).

We refer readers to the work by Pan et al. (2019) for applying our search-control strategy to policy gradient methods.

From Fig. 7 , one can see that our algorithm (DynaNAF-Frequency) finds a better policy comparing with the model-free NAF.

A.6 REPRODUCIBILITY DETAIL All of our implementations are based on tensorflow with version 1.13.0 (Abadi et al., 2015) .

For DQN update, we use Adam optimizer.

We use mini-batch size b = 32 except on the supervised learning experiment where we use 128.

For reinforcement learning experiment, we use buffer size 100k.

All activation functions are tanh except the output layer of the Q-value is linear.

Except the output layer parameters which were initialized from a uniform distribution [−0.003, 0.003] , all other parameters are initialized using Xavier initialization (Glorot & Bengio, 2010) .

For model learning, we use a 64 × 64 relu units neural network to predict s − s given a state-action pair with mini-batch size 128 and learning rate 0.0001.

For the supervised learning experiment shown in Section 3, we use 16 × 16 tanh units neural network, with learning rate 0.001 for all algorithms.

The learning curve is plotted by computing the testing error every 20 iterations.

When generating Fig. 2 , in order to sample points according to p(x) ∝ |f (x)| or p(x) ∝ |f (x)|, we use 10, 000 even spaced points on the domain [−2, 2] and the probabilities are computed by normalization across the 10k points.

The experiment on MountainCar is based on the implementation from OpenAI (Brockman et al., 2016) , we use 32 × 32 tanh layer, with target network moving rate 1000 and learning rate 0.001.

Exploration noise is 0.1 without decaying.

For all algorithms, we use warm up steps = 5000 (i.e. random action is taken in the first 5k time steps).

Prioritized experience replay (PrioritizedER) is implemented as the proportional version with sum tree data structure.

We use prioritized ER without importance ratio but half of mini-batch samples are uniformly sampled from the buffer as a strategy for bias correction.

For Dyna-Value and Dyna-Frequency, we use: gradient ascent step size (in search-control) 0.01, mixing rate β = 0.5 and m = 20, i.e., at each environment time step we fetch 20 states by hill climbing.

We fix p = 0.5 across all experiments, hence the hill climbing rules (8a) and (8b) are selected with equal probability.

We use natural projected gradient ascent for hill climbing as introduced by Pan et al. (2019) .

For the experiment on MazeGridWorld, each wall's width is 0.1 and each hole has height 0.1.

The left-top point of the hole in the first wall (counting from left to right) has coordinate (0.2, 0.5); the hole in the second wall has coordinate (0.4, 1.0) and the third one is 0.7, 0.2.

Each action leads to 0.05 unit move perturbed by a Gaussian noise from N (0, 0.01).

On this domain, for both DynaValue and Dyna-Frequency, all parameters are set the same with that used on MountainCar except that we use 64 × 64 tanh units for Q network, and number of search-control samples is set as m = 50, number of planning updates is 30.

As a supplement to the Section 5.2, we also provide the state distribution from ER buffer in Figure 8 .

One can see that ER buffer has very different state distribution with search-control queue.

We provide the pseudo-code in Algorithm 4 with sufficient details to recreate our experimental results.

The hill climbing rules we used is the same as introduced by Pan et al. (2019 Note that we use a squared norm to ensure numerical stability when taking gradient.

Then for value-based search-control, we use

and for frequency-based search-control, we use

whereΣ s is empirical covariance matrix estimated from visited states, and we set η = 0.01, α = 0.01 across all experiments.

Notice that comparing with the previous work, we omitted the projection step as we found it is unnecessary in our experiments.

Algorithm 4 Dyna architecture with Frequency-based search-control with additional details B s : search-control queue, B: the experience replay buffer M : S × A → S × R, the environment model m: number of search-control samples to fetch at each step p: probability of choosing value-based hill climbing rule (we set p = 0.5 for all experiments) β ∈ [0, 1]: mixing factor in a mini-batch, i.e. βb samples in a mini-batch are simulated from model n: number of state variables, i.e. S ⊂ R n a : empirically learned threshold as sample average of ||s t+1 − s t || 2 / √ n d: number of planning steps Q, Q : current and target Q networks, respectively b: the mini-batch size τ : update target network Q every τ updates to Q t ← 0 is the time step n τ ← 0 is the number of parameter updates // Gradient ascent hill climbing With probability p, 1 − p, choose hill climbing Eq. (14) sample βb states from B s and pair them with on-policy actions, and query M to get next states and rewards sample b(1 − β) transitions from B an stack these with the simulated transitions use the mixed mini-batch for parameter (i.e. DQN) update n τ ← n τ + 1 if mod(n τ , τ ) == 0 then: Q ← Q t ← t + 1

@highlight

Acquire states from high frequency region for search-control in Dyna.

@highlight

The authors propose to do sampling in the high-frequency domain to increase the sample efficiency

@highlight

This paper proposes a new way to select states from which do do transitions in dyna algorithm.