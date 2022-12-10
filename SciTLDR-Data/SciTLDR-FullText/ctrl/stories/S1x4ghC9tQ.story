To act and plan in complex environments, we posit that agents should have a mental simulator of the world with three characteristics: (a) it should build an abstract state representing the condition of the world; (b) it should form a belief which represents uncertainty on the world; (c) it should go beyond simple step-by-step simulation, and exhibit temporal abstraction.

Motivated by the absence of a model satisfying all these requirements, we propose TD-VAE, a generative sequence model that learns representations containing explicit beliefs about states several steps into the future, and that can be rolled out directly without single-step transitions.

TD-VAE is trained on pairs of temporally separated time points, using an analogue of temporal difference learning used in reinforcement learning.

Generative models of sequential data have received a lot of attention, due to their wide applicability in domains such as speech synthesis BID18 , neural translation BID3 , image captioning BID22 , and many others.

Different application domains will often have different requirements (e.g. long term coherence, sample quality, abstraction learning, etc.), which in turn will drive the choice of the architecture and training algorithm.

Of particular interest to this paper is the problem of reinforcement learning in partially observed environments, where, in order to act and explore optimally, agents need to build a representation of the uncertainty about the world, computed from the information they have gathered so far.

While an agent endowed with memory could in principle learn such a representation implicitly through model-free reinforcement learning, in many situations the reinforcement signal may be too weak to quickly learn such a representation in a way which would generalize to a collection of tasks.

Furthermore, in order to plan in a model-based fashion, an agent needs to be able to imagine distant futures which are consistent with the agent's past.

In many situations however, planning step-by-step is not a cognitively or computationally realistic approach.

To successfully address an application such as the above, we argue that a model of the agent's experience should exhibit the following properties:• The model should learn an abstract state representation of the data and be capable of making predictions at the state level, not just the observation level.• The model should learn a belief state, i.e. a deterministic, coded representation of the filtering posterior of the state given all the observations up to a given time.

A belief state contains all the information an agent has about the state of the world and thus about how to act optimally.• The model should exhibit temporal abstraction, both by making 'jumpy' predictions (predictions several time steps into the future), and by being able to learn from temporally separated time points without backpropagating through the entire time interval.

To our knowledge, no model in the literature meets these requirements.

In this paper, we develop a new model and associated training algorithm, called Temporal Difference Variational Auto-Encoder (TD-VAE), which meets all of the above requirements.

We first develop TD-VAE in the sequential, non-jumpy case, by using a modified evidence lower bound (ELBO) for stochastic state space models (Krishnan et al., 2015; BID12 BID8 which relies on jointly training a filtering posterior and a local smoothing posterior.

We demonstrate that on a simple task, this new inference network and associated lower bound lead to improved likelihood compared to methods classically used to train deep state-space models.

Following the intuition given by the sequential TD-VAE, we develop the full TD-VAE model, which learns from temporally extended data by making jumpy predictions into the future.

We show it can be used to train consistent jumpy simulators of complex 3D environments.

Finally, we illustrate how training a filtering a posterior leads to the computation of a neural belief state with good representation of the uncertainty on the state of the environment.2 MODEL DESIDERATA

Autoregressive models.

One of the simplest way to model sequential data (x 1 , . . . , x T ) is to use the chain rule to decompose the joint sequence likelihood as a product of conditional probabilities, i.e. log p(x 1 , . . .

, x T ) = t log p(x t | x 1 , . . . , x t−1 ).

This formula can be used to train an autoregressive model of data, by combining an RNN which aggregates information from the past (recursively computing an internal state h t = f (h t−1 , x t )) with a conditional generative model which can score the data x t given the context h t .

This idea is used in handwriting synthesis BID15 , density estimation (Uria et al., 2016) , image synthesis (van den BID19 , audio synthesis (van den BID20 , video synthesis (Kalchbrenner et al., 2016) , generative recall tasks BID13 , and environment modeling (Oh et al., 2015; BID9 .While these models are conceptually simple and easy to train, one potential weakness is that they only make predictions in the original observation space, and don't learn a compressed representation of data.

As a result, these models tend to be computationally heavy (for video prediction, they constantly decode and re-encode single video frames).

Furthermore, the model can be computationally unstable at test time since it is trained as a next step model (the RNN encoding real data), but at test time it feeds back its prediction into the RNN.

Various methods have been used to alleviate this issue Lamb et al., 2016; BID14 BID0 .State-space models.

An alternative to autoregressive models are models which operate on a higher level of abstraction, and use latent variables to model stochastic transitions between states (grounded by observation-level predictions).

This enables to sample state-to-state transitions only, without needing to render the observations, which can be faster and more conceptually appealing.

They generally consist of decoder or prior networks, which detail the generative process of states and observations, and encoder or posterior networks, which estimate the distribution of latents given the observed data.

There is a large amount of recent work on these type of models, which differ in the precise wiring of model components BID4 BID10 Krishnan et al., 2015; BID1 BID12 Liu et al., 2017; Serban et al., 2017; BID8 Lee et al., 2018; BID17 .Let z = (z 1 , . . . , z T ) be a state sequence and x = (x 1 , . . . , x T ) an observation sequence.

We assume a general form of state-space model, where the joint state and observation likelihood can be written as p(x, z) = t p(z t | z t−1 )p(x t | z t ).1 These models are commonly trained with a VAEinspired bound, by computing a posterior q(z | x) over the states given the observations.

Often, the posterior is decomposed autoregressively: q(z | x) = t q(z t | z t−1 , φ t (x)), where φ t is a function of (x 1 , . . .

, x t ) for filtering posteriors or the entire sequence x for smoothing posteriors.

This leads to the following lower bound: DISPLAYFORM0 A key feature of sequential models of data is that they allow to reason about the conditional distribution of the future given the past: p(x t+1 , . . .

, x T | x 1 , . . .

, x t ).

For reinforcement learning in partially observed environments, this distribution governs the distribution of returns given past observations, and as such, it is sufficient to derive the optimal policy.

For generative sequence modeling, it enables conditional generation of data given a context sequence.

For this reason, it is desirable to compute sufficient statistics b t = b t (x 1 , . . .

, x t ) of the future given the past, which allow to rewrite the conditional distribution as p(x t+1 , . . . , x T | x 1 , . . . , x t ) ≈ p(x t+1 , . . . , x T | b t ).

For an autoregressive model as described in section 2.1, the internal RNN state h t can immediately be identified as the desired sufficient statistics b t .

However, for the reasons mentioned in the previous section, we would like to identify an equivalent quantity for a state-space model.

For a state-space model, the filtering distribution p(z t | x 1 , . . . , x t ), also known as the belief state in reinforcement learning, is sufficient to compute the conditional future distribution, due to the Markov assumption underlying the state-space model and the following derivation: DISPLAYFORM1 Thus, if we train a network that extracts a code DISPLAYFORM2 , b t would contain all the information about the state of the world the agent has, and would effectively form a neural belief state, i.e. a code fully characterizing the filtering distribution.

Classical training of state-space model does not compute a belief state: by computing a joint, autoregressive posterior q(z | x) = t q(z t | z t−1 , x), some of the uncertainty about the marginal posterior of z t may be 'leaked' in the sample z t−1 .

Since that sample is stochastic, to obtain all information from (x 1 , . . .

, x t ) about z t , we would need to re-sample z t−1 , which would in turn require re-sampling z t−2 all the way to z 1 .While the notion of a belief state itself and its connection to optimal policies in POMDPs is well known BID2 Kaelbling et al., 1998; Hauskrecht, 2000) , it has often been restricted to the tabular case (Markov chain), and little work investigates computing belief states for learned deep models.

A notable exception is (Igl et al., 2018) , which uses a neural form of particle filtering, and represents the belief state more explicitly as a weighted collection of particles.

Related to our definition of belief states as sufficient statistics is the notion of predictive state representations (PSRs) (Littman & Sutton, 2002) ; see also BID21 for a model that learns PSRs which, combined with a decoder, can predict future observations.

Our last requirement for the model is that of temporal abstraction.

We postpone the discussion of this aspect until section 4.

In this section, we develop a sequential model that satisfies the requirements given in the previous section, namely (a) it constructs a latent state-space, and (b) it creates a online belief state.

We consider an arbitrary state space model with joint latent and observable likelihood given by DISPLAYFORM0 , and we aim to optimize the data likelihood log p(x).

We begin by autoregressively decomposing the data likelihood as: log p(x) = t log p(x t | x <t ).

For a given t, we evaluate the conditional likelihood p(x t | x <t ) by inferring over two latent states only: z t−1 and z t , as they will naturally make belief states appear for times t − 1 and t: DISPLAYFORM1 Because of the Markov assumptions underlying the state-space model, we can simplify DISPLAYFORM2 .

Next, we choose to decompose q(z t−1 , z t | x ≤t ) as a belief over z t and a one-step smoothing distribution DISPLAYFORM3 .

We obtain the following belief-based ELBO for state-space models: DISPLAYFORM4 Both quantities p(z t−1 | x ≤t−1 ) and q(z t | x ≤t ) represent the belief state of the model at different times, so at this stage we approximate them with the same distribution DISPLAYFORM5 representing the belief state code for z t .

Similarly, we represent the smoothing posterior over z t−1 as q(z t−1 | z t , b t−1 , b t ).

We obtain the following loss: DISPLAYFORM6 We provide an intuition on the different terms of the ELBO in the next section.

The model derived in the previous section expresses a state model p(z t | z t−1 ) that describes how the state of the world evolves from one time step to the next.

However, in many applications, the relevant timescale for planning may not be the one at which we receive observations and execute simple actions.

Imagine for example planning for a trip abroad; the different steps involved (discussing travel options, choosing a destination, buying a ticket, packing a suitcase, going to the airport, and so on), all occur at vastly different time scales (potentially months in the future at the beginning of the trip, and days during the trip).

Certainly, making a plan for this situation does not involve making second-by-second decisions.

This suggests that we should look for models that can imagine future states directly, without going through all intermediate states.

Beyond planning, there are several other reasons that motivate modeling the future directly.

First, training signal coming from the future can be stronger than small changes happening between time steps.

Second, the behavior of the model should ideally be independent from the underlying temporal sub-sampling of the data, if the latter is an arbitrary choice.

Third, jumpy predictions can be computationally efficient; when predicting several steps into the future, there may be some intervals where the prediction is either easy (e.g. a ball moving straight), or the prediction is complex but does not affect later time steps -which Neitz et al. FORMULA1 call inconsequential chaos.

There is a number of research directions that consider temporal jumps.

Koutnik et al. FORMULA1 and BID11 consider recurrent neural network with skip connections, making it easier to bridge distant timesteps.

BID8 temporally sub-sample the data and build a jumpy model (for fixed jump size) of this data; but by doing so they also drop the information contained in the skipped observations.

Neitz et al. FORMULA1 and Jayaraman et al. FORMULA1 predict sequences with variable time-skips, by choosing as target the most predictable future frames.

They predict the observations directly without learning appropriate states, and only focus on nearly fully observed problems (and therefore do not need to learn a notion of belief state).

For more general problems, this is a fundamental limitation, as even if one could in principle learn a jumpy observation model p(x t+δ |x ≤t ), it cannot be used recursively (feeding x t+δ back to the RNN and predicting x t+δ+δ ).

This is because x t+δ does not capture the full state of the system and so we would be missing information from t to t + δ to fully characterize what happens after time t + δ.

In addition, x t+δ might not be appropriate even as target, because some important information can only be extracted from a number of frames (potentially arbitrarily separated), such as a behavior of an agent.

Motivated by the model derived in section 3, we extend sequential TD-VAE to exhibit time abstraction.

We start from the same assumptions and architectural form: there exists a sequence of states z 1 , . . .

, z T from which we can predict the observations x 1 , . . . , x T .

A forward RNN encodes a belief state b t from past observations x ≤t .

The main difference is that, instead of relating information known at times t and t + 1 through the states z t and z t+1 , we relate two distant time steps t 1 and t 2 through their respective states z t1 and z t2 , and we learn a jumpy, state-to-state model p(z t2 | z t1 ) between z t1 and z t2 .

Following equation 5, the negative loss for the TD-VAE model is: DISPLAYFORM0 To train this model, one should choose the distribution of times t 1 , t 2 ; for instance, t 1 can be chosen uniformly from the sequence, and t 2 − t 1 uniformly over some finite range [1, D]; other approaches could be investigated.

FIG2 describes in detail the computation flow of the model.

Finally, it would be desirable to model the world with different hierarchies of state, the higher-level states predicting the same-level or lower-level states, and ideally representing more invariant or abstract information.

For this reason, we also develop stacked (hierarchical) version of TD-VAE, which uses several layers of latent states.

Hierarchical TD-VAE is detailed in the appendix.

In this section, we provide a more intuitive explanation behind the computation and loss of the model.

Assume we want to predict a future time step t 2 from all the information we have up until time t 1 .

All relevant information up until time t 1 (respectively t 2 ) has been compressed into a code b t1 (respectively b t2 ).

We make an observation x t of the world 2 at every time step t, but posit the existence of a state z t which fully captures the full condition of the world at time t.

Consider an agent at the current time t 2 .

At that time, the agent can make a guess of what the state of the world is by sampling from its belief model p B (z t2 | b t2 ).

Because the state z t2 should entail the corresponding observation x t2 , the agent aims to maximize p(x t2 | z t2 ) (first term of the loss), with a variational bottleneck penalty − log p(z t2 | b t2 ) (second term of the loss) to prevent too much information from the current observation x t2 from being encoded into z t2 .

Then follows the question 'could the state of the world at time t 2 have been predicted from the state of the world at time t 1 ?'.

In order to ascertain this, the agent must estimate the state of the world at time t 1 .

By time t 2 , the agent has aggregated observations between t 1 and t 2 that are informative about the state of the world at time t 1 , which, together with the current guess of the state of the world z t2 , can be used to form an ex post guess of the state of the world.

This is done by computing a smoothing distribution q(z t1 |z t2 , b t1 , b t2 ) and drawing a corresponding sample z t1 .

Having guessed states of the world z t1 and z t2 , the agent optimizes its predictive jumpy model of the world state p(z t2 | z t1 ) (third term of the loss).

Finally, it should attempt to see how predictable the revealed information was, or in other words, to assess whether the smoothing distribution q(z t1 | z t2 , b t2 ) could have been predicted from information only available at time t 1 (this is indirectly predicting z t2 from the state of knowledge b t1 at time t 1 -the problem we started with).

The agent can do so by minimizing the KL between the smoothing distribution and the belief distribution at time DISPLAYFORM0 (fourth term of the loss).

Summing all the losses described so far, we obtain the TD-VAE loss.

In reinforcement learning, the state of an agent represents a belief about the sum of discounted rewards R t = τ r t+τ γ τ .

In the classic setting, the agent only models the mean of this distribution represented by the value function V t or action dependent Q-function Q a t (Sutton & Barto, 1998) .

Recently in (Bellemare et al., 2017), a full distribution over R t has been considered.

To estimate V t1 or Q a t1 at time t 1 , one does not usually wait to get all the rewards to compute R t1 .

Instead, one uses an estimate at some future time t 2 as a bootstrap to estimate V t1 or Q a t1 (temporal difference).

In our case, the model expresses a belief p B (z t | b t ) about possible future states instead of the sum of discounted rewards.

The model trains the belief p B (z t1 | b t1 ) at time t 1 using belief p B (z t2 | b t2 ) at some time t 2 in the future.

It accomplishes this by (variationally) auto-encoding a sample z t2 of the future state into a sample z t1 , using the approximate posterior distribution q(z t1 | z t2 , b t1 , b t2 ) and the decoding distribution p(z t2 | z t1 ).

This auto-encoding mapping translates between states at t 1 and t 2 , forcing beliefs at the two time steps to be consistent.

Sample z t1 forms the target for training the belief p B (z t1 | b t1 ), which appears as a prior distribution over z t1 .

The first experiment using sequential TD-VAE, which enables a direct comparison to related algorithms for training state-space models.

Subsequent experiments use the full TD-VAE model.

We use a partially observed version of the MiniPacman environment (Racanière et al., 2017), shown in FIG0 .

The agent (Pacman) navigates a maze, and tries to eat all the food while avoiding being eaten by a ghost.

Pacman sees only a 5 × 5 window around itself.

To achieve a high score, the agent needs to form a belief state that captures memory of past experience (e.g. which parts of the maze have been visited) and uncertainty on the environment (e.g. where the ghost might be).We evaluate the performance of sequential (non-jumpy) TD-VAE on the task of modeling a sequence of the agent's observations.

We compare it with two state-space models trained using the standard ELBO of equation 1:• A filtering model with encoder q(z | x) = t q(z t | z t−1 , b t ), where b t = RNN(b t−1 , x t ).•

A mean-field model with encoder q(z | x) = t q(z t | b t ), where b t = RNN(b t−1 , x t ).

FIG0 shows the ELBO and estimated negative log probability on a test set of MiniPacman sequences for each model.

TD-VAE outperforms both baselines, whereas the mean-field model is the least well-performing.

We note that b t is a belief state for the mean-field model, but not for the filtering model; the encoder of the latter explicitly depends on the previous latent state z t−1 , hence b t ELBO and estimated negative log probability on a test set of MiniPacman sequences.

Lower is better.

Log probability is estimated using importance sampling with the encoder as proposal.

is not its sufficient statistics.

This comparison shows that naively restricting the encoder in order to obtain a belief state hurts the performance significantly; TD-VAE overcomes this difficulty.

In this experiment, we show that the model is able to learn the state and roll forward in jumps.

We consider sequences of length 20 of images of MNIST digits.

For each sequence, a random digit from the dataset is chosen, as well as the direction of movement (left or right).

At each time step, the digit moves by one pixel in the chosen direction, as shown in FIG1 .

We train the model with t 1 and t 2 separated by a random amount t 2 − t 1 from the interval [1, 4] .

We would like to see whether the model at a given time can roll out a simulated experience in time steps t 1 = t + δ 1 , t 2 = t 1 + δ 2 , . . .

with δ 1 , δ 2 , . . .

> 1, without considering the inputs in between these time points.

Note that it is not sufficient to predict the future inputs x t1 , . . .

as they do not contain information about whether the digit moves left or right.

We need to sample a state that contains this information.

We roll out a sequence from the model as follows: (a) b t is computed by the aggregation recurrent network from observations up to time t; (b) a state z t is sampled from p B (z t | b t ); (c) a sequence of states is rolled out by repeatedly sampling z ← z ∼ p(z | z) starting with z = z t ; (d) each z is decoded by p(x | z), producing a sequence of frames.

The resulting sequences are shown in FIG1 .

We see that indeed the model can roll forward the samples in steps of more than one elementary time step (the sampled digits move by more than one pixel) and that it preserves the direction of motion, demonstrating that it rolls forward a state.

We would like to demonstrate that the model can build a state even when little information is present in each observation, and that it can sample states far into the future.

For this we consider a 1D sequence obtained from a noisy harmonic oscillator, as shown in Figure 4 (first and fourth rows).

The frequencies, initial positions and initial velocities are chosen at random from some range.

At every update, noise is added to the position and the velocity of the oscillator, but the energy is approximately preserved.

The model observes a noisy version of the current position.

Attempting to predict the input, which consists of one value, 100 time steps in the future would be uninformative; such a Figure 4 : Skip-state prediction for 1D signal.

The input is generated by a noisy harmonic oscillator.

Rollouts consist of (a) a jumpy state transition with either dt = 20 or dt = 100, followed by 20 state transitions with dt = 1.

The model is able to create a state and predict it into the future, correctly predicting frequency and magnitude of the signal.prediction wouldn't reveal what the frequency or the magnitude of the signal is, and because the oscillator updates are noisy, the phase information would be nearly lost.

Instead, we should try to predict as much as possible about the state, which consists of frequency, magnitude and position, and it is only the position that cannot be accurately predicted.

The aggregation RNN is an LSTM; we use a hierarchical TD-VAE with two layers, where the latent variables in the higher layer are sampled first, and their results are passed to the lower layer.

The belief, smoothing and state-transition distributions are feed-forward networks, and the decoder simply extracts the first component from the z of the first layer.

We also feed the time interval t 2 − t 1 into the smoothing and state-transition distributions.

We train on sequences of length 200, with t 2 − t 1 taking values chosen at random from [1, 10] with probability 0.8 and from We analyze what the model has learned as follows.

We pick time t 1 = 60 and sample z t1 ∼ p B (z t1 | b t1 ).

Then, we choose a time interval δ t ∈ {20, 100} to skip, sample from the forward model p(z 2 | z 1 , δ t ) to obtain z t2 at t 2 = t 1 + δ t .

To see the content of this state, we roll forward 20 times with time step δ = 1 and plot the result, shown in Figure 4 .

We see that indeed the state z t2 is predicted correctly, containing the correct frequency and magnitude of the signal.

We also see that the position (phase) is predicted well for dt = 20 and less accurately for dt = 100 (at which point the noisiness of the system makes it unpredictable).Finally, we show that TD-VAE training can improve the quality of the belief state.

For this experiment, the harmonic oscillator has a different frequency in each interval [0, 10), [10, 20) , [20, 120) , [120, 140) .

The first three frequencies f 1 , f 2 , f 3 are chosen at random.

The final frequency f 4 is chosen to be one fixed value f a if f 1 > f 2 and another fixed value f b otherwise (f a and f b are constants).

In order to correctly model the signal in the final time interval, the model needs to learn the relation between f 1 and f 2 , store it over length of 100 steps, and apply it over a number of time steps (due to the noise) in the final interval.

To test whether the belief state contains the information about this relationship, we train a binary classifier from the belief state to the final frequency f 4 at points just before the final interval.

We compare two models with the same recurrent architecture (an LSTM), but trained with different objective: next-step prediction vs TD-VAE loss.

The figure on the right shows the classification accuracy for the two methods, averaged over 20 runs.

We found that the longer the separating time interval (containing frequency f 3 ) and the smaller the size of the LSTM, the better TD-VAE is compared to next-step predictor.

In the final experiment, we analyze the model on a more visually complex domain.

We use sequences of frames seen by an agent solving tasks in the DeepMind Lab environment BID5 .

We aim to demonstrate that the model holds explicit beliefs about various possible futures, and that it can roll out in jumps.

We suggest functional forms inspired by convolutional DRAW: we use convolutional LSTMs for all the circles in FIG5 and make the model 16 layers deep (except for the forward updating LSTMs which are fully connected with depth 4).We use time skips t 2 − t 1 sampled uniformly from [1, 40] and analyze the content of the belief state b. We take three samples z 1 , z 2 , z 3 from p B (z | b), which should represent three instances of possible futures.

FIG3 (left) shows that they decode to roughly the same frame.

To see what they represent about the future, we draw 5 samples z k i ∼ p(ẑ | z), k = 1, . . .

, 5 and decode them, as shown in FIG3 (right).

We see that for a given i, the predicted samples decode to similar frames (images in the same row).

However z's for different i's decode to different frames.

This means b represented a belief about several different possible futures, while different z i each represent a single possible future.

Finally, we show what rollouts look like.

We train on time separations t 2 − t 1 chosen uniformly from [1, 5] on a task where the agent tends to move forward and rotate.

FIG4 shows 4 rollouts from the model.

We see that the motion appears to go forward and into corridors and that it skips several time steps (real single step motion is slower).

In this paper, we argued that an agent needs a model that is different from an accurate step-by-step environment simulator.

We discussed the requirements for such a model, and presented TD-VAE, a sequence model that satisfies all requirements.

TD-VAE builds states from observations by bridging time points separated by random intervals.

This allows the states to relate to each other directly over longer time stretches and explicitly encode the future.

Further, it allows rolling out in state-space and in time steps larger than, and potentially independent of, the underlying temporal environment/data step size.

In the future, we aim to apply TD-VAE to more complex settings, and investigate a number of possible uses in reinforcement learning such are representation learning and planning.

In section 3, we derive an approximate ELBO which forms the basis of the training loss of the one-step TD-VAE.

One may wonder whether a similar idea may underpin the training loss of the jumpy TD-VAE.

Here we show how to modify the derivation to provide an approximate ELBO for a slightly different training regime.

Assume a sequence (x 1 , . . .

, x T ), and an arbitrary distribution S over subsequences x s = (x t1 , . . .

, x tn ) of x. For each time index t i , we suppose a state z ti , and model the subsequence x s with a jumpy state-space model p(x s ) = i p(z ti |z ti−1 )p(x ti |z ti ); denote z s = (z t1 , . . .

, z tn ) the state subsequence.

We use the exact same machinery as the next-step ELBO, except that we enrich the posterior distribution over z s by making it depend not only on observation subsequence x s , but on the entire sequence x. This is possible because posterior distributions can have arbitrary contexts; the observations which are part of x but not x s effectively serve as auxiliary variable for a stronger posterior.

We use the full sequence x to form a sequence of belief states b t at all time steps.

We use in particular the ones computed at the subsampled times t i .

By following the same derivation as the one-step TD-VAE, we obtain: DISPLAYFORM0 which, using the same belief approximations as the next step TD-VAE, becomes: DISPLAYFORM1 which is the same loss as the TD-VAE for a particular choice of the sampling scheme S (only sampling pairs).

In this section we start with a general recurrent variational auto-encoder and consider how the desired properties detailed in sections 1 and 2 constrain the architecture.

We will find that these constraints in fact naturally lead to the TD-VAE model.

Let us first consider a relatively general form of temporal variational auto-encoder.

We consider recurrent models where the same module is applied at every step, and where outputs are sampled one at a time (so that arbitrarily long sequences can be generated).

A very general form of such an architecture consist of forward-backward encoder RNNs and a forward decoder RNN (Figure 7 ) but otherwise allowing for all the connections.

Several works BID10 Lee et al., 2018; BID1 BID12 Liu et al., 2017; BID14 BID8 Serban et al., 2017) fall into this framework.

Now let us consider our desired properties.

In order to sample forward in latent space, the encoder must not feed into the decoder or the prior of the latent variables, since observations are required to compute the encoded state, and we would therefore require the sampled observations to compute the distribution over future states and observations.

We next consider the constraint of computing a belief state b t .

The belief state b t represents the state of knowledge up to time t, and therefore cannot receive an input from the backwards decoder.

Figure 7 : Recurrent variational auto-encoder.

General recurrent variational auto-encoder, obtained by imposing recurrent structure, forward sampling and allowing all potential connections.

Note that the encoder can have several alternating layers of forward and backward RNNs.

Also note that the connection 1 has to be absent if the backwards encoder is used.

Possible skip connections are not shown as they can directly be implemented in the RNN weights.

If connections 2 are absent, the model is capable of forward sampling in latent space without going back to observations.

Furthermore, b t should have an unrestricted access to information; it should ideally not be disturbed by sampling (two identical agents with the same information should compute the same information; this will not be the case if the computation involves sampling), nor go through information bottlenecks.

This suggests using the forward encoder for computing the belief state.

This prevents running the backwards inference from the end of the sequence.

However if we assume that p B represents our best belief about the future, we can take a sample from it as an instance of the future: z t2 ∼ p B (z t2 |b t2 ).

It forms a type of bootstrap information.

Then we can go backwards and infer what would the world have looked like given this future (e.g. the object B was still in the box even if we don't see it).

Using VAE training, we sample z 1 from its posterior q(z t1 |z t2 , b t2 , b t1 ) (the conditioning variables are the ones we have available locally), using p B (z t1 |b t1 ) as prior.

Conversely, for t 2 , we sample from p B (z t2 |b t2 ) as posterior, but with p(z t2 |z t1 ) as prior.

We therefore obtain the VAE losses log q(z 1 |z 2 , s 1 , s 2 ) − log p B (z 1 |s 1 ) at t 1 and log p B (z 2 |s 2 ) − log p P (z 2 |z 1 ) at t 2 .

In addition we have the reconstruction term p D (x 2 |z 2 ) that grounds the latent in the input.

The whole algorithm is presented in the FIG2

In the main paper we detailed a framework for learning models by bridging two temporally separated time points.

It would be desirable to model the world with different hierarchies of state, the higherlevel states predicting the same-level or lower-level states, and ideally representing more invariant or abstract information.

In this section we describe a stacked (hierarchical) version of the model.

The first part to extend to L layers is the RNN that aggregates observations to produce the belief state b. Here we simply use a deep LSTM, but with layer l receiving inputs also from layer l + 1 from the previous time step.

This is so that the higher layers can influence the lower ones (and vice versa). and setting b 0 = b L and b L+1 = ∅.We create a deep version of the belief part of the model by stacking the shallow one, as shown in FIG5 .

In the usual spirit of deep directed models, the model samples downwards, generating higher level representations before the lower level ones (closer to pixels).

The model implements deep inference, that is, the posterior distribution of one layer depends on the samples from the posterior distribution in previously sampled layers.

The order of inference is a design choice, and we use the same direction as that of generation, from higher to lower layers, as done for example by BID16 ; Kingma et al. (2016); Rasmus et al. (2015) .

We implement the dependence of various distributions on latent variables sampled so far using a recurrent neural network that summarizes all such variables (in a given group of distributions).

We don't share the weights between different layers.

Given these choices, we can allow all connections consistent with the model.

Next we describe the functional forms used in our model.= KL(q DISPLAYFORM0 The hidden layer of the D maps is 50; the size of each z l t is 8.

Belief states have size 50.

We use the Adam optimizer with learning rate 0.0005.The same network works for the MNIST experiment with the following modifications.

Observations are pre-processed by a two hidden layer MLP with ReLU nonlinearity.

The decoder p D also have a two layer MLP, which outputs the logits of a Bernoulli distribution.

δ t was not passed as input to any network.

<|TLDR|>

@highlight

Generative model of temporal data, that builds online belief state, operates in latent space, does jumpy predictions and rollouts of states.