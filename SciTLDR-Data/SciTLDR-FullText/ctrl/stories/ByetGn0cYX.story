In this work, we propose a novel formulation of planning which views it as a probabilistic inference problem over future optimal trajectories.

This enables us to use sampling methods, and thus, tackle planning in continuous domains using a fixed computational budget.

We design a new algorithm,  Sequential Monte Carlo Planning, by leveraging classical methods in Sequential Monte Carlo and Bayesian smoothing in the context of control as inference.

Furthermore, we show that Sequential Monte Carlo Planning can capture multimodal policies and can quickly learn continuous control tasks.

To exhibit intelligent behaviour machine learning agents must be able to learn quickly, predict the consequences of their actions, and explain how they will react in a given situation.

These abilities are best achieved when the agent efficiently uses a model of the world to plan future actions.

To date, planning algorithms have yielded very impressive results.

For instance, Alpha Go BID36 relied on Monte Carlo Tree Search (MCTS) BID23 ) to achieve super human performances.

Cross entropy methods (CEM) BID34 have enabled robots to perform complex nonprehensile manipulations BID11 and algorithms to play successfully Tetris BID39 .

In addition, iterative linear quadratic regulator (iLQR) BID21 BID20 BID41 enabled humanoid robots tasks to get up from an arbitrary seated pose .Despite these successes, these algorithms make strong underlying assumptions about the environment.

First, MCTS requires a discrete setting, limiting most of its successes to discrete games with known dynamics.

Second, CEM assumes the distribution over future trajectories to be Gaussian, i.e. unimodal.

Third, iLQR assumes that the dynamics are locally linear-Gaussian, which is a strong assumption on the dynamics and would also assume the distribution over future optimal trajectories to be Gaussian.

For these reasons, planning remains an open problem in environments with continuous actions and complex dynamics.

In this paper, we address the limitations of the aforementioned planning algorithms by creating a more general view of planning that can leverage advances in deep learning (DL) and probabilistic inference methods.

This allows us to approximate arbitrary complicated distributions over trajectories with non-linear dynamics.

We frame planning as density estimation problem over optimal future trajectories in the context of control as inference BID10 BID45 BID43 Rawlik et al., 2010; BID47 BID31 .

This perspective allows us to make use of tools from the inference research community and, as previously mentioned, model any distribution over future trajectories.

The planning distribution is complex since trajectories consist of an intertwined sequence of states and actions.

Sequential Monte Carlo (SMC) BID38 BID13 BID27 methods are flexible and efficient to model such a T −1 t≥1 p env (s t+1 |s t , a t ) T t≥1 π θ (a t |s t ) denotes the probability of a trajectory x 1:T under policy π θ .

FIG8 .1: O t is an observed optimality variable with probability p(O t |s t , a t ) = exp(r(s t , a t )).x t = (s t , a t ) are the state-action pair variables considered here as latent.

Traditionally, in reinforcement learning (RL) problems, the goal is to find the optimal policy that maximizes the expected return E q θ [ T t=1 γ t r t ].

However, it is useful to frame RL as an inference problem within a probabilistic graphical framework BID33 BID45 BID30 .

First, we introduce an auxiliary binary random variable O t denoting the "optimality" of a pair (s t , a t ) at time t and define its probability 1 as p(O t = 1|s t , a t ) = exp(r(s t , a t )).

O is a convenience variable only here for the sake of modeling.

By considering the variables (s t , a t ) as latent and O t as observed, we can construct a Hidden Markov Model (HMM) as depicted in figure 2.1.

Notice that the link s → a is not present in figure 2.1 as the dependency of the optimal action on the state depends on the future observations.

In this graphical model, the optimal policy is expressed as p(a t |s t , O t:T ).The posterior probability of this graphical model can be written as 2 : DISPLAYFORM0 r(s t , a t ) + log p(a t ) .(2.1)It appears clearly that finding optimal trajectories is equivalent to finding plausible trajectories yielding a high return.1 as in BID30 , if the rewards are bounded above, we can always remove a constant so that the probability is well defined.2 Notice that in the rest of the paper, we will abusively remove the product of the action priors T t=1 p(at) = exp T t=1 log p(at) from the joint as in BID30 .

We typically consider this term either constant or already included in the reward function.

See Appendix A.2 for details.

Many control as inference methods can be seen as approximating the density by optimizing its variational lower bound: BID43 .

Instead of directly differentiating the variational lower bound for the whole trajectory, it is possible to take a message passing approach such as the one used in Soft Actor-Critic (SAC) BID17 and directly estimate the optimal policy p(a t |s t , O t:T ) using the backward message, i.e a soft Q function instead of the Monte Carlo return.

DISPLAYFORM1

Since distributions over trajectories are complex, it is often difficult or impossible to directly draw samples from them.

Fortunately in statistics, there are successful strategies for drawing samples from complex sequential distributions, such as SMC methods.

For simplicity, in the remainder of this section we will overload the notation and refer to the target distribution as p(x) and the proposal distribution as q(x).

We wish to draw samples from p but we only know its unnormalized density.

We will use the proposal q to draw samples and estimate p.

In the next section, we will define the distributions p and q in the context of planning.

Importance sampling (IS): When x can be efficiently sampled from another simpler distribution q i.e. the proposal distribution, we can estimate the likelihood of any point x under p straightforwardly by computing the unnormalized importance sampling weights w(x) ∝ p(x) q(x) and using the identity DISPLAYFORM0 is defined as the normalized importance sampling weights.

In practice, one draws N samples from q: {x (n) } N n=1 ∼ q; these are referred to as particles.

The set of particles {x (n) } N n=1 associated with their weights {w (n) } N n=1 are simulations of samples from p. That is, we approximate the density p with a weighted sum of diracs from samples of q: DISPLAYFORM1 , with x (n) sampled from q where δ x0 (x) denotes the Dirac delta mass located as x 0 .Sequential Importance Sampling (SIS): When our problem is sequential in nature x = x 1:T , sampling x 1:T at once can be a challenging or even intractable task.

By exploiting the sequential structure, the unnormalized weights can be updated iteratively in an efficient manner: w t (x 1:t ) = w t−1 (x 1:t−1 ) p(xt|x1:t−1) q(xt|x1:t−1) .

We call this the update step.

This enables us to sample sequentially x t ∼ q(x t |x 1:t−1 ) to finally obtain the set of particles {x Sequential Importance Resampling (SIR): When the horizon T is long, samples from q usually have a low likelihood under p, and thus the quality of our approximation decreases exponentially with T .

More concretely, the unnormalized weights w (n) t converge to 0 with t → ∞. This usually causes the normalized weight distribution to degenerate, with one weight having a mass of 1 and the others a mass of 0.

This phenomenon is known as weight impoverishment.

One way to address weight impoverishment is to add a resampling step where each particle is stochastically resampled to higher likelihood regions at each time step.

This can typically reduce the variance of the estimation from growing exponentially with t to growing linearly.

In the context of control as inference, it is natural to see planning as the act of approximating a distribution of optimal future trajectories via simulation.

In order to plan, an agent must possess a model of the world that can accurately capture consequences of its actions.

In cases where multiple trajectories have the potential of being optimal, the agent must rationally partition its computational resources to explore each possibility.

Given finite time, the agent must limit its planning to a finite horizon h. We, therefore, define planning as the act of approximating the optimal distribution over trajectories of length h. In the control-as-inference framework, this distribution is naturally expressed as p(a 1 , s 2 , . . .

s h , a h |O 1:T , s 1 ), where s 1 represents our current state.

As we consider the current state s 1 given, it is equivalent and convenient to focus on the planning distribution with horizon h: p(x 1:h |O 1:T ).

Bayesian smoothing is an approach to the problem of estimating the distribution of a latent variable conditioned on all past and future observations.

One method to perform smoothing is to decompose the posterior with the two-filter formula BID4 BID26 : DISPLAYFORM0 This corresponds to a forward-backward messages factorization in a Hidden Markov Model as depicted in figure 3.1.

We broadly underline in orange forward variables and in blue backward variables in the rest of this section.

DISPLAYFORM1 Figure 3.1: Factorization of the HMM into forward (orange) and backward (blue) messages.

Estimating the forward message is filtering, estimating the value of the latent knowing all the observations is smoothing.

Filtering is the task of estimating p(x 1:t |O 1:t ): the probability of a latent variable conditioned on all past observations.

In contrast, smoothing estimates p(x 1:t |O 1:T ): the density of a latent variable conditioned on all the past and future measurements.

In the belief propagation algorithm for HMMs, these probabilities correspond to the forward message α h (x h ) = p(x 1:h |O 1:h ) and backward message β h (x h ) = p(O h+1:T |x h ) , both of which are computed recursively.

While in discrete spaces these forward and backward messages can be estimated using the sumproduct algorithm, its complexity scales with the square of the space dimension making it unsuitable for continuous tasks.

We will now devise efficient strategies for estimating reliably the full posterior using the SMC methods covered in section 2.2.

The backward message p(O h+1:T |x h ) can be understood as the answer to: "What is the probability of following an optimal trajectory from the next time step on until the end of the episode, given my current state?".

Importantly, this term is closely related to the notion of value function in RL.

Indeed, in the control-as-inference framework, the state-and action-value functions are defined as DISPLAYFORM0 T |s h , a h ) respectively.

They are solutions of a soft-Bellman equation that differs a little from the traditional Bellman equation (O'Donoghue et al., 2016; Nachum et al., 2017; BID35 BID0 .

A more in depth explanation can be found in BID30 .

We can show subsequently that: DISPLAYFORM1 Full details can be found in Appendix A.3.

Estimating the backward message is then equivalent to learning a value function.

This value function as defined here is the same one used in Maximum Entropy RL (Ziebart, 2010).

Using the results of the previous subsections we can now derive the full update of the sequential importance sampling weights.

To be consistent with the terminology of section 2.2, we call p(x 1:h |O 1:T ) the target distribution and q θ (x 1:h ) the proposal distribution.

The sequential weight update formula is in our case: DISPLAYFORM0 3) is akin to a maximum entropy advantage function.

The change in weight can be interpreted as sequentially correcting our expectation of the return of a trajectory.

The full derivation is available in Appendix A.4.

Our algorithm is similar to the Auxilliary Particle Filter (Pitt & Shephard, 1999) which uses a one look ahead simulation step to update the weights.

Note that we have assumed that our model of the environment was perfect to obtain this slightly simplified form.

This assumption is made by most planning algorithms (LQR, CEM . . .

): it entails that our plan is only as good as our model is.

A typical way to mitigate this issue and be more robust to model errors is to re-plan at each time step; this technique is called Model Predictive Control (MPC) and is commonplace in control theory.

We can now use the computations of previous subsections to derive the full algorithm.

We consider the root state of the planning to be the current state s t .

We aim at building a set of particles {x DISPLAYFORM0 and their weights {w DISPLAYFORM1 representative of the planning density p(x t:t+h |O 1:T ) over optimal trajectories.

We use SAC BID17 for the policy and value function, but any other Maximum Entropy policy can be used for the proposal distribution.

Note that we used the value function estimated by SAC as a proxy the optimal one as it is usually done by actor critic methods.

Algorithm 1 SMC Planning using SIR 1: for t in {1, . . . , T } do 2: DISPLAYFORM2 3: DISPLAYFORM3 for i in {t, . . .

, t + h} do

// Update 6: DISPLAYFORM0 7: DISPLAYFORM1 8: DISPLAYFORM2 // Resampling 10: DISPLAYFORM3 12: end for

Sample n ∼ Uniform(1, N ).14: We summarize the proposed algorithm in Algorithm 1.

At each step, we sample from the proposal distribution or model-free agent (line 6) and use our learned model to sample the next state and reward (line 7).

We then update the weights (line 8).

In practice we only use one sample to estimate the expectations, thus we may incur a small bias.

The resampling step is then performed (line 10-11) by resampling the trajectories according to their weight.

After the planning horizon is reached, we sample one of our trajectories (line 13) and execute its first action into the environment (line 15-16).

The observations (s t , a t , r t , s t+1 ) are then collected and added to a buffer (line 17) used to train the model as well as the policy and value function of the model-free agent.

An alternative algorithm that does not use the resampling step (SIS) is highlighted in Algorithm 2 in Appendix A.6.

DISPLAYFORM0 A schematic view of the algorithm can also be found on figure 3.2.

We now discuss shortcomings our approach to planning as inference may suffer from, namely encouraging risk seeking policies.

DISPLAYFORM0 we have the root white node st−1, the actions a (n) t−1 are black nodes and the leaf nodes are the s (n) t .

We have one particle on the leftmost branch, two on the central branch and one on the rightmost branch.• In each tree, the white nodes represent states and black nodes represent actions.

Each bullet point near a state represents a particle, meaning that this particle contains the total trajectory of the branch.

The root of the tree represents the root planning state, we expand the tree downward when planning.

Bias in the objective: Trajectories having a high likelihood under the posterior defined in Equation 2.1 are not necessarily trajectories yielding a high mean return.

Indeed, as log E p exp R(x) ≥ E p R(x) we can see that the objective function we maximize is an upper bound on the quantity of interest: the mean return.

This can lead to seeking risky trajectories as one very good outcome in log E exp could dominate all the other potentially very low outcomes, even if they might happen more frequently.

This fact is alleviated when the dynamics of the environment are close to deterministic BID30 .

Thus, this bias does not appear to be very detrimental to us in our experiments 4 as our environments are fairly close to deterministic.

The bias in the objective also appears in many control as inference works such as Particle Value Functions (Maddison et al., 2017a) and the probabilistic version of LQR proposed in BID43 .Bias in the model: A distinct but closely related problem arises when one trains jointly the policy π θ and the model p model , i.e if q(x 1:T ) is directly trained to approximate p(x 1:T |O 1:T ).

In that case, p model (s t+1 |s t , a t ) will not approximate p env (s t+1 |s t , a t ) but p env (s t+1 |s t , a t , O t:T ) BID30 .

This means the model we learn has an optimism bias and learns transitions that are overly optimistic and do no match the environment's behavior.

This issue is simply solved by training the model separately from the policy, on transition data contained in a buffer as seen on line 18 of Algorithm 1.

In this section, we show how SMCP can deal with multimodal policies when planning.

We believe multimodality is useful for exploring since it allows us to keep a distribution over many promising trajectories and also allows us to adapt to changes in the environment e.g. if a path is suddenly blocked.

We applied two version of SMCP: i) with a resampling step (SIR) ii) without a resampling step (SIS) and compare it to CEM on a simple 2D point mass environment 4.1.

Here, the agent can control the displacement on (x, y) within the square [0, 1] 2 , a = (∆x, ∆y) with maximum magnitude ||a|| = 0.05.

The starting position (•) of the agent is (x = 0, y = 0.5), while the goal ( ) is at g = (x = 1, y = 0.5).

The reward is the agent's relative closeness increment to the goal: (a) Sequential Importance Resampling (SIR): when resampling the trajectories at each time step, the agent is able to focus on the promising trajectories and does not collapse on a single mode.(b) Sequential Importance Sampling (SIS): if we do not perform the resampling step the agent spends most of its computation on uninteresting trajectories and was not able to explore as well.(c) CEM: here the agent samples all the actions at once from a Gaussian with learned mean and covariance.

We needed to update the parameters 50 times for the agent to find one solution, but it forgot the other one.

The proposal distribution is taken to be an isotropic gaussian.

Here we plot the planning distribution imagined at t = 0 for three different agents.

A darker shade of blue indicates a higher likelihood of the trajectory.

Only the agent using Sequential Importance Resampling was able to find good trajectories while not collapsing on a single mode.

DISPLAYFORM0 ||st−g|| 2 .

However, there is a partial wall at the centre of the square leading to two optimal trajectories, one choosing the path below the wall and one choosing the path above.

The proposal is an isotropic normal distribution for each planning algorithm, and since the environment's dynamics are known, there is no need for learning: the only difference between the three methods is how they handle planning.

We also set the value function to 0 for SIR and SIS as we do not wish to perform any learning.

We used 1500 particles for each method, and updated the parameters of CEM until convergence.

Our experiment 4.1 shows how having particles can deal with multimodality and how the resampling step can help to focus on the most promising trajectories.

The experiments were conducted on the Open AI Gym Mujoco benchmark suite BID5 .

To understand how planning can increase the learning speed of RL agents we focus on the 250000 first time steps.

The Mujoco environments provide a complex benchmark with continuous states and actions that requires exploration in order to achieve state-of-the-art performances.

The environment model used for our planning algorithm is the same as the probabilistic neural network used by BID8 , it minimizes a gaussian negative log-likelihood model: DISPLAYFORM0 where Σ θ is diagonal and the transitions (s n , a n , s n+1 ) are obtained from the environment.

We added more details about the architecture and the hyperparameters in the appendix A.5.We included two popular planning algorithms on Mujoco as baselines: CEM BID8 and Random Shooting (RS) (Nagabandi et al., 2017) .

Furthermore, we included SAC BID17 , a model free RL algorithm, since i) it has currently one of the highest performances on Mujoco tasks, which make it a very strong baseline, and ii) it is a component of our algorithm, as we use it as a proposal distribution in the planning phase.

Our results suggest that SMCP does not learn as fast as CEM and RS initially as it heavily relies on estimating a good value function.

However, SMCP quickly achieves higher performances than CEM and RS.

SMCP also learns faster than SAC because it was able to leverage information from the model early in training.

Note that our results differ slightly from the results usually found in the model-based RL literature.

This is because we are tackling a more difficult problem: estimating the transitions and the reward function.

We are using unmodified versions of the environments which introduces many hurdles.

For instance, the reward function is challenging to learn from the state and very noisy.

Usually, the environments are modified such that their reward can be computed directly from the state e.g. BID8 3 .As in BID18 , we assess the significance of our results by running each algorithm with multiple seeds (20 random seeds in our case, from seed 0 to seed 19) and we perform a statistical significance test following BID9 .

We test the hypothesis that our mean return on the last 100k steps is higher than the one obtained by SAC.

Our results are significant to the 5% for HalfCheetah and Walker2d.

See Appendix A.7 for additional details.

We also report some additional experimental results such as effective sample size and model loss in Appendix A.8.

Planning as inference: Seeing planning as an inference problem has been explored in cognitive neuroscience by BID3 and BID37 .

While shedding light on how Bayesian inference could be used in animal and human reasoning, it does not lead to a practical algorithm usable in complex environments.

In the reinforcement learning literature, we are only aware of Attias (2003) and BID44 that initially framed planning as an inference problem.

However, both works make simplifying assumptions on the dynamics and do not attempt to capture the full posterior distribution.

In the control theory literature, particle filters are usually used for inferring the true state of the system which is then used for control BID1 .

BID22 also combined SMC and MPC methods.

While their algorithm is similar to ours, the distribution they approximate is not the Bayesian posterior, but a distribution which converges to a Dirac on the best trajectory.

More recently, BID28 achieved promising results on a rope manipulation task using generative adversarial network BID12 to generate future trajectories.

Model based RL: Recent work has been done in order to improve environment modeling and account for different type of uncertainties.

BID8 compared the performance of models that account for both aleatoric and epistemic uncertainties by using an ensemble of probabilistic models.

BID16 combined the variational autoencoder BID25 ) and a LSTM BID19 to model the world.

BID6 used a model to improve the target for temporal difference (TD) learning.

Note that this line of work is complementary to ours as SMCP could make use of such models.

Other works have been conducted in order to directly learn how to use a model BID15 BID46 BID7 .Particle methods and variational inference: BID14 learn a good proposal distribution for SMC methods by minimizing the KL divergence with the optimal proposal.

It is conceptually similar to the way we use SAC BID17 but it instead minimizes the reverse KL to the optimal proposal.

Further works have combined SMC methods and variational inference (Naesseth et al., 2017; Maddison et al., 2017b; BID29 to obtain lower variance estimates of the distribution of interest.

In this work, we have introduced a connection between planning and inference and showed how we can exploit advances in deep learning and probabilistic inference to design a new efficient and theoretically grounded planning algorithm.

We additionally proposed a natural way to combine model-free and model-based reinforcement learning for planning based on the SMC perspective.

We empirically demonstrated that our method achieves state of the art results on Mujoco.

Our result suggest that planning can lead to faster learning in control tasks.

However, our particle-based inference method suffers some several shortcomings.

First, we need many particles to build a good approximation of the posterior, and this can be computationally expensive since it requires to perform a forward pass of the policy, the value function and the model for every particle.

Second, resampling can also have adverse effects, for instance all the particles could be resampled on the most likely particle, leading to a particle degeneracy.

More advanced SMC methods dealing with this issue such as backward simulation BID32 or Particle Gibbs with Ancestor Sampling (PGAS) (Lindsten et al., 2014) have been proposed and using them would certainly improve our results.

Another issue we did not tackle in our work is the use of models of the environment learned from data.

Imperfect model are known to result in compounding errors for prediction over long sequences.

We chose to re-plan at each time step (Model Predictive Control) as it is often done in control to be more robust to model errors.

More powerful models or uncertainty modeling techniques can also be used to improve the accuracy of our planning algorithm.

While the inference and modeling techniques used here could be improved in multiple ways, SMCP achieved impressive learning speed on complex control tasks.

The planning as inference framework proposed in this work is general and could serve as a stepping stone for further work combining probabilistic inference and deep reinforcement learning.

A.1 ABBREVIATION AND NOTATION p(x) Density of interest.

Approximation of the density of interest.t ∈ {1, . . .

T } time steps.n ∈ {1, . . .

N } particle number.h horizon length.

The true joint distribution 2.1 in section 2.1 should actually be written: DISPLAYFORM0 In Mujoco environments, the reward is typically written as DISPLAYFORM1 where f is a function of the state (velocity for HalfCheetah on Mujoco for example).

The part α||a t || 2 2can be seen as the contribution from the action prior (here a gaussian prior).

One can also consider the prior to be constant (and potentially improper) so that is does not change the posterior p(x 1:T |O 1:T ).

DISPLAYFORM2 By definition of the optimal value function in BID30 .

DISPLAYFORM3 We use there the forward-backward equation 3.1 for the numerator and the denominator DISPLAYFORM4 A.5 EXPERIMENT DETAILS Random samples: 1000 transitions are initially collected by a random policy to pretrain the model and the proposal distribution.

After which the agents start following their respective policy.

Data preprocessing: We normalize the observations to have zero mean and standard deviation 1.

The model is used to predict the planning distribution for the horizon h of N particles.

We then sample a trajectory according to its weight and return the first action of this trajectory.

In our experiments, we fix the maximum number of particles for every method to 2500.

For SMCP, the temperature and horizon length are described in TAB2 .3.

We used a custom implementation with a Gaussian policy for both the SAC baseline and the proposal distribution used for both versions of SMCP.

We used Adam (Kingma & Ba, 2014) with a learning rate of 0.001.

The reward scaling suggested by BID17 for all experiments and used an implementation inspired by Pong (2018).

We used a two hidden layers with 256 hidden units for the three networks: the value function, the policy and the soft Q functions.

Model: We train the model p model to minimize the negative log likelihood of p(s t+1 |s t + ∆ t (s t , a t ), σ t (s t , a t )).

The exact architectures are detailed in TAB2 .3.

We train the model to predict the distribution of the change in states and learn a deterministic reward function from the current state and predict the change in state.

Additionally, we manually add a penalty on the action magnitude in the reward function to simplify the learning.

At the end of each episode we train the model for 10 epochs.

Since the training is fairly short, we stored every transitions into the buffer.

The model is defined as: DISPLAYFORM0 DISPLAYFORM1 7: DISPLAYFORM2 8: DISPLAYFORM3 9:end for 10:Sample n ∼ Categorical(w The significance of our results is done following guidelines from BID9 .

We test the hypothesis that the mean return of our method is superior to the one of SAC.

We use 20 random seeds (from 0 to 19pro) for each method on each environment.

For this we look at the average return from steps 150k to 250k for SIR-SAC and SAC, and conduct a Welch's t-test with unknown variance.

We report the p-value for each environment tested on Mujoco.

A p val < 0.05 usually indicates that there is strong evidence to suggest that our method outperforms SAC.• HalfCheetah-v2: p val = 0.003.

There is very compelling evidence suggesting we outperform SAC.• Hopper-v2: p val = 0.09.

There is no significant evidence suggesting we outperform SAC.• Walker2d-v2: p val = 0.03.

There is compelling evidence suggesting we outperform SAC.

A.8.1 EFFECTIVE SAMPLE SIZE More precisely the values are DISPLAYFORM0 where i is the depth of the planning, N is the number of particles and DISPLAYFORM1 We can see that as the proposal distribution improves the ESS also increases.

The ESS on HalfCheetah is representative of the one obtained on the other environments.

While these values are not high, we are still around 15% thus we do not suffer heavily from weight degeneracy.

We also report the negative log likelihood loss of the environment's model during the training on Figure A .2.

<|TLDR|>

@highlight

Leveraging control as inference and Sequential Monte Carlo methods, we proposed a probabilistic planning algorithm.