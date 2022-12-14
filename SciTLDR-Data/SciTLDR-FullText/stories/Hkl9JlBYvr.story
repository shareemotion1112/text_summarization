Trading off exploration and exploitation in an unknown environment is key to maximising expected return during learning.

A Bayes-optimal policy, which does so optimally, conditions its actions not only on the environment state but on the agent's uncertainty about the environment.

Computing a Bayes-optimal policy is however intractable for all but the smallest tasks.

In this paper, we introduce variational Bayes-Adaptive Deep RL (variBAD), a way to meta-learn to perform approximate inference in an unknown environment, and incorporate task uncertainty directly during action selection.

In a grid-world domain, we illustrate how variBAD performs structured online exploration as a function of task uncertainty.

We also evaluate variBAD on MuJoCo domains widely used in meta-RL and show that it achieves higher return during training than existing methods.

Reinforcement learning (RL) is typically concerned with finding an optimal policy that maximises expected return for a given Markov decision process (MDP) with an unknown reward and transition function.

If these were known, the optimal policy could in theory be computed without environment interactions.

By contrast, learning in an unknown environment usually requires trading off exploration (learning about the environment) and exploitation (taking promising actions).

Balancing this trade-off is key to maximising expected return during learning, which is desirable in many settings, particularly in high-stakes real-world applications like healthcare and education (Yauney & Shah, 2018; Liu et al., 2014) .

A Bayes-optimal policy, which does this trade-off optimally, conditions actions not only on the environment state but on the agent's own uncertainty about the current MDP.

In principle, a Bayes-optimal policy can be computed using the framework of Bayes-adaptive Markov decision processes (BAMDPs) (Martin, 1967; Duff & Barto, 2002) , in which the agent maintains a belief distribution over possible environments.

Augmenting the state space of the underlying MDP with this belief yields a BAMDP, a special case of a belief MDP (Kaelbling et al., 1998) .

A Bayes-optimal agent maximises expected return in the BAMDP by systematically seeking out the data needed to quickly reduce uncertainty, but only insofar as doing so helps maximise expected return.

Its performance is bounded from above by the optimal policy for the given MDP, which does not need to take exploratory actions but requires prior knowledge about the MDP to compute.

Unfortunately, planning in a BAMDP, i.e., computing a Bayes-optimal policy that conditions on the augmented state, is intractable for all but the smallest tasks.

A common shortcut is to rely instead on posterior sampling (Thompson, 1933; Strens, 2000; Osband et al., 2013) .

Here, the agent periodically samples a single hypothesis MDP (e.g., at the beginning of an episode) from its posterior, and the policy that is optimal for the sampled MDP is followed until the next sample is drawn.

Planning is far more tractable since it is done on a regular MDP, not a BAMDP.

However, posterior sampling's exploration can be highly inefficient and far from Bayes-optimal.

Consider the example of a gridworld in Figure 1 , where the agent must navigate to an unknown goal located in the grey area (1a).

To maintain a posterior, the agent can uniformly assign non-zero probability to cells where the goal could be, and zero to all other cells.

A Bayes-optimal strategy strategically searches the set of goal positions that the posterior considers possible, until the goal is found (1b).

Posterior sampling by contrast samples a possible goal position, takes the shortest route there, and then resamples a different goal position from the updated posterior (1c).

Doing so is much less efficient since the agent's uncertainty is not reduced optimally (e.g., states are revisited).

Average return over all possible environments, over six episodes with 15 steps each (after which the agent is reset to the starting position).

The performance of any exploration strategy is bounded above by the optimal behaviour (of a policy with access to the true goal position).

The Bayes-optimal agent matches this behaviour from the third episode, whereas posterior sampling needs six rollouts.

VariBAD closely approximates Bayes-optimal behaviour in this environment.

As this example illustrates, Bayes-optimal policies can explore much more efficiently than posterior sampling.

Hence, a key challenge is to find ways to learn approximately Bayes-optimal policies while retaining the tractability of posterior sampling.

In addition, many complex tasks pose another key challenge: the inference involved in maintaining a posterior belief, needed even for posterior sampling, may itself be intractable.

In this paper, we combine ideas from Bayesian reinforcement learning, approximate variational inference, and meta-learning to tackle these challenges, and equip an agent with the ability to strategically explore unseen (but related) environments for a given distribution, in order to maximise its expected return.

More specifically, we propose variational Bayes-Adaptive Deep RL (variBAD), a way to meta-learn to perform approximate inference on a new task 1 , and incorporate task uncertainty directly during action selection.

We represent a single MDP using a learned, low-dimensional stochastic latent variable m. Given a set of tasks sampled from a distribution, we jointly meta-train: (1) a variational auto-encoder that can infer the posterior distribution over m in a new task while interacting with the environment, and (2) a policy that conditions on this posterior distribution over the MDP embeddings, and thus learns how to trade off exploration and exploitation when selecting actions under task uncertainty.

Figure 1e shows the performance of our method versus the hard-coded optimal (i.e., given privileged goal information), Bayes-optimal, and posterior sampling exploration strategies.

VariBAD's performance closely matches that of Bayes-optimal action selection, matching optimal performance from the third rollout.

Previous approaches to BAMDPs are only tractable in environments with small action and state spaces.

VariBAD offers a tractable and dramatically more flexible approach for learning Bayesadaptive policies tailored to the task distribution seen during training, with the only assumption that such a task distribution is available for meta-training.

We evaluate our approach on the gridworld shown above and on MuJoCo domains that are widely used in meta-RL, and show that variBAD exhibits superior exploratory behaviour at test time compared to existing meta-learning methods, achieving higher returns during learning.

As such, variBAD opens a path to tractable approximate Bayes-optimal exploration for deep reinforcement learning.

We define a Markov decision process (MDP) as a tuple M = (S, A, R, T, T 0 , ??, H) with S a set of states, A a set of actions, R(r t+1 |s t , a t , s t+1 ) a reward function, T (s t+1 |s t , a t ) a transition function, T 0 (s 0 ) an initial state distribution, ?? a discount factor, and H the horizon.

In the standard RL setting, we want to learn a policy ?? that maximises J (??) = E T0,T,?? H???1 t=0 ?? t R(r t+1 |s t , a t , s t+1 ) , the expected return.

Here, we consider a multi-task meta-learning setting, which we introduce next.

We adopt the standard meta-learning setting where we have a distribution p(M ) over MDPs, with an

.

Across tasks, the reward and transition functions vary but typically share some structure.

The index i represents a task description (e.g., a goal position or natural language instruction) or a task ID.

Sampling an MDP from p(M ) is typically done by sampling a reward and transition function from a distribution p(R, T ).

The distribution over MDPs is unknown to the agent, but single MDPs can be sampled for metatraining.

At each iteration, a batch

we are given a limited number of environment interactions for learning (how to learn), i.e., to maximise performance within that initially unknown MDP.

How those environment interactions are used depends on the meta-learning method (see Sec 4).

At meta-test time, the agent is evaluated based on the average return it achieves during learning, for tasks drawn from p. Doing this well requires at least two things: (1) incorporating prior knowledge obtained in related tasks, and (2) reasoning about (task) uncertainty when selecting actions to trade off exploration and exploitation.

In the following, we combine ideas from meta-learning and Bayesian RL to tackle these challenges.

When the MDP is unknown, optimal decision making has to trade off exploration and exploitation when selecting actions.

In principle, this can be done by taking a Bayesian approach to reinforcement learning (Bellman, 1956; Duff & Barto, 2002; Ghavamzadeh et al., 2015) .

In the Bayesian formulation of RL, we assume that the transition and reward functions are distributed according to a prior b 0 = p(R, T ).

Since the agent does not have access to the true reward and transition function, it can maintain a belief b t (R, T ) at every timestep, which is the posterior over the MDP given the agent's experience: given a trajectory of states, actions, and rewards, ?? :t = {s 0 , a 0 , r 1 , s 1 , a 1 , . . .

, s t }, the prior p(R, T ) can be updated to form a posterior belief b t (R, T ) = p(R, T |?? :t ).

This is often done by maintaining a distribution over the model parameters.

To allow the agent to incorporate the task uncertainty into its decision-making, this belief can be augmented to the state space, S + = S ?? B where B is the belief space.

States in S + are often called hyper-states, and they transition according to

i.e., the new environment state s t is the expected new state w.r.t.

the current posterior distribution of the transition function, and the belief is updated deterministically according to Bayes rule.

The reward function on hyperstates is defined as the expected reward under the current posterior (after the state transition) over reward functions,

We can now formulate the Bayes-Adaptive Markov decision process (BAMDP, Duff & Barto (2002) ), which consists of the tuple M = (S

.

This is a special case of a belief MDP, i.e, the MDP formed by taking the posterior beliefs maintained by an agent in a partially observable MDP (POMDP) and reinterpreting them as Markov states (Cassandra et al., 1994) .

In an arbitrary belief MDP, the belief is over a hidden state that can change at each timestep.

In a BAMDP, the belief is over the transition and reward functions, which are constant for a given task.

The agent's objective is now to maximise the expected return in the BAMDP,

i.e., maximise the expected return in an initially unknown environment, while learning, within the horizon H + .

Note that we distinguish between the MDP horizon H and the BAMDP horizon H + .

Although often H + = H, we might instead want the agent to act Bayes-optimal within the first N MDP episodes, so H + = N ?? H. Trading off exploration and exploitation optimally depends heavily on how much time the agent has left (e.g., to decide whether purely information-seeking actions are worth it).

The objective in (3) is maximised by the Bayes-optimal policy, which automatically trades off exploration and exploitation: it takes exploratory actions to reduce its task uncertainty only insofar as it helps to maximise the expected return within the horizon.

The BAMDP framework is powerful because it provides a principled way of formulating Bayes-optimal behaviour.

However, solving the BAMDP is hopelessly intractable for most interesting problems.

The main challenges are as follows.

??? We typically do not have access to the prior distribution p(M ) but can only sample from it.

??? We typically do not know the parameterisation of the true reward and/or transition model.

Instead, we can choose to approximate them, for example using deep neural networks.

??? The belief update (computing the posterior p(R, T |?? :t )) is often intractable.

??? Even given the posterior over the reward and transition model, planning in belief space is typically intractable.

In the following, we propose a method that uses meta-learning to do Bayesian reinforcement learning, which amounts to meta-learning a prior over tasks and performing inference over reward and transition functions using deep learning.

Crucially, our Bayes-adaptive policy is learned end-to-end with the inference framework, which means that no planning is necessary at test time.

It can be applied to the typical meta-learning setting and makes minimal assumptions (no task ID or description is required), resulting in a highly flexible and scalable approach to Bayes-adaptive Deep RL.

In this section, we present variBAD, and describe how we tackle the challenges outlined above.

We start by describing how to represent reward and transition functions, and (posterior) distributions over these.

We then consider how to meta-learn to perform approximate variational inference in a given task, and finally put all the pieces together to form our training objective.

In the typical meta-learning setting, the reward and transition functions that are unique to each MDP are unknown, but also share some structure across the MDPs M i in p(M ).

We know that there exists a true i which represents either a task description or task ID, but we do not have access to this information.

We therefore represent this value using a learned stochastic latent variable m i .

For a given MDP M i we can then write

where R and T are shared across tasks.

Since we do not have access to the true task description or ID, we need to infer m i given the agent's experience up to time step t collected in M i ,

i.e., we want to infer the posterior distribution p(m i |??

:t (from now on, we drop the sub-and superscript i for ease of notation).

Recall that our goal is to learn a distribution over the MDPs, and given a posteriori knowledge of the environment compute the optimal action.

Given the above reformulation, it is now sufficient to reason about the embedding m, instead of the transition and reward dynamics.

This is particularly useful when deploying deep learning strategies, where the reward and transition function can consist of millions of parameters, but the embedding m can be a small vector.

A trajectory of states, actions and rewards is processed online using an RNN to produce the posterior over task embeddings, q ?? (m|?? :t ).

The posterior is trained using a decoder which attempts to predict future states and rewards from current states and actions.

The policy conditions on the posterior in order to act in the environment and is trained using RL.

Computing the exact posterior is typically not possible: we do not have access to the MDP (and hence the transition and reward function), and marginalising over tasks is computationally infeasible.

Consequently, we need to learn a model of the environment p ?? (?? :H + |a :H + ???1 ), parameterised by ??, together with an amortised inference network q ?? (m|?? :t ), parameterised by ??, which allows fast inference at runtime at each timestep t.

The action-selection policy is not part of the MDP, so an environmental model can only give rise to a distribution of trajectories when conditioned on actions, which we typically draw from our current policy, a ??? ??.

At any given time step t, our model learning objective is thus to maximise

where ??(M, ?? :H + ) is the trajectory distribution induced by our policy and we slightly abuse notation by denoting by ?? the state-reward trajectories, excluding the actions.

In the following, we drop the conditioning on a :H + ???1 to simplify notation.

Instead of optimising (7), which is intractable, we can optimise a tractable lower bound, defined with a learned approximate posterior q ?? (m|?? :t ) which can be estimated by Monte Carlo sampling (for the full derivation see AppendixA):

The term E q [log p(?? :H + |m)] is often referred to as the reconstruction loss, and p(?? :t |m) as the decoder.

The term KL(q(m|?? :t )||p ?? (m)) is the KL-divergence between our variational posterior q ?? and the prior over the embeddings p ?? (m).

For sufficiently expressive decoders, we are free to choose p ?? (m).

We set the prior to our previous posterior, q ?? (m|?? :t???1 ), with initial prior q ?? (m) = N (0, I).

As can be seen in Equation 8 and Figure 2 , when the agent is at timestep t, we encode the past trajectory ?? :t to get the current posterior q(m|?? :t ) since this is all the information available to perform inference about the current task.

We then decode the entire trajectory ?? :T including the future, i.e., model E q [p(?? :T |m)].

This is different than the conventional VAE setup (and possible since we have access to this information during training).

Decoding not only the past but also the future is important because this way, variBAD learns to perform inference about unseen states given the past.

The reconstruction term log p(?? :H |m) factorises as log p(?? :H + |m, a :H + ???1 ) = log p((s 0 , r 0 , . . .

, s t???1 , r t???1 , s t )|m, a :H + ???1 )

= log p(s 0 |m) +

Here, p(s 0 |m) is the initial state distribution T 0 , p(s i+1 |s i , a i ; m) the transition function T , and p(r i+1 |s t , a t , s i+1 ; m) the reward function R .

From now, we include T 0 in T for ease of notation.

We can now formulate a training objective for learning the approximate posterior distribution over task embeddings, the policy, and the generalised reward and transition functions R and T .

We use deep neural networks to represent the individual components.

These are:

1.

The encoder q ?? (m|?? :t ), parameterised by ??;

2.

An approximate transition function T = p ?? (s i+1 |s i , a i ; m) and an approximate reward function R = p ?? (r i+1 |s t , a t , s i+1 ; m) which are jointly parameterised by ??; and 3.

A policy ?? ?? (a t |s t , q ?? (m|?? :t )) parameterised by ?? and dependent on ??.

The policy is conditioned on both the environment state and the posterior over m, ??(a t |s t , q(m|?? :t )).

This is similar to the formulation of BAMDPs introduced in 2.2, with the difference that we learn a unifying distribution over MDP embeddings, instead of the transition/reward function directly.

This makes learning easier since there are fewer parameters to perform inference over, and we can use data from all tasks to learn the shared reward and transition function.

The posterior can be represented by the distribution's parameters (e.g., mean and standard deviation if q is Gaussian).

Our overall objective is to maximise

Expectations are approximated by Monte Carlo samples, and the ELBO can be optimised using the reparameterisation trick (Kingma & Welling, 2014) .

For t = 0, we use the prior q ?? (m) = N (0, I).

Past trajectories can be encoded using, e.g., a recurrent network as in Duan et al. (2016) ; Wang et al. (2016) (which we did in our experiments) or using an encoder that computes an encoding per (s, a, s , r)-tuple and aggregates them in some way (Zaheer et al., 2017; Garnelo et al., 2018; Rakelly et al., 2019) .

The network architecture is shown in Figure 2 .

In Equation (10), we see that the ELBO appears for all possible context lengths t. This way, variBAD can learn how to perform inference online (while the agent is interacting with an environment), and decrease its uncertainty over time given more data.

In practice, we may subsample a fixed number of ELBO terms (for random time steps t) for computational efficiency if H + is large.

Equation (10) is trained end-to-end, and ?? weights the supervised model learning objective against the RL loss.

This is necessary since parameters ?? are shared between the model and the policy.

However, we found that not backpropagating the RL loss through the encoder is typically sufficient in practice, which speeds up training considerably.

In our experiments, we therefore optimise the policy and the VAE using different optimisers and learning rates.

We typically train the RL agent and the VAE using different data buffers: the policy is only trained with the most recent data since we use on-policy algorithms in our experiments; and for the VAE we maintain a separate, larger buffer of trajectories to compute the ELBO.

At meta-test time, we roll out the policy in randomly sampled test tasks (via forward passes through the encoder and policy) to evaluate performance.

The decoder is not used at test time, and no gradient adaptation is done: the policy has learned to act approximately Bayes-optimal during meta-training.

Meta Reinforcement Learning.

A prominent model-free meta-RL approach is to utilise the dynamics of recurrent networks for fast adaptation (RL 2 , Wang et al. (2016) ; Duan et al. (2016) ).

At every time step, the network gets an auxiliary comprised of the preceding action and reward.

This allows learning within a task to happen online, entirely in the dynamics of the recurrent network.

If we remove the decoder (Fig 2) and the VAE objective (Eq (7)), variBAD reduces to this setting, i.e., the main differences are that we use a stochastic latent variable (an inductive bias for representing uncertainty) together with a decoder to reconstruct previous and future transitions / rewards (which acts as an auxiliary loss (Jaderberg et al., 2017) to encode the task in latent space and deduce information about unseen states).

provide an in-depth discussion of meta-learning sequential strategies and how to recast memory-based meta-learning within a Bayesian framework.

Another popular approach to meta RL is to learn an initialisation of the model, such that at test time, only a few gradient steps are necessary to achieve good performance (Finn et al., 2017; Nichol & Schulman, 2018) .

These methods do not directly account for the fact that the initial policy needs to explore, a problem addressed, a.o., by (E-MAML) and Rothfuss et al. (2019) (ProMP) .

In terms of model complexity, MAML and ProMP are relatively lightweight, since they typically consist of a feedforward policy.

RL 2 and variBAD use recurrent modules, which increases model complexity but allows online adaptation.

Other methods that perform gradient adaptation at test time are, e.g., who meta-learn a loss function conditioned on the agent's experience that is used at test time so learn a policy (from scratch); and Sung et al. (2017) who learn a meta-critic that can criticise any actor for any task, and is used at test time to train a policy.

Compared to variBAD, these methods usually separate exploration (before gradient adaptation) and exploitation (after gradient adaptation) at test time by design, making them less sample efficient.

Skill / Task Embeddings.

Learning (variational) task or skill embeddings for meta / transfer reinforcement learning is used in a variety of approaches.

Hausman et al. (2018) use approximate variational inference learn an embedding space of skills (with a different lower bound than variBAD).

At test time the policy is fixed, and a new embedder is learned that interpolates between already learned skills.

Arnekvist et al. (2019) learn a stochastic embedding of optimal Q-functions for different skills, and condition the policy on (samples of) this embedding.

Adaptation at test time is done in latent space.

Co-Reyes et al. (2018) learn a latent space of low-level skills that can be controlled by a higher-level policy, framed within the setting of hierarchical RL.

This embedding is learned using a VAE to encode state trajectories and decode states and actions.

Zintgraf et al. (2019) learn a deterministic task embedding trained similarly to MAML (Finn et al., 2017) .

Similar to variBAD, Zhang et al. (2018) use learned dynamics and reward modules to learn a latent representation which the policy conditions on and show that transferring the (fixed) encoder to new environments helps learning.

Perez et al. (2018) learn dynamic models with auxiliary latent variables, and use them for model-predictive control.

Lan et al. (2019) learn a task embedding with an optimisation procedure similar to MAML, where the encoder is updated at test time, and the policy is fixed.

Saemundsson et al. (2018) explicitly learn an embedding of the environment model, which is subsequently used for model predictive control (and not, like in variBAD, for exploration).

In the field of imitation learning, some approaches embed expert demonstrations to represent the task; e.g., Wang et al. (2017) use variational methods and Duan et al. (2017) learn deterministic embeddings.

VariBAD differs from the above methods mainly in what the embedding represents (i.e., task uncertainty) and how it is used: the policy conditions on the posterior distribution over MDPs, allowing it to reason about task uncertainty and trade off exploration and exploitation online.

Our objective (8) explicitly optimises for Bayes-optimal behaviour.

Unlike some of the above methods, we do not use the model at test time, but model-based planning is a natural extension for future work.

Recent work by Humplik et al. (2019) also exploits the idea of conditioning the policy on a latent distribution over task embeddings, which is trained using privileged information such as a task ID, or an expert policy.

VariBAD on the other hand can be applied even when such information is not available.

Bayesian Reinforcement Learning.

Bayesian methods for RL can be used to quantify uncertainty to support action-selection, and provide a way to incorporate prior knowledge into the algorithms (see Ghavamzadeh et al. (2015) for a review).

A Bayes-optimal policy is one that optimally trades off exploration and exploitation, and thus maximises expected return during learning.

While such a policy can in principle be computed using the BAMDP framework, it is hopelessly intractable for all but the smallest tasks.

Existing methods are therefore restricted to small and discrete state / action spaces (Asmuth & Littman, 2011; Guez et al., 2012; , or a discrete set of tasks (Brunskill, 2012; Poupart et al., 2006) .

VariBAD opens a path to tractable approximate Bayes-optimal exploration for deep RL by leveraging ideas from meta-learning and approximate variational inference, with the only assumption that we can meta-train on a set of related tasks.

Existing approximate Bayesian RL methods often require us to define a prior / belief update on the reward / transition function, and rely on (possibly expensive) sample-based planning procedures.

Due to the use of deep neural networks however, variBAD lacks the formal guarantees enjoyed by some of the methods mentioned above.

Posterior sampling (Strens, 2000; Osband et al., 2013) , which extends Thompson sampling (Thompson, 1933 ) from bandits to MDPs, estimates a posterior distribution over MDPs (i.e., model and reward functions), in the same spirit as variBAD.

This posterior is used to periodically sample a single hypothesis MDP (e.g., at the beginning of an episode), and the policy that is optimal for the sampled MDP is followed subsequently.

This approach is less efficient than Bayes-optimal behaviour and therefore typically has lower expected return during learning.

A related approach for inter-task transfer of abstract knowledge is to pose policy search with priors as Markov Chain Monte Carlo inference (Wingate et al., 2011) .

Similarly Guez et al. (2013) propose a Monte Carlo Tree Search based method for Bayesian planning to get a tractable, samplebased method for obtaining approximate Bayes-optimal behaviour.

Osband et al. (2018) note that non-Bayesian treatment for decision making can be arbitrarily suboptimal and propose a simple randomised prior based approach for structured exploration.

Some recent deep RL methods use stochastic latent variables for structured exploration Rakelly et al., 2019) , which gives rise to behaviour similar to posterior sampling.

Other ways to use the posterior for exploration are, e.g., certain reward bonuses Kolter & Ng (2009) ; Sorg et al. (2012) and methods based on optimism in the face of uncertainty (Kearns & Singh, 2002; Brafman & Tennenholtz, 2002) .

NonBayesian methods for exploration are often used in practice, such as other exploration bonuses (e.g., via state-visitation counts) or using uninformed sampling of actions (e.g., -greedy action selection).

Such methods are prone to wasteful exploration that does not help maximise expected reward.

Variational Inference and Meta-Learning.

A main difference of variBAD to many existing Bayesian RL methods is that we meta-learn the inference procedure, i.e., both the (meaning of the) prior and how to update the posterior, instead of assuming that we have access to the prior or wellbehaved distributions for which we can update the posterior analytically.

Apart from (RL) methods mentioned above, related work in this direction can be found, a.o., in Garnelo et al. (2018) ; Gordon et al. (2019); Choi et al. (2019) .

By comparison, variBAD has a different inference procedure tailored to the setting of learning Bayes-optimal policies for a given distribution over MDPs.

POMDPs.

Several deep learning approaches to model-free reinforcement learning (Igl et al., 2019) and model learning for planning (Tschiatschek et al., 2018) in partially observable Markov decision processes have recently been proposed and utilise approximate variational inference methods.

VariBAD by contrast focuses on BAMDPs (Martin, 1967; Duff & Barto, 2002; Ghavamzadeh et al., 2015) , a special case of POMDPs where the transition and reward functions constitute the hidden state and the agent must maintain a belief over them.

While in general the hidden state in a POMDP can change at each time-step, in a BAMDP the underlying task, and therefore the hidden state, is fixed per task.

We exploit this property by learning an embedding that is fixed over time, unlike approaches like Igl et al. (2019) which use filtering to track the changing hidden state.

While we utilise the power of deep approximate variational inference, other approaches for BAMDPs often use more accurate but less scalable methods, e.g., discretise the latent distribution and use Bayesian filtering for the posterior update.

In this section we first investigate the properties of variBAD on a didactic gridworld domain.

We show that variBAD performs structured and online exploration as it attempts to identify the task at hand.

Then we consider more complex meta-learning settings by employing on two MuJoCo continuous control tasks commonly used in the meta-RL literature.

We show that variBAD can learn to adapt to the task during the first rollout, unlike most existing meta-learning techniques.

Details and hyperparameters can be found in the appendix.

We will provide an open-source reference implementation together with the paper.

To gain insight into variBAD's properties, we start with a didactic gridworld environment.

The task is to reach a goal (selected uniformly at random) in a 5 ?? 5 gridworld.

The goal is unobserved by the agent, inducing task uncertainty and necessitating exploration.

The goal can be anywhere except around the starting cell, which is at the bottom left.

Actions are: up, right, down, left, stay (executed deterministically), and after 15 steps the agent is reset.

The horizon within the MDP is H = 15, but we choose a horizon of H + = 3 ?? H = 45 for the BAMDP.

I.e., we train our agent to maximise performance for 3 MDP episodes.

The agent gets a sparse reward signal: ???0.1 on non-goal cells, and +1 on the goal cell.

The best strategy is to explore until the goal is found, and stay at the goal or return to it when reset to the initial position.

We use a latent dimensionality of 5.

Figure 3 illustrates how variBAD behaves at test time with deterministic actions (i.e., all exploration is done by the policy).

In 3a we see how the agent interacts with the environment, with the blue background visualising the posterior belief by using the learned reward function.

VariBAD learns the correct prior and adjusts its belief correctly over time.

It predicts no reward for cells it has visited, and explores the remaining cells until it finds the goal.

A nice property of variBAD is that we can gain insight into the agent's belief about the environment by analysing what the decoder predicts, and how the latent space changes while the agent interacts with the environment.

Figure 3b show the reward predictions: each line represents a grid cell and its value the probability of receiving a reward at that cell.

As the agent gathers more data, more and more cells are excluded (p(rew = 1) = 0), until eventually the agent finds the goal.

In Figure 3c we visualise the 5-dimensional latent space.

We see that once the agent finds the goal, the posterior concentrates: the variance drops close to zero, and the mean settles on a value.

As we showed in Figure 1e , the behaviour of variBAD closely matches that of the Bayes-optimal policy.

Recall that the Bayes-optimal policy is the one which optimally trades off exploration and exploitation in an unknown environment, and outperforms posterior sampling.

Our results on this gridworld indicate that variBAD is an effective way to approximate Bayes-optimal control.

We also tried a similar approach to Duan et al. (2016); Wang et al. (2016) , by using a similar architecture as variBAD but a deterministic embedding (of size 64) and no decoder.

We observed that this performs worse overall compared to variBAD (see Appendix B for a qualitative and quantitative comparison).

We show that variBAD is capable of scaling to more complex meta learning settings by employing it on MuJoCo (Todorov et al., 2012) locomotion tasks commonly used in the meta-RL literature.

2 We consider the HalfCheetahDir environment where the agent has to run either forwards or backwards (i.e., there are only two tasks), and the HalfCheetahVel environment where the agent has to run at different velocities (i.e., there are infinitely many tasks).

Both environments have a horizon H = H + = 200, i.e., we aim to maximise performance within a single rollout.

Figure 4a shows the performance at test time compared to existing methods.

While we show performance for multiple rollouts for the sake of completeness, anything beyond the first rollout is not directly relevant to our goal, which is to maximise performance on a new task, while learning, within a single episode.

Only variBAD and RL 2 are able to adapt to the task at hand within a single episode.

RL 2 underperforms variBAD on the HalfCheetahDir environment, and learning is slower and less stable (see learning curves and runtime comparisons in Appendix C).

Even though the first rollout includes exploratory steps, this matches the optimal oracle policy (which is conditioned on the true (2019)) are not designed to maximise reward during a single rollout, and perform poorly in this case.

They all require substantially more environment interactions in each new task to achieve good performance.

PEARL, which is akin to posterior sampling, only starts performing well starting from the third episode (Note: PEARL outperforms our oracle slightly, likely since our oracle is based on PPO, and PEARL is based on SAC).

E-MAML and ProMP use 20-40 rollouts for the gradient update, which is far less sample efficient than variBAD.

In Figure 4a , we show the performance with smaller batch sizes (0 ??? 4 rollouts), collected with the initial policy, and by performing a gradient update also on the learned initialisation.

To get a sense for where these differences might stem from, consider Figure 4b which shows example behaviour of the policies during the first three rollouts in the HalfCheetahDir environment, when the task is "go left".

Both variBAD and RL 2 adapt to the task online, whereas PEARL acts according to the current sample, which in the first two rollouts can mean walking in the wrong direction.

For a visualisation of the variBAD latent space at test time for this environment see Appendix C.3.

While we outperform at meta-test time, PEARL is more sample efficient during meta-training (see Appendix C.1), since it is an off-policy method.

Extending variBAD to off-policy methods is an interesting but orthogonal direction for future work.

Overall, our empirical results confirm that variBAD can scale up to current benchmarks and maximise expected reward within a single episode.

We presented variBAD, a novel deep RL method to approximate Bayes-optimal behaviour, which uses meta-learning to utilise knowledge obtained in related tasks and perform approximate inference in unknown environments.

In a didactic gridworld environment, our agent closely matches Bayesoptimal behaviour, and in more challenging MuJoCo tasks, variBAD outperforms existing methods in terms of achieved reward during a single episode.

In summary, we believe variBAD opens a path to tractable approximate Bayes-optimal exploration for deep reinforcement learning.

There are several interesting directions of future work based on variBAD.

For example, we currently do not use the decoder at test time.

One could instead use the decoder for model-predictive planning, or to get a sense for how wrong the predictions are (which might indicate we are out of distribution, and further training is necessary).

Another exciting direction for future research is considering settings where the training and test distribution of environments are not the same.

Generalising to out-of-distribution tasks poses additional challenges and in particular for variBAD two problems are likely to arise: the inference procedure will be wrong (the prior and/or posterior update) and the policy will not be able to interpret a changed posterior.

In this case, further training of both the encoder/decoder might be necessary, together with updates to the policy and/or explicit planning.

A FULL ELBO DERIVATION Equation (8) can be derived as follows.

B EXPERIMENTS: GRIDWORLD B.1 ADDITIONAL REMARKS Figure 3c visualises how the latent space changes as the agent interacts with the environment.

As we can see, the value of the latent dimensions starts around mean 1 and variance 0, which is the prior we chose for the beginning of an episode.

Given that the variance increases for a little bit before the agent finds the goal, this prior might not be optimal.

A natural extension of variBAD is therefore to also learn the prior to match the task at hand.

We used the PyTorch framework for our experiments.

Hyperparameters can be found below.

Figure 5 shows the behaviour of variBAD in comparison to a recurrent policy, an architecture which resembles the approach presented by Duan et al. (2016) and Wang et al. (2016) .

As we can see, the recurrent policy re-visits states it has already seen before, indicating that its does task inference less efficiently.

We believe that the stochastic latent embedding helps the policy express task uncertainty better, and that the auxiliary loss of decoding the embedding help learning.

Figure 6 shows the learning curves for this environment (for the full horizon 45) for variBAD and RL2, in comparison to the hard-coded optimal policy (which has access to the goal position), Bayesoptimal policy, and posterior sampling policy (evaluated on a horizon of H = 45, i.e., the first three "episodes" shown in Figure 1 ).

As we can see, variBAD closely approximates Bayes-optimal behaviour, and outperforms RL2.

C.1 LEARNING CURVES Figure 7 shows the learning curves for the MuJoCo environments for all approaches.

The oracle policy was trained using PPO.

Our approach for hyperparameter-tuning was to tune the hyperparameters for PPO using the oracle policy, and then using these for both variBAD and RL2, only further tuning those aspects particular to those methods (mostly VAE parameters and latent dimension).

PEARL (Rakelly et al., 2019) was trained using the reference implementation provided by the authors.

The environments we used are also taken from this implementation.

E-MAML and ProMP (Rothfuss et al., 2019) were trained using the reference implementation provided by Rothfuss et al. (2019) .

As we can see, PEARL is much more sample efficient in terms of number of frames than the other methods (Fig 7) , which is because it is an off-policy method.

On-policy vs off-policy training is an orthogonal issue to our contribution, but an extension of variBAD to off-policy methods is an interesting direction for future work.

Doing posterior sampling using off-policy methods also requires PEARL to use a different encoder (to maintain order invariance of the sampled trajectories) which is non-recurrent (and hence faster to train, see next section) but restrictive since it assumes independence between individual transitions.

The following are rough estimates of average run-times for the HalfCheetah-Dir environment (from what we have experienced; we often ran multiple experiments per machine, so some of these might be overestimated and should be mostly understood as giving a relative sense of ordering).

??? ProMP, E-MAML: 5-8 hours ??? variBAD: 48 hours ??? RL 2 : 60 hours

??? PEARL: 24 hours E-MAML and ProMP have the advantage that they do not have a recurrent part such as variBAD or RL 2 .

Forward and backward passes through recurrent networks can be slow, especially with large horizons.

2 use recurrent modules, we observed that variBAD is faster when training the policy with PPO.

This is because we do not backpropagate the RL-loss through the recurrent part, which allows us to make the PPO mini-batch updates without having to re-compute the embeddings (so it saves us a lot of forward/backward passes through the recurrent model).

Additionally, we This difference should be less pronounced with other RL methods that do not rely on this many forward/backward passes per policy update.

Figure 8 shows the latent space for the HalfCheetahDir tasks "go right" (top row) and "go left" (bottom row).

We observe that the latent mean and log-variance adapt rapidly, within just a few environment steps (left and middle figures).

This is also how fast the agent adapts to the current task (right figures).

It is interesting to note that the values of the latent dimensions swap signs between the two tasks.

Even though the latent mean/logvariance change rapidly within just a few steps, we can see that the variance increases slightly as time progresses.

We believe that this is on the one hand because this does not get penalised by the ELBO objective (as long as the means values are far enough apart, a little bit of variance on the sampled values does not hurt prediction) and because the VAE might overfit to the agent: once the agent is sufficiently far away from the origin, due to it being trained, the reward can be predicted just by considering the agent's position.

Visualising the belief in the reward/state space directly, as we have done in the gridworld example, is more difficult for MuJoCo tasks, since we now have continuous states and actions.

What we could do however, is additionally train a model that predicts a ground-truth task description or ID (separate from the main objective and just for further analysis, since we do not want to use this privileged information for meta-training).

This would give us a sense directly of how certain the agent is about the task (without artefacts such as increasing latent variance in the logspace as observed above).

Note that even though we maximise performance for a horizon of H = 200 for both RL2 and variBAD, we sometimes keep the posterior distribution (or the hidden state of the RNN) which was obtained during one rollout when resetting the environment.

This is to make sure that the agent can learn how to act for more than one episode.

We observe that RL2 is unstable when it comes to maintaining its performance over multiple rollouts.

We hypothesize this is due to the specific property of the HalfCheetah environments that the state can give information about the task if the agent has already adapted.

E.g., in the HalfCheetahDir environment, the x-position is part of the state observed by the agent.

At the beginning of the episode, when starting close to x = 0, the agent has to infer in which direction to run.

However, once it moves further and further away from the origin, it is sufficient to rely on the actual environment state to infer which task the agent is currently in.

This could lead to the hidden state of the recurrent part of the network being ignored (and taking values that the agent cannot interpret), such that when reset to the origin, the inference procedure has to be re-done.

VariBADdoes not have this problem, since we train the latent embedding to represent the task, and only the task.

Therefore, the agent does not have to do the inference procedure again when reset to the starting position, but can rely on the latent task description that is given by the approximate posterior.

We used the PyTorch framework for our experiments.

Hyperparameters can be found below.

@highlight

VariBAD opens a path to tractable approximate Bayes-optimal exploration for deep RL using ideas from meta-learning, Bayesian RL, and approximate variational inference.

@highlight

This paper presents a new deep reinforcement learning method that can efficiently trade-off exploration and exploitation that combines meta-learning, variational inference, and bayesian RL.