In reinforcement learning, we can learn a model of future observations and rewards, and use it to plan the agent's next actions.

However, jointly modeling future observations can be computationally expensive or even intractable if the observations are high-dimensional (e.g. images).

For this reason, previous works have considered partial models, which model only part of the observation.

In this paper, we show that partial models can be causally incorrect: they are confounded by the observations they don't model, and can therefore lead to incorrect planning.

To address this, we introduce a general family of partial models that are provably causally correct, but avoid the need to fully model future observations.

The ability to predict future outcomes of hypothetical decisions is a key aspect of intelligence.

One approach to capture this ability is via model-based reinforcement learning (MBRL) (Munro, 1987; Werbos, 1987; Nguyen & Widrow, 1990; Schmidhuber, 1991) .

In this framework, an agent builds an internal representation s t by sensing an environment through observational data y t (such as rewards, visual inputs, proprioceptive information) and interacts with the environment by taking actions a t according to a policy π(a t |s t ).

The sensory data collected is used to build a model that typically predicts future observations y >t from past actions a ≤t and past observations y ≤t .

The resulting model may be used in various ways, e.g. for planning (Oh et al., 2015; Silver et al., 2017a) , generation of synthetic training data (Weber et al., 2017) , better credit assignment (Heess et al., 2015) , learning useful internal representations and belief states (Gregor et al., 2019; Guo et al., 2018) , or exploration via quantification of uncertainty or information gain (Pathak et al., 2017) .

Within MBRL, commonly explored methods include action-conditional, next-step models (Oh et al., 2015; Ha & Schmidhuber, 2018; Chiappa et al., 2017; Schmidhuber, 2010; Xie et al., 2016; Deisenroth & Rasmussen, 2011; Lin & Mitchell, 1992; Li et al., 2015; Diuk et al., 2008; Igl et al., 2018; Ebert et al., 2018; Kaiser et al., 2019; Janner et al., 2019) .

However, it is often not tractable to accurately model all the available information.

This is both due to the fact that conditioning on high-dimensional data such as images would require modeling and generating images in order to plan over several timesteps (Finn & Levine, 2017) , and to the fact that modeling images is challenging and may unnecessarily focus on visual details which are not relevant for acting.

These challenges have motivated researchers to consider simpler models, henceforth referred to as partial models, i.e. models which are neither conditioned on, nor generate the full set of observed data (Guo et al., 2018; Gregor et al., 2019; Amos et al., 2018) .

In this paper, we demonstrate that partial models will often fail to make correct predictions under a new policy, and link this failure to a problem in causal reasoning.

Prior to this work, there has been a growing interest in combining causal inference with RL research in the directions of non-model based bandit algorithms (Bareinboim et al., 2015; Forney et al., 2017; Zhang & Bareinboim, 2017; Lee & Bareinboim, 2018; Bradtke & Barto, 1996) and causal discovery with RL (Zhu & Chen, 2019) .

Contrary to previous works, in this paper we focus on model-based approaches and propose a novel framework for learning better partial models.

A key insight of our methodology is the fact that any piece of information about the state of the environment that is used by the policy to make a decision, but is not available to the model, acts as a confounding variable for that model.

As a result, the learned model is causally incorrect.

Using such a model to reason may lead to the wrong conclusions about the optimal course of action as we demonstrate in this paper.

We address these issues of partial models by combining general principles of causal reasoning, probabilistic modeling and deep learning.

Our contributions are as follows.

• We identify and clarify a fundamental problem of partial models from a causal-reasoning perspective and illustrate it using simple, intuitive Markov Decision Processes (MDPs) (Section 2).

• In order to tackle these shortcomings we examine the following question: What is the minimal information that we have to condition a partial model on such that it will be causally correct with respect to changes in the policy? (Section 4)

• We answer this question by proposing a family of viable solutions and empirically investigate their effects on models learned in illustrative environments (simple MDPs and 3D environments).

Our method is described in Section 4 and the experiments are in Section 5.

We illustrate the issues with partial models using a simple example.

Consider the FuzzyBear MDP shown in Figure 1 (a): an agent at initial state s 0 transitions into an encounter with either a teddy bear or a grizzly bear with 50% random chance, and can then take an action to either hug the bear or run away.

In order to plan, the agent may learn a partial model q θ (r 2 |s 0 , a 0 , a 1 ) that predicts the reward r 2 after performing actions {a 0 , a 1 } starting from state s 0 .

This model is partial because it conditions on a sequence of actions without conditioning on the intermediate state s 1 .

The model is suitable for deterministic environments, but it will have problems on stochastic environments, as we shall see.

Such a reward model is usually trained on the agent's experience which consists of sequences of past actions and associated rewards.

Now, suppose the agent wishes to evaluate the sequence of actions {a 0 = visit forest, a 1 = hug} using the average reward under the model q θ (r 2 |s 0 , a 0 , a 1 ).

From Figure 1 (a), we see that the correct average reward is 0.5 × 1 + 0.5 × (−0.5) = 0.25.

However, if the model has been trained on past experience in which the agent has mostly hugged the teddy bear and ran away from the grizzly bear, it will learn that the sequence {visit forest, hug} is associated with a reward close to 1, and that the sequence {visit forest, run} is associated with a reward close to 0.

Mathematically, the model will learn the following conditional probability:

where s 1 is the state corresponding to either teddy bear or grizzly bear.

In the above expression, p(s 1 |s 0 , a 0 ) and p(r 2 |s 1 , a 1 ) are the transition and reward dynamics of the MDP, and π(a 1 |s 1 ) is the agent's behavior policy that generated its past experience.

As we can see, the behavior policy affects what the model learns.

The fact that the reward model q θ (r 2 |s 0 , a 0 , a 1 ) is not robust to changes in the behavior policy has serious implications for planning.

For example, suppose that instead of visiting the forest, the agent could have chosen to stay at home as shown in Figure 1 (b) .

In this situation, the optimal action is to stay home as it gives a reward of 0.6, whereas visiting the forest gives at most a reward of 0.5×1+0.5×0 = 0.5.

However, an agent that uses the above reward model to plan will overestimate the reward of going into the forest as being close to 1 and choose the suboptimal action.

One way to avoid this bias is to use a behavior policy that doesn't depend on the state s 1 , i.e. π(a 1 |s 1 ) = π(a 1 ).

Unfortunately, this approach does not scale well to complex environments as it requires an enormous amount of training data for the behavior policy to explore interesting states.

A better approach is to make the model robust to changes in the behavior policy.

Fundamentally, the problem is due to causally incorrect reasoning: the model learns the observational conditional p(r 2 |s 0 , a 0 , a 1 ) instead of the interventional conditional given by:

where the do-operator do(·) means that the actions are performed independently of the unspecified context (i.e. independently of s 1 ).

The interventional conditional is robust to changes in the policy and is a more appropriate quantity for planning.

In contrast, the observational conditional quantifies the statistical association between the actions a 0 , a 1 and the reward r 2 regardless of whether the actions caused the reward or the reward caused the actions.

In Section 3, we review relevant concepts from causal reasoning, and based on them we propose solutions that address the problem.

Finally, although using p(r 2 |s 0 , do(a 0 ), do(a 1 )) leads to causally correct planning, it is not optimal either: it predicts a reward of 0.25 for the sequence {visit forest, hug} and 0 for the sequence {visit forest, run}, whereas the optimal policy obtains a reward of 0.5.

The optimal policy makes the decision after observing s 1 (teddy bear vs grizzly bear); it is closed-loop as opposed to open-loop.

The solution is to make the intervention at the policy level instead of the action level, as we discuss in the following sections.

Many applications of machine learning involve predicting a variable y (target) from a variable x (covariate).

A standard way to make such a prediction is by fitting a model q θ (y|x) to a dataset of (x, y)-pairs.

Then, if we are given a new x and the data-generation process hasn't changed, we can expect that a well trained q θ (y|x) will make an accurate prediction of y.

(a) (b) (c) (d) (e) (f)

In many situations however, we would like to use the data to make different kinds of predictions.

For example, what prediction of y should we make, if something in the environment has changed, or if we set x ourselves?

In the latter case x didn't come from the original data-generation process.

This may cause problems in our prediction, because there may be unobserved variables u, known as confounders, that affected both x and y during the data-generation process.

That is, the actual process was of the form p(u)p(x|u)p(y|x, u) where we only observed x and y as shown in Figure 2 (b) .

Under this assumption, a model q θ (y|x) fitted on (x, y)-pairs will converge to the target p(y|x) ∝ p(u)p(x|u)p(y|x, u)du.

However, if at prediction time we set x ourselves, the actual distribution of y will be p(y|do(x)) = p(u)p(y|x, u)du.

This is because setting x ourselves changes the original graph from Figure 2 (b) to the one in Figure 2 (c).

Interventions: The operation of setting x to a fixed value x 0 independently of its parents, known as the do-operator (Pearl et al., 2016) , changes the data-generation process to p(u)δ(x − x 0 )p(y|x, u), where δ(x − x 0 ) is the delta-function.

As explained above, this results in a different target distribution p(u)p(y|x 0 , u)du, which we refer to as p(y|do(x = x 0 )), or simply p(y|do(x)) when x 0 is implied.

Let par j be the parents of x j .

The do-operator is a particular case of the more general concept of an intervention: given a generative process p(x) = j p j (x j |par j ), an intervention is defined as a change that replaces one or more factors by new factors.

For example, the intervention

The do-operator is a "hard" intervention whereby we replace a node by a delta function; that is, p(

, where x /k denotes the collection of all variables except x k .

In general, for graphs of the form of Figure 2 (b), p(y|x) does not equal p(y|do(x)).

As a consequence, it is not generally possible to recover p(y|do(x)) using observational data, i.e. (x, y)-pairs sampled from p(x, y), regardless of the amount of data available or the expressivity of the model.

However, recovering p(y|do(x)) from observational data alone becomes possible if we assume additional structure in the data-generation process.

Suppose there exists another observed variable z that blocks all paths from the confounder u to the covariate x as shown in Figure 2 (d) .

This variable is a particular case of the concept of a backdoor (Pearl et al., 2016, Chapter 3.3) and is said to be a backdoor for the pair x − y. In this case, we can express p(y|do(x)) entirely in terms of distributions that can be obtained from the observational data as:

This formula holds as long as p(x|z) > 0 and is referred to as backdoor adjustment.

The same formula applies when z blocks the effect of the confounder u on y as in Figure 2 (f).

More generally, we can use p(z) and p(y|z, x) from Equation (1) to compute the marginal distribution p(y) under an arbitrary intervention of the form p(x|z) → ψ(x|z) on the graph in Figure 2 (b).

We refer to the new marginal as p do(ψ) (y) and obtain it by:

A similar formula can be derived when there is a variable z blocking the effect of x on y, which is known as a frontdoor, shown in Figure 2 (e).

Derivations for the backdoor and frontdoor adjustment formulas are provided in Appendix A.

Causally correct models: Given data generated by an underlying generative process p(x), we say that a learned model q θ (x) is causally correct with respect to a set of interventions I if the model remains accurate after any intervention in I. That is, if q θ (x) ≈ p(x) and q θ (x) is causally correct with respect to I, then

Backdoor-adjustment and importance sampling: Given a dataset of N tuples (z n , x n , y n ) generated from the joint distribution p(u)p(z|u)p(x|z)p(y|x, u), we could alternatively approximate the marginal distribution p do(ψ) (y) after an intervention p(x|z) → ψ(x|z) by fitting a distribution q θ (y) to maximize the re-weighted likelihood:

where w(x, z) = ψ(x|z)/p(x|z) are the importance weights.

While this solution is a mathematically sound way of obtaining p do(ψ) (y), it requires re-fitting of the model for any new ψ(x|z).

Moreover, if ψ(x|z) is very different from p(x|z) the importance weights w(x, z) will have high variance.

By fitting the conditional distribution p(y|z, x) and using Equation (2) we can avoid these limitations.

Connection to MBRL: As we will see in much greater detail in the next section, there is a direct connection between partial models in MBRL and the causal concepts discussed above.

In MBRL we are interested in making predictions about some aspect of the future (observed frames, rewards, etc.); these would be the dependent variables y. Such predictions are conditioned on actions which play the role of the covariates x. When using partial models, the models will not have access to the full state of the policy and so the policy's state will be a confounding variable u. Any variable in the computational graph of the policy that mediates the effect of the state in the actions will be a backdoor with respect to the action-prediction pair.

We consider environments with a hidden state e t and dynamics specified by an unknown transition probability of the form p(e t |e t−1 , a t−1 ).

At each step t, the environment receives an action a t−1 , updates its state to e t and produces observable data y t ∼ p(y t |e t ) which includes a reward r t and potentially other forms of data such as images.

An agent with internal state s t interacts with the environment via actions a t produced by a policy π(a t |s t ) and updates its state using the observations y t+1 by s t+1 = f s (s t , a t , y t+1 ), where f s can for instance be implemented with an RNN.

The agent will neither observe nor model the environment state e t ; it is a confounder on the data generation process.

Figure 3 (a) illustrates the interaction between the agent and the environment.

Consider an agent at an arbitrary point in time and whose current state 2 is s 0 , and assume we are interested in generative models that can predict the outcome 3 y T of a sequence of actions {a 0 , . . .

, a T −1 } on the environment, for an arbitrary time T .

A first approach, shown in Figure 3 (c), would be to use an action-conditional autoregressive model of observations; initializing the model state h 1 to a function of (s 0 , a 0 ), sample y 1 from p(.|h 1 ), update the state h 2 = f s (h 1 , a 1 , y 1 ), sample y 2 from p(.|h 2 ), and so on until y T is sampled.

In other words, the prediction of observation y T is conditioned on all available observations (s 0 , y <T ) and actions a <T .

This approach is for instance found in (Oh et al., 2015) .

In contrast, another approach is to predict observation y T given actions but using no observation data beyond s 0 .

This family of models, sometimes called models with overshoot, can for instance be found in (Silver et al., 2017b; Oh et al., 2017; Luo et al., 2019; Guo et al., 2018; Hafner et al., 2018; Gregor et al., 2019; Asadi et al., 2019) and is illustrated in Figure 3 (d).

The model deterministically updates its state h t+1 = f h (h t , a t ), and generates y T from p(.|h T ).

An advantage of those models is that they can generate y T directly without generating intermediate observations.

More generally, we define a partial view v t as any function of past observations y ≤t and actions a ≤t .

We define a partial model as a generative model whose predictions are only conditioned on s 0 , the partial views v <t and the actions a <t : to generate y T , the agent generates v 1 from p(.|h 1 ), updates the state to h 2 = f h (h 1 , v 1 , a 1 ), and so on, until it has computed h T and sampled y T from p(.|h T ).

Both previous examples can be seen as special cases of a partial model, with v t = y t and v t = ∅ respectively.

NCPM architecture (overshoot) CPM architecture

A subtle consequence of conditioning the model only on a partial view v t is that the variables y <T become confounders for predicting y T , in addition to the state of the environment which is always a confounder.

In Section 3 we showed that the presence of confounders makes it impossible to correctly predict the target distribution after changes in the covariate distribution.

In the context of partial models, the covariates are the actions a <T executed by the agent and the agent's initial state s 0 , whereas the targets are the predictions y T we want to make at time T .

A corollary of this is that the learned partial model may not be robust against changes in the policy and thus cannot be used to make predictions under different policies π, and therefore should not be used for planning.

In Section 3 we saw that if there was a variable blocking the influence of the confounders on the covariates (a backdoor) or a variable blocking the influence of the covariates on the targets (a frontdoor), it may be possible to make predictions under a broad range of interventions if we learn the correct components from data, e.g. using the backdoor-adgustment formula in Equation (2).

In general it may not be straightforward to apply the backdoor-adjustment formula because we may not have enough access to the graph details to know which variable is a backdoor.

In reinforcement learning however, we can fully control the agent's graph.

This means that we can choose any node in the agent's computational graph that is between its internal state s t and the produced action a t as a backdoor variable for the actions.

Given the backdoor z t , the action a t is conditionally independent of the agent state s t .

To make partial models causally correct, we propose to choose the partial view v t to be equal to the backdoor z t .

This allows us to learn all components we need to make predictions under an arbitrary new policy.

In the rest of this paper we will refer to such models as Causal Partial Models (CPM), and all other partial models will be henceforth referred to as Non-Causal Partial Models (NCPM).

We assume the backdoor z t is sampled from a distribution m(z t |s t ) and the policy is a distribution conditioned on z t , π(a t |z t ).

This is illustrated in Figure 3 (b) and described in more details in Table 1 (right).

We can perform a simulation under a new policy ψ(a t |h t , z t ) by directly applying the backdoor-adjustment formula, Equation (1), to the RL graph as follows:

where the components p(z t |h t ) and p(y t+1 |h t+1 ) with h t+1 = f h (h t , z t , a t ) can be learned from observational data produced by the agent.

Modern deep-learning agents (e.g. as in Espeholt et al. (2018) ; Gregor et al. (2019) ; Ha & Schmidhuber (2018) ) have complex graphs, which means that there are many possible choices for the backdoor z t .

So an important question is: what are the simplest choices of z t ?

Below we list a few of the simplest choices we can use and discuss their advantages and trade-offs; more choices for z t are listed in Appendix C.

Agent state: Identifying z t with the agent's state s t can be very informative about the future, but this comes at a cost.

As part of the generative model, we have to learn the component p(z t |h t ).

This may be difficult in practice when z t = s t due to the high-dimensionality of s t , hence and performing simulations would be computationally expensive.

Policy probabilities: The z t can be the vector of probabilities produced by a policy when we have discrete actions.

The vector of probabilities is informative about the underlying state, if different states produce different probabilities.

Intended action: The z t can be the intended action before using some form of exploration, e.g. ε-greedy exploration.

This is an interesting choice when the actions are discrete, as it is simple to model and, when doing planning, results in a low branching factor which is independent of the complexity of the environment (e.g. in 3D, visually rich environments).

The causal correction methods presented in this section can be applied to any partial model.

In our experiments, we will focus on environment models of the form proposed by Gregor et al. (2019) .

These models consist of a deterministic "backbone" RNN that integrates actions and other contextual information.

The states of this RNN are then used to condition a generative model of the observed data y t , but the observations are not fed back to the model autoregressively, as shown in Table 1 (left).

This corresponds to learning a model of the form p(y t |s 0 , a 0 , . . . , a t−1 ).

We will compare this against our proposed model, which allows us to simulate the outcome of any policy using Equation (4).

In this setup, a policy network produces z t before an action a t .

For example, if the z t is the intended action before ε-exploration, z t will be sampled from a policy m(z t |s t ) and the executed action a t will then be sampled from an ε-exploration policy π(a t |z t ) = (1 − ε)δ zt,at + ε 1 na , where n a is the number of actions and ε is in (0, 1).

Acting with the sampled actions is diagrammed in Figure 3 (b) and the mathematical description is provided in Table 1 .

The model components p(z t |h t ) and p(y t |h t ) are trained via maximum likelihood on observational data collected by the agent.

The partial model does not need to model all parts of the y t observation.

For example, a model to be used for planning can model just the reward and the expected return.

In any case, it is imperative that we use some form of exploration to ensure that π(a t |z t ) > 0 for all a t and z t as this is a necessary to allow the model to learn the effects of the actions.

The model usage is summarized in Algorithms 1 and 2 in Appendix D and we discuss the model properties in Appendix E.

We analyse the effect of the proposed corrections on a variety of models and environments.

When the enviroment is an MDP, such as the FuzzyBear MDP from Section 2, we can compute exactly both the non-causal and the causal model directly from the MDP transition matrix and the behavior policy.

In Section 5.1, we compare the optimal policies computed from the non-causal and the causal model via value iteration.

For this analysis, we used the intended-action backdoor, since it's compatible with a tabular representation.

In Section 5.2, we repeat the analysis using a learned model instead.

For these experiments, we used the policy-probabilities backdoor.

The optimal policies corresponding to a given model were computed using a variant of the Dyna algorithm (Sutton, 1991) or expectimax (Michie, 1966) .

Finally in Section 5.3, we provide an analysis of the model rollouts in a visually rich 3D environment.

Given an MDP and a behavior policy π, the optimal values V * M (π) of planning based on a NCPM and CPM are derived in Appendix I. The theoretical analysis of the MDP does not use empirically trained models from the policy data, but rather assumes that the transition probabilities of the MDP and the policy from which training data are collected are accurately learned by the model.

This allows us to isolate the quality of planning using the model from how accurate the model is.

Optimal behavior policy: The optimal policy of the FuzzyBear MDP (Figure 1(a) ) is to always hug the teddy bear and run away from the grizzly bear.

Using training data from this behavior policy, we show in Figure 7 (Appendix I) the difference in the optimal planning based on the NCPM (Figure 3(d) ) and CPM with the backdoor z t being the intended action (Figure 3(e) ).

Learning from optimal policies with ε-exploration, the converged causal model is independent of the exploration parameter ε.

We see effects of varying ε on learned models in Figure 8 (Appendix I).

Sub-optimal behavior policies: We empirically show the difference between the causal and noncausal models when learning from randomly generated policies.

For each policy, we derive the corresponding converged model M (π) using training data generated by the policy.

We then compute the optimal value of V * M (π) using this model.

On FuzzyBear (Figure 4(a) ), we see that the causal model always produces a value greater than or equal to the value of the behavior policy.

The value estimated by the causal model can always be achieved in the real environment.

If the behavior policy was already good, the simulation policy used inside the model can reproduce the behavior policy by respecting the intended action.

If the behavior policy is random, the intended action is uninformative about the underlying state, so the simulation policy has to choose the most rewarding action, independently of the state.

And if the behavior policy is bad, the simulation policy can choose the opposite of the intended action.

This allows to find a very good simulation policy, when the behavior policy is very bad.

To further improve the policy, the search for better policies should be done also in state s 1 .

And the model can then be retrained on data from the improved policies.

If we look at the non-causal model, we see that it displays the unfortunate property of becoming more unrealistically optimistic as the behavior policy becomes better.

Similarly, the worse the policy is, i.e. the lower V π env is, the non-causal model becomes less able to improve the policy.

On AvoidFuzzyBear (Figure 4(b) ), the optimal policy is to stay at home.

Learning from data generated by random policies, the causal model indeed always prefers to stay home with any sampled intentions, resulting in a constant evaluation for all policies.

On the other hand, the non-causal model gives varied, overly-optimistic evaluations, while choosing the wrong action (visit forest).

We previously analyzed the case where the transition probabilities and theoretically optimal policy are known.

We will now describe experiments with learned models trained by gradient descent, using the same training setup as described in Section 4.

In this experiment we demonstrate that we can learn the optimal policy purely from off-policy experience using a general n-step-return algorithm derived from a causal model.

The algorithm is described in detail in Appendix F. In short, we simulate experiences from the partial model, and use policy gradient to learn the optimal policy on these experiences as if they were real experiences (this is possible since the policy gradient only needs action probabilities, values, predicted rewards and ends of episodes).

We compare a non-causal model and a causal model where the backdoor z t is the intended action.

For the environment we use AvoidFuzzyBear (Figure 1(b) ).

We collect experiences that are sub-optimal: half the time the agent visits the forest and half the time it stays home, but once in the forest it acts optimally with probability 0.9.

This is meant to simulate situations either where the agent has not yet learned the optimal policy but is acting reasonably, or where it is acting with a different objective (such as exploration or intrinsic reward), but would like to derive the optimal policy.

We expect the non-causal model to choose the sub-optimal policy of visiting the forest, since the sequence of actions of visiting the forest and hugging typically yields high reward.

This is what we indeed find, as shown in Figure 5(a) .

We see that the non-causal model indeed achieves a sub-optimal reward (less than 0.6), but believes that it will achieve a high reward (more than 0.6).

On the other hand, the causal model achieves the optimal reward and correctly predicts that it will achieve the corresponding value.

AvoidFuzzyBear with Expectimax: In this experiment, we used the classical expectimax search (Michie, 1966; Russell & Norvig, 2009 ).

On the simple AvoidFuzzyBear MDP, it is enough to use a search depth of 3: a decision node, a chance node and a decision node.

The behavior policy was progressively improving as the model was trained.

In Figure 5 (b), we see the results for the different models.

Only the non-causal model was not able to solve the task.

Planning with the non-causal model consistently preferred the stochastic path with the fuzzy bear, as predicted by our theoretical analysis with value iteration.

The models with clustered probabilities and clustered observations approximate modeling of the probabilities or observations.

These models are described in Appendix H.

The setup for these experiments is similar to Gregor et al. (2019) , where the agent is trained using the IMPALA algorithm (Espeholt et al., 2018) , and the model is trained alongside the agent via ELBO optimization on the data collected by the agent.

The architecture of the agent and model is based on Gregor et al. (2019) and follows the description in Table 1 (right).

For these experiments, the backdoor z t was chosen to be the policy probabilities, and p(z t |h t ) was parametrized as a mixture of Dirichlet distributions.

See Appendix J for more details.

We demonstrate the effect of the causal correction on the 3D T-Maze environment where an agent walks around in a 3D world with the goal of collecting the reward blocks (food).

The layout of this environment is shown in Figure 6 (a).

From our previous results, we expect NCPMs to be unrealistically optimistic.

This is indeed what we see in Figure 6 (b).

Compared to NCPM, CPM with generated z generates food at the end of a rollout with around 50% chance, as expected given that the environment randomly places the food on either side.

In Figure 6 (c) and Figure 6 In all rollouts depicted, the top row shows the real frames observed by an agent following a fixed policy (Ground Truth, GT).

Bottom 5 rows indicate model rollouts, conditioned on 3 previous frames without revealing the location of the food.

CPM and NCPM differ in their state-update formula and action generation (see Table 1 ), but frame generation yt ∼ p(yt|ht) is the same for both, as introduced in Gregor et al. (2019) .

For CPM, we compare rollouts with forced actions and generated z to rollouts with forced actions and forced z from the ground truth.

We can observe that rollouts with the generated z (left) respect the randomness in the food placement (with and without food), while the rollouts with forced z (right) consistently generate food blocks, if following actions consistent with the backdoor z from the well-trained ground truth policy.

We have characterized and explained some of the issues of partial models in terms of causal reasoning.

We proposed a simple, yet effective, modification to partial models so that they can still make correct predictions under changes in the behavior policy, which we validated theoretically and experimentally.

The proposed modifications address the correctness of the model against policy changes, but don't address the correctness/robustness against other types of intervention in the environment.

We will explore these aspects in future work.

Starting from a data-generation process of the form illustrated in Figure 2 (b), p(x, y, u) = p(u)p(x|u)p(y|x, u), we can use the do-operator to compute p(y|do(x)) = p(u)p(y|x, u)du.

Without assuming any extra structure in p(x|u) or in p(y|x, u) it is not possible to compute p(y|do(x)) from the knowledge of the joint p(x, y) alone.

If there was a variable z blocking all the effects of u on x, as illustrated in Figure 2(d) , then p(y|do(x)) can be derived as follows:

Conditioning the new joint

where we used the formula

If instead of just fixing the value of x, we perform a more general intervention p(x|z) → ψ(x|z), then p do(ψ(x|z)) (y) can be derived as follows:

New marginal

Applying the same reasoning to the graph shown in Figure 2 (e), we obtain the formula

where p(z|x), p(x ) and p(y|x , z) can be directly measured from the available (x, y, z) data.

This formula holds as long as p(z|x) > 0, ∀x, z and it is a simple instance of frontdoor adjustment (Pearl et al., 2016) .

Here, we will show in more detail that the models (c) and (e) in Figure 3 are causally correct, whereas model (d) is causally incorrect.

Specifically, we will show that given an initial state s 0 and after setting the actions a 0 to a T to specific values, models (c) and (e) make the same prediction about the future observation y T +1 as performing the intervention in the real world, whereas model (d) does not.

Using the do-operator, a hard intervention in the model is given by:

where h t is a deterministic function of s 0 , a 0:t−1 and y 1:t−1 .

The same hard intervention in the real world is given by:

p(y t |s 0 , a 0:t−1 , y 1:t−1 ) dy 1:T .

If the model is trained perfectly, the factors q θ (y t |h t ) will become equal to the conditionals p(y t |s 0 , a 0:t−1 , y 1:t−1 ).

Hence, an intervention in a perfectly trained model makes the same prediction as in the real world, which means that the model is causally correct.

The interventional conditional in the model is simply:

where h T +1 is a deterministic function of s 0 and a 0:T .

In a perfectly trained model, we have that q θ (y T +1 |h T +1 ) = p(y T +1 |s 0 , a 0:T ).

However, the observational conditional p(y T +1 |s 0 , a 0:T ) is not generally equal to the inverventional conditional p(y T +1 |s 0 , do(a 0:T )), which means that the model is causally incorrect.

Model (e) Finally, the interventional conditional in this model is:

where h t is a deterministic function of s 0 , a 0:t−1 and z 1:t−1 .

The same intervention in the real world can be written as follows:

In a perfectly trained model, we have that q θ (y T +1 |h T +1 ) = p(y T +1 |s 0 , a 0:T , z 1:T ) and q θ (z t |h t ) = p(z t |s 0 , a 0:t−1 , z 1:t−1 ).

That means that the intervention in a perfectly trained model makes the same prediction as the same intervention in the real world, hence the model is causally correct.

The first alternative backdoor we consider is the empty backdoor:

Empty backdoor z t = ∅: This backdoor is in general not appropriate; it is however appropriate when the behavior policy does in fact depend on no information, i.e. is not a function of the state s t .

For example, the policy can be uniformly random (or any non-state dependent distribution over actions).

This severely limits the behavior policy.

Because the backdoor contains no information about the observations, the simulations are open-loop, i.e. we can only consider plans which consist of a sequence of fixed actions, not policies.

In principle, the z t can be any layer from the policy.

To model the layer with a p(z t |h t ) distribution, we would need to know the needed numerical precision of the considered layer.

For example, a quantized layer can be modeled by a discrete distribution.

Alternatively, if the layer is produced by a variational encoder or variational information bottleneck, we can train p(z t |h t ) to minimize the KL(p encoder (z t |s t ) p(z t |h t )).

Finally, if a backdoor is appropriate, we can combine it with additional information:

Combinations: It is possible to combine a layer with information from other layers.

For example, the intended action can be combined with extra bits from the input layer.

Such z t can be more informative.

For example, the extra bits can hold a downsampled and quantized version of the input layer.

Algorithms 1 and 2 describe how the model is trained and used to simulate trajectories.

The algorithm for training assumes a distributed actor-learner setup (Espeholt et al., 2018) .

Data collection on an actor: For each step: z t ∼ m(z t |s t ) . . .

sample the backdoor (e.g., the partial view with the intended action) a t ∼ π(a t |z t ) . . .

sample the executed action (e.g., add ε-exploration) Table 2 provides an overview of properties of autoregressive models, deterministic non-causal models and the causal partial models.

The causal partial models have to generate only a partial view.

The partial view can be small and easy to model.

For example, a partial view with the discrete intended action can be flexibly modeled by a categorical distribution.

The causal partial models are fast and causally correct in stochastic environments.

The causal partial models have a low simulation variance, because they do not need to model and generate unimportant background distractors.

If the environment has deterministic regions, the model can quickly learn to ignore the small partial view and collect information only from the executed action.

It is interesting that the causal partial models are invariant of the π(a t |z t ) distribution.

For example, if the partial view z t is the intended action, the optimally learned model would be invariant of the used ε-exploration: π(a t |z t ).

Analogously, the autoregressive models are invariant of the whole policy π(a t |s t ).

This allows the autoregressive models to evaluate any other policy inside of the model.

The causal partial model can run inside the simulation only policies conditioned on the starting state s 0 , the actions a <t and the partial views z ≤t .

If we want to evaluate a policy conditioned on different features, we can collect trajectories from the policy and retrain the model.

The model can always evaluate the policy used to produce the training data.

We can also improve the policy, because the model allows to estimate the return for an initial (s 0 , a 0 ) pair, so the model can be used as a critic for a policy improvement.

In this section we derive an algorithm for learning an optimal policy given a (non-optimal) experience that utilizes n-step returns from partial models presented in this paper.

In general, a model of the environment can be used in a number of ways for reinforcement learning.

In Dyna (Sutton, 1990) , we sample experiences from the model, and apply a model-free algorithm (Q-learning in the original implementation, but more generally we could consider SARSA or policy gradient) as if these were real experiences.

In Dyna-2 (Silver et al., 2008) , the same process is applied but in the context the agent is currently in-starting the simulations from the current state-and adapting the policy locally (for example through separate fast weights).

In MCTS, the model is used to build a tree of possibilities.

Can we apply our model directly in these scenarios?

While we don't have a full model of the environment, we can produce a causally correct simulation of rewards and values; one that should generalize to policies different from those the agent was trained on.

Policy probabilities, values, rewards and ends of episodes are the only variables that the above RL algorithms need.

Here we propose a specific implementation of Dyna-style policy-gradient algorithm based on the models discussed in the paper.

This is meant as a proof of principle, and more exploration is left for future work.

As the agent sees an observation y t+1 , it forms an internal agent state s t from this observation and the previous agent state: s t+1 = RNN s (s t , a t , y t+1 ).

The agent state in our implementation is the state of the recurrent network, typically LSTM (Hochreiter & Schmidhuber, 1997) .

Next, let us assume that at some point in time with state s 0 the agent would like to learn to do a simulation from the model.

Let h t be the state of the simulation at time t. The agent first sets h 1 = g(s 0 , a 0 ) and proceeds with n-steps of the simulation recurrent network update h t+1 = RNN(h t , z t , a t ).

The agent learns the model p(z t |h t ) which it can use to simulate forward.

We assume that the model was trained on some (non-optimal) policy/experience.

We would like to derive an optimal policy and value function.

Since these need to be used during acting (if the agent were to then act optimally in the real environment), they are functions of the agent state s t : π(a t |s t ), V (s t ).

Now in general, h t = s t but we would like to use the simulation to train an optimal policy and value function.

Thus we define a second pair of functions π h (a t |h t , z t ), V h (h t , z t ).

Here the extra z t 's are needed, since the h t has seen z's only up to point z t−1 .

Next we are going to train these functions using policy gradients on simulated experiences.

We start with some state s t and produce a simulation h t+1 , . . .

, h T by sampling z t from the model at each step and action a t ∼ π h (a t |h t , z t ).

However at the initial point t, we sample from π, not π h , and compute the value V , not V h .

Sequence of actions, values and policy parameters are the quantities needed to compute a policy gradient update.

We use this update to train all these quantities.

There is one last element that the algorithm needs.

The values and policy parameters are trained at the start state and along the simulation by n-step returns, computed from simulated rewards and the bootstrap value at the end of the simulation.

However this last value is not trained in any way because it depends on the simulated state V h (h T ) not the agent state s T .

We would like this value to equal to what the agent state would produce: V (s T ).

Thus, during training of the model, we also train V h (h T ) to be close to V (s T ) by imposing an L 2 penalty.

In our implementation, we actually impose a penalty at every point t during simulation but we haven't experimented with which choice is better.

Variance reduction.

To reduce the variance of a simulation, it is possible to sample the z t from a proposal distribution q(z t |h t ).

The correct expectation can be still recovered by using an importance weight: w = p(zt|ht) q(zt|ht) .

Data efficient training.

Usually, we know the distribution of the used partial view: z t ∼ m(z t |s t ).

When training the p(z t |h t ) model, we can then minimize the exact KL(m(Z t |s t ) p(Z t |h t )).

When using a tree-search, we want to have a small branching factor at the chance nodes.

A good z t variable would be discrete with a small number of categories.

This is satisfied, if the z t is the intended action and the number of the possible actions is small.

We do not have such compact discrete z t , if using as z t the observation, the policy probabilities or some other modeled layer.

Here, we will present a model that approximates such causal partial models.

The idea is to cluster the modeled layers and use just the cluster index as z t .

The cluster index is discrete and we can control the branching factor by choosing the the number of clusters.

Concretely, let's call the modeled layer x t .

We will model the layer with a mixture of components.

The mixture gives us a discrete latent variable z t to represent the component index.

To train the mixture, we use a clustering loss to train only the best component to model the x t , given h t and z t :

where p(z t |h t ) is a model of the categorical component index and β clustering ∈ (0, 1) is a hyperparameter to encourage moving the information bits to the latent z t .

During training, we use the index of the best component as the inferred z t .

In theory, a better inference can be obtained by smoothing.

In contrast to training by maximum likelihood, the clustering loss uses just the needed number of the mixture components.

This helps to reduce the branching factor in a search.

In general, the cluster index is not guaranteed to be sufficient as a backdoor, if the reconstruction loss − log p(x t |h t , z t ) is not zero.

For example, if x t is the next observation, the number of mixture components may need to be unrealistically large, if the observation can contains many distractors.

We derive the following two model-based evaluation metrics for the MDP environments.

• V * NCPM(π) (s 0 ): optimal value computed with the non-causal model, which is trained with training data from policy π, starting from state s 0 .

• V * CPM(π) (s 0 ): optimal value computed with the causal model, which is trained with training data from policy π, starting from state s 0 .

The theoretical analysis of the MDP does not use empirically trained models from the policy data but rather assumes that the transition probabilities p(s i+1 | s i , a i ) of the MDP, and the policy, π(a i | s i ) or π(z i | s i ), from which training data are collected are accurately learned by the model.

Notice that the probability of s i is affected by a i here, because the network gets a i as an input, when predicting the r i+1 .

This will introduce the non-causal bias.

The network implements the expectation implicitly by learning the mean of the reward seen in the training data.

We can compute the expectation exactly, if we know the MDP.

The p(s i | s 0 , a 0 , . . . , a i ) can be computed recursively in two-steps as:

Here, we see the dependency of the learned model on the policy π.

The remaining terms can be expressed as:

Denoting p(s i | s 0 , a 0 , . . . , a j ) by S i,j , we have the two-step recursion

with

where

Denoting p(s i | s 0 , a 0 , z 1 . . .

, z i−1 , a i−1 ) by Z i , we have

where we used the fact that s i−1 is independent of a i−1 , given z i−1 .

Furthermore,

Therefore we can compute Z i recursively,

with In Figure 7 (a), the non-causal agent always chooses hug at step t = 1, since it has learned from the optimal policy that a reward of +1 always follows after taking a 1 = hug.

Thus from the noncausal agent's point of view, the expected reward is always 1 after hugging.

This is wrong since only hugging a teddy bear gives reward 1.

Moreover it exceeds the maximum expected reward 0.5 of the FuzzyBear MDP.

In Figure 7 (b), the causal agent first samples the intention z 1 from the optimal policy, giving equal probability of landing in either of the two chance nodes.

Then it chooses hug if z 1 = 0, indicating a teddy bear since the optimal policy intends to hug only if it observes a teddy bear.

Likewise, it chooses run if z 1 = 1, indicating a grizzly bear.

While the non-causal model expects unrealistically high reward, the causal model never over-estimates the expected reward.

We analyze learning from optimal policy with varying amounts of ε-exploration for models on FuzzyBear (Figure 8(a) ) and AvoidFuzzyBear (Figure 8(b) ).

As the parameter ε-exploration varies in range (0, 1], the causal model has a constant evaluation since the intended action is not affected by the randomness in exploration.

The non-causal model, on the other hand, evaluates based on the deterministic optimal policy data (i.e. at ε = 0) at an unrealistically high value of 1.0 when the maximum expected reward is 0.5.

As ε → 1, the training data becomes more random and its optimal evaluation expectantly goes down to match the causal evaluation based on a uniformly random policy.

The causal evaluation based on the optimal policy V gives an unrealistically high value 1.0 learned from the deterministic optimal policy (ε = 0).

Expectantly, it decreases to the level of CPM optimal value V * CPM(π rand ) learned from the uniformly random policy as ε → 1.

The CPM optimal values V * CPM(π * ) are constant for any value of ε based on the theoretical analysis in Section I.1.

(b) shows the same plots as (a) for the AvoidFuzzyBear environment.

Learning from any policy π, the CPM optimal value always equals the maximum expected reward 0.6, by correctly choosing to stay home.

When the backdoor variable z t was chosen to be the action probabilities, the distribution p(z t |h t ) was chosen as a mixture-network with N c Dirichlet components.

The concentration parameters α k (h t ) of each component were parametrized as α k (h t ) = α softmax(f k (h t )), where f k is the output of a relu-MLP with layer sizes [256, 64, N c × N a ], α is a total concentration parameter and N a is the number of actions.

The hyper-parameter value ranges used in our 3D experiments are similar to Gregor et al. (2019) and are shown in Table 3 .

To speed up training, we interleaved training on the T-maze level with a simple "Food" level, in which the agent simply had to walk around and eat food blocks (described by Gregor et al. (2019) ).

For each episode, 5 rollouts are generated after having observed the first 3 frames from the environment.

For the 5 rollouts, we processed the first 25 frames to classify the presence of food blocks by performing color matching of RGB values, using K-means and assuming 7 clusters.

Rollouts were generated shortly after the policy had achieved ceiling performance (15-20 million frames seen), but before the entropy of the policy reduces to the point that there is no longer sufficient exploration.

See Figure 9 for these same results for later training.

action + forced z action + gen.

z Figure 9 : While earlier in training, CPM generates a diverse range of outcomes (food or no food), as the policy becomes more deterministic (as seen in the right plot of the policy entropy over training), CPM starts to generate more food and becomes overoptimistic, similar to NCPM.

This can be avoided by training the model with non-zero ε-exploration.

<|TLDR|>

@highlight

Causally correct partial models do not have to generate the whole observation to remain causally correct in stochastic environments.