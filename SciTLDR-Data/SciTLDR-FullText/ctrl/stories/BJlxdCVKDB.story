Deep Reinforcement Learning (DRL) has led to many recent breakthroughs on complex control tasks, such as defeating the best human player in the game of Go.

However, decisions made by the DRL agent are not explainable, hindering its applicability in safety-critical settings.

Viper, a recently proposed technique, constructs a decision tree policy by mimicking the DRL agent.

Decision trees are interpretable as each action made can be traced back to the decision rule path that lead to it.

However, one global decision tree approximating the DRL policy has significant limitations with respect to the geometry of decision boundaries.

We propose MoET, a more expressive, yet still interpretable model based on Mixture of Experts, consisting of a gating function that partitions the state space, and multiple decision tree experts that specialize on different partitions.

We propose a training procedure to support non-differentiable decision tree experts and integrate it into imitation learning procedure of Viper.

We evaluate our algorithm on four OpenAI gym environments, and show that the policy constructed in such a way is more performant and better mimics the DRL agent by lowering mispredictions and increasing the reward.

We also show that MoET policies are amenable for verification using off-the-shelf automated theorem provers such as Z3.

Deep Reinforcement Learning (DRL) has achieved many recent breakthroughs in challenging domains such as Go (Silver et al., 2016) .

While using neural networks for encoding state representations allow DRL agents to learn policies for tasks with large state spaces, the learned policies are not interpretable, which hinders their use in safety-critical applications.

Some recent works leverage programs and decision trees as representations for interpreting the learned agent policies.

PIRL (Verma et al., 2018) uses program synthesis to generate a program in a Domain-Specific Language (DSL) that is close to the DRL agent policy.

The design of the DSL with desired operators is a tedious manual effort and the enumerative search for synthesis is difficult to scale for larger programs.

In contrast, Viper (Bastani et al., 2018 ) learns a Decision Tree (DT) policy by mimicking the DRL agent, which not only allows for a general representation for different policies, but also allows for verification of these policies using integer linear programming solvers.

Viper uses the DAGGER (Ross et al., 2011) imitation learning approach to collect state action pairs for training the student DT policy given the teacher DRL policy.

It modifies the DAGGER algorithm to use the Q-function of teacher policy to prioritize states of critical importance during learning.

However, learning a single DT for the complete policy leads to some key shortcomings such as i) less faithful representation of original agent policy measured by the number of mispredictions, ii) lower overall performance (reward), and iii) larger DT sizes that make them harder to interpret.

In this paper, we present MOËT (Mixture of Expert Trees), a technique based on Mixture of Experts (MOE) (Jacobs et al., 1991; Jordan and Xu, 1995; Yuksel et al., 2012) , and reformulate its learning procedure to support DT experts.

MOE models can typically use any expert as long as it is a differentiable function of model parameters, which unfortunately does not hold for DTs.

Similar to MOE training with Expectation-Maximization (EM) algorithm, we first observe that MOËT can be trained by interchangeably optimizing the weighted log likelihood for experts (independently from one another) and optimizing the gating function with respect to the obtained experts.

Then, we propose a procedure for DT learning in the specific context of MOE.

To the best of our knowledge we are first to combine standard non-differentiable DT experts, which are interpretable, with MOE model.

Existing combinations which rely on differentiable tree or treelike models, such as soft decision trees (Irsoy et al., 2012) and hierarchical mixture of experts (Zhao et al., 2019) are not interpretable.

We adapt the imitation learning technique of Viper to use MOËT policies instead of DTs.

MOËT creates multiple local DTs that specialize on different regions of the input space, allowing for simpler (shallower) DTs that more accurately mimic the DRL agent policy within their regions, and combines the local trees into a global policy using a gating function.

We use a simple and interpretable linear model with softmax function as the gating function, which returns a distribution over DT experts for each point in the input space.

While standard MOE uses this distribution to average predictions of DTs, we also consider selecting just one most likely expert tree to improve interpretability.

While decision boundaries of Viper DT policies must be axis-perpendicular, the softmax gating function supports boundaries with hyperplanes of arbitrary orientations, allowing MOËT to more faithfully represent the original policy.

We evaluate our technique on four different environments: CartPole, Pong, Acrobot, and Mountaincar.

We show that MOËT achieves significantly better rewards and lower misprediction rates with shallower trees.

We also visualize the Viper and MOËT policies for Mountaincar, demonstrating the differences in their learning capabilities.

Finally, we demonstrate how a MOËT policy can be translated into an SMT formula for verifying properties for CartPole game using the Z3 theorem prover (De Moura and Bjørner, 2008) under similar assumptions made in Viper.

In summary, this paper makes the following key contributions: 1) We propose MOËT, a technique based on MOE to learn mixture of expert decision trees and present a learning algorithm to train MOËT models.

2) We use MOËT models with a softmax gating function for interpreting DRL policies and adapt the imitation learning approach used in Viper to learn MOËT models.

3) We evaluate MOËT on different environments and show that it leads to smaller, more faithful, and performant representations of DRL agent policies compared to Viper while preserving verifiability.

Interpretable Machine Learning:

In numerous contexts, it is important to understand and interpret the decision making process of a machine learning model.

However, interpretability does not have a unique definition that is widely accepted.

Accoding to Lipton (Lipton, 2016) , there are several properties which might be meant by this word and we adopt the one which Lipton names transparency which is further decomposed to simulability, decomposability, and algorithmic transparency.

A model is simulable if a person can in reasonable time compute the outputs from given inputs and in that way simulate the model's inner workings.

That holds for small linear models and small decision trees (Lipton, 2016) .

A model is decomposable if each part of a models admits an intuitive explanation, which is again the case for simple linear models and decision trees (Lipton, 2016) .

Algorithmic transparency is related to our understanding of the workings of the training algorithm.

For instance, in case of linear models the shape of the error surface and properties of its unique minimum towards which the algorithm converges are well understood (Lipton, 2016) .

MOËT models focus on transparency (as we discuss at the end of Section 5).

Explainable Machine Learning: There has been a lot of recent interest in explaining decisions of black-box models (Guidotti et al., 2018a; Doshi-Velez and Kim, 2017) .

For image classification, activation maximization techniques can be used to sample representative input patterns (Erhan et al., 2009; Olah et al., 2017) .

TCAV uses human-friendly high-level concepts to associate their importance to the decision.

Some recent works also generate contrastive robust explanations to help users understand a classifier decision based on a family of neighboring inputs (Zhang et al., 2018; Dhurandhar et al., 2018) .

LORE (Guidotti et al., 2018b) explains behavior of a blackbox model around an input of interest by sampling the black-box model around the neighborhood of the input, and training a local DT over the sampled points.

Our model presents an approach that combines local trees into a global policy.

Tree-Structured Models:

Irsoy et al. (Irsoy et al., 2012) propose a a novel decision tree architecture with soft decisions at the internal nodes where both children are chosen with probabilities given by a sigmoid gating function.

Similarly, binary tree-structured hierarchical routing mixture of experts (HRME) model, which has classifiers as non-leaf node experts and simple regression models as leaf node experts, were proposed in (Zhao et al., 2019) .

Both models are unfortunately not interpretable.

Knowledge Distillation and Model Compression: We rely on ideas already explored in fields of model compression (Bucilu et al., 2006) and knowledge distillation (Hinton et al., 2015) .

The idea is to use a complex well performing model to facilitate training of a simpler model which might have some other desirable properties (e.g., interpretability).

Such practices have been applied to approximate decision tree ensemble by a single tree (Breiman and Shang, 1996) , but this is different from our case, since we approximate a neural network.

In a similar fashion a neural network can be used to train another neural network (Furlanello et al., 2018) , but neural networks are hard to interpret and even harder to formally verify, so this is also different from our case.

Such practices have also been applied in the field of reinforcement learning in knowledge and policy distillation (Rusu et al., 2016; Koul et al., 2019; Zhang et al., 2019) , which are similar in spirit to our work, and imitation learning (Bastani et al., 2018; Ross et al., 2011; Abbeel and Ng, 2004; Schaal, 1999) , which provide a foundation for our work.

We now present a simple motivating example to showcase some of the key differences between Viper and MOËT approaches.

Consider the N × N Gridworld problem shown in Figure 1a (for N = 5).

The agent is placed at a random position in a grid (except the walls denoted by filled rectangles) and should find its way out.

To move through the grid the agent can choose to go up, left, right or down at each time step.

If it hits the wall it stays in the same position (state).

State is represented using two integer values (x, y coordinates) which range from (0, 0)-bottom left to (N − 1, N − 1)-top right.

The grid can be escaped through either left doors (left of the first column), or right doors (right of the last column).

A negative reward of −0.1 is received for each agent action (negative reward encourages the agent to find the exit as fast as possible).

An episode finishes as soon as an exit is reached or if 100 steps are made whichever comes first.

The optimal policy (π * ) for this problem consists of taking the left (right resp.) action for each state below (above resp.) the diagonal.

We used π * as a teacher and used imitation learning approach of Viper to train an interpretable DT policy that mimics π * .

The resulting DT policy is shown in Figure 1b .

The DT partitions the state space (grid) using lines perpendicular to x and y axes, until it separates all states above diagonal from those below.

This results in a DT of depth 3 with 9 nodes.

On the other hand, the policy learned by MOËT is shown in Figure 1c .

The MOËT model with 2 experts learns to partition the space using the line defined by a linear function 1.06x + 1.11y = 4 (roughly the diagonal of the grid).

Points on the different sides of the line correspond to two different experts which are themselves DTs of depth 0 always choosing to go left (below) or right (above).

We notice that DT policy needs much larger depth to represent π * while MOËT can represent it as only one decision step.

Furthermore, with increasing N (size of the grid), complexity of DT will grow, while MOËT complexity stays the same; we empirically confirm this for N = [5, 10] .

5, 6, 7, 8, 9 , 10 DT depths are 3, 4, 4, 4, 4, 5 and number of nodes are 9, 11, 13, 15, 17, 21 respectively.

In contrast, MOËT models of same complexity and structure as the one shown in Figure 1c are learned for all values of N (models differ in the learned partitioning linear function).

In this section we provide description of two relevant methods we build upon: (1) Viper, an approach for interpretable imitation learning, and (2) MOE learning framework.

Viper.

Viper algorithm (included in appendix) is an instance of DAGGER imitation learning approach, adapted to prioritize critical states based on Q-values.

Inputs to the Viper training algorithm are (1) environment e which is an finite horizon (T -step) Markov Decision Process (MDP) (S, A, P, R) with states S, actions A, transition probabilities P : S × A × S → [0, 1], and rewards R : S → R; (2) teacher policy π t : S → A; (3) its Q-function Q πt : S × A → R and (4) number of training iterations N .

Distribution of states after T steps in environment e using a policy π is d (π) (e) (assuming randomly chosen initial state).

Viper uses the teacher as an oracle to label the data (states with actions).

It initially uses teacher policy to sample trajectories (states) to train a student (DT) policy.

It then uses the student policy to generate more trajectories.

Viper samples training points from the collected dataset D giving priority to states s having higher importance I(s), where

.

This sampling of states leads to faster learning and shallower DTs.

The process of sampling trajectories and training students is repeated for number of iterations N , and the best student policy is chosen using reward as the criterion.

MOE is an ensemble model (Jacobs et al., 1991; Jordan and Xu, 1995; Yuksel et al., 2012 ) that consists of expert networks and a gating function.

Gating function divides the input (feature) space into regions for which different experts are specialized and responsible.

MOE is flexible with respect to the choice of expert models as long as they are differentiable functions of model parameters (which is not the case for DTs).

In MOE framework, probability of outputting y ∈ IR m given an input x ∈ IR n is given by:

where E is the number of experts, g i (x, θ g ) is the probability of choosing the expert i (given input x), P (y|x, θ i ) is the probability of expert i producing output y (given input x).

Learnable parameters are θ = (θ g , θ e ), where θ g are parameters of the gating function and θ e = (θ 1 , θ 2 , ..., θ E ) are parameters of the experts.

Gating function can be modeled using a softmax function over a set of linear models.

Let θ g consist of parameter vectors (θ g1 , . . .

, θ gE ), then the gating function can be defined as

.

In the case of classification, an expert i outputs a vector y i of length C, where C is the number of classes.

Expert i associates a probability to each output class c (given by y ic ) using the gating function.

Final probability of a class c is a gate weighted sum of y ic for all experts i ∈ 1, 2, ..., E. This creates a probability vector y = (y 1 , y 2 , ..., y C ), and the output of MOE is arg max i y i .

MOE is commonly trained using EM algorithm, where instead of direct optimization of the likelihood one performs optimization of an auxiliary functionL defined in a following way.

Let z denote the expert chosen for instance x.

Then joint likelihood of x and z can be considered.

Since z is not observed in the data, log likelihood of samples (x, z, y) cannot be computed, but instead expected log likelihood can be considered, where expectation is taken over z. Since the expectation has to rely on some distribution of z, in the iterative process, the distribution with respect to the current estimate of parameters θ is used.

More precisely functionL is defined by (Jordan and Xu, 1995) :

where θ (k) is the estimate of parameters θ in iteration k. Then, for a specific sample D = {(x i , y i ) | i = 1, . . .

, N }, the following formula can be derived (Jordan and Xu, 1995) :

where it holds

In this section we explain the adaptation of original MOE model to mixture of decision trees, and present both training and inference algorithms.

Considering that coefficients h (k) ij (Eq. 4) are fixed with respect to θ and that in Eq. 3 the gating part (first double sum) and each expert part depend on disjoint subsets of parameters θ, training can be carried out by interchangeably optimizing the weighted log likelihood for experts (independently from one another) and optimizing the gating function with respect to the obtained experts.

The training procedure for MOËT, described by Algorithm 1, is based on this observation.

First, the parameters of the gating function are randomly initialized (line 2).

Then the experts are trained one by one.

Each expert j is trained on a dataset D w of instances weighted by coefficients h (k) ij (line 5), by applying specific DT learning algorithm (line 6) that we adapted for MOE context (described below).

After the experts are trained, an optimization step is performed (line 7) in order to increase the gating part of Eq. 3.

At the end, the parameters are returned (line 8).

Our tree learning procedure is as follows.

Our technique modifies original MOE algorithm in that it uses DTs as experts.

The fundamental difference with respect to traditional model comes from the fact that DTs do not rely on explicit and differentiable loss function which can be trained by gradient descent or Newton's methods.

Instead, due to their discrete structure, they rely on a specific greedy training procedure.

Therefore, the training of DTs has to be modified in order to take into account the attribution of instances to the experts given by coefficients h (k) ij , sometimes called responsibility of expert j for instance i. If these responsibilities were hard, meaning that each instance is assigned to strictly one expert, they would result in partitioning the feature space into disjoint regions belonging to different experts.

On the other hand, soft responsibilities are fractionally distributing each instance to different experts.

The higher the responsibility of an expert j for an instance i, the higher the influence of that instance on that expert's training.

In order to formulate this principle, we consider which way the instance influences construction of a tree.

First, it affects the impurity measure computed when splitting the nodes and second, it influences probability estimates in the leaves of the tree.

We address these two issues next.

A commonly used impurity measure to determine splits in the tree is the Gini index.

Let U be a set of indices of instances assigned to the node for which the split is being computed and D U set of corresponding instances.

Let categorical outcomes of y be 1, . . .

, C and for l = 1, . . .

, C denote p l fraction of assigned instances for which it holds y = l. More formally

, where I denotes indicator function of its argument expression and equals 1 if the expression is true.

Then the Gini index G of the set D U is defined by:

Considering that the assignment of instances to experts are fractional as defined by responsibility coefficients h (k) ij (which are provided to tree fitting function as weights of instances computed in line 5 of the algorithm), this definition has to be modified in that the instances assigned to the node should not be counted, but instead, their weights should be summed.

Hence, we propose the following definition:

and compute the Gini index for the set D U as G(p 1 , . . .

,p C ).

Similar modification can be performed for other impurity measures relying on distribution of outcomes of a categorical variable, like entropy.

Note that while the instance assignments to experts are soft, instance assignments to nodes within an expert are hard, meaning sets of instances assigned to different nodes are disjoint.

Probability estimate for y in the leaf node is usually performed by computing fractions of instances belonging to each class.

In our case, the modification is the same as the one presented by Eq. 5.

That way, estimates of probabilities P (y|x, θ

j ) needed by MOE are defined.

In Algorithm 1, function f it tree performs decision tree training using the above modifications.

We consider two ways to perform inference with respect to the obtained model.

First one which we call MOËT, is performed by maximizing P (y|x, θ) with respect to y where this probability is defined by Eq. 1.

The second way, which we call MOËT h , performs inference as arg max y P (y|x, θ arg max j gj (x,θg) ), meaning that we only rely on the most probable expert.

Algorithm 1 MOËT training.

for e ← 1 to N E do 4:

Adaptation of MOËT to imitation learning.

We integrate MOËT model into imitation learning approach of Viper by substituting training of DT with the MOËT training procedure.

Expressiveness.

Standard decision trees make their decisions by partitioning the feature space into regions which have borders perpendicular to coordinate axes.

To approximate borders that are not perpendicular to coordinate axes very deep trees are usually necessary.

MOËT h mitigates this shortcoming by exploiting hard softmax partitioning of the feature space using borders which are still hyperplanes, but need not be perpendicular to coordinate axes (see Section 3).

This improves the expressiveness of the model.

A MOËT h model is a combination of a linear model and several decision tree models.

For interpretability which is preserved in Lipton's sense of transparency, it is important that a single DT is used for each prediction (instead of weighted average).

Simultability of MOËT h consisting of DT and linear models is preserved because our models are small (2 ≤ depth ≤ 10) and we do not use high dimensional features (Lipton, 2016), so a person can easily simulate the model.

Similarly, decomposability is preserved because simple linear models without heavily engineered features and decision trees are decomposable (Lipton, 2016) and MOËT h is a simple combination of the two.

Finally, algorithmic transparency is achieved because MOËT training relies on DT training for the experts and linear model training for the gate, both of which are well understood.

However, the alternating refinement of initial feature space partitioning and experts makes the procedure more complicated, so our algorithmic transparency is partially achieved.

Importantly, we define a well-founded translation of MOËT h models to SMT formulas, which opens a new range of possibilities for interpreting and validating the model using automated reasoning tools.

SMT formulas provide a rich means of logical reasoning, where a user can ask the solver questions such as: "On which inputs do the two models differ?", or "What is the closest input to the given input on which model makes a different prediction?", or "Are the two models equivalent?", or "Are the two models equivalent in respect to the output class C?".

Answers to these and similar questions can help better understand and compare models in a rigorous way.

Also note that our symbolic reasoning of the gating function and decision trees allows us to construct SMT formulas that are readily handled by off-the-shelf tools, whereas direct SMT encodings of neural networks do not scale for any reasonably sized network because of the need for non-linear arithmetic reasoning.

We now compare MOËT and Viper on four OpenAI Gym environments: CartPole, Pong, Acrobot and Mountaincar.

For DRL agents, we use policy gradient model in CartPole, in other environments we use a DQN (Mnih et al., 2015) (training parameters provided in appendix).

The rewards obtained by the agents on CartPole, Pong, Acrobot and Mountaincar are 200.00, 21.00, −68.60 and −105.27, respectively (higher reward is better).

Rewards are averaged across 100 runs (250 in CartPole).

Comparison of MOËT, MOËT h , and Viper policies.

For CartPole, Acrobot, and Mountaincar environments, we train Viper DTs with maximum depths of {1, 2, 3, 4, 5}, while in the case of Pong we use maximum depths of {4, 6, 8, 10} as the problem is more complex and requires deeper trees.

For experts in MOËT policies we use the same maximum depths as in Viper (except for Pong for which we use depths 1 to 9) and we train the policies for 2 to 8 experts (in case of Pong we train for {2, 4, 8} experts).

We train all policies using 40 iterations of Viper algorithm, and choose the best performing policy in terms of rewards (and lower misprediction rate in case of equal rewards).

We use two criteria to compare policies: rewards and mispredictions (number of times the student performs an action different from what a teacher would do).

High reward indicates that the student learned more crucial parts of the teacher's policy, while a low misprediction rate indicates that in most cases student performs the same action as the teacher.

In order to measure mispredictions, we run the student for number of runs, and compare actions it took to the actions teacher would perform.

To ensure comparable depths for evaluating Viper and MOËT models while accounting for the different number of experts in MOËT, we introduce the notion of effective depth of a MOËT model as log 2 (E) + D, where E denotes the number of experts and D denotes the depth of each expert.

Table 1 compares the performance of Viper, MOËT and MOËT h .

The first column shows the depth of Viper decision trees and the corresponding effective depth for MOËT, rewards and mispredictions are shown in R and M columns resp.

We show results of the best performing MOËT configuration for a given effective depth chosen based on average results for rewards and mispredictions, where e.g. E3:D2 denotes 3 experts with DTs of depth 2.

All results shown are averaged across 10 runs 1 .

For CartPole, Viper, MOËT and MOËT h all achieve perfect reward (200) with depths of 2 and greater.

More interestingly, for depth 2 MOËT and MOËT h obtain significantly lower average misprediction rates of 0.84% and 0.91% respectively compared to 16.65% for Viper.

Even for larger depths, the misprediction rates for MOËT and MOËT h remain significantly lower.

For Pong, we observe that MOËT and MOËT h consistently outperform Viper for all depths in terms of rewards and mispredictions, whereas MOËT and MOËT h have similar performance.

For Acrobot, we similarly notice that both MOËT and MOËT h achieve consistently better rewards compared to Viper for all depths.

Moreover, the misprediction rates are also significantly lower for MOËT and MOËT h in majority of the cases.

Finally, for Mountaincar as well, we observe that MOËT and MOËT h both consistently outperform Viper with significantly higher rewards and lower misprediction rates.

Moreover, in both of these environments, we observe that both MOËT and MOËT h achieve comparable reward and misprediction rates.

Additional results are presented in appendix.

Analyzing the learned Policies.

We analyze the learned student policies (Viper and MOËT h ) by visualizing their state-action space, the differences between them, and differences with the teacher policy.

We use the Mountaincar environment for this analysis because of the ease of visualizing its 2-dimensional state space comprising of car position (p) and car velocity (v) features, and 3 allowed actions left, right, and neutral.

We visualize DRL, Viper and MOËT h policies in Figure 2 , showing the actions taken in different parts of the state space (additional visualizations are in appendix).

The state space is defined by feature bounds p ∈ [−1.2, 0.6] and v ∈ [−0.07, 0.07], which represent sets of allowed feature values in Mountaincar.

We sample the space uniformly with a resolution 200 × 200.

The actions left, neutral, and right are colored in green, yellow, and blue, respectively.

Recall that MOËT h can cover regions whose borders are hyperplanes of arbitrary orientation, while Viper, i.e. DT can only cover regions whose borders are perpendicular to coordinate axes.

This manifests in MOËT h policy containing slanted borders in yellow and green regions to capture more precisely the geometry of DRL policy, while the Viper policy only contains straight borders.

Furthermore, we visualize mispredictions for Viper and MOËT h policies.

While in previous section we calculated mispredictions by using student policy for playing the game, in this analysis we visualize mispredictions across the whole state space by sampling.

Note that in some states (critical states) it is more important to get the action right, while in other states choosing non-optimal action does not affect the overall score much.

Viper authors make use of this observation to weight states by their importance, and they use difference between Q values of optimal and non-optimal actions as a proxy for calculating how important (critical) state is.

Importance score is calculated as follows: I(s) = max a∈A Q(s, a)−min a∈A Q(s, a), where Q(s, a) denotes the Q value of action a in state s, and A is a set of all possible actions.

Using I function we weight mispredictions by their importance.

We create a vector i consisting of importance scores for sampled points, and normalize it to range [0, 1].

We also create a binary vector z which is 1 in the case of misprediction (student policy decision is different from DRL decision) and 0 otherwise.

We visualize m = z i (element-wise multiplication), where higher value indicates misprediction of higher importance and is denoted by a red color of higher intensity.

Such normalized mispredictions (m) for Viper and MOËT h policies are shown in Figure 2d and Figure 2e respectively.

We can observe that the MOËT h policy has fewer high intensity regions leading to fewer overall mispredictions.

To provide a quantitative difference between the mispredictions of two policies, we compute M = ( j m j / j i j ) · 100, which is measure in bounds [0, 100] such that its value is 0 in the case of no mispredictions, and 100 in the case of all mispredictions.

For the policies shown in Figure 2d and Figure 2e , we obtain M = 15.51 for Viper and M = 11.78 for MOËT h policies.

We also show differences in mispredictions between Viper and MOËT h (Figure 2f) Translating MOËT to SMT.

We now show the translation of MOËT policy to SMT constraints for verifying policy properties.

We present an example translation of MOËT policy on CartPole environment with the same property specification that was proposed for verifying Viper policies (Bastani et al., 2018) .

The goal in CartPole is to keep the pole upright, which can be encoded as a formula: where s i represents state after i steps, φ is the deviation of pole from the upright position.

In order to encode this formula it is necessary to encode the transition function f t (s, a) which models environment dynamics: given a state and action it returns the next state of the environment.

Also, it is necessary to encode the policy function π(s) that for a given state returns action to perform.

There are two issues with verifying ψ: (1) infinite time horizon; and (2) the nonlinear transition function f t .

To solve this problem, Bastani et al. (2018) use a finite time horizon T max = 10 and linear approximation of the dynamics and we make the same assumptions.

To encode π(s) we need to translate both the gating function and DT experts to logical formulas.

Since the gating function in MOËT h uses exponential function, it is difficult to encode the function directly in Z3 as SMT solvers do not have efficient decision procedures to solve non-linear arithmetic.

The direct encoding of exponentiation therefore leads to prohibitively complex Z3 formulas.

We exploit the following simplification of gating function that is sound when hard prediction is used:

First simplification is possible since the denominators for gatings of all experts are same, and second simplification is due to the monotonicity of the exponential function.

For encoding DTs we use the same encoding as in Viper.

To verify that ψ holds we need to show that ¬ψ is unsatisfiable.

We run the verification with our MOËT h policies and show that ¬ψ is indeed unsatisfiable.

To better understand the scalability of our verification procedure, we report on the verification times needed to verify policies for different number of experts and different expert depths in Figure 3 .

We observe that while MOËT h policies with 2 experts take from 2.6s to 8s for verification, the verification times for 8 experts can go up to as much as 319s.

This directly corresponds to the complexity of the logical formula obtained with an increase in the number of experts.

We introduced MOËT, a technique based on MOE with expert decision trees and presented a learning algorithm to train MOËT models.

We then used MOËT models for interpreting DRL agent policies, where different local DTs specialize on different regions of input space and are combined into a global policy using a gating function.

We showed that MOËT models lead to smaller, more faithful and performant representation of DRL agents compared to previous state-of-the-art approaches like Viper while still maintaining interpretability and verifiability.

Algorithm 2 Viper training (Bastani et al., 2018) 1: procedure VIPER (MDP e, TEACHER π t , Q-FUNCTION Q πt , ITERATIONS N ) 2:

Initialize dataset and student: D ← ∅, π s0 ← π t 3:

Sample trajectories and aggregate:

Sample dataset using Q values:

Train decision tree:

return Best policy π s ∈ {π s1 , ..., π s N }.

Viper algorithm is shown in Algorithm 2.

Here we present parameters we used to train DRL agents for different environments.

For CartPole, we use policy gradient model as used in Viper.

While we use the same model, we had to retrain it from scratch as the trained Viper agent was not available.

For Pong, we use a deep Q-network (DQN) network (Mnih et al., 2015) , and we use the same model as in Viper, which originates from OpenAI baselines (OpenAI Baselines).

For Acrobot and Mountaincar, we implement our own version of dueling DQN network following (Wang et al., 2015) .

We use 3 hidden layers with 15 neurons in each layer.

We set the learning rate to 0.001, batch size to 30, step size to 10000 and number of epochs to 80000.

We checkpoint a model every 5000 steps and pick the best performing one in terms of achieved reward.

In this section we provide a brief description of environments we used in our experiments.

We used four environments from OpenAI Gym: CartPole, Pong, Acrobot and Mountaincar.

This environment consists of a cart and a rigid pole hinged to the cart, based on the system presented by Barto et al. (Barto et al., 1983) .

At the beginning pole is upright, and the goal is to prevent it from falling over.

Cart is allowed to move horizontally within predefined bounds, and controller chooses to apply either left or right force to the cart.

State is defined with four variables: x (cart position),ẋ (cart velocity), θ (pole angle), andθ (pole angular velocity).

Game is terminated when the absolute value of pole angle exceeds 12

• , cart position is more than 2.4 units away from the center, or after 200 successful steps; whichever comes first.

In each step reward of +1 is given, and the game is considered solved when the average reward is over 195 in over 100 consecutive trials.

This is a classical Atari game of table tennis with two players.

Minimum possible score is −21 and maximum is 21.

This environment is analogous to a gymnast swinging on a horizontal bar, and consists of a two links and two joins, where the joint between the links is actuated.

The environment is based on the system presented by Sutton (Sutton, 1996) .

Initially both links are pointing downwards, and the goal is to swing the end-point (feet) above the bar for at least the length of one link.

The state consists of six variables, four variables consisting of sin and cos values of the joint angles, and two variables for angular velocities of the joints.

The action is either applying negative, neutral, or positive torque on the joint.

At each time step reward of −1 is received, and episode is terminated upon successful reaching the height, or after 200 steps, whichever comes first.

Acrobot is an unsolved environment in that there is no reward limit under which is considered solved, but the goal is to achieve high reward.

This environment consists of a car positioned between two hills, with a goal of reaching the hill in front of the car.

The environment is based on the system presented by Moore (Moore, 1990) .

Car can move in a one-dimensional track, but does not have enough power to reach the hill in one go, thus it needs to build momentum going back and forth to finally reach the hill.

Controller can choose left, right or neutral action to apply left, right or no force to the car.

State is defined by two variables, describing car position and car velocity.

In each step reward of −1 is received, and episode is terminated upon reaching the hill, or after 200 steps, whichever comes first.

The game is considered solved if average reward over 100 consecutive trials is no less than −110.

In this section we provide visualization of a gating function.

Figure 4 shows how gating function partitions the state space for which different experts specialize.

Gatings of MOËT h policy with 4 experts and depth 1 are shown.

E ADDITIONAL TABLES Table 2 shows results similar to Table 1 , but here in addition to averaging results across multiple trained models, it averages results across multiple MOËT configurations that have the same effective depth.

Table 3 shows the results of best performing DRL, MOËT and MOËT h models on the evaluation subjects.

<|TLDR|>

@highlight

Explainable reinforcement learning model using novel combination of mixture of experts with non-differentiable decision tree experts.