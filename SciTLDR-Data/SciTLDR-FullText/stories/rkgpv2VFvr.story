We study the benefit of sharing representations among tasks to enable the effective use of deep neural networks in Multi-Task Reinforcement Learning.

We leverage the assumption that learning from different tasks, sharing common properties, is helpful to generalize the knowledge of them resulting in a more effective feature extraction compared to learning a single task.

Intuitively, the resulting set of features offers performance benefits when used by Reinforcement Learning algorithms.

We prove this by providing theoretical guarantees that highlight the conditions for which is convenient to share representations among tasks, extending the well-known finite-time bounds of Approximate Value-Iteration to the multi-task setting.

In addition, we complement our analysis by proposing multi-task extensions of three Reinforcement Learning algorithms that we empirically evaluate on widely used Reinforcement Learning benchmarks showing significant improvements over the single-task counterparts in terms of sample efficiency and performance.

Multi-Task Learning (MTL) ambitiously aims to learn multiple tasks jointly instead of learning them separately, leveraging the assumption that the considered tasks have common properties which can be exploited by Machine Learning (ML) models to generalize the learning of each of them.

For instance, the features extracted in the hidden layers of a neural network trained on multiple tasks have the advantage of being a general representation of structures common to each other.

This translates into an effective way of learning multiple tasks at the same time, but it can also improve the learning of each individual task compared to learning them separately (Caruana, 1997) .

Furthermore, the learned representation can be used to perform Transfer Learning (TL), i.e. using it as a preliminary knowledge to learn a new similar task resulting in a more effective and faster learning than learning the new task from scratch (Baxter, 2000; Thrun & Pratt, 2012) .

The same benefits of extraction and exploitation of common features among the tasks achieved in MTL, can be obtained in Multi-Task Reinforcement Learning (MTRL) when training a single agent on multiple Reinforcement Learning (RL) problems with common structures (Taylor & Stone, 2009; Lazaric, 2012) .

In particular, in MTRL an agent can be trained on multiple tasks in the same domain, e.g. riding a bicycle or cycling while going towards a goal, or on different but similar domains, e.g. balancing a pendulum or balancing a double pendulum 1 .

Considering recent advances in Deep Reinforcement Learning (DRL) and the resulting increase in the complexity of experimental benchmarks, the use of Deep Learning (DL) models, e.g. deep neural networks, has become a popular and effective way to extract common features among tasks in MTRL algorithms (Rusu et al., 2015; Liu et al., 2016; Higgins et al., 2017) .

However, despite the high representational capacity of DL models, the extraction of good features remains challenging.

For instance, the performance of the learning process can degrade when unrelated tasks are used together (Caruana, 1997; Baxter, 2000) ; another detrimental issue may occur when the training of a single model is not balanced properly among multiple tasks (Hessel et al., 2018) .

Recent developments in MTRL achieve significant results in feature extraction by means of algorithms specifically developed to address these issues.

While some of these works rely on a single deep neural network to model the multi-task agent (Liu et al., 2016; Yang et al., 2017; Hessel et al., 2018; Wulfmeier et al., 2019) , others use multiple deep neural networks, e.g. one for each task and another for the multi-task agent (Rusu et al., 2015; Parisotto et al., 2015; Higgins et al., 2017; Teh et al., 2017) .

Intuitively, achieving good results in MTRL with a single deep neural network is more desirable than using many of them, since the training time is likely much less and the whole architecture is easier to implement.

In this paper we study the benefits of shared representations among tasks.

We theoretically motivate the intuitive effectiveness of our method, deriving theoretical guarantees that exploit the theoretical framework provided by Maurer et al. (2016) , in which the authors present upper bounds on the quality of learning in MTL when extracting features for multiple tasks in a single shared representation.

The significancy of this result is that the cost of learning the shared representation decreases with a factor O( 1 / ??? T ), where T is the number of tasks for many function approximator hypothesis classes.

The main contribution of this work is twofold.

Policy-Iteration (API) 2 (Farahmand, 2011) in the MTRL setting, and we extend the approximation error bounds in Maurer et al. (2016) to the case of multiple tasks with different dimensionalities.

Then, we show how to combine these results resulting in, to the best of our knowledge, the first proposed extension of the finite-time bounds of AVI/API to MTRL.

Despite being an extension of previous works, we derive these results to justify our approach showing how the error propagation in AVI/API can theoretically benefit from learning multiple tasks jointly.

2.

We leverage these results proposing a neural network architecture, for which these bounds hold with minor assumptions, that allow us to learn multiple tasks with a single regressor extracting a common representation.

We show an empirical evidence of the consequence of our bounds by means of a variant of Fitted Q-Iteration (FQI) (Ernst et al., 2005) , based on our shared network and for which our bounds apply, that we call Multi Fitted Q-Iteration (MFQI).

Then, we perform an empirical evaluation in challenging RL problems proposing multitask variants of the Deep Q-Network (DQN) (Mnih et al., 2015) and Deep Deterministic Policy Gradient (DDPG) (Lillicrap et al., 2015) algorithms.

These algorithms are practical implementations of the more general AVI/API framework, designed to solve complex problems.

In this case, the bounds apply to these algorithms only with some assumptions, e.g. stationary sampling distribution.

The outcome of the empirical analysis joins the theoretical results, showing significant performance improvements compared to the singletask version of the algorithms in various RL problems, including several MuJoCo (Todorov et al., 2012) domains.

Let B(X ) be the space of bounded measurable functions w.r.t.

the ??-algebra ?? X , and similarly B(X , L) be the same bounded by L < ???.

A Markov Decision Process (MDP) is defined as a 5-tuple M =< S, A, P, R, ?? >, where S is the state space, A is the action space, P : S ?? A ??? S is the transition distribution where P(s |s, a)

is the probability of reaching state s when performing action a in state s, R : S ?? A ?? S ??? R is the reward function, and ?? ??? (0, 1] is the discount factor.

A deterministic policy ?? maps, for each state, the action to perform: ?? : S ??? A. Given a policy ??, the value of an action a in a state s represents the expected discounted cumulative reward obtained by performing a in s and following ?? thereafter:

, where r i+1 is the reward obtained after the i-th transition.

The expected discounted cumulative reward is maximized by following the optimal policy ?? * which is the one that determines the optimal action values, i.e., the ones that satisfy the Bellman optimality equation (Bellman, 1954) :

>

where t ??? {1, . . . , T } and T is the number of MDPs.

For each MDP M (t) , a deterministic policy ?? t :

In this setting, the goal is to maximize the sum of the expected cumulative discounted reward of each task.

In our theoretical analysis of the MTRL problem, the complexity of representation plays a central role.

As done in Maurer et al. (2016), we consider the Gaussian complexity, a variant of the well-known Rademacher complexity, to measure the complexity of the representation.

Given a setX ??? X T n of n input samples for each task t ??? {1, . . .

, T }, and a class H composed of k ??? {1, . . . , K} functions, the Gaussian complexity of a random set H(X) = {(h k (X ti )) : h ??? H} ??? R KT n is defined as follows:

where ?? tki are independent standard normal variables.

We also need to define the following quantity, taken from Maurer (2016): let ?? be a vector of m random standard normal variables, and f ??? F :

Equation 2 can be viewed as a Gaussian average of Lipschitz quotients, and appears in the bounds provided in this work.

Finally, we define L(F) as the upper bound of the Lipschitz constant of all the functions f in the function class F.

The following theoretical study starts from the derivation of theoretical guarantees for MTRL in the AVI framework, extending the results of Farahmand (2011) in the MTRL scenario.

Then, to bound the approximation error term in the AVI bound, we extend the result described in Maurer (2006) to MTRL.

As we discuss, the resulting bounds described in this section clearly show the benefit of sharing representation in MTRL.

To the best of our knowledge, this is the first general result for MTRL; previous works have focused on finite MDPs (Brunskill & Li, 2013) or linear models (Lazaric & Restelli, 2011) .

problem:

where we use f = (f 1 , . . .

, f T ), w = (w 1 , . . . , w T ), and define the minimizers of Equation (3) as??, h, andf .

We assume that the loss function : R ?? R ??? [0, 1] is 1-Lipschitz in the first argument for every value of the second argument.

While this assumption may seem restrictive, the result obtained can be easily scaled to the general case.

To use the principal result of this section, for a generic loss function , it is possible to use (??) = (??) / max , where max is the maximum value of .

The expected loss over the tasks, given w, h and f is the task-averaged risk:

The minimum task-averaged risk, given the set of tasks ?? and the hypothesis classes W, H and F is ?? * avg , and the corresponding minimizers are w * , h * and f * .

We start by considering the bound for the AVI framework which applies for the single-task scenario.

Theorem 1. (Theorem 3.4 of Farahmand (2011)) Let K be a positive integer, and

?? , we have:

where

with E(?? 0 , . . . , ?? K???1 ; r) = K???1 k=0 ?? 2r k ?? k , the two coefficients c VI1,??,?? , c VI2,??,?? , the distributions ?? and ??, and the series ?? k are defined as in Farahmand (2011).

In the multi-task scenario, let the average approximation error across tasks be:

where Q t,k+1 =f t,k ????? k ????? t,k , and T * t is the optimal Bellman operator of task t. In the following, we extend the AVI bound of Theorem 1 to the multi-task scenario, by computing the average loss across tasks and pushing inside the average using Jensen's inequality.

Theorem 2.

Let K be a positive integer, and

we have:

Remarks Theorem 2 retains most of the properties of Theorem 3.4 of Farahmand (2011), except that the regression error in the bound is now task-averaged.

Interestingly, the second term of the sum in Equation (8) depends on the average maximum reward for each task.

In order to obtain this result we use an overly pessimistic bound on ?? and the concentrability coefficients, however this approximation is not too loose if the MDPs are sufficiently similar.

We bound the task-averaged approximation error ?? avg at each AVI iteration k involved in (8) following a derivation similar to the one proposed by Maurer et al. (2016) , obtaining: Theorem 3.

Let ??, W, H and F be defined as above and assume 0 ??? H and f (0) = 0, ???f ??? F. Then for ?? > 0 with probability at least 1 ??? ?? in the draw ofZ ??? T t=1 ?? n t we have that

Remarks The assumptions 0 ??? H and f (0) = 0 for all f ??? F are not essential for the proof and are only needed to simplify the result.

For reasonable function classes, the Gaussian complexity

If sup w w(X) and sup h,w h(w(X)) can be uniformly bounded, then they are O( ??? nT ).

For some function classes, the Gaussian average of Lipschitz quotients O(??) can be bounded independently from the number of samples.

Given these assumptions, the first and the fourth term of the right hand side of Equation (9), which represent respectively the cost of learning the meta-state space w and the task-specific f mappings, are both O( 1 / ??? n).

The second term represents the cost of learning the multi-task representation h and is O( 1 / ??? nT ), thus vanishing in the multi-task limit T ??? ???. The third term can be removed if ???h ??? H, ???p 0 ??? P : h(p) = 0; even when this assumption does not hold, this term can be ignored for many classes of interest, e.g. neural networks, as it can be arbitrarily small.

The last term to be bounded in (9) is the minimum average approximation error ?? * avg at each AVI iteration k. Recalling that the task-averaged approximation error is defined as in (7), applying Theorem 5.3 by Farahmand (2011) we obtain:

with C AE defined as in Farahmand (2011).

Final remarks The bound for MTRL is derived by composing the results in Theorems 2 and 3, and Lemma 4.

The results above highlight the advantage of learning a shared representation.

The bound in Theorem 2 shows that a small approximation error is critical to improve the convergence towards the optimal action-value function, and the bound in Theorem 3 shows that the cost of learning the shared representation at each AVI iteration is mitigated by using multiple tasks.

This is particularly beneficial when the feature representation is complex, e.g. deep neural networks.

As stated in the remarks of Equation (9), the benefit of MTRL is evinced by the second component of the bound, i.e. the cost of learning h, which vanishes with the increase of the number of tasks.

Obviously, adding more tasks require the shared representation to be large enough to include all of them, undesirably causing the term sup h,w h(w(X)) in the fourth component of the bound to increase.

This introduces a tradeoff between the number of features and number of tasks; however, for Figure 1 : (a) The architecture of the neural network we propose to learn T tasks simultaneously.

The w t block maps each input x t from task ?? t to a shared set of layers h which extracts a common representation of the tasks.

Eventually, the shared representation is specialized in block f t and the output y t of the network is computed.

Note that each block can be composed of arbitrarily many layers.

a reasonable number of tasks the number of features used in the single-task case is enough to handle them, as we show in some experiments in Section 5.

Notably, since the AVI/API framework provided by Farahmand (2011) provides an easy way to include the approximation error of a generic function approximator, it is easy to show the benefit in MTRL of the bound in Equation (9).

Despite being just multi-task extensions of previous works, our results are the first one to theoretically show the benefit of sharing representation in MTRL.

Moreover, they serve as a significant theoretical motivation, besides to the intuitive ones, of the practical algorithms that we describe in the following sections.

We want to empirically evaluate the benefit of our theoretical study in the problem of jointly learning T different tasks ?? t , introducing a neural network architecture for which our bounds hold.

Following our theoretical framework, the network we propose extracts representations w t from inputs x t for each task ?? t , mapping them to common features in a set of shared layers h, specializing the learning of each task in respective separated layers f t , and finally computing the output (Figure 1(a) ).

The idea behind this architecture is not new in the literature.

For instance, similar ideas have already been used in DQN variants to improve exploration on the same task via bootstrapping (Osband et al., 2016) and to perform MTRL (Liu et al., 2016) .

The intuitive and desirable property of this architecture is the exploitation of the regularization effect introduced by the shared representation of the jointly learned tasks.

Indeed, unlike learning a single task that may end up in overfitting, forcing the model to compute a shared representation of the tasks helps the regression process to extract more general features, with a consequent reduction in the variance of the learned function.

This intuitive justification for our approach, joins the theoretical benefit proven in Section 3.

Note that our architecture can be used in any MTRL problem involving a regression process; indeed, it can be easily used in value-based methods as a Q-function regressor, or in policy search as a policy regressor.

In both cases, the targets are learned for each task ?? t in its respective output block f t .

Remarkably, as we show in the experimental Section 5, it is straightforward to extend RL algorithms to their multi-task variants only through the use of the proposed network architecture, without major changes to the algorithms themselves.

To empirically evince the effect described by our bounds, we propose an extension of FQI (Ernst et al., 2005; Riedmiller, 2005) , that we call MFQI, for which our AVI bounds apply.

Then, to empirically evaluate our approach in challenging RL problems, we introduce multi-task variants of two well-known DRL algorithms: DQN (Mnih et al., 2015) and DDPG (Lillicrap et al., 2015) , which we call Multi Deep Q-Network (MDQN) and Multi Deep Deterministic Policy Gradient (MDDPG) respectively.

Note that for these methodologies, our AVI and API bounds hold only with Figure 2: Discounted cumulative reward averaged over 100 experiments of DQN and MDQN for each task and for transfer learning in the Acrobot problem.

An epoch consists of 1, 000 steps, after which the greedy policy is evaluated for 2, 000 steps.

The 95% confidence intervals are shown.

the simplifying assumption that the samples are i.i.d.; nevertheless they are useful to show the benefit of our method also in complex scenarios, e.g. MuJoCo (Todorov et al., 2012) .

We remark that in these experiments we are only interested in showing the benefit of learning multiple tasks with a shared representation w.r.t.

learning a single task; therefore, we only compare our methods with the single task counterparts, ignoring other works on MTRL in literature.

Experiments have been developed using the MushroomRL library (D'Eramo et al., 2020), and run on an NVIDIA R DGX Station TM and Intel R AI DevCloud.

Refer to Appendix B for all the details and our motivations about the experimental settings.

As a first empirical evaluation, we consider FQI, as an example of an AVI algorithm, to show the effect described by our theoretical AVI bounds in experiments.

We consider the Car-On-Hill problem as described in Ernst et al. (2005) , and select four different tasks from it changing the mass of the car and the value of the actions (details in Appendix B).

Then, we run separate instances of FQI with a single task network for each task respectively, and one of MFQI considering all the tasks simultaneously.

Figure 1(b) shows the L 1 -norm of the difference between Q * and Q ?? K averaged over all the tasks.

It is clear how MFQI is able to get much closer to the optimal Q-function, thus giving an empirical evidence of the AVI bounds in Theorem 2.

For completeness, we also show the advantage of MFQI w.r.t.

FQI in performance.

Then, in Figure 1 (c) we provide an empirical evidence of the benefit of increasing the number of tasks in MFQI in terms of both quality and stability.

As in Liu et al. (2016) , our MDQN uses separate replay memories for each task and the batch used in each training step is built picking the same number of samples from each replay memory.

Furthermore, a step of the algorithm consists of exactly one step in each task.

These are the only minor changes to the vanilla DQN algorithm we introduce, while all other aspects, such as the use of the target network, are not modified.

Thus, the time complexity of MDQN is considerably lower than vanilla DQN thanks to the learning of T tasks with a single model, but at the cost of a higher memory complexity for the collection of samples for each task.

We consider five problems with similar state spaces, sparse rewards and discrete actions: Cart-Pole, Acrobot, Mountain-Car, Car-On-Hill, and Inverted-Pendulum.

The implementation of the first three problems is the one provided by the OpenAI Gym library Brockman et al. (2016) , while Car-On-Hill is described in Ernst et al. (2005) and Inverted-Pendulum in Lagoudakis & Parr (2003) .

Figure 2(a) shows the performance of MDQN w.r.t.

to vanilla DQN that uses a single-task network structured as the multi-task one in the case with T = 1.

The first three plots from the left show good performance of MDQN, which is both higher and more stable than DQN.

In Car-On-Hill, MDQN is slightly slower than DQN to reach the best performance, but eventually manages to be more stable.

Finally, the Inverted-Pendulum experiment is clearly too easy to solve for both approaches, but it is still useful for the shared feature extraction in MDQN.

The described results provide important hints about the better quality of the features extracted by MDQN w.r.t.

DQN.

To further demonstrate this, we evaluate the performance of DQN on Acrobot, arguably the hardest of the five problems, using a single-task network with the shared parameters in h initialized with the weights of a multi-task network trained with MDQN on the other four problems.

Arbitrarily, the pre-trained weights can be adjusted during the learning of the new task or can be kept fixed and only the remaining randomly initialized parameters in w and f are trained.

From Figure 2(b) , the advantages of initializing the weights are clear.

In particular, we compare the performance of DQN without initialization w.r.t.

DQN with initialization in three settings: in Unfreeze-0 the initialized weights are adjusted, in NoUnfreeze they are kept fixed, and in Unfreeze-10 they are kept fixed until epoch 10 after which they start to be optimized.

Interestingly, keeping the shared weights fixed shows a significant performance improvement in the earliest epochs, but ceases to improve soon.

On the other hand, the adjustment of weights from the earliest epochs shows improvements only compared to the uninitialized network in the intermediate stages of learning.

The best results are achieved by starting to adjust the shared weights after epoch 10, which is approximately the point at which the improvement given by the fixed initialization starts to lessen.

In order to show how the flexibility of our approach easily allows to perform MTRL in policy search algorithms, we propose MDDPG as a multi-task variant of DDPG.

As an actor-critic method, DDPG requires an actor network and a critic network.

Intuitively, to obtain MDDPG both the actor and critic networks should be built following our proposed structure.

We perform separate experiments on two sets of MuJoCo Todorov et al. (2012) problems with similar continuous state and action spaces: the first set includes Inverted-Pendulum, Inverted-Double-Pendulum, and Inverted-Pendulum-Swingup as implemented in the pybullet library, whereas the second set includes Hopper-Stand, Walker-Walk, and Half-Cheetah-Run as implemented in the DeepMind Control SuiteTassa et al. (2018) .

Figure 3(a) shows a relevant improvement of MDDPG w.r.t.

DDPG in the pendulum tasks.

Indeed, while in Inverted-Pendulum, which is the easiest problem among the three, the performance of MDDPG is only slightly better than DDPG, the difference in the other two problems is significant.

The advantage of MDDPG is confirmed in Figure 3 (c) where it performs better than DDPG in Hopper and equally good in the other two tasks.

Again, we perform a TL evaluation of DDPG in the problems where it suffers the most, by initializing the shared weights of a single-task network with the ones of a multi-task network trained with MDDPG on the other problems.

Figures 3(b) and 3(d) show evident advantages of pre-training the shared weights and a significant difference between keeping them fixed or not.

Our work is inspired from both theoretical and empirical studies in MTL and MTRL literature.

In particular, the theoretical analysis we provide follows previous results about the theoretical properties of multi-task algorithms.

For instance, Cavallanti et al. (2010) and Maurer (2006) prove the theoretical advantages of MTL based on linear approximation.

More in detail, Maurer (2006) derives bounds on MTL when a linear approximator is used to extract a shared representation among tasks.

Then, Maurer et al. (2016), which we considered in this work, describes similar results that extend to the use of non-linear approximators.

Similar studies have been conducted in the context of MTRL.

Among the others, Lazaric & Restelli (2011) and Brunskill & Li (2013) give theoretical proofs of the advantage of learning from multiple MDPs and introduces new algorithms to empirically support their claims, as done in this work.

Generally, contributions in MTRL assume that properties of different tasks, e.g. dynamics and reward function, are generated from a common generative model.

About this, interesting analyses consider Bayesian approaches; for instance Wilson et al. (2007) assumes that the tasks are generated from a hierarchical Bayesian model, and likewise Lazaric & Ghavamzadeh (2010) considers the case when the value functions are generated from a common prior distribution.

Similar considerations, which however does not use a Bayesian approach, are implicitly made in Taylor et al. (2007), Lazaric et al. (2008) , and also in this work.

In recent years, the advantages of MTRL have been empirically evinced also in DRL, especially exploiting the powerful representational capacity of deep neural networks.

For instance, Parisotto et al. (2015) and Rusu et al. (2015) propose to derive a multi-task policy from the policies learned by DQN experts trained separately on different tasks.

Rusu et al. (2015) compares to a therein introduced variant of DQN, which is very similar to our MDQN and the one in Liu et al. (2016) , showing how their method overcomes it in the Atari benchmark Bellemare et al. (2013) .

Further developments, extend the analysis to policy search (Yang et al., 2017; Teh et al., 2017) , and to multi-goal RL (Schaul et al., 2015; Andrychowicz et al., 2017) .

Finally, Hessel et al. (2018) addresses the problem of balancing the learning of multiple tasks with a single deep neural network proposing a method that uniformly adapts the impact of each task on the training updates of the agent.

We have theoretically proved the advantage in RL of using a shared representation to learn multiple tasks w.r.t.

learning a single task.

We have derived our results extending the AVI/API bounds (Farahmand, 2011) to MTRL, leveraging the upper bounds on the approximation error in MTL provided in Maurer et al. (2016) .

The results of this analysis show that the error propagation during the AVI/API iterations is reduced according to the number of tasks.

Then, we proposed a practical way of exploiting this theoretical benefit which consists in an effective way of extracting shared representations of multiple tasks by means of deep neural networks.

To empirically show the advantages of our method, we carried out experiments on challenging RL problems with the introduction of multi-task extensions of FQI, DQN, and DDPG based on the neural network structure we proposed.

As desired, the favorable empirical results confirm the theoretical benefit we described.

A PROOFS

Proof of Theorem 2.

We compute the average expected loss across tasks: (11) with ?? = max

VI,??,?? (K; t, r), and R max,avg = 1 /T T t=1 R max,t .

Considering the term 1 /T

Using Jensen's inequality:

So, now we can write (11) as

avg (?? avg,0 , . . . , ?? avg,K???1 ; r)

with ?? avg,k = 1 /T T t=1 ?? t,k and E avg (?? avg,0 , . . . , ?? avg,K???1 ; r) =

Proof of Lemma 4.

Let us start from the definition of optimal task-averaged risk:

where Q * t,k , with t ??? [1, T ], are the minimizers of ?? avg,k .

Consider the task?? such that??

we can write the following inequality:

By the application of Theorem 5.3 by Farahmand (2011) to the right hand side, and defining

Q?? k ,i ?? , we obtain:

Squaring both sides yields the result:

We start by considering the bound for the API framework:

?? , we have:

where

with E(?? 0 , . . . , ?? K???1 ; r) = K???1 k=0 ?? 2r k ?? k , the three coefficients c PI1,??,?? , c PI2,??,?? , c PI3,??,?? , the distributions ?? and ??, and the series ?? k are defined as in Farahmand (2011).

From Theorem 5, by computing the average loss across tasks and pushing inside the average using Jensen's inequality, we derive the API bounds averaged on multiple tasks.

?? , we have:

avg (?? avg,0 , . . . , ?? avg,K???1 ; r)

with

Proof of Theorem 6.

The proof is very similar to the one for AVI.

We compute the average expected loss across tasks:

with ?? avg,k = 1 /T T t=1 ?? t,k and E avg (?? avg,0 , . . . , ?? avg,K???1 ; r) = K???1 k=0 ?? 2r k ?? avg,k .

Proof of Theorem 3.

Let w * 1 , . . .

, w * T , h * and f * 1 , . . . , f * T be the minimizers of ?? * avg , then:

We proceed to bound the three components individually:

??? C can be bounded using Hoeffding's inequality, with probability 1 ??? ?? /2 by ln( 2 /??) /(2nT ), as it contains only nT random variables bounded in the interval [0, 1];

Firstly, to bound L(F ), let y, y ??? R KT n , where y = (y ti ) with y ti ??? R K and y = (y ti ) with y ti ??? R K .

We can write the following:

whence L(F ) ??? L(F).

Then, we bound:

Then, since it is possible to bound the Euclidean diameter using the norm of the supremum value in the set, we bound D(S ) ??? 2 sup h,w h(w(X)) and D(W (X)) ??? 2 sup w???W T w(X) .

E sup g???F ??, g(y) ??? g(y ) = E sup f ???F T ti ?? ti (f t (y ti ) ??? f t (y ti ))

whence O(F ) ??? ???

To minimize the last term, it is possible to choose y 0 = 0, as f (0) = 0, ???f ??? F, resulting in min y???Y G(F(y)) = G(F(0)) = 0.

Then, substituting in (21), and recalling that G(S) ??? G(S ):

@highlight

A study on the benefit of sharing representation in Multi-Task Reinforcement Learning.