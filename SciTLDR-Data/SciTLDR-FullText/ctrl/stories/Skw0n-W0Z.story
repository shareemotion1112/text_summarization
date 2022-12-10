Model-free reinforcement learning (RL) has been proven to be a powerful, general tool for learning complex behaviors.

However, its sample efficiency is often impractically large for solving challenging real-world problems, even for off-policy algorithms such as Q-learning.

A limiting factor in classic model-free RL is that the learning signal consists only of scalar rewards, ignoring much of the rich information contained in state transition tuples.

Model-based RL uses this information, by training a predictive model, but often does not achieve the same asymptotic performance as model-free RL due to model bias.

We introduce temporal difference models (TDMs), a family of goal-conditioned value functions that can be trained with model-free learning and used for model-based control.

TDMs combine the benefits of model-free and model-based RL: they leverage the rich information in state transitions to learn very efficiently, while still attaining asymptotic performance that exceeds that of direct model-based RL methods.

Our experimental results show that, on a range of continuous control tasks, TDMs provide a substantial improvement in efficiency compared to state-of-the-art model-based and model-free methods.

Reinforcement learning (RL) algorithms provide a formalism for autonomous learning of complex behaviors.

When combined with rich function approximators such as deep neural networks, RL can provide impressive results on tasks ranging from playing games BID23 BID29 , to flying and driving BID36 , to controlling robotic arms BID14 .

However, these deep RL algorithms often require a large amount of experience to arrive at an effective solution, which can severely limit their application to real-world problems where this experience might need to be gathered directly on a real physical system.

Part of the reason for this is that direct, model-free RL learns only from the reward: experience that receives no reward provides minimal supervision to the learner.

In contrast, model-based RL algorithms obtain a large amount of supervision from every sample, since they can use each sample to better learn how to predict the system dynamics -that is, to learn the "physics" of the problem.

Once the dynamics are learned, near-optimal behavior can in principle be obtained by planning through these dynamics.

Model-based algorithms tend to be substantially more efficient BID9 BID24 , but often at the cost of larger asymptotic bias: when the dynamics cannot be learned perfectly, as is the case for most complex problems, the final policy can be highly suboptimal.

Therefore, conventional wisdom holds that model-free methods are less efficient but achieve the best asymptotic performance, while model-based methods are more efficient but do not produce policies that are as optimal.

Can we devise methods that retain the efficiency of model-based learning while still achieving the asymptotic performance of model-free learning?

This is the question that we study in this paper.

The search for methods that combine the best of model-based and model-free learning has been ongoing for decades, with techniques such as synthetic experience generation BID31 , partial modelbased backpropagation BID25 , and layering model-free learning on the residuals of model-based estimation BID6 ) being a few examples.

However, a direct connection between model-free and model-based RL has remained elusive.

By effectively bridging the gap between model-free and model-based RL, we should be able to smoothly transition from learning models to learning policies, obtaining rich supervision from every sample to quickly gain a moderate level of proficiency, while still converging to an unbiased solution.

To arrive at a method that combines the strengths of model-free and model-based RL, we study a variant of goal-conditioned value functions BID32 BID28 BID0 .

Goal-conditioned value functions learn to predict the value function for every possible goal state.

That is, they answer the following question: what is the expected reward for reaching a particular state, given that the agent is attempting (as optimally as possible) to reach it?

The particular choice of reward function determines what such a method actually does, but rewards based on distances to a goal hint at a connection to model-based learning: if we can predict how easy it is to reach any state from any current state, we must have some kind of understanding of the underlying "physics." In this work, we show that we can develop a method for learning variable-horizon goalconditioned value functions where, for a specific choice of reward and horizon, the value function corresponds directly to a model, while for larger horizons, it more closely resembles model-free approaches.

Extension toward more model-free learning is thus achieved by acquiring "multi-step models" that can be used to plan over progressively coarser temporal resolutions, eventually arriving at a fully model-free formulation.

The principle contribution of our work is a new RL algorithm that makes use of this connection between model-based and model-free learning to learn a specific type of goal-conditioned value function, which we call a temporal difference model (TDM).

This value function can be learned very efficiently, with sample complexities that are competitive with model-based RL, and can then be used with an MPC-like method to accomplish desired tasks.

Our empirical experiments demonstrate that this method achieves substantially better sample complexity than fully model-free learning on a range of challenging continuous control tasks, while outperforming purely model-based methods in terms of final performance.

Furthermore, the connection that our method elucidates between model-based and model-free learning may lead to a range of interesting future methods.

In this section, we introduce the reinforcement learning (RL) formalism, temporal difference Qlearning methods, model-based RL methods, and goal-conditioned value functions.

We will build on these components to develop temporal difference models (TDMs) in the next section.

RL deals with decision making problems that consist of a state space S, action space A, transition dynamics P (s | s, a), and an initial state distribution p 0 .

The goal of the learner is encapsulated by a reward function r(s, a, s ).

Typically, long or infinite horizon tasks also employ a discount factor γ, and the standard objective is to find a policy π(a | s) that maximizes the expected discounted sum of rewards, E π [ t γ t r(s t , a t , s t+1 )], where s 0 ∼ p 0 , a t ∼ π(a t |s t ), and s t+1 ∼ P (s | s, a).Q-functions.

We will focus on RL algorithms that learn a Q-function.

The Q-function represents the expected total (discounted) reward that can be obtained by the optimal policy after taking action a t in state s t , and can be defined recursively as following: DISPLAYFORM0 The optimal policy can then recovered according to π(a t |s t ) = δ(a t = arg max a Q(s t , a)).

Qlearning algorithms BID35 BID27 learn the Q-function via an offpolicy stochastic gradient descent algorithm, estimating the expectation in the above equation with samples collected from the environment and computing its gradient.

Q-learning methods can use transition tuples (s t , a t , s t+1 , r t ) collected from any exploration policy, which generally makes them more efficient than direct policy search, though still less efficient than purely model-based methods.

Model-based RL and optimal control.

Model-based RL takes a different approach to maximize the expected reward.

In model-based RL, the aim is to train a model of the form f (s t , a t ) to predict the next state s t+1 .

Once trained, this model can be used to choose actions, either by backpropagating reward gradients into a policy, or planning directly through the model.

In the latter case, a particularly effective method for employing a learned model is model-predictive control (MPC), where a new action plan is generated at each time step, and the first action of that plan is executed, before replanning begins from scratch.

MPC can be formalized as the following optimization problem: DISPLAYFORM1 We can also write the dynamics constraint in the above equation in terms of an implicit dynamics, according to DISPLAYFORM2 where C(s i , a i , s i+1 ) = 0 if and only if s i+1 = f (s i , a i ).

This implicit version will be important in understanding the connection between model-based and model-free RL.Goal-conditioned value functions.

Q-functions trained for a specific reward are specific to the corresponding task, and learning a new task requires optimizing an entirely new Q-function.

Goalconditioned value functions address this limitation by conditioning the Q-value on some task description vector s g ∈ G in a goal space G. This goal vector induces a parameterized reward r(s t , a t , s t+1 , s g ), which in turn gives rise to parameterized Q-functions of the form Q(s, a, s g ).A number of goal-conditioned value function methods have been proposed in the literature, such as universal value functions BID28 and Horde (Sutton et al., 2011) .

When the goal corresponds to an entire state, such goal-conditioned value functions usually predict how well an agent can reach a particular state, when it is trying to reach it.

The knowledge contained in such a value function is intriguingly close to a model: knowing how well you can reach any state is closely related to understanding the physics of the environment.

With Q-learning, these value functions can be learned for any goal s g using the same off-policy (s t , a t , s t+1 ) tuples.

Relabeling previously visited states with the reward for any goal leads to a natural data augmentation strategy, since each tuple can be replicated many times for many different goals without additional data collection.

BID0 used this property to produce an effective curriculum for solving multi-goal task with delayed rewards.

As we discuss below, relabeling past experience with different goals enables goal-conditioned value functions to learn much more quickly from the same amount of data.

In this section, we introduce a type of goal-conditioned value functions called temporal difference models (TDMs) that provide a direct connection to model-based RL.

We will first motivate this connection by relating the model-based MPC optimizations in Equations (2) and (3) to goal-conditioned value functions, and then present our temporal difference model derivation, which extends this connection from a purely model-based setting into one that becomes increasingly model-free.

Let us consider the choice of reward function for the goal conditioned value function.

Although a variety of options have been explored in the literature BID32 BID28 BID0 , a particularly intriguing connection to model-based RL emerges if we set G = S, such that g ∈ G corresponds to a goal state s g ∈ S, and we consider distance-based reward functions r d of the following form: DISPLAYFORM0 ) at convergence of Q-learning, which means that Q(s t , a t , s g ) = 0 implies that s t+1 = s g .

Plug this Q-function into the model-based planning optimization in Equation (3), denoting the task control reward as r c , such that the solution to DISPLAYFORM1 yields a model-based plan.

We have now derived a precise connection between model-free and model-based RL, in that model-free learning of goal-conditioned value functions can be used to directly produce an implicit model that can be used with MPC-based planning.

However, this connection by itself is not very useful: the resulting implicit model is fully model-based, and does not provide any kind of long-horizon capability.

In the next section, we show how to extend this connection into the long-horizon setting by introducing the temporal difference model (TDM).

If we consider the case where γ > 0, the optimization in Equation (4) no longer corresponds to any optimal control method.

In fact, when γ = 0, Q-values have well-defined units: units of distance between states.

For γ > 0, no such interpretation is possible.

The key insight in temporal difference models is to introduce a different mechanism for aggregating long-horizon rewards.

Instead of evaluating Q-values as discounted sums of rewards, we introduce an additional input τ , which represents the planning horizon, and define the Q-learning recursion as DISPLAYFORM0 The Q-function uses a reward of −D(s t+1 , s g ) when τ = 0 (at which point the episode terminates), and decrements τ by one at every other step.

Since this is still a well-defined Q-learning recursion, it can be optimized with off-policy data and, just as with goal-conditioned value functions, we can resample new goals s g and new horizons τ for each tuple (s t , a t , s t+1 ), even ones that were not actually used when the data was collected.

In this way, the TDM can be trained very efficiently, since every tuple provides supervision for every possible goal and every possible horizon.

The intuitive interpretation of the TDM is that it tells us how close the agent will get to a given goal state s g after τ time steps, when it is attempting to reach that state in τ steps.

Alternatively, TDMs can be interpreted as Q-values in a finite-horizon MDP, where the horizon is determined by τ .

For the case where τ = 0, TDMs effectively learn a model, allowing TDMs to be incorporated into a variety of planning and optimal control schemes at test time as in Equation (4).

Thus, we can view TDM learning as an interpolation between model-based and model-free learning, where τ = 0 corresponds to the single-step prediction made in model-based learning and τ > 0 corresponds to the long-term prediction made by typical Q-functions.

While the correspondence to models is not the same for τ > 0, if we only care about the reward at every K step, then we can recover a correspondence by replace Equation (4) with DISPLAYFORM1 where we only optimize over every K th state and action.

As the TDM becomes effective for longer horizons, we can increase K until K = T , and plan over only a single effective time step: DISPLAYFORM2 This formulation does result in some loss of generality, since we no longer optimize the reward at the intermediate steps.

This limits the multi-step formulation to terminal reward problems, but does allow us to accommodate arbitrary reward functions on the terminal state s t+T , which still describes a broad range of practically relevant tasks.

In the next section, we describe how TDMs can be implemented and used in practice for continuous state and action spaces.

The TDM can be trained with any off-policy Q-learning algorithm, such as DQN BID23 , DDPG (Lillicrap et al., 2015) , NAF BID13 , and SDQN BID21 .

During off-policy Q-learning, TDMs can benefit from arbitrary relabeling of the goal states g and the horizon τ , given the same (s t , a t , s t+1 ) tuples from the behavioral policy as done in BID0 .

This relabeling enables simultaneous, data-efficient learning of short-horizon and long-horizon behaviors for arbitrary goal states, unlike previously proposed goal-conditioned value functions that only learn for a single time scale, typically determined by a discount factor BID28 BID0 .

In this section, we describe the design decisions needed to make practical a TDM algorithm.

Q-learning typically optimizes scalar rewards, but TDMs enable us to increase the amount of supervision available to the Q-function by using a vector-valued reward.

Specifically, if the distance D(s, s g ) factors additively over the dimensions, we can train a vector-valued Q-function that predicts per-dimension distance, with the reward function for dimension j given by −D j (s j , s g,j ).

We use the 1 norm in our implementation, which corresponds to absolute value reward −|s j − s g,j |.The resulting vector-valued Q-function can learn distances along each dimension separately, providing it with more supervision from each training point.

Empirically, we found that this modifications provides a substantial boost in sample efficiency.

We can optionally make an improvement to TDMs if we know that the task reward r c depends only on some subset of the state or, more generally, state features.

In that case, we can train the TDM to predict distances along only those dimensions or features that are used by r c , which in practice can substantially simplify the corresponding prediction problem.

In our experiments, we illustrate this property by training TDMs for pushing tasks that predict distances from an end-effector and pushed object, without accounting for internal joints of the arm, and similarly for various locomotion tasks.

While the TDM optimal control formulation Equation FORMULA7 drastically reduces the number of states and actions to be optimized for long-term planning, it requires solving a constrained optimization problem, which is more computationally expensive than unconstrained problems.

We can remove the need for a constrained optimization through a specific architectural decision in the design of the function approximator for Q(s, a, s g , τ ).

We define the Q-function as Q(s, a, s g , τ ) = − f (s, a, s g , τ ) − s g , where f (s, a, s g , τ ) outputs a state vector.

By training the TDM with a standard Q-learning method, f (s, a, s g , τ ) is trained to explicitly predict the state that will be reached by a policy attempting to reach s g in τ steps.

This model can then be used to choose the action with fully explicit MPC as below, which also allows straightforward derivation of a multi-step version as in Equation FORMULA6 .

DISPLAYFORM0 In the case where the task is to reach a goal state s g , a simpler approach to extract a policy is to use the TDM directly: DISPLAYFORM1 In our experiments, we use Equations FORMULA8 and FORMULA9 to extract a policy.

The algorithm is summarized as Algorithm 1.

A crucial difference from prior goal-conditioned value function methods BID28 BID0 ) is that our algorithm can be used to act according to an arbitrary terminal reward function r c , both during exploration and at test time.

Like other off-policy algorithms BID23 , it consists of exploration and Q-function fitting.

Noise is injected for exploration, and Q-function fitting uses standard Qlearning techniques, with target networks Q and experience replay BID23 .

If we view the Q-function fitting as model fitting, the algorithm also resembles iterative model-based RL, which alternates between collecting data using the learned dynamics model for planning BID8 ) and fitting the model.

Since we focus on continuous tasks, we use DDPG , though any Q-learning method could be used.

The computation cost of the algorithm is mostly determined by the number of updates to fit the Q-function per transition, I. In general, TDMs can benefit from substantially larger I than classic

Require: Task reward function rc(s, a), parameterized TDM Qw(s, a, sg, τ ), replay buffer B 1: for n = 0, ..., N − 1 episodes do 2: s0 ∼ p(s0) 3: for t = 0, ..., T − 1 time steps do 4: a * t = MPC(rc, st, Qw, T − t) // Eq. 6, Eq. 7, Eq. 8, or Eq. 9 5: at = AddNoise(a * t ) // Noisy exploration 6: st+1 ∼ p(st, at), and store {st, at, st+1} in the replay buffer B // Step environment 7:for i = 0, I − 1 iterations do 8:Sample M transitions {sm, am, s m } from the replay B.

Relabel time horizons and goal states τm, sg,m // Section A.1 10: DISPLAYFORM0 Minimize(w, L(w)) // Optimize 13: end for 14:end for 15: end for model-free methods such as DDPG due to relabeling increasing the amount of supervision signals.

In real-world applications such as robotics where we care most of the sample efficiency BID14 , the learning is often bottlenecked by the data collection rather than the computation, and therefore large I values are usually not a significant problem and can continuously benefit from the acceleration in computation.

Combining model-based and model-free reinforcement learning techniques is a well-studied problem, though no single solution has demonstrated all of the benefits of model-based and model-free learning.

Some methods first learn a model and use this model to simulate experience BID31 BID13 or compute better gradients for model-free updates BID25 .

Other methods use model-free algorithms to correct for the local errors made by the model BID6 BID1 .

While these prior methods focused on combining different model-based and model-free RL techniques, our method proposes an equivalence between these two branches of RL through a specific generalization of goal-conditioned value function.

As a result, our approach achieves much better sample efficiency in practice on a variety of challenging reinforcement learning tasks than model-free alternatives, while exceeding the performance of purely model-based approaches.

We are not the first to study the connection between model-free and model-based methods, with Boyan (1999) and BID26 being two notable examples.

BID3 shows that one can extract a model from a value function when using a tabular representation of the transition function.

BID26 shows that, for linear function approximators, the model-free and model-based RL approaches produce the same value function at convergence.

Our contribution differs substantially from these: we are not aiming to show that model-free RL performs similarly to model-based RL at convergence, but rather how we can achieve sample complexity comparable to model-based RL while retaining the favorable asymptotic performance of model-free RL in complex tasks with nonlinear function approximation.

BID10 .

Critically, unlike the works on contextual policies BID5 BID7 BID18 which require onpolicy trajectories with each new goal, the value function approaches such as Horde BID32 and UVF BID28 can reuse off-policy data to learn rich contextual value functions using the same data.

TDMs condition on a policy trying to reach a goal and must predict τ steps into the future.

This type of prediction is similar to the prediction made by prior work on multi-step models BID22 BID34 : predict the state after τ actions.

An important difference is that multi-step models do not condition on a policy reaching a goal, and so they require optimizing over a sequence of actions, making the input space grow linearly with the planning horizon.

A particularly related UVF extension is hindsight experience replay (HER) BID0 .

Both HER and our method retroactively relabel past experience with goal states that are different from the goal aimed for during data collection.

However, unlike our method, the standard UVF in HER uses a single temporal scale when learning, and does not explicitly provide for a connection between model-based and model-free learning.

The practical result of these differences is that our approach empirically achieves substantially better sample complexity than HER on a wide range of complex continuous control tasks, while the theoretical connection between modelbased and model-free learning suggests a much more flexible use of the learned Q-function inside a planning or optimal control framework.

Lastly, our motivation is shared by other lines of work besides goal-conditioned value functions that aim to enhance supervision signals for model-free RL BID16 BID2 .

Predictions ) augment classic RL with multi-step reward predictions, while UNREAL BID16 ) also augments it with pixel control as a secondary reward objective.

These are substantially different methods from our work, but share the motivation to achieve efficient RL by increasing the amount of learning signals from finite data.

Our experiments examine how the sample efficiency and performance of TDMs compare to both model-based and model-free RL algorithms.

We expect to have the efficiency of model-based RL but with less model bias.

We also aim to study the importance of several key design decisions in TDMs, and evaluate the algorithm on a real-world robotic platform.

For the model-free comparison, we compare to DDPG , which typically achieves the best sample efficiency on benchmark tasks BID11 ; HER, which uses goal-conditioned value functions BID0 ; and DDPG with the same sparse rewards of HER.

For the modelbased comparison, we compare to the model-based component in BID24 , a recent work that reports highly efficient learning with neural network dynamics models.

Details of the baseline implementations are in the Appendix.

We perform the comparison on five simulated tasks: (1) a 7 DoF arm reaching various random end-effector targets, (2) an arm pushing a puck to a target location, (3) a planar cheetah attempting to reach a goal velocity (either forward or backward), (4) a quadrupedal ant attempting to reach a goal position, and (5) an ant attempting to reach a goal position and velocity.

The tasks are shown in Figure 1 and terminate when either the goal is reached or the time horizon is reached.

The pushing task requires long-horizon reasoning to reach and push the puck.

The cheetah and ant tasks require handling many contact discontinuities which is challenging for model-based methods, with the ant environment having particularly difficult dynamics given the larger state and action space.

The ant position and velocity task presents a scenario where reward shaping as in traditional RL methods may not lead to optimal behavior, since one cannot maintain both a desired position and velocity.

However, such a task can be very valuable in realistic settings.

For example, if we want the ant to jump, we might instruct it to achieve a particular velocity at a particular location.

We also tested TDMs on a real-world robot arm reaching end-effector positions, to study its applicability to real-world tasks.

For the simulated and real-world 7-DoF arm, our TDM is trained on all state components.

For the pushing task, our TDM is trained on the hand and puck XY-position.

For the half cheetah task, our TDM is trained on the velocity of the cheetah.

For the ant tasks, our TDM is trained on either the position or the position and velocity for the respective task.

Full details are in the Appendix.

The results are shown in Figure 2 .

When compared to the model-free baselines, the pure modelbased method learns learns much faster on all the tasks.

However, on the harder cheetah and ant tasks, its final performance is worse due to model bias.

TDMs learn as quickly or faster than the model-based method, but also always learn policies that are as good as if not better than the modelfree policies.

Furthermore, TDMs requires fewer samples than the model-free baselines on ant tasks and drastically fewer samples on the other tasks.

We also see that using HER does not lead to , model-based, and goal-conditioned value functions (HER -Dense) on various tasks.

All plots show the final distance to the goal versus 1000 environment steps (not rollouts).

The bold line shows the mean across 3 random seeds, and the shaded region show one standard deviation.

Our method, which uses model-free learning, is generally more sample-efficient than model-free alternatives including DDPG and HER and improves upon the best model-based performance.an improvement over DDPG.

While we were initially surprised, we realized that a selling point of HER is that it can solve sparse tasks that would otherwise be unsolvable.

In this paper, we were interested in improving the sample efficiency and not the feasibility of model-free reinforcement learning algorithms, and so we focused on tasks that DDPG could already solve.

In these sorts of tasks, the advantage of HER over DDPG with a dense reward is not expected.

To evaluate HER as a method to solve sparse tasks, we included the DDPG-Sparse baseline and we see that HER significantly outperforms it as expected.

In summary, TDMs converge as fast or faster than modelbased learning (which learns faster than the model-free baselines), while achieving final performance that is as good or better that the model-free methods on all tasks.

Lastly, we ran the algorithm on a 7-DoF Sawyer robotic arm to learn a real-world analogue of the reaching task.

Figure 2f shows that the algorithm outperforms and learns with fewer samples than DDPG, our model-free baseline.

These results show that TDMs can scale to real-world tasks.

In this section, we discuss two key design choices for TDMs that provide substantially improved performance.

First, FIG2 examines the tradeoffs between the vectorized and scalar rewards.

The results show that the vectorized formulation learns substantially faster than the naïve scalar variant.

Second, FIG2 compares the learning speed for different horizon values τ max .

Performance degrades when the horizon is too low, and learning becomes slower when the horizon is too high.

In this paper, we derive a connection between model-based and model-free reinforcement learning, and present a novel RL algorithm that exploits this connection to greatly improve on the sample efficiency of state-of-the-art model-free deep RL algorithms.

Our temporal difference models can be viewed both as goal-conditioned value functions and implicit dynamics models, which enables them to be trained efficiently on off-policy data while still minimizing the effects of model bias.

As a result, they achieve asymptotic performance that compares favorably with model-free algorithms, but with a sample complexity that is comparable to purely model-based methods.

While the experiments focus primarily on the new RL algorithm, the relationship between modelbased and model-free RL explored in this paper provides a number of avenues for future work.

We demonstrated the use of TDMs with a very basic planning approach, but further exploring how TDMs can be incorporated into powerful constrained optimization methods for model-predictive control or trajectory optimization is an exciting avenue for future work.

Another direction for future is to further explore how TDMs can be applied to complex state representations, such as images, where simple distance metrics may no longer be effective.

Although direct application of TDMs to these domains is not straightforward, a number of works have studied how to construct metric embeddings of images that could in principle provide viable distance functions.

We also note that while the presentation of TDMs have been in the context of deterministic environments, the extension to stochastic environments is straightforward: TDMs would learn to predict the expected distance between the future state and a goal state.

Finally, the promise of enabling sample-efficient learning with the performance of model-free RL and the efficiency of model-based RL is to enable widespread RL application on real-world systems.

Many applications in robotics, autonomous driving and flight, and other control domains could be explored in future work.

The maximum distance was set to 5 rather than 6 for this experiment, so the numbers should be lower than the ones reported in the paper.

A.1 GOAL STATE AND τ SAMPLING STRATEGY While Q-learning is valid for any value of s g and τ for each transition tuple (s t , a t , s t+1 ), the way in which these values are sampled during training can affect learning efficiency.

Some potential strategies for sampling s g are: (1) uniformly sample future states along the actual trajectory in the buffer (i.e., for s t , choose s g = s t+k for a random k > 0) as in BID0 ; (2) uniformly sample goal states from the replay buffer; (3) uniformly sample goals from a uniform range of valid states.

We found that the first strategy performed slightly better than the others, though not by much.

In our experiments, we use the first strategy.

The horizon τ is sampled uniformly at random between 0 and the maximum horizon τ max .

In all our experiments, we used DDPG as the base off-policy model-free RL algorithm for learning the TDMs Q(s, a, g, s τ ).

Experience replay BID23 has size of 1 million transitions, and the soft target networks are used with a polyak averaging coefficient of 0.999 for DDPG and TDM and 0.95 for HER and DDPG-Sparse.

For HER and DDPG-Sparse, we also added a penalty on the tanh pre-activation, as in BID0 .

Learning rates of the critic and the actor are chosen from {1e-4, 1e-3} and {1e-4, 1e-3} respectively.

Adam BID17 ) is used as the base optimizer with default parameters except the learning rate.

The batch size was 128.

The policies and networks are parmaeterized with neural networks with ReLU hidden activation and two hidden layers of size 300 and 300.

The policies have a tanh output activation, while the critic has no output activation (except for TDM, see A.5).

For the TDM, the goal was concatenated to the observation.

The planning horizon τ is also concatenated as an observation and represented as a single integer.

While we tried various representations for τ such as one-hot encodings and binary-string encodings, we found that simply providing the integer was sufficient.

While any distance metric for the TDM reward function can be used, we chose L1 norm − s t+1 − s g 1 to ensure that the scalar and vectorized TDMs are consistent.

For the model-based comparison, we trained a neural network dynamics model with ReLU activation, no output activation, and two hidden units of size 300 and 300.

The model was trained to predict the difference in state, rather than the full state.

The dynamics model is trained to minimize the mean squared error between the predicted difference and the actual difference.

After each state is observed, we sample a minibatch of size 128 from the replay buffer (size 1 million) and perform one step of gradient descent on this mean squared error loss.

Twenty rollouts were performed to compute the (per-dimension) mean and standard deviation of the states, actions, and state differences.

We used these statistics to normalize the states and actions before giving them to the model, and to normalize the state differences before computing the loss.

For MPC, we simulated 512 random action sequences of length 15 through the learned dynamics model and chose the first action of the sequence with the highest reward.

For TDMs, we found the most important hyperparameters to be the reward scale, τ max , and the number of updates per observations, I. As shown in FIG4 , TDMs can greatly benefit from larger values of I, though eventually there are diminishing returns and potentially negative impact, mostly likely due to over-fitting.

We found that the baselines did not benefit, except for HER which did benefit from larger I values.

For all the model-free algorithms (DDPG, DDPG-Sparse, HER, and TDMs), we performed a grid search over the reward scale in the range {0.01, 1, 100, 10000} and the number of updates per observations in the range {1, 5, 10}. For HER, we also tuned the weight given to the policy pre-tanh-activation {0, 0.01, 1}, which is described in BID0 .

For TDMs, we also tuned the best τ max in the range {15, 25, Horizon − 1}. For the half cheetah task, we performed extra searches over τ max and found τ max = 9 to be effective.

For TDMs, since we know that the true Q-function must learn to predict (negative) distances, we incorporate this prior knowledge into the Q-function by parameterizing it as Q(s, a, s g , τ ) = − f (s, a, s g , τ ) − s g 1 .

Here, f is a vector outputted by a feed-forward neural network and has the same dimension as the goal.

This parameterization ensures that the Q-function outputs non-positive values, while encouraging the Q-function to learn what we call a goal-conditioned model: f is encouraged to predict what state will be reached after τ , when the policy is trying to reach goal s g in τ time steps.

For the 1 norm, the scalar supervision regresses Q(s t , a t , s g , τ ) = − for each dimension j of the state.

A.6 TASK AND REWARD DESCRIPTIONS Benchmark tasks are designed on MuJoCo physics simulator BID33 and OpenAI Gym environments BID4 .

For the simulated reaching and pushing tasks, we use (8) and for the other tasks we use (9) for policy extraction.

The horizon (length of episode) for the pusher and ant tasks are 50.

The reaching tasks has a horizon of 100.

The half-cheetah task has a horizon of 99.7-DoF reacher.

: The state consists of 7 joint angles, 7 joint angular velocities, and 3 XYZ observation of the tip of the arm, making it 17 dimensional.

The action controls torques for each joint, totally 7 dimensional.

The reward function during optimization control and for the model-free baseline is the negative Euclidean distance between the XYZ of the tip and the target XYZ.

The targets are sampled randomly from all reachable locations of the arm at the beginning of each episode.

<|TLDR|>

@highlight

We show that a special goal-condition value function trained with model free methods can be used within model-based control, resulting in substantially better sample efficiency and performance.