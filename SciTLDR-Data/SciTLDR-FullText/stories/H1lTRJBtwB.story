The successful application of flexible, general learning algorithms to real-world robotics applications is often limited by their poor data-efficiency.

To address the challenge, domains with more than one dominant task of interest encourage the sharing of information across tasks to limit required experiment time.

To this end, we investigate compositional inductive biases in the form of hierarchical policies as a mechanism for knowledge transfer across tasks in reinforcement learning (RL).

We demonstrate that this type of hierarchy enables positive transfer while mitigating negative interference.

Furthermore, we demonstrate the benefits of additional incentives to efficiently decompose task solutions.

Our experiments show that these incentives are naturally given in multitask learning and can be easily introduced for single objectives.

We design an RL algorithm that enables stable and fast learning of structured policies and the effective reuse of both behavior components and transition data across tasks in an off-policy setting.

Finally, we evaluate our algorithm in simulated environments as well as physical robot experiments and  demonstrate substantial improvements in data data-efficiency over competitive baselines.

While recent successes in deep (reinforcement) learning for computer games (Atari (Mnih et al., 2013) , StarCraft (Vinyals et al., 2019) ), Go (Silver et al., 2017) and other high-throughput domains, e.g. (OpenAI et al., 2018) , have demonstrated the potential of these methods in the big data regime, the high cost of data acquisition has so far limited progress in many tasks of real-world relevance.

Data efficiency in machine learning generally relies on inductive biases to guide and accelerate the learning process; e.g. by including expert domain knowledge of varying granularity.

Incorporating such knowledge can accelerate learning -but when inaccurate it can also inappropriately bias the space of solutions and lead to sub-optimal results.

Robotics represents a domain in which data efficiency is critical, and human prior knowledge is commonly provided.

However, for scalability and reduced dependency on human accuracy, we can instead utilise an agent's permanent embodiment and shared environment across tasks.

Intuitively, such a scenario suggests the natural strategy of focusing on inductive biases that facilitate the sharing and reuse of experience and knowledge across tasks while other aspects of the domain can be learned.

As a general principle this relieves us from the need to inject detailed knowledge about the domain, instead we can focus on general principles that facilitate reuse (Caruana, 1997) .

Successes for transfer learning have, for example, built on optimizing initial parameters (e.g. Finn et al., 2017) , sharing models and parameters across tasks either in the form of policies or value functions (e.g. Rusu et al., 2016; Teh et al., 2017; Galashov et al., 2018) , data-sharing across tasks (e.g. Riedmiller et al., 2018; Andrychowicz et al., 2017) , or through the use of task-related auxiliary objectives (Jaderberg et al., 2016; Wulfmeier et al., 2017) .

Transfer between tasks can, however, lead to either constructive or destructive transfer for humans (Singley and Anderson, 1989) as well as for machines (Pan and Yang, 2010; Torrey and Shavlik, 2010) .

That is, jointly learning to solve different tasks can provide both benefits and disadvantages for individual tasks, depending on their similarity.

Finding a mechanism that enables transfer where possible but avoids interference is one of the long-standing research challenges.

In this paper we explore the benefits and limitations of hierarchical policies in single and multitask reinforcement learning.

Similar to Mixture Density Networks (Bishop, 1994) our models represent policies as state-conditional Gaussian mixture distributions, with separate Gaussian mixture components as low-level policies which can be selected by the high-level controller via a categorical action choice.

In the multitask setting, to obtain more robust and versatile low-level behaviors, we additionally shield the mixture components from information about the task at hand.

In this case, task information is only communicated through the choice of mixture component by the high-level controller, and the mixture components can be seen as domain-dependant, task-independent skills although the nature of these skills is not predefined and emerges during end-to-end training.

We implement this idea by building on three forms of transfer: targeted exploration via the concatenation of tasks within one episode (Riedmiller et al., 2018) , sharing transition data across tasks (Andrychowicz et al., 2017; Riedmiller et al., 2018 ), and reusing low-level components of the aforementioned policy class.

To this end we develop a novel robust and data-efficient multitask actor-critic algorithm, Regularized Hierarchical Policy Optimization (RHPO).

Our algorithm uses the multitask learning aspects of SAC (Riedmiller et al., 2018) to improve data-efficiency and robust policy optimization properties of MPO (Abdolmaleki et al., 2018a) in order to optimize hierarchical policies.

We furthermore demonstrate the generality of hierarchical policies for multitask learning via improving results also after replacing MPO as policy optimizer with another gradient-based, entropy-regularized policy optimizer (Heess et al., 2015) (see Appendix A.10).

We demonstrate that compositional, hierarchical policies -while strongly reducing training time in multitask domains -can fail to improve performance in single task domains if no additional inductive biases are given.

While multitask domains provide sufficient pressure for component specialization, and the related possibility for composition, we are required to introduce additional incentives to encourage similar developments for single task domains.

In the multitask setting, we demonstrate considerably improved performance, robustness and learning speed compared to competitive continuous control baselines demonstrating the relevance of hierarchy for data-efficiency and transfer.

We finally evaluate our approach on a physical robot for robotic manipulation tasks where RHPO leads to a significant speed up in training, enabling it to solve challenging stacking tasks on a single robot 1 .

We consider a multitask reinforcement learning setting with an agent operating in a Markov Decision Process (MDP) consisting of the state space S, the action space A, the transition probability p(s t+1 |s t , a t ) of reaching state s t+1 from state s t when executing action a t at the previous time step t. The actions are drawn from a probability distribution over actions ??(a|s) referred to as the agent's policy.

Jointly, the transition dynamics and policy induce the marginal state visitation distribution p(s).

Finally, the discount factor ?? together with the reward r(s, a) gives rise to the expected reward, or value, of starting in state s (and following ?? thereafter) V ?? (s) = E ?? [ ??? t=0 ?? t r(s t , a t )|s 0 = s, a t ??? ??(??|s t ), s t+1 ??? p(??|s t , a t )].

Furthermore, we define multitask learning over a set of tasks i ??? I with common agent embodiment as follows.

We assume shared state, action spaces and shared transition dynamics across tasks; tasks only differ in their reward function r i (s, a).

Furthermore, we consider task conditional policies ??(a|s, i).

The overall objective is defined as J(??) = E i???I E ??,p(s0) ??? t=0 ?? t r i (s t , a t ) |s t+1 ??? p(??|s t , a t ) = E i???I E ??,p(s) Q ?? (s, a, i) ,

(1) where all actions are drawn according to the policy ??, that is, a t ??? ??(??|s t , i) and we used the common definition of the state-action value function -here conditioned on the task -Q ?? (s, a, i) = E ?? [ ??? t=0 ?? t r i (s t , a t ) |a 0 = a, s 0 = s, a t ??? ??(??|s t , i), s t+1 ??? p(??|s t , a t )].

This section introduces Regularized Hierarchical Policy Optimization (RHPO) which focuses on efficient training of modular policies by sharing data across tasks; extending the data-sharing and scheduling mechanisms from Scheduled Auxiliary Control with randomized scheduling (SAC-U) (Riedmiller et al., 2018) .

We start by introducing the considered class of policies, followed by the required combination -and extension -of MPO (Abdolmaleki et al., 2018a) and SAC-U (Riedmiller et al., 2018) for training structured hierarchical policies in a multitask, off-policy setting.

We start by defining the hierarchical policy class which supports sharing sub-policies across tasks.

Formally, we decompose the per-task policy ??(a|s, i) as

with ?? H and ?? L respectively representing a "high-level" switching controller (a categorical distribution) and a "low-level" sub-policy (components of the resulting mixture distribution), where o is the index of the sub-policy.

Here, ?? denotes the parameters of both ?? H and ?? L , which we will seek to optimize.

While the number of components has to be decided externally, the method is robust with respect to this parameter (Appendix A.8.3) .

Note that, in the above formulation only the high-level controller ?? H is conditioned on the task information i; i.e. we employ a form of information asymmetry (Galashov et al., 2018; Tirumala et al., 2019; Heess et al., 2016 ) to enable the low-level policies to acquire general, task-independent behaviours.

This choice strengthens decomposition of tasks across domains and inhibits degenerate cases of bypassing the high-level controller.

Intuitively, these sub-policies can be understood as building reflex-like low-level control loops, which perform domain-dependent but task-independent behaviours and can be modulated by higher cognitive functions with knowledge of the task at hand.

In the following sections, we present the equations underlying RHPO.

For the complete pseudocode algorithm the reader is referred to the Appendix A.2.1.

To optimize the policy class described above we build on the MPO algorithm (Abdolmaleki et al., 2018a) which decouples the policy improvement step (optimizing J independently of the policy structure) from the fitting of the hierarchical policy.

Concretely, we first introduce an intermediate non-parametric policy q(a|s, i) and consider optimizing J(q) while staying close, in expectation, to a reference policy ?? ref (a|s, i)

where KL(?? ??) denotes the Kullback Leibler divergence, defines a bound on the KL, D denotes the data contained in a replay buffer, and assuming that we have an approximation of the ground-truth state-action value functionQ(s, a, i) ??? Q ?? (s, a, i) available (see Equation (4) for details on learnin?? Q from off-policy data).

Starting from an initial policy ?? ??0 we can then iterate the following steps to improve the policy ?? ?? k :

Policy Evaluation: UpdateQ such thatQ(s, a, i) ???Q ?? ?? k (s, a, i), see Equation (4).

Policy Improvement:

-Step 1: Obtain q k = arg max q J(q), under KL constraints with ?? ref = ?? ?? k (Equation (3)).

-Step 2: Obtain ?? k+1 = arg min ?? E s???D,i???I KL q k (??|s, i) ?? ?? (??|s, i) , under additional regularization (Equation (6)).

Multitask Policy Evaluation For data-efficient off-policy learning ofQ we build on scheduled auxiliary control with uniform scheduling (SAC-U) (Riedmiller et al., 2018) which exploits two main ideas to obtain data-efficiency: i) experience sharing across tasks; ii) switching between tasks within one episode for improved exploration.

Formally, we assume access to a replay buffer containing data gathered from all tasks, which is filled asynchronously to the optimization (similar to e.g. Espeholt et al. (2018) ) where for each trajectory snippet ?? = {(s 0 , a 0 , R 0 ), . . .

, (s L , a L , R L )} we record the rewards for all tasks R t = [r i1 (s t , a t ), . . .

, r i |I| (s t , a t )] as a vector in the buffer.

Using this data we define the retrace objective for learningQ, parameterized via ??, following (Munos et al., 2016; Riedmiller et al., 2018) as min

where Q ret is the L-step retrace target (Munos et al., 2016) , see the Appendix A.2.2 for details.

Multitask Policy Improvement 1:

Obtaining Non-parametric Policies We first find the intermediate policy q by maximizing Equation (3).

We obtain a closed-form solution with a non-parametric policy for each task, as

where ?? is a temperature parameter (corresponding to a given bound ) which is optimized alongside the policy optimization (see Appendix A.1.1 for a detailed derivation of the multitask case).

As mentioned above, this policy representation is independent of the form of the parametric policy ?? ?? k ; i.e. q only depends on ?? ?? k through its density.

This, crucially, makes it easy to employ complicated structured policies (such as the one introduced in Section 3.1).

The only requirement here, and in the following steps, is that we must be able to sample from ?? ?? k and calculate the gradient (w.r.t.

?? k ) of its log density (but the sampling process itself need not be differentiable).

In the second step we fit a policy to the non-parametric distribution obtained from the previous calculation by minimizing the divergence E s???D,i???I [KL(q k (??|s, i) ?? ?? (??|s, i))].

Assuming that we can sample from q k this step corresponds to maximum likelihood estimation (MLE).

Furthermore, we can regularize towards smoothly changing distributions during training -effectively mitigating optimization instabilities and introducing an inductive bias -by limiting the change of the policy (a trust-region constraint).

The idea is commonly used in on-as well as in off-policy RL (Schulman et al., 2015; Abdolmaleki et al., 2018b; a) .

The application to hierarchical policy classes highlights the importance of this constraint as investigated in Section 4.2.

Formally, we aim to obtain the solution (6), we first employ Lagrangian relaxation to make it amenable to gradient based optimization and then perform a fixed number of gradient descent steps (using Adam (Kingma and Ba, 2014)); details on this step, as well as an algorithm listing, can be found in the Appendix A.1.2.

In the following sections, we investigate the effects of training hierarchical policies in single and multitask domains, finally demonstrating how RHPO can provide compelling benefits for multitask learning in real and simulated robotic manipulation tasks and significantly reduce platform interaction time.

In the context of single-task domains from the DeepMind Control Suite (Tassa et al., 2018 ), we first demonstrate how this type of hierarchy on its own fails to improve performance and that for the model to exploit compositionality, additional incentives for component specialization are required.

Subsequently, we introduce suited incentives leading to improved performance and demonstrate that the variety of objectives in multitask domains can serve the same purpose.

The evaluation includes experiments on physical hardware with robotic manipulation tasks for the Sawyer arm, emphasizing the importance of data-efficiency.

More details on task hyperparameters as well as the results for additional ablations and all tasks from the multitask domains are provided in the Appendix A.4.

Across all tasks, we build on a distributed actor-critic framework (similar to (Espeholt et al., 2018) ) with flexible hardware assignment (Buchlovsky et al., 2019) to train all agents, performing critic and policy updates from a replay buffer, which is asynchronously filled by a set of actors.

In all figures with error bars, we visualize mean and variance derived from 3 runs.

We consider two high-dimensional tasks for continuous control: humanoid-run and humanoid-stand from Tassa et al. (2018) and compare a flat Gaussian policy to a hierarchical policy, a mixture of Gaussians with three components.

We align the update rates of all approaches for fair comparison and to focus the comparison of the algorithm and not its specific implementation 2 .

Figure  1 visualizes the results in terms of the number of actor episodes.

As can be observed, the hierarchical policy performs comparable to a flat policy with well aligned means and variances for all components as the model fails to decompose the problem.

While both the flat and hierarchical policy are initialized with means close to zero, we now include another hierarchical policy with distributed initial means for the three components ranging for all dimensions from minimum to maximum of the allowed action range (here: -1, 0, 1).

This simple change suffices to enable component specialization and significantly improved performance.

We use three simulated multitask scenarios with the Kinova Jaco and Rethink Robotics Sawyer robot arms to test in a variety of conditions.

Pile1: Here, the seven tasks of interest range from simple reaching for a block over tasks like grasping it, to the final task of stacking the block on top of another block.

In addition to the experiments in simulation, which are executed with 5 actors in a distributed setting, the same Pile1 multitask domain (same rewards and setup) is investigated with a single, physical robot in Section 4.3.

We further extend the evaluation towards two more complex multitask domains in simulation.

The first extension includes stacking with both blocks on top of the respective other block, resulting in a setting with 10 tasks (Pile2).

And a last domain including harder tasks such as opening a box and placing blocks into this box, consisting of a total of 13 tasks (Cleanup2).

We compare RHPO for training hierarchical policies against a flat, monolithic policy shared across all tasks which is provided with the additional task id as input (displayed as Monolithic in the plot) as well as policies with task dependent heads (displayed as Independent in the plots) following (Riedmiller et al., 2018) -both using MPO as the optimizer and a re-implementation of SAC-U using SVG (Heess et al., 2015) (which is related to a version of the option critic (Bacon et al., 2017) without temporal abstraction).

The baselines provide the two opposite, naive perspectives on transfer: by using the same monolithic policy across tasks we enable positive as well as negative interference and independent policies prevent policy-based transfer.

After experimentally confirming the robustness of RHPO with respect to the number of low-level sub-policies (see Appendix A.8.3), we set M proportional to the number of tasks in each domain.

Figure 2 demonstrates that the hierarchical policy (RHPO) outperforms the monolithic as well as the independent baselines.

For simple tasks such as Pile1, the difference is smaller, but the more tasks are trained and the more complex the domain becomes (cf.

Pile2 and Cleanup2), the greater is the advantage of composing learned behaviours across tasks.

Compared to SVG (Heess et al., 2015) , we observe that the baselines based on MPO already result in an improvement, which becomes even bigger with the hierarchical policies.

The results across all domains exhibit performance gains for the hierarchical model without the additional incentives from Section 4.1, demonstrating the sufficiency of variety in the training objectives to encourage component specialization and problem decomposition.

For real-world experiments, data-efficiency is crucial.

We perform all experiments in this section relying on a single robot (single actor) -demonstrating the benefits of RHPO in the low data regime.

The performed task is the real world version of the Pile1 task described in Section 4.2.

The main task objective is to stack one cube onto a second one and move the gripper away from it.

We introduce an additional third cube which serves purely as a distractor.

The setup for the experiments consists of a Sawyer robot arm mounted on a table, equipped with a Robotiq 2F-85 parallel gripper.

A basket of size 20cm 2 in front of the robot contains the three cubes.

Three cameras on the basket track the cubes using fiducials (augmented reality tags).

As in simulation, the agent is provided with proprioception information (joint positions, velocities and torques), a wrist sensor's force and torque readings, as well as the cubes' poses -estimated via the fiducials.

The agent action is five dimensional and consists of the three Cartesian translational velocities, the angular velocity of the wrist around the vertical axis and the speed of the gripper's fingers.

Figure 4 plots the learning progress on the real robot for two (out of 7) of the tasks, the simple reach tasks and the stack task -which is the main task of interest.

Plots for the learning progress of all tasks are given in the appendix A.6.

As can be observed, all methods manage to learn the reach task quickly (within about a few thousand episodes) but only RHPO with a hierarchical policy is able to learn the stacking task (taking about 15 thousand episodes to obtain good stacking success), which takes about 8 days of training on the real robot with considerably slower progress for all baselines.

In addition, we compute distributions for each component over the tasks which activate it, as well as distributions for each task over which components are being used.

For each set of distributions, we determine the Battacharyya distance metric to determine the similarity between tasks and the similarity between components in Figure 4 (right).

The plots demonstrate how the components specialise, but also provide a way to investigate our tasks, showing e.g. that the first reach task is fairly independent and that the last four tasks are comparably similar regarding the high-level components applied for their solution.

We perform a series of ablations based on the earlier introduced Pile1 domain, providing additional insights into benefits and shortcomings of RHPO and important factors for robust training.

The algorithm is well suited for sequential transfer learning based on solving new tasks with pre-trained low-level components (Appendix A.9).

We demonstrate the robustness of RHPO with respect to the number of sub-policies as well as importance of choice of regularization respectively in Appendix A.8.1 and A.8.3.

Finally, we ablate over the number of data-generating actors to evaluate all approaches with respect to data rate and illustrate how hierarchical policies are particularly relevant at lower data rates such as given by real-world robotics applications in Appendix A.8.2.

Transfer learning, in particular in the multitask context, has long been part of machine learning (ML) for data-limited domains (Caruana, 1997; Torrey and Shavlik, 2010; Pan and Yang, 2010; Taylor and Stone, 2009) .

Commonly, it is not straightforward to train a single model jointly across different tasks as the solutions to tasks might not only interfere positively but also negatively (Wang et al., 2018) .

Preventing this type of forgetting or negative transfer presents a challenge for biological (Singley and Anderson, 1989) as well as artificial systems (French, 1999) .

In the context of ML, a common scheme is the reduction of representational overlap (French, 1999; Rusu et al., 2016; Wang et al., 2018) .

Bishop (1994) utilize neural networks to parametrize mixture models for representing multi-modal distributions thus mitigating shortcomings of non-hierarchical approaches.

Rosenstein et al. (2005) demonstrate the benefits of hierarchical classification models to limit the impact of negative transfer.

Hierarchical approaches have a long history in the reinforcement learning literature (e.g. Sutton et al., 1999; Dayan and Hinton, 1993) .

Prior work commonly benefits from combining hierarchy with additional inductive biases such as (Vezhnevets et al., 2017; Nachum et al., 2018a; b; Xie et al., 2018) which employ different rewards for different levels of the hierarchy rather than optimizing a single objective for the entire model as we do.

Other works have shown the additional benefits for the stability of training and data-efficiency when sequences of high-level actions are given as guidance during optimization in a hierarchical setting (Shiarlis et al., 2018; Andreas et al., 2017; Tirumala et al., 2019) .

Instead of introducing additional training signals, we directly investigate the benefits of compositional hierarchy as provided structure for transfer between tasks.

Hierarchical models for probabilistic trajectory modelling have been used for the discovery of behavior abstractions as part of an end-to-end reinforcement learning paradigm (e.g. Teh et al., 2017; Igl et al., 2019; Tirumala et al., 2019; Galashov et al., 2018) where the models act as learned inductive biases that induce the sharing of behavior across tasks.

In a vein similar to the presented algorithm, (e.g Heess et al., 2016; Tirumala et al., 2019) share a low-level controller across tasks but modulate the low-level behavior via a continuous embedding rather than picking from a small number of mixture components.

In related work Hausman et al. (2018) Similar to our work, the options framework (Sutton et al., 1999; Precup, 2000) supports behavior hierarchies, where the higher level chooses from a discrete set of sub-policies or "options" which commonly are run until a termination criterion is satisfied.

The framework focuses on the notion of temporal abstraction.

A number of works have proposed practical and scalable algorithms for learning option policies with reinforcement learning (e.g. Bacon et al., 2017; Zhang and Whiteson, 2019; Smith et al., 2018; Riemer et al., 2018; Harb et al., 2018) or criteria for option induction (e.g. Harb et al., 2018; Harutyunyan et al., 2019) .

Rather than the additional inductive bias of temporal abstraction, we focus on the investigation of composition as type of hierarchy in the context of single and multitask learning while demonstrating the strength of hierarchical composition to lie in domains with strong variation in the objectives -such as in multitask domains.

We additionally introduce a hierarchical extension of SVG (Heess et al., 2015) , to investigate similarities to work on the option critic (Bacon et al., 2017) .

With the use of KL regularization to different ends in RL, work related to RHPO focuses on contextual bandits (Daniel et al., 2016) .

The algorithm builds on a 2-step EM like procedure to optimize linearly parametrized mixture policies.

However, their algorithm has been used only with low dimensional policy representations, and in contextual bandit and other very short horizon settings.

Our approach is designed to be applicable to full RL problems in complex domains with long horizons and with high-capacity function approximators such as neural networks.

This requires robust estimation of value function approximations, off-policy correction, and additional regularization for stable learning.

We introduce a novel framework to enable robust training and investigation of hierarchical, compositional policies in complex simulated and real-world tasks as well as provide insights into the learning process and its stability.

In simulation as well as on real robots, RHPO outperforms baseline methods which either handle tasks independently or utilize implicit sharing.

Especially with increasingly complex tasks or limited data rate, as given in real-world applications, we demonstrate hierarchical inductive biases to provide a compelling foundation for transfer learning, reducing the number of environment interactions significantly and often leading to more robust learning as well as improved final performance.

For single tasks with a single training objective all components can remain aligned, preventing problem decomposition and the hierarchical policy replicates a flat policy.

Performance improvements appear only when the individual components specialize, either via variety in the training objectives or additional incentives.

Furthermore, as demonstrated in Appendix A.9, a pre-trained set of specialized components can notably improve performance when learning new tasks.

One important next step is identifying how to optimize a basis set of components which transfers well to a wide range of tasks Since with mixture distributions, we are able to marginalize over components when optimizing the weighted likelihood over action samples in Equation 6, the extension towards multiple levels of hierarchy is trivial but can provide a valuable direction for practical future work.

While this approach partially mitigates negative interference between tasks in a parallel multitask learning scenario, addressing catastrophic inference in sequential settings remains a challenge.

We believe that especially in domains with consistent agent embodiment and high costs for data generation learning tasks jointly and information sharing is imperative.

RHPO combines several ideas that we believe will be important: multitask learning with hierarchical and compositional policy representations, robust optimization, and efficient off-policy learning.

Although we have found this particular combination of components to be very effective we believe it is just one instance of -and step towards -a spectrum of efficient learning architectures that will unlock further applications of RL both in simulation and, importantly, on physical hardware.

In this section we explain the detailed derivations for training hierarchical policies parameterized as a mixture of Gaussians.

In each policy improvement step, to obtain non-parametric policies for a given state and task distribution, we solve the following program:

To make the following derivations easier to follow we open up the expectations, writing them as integrals explicitly.

For this purpose let us define the joint distribution over states s ??? ??(s) together with randomly sampled tasks i ??? I as ??(s, i) = p(s|D)U(i ??? I), where U denotes the uniform distribution over possible tasks.

This allows us to re-write the expectations that include the corresponding distributions, i.e.

, but again, note that i here is not necessarily the task under which s was observed.

We can then write the Lagrangian equation corresponding to the above described program as

Next we maximize the Lagrangian L w.r.t the primal variable q. The derivative w.r.t q reads,

Setting it to zero and rearranging terms we obtain

However, the last exponential term is a normalization constant for q.

Therefore we can write,

Now, to obtain the dual function g(??), we plug in the solution to the KL constraint term (second term) of the Lagrangian which yields

After expanding and rearranging terms we get

Most of the terms cancel out and after rearranging the terms we obtain

Note that we have already calculated the term inside the integral in Equation 7.

By plugging in equation 7 we obtain the dual function

which we can minimize with respect to ?? based on samples from the replay buffer.

After obtaining the non parametric policies, we fit a parametric policy to samples from said nonparametric policies -effectively employing using maximum likelihood estimation with additional regularization based on a distance function T , i.e,

where D is an arbitrary distance function to evaluate the change of the new policy with respect to a reference/old policy, and denotes the allowed change for the policy.

To make the above objective amenable to gradient based optimization we employ Lagrangian Relaxation, yielding the following primal:

We solve for ?? by iterating the inner and outer optimization programs independently: We fix the parameters ?? to their current value and optimize for the Lagrangian multipliers (inner minimization) and then we fix the Lagrangian multipliers to their current value and optimize for ?? (outer maximization).

In practice we found that it is effective to simply perform one gradient step each in inner and outer optimization for each sampled batch of data.

The optimization given above is general, i.e. it works for any general type of policy.

As described in the main paper, we consider hierarchical policies of the form

In particular, in all experiments we made use of a mixture of Gaussians parametrization, where the high level policy ?? H ?? is a categorical distribution over low level ?? L ?? Gaussian policies, i.e,

where j denote the index of components and ?? is the high level policy ?? H assigning probabilities to each mixture component for a state s given the task and the low level policies are all Gaussian.

Here ?? j s are the probabilities for a categorical distribution over the components.

We also define the following distance function between old and new mixture of Gaussian policies

where T H evaluates the KL between categorical distributions and T L corresponds to the average KL across Gaussian components, as also described in the main paper (c.f.

Equation 5 in the main paper).

In order to bound the change of categorical distributions, means and covariances of the components independently -which makes it easy to control the convergence of the policy and which can prevent premature convergence as argued in Abdolmaleki et al. (2018a) log ?? ?? (a j |s t , i) following Eq. 6

following Eq. 8

As described in the main paper we consider the same setting as scheduled auxiliary control setting (SAC-X) (Riedmiller et al., 2018) to perform policy improvement (with uniform random switches between tasks every N steps within an episode, the SAC-U setting).

Given a replay buffer containing data gathered from all tasks, where for each trajectory a t ) , . . .

, r i |I| (s t , a t )] as a vector in the buffer, we define the retrace objective for learningQ, parameterized via ??, following Riedmiller et al. (2018) as

where the importance weights are defined as c k = min (1, ?? ?? k (a k |s k ,i) /b(a k |s k )), with b(a k |s k ) denoting an arbitrary behavior policy; in particular this will be the policy for the executed tasks during an

Input: N trajectories number of total trajectories requested, T steps per episode, ?? scheduling period initialize N = 0 while N < N trajectories do fetch parameters ?? // collect new trajectory from environment ?? = {} for t in [0...T ] do if t (mod ??) ??? 0 then // sample active task from uniform distribution i act ??? I end if a t ??? ?? ?? (??|s t , i act ) // execute action and determine rewards for all tasks r = [r i1 (s t , a t ), . . .

, r i |I| (s t , a t )]

?? ??? ?? ??? {(s t , a t ,r, ?? ?? (a 0 |s t , i act ))} end for send batch trajectories ?? to replay buffer N = N + 1 end while episode as in (Riedmiller et al., 2018) .

Note that, in practice, we truncate the infinite sum after L steps, bootstrapping withQ. We further perform optimization of Equation (4) via gradient descent and make use of a target network (Mnih et al., 2015) , denoted with parameters ?? , which we copy from ?? after a couple of gradient steps.

We reiterate that, as the state-action value functionQ remains independent of the policy's structure, we are able to utilize any other off-the-shelf Q-learning algorithm such as TD(0) (Sutton, 1988) .

Given that we utilize the same policy evaluation mechanism as SAC-U it is worth pausing here to identify the differences between SAC-U and our approach.

The main difference is in the policy parameterization: SAC-U used a monolithic policy for each task ??(a|s, i) (although a neural network with shared components, potentially leading to some implicit task transfer, was used).

Furthermore, we perform policy optimization based on MPO instead of using stochastic value gradients (SVG (Heess et al., 2016)).

We can thus recover a variant of plain SAC-U using MPO if we drop the hierarchical policy parameterization, which we employ in the single task experiments in the main paper.

To represent the Q-function in the multitask case we use the network architecture from SAC-X (see right sub-figure in Figure 5 ).

The proprioception of the robot, the features of the objects and the actions are fed together in a torso network.

At the input we use a fully connected first layer of 200 units, followed by a layer normalization operator, an optional tanh activation and another fully connected layer of 200 units with an ELU activation function.

The output of this torso network is shared by independent head networks for each of the tasks (or intentions, as they are called in the SAC-X paper).

Each head has two fully connected layers and outputs a Q-value for this task, given the input of the network.

Using the task identifier we then can compute the Q value for a given sample by discrete selection of the according head output.

While we use the network architecture for the Q function for all multitask experiments, we investigate different architectures for the policy in this paper.

The original SAC-X policy architecture is shown in Figure 5 (left sub-figure) .

The main structure follows the same basic principle that we use in the Q function architecture.

The only difference is that the heads compute the required parameters for the policy distribution we want to use (see subsection A.2.4).

This architecture is referenced as the independent heads (or task dependent heads).

The alternatives we investigate in this paper are the monolithic policy architecture (see Figure 6 , left sub- figure) and the hierarchical policy architecture (see Figure 6 , right sub-figure).

For the monolithic policy architecture we reduce the original policy architecture basically to one head and architecture in all multitask experiments, we investigate variations of the policy architecture (left sub-figure) in this paper (see Figure 6 ).

append the task-id as a one-hot encoded vector to the input.

For the hierarchical architecture, we build on the same torso and create a set of networks parameterizing the Gaussians which are shared across tasks and a task-specific network to parameterize the categorical distribution for each task.

The final mixture distribution is task-dependent for the high-level controller but task-independent for the low-level policies.

In this section we outline the details on the hyperparameters used for RHPO and baselines in both single task and multitask experiments.

All experiments use feed-forward neural networks.

We consider a flat policy represented by a Gaussian distribution and a hierarchical policy represented by a mixture of Gaussians distribution.

The flat policy is given by a Gaussian distribution with a diagonal covariance matrix, i.e, ??(a|s, ??) = N (??, ??).

The neural network outputs the mean ?? = ??(s) and diagonal Cholesky factors A = A(s), such that ?? = AA T .

The diagonal factor A has positive diagonal elements enforced by the softplus transform A ii ??? log(1 + exp(A ii )) to ensure positive definiteness of the diagonal covariance matrix.

Mixture of Gaussian policy has a number of Gaussian components as well as a categorical distribution for selecting the components.

The neural network outputs the Gaussian components based on the same setup described above for a single Gaussian and outputs the logits for representing the categorical distribution.

Tables 1 show the hyperparameters we used for the single tasks experiments.

We found layer normalization and a hyperbolic tangent (tanh) on the layer following the layer normalization are important for stability of the algorithms.

For RHPO the most important hyperparameters are the constraints in Step 1 and Step 2 of the algorithm.

For the SAC-U baseline we used a re-implementation of the method from (Riedmiller et al., 2018) using SVG (Heess et al., 2015) for optimizing the policy.

Concretely we use the same basic network structure as for the "Monolithic" baseline with MPO and parameterize the policy as

, where I denotes the identity matrix and ?? ?? (s, i) is computed from the network output via a softplus activation function.

Together with entropy regularization, as described in (Riedmiller et al., 2018 ) the policy can be optimized via gradient ascent, following the reparameterized gradient for a given states sampled from the replay:

which can be computed, using the reparameterization trick, as

where For the simulation of the robot arm experiments the numerical simulator MuJoCo 3 was used -using a model we identified from the real robot setup.

We run experiments of length 2 -7 days for the simulation experiments (depending on the task) with access to 2-5 recent CPUs with 32 cores each (depending on the number of actors) and 2 recent NVIDIA GPUs for the learner.

Computation for data buffering is negligible.

Compared to simulation where the ground truth position of all objects is known, in the real robot setting, three cameras on the basket track the cube using fiducials (augmented reality tags).

For safety reasons, external forces are measured at the wrist and the episode is terminated if a threshold of 20N on any of the three principle axes is exceeded (this is handled as a terminal state with reward 0 for the agent), adding further to the difficulty of the task.

The real robot setup differs from the simulation in the reset behaviour between episodes, since objects need to be physically moved around when randomizing, which takes a considerable amount of time.

To keep overhead small, object positions are randomized only every 25 episodes, using a hand-coded controller.

Objects are also placed back in the basket if they were thrown out during the previous episode.

Other than that, objects start in the same place as they were left in the previous episode.

The robot's starting pose is randomized each episode, as in simulation.

For this task we have a real setup and a MuJoCo simulation that are well aligned.

It consists of a Sawyer robot arm mounted on a table and equipped with a Robotiq 2F-85 parallel gripper.

In front of the robot there is a basket of size 20x20 cm which contains three cubes with an edge length of 5 cm (see Figure 7) .

The agent is provided with proprioception information for the arm (joint positions, velocities and torques), and the tool center point position computed via forward kinematics.

For the gripper, it receives the motor position and velocity, as well as a binary grasp flag.

It also receives a wrist sensor's force and torque readings.

Finally, it is provided with the cubes' poses as estimated via the fiducials, and the relative distances between the arm's tool center point and each object.

At each time step, a history of two previous observations is provided to the agent, along with the last two joint control commands, in order to account for potential communication delays on the real robot.

The observation space is detailed in Table 4 .

The robot arm is controlled in Cartesian mode at 20Hz.

The action space for the agent is 5-dimensional, as detailed in Table 3 .

The gripper movement is also restricted to a cubic volume above the basket using virtual walls.

For the Pile1 experiment we use 7 different task to learn, following the SAC-X principles.

The first 6 tasks are seen as auxiliary tasks that help to learn the final task (STACK_AND_LEAVE(G, Y)) of stacking the green cube on top of the yellow cube.

Overview of the used tasks:

Minimize the distance of the TCP to the green cube.

??? GRASP: Activate grasp sensor of gripper ("inward grasp signal" of Robotiq gripper)

??? LIFT(G): slin(G, 0.03, 0.10) Increase z coordinate of an object more than 3cm relative to the table.

??? ???

Sparse binary reward for bringing the green cube on top of the yellow one (with 3cm tolerance horizontally and 1cm vertically) and disengaging the grasp sensor.

??? STACK_AND_LEAVE(G, Y): stol(d z (T CP, G) + 0.10, 0.03, 0.10) * STACK(G, Y) Like STACK(G, Y), but needs to move the arm 10cm above the green cube.

Let d(o i , o j ) be the distance between the reference of two objects (the reference of the cubes are the center of mass, TCP is the reference of the gripper), and let d A be the distance only in the dimensions denoted by the set of axes A. We can define the reward function details by:

A.5.2 PILE2 Figure 8 : The Pile2 set-up in simulation with two main tasks: The first is to stack the blue on the red cube, the second is to stack the red on the blue cube.

For the Pile2 task, taken from Riedmiller et al. (2018), we use a different robot arm, control mode and task setup to emphasize that RHPO's improvements are not restricted to cartesian control or a specific robot and that the approach also works for multiple external tasks.

Here, the agent controls a simulated Kinova Jaco robot arm, equipped with a Kinova KG-3 gripper.

The robot faces a 40 x 40 cm basket that contains a red cube and a blue cube.

Both cubes have an edge length of 5 cm (see Figure 8 ).

The agent is provided with proprioceptive information for the arm and the fingers (joint positions and velocities) as well as the tool center point position (TCP) computed via forward kinematics.

Further, the simulated gripper is equipped with a touch sensor for each of the three fingers, whose value is provided to the agent as well.

Finally, the agent receives the cubes' poses, their translational and rotational velocities and the relative distances between the arm's tool center point and each object.

Neither observation nor action history is used in the Pile2 experiments.

The cubes are spawned at random on the table surface and the robot hand is initialized randomly above the table-top with a height offset of up to 20 cm above the table (minimum 10 cm).

The observation space is detailed in Table 6 .

The robot arm is controlled in raw joint velocity mode at 20 Hz.

The action space is 9-dimensional as detailed in Table 5 .

There are no virtual walls and the robot's movement is solely restricted by the velocity limits and the objects in the scene.

Analogous to Pile1 and the SAC-X setup, we use 10 different task for Pile2.

The first 8 tasks are seen as auxiliary tasks, that the agent uses to learn the main two tasks PILE_RED and PILE_BLUE, which represent stacking the red cube on the blue cube and stacking the blue cube on the red cube respectively.

The tasks used in the experiment are:

Minimize the distance of the TCP to the red cube.

Minimize the distance of the TCP to the blue cube.

Move the red cube.

Move the blue cube.

??? LIFT(R) = btol(pos z (R), 0.05)

Increase the z-coordinate of the red cube to more than 5cm relative to the table.

??? LIFT(B) = btol(pos z (B), 0.05)

Increase the z-coordinate of the blue cube to more than 5cm relative to the table.

??? ABOVE_CLOSE(R, B) = above(R, B) * stol(d (R, B) , 0.05, 0.2)

Bring the red cube to a position above of and close to the blue cube.

Bring the blue cube to a position above of and close to the red cube.

Place the red cube on another object (touches the top).

Only given when the cube doesn't touch the robot or the table.

??? PILE(B):

Place the blue cube on another object (touches the top).

Only given when the cube doesn't touch the robot or the table.

The sparse reward above(A, B) is given by comparing the bounding boxes of the two objects A and B.

If the bounding box of object A is completely above the highest point of object B's bounding box, above(A, B) is 1, otherwise above(A, B) is 0.

The Clean-Up task is also taken from Riedmiller et al. (2018) and builds on the setup described for the Pile2 task.

Besides the two cubes, the work-space contains an additional box with a moveable lid, that is always closed initially (see Figure 9 ).

The agent's goal is to clean up the scene by placing the cubes inside the box.

In addition to the observations used in the Pile2 task, the agent observes the lid's angle and it's angle velocity.

Figure 9 : The Clean-Up task set-up in simulation.

The task is solved when both bricks are in the box.

Analogous to Pile2 and the SAC-X setup, we use 13 different task for Clean-Up.

The first 12 tasks are seen as auxiliary tasks, that the agent uses to learn the main task ALL_INSIDE_BOX.

The tasks used in this experiments are:

Minimize the distance of the TCP to the red cube.

???

Minimize the distance of the TCP to the blue cube.

???

Move the red cube.

??? MOVE(B) = slin(| linvel(B) |, 0, 1):

Move the blue cube.

??? NO_TOUCH = 1 ??? GRASP Sparse binary reward, given when neither of the touch sensors is active.

??? LIFT(R) = btol(pos z (R), 0.05) Increase the z-coordinate of the red cube to more than 5cm relative to the table.

??? LIFT(B) = btol(pos z (B), 0.05) Increase the z-coordinate of the blue cube to more than 5cm relative to the table.

??? OPEN_BOX = slin(angle(lid), 0.01, 1.5) Open the lid up to 85 degrees.

??? ABOVE_CLOSE(R, BOX) = above(R, BOX) * btol(|d(R, BOX)|, 0.2) Bring the red cube to a position above of and close to the box.

??? ABOVE_CLOSE(B, BOX) = above(B, BOX) * btol(|d(B, BOX)|, 0.2) Bring the blue cube to a position above of and close to the box.

??? INSIDE(R, BOX) = inside(R, BOX) Place the red cube inside the box.

??? INSIDE(B, BOX) = inside(R, BOX) Place the blue cube inside the box.

??? Coordinating convergence progress in hierarchical models can be challenging but can be effectively moderated by the KL constraints.

We perform an ablation study varying the strength of KL constraints on the high-level controller between prior and the current policy during training -demonstrating a range of possible degenerate behaviors.

As depicted in Figure 14 , with a weak KL constraint, the high-level controller can converge too quickly leading to only a single sub-policy getting a gradient signal per step.

In addition, the categorical distribution tends to change at a high rate, preventing successful convergence for the low-level policies.

On the other hand, the low-level policies are missing task information to encourage decomposition as described in Section 3.2.

This fact, in combination with strong KL constraints, can prevent specialization of the low-level policies as the categorical remains near static, finally leading to no or very slow convergence.

As long as a reasonable constraint is picked (here a range of over 2 orders of magnitude), convergence is fast and the final policies obtain high quality for all tasks.

We note that no tuning of the constraints is required across domains and the range of admissible constraints is quite broad.

A.8.2 IMPACT OF DATA RATE Evaluating in a distributed off-policy setting enables us to investigate the effect of different rates for data generation by controlling the number of actors.

Figure 15 demonstrates how the different agents converge slower lower data rates (changing from 5 to 1 actor).

These experiments are highly relevant for the application domain as the number of available physical robots for real-world experiments is typically highly limited.

To limit computational cost, we focus on the simplest domain from Section 4.2, Pile1, in this comparison.

respect to the number of sub-policies and we will build all further experiments on setting the components equal to the number of tasks.

To additionally investigate performance in adapting trained multitask policies to novel tasks, we train agents to fulfill all but the final task in the Pile1 and Cleanup2 domains and subsequently evaluate training the models on the final task.

We consider two settings for the final policy by introducing only a new high-level controller (Sequential-Only-HL) or both an additional shared component as well as a new high-level controller (Sequential).

Figure 17 displays that in the sequential transfer setting, starting from a policy trained on a set of related tasks results in up to 5 times more data-efficiency in terms of actor episodes on the final task than training the same policy from scratch.

We observe that the final task can be solved by only reusing low-level components from previous tasks if the final task is the composition of previous tasks.

This is the case for the final task in Cleanup2 which can be completed by sequencing the previously learned components and in contrast to Pile1 where the final letting go of the block after stacking is not required for earlier tasks.

and Cleanup2 domains, and finally we train the models to adapt to the final task by either training 1-only a high-level controller or 2-a high-level controller as well as an additional component.

To test whether the benefits of a hierarchical policy transfer to a setting where a different algorithm is used to optimize the policy we performed additional experiments using SVG (Heess et al., 2015) in place of MPO.

For this purpose we use the same hierarchical policy structure as for the MPO experiments but change the categorical to an implementation that enables reparameterization with the Gumbel-Softmax trick (Maddison et al., 2016; Jang et al., 2016) .

We then change the entropy regularization from Equation (15) to a KL towards a target policy (as entropy regularization did not give stable learning in this setting) and use a regularizer equivalent to the distance function (per component KL's from Equation (12)) -using a multiplier of 0.05 for the regularization multiplier was found to be the best setting via a coarse grid search.

This is similar to previous work on hierarchical RL with SVG (Tirumala et al., 2019) .

This extension of SVG is conceptually similar to a single-step-option version of the option-critic (Bacon et al., 2017) .

Simplified, SVG is an off-policy actor-critic algorithm which builds on the reparametrisation instead of likelihood ratio trick (commonly leading to lower variance (Mohamed et al., 2019) ).

Since we do not build on temporally extended sub-policies, the algorithm simplifies to using a single critic (see Section 3.2).

The results of this experiment are depicted in Figure 18 , as can be seen, for this simple domain results in mild improvements over standard SAC-U. Similarly to the experiments in the main paper, we can see that the hierarchical policy leads to better final performance -here for a gradient-based approach.

All plots are generated by running 5 actors in parallel.

@highlight

We develop a hierarchical, actor-critic algorithm for compositional transfer by sharing policy components and demonstrate component specialization and related direct benefits in multitask domains as well as its adaptation for single tasks.

@highlight

A combination of different learning techniques for acquiring structure and learning with asymmetric data, used to train an HRL policy.

@highlight

The authors introduce a hierarchical policy structure for use in both single task and multitask reinforcement learning, and assess the structure's usefulness on complex robotic tasks.