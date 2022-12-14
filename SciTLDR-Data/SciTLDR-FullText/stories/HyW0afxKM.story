We propose an active learning algorithmic architecture, capable of organizing its learning process in order to achieve a field of complex tasks by learning sequences of primitive motor policies : Socially Guided Intrinsic Motivation with Procedure Babbling (SGIM-PB).

The learner can generalize over its experience to continuously learn new outcomes, by choosing actively what and how to learn guided by empirical measures of its own progress.

In this paper, we are considering the learning of a set of interrelated complex outcomes hierarchically organized.



We introduce a new framework called "procedures", which enables the autonomous discovery of how to combine previously learned skills in order to learn increasingly more complex motor policies (combinations of primitive motor policies).

Our architecture can actively decide which outcome to focus on and which exploration strategy to apply.

Those strategies could be autonomous exploration, or active social guidance, where it relies on the expertise of a human teacher providing demonstrations at the learner's request.

We show on a simulated environment that our new architecture is capable of tackling the learning of complex motor policies, to adapt the complexity of its policies to the task at hand.

We also show that our "procedures" increases the agent's capability to learn complex tasks.

Recently, efforts in the robotic industry and academic field have been made for integrating robots in previously human only environments.

In such a context, the ability for service robots to continuously learn new skills, autonomously or guided by their human counterparts, has become necessary.

They would be needed to carry out multiple tasks, especially in open environments, which is still an ongoing challenge in robotic learning.

Those tasks can be independent and self-contained but they can also be complex and interrelated, needing to combine learned skills from simpler tasks to be tackled efficiently.

The range of tasks those robots need to learn can be wide and even change after the deployment of the robot, we are therefore taking inspiration from the field of developmental psychology to give the robot the ability to learn.

Taking a developmental robotic approach BID13 , we combine the approaches of active motor skill learning of multiple tasks, interactive learning and strategical learning into a new learning algorithm and we show its capability to learn a mapping between a continuous space of parametrized outcomes (sometimes referred to as tasks) and a space of parametrized motor policies (sometimes referred to as actions).

Classical techniques based on Reinforcement Learning BID19 BID21 still need an engineer to manually design a reward function for each particular task.

Intrinsic motivation, which triggers curiosity in human beings according to developmental psychology BID8 , inspired knowledge-based algorithms BID17 .

The external reward is replaced by a surprise function: the learner is driven by unexpected outcomes BID4 .

However those approaches encounter limitations when the sensorimotor dimensionality increases.

Another approach using competence progress measures successfully drove the learner's exploration through goal-babbling BID2 BID20 .However when the dimension of the outcome space increases, these methods become less efficient BID3 due to the curse of dimensionality, or when the reachable space of the robot is small compared to its environment.

To enable robots to learn a wide range of tasks, and even an infinite number of tasks defined in a continuous space, heuristics such as social guidance can help by driving its exploration towards interesting and reachable space fast.

Combining intrinsically motivated learning and imitation BID16 has bootstrapped exploration by providing efficient human demonstrations of motor policies and outcomes.

Also, such a learner has been shown to be more efficient if requesting actively a human for help when needed instead of being passive, both from the learner or the teacher perspective BID6 .

This approach is called interactive learning and it enables a learner to benefit from both local exploration and learning from demonstration.

Information could be provided to the robot using external reinforcement signals (Thomaz & Breazeal, 2008) , actions BID11 , advice operators BID0 , or disambiguation among actions BID7 .

Another advantage of introducing imitation learning techniques is to include non-robotic experts in the learning process BID7 .

One of the key element of these hybrid approaches is to choose when to request human information or learn in autonomy such as to diminish the teacher's attendance.

Approaches enabling the learner to choose either what to learn (which outcome to focus on) or how to learn (which strategy to use such as imitation) are called strategic learning BID12 .

They aim at giving an autonomous learner the capability to self-organize its learning process.

Some work has been done to enable a learner to choose on which task space to focus.

The SAGG-RIAC algorithm BID2 , by self-generating goal outcomes fulfills this objective.

Other approaches focused on giving the learner the ability to change its strategy BID1 and showed it could be more efficient than each strategy taken alone.

Fewer studies have been made to enable a learner to choose both its strategy and its target outcome.

The problem was introduced and studied in BID12 , and was implemented for an infinite number of outcomes and policies in continuous spaces by the SGIM-ACTS algorithm BID14 .

This algorithm is capable of organizing its learning process, by choosing actively both which strategy to use and which outcome to focus on.

It relies on the empirical evaluation of its learning progress.

It could choose among autonomous exploration driven by intrinsic motivation and low-level imitation of one of the available human teachers to learn more efficiently.

It showed its potential to learn on a real high dimensional robot a set of hierarchically organized tasks BID9 , which is why we consider it to learn complex motor policies.

In this article, we tackle the learning of complex motor policies, which we define as combinations of simpler policies.

We describe it more concretely as a sequence of primitive policies.

A first approach to learning complex motor policies is to use via-points BID21 .

Via-points enable the definition of complex motor policies.

Those via-points are in the robot motor policy space.

When increasing the size of the complex policy (by chaining more primitive actions together), we can tackle more complex tasks.

However, this would increase the difficulty for the learner to tackle simpler tasks which would be reachable using less complex policies.

Enabling the learner to decide autonomously the complexity of the policy necessary to solve a task would allow the approach to be adaptive, and suitable to a greater number of problems.

Options BID22 are a different way to tackle this problem by offering temporally abstract actions to the learner.

However each option is built to reach one particular task and they have only been tested for discrete tasks and actions, in which a small number of options were used, whereas our new proposed learner is to be able to create an unlimited number of complex policies.

As we aim at learning a hierarchical set of interrelated complex tasks, our algorithm could also benefit from this task hierarchy (as BID10 did for learning tool use with simple primitive policies only), and try to reuse previously acquired skills to build more complex ones.

BID5 showed that building complex actions made of lower-level actions according to the task hierarchy can bootstrap exploration by reaching interesting outcomes more rapidly.

In this paper, we would like to enable a robot learner to achieve a wide range of tasks that can be inter-related and complex.

We allow the robot to use sequences of actions of undetermined length to achieve these tasks.

We propose an algorithm for the robot to learn which sequence of actions to use to achieve any task in the infinite field of tasks.

The learning algorithm has to face the problem of unlearnability of infinite task and policy spaces, and the curse of dimensionality of sequences of high-dimensionality policy spaces.

Thus, we extend SGIM-ACTS so as to learn complex motor policies of unlimited size.

We develop a new framework called "procedures" (see Section 2.2) which proposes to combine known policies according to their outcome.

Combining these, we developed a new algorithm called Socially Guided Intrinsic Motivation with Procedure Babbling (SGIM-PB) capable of taking task hierarchy into account to learn a set of complex interrelated tasks using adapted complex policies.

We will describe an experiment, on which we have tested our algorithm, and we will present and analyze the results.

Inspired by developmental psychology, we combine interactive learning and autonomous exploration in a strategic learner, which learning process is driven by intrinsic motivation.

This learner also takes task hierarchy into account to reuse its previously learned skills while adapting the complexity of its policy to the complexity of the task at hand.

In this section, we formalize our learning problem and explain the principles of SGIM-PB.

In our approach, an agent can perform motions through the use of policies ?? ?? , parametrized by ?? ??? ??, and those policies have an effect on the environment, which we call the outcome ?? ??? ???. The agent is then to learn the mapping between ?? and ???: it learns to predict the outcome ?? of each policy ?? ?? (the forward model M ), but more importantly, it learns which policy to choose for reaching any particular outcome (an inverse model L).

The outcomes ?? can be of different dimensionality and thus be split in subspaces ??? i ??? ???. The policies do not only consist of one primitive but of a succession of primitives (each encoded by the parameters in ??) that are executed sequentially by the agent.

Hence, policies also are of different dimensionality and are split in policy spaces ?? i ??? ?? (where i corresponds to the number of primitives used in each policy of ?? i ).

Complex policies are represented by concatenating the parameters of each of its primitive policies in execution order.

We take the trial and error approach, and suppose that ??? is a metric space, meaning the learner has a means of evaluating a distance between two outcomes d(?? 1 , ?? 2 ).

As this algorithm tackles the learning of complex hierarchically organized tasks, exploring and exploiting this hierarchy could ease the learning of the more complex tasks.

We define procedures as a way to encourage the robot to reuse previously learned skills, and chain them to build more complex ones.

More formally, a procedure is built by choosing previously known outcomes (t 1 , t 2 , ..., t n ??? ???) and is noted t 1 t 1 ... t n .Executing a procedure t 1 t 1 ... t n means building the complex policy ?? ?? corresponding to the succession of policies ?? ??i , i ??? 1, n (potentially complex as well) and execute it (where the ?? ??i reach best the t i ???i ??? 1, n respectively).

As the subtasks t i are generally unknown from the learner, the procedure is updated before execution (see Algo.

1) to subtasks t i which are the closest known by the learner according to its current skill set (by executing respectively ?? ?? 1 to ?? ?? n ).

When the agent selects a procedure to be executed, this latter is only a way to build the complex policy which will actually be executed.

So the agent does not check if the subtasks are actually reached when executing a procedure.

Algorithm 1 Procedure modification before execution DISPLAYFORM0

Outcome & The SGIM-PB algorithm (see Algo.

2, FIG0 ) learns by episodes, where an outcome ?? g ??? ??? to target and an exploration strategy ?? have been selected.

It is an extension of SGIM-ACTS BID14 , which can perform complex motor policies and size 2 procedures (sequences of 2 subtasks only).

It uses the same interest model and memory based inverse model.

In an episode under the policy space exploration strategy, the learner tries to optimize the policy ?? ?? to produce ?? g by choosing between random exploration of policies and local optimization, following the SAGG-RIAC algorithm BID2 ) (Goal-Directed Policy Optimization(?? g )).Local optimization uses local linear regression.

This is a slightly modified version of the SGIM-ACTS autonomous exploration strategy.

In an episode under the procedural space exploration strategy, the learner builds a size 2 procedure t i t j such as to reproduce the goal outcome ?? g the best (Goal-Directed Procedure Optimization(?? g )).

It chooses either random exploration of procedures (which builds procedures by generating two subtasks at random) when the goal outcome is far from any previously reached one, or local procedure optimization, which optimizes a procedure using local linear regression.

The procedure built is then modified and executed, following Algo.

1.In an episode under the mimicry of one policy teacher strategy, the learner requests a demonstration ?? d from the chosen teacher.

?? d is selected by the teacher as the closest from the goal outcome ?? g in its demonstration repertoire.

The learner then repeats the demonstrated policy (Mimic Policy(?? d )).

It is a strategy directly also available in the SGIM-ACTS algorithm.

In an episode under the mimicry of one procedural teacher strategy, the learner requests a procedural demonstration of size 2 t di t dj which is built by the chosen teacher according to a preset function which depends on the target outcome ?? g .

Then the learner tries to reproduce the demonstrated procedure by refining and executing it, following Algo.

1 (Mimic Procedure(t di t dj )).In both autonomous exploration strategies, the learner uses a method, Goal-Directed Optimization, to optimize its input parameters (procedure for the procedure exploration and policy for the policy exploration) to reach ?? g best.

This generic method works similarly in both cases, by creating random inputs, if the goal outcome ?? g is far from any previously reached one, or local optimization which uses linear regression.

After each episode, the learner stores the policies and modified procedures executed along with their reached outcomes in its episodic memory.

It computes its competence in reaching the goal outcome ?? g by computing the distance d(?? r , ?? g )) with the outcome ?? r it actually reached.

Then it updates its interest model by computing the interest interest(??, ??) of the goal outcome and each outcome reached (including the outcome spaces reached but not targeted): DISPLAYFORM0 where K(??) is the cost of the strategy used and the progress p(??) is the derivate of the competence.

The learning agent then uses these interest measures to partition the outcome space ??? into regions of high and low interest.

For each strategey ??, the outcomes reached and the goal are added to their partition region.

Over a fixed number of measures of interest in the region, it is then partitioned into 2 subregions so as maximise the difference in interest between the 2 subregions.

The method used is detailed in BID15 .

Thus, the learning agent discovers by itself how to organise its learning process and partition its task space into unreachable regions, easy regions and difficult regions, based on empirical measures of competence and interest.

The choice of strategy and goal outcome is based on the empirical progress measured in each region R n of the outcome space ???. ?? g , ?? are chosen stochastically (with respectively probabilities p 1 , p 2 , p 3 ), by one of the sampling modes:??? mode 1: choose ?? and ?? g ??? ??? at random; ??? mode 2: choose an outcome region R n and a strategy ?? with a probability proportional to its interest value.

Then generate ?? g ??? R n at random; ??? mode 3: choose ?? and R n like in mode 2, but generate a goal ?? g ??? R n close to the outcome with the highest measure of progress.

When the learner computes nearest neighbours to select policies or procedures to optimize (when choosing local optimization in any of both autonomous exploration strategies and when refining procedures), it actually uses a performance metric (1) which takes into account the complexity of the policy chosen: DISPLAYFORM1 where d(??, ?? g ) is the normalized Euclidean distance between the target outcome ?? g and the outcome ?? reached by the policy, ?? is a constant and n is equal to the size of the policy (the number of primitives chained).In this section, we have formalized the problem of learning an inverse model between an infinite space of outcomes and an infinite space of policies.

We have introduced the framework of procedures to allow the learning agent to learn sequences of primitive policies as task compositions.

We have then proposed SGIM-PB as a learning algorithm that leverages goal-babbling for autonomous exploration, sequences to learn complex policies, and social guidance to bootstrap the learning.

SGIM-PB learns to reach an ensemble of outcomes, by mapping them with policies, but also with subgoal outcomes.

The formalization and algorithmic architecture proposed are general and can apply to a high number of problems.

The requirements for an experimental setup are:??? to define the primitive policies of the robot??? to define the different outcomes the user is interested in by defining the variables from the sensors needed and a rough range of their values (we do not need a precise estimation as the algorithm is robust to overestimations of these ranges, see Nguyen & Oudeyer FORMULA2 ).??? a measure for the robot to assess its own performance such as a distance, as in all intrinsic motivation based algorithms.

This measure is used as an internal reward function.

Contrarily to classical reinforcement learning problems, this reward function is not fine tuned to the specific goal at hand, but is a generic function for all the goals in the outcome space.??? the environment and robot can reset to an initial state, as in most reinforcement learning algorithms.

In this study, we designed an experiment with a simulated robotic arm, which can move in its environment and interact with objects in it.

It can learn an infinite number of tasks, organized as 6 hierarchically organized types of tasks.

The robot is capable of performing complex policies of unrestricted size (i.e. consisting of any number of primitives), with primitive policies highly redundant and of high dimensionality.

The FIG1 shows environmental setup (contained in a cube delimited by (x, y, z) ??? [???1; 1] 3 ).

The learning agent is a planar robotic arm of 3 joints with the base centred on the horizontal plane, able to rotate freely around the vertical axis (each link has a length of 0.33) and change its vertical position.

The robot can grab objects in this environment, by hovering its arm tip (blue in the FIG1 close to them, which position is noted (x 0 , y 0 , z 0 ).

The robot can interact with:??? Floor (below z = 0.0): limits the motions of the robot, slightly elastic which enable the robot to go down to z = ???0.2 by forcing on it;??? Pen: can be moved around and draw on the floor, broken if forcing too much on the floor (when z <= ???0.3);??? Joystick 1 (the left one on the figure): can be moved inside a cube-shaped area (automatically released otherwise, position normalized for this area), its x-axis position control a video-game character x position on the screen when grabbed by the robot; DISPLAYFORM0 Figure 2: Experimental setup: a robotic arm, can interact with the different objects in its environment (a pen and two joysticks).

Both joysticks enable to control a video-game character (represented in top-right corner).

A grey floor limits its motions and can be drawn upon using the pen (a possible drawing is represented).??? Joystick 2 (the right one on the figure): can be moved inside a cube-shaped area (automatically released otherwise, position normalized for this area), its y-axis position control a video-game character y position on the screen when grabbed by the robot;??? Video-game character: can be moved on the screen by using the two joysticks, its position is refreshed only at the end of a primitive policy execution for the manipulated joystick.

The robot grabber can only handle one object.

When it touches a second object, it breaks, releasing both objects.

The robot always starts from the same position before executing a policy, and primitives are executed sequentially without getting back to this initial position.

Whole complex policies are recorded with their outcomes, but each step of the complex policy execution is recorded.

The motions of each of the three joints of the robot are encoded using a one-dimensional Dynamic Movement Primitive BID18 , defined by the system: DISPLAYFORM0 where x and v are the position and velocity of the system; s is the phase of the motion; x 0 and g are the starting and end position of the motion; ?? is a factor used to temporally scale the system (set to fix the length of a primitive execution); K and D are the spring constant and damping term fixed for the whole experiment; ?? is also a constant fixed for the experiment; and f is a non-linear term used to shape the trajectory called the forcing term.

This forcing term is defined as: DISPLAYFORM1 where ?? i (s) = exp(???h i (s ??? c i ) 2 ) with centers c i and widths h i fixed for all primitives.

There are 3 weights ?? i per DMP.The weights of the forcing term and the end positions are the only parameters of the DMP used by the robot.

The starting position of a primitive is set by either the initial position of the robot (if it is starting a new complex policy) or the end position of the preceding primitive.

The robot can also set its position on the vertical axis z for every primitive.

Therefore a primitive policy ?? ?? is parametrized by: DISPLAYFORM2 where DISPLAYFORM3 ) corresponds to the DMP parameters of the joint i, ordered from base to tip, and z is the fixed vertical position.

When combining two or more primitive policies (?? ??0 , ?? ??1 , ...), in a complex policies ?? ?? , the parameters (?? 0 , ?? 1 , ...) are simply concatenated together from the first primitive to the last.

The outcome subspaces the robot learns to reach are hierarchically organized and defined as:??? ??? 0 : the position (x 0 , y 0 , z 0 ) of the end effector of the robot in Cartesian coordinates at the end of a policy execution; ??? ??? 1 : the position (x 1 , y 1 , z 1 ) of the pen at the end of a policy execution if the pen is grabbed by the robot; ??? ??? 2 : the first (x a , y a ) and last (x b , y b ) points of the last drawn continuous line on the floor if the pen is functional (x a , y a , x b , y b ); ??? ??? 3 : the position (x 3 , y 3 , z 3 ) of the first joystick at the end of a policy execution if it is grabbed by the robot; ??? ??? 4 : the position (x 4 , y 4 , z 4 ) of the second joystick at the end of a policy execution if it is grabbed by the robot; ??? ??? 5 : the position (x 5 , y 5 ) of the video-game character at the end of a policy execution if moved.

To help the SGIM-PB learner, procedural teachers were available so as to provide procedures for every outcome subspaces but ??? 0 .

Each teacher was only giving procedures useful for its own outcome space, and was aware of the task representation.

They all had a cost of 5.

The rules used to provide procedures are the following:??? ProceduralTeacher1 (??? 1 ): t 1 t 0 with t 1 ??? ??? 1 corresponding to the pen initial position and t 0 ??? ??? 0 corresponding to the desired final pen position; ??? ProceduralTeacher2 (??? 2 ): t 1 t 0 with t 1 ??? ??? 1 corresponding to the point on the z = 1.0 plane above the first point of the desired drawing, and t 0 ??? ??? 0 corresponding to the desired final drawing point; ??? ProceduralTeacher3 (??? 3 ): t 3 t 0 with t 3 ??? ??? 3 and t 3 = (0, 0, 0), t 0 ??? ??? 0 corresponding to the end effector position corresponding to the desired final position of the first joystick; ??? ProceduralTeacher4 (??? 4 ): t 4 t 0 with t 4 ??? ??? 4 and t 4 = (0, 0, 0), t 0 ??? ??? 0 corresponding to the end effector position corresponding to the desired final position of the second joystick;??? ProceduralTeacher5 (??? 5 ): t 3 t 4 with t 3 ??? ??? 3 and t 3 = (x, 0, 0) with x corresponding to the desired x-position of the video-game character, t 4 ??? ??? 4 and t 4 = (0, y, 0) with y corresponding to the desired y-position of the video-game character.

We also added policy teachers corresponding to the same outcome spaces to bootstrap the robot early learning process.

The strategy attached to each teacher has a cost of 10.

Each teacher was capable to provide demonstrations (as policies executable by the robot) linearly distributed in its outcome space: To evaluate our algorithm, we created a benchmark dataset for each outcome space ??? i , linearly distributed across the outcome space dimensions, for a total of 27,600 points.

The evaluation consists in computing the normalized Euclidean distance between each of the benchmark outcome and their nearest neighbour in the learner dataset.

Then we compute the mean distance to benchmark for each outcome space.

The global evaluation is the mean evaluation for the 6 outcome spaces.

This process is then repeated across the learning process at predefined and regularly distributed timestamps.

DISPLAYFORM0 Then to asses our algorithm efficiency, we compare its results with 3 other algorithms:??? SAGG-RIAC: performs autonomous exploration of the policy space ?? guided by intrinsic motivation; ??? SGIM-ACTS: interactive learner driven by intrinsic motivation.

Choosing between autonomous exploration of the policy space ?? and mimicry of one of the available policy teachers; ??? IM-PB: performs both autonomous exploration of the procedural space and the policy space, guided by intrinsic motivation; ??? SGIM-PB: interactive learner driven by intrinsic motivation.

Choosing between autonomous exploration strategies (either of the policy space or the procedural space) and mimicry of one of the available teachers (either policy or procedural teachers).Each algorithm was run 5 times on this setup.

Each run, we let the algorithm performs 25,000 iterations (complex policies executions).

The value of ?? for this experiment is 1.2.

The probabilities to choose either of the sampling mode of SGIM-PB are p 1 = 0.15, p 2 = 0.65, p 3 = 0.2.

The code run for this experiment can be found here.

The Fig. 3 shows the global evaluation of all the tested algorithms, which corresponds to the mean error made by each algorithm to reproduce the benchmarks with respect to the number of complete complex policies tried.

The algorithms capable of performing procedures (IM-PB and SGIM-PB) have errors that drop to levels lower than the their non-procedure equivalents (respectively SAGG-RIAC and SGIM-ACTS), moreover since the beginning of the learning process (shown on Fig. 3 ).

It seems that the procedures bootstrap the exploration, enabling the learner to progress further.

Indeed, the autonomous learner SAGG-RIAC has significantly better performance when it can use procedures and is thus upgraded to IM-PB.We can also see that the SGIM-PB algorithm has a very quick improvement in global evaluation owing to the bootstrapping effect of the different teachers.

It goes lower to the final evaluation of SAGG-RIAC (0.17) after only 500 iterations.

This bootstrapping effect comes from the mimicry teachers, as it is also observed for SGIM-ACTS which shares the same mimicry teachers.

If we look at the evaluation on each individual outcome space (Fig. 4) , we can see that the learners with demonstrations (SGIM-PB and SGIM-ACTS) outperforms the other algorithms, except for the outcome space ??? 5 where IM-PB is better, due to the fact that IM-PB practiced much more on this outcome space (500 iterations where the goal was in ??? 5 against 160 for SGIM-PB) on this outcome space.

SGIM-PB and SGIM-ACTS are much better than the other algorithms on the two joysticks outcome spaces (??? 3 and ??? 4 ).

This is not surprising given the fact that those outcome spaces require precise policies.

Indeed, if the end-effector gets out of the area where it can control the joystick, the latter is released, thus potentially ruining the attempt.

So on these outcome spaces working directly on carefully crafted policies can alleviate this problem, while using procedures might be tricky, as the outcomes used don't take into account the motion trajectory but merely its final state.

SGIM-PB was provided with such policies by the policy teachers.

Also if we compare the results of the autonomous learner without procedures (SAGG-RIAC) with the one with procedures (IM-PB), we can see that it learn less on any outcome space but ??? 0 (which was the only outcome space reachable using only single primitive policies and that could not benefit from using the task hierarchy to be learned) and especially for ??? 1 , ??? 2 and ??? 5 which were the most hierarchical in this setup.

More generally, it seems than on this highly hierarchical ??? 5 , the learners with procedures were better.

So the procedures helped when learning any potentially hierarchical task in this experiment.

We further analyzed the results of our SGIM-PB learner.

We looked in its learning process to see which pairs of teachers and target outcomes it has chosen (Fig. 5) .

It was capable to request demonstrations from the relevant teachers depending on the task at hand, except for the outcome space ??? 0 which had no human teachers and therefore could not find a better teacher to help it.

Indeed, for the outcome space ??? 2 , the procedural teacher (ProceduralTeacher2) specially built for this outcome space was greatly chosen.

Figure 5: Choices of teachers and target outcomes of the SGIM-PB learnerWe wanted to see if our SGIM-PB learner adapts the complexity of its policies to the working task.

We draw 1,000,000 goal outcomes for each of the ??? 0 , ??? 1 and ??? 2 subspaces (chosen because they are increasingly complex) and we let the learner choose the known policy that would reach the closest outcome.

Fig. 6 shows the results of this analysis.

Figure 6: Number of policies selected per policy size for three increasingly more complex outcome spaces by the SGIM-PB learnerAs we can see on those three interrelated outcome subspaces (Fig. 6) , the learner is capable to adapt the complexity of its policies to the outcome at hand.

It chooses longer policies for the ??? 1 subspace (policies of size 2 and 3 while using mostly policies of size 1 and 2 for ??? 0 ) and even longer for the ??? 2 subspace (using far more policies of size 3 than for the others).

It shows that our learner is capable to correctly limit the complexity of its policies instead of being stuck into always trying longer and longer policies.

With this experiment, we show the capability of SGIM-PB to tackle the learning of a set of multiple interrelated complex tasks.

It successfully discovers the hierarchy between tasks and uses complex motor policies to learn a wider range of tasks.

It is capable to correctly choose the most adapted teachers to the target outcome when available.

Though it is not limited in the size of policies it could execute, the learner shows it could adapt the complexity of its policies to the task at hand.

The procedures greatly improved the learning capability of autonomous learners, as shows the difference between IM-PB and SAGG-RIAC .

Our SGIM-PB shows it is capable to use procedures to discover the task hierarchy and exploit the inverse model of previously learned skills.

More importantly, it shows it can successfully combine the ability of SGIM-ACTS to progress quickly in the beginning (owing to the mimicry teachers) and the ability of IM-PB to progress further on highly hierarchical tasks (owing to the procedure framework).In this article, we aimed to enable a robot to learn sequences of actions of undetermined length to achieve a field of outcomes.

To tackle this high-dimensionality learning between a continuous high-dimensional space of outcomes and a continuous infinite dimensionality space of sequences of actions , we used techniques that have proven efficient in previous studies: goal-babbling, social guidance and strategic learning based on intrinsic motivation.

We extended them with the procedures framework and proposed SGIM-PB algorithm, allowing the robot to babble in the procedure space and to imitate procedural teachers.

We showed that SGIM-PB can discover the hierarchy between tasks, learn to reach complex tasks while adapting the complexity of the policy.

The study shows that :??? procedures allow the learner to learn complex tasks, and adapt the length of sequences of actions to the complexity of the task ??? social guidance bootstraps the learning owing to demonstrations of primitive policy in the beginning, and then to demonstrations of procedures to learn how to compose tasks into sequences of actions ??? intrinsic motivation can be used as a common criteria for active learning for the robot to choose both its exploration strategy, its goal outcomes and the goal-oriented procedures.

However a precise analysis of the impact of each of the different strategies used by our learning algorithm could give us more insight in the roles of the teachers and procedures framework.

Also, we aim to illustrate the potency of our SGIM-PB learner on a real-world application.

We are currently designing such an experiment with a real robotic platform.

Besides, the procedures are defined as combinations of any number of subtasks but are used in the illustration experiment as only combinations of two subtasks.

It could be a next step to see if the learning algorithm can handle the curse of dimensionality of a larger procedure space, and explore combinations of any number of subtasks.

Moreover, the algorithm can be extended to allow the robot learner to decide on how to execute a procedure.

In the current version, we have proposed the "refinement process" to infer the best policy.

We could make this refinement process more recursive, by allowing the algorithm to select, not only policies, but also lower-level procedures as one of the policy components.

@highlight

The paper describes a strategic intrinsically motivated learning algorithm which tackles the learning of complex motor policies.