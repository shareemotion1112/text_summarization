Hierarchical planning, in particular, Hierarchical Task Networks, was proposed as a method to describe plans by decomposition of tasks to sub-tasks until primitive tasks, actions, are obtained.

Plan verification assumes a complete plan as input, and the objective is finding a task that decomposes to this plan.

In plan recognition, a prefix of the plan is given and the objective is finding a task that decomposes to the (shortest) plan with the given prefix.

This paper describes how to verify and recognize plans using a common method known from formal grammars, by parsing.

Hierarchical planning is a practically important approach to automated planning based on encoding abstract plans as hierarchical task networks (HTNs) BID3 .

The network describes how compound tasks are decomposed, via decomposition methods, to sub-tasks and eventually to actions forming a plan.

The decomposition methods may specify additional constraints among the subtasks such as partial ordering and causal links.

There exist only two systems for verifying if a given plan complies with the HTN model (a given sequence of actions can be obtained by decomposing some task).

One system is based on transforming the verification problem to SAT BID2 and the other system is using parsing of attribute grammars BID1 .

Only the parsing-based system supports HTN fully (the SAT-based system does not support the decomposition constraints).Parsing became popular in solving the plan recognition problem BID5 as researchers realized soon the similarity between hierarchical plans and formal grammars, specifically context-free grammars with parsing trees close to decomposition trees of HTNs.

The plan recognition problem can be formulated as the problem of adding a sequence of actions after some observed partial plan such that the joint sequence of actions forms a complete plan generated from some task (more general formulations also exist).

Hence plan recognition can be seen as a generalization of plan verification.

There exist numerous approaches to plan recognition using parsing or string rewriting (Avrahami-Zilberbrand and Kaminka 2005; BID5 BID4 BID5 ), but they use hierarchical models that are weaker than HTNs.

The languages defined by HTN planning problems (with partial-order, preconditions and effects) lie somewhere between context-free (CF) and context-sensitive (CS) languages BID5 so to model HTNs one needs to go beyond the CF grammars.

Currently, the only grammar-based model covering HTNs fully uses attribute grammars BID0 .

Moreover, the expressivity of HTNs makes the plan recognition problem undecidable BID2 .

Currently, there exists only one approach for HTN plan recognition.

This approach relies on translating the plan recognition problem to a planning problem BID5 , which is a method invented in BID5 .In this paper we focus on verification and recognition of HTN plans using parsing.

The uniqueness of the proposed methods is that they cover full HTNs including task interleaving, partial order of sub-tasks, and other decomposition constraints (prevailing constraints, specifically).

The methods are derived from the plan verification technique proposed in BID1 .There are two novel contributions of this paper.

First, we will simplify the above mentioned verification technique by exploiting information about actions and states to improve practical efficiency of plan verification.

Second, we will extend that technique to solve the plan (task) recognition problem.

For plan verification, only the method in BID1 supports HTN fully.

We will show that the verification algorithm can be much simpler and, hence, it is expected to be more efficient.

For plan recognition, the method proposed in BID5 can in principle support HTN fully, if a full HTN planner is used (which is not the case yet as prevailing conditions are not supported).

However, like other plan recognition techniques it requires the top task (the goal) and the initial state to be specified as input.

A practical difference of out methods is that they do not require information about possible top (root) tasks and an initial state as their input.

This is particularly interesting for plan/task recognition, where existing methods require a set of candidate tasks (goals) to select from (in principle, they may use all tasks as candidates, but this makes them inefficient).

In this paper we work with classical STRIPS planning that deals with sequences of actions transferring the world from a given initial state to a state satisfying certain goal condition.

World states are modelled as sets of propositions that are true in those states and actions are changing validity of certain propositions.

Formally, let P be a set of all propositions modelling properties of world states.

Then a state S ⊆ P is a set of propositions that are true in that state (every other proposition is false).

Later, we will use the notation S + = S to describe explicitly the valid propositions in the state S and S − = P \ S to describe explicitly the propositions not valid in the state S.Each action a is described by three sets of propositions DISPLAYFORM0 a describes positive preconditions of action a, that is, propositions that must be true right before the action a. Some modeling approaches allow also negative preconditions, but these preconditions can be compiled away.

For simplicity reasons we assume positive preconditions only (the techniques presented in this paper can also be extended to cover negative preconditions directly).

Action a is applicable to state S iff B + a ⊆ S. Sets A + a and A − a describe positive and negative effects of action a, that is, propositions that will become true and false in the state right after executing the action a. If an action a is applicable to state S then the state right after the action a is: DISPLAYFORM1 (1) γ(S, a) is undefined if an action a is not applicable to state S. The classical planning problem, also called a STRIPS problem, consists of a set of actions A, a set of propositions S 0 called an initial state, and a set of goal propositions G + describing the propositions required to be true in the goal state (again, negative goal is not assumed as it can be compiled away).

A solution to the planning problem is a sequence of actions a 1 , a 2 , . . .

, a n such that S = γ(...γ(γ(S 0 , a 1 ), a 2 ), ..., a n ) and G + ⊆ S. This sequence of actions is called a plan.

The plan verification problem is formulated as follows: given a sequence of actions a 1 , a 2 , . . .

, a n , and goal propositions G + , is there an initial state S 0 such that the sequence of actions forms a valid plan leading from S 0 to a goal state?

In some formulations, the initial state might also be given as an input to the verification problem.

To simplify the planning process, several extensions of the basic STRIPS model were proposed to include some control knowledge.

Hierarchical Task Networks BID3 were proposed as a planning domain modeling framework that includes control knowledge in the form of recipes how to solve specific tasks.

The recipe is represented as a task network, which is a set of sub-tasks to solve a given task together with the set of constraints between the sub-tasks.

Let T be a compound task and ({T 1 , ..., T k }, C) be a task network, where C are its constraints (see later).

We can describe the decomposition method as a derivation (rewriting) rule: DISPLAYFORM0 The planning problem in HTN is specified by an initial state (the set of propositions that hold at the beginning) and by an initial task representing the goal.

The compound tasks need to be decomposed via decomposition methods until a set of primitive tasks -actions -is obtained.

Moreover, these actions need to be linearly ordered to satisfy all the constraints obtained during decompositions and the obtained plan -a linear sequence of actions -must be applicable to the initial state in the same sense as in classical planning.

We denote action as a i , where the index i means the order number of action in the plan (a i is the i-th action in the plan).

The state right after the action a i is denoted S i , S 0 is the initial state.

We denote the set of actions to which a task T decomposes as act(T ).

If U is a set of tasks, we define act(U ) = ∪ T ∈U act(T ).

The index of the first action in the decomposition of T is denoted start(T ), that is, start(T ) = min{i|a i ∈ act(T )}.

Similarly, end(T ) means the index of the last action in the decomposition of T , that is, end(T ) = max{i|a i ∈ act(T )}.We can now define formally the constraints C used in the decomposition methods.

The constraints can be of the following three types:• t 1 ≺ t 2 : a precedence constraint meaning that in every plan the last action obtained from task t 1 is before the first action obtained from task t 2 , end(t 1 ) < start(t 2 ), • before(U, p): a precondition constraint meaning that in every plan the proposition p holds in the state right before the first action obtained from tasks U , p ∈ S start(U )−1 ,• between(U, V, p): a prevailing condition meaning that in every plan the proposition p holds in all the states between the last action obtained from tasks U and the first action obtained from tasks V , DISPLAYFORM1 The HTN plan verification problem is formulated as follows: given a sequence of actions a 1 , a 2 , . . .

, a n , is there an initial state S 0 such that the sequence of actions is a valid plan applicable to S 0 and obtained from some compound task?

Again, the initial state might also be given as an input in some formulations.

The HTN plan recognition problem is formulated as follows: given a sequence of actions a 1 , a 2 , . . .

, a n , are there an initial state S 0 and actions a n+1 , . . .

, a n+m for some m ≥ 0 such that the sequence of actions a 1 , a 2 , . . . , a n+m is a valid plan applicable to S 0 and obtained from some compound task?

In other words, the given actions form a prefix of some plan obtained from some compound task T .

We will be looking for such a task T minimizing the value m (the number of added actions to complete the plan).

If only the task T is of interest (not the actions a n+1 , . . .

, a n+m ) then we are talking about the task (goal) recognition problem.

The existing parsing-based HTN verification algorithm (Barták, Maillard, and Cardoso 2018) uses a complex structure of a timeline, that maintains the decomposition constraints so they can be checked when composing sub-tasks to a compound task.

We propose a simplified verification method, that does not require this complex structure, as it checks all the constraints directly in the input plan.

This makes the algorithm easier for implementation and presumably also faster.

The novel hierarchical plan verification algorithm is shown in Algorithm 1.

It first calculates all intermediate states (lines 2-8) by propagating information about propositions in action preconditions and effects.

At this stage, we actually solve the classical plan validation problem as the algorithm verifies that the given plan is causally consistent (action precondition is provided by previous actions or by the initial state).

The original verification algorithm did this calculation repeatedly each time it composed a compound task.

It is easy to show that every action is applicable, that is, B + ai ⊆ S i−1 (lines 2 and 4).

Next, we will show that DISPLAYFORM0 .

Right-to-left propagation (line 6) ensures that preconditions are propagated to earlier states if not provided by the action at a given position.

In other words, if there is a proposition p ∈ S i+1 \ A + ai+1 then this proposition should be at S i .

Line 6 adds such propositions to S i so it holds DISPLAYFORM1 then p would be deleted by the action a i+1 , which means that the plan is not valid.

The algorithm detects such situations (line 8).When the states are calculated, we apply a parsing algorithm to compose tasks.

Parsing starts with the set of primitive tasks (line 9), each corresponding to an action from the input plan.

For each task T , we keep a data structure describing the set act(T ), that is, the set of actions to which the task decomposes.

We use a Boolean vector I of the same size as the plan to describe this set; a i ∈ act(T ) ⇔ I(i) = 1.

To simplify checks of decomposition constraints, we also keep information about the index of first and last actions from act(T ).

Together, the task is represented using a quadruplet (T, s, e, I) in which T is a task, s is the index in the plan of the first action generated by T , e is the index in the plan of the last action generated by T (we say that [i, j] represents the interval of actions over which T spans), and I is a Boolean vector as described above.

The algorithm applies each decomposition rule to compose a new task from already known sub-tasks (line 12).

The composition consists of merging the sub-tasks, when we check that every action in the decomposition is obtained from a single sub-task (line 20) , that is, act(T 0 ) = k j=1 act(T j ) and ∀i = j : act(T i ) ∩ act(T j ) = ∅. We also check all the decomposition constraints; the code is direct Data: a plan P = (a 1 , ..., a n ) and a set of decomp.

methods Result: a Boolean equal to true if the plan can be derived from some compound task, false DISPLAYFORM2 A i is a primitive task corresponding to action a i , I i is a Boolean vector of size n, such that ∀i ∈ 1..n, DISPLAYFORM3 foreach decomposition method R of the form If all tests pass, the new task is added to a set of tasks (line 25).

Then we know that the task decomposes to actions, which form a sub-sequence (non necessarily continuous) of the plan to be verified.

The process is repeated until a task that decomposes to all actions is obtained (line 27) or no new task can be composed (line 10).

The algorithm is sound as the returned task decomposes to all actions in the input plan.

If the algorithm finishes with the value false then no other task can be derived.

As there is a finite number of possible tasks, the algorithm has to finish, so it is complete.

DISPLAYFORM4

Any plan verification algorithm, for example, the one from the previous section, can be extended to plan recognition by feeding the verification algorithm with actions a 1 , . . .

, a n+k , where we progressively increase k. The actions a 1 , . . .

, a n are given as an input, while the actions a n+1 , . . .

, a n+k need to be generated (planned).

However, this generate-and-verify approach would be inefficient for larger k as it requires exploration of all valid sequences of actions with the prefix a 1 , . . .

, a n .

Assume that there could be 5 actions at the position n + 1 and 6 actions at the position n + 2.

Then the generate-and-verify approach needs to explore up to 30 plans (not every action at the position n + 2 could follow every action at the position n + 1) and for each plan the verification part starts from scratch as the plans are different.

This is where the verification algorithm from BID1 can be used as it does not require exactly one action at each position.

The algorithm stores actions (sub-tasks) independently and only when it combines them to form a new task, it generates the states between the actions and checks the constraints for them.

This resembles the idea of the Graphplan algorithm (Blum and Furst 1997).

There are also sets of candidate actions for each position in the plan and the planextraction stage of the algorithm selects some of them to form a causally valid plan.

We use compound tasks together with their decomposition constraints to select and combine the actions (we do not use parallel actions in the plan).

The algorithm from (Barták, Maillard, and Cardoso 2018) extended to solve the plan recognition problem is shown in Algorithm 2.

It starts with actions a 1 , . . .

, a n (line 2) and it finds all compound tasks that decompose to subsets of these actions (lines 4-30).

This inner while-loop is taken from (Barták, Maillard, and Cardoso 2018), we only syntactically modified it to highlight the similarity with the verification algorithm from the previous section.

If a task that decomposes to all current actions is found (line 30) then we are done.

This is the goal task that we looked for and its timeline describes the recognized plan.

Otherwise, we add all primitive tasks corresponding to possible actions at position n + 1 (line 33).

Note that these are not parallel actions, the algorithm needs to select exactly one of them for the plan.

Now, the parsing algorithm continues as it may compose new tasks that include one of those just added primitive tasks.

Notice that the algorithm uses all composed tasks from previous iterations in succeeding iterations so it does not start from scratch when new actions are added.

This process is repeated until the goal task is found.

The algorithm is clearly sound as the task found is the task that decomposes to the shortest plan with a given prefix.

This goes from the soundness and completeness of the verification algorithm (in particular, no task that decomposes to a shorter plan exists).

The algorithm is semi-complete as if there exists a plan with the length n + k and with a given prefix, the algorithm will eventually find it at the (k + 1)-th iteration.

If no plan with a given prefix exists then the algorithm will not stop.

However, recall that the plan recognition problem is undecidable BID2 so any plan recognition approach suffers from this deficiency.

Data: a plan P = (a 1 , ..., a n ), A i is a primitive task corresponding to action a i , and a set of decomposition methods Result: a Task that decomposes to a plan with prefix DISPLAYFORM0 DISPLAYFORM1 action a can be at position l; A is a primitive task for a} 34 goto 4Algorithm 2: Plan recognitionThe algorithm maintains a timeline for each compound task to verify all the constraints.

This is the major difference from the above verification algorithm that points to the original plan.

This timeline has been introduced in BID1 , where all technical details can be found.

We include a short description to make the paper selfcontained.

A timeline is an ordered sequence of slots, where each slot describes an action, its effects, and the state right before the action.

For task T , the actions in slots are exactly the actions from act(T ).

Both effects and states are modelled using two sets of propositions, Post + and Post − modeling positive and negative effects of the action and Pre + and Pre − modeling propositions that must and must not be the true in the state right before the action.

Two sets are used as the state is specified only partially and propositions are added to it during propagation so it is necessary to keep information about propositions that must not be true in the state.

The timeline always spans from the first to the last action of the task.

Due to interleaving of tasks (actions from one task might be located between the actions of another task in the plan), some slots of the task might be empty.

These empty slots describe "space" for actions of other tasks.

When we are merging sub-tasks (lines 12-22), we merge their timelines, slot by slot.

This is how the actions from sub-tasks are put together in a compound task.

Notice, specifically, that it is not allowed for two merged sub-tasks to have actions in the same slot (line 15).

This ensures that each action is generated by exactly one task.

Data: a set of slots, a set of bef ore constraints Result: an updated set of slots 1 Function APPLYPRE(slots, pre) Propositions from bef ore and between constraints are "stored" in the corresponding slots (Algorithms 3 and 4) and their consistency is checked each time the slots are modified (line 26 of Algorithm 2).

Consistency means that no proposition is true and false at the same state.

Information between subsequent slots is propagated similarly to the verification algorithm (see Algorithm 5).

Positive and negative propositions are now propagated separately taking in account empty slots.

If there is no action in the slot then effects are unknown and hence propositions cannot be propagated.

DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 Algorithm 5: Propagate

A unique property of the proposed techniques is handling task interleaving -actions generated from different tasks may interleave to form a plan.

This is the property that parsing techniques based on CF grammars cannot handle.

Example in FIG4 demonstrates how the timelines are filled by actions as the tasks are being derived/composed from the plan.

Assume, first, that a complete plan consisting of actions a 1 , a 2 , . . .

, a 7 is given.

The plan recognition algorithm can also handle such situations, when a complete plan is given, so it can serve for plan verification too (the verification variant of the Algorithm 2 should stop with a failure at line 33 as no action can be added during plan verification).

In the first iteration, the algorithm will compose tasks T 2 , T 3 , T 4 as these tasks decompose to actions directly.

Notice, how the timelines with empty slots are constructed.

We know where the empty slots are located as we know the exact location of actions in the plan.

In the second iteration, only the task T 1 is composed from already known tasks T 3 and T 4 .

Notice how the slots from these tasks are copied to the slots of a new timeline for T 1 .

By contrary, the slots in original tasks remain untouched as these tasks may merge with other tasks to form alternative decomposition trees (see the discussion below).

Finally, in the third iteration, tasks T 1 and T 2 are merged to a new task T 0 and the algorithm stops there as a complete timeline that spans the plan fully is obtained (condition at line 30 of Algorithm 2 is satisfied).Let us assume that there is a constraint between({a 1 }, {a 3 }, p) in the decomposition method for T 3 .

This constraint may model a causal link between a 1 and a 3 .

When composing the task T 3 , the second slot of its timeline remains empty, but the proposition p is placed there (see Algorithm 4).

This proposition is then copied to the timeline of task T 1 , when merging the timelines (line 17 of Algorithm 2), and finally also to the timeline of task T 0 .

During each merge operation, the algorithm checks that p can still be in the slot, in particular, that p is not required to be false at the same slot (line 26 repeatedly checks the constraints from methods.

The new plan verification algorithm (Algorithm 1) handles the method constraints more efficiently as it uses the complete plan with states to check them.

Moreover, the propagation of states is run just once in Algorithm 1 (lines 2-8), while Algorithm 2 runs it repeatedly each time the task is composed from subtasks.

Hence, each constraint is verified just once in Algorithm 1, when a new task is composed.

In particular, the constraint between({a 1 }, {a 3 }, p) is verified with respect to the states when task T 3 is introduced.

Otherwise, both Algorithm 1 and Algorithm 2 derive the tasks in the same order (if the decomposition methods are explored in the same order).

Instead of timelines, Algorithm 1 uses the Boolean vector I to identify actions belonging to each task.

For example, for task T 3 the vector is [1, 0, 1, 0, 1, 0, 0] and for task T 4 it is [0, 0, 0, 1, 0, 1, 0] .

When composing task T 1 from T 3 and T 4 the vectors are merged to get [1, 0, 1, 1, 1, 1, 0] (see the loop at line 17).

Notice that the vector always spans the whole plan, while the timelines start at the first action and finish with the last action of the task (and hence the same timeline can be used for different plan lengths).Assume now that only plan prefix consisting of a 1 , a 2 , . . .

, a 6 is given.

The plan recognition algorithm (Algorithm 2) will first derive tasks T 3 and T 4 only.

Specifically, task T 2 cannot be derived yet as action a 7 is not in the plan.

In the second iteration, the algorithm will derive task T 1 by merging tasks T 3 and T 4 , exactly as we described above.

As no more tasks can be derived, the inner loop finishes and the algorithm attempts to add actions that can follow the prefix a 1 , a 2 , . . .

, a 6 (line 33).

Let action a 7 be added at the 7-th position in the plan; actually all actions, that can follow the prefix, will be added as separate primitive tasks at position 7.

Now the inner loop is restarted and task T 2 will be added in its first iteration.

In the next iteration, task T 0 will be added and this will be the final task as it satisfies the condition at line 30.Assume, hypothetically, that the verification Algorithm 1 is used there.

When it is applied to plan a 1 , a 2 , . . .

, a 6 , the algorithm derives tasks T 1 , T 3 , T 4 and fails as no task spans the whole plan and no more tasks can be derived.

After adding action a 7 , the algorithm will start from scratch as the states might be different due to propagating some propositions from the precondition of a 7 .

Hence, the algorithm needs to derive the tasks T 1 , T 3 , T 4 again and it will also add tasks T 0 , T 2 and then it will finish with success.

It may happen, that action a 5 can also be consistently placed to position 7.

Then, we can derive two versions of task Let us denote the second version as T 1 .

The Algorithm 1 will stop there as no more tasks can be derived.

Notice that tasks T 1 , T 3 , T 4 were derived repeatedly.

If we try a 5 earlier than a 7 at position 7 then tasks T 1 , T 3 , T 4 will actually be generated three times before the algorithm finds a complete plan.

Contrary, Algorithm 2 will add actions a 5 and a 7 together as two possible primitive tasks at position 7.

It will use tasks T 1 , T 3 , T 4 from the previous iteration, it will add tasks T 1 , T 3 as they can be composed from the primitive tasks (using the last a 5 ), it will also add tasks T 0 , T 2 (using the last a 7 ), and will finish with success.

Notice that T 1 cannot be merged with T 2 to get a new T 0 as T 1 has action a 5 at the 7-th slot while T 2 has a 7 there so the timelines cannot be merged (line 15 of Algorithm 2).

In the paper, we proposed two versions of the parsing technique for verification of HTN plans and for recognition of HTN plans.

As far as we know, these are the only approaches that currently cover HTN fully including all decomposition constraints.

Both versions can be applied to solve both verification and recognition problems, but as we demonstrated using an example, each of them has some deficiencies when applied to the other problem.

The next obvious step is implementation and empirical evaluation of both techniques.

There is no doubt that the novel verification algorithm is faster than the previous approaches BID2 and BID1 .

The open question is how much faster it will be, in particular for large plans.

The efficiency of the novel plan recognition technique in comparison to existing compilation technique BID5 ) is less clear as both techniques use different approaches, bottom-up vs. top-down.

The disadvantage of the compilation technique is that it needs to re-generate the known plan prefix, but it can exploit heuristics to remove some overhead there.

Contrary, the parsing techniques looks more like generate-and-test, but controlled by the hierarchical structure.

It also guarantees finding the shortest extension of plan prefix.

<|TLDR|>

@highlight

The paper describes methods to verify and recognize HTN plans by parsing of attribute grammars.