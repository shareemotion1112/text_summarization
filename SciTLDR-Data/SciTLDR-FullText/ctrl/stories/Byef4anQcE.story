Providing transparency of AI planning systems is crucial for their success in practical applications.

In order to create a transparent system, a user must be able to query it for explanations about its outputs.

We argue that a key underlying principle for this is the use of causality within a planning model, and that argumentation frameworks provide an intuitive representation of such causality.

In this paper, we discuss how argumentation can aid in extracting causalities in plans and models, and how they can create explanations from them.

Explainability of AI decision-making is crucial for increasing trust in AI systems, efficiency in human-AI teaming, and enabling better implementation into real-world settings.

Explainable AI Planning (XAIP) is a field that involves explaining AI planning systems to a user.

Approaches to this problem include explaining planner decision-making processes as well as forming explanations from the models.

Past work on model-based explanations includes an iterative approach BID14 as well as using explanations for more intuitive communication with the user BID5 .

With respect to human-AI teaming, the more helpful and illustrative the explanations, the better the performance of the system overall.

Research into the types of questions and motivations a user might have includes work with contrastive questions BID9 .

These questions are structured as '

Why F rather than G?', where F is some part (i.e. action(s) in a plan) of the original solution and G is something the user imagines to be better.

While contrastive questions are useful, they do not consider the case when a user doesn't have something else in mind (i.e. G) or has a more general question about the model.

This includes the scenario in which the user's understanding of the model is incomplete or inaccurate.

Research in the area of model reconciliation attempts to address this knowledge gap BID1 .More broadly, questions such as '

Why A?', where A is an action in the plan, or 'How G?', where G is a (sub)goal, must be answerable and explainable.

Questions like these are inherently based upon definitions held in the domain related to a particular problem and solution.

The user's motivation behind such questions can vary: he could think the action is unnecessary, be unsure as to its effects, or think there is a better option.

Furthermore, questions regarding particular state information may arise, such as 'Why A here?' and 'Why can't A go here?'.

For these, explanations that include relevant state information would vastly improve their efficiency when communicating with a user BID9 .

This is especially true for long plans, when a user does not have access to a domain, or the domain is too complex to be easily understood.

Thus, extracting relevant information about action-state causality from the model is required.

In the space of planning, causality underpins a variety of research areas including determining plan complexity BID6 and heuristics BID7 .

Many planners also can create causal graph visualizations of plans for a user to interact with BID12 .

The general structure of causality in planning is 'action causes state'.

Indirectly, this can be seen as 'action enables action', where the intermediary state is sufficient for the second action to occur.

Hilton describes different 'causal chains' which mirror the types of causality found in planning; action-state causality can be identified as either a 'temporal' or 'unfolding' chain, while action-action causality is similar to an 'opportunity chain' BID8 .

For now, we will focus on these two types of general causality.

To represent the causality of a model, argumentation is a good candidate; as detailed by BID0 , argumentation frameworks and causal models can be viewed as two versions of one entity.

A recent related work uses argumentation for explainable scheduling (Cyras et al. 2019) .

We consider an ASPIC + (Modgil and Prakken 2013) style framework with defeasible rules capturing the relationships between actions in a plan and strict rules capturing actionstate causality.

This structure allows more than a causal representation of a plan; it allows multiple types of causality to be distinguished and different causal 'chunks' to be created and combined to be used as justification for explanations.

In this paper we present an initial approach for using argumentation to represent causality, which can then be used to form more robust explanations.

In the following sections, a motivating scenario will be introduced and used to showcase our current approaches of abstracting causalities and state information into argumentation frameworks.

Consider a simple logistics scenario in which three trucks are tasked with delivering three packages to different locations.

The user analyzing the planner output has the plan as well as a general, non-technical understanding of the model and the goals of the problem; the user knows that trucks can move between certain waypoints that have connecting roads of differing lengths, there are refueling stations at waypoints B and E, and some subgoals of the problem are to have package 1 delivered to waypoint C, package 2 delivered to waypoint G, and package 3 delivered to waypoint D. The user is also aware that the three trucks and three packages are at waypoint A in the initial state.

A basic map of the domain and plan are shown in FIG1 , respectively.

Even with a simple and intuitive problem such as this, questions may arise which cannot be answered trivially.

One such question is 'Why drive truck 1 to waypoint E?'.

Addressing this question requires the causal consequence of applying the action; in other words, how does driving truck 1 to waypoint E help in achieving the goal(s)?As discussed previously, tracking state information throughout a plan can be useful for explanations.

This is especially true when values of state variables are not obvious at any given point in a plan and their relevance to a question is not known.

A question such as 'Why drive truck 3 to waypoint B?

' has this property.

These two questions will be addressed in the following sections.

As mentioned above, in this paper we will make use of ASPIC + as the underlying argumentation system from which explanations are constructed.

However, what we are suggesting is not limited to ASPIC + ; we can imagine using most formal argumentation systems to reason in this way.

For a full description of ASPIC + see BID10 .

In this paper we only make use of the ability to construct arguments, and so that is the only aspect of the system that we describe.

We start with a language L, closed under negation.

A reasoner is then equipped with a set Rules of strict rules, denoted ?? 1 , . . .

, ?? n ??? ??, and defeasible rules, denoted ?? 1 , . . .

, ?? n ??? ??, where ?? 1 , . . .

, ?? n , ?? are all elements of L. A knowledge base ??? is then a set of elements K from L and a set Rules.

From ??? it is possible to construct a set of arguments A(???), where an argument A is made up of some subset of K, along with a sequence of rules, that lead to a conclusion.

Given this, Prem(??) returns all the premises, Conc(??) returns the conclusion and TopRule(??) returns the last rule in the argument.

An argument A is then: DISPLAYFORM0 Sub(A) = {A}; and TopRule(A) = undefined.??? A 1 , . . .

, A n ??? ?? if A i , 1 ??? i ??? n, are arguments and there exists a strict rule of the form Conc(A 1 ), . . .

, DISPLAYFORM1 ??? A 1 , . . .

, A n ??? ?? if A i , 1 ??? i ??? n, are arguments and there exists a defeasible rule of the form Conc(A 1 ), . . .

, DISPLAYFORM2 Then, given K = {a; b} and Rules = {a ??? c; b, c ??? d}, we have the following arguments: DISPLAYFORM3 When applied to planning, these arguments define a subsection of a causal chain, as will be described below.

In order to utilize causality in explanations, the causal links between actions in a plan need to be extracted and abstracted into a framework.

This process is planner-independent, so it requires only the plan, problem, and domain as inputs.

An algorithm is used to extract the causalities which then form a knowledge base of causal links.

This can then be used by an argumentation engine to construct arguments representing the causal 'chunks' in a plan.

From this, questions of the forms '

Why A?' and 'How G?' can be addressed.

This process is described in the following sections.

To extract causal relationships between actions in a plan, an algorithm similar to the one used in BID2 for detecting action dependencies is utilized: 1.

Finds connections between one action's effects and another's preconditions from the domain to form a knowledge base.

In general terms we can think of these chunks as being statements in some logical language of the form: ((load truck t1 p1), (drive truck t1 wpC)) ??? (unload truck t1 p1) (drive truck t1 wpC) ??? (drive truck t1 wpD) (unload truck t1 p1) ??? p1 at wpC (drive truck t1 wpD) ??? (drive truck t1 wpE) DISPLAYFORM0

Given a knowledge base, the argumentation engine can construct a sequence of arguments with defeasible rules:A 1 :(load truck t1 p1) A 2 :(drive truck t1 wpC) A 3 :A 1 , A 2 ??? (unload truck t1 p1) A 4 :A 3 ??? p1 at wpC A 5 :A 2 ??? (drive truck t1 wpD) A 6 :A 5 ??? (drive truck t1 wpE) A 7 :A 6 ??? (ref uel truck t1) A 8 :A 7 ??? (drive truck t1 wpF ) A 9 :A 8 ??? (drive truck t1 wpG) A 10 :A 9 ??? (unload truck t1 p2) A 11 :A 10 ??? p2 at wpG These summarize the causal structure of part of the plan (i.e. a 'causal chunk' as defined in Secion 4.3), summarized in argument A 11 , which can then be presented to a user who is seeking explanations.

A visualization of these arguments can be seen in FIG4

We define the notion of a causal 'chunk' as any subsection(s) of the causal chain(s) extracted from the plan or model and then combined.

Intuitively, these chunks can focus on one 'topic' (e.g. state variable, object instance) to provide a higher-level abstraction of causality rather than just the individual causal links.

The argument A 11 which represents such a causal chunk shows only the action-action causalities (i.e. from just one causal chain) involving the object truck 1.

These chunks are created by searching through the Rules of the framework for those pertaining to a specific 'topic'.Given arguments such as A 11 , we propose two methods of structuring explanations.

The first method is allowing the user to engage the system in a dialogue.

For our example, the question, 'Why e? where e is the action of driving truck 1 to waypoint E could be used to query the system: why e Following work such as BID11 , the system replies to this query by building an argument for e, in this case A 6 , and using this to provide a suitable response, which might be by returning Conc(A 5 ), since A 5 ??? e.

Thus the system could reply with:d, which leads to e where d is drive truck t1 wpD. The user could then continue to unpack the causal chunk by asking: why d and so on.

This would provide the user with the causalities which enabled action e to be applied.

The same could be done using a forward approach where the argument A 6 is expanded until a subgoal is reached, if possible (e.g. A 11 ).

The user can then ask:why e and the system responds with:e leads to f as in A 7 : A 6 ??? f .

Iteratively, this would show how e leads to some goal or subgoal.

Reversing this process will also explain how a goal is reached.

The second method of structuring explanations is detailed in Section 5.2, and can be applied to this example similarly.

Using a similar method as above, causalities held within the state space of the plan are extracted and represented as a knowledge base.

An algorithm is used that iterates through the effects of actions from a plan and extracts the state variables they alter.

They can then be used to answer questions such as '

Why A here?

' and 'Why can't A go here?'.

In general terms, we define these dependencies as being statements in some logical language of the form: DISPLAYFORM0 which denote the statements 'a causes ???x a ' and 'b causes ???y c and ???z c '.

Here, a, b are actions in the plan, and x, y, z are state variables.

The x 0 , y 0 , z 0 denote the values of those variables in the initial state while x f , y f , z f denote the final values in the goal state; ???x a denotes the change in x after applying action a.

Applying this to our logistics example and the question, 'Why drive truck 3 to waypoint B?', these strict rules are relevant: DISPLAYFORM1 From these, it is clear the truck's fuel level is too low in the initial state to go anywhere besides waypoint B (see FIG1 .

However, it is not clear why the truck does not just stay put.

Alone, these rules do not provide a full explanation, but they can be added to the action-action causal chains for more complete explanations.

When used in conjunction, the causal traces and opportunity traces form a strong basis of justification for an explanation (see FIG5 for a visual representation).

Using the example from before, the relevant defeasible rules from the causal chain are: DISPLAYFORM0 where the conclusion of the second rule is a subgoal of the problem, perhaps previously unknown to the user.

That is, because the problem requires all trucks to have a minimum amount of fuel at the end, truck 3 had to refuel but could not deliver any packages due to its low initial fuel amount.

Thus, combining arguments from both types of causal chains more aptly answers this question.

A method for seamlessly creating explanations from this structure is an intended future work.

For now, it is possible to extract both the defeasible rules and strict rules governing the causal effects related to a specific topic and present them to a user.

How to determine which rules are relevant to a specific user question and how to combine the rules to form higher-level causal chunks are ongoing works.

One possible method of creating relevant causal chunks is to extract all rules related to a specific 'topic' (e.g. state variable).

For the variable 't3 fuel', all actions which alter it will be extracted along with any actions that enable the altering actions from the defeasible rules.

Additionally, any (sub)goals containing 't3 fuel' will be extracted.

Together, these form a chunk representing the causes of changes to 't3 fuel' as well as its relationship to the (sub)goals.

The arguments below represent the causal 'chunk': DISPLAYFORM1 where the conclusion of A 3 is a subgoal of the problem.

When unpacked iteratively, the arguments in the causal chunk centred on 't3 fuel' would give a similar output explanation as in the example in Section 4.3.

For example, a user asking the question '

Why b?' where b is the action (drive truck 3 to waypoint B) would either receive the response: t3 fuel is 2 enables b or the response:b causes t3 fuel decrease 2 and enables c if using a forward chaining approach, where c is the premise of the conclusion of A 2 , (refuel truck t3).

This process would continue until the subgoal t3 fuel >5 is reached.

However, identifying what state variables are relevant given a user question is not trivial.

The question 'Why drive truck 3 to waypoint B?' has no mention of the truck's fuel, so its relevance must be deduced from the plan, problem and domain.

Another method of providing explanations is through a graph structure, as depicted in Figure 5 .

Given a query, the relevant causal chunks would be identified and represented in the graph with individual actions and state changes as nodes and the causal rules between them as edges.

This approach could also help explain question of the form, Why can't A go here?, as inapplicable actions (ones not in the plan) can be shown.

Developing a robust system such as this is important future work.

Figure 5: An example graph with the queried action in blue and nodes contained in the 't3 fuel' chunk in orange, and I and G the initial and goal states.

Dashed edges denote defeasible rules; solid edges denote strict rules.

We acknowledge that this is a preliminary step and more work is required to expand on the ideas presented in this paper.

One such future work involves defining exactly what questions, which range from action-specific to model-based, can be answered and explained using our approach.

Also, how these questions are captured from a user is an open question.

The query, 'Why didn't truck 3 deliver any packages?

' can be answered using the causal information captured in the framework, but how one converts this question to a form that the system understands requires further research.

Potential methods for communicating a user question include a dialogue system or Natural Language Processing techniques.

Along with expanding the set of questions that can be addressed, extensions to the argumentation framework itself should be considered.

Better methods for creating causal 'chunks' for specific user questions are needed.

It may be advantageous to use argumentation schemes to help identify relevant topics of chunks and which causal chains should be included from the framework.

This relates to the idea of 'context' and identifying the motivation of a question.

If the system can be more precise in extracting the relevant information, the explanations themselves will be more effective.

Related to this is the need to explore other ways of presenting an explanation to a user.

Research into the efficacy of explanations and how to properly assess the effectiveness of the explanations in practice are future areas of research, and will require user studies.

Our starting point will be the approach outlined in Section 4.3 which has been shown empirically to be effective in contexts such as human-robot teaming BID13 .

In this paper we proposed an initial approach to explainable planning using argumentation in which causal chains are extracted from a plan and model and abstracted into an argumentation framework.

Our hypothesis is that this allows ease of forming and communicating explanations to a user.

Furthermore, causal 'chunks' can be created by combining relevant causal links from the chains which explain the causalities surrounding one 'topic'.

We believe these help with making more precise explanations, and that chunks can be used to provide hierarchical explanations.

Overall, the approach is a first step towards exploiting the intuitive functionality of argumentation in order to use causality for explanations.

<|TLDR|>

@highlight

Argumentation frameworks are used to represent causality of plans/models to be utilized for explanations.