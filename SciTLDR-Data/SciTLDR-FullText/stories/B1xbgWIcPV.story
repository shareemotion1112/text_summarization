Most approaches to learning action planning models heavily rely on a significantly large volume of training samples or plan observations.

In this paper, we adopt a different approach based on deductive learning from domain-specific knowledge, specifically from logic formulae that specify constraints about the possible states of a given domain.

The minimal input observability required by our approach is a single example composed of a full initial state and a partial goal state.

We will show that exploiting specific domain knowledge enables to constrain the space of possible action models as well as to complete partial observations, both of which turn out helpful to learn good-quality action models.

The learning of action models in planning has been typically addressed with inductive learning data-intensive approaches.

From the pioneer learning system ARMS BID13 ) to more recent ones BID8 Zhuo and Kambhampati 2013; Kucera and Barták 2018) , all of them require thousands of plan observations or training samples, i.e., sequences of actions as evidence of the execution of an observed agent, to obtain and validate an action model.

These approaches return the statistically significant model that best explains the plan observations by minimizing some error metric.

A model explains an observation if a plan containing the observed actions is computable with the model and the states induced by this plan also include the possibly partially observed states.

The limitation of posing model learning and validation as optimization tasks over a set of observations is that it neither guarantees completeness (the model may not explain all the observations) nor correctness (the states induced by the execution of the plan generated with the model may contain contradictory information).Differently, other approaches rely on symbolic-via learning.

The Simultaneous Learning and Filtering (SLAF) approach BID2 exploits logical inference and builds a complete explanation through a CNF formula that represents the initial belief state, and a plan observation.

The formula is updated with every action and state of Copyright c 2019, Association for the Advancement of Artificial Intelligence (www.aaai.org).

All rights reserved.

the observation, thus representing all possible transition relations consistent with it.

SLAF extracts all satisfying models of the learned formula with a SAT solver although the algorithm cannot effectively learn the preconditions of actions.

A more recent approach addresses the learning of action models from plan observations as a planning task which searches the space of all possible action models BID0 .

A plan here is conceived as a series of steps that determine the preconditions and effects of the action models plus other steps that validate the formed actions in the observations.

The advantage of this approach is that it only requires input samples of about a total of 50 actions.

This paper studies the impact of using mixed input data, i.e, automatically-collected plan observations and humanencoded domain-specific knowledge, in the learning of action models.

Particularly, we aim to stress the extreme case of having a single observation sample and answer the question to whether the lack of training samples can be overcome with the supply of domain knowledge.

The question is motivated by (a) the assumption that obtaining enough training observations is often difficult and costly, if not impossible in some domains (Zhuo 2015); (b) the fact that although the physics of the real-world domain being modeled are unknown, the user may know certain pieces of knowledge about the domain; and (c) the desire for correct action models that are usable beyond their fitness to a set of testing observations.

To this end, we opted for checking our hypothesis in the framework proposed in BID0 since this planning-based satisfiability approach allows us to configure additional constraints in the compilation scheme, it is able to work under a minimal set of observations and uses an off-the-shelf planner 1 .

Ultimately, we aim to compare the informational power of domain observations (information quantity) with the representational power of domainspecific knowledge (information quality).

Complementarily, we restrict our attention to solely observations over fluents as in many applications the actual actions of an agent may not be observable BID11 .Next section summarizes basic planning concepts and outlines the baseline learning approach BID0 ).

Then we formalize our one-shot learning task with domain knowledge and subsequently we explain the task-solving process.

Section 5 presents the experimental evaluation and last section concludes.

We denote as F (fluents) the set of propositional state variables.

A partial assignment of values to fluents is represented by L (literals).

We adopt the open world assumption (what is not known to be true in a state is unknown) to implicitly represent the unobserved literals of a state.

Hence, a state s includes positive literals (f ) and negative literals (¬f ) and it is defined as a full assignment of values to fluents; |s| = |F |.

We use L(F ) to denote the set of all literal sets on F ; i.e. all partial assignments of values to fluents.

A planning action a has a precondition list pre(a) ∈ L(F ) and a effect list eff(a) ∈ L(F ).

The semantics of an action a is specified with two functions: ρ(s, a) denotes whether a is applicable in a state s and θ(s, a) denotes the successor state that results from applying a in a state s.

Then, ρ(s, a) holds iff pre(a) ⊆ s, i.e. if its preconditions hold in s. The result of executing an applicable action a in a state s is a new state θ(s, a) = {s \ ¬eff(a) ∪ eff(a)}, where ¬eff(a) is the complement of eff(a), which is subtracted from s so as to ensure that θ(s, a) remains a well-defined state.

The subset of effects of an action a that assign a positive value to a fluent is called positive effects and denoted by eff DISPLAYFORM0 while eff − (a) ∈ eff(a) denotes the negative effects.

A planning problem is a tuple P = F, A, I, G , where I is the initial state and G ∈ L(F ) is the set of goal conditions over the state variables.

A plan π is an action sequence π = a 1 , . . .

, a n , with |π| = n denoting its plan length.

The execution of π in I induces a trajectory s 0 , a 1 , s 1 , . . .

, a n , s n such that s 0 = I and, for each 1 ≤ i ≤ n, it holds ρ(s i−1 , a i ) and s i = θ(s i−1 , a i ).

A plan π solves P iff the induced trajectory reaches a final state s n such that G ⊆ s n .The baseline learning approach our proposal draws upon uses actions with conditional effects BID0 ).

The conditional effects of an action a c is composed of two sets of literals: C ∈ L(F ), the condition, and E ∈ L(F ), the effect.

The triggered effects resulting from the action application (conditional effects whose conditions hold in s) is defined as eff c (s, a) = C£E∈cond(ac),C⊆s E.

The approach for learning STRIPS action models presented in BID0 , which we will use as our baseline learning system (hereafter BLS, for short), is a compilation scheme that transforms the problem of learning the preconditions and effects of action models into a planning task P .

A STRIPS action model ξ is defined as ξ = name(ξ), pars(ξ), pre(ξ), add(ξ), del(ξ) , where name(ξ) and parameters, pars(ξ), define the header of ξ; and pre(ξ), del(ξ) and add(ξ)) are sets of fluents that represent the preconditions, negative effects and positive effects, respectively, of the actions induced from the action model ξ.

The BLS receives as input an empty domain model, which only contains the headers of the action models, and a set of observations of plan executions, and creates a propositional encoding of the planning task P .

Let Ψ be the set of predicates 2 that shape the variables F .

The set of propositions of P that can appear in pre(ξ), del(ξ) and add(ξ) of a given ξ, denoted as I ξ,Ψ , are FOL interpretations of Ψ over the parameters pars(ξ).

For instance, in a four-operator blocksworld BID10 , the I ξ,Ψ set contains five elements for the pickup(v1) model, I pickup,Ψ ={handempty, holding(v1),clear(v1),ontable(v1), on(v1, v1)} and eleven elements for the model of stack(v1,v2), I stack,Ψ ={handempty, holding(v1), holding(v2), clear(v1),clear(v2),ontable(v1),ontable(v2), on(v1, v1),on(v1, v2), on(v2, v1), on(v2, v2)}. Hence, solving P consists in determining which elements of I ξ,Ψ will shape the preconditions, positive and negative effects of each action model ξ.

The decision as to whether or not an element of I ξ,Ψ will be part of pre(ξ), del(ξ) or add(ξ) is given by the plan that solves P .

Specifically, two different sets of actions are included in the definition of P : insert actions, which insert preconditions and effects on an action model; and apply actions, which validate the application of the learned action models in the input observations.

Roughly speaking, in the blocksworld domain, the insert actions of a plan that solves P will look like (insert pre stack holding v1), (insert eff stack clear v1), (insert eff stack clear v2), where the second action denotes a positive effect and the third one a negative effect both to be inserted in the model of stack; and the second set of actions of the plan that solves P will be like (apply unstack blockB blockA),(validate 1),(apply putdown blockB),(validate 2), where the validate actions denote the points at which the states generated through the apply actions must be validated with the observations of plan executions.

In a nutshell, the output of the BLS compilation is a plan that completes the empty input domain model by specifying the preconditions and effects of each action model such that the validation of the completed model over the input observations is successful.

The one-shot learning task to learn action models from domain-specific knowledge is defined as a tuple Λ = M, O, Φ , where:• M is the initial empty model that contains only the header of each action model to be learned.• O is a single learning example or plan observation; i.e. a sequence of (partially) observable states representing the evidence of the execution of an observed agent.• Φ is a set of logic formulae that define domain-specific knowledge.

Combination Meaning ¬pre ξ e ∧ ¬ef f ξ e e belongs neither to the preconditions nor effects of ξ (e / ∈ pre(ξ) ∧ e / ∈ add(ξ) ∧ e / ∈ del(ξ)) pre ξ e ∧ ¬ef f ξ e e is only a precondition of ξ (e ∈ pre(ξ) ∧ e / ∈ add(ξ) ∧ e / ∈ del(ξ)) ¬pre ξ e ∧ ef f ξ e e is a positive effect of ξ (e / ∈ pre(ξ) ∧ e ∈ add(ξ) ∧ e / ∈ del(ξ)) pre ξ e ∧ ef f ξ e e is a negative effect of ξ (e ∈ pre(ξ) ∧ e / ∈ add(ξ) ∧ e ∈ add(ξ))

We analyze here the solution space of a learning task Λ = M, O, Φ ; i.e., the space of STRIPS action models.

In principle, for a given action model ξ, any element of I ξ,Ψ can potentially appear in pre(ξ), del(ξ) and add(ξ).

In practice, the actual space of possible STRIPS schemata is bounded by:1.

Syntactic constraints.

The solution M must be consistent with the STRIPS constraints: del(ξ) ⊆ pre(ξ), del(ξ)∩add(ξ) = ∅ and pre(ξ)∩add(ξ) = ∅. Typing constraints would also be a type of syntactic constraint (McDermott et al. 1998).2.

Observation constraints.

The solution M must be consistent with these semantic constraints derived from the learning samples O, which in our case is a single plan observation.

Specifically, the states induced by the plan computable with M must comprise the observed states of the sample, which further constrains the space of possible action models.

Considering only the syntactic constraints, the size of the space of possible STRIPS models is given by 2 2×|I Ψ,ξ | because one element in I ξ,Ψ can appear both in the preconditions and effects of ξ.

In this work, the belonging of an e ∈ I Ψ,ξ to the preconditions, positive effects or negative effects of ξ is handled with a refined propositional encoding that uses fluents of two types, pre ξ e and ef f ξ e, instead of the three fluents used in the BLS.

The four possible combinations of these two fluents are sumarized in FIG0 .

This compact encoding allows for a more effective exploitation of the syntactic constraints, and also yields the solution space of Λ to be the same as its search space.

Our approach is to introduce domain-specific knowledge in the form of state constraints to further restrict the space of the action models.

Back to the blocksworld domain, one can argue that on(v1,v1) and on(v2,v2) will not appear in the pre(ξ), del(ξ) and add(ξ) of any action model ξ because, in this specific domain, a block cannot be on top of itself.

The notion of state constraint is very general and has been used in different areas of AI and for different purposes.

In planning, state constraints are compact and abstract representations that relate the values of variables in each state traversed by a plan, and allow to specify the set of states where a given action is applicable, the set of states where a given axiom or derived predicate holds or the set of states that are considered goal states (Haslum et al. 2018) .

State invariants is a useful type of state constraints for computing more compact state representations of a given planning problem BID6 ) and for making satisfiability planning or backward search more efficient BID9 BID1 .

Given a planning problem P = F, A, I, G , a state invariant is a formula φ that holds in I, I |= φ, and in every state s built out of F that is reachable by applying actions of A in I.A mutex (mutually exclusive) is a state invariant that takes the form of a binary clause and indicates a pair of different properties that cannot be simultaneously true BID7 .

For instance in a three-block blocksworld, ¬on(block A , block B ) ∨ ¬on(block A , block C ) is a mutex because block A can only be on top of a single block.

Recently, some works point at extracting lifted invariants, also called schematic invariants BID9 , that hold for any possible state and any possible set of objects.

Invariant templates obtained by inspecting the lifted representation of the domain have also been exploited for deriving lifted mutex BID3 .

In this work we exploit domain-specific knowledge that is given as schematic mutex.

We pay special attention to schematic mutex because they identify mutually exclusive properties of a given type of objects (Fox and Long 1998) and because they enable (1) an effective completion of a partially observed state and (2) an effective pruning of inconsistent STRIPS action models.

We define a schematic mutex as a p, q pair where both p, q ∈ I ξ,Ψ are predicates that shape the preconditions or effects of a given action scheme ξ and they satisfy the formulae ¬p ∨ ¬q, considering that their corresponding variables are universally quantified.

For instance, holding(v 1 ) and clear(v 1 ) from the blocksworld are schematic mutex while clear(v 1 ) and ontable(v 1 ) are not because ∀v 1 , ¬clear(v 1 ) ∨ ¬ontable(v 1 ) does not hold for every possible state.

FIG2 shows an example of four clauses that define schematic mutexes for the blocksworld domain.∀x1, x2 ¬ontable(x1) ∨ ¬on(x1, x2).

∀x1, x2 ¬clear(x1) ∨ ¬on(x2, x1).

∀x1, x2, x3 ¬on(x1, x2) ∨ ¬on(x1, x3) such that x2 = x3.

∀x1, x2, x3 ¬on(x2, x1) ∨ ¬on(x3, x1) such that x2 = x3.

In this section, we show how to exploit schematic mutexes for solving the learning task Λ = M, O, Φ .

The addition of new literals to complete the partial states s o 1 . . .

, s o m of an observation O using a set of schematic mutexes Φ is done in a pre-processing stage.

Let Ω be the set of objects that appear in F as the values of the arguments of the predicates Ψ, and φ = p, q a schematic mutex.

There exist many possible instantiations of φ of the type p(ω), q(ω ) with objects of Ω, where ω ⊆ Ω |args(p)| and ω ⊆ Ω |args(q)| .

Let us now assume that the instantiation p(ω) ∈ s literal.

For instance, if the literal holding(blockA) is observed in a particular state and Φ contains the schematic mutex ¬holding(v 1 ) ∨ ¬clear(v 1 ), we extend the state observation with literal ¬clear(blockA) (despite this particular literal being initially unknown).

Our approach to learning action models consistent with the schematic mutexes in Φ is to ensure that newly generated states induced by the learned actions do not introduce any inconsistency.

This is implemented by adding new conditional effects to the insert and apply actions of the BLS compilation.

FIG3 summarizes the new conditional effects added to the compilation and next, we describe them in detail:1-3 For every schematic mutex p, q , where both p and q belong to I ξ,Ψ , one conditional effect is added to the (insert pre) ξ,p actions to prevent the insertion of two preconditions that are schematic mutex.

Likewise, two conditional effects are added to the (insert eff) ξ,p actions, one to prevent the insertion of two positive effects that are schematic mutex and another one to prevent two mutex negative effects.4-5 For every schematic mutex p, q , where both p and q belong to I ξ,Ψ , two conditional effects are added to the (apply) ξ,ω actions to prevent positive effects that are inconsistent with an input observation (in (apply) ξ,ω actions the variables in pars(ξ) are bounded to the objects in ω that appear in the same position).In theory, conditional effects of the type 4-5 are sufficient to guarantee that all the states traversed by a plan produced by the compilation are consistent with the input set of schematic mutexes Φ (obviously provided that the input initial state s o 0 is a valid state).

In practice we include also conditional effects of the type 1-3 because they prune invalid action models at an earlier stage of the planning process (these effects extend the insert actions that always appear first in the solution plans).Compilation properties Lemma 1.

Soundness.

Any classical plan π that solves P (planning task that results from the compilation) produces a model M that solves the Λ = M, O, Φ learning task.

Proof.

According to the P compilation, once a given precondition or effect is inserted into the domain model M it cannot be undone.

In addition, once an action model is applied it cannot be modified.

In the compiled planning task P , only (apply) ξ,ω actions can update the value of the state fluents F .

This means that a state consistent with an observation s o m can only be achieved executing an applicable sequence of (apply) ξ,ω actions that, starting in the corresponding initial state s o 0 , validates that every generated intermediate state sj (0 < j ≤ m), is consistent with the input state observations and state-invariants.

This is exactly the definition of the solution condition for model M to solve the Λ = M, O, Φ learning task.

Lemma 2.

Completeness.

Any model M that solves the Λ = M, O, Φ learning task can be computed with a classical plan π that solves P .Proof.

By definition I ξ,Ψ fully captures the set of elements that can appear in an action model ξ using predicates Ψ. In addition the P compilation does not discard any model M definable within I ξ,Ψ that satisfies the mutexes in Φ. This means that, for every model M that solves the Λ = M, O, Φ , we can build a plan π that solves P by selecting the appropriate (insert pre) ξ,e and (insert eff) ξ,e actions for programming the precondition and effects of the corresponding action models in M and then, selecting the corresponding (apply) ξ,ω actions that transform the initial state observation s The size of P depends on the arity of the predicates in Ψ, that shape variables F , and the number of parameters of the action models, |pars(ξ)|.

The larger these arities, the larger |I ξ,Ψ |.

The size of I ξ,Ψ is the most dominant factor of the compilation because it defines the pre ξ e/ef f ξ e fluents, the corresponding set of insert actions, and the number of conditional effects in the (apply) ξ,ω actions.

Note that typing can be used straightforward to constrain the FOL interpretations of Ψ over the parameters pars(ξ), which will significantly reduce |I ξ,Ψ | and hence the size of P output by the compilation.

Classical planners tend to prefer shorter solution plans, so our compilation (as well as the BLS) may introduce a bias to Λ = M, O, Φ learning tasks preferring solutions that are referred to action models with a shorter number of preconditions/effects.

In more detail, all {pre ξ e, ef f ξ e} ∀e∈I ξ,Ψ fluents are false at the initial state of our P compilation so classical planners tend to solve P with plans that require a smaller number of insert actions.

This bias can be eliminated defining a cost function for the actions in P (e.g. insert actions have zero cost while (apply) ξ,ω actions have a positive constant cost).

In practice we use a different approach to disregard the cost of insert actions since classical planners are not proficient at optimizing plan cost with zero-cost actions.

Instead, our approach is to use a SAT-based planner BID9 ) that can apply all actions for inserting preconditions in a single planning step (these actions do not interact).

Further, the actions for inserting action effects are also applied in another single planning step.

The plan horizon for programming any action model is then always bounded to 2.

The SAT-based planning approach is also convenient for its ability to deal with planning problems populated with dead-ends and because symmetries in the insertion of preconditions/effects into an action model do not affect the planning performance.

This section evaluates the improvement when using domainspecific knowledge for learning action models.

Reproducibility The domains used in the evaluation are IPC domains that satisfy the STRIPS requirement BID4 , taken from the PLANNING.DOMAINS repository (Muise 2016).

For each domain we generated 10 learning examples of length 10 via random walks and report average values (only a single example is considered at each learning episode).

We also introduce a new parameter, the degree of observability σ, which indicates de probability of observing a literal in an intermediate state.

This parameter is used to build observations with varying degrees of incompleteness.

All experiments are run on an Intel Core i5 3.10 GHz x 4 with 16 GB of RAM.For the sake of reproducibility, the compilation source code, evaluation scripts, used benchmarks and input state-invariants are fully available at the repository https://github.com/anonsub/oneshot-learning.Metrics The learned models are evaluated using the precision and recall metrics for action models proposed in BID0 , which compare the learned models against the reference model.

Precision measures the correctness of the learned models.

Formally, P recision = tp tp+f p , where tp is the number of true positives (predicates that appear in both the learned and reference action models) and f p is the number of false positives (predicates that appear in the learned action model but not in the reference model).

Recall, on the other hand, measures the completeness of the model and is formally defined as Recall = tp tp+f n where f n is the number of false negatives (predicates that should appear in the learned action model but are missing).

In our first experiment, we seek to answer the question as to whether the plan observation (single learning example) of O is replaceable by the domain knowledge encoded in Φ. To this end, we evaluate the following 4 settings:1.

Minimal observability: This is the baseline setting where we use the minimal expression of the single learning example; i.e., a fully observed initial state and a partially observed final state.

This setting is labeled as σ = 0 with no Φ. Table 1 shows the average values of precision (P) and recall (R) for each domain in the four tested settings.

The table also reports the number of schematic mutexes (|Φ|) used for each domain.

Comparing the settings only domain knowledge (setting 2) with only observability (setting 3), we can see that slightly better results are obtained with the latter, meaning that observability is more informative than the used domain knowledge.

On the other hand, the gain of using Φ under minimal observability (setting 1 compared to setting 2) is rather marginal.

While these results might indicate a general preference for observations over knowledge, when comparing setting 3 with setting 4, we can observe a significant improvement in the quality of the learned models.

This indicates that the payoff of using Φ is noticeable when the learning example has a certain degree of observability.

DISPLAYFORM0

The previous experiment reveals that observations are not totally replaceable by domain knowledge; but also shows that given a minimum degree of observability, using Φ enriches both the observations and the learning process and better models are learnable.

In this next experiment we measure the improvement provided by Φ at increasing degrees of observability of the learning example.

learned models with and without domain knowledge.

The points plotted in these figures are average values across all the domains presented in Table 1 .

The results show that using Φ significantly improves the learned models no matter how complete the learning examples are.

An interesting and revealing aspect from the figures is that the quality of the action models learned with 30%-observable learning examples and Φ is comparable to the quality obtained with a 100%-observable example.

Hence, domain knowledge can make up for the lack of completeness in the learning examples.

We present an approach to learn action models that builds upon a former compilation-to-planning learning system BID0 .

Our proposal studies the gains of using domain-specific knowledge when the availability (amount and observability) of learning examples is very limited.

Introducing domain knowledge encoded as schematic mutexes allows to narrow down the search space of the learning task and improve overall the performance of the learning system to the point that it offsets the lack of learning examples.

In a theoretical work that analyzes the relation between the number of observed trajectory plans and the guarantee for a learned action model to achieve the goal BID12 , authors conclude that the number of trajectories needed scales gracefully and the guarantee grows linearly with the number of predicates and quasi-linearly with the number of actions.

This evidences that learning accurate models is heavily dependent on the number and quality (observability) of the learning examples.

In this sense, our proposal comes to alleviate this dependency by relying on easily deducible domain knowledge.

It is not only capable of learning from a single non-fully observable learning example but also proves that learning from a 30%-observable example with domain-specific knowledge is comparable to learning from a complete plan observation.

@highlight

Hybrid approach to model acquisition that compensates a lack of available data with domain specific knowledge provided by experts

@highlight

A domain acquisition approach that considers using a different representation for the partial domain model by using schematic mutex relations in place of pre/post conditions.