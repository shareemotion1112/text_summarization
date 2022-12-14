In an explanation generation problem, an agent needs to identify and explain the reasons for its decisions to another agent.

Existing work in this area is mostly confined to planning-based systems that use automated planning approaches to solve the problem.

In this paper, we approach this problem from a new perspective, where we propose a general logic-based framework for explanation generation.

In particular, given a knowledge base $KB_1$ that entails a formula $\phi$ and a second knowledge base $KB_2$ that does not entail $\phi$, we seek to identify an explanation $\epsilon$ that is a subset of $KB_1$ such that the union of $KB_2$ and $\epsilon$ entails $\phi$. We define two types of explanations, model- and proof-theoretic explanations, and use cost functions to reflect preferences between explanations.

Further, we present our algorithm implemented for propositional logic that compute such explanations and empirically evaluate it in random knowledge bases and a planning domain.

With increasing proliferation and integration of AI systems in our daily life, there is a surge of interest in explainable AI, which includes the development of AI systems whose actions can be easily understood by humans.

Driven by this goal, machine learning (ML) researchers have begun to classify commonly used ML algorithms according to different dimensions of explainability (Guidotti et al. 2018) ; improved the explainability of existing ML algorithms BID3 BID0 BID3 ; as well as proposed new ML algorithms that trade off accuracy for increasing explainability (Dong et al. 2017; BID1 .

1 While the term interpretability is more commonly used in the ML literature and can be used interchangeably with explainability, we use the latter term as it is more commonly used broadly across different subareas of AI.In contrast, researchers in the automated planning community have mostly taken a complementary approach.

While there is some work on adapting planning algorithms to find easily explainable plans 2 (i.e., plans that are easily understood and accepted by a human user) BID0 , most work has focused on the explanation generation problem (i.e., the problem of identifying explanations of plans found by planning agents that when presented to users, will allow them to understand and accept the proposed plan) (Langley 2016; Kambhampati 1990) .

Within this context, researchers have tackled the problem where the model of the human user may be (1) inconsistent with the model of the planning agent (Chakraborti et al. 2017b) ; (2) must be learned BID0 ; and (3) a different form or abstraction than that of the planning agent BID0 Tian et al. 2016) .

However, a common thread across most of these works is that they, not surprisingly, employ mostly automated planning approaches.

For example, they often assume that the models of both the agent and human are encoded in PDDL format.

In this paper, we approach the explanation generation problem from a different perspective -one based on knowledge representation and reasoning (KR).

We propose a general logic-based framework for explanation generation, where given a knowledge base KB 1 (of an agent) that entails a formula ?? and a knowledge base KB 2 (of a human user) that does not entail ??, the goal is to identify an explanation ??? KB 1 such that KB 2 ??? entails ??.

We define two types of explanations, model-and proof-theoretic explanations, and use cost functions to reflect preferences between explanations.

Further, we present an algorithm, implemented for propositional logic, that computes such explanations and evaluate its performance experimentally in random knowledge bases as well as in a planning domain.

In addition to providing an alternative approach to solve the same explanation generation problem tackled thus far by the automated planning community, our approach has the merit of being more generalizable to other problems beyond planning problems as long as they can be modeled using a logical KR language.

where KB L is the set of well-formed knowledge bases (or theories) of L -each being a set of formulae.

BS L is the set of possible belief sets; each element of BS L is a set of syntactic elements representing the beliefs L may adopt.

ACC L : KB L ??? 2 BS L describes the "semantics" of L by assigning to each element of KB L a set of acceptable sets of beliefs.

For each KB ??? KB L and B ??? ACC L (KB), we say that B is a model of DISPLAYFORM0 Example 1 Assume that L refers to the propositional logic over an alphabet P .

Then, KB L is the set of propositional theories over P , BS L = 2 P , and ACC L maps each theory KB into the set of its models in the usual sense.

DISPLAYFORM1 For our later use, we will assume that a negation operator ?? over formulas exists; and ?? and ???? are contradictory with each other in the sense that for any KB and B ??? ACC L (KB), if ?? ??? B then ???? ??? B; and if ???? ??? B then ?? ??? B. ??? KB is called a sub-theory of KB.

A theory KB subsumes a theory KB , denoted DISPLAYFORM2 Conclusions of a knowledge base can also be derived using rules.

A rule system ?? L of a logic L is a set of rules of the form DISPLAYFORM3 where ?? i are formulas.

The left hand side could be empty.

For a rule r of the form (1), body(r) (resp.

head(r)) denotes the left (resp.

right) side of r. Intuitively, a rule r states that if the body is true then the head is also true.

Given a knowledge base KB and a rule system ?? L , we say KB ?? L ?? if either ?? ??? KB or there exists a sequence of rules r 1 , . . .

, r n in ?? L such that body(r 1 ) ??? KB, head(r n ) = ??, head(r i ) ??? KB for i = 1, . . .

, n ??? 1, and body(r i ) ??? KB or body(r i ) ??? body(r 1 ) ??? {head(r j ) | j = 1, . . . , i ??? 1} for every i = 2, . . .

, n. We call the sequence = r 1 ; . . . ; r n as a proof from KB for ?? w.r.t.

?? L and say that the proof has the length n. DISPLAYFORM4

A classical planning problem (Russell and Norvig 2009) can be naturally encoded as an instance of propositional satisfiability BID2 .

The basic idea is the following: Given a planning problem P , find a solution for P of length n by creating a propositional formula that represent the initial state, goal, and the action dynamics for n time steps.

This is referred to as the bounded planning problem (P, n), and we define the formula for (P, n) such that: any model of the formula represents a solution to (P, n) and if (P, n) has a solution, then the formula is satisfiable.

We encode (P, n) as a formula ?? such that a 0 , a 1 , . . . , a n???1 is a solution for (P, n) if and only if ?? can be satisfied in a way that makes the fluents a 0 , a 1 , . . . , a n???1 true.

The formula ?? is a conjunction of the following formulae:??? Initial State: Let F be the set of possible facts in the planning problem: DISPLAYFORM0 ??? Goal: Let G be the set of goal facts:{f n |f ??? G}??? Action Scheme:

For every action a i at time step i: DISPLAYFORM1 ??? Explanatory Frame Axioms: Formulae describing what does not change between steps i and i + 1: DISPLAYFORM2 ??? Complete Exclusion Axioms: Only one action can occur at each time step: DISPLAYFORM3 Finally, we can extract a plan by finding an assignment of truth values that satisfies ?? (i.e., for i = 0, . . .

, n ??? 1, there will be exactly one action a such that a i = True).

This could be easily done by using a satisfiability algorithm, such as the well-known DPLL algorithm BID0 .In this paper, we will mostly use examples from propositional logic.

We make use of the fact that the resolution rule is sound and complete in first-order logic BID3 , and hence, in propositional logic.

This allows us to utilize the DPLL algorithm in computing proofs for a formula given a knowledge base.

In this section, we introduce the notion of an explanation in the following setting: Explanation Generation Problem: Given two knowledge bases KB 1 and KB 2 and a formula ?? in a logic L. Assume that KB 1 |= L ?? and KB 2 |= L ??.

The goal is to identify an explanation (i.e., a set of formulas) ??? KB 1 such that KB 2 ??? |= ??.We first define the notion of a support of a formula w.r.t.

a knowledge base.

DISPLAYFORM0 Assume that is a support of ?? w.r.t.

KB.

We say that ??? KB is a ???-minimal support of ?? if no proper subtheory of is a support of ??. Furthermore, is a ??-general support of ?? if there is no support of ?? w.r.t.

KB such that subsumes .We now define below two types of explanationsmodel-theoretic and proof-theoretic explanations.

Definition 2 (m-Explanation) Given two knowledge bases KB 1 and KB 2 in logic L and a formula ??. Assume that KB 1 |= L ?? and KB 2 |= L ??.A model-theoretic explanation (or m-explanation) for ?? from KB 1 for KB 2 is a support w.r.t.

DISPLAYFORM0 Example 2 Consider proposition logic theories over the set of propositions {a, b, c} with the usual definition of models, satisfaction, etc.

Assume KB 1 = {a, b, a ??? c, a ??? b ??? c} and KB 2 = {a}. We have that 1 = {a, a ??? c} and 2 = {a, b, a ??? b ??? c} are two ???-minimal supports of c w.r.t.

KB 1 .

Only 1 is a ??-general support of c w.r.t.

KB 1 since 2 ?? 1 .Both 1 and 2 can serve as m-explanations for c from KB 1 for KB 2 .

Of course, KB 1 is itself an mexplanation for c from KB 1 for KB 2 .Consider KB 3 = {a, ??b}. In this case, we have that only 1 is an m-explanation for c from KB 1 for KB 3 .

Now, consider KB 4 = {??a}. In this case, we have no m-explanation for c from KB 1 for KB 4 .Proposition 1 For two knowledge bases KB 1 and KB 2 in a monotonic logic L, if KB 1 |= L ?? and KB 2 |= L ????, then there exists no m-explanation for ?? from KB 1 for KB 2 .The KB 4 in Example 2 and Proposition 1 show that m-explanations alone might be insufficient.

Sometimes, we also need to persuade the other agent that its knowledge base is not correct.

We leave this for the future.

In this paper, we assume that KB 2 |= L ???? and KB 2 |= L ?? and, thus, an m-explanation always exists.

Definition 3 (p-explanation) Given a logic L with a sound and complete rule system ?? L and two knowledge bases KB 1 and KB 2 in logic L and a formula ??. Assume that KB 1 L ?? and KB 2 L ??.A proof-theoretic explanation (or p-explanation) for ?? from KB 1 for KB 2 is a proof r 1 ; . . . ; r n from KB 1 for ?? such that DISPLAYFORM0 Example 3 Consider the theories KB 1 = {a, b, a ??? c, a ??? b ??? c} and KB 2 = {a} from Example 2.

Let us assume that ?? L is the set of rules of the form l L l and l, ??l ??? p p for any literals l, p in the language of KB 1 and KB 2 .

Then, a, ??a ??? c L c is a proof from KB 1 for c, which is also a p-explanation for ?? from KB 1 for DISPLAYFORM1 Proposition 2 Assume that ?? L is a sound and complete rule system of a logic L, KB 1 is a knowledge base, and ?? is a formula in L. For each proof r 1 ; . . . ; r n from DISPLAYFORM2 Proposition 2 implies that each proof from KB 1 for ?? could be identified as a p-explanation for ?? from KB 1 if ?? L is sound and complete.

This provides the following relationship between m-explanations and pexplanations.

Proposition 3 Assume that ?? L is a sound and complete rule system of a logic L, KB 1 and KB 2 are two knowledge bases in L, and ?? is a formula in L. Then,??? for each m-explanation for ?? from KB 1 for KB 2 , there exists a p-explanation r 1 ; . . . ; r n for ?? from DISPLAYFORM3

Given KB 1 and KB 2 and a formula ??, there might be several (m-or p-) explanations for ?? from KB 1 for KB 2 .

For brevity, we will now use the term xexplanation for x ??? {m, p} to refer to an x-explanation for ?? from KB 1 for KB 2 .

Obviously, not all explanations are equal.

One might preferred a subset minimal m-explanation or a shortest length p-explanation over others.

We will next define a general preferred relation among explanations.

We assume a cost function C x L that maps pairs of knowledge bases and sets of explanations to nonnegative real values, i.e., C DISPLAYFORM0 where ??? is the set of x-explanations and R ???0 denotes the set of non-negative real numbers.

Intuitively, this function can be used to characterize different complexity measurements of an explanation.

A cost function C m L is monotonic if for any two m- DISPLAYFORM1 .

C x L induces a preference relation ??? KB over explanations as follows.

Definition 4 (Preferred Explanation) Given a cost function C x L , a knowledge base KB 2 , and two xexplanations 1 and 2 for KB 2 , we say 1 is preferred over 2 w.r.t.

KB 2 (denoted by 1 DISPLAYFORM2 (3) and 1 is strictly preferred over 2 w.r.t.

KB 2 (denoted DISPLAYFORM3

At a high level, Algorithms 1 and 2 can be used for computing most-preferred explanations given a formula ?? and two knowledge bases KB 1 and KB 2 of a logic L with the cost function C x L .

We assume that when computing for p-explanations, a sound and complete rule system is available.

Our algorithms rely on the existence of an algorithm for checking entailment between knowledge bases and formulas (Lines 1 and 3 in Algorithm 1 and Line 4 in Algorithm 2) and an algorithm for computing a potential explanation that is minimal with respect to a cost function and a knowledge base (Lines 2-3 in Algorithm 2).

These two algorithms depend on the logic L and the cost function C x L and need to be implemented for specific logic L and function C x L .

In the rest of this section, we discuss the implementation of our algorithms for propositional logic and different cost functions.

With propositional logic, it is easy to see that checking for entailment can be done by a SAT solver (e.g., MiniSat BID1 ).

We next discuss two algorithm implementations, one for mexplanations and one for p-explanations, that find an explanation that is minimal with respect to a cost function and a knowledge base.

The key data structures in the algorithm is a priority queue q, initialized to only include the empty set, of potential explanations ordered by their costs (Line 1) and a set checked of invalid explanations that have been considered thus far (line 2).

The algorithm repeatedly loops the following steps: (i) move the explanation with the smallest cost from the priority queue q to checked (Lines 4-5); (ii) check if it is a valid m-explanation and return if it is (Lines 6-7); (iii) if not, extend the explanation by 1 (with each clause from KB 1 ) and insert the extended explanations into the priority queue q (Lines 8-12).

If all potential explanations are exhausted, which means that there are no valid m-explanations, then the algorithm returns nil (Line 14).

It is straightforward to see that the following proposition holds.

Proposition 5 For two propositional theories KB 1 and KB 2 and a formula ??, Algorithm 3 returns a most preferred m-explanation w.r.t.

C m L for ?? from KB 1 to KB 2 if one exists.

Given a cost function C p L on p-explanations, Algorithm 4 computes a most-preferred p-explanation w.r.t.

C p L from KB 1 to KB 2 for ?? or returns nil if none exists.

We use the following notations in the pseudocode: For a proof , where is the sequence r 1 ; . . . ; r n , we write c( ) = head(r n ) and b( ) = n i=1 body(r i ).

We also write ??1,??2 ?? to indicate that ?? is the result of The algorithm uses the same two data structurespriority queue q and set checked -as in Algorithm 3.

The algorithm first populates the queue q with singlerule proofs consist of single clauses in KB 1 (Lines 2-4).

Then, it repeatedly loops the following steps: (i) move the proof with the smallest cost from the priority queue q to checked (Lines 8-9); (ii) check if it is a valid pexplanation and return if it is (Lines 10-11); (iii) if not, extend the proof by 1 and insert the extended proofs into the priority queue q FIG2 .

If all potential proofs are exhausted, which means that there are no valid pexplanations, then the algorithm returns nil (Line 19).

It is straightforward to see that the following proposition holds.

DISPLAYFORM0 Proposition 6 For two propositional theories KB 1 and KB 2 and a formula ??, Algorithm 4 returns a most preferred p-explanation w.r.t.

C p L for ?? from KB 1 to KB 2 if one exists.

As presented in the preliminaries, we can model a planning problem using the propositional logic language and thus utilize the proposed framework to generate explanations.

Particularly, we form the knowledge base of (a) Experimental Results on Random Knowledge Bases the agent, namely KB, by adding the encoded formula ?? (represented in CNF clauses) as well as the optimal plan of the specific planning problem.

Then, we define the explanation in terms of KB and plan optimality as follows: DISPLAYFORM0 DISPLAYFORM1 Definition 6 (Optimal Plan Explanation) Given a knowledge base KB and a plan ?? n = a 0 , a 1 , . . .

, a n???1 , we say that ?? n is optimal in KB if and only if KB |= ??, where ???t = 1, . . .

n ??? 1 : ?? = ??goal t .In other words, the formula ?? that we seek to explain is that no plan of lengths 1 to n???1 exists, and that a plan of length n exists.

Therefore, combined, that plan must be an optimal plan.

Now, given a second knowledge base KB 2 (i.e that of a human user), where KB 2 |= ??, we can compute a model-or proof-theoretic explanation as defined in Definitions 2 and 3.

We empirically evaluate our implementation of Algorithm 3 to find m-explanations on two synthetically generated benchmarks -random knowledge bases and a planning domain called BLOCKSWORLD -both encoded in propositional logic.

3 We evaluated our algorithm using the four cost functions described in Section .

Our algorithm was implemented in Python and experiments were performed on a machine with an Intel i7 2.6GHz processor and 16GB of RAM.

We report both the cost of the optimal m-explanation found as well as the runtime of the algorithm.

We first evaluated our algorithm on random knowledge bases with clauses in Horn form, where we varied the cardinality of KB 1 (the KB of the agent providing the explanation) from 20 to 1000.

To construct KB 2 (the KB of the agent receiving the explanation), we randomly chose 25% of the clauses from KB 1 .To construct each KB 1 , we first generated |KB1| 2 random symbols, which will be used in the KB.

Then, we iteratively generated clauses of increasing length l from 2 to 7.

For each length l, we generated |KB1| 2??l clauses using the symbols we previously generated such that each symbol is used at most once in these clauses of length l.

Each clause is a conjunction of l ??? 1 elements as the premise and the final l th element as the conclusion.

For example, a KB with a cardinality of 20, 10 symbols are first generated.

Then, 5 clauses of length 2, 3 clauses of length 3, 2 clauses of lengths 4 and 5, and 1 clause of lengths 6 and 7 are generated.

Finally, to complete the KB, we add all the symbols that are exclusively in the premise of the clauses generated as facts in the KB.

The formula ?? that we seek to explain is one of the randomly chosen conclusions in the clauses generated, which we ensure is entailed by KB 1 .

Table 1 (a) tabulates our results.

We make the following observations:??? As expected, the runtimes increase as |KB 1 | increases since the algorithm will need to search over a larger search space.??? As expected, the costs of explanations also increase as |KB 1 | increases since the explanations are presumably longer and more complex.

The reason is the computation of the costs of possible explanations is faster with the former two cost functions since they are not dependent on KB 2 while the computation for the latter two cost functions are dependent on KB 2 .

As we were motivated by the explanation generation problem studied in the automated planning community, we also conducted experiments on BLOCKSWORLD, a planning domain where multiple blocks must be stacked in a particular order on a table.

4 For these planning problems, we first used FAST-DOWNWARD BID2 to find optimal solutions to the planning problem.

Then, we translate the planning problem into a SAT problem with horizon h (Kautz et al. 1992), where h is the length of the optimal plan.

These CNF clauses then form our KB 1 (the KB of the agent providing the explanation).

Similar to random knowledge bases, we construct KB 2 (the KB of the agent receiving the explanation) by randomly choosing 25% of the clauses from KB 1 .

The formula ?? that we seek to explain is then that no plan of lengths 1 to h ??? 1 exists, and that a plan of length h (i.e., the plan found by FASTDOWNWARD) exists.

Therefore, combined, that plan must be an optimal plan.

Table 1(b) tabulates our results, where we observe similar trends as in the experiment on random knowledge bases.

The key difference is that the runtimes for all four cost functions here are a lot closer to each other, and the reason is because there was only one valid explanation in each problem instance.

Thus, regardless of the choice of cost function, that explanation had to be found.

Our experiments for larger problems are omitted as they timed out after 6 hours.

There is a very large body of work related to the very broad area of explainable AI.

We have briefly discussed some of them from the ML literature in Section .

We refer readers to surveys by BID0 and (Dosilovic et al. 2018 ) for more in-depth discussions of this area.

We focus below on related work from the KR and planning literature only since we employ KR techniques to solve explainable planning problems in this paper.

Related Work from the KR Literature: We note that the notion of an explanation proposed in this paper might appear similar to the notion of a diagnosis that has been studied extensively in the last several decades (e.g., (Reiter 1987)) as both aim at explaining something to an agent.

Diagnosis focuses on identifying the reason for the inconsistency of a theory whereas an mor p-explanation aims at identifying the support for a formula.

The difference lies in that a diagnosis is made with respect to the same theory and m-or p-explanation is sought for the second theory.

Another earlier research direction that is closely related to the proposed notion of explanation is that of developing explanation capabilities of knowledge-based systems and decision support systems, which resulted in different notions of explanation such as trace, strategic, deep, or reasoning explanations (see review by BID3 for a discussion of these notions).

All of these types of explanations focus on answering why certain rules in a knowledge base are used and how a conclusion is derived.

This is not our focus in this paper.

The present development differs from earlier proposals in that m-or p-explanations are identified with the aim of explaining a given formula to a second theory.

Furthermore, the notion of an optimal explanation with respect to the second theory is proposed.

There have been attempts to using argumentation for explanation (Cyras et al. 2017; Cyras et al. 2019) because of the close relation between argumentation and explanation.

For example, argumentation was used by (Cyras et al. 2019) to answer questions such as why a schedule does (does not) satisfy a criteria (e.g., feasibility, efficiency, etc.); the approach was to develop for each type of inquiry, an abstract argumentation framework (AF) that helps explain the situation by extracting the attacks (non-attacks) from the corresponding AF.

Our work differs from these works in that it is more general and does not focus on a specific question.

It is worth to pointing out that the problem of computing a most preferred explanation for ?? from KB 1 to KB 2 might look similar to the problem of computing a weakest sufficient condition of ?? on KB 1 under KB 2 as described by BID3 .

As it turns out, the two notions are quite different.

Given that KB 1 = {p, q} and KB 2 = {p}. It is easy to see that q is the unique explanation for q from KB 1 to KB 2 .

On the other hand, the weakest sufficient condition of q on KB 1 under KB 2 is ??? (Proposition 8, BID3 ).Related Work from the Planning Literature: In human-aware planning, the (planning) agent must have knowledge of the human model in order to be able to contemplate the goals of the humans as well as foresee how its plan will be perceived by them.

This is of the highest importance in the context of explainable planning since an explanation of a plan cannot be onesided (i.e., it must incorporate the human's beliefs of the planner).

In a plan generation process, a planner performs argumentation over a set of different models (Chakraborti et al. 2017a ); these models usually are the model of the agent incorporating the planner, the model of the human in the loop, the model the agent thinks the human has, the model the human thinks the agent has, and the agent's approximation of the latter.

Therefore, the necessity for plan explanations arises when the model of the agent and the model the human thinks the agent has diverge so that the optimal plans in the agent's model are inexplicable to the human.

During a collaborative activity, an explainable planning agent BID1 ) must be able to account for such model differences and maintain an explanatory dialogue with the human so that both of them agree on the same plan.

This forms the nucleus of explanation generation of an explainable planning agent, and is referred to as model reconciliation (Chakraborti et al. 2017b) .

In this approach, the agent computes the optimal plan in terms of his model and provides an explanation of that plan in terms of model differences.

Essentially, these explanations can be viewed as the agent's attempt to move the human's model to be in agreement with its own.

Further, for computing explanations using this approach the following four requirements are considered:??? Completeness -No better solution exists.

This is achieved by enforcing that the plan being explained is optimal in the updated human model.??? Conciseness -Explanations should be easily understandable to the human.??? Monotonicity -The remaining model differences cannot change the completeness of an explanation.??? Computability -Explanations should be easy to compute (from the agent's perspective).As our work is motivated by these ideas, we now identify some similarities and connections with our proposed approach.

First, it is easy to see that we implicitly enforce the first three requirements when computing an explanation -the notions of completeness and conciseness are captured through the use of our cost functions.

We do not claim to satisfy the computability requirement as it is more subjective and is more domain dependent.

In a nutshell, the model reconciliation approach works by providing a model update such that the optimal plan is feasible and optimal in the updated model of the human.

This is similar to our definition of the explanation generation problem where we want to identify an explanation ??? KB 1 (i.e., a set of formulae) such that KB 2 ??? |= ??.

In addition, the ???-minimal support in Definition 1 is equivalent to minimally complete explanations (MCEs) (the shortest explanation).

The -general support can be viewed as similar to the minimally monotonic explanations (MMEs) (the shortest explanation such that no further model updates invalidate it), with the only difference being that in the general support scenario, the explanations are such that all subsuming are also valid supports.

In contrast, model patch explanations (MPEs) (includes all the model updates) are trivial explanations and are equivalent to our definition that KB 1 itself serves as an m-explanation for KB 2 .

Note that, in our approach, we do not allow for explanations on "mistaken" expectations in the human model, as it can be inferred from Proposition 1 (monotonic language L).

From the model reconciliation perspective, such restriction is relaxed and allowed.

However, a similar property can be seen if the mental model is not known and, therefore, by taking an "empty" model as starting point explanations can only add to the human's understanding but not mend mistaken ones.

Explanation generation is an important problem within the larger explainable AI thrust.

Existing work on this problem has been done in the context of automated planning domains, where researchers have primarily employed, unsurprisingly, automated planning approaches.

In this paper, we approach the problem from the perspective of KR, where we propose a general logic-based framework for explanation generation.

We further define two types of explanations, model-and proof-theoretic explanations, and use cost functions to reflect preferences between explanations.

Our empirical results with algorithms implemented for propositional logic on both random knowledge bases as well as a planning domain demonstrate the generality of our approach beyond planning problems.

Future work includes investigating more complex scenarios, such as one where an agent needs to persuade another that its knowledge base is incorrect.

@highlight

A general framework for explanation generation using Logic.

@highlight

This paper researches explanation generation from a KR point of view and conducts experiments measuring explanation size and runtime on random formulas and formulas from a Blocksworld instance.

@highlight

This paper provides a perspective on explanations between two knowledge bases, and runs parallel to work on model reconciliation in planning literature.