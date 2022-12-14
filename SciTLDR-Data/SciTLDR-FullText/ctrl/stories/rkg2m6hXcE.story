Temporal logics are useful for describing dynamic system behavior, and have been successfully used as a language for goal definitions during task planning.

Prior works on inferring temporal logic specifications have focused on "summarizing" the input dataset -- i.e., finding specifications that are satisfied by all plan traces belonging to the given set.

In this paper, we examine the problem of inferring specifications that describe temporal differences between two sets of plan traces.

We formalize the concept of providing such contrastive explanations, then present a Bayesian probabilistic model for inferring contrastive explanations as linear temporal logic specifications.

We demonstrate the efficacy, scalability, and robustness of our model for inferring correct specifications across various benchmark planning domains and for a simulated air combat mission.

In a meeting where multiple plan options are under deliberation by a team, it would be helpful for that team's resolution process if someone could intuitively explain how the plans under consideration differ from one another.

Also, given a need to identify differences in execution behavior between distinct groups of users (e.g., a group of users who successfully completed a task using a particular system versus those who did not), explanations that identify distinguishing patterns between group behaviors can yield valuable analytics and insights toward iterative system refinement.

In this paper, we seek to generate explanations for how two sets of divergent plans differ.

We focus on generating such contrastive explanations by discovering specifications satisfied by one set of plans, but not the other.

Prior works on plan explanations include those related to plan recognition for inferring latent goals through observations BID25 BID35 , works on system diagnosis and excuse generation in order to explain plan failures BID29 BID10 , and those focused on synthesizing "explicable" plans -i.e., plans that are self-explanatory with respect to a human's mental model BID16 .

The aforementioned works, however, only involve the explanation or generation of a single plan; we instead focus on explaining differences between multiple plans, which can be helpful in various applications, such as the analysis of competing systems and compliance models, and detecting anomalous behaviour of users.

A specification language should be used in order to achieve clear and effective plan explanations.

Prior works have considered surface-level metrics such as plan cost and action (or causal link) similarity measures to describe plan differences BID23 BID3 .

In this work, we leverage linear temporal logic (LTL) BID24 which is an expressive language for capturing temporal relations of state variables.

We use a plan's individual satisfaction (or dissatisfaction) of LTL specifications to describe their differences.

LTL specifications have been widely used in both industrial systems and planning algorithms to compactly describe temporal properties BID32 .

They are human interpretable when expressed as compositions of predefined templates; inversely, they can be constructed from natural language descriptions BID7 ) and serve as natural patterns when encoding high-level human strategies for planning constraints BID14 .Although a suite of LTL miners have been developed for software engineering and verification purposes BID32 BID17 BID28 , they primarily focus on mining properties that summarize the overall behavior on a single set of plan traces.

Recently, BID22 presented SAT-based algorithms to construct a LTL specification that asserts contrast between two sets of traces.

The algorithms, however, are designed to output only a single explanation, and are susceptible to failure when the input contains imperfect traces.

Similar to Neider and Gavran, our problem focuses on mining contrastive explanations between two sets of traces, but we adopt a probabilistic approach -we present a Bayesian inference model that can generate multiple explanations while demonstrating robustness to noisy input.

The model also permits scalability when searching in large hypothesis spaces and allows for flexibility in incorporating various forms of prior knowledge and system designer preferences.

We demonstrate the efficacy of our model for extracting correct explanations on plan traces across various benchmark planning domains and for a simulated air combat mission.

Plan explanations are becoming increasingly important as automated planners and humans collaborate.

This first involves humans making sense of the planner's output (e.g., PDDL plans), where prior work has focused on developing user-friendly interfaces that provide graphical visualizations to describe the causal links and temporal relations of plan steps BID1 BID26 BID21 .

The outputs of these systems, however, require an expert for interpretation and do not provide a direct explanation as to why the planner made certain decisions to realize the outputted plan.

Automatic generation of explanations has been studied in goal recognition settings, where the objective is to infer the latent goal state that best explains the incomplete sequence of observations BID25 BID30 .

Works on explicable planning emphasize the generation of plans that are deemed selfexplanatory, defined in terms of optimizing plan costs for a human's mental model of the world BID16 .

Mixed-initiative planners iteratively revise their plan generation based on user input (e.g. action modifications), indirectly promoting an understanding of differences across newly generated plans through continual user interaction BID27 BID3 .

All aforementioned works deal with explainability with respect to a single planning problem specification, whereas our model deals with explaining differences in specifications governing two distinct sets of plans given as input.

Works on model reconciliation focus on producing explanations for planning models (i.e. predicates, preconditions and effects), instead of the realized plans .

Explanations are specified in the form of model updates, iteratively bringing an incomplete model to a more complete world model.

The term, "contrastive explanation," is used in these works to identify the relevant differences between the input pair of models.

Our work is similar in spirit but focuses on producing a specification of differences in the constraints satisfied among realized plans.

Our approach takes sets of observed plans as input rather than planning models.

While model updates are an important modality for providing plan explanations, there are certain limitations.

We note that an optimal plan generated with respect to a complete environment/world model is not always explicable or self-explanatory.

The space of optimal plans may be large, and the underlying preference or constraint that drives the generation of a particular plan may be difficult to pre-specify and incorporate within the planning model representation.

We focus on explanations stemming directly from the realized plans themselves.

Environment/world models (e.g. PDDL domain files) can be helpful in providing additional context, but are not necessary for our approach.

Our work leverages LTL as an explanation language.

Temporal patterns can offer greater expressivity and explanatory power in describing why a set of plans occurred and how they differ, and may reveal hidden plan dynamics that cannot be captured by the use of surface-level metrics like plan cost or action similarities.

Our work on using LTL for contrastive explanations directly contributes to exploring how we can answer the following roadmap questions for XAIP BID9 : "why did you do that?

why didn't you do something else (that I would have done)?"

Prior research into mining LTL specifications has focused on generating a "summary" explanation of the observed traces.

BID12 explored mining globally persistent specifications from demonstrated action traces for a finite state Markov decision process.

BID17 introduced Texada, a system for mining all possible instances of a given LTL template from an output log where each unique string is represented as a new proposition.

BID28 proposed a template-based probabilistic model to infer task specifications given a set of demonstrations.

However, all of these approaches focus on inferring a specification that all the demonstrated traces satisfy.

For contrastive explanations, Neider and Gavran (2018) presented SAT-based algorithms to infer a LTL specification that delineates between the positive and negative sets of traces.

Unlike existing LTL miners, the algorithms construct an arbitrary, minimal LTL specification without requiring predefined templates.

However, they are designed to output only a single specification, and can fail when the sets contain imperfect traces (i.e., if there exists no specification consistent with every single input trace.).

We present a probabilistic model for the same problem and generate multiple contrastive explanations while offering robustness to noisy input.

Some works have proposed algorithms to infer contrastive explanations for continuous valued time-series data based on restricted signal temporal logic (STL) grammar BID33 BID15 .

However, the continuous space semantics of STL and a restricted subset of temporal operators make the grammar unsuitable for use with planning domain problems.

To the best of our knowledge, our proposed model is the first probabilistic model to infer contrastive LTL specifications for sets of traces in domains defined by PDDL.

Linear Temporal Logic (LTL) provides an expressive grammar for describing temporal behavior BID24 ).

An LTL specification ?? is constructed from a set of propositions V , the standard Boolean operators, and a set of temporal operators.

Its truth value is determined with respect to a trace, ??, which is an infinite or finite sequence of truth assignments for all propositions in V .

The notation ??, t |= ?? indicates that ?? holds at time t.

The trace ?? satisfies ?? (denoted by ?? |= ??) iff ??, 0 |=

??. The minimal syntax for LTL can be described as follows: DISPLAYFORM0 where p is a proposition, and ?? 1 and ?? 2 are valid LTL specifications.

DISPLAYFORM1 Only one contiguous interval exists where pi is true.

DISPLAYFORM2 If pi occurs, pj occurred in the past.

Table 1 : An example set of LTL templates.

n T corresponds to the number of free propositions for each template.

X reads as "next" where X?? evaluates as true at t if ?? holds in the next time step t + 1.

U reads as "until" where ?? 1 U?? 2 evaluates as true at time step t if ?? 1 is true at that time and going forward, until a time step is reached where ?? 2 becomes true.

In addition to the minimal syntax, we also use higher-order temporal operators, F (eventually), G (global), and R (release).

F?? holds true at t if ?? holds for some time step ??? t. G?? holds true at t if ?? holds for all time steps ??? t. ?? 1 R?? 2 holds true at time step t if either there exists a time step t 1 ??? t such that ?? 2 holds true until t 1 where both ?? 1 and ?? 2 hold true simultaneously, or no such t 1 exists and ?? 2 holds true for all time steps ??? t.

Interpretable sets of LTL templates have been defined and successfully integrated for a variety of software verification systems BID32 BID20 .

Some of the widely used templates are shown in Table 1.

According to BID8 , a contrastive explanation describes "why event A occurred as opposed to some alternative event B." In our problem, events A and B represent two sets of plan traces (can be seen as traces generated from different systems or different group behavior).

The form of why may be expressed in various ways BID18 ); our choice is to define it according to the plans' satisfaction of a constraint.

Then, formally: Definition 3.1.

A contrastive explanation is a constraint ?? that it is satisfied by one set of plan traces (positive set, ?? A ), but not by the other (negative set, ?? B ).The constraint ?? can be seen as a classifier trying to separate the provided positive and negative traces.

Its performance measure corresponds to standard classification accuracy, computed by counting the number of traces in ?? A that satisfy ?? and, conversely, the number of traces in ?? B where ?? is unsatisfied.

Formally, accuracy of ?? is: DISPLAYFORM0 Accuracy is 1 for a perfect contrastive explanation, and approaches zero if both sets contains no valid trace with respect to ?? (i.e., all traces in ?? A dissatisfy ?? and all traces in ?? B satisfy ??).

The input to the problem is a pair of sets of traces (?? A , ?? B ).

Each ?? i ??? ?? is a trace on the set of propositions V (we refer to V as the vocabulary).

The output is a set of specifications, {??}, where each ?? achieves perfect or near-perfect contrastive explanation.

This is an unsupervised classification problem.

We use LTL specifications for the choice of ??. Planning is sequential, and so temporal patterns can offer greater expressivity and explanatory power for identifying plan differences rather than static facts.

We utilize a set of interpretable LTL templates, such as those shown in Table 1 .A LTL template T is instantiated with a selection of n T propositions denoted by p ??? V n T .

The candidate formula ?? is then composed as a conjunction of multiple instantiations of a template T based on a set of selections {p} ??? V n T .

For example, an instantiation of T ="stability" with p = [apple] is written as FG(apple).

If the selected subset of DISPLAYFORM0 , asserting the stability condition for all three propositions.

Conjunctions provide powerful semantics with the ability to capture a notion of quantification.

Formally, our LTL specification is written as follows: DISPLAYFORM1 Note that the number of free propositions, n T , varies per LTL template.

The number of possible specifications for a given LTL template T is 2 |V | n T .

Instead of extracting specifications narrowed down to a single template query, our hypothesis space ?? is set to include a number of predefined templates, T 1 , T 2 , ...T k .

With k representing the number of possible templates, the full hypothesis space of ?? grows with O(k ?? 2 |V | n T ).

Employing brute force enumeration to find {??} that achieves the contrastive explanation criterion rapidly becomes intractable with increasing vocabulary size.

We model specification learning as a Bayesian inference problem, building on the fundamental Bayes theorem: DISPLAYFORM0 Our goal is to infer ?? * = argmax ?? P (??|X).

P (??) represents the prior distribution over the hypothesis space, and Figure 1 : A graphical model of the generative distribution.

?? represents the latent LTL specification that we seek to infer given the evidence X (in our case, the traces).P (X | ??) is the likelihood of observing the evidence X = (?? A , ?? B )

given ??. We adopt a probabilistic generative modeling approach that has been used extensively in topic modeling BID2 .

Below, we describe each component of our generative model, depicted in Figure 1 .Prior Function ?? is generated by choosing a LTL template, T , the number of conjunctions, N , and then the proposition instantiations, p for each conjunct.

The generative process for each of those components is as follows: DISPLAYFORM1 T is generated with respect to a categorical distribution with weights w T ??? R k over the k possible LTL templates.

w T is a hyperparameter that the designer can set to assert preferences for mining certain types of templates over others (e.g., preferring templates with "global" operators than "until" operators).The number of conjunctions, N = |{p}|, is generated using a geometric distribution with a decay rate of ??.

Thus, the the probability of ?? is reduced by ?? for each addition of a conjunct, incentivizing low-complexity specifications defined in terms of having a fewer number of conjunctions (which also implies fewer total propositions).

This promotes conciseness and prevents over-fitting to the input traces (i.e., to avoid restating the input as a long, convoluted LTL formula).Similar to the method used for template selection, we use a separate categorical distribution for selecting propositions p for each conjunct in ??. Propositions are generated with respect to the probability weights, w p ??? R |V | , defined for all p in V. The designer can likewise control w p to favor specifications instantiated with certain types of propositions over others.

w p may be interpreted as the level of saliency of propositions for an application.

(For example, propositions that are landmarks for planning problems BID11 , or a part of the causal links set BID31 , may be deemed more important to express in plan explanations than other auxiliary state variables.)

Several forms of variable importance, corresponding to the saliency of that importance in an explanation, may be applied to set w p .

This opens the door to hypothesizing which propositions are most salient for a given domain, and generating explanations restricted to those propositions exclusively.

The full prior function, P (??), is evaluated as follows: DISPLAYFORM2 The derivation follows from the definition that T , N , {p} completely describe ?? (i.e. P (?? | T, N, {p}) = 1), and the assumption that the three probability distributions are independent of each other.

P (T ) and P (N ) are calculated using categorical and geometric distributions outlined in Equations 5 and 6, respectively.

P ({p}) denotes the probability of the full set of proposition instantiations (over all conjuncts); it is calculated by the average categorical weight, w p , over all propositions.

Formally: Likelihood Function The likelihood function P (X | ??) is the probability of observing the input sets of traces in the satisfying set ?? A and the non-satisfying set ?? B given the contrastive specification.

The traces in ?? A and ?? B are generated by different solutions to the planning problem that satisfy the problem specification.

As the problem specification is the only input needed to generate a set of plans, we assume that the individual traces are conditionally independent of each other, given the planning problem specification.

With the conditional independence assumption, the likelihood can then be factored as follows: DISPLAYFORM3 DISPLAYFORM4 LTL satisfaction checks are conducted over all traces belonging to sets ?? A and ?? B ; P (?? i |??) is set equal to 1 ??? ?? if ?? i |= ??, and ?? otherwise.

Conversely, P (?? j |??) is set equal to 1????? if ?? j ??, and ?? otherwise.

?? and ?? permit non-zero probability to traces not adhering to the constrastive explanation criterion, thereby providing robustness to noisy traces and outliers.

?? and ?? may be set to different values to reflect the relative importance of the positive and negative sets (e.g., may be used to counteract imbalanced sets).In order to perform LTL satisfaction checks on a trace, we follow the method developed by BID17 , in which ?? is represented as a tree and each temporal operator is recursively evaluated according to its semantics.

Since sub-trees of two different ?? may be identical, we memoize and re-use evaluation results to significantly speed up LTL satisfaction checks.

Proposal Function Exact inference methods to find maximum a posterior (MAP) estimates, {?? * }, are intractable.

Thus we implement a Markov Chain Monte Carlo method, specifically the Metropolis-Hasting (MH) algorithm BID6 , to iteratively draw samples whose collection approximates the true posterior distribution.

MH sampling requires a user-defined proposal function F (?? |??) that samples a new candidate ?? given the current ??. Our F behaves similar to an -greedy search, utilizing a drift kernel (i.e. a random walk) with a probability of 1-or sampling from the prior distribution (i.e. a restart) with a probability of .

The drift kernel operates by performing one of the following moves on the current candidate LTL ??:??? Remain within the current template T , add a new conjunct, and instantiate that conjunct with a randomly sampled p that is currently not in ??. The probability associated with this move, Q add , is equal to 1/(|V n T | ??? N ).???

Remain within the current template T and randomly remove one of the existing conjuncts.

The probability associated with this move, Q remove , is equal to 1/N .

The selection between these two moves is conducted uniformly, though there is no issue with allowing the designer to weight one more likely than the other.

Note that the drift kernel perturbs ??, but stays within the current template.

?? transitions to a new template (probabilistically) when choosing to sample from the prior.

The probability distribution associated with F , denoted by Q(?? |??), is then outlined as follows: DISPLAYFORM5 , sample prior function Our proposal function F fulfills the ergodicity condition of the Markov process (the transition from any ?? to ?? is aperiodic and occurs within a finite number of steps), thus asymptotically guarantees the sampling process from the true posterior distribution.

A new sample ?? is accepted at every MH iteration with the following probability: DISPLAYFORM6 The set of accepted samples approximates the true posterior, and the MAP estimates (the output {?? * }) are determined from the relative frequencies of accepted samples.

We evaluated the effectiveness of our model for inferring contrastive explanations from sets of traces generated from a number of International Planning Competition (IPC) planning domains BID19 .

The plan traces in ?? A were generated by first injecting the ground truth ?? ground into the original PDDL domain and problem files, enforcing valid plans on the modified domain/problem files to satisfy ?? ground .

The LTL injection to create modified planning files was performed using the LTLFOND2FOND tool BID4 .

Second, a state-of-the-art top-k planner 1 BID13 ) was used to produce a set of distinct, valid plans and their accompanying state execution traces.

Similarly, the above steps were repeated to generate execution traces for ?? B , wherein the negation of the ground truth specification, ???? ground , was injected to the planning files, and then a set of traces was collected.

Such a setup guarantees the existence of contrastive explanation solutions on (?? A , ?? B ), which includes (but is not limited to) ?? ground .

We collected twenty traces for each set.

We evaluated our model using six different IPC benchmark domains, containing problems related to mission planning, vehicle routing, and resource allocation.

For each of these domains, we tested three different problem instances of increasing vocabulary size, and on twenty randomly generated ?? ground specifications for each problem instance.

For each test case, ?? ground was randomly generated using one of the seven LTL templates listed in Table 1 ; thus the hypothesis space ?? was set to include all possible specifications over the predefined templates.

The categorical distribution weights, w T and w p , were set to be uniform.

Other hyperparameters were set as follows: ?? = ?? = 0.01, to put equal importance of positive and negative sets, ?? = 0.7 to penalize ?? for every additional conjunct, and = 0.2 to apply -greedy search in the the proposal function.

We ran the MH sampler with num M H = 2, 000 iterations with the first 300 used as a burn-in period.

These hyperparameters were set apriori, similar to how a wide range of probablistic graphical models are designed.

However, our experimental results were found to be robust to the various settings of these parameters.

We evaluated our model against the SAT-based miner developed by BID22 , the state-of-the-art for extracting contrastive LTL specifications.

We also evaluated our model against brute force enumeration, a common approach employed by existing LTL miners used for summarization BID32 BID17 .

Because enumerating through full space of ?? would result in a time out, we tested delimited enumeration with only a random subset of brute force samples.

This baseline selects a random subset of size num brute from ??. Then, a function proportional to the posterior distribution (numerator in Equation 4) is evaluated for each of the samples to determine {?? * }.

num brute was set equal to num M H to enable a fair baseline in terms of having the same amount of allotted computation.

TAB2 shows the inference results on the tested domains and on problem instances of varying complexity (reflected by an increase in |V |).

For evaluation, we measured M = |{?? * }|, the number of unique contrastive explanations extracted by the different approaches, along with the explanations' accuracy.

Each domain-problem combination row shows the average statistics over twenty ?? ground test cases.

High M and high accuracy across all domain-problem combinations demonstrate how our probabilistic model was able to generate multiple, near-perfect contrastive explanations.

The solution set {?? * } almost always included ?? ground .

Our model outperformed the baseline and the stateof-the-art miner by producing more contrastive explanations within an allotted amount of computation / runtime.

The runtime for our model and the delimited enumeration baseline with 2,000 samples ranged between 1.2-4.7 seconds (increase in |V | only had marginal effect on the runtime).

The SAT-based miner by Neider and Gavran often failed to generate a solution within a five minute cutoff (see the number of its timeout cases in the last column of TAB2 ).

The prior work can only output a single ?? * , which frequently took on a form of Fp i .

It did not scale well to problems that required more complex ?? as solutions.

This is because increasing the "depth" of ?? (the number of temporal / Boolean operators and propositions) exponentially increased the size of the compiled SAT problem.

In our experiments, the prior work timed out for problems requiring solutions with depth ??? 3 (note that Fp i has depth of 2).

Robustness to Noisy Input In order to test robustness, we perturbed the input X by randomly swapping traces between ?? A and ?? B .

For example, a noise rate of 0.2 would swap 20% of the traces, where the accuracy of ?? ground on the perturbed data, X = ( ?? A , ?? B ), would evaluate to 0.8 (note that it may be possible to discover other ?? that achieve better accuracy on X).

The MAP estimates inferred from X, { ?? * }, were evaluated on the original input X to assess any loss of ability to provide contrast.

Figure 3 shows the average accuracy of { ?? * }, evaluated on both X and X, across varying noise rate.

Even at a moderate noise rate of 0.25, the inferred ?? * s were able to maintain an average accuracy greater than 0.9 on X. Such a threshold is promising for real-world applications.

The robustness did start to sharply decline as noise rate increased past 0.4.

For all test cases, the Neider and Gavran miner failed to generate a solution for anything with a noise rate ??? 0.1.

Large values of M signify how there are often various ways to express how plan traces differ using the LTL semantics.

Some LTL specifications are logically dependent.

For example, the global template subsumes both the stability and the eventuality template.

LTL specifications may also be related through substitutions of propositions.

For example, on Figure 3 : The accuracy of ?? * with respect to increasing noise rate.

?? * is inferred from the perturbed, noisy data and then is evaluated (generalized) on the original input X. Each domain subplot shows the averages across all three problem instances and all twenty ?? ground test cases.

95% confidence intervals are displayed.problems where holding a block is a prerequisite to placing it onto a table, ?? 1 = F(holding A) ??? F(holding B) will be satisfied in concert with the satisfaction of ?? 2 = F(ontable A)???F(ontable B).

For contrastive explanation, however, one needs to be mindful of both positive and negative sides of satisfaction which affect the accuracy.

Relations like template subsumptions or precondition / effect pairs should not be simply favored during search without understanding that the converse may not hold and may result in worse accuracy.

For a contrastive ??, it is possible to create a new contrastive ?? that includes stationary propositions or tautologies specific to the planning problem.

For example, if ?? 1 = F(holding A) ??? F(holding B) is a contrastive explanation, so is ?? 3 = F(holding A) ??? F(holding B) ??? F(earth is round).

Our posterior distribution assigns a lower probability to ?? 3 than ?? 1 based on the decay rate on the number of conjunctions.

Also, tautologies by themselves cannot be contrastive explanations, because they can never be dissatisfied.

The output of our model appropriately excluded such vacuous explanations.

TAB2 shows how M generally increased as |V | increased.

This opens up interesting research avenues for determining a minimal set of {??}. Assessing logical dependence or metric space between two arbitrary LTL specifications, however, is non-trivial.

Evaluation on Real-world Inspired Domain We applied our inference model on a large force exercise (LFE) domain, which simulate air-combat games used to train pilots.

Through the use of Joint Semi-Automated Forces environment BID0 , realistic aircraft behavior and their state execution traces were collected for the mission objective of "gain and maintain air superiority."

A total of 24 instances (i.e. traces) of LFEs were separated into positive and negative sets by a subject matter expert.

The detail of the input was as follows: |?? A |=16, |?? B |=8, |V |=15, and the average length of traces involved 11 time steps.

Within a second (2,000 samples), our model generated ten unique contrastive explanations, all with accuracy of 0.96.

?? * 1 = G(attrition < 0.25) ??? G(striker not shot) represents how friendly attrition rate should be always less than 25% and that the striker aircraft should never be shot upon.

?? * 2 = (attrition < 0.25) U (weapon release) asserts how friendly attrition rate has to be less than 25% before releasing the weapon.

The model also inferred rules of the environment, for example, asserting that propositions (attrition < 0.75) and (attrition < 0.50) precede (attrition < 0.25) (which makes sense because attrition can only increase throughout the mission).

After discussion with the expert, we discovered that the model could not generate the perfect contrastive ?? ground , because it required having multiple conjuncts that incorporate different LTL templates (which is not part of our defined solution space).

Nevertheless, the generated explanations were consistent with the expert's interpretation of achieving the mission objective of air superiority.

We have presented a probabilistic Bayesian model to infer contrastive LTL specifications describing how two sets of plan traces differ.

Our model generates multiple contrastive explanations more efficiently than the state-of-the-art and demonstrates robustness to noisy input.

It also provides a principled approach to incorporate various forms of prior knowledge or preferences during search.

It can serve as a strong foundation that can be naturally extended to multiple input sets by repeating the algorithm for all pairwise or one-vs.-rest comparisons.

Interesting avenues for future work include gauging the saliency of propositions, as well as deriving a minimal set of contrastive explanations.

Furthermore, we seek to test the model in human-in-the-loop settings, with the goal of understanding the relationship between different planning heuristics for the saliency of propositions (e.g. landmarks and causal links) to their actual explicability when the explanation is communicated to a human.

<|TLDR|>

@highlight

We present a Bayesian inference model to infer contrastive explanations (as LTL specifications) describing how two sets of plan traces differ.