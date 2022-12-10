Synthesizing user-intended programs from a small number of input-output exam- ples is a challenging problem with several important applications like spreadsheet manipulation, data wrangling and code refactoring.

Existing synthesis systems either completely rely on deductive logic techniques that are extensively hand- engineered or on purely statistical models that need massive amounts of data, and in general fail to provide real-time synthesis on challenging benchmarks.

In this work, we propose Neural Guided Deductive Search (NGDS), a hybrid synthesis technique that combines the best of both symbolic logic techniques and statistical models.

Thus, it produces programs that satisfy the provided specifications by construction and generalize well on unseen examples, similar to data-driven systems.

Our technique effectively utilizes the deductive search framework to reduce the learning problem of the neural component to a simple supervised learning setup.

Further, this allows us to both train on sparingly available real-world data and still leverage powerful recurrent neural network encoders.

We demonstrate the effectiveness of our method by evaluating on real-world customer scenarios by synthesizing accurate programs with up to 12× speed-up compared to state-of-the-art systems.

Automatic synthesis of programs that satisfy a given specification is a classical problem in AI BID29 , with extensive literature in both machine learning and programming languages communities.

Recently, this area has gathered widespread interest, mainly spurred by the emergence of a sub-area -Programming by Examples (PBE) BID10 .

A PBE system synthesizes programs that map a given set of example inputs to their specified example outputs.

Such systems make many tasks accessible to a wider audience as example-based specifications can be easily provided even by end users without programming skills.

See Figure 1 for an example.

PBE systems are usually evaluated on three key criteria: (a) correctness: whether the synthesized program

Yann LeCunn Y LeCunn Hugo Larochelle H Larochelle Tara Sainath T SainathYoshua Bengio ?

Figure 1 : An example input-output spec; the goal is to learn a program that maps the given inputs to the corresponding outputs and generalizes well to new inputs.

Both programs below satisfy the spec: (i) Concat(1 st letter of 1 st word, 2 nd word), (ii) Concat(4 th -last letter of 1 st word, 2 nd word).

However, program (i) clearly generalizes better: for instance, its output on "Yoshua Bengio" is "Y Bengio" while program (ii) produces "s Bengio". satisfies the spec i.e. the provided example input-output mapping, (b) generalization: whether the program produces the desired outputs on unseen inputs, and finally, (c) performance: synthesis time.

State-of-the-art PBE systems are either symbolic, based on enumerative or deductive search BID10 BID21 or statistical, based on data-driven learning to induce the most likely program for the spec BID8 BID1 BID6 .

Symbolic systems are designed to produce a correct program by construction using logical reasoning and domain-specific knowledge.

They also produce the intended program with few input-output examples (often just 1).

However, they require significant engineering effort and their underlying search processes struggle with real-time performance, which is critical for user-facing PBE scenarios.

In contrast, statistical systems do not rely on specialized deductive algorithms, which makes their implementation and training easier.

However, they lack in two critical aspects.

First, they require a lot of training data and so are often trained using randomly generated tasks.

As a result, induced programs can be fairly unnatural and fail to generalize to real-world tasks with a small number of examples.

Second, purely statistical systems like RobustFill BID6 do not guarantee that the generated program satisfies the spec.

Thus, solving the synthesis task requires generating multiple programs with a beam search and post-hoc filtering, which defeats real-time performance.

Neural-Guided Deductive Search Motivated by shortcomings of both the above approaches, we propose Neural-Guided Deductive Search (NGDS), a hybrid synthesis technique that brings together the desirable aspects of both methods.

The symbolic foundation of NGDS is deductive search BID21 and is parameterized by an underlying domain-specific language (DSL) of target programs.

Synthesis proceeds by recursively applying production rules of the DSL to decompose the initial synthesis problem into smaller sub-problems and further applying the same search technique on them.

Our key observation I is that most of the deduced sub-problems do not contribute to the final best program and therefore a priori predicting the usefulness of pursuing a particular sub-problem streamlines the search process resulting in considerable time savings.

In NGDS, we use a statistical model trained on real-world data to predict a score that corresponds to the likelihood of finding a generalizable program as a result of exploring a sub-problem branch.

Our key observation II is that speeding up deductive search while retaining its correctness or generalization requires a close integration of symbolic and statistical approaches via an intelligent controller.

It is based on the "branch & bound" technique from combinatorial optimization BID5 .

The overall algorithm integrates (i) deductive search, (ii) a statistical model that predicts, a priori, the generalization score of the best program from a branch, and (iii) a controller that selects sub-problems for further exploration based on the model's predictions.

Since program synthesis is a sequential process wherein a sequence of decisions (here, selections of DSL rules) collectively construct the final program, a reinforcement learning setup seems more natural.

However, our key observation III is that deductive search is Markovian -it generates independent sub-problems at every level.

In other words, we can reason about a satisfying program for the sub-problem without factoring in the bigger problem from which it was deduced.

This brings three benefits enabling a supervised learning formulation: (a) a dataset of search decisions at every level over a relatively small set of PBE tasks that contains an exponential amount of information about the DSL promoting generalization, (b) such search traces can be generated and used for offline training, (c) we can learn separate models for different classes of sub-problems (e.g. DSL levels or rules), with relatively simpler supervised learning tasks.

Evaluation We evaluate NGDS on the string transformation domain, building on top of PROSE, a commercially successful deductive synthesis framework for PBE BID21 .

It represents one of the most widespread and challenging applications of PBE and has shipped in multiple mass-market tools including Microsoft Excel and Azure ML Workbench.1 We train and validate our method on 375 scenarios obtained from real-world customer tasks BID10 BID6 .

Thanks to the Markovian search properties described above, these scenarios generate a dataset of 400, 000+ intermediate search decisions.

NGDS produces intended programs on 68% of the scenarios despite using only one input-output example.

In contrast, state-of-the-art neural synthesis techniques BID1 BID6 learn intended programs from a single example in only 24-36% of scenarios taking ≈ 4× more time.

Moreover, NGDS matches the accuracy of baseline PROSE while providing a speed-up of up to 12× over challenging tasks.

Contributions First, we present a branch-and-bound optimization based controller that exploits deep neural network based score predictions to select grammar rules efficiently (Section 3.2).

Second, we propose a program synthesis algorithm that combines key traits of a symbolic and a statistical approach to retain desirable properties like correctness, robust generalization, and real-time performance (Section 3.3).

Third, we evaluate NGDS against state-of-the-art baselines on real customer tasks and show significant gains (speed-up of up to 12×) on several critical cases (Section 4).

In this section, we provide a brief background on PBE and the PROSE framework, using established formalism from the programming languages community.

Domain-Specific Language A program synthesis problem is defined over a domain-specific language (DSL).

A DSL is a restricted programming language that is suitable for expressing tasks in a given domain, but small enough to restrict a search space for program synthesis.

For instance, typical real-life DSLs with applications in textual data transformations BID10 often include conditionals, limited forms of loops, and domain-specific operators such as string concatenation, regular expressions, and date/time formatting.

DSLs for tree transformations such as code refactoring BID24 and data extraction BID16 include list/data-type processing operators such as Map and Filter, as well as domain-specific matching operators.

Formally, a DSL L is specified as a context-free grammar, with each non-terminal symbol N defined by a set of productions.

The right-hand side of each production is an application of some operator F (N 1 , . . .

, N k ) to some symbols of L. All symbols and operators are strongly typed.

FIG0 shows a subset of the Flash Fill DSL that we use as a running example in this paper.

The task of inductive program synthesis is characterized by a spec.

A spec ϕ is a set of m input-output constraints {σ DISPLAYFORM0 , where: • σ, an input state is a mapping of free variables of the desired program P to some correspondingly typed values.

At the top level of L, a program (and its expected input state) has only one free variable -the input variable of the DSL (e.g., inputs in FIG0 ).

Additional local variables are introduced inside L with a let construct.• ψ is an output constraint on the execution result of the desired program P (σ i ).

At the top level of L, when provided by the user, ψ is usually the output example -precisely the expected result of P (σ i ).

However, other intermediate constraints arise during the synthesis process.

For instance, ψ may be a disjunction of multiple allowed outputs.

The overall goal of program synthesis is thus: given a spec ϕ, find a program P in the underlying DSL L that satisfies ϕ, i.e., its outputs P (σ i ) satisfy all the corresponding constraints ψ i .

Example 1.

Consider the task of formatting a phone number, characterized by the spec ϕ = {inputs : ["(612) 8729128"]} "612-872-9128".

It has a single input-output example, with an input state σ containing a single variable inputs and its value which is a list with a single input string.

The output constraint is simply the desired program result.

The program the user is most likely looking for is the one that extracts (a) the part of the input enclosed in the first pair of parentheses, (b) the 7 th to 4 th characters from the end, and (c) the last 4 characters, and then concatenates all three parts using hyphens.

In our DSL, this corresponds to: DISPLAYFORM1 where ε is an empty regex, SubStr 0 (pos 1 , pos 2 ) is an abbreviation for "let x = std.

Kth(inputs, 0) in Substring(x, pos 1 , pos 2 )", and · is an abbreviation for std.

Pair.

However, many other programs in the DSL also satisfy ϕ. For instance, all occurrences of "8" in the output can be produced via a subprogram that simply extracts the last character.

Such a program overfits to ϕ and is bound to fail for other inputs where the last character and the 4 th one differ.

BID10 , used as a running example in this paper.

Every program takes as input a list of strings inputs, and returns an output string, a concatenation of atoms.

Each atom is either a constant or a substring of one of the inputs (x), extracted using some position logic.

The RegexOccurrence position logic finds k th occurrence of a regex r in x and returns its boundaries.

Alternatively, start and end positions can be selected independently either as absolute indices in x from left or right (AbsolutePosition) or as the k th occurrence of a pair of regexes surrounding the position (RegexPosition).

See BID10 for an in-depth DSL description.

As Example 1 shows, typical real-life problems are severely underspecified.

A DSL like FlashFill may contain up to 10 20 programs that satisfy a given spec of 1-3 input-output examples BID21 .

Therefore, the main challenge lies in finding a program that not only satisfies the provided input-output examples but also generalizes to unseen inputs.

Thus, the synthesis process usually interleaves search and ranking: the search phase finds a set of spec-satisfying programs in the DSL, from which the ranking phase selects top programs ordered using a domain-specific ranking function h : L × Σ → R where Σ is the set of all input states.

The ranking function takes as input a candidate program P ∈ L and a set of input states σ ∈ Σ (usually σ = inputs in the given spec + any available unlabeled inputs), and produces a score for P 's generalization.

The implementation of h expresses a subtle balance between program generality, complexity, and behavior on available inputs.

For instance, in FlashFill h penalizes overly specific regexes, prefers programs that produce fewer empty outputs, and prioritizes lower Kolmogorov complexity, among other features.

In modern PBE systems like PROSE, h is usually learned in a data-driven manner from customer tasks BID26 BID7 .

While designing and learning such a ranking is an interesting problem in itself, in this work we assume a black-box access to h. Finally, the problem of inductive program synthesis can be summarized as follows: DISPLAYFORM2 , optionally a set of unlabeled inputs σ u , and a target number of programs DISPLAYFORM3 .

The goal of inductive program synthesis is to find a program set S = {P 1 , . . .

, P K } ⊂ L such that (a) every program in S satisfies ϕ, and (b) the programs in S generalize best: h(P i , σ) ≥ h(P, σ) for any other P ∈ L that satisfies ϕ.Search Strategy Deductive search strategy for program synthesis, employed by PROSE explores the grammar of L top-down -iteratively unrolling the productions into partial programs starting from the root symbol.

Following the divide-and-conquer paradigm, at each step it reduces its synthesis problem to smaller subproblems defined over the parameters of the current production.

Formally, given a spec ϕ and a symbol N , PROSE computes the set Learn(N, ϕ) of top programs w.r.t.

h using two guiding principles: DISPLAYFORM4 , PROSE finds a ϕ-satisfying program set for every F i , and unites the results, i.e., Learn(N, ϕ) = ∪ i Learn(F i (. . .), ϕ).

2.

For a given production N := F (N 1 , . . . , N k ), PROSE spawns off k smaller synthesis problems Learn(N j , ϕ j ), 1 ≤ j ≤ k wherein PROSE deduces necessary and sufficient specs ϕ j for each N j such that every program of type F (P 1 , . . .

, P k ), where P j ∈ Learn(N j , ϕ j ), satisfies ϕ. The deduction logic (called a witness function) is domain-specific for each operator F .

PROSE then again recursively solves each subproblem and unites a cross-product of the results.

Example 2.

Consider a spec ϕ = {"Yann" "Y.L"} on a transf orm program.

Via the first production transf orm := atom, the only ϕ-satisfying program is ConstStr("Y.L").

The second production on the same level is Concat(atom, transf orm).

A necessary & sufficient spec on the atom sub-program is that it should produce some prefix of the output string.

Thus, the witness function for the Concat operator produces a disjunctive spec ϕ a = {"Yann" DISPLAYFORM5 of these disjuncts, in turn, induces a corresponding necessary and sufficient suffix spec on the second parameter: ϕ t1 = {"Yann" ".L"}, and ϕ t2 = {"Yann" "L"}, respectively.

The disjuncts in ϕ a will be recursively satisfied by different program sets: "Y." can only be produced via an atom path with a ConstStr program, whereas "Y" can also be extracted from the input using many Substring logics (their generalization capabilities vary).

FIG1 shows the resulting search DAG.

DISPLAYFORM6 . . .

Notice that the above mentioned principles create logical non-determinism due to which we might need to explore multiple alternatives in a search tree.

As such non-determinism arises at every level of the DSL with potentially any operator, the search tree (and the resulting search process) is exponential in size.

While all the branches of the tree by construction produce programs that satisfy the given spec, most of the branches do not contribute to the overall top-ranked generalizable program.

During deductive search, PROSE has limited information about the programs potentially produced from each branch, and cannot estimate their quality, thus exploring the entire tree unnecessarily.

Our main contribution is a neural-guided search algorithm that predicts the best program scores from each branch, and allows PROSE to omit branches that are unlikely to produce the desired program a priori.

Consider an arbitrary branching moment in the top-down search strategy of PROSE.

For example, let N be a nonterminal symbol in L, defined through a set of productions N := F 1 (. . .) | . . .

| F n (. . .), and let ϕ be a spec on N , constructed earlier during the recursive descent over L. A conservative way to select the top k programs rooted at N (as defined by the ranking function h), i.e., to compute Learn(N, ϕ), is to learn the top k programs of kind F i (. . .) for all i ∈ [k] and then select the top k programs overall from the union of program sets learned for each production.

Naturally, exploring all the branches for each nonterminal in the search tree is computationally expensive.

In this work, we propose a data-driven method to select an appropriate production rule N := F i (N 1 , . . .

, N k ) that would most likely lead to a top-ranked program.

To this end, we use the current spec ϕ to determine the "optimal" rule.

Now, it might seem unintuitive that even without exploring a production rule and finding the best program in the corresponding program set, we can a priori determine optimality of that rule.

However, we argue that by understanding ϕ and its relationship with the ranking function h, we can predict the intended branch in many real-life scenarios.

Example 3.

Consider a spec ϕ = {"alice" "alice@iclr.org", "bob" "bob@iclr.org"}. While learning a program in L given by FIG0 that satisfies ϕ, it is clear right at the beginning of the search procedure that the rule transf orm := atom does not apply.

This is because any programs derived from transf orm := atom can either extract a substring from the input or return a constant string, both of which fail to produce the desired output.

Hence, we should only consider transf orm := Concat(. . .), thus significantly reducing the search space.

Similarly, consider another spec ϕ = {"alice smith""alice", "bob jones" "bob"}. In this case, the output appears to be a substring of input, thus selecting transf orm := atom at the beginning of the search procedure is a better option than transf orm := Concat(. .

.).However, many such decisions are more subtle and depend on the ranking function h itself.

For example, consider a spec ϕ = {"alice liddell""al", "bob ong" "bo"}. Now, Figure 4 : LSTM-based model for predicting the score of a candidate production for a given spec ϕ.both transf orm := atom and transf orm := Concat(. . .) may lead to viable programs because the output can be constructed using the first two letters of the input (i.e. a substring atom) or by concatenating the first letters of each word.

Hence, the branch that produces the best program is ultimately determined by the ranking function h since both branches generate valid programs.

Example 3 shows that to design a data-driven search strategy for branch selection, we need to learn the subtle relationship between ϕ, h, and the candidate branch.

Below, we provide one such model.

As mentioned above, our goal is to predict one or more production rules that for a given spec ϕ will lead to a top-ranked program (as ranked a posteriori by h).

Formally, given black-box access to h, we want to learn a function f such that, DISPLAYFORM0 where Γ is a production rule in L, and S(Γ, ϕ) is a program set of all DSL programs derived from the rule Γ that satisfy ϕ. In other words, we want to predict the score of the top-ranked ϕ-satisfying program that is synthesized by unrolling the rule Γ .

We assume that the symbolic search of PROSE handles the construction of S(Γ, ϕ) and ensures that programs in it satisfy ϕ by construction.

The goal of f is to optimize the score of a program derived from Γ assuming this program is valid.

If no program derived from Γ can satisfy ϕ, f should return −∞. Note that, drawing upon observations mentioned in Section 1, we have cast the production selection problem as a supervised learning problem, thus simplifying the learning task as opposed to end-to-end reinforcement learning solution.

We have evaluated two models for learning f .

The loss function for the prediction is given by: DISPLAYFORM1 Figure 4 shows a common structure of both models we have evaluated.

Both are based on a standard multi-layer LSTM architecture BID13 and involve (a) embedding the given spec ϕ, (b) encoding the given production rule Γ , and (c) a feed-forward network to output a score f (Γ, ϕ).

One model attends over input when it encodes the output, whereas another does not.

A score model f alone is insufficient to perfectly predict the branches that should be explored at every level.

Consider again a branching decision moment N := F 1 (. . .) | . . .

| F n (. . .) in a search process for top k programs satisfying a spec ϕ. One naïve approach to using the predictions of f is to always follow the highest-scored production rule argmax i f (F i , ϕ).

However, this means that any single incorrect decision on the path from the DSL root to the desired program will eliminate that program from the learned program set.

If our search algorithm fails to produce the desired program by committing to a suboptimal branch anytime during the search process, then the user may never discover that such a program exists unless they supply additional input-output example.

Thus, a branch selection strategy based on the predictions of f must balance a trade-off of performance and generalization.

Selecting too few branches (a single best branch in the extreme case) risks committing to an incorrect path early in the search process and producing a suboptimal program or no program at all.

Selecting too many branches (all n branches in the extreme case) is no different from baseline PROSE and fails to exploit the predictions of f to improve its performance.

Formally, a controller for branch selection at a symbol N := F 1 (. . .)

| . . .

| F n (. . .) targeting k best programs must (a) predict the expected score of the best program from each program set: DISPLAYFORM0 DISPLAYFORM1 if k ≤ 0 then break 8: return S * Figure 5 : The controllers for guiding the search process to construct a most generalizable ϕ-satisfying program set S of size k given the f -predicted best scores s 1 , . . .

, s n of the productions F 1 , . . .

, F n .Given: DSL L, ranking function h, controller C from Figure 5 (THRESHOLDBASED or BNBBASED), symbolic search algorithm LEARN(Production rule Γ , spec ϕ, target k) as in PROSE BID21 , Figure 7 ) with all recursive calls to LEARN replaced with LEARNNGDS function LEARNNGDS(Symbol N := F1(. . .) | . . .

| Fn(. . .), spec ϕ, target number of programs k) 1: if n = 1 then return LEARN(F1, ϕ, k) 2: Pick a score model f based on depth(N, L) 3: s1, . . .

, sn ← f (F1, ϕ) , . . .

, f (Fn, ϕ) 4: return C(ϕ, h, k, s1, . . .

, sn)Figure 6: Neural-guided deductive search over L, parameterized with a branch selection controller C. DISPLAYFORM2 and (b) use the predicted scores s i to narrow down the set of productions F 1 , . . .

, F n to explore and to obtain the overall result by selecting a subset of generated programs.

In this work, we propose and evaluate two controllers.

Their pseudocode is shown in Figure 5 .Threshold-based: Fix a score threshold θ, and explore those branches whose predicted score differs by at most θ from the maximum predicted score.

This is a simple extension of the naïve "argmax" controller discussed earlier that also explores any branches that are predicted "approximately as good as the best one".

When θ = 0, it reduces to the "argmax" one.

This controller is based on the "branch & bound" technique in combinatorial optimization BID5 .

Assume the branches F i are ordered in the descending order of their respective predicted scores s i .

After recursive learning produces its program set S i , the controller proceeds to the next branch only if s i+1 exceeds the score of the worst program in S i .

Moreover, it reduces the target number of programs to be learned, using s i+1 as a lower bound on the scores of the programs in S i .

That is, rather than relying blindly on the predicted scores, the controller guides the remaining search process by accounting for the actual synthesized programs as well.

We now combine the above components to present our unified algorithm for program synthesis.

It builds upon the deductive search of the PROSE system, which uses symbolic PL insights in the form of witness functions to construct and narrow down the search space, and a ranking function h to pick the most generalizable program from the found set of spec-satisfying ones.

However, it significantly speeds up the search process by guiding it a priori at each branching decision using the learned score model f and a branch selection controller, outlined in Sections 3.1 and 3.2.

The resulting neural-guided deductive search (NGDS) keeps the symbolic insights that construct the search tree ensuring correctness of the found programs, but explores only those branches of this tree that are likely to produce the user-intended generalizable program, thus eliminating unproductive search time.

A key idea in NGDS is that the score prediction model f does not have to be the same for all decisions in the search process.

It is possible to train separate models for different DSL levels, symbols, or even productions.

This allows the model to use different features of the input-output spec for evaluating the fitness of different productions, and also leads to much simpler supervised learning problems.

Figure 6 shows the pseudocode of NGDS.

It builds upon the deductive search of PROSE, but augments every branching decision on a symbol with some branch selection controller from Section 3.2.

We present a comprehensive evaluation of different strategies in Section 4.

Table 1 : Accuracy and average speed-up of NGDS vs. baseline methods.

Accuracies are computed on a test set of 73 tasks.

Speed-up of a method is the geometric mean of its per-task speed-up (ratio of synthesis time of PROSE and of the method) when restricted to a subset of tasks with PROSE's synthesis time is ≥ 0.5 sec.

In this section, we evaluate our NGDS algorithm over the string manipulation domain with a DSL given by FIG0 ; see Figure 1 for an example task.

We evaluate NGDS, its ablations, and baseline techniques on two key metrics: (a) generalization accuracy on unseen inputs, (b) synthesis time.

Dataset.

We use a dataset of 375 tasks collected from real-world customer string manipulation problems, split into 65% training, 15% validation, and 20% test data.

Some of the common applications found in our dataset include date/time formatting, manipulating addresses, modifying names, automatically generating email IDs, etc.

Each task contains about 10 inputs, of which only one is provided as the spec to the synthesis system, mimicking industrial applications.

The remaining unseen examples are used to evaluate generalization performance of the synthesized programs.

After running synthesis of top-1 programs with PROSE on all training tasks, we have collected a dataset of ≈ 400,000 intermediate search decisions, i.e. triples production Γ, spec ϕ, a posteriori best score h(P, ϕ) .Baselines.

We compare our method against two state-of-the-art neural synthesis algorithms: RobustFill BID6 and DeepCoder BID1 .

For RobustFill, we use the best-performing Attention-C model and use their recommended DP-Beam Search with a beam size of 100 as it seems to perform the best; TAB5 in Appendix A presents results with different beam sizes.

As in the original work, we select the top-1 program ranked according to the generated log-likelihood.

DeepCoder is a generic framework that allows their neural predictions to be combined with any program synthesis method.

So, for fair comparison, we combine DeepCoder's predictions with PROSE.

We train DeepCoder model to predict a distribution over L's operators and as proposed, use it to guide PROSE synthesis.

Since both RobustFill and DeepCoder are trained on randomly sampled programs and are not optimized for generalization in the real-world, we include their variants trained with 2 or 3 examples (denoted RF m and DC m ) for fairness, although m = 1 example is the most important scenario in real-life industrial usage.

Ablations.

As mentioned in Section 3, our novel usage of score predictors to guide the search enables us to have multiple prediction models and controllers at various stages of the synthesis process.

Here we investigate ablations of our approach with models that specialize in predictions for individual levels in the search process.

The model T 1 is trained for symbol transf orm FIG0 ) when expanded in the first level.

Similarly, P P , P OS refer to models trained for the pp and pos symbol, respectively.

Finally, we train all our LSTM-based models with CNTK BID25 using Adam BID15 ) with a learning rate of 10 −2 and a batch size of 32, using early stopping on the validation loss to select the best performing model (thus, 100-600 epochs).We also evaluate three controllers: threshold-based (Thr) and branch-and-bound (BB) controllers given in Figure 5 , and a combination of them -branch-and-bound with a 0.2 threshold predecessor (BB 0.2 ).

In Tables 1 and 2 we denote different model combinations as NGDS(f , C) where f is a symbol-based model and C is a controller.

The final algorithm selection depends on its accuracyperformance trade-off.

In Table 1 , we use NGDS(T 1 + P OS, BB), the best performing algorithm on the test set, although NGDS(T 1 , BB) performs slightly better on the validation set.

Evaluation Metrics.

Generalization accuracy is the percentage of test tasks for which the generated program satisfies all unseen inputs in the task.

Synthesis time is measured as the wall-clock time taken by a synthesis method to find the correct program, median over 5 runs.

We run all the methods on the same machine with 2.3 GHz Intel Xeon processor, 64GB of RAM, and Windows Server 2016.Results.

Table 1 presents generalization accuracy as well as synthesis time speed-up of various methods w.r.t.

PROSE.

As we strive to provide real-time synthesis, we only compare the times for tasks which require PROSE more than 0.5 sec. Note that, with one example, NGDS and PROSE are Table 2 : Accuracies, mean speed-ups, and % of branches taken for different ablations of NGDS.significantly more accurate than RobustFill and DeepCoder.

This is natural as those methods are not trained to optimize generalization, but it also highlights advantage of a close integration with a symbolic system (PROSE) that incorporates deep domain knowledge.

Moreover, on an average, our method saves more than 50% of synthesis time over PROSE.

While DeepCoder with one example speeds up the synthesis even more, it does so at the expense of accuracy, eliminating branches with correct programs in 65% of tasks.

Table 2 presents speed-up obtained by variations of our models and controllers.

In addition to generalization accuracy and synthesis speed-up, we also show a fraction of branches that were selected for exploration by the controller.

Our method obtains impressive speed-up of > 1.5× in 22 cases.

One such test case where we obtain 12× speedup is a simple extraction case which is fairly common in Web mining: {"alpha,beta,charlie,delta" "alpha"}. For such cases, our model determine transf orm := atom to be the correct branch (that leads to the final Substring based program) and hence saves time required to explore the entire Concat operator which is expensive.

Another interesting test case where we observe 2.7× speed-up is: {"457 124th St S, Seattle, WA 98111" "Seattle-WA"}. This test case involves learning a Concat operator initially followed by Substring and RegexPosition operator.

Appendix B includes a comprehensive table of NGDS performance on all the validation and test tasks.

All the models in Table 2 run without attention.

As measured by score flip accuracies (i.e. percentage of correct orderings of branch scores on the same level), attention-based models perform best, achieving 99.57/90.4/96.4% accuracy on train/validation/test, respectively (as compared to 96.09/91.24/91.12% for non-attention models).

However, an attention-based model is significantly more computationally expensive at prediction time.

Evaluating it dominates the synthesis time and eliminates any potential speed-ups.

Thus, we decided to forgo attention in initial NGDS and investigate model compression/binarization in future work.

Error Analysis.

As Appendix B shows, NGDS is slower than PROSE on some tasks.

This occurs when the predictions do not satisfy the constraints of the controller i.e. all the predicted scores are within the threshold or they violate the actual scores during B&B exploration.

This leads to NGDS evaluating the LSTM for branches that were previously pruned.

This is especially harmful when branches pruned out at the very beginning of the search need to be reconsidered -as it could lead to evaluating the neural network many times.

While a single evaluation of the network is quick, a search tree involves many evaluations, and when performance of PROSE is already < 1 s, this results in considerable relative slowdown.

We provide two examples to illustrate both the failure modes:(a) " 41.7114830017,-91.41233825683,41.60762786865,-91.63739013671" "41.7114830017 ".

The intended program is a simple substring extraction.

However, at depth 1, the predicted score of Concat is much higher than the predicted score of Atom, and thus NGDS explores only the Concat branch.

The found Concat program is incorrect because it uses absolute position indexes and does not generalize to other similar extraction tasks.

We found this scenario common with punctuation in the output string, which the model considers a strong signal for Concat.(b) "type size = 36: Bartok.

Analysis.

CallGraphNode type size = 32: Bartok.

Analysis.

CallGraphNode CallGraphNode" "36->32".

In this case, NGDS correctly explores only the Concat branch, but the slowdown happens at the pos symbol.

There are many different logics to extract the "36" and "32" substrings.

NGDS explores the RelativePosition branch first, but the score of the resulting program is less then the prediction for RegexPositionRelative.

Thus, the B&B controller explores both branches anyway, which leads to a relative slowdown caused by the network evaluation time.

Neural Program Induction systems synthesize a program by training a new neural network model to map the example inputs to example outputs BID9 BID23 BID30 .

Examples include Neural Turing Machines BID9 ) that can learn simple programs like copying/sorting, work of BID14 that can perform more complex computations like binary multiplications, and more recent work of BID3 that can incorporate recursions.

While we are interested in ultimately producing the right output, all these models need to be re-trained for a given problem type, thus making them unsuitable for real-life synthesis of different programs with few examples.

Neural Program Synthesis systems synthesize a program in a given L with a pre-learned neural network.

Seminal works of BID2 and BID8 proposed first producing a high-level sketch of the program using procedural knowledge, and then synthesizing the program by combining the sketch with a neural or enumerative synthesis engine.

In contrast, R3NN BID20 and RobustFill BID6 systems synthesize the program end-to-end using a neural network; BID6 show that RobustFill in fact outperforms R3NN.

However, RobustFill does not guarantee generation of spec-satisfying programs and often requires more than one example to find the intended program.

In fact, our empirical evaluation (Section 4) shows that our hybrid synthesis approach significantly outperforms the purely statistical approach of RobustFill.

DeepCoder BID1 is also a hybrid synthesis system that guides enumerative program synthesis by prioritizing DSL operators according to a spec-driven likelihood distribution on the same.

However, NGDS differs from DeepCoder in two important ways: (a) it guides the search process at each recursive level in a top-down goal-oriented enumeration and thus reshapes the search tree, (b) it is trained on real-world data instead of random programs, thus achieving better generalization.

Symbolic Program Synthesis has been studied extensively in the PL community , dating back as far as 1960s BID29 .

Most approaches employ either bottom-up enumerative search BID28 , constraint solving BID27 , or inductive logic programming BID17 , and thus scale poorly to real-world industrial applications (e.g. data wrangling applications).

In this work, we build upon deductive search, first studied for synthesis by BID19 , and primarily used for program synthesis from formal logical specifications BID22 BID4 .

BID10 and later BID21 used it to build PROSE, a commercially successful domain-agnostic system for PBE.

While its deductive search guarantees program correctness and also good generalization via an accurate ranking function, it still takes several seconds on complex tasks.

Thus, speeding up deductive search requires considerable engineering to develop manual heuristics.

NGDS instead integrates neural-driven predictions at each level of deductive search to alleviate this drawback.

Work of BID18 represents the closest work with a similar technique but their work is applied to an automated theorem prover, and hence need not care about generalization.

In contrast, NGDS guides the search toward generalizable programs while relying on the underlying symbolic engine to generate correct programs.

We studied the problem of real-time program synthesis with a small number of input-output examples.

For this problem, we proposed a neural-guided system that builds upon PROSE, a state-of-the-art symbolic logic based system.

Our system avoids top-down enumerative grammar exploration required by PROSE thus providing impressive synthesis performance while still retaining key advantages of a deductive system.

That is, compared to existing neural synthesis techniques, our system enjoys following advantages: a) correctness: programs generated by our system are guaranteed to satisfy the given input-output specification, b) generalization: our system learns the user-intended program with just one input-output example in around 60% test cases while existing neural systems learn such a program in only 16% test cases, c) synthesis time: our system can solve most of the test cases in less than 0.1 sec and provide impressive performance gains over both neural as well symbolic systems.

The key take-home message of this work is that a deep integration of a symbolic deductive inference based system with statistical techniques leads to best of both the worlds where we can avoid extensive engineering effort required by symbolic systems without compromising the quality of generated programs, and at the same time provide significant performance (when measured as synthesis time) gains.

For future work, exploring better learning models for production rule selection and applying our technique to diverse and more powerful grammars should be important research directions.

A ROBUSTFILL PERFORMANCE WITH DIFFERENT BEAM SIZES

<|TLDR|>

@highlight

We integrate symbolic (deductive) and statistical (neural-based) methods to enable real-time program synthesis with almost perfect generalization from 1 input-output example.

@highlight

The paper presents a branch-and-bound approach to learn good programs where an LSTM is used to predict which branches in the search tree should lead to good programs

@highlight

Proposes system that synthesizes programs from a single example that generalize better than prior state-of-the-art