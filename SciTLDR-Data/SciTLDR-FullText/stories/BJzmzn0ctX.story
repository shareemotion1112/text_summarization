Reasoning over text and Knowledge Bases (KBs) is a major challenge for Artificial Intelligence, with applications in machine reading, dialogue, and question answering.

Transducing text to logical forms which can be operated on is a brittle and error-prone process.

Operating directly on text by jointly learning representations and transformations thereof by means of neural architectures that lack the ability to learn and exploit general rules can be very data-inefficient and not generalise correctly.

These issues are addressed by Neural Theorem Provers (NTPs) (Rocktäschel & Riedel, 2017), neuro-symbolic systems based on a continuous relaxation of Prolog’s backward chaining algorithm, where symbolic unification between atoms is replaced by a differentiable operator computing the similarity between their embedding representations.

In this paper, we first propose Neighbourhood-approximated Neural Theorem Provers (NaNTPs) consisting of two extensions toNTPs, namely a) a method for drastically reducing the previously prohibitive time and space complexity during inference and learning, and b) an attention mechanism for improving the rule learning process, deeming them usable on real-world datasets.

Then, we propose a novel approach for jointly reasoning over KB facts and textual mentions, by jointly embedding them in a shared embedding space.

The proposed method is able to extract rules and provide explanations—involving both textual patterns and KB relations—from large KBs and text corpora.

We show that NaNTPs perform on par with NTPs at a fraction of a cost, and can achieve competitive link prediction results on challenging large-scale datasets, including WN18, WN18RR, and FB15k-237 (with and without textual mentions) while being able to provide explanations for each prediction and extract interpretable rules.

The main focus in Artificial Intelligence is building systems that exhibit intelligent behaviour BID38 .

In particular, Natural Language Understanding (NLU) and Machine Reading (MR) aim at building models and systems with the ability to read text, extract meaningful knowledge, and actively reason with it BID18 BID45 .

This ability enables both the synthesis of new knowledge and the possibility to verify and update a given assertion.

For example, given the following statement:The River Thames is in the United Kingdom.

London is the capital and most populous city of England and the United Kingdom.

Standing on the River Thames in the south east of the island of Great Britain, London has been a major settlement for two millennia.a reader can verify that the statement is consistent since London is standing on the River Thames and London is in the United Kingdom.

Automated reasoning applied on text requires Natural Language Processing (NLP) tools capable of extracting meaningful knowledge from free-form text and compiling it into KBs BID51 .

However, the compiled KBs tend to be incomplete, ambiguous, and noisy, impairing the application of standard deductive reasoners BID29 .

Overall architecture of NaNTPs: the two main contributions consist in a faster inference mechanism (represented by the K-NN OR component, discussed in Section 3) and two dedicated encoders, one for KB facts and rules, and another for text (discussed in Section 4).A rich and broad literature in MR has approached this problem within a variety of frameworks, including Natural Logic BID41 BID2 and Semantic Parsing BID16 BID6 , and by framing the problem as Natural Language Inference-also referred to as Recognising Textual Entailment BID20 BID10 BID11 BID9 BID59 -and Question Answering .

However, such methods suffer from several limitations.

For instance, they rely on significant amounts of annotated data to suitably approximate the implicit distribution from which training and test data are drawn, and thus are often unable to generalise correctly in the absence of a sufficient quantity of training data or appropriate priors on model paramaters (e.g. via regularisation).

Orthogonally, even if accurate, such methods also cannot provide explanations for a given prediction BID19 BID44 .A promising strategy for overcoming these issues consists of combining neural models and symbolic reasoning, given their complementary strengths and weaknesses BID19 .

While symbolic models can generalise well from a small number of examples when the problem domain fits the inductive biases presented by the symbolic system at hand, they are brittle and prone to failure when the observations are noisy and ambiguous, or when the domain's properties are not known or formalisable, all of which being the case for natural language BID55 .

On the other hand, neural models are robust to noise and ambiguity but prone to overfitting BID44 and not easily interpretable BID40 , making them incapable of providing explanations or incorporating background knowledge.

Recent work in neuro-symbolic systems BID22 has made progress in learning neural representations that allow for comparison of symbols not on the basis of identity, but of their semantics (as learned in continuous representations of said symbols), while maintaining interpretability and generalisation, thereby inheriting the best of both worlds.

Among such systems, NTPs are end-to-end differentiable deductive reasoners based on Prolog's backward chaining algorithm, where unification between atoms is replaced by a differentiable operator computing their similarity between their embedding representations.

NTPs are especially interesting since they allow learning interpretable rules from data, by back-propagating a KB reconstruction error to the rule representations.

Furthermore, NTPs are explainable: by looking at the proof tree associated with the highest proof score, it is possible to know which rules are activated during the reasoning process-this enables providing explanations for a given reasoning outcome, performing error analysis, and driving modelling choices.

So far, due to their computational complexity, NTPs have only been successfully applied to learning tasks involving very small KBs.

However, most human knowledge is still stored in large KBs and natural language corpora, which are difficult to reason over automatically.

With this paper we aim at addressing these issues, by proposing:a) An efficient method for significantly reducing the time and space complexity required by NTPs by reducing the number of candidate proof scores and by using an attention mechanism for reducing the number of parameters required for learning new rules (Section 3), and b) An extension of NTPs towards text, by jointly embedding predicates and textual surface patterns in a shared embedding space by means of an efficient reading component (Section 4).

Mimicking backward chaining, NTPs recursively build a neural network enumerating all the possible proof states for proving a goal over the KB.

NTPs rely on three modules for building this neural network; the Unification module, which compares subsymbolic representations of symbols, and mutually recursive OR and AND modules, which jointly enumerate all the proof paths, before the final aggregation choses the single, highest scoring state.

We briefly overview these modules and the training procedure in the following.

Unification Module.

In backward chaining, unification is the operator that matches two atoms like locatedIn(LONDON, UK) and situatedIn(X, Y).

Discrete unification checks for equality between the elements of the atom (e.g. locatedIn = situatedIn) and binds variables to symbols via substitution (e.g. {X/LONDON, Y/UK}).

Unification in NTPs matches two atoms by comparing their embedding representations via a differentiable similarity function, which enables matching different symbols with similar semantics, such as locatedIn and situatedIn.

The unify θ (H, G, S) operator creates a neural network module that does exactly that.

Given two atoms H = [locatedIn, LONDON, UK] and G = [situatedIn, X, Y], and a proof state S = (S ψ , S ρ ) consisting of a set of substitutions S ψ and a proof score S ρ , the unify module compares the embedding representations θ locatedIn: and θ situatedIn: with a Radial Basis Function (RBF) kernel k, updates the variable binding substitution set S ψ = S ψ ∪ {X/LONDON, Y/UK}, and calculates the new proof score S ρ = min (S ρ , k (θ locatedIn: , θ situatedIn: )).

The resulting proof state S = (S ψ , S ρ ) is further expanded with the or and and modules.

AND Module.

The and module recursively tries to prove a list of sub-goals for a rule body.

Concretely, given the first sub-goal G and the following sub-goals G, the and K θ (G : G, d, S) module will substitute variables in G with constants according to the substitutions in S, and invoke the or module on G. The resulting state be used to prove the atoms in G, by recursively invoking the and module.

For example, when invoked on the rule body B mentioned above, the and module will first substitute variables with constants for the sub-goal [locatedIn, X, Z] and invoke the or module, whose resulting state will be the basis of the next invocation of and module on [locatedIn, Z, Y].Proof Aggregation.

After building a neural network that enumerates all the proof paths of the goal G on a KB K, NTPs select the proof path with the maximum proof score: DISPLAYFORM0 where d is a predefined maximum proof depth.

The initial proof state is set to (∅, 1), an empty substitution set, and a proof score of 1.Training In NTPs, predicate and constant embeddings are learned by optimising a cross-entropy loss on the final proof score, by iteratively masking facts in the KB and trying to prove them using available facts and rules .

Negative examples are sampled from the positive ones by corrupting the entities BID50 .Other than learning embeddings of predicates and constants, NTPs can also learn interpretable rules from data.

Scaling up Inference.

The model in Section 2 is capable of deductive reasoning, and the proof paths with the highest score can provide human-readable explanations for a given prediction.

However, a significant computational bottleneck lies in the or operator.

For instance, assume a KB K, composed of |K| facts and no rules.

The number of facts in a real-world KB can be quite large-for instance, Freebase contains 637 million facts BID17 , while the Google Knowledge Graph contains 18 billion facts BID50 .

Given a query G, in the absence of rules, NTP reduces to solving the following optimisation problem: DISPLAYFORM0 that is, it finds the fact F ∈ K in the KB K that, unified with the goal G, yields the maximum unification score.

Recall from Section 2 that the unification score between a fact DISPLAYFORM1 given by the similarity of their representations in a Euclidean space: DISPLAYFORM2 where k denotes a RBF kernel, and θ Gp: , θ Gs: , θ Go: ∈ R k (resp.

θ Fp: , θ Fs: , θ Fo: ∈ R k ) denote the embedding representation of the predicate, first and second argument of the goal G (resp.

fact F).Given a goal G, the NTPs proposed by will compute the unification score in Eq. 2 between G and every fact F ∈ K in the KB.

This is problematic, since computing the similarity between the representations of the goal G and every fact F ∈ K is computationally prohibitive-the number of comparisons is O(|K|n), where n is the number of goals and sub-goals in the proving process.

However, ntp K θ (G, d) only returns the single largest proof score, implying that every lower scoring proof is discarded during both inference and training.

One of the core contributions in this paper is to exactly compute ntp K θ (G, m) by only considering a subset of proof scores that contains the largest one.

Specifically, we make the following observation: given a goal G, if we know the most similar fact F ∈ K in embedding space as measured by unify θ , the number of comparisons needed for computing the final proof score ntp DISPLAYFORM3 The same reasoning can be extended to rules as well.

We argue that, given G, we can restrict the search of the closest fact F ∈ K to a Euclidean local neighbourhood of size n of G, DISPLAYFORM4 Then, the matching fact F will be very likely to be contained across the n most similar facts: DISPLAYFORM5 where the size of the neighbourhood is much lower than the size of the whole KB, i.e., |N K (G)| |K|.

The same idea can be extended from facts to rules, by selecting only the rules H :-B ∈ K such that their head H is closer to the goal.

However, finding the exact neighbourhood of a point in a Euclidean space is very costly, due to the curse of dimensionality BID30 .

Experiments showed that methods for identifying the exact neighbourhood can rarely outperform brute-force linear scan methods when dimensionality is high BID64 .A practical solution consists in Approximate Nearest Neighbour Search (ANNS) algorithms, which focus on finding an approximate solution to the k-nearest neighbour search problem outlined in Eq. 3 on high dimensional data.

Several families of ANNS algorithms exist, such as Locally-Sensitive Hashing BID0 , Product Quantisation BID31 BID32 , and Proximity Graphs BID42 .In this work, we use Hierarchical Navigable Small World (HNSW) BID43 , a graph-based incremental ANNS structure which can offer significantly better logarithmic complexity scaling during neighbourhood search than other approaches .

Specifically, given a subset of the KB P ⊆ K-for instance, containing all facts in K-we construct a HNSW graph for all elements in P, which has a O(|P| log |P|) time complexity.

Then, given a goal G, the HNSW graph is used for identifying its neighbourhood N P (G) within P, which has a O(log |P|) time complexity.

In our implementation, we construct the HNSW graph-based indexing structure when instantiating the model and, during training, we update the index every b batches.

Specifically, in our implementation, we generate a partitioning P ∈ 2 K of the KB K, where each element in P groups all facts and rules in K sharing the same signature.

Then, we redefine the or operator as follows: DISPLAYFORM6 where, instead of trying to unify a goal or sub-goal G with all facts and rule heads in the KB, we constrain the unification with ANNS to only facts and rule heads in its local neighbourhood N K (G).Improving Rule Learning via Attention.

Although NTPs can be used for learning interpretable rules from data, the solution proposed by can be quite data-inefficient, as the number of parameters associated to a rule can be quite large.

For instance, assume the rule DISPLAYFORM7 Such a rule introduces 3k parameters in the model, and it may be computationally inefficient to learn each of the embedding vectors.

We propose using an attention mechanism BID4 for attending over known predicates for defining the predicate embeddings θ p: , θ q: , θ r: .

Let R be the set of known predicates, and let R ∈ R |R|×k be a matrix representing the embeddings for the predicates in R. We define the θ p: as: DISPLAYFORM8 where a p: ∈ R |R| is a set of trainable attention weights associated with the predicate p.

This sensibly improves the parameter efficiency of the model in cases where the number of known predicates is low, i.e. |R| k, by introducing c|R| parameters for each rule rather than ck, where c is the number of trainable predicate embeddings in the rule.

In this section, we show we can use NaNTPs for jointly reasoning over KBs and natural language corpora.

In the following, we assume that our KB K is composed by facts, rules, and mentions.

A fact is composed by a predicate symbol and a sequence of arguments, e.g. [locationOf, LONDON, UK] .

On the other hand, a mention is a textual pattern between two co-occurring entities in the KB BID21 , such as "LONDON is located in the UK".We represent mentions jointly with facts and rules in K by considering each textual surface pattern linking two entities as a new predicate, and embedding it in a d-dimensional space by means of an end-to-end differentiable reading component.

For instance, the sentence "United Kingdom borders with Ireland" is translated into the following mention in K: [[[arg1], borders, with, [arg2] ], UK, IRELAND] by first identifying sentences or paragraphs containing KB entities, and then considering the textual surface pattern connecting such entities as an extra relation type.

While predicates in R are encoded by a look-up operation to a predicate embedding matrix R ∈ R |R|×k , textual surface patterns are encoded by an encode θ module.

The signature of encode θ is V * → R k , where V is the vocabulary of words and symbols occurring in textual surface patterns: it takes a sequence of tokens, and maps it to a k-dimensional embedding space.

More formally, given a textual surface pattern t borders, with, [arg2] ]-the encode θ module first encodes each token w in t by means of a token embedding matrix V ∈ R |V|×k , resulting in a pattern matrix W t ∈ R |t|×k .

Then, the module produces a textual surface pattern embedding vector θ t: ∈ R k from W t by means of an end-to-end differentiable encoder.

In this paper, we use a simple encode θ module that computes the average of the token embedding vectors composing a textual surface pattern: DISPLAYFORM0 DISPLAYFORM1 Albeit the encoder encode can be implemented by using other differentiable architectures, such as Recurrent Neural Networks (RNNs), we opted for a simple averaging model, for the sake of simplicity and efficiency, knowing that such a model performs on par or better than more complex models, thanks to a lower tendency to overfit to training data BID66 BID3 .

A significant corpus of literature aims at addressing the limitations of neural architectures in terms of generalisation and reasoning abilities.

A line of research consists of enriching neural network architectures with a differentiable external memory BID60 BID25 BID33 BID34 BID47 .

The underlying idea is that a neural network can learn to represent and manipulate complex data structures, thus disentangling the algorithmic part of the process from the representation of the inputs.

Another way of improving the generalisation and extrapolation abilities of neural networks consists of designing architectures capable of learning general, reusable programs-atomic primitives that can be reused across a variety of environments and tasks BID56 BID53 .

By doing so, it becomes also possible to train such models from enriched supervision signals, such as from program traces rather than simple input-output pairs.

Yet another line of work is differentiable interpreters-program interpreters where declarative or procedural knowledge, e.g., a sorting program, is compiled into a neural network architecture BID8 BID24 BID19 )-NTPs fall in this category.

This family of models allows imposing strong inductive biases on the models by partially defining the program structure used for constructing the network, e.g., in terms of instruction sets or rules.

A major problem with differentiable interpreters, however, is their computational complexity, that so far deemed them unusable except for smaller-scale learning problems.

This work is also related to BID54 , which use an approximate nearest neighbour data structure for sparsifying read operations in memory networks.

Furthermore, BID57 pioneered the idea of jointly embedding KB facts and textual mentions in a shared embedding space, by considering mentions as additional relations in a KB factorisation setting.

This idea was later extended to more elaborate mention encoders by BID46 .

Our work is also related to path encoding models BID12 and random walk approaches BID37 BID23 , which both lack rule induction mechanisms.

Lastly, our work is related to BID68 which is a scalable rule induction approach for knowledge base completion, but has not been applied to textual surface patterns.

We report the results of experiments on benchmark datasets -Countries BID7 , Nations, UMLS, and Kinship BID35 -following the same evaluation protocols as .

Furthermore, since our scalability improvements described in Section 3 allow us to experiment on significantly larger datasets, we also report results on the WN18 BID5 , WN18RR BID15 and FB15k-237 datasetswhose characteristics are outlined in TAB4 .

For evaluating our natural language reading component proposed in Section 4, we use FB15k-237.E -the FB15k-237 dataset augmented with a set of textual mentions for all entity pairs derived from ClueWeb12 with Freebase entity mention annotations BID21 -and a set of manually generated mentions BID63 BID5 ).

All datasets are described in detail in Appendix B.Baselines.

We compare NaNTPs with NTPs on benchmark datasets, and with DistMult BID67 and ComplEx BID63 , two state-of-the-art Neural Link Predictors used for identifying missing facts in potentially very large KBs, on WordNet and FreeBase.

For computing likelihoods of facts, DistMult and ComplEx embed each entity and relation type in a d-dimensional embedding space and use a differentiable scoring function based on the embeddings corresponding to the entity and relation of a fact.

Embedding representations and scoring function parameters are learned jointly by minimising a KB reconstruction error.

Note that, while the complexity of scoring a triple in DistMult and ComplEx is O(1)-they only need the embeddings of the symbols within a fact for computing the ranking score-instead in our model, it is O(log |K|).

For such a reason, instead of performing a full hyperparameter search, we fix some of the hyperparameters-i.e.we fix the embedding size d = 100, and train them for 100 epochs-and report results using these hyperparameters for NaNTPs.

For the sake of comparison, we also train ComplEx and DistMult while fixing d = 100 and the number of training epochs to 100, similarly to NaNTPs.

Figure 2: Given the Countries dataset, we replaced a varying number of training triples with mentions (see Appendix B.1.2 for details) and integrated the mentions using two different strategies: by encoding the mentions using the encoder introduced in Section 4 (Facts and Mentions) and by simply adding them to the KB (Facts).Experiments were conducted with the attention mechanism proposed in Section 3 (right) and the standard rule-learning procedure (left), each with 10 different random seeds.

We can see that, on each of the datasets, using the encoder yields consistently better AUC-PR values than simply adding the mentions to the KB.

Evaluation Performance Comparison In order to verify the correctness of our approximation, we compare the evaluation performance of NaNTP and NTP on the set of benchmark datasets presented in .

Results, presented in TAB3 show that NaNTP achieves on par or better results than NTP, consistently through benchmark datasets.

Please note that results reported in were calculated with an incorrect ranking function, which caused them to report artificially better ranking results.

Run-Time Performance comparison To assess the run-time gains of NaNTP, we compare it to NTP with respect to time and memory performance during training.

In our experiments, we vary the n of the ANNS approximation to assess the computational demands by increasing n. First, we compare the average number of examples (queries) per second by running 10 training batches with a maximum batch to fit the memory of NVIDIA GeForce GTX 1080 Ti, for all models.

Second, we compare the maximum memory usage of both models on a CPU, over 10 training batches with same batch sizes.

The comparison is done on a CPU to ensure that we include the size of the ANNS index in NaNTP measures and as a fail-safe, in case the model does not fit on the GPU memory.

The results, presented in FIG4 , demonstrate that, compared to NTP, NaNTP is considerably more time and memory efficiency.

In particular, we observe that NaNTP yields significant speedups of an order of magnitude for smaller datasets (Countries S1 and S2), and more than two orders of magnitude for larger datasets (Kinship and Nations).

Interestingly, with the increased size of the dataset, NaNTP consistently achieves higher speedups, when compared to NTP.

Similarly, NaNTP is more memory efficient, with savings bigger than an order of magnitude, making them readily applicable to larger datasets, even when augmented with textual surface forms.

Experiments with Generated Mentions.

For evaluating different strategies of integrating textual surface patterns, in the form of mentions, in NTPs, we proceeded as follows.

We replaced a varying number of training set triples from each of the Countries S1-3 datasets with human-generated textual mentions.

For instance, the fact neighbourOf(UK, IRELAND) may be replaced by the mention "UK is neighbouring with IRELAND".Then, we evaluate two ways of integrating textual mentions in NaNTPs, either by i) adding them as facts to the KB, or by ii) parsing the mention by means of an encoder, as described in Section 4.

The results, presented in Fig. 2 , testify that the proposed encoding module yields consistent improvements of the ranking accuracy in comparison to simply adding the mentions as facts to the KB.

This is particularly obvious in cases where the number of held-out facts is higher, implying that the added mentions can replace a large missing number of original facts in the KB.Explanations Involving Mentions.

NaNTPs are extremely efficient at learning rules involving both logic atoms and textual mentions.

For instance, by analysing the learned models and their explanations, we can see that NaNTPs learn patterns such as: DISPLAYFORM0 locatedIn(X, Y) :-"E 1 was a neighboring state to E 2 "(X, Z), "E 1 was located in E 2 "(Z, Y) locatedIn(X, Y) :-"E 1 can be found in E 2 "(X, Z), "E 1 is located in E 2 "(Z, Y)where E 1 and E 2 denote the position of the entities in the text surface patterns, and leverage them during their reasoning process, providing human-readable explanations for a given prediction.

Link prediction results are summarised in TAB1 shows a sample of explanations for the facts in the validation set of WN18 and WN18RR provided by NaNTPs by analysing the proof paths associated with the largest proof scores.

We can see that NaNTPs is capable of learning rules, such as has_part(X, Y) :-part_of(Y, X), and hyponym(X, Y) :-hypernym(Y, X).Interestingly, it is also able to find an alternative, non-trivial explanations for a given fact, based on the similarity between entity representations.

For instance, it can explain that CONGO is part of AFRICA by leveraging the similarity between AFRICA and AFRICAN_ COUNTRY, and the fact that the latter is a hyponym of CONGO.

It is also able to explain that CHAPLIN is a FILM_MAKER by leveraging the prior knowledge that CHAPLIN is a COMEDIAN, and the similarity between FILM_MAKER and COMEDIAN.

We analysed the effect of using attention for rule learning, introduced in Section 3, on NaNTP's accuracy, on both the benchmark datasets-outlined in TAB5 -and WordNet-outlined in TAB6 .

TAB5 shows that using attention in NaNTP for learning rule representations yields higher average ranking accuracy and lower variance on Countries S1-3 and Kinship, while yielding comparable results to not using attention on Nations and UMLS.

This is consistent with the observation in Fig. 2 , where NaNTPs with attention yield better performance on Countries S1-3.Results in TAB6 show the results of ablations on two large datasets derived from WordNet, namely WN18 and WN18RR.

For these two datasets, attention for learning rules greatly increases the ranking accuracy.

For instance Hits@10 increases from 83.2% to 93.7% in the case of WN18, and from 25% to 43.2% in the case of WN18RR.

Please note that state-of-the-art Neural Link Predictors such as ComplEx BID63 still yield an Hits@10 lower than 95% on WN18: this shows that WN18 yields results on par with other classes of models, while providing explanations for each prediction (as shown in Section 6.4).

Note that Neural Link Predictors are a class of model that was investigated for more than a decade now BID52 BID5 .Similarly, MRR increases from 0.539 to 0.769 in the case of WN18, and from 0.137 to 0.398 in the case of WN18RR.

An explanation for this phenomenon is that using attention drastically reduces the number of parameters required to learn each of the rule predicates from 100 to 18 in the case of WN18, and to 11 in the case of WN18RR, introducing an inductive bias that reveals being extremely beneficial in terms of ranking accuracy.

We can also note that, for FB15k-237, attention did not improve the ranking accuracy.

An explanation is that this dataset is very high relational (237 relation types), and using attention actually increased the number of parameters to be learned.

NTPs combine the strengths of rule-based and neural models but, so far, they were unable to reason over large KBs, and therefore over natural language.

In this paper, we proposed NaNTPs that utilise ANNS and attention as a solution to scaling issues of NTP.

By efficiently considering only the subset of proof paths associated with the highest proof scores during the construction of a dynamic computation graph, NaNTPs yield drastic speedups and memory efficiency, while yielding the same or a better predictive accuracy than NTPs.

This enables application of NaNTPs to mixed KB and natural language data by embedding logic atoms and textual mentions in a joint embedding space.

Albeit results are still slightly lower than those yielded by state-of-the-art Neural Link Predictors on large datasets, NaNTPs is interpretable and is able to provide explanations of its reasoning at scale.

In NTPs, the neural network structure is built recursively, and its construction is defined in terms of modules similarly to dynamic neural module networks BID1 .

Given a goal, a KB, and a current proof state as inputs, each module produces a list of new proof states, i.e., neural networks representing partial proof success scores and variable substitutions.

In the following we briefly overview the modules constituting NTPs.

In backward chaining, unification between two logic atoms is used for checking whether they can represent the same structure.

In discrete unification, non-variable symbols are checked for equality, and the proof fails if two symbols differ.

Rather than comparing symbols, NTPs compare their embedding representations by means of an end-to-end differentiable similarity function, such as a RBF kernel.

This allows matching different symbols with similar semantics, such as relations like locatedIn and situatedIn.

Unification is carried out by the unify operator, which updates a substitution set S, and creates a neural network for comparing the vector representations of non-variable symbols in two sequences of terms.

The signature of unify is L × L × S → S, where L is the domain of lists of terms: it takes two atoms, represented as lists of terms, and an upstream proof state, and returns a new proof state S .More formally, let H and G denote two lists of terms, each denoting a logical atom, such as [p, A, B] and [q, X, Y] for respectively denoting the atoms p(A, B) and q(X, Y).

Given a proof state S = (S ψ , S ρ ), where S ψ and S ρ respectively denote a substitution set and a proof score, unification is computed as follows: DISPLAYFORM0 with S = (S ψ , S ρ ) where: DISPLAYFORM1 Here, S refers to the new proof state, V refers to the set of variable symbols, h/g is a substitution from the variable symbol h to the symbol g, and θ g: denotes the embedding look-up of the non-variable symbol with index g.

For example, given two atoms H = [locatedIn, LONDON, UK] and G = [situatedIn, X, Y], the result of unify θ (H, G, (∅, 1)) with S = (∅, 1) will be a new substitution set S = (S ψ , S ρ ) where S ψ = {X/LONDON, Y/UK} and S ρ = min (1, k (θ locatedIn: , θ situatedIn: ))

Given a goal G, the or module unifies G with all facts and rules in a KB: for each rule H :-B ∈ K, after unifying G with the head H, it also attempts to prove the atoms in the body H by invoking the and module.

The signature of or is L × N × S → S N , where L is the domain of goal atoms, the second argument specifies the maximum proof depth, and N denotes the number of possible output proof states.

This operator is implemented as follows: DISPLAYFORM0 where H :-B denotes a rule in a given KB K with a head atom H and a list of body atoms B. DISPLAYFORM1 by first unifying the goal G with the head of the rule H, and then proving the sub-goals in the body of the rule by using and.

The and module recursively tries to prove a list of sub-goals, by invoking the or module.

Its signature is L × N × S → S N , where L is the domain of lists of atoms, and N is the number of possible output proof states for a list of atoms with a known structure and a provided KB.

This module is implemented as: DISPLAYFORM0 where substitute is an auxiliary function that applies substitutions to variables in an atom whenever possible.

Line 3 defines the recursion-the first sub-goal is proven by instantiating an or module after substitutions are applied, and every resulting proof state is used for proving the remaining sub-goals by instantiating an and module.

Finally, NTPs define the overall success score of proving a goal G using a KB K with parameters θ as:ntp DISPLAYFORM0 where d is a predefined maximum proof depth, and the initial proof state is set to (∅, 1), denoting an empty substitution set and a proof success score of 1.

The model is then trained using a Leave-One-Out cross-entropy loss-by removing one fact from the KB, and predicting its score.

Since this procedure only generates positive examples, negative examples are generated by corrupting positive examples, by randomly changing the entities BID50 .

We refer to

We run experiments on the following datasets-also outlined in TAB4 -and report results in terms of Area Under the Precision-Recall Curve BID14 ) (AUC-PR), MRR, and HITS@m BID5 .

Countries is a dataset introduced by BID7 for testing reasoning capabilities of neural link prediction models.

It consists of 244 countries, 5 regions (e.g. EUROPE), 23 sub-regions (e.g. WESTERN EUROPE, NORTH AMERICA), and 1158 facts about the neighbourhood of countries, and the location of countries and sub-regions.

As in , we randomly split a is adjacent to b, a borders with b, a is butted against b, a neighbours b, a is a neighbor of b, a is a neighboring country of b, a is a neighboring state to b, a was adjacent to b, a borders b, a was butted against b, a neighbours with b, a was a neighbor of b, a was a neighboring country of b, a was a neighboring state to b Table 4 : Mentions used for replacing a varying number of training triples in the Countries S1, S2, and S3 datasets.countries into a training set of 204 countries (train), a development set of 20 countries (validation), and a test set of 20 countries (test), such that every validation and test country has at least one neighbour in the training set.

Subsequently, three different task datasets are created, namely S1, S2, and S3.

For all tasks, the goal is to predict locatedIn(c, r) for every test country c and all five regions r, but the access to training atoms in the KB varies.

S1: All ground atoms locatedIn(c, r), where c is a test country and r is a region, are removed from the KB.

Since information about the sub-region of test countries is still contained in the KB, this task can be solved by using the transitivity rule locatedIn(X, Y) :-locatedIn(X, Z), locatedIn(Z, Y).

In addition to S1, all ground atoms locatedIn(c, s) are removed where c is a test country and s is a sub-region.

The location of countries in the test set needs to be inferred from the location of its neighbouring countries: locatedIn(X, Y) :-neighborOf(X, Z), locatedIn(Z, Y).

This task is more difficult than S1, as neighbouring countries might not be in the same region, so the rule above will not always hold.

In addition to S2, also all ground atoms locatedIn(c, r) are removed where r is a region and c is a country from the training set training that has a country from the validation or test sets as a neighbour.

The location of test countries can for instance be inferred using the rule locatedIn(X, Y) :-neighborOf(X, Z), neighborOf(Z, W), locatedIn(W, Y).

We generated a set of variants of Countries S1, S2, and S3, by randomly replacing a varying number of training set triples with mentions.

The employed mentions are outlined in Table 4 .

Furthermore, we consider the Nations, and the Unified Medical Language System (UMLS) datasets BID36 .

UMLS contains 49 predicates, 135 constants and 6529 true facts, while Nations contains 56 binary predicates, 111 unary predicates, 14 constants and 2565 true facts.

We follow the protocol used by and split every dataset into training, development, and test facts, with a 80%/10%/10% ratio.

For evaluation, we take a test fact and corrupt its first and second argument in all possible ways such that the corrupted fact is not in the original KB.

Subsequently, we predict a ranking of the test fact and its corruptions to calculate MRR and HITS@m.

Note that neither Countries nor UMLS and Nations have mentions.

For such a reason, for each of these datasets, we generated one equivalent mention-using a natural language sentence rather than a predicate name-and replaced a varying amount of training set triples with equivalent mentions.

BID5 ) is a subset of WordNet BID48 , a lexical KB for the English language, where entities correspond to word senses and relationships define lexical relations between them.

We also consider WN18RR BID15 , a dataset derived from WN18 where predicting missing links is sensibly harder.

For evaluating the impact of also using mentions, we use the FB15k-237.E dataset , a subset of FB15k BID5 ) that excludes redundant relations and direct training links for held-out triples, with the goal of making the task more realistic.

Textual relations for FB15k-237.E are extracted from 200 million sentences in the ClueWeb12 corpus, coupled with Freebase mention annotations BID21 , and include textual links of all co-occurring entities from the KB set.

After pruning 3 , there are 2.7 million unique textual relations that are added to the KB.

The number of relations and triples in the training, validation, and test portions of the data are given in TAB4 .

In particular, FB15k-237.E denotes the FB15k-237 dataset augmented with textual relations proposed by .

In order to grasp the magnitude of WN18, WN18RR and FB15k-237.E datasets, we provide their basic statistics in TAB4 C ADDITIONAL EXPERIMENTS C.1 ABLATION STUDIES The effect of attention over rules to this framework is quantised by two ablation studies, on benchmark datasets, in TAB5 , and on the large datasets, in TAB6 shows attention achieving higher or on-par performance with NaNTP without attention.

TAB5 , however, reports that NaNTP with attention significantly outperforms NaNTP without it on WordNet datasets.

In order to analyse the impact of using ANNS as a choice of heuristic, we ran additional experiments on the baseline datasets.

In particular, we compared ANNS to exact nearest neighbours search, since ANNS models may not return the exact nearest neighbours.

We also compared ANNS to random neighbour selection, for analysing the behaviour of the model with random neighbourhood choices.

Results are outlined in TAB7 : they show that the random neighbour, as expected, yield sensibly worse ranking results in comparison with ANNS.A surprising exception is Nations, where ranking results were apparently higher in comparison with UMLS and Kinship: a possible explanation is that Nations only contains 14 entities, so the random neighbourhood can sometimes correspond to the exact neighbourhood.

DistMult BID67 0 We can also observe that ANNS, yield very close ranking results in comparison with Exact NNS, but orders of magnitude faster.

This implies that, compared to a costly Exact NNS, ANNS is an optimal choice for a heuristic, since it greatly decreases the computational complexity of the method.

Please note that experiments with Exact NNS were extremely computationally demanding and, for such a reason, we limited the neighbourhood size k to k = 1.

@highlight

We scale Neural Theorem Provers to large datasets, improve the rule learning process, and extend it to jointly reason over text and Knowledge Bases.

@highlight

Proposes an extension of the Neural Theorem Provers system that addresses the main issues of this model by reducing the time and space complexity of the model

@highlight

Scales NTPs by using approximate nearest neighbour search over facts and rules during unification and suggests parameterizing predicates using attention over known predicates

@highlight

improves upon the previously proposed Neural Theorem Prover approach by using nearest neighbor search.