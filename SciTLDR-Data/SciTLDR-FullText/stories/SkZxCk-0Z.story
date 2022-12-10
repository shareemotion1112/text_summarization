We introduce a new dataset of logical entailments for the purpose of measuring models' ability to capture and exploit the structure of logical expressions against an entailment prediction task.

We use this task to compare a series of architectures which are ubiquitous in the sequence-processing literature, in addition to a new model class---PossibleWorldNets---which computes entailment as a ``convolution over possible worlds''.

Results show that convolutional networks present the wrong inductive bias for this class of problems relative to LSTM RNNs, tree-structured neural networks outperform LSTM RNNs due to their enhanced ability to exploit the syntax of logic, and PossibleWorldNets outperform all benchmarks.

This paper seeks to answer two questions: "Can neural networks understand logical formulae well enough to detect entailment?", and, more generally, "Which architectures are best at inferring, encoding, and relating features in a purely structural sequence-based problem?".

In answering these questions, we aim to better understand the inductive biases of popular architectures with regard to structure and abstraction in sequence data.

Such understanding would help pave the road to agents and classifiers that reason structurally, in addition to reasoning on the basis of essentially semantic representations.

In this paper, we provide a testbed for evaluating some aspects of neural networks' ability to reason structurally and abstractly.

We use it to compare a variety of popular network architectures and a new model we introduce, called PossibleWorldNet.

Neural network architectures lie at the heart of a variety of applications.

They are practically ubiquitous across vision tasks BID19 BID17 BID29 and natural language understanding, from machine translation BID13 BID33 BID2 to textual entailment BID3 BID27 via sentiment analysis BID31 BID14 and reading comprehension BID9 BID25 .

They have been used to synthesise programs BID20 BID23 BID5 or internalise algorithms BID6 BID11 BID12 BID26 .

They form the basis of reinforcement learning agents capable of playing video games BID22 , difficult perfect information games BID28 BID35 , and navigating complex environments from raw pixels BID21 ).

An important question in this context is to find the inductive and generalisation properties of different neural architectures, particularly towards the ability to capture structure present in the input, an ability that might be important for many language and reasoning tasks.

However, there is little work on studying these inductive biases in isolation by running these models on tasks that are primarily or purely about sequence structure, which we intend to address.

The paper's contribution is three-fold.

First, we introduce a new dataset for training and evaluating models.

Second, we provide a thorough evaluation of the existing neural models on this dataset.

Third, inspired by the semantic (model-theoretic) definition of entailment, we propose a variant of the TreeNet that evaluates the formulas in multiple different "possible worlds", and which significantly outperforms the benchmarks.

The structure of this paper is as follows.

In Section 2, we introduce the new dataset and describe a generic data generation process for entailment datasets, which offers certain guarantees against the presence of superficial exploitable biases.

In Section 3, we describe a series of baseline models used to validate the dataset, benchmarks from which we will derive our analyses of popular model architectures, and also introduce our new neural model, the PossibleWorldNet.

In Section 4, we describe the structure of experiments, from which we obtained the results presented and discussed in Section 5.

We offer a brief survey of related work in Section 6, before making concluding remarks in Section 7.

Formal logics provide a symbolic toolkit for encoding and examining patterns of reasoning.

They are structural calculi aiming to codify the norms of correct thought.

The meanings of such statements are invariant to what the particular propositions stand for: to understand the entailment (p ∧ q) q, we only need to understand the semantics of-or related syntactic rules governing-a finite set of logical connectives, while p and q are meaningless arbitrary symbols selected to stand for distinct propositions.

In other words, the problem of determining whether an entailment holds is a purely structural sequence-based problem: to evaluate whether an entailment is true, only the meaning ofor inference rules governing-the connectives is relevant.

Everything else only has meaning via its place in the structure specified by an expression.

These qualities suggest that detecting logical entailment is an excellent task for measuring the ability of models to capture, understand, or exploit structure.

We present in this paper a generic process for generating entailment datasets, explained in detail in Appendix A, for any given logical system.

In the specific dataset-generated through this process-presented in this section, we will focus on propositional logic, which is decidable but requires a worst case of O(2 n ) operations (e.g. resolution steps, truth table rows), where n is the number of unique propositional variables, to verify entailment.

* D is composed of triples of the form (A, B, A B), where A and B are formulas of propositional logic, and A B is 1 if A entails B, and 0 otherwise.

For example, the data point (p ∧ q, q, 1) is positive because p ∧ q entails q, whereas (q ∨ r, r, 0) is negative because q ∨ r does not entail r. Entailment is primarily a semantic notion: A entails B if every model in which A is true is also a model in which B is true.

We impose various requirements on the dataset, to rule out superficial structural differences between D + and D − that can be easily exploited by "trivial" baselines † .

We impose the following high level constraints on our data through the generative process, explained in detail in Appendix A: our classes must be balanced, and formulas in positive and negative examples must have the same distribution over length.

Furthermore, we attempt to ensure that there are no recognisable differences in the distributions of lexical or syntactic features between the positive and negative examples.

It would not be acceptable, for example, if a typical B formula in a positive entailment (A, B, 1) had more disjunctions than a B formula in a negative entailment (A , B , 0).If we simply sample formulas A and B and evaluate whether A B, there are significant differences between the distributions of formulas for the positive and negative examples, which models can learn to exploit without needing to understand the structure of the problem.

To avoid these issues, we use a different approach, that satisfies the above requirements.

We sample 4-tuples of formulas (A 1 , B 1 , A 2 , B 2 ) such that: DISPLAYFORM0 Here, each of the four formulas appears in one positive entailment and one negative entailment.

This way, we minimise crude structural differences between the positive and negative examples.

Here is a simple example (although the actual dataset has much longer formulas) of such a 4-tuple of datapoints: BID32 ).

Then we search through the set of pairs, looking for pairs of pairs, (A 1 , B 1 ) and (A 2 , B 2 ), such that A 1 B 2 and A 2 B 1 .

We present, in Appendix A, the full details of this generative process, its constraints and guarantees, and how we used particular baselines to validate the data.

DISPLAYFORM1

We produced train, validation, and test (easy) by generating one large set of 4-tuples, and splitting them into groups of sizes 100000, 5000, and 5000.

The difficulty of evaluating an entailment depends on the number of propositional variables and the number of operators in the two formulas.

In training, validation, and test (easy), we sample the number of propositional variables uniformly between 1 and 10 (there are 26 propositional variables in total: a to z).

In test (hard), we sample uniformly between 5 and 10.

Our formula sampling method takes a parameter specifying the desired number of operators in the formula.

In training, validation, and test (easy), the number of operators in a formula is sampled uniformly between 1 and 10.

In our hard test set, the number of operators in a formula is sampled uniformly between 15 and 20.For the test (big) dataset, we sampled formulas using between 1 and 20 variables (uniformly), and between 10 and 30 operators (again, uniformly).

For test (massive), we used a different generating mechanism.

We first sampled pairs of formulas A, B such that A |= B. These had between 20 and 26 variables, and between 20 and 30 operators each.

Then we generated a B * by mutating B and checking that A B * .

See TAB0 for detailed statistics of the dataset sections, including the average difficulty (based on a complexity of O(2 # Vars )) of sequents in each fold.

The test (exam) dataset was assembled from 100 examples of logical entailment in the wild.

We looked through various logic textbooks for classic examples of entailments.

From these textbooks, we extracted true entailment triples (A, B, 1) where A |= B. We added false triples (A, B * , 0), by mutating B into B * and checking that A B * .In order to test models' ability to generalise to new unseen formulas, we pruned out cases where formulas seen in validation and test were α-equivalent (equivalent up to renaming of symbols) to formulas seen in training.

So, for example, if it had seen p |= (¬q ∧ p) in training, we did not want r |= (¬s ∧ r) to appear in either the test or validation sets.

To do this, we converted all formulas to de-Bruijn form (see BID24 , Chapter 6), and filtered out formulas in validation and test whose de-Bruijn form was identical to one of those in training.

This prevents the system from being able to simply memorise examples it has seen in training.

As discussed above, the logical connectives (∨, ∧, . . . ) are the only elements of the language in each dataset that have consistent implicit semantics across expressions.

In this sense, two entailments p ∧ q q and a ∧ b b should ideally be treated as identical by the model.

To encourage models to capture this invariance, we add an optional data processing layer during training (not testing) whereby symbols are consistently replaced by other symbols of the same type within individual entailments before being input to the network according to the process described below.

This is achieved by randomly sampling a permutation of a, . . .

, z (the propositional variables used) for every training example, and applying this permutation to the left and right sequents.

This process is analogous to augmenting image classification training with random reflections and crops.

In this section, we first describe a couple of baseline models that verify the basic difficulty of the dataset, followed by a description of benchmark models which are commonly used (with some variation) in a variety of problems, and finally by a description of our new model, PossibleWorldNet.

The classes in the dataset are balanced in training, validation, and both test sets, so a random baseline (and a constant, majority-class predicting baseline) will obtain an accuracy of 50% on the test sets.

We define two neural baselines which, we believe, should not be able to perform competitively on this task, but may do better than random.

The first is a linear bag of words (Linear BoW) model which embeds each symbol to a vector, and averages them, to produce a representation of each side of the sequent.

These representations are then passed through a linear layer: DISPLAYFORM0 The second is a similar architecture, where the final linear layer is replaced with a multi-layer perceptron (MLP BoW): DISPLAYFORM1 In both of these cases, the baselines are expected to have limited performance since they can only capture entailment by modelling the contribution of symbols individually, rather than by modelling structure, since the summation in g destroys all structural information (including word order).

We use these results to provide an indication of the difficulty of the dataset.

We present here a series of benchmark models, not only to serve the purpose of being grounds for comparison for new models tested against this dataset, but also to compare and contrast the performance of fairly ubiquitous model architectures on this purely syntactic problem.

We distinguish two categories of models: encoding models and relational models.

Encoding models, with exceptions specified below, jointly learn an encoding function f and an MLP, such that given a sequent A B, the model expresses DISPLAYFORM0 In this sense f produces a representation of each side of the sequent which contains all the information needed for the MLP to decide on entailment.

In contrast, relational models will observe the pair of expressions and make a decision, perhaps by traversing both expressions, or by relating substructure of one expression to that of the other.

These models express a more general formulation DISPLAYFORM1

The first encoder benchmark implemented is a Deep Convolutional Network Encoder (ConvNet Encoders), akin to architectures described in the convolutional networks for text literature BID14 Zhang et al., 2015; BID15 .

Here, the encoder function f is a stack of one dimensional convolutions over sequence symbols embedded by an embedding operation embedSeq, interleaved with max pooling layers every k layers (which is a model hyperparameter), followed by n (also a hyperparameter) fully connected layers: DISPLAYFORM0 The second and third encoder benchmarks are an LSTM BID10 encoder network (LSTM Encoders), and its bidirectional LSTM variant (BiDirLSTM Encoders).

For the LSTM encoder, we embed the sequence symbols, and run an LSTM RNN over them, ignoring the output until the final state: DISPLAYFORM1 For the bidirectional variant, two separate LSTM RNNs LSTM ← and LSTM → are run over the sequence in opposite directions.

Their respective final states are concatenated to form a representation of the expression: DISPLAYFORM2 and h DISPLAYFORM3 The benchmarks described thus far do not explicitly condition on structure, even when it is known, as they are designed to traverse a sequence from left to right and model dependencies in the data implicitly.

In contrast, we now consider encoder benchmarks which rely on the provision of the syntactic structure of the sequence they encode, and exploit it to determine the order of composition.

This inductive bias, which may be incorrect in certain domains (e.g., where no syntax is defined) or difficult to achieve in domains such as natural language text (where syntactic structure is latent and ambiguous), is easy to achieve for logic (where the syntax is known).

The experiments below will seek to demonstrate whether is a helpful inductive architectural bias.

The fourth and fifth encoding benchmarks are (tree) recursive neural networks BID34 BID18 BID35 BID1 , also known as TreeRNNs.

These recursively encode the logical expression using the parse structure ‡ , where leaf nodes of the tree (propositional variables) are embedded as learnable vectors, and each logical operator then combines one or more of these embedded values to produce a new embedding.

For example, the expression (¬a) ∨ b is parsed as the tree with leaves a and b, a unary node ¬ (with input the embedding of a), and a binary node ∨ (with inputs the embeddings of ¬a and b).

Following BID1 , the fourth encoding benchmark is a simple TreeRNN (TreeNet Encoders), where each operator 'op' concatenates its inputs to a vector x, and produces the output DISPLAYFORM4 The fifth and final encoding benchmark (TreeLSTM Encoders) is a variant of TreeRNNs which adapts LSTM cell updates.

This helps capture long range dependencies and propagate gradient within the tree.

Our implementation follows BID34 , modified to have per-op parameters as per TreeRNNs (see, also, the work by BID18 and Zhu et al. FORMULA17 ).

In addition to these encoding benchmarks, we define a pair of relational benchmarks, following BID27 .

We will traverse the entire sequent with LSTM RNNs or bidirectional LSTM RNNs but concatenating the left hand side and right hand side sequences into a single sequence separated by a held-out symbol (effectively standing for ).

For the LSTM variant (LSTM Traversal), the model is: DISPLAYFORM0 For the bidirectional case (BiDirLSTM Traversal), the extension is DISPLAYFORM1 and h → final = LSTM → (embedSeq(X)) ‡ Completely accurate parses of logical expressions are trivial to obtain, and these are provided to the model rather than learned.

We also benchmark the Transformer model, also known as Attention Is All You Need BID36 , which is a sequence-to-sequence model achieving state-of-the-art results in machine translation.

As in the relational LSTM models, we concatenate and embed the sequents, but instead of separating the sequents by a held-out symbol, we add a learnable bias to the right sequent in this embedding.

This augments the Transformer's method of adding timing signals to distinguishing symbols at different positions.

We then decode a sequence of length 1 and apply a linear transformation to get the final entailment prediction logits.

In this section, we introduce our new model.

Inspired by the semantic (model-theoretic) definition of entailment, we propose a variant on TreeNets that evaluates the pair of formulas in different "possible worlds".Entailment is, first and foremost, a semantic notion.

Given a set W of worlds, A |= B iff for every world w ∈ W, sat(w, A) implies sat(w, B)Here sat : W orld × F ormula →

Bool indicates whether a formula is satisfied in a particular world.

We shall first define a variant of sat that produces integers, and then define another variant that operates on real values.

First, define sat 2 : W orld × F ormula → {0, 1}: DISPLAYFORM0 Using sat 2 , we can redefine entailment as: DISPLAYFORM1 Assume we have a finite set of worlds W = {w 1 , ..., w n }; then we can recast as: DISPLAYFORM2 We are going to produce a relaxation of Proposition 1 by replacing sat 2 and ≤ with continuous functions.

Assume we have a variant of sat 2 that produces vectors of real values: DISPLAYFORM3 that generalises ≤ to vectors of real values.

Now we can rewrite as: DISPLAYFORM4 In our neural model, f is implemented by a simple linear layer using learnable weights W f and b f : DISPLAYFORM5 We use a set of random vectors to represent our worlds {w 1 , ..., w n }, where w i ∈ R k is a vector of length k of values drawn uniformly randomly.

We implement sat 3 using a simplified TreeNN (see Section 3.2) as described below.

Since sat 3 depends on the particular world w i we are currently evaluating, we add an additional parameter to the TreeNN so that the embedder has access to the current world w i .

We add an additional weight matrix W op 4 so that propositional variables can learn which aspect of the current world to focus on.

If the formula is of the form op(l, r), where op is nullary (a propositional variable), unary (e.g., negation), or binary (e.g., conjunction), and l and r are the embeddings of the constituents of the expression, then To evaluate whether A B, the PossibleWorldNet generates a set of imagined "worlds", and then evaluates A and B in each of those worlds.

It is a form of "convolution over possible worlds".

As we will see in Section 5, the quality of the model increases steadily as we increase the number of imagined worlds.

This architecture was inspired by semantic (model-theoretic) approaches to detecting entailment, but it does not encode any constraint on propositional logic in particular or formal logic in general.

The procedure of evaluating sentences in multiple worlds, and combining those evaluations in one product, is just what "entailment" means; so we speculate that an architecture like this should, in principle, be equally applicable to other logics (e.g., intuitionistic logic, modal logics, first-order logic) and also to non-formal entailments in natural language sentences.

Abstracting away from the particular interpretation of these vectors as "worlds", this method generates n copies of the model with shared weights, one for each vector w i ; each nullary operator learns a different projection on w i .

It makes predictions via a linear layer combining two representations, and then takes the product of the predictions as the overall prediction.

For each encoder benchmark architecture, the parameters of the encoders for the left and right hand sides of the sequent are shared.

The MLP which performs binary classification to detect entailment based on the expression representations produced by the encoders is model-specific (re-initialised for each model) and jointly trained.

Symbol embedding matrices are also model-specific, shared across encoders, and jointly trained.

We implemented all architectures in TensorFlow BID0 .

We optimised all models with Adam BID16 .

We grid searched across learning rates in [1e−5, 1e−4, 1e−3], minibatch sizes in [64, 128] , and trained each model thrice with different random seeds.

Per architecture, we grid-searched across specific hyperparameters as follows.

We searched across 2 and 3 layer MLPs wherever an MLP existed in a benchmark, and across layer sizes in [32, 64] for MLP hidden layers, embedding sizes, and RNN cell size (where applicable).

Additionally for convolutional networks, we searched across a number of convolutional layers in [4, 6, 8] , across kernel size in [5, 7, 9] , across number of channels in [32, 64] , and across pooling interval in [0, 5, 3, 1] (where 0 indicates no pooling).

For the Transformer model, we searched across the number of encoder and decoder layers in the range [6, 8, 10] , dropout probability in the range [0, 0.1, 0.5], and filter size in the range [128, 256, 384] .

Finally, for all models, we ran them with and without the symbol permutation data augmentation technique described in Section 2.2.As a result of the grid search, we selected the best model for each architecture against validation results, and record training, validation, and all test accuracies for the associated time step, which we present below.

Experimental results are shown in TAB1 .

The test scores of the best performing overall model are indicated in bold.

The test scores of the best performing model which does not have privileged access to the syntax or semantics of the logic (i.e. excluding TreeRNN-based models) are italicised.

The best benchmark test results are underlined.

We observe that the baselines are doing better than random (8.2 points above for the easy test set, for the MLP BoW, and 2.6 above random for the hard test set).

This indicates that there are some small number of exploitable regularities at the symbolic level in this dataset, but that they do not provide significant information.

The baseline results show that convolution networks and BiDirLSTMs encoders obtain relatively mediocre results compared to other models, as do LSTM and BiDirLSTM Traversal models.

LSTM encoders is the best performing model which does not have privileged access to the syntax trees.

Their success relative to BiDirLSTMs Encoders could be due to their reduced number of parameters guarding against overfitting, and rendering them easier to optimise, but it is plausible BiDirLSTMs Encoders would perform similarly with a more fine-grained grid search.

Both tree-based models take the lead amongst the benchmarks, with the TreeLSTM being the best performing benchmark overall on both test sets.

For most models except baselines, the symbol permutation data augmentation yielded 2-3 point increase in accuracy on weaker models (BiDirLSTM encoders and traversals, an convolutional networks) and between 7-15 point increases for the Tree-based models.

This indicates that this data augmentation strategy is particularly well fitted for letting structure-aware models capture, at the representational level, the arbitrariness of symbols indicating unbound variables.

Overall, these results show clearly that models that exploit structure in problems where it is provided, unambiguous, and a central feature of the task, outperform models which must implicitly model the structure of sequences.

LSTM-based encoders provide robust and competitive results, although bidirectionality is not necessarily always the obvious choice due to optimisation and overfitting problems.

Perhaps counter-intuitively, given the results of BID27 , traversal models do not outperform encoding models in this pair-of-sequences traversal problem, indicating that they may be better at capturing the sort of long-range dependencies need to recognise textual entailment better than they are at capturing structure in general.

We conclude, from these benchmark results, that tree structured networks may be a better choice for domains with unambiguous syntax, such as analysing formal languages or programs.

For domains such as natural language understanding, both convolutional and recurrent network architectures have had some success, but our experiments indicate that this may be due to the fact that existing tasks favour models which capture representational or semantic regularities, and do not adequately test for structural or syntactic reasoning.

In particular, the poor performance of convolutional nets on this task serves as a useful indicator that while they present the right inductive bias for capturing structure in images, where topological proximity usually indicates a joint semantic contribution (pixels close by are likely to contribute to the same "part" of an image, such as an edge or pattern), this inductive bias does not carry over to sequences particularly well (where dependencies may be significantly more sparse, structured, and distant) § .

The results for the transformer benchmark indicate that while this architecture can capture sufficient structure for machine translation, allowing for the appropriate word order in the output, and accounting for disambiguation or relational information where it exists within sentences, it does not capture with sufficient precision the more hierarchical structure which exists in logical expressions.

The best performing model overall is the PossibleWorldNet, which achieves significantly higher results than the other models, with 99.3% accuracy on test (easy), and 97.3% accuracy on test (hard).

This is as to be expected, as it has the strongest inductive bias.

This inductive bias has two components.

First, the model has knowledge of the syntactic structure of the expression, since it is a variant of a TreeNet.

Second, inspired by the definition of semantic (model-theoretic) entailment in § Related to this point, BID15 show that convolutional networks make for good character-level encoders, to produce word representations, which are in turn better exploited by RNNs.

This is consistent with our interpretation of our results, since at the character level, topological distance is-like for images-a good indicator of semantic grouping (characters that are close are usually part of the same word or n-gram).

general, the model evaluates the pair of formulas in lots of different situations ("possible worlds") and combines the various results together in a product ¶ .The quality of the PossibleWorldNet depends directly on the number of "possible worlds" it considers (see FIG1 .

As we increase the number of possible worlds, the validation error rate goes down steadily.

Note that the data-efficiency also increases as we increase the number of worlds.

This is because adding worlds to the model does not increase the number of model parameters-it just increases the number of different "possibilities" that are considered.

In propositional logic, of course, if we are allowed to generate every single truth-value assignment, then it is trivial to detect entailment by checking each one.

In our big test set, there are on average more than 3,000 possible truth-value assignments.

In our massive test set, there are on average over 800,000 possible assignments. (See TAB0 ).

The PossibleWorldNet considers at most 256 different worlds, which is only 7% of the expected total number of rows needed in the big test set, and only 0.03% of the expected number of rows needed for the massive test set.

To understand this result, we sample 32, 64, 128 and 256 truth table rows (variable truth-value assignments) for each pair of formulas in Test (hard), and reject entailment if a single evaluation for the formulas amongst these finds the left hand side to be true while the right hand side is false.

This gives us an estimate of the accuracy of sampling a number of truth table rows equal to the number of possible worlds in our model.

We estimate that these statistical methods have 75.9%, 86.5%, 93.4% and 97.2% chance of finding a countermodel, respectively.

This seems to indicate that PossibleWorldNet is capable of exploiting repeated computation across projections of random noise in order to learn, solely based on the label likelihood objective, something akin to a modelbased solution to entailment by treating the random-noise as variable valuations.6 RELATED WORK BID39 show how a neural architecture can be used to optimise matrix expressions.

They generate all expressions up to a certain depth, group them into equivalence classes, and train a recursive neural network classifier to detect whether two expressions are in the same equivalence class.

They use a recursive neural network BID30 to guide the search for an optimised equivalent expression.

There are two major differences between this work and ours.

First, the classifier is predicting whether two matrix expressions (e.g. A and (A T ) T ) compute the same values; this is an equivalence relation, while entailment is a partial order.

Second, their dataset consists of matrix expressions containing at most one variable, while our formulas contain many variables.

BID1 use a recursive neural network to learn whether two expressions are equivalent.

They tested on two datasets: propositional logic and polynomials.

There are two main differences between their approach and ours.

First, we consider entailment while they consider equivalence; equivalence is a symmetric relation, while entailment is not symmetric.

Second, we consider entailment as a relational classification problem: given a pair of expressions A and B, predict whether A entails B. In their paper, by contrast, they generate a set of k equivalence-classes of ¶ See Formula 2 above.

This general notion of entailment as truth-in-all-worlds is not dependent on any particular formal logic, and applies to entailment in both formal logics and natural languages.formulas with the same truth-conditions, and ask the network to predict which of these k classes a single formula falls into.

Their task is more specific: their network is only able to classify a formula from a new equivalence class that has not been seen during training if it has additional auxiliary information about that class (e.g. exemplar members of the class).Recognizing textual entailment (RTE) between natural language sentences is a central task in natural language processing. (See Dagan et al. (2006) ; for a recent dataset, see BID3 ).

Some approaches (e.g., BID37 and BID27 ) use LSTMs with attention, while others (e.g., BID38 ) use a convolutional neural network with attention.

Of course, recognizing entailment between natural language sentences is a very different task from recognizing entailment between logical formulas.

Evaluating an entailment between natural language sentences requires understanding the meaning of the non-logical terms in the sentence.

For example, the inference from "An ice skating rink placed outdoors is full of people" to "A lot of people are in an ice skating park" requires knowing the non-logical semantic information that an outdoors ice skating rink is also an ice skating park.

Current neural models do not always understand the structure of the sentences they are evaluating.

In BID3 , all the neural models they considered wrongly claimed that "A man wearing padded arm protection is being bitten by a German shepherd dog" entails "A man bit a dog".

We believe that isolating the purely structural sub-problem will be useful because only networks that can reliably predict entailment in a purely formal setting, such as propositional (or first-order) logic, will be capable of getting these sorts of examples consistently correct.

In this paper, we have introduced a new process for generating datasets for the purpose of recognising logical entailment.

This was used to compare benchmarks and a new model on a task which is primarily about understanding and exploiting structure.

We have established two clear results on the basis of this task.

First, and perhaps most intuitively, architectures which make explicit use of structure will perform significantly better than those which must implicitly capture it.

Second, the best model is the one that has a strong architectural bias towards capturing the possible world semantics of entailment.

In addition to these two points, experimental results also shed some light on the relative abilities of implicit structure models-namely LSTM and Convolution networkbased architectures-to capture structure, showing that convolutional networks may not present the right inductive bias to capture and exploit the heterogeneous and deeply structured syntax in certain sequence-based problems, both for formal and natural languages.

This conclusion is to be expected: the most successful models are those with the most prior knowledge about the generic structure of the task at hand.

But our dataset throws new light on this unsurprising thought, by providing a new data-point on which to evaluate neural models' ability to understand structural sequence problems.

Logical entailment, unlike textual entailment, depends only on the meaning of the logical operators, and of the place particular arbitrarily-named variables hold within a structure.

Here, we have a task in which a network's understanding of structure can be disentangled from its understanding of the meaning of words.

Xiang Zhang, Junbo Zhao, and Yann LeCun.

Character-level convolutional networks for text classification.

In Advances in neural information processing systems, pp.

649-657, 2015.Xiaodan Zhu, Parinaz Sobihani, and Hongyu Guo.

Long short-term memory over recursive structures.

In International Conference on Machine Learning, pp.

1604 Learning, pp.

-1612 Learning, pp. , 2015 A THE DATASET A.1 DATASET REQUIREMENTS Our dataset D is composed of triples of the form (A, B, A B) , where A B is 1 if A entails B, and 0 otherwise.

For example: (p ∧ q, q, 1) (q ∨ r, r, 0)We wanted to ensure that simple baseline models are unable to exploit simple statistical regularities to perform well in this task.

We define a series of baseline models which, due to their structure or the information they have access to, should not be able to solve the entailment recognition problem described in this paper.

We distinguish baselines for which we believe there is little chance of them detecting entailment, from those for which there categorically cannot be true modelling of entailment.

The baselines which categorically cannot detect entailment are encoding models which only observe one side of the sequent: DISPLAYFORM0 where f is a linear bag of words encoder, an MLP bag of words encoder, or a TreeNet.

Because the dataset contains a roughly balanced number of positive and negative examples, it follows that we should expect any model which only sees part of the sequent to perform in line with a random classifier.

If they outperform a random baseline on test, there is a structural or symbolic regularity on one side (or both) which is sufficient to identify some subset of positive or negative examples.

We use these baselines to verify the soundness of the generation process.

Let D + and D − be the positive and negative entailments: DISPLAYFORM1 We impose various requirements on the dataset, to rule out superficial syntactic differences between D + and D − that can be easily exploited by the simple baselines described above.

We require that our classes are balanced: and , we are guaranteed to produce balanced classes.

Unfortunately, this straightforward approach generates datasets that violate most of our requirements above.

See TAB3 for the details.

DISPLAYFORM2 In particular, the mean number of negations, conjunctions, and disjunctions at the top of the syntax tree (num at(·, 0, op)) is markedly different.

A + has significantly more conjunctions at the top of the syntax tree than A − , while B + has significantly fewer than B − .

Conversely, A + has significantly fewer disjunctions at the top of the syntax tree than A − , while B + has significantly more than DISPLAYFORM3 The mean number of satisfying truth-value assignments (sat(·)) is also markedly different: A + is true in on average 3.7 truth-value assignments (i.e. it is a very specific formula which is only true under very particular circumstances), while A − is true in 10.3 truth-value assignments (i.e. it is true in a wider range of circumstances).

We can use these statistics to develop simple heuristic baselines that will be unreasonably effective on the dataset described above: we can estimate whether A B by comparing the lengths of A and B, or by looking at the number of variables in B that do not appear in A, or by looking at the topmost connective in A and B. In order to satisfy our requirements above, we took a different approach to dataset generation.

In order to ensure that there are no crude statistical measurements that can detect differences between D + and D − , we change the generation procedure so that every formula appears in both D + and D − .

We sample 4-tuples of formulas FIG1 , B 2 ) such that: DISPLAYFORM4 Here, each of the four formulas appears in one positive entailment and one negative entailment * * .Using this alternative approach, we are able to satisfy the requirements above.

By construction, the mean length, number of operators at a certain level in the syntax tree, and the number of satisfying truth-value assignments is exactly the same for D + and D − .

See Table 4 .

DISPLAYFORM5 (r → c) → ((r → v) ∨ p) * * One consequence of this method is that it rules out A1 from being impossible (if it was impossible, we would not have A1 B2) and B1 from being a tautology (if it was a tautology, we would not have A2 B1).

@highlight

We introduce a new dataset of logical entailments for the purpose of measuring models' ability to capture and exploit the structure of logical expressions against an entailment prediction task.

@highlight

The paper proposes a new model to use deep models for detecting logical entailment as a product of continuous functions over possible worlds.

@highlight

Proposes a new model designed for machine learning with predicting logical entailment.