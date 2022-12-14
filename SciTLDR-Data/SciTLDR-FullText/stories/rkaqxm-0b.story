Answering compositional questions requiring multi-step reasoning is challenging for current models.

We introduce an end-to-end differentiable model for interpreting questions, which is inspired by formal approaches to semantics.

Each span of text is represented by a denotation in a knowledge graph, together with a vector that captures ungrounded aspects of meaning.

Learned composition modules recursively combine constituents, culminating in a grounding for the complete sentence which is an answer to the question.

For example, to interpret ‘not green’, the model will represent ‘green’ as a set of entities, ‘not’ as a trainable ungrounded vector, and then use this vector to parametrize a composition function to perform a complement operation.

For each sentence, we build a parse chart subsuming all possible parses, allowing the model to jointly learn both the composition operators and output structure by gradient descent.

We show the model can learn to represent a variety of challenging semantic operators, such as quantifiers, negation, disjunctions and composed relations on a synthetic question answering task.

The model also generalizes well to longer sentences than seen in its training data, in contrast to LSTM and RelNet baselines.

We will release our code.

Compositionality is a mechanism by which the meanings of complex expressions are systematically determined from the meanings of their parts, and has been widely assumed in the study of both natural languages BID10 , as well as programming and logical languages, as a means for allowing speakers to generalize to understanding an infinite number of sentences.

Popular neural network approaches to question answering use a restricted form of compositionality, typically encoding a sentence word-by-word from left-to-right, and finally executing the complete sentence encoding against a knowledge source BID13 .

Such models can fail to generalize from training sentences in surprising ways.

Inspired by linguistic theories of compositional semantics, we instead build a latent tree of interpretable expressions over a sentence, recursively combining constituents using a small set of neural modules.

When tested on longer questions than are found in the training data, we find that our model achieves higher performance than baselines using LSTMs and RelNets.

Our approach resembles Montague semantics, in which a tree of interpretable expressions is built over the sentence, with nodes combined by a small set of composition functions.

However, both the structure of the sentence and the neural modules that handle composition are learned by end-to-end gradient descent.

To achieve this, we define the parametric form of small set of neural modules, and then build a parse chart over each sentence subsuming all possible trees.

Each node in the chart represents a span of text with a distribution over groundings (in terms of booleans and knowledge base nodes and edges), as well as a vector representing aspects of the meaning that have not yet been grounded.

The representation for a node is built by taking a weighted sum over different ways of building the node (similarly to BID9 ).Typical neural network approaches to grounded question answering first encode a question from left-to-right with a recurrent neural network (RNNs), and then evaluate the encoding against an encoding of the knowledge source (for example, a knowledge base or image) BID14 .

In contrast to classical approaches to compositionality, constituents of complex expressions are not given explicit interpretations in isolation.

For example, in Which cubes are large or green?, an RNN encoder will not explicitly build an interpretation for the expression large or green.

We show that A correct parse for a question given the knowledge graph on the right, using our model.

We show the type for each node, and its denotation in terms of the knowledge graph.

The words or and not are represented by vectors, which parameterize composition modules.

The denotation for the complete question represents the answer to the question.

Nodes here have types E for sets of entities, R for relations, V for ungrounded vectors, EV for a combination of entities and a vector, and φ for semantically vacuous nodes.

While we show only one parse tree here, our model builds a parse chart subsuming all trees.

such approaches can generalize poorly when tested on more complex sentences than they were trained on.

In contrast, our approach imposes strong independence assumptions that give a linguistically motivated inductive bias.

In particular, it enforces that phrases are interpreted independently of surrounding words, allowing the model to generalize naturally to interpreting phrases in different contexts.

In the previous example, large or green will be represented as a particular set of entities in a knowledge graph, and be intersected with the set of entities represented by the cubes node.

Another perspective on our work is as a method for learning the layouts of Neural Module Networks (NMNs) BID1 .

Work on NMNs has focused on how to construct the structure of the network, variously using rules, parsers and reinforcement learning BID0 BID3 .

Our end-to-end differentiable model jointly learns structures and modules by gradient descent.

Our task is to answer a question q = w 1..|q| , with respect to a Knowledge Graph (KG) consisting of nodes E (representing entities) and labelled directed edges R (representing relationship between entities).

In our task, answers are either booleans, or specific subsets of nodes from the KG.Our model builds a parse for the sentence, in which phrases are grounded in the KG, and a small set of composition modules are used to combine phrases, resulting in a grounding for the complete question sentence that answers the question.

For example, in FIG0 , the phrases not and cylindrical are interpreted as a function word and an entity set, and then not cylindrical is interpreted by computing the complement of the entity set.

The node at the root of the parse tree is the answer to the question.

We describe a compositional neural model that answers such questions by:1.

Grounding individual tokens in a Knowledge Graph.

Tokens can either be grounded as particular sets of entities and relations in the KG, as ungrounded vectors, or marked as being semantically vacuous.

For each word, we learn parameters that are used to compute a distribution over semantic types and corresponding denotations in a KG ( § 4.1).

2.

Combining representations for adjacent phrases into representations for larger phrases, using trainable neural composition modules ( § 3.2).

This produces a denotation for the phrase.

3.

Assigning a binary-tree structure to the question sentence, which determines how words are grounded, and which phrases are combined using which modules.

We build a parse chart subsuming all possible structures, and train a parsing model to increase the likelihood of structures leading to the correct answer to questions.

Different parses leading to a denotation for a phrase of type t are merged into an expected denotation, allowing dynamic programming ( § 4).

4.

Answering the question, with the most likely grounding of the phrase spanning the sentence.

Our model classifies spans of text into different semantic types to represent their meaning as explicit denotations or ungrounded vectors.

All phrases are assigned a distribution over semantic types.

The semantic type determines how a phrase is grounded, and which composition modules can be used to combine it with other phrases.

A phrase spanning w i..

j has a denotation w i..j t KG for each semantic type t. For example, in FIG0 , red thing corresponds to a set of entities, left corresponds to a set of relations, and not is treated as an ungrounded vector.

The semantic types we define can be classified into the three different categories.

Below we describe these semantic types and their corresponding representations.

Grounded Semantic Types: Spans of text that can be fully grounded in the KG.

Ungrounded Semantic Types: Spans of text whose meaning cannot be grounded in the KG.1.

Vector (V): This type is used for spans representing functions that cannot yet be grounded in the KG, for example words such as and or every.

These spans are represented using 4 different real-valued vectors v 1 ∈ R 2 , v 2 ∈ R 3 , v 3 ∈ R 4 , v 4 ∈ R 5 that are used to parameterize different composition modules described below in § 3.2.2.

Vacuous (φ φ φ): Spans that are considered semantically vacuous, but are necessary syntactically, e.g. of in left of a cube.

During composition, these nodes act as identity functions.

Partially-Grounded Semantic Types: Spans of text that can only be partially grounded in the knowledge graph, such as and red or are four spheres.

Here, we represent the span by a combination of a grounding and vectors, representing grounded and ungrounded aspects of meaning respectively.

The grounded component of the representation will typically combine with another fully grounded representation, and the ungrounded vectors will parameterize the composition module.

We define 3 semantic types of this kind: EV, RV and TV, corresponding to the combination of entities, relations and boolean groundings with an ungrounded vector.

Here, the word represented by the vectors can be viewed as a binary function, one of whose arguments has been supplied.

Next, we describe how we compose phrase representations (from § 3.1) to create representations for larger phrases.

We define a small number of generic composition modules, that take as input two constituents of text with their corresponding semantic representations (grounded representations and ungrounded vectors), and outputs the semantic type and corresponding representation of the larger constituent.

The composition modules are parameterized by the trainable word vectors.

These can be divided into several categories:Composition modules resulting in fully grounded denotations: Described in FIG3 .

DISPLAYFORM0 This module performs a function on a pair of soft entity sets, parameterized by the model's global parameter vector [w1, w2, b] to produce a new soft entity set.

The composition function for a single entity's resulting attention value is shown.

Such a composition module can be used to interpret compound nouns and entity appositions.

For example, the composition module shown above learns to output the intersection of two entity sets.

DISPLAYFORM1 This module performs a function on a soft entity set, parameterized by a word vector, to produce a new soft entity set.

For example, the word not learns to take the complement of a set of entities.

The entity attention representation of the resulting span is computed by using the indicated function that takes the v1 ∈ R 2 vector of the V constituent as a parameter argument and the entity attention vector of the E constituent as a function argument.

DISPLAYFORM2 DISPLAYFORM3 This module composes a set of relations (represented as a single soft adjacency matrix) and a soft entity set to produce an output soft entity set.

The composition function uses the adjacency matrix representation of the R-span and the soft entity set representation of the E-span.

This module maps a soft entity set onto a soft boolean, parameterized by word vector (v3).

The module counts whether a sufficient number of elements are in (or out) of the set.

For example, the word any should test if a set is non-empty.

EV + E → T: This module combines two soft entity sets into a soft boolean, which is useful for modelling generalized quantifiers.

For example, in is every cylinder blue, the module can use the inner sigmoid to test if an element ei is in the set of cylinders (p L e i ≈ 1) but not in the set of blue things (p R e i ≈ 0), and then use the outer sigmoid to return a value close to 1 if the sum of elements matching this property is close to 0.

This module maps a pair of soft booleans into a soft boolean using the v2 word vector to parameterize the composition function.

Similar to EV + E → E, this module facilitates modeling a range of boolean set operations.

Using the same functional form for different composition functions, allows our model to use the same ungrounded word vector (v2) for compositions that are semantically analogous.

DISPLAYFORM0 DISPLAYFORM1 This module composes a pair of soft set of relations to a produce an output soft set of relations.

For example, the relations left and above are composed by the word or to produce a set of relations such that entities ei and ej are related if either of the two relations exists between them.

The functional form for this composition is similar to EV + E → E and TV + T → T modules.

Composition with φ φ φ-typed nodes: Phrases with type φ φ φ are treated as being semantically transparent identity functions.

Phrases of any other type can combined with these with no change to their type or representation.

Composition modules resulting in partially grounded denotations: We define several simple modules that combine fully grounded phrases with ungrounded phrases, by deterministically taking the union of the representations, giving phrases with partially grounded representations ( § 3.1).

These modules are useful for when words act as binary functions; here they combine with their first argument.

For example, in FIG0 , or and not cylindrical combine to make a phrase containing both the vectors for or and the entity set for not cylindrical.

Here, we describe how our model classifies question tokens into different semantic type spans and compute their representations ( § 4.1), recursively uses the composition modules defined above to parse the question appropriately into a soft latent tree that provides the answer ( § 4.2).

The model is trained end-to-end using only question-answer supervision ( § 4.3).

Each token in the question sentence is assigned a distribution over the semantic types, and given a grounding for each type.

Tokens can only be assigned the E, R, V, and φ φ φ semantic types.

For example, the token cylindrical in the question in FIG0 is assigned a distribution over the 4 semantic types (one shown) and for the E type, the representation computed is the set of cylindrical entities.

Semantic Type Distribution for Tokens: To compute the semantic type distribution, our model represents each word w in the word vocabulary V, and each semantic type t using an embedding vector; v w , v t ∈ R d .

The semantic type distribution is assigned with a softmax: DISPLAYFORM0 Grounding for Tokens: For each of the four semantic type assignments for question tokens, we need to compute/assign their corresponding representations.1.

E-Type Representation:

Each entity e ∈ E, is represented using an embedding vector v e ∈ R d based on the concatenation of vectors for its properties.

For each token w, we use its word vector to find the probability of each entity being part of the E-Type grounding: DISPLAYFORM1 For example, in FIG0 , the word red will be grounded as all the red entities.2.

R-Type Representation:

Each relation r ∈ R, is represented using an embedding vector v r ∈ R d .

For each token w i in the question, we first compute a distribution over relations it could refer to, and then use this distribution to compute the expected adjacency matrix that forms the R-type representation for this token.

DISPLAYFORM2 DISPLAYFORM3 For example, the word left in FIG0 is grounded as the subset of edges with the label 'left'.3.

V-Type Representation: For each word w ∈ V, we learn four vectors v 1 ∈ R 2 , v 2 ∈ R 3 , v 3 ∈ R 4 , v 4 ∈ R 5 , and use these as the representation for words with the V-Type.

Representation: This type is used for semantically vacuous words, which do not require a representation.

To learn the correct structure for applying composition modules, we use a simple parsing model.

We build a parse-chart over the question encompassing all possible trees by applying all composition modules, similar to a standard CRF-based PCFG parser using the CKY algorithm.

Each node in the parse-chart, for each span w i..j of the question, is represented as a distribution over different semantic types with their corresponding representations.

This distribution is computed by weighing the different ways of composing the span's constituents.

Phrase Semantic Type Potential: Each node in the parse-chart is associated with a potential value ψ(i, j, t), that is the score assigned by the model to the t semantic type for the w i..j span.

This is computed from all possible ways to form the span w i..j with type t. For a particular composition of span w i..k of type t 1 and w k+1..j of type t 2 , using the t 1 + t 2 → t module, the score is: DISPLAYFORM0 where, f (t1+t2→t) x (i, j, k|q) are six feature functions; a trainable weight for each word per module in the vocabulary, that correspond to: f 1 : word that appears before the start of the span w i−1 ; f 2 : first word in the span w i ; f 3 : last word in the left constituent w k ; f 4 : first word in the right constituent w k+1 ; f 5 : last word in the right constituent w j ; and f 6 : word that appears after the span w j+1 .The token semantic type potential of w i , ψ(i, i, i, t 1 + t 2 → t), is the same as p(t|w i ) (Eq. 1).The final t-type potential of w i..j is computed by summing over scores from all possible compositions: DISPLAYFORM1 Combining Phrase Representations: To compute the span w i..j 's denotation with type t, DISPLAYFORM2 where w i..j t KG , is the t-type representation of the span w i..

j , w i..k..

j t1+t2→t KG is the representation resulting from the composition of w i..k with w k+1..

j using the t 1 + t 2 → t composition module.

Answer Grounding: By recursively computing the phrase semantic-type potentials and representations, we can infer the semantic type distribution of the complete question sentence (Eq. 8) and the resulting grounding for different semantic type t, w 1..|q| DISPLAYFORM3 The answer-type (boolean or subset of entities) for the question is computed using: DISPLAYFORM4 The corresponding grounding is w 1..|q| t * KG , which answers the question.

Given a dataset D of (question, answer, knowledge-graph) tuples, DISPLAYFORM0 , we train our model to maximize the log-likelihood of the correct answers.

Answers are either booleans, or specific subsets of entities from the KG.

We denote the semantic type of the answer as a t .

If the answer is boolean, a ∈ {0, 1}, otherwise is a subset of entities from the KG, i.e. a = {e j }.

The model's answer to a question is found by taking its representation of the complete question, containing a distribution over types and the representation for each type.

We maximize the following objective: DISPLAYFORM1 Questions with boolean answers DISPLAYFORM2 Questions with entity set answers FORMULA7 We also add L 2 -regularization for the scalar parsing features introduced in § 4.2.

We generate a dataset of question-answers based on the CLEVR dataset BID4 , which contains knowledge graphs containing attribute information of objects and relations between them.

We generate a new set of questions for this data, as existing questions contain some biases that can be exploited by models BID4 found that many spatial relation questions can be answered only using absolute spatial information and many long questions can be answered correctly without performing all steps of reasoning), and many questions are over 40 words long, which is intractable given that the size of our computation graph is cubic in the question length.

Future work should explore scaling our approach to longer questions.

We generate 75K questions for training and 37.5K for validation.

Our question set tests various challenging semantic operators.

These include conjunctions (e.g. Is anything red or is anything large?), negations (e.g. What is not spherical?), counts (e.g. Are five spheres green?), quantifiers (e.g. Is every red thing cylindrical?), and relations (e.g. What is left of and above a cube?).

We employ some simple tests to remove trivial biases from the dataset.

We create two test sets: one drawn from the same distribution as the training data (37.5K), and another containing longer questions than the training data (22.5K).Our COMPLEX QUESTIONS test set contains the same words and constructions, but chained into longer questions.

For example, it contains questions such as What is a cube that is right of a metallic thing that is beneath a blue thing?

and Are two red things that are above a sphere metallic?.

These questions require more multi-step reasoning to solve.

In this section we describe our experimentation setting, the baseline models we compare to, and the various experiments demonstrating the ability of our model to answer compositional questions referring to KG and its ability to generalize to unseen longer questions and new attribute combinations.

Here we describe the training details of our model and the baseline models.

Representing Entities: Each entity in the CLEVR dataset consists of 4 attributes.

For each attribute-value, we learn an embedding vector and concatenate the 4-embedding vectors to form the representation for the entity.

Training Details: Training the model is complicated by the large number of poor local minima, as the model needs to learn both good syntactic structures and the complex semantics of neural modules.

To simplify training, we use Curriculum Learning BID2 ) to pre-train the model on an easier subset of questions.

We use a 2-step schedule where we first train our model on simple attribute match (What is a red sphere?), attribute existence (Is anything blue?) and boolean composition (Is anything green and is anything purple?) questions and in the second step on all questions jointly.

TAB0 : Results for Short Questions:

Performance of our model compared to baseline models on the Short Questions test set.

The LSTM (NO KG) has accuracy close to chance, showing that the questions lack trivial biases.

Our model almost perfectly solves all questions showing its ability to learn challenging semantic operators, and parse questions only using weak end-to-end supervision.

We tune the hyper-parameters using validation accuracy.

We train using SGD with learning rate of 0.5 and mini-batch size of 4, regularization constant of 0.3.

When assigning the semantic type distribution to the words at the leaves, we add a small positive bias of +1 for φ φ φ-type and a small negative bias of −1 for the E-type score before the softmax.

Our trainable parameters are: question word embeddings (64-dimensional), relation embeddings (64-dimensional), entity attribute-value embeddings (16-dimensional), four vectors per word for V-type representations, six scalar feature scores per module per word for the parsing model, and the global parameter vector for the E+E→E module.

Baseline Models: We use three baseline models for comparison.

A simple LSTM (NO KG) model that encodes the question using an LSTM network and answers questions without access to the KG.

Another LSTM based model, LSTM (NO RELATION), that has access only to the entities of the KG but not the relationship information between them.

Finally, we train a RELATION NETWORK BID14 augmented model, which achieved state-of-the-art performance on the CLEVR dataset using image state descriptions.

Details about the baseline models are given in the Appendix section.

Short Questions Performance: In TAB0 , we see that our model is able to perfectly answer all the questions in the test set.

This demonstrates our model can learn challenging semantic operators using composition modules, as well as learn to parse the questions from only using weak endto-end supervision.

The RELATION NETWORK also achieves good performance, particularly on questions involving relations, but is weaker than our model on some question types.

The LSTM (NO RELATION) model also achieves good performance on questions not involving relations, which are out of scope for the model.

Table 2 shows results on complex questions, which are constructed by combining components of shorter questions.

We use the same models as in TAB0 , which were trained and developed only on shorter questions.

Answering longer questions requires complex multi-hop reasoning, and the ability to generalize from the language seen in its training data to new types of questions.

Results show that all baselines achieve close to random performance on this task, despite high accuracy for shorter questions.

This shows the challenges in generalizing RNN encoders beyond their training data.

In contrast, the strong inductive bias from our model structure allows the model to generalize to complex questions much more easily than RNN encoders.

Generalization to Unseen Attribute Combination: We also measure how well models generalize to unseen attribute combinations in knowledge graphs (using the COGENT subset of CLEVR).

For example, the test set contains 'blue spheres' that are not found in the training set.

None of the models showed a significant reduction in performance in this setting.

Error Analysis:

Analyzing the errors of our model, we find that most errors are due to incorrect assignments of structure, rather than semantic errors from the modules.

For example, in the question Are four red spheres beneath a metallic thing small?, our model produces a parse where it composes metallic thing small into a single node instead of composing red spheres beneath a metallic thing into a single node.

Future work should use more sophisticated parsing models.

Table 2 : Results for Complex Questions: All baseline models fail to generalize to questions requiring longer chains of reasoning than seen during training.

Our model substantially outperforms the baselines, showing its ability to perform complex multi-hop reasoning, and generalize from its training data.

Analysis suggests that most errors from our model are due to assigning incorrect structures, not mistakes by the composition modules.

Many approaches have been proposed to perform question-answering against structured knowledge sources.

Semantic parsing models have attempted to learn structures over pre-defined discrete operators, to produce logical forms that can be executed to answer the question.

Early work trained using gold-standard logical forms BID15 BID6 , whereas later efforts have only used answers to questions BID8 BID5 BID12 .

A key difference is that our model must learn semantic operators from data, which may be necessary to model the fuzzy interpretations of some function words like many or few.

Another similar line of work is neural program induction models, such as Neural Programmer BID11 and Neural Symbolic Machine BID7 .

These models learn to produce programs composed of predefined operators using weak supervision to answer questions against semi-structured tables.

Neural module networks have recently been proposed for learning semantic operators BID1 for question answering.

This model assumes that the structure of the semantic parse is given, and must only learn a set of operators.

Dynamic Neural Module Networks (D-NMN) extend this approach by selecting from a small set of candidate module structures BID0 .

In contrast, our approach learns a model over all possible structures for interpreting a question.

Our work is most similar to the most recently proposed N2NMN BID3 model, an end-toend version of D-NMN.

This model learns both semantic operators and the layout in which to compose them.

However, optimizing the layouts requires reinforcement learning, which is challenging due to the high variance of policy gradients, whereas our approach is end-to-end differentiable.

We use a LSTM network to encode the question as a vector q. We also define three other parameter vectors, t, e and b that are used to predict the answer-type P (a = T) = σ(q · t), entity attention value p ei = σ(q · e), and the probability of the answer being True p true = σ(q · b).

Similar to LSTM (NO RELATION), the question is encoded using a LSTM network as vector q. Similar to our model, we learn entity attribute-value embeddings and represent each entity as the concatenation of the 4 attribute-value embeddings, v ei .

Similar to LSTM (NO RELATION), we also define the t parameter vector to predict the answer-type.

The entity-attention values are predicted as p ei = σ(v ei · q).

To predict the probability of the boolean-type answer being true, we first add the entity representations to form b = ei v ei , then make the prediction as p true = σ(q · b).

The original formulation of the relation network module is as follows: DISPLAYFORM0 where e i , e j are the representations of the entities and q is the question representation from an LSTM network.

The output of the Relation Network module is a scalar score value for the elements in the answer vocabulary.

Since our dataset contains entity-set valued answers, we modified the module in the following manner.

We concatenate the object pair representations with the representations of the pair of directed relationships between them 1 .

We then use the Relation Network module to produce an output representation for each entity in the KB, in the following manner: DISPLAYFORM1 Similar to the LSTM baselines, we define a parameter vector t to predict the answer-type as: DISPLAYFORM2 P (a = E) = 1 − P (a = T)To predict the probability of the boolean type answer being true, we define a parameter vector b and predict as following: DISPLAYFORM3 To predict the entity-attention values, we use a separate attribute-embedding matrix to first generate the output representation for each entity, e out i , then predict the output attention values as follows: DISPLAYFORM4 We tried other architectures as well, but this modification provided the best performance on the validation set.

We also tuned the hyper-parameters and found the setting from BID14 to work the best based on validation accuracy.

We used a different 2-step curriculum to train the RELATION NETWORK module, in which we replace the Boolean questions with the relation questions in the first-schedule and jointly train on all questions in the subsequent schedule.

@highlight

We describe an end-to-end differentiable model for QA that learns to represent spans of text in the question as denotations in knowledge graph, by learning both neural modules for composition and the syntactic structure of the sentence.

@highlight

This paper presents a model for visual question answering that can learn both parameters and structure predictors for a modular neural network, without supervised structures or assistance from a syntactic parser.

@highlight

Proposes for training a question answering model from answers only and a KB by learning latent trees that capture the syntax and learn the semantic of words