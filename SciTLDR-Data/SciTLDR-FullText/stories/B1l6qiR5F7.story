Natural language is hierarchically structured: smaller units (e.g., phrases) are nested within larger units (e.g., clauses).

When a larger constituent ends, all of the smaller constituents that are nested within it must also be closed.

While the standard LSTM architecture allows different neurons to track information at different time scales, it does not have an explicit bias towards modeling a hierarchy of constituents.

This paper proposes to add such inductive bias by ordering the neurons; a vector of master input and forget gates ensures that when a given neuron is updated, all the neurons that follow it in the ordering are also updated.

Our novel recurrent architecture, ordered neurons LSTM (ON-LSTM), achieves good performance on four different tasks: language modeling, unsupervised parsing, targeted syntactic evaluation, and logical inference.

Natural language has a sequential overt form as spoken and written, but the underlying structure of language is not strictly sequential.

This structure is usually tree-like.

Linguists agree on a set of rules, or syntax, that determine this structure BID10 BID11 BID46 and dictate how single words compose to form meaningful larger units, also called "constituents" BID30 .

The human brain can also implicitly acquire the latent structure of language BID14 : during language acquisition, children are not given annotated parse trees.

This observation brings more interest in latent structure induction with artificial neural network approaches, which are inspired by information processing and communication patterns in biological nervous systems.

From a practical point of view, integrating a tree structure into a neural network language model may be important for multiple reasons:(i) to obtain a hierarchical representation with increasing levels of abstraction, a key feature of deep neural networks BID1 BID34 BID48 ;(ii) to model the compositional effects of language BID30 BID54 and help with the long-term dependency problem BID1 BID56 by providing shortcuts for gradient backpropagation BID12 ;(iii) to improve generalization via a better inductive bias and at the same time potentially reducing the need of a large amount of training data.

The study of deep neural network techniques that can infer and use tree structures to form better representations of natural language sentences has received a great deal of attention in recent years BID5 BID61 BID50 BID24 BID9 BID58 BID51 .Given a sentence, one straightforward way of predicting the corresponding latent tree structure is through a supervised syntactic parser.

Trees produced by these parsers have been used to guide the composition of word semantics into sentence semantics BID54 BID4 , or even to help next word prediction given previous words BID59 .

However, supervised parsers are limiting for several reasons: i) few languages have comprehensive annotated data for supervised parser training; ii) in some domains, syntax rules tend to be broken (e.g. in tweets); and iii) languages change over time with use, so syntax rules may evolve.

On the other hand, grammar induction, defined as the task of learning the syntactic structure from raw corpora without access to expert-labeled data, remains an open problem.

Many such recent attempts suffer from inducing a trivial structure (e.g., a left-branching or right-branching tree BID58 ), or encounter difficulties in training caused by learning branching policies with Reinforcement Learning (RL) BID61 .

Furthermore, some methods are relatively complex to implement and train, like the PRPN model proposed in BID50 .Recurrent neural networks (RNNs) have proven highly effective at the task of language modeling BID41 BID39 .

RNNs explicitly impose a chain structure on the data.

This assumption may seem at odds with the latent non-sequential structure of language and may pose several difficulties for the processing of natural language data with deep learning methods, giving rise to problems such as capturing long-term dependencies BID1 , achieving good generalization BID4 , handling negation BID54 , etc.

Meanwhile, some evidence exists that LSTMs with sufficient capacity potentially implement syntactic processing mechanisms by encoding the tree structure implicitly, as shown by BID21 ; and very recently by BID33 .

We believe that the following question remains: Can better models of language be obtained by architectures equipped with an inductive bias towards learning such latent tree structures?In this work, we introduce ordered neurons, a new inductive bias for recurrent neural networks.

This inductive bias promotes differentiation of the life cycle of information stored inside each neuron: high-ranking neurons will store long-term information which is kept for a large number of steps, while low-ranking neurons will store short-term information that can be rapidly forgotten.

To avoid a strict division between high-ranking and low-ranking neurons, we propose a new activation function, the cumulative softmax, or cumax(), to actively allocate neurons to store long/short-term information.

We use the cumax() function to produce a vector of master input and forget gates ensuring that when a given neuron is updated (erased), all of the neurons that follow it in the ordering are also updated (erased).

Based on the cumax() and the LSTM architecture, we have designed a new model, ON-LSTM, that is biased towards performing tree-like composition operations.

Our model achieves good performance on four tasks: language modeling, unsupervised constituency parsing, targeted syntactic evaluation BID38 and logical inference BID4 .

The result on unsupervised constituency parsing suggests that the proposed inductive bias aligns with the syntax principles proposed by human experts better than previously proposed models.

The experiments also show that ON-LSTM performs better than standard LSTM models in tasks requiring capturing long-term dependencies and achieves better generalization to longer sequences.

There has been prior work leveraging tree structures for natural language tasks in the literature.

Socher et al. (2010) ; Alvarez-Melis & Jaakkola (2016); ; BID64 use supervised learning on expert-labeled treebanks for predicting parse trees.

BID54 and BID56 explicitly model the tree-structure using parsing information from an external parser.

Later, Bowman et al. (2016) exploited guidance from a supervised parser BID28 in order to train a stack-augmented neural network.

Theoretically, RNNs and LSTMs can model data produced by context-free grammars and contextsensitive grammars BID18 ).

However, recent results suggest that introducing structure information into LSTMs is beneficial. showed that RNNGs , which have an explicit bias to model the syntactic structures, outperform LSTMs on the subject-verb agreement task BID36 .

In our paper, we run a more extensive suite of grammatical tests recently provided by BID38 .

BID3 BID48 also demonstrate that tree-structured models are more effective for downstream tasks whose data was generated by recursive programs.

Interestingly, BID51 suggests that while the prescribed grammar tree may not be ideal, some sort of hierarchical structure, perhaps task dependent, might help.

However, the problem of efficiently inferring such structures from observed data remains an open question.

The task of learning the underlying grammar from data is known as grammar induction BID8 BID13 .

Early work incorporated syntactic structure in the context of language modeling BID45 BID6 BID7 .

More recently, there have been attempts at incorporating some structure for downstream tasks using neural models BID20 BID55 BID25 .

Generally, these works augment a main recurrent model with a stack and focus on solving algorithmic tasks.

focus on language modeling and syntactic evaluation tasks BID36 ) but they do not show the extent to which the structure learnt by the model align with gold-standard parse trees.

BID50 introduced the Parsing-Reading-Predict Networks (PRPN) model, which attempts to perform parsing by solving a language modeling task.

The model uses self-attention to compose previous states, where the range of attention is controlled by a learnt "syntactic distance".

The authors show that this value corresponds to the depth of the parse tree.

However, the added complexity in using the PRPN model makes it unwieldy in practice.

Another possible solution is to develop models with varying time-scales of recurrence as a way of capturing this hierarchy.

El BID16 ; Schmidhuber (1991); BID35 describe models that capture hierarchies at pre-determined time-scales.

More recently, BID31 proposed Clockwork RNN, which segments the hidden state of a RNN by updating at different time-scales.

These approaches typically make a strong assumption about the regularity of the hierarchy involved in modelling the data.

BID12 proposed a method that, unlike the Clockwork RNN, would learn a multi-scale hierarchical recurrence.

However, the model still has a pre-determined depth to the hierarchy, depending on the number of layers.

Our work is more closely related to BID44 , which propose to induce a hierarchy in the representation units by applying "nested" dropout masks: units are not dropped independently at random but whenever a unit is dropped, all the units that follow in the ordering are also dropped.

Our work can be seen as a soft relaxation of the dropout by means of the proposed cumax() activation.

Moreover, we propose to condition the update masks on the particular input and apply our overall model to sequential data.

Therefore, our model can adapt the structure to the observed data, while both Clockwork RNN and nested dropout impose a predefined hierarchy to hidden representations.

Given a sequence of tokens S = (x 1 , . . .

, x T ) and its corresponding constituency tree (Figure 2 (a)), our goal is to infer the unobserved tree structure while processing the observed sequence, i.e. while computing the hidden state h t for each time step t. At each time step, h t would ideally contain a information about all the nodes on the path between the current leaf node x t and the root S. In Figure 2 (c), we illustrate how h t would contain information about all the constituents that include the current token x t even if those are only partially observed.

This intuition suggests that each node in the tree can be represented by a set of neurons in the hidden states.

However, while the dimensionality of the hidden state is fixed in advance, the length of the path connecting the leaf to the root of the tree may be different across different time steps and sentences.

Therefore, a desiderata for the model is to dynamically reallocate the dimensions of the hidden state to each node.

Given these requirements, we introduce ordered neurons, an inductive bias that forces neurons to represent information at different time-scales.

In our model, high-ranking neurons contain long-term Figure 2 : Correspondences between a constituency parse tree and the hidden states of the proposed ON-LSTM.

A sequence of tokens S = (x 1 , x 2 , x 3 ) and its corresponding constituency tree are illustrated in (a).

We provide a block view of the tree structure in (b), where both S and VP nodes span more than one time step.

The representation for high-ranking nodes should be relatively consistent across multiple time steps.

(c) Visualization of the update frequency of groups of hidden state neurons.

At each time step, given the input word, dark grey blocks are completely updated while light grey blocks are partially updated.

The three groups of neurons have different update frequencies.

Topmost groups update less frequently while lower groups are more frequently updated.or global information that will last anywhere from several time steps to the entire sentence, representing nodes near the root of the tree.

Low-ranking neurons encode short-term or local information that only last one or a few time steps, representing smaller constituents, as shown in Figure 2 (b).

The differentiation between high-ranking and low-ranking neurons is learnt in a completely data-driven fashion by controlling the update frequency of single neurons: to erase (or update) high-ranking neurons, the model should first erase (or update) all lower-ranking neurons.

In other words, some neurons always update more (or less) frequently than the others, and that order is pre-determined as part of the model architecture.

In this section, we present a new RNN unit, ON-LSTM ("ordered neurons LSTM").

The new model uses an architecture similar to the standard LSTM, reported below: DISPLAYFORM0 The difference with the LSTM is that we replace the update function for the cell state c t with a new function that will be explained in the following sections.

The forget gates f t and input gates i t are used to control the erasing and writing operation on cell states c t , as before.

Since the gates in the LSTM act independently on each neuron, it may be difficult in general to discern a hierarchy of information between the neurons.

To this end, we propose to make the gate for each neuron dependent on the others by enforcing the order in which neurons should be updated.

To enforce an order to the update frequency, we introduce a new activation function: DISPLAYFORM0 where cumsum denotes the cumulative sum.

We will show that the vectorĝ can be seen as the expectation of a binary gate g = (0, ..., 0, 1, ..., 1).

This binary gate splits the cell state into two segments: the 0-segment and the 1-segment.

Thus, the model can apply different update rules on the two segments to differentiate long/short-term information.

Denote by d a categorical random variable representing the index for the first 1 in g: DISPLAYFORM1 The variable d represents the split point between the two segments.

We can compute the probability of the k-th value in g being 1 by evaluating the probability of the disjunction of any of the values before the k-th being the split point, that is DISPLAYFORM2 Since the categories are mutually exclusive, we can do this by computing the cumulative distribution function: DISPLAYFORM3 Ideally, g should take the form of a discrete variable.

Unfortunately, computing gradients when a discrete variable is included in the computation graph is not trivial BID49 , so in practice we use a continuous relaxation by computing the quantity p(d ≤ k), obtained by taking a cumulative sum of the softmax.

As g k is binary, this is equivalent to computing DISPLAYFORM4

Based on the cumax() function, we introduce a master forget gatef t and a master input gateĩ t : DISPLAYFORM0 DISPLAYFORM1 Following the properties of the cumax() activation, the values in the master forget gate are monotonically increasing from 0 to 1, and those in the master input gate are monotonically decreasing from 1 to 0.

These gates serve as high-level control for the update operations of cell states.

Using the master gates, we define a new update rule: DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 In order to explain the intuition behind the new update rule, we assume that the master gates are binary:• The master forget gatef t controls the erasing behavior of the model.

Supposef t = (0, . . . , 0, 1, . . . , 1) and the split point is d f t .

Given the Eq. FORMULA8 and FORMULA10 , the information stored in the first d f t neurons of the previous cell state c t−1 will be completely erased.

In a parse tree (e.g. Figure 2 (a)), this operation is akin to closing previous constituents.

A large number of zeroed neurons, i.e. a large d f t , represents the end of a high-level constituent in the parse tree, as most of the information in the state will be discarded.

Conversely, a small d f t represents the end of a low-level constituent as high-level information is kept for further processing.• The master input gateĩ t is meant to control the writing mechanism of the model.

Assume thatĩ t = (1, . . . , 1, 0, . . . , 0) and the split point is d i t .

Given Eq. FORMULA9 and FORMULA10 , a large d i t means that the current input x t contains long-term information that needs to be preserved for several time steps.

Conversely, a small d i t means that the current input x t just provides local information that could be erased byf t in the next few time steps.• The product of the two master gates ω t represents the overlap off t andĩ t .

Whenever an overlap exists (∃k, ω tk > 0), the corresponding segment of neurons encodes the incomplete constituents that contain some previous words and the current input word x t .

Since these constituents are incomplete, we want to update the information inside the respective blocks.

The segment is further controlled by the f t and i t in the standard LSTM model to enable more fine-grained operations within blocks.

For example, in Figure 2 , the word x 3 is nested into the constituents S and VP.

At this time step, the overlap gray blocks would represent these constituents, such thatf t andĩ t can decide whether to reset or update each individual neurons in these blocks.

As the master gates only focus on coarse-grained control, modeling them with the same dimensions as the hidden states is computationally expensive and unnecessary.

In practice, we setf t andĩ t to be D m = D C dimensional vectors, where D is the dimension of hidden state, and C is a chunk size factor.

We repeat each dimension C times, before the element-wise multiplication with f t and i t .

The downsizing significantly reduces the number of extra parameters that we need to add to the LSTM.

Therefore, every neuron within each C-sized chunk shares the same master gates.

We evaluate the proposed model on four tasks: language modeling, unsupervised constituency parsing, targeted syntactic evaluation BID38 , and logical inference BID4 .

Word-level language modeling is a macroscopic evaluation of the model's ability to deal with various linguistic phenomena (e.g. co-occurence, syntactic structure, verb-subject agreement, etc).

We evaluate our model by measuring perplexity on the Penn TreeBank (PTB) BID37 BID42 task.

For fair comparison, we closely follow the model hyper-parameters, regularization and optimization techniques introduced in AWD-LSTM BID41 .

Our model uses a three-layer ON-LSTM model with 1150 units in the hidden layer and an embedding of size 400.

For master gates, the downsize factor C = 10.

The total number of parameters was slightly increased from 24 millions to 25 millions with additional matrices for computing master gates.

We manually searched some of the dropout values for ON-LSTM based on the validation performance.

The values used for dropout on the word vectors, the output between LSTM layers, the output of the final LSTM layer, and embedding dropout where (0.5, 0.3, 0.45, 0.1) respectively.

A weight-dropout of 0.45 was applied to the recurrent weight matrices.

As shown in TAB0 , our model performs better than the standard LSTM while sharing the same number of layers, embedding dimensions, and hidden states units.

Recall that the master gates only control how information is stored in different neurons.

It is interesting to note that we can improve the performance of a strong LSTM model without adding skip connections or a significant increase in the number of parameters.

The unsupervised constituency parsing task compares the latent stree structure induced by the model with those annotated by human experts.

Following the experiment settings proposed in Htut et al. FORMULA4 , we take our best model for the language modeling task, and test it on WSJ10 dataset and WSJ test set.

WSJ10 has 7422 sentences, filtered from the WSJ dataset with the constraint of 10 words or less, after the removal of punctuation and null elements BID27 .

The WSJ test set contains 2416 sentences with various lengths.

It is worth noting that the WSJ10 test set contains sentences from the training, validation, and test set of the PTB dataset, while WSJ test uses the same set of sentences as the PTB test set.

To infer the tree structure of a sentence from a pre-trained model, we initialize the hidden states with the zero vector, then feed the sentence into the model as done in the language modeling task.

At each time step, we compute an estimate of d f t : DISPLAYFORM0 where p f is the probability distribution over split points associated to the master forget gate and D m is the size of the hidden state.

Givend f t , we can use the top-down greedy parsing algorithm proposed in BID50 for unsupervised constituency parsing.

We first sort the {d f t } in decreasing order.

For the firstd f i in the sorted sequence, we split the sentence into constituents ((x <i ), (x i , (x >i ))).

Then, we recursively repeat this operation for constituents (x <i ) and (x >i ), until each constituent contains only one word.

The performance is shown in TAB2 .

The second layer of ON-LSTM achieves state-of-the-art unsupervised constituency parsing results on the WSJ test set, while the first and third layers do not perform as well.

One possible interpretation is that the first and last layers may be too focused on capturing local information useful for the language modeling task as they are directly exposed to input tokens and output predictions respectively, thus may not be encouraged to learn the more abstract tree structure.

Since the WSJ test set contains sentences of various lengths which are unobserved during training, we find that ON-LSTM provides better generalization and robustness toward longer sentences than previous models.

We also see that ON-LSTM model can provide strong results for phrase detection, including ADJP (adjective phrases), PP (prepositional phrases), and NP (noun phrases).

This feature could benefit many downstream tasks, like question answering, named entity recognition, co-reference resolution, etc.

Targeted syntactic evaluation tasks have been proposed in BID38 .

It is a collection of tasks that evaluate language models along three different structure-sensitive linguistic phenomena: subject-verb agreement, reflexive anaphora and negative polarity items.

Given a large number of minimally different pairs of English sentences, each consisting of a grammatical and an ungrammatical sentence, a language model should assign a higher probability to a grammatical sentence than an ungrammatical one.

Using the released codebase 2 and the same settings proposed in BID38 , we train both our ON-LSTM model and a baseline LSTM language model on a 90 million word subset of Wikipedia.

Both language models have two layers of 650 units, a batch size of 128, a dropout rate of 0.2, a learning rate of 20.0, and were trained for 40 epochs.

The input embeddings have 200 dimensions and the output embeddings have 650 dimesions.

Table 3 shows that the ON-LSTM performs better on the long-term dependency cases, while the baseline LSTM fares better on the short-term ones.

This is possibly due to the relatively small num- BID57 .

PRPN models are evaluated on the WSJ test set BID22 .

We run the model with 5 different random seeds to calculate the average F1.

The Accuracy columns represent the fraction of ground truth constituents of a given type that correspond to constituents in the model parses.

We use the model with the best F1 score to report ADJP, NP, PP, and INTJ.

WSJ10 baselines are from Klein & Manning (2002, CCM) , Klein & Manning (2005, DMV+CCM) , and Bod (2006, UML-DOP) .

As the WSJ10 baselines are trained using POS tags, they are not strictly comparable with the latent tree learning results.

Italics mark results that are worse than the random baseline.

DISPLAYFORM0 ber of units in the hidden states, which is insufficient to take into account both long and short-term information.

We also notice that the results for NPI test cases have unusually high variance across different hyper-parameters.

This result maybe due to the non-syntactic cues discussed in BID38 .

Despite this, ON-LSTM actually achieves better perplexity on the validation set.

We also analyze the model's performance on the logical inference task described in BID4 .

This task is based on a language that has a vocabulary of six words and three logical operations, or, and, not.

There are seven mutually exclusive logical relations that describe the relationship between two sentences: two types of entailment, equivalence, exhaustive and non-exhaustive contradiction, and two types of semantic independence.

Similar to the natural language inference task, this logical inference task requires the model to predict the correct label given a pair of sentences.

The train/test split is as described in the original codebase 3 , and 10% of training set is set aside as the validation set.

We evaluate the ON-LSTM and the standard LSTM on this dataset.

Given a pair of sentences (s 1 , s 2 ), we feed both sentences into an RNN encoder, taking the last hidden state (h 1 , h 2 ) as the sentence embedding.

The concatenation of (h 1 , h 2 , h 1 • h 2 , abs(h 1 − h 2 )) is used as input to a multi-layer classifier, which gives a probability distribution over seven labels.

In our experiment, the RNN models were parameterised with 400 units in one hidden layer, and the input embedding size was 128.

A dropout of 0.2 was applied between different layers.

Both models are trained on sequences with 6 or less logical operations and tested on sequences with at most 12 operations.

Figure 3 shows the performance of ON-LSTM and standard LSTM on the logical inference task.

While both models achieve nearly 100% accuracy on short sequences (≤ 3), ON-LSTM attains Table 3 : Overall accuracy for the ON-LSTM and LSTM on each test case.

"Long-term dependency" means that an unrelated phrase (or a clause) exist between the targeted pair of words, while "shortterm dependency" means there is no such distraction.

Figure 3: Test accuracy of the models, trained on short sequences (≤ 6) in logic data.

The horizontal axis indicates the length of the sequence, and the vertical axis indicates the accuracy of models performance on the corresponding test set.

better performance on sequences longer then 3.

The performance gap continues to increase on longer sequences (≥ 7) that were not present during training.

Hence, the ON-LSTM model shows better generalization while facing structured data with various lengths and comparing to the standard LSTM.

A tree-structured model can achieve strong performance on this dataset BID4 , since it is provided with the ground truth structure as input.

The recursive application of the same composition function is well suited for this task.

We also include the result of RRNet BID24 , which can induce the latent tree structure from downstream tasks.

Note that the results may not be comparable, because the hyper-parameters for training were not provided.

In this paper, we propose ordered neurons, a novel inductive bias for recurrent neural networks.

Based on this idea, we propose a novel recurrent unit, the ON-LSTM, which includes a new gating mechanism and a new activation function cumax(·).

This brings recurrent neural networks closer to performing tree-like composition operations, by separately allocating hidden state neurons with long and short-term information.

The model performance on unsupervised constituency parsing shows that the ON-LSTM induces the latent structure of natural language in a way that is coherent with human expert annotation.

The inductive bias also enables ON-LSTM to achieve good performance on language modeling, long-term dependency, and logical inference tasks.

@highlight

We introduce a new inductive bias that integrates tree structures in recurrent neural networks.

@highlight

This paper proposes ON-LSTM, a new RNN unit that integrates the latent tree structure into recurrent models and that has good results on language modeling, unsupervised parsing, targeted syntactic evaluation, and logical inference.