We introduce a neural network that represents sentences by composing their words according to induced binary parse trees.

We use Tree-LSTM as our composition function, applied along a tree structure found by a fully differentiable natural language chart parser.

Our model simultaneously optimises both the composition function and the parser, thus eliminating the need for externally-provided parse trees which are normally required for Tree-LSTM.

It can therefore be seen as a tree-based RNN that is unsupervised with respect to the parse trees.

As it is fully differentiable, our model is easily trained with an off-the-shelf gradient descent method and backpropagation.

We demonstrate that it achieves better performance compared to various supervised Tree-LSTM architectures on a textual entailment task and a reverse dictionary task.

Finally, we show how performance can be improved with an attention mechanism which fully exploits the parse chart, by attending over all possible subspans of the sentence.

Recurrent neural networks, in particular the Long Short-Term Memory (LSTM) architecture BID10 and some of its variants BID8 BID1 , have been widely applied to problems in natural language processing.

Examples include language modelling BID35 BID13 , textual entailment BID2 BID30 , and machine translation BID1 BID36 amongst others.

The topology of an LSTM network is linear: words are read sequentially, normally in left-to-right order.

However, language is known to have an underlying hierarchical, tree-like structure BID4 .

How to capture this structure in a neural network, and whether doing so leads to improved performance on common linguistic tasks, is an open question.

The Tree-LSTM network BID37 BID41 provides a possible answer, by generalising the LSTM to tree-structured topologies.

It was shown to be more effective than a standard LSTM in semantic relatedness and sentiment analysis tasks.

Despite their superior performance on these tasks, Tree-LSTM networks have the drawback of requiring an extra labelling of the input sentences in the form of parse trees.

These can be either provided by an automatic parser BID37 , or taken from a gold-standard resource such as the Penn Treebank BID18 .

BID39 proposed to remove this requirement by including a shift-reduce parser in the model, to be optimised alongside the composition function based on a downstream task.

This makes the full model non-differentiable so it needs to be trained with reinforcement learning, which can be slow due to high variance.

Our proposed approach is to include a fully differentiable chart parser in the model, inspired by the CYK constituency parser BID5 BID40 BID15 .

Due to the parser being differentiable, the entire model can be trained end-to-end for a downstream task by using stochastic gradient descent.

Our model is also unsupervised with respect to the parse trees, similar to BID39 .

We show that the proposed method outperforms baseline Tree-LSTM architectures based on fully left-branching, right-branching, and supervised parse trees on a textual entailment task and a reverse dictionary task.

We also introduce an attention mechanism in the spirit of BID1 for our model, which attends over all possible subspans of the source sentence via the parse chart.

Our work can be seen as part of a wider class of sentence embedding models that let their composition order be guided by a tree structure.

These can be further split into two groups: (1) models that rely on traditional syntactic parse trees, usually provided as input, and (2) models that induce a tree structure based on some downstream task.

In the first group, BID27 take inspiration from the standard Montagovian semantic treatment of composition.

They model nouns as vectors, and relational words that take arguments (such as adjectives, that combine with nouns) as tensors, with tensor contraction representing application BID6 .

These tensors are trained via linear regression based on a downstream task, but the tree that determines their order of application is expected to be provided as input.

BID32 and BID33 also rely on external trees, but use recursive neural networks as the composition function.

Instead of using a single parse tree, BID20 propose a model that takes as input a parse forest from an external parser, in order to deal with uncertainty.

The authors use a convolutional neural network composition function and, like our model, rely on a mechanism similar to the one employed by the CYK parser to process the trees.

BID23 propose a related model, also making use of syntactic information and convolutional networks to obtain a representation in a bottom-up manner.

Convolutional neural networks can also be used to produce embeddings without the use of tree structures, such as in BID14 .

BID3 propose an RNN that produces sentence embeddings optimised for a downstream task, with a composition function that works similarly to a shift-reduce parser.

The model is able to operate on unparsed data by using an integrated parser.

However, it is trained to mimic the decisions that would be taken by an external parser, and is therefore not free to explore using different tree structures.

introduce a probabilistic model of sentences that explicitly models nested, hierarchical relationships among words and phrases.

They too rely on a shift-reduce parsing mechanism to obtain trees, trained on a corpus of gold-standard trees.

In the second group, BID39 shows the most similarities to our proposed model.

The authors use reinforcement learning to learn tree structures for a neural network model similar to BID3 , taking performance on a downstream task that uses the computed sentence representations as the reward signal.

BID16 take a slightly different approach: they formalise a dependency parser as a graphical model, viewed as an extension to attention mechanisms, and hand-optimise the backpropagation step through the inference algorithm.

All the models take a sentence as input, represented as an ordered sequence of words.

Each word w i ∈ V in the vocabulary is encoded as a (learned) word embedding w i ∈ R d .

The models then output a sentence representation h ∈ R D , where the output space R D does not necessarily coincide with the input space R d .

Our simplest baseline is a bag-of-words (BoW) model.

Due to its reliance on addition, which is commutative, any information on the original order of words is lost.

Given a sentence encoded by embeddings w 1 , . . .

, w n it computes DISPLAYFORM0 where W is a learned input projection matrix.

An obvious choice for a baseline is the popular Long Short-Term Memory (LSTM) architecture of BID10 .

It is a recurrent neural network that, given a sentence encoded by embeddings w 1 , . . .

, w T , runs for T time steps t = 1 . . .

T and computes DISPLAYFORM0 where σ(x) = 1 1+e −x is the standard logistic function.

The LSTM is parametrised by the matrices W ∈ R 4D×d , U ∈ R 4D×D , and the bias vector b ∈ R 4D .

The vectors σ(i t ), σ(f t ), σ(o t ) ∈ R D are known as input, forget, and output gates respectively, while we call the vector tanh(u t ) the candidate update.

We take h T , the h-state of the last time step, as the final representation of the sentence.

Following the recommendation of BID12 , we deviate slightly from the vanilla LSTM architecture described above by also adding a bias of 1 to the forget gate, which was found to improve performance.

Tree-LSTMs are a family of extensions of the LSTM architecture to tree structures BID37 BID41 .

We implement the version designed for binary constituency trees.

Given a node with children labelled L and R, its representation is computed as DISPLAYFORM0 where w in (1) is a word embedding, only nonzero at the leaves of the parse tree; and h L , h R and c L , c R are the node children's h-and c-states, only nonzero at the branches.

These computations are repeated recursively following the tree structure, and the representation of the whole sentence is given by the h-state of the root node.

Analogously to our LSTM implementation, here we also add a bias of 1 to the forget gates.

While the Tree-LSTM is very powerful, it requires as input not only the sentence, but also a parse tree structure defined over it.

Our proposed extension optimises this step away, by including a basic CYK-style BID5 BID40 BID15 chart parser in the model.

The parser has the property of being fully differentiable, and can therefore be trained jointly with the Tree-LSTM composition function for some downstream task.

The CYK parser relies on a chart data structure, which provides a convenient way of representing the possible binary parse trees of a sentence, according to some grammar.

Here we use the chart as an efficient means to store all possible binary-branching trees, effectively using a grammar with only a single non-terminal.

This is sketched in simplified form in TAB0 for an example input.

The chart is drawn as a diagonal matrix, where the bottom row contains the individual words of the input sentence.

The n th row contains all cells with branch nodes spanning n words (here each cell is represented simply by the span -see FIG0 below for a forest representation of the nodes in all possible trees).

By combining nodes in this chart in various ways it is possible to efficiently represent every binary parse tree of the input sentence.

The unsupervised Tree-LSTM uses an analogous chart to guide the order of composition.

Instead of storing sets of non-terminals, however, as in a standard chart parser, here each cell is made up of a pair of vectors (h, c) representing the state of the Tree-LSTM RNN at that particular node in the tree.

The process starts at the bottom row, where each cell is filled in by calculating the Tree-LSTM output (1)-(3) with w set to the embedding of the corresponding word.

These are the leaves of the parse tree.

Then, the second row is computed by repeatedly calling the Tree-LSTM with the appropriate children.

This row contains the nodes that are directly combining two leaves.

They might not all be needed for the final parse tree: some leaves might connect directly to higher-level nodes, which have not yet been considered.

However, they are all computed, as we cannot yet know whether there are better ways of connecting them to the tree.

This decision is made at a later stage.

Starting from the third row, ambiguity arises since constituents can be built up in more than one way: for example, the constituent "neuro linguistic programming" in TAB0 can be made up either by combining the leaf "neuro" and the second-row node "linguistic programming", or by combining the second-row node "neuro linguistic" and the leaf "programming".

In these cases, all possible compositions are performed, leading to a set of candidate constituents (c 1 , h 2 ), . . .

, (c n , h n ).

Each is assigned an energy, given by DISPLAYFORM0 where cos(·, ·) indicates the cosine similarity function and u is a (trained) vector of weights.

All energies are then passed through a softmax function to normalise them, and the cell representation is finally calculated as a weighted sum of all candidates using the softmax output: DISPLAYFORM1 DISPLAYFORM2 The softmax uses a temperature hyperparameter t which, for small values, has the effect of making the distribution sparse by making the highest score tend to 1.

In all our experiments the temperature is initialised as t = 1, and is smoothly decreasing as t = 1 /2 e , where e ∈ Q is the fraction of training epochs that have been completed.

In the limit as t → 0 + , this mechanism will only select the highest scoring option, and is equivalent to the argmax operation.

The same procedure is repeated for all higher rows, and the final output is given by the h-state of the top cell of the chart.

The whole process is sketched in FIG0 for an example sentence.

Note how, for instance, the final sentence representation can be obtained in three different ways, each represented by a coloured circle.

All are computed, and the final representation is a weighted sum of the three, represented by the dotted lines.

When the temperature t in (5) reaches very low values, this effectively reduces to the single "best" tree, as selected by gradient descent.

All models are implemented in Python 3.5.2 with the DyNet neural network library BID26 at commit 25be489.

The code for all following experiments will be made available on the first author's website 1 shortly after the publication date of this article.

Performance on the development data is used to determine when to stop training.

Each model is trained three times, and the test set performance is reported for the model performing best on the development set.

The textual entailment model was trained on a 2.2 GHz Intel Xeon E5-2660 CPU, and took three days to converge.

The reverse dictionary model was trained on a NVIDIA GeForce GTX TITAN Black GPU, and took five days to converge.

In addition to the baselines already described in §3, for the following experiments we also train two additional Tree-LSTM models that use a fixed composition order: one that uses a fully left-branching tree, and one that uses a fully right-branching tree.

We test our model and baselines on the Stanford Natural Language Inference task BID2 , consisting of 570 k manually annotated pairs of sentences.

Given two sentences, the aim is to predict whether the first entails, contradicts, or is neutral with respect to the second.

For example, given "children smiling and waving at camera" and "there are children present", the model would be expected to predict entailment.

For this experiment, we choose 100D input embeddings, initialised with 100D GloVe vectors BID28 and with out-of-vocabulary words set to the average of all other vectors.

This results in a 100 × 37 369 word embedding matrix, fine-tuned during training.

For the supervised Tree-LSTM model, we used the parse trees included in the dataset.

For training we used the Adam optimisation algorithm BID17 , with a batch size of 16.Given a pair of sentences, one of the models is used to produce the embeddings s 1 , s 2 ∈ R 100 .

Following BID39 and BID3 , we then compute DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 where B ∈ R 3×200 and b ∈ R 3 are trained parameters.

TAB1 lists the accuracy and number of parameters for our model, baselines, as well as other sentence embedding models in the literature.

When the information is available, we report both the number of intrinsic model parameters as well as the number of word embedding parameters.

For other models these figures are based on the data from the SNLI website 2 and the original papers.

Attention is a mechanism which allows a model to soft-search for relevant parts of a sentence.

It has been shown to be effective in a variety of linguistic tasks, such as machine translation BID1 BID38 , summarisation BID29 , and textual entailment BID31 .In the spirit of BID1 , we modify our LSTM model such that it returns not just the output of the last time step, but rather the outputs for all steps.

Thus, we no longer have a single pair of vectors s 1 , s 2 as in FORMULA6 , but rather two lists of vectors s 1,1 , . . .

, s 1,n1 and s 2,1 , . . .

, s 2,n2 .

Then, we replace s 1 in (6) with DISPLAYFORM3 where f is the attention mechanism, with vector parameter a and matrix parameters A i , A s .

This can be interpreted as attending over sentence 1, informed by the context of sentence 2 via the vector s 2,n2 .

Similarly, s 2 is replaced by an analogously defined s 2 , with separate attention parameters.

We also extend the mechanism of BID1 to the Unsupervised Tree-LSTM.

In this case, instead of attending over the list of outputs of an LSTM at different time steps, attention is over the whole chart structure described in §3.4.

Thus, the model is no longer attending over all words in the source sentences, but rather over all their possible subspans.

The results for both attention-augmented models are reported in TAB2 .

Table 4 : Median rank (lower is better) and accuracies (higher is better) at 10 and 100 on the three test sets for the reverse dictionary task: seen words (S), unseen words (U), and concept descriptions (C).

We also test our model and baselines on the reverse dictionary task of BID9 , which consists of 852 k word-definition pairs.

The aim is to retrieve the name of a concept from a list of words, given its definition.

For example, when provided with the sentence "control consisting of a mechanical device for controlling fluid flow", a model would be expected to rank the word "valve" above other confounders in a list.

We use three test sets provided by the authors: two sets involving word definitions, either seen during training or held out; and one set involving concept descriptions instead of formal definitions.

Performance is measured via three statistics: the median rank of the correct answer over a list of over 66 k words; and the proportion of cases in which the correct answer appears in the top 10 and 100 ranked words (top 10 accuracy and top 100 accuracy).As output embeddings, we use the 500D CBOW vectors BID25 provided by the authors.

As input embeddings we use the same vectors, reduced to 256 dimensions with PCA.

Given a training definition as a sequence of (input) embeddings w 1 , . . .

, w n ∈ R 256 , the model produces an embedding s ∈ R 256 which is then mapped to the output space via a trained projection matrix W ∈ R 500×256 .

The training objective to be maximised is then the cosine similarity cos(Ws, d) between the definition embedding and the (output) embedding d of the word being defined.

For the supervised Tree-LSTM model, we additionally parsed the definitions with Stanford CoreNLP to obtain parse trees.

We use simple stochastic gradient descent for training.

The first 128 batches are held out from the training set to be used as development data.

The softmax temperature in (5) is allowed to decrease as described in §3.4 until it reaches a value of 0.005, and then kept constant.

This was found to have the best performance on the development set.

Table 4 shows the results for our model and baselines, as well as the numbers for the cosine-based "w2v" models of BID9 , taken directly from their paper.4 Our bag-of-words model consists of 193.8 k parameters; our LSTM uses 653 k parameters; the fixed-branching, supervised, and unsupervised Tree-LSTM models all use 1.1 M parameters.

On top of these, the input word embeddings consist of 113 123 × 256 parameters.

Output embeddings are not counted as they are not updated during training.

The results in TAB1 show a strong performance of the Unsupervised Tree-LSTM against our tested baselines, as well as other similar methods in the literature with a comparable number of parameters.

For the textual entailment task, our model outperforms all baselines including the supervised Tree-LSTM, as well as some of the other sentence embedding models in the literature with a higher number of parameters.

The use of attention, extended for the Unsupervised Tree-LSTM to be over all possible subspans, further improves performance.

In the reverse dictionary task, the poor performance of the supervised Tree-LSTM can be explained by the unusual tokenisation used in the dataset of BID9 : punctuation is simply stripped, turning e.g. "(archaic) a section of a poem" into "archaic a section of a poem", or stripping away the semicolons in long lists of synonyms.

On the one hand, this might seem unfair on the supervised Tree-LSTM, which received suboptimal trees as input.

On the other hand, it demonstrates the robustness of our method to noisy data.

Our model also performed well in comparison to the LSTM and the other Tree-LSTM baselines.

Despite the slower training time due to the additional complexity, FIG2 shows how our model needed fewer training examples to reach convergence in this task.

Following BID39 , we also manually inspect the learned trees to see how closely they match conventional syntax trees, as would typically be assigned by trained linguists.

We analyse the same four sentences they chose.

The trees produced by our model are shown in Figure 3 .

One notable feature is the fact that verbs are joined with their subject noun phrases first, which differs from the standard verb phrase structure.

However, formalisms such as combinatory categorial grammar BID34 , through type-raising and composition operators, do allow such constituents.

The spans of prepositional phrases in (b), (c) and (d) are correctly identified at the highest level; but only in (d) does the structure of the subtree match convention.

As could be expected, other features such as the attachment of the full stops or of some determiners do not appear to match human intuition.

We presented a fully differentiable model to jointly learn sentence embeddings and syntax, based on the Tree-LSTM composition function.

We demonstrated its benefits over standard Tree-LSTM on a textual entailment task and a reverse dictionary task.

Introducing an attention mechanism over the parse chart was shown to further improve performance for the textual entailment task.

The model is conceptually simple, and easy to train via backpropagation and stochastic gradient descent with popular deep learning toolkits based on dynamic computation graphs such as DyNet BID26 and PyTorch.

The unsupervised Tree-LSTM we presented is relatively simple, but could be plausibly improved by combining it with aspects of other models.

It should be noted in particular that (4), the function assigning an energy to alternative ways of forming constituents, is extremely basic and does not rely on any global information on the sentence.

Using a more complex function, perhaps relying on a mechanism such as the tracking LSTM in BID3 , might lead to improvements in performance.

Techniques such as batch normalization BID11 or layer normalization BID0 might also lead to further improvements.

In future work, it may be possible to obtain trees closer to human intuition by training models to perform well on multiple tasks instead of a single one, an important feature for intelligent agents to demonstrate BID21 .

Elastic weight consolidation BID19 has been shown to help with multitask learning, and could be readily applied to our model.

<|TLDR|>

@highlight

Represent sentences by composing them with Tree-LSTMs according to automatically induced parse trees.