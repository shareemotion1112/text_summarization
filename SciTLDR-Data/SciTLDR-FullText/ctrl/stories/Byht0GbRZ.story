Many tasks in natural language processing involve comparing two sentences to compute some notion of relevance, entailment, or similarity.

Typically this comparison is done either at the word level or at the sentence level, with no attempt to leverage the inherent structure of the sentence.

When sentence structure is used for comparison, it is obtained during a non-differentiable pre-processing step, leading to propagation of errors.

We introduce a model of structured alignments between sentences, showing how to compare two sentences by matching their latent structures.

Using a structured attention mechanism, our model matches possible spans in the first sentence to possible spans in the second sentence, simultaneously discovering the tree structure of each sentence and performing a comparison, in a model that is fully differentiable and is trained only on the comparison objective.

We evaluate this model on two sentence comparison tasks: the Stanford natural language inference dataset and the TREC-QA dataset.

We find that comparing spans results in superior performance to comparing words individually, and that the learned trees are consistent with actual linguistic structures.

There are many tasks in natural language processing that require comparing two sentences: natural language inference BID1 BID28 and paraphrase detection BID44 are classification tasks over sentence pairs, and question answering often requires an alignment between a question and a passage of text that may contain the answer BID39 BID31 BID17 .Neural models for these tasks almost always perform comparisons between the two sentences either at the word level BID29 ), or at the sentence level BID1 .

Word-level comparisons ignore the inherent structure of the sentences being compared, at best relying on a recurrent neural network such as an LSTM BID15 to incorporate some amount of context from neighboring words into each word's representation.

Sentence-level comparisons can incorporate the structure of each sentence individually BID2 BID36 , but cannot easily compare substructures between the sentences, as these are all squashed into a single vector.

Some models do incorporate sentence structure by comparing subtrees between the two sentences BID46 BID4 , but require pipelined approaches where a parser is run in a non-differentiable preprocessing step, losing the benefits of end-to-end training.

In this paper we propose a method, which we call structured alignment networks, to perform comparisons between substructures in two sentences, without relying on an external, non-differentiable parser.

We use a structured attention mechanism BID18 BID25 to compute a structured alignment between the two sentences, jointly learning a latent tree structure for each sentence and aligning spans between the two sentences.

Our method constructs a CKY chart for each sentence using the inside-outside algorithm BID27 , which is fully differentiable BID22 BID13 .

This chart has a node for each possible span in the sentence, and a score for the likelihood of that span being a constituent in a parse of the sentence, marginalized over all possible parses.

We take these two charts and find alignments between them, representing each span in each sentence with a structured attention over spans in the other sentence.

These span representations, weighted by the span's likelihood, are then used to compare the two sentences.

In this way we can perform comparisons between sentences that leverage the internal structure of each sentence in an end-to-end, fully differentiable model, trained only on one final objective.

We evaluate this model on several sentence comparison datasets.

In experiments on SNLI BID1 and TREC-QA (Voorhees & Tice, 2000) , we find that comparing sentences at the span level consistently outperforms comparing at the word level.

Additionally, and in contrast to prior work , we find that learning sentence structure on the comparison objective results in well-formed trees that closely mimic syntax.

Our results provide strong motivation for incorporating latent structure into models that implicitly or expliclty compare two sentences.

We first describe a common word-level comparison model, called decomposable attention BID29 .

This model was first proposed for the natural language inference task, but similar mechanisms have been used in many other places, such as for aligning question and passage words in the bi-directional attention model for question answering BID35 .

This model serves as our main point of comparison, as our latent tree matching model simply replaces the word-level comparisons done in decomposable attention with span comparisons.

The decomposable attention model consists of three steps: attend, compare, and aggregate.

As input, the model takes two sentences a and b represented by sequences of word embeddings DISPLAYFORM0 In the attend step, the model computes attention scores for each pair of tokens across the two input sentences and normalizes them as a soft alignment from a to b (and vice versa): DISPLAYFORM1 DISPLAYFORM2 where F is a feed-forward neural network, ?? i is the weighted summation of the tokens in b that are softly aligned to token a i and vice versa for ?? i .In the compare step, the input vectors a i and b i are concatenated with their corresponding attended vector ?? i and ?? i , and fed into a feed-forward neural network, giving a comparison between each word and the words it aligns to in the other sentence: DISPLAYFORM3 The aggregate step is a simple summation of v ai and v bj for each token in sentence a and b, and the two resulting fixed-length vectors are concatenated and fed into a linear layer, followed by a softmax layer for classification: DISPLAYFORM4 The decomposable attention model completely ignores the order and context of words in the sequence.

There are some efforts strengthening decomposable attention model with a recurrent neural

Boeing is a company based in WA B:Figure 1: Example span alignments of a sentence pair, where different colors indicate matching spans.

Note that some spans overlap, which cannot happen in a single tree; our model considers all possible span comparisons, weighted by the spans' marginal likelihood.network BID25 or intra-sentence attention BID29 .

However, these models amount to simply changing the input vectors a and b, and still only perform a token-level alignment between the two sentences.

Language is inherently tree structured, and the meaning of sentences comes largely from composing the meanings of subtrees BID6 .

It is natural, then, to compare the meaning of two sentences by comparing their substructures BID26 .

For example, when determining the relationship between "Boeing is in Seattle" and "Boeing is a company based in WA", the ideal units of comparison are spans determined by subtrees: "in Seattle" compared to "based in WA", etc. (see Figure 1 ).The challenge with comparing spans drawn from subtrees is that the tree structure of the sentence is latent and must be inferred, either during pre-processing or in the model itself.

In this section we present a model that operates on the latent tree structure of each sentence, comparing all possible spans in one sentence with all possible spans in the second sentence, weighted by how likely each span is to appear as a constituent in a parse of the sentence.

We use the non-terminal nodes of a binary constituency parse to represent spans.

Because of this choice of representation, we can use the nodes in a CKY parsing chart to efficiently marginalize span likelihood over all possible parses for each sentence, and compare nodes in each sentence's chart to compare spans between the sentences.

A constituency parser can be partially formalized as a graphical model with the following cliques BID20 : the latent variables c ijk for all i < j, indicating the span from the i-th token to the j-th token (span ij ) is a constituency node built from the merging of sub-node span ik and span (k+1)j .

Given the sentence x = [x i , ?? ?? ?? , x n ], the probability of a parse tree z is, DISPLAYFORM0 where Z represents all possible constituency trees for x.

The parameters to the graph-based CRF constituency parser are the unary potentials ?? i , reflecting the score of the token x i forming a unary constituency node and ?? ikj reflecting the score of span ij forming a binary constituency node with k as the splitting point.

It is possible to calculate the marginal probability of each constituency node p(c ijk = 1|x) using the inside-outside algorithm BID19 , and marginalize on the splitting points with p(s ij = 1|x) = i???k<j p(c ijk = 1|x) to compute the probability for a span ij being a constituency node.

The inside-outside algorithm is constrained to generate a binary tree; this is not a severe limitation, however, as most structures can be easily binarized BID11 .In a typical constituency parser, the score ?? ikj is parameterized according to the production rules of a grammar, e.g., with normalized categorical distributions for each non-terminal.

Our unlabeled grammar effectively has only a single production rule, however, so we instead parameterize these scores as multi-layer perceptrons operating on the representations of the subtrees being combined.

For computational and statistical efficiency given this parameterization, we drop the dependence on the splitting point in this score, resulting in a score for each span ?? ij representing how "constituent-like" the span is, independent of the merging of its children in the tree.

This allows for a slightly-modified computation of the inside score in the inside-outside algorithm.

Where the inside score ?? ij is typically computed as ?? ij = i???k<j ?? ikj ?? ik ?? (k+1)j , we instead compute it as DISPLAYFORM1 Up to this point, the tags of constituency nodes are not considered 1 , leading to an unlabeled tree structure.

However, with the binary tree constraint, not all tree nodes are syntactically complete, and thus some nodes may not be useful for comparison between the sentences.

To overcome this, we introduce two artificial tags T 0 and T 1 , where the former tag represents that this is a comparable constituent and the latter represents that this is just an intermediate node.

In other words, the T 1 tag gives the model a fallback option when the span should not be compared to other spans, but is still helpful to building the tree structure.

The inside pass is described in Algorithm 1, where ?? for i:=1 to n ??? width + 1 do 7:j := i + width ??? 1 8: DISPLAYFORM2 10: The ?? values are the inside scores for all the spans in the sentence, which are basically the unnormalized scores indicating the whether the spans are proper constituents.

After feeding these values into the outside algorithm, we can obtain the normalized marginal probability for each span DISPLAYFORM3 DISPLAYFORM4 When computing the unary and binary potentials ?? and ??, we use Long Short-Term Memory Neural Networks (LSTMs) BID15 and LSTM span features BID8 BID24 for representing all the spans.

We represent each sentence as a sequence of word embeddings [w sos , w 1 , ?? ?? ?? , w t , ?? ?? ?? , w n , w eos ].

We run a bidirectional LSTM over the sentence and obtain the output vector sequence DISPLAYFORM5 is the output vector for the t th token, and h t and h t are the output vectors from the forward and backward directions, respectively.

We represent a constituent c from position i to j with a span vector sp ij which is the concatenation of the vector differences h j+1 ??? h i and h i???1 ??? h j :And the potentials are computed by: DISPLAYFORM6 DISPLAYFORM7 where M LP T0 and M LP T1 are two multilayer perceptions with a scalar output and ReLU as the activation function for the hidden layer.

After applying the parsing process on two sentences, we will get the marginal probability for all potential spans of the two constituency trees, which can then be used for aligning.

After learning latent constituency trees for each sentence, we are able to do span-level comparisons between the two sentences, instead of the word-level comparisons done by the decomposable attention model.

The structure of these two comparison models are the same, but the basic elements of our structured alignment model are spans instead of words, and the marginal probabilities output from the inside-outside algorithm are used as a re-normalization value for incorporating structural information into the alignments.

For sentence a, with LSTM span features, we can obtain the representation for all potential spans, [sp The attention scores are computed between all pairs of spans across the two sentences, and the attended vectors can be calculated as: DISPLAYFORM0 DISPLAYFORM1 here the method is similar to the process in the decomposable attention model, but the basic elements are text spans instead of tokens, and the marginal probabilities output from the inside-outside algorithm are used as a re-normalization value for incorporating structural information into the alignments.

Then, the span vectors are concatenated with the attended vectors and fed into a feed-forward neural network: DISPLAYFORM2 DISPLAYFORM3 To aggregate these vectors, instead of using a direct summation, here we apply a weighted summation with the marginal probabilities as weights: DISPLAYFORM4 Here ?? a and ?? b work like the self-attention mechanism in BID23 to replace the summation pooling step.

The final output will still be obtained by a softmax function: DISPLAYFORM5

We evaluate our structured alignment model with two natural language matching tasks: question answering as sentence selection and natural language inference.

Since our approach can be considered as a module for replacing the widely-used token-level alignment, and can be plugged into other neural models, the experiments are not intended to show that our approach can beat state-of-theart baselines, but to test whether these methods can be trained effectively in an end-to-end fashion, can yield improvements over standard token-level alignment models, and can learn plausible latent constituency tree structures.

We first study the effectiveness of our model for answer sentence selection tasks.

Given a question, answer sentence selection is the task of ranking a list of candidate answer sentences based on their relatedness to the question, and the performance is measured by the mean average precision (MAP) and mean reciprocal rank (MRR).

We experiment on the TREC-QA dataset BID40 , in which all questions with only positive or negative answers are removed.

This leaves us with 1162 training questions, 65 development questions and 68 test questions.

Experimental results of the stateof-the-art models and our structured alignment model are listed in Table 1 , where the performances are evaluated with the standard TREC evaluation script.

The baseline model is the token-level decomposable attention strengthened with a bidirectional LSTM at the bottom for obtaining a contextualized representation for each token.

For selecting the answer sentences, we consider this as a binary classification problem and the final ranking is based on the predicted possibility of being positive.

We use 300-dimensional 840B GloVe word embeddings BID30 for initialization.

The hidden size for BiLSTM is 150 and the feed-forward neural networks F and G are two-layer perceptrons with ReLU as activation function and 300 as hidden size.

We apply dropout to the output of the BiLSTM and two-layer perceptrons with dropout ratio as 0.2.

All parameters (including word embeddings) were updated with Adagrad BID10 , and the learning rate was set to 0.05.

Since the structure of the question and the answer sentence may be different, we use two variants of the structured alignment model in the experiment; the first shares parameters for computing the structures and the second uses separate parameters.

Models MAP MRR QA-LSTM BID38 0.730 0.824 Attentive Pooling Network 0.753 0.851 Pairwise Word Interaction 0.777 0.836 Lexical Decomposition and Composition BID43 0.771 0.845 Noise-Contrastive Estimation BID32 0.801 0.877 BiMPM BID44 0.802 0.875 Decomposable Attention BID29 0.764 0.842 Structured Alignment (Shared Parameters) (ours) 0.770 0.850 Structured Alignment (Separated Parameters) (ours) 0.776 0.850 Table 1 : Results of our models (bottom) and others (top) on the TREC-QA test set.

From the results we can see that on both the MAP and MRR metrics, structured alignment models perform better than the decomposable attention model, showing that the structural bias is helpful for matching the question to the correct answer sentence.

Furthermore, the setting of separated parameters achieves higher scores on both metrics.

The second task we consider is natural language inference, where the input is two sentences, a premise and a hypothesis, and the goal is to predict whether the premise entails the hypothesis, contradicts the hypothesis, or neither.

For this task, we use the Stanford NLI dataset BID1 .

After removing sentences with unknown labels, we obtained 549,367 pairs for training, 9,842 for development and 9,824 for testing.

The baseline decomposable attention model is the same as in the question answering task.

The hidden size of the LSTM was set to 150.

We used 300-dimensional Glove 840B vectors to initialize the word embeddings.

All parameters (including word embeddings) were updated with Adagrad BID10 , and the learning rate was set to 0.05.

The hidden size of the two-layer perceptrons was Models Acc Classifier with handcrafted features BID1 78.2 LSTM encoders BID1 80.6 Stack-Augmented Parser-Interpreter Neural Net BID2 83.2 LSTM with inter-attention BID33 83.5 Matching LSTMs BID41 86.1 LSTMN with deep attention fusion BID5 86.3 Enhanced BiLSTM Inference Model BID3 88.0 Densely Interactive Inference Network BID12 88.0 Decomposable Attention BID29 85.8 Structured Alignment (ours) 86.6 Table 2 : Test accuracy on the SNLI dataset.set to 300 and dropout was used with ratio 0.2.

The structured alignment model in this experiment uses shared parameters for computing latent tree structures, since both the premise and hypothesis are declarative sentences.

The results of our experiments are shown in Table 2 .

Our structured alignment model gains almost a full point of accuracy (a 6% error reduction) over the baseline word-level comparison model with no additional annotation, simply from introducing a structural bias in the alignment between the sentences.

Table 2 shows the performances of the state-of-the-art models and our approaches.

Similar to the answer selection task, the tree matching model outperforms the decomposable model stably.

Here we give a brief qualitative analysis of the automatically learned tree structures.

We present the CKY charts for two randomly-selected sentences in the SNLI test set in FIG2 .

Recall that the CKY chart shows the likelihood of each span appearing as a constituent in the parse of the sentence, marginalized over all possible parses.

By looking at these span probabilities, we can see that the model learned a model of sentence structure that corresponds well to known syntactic structures.

In the first example, we can see that "five children playing soccer" is a very likely span, as is "chase after a ball".

Nonsensical spans, such as "playing soccer chase", have very low probability.

In the second example, we can see that the model can even resolve some attachment ambiguities correctly.

The prepositional phrase "at a large venue", which our model correctly identifies as a likely constituent in this sentence, has a very low score for attaching to "music" to form the constituent "music at a large venue".

Instead, the model (correctly) prefers to attach "at a large venue" to "playing", giving the span "playing music at a large venue".Our model is able to recover tree structures that very closely mimic syntax, without ever being given any access to syntactic supervision.

This is in contrast to prior work by , who were unable to learn syntax trees from a semantic objective.

We use the same supervision as their model; we hypothesize that the difference in result is that they were trying to learn tree structures for each sentence independently, only performing comparisons at the sentence level.

Comparing spans directly forces the model to induce trees with comparable constituents, giving the model a strong signal that was lacking in prior work.

Sentence comparison models: The Stanford natural language inference dataset BID1 , and the expanded multi-genre natural language inference dataset BID28 , are the most well-known recent sentence comparison tasks.

The literature of models addressing this comparison task is far too extensive to include here, though the recent shared task on Multi-NLI gives a good survey of sentence-level comparison models BID28 .

Some of these sentence-level comparison models do use sentence structure, obtained either latently (Bowman et al., 2016) or during pre-processing BID46 , but they squash all of the structure into a single vector, losing the ability to easily compare substructures between the two sentences.

For models doing a word-level comparison, the decomposable attention model, which we have discussed already in this paper BID29 , is the most salient example, though many similar models exist in the literature BID4 BID44 .

The idea of word-level alignments between a question and a passage of text is also pervasive in the recent question answering literature BID35 BID42 ).Finally, and most similar to our model, there have been many sentence comparison models proposed that directly compare subtrees between the two sentences BID4 BID46 .

However, all of these models are pipelined; they obtain the sentence structure in a non-differentiable preprocessing step, losing the benefits of end-to-end training.

Ours is the first model to allow comparison between latent tree structures, trained end-to-end on the comparison objective.

Structured attention: While it has long been known that inference in graphical models is differentiable BID22 BID9 , and using inference in, e.g., a CRF (laf, 2001) as the last layer in a neural network is common practice BID24 BID21 , including inference algorithms as intermediate layers in end-to-end neural networks is a recent development.

BID18 were the first to use inference to compute structured attentions over latent sentence variables, inducing tree structures trained on the end-to-end objective.

BID25 showed how to do this more efficiently, though their work was still limited to structured attention over a single sentence.

Our model is the first to include latent structured alignments between two sentences.

Inferring latent trees: Unsupervised grammar induction is a well-studied problem BID7 .

The most recent work in this direction was the Neural E-DMV model of .

While our goal is not to induce a grammar, we do produce a probabilistic grammar as a byproduct of our model.

Our results suggest that training on more complex objectives may be a good way to pursue grammar induction in the future; forcing the model to construct consistent, comparable subtrees between the two sentences is a strong signal for grammar induction.

We have considered the problem of comparing two sentences in natural language processing models.

We have shown how to move beyond word-and sentence-level comparison to comparing spans between the two sentences, without the need for an external parser.

Through experiments on several sentence comparison datasets, we have seen that span comparisons consistently outperform wordlevel comparisons, with no additional supervision.

We additionally found our model was able to discover latent tree structures that closely mimic syntax, without any syntactic supervision.

Our results have several implications for future work.

First, the success of span comparisons over word-level comparisons suggests that it may be profitable to include such comparisons in more complex models, either for comparing two sentences directly, or as intermediate parts of models for more complex tasks, such as reading comprehension.

Second, though we have not yet done a formal comparison with prior work on grammar induction, our model's ability to infer trees that look like syntax from a semantic objective is intriguing, and suggestive of future opportunities in grammar induction research.

Also, the speed of the model remains a problem, with the insideoutside algorithm involved, the speed of the full model will be be 15-20 times slower than the decomposable attention model, mainly due the the fact this dynamic programming method can not be effectively accelerated on a GPU.

<|TLDR|>

@highlight

Matching sentences by learning the latent constituency tree structures with a variant of the inside-outside algorithm embedded as a neural network layer.

@highlight

This paper introduces a structured attention mechanisms to compute alignment scores among all possible spans in two given sentences

@highlight

This paper proposes a model of structured alignments between sentences as a means of comparing sentences by matching their latent structures.