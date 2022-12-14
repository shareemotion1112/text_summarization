We introduce a new type of deep contextualized word representation that models both (1) complex characteristics of word use (e.g., syntax and semantics), and (2) how these uses vary across linguistic contexts (i.e., to model polysemy).

Our word vectors are learned functions of the internal states of a deep bidirectional language model (biLM), which is pretrained on a large text corpus.

We show that these representations can be easily added to existing models and significantly improve the state of the art across six challenging NLP problems, including question answering, textual entailment and sentiment analysis.

We also present an analysis showing that exposing the deep internals of the pretrained network is crucial, allowing downstream models to mix different types of semi-supervision signals.

Due to their ability to capture syntactic and semantic information of words from large scale unlabeled text, pretrained word vectors BID59 BID39 BID44 are a standard component of most state-of-the-art NLP architectures, including for question answering BID60 , textual entailment BID5 and semantic role labeling .

However, these approaches for learning word vectors only allow a single, contextindependent representation for each word.

Another line of research focuses on global methods for learning sentence and document encoders from unlabeled data (e.g., BID30 BID25 BID18 BID11 , where the goal is to build one representation for an entire text sequence.

In contrast, as we will see Section 3, ELMo representations are associated with individual words, but also encode the larger context in which they appear.

Previously-proposed methods overcome some of the shortcomings of traditional word vectors by either enriching them with subword information (e.g., BID61 BID2 or learning separate vectors for each word sense (e.g., BID42 .

Our approach also benefits from subword units through the use of character convolutions, and we seamlessly incorporate multi-sense information into downstream tasks without explicitly training to predict predefined sense classes.

Other recent work has also focused on learning context-dependent representations.

context2vec BID36 ) uses a bidirectional Long Short Term Memory (LSTM; BID19 to encode the context around a pivot word.

Other approaches for learning contextual embeddings include the pivot word itself in the representation and are computed with the encoder of either a supervised neural machine translation (MT) system (CoVe; BID35 or an unsupervised language model BID45 .

Both of these approaches benefit from large datasets, although the MT approach is limited by the size of parallel corpora.

In this paper, we take full advantage of access to plentiful monolingual data, and train our biLM on a corpus with approximately 30 million sentences BID4 .

We also generalize these approaches to deep contextual representations, which we show work well across a broad range of diverse NLP tasks.

Previous work has also shown that different layers of deep biRNNs encode different types of information.

For example, introducing multi-task syntactic supervision (e.g., part-of-speech tags) at the lower levels of a deep LSTM can improve overall performance of higher level tasks such as dependency parsing BID16 or CCG super tagging BID56 .

In an RNN-based encoder-decoder machine translation system, BID1 showed that the representations learned at the first layer in a 2-layer LSTM encoder are better at predicting POS tags then second layer.

Finally, the top layer of an LSTM for encoding word context BID36 has been shown to learn representations of word sense.

We show that similar signals are also induced by the modified language model objective of our ELMo representations, and it can be very beneficial to learn models for downstream tasks that mix these different types of semi-supervision.

Similar to computer vision where representations from deep CNNs pretrained on ImageNet are fine tuned for other tasks BID26 BID54 , BID12 and BID51 pretrain encoder-decoder pairs and then fine tune with task specific supervision.

In contrast, after pretraining the biLM with unlabeled data, we fix the weights and add additional task-specific model capacity, allowing us to leverage large, rich and universal biLM representations for cases where downstream training data size dictates a smaller supervised model.

This section details how we compute ELMo representations and use them to improve NLP models.

We first present our biLM approach (Sec. 3.1) and then show how ELMo representations are computed on top of them (Sec. 3.2).

We also describe how to add ELMo to existing neural NLP architectures (Sec. 3.3) , and the details of how the biLM is pretrained (Sec. 3.4).

Given a sequence of N tokens, (t 1 , t 2 , ..., t N ), a forward language model computes the probability of the sequence by modeling the probability of token t k given the history (t 1 , ..., t k???1 ): DISPLAYFORM0 Recent state-of-the-art neural language models BID22 BID37 BID38 ) compute a context-independent token representation x LM k (via token embeddings or a CNN over characters) then pass it through L layers of forward LSTMs.

At each position k, each LSTM layer outputs a context-dependent representation DISPLAYFORM1 , is used to predict the next token t k+1 with a Softmax layer.

A backward LM is similar to a forward LM, except it runs over the sequence in reverse, predicting the previous token given the future context: DISPLAYFORM2 It can be implemented in an analogous way to a forward LM, with each backward LSTM layer j in a L layer deep model producing representations DISPLAYFORM3 A biLM combines both a forward and backward LM.

Our formulation jointly maximizes the log likelihood of the forward and backward directions: DISPLAYFORM4 We tie the parameters for both the token representation (?? x ) and Softmax layer (?? s ) in the forward and backward direction while maintaining separate parameters for the LSTMs in each direction.

Overall, this formulation is similar to the approach of BID45 , with the exception that we share some weights between directions instead of using completely independent parameters.

In the next section, we depart from previous work by introducing a new approach for learning word representations that are a linear combination of the biLM layers.

ELMo is a task specific combination of the intermediate layer representations in the biLM.

For each token t k , a L-layer biLM computes a set of 2L + 1 representations DISPLAYFORM0 is the token layer and h DISPLAYFORM1 For inclusion in a downstream model, ELMo collapses all layers in R into a single vector, DISPLAYFORM2 In the simplest case, ELMo just selects the top layer, DISPLAYFORM3 , as in TagLM BID45 and CoVe (McCann et al., 2017) .

Across the tasks considered, the best performance was achieved by weighting all biLM layers with softmax-normalized learned scalar weights s = Sof tmax(w): DISPLAYFORM4 The scalar parameter ?? allows the task model to scale the entire ELMo vector and is of practical importance to aid the optimization process (see the Appendix for details).

Considering that the activations of each biLM layer have a different distribution, in some cases it also helped to apply layer normalization BID0 to each biLM layer before weighting.

Given a pre-trained biLM and a supervised architecture for a target NLP task, it is a simple process to use the biLM to improve the task model.

All of the architectures considered in this paper use RNNs, although the method is equally applicable to CNNs.

We first consider the lowest layers of the supervised model without the biLM.

Most RNN based NLP models (including every model in this paper) share a common architecture at the lowest layers, allowing us to add ELMo in a consistent, unified manner.

Given a sequence of tokens (t 1 , . . .

, t N ), it is standard to form a context-independent token representation x k for each token position using pretrained word embeddings and optionally character-based representations (typically from a CNN Finally, we found it beneficial to add a moderate amount of dropout to ELMo BID57 and in some cases to regularize the ELMo weights by adding ?? w ??? 1 L+1 2 2 to the loss.

This regularization term imposes an inductive bias on the ELMo weights to stay close to an average of all biLM layers.

The pre-trained biLMs in this paper are similar to the architectures in BID22 and BID23 , but modified to support joint training of both directions and to include a residual connection between LSTM layers.

We focus on biLMs trained at large scale in this work, as BID45 highlighted the importance of using biLMs over forward-only LMs and large scale training.

To balance overall language model perplexity with model size and computational requirements for downstream tasks while maintaining a purely character-based input representation, we halved all embedding and hidden dimensions from the single best model CNN-BIG-LSTM in BID22 .

The resulting model uses 2048 character n-gram convolutional filters followed by two highway layers BID58 and a linear projection down to a 512 dimension token representation.

Each recurrent direction uses two LSTM layers with 4096 units and 512 dimension projections.

The average forward and backward perplexities on the 1B Word Benchmark BID4 ) is 39.7, compared to 30.0 for the forward CNN-BIG-LSTM.

Generally, we found the forward and backward perplexities to be approximately equal, with the backward value slightly lower.

Fine tuning on task specific data resulted in significant drops in perplexity and an increase in downstream task performance in some cases.

This can be seen as a type of domain transfer for the biLM.

As a result, in most cases we used a fine-tuned biLM in the downstream task.

See the Appendix for details.

TAB1 shows the performance of ELMo across a diverse set of six benchmark NLP tasks.

In every task considered, simply adding ELMo establishes a new state-of-the-art result, with relative error reductions ranging from 6 -20% over strong base models.

This is a very general result across a diverse set model architectures and language understanding tasks.

In the remainder of this section we provide high-level sketches of the individual task results; see the Appendix for full experimental details.

Textual entailment Textual entailment is the task of determining whether a "hypothesis" is true, given a "premise".

The Stanford Natural Language Inference (SNLI) corpus BID3 provides approximately 550K hypothesis/premise pairs.

Our baseline, the ESIM sequence model from BID5 , uses a biLSTM to encode the premise and hypothesis, followed by a matrix attention layer, a local inference layer, another biLSTM inference composition layer, and finally a pooling operation before the output layer.

Overall, adding ELMo to the ESIM model improves accuracy by an average of 0.7% across five random seeds, increasing the single model state-of-theart by 0.6% over the CoVe enhanced model from BID35 .

A five member ensemble pushes the overall accuracy to 89.3%, exceeding the previous ensemble best of 88.9% BID15 ) -see Appendix for details.

Question answering The Stanford Question Answering Dataset (SQuAD) BID50 contains 100K+ crowd sourced question-answer pairs where the answer is a span in a given Wikipedia paragraph.

Our baseline model BID8 ) is an improved version of the Bidirectional Attention Flow model in Seo et al. (BiDAF; .

It adds a self-attention layer after the bidirectional attention component, simplifies some of the pooling operations and substitutes the LSTMs for gated recurrent units (GRUs; BID7 .

After adding ELMo to the baseline model, test set F 1 improved by 4.2% from 81.1% to 85.3%, improving the single model state-ofthe-art by 1.0%.Semantic role labeling A semantic role labeling (SRL) system models the predicate-argument structure of a sentence, and is often described as answering "

Who did what to whom".

SRL is a challenging NLP task as it requires jointly extracting the arguments of a predicate and establishing their semantic roles.

modeled SRL as a BIO tagging problem and used an 8-layer deep biLSTM with forward and backward directions interleaved, following BID64 .

As shown in TAB1 , when adding ELMo to a re-implementation of the single model test set F 1 jumped 3.2% from 81.4% to 84.6% -a new state-of-the-art on the OntoNotes benchmark BID47 , even improving over the previous best ensemble result by 1.2% (see TAB1 in the Appendix).Coreference resolution Coreference resolution is the task of clustering mentions in text that refer to the same underlying real world entities.

Our baseline model is the end-to-end span-based neural model of .

It uses a biLSTM and attention mechanism to first compute span representations and then applies a softmax mention ranking model to find coreference chains.

In our experiments with the OntoNotes coreference annotations from the CoNLL 2012 shared task BID46 , adding ELMo improved the average F 1 by 3.2% from 67.2 to 70.4, establishing a new state of the art, again improving over the previous best ensemble result by 1.6% F 1 (see TAB1 in the Appendix).Named entity extraction The CoNLL 2003 NER task BID52 consists of newswire from the Reuters RCV1 corpus tagged with four different entity types (PER, LOC, ORG, MISC).

Fol- lowing recent state-of-the-art systems BID29 BID45 , the baseline model is a biLSTM-CRF based sequence tagger.

It forms a token representation by concatenating pre-trained word embeddings with a character-based CNN representation, passes it through two layers of biLSTMs, and then computes the sentence conditional random field (CRF) loss BID28 during training and decodes with the Viterbi algorithm during testing, similar to BID10 .

As shown in TAB1 , our ELMo enhanced biLSTM-CRF achieves 92.22% F 1 averaged over five runs.

The key difference between our system and the previous state of the art from Peters et al. FORMULA9 is that we allowed the task model to learn a weighted average of all biLM layers, whereas BID45 only use the top biLM layer.

As shown in Sec. 5.1, using all layers instead of just the last layer improves performance across multiple tasks.

The fine-grained sentiment classification task in the Stanford Sentiment Treebank (SST-5; BID55 involves selecting one of five labels (from very negative to very positive) to describe a sentence from a movie review.

The sentences contain diverse linguistic phenomena such as idioms, named entities related to film, and complex syntactic constructions (e.g., negations) that are difficult for models to learn directly from the training dataset alone.

Our baseline model is the biattentive classification network (BCN) from BID35 , which also held the prior state-of-the-art result when augmented with CoVe embeddings.

Replacing CoVe with ELMo in the BCN model results in a 1.0% absolute accuracy improvement over the state of the art.

This section provides an ablation analysis to validate our chief claims and to elucidate some interesting aspects of ELMo representations.

Sec. 5.1 shows that using deep contextual representations in downstream tasks improves performance over previous work that uses just the top layer, regardless of whether they are produced from a biLM or MT encoder, and that ELMo representations provide the best overall performance.

Sec. 5.3 explores the different types of contextual information captured in biLMs and confirms that syntactic information is better represented at lower layers while semantic information is captured a higher layers, consistent with MT encoders.

It also shows that our biLM consistently provides richer representations then CoVe.

Additionally, we analyze the sensitivity to where ELMo is included in the task model (Sec. 5.2), training set size (Sec. 5.4), and visualize the ELMo learned weights across the tasks (Sec. 5.5).

There are many alternatives to Equation 1 for combining the biLM layers.

Previous work on contextual representations use only the last layer, whether it be from a biLM BID45 or an MT encoder (CoVe; BID35 .

The choice of the regularization parameter ?? is also important, as large values such as ?? = 1 effectively reduce the weighting function to a simple average over the layers, while smaller values (e.g., ?? = 0.001) allows the layer weights to vary.

TAB2 compares these alternatives for SNLI, SRL and SQuAD.

Including representations from all layers improves overall performance over just using the last layer, and including contextual representations from the last layer improves performance over the baseline.

For example, in the case of SQuAD, using just the last biLM layer improves development F 1 by 1.7% over the baseline.

Aver- . . } they were actors who had been handed fat roles in a successful play , and had talent enough to fill the roles competently , with nice understatement .aging all biLM layers instead of using just the last layer improves F 1 another 1.1% (comparing "Last Only" to ??=1 columns), and allowing the task model to learn individual layer weights improves F 1 another 1.2% (??=1 vs. ??=0.001).

A small ?? is preferred in most cases with ELMo, although for NER, a task with a smaller training set, the results are insensitive to ?? (not shown).The overall trend is similar with CoVe but with smaller increases over the baseline.

In the case of SNLI, weighting all layers with ?? = 1 improves development accuracy from 88.2 to 88.7% over using just the last layer.

SRL F 1 increased a marginal 0.1% to 82.2 for the ?? = 1 case compared to using the last layer only.

All of the task architectures in this paper include word embeddings only as input to the lowest layer biRNN.

However, we find that including ELMo at the output of the biRNN in task-specific architectures improves overall results for some tasks.

As shown in TAB3 , including ELMo at both the input and output layers for SNLI and SQuAD improves over just the input layer, but for SRL (and coreference resolution, not shown) performance is highest when it is included at just the input layer.

One possible explanation for this result is that both the SNLI and SQuAD architectures use attention layers after the biRNN, so introducing ELMo at this layer allows the supervised model to attend directly to the biLM's internal representations.

In the SRL case, the task-specific context representations are likely more important than those from the biLM.

Since adding ELMo improves task performance over word vectors alone, the biLM's contextual representations must encode information generally useful for NLP tasks that is not captured in word vectors.

Intuitively, the biLM must be disambiguating the meaning of words using their context.

Consider "play", a highly polysemous word.

The top of TAB4 lists nearest neighbors to "play" using GloVe vectors.

They are spread across several parts of speech (e.g., "played", "playing" as verbs, and "player", "game" as nouns) but concentrated in the sports-related senses of "play".

In contrast, the bottom two rows show nearest neighbor sentences from the SemCor dataset (see below) using the biLM's context representation of "play" in the source sentence.

In these cases, the biLM is able to disambiguate both the part of speech and word sense in the source sentence.

These observations can be quantified using an approach similar to BID1 .

To isolate the information encoded by the biLM, the representations are used to directly make predictions for a fine grained word sense disambiguation (WSD) task and a POS tagging task.

Using this approach, it is also possible to compare to CoVe, and across each of the individual layers.

Word sense disambiguation Given a sentence, we can use the biLM representations to predict the sense of a target word using a simple 1-nearest neighbor approach, similar to BID36 .

To do so, we first use the biLM to compute representations for all words in SemCor 3.0, our training corpus BID40 , and then take the average representation for each sense.

At test time, we again use the biLM to compute representations for a given target word and take the nearest neighbor sense from the training set, falling back to the first sense from WordNet for lemmas not observed during training.

TAB5 compares WSD results using the evaluation framework from BID49 across the same suite of four test sets in BID48 .

Overall, the biLM top layer representations have F 1 of 69.0 and are better at WSD then the first layer.

This is competitive with a state-ofthe-art WSD-specific supervised model using hand crafted features BID20 ) and a task specific biLSTM that is also trained with auxiliary coarse-grained semantic labels and POS tags BID48 .

The CoVe biLSTM layers follow a similar pattern to those from the biLM (higher overall performance at the second layer compared to the first); however, our biLM outperforms the CoVe biLSTM, which trails the WordNet first sense baseline.

POS tagging To examine whether the biLM captures basic syntax, we used the context representations as input to a linear classifier that predicts POS tags with the Wall Street Journal portion of the Penn Treebank (PTB) BID34 .

As the linear classifier adds only a tiny amount of model capacity, this is direct test of the biLM's representations.

Similar to WSD, the biLM representations are competitive with carefully tuned, task specific biLSTMs with character representations BID32 BID33 .

However, unlike WSD, accuracies using the first biLM layer are higher than the top layer, consistent with results from deep biLSTMs in multi-task training BID56 BID16 and MT BID1 .

CoVe POS tagging accuracies follow the same pattern as those from the biLM, and just like for WSD, the biLM achieves higher accuracies than the CoVe encoder.

Implications for supervised tasks Taken together, these experiments confirm different layers in the biLM represent different types of information and explain why including all biLM layers is important for the highest performance in downstream tasks.

In addition, the biLM's representations are more transferable to WSD and POS tagging than those in CoVe, which helps illustrate why ELMo outperforms CoVe in downstream tasks.

Adding ELMo to a model increases the sample efficiency considerably, both in terms of number of parameter updates to reach state-of-the-art performance and the overall training set size.

For example, the SRL model reaches a maximum development F 1 after 486 epochs of training without ELMo.

After adding ELMo, the model exceeds the baseline maximum at epoch 10, a 98% relative decrease in the number of updates needed to reach the same level of performance.

In addition, ELMo-enhanced models use smaller training sets more efficiently than models without ELMo.

Figure 1 compares the performance of baselines models with and without ELMo as the percentage of the full training set is varied from 0.1% to 100%.

Improvements with ELMo are largest for smaller training sets and significantly reduce the amount of training data needed to reach a given level of performance.

In the SRL case, the ELMo model with 1% of the training set has about the same F 1 as the baseline model with 10% of the training set.

FIG0 visualizes the softmax-normalized learned layer weights across the tasks.

At the input layer, in all cases, the task model favors the first biLSTM layer, with the remaining emphasis split between the token layer and top biLSTM in task specific ways.

For coreference and SQuAD, the first LSTM layer is strongly favored, but the distribution is less peaked for the other tasks.

It is an interesting question for future work to understand why the first biLSTM layer is universally favored.

The output layer weights are relatively balanced, with a slight preference for the lower layers.

We have introduced a general approach for learning high-quality deep context-dependent representations from biLMs, and shown large improvements when applying ELMo to a broad range of NLP tasks.

Through ablations and other controlled experiments, we have also confirmed that the biLM layers efficiently encode different types of syntactic and semantic information about wordsin-context, and that using all layers improves overall task performance.

Our approach raises several interesting questions for future work, broadly organized into two themes."What is the best training regime for learning generally useful NLP representations?

" By choosing a biLM training objective, we benefit from nearly limitless unlabeled text and can immediately apply advances in language modeling, an active area of current research.

However, it's possible that further decreases in LM perplexity will not translate to more transferable representations, and that other objective functions might be more suitable for learning general purpose representations."What is the best way to use deep contextual representations for other tasks?" Our method of using a weighted average of all layers from the biLM is simple and empirically successful.

However, a deeper fusion of the biLM layers with a target NLP architecture may lead to further improvements.

This Appendix contains details of the model architectures, training routines and hyper-parameter choices for the state-of-the-art models in Section 4.All of the individual models share a common architecture in the lowest layers with a context independent token representation below several layers of stacked RNNs -LSTMs in every case except the SQuAD model that uses GRUs.

As noted in Sec. 3.4, fine tuning the biLM on task specific data typically resulted in significant drops in perplexity.

To fine tune on a given task, the supervised labels were temporarily ignored, the biLM fine tuned for one epoch on the training split and evaluated on the development split.

Once fine tuned, the biLM weights were fixed during task training.

TAB7 lists the development set perplexities for the considered tasks.

In every case except CoNLL 2012, fine tuning results in a large improvement in perplexity, e.g., from 72.1 to 16.8 for SNLI.The impact of fine tuning on supervised performance is task dependent.

In the case of SNLI, fine tuning the biLM increased development accuracy 0.6% from 88.9% to 89.5% for our single best model.

However, for sentiment classification development set accuracy is approximately the same regardless whether a fine tuned biLM was used.

The ?? parameter in Eqn.(1) was of practical importance to aid optimization, due to the different distributions between the biLM internal representations and the task specific representations.

It is especially important in the last-only case in Sec. 5.1.

Without this parameter, the last-only case performed poorly (well below the baseline) for SNLI and training failed completely for SRL.

Our baseline SNLI model is the ESIM sequence model from BID5 .

Following the original implementation, we used 300 dimensions for all LSTM and feed forward layers and pretrained 300 dimensional GloVe embeddings that were fixed during training.

For regularization, we added 50% variational dropout BID14 to the input of each LSTM layer and 50% dropout BID57 at the input to the final two fully connected layers.

All feed forward layers use ReLU activations.

Parameters were optimized using Adam BID24 with gradient norms clipped at 5.0 and initial learning rate 0.0004, decreasing by half each time accuracy on the development set did not increase in subsequent epochs.

The batch size was 32.The best ELMo configuration added ELMo vectors to both the input and output of the lowest layer LSTM, using (1) with layer normalization and ?? = 0.001.

Due to the increased number of parameters in the ELMo model, we added 2 regularization with regularization coefficient 0.0001 to all recurrent and feed forward weight matrices and 50% dropout after the attention layer.

TAB9 compares test set accuracy of our system to previously published systems.

Overall, adding ELMo to the ESIM model improved accuracy by 0.7% establishing a new single model state-of-theart of 88.7%, and a five member ensemble pushes the overall accuracy to 89.3%.

Our QA model is a simplified version of the model from BID8 .

It embeds tokens by concatenating each token's case-sensitive 300 dimensional GloVe word vector BID44 ) with a character-derived embedding produced using a convolutional neural network followed by max-pooling on learned character embeddings.

The token embeddings are passed through a shared bi-directional GRU, and then the bi-directional attention mechanism from BiDAF BID53 .

The augmented context vectors are then passed through a linear layer with ReLU activations, a residual self-attention layer that uses a GRU followed by the same attention mechanism applied context-to-context, and another linear layer with ReLU activations.

Finally, the results are fed through linear layers to predict the start and end token of the answer.

Variational dropout is used before the input to the GRUs and the linear layers at a rate of 0.2.

A dimensionality of 90 is used for the GRUs, and 180 for the linear layers.

We optimize the model using Adadelta with a batch size of 45.

At test time we use an exponential moving average of the weights and limit the output span to be of at most size 17.

We do not update the word vectors during training.

Performance was highest when adding ELMo without layer normalization to both the input and output of the contextual GRU layer and leaving the ELMo weights unregularized (?? = 0).

BID8 additionally uses GRUs for prediction, and is stronger than our model without ELMo (79.4 vs 80.7 F1 on the dev set), so we anticipate being able to make further performance improvements by incorporating these changes into our design.

Our baseline SRL model is an exact reimplementation of .

Words are represented using a concatenation of 100 dimensional vector representations, initialized using GloVe BID44 and a binary, per-word predicate feature, represented using an 100 dimensional embedding.

This 200 dimensional token representation is then passed through an 8 layer "interleaved" biLSTM with a 300 dimensional hidden size, in which the directions of the LSTM layers alternate per layer.

This deep LSTM uses Highway connections BID58 between layers and variational recurrent dropout BID14 ).

This deep representation is then projected using a final dense layer followed by a softmax activation to form a distribution over all possible tags.

Labels consist of semantic roles from PropBank BID43 augmented with a BIO labeling scheme to represent argument spans.

During training, we minimize the negative log likelihood of the tag sequence using Adadelta with a learning rate of 1.0 and ?? = 0.95 BID63 .

At test time, we perform Viterbi decoding to enforce valid spans using BIO constraints.

Variational dropout of 10% is added to all LSTM hidden layers.

Gradients are clipped if their value exceeds 1.0.

Models are trained for 500 epochs or until validation F1 does not improve for 200 epochs, whichever is sooner.

The pretrained GloVe vectors are fine-tuned during training.

The final dense layer and all cells of all LSTMs are initialized to be orthogonal.

The forget gate bias is initialized to 1 for all LSTMs, with all other gates initialized to 0, as per BID21 .

TAB1 compares test set F1 scores of our ELMo augmented implementation of with previous results.

Our single model score of 84.6 F1 represents a new state-of-the-art result on the CONLL 2012 Semantic Role Labeling task, surpassing the previous single model result by 2.9 F1 and a 5-model ensemble by 1.2 F1.

BID3 78.2 DIIN BID15 88.0 BCN+Char+CoVe BID35 88.1 ESIM BID5 88.0 ESIM+ELMo 88.7 ?? 0.17 ESIM+TreeLSTM BID5 88.6 DIIN ensemble BID15 88.9 ESIM+ELMo ensemble 89.37.6 COREFERENCE RESOLUTION Our baseline coreference model is the end-to-end neural model from with all hyperparameters exactly following the original implementation.

The best configuration added ELMo to the input of the lowest layer biLSTM and weighted the biLM layers using (1) without any regularization (?? = 0) or layer normalization.

50% dropout was added to the ELMo representations.

BID10 ) with a CNN character based representation.

The character representation uses 16 dimensional character embeddings and 128 convolutional filters of width three characters, a ReLU activation and by max pooling.

The token representation is passed through two biLSTM layers, the first with 200 hidden units and the second with 100 hidden units before a final dense layer and softmax layer.

During training, we use a CRF loss and at test time perform decoding using the Viterbi algorithm while ensuring that the output tag sequence is valid.

Variational dropout is added to the input of both biLSTM layers.

During training the gradients are rescaled if their 2 norm exceeds 5.0 and parameters updated using Adam with constant learning rate of 0.001.

The pre-trained Senna embeddings are fine tuned during training.

We employ early stopping on the development set and report the averaged test set score across five runs with different random seeds.

ELMo was added to the input of the lowest layer task biLSTM.

As the CoNLL 2003 NER data set is relatively small, we found the best performance by constraining the trainable layer weights to be effectively constant by setting ?? = 0.1 with (1).

TAB1 compares test set F 1 scores of our ELMo enhanced biLSTM-CRF tagger with previous results.

Overall, the 92.22% F 1 from our system establishes a new state-of-the-art.

When compared to BID45 , using representations from all layers of the biLM provides a modest improvement.

We use almost the same biattention classification network architecture described in BID35 , with the exception of replacing the final maxout network with a simpler feedforward network composed of two ReLu layers with dropout.

A BCN model with a batch-normalized maxout network reached significantly lower validation accuracies in our experiments, although there may FORMULA9 77.5 BID64 81.3 , single 81.7 , ensemble 83.4 , our impl.

81.4 ) + ELMo 84.6 BID13 60.

3 Wiseman et al. (2016) 64.2 BID9 65.7 BID29 90.94 BID33 91.2 BID6 ???,??? 91.62 ?? 0.33 BID45 ??? 91.93 ?? 0.19 biLSTM-CRF + ELMo 92.22 ?? 0.10 TAB1 :

Test set accuracy for SST-5.

Model Acc.

DMN BID27 52.1 LSTM-CNN BID65 52.4 NTI BID41 53.1 BCN+Char+CoVe (McCann et al., 2017) 53.7 BCN+ELMo 54.7 be discrepancies between our implementation and that of BID35 .

To match the CoVe training setup, we only train on phrases that contain four or more tokens.

We use 300-d hidden states for the biLSTM and optimize the model parameters with Adam Kingma & Ba (2015) using a learning rate of 0.0001.

The trainable biLM layer weights are regularized by ?? = 0.001, and we add ELMo to both the input and output of the biLSTM; the output ELMo vectors are computed with a second biLSTM and concatenated to the input.

@highlight

We introduce a new type of deep contextualized word representation that significantly improves the state of the art for a range of challenging NLP tasks.