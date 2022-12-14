Natural Language Inference (NLI) task requires an agent to determine the logical relationship between a natural language premise and a natural language hypothesis.

We introduce Interactive Inference Network (IIN), a novel class of neural network architectures that is able to achieve high-level understanding of the sentence pair by hierarchically extracting semantic features from interaction space.

We show that an interaction tensor (attention weight) contains semantic information to solve natural language inference, and a denser interaction tensor contains richer semantic information.

One instance of such architecture, Densely Interactive Inference Network (DIIN), demonstrates the state-of-the-art performance on large scale NLI copora and large-scale NLI alike corpus.

It's noteworthy that DIIN achieve a greater than 20% error reduction on the challenging Multi-Genre NLI (MultiNLI) dataset with respect to the strongest published system.

Natural Language Inference (NLI also known as recognizing textual entiailment, or RTE) task requires one to determine whether the logical relationship between two sentences is among entailment (if the premise is true, then the hypothesis must be true), contradiction (if the premise is true, then the hypothesis must be false) and neutral (neither entailment nor contradiction).

NLI is known as a fundamental and yet challenging task for natural language understanding , not only because it requires one to identify the language pattern, but also to understand certain common sense knowledge.

In TAB0 , three samples from MultiNLI corpus show solving the task requires one to handle the full complexity of lexical and compositional semantics.

The previous work on NLI (or RTE) has extensively researched on conventional approaches BID25 Bos & Markert, 2005; BID39 .

Recent progress on NLI is enabled by the availability of 570k human annotated dataset and the advancement of representation learning technique.

Among the core representation learning techniques, attention mechanism is broadly applied in many NLU tasks since its introduction: machine translation BID15 , abstractive summarization BID50 , Reading Comprehension , dialog system BID41 , etc.

As described by BID57 , "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.

The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key".

Attention mechanism is known for its alignment between representations, focusing one part of representation over another, and modeling the dependency regardless of sequence length.

Observing attention's powerful capability, we hypothesize that the attention weight can assist with machine to understanding the text.

A regular attention weight, the core component of the attention mechanism, encodes the crosssentence word relationship into a alignment matrix.

However, a multi-head attention weightVaswani et al. (2017) can encode such interaction into multiple alignment matrices, which shows a more powerful alignment.

In this work, we push the multi-head attention to a extreme by building a word- by-word dimension-wise alignment tensor which we call interaction tensor.

The interaction tensor encodes the high-order alignment relationship between sentences pair.

Our experiments demonstrate that by capturing the rich semantic features in the interaction tensor, we are able to solve natural language inference task well, especially in cases with paraphrase, antonyms and overlapping words.

We dub the general framework as Interactive Inference Network(IIN).

To the best of our knowledge, it is the first attempt to solve natural language inference task in the interaction space.

We further explore one instance of Interactive Inference Network, Densely Interactive Inference Network (DIIN), which achieves new state-of-the-art performance on both SNLI and MultiNLI copora.

To test the generality of the architecture, we interpret the paraphrase identification task as natural language inference task where matching as entailment, not-matching as neutral.

We test the model on Quora Question Pair dataset, which contains over 400k real world question pair, and achieves new state-of-the-art performance.

We introduce the related work in Section 2, and discuss the general framework of IIN along with a specific instance that enjoys state-of-the-art performance on multiple datasets in Section 3.

We describe experiments and analysis in Section 4.

Finally, we conclude and discuss future work in Section 5.

The early exploration on NLI mainly rely on conventional methods and small scale datasets BID40 .

The availability of SNLI dataset with 570k human annotated sentence pairs has enabled a good deal of progress on natural language understanding.

The essential representation learning techniques for NLU such as attention , memory and the use of parse structure are studied on the SNLI which serves as an important benchmark for sentence understanding.

The models trained on NLI task can be divided into two categories: (i) sentence encoding-based model which aims to find vector representation for each sentence and classifies the relation by using the concatenation of two vector representation along with their absolute element-wise difference and element-wise product .(ii) Joint feature models which use the cross sentence feature or attention from one sentence to another .After neural attention mechanism is successfully applied on the machine translation task, such technique has became widely used in both natural language process and computer vision domains.

Many variants of attention technique such as hard-attention , self-attention , multi-hop attention BID27 , bidirectional attention BID51 and multi-head attention BID57 are also introduced to tackle more complicated tasks.

Before this work, neural attention mechanism is mainly used to make alignment, focusing on specific part of the representation.

In this work, we want to show that attention weight contains rich semantic information required for understanding the logical relationship between sentence pair.

Though RNN or LSTM are very good for variable length sequence modeling, using Convolutional neural network in NLU tasks is very desirable because of its parallelism in computation.

Convolutional structure has been successfully applied in various domain such as machine translation BID26 , sentence classification BID34 ), text matching BID30 and sentiment analysis BID33 , etc.

The convolution structure is also applied on different level of granularity such as byte BID65 , character , word BID26 and sentences levels.

The Interactive Inference Network (IIN) is a hierarchical multi-stage process and consists of five components.

Each of the components is compatible with different type of implementations.

Potentially all exiting approaches in machine learning, such as decision tree, support vector machine and neural network approach, can be transfer to replace certain component in this architecture.

We focus on neural network approaches below.

1.

Embedding Layer converts each word or phrase to a vector representation and construct the representation matrix for sentences.

In embedding layer, a model can map tokens to vectors with the pre-trained word representation such as GloVe BID48 , word2Vec BID42 and fasttext BID32 .

It can also utilize the preprocessing tool, e.g. named entity recognizer, part-of-speech recognizer, lexical parser and coreference identifier etc., to incorporate more lexical and syntactical information into the feature vector.2.

Encoding Layer encodes the representations by incorporating the context information or enriching the representation with desirable features for future use.

For instance, a model can adopt bidirectional recurrent neural network to model the temporal interaction on both direction, recursive neural network BID54 (also known as TreeRNN) to model the compositionality and the recursive structure of language, or self-attention to model the long-term dependency on sentence.

Different components of encoder can be combined to obtain a better sentence matrix representation.3.

Interaction Layer creates an word-by-word interaction tensor by both premise and hypothesis representation matrix.

The interaction can be modeled in different ways.

A common approach is to compute the cosine similarity or dot product between each pair of feature vector.

On the other hand, a high-order interaction tensor can be constructed with the outer product between two matrix representations.4.

Feature Extraction Layer adopts feature extractor to extract the semantic feature from interaction tensor.

The convolutional feature extractors, such as AlexNet BID36 , VGG BID53 , Inception BID55 , ResNet BID28 and DenseNet BID31 , proven work well on image recognition are completely compatible under such architecture.

Unlike the work BID34 who employs 1-D sliding window, our CNN architecture allows 2-D kernel to extract semantic interaction feature from the word-by-word interaction between n-gram pair.

Sequential or tree-like feature extractors are also applicable in the feature extraction layer.

Output Layer decodes the acquired features to give prediction.

Under the setting of NLI, the output layer predicts the confidence on each class.

Here we introduce Densely Interactive Inference Network (DIIN) 1 , which is a relatively simple instantiation of IIN but produces state-of-the-art performance on multiple datasets.

Embedding Layer: For DIIN, we use the concatenation of word embedding, character feature and syntactical features.

The word embedding is obtained by mapping token to high dimensional vector space by pre-trained word vector (840B GloVe).

The word embedding is updated during training.

As in BID35 BID37 , we filter character embedding with 1D convolution kernel.

The character convolutional feature maps are then max pooled over time dimension for each token to obtain a vector.

The character features supplies extra information for some out-ofvocabulary (OOV) words.

Syntactical features include one-hot part-of-speech (POS) tagging feature and binary exact match (EM) feature.

The EM value is activated if there are tokens with same stem or lemma in the other sentence as the corresponding token.

The EM feature is simple while found useful as in reading comprehension task BID21 .

In analysis section, we study how EM feature helps text understanding.

Now we have premise representation P ??? R p??d and hypothesis representation H ??? R h??d , where p refers to the sequence length of premise, h refers to the sequence length of hypothesis and d means the dimension of both representation.

The 1-D convolutional neural network and character features weights share the same set of parameters between premise and hypothesis.

Encoding Layer: In the encoding layer, the premise representation P and the hypothesis representation H are passed through a two-layer highway network, thus havingP ??? R p??d and H ??? R h??d for new premise representation and new hypothesis representation.

These new representation are then passed to self-attention layer to take into account the word order and context information.

Take premise as example, we model self-attention by DISPLAYFORM0 whereP i is a weighted summation ofP .

We choose ??(a, DISPLAYFORM1 , where w a ??? R 3d is a trainable weight, ??? is element-wise multiplication, [;] is vector concatenation across row, and the implicit multiplication is matrix multiplication.

Then bothP andP are fed into a semantic composite fuse gate (fuse gate in short), which acts as a skip connection.

The fuse gate is implemented as DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 where DISPLAYFORM6 We do the same operation on hypothesis representation, thus havingH. The weights of intraattention and fuse gate for premise and hypothesis are not shared, but the difference between the weights of are penalized.

The penalization aims to ensure the parallel structure learns the similar functionality but is aware of the subtle semantic difference between premise and hypothesis.

Interaction Layer: The interaction layer models the interaction between premise encoded representation P enc and hypothesis encoded representation H enc as follows: DISPLAYFORM7 whereP i is the i-th row vector ofP , andH j is the j-th row vector ofH. Though there are many implementations of interaction, we find ??(a, b) = a ??? b very useful.

Feature Extraction Layer: We adopt DenseNet BID31 as convolutional feature extractor in DIIN.

Though our experiments show ResNet BID28 works well in the architecture, we choose DenseNet because it is effective in saving parameters.

One interesting observation with ResNet is that if we remove the skip connection in residual structure, the model does not converge at all.

We found batch normalization delays convergence without contributing to accuracy, therefore we does not use it in our case.

A ReLU activation function is applied after all convolution unless otherwise noted.

Once we have the interaction tensor I, we use a convolution with 1 ?? 1 kernel to scale down the tensor in a ratio, ??, without following ReLU.

If the input channel is k then the output channel is f loor(k ?? ??).

Then the generated feature map is feed into three sets of Dense block BID31 and transition block pair.

The DenseNet block contains n layers of 3 ?? 3 convolution layer with growth rate of g. The transition layer has a convolution layer with 1 ?? 1 kernel for scaling down purpose, followed by a max pooling layer with stride 2.

The transition scale down ratio in transition layer is ??.

Output Layer: DIIN uses a linear layer to classify final flattened feature representation to three classes.

In this section, we present the evaluation of our model.

We first perform quantitative evaluation, comparing our model with other competitive models.

We then conduct some qualitative analyses to understand how DIIN achieve the high level understanding through interaction.

Here we introduce three datasets we evaluate our model on.

The evaluation metric for all dataset is accuracy.

SNLI Stanford Natural Language Inference (SNLI; Bowman et al. 2015) has 570k human annotated sentence pairs.

The premise data is draw from the captions of the Flickr30k corpus, and the hypothesis data is manually composed.

The labels provided in are "entailment", "neutral', "contradiction" and "-". "-" shows that annotators cannot reach consensus with each other, thus removed during training and testing as in other works.

We use the same data split as in .MultiNLI Multi-Genre NLI Corpus (MultiNLI; ) has 433k sentence pairs, whose collection process and task detail are modeled closely to SNLI.

The premise data is collected from maximally broad range of genre of American English such as written non-fiction genres (SLATE, OUP, GOVERNMENT, VERBATIM, TRAVEL), spoken genres (TELEPHONE, FACE-TO-FACE), less formal written genres (FICTION, LETTERS) and a specialized one for 9/11.

Half of these selected genres appear in training set while the rest are not, creating in-domain (matched) and cross-domain (mismatched) development/test sets.

We use the same data split as provided by .

Since test set labels are not provided, the test performance is obtained through submission on Kaggle.com 2 .

Each team is limited to two submissions per day.

Quora question pair Quora question pair dataset contains over 400k real world question pair selected from Quora.com.

A binary annotation which stands for match (duplicate) or not match (not duplicate) is provided for each question pair.

In our case, duplicate question pair can be interpreted as entailment relation and not duplicate as neutral.

We use the same split ratio as mentioned in .

We implement our algorithm with Tensorflow BID14 framework.

An Adadelta optimizer(Zeiler, 2012) with ?? as 0.95 and as 1e???8 is used to optimize all the trainable weights.

The initial learning rate is set to 0.5 and batch size to 70.

When the model does not improve best indomain performance for 30,000 steps, an SGD optimizer with learning rate of 3e???4 is used to help model to find a better local optimum.

Dropout layers are applied before all linear layers and after word-embedding layer.

We use an exponential decayed keep rate during training, where the initial keep rate is 1.0 and the decay rate is 0.977 for every 10,000 step.

We initialize our word embeddings with pre-trained 300D GloVe 840B vectors BID48 while the out-of-vocabulary word are randomly initialized with uniform distribution.

The character embeddings are randomly initialized with 100D.

We crop or pad each token to have 16 characters.

The 1D convolution kernel size for character embedding is 5.

All weights are constraint by L2 regularization, and the L2 regularization at step t is calculated as follows: DISPLAYFORM0 where L2F ullRatio determines the maximum L2 regularization ratio, and L2F ullStep determines at which step the maximum L2 regularization ratio would be applied on the L2 regularization.

We choose L2F ullRatio as 0.9e ??? 5 and L2F ullStep as 100,000.

72.3 72.1 4.

Gated-Att BiLSTM BID23 73.2 73.6 5.

Shorcut-Stacked encoder BID46 74 Table 2 : MultiNLI result.difference of two encoder weights is set to 1e ??? 3.

For a dense block in feature extraction layer, the number of layer n is set to 8 and growth rate g is set to 20.

The first scale down ratio ?? in feature extraction layer is set to 0.3 and transitional scale down ratio ?? is set to 0.5.

The sequence length is set as a hard cutoff on all experiments: 48 for MultiNLI, 32 for SNLI and 24 for Quora Question Pair Dataset.

During the experiments on MultiNLI, we use 15% of data from SNLI as in .

We select the parameter by the best run of development accuracy.

Our ensembling approach considers the majority vote of the predictions given by multiple runs of the same model under different random parameter initialization.

We compare our result with all other published systems in Table 2 .

Besides ESIM, the state-of-theart model on SNLI, all other models appear at RepEval 2017 workshop.

RepEval 2017 workshop requires all submitted model to be sentence encoding-based model therefore alignment between sentences and memory module are not eligible for competition.

All models except ours share one common feature that they use LSTM as a essential building block as encoder.

Our approach, without using any recurrent structure, achieves the new state-of-the-art performance of 80.0%, exceeding current state-of-the-art performance by more than 5%.

Unlike the observation from , we find the out-of-domain test performance is consistently lower than in-domain test performance.

Selecting parameters from the best in-domain development accuracy partially contributes to this result.

In TAB4 , we compare our model to other model performance on SNLI.

Experiments (2-7) are sentence encoding based model.

Bowman et al. FORMULA0 provides a BiLSTM baseline.

adopts two layer GRU encoder with pre-trained "skip-thoughts" vectors.

To capture sentence-level semantics, use tree-based CNN and Bowman et al. (2016) propose a stack-augmented parser-interpreter neural network (SPINN) which incorporates parsing information in a sequential manner.

uses intra-attention on top of BiLSTM to generate sentence representation, and proposes an memory augmented neural network to encode the sentence.

The next group of model, experiments BID5 BID6 BID7 (11) BID8 BID9 BID10 (15) BID11 BID12 BID13 , uses cross sentence feature.

aligns each sentence word-by-word with attention on top of LSTMs.

enforces cross sentence attention word-by-word matching with the proprosed mL-STM model.

proposes long short-term memory-network(LSTMN) with deep attention fusion that links the current word to previous word stored in memory.

decomposes the task into sub-problems and conquer them respectively.

proposes neural tree indexer, a full n-ary tree whose subtrees can be overlapped.

Re-read LSTM proposed by considers the attention vector of one sentence as the inner-state of LSTM for another sentence. propose a sequential model that infers locally, and a ensemble with tree-like inference module that further improves performance.

We show our model, DIIN, achieves state-of-the-art performance on the competitive leaderboard.

Table 4 : Quora question dataset result.

First six rows are copied from and next two rows from BID56 .

In this subsection, we evaluate the effectiveness of our model for paraphrase identification as natural language inference task.

Other than our baselines, we compare with and BID56 .

BIMPM models different perspective of matching between sentence pair on both direction, then aggregates matching vector with LSTM.

DECATT word and DECATT char uses automatically collected in-domain paraphrase data to noisy pretrain n-gram word embedding and ngram subword embedding correspondingly on decomposable attention model proposed by .

In Table 4 , our experiment shows DIIN has better performance than all other models and an ensemble score is higher than the former best result for more than 1 percent.

Ablation Study We conduct a ablation study on our base model to examine the effectiveness of each component.

We study our model on MultiNLI dataset and we use Matched validation score as the standard for model selection.

The result is shown in Table 5 .

We studies how EM feature Table 5 : Ablation study result.contributes to the system.

After removing the exact match binary feature, we find the performance degrade to 78.2 on matched score on development set and 78.0 on mismatched score.

As observed in reading comprehension task BID21 , the simple exact match feature does help the model to better understand the sentences.

In the experiment 3, we remove the convolutional feature extractor and then model is structured as a sentence-encoding based model.

The sentence representation matrix is max-pooled over time to obtain a feature vector.

Once we have the feature vector p for premise and h for hypothesis, we use [p; h; |p ??? h|; p ??? h] as final feature vector to classify the relationship.

We obtain 73.2 for matched score and 73.6 on mismatched data.

The result is competitive among other sentence-encoding based model.

We further study how encoding layer contribute in enriching the feature space in interaction tensor.

If we remove encoding layer completely, then we'll obtain a 73.5 for matched score and 73.2 for mismatched score.

The result demonstrate the feature extraction layer have powerful capability to capture the semantic feature.

In experiment 5, we remove both self-attention and fuse gate, thus retaining only highway network.

The result improves to 77.7 and 77.3 respectively on matched and mismatched development set.

However, in experiment 6, when we only remove fuse gate, to our surprise, the performance degrade to 73.5 for matched score and 73.8 for mismatched.

On the other hand, if we use the addition of the representation after highway network and the representation after self-attention as skip connection as in experiment 7, the performance increase to 77.3 and 76.3.

The comparison indicates self-attention layer makes the training harder to converge while a skip connection could ease the gradient flow for both highway layer and self-attention layer.

By comparing the base model and the model the in experiment 6, we show that the fuse gate not only well serves as a skip connection, but also makes good decision upon which information the fuse for both representation.

To show that dense interaction tensor contains more semantic information, we replace the dense interaction tensor with dot product similarity matrix between the encoded representation of premise and hypothesis.

The result shows that the dot product similarity matrix has an inferior capacity of semantic information.

Another dimensionality study is provided in supplementary material.

In experiment 9, we share the encoding layer weight, and the result decrease from the baseline.

The result shows that the two set of encoding weights learn the subtle difference between premise and hypothesis.

Error analysis To analyze the model prediction, we use annotated subset of development set provided by that consists of 1,000 examples each tagged with zero or more following tags:??? CONDITIONAL: whether the sentence contains a conditional.??? WORD OVERLAP: whether both sentences share more than 70% of their tokens.??? NEGATION: whether a negation shows up in either sentence.??? ANTO: whether two sentences contain antonym pair.??? LONG SENTENCE: whether premise or hypothesis is longer than 30 or 16 tokens respectively.??? TENSE DIFFERENCE: whether any verb in two sentences uses different tense.??? ACTIVE/PASSIVE: whether there is an active-to-passive (or vice versa) transformation from the premise to the hypothesis.

Table 6 : MultiNLI result.??? PARAPHRASE: whether the two sentences are close paraphrases ??? QUANTITY/TIME REASONING: whether understanding the pair requires quantity or time reasoning.??? COREF: Whether the hypothesis contains a pronoun or referring expression that needs to be resolved using the premise.??? QUANTIFIER: Whether either sentence contains one of the following quantifier: much, enough, more, most, less, least, no, none, some, any, many, few, several, almost, nearly.??? MODAL: Whether one of the following modal verbs appears in either sentence: can, could, may, might, must, will, would, should.??? BELIEF: Whether one of the following belief verbs appear in either sentence: know, believe, understand, doubt, think, suppose, recognize, forget, remember, imagine, mean, agree, disagree, deny, promise.

For more detailed descriptions, please resort to .

The result is shown in Table 6 .

We find DIIN is consistently better on sentence pair with WORD OVERLAP, ANTO, LONG SENTENCE, PARAPHRASE and BELIEF tags by a large margin.

During investigation, we hypothesize exact match feature helps the model to better understand paraphrase, therefore we study the result from second ablation ablation study where exact match feature is not used.

Surprisingly, the model without exact model feature does not work worse on PARAPHRASE, instead, the accuracy on ANTO drops about 10%.

DIIN is also work well on LONG SENTENCE, partially because the receptive field is large enough to cover all tokens.

Visualization We also visualize the hidden representation from interaction tensor I and the feature map from first dense block in Figure 2 .

We pick a sentence pair whose premise is "South Carolina has no referendum right, so the Supreme Court canceled the vote and upheld the ban." and hypothesis is "South Carolina has a referendum right, so the Supreme Court was powerless over the state.".

The upper row of figures are sampled from hidden representation of interaction tensor I. We observe the values of neurons are highly correlated row-wise and column-wise in the interaction tensor I and different channel of hidden representation shows different aspect of interaction.

Though in certain channel same words, "referendum", or phrases, "supreme court", cause activation, different word or phrase pair, such as "ban" and "powerless over", also cause activation in other activation.

It shows the model's strong capacity of understanding text in different perspective.

The lower row of Figure 2 shows the feature map from first dense block.

After being convolved from the interaction tensor and previous feature map, new feature maps shows activation in different position, demonstrating different semantic features are found.

The first figure in the lower row has similar pattern as normal attention weight whereas others has no obvious pattern.

Different channels of feature maps indicate different kinds of semantic feature.

Figure 2 : A visualization of hidden representation.

The premise is "South Carolina has no referendum right, so the Supreme Court canceled the vote and upheld the ban." and the hypothesis is "South Carolina has a referendum right, so the Supreme Court was powerless over the state.".

The upper row are sampled from interaction tensor I and the lower row are sample from the feature map of first dense block.

We use viridis colormap, where yellow represents activation and purple shows the neuron is not active.

We show the interaction tensor (or attention weight) contains semantic information to understand the natural language.

We introduce Interactive Inference Network, a novel class of architecture that allows the model to solve NLI or NLI alike tasks via extracting semantic feature from interaction tensor end-to-end.

One instance of such architecture, Densely Interactive Inference Network (DIIN), achieves state-of-the-art performance on multiple datasets.

By ablating each component in DIIN and changing the dimensionality, we show the effectiveness of each component in DIIN.Though we have the initial exploration of natural language inference in interaction space, the full potential is not yet clear.

We will keep exploring the potential of interaction space.

Incorporating common-sense knowledge from external resources such as knowledge base to leverage the capacity of the mode is another research goal of ours.

<|TLDR|>

@highlight

show multi-channel attention weight contains semantic feature to solve natural language inference task.