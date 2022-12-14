This paper improves upon the line of research that formulates named entity recognition (NER) as a sequence-labeling problem.

We use so-called black-box long short-term memory (LSTM) encoders to achieve state-of-the-art results while providing insightful understanding of what the auto-regressive model learns with a parallel self-attention mechanism.

Specifically, we decouple the sequence-labeling problem of NER into entity chunking, e.g., Barack_B Obama_E was_O elected_O, and entity typing, e.g., Barack_PERSON Obama_PERSON was_NONE elected_NONE, and analyze how the model learns to, or has difficulties in, capturing text patterns for each of the subtasks.

The insights we gain then lead us to explore a more sophisticated deep cross-Bi-LSTM encoder, which proves better at capturing global interactions given both empirical results and a theoretical justification.

Named entity recognition is an important task in information extraction in which we seek to locate entity chunks in text and classify their entity types.

Originally a structured prediction task, NER has since been formulated as a task of sequential token labeling, much like text chunking and part-ofspeech tagging.

With the ability to compute representations of past and future context respectively for each token, bidirectional LSTM (Bi-LSTM) has proved a robust building block for sequencelabeling NER BID7 BID13 BID0 .

However, it has been predominantly used as a black box; research directed to understanding how the model learns to tackle the task is minimal.

In this work, we decouple sequence-labeling NER into the entity chunking and entity typing subtasks, and seek insight into what patterns LSTM learns to capture or has difficulties capturing.

We propose the use of a fast and effective parallel self-attention mechanism alongside Bi-LSTM.

Unlike traditional attention mechanisms used for tasks such as machine translation BID12 and sentence classification BID2 BID11 , our self-attentive Bi-LSTM uses the hidden state of each token as its own query vector and computes context vectors for all tokens in parallel.

For both subtasks, we then find important global patterns that cross past and future context, and in particular discover the way multi-chunk entities are handled.

Furthermore, we discover that the theoretical limitations of traditional Bi-LSTMs harms performance on the task, and hence propose using a cross construction of deep Bi-LSTMs.

As a result, with these cross structures, both selfattentive Bi-LSTM and cross-Bi-LSTM achieve new state-of-the-art results on sequence-labeling NER.In Section 3, the normal Bi-LSTM-CNN model is formulated.

Section 4 details the computation of the parallel self-attention mechanism.

Section 5 presents the empirical results and detailed analyses of the models, with a particular focus on patterns captured for {B, I, E} labels.

Finally in Section 6, cross-Bi-LSTM-CNN is formulated and evaluated on a theoretical basis.

Our contribution is threefold:??? We provide insightful understanding of how a sequence-labeling model tackles NER and the difficulties it faces;??? We propose using cross-Bi-LSTM-CNN for sequence-labeling NER with theoreticallygrounded improvements.

Many have attempted tackling the NER task with LSTM-based sequence encoders BID7 BID13 BID0 BID9 .

Among these, the most similar to the proposed Bi-LSTM-CNN is the model proposed by BID0 .

In contrast to previous work, BID0 stack multiple layers of LSTM cells per direction, and also use a CNN to compute character-level word vectors alongside pre-trained word vectors.

We largely follow their work in constructing the Bi-LSTM-CNN, including the selection of raw features, the CNN, and the multi-layer Bi-LSTM.

The subtle difference is that they send the output of each direction through separate affine-softmax classifiers and then sum their probabilities, effectively forming an ensemble of forward and backward LSTM-CNNs.

Another difference is that they focus on proposing a new representation of external lexicon features, which we do not make use of in this work.

The modeling of global context for sequential-labeling NER has been accomplished using traditional models with intensive feature engineering and conditional random fields (CRF).

BID17 build the Illinois NER tagger with feature-based perceptrons.

In their analysis, the usefulness of Viterbi decoding is minimal, as class transition patterns only occur in small chunks and greedy decoding can handle them comparatively well.

On the other hand, recent research on LSTM or CNNbased encoders report empirical improvements brought by CRF BID7 BID13 BID9 BID18 , as it discourages illegal predictions by explicitly modeling class transition probabilities.

In contrast, the cross structures of self-attention and crossBi-LSTM studied in this work provide for the direct capture of global patterns and extraction of better features to improve class observation likelihoods.

Various attention mechanisms have been proposed and shown success in natural language tasks.

They lighten the LSTM's burden of compressing all relevant information into a single hidden state by consulting past memory.

For seq2seq models, attention has been used for current decoder hidden states BID12 .

For models computing sentence representations, trainable weights are used for self-attention BID2 BID11 .

In this work, we propose using a token-level parallel self-attention mechanism for sequential token-labeling and show that it enables the model to capture cross interactions between past and future contexts.3 BI-LSTM-CNN FOR SEQUENCE LABELING

All models in our experiments use the same set of raw features: word embedding, word capitalization pattern type, character embedding, and character type.

For character embedding, 25d vectors are randomly initialized and trained end-to-end with the model.

Appended to these are 4d one-hot character-type features indicating whether a character is uppercase, lowercase, digit, or punctuation BID0 .

In addition, an unknown character vector and a padding character vector are also trained.

We unify the word token length to 20 by truncation and padding.

The resulting 20-by-(25+4) feature map of each token are applied to a character-trigram CNN with 20 kernels per length 1 to 3 and max-over-time pooling to compute a 60d character-based word vector BID8 BID0 BID13 .For word embedding, pre-trained 300d GloVe word vectors BID15 are used without further tuning.

In addition, 4d one-hot word capitalization features indicate whether a word is uppercase, upper-initial, lowercase, or mixed-caps BID1 BID0 .Throughout this paper, we use X to denote the n-by-d x matrix of raw sequence features, with n denoting the number of word tokens in a sentence and d x = 60 + 300 + 4.

Given a sequence of input feature vectors x 1 , x 2 , . . .

, x T ??? R d1 , an LSTM cell computes a sequence of hidden feature vectors h 1 , h 2 , . . .

, h T ??? R d2 by DISPLAYFORM0 are trainable weight matrices and biases, tanh denotes hyperbolic tangent, ?? denotes sigmoid function, and denotes element-wise multiplication.

Bidirectional LSTMs (Bi-LSTMs) are used to capture the future and the past for each time step.

Following BID0 , 4 distinct LSTM cells -two in each direction -are stacked to capture higher level representations: DISPLAYFORM1 where DISPLAYFORM2 H denote the resulting feature matrices of the stacked application, and || denotes row-wise concatenation.

In all our experiments, 100d LSTM cells are used, so H ??? R n??d h and d h = 200.

Finally, suppose there are d p token classes, the probability of each of which is given by the composition of affine and softmax transformations: DISPLAYFORM0 where H t is the t th row of H, W p ??? R d h ??dp , b ??? R dp are a trainable weight matrix and bias, and s ti and s tj are the i-th and j-th elements of s t .Following BID0 , we use the 5 chunk labels O, S, B, I, and E to denote if a word token is {O}utside any entities, the {S}ole token of an entity, the {B}eginning token of a multitoken entity, {I}n the middle of a multi-token entity, or the {E}nding token of a multi-token entity.

Hence when there are P types of named entities, the actual number of token classes d p = P ?? 4 + 1 for sequence labeling NER.

We propose using a token-level self-attention mechanism ( FIG0 ) that is computed after the autoregressive Bi-LSTM in Section 3.2.

This has two benefits over traditional auto-regressive attention, which wraps stacked LSTM cells to look at past tokens at each time step for each direction of Bi-LSTM.

First, it allows each token to look at both past and future sequences simultaneously with one combined hidden state of past and future, thus capturing cross interactions between the two contexts.

And secondly, since all time steps run in parallel with matrix computations, it introduces little computation time overhead.

Specifically, given the hidden features H of a whole sequence, we project each hidden state to different subspaces, depending on whether it is used as the {q}uery vector to consult other hidden states for each word token, the {k}ey vector to compute its dot-similarities with incoming queries, or the {v}alue vector to be weighted and actually convey information to the querying token.

Moreover, as different aspects of a task can call for different attention, multiple "attentions" running in parallel are used, i.e., multi-head attention BID19 .Formally, let m be the number of attention heads and d c be the subspace dimension.

For each head i ??? {1..m}, the attention weight matrix and context matrix are computed by DISPLAYFORM0 where W qi , W ki , W vi ??? R d h ??dc are trainable projection matrices and ?? performs softmax along the second dimension.

Each row of the resulting ?? 1 , ?? 2 , . . .

, ?? m ??? R n??n contains the attention weights of a token to its context, and each row of C 1 , C 2 , . . .

, C m ??? R n??dc is its context vector.

DISPLAYFORM1 H , the computation of ?? i and C i models the cross interaction between past and future.

Finally, for Bi-LSTM-CNN augmented with the attention mechanism, the hidden vector and context vectors of each token are considered together for classification: We conduct experiments on the challenging OntoNotes 5.0 English NER corpus BID6 BID16 .

OntoNotes is an ambitious project that collects large corpora from diverse sources and provides multi-layer annotations for joint research on constituency parsing, semantic role labeling, coreference resolution, and NER.

The data sources include newswires, web, broadcast news, broadcast conversations, magazines, and telephone conversations.

Some are transcriptions of talk shows and some are translated from Chinese or Arabic.

Such diversity and noisiness requires that models are robust and able to capture a multitude of linguistic patterns.

BID16 , excluding the New Testament corpus as it contains no entity annotations.

Despite this million-token corpus with over 100K annotated entities, previous work has struggled to reach state-of-the-art NER results on the dataset.

This is due partly to the fact that there are 18 types of entities to be classified.

Eleven of these are classes of general names, with NORP including nationalities such as American, FAC including facilities such as The White House, and WORK OF ART including titles of books, songs, and so on.

Moreover, various forms of values of the seven numerical classes must also be identified.

DISPLAYFORM0

The hyperparameters of our models were given in Sections 3 and 4.

When training the models, we minimized per-token cross-entropy loss with the Nadam optimizer BID3 .

In addition, we randomly dropped 35% hidden features (dropout) and upscaled the same amount during training.

Following previous lines of work, we evaluated NER performance with the per-entity F1 score.

The tokens for an entity were all to be classified correctly to count as a correct prediction; otherwise it was counted as either a false positive prediction or a false negative non-prediction.

We stopped training when the validation F1 had not improved for 20 epochs.

All models were initialized and trained 5 times; we report the mean precision, recall, and F1 scores (%) of the experiments.

Validation scores are also reported for future research on this task.

TAB1 shows the overall results of our models against notable previous work.

It can be seen that simple LSTM-based sequence encoders already beat the previous best results without using external lexicons BID0 , document-level context BID18 , or constituency parsers BID10 .

Furthermore, with the proposed parallel self-attention mechanism (ATT), we achieve a new state-of-the-art result (88.29 F1) with a clear margin over past systems.

More importantly, the attention mechanism allows us to conduct insightful analyses in the following sections, yielding important understanding of how Bi-LSTM learns or has difficulty tackling the different sequence-labeling NER subtasks: entity chunking and entity typing.

We decouple the entity chunking task from sequence-labeling NER.

Specifically, for a sentence such as {Barack Obama moves out of the White House .}, the task is to correctly label each token as TAB3 shows the performance of different setups on validation data.

We take the pre-trained models from TAB1 without re-training for this subtask.

{O, S, B, I, E} are the chunk classes.

The column of HC all lists the performance of the full Bi-LSTM-CNN+ATT model on each chunk class, where C all stands for C 1 , . . .

, C 5 .

Other columns list the performance of other setups compared to the full model.

Columns H to C 5 are when the full model is deprived of all other information by zeroing all other vectors for the affine-softmax classification layer in testing time, except for those specified by the column header.

NativeH is the native Bi-LSTM-CNN trained without attention.

The figures shown in the table are the per-token recalls for each chunk class, which tells if a part of the model is responsible for signaling the whole model to predict the class.

DISPLAYFORM0

Looking at the three columns on the left, the first thing we discover is that Bi-LSTM-CNN+ATT designates the task of predicting {I} to the attention mechanism.

The model performance on tokens {I}n the middle of an entity significantly degrades (-28.18 ) in the absence of global context C all , when token hidden state H is left alone.

On the other hand, without the information on the token itself, it is clear that the model strongly favors predicting I (-3.80) given its global context C all .Taking this one step further and zeroing out all other vectors except for each attention head, the roles of context for entity chunking become even clearer.

C 2 and C 3 send strong signals (-36.45,-39.19 ) on entity chunk {E}nding to the model, plus weak signals (-60.56,-50.19 ) on entity chunk {I}nside, while C 4 sends a strong signal (-12.21) on entity chunk {B}eginning plus weak signals (-57.19) on {I}nside.

When all these heads fire simultaneously, the model produces a strong signal to {I}.However, NativeH -Bi-LSTM-CNN trained without attention -underperforms in chunk labels {B} (-0.63), {I} (-0.41), {E} (-0.38) in comparison to HC all , the model trained with ATT.

This suggests that entity chunking is indeed a crucial aspect in sequence-labeling NER, and that it is difficult for pure LSTM encoders to compress all necessary information in each hidden state to correctly label all the tokens of a multi-token entity.

Aside from knowing that entity chunking is a crucial, challenging aspect in sequence-labeling NER for Bi-LSTM, one remaining question is how exactly the encoder is attempting to properly classify the {B}egin, {I}nside, and {E}nd of a multi-token entity.

To shed light on this question, we visualize samples from validation data and discover consistent patterns in the attention weight heat maps across sentences and entities.

FIG2 shows one of the samples, where the attention weights ?? 2 , ?? 3 , ?? 4 of a sentence containing the B White I house E are visualized.

The full Bi-LSTM-CNN+ATT (HC all ) classifies the tokens correctly, but when in the absence of the context vectors (H), the predictions become the B White S house E .

For Bi-LSTM-CNN trained without attention at all (NativeH ), the predictions are the O White S house O .

Each row of the matrix shows the attention weight distribution for the diagonal token in bold font.

We observe that ?? 2 and especially ?? 3 have a tendency to focus on the previous tokens: the diagonal shifted left.

In contrast, ?? 4 tends to look at the immediate following tokens: the diagonal shifted right.

By looking for previous tokens that belong to the same entity chunk and finding some, an attention head, via its context vector, can signal to the model that the token spoken of might be the {E}nding token or {I}nside token.

The same is true for an attention head looking at next tokens, but this time signaling for {B}egin and {I}nside.

This also dictates that both signals need to be weaker for {I} but stronger when combined.

This behavior can be observed throughout the heat maps of ?? 2 , ?? 3 , ?? 4 .

In particular for the White house, C all predicts the B White I house O as Saturday is wrongly focused by ?? 4 for house.

27.27 -9.09 From TAB3 , we already know that NativeH has some difficulties in handling multi-token entities, being more inclined to predict {S}ingle-token entities, and that HC all mitigates this problem by delegating work to C all , especially by relying on the latter to signal for {I}n tokens.

The heat maps further tell the story of how the related labels {B, I, E} are handled collectively.

In addition, this also suggests that modeling interactions between future and past contexts is crucial for sequencelabeling NER and motivates the use of a deep cross-Bi-LSTM encoder in Section 6.

DISPLAYFORM0

When the entity chunking task is decoupled from sequence-labeling NER, the remaining entity typing task requires a model to label {Barack Obama moves out of the White House .} as {Barack PERSON Obama PERSON moves NONE out NONE of NONE the FAC White FAC House FAC .

NONE }.

TAB4 shows the entity classes for which HC all yields notably different performance (> 2%) from that of NativeH .

Of particular interest is C 5 's strong signal (27.27) for LAN (language) in comparison to the NativeH 's struggles (-9.09) on this class without attention.

Qualitatively, we study the two sentences shown in Figure 3 , containing Dutch LAN into NONE English LAN and Chinese LAN and NONE English LAN .

HC all classifies the tokens correctly, but both H and N ativeH wrongly predict Dutch NORP into NONE English LAN and Chinese NORP and NONE English LAN .

Here NORP stands for nationality, meaning that both models without attention wrongly judge that Dutch and Chinese here refer to people from these countries.

With attention, in Figure 3 , we see that ?? 1 attends to Dutch and English at the same time for the two tokens and attends to Chinese and English at the same time for the other two.

On the other hand, ?? 5 focuses on all possible LAN tokens, including a small mis-attention to Taiwanese in the second sentence, which is actually a NORP in this case.

These attention weights signify that the model learns a pattern of cross interaction between entities: when two ambiguous entities of NORP , LAN occur together in the same context, the model predicts both as LAN .

In Section 4.1, we briefly mentioned that the computation of attention weights ?? i and context features C i models the cross interaction between past and future.

Mathematically, since H = ??? ??? H || ??? ??? H , the computation of attention scores can be rewritten as DISPLAYFORM0 The un-shifted covariance matrix of the projected ( ??? ??? H || ??? ??? H ) thus computes the interaction between past context and future context for each token, capturing cross-context patterns that the deep Bi-LSTM-CNN specified in Section 3 cannot.

The consequence of this inability has been empirically shown in Section 5.

Here, we further consider the following four simple phrases that form an XOR: {Key and Peele} WOA ; {You and I } WOA ; {Key and I }; {You and Peele} where WOA stands for WORK OF ART .

The first two phrases are respectively a show title and a song title.

The other two are not entities, where the last one actually occurs in an interview with Keegan-Michael Key.

Suppose the phrases themselves are the only available context for the classification of and .

Then the Bi-LSTM-CNN cannot capture good enough features to classify and correctly simultaneously for the four cases, even if they are the training data, no matter how many LSTM cells are stacked.

The key is that given the same half-context of past or future, and is sometimes {WOA :

I } but sometimes {NONE : O}.

It is only when patterns that cross past and future are captured that the model is able to decide the correct label.

Motivated by the limitation of the conventional Bi-LSTM-CNN for sequence labeling, we propose the use of Cross-Bi-LSTM-CNN by changing the deep structure in Section 3.2 to Note that when computing sentence embeddings for tasks such as sentence classification, both directions of a normal Bi-LSTM look at the whole sentence.

However, when computing hidden node features for sequence labeling, each direction of a normal Bi-LSTM looks at only half of the sentence.

Cross-Bi-LSTM remedies this problem by interleaving the hidden features between LSTM layers.

The output of the first layers of both directions are sent to the second layers of both directions, allowing higher layers to capture interactions between past and future contexts for each token.

Empirically, we experiment with cross construction 5 times and find it further improves the performance of Bi-LSTM-CNN from 87.56 (??0.07) to 88.09 (??0.16).

In this paper, we have decoupled named entity recognition into entity chunking and entity typing and demonstrated how sequence-labeling models can learn to handle each of these two subtasks.

By using a fast parallel self-attention mechanism, we have discovered how the beginning and ending of a multi-token entity is determined and how they are jointly correlated to locate the inside tokens.

Further, through our quantitative and qualitative analyses for both chunking and typing, we have shown that it is crucial to capture global patterns that cross both sides of a token.

We demonstrate the theoretical limitation of the conventional deep Bi-LSTM-CNN used in sequence labeling tasks.

In addition to the interpretability of the proposed parallel self-attention, it is shown that it constitutes a way to correlate past and future contexts.

We have also provided deep cross-Bi-LSTM-CNN as another way to extract global context features.

With their respective cross structures, both selfattentive Bi-LSTM and cross-Bi-LSTM achieve new state-of-the-art results on sequence-labeling NER.

@highlight

We provide insightful understanding of sequence-labeling NER and propose to use two types of cross structures, both of which bring theoretical and empirical improvements.