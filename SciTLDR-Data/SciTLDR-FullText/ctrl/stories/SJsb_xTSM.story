We present a novel multi-task training approach to learning multilingual distributed representations of text.

Our system learns word and sentence embeddings jointly by training a multilingual skip-gram model together with a cross-lingual sentence similarity model.

We construct sentence embeddings by processing word embeddings with an LSTM and by taking an average of the outputs.

Our architecture can transparently use both monolingual and sentence aligned bilingual corpora to learn multilingual embeddings, thus covering a vocabulary significantly larger than the vocabulary of the bilingual corpora alone.

Our model shows competitive performance in a standard cross-lingual document classification task.

We also show the effectiveness of our method in a low-resource scenario.

Learning distributed representations of text, whether it be at the level of words BID26 ; BID28 , phrases BID32 ; BID30 , sentences BID18 or documents BID22 , has been one of the most widely researched subjects in natural language processing in recent years.

Word/sentence/document embeddings, as they are now commonly referred to, have quickly become essential ingredients of larger and more complex NLP systems BID4 ; BID25 BID8 ; BID1 ; BID6 looking to leverage the rich semantic and linguistic information present in distributed representations.

One of the exciting avenues of research that has been taking place in the context of distributed text representations, which is also the subject of this paper, is learning multilingual text representations shared across languages BID11 ; BID3 ; BID24 .

Multilingual embeddings open up the possibility of transferring knowledge across languages and building complex NLP systems even for languages with limited amount of supervised resources BID0 ; BID17 .

By far the most popular approach to learning multilingual embeddings is to train a multilingual word embedding model that is then used to derive representations for sentences and documents by composition BID14 .

These models are typically trained solely on word or sentence aligned corpora and the composition models are usually simple predefined functions like averages over word embeddings BID14 ; BID27 or parametric coposition models learned along with the word embeddings.

In this work we learn word and sentence embeddings jointly by training a multilingual skip-gram model BID24 together with a cross-lingual sentence similarity model.

The multilingual skip-gram model transparently consumes (word, context word) pairs constructed from monolingual as well as sentence aligned bilingual corpora.

We use a parametric composition model to construct sentence embeddings from word embeddings.

We process word embeddings with a Bi-directional LSTM and then take an average of the LSTM outputs, which can be viewed as context dependent word embeddings.

Since our multilingual skip-gram and cross-lingual sentence similarity models are trained jointly, they can inform each other through the shared word embedding layer and promote the compositionality of learned word embeddings at training time.

Further, the gradients flowing back from the sentence similarity model can affect the embeddings learned for words outside the vocabulary of the parallel corpora.

We hypothesize these two aspects of our model lead to more robust sentence embeddings.

Our contributions are as follows :• Scalable approach: We show that our approach performs better as more languages are added, since represent the extended lexicon in a suitable manner.• Ability to perform well in low-resource scenario: Our approach produces representations comparable with the state-of-art multilingual sentence embeddings using a limited amount of parallel data.

Our sentence embedding model is trained end-to-end on a vocabulary significantly larger than the vocabulary of the parallel corpora used for learning crosslingual sentence similarity.• Amenable to Multi-task modeling: Our model can be trained jointly with proxy tasks, such as sentiment classification, to produce more robust embeddings for downstream tasks.

This section gives a brief survey of relevant literature.

For a through survey of cross-lingual text embedding models, please refer to BID31 .

BID33 and 4.

joint optimization: using both parallel and monolingual corpora BID19 ; BID24 ; BID36 ; BID9 .

We adopt the skip-gram architecture of BID24 and train a single multilingual model using monolingual data from each language as well as any sentence aligned bilingual data available for any language pair.

Cross-lingual Sentence Embeddings: Some works dealing with cross-lingual word embeddings have considered the problem of constructing sentence embeddings including BID34 ; BID29 BID14 .

In general, it is not trivial to construct crosslingual sentence embeddings by composing word embeddings as the semantics of a sentence is a complex language-dependent function of its component words as well as their ordering.

BID29 addresses this difficulty by extending the paragraph vector model of BID22 to the bilingual context which models the sentence embedding as a separate context vector used for predicting the n-grams from both sides of the parallel sentence pair.

At test time, the sentence vector is randomly initialized and trained as part of an otherwise fixed model to predict the n-grams of the given sentence.

Our sentence embedding model is closer to the approach taken in BID14 .

They construct sentence embeddings by taking average of word or bi-gram embeddings and use a noise-contrastive loss based on euclidean distance between parallel sentence embeddings to learn these embeddings.

Multi-task Learning: Multi-task learning has been employed in various NLP applications where the parameters are shared among tasks BID7 ; BID23 ; BID12 .

BID23 show the effectiveness of multi-task learning in multiple sentiment classification tasks by sharing an RNN layer across tasks while learning separate prediction layers for each task.

BID37 recently showed benefits of learning a common semantic space for multiple tasks which share a low level feature dictionary.

Our multi-task architecture treats training multilingual word embeddings as a separate task with a separate objective as opposed to training them beforehand or training them only as part of a larger model.

Our model is trained to optimize two separate objectives: multilingual skip-gram BID24 and cross-lingual sentence similarity.

These two tasks are trained jointly with a shared word embedding layer in an end-to-end fashion.

Overview of the architecture that we use for computing sentence representations R S and R T for input word sequences S and T .

Multilingual skip-gram model BID24 extends the traditional skip-gram model by predicting words from both the monolingual and the cross-lingual context.

The monolingual context consists of words neighboring a given word as in the case of the traditional skip-gram model.

The cross-lingual context, on the other hand, consists of words neighboring the target word aligned with a given source word in a parallel sentence pair.

FIG0 , shows an example alignment, where an aligned pair of words are attached to both their monolingual and bilingual contexts.

For a pair of languages L1 and L2, the word embeddings are learned by optimizing the traditional skip-gram objective with (word, context word) pairs sampled from monolingual neighbors in L1 → L1 and L2 → L2 directions as well as cross-lingual neighbors in L1 → L2 and L2 → L1 directions.

In our setup, cross-lingual pairs are sampled from parallel corpora while monolingual pairs are sampled from both parallel and monolingual corpora.

We use a parametric composition model to construct sentence embeddings from word embeddings.

We process word embeddings with a bi-directional LSTM Hochreiter et al. (2001) ; BID15 and then take an average of the LSTM outputs.

There are various implementations of LSTMs available; in this work we use an implementation based on BID40 .

The LSTM outputs (hidden states) contextualize input word embeddings by encoding the history of each word into its representation.

We hypothesize that this is better than averaging word embeddings as sentences generally have complex semantic structure and two sentences with different meanings can have exactly the same words.

In FIG1 , the word embeddings x i are processed with a bi-directional LSTM layer to produce h i .

Bi-directional LSTM outputs are then averaged to get a sentence representation.

Learning Method: Let R : S → R d denote our sentence encoder mapping a given sequence of words S to a continuous vector in R d .

Given a pair of parallel sentences (S, T ), we define the loss L of our cross-lingual sentence encoder model as: DISPLAYFORM0 Therefore, for similar sentences (S ≈ T ), we minimize the loss L ST between their embeddings.

We also use a noise-constrastive large-margin update to ensure that the representations of non-aligned sentences observe a certain margin from each other.

For every parallel sentence pair (S, T ) we randomly sample k negative sentences N i , i = 1 . . .

k. With high probability N i is not semantically equivalent to S or T .We define our loss for a parallel sentence pair as follows: DISPLAYFORM1 Without the LSTM layer, this sentence encoder is similar to the BiCVM Hermann & Blunsom (2014) except that we use also the reversed sample (T, S) to train the model, therefore showing each pair of sentences to the model two times per epoch.

Following the literature, we use The Europarl corpus v71 BID20 for initial development and testing of our approach.

We use the first 500K parallel sentences for each of the EnglishGerman (en-de), English-Spanish (en-es) and English-French (en-fr) language pairs.

We keep the first 90% for training and the remaining 10% for development purposes.

We also use additional 500K monolingual sentences from the Europarl corpus for each language.

These sentences do not overlap with the sentences in parallel data.

Words which have a frequency less than 5 for a language are replaced with the <unk> symbol.

In the joint multi-task setting, the word frequencies are counted using the combined monolingual and parallel corpora.

When using just the parallel data for the en-de pair, the vocabulary sizes are 39K for German (de) and 21K for English (en).

Vocabulary sizes are 120K for German and 68K for English when both the parallel and the monolingual data are used.

We evaluate our model on the RCV1/RCV2 cross-lingual document classification task where for each language we use 1K documents for training and 5K documents for testing.

A. Multilingual Skip-gram: We use stochastic gradient descent with a learning rate of 0.01 and exponential decay of 0.98 after 10k steps ( 1 step = 256 word pairs), negative sampling with 128 samples, skip-gram context window of size 5.

Reducing the learning rate of the skip-gram model helps in the multi-task scenario by allowing skip-gram objective to converge in parallel with the sentence similarity objective.

We do this modification to make sure that shared word embeddings receive enough supervision from the multilingual sentence similarity objective.

At every step, we sample equal number of monolingual and cross-lingual word pairs to make a mini-batch.

We keep the batch size to be 50 sentence pairs.

LSTM hidden dimension P is one of 100, 128, 512 depending on the model.

We use dropout at the embedding layer with drop probability 0.3.

Hinge-loss margin m is always kept to be sentence embedding size.

We sample 5 negative samples for the noise-contrastive loss.

The model is trained using the Adam optimizer with a learning rate of 0.001 and an exponential decay of 0.98 after 10k steps ( 1 step = 50 sentence pairs = 1 mini-batch ).The system is optimized by alternating between mini-batches of these two tasks.

All of our models project words from all input languages to a shared vector space.

We train four types of models.• Sent-Avg: This model simply averages word embeddings to get a sentence embedding.

It is similar to BiCVM-add model from Hermann & Blunsom FORMULA0 , but we also add sentence pairs in the opposite direction, so that the model performs well in both directions.• Sent-LSTM: Represents words in context using the bidirectional LSTM layer, which are then averaged to get sentence embeddings.• JMT-Sent-Avg: Multilingual skip-gram jointly trained with Sent-add.

In this setting, the model is optimized by alternating between mini-batches for the two models.

JMT refers to Joint Multi-task.• JMT-Sent-LSTM: Multilingual skip-gram jointly trained with Sent-LSTM.

We report results on the Reuters RCV1/RCV2 cross-lingual document classification (CLDC) task BID19 using the same experimental setup.

We learn the distributed representations on the Europarl corpus.

We construct document embeddings by averaging sentence embeddings.

Sentence representations are fixed vectors determined by a sentence encoder trained on parallel and monolingual Europarl corpora.

For a language pair L1-L2, a document classifier (single layer average perceptron) is trained using the document representations from L1, and tested on documents from L2.

Due to lack of supervision on the test side, CLDC setup relies on documents with similar meaning having similar representations.

Table 1 , shows the results for our systems and compares it to some state-of-the-art approaches.

When the sentence embedding dimension is 128, we outperform most of the systems compared.

When the sentence embedding dimension is increased to 512, our results are close to the best results obtained for this task.

Our models with an LSTM layer (Sent-LSTM and JMT-Sent-LSTM) are significantly better than those without one.

There are also significant gains when the document embeddings are obtained from sentence encoders trained in the multi-task setting.

The ablation experiments where we just use parallel corpora suggest that these gains are mostly due to additional monolingual data that we can exploit in the multi-task setting.

Table 2 : We compare our JMT-Sent-LSTM model trained on three languages to one trained on two languages.

Table 2 compares models trained on data from four languages (en, es, de, fr) to models trained on data from two languages.

The results suggest that models trained on multiple languages perform better when English is the source language used to train the CLDC system.

The multilingual systems also show promising results for es-de pair, for which there was no direct parallel data available.

Validation loss for JMT-sent-add model shows more stability and achieves a lower value than the one for Sent-add model in the low-resource scenario.

At every training step, the validation set is created by randomly choosing 50 sentences from the development set.

The main motivation behind the multi-task architecture is to create high quality multilingual embeddings for languages which have limited amount of parallel data available.

Therefore, we compare the effectiveness of our Joint multi-task models in the low resource scenario, where for each language pair we use 100k parallel sentences and 1 million monolingual sentences for training the sentence encoder.

We evaluate on the RCV1/RCV2 document classification task.

Like before, we keep the first 90% (90k parallel sentences) of parallel data for training and 10% (10k parallel sentences) for development purposes.

FIG2 shows the loss curves for sent-add and JMT-Sent-add models.

On the validation set, JMTSent-add model gives a smoother and lower loss curve.

Our results suggest that using a parametric composition model to derive sentence embeddings from word embeddings and joint multi-task learning of multilingual word and sentence embeddings are promising directions.

This paper is a snapshot of our current efforts and w e believe that our sentence embedding models can be improved further with straightforward modifications to the model architecture, for instance by using stacked LSTMs, and we plan to explore these directions in future work.

In our exploration of architectures for the sentence encoding model, we also tried using a selfattention layer following the intuition that not all words are equally important for the meaning of a sentence.

However, we later realized that the cross lingual sentence similarity objective is at odds with what we want the attention layer to learn.

When we used self attention instead of simple averaging of word embeddings, the attention layer learns to give the entire weight to a single word in both the source and the target language since that makes optimizing cross lingual sentence similarity objective easier.

Even though they are related tasks, multilingual skip-gram and cross-lingual sentence similarity models are always in a conflict to modify the shared word embeddings according to their objectives.

This conflict, to some extent, can be eased by careful choice of hyper-parameters.

This dependency on hyper-parameters suggests that better hyper-parameters can lead to better results in the multi-task learning scenario.

We have not yet tried a full sweep of the hyperparameters of our current models but we believe there may be easy gains to be had from such a sweep especially in the multi-task learning scenario.

<|TLDR|>

@highlight

We jointly train a multilingual skip-gram model and a cross-lingual sentence similarity model to learn high quality multilingual text embeddings that perform well in the low resource scenario.