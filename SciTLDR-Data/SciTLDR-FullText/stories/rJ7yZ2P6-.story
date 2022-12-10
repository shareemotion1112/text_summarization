Ubuntu dialogue corpus is the largest public available dialogue corpus to make it feasible to build end-to-end deep neural network models directly from the conversation data.

One challenge of Ubuntu dialogue corpus is  the large number of out-of-vocabulary words.

In this paper we proposed an algorithm which combines the general pre-trained word embedding vectors with those  generated on the task-specific training set to address this issue.

We integrated character embedding into Chen et al's Enhanced LSTM method (ESIM) and used it to evaluate the effectiveness of our proposed method.

For the task of next utterance selection, the proposed method has demonstrated a significant performance improvement against original ESIM and the new model has achieved state-of-the-art results on both Ubuntu dialogue corpus and Douban conversation corpus.

In addition, we investigated the performance impact of end-of-utterance and end-of-turn token tags.

The ability for a machine to converse with human in a natural and coherent manner is one of challenging goals in AI and natural language understanding.

One problem in chat-oriented humanmachine dialog system is to reply a message within conversation contexts.

Existing methods can be divided into two categories: retrieval-based methods BID31 BID8 BID36 and generation based methods BID28 .

The former is to rank a list of candidates and select a good response.

For the latter, encoder-decoder framework BID28 or statistical translation method BID19 are usually used to generate a response.

It is not easy to main the fluency of the generated texts.

Ubuntu dialogue corpus BID13 is the public largest unstructured multi-turns dialogue corpus which consists of about one-million two-person conversations.

The size of the corpus makes it attractive for the exploration of deep neural network modeling in the context of dialogue systems.

Most deep neural networks use word embedding as the first layer.

They either use fixed pre-trained word embedding vectors generated on a large text corpus or learn word embedding for the specific task.

The former is lack of flexibility of domain adaptation.

The latter requires a very large training corpus and significantly increases model training time.

Word out-of-vocabulary issue occurs for both cases.

Ubuntu dialogue corpus also contains many technical words (e.g. "ctrl+alt+f1", "/dev/sdb1").

The ubuntu corpus (V2) contains 823057 unique tokens whereas only 22% tokens occur in the pre-built GloVe word vectors 1 .

Although character-level representation which models sub-word morphologies can alleviate this problem to some extent BID7 BID3 BID11 , character-level representation still have limitations: learn only morphological and orthographic similarity, other than semantic similarity (e.g. 'car' and 'bmw') and it cannot be applied to Asian languages (e.g. Chinese characters).In this paper, we generate word embedding vectors on the training corpus based on word2vec BID16 .

Then we propose an algorithm to combine the generated one with the pre-trained word embedding vectors on a large general text corpus based on vector concatenation.

The new word representation maintains information learned from both general text corpus and taskdomain.

The nice property of the algorithm is simplicity and little extra computational cost will be added.

It can address word out-of-vocabulary issue effectively.

This method can be applied to most NLP deep neural network models and is language-independent.

We integrated our methods with ESIM(baseline model) .

The experimental results have shown that the proposed method has significantly improved the performance of original ESIM model and obtained state-ofthe-art results on both Ubuntu Dialogue Corpus and Douban Conversation Corpus BID34 .

On Ubuntu Dialogue Corpus (V2), the improvement to the previous best baseline model (single) on R 10 @1 is 3.8% and our ensemble model on R 10 @1 is 75.9%.

On Douban Conversation Corpus, the improvement to the previous best model (single) on P @1 is 3.6%.Our contributions in this paper are summarized below:1.

We propose an algorithm to combine pre-trained word embedding vectors with those generated on the training corpus to address out-of-vocabulary word issues and experimental results have shown that it is very effective.2.

ESIM with our method has achieved the state-of-the-art results on both Ubuntu Dialogue corpus and Douban conversation corpus.3.

We investigate performance impact of two special tags on Ubuntu Dialogue Corpus: endof-utterance and end-of-turn.

The rest paper is organized as follows.

In Section 2, we review the related work.

In Section 3 we provide an overview of ESIM (baseline) model and describe our methods to address out-ofvocabulary issues.

In Section 4, we conduct extensive experiments to show the effectiveness of the proposed method.

Finally we conclude with remarks and summarize our findings and outline future research directions.

Character-level representation has been widely used in information retrieval, tagging, language modeling and question answering.

BID23 represented a word based on character trigram in convolution neural network for web-search ranking.

BID3 represented a word by the sum of the vector representation of character n-gram.

Santos et al BID20 BID21 and BID11 used convolution neural network to generate character-level representation (embedding) of a word.

The former combined both word-level and character-level representation for part-of-speech and name entity tagging tasks while the latter used only character-level representation for language modeling.

BID39 employed a deep bidirectional GRU network to learn character-level representation and then concatenated word-level and character-level representation vectors together.

BID38 used a fine-grained gating mechanism to combine the word-level and character-level representation for reading comprehension.

Character-level representation can help address out-of-vocabulary issue to some extent for western languages, which is mainly used to capture character ngram similarity.

The other work related to enrich word representation is to combine the pre-built embedding produced by GloVe and word2vec with structured knowledge from semantic network ConceptNet BID25 and merge them into a common representation BID24 .

The method obtained very good performance on word-similarity evaluations.

But it is not very clear how useful the method is for other tasks such as question answering.

Furthermore, this method does not directly address out-of-vocabulary issue.

Next utterance selection is related to response selection from a set of candidates.

This task is similar to ranking in search, answer selection in question answering and classification in natural language inference.

That is, given a context and response pair, assign a decision score BID2 .

BID8 formalized short-text conversations as a search problem where rankSVM was used to select response.

The model used the last utterance (a single-turn message) for response selection.

On Ubuntu dialogue corpus, BID13 proposed Long Short-Term Memory(LSTM) (Hochreiter & Schmidhuber, 1997) siamese-style neural architecture to embed both context and response into vectors and response were selected based on the similarity of embedded vectors.

BID9 built an ensemble of convolution neural network (CNN) BID10 and Bi-directional LSTM.

BID2 employed a deep neural network structure BID27 where CNN was applied to extract features after bi-directional LSTM layer.

BID40 treated each turn in multi-turn context as an unit and joined word sequence view and utterance sequence view together by deep neural networks.

BID34 explicitly used multi-turn structural info on Ubuntu dialogue corpus to propose a sequential matching method: match each utterance and response first on both word and sub-sequence levels and then aggregate the matching information by recurrent neural network.

The latest developments have shown that attention and matching aggregation are effective in NLP tasks such as question/answering and natural language inference.

BID22 introduced context-to-query and query-to-context attentions mechanisms and employed bi-directional LSTM network to capture the interactions among the context words conditioned on the query.

Parikh et al.(2016) compared a word in one sentence and the corresponding attended word in the other sentence and aggregated the comparison vectors by summation.

enhanced local inference information by the vector difference and element-wise product between the word in premise an the attended word in hypothesis and aggregated local matching information by LSTM neural network and obtained the state-of-the-art results on the Stanford Natural Language Inference (SNLI) Corpus.

BID33 introduced several local matching mechanisms before aggregation, other than only word-by-word matching.

In this section, we first review ESIM model and introduce our modifications and extensions.

Then we introduce a string matching algorithm for out-of-vocabulary words.

In our notation, given a context with multi-turns DISPLAYFORM0 with length m and a response R = (r 1 , r 2 , · · · , r j , · · · , r n ) with length n where c i and r j is the ith and jth word in context and response, respectively.

For next utterance selection, the response is selected based on estimating a conditional probability P (y = 1|C, R) which represents the confidence of selecting R from the context C. FIG0 shows high-level overview of our model and its details will be explained in the following sections.

Word Representation Layer.

Each word in context and response is mapped into d-dimensional vector space.

We construct this vector space with word-embedding and character-composed embedding.

The character-composed embedding, which is newly introduced here and was not part of the original forumulation of ESIM, is generated by concatenating the final state vector of the forward and backward direction of bi-directional LSTM (BiLSTM).

Finally, we concatenate word embedding and character-composed embedding as word representation.

Context Representation Layer.

As in base model, context and response embedding vector sequences are fed into BiLSTM.

Here BiLSTM learns to represent word and its local sequence context.

We concatenate the hidden states at each time step for both directions as local context-aware new word representation, denoted byā andb for context and response, respectively.

DISPLAYFORM1 DISPLAYFORM2 where w is word vector representation from the previous layer.

Attention Matching Layer.

As in ESIM model, the co-attention matrix E ∈ R m×n where E ij = a T ib j .

E ij computes the similarity of hidden states between context and response.

For each word in context, we find the most relevant response word by computing the attended response vector in Equation 3.

The similar operation is used to compute attended context vector in Equation 4.

DISPLAYFORM3 DISPLAYFORM4 After the above attended vectors are calculated, vector difference and element-wise product are used to enrich the interaction information further between context and response as shown in Equation 5 , the diagram addes character-level embedding and replaces average pooling by LSTM last state summary vector. and 6.

DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 where the difference and element-wise product are concatenated with the original vectors.

Matching Aggregation Layer.

As in ESIM model, BiLSTM is used to aggregate response-aware context representation as well as context-aware response representation.

The high-level formula is given by DISPLAYFORM8 DISPLAYFORM9 Pooling Layer.

As in ESIM model, we use max pooling.

Instead of using average pooling in the original ESIM model, we combine max pooling and final state vectors (concatenation of both forward and backward one) to form the final fixed vector, which is calculated as follows: DISPLAYFORM10 DISPLAYFORM11 DISPLAYFORM12 Prediction Layer.

We feed v in Equation 11 into a 2-layer fully-connected feed-forward neural network with ReLu activation.

In the last layer the sigmoid function is used.

We minimize binary cross-entropy loss for training.

Many pre-trained word embedding vectors on general large text-corpus are available.

For domainspecific tasks, out-of-vocabulary may become an issue.

Here we propose algorithm 1 to combine pre-trained word vectors with those word2vec BID16 ) generated on the training set.

Here the pre-trainined word vectors can be from known methods such as GloVe BID18 , word2vec BID16 and FastText BID3 .Algorithm 1: Combine pre-trained word embedding with those generated on training set.

Input : Pre-trained word embedding set {U w |w ∈ S} where U w ∈ R d1 is embedding vector for word w. Word embedding {V w |w ∈ T } are generated on training set where V w ∈ R d2 .

P is a set of word vocabulary on the task dataset.

Output: A dictionary with word embedding vectors of dimension DISPLAYFORM0

We evaluate our model on the public Ubuntu Dialogue Corpus V2 2 BID14 ) since this corpus is designed for response selection study of multi turns human-computer conversations.

The corpus is constructed from Ubuntu IRC chat logs.

The training set consists of 1 million < context, response, label > triples where the original context and corresponding response are labeled as positive and negative response are selected randomly on the dataset.

On both validation and test sets, each context contains one positive response and 9 negative responses.

Some statistics of this corpus are presented in Table 1 .

Douban conversation corpus BID34 which are constructed from Douban group 3 (a popular social networking service in China) is also used in experiments.

Response candidates on the test set are collected by Lucene retrieval model, other than negative sampling without human judgment on Ubuntu Dialogue Corpus.

That is, the last turn of each Douban dialogue with additional keywords extracted from the context on the test set was used as query to retrieve 10 response candidates from the Lucene index set (Details are referred to section 4 in BID34 BID34 .

Our model was implemented based on Tensorflow BID0 .

ADAM optimization algorithm BID12 was used for training.

The initial learning rate was set to 0.001 and exponentially decayed during the training 4 .

The batch size was 128.

The number of hidden units of biLSTM for character-level embedding was set to 40.

We used 200 hidden units for both context representation layers and matching aggregation layers.

In the prediction layer, the number of hidden units with ReLu activation was set to 256.

We did not use dropout and regularization.

Word embedding matrix was initialized with pre-trained 300-dimensional GloVe vectors 5 BID18 .

For character-level embedding, we used one hot encoding with 69 characters (68 ASCII characters plus one unknown character).

Both word embedding and character embedding matrix were fixed during the training.

After algorithm 1 was applied, the remaining out-of-vocabulary words were initialized as zero vectors.

We used Stanford PTBTokenizer on the Ubuntu corpus.

The same hyper-parameter settings are applied to both Ubuntu Dialogue and Douban conversation corpus.

For the ensemble model, we use the average prediction output of models with different runs.

On both corpuses, the dimension of word2vec vectors generated on the training set is 100.

Since the output scores are used for ranking candidates, we use Recall@k (recall at position k in 10 candidates, denotes as R@1, R@2 below), P@1 (precision at position 1), MAP(mean average precision) BID1 , MRR (Mean Reciprocal Rank) BID29 ) to measure the model performance.

TAB3 show the performance comparison of our model and others on Ubuntu Dialogue Corpus V2 and Douban conversation corpus, respectively.

On Douban conversation corpus, FastText BID3 pre-trained Chinese embedding vectors 6 are used in ESIM + enhanced word vector whereas word2vec generated on training set is used in baseline model (ESIM).

It can be seen from table 3 that character embedding enhances Model 1 in 10 R@1 1 in 10 R@2 1 in 10 R@5 MRR TF-IDF BID14 0.488 0.587 0.763 -Dual Encoder w/RNN BID14 0.379 0.561 0.836 -Dual Encoder w/LSTM BID14 0.552 0.721 0.924 -RNN-CNN BID2 0.672 0.809 0.956 0.788 * MEMN2N BID5 0.637 ---* CNN + LSTM(Ensemble) BID9 0.683 0.818 0.957 -* Multi-view dual Encoder BID40 0.662 0.801 0.951 -* SMN dynamic BID34 0 BID34 0.208 0.390 0.422 CNN BID34 0.226 0.417 0.440 LSTM BID34 0.320 0.485 0.527 BiLSTM BID34 0.313 0.479 0.514 Multi-View BID40 BID34 0.342 0.505 0.543 DL2R BID35 BID34 0.330 0.488 0.527 MV-LSTM BID30 BID34 0.348 0.498 0.538 Match-LSTM BID32 BID34 0.345 0.500 0.537 Attentive-LSTM BID27 BID34 0.331 0.495 0.523 Multi-Channel BID34 0.349 0.506 0.543 SMN dynamic BID34 0.397 0.529 0.569 ESIM 0.407 0.544 0.588 ESIM + enhanced word vector (single) 0.433 0.559 0.607 Table 4 : Performance of the models on Douban Conversation Corpus.the performance of original ESIM.

Enhanced Word representation in algorithm 1 improves the performance further and has shown that the proposed method is effective.

Most models (RNN, CNN, LSTM, BiLSTM, Dual-Encoder) which encode the whole context (or response) into compact vectors before matching do not perform well.

SMN dynamic directly models sequential structure of multi utterances in context and achieves good performance whereas ESIM implicitly makes use of end-of-utterance( eou ) and end-of-turn ( eot ) token tags as shown in subsection 4.6.

In this section we evaluated word representation with the following cases on Ubuntu Dialogue corpus and compared them with that in algorithm 1.WP1 Used the fixed pre-trained GloVe vectors 7 .WP2 Word embedding were initialized by GloVe vectors and then updated during the training.

WP3 Generated word2vec embeddings on the training set BID16 and updated them during the training (dropout).

WP4 Used the pre-built ConceptNet NumberBatch BID26 8 .

eou and eot are missing from pre-trained GloVe vectors.

But this two tokens play an important role in the model performance shown in subsection 4.6.

For word2vec generated on the training set, the unique token coverage is low.

Due to the limited size of training corpus, the word2vec representation power could be degraded to some extent.

WP5 combines advantages of both generality and domain adaptation.

In order to check whether the effectiveness of enhanced word representation in algorithm 1 depends on the specific model and datasets, we represent a doc (context, response or query) as the simple average of word vectors.

Cosine similarity is used to rank the responses.

The performances of the simple model on the test sets are shown in FIG2 .

BID37 is an open-domain question answering dataset from Microsoft research.

The results on the enhanced vectors are better on the above three datasets.

This indicates that enhanced vectors may fuse the domain-specific info into pre-built vectors for a better representation.

There are two special token tags ( eou and eot ) on ubuntu dialogue corpus.

eot tag is used to denote the end of a user's turn within the context and eou tag is used to denote of a user utterance without a change of turn.

Table 7 shows the performance with/without two special tags.

Table 7 : Performance comparison with/without eou and eot tags on Ubuntu Dialogue Corpus (V2).It can be observed that the performance is significantly degraded without two special tags.

In order to understand how the two tags helps the model identify the important information, we perform a case study.

We randomly selected a context-response pair where model trained with tags succeeded and model trained without tags failed.

Since max pooling is used in Equations 9 and 10, we apply max operator to each context token vector in Equation 7 as the signal strength.

Then tokens are ranked in a descending order by it.

The same operation is applied to response tokens.

It can be seen from Table 8 that eou and eot carry useful information.

eou and eot captures utterance and turn boundary structure information, respectively.

This may provide hints to design a better neural architecture to leverage this structure information.

Model with tags i ca n't seem to get ssh to respect a changes(0.932) authorize keys file eou (0.920) is there anything i should do besides service ssh restart ?

eou (0.981) eot (0.957) restarting ssh should n't be necessary . .

sounds like there 's(0.935) a different problem .

are you sure the file is only readable by the owner ?

and the .

ssh directory is 700 ? eou eot (0.967)

yeah , it was set up initially by ubuntu/ec2 , i(0.784) just changed(0.851) the file(0.837) , but it 's neither(0.802) locking out the old key(0.896) nor(0.746) accepting the new one eouModel without tags i ca n't seem to get ssh to respect a changes authorize keys file is there anything i should do besides service ssh restart ?

restarting(0.930) ssh should n't be necessary .(0.958) .

sounds like there 's a different problem .

are(0.941) you sure(0.935) the file(0.973) is only readable by the owner ?

and the . ssh(0.949) directory is 700 ?

yeah , it was set up(0.787) initially by ubuntu/ec2 , i just changed the file(0.923) , but it 's neither locking(0.844) out the(0.816) old key nor(0.846)

accepting(0.933) the new one Table 8 : Tagged outputs from models trained with/without eou and eot tags.

The top 6 tokens with the highest signal strength are highlighted in blue color.

The value inside the parentheses is signal strength.

We propose an algorithm to combine pre-trained word embedding vectors with those generated on training set as new word representation to address out-of-vocabulary word issues.

The experimental results have shown that the proposed method is effective to solve out-of-vocabulary issue and improves the performance of ESIM, achieving the state-of-the-art results on Ubuntu Dialogue Corpus and Douban conversation corpus.

In addition, we investigate the performance impact of two special tags: end-of-utterance and end-of-turn.

In the future, we may design a better neural architecture to leverage utterance structure in multi-turn conversations.

@highlight

Combine information between pre-built word embedding and task-specific word representation to address out-of-vocabulary issue

@highlight

This paper proposes an approach to improve the out-of-vocabulary embedding prediction for the task of modeling dialogue conversations with sizable gains over the baselines.

@highlight

Proposes combining external pretrained word embeddings and pretrained word embeddings on training data by keeping them as two views.

@highlight

Proposes method to extend the coverage of pre-trained word embeddings to deal with the OOV problem that arises when applying them to conversational datasets and applies new variants of LSTM-based model to the task of response-selection in dialogue modeling.