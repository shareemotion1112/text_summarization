Learning word representations from large available corpora relies on the distributional hypothesis that words present in similar contexts tend to have similar meanings.

Recent work has shown that word representations learnt in this manner lack sentiment information which, fortunately, can be leveraged using external knowledge.

Our work addresses the question: can affect lexica improve the word representations learnt from a corpus?

In this work, we propose techniques to incorporate affect lexica, which capture fine-grained information about a word's psycholinguistic and emotional orientation, into the training process of Word2Vec SkipGram, Word2Vec CBOW and GloVe methods using a joint learning approach.

We use affect scores from Warriner's affect lexicon to regularize the vector representations learnt from an unlabelled corpus.

Our proposed method outperforms previously proposed methods on standard tasks for word similarity detection, outlier detection and sentiment detection.

We also demonstrate the usefulness of our approach for a new task related to the prediction of formality, frustration and politeness in corporate communication.

In natural language research, words, sentences and paragraphs are considered in context through vector space representations, rather than as atomic units with no relational information among them.

Although n-gram based methods trained on large volumes of data have been found to outperform more complex approaches both on computational cost and accuracy, the techniques do not scale well in cases where the corpus size is limited(for example, for labeled speech or affect corpora with a size of a few millions of words).

Recent work has attempted to improve the performance of word distributions for downstream tasks such as sentiment analysis BID27 and knowledge base completion BID16 using lexical knowledge to enrich word embeddings, by performing methods such as regularization or introducing a loss term in the learning objective.

Sentiment relationships between words can be considered transitive, where 'good' < 'better' < 'best' implies that 'good' < 'best'.

However, word representations based on traditional approaches such as Word2Vec BID19 and GloVe BID23 are agnostic to the associated sentiments, emotions, or more generally affects BID4 .

Furthermore, although words such as delighted and disappointed share similar vector representations given their similar contexts, these words are associated with opposite reactions (or sentiments) as well as have a fairly different interpreted meaning.

The challenge in using syntactic relational information for sentiment detection, is that sentiment relations are transitive and symmetric (i.e., if 'delighted' is the opposite of 'disappointed', then 'disappointed' is the opposite of 'delighted'.)

Ignoring the bipolar nature of words could lead to spurious results, especially in predictive tasks related to synonyms and antonyms and sentiment analysis.

On the other hand, incorporating affect-related information would make word distributions homogeneous and suitable for speech and text generation tasks that aim at capturing author or reader reactions.

Furthermore, by using a small sentiment lexicon, it is possible to develop an automatic way to rate words based on their vector space representations.

This could help reduce the time and cost required to gather word ratings, as well as eliminate the implicit biases that may be introduced in annotations, such as the high correlation between high valence ratings with high arousal reported by BID27 .We present an approach to build affect-enriched word representations.

In other words, we enhance word distributions by incorporating reactions and affect dimensions.

The output of this work produces word distributions that capture human reactions by modeling the affect information in the words.

The affective word representations distinguish between semantically similar words that have varying affective interpretations.

Affect is represented as a weighted relational information between two words, following the approach used by existing work.

BID27 identify words of opposite polarity by performing signed spectral clustering on pre-trained embeddings.

We present an approach to incorporate external affect and reaction signals in the pre-training step, using the hand-annotated affect lexica to learn from.

Our experiments are based on using the state-of-the-art Warriner's affect lexicon BID30 as the input.

The proposed approach builds on the intuition that relationships between synonyms and antonyms can be characterized using semantic dictionaries and the relationship can then be deterministically captured into the training loss functions.

We evaluate the proposed enriched word distributions on standard natural language tasks.

We predict formality, frustration and politeness on a labeled dataset and show improved results using the enriched word embeddings.

Further, we outperform the state-of-the-art for sentiment prediction on standard datasets.

The key contributions of this paper include:• Algorithm to incorporate affect sensors in the cost functions of distributional word representations (including Word2Vec SkipGram, Word2Vec CBOW, and GloVe) during training using semantic and external affect signals.• Establish the utility of affect enriched word-embeddings for linguistic tasks such as Sentiment and Formality prediction in text data.

Our method out performs the state-of-the-art with an 20% improvement in accuracy for the outlier detection methods.

Detailed results are reported in table 1.• Introduce a workflow to incorporate affective and reaction signals to word representations during pre-training.

We show the generalizability of the workflow through experiments on 3 existing embeddings; Word2Vec-CBOW, Word2Vec-SkipGram, and GloVe.

Section 2 covers the prior art in both pre-training and post-training approaches for distributional word representations.

Section 3 presents the proposed approach and detailed experiments are discussed in section 4.

We conclude with a discussion on the learnings and the observations through this process 5.

A huge amount of research has explored how to use external resources to improve on Word2Vec BID19 and GloVe BID23 embeddings.

Research has refined embeddings for various downstream tasks such as dependency parsing BID0 , sentiment analysis BID27 and knowledge base completion BID16 .

There are mainly two approaches: joint learning and post-training.

The former considers the incorporation of external knowledge into the training process of learning word embeddings itself.

On the other hand, post-training approaches take already trained embeddings and use additional information to modify them.

Our approach falls in the first category.

To the best of our knowledge, no prior work focuses on improving word embeddings by jointly learning from a corpus and an affect lexica.

define a new basis for word representation and explore syntactic, semantic and morphological knowledge to provide additional information.

A binary indicator function is used to define relations.

BID32 propose a Relation Constrained Model(RCM) which predicts one word from another related word.

They use a linear combination of the objectives in CBOW and RCM for joint learning.

BID15 build specialized word embeddings for either similarity or relatedness.

They use a joint learning approach by using additional context words from external sources with the SkipGram loss function.

Our Word2Vec approach is similar to the pre-training model used in BID15 .

Essentially, words in the pruned list L i pruned (section 3) can also be taken as additional context words, similar to their understanding.

However, in our case, this addition is from an affect lexica and includes a strength associated with it.

BID31 propose a general framework RC-NET to incorporate knowledge into the word distributions.

They encode external relational and categorical information into regularization functions and combine them with the original objective of Word2Vec SkipGram model.

Our modified loss function can also be thought of as similar to the one in "Categorical Knowledge Powered model" in BID31 , thinking of the strength as a similarity score, although the distance function there is just defined as the Euclidean distance.

BID3 use a similar approach as ours for GloVe method to include various relations like synonyms, antonyms, hyponyms and so on.

They make use of a binary function to indicate whether the relation between any two words exists or not.

For post-training, BID8 makes no assumption about input pre-trained word vectors.

This work proposes an objective to further refine the input embeddings using relational information from semantic lexica.

Another post training approach has been proposed in BID22 which defines the final objective as a weighted sum of "Antonym Repel", "Synonym Attract" and "Vector Space Preservation" objectives.3 JOINT LEARNING FROM UNLABELED CORPUS AND AFFECT LEXICA 3.1 NOTATIONS Consider a corpus C and an affect lexica L consisting of l(word, affect-score) pairs.

Let us denote the i th pair in L with p i = (w i , s i ).

We define a function S(i, j) which captures the strength of the relationship between any two words w i and w j in L. To incorporate an affect lexica into the process of learning word embeddings, we follow a three step approach (See FIG0 ):Step 1:

Identifying Synonyms and Antonyms -Given a pair p i in L, WordNet BID20 is used to identify all pairs p j in L such that w j is either a synonym or an antonym of w i .

Synonyms and antonyms are retrieved based on WordNet definitions.

Note that, S(i, j) will be defined as non-zero only when w j is returned by WordNet as the synonym or antonym of w i .

For example, consider a pair of words from the Warriner's affect lexicon (see section 4.1 for lexicon details): although 'confident' and 'funny' have similar valence scores of 7.56 and 7.59 respectively (on a scale of 1-9), they do not share a semantic relationship between them (i.e. not defined as synonyms or antonyms).

Here, S(i, j) is set to 0.Step 2: Defining strength S(i, j) for all possible (i, j) pairs -Polarity information is captured in our modeling by centering the scores around 0 instead of the 1-9 scale.

As already mentioned, S(i, j) is defined as 0 if w j is not a synonym or antonym of w i .

For all other cases, the strength models the difference in affect scores of the two words under consideration.

If the words have scores with the same sign, we define a positive strength inversely proportional to the relative distance between them.

If the words have scores with opposite signs, the strength is negative, with magnitude directly proportional to the difference in their scores.

Algorithm 1 describes the approach.

return strength 15: end functionStep 3: Loss function definition -We introduce a new loss function for the embedding training.

The loss functions defined for Word2Vec and GloVe are described here.

SkipGram and CBOW techniques.

We build on top of that intuition.

For the SkipGram approach, the model predicts the context of an input word in the corpus.

Using the same notation as introduced in Rong (2014), the loss function of Word2Vec model BID19 ) with negative sampling is defined as the following: DISPLAYFORM0 where w O is the output word (i.e. the positive sample) and v w O is it's output vector.

h is the output value of the hidden layer.

For the SkipGram approach, h is simply v w I , which is the input vector corresponding to the input word w I .

W neg = {w j |1, ..., K} is the set of all negative samples.

The standard unigram distribution raised to the 3 4 th power for this sampling is used for all reported experiments.

Information from affect lexica is incorporated using another loss function in the following manner: DISPLAYFORM1 where v wj is the output vector as already defined, h is the hidden layer output, i is the index of the input word w I in L and S(i, j) is the relation strength.

L i pruned is the pruned list of words obtained from WordNet corresponding to the input word w I (same as the word w i in L).Note that i will only be defined if the input word w I belongs to the affect lexica L. Hence, the loss function will only matter when a word present in L appears at the input end.

The final loss function is defined as: DISPLAYFORM2 where λ ≥ 0 is a hyper-parameter.

This modification results in a modified form of the derivative of E with respect to (v T wj h), as given by the following equation: The weight update equations can be further obtained using equation 4 as in Rong FORMULA0 .

Intuitively, if the strength is positive for a pair of words, the back-propagation algorithm makes their corresponding embeddings closer.

Alternatively, if the strength is negative, the algorithm tries to move them apart.

The magnitude of the strength controls the speed of this movement.

DISPLAYFORM3 Inspired from negative sampling done from the vocabulary, we also try sampling the words from the entire affect lexica instead of using WordNet.

This suffers from the semantic relationship problems discussed in step one, the performance, hence declines.

The model used in CBOW is the opposite of the one defined for SkipGram.

Given a context window around a word, the model predicts that word.

Hence, the hidden layer output is defined as h = 1 C C c=1 v wc , the combination of all input word vectors corresponding to the context of the output word.

The loss function is formulated similar to the one already described.

GloVe BID23 Loss Function: We achieve this by using a weighted regularized version of the original GloVe objective.

Unlike Word2Vec, which considers only local cooccurrences, this approach uses global co-occurrences over our entire corpus.

We use the original GloVe implementation 1 , first extracting the vocabulary and building a cooccurrence matrix X. Borrowing the notation from BID23 , the objective for GloVe is as follows: DISPLAYFORM4 where V is the vocabulary, b p and b q are real-valued bias terms, w p is the target vector for the word w p andw q is the corresponding context vector for the word w q .

f (X pq ) is a weighting function defined as: DISPLAYFORM5 Using the same values as in BID23 , we keep α as 3 4 and x max as 100.

With the strength function S(i, j)(defined in Step 2) in mind, we define a regularization term as follows: DISPLAYFORM6 where indices i and j in L correspond to p and q in V respectively.

The final objective is obtained as follows: DISPLAYFORM7 using the same hyperparameter λ as before.

The obtained expression is similar to the one used in BID3 , except that instead of using a strength function S(i, j), they use a binary function R(i, j), which indicates whether a relation exists between words w i and w j or not.

We direct the readers to BID3 for the changes in update equations, which also result in similar values.

FIG1 illustrates the proposed method through the t-distributed Stochastic Neighbor Embedding (t-SNE) visualization of the first two components of a subset of words.

The points are colored to identify their positive (blue) or negative (red) valence in the affect lexicon.

As shown in FIG1 , 'accept', 'reject' and 'refuse' were found close together in the original representation.

By adding valence information to baseline embeddings FIG1 , the model is able to (a) pull words of similar sentiment closer together and (b) pull words of opposite sentiment, further apart.

This shows that our method has face-validity in being able to identify and distinguish sentiment polarities within word embeddings.

In this section, we conduct experiments to examine whether incorporating affect information into learning continuous word representations can improve the intrinsic quality of word embeddings and the performance of trained models on downstream tasks.

First, we introduce the dataset and then describe the evaluation framework.

ukWaC corpus: We use the ukWaC BID1 ) corpus for building our word embeddings.

It is large enough to obtain embeddings of a good quality, while still being tractable in terms of time and space resources.

We flattened the dependency-parsed format of this corpus, resulting in 2.2 billion tokens and a vocabulary size of 569,574 words after removing the words having frequency count of less than 20.We experimented with different values for λ and found that the highest similarity score on RG word similarity dataset BID26 was with a λ = 2.

The details of this experiment are provided in the Appendix.

Accordingly, we have used this value for λ throughout.

We use a window size of 10 and keep the word embedding dimensions as 300.

For Word2Vec, we use negative sampling of 15 words for optimization.

For GloVe, we found that running the model for 5 iterations was sufficient.

On a machine with 8 core Intel 3.4GHz processor and 16GB RAM, the Word2Vec skipgram approach takes 15 hours, CBOW takes under 100 minutes while GloVe takes approximately 15 minutes per iteration.

We compare the performance of the word2vec skipgram, word2vec CBOW and GloVe models for the following settings:• The baseline approach corresponding to setting λ as 0, i.e. only training on the unlabeled ukWaC corpus • With λ = 2, incorporating either valence, arousal and dominance scores to the original corpus one at a time.

We refer to these models as '+V','+A' and '+D' respectively • Incorporating the mean weight of valence, arousal and dominance to the original corpus.

We refer to this approach as '+VAD'.• Comparison against the state of the art: We reimplemented the approach by BID3 on our dataset.

The authors use a binary indicator function for the incorporation of various relations such as synonyms and antonyms in the training process.

In the original paper, this approach, trained on GloVe embeddings, demonstrably improved the state of the art on standard word similarity and analogy prediction.

We pick their best performing model which uses synonym pairs and train it on our ukWaC corpus with the same parameter settings.• Comparison against post-training baseline (Append): We compare our results against a simple post-training baseline in which we concatenate the word-vectors (D-dimensional WV) with VAD-vectors (3-dimensional AV) resulting in a D + 3 dimensional vector.

The considered dataset might not contain VAD ratings for stop words and proper nouns.

Since they are neutral in sentiment (V), strength (D) and arousal (A), for these out-of-dictionary words the VAD vector is set to neutral ν = [5, 5, 5] in all our experiments.1.

Normalize both embeddings with their L2-Norms FORMULA8 ).

This reduces the individual vectors to unit-length.

DISPLAYFORM0 2.

These regularized distributional word-embeddings are then concatenated with regularized affect scores as shown in Equation FORMULA0 .

DISPLAYFORM1 3.

We also perform the standardization process on the resultant D + 3 dimensional embeddings.

This transforms each dimension in the vector to have unit variance and zero mean so that each dimension contributes approximately proportionately.

DISPLAYFORM2 where µ and σ represent the mean and std.

deviation respectively.

The resultant vectors capture affect properties of the word in a distributional setting.

Warriner affect word list: We use affective information from Warriner's affect lexicon BID30 ) which comprises 13,915 words tagged on Valence, Arousal and Dominance scores on a scale of 1-9.

Valence is the unhappy-happy scale, Arousal is the calm-excited scale and Dominance indicates the forcefulness of expressed affect.

In all our experiments, we use a normalized scale from -4 to 4 to take the signed information into account, similar to BID27 .

We evaluate the proposed method on three standard tasks: predicting the similarity of words on seven benchmark datasets, detecting outliers in semantic clusters on the 8-8-8 dataset (CamachoCollados & Navigli, 2016) and predicting sentiment on the Stanford Sentiment Treebank .

We then introduce a new dataset and task for formality, frustration and politeness detection for a labeled email corpus.

The task is to predict the similarity between given two words.

We compute the cosine similarity between the corresponding word embeddings of the two words and assign it as the similarity score.

We consider seven benchmark datasets:• SIMLEX: the 999 word pairs list BID12 • MC: 30 word pairs in Miller Charles BID21 • MEN: 3000 pairs of words BID5 • RG: 65 word pairs by Rubenstein-Goodenough BID26 • RW: 2034 pairs in the Rare-Words dataset BID18 ),• SCWS: 2023 word pairs in Stanford's contextual word similarities (Huang et al., 2012)• WordSim: 353 word-pairs in the WordSim test collection BID10 ).

Each word pair in these benchmarks has a manually assigned score which we consider as gold standard ratings.

We report the Spearman Correlation Coefficient between the gold standard similarity ratings and the cosine similarity scores assigned by our model.

The results are provided in TAB1 .

The rows correspond to the approaches, with baselines corresponding to λ = 0.

The columns correspond to the various datasets discussed above.

Our proposed modifications show reasonable improvements for GloVe and word2vec CBOW embeddings.

The poorest performance was observed for the word2vec skipgram approach, which did not outperform the baseline approach for most of the datasets.

In TAB1 we show that our approach showed modest improvements over the state of the art method by BID3 on word similarity for two datasets.

We surmise that the nature of the word similarity task and the absence of "opposite" words in the benchmark datasets could be the reason why non-affective approaches are hard to beat for predicting word similarity.

For outlier prediction, our approaches gives a performance improvement of 5% over the approach by BID3 .

Word similarity tasks have been widely used for measuring the semantic coherence of vector space models.

However, such tasks often suffer from low inter-annotator agreement scores of gold standard datasets.

Hence, we also report our results on an outlier detection task BID6 , in order to test the quality of semantic clusters in the vector space models.

These results are on the 8-8-8 outlier detection dataset BID6 containing 8 different topics, each made up of a cluster of 8 words and 8 possible outliers.

TAB1 summa-rizes the Outlier Position Percentage(OPP score) and Accuracy of our different models.

We refer BID6 to the readers for further details on the dataset and evaluation metrics.

Apart from Word2Vec SkipGram models, our approaches perform well on both OPP and Accuracy scores(higher is better).

We observe that the incorporation of affect information in terms of Valence, Arousal and Dominance scores, shows improvements in an Outlier Detection task on an unrelated dataset with topics such as Football Teams, Solar Systems and Car Manufacturers.

Next, we evaluate our approach on two tasks for sentiment detection.

Since our approach enriches word embeddings with affect information, we anticipate that our models would outperform the state of the art on these tasks.

For the sentiment prediction task, we make use of an available Deep Averaging Network 2 (DAN) model BID14 with its default parameter settings along with our modified word embeddings.

We report the results on Stanford Sentiment Treebank(SST) dataset which contains fine grained sentiment labels for 215,154 phrases in the parse trees of 11,855 sentences.

We use the standard splits of the datasets.

TAB2 summarizes the results for sentiment prediction.

We report the accuracy values for both fine and binary classification settings.

For the latter, we remove all the instances with neutral labels.

Our approaches show significant improvements over the baselines for both these metrics.

As evident from TAB2 , our approaches perform better on sentiment prediction task than the state of the art approach BID3 .

We attribute this improvement to our strength function which not only specifies the existence of a relation between two words, but also provides the signed strength associated with it.

Finally, we evaluate the quality our embeddings on an affect prediction task.

We use a new dataset consisting of 980 emails from the ENRON dataset (Cohen, 2009), tagged with formality, frustration and politeness scores by human annotators.

We use a CNN based regression model which takes our obtained embeddings as inputs and predicts formality, frustration and politeness in emails.

We use a model which stacks a convolutional layer(5 filters of 10X5 size), a pooling layer(5X5 size with stride 5) and a dense layer(size 50, dropout 0.2).

Rectified Linear Unit(ReLU) activation is used throughout with Stochastic Gradient Descent(SGD) optimizer and a mean squared loss.

Results are provided in Table 3 .

For the affect prediction task, we report the mean and standard deviation of Mean Square Error(MSE) over five different train-test splits(test-ratio:0.2) of the dataset corresponding to Formality, Frustration and Politeness.

In this case, along with GloVe and CBOW approaches, our modifications over SkipGram baseline (corpus only) show improvements with low standard deviations in MSE values.

We find reasonable improvements by our proposed approaches in all the task-based evaluations.

SkipGram based methods perform poorly in word similarity prediction and outlier detection, but do well on sentiment and affect prediction.

This difference in performance on downstream tasks, has been discussed before in BID9 and BID6 , who point out various issues with word similarity based evaluations such as task subjectivity, low inter annotator agreements and low correlations between the performance of word vectors on word similarity and NLP tasks like text classification, parsing and sentiment analysis.

Performance differences can also be attributed to corpus size, which are examined in the Appendix section.

Table 3 : Performance of proposed approaches on affect prediction task: (a) In terms of Mean Square Error (MSE) values for affect prediction on a labeled email corpus, (b) Comparison with prior work.

The baseline model refers to the corpus only approach, with λ = 0.

λ is set to 2 for all other approaches: using Valence list(+V), Arousal(+A), Dominance(+D) and average strength(+VAD).(a) The results suggest that different embeddings perform well for different tasks.

In word similarity tasks, the +V model performs well in GloVe setting but the +A model seems to perform the best for CBOW.

Similar results are observed in sentiment prediction: for binary sentiment prediction, arousal scores give the best performance with CBOW embeddings but dominance and valence give the best performance with skip-gram and GloVe embeddings respectively.

This suggests that the most flexible method could be an ensemble implementation that considers all these inputs before predicting a final class.

Also note that given the vocabulary of our ukWaC corpus as 569, 574 words, our affect lexica with 13, 915 words is relatively small.

We plan to take this work forward by further analysis in the future.

At the least, we expect superior word embeddings with better quality and larger affect lexica.

Words may have different affect in different contexts.

This happens because their meanings change from one context to another.

For example "A wicked woman came in" vs a slang "This game is wicked" should have different affect scores for both "wicked" because the meaning of the word almost turns on its head (from "devilish" to "cool/commendable").

But datasets like BID30 do not capture this as the scores attributed by human annotators inherently average out the contexts (to the best of their knowledge) in which the words tend to appear.

This problem (of "affect-polysemy") is related to "polysemy" which cripples today's word-level embedding models BID11 BID13 ).

Word-level vector embeddings like SkipGram combine multiple senses of a polysemous word into single embedding vector BID17 ).

Although, there have been a few attempts to address this shortcoming using multi-prototype vectorspace models BID24 BID13 , BID29 ), but they also face several challenges BID17 ).Since, single vector embeddings are still the most popular choice for NLP tasks, the proposed approach tries to improve the quality of these vanilla embeddings for tasks which can benefit by inducing affect information.

Therefore, in this paper, we evaluate our approach on sentence/phrase level prediction tasks like sentiment prediction and tone of emails.

The underlying state-of-theart recurrent/convolutional models try to create context-aware sentence level representations from word-embeddings to predict information like sentiment, formality etc.

Our affect induced wordembeddings perform better than vanilla embeddings on these tasks thus supporting our claim.

We consider three models for analysis: GloVe baseline(corpus only approach), GloVe + Valence and GloVe + synonyms BID3 .

TAB4 shows the true class label and predictions for a few instances from SST test set.

We discuss these results below.

For the first two sentences, although the overall sentiment is clearly positive, all the three the models fail to capture this.

The models fail in both fine-grained and binary(positive/negative) settings.

The reason for poor predictions can be attributed to the absence of any affect-sensitive words.

On the other hand, third and fourth sentences are rich with affect information.

Words like 'entertainment', 'smart', 'sweet' and 'playful' make it easier for the models to predict the positive sentiment in these two sentences.

The fifth sentence has a negative overall sentiment, with a true label of -2.

Surprisingly, baseline model and BID3 approaches fail to capture this.

However, our approach is able to make the correct prediction.

This observation can be explained as follows.

The main affect word in the sentence is "dreadful", which is present in Warriner affect lexicon but not in synonym word pairs used by BID3 .

This enables our model to get improved embedding for this word.

This infact can be observed in the neighborhood of 'dreadful' compared in TAB5 .

The embedding for 'dreadful' is a lot closer to negative words for our approach than others.

We observe a similar behavior in last two sentences.

The better quality embeddings learnt for 'gem' and 'hate' help to make better predictions while other models make errors.

Including various other instances which we observe, the presence or absence or affect words in the lexicon plays a vital role.

Since we work in a joint learning framework, embeddings for other words are also improved, but this improvement is significant for words present in the lexicon only.

The inconsistency in the performance of our models can be attributed to this along with the quality of affect lexicon itself.

This work proposes methods to incorporate information from an affect lexicon into Word2Vec and GloVe training process.

In a nutshell, we first use WordNet to identify word pairs in the affect lexicon which are semantically related.

We define the strength of this relationship using available affect scores.

Finally, we modify the training objectives to incorporate this information.

In order to evaluate our embeddings, we compare them with baseline approaches where the training completely ignores the affect information.

Our embeddings show improvements over baselines on not only Word Similarity benchmarks but also on a more complex, Outlier Detection task.

We also do this comparison extrinsically and show that our modified embeddings perform better over prior work in predicting sentiment and predicting formality, frustration and politeness in emails.

Among models using Valence, Arousal or Dominance score lists, there is no clear winner but overall addition of valence scores does a reasonable job in almost all of the cases.1.

Choosing an appropriate value for hyper-parameter λ:In order to choose a suitable value for λ, we take a 100 MB sample of ukWaC corpus.

The sample has close to 20 million tokens, with a vocabulary size of 27,978 words, eliminating all the words having the frequency count of less than 20.

We choose a smaller corpus for tuning as it is more manageable with respect to space and time resources.

We train a Word2Vec SkipGram model on the above 100MB sample and Valence affect lists by using all the λ value from the set (0, 0.5, 1, 2, 10, 100, 1000) one by one.

To pick the most suitable value, we compare the results on word similarity task on the Rubenstein-Goodenough(RG) dataset BID26 .

The results are given in FIG2 .

Since λ = 2.0 performs the best, we fix this value for all our experiments.

We conduct an error analysis of the poor performance of Word2Vec Skipgram by observing the effect of varying the corpus size.

We take different sized samples from the ukWaC corpus and report the word similarity performance on RG dataset in FIG3 .We observe irregularities in the performance of baseline approach(λ=0).

Adding the affective information has a negative impact for corpora with sizes 2.5GB and 4.5GB while shows minor improvements over baseline for larger corpora.

This improvement over baseline is the most for a smaller, 100MB corpus.

We believe better preprocessing on the corpus should help with these non-intuitive observations.

<|TLDR|>

@highlight

Enriching word embeddings with affect information improves their performance on sentiment prediction tasks.

@highlight

Proposes to use affect lexica to improve word embeddings to outperform the standard Word2vec and Glove.

@highlight

This paper proposes integrating information from a semantic resource quantifying the affect of words into a text-based word embedding algorithm to make language models more reflective of semantic and pragmatic phenomena.

@highlight

This paper introduces modifications the word2vec and GloVe loss functions to incorporate affect lexica to facilitate the learning of affect-sensitive word embeddings.