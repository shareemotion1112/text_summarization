We propose an effective multitask learning setup for reducing distant supervision noise by leveraging sentence-level supervision.

We show how sentence-level supervision can be used to improve the encoding of individual sentences, and to learn which input sentences are more likely to express the relationship between a pair of entities.

We also introduce a novel neural architecture for collecting signals from multiple input sentences, which combines the benefits of attention and maxpooling.

The proposed method increases AUC by 10% (from 0.261 to 0.284), and outperforms recently published results on the FB-NYT dataset.

Early work in relation extraction from text used fully supervised methods, e.g., BID2 , which motivated the development of relatively small datasets with sentence-level annotations such as ACE 2004 BID2 , BioInfer and SemEval 2010 .

Recognizing the difficulty of annotating text with relations, especially when the number of relation types of interest is large, BID16 pioneered the distant supervision approach to relation extraction, where a knowledge base (KB) and a text corpus are used to automatically generate a large dataset of labeled sentences which is then used to train a relation classifier.

Distant supervision provides a practical alternative to manual annotations, but introduces many noisy examples.

Although many methods have been proposed to reduce the noise in distantly supervised models for relation extraction (e.g., BID8 BID23 BID22 BID5 BID27 BID11 , a rather obvious approach has been understudied: using sentence-level supervision to augment distant supervision.

Intuitively, supervision at the sentence-level can help reduce the noise in distantly supervised models by identifying which of the input sentences for a given pair of entities are likely to express a relation.

We experiment with a variety of model architectures to combine sentence-and bag-level supervision and find it most effective to use the sentence-level annotations to directly supervise the sentence encoder component of the model in a multi-task learning framework.

We also introduce a novel maxpooling attention architecture for combining the evidence provided by different sentences where the entity pair is mentioned, and use the sentence-level annotations to supervise attention weights.

The contributions of this paper are as follows:??? We propose an effective multitask learning setup for reducing distant supervision noise by leveraging existing datasets of relations annotated at the sentence level.???

We propose maxpooled attention, a neural architecture which combines the benefits of maxpooling and soft attention, and show that it helps the model combine information about a pair of entities from multiple sentences.??? We release our library for relation extraction as open source.

1 The following section defines the notation we use, describes the problem and provides an overview of our approach.

Our goal is to predict which relation types are expressed between a pair of entities (e 1 , e 2 ), given 1 We attach the anonymized code as supplemental material in this submission instead of providing a github link in order to maintain author anonymity.

Figure 1: An overview of our approach for augmenting distant supervision with sentence-level annotations.

The left side shows one sentence in the labeled data and how it is used to provide direct supervision for the sentence encoder.

The right side shows snippets of the text corpus and the knowledge base, which are then combined to construct one training instance for the model, with a bag of three input sentences and two active relations: 'founder of' and 'ceo of'.all sentences in which both entities are mentioned in a large collection of unlabeled documents.

Following previous work on distant supervision, we use known tuples (e 1 , r, e 2 ) in a knowledge base K to automatically annotate sentences where both entities are mentioned.

In particular, we group all sentences s with one or more mentions of an entity pair e 1 and e 2 into a bag of sentences B e 1 ,e 2 , then automatically annotate this bag with the set of relation types L distant = {r ??? R : (e 1 , r, e 2 ) ??? K}, where R is the set of relations we are interested in.

We use 'positive instances' to refer to cases where |L| > 0, and 'negative instances' when |L| = 0.In this paper, we leverage existing datasets with sentence-level relation annotations in a similar domain, where each example consists of a token sequence s, token indexes for e 1 and e 2 in the sequence, and one relation type (or 'no relation').

Since the relation types annotated at the sentence level may not correspond one-to-one to those in the KB, we replace the relation label associated with each sentence with a binary indicator.

(1 indicates that the sentence s expresses one of the relationships of interest.)

We do not require the entities to match those in the KB either.

Fig. 1 illustrates how we modify neural architectures commonly used in distant supervision, e.g., BID13 to effectively incorporate sentence-level supervision.

The model consists of two components: 1) A sentence encoder (displayed in blue) reads a sequence of tokens and their relative distances from e 1 and e 2 , and outputs a vector s representing the sentence encoding, as well as P (e 1 ??? e 2 | s) representing the probability that the two entities are related given this sentence.

2) The bag encoder (displayed in green) reads the encoding of each sentence in the bag for the pair (e 1 , e 2 ) and predicts P (r = 1 | e 1 , e 2 ), ???r ??? R. We combine both bag-level (i.e., distant) and sentence-level (i.e., direct) supervision in a multi-task learning framework by minimizing the weighted sum of the cross entropy losses for P (e 1 ??? e 2 | s) and P (r = 1 | e 1 , e 2 ).

By sharing the parameters of sentence encoders used to compute either loss, the sentence encoders become less susceptible to the noisy bag labels.

The bag encoder also benefits from sentence-level supervision by using the supervised distribution P (e 1 ??? e 2 | s) to decide the weight of each sentence in the bag, using a novel architecture which we call maxpooled attention.

The model predicts a set of relation types L pred ??? R given a pair of entities e 1 , e 2 and a bag of sentences B e 1 ,e 2 .

In this section, we first describe the sentence encoder part of the model FIG2 , then describe the bag encoder FIG2 , then we explain how the two types of supervision are jointly used for training the model end-to-end.

Given a sequence of words w 1 , . . .

, w |s| in a sentence s, a sentence encoder translates this sequence into a fixed length vector s.

Input Representation.

The input representation is illustrated graphically with a table at the bottom of FIG2 .

We map word token i in the sentence w i to a pretrained word embedding vector w i .

2 Another crucial input signal is the position of entity mentions in each sentence s ??? B e 1 ,e 2 .

Following BID28 , we map the distance between each word in the sentence and the entity mentions 3 to a small vector of learned parameters, namely d i .

Instead of randomly initializing position embeddings with mean = 0, we obtain notable performance improvements by randomly initializing all dimensions of the position embedding for distance d around the mean value d. Intuitively, this makes it easier to learn useful parameters since the embedding of similar distances (e.g., d = 10 and d = 11) should be similar, without adding hard constraints on how they should be related.

We find that adding a dropout layer with a small probability (p = 0.1) before the sentence encoder reduces overfitting and improves the results.

To summarize, the input layer for a sentence s is a sequence of vectors: DISPLAYFORM0 Word Composition.

Word composition is illustrated with the block CNN in the bottom part of FIG2 , which represents a convolutional neural network (CNN) with multiple filter sizes.

The outputs of the maxpool operations for different filter sizes are concatenated then projected into a smaller vector using one feed forward linear layer.

This is in contrast to previous work BID19 which used Piecewise CNN (PCNN).

In PCNN, we convolve three segments of the sentence separately: windows before the left entity, windows inbetween the two entities and windows after the right entity.

Every split is maxpooled independently, then the three vectors are concatenated.

The intuition is that this helps the model put more emphasis on the middle segment which : Blue box is the sentence encoder, it maps a sentence to a fixed length vector (CNN output) and the probability it expresses a relation between e 1 and e 2 (sigmoid output).

Green box is the bag encoder, it takes encoded sentences and their weights and produces a fixed length vector (maxpool output), concatenates it with entity embeddings (pointwise mult.

output) then outputs a probability for each relation type r. White boxes contain parameters that the model learns while gray boxes do not have learnable parameters.

Sentence-level annotations supervise P (e 1 ??? e 2 | s).

Bag-level annotations supervise P (r = 1 | e 1 , e 2 ).connects the two entities.

As discussed later in Section 4.2, we compare CNN and PCNN and find the simpler CNN architecture works better.

Sentence encoding s is computed as follows: DISPLAYFORM1 where CNN x is a standard convolutional neural network with filter size x, W 1 and b 1 are model parameters and s is the sentence encoding.

We feed the sentence encoding s into a ReLU layer followed by a sigmoid layer with output size 1, representing P (e 1 ??? e 2 | s), as illustrated in DISPLAYFORM2 where ?? is the sigmoid function and W 2 , b 2 , W 3 , b 3 are model parameters.

Given a bag B e 1 ,e 2 of n ??? 1 sentences, we compute their encodings s 1 , . . .

, s n as described earlier and feed them into the bag encoder, which is responsible for combining the information in all sentence encodings and predict the probability P (r = 1 | e 1 , e 2 ), ???r ??? R. The bag encoder also makes use of p = P (e 1 ??? e 2 | s) from Eq. 1 as an estimate of the degree to which sentence s expresses the relation between e 1 and e 2 .Maxpooled Attention.

To aggregate the sentence encodings s 1 , . . .

, s n into a fixed length vector that captures the important features in the bag, BID11 used maxpooling, while BID13 used soft attention.

In this work, we propose maxpooled attention, a new form of attention which combines some of the characteristics of maxpooling and soft attention.

Given the encoding s j and an unnormalized weight u j for each sentence s j ??? B e 1 ,e 2 , the bag encoding g is a vector with the same dimensionality as s j with the k-th element computed as: DISPLAYFORM0 Maxpooled attention has the same intuition of soft attention; learning weights for sentences that enable the model to focus on the important sentences.

However, maxpooled attention differs from soft attention in two aspects.

The first is that every sentence s j is given a probability that indicates how useful the sentence is, independently of the other sentences.

Notice how this is different from soft attention where sentences compete for probability mass, i.e., probabilities must sum to 1.

This is implemented in maxpooled attention by normalizing the weight of each sentence with a sigmoid function rather than a softmax.

This is a better fit for the task at hand because the sentences are not competing.

It also makes the weights useful even when |B e 1 ,e 2 | = 1, while soft attention will always normalize such weights to 1.The second difference between maxpooled attention and soft attention is the use of weighted maxpooling instead of weighted average.

Maxpooling is more effective for this task because it can pick the useful features from different sentences.

As shown in FIG2 , we do not directly use the p from Eq. 1 as weight in maxpooled attention.

Instead, we found it useful to feed it into more non-linearities.

The unnormalized maxpooled attention weight for s j is computed as: DISPLAYFORM1 Entity Embeddings.

Following Ji et al. (2017) , we use entity embeddings to improve our model of relations in the distant supervision setting, although our formulation is closer to that of BID25 who used point-wise multiplication of entity embeddings: m = e 1 e 2 , where is point-wise multiplication, and e 1 and e 2 are the embeddings of e 1 and e 2 , respectively.

In order to improve the coverage of entity embeddings, we use pretrained GloVe vectors BID19 (same embeddings used in the input layer).For entities with multiple words, like "Steve Jobs", the vector for the entity is the average of the GloVe vectors of its individual words.

If the entity is expressed differently across sentences, we average the vectors of the different mentions.

As discussed in Section 4.2, this leads to big improvement in the results, and we believe there's still big room for improvement from having better representation for entities.

We feed the output m as additional input to the last block of our model.

Output Layer.

The final step is to use the bag encoding g and the entity pair encoding m to predict a set of relations L pred which is a standard multilabel classification problem.

We concatenate g and m and feed them into a feedforward layer with ReLU non-linearity, followed by a sigmoid layer with an output size of |R|: DISPLAYFORM2 where r is a vector of Bernoulli variables each of which corresponds to one of the relations in R. This is the final output of the model.

To train the model on the bag-level labels obtained with distant supervision, we use binary crossentropy loss between the model predictions and the labels obtained with distant supervision, i.e., bag loss = DISPLAYFORM0 where r distant [k] = 1 indicates that the tuple (e 1 , r k , e 2 ) is in the knowledge base.

In addition to the bag-level supervision commonly used in distant supervision, we also use sentence-level annotations.

One approach is to create a bag of size 1 for each sentence-level annotation, and add the bags to those obtained with distant supervision.

However, this approach requires mapping relations in the sentence-level annotations map to those in the KB.Instead, we found that the best use of the supervised data is to improve the model's ability to predict the the potential usefulness of a sentence by using sentence-level annotations to help supervise the sentence encoder module.

According to our analysis of baseline models, distinguishing between positive and negative examples is the real bottleneck.

This supervision serves two purposes: it improves our encoding of each sentence, and improves the weights used by the maxpooled attention to decide which sentences should contribute more to the bag encoding.

We maximize log loss of gold labels in the sentence-level data D according to the model described in Eq. 1: DISPLAYFORM1 where D consists of all the sentence-level annotations in addition to all distantly-supervised negative examples.

4 We jointly train the model on both types of supervision.

The model loss is a weighted sum of the sentence-level and the bag-level losses: DISPLAYFORM2 where ?? is a parameter that controls the contribution of each loss, tuned on a validation set.

This section discusses datasets, metrics, experiment configurations and the models we are comparing with.

Metrics.

Prior works used precision-recall (PR) curves to show their results on the FB-NYT dataset.

In this multilabel classification setting, all model predictions for all relation types are sorted by confidence from highest to lowest.

Then applying different thresholds gives the points on the PR curve.

We use the area under the PR curve (AUC) for early stopping and hyperparameter tuning.

Because we are interested in the high-precision extractions, we focus on the high-precision lowrecall part of the PR curve.

That is, in our experiments, we only keep points on the PR curve with recall below 0.4 which means that the largest possible value for AUC is 0.4.Configurations.

The FB-NYT dataset does not have a validation set for hyper-parameter tuning and early stopping.

For these, use the test set, and BID13 use 3-fold cross validation.

We use 90% of the training set for training and keep the other 10% for validation.

The pretrained word embeddings we use are 300-dimensional GloVe vectors, trained on 42B tokens.

Since we do not update word embeddings 6 while training the model, our vocabulary may include any word which appears in the training, validation or test sets with frequency greater than two.

When a word with a hyphen (e.g., 'five-star') is not in the GloVe vocabulary, we average the embeddings of its subcomponents.

Otherwise, all OOV words are assigned the same random vector (normal with mean 0 and standard deviation 0.05).Our model is implemented using PyTorch and AllenNLP BID6 and trained on machines with P100 GPUs.

Each run takes five hours on average.

We train for a large number of epochs and use early stopping with patience = 3.

The batches of the two datasets are randomly shuffled before every epoch.

The optimizer we use is Adam with its default PyTorch parameters.

We run every configuration three times with three different random seeds then report the PR curve for the run with the best validation (thresholded) AUC.

In the controlled experiments, we report the mean and standard deviation of the AUC metric.

Compared Models.

Our baseline for comparison is a model that is similar to what is described in Section 3 with the following configurations.

It uses our approach for position embedding initialization, encodes sentences using CNN, uses entity embeddings, aggregate sentences using maxpooling and does not use the sentence-level annotation.

Our best configuration adds the maxpooled attention and the sentence-level annotations.

We also compare with existing models in the literature.

The model by BID13 uses an attention mechanism that assigns weights to each sentence followed by a weighted average of sentence encodings.

The model by extends the model by BID13 by using soft labels during training.

Main Result.

FIG3 summarizes the main results of our experiments.

8 The AUC of our baseline (green) is comparable to that of Lin et al. (2016) (blue) , which verifies that we are building on a strong baseline.

Adding maxpooled attention and sentence-level supervision (i.e., the full model, in red) substantially improves over the baseline (green).

The figure also illustrates that our full model outperforms the strong baseline 8 Results of BID13 and are copied from their papers.

0.271 ?? 0.007 + additional bags 0.269 ?? 0.001 + sentence loss 0.284 ?? 0.007 Table 1 : The + and ??? signs indicate independent changes to the baseline configuration.of in orange.

9 We emphasize that the improved results reported here conflate both additional supervision and model improvements.

Next, we report the results of controlled experiments to quantify the contribution of each modification in Table 1 .

The first line in the table is the baseline model and configuration described in the previous section and in FIG3 , and the + and ??? signs indicate (independent) additions to and removals from that configuration, respectively.

Position Embedding Initialization.

The second line in Table 1 shows that removing the distance-based initialization of position embeddings results in a large drop in AUC.

We hypothesize that the position-based initialization reduces the burden of finding optimal values for position embeddings, without explicit constraints that guarantee similar distances to have similar embeddings.

Sentence Encoder.

In the next line of Table 1 , we replace the simpler CNN in our baseline with the more complex PCNN BID27 .

Both encoders use filters of sizes 2, 3, 4 and 5.

Table 1 shows that using CNN works markedly better than PCNN which is in contrast to the findings of BID27 .

This could be due to the use of multiple filter sizes and to the improved representation of entity positions in our model, which may obviate the need to have a separate encoding of each segment in the sentence.

Entity Embeddings.

The next line in Table 1 shows that entity embeddings (which are included in the baseline model) provide valuable information and help predict relations.

This information may encode entity type, entity compatibility with each others and entity bias to participate in a relation.

Given that our entity embeddings are simple GloVe vectors, we believe there is still a large room for improvement.

We compare different ways of aggregating sentences into a single vector including maxpooling (baseline, originally proposed in BID11 ), attention BID13 and our proposed maxpooled attention.

10 Maxpooling works better than soft attention because it is better at picking out useful features from multiple sentences, while attention can only weight the whole representation of the sentence.

We hypothesize that our proposed maxpooled attention works better than both because it combines the soft attention's ability to learn and use different weights for different sentences, and the maxpool's ability to pick out useful features from multiple sentence.

Another advantage of maxpooled attention over attention is that it helps in cases where bag size equals 1 because the softmax typically used in attention results in a weight of 1 for the sentence rendering that weight useless.

The last three lines in Table 1 compare different ways for using sentence-level annotations.

The line "baseline + maxpooled att." is copied from the pre-10 Our reimplementation of BID13 attention differs from what was described in the paper.

The unnormalized attention weights of BID13 are oj = sj ?? A ?? q where sj is the sentence encoding, A is a diagonal matrix and q is the query vector.

We tried this but found that implementing it as a feedforward layer with output size = 1 works better.

The results in Table 1 are for the feedforward implementation.

vious line and is the basis for the following two lines.

In "additional bags," we add the sentencelevel annotations as additional bags along with the distantly supervised data.

In "sentence loss," we use the method described in Section 3 for integrating sentence-level supervision.

The results show that simply adding the sentence-level supervised data to the distantly supervised data as additional bags has little effect on the performance.

This is probably because they change the distribution of the training to differ from the test set.

However, adding the sentence-level supervision following our proposed multitask learning improves the results considerably because it allows the model to better filter noisy sentences.

Selecting Lambda.

Although we did not spend much time tuning hyperparameters, we made sure to carefully tune ?? (Equation 3) which balances the contribution of the two losses.

Early experiments showed that sentence-level loss is typically smaller than bag-level loss, so we experimented with ?? ??? {0, 0.5, 1, 2, 4, 8, 16, 32, 64} .

FIG4 shows thresholded AUC for different values of ??, where each point is the average of three runs.

It is clear that picking the right value for ?? has a big impact on the final result.

Qualitative Analysis.

An example of a positive bag is shown in TAB3 .

Our model, which incorporates sentence-level supervision, assigns the most weight to the first sentence while the attention model assigns the most weight to the last sentence (which is less informative for the relation between the two entities).

Furthermore, the attention model does not use the other two sentences because their weights are dominated by the weight of the last sentence.

We also found that the weights from our model usually range between 0 and 0.08, suggesting the relative values of the weights are informative to the model, even when the absolute values are small.

Distant Supervision.

The term 'distant supervision' was coined by BID16 who used relation instances in a KB (Freebase, Bollacker et al., 2008) to identify any sentence in a text corpus where two related entities are mentioned, then developed a classifier to predict the relation.

Researchers have since extended this approach for relation extraction (e.g., BID24 BID15 BID21 BID12 .

A key source of noise in distant supervision is that sentences may mention two related entities without expressing the relation between them.

BID8 used multi-instance learning to address this problem by developing a graphical model for each entity pair which includes a latent variable for each sentence to explicitly indicate the relation expressed by that sentence, if any.

Our model can be viewed as an extension of BID8 where the sentence-bound latent variables can also be directly supervised in some of the training examples.

Neural Models for Distant Supervision.

More recently, neural models have been effectively used to model textual relations (e.g., BID7 BID28 BID17 .

Focusing on distantly supervised models, BID27 proposed a neural implementation of multi-instance learning to leverage multiple sentences which mention an entity pair in distantly supervised relation extraction.

However, their model picks only one sentence to represent an entity pair, which wastes the information in the neglected sentences.

BID11 addresses this limitation by maxpooling the vector encodings of all input sentences for a given entity pair.

BID13 independently proposed to use attention to address the same limitation.

Results in Section 4.2 suggest that maxpooling is more effective than attention for multi-instance learning.

BID26 proposed a method for leveraging dependencies between different relations in a pairwise ranking framework.

Sentence-Level Supervision.

Despite the substantial amount of work on both fully supervised and distantly supervised relation extraction, the question of how to combine both signals has been mostly ignored in the literature, with a few exceptions.

Nguyen and Moschitti (2011) first manually defined a mapping between relation types in YAGO to compatible relation types in ACE 2004 BID4 , then trained two separate SVM models using the training portion of ACE 2004 and the distantly supervised sentences.

Model predictions are then linearly combined to make the final prediction.

In contrast, we use a neural model which combines both sources of supervision in a multi-task learning framework BID3 .

We also do not require a strict mapping between the relation types of the KB and those annotated at the sentence level.

Another important distinction is the unit of prediction (at the sentence level vs. at the entity pair level), each of which has important practical applications.

Also related is BID0 who used active learning to improve the multi-instance multi-label model of BID23 .

We propose two complementary methods to improve performance and reduce noise in distantly supervised relation extraction.

The first is incorporating sentence-level supervision and the second is maxpooled attention, a novel form of attention.

The sentence-level supervision improves sentence encoding and provides supervision for attention weights, while maxpooled attention effectively combines sentence encodings and their weights into a bag encoding.

Our experiments show a 10% improvement in AUC (from 0.261 to 0.284) outperforming recently published results on the FB-NYT dataset .

@highlight

A new form of attention that works well for the distant supervision setting, and a multitask learning approach to add sentence-level annotations. 