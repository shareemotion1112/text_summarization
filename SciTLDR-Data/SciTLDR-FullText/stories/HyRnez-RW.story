Reading comprehension is a challenging task, especially when executed across longer or across multiple evidence documents, where the answer is likely to reoccur.

Existing neural architectures typically do not scale to the entire evidence, and hence, resort to selecting a single passage in the document (either via truncation or other means), and carefully searching for the answer within that passage.

However, in some cases, this strategy can be suboptimal,  since by focusing on a specific passage, it becomes difficult to leverage multiple mentions of the same answer throughout the document.

In this work, we take a different approach by constructing lightweight models that are combined in a cascade to find the answer.

Each submodel consists only of feed-forward networks equipped with an attention mechanism, making it trivially parallelizable.

We show that our approach can scale to approximately an order of magnitude larger evidence documents and can aggregate information from multiple mentions of each answer candidate across the document.

Empirically, our approach achieves state-of-the-art performance on both the Wikipedia and web domains of the TriviaQA dataset, outperforming more complex, recurrent architectures.

Reading comprehension, the task of answering questions based on a set of one more documents, is a key challenge in natural language understanding.

While data-driven approaches for the task date back to BID11 , much of the recent progress can be attributed to new largescale datasets such as the CNN/Daily Mail Corpus BID8 , the Children's Book Test Corpus BID9 and the Stanford Question Answering Dataset (SQuAD) BID21 .

These datasets have driven a large body of neural approaches BID24 BID16 BID22 BID27 , inter alia) that build complex deep models typically driven by long short-term memory networks BID12 .

These models have given impressive results on SQuAD where the document consists of a single paragraph and the correct answer span is typically only present once.

However, they are computationally intensive and cannot scale to large evidence texts.

Such is the case in the recently released TriviaQA dataset BID14 , which provides as evidence, entire webpages or Wikipedia articles, for answering independently collected trivia-style questions.

So far, progress on the TriviaQA dataset has leveraged existing approaches on the SQuAD dataset by truncating documents and focusing on the first 800 words BID14 BID18 .

This has the obvious limitation that the truncated document may not contain the evidence required to answer the question 1 .

Furthermore, in TriviaQA there is often useful evidence spread throughout the supporting documents.

This cannot be harnessed by approaches such as that greedily search for the best 1-2 sentences in a document.

For example, in Fig.1 the answer does not appear in the first 800 words.

The first occurrence of the answer string is not sufficient to answer the question.

The passage starting at token 4089 does contain all of the information required to infer the answer, but this inference requires us to resolve the two complex co-referential phrases in 'In the summer of that year they got married in a church'.

Access to other mentions of Krasner and Pollock and the year 1945 is important to answer this question.

Figure 1: Example from TriviaQA in which multiple mentions contain information that is useful in inferring the answer.

Only the italicized phrase completely answers the question (Krasner could have married multiple times) but contains complex coreference that is beyond the scope of current natural language processing.

The last phrase is more easy to interpret but it misses the clue provided by the year 1945.In this paper we present a novel cascaded approach to extractive question answering ( ??3) that can accumulate evidence from an order of magnitude more text than previous approaches, and which achieves state-of-the-art performance on all tasks and metrics in the TriviaQA evaluation.

The model is split into three levels that consist of feed-forward networks applied to an embedding of the input.

The first level submodels use simple bag-of-embeddings representations of the question, a candidate answer span in the document, and the words surrounding the span (the context).

The second level submodel uses the representation built by the first level, along with an attention mechanism BID2 that aligns question words with words in the sentence that contains the candidate span.

Finally, for answer candidates that are mentioned multiple times in the evidence document, the third level submodel aggregates the mention level representations from the second level to build a single answer representation.

At inference time, predictions are made using the output of the third level classifier only.

However, in training, as opposed to using a single loss, all the classifiers are trained using the multi-loss framework of BID1 , with gradients flowing down from higher to lower submodels.

This separation into submodels and the multi-loss objective prevents adaptation between features BID10 .

This is particularly important in our case where the higher level, more complex submodels could subsume the weaker, lower level models c.f.

BID1 .To summarize, our novel contributions are??? a non-recurrent architecture enabling processing of longer evidence texts consisting of simple submodels ??? the aggregation of evidence from multiple mentions of answer candidates at the representation level ??? the use of a multi-loss objective.

Our experimental results ( ??4) show that all the above are essential in helping our model achieve state-of-the-art performance.

Since we use only feed-forward networks along with fixed length window representations of the question, answer candidate, and answer context, the vast majority of computation required by our model is trivially parallelizable, and is about 45?? faster in comparison to recurrent models.

Most existing approaches to reading comprehension BID24 BID16 BID22 BID27 BID25 BID15 , inter alia) involve using recurrent neural nets (LSTMs BID12 or memory nets BID26 ) along with various flavors of the attention mechanism BID2 to align the question with the passage.

In preliminary experiments in the original TriviaQA paper, BID14 explored one such approach, the BiDAF architecture BID22 , for their dataset.

However, BiDAF is designed for SQuAD, where the evidence passage is much shorter (122 tokens on an average), and hence does not scale to the entire document in TriviaQA (2895 tokens on an average); to work around this, the document is truncated to the first 800 tokens.

Pointer networks with multi-hop reasoning, and syntactic and NER features, have been used recently in three architectures -Smarnet BID4 , Reinforced Mnemonic Reader BID13 and MEMEN BID18 for both SQuAD and TriviaQA.

Most of the above also use document truncation .Approaches such as first select the top (1-2) sentences using a very coarse model and then run a recurrent architecture on these sentences to find the correct span.

BID3 propose scoring spans in each paragraph with a recurrent network architecture separately and then take taking the span with the highest score.

Our approach is different from existing question-answering architectures in the following aspects.

First, instead of using one monolithic architecture, we employ a cascade of simpler models that enables us to analyze larger parts of the document.

Secondly, instead of recurrent neural networks, we use only feed-forward networks to improve scalability.

Third, our approach aggregates information from different mentions of the same candidate answer at the representation level rather than the score level, as in other approaches BID15 BID14 .

Finally, our learning problem also leverages the presence of several correct answer spans in the document, instead of considering only the first mention of the correct answer candidate.

For the reading comprehension task ( ??3.1), we propose a cascaded model architecture arranged in three different levels ( ??3.2).

Submodels in the lower levels ( ??3.3) use simple features to score candidate answer spans.

Higher level submodels select the best answer candidate using more expensive attention-based features ( ??3.4) and by aggregating information from different occurrences of each answer candidate in the document ( ??3.5).

The submodels score all the potential answer span candidates in the evidence document 2 , each represented using simple bags-of-embeddings.

Each submodel is associated with its own objective and the final objective is given by a linear combination of all the objectives ( ??3.6).

We conclude this section with a runtime analysis of the model ( ??3.7).

We take as input a sequence of question word embeddings q = {q 1 . . .

q m }, and document word embeddings d = {d 1 . . .

d n }, obtained from a dictionary of pre-trained word embeddings.

Each candidate answer span, s = {d s1 . . .

d so } is a collection of o ??? l consecutive word embeddings from the document, where l is the maximum length of a span.

The set of all candidate answer spans is S := {s i } nl i=1 .

Limiting spans to length l minimally affects oracle accuracy (see Section ??4) and allows the approach to scale linearly with document size.

Since the same spans of text can occur multiple times in each document, we also maintain the set of unique answer candidate spans, u ??? S u , and a mapping between each span and the unique answer candidate that it corresponds to, s u. In TriviaQA, each answer can have multiple alternative forms, as shown in Fig.1 .

The set of correct answer strings is S * and our task is to predict a single answer candidate?? ??? S.

We first describe our meta-architecture, which is a collection of simple submodels M k (??) organized in a cascade.

The idea of modeling separate classifiers that use complementary features comes from BID1 who found this gave considerable gains over combining all the features into a single model for a conversational system.

As shown in FIG0 , our architecture consists of two submodels M 1 , M 2 at the first level, one submodel M 3 at the second, and one submodel M 4 at the third level.

Each submodel M k returns a score, ?? DISPLAYFORM0 s that is fed as input to the submodel at the next level.

?? DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 Using their respective scores, ??s , the models M 1 ...M 3 define a distribution over all spans, while M 4 uses ?? (4) u to define a distribution over all unique candidates, as follows: DISPLAYFORM4 In training, our total loss is given by an interpolation of losses for each of M 1 , .., M 4 .

However, during inference we make a single prediction, simply computed as?? = arg max u???Su ??u .

The first level is the simplest, taking only bags of embeddings as input.

This level contains two submodels, one that looks at the span and question together ( ??3.3.1), and another that looks at the span along with the local context in which the span appears ( ??3.3.2).

We first describe the span representations used by both.

Span Embeddings: We denote a span of length o as a vector s, containing??? averaged document token embeddings, and ??? a binary feature ?? qs indicating whether the spans contains question tokens DISPLAYFORM0 The binary feature ?? qs is motivated by the question-in-span feature from BID3 , we use the question-in-span feature, motivated by the observation that questions rarely contain the answers that they are seeking and it helps the model avoid answers that are over-complete -containing information already known by the questioner.

The question + span component of the level 1 submodel predicts the correct answer using a feedforward neural network on top of fixed length question and span representations.

The span representation is taken from above and we represent the question with a weighted average of the question embeddings.

Question Embeddings: Motivated by BID16 we learn a weight ?? qi for each of the words in the question using the parameterized function defined below.

This allows the model to learn to focus on the important words in the question.

?? qi is generated with a two-layer feed-forward net with rectified linear unit (ReLU) activations BID17 BID7 , DISPLAYFORM0 where U, V, w, z, a and b are parameters of the feed-forward network.

Since all three submodels rely on identically structured feed-forward networks and linear prediction layers, from now on we will use the abbreviations ffnn and linear as shorthand for the functions defined above.

The scores, ?? qi are normalized and used to generate an aggregated question vectorq as follows.

DISPLAYFORM1 Now that we have representations of both the question and the span, the question + span model computes a hidden layer representation of the question-span pair as well as a scalar score for the span candidate as follows: DISPLAYFORM2 s ) where [x; y] represents the concatenation of vectors x and y.

The span + short context component builds a representation of each span in its local linguistic context.

We hypothesize that words in the left and right context of a candidate span are important for the modeling of the span's semantic category (e.g. person, place, date).

Unlike level 1 which builds a question representation independently of the document, the level 2 submodel considers the question in the context of each sentence in the document.

This level aligns the question embeddings with the embeddings in each sentence using neural attention BID2 BID16 BID27 BID22 , specifically the attend and compare steps from the decomposable attention model of .

The aligned representations are used to predict if the sentence could contain the answers.

We apply this attention to sentences, not individual span contexts, to prevent our computational requirements from exploding with the number of spans.

Subsequently, it is only because level 2 includes the level 1 representations h (1) s and h (2) s that it can assign different scores to different spans in the same sentence.

Sentence Embeddings: We define g s = {d gs,1 . . .

d g s,G } to be the embeddings of the G words of the sentence that contains s. First, we measure the similarity between every pair of question and sentence embeddings by passing them through a feed-forward net, ffnn att1 and using the resulting hidden layers to compute a similarity score, ??.

Taking into account this similarity, attended vectors q i andd gs,j are computed for each question and sentence token, respectively.

DISPLAYFORM0 The original vector and its corresponding attended vector are concatenated and passed through another feed-forward net, ffnn att2 the final layers from which are summed to obtain a question-aware sentence vector??? s , and a sentence context-aware question vector,q.

DISPLAYFORM1 Using these attended vectors as features, along with the hidden layers from level 1 and the questionspan feature, new scores and hidden layers are computed for level 2: DISPLAYFORM2 s ;q;??? s ; ?? qs ]), ?? DISPLAYFORM3 s )

In this level, we aggregate information from all the candidate answer spans which occur multiple times throughout the document.

The hidden layers of every span from level 2 (along with the question-in-span feature) are passed through a feed-forward net, and then summed if they correspond to the same unique span, using the s u map.

The sum, h u is then used to compute the score and the hidden layer 3 for each unique span, u in the document.

DISPLAYFORM0 DISPLAYFORM1 The hidden layer in level 3 is used only for computing the score ??u , mentioned here to preserve consistency of notation.

To handle distant supervision, previous approaches use the first mention of the correct answer span (or any of its aliases) in the document as gold BID14 .

Instead, we leverage the existence of multiple correct answer occurrences by maximizing the probability of all such occurrences.

Using Equation 1, the overall objective, (U * , V * , w * , z * , a * , b * ) is given by the total negative log likelihood of the correct answer spans under all submodels: DISPLAYFORM0 where ?? 1 , .., .?? 4 are hyperparameters, such that 4 i=1 ?? i = 1, to weigh the contribution of each loss term.

We briefly discuss the asymptotic complexity of our approach.

For simplicity assume all hidden dimensions and the embedding dimension are ?? and that the complexity of matrix(??????)-vector(????1) multiplication is O(?? 2 ).

Thus, each application of a feed-forward network has O(?? 2 ) complexity.

Recall that m is the length of the question, n is the length of the document, and l is the maximum length of each span.

We then have the following complexity for each submodel:Level 1 (Question + Span) :

Building the weighted representation of each question takes O(m?? 2 ) and running the feed forward net to score all the spans requires O(nl?? 2 ), for a total of O(m?? 2 + nl?? 2 ).

Level 2 : Computing the alignment between the question and each sentence in the document takes O(n?? 2 + m?? 2 + nm??) and then scoring each span requires O(nl?? 2 ).

This gives a total complexity of O(nl?? 2 + nm??), since we can reasonably assume that m < n.

Thus, the total complexity of our approach is O(nl?? 2 + mn??).

While the nl and nm terms can seem expensive at first glance, a key advantage of our approach is that each sub-model is trivially parallelizable over the length of the document (n) and thus very amenable to GPUs.

Moreover note that l is set to 5 in our experiments since we are only concerned about short answers.

The TriviaQA dataset BID14 contains a collection of 95k trivia question-answer pairs from several online trivia sources.

To provide evidence for answering these questions, documents are collected independently, from the web and from Wikipedia.

Performance is reported independently in either domain.

In addition to the answers from the trivia sources, aliases for the answers are collected from DBPedia; on an average, there are 16 such aliases per answer entity.

Answers and their aliases can occur multiple times in the document; the average occurrence frequency is almost 15 times per document in either domain.

The dataset also provides a subset on the development and test splits which only contain examples determined by annotators to have enough evidence in the document to support the answer.

In contrast, in the full development and test split of the data, the answer string is guaranteed to occur, but not necessarily with the evidence needed to answer the question.

Data preprocessing: All documents are tokenized using the NLTK 4 tokenizer.

Each document is truncated to contain at most 6000 words and at most 1000 sentences (average the number of sentences per document in Wikipedia is about 240).

Sentences are truncated to a maximum length of 50 (avg sentence length in Wikipedia is about 22).

Spans only up to length l = 5 are considered and cross-sentence spans discarded -this results in an oracle exact match accuracy of 95% on the Wikipedia development data.

To be consistent with the evaluation setup of BID14 , for the Wikipedia domain we create a training instance for every question (with all its associated documents), while on the web domain we create a training instance for every question-document pair.

Hyperparameters: We use GloVe embeddings BID20 of dimension 300 (trained on a corpus of 840 billion words) that are not updated during training.

Each embedding vector is normalized to have 2 norm of 1.

Out-of-vocabulary words are hashed to one of 1000 random embeddings, each initialized with a mean of 0 and a variance of 1.

Dropout regularization BID23 ) is applied to all ReLU layers (but not for the linear layers).

We additionally tuned the following hyperparameters using grid search and indicate the optimal values in parantheses: network size (2-layers, each with 300 neurons), dropout ratio (0.1), learning rate (0.05), context size (1), and loss weights (?? 1 = ?? 2 = 0.35, ?? 3 = 0.2, ?? 4 = 0.1).

We use Adagrad BID6 for optimization (default initial accumulator value set to 0.1, batch size set to 1).

Each hyperparameter setting took 2-3 days to train on a single NVIDIA P100 GPU.

The model was implemented in Tensorflow BID0 .

Table 1 presents our results on both the full test set as well as the verified subsets, using the exact match (EM) and F 1 metrics.

Our approach achieves state-of-the-art performance on both the Wikipedia and web domains outperforming considerably more complex models 5 .

In the web domain, except for the verified F 1 scores, we see a similar trend.

Surprisingly, we outperform approaches which use multi-layer recurrent pointer networks with specialized memories BID4 BID13 6 .

DISPLAYFORM0

Wikipedia Dev (EM) Table 2 : Model ablations on the full Wikipedia development set.

For row labeled **, explanation provided in Section ??4.3.

Table 2 shows some ablations that give more insight into the different contributions of our model components.

Our final approach (3-Level Cascade, Multiloss) achieves the best performance.

Training with only a single loss in level 3 (3-Level Cascade, Single Loss) leads to a considerable decrease in performance, signifying the effect of using a multi-loss architecture.

It is unclear if combining the two submodels in level 1 into a single feed-forward network that is a function of the question, span and short context (3-Level Cascade, Combined Level 1) is significantly better than our final model.

Although we found it could obtain high results, it was less consistent across different runs and gave lower scores on average (49.30) compared to our approach averaged over 4 runs (51.03).

Finally, the last three rows show the results of using only smaller components of our model instead of the entire architecture.

In particular, our model without the aggregation submodel (Level 1 + Level 2 Only) performs considerably worse, showing the value of aggregating multiple mentions of the same span across the document.

As expected, the level 1 only models are the weakest, showing that attending to the document is a powerful method for reading comprehension.

FIG2 shows the behavior of the k-best predictions of these smaller components.

While the difference between the level 1 models becomes more enhanced when considering the top-k candidates, the difference between the model without the aggregation submodel (Level 1 + Level 2 Only) and our full model is no longer significant, indicating that the former might not be able to distinguish between the best answer candidates as well as the latter.

The effect of truncation on Wikipedia in FIG2 (right) indicates that more scalable approaches that can take advantage of longer documents have the potential to perform better on the TriviaQA task.

Multiple Mentions: TriviaQA answers and their aliases typically reoccur in the document (15 times per document on an average).

To verify whether our model is able to predict answers which occur frequently in the document, we look at the frequencies of the predicted answer spans in FIG3 (left).

This distribution follows the distribution of the gold answers very closely, showing that our model learns the frequency of occurrence of answer spans in the document.

Speed: To demonstrate the computational advantages of our approach we implemented a simple 50-state bidirectional LSTM baseline (without any attention) that runs through the document and predicts the start/end positions separately.

FIG3 shows the speedup ratio of our approach compared to this LSTM baseline as the length of the document is increased (both approaches use a P100 GPU).

For a length of 200 tokens, our approach is about 8?? faster, but for a maximum length of 10,000 tokens our approach is about 45?? faster, showing that our method can more easily take advantage of GPU parallelization.

We observe the following phenomena in the results (see Table 3 ) which provide insight into the benefits of our architecture, and some directions of future work.

Lower levels get the prediction right, but not the upper levels.

Model predicts entities from the question.

Table 3 : Example predictions from different levels of our model.

Evidence context and aggregation are helpful for model performance.

The model confuses between entities of the same type, particularly in the lower levels.

Aggregation helps As motivated in Fig 1, we observe that aggregating mention representations across the evidence text helps (row 3).

Lower levels may contain, among the top candidates, multiple mentions of the correct answer (row 4).

However, since they cannot aggregate these mentions, they tend to perform worse.

Moreover, level 3 does not just select the most frequent candidate, it selects the correct one (row 2).Context helps Models which take into account the context surrounding the span do better (rows 1-4) than the level 1 (question + span) submodel, which considers answer spans completely out of context.

Entity-type confusion Our model still struggles to resolve confusion between different entities of the same type (row 4).

Context helps mitigate this confusion in many cases (rows 1-2).

How-ever, sometimes the lower levels get the answer right, while the upper levels do not (row 5) -this illustrates the value of using a multi-loss architecture with a combination of models.

Our model still struggles with deciphering the entities present in the question (row 5), despite the question-in-span feature.

We presented a 3-level cascaded model for TriviaQA reading comprehension.

Our approach, through the use of feed-forward networks and bag-of-embeddings representations, can handle longer evidence documents and aggregated information from multiple occurrences of answer spans throughout the document.

We achieved state-of-the-art performance on both Wikipedia and web domains, outperforming several complex recurrent architectures.

@highlight

We propose neural cascades, a simple and trivially parallelizable approach to reading comprehension, consisting only of feed-forward nets and attention that achieves state-of-the-art performance on the TriviaQA dataset.