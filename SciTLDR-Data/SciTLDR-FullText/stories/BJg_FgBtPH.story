Open-domain dialogue generation has gained increasing attention in Natural Language Processing.

Comparing these methods requires a holistic means of dialogue evaluation.

Human ratings are deemed as the gold standard.

As human evaluation is inefficient and costly, an automated substitute is desirable.

In this paper, we propose holistic evaluation metrics which capture both the quality and diversity of dialogues.

Our metrics consists of (1) GPT-2 based context coherence between sentences in a dialogue, (2) GPT-2 based fluency in phrasing, and, (3) $n$-gram based diversity in responses to augmented queries.

The empirical validity of our metrics is demonstrated by strong correlation with human judgments.

We provide the associated code, datasets and human ratings.

Learning to communicate is a key capacity of intelligent agents.

Research on enabling a machine to have meaningful and natural conversation with humans plays a fundamental role in developing artificial general intelligence, as can be seen in the formulation of Turing test (Turing, 1950) .

Recently open-domain or non-task-oriented dialogue systems have attracted a surge of research interest (Bessho et al., 2012; Sordoni et al., 2015; Shang et al., 2015; Vinyals & Le, 2015; Ghazvininejad et al., 2018; .

Moreover, dialogue generation has a wide range of industrial applications such as Microsoft's Xiaoice and Baidu's Dumi.

Evaluating models of dialogue generation in an efficient manner poses a significant challenge in developing dialogue systems.

The prevalent method of dialogue evaluation is human-based rating under a given rubric.

This method of evaluation is deemed impracticable, when various variations in the model and sets of hyperparameters are needed.

These drawbacks may hinder the research progress and render the human evaluation approach not scalable.

Previous automatic evaluation metrics generally focus on the quality of the dialogue generation (Tao et al., 2018; Ghazarian et al., 2019) .

In this work, we propose holistic metrics which considers both the quality and diversity of generated dialogues.

Specifically, we consider (1) context coherence of a dialogue (i.e., the meaningfulness of a response within the prior context of the dialogue), (2) language fluency of generated responses (i.e., the quality of phrasing relative to a human native speaker), and, (3) response diversity of a set of generated responses (i.e., the variety in meaning and word choice of responses).

A strong language model such as GPT-2 (Radford et al., 2019) naturally captures (1) and (2).

Therefore, we propose to recruit and fine-tune GPT-2 as a measure of quality.

Moreover, we utilize n-gram based entropy to capture (3).

Specifically, we propose to measure response diversity under augmented queries with controlled diversity.

Two such augmentation strategies are considered.

Finally, extensive human evaluations are conducted to substantiate the validity of our proposed metrics.

Evaluation metrics based on heuristics have been shown to align well with human judgments and widely applied in various language generation tasks.

For machine translation, BLUE (Papineni et al., 2002) computes n-gram precision, whereas METEOR (Banerjee & Lavie, 2005) takes into account both precision and recall.

For summarization, ROUGE (Lin, 2004 ) also considers both precision and recall by calculating F-measure.

These n-gram based metrics are well-suited for the generation tasks that are more source-determined or low conditional entropy such as translation, image captioning, and summarization.

Some dialogue studies adopted these metrics to evaluate the quality of generated conversation responses (Ritter et al., 2011; Su et al., 2018; Sordoni et al., 2015) .

They nevertheless are not suitable for open-ended generations or high conditional entropy task like dialogue generation where a diverse range of generations are acceptable conditional on a query.

Indeed, Liu et al. (2016) conduct extensive empirical studies on these metrics (e.g., BLEU, METEOR, and ROUGE) to test their effectiveness on evaluating dialogue generation and find limited relation between these automatic metrics and human judgments.

Table 1 : An example of low BLEU score and low semantic similarity between model response and reference response while the generated response appears reasonable within the dialogue.

The word-overlap metrics (e.g., BLUE) fail to capture the semantic similarity between model and reference responses.

The following works leverage the distributed representation learned in neural network models to capture semantic similarity among context, model response, and reference response.

collect a dataset of human scores and train a hierarchical recurrent neural network (RNN) to predict human-like scores to input responses given the context, resulting in an automatic metric that has a medium level correlation with human judgments.

Obtaining this metric however requires a large dataset of human-annotated scores, thus rendering this approach less flexible and extensible.

Tao et al. (2018) proposes a referenced metric and unreferenced metric blended evaluation routine (RUBER) for open-domain dialogue systems.

This blended metric is a combination of two metrics.

A referenced metric measures the similarity between model-generated and reference responses using word-embeddings.

An unreferenced metric captures the relevance between the query and response.

It is obtained by training a neural network classifier to determine whether a response is appropriate.

The positive examples are the references, while the negative examples are the reference responses randomly chosen from the dataset, hence avoiding the need of human-annotated data.

After training, the softmax score is utlized to measure whether the generated response is coherent with the query.

Attempting to improve RUBER, Ghazarian et al. (2019) explores to use contextualized embeddings from BERT.

The BERT-based unreferenced metric improves over the word-embedding-based RUBER unreferenced metric.

Interestingly, they show that the combined metric has a reduced correlation with human judgments than the unreferenced metric alone.

Although this finding is counterintuitive, it is consistent with the characteristics of opendomain dialogue that a range of diverse responses are reasonable given a query.

Hence a response can be acceptable even if it does not align well with the reference either in terms of word-overlap or semantic embedding.

See Table 1 for an example.

Prior art on automatic metrics focuses on the quality, mostly the relevance to the query, of the generated responses.

A good evaluation metric should not only measure the quality of generation, but also the diversity of generation, which is especially important for open-ended tasks like dialogue or story generation (Hashimoto et al., 2019) .

The current work proposes metrics to holistically evaluate the quality and diversity of open-domain dialogue generation.

One key component of dialogue response generation is its coherence to the query as explored in Tao et al. (2018) and Ghazvininejad et al. (2018) .

Prior work measures the coherence based on the Softmax score of a trained binary classifier.

Here we explore an alternative approach based on language modeling (Bengio et al., 2003) .

A language model can naturally capture the coherence of the response to the query without resorting to an ad-hoc classifier.

In particular, the query coherence metric is computed as the conditional probability of the response given the query, which reflects whether the response appropriately follows the query under a language model.

We adopt a transfer learning approach to obtain a powerful language model.

Besides coherence, a good response should be fluent.

Fluency is often measured by a language model (Holtzman et al., 2018; Xu et al., 2018) .

We define the response fluency score as negative perplexity of generated responses.

While the aforementioned metrics attempt to measure the quality of text generation, some n-gram based metric has also been utilized to measure diversity.

Mou et al. (2016) and compute unigram entropy across all generated utterances to measure the diversity.

This metric might be an improper metric for diversity since the generated utterances given various queries are generally diverse.

In our experiments, we observe constantly high diversity in terms of human ratings and n-gram based entropy.

Instead we approach diversity evaluation of a dialogue model with controlled queries, whereby we control the diversity of the queries while evaluating the diversity of the responses.

Controlling query diversity involves minimizing diversity in both meaning and word use and avoiding feeding the dialogue models identical inputs.

A dialogue model with poor diversity always generates responses with the same phrases and words, whereas an ideal model produces varying words and sentence structures.

The controlled queries are generated by augmenting the original query with sentences close in meaning and slightly different in word use.

For the purpose of generality, we propose WordNet substitution and Conditional Text Generator to generate controlled queries.

The n-gram entropy across the responses given the controlled queries is deemed as a diversity measure.

In this work, we propose a metric to holistically evaluate open-dialogue models by taking into consideration both quality and diversity of generated dialogues.

Our contributions are summarized below.

??? Both context coherence and response fluency (quality metrics) are naturally captured by metrics based on a strong language model.

Empirically, we demonstrate that the language model based metrics clearly outperform previous relevant metrics.

??? In view of the complexity of diversity evaluation, we propose two effective approaches to generate augmented utterances with controlled diversity: word substitution and text generator with k-best decoder.

Our experiments show that the diversity metric strongly correlates with human judgments on the response diversity.

Moreover, our proposed datasets significantly improve the agreement between human evaluation, leading to a more accurate and straightforward human annotation.

??? We release the datasets, human ratings and implementation of the metric as open-source contribution to pave the way towards further research.

2.1 CONTEXT COHERENCE Language models, which predict the next token given previous tokens, naturally capture the coherence between sentences and particularly the dialogue query and response in our case.

GPT-2 (Radford et al., 2019) is a large-scale pre-trained language model based on the transformer architecture (Vaswani et al., 2017) .

It is trained on a vast amount of diverse data and demonstrates impressive text generation capabilities.

In order to better capture the dependence between the queries and responses, GPT-2 can be fine-tuned on the dialogue dataset of interest.

Suppose a query q contains tokens {q t : t = 1, ..., T q } and a response r has tokens {r t : t = 1, ..., T r }.

Let P denote the fine-tuned GPT-2, then the context coherence is defined as the loglikelihood of the response conditional on the the query normalized by the length of the response length:

log P (r t |r <t , q).

(1)

Note that c raw (r|q) is some negative number and unbounded from below.

A single value is then hard to explain absolutely and can only be intepreted relative to other values.

Also, the unboundedness renders it prone to extreme values.

Hence, a normalized score is proposed instead.

Since the score distribution varies as a function of the dataset, the lower bound is defined as 5th percentile, denoted as c 5th , instead of some arbitrary value.

Then the normalized score, c(r|q), is

which ranges from 0 to 1.

To capture the fluency of responses, we also adopt the pretrained language model, GPT-2.

In particular, the raw response fluency score, f raw (r), is defined as,

Due to the negativeness and unboundedness of the raw score, a normalized version, f (r), similar to the normalized context coherence score is proposed,

We measure response diversity utilizing augmented queries with controlled diversity.

Controlling query diversity involves minimizing diversity in both meaning and word use and avoiding feeding the dialogue models identical inputs.

We thus aim to augment the original query with sentences close in meaning and slightly different in word use.

To achieve so, two augmentation approaches are proposed: (1) WordNet Substitution (WS) and (2) Conditional Text Generator (CTG).

WordNet Substitution (WS) is word-level manipulation method suitable for both single-turn and multi-turn datasets.

It is achieved by first using Part-Of-Speech (POS) tagger to tag tokens in a query.

Then four augmented inputs are generated by substituting verbs, nouns, adjectives & adverbs, or all of the above with synonyms in WordNet.

Different from WS, Conditional Text Generator (CTG) is an approach to testing language diversity using multi-turn datasets.

It requires a sequence-to-sequence or a transformer model to produce augments conditioned on the context, which is defined as the prior utterance history to the selected query.

For instance, suppose {u 1 , ..., u t???1 } denotes the utterance history and u t indicates the query to be augmented, then the top-5 beams, u

t , ..., u

t , from the CTG model with the concatenated utterance history [u 1 ; ...; u t???1 ] is input into a model to be evaluated.

Given a set of augmented queries for the ith query with controlled diversity, the responses, R i , are generated by the model under test.

Then n-gram entropy for the ith sample is computed as,

where p is the n-gram probability in R i .

The diversity metric is then defined as the averaged entropy over the dataset,

3 EXPERIMENTS

To facilitate comparison with prior work (Ghazarian et al., 2019) , the DailyDialog dataset (Li et al., 2017 ) is adopted for the empricial analysis of our proposed metrics.

This dataset contains 13,118 high-quality multi-turn dialogue dataset.

The dialogue is split into query-response pairs with a 42,000 / 3,700 / 3,900 train-test-validation split.

A sequence-to-sequence (seq2seq) with attention (Bahdanau et al., 2014) was trained with the train and validation partitions to generate dialogue responses.

The implementation in OpenNMT (Klein et al., 2017) was used to train the model.

The seq2seq consists of a 2-layer LSTM with 500 hidden units on both the encoder and decoder.

The model was trained with SGD and learning rate of 1.

To obtain responses on a wide spectrum of quality and diversity, we sample the data with top-k sampling where k = {1, 50, 500}.

The base GPT-2 model with 12 layers was used to compute our metrics.

We also experimented with the medium GPT-2 with 24 layers and found that the results were generally the same.

And larger models (the 36-and 48-layers GPT-2) might pose computational difficulty for some researchers and thus were not considered.

The GPT-2 model was fine-tuned on the training and validation data.

In fine-tuning, the query and response were concatenated together as a single sentence to feed into GPT-2.

The perplexity of the fine-tuned language model on the test dataset was 16.5.

WordNet substitution and conditional text generator were used to augment diversity-controlled queries.

The Stanford POS tagger (Toutanova & Manning, 2000) and the WordNet by Miller (1998) were utilized to do WordNet substitution.

As for conditional text generator, we trained an Open-NMT Transformer on the training and validation splits for query augmentation, which was applied to the testing dataset to augment the query with the top-4 beams.

To assess validity of our proposed metrics, we utilize Amazon Turk to collect high quality human ratings from 10 subjects.

For each metric, we select a set of generated query-response pairs (or responses only) to be presented to humans and each datapoint is to be rated from 1 to 5, with 1 being the worst and 5 being the best in generation quality corresponding to that metric.

On both Context Coherence and Fluency metrics, we select 200 datapoints with diverse range of generation quality.

There are 200 query-response pairs to be rated for Context Coherence and 200 responses to be rated for Fluency.

For Diversity metric, we select 100 datapoints, totaling 500 responses, to be rated in groups of 5 all of which are conditioned on the controlled inputs generated by a CTG given the same context.

After Amazon Turk results are collected, we then compute Pearson Correlation between our evaluation metrics and human ratings to assess the validity of our metric and selected datasets.

We normalize the human rating scores from 0 to 1.

Query Generated Reply Human Score RUBER Ours Of course.

A two-week paid vacation a year, a five-day workweek.

So, if I get a margin card, I could take a margin card for you to travel to a company as soon as possible.

0.20 0.97 0.19 Table 2 : Case study.

Both our coherence metric and the human evaluation agreed that the generated response is not coherent with the given query, while RUBER indicated this reply is coherent.

4.1 CONTEXT COHERENCE Table 3 demonstrates the Pearson and Spearman correlations between the proposed context coherence metric and human judgments.

Also, the results were compared to the previous best-performing automatic metric, RUBER with BERT embeddings (Ghazvininejad et al., 2018) .

Clearly both our language model based coherence metric show higher correlation with human judgments than the classifier-based metric, RUBER.

In addition, we compared the proposed metric with a similar metric based on a GPT-2 language model without fine-tuning on the target dataset.

The fine-tuned version improved the results, indicating that fine-tuning on the dialogue dataset enables the language model better capture the dependency between the queries and replies.

Interestingly, even the metric based on the language model without fine-tuning correlated with human ratings stronger than RUBER.

We also examined the inter-rater reliability.

It is computed by holding out the ratings of one rater at a time, calculating its correlation with the average of other rater's judgments, and finally averaging across and taking the maximum all held-out correlation scores.

The inter-rater reliability results also support the strong performance our proposed context coherence metric since the correlation between the automatic metric and human evaluation was close to the inter-rater correlations.

Table 2 displays a case study.

Both our coherence metric and the human evaluation agreed that the generated response is not coherent with the given query, while RUBER indicated this reply is coherent.

This might be because RUBER simply compares the embeddings of the query and response and business travel related words in the query such as vacation, workweek and in the reply such as travel, company make RUBER judge that they are similar.

Table 3 : Correlation between RUBER+BERT and context coherence metric c(r|q) with human ratings (without and with fine-tuning of GPT-2).

Our findings show that the proposed fluency metric f (r) is highly correlated with human judgments.

Table 4 summarizes the relation between our proposed fluency metric and the human-ratings in terms of Pearson and Spearman correlation.

The importance of fine-tuning GPT-2 (as outlined in Section 3.3) is evident.

We observe an increase from 0.43 to 0.82 in Pearson correlation.

In addition, Figure 2 details the effect of fine-tuning.

Notably, a correction of outliers occurs.

Moreover, the consistency of human ratings is demonstrated by high mean pair-wise correlations between pairs of ratings.

Table 4 : Correlation between fluency metric f (r) and human ratings without and with fine-tuning of GPT-2.

Pairwise mean and max correlations of human ratings.

Table 5 shows the evaluation of our generated datasets using WS and CTG.

Unigram, bigram, and trigram entropy are used to calculate responses' diversity and are compared to human ratings in Pearson and Spearman Correlation.

Note that automatic evaluations on our datasets consistently achieve higher correlation compared to the baseline dataset.

We also show our datasets evaluated using three different diversity metrics in Figure 3 .

The figures show correlations between normalized human ratings and corresponding n-gram entropy.

A line of best-fit is drawn to indicate their correlations, and for plotting purpose, each datapoint after normalization is added a random noise sampled from N (0, 0.05 2 ).

Clearly, WS and CTG Dataset show more clustered datapoints and slopes closer to 1 than our baseline dataset, a result consistent with the reported correlations.

Table 6 shows inter-rater Pearson Correlation, Spearman correlations, and variance in human ratings.

Interestingly, both WS Dataset and CTG Dataset display similarly high correlations, indicating that raters generally agree with each other.

WS Dataset is also lowest in Human Variance, suggesting human raters are more certain about their ratings.

Baseline Dataset, on the other hand, has poor inter-rater correlations.

This is most likely due to the uncontrolled nature of input sentences such that outputs of evaluated models are generally diverse, making it difficult for humans to judge diversity performance of the model.

Furthermore, both of our datasets achieve scores close to that of their corresponding mean inter-rater correlations, indicating that the evaluation metric on our datasets can reveal diversity of a dialog system consistent with humans.

This paper provides a holistic and automatic evaluation method of open-domain dialogue models.

In contrast to prior art, our means of evaluation captures not only the quality of generation, but also the diversity of responses.

We recruit GPT-2 as a strong language model to evaluate the fluency and context-coherency of a dialogue.

For diversity evaluation, the diversity of queries is controlled while the diversity of responses is evaluated by n-gram entropy.

Two methods for controlled diversity are proposed, WordNet Substitution and Conditional Text Generator.

The proposed metrics show strong correlation with human judgments.

We are providing the implementations of our proposed metrics, associated fine-tuned models and datasets to accelerate the research on open-domain dialogue systems.

It is our hope the proposed holistic metrics may pave the way towards comparability of open-domain dialogue methods.

@highlight

We propose automatic metrics to holistically evaluate open-dialogue generation and they strongly correlate with human evaluation.