We show that generating English Wikipedia articles can be approached as a multi- document summarization of source documents.

We use extractive summarization to coarsely identify salient information and a neural abstractive model to generate the article.

For the abstractive model, we introduce a decoder-only architecture that can scalably attend to very long sequences, much longer than typical encoder- decoder architectures used in sequence transduction.

We show that this model can generate fluent, coherent multi-sentence paragraphs and even whole Wikipedia articles.

When given reference documents, we show it can extract relevant factual information as reflected in perplexity, ROUGE scores and human evaluations.

The sequence-to-sequence framework has demonstrated success in natural-language sequence transduction tasks such as machine translation.

More recently, neural techniques have been applied to do single-document, abstractive (paraphrasing) text summarization of news articles BID15 , BID9 ).

In this prior work, the input to supervised models ranged from the first sentence to the entire text of an article, and they are trained end-to-end to predict reference summaries.

Doing this end-to-end requires a significant number of parallel article-summary pairs since language understanding is a pre-requisite to generate fluent summaries.

In contrast, we consider the task of multi-document summarization, where the input is a collection of related documents from which a summary is distilled.

Prior work has focused on extractive summarization, which select sentences or phrases from the input to form the summaries, rather than generating new text.

There has been limited application of abstractive neural methods and one possible reason is the paucity of large, labeled datasets.

In this work, we consider English Wikipedia as a supervised machine learning task for multidocument summarization where the input is comprised of a Wikipedia topic (title of article) and a collection of non-Wikipedia reference documents, and the target is the Wikipedia article text.

We describe the first attempt to abstractively generate the first section, or lead, of Wikipedia articles conditioned on reference text.

In addition to running strong baseline models on the task, we modify the Transformer architecture BID18 to only consist of a decoder, which performs better in the case of longer input sequences compared to recurrent neural network (RNN) and Transformer encoder-decoder models.

Finally we show our modeling improvements allow us to generate entire Wikipedia articles.

Neural abstractive summarization was pioneered in BID15 , where they train headline generation models using the English Gigaword corpus BID3 , consisting of news articles from number of publishers.

However, the task is more akin to sentence paraphrasing than summarization as only the first sentence of an article is used to predict the headline, another sentence.

RNN-based encoder-decoder models with attention (seq2seq) perform very well on this task in both ROUGE BID7 , an automatic metric often used in summarization, and human evaluation BID1 .In BID9 , an abstractive summarization dataset is proposed by modifying a questionanswering dataset of news articles paired with story highlights from Daily Mail and CNN.

This task is more difficult than headline-generation because the information used in the highlights may come from many parts of the article and not only the first sentence.

One downside of the dataset is that it has an order-of-magnitude fewer parallel examples (310k vs. 3.8M) to learn from.

Standard seq2seq models with attention do less well, and a number of techniques are used to augment performance.

Another downside is that it is unclear what the guidelines are for creating story highlights and it is obvious that there are significant stylistic differences between the two news publishers.

In our work we also train neural abstractive models, but in the multi-document regime with Wikipedia.

As can be seen in TAB0 , the input and output text are generally much larger, with significant variance depending on the article.

The summaries (Wikipedia lead) are multiple sentences and sometimes multiple paragraphs, written in a fairly uniform style as encouraged by the Wikipedia Manual of Style 1 .

However, the input documents may consist of documents of arbitrary style originating from arbitrary sources.

We also show in TAB0 the ROUGE-1 recall scores of the output given the input, which is the proportion of unigrams/words in the output co-occuring in the input.

A higher score corresponds to a dataset more amenable to extractive summarization.

In particular, if the output is completely embedded somewhere in the input (e.g. a wiki-clone), the score would be 100.

Given a score of only 59.2 compared to 76.1 and 78.7 for other summarization datasets shows that ours is the least amenable to purely extractive methods.

There is a rich body of work incorporating Wikipedia for machine learning tasks, including questionanswering BID4 , BID13 ) and information extraction BID6 , and text generation from structured data BID5 .The closest work to ours involving generating Wikipedia is BID16 , where articles are generated extractively (instead of abstractively in our case) from reference documents using learned templates.

The Wikipedia articles are restricted to two categories, whereas we use all article types.

The reference documents are obtained from a search engine, with the Wikipedia topic used as query similar to our search engine references.

However we also show results with documents only found in the References section of the Wikipedia articles.

Previous work on neural abstractive summarization relies on RNNs as fundamental modules, mirroring techniques successful in machine translation (MT).

Recently, state-of-the-art MT results were obtained using a non-recurrent architecture, called the Transformer BID18 .

The lack of recurrence enables greater within-training-example parallelization, at the cost of quadratic complexity in the input sequence length.

We find the Transformer transfers well to medium length, input sequence summarization and describe modifications to better handle longer sequences.

results from the Google search engine, using the article section titles as queries.

For each query, we collect 10 result pages.

From this collection we remove the Wikipedia article itself, which is often among the top results.

We also remove "clones", which are detected when there is a high-level of unigram overlap with the article (details provided in A.2.1).

We denote these refined search results for an article, a i , as S i ⊂ D. Similar to C i , we extract only the text to use as input.

TAB1 describes overall properties of our WikiSum dataset.

Many articles have few citations, motivating our supplementation of the source documents with web search results.

On the other hand, citations when available, tend to be of higher-quality.

When counting the total words in the entire dataset, it is orders-of-magnitude larger than previous summarization datasets.

To have consistent train/development/test data across corpus-comparison experiments, we restrict the articles to those with at least one crawlable citation.

We divide the articles roughly into 80/10/10 for train/development/test subsets, resulting in 1865750, 233252, and 232998 examples respectively.

Because the amount of text in input reference documents (C i , S i ) can be very large (see TAB1 ) it is infeasible to train an end-to-end abstractive model given the memory constraints of current hardware.

Hence, we first coarsely select a subset of the input using extractive summarization.

The second stage involves training an abstractive model that generates the Wikipedia text while conditioning on this extraction.

This two-stage process is inspired by by how humans might summarize multiple long documents: First highlight pertinent information, then conditionally generate the summary based on the highlights.

We investigate three extractive methods from the summarization literature, along with a trivial and cheating method, to assess the importance of this stage.

For each article, a i we create a ranked list of paragraphs, {p DISPLAYFORM0 is the rank of the jth paragraph p i j of (C i , S i ).

From this we select the first L tokens as input to the second abstractive stage.1.

Identity: As a trivial baseline extractor, we simply use the first L tokens of the input.2.

tf-idf : A non-trivial ranking is to consider ranking paragraphs as documents in a queryretrieval problem, where the query is the title of the article, T (a i ).

We compute tf-idf BID14 for the query, with respect to the documents, {p i j }.

That is, we summate for each word in the query DISPLAYFORM1 where N w , N d , and N dw are the count of the word in the document, total number of documents, and total number of documents containing the word, respectively.3.

TextRank BID8 : A weighted graph is defined where text units are nodes and edges are defined by a similarity measure based on word overlap.

An algorithm similar to PageRank BID11 is then used to compute the ranking of text units.

We used paragraphs for the text units.4.

SumBasic BID10 : Word frequencies in the input text are used to assign scores to words, which are in turn used to score sentences.

After selecting the best scoring sentence, words in it have their scores reduced, and the process is repeated until the desired summary length is reached.

To further demonstrate the quality of extraction on the final performance, we implement a cheating extractor that ranks {p i j } using recall of bigrams in the ground truth text:

DISPLAYFORM0 4.2 ABSTRACTIVE STAGE 4.2.1 DATA REPRESENTATION Given the ordered paragraphs {p i Ri(j) }, we derive the raw text input simply as the concatenation of the paragraphs in order, the most relevant at the beginning, and prefixed with the title.

We then encode the text using sub-word tokenization similar to BID19 with a vocabulary size of 32,000 yielding tokenized input, x i : DISPLAYFORM1 For various values of L in experiments, we truncate the tokens to form the input sequence: DISPLAYFORM2 For the output, we use the same vocabulary and tokenization for the Wikipedia lead text but do not do any truncation across experiments.

Next we describe the abstractive models, W , that learn to write articles, a i = W (m L i ), which we treat as a sequence transduction problem from very long input sequences (up to L = 11000) to medium output sequences (typically less than 500).

As a baseline we apply the standard LSTM encoder-decoder with attention (seq2seq-att) as in BID0 to this task.

As is typical we train to optimize the maximum-likelihood objective: DISPLAYFORM0 A stronger, more recent baseline that we use is the non-recurrent Transformer model described in 2.3, which also has symmetric encoder and decoder modules (T-ED).

We introduce a simple but effective modification to T-ED for long sequences that drops the encoder module (almost reducing model parameters by half for a given hyper-parameter set), combines the input and output sequences into a single "sentence" and is trained as a standard language model.

, where δ is a special separator token and train a model to predict the next word given the previous ones: DISPLAYFORM0 Since the model is forced to predict the next token in the input, m, as well as y, error signals are propagated from both input and output time-steps during training.

We also suspect that for monolingual text-to-text tasks redundant information is re-learned about language in the encoder and decoder.

We believe this allows for easier optimization and empirically observe this with longer sequences (see Section 5.3).

Note that because of the self-attention of the Transformer, when generating the next token, attention from both m and y are considered.

At inference we provide the input sequence, m i , initially, and auto-regressively generate the output, y i , as normal.

To re-use the terminology used to describe the Transformer, the attention is a function of a query (Q) and set of key (K) and value (V ) pairs.

To handle longer sequences, we modify the multi-head self-attention of the Transformer to reduce memory usage by limiting the dot products between Q and K in: DISPLAYFORM0 Local attention: Sequence tokens are divided into blocks of similar length and attention is performed in each block independently.

As the attention memory cost per block becomes constant, this modification allow us to keep the number of activations linear with respect to the sequence length.

In our experiments, we choose to have blocks of 256 tokens.

Memory-compressed attention: After projecting the tokens into the query, key, and value embeddings, we reduce the number of keys and values by using a strided convolution.

The number of queries remains unchanged.

This modification allows us to divide the number of activations by a compression factor.

In our experiments we use convolution kernels of size 3 with stride 3.

In contrast to local attention layers, which only capture the local information within a block, the memorycompressed attention layers are able to exchange information globally on the entire sequence.

These modifications (see FIG0 ) allow us in practice to process sequences 3x in length over the T-D model.

For both local and memory-compressed attention, masking is added to prevent the queries from attending to future keys and values.

Our final architecture is a 5-layer network (LMLML) alternating between local-attention (L) layers and memory-compressed attention (M) layers (in BID18 it is 6 identical layers).

We also added in some experiments one mixture of experts (MoE) layer to increase the network's capacity.

In experiments we evaluate based on perplexity (per-wordpiece), a common language modeling metric, and ROUGE-L F1 (version ROUGE-1.5.5), a common metric used in comparing candidate and reference summaries.

Note the F1 flavor of ROUGE is more appropriate in this setting as we do not explicitly constrain the output length in abstractive models; it is the harmonic mean of ROUGERecall (which favors long summaries) and ROUGE-Precision (which favors short summaries).Although optimizing ROUGE directly has been shown to not always yield the best summaries as evaluated by human judgment BID12 , we found that for our task optimizing for perplexity correlates with increased ROUGE and human judgment.

We suspect that the relatively uniform style of Wikipedia articles makes ROUGE more appropriate here than in general abstractive summarization tasks.

For all abstractive model training, we use the open-source tensor2tensor 2 library.

The seq2seq baseline had a hidden size of 128 with 2 layers (we use the hyper-parameter set defined in the library as lstm attention).For the Transformer encoder-decoder (T-ED), we use the hyper-parameter set transfomer base v1 and train for 1 million steps.

Models exhibited very little overfitting and did not require early-stopping.

The Transformer Decoder (T-D) was identical to the decoder part of T-ED.

The T-DMCA model is similar to T-D, but with the enhancements described in section 4.2.4.Unless otherwise stated, during decoding we use a beam search of size 4 and length penalty α = 0.6 BID19 and decode until an end-of-sequence token is reached.

Extractive-only is not enough: We investigate performance of extractive methods without the abstractive model by looking at the ROUGE-L F1 scores after running tf-idf, SumBasic, and TextRank in Figure 2 , without any abstractive model.

In the case of TextRank and SumBasic we matched the output length to the target length and observe the extractive methods perform roughly in-line with each other in terms of ROUGE-L F1.

Our best abstractive model more than doubled this metric.

Further, this model yields large improvements in perceived linguistic quality (elaborated below).Extractive method: From TAB2 we observe that smart extraction is critical for final abstractive performance.

There is a significant gap between doing nothing, identity, and extractive summarization, tf-idf.

Further, there is a significant gap between tf-idf and the cheating extractor, suggesting future work in improving the extraction step could result in significant improvements.

One possibility is to train a supervised model to predict relevance (Eq. 1), which we leave as future work.

For subsequent experiments we fix the extractive method to tf-idf.

Input Corpus: From table 3 we also observe that, unsurprisingly, the combined dataset performs best, but the gaps between it and using only one of citations or search results are both significant and their contributions are complementary.

In subsequent experiments, we report only the combined results.

Abstractive model architecture and input length: As we see from TAB3 , seq2seq-attention as a baseline does quite poorly on this task compared to the Transformer architectures.

As seen in FIG2 , we observe that the Transformer encoder-decoder, T-ED, architecture consistently improves in performance until a best of around L = 500 − 1000 and is unable to learn at L = 2000.

This motivated the Transformer-Decoder, which we found could learn and improve up to L = 4000, before running out of memory on our machines equipped with 16GB of GPU RAM (NVIDIA P100).

By using the T-DMCA modifications, we were able to train up to L = 11000 and continued to see improvements in performance.

We also found the MoE-layer helped performance by adding model capacity at high L, for example dropping log-perplexity from 2.05 to 1.93 at L = 11000 with 128 experts.

Our best model attempted uses 256 experts at L = 7500 (we were unable to use 256 experts with L = 11000 due to memory constraints) and achieves a perplexity of 1.90, Human Evaluation -Linguistic quality We conducted a DUC-style human evaluation of linguistic quality 3 of samples from a baseline abstractive (seq2seq), the best extractive (tf-idf ), and our best T-DMCA models.

Five different dimensions are assessed: grammaticality, non-redundancy, referential clarity, focus, and structure/coherence.

As seen in Table 5 , the T-DMCA model does statistically significantly better on all dimensions, except on non-redundancy where tf-idf does about as well.

Overall, we observed high fluency and coherence from our best abstractive model.

Occasionally we observed some repetition of phrases which hurt the non-redundancy and structure, but it was much rarer compared with the other abstractive method, seq2seq.

The biggest weakness of the extractive Table 5 : Linguistic quality human evaluation scores (scale 1-5, higher is better).

A score significantly different (according to the Welch Two Sample t-test, with p = 0.001) than the T-DMCA model is denoted by *.

Focus Grammar

Referential clarity

T-DMCA (best) 4.5 4.6 4.2 4.5 4.2 tf-idf -only 3.0* 3.6* 3.9 3.2* 2.7* seq2seq-attention 3.0* 3.4* 2.1* 3.4* 2.3* Table 6 : Side-by-side for two models pair with large automatic metric gaps DISPLAYFORM0 38.8 1.5 method compared with our best abstractive model was the lack of structure and coherence in the summaries.

Human Evaluation -side-by-side preference We validated our chosen metrics correlate with human preference by conducting two side-by-side human evaluation experiments, comparing models with large gaps in perplexity/ROUGE.

We observe in Table 6 that human judgment correlates with our automatic metrics, but it becomes more difficult to distinguish at the higher-end of model performance.

Details of the human evaluation experimental designs can be found in Appendix A.3.To summarize the quantitative results, we believe the highest impact future work will be from improving the extractive stage and extending the decoder-only architectures to learn from larger L while maintaining sufficient model capacity.

Comparison with BID16 : A direct comparison with BID16 is difficult for three reasons: (a) they report results only for two small subsets of Wikipedia, Diseases and American Actors; (b) we report on lead generation instead of full-articles; (c) we were unable to obtain the exact articles they used as input and output (in particular they make no claim of Wikiclone detection).

However, we make a best-effort comparison by finding the subset of articles of our test set that correspond to Diseases and American Actors, the two categories reported on by Sauper & Barzilay and reporting our ROUGE-1 scores TAB4 .

We observe that we perform better on American Actors than Diseases, probably because of the prevalence of the former (and biographies) in Wikipedia compared to the latter in our training set for our single, global model, whereas Sauper & Barzilay likely benefit from the category-specific templates.

On average our ROUGE-1 scores are higher but do worse on the less common and somewhat specific disease category.

In FIG3 , we show the predictions from three different models (using tf-idf extraction, and the combined corpus) along with the Wikipedia ground truth.

As the perplexity decreases we see improvements in the model outputs, in terms of fluency, factual accuracy, and narrative complexity.

In particular, the T-DMCA model offers a respectable alternative to the Wikipedia version and is more succinct, while mentioning key facts, such as where the law firm was located, when and how it was formed, and the rise and fall of the firm.

In manual inspection of model outputs, we noticed an unexpected side-effect: models learn to translate names from English into multiple languages, e.g. Rohit Viswanath into Hindi (see FIG4 ).

Although we did not do a systematic evaluation of the translations, we found they are often correct, and often they are not found in the Wikipedia article itself.

We also verified that in general the translation is not merely copied from the source, such as example cases where the target language is the incorrect one (e.g. translation of an English name into Ukrainian).

Given that we have shown it is possible to learn sequence transduction models on combined inputoutput sequence lengths of approximately 12000 using the T-D architecture, we show that it is possible to train a model to generate entire Wikipedia articles.

As a preliminary result, we trained two T-DMCA models: One is trained to use L = 6000 reference tokens to predict at most 2192 article tokens (longer examples are ignored) and another is conditioned only on the title and generates articles up to 4000 tokens long.

We show samples from both models in Appendix A.1.

Although the generated articles are not as good as the real Wikipedia or our lead section samples, the models can be seen to organize the article into plausible sections and exhibit global coherence over multi-paragraph text.

The model with access to reference documents inserts factual information in the generated article.

Although we did not focus or tune on the full-article task we see this as an interesting future work for abstractive summarization.

We have shown that generating Wikipedia can be approached as a multi-document summarization problem with a large, parallel dataset, and demonstrated a two-stage extractive-abstractive framework for carrying it out.

The coarse extraction method used in the first stage appears to have a significant effect on final performance, suggesting further research on improving it would be fruitful.

We introduce a new, decoder-only sequence transduction model for the abstractive stage, capable of handling very long input-output examples.

This model significantly outperforms traditional encoderdecoder architectures on long sequences, allowing us to condition on many reference documents and to generate coherent and informative Wikipedia articles.

To encourage further research on large-scale summarization, we will release the URLs used in our experiments (the Wikipedia URL as well as the URLs of its references).

We also provide code that that extracts content from the CommonCrawl dataset 4 , which is freely available for download.

We use the open-source tensor2tensor 5 library for training abstractive models and will be releasing our abstractive modeling code extensions.

Further details are available at https:// goo.gl/wSuuS9.

To assess linguistic quality, we randomly selected samples generated by models from the test set and ask raters to choose a score from 1 to 5 (higher is better) for five dimensions: Grammaticality, Non-redundancy, Referential clarity, Focus, and Structure and Coherence.

These were used in the past at DUC for evaluating summaries BID2 .

For each model we selected 25 examples and averaged the scores for each question across 3 raters (out of pool of 7).To compare two models by human evaluation, we randomly select examples from the test set and show model outputs side-by-side in the interface shown in Figure 9 .

Which side a model appears on is randomized per example and rater.

For the experiments in Table 6 we had 3 raters score 25 examples each and computed the ratio of ratings preferring one model over the other.

A.4 EXAMPLE ABSTRACTIVE MODEL INPUT Figure 9 : Screenshot of side-by-side human evaluation tool.

Raters are asked whether they prefer model output on the left or right, given a ground truth Wikipedia text.

Figure 10: Example extractive-output/abstractive-input for models in "dewey & lebeouf" example.

The extractive method used is tf-idf.

<|TLDR|>

@highlight

We generate Wikipedia articles abstractively conditioned on source document text.