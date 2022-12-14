Sequence-to-sequence (seq2seq) neural models have been actively investigated for abstractive summarization.

Nevertheless, existing neural abstractive systems frequently generate factually incorrect summaries and are vulnerable to adversarial information, suggesting a crucial lack of semantic understanding.

In this paper, we propose a novel semantic-aware neural abstractive summarization model that learns to generate high quality summaries through semantic interpretation over salient content.

A novel evaluation scheme with adversarial samples is introduced to measure how well a model identifies off-topic information, where our model yields significantly better performance than the popular pointer-generator summarizer.

Human evaluation also confirms that our system summaries are uniformly more informative and faithful as well as less redundant than the seq2seq model.

Automatic text summarization holds the promise of alleviating the information overload problem BID13 .

Considerable progress has been made over decades, but existing summarization systems are still largely extractive-important sentences or phrases are identified from the original text for inclusion in the output BID22 .

Extractive summaries thus unavoidably suffer from redundancy and incoherence, leading to the need for abstractive summarization methods.

Built on the success of sequence-to-sequence (seq2seq) learning models BID35 , there has been a growing interest in utilizing a neural framework for abstractive summarization BID28 BID20 BID41 BID36 BID5 .Although current state-of-the-art neural models naturally excel at generating grammatically correct sentences, the model structure and learning objectives have intrinsic difficulty in acquiring semantic interpretation of the input text, which is crucial for summarization.

Importantly, the lack of semantic understanding causes existing systems to produce unfaithful generations.

BID3 report that about 30% of the summaries generated from a seq2seq model contain fabricated or nonsensical information.

Furthermore, current neural summarization systems can be easily fooled by off-topic information.

For instance, FIG0 shows one example where irrelevant sentences are added into an article about "David Collenette's resignation".

Both the seq2seq attentional model BID20 and the popular pointer-generator model BID31 are particularly susceptible to unfaithful generation, partially because these models tend to rely on sentences at the beginning of the articles for summarization while being ignorant about their content.

Therefore, we design a novel adversarial evaluation metric to measure the robustness of each summarizer against small amounts of randomly inserted topic-irrelevant information.

The intuition is that if a summarization system truly understands the salient entities and events, it would ignore unrelated content.32nd Conference on Neural Information Processing Systems (NIPS 2018), Montr??al, Canada.

Article Snippet:

For years Joe DiMaggio was always introduced at Yankee Stadium as "baseball's greatest living player."

But with his memory joining those of Babe Ruth, Lou Gehrig, Mickey Mantle and Miller Huggins.

Canada's Minister of Defense resigned today, a day after an army official testified that top military officials had altered documents to cover up responsibility for the beating death of a Somali teen-ager at the hands of Canadian peacekeeping troops in 1993.

Defense minister David Collenette insisted that his resignation had nothing to do with the Somalia scandal.

Ted Williams was the first name to come to mind, and he's the greatest living hitter.

... Seq2seq:

George Vecsey sports of The Times column on New York State's naming of late baseball legend Joe DiMaggio as "baseball's greatest living player," but with his memory joining those of Babe Ruth, Lou Gehrig, Mickey Mantle and Miller dens.

Pointer-generator: Joe DiMaggio is first name to come to mind, and Ted Williams is first name to come to mind, and he's greatest living hitter; he will be replaced by human resources minister, Doug Young, and will keep his Parliament seat for governing Liberal Party.

Our Model: Former Canadian Defense Min David Collenette resigns day after army official testifies that top military officials altered documents to cover up responsibility for beating death of Somali teen-ager at hands of Canadian peacekeeping troops in 1993.

To address the above issues, we propose a novel semantic-aware abstractive summarization model, inspired by the human process of writing summaries-important events and entities are first identified, and then used for summary construction.

Concretely, taking an article as input, our model first generates a set of summary-worthy semantic structures consisting of predicates and corresponding arguments (as in semantic parsing), then constructs a fluent summary reflecting the semantic information.

Both tasks are learned under an encoder-decoder architecture with new learning objectives.

A dual attention mechanism for summary decoding is designed to consider information from both the input article and the generated predicate-argument structures.

We further present a novel decoder with a segment-based reranking strategy to produce diverse hypotheses and reduce redundancy under the guidance of generated semantic information.

Evaluation against adversarial samples shows that while performance by the seq2seq attentional model and the pointer-generator model is impacted severely by even a small addition of topic-irrelevant information to the input, our model is significantly more robust and consistently produces more on-topic summaries (i.e. higher ROUGE and METEOR scores for standard automatic evaluation).

Our model also achieves significantly better ROUGE and METEOR scores than both models on the benchmark dataset CNN/Daily Mail BID11 .

Specifically, our model's summaries use substantially fewer and shorter extractive fragments than the comparisons and have less redundancy, alleviating another common problem for the seq2seq framework.

Human evaluation demonstrates that our model generates more informative and faithful summaries than the seq2seq model.

To discourage the generation of fabricated content in neural abstractive models, a pointer-generator summarizer BID31 ) is proposed to directly reuse words from the input article via a copying mechanism BID9 .

However, as reported in their work BID31 and confirmed by our experiments, this model produces nearly extractive summaries.

While maintaining their model's rephrasing ability, here we improve the faithfulness and informativeness of neural summarization models by enforcing the generation of salient semantic structures via multi-task learning and a reranking-based decoder.

Our work is in part inspired by prior abstractive summarization work, where the summary generation process consists of a distinct content selection step (i.e., what to say) and a surface realization step (i.e., how to say it) BID40 BID26 .

Our model learns to generate salient semantic roles and a summary in a single end-to-end trained neural network.

Our proposed model also leverages the recent successes of multi-task learning (MTL) as applied to neural networks for a wide array of natural language processing tasks BID18 BID33 BID25 BID27 BID12 .

Most recent work BID23 leverages MTL to jointly improve performance on summarization and entailment generation.

Instead of treating the tasks equally, we employ semantic parsing for the sake of facilitating more informative and faithful summary generation.

Figure 2: Our semantic-aware summarization model with dual attention over both input and generated salient semantic roles.

In both shared and separate decoder models, the semantic roles are generated first, followed by the summary construction.

Best viewed in color.

In the standard seq2seq model, a sequence of input tokens x = {x 1 , ..., x n } is encoded by a recurrent neural network (RNN) with hidden states h i at every timestep i. The final hidden state of the encoder is treated as the initial state of the decoder, another RNN (h n = s 0 ).

The decoder will produce the output sequence y = {y 1 , ..., y T }, with hidden states s t at every timestep t. Our model outputs two sequences: the sequence of semantic role tokens y s = {y is calculated as the summation of the encoder hidden states, weighted by the attention distribution a t at every timestep t: DISPLAYFORM0 where v, W s , W h are learnable parameters.

The context vector, along with the decoder hidden state, is used to produce the vocabulary distribution via softmax: The loss is defined as the negative log likelihood of generating summary y over the training set D using model parameters ??: DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 The log probability for each training sample is the average log likelihood across decoder timesteps.

Encoder.

In our models, our encoder is a single-layer bidirectional long short-term memory (LSTM) unit BID7 , where the hidden state is a concatenation of the forwards and backwards LSTMs: DISPLAYFORM4 Decoders.

We propose two different decoder architectures-separate decoder and shared decoderto handle semantic information.

In the separate decoder model (See Figure 2c) , the semantic decoder and the summary decoder are each implemented as their own single-layer LSTM.

This setup, inspired by the one-to-many multi-task learning framework of BID18 , encourages each decoder to focus more on its respective task.

Decoder output and attention over the input are calculated using Eqs. We further study a shared decoder model (See Figure 2a) for the purpose of reducing the number of parameters as well as increasing the summary decoder's exposure to semantic information.

One single-layer LSTM is employed to sequentially produce the important semantic structures, followed by the summary.

Our output thus becomes y = [y s y a ], and the first timestep of the summary decoder is the last timestep of the semantic decoder.

Attention is calculated as in Eqs. 1-5.For both models, the loss becomes the weighted sum of the semantic loss and the summary loss: DISPLAYFORM5 In our experiments, we set ?? as 0.5 unless otherwise specified.

We also investigate multi-head attention BID37 over the input to acquire different language features.

To our knowledge, we are the first to apply it to the task of summarization.

As shown later in the results section, this method is indeed useful for summarization, with different heads learning different features.

In fact, the multi-head attention is particularly well-suited for our shared decoder model, as some heads learn to attend semantic aspects and others learn to attend summary aspects.

Target Semantic Output.

We use semantic role labeling (SRL) as the target semantic representation, which identifies predicate-argument structures.

To create the target data, articles in the training set are parsed in the style of PropBank BID14 using the DeepSRL parser from BID10 .

We then choose up to five SRL structures that have the most overlap with the reference summary.

Here we first consider matching headwords of predicates and arguments.

If no match is found, we consider all content words.

Note that the semantic labels are only used for model training.

At test time, no external resource beyond the article itself is used for summary decoding.

To further leverage semantic information, a dual attention mechanism is used to attend over the generated semantic output in addition to the input article during summary decoding.

Although attending over multiple sources of information has been studied for visual QA BID21 and sentence-level summarization BID3 ), these are computed over different encoder states.

In contrast, our dual attention mechanism considers decoding results from the semantic output.

Although our attention mechanism may appear close to intra-or self-attention BID30 , the function of our dual attention is to attend over a specific portion of the previously generated content that represents its own body of information, whileas traditional self-attention is predominantly used to discourage redundancy.

The context vector attending over the semantic decoder hidden states is calculated as follows: , segment-reranking BID32 , and hamming loss BID42 .

Here, we discuss our own segment-reranking beam search decoder, which encourages both beam diversity and semantic coverage.

Our reranker is applied only during summary decoding, where we rely on the generated semantic roles for global guidance.

The only modification made to the semantic decoder is a de-duplication strategy, where generated predicates seen in previous output are eliminated.

DISPLAYFORM0 Reranking.

Regular beam search chooses the hypotheses for the next timestep solely based on conditional likelihood (i.e., p(y a | x)).

In our proposed summary decoder, we also leverage our generated semantic information and curb redundancy during beam selection.

Reranking is performed every R timesteps based on the following scorer, where hypotheses with less repetition (r) and covering more content words from the generated semantics (s) are ranked higher: DISPLAYFORM1 where LRS is the longest repeating substring in the current hypotheses.

We define a repeating substring as a sequence of three or more tokens that appears in that order more than once in the hypothesis, with the intuition that long repeating fragments (ex. "the Senate proposed a tax bill; the Senate proposed a tax bill.") should be penalized more heavily than short ones (ex.

"the Senate proposed a tax bill; Senate proposed.").

s measures the percentage of generated semantic words reused by the current summary hypothesis, contingent on the predicate of semantic structure matching.

At every other timestep, we rank the hypotheses based on conditional likelihood and a weaker redundancy handler, r , that considers unigram novelty (i.e., percentage of unique content words): score = log (p(y a | x)) + ?? r .

We use R = 10, ?? = 0.4, ?? = 0.1, ?? = 0.1 in our experiments.

Beam Diversity.

In the standard beam search algorithm, all possible extensions to the beams are considered, resulting in a comparison of B ?? D hypotheses, where B is the number of beams and D is the vocabulary size.

In order to make the best use of the reranking algorithm, we develop two methods to enforce hypothesis diversity during non-reranking timesteps.

Beam Expansion.

Inspired by BID32 , we rank only the K highest scoring extensions from each beam, where K < B. This ensures that at least two unique hypotheses from the previous timestep will carry on to the next timestep.

Beam Selection.

We further reinforce hypothesis diversity through the two-step method of (1) likelihood selection and (2) dissimilarity selection.

In likelihood selection, we accept N hypotheses (where N < B) from our pool of B ?? K based solely on conditional probabilities, as in traditional beam search.

From the remaining hypotheses pool, we select B ??? N hypotheses on the basis of dissimilarity.

We choose the hypotheses with the highest dissimilarity score ???([h]

N , h ), where h is a candidate and [h] N are the hypotheses chosen during likelihood selection.

In our experiments, we use token-level Levenshtein edit distance BID15 as the dissimilarity metric, where (Lev(h, h ) ).

In experiments, we use B = 12, K = 6, N = 6.

DISPLAYFORM2

Datasets.

We experiment with two popular large datasets of news articles paired with human-written summaries: the CNN/Daily Mail corpus BID11 (henceforth CNN/DM) and the New York Times corpus BID29 ) (henceforth NYT).

For CNN/DM, we follow the experimental setup from BID31 and obtain a dataset consisting of 287,226 training pairs, 13,368 validation pairs, and 11,490 test pairs.

For NYT, we removed samples with articles of less than 100 words or summaries of less than 20 words.

We further remove samples with summaries containing information outside the article, e.g., "[AUTHOR]'s movie review on..." where the author's name does not appear in the article.

NYT consists of 280,146 training pairs and 15,564 pairs each for validation and test.

Training Details and Parameters.

For all experiments, a vocabulary of 50k words shared by input and output is used.

Model parameters and learning rate are adopted from prior work BID31 for comparison purpose.

All models are also trained in stages of increasing maximum token lengths to expedite training.

The models trained on the NYT dataset use an additional final training stage, where we optimized only on the summary loss (i.e., ?? = 1 in Eq. 7).

During decoding, unknown tokens are replaced with the highest scoring word in the corresponding attention distribution.

Baselines and Comparisons.

We include as our extractive baselines TEXTRANK BID19 and LEAD-2, the first 2 sentences of the input article, simulating the average length of the target summaries.

We consider as abstractive comparison models (1) vanilla seq2seq with attention (SEQ2SEQ) and (2) pointer-generator BID31 (POINT-GEN) , which is trained from scratch using the released code.

Results for variants of our model are also reported.1 Automatic Evaluation Metrics.

For automatic evaluation, we first report the F1 scores of ROUGE-1, 2, and L BID17 , and METEOR scores based on exact matching and full matching that considers paraphrases, synonyms, and stemming BID6 .We further measure two important aspects of summary quality: extractiveness-how much the summary reuses article content verbatim, and redundancy-how much the summary repeats itself.

For the first aspect, we utilize the density metric proposed by BID8 that calculates "the average length of extractive fragments": DISPLAYFORM0 where A represents the article, S represents the summary, and F (A, S) is a set of greedily matched extractive fragments from the article-summary pair.

Based on density, we propose a new redundancy metric: DISPLAYFORM1 where F (S) contains a set of fragments at least three tokens long that are repeated within the summary, and #f is the repetition frequency for fragment f .

Intuitively, longer fragments and more frequent repetition should be penalized more heavily.

Adversarial Evaluation.

Our pilot study suggests that the presence of minor irrelevant details in a summary often hurts a reader's understanding severely, but such an error cannot be captured or penalized by the recall-based ROUGE and METEOR metrics.

In order to test a model's ability to discern irrelevant information, we design a novel adversarial evaluation scheme, where we purposely mix a limited number of off-topic sentences into a test article.

The intuition is that if a summarization system truly understands the salient entities and events, it would ignore unrelated sentences.

We randomly select 5,000 articles from the CNN/DM test set with the "news" tag in the URL (mainly covering international or domestic events).

For each article, we randomly insert one to four sentences from articles in the test set with the "sports" tag.

For NYT, we randomly select 5,000 articles from the test set with government related tags ("U.S.", "Washington", "world") and insert one to four sentences from articles outside the domain ("arts", "sports", "technology").

We ensure that sentences containing a pronoun in the first five words are not interrupted from the previous sentence so that discourse chains are unlikely to be broken.

the content of the article, it would ignore unrelated sentences.

We took 15,000 articles from the NYT corpus with government related tags ("u.s.", "washington", "world"), and for each article, randomly inserted up to four sentences from articles outside the domain ("arts", "sports", "technology").

We ensured sentences that contained a pronoun in the first five words were not interrupted from the previous sentence so as not to break discourse or coreference chains.

DISPLAYFORM0 Figure 3: ROUGE-L and full METEOR scores on adversarial samples, where n irrelevant sentences are inserted into original test articles.

Our models (shared decoder for NYT, shared+MHA for CNN/DM) are sturdier against irrelevant information than seq2seq and pointer-generator.

separate decoder, and with or without multi-head attention (MHA).

In addition to ROUGE and METEOR, we also display extractive density (Dens.) and redundancy (Red.) (lower scores are preferred).

Best performing amongst our models are in bold.

All our ROUGE scores have a 95% confidence interval of at most ??0.04.

Our models all statistically significantly outperform seq2seq for ROUGE and METEOR (approximate randomization test, p < 0.01).

Our shared decoder with MHA model statistically significantly outperforms pointer-generator in ROUGE-1 and ROUGE-L on CNN/DM.

DISPLAYFORM1 The adversarial evaluation results on seq2seq, pointer-generator, and our shared decoder model are shown in Figure 3 , where our model consistently yields significantly better ROUGE-L and METEOR scores than the comparisons, and is less affected as more irrelevant sentences are added.

Sample summaries for an adversarial sample are shown in FIG0 .

We find that our semantic decoder plays an important role in capturing salient predicates and arguments, leading to higher quality summaries.

Automatic Evaluation.

The main results are displayed in Table 1 .

On CNN/DM, all of our models significantly outperform seq2seq across all metrics, and our shared decoder model with multi-head attention yields significantly better ROUGE (R-1 and R-L) scores than the pointer-generator on the same dataset (approximate randomization test, p < 0.01).

Despite the fact that both ROUGE and METEOR favor recall and thus reward longer summaries, our summaries that are often shorter than comparisons still produce significantly better ROUGE and METEOR scores than seq2seq on the NYT dataset.

Furthermore, our system summaries, by all model variations, reuse fewer and shorter phrases from the input (i.e., lower density scores) than the pointer-generator model, signifying a potentially stronger ability to rephrase.

Note that the density scores for seq2seq are likely deflated due to its inability to handle out-of-vocabulary words.

Our models also produce less redundant summaries than their abstractive comparisons.

Human Evaluation.

We further conducted a pilot human evaluation on 60 samples selected from the NYT test set.

For each article, the summaries by the seq2seq model, our shared decoder model, and the human reference were displayed in a randomized order.

Two human judges, who are native or fluent English speakers, were asked to read the article and rank the summaries against each other based on non-redundancy, fluency, faithfulness to input, and informativeness (whether the summary delivers the main points of the article).

Ties between system outputs were allowed but discouraged.

Table 2 , our model was ranked significantly higher than seq2seq across all metrics.

Surprisingly, our model's output was ranked higher than the reference summary in 26% of the samples for informativeness and fluency.

We believe this is due to the specific style of the NYT reference summaries and the shorter lengths of our summaries: the informal style of the reference summaries (e.g., frequently dropping subjects and articles) negatively affects its fluency rating, and readers find our shorter summaries to be more concise and to the point.

NON-RED.

FLUENCY FAITH.

INFORM.

HUMAN 1.4 ?? 0.7 1.5 ?? 0.7 1.5 ?? 0.8 1.6 ?? 0.8 SEQ2SEQ 2.0 ?? 0.8 2.4 ?? 0.7 2.1 ?? 0.8 2.4 ?? 0.7 OURS 1.6 ?? 0.7 2.1 ?? 0.8 1.9 ?? 0.7 2.0 ?? 0.7 Table 2 : Human ranking results on non-redundancy, fluency, faithfulness of summaries, and informativeness.

The mean (?? std.

dev.) for the rankings is shown (lower is better).

Across all metrics, the difference between our model (shared decoder) and human summary, as well as between our model and seq2seq is statistically significant (one-way ANOVA, p < 0.05).

Usage of Semantic Roles in Summaries.

We examine the utility of the generated semantic roles.

Across all models, approximately 44% of the generated predicates are part of the reference summary, indicating the adequacy of our semantic decoder.

Furthermore, across all models, approximately 65% of the generated predicates are reused by the generated summary, and approximately 53% of the SRL structures are reused by the system using a strict matching constraint, in which the predicate and head words for all arguments must match in the summary.

When gold-standard semantic roles are used for dual attention in place of our system generations, ROUGE scores increase by about half a point, indicating that improving semantic decoder in future work will further enhance the summaries.

Coverage.

We also conduct experiments using a coverage mechanism similar to the one used in BID31 .

We apply our coverage in two places: (1) over the input to handle redundancy, and (2) over the generated semantics to promote its reuse in the summary.

However, no significant difference is observed.

Our proposed reranker handles both issues in a more explicit way, and does not require the additional training time used to learn coverage parameters.

Alternative Semantic Representation.

Our summarization model can be trained with other types of semantic information.

For example, in addition to using the salient semantic roles from the input article, we also explore using SRL parses of the reference abstracts as training signals, but the higher level of abstraction required for semantic generation hurts performance by two ROUGE points for almost all models, indicating the type of semantic structure matters greatly for the ultimate summarization task.

For future work, other semantic representation along with novel model architecture will be explored.

For instance, other forms of semantic representation can be considered, such as frame semantics BID1 or Abstract Meaning Representation (AMR) BID2 .

Although previous work by has shown that seq2seq models are able to successfully generate linearized tree structures, we may also consider generating semantic roles with a hierarchical semantic decoder BID34 .

We presented a novel semantic-aware neural abstractive summarization model that jointly learns summarization and semantic parsing.

A novel dual attention mechanism was designed to better capture the semantic information for summarization.

A reranking-based decoder was proposed to promote the content coverage.

Our proposed adversarial evaluation demonstrated that our model was more adept at handling irrelevant information compared to popular neural summarization models.

Experiments on two large-scale news corpora showed that our model yielded significantly more informative, less redundant, and less extractive summaries.

Human evaluation further confirmed that our summaries were more informative and faithful than comparisons.

<|TLDR|>

@highlight

We propose a semantic-aware neural abstractive summarization model and a novel automatic summarization evaluation scheme that measures how well a model identifies off-topic information from adversarial samples.