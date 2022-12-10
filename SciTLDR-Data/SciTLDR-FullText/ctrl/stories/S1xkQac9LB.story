We review three limitations of BLEU and ROUGE – the most popular metrics used to assess reference summaries against hypothesis summaries, come up with criteria for what a good metric should behave like and propose concrete ways to assess the performance of a metric in detail and show the potential of Transformers-based Language Models to assess reference summaries against hypothesis summaries.

Evaluation metrics play a central role in the machine learning community.

They direct the efforts of the research community and are used to define the state of the art models.

In machine translation and summarization, the two most common metrics used for evaluating similarity between candidate and reference texts are BLEU [Papineni et al., 2002] and ROUGE [Lin, 2004] .

Both approaches rely on counting the matching n-grams in the candidates summary to n-grams in the reference text.

BLEU is precision focused while ROUGE is recall focused.

These metrics have posed serious limitations and have already been criticized by the academic community [Reiter, 2018] [Callison-Burch et al., 2006] [Sulem et al., 2018] [Novikova et al., 2017] .

In this work, we formulate an empirical criticism of BLEU and ROUGE, establish a criteria that a sound evaluation metric should have and propose concrete ways to test any metric towards these criteria.

We also use recent advances in NLP to design a data-driven metric addressing the weaknesses found in BLEU and ROUGE and scoring high on the criteria for a sound evaluation metric.

2 Related Work 2.1 BLEU, ROUGE and n-gram matching approaches BLEU (Bilingual Evaluation Understudy) [Papineni et al., 2002] and ROUGE (Recall-Oriented Understudy for Gisting Evaluation) [Lin, 2004] have been used to evaluate many NLP tasks for almost two decades.

The general acceptance of these methods depend on many factors including their simplicity and the intuitive interpretability.

Yet the main factor is the claim that they highly correlate with human judgement [Papineni et al., 2002] .

This has been criticised extensively by the literature and the shortcomings of these methods have been widely studied.

Reiter [Reiter, 2018] , in his structured review of BLEU, finds a low correlation between BLEU and human judgment.

Callison et al [Callison-Burch et al., 2006] examines BLEU in the context of machine translation and find that BLEU does neither correlate with human judgment on adequacy(whether the hypothesis sentence adequately captures the meaning of the reference sentence) nor fluency(the quality of language in a sentence).

Sulem et al [Sulem et al., 2018] examines BLEU in the context of text simplification on grammaticality, meaning preservation and simplicity and report BLEU has very low or in some cases negative correlation with human judgment.

Language modeling has become an important NLP technique thanks to the ability to apply it to various NLP tasks as explained in Radford et al [Radford et al., 2019] .

There are two leading architectures for language modeling Recurrent Neural Networks (RNNs) [Mikolov et al., 2010] and Transformers [Vaswani et al., 2017] .

RNNs handle the input tokens, words or characters, one by one through time to learn the relationship between them, whereas, transformers receive a segment of tokens and learn the dependencies between them using an attention mechanism.

While BLEU and ROUGE are defined in a discrete space new evaluation metric can be defined in this continuous space.

BERTscore [Zhang et al., 2019] uses word embeddings and cosine similarity to create a score array and use greedy matching to maximize the similarity score.

Sentence Mover's Similarity [Clark et al., 2019] uses the mover similarity, Wasserstein distance, between sentence embedding generated from averaging the word embeddings in a sentence.

One other evaluation method proposed is RUSE [Shimanaka et al., 2018] this method proposes embedding both sentences separately and pooling them to a given size.

After that they use a pre trained MLP to predict on different tasks.

This quality estimator metric is then proposed to be used in language evaluation.

Our proposed methodology is to take neural language evaluation beyond architecture specifications.

We are proposing a framework in which an evaluator's success can be determined.

In this part, we discuss three significant limitations of BLEU and ROUGE.

These metrics can assign: High scores to semantically opposite translations/summaries, Low scores to semantically related translations/summaries and High scores to unintelligible translations/summaries.

Suppose that we have a reference summary s1.

By adding a few negation terms to s1, one can create a summary s2 which is semantically opposite to s1 but yet has a high BLEU/ROUGE score.

In addition not to be sensitive to negation, BLEU and ROUGE score can give low scores to sentences with equivalent meaning.

If s2 is a paraphrase of s1, the meaning will be the same ;however, the overlap between words in s1 and s2 will not necessarily be significant.

A third weakness of BLEU and ROUGE is that in their simplest implementations, they are insensitive to word permutation and can give very high scores to unintelligible sentences.

Although higher order BLEU scores are expected to mitigate this effect, they make the metric more sensitive to paraphrasing.

To overcome the previously highlighted challenges and provide a framework by which metrics comparing reference summaries/translation can be assessed and improved, we established firstprinciples criteria on what a good evaluator should do.

The first one is that it should be highly correlated with human judgement in semantic similarity.

The second one is that it should be able to distinguish sentences which are in logical contradiction, logically unrelated or in logical agreement.

The third one is that given s1, s2 which are semantically similar, eval(s1,s2) > eval(s1,s2(corrupted) > eval(s1,s2(more corrupted)) where corruption here includes removing words, adding noise to the word order or including grammatical mistakes.

We will now give a more detailed example to how the scorecard can be implemented.

For every dimension of the scorecard the experiments are done with three metrics.

BLEU with equal weights between 1 to 4 grams.

ROUGE with averaging ROUGE-1 and ROUGE-2 and the a neural evaluator.

The evaluator is the RoBERTa large pre-trained model , which we fine tune it to predict sentence similarity (0-5 scale) on the STS-B benchmark dataset (8628 sentence pairs).

The first expectation from a google similarty metric is to correlate highly with human judgment in terms of assessing semantic similarity.

Here we assessed BLEU and ROUGE on the STS-B benchmark and compared their performance to a RoBERTa model fine tuned for semantic similarity (Table 1) .

Another characteristic of a good metric is to differentiate the argument, core meaning, in a sentence and take it into account when assessing hypothesis text with references.

Here we used the MNLI dataset where for each text we have three hypothesis text representing contradiction, neutral and entailment.

We expect a good metric to rank entailment higher than neutral and both of them higher than contradiction.

To assess the quality of a metric we propose to use the Spearman's ranked correlation and in Table 4 .2.2 we also experiment with Kendall's τ .

Here we observe that the RoBERTa model remarkably outperforms BLEU and ROUGE and both of these metrics show very little correlation with human judgment.

For assessing the third criteria.

We start with 3479 sentence pairs from the MNLI dataset that are labelled as entailment.

We introduce random corruptions such as random insertion, deletion and grammatical errors as in [Zhao et al., 2019] .

We use two different set of parameters for different corruption levels and expect that a good metric would rank the original similar sentence higher than the less corrupted and both higher than the more corrupted sentence.

Here we also propose to use the Spearman's ranked correlation and also experiment with Kendall's τ .

We report results on

In this work, we have established a framework to assess metrics comparing the quality of reference and hypothesis summary/translations.

Based on these criteria, we compare evaluators using recent Transformers to BLEU and ROUGE and highlight their potential to replace BLEU and ROUGE.

<|TLDR|>

@highlight

New method for assessing the quaility of similarity evaluators and showing potential of Transformer-based language models in replacing BLEU and ROUGE.