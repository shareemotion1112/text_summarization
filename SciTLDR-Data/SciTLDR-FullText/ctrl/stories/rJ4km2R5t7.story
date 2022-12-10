For natural language understanding (NLU) technology to be maximally useful, it must be able to process language in a way that is not exclusive to a single task, genre, or dataset.

In pursuit of this objective, we introduce the General Language Understanding Evaluation (GLUE) benchmark, a collection of tools for evaluating the performance of models across a diverse set of existing NLU tasks.

By including tasks with limited training data, GLUE is designed to favor and encourage models that share general linguistic knowledge across tasks.

GLUE also includes a hand-crafted diagnostic test suite that enables detailed linguistic analysis of models.

We evaluate baselines based on current methods for transfer and representation learning and find that multi-task training on all tasks performs better than training a separate model per task.

However, the low absolute performance of our best model indicates the need for improved general NLU systems.

The human ability to understand language is general, flexible, and robust.

In contrast, most NLU models above the word level are designed for a specific task and struggle with out-of-domain data.

If we aspire to develop models with understanding beyond the detection of superficial correspondences between inputs and outputs, then it is critical to develop a more unified model that can learn to execute a range of different linguistic tasks in different domains.

To facilitate research in this direction, we present the General Language Understanding Evaluation (GLUE) benchmark: a collection of NLU tasks including question answering, sentiment analysis, and textual entailment, and an associated online platform for model evaluation, comparison, and analysis.

GLUE does not place any constraints on model architecture beyond the ability to process single-sentence and sentence-pair inputs and to make corresponding predictions.

For some GLUE tasks, training data is plentiful, but for others it is limited or fails to match the genre of the test set.

GLUE therefore favors models that can learn to represent linguistic knowledge in a way that facilitates sample-efficient learning and effective knowledge-transfer across tasks.

None of the datasets in GLUE were created from scratch for the benchmark; we rely on preexisting datasets because they have been implicitly agreed upon by the NLP community as challenging and interesting.

Four of the datasets feature privately-held test data, which will be used to ensure that the benchmark is used fairly.

Table 1 : Task descriptions and statistics.

All tasks are single sentence or sentence pair classification, except STS-B, which is a regression task.

MNLI has three classes; all other classification tasks have two.

Test sets shown in bold use labels that have never been made public in any form.

To better understand the challenged posed by GLUE, we conduct experiments with simple baselines and state-of-the-art sentence representation models.

We find that unified multi-task trained models slightly outperform comparable models trained on each task separately.

Our best multi-task model makes use of ELMo BID2 , a recently proposed pre-training technique.

However, this model still achieves a fairly low absolute score.

Analysis with our diagnostic dataset reveals that our baseline models deal well with strong lexical signals but struggle with deeper logical structure.

In summary, we offer: (i) A suite of nine sentence or sentence-pair NLU tasks, built on established annotated datasets and selected to cover a diverse range of text genres, dataset sizes, and degrees of difficulty.(ii) An online evaluation platform and leaderboard, based primarily on privately-held test data.

The platform is model-agnostic, and can evaluate any method capable of producing results on all nine tasks. (iii) An expert-constructed diagnostic evaluation dataset. (iv) Baseline results for several major existing approaches to sentence representation learning.

Collobert et al. (2011) used a multi-task model with a shared sentence understanding component to jointly learn POS tagging, chunking, named entity recognition, and semantic role labeling.

More recent work has explored using labels from core NLP tasks to supervise training of lower levels of deep neural networks BID10 Hashimoto et al., 2017 ) and automatically learning cross-task sharing mechanisms for multi-task learning BID6 ).Beyond multi-task learning, much work in developing general NLU systems has focused on sentence-to-vector encoders (Le & Mikolov, 2014; Kiros et al., 2015, i.a.) , leveraging unlabeled data (Hill et al., 2016; BID2 , labeled data (Conneau & Kiela, 2018; McCann et al., 2017) , and combinations of these (Collobert et al., 2011; BID11 .

In this line of work, a standard evaluation practice has emerged, recently codified as SentEval (Conneau et al., 2017; Conneau & Kiela, 2018) .

Like GLUE, SentEval relies on a set of existing classification tasks involving either one or two sentences as inputs.

Unlike GLUE, SentEval only evaluates sentenceto-vector encoders, making it well-suited for evaluating models on tasks involving sentences in isolation.

However, cross-sentence contextualization and alignment are instrumental in achieving state-of-the-art performance on tasks such as machine translation BID0 BID13 , question answering BID8 , and natural language inference BID5 .

GLUE is designed to facilitate the development of these methods: It is model-agnostic, allowing for any kind of representation or contextualization, including models that use no explicit vector or symbolic representations for sentences whatsoever.

GLUE also diverges from SentEval in the selection of evaluation tasks that are included in the suite.

Many of the SentEval tasks are closely related to sentiment analysis, such as MR (Pang & Lee, 2005) , SST BID9 , CR (Hu & Liu, 2004) , and SUBJ (Pang & Lee, 2004) .

Other tasks are so close to being solved that evaluation on them is relatively uninformative, such as MPQA (Wiebe et al., 2005) and TREC question classification BID14 .

In GLUE, we attempt to construct a benchmark that is both diverse and difficult.

McCann et al. (2018) introduce decaNLP, which also scores NLP systems based on their performance on multiple datasets.

Their benchmark recasts the ten evaluation tasks as question answering, converting tasks like summarization and text-to-SQL semantic parsing into question answering using automatic transformations.

That benchmark lacks the leaderboard and error analysis toolkit of GLUE, but more importantly, we see it as pursuing a more ambitious but less immediately practical goal:

While GLUE rewards methods that yield good performance on a circumscribed set of tasks using methods like those that are currently used for those tasks, their benchmark rewards systems that make progress toward their goal of unifying all of NLU under the rubric of question answering.

GLUE is centered on nine English sentence understanding tasks, which cover a broad range of domains, data quantities, and difficulties.

As the goal of GLUE is to spur development of generalizable NLU systems, we design the benchmark such that good performance should require a model to share substantial knowledge (e.g., trained parameters) across all tasks, while still maintaining some taskspecific components.

Though it is possible to train a single model for each task with no pretraining or other outside sources of knowledge and evaluate the resulting set of models on this benchmark, we expect that our inclusion of several data-scarce tasks will ultimately render this approach uncompetitive.

We describe the tasks below and in Table 1 .

Appendix A includes additional details.

Unless otherwise mentioned, tasks are evaluated on accuracy and are balanced across classes.

CoLA The Corpus of Linguistic Acceptability BID15 consists of English acceptability judgments drawn from books and journal articles on linguistic theory.

Each example is a sequence of words annotated with whether it is a grammatical English sentence.

Following the authors, we use Matthews correlation coefficient (Matthews, 1975) as the evaluation metric, which evaluates performance on unbalanced binary classification and ranges from -1 to 1, with 0 being the performance of uninformed guessing.

We use the standard test set, for which we obtained private labels from the authors.

We report a single performance number on the combination of the in-and out-of-domain sections of the test set.

SST-2 The Stanford Sentiment Treebank BID9 consists of sentences from movie reviews and human annotations of their sentiment.

The task is to predict the sentiment of a given sentence.

We use the two-way (positive/negative) class split, and use only sentence-level labels.

MRPC The Microsoft Research Paraphrase Corpus (Dolan & Brockett, 2005 ) is a corpus of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent.

Because the classes are imbalanced (68% positive), we follow common practice and report both accuracy and F1 score.

QQP The Quora Question Pairs 2 dataset is a collection of question pairs from the community question-answering website Quora.

The task is to determine whether a pair of questions are semantically equivalent.

As in MRPC, the class distribution in QQP is unbalanced (63% negative), so we report both accuracy and F1 score.

We use the standard test set, for which we obtained private labels from the authors.

We observe that the test set has a different label distribution than the training set.

The Semantic Textual Similarity Benchmark (Cer et al., 2017 ) is a collection of sentence pairs drawn from news headlines, video and image captions, and natural language inference data.

Each pair is human-annotated with a similarity score from 1 to 5; the task is to predict these scores.

Follow common practice, we evaluate using Pearson and Spearman correlation coefficients.

MNLI The Multi-Genre Natural Language Inference Corpus (Williams et al., 2018 ) is a crowdsourced collection of sentence pairs with textual entailment annotations.

Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).

The premise sentences are gathered from ten different sources, including transcribed speech, fiction, and government reports.

We use the standard test set, for which we obtained private labels from the authors, and evaluate on both the matched (in-domain) and mismatched (cross-domain) sections.

We also use and recommend the SNLI corpus (Bowman et al., 2015) as 550k examples of auxiliary training data.

QNLI The Stanford Question Answering Dataset BID4 ) is a question-answering dataset consisting of question-paragraph pairs, where one of the sentences in the paragraph (drawn from Wikipedia) contains the answer to the corresponding question (written by an annotator).

We convert the task into sentence pair classification by forming a pair between each question and each sentence in the corresponding context, and filtering out pairs with low lexical overlap between the question and the context sentence.

The task is to determine whether the context sentence contains the answer to the question.

This modified version of the original task removes the requirement that the model select the exact answer, but also removes the simplifying assumptions that the answer is always present in the input and that lexical overlap is a reliable cue.

This process of recasting existing datasets into NLI is similar to methods introduced in BID16 and expanded upon in Demszky et al. (2018) .

We call the converted dataset QNLI (Question-answering NLI).

RTE The Recognizing Textual Entailment (RTE) datasets come from a series of annual textual entailment challenges.

We combine the data from RTE1 (Dagan et al., 2006) , RTE2 (Bar Haim et al., 2006) , RTE3 (Giampiccolo et al., 2007) , and RTE5 (Bentivogli et al., 2009) .

4 Examples are constructed based on news and Wikipedia text.

We convert all datasets to a two-class split, where for three-class datasets we collapse neutral and contradiction into not entailment, for consistency.

WNLI The Winograd Schema Challenge (Levesque et al., 2011 ) is a reading comprehension task in which a system must read a sentence with a pronoun and select the referent of that pronoun from a list of choices.

The examples are manually constructed to foil simple statistical methods: Each one is contingent on contextual information provided by a single word or phrase in the sentence.

To convert the problem into sentence pair classification, we construct sentence pairs by replacing the ambiguous pronoun with each possible referent.

The task is to predict if the sentence with the I have never seen a hummingbird not flying.

I have never seen a hummingbird.

N E Table 3 : Examples from the diagnostic set.

Fwd (resp.

Bwd) denotes the label when sentence 1 (resp.

sentence 2) is the premise.

Labels are entailment (E), neutral (N), or contradiction (C).

Examples are tagged with the phenomena they demonstrate, and each phenomenon belongs to one of four broad categories (in parentheses).pronoun substituted is entailed by the original sentence.

We use a small evaluation set consisting of new examples derived from fiction books 5 that was shared privately by the authors of the original corpus.

While the included training set is balanced between two classes, the test set is imbalanced between them (65% not entailment).

Also, due to a data quirk, the development set is adversarial: hypotheses are sometimes shared between training and development examples, so if a model memorizes the training examples, they will predict the wrong label on corresponding development set example.

As with QNLI, each example is evaluated separately, so there is not a systematic correspondence between a model's score on this task and its score on the unconverted original task.

We call converted dataset WNLI (Winograd NLI).

The GLUE benchmark follows the same evaluation model as SemEval and Kaggle.

To evaluate a system on the benchmark, one must run the system on the provided test data for the tasks, then upload the results to the website gluebenchmark.com for scoring.

The benchmark site shows per-task scores and a macro-average of those scores to determine a system's position on the leaderboard.

For tasks with multiple metrics (e.g., accuracy and F1), we use an unweighted average of the metrics as the score for the task when computing the overall macro-average.

The website also provides fine-and coarse-grained results on the diagnostic dataset.

See Appendix D for details.

Drawing inspiration from the FraCaS suite (Cooper et al., 1996) and the recent Build-It-Break-It competition (Ettinger et al., 2017), we include a small, manually-curated test set for the analysis of system performance.

While the main benchmark mostly reflects an application-driven distribution of examples, our diagnostic dataset highlights a pre-defined set of phenomena that we believe are interesting and important for models to capture.

We show the full set of phenomena in TAB1 .Each diagnostic example is an NLI sentence pair with tags for the phenomena demonstrated.

The NLI task is well-suited to this kind of analysis, as it can easily evaluate the full set of skills involved in (ungrounded) sentence understanding, from resolution of syntactic ambiguity to pragmatic reasoning with world knowledge.

We ensure the data is reasonably diverse by producing examples for a variety of linguistic phenomena and basing our examples on naturally-occurring sentences from several domains (news, Reddit, Wikipedia, academic papers).

This approaches differs from that of FraCaS, which was designed to test linguistic theories with a minimal and uniform set of examples.

A sample from our dataset is shown in Table 3 .Annotation Process We begin with a target set of phenomena, based roughly on those used in the FraCaS suite (Cooper et al., 1996) .

We construct each example by locating a sentence that can be easily made to demonstrate a target phenomenon, and editing it in two ways to produce an appropriate sentence pair.

We make minimal modifications so as to maintain high lexical and structural overlap within each sentence pair and limit superficial cues.

We then label the inference relationships between the sentences, considering each sentence alternatively as the premise, producing two labeled examples for each pair (1100 total).

Where possible, we produce several pairs with different labels for a single source sentence, to have minimal sets of sentence pairs that are lexically and structurally very similar but correspond to different entailment relationships.

The resulting labels are 42% entailment, 35% neutral, and 23% contradiction.

Evaluation Since the class distribution in the diagnostic set is not balanced, we use R 3 (Gorodkin, 2004), a three-class generalization of the Matthews correlation coefficient, for evaluation.

In light of recent work showing that crowdsourced data often contains artifacts which can be exploited to perform well without solving the intended task BID7 BID3 Tsuchiya, 2018, i.a.) , we audit the data for such artifacts.

We reproduce the methodology of Gururangan et al. (2018) , training two fastText classifiers (Joulin et al., 2016) to predict entailment labels on SNLI and MNLI using only the hypothesis as input.

The models respectively get near-chance accuracies of 32.7% and 36.4% on our diagnostic data, showing that the data does not suffer from such artifacts.

To establish human baseline performance on the diagnostic set, we have six NLP researchers annotate 50 sentence pairs (100 entailment examples) randomly sampled from the diagnostic set.

Interannotator agreement is high, with a Fleiss's κ of 0.73.

The average R 3 score among the annotators is 0.80, much higher than any of the baseline systems described in Section 5.Intended Use The diagnostic examples are hand-picked to address certain phenomena, and NLI is a task with no natural input distribution, so we do not expect performance on the diagnostic set to reflect overall performance or generalization in downstream applications.

Performance on the analysis set should be compared between models but not between categories.

The set is provided not as a benchmark, but as an analysis tool for error analysis, qualitative model comparison, and development of adversarial examples.

For baselines, we evaluate a multi-task learning model trained on the GLUE tasks, as well as several variants based on recent pre-training methods.

We briefly describe them here.

See Appendix B for details.

We implement our models in the AllenNLP library (Gardner et al., 2017) .

Original code for the baselines is available at https://github.com/nyu-mll/GLUE-baselines and a newer version is available at https://github.com/jsalt18-sentence-repl/jiant.Architecture Our simplest baseline architecture is based on sentence-to-vector encoders, and sets aside GLUE's ability to evaluate models with more complex structures.

Taking inspiration from Conneau et al. (2017) , the model uses a two-layer, 1500D (per direction) BiLSTM with max pooling and 300D GloVe word embeddings (840B Common Crawl version; BID1 .

For single-sentence tasks, we encode the sentence and pass the resulting vector to a classifier.

For sentence-pair tasks, we encode sentences independently to produce vectors u, v, and pass [u; v; |u − v|; u * v] to a classifier.

The classifier is an MLP with a 512D hidden layer.

We also consider a variant of our model which for sentence pair tasks uses an attention mechanism inspired by BID8 between all pairs of words, followed by a second BiLSTM with max pooling.

By explicitly modeling the interaction between sentences, these models fall outside the sentence-to-vector paradigm.

Pre-Training We augment our base model with two recent methods for pre-training: ELMo and CoVe.

We use existing trained models for both.

ELMo uses a pair of two-layer neural language models trained on the Billion Word Benchmark (Chelba et al., 2013 Training We train our models with the BiLSTM sentence encoder and post-attention BiLSTMs shared across tasks, and classifiers trained separately for each task.

For each training update, we sample a task to train with a probability proportional to the number of training examples for each task.

We train our models with Adam (Kingma & Ba, 2015) with initial learning rate 10 −4 and batch size 128.

We use the macro-average score as the validation metric and stop training when the learning rate drops below 10 −5 or performance does not improve after 5 validation checks.

We also train a set of single-task models, which are configured and trained identically, but share no parameters.

To allow for fair comparisons with the multi-task analogs, we do not tune parameter or training settings for each task, so these single-task models do not generally represent the state of the art for each task.

Sentence Representation Models Finally, we evaluate the following trained sentence-to-vector encoder models using our benchmark: average bag-of-words using GloVe embeddings (CBoW), Skip-Thought (Kiros et al., 2015) , InferSent (Conneau et al., 2017) , DisSent (Nie et al., 2017) , and GenSen BID11 .

For these models, we only train task-specific classifiers on the representations they produce.

We train three runs of each model and evaluate the run with the best macro-average development set performance (see TAB8 in Appendix C).

For single-task and sentence representation models, we evaluate the best run for each individual task.

We present performance on the main benchmark tasks in TAB4 .We find that multi-task training yields better overall scores over single-task training amongst models using attention or ELMo.

Attention generally has negligible or negative aggregate effect in single task training, but helps in multi-task training.

We see a consistent improvement in using ELMo embeddings in place of GloVe or CoVe embeddings, particularly for single-sentence tasks.

Using CoVe has mixed effects over using only GloVe.

Among the pre-trained sentence representation models, we observe fairly consistent gains moving from CBoW to Skip-Thought to Infersent and GenSen.

Relative to the models trained directly on the GLUE tasks, InferSent is competitive and GenSen outperforms all but the two best.

Looking at results per task, we find that the sentence representation models substantially underperform on CoLA compared to the models directly trained on the task.

On the other hand, for STS-B, models trained directly on the task lag significantly behind the performance of the best sentence representation model.

Finally, there are tasks for which no model does particularly well.

On WNLI, no model exceeds most-frequent-class guessing (65.1%) and we substitute the model predictions for the most-frequent baseline.

On RTE and in aggregate, even our best baselines leave room for improvement.

These early results indicate that solving GLUE is beyond the capabilities of current models and methods.

We analyze the baselines by evaluating each model's MNLI classifier on the diagnostic set to get a better sense of their linguistic capabilities.

Results are presented in TAB6 .Coarse Categories Overall performance is low for all models: The highest total score of 28 still denotes poor absolute performance.

Performance tends to be higher on Predicate-Argument Structure and lower on Logic, though numbers are not closely comparable across categories.

Unlike on the main benchmark, the multi-task models are almost always outperformed by their single-task counterparts.

This is perhaps unsurprising, since with our simple multi-task training regime, there is likely some destructive interference between MNLI and the other tasks.

The models trained on the GLUE tasks largely outperform the pretrained sentence representation models, with the exception of GenSen.

Using attention has a greater influence on diagnostic scores than using ELMo or CoVe, which we take to indicate that attention is especially important for generalization in NLI.Fine-Grained Subcategories Most models handle universal quantification relatively well.

Looking at relevant examples, it seems that relying on lexical cues such as "all" often suffices for good performance.

Similarly, lexical cues often provide good signal in morphological negation examples.

We observe varying weaknesses between models.

Double negation is especially difficult for the GLUE-trained models that only use GloVe embeddings.

This is ameliorated by ELMo, and to some degree CoVe.

Also, attention has mixed effects on overall results, and models with attention tend to struggle with downward monotonicity.

Examining their predictions, we found that the models are sensitive to hypernym/hyponym substitution and word deletion as a signal of entailment, but predict it in the wrong direction (as if the substituted/deleted word were in an upward monotone context).

This is consistent with recent findings by McCoy & Linzen (2019) that these systems use the subsequence relation between premise and hypothesis as a heuristic shortcut.

Restrictivity examples, which often depend on nuances of quantifier scope, are especially difficult for almost all models.

Overall, there is evidence that going beyond sentence-to-vector representations, e.g. with an attention mechanism, might aid performance on out-of-domain data, and that transfer methods like ELMo and CoVe encode linguistic information specific to their supervision signal.

However, increased representational capacity may lead to overfitting, such as the failure of attention models in downward monotone contexts.

We expect that our platform and diagnostic dataset will be useful for similar analyses in the future, so that model designers can better understand their models' generalization behavior and implicit knowledge.

We introduce GLUE, a platform and collection of resources for evaluating and analyzing natural language understanding systems.

We find that, in aggregate, models trained jointly on our tasks see better performance than the combined performance of models trained for each task separately.

We confirm the utility of attention mechanisms and transfer learning methods such as ELMo in NLU systems, which combine to outperform the best sentence representation models on the GLUE benchmark, but still leave room for improvement.

When evaluating these models on our diagnostic dataset, we find that they fail (often spectacularly) on many linguistic phenomena, suggesting possible avenues for future work.

In sum, the question of how to design general-purpose NLU models remains unanswered, and we believe that GLUE can provide fertile soil for addressing this challenge.

A ADDITIONAL BENCHMARK DETAILS QNLI To construct a balanced dataset, we select all pairs in which the most similar sentence to the question was not the answer sentence, as well as an equal amount of cases in which the correct sentence was the most similar to the question, but another distracting sentence was a close second.

Our similarity metric is based on CBoW representations with pre-trained GloVe embeddings.

This approach to converting pre-existing datasets into NLI format is closely related to recent work by BID16 , as well as to the original motivation for textual entailment presented by Dagan et al. (2006) .

Both argue that many NLP tasks can be productively reduced to textual entailment.

We implement our attention mechanism as follows: given two sequences of hidden states u 1 , u 2 , . . . , u M and v 1 , v 2 , . . .

, v N , we first compute matrix H where H ij = u i · v j .

For each u i , we get attention weights α i by taking a softmax over the i th row of H, and get the corresponding context vectorṽ i = j α ij v j by taking the attention-weighted sum of the v j .

We pass a second BiLSTM with max pooling over the sequence [u 1 ;ṽ 1 ], . . . [u M ;ṽ M ] to produce u .

We process the v j vectors analogously to obtain v .

Finally, we feed [u ; v ; |u − v |; u * v ] into a classifier.

We train our models with the BiLSTM sentence encoder and post-attention BiLSTMs shared across tasks, and classifiers trained separately for each task.

For each training update, we sample a task to train with a probability proportional to the number of training examples for each task.

We scale each task's loss inversely proportional to the number of examples for that task, which we found to improve overall performance.

We train our models with Adam (Kingma & Ba, 2015) with initial learning rate 10 −3 , batch size 128, and gradient clipping.

We use macro-average score over all tasks as our validation metric, and perform a validation check every 10k updates.

We divide the learning rate by 5 whenever validation performance does not improve.

We stop training when the learning rate drops below 10 −5 or performance does not improve after 5 validation checks.

We evaluate the following sentence representation models:1.

CBoW, the average of the GloVe embeddings of the tokens in the sentence. , a BiLSTM with max-pooling trained to predict the discourse marker (because, so, etc.) relating two sentences on data derived from TBC.

We use the variant trained for eight-way classification.

5.

GenSen BID11 , a sequence-to-sequence model trained on a variety of supervised and unsupervised objectives.

We use the variant of the model trained on both MNLI and SNLI, the Skip-Thought objective on TBC, and a constituency parsing objective on the Billion Word Benchmark.

We train task-specific classifiers on top of frozen sentence encoders, using the default parameters from SentEval.

See https://github.com/nyu-mll/SentEval for details and code.

The GLUE website limits users to two submissions per day in order to avoid overfitting to the private test data.

To provide a reference for future work on GLUE, we present the best development set results achieved by our baselines in TAB8 .

GLUE's online platform is built using React, Redux and TypeScript.

We use Google Firebase for data storage and Google Cloud Functions to host and run our grading script when a submission is made.

FIG1 shows the visual presentation of our baselines on the leaderboard.

Table 7 : Diagnostic dataset statistics by coarse-grained category.

Note that some examples may be tagged with phenomena belonging to multiple categories.

The dataset is designed to allow for analyzing many levels of natural language understanding, from word meaning and sentence structure to high-level reasoning and application of world knowledge.

To make this kind of analysis feasible, we first identify four broad categories of phenomena: Lexical Semantics, Predicate-Argument Structure, Logic, and Knowledge.

However, since these categories are vague, we divide each into a larger set of fine-grained subcategories.

Descriptions of all of the fine-grained categories are given in the remainder of this section.

These categories are just one lens that can be used to understand linguistic phenomena and entailment, and there is certainly room to argue about how examples should be categorized, what the categories should be, etc.

These categories are not based on any particular linguistic theory, but broadly based on issues that linguists have often identified and modeled in the study of syntax and semantics.

The dataset is provided not as a benchmark, but as an analysis tool to paint in broad strokes the kinds of phenomena a model may or may not capture, and to provide a set of examples that can serve for error analysis, qualitative model comparison, and development of adversarial examples that expose a model's weaknesses.

Because the distribution of language is somewhat arbitrary, it will not be helpful to compare performance of the same model on different categories.

Rather, we recommend comparing performance that different models score on the same category, or using the reported scores as a guide for error analysis.

We show coarse-grain category counts and label distributions of the diagnostic set in Table 7 .

These phenomena center on aspects of word meaning.

Lexical Entailment Entailment can be applied not only on the sentence level, but the word level.

For example, we say "dog" lexically entails "animal" because anything that is a dog is also an animal, and "dog" lexically contradicts "cat" because it is impossible to be both at once.

This relationship applies to many types of words (nouns, adjectives, verbs, many prepositions, etc.) and the relationship between lexical and sentential entailment has been deeply explored, e.g., in systems of natural logic.

This connection often hinges on monotonicity in language, so many Lexical Entailment examples will also be tagged with one of the Monotone categories, though we do not do this in every case (see Monotonicity, under Logic)

.Morphological Negation This is a special case of lexical contradiction where one word is derived from the other: from "affordable" to "unaffordable", "agree" to "disagree", etc.

We also include examples like "ever" and "never".

We also label these examples with Negation or Double Negation, since they can be viewed as involving a word-level logical negation.

Factivity Propositions appearing in a sentence may be in any entailment relation with the sentence as a whole, depending on the context in which they appear.

In many cases, this is determined by lexical triggers (usually verbs or adverbs) in the sentence.

For example,• "I recognize that X" entails "X".• "I did not recognize that X" entails "X".• "I believe that X" does not entail "X".• "I am refusing to do X" contradicts "I am doing X".• "I am not refusing to do X" does not contradict "I am doing X".• "I almost finished X" contradicts "I finished X".•

"I barely finished X" entails "I finished X".Constructions like "I recognize that X" are often called factive, since the entailment (of X above, regarded as a presupposition) persists even under negation.

Constructions like "I am refusing to do X" above are often called implicative, and are sensitive to negation.

There are also cases where a sentence (non-)entails the existence of an entity mentioned in it, for example "I have found a unicorn" entails "A unicorn exists" while "I am looking for a unicorn" doesn't necessarily entail "A unicorn exists".

Readings where the entity does not necessarily exist are often called intensional readings, since they seem to deal with the properties denoted by a description (its intension) rather than being reducible to the set of entities that match the description (its extension, which in cases of non-existence will be empty).We place all examples involving these phenomena under the label of Factivity.

While it often depends on context to determine whether a nested proposition or existence of an entity is entailed by the overall statement, very often it relies heavily on lexical triggers, so we place the category under Lexical Semantics.

Symmetry/Collectivity Some propositions denote symmetric relations, while others do not.

For example, "John married Gary" entails "Gary married John" but "John likes Gary" does not entail "Gary likes John".

Symmetric relations can often be rephrased by collecting both arguments into the subject: "John met Gary" entails "John and Gary met".

Whether a relation is symmetric, or admits collecting its arguments into the subject, is often determined by its head word (e.g., "like", "marry" or "meet"), so we classify it under Lexical Semantics.

Redundancy If a word can be removed from a sentence without changing its meaning, that means the word's meaning was more-or-less adequately expressed by the sentence; so, identifying these cases reflects an understanding of both lexical and sentential semantics.

Named Entities Words often name entities that exist in the world.

There are many different kinds of understanding we might wish to understand about these names, including their compositional structure (for example, the "Baltimore Police" is the same as the "Police of the City of Baltimore") or their real-world referents and acronym expansions (for example, "SNL" is "Saturday Night Live").

This category is closely related to World Knowledge, but focuses on the semantics of names as lexical items rather than background knowledge about their denoted entities.

Quantifiers Logical quantification in natural language is often expressed through lexical triggers such as "every", "most", "some", and "no".

While we reserve the categories in Quantification and Monotonicity for entailments involving operations on these quantifiers and their arguments, we choose to regard the interchangeability of quantifiers (e.g., in many cases "most" entails "many") as a question of lexical semantics.

An important component of understanding the meaning of a sentence is understanding how its parts are composed together into a whole.

In this category, we address issues across that spectrum, from syntactic ambiguity to semantic roles and coreference.

Syntactic Ambiguity: Relative Clauses, Coordination Scope These two categories deal purely with resolving syntactic ambiguity.

Relative clauses and coordination scope are both sources of a great amount of ambiguity in English.

Prepositional phrases Prepositional phrase attachment is a particularly difficult problem that syntactic parsers in NLP systems continue to struggle with.

We view it as a problem both of syntax and semantics, since prepositional phrases can express a wide variety of semantic roles and often semantically apply beyond their direct syntactic attachment.

Core Arguments Verbs select for particular arguments, particularly subjects and objects, which might be interchangeable depending on the context or the surface form.

One example is the ergative alternation: "Jake broke the vase" entails "the vase broke" but "Jake broke the vase" does not entail "Jake broke".

Other rearrangements of core arguments, such as those seen in Symmetry/Collectivity, also fall under the Core Arguments label.

Alternations: Active/Passive, Genitives/Partitives, Nominalization, Datives All four of these categories correspond to syntactic alternations that are known to follow specific patterns in English:• Active/Passive: "I saw him" is equivalent to "He was seen by me" and entails "

He was seen".• Genitives/Partitives: "the elephant's foot" is the same thing as "the foot of the elephant".• Nominalization:

"I caused him to submit his resignation" entails "I caused the submission of his resignation".• Datives: "I baked him a cake" entails "I baked a cake for him" and "I baked a cake" but not "I baked him".Ellipsis/Implicits Often, the argument of a verb or other predicate is omitted (elided) in the text, with the reader filling in the gap.

We can construct entailment examples by explicitly filling in the gap with the correct or incorrect referents.

For example, the premise "Putin is so entrenched within Russias ruling system that many of its members can imagine no other leader" entails "Putin is so entrenched within Russias ruling system that many of its members can imagine no other leader than Putin" and contradicts "Putin is so entrenched within Russias ruling system that many of its members can imagine no other leader than themselves."

This is often regarded as a special case of anaphora, but we decided to split out these cases from explicit anaphora, which is often also regarded as a case of coreference (and attempted to some degree in modern coreference resolution systems).Anaphora/Coreference Coreference refers to when multiple expressions refer to the same entity or event.

It is closely related to Anaphora, where the meaning of an expression depends on another (antecedent) expression in context.

These two phenomena have significant overlap; for example, pronouns ("she", "we", "it") are anaphors that are co-referent with their antecedents.

However, they also may occur independently, such as coreference between two definite noun phrases (e.g., "Theresa May "and the "British Prime Minister") that refer to the same entity, or anaphora from a word like "other" which requires an antecedent to distinguish something from.

In this category we only include cases where there is an explicit phrase (anaphoric or not) that is co-referent with an antecedent or other phrase.

We construct examples for these in much the same way as for Ellipsis/Implicits.

Intersectivity Many modifiers, especially adjectives, allow non-intersective uses, which affect their entailment behavior.

For example:• Intersective: "He is a violinist and an old surgeon" entails "He is an old violinist" and "He is a surgeon".• Non-intersective: "He is a violinist and a skilled surgeon" does not entail "He is a skilled violinist".• Non-intersective:

"He is a fake surgeon" does not entail "He is a surgeon".Generally, an intersective use of a modifier, like "old" in "old men", is one which may be interpreted as referring to the set of entities with both properties (they are old and they are men).

Linguists often formalize this using set intersection, hence the name.

Intersectivity is related to Factivity.

For example, "fake" may be regarded as a counter-implicative modifier, and these examples will be labeled as such.

However, we choose to categorize intersectivity under predicate-argument structure rather than lexical semantics, because generally the same word will admit both intersective and non-intersective uses, so it may be regarded as an ambiguity of argument structure.

Restrictivity Restrictivity is most often used to refer to a property of uses of noun modifiers.

In particular, a restrictive use of a modifier is one that serves to identify the entity or entities being described, whereas a non-restrictive use adds extra details to the identified entity.

The distinction can often be highlighted by entailments:• Restrictive: "I finished all of my homework due today" does not entail "I finished all of my homework".• Non-restrictive: "I got rid of all those pesky bedbugs" entails "I got rid of all those bedbugs".Modifiers that are commonly used non-restrictively are appositives, relative clauses starting with "which" or "who", and expletives (e.g. "pesky").

Non-restrictive uses can appear in many forms.

With an understanding of the structure of a sentence, there is often a baseline set of shallow conclusions that can be drawn using logical operators and often modeled using the mathematical tools of logic.

Indeed, the development of mathematical logic was initially guided by questions about natural language meaning, from Aristotelian syllogisms to Fregean symbols.

The notion of entailment is also borrowed from mathematical logic.

Propositional Structure: Negation, Double Negation, Conjunction, Disjunction, Conditionals All of the basic operations of propositional logic appear in natural language, and we tag them where they are relevant to our examples:• Negation: "The cat sat on the mat" contradicts "The cat did not sit on the mat".• Double negation: "The market is not impossible to navigate" entails "The market is possible to navigate".

<|TLDR|>

@highlight

We present a multi-task benchmark and analysis platform for evaluating generalization in natural language understanding systems.