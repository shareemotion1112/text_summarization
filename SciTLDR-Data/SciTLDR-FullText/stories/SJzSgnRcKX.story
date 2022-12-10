Contextualized representation models such as ELMo (Peters et al., 2018a) and BERT (Devlin et al., 2018) have recently achieved state-of-the-art results on a diverse array of downstream NLP tasks.

Building on recent token-level probing work, we introduce a novel edge probing task design and construct a broad suite of sub-sentence tasks derived from the traditional structured NLP pipeline.

We probe word-level contextual representations from four recent models and investigate how they encode sentence structure across a range of syntactic, semantic, local, and long-range phenomena.

We find that existing models trained on language modeling and translation produce strong representations for syntactic phenomena, but only offer comparably small improvements on semantic tasks over a non-contextual baseline.

Pretrained word embeddings BID30 BID32 are a staple tool for NLP.

These models provide continuous representations for word types, typically learned from cooccurrence statistics on unlabeled data, and improve generalization of downstream models across many domains.

Recently, a number of models have been proposed for contextualized word embeddings.

Instead of using a single, fixed vector per word type, these models run a pretrained encoder network over the sentence to produce contextual embeddings of each token.

The encoder, usually an LSTM BID18 or a Transformer BID47 , can be trained on objectives like machine translation BID29 or language modeling BID33 BID19 , for which large amounts of data are available.

The activations of this network-a collection of one vector per token-fit the same interface as conventional word embeddings, and can be used as a drop-in replacement input to any model.

Applied to popular models, this technique has yielded significant improvements to the state-of-the-art on several tasks, including constituency parsing BID22 , semantic role labeling BID44 , and coreference , and has outperformed competing techniques BID8 ) that produce fixed-length representations for entire sentences.

Our goal in this work is to understand where these contextual representations improve over conventional word embeddings.

Recent work has explored many token-level properties of these representations, such as their ability to capture part-of-speech tags BID4 BID3 BID42 , morphology BID2 b) , or word-sense disambiguation BID33 .

BID34 extends this to constituent phrases, and present a heuristic for unsuper- Figure 1 : Probing model architecture ( § 3.1).

All parameters inside the dashed line are fixed, while we train the span pooling and MLP classifiers to extract information from the contextual vectors.

The example shown is for semantic role labeling, where s(1) = [1, 2) corresponds to the predicate ("eat"), while s (2) = [2, 5) is the argument ("strawberry ice cream"), and we predict label A1 as positive and others as negative.

For entity and constituent labeling, only a single span is used.vised pronominal coreference.

We expand on this even further and introduce a suite of edge probing tasks covering a broad range of syntactic, semantic, local, and long-range phenomena.

In particular, we focus on asking what information is encoded at each position, and how well it encodes structural information about that word's role in the sentence.

Is this information primarily syntactic in nature, or do the representations also encode higher-level semantic relationships?

Is this information local, or do the encoders also capture long-range structure?We approach these questions with a probing model (Figure 1 ) that sees only the contextual embeddings from a fixed, pretrained encoder.

The model can access only embeddings within given spans, such as a predicate-argument pair, and must predict properties, such as semantic roles, which typically require whole-sentence context.

We use data derived from traditional structured NLP tasks: tagging, parsing, semantic roles, and coreference.

Common corpora such as OntoNotes BID49 provide a wealth of annotations for well-studied concepts which are both linguistically motivated and known to be useful intermediates for high-level language understanding.

We refer to our technique as "edge probing", as we decompose each structured task into a set of graph edges ( § 2) which we can predict independently using a common classifier architecture ( § 3.1) 2 .

We probe four popular contextual representation models ( § 3.2): CoVe BID29 , ELMo BID33 , OpenAI GPT , and BERT .We focus on these models because their pretrained weights and code are available, since these are most likely to be used by researchers.

We compare to word-level baselines to separate the contribution of context from lexical priors, and experiment with augmented baselines to better understand the role of pretraining and the ability of encoders to capture long-range dependencies.

To carry out our experiments, we define a novel "edge probing" framework motivated by the need for a uniform set of metrics and architectures across tasks.

Our framework is generic, and can be applied to any task that can be represented as a labeled graph anchored to spans in a sentence.

Formulation.

Formally, we represent a sentence as a list of tokens T = [t 0 , t 1 , . . .

, t n ], and a labeled edge as {s(1) , s (2) , L}. We treat DISPLAYFORM0 ) and, optionally, DISPLAYFORM1 ) as (end-exclusive) spans.

For unary edges such as constituent labels, s (2) is omitted.

We take L to be a set of zero or more targets from a task-specific label set L.

The important thing about Disney is that it is a global To cast all tasks into a common classification model, we focus on the labeling versions of each task.

Spans (gold mentions, constituents, predicates, etc.) are given as inputs, and the model is trained to predict L as a multi-label target.

We note that this is only one component of the common pipelined (or end-to-end) approach to these tasks, and that in general our metrics are not comparable to models that jointly perform span identification and labeling.

However, since our focus is on analysis rather than application, the labeling version is a better fit for our goals of isolating individual phenomena of interest, and giving a uniform metric -binary F1 score -across our probing suite.

Our experiments focus on eight core NLP labeling tasks: part-of-speech, constituents, dependencies, named entities, semantic roles, coreference, semantic proto-roles, and relation classification.

The tasks and their respective datasets are described below, and also detailed in TAB1 and Appendix B.Part-of-speech tagging (POS) is the syntactic task of assigning tags such as noun, verb, adjective, etc.

to individual tokens.

We let s 1 = [i, i + 1) be a single token, and seek to predict the POS tag.

Constituent labeling is the more general task concerned with assigning a non-terminal label for a span of tokens within the phrase-structure parse of the sentence: e.g. is the span a noun phrase, a verb phrase, etc.

We let s 1 = [i, j) be a known constituent, and seek to predict the constituent label.

Dependency labeling is similar to constituent labeling, except that rather than aiming to position a span of tokens within the phrase structure, dependency labeling seeks to predict the functional relationships of one token relative to another: e.g. is in a modifier-head relationship, a subjectobject relationship, etc.

We take s 1 = [i, i + 1) to be a single token and s 2 = [j, j + 1) to be its syntactic head, and seek to predict the dependency relation between tokens i and j.

Named entity labeling is the task of predicting the category of an entity referred to by a given span, e.g. does the entity refer to a person, a location, an organization, etc.

We let s 1 = [i, j) represent an entity span and seek to predict the entity type.

Semantic role labeling (SRL) is the task of imposing predicate-argument structure onto a natural language sentence: e.g. given a sentence like "Mary pushed John", SRL is concerned with identifying "Mary" as the pusher and "John" as the pushee.

We let s 1 = [i 1 , j 1 ) represent a known predicate and s 2 = [i 2 , j 2 ) represent a known argument of that predicate, and seek to predict the role that the argument s 2 fills-e.g.

ARG0 (agent, the pusher) vs. ARG1 (patient, the pushee).Coreference is the task of determining whether two spans of tokens ("mentions") refer to the same entity (or event): e.g. in a given context, do "Obama" and "the former president" refer to the same person, or do "New York City" and "there" refer to the same place.

We let s 1 and s 2 represent known mentions, and seek to make a binary prediction of whether they co-refer.

Semantic proto-role (SPR) labeling is the task of annotating fine-grained, non-exclusive semantic attributes, such as change of state or awareness, over predicate-argument pairs.

E.g.given the sentence "Mary pushed John", whereas SRL is concerned with identifying "Mary" as the pusher, SPR is concerned with identifying attributes such as awareness (whether the pusher is aware that they are doing the pushing).

We let s 1 represent a predicate span and s 2 a known argument head, and perform a multi-label classification over potential attributes of the predicateargument relation.

Relation Classification (Rel.) is the task of predicting the real-world relation that holds between two entities, typically given an inventory of symbolic relation types (often from an ontology or database schema).

For example, given a sentence like "Mary is walking to work", relation classification is concerned with linking "Mary" to "work" via the Entity-Destination relation.

We let s 1 and s 2 represent known mentions, and seek to predict the relation type.

We use the annotations in the OntoNotes 5.0 corpus BID49 for five of the above eight tasks: POS tags, constituents, named entities, semantic roles, and coreference.

In all cases, we simply cast the original annotation into our edge probing format.

For POS tagging, we simply extract these labels from the constituency parse data in OntoNotes.

For coreference, since OntoNotes only provides annotations for positive examples (pairs of mentions that corefer) we generate negative examples by generating all pairs of mentions that are not explicitly marked as coreferent.

The OntoNotes corpus does not contain annotations for dependencies, proto-roles, or semantic relations.

Thus, for dependencies, we use the English Web Treebank portion of the Universal Dependencies 2.2 release BID43 .

For SPR, we use two datasets, one (SPR1; BID46 ) derived from Penn Treebank and one (SPR2; BID40 ) derived from English Web Treebank.

For relation classification, we use the SemEval 2010 Task 8 dataset BID17 , which consists of sentences sampled from English web text, labeled with a set of 9 directional relation types.

In addition to the OntoNotes coreference examples, we include an extra "challenge" coreference dataset based on the Winograd schema BID26 .

Winograd schema problems focus on cases of pronoun resolution which are syntactically ambiguous and thus are intended to require subtler semantic inference in order to resolve correctly (see example in TAB1 ).

We use the version of the Definite Pronoun Resolution (DPR) dataset BID39 employed by BID50 , which contains balanced positive and negative pairs.3 EXPERIMENTAL SET-UP

Our probing architecture is illustrated in Figure 1 .

The model is designed to have limited expressive power on its own, as to focus on what information can be extracted from the contextual embeddings.

We take a list of contextual vectors [e 0 , e 1 , . . .

, e n ] and integer spans s DISPLAYFORM0 ) as inputs, and use a projection layer followed by the self-attention pooling operator of BID24 to compute fixed-length span representations.

Pooling is only within the bounds of a span, e.g. the vectors [e i , e i+1 , . . .

, e j−1 ], which means that the only information our model can access about the rest of the sentence is that provided by the contextual embeddings.

The span representations are concatenated and fed into a two-layer MLP followed by a sigmoid output layer.

We train by minimizing binary cross-entropy against the target label set L ∈ {0, 1} |L| .

Our code is implemented in PyTorch BID31 using the AllenNLP BID13 toolkit.

For further details on training, see Appendix C.

We explore four recent contextual encoder models: CoVe, ELMo, OpenAI GPT, and BERT.

Each model takes tokens [t 0 , t 1 , . . .

, t n ] as input and produces a list of contextual vectors [e 0 , e 1 , . . . , e n ].CoVe BID29 uses the top-level activations of a two-layer biLSTM trained on EnglishGerman translation, concatenated with 300-dimensional GloVe vectors.

The source data consists of 7 million sentences from web crawl, news, and government proceedings (WMT 2017; BID5 ).ELMo BID33 ) is a two-layer bidirectional LSTM language model, built over a contextindependent character CNN layer and trained on the Billion Word Benchmark dataset BID6 , consisting primarily of newswire text.

We follow standard usage and take a linear combination of the ELMo layers, using learned task-specific scalars (Equation 1 of BID33 .GPT ) is a 12-layer Transformer BID47 encoder trained as a left-to-right language model on the Toronto Books Corpus .

Departing from the original authors, we do not fine-tune the encoder 3 .BERT ) is a deep Transformer BID47 encoder trained jointly as a masked language model and on next-sentence prediction, trained on the concatenation of the Toronto Books Corpus and English Wikipedia.

As with GPT, we do not finetune the encoder weights.

We probe the publicly released bert-base-uncased (12-layer) and bert-large-uncased (24-layer) models 4 .For BERT and GPT, we compare two methods for yielding contextual vectors for each token: cat where we concatenate the subword embeddings with the activations of the top layer, similar to CoVe, and mix where we take a linear combination of layer activations (including embeddings) using learned task-specific scalars (Equation FORMULA0 The pretrained models expect different tokenizations and input processing.

We use a heuristic alignment algorithm based on byte-level Levenshtein distance, explained in detail in Appendix E, in order to re-map spans from the source data to the tokenization expected by the above models.

Again, we want to answer: What do contextual representations encode that conventional word embeddings do not?

Our experimental comparisons, described below, are intended to ablate various aspects of contextualized encoders in order to illuminate how the model captures different types of linguistic information.

Lexical Baselines.

In order to probe the effect of each contextual encoder, we train a version of our probing model directly on the most closely related context-independent word representations.

This baseline measures the performance that can be achieved from lexical priors alone, without any access to surrounding words.

For CoVe, we compare to the embedding layer of that model, which consists of 300-dimensional GloVe vectors trained on 840 billion tokens of CommonCrawl (web) text.

For ELMo, we use the activations of the context-independent character-CNN layer (layer 0) from the full model.

For GPT and for BERT, we use the learned subword embeddings from the full model.

Randomized ELMo.

Randomized neural networks have recently BID52 shown surprisingly strong performance on many tasks, suggesting that architecture may play a significant role in learning useful feature functions.

To help understand what is actually learned during the encoder pretraining, we compare with a version of the ELMo model in which all weights above the lexical layer (layer 0) are replaced with random orthonormal matrices 6 .

3 We note that there may be information not easily accessible without fine-tuning the LSTM weights.

This can be easily explored within our framework, e.g. using the techniques of BID19 or .

We leave this to future work, and hope that our code release will facilitate such continuations.4 recommend the cased BERT models for named entity recognition tasks; however, we find no difference in performance on our entity labeling variant and so report all results with uncased models.

5 For further details, see Appendix D. 6 This includes both LSTM cell weights and projection matrices between layers.

Non-square matrices are orthogonal along the smaller dimension.

Word-Level CNN.

To what extent do contextual encoders capture long-range dependencies, versus simply modeling local context?

We extend our lexical baseline by introducing a fixed-width convolutional layer on top of the word representations.

As comparing to the lexical baseline factors out word-level priors, comparing to this CNN baseline factors out local relationships, such as the presence of nearby function words, and allows us to see the contribution of long-range context to encoder performance.

To implement this, we replace the projection layer in our probing model with a fully-connected CNN that sees ±1 or ±2 tokens around the center word (i.e. kernel width 3 or 5).

Using the above experimental design, we return to the central questions originally posed.

That is, what types of syntactic and semantic information does each model encode at each position?

And is the information captured primarily local, or do contextualized embeddings encode information about long-range sentential structure?Comparison of representation models.

We report F1 scores for ELMo, CoVe, GPT, and BERT in Table 2 .

We observe that ELMo and GPT (with mix features) have comparable performance, with ELMo slightly better on most tasks but the Transformer scoring higher on relation classification and OntoNotes coreference.

Both models outperform CoVe by a significant margin (6.3 F1 points on average), meaning that the information in their word representations makes it easier to recover details of sentence structure.

It is important to note that while ELMo, CoVe, and the GPT can be applied to the same problems, they differ in architecture, training objective, and both the quantity and genre of training data ( § 3.2).

Furthermore, on all tasks except for Winograd coreference, the lexical representations used by the ELMo and GPT models outperform GloVe vectors (by 5.4 and 2.4 points on average, respectively).

This is particularly pronounced on constituent and semantic role labeling, where the model may be benefiting from better handling of morphology by character-level or subword representations.

We observe that using ELMo-style scalar mixing (mix) instead of concatenation improves performance significantly (1-3 F1 points on average) on both deep Transformer models (BERT and GPT).

We attribute this to the most relevant information being contained in intermediate layers, which agrees with observations by BID4 , BID33 , and , and with the finding of BID34 that top layers may be overly specialized to perform next-word prediction.

When using scalar mixing (mix), we observe that the BERT-base model outperforms GPT, which has a similar 12-layer Transformer architecture, by approximately 2 F1 points on average.

The 24-layer BERT-large model performs better still, besting BERT-base by 1.1 F1 points and ELMo by 2.7 F1 -a nearly 20% relative reduction in error on most tasks.

We find that the improvements of the BERT models are not uniform across tasks.

In particular, BERT-large improves on ELMo by 7.4 F1 points on OntoNotes coreference, more than a 40% reduction in error and nearly as high as the improvement of the ELMo encoder over its lexical baseline.

We also see a large improvement (7.8 F1 points) 7 on Winograd-style coreference from BERT-large in particular, suggesting that deeper unsupervised models may yield further improvement on difficult semantic tasks.

Genre Effects.

Our probing suite is drawn mostly from newswire and web text ( § 2).

This is a good match for the Billion Word Benchmark (BWB) used to train the ELMo model, but a weaker match for the Books Corpus used to train the published GPT model.

To control for this, we train a clone of the GPT model on the BWB, using the code and hyperparameters of .

We find that this model performs only slightly better (+0.15 F1 on average) on our probing suite than the Books Corpus-trained model, but still underperforms ELMo by nearly 1 F1 point.

Encoding of syntactic vs. semantic information.

By comparing to lexical baselines, we can measure how much the contextual information from a particular encoder improves performance on Table 2 : Comparison of representation models and their respective lexical baselines.

Numbers reported are micro-averaged F1 score on respective test sets.

Lex. denotes the lexical baseline ( § 4) for each model, and bold denotes the best performance on each task.

Lines in italics are subsets of the targets from a parent task; these are omitted in the macro average.

SRL numbers consider core and non-core roles, but ignore references and continuations.

Winograd (DPR) results are the average of five runs each using a random sample (without replacement) of 80% of the training data.

95% confidence intervals (normal approximation) are approximately ±3 (±6 with BERT-large) for Winograd, ±1 for SPR1 and SPR2, and ±0.5 or smaller for all other tasks.each task.

Note that in all cases, the contextual representation is strictly more expressive, since it includes access to the lexical representations either by concatenation or by scalar mixing.

We observe that ELMo, CoVe, and GPT all follow a similar trend across our suite (Table 2) , showing the largest gains on tasks which are considered to be largely syntactic, such as dependency and constituent labeling, and smaller gains on tasks which are considered to require more semantic reasoning, such as SPR and Winograd.

We observe small absolute improvements (+6.3 and +3.5 for ELMo Full vs. Lex.) on part-of-speech tagging and entity labeling, but note that this is likely due to the strength of word-level priors on these tasks.

Relative reduction in error is much higher (+66% for Part-of-Speech and +44% for Entities), suggesting that ELMo does encode local type information.

Semantic role labeling benefits greatly from contextual encoders overall, but this is predominantly due to better labeling of core roles (+19.0 F1 for ELMo) which are known to be closely tied to syntax (e.g. BID37 ; BID14 ).

The lexical baseline performs similarly on core and non-core roles (74 and 75 F1 for ELMo), but the more semantically-oriented non-core role labels (such as purpose, cause, or negation) see only a smaller improvement from encoded context (+8.8 F1 for ELMo).

The semantic proto-role labeling task (SPR1, SPR2) looks at the same type of core predicate-argument pairs but tests for higher-level semantic properties ( § 2), which we find to be only weakly captured by the contextual encoder (+1-5 F1 for ELMo).The SemEval relation classification task is designed to require semantic reasoning, but in this case we see a large improvement from contextual encoders, with ELMo improving by 22 F1 points on the lexical baseline (50% relative error reduction) and BERT-large improving by another 4.6 points.

We attribute this partly to the poor performance (51-58 F1) of lexical priors on this task, and to the fact that many easy relations can be resolved simply by observing key words in the sentence (for example, "caused" suggests the presence of a Cause-Effect relation).

To test this, we augment the lexical baseline with a bag-of-words feature, and find that for relation classification we capture more than 70% of the headroom from using the full ELMo model.

Effects of architecture.

Focusing on the ELMo model, we ask: how much of the model's performance can be attributed to the architecture, rather than knowledge from pretraining?

In FIG1 we compare to an orthonormal encoder ( § 4) which is structurally identical to ELMo but contains no information in the recurrent weights.

It can be thought of as a randomized feature function over the sentence, and provides a baseline for how the architecture itself can encode useful contextual information.

We find that the orthonormal encoder improves significantly on the lexical baseline, but that overall the learned weights account for over 70% of the improvements from full ELMo.

Encoding non-local context.

How much information is carried over long distances (several tokens or more) in the sentence?

To estimate this, we extend our lexical baseline with a convolutional layer, which allows the probing classifier to use local context.

In FIG1 we find that adding a CNN of width 3 (±1 token) closes 72% (macro average over tasks) of the gap between the lexical baseline and full ELMo; this extends to 79% if we use a CNN of width 5 (±2 tokens).

On nonterminal constituents, we find that the CNN ±2 model matches ELMo performance, suggesting that while the ELMo encoder propagates a large amount of information about constituents (+15.4 F1 vs. Lex., Table 2 ), most of it is local in nature.

We see a similar trend on the other syntactic tasks, with 80-90% of ELMo performance on dependencies, part-of-speech, and SRL core roles captured by CNN ±2.

Conversely, on more semantic tasks, such as coreference, SRL non-core roles, and SPR, the gap between full ELMo and the CNN baselines is larger.

This suggests that while ELMo does not encode these phenomena as efficiently, the improvements it does bring are largely due to long-range information.

We can test this hypothesis by seeing how our probing model performs with distant spans.

FIG2 shows F1 score as a function of the distance (number of tokens) between a token and its head for the dependency labeling task.

The CNN models and the orthonormal encoder perform best with nearby spans, but fall off rapidly as token distance increases.

The full ELMo model holds up better, with performance dropping only 7 F1 points between d = 0 tokens and d = 8, suggesting the pretrained encoder does encode useful long-distance dependencies.

Recent work has consistently demonstrated the strong empirical performance of contextualized word representations, including CoVe (McCann et al., 2017) , ULMFit BID19 , ELMo BID33 BID44 BID22 .

In response to the impressive results on downstream tasks, a line of work has emerged with the goal of understanding and comparing such pretrained representations.

SentEval BID7 and GLUE BID48 offer suites of application-oriented benchmark tasks, such as sentiment analysis or textual entailment, which combine many types of reasoning and provide valuable aggregate metrics which are indicative of practical performance.

A parallel effort, to which this work contributes, seeks to understand what is driving (or hindering) performance gains by using "probing tasks," i.e. tasks which attempt to isolate specific phenomena for the purpose of finer-grained analysis rather than application, as discussed below.

Much work has focused on probing fixed-length sentence encoders, such as InferSent BID8 , specifically their ability to capture surface properties of sentences such as length, word content, and word order BID0 , as well as a broader set of syntactic features, such as tree depth and tense .

Other related work uses perplexity scores to test whether language models learn to encode properties such as subject-verb agreement BID27 BID15 BID28 BID23 .Often, probing tasks take the form of "challenge sets", or test sets which are generated using templates and/or perturbations of existing test sets in order to isolate particular linguistic phenomena, e.g. compositional reasoning BID10 BID12 .

This approach is exemplified by the recently-released Diverse Natural Language Collection (DNC) BID36 , which introduces a suite of 11 tasks targeting different semantic phenomena.

In the DNC, these tasks are all recast into natural language inference (NLI) format BID50 , i.e. systems must understand the targeted semantic phenomenon in order to make correct inferences about en-tailment.

BID35 used an earlier version of recast NLI to test NMT encoders' ability to understand coreference, SPR, and paraphrastic inference.

Challenge sets which operate on full sentence encodings introduce confounds into the analysis, since sentence representation models must pool word-level representations over the entire sequence.

This makes it difficult to infer whether the relevant information is encoded within the span of interest or rather inferred from diffuse information elsewhere in the sentence.

One strategy to control for this is the use of minimally-differing sentence pairs BID36 BID12 ).

An alternative approach, which we adopt in this paper, is to directly probe the token representations for word-and phrase-level properties.

This approach has been used previously to show that the representations learned by neural machine translation systems encode token-level properties like part-of-speech, semantic tags, and morphology BID42 BID2 b) , as well as pairwise dependency relations BID1 .

BID4 goes further to explore how part-of-speech and hierarchical constituent structure are encoded by different pretraining objectives and at different layers of the model.

BID34 presents similar results for ELMo and architectural variants.

Compared to existing work, we extend sub-sentence probing to a broader range of syntactic and semantic tasks, including long-range and high-level relations such as predicate-argument structure.

Our approach can incorporate existing annotated datasets without the need for templated data generation, and admits fine-grained analysis by label and by metadata such as span distance.

We note that some of the tasks we explore overlap with those included in the DNC, in particular, named entities, SPR and Winograd.

However, our focus on probing token-level representations directly, rather than pooling over the whole sentence, provides a complementary means for analyzing these representations and diagnosing the particular advantages of contextualized vs. conventional word embeddings.

We introduce a suite of "edge probing" tasks designed to probe the sub-sentential structure of contextualized word embeddings.

These tasks are derived from core NLP tasks and encompass a range of syntactic and semantic phenomena.

We use these tasks to explore how contextual embeddings improve on their lexical (context-independent) baselines.

We focus on four recent models for contextualized word embeddings-CoVe, ELMo, OpenAI GPT, and BERT.Based on our analysis, we find evidence suggesting the following trends.

First, in general, contextualized embeddings improve over their non-contextualized counterparts largely on syntactic tasks (e.g. constituent labeling) in comparison to semantic tasks (e.g. coreference), suggesting that these embeddings encode syntax more so than higher-level semantics.

Second, the performance of ELMo cannot be fully explained by a model with access to local context, suggesting that the contextualized representations do encode distant linguistic information, which can help disambiguate longer-range dependency relations and higher-level syntactic structures.

We release our data processing and model code, and hope that this can be a useful tool to facilitate understanding of, and improvements in, contextualized word embedding models.

This version of the paper has been updated to include probing results on the popular BERT model, which was released after our original submission.

Aside from formatting and minor re-wording, the following changes have been made:• We include probing results on the BERT-base and BERT-large models .•

We add one additional task to Table 2 , relation classification on SemEval 2010 Task 8 BID17 , in order to better explore how pre-trained encoders capture semantic information.• We refer to the OpenAI Transformer LM as "GPT" to better reflect common usage.• We add experiments with ELMo-style scalar mixing (Section 3.2) on the OpenAI GPT model.

This improves performance slightly, and changes our conclusion that ELMo was overall superior to GPT; the two are approximately equal on average, with slight differences on some tasks.• To reduce noise, we report the average over five runs for experiments on Winograd coreference (DPR).

We use the self-attentional pooling operator from BID24 and .

This learns a weight z DISPLAYFORM0 for each token, then represents the span as a sum of the vectors e DISPLAYFORM1 Finally, the pooled span representations are fed into a two-layer MLP followed by a sigmoid output layer: DISPLAYFORM2 (s (2) )]) P (label = 1) = σ(W h + b) for = 0, . . .

, |L|We train by minimizing binary cross entropy against the set of true labels.

While convention on many tasks (e.g. SRL) is to use a softmax loss, this enforces an exclusivity constraint.

By using a per-label sigmoid our model can estimate each label independently, which allows us to stratify our analysis (see § 5) to individual labels or groups of labels within a task.

With the exception of ELMo scalars, we hold the weights of the sentence encoder ( § 3.2) fixed while we train our probing classifier.

We train using the Adam optimizer BID20 with a batch size 9 of 32, an initial learning rate of 1e-4, and gradient clipping with max L 2 norm of 5.0.

We evaluate on the validation set every 1000 steps (or every 100 for SPR1, SPR2, and Winograd), halve the learning rate if no improvement is seen in 5 validations, and stop training if no improvement is seen in 20 validations.

CoVe The CoVe model BID29 ) is a two-layer biLSTM trained as the encoder side of a sequence-to-sequence BID45 English-German machine translation model.

We use the original authors' implementation and the best released pre-trained model 10 .

This model is trained on the WMT2017 dataset BID5 which contains approximately 7 million sentences of English text.

Following BID29 , we concatenate the activations of the top-layer forward and backward LSTMs (d = 300 each) with the pre-trained GloVe BID32 embedding 11 (d = 300) of each token, for a total representation dimension of d = 900.ELMo The ELMo model BID33 ) is a two layer LSTM trained as the concatenation of a forward and a backward language model, and built over a context-independent character CNN layer.

We use the original authors' implementation as provided in the AllenNLP BID13 toolkit 12 and the standard pre-trained model trained on the Billion Word Benchmark (BWB) BID6 We take the (fixed, contextual) representation of token i to be the set of three vectors h 0,i , h 1,i , and h 2,i containing the activations of each layer of the ELMo model.

Following Equation 1 of BID33 , we learn task-specific scalar parameters and take a weighted sum:e i = γ (s 0 h 0,i + s 1 h 1,i + s 2 h 2,i ) for i = 0, 1, . . . , nto give 1024-dimensional representations for each token.

OpenAI GPT The GPT model was recently shown to outperform ELMo on a number of downstream tasks, and as of submission holds the highest score on the GLUE benchmark BID48 .

It consists of a 12-layer Transformer BID47 model, trained as a left-to-right language model using masked attention.

We use a PyTorch reimplementation of the alignments as boolean adjacency matricies, we can compose them to form a token-to-token alignment A = UÃV T .We then represent each source span as a boolean vector with 1s inside the span and 0s outside, e.g.[2, 4) = [0, 0, 1, 1, 0, 0, . . .]

∈ {0, 1} m , and project through the alignment A to the target side.

We recover a target-side span from the minimum and maximum nonzero indices.

@highlight

We probe for sentence structure in ELMo and related contextual embedding models. We find existing models efficiently encode syntax and show evidence of long-range dependencies, but only offer small improvements on semantic tasks.

@highlight

Proposes the "edge probing" method and focuses on the relationship between spans rather than individual words, enabling the authors to look at syntactic constituency, dependencies, entity labels, and semantic role labeling.

@highlight

Provides new insights on what is captured contextualized word embeddings by compiling a set of “edge probing” tasks. 