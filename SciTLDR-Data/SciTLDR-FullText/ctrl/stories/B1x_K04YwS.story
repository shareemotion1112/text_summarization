A large number of natural language processing tasks exist to analyze syntax, semantics, and information content of human language.

These seemingly very different tasks are usually solved by specially designed architectures.

In this paper, we provide the simple insight that a great variety of tasks can be represented in a single unified format consisting of labeling spans and relations between spans, thus a single task-independent model can be used across different tasks.

We perform extensive experiments to test this insight on 10 disparate tasks as broad as dependency parsing (syntax), semantic role labeling (semantics), relation extraction (information content), aspect based sentiment analysis (sentiment), and many others, achieving comparable performance as state-of-the-art specialized models.

We further demonstrate benefits in multi-task learning.

We convert these datasets into a unified format to build a benchmark, which provides a holistic testbed for evaluating future models for generalized natural language analysis.

A large number of natural language processing (NLP) tasks exist to analyze various aspects of human language, including syntax (e.g., constituency and dependency parsing), semantics (e.g., semantic role labeling), information content (e.g., named entity recognition and relation extraction), or sentiment (e.g. sentiment analysis).

At first glance, these tasks are seemingly very different in both the structure of their output and the variety of information that they try to capture.

To handle these different characteristics, researchers usually use specially designed neural network architectures.

In this paper we ask the simple questions: are the task-specific architectures really necessary?

Or with the appropriate representational methodology, can we devise a single model that can perform -and achieve state-of-the-art performance on -a large number of natural language analysis tasks?

Interestingly, in the domain of efficient human annotation interfaces, it is already standard to use unified representations for a wide variety of NLP tasks.

On the right we show one example of the annotation interface BRAT (Stenetorp et al., 2012) , which has been used for annotating data for tasks as broad as part-of-speech tagging, named entity recognition, relation extraction, and many others.

Notably, this interface has a single unified format that consists of spans (e.g. the span of an entity), labels on the spans (e.g. the variety of entity such as "person" or "location"), and labeled relations between the spans (e.g. "born-in").

These labeled relations can form a tree or graph structure (e.g., dependency tree), expressing the linguistic structure of sentences.

We detail this BRAT format and how it can be used to represent a wide number of natural language analysis tasks in Section 2.

The simple hypothesis behind our paper is: if humans can perform natural language analysis in a single unified format, then perhaps machines can as well.

Fortunately, there already exist NLP models that perform span prediction and prediction of relations between pairs of spans, such as the end-to-end neural coreference model of .

We extend this model with minor architectural modifications (which are not our core contributions) and pre-trained contextualized (Peters et al., 2018) BERT (Devlin et al., 2019) BERT baseline (Shi & Lin, 2019) SpanBERT (Joshi et al., 2019) Single Table 1 : The unified span-relation model can work on multiple NLP tasks, in contrast to previous works usually designed for a subset of tasks.

representations (e.g., BERT; Devlin et al. (2019) 1 ) then demonstrate the applicability and versatility of this single model on 10 tasks, including named entity recognition (NER), relation extraction (RE), coreference resolution (Coref.), open information extraction (OpenIE), part-of-speech tagging (POS), dependency parsing (Dep.), constituency parsing (Consti.), semantic role labeling (SRL), aspect based sentiment analysis (ABSA), and opinion role labeling (ORL).

While previous work has used similar formalisms to understand the representations learned by pre-trained embeddings (Tenney et al., 2019a; b) , to the best of our knowledge this is the first work that uses such a unified model to actually perform analysis.

Moreover, despite it simplicity we demonstrate that such a model can achieve comparable performance with special-purpose state-of-the-art models on the tasks above (Table 1 ).

We also demonstrate that this framework allows us to easily perform multi-task learning among different tasks, leading to improvements when there are related tasks to be learned from or data is sparse.

In summary, our contributions are:

• We provide the simple insight that a great variety of natural language analysis tasks can be represented and solved in a single unified format, i.e., span-relation representations.

This insight may seem obvious in hindsight, but it has not been examined, particularly to this scale, by previous work on model-building for NLP.

• We perform extensive experiments to test this insight on 10 disparate tasks, achieving comparable empirical results as the state-of-the-art, using a single task-independent modeling framework.

• We further use this framework to perform an analysis of the benefits from multi-task learning across all of the tasks above, gleaning various insights about task relatedness and how multi-task learning performs with different token representations.

• Upon acceptance of the paper, we will release our General Language Analysis Datasets (GLAD) benchmark with 8 datasets covering 10 tasks in the BRAT format, and provide a leaderboard to facilitate future work on generalized models for NLP.

Compared to the full sentence-level tasks in the GLUE leaderboard (Wang et al., 2019a ;b), we cover a wide variety of natural language analysis tasks that require analyzing of the finer grained text units (e.g., words, phrases, clauses).

In this section, we explain how the BRAT format can be used to represent a large number of tasks.

indices respectively, and L is a set of span labels.

A relation annotation (s j , s k , r jk ) refers to a relation r jk (r jk ∈ R) between the head span s j and the tail span s k , where R is a set of relation types.

This span-relation representation can easily express many tasks by defining different L and R, as summarized in Table 2a and Table 2b .

These tasks fall in two categories: span-oriented tasks, where the goal is to predict labeled spans (e.g., named entities in NER) and relation-oriented tasks, where the goal is to predict relations between two spans (e.g., relation between two entities in RE).

• Span-oriented Tasks (Table 2a) -Named Entity Recognition (Sang & Meulder, 2003) NER is traditionally considered as a sequence labeling task.

We model named entities as spans over one or more tokens.

-Constituency Parsing (Collins, 1997) Constituency parsing aims to produce a syntactic parse tree for each sentence.

Each node in the tree is an individual span, and spans are nested.

-Part-of-speech Tagging (Ratnaparkhi, 1996; Toutanova et al., 2003) POS tagging is another sequence labeling task, where every single token is an individual span with a POS tag.

-Aspect-based Sentiment Analysis (Pontiki et al., 2014) ABSA is a task that consists of identifying certain spans as aspect terms and predicting their associated sentiments.

• Relation-oriented Tasks (Table 2b) -Relation Extraction (Hendrickx et al., 2010) RE concerns the relation between two entities.

-Coreference (Pradhan et al., 2012) Coreference resolution is to link named, nominal, and pronominal mentions that refer to the same concept, within or beyond a single sentence.

-Semantic Role Labeling (Gildea & Jurafsky, 2002 ) SRL aims to identify arguments of a predicate (verb or noun) and classify them with semantic roles in relation to the predicate.

-Open Information Extraction (Banko et al., 2007; Niklaus et al., 2018) In contrast to the fixed relation types in RE, OpenIE aims to extract open-domain predicates and their arguments (usually subjects and objects) from a sentence.

-Dependency Parsing (Kübler et al., 2009 ) Spans are single-word tokens and a relation links a word to its syntactic parent with the corresponding dependency type.

-Opinion Role Labeling (Yang & Cardie, 2013 ) ORL detects spans that are opinion expressions, as well as holders and targets related to these opinions.

While the tasks above represent a remarkably broad swath of NLP, it is worth mentioning what we have not covered, to properly scope the work.

Notably, sentence-level tasks such as text classification and natural language inference are not covered, although they can also be formulated using this span-relation representation by treating the entire sentence as a span.

We chose to omit these tasks because they are already well-represented by previous work on generalized architectures (Lan & Xu, 2018) and multi-task learning (Devlin et al., 2019; , and thus we mainly focus on tasks using phrase-like spans.

In addition, the span-relation representations described here are designed for natural language analysis, and cannot handle tasks that require generation of text, such as machine translation (Bojar et al., 2014) , dialog response generation (Lowe et al., 2015) , and summarization (Nallapati et al., 2016) .

There are also a small number of analysis tasks such as semantic parsing to logical forms (Banarescu et al., 2013) where the outputs are not directly associated with spans in the input, and handling these tasks is beyond the scope of this work.

Now that it is clear that a very large number of analysis tasks can be formulated in a single format, we turn to devising a single model that can solve these tasks.

We base our model on a span-based model first designed for end-to-end coreference resolution , which is then adapted for other tasks Luan et al., 2018; Dixit & Al-Onaizan, 2019; Zhang & Zhao, 2019) .

At the core of the model is a module to represent each span as a fixed-length vector, which is used to predict labels for spans or span pairs.

We first briefly describe the span representation used and proven to be effective in previous works, then highlight some details we introduce to make this model generalize to a wide variety of tasks.

Span Representation Given a sentence x = [w 1 , w 2 , ..., w n ] of n tokens, a span s i = [w bi , w bi+1 , ..., w ei ] is represented by concatenating two components: a content representation z c i calculated as the weighted average across all token embeddings in the span, and a boundary representation z u i that concatenates the embeddings at the start/end positions of the span.

Specifically,

where TokenRepr could be non-contextualized embeddings, such as GloVe (Pennington et al., 2014) , or contextualized embeddings, such as ELMo (Peters et al., 2018) , BERT (Devlin et al., 2019) , and SpanBERT (Joshi et al., 2019) .

We refer to for further details.

Since we extract spans and relations in an end-to-end fashion, we introduce two additional labels NEG SPAN and NEG REL in L and R respectively.

NEG SPAN indicates invalid spans (e.g., spans that are not named entities in NER) and NEG REL indicates invalid span pairs without any relation between them (i.e., no relation exists between two arguments in SRL).

We first predict labels for all spans up to a length of l words using a multilayer perceptron (MLP): softmax(MLP span (z i )) ∈ ∆ |L| , where ∆ |L| is a |L|-dimensional simplex.

Then we keep the top K = τ · n spans with the lowest NEG SPAN probabilities in relation prediction for efficiency, where smaller pruning threshold τ indicates more aggressive pruning.

Another MLP is applied to pairs of the remaining spans to produce their relation score:

Application to Disparate Tasks For most of the tasks, we can simply maximize the probability of the ground truth relation for all pairs of the remaining spans.

However, some tasks might have different requirements, e.g., coreference resolution aims to cluster spans referring to the same concept and we do not care about which antecedent a span is linked to if there are multiple ones.

To accommodate different requirements, we provide two training loss functions:

1.

Pairwise Maximize the probabilities of the ground truth relations for all pairs of the remaining spans independently: softmax(o jk )

r jk , where r jk indexes the ground truth relation.

2.

Head Maximize the probability of selecting the ground truth head spans for a specific span

where head(·) returns indices of one or multiple heads and o j· is the corresponding scalar from o j· indicating how likely two spans are related.

We use option 1 for all tasks except for coreference resolution which uses option 2.

Note that the above loss functions only differ in how relation scores are normalized and the other parts of the model remain the same across different tasks.

At test time, we follow previous inference methods to generate valid outputs.

For coreference resolution, we link a span to the antecedent with highest score and build clusters .

For constituency parsing, we use the greedy top-down 2 The time complexity of span prediction is O(l · n) for a sentence of n tokens, and the time complexity of relation prediction is O(K 2 ) = O(τ 2 · n 2 ).

Another option for span prediction is to formulate it as a sequence labeling task, as did in previous works on SRL and many others (Lample et al., 2016; Stanovsky et al., 2018) , and their time complexity is O(n).

Although slower than token-based labeling models, span-based models offer the advantages of being able to model overlapping spans (e.g., overlapping arguments in SRL) and exploring span-level information for span prediction.

• Previous works use gold standard spans in these evaluations.

† We use standard bracket scoring program Evalb (Collins, 1997) in constituency parsing.

decoding (Stern et al., 2017) to generate a valid parse tree.

For dependency parsing, each word is linked to exactly one parent with the highest relation probability.

For other tasks, we predict relations for all span pairs and use those not predicted as NEG REL to construct outputs.

Our core insight is that the above formulation is largely task-agnostic, meaning that a task can be modeled in this framework as long as it can be formulated as a span-relation prediction problem with properly defined span labels L and relation labels R. As shown in Table 1 , this unified SpanRelation (SpanRel) model makes it simple to scale to a large number of language analysis tasks, with breadth far beyond that of previous work.

The SpanRel model makes it easy to perform multi-task learning (MTL) by sharing all parameters except for the MLPs used for label prediction.

With shared span representations, different tasks can learn from each other.

However, because different tasks capture different linguistic aspects, they are not equally beneficial to each other.

It is expected that jointly training on related tasks is helpful, while forcing the same model to solve unrelated tasks might even hurt the performance (Ruder, 2017) .

Compared to manually choosing source tasks based on prior knowledge, which might be sub-optimal when the number of tasks is large, SpanRel offers a systematic way to examine relative benefits of source-target task pairs, as we will show in Section 4.3.

We first describe our General Language Analysis Datasets (GLAD) benchmark and evaluation metrics, then conduct experiments to (1) verify that SpanRel can achieve comparable performance across all tasks (Section 4.2), and (2) demonstrate its benefits in multi-task learning (Section 4.3).

Table 3 , we convert 8 widely used datasets with annotations of 10 tasks into the BRAT format and include them in the GLAD benchmark.

It covers diverse domains, spans, and relations, and provides a holistic testbed for natural language analysis evaluation.

The major evaluation metric is span-based F 1 (denoted as F 1 unless otherwise noted), a standard metric for SRL.

Precision is the proportion of extracted spans (spans not predicted as NEG SPAN) that are consistent with the ground truth.

Recall is the proportion of ground truth spans that are correctly extracted.

F 1 is their harmonic mean.

Span F 1 is also applicable to the case of relations, where an extracted relation (relations not predicted as NEG REL) is correct iff both head and tail spans have correct boundaries and the predicted relation label is correct.

To make fair comparisons with existing works, we also compute standard metrics for different tasks, as listed in Table 3 .

We refer to the corresponding papers for details.

& Ba, 2015) with learning rate of 1e-3 and early stop with patience of 3.

For BERT and SpanBERT, we follow standard fine-tuning and use Adam with learning rate of 5e-5, β 1 = 0.9, β 2 = 0.999, L2 weight decay of 0.01, warmup over the first 10% steps, and number of epochs tuned on development set.

Task-specific hyperparameters maximal span length and pruning ratio are tuned on development set and listed in Appendix B.

We compare the SpanRel model with state-of-the-art task-specific models by training on data from a single task.

By doing so we attempt to answer the research question "can a single model with minimal task-specific engineering achieve competitive or superior performance to other models that have been specifically engineered?" We select competitive SOTA models mainly based on settings, e.g., single-task learning and end-to-end extraction of spans and relations.

To make fair comparisons, token embeddings (GloVe, ELMo, BERT) and other hyperparameters (e.g., the number of antecedents in Coref.

and the maximal span length in SRL) in our method are set to match those used by SOTA models, to focus on differences brought about by the model architecture.

As shown in Table 4 , the SpanRel model achieves comparable performances as the task-specific SOTA methods (regardless of whether the token representation is contextualized or not).

This indicates that the span-relation format can generically represent a large number of natural language analysis tasks and it is possible to devise a single unified model that can achieves strong performance on all of them.

It provides a strong and generic baseline for natural language analysis tasks and a way to examine the usefulness of task-specific designs.

To demonstrate the benefit of the SpanRel model in MTL, we perform single-task learning (STL) and MTL across all tasks using end-to-end settings.

4 Following , we perform MTL+fine-tuning and show the results in separate columns of Table 5 .

Contextualized token representations yield significantly better results than GloVe on all tasks, indicating that pre-training on large corpora 3 • The small version of 's method with 100 antencedents and no speaker features.

For OpenIE and ORL, we use span-based F1 instead of syntactic-head-based F1 and binary coverage F1 used in the original papers because they are biased towards extracting long spans.

† For SRL, we choose to compare with because they also extract predicates and arguments in an end-to-end way.

We follow Xu et al. (2019) to report accuracy of restaurant and laptop domain separately in ABSA.

4 Span-based F1 is used as the evaluation metric in SemEval-2010 Task 8 and SemEval-2014 Task 4 as opposed to macro F1 and accuracy reported in the original papers because we aim at end-to-end extractions.

Table 5 : Comparison between STL and MTL+fine-tuning of the SpanRel model across all tasks.

blue↑ indicates results better than STL, red↓ indicates worse, and black means almost the same (i.e., a difference within 0.5).

Constituency parsing requires more memory than other tasks so we restrict its maximal span length as 10 in MTL, thus it cannot form a valid tree.

is almost universally helpful to NLP tasks.

Comparing the results of MTL+fine-tuning with STL, we found that performance with GloVe drops on 8 out of 15 tasks, most of which are tasks with relatively sparse data.

It is probably because the capacity of the GloVe-based model is too small to store all the patterns required by different tasks.

The results of contextualized representations are mixed, with some tasks being improved and others remaining the same or degrading.

We hypothesize that this is because different tasks capture different linguistic aspects, thus are not equally helpful to each other.

Reconciling these seemingly different tasks in the same model might be harmful to some tasks.

Notably, as the contextualized representations become stronger, the performance of MTL+FT becomes more favorable.

5 out of 15 tasks (NER, RE, OpenIE, SRL, ORL) observe improvements with SpanBERT, a contextualized embedding pre-trained with span-based training objectives, while only one task degrades (ABSA), indicating its superiority in reconciling spans from different tasks.

The GLAD benchmark provides a holistic testbed for evaluating natural language analysis capability.

Task Relatedness Analysis To further investigate how different tasks interact with each other, we choose five source tasks (POS, NER, Consti., Dep., and SRL) that have been widely used in MTL (Hashimoto et al., 2017; Strubell et al., 2018) and six target tasks (OpenIE, NER, RE, ABSA, ORL, and SRL) to perform pairwise multi-task learning.

We hypothesize that although language modeling pre-training is theoretically orthogonal to MTL (Swayamdipta et al., 2018) , in practice the benefit of pre-training tends to overlap or even overshadow the benefit of MTL.

To analyze these two factors separately, we start with a weak representation GloVe to study task relatedness, then move to BERT to demonstrate how much we can still improve with MTL given strong and contextualized representations.

As shown in Table 6 (GloVe), tasks are not equally useful to each other.

Notably, (1) for OpenIE and ORL, multi-task learning with SRL improves the performance significantly, while other tasks lead to less or no improvements.

(2) Dependency parsing and SRL are generic source tasks that are beneficial to most of the target tasks.

(3) ABSA is quite different from the source tasks and no improvement is observed with MTL.

This unified SpanRel makes it easy to perform MTL and decide beneficial source tasks.

We analyze how token representations and sizes of the target dataset affect the performance of MTL.

Comparing BERT and GloVe in Table 6 , the improvements of MTL become smaller or vanish as the token representation becomes stronger, e.g., improvement on OpenIE with SRL reduces from 5.8 to 1.6 and improvement on SRL with Dep. reduces from 2.5 to 1.1.

This is expected because both large-scale pre-training and MTL aim to learn general representations and their benefits tend to overlap in practice.

Interestingly, some helpful source tasks even become harmful when we shift from GloVe to BERT, such as OpenIE paired with Consti.

or POS.

We conjecture that the gains of MTL might have already been achieved by BERT, but the taskspecific characteristics of Consti.

and POS hurt the performance of OpenIE.

MTL shrink as we increase the size of the SRL datasets, as shown in Figure 2 , indicating that MTL is more useful when the target data is sparse.

General Architectures for NLP There has been a rising interest in developing general architectures for different NLP tasks, with the most prominent examples being sequence labeling framework (Collobert et al., 2011; Ma & Hovy, 2016) used for tagging tasks (e.g., NER, POS) and sequenceto-sequence framework used for generation tasks (e.g., machine translation).

Moreover, researchers typically pick related tasks, motivated by either linguistic insights or empirical results, and create a general framework to perform MTL, several of which are summarized in Table 1 .

For example, based on the belief that semantic structure of a sentence should conform with syntactic structure, Swayamdipta et al. (2018) and Strubell et al. (2018) use constituency and dependency parsing to improve SRL.

Luan et al. (2018; ) use a span-based model to jointly solve three information-extraction-related tasks (NER, RE, and Coref.).

Compared to existing works, we aim to create an output representation that can solve nearly every natural language analysis task in one fell swoop, allowing us to cover a far broader range of tasks with a single model.

In addition, NLP has seen a recent burgeoning of contextualized token embeddings pre-trained on large corpra (e.g., ELMo (Peters et al., 2018) and BERT (Devlin et al., 2019) ).

These methods focus on learning generic input representations, but are agnostic to the output representation, requiring different predictors to be designed for different tasks.

In contrast, we present a methodology to formulate the output of different tasks in a unified format.

Thus our work is orthogonal to those on contextualized embeddings.

Indeed, in Section 4.3, we demonstrate that the SpanRel model can benefit from stronger contextualized representation models, and even provide a testbed for their use in natural language analysis.

Benchmarks for Evaluating Natural Language Understanding Due to the rapid development of NLP models, large-scale benchmarks, such as SentEval (Conneau & Kiela, 2018) , GLUE , and SuperGLUE (Wang et al., 2019a) have been proposed to facilitate fast and holistic evaluation of models' natural language understanding ability.

They mainly focus on sentence-level tasks, such as text classification and natural language inference, while our GLAD benchmark focuses on token/phrase-level analysis tasks with diverse coverage of different linguistic structures.

With different tasks represented under the same format, a model can be easily evaluated on all our tasks, reflecting various aspects of its natural language analysis capability.

New tasks and datasets can be conveniently added to our benchmark as long as they are in the BRAT standoff format, which is one of the most commonly used data format in the NLP community, e.g., it has been used in the BioNLP shared tasks (Kim et al., 2009 ) and the Universal Dependency project (McDonald et al., 2013) .

We provide the simple insight that a large number of natural language analysis tasks can be represented in a single format consisting of spans and relations between spans.

As a result, these tasks can be solved in a single modeling framework that first extracts spans and predicts their labels, then predicts relations between extracted spans.

We attempted 10 tasks with this SpanRel model under this unified representation and show that this generic task-independent model can achieve competitive performance as state-of-the-art methods tailored for each tasks.

We merge 8 datasets into our GLAD benchmark for evaluating future models for natural language analysis.

Table 7 : Single-task learning performance of the SpanRel model with different token representations.

BERT large requires a large amount of memory so we cannot feed the entire document to the model in coreference resolution.

As shown in Table 8 , a larger maximum span length is used for tasks with longer spans (e.g., OpenIE), and a larger pruning ratio is used for tasks with more spans (e.g., SRL).

Constituency parsing does not have span length limit because spans can be as long as the entire sentence.

Since relation extraction aims to extract exactly two entities and their relation from a sentence, we keep pruning ratio fixed (top 5 spans in this case) regardless of the length of the sentence.

Table 8 : Task-specific hyperparameters.

Span-oriented tasks do not need pruning ratio.

<|TLDR|>

@highlight

We use a single model to solve a great variety of natural language analysis tasks by formulating them in a unified span-relation format.

@highlight

This paper generalizes a wide range of natural language processing tasks as a single span-based framework and proposes a general architecture to solve all these problems.

@highlight

This work presents a unified formulation of various phrase and token level NLP tasks.