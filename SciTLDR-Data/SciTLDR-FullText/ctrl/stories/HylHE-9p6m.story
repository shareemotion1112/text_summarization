Fine-grained Entity Recognition (FgER) is the task of detecting and classifying entity mentions to a large set of types spanning diverse domains such as biomedical, finance and sports.

We  observe  that  when  the  type  set  spans  several  domains,  detection  of  entity mention becomes a limitation for supervised learning models.

The primary reason being lack  of  dataset  where  entity  boundaries  are  properly  annotated  while  covering  a  large spectrum of entity types.

Our work directly addresses this issue.

We propose Heuristics Allied with Distant Supervision (HAnDS) framework to automatically construct a quality dataset suitable for the FgER task.

HAnDS framework exploits the high interlink among Wikipedia  and  Freebase  in  a  pipelined  manner,  reducing  annotation  errors  introduced by naively using distant supervision approach.

Using HAnDS framework,  we create two datasets, one suitable for building FgER systems recognizing up to 118 entity types based on the FIGER type hierarchy and another for up to 1115 entity types based on the TypeNet hierarchy.

Our extensive empirical experimentation warrants the quality of the generated datasets.

Along with this, we also provide a manually annotated dataset for benchmarking FgER systems.

In the literature, the problem of recognizing a handful of coarse-grained types such as person, location and organization has been extensively studied BID18 Sekine, 2007, Marrero et al., 2013] .

We term this as Coarse-grained Entity Recognition (CgER) task.

For CgER, there exist several datasets, including manually annotated datasets such as CoNLL BID28 ] and automatically generated datasets such as WP2 BID21 .

Manually constructing a dataset for FgER task is an expensive and time-consuming process as an entity mention could be assigned multiple types from a set of thousands of types.

In recent years, one of the subproblems of FgER, the Fine Entity Categorization or Typing (Fine-ET) problem has received lots of attention particularly in expanding its type coverage from a handful of coarse-grained types to thousands of fine-grained types BID17 BID6 .

The primary driver for this rapid expansion is exploitation of cheap but fairly accurate annotations from Wikipedia and Freebase BID4 via the distant supervision process BID7 .

The Fine-ET problem assumes that the entity boundaries are provided by an oracle.

We observe that the detection of entity mentions at the granularity of Fine-ET is a bottleneck.

The existing FgER systems, such as FIGER BID12 , follow a two-step approach in which the first step is to detect entity mentions and the second step is to categorize detected entity mentions.

For the entity detection, it is assumed that all the fine-categories are subtypes of the following four categories: person, location, organization and miscellaneous.

Thus, a model trained on the CoNLL dataset BID28 ] which is annotated with these types can be used for entity detection.

Our analysis indicates that in the context of FgER, this assumption is not a valid assumption.

As a face value, the miscellaneous type should ideally cover all entity types other than person, location, and organization.

However, it only covers 68% of the remaining types of the FIGER hierarchy and 42% of the TypeNet hierarchy.

Thus, the models trained using CoNLL data are highly likely to miss a significant portion of entity mentions relevant to automatic knowledge bases construction applications.

Our work bridges this gap between entity detection and Fine-ET.

We propose to automatically construct a quality dataset suitable for the FgER, i.e, both Fine-ED and Fine-ET using the proposed HAnDS framework.

HAnDS is a three-stage pipelined framework wherein each stage different heuristics are used to combat the errors introduced via naively using distant supervision paradigm, including but not limited to the presence of large false negatives.

The heuristics are data-driven and use information provided by hyperlinks, alternate names of entities, and orthographic and morphological features of words.

Using the HAnDS framework and the two popular type hierarchies available for Fine-ET, the FIGER type hierarchy BID12 and TypeNet BID17 , we automatically generated two corpora suitable for the FgER task.

The first corpus contains around 38 million entity mentions annotated with 118 entity types.

The second corpus contains around 46 million entity mentions annotated with 1115 entity types.

Our extensive intrinsic and extrinsic evaluation of the generated datasets warrants its quality.

As compared with existing automatically generated datasets, supervised learning models trained on our induced training datasets perform significantly better (approx 20 point improvement on micro-F1 score).

Along with the automatically generated dataset, we provide a manually annotated corpora of around thousand sentences annotated with 117 entity types for benchmarking of FgER models.

Our contributions are highlighted as follows:• We analyzed that existing practice of using models trained on CoNLL dataset has poor recall for entity detection in the Fine-ET setting, where the type set spans several diverse domains. (Section 3)• We propose HAnDS framework, a heuristics allied with the distant supervision approach to automatically construct datasets suitable for FgER problem, i.e., both fine entity detection and fine entity typing. (Section 4)• We establish the state-of-the-art baselines on our new manually annotated corpus, which covers 2.7 times more finer-entity types than the FIGER gold corpus, the current de facto FgER evaluation corpus. (Section 5)The rest of the paper is organized as follows.

We describe the related work in Section 2, followed by a case study on entity detection problem in the Fine-ET setting, in Section 3.

Section 4 describes our proposed HAnDS framework, followed by empirical evaluation of the datasets in Section 5.

In Section 6 we conclude our work.

We majorly divide the related work into two parts.

First, we describe work related to the automatic dataset construction in the context of the entity recognition task followed by related work on noise reduction techniques in the context of automatic dataset construction task.

In the context of FgER task, BID12 proposed to use distant supervision paradigm BID2 ] to automatically generate a dataset for the Fine-ET problem, which is a sub-problem of FgER.

We term this as a Naive Distant Supervision (NDS) approach.

In NDS, the linkage between Wikipedia and Freebase is exploited.

If there is a hyperlink in a Wikipedia sentence, and that hyperlink is assigned to an entity present in Freebase, then the hyperlinked text is an entity mention whose types are obtained from Freebase.

However, this process can only generate positive annotations, i.e., if an entity mention is not hyperlinked, no types will be assigned to that entity mention.

The positive only annotations are suitable for Fine-ET problem but it is not suitable for learning entity detection models as there are large number of false negatives (Section 3).

This dataset is publicly available as FIGER dataset, along with a manually annotated evaluation corpra.

The NDS approach is also used to generate datasets for some variants of the Fine-ET problem such as the Corpus level Fine-Entity typing BID33 and FineEntity typing utilizing knowledge base embeddings BID31 .

Much recently, BID6 ] generated an entity typing dataset with a very large type set of size 10k using head words as a source of distant supervision as well as using crowdsourcing.

In the context of CgER task, BID19 BID20 BID21 proposed an approach to create a training dataset for CgER task using a combination of bootstrapping process and heuristics.

The bootstrapping was used to classify a Wikipedia article into five categories, namely PER, LOC, ORG, MISC and NON-ENTITY.

The bootstrapping requires initial manually annotated seed examples for each type, which limits its scalability to thousands of types.

The heuristics were used to infer additional links in un-linked text, however the proposed heuristics limit the scope of entity and non-entity mentions.

For example, one of the heuristics used mostly restricts entity mentions to have at least one character capitalized.

This assumption is not true in the context for FgER where entity mentions are from several diverse domains including biomedical domain.

There are other notable work which combines NDS with heuristics for generating entity recognition training dataset, such as BID1 and BID9 .

However, their scope is limited to the application of CgER.

Our work revisits the idea of automatic corpus construction in the context of FgER.

In HAnDS framework, our main contribution is to design data-driven heuristics which are generic enough to work for over thousands of diverse entity types while maintaining a good annotation quality.

An automatic dataset construction process involving heuristics and distant supervision will inevitably introduce noise and its characteristics depend on the dataset construction task.

In the context of the Fine-ET task BID24 BID10 , the dominant noise in false positives.

Whereas, for the relation extraction task both false negatives and false positives noise is present BID25 BID22 .

In this section, we systematically analyzed existing entity detection systems in the setting of Fine-Entity Typing.

Our aim is to answer the following question : How good are entity detection systems when it comes to detecting entity mentions belonging to a large set of diverse types?

We performed two analysis.

The first analysis is about the type coverage of entity detection systems and the second analysis is about actual performance of entity detection systems on two manually annotated FgER datasets.3.1 Is the Fine-ET type set an expansion of the extensively researched coarse-grained types?For this analysis we manually inspected the most commonly used CgER dataset, CoNLL 2003.

We analyzed how many entity types in the two popular Fine-ET hierarchies, FIGER and TypeNet are actual descendent of the four coarse-types present in the CoNLL dataset, namely person, location, organization and miscellaneous.

The results are available in FIG0 .

We can observe that in the FIGER typeset, 14% of types are not a descendants of the CoNLL types.

This share increases in TypeNet where 25% of types are not descendants Table 1 : Performance of entity detection models trained on existing datasets evaluated on the FIGER and 1k-WFB-g datasets.of CoNLL types.

These types are from various diverse domain, including bio-medical, legal processes and entertainment and it is important in the aspect of the knowledge base construction applications to detect entity mentions of these types.

These differences can be attributed to the fact that since 2003, the entity recognition problem has evolved a lot both in going towards finer-categorization as well as capturing entities from diverse domains.

For this analysis we evaluate two publicly available state-of-the-art entity detection systems, the Stanford CoreNLP BID15 and the NER Tagger system proposed in BID11 .

Along with these, we also train a LSTM-CNN-CRF based sequence labeling model proposed in BID13 on the FIGER dataset.

The learning models were evaluated on a manually annotated FIGER corpus and 1k-WFB-g corpus, a new in-house developed corpus specifically for FgER model evaluations.

The results are presented in Table 1 .From the results, we can observe that a state-of-the-art sequence labeling model, LSTM-CNN-CRF trained on a dataset generated using NDS approach, such as FIGER dataset has lower recall compared with precision.

On average the recall is 58% lower than precision.

This is primarily because the NDS approach generates positive only annotations and the remaining un-annotated tokens contains large number of entity mentions.

Thus the resulting dataset has large false negatives.

On the other hand, learning models trained on CoNLL dataset (CoreNLP and NER Tagger), have a much more balanced performance in precision and recall.

This is because, being a manually annotated dataset, it is less likely that any entity mention (according to the annotation guidelines) will remain un-annotated.

However, the recall is much lower (16% lower) on the 1k-WFB-g corpus as on the FIGER corpus.

This is because, when designing 1k-WFB-g we insured that it has sufficient examples covering 117 entity types.

Whereas, the FIGER evaluation corpus has only has 42 types of entity mentions and 80% of mentions are from person, location and organization coarse types.

These results also highlight the coverage issue, mentioned in section 3.1.

When the evaluation set is balanced covering a large spectrum of entity types, the performance of models trained on the CoNLL dataset goes down because of presence out-of-scope entity types.

An ideal entity detection system should be able to work on the traditional as well as other entities relevant to FgER problem, i.e., good performance across all types.

A statistical comparison of FIGER and 1k-WFB-g corpus is provided in Table 2 .The use of CoreNLP or learning models trained on CoNLL dataset is a standard practice to detect entity mentions in existing FgER research BID12 .

Our analysis conveys that this practice has its limitation in terms of detecting entities which are out of the scope of the CoNLL dataset.

In the next section, we will describe our approach of automatically creating a training dataset for the FgER task.

The same learning models, when trained on our generated training datasets will have a better and a balanced precision and recall.

The objective of the HAnDS framework is to automatically create a corpus of sentences where every entity mention is correctly detected and is being characterized into one or more entity types.

The scope of entities, i.e., what types of entities should be annotated is decided by a type hierarchy, which is one of the inputs of the framework.

FIG2 gives an overview of the HAnDS framework.

The framework requires three inputs, a linked text corpus, a knowledge base and a type hierarchy.

Linked text corpus: A linked text corpus is a collection of documents where sporadically important concepts are hyperlinked to another document.

For example, Wikipedia is a large-scale multi-lingual linked text corpus.

The framework considers the span of hyperlinked text (or anchor text) as potential candidates for entity mentions.

Knowledge base: A knowledge base (KB) captures concepts, their properties, and interconcept properties.

Freebase, WikiData BID29 and UMLS BID3 are examples of popular knowledge bases.

A KB usually has a type property where multiple fine-grained semantic types/labels are assigned to each concept.

Type hierarchy: A type hierarchy (T ) is a hierarchical organization of various entity types.

For example, an entity type city is a descendant of type geopolitical entity.

There have been various hierarchical organization schemes of fine-grained entity types proposed in literature, which includes, a 200 type scheme proposed in BID26 , a 113 type scheme proposed in BID12 , a 87 type scheme proposed in BID10 and a 1081 type scheme proposed in BID17 .

However, in our work, we use two such hierarchies, FIGER 2 and TypeNet.

FIGER being the most extensively used hierarchy and TypeNet being the latest and largest entity type hierarchy.

Automatic corpora creation using distant supervised methods inevitably will contain errors.

For example, in the context of FgER, the errors could be at annotating entity boundaries, i.e, entity detection errors, or assigning an incorrect type, i.e., entity linking errors or both.

The three-step process in our proposed HAnDS framework tries to reduce these errors.

The objective of this stage is to reduce false positives entity mentions, where an incorrect anchor text is detected as an entity mention.

To do so, we first categorize all hyperlinks of the document being processed as entity links and non-entity links.

Further, every link is assigned a tag of being a referential link or not.

Entity links: These are a subset of links whose anchor text represents candidate entity mentions.

If the labels obtained by a KB for a link, belongs to T , we categorize that link as an entity link.

Here, the T decides the scope of entities in the generated dataset.

For example, if T is the FIGER type hierarchy, then the hyperlink photovoltaic cell is not an entity link as its labels obtained by Freebase is not present in T .

However, if T is the TypeNet hierarchy, then photovoltaic cell is an entity link of type invention.

Non-entity links: These are a subset of links whose anchor text does not represent an entity mention.

Since knowledge bases are incomplete, if a link is not categorized as an entity link it does not mean that the link will not represent an entity.

We exploit corpus level context to categorize a link as a non-entity link using the following criteria: across complete corpus, the link should be mentioned at least 50 times (support threshold) and at least 50% of times (confidence threshold) with a lowercase anchor text.

The intuition of this criteria is that we want to be certain that a link actually represents a non-entity.

For example, this heuristic categorizes RBI as a non-entity link as there is no label present for this link in Freebase.

Here RBI refers to the term "run batted in", frequently used in the context of baseball and softball.

Unlike, BID19 which discards non entity mentions to have capitalized word, our data-driven heuristics does not put any hard constraints.

Referential links: A link is said to be referential if its anchor text has a direct caseinsensitive match with the list of allowed candidate names for the linked concept.

A KB can provide such list.

For example, for an entity Bill Gates, the candidate names provided by Freebase includes Gates and William Henry Gates.

However, in Wikipedia, there exists hyperlinks such as Bill and Melinda Gates linking to Bill Gates page, which is erroneous as the hyperlinked text is not the correct referent of the entity Bill Gates.

After categorization of links, except for referential entity links, we unlink all other links.

Unlinking non-referential links such as Bill and Melinda Gates reduce entity detection errors by eliminating false positive entity mentions.

The unlinked text span or a part of it can be referential mention for some other entities, as in the above example Bill and Melinda Gates.

FIG2 also illustrates this process where Lahti, Finland get unlinked after this stage.

The next stage tries to re-link the unlinked tokens correctly.

The objective of this stage is to reduce false negative entity mentions, where an entity mention is not annotated.

This is done by linking the correct referential name of the entity mention to the correct node in KB.To reduce entity linking errors, we use the document level context by restricting the candidate links (entities or non-entities) to the outgoing links of the current document being processed.

For example, in FIG2 , while processing an article about an FinnishAmerican luger Tristan Jeskanen, it is unlikely to observe mention of a 1903 German novel having the same name, i.e., Tristan.

To reduce false negative entity mentions, we construct two trie trees capturing the outgoing links and their candidates referential names for each document.

The first trie contains all links and the second trie only contains links of entities which are predominantly expressed in lowercase phrases 3 (e.g. names of diseases).

For each non-linked uppercase character, we match the longest matching prefix string within the first trie and assign the matching link.

In the remaining non-linked phrases, we match the longest matching prefix string within the second trie and assign the matching link.

Linking the candidate entities in unlinked phrases reduces entity detection error, by eliminating false negative entity mentions.

Unlike BID19 , the two step string matching process ensures the possibility of a lowercase phrase being an entity mention (e.g. lactic acid, apple juice, bronchoconstriction, etc.) and a word with a first uppercase character being a non-entity (e.g. Jazz, RBI, 4 etc.).

FIG2 shows an example of the input and output of this stage.

In this stage, the phrases Tristan, Lahti, Finland and Jeskanen gets linked.

The objective of this stage is to further reduce entity detection errors.

This stage is motivated by the incomplete nature of practical knowledge bases.

KBs do not capture all entities present in a linked text corpus and do not provide all the referential names for an entity mention.

Thus, after stage-II there will be still a possibility of having both types of entity detection errors, false positives, and false negatives.

To reduce such errors in the induced corpus, we select sentences where it is most likely that all entity mention are annotated correctly.

The resultant corpora of selected sentences Table 2 : Statistics of the different datasets generated or used in this work.will be our final dataset.

To select these sentences, we exploit sentence-level context by using POS tags and list of the frequent sentence starting words.

We only select sentences where all unlinked tokens are most likely to be a non-entity mention.

If an unlinked token has a capitalized characters, then it likely to be an entity mention.

We do not select such sentences, except in the following cases.

In the first case, the token is a sentence starter, and is either in a list of frequent sentence starter word 5 or its POS tag is among the list of permissible tags 6 .

In the second case, the token is an adjective, or belongs to occupational titles or is a name of day or month.

FIG2 shows an example of the input and output of this stage.

Here only the first sentence of the document is selected because in the other sentence the name Sami is not linked.

The sentence selection stage ensures that the selected sentences have high-quality annotations.

We observe that only around 40% of sentences are selected by stage III in our experimental setup.

7 Our extrinsic analysis in Section 5.2 shows that this stage helps models to have a significantly better recall.

In the next section, we describe the dataset generated using the HAnDS framework along with its evaluations.

Using the HAnDS framework we generated two datasets as described below: WikiFbF: A dataset generated using Wikipedia, Freebase and the FIGER hierarchy as an input for the HAnDS framework.

This dataset contains around 38 million entity mentions annotated with 118 different types.

WikiFbT: A dataset generated using Wikipedia, Freebase and the TypeNet hierarchy as an input for the HAnDS framework.

This dataset contains around 46 million entity mentions annotated with 1115 different types.

In our experiments, we use the September 2016 Wikipedia dump.

Table 2 lists various statistics of these datasets.

In the next subsections, we estimate the quality of the generated datasets, both intrinsically and extrinsically.

Our intrinsic evaluation is focused on 5.

150 most frequent words were used in the list.

6.

POS tags such as DT, IN, PRP, CC, WDT etc.

that are least likely to be candidate for entity mention.7.

An analysis of several characteristics of the discarded and retained sentences in available in the supplementary material at: https://github.com/abhipec/HAnDS.

Table 3 : Quantitative analysis of dataset generated using the HAnDS framework with the NDS approach of dataset generation.

Here H denotes a set of entity mentions in Table 3a and set of entities in Table 3b generated by the HAnDS framework, and N denotes a set of entity mentions in Table 3a and set of entities in Table 3b generated by the NDS approach.quantitative analysis, and the extrinsic evaluation is used as a proxy to estimate precision and recall of annotations.

In intrinsic evaluation, we perform a quantitative analysis of the annotations generated by the HAnDS framework with the NDS approach.

The result of this analysis is presented in Table 3 .

We can observe that on the same sentences, HAnDS framework is able to generate about 1.9 times more entity mention annotations and about 1.6 times more entities for the WikiFbT corpus compared with the NDS approach.

Similarly, there are around 1.8 times more entity mentions and about 1.6 time more entities in the WikiFbF corpus.

In Section 5.2.4, we will observe that despite around 1.6 to 1.9 times more new annotations, these annotations have a very high linking precision.

Also, there is a large overlap among annotations generated using HAnDS framework and NDS approach.

Around above 95% of entity mentions (and entities) annotations generated using the NDS approach are present in the HAnDS framework induced corpora.

This indicated that the existing links present in Wikipedia are of high quality.

The remaining 5% links were removed by the HAnDS framework as false positive entity mentions.

In extrinsic evaluation, we evaluate the performance of learning models when trained on datasets generated using the HAnDS framework.

Due to resource constraints, we perform this evaluation only for the WikiFbF dataset and its variants.

Following BID12 we divided the FgER task into two subtasks: Fine-ED, a sequence labeling problem and Fine-ET, a multi-label classification problem.

We use the existing state-of-the-art models for the respective sub-tasks.

The FgER model is a simple pipeline combination of a Fine-ED model followed by a Fine-ET model.

Fine-ED model: For Fine-ED task we use a state-of-the-art sequence labeling based LSTM-CNN-CRF model as proposed in BID13 .Fine-ET model: For Fine-ET task we use a state-of-the-art LSTM based model as proposed in BID0 .

Please refer to the respective papers for model details.

8 The values of various hyperparameters used in the models along with the training procedure is mentioned in the supplementary material available at: https://github.com/abhipec/HAnDS.

The two learning models are trained on the following datasets:(1) Wiki-FbF:

Dataset created by the HAnDS framework.(2) Wiki-FbF-w/o-III:

Dataset created by the HAnDS framework without using stage III of the pipeline.

(3) Wiki-NDS: Dataset created using the naive distant supervision approach with the same Wikipedia version used in our work..

(4) FIGER: Dataset created using the NDS approach but shared by BID12 .Except for the FIGER dataset, for other datasets, we randomly sampled two million sentences for model training due to computational constraints.

However, during model training as described in the supplementary material, we ensured that every model irrespective of the dataset, is trained for approximately same number of examples to reduce any bias introduced due to difference in the number of entity mentions present in each dataset.

All extrinsic evaluation experiments, subsequently reported in this section are performed on these randomly sampled datasets.

Also, the same dataset is used to train Fine-ED and Fine-ET learning model.

This setting is different from BID12 where entity detection model is trained on the CoNLL dataset.

Hence, the result reported in their work is not directly comparable.

We evaluated the learning models on the following two datasets: (1) FIGER: This is a manually annotated evaluation corpus which has been created by BID12 .

This contains 563 entity mentions and overall 43 different entity types.

The type distribution in this corpus is skewed as only 11 entity types are mentioned more than 10 times.

(2) 1k-WFB-g: This is a new manually annotated evaluation corpus developed specifically to cover large type set.

This contains 2420 entity mentions and overall 117 different entity types.

In this corpus 84 entity types are mentioned more than 10 types.

The sentences for this dataset construction were sampled from Wikipedia text.

The statistics of these datasets is available in Table 2 .

For the Fine-ED task, we evaluated model's performance using the precision, recall and F1 metrics as computed by the standard conll evaluation script 9 .

For the Fine-ET and 8.

Please note that there are several other models with competitive or better performance such as BID5 BID11 BID13 for sequence labeling problem and BID23 BID27 BID0 BID31 BID32 for multi-label classification problem.

Our criteria for model selection was simple; easy to use publicly available efficient implementation.

Table 4 : Performance of the entity detection models on the FIGER and 1k-WFB-g datasets.the FgER task, we use the strict, loose-macro-average and loose-micro-average evaluation metrics described in BID12 .

The results of the entity detection models on the two evaluation datasets are presented in Table 4 .

From these results we perform two analysis.

First, the effect of training datasets on model's performance and second, the performance comparison among the two manually annotated datasets.

In the first analysis, we observe that the LSTM-CNN-CRF model when trained on WikiFbF dataset has the highest F1 score on both the evaluation corpus.

Moreover, the average difference in precision and recall for this model is the lowest, which indicates a balanced performance across both evaluation corpus.

When compared with the models trained on the NDS generated datasets (Wiki-NDS and FIGER), we observe that these models have best precision across both corpus, however, lowest recall.

The result indicates that large number of false negatives entity mentions are present in the NDS induced datasets.

In the case of model trained on the dataset Wiki-FbF-w/o-III dataset the performance is in between the performance of model trained on Wiki-NDS and Wiki-FbF datasets.

However, they have a significantly lower recall on average around 28% lower than model trained on Wiki-FbF. This highlights the role of stage-III, by selecting only quality annotated sentence, erroneous annotations are removed, resulting in learning models trained on WikiFbF to have a better and a balanced performance.

In the second analysis, we observe that models trained on datasets generated using Wikipedia as sentence source, performs better on the 1k-WFB-g evaluation corpus as compared to the FIGER evaluation corpus.

These datasets are FIGER training corpus, WikiFbF, Wiki-NDS and Wiki-FbF-w/o-III.

The primarily reason for better performance is that the sentences constituting the 1k-WFB-g dataset were sampled from Wikipedia.

10 Thus, this evaluation is a same domain evaluation.

On the other hand, FIGER evaluation corpus is based on sentences sampled from news and specialized magazines (photography and veterinary domains).

It has been observed in the literature that in a cross domain evaluation setting, learning model performance is reduced compared to the same domain evaluation BID20 .

Moreover, this result also conveys that to some extent learning model trained on the large Wikipedia text corpus is also able to generalize on evaluation dataset consisting of sentences from news and specialized magazines.

Our analysis in this section as well as in Section 3.1 indicates that although the type coverage of FIGER evaluation corpus is low (43 types), it helps to better measure model's generalizability in a cross-domain evaluation.

Whereas, 1k-WFB-g helps to measure performance across a large spectrum of entity types (117 types).

Learning models trained on Wiki-FbF perform best on both of the evaluation corpora.

This warrants the usability of the generated corpus as well as the framework used to generate the corpus.

We observe that for the Fine-ET task, there is not a significant difference between the performance of learning models trained on the Wiki-NDS dataset and models trained on the Wiki-FbF dataset.

The later model performs approx 1% better in the micro-F1 metric computed on the 1k-WFB-g corpus.

This indicates that in the HAnDS framework stage-II, where false negative entity mentions were reduced by relinking them to Freebase, has a very high linking precision similar to NDS, which is estimated to be about 97-98% BID30 .The results for the complete FgER system, i.e., Fine-ED followed by Fine-ET are available in TAB5 .

These results supports our claim in Section 3.1, that the current bottleneck for the FgER task, is Fine-ED, specifically lack of resource with quality entity boundary annotations while covering large spectrum of entity types.

Our work directly addressed this issue.

In the FgER task performance measure, learning model trained on WikiFbF has an average absolute performance improvement of at least 18% on all of the there evaluation metrics.

In this work, we initiate a push towards moving from CgER systems to FgER systems, i.e., from recognizing entities from a handful of types to thousands of types.

We propose the HAnDS framework to automatically construct quality training dataset for different variants of FgER tasks.

The two datasets constructed in our work along with the evaluation resource are currently the largest available training and testing dataset for the entity recognition problem.

They are backed with empirical experimentation to warrants the quality of the constructed corpora.

The datasets generated in our work opens up two new research directions related to the entity recognition problem.

The first direction is towards an exploration of sequence labeling approaches in the setting of FgER, where each entity mention can have more than one type.

The existing state-of-the-art sequence labeling models for the CgER task, can not be directly applied in the FgER setting due to state space explosion in the multi-label setting.

The second direction is towards noise robust sequence labeling models, where some of the entity boundaries are incorrect.

For example, in our induced datasets, there are still entity detection errors, which are inevitable in any heuristic approach.

There has been some work explored in BID8 assuming that it is a priori known which tokens have noise.

This information is not available in our generated datasets.

Additionally, the generated datasets are much richer in entity types compared to any existing entity recognition datasets.

For example, the generated dataset contains entities from several domains such as biomedical, finance, sports, products and entertainment.

In several downstream applications where NER is used on a text writing style different from Wikipedia, the generated dataset is a good candidate as a source dataset for transfer learning to improve domain-specific performance.

<|TLDR|>

@highlight

We initiate a push towards building ER systems to recognize thousands of types by providing a method to automatically construct suitable datasets based on the type hierarchy. 