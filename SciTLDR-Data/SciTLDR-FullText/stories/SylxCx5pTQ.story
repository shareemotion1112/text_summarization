This paper presents the formal release of {\em MedMentions}, a new manually annotated resource for the recognition of biomedical concepts.

What distinguishes MedMentions from other annotated biomedical corpora is its size (over 4,000 abstracts and over 350,000 linked mentions), as well as the size of the concept ontology (over 3 million concepts from UMLS 2017) and its broad coverage of biomedical disciplines.

In addition to the full corpus, a sub-corpus of MedMentions is also presented, comprising annotations for a subset of UMLS 2017 targeted towards document retrieval.

To encourage research in Biomedical Named Entity Recognition and Linking, data splits for training and testing are included in the release, and a baseline model and its metrics for entity linking are also described.

One recognized challenge in developing automated biomedical entity extraction systems is the lack of richly annotated training datasets.

While there are a few such datasets available, the annotated corpus often contains no more than a few thousand annotated entity mentions.

Additionally, the annotated entities are limited to a few types of biomedical concepts such as diseases BID4 , gene ontology terms BID18 , or chemicals and diseases BID9 .

Researchers targeting the recognition of multiple biomedical entity types have had to resort to specialized machine learning techniques for combining datasets labelled with subsets of the full target set, e.g. using multi-task learning BID3 , or a modified Conditional Random Field cost which allows un-labeled tokens to take any labels not in the current dataset's target set BID5 .

To promote the development of state-of-the-art entity linkers targeting a more comprehensive coverage of biomedical concepts, we decided to create a large concept-mention annotated gold standard dataset named 'MedMentions' BID11 .With the release of MedMentions, we hope to address two key needs for developing better biomedical concept recognition systems: (i) a much broader coverage of the fields of biology and medicine through the use of the Unified Medical Language System (UMLS) as the target ontology, and (ii) a significantly larger annotated corpus than available today, to meet the data demands of today's more complex machine learning models for concept recognition.

The paper begins with an introduction to the MedMentions annotated corpus, including a subcorpus aimed at information retrieval systems.

This is followed by a comparison with a few other large datasets annotated with biomedical entities.

Finally, to promote further research on large ontology named entity recognition and linking, we present metrics for a baseline end-to-end concept recognition (entity type recognition and entity linking) model trained on the MedMentions corpus.

We randomly selected 5,000 abstracts released in PubMed ®1 between January 2016 and January 2017.

Upon review, some abstracts were found to be outside the biomedical fields or not written in English.

These were discarded, leaving a total of 4,392 abstracts in the corpus.

The Metathesaurus of UMLS BID2 combines concepts from over 200 source ontologies.

It is therefore the largest single ontology of biomedical concepts, and was a natural choice for constructing an annotated resource with broad coverage in biomedical science.

In this paper, we will use entities and concepts interchangeably, to refer to UMLS concepts.

The 2017 AA release of the UMLS Metathesaurus contains approximately 3.2 million unique concepts.

Each concept has a unique id (a "CUID") and primary name and a set of aliases, and is linked to all the source ontologies it was mapped from.

Each concept is also linked to one or more Semantic Types -the UMLS guidelines are to link each concept to the most specific type(s) available.

Each Semantic Type also has a unique identifier ("TUI") and a name.

The Metathesaurus contains 127 Semantic Types, arranged in a "is-a" hierarchy.

About 91.7% of the concepts are linked to exactly one semantic type, approximately 8% to two types, and a very small number to more than two types.

We recruited a team of professional annotators with rich experience in biomedical content curation to exhaustively annotate UMLS entity mentions from the abstracts.

The annotators used the text processing tool GATE 2 (version 8.2) to facilitate the curation.

All the relevant scientific terms from each abstract were manually searched in the 2017 AA (full) version of the UMLS metathesaurus 3 and the best matching concept was retrieved.

The annotators were asked to annotate the most specific concept for each mention, without any overlaps in mentions.

To gain insight on the annotation quality of MedMentions, we randomly selected eight abstracts from the annotated corpus.

Two biologists ("Reviewers") who did not participate in the annotation task then each reviewed four abstracts and the corresponding concepts in MedMentions.

The abstracts contained a total of 469 concepts.

Of these 469 concepts, the agreement between Reviewers and Annotators was 97.3%, estimating the precision of the annotation in MedMentions.

Due to the size of UMLS, we reasoned that no human curators would have knowledge of the entire UMLS, so we did not perform an evaluation on the recall.

We are working on getting more detailed IAA (Inter-annotator agreement) data, which will be released when that task is completed.

Entity linking / labeling methods have prominently been used as the first step towards relationship extraction, e.g. the BioCreative V CDR task for Chemical-Disease relationship extraction BID9 , and for indexing for entity-based document retrieval, e.g. as described in the BioASQ Task A for semantic indexing BID12 .

One of our goals in building a more comprehensive annotated corpus was to provide indexing models with a larger ontology than MeSH (used in BioASQ Task A and PubMed) for semantic indexing, to support more specific document retrieval queries from researchers in all biomedical disciplines.

UMLS does indeed provide a much larger ontology (see TAB8 ).

However UMLS also contains many concepts that are not as useful for specialized document retrieval, either because they are too broad so not discriminating enough (e.g. Groups [cuid = C0441833], Risk [C0035647]), or cover peripheral and supplementary topics not likely to be used by a biomedical researcher in a query (e.g. DISPLAYFORM0 Filtering UMLS to a subset most useful for semantic indexing is going to be an area of ongoing study, and will have different answers for different user communities.

Furthermore, targeting different subsets will also impact machine learning systems designed to recognize concepts in text.

As a first step, we propose the "ST21pv" subset of UMLS, and the corresponding annotated subcorpus MedMentions ST21pv.

Here "ST21pv" is an acronym for "21 Semantic Types from Preferred Vocabularies", and the ST21pv subset of UMLS was constructed as follows:1.

We eliminated all concepts that were only linked to semantic types at levels 1 or 2 in the UMLS Semantic Type hierarchy with the intuition that these concepts would be too broad.

We also limited the concepts to those in the Active subset of the 2017 AA release of UMLS.2.

We then selected 21 semantic types at levels 3-5 based on biomedical relevance, and whether MedMentions contained sufficient annotated examples.

Only concepts mapping into one of these 21 types (i.e. linked to one of these types or to a descendant in the type hierarchy) were considered for inclusion.

As an example, the semantic type Archaeon [T194] was excluded because MedMentions contains only 25 mentions for 15 of the 5,418 concepts that map into this type TAB1 ).Since our primary purpose for ST21pv is to use annotations from this subset as an aid for biomedical researchers to retrieve relevant papers, some types were eliminated if most of their member concepts were considered by our staff biologists as not useful for this task.

3. Finally, we selected 18 'prefered' source vocabularies (Table 1) , and excluded any concepts that were not linked in UMLS to at least one of these sources.

These vocabularies were selected based on usage and relevance to biomedical research 4 , with an emphasis on gene function, disease and phenotype, structure and anatomy, and drug and chemical entities.

been pruned and their counts rolled up.

The counts therefore are for concepts linked to the corresponding type for the non-bold rows, and mapped to the ST21pv types for the rows in bold.

Note that some concepts in UMLS are linked to multiple semantic types.

The prefix MM-in the column name indicates the counts are for concepts mentioned in MedMentions.

The full MedMentions corpus contains 2,473 mentions of 685 concepts that are not members of the 2017 AA Active release.

These were eliminated as part of step 1.

The other non-bold rows in the table represent semantic types excluded in steps 1 and 2, corresponding to a total of 135,986 mentions of 6,002 unique concepts.

A further 10,755 mentions of 2,618 concepts were eliminated in step 3.

As a result of all this filtering, the target ontology for MedMentions ST21pv (MM-ST21pv) contains 2,327,250 concepts and 203,282 concept mentions.

Examples of broad concepts eliminated by selecting semantic types at level 3 or higher:• C1707689: "Design", linked to T052: Activity, level=2• C0029235: "Organism" linked to T001: Organism, level=3• C0520510: "Materials" linked to T167: Substance, level=3

The MedMentions corpus consists of 4,392 abstracts randomly selected from those released on PubMed between January 2016 and January 2017.

the MedMentions corpus and its ST21pv subset.

The tokenization and sentence splitting were performed using Stanford CoreNLP 5 BID10 .

Due to the size of UMLS, only about 1% of its concepts are covered in MedMentions.

So a major part of the challenge for machine learning systems trained to recognize these concepts is 'unseen labels' (often called "zero-shot learning", e.g. BID14 BID17 BID20 ).

As part of the release, we also include a 60% -20% -20% random partitioning of the corpus into training, development (often called 'validation') and test subsets.

These are described in TAB5 .

As can be seen from the table, about 42% of the concepts in the test data do not occur in the training data, and 38% do not occur in either training or development subsets.

The MedMentions resource has been published at https://github.com/chanzuckerberg/ MedMentions.

The corpus itself is in PubTator BID19 ] format, which is described on the release site.

The corpus consists of PubMed abstracts, each identified with a unique PubMed identifier (PMID).

Each PubMed abstract has Title and Abstract texts, and a series of annotations of concept mentions.

Each concept mention identifies the portion of the document text comprising the mention, and the UMLS concept.

A separate file for the ST21pv sub-corpus is also included in the release.

The release also includes three lists of PMID's that partition the corpus into a 60% -20% -20% split defining the Training, Development and Test subsets.

Researchers are encouraged to train their models using the Training and Development portions of the corpus, and publish test results on the held-out Test subset of the corpus.

There have been several gold standard (manually annotated) corpora of biomedical scientific literature made publicly available.

Some of the larger ones are described below.

GENIA: BID13 One of the earliest 'large' biomedical annotated corpora, it is aimed at biomedical Named Entity Recognition, where the annotations are for 36 biomedical Entity Types.

The dataset consists of 2,000 MEDLINE abstracts about "biological reactions concerning transcription factors in human blood cells", collected by searching on MEDLINE using the MeSH terms human, blood cells and transcription factors.

An extended version (2,404 abstracts), with a smaller ontology (six types) was later used for the JNLPBA 2004 NER task BID7 .ITI TXM Corpora: BID0 Among the largest gold standard biomedical annotated corpora previously available, this consists of two sets of full-length papers obtained from PubMed and PubMed Central: 217 articles focusing on protein-protein interactions (PPI) and 238 articles on tissue expressions (TES).

The PPI and TES corpora were annotated with entities from NCBI Taxonomy, NCBI Reference Sequence Database, and Entrez Gene.

The TES corpus was also annotated with entities from Chemical Entities of Biological Interest (ChEBI) and Medical Subject Headings (MeSH).

The concepts were grouped into 15 entity types, and these type labels were included in the annotations.

In addition to concept mentions, the corpus also includes relations between entities.

The statistics TAB4 ) for this corpus BID0 are a little confusing, since not all sections of the articles were annotated.

Furthermore some articles were annotated by more than one biologist, and each annotated version was incorporated into the corpus as a separate document.

CRAFT: BID1 The Colorado Richly Annotated Full-Text (CRAFT) Corpus is another large gold standard corpus annotated with a diverse set of biomedical concepts.

It consists of 67 full-text open-access biomedical journal articles, downloaded from PubMed Central, covering a wide range of disciplines, including genetics, biochemistry and molecular biology, cell biology, developmental biology, and computational biology.

The text is annotated with concepts from 9 biomedical ontologies: ChEBI, Cell Ontology, Entrez Gene, Gene Ontology (GO) Biological Process, GO Cellular Component, GO Molecular Function, NCBI Taxonomy, Protein Ontology, and Sequence Ontology.

The latest release of CRAFT 6 reorganizes this into ten Open Biomedical Ontologies.

The corpus also contains exhaustive syntactic annotations.

TAB4 gives a comparison of the sizes of CRAFT against the other corpora mentioned here.

MedMentions can be viewed as a supplement to the CRAFT corpus, but with a broader coverage of biomedical research (over four thousand abstracts compared to the 67 articles in CRAFT).

Through the larger set of ontologies included within UMLS, MedMentions also contains more comprehensive annotation of concepts from some biomedical fields, e.g. diseases and drugs (see Table 1 for a partial list of the ontologies included in UMLS).BioASQ Task A: BID12 The Large Scale Semantic Indexing task considers assigning MeSH headings for 'important' concepts to each document.

The training data is very large, but with a smaller target concept vocabulary (see TAB8 ), and annotation (by NCBI) is at the document level rather than at the mention level.

Relation / Event Extraction Corpora: Most recently developed manually annotated datasets of biomedical scientific literature have focused on the task of extracting biomedical events or relations between entities.

These datasets have been used for shared tasks in biomedical NLP workshops like BioCreative, e.g. BC5-CDR BID9 which focuses on ChemicalDisease relations, and BioNLP, e.g. the BioNLP 2013 Cancer Genetics (CG) and Pathway Curation tasks BID15 where the main goal is to identify events involving entities.

While these datasets include entity mention annotations, they are typically focused on a small set of entity types, and the sizes of the corpora are also smaller (1,500 document abstracts in BC5-CDR, 600 abstracts in CG, and 525 abstracts in PC).

Machine learning mod-

Our main goal in constructing and releasing MedMentions is to promote the development of models for recognizing biomedical concepts mentioned in scientific literature.

To help jumpstart this research, we now present a baseline modeling approach trained using the Training and Development splits of MedMentions ST21pv, and its metrics on the MM-ST21pv Test set.

A subset of a pre-release version of MedMentions was also used by BID11 ] to test their hierarchical entity linking model.

We measure the performance of the model described below at both the mention level (also referred to as phrase level) and the document level.

Concept annotations in MedMentions identify an exact span of text using start and end positions, and annotate that span with an entity type identifier and entity identifier.

Concept recognition models like the one described below will output predictions in a similar format.

The performance of such models is usually measured using mention level precision, recall and F1 score as described in BID16 ].

Here we are interested in measuring the entity resolution performance of the model: a prediction is counted as a true positive (tp) only when the predicted text span as well as the linked entity (and by implication the entity type) matches with the gold standard reference.

All other predicted mentions are counted as falsepositives (f p), and all un-matched reference entity mentions as false-negatives (f n).

These counts are used to compute the following metrics: Mention level metrics would be the primary concept recognition metrics of interest when, for example, the model is used as a component in a relation extraction system.

As another example, the Disease recognition task in BC5-CDR described above uses mention level metrics.

DISPLAYFORM0 Document level metrics are computed in a similar manner, after mapping all concept mentions as entity labels directly to the document, and discarding all associations with spans of text that identify the locations of the mentions in the document.

For example, a document may contain three mentions of the concept Breast Carcinoma in three different parts (spans) of the document text; for document level metrics they are all mapped to one label on the document.

Document level metrics are useful in information retrieval when the goal is simply to retrieve the entire matching document, and are used in the BioASQ Large Scale Semantic Indexing task mentioned earlier.

TaggerOne ] is a semi-Markov model doing joint entity type recognition and entity linking, with perceptron-style parameter estimation.

It is a flexible package that handles simultaneous recognition of multiple entity types, with published results near state-of-the-art, e.g. for joint Chemical and Disease recognition on the BC5-CDR corpus.

We used the package without any changes to its modeling features.

The MM-ST21pv data presented to TaggerOne was modified as follows: for each mention of a concept in the data, the Semantic Type label was modified to one of the 21 semantic types ( TAB1 that concept mapped into.

Thus each mention was labeled with one of 21 entity types, as well as linked to a specific concept from the ST21pv subset of UMLS.

Twenty one lexicons of primary and alias names for each concept in the 21 types, extracted from UMLS 2017 AA Active, were also provided to TaggerOne.

Training was performed on the Training split, and the Development split was provided as holdout data (validation data, used for stopping training).

The model was trained with the parameters: REGULARIZATION = 0, MAX STEP SIZE = 1.5, for a maximum of 10 epochs with patience (iterationsPastLastImprovement) of 1 epoch.

Our model took 9 days to train on a machine equipped with Intel Xeon Broadwell processors and over 900GB of RAM.When TaggerOne detects a concept in a document it identifies a span of text (start and end positions) within the document, and labels it with an entity type and links it to a concept from that type.

Metrics are calculated by comparing these concept predictions against the reference or ground truth (i.e. the annotations in MM-ST21pv).

As a baseline for future work on biomedical concept recognition, both mention level and document level metrics for the TaggerOne model, computed on the MM-ST21pv Test subset, are reported in TAB9 .

We presented the formal release of a new resource, named MedMentions, for biomedical concept recognition, with a large manually annotated annotated corpus of over 4,000 abstracts targeting a very large fine-grained concept ontology consisting of over 3 million concepts.

We also included in this release a targeted sub-corpus (MedMentions ST21pv), with standard training, development and test splits of the data, and the metrics of a baseline concept recognition model trained on this subset, to allow researchers to compare the metrics of their concept recognition models.

@highlight

The paper introduces a new gold-standard corpus corpus of biomedical scientific literature manually annotated with UMLS concept mentions.

@highlight

Details the construction of a manually annotated dataset covering biomedical concepts that is larger and covered by a larger ontology than previous datasets.

@highlight

This paper uses MedMentions, a TaggerOne semi-Markov model for end-to-end concept recognition and linking on a set of Pubmed abstracts to label papers with biomedical concepts/entities