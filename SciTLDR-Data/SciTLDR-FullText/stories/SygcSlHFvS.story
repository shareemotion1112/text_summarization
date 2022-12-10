Many methods have been developed to represent knowledge graph data, which implicitly exploit low-rank latent structure in the data to encode known information and enable unknown facts to be inferred.

To predict whether a relationship holds between entities, their embeddings are typically compared in the latent space following a relation-specific mapping.

Whilst link prediction has steadily improved, the latent structure, and hence why such models capture semantic information, remains unexplained.

We build on recent theoretical interpretation of word embeddings as a basis to consider an explicit structure for representations of relations between entities.

For identifiable relation types, we are able to predict properties and justify the relative performance of leading knowledge graph representation methods, including their often overlooked ability to make independent predictions.

Knowledge graphs are large repositories of binary relations between words (or entities) in the form of fact triples (subject, relation, object).

Many models have been developed for learning representations of entities and relations in knowledge graphs, such that known facts can be recalled and previously unknown facts can be inferred, a task known as link prediction.

Recent link prediction models (e.g. Bordes et al., 2013; Trouillon et al., 2016; Balažević et al., 2019b ) learn entity representations, or embeddings, of far lower dimensionality than the number of entities, by capturing latent structure in the data.

Relations are typically represented as a mapping from the embedding of a subject entity to its related object entity embedding(s).

Although the performance of knowledge graphlink prediction models has steadily improved for nearly a decade, relatively little is understood of the low-rank latent structure that underpins these models, which we address in this work.

We start by drawing a parallel between entity embeddings in knowledge graphs and unsupervised word embeddings, as learned by algorithms such as Word2Vec (W2V) (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) .

We assume that words have latent features, e.g. meaning(s), tense, grammatical type, that are innate and fixed, irrespective of what an embedding may capture (which may be only a part, subject to the embedding method and/or the data source); and that this same latent structure gives rise to patterns observed in the data, e.g. in word co-occurrence statistics and in which words are related to which.

As such, an understanding of the latent structure from one embedding task (e.g. word embedding) might be useful to another (e.g. knowledge graph entity embedding).

Recent work theoretically explains how semantic properties are encoded in word embeddings that (approximately) factorise a matrix of word cooccurrence pointwise mutual information (PMI), e.g. as is known for W2V (Levy & Goldberg, 2014) .

Semantic relationships between words (specifically similarity, relatedness, paraphrase and analogy) are proven to manifest as linear relationships between rows of the PMI matrix (subject to known error terms), of which word embeddings can be considered low-rank projections.

This explains why similar words (e.g. synonyms) have similar embeddings; and embeddings of analogous word pairs share a common "vector offset".

Importantly, this insight allows us to identify geometric relationships between such word embeddings necessary for other semantic relations to hold, such as those of knowledge graphs.

These relation conditions describe relation-specific mappings between entity embeddings, i.e. relation representations, providing a "blue-print" against which to consider knowledge graph representation models.

We find that various properties of knowledge graph representation models, including the relative DistMult (Yang et al., 2015) multiplicative (diagonal) e s Re o TuckER (Balažević et al., 2019b) multiplicative W × 1 e s × 2 r × 3 e o MuRE (Balažević et al., 2019a) performance of leading link prediction models, accord with predictions based on these relation conditions, suggesting a commonality to the latent structure learned in word embedding models and knowledge graph representation models, despite the significant differences between their training data and methodology.

In summary, the key contributions of this work are:

• to use recent understanding of PMI-based word embeddings to derive what a relation representation must achieve to map a subject word embedding to all related object word embeddings (relation conditions), based on which relations can be categorised into three types; • to show that properties of knowledge graph models fit predictions made from relation conditions, e.g. strength of a relation's relatedness aspect is reflected in the eigenvalues of its relation matrix; • to show that the performance per relation of leading link prediction models corresponds to the ability of the model's architecture to meet the relation conditions of the relation's type, i.e. the better the architecture of a knowledge graph representation model aligns with the form theoretically derived for PMI-based word embeddings, the better the model performs; and • noting how ranking metrics can be flawed, to provide novel insight into the prediction accuracy per relation of recent knowledge graph models, an evaluation metric we recommend in future.

Our work draws on knowledge graph representation and word embedding.

Whilst related, these tasks differ materially in their training data.

The former is restricted to datasets crafted by hand or automatically generated, the latter has the vast abundance of natural language text (e.g. Wikipedia).

Almost all recent knowledge graph models represent entities e s , e o as vectors e s , e o ∈ R de of low dimension (e.g. d e = 200) relative to the number of entities n e (typically of order 10 4 ), and relations as transformations in the latent space from subject entity embedding to object.

These models are distinguished by their score function, which defines (i) the form of the relation transformation, e.g. matrix multiplication, vector addition; and (ii) how "closeness" between the transformed subject embedding and an object embedding is evaluated, e.g. dot product, Euclidean distance.

Score functions can be non-linear (e.g. Dettmers et al. (2018) ), or linear and sub-categorised as additive, multiplicative or both.

We focus on linear models due to their simplicity and strong performance at link prediction (including state-of-the-art).

Table 1 shows the score functions of competitive linear models that span all linear sub-categories: TransE (Bordes et al., 2013) , DistMult (Yang et al., 2015) , TuckER (Balažević et al., 2019b) and MuRE (Balažević et al., 2019a) .

Additive models typically use Euclidean distance and contain a relation-specific translation from a (possibly transformed) subject to a (possibly transformed) object entity embedding.

A generic additive score function is given by φ(e s , r, e o ) = − R s e s +r−R o e o 2 2 +b s +b o = − e Multiplicative models have the generic score function φ(e s , r, e o ) = e s Re o = e (r) s , e o , i.e. a bilinear product of the entity embeddings and a relation-specific matrix R. DistMult is a simple example with diagonal R and so cannot model asymmetric relations (Trouillon et al., 2016) .

In TuckER, each relation-specific R = W × 3 r is a linear combination of d r "prototype" relation matrices in a core tensor W ∈ R de×dr×de (where × n denotes tensor product along mode n), facilitating multi-task learning across relations.

Algorithms such as Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) generate succinct low-rank word embeddings that perform well on downstream tasks (Baroni et al., 2014) .

Such models predict the context words (c j ) observed around each target word (w i ) in a text corpus using shallow neural networks.

Whilst recent language models (e.g. Devlin et al. (2018); Peters et al. (2018) ) create impressive context-specific word embeddings, we focus on the former embeddings since knowledge graph entities have no obvious context and, more importantly, they are interpretable.

Levy & Goldberg (2014) show that, for a dictionary of unique words D and embedding dimension d |D|, W2V's loss function is minimised when its weight matrices W, C ∈ R d×|D| (whose columns are word embeddings w i , c j ) factorise a word co-occurrence pointwise mutual information (PMI) matrix, subject to a shift term (PMI(w i , c j ) = log Recent work shows why word embeddings that factorise such PMI matrix encode semantic word relationships .

The authors show that word embeddings can be seen as low-rank projections of high dimensional PMI vectors (rows of the PMI matrix), between which the semantic relationships of similarity, relatedness, paraphrase and analogy provably manifest as linear geometric relationships (subject to defined error terms), which are then preserved, under a sufficiently linear projection, between word embeddings.

Thus similar words have similar embeddings, and the embeddings of analogous word pairs share a common vector offset.

Specifically, the PMI vectors (p x ) of an analogy "man is to king as woman is to queen" satisfy p Q −p W ≈ p K −p M because the difference between words associated with king and man (e.g. reign, crown) mirrors that between queen and woman.

This leads to a common difference between their co-occurrence distributions (over all words), giving a common difference between their PMI vectors, which projects to a common difference between embeddings.

Any discrepancy in the mirroring of word associations is shown to introduce error, weakening the analogy, as does a lack of statistical independence within certain word pairs (see ).

The common difference in word co-occurrence distributions, e.g. the increased association with words {reign, crown, etc.}, can be interpreted semantically as a common change in context (context-shift) that transforms man to king and woman to queen by adding a royal context.

Under this interpretation, context can also be subtracted, e.g. "king is to man as queen is to woman" (minus royal); or both, e.g. "boy is to king as girl is to queen" (minus youth plus royal).

Adding context can also be interpreted as specialisation, and subtracting context as generalisation.

This establishes a correspondence between common word embedding vector offsets and semantic context-shifts.

Although the projection from PMI vectors to word embeddings preserves the relative relationships, and thus the above semantic interpretability of common embedding differences, a direct interpretation of dimensions themselves is obscured, not least because any embedding matrix can be arbitrarily scaled/rotated if the other is inversely transformed.

Our aim is to build on the understanding of PMI-based word embeddings (henceforth word embeddings), to identify what a knowledge graph relation representation needs to achieve to map all subject word embeddings to all related object word embeddings.

We note that if a semantic relation between two words implies a particular geometric relationship between their embeddings, then the latter serves as a necessary quantitative condition for the former to hold (a relation condition).

Relation conditions implicitly define a relation-specific mapping by which all subject embeddings are mapped to all related object embedding(s), allowing related entities to be identified by a proximity measure (e.g. Euclidean distance or dot product).

Since this is the approach of many knowledge graph representation models, their performance can be contrasted with their ability to express mappings that satisfy required relation conditions.

For example, similarity and context-shift relations respectively imply closeness and a relation-specific vector offset between embeddings (S2.2).

Such relation conditions can be tested for by respectively making no change to the subject entity or adding the relation-specific offset, before measuring proximity with the object.

We note that since relation conditions are not necessarily sufficient, they do not guarantee a relation holds, i.e. false positives may arise.

In general, the data from which knowledge graph embeddings are derived differs greatly to the co-occurrence data used for word embeddings, and the latter would not be anticipated to be learned by knowledge graph models.

However, word embeddings provide a known solution (i.e. minimise the loss function) of any knowledge graph model able to express the required mapping(s) derived from relation conditions, where the loss function measures proximity between mapped entities.

The relation conditions for certain relation types (underlined) follow readily from S2.2:

• Similarity: Semantically similar words induce similar distributions over the words they co-occur with.

Thus their PMI vectors (Fig 1a) and word embeddings are similar.

• Relatedness:

The relatedness of two words can be defined in terms of the words with which both co-occur similarly (S ∈ D), which define the nature of relatedness, e.g. milk and cheese are related by S = {dairy, breakfast, ...}; and |S| reflects the strength of relatedness.

Since PMI vector components corresponding to S are similar (Fig 1b) , embeddings of "S-related" words have similar components in the subspace V S that spans the projected PMI vector dimensions corresponding to S. The rank of V S might be expected to reflect relatedness strength.

In general, relatedness is a weaker, more variable relation than similarity, its limiting case with S = D and rank(V S ) = d.

• Context-shift: In the context of word embeddings, analogy typically refers to relational similarity (Turney, 2006; Gladkova et al., 2016) .

More specifically, the relations within analogies that give a common vector offset between word embeddings require related words to have a common difference between their distributions of co-occurring words, defined as a context-shifts (see S2.2).

These relations are strictly 1-to-1 and include an aspect of relatedness due to the word associations in common (Fig 1d) .

A specialisation relation is a context-shift in which context is only added (Fig 1c) .

• Generalised context-shift: Context-shift relations are generalised to 1-to-many, many-to-1 and many-to-many relations by letting the fully-specified added or subtracted context be one of a (relationspecific) context set (Fig 1e) , e.g. allowing an entity to be any colour or anything blue.

The potential scope and size of each context set means these relations can vary greatly.

The limiting case for small context sets has a single context in each, whereby the relation is an explicit context-shift (as above), and the difference between embeddings is a known vector offset.

In the limiting case where context sets are large, the added/subtracted context is so loosely defined that, in effect, only the relatedness aspect of the relation and thus only the common subspace component of embeddings is known.

Link to set theory: Viewing PMI vectors as sets of word associations and taking intuition from Fig 1, the above relations can be seen to reflect set operations: similarity as set equality; relatedness as equality of a subset; and context-shift as the set difference equalling a relation-specific set.

This highlights how the relatedness aspect of a relation reflects features that must be common, and contextshift reflects features that must differ.

Whilst this mirrors an intuitive notion of "feature vectors", we emphasise that this is grounded in the co-occurrence statistics of PMI-based word embeddings.

Analysing the relations of popular knowledge graph datasets with the above perspective, we find that they comprise (i) a relatedness aspect that reflects a common theme (e.g. both entities are animals or geographic terms); and (ii) specific word associations of the subject and/or object entities.

Specifically, relations appear to fall under a hierarchy of three relation types: highly related (R); (generalised) specialisation (S); and (generalised) context-shift (C).

As above, "generalised" indicates that any added/subtracted contexts can be from a set.

From Fig 1, type R relations can be seen as a special case of S, which, in turn, is a special case of C. Type C is therefore a generalised case of all considered relations.

Whilst there are several other ways to classify relations (e.g. by their hierarchy, Examples (subject entity, object entity)

transitivity), by considering relation conditions, we delineate by the required mathematical form (and complexity) of their representation.

Table 2 shows a categorisation of the relations of the WN18RR dataset (Dettmers et al., 2018) , containing 11 relations between 40,943 entities.

1 An explanation for this assignment is in Appx.

A and that for NELL-995 (Xiong et al., 2017 ) is in Appx.

B. A review of the FB15k-237 dataset (Toutanova et al., 2015) shows the vast majority of relations to be of type C preventing a contrast between relation types being drawn, hence we do not consider that dataset.

Given the relation conditions of a particular relation type, we can recognise mappings that meet them and thus loss functions (that evaluate the proximity of mapped entity embeddings by dot product or Euclidean distance) able to identify relations of that type between PMI-based word embeddings.

We then contrast these theoretically inspired loss functions (one per relation type) with those of knowledge graph models (Table 1) and, on the outline assumption that a common low-rank latent structure is exploited by both word embeddings and knowledge graph models, predict properties and the relative performance of different knowledge graphs models for different relation types.

R: To evidence S-relatedness, both entity embeddings e s , e o must be projected onto a subspace V S , where their images are compared.

Projection requires multiplication by a matrix P r ∈ R d×d and cannot be achieved additively, except in the limiting case of similarity, when P r = I or vector r ≈ 0 is added.

Comparison by dot product gives (P r e s ) (P r e o ) = e s P r P r e o = e s M r e o (for a relation-specific symmetric M r = P r P r ).

Euclidean distance gives P r e s − P r e o 2 = (e s −e o ) M r (e s −e o ) = P r e s 2 − 2e s M r e o + P r e o 2 .

S/C: Evidencing these relations requires a test both for S-relatedness and for relation-entity-specific embeddings component(s) (v s r , v o r ).

This can be achieved by (i) multiplying both entity embeddings by a relation-specific projection matrix P r that projects onto the subspace that spans the low-rank projection of dimensions corresponding to S, v s r and v o r , (which tests for S-relatedness whilst preserving any entity-specific embedding components); and (ii) adding a relation-specific vector r = v o r − v s r to the transformed subject entity embeddings.

Comparison of the final transformed entity embeddings by dot product equates to (P r e s + r) P r e o ; and by Euclidean distance to P r e s + r − P r e o 2 = P r e s + r 2 − 2(P r e s + r) P r e o + P r e o 2 (cf MuRE: Re s + r − e o 2 ).

Contrasting the above loss functions with those of knowledge graph models (Table 1) , we make the following predictions: (P1) the ability to learn the representation of a particular relation is expected to reflect the complexity of its type (R>S>C), and whether all relation conditions (e.g. additive or multiplicative interactions) can be met under a given model; (P2) relation matrices for relatedness (type R) relations are highly symmetric; (P3) offset vectors for relatedness relations have low norm; and (P4) as a proxy to the rank of V S , the eigenvalues of a relation matrix reflect a relation's strength of relatedness.

To elaborate: P1 anticipates that additive-only models (e.g. TransE) are not suited to identifying the relatedness aspect of relations (except in limiting cases of similarity, requiring a zero vector); and multiplicative-only models (e.g. DistMult) should perform well on type R but are not suited to identifying entity-specific features of type S/C, for which an asymmetric relation matrix in TuckER may help compensate.

Further, the loss function of MuRE closely resembles that derived for type C relations (which generalise all others) and is thus expected to perform best overall.

We test the predictions made on the basis of word embeddings by comparing the performance of competitive knowledge graph models, TransE, DistMult, TuckER and MuRE (see S2), which entail different forms of relation representation, on all WN18RR relations and a similar number of NELL-995 relations (selected to represent each relation type).

Since applying the logistic sigmoid to the score function of TransE does not give a probabilistic interpretation comparable to other models, we include MuRE I , a constrained variant of MuRE with R s = R o = I, as a proxy to TransE for a fairer comparison.

Implementation details are included in Appx.

D. For evaluation, we generate 2n e evaluation triples for each test triple (for the number of entities n e ) by fixing the subject entity e s and relation r and replacing the object entity e o with all possible entities and then keeping e o and r fixed and varying e s .

The obtained scores are ranked to give the standard metric hits@10 (Bordes et al., 2013), i.e. the fraction of times a true triple appears in the top 10 ranked evaluation triples.

Tables 3 and 4 report results (hits@10) for each relation and include the relation type and known confounding influences: percentage of relation instances in the training and test sets (approximately equal), number of instances in the test set, Krackhardt hierarchy score (see Appx.

E) (Krackhardt, 2014; Balažević et al., 2019a) and maximum and average shortest path between any two related nodes.

A further confounding effect is dependence between relations.

Balažević et al. (2019b) and Lacroix et al. (2018) show that constraining the rank of relation representations benefits datasets with many relations (particularly when the number of instances per relation is low) due to multi-task learning, which is expected to benefit TuckER on the NELL-995 dataset (200 relations).

Note that all models have a comparable number of free parameters.

As predicted, all models tend to perform best at type R relations, with a clear performance gap to other relation types.

Also, performance on type S relations appears higher in general than type C. Additive-only models (TransE, MuRE I ) perform most poorly on average, in line with prediction since all relation types involve a relatedness component.

They achieve their best results on type R relations, where the relation vector can be zero/small.

Multiplicative-only DistMult performs well, sometimes best, on type R relations, fitting expectation as it can fully represent those relations and has no additional parameters that may overfit to noise (which may explain where MuRE performs slightly worse).

As expected, MuRE, performs best on average (particularly on WN18RR), and most strongly on S and C type relations that require both multiplicative and additive components.

The comparable performance of TuckER on NELL-995 is explained by its ability for multi-task learning.

Other unexpected results also closely align with confounding factors, e.g. that all models perform poorly on the hypernym relation, despite it having type S and a relative abundance of training data (40% of all instances), might be explained by its hierarchical nature (Khs ≈ 1 and long paths).

The same may explain the reduced performance on type R relations also_see and agentcollaborateswithagent.

As found previously, none of the models considered are well suited to modelling hierarchical structures (Balažević et al., 2019a) .

We also note that the percentage of training instances of a relation does not seem to correlate with its performance, as might typically be expected.

Table 5 shows the symmetry score (∈ [-1, 1] indicating perfect anti-symmetry to symmetry; see Appx.

F) for the relation matrix of TuckER and the norm of relation vectors of TransE, MuRE I and MuRE on the WN18RR dataset.

As expected, type R relations have high symmetry, whereas both other relation types have lower scores, fitting the expectation that TuckER compensates for having no additive component.

All additive models learn relation vectors of a noticeably lower norm for type R relations (where, in the extreme, no additive component is required) than for types S and C. Fig 2 shows eigenvalue magnitudes (scaled relative to their largest and ordered) of the relationspecific matrices R of MuRE, labelled by relation type.

Predicted to reflect strength of a relation's relatedness, they should be highest for type R relations, as observed.

For relation types S and C the profiles are more varied, fitting the expectation that the relatedness of those types has greater variability in both choice and size of S, i.e. in the nature and strength of relatedness.

In summary, the results support all predictions made based on the assumption that knowledge graph models benefit from the same latent semantic structure as word embeddings and the relation conditions theoretically derived from them.

Our analysis identifies the best performing model per relation type: multiplicative-only DistMult for type R, additive-multiplicative MuRE for types S/C; providing a basis for dataset-dependent model selection.

The per-relation insight into where models perform poorly, e.g. on hierarchical or type C relations, can also be used to aid and direct future model design.

Even though in practice we want to know whether a particular triple is true or false, such independent predictions are not commonly reported or evaluated.

Despite many recent link prediction models being able to independently predict the truth of each triple, it is common practice to report rankingbased metrics, e.g. mean reciprocal rank, hits@k, which compare the prediction of a test triple to those of all evaluation triples (see S4).

Not only is this computationally costly, the evaluation is flawed if entities are related to more than k others and does not evaluate a model's ability to independently predict whether "a is related to b".

We address this by considering actual model predictions.

Since for each relation there are n 2 e possible entity-entity relationships, we sub-sample by computing predictions for all (e s , r, e o ) triples only for each e s , r pair seen in the test set.

We split positive predictions (σ(φ(e s , r, e o )) > 0.5) between (i) training and test/validation instances (known truths); and (ii) other, the truth of which is not known.

Per relation, we then compute accuracy over the training instances (train) and the test/validation instances (test); and the average number of other truths predicted per e s , r pair.

Table 6 shows results for MuRE I , DistMult, TuckER and MuRE.

All Table 6 : Per relation prediction accuracy for MuRE I (M I ), (D)istMult, (T)uckER and (M)uRE (WN18RR).

models achieve almost perfect training accuracy.

The additive-multiplicative MuRE gives best test set performance, followed (surprisingly) closely by MuRE I , with multiplicative models (DistMult and TuckER) performing poorly on all but type R relations.

Analysing a sample of "other" positive predictions for a relation of each type (see Appx.

G), we estimate that TuckER is relatively accurate but pessimistic (∼0.3 correct of the 0.5 predictions ≈ 60%), MuRE I is optimistic but inaccurate (∼2.3 of 7.5≈ 31%), whereas MuRE is both optimistic and accurate (∼1.1 of 1.5≈ 73%).

Fig 3 shows histograms of MuRE prediction probabilities for the same sample relations, split by known truths (training and test/validation) and other instances.

There is a clear distinction between relation types: for type R, most train and test triples are classified correctly with high confidence; for types S and C, an increasing majority of incorrect test predictions are far below the decision boundary, i.e. the model is confidently incorrect.

For relation types where the model is less accurate, fewer positive predictions are made overall and the prediction distribution is more peaked towards zero.

This analysis probes further into the difficulty models have representing type S/C relations.

Given the additional insight provided and the benefit of standalone predictions, we recommend the inclusion of predictive performance in future link prediction work.

Many models learn low-rank representations for knowledge graph link prediction, yet little is known about the latent structure they learn.

We build on recent understanding of PMI-based word embeddings to theoretically establish what a relation representation must achieve to map a word embedding to those it is related to for the relations of knowledge graphs (relation conditions).

Such conditions partition relations into three types and also provide a framework to assess loss functions of knowledge graph models.

Any model that satisfies a relation's conditions can represent it if its entity embeddings are set to PMI-based word embeddings, i.e. a solution is known to exist.

Whilst knowledge graph models do not learn the parameters of word embeddings, we show that the better a model's architecture satisfies a relation's conditions, the better its link prediction performance, fitting the premise that similar latent structure is exploited.

Overall, we extend previous understanding of how semantic relations are encoded in relationships between PMI-based word embeddings -generalising from a limited set, e.g. similarity and analogy; we demonstrate commonality between the latent structure learned by PMI-based word embeddings (e.g. W2V) and knowledge graph representation models; and we provide novel insight into knowledge graph models by evaluating their predictive performance.

A CATEGORISING WORDNET RELATIONS Table 7 describes how each WN18RR relation was assigned to its respective category.

Carlson et al., 2010) ), which span our identified relation types (see Table 8 ).

Explanation for the relation category assignment is shown in Table 9 . (Balažević et al., 2019b ).

The Krackhardt hierarchy score measures the proportion of node pairs (x, y) where there exists a directed path x → y, but not y → x; and it takes a value of one for all directed acyclic graphs, and zero for cycles and cliques (Krackhardt, 2014; Balažević et al., 2019a) .

Let M ∈ R n×n be the binary reachability matrix of a directed graph G with n nodes, with M i,j = 1 if there exists a directed path from node i to node j and 0 otherwise.

The Krackhardt hierarchy score of G is defined as:

.

(1)

The symmetry score∈ [−1, 1] (Hubert & Baker, 1979) for a relation matrix R ∈ R de×de is defined as:

where 1 indicates a symmetric and -1 an anti-symmetric matrix.

Tables 10 to 13 shows a sample of the unknown triples (i.e. those formed using the WN18RR entities and relations, but not present in the dataset) for the derivationally_related form (R), instance_hypernym (S) and synset_domain_topic_of (C) relations at a range of probabilities (σ(φ(e s , r, e o )) ≈ {0.4, 0.6, 0.8, 1}), as predicted by each model.

True triples are indicated in bold; instances where a model predicts an entity is related to itself are indicated in blue.

@highlight

Understanding the structure of knowledge graph representation using insight from word embeddings.

@highlight

This paper attempts to understand the latent structure underlying knowledge graph embedding methods, and demonstrates that a model's ability to represent a relation type depends on the model architecture's limitations with respect to relation conditions.

@highlight

This paper proposes a detailed study on the explainability of link prediction (LP) models by utilizing a recent interpretation of word embeddings to provide a better understanding of LPs' model performance.