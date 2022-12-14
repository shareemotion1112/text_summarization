We study the problem of learning representations of entities and relations in knowledge graphs for predicting missing links.

The success of such a task heavily relies on the ability of modeling and inferring the patterns of (or between) the relations.

In this paper, we present a new approach for knowledge graph embedding called RotatE, which is able to model and infer various relation patterns including: symmetry/antisymmetry, inversion, and composition.

Specifically, the RotatE model defines each relation as a rotation from the source entity to the target entity in the complex vector space.

In addition, we propose a novel self-adversarial negative sampling technique for efficiently and effectively training the RotatE model.

Experimental results on multiple benchmark knowledge graphs show that the proposed RotatE model is not only scalable, but also able to infer and model various relation patterns and significantly outperform existing state-of-the-art models for link prediction.

Knowledge graphs are collections of factual triplets, where each triplet (h, r, t) represents a relation r between a head entity h and a tail entity t. Examples of real-world knowledge graphs include Freebase BID0 , Yago (Suchanek et al., 2007) , and WordNet (Miller, 1995) .

Knowledge graphs are potentially useful to a variety of applications such as question-answering BID10 , information retrieval BID30 , recommender systems BID34 , and natural language processing BID31 .

Research on knowledge graphs is attracting growing interests in both academia and industry communities.

Since knowledge graphs are usually incomplete, a fundamental problem for knowledge graph is predicting the missing links.

Recently, extensive studies have been done on learning low-dimensional representations of entities and relations for missing link prediction (a.k.a., knowledge graph embedding) BID3 BID28 BID7 .

These methods have been shown to be scalable and effective.

The general intuition of these methods is to model and infer the connectivity patterns in knowledge graphs according to the observed knowledge facts.

For example, some relations are symmetric (e.g., marriage) while others are antisymmetric (e.g., filiation); some relations are the inverse of other relations (e.g., hypernym and hyponym); and some relations may be composed by others (e.g., my mother's husband is my father).

It is critical to find ways to model and infer these patterns, i.e., symmetry/antisymmetry, inversion, and composition, from the observed facts in order to predict missing links.

Indeed, many existing approaches have been trying to either implicitly or explicitly model one or a few of the above relation patterns BID3 BID29 BID17 Table 1: The score functions f r (h, t) of several knowledge graph embedding models, where ?? denotes the generalized dot product, ??? denotes the Hadamard product, ??? denotes circular correlation, ?? denotes activation function and * denotes 2D convolution.

?? denotes conjugate for complex vectors, and 2D reshaping for real vectors in ConvE model.

TransX represents a wide range of TransE's variants, such as TransH BID29 , TransR BID17 , and STransE BID23 , where g r,i (??) denotes a matrix multiplication with respect to relation r. BID32 BID28 .

For example, the TransE model BID2 , which represents relations as translations, aims to model the inversion and composition patterns; the DisMult model BID32 , which models the three-way interactions between head entities, relations, and tail entities, aims to model the symmetry pattern.

However, none of existing models is capable of modeling and inferring all the above patterns.

Therefore, we are looking for an approach that is able to model and infer all the three types of relation patterns.

In this paper, we propose such an approach called RotatE for knowledge graph embedding.

Our motivation is from Euler's identity e i?? = cos ?? + i sin ??, which indicates that a unitary complex number can be regarded as a rotation in the complex plane.

Specifically, the RotatE model maps the entities and relations to the complex vector space and defines each relation as a rotation from the source entity to the target entity.

Given a triplet (h, r, t), we expect that t = h ??? r, where h, r, t ??? C k are the embeddings, the modulus |r i | = 1 and ??? denotes the Hadamard (element-wise) product.

Specifically, for each dimension in the complex space, we expect that:t i = h i r i , where h i , r i , t i ??? C and |r i | = 1.It turns out that such a simple operation can effectively model all the three relation patterns: symmetric/antisymmetric, inversion, and composition.

For example, a relation r is symmetric if and only if each element of its embedding r, i.e. r i , satisfies r i = e 0/i?? = ??1; two relations r 1 and r 2 are inverse if and only if their embeddings are conjugates: r 2 =r 1 ; a relation r 3 = e i??3 is a combination of other two relations r 1 = e i??1 and r 2 = e i??2 if and only if r 3 = r 1 ??? r 2 (i.e. ?? 3 = ?? 1 + ?? 2 ).

Moreover, the RotatE model is scalable to large knowledge graphs as it remains linear in both time and memory.

To effectively optimizing the RotatE, we further propose a novel self-adversarial negative sampling technique, which generates negative samples according to the current entity and relation embeddings.

The proposed technique is very general and can be applied to many existing knowledge graph embedding models.

We evaluate the RotatE on four large knowledge graph benchmark datasets including FB15k BID3 , WN18 BID3 ), FB15k-237 (Toutanova & Chen, 2015 and WN18RR BID7 .

Experimental results show that the RotatE model significantly outperforms existing state-of-the-art approaches.

In addition, RotatE also outperforms state-of-the-art models on Countries BID4 , a benchmark explicitly designed for composition pattern inference and modeling.

To the best of our knowledge, RotatE is the first model that achieves state-of-the-art performance on all the benchmarks.

2 The p-norm of a complex vector v is defined as v p = p |vi| p .

We use L1-norm for all distancebased models in this paper and drop the subscript of ?? 1 for brevity.

Score Function Symmetry Antisymmetry Inversion Composition SE ??? Wr,1h ??? Wr,2t Predicting missing links with knowledge graph embedding (KGE) methods has been extensively investigated in recent years.

The general methodology is to define a score function for the triplets.

Formally, let E denote the set of entities and R denote the set of relations, then a knowledge graph is a collection of factual triplets (h, r, t), where h, t ??? E and r ??? R. Since entity embeddings are usually represented as vectors, the score function usually takes the form f r (h, t), where h and t are head and tail entity embeddings.

The score function f r (h, t) measures the salience of a candidate triplet (h, r, t).

The goal of the optimization is usually to score true triplet (h, r, t) higher than the corrupted false triplets (h , r, t) or (h, r, t ).

Table 1 summarizes different score functions f r (h, t) in previous state-of-the-art methods as well as the model proposed in this paper.

These models generally capture only a portion of the relation patterns.

For example, TransE represents each relation as a bijection between source entities and target entities, and thus implicitly models inversion and composition of relations, but it cannot model symmetric relations; ComplEx extends DistMult by introducing complex embeddings so as to better model asymmetric relations, but it cannot infer the composition pattern.

The proposed RotatE model leverages the advantages of both.

DISPLAYFORM0 A relevant and concurrent work to our work is the TorusE BID8 model, which defines knowledge graph embedding as translations on a compact Lie group.

The TorusE model can be regarded as a special case of RotatE, where the modulus of embeddings are set fixed; our RotatE is defined on the entire complex space, which has much more representation capacity.

Our experiments show that this is very critical for modeling and inferring the composition patterns.

Moreover, TorusE focuses on the problem of regularization in TransE while this paper focuses on modeling and inferring multiple types of relation patterns.

There are also a large body of relational approaches for modeling the relational patterns on knowledge graphs BID15 BID21 BID6 BID25 .

However, these approaches mainly focus on explicitly modeling the relational paths while our proposed RotatE model implicitly learns the relation patterns, which is not only much more scalable but also provides meaningful embeddings for both entities and relations.

Another related problem is how to effectively draw negative samples for training knowledge graph embeddings.

This problem has been explicitly studied by BID5 , which proposed a generative adversarial learning framework to draw negative samples.

However, such a framework requires simultaneously training the embedding model and a discrete negative sample generator, which are difficult to optimize and also computationally expensive.

We propose a self-adversarial sampling scheme which only relies on the current model.

It does require any additional optimization component, which make it much more efficient.

In this section, we introduce our proposed RotatE model.

We first introduce three important relation patterns that are widely studied in the literature of link prediction on knowledge graphs.

Afterwards, we introduce our proposed RotatE model, which defines relations as rotations in complex vector space.

We also show that the RotatE model is able to model and infer all three relation patterns.

The key of link prediction in knowledge graph is to infer the connection patterns, e.g., relation patterns, with observed facts.

According to the existing literature BID28 BID27 BID9 BID16 , three types of relation patterns are very important and widely spread in knowledge graphs: symmetry, inversion and composition.

We give their formal definition here: Definition 1.

A relation r is symmetric (antisymmetric) if ???x, y r(x, y) ??? r(y, x) ( r(x, y) ??? ??r(y, x) ) A clause with such form is a symmetry (antisymmetry) pattern.

Definition 2.

Relation r 1 is inverse to relation r 2 if ???x, y r 2 (x, y) ??? r 1 (y, x) A clause with such form is a inversion pattern.

Definition 3.

Relation r 1 is composed of relation r 2 and relation r 3 if ???x, y, z r 2 (x, y) ??? r 3 (y, z) ??? r 1 (x, z) A clause with such form is a composition pattern.

According to the definition of the above three types of relation patterns, we provide an analysis of existing models on their abilities in inferring and modeling these patterns.

Specifically, we provide an analysis on TransE, TransX, DistMult, and ComplEx.

3 We did not include the analysis on HolE and ConvE since HolE is equivalent to ComplEx BID11 , and ConvE is a black box that involves two-layer neural networks and convolution operations, which are hard to analyze.

The results are summarized into TAB0 .

We can see that no existing approaches are capable of modeling all the three relation patterns.

In this part, we introduce our proposed model that is able to model and infer all the three types of relation patterns.

Inspired by Euler's identity, we map the head and tail entities h, t to the complex embeddings, i.e., h, t ??? C k ; then we define the functional mapping induced by each relation r as an element-wise rotation from the head entity h to the tail entity t. In other words, given a triple (h, r, t), we expect that: DISPLAYFORM0 and ??? is the Hadmard (or element-wise) product.

Specifically, for each element in the embeddings, we have t i = h i r i .

Here, we constrain the modulus of each element of r ??? C k , i.e., r i ??? C, to be |r i | = 1.

By doing this, r i is of the form e i??r,i , which corresponds to a counterclockwise rotation by ?? r,i radians about the origin of the complex plane, and only affects the phases of the entity embeddings in the complex vector space.

We refer to the proposed model as RotatE due to its rotational nature.

According to the above definition, for each triple (h, r, t), we define the distance function of RotatE as: DISPLAYFORM1 By defining each relation as a rotation in the complex vector spaces, RotatE can model and infer all the three types of relation patterns introduced above.

These results are also summarized into TAB0 .

We can see that the RotatE model is the only model that can model and infer all the three types of relation patterns.

Connection to TransE. From TAB0 , we can see that TransE is able to infer and model all the other relation patterns except the symmetry pattern.

The reason is that in TransE, any symmetric relation will be represented by a 0 translation vector.

As a result, this will push the entities with symmetric relations to be close to each other in the embedding space.

RotatE solves this problem and is a able to model and infer the symmetry pattern.

An arbitrary vector r that satisfies r i = ??1 can be used for representing a symmetric relation in RotatE, and thus the entities having symmetric relations can be distinguished.

Different symmetric relations can be also represented with different embedding vectors.

Figure 1 provides illustrations of TransE and RotatE with only 1-dimensional embedding and shows how RotatE models a symmetric relation.

Negative sampling has been proved quite effective for both learning knowledge graph embedding BID28 and word embedding BID19 .

Here we use a loss function similar to the negative sampling loss BID19 for effectively optimizing distance-based models: DISPLAYFORM0 where ?? is a fixed margin, ?? is the sigmoid function, and (h i , r, t i ) is the i-th negative triplet.

We also propose a new approach for drawing negative samples.

The negative sampling loss samples the negative triplets in a uniform way.

Such a uniform negative sampling suffers the problem of inefficiency since many samples are obviously false as training goes on, which does not provide any meaningful information.

Therefore, we propose an approach called self-adversarial negative sampling, which samples negative triples according to the current embedding model.

Specifically, we sample negative triples from the following distribution: DISPLAYFORM1 where ?? is the temperature of sampling.

Moreover, since the sampling procedure may be costly, we treat the above probability as the weight of the negative sample.

Therefore, the final negative sampling loss with self-adversarial training takes the following form: DISPLAYFORM2 In the experiments, we will compare different approaches for negative sampling.

We evaluate our proposed model on four widely used knowledge graphs.

The statistics of these knowledge graphs are summarized into Table 3 .??? FB15k BID3 ) is a subset of Freebase BID0 , a large-scale knowledge graph containing general knowledge facts.

BID27 showed that almost 81% of the test triplets (x, r, y) can be inferred via a directly linked triplet (x, r , y) or (y, r , x).

Therefore, the key of link prediction on FB15k is to model and infer the symmetry/antisymmetry and inversion patterns.??? WN18 BID3 ) is a subset of WordNet BID20 , a database featuring lexical relations between words.

This dataset also has many inverse relations.

So the main relation patterns in WN18 are also symmetry/antisymmetry and inversion.??? FB15k-237 BID27 ) is a subset of FB15k, where inverse relations are deleted.

Therefore, the key of link prediction on FB15k-237 boils down to model and infer the symmetry/antisymmetry and composition patterns.??? WN18RR BID7 ) is a subset of WN18.

The inverse relations are deleted, and the main relation patterns are symmetry/antisymmetry and composition.

Hyperparameter Settings.

We use Adam BID14 as the optimizer and fine-tune the hyperparameters on the validation dataset.

The ranges of the hyperparameters for the grid search are set as follows: embedding dimension 5 k ??? {125, 250, 500, 1000}, batch size b ??? {512, 1024, 2048}, self-adversarial sampling temperature ?? ??? {0.5, 1.0}, and fixed margin ?? ??? {3, 6, 9, 12, 18, 24, 30}. Both the real and imaginary parts of the entity embeddings are uniformly initialized, and the phases of the relation embeddings are uniformly initialized between 0 and 2??.

No regularization is used since we find that the fixed margin ?? could prevent our model from over-fitting.

Evaluation Settings.

We evaluate the performance of link prediction in the filtered setting: we rank test triples against all other candidate triples not appearing in the training, validation, or test set, where candidates are generated by corrupting subjects or objects: (h , r, t) or (h, r, t ).

Mean Rank (MR), Mean Reciprocal Rank (MRR) and Hits at N (H@N) are standard evaluation measures for these datasets and are evaluated in our experiments.

Baseline.

Apart from RotatE, we propose a variant of RotatE as baseline, where the modulus of the entity embeddings are also constrained: |h i | = |t i | = C, and the distance function is thus 2C sin DISPLAYFORM0 (See Equation 17 at Appendix F for a detailed derivation).

In this way, we can investigate how RotatE works without modulus information and with only phase information.

We refer to the baseline as pRotatE. It is obvious to see that pRotatE can also model and infer all the three relation patterns.

We compare RotatE to several state-of-the-art models, including TransE BID3 , DistMult BID32 , ComplEx BID28 , HolE BID24 , and ConvE BID7 , as well as our baseline model pRotatE, to empirically show the importance of modeling and inferring the relation patterns for the task of predicting missing links.

TAB4 summarizes our results on FB15k and WN18.

We can see that RotatE outperforms all the state-of-the-art models.

The performance of pRotatE and RotatE are similar on these two datasets.

Moreover, the performance of these models on different datasets is consistent with our analysis on the three relation patterns TAB0 :??? On FB15K, the main relation patterns are symmetry/antisymmetry and inversion.

We can see that ComplEx performs well while TransE does not perform well since ComplEx can infer both symmetry/antisymmetry and inversion patterns while TransE cannot infer symmetry pattern.

Surprisingly, DistMult achieves good performance on this dataset although it cannot model the antisymmetry and inversion patterns.

The reason is that for most of the relations in FB15K, the types of head entities and tail entities are different.

Although DistMult gives the same score to a true triplet (h, r, t) and its opposition triplet (t, r, h), (t, r, h) is usually impossible to be valid since the entity type of t does not match the head entity type of h. For example, DistMult assigns the same score to (Obama, nationality, USA) and (USA, nationality, Obama).

But (USA, nationality, Obama) can be simply predicted as false since USA cannot be the head entity of the relation nationality.??? On WN18, the main relation patterns are also symmetry/antisymmetry and inversion.

As expected, ComplEx still performs very well on this dataset.

However, different from the results on FB15K, the performance of DistMult significantly decreases on WN18.

The reason is that DistMult cannot model antisymmetry and inversion patterns, and almost all the entities in WN18 are words and belong to the same entity type, which do not have the same problem as FB15K.??? On FB15k-237, the main relation pattern is composition.

We can see that TransE performs really well while ComplEx does not perform well.

The reason is that, as discussed before, TransE is able to infer the composition pattern while ComplEx cannot infer the composition pattern.??? On WN18RR, one of the main relation patterns is the symmetry pattern since almost each word has a symmetric relation in WN18RR, e.g., also see and similar to.

TransE does not well on this dataset since it is not able to model the symmetric relations.

ComplEx ConvE RotatE S1 1.00 ?? 0.00 0.97 ?? 0.02 1.00 ?? 0.00 1.00 ?? 0.00 S2 0.72 ?? 0.12 0.57 ?? 0.10 0.99 ?? 0.01 1.00 ?? 0.00 S3 0.52 ?? 0.07 0.43 ?? 0.07 0.86 ?? 0.05 0.95 ?? 0.00 Table 6 : Results on the Countries datasets.

Other results are taken from BID7 .

Figure 2: Histograms of relation embedding phases {?? r,i } (r i = e i??r,i ), where for 1 represents relation award nominee/award nominations./award/award nomination/nominated for, winner represents relation award category/winners./award/award honor/award winner and for 2 represents award category/nominees./award/award nomination/nominated for.

The symmetry, inversion and composition pattern is represented in Figure 2a , 2c and 2g, respectively.

We also evaluate our model on the Countries dataset BID4 BID24 , which is carefully designed to explicitly test the capabilities of the link prediction models for composition pattern modeling and inferring.

It contains 2 relations and 272 entities (244 countries, 5 regions and 23 subregions).

Unlike link prediction on general knowledge graphs, the queries in Countries are of the form locatedIn(c, ?), and the answer is one of the five regions.

The Countries dataset has 3 tasks, each requiring inferring a composition pattern with increasing length and difficulty.

For example, task S2 requires inferring a relatively simpler composition pattern: DISPLAYFORM0 while task S3 requires inferring the most complex composition pattern: DISPLAYFORM1 In Table 6 , we report the results with respect to the AUC-PR metric, which is commonly used in the literature.

We can see that RotatE outperforms all the previous models.

The performance of RotatE is significantly better than other methods on S3, which is the most difficult task.

In this section, we verify whether the relation patterns are implicitly represented by RotatE relation embeddings.

We ignore the specific positions in the relation embedding ?? r and plot the histogram of the phase of each element in the relation embedding, i.e., {?? r,i }

.Symmetry pattern requires the symmetric relations to have property r ??? r = 1, and the solution is r i = ??1.

We investigate the relation embeddings from a 500-dimensional RotatE trained on WN18.

Figure 2a gives the histogram of the embedding phases of a symmetric relation similar to.

We can find that the embedding phases are either ?? (r i = ???1) or 0, 2?? (r i = 1 BID5 1.00 ?? 0.00 1.00 ?? 0.00 0.95 ?? 0.00 model does infer and model the symmetry pattern.

Figure 2b is the histogram of relation hypernym, which shows that the embedding of a general relation does not have such a ??1 pattern.

Inversion pattern requires the embeddings of a pair of inverse relations to be conjugate.

We use the same RotatE model trained on WN18 for an analysis.

Figure 2c illustrates the element-wise addition of the embedding phases from relation r 1 = hypernym and its inversed relation r 2 = hyponym.

All the additive embedding phases are 0 or 2??, which represents that r 1 = r ???12 .

This case shows that the inversion pattern is also inferred and modeled in the RotatE model.

Composition pattern requires the embedding phases of the composed relation to be the addition of the other two relations.

Since there is no significant composition pattern in WN18, we study the inference of the composition patterns on FB15k-237, where a 1000-dimensional RotatE is trained.

Figure 2d -2g illustrate such a r 1 = r 2 ??? r 3 case, where ?? 2,i + ?? 3,i = ?? 1,i or ?? 2,i + ?? 3,i = ?? 1,i + 2??.

More results of implicitly inferring basic patterns are presented in the appendix.

In this part, we compare different negative sampling techniques including uniform sampling, our proposed self-adversarial technique, and the KBGAN model BID5 , which aims to optimize a generative adversarial network to generate the negative samples.

We re-implement a 50-dimension TransE model with the margin-based ranking criterion that was used in BID5 , and evaluate its performance on FB15k-237, WN18RR and WN18 with self-adversarial negative sampling.

Table 7 summarizes our results.

We can see that self-adversarial sampling is the most effective negative sampling technique.

One may argue that the contribution of RotatE comes from the self-adversarial negative sampling technique.

In this part, we conduct further experiments on TransE and ComplEx in the same setting as RotatE to make a fair comparison among the three models.

TAB8 shows the results of TransE and ComplEx trained with the self-adversarial negative sampling technique on FB15k and FB15k-237 datasets, where a large number of relations are available.

In addition, we evaluate these three models on the Countries dataset, which explicitly requires inferring the composition pattern.

We also provide a detailed ablation study on TransE and RotatE in the appendix.

From TAB8 , we can see that similar results are observed as TAB4 and 5.

The RotatE model achieves the best performance on both FB15k and FB15k-237, as it is able to model all the three relation patterns.

The TransE model does not work well on the FB15k datasets, which requires modeling the symmetric pattern; the ComplEx model does not work well on FB15k-237, which requires modeling the composition pattern.

The results on the Countries dataset are a little bit different, where the TransE model slightly outperforms RoateE on the S3 task.

The reason is that Relation Category 1-to-1 1-to-N N-to-1 N-to-N 1-to-1 1-to-N N-to-1 N-to-N Table 9 : Experimental results on FB15k by relation category.

The first three rows are taken from BID12 .

The rest of the results are from RotatE trained with the self-adversarial negative sampling technique.the Countries datasets do not have the symmetric relation between different regions, and all the three tasks in the Countries datasets only require inferring the region for a given city.

Therefore, the TransE model does not suffer from its inability of modeling symmetric relations.

For ComplEx, we can see that it does not perform well on Countries since it cannot infer the composition pattern.

We also did some further investigation on the performance of RotatE on different relation categories: one-to-many, many-to-one, and many-to-many relations 6 .

The results of RotatE on different relation categories on the data set FB15k are summarized into Table 9 .

We also compare an additional approach KG2E KL BID12 , which is a probabilistic framework for knowledge graph embedding methods and aims to model the uncertainties of the entities and relations in knowledge graphs with the TransE model.

We also summarize the statistics of different relation categories into TAB11 in the appendix.

We can see that besides the one-to-one relation, the RotatE model also performs quite well on the non-injective relations, especially on many-to-many relations.

We also notice that the probabilistic framework KG2E KL(bern) BID12 is quite powerful, which consistently outperforms its corresponding knowledge graph embedding model, showing the importance of modeling the uncertainties in knowledge graphs.

We leave the work of modeling the uncertainties in knowledge graphs with RotatE as our future work.

We have proposed a new knowledge graph embedding method called RotatE, which represents entities as complex vectors and relations as rotations in complex vector space.

In addition, we propose a novel self-adversarial negative sampling technique for efficiently and effectively training the RotatE model.

Our experimental results show that the RotatE model outperforms all existing state-of-theart models on four large-scale benchmarks.

Moreover, RotatE also achieves state-of-the-art results on a benchmark that is explicitly designed for composition pattern inference and modeling.

A deep investigation into RotatE relation embeddings shows that the three relation patterns are implicitly represented in the relation embeddings.

In the future, we plan to evaluate the RotatE model on more datasets and leverage a probabilistic framework to model the uncertainties of entities and relations.

No existing models are capable of modeling all the three relation patterns.

For example, TransE cannot model the symmetry pattern because it would yield r = 0 for symmetric relations; TransX can infer and model the symmetry/antisymmetry pattern when g r,1 = g r,2 , e.g. in TransH BID29 , but cannot infer inversion and composition as g r,1 and g r,2 are invertible matrix multiplications; due to its symmetric nature, DistMult is difficult to model the asymmetric and inversion pattern; ComplEx addresses the problem of DisMult and is able to infer both the symmetry and asymmetric patterns with complex embeddings.

Moreover, it can infer inversion rules because the complex conjugate of the solution to arg max r Re( x, r, y ) is exactly the solution to arg max r Re( y, r, x ).

However, ComplEx cannot infer composition rules, since it does not model a bijection mapping from h to t via relation r. These concerns are summarized in TAB0 .B PROOF OF LEMMA 1Proof.

if r(x, y) and r(y, x) hold, we have y = r ??? x ??? x = r ??? y ??? r ??? r = 1 Otherwise, if r(x, y) and ??r(y, x) hold, we have DISPLAYFORM0 Proof.

if r 1 (x, y) and r 2 (y, x) hold, we have DISPLAYFORM1 Proof.

if r 1 (x, z), r 2 (x, y) and r 3 (y, z) hold, we have DISPLAYFORM2

A useful property for RotatE is that the inverse of a relation can be easily acquired by complex conjugate.

In this way, the RotatE model treats head and tail entities in a uniform way, which is potentially useful for efficient 1-N scoring BID7 : DISPLAYFORM0 Moreover, considering the embeddings in the polar form, i.e., h i = m h,i e i?? h,i , r i = e i??r,i , t i = m t,i e i??t,i , we can rewrite the RotatE distance function as: DISPLAYFORM1 This equation provides two interesting views of the model:(1) When we constrain the modulus m h,i = m t,i = C, the distance function is reduced to 2C sin DISPLAYFORM2 .

We can see that this is very similar to the distance function of TransE: Table 11 : Results of several models evaluated on the YAGO3-10 datasets.

Other results are taken from BID7 .

DISPLAYFORM3

Proof.

By further restricting |h i | = |t i | = C, we can rewrite h, r, t by DISPLAYFORM0 r = e i??r = cos ?? r + i sin ?? r (10) DISPLAYFORM1 Therefore, we have DISPLAYFORM2 DISPLAYFORM3 If the embedding of (h, r, t) in TransE is h , r , t , let ?? h = ch , ?? r = cr , ?? t = ct and C = 1/c , we have DISPLAYFORM4 YAGO3-10 is a subset of YAGO3 BID18 , which consists of entities that have a minimum of 10 relations each.

It has 123,182 entities and 37 relations.

Most of the triples deal with descriptive attributes of people, such as citizenship, gender, profession and marital status.

Table 11 shows that the RotatE model also outperforms state-of-the-art models on YAGO3-10.

We list the best hyperparameter setting of RotatE w.r.t the validation dataset on several benchmarks in Table 13 : Results of ablation study on FB15k-237, where "adv" represents "self-adversarial".I ABLATION STUDY Table 13 shows our ablation study of self-adversarial sampling and negative sampling loss on FB15k-237.

We also re-implement a 1000-dimension TransE and do ablation study on it.

From the table, We can find that self-adversarial sampling boosts the performance for both models, while negative sampling loss is only effective on RotatE; in addition, our re-implementation of TransE also outperforms all the state-of-the-art models on FB15k-237.

In TAB4 , We provide the average and variance of the MRR results on FB15k, WN18, FB15k-237 and WN18RR.

Both the average and the variance is calculated by three runs of RotatE with difference random seeds.

We can find that the performance of RotatE is quite stable for different random initialization.

We provide more histograms of embedding phases in FIG2 -5.

<|TLDR|>

@highlight

A new state-of-the-art approach for knowledge graph embedding.

@highlight

Presents a neural link prediction scoring function that can infer symmetry, anti-symmetry, inversion and composition patterns of relations in a knowledge base.

@highlight

This paper proposes an approach to knowledge graph embedding by modeling relations as rotations in the complex vector space.

@highlight

Proposes a method for graph embedding to be used for link prediction