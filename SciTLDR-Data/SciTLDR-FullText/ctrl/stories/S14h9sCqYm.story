Aligning knowledge graphs from different sources or languages, which aims to align both the entity and relation, is critical to a variety of applications such as knowledge graph construction and question answering.

Existing methods of knowledge graph alignment usually rely on a large number of aligned knowledge triplets to train effective models.

However, these aligned triplets may not be available or are expensive to obtain for many domains.

Therefore, in this paper we study how to design fully-unsupervised methods or weakly-supervised methods, i.e., to align knowledge graphs without or with only a few aligned triplets.

We propose an unsupervised framework based on adversarial training, which is able to map the entities and relations in a source knowledge graph to those in a target knowledge graph.

This framework can be further seamlessly integrated with existing supervised methods, where only a limited number of aligned triplets are utilized as guidance.

Experiments on real-world datasets prove the effectiveness of our proposed approach in both the weakly-supervised and unsupervised settings.

Knowledge graphs represent a collection of knowledge facts and are quite popular in the real world.

Each fact is represented as a triplet (h, r, t), meaning that the head entity h has the relation r with the tail entity t. Examples of real-world knowledge graphs include instances which contain knowledge facts from general domain in different languages (Freebase 1 , DBPedia BID2 , Yago BID19 , WordNet 2 ) or facts from specific domains such as biomedical ontology (UMLS 3 ).

Knowledge graphs are critical to a variety of applications such as question answering BID4 ) and semantic search BID13 ), which are attracting growing interest recently in both academia and industry communities.

In practice, each knowledge graph is usually constructed from a single source or language, the coverage of which is limited.

To enlarge the coverage and construct more unified knowledge graphs, a natural idea is to integrate multiple knowledge graphs from different sources or languages BID0 ).

However, different knowledge graphs use distinct symbol systems to represent entities and relations, which are not compatible.

As a result, it is necessary to align entities and relations across different knowledge graphs (a.k.a., knowledge graph alignment) before integrating them.

Indeed, there are some recent studies focusing on aligning entities and relations from a source knowledge graph to a target knowledge graph ( BID23 ; BID6 ; BID7 ).

These methods typically represent entities and relations in a low-dimensional space, and meanwhile learn a mapping function to align entities and relations from the source knowledge graph to the target one.

However, these methods usually rely on a large number of aligned triplets as labeled data to train effective alignment models.

In reality, the aligned triplets may not be available or can be expensive to obtain, making existing methods fail to achieve satisfactory results.

Therefore, we are seeking for an unsupervised or weakly-supervised approach, which is able to align knowledge graphs with a few or even without labeled data.

In this paper, we propose an unsupervised approach for knowledge graph alignment with the adversarial training framework BID11 .

Our proposed approach aims to learn alignment functions, i.e., P e (e tgt |e src ) and P r (r tgt |r src ), to map the entities and relations (e src and r src ) from the source knowledge graph to those (e tgt and r tgt ) in the target graph, without any labeled data.

Towards this goal, we notice that we can align each triplet in the source knowledge graph with one in the target knowledge graph by aligning the head/tail entities and relation respectively.

Ideally, the optimal alignment functions would align all the source triplets to some valid triplets (i.e., triplets expressing true facts).

Therefore, we can enhance the alignment functions by improving the plausibility of the aligned triplets.

With this intuition, we train a triplet discriminator to distinguish between the real triplets in the target knowledge graph and those aligned from the source graph, which provides a reward function to measure the plausibility of a triplet.

Meanwhile, the alignment functions are optimized to maximize the reward.

The above process naturally forms an adversarial training procedure BID11 ).

By alternatively optimizing the alignment functions and the discriminator, the discriminator can consistently enhance the alignment functions.

However, the above approach may suffer from the problem of mode collapse BID17 ).

Specifically, many entities in the source knowledge graph may be aligned to only a few entities in the target knowledge graph.

This problem can be addressed if the aggregated posterior entity distribution e src P e (e tgt |e src )P (e src ) derived by the alignment functions matches the prior entity distribution P (e tgt ) in the target knowledge graph.

Therefore, we match them with another adversarial training framework, which shares similar idea with adversarial auto-encoders BID16 ).The whole framework can also be seamlessly integrated with existing supervised methods, in which we can use a few aligned entities or relations as guidance, yielding a weakly-supervised approach.

Our approach can be effectively optimized with stochastic gradient descent, where the gradient for the alignment functions is calculated by the REINFORCE algorithm (Williams (1992)).

We conduct extensive experiments on several real-world knowledge graphs.

Experimental results prove the effectiveness of our proposed approach in both the weakly-supervised and unsupervised settings.

Our work is related to knowledge graph embedding, that is, embedding knowledge graphs into lowdimensional spaces, in which each entity and relation is represented as a low-dimensional vector (a.k.a., embedding).

A variety of knowledge graph embedding approaches have been proposed BID3 ; BID21 ), which can effectively preserve the semantic similarities of entities and relations into the learned embeddings.

We treat these techniques as tools to learn entity and relation embeddings, which are further used as features for knowledge graph alignment.

In literature, there are also some studies focusing on knowledge graph alignment.

Most of them perform alignment by considering contextual features of entities and relations, such as their names BID15 ) or text descriptions BID8 ; BID20 ; BID21 ).

However, such contextual features are not always available, and therefore these methods cannot generalize to most knowledge graphs.

In this paper, we consider the most general case, in which only the triplets in knowledge graphs are used for alignment.

The studies most related to ours are BID23 and BID6 .

Similar to our approach, they treat the entity and relation embeddings as features, and jointly train an alignment model.

However, they totally rely on the labeled data (e.g., aligned entities or relations) to train the alignment model, whereas our approach incorporates additional signals by using adversarial training, and therefore achieves better results in the weakly-supervised and unsupervised settings.

More broadly, our work belongs to the family of domain alignment, which aims at mapping data from one domain to data in the other domain.

With the success of generative adversarial networks BID11 ), many researchers have been bringing the idea to domain alignment, getting impressive results in many applications, such as image-to-image translation BID24 ; BID25 ), word-to-word translation BID9 ) and text style transfer BID18 ).

They typically train a domain discriminator to distinguish between data points from different domains, and then the alignment function is optimized by fooling the discriminator.

Our approach shares similar idea, but is designed with some specific intuitions in knowledge graphs.

Source Knowledge Graph

Aligned Knowledge Graph Triplet Discriminator

Figure 1: Framework overview.

By applying the alignment functions to the triplets in the source knowledge graph, we obtain an aligned knowledge graph.

The alignment functions are learned through two GANs.(1) We expect all triplets in the aligned knowledge graph are valid, therefore we train a triplet discriminator to distinguish between valid and invalid triplets, and further use it to facilitate the alignment functions.

FORMULA1 We also expect the entity distribution in the aligned knowledge graph matches the one in the target knowledge graph, which is achieved with another GAN.

Definition 1 (KNOWLEDGE GRAPH.)

A knowledge graph is denoted as G = (E, R, T ), where E is a set of entities, R is a set of relations and T is a set of triplets.

Each triplet (h, r, t) consists of a head entity h, a relation r and a tail entity t, meaning that entity h has relation r with entity t.

In practice, the coverage of each individual knowledge graph is usually limited, since it is typically constructed from a single source or language.

To construct knowledge graphs with broader coverage, a straightforward way is to integrate multiple knowledge graphs from different sources or languages.

However, each knowledge graph uses a unique symbol system to represent entities and relations, which is not compatible with other knowledge graphs.

Therefore, a prerequisite for knowledge graph integration is to align entities and relations across different knowledge graphs (a.k.a., knowledge graph alignment).

In this paper, we study how to align entities and relations from a source knowledge graph to those in a target knowledge graph, and the problem is formally defined below:Definition 2 (KNOWLEDGE GRAPH ALIGNMENT.)

Given a source knowledge graph G src = (E src , R src , T src ) and a target knowledge graph G tgt = (E tgt , R tgt , T tgt ), the problem aims at learning an entity alignment function P e and a relation alignment function P r .

Given an entity e src in the source knowledge graph and an entity e tgt in the target knowledge graph, P e (e tgt |e src ) gives the probability that e src aligns to e tgt .

Similarly, for a source relation r src and a target relation r tgt , P r (r tgt |r src ) gives the probability that r src aligns to r tgt .

In this paper we propose an unsupervised approach to learning the alignment functions, i.e., P e (e tgt |e src ) and P r (r tgt |r src ), for knowledge graph alignment.

To learn them without any supervision, we notice that we can align each triplet in the source knowledge graph with one in the target knowledge graph by aligning the head/tail entities and relation respectively.

For an ideal alignment model, all the aligned triplets should be valid ones (i.e., triplets expressing true facts).

As a result, we can improve the alignment functions by raising the plausibility of the aligned triplets.

With the intuition, our approach trains a triplet discriminator to distinguish between valid triplets and other ones.

Then we build a reward function from the discriminator to facilitate the alignment functions.

However, using the triplet discriminator alone may cause the problem of mode collapse.

More specifically, many entities in the source knowledge graph are aligned to only a few entities in the target knowledge graph.

This problem can be addressed if the aggregated posterior distribution of entities derived by the alignment functions matches the prior entity distribution from the target knowledge graph.

Therefore, we follow the idea in adversarial auto-encoders BID16 ), and leverage another adversarial training framework to regularize the distribution.

The above strategies yield an unsupervised approach.

However, in many cases, the structures of the source and target knowledge graphs (e.g., entity and triplet distributions) can be very different, making our unsupervised approach unable to perform effective alignment.

In such cases, we can integrate our approach with existing supervised methods, and use a few labeled data as guidance, which further yields a weakly-supervised approach.

Formally, our approach starts by learning entity and relation embeddings with existing knowledge graph embedding techniques, which are denoted as {x e src } e src ∈E src , {x e tgt } e tgt ∈E tgt and {x r src } r src ∈R src , {x r tgt } r tgt ∈R tgt .

The learned embeddings preserve the semantic correlations of entities and relations, hence we treat them as features and build our alignment functions on top of them.

Specifically, we define the probability that a source entity e src or relation r src aligns to a target entity e tgt or relation r tgt as follows: DISPLAYFORM0 where γ is a temperature parameter, Z is a normalization term.

W is a linear projection matrix, which maps an embedding in the source knowledge graph (e.g., x e src ) to one in the target graph (e.g., W x e src ), so that we can perform alignment by calculating the distance between the mapped source embeddings (e.g., W x e src ) and the embeddings in the target graph (e.g., x e tgt ).

Note that W is the only parameter to be learned, and it is shared across the entity and relation alignment functions.

We also try independent projection matrices or nonlinear projections, but get inferior results.

In the following chapters, we first briefly introduce the method for learning entity and relation embeddings (Section 4.1).

Then, we introduce how we leverage the triplet discriminator (Section 4.2) and the regularization mechanism (Section 4.3) to facilitate training the alignment functions.

Afterwards, we introduce a simple supervised method as an example, to illustrate how to incorporate labeled data (Section 4.4).

Finally, we introduce our optimization algorithm (Section 4.5).

In this paper, we leverage the TransE algorithm BID3 ) for entity and relation embedding learning, due to its simplicity and effectiveness in a wide range of datasets.

In general, we can also use other knowledge graph embedding algorithms as well.

Given a triplet t = (e h , r, e t ), TransE defines its score as follows: DISPLAYFORM0 Then the model is trained by maximizing the margin between the scores of real triplets and random triplets, and the objective function is given below: DISPLAYFORM1 where T is the set of real triplets in the knowledge graph, T is the set of random triplets, and m is a parameter controlling the margin.

By defining the alignment functions for entity and relation (Eqn.

1), we are able to align each triplet in the source knowledge graph to the target knowledge graph by aligning the entities and relation respectively.

An ideal alignment function would align all the source triplets to some valid triplets.

Therefore, we can enhance the alignment functions by raising the plausibility of the aligned triplets.

Ideally, we would wish that all the aligned triplets are valid ones.

Towards this goal, we train a triplet discriminator to distinguish between valid triplets and other ones.

Then the discriminator is used to define different reward functions for guiding the alignment functions.

In our approach, we train the discriminator by treating the real triplets in knowledge graphs as positive examples, and the aligned triplets generated by our approach as negative examples.

Following existing studies BID11 ), we define the objective function below: DISPLAYFORM0 where t ∼ A(t src ) is a triplet aligned from t src and A is defined in Eqn.

4.

D t is the triplet discriminator, which concatenates the embeddings of the head/tail entities and relation in a triplet t, and further predicts the probability that t is a valid triplet.

Based on the discriminator, we can construct a scalar-to-scalar reward function R to measure the plausibility of a triplet.

Then the alignment functions can be trained by maximizing the reward: DISPLAYFORM1 There are several ways to define the reward function R, which essentially yields different adversarial training frameworks.

For example, BID11 and BID14 Besides, we may also leverage R(x) = x, which is the first-order Taylor's expansion of − log(1−x) at x = 1 and has a limited range when x ∈ (0, 1).

All different reward functions have the same optimal solution, i.e, the derived distribution of the aligned triplets matching the real triplet distribution in the target knowledge graph.

In practice, these reward functions may have different variance, and we empirically compare them in the experiments TAB6 ).During optimization, the gradient with respect to the alignment functions cannot be calculated directly, as the triplets sampled from the alignment functions are discrete variables.

Therefore, we leverage the REINFORCE algorithm (Williams (1992)), which calculates the gradient as follows: DISPLAYFORM2 where P (t|t src ) = P e (e h |e src h )P r (r|r src )P e (e t |e src t ) with t = (e h , r, e t ), t src = (e src h , r src , e src t ).

Although the triplet discriminator provides effective reward to the alignment functions, many entities in the source knowledge graph can be aligned to only a few entities in the target knowledge graph.

Such problems can be solved by constraining the aggregated posterior entity distribution derived by the alignment functions to match the prior entity distribution in the target knowledge graph.

Formally, the aggregated posterior distribution of entities is given below: DISPLAYFORM0 where P (e src ) is the entity distribution in the source knowledge graph.

We expect this distribution to match the prior distribution P (e tgt ), which is the entity distribution in the target knowledge graph.

Following BID16 , we regularize the distribution with another adversarial training framework BID11 ).

During training, an entity discriminator D e is learned to distinguish between the posterior and prior distributions using the following objective function: DISPLAYFORM1 where D e takes the embedding of an entity as features to predict the probability that the entity is sampled from prior distribution P (e tgt ).

To enforce the posterior distribution to match the prior distribution, the entity alignment function is trained to fool the discriminator by maximizing the following objective: DISPLAYFORM2 where R is the same reward function as used in the triplet discriminator (Eqn.

6), and the gradient for the alignment functions can be similarly calculated with the REINFORCE algorithm.

The above sections introduce an unsupervised approach to knowledge graph alignment.

In many cases, the source and target knowledge graphs may have very different structures (e.g., entity or triplet distributions), making our approach fail to perform effective alignment.

In these cases, we can integrate our approach with any supervised methods, and leverage a few labeled data (e.g., aligned entity or relation pairs) as guidance, which yields a weakly-supervised approach.

In this section, we introduce a simple yet effective method to show how to utilize the labeled data.

Suppose we are given some aligned entity pairs, and the aligned relation pairs can be handled in a similar way.

We define our objective function as follows:O L = E (e src ,e tgt )∈S log P e (e tgt |e src ) − λH(P e (e tgt |e src )),where S is the set of aligned entity pairs, e src and e tgt are random variables of entities in the source and target knowledge graphs, H is the entropy of a distribution.

The first term corresponds to a softmax classifier, which aims at maximizing the probability of aligning a source entity to the ground-truth target entity.

The second term minimizes the entropy of the probability distribution calculated by the alignment function, which encourages the alignment function to make confident predictions.

Such an entropy minimization strategy is used in many semi-supervised learning studies BID12 ).

We leverage the stochastic gradient descent algorithm for optimization.

In practice, we find that first pre-training the alignment functions with the given labeled data (Eqn.

11), then fine-tuning them with the triplet discriminator (Eqn.

6) and the regularization mechanism (Eqn.

8) leads to better performance, compared with jointly training all of them TAB7 ).

Consequently, we adopt the pre-training and fine-tuning framework for optimization, which is summarized in Alg.

1.

Update the triplet discriminator D t and the alignment functions with Eqn.

5 6.

Update the entity discriminator D e and the alignment functions with Eqn.

9 10.

6: end while

In experiment, we use four datasets for evaluation.

In FB15k-1 and FB15k-2, the knowledge graphs have very different triplets, which can be seen as constructed from different sources; in WK15k(enfr) and WK15k(en-de), the knowledge graphs are from different languages.

The statistics are summarized in TAB2 .

Following existing studies BID23 ; BID6 ), we consider the task of entity alignment, and three different settings are considered, including supervised, weakly-supervised and unsupervised settings.

Hit@1 and Hit@10 are reported.

• FB15k-1, FB15k-2: Following Zhu et al. (2017a), we construct two datasets from the FB15k dataset BID3 ).

In FB15k-1, the two knowledge graphs share 50% triplets, and in FB15k-2 10% triplets are shared.

According to the study, we use 5000 and 500 aligned entity pairs as labeled data in FB15k-1 and FB15k-2 respectively, and the rest for evaluation.• WK15k(en-fr): A bi-lingual (English and French) dataset in BID6 .

Some aligned triplets are provided as labeled data, and some aligned entity pairs as test data.

The labeled data and test data have some overlaps, so we delete the overlapped pairs from labeled data.• WK15k(en-de): A bi-lingual (English and German) dataset used in BID6 .

The dataset is similar to WK15k(en-fr), so we perform preprocessing in the same way.

Compared Algorithms FORMULA0 ): A supervised method for word translation, which learns the translation in a bootstrapping way.

We apply the method on the pre-trained entity and relation embeddings to perform knowledge graph alignment.

(4) UWT BID9 ): An unsupervised word translation method, which leverages adversarial training and a refinement strategy.

We apply the method to the entity and relation embeddings to perform alignment.

(5) KAGAN-sup: The supervised method introduced in Section 4.4, which is simple but effective, and can be easily integrated with our unsupervised approach.

(6) KAGAN: Our proposed Knowledgegraph Alignment GAN, which leverages the labeled data for pre-training, and then fine-tunes the model with the triplet discriminator and the regularization mechanism.

(7) KAGAN-t: A variant with only the triplet GAN, which first performs pre-training with the labeled data, and then performs fine-tuning with the triplet discriminator. (8) KAGAN-e: A variant with only the entity GAN, which first pre-trains with the labeled data, and then fine-tunes with the regularization mechanism.

The dimension of entity and relation embeddings is set as 512 for all compared methods.

For our proposed approach, λ is set as 1, the temperature parameter γ is set as 1, and the reward function is set as x by default.

SGD is used for optimization.

The learning rate is set as 0.1 for pre-training and 0.001 for fine-tuning.

10% labeled pairs are treated as the validation set.

The training process is terminated if the performance on the validation set drops.

For the compared methods, the parameters are chosen according to the performance on the validation set.

The main results are presented in TAB3 .

In the supervised setting, our approach significantly outperforms all the compared methods.

On the FB15k-1 and FB15k-2 datasets, without using any labeled data, our approach already achieves close results as in the supervised setting.

On the WK15k datasets under the weakly-supervised setting, our performance is comparable or even superior to the performance of other methods in the supervised setting, but with much fewer labeled data (about 13% in WK15k(en-fr) and 1% in WK15k(en-de)).

Overall, our approach is quite effective in the weakly-supervised and unsupervised settings, outperforming all the baseline approaches.

To understand the effect of each part in our approach, we further conduct some ablation studies.

Table 3 presents the results in the supervised setting.

Both the triplet discriminator (KAGAN-t) and the regularization mechanism (KAGAN-e) improves the pre-trained alignment models (KAGAN-pre).

Combining them (KAGAN) leads to even better results.

TAB5 gives the results in the unsupervised (FB15k-2) and weakly-supervised (WK15k-fr2en, WK15k-de2en) settings.

On the FB15k-2 dataset, using the regularization mechanism alone (KAGAN-e) already achieves impressive results.

This is because the source and target knowledge graphs in FB15k-2 share similar structures, and our regularization mechanism can effectively leverage such similarity for alignment.

However, the performance of using only the triplet discriminator (KAGAN-t) is very poor, which is caused by the problem of mode collapse.

The problem is effectively solved by integrating the approach with the regularization mechanism (KAGAN), which achieves the best results in all cases.

Comparison of the reward functions.

In our approach, we can choose different reward functions, leading to different adversarial training frameworks.

These frameworks have the same optimal solutions, but with different variance.

In this section we compare them on the WK15k datasets, and the results of Hit@1 are presented in TAB6 .

We notice that all reward functions lead to significant improvement compared with using no reward.

Among them, x 1−x and x obtain the best results.

Comparison of the optimization methods.

During training, our approach fixes the entity/relation embeddings, and uses a pre-training and fine-tuning framework for optimization.

In this section, we compare the framework with some variants, and the results of Hit@1 are presented in TAB7 .

We see that our framework (pre-training and fine-tuning) outperforms the joint training framework.

Besides, fine-tuning entity/relation embeddings yields worse results than fixing them during training.

Case study.

In this section, we present some visualization results to intuitively show the effect of the triplet discriminator and regularization mechanism in our approach.

We consider the unsupervised setting on the FB15k-2 dataset, and leverage the PCA algorithm to visualize certain embeddings.

Figure 2 compares the entity embeddings obtained with and without the regularization mechanism, where red is for the mapped source entity embeddings (W x e src ) , and green for the target embeddings (x e tgt ).

We see that without the mechanism, many entities from the source knowledge graph are mapped to a small region (the red region), leading to the problem of mode collapse.

The problem is addressed with the regularization mechanism.

FIG2 compares the triplet embeddings obtained with and without the triplet discriminator, where the triplet embedding is obtained by concatenating the entity and relation embeddings in a triplet.

Red color is for triplets aligned from the source knowledge graph, and green is for triplets in the target graph.

Without the triplet discriminator, the aligned triplets look quite different from the real ones (under different distributions).

With the triplet discriminator, the aligned triplets look like the real ones (under similar distributions).

6 CONCLUSION This paper studied knowledge graph alignment.

We proposed an unsupervised approach based on the adversarial training framework, which is able to align entities and relations from a source knowledge graph to those in a target knowledge graph.

In practice, our approach can be seamlessly integrated with existing supervised methods, which enables us to leverage a few labeled data as guidance, leading to a weakly-supervised approach.

Experimental results on several real-world datasets proved the effectiveness of our approach in both the unsupervised and weakly-supervised settings.

In the future, we plan to learn alignment functions from two directions (source to target and target to source) to further improve the results, which is similar to CycleGAN BID24 ).

<|TLDR|>

@highlight

This paper studies weakly-supervised knowledge graph alignment with adversarial training frameworks.