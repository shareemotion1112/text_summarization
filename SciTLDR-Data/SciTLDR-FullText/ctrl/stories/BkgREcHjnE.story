Knowledge graphs are structured representations of real world facts.

However, they typically contain only a small subset of all possible facts.

Link prediction is the task of inferring missing facts based on existing ones.

We propose TuckER, a relatively simple yet powerful linear model based on Tucker decomposition of the binary tensor representation of knowledge graph triples.

By using this particular decomposition, parameters are shared between relations, enabling multi-task learning.

TuckER outperforms previous state-of-the-art models across several standard link prediction datasets.

Vast amounts of information available in the world can be represented succinctly as entities and relations between them.

Knowledge graphs are large, graph-structured databases which store facts in triple form (e s , r, e o ), with e s and e o representing subject and object entities and r a relation.

However, far from all available information is stored in existing knowledge graphs, which creates the need for algorithms that automatically infer missing facts.

Knowledge graphs can be represented by a third-order binary tensor, where each element corresponds to a triple, 1 indicating a true fact and 0 indicating the unknown (either a false or a missing fact).

The task of link prediction is to infer which of the 0 entries in the tensor are indeed false, and which are missing but actually true.

A large number of approaches to link prediction so far have been linear, based on various methods of factorizing the third-order binary tensor BID12 BID22 BID19 BID7 .

Recently, state-of-the-art results have been achieved using non-linear convolutional models BID3 BID0 .

Despite achieving very good per- formance, the fundamental problem with deep, non-linear models is that they are non-transparent and poorly understood, as opposed to more mathematically principled and widely studied tensor decomposition models.

In this paper, we introduce TuckER (E stands for entities, R for relations), a simple linear model for link prediction in knowledge graphs, based on Tucker decomposition BID21 of the binary tensor of triples.

Tucker decomposition factorizes a tensor into a core tensor multiplied by a matrix along each mode.

In our case, rows of the matrices contain entity and relation embeddings, while entries of the core tensor determine the level of interaction between them.

Due to having the core tensor, unlike simpler models, such as RESCAL, DistMult and ComplEx, where parameters for each relation are often learned separately, TuckER makes use of multi-task learning between different relations BID24 .

Subject and object entity embedding matrices are assumed equivalent, i.e. we make no distinction between the embeddings of an entity depending on whether it appears as a subject or as an object in a particular triple.

Our experiments show that TuckER achieves state-of-the-art results across all standard link prediction datasets.

Several linear models for link prediction have previously been proposed.

An early linear model, RESCAL BID12 , optimizes a scoring function containing a bilinear product between subject and object entity vectors and a full rank matrix for each relation.

RESCAL is prone to overfitting due to its large number of parameters, which increases quadratically in the embedding dimension with the number of relations in a knowledge graph.

DistMult BID22 ) is a special case of RESCAL with a diagonal matrix per relation, which reduces overfitting.

However, DistMult cannot model asymmetric relations.

ComplEx BID19 extends DistMult to the complex domain.

Subject and object entity embeddings for the same entity are complex conjugates, which enables ComplEx to model asymmetric relations.

SimplE BID7 is a model based on Canonical Polyadic (CP) decomposition BID5 .Scoring functions of all models described above and TuckER are summarized in Table 1 .

Table 1 .

Scoring functions of state-of-the-art link prediction models, the dimensionality of their relation parameters, and significant terms of their space complexity.

de and dr are the dimensionalities of entity and relation embeddings, while ne and nr denote the number of entities and relations respectively.

eo ∈ C de is the complex conjugate of eo, he s , te s ∈ R de are the head and tail entity embedding of entity es, and w r −1 ∈ R dr is the embedding of relation r −1 which is the inverse of relation r. · denotes the dot product and ×n denotes the tensor product along the n-th mode, f is a non-linear function, and W ∈ R de×de×dr is the core tensor of a Tucker decomposition.

Relation Parameters Space Complexity BID22 e s , w r , e o w r ∈ R de O(n e d e + n r d e ) ComplEx BID19 Re( e s , w r , e o ) DISPLAYFORM0 DISPLAYFORM1

Let E denote the set of all entities and R the set of all relations present in a knowledge graph.

A triple is represented as (e s , r, e o ), with e s , e o ∈ E denoting subject and object entities respectively and r ∈ R the relation between them.

In link prediction, we are given a subset of all true triples and the aim is to learn a scoring function φ that assigns a score s = φ(e s , r, e o ) ∈ R which indicates whether a triple is true, with the ultimate goal of being able to correctly score all missing triples.

The scoring function is either a specific form of tensor factorization in the case of linear models or a more complex (deep) neural network architecture for nonlinear models.

Typically, a positive score for a particular triple indicates a true fact predicted by the model, while a negative score indicates a false one.

Tucker decomposition, named after Ledyard R. Tucker BID20 , decomposes a tensor into a set of matrices and a smaller core tensor.

In a three-mode case, given the original tensor X ∈ R I×J×K , Tucker decomposition outputs a tensor Z ∈ R P ×Q×R and three matrices A ∈ R I×P , B ∈ R J×Q , C ∈ R K×R : DISPLAYFORM0 with × n indicating the tensor product along the n-th mode.

Elements of the core tensor Z show the level of interaction between the different components.

Typically, P , Q, R are smaller than I, J, K respectively, so Z can be thought of as a compressed version of X BID9 ).

We propose a model that uses Tucker decomposition for link prediction on the binary tensor representation of a knowledge graph, with entity embedding matrix E that is equivalent for subject and object entities, i.e. E = A = C ∈ R ne×de and relation embedding matrix R = B ∈ R nr×dr , where n e and n r represent the number of entities and relations and d e and d r the dimensionality of entity and relation embedding vectors respectively.

We define the scoring function for TuckER as: DISPLAYFORM0 where e s , e o ∈ R de are the rows of E representing the subject and object entity embedding vectors, w r ∈ R dr the rows of R representing the relation embedding vector and W ∈ R de×dr×de is the core tensor.

We apply logistic sigmoid to each score φ(e s , r, e o ) to obtain the predicted probability p of a triple being true.

Visualization of the TuckER model architecture can be seen in FIG1 .

The number of parameters of TuckER increases linearly with respect to entity and relation embedding dimensionality d e and d r , as the number of entities and relations increases, since the number of parameters of W depends only on the entity and relation embedding dimensionality and not on the number of entities or relations.

By having the core tensor W, unlike simpler models such as DistMult, ComplEx and SimplE, TuckER does not encode all the learned knowledge into the embeddings; some is stored in the core tensor and shared between all entities and relations through multi-task learning BID24 .

Following the training procedure introduced by BID3 , we use 1-N scoring, i.e. we simultaneously score a pair (e s , r) with all entities e o ∈ E, in contrast to 1-1 scoring, where individual triples (e s , r, e o ) are trained one at a time.

We assume that a knowledge graph is only locally complete by including only the non-existing triples (e s , r, ·) and (·, r, e o ) of the observed pairs (e s , r) and (r, e o ) as negative samples and all observed triples as positive samples.

We train our model to minimize the Bernoulli negative loglikelihood loss function.

A component of the loss for one subject entity and all the object entities is defined as: DISPLAYFORM0 where p ∈ R ne is the vector of predicted probabilities and y ∈ R ne is the binary label vector.

We evaluate TuckER using standard link prediction datasets.

FB15k BID1 ) is a subset of Freebase, a large database of real world facts.

FB15k-237 BID18 was created from FB15k by removing the inverse of many relations that are present in the training set from validation and test sets.

WN18 BID1 ) is a subset of WordNet, containing lexical relations between words.

WN18RR BID3 ) is a subset of WN18, created by removing the inverse relations.

We implement TuckER in PyTorch BID13 and make our code available on Github 1 .

We choose all hyper-parameters by random search based on validation set performance.

For FB15k and FB15k-237, we set entity and relation embedding dimensionality to d e = d r = 200.

For WN18 and WN18RR, which both contain a significantly smaller number of relations relative to the number of entities as well as a small number of relations compared to FB15k and FB15k-237, we set d e = 200 and d r = 30.

We use batch normalization BID6 and dropout BID16 to speed up training.

We choose the learning rate from {0.01, 0.005, 0.003, 0.001, 0.0005} and learning rate decay from {1, 0.995, 0.99}. We find the following combinations of learning rate and learning rate decay to give the best results: (0.003, 0.99) for FB15k, (0.0005, 1.0) for FB15k-237, (0.005, 0.995) for WN18 and (0.01, 1.0) for WN18RR.

We train the model using Adam BID8 with the batch size 128.1 https://github.com/ibalazevic/TuckERWe evaluate each triple from the test set as in BID1 : for a given triple, we generate 2n e test triples by keeping the subject entity e s and relation r fixed and replacing the object entity e o with all possible entities E and vice versa.

We then rank the scores obtained.

We use the filtered setting, i.e. we remove all true triples apart from the currently observed test triple.

For evaluation, we use the evaluation metrics used across the link prediction literature: mean reciprocal rank (MRR) and hits@k, k ∈ {1, 3, 10}. Mean reciprocal rank is the average of the inverse of a mean rank assigned to the true triple over all n e generated triples.

Hits@k measures the percentage of times the true triple is ranked in the top k of the n e generated triples.

Link prediction results on all datasets are shown in Tables 2 and 3 .

Overall, TuckER outperforms previous state-ofthe-art models on all metrics across all datasets (apart from hits@10 on WN18).

Results achieved by TuckER are not only better than those of other linear models, such as DistMult, ComplEx and SimplE, but also better than those of many more complex deep neural network and reinforcement learning architectures, e.g. MINERVA, ConvE and HypER, demonstrating the expressive power of linear models.

Even though at entity embedding dimensionality d e = 200 and relation embedding dimensionality d r = 30 on WN18RR TuckER has fewer parameters (∼9.4 million) than ComplEx and SimplE (∼16.4 million), it consistently obtains better results than any of those models.

We believe this is achieved by exploiting knowledge sharing between relations through the core tensor.

We find that lower dropout values (0.1, 0.2) are required for datasets with a higher number of training triples per relation and thus less risk of overfitting (WN18 and WN18RR) and higher dropout values (0.3, 0.4, 0.5) are required for FB15k and FB15k-237.

We further note that TuckER improves the results of all other linear models by a larger margin on datasets with a large number of relations (e.g. +14% improvement on FB15k results over ComplEx, +8% improvement over SimplE on the toughest hits@1 metric), which supports our belief that TuckER makes use of the parameters shared between similar relations to improve predictions by multi-task learning.

The presence of the core tensor which allows for knowledge sharing between relations suggests that TuckER should need a lower number of parameters for obtaining good results than ComplEx or SimplE. To test this, we re-implement ComplEx and SimplE with 1-N scoring, batch normalization and dropout for fair comparison, perform random search to choose best hyper-parameters and train all three models on FB15k-237 with embedding sizes d e = d r ∈ Table 2 .

Link prediction results on WN18RR and FB15k-237.

We report results for ComplEx-N3 BID10 at de = 115 for WN18RR and de = 400 for FB15k-237 to ensure comparability with TuckER in terms of the overall number of parameters (original paper reports results at de = 2000).

The RotatE BID17 results are reported without their self-adversarial negative sampling (see Appendix H in the original paper) for fair comparison, given that it improves the results by ∼ 4% and it is not specific to that model only.

FB15k-237Linear MRR Hits@10 Hits@3 Hits@1 MRR Hits@10 Hits@3 Hits@1DistMult BID22 no BID14 no DISPLAYFORM0 DISPLAYFORM1 .151 MINERVA BID2 no BID3 FIG2 shows the obtained MRR on the test set for each model.

It is important to note that at embedding dimensionalities 20, 50 and 100, TuckER has fewer parameters than ComplEx and SimplE (e.g. ComplEx and SimplE have ∼3 million and TuckER has ∼2.5 million parameters for embedding dimensionality 100).

DISPLAYFORM2 We can see that the difference between the MRRs of ComplEx, SimplE and TuckER is approximately constant for embedding sizes 100 and 200.

However, for lower embedding sizes, the difference between MRRs increases e.g. by 4.2% for embedding size 20 for ComplEx and by 9.9% for embedding size 20 for SimplE. At embedding size 20 (∼300k parameters), the performance of TuckER is almost as good as the performance of ComplEx and SimplE at embedding size 200 (∼6 million parameters), which supports our initial assumption.

In this work, we introduce TuckER, a relatively simple yet highly flexible linear model for link prediction in knowledge graphs based on the Tucker decomposition of a binary tensor of training set triples, which achieves state-of-the-art results on several standard link prediction datasets.

TuckER's number of parameters grows linearly with respect to embedding dimension as the number of entities or relations in a knowledge graph increases, which makes it easily scalable to large knowledge graphs.

Future work might include exploring how to incorporate background knowledge on individual relation properties into the existing model.

<|TLDR|>

@highlight

 We propose TuckER, a relatively simple but powerful linear model for link prediction in knowledge graphs, based on Tucker decomposition of the binary tensor representation of knowledge graph triples. 