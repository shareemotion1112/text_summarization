Over the past decade, knowledge graphs became popular for capturing structured domain knowledge.

Relational learning models enable the prediction of missing links inside knowledge graphs.

More specifically, latent distance approaches model the relationships among entities via a distance between latent representations.

Translating embedding models (e.g., TransE) are among the most popular latent distance approaches which use one distance function to learn multiple relation patterns.

However, they are mostly inefficient in capturing symmetric relations since the representation vector norm for all the symmetric relations becomes equal to zero.

They also lose information when learning relations with reflexive patterns since they become symmetric and transitive.

We propose the Multiple Distance Embedding model (MDE) that addresses these limitations and a framework which enables collaborative combinations of latent distance-based terms (MDE).

Our solution is based on two principles: 1) using limit-based loss instead of margin ranking loss and 2) by learning independent embedding vectors for each of terms we can collectively train and predict using contradicting distance terms.

We further demonstrate that MDE allows modeling relations with (anti)symmetry, inversion, and composition patterns.

We propose MDE as a neural network model which allows us to map non-linear relations between the embedding vectors and the expected output of the score function.

Our empirical results show that MDE outperforms the state-of-the-art embedding models on several benchmark datasets.

While machine learning methods conventionally model functions given sample inputs and outputs, a subset of Statistical Relational Learning (SRL) (De Raedt, 2008; Nickel et al., 2015) approaches specifically aim to model "things" (entities) and relations between them.

These methods usually model human knowledge which is structured in the form of multi-relational Knowledge Graphs (KG).

KGs allow semantically rich queries and are used in search engines, natural language processing (NLP) and dialog systems.

However, they usually miss many of the true relations (West et al., 2014) , therefore, the prediction of missing links/relations in KGs is a crucial challenge for SRL approaches.

A KG usually consists of a set of facts.

A fact is a triple (head, relation, tail) where heads and tails are called entities.

Among the SRL models, distance-based KG embeddings are popular because of their simplicity, their low number of parameters, and their efficiency on large scale datasets.

Specifically, their simplicity allows integrating them into many models.

Previous studies have integrated them with logical rule embeddings (Guo et al., 2016) , have adopted them to encode temporal information (Jiang et al., 2016) and have applied them to find equivalent entities between multi-language datasets (Muhao et al., 2017) .

Soon after the introduction of the first multi-relational distance-based method TransE (Bordes et al., 2013) it was acknowledged that it is inefficient in learning of symmetric relations, since the norm of the representation vector for all the symmetric relations in the KG becomes close to zero.

This means the model cannot distinguish well between different symmetric relations in a KG.

To extend this model many variations are studied afterwards, e.g., TransH (Wang et al., 2014b) , TransR (Lin et al., 2015b) , TransD (Ji et al., 2015) , and STransE (Dat et al., 2016) .

Even though they solved the issue of symmetric relations, they introduced a new problem: these models were no longer efficient in learning the inversion and composition relation patterns that originally TransE could handle.

Besides, as noted in (Kazemi & Poole, 2018; Sun et al., 2019) , within the family of distancebased embeddings, usually reflexive relations are forced to become symmetric and transitive.

In this study, we take advantage of independent vector representations of vectors that enable us to view the same relations from different aspects and put forward a translation-based model that addresses these limitations and allows the learning of all three relation patterns.

In addition, we address the issue of the limit-based loss function in finding an optimal limit and suggest an updating limit loss function to be used complementary to the current limit-based loss function which has fixed limits.

Moreover, we frame our model into a neural network structure that allows it to learn non-linear patterns between embedding vectors and the expected output which substantially improves the generalization power of the model in link prediction tasks.

The model performs well in the empirical evaluations, improving upon the state-of-the-art results in link prediction benchmarks.

Since our approach involves several elements that model the relations between entities as the geometric distance of vectors from different views, we dubbed it multipledistance embeddings (MDE).

Given the set of all entities E and the set of all relations R, we formally define a fact as a triple of the form (h, r, t) in which h is the head and t is the tail, h, t ∈ E and r ∈ R is a relation.

A knowledge graph KG is a subset of all true facts KG ⊂ ζ and is represented by a set of triples.

An embedding is a mapping from an entity or a relation to their latent representation.

A latent representation is usually a (set of) vector(s), a matrix or a tensor of numbers.

A relational learning model is made of an embedding function and a prediction function that given a triple (h, r, t) it determines if (h, r, t) ∈ ζ.

We represent the embedding representation of an entity h with a lowercase letter h if it is a vector and with an uppercase letter H if it is a matrix.

The ability to encode different patterns in the relations can show the generalization power of a model: Definition 1.

A relation r is symmetric (antisymmetric) if ∀x, y r(x, y) ⇒ r(y, x) ( r(x, y) ⇒ ¬r(y, x) ).

A clause with such a structure has a symmetry (antisymmetry) pattern.

Definition 2.

A relation r 1 is inverse to relation r 2 if ∀x, y r 2 (x, y) ⇒ r 1 (y, x).

A clause with such a form has an inversion pattern.

Definition 3.

A relation r 1 is composed of relation r 2 and relation r 3 if ∀x, y, z

A clause with such a form has a composition pattern.

Tensor Factorization and Multiplicative Models define the score of triples via pairwise multiplication of embeddings.

DistMult (Yang et al., 2015) simply multiplies the embedding vectors of a triple element by element h, r, t as the score function.

Since multiplication of real numbers is symmetric, DistMult can not distinguish displacement of head relation and tail entities and therefore, it can not model anti-symmetric relations.

ComplEx (Trouillon et al., 2016) solves the issue of DistMult by the idea that the complex conjugate of the tail makes it non-symmetric.

By introducing complex-valued embeddings instead of realvalued embeddings to DistMult, the score of a triple in ComplEx is Re(h diag(r)t) witht the conjugate of t and Re(.) is the real part of a complex value.

ComplEx is not efficient in encoding composition rules (Sun et al., 2019) .

In RESCAL (Nickel et al., 2011) instead of a vector, a matrix represents the relation r, and performs outer products of h and t vectors to this matrix so that its score function becomes h Rt.

A simplified version of RESCAL is HolE (Nickel et al., 2016) that defines a vector for r and performs circular correlation of h and t has been found equivalent (Hayashi & Shimbo, 2017) to ComplEx.

Another tensor factorization model is Canonical Polyadic (CP) (Hitchcock, 1927) .

In CP decomposition, each entity e is represented by two vectors h e , t e ∈ R d , and each relation r has a single embedding vector v r ∈ R d .

MDE is similarly based on the idea of independent vector embeddings.

A study (Trouillon et al., 2017) suggests that in CP, the independence of vectors causes the poor performance of CP in KG completion, however, we show that the independent vectors can strengthen a model if they are combined complementarily.

SimplE (Kazemi & Poole, 2018) analogous to CP, trains on two sets of subject and object entity vectors.

SimplE's score function, 1 2 h ei , r, t ej + 1 2 h ej , r −1 , t ej , is the average of two terms.

The first term is similar to DistMult.

However, its combination with the second term and using a second set of entity vectors allows SimplE to avoid the symmetric issue of DistMult.

SimplE allows learning of symmetry, anti-symmetry and inversion patterns.

However, it is unable to efficiently encode composition rules, since it does not model a bijection mapping from h to t through relation r.

In Latent Distance Approaches the score function is the distance between embedding vectors of entities and relations.

In the view of social network analysis, (Hoff et al., 2002) originally proposed distance of entities −d(h, t) as the score function for modeling uni-relational graphs where d(., .) means any arbitrary distance, such as Euclidean distance.

SE (Bordes et al., 2011) generalizes the distance for multi-relational data by incorporating a pair of relation matrices into it.

TransE (Bordes et al., 2013) represents relation and entities of a triple by a vector that has this relation

where .

p is the p-norm.

To better distinguish entities with complex relations, TransH (Wang et al., 2014a) projects the vector of head and tail to a relation-specific hyperplane.

Similarly, TransR follows the idea with relation-specific spaces and extends the distance function to M r h + r − M r t p .

RotatE (Sun et al., 2019) combines translation and rotation and defines the distance of a t from tail h which is rotated the amount r as the score function of a triple −d(h • r, t) where • is Hadamard product.

Neural Network Methods train a neural network to learn the interaction of the h, r and t. ER-MLP (Dong et al., 2014 ) is a two layer feedforward neural network considering h, r and t vectors in the input.

NTN (Socher et al., 2013 ) is neural tensor network that concatenates head h and tail t vectors and feeds them to the first layer that has r as weight.

In another layer, it combines h and t with a tensor R that represents r and finally, for each relation, it defines an output layer r to represent relation embeddings.

In SME (Bordes et al., 2014) relation r is once combined with the head h to get g u (h, r), and similarly it is combined with the tail t to get g v (t, r).

SME defines a score function by the dot product of this two functions in the hidden layer.

In the linear SME, g(e, r) is equal to M

The score function of MDE involves multiple terms.

We first explain the intuition behind each term and then explicate a framework that we suggest to efficiently utilize them such that we benefit from their strengths and avoid their weaknesses.

Inverse Relation Learning: Inverse relations can be a strong indicator in knowledge graphs.

For example, if IsP arentOf (m, c) represents that a person m is a parent of another person c, then this could imply IsChildOf (c, m) assuming that this represents the person c being the child of m. This indication is also valid in cases when this only holds in one direction, e.g. for the relations IsM otherOf and IsChildOf .

In such a case, even though the actual inverse IsP arentOf may not even exist in the KG, we can still benefit from inverse relation learning.

To learn the inverse of the relations, we define a score function S 2 :

Symmetric Relations Learning:

It is possible to easily check that the formulation h + r − t allows 1 learning of anti-symmetric pattern but when learning symmetric relations, r tends toward zero which limits the ability of the model in separating entities specially if symmetric relations are frequent in the KG.

For learning symmetric relations, we suggest the term S 3 as a score function.

It learns such relations more efficiently despite it is limited in the learning of antisymmetric relations.

Lemma 1.

S 1 allows modeling antisymmetry, inversion and composition patterns and S 2 allows modeling symmetry patterns. (See proof in Appendix A)

Relieving Limitations on Learning of Reflexive Relations: A previous study (Kazemi & Poole, 2018) highlighted the common limitations of TransE, FTransE, STransE, TransH and TransR for learning reflexive relations where these translation-based models force the reflexive relations to become symmetric and transitive.

To relieve these limitations, we define S 4 as a score function which is similar to the score of RotatE i.e., h • r − t p but with the Hadamard operation on the tail.

In contrast to RotatE which represents entities as complex vectors, S 4 only holds in the real space:

Lemma 2.

The following restrictions of translation based embeddings approaches do not apply to the S 4 score function.

R1: if a relation r is reflexive, on ∆ ∈ E, r it will be also symmetric on ∆. R2: if r is reflexive on ∆ ∈ E, r it will be also be transitive on ∆. (See proof in Appendix B)

Model Definition: To incorporate different views to the relations between entities, we define these settings for the model:

1. Using limit-based loss instead of margin ranking loss.

2.

Each aggregated term in the score represents a different view of entities and relations with an independent set of embedding vectors.

3.

In contrast to ensemble approaches that incorporate models by training independently and testing them together, MDE is based on multi-objective optimization (Marler & Arora, 2004 ) that jointly minimizes the objective functions.

However, when aggregating different terms in the score function, the summation of opposite vectors can cause the norm of these vectors to diminish during the optimization.

For example if S 1 and S 3 are added together, the minimization would lead to relation(r) vectors with zero norm value.

To address this issue, we represent the same entities with independent variables in different distance functions.

Based on CP, MDE considers four vectors e i , e j , e k , e l , ∈ R d as the embedding vector of each entity e , and four vectors r i , r j , r k , r l ∈ R d for each relation r.

The score function of MDE for a triple (h, r, t) is defined as weighted sum of listed score functions:

where ψ, w 1 , w 2 , w 3 , w 4 ∈ R are constant values.

In the following, we show using ψ and limitbased loss, the combination of the terms in equation 5 is efficient, such that if one of the terms recognises if a sample is true F M DE would also recognize it.

Limit-based Loss: Because margin ranking loss minimizes the sum of error from directly comparing the score of negative to positive samples, when applying it to translation embeddings, it is possible that the score of a correct triplet is not small enough to hold the relation of the score function (Zhou et al., 2017) .

To enforce the scores of positive triples become lower than those of negative ones, (Zhou et al., 2017 ) defines limited-based loss which minimizes the objective function such that Figure 1 : Geometric illustration of the translation terms considered in MDE the score for all the positive samples become less than a fixed limit.

Sun et al. (2018) extends the limit-based loss so that the score of the negative samples become greater than a fixed limit.

We train our model with the same loss function which is:

where

− are the set of positive and negative samples and β 1 , β 2 > 0 are constants denoting the importance of the positive and negative samples.

This version of limit-based loss minimizes the aggregated error such that the score for the positive samples become less than γ 1 and the score for negative samples become greater than γ 2 .

To find the optimal limits for the limit-based loss, we suggest updating the limits during the training. (See the explanation in Appendix D).

Lemma 3.

There exist ψ and γ 1 , γ 2 ≥ 0 (γ 1 ≥ γ 2 ), such that only if one of the terms in f M DE estimates a fact as true, f M DE also predicts it as a true fact.

Consequently, the same also holds for the capability of MDE to allow learning of different relation patterns. (See proof in Appendix C)

It is notable that without the introduction of ψ and the limits γ 1 , γ 2 from the limit-based loss, Lemma 3 does not hold and framing the model with this settings makes the efficient combination of the terms in f M DE possible.

In contrast to SimplE that ties the relation vectors of two terms in the score together, MDE does not directly relate them to take advantage of the independent relation and entity vectors in combining opposite terms.

The learning of the symmetric relations is previously studied (e.g. in (Yang et al., 2014; Sun et al., 2019) ) and (Lin et al., 2015a) studied the training over the inverse of relations, however providing a way to gather all these benefits in one model is a novelty of MDE.

Besides, complementary modeling of different vector-based views of a knowledge graph is a novel contribution.

The score of MDE is already aggregating a multiplication of vectors to weights.

We take advantage of this setting to model MDE as a layer of a neural network that allows learning the embedding vectors and multiplied weights jointly during the optimization.

To create such a neural network we multiply ψ by a weight w 5 and we feed the MDE score to an activation function.

We call this extension of MDE as MDE N N :

where σ is logistic sigmoid function and w 1 , w 2 , . . . , w 5 are elements of the latent vector w that are estimated during the training of the model.

This framing of MDE reduces the number of hyperparameters.

The major advantage of MDE N N in comparison to the current distance-based models is that the logistic sigmoid activation function allows the non-linear mappings between the embedding vectors and the expected output for positive and the negative samples.

Considering the ever growth of KGs and the expansion of the web, it is crucial that the time and memory complexity of a relational mode be minimal.

Despite the limitations in expressivity, TransE is one of the popular models on large datasets due to its scalability.

With O(d) time complexity (of one mini-batch), where d is the size of embedding vectors, it is more efficient than RESCAL, NTN, and the neural network models.

Similar to TransE, the time complexity of MDE is O(d).

Due to the additive construction of MDE, the inclusion of more distance terms keeps the time complexity linear in the size of vector embeddings.

Datasets: We experimented on four standard datasets: WN18 and FB15k are extracted by (Bordes et al., 2013) from Wordnet (Miller, 1995) Freebase (Bollacker et al., 2008) .

We used the same train/valid/test sets as in (Bordes et al., 2013 (Trouillon et al., 2016) and the results of TransR and NTN from (Nguyen, 2017), and ER-MLP from (Nickel et al., 2016) .

The results on the inverse relation excluded datasets are from (Sun et al., 2019) , Table 13 for TransE and RotatE and the rest are from (Dettmers et al., 2018) 2 .

Evaluation Settings: We evaluate the link prediction performance by ranking the score of each test triple against its versions with replaced head, and once for tail.

Then we compute the hit at N (Hit@N), mean rank (MR) and mean reciprocal rank (MRR) of these rankings.

We report the evaluations in the filtered setting.

Implementation: We implemented MDE in PyTorch 3 .

Following (Bordes et al., 2011) , we generated one negative example per positive example for all the datasets.

We used Adadelta (Zeiler, 2012) as the optimizer and fine-tuned the hyperparameters on the validation dataset.

The ranges of the hyperparameters are set as follows: embedding dimension 25, 50, 100, 200, batch size 100, 150, and iterations 50, 100, 1000, 1500, 2500, 3600.

We set the initial learning rate on all datasets to 10.

For MDE, the best embedding size and γ 1 and γ 2 and β 1 and β 2 values on WN18 were 50 and 1.9, 1.9, 2 and 1 respectively and for FB15k were 200, 10, 13, 1, 1.

The best found embedding size and γ 1 and γ 2 and β 1 and β 2 values on FB15k-237 were 100, 9, 9, 1 and 1 respectively and for WN18RR were 50, 2, 2, 5 and 1.

We selected the coefficient of terms in equation 5, by grid search in the range 0.1 to 1.0 and testing those combinations of the coefficients where they create a convex combination.

Found values are w 1 = 0.16, w 2 = 0.33, w 3 = 0.16, w 4 =0.33.

We also tested for the best value for ψ between {0.1, 0.2,. . .

, 1.5}. We use ψ = 1.2 for all the experiments.

For MDE N N , we use the same γ 1 , γ 2 , β 1 and β 2 values except for WN18 that the γ 1 and γ 2 are 4.

We use the embedding size 50 for WN18RR, 200 for WN18, 200 for FB15k-237 and 200 for FB15k.

We use ψ = 2 for all the MDE N N experiments.

To regulate the loss function and to avoid over-fitting, we estimate the score function for two sets of independent vectors and we take their average in the prediction.

Another advantage of this operation is the reduction of required training iterations.

As a result, MDE reaches to the 99 percent of its ranking performance in 100 iterations, and MDE N N reaches its best performance in the benchmarks in just 50 iterations.

Table 2 shows the result of our experiment on FB15k-237 and WN18RR, where the improvement is much more significant.

Due to the existence of hard limits in the limit-based loss, the mean rank in both MDE and MDE N N is much lower than other methods.

The comparison of MDE to other state-of-the-art models, regardless of the MDE N N , shows the competitive performance of MDE.

It is observable that while MDE generates only one negative sample per positive sample and is using vector sizes between 50 to 200, it challenges RotatE which employs relatively large embedding dimensions (from 125 up to 1000) and high number of negative samples (up to 1024).

We observe that the application of sigmoid in MDE N N improves it significantly in all the benchmarks.

Particularly, in the more challenging tests over WN18RR and FB15k-237, the improvement is more significant.

For example, we can see that the construction of the neural network from the model increased its Hit@10 result on FB15k-237 from 0.484 to 0.999.

From analyzing the MRR scores, we can see that RotatE must be totally off in few cases whereas the MDE N N model almost never seems to be far off, but frequently fails to put the correct entity on top.

To our knowledge, MDE N N outperforms all the current embedding models in the MR and Hit@10 measures and specially performs better than all the existing models in all the measures on WN18RR and FB15k-237 benchmarks.

To better understand the role of each term in the score function of MDE, we embark two ablation experiments.

First, we train MDE using one of the terms alone, and observe the link prediction performance of each term in the filtered setting.

In the second experiment, we remove one of the terms at a time and test the effect of the removal of that term on the model after 100 iterations.

Table 3 summarizes the results of the first experiment on WN18RR and FB15k-237.

We can see that S 4 outperforms the other terms while S 1 and S 3 performs very similar on these two datasets.

Between the four terms, S 2 performs the worst since most of the relations in the test datasets follow an antisymmetric pattern and S 2 is not efficient in modeling them.

Table 4 shows the results of the second experiment.

The evaluations on WN18RR and WN18 show that removal of S 4 has the most negative effect on the performance of MDE.

The removal of S 1 that was one of the good performing terms in the last experiment has the least effect.

Nevertheless, S 1 improves the MRR in the MDE.

Also, when we remove S 2 , the MRR and Hit@10 are negatively influenced, indicating that there exist cases that S 2 performs better than the other terms, although, in the individual tests, it performed the worst between all the terms.

In this study, we showed how MDE relieves the expressiveness restrictions of the distance-based embedding models and proposed a general method to override these limitations for the older models.

Beside MDE and RotatE, most of the existing KG embedding approaches are unable to allow modeling of all the three relation patterns.

We framed MDE into a Neural Network structure and validated our contributions via both theoretical proofs and empirical results.

We demonstrated that with multiple views to translation embeddings and using independent vectors (that previously were suggested to cause poor performance (Trouillon et al., 2017; Kazemi & Poole, 2018) ) a model can outperform the existing state-of-the-art models for link prediction.

Our experimental results confirm the competitive performance of MDE and particularly MDE N N that achieves state-of-the-art MR and Hit@10 performance on all the benchmark datasets.

A PROOF OF LEMMA 1.

Let r 1 , r 2 , r 3 be relation vector representations and e i , e j , e k are entity representations.

A relation r 1 between (e i , e k ) exists when a triple (e i , r 1 , e k ) exists and we show it by r 1 (e i , e k ).

Formally, we have the following results:

Antisymmetric Pattern.

If r 1 (e i , e j ) and r 1 (e j , e i ) hold, in equation 1 for S 1 , then: e i + r 1 = e j ∧ e j + r 1 = e i ⇒ e i + 2r 1 = e i Therefore S 1 allows encoding of relations with antisymmetric patterns.

Symmetric Pattern.

If r 1 (e i , e j ) and r 1 (e j , e i ) hold, for S 2 we have: e i + e j − r 1 = 0 ∧ e j + e i − r 1 = 0 ⇒ e j + e i = r 1 Therefore S 2 allows encoding relations with symmetric patterns.

For S 1 we have:

Inversion Pattern.

If r 1 (e i , e j ) and r 2 (e j , e i ) hold, from Equation 1 we have:

Therefore S 1 allows encoding relations with inversion patterns.

Composition Pattern.

If r 1 (e i , e k ) , r 2 (e i , e j ) and, r 3 (e j , e k ) hold, from equation 1 we have: e i + r 1 = e k ∧ e i + r 2 = e j ∧ e j + r 3 = e k ⇒ r 2 + r 3 = r 1

Therefore S 1 allows encoding relations with composition patterns.

B PROOF OF LEMMA 2.

Proof.

R1: For such reflexive r 1 , if r 1 (e i , e i ) then r l (e j , e j ).

In this equation we have: e i = r 1 e i ∧ e j = r 1 e j ⇒ r 1 = U ⇒ e i = r 1 e j where U is unit tensor.

R2: For such reflexive r 1 , if r 1 (e i , e j ) and r l (e j , e k ) then r 1 (e j , e i ) and r l (e k , e j ).

In the above equation we have: e i = r 1 e j ∧ e j = r 1 e k ⇒ e i = r 1 r 1 e j e k ∧ r i = U ⇒ e i = e j e k ⇒ e i + e k = r l C PROOF OF LEMMA 3.

We show there is boundries for γ 1 , γ 2 , w 1 , w 2 , w 3 , w 4 , such that learning a fact by one of the terms in f M DE is enough to classify a fact correctly.

Proof.

We show the boundaries for three aggregated terms in the the distance function, it is easily possible to extend it to four and more terms.

It is enough to show that there is at least one set of boundaries for the positive and negative samples that follows the constraints.

The case to prove is when three of the distance functions classify a fact negative N and the one distance function e.g. s 2 classify it as positive P , and the case that s 1 and s 3 classify a fact as positive and s 2 classify it as negative.

We set w 1 = w 3 = 1/4 and w 2 = 1/2 and assume that Sum is the value estimated by the score function of MDE, we have:

There exist a = 2 and γ 1 = γ 2 = 2 and ψ = 1 that satisfy γ 1 > Sum ≥ 0 and the inequality 8.

loss = the result from equation 6 It can be easily checked that without introduction of ψ, there is no value of Sum that can satisfy both γ 1 > Sum ≥ 0 and the inequality 8 and we calculated the value of ψ based on the values of γ 1 , γ 2 and a. In case that future studies discover new interesting distances, this Lemma shows how to basically integrate them into MDE.

While the limit-based loss resolves the issue of margin ranking loss with distance based embeddings, it does not provide a way to find the optimal limits.

Therefore the mechanism to find limits for each dataset and hyper-parameter is the try and error.

To address this issue, we suggest updating the limits in the limit-based loss function during the training iterations.

We denote the moving-limit loss by loss guide .

loss guide = lim δ,δ →γ1

where the initial value of δ 0 , δ 0 is 0.

In this formulation, we increase the δ 0 , δ 0 toward γ 1 and γ 2 during the training iterations such that the error for positive samples minimizes as much as possible.

We test on the validation set after each 50 epoch and take those limits that give the best value during the tests.

The details of the search for limits is explained in Algorithm 1.

After observing the most promising values for limits in the preset number of iterations, we stop the search and perform the training while having the δ values fixed(fixed limit-base loss) to allow the adaptive learning to reach loss values smaller than the threshold.

We based this approach on the idea of adaptive learning rate (Zeiler, 2012) , where the Adadelta optimizer adapts the learning rate after each iteration, therefore in the loss guided we can update the limits without stopping the training iterations.

In our experiments, the variables in the Algorithm 1, are as follows.

threshold = 0.05, ξ = 0.1.

<|TLDR|>

@highlight

A novel method of modelling Knowledge Graphs based on Distance Embeddings and Neural Networks