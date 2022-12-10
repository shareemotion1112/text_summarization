Knowledge Graphs (KG), composed of entities and relations, provide a structured representation of knowledge.

For easy access to statistical approaches on relational data, multiple methods to embed a KG as components of R^d have been introduced.

We propose TransINT, a novel and interpretable KG embedding method that isomorphically preserves the implication ordering among relations in the embedding space.

TransINT maps set of entities (tied by a relation) to continuous sets of vectors that are inclusion-ordered isomorphically to relation implications.

With a novel parameter sharing scheme, TransINT enables automatic training on missing but implied facts without rule grounding.

We achieve new state-of-the-art performances with signficant margins in Link Prediction and Triple Classification on FB122 dataset, with boosted performance even on test instances that cannot be inferred by logical rules.

The angles between the continuous sets embedded by TransINT provide an interpretable way to mine semantic relatedness and implication rules among relations.

Recently, learning distributed vector representations of multi-relational knowledge has become an active area of research (Bordes et al.; Nickel et al.; Kazemi & Poole; Wang et al.; Bordes et al.) .

These methods map components of a KG (entities and relations) to elements of R d and capture statistical patterns, regarding vectors close in distance as representing similar concepts.

However, they lack common sense knowledge which are essential for reasoning (Wang et al.; Guo et al.; Nickel & Kiela) .

For example, "parent" and "father" would be deemed similar by KG embeddings, but by common sense, "parent ⇒ father" yet not the other way around.

Thus, one focus of current research is to bring common sense rules to KG embeddings (Guo et al.; Wang et al.; Wei et al.( .

Some methods impose hard geometric constraints and embed asymmetric orderings of knowledge (Nickel & Kiela; Vendrov et al.; Vilnis et al.( .

However, they only embed hierarchy (unary Is_a relations), and cannot embed n-ary relations in KG's.

Moreover, their hierarchy learning is largely incompatible with conventional relational learning, because they put hard constraints on distance to represent partial ordering, which is a common metric of similarity/ relatedness in relational learning.

We propose TransINT, a new KG embedding method that isomorphically preserves the implication ordering among relations in the embedding space.

TransINT restrict entities tied by a relation to be embedded to vectors in a particular region of R d included isomorphically to the order of relation implication.

For example, we map any entities tied by is_father_of to vectors in a region that is part of the region for is_parent_of; thus, we can automatically know that if John is a father of Tom, he is also his parent even if such a fact is missing in the KG.

Such embeddings are constructed by sharing and rank-ordering the basis of the linear subspaces where the vectors are required to belong.

Mathematically, a relation can be viewed as sets of entities tied by a constraint (Stoll) .

We take such a view on KG's, since it gives consistancy and interpretability to model behavior.

Furthermore, for the first time in KG embedding, we map sets of entitites under relation constraint to a continuous set of points (whose elements are entity vectors) -which learns relationships among not only individual entity vectors but also sets of entities.

We show that angles between embedded relation sets can identify semantic patterns and implication rules -an extension of the line of thought as in word/ image embedding methods such as Mikolov et al., Frome et al. to relational embedding.

Such mining is both limited and less interpretable if embedded sets are discrete (Vilnis et al.; Vendrov et al.) or each entitity itself is embedded to a region, not a member vector of it (Vilnis et al.) .

1 TransINT's such interpretable meta-learning opens up possibilities for explainable reasoning in applications such as recommender systems (Ma et al.) and question answering (Hamilton et al.

In this section, we describe the intuition and justification of our method.

We first define relation as sets, and revisit TransH as mapping relations to sets in R d .

Finally, we propose TransINT, which connects the ordering of the two aforementioned sets.

We put * next to definitions and theorems we propose/ introduce.

Otherwise, we use existing definitions and cite them.

We define relations as sets and implication as inclusion of sets, as in set-theoretic logic.

Definition (Relation Set): Let r i be a binary relation x, y entities.

Then, r i (x, y) iff there exists some set R i such that the pair (x, y) ∈ R i .

R i is called the relation set of r i . (Stoll) For example, consider the distinct relations in Figure 1a , and their corresponding sets in Figure 1b ; Is_Father_Of(Tom, Harry) is equivalent to (Tom, Harry) ∈ R Is_Father_Of .

Definition (Logical Implication): For two relations, r 1 implies r 2 (or r 1 ⇒ r 2 ) iff ∀x, y, ; the orange dot is the origin, to emphasize that a vector is really a point from the origin but can be translated and considered equivalently.

(a): first projecting #» h and #» t onto H is_f amily_of , and then requiring

: first substracting #» t from #» h , and then projecting the distance ( # » t − h) to H is_f amily_of and requiring ( # » t − h) ⊥ ≈ rj.

The red line is unique because it is when # » r is_f amily_of is translated to the origin.

2.2 BACKGROUND: TRANSE AND TRANSH Given a fact triple (h, r, t) in a given KG (i.e. (Harry, is_father_of, Tom)), TransE wants #» h + #» r ≈ #» t where #» h , #» r , #» t are embeddings of h, r, t. In other words, the distance between two entity vectors is equal to a fixed relation vector.

TransE applies well to 1-to-1 relations but has issues for N-to-1, 1-to-N and N-to-N relations, because the distance between two vectors are unique and thus two entities can only be tied with one relation.

To address this, TransH constraints the distance of entities in a multi-relational way, by decomposing distance with projection ( Figure 2a) .

TransH first projects an entity vector into a hyperplane unique to each relation, and then requires their difference is some constant value.

Like TransE, it embeds an entity to a vector.

However, for each relation r j , it assigns two components: a relation-specific hyperplane H j and a fixed vector #» r j on H j .

For each fact triple (h, r j , t), TransH wants ( Figure 2 )

where (Figure 2a ).

Revisiting TransH We interpret TransH in a novel perspective.

An equivalent way to put Eq.1 is to change the order of subtraction and projection:

This means that all entity vectors ( #» h , #» t ) such that their distance # » t − h belongs to the red line are considered to be tied by relation r j (Figure 2b) i.e. R j ≈ the red line.

For example,

The red line is the set of all vectors whose projection onto H j is the fixed vector #» r j .

Thus, upon a deeper look, TransH actually embeds a relation set in KG (figure 1b) to a particular set in R d .

We call such sets relation space for now; in other words, a relation space of some relation r i is the space where each (h, r i , t)'s # » t − h can exist.

We formally visit it later in Section 3.1.

. (Figure 2b ) perspective.

The blue line, red line, and the green plane is respectively is_father_of, is_mother_of and is_parent_of 's relation space -where # » t − h's of h, t tied by these relations can exist.

The blue and the red line lie on the green plane -is_parent_of 's relation space includes the other two's.

2.3 TRANSINT Like TransH, TransINT embeds a relation r j to a (subspace, vector) pair (H j , #» r j ).

However, TransINT modifies the relation embeddings (H j , #» r j ) so that the relation spaces (i.e. red line of Figure 2b ) are ordered by implication; we do so by intersecting the H j 's and projecting the #» r j 's ( Figure 3a ).

We explain with familial relations as a running example.

Intersecting the H j 's TransINT assigns distinct hyperplanes H is_f ather_of and H is_mother_of to is_father_of and is_mother_of.

However, because is_parent_of is implied by the aforementioned relations, we assign H is_parent_of = H is_f ather_of ∩ H is_mother_of .

TrainsINT's H is_parent_of is not a hyperplane but a line (Figure 3a ), unlike in TransH where all H j 's are hyperplanes.

Projecting the #» r j 's TransH constrains the #»

r j 's with projections ( Figure 3a 's dotted orange lines).

First, # » r is_f ather_of and # » r is_mother_of are required to have the same projection onto H is_parent_of .

Second, # » r is_parent_of is that same projection onto H is_parent_of .

We connect the two above constraints to ordering relation spaces.

Figure 3b graphically illustrates that is_parent_of 's relation space (green hyperplane) includes those of is_father_of (blue line) and is_mother_of (red line).

More generally, TransINT requires two hard geometric constraints on (H j , #» r j )'s that For distinct relations r i , r j , require the following if and only if r i ⇒ r j : Intersection Constraint:

We prove that these two constraints guarantee that an ordering isomorphic to implication holds in the embedding space: (r i ⇒ r j ) iff (r i 's rel.

space ⊂ r j 's rel.

space) or equivalently,

The orderings are isomorphic, because for example, if is_parent_of subsumes is_father_of, the first relation space also subsumes the latter's ( Figure 3 ).

At first sight, it may look paradoxical that the H j 's and the relation spaces are inversely ordered; however, it is a natural consequence of the rank-based geometry in R d .

In this section, we formally state TransINT's isomorphic guarantee and its grounds.

We also discuss the intuitive meaning of our method.

We denote all d × d matrices with capital letters (ex) A) and vectors with arrows on top (ex) #» b ).

In R d , points are projected to linear subspaces by projection matrices; each linear subspace H i has a projection matrix Strang) .

For example, in Figure 4 , a random point #» a ∈ R d is projected onto H 1 when multiplied by P 1 ; i.e. P 1 a = #» b ∈ H 1 .

In the rest of the paper, denote P i as the projection matrix onto subspace H i .

Now, we algebraically define a general concept that subsumes relation space( Figure 3b ).

Let H be a linear subspace and P its projection matrix.

Then, given #» k on H, the set of vectors that become #» k when projected on to H, or the solution space of

With this definition, relation space ( Figure 3b ) is (Sol(P i , #» r i )), where P i is the projection matrix of H i (subspace for relation r i ); it is the set of points

Main Theorem 1 (Isomorphism):

Let {(H i , #» r i )} n be the (subspace, vector) embeddings assigned to relations {R i } n by the Intersection Constraint and the Projection Constraint; P i the projection matrix of

In actual implementation and training, TransINT requires something less strict than

for some non-negative and small .

This bounds # » t − h − #» r i to regions with thickness 2 , centered around Sol(P i , #» r i ) ( Figure 5 ).

We prove that isomorphism still holds with this weaker requirement.

Main Theorem 2 (Margin-aware Isomorphism):

For all non-negative scalar , ({Sol (P i , #» r i )} n , ⊂) is isomorphic to ({R i } n , ⊂).

At a first glance, it may look paradoxical that a relation whose H i is the intersection of other relations' H j 's (i.e. is_parent_of of Figure 3a) actually subsumes all the relations that were intersected (i.e. is_fahter_of, is_mohter_of).

This inverse ordering of H j 's and Sol(P j , #» r j ) arise from the fact that the two are orthocomplements (Strang) .

Geometrically, projection is decomposition into independent directions; #» x = P #» x + (I − P ) #» x holds for all #» x .

In Fig. 4a , one can see that P and I − P are orthogonal.

Algebraically, a vector #» x ∈ R d bound by P #» x = b, composed of k independent constraints (rank k), #» x is free in all other d − k directions of I − P (Fig. 4b) .

Thus, the lesser constraint the space to be projected onto, the more freedom a vector is given; which is isomorphic to that, for example, is_f amily_of puts more freedom on who can be tied by it than is_f ather_of . (Fig. 1b) .

Thus, the intuitive meaning of the above proof is that we can map degree of freedom in the logical space to that in R d .

The intersection and projection constraints can be imposed with parameter sharing.

We describe how shared parameters are initialized and trained.

From initialization, we bind parameters so that they satisfy the two constraints.

For each entity e j , we assign a d-dimensional vector #» e j .

To each R i , we assign (H i , #» r i ) (or (A i , #» r i )) with parameter sharing.

We first construct the H's.

Intersection constraint We define the H's top-down, first defining the intersections and then the subspaces that go through it.

To the head R h , assign a h linearly independent rows for the basis of H h .

Then, to each R i that is not a head, additionally assign a i rows linearly independent to the bases of all of its parents, and construct H i with its bases and the bases of all of its parents.

Projection matrices can be uniquely constructed given the bases (Strang) .

Now, we initlialize the #» r i 's. Projection Constraint To the head R h , pick any random x h ∈ R d and assign #» r h = P h x. To each non-head R i whose parent is R p , assign

Parameters to be trained Such initialization leaves the following parameters given a KG with entities e j 's and relations r i 's: (1) A h for the head relation, (2) c i for each non-head relation, (3) #»

x i for each head and non-head relation, (4) #» e j for each entity e j .

4.1.1 TRAINING We construct negative examples (wrong fact triplets) and train with a margin-based loss, following the same protocols as in TransE and TransH. Training Objective We adopt the same loss function as in TransH. For each fact triplet (h, r i , t), we define the score function f (h, r i , t) = ||P i ( # » t − h) − #» r i || 2 and train a margin-based loss L which is aggregates f 's and discriminates between correct and negative examples.

where G is the set of all triples in the KG and (h , r i , t ) is a negative triple made from corrupting (h, r i , t).

We minimize this objective with stochastic gradient descent.

Automatic Grounding of Positive Triples The parameter sharing scheme guarantees two advantages during all steps of training.

First, the intersection and projection constraint are met not only at initialization but always.

Second, traversing through a particular (h, r i , t) also automatically executes training with (h, r p , t) for any r i ⇒ r p .

For example, by traversing (Tom, is_father_of, Harry) in the KG, the model automatically also traverses (Tom, is_parent_of, Harry), (Tom, is_family_of, Harry), even if the two triples are missing in the KG.

This is because P p P i = P p with the given initialization (section 4.1.1) and thus,

= f (h, r i , t) In other words, training f (h, r i , t) towards less than automatically guarantees, or has the effect of training f (h, r p , t) towards less than .

This enables the model to be automatically trained with what exists in the KG, eliminating the need to manually create missing triples that are true by implication rule.

We evaluate TransINT on Freebase 122 (respectively created by Vendrov et al. and Guo et al.) against the current state-of-the-art method.

The task is to predict the gold entity given a fact triple with missing head or tail -if (h, r, t ) is a fact triple in the test set, predict h given (r, t) or predict t given (h, r).

We follow TransE and KALE's protocol.

For each test triple (h, r, t ), we rank the similarity score (f (e, r, t) when h is replaced with e for every entity e in the KG, and identify the rank of the gold head entity h; we do the same for the tail entity t. Aggregated over all test triples, we report: (i) the mean reciprocal rank (MRR), (ii) the median of the ranks (MED), and (iii) the proportion of ranks no larger than n (HITS@N), which are the same metrics reported by KALE.

A lower MED and higher MRR and Hits HITS@N are better.

TransH and KALE adopt a "filtered" setting that addresses when entities that are correct, albeit not gold, are ranked before the gold entity.

For example, if the gold entity is (Tom, is_parent_of, John) and we rank every entity e for being the head of (?, is_parent_of, John), it is possible that Sue, John's mother, gets ranked before Tom.

To avoid this, the "filtered setting" ignore corrupted triplets that exist in the KG when counting the rank of the gold entity. (The setting without this is called the "raw setting").

We compare our performance with that of KALE and previous methods (TransE, TransH, TransR) that were compared against it, using the same dataset (FB122).

FB122 is a subset of FB15K (Bordes et al.) accompanied by 47 implication and transitive rules; it consists of 122 Freebase relations on "people", "location", and "sports" topics.

Since we use the same train/ test/ validation sets, we directly copy from Guo et al. for reporting on these baselines.

5.1.1 DETAILS OF TRAINING TransINT's hyperparameters are: learning rate (η), margin (γ), embedding dimension (d), and learning rate decay (α), applied every 10 epochs to the learning rate.

We find optimal configurations among the following candidates: η ∈ {0.003, 0.005, 0.01}, γ ∈ {1, 2, 5, 10}, d ∈ {50, 100}, α, ∈ {1.0, 0.98, 0.95}. We create 100 mini-batches of the training set and train for maximum of 1000 epochs with early stopping based on the best median rank.

Furthermore, we try training with and without normalizing each of entity vectors, relation vectors, and relation subspace bases after every batch of training.

5.1.2 EXPERIMENT SETTINGS Out of the 47 rules in FB122, 9 are transitive rules (such as person/nationality(x,y) ∧ country/official_language(y,z) ⇒ person/languages(x,z)) to be used for KALE.

However, since TransINT only deals with implication rules, we do not take advantage of them, unlike KALE.

We also put us on some intentional disadvantages against KALE to assess TransINT's robustness to absence of negative example grounding.

In constructing negative examples for the margin-based loss L, KALE both uses rules (by grounding) and their own scoring scheme to avoid false negatives.

While grounding with FB122 is not a burdensome task, it known to be very inefficient and difficult for extremely large datasets (Ding et al.) .

Thus, it is a great advantage for a KG model to perform well without grounding of training/ test data.

We evaluate TransINT on two settings for avoiding false negative examples; using rule grounding and only avoiding ones that exist in the KG.

We call them respectively TransINT G (grounding), TransINT N G (no grounding).

Table 1 .

While the filtered setting gives better performance (as expected), the trend is generally similar between raw and filtered.

TransINT outperforms all other models by large margins in all metrics, even without grounding; especially in the filtered setting, the Hits@N gap between TransINT G and KALE is around 4∼6 times that between KALE and the best performing Trans Baseline (TransR).

Also, while TransINT G performs higher than TransINT N G in all settings/metrics, the gap between them is much smaller than the that between TransINT N G and KALE, showing that TransINT robustly brings state-of-the-art performance even without grounding.

The results suggest two possibilities in a more general sense.

First, the emphasis of true positives could be as important as/ more important than avoiding false negatives.

Even without manual grounding, TransINT N G has automatic grounding of positive training instances enabled (Section 4.1.1.) due to model properties, and this could be one of its success factors.

Second, hard constraint on parameter structures can bring performance boost uncomparable to that by regularization or joint learning, which are softer constraints.

We also note that norm regularization of any parameter did not help in training TransINT, unlike stated in TransE, TransH, and KALE.

Instead, it was important to use a large margin (either γ = 5 or γ = 10).

The task is to classify whether an unobserved instance (h, r, t) is correct or not, where the test set consists of positive and negative instances.

We use the same protocol and test set provided by KALE; for each test instance, we evaluate its similarity score f (h, r, t) and classify it as "correct" if f (h, r, t) is below a certain threshold (σ), a hyperparameter to be additionally tuned for this task.

We report on mean average precision (MAP), the mean of classification precision over all distinct relations (r's) of the test instances.

We use the same experiment settings/ training details as in Link Prediction other than additionally finding optimal σ.

5.2.1 RESULTS Triple Classification results are shown in Table 2 .

Again, TransINT G and TransINT N G both significantly outperform all other baselines.

We also separately analyze MAP for relations that are/ are not affected by the implication rules (those that appear/ do not appear in the rules), shown in parentheses of Table 2 with the order of (influenced relations/ uninfluenced relations).

We can see that both TransINT's have MAP higher than the overall MAP of KALE, even when the TransINT's have the penalty of being evaluated only on uninfluenced relations; this shows that TransINT generates better embeddings even for those not affected by rules.

Furthermore, we comment on the role of negative example grounding; we can see that grounding does not help performance on unaffected relations (i.e. 0.752 vs 0.761), but greatly boosts performance on those affected by rules (0.839 vs 0.709).

While TransINT does not necessitate negative example grounding, it does improve the quality of embeddings for those affected by rules.

Traditional embedding methods that map an object (i.e. words, images) to a singleton vector learn soft tendencies between embedded vectors, such as semantic similarity (Mikolov et al., Frome et al.) .

83.5 n/a 7 A common metric for such tendency is cosine similarity, or angle between two embddings.

TransINT extends such line of thought to semantic relatedness between groups of objects, with angles between relation spaces.

In Fig. 5b , one can observe that the closer the angle between two embedded regions, the larger the overlap in area.

For entities h and t to be tied by both relations r 1 , r 2 , t − h has to belong to the intersection of their relation spaces.

Thus, we hypothesize the following over any two relations r 1 , r 2 that are not explicitly tied by the pre-determined rules: Let V 1 be the set of # » t − h's in r 1 's relation space (denoted as Rel 1 ) and V 2 that of r 2 '

s.

Then,

(1) Angle between Rel 1 and Rel 2 represents semantic "disjointness" of r 1 , r 2 ; the more disjoint two relations, the closer their angle to 90

• .

When the angle between Rel 1 and Rel 2 is small, (2) if majority of V 1 belongs to the overlap of V 1 and V 2 but not vice versa, r 1 implies r 2 .

(3) if majority of V 1 and V 2 both belong to their overlap, r 1 and r 2 are semantically related.

Hypotheses (2) and (3) consider the imbalance of membership in overlapped regions.

Exact calculation of this involves specifying an appropriate (Fig. 3) .

As a proxy for deciding whether an element of V 1 (denote v 1 ) belongs in the overlapped region, we can consider the distance between v 1 to and its projection to Rel 2 ; the further away v 1 is from the overlapped region, the larger the projected distance (visualization available in our code repository).

We call the mean of such distances from V 1 to For hypothesis (1), we verified that the vast majority of relation pairs have angles near to 90

• , with the mean and median respectively 83.0

• and 85.4

• ; only 1% of all relation pairs had angles less than 50

• .

We observed that relation pairs with angle less than 20

• were those that can be inferred by transitively applying the pre-determined implication rules.

Relation pairs with angles within the range of [20

• ] had strong tendencies of semantic relatedness or implication; such tendency drastically weakened past 70

• .

Table 3 shows the angle and imb of relations with respect to /people/person/place_of_birth, whose trend agrees with our hypotheses.

While we only show a subset of the complete list, we note that almost all relation pairs generally follow such a tendency; the complete list can be accessed in our code repository.

Finally, we note that such analysis could be possible with TransH as well, since their method too maps # » t − h's to lines (Fig. 2b) .

Throughout target tasks (Link Prediction, Triple Classification) and semantics mining, TransINT's theme of optimal regions to bound entity sets is unified and consistent.

Furthermore, the integration of rules into embedding space geometrically coherent with KG embeddings alone.

These two qualities were missing in existing works such as TransE or KALE, and TransINT opens up new possibilities for applying KG embeddings to explainable reasoning in applications such as recommender systems (Ma et al.) and question answering (Hamilton et al.) .

Our work is related to three strands of work.

The first strand is Order Embeddings (Vendrov et al.) and their extensions (Vilnis et al.; Athiwaratkun & Wilson) , whose limitation we discussed in the introduction.

While Nickel & Kiela also approximately embed unary partial ordering, their focus is on achieving reasonably competent result with unsupervised learning of rules in low dimensions, while ours is achieving state-of-the-art in a supervised setting.

The second strand is those that enforce the satisfaction of common sense logical rules in the embedded KG.

Wang et al. explicitly constraints the resulting embedding to satisfy logical implications and type constraints via linear programming, but it only requires to do so during inference, not learning.

On the other hand,Guo et al. induces that embeddings follow a set of logical rules during learning, but their approach is soft induction not hardly constrain.

Our work combines the advantages of both works.

We presented TransINT, a new KG embedding method that embed sets of entities (tied by relations) to continuous sets in R d that are inclusion-ordered isomorphically to relation implications.

Our method achieved new state-of-the-art performances with signficant margins in Link Prediction and Triple Classification on the FB122 dataset, with boosted performance even on test instances that are not affected by rules.

We further propose and interpretable criterion for mining semantic similairty among sets of entities with TransINT.

Here, we provide the proofs for Main Theorems 1 and 2.

We also explain some concepts necessary in explaining the proofs.

We put * next to definitions and theorems we propose/ introduce.

Otherwise, we use existing definitions and cite them.

We explain in detail elements of R d that were intuitively discussed.

In this and later sections, we mark all lemmas and definitions that we newly introduce with * ; those not marked with * are accompanied by reference for proof.

We denote all d × d matrices with capital letters (ex) A) and vectors with arrows on top (ex) #» b ).

The linear subspace given by

that are solutions to the equation; its rank is the number of constraints A(x − #» b ) = 0 imposes.

For example, in R 3 , a hyperplane is a set of #» x = [ x 1 , x 2 , x 3 ] ∈ R 3 such that ax 1 + bx 2 + cx 3 − d = 0 for some scalars a, b, c, d; because vectors are bound by one equation (or its "A" only really contains one effective equation), a hyperplane's rank is 1 (equivalently rank(A) = 1).

On the other hand, a line in R 3 imposes to 2 constraints, and its rank is 2 (equivalently rank(A) = 2).

Consider two linear subspaces H 1 , H 2 , each given by A 1 ( #» x − #» b 1 ) = 0, A 2 ( #» x − #» b 2 ) = 0.

Then,

by definition.

In the rest of the paper, denote H i as the linear subspace given by some A i ( #» x − #» b i ) = 0.

Invariance For all #» x on H, projecting #» x onto H is still #» x ; the converse is also true.

Lemma 1 P #» x = #» x ⇔ #» x ∈ H (Strang).

Orthogonality Projection decomposes any vector #» x to two orthogonal components -P #» x and (I − P ) #» x (Figure 4) .

Thus, for any projection matrix P , I − P is also a projection matrix that is orthogonal to P (i.e. P (I − P ) = 0) (Strang).

Lemma 2 Let P be a projection matrix.

Then I −P is also a projection matrix such that P (I −P ) = 0 (Strang).

The following lemma also follows.

Lemma 3 ||P #» x || ≤ ||P #» x + (I − P ) #» x || = || #» x || (Strang).

Projection onto an included space If one subspace H 1 includes H 2 , the order of projecting a point onto them does not matter.

For example, in Figure 3 , a random point #» a in R 3 can be first projected onto H 1 at #» b , and then onto H 3 at #» d .

On the other hand, it can be first projected onto H 3 at #» d , and then onto H 1 at still #» d .

Thus, the order of applying projections onto spaces that includes one another does not matter.

If we generalize, we obtain the following two lemmas (Figure 6 ):

Lemma 4 * Every two subspaces H 1 ⊂ H 2 if and only if P 1 P 2 = P 2 P 1 = P 1 .

proof) By Lemma 1, if H 1 ⊂ H 2 , then P 2 #» x = #» x ∀ #» x ∈ H 1 .

On the other hand, if H 1 ⊂ H 2 , then there is some #» x ∈ H 1 , #» x ∈ H 2 such that P 2 #» x = #» x .

Thus,

⇔ ∀ #» y , P 2 (P 1 #» y ) = P 1 #» y ⇔ P 2 P 1 = P 1 .

Because projection matrices are symmetric (Strang), P 2 P 1 = P 1 = P 1 T = P 1 T P 2 T = P 1 P 2 .

Lemma 5 * For two subspaces H 1 , H 2 and vector #» k ∈ H 2 ,

<|TLDR|>

@highlight

We propose TransINT, a novel and interpretable KG embedding method that isomorphically preserves the implication ordering among relations in the embedding space in an explainable, robust, and geometrically coherent way.