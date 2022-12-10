Answering complex logical queries on large-scale incomplete knowledge graphs (KGs) is a fundamental yet challenging task.

Recently, a promising approach to this problem has been to embed KG entities as well as the query into a vector space such that entities that answer the query are embedded close to the query.

However, prior work models queries as single points in the vector space, which is problematic because a complex query represents a potentially large set of its answer entities, but it is unclear how such a set can be represented as a single point.

Furthermore, prior work can only handle queries that use conjunctions ($\wedge$) and existential quantifiers ($\exists$).

Handling queries with logical disjunctions ($\vee$) remains an open problem.

Here we propose query2box, an embedding-based framework for reasoning over arbitrary queries with $\wedge$, $\vee$, and $\exists$ operators in massive and incomplete KGs.

Our main insight is that queries can be embedded as boxes (i.e., hyper-rectangles), where a set of points inside the box corresponds to a set of answer entities of the query.

We show that conjunctions can be naturally represented as intersections of boxes and also prove a negative result that handling disjunctions would require embedding with dimension proportional to the number of KG entities.

However, we show that by transforming queries into a Disjunctive Normal Form, query2box is capable of handling arbitrary logical queries with $\wedge$, $\vee$, $\exists$ in a scalable manner.

We demonstrate the effectiveness of query2box on two large KGs and show that query2box achieves up to 25% relative improvement over the state of the art.

Knowledge graphs (KGs) capture different types of relationships between entities, e.g., Canada citizen −−−−→ Hinton.

Answering arbitrary logical queries, such as "where did Canadian citizens with Turing Award graduate?", over such KGs is a fundamental task in question answering, knowledge base reasoning, as well as AI more broadly.

First-order logical queries can be represented as Directed Acyclic Graphs (DAGs) ( Fig. 1(A) ) and be reasoned according to the DAGs to obtain a set of answers ( Fig. 1(C) ).

While simple and intuitive, such approach has many drawbacks: (1) Computational complexity of subgraph matching is exponential in the query size, and thus cannot scale to modern KGs; (2) Subgraph matching is very sensitive as it cannot correctly answer queries with missing relations.

To remedy (2) one could impute missing relations (Koller et al., 2007; Džeroski, 2009; De Raedt, 2008; Nickel et al., 2016) but that would only make the KG denser, which would further exacerbate issue (1) (Dalvi & Suciu, 2007; Krompaß et al., 2014) .

Recently, a promising alternative approach has emerged, where logical queries as well as KG entities are embedded into a low-dimensional vector space such that entities that answer the query are embedded close to the query (Guu et al., 2015; Hamilton et al., 2018; Das et al., 2017) .

Such approach robustly handles missing relations (Hamilton et al., 2018) and is also orders of magnitude faster, as answering an arbitrary logical query is reduced to simply identifying entities nearest to the embedding of the query in the vector space.

However, prior work embeds a query into a single point in the vector space.

This is problematic because answering a logical query requires modeling a set of active entities while traversing the KG ( Fig. 1(C) ), and how to effectively model a set with a single point is unclear.

Furthermore, it We then obtain query embedding according to the computation graph (B) as a sequence of box operations: start with two nodes TuringAward and Canada and apply Win and Citizen projection operators, followed by an intersection operator (denoted as a shaded intersection of yellow and orange boxes) and another projection operator.

The final embedding of the query is a green box and query's answers are the entities inside the box.

is also unnatural to define logical operators (e.g., set intersection) of two points in the vector space.

Another fundamental limitation of prior work is that it can only handle conjunctive queries, a subset of first-order logic that only involves conjunction (∧) and existential quantifier (∃), but not disjunction (∨).

It remains an open question how to handle disjunction effectively in the vector space.

Here we present QUERY2BOX, an embedding-based framework for reasoning over KGs that is capable of handling arbitrary Existential Positive First-order (EPFO) logical queries (i.e., queries that include any set of ∧, ∨, and ∃) in a scalable manner.

First, to accurately model a set of entities, our key idea is to use a closed region rather than a single point in the vector space.

Specifically, we use a box (axis-aligned hyper-rectangle) to represent a query ( Fig. 1(D) ).

This provides three important benefits: (1) Boxes naturally model sets of entities they enclose; (2) Logical operators (e.g., set intersection) can naturally be defined over boxes similarly as in Venn diagrams (Venn, 1880); (3) Executing logical operators over boxes results in new boxes, which means that the operations are closed; thus, logical reasoning can be efficiently performed in QUERY2BOX by iteratively updating boxes according to the query computation graph ( Fig. 1(B)(D) ).

We show that QUERY2BOX can naturally handle conjunctive queries.

We first prove a negative result that embedding EPFO queries to only single points or boxes is intractable as it would require embedding dimension proportional to the number of KG entities.

However, we provide an elegant solution, where we transform a given EPFO logical query into a Disjunctive Normal Form (DNF) (Davey & Priestley, 2002) , i.e., disjunction of conjunctive queries.

Given any EPFO query, QUERY2BOX represents it as a set of individual boxes, where each box is obtained for each conjunctive query in the DNF.

We then return nearest neighbor entities to any of the boxes as the answers to the query.

This means that to answer any EPFO query we first answer individual conjunctive queries and then take the union of the answer entities.

We evaluate QUERY2BOX on standard KG benchmarks and show: (1) QUERY2BOX provides strong generalization as it can answer complex queries that it has never seen during training; (2) QUERY2BOX is robust as it can answer any EPFO query with high accuracy even when relations involving answering the query are missing in the KG; (3) QUERY2BOX provides up to 25% relative improvement in accuracy of answering EPFO queries over state-of-the-art baselines.

Most related to our work are embedding approaches for multi-hop reasoning over KGs (Bordes et al., 2013; Das et al., 2017; Guu et al., 2015; Hamilton et al., 2018) .

Crucial difference is that we provide a way to tractably handle a larger subset of the first-order logic (EPFO queries vs. conjunctive queries) and that we embed queries as boxes, which provides better accuracy and generalization.

Second line of related work is on structured embeddings, which associate images, words, sentences, or knowledge base concepts with geometric objects such as regions (Erk, 2009; Vilnis et al., 2018; Li et al., 2019) , densities (Vilnis & McCallum, 2014; He et al., 2015; Athiwaratkun & Wilson, 2018) , and orderings (Vendrov et al., 2015; Lai & Hockenmaier, 2017; Li et al., 2017) .

While the above work uses geometric objects to model individual entities and their pairwise relations, we use the geometric objects to model sets of entities and reason over those sets.

In this sense our work is also related to classical Venn Diagrams (Venn, 1880) , where boxes are essentially the Venn Diagrams in vector space, but our boxes and entity embeddings are jointly learned, which allows us to reason over incomplete KGs.

Here we present the QUERY2BOX, where we will define an objective function that allows us to learn embeddings of entities in the KG, and at the same time also learn parameterized geometric logical operators over boxes.

Then given an arbitrary EPFO query q ( Fig. 1(A) ), we will identify its computation graph ( Fig. 1(B) ), and embed the query by executing a set of geometric operators over boxes ( Fig. 1(D) ).

Entities that are enclosed in the final box embedding are returned as answers to the query ( Fig. 1(D) ).

In order to train our system, we generate a set of queries together with their answers at training time and then learn entity embeddings and geometric operators such that queries can be accurately answered.

We show in the following sections that our approach is able to generalize to queries and query structures never seen during training.

We denote a KG as G = (V, R), where v ∈ V represents an entity, and r ∈ R is a binary function r : V × V → {True, False}, indicating whether the relation r holds between a pair of entities or not.

In the KG, such binary output indicates the existence of the directed edge between a pair of entities, i.e., v

Conjunctive queries are a subclass of the first-order logical queries that use existential (∃) and conjunction (∧) operations.

They are formally defined as follows.

where v a represents non-variable anchor entity, V 1 , . . .

, V k are existentially quantified bound variables, V ? is the target variable.

The goal of answering the logical query q is to find a set of entities q ⊆ V such that v ∈ q iff q[v] = True.

We call q the denotation set (i.e., answer set) of query q.

As shown in Fig. 1(A) , the dependency graph (DG) is a graphical representation of conjunctive query q, where nodes correspond to variable or non-variable entities in q and edges correspond to relations in q. In order for the query to be valid, the corresponding DG needs to be a Directed Acyclic Graph (DAG), with the anchor entities as the source nodes of the DAG and the query target V ?

as the unique sink node (Hamilton et al., 2018) .

From the dependency graph of query q, one can also derive the computation graph, which consists of two types of directed edges that represent operators over sets of entities:

• Projection:

Given a set of entities S ⊆ V, and relation r ∈ R, this operator obtains

• Intersection:

Given a set of entity sets {S 1 , S 2 , . . .

, S n }, this operator obtains ∩ n i=1 S i .

For a given query q, the computation graph specifies the procedure of reasoning to obtain a set of answer entities, i.e., starting from a set of anchor nodes, the above two operators are applied iteratively until the unique sink target node is reached.

The entire procedure is analogous to traversing KGs following the computation graph (Guu et al., 2015) .

So far we have defined conjunctive queries as computation graphs that can be executed directly over the nodes and edges in the KG.

Now, we define logical reasoning in the vector space.

Our intuition follows Fig. 1 :

Given a complex query, we shall decompose it into a sequence of logical operations, and then execute these operations in the vector space.

This way we will obtain the embedding of the query, and answers to the query will be entities that are enclosed in the final query embedding box.

In the following, we detail our two methodological advances: (1) the use of box embeddings to efficiently model and reason over sets of entities in the vector space, and (2) how to tractably handle disjunction operator (∨), expanding the class of first-order logic that can be modeled in the vector space (Section 3.3).

Box embeddings.

To efficiently model a set of entities in the vector space, we use boxes (i.e., axis-aligned hyper-rectangles).

The benefit is that unlike a single point, the box has the interior; thus, if an entity is in a set, it is natural to model the entity embedding to be a point inside the box.

Formally, we operate on R d , and define a box in R d by p = (Cen(p), Off(p)) ∈ R 2d as:

where is element-wise inequality, Cen(p) ∈ R d is the center of the box, and Off(p) ∈ R d ≥0 is the positive offset of the box, modeling the size of the box.

Each entity v ∈ V in KG is assigned a single vector v ∈ R d (i.e., a zero-size box), and the box embedding p models {v ∈ V : v ∈ Box p }, i.e., a set of entities whose vectors are inside the box.

For the rest of the paper, we use the bold face to denote the embedding, e.g., embedding of v is denoted by v.

Our framework reasons over KGs in the vector space following the computation graph of the query, as shown in Fig. 1 (D): we start from the initial box embeddings of the source nodes (anchor entities) and sequentially update the embeddings according to the logical operators.

Below, we describe how we set initial box embeddings for the source nodes, as well as how we model projection and intersection operators (defined in Sec. 3.1) as geometric operators that operate over boxes.

After that, we describe our entity-to-box distance function and the overall objective that learns embeddings as well as the geometric operators.

Initial boxes for source nodes.

Each source node represents an anchor entity v ∈ V, which we can regard as a set that only contains the single entity.

Such a single-element set can be naturally modeled by a box of size/offset zero centerd at v. Formally, we set the initial box embedding as (v, 0), where v ∈ R d is the anchor entity vector and 0 is a d-dimensional all-zero vector.

Geometric projection operator.

We associate each relation r ∈ R with relation embedding r = (Cen(r), Off(r)) ∈ R 2d with Off(r) 0.

Given an input box embedding p, we model the projection by p + r, where we sum the centers and sum the offsets.

This gives us a new box with the translated center and larger offset because Off(r) 0, as illustrated in Fig. 2 (A).

The adaptive box size effectively models a different number of entities/vectors in the set.

Geometric intersection operator.

We model the intersection of a set of box embeddings {p 1 , . . .

, p n } as p inter = (Cen(p inter ), Off(p inter )), which is calculated by performing attention over the box centers (Bahdanau et al., 2014) and shrinking the box offset using the sigmoid function:

where is the dimension-wise product, MLP(·) :

is the sigmoid function, DeepSets(·) is the permutation-invariant deep architecture (Zaheer et al., 2017) , and both Min(·) and exp(·) are applied in a dimension-wise manner.

Following Hamilton et al. (2018) , we model all the deep sets by DeepSets({x 1 , . . .

,

where all the hidden dimensionalities of the two MLPs are the same as the input dimensionality.

The intuition behind our geometric intersection is to generate a smaller box that lies inside a set of boxes, as illustrated in Fig. 2

1 Different from the generic deep sets to model the intersection (Hamilton et al., 2018) , our geometric intersection operator effectively constrains the center position and models the shrinking set size.

Entity-to-box distance.

Given a query box q ∈ R 2d and an entity vector v ∈ R d , we define their distance as

where

and 0 < α < 1 is a fixed scalar, and

As illustrated in Fig. 2(C) , dist outside corresponds to the distance between the entity and closest corner/side of the box.

Analogously, dist inside corresponds to the distance between the center of the box and its side/corner (or the entity itself if the entity is inside the box).

The key here is to downweight the distance inside the box by using 0 < α < 1.

This means that as long as entity vectors are inside the box, we regard them as "close enough" to the query center (i.e., dist outside is 0, and dist inside is scaled by α).

When α = 1, dist box reduces to the ordinary L 1 distance, i.e., Cen(q) − v 1 , which is used by the conventional TransE (Bordes et al., 2013) as well as prior query embedding methods (Guu et al., 2015; Hamilton et al., 2018) .

Training objective.

Our next goal is to learn entity embeddings as well as geometric projection and intersection operators.

Given a training set of queries and their answers, we optimize a negative sampling loss (Mikolov et al., 2013) to effectively optimize our distance-based model (Sun et al., 2019) :

where γ represents a fixed scalar margin, v ∈ q is a positive entity (i.e., answer to the query q), and v i / ∈ q is the i-th negative entity (non-answer to the query q) and k is the number of negative entities.

So far we have focused on conjunctive queries, and our aim here is to tractably handle in the vector space a wider class of logical queries, called Existential Positive First-order (EPFO) queries (Dalvi & Suciu, 2012) that involve ∨ in addition to ∃ and ∧. We specifically focus on EPFO queries whose computation graphs are a DAG, same as that of conjunctive queries (Section 3.1), except that we now have an additional type of directed edge, called union defined as follows:

• Union:

Given a set of entity sets {S 1 , S 2 , . . .

, S n }, this operator obtains

A straightforward approach here would be to define another geometric operator for union and embed the query as we did in the previous sections.

An immediate challenge for our box embeddings is that boxes can be located anywhere in the vector space, so their union would no longer be a simple box.

In other words, union operation over boxes is not closed.

Theoretically, we can prove a general negative result for any embedding-based method that maps query q into q such that dist(v; q) ≤ β iff v ∈ q .

Here, dist(v; q) is the distance between entity and query embeddings, e.g., dist box (v; q) or v − q 1 , and β is a fixed threshold.

Theorem 1.

Consider any M conjunctive queries q 1 , . . .

, q M whose denotation sets q 1 , . . .

, q M are disjoint with each other, ∀ i = j, q i ∩ q j = ∅. Let D be the VC dimension of the function class {sign(β − dist(·; q)) : q ∈ Ξ}, where Ξ represents the query embedding space and sign(·) is the sign function.

Then, we need D ≥ M to model any EPFO query, i.e., dist(v; q) ≤ β ⇔ v ∈ q is satisfied for every EPFO query q.

The proof is provided in Appendix A, where the key is that the introduction of the union operation forces us to model the powerset {∪ q i ∈S q i : S ⊆ {q 1 , . . .

, q M }} in a vector space.

For a real-world KG, there are M ≈ |V| conjunctive queries with non-overlapping answers.

For example, in the commonly-used FB15k dataset (Bordes et al., 2013) , derived from the Freebase (Bollacker et al., 2008) , we find M = 13,365, while |V| is 14,951 (see Appendix B for the details).

Theorem 1 shows that in order to accurately model any EPFO query with the existing framework, the complexity of the distance function measured by the VC dimension needs to be as large as the number of KG entities.

This implies that if we use common distance functions based on hyper-plane, Enclidian sphere, or axis-aligned rectangle, 2 their parameter dimensionality needs to be Θ(M ), which is Θ(|V|) for real KGs we are interested in.

In other words, the dimensionality of the logical query embeddings needs to be Θ(|V|), which is not low-dimensional; thus not scalable to large KGs and not generalizable in the presence of unobserved KG edges.

To rectify this issue, our key idea is to transform a given EPFO query into a Disjunctive Normal Form (DNF) (Davey & Priestley, 2002) , i.e., disjunction of conjunctive queries, so that union operation only appears in the last step.

Each of the conjunctive queries can then be reasoned in the low-dimensional space, after which we can aggregate the results by a simple and intuitive procedure.

In the following, we describe the transformation to DNF and the aggregation procedure.

Transformation to DNF.

Any first-order logic can be transformed into the equivalent DNF (Davey & Priestley, 2002) .

We perform such transformation directly in the space of computation graph, i.e., moving all the edges of type "union" to the last step of the computation graph.

Let G q = (V q , E q ) be the computation graph for a given EPFO query q, and let V union ⊂ V q be a set of nodes whose in-coming edges are of type "union".

For each v ∈ V union , define P v ⊂ V q as a set of its parent nodes.

We first generate N = v∈Vunion |P v | different computation graphs G q (1) , . . .

, G q (N ) as follows, each with different choices of v parent in the first step.

1.

For every v ∈ V union , select one parent node v parent ∈ P v .

2.

Remove all the edges of type 'union.' 3.

Merge v and v parent , while retaining all other edge connections.

We then combine the obtained computation graphs G q (1) , . . .

, G q (N ) as follows to give the final equivalent computation graph.

1.

Convert the target sink nodes of all the obtained computation graphs into the existentially quantified bound variables nodes.

2.

Create a new target sink node V ? , and draw directed edges of type "union" from all the above variable nodes to the new target node.

An example of the entire transformation procedure is illustrated in Fig. 3 .

By the definition of the union operation, our procedure gives the equivalent computation graph as the original one.

Furthermore, as all the union operators are removed from G q (1) , . . .

, G q (N ) , all of these computation graphs represent conjunctive queries, which we denote as q (1) , . . .

, q (N ) .

We can then apply existing framework to obtain a set of embeddings for these conjunctive queries as q (1) , . . .

, q (N) .

Aggregation.

Next we define the distance function between the given EPFO query q and an entity v ∈ V. Since q is logically equivalent to q (1) ∨ · · · ∨ q (N ) , we can naturally define the aggregated distance function using the box distance dist box :

where dist agg is parameterized by the EPFO query q. When q is a conjunctive query, i.e., N = 1, dist agg (v; q) = dist box (v; q).

For N > 1, dist agg takes the minimum distance to the closest box as the distance to an entity.

This modeling aligns well with the union operation; an entity is inside the union of sets as long as the entity is in one of the sets.

Note that our DNF-query rewriting scheme is general and is able to extend any method that works for conjunctive queries (e.g., (Hamilton et al., 2018) ) to handle more general class of EPFO queries.

Computational complexity.

The computational complexity of answering an EPFO query with our framework is equal to that of answering the N conjunctive queries.

In practice, N might not be so large, and all the N computations can be parallelized.

Furthermore, answering each conjuctive query is very fast as it requires us to execute a sequence of simple box operations (each of which takes constant time) and then perform a range search (Bentley & Friedman, 1979) in the embedding space, which can also be done in constant time using techniques based on Locality Sensitive Hashing (Indyk & Motwani, 1998) .

Our goal in the experiment section is to evaluate the performance of QUERY2BOX on discovering answers to complex logical queries that cannot be obtained by traversing the incomplete KG.

This means, we will focus on answering queries where one or more missing edges in the KG have to be successfully predicted in order to obtain the additional answers.

Figure 4: Query structures considered in the experiments, where anchor entities and relations are to be specified to instantiate logical queries.

Naming for each query structure is provided under each subfigure, where 'p', 'i', and 'u' stand for 'projection', 'intersection', and 'union', respectively.

Models are trained on the first 5 query structures, and evaluated on all 9 query structures.

Table 2 : Number of training, validation, and test queries generated for different query structures.

We perform experiments on standard KG benchmarks, FB15k (Bordes et al., 2013) and FB15k-237 (Toutanova & Chen, 2015) .

Both are subsets of Freebase (Bollacker et al., 2008) , a large-scale KG containing general facts.

Dataset statistics are shown in Table 1 .

We follow the standard evaluation protocol in KG literture: Given the standard split of edges into training, test, and validation sets (Table 1) , we first augment the KG to also include inverse relations and effectively double the number of edges in the graph.

We then create three graphs: G train , which only contains training edges and we use this graph to train node embeddings as well as box operators.

We then also generate two bigger graphs: G valid which contains G train plus the validation edges, and G test , which includes G valid as well as the test edges.

We consider 9 kinds of diverse query structures shown and named in Fig. 4 .

We use 5 query structures for training and then evaluate on all the 9 query structures.

Given a query q, let q train , q val , and q test denote a set of answer entities obtained by running subgraph matching of q on G train , G valid , and G test , respectively.

(We refer the reader to Appendix C for full details on query generation.)

At the training time, we use q train as positive examples for the query and other random entities as negative examples (Table 2) .

However, at the test/validation time we proceed differently.

Note that we focus on answering queries where generalization performance is crucial and at least one edge needs to be imputed in order to answer the queries.

Thus, rather than evaluating a given query on the full validation (or test) set q val ( q test ) of answers, we validate the method only on non-trivial answers that include missing relations.

Given how we constructed G train ⊆ G valid ⊆ G test , we have q train ⊆

q val ⊆ q test and thus we evaluate the method on q val \ q train to tune hyper-parameters and then report results identifying answer entities in q test \ q val .

This means we always evaluate on queries/entities that were not part of the training set and the method has not seen them before.

Given a test query q, for each of its non-trivial answers v ∈ q test \ q val , we use dist box in Eq. 3 to rank v among V\ q test .

Denoting the rank of v by Rank(v), we then calculate evaluation metrics for answering query q, such as Mean Reciprocal Rank (MRR) and Hits at K (H@K):

where f metrics (x) = We then average Eq. 6 over all the queries within the same query structure, 3 and report the results separately for different query structures.

The same evaluation protocol is applied to the validation stage except that we evaluate on q val \ q train rather than q test \ q val .

We compare our framework QUERY2BOX against the state-of-the-art GQE (Hamilton et al., 2018) .

GQE embeds a query to a single vector, and models projection and intersection operators as translation and deep sets (Zaheer et al., 2017) , respectively.

The L 1 distance is used as the distance between query and entity vectors.

For a fair comparison, we also compare with GQE-DOUBLE (GQE with doubled embedding dimensionality) so that QUERY2BOX and GQE-DOUBLE have the same amount of parameters.

Although the original GQE cannot handle EPFO queries, we apply our DNF-query rewriting strategy and in our evaluation extend GQE to handle general EPFO queries as well.

Furthermore, we perform extensive ablation study by considering several variants of QUERY2BOX (abbreviated as Q2B).

We list our method as well as its variants below.

• Q2B (our method).

The box embeddings are used to model queries, and the attention mechanism is used for the intersection operator.

• Q2B-AVG.

The attention mechanism for intersection is replaced with averaging.

• Q2B-DEEPSETS.

The attention mechanism for intersection is replaced with the deep sets.

• Q2B-AVG-1P.

The variant of Q2B-AVG that is trained with only 1p queries (see Fig. 4) ; thus, logical operators are not explicitly trained.

• Q2B-SHAREDOFFSET.

The box offset is shared across all queries (every query is represented by a box with the same size).

We use embedding dimensionality of d = 400 and set γ = 24, α = 0.2 for the loss in Eq. 4.

We train all types of training queries jointly.

In every iteration, we sample a minibatch size of 512 queries for each query structure (details in Appendix C), and we sample 1 answer entity and 128 negative entities for each query.

We optimize the loss in Eq. 4 using Adam Optimizer (Kingma & Ba, 2014) with learning rate = 0.0001.

We train all models for 250 epochs, monitor the performance on the validation set, and report the test performance.

We start by comparing our Q2B with state-of-the-art query embedding method GQE (Hamilton et al., 2018) on FB15k and FB15k-237.

As listed in Table 3 and Table 4 , our method significantly and consistently outperforms the state-of-the-art baseline across all the query structures, including those not seen during training as well as those with union operations.

On average, we obtain Table 6 : H@3 on test set for QUERY2BOX vs. several of its variants on FB15k.

9.8% (25% relative) and 3.8% (15% relative) higher H@3 than the best baselines on FB15k and FB15k-237, respectively.

Notice that naïvely increasing embedding dimensionality in GQE yields limited performance improvement.

Our Q2B is able to effectively model a set of entities by the box embedding, and achieves a large performance gain compared with GQE-DOUBLE (with same number of parameters) that represents queries as point vectors.

Also notice that Q2B performs well on new queries with the same structure as the training queries as well as on new query structures never seen during training.

We also conduct extensive ablation studies, which are summarized in Tables 5 and 6 :

Importance of attention mechanism.

First, we show that our modeling of intersection using the attention mechanism is important.

Given a set of box embeddings {p 1 , . . .

, p n }, Q2B-AVG is the most naïve way to calculate the center of the resulting box embedding p inter while Q2B-DEEPSETS is too flexible and neglects the fact that the center should be a weighted average of Cen(p 1 ), . . .

, Cen(p n ).

Compared with the two methods, Q2B achieves better performance in answering queries that involve intersection operation, e.g., 2i, 3i, pi, ip.

Specifically, on FB15k-237, Q2B obtains more than 4% and 2% absolute gain in H@3 compared to Q2B-AVG and Q2B-DEEPSETS, respectively.

Necessity of training on complex queries.

Second, we observe that explicitly training on complex logical queries beyond one-hop path queries (1p in Fig. 4) improves the reasoning performance.

Although Q2B-AVG-1P is able to achieve strong performance on 1p and 2u, where answering 2u is essentially answering two 1p queries with an additional minimum operation (see Eq. 5 in Section 3.3), Q2B-AVG-1P fails miserably in answering other types of queries involving logical operators.

On the other hand, other methods (Q2B, Q2B-AVG, and Q2B-DEEPSETS) that are explicitly trained on the logical queries achieve much higher accuracy, with up to 10% absolute average improvement of H@3 on FB15k.

Adaptive box size for different queries.

Third, we investigate the importance of learning adaptive offsets (box size) for different queries.

Q2B-SHAREDOFFSET is a variant of our Q2B where all the box embeddings share the same learnable offset.

Q2B-SHAREDOFFSET does not work well on all types of queries.

This is possibly because different queries have different numbers of answer entities, and the adaptive box size enables us to better model it.

In this paper we proposed a reasoning framework called QUERY2BOX that can effectively model and reason over sets of entities as well as handle EPFO queries in a vector space.

Given a logical query, we first transform it into DNF, embed each conjunctive query into a box, and output entities closest to their nearest boxes.

Our approach is capable of handling all types of EPFO queries scalably and accurately.

Experimental results on standard KGs demonstrate that QUERY2BOX significantly outperforms the existing work in answering diverse logical queries.

Proof.

To model any EPFO query, we need to at least model a subset of EPFO queries Q = {∨ q i ∈S q i : S ⊆ {q 1 , . . .

, q M }}, where the corresponding denotation sets are {∪ q i ∈S q i : S ⊆ {q 1 , . . .

, q M }}.

For the sake of modeling Q, without loss of generality, we consider assigning a single entity embedding v q i to all v ∈ q i , so there are M kinds of entity vectors, v q1 , . . .

, v q M .

To model all queries in Q, it is necessary to satisfy the following.

where q S is the embedding of query ∨ q i ∈S q i .

Eq. 7 means that we can learn the M kinds of entity vectors such that for every query in Q, we can obtain its embedding to model the corresponding set using the distance function.

Notice that this is agnostic to the specific algorithm to embed query ∨ q∈S q into q S ; thus, our result is generally applicable to any method that embeds the query into a single vector.

Crucially, satisfying Eq. 7 is equivalent to {sign(β − dist(·; q)) : q ∈ Ξ} being able to shutter {v q1 , . . .

, v q M }, i.e., any binary labeling of the points can be perfectly fit by some classifier in the function class.

To sum up, in order to model any EPFO query, we need to at least model any query in Q, which requires the VC dimension of the distance function to be larger than or equal to M .

Given the full KG G test for the FB15k dataset, our goal is to find conjunctive queries q 1 , . . .

, q M such that q 1 , . . .

, q M are disjoint with each other.

For conjunctive queries, we use two types of queries: '1p' and '2i' whose query structures are shown in Figure 4 .

On the FB15k, we instantiate 308,006 queries of type '1p', which we denote by S 1p .

Out of all the queries in S 1p , 129,717 queries have more than one answer entities, and we denote such a set of the queries by S 1p .

We then generate a set of queries of type '2i' by first randomly sampling two queries from S 1p and then taking conjunction; we denote the resulting set of queries by S 2i .

Now, we use S 1p and S 2i to generate a set of conjunctive queries whose denotation sets are disjoint with each other.

First, we prepare two empty sets V seen = ∅, and Q = ∅. Then, for every q ∈ S 1p , if V seen ∩ q = ∅ holds, we let Q ← Q ∪ {q} and V seen ← V seen ∪ q .

This procedure already gives us Q, where we have 10, 812 conjunctive queries whose denotation sets are disjoint with each other.

We can further apply the analogous procedure for S 2i , which gives us a further increased Q, where we have 13, 365 conjunctive queries whose denotation sets are disjoint with each other.

Therefore, we get M = 13, 365.

Given G train , G valid , and G test as defined in Section 4.1, we generate training, validation and test queries of different query structures.

During training, we consider the first 5 kinds of query structures.

For evaluation, we consider all the 9 query structures in Fig. 4 , containing query structures that are both seen and unseen during training time.

We instantiate queries in the following way.

Given a KG and a query structure (which is a DAG), we use pre-order traversal to assign an entity and a relation to each node and edge in the DAG of query structure to instantiate a query.

Namely, we start from the root of the DAG (which is the target node), we sample an entity e uniformly from the KG to be the root, then for every node connected to the root in the DAG, we choose a relation r uniformly from the in-coming relations of e in the KG, and a new entity e from the set of entities that reaches e by r in the KG.

Then we assign the relation r to the edge and e to the node, and move on the process based on the pre-order traversal.

This iterative process stops after we assign an entity and relation to every node and edge in DAG.

The leaf nodes in the DAG serve as the anchor nodes.

Note that during the entity and relation assignment, we specifically filter out all the degenerated queries, as shown in Fig. C .

Then we perform a post-order traversal of the DAG on the KG, starting from the anchor nodes, to obtain a set of answer entities to this query.

All of our generated datasets will be made publicly available.

Figure 5: Example of the degenerated queries, including (1) r and r −1 appear along one path and (2) same anchor node and relation in intersections.

When generating validation/test queries, we explicitly filter out trivial queries that can be fully answered by subgraph matching on G train /G valid .

We perform additional experiments on NELL995, which is presented in Xiong et al. (2017) .

Query generation and statistics.

Following Anonymous (2020), we first combine the validation and test sets with the training set to create the whole knowledge graph for NELL995.

Then we create new validation and test set splits by randomly selecting 20,000 triples each from the whole knowledge graph.

Note that we filter out all the entities that only appear in the validation and test sets but not in the training set.

The statistics of NELL995 are shown in Table 11 .

Based on the new splits, we sample queries in the same way as in FB15k and FB15k-237.

The statistics of the queries are listed in Table 12 .

Results.

The results comparing query2box and its baselines (GQE and the query2box variants) are shown in Tables 13, 14 , 15, 16.

Overall, we see the results follow the similar trend as the two FB15k datasets; our query2box outperforms GQE as well as its variants by a large margin.

<|TLDR|>

@highlight

Answering a wide class of logical queries over knowledge graphs with box embeddings in vector space