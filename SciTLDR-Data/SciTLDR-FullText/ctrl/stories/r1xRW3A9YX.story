Multi-relational graph embedding which aims at achieving effective representations with reduced low-dimensional parameters, has been widely used in knowledge base completion.

Although knowledge base data usually contains tree-like or cyclic structure, none of existing approaches can embed these data into a compatible space that in line with the structure.

To overcome this problem, a novel framework, called Riemannian TransE, is proposed in this paper to embed the entities in a Riemannian manifold.

Riemannian TransE models each relation as a move to a point and defines specific novel distance dissimilarity for each relation, so that all the relations are naturally embedded in correspondence to the structure of data.

Experiments on several knowledge base completion tasks have shown that, based on an appropriate choice of manifold, Riemannian TransE achieves good performance even with a significantly reduced parameters.

1.1 BACKGROUND Multi-relational graphs, such as social networks and knowledge bases, have a variety of applications, and embedding methods for these graphs are particularly important for these applications.

For instance, multi-relational graph embedding has been applied to social network analysis (KrohnGrimberghe et al., 2012) and knowledge base completion BID2 .

A multi-relational graph consists of entities V, a set R of relation types, and a collection of real data triples, where each triple (h, r, t) ∈ V × R × V represents some relation r ∈ R between a head entity h ∈ V and a tail entity t ∈ V. Embedding a multi-relational graph refers to a map from the entity and the relation set to some space.

Mathematical operations in this space enable many tasks, including clustering of entities and completion, prediction, or denoising of triples.

Indeed, completion tasks for knowledge bases attract considerable attention, because knowledge bases are known to be far from complete, as discussed in (West et al., 2014) BID13 .

Multi-relational graph embedding can help its completion and improve the performance of applications that use the graph.

This is the reason why much work focuses on multi-relational graph embedding.

FIG0 shows an example of a multi-relational graph and a completion task.

In multi-relational graph embedding, reducing the number of parameters is an important problem in the era of big data.

Many parameters are needed with tensor-factorization-based methods, such as Bayesian clustered tensor factorization (BCTF) (Sutskever et al., 2009) , RESCAL (Nickel et al., 2011) , and a neural tensor network (NTN) (Socher et al., 2013) , where each relation has a dense matrix or tensors (O D 2 or more parameters, where D is dimensionality of the space).

Thus, TransE BID2 was proposed to reduce the number of parameters, to overcome this problem.

In TransE, each entity is mapped to a point in Euclidean space and each relation is no more than a vector addition (O (D) parameters), rather than a matrix operation.

The successors to TransE, TransH (Wang et al., 2014) and TransD BID11 , also use only a small number of parameters.

Some methods succeeded in reducing parameters using diagonal matrices instead of dense matrices: e.g. DISTMULT (Yang et al., 2015) , ComplEx (Trouillon et al., 2016) , HolE (through the Fourier transform) (Nickel et al., 2016) , and ANALOGY BID15 .

In these methods, all relations share one space for embedding, but each relation uses its own dissimilarity criterion.

The success of these methods implies that one common space underlies whole data, and each relation can be regarded as a dissimilarity criterion in the space.

Whereas these methods use distances or inner products in Euclidean space as dissimilarity criteria, recent work has shown that using non-Euclidean space can further reduce the number of parameters.

One typical example of this is Poincaré Embedding (Nickel & Kiela, 2017) for hierarchical data, where a hyperbolic space is used as a space for embedding.

Here, the tree structure of hierarchical data has good compatibility with the exponential growth of hyperbolic space.

Recall the circumference with radius R is given by 2π sinh R(≈ 2π exp R) in a hyperbolic plane.

As a result, Poincaré embedding achieved good graph completion accuracy, even in low dimensionality such as 5 or 10.

On the other hand, spheres (circumference: 2π sin R) are compatible with cyclic structures.

Since Poincaré embedding, several methods have been proposed for single-relational graph embedding in non-Euclidean space (e.g. BID8 , (Nickel & Kiela, 2018) ) and shown good results.

The success of these methods suggests that the appropriate choice of a manifold (i.e., space) can retain low dimensionality, although these methods are limited to single-relational graph embedding.

According to the success of the TransE and its derivation and Poincaré embedding, it is reasonable in multi-relational graph embedding to assume the existence of a single structure compatible with a non-Euclidean manifold.

For example, we can consider a single tree-like structure, which contains multiple hierarchical structures, where root selection gives multiple hierarchical structures from a single tree, which is compatible with hyperbolic spaces (See Figure 2) .

Therefore, embedding in a single shared non-Euclidean manifold with multiple dissimilarity criteria used in TransE is promising.

Taking Poincaré embedding's success with low dimensionality into consideration, this method should work well (e.g., in graph completion tasks) with small number of parameters.

This is the main idea of this paper.

There are five entities and two kinds of relation (hypernym and synonym).

Graph completion refers to answering questions such as "is mammal a hypernym of cannis?"Figure 2: Multiple hierarchical relations in a single tree.

As this example shows, it is possible that multiple relations are given by multiple dissimilarity criteria in a single structure.

We propose a novel method, called Riemannian TransE, for multi-relation graph embedding using a non-Euclidean manifold.

In Riemannian TransE, the relations share one non-Euclidean space and the entities are mapped to the space, whereas each relation has its own dissimilarity criterion based on the distance in the space.

Specifically, the dissimilarity criteria in Riemannian TransE are similar to those in TransE BID2 ) based on vector addition, which is known to be effective.

Unfortunately, we cannot straightforwardly use TransE's dissimilarity criteria.

This is due to non-existence of a parallel vector field (See Figure 4) , which is implicitly but essentially used in "vector addition."

However, the parallel condition is not essential in TransE's idea.

For example, hierarchical bottom to top relations should be regarded as attraction to the top in the hierarchy, which is not parallel but has an attractive point.

Moreover, parallel vector fields can be regarded as a vector field attracted to a point at infinity.

Therefore, we replace parallel vector fields in TransE by vector fields with an attractive point that are well-defined in Riemannian manifolds, and as a result, we obtain Riemannian TransE. Advantages of non-Euclidean spaces enable our Riemannian TransE to achieve good performance (e.g. in graph completion) with low-dimensional parameters.

Riemannian TransE further exploits the advantages of TransE: that is, the method needs only O (D) parameters for each relation.

Numerical experiments on graph completion tasks show that with an appropriate choice of manifold, our method can improve the performance of multi-relational graph embedding with few parameters.

Let V and R denote the entities and relations in a multi-relational graph, and let T ⊂ V × R × V denote the triples in the graph.

Multi-relational graph embedding refers to a pair of maps from V and R into M e and M r , respectively.

Particularly, learning multi-relational graph embedding refers to obtaining an appropriate pair of maps v → p v (v ∈ V, p v ∈ M e ) and r → w r (r ∈ R, w r ∈ M r ) from the triples T .

In this paper, we call p v the planet of entity v, w r the launcher of relation r, and M e and M r the planet manifold and launcher manifold, respectively.

The quality of embedding is measured through a score function f : (M e × M e ) × M r → R, which is designed by each method.

Embedding is learned such that the value score function f (p h , p t ; w r ) will be low when p h , p t ; w r ∈ T and high when p h , p t ; w r / ∈ T .

For specific loss functions designed from the score function, see Subsection 2.3.

We interpret the score function of multi-relational graph embedding as dissimilarity in a manifold, which we call a satellite manifold M s .

We rewrite the score function f in multi-relational graph embedding using two maps H , T : M e × M r → M s and the dissimilarity measure function D : M s × M s → R as follows: DISPLAYFORM0 We call H and T the head and tail launch map, respectively, and call s H v;r and s T v;r the head and tail satellite of entity v (or of planet p v ) with respect to relation r. The idea of this formulation is embedding in one shared space with multiple dissimilarity criteria.

Specifically, each entity has only one planet and their satellite pairs give multiple dissimilarity criteria, each of which corresponds to a relation.

In other words, all of the relations shares one space and the planets in it, and the differences among the relations are reduced to the difference of their launcher maps and the satellites given by them.

We regard the planets as the embeddings of the entities, whereas dissimilarity between entities with respect to a relation is evaluated through their satellites which correspond to the relation.

A simple example of this is TransE BID2 , where all of the planets, satellites, and launchers share the same Euclidean space, i.e. M e = M s = M r = R D , the launch maps are given by vector addition as H (p; w) = p + w and T (p; w) = p, and the distance in a norm spacei.e.

the norm of the difference-is used as a dissimilarity criterion i.e. D s DISPLAYFORM1 (the L1 or L2 norm is often used in practice).

See Figure 5 (left).

As Nguyen (2017) suggested, one can associate the idea of representing relations as vector additions with the fact that we can find a relation through a substraction operator in Word2Vec Mikolov et al. (2013) .

That is, we can find relations such as p France − p Paris ≈ p Italy − p Rome in Word2Vec.

As explained above, TransE is based on the distance between satellites, and each satellite is given by simple vector addition.

Regardless of this simplicity, the performance of TransE has been exemplified in review papers (Nickel et al., 2016 ) (Nguyen, 2017 .

Indeed, the addition operation in a linear space is essential in the launcher map, and hence TransE can easily be extended to a Lie group, which is a manifold equipped with an addition operator, as suggested in BID5 .

Some methods, such as TransH (Wang et al., 2014) , TransR BID14 , and TransD BID11 , also use a norm in linear space as a dissimilarity measure, integrating a linear map into a latent space.

Another simple example is RESCAL (Nickel et al., 2011) , which uses the negative inner product as a dissimilarity measure.

In RESCAL, the launcher of relation r is a matrix W ∈ M r = R D×D , the launch maps are given by a linear map, i.e. H (p; (W , w)) = W p and T (p; (W , w)) = p, and the dissimilarity measure is the negative inner product D s DISPLAYFORM2 Other methods are also based on the (negative) inner product dissimilarity: e.g., DISTMULT (Yang et al., 2015) , ComplEx (Trouillon et al., 2016) , HolE (through the Fourier transform) (Nickel et al., 2016) , and ANALOGY BID15 .

Table 1 shows score functions of these methods.

Whereas some methods are based on a neural network (e.g., the neural tensor network (Socher et al., 2013) and ConvE BID3 ), their score function consists of linear operations and element-wise nonlinear functions.

Graph embedding using non-Euclidean space has attracted considerable attention, recently.

Specifically, embedding methods using hyperbolic space have achieved outstanding results (Nickel & Kiela, Table 1 : Score Functions.

The launcher w r of r determines the dissimilarity criterion of r through satellites.

In this table, the dimensionality is set so that the (real) dimensionality of the planets is D. † denotes conjugate transpose.

F denotes the discrete Fourier Transform.

The interpretation here of HolE is given by BID15 and BID10 BID8 ) (Nickel & Kiela, 2018 .

With these methods, each node in the graph is mapped to a point in hyperbolic space and the dissimilarity is measured by a distance function in the space.

Although these methods exploit the advantages of non-Euclidean space, specifically those of a negative curvature space, they focus on single-rather than multi-relational graph embedding.

DISPLAYFORM0 By contrast, TransE has been extended to an embedding method in a Lie group-that is, a manifold with the structure of a group BID5 .

As such, the regularization problem in TransE is avoided by using torus, which can be regarded as a Lie group.

Although this extension to TransE deals with multi-relational embedding, it cannot be applied to all manifolds.

This is because not all manifolds have the structure of a Lie group.

Indeed, we cannot regard a hyperbolic space (if D = 1) or a sphere (if D = 1, 3) as a Lie group.

We can simply design a loss function on the basis of the negative log likelihood of a Bernoulli model as follows: DISPLAYFORM0 DISPLAYFORM1 where Q is the set of the triples with its corrupted head and tail.

That is, DISPLAYFORM2 where δ ∈ R ≥0 is the margin hyperparameter, and [·] + denotes the negative value clipping-i.e.

for all x ∈ R, [x] + := max(x, 0).

We use this loss function throughout this paper.

In this section, we formulate Riemannian TransE exploiting the advantages of TransE in nonEuclidean manifolds.

Firstly, we give a brief introduction of Riemannian geometry.

Secondly, we explain the difficulty in application of TransE in non-Euclidean manifolds.

Lastly, we formulate Riemannian TransE.

Let (M, g) be a Riemannian manifold with metric g.

We denote the tangent and cotangent space of M on p by T p M and T * p M, respectively, and we denote the collection of all smooth vector DISPLAYFORM0 denote the LeviCivita connection, the unique metric-preserving torsion-free affine connection.

A smooth curve γ : (− , ) → M is a geodesic when ∇γγ = 0 on curve γ, whereγ is the differential of curve γ.

Geodesics are generalizations of straight lines, in the sense that they are constant speed curves that are locally distance-minimizing.

We define the exponential map Exp p , which moves point p ∈ M towards a vector by the magnitude of the vector.

In this sense, the exponential map is regarded as an extension of vector addition in a Riemannian manifold.

FIG2 shows an intuitive example of an exponential map on a sphere.

Let DISPLAYFORM1 We define the logarithmic map Log p : M → T p M as the inverse of the exponential map.

Note that the exponential map is not always bijective, and we have to limit the domain of the exponential and logarithmic map appropriately, while some manifolds, such as Euclidean and hyperbolic space, do not suffer from this problem.

In TransE, a single vector w r determines the head and tail launch maps H , T as a transform: DISPLAYFORM0 In fact, these launch maps are given by vector addition.

Note that this constitution of the launcher maps implicitly but essentially uses the fact that a vector is identified with a parallel vector field in Euclidean space.

Specifically, a vector w determines a parallel vector field, denoted by W r here, which gives a tangent vector [W r ] p ∈ T p R D on every point p ∈ R D , and each tangent vector determines the exponential map Exp p ([W r ] p ) at p, which is used as a launch map in TransE. However, because there is no parallel vector field in non-zero curvature spaces, we cannot apply TransE straightforwardly in non-zero curvature spaces.

Thus, extention of TransE in non-Euclidean space non-trivial.

This is the difficulty in Riemannian Manifolds.

As we have explained in Introduction, our idea is replacing parallel vector fields in TransE by vector fields attracted to a point.

Specifically, we obtain the Riemannian TransE as an extension of TransE, replacing the launchers w r ∈ R D in TransE by pairs w r = ( r , p r ) ∈ R × M of a scalar value and point, indicating the length and destination of the satellites' move, respectively.

We call p r the attraction point of relation r. In other words, we replace parallel vector field W r = w r in TransE by r DISPLAYFORM0 Note that, we use a fixed manifold M e = M for entity embedding and use direct product manifold M r = R × M for relation embedding.

However, the extension still has arbitrariness.

For instance, we could launch the tail satellite instead of the head satellite in TransE; in other words, the following launching map also gives us a score function equivalent to that of the original TransE: H (p; w) = p and T (p; w) = p − w ( Figure 5 center).

On the other hand, the score function depends on whether we move the head or tail satellites In these examples, the number |V| of entities is three (1, 2, 3) and the number |R| of relations is two (red and orange), with triples (1, orange, 2) and (1, red, 3).

Hence, these models learn that the orange head satellite of Entity 1 is close to the orange tail satellite of Entity 2 and the red head satellite of Entity 1 is close to the red tail satellite of Entity 3.

In addition, the distance of the other pair of satellites should be long in the representation learned by each method.

The figure on the left shows the original formulation of TransE, where the satellites are given by vector addition.

In other words, the satellites are given by a move towards a point at infinity from the planet.

The center figure shows an alternative formulation of TransE, which is equivalent to the original TransE. Here, the tail satellites are launched and the head satellites are fixed in the red relation.

In Riemannian TransE in the figure on the right, the vector additions are replaced by a move towards a (finite) point.

Figure 6: Relation of the sign for .

If is positive (e.g. the orange relation), the relation runs from low (e.g. Entity 2 and 3) to high hierarchy (e.g. Entity 1), and vice versa (e.g. the red relation).in our case, where the attraction points are not at infinity.

With hierarchical data, an entity at a higher hierarchy has many related entities in a lower hierarchy.

Therefore, it is best to always launch the satellites of "children," the entities in a lower hierarchy, toward their parent.

Hence, we move the head satellites when r > 0 and fix the tail satellites, and vice versa when r < 0; specifically, we move the head satellites by length λ = [ r ] + and move the tail satellites by length λ = [− r ] + .

Thus, bottom-to-top relation cases correspond to r > 0 (Figure 6, left) , and top-to-bottom relation cases correspond to r < 0 (Figure 6 , right).

Another problem pertains to launching the satellites near the attraction point.

If λ > ∆ (p r , p v ), the naive rule causes overrun.

In this case, we simply clip the move and set the satellite in the place of p r .We turn now to the score function of Riemannian TransE. The score function f : (M × M) × (R, M) → R in Riemannian TransE is given as follows: DISPLAYFORM1 where transform m λ,p: M → M denotes a move, defined as follows: DISPLAYFORM2 Here, note that m ,p (q) is on the geodesic that passes through p and q. Figure 5 (right) shows the Riemannian TransE model.

If M = R D and the attraction points are at infinity, the score function is equivalent to that of TransE (without the sphere constraint).

Although the exponential map and logarithmic map in closed form are required to implement Riemannian TransE, we can obtain them when the manifold M is a sphere S D (positive curvature), Euclidean space R D (zero curvature), and hyperbolic space H D (negative curvature), or a direct product of them.

These are practically sufficient.

Also note that the computation costs of these maps are O(D), which is small enough.

In typical cases, the number of entities is very large.

Therefore, stochastic gradient methods are effective for optimization.

Although we can directly apply stochastic gradient methods of Euclidean space or the natural gradient method (Amari, 1998), Riemannian gradient methods (e.g. (Zhang & Sra, 2016) (Zhang et al., 2016) ) work better for non-Euclidean embedding BID6 .

In this paper, we use stocastic Riemannian sub gradient methods Zhang & Sra (2016) with norm clipping (See Appendix).

Note that in spheres or hyperbolic spaces, the computation costs of the gradient is O(D), which is as small as TransE.

Evaluation Tasks We evaluated the performance of our method for a triple classification task (Socher et al., 2013) on real knowledge base datasets.

The triple classification task involved predict-ing whether a triple in the test data is correct.

We label a triple positive when f (p h , p t ; ( r , p r )) > θ r , and vice versa.

Here, θ r ∈ R ≥0 denotes the threshold for each relation r, which is determined by the accuracy of the validation set.

We evaluated the accuracy of classification with the FB13 and WN11 datasets (Socher et al., 2013) .

Although we do not report the results of link prediction tasks BID2 here because there are many evaluation criteria for the task, which makes it difficult to interpret the results, we report the results in Appendix.

BID15 .

We used implementations of these methods on the basis of OpenKE http://openke.thunlp.org/static/index.html, and we used the evaluation scripts there.

Note that we compensated for some missing constraints (for example, in TransR and TransD) and regularizers (for example, in DISTMULT and Analogy) in OpenKE.

We also found that omitting the constraint of the entity planets onto the sphere in TransE gave much better results in our setting, so we also provide these unconstrained results (UnconstraintTransE).

We determined the hyperparameters by following each paper.

For details, see the Appendix.

Results TAB1 shows the results for the triple classification task in each dimensionality.

In WN11, the sphere-based Riemannian TransEs achieved good accuracy.

The accuracy did not degrade dramatically even with low dimensionality.

On the other hand, in FB13, the hyperbolic-space-based Riemannian TransEs was more accurate than other methods.

Moreover for each dimensionality, these results with the proposed Riemannian TransE were at least comparable to those of the baselines.

The accuracy of Euclidean-space-based methods (e.g. the original TransE, and Euclidean TransE) are between that of the sphere-based Riemannian TransEs and that of the hyperbolic-spacebased Riemannian TransEs in most cases.

Note that these results are compatible with the curvature of each space (i.e. Sphere: positive, Euclidean space: 0, a hyperbolic space: negative).

Note that Euclidean methods are sometimes better than non-Euclidean methods.

In Appendix, we also report the triple classification task results in FB15k, where Euclidean TransE as well as baseline methods outperformed Riemannian TransE did not always outperform the baseline methods.

In summary, positive curvature spaces were good in WN11 and negative curvature spaces were good in FB13, and zero curvature spaces were good in FB15k.

These results show that Riemannian TransE can attain good accuracy with small dimensionality provided that an appropriate manifold is selected.

What determines the appropriate manifold?

Spheres are compatible with cyclic structure and hyperbolic spaces are compatible with tree-like structure.

One possible explanation is that WN11 has cyclic structure and FB13 has tree-like structure and the structure of FB15k is between them.

However, further discussion remains future work.

We proposed Riemannian TransE, a novel framework for multi-relational graph embedding, by extending TransE to a Riemannian TransE. Numerical experiments showed that Riemannian TransE outperforms baseline methods in low dimensionality, although its performance depends significantly on the choice of manifold.

Hence, future research shall clarify which manifolds work well with particular kinds of data, and develop a methodology for choosing the appropriate manifold.

This is important work not only for graph completion tasks but also for furthering our understanding of the global characteristics of a graph.

In other words, observing which manifold is effective can help us to understand the global "behavior" of a graph.

Other important work involves using "subspaces" in non-Euclidean space.

Although the notion of a subspace in a non-Euclidean manifold is nontrivial, it may be that our method offers advantages over TransH and TransD, which exploit linear subspaces.

In this paper, we use the following simple (projected) stochastic (Riemannian) (sub-) gradient methods Zhang & Sra (2016) DISPLAYFORM0 where DISPLAYFORM1 |R| denotes the parameter in the τ -th step, η ∈ R ≥0 is the DISPLAYFORM2 |R| is a stochastic gradient that satisfies DISPLAYFORM3 Recall that denotes index raising.

Specifically, we use the following stochastic loss function based on the mini-batch method: DISPLAYFORM4 where the stochastic quintet set Q (τ ) ⊂ Q is a set of uniform-distributed random variables on Q. DISPLAYFORM5 .

We obtain a stochastic gradient as follows: DISPLAYFORM6 where θ is a local coordinate representation of θ.

We obtain∇ (τ ) easily using an automatic differentiation framework.

Algorithm 1 shows the learning algorithm for Riemannian TransE. In the experiments, we applied norm clipping such that the norm of a stochastic gradient is smaller than 1.

end for return θ (τ )

We give additional explanations of the reason why we cannot define a parallel vector field on a non-Euclidean manifold.

Specifically we describe the relationship between parallel vector fields and parallel transform.

We can define a parallel transform along a geodesic.

This parallel transform maps a tangent vector in a tangent space to one in another.

At one glance, it seems that we can define a parallel vector field using the parallel transform.

However, a parallel transform is not determined only by the origin and destination but depends on the path i.e. the geodesic.

Figure 7 shows an example on a sphere, where two ways to map a vector from a tangent space to another are shown and these two give different maps.

As this figure shows we cannot obtain a well-defined vector on more than two points.

Figure 7 : Parallel transforms in a sphere S 2 .

This figure shows two ways to transform vector v ∈ T p S 2 to T r S 2 .

We denote the parallel transform from along segment pq by Π q p : T p S 2 → T q S 2 .

The red vector on T r S 2 denotes the vector obtained by the direct transform along segment pr.

The blue vector T r S 2 denotes the vector obtained by the transform via q. As this figure shows we cannot obtain a well-defined vector on more than two points.

We introduces some Riemannian manifolds useful in applications, and the formula of the exponential map and logarithmic map in these manifolds.

The closed form of exponential map and logarithmic map enables implementation of Riemannian TransE in these manifolds.

In the following, we omit symbols ∂ ∂x and d x of the basis in a tangent and cotangent space, respectively, for notation simplicity.

Moreover, we give the composition of the exponential map and index raising and that of the index lowering and logarithmic map instead of the exponential map and logarithmic map themselves.

This is because we use a cotangent vector rather than a tangent vector in a practical implementation and map from/to cotangent space is more useful (Recall that ∂ ∂θ L is not the coordinate of a tangent but the coordinate of a cotangent vector).

In a D-dimensional Euclidean Space, the exponential map (with the index raising) Exp p • : DISPLAYFORM0 Apparently, the logarithmic map (with the index lowering) DISPLAYFORM1 C.2 SPHERE A D-dimensional (unit) sphere is given by point set S D := p ∈ R (D+1) p p = 1 , and the DISPLAYFORM2 between two points p ∈ S D and q ∈ S D is given as follows: DISPLAYFORM3 where arccos : [−1, 1] → [0, π] denote arc-cosine function.

The exponential map (with the index raising) Exp p • : DISPLAYFORM4 where sinc denotes the cardinal sine function defined as follows: DISPLAYFORM5 The logarithmic map (with the index lowering) DISPLAYFORM6 Note that in optimization, we need the projection of the differentialδ = ∂ ∂θ L (θ)| θ=p of the loss function L to cotangent vector δ given by: DISPLAYFORM7

In this subsection, we introduces models of a hyperbolic space, which are mathematically equivalent to each other, but have practically different aspects.

There are many models of a hyperbolic space.

We introduce two of them: the hyperboloid model and Poincaré disk model.

Some formulae here are also given and used in Nickel & Kiela (2018) .

Let G M denote diagonal matrix DISPLAYFORM0 In the hyperboloid model, a (canonical) hyperbolic space is given by point set DISPLAYFORM1 p, δ = 0 , and the metric g p : DISPLAYFORM2 The distance ∆ (p, q) between two points p ∈ H D and q ∈ H D is given as follows: DISPLAYFORM3 where, arcosh : [1, ∞) → [0, ∞) denotes the area hyperbolic cosine function, i.e. the inverse fucntion of the hyperbolic cosine function.

The exponential map (with the index raising) Exp p • : DISPLAYFORM4 where sinhc denotes the hyperbolic sine cardinal function defined as follows: DISPLAYFORM5 The logarithmic map (with the index lowering) DISPLAYFORM6 D.1.1 LINK PREDICTION TASK In the link prediction task, we predict the head or the tail entity given the relation type and the other entity.

We evaluate the ranking of each correct test triple (h, r, t) in the corrupted triples.

We corrupt each triple as follows.

In our setting, either its head or tail is replaced by one of the possible head or entity, respectively.

In addition, we applied "filtered" setting proposed by BID2 , where the correct triples, that is, the triples T in the original multi-relational graph are excluded.

Thus, the corrupted triples are given by (h , r, t) h ∈ V h ∧ (h , r, t) / ∈ T (head corruption) or {(h, r, t ) | t ∈ V t ∧ (h, r, t ) / ∈ T } (tail corruption).

where V h r and V t r denote the possible heads and tails in relation r, given as follows: DISPLAYFORM7 As evaluation metrics, we use the following:Mean rank (MR) the mean rank of the correct test triples.

The value of this metric is always equal to or greater than 1, and the lower, the better.

Hits @ n (@n) the propotion of correct triples ranked in the top n predictions (n = 1, 3, 10).

The value ranges from 0 to 1, and the higher, the better.

Mean reciprocal rank (MRR) the mean of the reciprocal rank of the correct test triples.

The value ranges from 0 to 1, and the higher, the better.

In triple classification tasks, we predict whether a triple in the test data is correct or not.

The classification is simply based on the score function i.e. we label a triple positive when f (p h , p t ; ( r , p r )) > θ r , and the other way around.

Here, θ r ∈ R ≥0 denotes the threshold for each relation r, which is determined by the accuracy in the validation set.

In link prediction tasks, we used WN18 and FB15k Bordes et al. (2013) datasets, and WN11 and FB13 datasets Socher et al. (2013) .

In triple classification tasks, we used WN11 and FB13 datasets, as well as FB15k.

Note that WN18 and FB15k are originally used for link prediction tasks, whereas WN11 and FB13 are originally used for triple classification tasks.

Also note that WN18 cannot be used for the triple classification task because WN18 does not have test negative data.

TAB4 shows the number of the entities, relations, and triples in each dataset.

Manifolds in Riemannian TransE To evaluate the dependency of performance of Riemannian TransE, we compared Riemannian TransE using the following five kinds of manifolds: Euclidean space R D (Euclidean TransE), hyperbolic spaces H D (HyperbolicTransE), spheres S D (SphericalTransE), the direct product H 4 × H 4 × · · · × H 4 of hyperbolic spaces (PHyperbolicTransE), and the direct product S 4 × S 4 × · · · × S 4 of spheres (PSphericalTransE).

We compared our method with baselines.

As baselines, we used RESCAL Nickel et al. et al. (2016) .

We used implementations of the baselines in OpenKE http://openke.thunlp.

org/static/index.html, a Python library of knowledge base embedding based on Tensorflow BID0 , and moreover, we implemented some lacked constraints (for example, in TransR, TransD) and regularizers (for example, in DistMult, Analogy) in OpenKE.

We also found that omitting the constraint of the entity planets onto sphere in TransE gives much better results in our setting, and this is why we also show the result without the constraint (UnconstraintTransE).

We also implemented Riemannian TransEs as derivations of the base class of OpenKE.We set the dimensionality of the entity manifold as D = 8, 16, 32, 64, 128.

Although we also have to determine the dimensionality of the projected space in TransR and TransD, we let them be equal to D. Due to limitation of the computational costs, we fixed the batch size in baselines and Riemannian TransEs such that that the training data are split to 100 batches.

We also fixed the number of epochs to 1000.

Note that in the first 100 epochs in Riemannian TransEs, we fixed the launchers.

Also note that we applied norm clipping such that the norm of a stochastic gradient in the tangent space is smaller than 1.

We did not use "bern" setting introduced in Wang et al. (2014) , where the ratio between head and tail corruption is not fixed to one to one; in other words, we replaced head and tail with equal probability.

Other than the dimensionality and batch sizes, we used hyperparameters such as learning rate η and margin paremeter δ of baselines used in each paper.

Note that some methods only reports link prediction tasks, and reports hyperparameters for WN18 and FB15k and do not reports ones for WN11 and FB13.

Some methods do not mention settings of hyperparameters, and in these cases, we used the default parameters in OpenKE.

In these cases, we used hyperparameters of WN18 and FB15k also for WN11 and FB13, respectively.

Note that the parameters of TorusE is supposed to be used with very high dimensionality, and the hyperparameters are designed for high dimensionality settings.

In Riemannian TransEs, we simply followed the hyperparameters in TransE.We used the Xavier initializer BID9 as an initializer.

When we have to use the points on a sphere (in the original TransE and Spherical TransEs), we projected the points generated by the initialization onto the sphere.

We found that choice of an initializer has significant effect on embedding performance, and the Xavier initializer achieves very good performance.

We selected optimizers in baselines following each paper.

Note that while using ADADELTA (Zeiler, 2012) is also proposed in TransD, we used SGD in TransD. In Riemannian TransEs, we used we simply followed the hyperparameters in TransE. TAB5 shows the hyperparameters and optimization method for each method.

BID2 , it still contain many reversible triples, as noted by Toutanova & Chen (2015) .

By contrast, these are removed in WN11 and FB13.

Recall that projection-based methods such as TransH, TransR and TransD, and inner-product-based methods such as ComplEx and DISTMULT can exploit a linear subspace.

When a dataset has apparent clusters inside which one relation is easily recovered from the others, we can allocate each cluster to a subspace and separate subspaces from one another.

This separation is easily realized by setting some elements in the launchers to zero in these methods.

Indeed, the TransE without the sphere constraint attains good accuracies in WN11 and FB13.Differences between criteria are also interesting phenomena.

Note that MRR and hit@10 is generous for heavy mistakes.

It is possible that inner-product-based methods earn good scores in trivial relations, but further intensive investigation is needed.

<|TLDR|>

@highlight

Multi-relational graph embedding with Riemannian manifolds and TransE-like loss function. 