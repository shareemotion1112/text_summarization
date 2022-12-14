There is growing interest in geometrically-inspired embeddings for learning hierarchies, partial orders, and lattice structures, with natural applications to transitive relational data such as entailment graphs.

Recent work has extended these ideas beyond deterministic hierarchies to probabilistically calibrated models, which enable learning from uncertain supervision and inferring soft-inclusions among concepts, while maintaining the geometric inductive bias of hierarchical embedding models.

We build on the Box Lattice model of Vilnis et al. (2018), which showed promising results in modeling soft-inclusions through an overlapping hierarchy of sets, parameterized as high-dimensional hyperrectangles (boxes).

However, the hard edges of the boxes present difficulties for standard gradient based optimization; that work employed a special surrogate function for the disjoint case, but we find this method to be fragile.

In this work, we present a novel hierarchical embedding model, inspired by a relaxation of box embeddings into parameterized density functions using Gaussian convolutions over the boxes.

Our approach provides an alternative surrogate to the original lattice measure that improves the robustness of optimization in the disjoint case, while also preserving the desirable properties with respect to the original lattice.

We demonstrate increased or matching performance on WordNet hypernymy prediction, Flickr caption entailment, and a MovieLens-based market basket dataset.

We show especially marked improvements in the case of sparse data, where many conditional probabilities should be low, and thus boxes should be nearly disjoint.

Embedding methods have long been a key technique in machine learning, providing a natural way to convert semantic problems into geometric problems.

Early examples include the vector space BID17 and latent semantic indexing BID4 ) models for information retrieval.

Embeddings experienced a renaissance after the publication of Word2Vec BID12 , a neural word embedding method BID2 BID13 ) that could run at massive scale.

Recent years have seen an interest in structured or geometric representations.

Instead of representing e.g. images, words, sentences, or knowledge base concepts with points, these methods instead associate them with more complex geometric structures.

These objects can be density functions, as in Gaussian embeddings BID21 BID0 , convex cones, as in order embeddings BID20 BID9 , or axis-aligned hyperrectangles, as in box embeddings BID22 BID18 .

These geometric objects more naturally express ideas of asymmetry, entailment, ordering, and transitive relations than simple points in a vector space, and provide a strong inductive bias for these tasks.

In this work, we focus on the probabilistic Box Lattice model of BID22 , because of its strong empirical performance in modeling transitive relations, probabilistic interpretation (edges in a relational DAG are replaced with conditional probabilities), and ability to model complex joint probability distributions including negative correlations.

Box embeddings (BE) are a generalization of order embeddings (OE) BID20 and probabilistic order embeddings (POE) BID9 that replace the vector lattice ordering (notions of overlapping and enclosing convex cones) in OE and POE with a more general notion of overlapping boxes (products of intervals).While intuitively appealing, the "hard edges" of boxes and their ability to become easily disjoint, present difficulties for gradient-based optimization: when two boxes are disjoint in the model, but have overlap in the ground truth, no gradient can flow to the model to correct the problem.

This is of special concern for (pseudo-)sparse data, where many boxes should have nearly zero overlap, while others should have very high overlap.

This is especially pronounced in the case of e.g. market basket models for recommendation, where most items should not be recommended, and entailment tasks, most of which are currently artificially resampled into a 1:1 ratio of positive to negative examples.

To address the disjoint case, BID22 introduce an ad-hoc surrogate function.

In contrast, we look at this problem as inspiration for a new model, based on the intuition of relaxing the hard edges of the boxes into smoothed density functions, using a Gaussian convolution with the original boxes.

We demonstrate the superiority of our approach to modeling transitive relations on WordNet, Flickr caption entailment, and a MovieLens-based market basket dataset.

We match or beat existing state of the art results, while showing substantial improvements in the pseudosparse regime.

As mentioned in the introduction, there is much related work on structured or geometric embeddings.

Most relevant to this work are the order embeddings of BID20 , which embed a nonprobabilistic DAG or lattice in a vector space with order given by inclusion of embeddings' forward cones, the probabilistic extension of that model due to BID9 , and the box lattice or box embedding model of BID22 , which we extend.

Concurrently to BID22 , another hyperrectangle-based generalization of order embeddings was proposed by BID18 , also called box embeddings.

The difference between the two models lies in the interpretation: the former is a probabilistic model that assigns edges conditional probabilities according to degrees of overlap, while the latter is a deterministic model in the style of order embeddings -an edge is considered present only if one box entirely encloses another.

Methods based on embedding points in hyperbolic space BID16 BID5 have also recently been proposed for learning hierarchical embeddings.

These models, similar to order embeddings and the box embeddings of BID18 , are nonprobabilistic and optimize an energy function.

Additionally, while the negative curvature of hyperbolic space is attractively biased towards learning tree structures (since distances between points increase the farther they are from the origin), this constant curvature makes the models not as suitable for learning non-treelike DAGs.

Our approach to smoothing the energy landscape of the model using Gaussian convolution is common in mollified optimization and continuation methods, and is increasingly making its way into machine learning models such as Mollifying Networks BID7 , diffusion-trained networks BID14 , and noisy activation functions BID6 .Our focus on embedding orderings and transitive relations is a subset of knowledge graph embedding.

While this field is very large, the main difference of our probabilistic approach is that we seek to learn an embedding model which maps concepts to subsets of event space, giving our model an inductive bias especially suited for transitive relations as well as fuzzy concepts of inclusion and entailment.

We begin with a brief overview of two methods for representing ontologies as geometric objects.

First, we review some definitions from order theory, a useful formalism for describing ontologies, then we introduce the vector and box lattices.

FIG0 shows a simple two-dimensional example of these representations.

A non-strict partially ordered set (poset) is a pair P, , where P is a set, and is a binary relation.

For all a, b, c ??? P , Reflexivity: a a Antisymmetry: a b a implies a = b Transitivity: a b c implies a cThis generalizes the standard concept of a totally ordered set to allow some elements to be incomparable.

Posets provide a good formalism for the kind of acyclic directed graph data found in many knowledge bases with transitive relations.

A lattice is a poset where any subset of elements has a single unique least upper bound, and greatest lower bound.

In a bounded lattice, the set P contains two additional elements, (top), and ??? (bottom), which denote the least upper bound and greatest lower bound of the entire set.

A lattice is equipped with two binary operations, ??? (join), and ??? (meet).

a???b denotes the least upper bound of a, b ??? P , and a ??? b denotes their greatest lower bound.

A bounded lattice must satisfy these properties: DISPLAYFORM0 Note that the extended real numbers, R ??? {??????, ???}, form a bounded lattice (and in fact, a totally ordered set) under the min and max operations as the meet (???) and join (???) operations.

So do sets partially ordered by inclusion, with ??? and ??? as ??? and ???. Thinking of these special cases gives the intuition for the fourth property, absorption.

The ??? and ??? operations can be swapped, along with reversing the poset relation , to give a valid lattice, called the dual lattice.

In the real numbers this just corresponds to a sign change.

A semilattice has only a meet or join, but not both.

Note.

In the rest of the paper, when the context is clear, we will also use ??? and ??? to denote min and max of real numbers, in order to clarify the intuition behind our model.

A vector lattice, also known as a Riesz space (Zaanen, 1997), or Hilbert lattice when the accompanying vector space has an inner product, is a vector space endowed with a lattice structure.

A standard choice of partial order for the vector lattice R n is to use the product order from the underlying real numbers, which specifies for all x, y ??? R n x y ?????? ???i ??? {1..n}, x i ??? y i Under this order, meet and join operations are pointwise min and max, which gives a lattice structure.

In this formalism, the Order Embeddings of BID20 embed partial orders as vectors using the reverse product order, corresponding to the dual lattice, and restrict the vectors to be positive.

The vector of all zeroes represents , and embedded objects become "more specific" as they get farther away from the origin.

FIG0 demonstrates a toy, two-dimensional example of the Order Embedding vector lattice representation of a simple ontology.

Shading represents the probability measure assigned to this lattice in the probabilistic extension of BID9 .

Vilnis et al. FORMULA2 introduced a box lattice, wherein each concept in a knowledge graph is associated with two vectors, the minimum and maximum coordinates of an axis-aligned hyperrectangle, or box (product of intervals).Using the notion of set inclusion between boxes, there is a natural partial order and lattice structure.

To represent a box x, let the pairs (x m,i , x M,i ) be the maximum and minimum of the interval at each coordinate i.

Then the box lattice structure (least upper bounds and greatest lower bounds), with ??? and ??? denoting max and min when applied to the scalar coordinates, is DISPLAYFORM0 Here, denotes a set (cartesian) product -the lattice meet is the largest box contained entirely within both x and y, or bottom (the empty set) where no intersection exists, and the lattice join is the smallest box containing both x and y.

To associate a measure, marginal probabilities of (collections of) events are given by the volume of boxes, their complements, and intersections under a suitable probability measure.

Under the uniform measure, if event x has an associated box with interval boundaries (x m , x M ), the probability p(x) is given by n i (x M,i ??? x m,i ).

Use of the uniform measure requires the boxes to be constrained to the unit hypercube, so that p(x) ??? 1.

p(???) is taken to be zero, since ??? is an empty set.

As boxes are simply special cases of sets, it is intuitive that this is a valid probability measure, but it can also be shown to be compatible with the meet semilattice structure in a precise sense BID10 .

When using gradient-based optimization to learn box embeddings, an immediate problem identified in the original work is that when two concepts are incorrectly given as disjoint by the model, no gradient signal can flow since the meet (intersection) is exactly zero, with zero derivative.

To see this, note that for a pair of 1-dimensional boxes (intervals), the volume of the meet under the uniform measure p as given in Section 3.3 is DISPLAYFORM1 where m h is the standard hinge function, m h (x) = 0 ??? x = max(0, x).The hinge function has a large flat plateau at 0 when intervals are disjoint.

This issue is especially problematic when the lattice to be embedded is (pseudo-)sparse, that is, most boxes should have very little or no intersection, since if training accidentally makes two boxes disjoint there is no way to recover with the naive measure.

The authors propose a surrogate function to optimize in this case, but we will use a more principled framework to develop alternate measures that avoid this pathology, improving both optimization and final model quality.

DISPLAYFORM2 Figure 2: One-dimensional example demonstrating two disjoint indicators of intervals before and after the application of a smoothing kernel.

The area under the purple product curve is proportional to the degree of overlap.

The intuition behind our approach is that the "hard edges" of the standard box embeddings lead to unwanted gradient sparsity, and we seek a relaxation of this assumption that maintains the desirable properties of the base lattice model while enabling better optimization and preserving a geometric intuition.

For ease of exposition, we will refer to 1-dimensional intervals in this section, but the results carry through from the representation of boxes as products of intervals and their volumes under the associated product measures.

The first observation is that, considering boxes as indicator functions of intervals, we can rewrite the measure of the joint probability p(x ??? y) between intervals x = [a, b] and y = [c, d] as an integral of the product of those indicators: DISPLAYFORM3 since the product has support (and is equal to 1) only in the areas where the two intervals overlap.

A solution suggests itself in replacing these indicator functions with functions of infinite support.

We elect for kernel smoothing, specifically convolution with a normalized Gaussian kernel, equivalent to an application of the diffusion equation to the original functional form of the embeddings (indicator functions) and a common approach to mollified optimization and energy smoothing BID15 BID7 BID14 .

This approach is demonstrated in one dimension in Figure 2 .Specifically, given x = [a, b], we associate the smoothed indicator function DISPLAYFORM4 We then wish to evaluate, for two lattice elements x and y with associated smoothed indicators f and g, DISPLAYFORM5 This integral admits a closed form solution.

Proposition 1.

Let m ?? (x) = ??(x)dx be an antiderivative of the standard normal CDF.

Then the solution to equation 2 is given by, DISPLAYFORM6 where ?? = ?? 2 1 + ?? 2 2 , soft(x) = log(1 + exp(x)) is the softplus function, the antiderivative of the logistic sigmoid, and ?? = ?? 1.702 .Proof.

The first line is proved in Appendix A, the second approximation follows from the approximation of ?? by a logistic sigmoid given in BID3 .Note that, in the zero-temperature limit, as ?? goes to zero, we recover the formula DISPLAYFORM7 with equality in the last line because (a, b) and (c, d) are intervals.

This last line is exactly our original equation equation 1, which is expected from convolution with a zero-bandwidth kernel (a Dirac delta function, the identity element under convolution).

This is true for both the exact formula using ??(x)dx, and the softplus approximation.

Unfortunately, for any ?? > 0, multiplication of Gaussian-smoothed indicators does not give a valid meet operation on a function lattice, for the simple reason that f 2 = f , except in the case of indicator functions, violating the idempotency requirement of Section 3.1.More importantly, for practical considerations, if we are to treat the outputs of p ?? as probabilities, the consequence is DISPLAYFORM8 which complicates our applications that train on conditional probabilities.

However, by a modification of equation 3, we can obtain a function p such that p(x ??? x) = p(x), while retaining the smooth optimization properties of the Gaussian model.

Recall that for the hinge function m h and two intervals (a, b) and (c, d), we have DISPLAYFORM9 where the left hand side is the zero-temperature limit of the Gaussian model from equation 3.

This identity is true of the hinge function m h , but not the softplus function.

However, an equation with a similar functional form as equation 6 (on both the left-and right-hand sides) is true not only of the hinge function from the unsmoothed model, but also true of the softplus.

For two intervals x = (a, b) an y = (c, d), by the commutativity of min and max with monotonic functions, we have DISPLAYFORM10 In the zero-temperature limit, all terms in equations 3 and 7 are equivalent.

However, outside of this, equation 7 is idempotent for x = y = (a, b) = (c, d) (when considered as a measure of overlap, made precise in the next paragraph), while equation 3 is not.

This inspires us to define the probabilities p(x) and p(x, y) using a normalized version of equation 7 in place of equation 3.

For the interval (one-dimensional box) case, we define DISPLAYFORM11 which satisfies the idempotency requirement, p(x) = p(x, x).Because softplus upper-bounds the hinge function, it is capable of outputting values that are greater than 1, and therefore must be normalized.

In our experiments, we use two different approaches to normalization.

For experiments with a relatively small number of entities (all besides Flickr), we allow the boxes to learn unconstrained, and divide each dimension by the measured size of the global minimum and maximum (G DISPLAYFORM12 For data where computing these values repeatedly is infeasible, we project onto the unit hypercube and normalize by m soft (1).

The final probability p(x) is given by the product over dimensions DISPLAYFORM13 Note that, while equivalent in the zero temperature limit to the standard uniform probability measure of the box model, this function, like the Gaussian model, is not a valid probability measure on the entire joint space of events (the lattice).

However, neither is factorization of a conditional probability table using a logistic sigmoid link function, which is commonly used for the similar tasks.

Our approach retains the inductive bias of the original box model, is equivalent in the limit, and satisfies the necessary condition that p(x, x) = p(x).

A comparison of the 3 different functions is given in FIG2 , with the softplus overlap showing much better behavior for highly disjoint boxes than the Gaussian model, while also preserving the meet property.

Note that in order to achieve high overlap, the Gaussian model must drastically lower its temperature, causing vanishing gradients in the tails.

We perform experiments on the WordNet hypernym prediction task in order to evaluate the performance of these improvements in practice.

The WordNet hypernym hierarchy contains 837,888-edges after performing the transitive closure on the direct edges in WordNet.

We used the same train/dev/test split as in BID20 .

Positive examples are randomly chosen from the 837k edges, while negative examples are generated by swapping one of the terms to a random word in the dictionary.

Experimental details are given in Appendix D.1.

The smoothed box model performs nearly as well as the original box lattice in terms of test accuracy 1 .

While our model requires less hyper-parameter tuning than the original, we suspect that our performance would be increased on a task with a higher degree of sparsity than the 50/50 positive/negative split of the standard WordNet data, which we explore in the next section.

In order to confirm our intuition that the smoothed box model performs better in the sparse regime, we perform further experiments using different numbers of positive and negative examples from the WordNet mammal subset, comparing the box lattice, our smoothed approach, and order embeddings (OE) as a baseline.

The training data is the transitive reduction of this subset of the mammal WordNet, while the dev/test is the transitive closure of the training data.

The training data contains 1,176 positive examples, and the dev and test sets contain 209 positive examples.

Negative examples are generated randomly using the ratio stated in the table.

As we can see from the table, with balanced data, all models include OE baseline, Box, Smoothed Box models nearly match the full transitive closure.

As the number of negative examples increases, the performance drops for the original box model, but Smoothed Box still outperforms OE and Box in all setting.

This superior performance on imbalanced data is important for e.g. real-world entailment graph learning, where the number of negatives greatly outweigh the positives.

Table 5 : F1 scores of the box lattice, order embeddings, and our smoothed model, for different levels of label imbalance on the WordNet mammal subset.

We conduct experiments on the Flickr entailment dataset.

Flickr is a large-scale caption entailment dataset containing of 45 million image caption pairs.

In order to perform an apples-to-apples comparison with existing results we use the exact same dataset from BID22 .

In this case, we do constrain the boxes to the unit cube, using the same experimental setup as BID22 , except we apply the softplus function before calculating the volume of the boxes.

Experimental details are given in Appendix D.3.We report KL divergence and Pearson correlation on the full test data, unseen pairs (caption pairs which are never occur in training data) and unseen captions (captions which are never occur in training data).

As shown in TAB2 , we see a slight performance gain compared to the original model, with improvements most concentrated on unseen captions.

We apply our method to a market-basket task constructed using the MovieLens dataset.

Here, the task is to predict users' preference for movie A given that they liked movie B. We first collect all pairs of user-movie ratings higher than 4 points (strong preference) from the MovieLens-20M dataset.

From this we further prune to just a subset of movies which have more than 100 user ratings to make sure that counting statistics are significant enough.

This leads to 8545 movies in our dataset.

We calculate the conditional probability P (A|B) = P (A,B) We compare with several baselines: low-rank matrix factorization, complex bilinear factorization BID19 , and two hierarchical embedding methods, POE BID9 and the Box Lattice BID22 .

Since the training matrix is asymmetric, we used separate embeddings for target and conditioned movies.

For the complex bilinear model, we added one additional vector of parameters to capture the "imply" relation.

We evaluate on the test set using KL divergence, Pearson correlation, and Spearman correlation with the ground truth probabilities.

Experimental details are given in Appendix D.4.

DISPLAYFORM0 From the results in TAB4 , we can see that our smoothed box embedding method outperforms the original box lattice as well as all other baselines' performances, especially in Spearman correlation, the most relevant metric for recommendation, a ranking task.

We perform an additional study on the robustness of the smoothed model to initialization conditions in Appendix C.

We presented an approach to smoothing the energy and optimization landscape of probabilistic box embeddings and provided a theoretical justification for the smoothing.

Due to a decreased number of hyper-parameters this model is easier to train, and, furthermore, met or surpassed current state-ofthe-art results on several interesting datasets.

We further demonstrated that this model is particularly effective in the case of sparse data and more robust to poor initialization.

Tackling the learning problems presented by rich, geometrically-inspired embedding models is an open and challenging area of research, which this work is far from the last word on.

This task will become even more pressing as the embedding structures become more complex, such as unions of boxes or other non-convex objects.

To this end, we will continue to explore both function lattices, and constraint-based approaches to learning.

A PROOF OF GAUSSIAN OVERLAP FORMULA We wish to evaluate, for two lattice elements x and y, with associated smoothed indicators f and g, DISPLAYFORM0 Since the Gaussian kernel is normalized to have total integral equal to 1, so as not to change the overall areas of the boxes, the concrete formula is DISPLAYFORM1 Since the antiderivative of ?? is the normal CDF, this may be recognized as the difference ??(x; a, ?? 2 ) ??? ??(x; b, ?? 2 ), but this does not allow us to easily evaluate the integral of interest, which is the integral of the product of two such functions.

To evaluate equation 8, recall the identity BID8 BID21 DISPLAYFORM2 For convenience, let ?? : DISPLAYFORM3 .

Applying Fubini's theorem and using equation 9, we have DISPLAYFORM4 and therefore, with ?? = ?? ???1 , DISPLAYFORM5

The MovieLens dataset, while not truly sparse, has a large proportion of small probabilities which make it especially suitable for optimization by the smoothed model.

The rough distribution of probabilities, in buckets of width 0.1, is shown in FIG0 .

We perform an additional set of experiments to determine the robustness of the smoothed box model to initialization.

While the model is normally initialized randomly so that each box is a product of intervals that almost always overlaps with the other boxes, we would like to determine the models robustness to disjoint boxes in a principled way.

While we can control initialization, we cannot always control the intermediate results of optimization, which may drive boxes to be disjoint, a condition from which the original, hard-edged box model may have difficulty recovering.

So, parametrizing the initial distribution of boxes with a minimum coordinate and a positive width, we adjust the width parameter so that approximately 0%, 20%, 50%, and 100% of boxes are disjoint at initialization before learning on the MovieLens dataset as usual.

These results are presented in table 8.

The smoothed model does not seem to suffer at all from disjoint initialization, while the performance of the original box model degrades significantly.

From this we can speculate that part of the strength of the smoothed box model is its ability to smoothly optimize in the disjoint regime.

We give a brief overview of our methodology and hyperparameter selection methods for each experiment.

Detailed hyperparameter settings and code to reproduce experiments can be found at https://github.com/Lorraine333/smoothed_box_embedding.

For the WordNet experiments, the model is evaluated every epoch on the development set for a large fixed number of epochs, and the best development model is used to score the test set.

Baseline models are trained using the parameters of BID22 , with the smoothed model using hyperparameters determined on the development set.

We follow the same routine as the WordNet experiments section to select best parameters.

For the 12 experiments we conducted in this section, negative examples are generated randomly based on the ratio for each batch of positive examples.

We do a parameter sweep for all models then choose the best result for each model as our final result.

The experimental setup uses the same architecture as BID22 and BID9 , a single-layer LSTM that reads captions and produces a box embedding parameterized by min and delta.

Embeddings are produced by feedforward networks on the output of the LSTM.

The model is trained for a large fixed number of epochs, and tested on the development data at each epoch.

The best development model is used to report test set score.

Hyperparameters were determined on the development set.

For all MovieLens experiments, the model is evaluated every 50 steps on the development set, and optimization is stopped if the best development set score fails to improve after 200 steps.

The best development model is used to score the test set.

<|TLDR|>

@highlight

Improve hierarchical embedding models using kernel smoothing