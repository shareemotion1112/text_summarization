Adversarial examples are a pervasive phenomenon of machine learning models where seemingly imperceptible perturbations to the input lead to misclassifications for otherwise statistically accurate models.

We propose a geometric framework, drawing on tools from the manifold reconstruction literature, to analyze the high-dimensional geometry of adversarial examples.

In particular, we highlight the importance of codimension: for low-dimensional data manifolds embedded in high-dimensional space there are many directions off the manifold in which to construct adversarial examples.

Adversarial examples are a natural consequence of learning a decision boundary that classifies the low-dimensional data manifold well, but classifies points near the manifold incorrectly.

Using our geometric framework we prove (1) a tradeoff between robustness under different norms, (2) that adversarial training in balls around the data is sample inefficient, and (3) sufficient sampling conditions under which nearest neighbor classifiers and ball-based adversarial training are robust.

Deep learning at scale has led to breakthroughs on important problems in computer vision (Krizhevsky et al. (2012) ), natural language processing (Wu et al. (2016) ), and robotics (Levine et al. (2015) ).

Shortly thereafter, the interesting phenomena of adversarial examples was observed.

A seemingly ubiquitous property of machine learning models where perturbations of the input that are imperceptible to humans reliably lead to confident incorrect classifications (Szegedy et al. (2013) ; BID21 ).

What has ensued is a standard story from the security literature: a game of cat and mouse where defenses are proposed only to be quickly defeated by stronger attacks BID3 ).

This has led researchers to develop methods which are provably robust under specific attack models (Madry et al. (2018) ; Wong & Kolter (2018) ; Sinha et al. (2018) ; Raghunathan et al. (2018) ).

As machine learning proliferates into society, including security-critical settings like health care BID18 ) or autonomous vehicles BID10 ), it is crucial to develop methods that allow us to understand the vulnerability of our models and design appropriate counter-measures.

In this paper, we propose a geometric framework for analyzing the phenomenon of adversarial examples.

We leverage the observation that datasets encountered in practice exhibit low-dimensional structure despite being embedded in very high-dimensional input spaces.

This property is colloquially referred to as the "Manifold Hypothesis": the idea that low-dimensional structure of 'real' data leads to tractable learning.

We model data as being sampled from class-specific low-dimensional manifolds embedded in a high-dimensional space.

We consider a threat model where an adversary may choose any point on the data manifold to perturb by in order to fool a classifier.

In order to be robust to such an adversary, a classifier must be correct everywhere in an -tube around the data manifold.

Observe that, even though the data manifold is a low-dimensional object, this tube has the same dimension as the entire space the manifold is embedded in.

Our analysis argues that adversarial examples are a natural consequence of learning a decision boundary that classifies all points on a low-dimensional data manifold correctly, but classifies many points near the manifold incorrectly.

The high codimension, the difference between the dimension of the data manifold and the dimension of the embedding space, is a key source of the pervasiveness of adversarial examples.

Our paper makes the following contributions.

First, we develop a geometric framework, inspired by the manifold reconstruction literature, that formalizes the manifold hypothesis described above and our attack model.

Second, we highlight the role codimension plays in vulnerability to adversarial DISPLAYFORM0 rch 2 ⇤ 2 rch 2 ⇤ 2 rch 2 ⇤ 2 rch 2 ⇤ 2 rch 2 ⇤ 2 rch 2 ⇤ 2 Figure 1 : Examples of the decision axis Λ 2 , shown here in green, for different data manifolds.

Intuitively, the decision axis captures an optimal decision boundary between the data manifolds.

It's optimal in the sense that each point on the decision axis is as far away from each data manifold as possible.

Notice that in the first example, the decision axis coincides with the maximum margin line.examples.

As the codimension increases, there are an increasing number of directions off the data manifold in which to construct adversarial perturbations.

Prior work has attributed vulnerability to adversarial examples to input dimension BID20 ).

This is the first work that investigates the role of codimension in adversarial examples.

Interestingly, we find that different classification algorithms are less sensitive to changes in codimension.

Third, we apply this framework to prove the following results: (1) we show that the choice of norm to restrict an adversary is important in that there exists a tradeoff between being robust to different norms: we present a classification problem where improving robustness under the · ∞ norm requires a loss of Ω(1−1/ √ d) in robustness to the · 2 norm; (2) we show that a common approach, training against adversarial examples drawn from balls around the training set, is insufficient to learn robust decision boundaries with realistic amounts of data; and (3) we show that nearest neighbor classifiers do not suffer from this insufficiency, due to geometric properties of their decision boundary away from data, and thus represent a potentially robust classification algorithm.

Finally we provide experimental evidence on synthetic datasets and MNIST that support our theoretical results.

This paper approaches the problem of adversarial examples using techniques and intuition from the manifold reconstruction literature.

Both fields have a great deal of prior work, so we focus on only the most related papers here.

Some previous work has considered the relationships between adversarial examples and high dimensional geometry.

BID19 explore the robustness of classifiers to random noise in terms of distance to the decision boundary, under the assumption that the decision boundary is locally flat.

The work of BID20 experimentally evaluated the setting of two concentric under-sampled 499-spheres embedded in R 500 , and concluded that adversarial examples occur on the data manifold.

In contrast, we present a geometric framework for proving robustness guarantees for learning algorithms, that makes no assumptions on the decision boundary.

We carefully sample the data manifold in order to highlight the importance of codimension; adversarial examples exist even when the manifold is perfectly classified.

Additionally we explore the importance of the spacing between the constituent data manifolds and sampling requirements for learning algorithms.

Wang et al. (2018) explore the robustness of k-nearest neighbor classifiers to adversarial examples.

In the setting where the Bayes optimal classifier is uncertain about the true label of each point, they show that k-nearest neighbors is not robust if k is a small constant.

They also show that if k ∈ Ω( √ dn log n), then k-nearest neighbors is robust.

Using our geometric framework we show a complementary result: in the setting where each point is certain of its label, 1-nearest neighbors is robust to adversarial examples.

The decision and medial axes defined in Section 3 are maximum margin decision boundaries.

Hard margin SVMs define define a linear separator with maximum margin, maximum distance from the training data BID11 ).

Kernel methods allow for maximum margin decision boundaries that are non-linear by using additional features to project the data into a higher-dimensional feature space (Shawe-Taylor & Cristianini (2004) ).

The decision and medial axes generalize the notion of maximum margin to account for the arbitrary curvature of the data manifolds.

There have been attempts to incorporate maximum margins into deep learning (Sun et al. (2016); Liu et al. (2016); Liang et al. (2017); BID17 ), often by designing loss functions that encourage large margins at either the output (Sun et al. (2016) ) or at any layer BID17 ).

In contrast, the decision axis is defined on the input space and we use it as an analysis tool for proving robustness guarantees.

Manifold reconstruction is the problem of discovering the structure of a k-dimensional manifold embedded in R d , given only a set of points sampled from the manifold.

A large vein of research in manifold reconstruction develops algorithms that are provably good: if the points sampled from the underlying manifold are sufficiently dense, these algorithms are guaranteed to produce a geometrically accurate representation of the unknown manifold with the correct topology.

The output of these algorithms is often a simplicial complex, a set of simplices such as triangles, tetrahedra, and higher-dimensional variants, that approximate the unknown manifold.

In particular these algorithms output subsets of the Delaunay triangulation, which, along with their dual the Voronoi diagram, have properties that aid in proving geometric and topological guarantees BID16 ).The field first focused on curve reconstruction in R 2 BID1 ) and subsequently in BID14 ).

Soon after algorithms were developed for surface reconstruction in R 3 , both in the noise-free setting BID0 ; BID2 ) and in the presence of noise BID13 ).

We borrow heavily from the analysis tools of these early works, including the medial axis and the reach.

However we emphasize that we have adapted these tools to the learning setting.

To the best of our knowledge, our work is the first to consider the medial axis under different norms.

DISPLAYFORM0 In higher-dimensional embedding spaces (large d), manifold reconstruction algorithms face the curse of dimensionality.

In particular, the Delaunay triangulation, which forms the bedrock of algorithms in low-dimensions, of n vertices in R d can have up to Θ(n d/2 ) simplices.

To circumvent the curse of dimensionality, algorithms were proposed that compute subsets of the Delaunay triangulation restricted to the k-dimensional tangent spaces of the manifold at each sample point BID5 ).

Unfortunately, progress on higher-dimensional manifolds has been limited due to the presence of so-called "sliver" simplices, poorly shaped simplices that cause in-consistences between the local triangulations constructed in each tangent space BID8 ; BID5 ).

Techniques that provably remove sliver simplices have prohibitive sampling requirements BID7 ; BID5 ).

Even in the special case of surfaces (k = 2) embedded in high dimensions (d > 3), algorithms with practical sampling requirements have only recently been proposed (Khoury & Shewchuk (2016) ).

Our use of tubular neighborhoods as a tool for analysis is borrowed from and Khoury & Shewchuk (2016) .In this paper we are interested in learning robust decision boundaries, not reconstructing the underlying data manifolds, and so we avoid the use of Delaunay triangulations and their difficulties entirely.

In Section 6 we present robustness guarantees for two learning algorithms in terms of a sampling condition on the underlying manifold.

These sampling requirements scale with the dimension of the underlying manifold k, not with the dimension of the embedding space d.

We model data as being sampled from a set of low-dimensional manifolds (with or without boundary) embedded in a high-dimensional space R d .

We use k to denote the dimension of a manifold M ⊂ R d .

The special case of a 1-manifold is called a curve, and a 2-manifold is a surface.

The codimension of M is d−k, the difference between the dimension of the manifold and the dimension of the embedding space.

The "Manifold Hypothesis" is the observation that in practice, data is often sampled from manifolds, usually of high codimension.

In this paper we are primarily interested in the classification problem.

Thus we model data as being sampled from C class manifolds M 1 , . . .

, M C , one for each class.

When we wish to refer to the entire space from which a dataset is sampled, we refer to the data manifold M = ∪ 1≤j≤C M j .

We often work with a finite sample of n points, X ⊂ M, and we write X = {X 1 , X 2 , . . .

, X n }.

Each sample point X i has an accompanying class label y i ∈ {1, 2, . . .

, C} indicating which manifold M yi the point X i is sampled from.

Consider a · p -ball B centered at some point c ∈ R d and imagine growing B by increasing its radius starting from zero.

For nearly all starting points c, the ball B eventually intersects one, and only one, of the M i '

s. Thus the nearest point to c on M, in the norm · p , lies on M i . (Note that the nearest point on M i need not be unique.)The decision axis Λ p of M is the set of points c such that the boundary of B intersects two or more of the M i , but the interior of B does not intersect M at all.

In other words, the decision axis Λ p is the set of points that have two or more closest points, in the norm · p , on distinct class manifolds.

See Figure 1 .

The decision axis is inspired by the medial axis, which was first proposed by BID4 in the context of image analysis and subsequently modified for the purposes of curve and surface reconstruction by BID1 BID2 .

We have modified the definition to account for multiple class manifolds and have renamed our variant in order to avoid confusion in the future.

The decision axis Λ p can intuitively be thought of as a decision boundary that is optimal in the following sense.

First, Λ p separates the class manifolds when they do not intersect (Lemma 7).

Second, each point of Λ p is as far away from the class manifolds as possible in the norm · p .

As shown in the leftmost example in Figure 1 , in the case of two linearly separable circles of equal radius, the decision axis Λ 2 is exactly the line that separates the data with maximum margin.

For arbitrary manifolds, Λ p generalizes the notion of maximum margin to account for the arbitrary curvature of the class manifolds.

Let T ⊂ R d be any set.

The reach rch p (T ; M) of M is defined as inf x∈M,y∈T x − y p .

When M is compact, the reach is achieved by the point on M that is closest to T under the · p norm.

We will drop M from the notation when it is understood from context.

DISPLAYFORM0 ,p is the set of all points whose distance to M under the metric induced by · p is less than .

Note that while M is k-dimensional, M ,p is always d-dimensional.

Tubular neighborhoods are how we rigorously define adversarial examples.

Consider a classifier f : DISPLAYFORM1 such that f (x) = i. A classifier f is robust to all -adversarial examples when f correctly classifies not only M, but all of M ,p .

Thus the problem of being robust to adversarial examples is rightly seen as one of generalization.

In this paper we will be primarily concerned with exploring the conditions under which we can provably learn a decision boundary that correctly classifies M ,p .

When < rch p Λ p , the decision axis Λ p is one decision boundary that correctly classifies M ,p (Corollary 9).

Throughout the remainder of the paper we will drop the p in M ,p from the notation, instead writing M ; the norm will always be clear from context.

The geometric quantities defined above can be defined more generally for any distance metric d(·, ·).

In this paper we will focus exclusively on the metrics induced by the norms · p for p > 0.

The decision axis under · 2 is in general not identical to the decision axis under · ∞ .

In Section 4 we will prove that since Λ 2 is not identical to Λ ∞ there exists a tradeoff in the robustness of any decision boundary between the two norms.

Schott et al. (2018) explore the vulnerability of robust classifiers to attacks under different norms.

In particular, they take the robust pretrained classifier of Madry et al. (2018) , which was trained to be robust to · ∞ -perturbations, and subject it to · 0 and · 2 attacks.

They show that accuracy drops to 0% under · 0 attacks and to 35% under · 2 .

Here we explain why poor robustness under the norm · 2 should be expected.

Figure 2: As the dimension increases, the rch 2 (Λ ∞ ; S 1 ∪ S 2 ) decreases, and so an · ∞ robust classifier is less robust to · 2 attacks.

The dashed lines are placed at 1/ √ d, where our theoretical results predict we should start finding · 2 adversarial examples.

We use the robust · ∞ loss of Wong & Kolter (2018) We say a decision boundary D f for a classifier f is -robust in the · p norm if < rch p D f .

In words, starting from any point x ∈ M, a perturbation η x must have p-norm greater than rch p D f to cross the decision boundary.

The most robust decision boundary to · p -perturbations is Λ p .

In Theorem 1 we construct a learning setting where Λ 2 is distinct from Λ ∞ .

Thus, in general, no single decision boundary can be optimally robust in all norms.

Theorem 1.

Let S 1 , S 2 ⊂ R d+1 be two concentric d-spheres with radii r 1 < r 2 respectively.

Let S = S 1 ∪S 2 and let Λ 2 , Λ ∞ be the · 2 and · ∞ decision axes of S.

From Theorem 1 we conclude that the minimum distance from S 1 to Λ ∞ under the · 2 norm is upper bounded as DISPLAYFORM0 If a classifier f is trained to learn Λ ∞ , an adversary, starting on S 1 , can construct an · 2 adversarial example for a perturbation as small as O(1/ √ d).

Thus we should expect f to be less robust to · 2 -perturbations.

Figure 2 verifies this result experimentally.

The proof of Theorem 1 is provided in Appendix A We expect that Λ 2 = Λ ∞ is the common case in practice.

For example, Theorem 1 extends immediately to concentric cylinders and intertwined tori by considering 2-dimensional planar crosssections.

In general, we expect that Λ 2 = Λ ∞ in situations where a 2-dimensional cross-section with M has nontrivial curvature.

Theorem 1 is important because, even in recent literature, researchers have attributed this phenomena to overfitting.

Schott et al. (2018) state that "the widely recognized and by far most successful defense by Madry et al. FORMULA8 overfits on the L ∞ metric (it's highly susceptible to L 2 and L 0 perturbations)" (emphasis ours).

We disagree; the Madry et al. FORMULA8 classifier performed exactly as intended.

It learned a decision boundary that is robust under · ∞ , which we have shown is quite different from the most robust decision boundary under · 2 .Interestingly, the proposed models of Schott et al. FORMULA8 also suffer from this tradeoff.

Their model ABS has accuracy 80% to · 2 attacks but drops to 8% for · ∞ .

Similarly their model ABS Binary has accuracy 77% to · ∞ attacks but drops to 39% for · 2 attacks.

We reiterate, in general, no single decision boundary can be optimally robust in all norms.

Madry et al. FORMULA8 suggest training a robust classifier with the help of an adversary which, at each iteration, produces -perturbations around the training set that are incorrectly classified.

In our notation, this corresponds to learning a decision boundary that correctly classifies X = {x ∈ R d : x−X i 2 ≤ for some training point X i }.

We believe this approach is insufficiently robust in practice, as X is often a poor model for M .

In this section, we show that the volume vol X is often DISPLAYFORM0 To construct an δ-cover we place sample points, shown here in black, along a regular grid with spacing ∆. The blue points are the furthest points of Π from the sample.

To cover Π we need ∆ = 2δ/ √ k. Right: An illustration of the lower bound technique used in Equation 3.

The volume vol Π δ shown in the black dashed lines, is bounded from below by placing a (d − k)-dimensional ball of radius δ at each point of Π, shown in green.

In this illustration, a 1-dimensional manifold is embedded in 2 dimensions, so these balls are 1-dimensional line segments.a vanishingly small percentage of vol M .

These results shed light on why the ball-based learning algorithm L defined in Section 6 is so much less sample-efficient than nearest neighbor classifiers.

In Section 7.1 we experimentally verify these observations by showing that in high-dimensional space it is easy to find adversarial examples even after training against a strong adversary.

For the remainder of this section we will consider the · 2 norm.

DISPLAYFORM1 Let X ⊂ M be a finite set of points sampled from M. Suppose that ≤ rch 2 Ξ where Ξ is the medial axis of M, defined as in BID12 .

Then the percentage of M covered by X is upper bounded by DISPLAYFORM2 As the codimension (d − k) → ∞, Equation 1 approaches 0, for any fixed |X|.In high codimension, even moderate under-sampling of M leads to a significant loss of coverage of M because the volume of the union of balls centered at the samples shrinks faster than the volume of M .

Theorem 2 states that in high codimensions the fraction of M covered by X goes to 0.

Almost nothing is covered by X for training set sizes that are realistic in practice.

Thus X is a poor model of M , and high classificaiton accuracy on X does not imply high accuracy in M .

The proof of Theorem 2 is given in Appendix A.Note that an alternative way of defining the ratio vol X / vol M is as vol (X ∩ M )/ vol M .

This is equivalent in our setting since X ⊂ M and so X ⊂ M .For the remainder of the section we provide intuition for Theorem 2 by considering the special case DISPLAYFORM3 ; that is Π is a subset of the x 1 -. .

.-x k -plane bounded between the coordinates [ , µ].

A δ-cover of a manifold M in the norm · 2 is a finite set of points X such that for every x ∈ M there exists X i such that x − X i 2 ≤ δ.

It is easy to construct an explicit δ-cover X of Π: place sample points at the vertices of a regular grid, shown in FIG1 by the black vertices.

The centers of the cubes of this regular grid, shown in blue in FIG1 , are the furthest points from the samples.

The distance from the vertices of the grid to the centers is √ k∆/2 where ∆ is the spacing between points along an axis of the grid.

To construct a δ-cover we need √ k∆/2 = δ which gives a spacing of ∆ = 2δ/ √ k. The size of this sample is |X| = DISPLAYFORM4 Note that |X| scales exponentially in k, the dimension of Π, not in d, the dimension of the embedding space.

Recall that Π δ is the δ-tubular neighborhood of Π. The δ-balls around X, which comprise X δ , cover Π and so any robust approach that guarantees correct classification within X δ will achieve perfect accuracy on Π. However, we will show that X δ covers only a vanishingly small fraction of Π δ .

Let B δ denote the d-ball of radius δ centered at the origin.

An upper bound on the volume of X δ is DISPLAYFORM5 Next we bound the volume vol Π δ from below.

Intuitively, a lower bound on the volume can be derived by placing a (d − k)-dimensional ball in the normal space at each point of Π and integrating the volumes.

DISPLAYFORM6 Combining Equations 2 and 3 gives an upper bound on the percentage of Π δ that is covered by X .

DISPLAYFORM7 Notice that the factors involving δ and (µ − ) cancel.

FIG3 (Left) shows that this expression approaches 0 as the codimension DISPLAYFORM8 Suppose we set δ = 1 and construct a 1-cover of Π. The number of points necessary to cover Π with balls of radius 1 depends only on k, not the embedding dimension d. However the number of points necessary to cover the tubular neighborhood Π 1 with balls of radius 1 increases depends on both k and d. In Theorem 3 we derive a lower bound on the number of samples necessary to cover Π 1 .Theorem 3.

Let Π be a bounded k-flat as described above, bounded along each axis by < µ.

Let n denote the number of samples necessary to cover the 1-tubular neighborhood Π 1 of Π with · 2 -balls of radius 1.

That is let n be the minimum value for which there exists a finite sample X of size n such that DISPLAYFORM9 Theorem 3 states that, in general, it takes many fewer samples to accurately model M than to model M .

FIG3 (Right) compares the number of points necessary to construct a 1-cover of Π with the lower bound on the number necessary to cover Π 1 from Theorem 3.

The number of points necessary to cover Π 1 increases as Ω (d − k) k/2 , scaling polynomially in d and exponentially in k. In contrast, the number necessary to construct a 1-cover of Π remains constant as d increases, depending only on k. The proof of Theorem 3 is given in Appendix A. Approaches that produce robust classifiers by generating adversarial examples in the -balls centered on the training set do not accurately model M , and it will take many more samples to do so.

If the method behaves arbitrarily outside of the -balls that define X , adversarial examples will still exist and it will likely be easy to find them.

The reason deep learning has performed so well on a variety of tasks, in spite of the brittleness made apparent by adversarial examples, is because it is much easier to perform well on M than it is to perform well on M .

Adversarial training, the process of training on adversarial examples generated in a · p -ball around the training data, is a very natural approach to constructing robust models BID21 ; Madry et al. FORMULA8 ).

In our notation this corresponds to training on samples drawn from X for some .

While natural, we show that there are simple settings where this approach is much less sample-efficient than other classification algorithms, if the only guarantee is correctness in X .Define a learning algorithm L with the property that, given a training set X ⊂ M sampled from a manifold M, L outputs a model f L such that for every x ∈ X with label y, and everyx DISPLAYFORM0 Here B(x, r) denotes the ball centered at x of radius r in the relevant norm.

That is, L learns a model that outputs the same label for any · p -perturbation of x up to rch p Λ p as it outputs for x. L is our theoretical model of adversarial training BID21 ; Madry et al. FORMULA8 ).

Theorem 4 states that L is sample inefficient in high codimensions.

Theorem 4.

There exists a classification algorithm A that, for a particular choice of M, correctly classifies M using exponentially fewer samples than are required for L to correctly classify M .Theorem 4 follows from Theorems 5 and 6.

In Theorems 5 and 6 we will prove that a nearest neighbor classifier f nn is one such classification algorithm.

Nearest neighbor classifiers are naturally robust in high codimensions because the Voronoi cells of X are elongated in the directions normal to M when X is dense (Dey FORMULA11 ).Recall that a δ-cover of a manifold M in the norm · p is a finite set of points X such that for every x ∈ M there exists X i such that x − X i p ≤ δ.

Theorem 5 gives a sufficient sampling condition for f L to correctly classify M for all manifolds M. Theorem 5 also provides a sufficient sampling condition for a nearest neighbor classifier f nn to correctly classify M , which is substantially less dense than that of f L .

Thus different classification algorithms have different sampling requirements in high codimensions.

Theorem 5.

Let M ⊂ R d be a k-dimensional manifold and let < rch p Λ p for any p. Let f nn be a nearest neighbor classifier and let f L be the output of a learning algorithm L as described above.

Let X nn , X L ⊂ M denote the training sets for f nn and L respectively.

We have the following sampling guarantees: DISPLAYFORM1 The bounds on δ in Theorem 5 are sufficient, but they are not always necessary.

There exist manifolds where the bounds in Theorem 5 are pessimistic, and less dense samples corresponding to larger values of δ would suffice.

In Theorem 6 we show a setting where bounds on δ similar to those in Theorem 5 are necessary.

In this setting, the difference of a factor of 2 in δ between the sampling requirements of f nn and f L leads to an exponential gap between the sizes of X nn and X L necessary to achieve the same amount of robustness.

Consider two subsets of k-flats Π 1 , Π 2 , as defined in Section 5, where Π 1 lies in the subspace x d = 0 and Π 2 lies in the subspace x d = 1; thus rch 2 Λ 2 = 1.

In the · 2 norm we can show that the gap in Theorem 5 is necessary for Π = Π 1 ∪ Π 2 .

Furthermore the bounds we derive for δ-covers for Π for both f nn and f L are tight.

Combined with well-known properties of covers, we get that the ratio |X L |/|X nn | is exponential in k. FORMULA8 is no guarantee of robustness; as the codimension increases it becomes easier to find adversarial examples using BIM attacks.

Appendix B.4 shows the performance on nearest neighbor on this data, which is essentially perfect accuracy for all .Theorem 6.

Let Π = Π 1 ∪ Π 2 as described above.

Let X nn , X L ⊂ Π be minimum training sets necessary to guarantee that f nn and f L correctly classify M .

Then we have that DISPLAYFORM2 We have shown that both L and nearest neighbor classifiers learn robust decision boundaries when provided sufficiently dense samples of M. However there are settings where nearest neighbors is exponentially more sample-efficient than L in achieving the same amount of robustness.

We experimentally verify these theoretical results in Section 7.1.

Proofs for all of the results in this section are provided in Appendix A.

Section 5 suggests that as the codimension increases it should become easier to find adversarial examples.

To verify this, we introduce two synthetic datasets, CIRCLES and PLANES, which allow us to carefully vary the codimension while maintaining dense samples.

The CIRCLES dataset consists of two concentric circles in the x 1 -x 2 -plane, with rch 2 Λ 2 = 1.

We densely sample 1000 random points on each circle for both the training and the test sets.

The PLANES dataset consists of two 2-dimensional planes, the first in the x d = 0 and the second in x d = 2, so that rch 2 Λ 2 = 1.

We sample the training set at the vertices of the grid described in Section 5, and the test set at the centers of the grid cubes, the blue points in FIG1 .

Further details are provided in Appendix E and visualizations in Appendix H.We consider two attacks, the fast gradient sign method (FGSM) BID21 ) and the basic iterative method (BIM) (Kurakin et al. (2016) ) under · 2 .

We use the implementations provided in the cleverhans library (Papernot et al. FORMULA8 ).

Further implementation details are provided in Appendix E. Our experimental results are averaged over 20 retrainings of our model architecture, using Adam (Kingma & Ba FORMULA8 ).

Further implementation details are provided in Appendix E. FIG4 (Left, Center) shows FGSM and BIM attacks on the CIRCLES dataset as we vary the codimension.

For both attacks we see a steady decrease in robustness as we increase the codimension, on average.

et al. (2018) propose training against a PGD adversary to improve robustness.

Section 5 suggests that this should be insufficient to guarantee robustness, as X is often a poor model for M .

We train against a PGD adversary with = 1 under · 2 -perturbations on the PLANES dataset.

FIG4 (Right) shows that it is still easy to find adversarial examples for < 1 and that as the codimension increases we can find adversarial examples for decreasing values of .

In contrast, nearest neighbor achieves perfect robustness for all epsilon on this data (see Appendix B.4 for details).

To explore performance on a more realistic dataset, we compared nearest neighbors with robust and natural models on MNIST.

We considered two attacks: BIM under l ∞ norm against the natural and robust models as well as a custom attack against nearest neighbors.

Each of these attacks are generated from the MNIST test set.

Architecture details can be found in Appendix E. FIG5 (Left) shows that nearest neighbors is substantially more robust to BIM attacks than the naturally trained model.

FIG5 (Center) shows that nearest neighbors is comparable to the robust model up to = 0.3, which is the value for which the robust model was trained.

After = 0.3, nearest neighbors is substantially more robust to BIM attacks than the robust model.

At = 0.5, nearest neighbors maintains accuracy of 78% to adversarial perturbations that cause the accuracy of the robust model to drop to 0%.

In Appendix B.2 we provide a similar result for FGSM attacks.

Figure 6 (Right) shows the performance of nearest neighbors and the robust model on adversarial examples generated for nearest neighbors.

The nearest neighbor attacks are generated as follows: iteratively find the k nearest neighbors and compute an attack direction by walking away from the neighbors in the true class and toward the neighbors in other classes.

We find that nearest neighbors is able to be tricked by this approach, but the robust model is not.

This indicates that the errors of these models are distinct and suggests that ensemble methods may effectively get the best of both worlds.

Additionally, a closer investigation shows strong qualitative differences between the BIM adversarial examples and the examples generated for nearest neighbors.

Appendix J argues that the adversarial examples that fool nearest neighbor line up better with human intuition.

We have presented a geometric framework for proving robustness guarantees for learning algorithms.

Our framework is general and can be used to describe the robustness of any classifier.

We have shown that no single model can be simultaneously robust to attacks under all norms and that nearest neighbor classifiers are theoretically more sample efficient than adversarial training.

Most importantly, we have highlighted the role of codimension in contributing to adversarial examples and verified our theoretical contributions with experimental results.

We believe that a geometric understanding of the decision boundaries learned by deep networks will lead to both new geometrically inspired attacks and defenses.

In Appendix C we provide a novel gradient-free geometric attack in support of this claim.

Finally we believe future work into the geometric properties of decision boundaries learned by various optimization procedures will provide new techniques for black-box attacks.

A.1 AUXILIARY LEMMAS DISPLAYFORM0 Let Λ p be their decision axis for any p and let γ : [0, 1] → R d be any path such that γ(0) ∈ M 1 and γ(1) ∈ M 2 .

Then γ ∩ M = ∅, that is γ must cross the decision axis.

DISPLAYFORM1 Consider the function g(t) = f 1 (t) − f 2 (t).

Since M 1 ∩ M 2 = ∅ and γ starts on M 1 and terminates on M 2 the function g(0) < 0 and g(1) > 0.

Then, since g is continuous, the Intermediate Value Theorem implies that there exists t 1 ∈ [0, 1] such that g(t 1 ) = 0.

Thus d(γ(t 1 ), M 1 ) = d(γ(t 1 ), M 2 ), which implies that γ(t 1 ) is on the decision axis Λ.Theorem 8.

Let f be any classifier on M = M 1 ∪ M 2 .

The maximum accuracy achievable, assuming a uniform distribution, on M is DISPLAYFORM2 Proof.

It is clearly optimal to classify points in vol(M 1 \ M 2 ) as class 1 and to classify points in vol(M 2 \ M 1 ) as class 2.

Such a classifier can only be wrong when points lie in this intersection.

For points in this intersection, the probability of a misclassification is 1 2 for any classification that f makes.

Thus, the probability of misclassification is DISPLAYFORM3 Corollary 9.

For < rch p (Λ p ; M) there exists a decision boundary that correctly classifies M .Proof.

For < rch p Λ p , M ∩ Λ p = ∅ and so Λ p is one such decision boundary.

Proof.

The decision axis under · 2 , Λ 2 , is just the d-sphere with radius (r 1 + r 2 )/2.

However, Λ ∞ is not identical to Λ 2 in this setting; in fact most Λ ∞ of approaches S 1 as d increases.

The geometry of a · ∞ -ball B ∆ centered at m ∈ R d with radius ∆ is that of a hypercube centered at m with side length 2∆. To find a point on Λ ∞ we place B ∆ tangent to the north pole q of S 1 so that the corners of B ∆ touch S 2 .

The north pole has coordinate representation q = (0, . . .

, 0, r 1 ), the center m = (0, . . .

, 0, r 1 + ∆), and a corner of B ∆ can be expressed as p = (∆, . . .

, ∆, r 1 + 2∆).

Additionally we have the constraint that p 2 = r 2 since p ∈ S 2 .

Then we can solve for ∆ as DISPLAYFORM0 where the last step follows from the quadratic formula and the fact that ∆ > 0.

For fixed r 1 , r 2 , the value ∆ scales as DISPLAYFORM1 A.3 PROOF OF THEOREM 2Proof.

Assuming the balls centered on the samples in X are disjoint we get the upper bound DISPLAYFORM2 This is identical to the reasoning in Equation 2.The medial axis Ξ of M is defined as the closure of the set of all points in R d that have two or more closest points on M in the norm · 2 .

The medial axis Ξ is similar to the decision axis Λ 2 , except that the nearest points do not need to be on distinct class manifolds.

For ≤ rch 2 Ξ, we have the lower bound DISPLAYFORM3 Combining Equations 8 and 9 gives the result.

To get the asymptotic result we apply Stirling's approximation to get Γ( DISPLAYFORM4 The last step follows from the fact that lim DISPLAYFORM5 , where e is the base of the natural logarithm.

Proof.

We first construct an upper bound by generously assuming that the balls centered at the samples are disjoint.

That is DISPLAYFORM0 To guarantee that Π 1 ⊂ ∪ x∈X B(x, 1) = X 1 we set the left hand side of Equation 10 equal to 1 and solve for n. DISPLAYFORM1 The last inequality follows from Equation 3.

Setting δ = 1 gives the result.

The asymptotic result is similar to the argument in the proof of Theorem 2.

Proof.

We begin by proving (1).

Let q ∈ M be any point in M .

Suppose without loss of generality that q ∈ M i for some class i. The distance d(q, M j ) from q to any other data manifold M j , and thus any sample on M j , is lower bounded by d(q, M j ) ≥ 2 rch p Λ p − .

It is then both necessary and sufficient that there exists a x ∈ M i such that d(q, x) < 2 rch p Λ p − for f nn (q) = i.(Necessary since a properly placed sample on M j can achieve the lower bound on d(q, M j ).)

The distance from q to the nearest sample x on M i is d(q, x) ≤ + δ for some δ > 0.

The question is how large can we allow δ to be and still guarantee that f nn correctly classifies M ?

We need DISPLAYFORM0 which implies that δ ≤ 2(rch p Λ p − ).

It follows that a δ-cover with δ = 2(rch p Λ p − ) is sufficient, and in some cases necessary, to guarantee that f nn correctly classifies M .Next we prove (2).

As before let q ∈ M i .

It is both necessary and sufficient for q ∈ B rchp Λp (x) for some sample x ∈ M i to guarantee that f L (q) = i, by definition of L. The distance to the nearest sample x on M i is d(q, x) ≤ + δ for some δ > 0.

Thus it suffices that δ ≤ rch p Λ p − .

Proof.

Let q ∈ Π 1 .

Since Π 1 is flat, the distance to from q to the nearest sample x ∈ Π 1 is bounded as q − x 2 ≤ √ 2 + δ 2 .

For f nn (q) = 1 we need that q − x 2 ≤ 2 − , and so it suffices that δ ≤ 2 √ 1 − .

In this setting, this is also necessary; should δ be any larger a property placed sample on Π 2 can claim q in its Voronoi cell.

Similarly for f L (q) = 1 we need that q − x 2 ≤ 1, and so it suffices that δ ≤ √ 1 − 2 .

In this setting, this is also necessary; should δ be any larger, q lies outside of every · 2 -ball B 1 (x) and so L is free to learn a decision boundary that misclassifies q.

Let N (δ, M) denote the size of the minimum δ-cover of M. Since Π is flat (has no curvature) and since the intersection of Π with a d-ball centered at a point on Π is a k-ball, a standard volume argument can be applied in the affine subspace aff Π to conclude that N (δ, Π) ∈ Θ vol k Π/δ k .

So we have DISPLAYFORM0 Since Π is constant in both settings, the factor vol k Π as well as the constant factors hidden by Θ(·) cancel.

(Note that we are using the fact that Π 1 , Π 2 have finite k-dimensional volume.)

The inequality follows from the fact that the expression (1 + ) −k/2 is monotonically decreasing on the interval [0, 1] and takes value 2 −k/2 at = 1.

We present additional experiments to support our theoretical predictions.

We reproduce the results of Section 7 using different optimization algorithms (Section B.1) and attack methods (Section B.2).

These additional experiments are consistent with our conclusions in Section 7.

Additionally we provide evidence that adversarial perturbations lie mostly in the directions of the normal space (Section B.3).

We show that a nearest neighbor classifier is robust in high codimensions (Section B.4).

Finally we show that increasing the sampling density substantially does not notably improve the robustness of adversarial training (Section B.5).

In Section 7.1 we showed that increasing the codimension reduces the robustness of the decision boundaries learned by Adam on CIRCLES.

In FIG6 we reproduce this result using SGD.

Again we see that as we increase the codimension the robustness decreases.

SGD presents with much less variances than Adam, which we attribute to implicit regularization that has been observed for SGD (Soudry et al. FORMULA8 Figure 8: Adverarial training with a PGD adversary, as in FIG4 , using SGD.

Similarly we see a drop in robustness as the codimension increases.

In Section 7.1 we evaluated the robustness of nearest neighbors against BIM attacks under the · ∞ on MNIST.

In FIG7 we evaluate the robustness of nearest neighbors against FGSM attacks under the · ∞ on MNIST.

We use the naturally pretrained (natural) and adversarially pretrained (robust) convolutional models provided by Madry et al. (2018) 1 .

FIG7 (Left) shows that nearest neighbors is substantially more robust to FGSM attacks than the naturally trained model.

FIG7 (Right) shows that nearest neighbors is comparable to the robust model up to = 0.3, which is the value for which the robust model was trained.

After = 0.3, nearest neighbors is substantially more robust to FGSM attacks than the robust model.

At = 0.5, nearest neighbors maintains accuracy of 78% to adversarial perturbations that cause the accuracy of the robust model to drop to 39%.

Let η x be an adversarial perturbation generated by FGSM with = 1 at x ∈ M. Note that the adversarial example is constructed asx = x + η x .

In Figure 10 we plot a histogram of the angles ∠(η x , N x M) between η x and the normal space N x M for the CIRCLES dataset in codimensions 1, 10, 100, and 500.

In codimension 1, 88% of adversarial perturbations make an angle of less than 10 • with the normal space.

Similarly in codimension 10, 97%, in codimension 100, 96%, and in codimension 500, 93%.

As Figure 10 shows, nearly all adversarial perturbations make an angle less than 20• with the normal space.

Our results are averaged over 20 retrainings of the model using SGD.Throughout this paper we've argued that high codimension is a key source of the pervasiveness of adversarial examples.

Figure 10 shows that adversarial perturbations are well aligned with the normal space.

When the codimension is high, there are many directions normal to the manifold and thus many directions in which to construct adversarial perturbations.

In Section 7.1 we showed that the robustness of learned decision boundaries decreased as the codimension increased.

In Figure 11 we repeat the experiment in FIG4 , in which we measured the robustnesss of our neural network models to FGSM attacks as the codimension increased.

We repeat this experiment using nearest neighbors to classify the adversarial examples generated by FGSM.

Figure 11 shows that nearest neighbors is robust even when the codimension is high, as long as the low-dimensional data manifold is well sampled.

This is a consquence of the fact that the Voronoi cells of the samples are elongated in the directions normal to the data manifold when the sample is dense.

The PLANES dataset is sampled so that the trianing set is a 1-cover of the underlying planes, which requires 450 sample points.

FIG1 shows the results of increasing the sampling density to a 0.5-cover (1682 samples) and a 0.25-cover (6498 samples).

Increasing the sampling density improves the robustness of adversarial training at the same codimension and particularly in low-codimension.

However adversarial training with a substantially larger training set does not produce a classifier as robust as a nearest neighbor classifier on a much smaller training set.

Nearest neighbors is much more sample efficient than adversarial training, as predicted by Theorem 5 and experimentally verified in Section B.4.

C A GRADIENT-FREE GEOMETRIC ATTACK Most current attacks rely on the gradient of the loss function at a test sample to find a direction towards the decision boundary.

Partial resistance against such attacks can be achieved by obfuscating the gradients, but BID3 showed how to circumvent such defenses.

BID6 propose a gradient-free attack for · 2 , that starts from a misclassifed point and walks toward the original point.

In this section we propose a gradient-free attack that only requires oracle access to a model, meaning we only query the model for a prediction.

Consider a point x ∈ X test and the · p -ball B r (x) centered at x of radius r. To construct an adversarial perturbation η x ∈ B r (x), giving an adversarial examplex = x+η x , we project every point in X test onto B r (x) and query the oracle for a prediction Increasing the sampling density improves robustness at the same codimension.

However even training on a significantly denser training set does not produce a classifier as robust as a nearest neighbor classifier on a much sparser training set, Figure 12 for each point.

If y ∈ X test projected to a point y that the model classified differently than x, we take η x = y − x, otherwise η x = 0.

This incredibly simple attack reduces the accuracy of the pretrained robust model of Madry et al. (2018) for · ∞ and = 0.3 to 90.6%, less than two percent shy of the current SOTA for whitebox attacks, 88.79% (Zheng et al. (2018) ).Simple datasets, such as CIRCLES and PLANES, allow us to diagnose issues in learning algorithms in settings where we understand how the algorithm should behave.

For example BID3 state that the work of Madry et al. FORMULA8 does not suffer from obfuscated gradients.

In Appendix D we provide evidence that Madry et al. (2018) does suffer from the obfuscated gradients problem, failing one of BID3 's criteria for detecting obfuscated gradients.

D THE MADRY DEFENSE SUFFERS FROM OBFUSCATED GRADIENTS BID3 identified the problem of "obfuscated gradients", a type of a gradient masking (Papernot et al. (2017) ) that many proposed defenses employed to defend against adversarial examples.

They identified three different types of obfuscated gradients: shattered gradients, stochastic gradeints, and exploding/vanishing gradients.

They examined nine recently proposed defenses, concluded that seven suffered from at least one type of obfuscated gradient, and showed how to circumvent each type of obfuscated gradient and thus each defense that employed obfuscated gradients.

Regarding the work of Madry et al. (2018) , BID3 stated "We believe this approach does not cause obfuscated gradients".

They note that "our experiments with optimization based attacks do succeed with some probability".

In this section we provide evidence that the defense of Madry et al. (2018) does suffer from obfuscated gradients, specifically shattered gradients.

Shattered gradients occur when a defence causes the gradient field to be "nonexistent or incorrect" BID3 ).

Specifically we provide evidence that the defense of Madry et al. (2018) works by shattering the gradient field of the loss function around the data manifolds.

In FIG3 (Left) we show the normalized gradient field of the loss function for a network trained on a 2-dimensional version of our PLANES dataset using the adversarial training procedure of Madry et al. (2018) with a PGD adversary.

While the gradients have meaningful directions, FIG3 (Left) shows that magnitude of the gradient field is nearly 0 everywhere around the data manifolds, which are at y = 0 and y = 2.

The only notable gradients are near the decision axis which is at y = 1.

One criteria that BID3 propose for identifying obfuscated gradients is whether onestep attacks perform better than iterative attacks.

The reason this criteria is useful for identifying obfuscated gradients is because one-step attacks like FGSM first normalize the gradient, ignoring its magnitude, then take as large of a step as allowed in the direction of the normalized gradient.

So long as the gradient on the manifold points towards the decision boundary, FGSM will be effective at finding an adversarial example.

In FIG4 we show the adversarial examples generated using PGD (left), FGSM (center), and BIM (right) for = 1 starting at the test set for the PLANES dataset.

FGSM produces adversarial examples at the decision axis y = 1, exactly where we would expect.

Notice that all of the adversarial perturbation is normal to the data manifold, suggesting that the gradient on the manifold points towards the decision boundary.

However the adversarial examples produced by PGD lie closer to the manifold from which the example was generated.

PGD splits the total perturbation between both the normal and the tangent spaces of the data manifold, as shown by the arrows in FIG4 .

This suggests that, when trained adversarially, the network learned a gradient field that has small but correct gradients on the data manifold, but gradients that curve in the tangent directions immediately off the manifold.

Lastly notice that BIM, another iterative method, also produces adversarial examples that are near the decision axis.

BID3 cite success with iterative based optimization procedures as evidence against obfuscated gradients.

However BIM also ignores the magnitude of the gradient, as it simply applies FGSM iteratively.

The network has learned a gradient field that is overfit to the particulars of the PGD attack.

BIM successfully navigates this gradient field, while PGD does not.

While the network is robust to PGD attacks at test time, it is less robust to FGSM and BIM attacks.

In Section 7 we introduced two synthetic datasets, CIRCLES and PLANES.

The CIRCLES dataset consists of two concentric circles, the first with radius r 1 = 1 and the second with radius r 2 = 3, so that the rch = 1.

The PLANES dataset consists of two 2-dimensional planes, the first in the subspace defined by x d = 0, and the second in x d = 2, so that rch = 1.

The first two axis of both planes are

Figure 15: Adverarial examples generated using PGD (left), FGSM (center), and BIM (right).

While the network is robust to PGD attacks, FGSM and BIM attacks are more effective because they ignore the magnitude of the gradient.

For PGD we draw arrows from the test sample to the adversarial example generated from that point to aid the reader.bounded as −10 ≤ x 1 , x 2 ≤ 10, while x 3 = . . .

= x d−1 = 0.

Both planes are sampled as described in Section 5, so that X 1 covers the underlying planes, where X is the training set.

We consider three attacks, FGSM, BIM, and PGD, primarily under the · 2 norm.

For the iterative attacks BIM and PGD, we set the number of iterations to 30 with a step size of step = 0.05 per iteration.

Our controlled experiments on synthetic data consider a fully connected network with 1 hidden layer, 100 hidden units, and ReLU activations.

This model architecture is more than capable of representing a nearly perfect robust decision boundary for both CIRCLES and PLANES, the latter of which is linearly separable.

We set the learning rate for Adam as α = 0.1, which we found to work best for our datasets.

The parameters for the exponential decay of the first and second moment estimates were set to β 1 = 0.9 and β 2 = 0.999.

We set the learning rate for SGD as α = 0.1 and decrease the learning rate by a factor of 10 every 100 epochs.

We train all of our models for 250 epochs, following Wilson et al. (2017) .All of our experiments are implemented using PyTorch.

When comparing against a published result we use publicly available repositories, if able.

For the robust loss of Wong & Kolter (2018), we use the code provided by the authors 2 .The provided implementation 3 of the adversarial training procedure of Madry et al. (2018) considers a PGD adversary with · ∞ -perturbations.

We reimplemented their adversarial training procedure for · 2 -perturbations following their implementation and using the PGD attack implemented in the cleverhans library (Papernot et al. (2018) ).The models of Madry et al. (2018) consist of two convolutional layers with 32 and 64 filters respectively, each followed by 2 × 2 max pooling.

After the two convolutional layers, there are two fully connected layers each with 1024 hidden units.

DISPLAYFORM0 The volume of S is given by DISPLAYFORM1 where Γ denotes the gamma function.

Let X ⊂ S be a finite sample of size n of S. The set X is the set of all perturbations of points in X under the norm · 2 .

How well does X approximate S as a function of n, d and ?

To answer this question we upper bound the ratio vol X / vol S by generously assuming that the balls B(X i , ) are disjoint.

The resulting upper bound is DISPLAYFORM2 In FIG5 we show three different views of this bound.

In FIG5 (Left) we set n = 10 12 and plot four different values of ; in each case the percentage of volume of S that is covered by X quickly approaches 0.

Similarly, in FIG5 (Center), if we fix = 1 and plot four different values of n, in each case we have the same result.

Finally in FIG5 (Right) we plot a lower bound on number of samples necessary to cover S by X for four different values of ; in each case the number of samples necessary grows exponentially with the dimension.

The Delaunay triangulation of X, denoted Del X is a triangulation of the convex hull of X into d-simplices.

Every d-simplex τ ∈ Del X, as well as every lower-dimensional face of τ , has the defining property that there exists an empty circumscribing ball B such that the vertices of τ lie on the boundary of B and the interior of B is free from any points in X. See FIG6 .

This empty circumscribing ball property of Delaunay triangulations implies many desirable properties that are useful in mesh generation BID9 ) and manifold reconstruction BID16 ).

The Delaunay triangulation of a point set always exists, but is not unique in general.

There exists a well known duality between the Voronoi diagram and the Delaunay triangulation of X. For every j-dimensional face σ ∈ Vor X there exist a dual (d − j)-dimensional simplex denoted σ * ∈ Del X whose d − j + 1 vertices are the d − j + 1 vertices of X whose Voronoi cells intersect at σ.

In particular, every d-cell of Vor X is dual to the vertex of Del X that generates that cell, and every (d − 1)-face of Vor X is dual to an edge of Del X.

In Figure 18 we provide visualizations of our two synthetic datasets, CIRCLES (left) and PLANES (right).

Figure 18 : We create two synthetic datasets which allow us to perform controlled experiments on the affect of codimension on adversarial examples.

In FIG7 we provide visualizations of the decision boundaries learned by (a-d) our fully connected network architecture with cross entropy loss for various optimization procedures and various training lengths, (e) our fully connected network architecture trained using the robust loss of Wong & Kolter (2018) for · ∞ -perturbations, and (f) a nearest neighbor classifier for · 2 on the training set.

Specifically we train on the CIRCLES dataset, embedded in R 3 .

The training set is entirely contained in the xy-plane.

We then visualize cross sections of the decision boundary for various values of z ∈ [−5, 5].

We color points labeled as in the same class as the outer circle with the color blue and points labeled as in the same class as the inner circle as orange.

FIG7 shows the cross sections of the decision boundaries, averaged over 20 retrainings.

The visualization shows how various optimization algorithms learn decision boundaries that extend into the normal directions where no data is provided. (2018) .

The top row is the original data that the examples were generated from.

Each figure is labelled with the predictions from robust neural network.

We observe an immediate qualitative difference between the nearest neighbor examples and the BIM examples: the nearest neighbors ones are starting to look like numbers from a target class!

In fact, we can reasonably argue that the classifications of the robust model that don't change represent as much of an error and being fooled by a standard adversarial example.

For example the center right image would be classified as an 8 by most people, but the neural network is confident it is a 0.

This provides evidence that nearest neighbors is doing a better job of the learning the human decision boundary between numbers.

<|TLDR|>

@highlight

We present a geometric framework for proving robustness guarantees and highlight the importance of codimension in adversarial examples. 

@highlight

This paper gives a theoretical analysis of adversarial examples, showing that  there exists a tradeoff between robustness in different norms, adversarial training is sample inefficient, and the nearest neighbor classifier can be robust under certain conditions.