We study the problem of learning permutation invariant representations that can capture containment relations.

We propose training a model on a novel task: predicting the size of the symmetric difference between pairs of multisets, sets which may contain multiple copies of the same object.

With motivation from fuzzy set theory, we formulate both multiset representations and how to predict symmetric difference sizes given these representations.

We model multiset elements as vectors on the standard simplex and multisets as the summations of such vectors, and we predict symmetric difference as the l1-distance between multiset representations.

We demonstrate that our representations more effectively predict the sizes of symmetric differences than DeepSets-based approaches with unconstrained object representations.

Furthermore, we demonstrate that the model learns meaningful representations, mapping objects of different classes to different standard basis vectors.

Tasks for which the input is an unordered collection, i.e. a set, are ubiquitous and include multipleinstance learning Ilse et al. (2018) , point-cloud classification Zaheer et al. (2017) ; Qi et al. (2017) , estimating cosmological parameters Zaheer et al. (2017) ; Ravanbakhsh et al. (2016) , collaborative filtering Hartford et al. (2018) , and relation extraction Verga et al. (2017) ; Rossiello et al. (2019) .

Recent work has demonstrated the benefits of permutation invariant models that have inductive biases well aligned with the set-based input of the tasks (Ilse et al., 2018; Qi et al., 2017; Zaheer et al., 2017; Lee et al., 2019) .

The containment relationship between sets -and intersection more generally -is often considered as a measure of relatedness.

For instance, when comparing the keywords for two documents, we may wish to model that {currency, equilibrium} describes a more specific set of topics than (i.e. is "contained" in) {money, balance, economics}. The containment order is a natural partial order on sets.

However, we are often interested not in sets, but multisets, which may contain multiple copies of the same object; examples include bags-of-words, geo-location data over a time period, and data in any multiple-instance learning setting (Ilse et al., 2018) .

The containment order can be extended to multisets.

Learning to represent multisets in a way that respects this partial order is a core representation learning challenge.

Note that this may require modeling not just exact containment, but relations that consider the relatedness of individual objects.

We may want to learn representations of the multisets' elements which induce the desired multiset relations.

In the aforementioned example, we may want money ≈ currency and balance ≈ equilibrium.

Previous work has considered modeling hierarchical relationships or orderings between pairs of individual items (Ganea et al., 2018; Lai and Hockenmaier, 2017; Nickel and Kiela, 2017; Suzuki et al., 2019; Vendrov et al., 2015; Vilnis et al., 2018; Vilnis and McCallum, 2015; Li et al., 2019; Athiwaratkun and Wilson, 2018) .

However, this work does not naturally extend from representing individual items to modeling relations between multisets via the elements' learned representations.

Furthermore, we may want to consider richer information about the relationship between two multisets beyond containment, such as the size of their intersection.

In this paper, we present a measure-theoretic definition of multisets, which lets us formally define the "flexible containment" notion exemplified above.

The theory lets us derive method for learning representations of multisets and their elements, given the relationships between pairs of multisets -in particular, we propose to use the sizes of their symmetric differences or of their intersections.

We learn these representations with the goal of predicting the relationships between unseen pairs of multisets (whose elements may themselves have been unseen during training).

We prove that this allows us to predict containment relations between unseen pairs of multisets.

We show empirically that the theoretical basis of our model is important for being able to capture these relations, comparing our approach to DeepSets-based approaches (Zaheer et al., 2017) with unconstrained item representations.

Furthermore, we demonstrate that our model learns "meaningful" representations.

2 RELATED WORK 2.1 SET REPRESENTATION Qi et al. (2017) and Zaheer et al. (2017) both explore learning functions on sets.

Importantly, they arrive at similar theoretical statements about the approximation of such functions, which rely on permutation invariant pooling functions.

In particular, Zaheer et al. (2017) show that any set function f (A) can be approximated by a model of the form ρ a∈A φ(a) for some learned ρ and φ, which they call DeepSets.

They note that the sum can be replaced by a max-pool (which is essentially the formulation of Qi et al. (2017) ), and observe empirically that this leads to better performance.

1 More recently, there has been some very interesting work on leveraging the relationship between sets.

Probst (2018) proposes a set autoencoder, while Skianis et al. (2019) learn set representations with a network that compares the input set to trainable "hidden sets."

However, both these approaches require solving computationally expensive matching problems at each iteration.

Vendrov et al. (2015) and Ganea et al. (2018) seek to model partial orders on objects via geometric relationships between their embeddings -namely, using cones in Euclidean space and hyperbolic space, respectively.

Nickel and Kiela (2017) use a similar idea to embed hierarchical network structures in hyperbolic space, simply using the hyperbolic distance between embeddings.

These approaches are unified under the framework of "disk embeddings" by Suzuki et al. (2019) .

The idea is to map each object to the product space X × R, where X is a (pseudo-)metric space.

This mapping can be expressed as A → (f (A), r(A)), and it is trained with the objective that A B if and only if d X (f (A), f (B)) ≤ r(B) − r(A).

An equivalent statement can be made for multisets (see Proposition 3.2.4) .

Other work has taken a probabilistic approach to the problem of representing hierarchical relationships.

Lai and Hockenmaier (2017) attempt to formulate the Order Embeddings of Vendrov et al. (2015) probabilistically, modeling joint probabilities as the volumes of cone intersections.

Vilnis et al. (2018) represent entities as "box embeddings," or rectangular volumes, where containment of one box inside another models order relationships between the objects.

(Marginal and conditional probabilities can be computed from intersections of boxes.)

Vilnis and McCallum (2015) propose modeling words as Gaussian distributions in order to capture notions of entailment and generality, and this work has been extended to mixtures of Gaussians by Athiwaratkun and Wilson (2017) .

The theory of fuzzy sets can be traced back to Zadeh (1965) .

A fuzzy set A of objects from a universe U is defined via its membership function µ A : U → [0, 1].

Fuzzy set operations -such as intersection -are then defined in terms of this function.

In modern fuzzy set theory, intersection is usually defined via a t-norm, which is a function T : [0, 1] 2 → [0, 1] satisfying certain properties.

The intersection of two fuzzy sets A and B is defined via the membership function µ A∩B (x) = T (µ A (x), µ B (x)). (More in-depth background, including the defining properties of t-norms, is provided in Appendix B.) There is also more recent literature on extending fuzzy set theory to multisets (Casasnovas and Mayor, 2008; Miyamoto, 2000) , using a membership function of the form µ A : U × [0, 1]

→ N, where µ A (x, α) is the number of appearances in A of an object x with membership α.

Our goal is to learn to represent and predict a notion of containment between multisets.

We begin with a brief motivating example, and then move on to provide the formalization of the problem.

Recall our example, where for A = {currency, equilibrium} and B = {money, balance, economics}, we have a sense in which A is "contained" in B. After seeing many such example pairs of multisets, we want to be able to deduce that for A = {currency, food} and B = {money, food}, the relation A ⊆ B holds.

In general, there exists a universe Ω of objects (in our above example, words).

We let Ω * denote the set of all multisets of objects from Ω, formally defined as follows.

Definition 3.1.1 A multiset A is defined by a its membership function m A : Ω → M , where M ⊆ R + is a subset of the non-negative reals, and m A maps each object to the "number of times" it occurs in A.

The choice of M dictates the kind of multiset A is.

In particular, M = {0, 1} gives classical sets, and M = N gives the traditional notion of multiset -a set which may contain multiple copies of the same object.

If M = R + , then we call A a "fuzzy multiset." 2 The cardinality (or size) of a multiset is defined with respect to a measure λ on Ω.

We will always fix some measure λ on Ω (which may be called the dominating measure) and take all cardinalities with respect to λ.

Note that we can always view m A as a density (i.e. the RadonNikodym derivative) of some measure µ A on Ω with respect to λ.

We can thus identify a multiset A with the measure µ A on Ω, and write |A| = µ A (Ω).

In the case that the universe Ω is countable, we simply let λ be the counting measure, in which case the cardinality of any multiset A is |A| = x∈Ω m a (x).

All the usual operations on pairs of multisets are defined in terms of their membership functions.

These definitions are standard for multisets with both whole-number and real-valued memberships (Casasnovas and Mayor, 2008; Miyamoto, 2000; Blizard, 1989) .

To those familiar with fuzzy set theory, it should immediately stand out the intersection and union are given by the standard T-norm and T-conorm (functions used to define these operations on fuzzy sets; see Appendix B).

This means that our definition of fuzzy multiset contains a copy of fuzzy set theory.

Unfortunately, there is no intuitive way to use other T-norms in order to define multiset operations. (For intuition on the above operations and why this is the case, see Appendix A.)

Finally, for multisets, containment is formally defined as follows.

2 Note that this is not the same formulation of "fuzzy multisets" usually given in literature (Casasnovas and Mayor, 2008; Miyamoto, 2000) .

However, this formulation will be much more easily amenable to the machine-learning setting.

Our notion here is also more closely related to the "real-valued multisets" of Blizard (1989) , although the author approaches the subject from a standpoint of formal logic and axiomatic set theory.

Note however that as demonstrated by our motivating example above, for two multisets A and B, we want to have a more "flexible" notion than A ⊆ B. (It is not actually the case that {currency, equilibrium} ⊆ {money, balance, economics}.) We will now formally provide a structure allowing for this flexibility.

Let us first make two observations.

Firstly, the desired flexibility will depend on some notion of "similairty" between the objects in Ω. Secondly, this similarity must be externally provided by our observations of these "subset" relations.

In our example above, we had a sense that money ≈ currency and balance ≈ equilibrium, because we observed that A is "contained" in B (perhaps along with many other similar examples).

We now formalize this idea.

Definition 3.2.1 For two universes Ω and U, a map T : Ω * → U * is a called a multiset transformation from Ω to U.

The idea is that there exists some multiset transformation T from Ω to U, but we may not observe the structure of T or this new universe U. However, we indirectly observe this structure, because our notion of subsets will be taken in U * rather than in Ω * .

In particular, in our example above, the sense in which A is "contained" in B is that T (A) ⊆ T (B).

The simplest example of such a setting, on which we focus, is when U = {1, . . .

, k} = [k].

That is, T maps each multiset in Ω * to a multiset of numbers from 1 to k. These numbers can be thought of as "tags," "classes," or "labels," meaning that each multiset has associated to it some tags, each of which may occur more than once.

Say for our running example, T (A) = {1, 2} and T (B) = {1, 2, 3}. We then obtain that T (A) ⊆ T (B), as desired.

The example mapping T above is suggestive.

It suggests a category of such T functions that are commonly useful: when each object in Ω is itself associated with a "tag" in U. Here, money and currency both are associated to tag 1, balance and equilibrium are both associated to tag 2, and economics to tag 3.

Our map T is induced by this element-wise mapping.

We formalize this notion as follows.

We call a function t : Ω → U a universe transformation.

Each universe transformation t induces a pushforward multiset transformation T , where the membership function of

Proposition 3.2.3 Every pushforward multiset transformation T preserves cardinalities; that is, for any A, we have |A| = |T (A)|.

Note that if we view A as the measure µ A on Ω, then the measure µ T (A) on U is the honest-togoodness pushforward measure µ A • t −1 .

The above result follows easily.

Let's summarize where we are so far.

We observe some relations between pairs of multisets over Ω. We also assume there is a multiset transformation T from Ω to some "latent" universe U, and that our observed relations are explained by relations that hold in U * .

In general, we wish to understand the structure of T , in order to predict similar relations for unobserved pairs of multisets.

We may also hope that in the process we learn something about the structure of Ω, and that in general this learning process is feasible due to U being much smaller or simpler than Ω. For example, we might assume that T is in fact a pushforward transformation induced by an unobserved labeling of the elements in Ω by elements in U = [k].

However, there is a problem with this setup: it is unlikely that for two multisets A ∈ Ω * and B ∈ Ω * , we have either T (A) ⊆ T (B) or T (B) ⊆ T (A).

However, there are richer relations that can exist between two multisets than just containment, and which can be observed for any such pair.

For example, regardless of whether either of T (A) or T (B) is a subset of the other, we can ask about how much they overlap -i.e.

the size of either their intersection |T (A) ∩ T (B)|, or of their symmetric difference, |T (A) T (B)|.

Note that if we also know their sizes |T (A)| and |T (B)|, then we can know whether one contains the other (Theorem 3.2.6).

This follows directly from the following: (1) that the size of the symmetric difference gives rise to a (pseudo-)metric on multisets, which can then be used to express a "disk embedding" inequality (Suzuki et al., 2019) relating containment to cardinalities (Proposition 3.2.4); and (2), the sizes of the symmetric difference and intersection are related via the sizes of the multisets themselves (Lemma 3.2.5).

See Appendices C and D for proofs; note also that for each of the following three statements, we assume A and B are multisets over the same universe. .

We therefore see that we can use a training signal more readily available than binary yes-no containment -measurements of overlap between multisets -to still learn to predict containment relations.

Thus, the problem we will be solving here is learning to predict either |T (A) T (B)| or |T (A) ∩ T (B)| from examples.

Importantly, the error on these predictions will indicate how well we learned to capture our "flexible" notion of containment.

Formally, our learning task will therefore be as follows.

There exists a universe Ω, which we assume for practical purposes can be embedded in R d for some known d. There is also some latent universe U = [k] together with an unknown multiset transformation T : Ω * → U * .

We will assume that T preserves cardinalities -in practice this means either that T is a pushfoward transformation induced by some t : Ω → U, or an "expectation transformation," which we define in Section 4.2.

We then observe samples (A, B) from a training distribution D over pairs of multisets in Ω * .

For practical reasons, our sampled multisets will wholenumber multiplicities.

For each such pair, we also observe the overlap via either |T (A) T (B)| or |T (A) ∩ T (B)|.

Which of these is used is fixed beforehand for the entire task, and we assume this choice is known.

(We test both choices in our experiments.)

Our assumption that T preserves cardinality is important, because together with these observations, it allows us to conclude whether T (A) is a subset of T (B).

We then pick a hypothesis target universeÛ = [k], and we letk = k if k is known. (Experimentally, we examine the casesk < k andk = k.) Finally, our goal is to learn a model -i.e.

a mapT : Ω * →Û * -that minimizes squared error in the predicted overlaps.

That is, we learnT in order to minimize the appropriate choice of the following two losses:

Having formulated our learning task, we now define our learnable modelT : Ω * →Û * .

In order to do so, we want our model to give us "representations" of the multisets inÛ * in the most common machine-learning sense -i.e.

vectors in some Euclidean space.

We begin this section by defining how we obtain and use such representations, and then conclude by defining our modelT itself that gives us these representations.

In general, we want our representations of multisets to be "useful," in the sense that we can use them to perform common operations -such as those in Definition 3.1.3.

More importantly for our task, we need to be able to calculate the size of either the symmetric difference or the intersection of two multisets.

Our choice of target universeÛ = [k] gives us such a representation function.

Definition 4.1.2 LetÛ be the finite universe [k] .

The natural representation function Ψk :

This should be an intuitive concept.

For example, the natural representation function for classical sets gives the familiar indicator vector representation Ψk(S) = [1 1∈S , . . .

, 1k ∈S ].

Furthermore, we get "usefullness" of these representations for free, since all operations defined via membership functions (e.g. those in Definition 3.1.3) can be performed coordinate-wise.

Furthermore, the cardinality of a multiset S ∈Û * is given by thus sum of the entries in Ψk(S).

Together with the non-negativity of membership functions, this gives us the following.

(As these two results are essentially immediate, we omit their proofs.)

Proposition 4.1.4 For any two multisets R and S over [k], we have |R S| = ||Ψk(R) − Ψk(S)|| 1 and |R ∩ S| = min{Ψk(R), Ψk(S)}, where the minimum is applied coordinate-wise.

We thus use the natural representation function onÛ to train our model.

We note that Proposition 4.1.4 could provide a reason to prefer the size of the symmetric difference over the size of the intersection as the training signal.

The reasoning is that 1 -distance has a gradient which depends on both the representations Ψk(R) and Ψk(S) in each coordinate (except at 0), while the coordinate-wise minimum can only depend on one of the representations in each coordinate.

We test this idea in our experiments.

Recall that the unobserved multiset transformation T : Ω * → U * preserves cardinalities.

In particular, suppose for the purpose of exposition that T is the pushforward transformation induced by some labeling t : Ω → U.

We both want our hypothesis class of modelsT to contain all such pushforward multiset transformations, and to potentially be restricted to thoseT which preserve cardinalities.

Unfortunately, we cannot directly learn over the set of all pushforward transformations, as this is equivalent to learning the correct discrete labeling t : Ω → [k], which is both a hard and nondifferentiable problem.

4 Instead, we take a probabilistic approach.

Definition 4.2.1 For two universes Ω andÛ, a probabilistic universe transformation is a map : Ω → ∆(Û) , where ∆(Û) is the space of probability measures onÛ.

As we will see, probabilistic universe transformations to [k] have the advantage of being smoothly parametrizable.

In analogy to the pushforward multiset transformation induced by a t : Ω → U, we leverage our probabilistic transformation above to define a different kind of induced multiset transformation.

Definition 4.2.2 Let : Ω → ∆(Û) be a probabilistic universe transformation.

The expectation multiset transformation L : Ω * →Û * is defined to be the map

We first note that L is well defined, in the sense that L(A) is always a valid measure onÛ. This can easily be seen by re-writing the expression as follows:

L(A) has a natural interpretation, as the expected multiset obtained by sampling an element ofÛ for each x ∈ A according to its corresponding distribution (x) (with contribution weighted by m A (x)).

Additionally, we have the desirable property that L preserves cardinalities (see Appendix E for proof).

Theorem 4.2.3 Any expectation multiset transformation L : Ω * →Û * preserves cardinalities, i.e. for any A ∈ Ω * , |A| = |L(A)|.

We will thus let our learned modelT : Ω * →Û * be an expectation transformation L induced by some probabilistic universe transformation .

Note that we expect to be able to learn not only in the case where the unknown T is a pushforward transformation, but in fact when T itself is some expectation transformation (although we do not test the latter experimentally.)

The outstanding question is: how do we parametrize our representations Ψk(L(A)) for any given A ∈ Ω * ?

By the definition of the natural representation function Ψk, we can write the i-th component of the representation of L(A) as

where the last two equalities come from viewing (x) as a general measure and thus multiset onÛ. Now, assuming that A is in fact one of the multisets sampled from our training distribution D, we know that A has whole-number multiplicities.

(This will also be the case during evaluation, and thus in fact for any multiset we are trying to represent.)

The above can then by simply written as Ψk(L(A)) i = x∈A Ψk( (x)) i , where each x occurs in the sum m A (x) times.

More simply, we have Ψk(L(A)) = x∈A Ψk( (x)).

Since each (x) is just a distribution overk elements, it suffices to learn a map from Ω to the probability simplex in Rk -the non-negative vectors whose components sum to 1.

Recalling that we assumed our input universe Ω consists of vectors in R d , we pick a favorite object-featurization

We then guarantee than we obtain a point in the probability simplex by taking φ(a) to

||f (φ(a))||1 , where f : R → R + is a function applied component-wise.

For differentiability, we choose the softplus function f (x) = log(1 + e x ).

Our complete model is thus the representation

||f (φ(a))||1 .

Our losses, in terms of these representations, are

We begin here with an overview what we want to test about our model.

In Section 5.1 we move on to describe our training and evaluation procedures.

The experimental results themselves follow.

A clear question to seek the answer to empirically is whether the size of the symmetric difference or of the intersection works better in practice.

(Recall that the symmetric difference may have more informative gradients, possibly leading to better learning and performance.)

We thus compare these two approaches, both in terms of the error on the respective tasks themselves, and in terms of the error on predicting containment.

More generally, the theory motivating our model suggest that there is a delicate balance in the properties that make the model well-posed.

5 We tackle this idea from two directions.

First, we ask how important is the precise definition of our model, Ψ(A) = x∈A f (φ(a)) ||f (φ(a))||1 .

An obvious baseline to compare against is Ψ(A) = x∈A φ(x), which should help us answer the question of how important it is that each object is mapped to a point in the probability simplex.

Looking at this formulation, an immediate connection one might make is to the DeepSets model of Zaheer et al. (2017) : Ψ(A) = ρ 1 ( x∈A φ(x)) for some learnable function ρ 1 .

The authors prove this model can learn any permutation invariant function -e.g.

the size of the intersection or symmetric difference of two multisets.

We thus use both the models above as baselines in all our experiments, calling the former "unrestricted multisets" and the latter "DeepSets"

The second category of question we ask here is whether we gain anything from our construction of the multiset operations on representations.

We tackle this question replacing the terms ||Ψ(A) − Ψ(B)|| 1 and min{Ψ(A)lΨ(B)} in our losses with ρ 2 (Ψ(A)+Ψ(B)) for a learnable function ρ 2 .

The intuition here is that this new prediction is in fact a second DeepSets model trying to learn of our prediction functions -where we choose DeepSets because both intersection and symmetric difference are permutation invariant (i.e. commutative).

This setting will be called the "learned operation" setting.

We further test whether our parametrizations of the multisets operations are somehow intrisically good via a scheme of "cross-wiring" them -using one for a task where we should use the otheron which we elaborate in Section 5.4.

Finally, we will also examine the learned representations φ(x) of elements x ∈ Ω.

We use MNIST (LeCun, 1998) as our dataset.

The training set consists of 60,000 handwritten images of digits, and the test set of 10,000.

We train all the models on 3×10 5 training pairs of multisets (A, B) ∈ Ω. At each iteration of training, both A and B are generated randomly, as follows.

First a whole-number cardinality is uniformly sampled in some chosen range -in our experiments we use [2, 5] .

(We exclude singleton sets to ensure that the models aren't just learning from comparing pairs of singletons.)

Once the cardinality is chosen, then that number of images x ∈ Ω is then chosen uniformly at random (with replacement) from the training set.

The multiset representation is calculated as usual, via one of the representation functions Ψ defined above, and the predicted cardinality of the symmetric difference or intersection is then calculated using these representations.

The value to be predicted is calculated directly from the labels of the images in the multisets -e.g.

if A is two images of ones, and B is a one and a three, the target value will be 1 for intersection, and 2 for symmetric difference.

The squared error is minimized using Adam (Kingma and Ba, 2015) , with the default parameters β 1 = 0.9 and β 2 = 0.999, and a learning rate of 5 × 10 −5 .

The learning rate was chosen by logarithmic grid search from 1 down to 5 × 10 −6 , training on up to 10 4 pairs during the search.

(All models performed best with the chosen learning rate -or at least no worse than any of the other learning rates.)

Given this learning rate, we chose to train the models for 3 × 10 5 iterations, finding that almost all of the models converged by this point.

Evaluation is performed similarly to training, with the addition of multiset sizes uniform on [2, 20] , and with images sampled from the test set.

Importantly, this means that the none of the images seen during training appear during evaluation.

Each model is evaluated on 3 × 10 4 such multiset pairs, and unless otherwise stated we let φ : Ω → R k (that is,ĥ = k = 10).

For the object featurizing function φ, we use a variant of the LeNet-5 neural network (LeCun, 1998) .

Specifically, we adopt the same architecture as used by Ilse et al. (Ilse et al., 2018) .

(See Appendix F for network architectures, including those used for ρ 1 and ρ 2 .)

We examine the performance of six kinds of models -multisets, unrestricted multisets, and DeepSets, each with or without learned multiset operations -on the tasks of predicting cardinality, either of the symmetric difference of the intersection.

We will refer here to Tables 1 (symmetric difference) and 2 (intersection), which report the mean absolute errors of the predictions.

Within each of the tables, two patterns are immediately clear.

First, a amount portion of the prediction error may be explained by whether the multiset operations are learned, or taken to be the theoretically-motivated parametrizations; the models with learned operations exhibit more than twice the prediction error.

While this shows there is a benefit to using our theoretically-motivated definitions, it does not necessarily mean that our definitions are intrinsically or uniquely well-suited for the task.

We will revisit this point later.

The second salient pattern is that as we move away from the expectation-transformation model (which we simply call "multisets" in our tables), first to the unrestricted multiset model, and then to DeepSets, there is a rapid decrease in performance (in some cases almost ten-fold).

This suggests that the theory behind our model is indeed useful.

Finally, when we compare across the two tables -that is, compare cardinality prediction for symmetric difference and for intersection -we observe a surprisingly large gap in error.

The error on intersection size prediction is consistently about twice as small as on the other task.

It is worth noting that if anything, we expected an opposite effect.

This gap is intriguing, and we believe that it should be explored further.

We now compare the same models above on what is perhaps the more important task: prediction whether there exists a containment relation between T (A) and T (B) for some A and B. In particular, for any such pair, we predict whether

T (A), or there is containment relation (i.e. we treat this as a classification problem).

We perform this prediction by relying of Theorem 3.2.6 (and the assumptions on our representations).

In particular, for any pair A and B, we predict the containment relation implied by Proposition 3.2.4, where we take the cardinality of the symmetric difference predicted by our model.

(If the model predicts intersection, we just use Lemma 3.2.5 to go from one to the other.)

Note that in this experiment we sample pairs A and B such that the probability of each kind of containment is essentially uniform.

Referring to Tables 3  and 4 , we observe almost exactly the same patters as above -with over 96% accuracy achieved by both multiset models.

The one difference is that for the unnormalized model and the DeepSets model (with non-learned operations), the version of the models trained on the intersection task perform noticeably worse than the corresponding models trained on symmetric difference.

Furthermore, on all other models, the performances are comparable.

This further complicates the picture from above, as it suggests that while intersection may somehow be easier to learn to predict the cardinality of, perhaps the task itself is a worse way to capture containment relations.

Motivated by our observation that models less closely aligned with our theory seems to perform worse, we devise a small test to see whether our symmetric difference and intersection cardinality operations are somehow intrinsic.

To do so, we perform two experiments.

First, we train our regular expectationtransformation-based multiset model to predict symmetric difference cardinality, but where it's prediction function is given by min{Ψ(A), Ψ(B)}. Similarly, in the second experiment, we train the model to predict intersection cardinality, but where the prediction function is ||Ψ(A) − Ψ(B)|| 1 .

We observe that the former model achieves and mean-absolute error of 2.3710 on the test set, while the second achieves ones of 0.9929.

There values are significantly higher than the errors achieved with the "correct" prediction functions, suggesting that there indeed a sense in which these are the "right" functions.

We finally turn to examining the object-representations learned by our model.

As one would expect, the learned representations of objects are approximate the standard basis vectors (as shown in Figure 1 for n = 3).

This suggests our expectation-transformation model is learning appropriate point-mass probabilities corresponding to each object's label in U.

We also examine the casek < k, which may occur when we don't know the true size of U. Here, the "pinched" nature of the restricted representations may be undesirable (Figure 2a ).

This problem, of course, gets worse with the discrepancy between number of objects and dimension (Figure 2b) .

On the other hand, the unrestricted multiset model is able to learn more balanced-looking clusters.

However, the clusters for d = n appear slightly less well-separated (Figures 3 and 4) .

The DeepSets model didn't learn interpretable representations (Figure 4c) .

Furthermore, when we measure the accuracy of the regular multiset model on the containment prediction task above of each of the models, we obtain good results even whenk is "too small."

In particular, fixing k = 5:k = 5 we obtain an accuracy of 0.9586, andk = 3 gives 0.9045.

This suggest that the representations learned are in fact robust to small discrepancies in dimension.

Figure 1 : Three-dimensional representations of test-set MNIST images generated by the restricted multiset model trained on multisets of sizes ∈ [2, 5] ; the model is trained on images of zeros, ones, twos.

We propose a novel task: predicting the size of either the symmetric difference of the intersection between pairs of multisets.

We motivate this construction via a measure-theoretic notion of "flexible containment."

We demonstrate the utility of this idea, developing a theoretically-motivated model that given only the sizes of symmetric differences between pairs of multisets, learns representations of such multisets and their elements.

These representations allow us to predict containment relations with extremely high accuracy.

Our model learns to map each type of object to a standard basis vector, thus essentially performing semi-supervised clustering.

One interesting area for future theoretical work is understanding a related problem: clustering n objects given multiset difference sizes.

As a first step, we show in Appendix H that n − 1 specific multiset comparisons are sufficient to recover the clusters.

We would also be curious to see if one can learn the latent multiset space U. Following similar reasoning, we can convince ourselves that multiset union should be defined as

It is important to differentiate this from "multiset addition," which simply combines two multisets directly: A + B = {1, 1, 1, 1, 1, 2, 2, 2, 3} for our example above, and in general m A+B = m A (x) + m B (x).

Multiset difference is a little harder to define.

The main problem is that we cannot rely on a notion of "complement" for multisets.

Instead, let us again try to reason by example.

For our example multisets above, we have A \ B = {1, 2}. To arrive at this result, we remove from A each copy of an element which also appears in B. Note that if B had more of a certain element than A, that element would not appear in the final result.

In other words, we are performing a subtraction of counts which is "glued" to a minimum value of zero.

That is, m A\B (x) = max{m A (x) − m B (x), 0}. We can further convince ourselves of the correctness of this expression by noting that we recover the identity

Finally, symmetric multiset difference can be defined using our expression for multiset difference, combined with either multiset addition or union.

In particular, note that A B = (A\B)+(B \A) = (A \ B) ∪ (B \ A) -addition and union both work because (A \ B) and (B \ A) are necessarily disjoint.

This gives us:

(The equation still holds if we replace the addition with a maximum.)

A fuzzy set A over a universe Ω is given by a function m A : Ω → [0, 1].

Intuitively, m A maps each x ∈ Ω to "how much of a member" x is of A, on a scale from 0 to 1.

With this simple idea, fuzzy set operations can be defined.

This is traditionally done by leveraging element-wise fuzzy logical operations, which we define below.

1] , satisfying the following properties:

• 1 is the identity: T (a, 1) = a T-norms generalize the notion of conjunction.

Note that the above conditions imply that for any a, T (a, 0) = 0, and that T (1, 1) = 1.

These two observations show that t-norms are "compatible" with classical, non-fuzzy logic -where we identify 0 with "false" and 1 with "true."

The standard t-norm is T (a, b) = min{a, b}.

Definition B.0.2 A strong negator is a strictly monotonic, decreasing function n :

Unsurprisingly, strong negators generalize logical negation.

The standard strong negator is n(x) = 1 − x.

Definition B.0.3 An S-norm (also called a t-conorm) is a function with the same properties as a t-norm, except that the identity element is 0.

S-norms generalize disjunction.

For every t-norm (and a given negator), we can define a complementary s-norm: S(a, b) = n(T (n(a), n(b))).

This is a generalization of De Morgan's laws.

The standard s-norm, complementary to the min t-norm, is S(a, b) = max{a, b}.

The membership function for the intersection of two fuzzy sets A and B is naturally defined as µ A∩B (x) = T (µ A (x), µ B (x)) for a t-norm T .

Similarly, the complement of a fuzzy set is given by µ A (x) = n(µ A (x)) for a strong negator n, and the union of two fuzzy sets is given by µ A∪B (x) = S(µ A (x), µ B (x)) for an s-norm S. Usually, we want T and S to be complementary with respect to n. Then, we can generalize all the usual set operations to fuzzy sets by combining the three basic operations above.

C PROOF OF PROPOSITION 3.2.4 We show that for any two multisets A and B over the same universe Ω, A ⊆ B if and only if |A B| ≤ |B| − |A|.

In fact, noting that it is always the case that |A B| ≥ |B| − |A| (which we will not prove but is easy to show), the following proof shows this holds with equality.

Proof.

Let λ be the dominating measure with respect to which the cardinalities are taken.

We first show the forward direction.

Suppose A ⊆ B, that is, for every x ∈ Ω, we have m A (x) ≤ m B (x).

The result follows directly (with equality):

For the converse direction, suppose on the other hand that |A B| ≤ |B| − |A|.

Now suppose for the sake of contradiction that for some

which is a contradiction.

D PROOF OF LEMMA 3.2.5

We show that for any two multisets A and B over the same universe Ω, |A B| = |A|+|B|−2|A∩B|.

Proof.

Let λ be the dominating measure with respect to which the cardinalities are taken.

Let Ω A ⊆ Ω be the elements x ∈ Ω on which m A (x) > m B (x), and similarly let Ω B ⊆ Ω be the elements x ∈ Ω on which m B (x) > m A (x).

Finally let Ω 0 be those elements x ∈ Ω for which m A (x) = m B (x).

Note that Ω A , Ω B , and Ω 0 are disjoint, and that their union is the entire universe Ω. We can then write the cardinality of the intersection of A and B as

We then observe that

Simplifying, we obtain

But this is exactly the cardinality of the symmetric difference |A B|.

We will show that for any probabilistic universe transformation : Ω → ∆(U), the induced expectation transformation L : Ω * → U * preserves cardinalities.

That is, |A| = |L(A)|.

(For easy of notation, we write U here rather than theÛ used in statement of the result in the main text.)

Proof.

Recall that since L(A) is a multiset and thus a measure, we have

where the second equality holds because (x) is a probability measure over U.

Given input images with c channels, and an output dimension d, the function φ is parametrized by the network: Figure 2: Three-dimensional representations of test-set MNIST images generated by the restricted multiset model trained on multisets of sizes ∈ [2, 5] .

Note that in (b) the representations of twos and threes are essentially inseparable.

H CLUSTERING n OBJECTS GIVEN n − 1 SYMMETRIC SET DIFFERENCE SIZES

We are interested in the following problem.

Suppose we have a set of n objects U, each of which belongs to one of k clusters, C 1 , . . .

, C k .

Let M : 2 U → {1, . . .

, k} * be the function which takes any subset of U, and gives the multiset of cluster labels represented in that subset.

We are given oracle access to the function ∆ : 2 U × 2 U → N which gives the size of the symmetric set difference between the cluster-label multisets: ∆(A, B) = |M (A) M (B)|.

How many queries are required to determine the clusters C 1 , . . .

, C k (up to permutation)?

We show that the clusters can be determined with n − 1 specific queries.

(Another way to think of this is as a training data problem, rather than an oracle querying problem; we show n − 1 training examples can be sufficient.)

We do this in two steps.

The step lets us identify k disjoint subsets of U, such that no two of these subsets contain objects from the same cluster.

The second step confirms that these subsets are in fact the clusters C 1 , . . .

, C k .

Three-dimensional representations of test-set MNIST images generated by the unrestricted multiset model trained on multisets of sizes ∈ [2, 5] .

Note that in (c), the clusters essentially form a tetrahedron, with one of the vertices being the combination of twos and threes.

The first step consists of logarithmically "splitting" U. The very first query in this step is ∆ k/2 i=1 C i , k i= k/2 C i , which tells us that k/2 i=1 C i and k i= k/2 C i are disjoint in terms of represented clusters.

We proceed recursively, each query "splitting" the sets in half (in terms of which clusters they contain).

The number of such steps required is k − 1 (which is the number of internal nodes in a balanced binary search tree for k objects).

We'll call the resulting disjoint setsC 1 , . . .

,C k (since we technically don't yet know they correspond to the true clusters).

For the second step, we must verify that the objects in each of our sets resulting from step one all belong to the same cluster.

This can be done by ordering the objects within each set, and comparing each consecutive pair as singletons.

For each of our setsC i , we thus make |C i | − 1 such queries.

Across all such sets, we thus make k i=1 |C i | − 1 = n − k queries.

So, the total number of queries made is (k − 1) + (n − k) = n − 1.

<|TLDR|>

@highlight

Based on fuzzy set theory, we propose a model that given only the sizes of symmetric differences between pairs of multisets, learns representations of such multisets and their elements.

@highlight

This paper proposes a new task of set learning, predicting the size of the symmetric difference between multisets, and gives a method to solve the task based on fuzzy set theory.