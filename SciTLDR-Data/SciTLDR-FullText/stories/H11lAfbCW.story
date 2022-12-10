The learnability of different neural architectures can be characterized directly by computable measures of data complexity.

In this paper, we reframe the problem of architecture selection as understanding how data determines the most expressive and generalizable architectures suited to that data, beyond inductive bias.

After suggesting algebraic topology as a measure for data complexity, we show that the power of a network to express the topological complexity of a dataset in its decision boundary is a strictly limiting factor in its ability to generalize.

We then provide the first empirical characterization of the topological capacity of neural networks.

Our empirical analysis shows that at every level of dataset complexity, neural networks exhibit topological phase transitions and stratification.

This observation allowed us to connect existing theory to empirically driven conjectures on the choice of architectures for a single hidden layer neural networks.

Deep learning has rapidly become one of the most pervasively applied techniques in machine learning.

From computer vision BID15 ) and reinforcement learning BID18 ) to natural language processing BID27 ) and speech recognition ), the core principles of hierarchical representation and optimization central to deep learning have revolutionized the state of the art; see BID10 .

In each domain, a major difficulty lies in selecting the architectures of models that most optimally take advantage of structure in the data.

In computer vision, for example, a large body of work BID24 , BID25 , BID12 , etc.) focuses on improving the initial architectural choices of BID15 by developing novel network topologies and optimization schemes specific to vision tasks.

Despite the success of this approach, there are still not general principles for choosing architectures in arbitrary settings, and in order for deep learning to scale efficiently to new problems and domains without expert architecture designers, the problem of architecture selection must be better understood.

Theoretically, substantial analysis has explored how various properties of neural networks, (eg.

the depth, width, and connectivity) relate to their expressivity and generalization capability , BID6 , BID11 ).

However, the foregoing theory can only be used to determine an architecture in practice if it is understood how expressive a model need be in order to solve a problem.

On the other hand, neural architecture search (NAS) views architecture selection as a compositional hyperparameter search BID23 , BID9 , BID31 ).

As a result NAS ideally yields expressive and powerful architectures, but it is often difficult to interperate the resulting architectures beyond justifying their use from their emperical optimality.

We propose a third alternative to the foregoing: data-first architecture selection.

In practice, experts design architectures with some inductive bias about the data, and more generally, like any hyperparameter selection problem, the most expressive neural architectures for learning on a particular dataset are solely determined by the nature of the true data distribution.

Therefore, architecture selection can be rephrased as follows: given a learning problem (some dataset), which architectures are suitably regularized and expressive enough to learn and generalize on that problem?A natural approach to this question is to develop some objective measure of data complexity, and then characterize neural architectures by their ability to learn subject to that complexity.

Then given some new dataset, the problem of architecture selection is distilled to computing the data complexity and chosing the appropriate architecture.

For example, take the two datasets D 1 and D 2 given in FIG0 (ab) and FIG0 (cd) respectively.

The first dataset, D 1 , consists of positive examples sampled from two disks and negative examples from their compliment.

On the right, dataset D 2 consists of positive points sampled from two disks and two rings with hollow centers.

Under some geometric measure of complexity D 2 appears more 'complicated' than D 1 because it contains more holes and clusters.

As one trains single layer neural networks of increasing hidden dimension on both datasets, the minimum number of hidden units required to achieve zero testing error is ordered according to this geometric complexity.

Visually in FIG0 , regardless of initialization no single hidden layer neural network with ≤ 12 units, denoted h ≤12 , can express the two holes and clusters in D 2 .

Whereas on the simpler D 1 , both h 12 and h 26 can express the decision boundary perfectly.

Returning to architecture selection, one wonders if this characterization can be extrapolated; that is, is it true that for datasets with 'similar' geometric complexity to D 1 , any architecture with ≥ 12 hidden learns perfectly, and likewise for those datasets similar in complexity to D 2 , architectures with ≤ 12 hidden units can never learn to completion?

In this paper, we formalize the above of geometric complexity in the language of algebraic topology.

We show that questions of architecture selection can be answered by understanding the 'topological capacity' of different neural networks.

In particular, a geometric complexity measure, called persistent homology, characterizes the capacity of neural architectures in direct relation to their ability to generalize on data.

Using persistent homology, we develop a method which gives the first empirical insight into the learnability of different architectures as data complexity increases.

In addition, our method allows us to generate conjectures which tighten known theoretical bounds on the expressivity of neural networks.

Finally, we show that topological characterizations of architectures are possible and useful for architecture selection in practice by computing the persistent homology of CIFAR-10 and several UCI datasets.

In order to more formally describe notions of geometric complexity in datasets, we will turn to the language of topology.

Broadly speaking, topology is a branch of mathematics that deals with characterizing shapes, spaces, and sets by their connectivity.

In the context of characterizing neural networks, we will work towards defining the topological complexity of a dataset in terms of how that dataset is 'connected', and then group neural networks by their capacity to produce decision regions of the same connectivity.

BID14 ).In topology, one understands the relationships between two different spaces of points by the continuous maps between them.

Informally, we say that two topological spaces A and B are equivalent (A ∼ = B) if there is a continuous function f : A → B that has an inverse f −1 that is also continuous.

When f exists, we say that A and B are homeomorphic and f is their homeomorphism; for a more detailed treatment of general topology see BID2 .

Take for example, the classic example of the coffee cup and the donut in FIG1 .

They are homeomorphic because one can define a continuous deformation of one into the other which shrinks, twists, and morphs without tearing or gluing, as in FIG1 .

Note that if the donut had two holes, it would no longer be equivalent to the mug.

Likewise, in an informal way, FIG0 since if there were a homeomorphism f : D 1 → D 2 at least one of the clusters in D 1 would need to be split in order to produce the four different regions in D 2 .

DISPLAYFORM0 The power of topology lies in its capacity to differentiate sets (topological spaces) in a meaningful geometric way that discards certain irrelevant properties such as rotation, translation, curvature, etc.

For the purposes of defining geometric complexity, non-topological properties 1 like curvature would further fine-tune architecture selection-say if D 2 had the same regions but with squigly (differentially complex) boundaries, certain architectures might not converge-but as we will show, grouping neural networks by 'topological capacity' provides a powerful minimality condition.

That is, we will show that if a certain architecture is incapable of expressing a decision region that is equivalent in topology to training data, then there is no hope of it ever generalizing to the true data.

Algebraic topology provides the tools necessary to not only build the foregoing notion of topological equivalence into a measure of geometric complexity, but also to compute that measure on real data (Betti (1872) , BID7 BID2 ).

At its core, algebraic topology takes topological spaces (shapes and sets with certain properties) and assigns them algebraic objects such as groups, chains, and other more exotic constructs.

In doing so, two spaces can be shown to be topologically equivalent (or distinct) if the algebraic objects to which they are assigned are isomorphic (or not).

Thus algebraic topology will allow us to compare the complexity of decision boundaries and datasets by the objects to which they are assigned.

Although there are many flavors of algebraic topology, a powerful and computationally realizable tool is homology.

Definition 2.1 (Informal, Bredon (2013) ).

If X is a topological space, then H n (X) = Z βn is called the nth homology group of X if the power β n is the number of 'holes' of dimension n in X. Note that β 0 is the number of separate connected components.

We call β n (X) the nth Betti number of X. Finally, the homology 2 of X is defined as DISPLAYFORM0 Immediately homology brings us closer to defining the complexity of D 1 and D 2 .

If we assume that D 1 is not actually a collection of N datapoints, but really the union of 2 solid balls, and likewise that D 2 is the union of 2 solid balls and 2 rings, then we can compute the homology directly.

In this case H 0 (D 1 ) = Z 2 since there are two connected components 3 ; H 1 (D 1 ) = {0} since there are no circles (one-dimensional holes); and clearly, H n (D 1 ) = {0} for n ≥ 2.

Performing the same computation in the second case, we get H 0 (D 2 ) = Z 4 and H 1 (D 2 ) = Z 2 as there are 4 seperate clusters and 2 rings/holes.

With respect to any reasonable ordering on homology, D 2 is more complex than D 1 .

The measure yields non-trivial differentiation of spaces in higher dimension.

For example, the homology of a hollow donut is DISPLAYFORM1 Surprisingly, the homology of a space contains a great deal of information about its topological complexity 1 .

The following theorem suggests the absolute power of homology to group topologically similar spaces, and therefore neural networks with topologically similar decision regions.

DISPLAYFORM2 Intuitively, Theorem 2.2 states that number of 'holes' (and in the case of H 0 (X), connected components) are topologically invariant, and can be used to show that two shapes (or decision regions) are different.

In order to compute the homology of both D 1 and D 2 we needed to assume that they were actually the geometric shapes from which they were sampled.

Without such assumptions, for any dataset DISPLAYFORM0 where N is the number of data points.

This is because, at small enough scales each data point can be isolated as its own connected component; that is, as sets each pair of different positive points d 1 , d 2 ∈ D are disjoint.

To properly utilize homological complexity in better understanding architecture selection, we need to be able to compute the homology of the data directly and still capture meaningful topological information.

Persistent homology, introduced in BID30 , avoids the trivialization of computation of dataset homology by providing an algorithm to calculate the homology of a filtration of a space.

Specifically, a filtration is a topological space X equipped with a sequence of subspaces X 0 ⊂ X 1 ⊂ · · · ⊂ X. In FIG2 one such particular filtration is given by growing balls of size centered at each point, and then letting X be the resulting subspace in the filtration.

Define β n (X) to be the nth Betti number of the homology H(X ) of X .

Then for example at = 1.5, β 0 (X ) = 19 and β 1 (X ) = 0 as every ball is disjoint.

At = 5.0 some connected components merge and β 0 (X ) = 12 and β 1 (X ) = 0.

Finally at = 7, the union of the balls forms a hole towards the center of the dataset and β 1 (X ) > 0 with β 0 (X ) = 4.All together the change in homology and therefore Betti numbers for X as changes can be summarized succinctly in the persistence barcode diagram given in FIG2 .

Each bar in the section β n (X) denotes a 'hole' of dimension n.

The left endpoint of the bar is the point at which homology detects that particular component, and the right endpoint is when that component becomes indistinguishable in the filtration.

When calculating the persistent homology of datasets we will frequently use these diagrams.

With the foregoing algorithms established, we are now equipped with the tools to study the capacity of neural networks in the language of algebraic topology.

In the forthcoming section, we will apply persistent homology to emperically characterize the power of certain neural architectures.

To understand why homological complexity is a powerful measure for differentiating architectures, we present the following principle.

Suppose that D is some dataset drawn from a joint distribution F with continuous CDF on some topological space X × {0, 1}. Let X + denote the support of the distribution of points with positive labels, and X − denote that of the points with negative labels.

Then let H S (f ) := H[f −1 ((0, ∞))] denote the support homology of some function f : X → {0, 1}. Essentially H S (f ) is homology of the set of x such that f (x) > 0.

For a binary classifier, f , H S (f ) is roughly a characterization of how many 'holes' are in the positive decision region of f .

We will sometimes use β n (f ) to denote the nth Betti number of this support homology.

Finally let F = {f : X → {0, 1}} be some family of binary classifiers on X. Theorem 3.1 (The Homological Principle of Generalization).

If X = X − X + and for all f ∈ F with H S (f ) = H(X + ), then for all f ∈ F there exists A ⊂ X + so f misclassifies every x ∈ A.Essential Theorem 3.1 says that if an architecture (a family of models F) is incapable of producing a certain homological complexity, then for any model using that architecture there will always be a set A of true data points on which the model will fail.

Note that the above principle holds regardless of how f ∈ F is attained, learned or otherwise.

However, the principle does imply that no matter how well some F learns to correctly classify D there will always be a counter examples in the true data.

In the context of architecture selection, the foregoing minimality condition significantly reduces the size of the search space by eliminating smaller architectures which cannot even express the 'holes' (persistent homology) of the data H(D).

This allows us to return to our original question of finding suitably expressive and generalizeable architectures but in the very computable language of homological complexity: Let F A the set of all neural networks with 'architecture' A, then Given a dataset D, for which architectures A does there exist a neural network f ∈ F A such that H S (f ) = H(D)?

We will resurface a contemporary theoretical view on this question, and thereafter make the first steps towards an emperical characterization of the capacity of neural architectures in the view of topology.

Theoretically, the homological complexity of neural network can be framed in terms of the sum of the number of holes expressible by certain architectures.

In particular, BID1 gives an analysis of how the maximum sum of Betti numbers grows as F A changes.

The results, summarized in TAB0 , show that the width, depth, and activation of a fully connected architecture effect its topological expressivity to varying polynomial and exponential degrees.

What is unclear from this analysis is how these bounds describe expressivity in terms of individual Betti numbers.

For example, with a tanh activation function, n inputs, layers, and h hidden units, there is no description of what the number of connected components max f ∈F A β 0 (f ) or 1-dimensional holes max f ∈F A β 1 (f ) actually is.

With regards to tighter bounds BID1 stipulate that improvements to their results are deeply tied to several unsolved problems in algebraic topology.

BID1 .) DISPLAYFORM0 Figure 4: Topological phase transitions in low dimensional neural networks as the homological complexity of the data increases.

The upper right corner of each plot is a dataset on which the neural networks of increasing hidden dimension are trained.

To understand how the homology of data determines expressive architectures we turn to an empirical characterization of neural networks.

In this setting, we can tighten the bounds given in TAB0 by training different architectures on datasets with known homologies and then recording the decision regions observed over the course of training.

In the most basic case, one is interested in studying how the number of hidden units in a single hidden layer neural network affect its homological capacity.

The results of BID1 say for certain activation functions we should expect a polynomial dependence on the sum of Betti numbers β n , but is this true of individual numbers?

Having an individual characterization would allow for architecture selection by computing the homology of the dataset, and then finding which architectures meet the minimal criterion for each Betti number β n .

Restricting 5 our analysis to the case of two inputs, n = 2, we characterize the capacities of architectures with an increasing number of hidden units to learn on datasets with homological complexities ranging from {Z 1 , 0} to {Z 20 , Z 20 }.

In our experiment, we generate datasets of each particular homological complexity by sampling different combinations of balls, rings, and scaling, twisting, and gluing them at random.

After generating the foregoing datasets with N ≈ 90000 samples we train 100 randomly (truncated normal) initialized single hidden layer architectures with hidden units h ∈ {1, . . .

, 255} and tanh activation functions for 10 6 minibatches of size 128.

During training, every 2500 batches we sample the decision boundary of each neural network over a grid of 500 × 500 samples, producing 1.02 × 10 6 recorded decision boundaries.

Using the resulting data, we not only characterize different architectures but observed interesting topological phenomena during learning.

First, neural networks exhibit a statistically significant topological phase transition in their convergence which depends directly on the homological complexity of the data.

For any dataset in the experiment and any random homeomorphism applied thereto, the best test error of architectures with h hidden units is strictly ordered in magnitude and convergence time for h < h phase where h phase is a number of hidden units required to express the homology of the data.

In Figure 4 we plot the best performing test error of architectures h ∈ {1, . . .

, 15} on some example datasets DISPLAYFORM0 In this example h phase (D 0 ) = 4, h phase (D 1 ) = 6, and h phase (D 2 ) = 10.

Surprisingly, leading up to the phase transition point, each different architecture falls into its own band of optimal convergence.

This suggests that additional hidden units do in fact add to the topological capacity of an architecture in a consistent way.

Using topological phase transitions we now return to the original question of existence of expressive architectures.

In FIG3 , we accumulate the probabilities that neural networks of varying hidden dimension train to zero-error on datasets of different homological complexities.

The table gives different views into how expressive an architecture need be in order to converge, and therefore we are able to conjecture tighter bounds on the capacity of hidden units.

Extrapolating from the first view, if H 0 (D) = Z m then there exists a single hidden layer neural network with h = m + 2 that converges to zero error on D. Likewise we claim that if H 0 (D) = Z m and H 1 (D) = 1 then the same holds with h ≥ 3m − 1.

Further empirical analysis of convergence probabilities yields additional conjectures.

However, claiming converse conjectures about a failure to generalize in the view of Theorem 3.1 requires exhaustive computation of decision boundary homologies.

By applying persistent homology to the decision boundaries of certain networks during training, we observe that given sufficient data, neural networks exhibit topological stratification.

For example, consider the homologies of different architecture decision regions as training progresses in FIG4 .

At the beginning of training every model captures the global topological information of the dataset and is homologically correlated with one another.

However as training continues, the architectures stratify into two groups with homological complexities ordered by the capacities of the models.

In this example, h 3 , h 4 , and h 5 are unable to express as many holes as the other architectures and so never specialize to more complex and local topological properties of the data.

FIG4 (b) depicts topological stratification in terms of the correlation between Betti numbers.

Topologically speaking, networks with less than 6 hidden units are distinct from those with more for most of training.

Furthermore, this correlative view shows that stratification is consistent with topological phase transition; that is, across all decision boundary homologies recorded during the experiment stratification occurs just when the number of hidden units is slightly less than h phase .

We have thus far demonstrated the discriminatory power of homological complexity in determining the expressivity of architectures.

However, for homological complexity to have any practical use in architecture selection, it must be computable on real data, and more generally real data must have non-trivial homology; if all data were topologically simple our characterization would have no predictive power.

In the following section we will compute the persistent homologies up to dimension 2 of different real world datasets.

CIFAR-10.

We compute the persistent homology of several classes of CIFAR-10 using the Python library Dionysus.

Currently algorithms for persistent homology do not deal well with high dimensional data, so we embed the entire dataset in R 3 using local linear embedding (LLE; Saul & Roweis (2000) ) with K = 120 neighbors.

After embedding the dataset, we take a sample of 1000 points from example class 'car' and build a persistent filtration by constructing a Vietoris-Rips complex on the data.

The resulting complex has 20833750 simplices and took 4.3 min.

to generate.

UCI Datasets.

We further compute the homology of three low dimensional UCI datasets and attempt to assert the of non-trivial , h phase .

Specifically, we compute the persistent homology of the majority classes in the Yeast Protein Localization Sites, UCI Ecoli Protein Localization Sites, and HTRU2 datasets.

For these datasets no dimensionality reduction was used.

In FIG5 (left), the persistence barcode exhibits two seperate significant loops (holes) at ∈ [0.19, 0.31] and ∈ [0.76, 0.85] , as well as two major connected components in β 0 (D).

The Other persistence diagrams are relegated to the appendix.

Existing Data.

Outside of the primary machine learning literature, topological data analysis yields non-trivial computations in wide variety of fields and datasets.

Of particular interest is the work of BID3 , which computes the homological complexity of collections of n × n patches of natural images.

Even in these simple collections of images, the authors found topologies of Klein Bottles (H(·) = {Z, Z 2 /2Z, 0 . . . }) and other exotic topological objects.

Other authors have calculated non-trivial dataset homologies in biological BID26 ), natural language BID17 ), and other domains BID28 , BID29 ).

We will place this work in the context of deep learning theory as it relates to expressivity.

Since the seminal work of BID5 which established standard universal approximation results for neural networks, many theorists have endeavored to understand the expressivity of certain neural architectures.

BID19 BID16 provided the first analysis relating the depth and width of architectures to the complexitity of the sublevel sets they can express.

Motivated therefrom, BID1 expressed this theme in the language of Pfefferian functions, thereby bounding the sum of Betti numbers expressed by sublevel sets.

Finally Guss (2016) gave an account of how topological assumptions on the input data lead to optimally expressive architectures.

In parallel, BID8 presented the first analytical minimality result in expressivity theory; that is, the authors show that there are simple functions that cannot be expressed by two layer neural networks with out exponential dependence on input dimension.

This work spurred the work ofPoole et al. (2016) , which reframed expressivity in a differential geometric lense.

Our work presents the first method to derive expressivity results empirically.

Our topological viewpoint sits dually with its differential geometric counterpart, and in conjunction with the work of and BID1 , this duallity implies that when topological expression is not possible, exponential differential expressivity allows networks to bypass homological constaints at the cost of adversarial sets.

Furthermore, our work opens a practical connectio nbetween the foregoing theory on neural expressivity and architecture selection, with the potential to drastically improve neural architecture search BID31 ) by directly computing the capacities of different architectures.

Architectural power is deeply related to the algebraic topology of decision boundaries.

In this work we distilled neural network expressivity into an empirical question of the generalization capabilities of architectures with respect to the homological complexity of learning problems.

This view allowed us to provide an empirical method for developing tighter characterizations on the the capacity of different architectures in addition to a principled approach to guiding architecture selection by computation of persistent homology on real data.

There are several potential avenues of future research in using homological complexity to better understand neural architectures.

First, a full characterization of neural networks with many layers or convolutional linearities is a crucial next step.

Our empirical results suggest that the their are exact formulas describing the of power of neural networks to express decision boundaries with certain properties.

Future theoretical work in determining these forms would significantly increase the efficiency and power of neural architecture search, constraining the search space by the persistent homology of the data.

Additionally, we intend on studying how the topological complexity of data changes as it is propagated through deeper architectures.

A.1 HOMOLOGY Homology is naturally described using the language of category theory.

Let T op 2 denote the category of topological spaces and Ab the category of abelian groups.

A → X and j : X → (X, A) the sequence sequence of inclusions and connecting homomorphisms are exact.3.

Given the pair (X, A) and an open set U ⊂ X such that cl(U ) ⊂ int(A) then the inclusion k : (X − U, A − U ) → (X, A) induces an isomorphism k * : H * (X − U, A − U ) → H * (X, A)4.

For a one point space P, H i (P ) = 0 for all i = 0.5.

For a topological sum X = + α X α the homomorphism DISPLAYFORM0 is an isomorphism, where i α : X α → X is the inclusion.

For related definitions and requisite notions we refer the reader to Bredon (2013).A.2 PROOF OF THEOREM 3.1 Theorem A.2.

Let X be a topological space and X + be some open subspace.

If F ⊂ 2 X such that f ∈ F implies H S (f ) = H(X + ), then for all f ∈ F there exists A ⊂ X so that f (A ∩ X + ) = {0} and f (A ∩ (X \ X + )) = {1}.Proof.

Suppose the for the sake of contraiction that for all f ∈ F, H S (f ) = H(X + ) and yet there exists an f such that for all A ⊂ X, there exists an x ∈ A such that f (x) = 1.

Then take A = {x} x∈X , and note that f maps each singleton into its proper partition on X. We have that for any open subset of V ⊂ X + , f (V ) = {1}, and for any closed subset W ⊂ X \ X + , f (W ) = {0}. Therefore X + = A∈τ X + ∩X A ⊂ supp(f ) as the subspace topology τ X + ∩X = τ X + ∩ τ X where τ X + = {A ∈ τ X | A ⊂ X + } and τ X denotes the topology of X. Likewise, int(X − ) ⊂ X \supp(F ) under the same logic.

Therefore supp(f ) has the exact same topology as X + and so by Theorem 2.2 H(X + ) = H(supp(f )) but this is a contradiction.

This completes the proof.

@highlight

We show that the learnability of different neural architectures can be characterized directly by computable measures of data complexity.