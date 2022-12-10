In this paper, we study the representational power of deep neural networks (DNN) that belong to the family of piecewise-linear (PWL) functions, based on PWL activation units such as rectifier or maxout.

We investigate the complexity of such networks by studying the number of linear regions of the PWL function.

Typically, a PWL function from a DNN can be seen as a large family of linear functions acting on millions of such regions.

We directly build upon the work of Mont´ufar et al. (2014), Mont´ufar (2017), and Raghu et al. (2017) by refining the upper and lower bounds on the number of linear regions for rectified and maxout networks.

In addition to achieving tighter bounds, we also develop a novel method to perform exact numeration or counting of the number of linear regions with a mixed-integer linear formulation that maps the input space to output.

We use this new capability to visualize how the number of linear regions change while training DNNs.

We have witnessed an unprecedented success of deep learning algorithms in computer vision, speech, and other domains (Krizhevsky et al., 2012; Ciresan et al., 2012; Goodfellow et al., 2013; Hinton et al., 2012) .

While the popular deep learning architectures such as AlexNet (Krizhevsky et al., 2012) , GoogleNet (Szegedy et al., 2015) , and residual networks (He et al., 2016) have shown record beating performance on various image recognition tasks, empirical results still govern the design of network architecture in terms of depth and activation functions.

Two important practical considerations that are part of most successful architectures are greater depth and the use of PWL activation functions such as rectified linear units (ReLUs).

Due to the large gap between theory and practice, many researchers have been looking at the theoretical modeling of the representational power of DNNs (Cybenko, 1989; BID0 Pascanu et al., 2014; Montúfar et al., 2014; BID4 Eldan & Shamir, 2016; Telgarsky, 2015; Mhaskar et al., 2016; Raghu et al., 2017; Montúfar, 2017) .Any continuous function can be approximated to arbitrary accuracy using a single hidden layer of sigmoid activation functions (Cybenko, 1989 ).

This does not imply that shallow networks are sufficient to model all problems in practice.

Typically, shallow networks require exponentially more number of neurons to model functions that can be modeled using much fewer activation functions in deeper ones (Delalleau & Bengio, 2011) .

There have been a wide variety of activation functions such as threshold (f (z) = (z > 0)), logistic (f (z) = 1/(1 + exp(−e))), hyperbolic tangent (f (z) = tanh(z)), rectified linear units (ReLUs f (z) = max{0, z}), and maxouts (f (z 1 , z 2 , . . .

, z k ) = max{z 1 , z 2 , . . .

, z k }).

The activation functions offer different modeling capabilities.

For example, sigmoid networks are shown to be more expressive than similar-sized threshold networks (Maass et al., 1994) .

It was recently shown that ReLUs are more expressive than similar-sized threshold networks by deriving transformations from one network to another (Pan & Srikumar, 2016) .The complexity of neural networks belonging to the family of PWL functions can be analyzed by looking at how the network can partition the input space to an exponential number of linear response regions (Pascanu et al., 2014; Montúfar et al., 2014) .

The basic idea of a PWL function is simple: we can divide the input space into several regions and we have individual linear functions for each of these regions.

Functions partitioning the input space to a larger number of linear regions are considered to be more complex ones, or in other words, possess better representational power.

In the case of ReLUs, it was shown that deep networks separate their input space into exponentially more linear response regions than their shallow counterparts despite using the same number of activation functions (Pascanu et al., 2014) .

The results were later extended and improved (Montúfar et al., 2014; Raghu et al., 2017; Montúfar, 2017; BID1 .

In particular, Montúfar et al. (2014) shows both upper and lower bounds on the maximal number of linear regions for a ReLU DNN and a single layer maxout network, and a lower bound for a maxout DNN.

Furthermore, Raghu et al. (2017) and Montúfar (2017) improve the upper bound for a ReLU DNN.

This upper bound asymptotically matches the lower bound from Montúfar et al. (2014) when the number of layers and input dimension are constant and all layers have the same width.

Finally, BID1 improves the lower bound by providing a family of ReLU DNNS with an exponential number of regions given fixed size and depth.

In this work, we directly improve on the results of Montúfar et al. (Pascanu et al., 2014; Montúfar et al., 2014; Montúfar, 2017) and Raghu et al. (Raghu et al., 2017) in better understanding the representational power of DNNs employing PWL activation functions.

We will only consider feedforward neural networks in this paper.

Let us assume that the network has n 0 input variables given by x = {x 1 , x 2 , . . .

, x n0 }, and m output variables given by y = {y 1 , y 2 , . . .

, y m }.

Each hidden layer l = {1, 2, . . . , L} has n l hidden neurons whose activations are given by h l = {h }.

Let W l be the n l × n l−1 matrix where each row corresponds to the weights of a neuron of layer l. Let b l be the bias vector used to obtain the activation functions of neurons in layer l. Based on the ReLU(x) = max{0, x} activation function, the activations of the hidden neurons and the outputs are given below: DISPLAYFORM0 As considered in Pascanu et al. (2014) , the output layer is a linear layer that computes the linear combination of the activations from the previous layer without any ReLUs.

We can treat the DNN as a piecewise linear (PWL) function F : R n0 → R m that maps the input x in R n0 to y in R m .

This paper primarily deals with investigating the bounds on the linear regions of this PWL function.

There are two subtly different definitions for linear regions in the literature and we will formally define them.

Definition 1.

Given a PWL function F : R n0 → R m , a linear region is defined as a maximal connected subset of the input space R n0 , on which F is linear (Pascanu et al., 2014; Montúfar et al., 2014) .Activation Pattern: Let us consider an input vector x = {x 1 , x 2 , . . .

, x n0 }.

For every layer l we define an activation set S l ⊆ {1, 2, . . .

, n l } such that e ∈ S l if and only if the ReLU e is active, that is, h l e > 0.

We aggregate these activation sets into a set S = (S 1 , . . .

, S l ), which we call an activation pattern.

Note that we may consider activation patterns up to a layer l ≤ L. Activation patterns were previously defined in terms of strings (Raghu et al., 2017; Montúfar, 2017) .We say that an input x corresponds to an activation pattern S in a DNN if feeding x to the DNN results in the activations in S. Definition 2.

Given a PWL function F : R n0 → R m represented by a DNN, a linear region is the set of input vectors x that corresponds to an activation pattern S in the DNN.We prefer to look at linear regions as activation patterns and we interchangeably refer to S as an activation pattern or a region.

Definitions 1 and 2 are essentially the same, except in a few degenerate cases.

There could be scenarios where two different activation patterns may correspond to two adjacent regions with the same linear function.

In this case, Definition 1 will produce only one linear region whereas Definition 2 will yield two linear regions.

This has no effect on the bounds that we derive in this paper.

In FIG1 we show a simple ReLU DNN with two inputs {x 1 , x 2 } and 3 hidden layers.

The activation units {a, b, c, d, e, f } in the hidden layers can be thought of as hyperplanes that each divide the space in two.

On one side of the hyperplane, the unit outputs a positive value.

For all points on the other side of the hyperplane including itself, the unit outputs 0.One may wonder: into how many regions do n hyperplanes split a space?

Zaslavsky (1975) shows that an arrangement of n hyperplanes divides a d-dimensional space into at most d s=0 n s regions, a bound that is attained when they are in general position.

The term general position basically means that a small perturbation of the hyperplanes does not change the number of regions.

This corresponds to the exact maximal number of regions of a single layer DNN with n ReLUs and input dimension d.

In FIG1 -(g), we provide a visualization of how ReLUs partition the input space.

Figs. 1(e), (f), and (g) show the hyperplanes corresponding to the ReLUs at layers l = 1, 2, and 3 respectively.

Figs. 1(b), (c), and (d) consider these same hyperplanes in the input space x. In FIG1 , as per Zaslavsky (1975) , the 2D input space is partitioned into 4 regions ( and (d), we add the hyperplanes from the second and third layers respectively, which are affected by the transformations applied in the earlier hidden layers.

The regions are further partitioned as we consider additional layers.

FIG1 also highlights that activation boundaries behave like hyperplanes when inside a region and may bend whenever they intersect with a boundary from a previous layer.

This has also been pointed out by Raghu et al. (2017) .

In particular, they cannot appear twice in the same region as they are defined by a single hyperplane if we fix the region.

Moreover, these boundaries do not need to be connected, as illustrated in FIG3 .

We summarize the main contributions of this paper below:• We achieve tighter upper and lower bounds on the maximal number of linear regions of the PWL function corresponding to a DNN that employs ReLUs.

As a special case, we present the exact maximal number of regions when the input dimension is one.

We ad- ditionally provide the first upper bound on the number of linear regions for multi-layer maxout networks (See Sections 3 and 4).• We show for ReLUs that the exact maximal number of linear regions of shallow networks is larger than that of deep networks if the input dimension exceeds the number of neurons.

This result is particularly interesting, since it cannot be inferred from the bounds derived in prior work.• We use a mixed-integer linear formulation to show that exact counting of the linear regions is indeed possible.

For the first time, we show the exact counting of the number of linear regions for several small-sized DNNs during the training process.

This new capability can be used to evaluate the tightness of the bounds and potentially analyze the correlation between validation accuracy and the number of linear regions.

It also provides new insights as to how the linear regions vary during the training process (See Section 5 and 6).

appropriately, this lower bound is Ω(s n0 ) where s is the total size of the network.

We derive both upper and lower bounds that improve upon these previous results.

In this section, we prove the following upper bound on the number of regions.

Theorem 1.

Consider a deep rectifier network with L layers, n l rectified linear units at each layer l, and an input of dimension n 0 .

The maximal number of regions of this neural network is at most DISPLAYFORM0 Note that this is a stronger upper bound than the one that appeared in Montúfar (2017), which can be derived from this bound by relaxing the terms n l − j l to n l and factoring the expression.

When n 0 = O(1) and all layers have the same width n, this expression has the same best known asymptotic bound O(n Ln0 ) first presented in Raghu et al. (2017) .Two insights can be extracted from the above expression:1.

Bottleneck effect.

The bound is sensitive to the positioning of layers that are small relative to the others, a property we call the bottleneck effect.

If we subtract a neuron from one of two layers with the same width, choosing the one closer to the input layer will lead to a larger (or equal) decrease in the bound.

This occurs because each index j l is essentially limited by the widths of the current and previous layers, n 0 , n 1 , . . .

, n l .

In other words, smaller widths in the first few layers of the network imply a bottleneck on the bound.

In particular for a 2-layer network, we show in Appendix A that if the input dimension is sufficiently large to not create its own bottleneck, then moving a neuron from the first layer to the second layer strictly decreases the bound, as it tightens a bottleneck.

FIG5 illustrates this behavior.

For the solid line, we keep the total size of the network the same but shift from a small-to-large network (i.e., smaller width near the input layer and larger width near the output layer) to a large-to-small network in terms of width.

We see that the bound monotonically increases as we reduce the bottleneck.

If we add a layer of constant width at the end, represented by the dashed line, the bound decreases when the layers before the last become too small and create a bottleneck for the last layer.

While this is a property of the upper bound rather than one of the exact maximal number of regions, we observe in Section 6 that empirical results for the number of regions of a trained network exhibit a behavior that resembles the bound as the width of the layers vary.2.

Deep vs shallow for large input dimensions.

In several applications such as imaging, the input dimension can be very large.

Montúfar et al. (2014) show that if the input dimension n 0 is constant, then the number of regions of deep networks is asymptotically larger than that of shallow (single-layer) networks.

We complement this picture by establishing that if the input dimension is large, then shallow networks can attain more regions than deep networks.

More precisely, we compare a deep network with L layers of equal width n and a shallow network with one layer of width Ln.

In Appendix A, we show using Theorem 1 that if the input dimension n 0 exceeds the size of the network Ln, then the ratio between the exact maximal number of regions of the deep and of the shallow network goes to zero as L approaches infinity.

We also show in Appendix A that in a 2-layer network, if the input dimension n 0 is larger than both widths n 1 and n 2 , then turning it into a shallow network with a layer of n 1 + n 2 ReLUs increases the exact maximal number of regions.

FIG5 illustrates this behavior.

As we increase the number of layers while keeping the total size of the network constant, the bound plateaus at a value lower than the exact maximal number of regions for shallow networks.

Moreover, the number of layers that yields the highest bound decreases as we increase the input dimension n 0 .

It is important to note that this property cannot be inferred from previous upper bounds derived in prior work, since they are at least 2 N when n 0 ≥ max{n 1 , . . .

, n L }, where N is the total number of neurons.

We remark that asymptotically both deep and shallow networks can attain exponentially many regions when the input dimension is at least n (see Appendix B).

We now build towards the proof of Theorem 1.

For a given activation set S l and a matrix W with n l rows, let σ S l (W ) be the operation that zeroes out the rows of W that are inactive according to S l .

This represents the effect of the ReLUs.

For a region S at layer l − 1, defineW DISPLAYFORM1 Each region S at layer l − 1 may be partitioned by a set of hyperplanes defined by the neurons of layer l.

When viewed in the input space, these hyperplanes are the rows ofW l S x + b = 0 for some b. To verify this, note that, if we recursively substitute out the hidden variables h l−1 , . . .

, h 1 from the original hyperplane W l h l−1 + b l = 0 following S, the resulting weight matrix applied to x isW l S .

Finally, we define the dimension of a region S at layer l − 1 as dim(S) : DISPLAYFORM2 ).

This can be interpreted as the dimension of the space corresponding to S that W l effectively partitions.

The proof of Theorem 1 focuses on the dimension of each region S. A key observation is that once it falls to a certain value, the regions contained in S cannot recover to a higher dimension.

Zaslavsky (1975) showed that the maximal number of regions in R d induced by an arrangement of m hyperplanes is at most The proof is given in Appendix C. Its key idea is that it suffices to count regions within the row space of W .

The next lemma brings Lemma 2 into our context.

Lemma 3.

The number of regions induced by the n l neurons at layer l within a certain region S is at most DISPLAYFORM3 Proof.

The hyperplanes in a region S of the input space are given by the rows ofW DISPLAYFORM4 By the definition ofW l S , the rank ofW DISPLAYFORM5 Applying Lemma 2 yields the result.

In the next lemma, we show that the dimension of a region S can be bounded recursively in terms of the dimension of the region containing S and the number of activated neurons defining S. Lemma 4.

Let S be a region at layer l and S be the region at layer l − 1 that contains it.

Then DISPLAYFORM6 The last inequality comes from the fact that the zeroed out rows do not count towards the rank of the matrix.

In the remainder of the proof of Theorem 1, we combine Lemmas 3 and 4 to construct a recurrence R(l, d) that bounds the number of regions within a given region of dimension d. Simplifying this recurrence yields the expression in Theorem 1.

We formalize this idea and complete the proof of Theorem 1 in Appendix D.As a side note, Theorem 1 can be further tightened if the weight matrices are known to have small rank.

The bound from Lemma 3 can be rewritten as DISPLAYFORM7 n l j if we do not relax rank(W l ) to n l in the proof.

The term rank(W l ) follows through the proof of Theorem 1 and the index set J in the theorem becomes DISPLAYFORM8 A key insight from Lemmas 3 and 4 is that the dimensions of the regions are non-increasing as we move through the layers partitioning it.

In other words, if at any layer the dimension of a region becomes small, then that region will not be able to be further partitioned into a large number of regions.

For instance, if the dimension of a region falls to zero, then that region will never be further partitioned.

This suggests that if we want to have many regions, we need to keep dimensions high.

We use this idea in the next section to construct a DNN with many regions.

If the input dimension n 0 is equal to 1 and n l = n for all layers l, the upper bound presented in the previous section reduces to (n + 1) L .

On the other hand, the lower bound given by Montúfar et al. FORMULA16 becomes n L−1 (n + 1).

It is then natural to ask: are either of these bounds tight?

The answer is that the upper bound is tight in the case of n 0 = 1, assuming there are sufficiently many neurons.

Theorem 5.

Consider a deep rectifier network with L layers, n l ≥ 3 rectified linear units at each layer l, and an input of dimension 1.

The maximal number of regions of this neural network is exactly DISPLAYFORM0 The expression above is a simplified form of the upper bound from Theorem 1 in the case n 0 = 1.The proof of this theorem in Appendix E has a construction with n + 1 regions that replicate themselves as we add layers, instead of n as in Montúfar et al. (2014) .

That is motivated by an insight from the previous section: in order to obtain more regions, we want the dimension of every region to be as large as possible.

When n 0 = 1, we want all regions to have dimension one.

This intuition leads to a new construction with one additional region that can be replicated with other strategies.

Both the lower bound from Montúfar et al. (2014) and from BID1 can be slightly improved, since their approaches are based on extending a 1-dimensional construction similar to the one in Section 3.2.

We do both since they are not directly comparable: the former bound is in terms of the number of neurons in each layer and the latter is in terms of the total size of the network.

Theorem 6.

The maximal number of linear regions induced by a rectifier network with n 0 input units and L hidden layers with n l ≥ 3n 0 for all l is lower bounded by DISPLAYFORM0 The proof of this theorem is in Appendix F. For comparison, the differences between the lower bound theorem (Theorem 5) from Montúfar et al. (2014) and the above theorem is the replacement of the condition n l ≥ n 0 by the more restrictive n l ≥ 3n 0 , and of n l /n 0 by n l /n 0 + 1.

Theorem 7.

For any values of m ≥ 1 and w ≥ 2, there exists a rectifier network with n 0 input units and L hidden layers of size 2m + w(L − 1) that has 2 n0−1 j=0 DISPLAYFORM1 The proof of this theorem is in Appendix G. The differences between Theorem 2.11(i) from BID1 and the above theorem is the replacement of w by w + 1.

They construct a 2m-width layer with many regions and use a one-dimensional construction for the remaining layers.

We now consider a deep neural network composed of maxout units.

Given weights W l j for j = 1, . . .

, k, the output of a rank-k maxout layer l is given by DISPLAYFORM0 In terms of bounding number of regions, a major difference between the next result for maxout units and the previous one for ReLUs is that reductions in dimensionality due to inactive neurons with zeroed output become a particular case now.

Nevertheless, using techniques similar to the ones from Section 3.1, the following theorem can be shown (see Appendix H for the proof).

Theorem 8.

Consider a deep neural network with L layers, n l rank-k maxout units at each layer l, and an input of dimension n 0 .

The maximal number of regions of this neural network is at most DISPLAYFORM1 Asymptotically, if n l = n for all l = 1, . . .

, L, n ≥ n 0 , and n 0 = O(1), then the maximal number of regions is at most O((k 2 n) Ln0 ).

If the input space x ∈ R n0 is bounded by minimum and maximum values along each dimension, or else if x corresponds to a polytope more generally, then we can define a mixed-integer linear formulation mapping polyhedral regions of x to the output space y ∈ R m .

The assumption that x is bounded and polyhedral is natural in most applications, where each value x i has known lower and upper bounds (e.g., the value can vary from 0 to 1 for image pixels).

Among other things, we can use this formulation to count the number of linear regions.

In the formulation that follows, we use continuous variables to represent the input x, which we can also denote as h 0 , the output of each neuron i in layer l as h l i , and the output y as h L+1 .

To simplify the representation, we lift this formulation to a space that also contains the output of a complementary set of neurons, each of which is active when the corresponding neuron is not.

Namely, for each neuron i in layer l we also have a variable h DISPLAYFORM0 We use binary variables of the form z l i to denote if each neuron i in layer l is active or else if the complement of such neuron is.

Finally, we assume M to be a sufficiently large constant.

For a given neuron i in layer l, the following set of constraints maps the input to the output: DISPLAYFORM1 Theorem 9.

Provided that |w DISPLAYFORM2 for any possible value of h l−1 , a formulation with the set of constraints (1) for each neuron of a rectifier network is such that a feasible solution with a fixed value for x yields the output y of the neural network.

The proof for the statement above is given in Appendix I. More details on the procedure for exact counting are in Appendix J. In addition, we show the theory for unrestricted inputs and a mixedinteger formulation for maxout networks in Appendices K and L, respectively.

These results have important consequences.

First, they allow us to tap into the literature of mixedinteger representability (Jeroslow, 1987) and disjunctive programming BID2 to understand what can be modeled on rectifier networks with a finite number of neurons and layers.

To the best of our knowledge, that has not been discussed before.

Second, they imply that we can use mixedinteger optimization solvers to analyze the (x, y) mapping of a trained neural network.

We perform two different experiments for region counting using small-sized networks with ReLU activation units on the MNIST benchmark dataset (LeCun et al., 1998) .

In the first experiment, we generate rectifier networks with 1 to 4 hidden layers having 10 neurons each, with final test error between 6 and 8%.

The training was carried out for 20 epochs or training steps, and we count the number of linear regions during each training step.

For those networks, we count the number of linear regions within 0 ≤ x ≤ 1 in which a single neuron is active in the output layer, hence partitioning these regions in terms of the digits that they classify.

In Fig. 4 , we show how the number of regions classifying each digit progresses during training.

Some digits have zero linear regions in the beginning, which explains why they begin later in the plot.

The total number of such regions per training step is presented in Fig. 5(a) and error measures are found in Appendix M. Overall, we observe that the number of linear regions jumps orders of magnitude are varies more widely for each added layer.

Furthermore, there is an initial jump in the number of linear regions classifying each digit that seems proportional to the number of layers.

In the second experiment, we train rectifier networks with two hidden layers summing up to 22 neurons.

We train a network for each width configuration under the same conditions as above, with the test error in half of them ranging from 5 to 6%.

In this case, we count all linear regions within 0 ≤ x ≤ 1, hence not restricting by activation in output layer as before.

The number of linear regions of these networks are plotted in Fig. 5(b) , along with the upper bound from Theorem 1 and the upper bounds from Montúfar et al. (2014) and Montúfar (2017) .

Error measures of both experiments can be found in Appendix M and runtimes for counting the linear regions in Appendix N.

The representational power of a DNN can be studied by observing the number of linear regions of the PWL function that the DNN represents.

In this work, we improve on the upper and lower bounds on the linear regions for rectified networks derived in prior work (Montúfar et al., 2014; Raghu et al., 2017; Montúfar, 2017; BID1 and introduce a first upper bound for multi-layer maxout networks.

We obtain several valuable insights from our extensions.

Our ReLU upper bound indicates that small widths in early layers cause a bottleneck effect on the number of regions.

If we reduce the width of an early layer, the dimensions of the linear regions become irrecoverably smaller throughout the network and the regions will not be able to be partitioned as much.

Moreover, the dimensions of the linear regions are not only driven by width, but also the number of activated ReLUs corresponding to the region.

This intuition allowed us to create a 1-dimensional construction with the maximal number of regions by eliminating a zero-dimensional bottleneck.

An unexpected and useful consequence of our result is that shallow networks can attain more linear regions when the input dimensions exceed the number of neurons of the DNN.In addition to achieving tighter bounds, we use a mixed-integer linear formulation that maps the input space to the output to show the exact counting of the number of linear regions for several small-sized DNNs during the training process.

In the first experiment, we observed that the number of linear regions correctly classifying each digit of the MNIST benchmark increases and vary in proportion to the depth of the network during the first training epochs.

In the second experiment, we count the total number of linear regions as we vary the width of two layers with a fixed number of neurons, and we experimentally validate the bottleneck effect by observing that the results follow a similar pattern to the upper bound that we show.

Our current results suggest new avenues for future research.

First, we believe that the study of linear regions may eventually lead to insights in how to design better DNNs in practice, for example by further validating the bottleneck effect found in this study.

Other properties of the bounds may turn into actionable insights if confirmed as these bounds get sufficiently close to the actual number of regions.

For example, the plots in Appendix O show that there are particular network depths that maximize our ReLU upper bound for a given input dimension and number of neurons.

In a sense, the number of neurons is a proxy to the computational resources available.

We also believe that analyzing the shape of the linear regions is a promising idea for future work, which could provide further insight in how to design DNNs.

Another important line of research is to understand the exact relation between the number of linear regions and accuracy, which may also involve the potential for overfitting.

We conjecture that the network training is not likely to generalize well if there are so many regions that each point can be singled out in a different region, in particular if regions with similar labels are unlikely to be compositionally related.

Second, applying exact counting to larger networks would depend on more efficient algorithms or on using approximations instead.

In any case, the exact counting at a smaller scale can assess the quality of the current bounds and possibly derive insights for tighter bounds in future work, hence leading to insights that could be scaled up.

Most of the proofs for theorems and lemmas associated with the upper and lower bounds on the linear regions are provided below.

The theory for mixed-integer formulation for exact counting in the case of maxouts and unrestricted inputs are also provided below.

A ANALYSIS OF THE BOUND FROM THEOREM 1In this section, we present properties of the upper bound for the number of regions of a rectifier network from Theorem 1.

Denote the bound by B(n 0 , n 1 , . . .

, n L ), where n 0 is the input dimension and n 1 , . . .

, n L are the widths of layers 1 through L of the network.

That is, DISPLAYFORM0 Instead of expressing J as in Theorem 1, we rearrange it to a more convenient form for the proofs in this section: DISPLAYFORM1 Note that whenever we assume n 0 ≥ max{n 1 , . . .

, n l }, then the bound inequality for n 0 becomes redundant and can be removed.

Some of the results have implications in terms of the exact maximal number of regions.

We denote it by R(n 0 , n 1 , . . .

, n L ), following the same notation above.

Moreover, the following lemma is useful throughout the section.

Lemma 10.

DISPLAYFORM2 Proof.

The result comes from taking a generalization of Vandermonde's identity and adding the summation of j from 0 to k as above.

We first examine some properties related to 2-layer networks.

The proposition below characterizes the bound when L = 2 for large input dimensions.

Proposition 11.

Consider a 2-layer network with widths n 1 , n 2 and input dimension n 0 ≥ n 1 and n 0 ≥ n 2 .

Then DISPLAYFORM3 If n 0 < n 1 or n 0 < n 2 , the above holds with inequality: B(n 0 , n 1 , n 2 ) ≤ n1 j=0 n1+n2 j .Proof.

If n 0 ≥ n 1 and n 0 ≥ n 2 , the bound inequalities for n 0 in the index set J become redundant.

By applying Lemma 10, we obtain DISPLAYFORM4 If n 0 < n 1 or n 0 < n 2 , then its index set J is contained by the one above, and thus the first equal sign above becomes a less-or-equal sign.

Recall that the expression on the right-hand side of Proposition 11 is equal to the maximal number of regions of a single-layer network with n 1 + n 2 ReLUs and input dimension n 1 , as discussed in Section 2.

Hence, the proposition implies that for large input dimensions, a two-layer network has no more regions than a single-layer network with the same number of neurons, as formalized below.

Corollary 12.

Consider a 2-layer network with widths n 1 , n 2 ≥ 1 and input dimension n 0 ≥ n 1 and n 0 ≥ n 2 .

Then R(n 0 , n 1 , n 2 ) ≤ R(n 0 , n 1 + n 2 ).Moreover, this inequality is strict when n 0 > n 1 .Proof.

This is a direct consequence of Proposition 11: DISPLAYFORM5 Note that if n 0 > n 1 , then the second inequality can be turned into a strict inequality.

The next corollary illustrates the bottleneck effect for two layers.

It states that for large input dimensions, moving a neuron from the second layer to the first strictly increases the bound.

Corollary 13.

Consider a 2-layer network with widths n 1 , n 2 and input dimension n 0 ≥ n 1 + 1 and n 0 ≥ n 2 + 1.

Then B(n 0 , n 1 + 1, n 2 ) > B(n 0 , n 1 , n 2 + 1).Proof.

By Proposition 11, DISPLAYFORM6 The assumption that n 0 must be large is required for the above proposition; otherwise, the input itself may create a bottleneck with respect to the second layer as we decrease its size.

Note that the bottleneck affects all subsequent layers, not only the layer immediately after it.

However, it is not true that moving neurons to earlier layers always increases the bound.

For instance, with three layers, B(4, 3, 2, 1) = 47 > 46 = B(4, 4, 1, 1).In the remainder of this section, we consider deep networks of equal widths n.

The next proposition can be viewed as an extension of Proposition 11 for multiple layers.

It states that for a network with widths and input dimension n and at least 4 layers, if we halve the number of layers and redistribute the neurons so that the widths become 2n, then the bound increases.

In other words, if we assume the bound to be close to the maximal number of regions, it suggests that making a deep network shallower allows for more regions when the input dimension is equal to the width.

Proposition 14.

Consider a 2L-layer network with equal widths n and input dimension n 0 = n. Then DISPLAYFORM7 This inequality is met with equality when L = 1 and strict inequality when L ≥ 2.Proof.

When n 0 = n, the inequalities j l ≤ min{n 0 , 2n − j 1 , . . .

, 2n − j l−1 , 2n} appearing in J (in the form presented in Theorem 1) can be simplified to j l ≤ n. Therefore, using Lemma 10, the bound on the right-hand side becomes DISPLAYFORM8 where J above is the index set from Theorem 1 applied to n 0 = n l = n for all l = 1, . . .

, 2L.Note that we can turn the inequality into equality when L = 1 (also becoming a consequence of Proposition 11) and into strict inequality when L ≥ 2.Next, we provide an upper bound that is independent of n 0 .Proposition 15.

Consider an L-layer network with equal widths n and any input dimension n 0 ≥ 0.

DISPLAYFORM9 Proof.

Since we are deriving an upper bound, we can assume n 0 ≥ n, as the bound is nondecreasing on n 0 .

We first assume that L is even.

We relax some of the constraints of the index set J from Theorem 1 and apply Vandermonde's identity on each pair: DISPLAYFORM10 The bound on 2n n is a direct application of Stirling's approximation (Stirling, 1730) .

If L is odd, then we can write DISPLAYFORM11 where the last inequality is analogous to the even case.

Hence, the result follows.

Corollary 16.

Consider an L-layer network with equal widths n and any input dimension n 0 ≥ 0.

DISPLAYFORM12 Proof.

By Proposition 15 and Theorem 1, the ratio between R(n 0 , n, . . . , n) and 2 Ln is at most DISPLAYFORM13 Since the base of the first term is less than 1 for all n ≥ 1 and √ 2 is a constant, the ratio goes to 0 as L goes to infinity.

In particular, Corollary 16 implies that if n 0 exceeds the total size of the network, that is, n 0 ≥ Ln, then lim L→∞ R(n0,n,...,n) R(n0,Ln) = 0.

In other words, the ratio between the maximal number of regions of a deep network and a shallow network goes to zero as L goes to infinity.

Proposition 17.

Consider an L-layer rectifier network with equal widths n and input dimension n 0 ≥ n/3.

Then the maximal number of regions is Ω(2 2 3 Ln ).Proof.

It suffices to show that a lower bound such as the one from Theorem 6 grows exponentially large.

For simplicity, we consider the lower bound ( L l=1 ( n l /n 0 + 1)) n0 , which is the bound obtained before the last tightening step in the proof of Theorem 6 (see Appendix F).Note that replacing n 0 in the above expression by a value n 0 smaller than the input dimension still yields a valid lower bound.

This holds because increasing the input dimension of a network from n 0 to n 0 cannot decrease its maximal number of regions.

Choose n 0 = n/3 , which satisfies n 0 ≤ n 0 and the condition n ≥ 3n 0 of Theorem 6.

The lower bound can be expressed as ( n/ n/3 + 1) L n/3 ≥ 4 L n/3 .

This implies that the maximal number of regions is Ω(2 Proof.

Consider the row space R(W ) of W , which is a subspace of R d of dimension rank(W ).

We show that the number of regions DISPLAYFORM0 .

This suffices to prove the lemma since R(W ) has at most rank(W ) j=0 m j regions according to Zaslavsky's theorem.

DISPLAYFORM1 To show the converse, we apply the orthogonal decomposition theorem from linear algebra: any pointx ∈ R d can be expressed uniquely asx =x + y, wherex ∈ R(W ) and y ∈ R(W ) ⊥ .

Here, R(W ) ⊥ = Ker(W ) := {y ∈ R d : W y = 0}, and thus Wx = Wx + W y = Wx.

This meansx andx lie on the same side of each hyperplane of W x = b and thus belong to the same region.

In other words, given anyx ∈ R d , its region is the same one thatx ∈ R(W ) lies in.

Therefore, DISPLAYFORM2 and the result follows.

Theorem 1.

Consider a deep rectifier network with L layers, n l rectified linear units at each layer l, and an input of dimension n 0 .

The maximal number of regions of this neural network is at most DISPLAYFORM0 . .

, n l−1 − j l−1 , n l } ∀l = 1, . . . , L}. This bound is tight when L = 1.Proof.

As illustrated in FIG1 , the partitioning can be viewed as a sequential process: at each layer, we partition the regions obtained from the previous layer.

When viewed in the input space, each region S obtained at layer l − 1 is potentially partitioned by n l hyperplanes given by the rows ofW l S + b = 0 for some bias b. Some of these hyperplanes may fall outside the interior of S and do not partition the region.

With this process in mind, we recursively bound the number of subregions within a region.

More precisely, we construct a recurrence R(l, d) to be an upper bound to the maximal number of regions obtained from partitioning a region of dimension d with layers l, l + 1, . . .

, L. The base case of the recurrence is given by Lemma 3: DISPLAYFORM1 Based on Lemma 4, we can write the recurrence by grouping together regions with the same activation set size |S l |, as follows: DISPLAYFORM2 ,j represents the maximum number of regions with |S l | = j obtained by partitioning a space of dimension d with n l hyperplanes.

We bound this value next.

For each j, there are at most n l j regions with |S l | = j, as they can be viewed as subsets of n l neurons of size j.

In total, Lemma 3 states that there are at most min{n l ,d} j=0 n l j regions.

If we allow these regions to have the highest |S l | possible, for each j from 0 to min{n l , d} we have at most DISPLAYFORM3 Therefore, we can write the recurrence as DISPLAYFORM4 The recurrence R(1, n 0 ) can be unpacked to DISPLAYFORM5 This can be made more compact, resulting in the final expression.

The bound is tight when L = 1 since it becomes min{n0,n1} j=0 n1 j , which is the maximal number of regions of a single-layer network.

Theorem 5.

Consider a deep rectifier network with L layers, n l ≥ 3 rectified linear units at each layer l, and an input of dimension 1.

The maximal number of regions of this neural network is exactly L l=1 (n l + 1).Proof.

Section 3 provides us with a helpful insight to construct an example with a large number of regions.

It tells us that we want regions to have large dimension in general.

In particular, regions of dimension zero cannot be further partitioned.

This suggests that the one-dimensional construction from Montúfar et al. (2014) can be improved, as it contains n regions of dimension one and 1 region of dimension zero.

This is because all ReLUs point to the same direction as depicted in FIG10 , leaving one region with an empty activation pattern.

Our construction essentially increases the dimension of this region from zero to one.

This is done by shifting the neurons forward and flipping the direction of the third neuron, as illustrated in FIG10 .

We assume n ≥ 3.We review the intuition behind the construction strategy from Montúfar et al. (2014) .

They construct a linear functionh : R → R with a zigzag pattern from [0, 1] to [0, 1] that is composed of n ReLUs.

More precisely,h(x) = (1, −1, 1, . . . , ±1) (h 1 (x), h 2 (x), . . . , h n (x)), where h i (x) for i = 1, . . . , n are ReLUs.

This linear function can be absorbed in the preactivation function of the next layer.

The zigzag pattern allows it to replicate in each slope a scaled copy of the function in the domain [0, 1].

FIG11 shows an example of this effect.

Essentially, when we composeh with itself, each linear piece in [t 1 , t 2 ] such thath(t 1 ) = 0 andh(t 2 ) = 1 maps the entire functionh to the interval [t 1 , t 2 ], and each piece such thath(t 1 ) = 1 andh(t 2 ) = 2 does the same in a backward manner.

In our construction, we want to use n ReLUs to create n + 1 regions instead of n. In other words, we want the construct this zigzag pattern with n + 1 slopes.

In order to do that, we take two steps to give ourselves more freedom.

First, observe that we only need each linear piece to go from zero to one or one to zero; that is, the construction works independently of the length of each piece.

Therefore, we turn the breakpoints into parameters t 1 , t 2 , . . .

, t n , where 0 < t 1 < t 2 < . . .

< t n < 1.

Second, we add sign and bias parameters to the functionh.

That is,h(x) = (s 1 , s 2 , . . . , s n ) (h 1 (x), h 2 (x), . . .

, h n (x)) + d, where s i ∈ {−1, +1} and d are parameters to be set.

Here, h i (x) = max{0,w i x +b i } since it is a ReLU.We define w i = s iwi and b i = s ibi , which are the weights and biases we seek in each interval to form the zigzag pattern.

The parameters s i are needed because the signs ofw i cannot be arbitrary: it must match the directions the ReLUs point towards.

In particular, we need a positive slope (w i > 0) if we want i to point right, and a negative slope (w i < 0) if we want i to point left.

Hence, without loss of generality, we do not need to consider the s i 's any further since they will be directly defined from the signs of the w i 's and the directions.

More precisely, s i = 1 if w i ≥ 0 and s i = −1 otherwise for i = 1, 2, 4, . . .

, n, and s 3 = −1 if w 3 ≥ 0 and s 3 = 1 otherwise.

To summarize, our parameters are the weights w i and biases b i for each ReLU, a global bias d, and the breakpoints 0 < t 1 < . . .

< t n < 1.

Our goal is to find values for these parameters such that each piece in the functionh with domain in [0, 1] is linear from zero to one or one to zero.

More precisely, if the domain is [s, t], we want each linear piece to be either 1 t−s x− s t−s or − 1 t−s x+ t t−s , which define linear functions from zero to one and from one to zero respectively.

Since we want a zigzag pattern, the former should happen for the interval [t i , t i−1 ] when i is odd and the latter should happen when i is even.

There is one more set of parameters that we will fix.

Each ReLU corresponds to a hyperplane, or a point in dimension one.

In fact, these points are the breakpoints t 1 , . . .

, t n .

They have directions that define for which inputs the neuron is activated.

For instance, if a neuron h i points to the right, then the neuron h i (x) outputs zero if x ≤ t i and the linear function w i x + b i if x > t i .As previously discussed, in our construction all neurons point right except for the third neuron h 3 , which points left.

This is to ensure that the region before t 1 has one activated neuron instead of zero, which would happen if all neurons pointed left.

However, although ensuring every region has dimension one is necessary to reach the bound, not every set of directions yields valid weights.

These directions are chosen so that they admit valid weights.

The directions of the neurons tells us which neurons are activated in each region.

From left to right, we start with h 3 activated, then we activate h 1 and h 2 as we move forward, we deactivate h 3 , and finally we activate h 4 , . . .

, h n in sequence.

This yields the following system of equations, where t n+1 is defined as 1 for simplicity: DISPLAYFORM0 It is left to show that there exists a solution to this system of linear equations such that 0 < t 1 < . . .

< t n < 1.First, note that all of the biases b 1 , . . .

, b n , d can be written in terms of t 1 , . . .

, t n .

Note that if we subtract (R 4 ) from (R 3 ), we can express b 3 in terms of the t i variables.

The remaining equations become triangular, and therefore given any values for t i 's we can back-substitute the remaining bias variables.

The same subtraction yields w 3 in terms of t i '

s.

However, both (R 1 ) and (R 3 ) − (R 4 ) define w 3 in terms of the t i variables, so they must be the same: DISPLAYFORM1 If we find values for t i 's satisfying this equation and 0 < t 1 < . . .

< t n < 1, all other weights can be obtained by back-substitution since eliminating w 3 yields a triangular set of equations.

In particular, the following values are valid: DISPLAYFORM2 2n+1 for all i = 2, . . .

, n. The remaining weights and biases can be obtained as described above, which completes the desired construction.

As an example, a construction with four units is depicted in FIG10 .

Its breakpoints are t 1 = 1 9 , t 2 = 3 9 , t 3 = 5 9 , and t 4 = 7 9 .

Its ReLUs are h 1 (x) = max{0, − 27 2 x + 3 2 }, h 2 (x) = max{0, 9x − 3}, h 3 (x) = max{0, 9x − 5}, and h 4 (x) = max{0, 9x}. Finally,h(x) = (−1, 1, −1, 1) (h 1 (x), h 2 (x), h 3 (x), h 4 (x)) + 5.

Theorem 6.

The maximal number of linear regions induced by a rectifier network with n 0 input units and L hidden layers with n l ≥ 3n 0 for all l is lower bounded by DISPLAYFORM0 Proof.

We follow the proof of Theorem 5 from (Montúfar et al., 2014 ) except that we use a different 1-dimensional construction.

The main idea of the proof is to organize the network into n 0 independent networks with input dimension 1 each and apply the 1-dimensional construction to each individual network.

In particular, for each layer l we assign n l /n 0 ReLUs to each network, ignoring any remainder units.

In (Montúfar et al., 2014) , each of these networks have at least L l=1 n l /n 0 regions.

We instead use Theorem 5 to attain L l=1 ( n l /n 0 + 1) regions in each network.

Since the networks are independent from each other, the number of activation patterns of the compound network is the product of the number of activation patterns of each of the n 0 networks.

Hence, the same holds for the number of regions.

Therefore, the number of regions of this network is at least ( DISPLAYFORM1 n0 .In addition, we can replace the last layer by a function representing an arrangement of n L hyperplanes in general position that partitions (0, 1) n0 into n0 j=0 n L j regions.

This yields the lower bound of DISPLAYFORM2 G PROOF OF THEOREM 7 there is one region per linear function.

The boundaries between the regions are composed by pieces that are each contained in a hyperplane.

Each piece is part of the boundary of at least two regions and conversely each pair of regions corresponds to at most one piece.

Extending these pieces into hyperplanes cannot decrease the number of regions.

Therefore, if we now consider n maxout units in a single layer, we can have at most the number of regions of an arrangement of k 2 n hyperplanes.

In the results below we replace k 2 by k 2 , as only pairs of distinct functions need to be considered.

We need to define more precisely these k 2 n hyperplanes in order to apply a strategy similar to the one from the Section 3.1.

In a single layer setting, they are given by w j x + b j = w j + b j for each distinct pair j, j within a neuron.

In order to extend this to multiple layers, consider a k 2 n l × n l−1 matrixŴ l where its rows are given by w j − w j for every distinct pair j, j within a neuron i and for every neuron i = 1, . . .

, n l .

Given a region S, we can now write the weight matrix corresponding to the hyperplanes described above:Ŵ .

In other words, the hyperplanes that extend the boundary pieces within region S are given by the rows ofŴ l S x + b = 0 for some bias b. A main difference between the maxout case and the ReLU case is that the maxout operator φ does not guarantee reductions in rank, unlike the ReLU operator σ.

We show the analogous of Lemma 3 for the maxout case.

However, we fully relax the rank.

Lemma 18.

The number of regions induced by the n l neurons at layer l within a certain region S is at most DISPLAYFORM3 , where d l = min{n 0 , n 1 , . . .

, n l }.Proof.

For a fixed region S, an upper bound is given by the number of regions of the hyperplane arrangement corresponding toŴ It suffices to prove that the constraints for each neuron map the input to the output in the same way that the neural network would.

If W

The formulation above generalizes that for ReLUs with some small modifications.

First, we are computing the output of each term with constraint (14).

The output of the neuron is lower bounded by that of each term with constraint (15).

Finally, we have a binary variable z li m per term of each neuron, which denotes which neuron is active.

Constraint (18) enforces that only one variable is at one per neuron, whereas constraint (16) equates the output of the neuron with the active term.

Each constant M should be chosen in a way that the other terms can vary freely, hence effectively disabling the constraint when the corresponding binary variable is at zero.

Figure 8 shows the error during training for different configurations in the first experiment.

Figure 9 shows the errors after training for different configurations in the second experiment.

In both, we observe some relation between accuracy and the order of magnitude of the linear regions, which suggest that linear regions represent a reasonable proxy to the representational power of DNNs.

O UPPER BOUND BY VARYING THE TOTAL NUMBER OF NEURONS FIG1 shows that the upper bound from Theorem 1 can only be maximized if more layers are added as the number of neurons increase.

In contrast, FIG1 shows that the smallest depth preserving such growth is better because there is a secondary, although still exponential, effect that starts shrinks the bound if the number of layers is too large for the total number of neurons.

@highlight

We empirically count the number of linear regions of rectifier networks and refine upper and lower bounds.

@highlight

This paper presents improved bounds for counting the number of linear regions in ReLU networks.