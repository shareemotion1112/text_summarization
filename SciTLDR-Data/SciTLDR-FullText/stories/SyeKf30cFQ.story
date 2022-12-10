Understanding theoretical properties of deep and locally connected nonlinear network, such as deep convolutional neural network (DCNN), is still a hard problem despite its empirical success.

In this paper, we propose a novel theoretical framework for such networks with ReLU nonlinearity.

The framework bridges data distribution with gradient descent rules, favors disentangled representations and is compatible with common regularization techniques such as Batch Norm, after a novel discovery of its projection nature.

The framework is built upon teacher-student setting, by projecting the student's forward/backward pass onto the teacher's computational graph.

We do not impose unrealistic assumptions (e.g., Gaussian inputs, independence of activation, etc).

Our framework could help facilitate theoretical analysis of many practical issues, e.g. disentangled representations in deep networks.

Deep Convolutional Neural Network (DCNN) has achieved a huge empirical success in multiple disciplines (e.g., computer vision BID0 BID10 He et al., 2016) , Computer Go BID8 BID12 BID13 , and so on).

On the other hand, its theoretical properties remain an open problem and an active research topic.

Learning deep models are often treated as non-convex optimization in a high-dimensional space.

From this perspective, many properties in deep models have been analyzed: landscapes of loss functions (Choromanska et al., 2015b; BID1 BID3 , saddle points (Du et al., 2017; Dauphin et al., 2014) , relationships between local minima and global minimum (Kawaguchi, 2016; Hardt & Ma, 2017; BID5 , trajectories of gradient descent (Goodfellow et al., 2014) , path between local minima BID15 , etc.

However, such a modeling misses two components: neither specific network structures nor input data distribution is considered.

Both are critical in practice.

Empirically, deep models work particular well for certain forms of data (e.g., images); theoretically, for certain data distribution, popular methods like gradient descent is shown to fail to recover network parameters (Brutzkus & Globerson, 2017) .Along this direction, previous theoretical works assume specific data distributions like spherical Gaussian and focus on shallow nonlinear networks BID12 Brutzkus & Globerson, 2017; Du et al., 2018) .

These assumptions yield nice gradient forms and enable analysis of many properties such as global convergence.

However, it is also nontrivial to extend such approaches to deep nonlinear neural networks that yield strong empirical performance.

In this paper, we propose a novel theoretical framework for deep and locally connected ReLU network that is applicable to general data distributions.

Specifically, we embrace a teacher-student setting.

The teacher computes classification labels via a computational graph that has local structures (e.g., CNN): intermediate variables in the graph, (called summarization variables), are computed from a subset of the input dimensions.

The student network, with similar local structures, updates the weights to fit teacher's labels with gradient descent, without knowing the summarization variables.

One ultimate goal is to show that after training, each node in the student network is highly selective with respect to the summarization variable in the teacher.

Achieving this goal will shed light to how the training of practically effective methods like CNN works, which remains a grand challenge.

As a first step, we reformulate the forward/backward pass in gradient descent by marginalizing out the input data conditioned on the graph variables of the teacher at each layer.

The reformulation has nice properties: (1) it relates data distribution with gradient update rules, (2) it is compatible with existing Receptive fields form a hierarchy.

The entire input is denoted as x (or x ω ).

A local region of an input x is denoted as x α .

(b) For each region α, we have a latent multinomial discrete variable z α which is computed from its immediate children {z β } β∈ch (α) .

Given the input x, z α = z α (x α ) is a function of the image content x α at α.

Finally, z ω at the top level is the class label.

(c) A locally connected neural network is trained with pairs (x, z ω (x)), where z ω (x) is the class label generated from the teacher.

(d) For each node j, f j (x) is the activation while g j (x) is the back-propagated gradient, both as function of input x (and weights at different layers).state-of-the-art regularization techniques such as Batch Normalization (Ioffe & Szegedy, 2015) , and (3) it favors disentangled representation when data distributions have factorizable structures.

To our best knowledge, our work is the first theoretical framework to achieve these properties for deep and locally connected nonlinear networks.

Previous works have also proposed framework to explain deep networks, e.g., renormalization group for restricted Boltzmann machines BID2 , spin-glass models (Amit et al., 1985; Choromanska et al., 2015a) , transient chaos models BID4 , differential equations BID11 BID6 , information bottleneck (Achille & Soatto, 2017; BID14 BID7 , etc.

In comparison, our framework (1) imposes mild assumptions rather than unrealistic ones (e.g., independence of activations), (2) explicitly deals with back-propagation which is the dominant approach used for training in practice, and relates it with data distribution, and (3) considers spatial locality of neurons, an important component in practical deep models.

We consider multi-layer (deep) and locally connected network with ReLU nonlinearity.

We consider supervised setting, in which we have a dataset {(x, y)}, where x is the input image and y is its label computed from x deterministically.

It is hard to analyze y which does not have a structure (e.g., random labels).

Here our analysis assumes the generation of y from x has a specific hierarchical structure.

We use teacher-student setting to study the property: a student network learns teacher's label y via gradient descent, without knowing teacher's internal representations.

An interesting characteristics in locally connected network is that each neuron only covers a fraction of the input dimension.

Furthermore, for deep and locally connected network, neurons in the lower layer cover a small region while neurons in the upper layer cover a large region.

We use Greek letters {α, β, . . .

, ω} to represent receptive fields.

For a receptive field α, x α is the content in that region.

We use ω to represent the entire image ( FIG0 ).Receptive fields form a hierarchy: α is a parent of β, denoted as α ∈ pa(β) or β ∈ ch(α), if α ⊇ β and there exists no other receptive field γ / ∈ {α, β} so that α ⊇ γ ⊇ β.

Note that siblings can have substantial overlaps (e.g., β 1 and β 2 in FIG0 ).

With this partial ordering, we can attach layer number l to each receptive field: α ∈ pa(β) implies l(β) = l(α) + 1.

For top-most layer (closest to classification label), l = 0 and for bottom-most layer, l = L.For locally connected network, a neuron (or node) j ∈ α means its receptive field is α.

Denote n α as the number of nodes covering the same region (e.g., multi-channel case, Fig. 2(a) ).

The image content is x α(j) , abbreviated as x j if no ambiguity.

The parent j's receptive field covers its children's.

We assume the label y of the input x is computed by a teacher in a bottom-up manner: for each region α, we compute a summarization variable z α from the summarization variables of its children: DISPLAYFORM0 Figure 2: (a) Multiple nodes (neurons) share the same receptive field α.

Note that n α is the number of nodes sharing the receptive field α.

(b) Grouping nodes with the same receptive fields together.

By abuse of notation, α also represents the collection of all nodes with the same receptive field.

DISPLAYFORM1 ).

This procedure is repeated until the top-level summarization z ω is computed, which is the class label y.

We denote φ = {φ α } as the collection of all summarization functions.

For convenience, we assume z α be discrete variables that takes m α possible values.

Intuitively, m α is exponential w.r.t the area of the receptive field sz(α), for binary input, m α ≤ 2 sz(α) .

We call a particular assignment of z α , z α = a, an event.

For the bottom-most layers, z is just the (discretized) value in each dimension.

At each stage, the upward function is deterministic but lossy: z α does not contain all the information in {z β } for β ∈ ch(α).

Indeed, it keeps relevant information in the input region x α with respect to the class label, and discards the irrelevant part.

During training, all summarization variables Z = {z α } are unknown to the student, except for the label y.

Example of teacher networks.

Locally connected network itself is a good example of teacher network, in which nodes of different channels located at one specific spatial location form some encoding of the variable z α .

Note that the relationship between a particular input x and the corresponding values of the summarization variable z at each layer is purely deterministic.

The reason why probabilistic quantities (e.g., P(z α ) and P(z α |z β )) appear in our formulation, is due to marginalization over z (or x).

This marginalization implicitly establishes a relationship between the conditional probabilities P(z α |z β ) and the input data distribution P(x).

If we have specified P(z α |z β ) at each layer, then we implicitly specify a certain kind of data distribution P(x).

Conversely, given a certain kind of P(x) and summarization function φ, we can compute P(z α |z β ) by sampling x, compute summarization variable z α , and accumulate frequency statistics of P(z α |z β ).

If there is an overlap between sibling receptive fields, then it is likely that some relationship among P(z α |z β ) might exist, which we leave for future work.

Although such an indirect specification may not be as intuitive and mathematically easy to deal with as common assumptions used in previous works (e.g., assuming Gaussian input BID12 Du et al., 2018; Brutzkus & Globerson, 2017) ), it gives much more flexibility of the distribution x and is more likely to be true empirically.

Comparison with top-down generative model.

An alternative (and more traditional) way to specify data distribution is to use a top-down generative model: first sample the label y, then sample the latent variables z α at each layer in a top-down manner, until the input layer.

Marginalizing over all the latent variables z α yields a class-conditioned data distribution P(x|y).The main difficulty of this top-down modeling is that when the receptive fields α and α of sibling latent variables overlap, the underlying graphical model becomes loopy.

This makes the population loss function, which involves an integral over the input data x, very difficult to deal with.

As a result, it is nontrivial to find a concise relationship between the parameters in the top-down modeling (e.g., conditional probability) and the optimization techniques applied to neural network (e.g., gradient descent).

In contrast, as we will see in Sec. 3, our modeling naturally gives relationship between gradient descent rules and conditional probability between nearby summarization variables.

We consider a neuron (or node)

j. Denote f j as its activation after nonlinearity and g j as the (input) gradient it receives after filtered by ReLU's gating ( FIG0 ).

Note that both f j and g j are deterministic functions of the input x and label y, and are abbreviated as f j (x) and g j (x).1 .The activation f j and gradient g k can be written as (note that f j is the binary gating function): DISPLAYFORM0 And the weight update for gradient descent is DISPLAYFORM1 Here is the expectation is with respect to a training dataset (or a batch), depending on whether GD or SGD has been used.

We also use f raw j and g raw j as the counterpart of f j and g j before nonlinearity.

For locally connected network, the activation f j of node j is only dependent on the region x j , rather than the entire image x.

This means that DISPLAYFORM2 However, the gradient g j is determined by the entire image x, and its label y, i.e., g j = g j (x, y).Note that since the label y is a deterministic (but unknown) function of x, for gradient we just write DISPLAYFORM3 Marginalized Gradient.

For locally connected network, the gradient g j has some nice structures.

From Eqn.

17 we knows that DISPLAYFORM4 .

Define x −k = x\x k as the input image x except for x k .

Then we can define the marginalized gradient: DISPLAYFORM5 as the marginalization (average) of x −k , while keep x k fixed.

With this notation, we can write DISPLAYFORM6 On the other hand, the gradient which back-propagates to a node k can be written as DISPLAYFORM7 where f k is the derivative of activation function of node k (for ReLU it is just a gating function).

If we take expectation with respect to x −k |x k on both side, we get DISPLAYFORM8 Note that all marginalized gradients g j (x k ) are independently computed by marginalizing with respect to all regions that are outside the receptive field x k .

Interestingly, there is a relationship between these gradients that respects the locality structure: Theorem 1 (Recursive Property of marginalized gradient).

DISPLAYFORM9 This shows that there is a recursive structure in marginal gradient: we can first compute g j (x j ) for top node j, then by marginalizing over the region within x j but outside x k , we get its projection g j (x k ) on child k, then by Eqn.

20 we collect all projections from all the parents of node k, to get g k (x k ).

This procedure can be repeated until we arrive at the leaf nodes.

Let's first consider the following quantity.

For each neural node j, we want to compute the expected gradient given a particular factor z α , where α = rf(j) (the reception field of node j): DISPLAYFORM0 Note that P(x j |z α ) is the frequency count of x j for z α .

If z α captures all information of x j , then P(x j |z α ) is a delta function.

Throughout the paper, we use frequentist interpretation of probabilities.

Goal.

Intuitively, if we have g j (z α = a) > 0 and g j (z α = a) < 0, then the node j learns about the hidden event z α = a. For multi-class classification, the top level nodes (just below the softmax layer) already embrace such correlations (here j is the class label): g j (y = j) > 0 and g j (y = j) < 0, where we know z ω = y is the top level factor.

A natural question now arises:Does gradient descent automatically push g j (z α ) to be correlated with the factor z α ?

DISPLAYFORM1 n β -by-n α Weight matrix that links group α and β P αβ m α -by-m β Prob P(z β |z α ) of events at group α and β If this is true, then gradient descent on deep models is essentially a weak-supervised approach that automatically learns the intermediate events at different levels.

Giving a complete answer of this question is very difficult and is beyond the scope of this paper.

As a first step, we build a theoretical framework that enables such analysis.

We start with the relationship between neighboring layers: Theorem 2 (Reformulation).

For node j and k and their receptive field α and β.

If the following two conditions holds: DISPLAYFORM2 Then the following iterative equations hold: DISPLAYFORM3 The reformulation becomes exact if z α contains all information of the region.

Theorem 3.

If P(x j |z α ) is a delta function for all α, then all conditions in Thm.

2 hold.

While Thm.

3 holds in the ideal (and maybe trivial) case, both assumptions are still practically reasonable.

For assumption (1), the main idea is that the image content x α is most related to the summarization variable z α located at the same receptive field α, and less related to others.

On the other hand, assumptions (2) holds approximately if the summarization variable is fine-grained.

Intuitively, P(x j |z α ) is a distribution encoding how much information gets lost if we only know the factor z α .

Climbing up the ladder, more and more information is lost while keeping the critical part for the classification.

This is consistent with empirical observations (Bau et al., 2017) , in which the low-level features in DCNN are generic, and high-level features are more class-specific.

One key property of this formulation is that, it relates conditional probabilities P(z α , z β ), and thus input data distribution P(x) into the gradient descent rules.

This is important since running backpropagation on different dataset is now formulated into the same framework with different probability, i.e., frequency counts of events.

By studying which family of distribution leads to the desired property, we could understand backpropagation better.

Furthermore, the property of stochastic gradient descent (SGD) can be modeled as using an imperfect estimateP(z α , z β ) of the true probability P(z α , z β ) when running backpropagation.

This is because each batch is a rough sample of the data distribution so the resulting P(z α , z β ) will also be different.

This could also unite GD and SGD analysis.

For boundary conditions, in the lowest level L, we could treat each input pixel (or a group of pixels) as a single event: DISPLAYFORM4 For top level, each node j corresponds to a class label j while the summarization variable z α also take class labels: DISPLAYFORM5 If we group the nodes with the same reception field at the same level together (Fig. 2) , we have the matrix form of Eqn.

7 (• is element-wise multiplication): Theorem 4 (Matrix Representation of Reformulation).

DISPLAYFORM6 See Tbl.

3 for the notation.

For this dynamics, we want F * ω = I nω , i.e., the top n ω neurons faithfully represents the classification labels.

Therefore, the top level gradient is G ω = I nω − F ω .

On the other side, for each region β at the bottom layer, we have F β = I n β , i.e., the input contains all the preliminary factors.

For all regions α in the top-most and bottom-most layers, we have n α = m α .

Our reformulation naturally incorporates empirical regularization technique like Batch Normalization (BN) (Ioffe & Szegedy, 2015) .

We start with a novel finding of Batch Norm: the back-propagated gradient through Batch Norm layer at a node j is a projection onto the orthogonal complementary subspace spanned by all one vectors and the current activations of node j.

Denote pre-batchnorm activations as DISPLAYFORM0 where N is the batchsize.

In Batch Norm, f is whitened to bef , then linearly transformed to yield the output f bn (note that we omit node subscript j for clarity):f DISPLAYFORM1 where µ = Theorem 5 (Backpropagation of Batch Norm).

For a top-down pre-BN gradient g bn (a vector of size N -by-1,N is the batchsize), the gradient after passing BN layer is the following: DISPLAYFORM2 Here P ⊥ f ,1 is the orthogonal complementary projection onto subspace {f , 1} and DISPLAYFORM3 Intuitively, the back-propagated gradient g is zero-mean and perpendicular to the input activation f of BN layer, as illustrated in FIG1 .

Unlike (Kohler et al., 2018) that analyzes BN in an approximate manner, in Thm.

5 we do not impose any assumptions.

In our reformulation, we take the expectation of input x so there is no explicit notation of batch.

However, we could regard each sample in the batch as i.i.d.

samples from the data distribution P(x).

Then the analysis of Batch Norm in Sec. 4.1 could be applied in the reformulation and yield similar results, using the quantity that DISPLAYFORM0 In this case, we have DISPLAYFORM1 f ,1 .

Note that the projection matrix P DISPLAYFORM2 zα .

In comparison, Sec. 4.1 is a special case with P( DISPLAYFORM3 , where x 1 , . . .

, x N are the batch samples.

One consequence is that forG α , we have 1 DISPLAYFORM4 is in the null space of 1 under the inner product ·, · zα .

This property will be used in Sec. 5.2.

With the help of the theoretical framework, we now can analyze interesting structures of gradient descent in deep models, when the data distribution P(z α , z β ) satisfies specific conditions.

Here we give two concrete examples: the role played by nonlinearity and in which condition disentangled representation can be achieved.

Besides, from the theoretical framework, we also give general comments on multiple issues (e.g., overfitting, GD versus SGD) in deep learning.

In the formulation, m α is the number of possible events within a region α, which is often exponential with respect to the size sz(α) of the region.

The following analysis shows that a linear model cannot handle it, even with exponential number of nodes n α , while a nonlinear one with ReLU can.

Definition 1 (Convex Hull of a Set).

We define the convex hull Conv(P ) of m points P ⊂ R n to be Conv(P ) = P a, a ∈ ∆ n−1 , where DISPLAYFORM0 ∈ Conv(P \p j ).

Definition 2.

A matrix P of size m-by-n is called k-vert, or vert(P ) = k ≤ m, if its k rows are vertices of the convex hull generated by its rows.

P is called all-vert if k = m. Theorem 6 (Expressibility of ReLU Nonlinearity).

Assuming m α = n α = O(exp(sz(α))), where sz(α) is the size of receptive field of α.

If each P αβ is all-vert, then: (ω is top-level receptive field) DISPLAYFORM1 Note that here Loss(W ) ≡ F ω − I 2 F .

This shows the power of nonlinearity, which guarantees full rank of output, even if the matrices involved in the multiplication are low-rank.

The following theorem shows that for intermediate layers whose input is not identity, the all-vert property remains.

DISPLAYFORM2 This means that if all P αβ are all-vert and its input F β is full-rank, then with the same construction of Thm.

6, F α can be made identity.

In particular, if we sample W randomly, then with probability 1, all F β are full-rank, in particular the top-level input F 1 .

Therefore, using top-level W 1 alone would be sufficient to yield zero generalization error, as shown in the previous works that random projection could work well.

The analysis in Sec. 5.1 assumes that n α = m α , which means that we have sufficient nodes, one neuron for one event, to convey the information forward to the classification level.

In practice, this is never the case.

When n α m α = O(exp(sz(α))) and the network needs to represent the information in a proper way so that it can be sent to the top level.

Ideally, if the factor z α can be written down as a list of binary factors: DISPLAYFORM0 , the output of a node j could represent z α [j] , so that all m α events can be represented concisely with n α nodes.

DISPLAYFORM1 The j-th binary factor of region α.

z α[j] can take 0 or 1.

DISPLAYFORM2 2-by-1 marginal probability vector of binary factor DISPLAYFORM3 The j-th column of F α , G α andG α corresponding to j-th binary factor z α [j] .1 / 0 All-1 / All-0 vector.

Its dimension depends on context.

DISPLAYFORM4 Out (or tensor) product of F 1 and DISPLAYFORM5 are the indices of downstream nodes in β to i-th binary factor in α FIG3 ).

DISPLAYFORM6 The j-th subcolumn of weight matrix W βα , whose rows are selected by S αβ j .

To come up with a complete theory for disentangled representation in deep nonlinear network is far from trivial and beyond the scope of this paper.

In the following, we make an initial attempt by constructing factorizable P αβ so that disentangled representation is possible in the forward pass.

First we need to formally define what is disentangled representation: DISPLAYFORM7 and 1 is a 2-by-1 vector.

Definition 4.

The gradientG α is disentangled, if its j-th columnG α,: DISPLAYFORM8 is a 2-by-1 vector.

Intuitively, this means that each node j represents the binary factor z α [j] .

A follow-up question is whether such disentangled properties carries over layers in the forward pass.

It turns out that the disentangled structure carries if the data distribution and weights have compatible structures:Definition 5.

The weights W βα is separable with respect to a disjoint set {S DISPLAYFORM9 Theorem 8 (Disentangled Forward).

If for each β ∈ ch(α), P αβ can be written as a tensor product DISPLAYFORM10 where {S αβ i } are αβ-dependent disjointed set, W βα is separable with respect to {S αβ i }, F β is disentangled, then F α is also disentangled (with/without ReLU /Batch Norm).

If the bottom activations are disentangled, by induction, all activations will be disentangled.

The next question is whether gradient descent preserves such a structure.

The answer is also conditionally yes: DISPLAYFORM11 , F β andG α are both disentangled, 1 TG α = 0, then the gradient update ∆W βα is separable with respect to {S i }.

Therefore, with disentangled F β andG α and centered gradient 1 TG α = 0, the separable structure is conserved over gradient descent, given the initial W (0) βα is separable.

Note that centered gradient is guaranteed if we insert Batch Norm (Eqn.

83) after linear layers.

And the activation F remains disentangled if the weights are separable.

The hard part is whetherG β remains disentangled during backpropagation, if {G α } α∈pa(β) are all disentangled.

If so, then the disentangled representation is self-sustainable under gradient descent.

This is a non-trivial problem and generally requires structures of data distribution.

We put some discussion in the Appendix and leave this topic for future work.

In the proposed formulation, the input x in Eqn.

7 is integrated out, and the data distribution is now encoded into the probabilistic distribution P(z α , z β ), and their marginals.

A change of such distribution means the input distribution has changed.

For the first time, we can now analyze many practical factors and behaviors in the DL training that is traditionally not included in the formulation.

Over-fitting.

Given finite number of training samples, there is always error in estimated factor-factor distributionP(z α , z β ) and factor-observation distributionP(x α |z α ).

In some cases, a slight change of distribution would drastically change the optimal weights for prediction, which is overfitting.

is a noisy factor.

Here is one example.

Suppose there are two different kinds of events at two disjoint reception fields: z α and z γ .

The class label is z ω , which equals z α but is not related to z γ .

Therefore, we have: DISPLAYFORM0 Although z γ is unrelated to the class label z ω , with finite samples z γ could show spurious correlation: DISPLAYFORM1 On the other hand, as shown in Fig. 5 , P(x α |z α ) contains a lot of detailed structures and is almost impossible to separate in the finite sample case, while P(x γ |z γ ) could be well separated for z γ = 0/1.

Therefore, for node j with rf(j) = α, f j (z α ) ≈ constant (input almost indistinguishable): DISPLAYFORM2 where DISPLAYFORM3 , which is a strong gradient signal backpropagated from the top softmax level, since z α is strongly correlated with z ω .

For node k with rf(k) = γ, an easy separation of the input (e.g., random initialization) yields distinctive f k (z γ ).

Therefore, DISPLAYFORM4 where g 0 (z γ ) = E zω|zγ [g 0 (z ω )] = 2 z γ = 1 −2 z γ = 0 , a weak signal because of z γ is (almost) unrelated to the label.

Therefore, we see that the weight w j that links to meaningful receptive field z α does not receive strong gradient, while the weight w k that links to irrelevant (but spurious) receptive field z γ receives strong gradient.

This will lead to overfitting.

With more data, over-fitting is alleviated since (1)P(z ω |z γ ) becomes more accurate and → 0; (2) P(x α |z α ) starts to show statistical difference for z α = 0/1 and thus f j (z α ) shows distinctiveness.

Note that there exists a second explanation: we could argue that z γ is a true but weak factor that contributes to the label, while z α is a fictitious discriminative factor, since the appearance difference between z α = 0 and z α = 1 (i.e.,P(x α |z α ) for α = 0/1) could be purely due to noise and thus should be neglected.

With finite number of samples, these two cases are essentially indistinguishable.

Models with different induction bias might prefer one to the other, yielding drastically different generalization error.

For neural network, SGD prefers the second explanation but if under the pressure of training, it may also explore the first one by pushing gradient down to distinguish subtle difference in the input.

This may explain why the same neural networks can fit random-labeled data, and generalize well for real data BID16 .

Gradient Descent: Stochastic or not?

Previous works (Keskar et al., 2017) show that empirically stochastic gradient decent (SGD) with small batch size tends to converge to "flat" minima and offers better generalizable solution than those uses larger batches to compute the gradient.

From our framework, SGD update with small batch size is equivalent to using a perturbed/noisy version of P(z α , z β ) at each iteration.

Such an approach naturally reduces aforementioned over-fitting issues, which is due to hyper-sensitivity of data distribution and makes the final weight solution invariant to changes in P(z α , z β ), yielding a "flat" solution.

In this paper, we propose a novel theoretical framework for deep (multi-layered) nonlinear network with ReLU activation and local receptive fields.

The framework utilizes the specific structure of neural networks, and formulates input data distributions explicitly.

Compared to modeling deep models as non-convex problems, our framework reveals more structures of the network; compared to recent works that also take data distribution into considerations, our theoretical framework can model deep networks without imposing idealistic analytic distribution of data like Gaussian inputs or independent activations.

Besides, we also analyze regularization techniques like Batch Norm, depicts its underlying geometrical intuition, and shows that BN is compatible with our framework.

Using this novel framework, we have made an initial attempt to analyze many important and practical issues in deep models, and provides a novel perspective on overfitting, generalization, disentangled representation, etc.

We emphasize that in this work, we barely touch the surface of these core issues in deep learning.

As a future work, we aim to explore them in a deeper and more thorough manner, by using the powerful theoretical framework proposed in this paper.

We consider a neuron (or node)

j. Denote f j as its activation after nonlinearity and g j as the (input) gradient it receives after filtered by ReLU's gating.

Note that both f j and g j are deterministic functions of the input x and label y. Since y is a deterministic function of x, we can write f j = f j (x) and g j = g j (x).

Note that all analysis still holds with bias terms.

We omit them for brevity.

The activation f j and gradient g k can be written as (note that f j is the binary gating function): DISPLAYFORM0 And the weight update for gradient descent is: DISPLAYFORM1 Here is the expectation is with respect to a training dataset (or a batch), depending on whether GD or SGD has been used.

We also use f raw j and g raw j as the counterpart of f j and g j before nonlinearity.

Given the structure of locally connected network, the gradient g j has some nice structures.

From Eqn.

17 we knows that DISPLAYFORM0 .

Define x −k = x\x k as the input image x except for x k .

Then we can define the marginalized gradient: DISPLAYFORM1 as the marginalization (average) of x −k , while keep x k fixed.

With this notation, we can write DISPLAYFORM2 On the other hand, the gradient which back-propagates to a node k can be written as DISPLAYFORM3 where f k is the derivative of activation function of node k (for ReLU it is just a gating function).

If we take expectation with respect to x −k |x k on both side, we get DISPLAYFORM4 Note that all marginalized gradients g j (x k ) are independently computed by marginalizing with respect to all regions that are outside the receptive field x k .

Interestingly, there is a relationship between these gradients that respects the locality structure: Theorem 1 (Recursive Property of marginalized gradient).

DISPLAYFORM5 Proof.

We have: DISPLAYFORM6

Theorem 2 (Reformulation).

Denote α = rf(j) and β = rf(k).

k is a child of j. If the following two conditions hold:• Focus of knowledge.

P(x k |z α , z β ) = P(x k |z β ).•

Broadness of knowledge.

P(x j |z α , z β ) = P(x j |z α ).• Decorrelation.

Given z β , (g raw k (·) and f k (·)) and (f raw k (·) and f k (·)) are uncorrelatedThen the following two conditions holds: DISPLAYFORM0 Proof.

For Eqn.

22a, we have: DISPLAYFORM1 And for each of the entry, we have: DISPLAYFORM2 For P(x k |z α ), using focus of knowledge, we have: DISPLAYFORM3 Therefore, following Eqn.

26, we have: DISPLAYFORM4 Putting it back to Eqn.

25 and we have: DISPLAYFORM5 For Eqn.

22b, similarly we have: DISPLAYFORM6 Notice that we have: DISPLAYFORM7 since x j covers x k which determines z β .

Therefore, for each item we have: DISPLAYFORM8 Then we use the broadness of knowledge: DISPLAYFORM9 DISPLAYFORM10 Following Eqn.

40, we now have: DISPLAYFORM11 DISPLAYFORM12 DISPLAYFORM13 Putting it back to Eqn.

36 and we have: DISPLAYFORM14 Using the definition of g k (z β ): DISPLAYFORM15 The un-correlation between g raw k (·) and f k (·) means that DISPLAYFORM16 Similarly for f j (z α ).

The following theorem shows that the reformulation is exact if z α has all information of the region.

Theorem 3.

If P(x j |z α ) is a delta function for all α, then the conditions of Thm.

2 hold and the reformulation becomes exact.

Proof.

The fact that P(x j |z α ) is a delta function means that there exists a function φ j so that: DISPLAYFORM0 That is, z α contains all information of x j (or x α ).

Therefore,• Broadness of knowledge.

z α contains strictly more information than z β for β ∈ ch(α), therefore P(x j |z α , z β ) = P(x j |z α ).•

Focus of knowledge.

z β captures all information of z k , so P(x k |z α , z β ) = P(x k |z β ).• Decorrelation.

For any h 1 (x j ) and h 2 (x j ) we have DISPLAYFORM1

Theorem 4 (Matrix Representation of Reformulation).

DISPLAYFORM0 n β -by-n α Weight matrix that links group β and α.

P αβ , P b αβ m α -by-m β Prob P(z β |z α ), P(z α |z β ) of events between group β and α.

Λ α m α -by-m α Diagonal matrix encoding prior prob P(z α ).

Proof.

We first consider one certain group α and β, which uses x α and x β as the receptive field.

For this pair, we can write Eqn.

22 in the following matrix form: DISPLAYFORM1 we could simplify Eqn.

60 as follows: DISPLAYFORM2 Therefore, using the fact that j∈pa(k) = α∈pa(β) j∈α (where β = rf(k)) and k∈ch(j) = β∈ch(α) k∈β (where α = rf(j)), and group all nodes that share the receptive field together, we have: DISPLAYFORM3 For the gradient update rule, from Eqn.

17 notice that: DISPLAYFORM4 We assume decorrelation so we have: DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 , again we use focus of knowledge: DISPLAYFORM8 Put them together and we have: DISPLAYFORM9 Write it in concise matrix form and we get: DISPLAYFORM10

Theorem 5 (Backpropagation of Batch Norm).

For a top-down gradient g, BN layer gives the following gradient update (P ⊥ f ,1 is the orthogonal complementary projection of subspace {f , 1}): DISPLAYFORM0 Proof.

We denote pre-batchnorm activations as DISPLAYFORM1 whitened to bef (i) , then linearly transformed to yield the output f DISPLAYFORM2 DISPLAYFORM3 2 and c 1 , c 0 are learnable parameters.

While in the original batch norm paper, the weight update rules are super complicated and unintuitive (listed here for a reference):Figure 7: Original BN rule from (Ioffe & Szegedy, 2015) .It turns out that with vector notation, the update equations have a compact vector form with clear geometric meaning.

To achieve that, we first write down the vector form of forward pass of batch normalization: DISPLAYFORM4 where f ,f ,f and f bn are vectors of size N , P DISPLAYFORM5 is 2-by-2 identity matrix) and thus S(x) is an column-orthogonal N -by-2 matrix.

If we put everything together, then we have: DISPLAYFORM6 Using this notation, we can compute the Jacobian of batch normalization layer.

Specifically, for any vector f , we have: DISPLAYFORM7 where P ⊥ f projects a vector into the orthogonal complementary space of f .

Therefore we have: DISPLAYFORM8 where DISPLAYFORM9 is a symmetric projection matrix that projects the input gradient to the orthogonal complement space spanned byx and 1 FIG1 ).

Note that the space spanned byf and 1 is also the space spanned by f and 1, sincef = (f − µ1)/σ can be represented linearly by f and 1.

DISPLAYFORM10 An interesting property is that since f bn returns a vector in the subspace of f and 1, for the N -by-N Jacobian matrix of Batch Normalization, we have: DISPLAYFORM11 Following the backpropagation rule, we get the following gradient update for batch normalization.

If g bn = ∂L/∂f is the gradient from top, then DISPLAYFORM12 Therefore, any gradient (vector of size N ) that is back-propagated to the input of BN layer will be automatically orthogonal to that activation (which is also a vector of size N ).

The analysis of Batch Norm is compatible with the reformulation and we arrive at similar backpropagation rule, by noticing that DISPLAYFORM0 Note that we still have the projection property, but under the new inner product f j , g j zα = DISPLAYFORM1 zα .

One can find an interesting quantity, by multiplying g j (x) on both side of the forward equation in Eqn.

16 and taking expectation: DISPLAYFORM0 Using the language of differential equation, we know that: DISPLAYFORM1 where DISPLAYFORM2 If we place Batch Normalization layer just after ReLU activation and linear layer, by BN property, since E x [g j f j ] ≡ 0 for all iterations, the row energy E j (t) of weight matrix W of the linear layer is conserved over time.

This might be part of the reason why BN helps stabilize the training.

Otherwise energy might "leak" from one layer to nearby layers.

With the help of the theoretical framework, we now can analyze interesting structures of gradient descent in deep models, when the data distribution P(z α , z β ) satisfies specific conditions.

Here we give two concrete examples: the role played by nonlinearity and in which condition disentangled representation can be achieved.

Besides, from the theoretical framework, we also give general comments on multiple issues (e.g., overfitting, GD versus SGD) in deep learning.

In the formulation, m α is the number of possible events within a region α, which is often exponential with respect to the size sz(α) of the region.

The following analysis shows that a linear model cannot handle it, even with exponential number of nodes n α , while a nonlinear one with ReLU can.

Definition 1 (Convex Hull of a Set).

We define the convex hull Conv(P ) of m points P ⊂ R n to be Conv(P ) = P a, a ∈ ∆ n−1 , where DISPLAYFORM0 ∈ Conv(P \p j ).

Definition 2.

A matrix P of size m-by-n is called k-vert, or vert(P ) = k ≤ m, if its k rows are vertices of the convex hull generated by its rows.

P is called all-vert if k = m. Theorem 6 (Expressibility of ReLU Nonlinearity).

Assuming m α = n α = O(exp(sz(α))), where sz(α) is the size of receptive field of α.

If each P αβ is all-vert, then: (ω is top-level receptive field) DISPLAYFORM1 Here we define DISPLAYFORM2 Proof.

We prove that in the case of nonlinearity, there exists a weight so that the activation F α = I for all α.

We prove by induction.

The base case is trivial since we already know that F α = I for all leaf regions.

Suppose F β = I for any β ∈ ch(α).

Since P αβ is all-vert, every row is a vertex of the convex hull, which means that for i-th row p i , there exists a weight w i and b i so that w DISPLAYFORM3 Put these weights and biases together into W βα and we have DISPLAYFORM4 All diagonal elements of F raw α are 1 while all off-diagonal elements are negative.

Therefore, after ReLU, F α = I. Applying induction, we get F ω = I and G ω = I − F ω = 0.

Therefore, DISPLAYFORM5 In the linear case, we know that rank(F α ) ≤ β rank(P αβ F β W βα ) ≤ β rank(F β ), which is on the order of the size sz(α) of α's receptive field (Note that the constant relies on the overlap between receptive fields).

However, at the top-level, m ω = n ω = O(exp(sz(ω))), i.e., the information contained in α is exponential with respect to the size of the receptive field.

By Eckart-Young-Mirsky theorem, we know that there is a lower bound for low-rank approximation.

Therefore, the loss for linear network Loss linear is at least on the order of m 0 , i.e., Loss linear = O(m ω ).

Note that this also works if we have BN layer in-between, since BN does a linear transform in the forward pass.

This shows the power of nonlinearity, which guarantees full rank of output, even if the matrices involved in the multiplication are low-rank.

The following theorem shows that for intermediate layers whose input is not identity, the all-vert property remains.

Theorem 7.(1) If F is full row rank, then vert(P F ) = vert(P ).

(2) P F is all-vert iff P is all-vert.

Proof.

For (1), note that each row of P F is p T i F .

If F is row full rank, then F has pseudo-inverse F so that F F = I. Therefore, if p i is not a vertex: DISPLAYFORM6 then p T i F is also not a vertex and vice versa.

Therefore, vert(P F ) = vert(P ).

(2) follows from (1).This means that if all P αβ are all-vert and its input F β is full-rank, then with the same construction of Thm.

6, F α can be made identity.

In particular, if we sample W randomly, then with probability 1, all F β are full-rank, in particular the top-level input F 1 .

Therefore, using top-level W 1 alone would be sufficient to yield zero generalization error, as shown in the previous works that random projection could work well.

The analysis in the previous section assumes that n α = m α , which means that we have sufficient nodes, one neuron for one event, to convey the information forward to the classification level.

In practice, this is never the case.

When n α m α = O(exp(sz(α))) and the network needs to represent the information in a proper way so that it can be sent to the top level.

Ideally, if the factor z α can be written down as a list of binary factors: DISPLAYFORM0 , the output of a node j could represent z α [j] , so that all m α events can be represented concisely with n α nodes.

To come up with a complete theory for disentangled representation in deep nonlinear network is far from trivial and beyond the scope of this paper.

In the following, we make an initial attempt by constructing factorizable P αβ so that disentangled representation is possible in the forward pass.

First we need to formally define what is disentangled representation: Definition 3.

The activation F α is disentangled, if its j-th column If the bottom activations are disentangled, by induction, all activations should be disentangled.

The next question is whether gradient descent preserves such a structure.

Here we provide a few theorems to discuss such issues.

We first start with two lemmas.

Both of them have simple proofs.

Lemma 1.

Distribution representations have the following property: DISPLAYFORM1 α is also disentangled.(2) If F α is disentangled and h is any per-column element-wise function, then h(F α ) is disentangled.

DISPLAYFORM2 Proof.(1) follows from properties of tensor product.

For FORMULA7 and FORMULA9 , note that the j-th column of F α is F α,:j = 1⊗. . .

f j . .

.⊗1, therefore h j (F α,:j ) = 1⊗. . .

h j (f j ) . .

.⊗1, and h We have P Sj 1 = 1 and 1 T p α[j] = 1.

Note here for simplicity, 1 represents all-one vectors of any length, determined by the context.

Since F α and G β are disentangled, their j-th column can be written as: For simplicity, in the following proofs, we just show the case that n α = 2, n β = 3, z α = z α[1] , z α [2] and S = {S 1 , S 2 } = {{1, 2}, {3}}. We write f 1,2 = [f 1 ⊗ 1, 1 ⊗ f 2 ] as a 2-column matrix.

The general case is similar and we omit here for brevity.

Theorem 8 (Disentangled Forward).

If for each β ∈ ch(α), P αβ can be written as a tensor product DISPLAYFORM3 where {S αβ i } are αβ-dependent disjointed set, W βα is separable with respect to {S αβ i }, F β is disentangled, then F α is also disentangled (with/without ReLU /Batch Norm).Proof.

For a certain β ∈ ch(α), we first compute the quantity P αβ F β : P αβ F β = (P 1,2 ⊗ P 3 ) [f 1,2 ⊗ 1, 1 ⊗ f 3 ] = [P 1,2 f 1,2 ⊗ 1, 1 ⊗ P 3 f 3 ]

Therefore, the forward information sent from β to α is: One hope here is that if we consider α∈pa(β)G raw α→β , the summation over parent α could lead to a better structure, even for individual α, P .If each α ∈ pa(β) is informative in a diverse way, and |S 1 | is relatively small (e.g., 4), then v + α,S1 − v − α,S1 = 0 and spans the probability space of dimension 2 |S1| − 1.

Then we can always find c α (or equivalently, weights) so that Eqn.

109 becomes rank-1 tensor (or disentangled).

Besides, the gating D β , which is disentangled as it is an element-wise function of F β , will also play a role in regularizingG β .We will leave this part to future work.

@highlight

This paper presents a theoretical framework that models data distribution explicitly for deep and locally connected ReLU network