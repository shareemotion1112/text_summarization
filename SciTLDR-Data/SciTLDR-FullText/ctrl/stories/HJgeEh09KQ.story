We present a novel approach for the certification of neural networks against adversarial perturbations which combines scalable overapproximation methods with precise (mixed integer) linear programming.

This results in significantly better precision than state-of-the-art verifiers on challenging feedforward and convolutional neural networks with piecewise linear activation functions.

Neural networks are increasingly applied in critical domains such as autonomous driving BID1 , medical diagnosis BID0 , and speech recognition BID13 .

However, it has been shown by BID11 that neural networks can be vulnerable against adversarial attacks, i.e., imperceptible input perturbations cause neural networks to misclassify.

To address this challenge and prove that a network is free of adversarial examples (usually, in a region around a given input), recent work has started investigating the use of certification techniques.

Current verifiers can be broadly classified as either complete or incomplete.

Complete verifiers are exact, i.e., if the verifier fails to certify a network then the network is nonrobust (and vice-versa) .

Existing complete verifiers are based on Mixed Integer Linear Programming (MILP) BID19 BID8 BID5 BID3 or SMT solvers BID16 BID7 .

Although precise, these can only handle networks with a small number of layers and neurons.

To scale, incomplete verifiers usually employ overapproximation methods and hence they are sound but may fail to prove robustness even if it holds.

Incomplete verifiers use methods such as duality BID6 , abstract interpretation BID23 , linear approximations BID29 , semidefinite relaxations BID21 , combination of linear and non-linear approximation BID30 , or search space discretization BID14 .

Incomplete verifiers are more scalable than complete ones, but can suffer from precision loss for deeper networks.

In principle, incomplete verifiers can be made asymptotically complete by iteratively refining the input space BID26 or the neurons BID27 ; however, in the worst case, this may eliminate any scalability gains and thus defeat the purpose of using overapproximation in the first place.

This work: boosting complete and incomplete verifiers.

A key challenge then is to design a verifier which improves the precision of incomplete methods and the scalability of complete ones.

In this work, we make a step towards addressing this challenge based on two key ideas: (i) a combination of state-of-the-art overapproximation techniques used by incomplete methods, including LP relaxations, together with MILP solvers, often employed in complete verifiers; (ii) a novel heuristic, which points to neurons whose approximated bounds should be refined.

We implemented these ideas in a system called RefineZono, and showed that is is faster than state-of-the-art complete verifiers on small networks while improving precision of existing incomplete verifiers on larger networks.

The recent works of BID27 and BID25 have also explored the combination of linear programming with overapproximation.

However, both use simpler and coarser overapproximations than ours.

Our evaluation shows that RefineZono is faster than both for complete verification.

For example, RefineZono is faster than the work of BID25 for the complete x 7x 8 x 9x 10 x 11x 12x 13 Figure 1 : Robustness analysis of a toy example neural network using our method.

Here, approximation results computed with DeepZ (blue box) are refined using MILP whereas those in green are refined using LP.

verification of a 3 × 50 network, while for the larger 9 × 200 network their method does not finish within multiple days on images which RefineZono verifies in ≈ 14 minutes.

DISPLAYFORM0

• A refinement-based approach for certifying neural network robustness that combines the strengths of fast overapproximation methods with MILP solvers and LP relaxations.• A novel heuristic for selecting neurons whose bounds should be further refined.• A complete end-to-end implementation of our approach in a system called RefineZono, publicly available at https://github.com/eth-sri/eran.

• An evaluation, showing that RefineZono is more precise than existing state-of-the-art incomplete verifiers on larger networks and faster (while being complete) than complete verifiers on smaller networks.

We now demonstrate how our method improves the precision of a state-of-the-art incomplete verifier.

The main objective here is to provide an intuitive understanding of our approach; full formal details are provided in the next section.

Consider the simple fully connected feedforward neural network with ReLU activations shown in Fig. 1 .

There are two inputs to the network, both in the range [0, 1].

The network consists of an input layer, two hidden layers, and one output layer.

Each layer consist of two neurons each.

For our explanation, we separate each neuron into two parts: one represents the output of the affine transformation while the other captures the output of the ReLU activation.

The weights for the affine transformation are represented by weights on the edges.

The bias for each node is shown above or below it.

Our goal is to verify that for any input in [0, 1] × [0, 1], the output at neuron x 13 is greater than the output at x 14 .We now demonstrate how our verifier operates on this network.

We assume that the analysis results after the second affine transformation are refined using a MILP formulation of the network whereas the results after the third affine transformation are refined by an LP formulation of the network.

In DISPLAYFORM0 Figure 2: ReLU transformers, computing an affine form.

Here, l x , u x are the original bounds, whereas l x , u x are the refined bounds.

The slope of the two non-vertical parallel blue lines is λ = u x /(u x − l x ) and the slope of the two non-vertical parallel green lines is λ = u x /(u x − l x ).

The blue parallelogram is used to compute an affine form in DeepZ, whereas the green parallelogram is used to compute the output of the refined ReLU transformer considered in this work.the next section, we will explain our heuristic for selecting MILP or LP formulations of different neurons in the network.

Our analysis leverages the Zonotope domain BID10 ) together with the abstract Zonotope transformers specialized to neural network activations as used in DeepZ BID23 , a state of the art verifier for neural network robustness.

The Zonotope domain associates an affine formx with each neuron x in the network: DISPLAYFORM1 Here, c 0 , c i ∈ R are real coefficients and η i ∈ [s i , t i ] ⊆ [−1, 1] are the noise symbols, which are shared between the affine forms for different neurons.

This sharing makes the domain relational and thus more precise than non-relational domains such as Interval (Box).

An abstract element in our analysis is an intersection between a Zonotope (given as a list of affine forms) and a bounding box.

Thus, for each neuron x, we keep the affine formx and an interval [l x , u x ].First layer.

Our analysis starts by settinĝ DISPLAYFORM2 andx 2 = 0.5 + 0.5 · η 2 , l 2 = 0, u 2 = 1, representing the input [0, 1] at x 1 and [0, 1] at x 2 in our domain, respectively.

Next, an affine transformation is applied on the inputs resulting in the output DISPLAYFORM3 Note that the Zonotope affine transformer is exact for this transformation.

Next, the Zonotope ReLU transformer is applied.

We note that as l 3 ≥ 0, the neuron x 3 provably takes only non-negative values.

Thus, the ReLU Zonotope transformer outputsx 5 =x 3 and we set l 5 = l 3 , u 5 = l 3 which is the exact result.

For x 4 , l 4 < 0 and u 4 > 0 and thus neuron x 4 can take both positive and negative values.

The corresponding output does not have a closed affine form and hence the approximation in blue shown in Fig. 2 is used to compute the result.

This approximation minimizes the area of the result in the input-output plane and introduces a new noise symbol η 3 ∈ [−1, 1].

The result iŝ DISPLAYFORM4 Note that the Zonotope approximation for x 6 from Fig. 2 permits negative values whereas x 6 can only take non-negative values in the concrete.

This overapproximation typically accumulates as the analysis progresses deeper into the network, resulting in overall imprecision and failure to prove properties that actually hold.

MILP-based refinement at second layer.

Next, the analysis handles the second affine transformation and computeŝ DISPLAYFORM5 Here, x 7 is provably positive, whereas x 8 can take both positive and negative values.

Due to the approximation for x 6 , the bounds for x 7 and x 8 are imprecise.

Note that the DeepZ ReLU transformer for x 8 applied next will introduce more imprecision and although the ReLU transformer for provably positive inputs such as x 7 does not lose precision with respect to the input, it still propagates the imprecision in the computation of the abstract values for x 7 .Thus, to reduce precision loss, in our method we refine the bounds for both x 7 and x 8 by formulating the network up to (and including) the second affine transformation as a MILP instance based on a formulation from BID25 and compute bounds for x 7 and x 8 , respectively.

The MILP solver improves the lower bounds for x 7 and x 8 to 1 and −1, respectively, which then updates the corresponding lower bounds in our abstraction, i.e., l 7 = 1 and l 8 = −1.Next, the ReLU transformer is applied.

Since x 7 is provably positive, we getx 9 =x 7 , l 9 = l 7 , and u 9 = u 7 .

We note that x 8 can take both positive and negative values and is therefore approximated.

However, the ReLU transformer now uses the refined bounds instead of the original bounds and thus the approximation shown in green from Fig. 2 is used.

This approximation has smaller area in the input-output plane compared to the blue one and thus reduces the approximation error.

The result iŝ DISPLAYFORM6 LP-based refinement at final layer.

Continuing with the analysis, we now process the final affine transformation, which yieldŝ Due to the approximations from previous layers, the computed values can be imprecise.

We note that, as the analysis proceeds deeper into the network, refining bounds with MILP becomes expensive.

Thus, we refine the bounds by encoding the network up to (and including) the third affine transformation using the faster LP relaxation of the network based on BID7 and compute the bounds for x 11 and x 12 , respectively.

This leads to better results for l 11 = 3.25, l 12 = 2, and u 12 = 3.

As both x 11 and x 12 are provably positive, the subsequent ReLU transformations set x 13 =x 11 , l 13 = l 11 , u 13 = u 11 andx 14 =x 12 , l 14 = l 12 , u 14 = u 12 .Proving robustness.

Since the lower bound l 13 for x 13 is greater than the upper bound u 14 for x 14 , our analysis can prove that the given neural network provides the same label for all inputs in [0, 1] × [0, 1] and is thus robust.

In contrast, DeepZ without our refinement would computê x 13 = 4.95 + 0.6 · η 1 + 1.8 · η 2 − 0.6 · η 3 + 0.3 · η 4 , l 13 = 1.65, u 13 = 8.25 andx 14 = 2.55 + 0.15 · η 1 + 0.45 · η 2 − 0.15 · η 3 − 0.3 · η 4 , l 14 = 1.5, u 14 = 3.6.

As a result, DeepZ fails to prove that x 13 is greater than x 14 , and thus fails to prove robustness.

Generalization to other abstractions.

We note that our refinement-based approach is not restricted to the Zonotope domain.

It can be extended for refining the results computed by other abstractions such as Polyhedra BID22 or the abstraction used in DeepPoly BID24 .

For example, the ReLU transformer in BID24 also depends on the bounds of input neurons and thus it will benefit from the precise bounds computed using our refinement.

Since DeepPoly often produces more precise results than DeepZ, we believe a combination of this work with DeepPoly will further improve verification results.

We now describe our approach in more formal terms.

As in the previous section, we will consider affine transformations and ReLU activations as separate layers.

As illustrated earlier, the key idea will be to combine abstract interpretation BID4 with exact and inexact MILP formulations of the network, which are then solved, in order to compute more precise results for neuron bounds.

We begin by describing the core ingredients of abstract interpretation.

Our approach requires an abstract domain A n over n variables (i.e., some set whose elements can be encoded symbolically) such as Interval, Zonotope, the abstraction in DeepPoly, or Polyhedra.

An abstract domain has a bottom element ⊥ ∈ A n as well as the following components:• A (potentially non-computable) concretization function γ n : A n → P(R n ) that associates with each abstract element a ∈ A n the set of concrete points from R n that it abstracts.

We have γ n (⊥) = ∅.• An abstraction function α n : B n →

A n , where X ⊆

γ n (α n (X)) for all X ∈ B n .

We assume DISPLAYFORM0 (For many abstract domains, α n can be defined on a larger domain B n , but in this work, we only consider Interval input regions.)

DISPLAYFORM1 • A meet operation a L for each a ∈ A n and linear constraints L over n real variables, where DISPLAYFORM2 • An affine abstract transformer DISPLAYFORM3 • A ReLU abstract transformer T DISPLAYFORM4 for all abstract elements a ∈ A n and for all lower and upper bounds l, u ∈ R n on input activations of the ReLU operation.

Verification via Abstract interpretation.

As first shown by , any such abstract domain induces a method for robustness certification of neural networks with ReLU activations.

For example, assume that we want to certify that a given neural network f : R m → R n considers class i more likely than class j for all inputsx with ||x − x|| ∞ ≤ for a given x and .

We can first use the abstraction function α m to compute a symbolic overapproximation of the set of possible inputsx, namely DISPLAYFORM5 Given that the neural network can be written as a composition of affine functions and ReLU layers, we can then propagate the abstract element a in through the corresponding abstract transformers to obtain a symbolic overapproximation a out of the concrete outputs of the neural network.

For example, if the neural network f (x) = A · ReLU(Ax + b) + b has a single hidden layer with h hidden neurons, we first compute a = T # x →Ax+b (a in ), which is a symbolic overapproximation of the input to the ReLU activation function.

We then compute (l, u) = ι h (a ) to obtain opposite corners of a bounding box of all possible ReLU input activations, such that we can apply the ReLU abstract transformer: DISPLAYFORM6 Finally, we apply the affine abstract transformer again to obtain a out = T # x →A x+b (a ).

Using our assumptions, we can conclude that the set γ n (a out ) contains all output activations that f can possibly produce when given any of the inputsx.

Therefore, if a out (x i ≤ x j ) = ⊥, we have proved the property: for allx, the neural network considers class i more likely than class j.

Incompleteness.

While this approach is sound (i.e., whenever we prove the property, it actually holds), it is incomplete (i.e., we might not prove the property, even if it holds), because the abstract transformers produce a superset of the set of concrete outputs that the corresponding concrete executions produce.

This can be quite imprecise for deep neural networks, because the overapproximations introduced in each layer accumulate.

Refining the bounds.

To combat spurious overapproximation, we use mixed integer linear programming (MILP) to compute refined lower and upper bounds l , u after applying each affine abstract transformer (except for the first layer).

We then refine the abstract element using the meet operator of the underlying abstract domain and the linear constraints l i ≤ x i ≤ u i for all input activations i, i.e., we replace the current abstract element a by a = a ( i l i ≤ x i ≤ u i ), and continue analysis with the refined abstract element.

Importantly, we obtain a more refined abstract transformer for ReLU than the one used in DeepZ by leveraging the new lower and upper bounds.

That is, using the tighter bounds l x , u x for x, we define the ReLU transformer for y := max(0, x) as follows: DISPLAYFORM7 , and new ∈ [−1, 1] is a new noise symbol.

The refined ReLU transformer benefits from the improved bounds.

For example, when l x < 0 and u x > 0 holds for the original bounds then after refinement:• If l x > 0, then the output is the same as the input and no overapproximation is added.• Else if u x ≤ 0, then the output is exact.• Otherwise, as shown in Fig. 2 , the approximation with the tighter l x and u x has smaller area in the input-output plane than the original transformer that uses the imprecise l x and u x .Obtaining constraints for refinement.

To enable refinement with MILP, we need to obtain constraints which fully capture the behavior of the neural network up to the last layer whose abstract transformer has been executed.

In our encoding, we have one variable for each neuron and we write x (k) i to denote the variable corresponding to the activation of the i-th neuron in the k-th layer, where the input layer has k = 0.

Similarly, we write l From the input layer, we obtain constraints of the form l DISPLAYFORM8 , from affine layers, we obtain constraints of the form x DISPLAYFORM9 and from ReLU layers we obtain constraints of the form x DISPLAYFORM10 MILP.

Let ϕ (k) denote the conjunction of all constraints up to and including those from layer k. To obtain the best possible lower and upper bounds for layer k with p neurons, we need to solve the following 2 · p optimization problems: DISPLAYFORM11 i , for i = 1, . . .

, p.

As was shown by BID25 , such optimization problems can be encoded exactly as MILP instances using the bounds computed by abstract interpretation and the instances can then be solved using off-the-shelf MILP solvers to compute l DISPLAYFORM12 Published as a conference paper at ICLR 2019 LP relaxation.

While not introducing any approximation, unfortunately, current MILP solvers do not scale to larger neural networks.

It becomes increasingly more expensive to refine bounds with the MILP-based formulation as the analysis proceeds deeper into the network.

However, for soundness it is not crucial that the produced bounds are the best possible: for example, plain abstract interpretation uses sound bounds produced by the bounding box function ι instead.

Therefore, for deeper layers in the network, we explore the trade-off between precision and scalability by also considering an intermediate method, which is faster than exact MILP, but also more precise than abstract interpretation.

We relax the constraints in ϕ (k) using the bounds computed by abstract interpretation in the same way as Ehlers (2017) to obtain a set of weaker linear constraints ϕ DISPLAYFORM13 LP .

We then use the solver to solve the relaxed optimization problems that are constrained by ϕ (k) LP instead of ϕ (k) , producing possibly looser bounds l (k) and u (k) .

Note that the encoding of subsequent layers depends on the bounds computed in previous layers, where tighter bounds reduce the amount of newly introduced approximation.

Anytime MILP relaxation.

MILP solvers usually provide the option to provide an explicit timeout after which the solver must terminate.

In return, the solver may not be able to solve the instance exactly, but it will instead provide lower and upper bounds on the objective function in a best-effort fashion.

This provides another way to compute sound but inexact bounds l (k) and u (k) .In practice, we choose a fraction θ ∈ (0, 1] of neurons in a given layer k and compute bounds for them using MILP with a timeout T in a first step.

In the second step, for a fraction δ ∈ [0, 1 − θ] of neurons in the layer, we set the timeout to β · T , where T is the average time taken by the MILP solver to solve one of the instances from the first step and β ∈ [0, 1] is a parameter.

Neuron selection heuristic.

To select the θ-fraction of neurons for the first step of the anytime MILP relaxation for the k-th layer, we rank the neurons.

If the next layer is a ReLU layer, we first ignore all neurons whose activations can be proven to be non-positive using abstract interpretation (i.e., using the bounds produced by ι), because in this case it is already known that ReLU will map the activation to 0.

The remaining neurons are ordered in up to two different ways, once by width (i.e. neuron i has key u DISPLAYFORM14 i ), and possibly once by the sum of absolute output weights.

i.e., if the next layer is a fully connected layer x →

Ax + b, the key of neuron i is j |A i,j |.

If the next layer is a ReLU layer, we skip the ReLU layer and use the weights from the fully connected layer that follows it (if any).

The two ranks of a neuron in both orders are added, and the θ-fraction with smallest rank sum is selected and their bounds are refined with a timeout of T whereas the next δ-fraction of neurons are refined with a timeout of β · T .RefineZono: end-to-end approach.

To certify robustness of deep neural networks, we combine MILP, LP relaxation, and abstract interpretation.

We first pick numbers of layers k MILP , k LP , k AI that sum to the total number of layers of the neural network.

For the analysis of the first k MILP layers, we refine bounds using anytime MILP relaxation with the neuron selection heuristic.

As an optimization, we do not perform refinement after the abstract transformer for the first layer in case it is an affine transformation, as the abstract domain computes the tightest possible bounding box for an affine transformation of a box (this is always the case in our experiments).

For the next k LP layers, we refine bounds using LP relaxation (i.e., the network up to the layer to be refined is encoded using linear constraints) combined with the neuron selection heuristic.

For the remaining k AI layers, we use abstract interpretation without additional refinement (however, this also benefits from refinement that was performed in previous layers), and compute the bounds using ι.

Final property certification.

Let k be the index of the last layer and p be the number of output classes.

We can encode the final certification problem using the output abstract element a out obtained after applying the abstract transformer for the last layer in the network.

If we want to prove that class i is assigned a higher probability than class j, it suffices to show that a out (x DISPLAYFORM15 If this fails, one can resort to complete verification using MILP: the property is satisfied if and only if the set of constraints DISPLAYFORM16

We evaluate the effectiveness of our approach for the robustness verification of ReLU-based feedforward and convolutional neural networks.

The results show that our approach enables faster complete verification than the state-of-the-art complete verifiers: BID27 and BID25 , and produces more precise results than state-of-the-art incomplete verifiers: DeepZ BID23 and DeepPoly BID24 , when complete certification becomes infeasible.

We implemented our approach in a system called RefineZono.

RefineZono uses Gurobi (Gurobi Optimization, LLC, 2018) for solving MILP and LP instances and is built on top of the ELINA library (eli, 2018; BID22 for numerical abstract domains.

All of our code, neural networks, and images used in our experiments are publicly available at https://github.com/eth-sri/eran.Evaluation datasets.

We used the popular MNIST BID18 , CIFAR10 BID17 , and ACAS Xu BID15 ) datasets in our experiments.

MNIST contains grayscale images of size 28 × 28 pixels whereas CIFAR10 contains RGB images of size 32 × 32.

ACAS Xu contains 5 inputs representing aircraft sensor data.

Neural networks.

TAB0 shows 12 different MNIST, CIFAR10, and ACAS Xu feedforward (FNNs) and convolutional networks (CNNs) with ReLU activations used in our experiments.

Out of these 4 were trained to be robust against adversarial attacks using DiffAI whereas the remaining 8 had no adversarial training.

The largest network in our experiments contains > 88K neurons whereas the deepest network contains 9 layers.

Robustness properties.

For MNIST and CIFAR10, we consider the L ∞ -norm BID2 based adversarial region parameterized by ∈ R. Our goal here is to certify that the network produces the correct label on all points in the adversarial region.

For ACAS Xu, our goal is to verify that the property φ 9 BID16 holds for the 6 × 50 network (known to be hard).Experimental setup.

All experiments for the 3 × 50 MNIST FNN and all CNNs were carried out on a 2.6 GHz 14 core Intel Xeon CPU E5-2690 with 512 GB of main memory; the remaining FNNs were evaluated on a 3.3 GHz 10 Core Intel i9-7900X Skylake CPU with a main memory of 64 GB.Benchmarks.

For each MNIST and CIFAR10 network, we selected the first 100 images from the respective test set and filtered out those images that were not classified correctly.

We consider complete certification with RefineZono on the ACAS Xu network and the 3 × 50 MNIST network.

For the 3 × 50 network, we choose an for which the incomplete verifier DeepZ certified < 40% of all candidate images.

We consider incomplete certification for the remaining networks and choose an for which complete certification with RefineZono becomes infeasible.

RefineZono first runs DeepZ analysis on the whole network collecting the bounds for all neurons in the network.

If DeepZ fails to certify the network, then the collected bounds are used to encode the robustness certification as a MILP instance (discussed in section 3).ACAS Xu 6×50 network.

As this network has only 5 inputs, we uniformly split the pre-condition defined by φ 9 to produce 6 300 smaller input regions.

We certify that the post-condition defined by φ 9 holds for each region with RefineZono.

RefineZono certifies that φ 9 holds for the network in 227 seconds which is > 4x faster than the fastest verifier for ACAS Xu from BID27 .MNIST 3 × 50 network.

We use = 0.03 for the L ∞ -norm attack.

We compare RefineZono against the state-of-the-art complete verifier for MNIST from BID25 .

This approach is also MILP-based like ours, but it uses Interval analysis and LP to determine neuron bounds.

We implemented the Interval analysis and LP-based analysis to determine the initial bounds.

We call the MILP solver only if LP analysis (or Interval analysis) fails to certify.

All complete verifiers certify the neural network to be robust against L ∞ -norm perturbations on 85% of the images.

The average runtime of RefineZono, MILP with bounds from the Interval analysis, and MILP with bounds from the LP analysis are 28, 123, and 35 seconds respectively.

Based on our result, we believe that the Zonotope analysis offers a good middle ground between the speed of the Interval analysis and the precision of LP for bound computation, as it produces precise bounds faster than LP.

We next compare RefineZono against DeepZ and DeepPoly for the incomplete robustness certification of the remaining networks.

We note that DeepZ has the same precision as Fast-Lin and DeepPoly has the same precision as CROWN .

The values used for the L ∞ -norm attack are shown in TAB1 .

The values for networks trained to be robust are larger than for networks that are not.

For each verifier, we report the average runtime per image in seconds and the precision measured by the % of images for which the verifier certified the network to be robust.

We note that running the Interval analysis to obtain initial bounds is too imprecise for these large networks with the values considered in our experiments.

As a result, the approach from BID25 has to rely on applying LP per neuron to obtain precise bounds for the MILP solver which does not scale.

For example, on the 9 × 200 network, determining bounds with LP already takes > 20 minutes (without calling the MILP solver which is more expensive than LP) whereas RefineZono has an average running time of ≈ 14 minutes.

Parameter values.

We experimented with different values of the analysis parameters k MILP , k LP , k AI , θ, δ, β, T and chose values that offered the best tradeoff between performance and precision for the certification of each neural network.

We refine the neuron bounds after all affine transformations that are followed by a ReLU except the first one.

In a given layer, we consider all neurons that can take positive values after the affine transformation as refinement candidates.

For the MNIST FNNs, we refine the bounds of the candidate neurons in layers 2-4 with MILP and those in the remaining layers using LP.

For MILP based refinement, we use θ = DISPLAYFORM0 where ω is the number of candidates and p is the total number of neurons in layer k. For LP based refinement, we use θ = ω 2 k−5 ·p .

We use timeout T = 1 second, β = 0.5, and δ = ω p − θ for both MILP and LP based refinements.

For the CIFAR10 FNN, we use the same values except that we use θ = ω 2 k−2 ·p for MILP refinement and set T = 6 seconds for both MILP and LP based refinement as it is more expensive to refine neuron bounds in CIFAR10 networks due to these having more input neurons.

For the CNNs, the convolutional layers have large number of candidates so we do not refine these.

Instead, we refine all candidates in the fully connected layers with a larger timeout so to compensate for the more difficult problem instances for the solver.

For the MNIST ConvSmall, ConvBig and CIFAR10 ConvSmall networks, we refine all the candidate neurons using MILP with T = 10 seconds.

For the MNIST ConvSuper network, we refine similarly but use LP with T = 15 seconds.

Results for incomplete certification.

TAB1 shows the precision and the average runtime of all three verifiers.

RefineZono either improves or achieves the precision of the state-of-the-art verifiers on all neural networks.

It certifies more images than DeepZ on all networks except the MNIST ConvSuper network.

This is because DeepZ is already very precise for the considered.

We could not try larger for this network, as the DeepZ analysis becomes too expensive.

RefineZono certifies the network to be more robust on more images than DeepPoly on 6 out of 10 networks.

It can be seen that the number of neurons in the network is not the determining factor for the average runtime of RefineZono.

We observe that RefineZono runs faster on the networks trained to be robust and the top three networks with the largest runtime for RefineZono are all networks not trained to be robust.

This is because robust networks are relatively easier to certify and produce only a small number of candidate neurons for refinement, which are easier to refine by the solver.

For example, even though the same parameter values are used for refining the results on the MNIST ConvSmall and ConvBig networks, the average runtime of RefineZono on the robustly trained ConvBig network with ≈ 35K neurons, 6 layers and a perturbation region defined using = 0.2 is almost 4 times less than on the non-robust ConvSmall network with only 3 604 neurons, 3 layers and a smaller = 0.12.

We use the neuron selection heuristic from section 3 to determine neurons which need to be refined more than others for FNNs, as refining all neurons in a layer with MILP can significantly slow down the analysis.

To check whether our heuristic can identify important neurons, we ran the analysis on the MNIST 9 × 200 FNN by keeping all analysis parameters the same, except instead of selecting the neurons with the smallest rank sum first we selected the neurons with the largest rank sum first (thus refining neurons more if our heuristic deems them unimportant).

With this change, the average runtime does not change significantly.

However, the modified analysis loses precision and fails to certify two images that the analysis refining with our neuron selection heuristic succeeds on.

We presented a novel refinement-based approach for effectively combining overapproximation techniques used by incomplete verifiers with linear-programming-based methods used in complete verifiers.

We implemented our method in a system called RefineZono and showed its effectiveness on verification tasks involving feedforward and convolutional neural networks with ReLU activations.

Our evaluation demonstrates that RefineZono can certify robustness properties beyond the reach of existing state-of-the-art complete verifiers (these can fail due to scalability issues) while simultaneously improving on the precision of existing incomplete verifiers (which can fail due to using too coarse of an overapproximation).Overall, we believe combining the strengths of overapproximation methods with those of mixed integer linear programming as done in this work is a promising direction for further advancing the state-of-the-art in neural network verification.

<|TLDR|>

@highlight

We refine the over-approximation results from incomplete verifiers using MILP solvers to prove more robustness properties than state-of-the-art. 

@highlight

Introduces a verifier that obtains improvement on precision of incomplete verifiers and scalability of the complete verifiers using over-parameterization, mixed integer linear programming and linear programming relaxation.

@highlight

A mixed strategy to obtain better precision on robustness verifications of feed-forward neural networks with piecewise linear activation functions, achieving better precision than incomplete verifiers and more scalability than complete verifiers.