Neural networks trained only to optimize for training accuracy can often be fooled by adversarial examples --- slightly perturbed inputs misclassified with high confidence.

Verification of networks enables us to gauge their vulnerability to such adversarial examples.

We formulate verification of piecewise-linear neural networks as a mixed integer program.

On a representative task of finding minimum adversarial distortions, our verifier is two to three orders of magnitude quicker than the state-of-the-art.

We achieve this computational speedup via tight formulations for non-linearities, as well as a novel presolve algorithm that makes full use of all information available.

The computational speedup allows us to verify properties on convolutional and residual networks with over 100,000 ReLUs --- several orders of magnitude more than networks previously verified by any complete verifier.

In particular, we determine for the first time the exact adversarial accuracy of an MNIST classifier to perturbations with bounded l-∞ norm ε=0.1: for this classifier, we find an adversarial example for 4.38% of samples, and a certificate of robustness to norm-bounded perturbations for the remainder.

Across all robust training procedures and network architectures considered, and for both the MNIST and CIFAR-10 datasets, we are able to certify more samples than the state-of-the-art and find more adversarial examples than a strong first-order attack.

Neural networks trained only to optimize for training accuracy have been shown to be vulnerable to adversarial examples: perturbed inputs that are very similar to some regular input but for which the output is radically different BID14 .

There is now a large body of work proposing defense methods to produce classifiers that are more robust to adversarial examples.

However, as long as a defense is evaluated only via heuristic attacks (such as the Fast Gradient Sign Method (FGSM) (Goodfellow et al., 2015) or BID6 's attack (CW)), we have no guarantee that the defense actually increases the robustness of the classifier produced.

Defense methods thought to be successful when published have often later been found to be vulnerable to a new class of attacks.

For instance, multiple defense methods are defeated in BID5 by constructing defense-specific loss functions and in BID0 by overcoming obfuscated gradients.

Fortunately, we can evaluate robustness to adversarial examples in a principled fashion.

One option is to determine (for each test input) the minimum distance to the closest adversarial example, which we call the minimum adversarial distortion BID7 .

Alternatively, we can determine the adversarial test accuracy BID1 , which is the proportion of the test set for which no perturbation in some bounded class causes a misclassification.

An increase in the mean minimum adversarial distortion or in the adversarial test accuracy indicates an improvement in robustness.

1 We present an efficient implementation of a mixed-integer linear programming (MILP) verifier for properties of piecewise-linear feed-forward neural networks.

Our tight formulation for nonlinearities and our novel presolve algorithm combine to minimize the number of binary variables in the MILP problem and dramatically improve its numerical conditioning.

Optimizations in our MILP implementation improve performance by several orders of magnitude when compared to a naïve MILP implementation, and we are two to three orders of magnitude faster than the state-of-the-art Satisfiability Modulo Theories (SMT) based verifier, Reluplex BID7 We make the following key contributions:• We demonstrate that, despite considering the full combinatorial nature of the network, our verifier can succeed at evaluating the robustness of larger neural networks, including those with convolutional and residual layers.• We identify why we can succeed on larger neural networks with hundreds of thousands of units.

First, a large fraction of the ReLUs can be shown to be either always active or always inactive over the bounded input domain.

Second, since the predicted label is determined by the unit in the final layer with the maximum activation, proving that a unit never has the maximum activation over all bounded perturbations eliminates it from consideration.

We exploit both phenomena, reducing the overall number of non-linearities considered.• We determine for the first time the exact adversarial accuracy for MNIST classifiers to perturbations with bounded l ∞ norm .

We are also able to certify more samples than the state-of-the-art and find more adversarial examples across MNIST and CIFAR-10 classifiers with different architectures trained with a variety of robust training procedures.

Our code is available at https://github.com/vtjeng/MIPVerify.jl.

Our work relates most closely to other work on verification of piecewise-linear neural networks; BID4 provides a good overview of the field.

We categorize verification procedures as complete or incomplete.

To understand the difference between these two types of procedures, we consider the example of evaluating adversarial accuracy.

As in Kolter & Wong (2017) , we call the exact set of all final-layer activations that can be achieved by applying a bounded perturbation to the input the adversarial polytope.

Incomplete verifiers reason over an outer approximation of the adversarial polytope.

As a result, when using incomplete verifiers, the answer to some queries about the adversarial polytope may not be decidable.

In particular, incomplete verifiers can only certify robustness for a fraction of robust input; the status for the remaining input is undetermined.

In contrast, complete verifiers reason over the exact adversarial polytope.

Given sufficient time, a complete verifier can provide a definite answer to any query about the adversarial polytope.

In the context of adversarial accuracy, complete verifiers will obtain a valid adversarial example or a certificate of robustness for every input.

When a time limit is set, complete verifiers behave like incomplete verifiers, and resolve only a fraction of queries.

However, complete verifiers do allow users to answer a larger fraction of queries by extending the set time limit.

Incomplete verifiers for evaluating network robustness employ a range of techniques, including duality (Dvijotham et al., 2018; Kolter & Wong, 2017; BID11 , layer-by-layer approximations of the adversarial polytope BID18 , discretizing the search space (Huang et al., 2017) , abstract interpretation (Gehr et al., 2018) , bounding the local Lipschitz constant BID16 , or bounding the activation of the ReLU with linear functions BID16 .Complete verifiers typically employ either MILP solvers as we do (Cheng et al., 2017; Dutta et al., 2018; Fischetti & Jo, 2018; Lomuscio & Maganti, 2017) or SMT solvers BID7 Ehlers, 2017; BID7 BID12 .

Our approach improves upon existing MILP-based approaches with a tighter formulation for non-linearities and a novel presolve algorithm that makes full use of all information available, leading to solve times several orders of magnitude faster than a naïvely implemented MILP-based approach.

When comparing our approach to the state-of-the-art SMT-based approach (Reluplex) on the task of finding minimum adversarial distortions, we find that our verifier is two to three orders of magnitude faster.

Crucially, these improvements in performance allow our verifier to verify a network with over 100,000 units -several orders of magnitude larger than the largest MNIST classifier previously verified with a complete verifier.

A complementary line of research to verification is in robust training procedures that train networks designed to be robust to bounded perturbations.

Robust training aims to minimize the "worst-case loss" for each example -that is, the maximum loss over all bounded perturbations of that example (Kolter & Wong, 2017) .

Since calculating the exact worst-case loss can be computationally costly, robust training procedures typically minimize an estimate of the worst-case loss: either a lower bound as is the case for adversarial training (Goodfellow et al., 2015) , or an upper bound as is the case for certified training approaches (Hein & Andriushchenko, 2017; Kolter & Wong, 2017; BID11 .

Complete verifiers such as ours can augment robust training procedures by resolving the status of input for which heuristic attacks cannot find an adversarial example and incomplete verifiers cannot guarantee robustness, enabling more accurate comparisons between different training procedures.

We denote a neural network by a function f (·; θ) : R m → R n parameterized by a (fixed) vector of weights θ.

For a classifier, the output layer has a neuron for each target class.

Verification as solving an MILP.

The general problem of verification is to determine whether some property P on the output of a neural network holds for all input in a bounded input domain C ⊆ R m .

For the verification problem to be expressible as solving an MILP, P must be expressible as the conjunction or disjunction of linear properties P i,j over some set of polyhedra C i , where C = ∪C i .In addition, f (·) must be composed of piecewise-linear layers.

This is not a particularly restrictive requirement: piecewise-linear layers include linear transformations (such as fully-connected, convolution, and average-pooling layers) and layers that use piecewise-linear functions (such as ReLU or maximum-pooling layers).

We provide details on how to express these piecewise-linear functions in the MILP framework in Section 4.1.

The "shortcut connections" used in architectures such as ResNet (He et al., 2016) are also linear, and batch normalization (Ioffe & Szegedy, 2015) or dropout BID13 are linear transformations at evaluation time BID4 .

Evaluating Adversarial Accuracy.

Let G(x) denote the region in the input domain corresponding to all allowable perturbations of a particular input x. In general, perturbed inputs must also remain in the domain of valid inputs X valid .

For example, for normalized images with pixel values ranging from 0 to 1, X valid = [0, 1] m .

As in Madry et al. (2018) , we say that a neural network is robust to perturbations on x if the predicted probability of the true label λ(x) exceeds that of every other label for all perturbations: DISPLAYFORM0 Equivalently, the network is robust to perturbations on x if and only if Equation 2 is infeasible for x .

DISPLAYFORM1 where f i (·) is the i th output of the network.

For conciseness, we call x robust with respect to the network if f (·) is robust to perturbations on x. If x is not robust, we call any x satisfying the constraints a valid adversarial example to x. The adversarial accuracy of a network is the fraction of the test set that is robust; the adversarial error is the complement of the adversarial accuracy.

As long as G(x)

∩ X valid can be expressed as the union of a set of polyhedra, the feasibility problem can be expressed as an MILP.

The four robust training procedures we consider (Kolter & Wong, 2017; BID17 Madry et al., 2018; BID11 are designed to be robust to perturbations with bounded l ∞ norm, and the l ∞ -ball of radius around each input x can be succinctly represented by the set of linear constraints DISPLAYFORM2 Evaluating Mean Minimum Adversarial Distortion.

Let d(·, ·) denote a distance metric that measures the perceptual similarity between two input images.

The minimum adversarial distortion under d for input x with true label λ(x) corresponds to the solution to the optimization: DISPLAYFORM3 subject to DISPLAYFORM4 We can target the attack to generate an adversarial example that is classified in one of a set of target labels T by replacing Equation 4 with DISPLAYFORM5 The most prevalent distance metrics in the literature for generating adversarial examples are the l 1 BID6 , l 2 BID14 , and l ∞ (Goodfellow et al., 2015; BID10 norms.

All three can be expressed in the objective without adding any additional integer variables to the model BID3 ; details are in Appendix A.3.

Tight formulations of the ReLU and maximum functions are critical to good performance of the MILP solver; we thus present these formulations in detail with accompanying proofs.

Formulating ReLU Let y = max(x, 0), and l ≤ x ≤ u. There are three possibilities for the phase of the ReLU.

If u ≤ 0, we have y ≡ 0.

We say that such a unit is stably inactive.

Similarly, if l ≥ 0, we have y ≡ x. We say that such a unit is stably active.

Otherwise, the unit is unstable.

For unstable units, we introduce an indicator decision variable a = 1 x≥0 .

As we prove in Appendix A.1, y = max(x, 0) is equivalent to the set of linear and integer constraints in Equation FORMULA6 .

DISPLAYFORM0 Formulating the Maximum Function Let y = max(x 1 , x 2 , . . .

, x m ), and DISPLAYFORM1 . .

, l m ).

We can eliminate from consideration all x i where DISPLAYFORM2 We introduce an indicator decision variable a i for each of our input variables, where a i = 1 =⇒ y = x i .

Furthermore, we define u max,−i max j =i (u j ).

As we prove in Appendix A.2, the constraint y = max(x 1 , x 2 , . . . , x m ) is equivalent to the set of linear and integer constraints in Equation 7 .

DISPLAYFORM3

We previously assumed that we had some element-wise bounds on the inputs to non-linearities.

In practice, we have to carry out a presolve step to determine these bounds.

Determining tight bounds is critical for problem tractability: tight bounds strengthen the problem formulation and thus improve solve times BID15 .

For instance, if we can prove that the phase of a ReLU is stable, we can avoid introducing a binary variable.

More generally, loose bounds on input to some unit will propagate downstream, leading to units in later layers having looser bounds.

We used two procedures to determine bounds: INTERVAL ARITHMETIC (IA), also used in Cheng et al. FORMULA0 ; Dutta et al. (2018) , and the slower but tighter LINEAR PROGRAMMING (LP) approach.

Implementation details are in Appendix B.Since faster procedures achieve efficiency by compromising on tightness of bounds, we face a tradeoff between higher build times (to determine tighter bounds to inputs to non-linearities), and higher solve times (to solve the main MILP problem in Equation 2 or Equation 3-5).

While a degree of compromise is inevitable, our knowledge of the non-linearities used in our network allows us to reduce average build times without affecting the strength of the problem formulation.

The key observation is that, for piecewise-linear non-linearities, there are thresholds beyond which further refining a bound will not improve the problem formulation.

With this in mind, we adopt a progressive bounds tightening approach: we begin by determining coarse bounds using fast procedures and only spend time refining bounds using procedures with higher computational complexity if doing so could provide additional information to improve the problem formulation.

4 Pseudocode demonstrating how to efficiently determine bounds for the tightest possible formulations for the ReLU and maximum function is provided below and in Appendix C respectively.

GETBOUNDSFORRELU(x, f s) 1 £ f s are the procedures to determine bounds, sorted in increasing computational complexity.

2 l best = −∞; u best = ∞ £ initialize best known upper and lower bounds on x 3 for f in f s: £ carrying out progressive bounds tightening 4 Using one of these procedures in addition to IA and LP has the potential to further reduce build times.

DISPLAYFORM0

Dataset.

All experiments are carried out on classifiers for the MNIST dataset of handwritten digits or the CIFAR-10 dataset of color images.

Architectures.

We conduct experiments on a range of feed-forward networks.

In all cases, ReLUs follow each layer except the output layer.

MLP-m×[n] refers to a multilayer perceptron with m hidden layers and n units per hidden layer.

We further abbreviate and as MLP A and MLP B respectively.

CNN A and CNN B refer to the small and large ConvNet architectures in BID17 .

CNN A has two convolutional layers (stride length 2) with 16 and 32 filters (size 4 × 4) respectively, followed by a fully-connected layer with 100 units.

CNN B has four convolutional layers with 32, 32, 64, and 64 filters, followed by two fully-connected layers with 512 units.

RES refers to the ResNet architecture used in BID17 , with 9 convolutional layers in four blocks, followed by two fully-connected layers with 4096 and 1000 units respectively.

Training Methods.

We conduct experiments on networks trained with a regular loss function and networks trained to be robust.

Networks trained to be robust are identified by a prefix corresponding to the method used to approximate the worst-case loss: LP d 5 when the dual of a linear program is used, as in Kolter & Wong (2017) ; SDP d when the dual of a semidefinite relaxation is used, as in BID11 ; and Adv when adversarial examples generated via Projected Gradient Descent (PGD) are used, as in Madry et al. (2018) .

Full details on each network are in Appendix D.1.Experimental Setup.

We run experiments on a modest 8 CPUs@2.20 GHz with 8GB of RAM.

Appendix D.2 provides additional details about the computational environment.

Maximum build effort is LP.

Unless otherwise noted, we report a timeout if solve time for some input exceeds 1200s.

Our MILP approach implements three key optimizations: we use progressive tightening, make use of the information provided by the restricted input domain G(x), and use asymmetric bounds in the ReLU formulation in Equation 6.

None of the four other MILP-based complete verifiers implement progressive tightening or use the restricted input domain, and only Fischetti & Jo (2018) uses asymmetric bounds.

Since none of the four verifiers have publicly available code, we use ablation tests to provide an idea of the difference in performance between our verifier and these existing ones.

When removing progressive tightening, we directly use LP rather than doing IA first.

When removing using restricted input domain, we determine bounds under the assumption that our perturbed input could be anywhere in the full input domain X valid , imposing the constraint x ∈ G(x) only after all bounds are determined.

Finally, when removing using asymmetric bounds, we replace l and u in Equation 6 with −M and M respectively, where M max(−l, u), as is done in Cheng et al. FORMULA0 ; Dutta et al. (2018); Lomuscio & Maganti (2017) .

We carry out experiments on an MNIST classifier; results are reported in TAB0 .

The ablation tests demonstrate that each optimization is critical to the performance of our verifier.

In terms of performance comparisons, we expect our verifier to have a runtime several orders of magnitude faster than any of the three verifiers not using asymmetric bounds.

While Fischetti & Jo (2018) do use asymmetric bounds, they do not use information from the restricted input domain; we thus expect our verifier to have a runtime at least an order of magnitude faster than theirs.

We also compared our verifier to other verifiers on the task of finding minimum targeted adversarial distortions for MNIST test samples.

Verifiers included for comparison are 1) Reluplex (Katz et al., 2017), a complete verifier also able to find the true minimum distortion; and 2) LP 6 , Fast-Lip, Fast-Lin (Weng et al., 2018), and LP-full (Kolter & Wong, 2017) , incomplete verifiers that provide a certified lower bound on the minimum distortion.

Verification Times, vis-à-vis the state-of-the-art SMT-based complete verifier Reluplex.

Figure 1 presents average verification times per sample.

All solves for our method were run to completion.

On the l ∞ norm, we improve on the speed of Reluplex by two to three orders of magnitude.

Minimum Targeted Adversarial Distortions, vis-à-vis incomplete verifiers.

FIG2 compares lower bounds from the incomplete verifiers to the exact value we obtain.

The gap between the best lower bound and the true minimum adversarial distortion is significant even on these small networks.

This corroborates the observation in BID11 that incomplete verifiers provide weak bounds if the network they are applied to is not optimized for that verifier.

For example, under the l ∞ norm, the best certified lower bound is less than half of the true minimum distortion.

In context: a network robust to perturbations with l ∞ norm-bound = 0.1 would only be verifiable to = 0.05.

The gap between the true minimum adversarial distortion and the best lower bound is significant in all cases, increasing for deeper networks.

We report mean values over 100 samples.

We use our verifier to determine the adversarial accuracy of classifiers trained by a range of robust training procedures on the MNIST and CIFAR-10 datasets.

Table 2 presents the test error and estimates of the adversarial error for these classifiers.7 For MNIST, we verified a range of networks trained to be robust to attacks with bounded l ∞ norm = 0.1, as well as networks trained to be robust to larger attacks of = 0.2, 0.3 and 0.4.

Lower bounds on the adversarial error are proven by providing adversarial examples for input that is not robust.

We compare the number of samples for which we successfully find adversarial examples to the number for PGD, a strong first-order attack.

Upper bounds on the adversarial error are proven by providing certificates of robustness for input that is robust.

We compare our upper bounds to the previous state-of-the-art for each network.

While performance depends on the training method and architecture, we improve on both the lower and upper bounds for every network tested.8 For lower bounds, we successfully find an adversarial example for every test sample that PGD finds an adversarial example for.

In addition, we observe that PGD 'misses' some valid adversarial examples: it fails to find these adversarial examples even though they are within the norm bounds.

As the last three rows of Table 2 show, PGD misses for a larger fraction of test samples when is larger.

We also found that PGD is far more likely to miss for some test sample if the minimum adversarial distortion for that sample is close to ; this observation is discussed in more depth in Appendix G. For upper bounds, we improve on the bound on adversarial error even when the upper bound on the worst-case loss -which is used to generate the certificate of robustness -is explicitly optimized for during training (as is the case for LP d and SDP d training).

Our method also scales well to the more complex CIFAR-10 dataset and the larger LP d -RES network (which has 107,496 units), with the solver reaching the time limit for only 0.31% of samples.

Most importantly, we are able to determine the exact adversarial accuracy for Adv-MLP B and LP d -CNN A for all tested, finding either a certificate of robustness or an adversarial example for every test sample.

For Adv-MLP B and LP d -CNN A , running our verifier over the full test set takes approximately 10 hours on 8 CPUs -the same order of magnitude as the time to train each network on a single GPU.

Better still, verification of individual samples is fully parallelizable -so verification time can be reduced with more computational power.

Table 2 : Adversarial accuracy of MNIST and CIFAR-10 classifiers to perturbations with l ∞ normbound .

In every case, we improve on both 1) the lower bound on the adversarial error, found by PGD, and 2) the previous state-of-the-art (SOA) for the upper bound, generated by the following methods:[ The key lies in the restricted input domain G(x) for each test sample x. When input is restricted to G(x), we can prove that many ReLUs are stable (with respect to G).

Furthermore, we can eliminate some labels from consideration by proving that the upper bound on the output neuron corresponding to that label is lower than the lower bound for some other output neuron.

As the results in TAB2 show, a significant number of ReLUs can be proven to be stable, and a significant number of labels can be eliminated from consideration.

Rather than being correlated to the total number of ReLUs, solve times are instead more strongly correlated to the number of ReLUs that are not provably stable, as well as the number of labels that cannot be eliminated from consideration.

This paper presents an efficient complete verifier for piecewise-linear neural networks.

While we have focused on evaluating networks on the class of perturbations they are designed to be robust to, defining a class of perturbations that generates images perceptually similar to the original remains an important direction of research.

Our verifier is able to handle new classes of perturbations (such as convolutions applied to the original image) as long as the set of perturbed images is a union of polytopes in the input space.

We close with ideas on improving verification of neural networks.

First, our improvements can be combined with other optimizations in solving MILPs.

For example, BID4 DISPLAYFORM0 We consider two cases.

Recall that a is the indicator variable a = 1 x≥0 .When a = 0, the constraints in Equation FORMULA0 This formulation for rectified linearities is sharp BID15 if we have no further information about x. This is the case since relaxing the integrality constraint on a leads to (x, y) being restricted to an area that is the convex hull of y = max(x, 0).

However, if x is an affine expression x = w T z + b, the formulation is no longer sharp, and we can add more constraints using bounds we have on z to improve the problem formulation.

We reproduce our formulation for the maximum function below.

DISPLAYFORM0 Equation FORMULA0 ensures that exactly one of the a i is 1.

It thus suffices to consider the value of a i for a single variable.

When a i = 1, Equations 13 and 14 are binding, and together imply that y = x i .

We thus have DISPLAYFORM1 When a i = 0, we simply need to show that the constraints involving When d(x , x) = x − x 1 , we introduce the auxiliary variable δ, which bounds the elementwise absolute value from above: DISPLAYFORM2 we introduce the auxiliary variable , which bounds the l ∞ norm from above: DISPLAYFORM3 DISPLAYFORM4 B DETERMINING TIGHT BOUNDS ON DECISION VARIABLES Our framework for determining bounds on decision variables is to view the neural network as a computation graph G. Directed edges point from function input to output, and vertices represent variables.

Source vertices in G correspond to the input of the network, and sink vertices in G correspond to the output of the network.

The computation graph begins with defined bounds on the input variables (based on the input domain (G(x) ∩ X valid )), and is augmented with bounds on intermediate variables as we determine them.

The computation graph is acyclic for the feed-forward networks we consider.

Since the networks we consider are piecewise-linear, any subgraph of G can be expressed as an MILP, with constraints derived from 1) input-output relationships along edges and 2) bounds on the values of the source nodes in the subgraph.

Integer constraints are added whenever edges describe a non-linear relationship.

We focus on computing an upper bound on some variable v; computing lower bounds follows a similar process.

All the information required to determine the best possible bounds on v is contained in the subtree of G rooted at v, G v .

(Other variables that are not ancestors of v in the computation graph cannot affect its value.)

Maximizing the value of v in the MILP M v corresponding to G v gives the optimal upper bound on v.

We can reduce computation time in two ways.

Firstly, we can prune some edges and vertices of G v .

Specifically, we select a set of variables with existing bounds V I that we assume to be independent (that is, we assume that they each can take on any value independent of the value of the other variables in V I ).

We remove all in-edges to vertices in V I , and eliminate variables without children, resulting in the smaller computation graph G v,V I .

Maximizing the value of v in the MILP M v,V I corresponding to G v,V I gives a valid upper bound on v that is optimal if the independence assumption holds.

We can also reduce computation time by relaxing some of the integer constraints in M v to obtain a MILP with fewer integer variables M v .

Relaxing an integer constraint corresponds to replacing the relevant non-linear relationship with its convex relaxation.

Again, the objective value returned by maximizing the value of v over M v may not be the optimal upper bound, but is guaranteed to be a valid bound.

FULL considers the full subtree G v and does not relax any integer constraints.

The upper and lower bound on v is determined by maximizing and minimizing the value of v in M v respectively.

FULL is also used in Cheng et al. FORMULA0 and Fischetti & Jo (2018) .If solves proceed to optimality, FULL is guaranteed to find the optimal bounds on the value of a single variable v. The trade-off is that, for deeper layers, using FULL can be relatively inefficient, since solve times in the worst case are exponential in the number of binary variables in M v .Nevertheless, contrary to what is asserted in Cheng et al. FORMULA0 , we can terminate solves early and still obtain useful bounds.

For example, to determine an upper bound on v, we set the objective of M v to be to maximize the value of v. As the solve process proceeds, we obtain progressively better certified upper bounds on the maximum value of v. We can thus terminate the solve process and extract the best upper bound found at any time, using this upper bound as a valid (but possibly loose) bound on the value of v.

LP considers the full subtree G v but relaxes all integer constraints.

This results in the optimization problem becoming a linear program that can be solved more efficiently.

LP represents a good middle ground between the optimality of FULL and the performance of IA.

IA selects V I to be the parents of v. In other words, bounds on v are determined solely by considering the bounds on the variables in the previous layer.

We note that this is simply interval arithmetic BID9 .

IA is efficient (since it only involves matrix operations for our applications).

However, for deeper layers, using interval arithmetic can lead to overly conservative bounds.

GETBOUNDSFORMAX finds the tightest bounds required for specifying the constraint y = max(xs).Using the observation in Proposition 1, we stop tightening the bounds on a variable if its maximum possible value is lower than the minimum value of some other variable.

GETBOUNDSFORMAX returns a tuple containing the set of elements in xs that can still take on the maximum value, as well as a dictionary of upper and lower bounds.

GETBOUNDSFORMAX(xs, fs) 1 £ fs are the procedures to determine bounds, sorted in increasing computational complexity.

2 d l = {x : −∞ for x in xs} 3 d u = {x : ∞ for x in xs} 4 £ initialize dictionaries containing best known upper and lower bounds on xs 5 l max = −∞ £ l max is the maximum known lower bound on any of the xs 6 a = {xs} 7 £ a is a set of active elements in xs that can still potentially take on the maximum value.

8 for f in fs: £ carrying out progressive bounds tightening 9 do for x in xs: DISPLAYFORM0 then a.remove(x) £ x cannot take on the maximum value 12 DISPLAYFORM1 The source of the weights for each of the networks we present results for in the paper are provided below.• MNIST classifiers not designed to be robust:-MLP-2× [20] and are the MNIST classifiers in Weng et al.(2018), and can be found at https://github.com/huanzhang12/ CertifiedReLURobustness.• MNIST classifiers designed for robustness to perturbations with l ∞ norm-bound = 0.1:-LP d -CNN B is the large MNIST classifier for = 0.1 in BID17 , and can be found at https://github.com/locuslab/convex_adversarial/ blob/master/models_scaled/mnist_large_0_1.pth.

-LP d -CNN A is the MNIST classifier in Kolter & Wong (2017) , and can be found at https://github.com/locuslab/convex_adversarial/ blob/master/models/mnist.pth.

-Adv-CNN A was trained with adversarial examples generated by PGD.

PGD attacks were carried out with l ∞ norm-bound = 0.1, 8 steps per sample, and a step size of 0.334.

An l 1 regularization term was added to the objective with a weight of 0.0015625 on the first convolution layer and 0.003125 for the remaining layers.

• MNIST classifiers designed for robustness to perturbations with l ∞ norm-bound = 0.2, 0.3, 0.4:-LP d -CNN A was trained with the code available at https://github.com/ locuslab/convex_adversarial at commit 4e9377f.

Parameters selected were batch size=20, starting epsilon=0.01, epochs=200, seed=0.-LP d -CNN B is the large MNIST classifier for = 0.3 in BID17 , and can be found at https://github.com/locuslab/convex_adversarial/ blob/master/models_scaled/mnist_large_0_3.pth.• CIFAR-10 classifiers designed for robustness to perturbations with l ∞ norm-bound = 2 255-LP d -CNN A is the small CIFAR classifier in BID17 , courtesy of the authors.• CIFAR-10 classifiers designed for robustness to perturbations with l ∞ norm-bound = 8 255-LP d -RES is the resnet CIFAR classifier in BID17 , and can be found at https://github.com/locuslab/convex_adversarial/ blob/master/models_scaled/cifar_resnet_8px.pth.

We construct the MILP models in Julia BID2 ) using JuMP (Dunning et al., 2017 , with the model solved by the commercial solver Gurobi 7.5.2 (Gurobi Optimization, 2017) .

All experiments were run on a KVM virtual machine with 8 virtual CPUs running on shared hardware, with Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz processors, and 8GB of RAM.

To give a sense for how our verifier performs with other solvers, we ran a comparison with the Cbc (Forrest et al., 2018) and GLPK (Makhorin, 2012) solvers, two open-source MILP solvers.

When we use GLPK as the solver, our performance is significantly worse than when using Gurobi, with the solver timing out on almost 4% of samples.

While we time out on some samples with Cbc, our verifier still provides a lower bound better than PGD and an upper bound significantly better than the state-of-the-art for this network.

Overall, verifier performance is affected by the underlying MILP solver used, but we are still able to improve on existing bounds using an open-source solver.

F ADDITIONAL SOLVE STATISTICS F.1 NODES EXPLORED TAB7 presents solve statistics on nodes explored to supplement the results reported in Table 2 .

If the solver explores zero nodes for a particular sample, it proved that the sample was robust (or found an adversarial example) without branching on any binary variables.

This occurs when the bounds we find during the presolve step are sufficiently tight.

We note that this occurs for over 95% of samples for LP d -CNN A for = 0.1.

TAB8 presents additional information on the determinants of verification time for networks we omit in TAB2 .

Figure 3: Fraction of samples in the MNIST test set vulnerable to attack for which PGD succeeds at finding an adversarial example.

Samples are binned by their minimum adversarial distortion (as measured under the l ∞ norm), with bins of size 0.01.

Each of these are LP d -CNN A networks, and were trained to optimize for robustness to attacks with l ∞ norm-bound .

For any given network, the success rate of PGD declines as the minimum adversarial distortion increases.

Comparing networks, success rates decline for networks with larger even at the same minimum adversarial distortion.

PGD succeeds in finding an adversarial example if and only if the starting point for the gradient descent is in the basin of attraction of some adversarial example.

Since PGD initializes the gradient descent with a randomly chosen starting point within G(x)

∩ X valid , the success rate (with a single random start) corresponds to the fraction of G(x)

∩ X valid that is in the basin of attraction of some adversarial example.

Intuitively, the success rate of PGD should be inversely related to the magnitude of the minimum adversarial distortionδ: ifδ is small, we expect more of G(x)

∩ X valid to correspond to adversarial examples, and thus the union of the basins of attraction of the adversarial examples is likely to be larger.

We investigate here whether our intuition is substantiated.

To obtain the best possible empirical estimate of the success rate of PGD for each sample, we would need to re-run PGD initialized with multiple different randomly chosen starting points within G(x)

∩ X valid .However, since we are simply interested in the relationship between success rate and minimum adversarial distortion, we obtained a coarser estimate by binning the samples based on their minimum adversarial distortion, and then calculating the fraction of samples in each bin for which PGD with a single randomly chosen starting point succeeds at finding an adversarial example.

• PGD is very successful at finding adversarial examples when the magnitude of the minimum adversarial distortion,δ, is small.• The success rate of PGD declines significantly for all networks asδ approaches .•

For a given value ofδ, and two networks a and b trained to be robust to attacks with l ∞ norm-bound a and b respectively (where a < b ), PGD is consistently more successful at attacking the network trained to be robust to smaller attacks, a, as long asδ a .The sharp decline in the success rate of PGD asδ approaches is particularly interesting, especially since it is suggests a pathway to generating networks that appear robust when subject to PGD attacks of bounded l ∞ norm but are in fact vulnerable to such bounded attacks: we simply train the network to maximize the total number of adversarial examples with minimum adversarial distortion close to .

When verifying the robustness of SDP d -MLP A , we observed that a significant proportion of kernel weights were close to zero.

Many of these tiny weights are unlikely to be contributing significantly to the final classification of any input image.

Having said that, setting these tiny weights to zero could potentially reduce verification time, by 1) reducing the size of the MILP formulation, and by 2) ameliorating numerical issues caused by the large range of numerical coefficients in the network (Gurobi, 2017).We generated sparse versions of the original network to study the impact of sparseness on solve times.

Our heuristic sparsification algorithm is as follows: for each fully-connected layer i, we set a fraction f i of the weights with smallest absolute value in the kernel to 0, and rescale the rest of the weights such that the l 1 norm of the kernel remains the same.

9 Note that MLP A consists of only two layers: one hidden layer (layer 1) and one output layer (layer 2).

TAB9 summarizes the results of verifying sparse versions of SDP d -MLP A ; the first row presents results for the original network, and the subsequent rows present results when more and more of the kernel weights are set to zero.

When comparing the first and last rows, we observe an improvement in both mean time and fraction timed out by an order of magnitude.

As expected, sparsifying weights increases the test error, but the impact is not significant until f 1 exceeds 0.8.

We also find that sparsification significantly improves our upper bound on adversarial error -to a point: the upper bound on adversarial error for f 1 = 0.9 is higher than that for f 1 = 0.8, likely because the true adversarial error has increased significantly.

Starting with a network that is robust, we have demonstrated that a simple sparsification approach can already generate a sparsified network with an upper bound on adversarial error significantly lower than the best upper bound that can be determined for the original network.

Adopting a more principled sparsification approach could achieve the same improvement in verifiability but without compromising on the true adversarial error as much.

Networks that are designed to be robust need to balance two competing objectives.

Locally, they need to be robust to small perturbations to the input.

However, they also need to retain sufficient global expressiveness to maintain a low test error.

For the networks in TAB2 , even though each robust training approach estimates the worst-case error very differently, all approaches lead to a significant fraction of the ReLUs in the network being provably stable with respect to perturbations with bounded l ∞ norm.

In other words, for the input domain G(x) consisting of all bounded perturbations of the sample x, we can show that, for many ReLUs, the input to the unit is always positive (and thus the output is linear in the input) or always negative (and thus the output is always zero).

As discussed in the main text, we believe that the need for the network to be robust to perturbations in G drives more ReLUs to be provably stable with respect to G.To better understand how networks can retain global expressiveness even as many ReLUs are provably stable with respect to perturbations with bounded l ∞ norm , we study how the number of ReLUs that are provably stable changes as we vary the size of G(x) by changing the maximum allowable l ∞ norm of perturbations.

The results are presented in FIG7 .As expected, the number of ReLUs that cannot be proven to be stable increases as the maximum allowable l ∞ norm of perturbations increases.

More interestingly, LP d -CNN A is very sensitive to the = 0.1 threshold, with a sharp increase in the number of ReLUs that cannot be proven to be stable when the maximum allowable l ∞ norm of perturbations increases beyond 0.102.

An increase of the same abruptness is not seen for the other two networks.(a) LPd-CNNA.

Note the sharp increase in the number of ReLUs that cannot be proven to be stable when the maximum l∞ norm increases beyond 0.102.

TAB2 are marked by a dotted line.

As we increase the maximum allowable l ∞ norm of perturbations, the number of ReLUs that cannot be proven to be stable increases across all networks (as expected), but LP d -CNN A is far more sensitive to the = 0.1 threshold.

<|TLDR|>

@highlight

We efficiently verify the robustness of deep neural models with over 100,000 ReLUs, certifying more samples than the state-of-the-art and finding more adversarial examples than a strong first-order attack.

@highlight

Performs a careful study of mixed integer linear programming approaches for verifying robustness of neural networks to adversarial perturbations and proposes three enhancements to MILP formulations of neural network verification.