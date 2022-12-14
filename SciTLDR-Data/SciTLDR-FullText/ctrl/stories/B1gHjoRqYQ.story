There are two major paradigms of white-box adversarial attacks that attempt to impose input perturbations.

The first paradigm, called the fix-perturbation attack, crafts adversarial samples within a given perturbation level.

The second paradigm, called the zero-confidence attack, finds the smallest perturbation needed to cause misclassification, also known as the margin of an input feature.

While the former paradigm is well-resolved, the latter is not.

Existing zero-confidence attacks either introduce significant approximation errors, or are too time-consuming.

We therefore propose MarginAttack, a zero-confidence attack framework that is able to compute the margin with improved accuracy and efficiency.

Our experiments show that MarginAttack is able to compute a smaller margin than the state-of-the-art zero-confidence attacks, and matches the state-of-the-art fix-perturbation attacks.

In addition, it runs significantly faster than the Carlini-Wagner attack, currently the most accurate zero-confidence attack algorithm.

Adversarial attack refers to the task of finding small and imperceptible input transformations that cause a neural network classifier to misclassify.

White-box attacks are a subset of attacks that have access to gradient information of the target network.

In this paper, we will focus on the white-box attacks.

An important class of input transformations is adding small perturbations to the input.

There are two major paradigms of adversarial attacks that attempt to impose input perturbations.

The first paradigm, called the fix-perturbation attack, tries to find perturbations that are most likely to cause misclassification, with the constraint that the norm of the perturbations cannot exceed a given level.

Since the perturbation level is fixed, fix-perturbation attacks may fail to find any adversarial samples for inputs that are far away from the decision boundary.

The second paradigm, called the zero-confidence attack, tries to find the smallest perturbations that are guaranteed to cause misclassification, regardless of how large the perturbations are.

Since they aim to minimize the perturbation norm, zero-confidence attacks usually find adversarial samples that ride right on the decision boundaries, and hence the name "zero-confidence".

The resulting perturbation norm is also known as the margin of an input feature to the decision boundary.

Both of these paradigms are essentially constrained optimization problems.

The former has a simple convex constraint (perturbation norm), but a non-convex target (classification loss or logit differences).

In contrast, the latter has a non-convex constraint (classification loss or logit differences), but a simple convex target (perturbation norm).Despite their similarity as optimization problems, the two paradigms differ significantly in terms of difficulty.

The fix-perturbation attack problem is easier.

The state-of-the-art algorithms, including projected gradient descent (PGD) BID10 and distributional adversarial attack (Zheng et al., 2018) , can achieve both high efficiency and high success rate, and often come with theoretical convergence guarantee.

On the other hand, the zero-confidence attack problem is much more challenging.

Existing methods are either not strong enough or too slow.

For example, DeepFool BID11 and fast gradient sign method (FGSM) BID3 BID7 b) linearizes the constraint, and solves the simplified optimization problem with a simple convex target and a linear constraint.

However, due to the linearization approximation errors, the solution can be far from optimal.

As another extreme, L-BFGS BID18 and Carlini-Wagner (CW) BID1 convert the optimization problem into a Lagrangian, and the Lagrangian multiplier is determined through grid search or binary search.

These attacks are generally much stronger and theoretically grounded, but can be very slow.

The necessity of developing a better zero-confidence attack is evident.

The zero-confidence attack paradigm is a more realistic attack setting.

More importantly, it aims to measure the margin of each individual token, which lends more insight into the data distribution and adversarial robustness.

Motivated by this, we propose MARGINATTACK, a zero-confidence attack framework that is able to compute the margin with improved accuracy and efficiency.

Specifically, MARGINATTACK iterates between two moves.

The first move, called restoration move, linearizes the constraint and solves the simplified optimization problem, just like DeepFool and FGSM; the second move, called projection move, explores even smaller perturbations without changing the constraint values significantly.

By construction, MARGINATTACK inherits the efficiency in DeepFool and FGSM, and improves over them in terms of accuracy with a convergence guarantee.

Our experiments show that MARGINAT-TACK attack is able to compute a smaller margin than the state-of-the-art zero-confidence attacks, and matches the state-of-the-art fix-perturbation attacks.

In addition, it runs significantly faster than CW, and in some cases comparable to DeepFool and FGSM.

In addition to the aforementioned state-of-the-art attacks, there are a couple of other works that attempt to explore the margin.

Jacobian-based saliency map attack BID13 is among the earliest works that apply gradient information to guide the crafting of adversarial examples.

It chooses to perturb the input features whose gradient is consistent with the adversarial goal.

Onepixel attack BID17 finds adversarial examples by perturbing only one pixel, which can be regarded as finding the 0 margin of the inputs.

BID5 converts PGD into a zeroconfidence attack by searching different perturbation levels, but this again can be time-consuming because it needs to solve multiple optimization subproblems.

Weng et al. proposed a metric called CLEVER (Weng et al., 2018) , which estimates an upper-bound of the margins.

Unfortunately, recent work BID2 has shown that CLEVER can overestimate the margins due to gradient masking BID14 .

The above are a just a small subset of white-box attack algorithms that are relevant to our work.

For an overview of the field, we refer readers to BID0 .The MARGINATTACK framework is inspired by the Rosen's algorithm BID15 for constraint optimization problems.

However, there are several important distinctions.

First, the Rosen's algorithm rests on some unrealistic assumptions for neural networks, e.g. continuously differentiable constraints, while MARGINATTACK has a convergence guarantee with a more realistic set of assumptions.

Second, the Rosen's algorithm requires a step size search for each iteration, which can be time-consuming, whereas MARGINATTACK will work with a simple diminishing step size scheme.

Most importantly, as will be shown later, MARGINATTACK refers to a large class of attack algorithms depending on how the two parameters, a (k) and b (k) , are set, and the Rosen's algorithm only fits into one of the settings, which only works well under the 2 norm.

For other norms, there exist other parameter settings that are much more effective.

As another highlight, the convergence guarantee of MARGINATTACK holds for all the settings that satisfy some moderate assumptions.

In this section, we will formally introduce the algorithm and discuss its convergence properties.

In the paper, we will denote scalars with non-bolded letters, e.g. a or A; column vectors with lowercased, bolded letters, e.g. a; matrix with upper-cased, bolded letters, e.g. A; sets with upper-cased double-stoke letters, e.g. A; gradient of a function f (x) evaluated at x = x 0 as ???f (x 0 ).

Given a classifier whose output logits are denoted as l 0 (x), l 1 (x), ?? ?? ?? , l C???1 (x), where C is the total number of classes, for any data token (x 0 , t), where x 0 is an n-dimensional input feature vector, and t ??? {0, ?? ?? ?? , C ??? 1} is its label, MARGINATTACK computes DISPLAYFORM0 where d(??) is a norm.

In this paper we only consider 2 and ??? norms, but the proposed method is generalizable to other norms.

For non-targeted adversarial attacks, the constraint is defined as DISPLAYFORM1 where ?? is the offset parameter.

As a common practice, ?? is often set to a small negative number to ensure that the adversarial sample lies on the incorrect side of the decision boundary.

In this paper, we will only consider non-targeted attack, but all the discussions are applicable to targeted attacks (i.e. c(x) = max i =a l i (x) ??? l a (x) ??? ?? for a target class a).

MARGINATTACK alternately performs the restoration move and the projection move.

Specifically, denote the solution after the k-th iteration as x (k) .

Then the two steps are:Restoration Move: The restoration move tries to hop to the constraint boundary, i.e. c(x) = 0 with the shortest hop.

Formally, it solves: DISPLAYFORM0 where ?? (k) is the step size within [0, 1] .

Notice that the left hand side of the constraint in Eq. (3) is the first-order Taylor approximation of c(z DISPLAYFORM1 , so this constraint tries to move point closer to c(x) = 0 by ?? (k) .

It can be shown, from the dual-norm theory, 1 that the solution to (3) is DISPLAYFORM2 .

Specifically, noticing that the dual norm of the p norm is the (1???p ???1 ) ???1 norm, we have DISPLAYFORM3 As mentioned, Eq. (4) is similar to DeepFool under 2 norm, and to FGSM under ??? norm.

Therefore, we can expect that the restoration move should effectively hop towards the decision boundary, but the hop direction may not be optimal.

That is why we need the next move.

Projection Move: The projection move tries to move closer to x 0 while ensuring that c(x) will not change drastically.

Formally, DISPLAYFORM4 where ?? (k) is the step size within [0, 1]; a (k) and b (k) are two scalars, which will be specified later.

As an intuitive explanation on Eq. (3), notice that the second term, which we will call the distance reduction term, reduces the distance to x 0 , whereas the third term, which we will call the constraint reduction term, reduces the the constraint (because s(z (k) ) and ???c(z (k) ) has a positive inner product).

Therefore, the projection move essentially strikes a balance between reduction in distance and reduction in constraint.

a (k) and b (k) can have two designs.

The first design is to ensure the constraint values are roughly the same after the move, i.e. c(z DISPLAYFORM5 whose solution is DISPLAYFORM6 Another design is to ensure the perturbation norm reduces roughly by DISPLAYFORM7 .

By Taylor approximation, we have DISPLAYFORM8 whose solution is DISPLAYFORM9 It should be noted that Eqs. FORMULA8 and FORMULA0 are just two specific choices for a (k) and b (k) .

It turns out that MARGINATTACK will work with a convergence guarantee for a wide range of bounded a (k) s and b (k) s that satisfy some conditions, as will be shown in section 3.4.

Therefore, MARGINAT-TACK provides a general and flexible framework for zero-confidence adversarial attack designs.

In practice, we find that Eq. (8) works better for 2 norm, and Eq. (8) works better for ??? norm.

FIG1 illustrates a typical convergence path of MARGINATTACK using 2 norm and Eq. FORMULA8 as an example.

The red dots on the right denote the original inputs x 0 and its closest point on the decision boundary, x * .

Suppose after iteration k, MARGINATTACK reaches x (k) , denoted by the green dot on the left.

The restoration move travels directly towards the decision boundary by finding the normal direction to the current constraint contour.

Then, the projection move travels along the tangent plane of the current constraint contour to reduce the distance to x 0 while preventing the constraint value from deviating much.

As intuitively expected, the iteration should eventually approach x * .

FIG2 plots an empirical convergence curve of the perturbation norm and constraint value of MARGINATTACK-2 on a randomly chosen CIFAR image.

Each move from a triangle to a circle dot is a restoration move, and from circle to triangle a projection move.

The red line is the smoothed version.

As can be seen, a restoration move reduces the constraint value while slightly increasing the constraint norm, and a projection move reduces the perturbation norm while slightly affecting the constraint value.

Both curves can eventually converge.

The constraint function c(x) in Eq. FORMULA1 is nonconvex, thus the convergence analysis for MARGINAT-TACK is limited to the vicinity of a unique local optimum, as stated in the following theorem.

Theorem 1.

Denote x * as one local optimum for Eq.(1).

Assume ???c(x * ) exists.

Define projection matrices DISPLAYFORM0 Consider the neighborhood B = {x : DISPLAYFORM1 2 ??? X, |c(x)| ??? C} that satisfies the following assumptions:1. (Differentiability) ???x ??? B, ???c(x) exists, but can be discontinuous, i.e. all the discontinuity points of the gradient in B are jump discontinuities; DISPLAYFORM2 6. (Unique Optimality) x * is the only global optimum within B; DISPLAYFORM3 Then we have the convergence guarantee lim k?????? DISPLAYFORM4 The proof will be presented in the appendix.

Here are a few remarks.

First, assumption 1 allows jump discontinuities in ???c(x) almost everywhere, which is a very practical assumption for deep neural networks.

Most neural network operations, such as ReLU and max-pooling, as well as the max operation in Eq. (2), introduce nothing beyond jump discontinuities in gradient.

Second, assumption 3 does require the constraint gradient to be lower bounded, which may lead to concerns that MARGINATTACK may fail in the presence of gradient masking BID14 .

However, notice that the gradient boundedness assumption is only imposed in B, which is in the vicinity of the decision boundary, whereas gradient masking is most likely to appear away from the decision boundary and where the input features are populated.

Besides, as will be discussed later, a random initialization as in PGD will be adopted to bypass regions with gradient masking.

Experiments on adversarially trained models also verify the robustness of MARGINATTACK.Finally, assumption 5 essentially stipulates that c(x) is convex or "not too concave" in B (and thus so is the constraint set c(x) ??? 0), so that the first order optimality condition can readily imply local minimum instead of a local maximum.

In fact, it can be shown that assumption 5 can be implied if c(x) is convex in B. There are a few additional implementation details as outlined below.

Box Constraint:

In many applications, each dimension of the input features should be bounded, i.e. x ??? [x min , x max ] n .

To impose the box constraint, the restoration move problem as in Eq. FORMULA2 is modified as DISPLAYFORM5 whose solution is DISPLAYFORM6 Proj(??) is an operator that projects the vector in its argument onto the subset in its subscript.

I is a set of indices with which the elements inz (k) satisfy the box constraint, and I C is its complement.

I is determined by running Eq. (13) iteratively and updating I after each iterations.

Unlike other attack algorithms that simply project the solution onto the constraint box, MARGINAT-TACK incorporates the box constraint in a principled way, such that any local optimal solution x * will be an invariant point of the restoration move.

Thus the convergence is faster.

Target Scan: According to Eq. (2), each restoration move essentially approaches the adversarial class with the highest logit, but the class with the highest logit may not be the closest.

To mitigate the problem, we follow a similar approach adopted in DeepFool, which we call target scan.

Target scan performs a target-specific restoration move towards each class, and chooses the move with the shortest distance.

Formally, target scan introduces a set of target-specific constraints {c i (x) = l t (x) ??? l i (x) ??? ??}. A restoration move with target scan solves DISPLAYFORM7 where z (k,i) is the solution to Eqs. FORMULA2 or FORMULA0 with c(x (k) ) replaced with c i (x (k) ), and thus is equal to Eqs. (4) or (13) with c( DISPLAYFORM8 A is a set of candidate adversarial calsses, which can be all the incorrect classes if the number of classes is small, or which can be a subset of the adversarial classes with the highest logits otherwise.

Experiments show that target scan is necessary only in the first few restoration moves, when the closest and highest adversarial classes are likely to be distinct.

Therefore, the computation cost will not increase too much.

Initialization: The initialization of x (0) can be either deterministic or random as follows DISPLAYFORM9 where U{ [???u, u] n } denotes the uniform random distribution in [???u, u] n .

Similar to PGD, we can perform multiple trials with random initialization to find a better local optimum.

Final Tuning MARGINATTACK can only cause misclassification when c(x) ??? ??.

To make sure the attack is successful, the final iterations of MARGINATTACK consists of restoration moves only, DISPLAYFORM10 and no projection moves, until a misclassification is caused.

This can also ensure the final solution satisfies the box constraint (because only the restoration move incorporates the box constraint).Summary: Alg.

1 summarizes the MARGINATTACK procedure.

As for the complexity, each restoration move or projection move requires only one backward propagation, and thus the computational complexity of each move is comparable to one iteration of most attack algorithms.

This section compares MARGINATTACK with several state-of-the-art adversarial attack algorithms in terms of the perturbation norm and computation time on image classification benchmarks.

Three regularly trained models are evaluated on.??? MNIST (LeCun et al., 1998): The classifier is a stack of two 5 ?? 5 convolutional layers with 32 and 64 filters respectively, followed by two fully-connected layers with 1,024 hidden units.??? CIFAR10 BID6 ): The classifier is a pre-trained ResNet32 BID4 provided by TensorFlow.

5 .??? ImageNet BID16 :

The classifier is a pre-trained ResNet50 BID4 provided by TensorFlow Keras 6 .

Evaluation is on a validation subset containing 10,000 images.

The range of each pixel is [0, 1] for MNIST, and [0, 255] for CIFAR10 and ImageNet.

The settings of MARGINATTACK and baselines are listed below.

Unless stated otherwise, the baseline algorithms are implemented by cleverhans BID14 .

The hyperparameters are set to defaults if not specifically stated.??? CW BID1 :

The target and evaluation norm is 2 .

The learning rate is set to 0.05 for MNIST, 0.001 for CIFAR10 and 0.01 for ImageNet, which are tuned to its best performance.

The number of binary steps for multiplier search is 10.

??? DeepFool (Moosavi Dezfooli et al., 2016): The evaluation norm is 2 .??? FGSM BID3 : FGSM is implemented by authors.

The step size is searched to achieve zero-confidence attack.

The evaluation distance metric is ??? .???

PGD BID10 :

The target and evaluation norm are ??? .

The learning rate is set to 0.01 for MNIST, and 0.05 for CIFAR10 and 0.1 for ImageNet.??? MARGINATTACK: Two versions of MARGINATTACK are implemented, whose target and evaluation norms are 2 , and ??? , respectively.

The hyperparmeters are detailed in TAB4 in the appendix.

The first 10 restoration moves are with target scan, and the last 20 moves are all restoration moves.

The number of iterations/moves is set to 2,000 for CW, 200 with 10 random starts for PGD and MARGINATTACK (except for ImageNet where there is only one random run), and 200 for the rest.

Except for PGD, all the other attacks are zero-confidence attacks.

For these attacks, we plot the CDF of the margins of the validation data, which can also be interpreted as the percentage success rate of these attacks as a function of perturbation level.

FIG3 plots the success rate curves, where the upper panel shows the 2 attacks, and the lower one shows ??? attacks.

As can be observed, the MARGINATTACK curves are above all other algorithms at all perturbation levels and in all datasets.

CW is very close to MARGINATTACK on MNIST and CIFAR10, but MARGINATTACK maintains a 3% advantage on MNIST and 1% on CIFAR10.

It seems that CW is unable to converge well within 2,000 iterations on ImageNet, although the learning rate has been tuned to maximize its performance.

MARGINATTACK, on the other hand, converges more efficiently and consistently.

To obtain a success rate curve for PGD, we have to run the attack again and again for many different perturbation levels, which can be time-consuming for large datasets (this shows an advantage of zero-confidence attacks over fix-perturbation attacks).

Instead, we choose four perturbation levels for each attack scenario to compare.

The perturbation levels are chosen to roughly follow the 0.2, 0.4, 0.6 and 0.8 quantiles of the MARGINATTACK margins.

TAB1 compares the success rates under the chosen quantiles among the ??? attacks.

We can see that MARGINATTACK outperforms PGD under all the perturbation levels, and that both significantly dominate FGSM.

We also evaluate MARGINATTACK on the MNIST Adversarial Examples Challenge 7 , which is a challenge of attacking an MNIST model adversarially trained using PGD with 0.3 perturbation level.

Same as the PGD baseline listed, MARGINATTACK is run with 50 random starts, and the initialization perturbation range u = 0.3.

The number of moves is 500.

The target norm is ??? .

b n = 5 and a n is set as in Eq. (10).

The rest of the configuration is the same as in the previous experiments.

Table 2 lists the success rates of different attacks under 0.3 perturbation level.

The baseline algorithms are all fix-perturbation attacks, and their results are excerpted from the challenge white-box attack leaderboard.

As can be seen, MARGINATTACK, as the only zero-confidence attack algorithm, has the second best result, which shows that it performs competitively against the state-of-the-art fix-perturbation attacks.

We would like to revisit the convergence plot of the constraint value c(x) and perturbation norm d(x) of as in FIG2 .

We can see that MARGINATTACK converges very quickly.

In the example shown in the figure, it is able to converge within 20 moves.

Therefore, MARGINATTACK can be greatly accelerated.

If margin accuracy is the priority, a large number of moves, e.g. 200 as in our experiment, would help.

However, if efficiency is the priory, a small number of moves, e.g. 30, suffices to produce a decent attack.

To further assess the efficiency of MARGINATTACK, Tab.

3 compares the running time (in seconds) of attacking one batch of images, implemented on a single NVIDIA TESLA P100 GPU.

The batch size is 200 for MNIST and CIFAR10, and 100 for ImageNet.

The settings are the same as stated in section 4.1, except that for a better comparison, the number of iterations of CW is cut down to 200, and PGD and MARGINATTACK runs one random pass, so that all the algorithms have the same iteration/moves.

Only the 2 versions of MARGINATTACK are shown because the other versions have similar run times.

As shown, running time of MARGINATTACK is much shorter than CW, and is comparable to DeepFool and PGD.

CW is significantly slower that the other algorithms because it has to run multiple trials to search for the best Lagrange multiplier.

Note that DeepFool and CW enable early stop, but MARGINATTACK does not.

Considering MARGINATTACK's fast convergence rate, the running time can be further reduced by early stop.

We have proposed MARGINATTACK, a novel zero-confidence adversarial attack algorithm that is better able to find a smaller perturbation that results in misclassification.

Both theoretical and empirical analyses have demonstrated that MARGINATTACK is an efficient, reliable and accurate adversarial attack algorithm, and establishes a new state-of-the-art among zero-confidence attacks.

What is more, MARGINATTACK still has room for improvement.

So far, only two settings of a (k) and b (k) are developed, but MARGINATTACK will work for many other settings, as long as assumption 5 is satisfied.

Authors hereby encourage exploring novel and better settings for the MARGINATTACK framework, and promote MARGINATTACK as a new robustness evaluation measure or baseline in the field of adversarial attack and defense.

This supplementary material aims to prove Thm.

1.

Without the loss of generality, K in Eq. (9) in set to 0.

Before we prove the theorem, we need to introduce some lemmas.

Lemma 1.1.

If assumption 3 in Thm.

1 holds, then ???x ??? B DISPLAYFORM0 Proof.

According to Eq. (5), for 2 norm, DISPLAYFORM1 for ??? norm, DISPLAYFORM2 Lemma 1.2.

Given all the assumptions in Thm.

1, where DISPLAYFORM3 and assuming DISPLAYFORM4 where DISPLAYFORM5 A and B are defined in Eq. (32).According to assumption 8, this implies DISPLAYFORM6 at the rate of at least 1/n ?? .Proof.

As a digression, the second term in Eq. FORMULA0 is well defined, because DISPLAYFORM7 is upper bounded by Lem.

1.1 and assumptions 3.Back to proving the lemma, we will prove that each restoration move will bring c(x (k) ) closer to 0, while each projection move will not change c(x (k) ) much.

First, for the restoration move DISPLAYFORM8 The first line is from the generalization of Mean-Value Theorem with jump discontinuities, and ?? = tz (k) + (1 ??? t)x (k) and t is a real number in [0, 1].

The second line is from Eq. (4).

The last line is from assumptions 4 and 7 and Eq. (19).Next, for the projection move DISPLAYFORM9 The first line is from the fact that assumption 3 implies that c(x) is M -Lipschitz continuous.

DISPLAYFORM10 for some M d and M s .

To see this, for 2 norm DISPLAYFORM11 where b is defined as the maximum perturbation norm ( 2 ) within B, i.e. DISPLAYFORM12 which is well defined because B is a tight set.

For ??? norm, DISPLAYFORM13 Note that Eq. (26) also holds for other norms.

With Eq. (26) and assumption 8, Eq. FORMULA1 becomes DISPLAYFORM14 Combining Eqs. FORMULA1 and FORMULA2 we have DISPLAYFORM15 where DISPLAYFORM16 According to assumption 7, 0 < A < 1.

Also, according to Eq. FORMULA1 , DISPLAYFORM17 and thus DISPLAYFORM18 If DISPLAYFORM19 Otherwise, Eq. (34) implies DISPLAYFORM20 This concludes the proof.

Lemma 1.3.

Given all the assumptions in Thm.

1, and assuming DISPLAYFORM21 Proof.

First, for restoration move DISPLAYFORM22 ??m 2 (38) Line 4 is given by Eq. (3).

Line 5 is derived from Lem.

1.1.

The last line is from Lem.

1.2.

??m 2 (39) where DISPLAYFORM0 It can easily be shown that DISPLAYFORM1 Therefore DISPLAYFORM2 Combining Eqs. FORMULA2 and FORMULA1 , we have DISPLAYFORM3 Step Case: Assume Eq. (51) holds ???k ??? K, then Eqs. (31) and (45) holds ???k ??? K.??? Proving |c(x (K+1) )| ??? C: DISPLAYFORM4 where the last inequality is given by Eq. (50).

DISPLAYFORM5 Notice that from Eq. (50), 0 DISPLAYFORM6 where the last inequality is given by Eq. (50).

DISPLAYFORM7 Lemma 1.5.

Under the assumptions in Thm.

1 DISPLAYFORM8 Proof.

From Thm.

2, a solution, denoted as x , to min DISPLAYFORM9 would satisfy DISPLAYFORM10 If P [x * ??? x 0 ] = 0, there are two possibilities.

The first possibility is that x * is not a solution to Eq. (62), which contradicts with the first order optimality condition that x * must satisfy.

The second possibility is there are multiple solutions to the problem in Eq. FORMULA1 , and x and x * are both its solutions.

This can happen if d(??) is 1 or ??? norm.

By definition DISPLAYFORM11 * is a local minimum to Eq. (1), ???j ??? I, ?? < 1, ????? < ??, DISPLAYFORM12 65) Otherwise, if c(x ?? ) ??? 0, then x ?? is a feasible solution to the problem in Eq. FORMULA0 and DISPLAYFORM13 which contradicts with the assumption that x * is a unique local optimum in B. DISPLAYFORM14 takes discrete values.

Therefore, to satisfy assumption 2, s j (x ?? ) = s j (x * ), which implies DISPLAYFORM15 The first inequality is because DISPLAYFORM16 Eqs. FORMULA6 and FORMULA6 cause a contradiction.

Now we are ready to prove Thm.

1.Proof of Thm.

1.

From Lems.

1.2, 1.3 and 1.4, we can established that Eqs. FORMULA0 and FORMULA2 holds under all the assumptions in Thm.

1.

The only thing we need to prove is that Eqs. FORMULA0 and FORMULA2 necessarily implies lim k??? x (k) ??? x 0 2 = 0.First, from Lem.

1.5 DISPLAYFORM17 Then, ???x ??? B s.t.

P [x ??? x 0 ] 2 2 = 0, we have x ??? x = ??s(x ).

From assumption 4, we know that c(x ) is monotonic along x ??? x = ??s(x ).

Therefore, x * is the only point in B that satisfies P [x ??? x 0 ] 2 2 = 0 and c(x ) = 0.

Also, notice that P [x ??? x 0 ] and c(x) are both continuous mappings.

This concludes the proof.

Notice that the product of ?? and ??? T c(x )

y is constant, so if ?? is to be minimized, then ??? T c(x )y needs to be maximized.

Namely, y can be determined by solving which is the definition of dual norm.

Therefore y = s(x ) (74) Plug Eq. (74) into the constraint in Eq. (72), we can solve for ??.

This concludes the proof.

As a remark, Thm.

2 is applicable to the optimization problems in Eqs. FORMULA2 and FORMULA1 Proof.

Since c(x) is convex in B, we have ???x DISPLAYFORM18 Further, assume x satisfies ??? T c(x * )(x ??? x * ) = 0 (77) Then we have P T P (x ??? x * ) = P (x ??? x * ) = x ??? x * (78) where the first equality is from the fact that P is an orthogonal projection matrix under 2 norm; the second equality is from the fact that the projection subspace of P is orthogonal to ???c(x * ) by construction.

Also, from Lem.

1.5, we have DISPLAYFORM19 Plug Eqs. FORMULA7 to FORMULA7 into FORMULA6 , we have DISPLAYFORM20 On the other hand, let ?? = m a /2 x * ??? x 0 2 , then ???x satisfying Eq. (77) and DISPLAYFORM21 where the second line comes from Eq. (75) and the fact that x * is the optimal solution to the problem in Eq. (62).

Combining Eqs. FORMULA8 and FORMULA0 , we know that assumption 5 holds with strict inequality for x satisfying Eq. (77) and x = x * .???

T c(x)P T P (x ??? x * ), ??? T d(x ??? x 0 )P T P (x ??? x 0 ) and ??(x ??? x 0 ) T P T P (x ??? x 0 ) are continuous functions, and therefore ???B where assumption 5 also holds.

This concludes the proof.

<|TLDR|>

@highlight

This paper introduces MarginAttack, a stronger and faster zero-confidence adversarial attack.