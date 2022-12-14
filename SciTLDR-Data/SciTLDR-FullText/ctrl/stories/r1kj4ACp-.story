Deep learning achieves remarkable generalization capability with overwhelming number of model parameters.

Theoretical understanding of deep learning generalization receives recent attention yet remains not fully explored.

This paper attempts to provide an alternative understanding from the perspective of maximum entropy.

We first derive two feature conditions that softmax regression strictly apply maximum entropy principle.

DNN is then regarded as approximating the feature conditions with multilayer feature learning, and proved to be a recursive solution towards maximum entropy principle.

The connection between DNN and maximum entropy well explains why typical designs such as shortcut and regularization improves model generalization, and provides instructions for future model development.

Deep learning has achieved significant success in various application areas.

Its success has been widely ascribed to the remarkable generalization ability.

Recent study shows that with very limited training data, a 12-layer fully connected neural network still generalizes well while kernel ridge regression easily overfits with polynomial kernels of more than 6 orders (Wu et al., 2017) .

Classical statistical learning theories like Vapnik-Chervonenkis (VC) dimension (Maass, 1994) and Rademacher complexity (Neyshabur et al., 2015) evaluate generalization based on the complexity of the target function class.

It is suggested that the models with good generalization capability are expected to have low function complexity.

However, most successful deep neural networks already have over 100 hidden layers, e.g., ResNet BID2 and DenseNet BID3 for image recognition.

The number of model parameters in these cases is even larger than the number of training samples.

Statistical learning theory cannot well explain the generalization capability of deep learning models (Zhang et al., 2017) .Maximum Entropy (ME) is a general principle for designing machine learning models.

Models fulfilling the principle of ME make least hypothesis beyond the stated prior data, and thus lead to least biased estimate possible on the given information BID5 .

Appropriate feature functions are critical in applying ME principle and largely decide the model generalization capability BID1 .

Different selections of feature functions lead to different instantiations of maximum entropy models (Malouf, 2002; Yusuke & Jun'ichi, 2002) .

The most simple and wellknown instantiation is that ME principle invents identical formulation of softmax regression by selecting certain feature functions and treating data as conditionally independent (Manning & Klein, 2003) .

It is obvious that softmax regression has no guaranty of generalization, indicating that inappropriate feature functions and data hypothesis violates ME principle and undermines the model performance.

It remains not fully studied how to select feature functions to maximally fulfill ME principle and guarantee the generalization capability of ME models.

Maximum entropy provides a potential but not-ready way to understand deep learning generalization.

This paper is motivated to improve the theory behind applying ME principle and use it to understand deep learning generalization.

We research on the feature conditions to equivalently apply ME principle, and indicates that deep neural networks (DNN) is essentially a recursive solution to approximate the feature conditions and thus maximally fulfill ME principle.??? In Section 2, we first revisit the relation between generalization and ME principle, and conclude that models well fulfilling ME principle requires least data hypothesis so to possess good generalization capability.

One general guideline for feature function selection is to transfer the hypothesis on input data to the constrain on model features 1 .

This demonstrates the role of feature learning in designing ME models.??? Section 3 addresses what features to learn.

Specifically, we derive two feature conditions to make softmax regression strictly equivalent to the original ME model (denoted as Maximum Entropy Equivalence Theorem).

That is, if the utilized features meet the two conditions, simple softmax regression model can fulfill ME principle and guarantee generalization.

These two conditions actually specify the goal of feature learning.??? Section 4 resolves how to meet the feature conditions and connects DNN with ME.

Based on Maximum Entropy Equivalence Theorem, viewing the output supervision layer as softmax regression, the DNN hidden layers before the output layer can be regarded as learning features to meet the feature conditions.

Since the feature conditions are difficult to be directly satisfied, they are optimized and recursively decomposed to a sequence of manageable problems.

It is proved that, standard DNN uses the composition of multilayer non-linear functions to realize the recursive decomposition and uses back propagation to solve the corresponding optimization problem.??? Section 5 employs the above ME interpretation to explain some generalization-related observations of DNN.

Specifically, from the perspective of ME, we provide an alternative way to understand the connection between deep learning and Information Bottleneck (Shwartz-Ziv & Tishby, 2017) .

Theoretical explanations on typical generalization design of DNN, e.g., shortcut, regularization, are also provided at last.

The contributions are summarized in three-fold:1.

We derive the feature conditions that softmax regression strictly apply maximum entropy principle.

This helps understanding the relation between generalization and ME models, and provides theoretical guidelines for feature learning in these models.2.

We introduce a recursive decomposition solution for applying ME principle.

It is proved that DNN maximally fulfills maximum entropy principle by multilayer feature learning and softmax regression, which guarantees the model generalization performance.3.

Based on the ME understanding of DNN, we provide explanations to the information bottleneck phenomenon in DNN and typical DNN designs for generalization improvement.

In machine learning, one common task is to fit a model to a set of training data.

If the derived model makes reliable predictions on unseen testing data, we think the model has good generalization capability.

Traditionally, overfitting refers to a model that fits the training data too well but generalize poor to testing data, while underfitting refers to a model that can neither fits the training data nor generalize to testing data (Vapnik & Vapnik, 1998) .As a criterion for learning machine learning models, ME principle makes null hypothesis beyond the stated prior data (X, Y ) where X, Y denote the original sample representation and label respectively.

To facilitate the discussion between generalization and maximum entropy, we revisit generalization, overfitting and underfitting by how much data hypothesis is assumed by the model:??? Underfitting: Underfitting occurs when the model's data hypothesis is not satisfied by the training data.??? Overfitting: Overfitting occurs when the model's data hypothesis is satisfied by the training data, but not satisfied by the testing data.??? Generalization: According to ME principle, a model with good generalization capability is expected to have as less extra hypothesis on data (X, Y ) as possible.

The above interpretation of underfitting and overfitting can be illustrated with the toy example in FIG0 .

The underfitting model in solid line assumes linear relation on (X, Y ), which is not satisfied by the training data.

The model in dot dash line assumes 5-order polynomial relation on (X, Y ), which perfectly fits to the training data.

However, it is obvious that the hypothesis generalizes poorly to testing data and the 5-order polynomial model tends to overfitting.

A coarse conclusion reaches that, introducing extra data hypothesis, whether or not fitting well to the training data, will lead to degradation of model generalization capability.

non-ME model with data hypothesis, original ME model without data hypothesis, simple model with feature constraint (equivalent to original ME).One question arises: why ME models cannot guarantee good generalization?

Continuing the discussion in Introduction, to enable the enumeration of predicate states, most ME models explicitly or implicitly introduce extra data hypothesis, e.g., softmax regression assumes independent observations when applying ME principle.

Imposing extra data hypothesis actually violates the ME principle and degrades the model to non-ME (Maximum Entropy) model.

The dilemma is: generalization requires no extra data hypothesis, but it is difficult to derive simple models without data hypothesis.

Is there solution to apply ME principle without imposing hypothesis on the original data?While the input original data (X, Y ) is fixed and maybe not compatible with the hypothesis, we can introduce model feature T sufficient to represent data, and transfer the data hypothesis to feature constraint.

Ideally the model defined on feature T is a simple ME model (e.g., softmax regression), so that we can easily apply ME principle without imposing extra data hypothesis.

In this case, the simple model plus feature constraint constitutes an equivalent implementation to ME principle and possesses good generalization capability.

FIG0 (right) illustrates these model settings with/without data hypothesis.

It is easy to see that, from the perspective of applying ME, feature learning works between data and feature, with goal to realizing the feature constraints.

According to the above discussions, when applying ME principle, the problem becomes how to identify the equivalent feature constraints and simple models.

Since the output layer of DNN is usually softmax regression, this section will explore what feature constraints can make softmax regression equivalent to the original ME model.

We first review the definition of original ME model and feature-based softmax model in this subsection.

Note that in defining the original ME model, instead of using predicate functions as most ME models, we deliver the constraints of (X, Y ) with joint distribution equality 2 .

Before defining the softmax model, to facilitate the transfer of data hypothesis to feature constraint, we first provide the definition of feature T over input data X, and then derive the general formulation of feature-based ME model.

Feature-based softmax model can be seen as a special case of feature-based ME model.

Y , the task is to find a good prediction of Y using X. The prediction?? needs to maximize the conditional entropy H(?? |X) while preserving the same distribution with data (X, Y ).

This is formulated as: DISPLAYFORM0 This optimization question can be solve by lagrangian multiplier method : DISPLAYFORM1 The above equation can be equivalently written with the original defined predicate function in BID1 : DISPLAYFORM2 is predicate function, which equalizes 1 when (X, Y ) satisfies a certain status: DISPLAYFORM3 The solution to the above problem is: DISPLAYFORM4

According to the above definition of feature T , feature-based maximum entropy model can be formulated as : DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 where ?? i (y) and b(y) denote functions of y: ?? i (y) is weight for feature T i , and b(y) is the bias term in softmax regression.

From Eqn.

(3) and Eqn.

FORMULA4 , we find it impossible to traverse all status of (X, Y ), making the original ME problem difficult to solve.

To address this, many studies are devoted to designing special kind of predicate functions to make the problem solvable.

However, recalling the discussion on ME and generalization in Section 2, if extra data hypothesis is imposed on (X, Y ), the generalization capability of the derived ME model will be undermined.

An alternative solution is to design the predicate function by imposing constraints on intermediate feature T instead of directly on input data (X, Y ).On imposing the feature constraints, two issues need to be considered: (1) not arbitrary T makes the feature-based ME model equivalent to the original ME model; (2) under the premise of equivalence, T should make the derived ME model solvable (like the softmax regression).

Based on these considerations, we prove and derive two necessary and sufficient feature conditions to make feature-based softmax regression (Definition 3) strictly equivalent to the original ME model (Definition 1).

DISPLAYFORM0 The proof to the theorem is given in Section A in the Appendix.

The first condition ensures that feature-based ME model is equivalent to the original ME model, and thus be denoted as equivalent condition.

The second condition makes feature-based ME model solvable and converted as featurebased softmax regression problem.

We denote the second condition as solvable condition.

This theorem on one hand derives operable feature constraints that softmax regression is equivalent to the original ME model, on the other hand provides theoretical guidance to feature learning with goal of improving model generalization.

Based on the derived Maximum Entropy Equivalence Theorem, the original ME model is equivalent to a feature-based softmax model with two feature constraints.

In this way, from the perspective of maximum entropy, if DNN uses softmax as the output layer, the previous latent layers can be seen as the process of feature learning to approach these constraints.

However, these feature constraints are difficult to be satisfied directly, and therefore being decomposed to many smaller and manageable problems for approximation.

This section claims that DNN actually uses the composition of multilayer non-linear functions to realize a recursive decomposition towards these feature constraints.

In the following we will first introduce the recursive decomposition operation to difficult problem, and then prove that DNN with sigmoid-activated hidden layers and softmax output layer is exactly a recursive decomposition solution towards the original ME model.

A common way to solve a difficult problem is relaxing the problem to an easier one, like majorizeminimize algorithms BID4 Obviously, according to Maximum Entropy Equivalence Theorem, the original ME problem is such a decomposable problem.

If the original problem P is decomposable, and P is equivalent to a manageable problem P 1 with additional constraints C 1 , we denote it as P = P 1 + C 1 .

In this case, we can solve P 1 + C 1 instead of directly solving P .Since P 1 is easy to solve, it remains to satisfy the constraint C 1 .

The constrain C 1 can be approximately satisfied by an optimization problem p 1 as its upper bound.

From Definition 4, we know that p 1 is only related to the extra added parameters.

Now, we have P = P 1 + p 1 .If p 1 is solvable, we can use an algorithm similar to EM to solve P 1 + p 1 :(1) fix parameters in p 1 and optimize P 1 ; (2) fix parameters in P 1 and optimize p 1 ; (3) iterate (1) and FORMULA3 until convergence.

However, sometimes p 1 is still difficult to solve but decomposable.

In this case, we need further decompose p 1 to a manageable problem P 2 with smaller problem p 2 under condition that p 1 = P 2 + p 2 .

The problem transfers to solve P = P 1 + P 2 + p 2 in a similar iterative way.

If p 2 is still difficult, we can repeat this process to get DISPLAYFORM0 Since this constitutes a recursive process, we denote this way of relaxation as recursive decomposition.

The optimization process of recursive decomposition is also recursive.

Given the decomposition of difficult problem P = P 1 + ?? ?? ?? + P l + +P L , we have the following optimization process:(1) fix parameters in P 2 , ?? ?? ?? , P L and optimize P 1 ; (2) fix parameters in P 1 , P 3 , ?? ?? ?? , P L and optimize DISPLAYFORM1 The premise behind this method is that, if we change the constraints of problem to a minimum problem of its upper bound, the new problem is still a better approximation than the original problem without constraint.

This subsection will explain that DNN is actually a recursive decomposition solution towards maximum entropy, and the back propagation algorithm is a realization of parameter optimization to the model.

According to Maximum Entropy Equivalence Theorem, the original ME model is equivalent to softmax model with two feature constraints, which is a typical decomposable problem.

In the following we employ the above introduced recursive decomposition method to solve it: the original ME problem is the difficult problem P , softmax model is the manageable problem P 1 , and the two conditions constitutes the constraints C 1 related only to feature T .While the feature constraints C 1 are still difficult to be satisfied, we relax the constraints to smaller problems using the following Feature Constraint Relaxation Theorem.

can be relaxed to the following optimization problem: DISPLAYFORM0

This theorem is proved in Section B in the Appendix.

The above relaxed minimization problem constitutes p 1 , which optimizes feature T = T 1 , T 2 , ..., T n .

Using the derivation from the proof for the above theorem, we know that minimization of DISPLAYFORM0 The fact that T i , T j is independent allows to split p 1 further to n smaller problems p 11 , ?? ?? ?? , p 1i , ?? ?? ?? , p 1n , where p 1i is an optimization problem with the same formulation as Eqn. (8) but defined over T i .Note that the new optimization problems p 1i are still difficult ME problems, which need to be decomposed and relaxed recursively till problem p Li ??? P Li where P Li is manageable.

According to Maximum Entropy Equivalence Theorem, each decomposed manageable problem P li is realized by a softmax regression.

Since feature T i is binary random variable, the models for feature learning change to logistic regression.

For a L-depth recursive decomposition, the original ME model is approximated by ??? L l=1 n l logistic regression models and one softmax regression model (n l denotes the number of features at the l-th recursion depth).

It is easy to find that this structure perfectly matches a basic DNN model: the depth of recursion corresponds to the network hidden layer (but in opposite index, i.e., the L-th depth recursion corresponds to the 1st hidden layer), the number of features at each recursion correspond to the number of hidden neurons at each layer, and the logistic regression corresponds to one layer of linear regression with sigmoid activation function.

Therefore, we reach a conclusion that DNN is a recursive decomposition solution towards maximum entropy.

The generalization capability is thus guaranteed under the ME principle.

This explains why DNN is designed as composition of multilayer non-linear functions.

Moreover, the model learning technique, backpropagation, actually follows the same spirits as the optimization process in recursive decomposition for DNN parameter optimization.

After modeling DNN as a recursive decomposition solution towards ME, in this section, we use the ME theory to explain some generalization-related phenomenon about DNN and provide interpretations on DNN structure design.

Specifically, Section 5.1 explains why Information Bottleneck exists in DNN, and Section 5.2 explains why certain DNN structure design can improve generalization.

In the Information Bottleneck (IB) theory (Tishby et al., 1999) , given data (X, Y ), the optimization target is to minimize mutual information I(X; T ) while T is a sufficient statistic satisfying

Now, we prove that the output of constraint problem in ME model is sufficient to satisfy the Information Bottleneck theory.

In other words, basic DNN model with softmax output fulfills IB theory.

Corollary (Corollary of ME's interpretation on Information Bottleneck).

The output of maximum entropy problem min DISPLAYFORM0 is sufficient condition to the IB optimization problem: min T

The proof of this corollary is available in Section C in the Appendix.

Since DNN is an approximation towards ME, this result explains why DNN tends to increase I(T ; Y ) while reduce I(X; T ) and the Information Bottleneck phenomenon in DNN.

DNN has some typical generalization designs, e.g., shortcut, regularization, etc.

This subsection explains why these designs can improve model generalization capability.

Shortcut is widely used in many CNN framework.

The traditional explanation is that shortcut makes information flow more convenient, so we can train deeper networks BID2 .

But this cannot explain why shortcut contributes to a better performance.

According to the above modeling of DNN as ME, CNN is a special kind of DNN where we use part of input X at each layer to construct the model.

The actual input of CNN is related to the size of corresponding convolution kernel, and receives only part of X within its receptive field.

Shortcut enriches different size of receptive fields and thus reserve more information from X during problem decomposition in the recursion process.

The regularization in DNN can be seen as playing similar role as the feature conditions in Maximum Entropy Equivalence Theorem.

BID0 demonstrated that the regularization design, like sgd, L2-Norm, dropout, is equal to minimizing the mutual information I(X; T ).

The ME modeling of DNN also sheds some light on the role of network depth in generalization performance.

Following the recursive decomposition discussion, it seems network with more layers leads to deeper recursion and thus closer approximation towards ME.

However, it is noted that we are using relaxed optimization to replace the original constraints.

Considering the continuous minimization of upper bound, simple DNN with too many hidden layers may not always guarantees the performance.

We emphasize that for those CNNs with good architecture, more hidden layers bring richer receptive fields and less loss of information in X. In this case, increasing network depth will contribute to generalization improvement.

This paper regards DNN as a solution to recursively decomposing the original maximum entropy problem.

From the perspective of maximum entropy, we ascribe the remarkable generalization capability of DNN to the introduction of least extra data hypothesis.

The future work goes in two directions: (1) first efforts will be payed to identifying connections with other generalization theories and explaining more DNN observations like the role of ReLu activation and redundant features; (2) the second direction is to improve and exploit the new theory to provide instructions for future model development of traditional machine learning as well as deep learning methods.

The two feature conditions can be separately proved.

Firstly, we prove the necessity and sufficiency of condition 1 (equivalent condition) for equivalence of feature-based ME model and original ME model.

Secondly, condition 2 (solvable condition) guarantees the solution of feature-based ME model in a manageable form (i.e., softmax regression).To prove this theorem, we first prove the following three Lemmas.

Lemma 1.

If T is a set of random variables only related to X, and T satisfies condition 1, i.e., mutual information I(X; Y |T ) = 0, then DISPLAYFORM0 Proof.

Since T is a set of random variables only related to X, it is obvious to have DISPLAYFORM1 So the task leaves to prove DISPLAYFORM2 Recall that T is a set of random variables only related to X, then DISPLAYFORM3 We further have T satisfying condition 1: DISPLAYFORM4 Similarly, X ??? T ????? is Marcov chain, hence we have: DISPLAYFORM5 T is defined feature function on X, so P (X|T ) is a constant.

We further have: DISPLAYFORM6 Note that E P (X,Y ) = E P (X,?? ) indicates that the predicate functions satisfy Eqn.

FORMULA3 in the definition of original ME model, and thus is equivalent to P (X, Y ) = P (X,?? ).With Eqn.

FORMULA18 and Eqn.

FORMULA23 , we finally have: DISPLAYFORM7 Lemma 2.

If T is a set of random variables only related to X, and DISPLAYFORM8 Proof.

Since T is a set of random variables only related to X, we have DISPLAYFORM9 X ??? T ????? is Marcov chain, we have: DISPLAYFORM10 Additionally , DISPLAYFORM11 So we can derive: DISPLAYFORM12 Lemma 3.

If T is a set of random variables only related to X that satisfies condition 1, and DISPLAYFORM13 Proof.

T is a set of random variables only related to X: DISPLAYFORM14 T satisfies condition 1, so: DISPLAYFORM15 With Eqn.

FORMULA31 Further using Lemma1, we can derive: DISPLAYFORM16 Therefore, we get DISPLAYFORM17 DISPLAYFORM18 Proof.

With Lemma1, Lemma2 and Lemma3, we derive that condition 1 is necessary and sufficient for the equivalence of original ME model and the following feature-based ME model: DISPLAYFORM19 The above optimization problem can be solved with the following solution: DISPLAYFORM20 DISPLAYFORM21 However, this solution is too complex to apply.

With n features T = {T 1 , T 2 , ..., T n } and m different classes of Y , there will be m * 2 n different f i (T, Y ).

Condition 2 assumes the conditional independence among feature (T i , T j ), which derives that the joint distribution equation DISPLAYFORM22 According to definition, for each T i , we have P (T i = 1|X = x) = t i (x) and P (T i = 0|X = x) = 1 ??? t i (x).

Therefore, under condition 2, the predicate functions will be: DISPLAYFORM23

We then have: DISPLAYFORM0 where ?? denotes variable about y.

We further define b(y) = ??? i ?? i0 and ?? i (y) = (?? i1 ??? ?? i0 ), then the solution of Eqn.

FORMULA9 and Eqn. (18) change to: DISPLAYFORM1 DISPLAYFORM2 This is the identical formulation to the general softmax regression model as in Definition 3.

It also explains why we have bias term in the softmax model.

Note that t i (x) need not to be in range [0, 1] when we use the softmax model, as we can change ?? and b to achieve translation and scaling.

where S(T ) denotes the output of softmax model if input is T .Proof.

Since T is only related to X, T i ??? X ??? T j is Marcov chain, and DISPLAYFORM3 We can relax the minimization problem to minimize its upper bound instead, so Note that?? = S(T ): X ??? T ????? is M arcov chain Recall that?? is solution to the problem P 1 , and P 1 has constraint E P (X,Y ) = E P (X,?? ) .

Same as E P (X,Y ) = E P (X,S(T )) , we have is sufficient condition to the IB optimization problem: DISPLAYFORM4

Proof.

Summing up Lemma4 and Lemma5, the output of the constraint problem is sufficient to solving the IB optimization problem .

<|TLDR|>

@highlight

We prove that DNN is a recursively approximated solution to the maximum entropy principle.

@highlight

Presents a derivation which links a DNN to recursive application of maximum entropy model fitting.

@highlight

The paper aims to provide a view of deep learning from the perspective of maximum entropy principle.