Neural networks could misclassify inputs that are slightly different from their training data, which indicates a small margin between their decision boundaries and the training dataset.

In this work, we study the binary classification of linearly separable datasets and show that linear classifiers could also have decision boundaries that lie close to their training dataset if cross-entropy loss is used for training.

In particular, we show that if the features of the training dataset lie in a low-dimensional affine subspace and the cross-entropy loss is minimized by using a gradient method, the margin between the training points and the decision boundary could be much smaller than the optimal value.

This result is contrary to the conclusions of recent related works such as (Soudry et al., 2018), and we identify the reason for this contradiction.

In order to improve the margin, we introduce differential training, which is a training paradigm that uses a loss function defined on pairs of points from each class.

We show that the decision boundary of a linear classifier trained with differential training indeed achieves the maximum margin.

The results reveal the use of cross-entropy loss as one of the hidden culprits of adversarial examples and introduces a new direction to make neural networks robust against them.

Training neural networks is challenging and involves making several design choices.

Among these are the architecture of the network, the training loss function, the optimization algorithm used for training, and their hyperparameters, such as the learning rate and the batch size.

Most of these design choices influence the solution obtained by the training procedure and have been studied in detail BID9 BID4 BID5 Wilson et al., 2017; BID17 BID19 .

Nevertheless, one choice has been mostly taken for granted when the network is trained for a classification task: the training loss function.

Cross-entropy loss function is almost the sole choice for classification tasks in practice.

Its prevalent use is backed theoretically by its association with the minimization of the Kullback-Leibler divergence between the empirical distribution of a dataset and the confidence of the classifier for that dataset.

Given the particular success of neural networks for classification tasks BID11 BID18 BID5 , there seems to be little motivation to search for alternatives for this loss function, and most of the software developed for neural networks incorporates an efficient implementation for it, thereby facilitating its use.

Recently there has been a line of work analyzing the dynamics of training a linear classifier with the cross-entropy loss function BID15 b; BID7 .

They specified the decision boundary that the gradient descent algorithm yields on linearly separable datasets and claimed that this solution achieves the maximum margin.1 However, these claims were observed not to hold in the simple experiments we ran.

For example, FIG6 displays a case where the cross-entropy minimization for a linear classifier leads to a decision boundary which attains an extremely poor margin and is nearly orthogonal to the solution given by the hard-margin support vector machine (SVM).We set out to understand this discrepancy between the claims of the previous works and our observations on the simple experiments.

We can summarize our contributions as follows.

Cross-entropy min.

decision boundary (poor margin)Figure 1: Orange and blue points represent the data from two different classes in R 2 .

Cross-entropy minimization for a linear classifier on the given training points leads to the decision boundary shown with the solid line, which attains a very poor margin and is almost orthogonal to the solution given by the SVM.1.

We analyze the minimization of the cross-entropy loss for a linear classifier by using only two training points, i.e., only one point from each of the two classes, and we show that the dynamics of the gradient descent algorithm could yield a poor decision boundary, which could be almost orthogonal to the boundary with the maximum margin.2.

We identify the source of discrepancy between our observations and the claims of the recent works as the misleading abbreviation of notation in the previous works.

We clarify why the solution obtained with cross-entropy minimization is different from the SVM solution.3.

We show that for linearly separable datasets, if the features of the training points lie in an affine subspace, and if the cross-entropy loss is minimized by a gradient method with no regularization to train a linear classifier, the margin between the decision boundary of the classifier and the training points could be much smaller than the optimal value.

We verify that when a neural network is trained with the cross-entropy loss to classify two classes from the CIFAR-10 dataset, the output of the penultimate layer of the network indeed produces points that lie on an affine subspace.4.

We show that if there is no explicit and effective regularization, the weights of the last layer of a neural network could grow to infinity during training with a gradient method.

Even though this has been observed in recent works as well, we are the first to point out that this divergence drives the confidence of the neural network to 100% at almost every point in the input space if the network is trained for long.

In other words, the confidence depends heavily on the training duration, and its exact value might be of little significance as long as it is above 50%.5.

We introduce differential training, which is a training paradigm that uses a loss function defined on pairs of points from each class -instead of only one point from any class.

We show that the decision boundary of a linear classifier trained with differential training indeed produces the SVM solution with the maximum hard margin.

We start with a simple binary classification problem.

Given two points x ∈ R d and −y ∈ R d from two different classes, we can find a linear classifier by minimizing the cross-entropy loss function log(e −w x + 1) + log(e −w ỹ + 1) , DISPLAYFORM0 .

Unless the two points x and −y are equal, the function (1) does not attain its minimum at a finite value ofw.

Consequently, if the gradient descent algorithm is used to minimize (1), the iterate at time k,w[k], diverges as k increases.

The following theorem characterizes the growth rate ofw[k] and its direction in the limit by using a continuous-time approximation to the gradient descent algorithm.

Theorem 1.

Given two points x ∈ R d and −y ∈ R d , letx and −ỹ denote [x 1] and [−y 1], respectively.

Without loss of generality, assume x ≤ y .

If the two points are in different classes and we minimize the cross-entropy loss DISPLAYFORM1 where σ x = x 2 , σ xy =x ỹ and σ y = ỹ 2 .Note that first d coordinates of (2) represent the normal vector of the decision boundary obtained by minimizing the cross-entropy loss (1).

This vector is different from x + y, which is the direction of the maximum-margin solution given by the SVM.

In fact, the direction in (2) could be almost orthogonal to the SVM solution in certain cases, which implies that the margin between the points and the decision boundary could be much smaller than the optimal value.

Corollary 1 describes a subset of these cases.

Corollary 1.

Given two points x and −y in R d , let ψ denote the angle between the solution given by (2) and the solution given by the SVM, i.e., (x + y).

If x y = 1, then DISPLAYFORM2 where σ x = x 2 +1 and σ y = y 2 +1.

Consequently, as x / y approaches 0 while maintaining the condition x y = 1, the angle ψ converges to π/2.

Remark 1.

Corollary 1 shows that if x and −y have disparate norms, the minimization of the crossentropy loss with gradient descent algorithm could lead to a direction which is almost orthogonal to the maximum-margin solution.

It may seem like this problem could be avoided with preprocessing the data so as to normalize the data points.

However, this approach will not be effective for neural networks: if we consider an L-layer neural network, w φ L−1 (x), and regard the first L − 1 layers, φ L−1 (·), as a feature mapping, preprocessing a dataset {x i } i∈I will not produce a normalized set of features {φ L−1 (x i )} i∈I .

Note that we could not normalize {φ L−1 (x i )} i∈I directly either, since the mapping φ L−1 (·) evolves during training.

Remark 2.

Theorem 1 shows that the norm of w keeps growing unboundedly as the training continues.

The same behavior will be observed for larger datasets in the next sections as well.

Since the "confidence" of the classifier for its prediction at a point x is given by DISPLAYFORM3 this unbounded growth of w drives the confidence of the classifier to 100% at every point in the input space, except at the points on the decision boundary, if the algorithm is run for long.

Given the lack of effective regularization for neural networks, a similar unbounded growth is expected to be observed in neural network training as well, which is mentioned in BID1 .

As a result, the confidence of a neural network might be highly correlated with the training duration, and whether a neural network gives 99% or 51% confidence for a prediction might be of little importance as long as it is above 50%.

In other words, regarding this confidence value as a measure of similarity between an input and the training dataset from the most-likely class should be reconsidered.

In this section, we examine the binary classification of a linearly separable dataset by minimizing the cross-entropy loss function.

Recently, this problem has also been studied in BID16 a; BID7 .

We restate an edited version of the main theorem of , followed by the reason of the edition.

Theorem 2 (Adapted from Theorem 3 of ).

Given two sets of points {x i } i∈I and {−y j } j∈J that are linearly separable in R d , letx i and −ỹ j denote [x i 1] and [−y j 1] , respectively, for all i ∈ I, j ∈ J. Then the iterate of the gradient descent algorithm,w(t), on the cross-entropy loss function DISPLAYFORM0 with a sufficiently small step size will converge in direction: DISPLAYFORM1 where w is the solution to DISPLAYFORM2 The solution (4) given in Theorem 2 was referred in , and consequently in the other works, as the maximum-margin solution.

However, due to the misleading absence of the bias term in the notation, this is incorrect.

Given the linearly separable sets of points {x i } i∈I and {−y j } j∈J , the maximum-margin solution given by the SVM solves DISPLAYFORM3 On the other hand, the solution given by Theorem 2 corresponds to DISPLAYFORM4 Even though the sets of constraints for both problems are identical, their objective functions are different, and consequently, the solutions are different.

As a result, the decision boundary obtained by crossentropy minimization does not necessarily attain the maximum hard margin.

In fact, as the following theorem shows, its margin could be arbitrarily worse than the maximum margin.

Theorem 3.

Assume that the points {x i } i∈I and {−y j } j∈J are linearly separable and lie in an affine subspace; that is, there exist a set of orthonormal vectors {r k } k∈K and a set of scalars DISPLAYFORM5 Let w, · + B = 0 denote the decision boundary obtained by minimizing the cross-entropy loss, i.e. the pair (w, B) solves DISPLAYFORM6 Then the minimization of the cross-entropy loss (3) yields a margin smaller than or equal to DISPLAYFORM7 where γ denotes the optimal hard margin given by the SVM solution.

Remark 3.

Theorem 3 shows that if the training points lie in an affine subspace, the margin obtained by the cross-entropy minimization will be smaller than the optimal margin value.

As the dimension of this affine subspace decreases, the cardinality of the set K increases and the term k∈K ∆ 2 k could become much larger than 1/γ 2 .

Therefore, as the dimension of the subspace containing the training points gets smaller compared to the dimension of the input space, cross-entropy minimization with a gradient method becomes more likely to yield a poor margin.

Note that this argument also holds for classifiers of the form w φ(x) with the fixed feature mapping φ(·).The next theorem relaxes the condition of Theorem 3 and allows the training points to be near an affine subspace instead of being exactly on it.

Note that the ability to compare the margin obtained by cross-entropy minimization with the optimal value is lost.

Nevertheless, it highlights the fact that same set of points could be assigned a different margin by cross-entropy minimization if all of them are shifted away from the origin by the same amount in the same direction.

Theorem 4.

Assume that the points {x i } i∈I and {−y j } j∈J in R d are linearly separable and there exist a set of orthonormal vectors {r k } k∈K and a set of scalars {∆ k } k∈K such that DISPLAYFORM8 Let w, · + B = 0 denote the decision boundary obtained by minimizing the cross-entropy loss, i.e. the pair (w, B) solves DISPLAYFORM9 Then the minimization of the cross-entropy loss (3) yields a margin smaller than or equal to DISPLAYFORM10 Remark 4.

Both Theorem 3 and Theorem 4 consider linearly separable datasets.

If the dataset is not linearly separable, BID7 predicts that the normal vector of the decision boundary, w, will have two components, one of which converges to a finite vector and the other diverges.

The diverging component still has the potential to drive the decision boundary to a direction with a poor margin.

In fact, the margin is expected to be small especially if the points intruding into the opposite class lie in the same subspace as the optimal normal vector for the decision boundary.

In this work, we focus on the case of separable datasets as this case provides critical insight into the issues of state-of-the-art neural networks, given they can easily attain zero training error even on randomly generated datasets, which indicates the linear separability of the features obtained at their penultimate layers (Zhang et al., 2017) .

In previous sections, we saw that the cross-entropy minimization could lead to poor margins, and the main reason for this was the appearance of the bias term in the objective function of (P2).

In order to remove the effect of the bias term, consider the SVM problem (P1) and note that this problem could be equivalently written as minimize w w 2 2 subject to w, x i + y j ≥ 2 ∀i ∈ I, ∀j ∈ Jif we only care about the weight parameter w.

This gives the hint that if we use the set of differences {x i + y j : i ∈ I, j ∈ J} instead of the individual sets {x i } i∈I and {−y j } j∈J , the bias term could be excluded from the problem.

This was also noted in BID8 BID6 previously.

Indeed, this approach allows obtaining the SVM solution with a loss function similar to the cross-entropy loss, as the following theorem shows.

Theorem 5.

Given two sets of points {x i } i∈I and {−y j } j∈J that are linearly separable in R d , if we solve min DISPLAYFORM0 by using the gradient descent algorithm with a sufficiently small learning rate, the direction of w converges to the direction of maximum-margin solution, i.e. DISPLAYFORM1 where w SVM is the solution of (P3).Proof.

Apply Theorem 2 by replacing the sets {x i } i∈I and {−y j } j∈J with {x i + y j } i∈I,j∈J and the empty set, respectively.

Then the minimization of the loss function FORMULA16 Since w SVM is the solution of (P3), we obtain w = 1 2 w SVM , and the claim of the theorem holds.

Remark 5.

Theorem 5 is stated for the gradient descent algorithm, but the identical statement could be made for the stochastic gradient method as well by invoking the main theorem of BID20 .Minimization of the cost function (5) yields the weight parameterŵ of the decision boundary.

The bias parameter, b, could be chosen by plotting the histogram of the inner products { ŵ, x i } i∈I and { ŵ, −y j } j∈J and fixing a value forb such that DISPLAYFORM2 DISPLAYFORM3 The largest hard margin is achieved bŷ DISPLAYFORM4 However, by choosing a larger or smaller value forb, it is possible to make a tradeoff between the Type-I and Type-II errors.

The cost function (5) includes a loss defined on every pair of data points from the two classes.

This cost function can be considered as the cross-entropy loss on a new dataset which contains |I| × |J| points.

There are two aspects of this fact:1.

When standard loss functions are used for classification tasks, we need to oversample or undersample either of the classes if the training dataset contains different number of points from different classes.

This problem does not arise when we use the cost function (5).

2.

Number of pairs in the new dataset, |I| × |J|, will usually be much larger than the original dataset, which contains |I| + |J| points.

Therefore, the minimization of (5) might appear more expensive than the minimization of the standard cross-entropy loss computationally.

However, if the points in different classes are well separated and the stochastic gradient method is used to minimize (5), the algorithm achieves zero training error after using only a few pairs, which is formalized in Theorem 6.

Further computation is needed only to improve the margin of the classifier.

In addition, in our experiments to train a neural network to classify two classes from the CIFAR-10 dataset, only a few percent of |I| × |J| points were observed to be sufficient to reach a high accuracy on the training dataset.

Theorem 6.

Given two sets of points {x i } i∈I and {−y j } j∈J that are linearly separable in R d , assume the cost function (5) is minimized with the stochastic gradient method.

Define R x = max{ x i −

x i : i, i ∈ I}, R y = max{ y j − y j : j, j ∈ J} and let γ denote the hard margin that would be obtained with the SVM: DISPLAYFORM5 If 2γ ≥ 5 max(R x , R y ), then the stochastic gradient algorithm produces a weight parameter,ŵ, only in one iteration which satisfies the inequalities (7a)-(7b) along with the bias,b, given by (8).

In this section, we present numerical experiments supporting our claims.

Differential training.

In Figure 2 , we show the decision boundaries of two linear classifiers, where one of them is trained by minimizing the cross-entropy loss, and the other through differential training.

Unlike the example shown in FIG6 , here the data do not exactly lie in an affine subspace.

In particular, one of the classes is composed of 10 samples from a normal distribution with mean (2, 12) and variance 25, and the other class is composed of 10 samples from a normal distribution with mean (40, 50) and variance 25.

As can be seen from the figure, the cross-entropy minimization yields a margin that is smaller than differential training, even though when the training dataset is not low-dimensional, which is predicted by Theorem 4.

Cross-entropy min.

boundary Figure 2 : Classification boundaries obtained using differential training and cross-entropy minimization.

The margin recovered by cross-entropy minimization is worse than differential training even when the training dataset is not low-dimensional.

Low-dimensionality.

We empirically evaluated if the features obtained at the penultimate layer of a neural network indeed lie in a low-dimensional affine subspace.

For this purpose, we trained a convolutional neural network architecture to classify horses and planes from the CIFAR-10 dataset BID10 .

FIG3 shows the cumulative variance explained for the features that feed into the soft-max layer as a function of the number of principle components used.

We observe that the features, which are the outputs of the penultimate layer of the network, lie in a low-dimensional affine subspace, and this holds for a variety of training modalities for the network.

This observation is relevant to Remark 3.

The dimension of the subspace containing the training points is at most 20, which is much smaller than the dimension of the feature space, 84.

Consequently, cross-entropy minimization with a gradient method is expected to yield a poor margin on these features.

We compare our results with related works and discuss their implications for the following subjects.

Adversarial examples.

State-of-the-art neural networks have been observed to misclassify inputs that are slightly different from their training data, which indicates a small margin between their decision boundaries and the training dataset (Szegedy et al., 2013; BID3 MoosaviDezfooli et al., 2017; .

Our results reveal that the combination of gradient methods, cross-entropy loss function and the low-dimensionality of the training dataset (at least in some domain) has a responsibility for this problem.

Note that SVM with the radial basis function was shown to be robust against adversarial examples, and this was attributed to the high nonlinearity of the radial basis function in BID3 .

Given that the SVM uses neither the cross entropy loss function nor the gradient descent algorithm for training, we argue that the robustness of SVM is no surprise -independent of its nonlinearity.

Lastly, effectiveness of differential training for neural networks against adversarial examples is our ongoing work.

The activations feeding into the soft-max layer could be considered as the features for a linear classifier.

Plot shows the cumulative variance explained for these features as a function of the number of principle components used.

Almost all the variance in the features is captured by the first 20 principle components out of 84, which shows that the input to the soft-max layer resides predominantly in a low-dimensional subspace.

Low-dimensionality of the training dataset.

As stated in Remark 3, as the dimension of the affine subspace containing the training dataset gets very small compared to the dimension of the input space, the training algorithm will become more likely to yield a small margin for the classifier.

This observation confirms the results of BID13 , which showed that if the set of training data is projected onto a low-dimensional subspace before feeding into a neural network, the performance of the network against adversarial examples is improved -since projecting the inputs onto a low-dimensional domain corresponds to decreasing the dimension of the input space.

Even though this method is effective, it requires the knowledge of the domain in which the training points are low-dimensional.

Because this knowledge will not always be available, finding alternative training algorithms and loss functions that are suited for low-dimensional data is still an important direction for future research.

Robust optimization.

Using robust optimization techniques to train neural networks has been shown to be effective against adversarial examples BID12 BID0 .

Note that these techniques could be considered as inflating the training points by a presumed amount and training the classifier with these inflated points.

Consequently, as long as the cross-entropy loss is involved, the decision boundaries of the neural network will still be in the vicinity of the inflated points.

Therefore, even though the classifier is robust against the disturbances of the presumed magnitude, the margin of the classifier could still be much smaller than what it could potentially be.

Differential training.

We introduced differential training, which allows the feature mapping to remain trainable while ensuring a large margin between different classes of points.

Therefore, this method combines the benefits of neural networks with those of support vector machines.

Even though moving from 2N training points to N 2 seems prohibitive, it points out that a true classification should in fact be able to differentiate between the pairs that are hardest to differentiate, and this search will necessarily require an N 2 term.

Some heuristic methods are likely to be effective, such as considering only a smaller subset of points closer to the boundary and updating this set of points as needed during training.

If a neural network is trained with this procedure, the network will be forced to find features that are able to tell apart between the hardest pairs.

Nonseparable data.

What happens when the training data is not linearly separable is an open direction for future work.

However, as stated in Remark 4, this case is not expected to arise for the state-of-the-art networks, since they have been shown to achieve zero training error even on randomly generated datasets (Zhang et al., 2017) , which implies that the features represented by the output of their penultimate layer eventually become linearly separable.

A PROOF OF THEOREM 1Theorem 1 could be proved by using Theorem 2, but we provide an independent proof here.

Gradient descent algorithm with learning rate δ on the cross-entropy loss (1) yields DISPLAYFORM0 1 + e −w x + δỹ e −w ỹ 1 + e −w ỹ .Ifw(0) = 0, thenw(t) = p(t)x + q(t)ỹ for all t ≥ 0, wherė DISPLAYFORM1 Then we can writeα Lemma 2.

If b < 0, then there exists t 0 ∈ (0, ∞) such that DISPLAYFORM2 Proof.

Note that DISPLAYFORM3 which implies that DISPLAYFORM4 as long as DISPLAYFORM5 By using Lemma 2, DISPLAYFORM6 Proof.

Solving the set of equations DISPLAYFORM7 , DISPLAYFORM8 Proof.

Note thatż ≥ a/2 andv ≥ c/2; therefore, DISPLAYFORM9 if either side exists.

Remember thaṫ DISPLAYFORM10 We can compute f (w) = 2acw + bcw 2 + ab b 2 w 2 + 2abw + a 2 .

The function f is strictly increasing and convex for w > 0.

We have DISPLAYFORM11 Therefore, when b ≥ a, the only fixed point of f over [0, ∞) is the origin, and when a > b, 0 and (a − b)/(c − b) are the only fixed points of f over [0, ∞).

Figure 4 shows the curves over whichu = 0 andẇ = 0.

Since lim t→∞ u = lim t→∞ w, the only points (u, w) can converge to are the fixed points of f .

Remember thaṫ DISPLAYFORM12 so when a > b, the origin (0, 0) is unstable in the sense of Lyapunov, and (u, w) cannot converge to it.

Otherwise, (0, 0) is the only fixed point, and it is stable.

As a result, DISPLAYFORM13 Figure 4: Stationary points of function f .

DISPLAYFORM14 Proof.

From Lemma 6, DISPLAYFORM15 Consequently, DISPLAYFORM16 which gives the same solution as Lemma 5: DISPLAYFORM17 Proof.

We can obtain a lower bound for square of the denominator as DISPLAYFORM18 DISPLAYFORM19 As a result, Then, we can write w as DISPLAYFORM20 Remember, by definition, w SVM = arg min w 2 s.t.

w, x i + y j ≥ 2 ∀i ∈ I, ∀j ∈ J.Since the vector u also satisfies u, x i + y j = w, x i + y j ≥ 2 for all i ∈ I, j ∈ J, we have u ≥ w SVM = 1 γ .

As a result, the margin obtained by minimizing the cross-entropy loss is DISPLAYFORM21

If B < 0, we could consider the hyperplane w, · − B = 0 for the points {−x i }

i∈I and {y j } j∈J , which would have the identical margin due to symmetry.

Therefore, without loss of generality, assume B ≥ 0.

As in the proof of Theorem 3, KKT conditions for the optimality of w and B requires w = i∈I µ i x i + j∈J ν j y j , B = i∈I µ i − j∈J ν j where µ i ≥ 0 and ν j ≥ 0 for all i ∈ I, j ∈ J. Note that for each k ∈ K, w, r k = i∈I µ i x i , r k − j∈J ν j −y j , r k DISPLAYFORM0 Since {r k } k∈K is an orthonormal set of vectors, DISPLAYFORM1 The result follows from the fact that w −1 is an upper bound on the margin.

E PROOF OF THEOREM 6In order to achieve zero training error in one iteration of the stochastic gradient algorithm, it is sufficient to have min i ∈I x i , x i + y j > max j ∈J −y j , x i + y j ∀i ∈ I, ∀j ∈ J, or equivalently, x i + y j , x i + y j > 0 ∀i, i ∈ I, ∀j, j ∈ J.By definition of the margin, there exists a vector w SVM ∈ R d with unit norm which satisfies 2γ = min i∈I,j∈J x i + y j , w SVM .Note that w SVM is orthogonal to the decision boundary given by the SVM.

Then we can write every x i + y j as x i + y j = 2γw SVM + δ If we choose γ > 5 2 max(R x , R y ), we have 4γ 2 − 2γ(2R x + 2R y ) − (R x + R y ) 2 > 0, which guarantees (10) and completes the proof.

@highlight

We show that minimizing the cross-entropy loss by using a gradient method could lead to a very poor margin if the features of the dataset lie on a low-dimensional subspace.