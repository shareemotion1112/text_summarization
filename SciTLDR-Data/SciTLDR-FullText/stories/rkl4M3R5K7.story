We study the problem of designing provably optimal adversarial noise algorithms that induce misclassification in settings where a learner aggregates decisions from multiple classifiers.

Given the demonstrated vulnerability of state-of-the-art models to adversarial examples, recent efforts within the field of robust machine learning have focused on the use of ensemble classifiers as a way of boosting the robustness of individual models.

In this paper, we design provably optimal attacks against a set of classifiers.

We demonstrate how this problem can be framed as finding strategies at equilibrium in a two player, zero sum game between a learner and an adversary and consequently illustrate the need for randomization in adversarial attacks.

The main technical challenge we consider is the design of best response oracles that can be implemented in a Multiplicative Weight Updates framework to find equilibrium strategies in the zero-sum game.

We develop a series of scalable noise generation algorithms for deep neural networks, and show that it outperforms state-of-the-art attacks on various image classification tasks.

Although there are generally no guarantees for deep learning, we show this is a well-principled approach in that it is provably optimal for linear classifiers.

The main insight is a geometric characterization of the decision space that reduces the problem of designing best response oracles to minimizing a quadratic function over a set of convex polytopes.

In this paper, we study adversarial attacks that induce misclassification when a learner has access to multiple classifiers.

One of the most pressing concerns within the field of AI has been the welldemonstrated sensitivity of machine learning algorithms to noise and their general instability.

Seminal work by has shown that adversarial attacks that produce small perturbations can cause data points to be misclassified by state-of-the-art models, including neural networks.

In order to evaluate classifiers' robustness and improve their training, adversarial attacks have become a central focus in machine learning and security BID21 BID17 BID23 .Adversarial attacks induce misclassification by perturbing data points past the decision boundary of a particular class.

In the case of binary linear classifiers, for example, the optimal perturbation is to push points in the direction perpendicular to the separating hyperplane.

For non-linear models there is no general characterization of an optimal perturbation, though attacks designed for linear classifiers tend to generalize well to deep neural networks BID21 .Since a learner may aggregate decisions using multiple classifiers, a recent line of work has focused on designing attacks on an ensemble of different classifiers BID31 BID0 BID13 .

In particular, this line of work shows that an entire set of state-of-the-art classifiers can be fooled by using an adversarial attack on an ensemble classifier that averages the decisions of the classifiers in that set.

Given that attacking an entire set of classifiers is possible, the natural question is then:What is the most effective approach to design attacks on a set of multiple classifiers?The main challenge when considering attacks on multiple classifiers is that fooling a single model, or even the ensemble classifier (i.e. the model that classifies a data point by averaging individual predictions), provides no guarantees that the learner will fail to classify correctly.

Models may have different decision boundaries, and perturbations that affect one may be ineffective on another.

Furthermore, a learner can randomize over classifiers and avoid deterministic attacks (see Figure 1 ).

c 2 c 1 Figure 1 : Illustration of why randomization is necessary to compute optimal adversarial attacks.

In this example using binary linear classifiers, there is a single point that is initially classified correctly by two classifiers c1, c2, and a fixed noise budget α in the ℓ2 norm.

A naive adversary who chooses a noise perturbation deterministically will always fail to trick the learner since she can always select the remaining classifier.

An optimal adversarial attack in this scenario consists of randomizing with equal probability amongst both noise vectors.

In this paper, we present a principled approach for attacking a set of classifiers which proves to be highly effective.

We show that constructing optimal adversarial attacks against multiple classifiers is equivalent to finding strategies at equilibrium in a zero sum game between a learner and an adversary.

It is well known that strategies at equilibrium in a zero sum game can be obtained by applying the celebrated Multiplicative Weights Update framework, given an oracle that computes a best response to a randomized strategy.

The main technical challenge we address pertains to the characterization and implementation of such oracles.

Our main contributions can be summarized as follows:• We describe the Noise Synthesis FrameWork (henceforth NSFW) for generating adversarial attacks.

This framework reduces the problem of designing optimal adversarial attacks for a general set of classifiers to constructing a best response oracle in a two player, zero sum game between a learner and an adversary; • We show that NSFW is an effective approach for designing adversarial noise that fools neural networks.

In particular, applying projected gradient descent on an appropriately chosen loss function as a proxy for a best response oracle achieves performance that significantly improves upon current state-of-the-art attacks (see results in Figure 2 ); • We show that applying projected gradient descent on an appropriately chosen loss function is a well-principled approach.

We do so by proving that for linear classifiers such an approach yields an optimal adversarial attack if the equivalent game has a pure Nash equilibrium.

This result is shown via a geometric characterization of the decision boundary space which reduces the problem of designing optimal attacks to a convex program; • If the game does not have a pure Nash equilibrium, there is an algorithm for finding an optimal adversarial attack for linear classifiers whose runtime is exponential in the number of classifiers.

We show that finding an optimal strategy in this case is NP-hard.

Paper organization.

Following a discussion on related work, in Section 2 we formulate the problem of designing optimal adversarial noise and show how it can be modeled as finding strategies at equilibrium in a two player, zero sum game.

Afterwards, we discuss our approach for finding such strategies using MWU and proxies for best response oracles.

In Section 2 .1, we justify our approach by proving guarantees for linear classifiers.

Lastly, in Section 3, we present our experiments.

Additional related work.

The field of adversarial attacks on machine learning classifiers has recently received widespread attention from a variety of perspectives BID1 BID9 BID25 BID3 .

In particular, a significant amount of effort has been devoted to computing adversarial examples that induce misclassification across multiple models BID22 BID21 .

There has been compelling evidence which empirically demonstrates the effectiveness of ensembles as way of both generating and defending against adversarial attacks.

For example, BID31 establish the strengths of ensemble training as a defense against adversarial attacks.

Conversely, provide the first set of experiments showing that attacking an ensemble classifier is an effective way of generating adversarial examples that transfer to the underlying models.

Relative to their investigation, our work differs in certain key aspects.

Rather than analyzing adversarial noise from a security perspective and developing methods for black-box attacks, we approach the problem from a theoretical point of view and introduce a formal characterization of the optimal attack against a set of classifiers.

Furthermore, by analyzing noise in the linear setting, we design algorithms for this task that have strong guarantees of performance.

Through our experiments, we demonstrate how these algorithms motivate a natural extension for noise in deep learning that achieves state-of-the-art results.

Given a set of point-label pairs {( DISPLAYFORM0 where DISPLAYFORM1 , a deterministic adversarial attack is a totally ordered set of noise vectors, V = (v 1 , . . .

, v m ) ∈ R d×m .

We say that q is an adversarial attack if q is a distribution over sets of noise vectors.

An adversarial attack q is α-bounded if for all sets V that have non-zero probability under q, each individual noise vector v i ∈ V has bounded norm, e.g ||v i || p ≤ α.

We focus on the case where each vector v i is bounded to have ℓ 2 norm less than a fixed value α, however, our model can be easily extended to a variety of norms.

For a given classifier c : DISPLAYFORM0 , a realization of the adversarial attack, V = (v 1 , . . .

, v m ), induces misclassification on (x j , y j ) if c(x j + v j ) ∕ = y j .

Given a finite set of classifiers C and a data set S = {(x i , y i )} m i=1 of point-label pairs as above, an optimal adversarial attack is a distribution q over sets of noise vectors that maximizes the minimum 0-1 loss of the classifiers in C: DISPLAYFORM1 Optimal adversarial attacks are equilibrium strategies in a zero sum game.

An equivalent interpretation of the optimization problem described in Equation FORMULA3 is that of a best response in a two player, zero sum game played between a learner and an adversary.

When the learner plays classifier c ∈ C and the adversary plays an attack V , the payoff to the adversary is M (c, DISPLAYFORM2 , which is the average 0-1 loss of the learner.2 The learner and the adversary can choose to play randomized strategies p, q over classifiers and noise vectors yielding expected payout E(c,V )∼(p,q) M (c, V ).

The (mixed) equilibrium strategy of the game is the pair of distributions p, q that maximize the minimum loss DISPLAYFORM3 Computing optimal adversarial attacks via MWU.

As discussed above, the optimization problem of designing optimal adversarial attacks reduces to that of finding strategies at equilibrium in a zero sum game.

It is well known that the celebrated Multiplicative Weight Updates algorithm can be used to efficiently compute equilibrium strategies of zero sum games when equipped with a best response oracle that finds an optimal set of perturbations for any strategy chosen by the learner: DISPLAYFORM4 Our framework for generating adversarial noise applies the Multiplicative Weight Updates algorithm as specified in Algorithm 1.

The algorithm returns distributions p , q that are within δ of the equilibrium value of the game λ = min DISPLAYFORM5 ln n δ 2 ) calls to a best response oracle.3 In this work, we focus on developing attacks on neural networks and linear models.

Yet, our framework is general enough to generate optimal attacks for any domain in which one can approximate a best response.

We analyze the convergence of NSFW in Appendix G.Approximating a best response.

Given the framework described above, the main challenge is in computing a best response strategy.

To do so, at every iteration, as a proxy for a best response, we apply projected gradient descent (PGD) to an appropriately chosen surrogate loss function.

In particular, given DISPLAYFORM6 we aim to solve: DISPLAYFORM7 ℓ is a loss function that depends on the type of attack (targeted vs. untargeted) and the type of classifiers in C (linear vs. deep).

We introduce a series of alternatives for ℓ in the following section.

As we will now show, maximizing the loss of the learner by applying PGD to a weighted sum of loss functions is a well-principled approach to computing best responses as it is guaranteed to converge to the optimal solution in the case where C is composed of linear classifiers.

While there are generally no guarantees for solving non-convex optimization problems of this sort for deep neural networks, in Section 3 , we demonstrate the effectiveness of our approach by showing that it experimentally improves upon current state-of-the-art attacks.

Input: DISPLAYFORM0

The main theoretical insight that leads to provable guarantees for generating adversarial noise is a geometric characterization of the underlying structure of adversarial attacks.

Regardless of the type of model, selecting a distribution over classifiers partitions the input space into disjoint regions, each of which is associated with a single loss value for the learner.

Given a distribution over classifiers played by the learner, computing a best response strategy for the adversary then reduces to a search problem.

In this problem, the search is for points in each region that lie within the noise budget and can be misclassified.

The best response is to select the region which induces the maximal loss.

In the case of linear classifiers, the key observation is that the regions are convex.

As a result, designing optimal adversarial attacks reduces to solving a series of quadratic programs.

Lemma 1.

Selecting a distribution p over a set C of n linear classifiers, partitions the input space R d into k n disjoint, convex sets T j such that: DISPLAYFORM0 2.

There exists a finite set of numbers a 1 , . . .

a k n , not necessarily all unique, such that DISPLAYFORM1 Proof Sketch (see full proof in Appendix C).

Each set T j is defined according to the predictions of the classifiers c i ∈ C on points x ∈ T j .

In particular, each region T j is associated with a unique label vector DISPLAYFORM2 Since the prediction of each classifier is the same for all points in a particular region, the loss of the learner i∈[n] p[i]ℓ 0-1 (c i , x, y) is constant over the entire region.

Convexity then follows by showing that each T j is an intersection of hyperplanes.

This characterization of the underlying geometry now allows us to design best response oracles for linear classifiers via convex optimization.

For our analysis, we focus on the case where C consists of "one-vs-all" classifiers.

In the appendix, we show how our results can be generalized to other methods for multilabel classification by reducing these other approaches to the "one-vs-all" case.

Given k classes, a "one-vs-all" classifier c i consists of k linear functions c i, DISPLAYFORM3 On input x, predictions are made according to the rule c i (x) = arg max j c i,j (x).

Lemma 2.

For linear classifiers, implementing a best response oracle reduces to the problem of minimizing a quadratic function over a set of k n convex polytopes.

Proof Sketch (see full proof in Appendix C).

The main idea behind this lemma is that given a distribution over classifiers, the loss of the learner can be maximized individually for each point (x, y) ∈ S. Furthermore, by Lemma 1, the loss can assume only finitely many values, each of which is associated with a particular convex region T j of the input space.

Therefore, to compute a best response, we can iterate over all regions and choose the one associated with the highest loss.

To find points in each region T j , we can simply minimize the ℓ 2 norm of a perturbation v such that x + v ∈ T j , which can be framed as minimizing a quadratic function over a convex set.

These results give an important characterization, but it also shows that the number of polytopes is exponential in the number of classifiers.

To overcome this difficulty, we demonstrate how when there exists a pure strategy Nash equilibrium (PSNE), that is a single set of noise vectors V where every vector is bounded by α and min ci∈C M (c i , V ) = 1, PGD applied to the reverse hinge loss, ℓ r , is guaranteed to converge to a point that achieves this maximum for binary classifiers.

More generally, given a label vector s j ∈ [k] n , PGD applied to the targeted reverse hinge loss, ℓ t , converges to a point within the noise budget that lies within the specified set T j .

We define ℓ r and ℓ t as follows: DISPLAYFORM4 The proof follows standard arguments for convergence of convex and β-smooth functions.

Theorem 1.

Given any precision > 0 and noise budget α > 0:• For a finite set of linear binary classifiers C and a point (x, y), running PGD for T = 4α/ iterations on the objective DISPLAYFORM5 converges to a point that is within of the pure strategy Nash equilibrium f (x + v * ), if such an equilibrium exists;• For a finite set of linear multilabel classifiers C, given a label vector s j ∈ [k] n and a distribution p over C, running PGD for T = 4α/ iterations on the objective DISPLAYFORM6 Proof Sketch.

From the definition of the reverse hinge loss, we see that ℓ r (c i , x ′ , y) = 0 if and only if ℓ 0-1 (c i , x ′ , y) = 1.

Similarly, the targeted loss ℓ t (c i , x ′ , j) is 0 if and only if c i predicts x ′ to have label j. For linear classifiers, both of these functions are convex and β-smooth.

Hence PGD converges to a global minimum, which is zero if there exists a pure equilibrium in the game.

The requirement that there exist a feasible point x ′ within T j is not only sufficient, it is also necessary in order to avoid a brute force search.

Designing an efficient algorithm to find the region associated with the highest loss is unlikely as the decision version of the problem is NP-hard even for binary linear classifiers.

We state the theorem below and defer the proof to the appendix.

Theorem 2.

Given a set C of n binary, linear classifiers, a number B, a point (x, y), noise budget α, and a distribution p, finding v with ||v|| 2 ≤ α s.t.

the loss of the learner is exactly B is NP-complete.

As we show in the following section, this hardness result does not limit our ability to compute optimal adversarial examples.

Most of the problems that have been examined in the context of adversarial noise suppose that the learner has access only to a small number of classifiers (e.g less than 5) BID8 BID0 BID31 BID13 .

In such cases we can solve the convex program over all regions and find an optimal adversarial attack, even when a pure Nash equilibrium does not exist.

We evaluate the performance of NSFW at fooling a set of classifiers by comparing against noise generated by using state-of-the-art attacks against an ensemble classifier.

Recent work by and BID31 , demonstrates how attacking an ensemble of a set of classifiers generates noise that improves upon all previous attempts at fooling multiple classifiers.

We test our methods on deep neural networks on MNIST and ImageNet, as well as on linear classifiers where we know that NSFW is guaranteed to converge to the optimal adversarial attack.

We use the insights derived from our theoretical analysis of linear models to approximate a best response oracle for this new setting.

Specifically, at each iteration of NSFW we compute a best response as in Equation FORMULA9 by running PGD on a weighted sum of untargeted reverse hinge losses, ℓ ut , introduced in this domain by BID4 .

Given a network c i , we denote c i,j (x) to be the probability assigned by the model to input x belonging to class j (the jth output of the softmax layer of the model).

DISPLAYFORM0 For MNIST, the set of classifiers C consists of 5 convolutional neural networks, each with a different architecture, that we train on the full training set of 55k images (see Appendix for details).

All classifiers (models) were over 97% accurate on the MNIST test set.

For ImageNet, C consists of the InceptionV3, DenseNet121, ResNet50, VGG16, and Xception models with pre-trained weights Figure 2 : Visual comparison of misclassification using state-of-the-art adversarial attacks.

We compare the level of noise necessary to induce similar levels of misclassification by attacking an ensemble classifier using the (from left to right) Fast Gradient Method (FGM), the Madry attack, and the Momentum Iterative Method (MIM) versus applying NSFW (rightmost column) on the same set of classifiers.

To induce a maximum of 17% accuracy across all models, we only need to set α to be 300 for NSFW.

For the MIM attack on the ensemble we need to set α = 2000.

For FGM and the Madry attack, the noise budget must be further increased to 8000.downloaded from the Keras repository BID6 BID12 BID27 BID7 BID30 BID14 .

To evaluate the merits of our approach, we compare our results against attacks on the ensemble composed of C as suggested by .

More specifically, we create an ensemble by averaging the outputs of the softmax layers of the different networks using equal weights.

We generate baseline attacks by attacking the ensemble using (1) the Fast Gradient Method by BID11 , (2) the Projected Gradient Method by , and (3) the Momentum Iterative Method by BID8 which we download from the Cleverhans library BID24 .

We select the noise budget α by comparing against the average ℓ 2 distortion reported by similar papers in the field.

For MNIST, we base ourselves off the values reported by BID4 and choose a noise budget of 3.0.

For ImageNet, we compare against .

In their paper, they run similar untargeted experiments on ImageNet with 100 images and report a noise budget of 22 when measured as the root mean squared deviation.

Converted to the ℓ 2 norm, this corresponds to α ≥ 8500.6 We found this noise budget to be excessive, yielding images comparable to those in the leftmost column in Figure 2 .

Therefore, we chose α = 300 (roughly 3.5% of the total distortion used in ) which ensures that the perturbed images are visually indistinguishable from the originals to the human eye (see rightmost column in Figure 2 Table 1 :

Accuracies of ImageNet models under different noise algorithms using a noise budget of 300.0 in the ℓ2 norm.

Entry (i, j) indicates the accuracy of each model j when evaluated on noise from attack i.

The last two columns report the mean and max accuracy of the classifiers on a particular attack.

We see that NSFW significantly outperforms noise generated by an ensemble classifier for all choices of attack algorithms.4 Specific details regarding model architectures as well as the code for all our experiments can be found in our repository which will be made public after the review period in order to comply with anonymity guidelines.

The test set accuracies of all ImageNet classifiers are displayed on the Keras website.5 Momentum Iterative Method won the 2017 NIPS adversarial attacks competition .

6 In their paper, define the root mean squared deviation of two points x, x as (xi − x i ) 2 /N where N is the dimension of x. For ImageNet, our images are of dimension 224×224×3, while for MNIST they are of size 28 × 28 × 1.

For further perspective, if we convert our noise budgets from the ℓ2 norm to RMSD, our budgets would correspond to .77 and .11 for ImageNet and MNIST respectively.

For our experiments, we ran NSFW for 50 MWU iterations on MNIST models and for 10 iterations on ImageNet classifiers.

We use far fewer iterations than the theoretical bound since we found that in practice NSFW converges to the equilibrium solution in only a small number of iterations (see Figure 5 in Appendix A).

At each iteration of the MWU we approximate a best response as described in Equation 3 by running PGD using the Adam optimizer (Kingma & Ba, 2014) on a sum of untargeted reverse hinge losses.

Specifically, we run the optimizer for 5k iterations with a learning rate of .01.

At each iteration, we clip images to lie in the range Finally, for evaluation, for both MNIST and ImageNet we selected 100 images uniformly at random from the set of images in the test sets that were correctly classified by all models.

In Table 1 , we report the empirical accuracy of all classifiers in the set C when evaluated on NSFW as well as on the three baseline attacks.

To compare their performance, we highlight the average and maximum accuracies of models in C when attacked using a particular noise solution.

From Table 1 , we see that on ImageNet our algorithm results in solutions that robustly optimize over the entire set of models using only a small amount of noise.

The maximum accuracy of any classifier is 17% under NSFW, while the best ensemble attack yields a max accuracy of only 68%.

If we wish to generate a similar level of performance from the ensemble baselines, we would need to increase the noise budget to 8000 for FGM and the Madry attack and to 2000 for the Momentum Iterative Method.

We present a visual comparison of the different attacks under these noise budgets required to achieve accuracy of 17% in Figure 2 .

On MNIST, we find similar results.

NSFW yields a max accuracy of 22.6% compared to the next best result of 48% generated by the Madry attack on the ensemble.

We summarize the results for MNIST in Table 2 presented in Appendix A.

As seen in the previous section, noise generated by directly attacking an ensemble of classifiers significantly underperforms NSFW at robustly fooling the underlying models.

In this section, we aim to understand this phenomenon by analyzing how the decision boundary of the ensemble model compares to that of the different networks.

In particular, we visualize the class boundaries of convolutional neural networks using the algorithm proposed by BID28 for generating saliency maps.

8 The class saliency map indicates which features (pixels) are most relevant in classifying an image to have a particular label.9 Therefore, they serve as one way of understanding the decision boundary of a particular model by highlighting which dimensions carry the highest weight.

In FIG0 , we see that the class saliency maps for individual models exhibit significant diversity.

The ensemble of all 5 classifiers appears to contain information from all models, however, certain regions that are of central importance for individual models are relatively less prominent in the ensemble saliency map.

Compared to our approach which calculates individual gradients for classifiers in C, creating an ensemble classifier obfuscates key information regarding the decision boundary of individual models.

We make this discussion rigorous by analyzing the linear case in Appendix B. NSFW on linear multiclass models using different noise functions and varying the noise budget α.

NSFWOracle corresponds to running Algorithm 1 using the best response oracle described in Lemma 2.

Similarly, NSFW-Untargeted shows the results of running NSFW and applying PGD to a weighted sum of untargeted losses as in Equation FORMULA9 .

The label iteration method is described below.

Lastly, the ensemble attack corresponds to the optimal noise on an equal weights ensemble of models in C. On the right, we illustrate the convergence of NSFW on linear binary classifiers with maximally different decision boundaries to compare against the convergence rate observed for neural nets in Figure 5 and better understand when weight adaptivity is necessary.

In addition to evaluating our approach on neural networks, we performed experiments with linear classifiers.

Since we have a precise characterization of the optimal attack on a set of linear classifiers, we can rigorously analyze the performance of different methods in comparison to the optimum.

We train two sets of 5 linear SVM classifiers on MNIST, one for binary classification (digits 4 and 9) and another for multiclass (first 4 classes, MNIST 0-3).

To ensure a diversity of models, we randomly zero out up to 75% of the dimensions of the training set for each classifier.

Hence, each model operates on a random subset of features.

All models achieve test accuracies of above 90%.

For our experiments, we select 1k points from each dataset that are correctly classified by all models.

In order to better compare across different best response proxies, we further extend NSFW by incorporating the label iteration method as another heuristic to generate untargeted noise.

Given a point (x, y), the iterative label method attempts to calculate a best response by running PGD on the targeted reverse hinge loss for every label j ∈ [k] \ {y} and choosing the attack associated with the minimal loss.

Compared to the untargeted reverse hinge loss, it has the benefit of being convex.

As for deep learning classifiers, we compare our results to the noise generated by the optimal attack on an ensemble of models in C. Since the class of linear classifiers is convex, creating an equal weights ensemble by averaging the weight vectors results in just another linear classifier.

We can compute the optimal attack by running the best response oracle described in Section 2 .1 for the special case where C consists of a single model and then scaling the noise to have norm equal to α.

As seen in the leftmost plot in FIG3 , even for linear models there is a significant difference between the optimal attack and other approaches.

Specifically, we observe an empirical gap between NSFW equipped with the best response oracle as described in Lemma 2 vs. NSFW with proxy best response oracles, e.g. the oracle that runs PGD on appropriately chosen loss functions.10 This difference in performance is consistent across a variety of noise budgets.

Our main takeaway is that in theory and in practice, there is a significant benefit in applying appropriately designed best response oracles.

Lastly, on the right in FIG3 , we illustrate how the adaptivity of MWU is in general necessary to compute optimal attacks.

While for most cases, NSFW converges to the equilibrium solution almost immediately, if the set of classifiers is sufficiently diverse, running NSFW for a larger number of rounds drastically boosts the quality of the attack.

(See Appendix A for details.)

Designing adversarial attacks when a learner has access to multiple classifiers is a non-trivial problem.

In this paper we introduced NSFW which is a principled approach that is provably optimal on linear classifiers and empirically effective on neural networks.

The main technical crux is in designing best response oracles which we achieve through a geometrical characterization of the optimization landscape.

We believe NSFW can generalize to domains beyond those in this paper.

A ADDITIONAL EXPERIMENTS AND DETAILS ON EXPERIMENTAL SETUP Figure 5 : Fast convergence of NSFW on MNIST and ImageNet deep learning models.

NSFW-Untargeted corresponds to running NSFW and applying PGD on a sum of untargeted reverese hinge losses as described in Section 3 .1.

The dotted lines correspond to running the indicated attack on the ensemble of models in C. For both datasets, we find that NSFW converges almost immediately to the equilibrium noise solution.

These misclassification results can also be examined in Tables 1 and 2 .We now discuss further details regarding the setup of the experiments presented in Section 3 .

In the case of deep learning, we set hyperparameters for of all the baseline attacks (Fast Gradient Method, Madry attack, and the Momentum Iterative Method) by analyzing the values reported in the original papers.

When running the Projected Gradient Method by Madry et al., given a noise budget α, we run the algorithm for 40 iterations with a step size of α/40 × 1.25 so as to mimic the setup of the authors.

In the case of the Momentum Iterative Method, we run the attack for 5 iterations with a decay factor µ = 1.0 and a step size of α/5 as specified in BID8 .

FGM has no hyperparameters other than the noise budget.

For all methods, we clip solutions to lie within the desired pixel range and noise budget.

When comparing different algorithms to compute best responses for linear multiclass classifiers as described in Section 3 .3, we run the NSFW algorithm with α = .2k for k ∈ [5].

In the case of binary classifiers FIG4 ), we find that the margins are smaller, and hence run NSFW with α = .05 + .1k for k ∈ [5].

For each value of α and choice of noise function, we run NSFW for 50 iterations.

The one exception is that, for the multiclass experiments with α equal to .2 or .4, we ran the best response oracle for only 20 iterations due to computational constraints.

When optimizing the loss of the learner through gradient descent (e.g when using PGD on appropriately chosen loses), we set the number of iterations to 3k and the learning rate to .01.We set up the weight adaptivity experiment described at the end of Section 3 .3 (rightmost plot of FIG3 ) as follows.

We train 5 linear binary SVM classifiers on our binary version of the MNIST dataset.

For each classifier, we zero out 80% of the input dimensions so that each model has nonzero weights for a strictly different subset of features, thereby ensuring maximum diversity in the decision Table 2 : Classification accuracies for deep learning MNIST models under different noise algorithms.

As in the ImageNet case, we find that the NSFW algorithm improves upon the performance of state-of-the-art attacks and robustly optimizes over the entire set of classifiers.

Moreover, we find that, for all attacks, there is a significant difference between the average and maximum accuracy of classifiers in C, further highlighting the need to design noise algorithms that are guaranteed to inhibit the performance of the best possible model.

FIG3 NSFW equipped with the best response outperforms other approaches at generating noise for linear models.

Furthermore, we see there is a performance gap between gradient based approaches and the theoretically optimal one that leverages convex programming.boundaries across models.

In order to generate noise, we select 500 points uniformly at random from the test set that were correctly classified by all modes.

We then run NSFW equipped with the best response oracle described in Lemma 2 for 50 iterations with a noise budget of .4.

In this section, we provide a brief theoretical justification as to why methods designed to attack an ensemble constructed by averaging individual models in a set of classifiers (e.g. Attacks on ensemble classifiers, as seen in , typically consist of applying gradient based optimization to an ensemble model E(C, p) made up of classifiers C and ensemble weights p.

For concreteness, consider the simplest case where C is composed of linear binary classifiers.

To find adversarial examples, we run gradient descent on a loss function such as the reverse hinge loss that is 0 if and only if the perturbed example x ′ = x + v with true label y is misclassified by c i .Assuming x ′ is not yet misclassified by the ensemble, running SGD on the ensemble classifier with the reverse hinge loss function results in a gradient update step of ∇ℓ r (E(C, p), x ′ , y) = i p[i]w i .

This is undesirable for two main reasons:• First, the ensemble obscures valuable information about the underlying objective.

If x ′ is misclassified by a particular model c i but not the ensemble, c i still contributes p[i]w i to the resulting gradient and biases exploration away from promising regions of the search space;• Second, fooling the ensemble does not guarantee that the noise will transfer across the underlying models.

Assuming the true label y is -1 and that x ′ is correctly classified by all models, ℓ r (E(C, p), x ′ , y) = 0 if and only if there exists a subset of classifiers DISPLAYFORM0 Hence, the strength of an ensemble classifier is only as good as its weakest weighted majority.

Lemma 1.

Selecting a distribution p over a set C of n linear classifiers, partitions the input space R d into k n disjoint, convex sets T j such that:1.

For each T j , there exists a unique label vector s j ∈ [k] n such that for all x ∈ T j and c i ∈ C, c i (x) = s j,i , where s j,i is a particular label in [k].2.

There exists a finite set of numbers a 1 , . . .

a k n , not necessarily all unique, such that n i=1 p[i]ℓ 0-1 (c i , x, y) = a j for a fixed y and all x ∈ T j 3.

R d \ j T j is a set of measure zero.

Proof.

Given a label vector s j , we define each T j as the set of points x where c i (x) = s j,i for all i ∈ [n].

This establishes a bijection between the elements of [k] n and the sets T j .

All the T j are pairwise disjoint since their corresponding label vectors in [k] n must differ in at least one index and by construction each classifier can only predict a single label for x ∈ T j .To see that these sets are convex, consider points x 1 , x 2 ∈ T j and an arbitrary classifier c i ∈ C s.t.

c i (x) = z for all x ∈ T j .

If we let x ′ = γx 1 + (1 − γ)x 2 where γ ∈ [0, 1] then the following holds for all j ∈ [k] where j ∕ = z: DISPLAYFORM0 Furthermore, for each T j , there exists a number a j ∈ R ≥0 such that the expected loss of the learner DISPLAYFORM1 Since the distribution p is fixed, the loss of the learner is uniquely determined by the correctness of the predictions of all the individual classifiers c i .

Since these are the same for all points in T j , the loss of the learner remains constant.

Lastly, the set R d \ i T i is equal to the set of points x where there are ties for the maximum valued classifier.

This set is a subset of the set of points K that lie at the intersection of two hyperplanes: DISPLAYFORM2 Finally, we argue that K has measure zero.

For all ε > 0, x ∈ K, there exists an x ′ such that ||x − x ′ || 2 < ε and x ′ / ∈ K since the intersection of two distinct hyperplanes is of dimension two less than the overall space.

Therefore, R d \

i T i must also have measure zero.

Lemma 2.

For linear classifiers, implementing a best response oracle reduces to the problem of minimizing a quadratic function over a set of k n convex polytopes.

Proof.

We outline the proof as follows.

Given a distribution p over C, the loss of the learner DISPLAYFORM3 can be optimized individually for each v t since the terms in the sum are independent from one another.

We leverage our results from Lemma 1 to demonstrate how we can frame the problem of finding the minimum perturbation v j such that x + v j ∈ T j as the minimization of a convex function over a convex set.

Since the loss of the learner is constant for points that lie in a particular set T j , we can find the optimal solution by iterating over all sets T j and selecting the perturbation with ℓ 2 norm less than α that is associated with the highest loss.

The best response oracle then follows by repeating the same process for each point (x, y).Given a point (x, y) solving for the minimal perturbation v such that x + v ∈ T j can be expressed as the minimization of a quadratic function subject to n(k − 1) linear inequalities.

DISPLAYFORM4 Each constraint in (7) can be expressed as k − 1 linear inequalities.

For a particular z ∈ [k], c i ∈ C we write c i (x + v) = z as c i,z (x + v) > c i,l (x + v) for all l ∕ = z. Lastly, squaring the norm of the vector is a monotonic transformation and hence does not alter the underlying minimum.

Here we extend the results from our analysis of linear classifiers to other methods for multilabel classification.

In particular, we show that any "all-pairs" or multivector model can be converted to an equivalent "one-vs-all" classifier and hence all of our results also apply to these other approaches.

All-Pairs.

In the "all-pairs" approach, each linear classifier c consists of k 2 linear predictors c i,j trained to predict between labels i, j ∈ [k].

As per convention, we let c i,j (x) = −c j,i (x).

Labels are chosen according to the rule: DISPLAYFORM0 Given an "all-pairs" classifier c, we show how it can be transformed into a "one-vs-all" classifier c DISPLAYFORM1 Multivector.

Lastly, we extend our results to multilabel classification done via class-sensitive feature mappings and the multivector construction by again reducing to the "one-vs-all" case.

Given a function Ψ : DISPLAYFORM2 n , labels are predicted according to the rule: DISPLAYFORM3 While there are several choices for the Ψ, we focus on the most common, the multivector construction:Ψ(x, y) = 0, . . .

, 0 DISPLAYFORM4 , 0, . . . , 0 DISPLAYFORM5 DISPLAYFORM6 This in effect ensures that (9) becomes equivalent to that of the "one-vs-all" approach: DISPLAYFORM7 E CONVERGENCE ANALYSIS OF PROJECTED GRADIENT DESCENT Theorem 1.

Given any precision > 0 and noise budget α > 0:• For a finite set of linear binary classifiers C and a point (x, y), running PGD for T = 4α/ iterations on the objective f (v) = n i=i p[i]ℓ r (c i , x + v, y) converges to a point that is within of the pure strategy Nash equilibrium f (x + v * ), if such an equilibrium exists;• For a finite set of linear multilabel classifiers C, given a label vector s j ∈ [k] n and a distribution p over C, running PGD for T = 4α/ iterations on the objective f (v) = n i=i p[i]ℓ t (c i , x+v, s j,i ) converges to a point x+v (T ) such that f (x+v (T ) )−f (x+v * ) ≤ where x + v * ∈ T j and ||v * || 2 ≤ α, if such a point exists.

Proof.

We know that if a function f is convex and β-smooth, then running projected gradient descent over a convex set, results in the following rate of convergence, where v is the optimal solution and v (1) is the initial starting point (See Theorem 3.7 in BID2 ).

DISPLAYFORM8 Given n classifiers, the objective is Furthermore, since v * is a pure strategy Nash equilibrium, f (v ) = 0 and the maximum difference between f (v) − f (v ), for any v, is bounded by: DISPLAYFORM9 Since ||v (T ) − v || 2 ≤ α, we have that: DISPLAYFORM10 Lastly, we can normalize all the w i such that ||w i || 2 = 1 without changing the predictions of the c i and arrive at our desired result.

For the multiclass case, we have that: Using the fact that all weight vectors w i,j can be transformed to have ℓ 2 norm equal to 1, we have that DISPLAYFORM11 DISPLAYFORM12 .

Lastly, we can check that ℓ t is β-smooth with β = α n i=1 p[i], which yields the same bound as in the binary case.

Theorem 2.

Given a set C of n binary, linear classifiers, a number B, a point (x, y), noise budget α, and a distribution p, finding v with ||v|| 2 ≤ α s.t.

the loss of the learner is exactly B is NP-complete.

Proof.

We can certainly verify in polynomial time that a vector v induces a loss of B simply by calculating the 0-1 loss of each classifier.

Therefore the problem is in NP.To show hardness, we reduce from Subset Sum.

Given n numbers p 1 , . . .

p n and a target number B, 11 we determine our input space to be R n , the point x to be the origin, the label y = −1, and the noise budget α = 1.

Next, we create n binary classifiers of the form c i (x) = 〈e i , x〉 where e i is the ith standard basis vector.

We let p i be the probability with which the learner selects classifier c i .

We claim that there is a subset that sums to B if and only if there exists a region T j ⊂ R n on which the learner achieves loss B. Given the parameters of the reduction, the loss of the learner is determined by the sum of the probability weights of classifiers c i such that c i (x + v) = +1 for points x + v ∈ T j .

If we again identify sets T j with sign vectors s j ∈ {±1} n as per Lemma 2, there is a bijection between the sets T j and the power set of {p 1 , . . .

, p n }.

A number p i is in a subset U j if the ith entry of s j is equal to +1.Lastly, we can check that there are feasible points within each set T j and hence that all subsets within the original Subset Sum instance are valid.

Each T j simply corresponds to a quadrant of R n .

For any ε > 0 and for any T j , there exists a v j with ℓ 2 norm less than ε such that x + v j ∈ T j .

Therefore, there is a subset U j that sums to B if and only if there is a region T j in which the learner achieves loss B.11 Without loss of generality, we can assume that instances of Subset Sum only have values in the range [0, 1].

We can reduce from the more general case by simply normalizing inputs to lie in this range.12 We can again normalize values so that they form a valid probability distribution.

@highlight

Paper analyzes the problem of designing adversarial attacks against multiple classifiers, introducing algorithms that are optimal for linear classifiers and which provide state-of-the-art results for deep learning.