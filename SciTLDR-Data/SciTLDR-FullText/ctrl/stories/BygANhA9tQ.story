Several recent works have developed methods for training classifiers that are certifiably robust against norm-bounded adversarial perturbations.

These methods assume that all the adversarial transformations are equally important, which is seldom the case in real-world applications.

We advocate for cost-sensitive robustness as the criteria for measuring the classifier's performance for tasks where some adversarial transformation are more important than others.

We encode the potential harm of each adversarial transformation in a cost matrix, and propose a general objective function to adapt the robust training method of Wong & Kolter (2018) to optimize for cost-sensitive robustness.

Our experiments on simple MNIST and CIFAR10 models with a variety of cost matrices show that the proposed approach can produce models with substantially reduced cost-sensitive robust error, while maintaining classification accuracy.

Despite the exceptional performance of deep neural networks (DNNs) on various machine learning tasks such as malware detection BID24 , face recognition BID22 and autonomous driving BID2 , recent studies BID26 BID9 have shown that deep learning models are vulnerable to misclassifying inputs, known as adversarial examples, that are crafted with targeted but visually-imperceptible perturbations.

While several defense mechanisms have been proposed and empirically demonstrated to be successful against existing particular attacks BID21 BID9 , new attacks BID3 BID27 BID1 are repeatedly found that circumvent such defenses.

To end this arm race, recent works BID23 BID28 propose methods to certify examples to be robust against some specific norm-bounded adversarial perturbations for given inputs and to train models to optimize for certifiable robustness.

However, all of the aforementioned methods aim at improving the overall robustness of the classifier.

This means that the methods to improve robustness are designed to prevent seed examples in any class from being misclassified as any other class.

Achieving such a goal (at least for some definitions of adversarial robustness) requires producing a perfect classifier, and has, unsurprisingly, remained elusive.

Indeed, BID20 proved that if the metric probability space is concentrated, overall adversarial robustness is unattainable for any classifier with initial constant error.

We argue that overall robustness may not be the appropriate criteria for measuring system performance in security-sensitive applications, since only certain kinds of adversarial misclassifications pose meaningful threats that provide value for potential adversaries.

Whereas overall robustness places equal emphasis on every adversarial transformation, from a security perspective, only certain transformations matter.

As a simple example, misclassifying a malicious program as benign results in more severe consequences than the reverse.

In this paper, we propose a general method for adapting provable defenses against norm-bounded perturbations to take into account the potential harm of different adversarial class transformations.

Inspired by cost-sensitive learning BID5 BID7 ) for non-adversarial contexts, we capture the impact of different adversarial class transformations using a cost matrix C, where each entry represents the cost of an adversary being able to take a natural example from the first class and perturb it so as to be misclassified by the model as the second class.

Instead of reducing the overall robust error, our goal is to minimize the cost-weighted robust error (which we define for both binary and real-valued costs in C).

The proposed method incorporates the specified cost matrix into the training objective function, which encourages stronger robustness guarantees on cost-sensitive class transformations, while maintaining the overall classification accuracy on the original inputs.

Contributions.

By encoding the consequences of different adversarial transformations into a cost matrix, we introduce the notion of cost-sensitive robustness (Section 3.1) as a metric to assess the expected performance of a classifier when facing adversarial examples.

We propose an objective function for training a cost-sensitive robust classifier (Section 3.2).

The proposed method is general in that it can incorporate any type of cost matrix, including both binary and real-valued.

We demonstrate the effectiveness of the proposed cost-sensitive defense model for a variety of cost scenarios on two benchmark image classification datasets: MNIST (Section 4.1) and CIFAR10 (Section 4.2).

Compared with the state-of-the-art overall robust defense model , our model achieves significant improvements in cost-sensitive robustness for different tasks, while maintaining approximately the same classification accuracy on both datasets.

Notation.

We use lower-case boldface letters such as x for vectors and capital boldface letters such as A to represent matrices.

Let [m] be the index set {1, 2, . . .

, m} and A ij be the (i, j)-th entry of matrix A. Denote the i-th natural basis vector, the all-ones vector and the identity matrix by e i , 1 and I respectively.

For any vector x ∈ R d , the ∞ -norm of x is defined as DISPLAYFORM0

In this section, we provide a brief introduction on related topics, including neural network classifiers, adversarial examples, defenses with certified robustness, and cost-sensitive learning.

A K-layer neural network classifier can be represented by a function f : DISPLAYFORM0 , for any x ∈ X .

For k ∈ {1, 2, . . .

, K −2}, the mapping function f k (·) typically consists of two operations: an affine transformation (either matrix multiplication or convolution) and a nonlinear activation.

In this paper, we consider rectified linear unit (ReLU) as the activation function.

If denote the feature vector of the k-th layer as z k , then f k (·) is defined as DISPLAYFORM1 where W k denotes the weight parameter matrix and b k the bias vector.

The output function f K−1 (·) maps the feature vector in the last hidden layer to the output space Y solely through matrix multiplication: DISPLAYFORM2 , where z K can be regarded as the estimated score vector of input x for different possible output classes.

In the following discussions, we use f θ to represent the neural network classifier, where DISPLAYFORM3 To train the neural network, a loss function DISPLAYFORM4 , where x i is the i-th input vector and y i denotes its class label.

Cross-entropy loss is typically used for multiclass image classification.

With proper initialization, all model parameters are then updated iteratively using backpropagation.

For any input example x, the predicted label y is given by the index of the largest predicted score among all classes, argmax j [f θ ( x)] j .

An adversarial example is an input, generated by some adversary, which is visually indistinguishable from an example from the natural distribution, but is able to mislead the target classifier.

Since "visually indistinguishable" depends on human perception, which is hard to define rigorously, we consider the most popular alternative: input examples with perturbations bounded in ∞ -norm BID9 .

More formally, the set of adversarial examples with respect to seed example {x 0 , y 0 } and classifier f θ (·) is defined as DISPLAYFORM0 where > 0 denotes the maximum perturbation distance.

Although p distances are commonly used in adversarial examples research, they are not an adequate measure of perceptual similarity BID25 and other minimal geometric transformations can be used to find adversarial examples BID8 BID11 BID31 .

Nevertheless, there is considerable interest in improving robustness in this simple domain, and hope that as this research area matures we will find ways to apply results from studying simplified problems to more realistic ones.

A line of recent work has proposed defenses that are guaranteed to be robust against norm-bounded adversarial perturbations.

BID10 proved formal robustness guarantees against 2 -norm bounded perturbations for two-layer neural networks, and provided a training method based on a surrogate robust bound.

BID23 developed an approach based on semidefinite relaxation for training certified robust classifiers, but was limited to two-layer fullyconnected networks.

Our work builds most directly on , which can be applied to deep ReLU-based networks and achieves the state-of-the-art certified robustness on MNIST dataset.

Following the definitions in , an adversarial polytope Z (x) with respect to a given example x is defined as DISPLAYFORM0 which contains all the possible output vectors for the given classifier f θ by perturbing x within an ∞ -norm ball with radius .

A seed example, {x 0 , y 0 }, is said to be certified robust with respect to maximum perturbation distance , if the corresponding adversarial example set A (x 0 , y 0 ; θ) is empty.

Equivalently, if we solve, for any output class y targ = y 0 , the optimization problem, DISPLAYFORM1 then according to the definition of A (x 0 , y 0 ; θ) in (2.1), {x 0 , y 0 } is guaranteed to be robust provided that the optimal objective value of (2.3) is positive for every output class.

To train a robust model on a given dataset {x i , y i } N i=1 , the standard robust optimization aims to minimize the sample loss function on the worst-case locations through the following adversarial loss DISPLAYFORM2 where L(·, ·) denotes the cross-entropy loss.

However, due to the nonconvexity of the neural network classifier f θ (·) introduced by the nonlinear ReLU activation, both the adversarial polytope (2.2) and training objective (2.4) are highly nonconvex.

In addition, solving optimization problem (2.3) for each pair of input example and output class is computationally intractable.

Instead of solving the optimization problem directly, proposed an alternative training objective function based on convex relaxation, which can be efficiently optimized through a dual network.

Specifically, they relaxed Z (x) into a convex outer adversarial polytope Z (x) by replacing the ReLU inequalities for each neuron z = max{ z, 0} with a set of inequalities, 5) where u, denote the lower and upper bounds on the considered pre-ReLU activation.

1 Based on the relaxed outer bound Z (x), they propose the following alternative optimization problem, 6) which is in fact a linear program.

Since Z (x) ⊆ Z (x) for any x ∈ X , solving (2.6) for all output classes provides stronger robustness guarantees compared with (2.3), provided all the optimal objective values are positive.

In addition, they derived a guaranteed lower bound, denoted by J x 0 , g θ (e y0 − e ytarg ) , on the optimal objective value of Equation 2.6 using duality theory, where g θ (·) is a K-layer feedforward dual network (Theorem 1 in ).

Finally, according to the properties of cross-entropy loss, they minimize the following objective to train the robust model, which serves as an upper bound of the adversarial loss (2.4): DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 where g θ (·) is regarded as a columnwise function when applied to a matrix.

Although the proposed method in achieves certified robustness, its computational complexity is quadratic with the network size in the worst case so it only scales to small networks.

Recently, extended the training procedure to scale to larger networks by using nonlinear random projections.

However, if the network size allows for both methods, we observe a small decrease in performance using the training method provided in .

Therefore, we only use the approximation techniques for the experiments on CIFAR10 ( §4.2), and use the less scalable method for the MNIST experiments ( §4.1).

Cost-sensitive learning BID5 BID7 BID18 was proposed to deal with unequal misclassification costs and class imbalance problems commonly found in classification applications.

The key observation is that cost-blind learning algorithms tend to overwhelm the major class, but the neglected minor class is often our primary interest.

For example, in medical diagnosis misclassifying a rare cancerous lesion as benign is extremely costly.

Various cost-sensitive learning algorithms BID15 BID32 BID33 BID12 have been proposed in literature, but only a few algorithms, limited to simple classifiers, considered adversarial settings.

2 BID4 studied the naive Bayes classifier for spam detection in the presence of a cost-sensitive adversary, and developed an adversary-aware classifier based on game theory.

BID0 proposed a cost-sensitive robust minimax approach that hardens a linear discriminant classifier with robustness in the adversarial context.

All of these methods are designed for simple linear classifiers, and cannot be directly extended to neural network classifiers.

In addition, the robustness of their proposed classifier is only examined experimentally based on the performance against some specific adversary, so does not provide any notion of certified robustness.

Recently, BID6 advocated for the idea of using application-level semantics in adversarial analysis, however, they didn't provide a formal method on how to train such classifier.

Our work provides a practical training method that hardens neural network classifiers with certified cost-sensitive robustness against adversarial perturbations.

The approach introduced in penalizes all adversarial class transformations equally, even though the consequences of adversarial examples usually depends on the specific class transformations.

Here, we provide a formal definition of cost-sensitive robustness ( §3.1) and propose a general method for training cost-sensitive robust models ( §3.2).

Our approach uses a cost matrix C that encodes the cost (i.e., potential harm to model deployer) of different adversarial examples.

First, we consider the case where there are m classes and C is a m × m binary matrix with C jj ∈ {0, 1}. The value C jj indicates whether we care about an adversary transforming a seed input in class j into one recognized by the model as being in class j .

If the adversarial transformation j → j matters, C jj = 1, otherwise C jj = 0.

Let Ω j = {j ∈ [m] : C jj = 0} be the index set of output classes that induce cost with respect to input class j.

For any j ∈ [m], let δ j = 0 if Ω j is an empty set, and δ j = 1 otherwise.

We are only concerned with adversarial transformations from a seed class j to target classes j ∈ Ω j .

For any example x in seed class j, x is said to be certified cost-sensitive robust if the lower bound J (x, g θ (e j − e j )) ≥ 0 for all j ∈ Ω j .

That is, no adversarial perturbations in an ∞ -norm ball around x with radius can mislead the classifier to any target class in Ω j .The cost-sensitive robust error on a dataset {x i , y i } N i=1 is defined as the number of examples that are not guaranteed to be cost-sensitive robust over the number of non-zero cost candidate seed examples: DISPLAYFORM0 where #A represents the cardinality of a set A, and N j is the total number of examples in class j.

Next, we consider a more general case where C is a m×m real-valued cost matrix.

Each entry of C is a non-negative real number, which represents the cost of the corresponding adversarial transformation.

To take into account the different potential costs among adversarial examples, we measure the costsensitive robustness by the average certified cost of adversarial examples.

The cost of an adversarial example x in class j is defined as the sum of all C jj such that J (x, g θ (e j − e j )) < 0.

Intuitively speaking, an adversarial example will induce more cost if it can be adversarially misclassified as more target classes with high cost.

Accordingly, the robust cost is defined as the total cost of adversarial examples divided by the total number of valued seed examples: DISPLAYFORM1 where 1(·) denotes the indicator function.

Recall that our goal is to develop a classifier with certified cost-sensitive robustness, as defined in §3.1, while maintaining overall classification accuracy.

According to the guaranteed lower bound, J x 0 , g θ (e y0 − e ytarg ) on Equation 2.6 and inspired by the cost-sensitive CE loss BID12 , we propose the following robust optimization with respect to a neural network classifier f θ : DISPLAYFORM0 where α ≥ 0 denotes the regularization parameter.

The first term in Equation 3.2 denotes the cross-entropy loss for standard classification, whereas the second term accounts for the cost-sensitive robustness.

Compared with the overall robustness training objective function (2.7), we include a regularization parameter α to control the trade-off between classification accuracy on original inputs and adversarial robustness.

To provide cost-sensitivity, the loss function selectively penalizes the adversarial examples based on their cost.

For binary cost matrixes, the regularization term penalizes every cost-sensitive adversarial example equally, but has no impact for instances where C jj = 0.

For the real-valued costs, a larger value of C jj increases the weight of the corresponding adversarial transformation in the training objective.

This optimization problem (3.2) can be solved efficiently using gradient-based algorithms, such as stochastic gradient descent and ADAM BID13 .

We evaluate the performance of our cost-sensitive robustness training method on models for two benchmark image classification datasets: MNIST (LeCun et al., 2010) and CIFAR10 BID14 ( §2.3) as a baseline.

For both datasets, the relevant family of attacks is specified as all the adversarial perturbations that are bounded in an ∞ -norm ball.

Our goal in the experiments is to evaluate how well a variety of different types of cost matrices can be supported.

MNIST and CIFAR-10 are toy datasets, thus there are no obvious cost matrices that correspond to meaningful security applications for these datasets.

Instead, we select representative tasks and design cost matrices to capture them.

For MNIST, we use the same convolutional neural network architecture BID16 as , which includes two convolutional layers, with 16 and 32 filters respectively, and a two fully-connected layers, consisting of 100 and 10 hidden units respectively.

ReLU activations are applied to each layer except the last one.

For both our cost-sensitive robust model and the overall robust model, we randomly split the 60,000 training samples into five folds of equal size, and train the classifier over 60 epochs on four of them using the Adam optimizer BID13 with batch size 50 and learning rate 0.001.

We treat the remaining fold as a validation dataset for model selection.

In addition, we use the -scheduling and learning rate decay techniques, where we increase from 0.05 to the desired value linearly over the first 20 epochs and decay the learning rate by 0.5 every 10 epochs for the remaining epochs.

Baseline: Overall Robustness.

FIG0 illustrates the learning curves of both classification error and overall robust error during training based on robust loss (2.7) with maximum perturbation distance = 0.2.

The model with classification error less than 4% and minimum overall robust error on the validation dataset is selected over the 60 training epochs.

The best classifier reaches 3.39% classification error and 13.80% overall robust error on the 10,000 MNIST testing samples.

We report the robust test error for every adversarial transformation in FIG0 (b) (for the model without any robustness training all of the values are 100%).

The (i, j)-th entry is a bound on the robustness of that seed-target transformation-the fraction of testing examples in class i that cannot be certified robust against transformation into class j for any norm-bounded attack.

As shown in FIG0 , the vulnerability to adversarial transformations differs considerably among class pairs and appears correlated with perceptual similarity.

For instance, only 0.26% of seeds in class 1 cannot be certified robust for target class 9 compare to 10% of seeds from class 9 into class 4.Binary Cost Matrix.

Next, we evaluate the effectiveness of cost-sensitive robustness training in producing models that are more robust for adversarial transformations designated as valuable.

We consider four types of tasks defined by different binary cost matrices that capture different sets of adversarial transformations: single pair: particular seed class s to particular target class t; single seed: particular seed class s to any target class; single target: any seed class to particular target class t; and Figure 2 : Cost-sensitive robust error using the proposed model and baseline model on MNIST for different binary tasks: (a) treat each digit as the seed class of concern respectively; (b) treat each digit as the target class of concern respectively.

multiple: multiple seed and target classes.

For each setting, the cost matrix is defined as C ij = 1 if (i, j) is selected; otherwise, C ij = 0.

In general, we expect that the sparser the cost matrix, the more opportunity there is for cost-sensitive training to improve cost-sensitive robustness over models trained for overall robustness.

For the single pair task, we selected three representative adversarial goals: a low vulnerability pair (0, 2), medium vulnerability pair (6, 5) and high vulnerability pair (4, 9).

We selected these pairs by considering the robust error results on the overall-robustness trained model FIG0 ) as a rough measure for transformation hardness.

This is generally consistent with intuitions about the MNIST digit classes (e.g., 9 and 4 look similar, so are harder to induce robustness against adversarial transformation), as well as with the visualization results produced by dimension reduction techniques, such as t-SNE BID19 .Similarly, for the single seed and single target tasks we select three representative examples representing low, medium, and high vulnerability to include in TAB1 and provide full results for all the single-seed and single target tasks for MNIST in Figure 2 .

For the multiple transformations task, we consider four variations: (i) the ten most vulnerable seed-target transformations; (ii) ten randomly-selected seed-target transformations; (iii) all the class transformations from odd digit seed to any other class; (iv) all the class transformations from even digit seed to any other class.

TAB1 summarizes the results, comparing the cost-sensitive robust error between the baseline model trained for overall robustness and a model trained using our cost-sensitive robust optimization.

The cost-sensitive robust defense model is trained with = 0.2 based on loss function (3.2) and the corresponding cost matrix C. The regularization parameter α is tuned via cross validation (see Appendix A for details).

We report the selected best α, classification error and cost-sensitive robust error on the testing dataset.

Our model achieves a substantial improvement on the cost-sensitive robustness compared with the baseline model on all of the considered tasks, with no significant increases in normal classification error.

The cost-sensitive robust error reduction varies from 30% to 90%, and is generally higher for sparse cost matrices.

In particular, our classifier reduces the number of cost-sensitive adversarial examples from 198 to 12 on the single target task with digit 1 as the target class.

Real-valued Cost Matrices.

Loosely motivated by a check forging adversary who obtains value by changing the semantic interpretation of a number BID21 , we consider two real-valued cost matrices: small-large, where only adversarial transformations from a smaller digit class to a larger one are valued, and the cost of valued-transformation is quadratic with the absolute difference between the seed and target class digits: C ij = (i − j) 2 if j > i, otherwise C ij = 0; large-small: only adversarial transformations from a larger digit class to a smaller one are valued: C ij = (i − j) 2 if i > j, otherwise C ij = 0.

We tune α for the cost-sensitive robust model on the training MNIST dataset via cross validation, and set all the other parameters the same as in the binary case.

The certified robust error for every adversarial transformation on MNIST testing dataset is shown in Figure 3 , and the classification error and robust cost are given in TAB2 .

Compared with the model trained for overall robustness FIG0 ), our trained classifier achieves stronger robustness guarantees on the adversarial transformations that induce costs, especially for those with larger costs.

DISPLAYFORM0 0.7% 1.3% 1.1% 0.6% 1.3% 0.8% 0.6% 0.5% 0.2% 1.7% 2.2% 1.0% 0.4% 0.6% 0.5% 0.1% 0.5% 0.0% 11.8% 17.6% 5.6% 1.2% 1.1% 0.9% (b) large-small Figure 3 : Heatmaps of robust test error using our cost-sensitive robust classifier on MNIST for various real-valued cost tasks: (a) small-large; (b) large-small.

We use the same neural network architecture for the CIFAR10 dataset as , with four convolutional layers and two fully-connected layers.

For memory and computational efficiency, we incorporate the approximation technique based on nonlinear random projection during the training phase , §3.2).

We train both the baseline model and our model using random projection of 50 dimensions, and optimize the training objective using SGD.

Other parameters such as learning rate and batch size are set as same as those in .Given a specific task, we train the cost-sensitive robust classifier on 80% randomly-selected training examples, and tune the regularization parameter α according to the performance on the remaining examples as validation dataset.

The tasks are similar to those for MNIST ( §4.1), except for the multiple transformations task we cluster the ten CIFAR10 classes into two large groups: animals and vehicles, and consider the cases where only transformations between an animal class and a vehicle class are sensitive, and the converse.

TAB4 shows results on the testing data based on different robust defense models with = 2/255.

For all of the aforementioned tasks, our models substantially reduce the cost-sensitive robust error while keeping a lower classification error than the baseline.

For the real-valued task, we are concerned with adversarial transformations from seed examples in vehicle classes to other target classes.

In addition, more cost is placed on transformations from vehicle to animal, which is 10 times larger compared with that from vehicle to vehicle.

and 4(b) illustrate the pairwise robust test error using overall robust model and the proposed classifier for the aforementioned real-valued task on CIFAR10.

We investigate the performance of our model against different levels of adversarial strength by varying the value of that defines the ∞ ball available to the adversary.

FIG2 show the overall classification and cost-sensitive robust error of our best trained model, compared with the baseline model, on the MNIST single seed task with digit 9 and CIFAR single seed task with dog as the seed class of concern, as we vary the maximum ∞ perturbation distance.

Under all the considered attack models, the proposed classifier achieves better cost-sensitive adversarial robustness than the baseline, while maintaining similar classification accuracy on original data points.

As the adversarial strength increases, the improvement for cost-sensitive robustness over overall robustness becomes more significant.

By focusing on overall robustness, previous robustness training methods expend a large fraction of the capacity of the network on unimportant transformations.

We argue that for most scenarios, the actual harm caused by an adversarial transformation often varies depending on the seed and target class, so robust training methods should be designed to account for these differences.

By incorporating a cost matrix into the training objective, we develop a general method for producing a cost-sensitive robust classifier.

Our experimental results show that our cost-sensitive training method works across a variety of different types of cost matrices, so we believe it can be generalized to other cost matrix scenarios that would be found in realistic applications.

There remains a large gap between the small models and limited attacker capabilities for which we can achieve certifiable robustness, and the complex models and unconstrained attacks that may be important in practice.

The scalability of our techniques is limited to the toy models and simple attack norms for which certifiable robustness is currently feasible, so considerable process is needed before they could be applied to realistic scenarios.

However, we hope that considering cost-sensitive robustness instead of overall robustness is a step towards achieving more realistic robustness goals.

Our implementation, including code for reproducing all our experiments, is available as open source code at https://github.com/xiaozhanguva/Cost-Sensitive-Robustness.

For experiments on the MNIST dataset, we first perform a coarse tuning on regularization parameter α with searching grid {10 −2 , 10 −1 , 10 0 , 10 1 , 10 2 }, and select the most appropriate one, denoted by α coarse , with overall classification error less than 4% and the lowest cost-sensitive robust error on validation dataset.

Then, we further finely tune α from the range {2 −3 , 2 −2 , 2 −1 , 2 0 , 2 1 , 2 2 , 2 3 } · α coarse , and choose the best robust model according to the same criteria.

Figures 6(a) and 6(b) show the learning curves for task B with digit 9 as the selected seed class based on the proposed cost-sensitive robust model with varying α (we show digit 9 because it is one of the most vulnerable seed classes).

The results suggest that as the value of α increases, the corresponding classifier will have a lower cost-sensitive robust error but a higher classification error, which is what we expect from the design of (3.2).We observe similar trends for the learning curves for the other tasks, so do not present them here.

For the CIFAR10 experiments, a similar tuning strategy is implemented.

The only difference is that we use 35% as the threshold of overall classification error for selecting the best α.

As discussed in Section 2.4, prior work on cost-sensitive learning mainly focuses on the nonadversarial setting.

In this section, we investigate the robustness of the cross-entropy based costsensitive classifier proposed in BID12 , and compare the performance of their classifier with our proposed cost-sensitive robust classifier.

Given a set of training examples {(x i , y i )} DISPLAYFORM0 and cost matrix C with each entry representing the cost of the corresponding misclassification, the evaluation metric for cost-sensitive learning is defined as the average cost of misclassifications, or more concretely DISPLAYFORM1 C yi yi , where y i = argmax DISPLAYFORM2 where m is the total number of class labels and f θ (·) denotes the neural network classifier as introduced in Section 2.1.

In addition, the cross-entropy based cost-sensitive training objective takes the following form: To provide a fair comparison, we assume the cost matrix used for (B.1) coincides with the cost matrix used for cost-sensitive robust training (3.2) in our experiment, whereas they are unlikely to be the same for real security applications.

For instance, misclassifying a benign program as malicious may still induce some cost in the non-adversarial setting, whereas the adversary may only benefit from transforming a malicious program into a benign one.

We consider the small-large real-valued task for MNIST, where the cost matrix C is designed as C ij = 0.1, if i > j; C ij = 0, if i = j; C ij = (i − j) 2 , otherwise.

TAB5 demonstrates the comparison results of different classifiers in such setting: the baseline standard deep learning classifier, a standard cost-sensitive classifier BID12 trained using (B.1), classifier trained for overall robustness and our proposed classifier trained for cost-sensitive robustness.

Compared with baseline, the standard cost-sensitive classifier indeed reduces the misclassification cost.

But, it does not provide any improvement on the robust cost, as defined in (3.1).

In sharp contrast, our robust training method significantly improves the cost-sensitive robustness.

<|TLDR|>

@highlight

A general method for training certified cost-sensitive robust classifier against adversarial perturbations

@highlight

Calculates and plugs in the costs of adversarial attack into the objective of optimization to get a model that is cost-sensitively robust against adversarial attacks. 

@highlight

Build on semnial work by Dalvi et al. and extends approach to certifiable robustness with a cost matrix that specifies for each pair of source-target classes whether the model should be robust to adversarial examples.