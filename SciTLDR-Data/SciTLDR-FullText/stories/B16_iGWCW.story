In this paper, a deep boosting algorithm is developed to learn more discriminative ensemble classifier by seamlessly combining a set of base deep CNNs (base experts) with diverse capabilities, e.g., these base deep CNNs are sequentially trained to recognize a set of  object classes in an easy-to-hard way according to their learning complexities.

Our experimental results have demonstrated that our deep boosting algorithm can significantly improve the accuracy rates on large-scale visual recognition.

The rapid growth of computational powers of GPUs has provided good opportunities for us to develop scalable learning algorithms to leverage massive digital images to train more discriminative classifiers for large-scale visual recognition applications, and deep learning BID19 BID20 BID3 has demonstrated its outstanding performance because highly invariant and discriminant features and multi-way softmax classifier are learned jointly in an end-to-end fashion.

Before deep learning becomes so popular, boosting has achieved good success on visual recognition BID21 .

By embedding multiple weak learners to construct an ensemble one, boosting BID15 can significantly improve the performance by sequentially training multiple weak learners with respect to a weighted error function which assigns larger weights to the samples misclassified by the previous weak learners.

Thus it is very attractive to invest whether boosting can be integrated with deep learning to achieve higher accuracy rates on large-scale visual recognition.

By using neural networks to replace the traditional weak learners in the boosting frameworks, boosting of neural networks has received enough attentions BID23 BID10 BID7 BID9 .

All these existing deep boosting algorithms simply use the weighted error function (proposed by Adaboost (Schapire, 1999) ) to replace the softmax error function (used in deep learning ) that treats all the errors equally.

Because different object classes may have different learning complexities, it is more attractive to invest new deep boosting algorithm that can use different weights over various object classes rather than over different training samples.

Motivated by this observation, a deep boosting algorithm is developed to generate more discriminative ensemble classifier by combining a set of base deep CNNs with diverse capabilities, e.g., all these base deep CNNs (base experts) are sequentially trained to recognize different subsets of object classes in an easy-to-hard way according to their learning complexities.

The rest of the paper is organized as: Section 2 briefly reviews the related work; Section 3 introduce our deep boosting algorithm; Section 4 reports our experimental results; and we conclude this paper at Section 5.

In this section, we briefly review the most relevant researches on deep learning and boosting.

Even deep learning has demonstrated its outstanding abilities on large-scale visual recognition BID6 BID19 BID20 BID3 BID4 , it still has room to improve: all the object classes are arbitrarily assumed to share similar learning complexities and a multi-way softmax is used to treat them equally.

For recognizing large numbers of object classes, there may have significant differences on their learning complexi-ties, e.g., some object classes may be harder to be recognized than others.

Thus learning their deep CNNs jointly may not be able to achieve the global optimum effectively because the gradients of their joint objective function are not uniform for all the object classes and such joint learning process may distract on discerning some object classes that are hard to be discriminated.

For recognizing large numbers of object classes with diverse learning complexities, it is very important to organize them in an easy-to-hard way according to their learning complexities and learn their deep CNNs sequentially.

By assigning different weights to the training samples adaptively, boosting BID15 BID1 BID16 has provided an easy-to-hard approach to train a set of weak learners sequentially.

Thus it is very attractive to invest whether we can leverage boosting to learn a set of base deep CNNs sequentially for recognizing large numbers of object classes in an easy-to-hard way.

Some deep boosting algorithms have been developed by seamlessly integrating boosting with deep neural networks to improve the performance in practice.

BID17 BID18 proposed the first work to integrate Adaboost with neural networks for online character recognition application.

BID23 extended the Adaboosting neural networks algorithm for credit scoring.

Recently, BID11 developed a deep incremental boosting method which increases the size of neural network at each round by adding new layers at the end of the network.

Moreover, BID10 integrated residual networks with incremental boosting and built an ensemble of residual networks via adding one more residual block to the previous residual network at each round of boosting.

All these methods combine the merits of boosting and neural networks; they train each base network either using a different training set by resampling with a probability distribution derived from the error weight, or directly using the weighted cost function for the base network.

Alternatively, BID14 proposed a margin enforcing loss for multi-class boosting and presented two ways to minimize the resulting risk: the one is coordinate descent approach which updates one predictor component at a time, the other way is based on directional functional derivative and updates all components jointly.

By applying the first way, i.e., coordinate descent, BID0 designed ensemble learning algorithm for binary-class classification using deep decision trees as base classifiers and gave the data-dependent learning bound of convex ensembles, and BID7 furthermore extended it to multi-class version.

By applying the second way, i.e., directional derivative descent, BID9 developed an algorithm for boosting deep convolutional neural networks (CNNs) based on least squares between weights and directional derivatives, which differs from the original method based on inner product of weights and directional derivative in BID14 .

All above algorithms focus on seeking the optimal ensemble predictor via changing the error weights of samples; they either update one component of the predictor per boosting iteration, or update all components simultaneously.

On the other hand, our deep boosting algorithm focuses on combining a set of base deep CNNs with diverse capabilities: (1) large numbers of object classes are automatically organized in an easyto-hard way according to their learning complexities; (2) all these base deep CNNs (base experts) are sequentially learned to recognize different subsets of object classes; and (3) these base deep CNNs with diverse capabilities are seamlessly combined to generate more discriminative ensemble classifier.

In this paper, a deep boosting algorithm is developed by seamlessly combining a set of base deep CNNs with various capabilities, e.g., all these base deep CNNs are sequentially trained to recognize different subsets of object classes in an easy-to-hard way according to their learning complexities.

Our deep boosting algorithm uses the base deep CNNs as its weak learners, and many well-designed deep networks (such as AlexNet BID6 , VGG BID19 , ResNet BID3 , and huang2016densely), can be used as its base deep CNNs.

It is worth noting that all these well-designed deep networks [] optimize their structures (i.e., numbers of layers and units in each layer), their node weights and their softmax jointly in an end-to-end manner for recognizing the same set of object classes.

Thus our deep boosting algorithm is firstly implemented for recognizing 1,00 object classes, however, it is straightward to extend our current implementation Normalization: DISPLAYFORM0 Training the t th base deep CNNs f t (x) via Loss t with respect to the importance distribution DISPLAYFORM1 Calculating the error per category for f t (x): ε t (l), (l = 1, ..., C); DISPLAYFORM2 Computing the weighted error for f t (x): DISPLAYFORM3 Setting DISPLAYFORM4 .., C), so that hard object classes misclassified by f t (x) can receive larger weights (importances) when training the (t + 1) th base deep CNNs at the next round; 8: end for 9: Ensembling: DISPLAYFORM5 when huge deep networks (with larger capacities) are available in the future and being used as the base deep CNNs.

As illustrated in Algorithm 1, our deep boosting algorithm contains the following key components: (a) Training the t th base deep CNNs (base expert) f t (x) by focusing on achieving higher accuracy rates for some particular object classes; (b) Estimating the weighted error function for the t th base deep CNNs f t (x) according to the distribution of importances D t for C object classes; (c) Updating the distribution of importances D t+1 for C object classes to train the (t + 1) th base deep CNNs by spending more efforts on distinguishing the hard object classes which are not classified very well by the previous base deep CNNs; (d) Such iterative training process stops when the maximum number of iterations is reached or a certain level of the accuracy rates is achieved.

For the t th base expert f t (x), we firstly employ deep CNNs to map x into more separable feature space h t (x; θ t ), followed by a fully connected discriminant layer and a C-way softmax layer.

The output of the t th base expert is the predicted multi-class distribution, denoted as DISPLAYFORM0 ⊤ , whose each component p t (l|x) is the probability score of x assigned to the object class l, (l = 1, ..., C): DISPLAYFORM1 where θ t and w lt , (l = 1, ..., C) are the model parameters for the t th base expert f t (x).

Based on the above probability score, the category label of x can be predicted by the t th base expert as follows: DISPLAYFORM2 Suppose that training set consists of N labeled samples from C classes: DISPLAYFORM3 .

To train the t th base expert f t (x), the model parameters can be learned by maximizing the objective function in the form of weighted margin as follows: DISPLAYFORM4 where DISPLAYFORM5 Herein the indicator function 1(y i = l) is equal to 1 if y i = l; otherwise zero.

N l denotes the number of samples belonging to the l th object class.

D t (l) is the normalized importance score for class l in the t th base expert f t (x).

By using the distribution of importances [ D t (1), ..., D t (C)] to approximate the learning complexities for C object classes, our deep boosting algorithm can push the current base deep CNNs to focus on distinguishing the object classes which are hard classified by the previous base deep CNNs, thus it can support an easy-to-hard solution for large-scale visual recognition.ξ lt measures the margin between the average confidence on correctly classified examples and the average confidence on misclassified examples for the l th object class.

If the second item in Eq. FORMULA11 is small enough and negligible, DISPLAYFORM6 , then maximizing the objective function in Eq. FORMULA10 is equivalent to maximizing the weighted likelihood.

For the t th base expert f t (x), the classification error rate over the training samples in l th object class is as follows: DISPLAYFORM7 This error rate is used to update category weight and the loss function of the next weak learner, and above definition encourages predictors with large margin to improve the discrimination between correct class and incorrect classe competing with it.

Error rate calculated by Eq. FORMULA15 is in soft decision with probability; alternatively, we can also simply compute the error rate in hard decision as DISPLAYFORM8 where the hyperparameter λ controls the threshold, and we constrain λ > 1 2 (i.e., 1 2λ < 1 ) such that the threshold makes sense.

The larger the hyper-parameter λ is, the more strict the precision requirement becomes.

We then compute the weighted error rate ε t over all classes for f t (x) such that hard object classes are focused on by the next base expert.

DISPLAYFORM9 The distribution of importances is initialized equally for all C object classes: DISPLAYFORM10 , and it is updated along the iterative learning process by emphasizing the object classes which are heavily misclassified by the previous base deep CNNs: DISPLAYFORM11 where β t should be an increasing function of ε t , and its range should be 0 < β t < 1.

It should be pointed out that λϵ t (l) denotes the product of λ and ϵ t (l).

Such update of distribution encourages the next base network focusing on the categories that are hard to classify.

As shown in Section 4, to guarantee the upper boundary of ratio (the number of heavily misclassified categories over the number of all classes) to be minimized, we set DISPLAYFORM12 Normalization of the updated importances distribution can be easily carried out: DISPLAYFORM13 The distribution of importances is used to: (a) separate the hard object classes (heavily misclassified by the previous base deep CNNs) from the easy object classes (which have been classified correctly by the previous base deep CNNs); (b) estimate the weighted error function for the (t + 1) th base deep CNNs f t+1 (x), so that it can spend more efforts on distinguishing the hard object classes misclassified by the previous base deep CNNs.

After T iterations, we can obtain T base deep CNNs (base experts) {f 1 , · · · , f t , · · · , f T }, which are sequentially trained to recognize different subsets of C object classes in an easy-to-hard way according to their learning complexities.

All these T base deep CNNs are seamlessly combined to generate more discriminative ensemble classifier g(x) for recognizing C object classes: DISPLAYFORM14 where DISPLAYFORM15 is a normalization factor.

By diversifying a set of base deep CNNs on their capabilities (i.e., they are trained to recognize different subsets of C object classes in an easyto-hard way), our deep boosting algorithm can obtain more discriminative ensemble classifier g(x) to significantly improve the accuracy rates on large-scale visual recognition.

To apply such ensembled classifier for recognition, for a given test sample x test , it firstly goes through all these base deep CNNs to obtain T deep representations {h 1 , · · · , h T } and then its final probability score p(l|x test ) to be assigned into the lth object class is calculated as follows: DISPLAYFORM16 3.2 SELECTION OF β tIn our deep boosting algorithm, β t is selected to be an increasing function of error rate ε t , with its range [0, 1].

β t is employed in two folds: (i) As seen in Eq. (7) , β t helps to update the importance of different categories such that hard object classes are emphasized; (ii) As seen in Eq. FORMULA7 and Eq.(11), reciprocals of β t are the combination coefficients for the final ensemble classifier such that those base experts with low error rate have large weight.

The criterion of hard object classes for the tth expert is DISPLAYFORM17 for each t, (t = 1, ..., T ); it implies that the lth object class is hard for all T experts.

Let ♯{l : ϵ min (l) > 1 2λ } denote the the number of hard object classes for all T experts.

Inspired by BID1 , we now show that the selection of β t as in Eq.(8) guarantees the upper boundary of ratio (the number of heavily misclassified categories over the number of all classes) to be minimized.

It can be shown that for 0 < x < 1 and 0 < α < 1, we have x α ≤ 1 − (1 − x)α.

According to Eq. (7) : DISPLAYFORM18 According to Eq.(6) and Eq.(9), we get: DISPLAYFORM19 By substituting Eq. (7) into Eq. FORMULA7 , we get DISPLAYFORM20 DISPLAYFORM21 Combining Eq. FORMULA7 with Eq.(16), we get DISPLAYFORM22 To minimize the rightside, we set its partial derivative with respect to β t to zero: DISPLAYFORM23 Since β t only exists in the tth factor, above equation is equivalent to DISPLAYFORM24 Solving it, we find that β t can be optimally selected as: DISPLAYFORM25

We substitute β t = λεt 1−λεt into Eq.(17), and get the upper boundary of ratio (the number of hard object categories over the number of all classes): DISPLAYFORM0 Now we discuss the range for the hyper-parameter λ.

Recall that the criterion of hard object classes for the tth expert is ϵ t (l) > From the relation between λε t and λε t (1 − λε t ), as illustrated in Fig.1 , we can see the effect of λ on the upper boundary of ratio (the number of hard object categories over the number of all classes) in Eq.(18).• In the yellow shaded region, λ ∈ [ 1 2 , 1 2εt ], i.e., εt 2 < λε t < 1 2 , the condition 0 < β t < 1 is satisfied, and the upper boundary of hard category percentage in Eq.(18) increases with λ increasing, the reason for which is that when λ increases, the precision requirement increases, thus the number of hard categories increases too.• On the right side of the yellow shaded region, λ > 1 2εt , i.e., λε t > 1 2 .

In this case, the condition 0 < β t = λεt 1−λεt < 1 is not satisfied, thus the update of importance distribution in Eq. (7) can not effectively emphasize the object classes which are heavily misclassified by the previous experts.

In hard classification task, large error rates ε t tend to result in λε t larger than or approaching 1 2 , and β t larger than or approaching 1.

The value of λ should be set smaller to alleviate large ε t such that λε t < 1 2 and 0 < β t < 1.• On the left side of the yellow shaded region, λ < 1 2 , i.e.,

The procedure of learning the t th base expert repeatedly adjusts the parameters of the corresponding deep network so as to maximize the objective function O t in Eq.(3).

To maximize O t , it is necessary to calculate its gradients with respect to all parameters, including the weights {w lt } C l=1 and the set of model parameters θ t .For clearance, we denote DISPLAYFORM0 Thus, the probability score of x assigned to the object class l, (l = 1, ..., C), in Eq.(1) can be written as DISPLAYFORM1 Then, the objective function in Eq.(3) can be denoted as DISPLAYFORM2 From above presentations, it can be more clearly seen that the objective is a composite function.

DISPLAYFORM3 Herein, J is Jacobi matrix.

Such gradients are back-propagated [] through the t th base deep CNNs to fine-tune the weights {w lt } C l=1 and the set of model parameters θ t simultaneously.

Denote X as the instance space, denote Ω as the distribution over X , and denote S as a training set of N examples chosen i.i.d according to Ω. We are to investigate the gap between the generalization error on Ω and the empirical error on S.Suppose that F is the set from which the base deep experts are chosen, and let G = Note that g is a C-dim vetor, and each component of g is the category confidence, i.e., g y (x) = p(y|x), (y = 1, ..., C).

Based on Eq.(11), the category label of test sample can be predicted by arg max y g y (x) = p(y|x).

The ensembled classifier g predicts wrong if g y (x) ≤ maxȳ ̸ =y gȳ(x).

The generalization error rate for the final ensembled classifier can be measured by the probability DISPLAYFORM0 DISPLAYFORM1 According to probability theory, for any events B 1 and B 2 , P(B 1 ) ≤ P(B 2 ) + P(B 2 |B 1 ), therefore DISPLAYFORM2 where ξ > 0 measures the margin between the confidences from ground-truth and incorrect categories.

Using Chernoff bound BID16 , the the second term in the right side of Eq. FORMULA7 is bounded as: DISPLAYFORM3 Assume that the base-classifier space F is with VC-dimension d, which can be approximately estimated by the number of neurons ν and the number of weigths ω in the base deep network, i.e., d = O(νω).

Recall that S is a sample set of N examples from C categories.

Then the effective number of hypotheses for F over S is at most DISPLAYFORM4 Thus, the effective number of hypotheses over S forĜ = DISPLAYFORM5 Applying Devroye Lemma as in BID16 , it holds with probability at least 1 − δ Γ that DISPLAYFORM6 where DISPLAYFORM7 Likewise, in probability theory for any events B 1 and B 2 , P(B 1 ) ≤ P(B 2 ) + P(B 1 |B 2 ), thus DISPLAYFORM8 Because DISPLAYFORM9 So, combining Eq.(19−23) together, it can be derived that DISPLAYFORM10 As can be seen from above, large margin ξ over the training set corresponds to narrow gap between the generalization error on Ω and the empirical error on S, which leads to the better upper bound of generalization error.

In this section we evaluate the proposed algorithms on three real world datasets MNIST (LeCun, 1998), CIFAR-100 BID5 , and ImageNet (Russakovsky et al., 2015) .

For MNIST and CIFAR-100, we train all networks from scrach in each AdaBoost iteration stage.

On ImageNet, we use the pretrained model as the result of iteration #1 and then train weighted models sequentially.

The pretrained model is available in TorchVision 1 .

In each iteration, we adopt the weight initialization menthod proposed by BID2 .

All the networks are trained using stochastic gradient descent (SGD) with the weight decay 0f 10 −4 and the momentum of 0.9 in the experiments.

MNIST dataset consists of 60,000 training and 10,000 test handwritten digit samples.

BID18 showed the accuracy improvement of MLP via AdaBoost on MNIST dataset by updating sample weights according to classification errors.

For fair comparison, we firstly use the similar network architecture (MLP) as the base experts in experiments.

We train two sets of networks with the only difference that one updates weights w.r.t the class errors on training datasets while the other one updates weights w.r.t the sample errors on training datasets.

The former is the proposed method in this paper, and the latter is the traditional AdaBoost method.

In the two sets of weak learners, we share the same weak learner in iteration #1 and train other two weak learners seperately.

For data pre-processing, we normalize data via subtracting means and dividing standard deviations.

In the experiment on MNIST, we simply train the network with learning rate 0.01 through out the whole 120 epoches.

With our proposed method, the top 1 error on test datasets decreases from 4.73% to 1.87 % after three iterations (table 1) .

After the interation #1, the top 1 error of our method drops more quickly than the method which update weights w.r.t sample errors.

Our method, which updates weights w.r.t the class errors, leverages the idea that different class should have different learning comlexity and should not be treated equally.

Through the iterations, our method trains a set of classifiers in an easy-to-hard way.

Class APs vary from each weak learner in each iteration to others, increasing for marjor weighted classes while decreasing for minor weighted classes( FIG1 .

Therefore, in each iteration, the weighted learner classifier behaves like a expert different from the classfier in the previous iteration.

Though some APs for certain classes may decrease in some degree with each weak learner, the boosting models improve the accuracy for hard classes while preservering the accuracy for easy classes ( FIG1 .

Our method cordinates the set of weak learners trained sequeentially with diversified capabilities to improve the classfication capability of boosting model.

We also carry out experiments on CIFAR-100 dataset.

CIFAR-100 dataset consists of 60,000 images from 100 classes.

There are 500 training images and 100 testing images per class.

We adopt padding, mirroring, shifting for data augumentation and normalization as in BID3 BID4 .

In training stage, we hold out 5,000 training images for validation and leave 45,000 for training.

Because the error per class on training datasets approaches zero and training errors could be all zeros with even simple networks , we update the category distribution w.r.t the class errors on validation datasets.

We do not use any sample of validation datasets to update parameters of the networks itself.

When training networks on CIFAR-100, the initial learning rate is set to 0.1 and divided by 0.1 at epoch [150, 225] .

Similar to BID2 BID4 , we train the network for 300 epoches.

We show the results with various models including ResNet56(λ = 0.7) and DenseNet-BC(k=12) BID4 on test set.

The performances of emsembled classifier with different number of base networks are shown in the middle two rows of (table 2).As illustrated in section 3.3, λ controls the weight differences among classes.

In comparison, we use λ={0.7, 0.5, 0.1}. As shown in FIG2 -left, with smaller lambda, the weitht differences become bigger.

We use ResNet model in BID3 with 56 layers on CIFAR-100 datasets.

Overall, the models with lambda=0.7 performs the best, resulting in 24.15% test error after four iterations.

Comparing with lambda=0.5 and lambda=0.7, we find that both model performs well in the initial several iterations, but the model with lambda=0.7 would converge to a better optimal( FIG2 .

However, with lambda=0.1 which weights classes more discriminately, top 1 error fluctuates along the iterations ( FIG2 .

We conclude that lambda should be merely used to insure that the value of β is below 0.5 and may harm the performance in the ensemble models if set to a low value.

In FIG3 , we show the comparison of weak leaner #1 and weak learner #2 without boosting.

Though with minor exeptions, most classes with low APs improve their class APs in the proceeding weak learner.

Our method is based on the motivation that different classes should have different learning comlexity.

Thus, those classes with higher learning complexity should be paid more attention along the iterations.

Based on the class AP result of the privious iteration, we suppose those classes with lower APs should have higher learning complexity and be paid for attention in the subsequent iterations.

We furthermore carry out experiments on ILSVRC2012 Classification dataset (Russakovsky et al., 2015) which consists of 1.2 million images for training, and 50,000 for validation.

There are 1,000 classes in the dataset.

For data augmentation and normalization, we adopt scaling, ramdom cropping and horizontal flipping as in BID3 BID4 .

Similar to the experiments on CIFAR-100, the error per class on training datasets approaches zero, we update the category distribution w.r.t the class errors on validation datasets.

Since the test dataset of ImageNet are not available, we just report the results on the validation sets, following BID3 ; BID4 for ImageNet.

When we train ResNet50 networks on ImageNet, the initial learning rates are set to 0.1 and divided by 0.1 at epoch [30, 60] .

Similar to BID2 BID4 again, we train the network for 90 epoches.

The performances of emsembled classifier with different number of base networks are shown in the bottom rows of (table 2).

These base ResNet networks with diverse capabilities are combined to generate more discriminative ensemble classifier.

In this paper, we develop a deep boosting algorithm is to learn more discriminative ensemble classifier by combining a set of base experts with diverse capabilities.

The base experts are from the family of deep CNNs and they are sequentially trained to recognize a set of object classes in an easy-to-hard way according to their learning complexities.

As for the future network, we would like to investigate the performance of heterogeneous base deep networks from different families.

@highlight

 A deep boosting algorithm is developed to learn more discriminative ensemble classifier by seamlessly combining a set of base deep CNNs.