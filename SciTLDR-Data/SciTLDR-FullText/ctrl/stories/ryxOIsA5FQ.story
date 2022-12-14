In machine learning tasks, overtting frequently crops up when the number of samples of target domain is insufﬁcient, for the generalization ability of the classiﬁer is poor in this circumstance.

To solve this problem, transfer learning utilizes the knowledge of similar domains to improve the robustness of the learner.

The main idea of existing transfer learning algorithms is to reduce the dierence between domains by sample selection or domain adaptation.

However, no matter what transfer learning algorithm we use, the difference always exists and the hybrid training of source and target data leads to reducing ﬁtting capability of the learner on target domain.

Moreover, when the relatedness between domains is too low, negative transfer is more likely to occur.

To tackle the problem, we proposed a two-phase transfer learning architecture based on ensemble learning, which uses the existing transfer learning algorithms to train the weak learners in the ﬁrst stage, and uses the predictions of target data to train the ﬁnal learner in the second stage.

Under this architecture, the ﬁtting capability and generalization capability can be guaranteed at the same time.

We evaluated the proposed method on public datasets, which demonstrates the effectiveness and robustness of our proposed method.

Transfer learning has attracted more and more attention since it was first proposed in 1995 BID11 and is becoming an important field of machine learning.

The main purpose of transfer learning is to solve the problem that the same distributed data is hard to get in practical applications by using different distributed data of similar domains.

Several different kinds of transfer stratagies are proposed in recent years, transfer learning can be devided into 4 categories BID17 , including instance-based transfer learning, feature-based transfer learning, parameter-based transfer learning and relation-based transfer learning.

In this paper, we focus on how to enhance the performance of instance-based transfer learning and feature-based transfer learning when limited labeled data from target domain can be obtained.

In transfer learning tasks, when diff-distribution data is obtained to improve the generalization ability of learners, the fitting ability on target data set will be affected more or less, especially when the domains are not relative enough, negative transfer might occur BID11 , it's hard to trade off between generalization and fitting.

Most of the existing methods to prevent negative transfer learning are based on similarity measure(e.g., maximum mean distance(MMD), KL divergence), which is used for choosing useful knowledge on source domains.

However, similarity and transferability are not equivalent concepts.

To solve those problems, we proposed a novel transfer learning architecture to improve the fitting capability of final learner on target domain and the generalization capability is provided by weak learners.

As shown in FIG0 , to decrease the learning error on target training set when limited labeled data on target domain can be obtained, ensemble learning is introduced and the performances of transfer learning algorithms are significantly improved as a result.

In the first stage, traditional transfer learning algorithms are applied to diversify training data(e.g., Adaptive weight adjustment of boosting-based transfer learning or different parameter settings of domain adaptation).

Then diversified training data is fed to several weak classifiers to improve the generalization ability on target data.

To guarantee the fitting capability on target data, the predictions of target data is vectorized to be fed to the final estimator.

This architecture brings the following advantages:• When the similarity between domains is low, the final estimator can still achieve good performance on target training set.

Firstly, source data and target data are hybrid together to train the weak learners, then super learner is used to fit the predictions of target data.• Parameter setting is simplified and performance is better than individual estimators under normal conditions.

To test the effectiveness of the method, we respectively modified TrAdaboost BID1 and BDA BID16 as the base algorithms for data diversification and desired result is achieved.1.1 RELATED WORK

TrAdaboost proposed by BID1 is a typical instance-based transfer learning algorithm, which transfer knowledge by reweighting samples of target domain and source domain according to the training error.

In this method, source samples are used directly for hybrid training.

As the earliest boosting based transfer learning method, there are many inherent defects in TrAdaboost (e.g., high requirements for similarity between domains, negative transfer can easily happen).

Moreover, TrAdaboost is extended from Adaboost and use WMA(Weighted Majority Algorithm) BID6 to update the weights, and the source instance that is not correctly classified on a consistent basis would converge to zero by ⌈N/2⌉ and would not be used in the final classifier's output since that classifier only uses boosting iterations⌈N/2⌉ → N .

Two weakness caused by disregarding first half of ensembles analysised in BID0 are in the list below:• As the source weights convergence rapidly, after ⌈N/2⌉ iterations, the source weights will be too low to make full use of source knowledge.• Later classifiers merely focus on the harder instances.

To deal with rapid convergence of TrAdaboost, BID2 proposed TransferBoost, which apply a 2-phase training process at each iteration to test whether negative transfer has occurred and adjust the weights according to the results.

BID0 introduces an adaptive factor in weights update to slow down the convergence.

BID19 proposed multisource TrAdaboost, aimed at utilize instances from multiple source domains to improve the transfer performance.

In this paper, we still use the WMA to achieve data diversification in experiment of instance-based transfer learning, but stacking rather than boosting is used in final predictions.

Feature based transfer learning mainly realizes transfer learning by reducing the distribution difference(e.g., MMD) between source domain and target domain by feature mapping, which is the most studied method for transfer knowledge in recent years.

BID12 proposed transfer components analysis(TCA) as early as 2011, TCA achieve knowledge transfer by mapping the feature of source domain and target domain to a new feature space where the MMD between domains can be minimized.

However, not using labels brings a defect that only marginal distribution can be matched.

To address the problem, BID8 proposed joint distribution adaptation(JDA) which fit the marginal distribution and conditional distribution at the same time, for unlabeled target data, it utilizes pseudo-labels provided by classifier trained on source data.

After that, BID16 extended JDA for imbalanced data.

In neural networks, it's easy to transfer knowledge by pre-train and fine-tune because feature extracted by lower layers are mostly common for different tasks BID20 , to transfer knowledge in higher layers which extract task-specific features, BID15 , BID9 and BID10 add MMD to the optimization target in higher layers.

The learning performance in the target domain using the source data could be poorer than that without using the source data, This phenomenon is called negative transfer BID11 .

To avoid negative transfer, BID7 point out that one of the most important research issues in transfer learning is to determine whether a given source domain is effective in transferring knowledge to a target domain, and then to determine how much of the knowledge should be transferred from a source domain to a target domain.

Researchers have proposed to evaluate the relatedness between the source and target domains.

When limited labeled target data can be obtained, two of the methods are listed below:• Introduce the predened parameters to qualify the relevance between the source and target domains.

However, it is very labor consuming and time costing to manually select their proper values.• Examine the relatedness between domains directly to guide transfer learning.

The notion of positive transferability was first introduced in BID13 for the assessment of synergy between the source and target domains in their prediction models, and a criterion to measure the positive transferability between sample pairs of different domains in terms of their prediction distributions is proposed in that research.

BID14 proposed a kernel method to evaluate the task relatedness and the instance similarities to avoid negative transfer BID4 proposed a method to detection the occurance of negative transfer which can also delay the point of negative transfer in the process of transfer learning.

BID3 remind that most previous work treats knowledge from every source domain as a valuable contribution to the task on the target domain could increase the risk of negative transfer.

A two-phase multiple source transfer framework is proposed, which can effectively downgrade the contributions of irrelevant source domains and properly evaluate the importance of source domains even when the class distributions are imbalanced.

Stacking is one of the ensemble learning methods that fuses multiple weak classifiers to get a better performance than any single one BID5 .

When using stacking, diversification of weak learners has an important impact on the performance of ensemble BID18 .

Here are some common ways to achieve diversification:• Diversifying input data: using different subset of samples or features.• Diversifying outputs: classifiers are only for certain categories.• Diversifying models: using different classification models.

In this paper, we can also regard the proposed architecture as a stacking model which uses transfer learning algorithms to achieve input diversification.

In this section, we introduce how the instance-based transfer learning is applied to the proposed architecture.

we use TrAdaboost as an example and make a simple modification to turn it to stacking.

In TrAdaboost, we need a large number of labeled data on source domain and limited labeled data on the target domain.

We use X = X S ∪ X T to represent the feature space.

Source space(X S ) and target space(X T ) are defined as DISPLAYFORM0 respectively.

Then the hybrid training data set is defined in equation 1.

DISPLAYFORM1 Weight vector is initialized firstly by w 1 = (w DISPLAYFORM2 In t th iteration, the weights are updated by Equation 3.

DISPLAYFORM3 Here, β t = ϵ t 1−ϵt and β = 1/(1+ √ 2 ln n/N ).

It is noteworthy that original TrAdaboost is for binary classification.

In order to facilitate experimental analysis and comparison, we extend the traditional TrAdaboost to a multi-classification algorithm according to Multi-class Adaboost proposed in BID21 , then β t and beta defined as: DISPLAYFORM4 K is the class number.

Equation 5 defines the final output for each class.

DISPLAYFORM5 Moreover, for single-label problem, we use softmax to transfer P k (x) to probabilities.

To address the rapid convergence problem, BID2 proposed TransferBoost, which utilizes all the weak classifiers for ensemble, but in the experiment, early stop can improve the final performance, so which classifiers should be chosen is still a problem.

BID0 proposed dynamic TrAdaboost, In this algorithm, an adaptive factor is introduced to limit the convergence rate of source sample weight.

However, it's not always effective in practical use.

Theoretical upper bound of training error in target domain is not changed in dynamic TrAdaboost, which is related to the training error on target domain, we have: DISPLAYFORM6 Algorithm 1 stacked generalization for instance-based transfer learning.

Call Learner, providing it labeled target data set with the distribution w t .

Then get back a hypothesis of S. DISPLAYFORM7

Calculate the error on S:6: DISPLAYFORM0 Get a subset of source task DISPLAYFORM1 Call learner, providing it S t ∪ T .

Then get back a hypothesis of T.

Calculate the error on T using equation.

2.10:updata β t using equation.

4.

Update the new weight vector using equation.

3. 12: end for 13: Construct probability vectors by concatenating DISPLAYFORM0 Although dynamic TrAdaboost can improve the weights of source samples after iteration ⌈N/2⌉ and the generalization capability is improved, it's very likely that the error rate on source domain ϵ t increases, sometimes it even aggravates the occurrence of negative transfer when the domains are not similar enough.

We use stacking to address the problems above in this section, in the data diversification stage, TrAdaboost is used to reweight samples for each weak classifier.

Meanwhile, because we make use of all the weak classifiers, to avoid the high source weights of irrelative source samples negatively effects on the task in early iterations, a two-phase sampling mechanism is introduced in our algorithm.

A formal description of stacking for instance-based transfer learning is given in Algorithm 1.

The main difference between stacking for instance-based transfer learning and TrAdaboost are listed as follows:• A two-phase sampling process and an extra parameter λ is introduced.

Firstly, target data is fed to weak learner and the weighted error rate of source samples are used to decide which samples can be used for hybrid learning by comparing with the threshold λ.

As the source weights reduces with the number of iterations increasing, more and more source samples will be utilized.• Stacking rather than TrAdaboost is used to get the final output.

We construct a feature matrix by the outputs of weak learners on target data, then use a super learner(e.g., LogitRegression in our experiment) to fit the labels.

In this way, training error on target set can be minimized.

When compared with TrAdaboost, stacking is insensitive to the performance of each weak classifier because the training error on target data can be minimized in stacking, which means it's more robust in most cases and brings some benefits:• When using stacking, all of the weak classifiers could be used.• When source domain is not related enough, stacking performs better.

One of a popular methods for feature-based transfer learning to achieve knowledge transfer is domain adaptation, which minimizes the distribution difference between domains by mapping features to a new space, where we could measure the distribution difference by MMD.

Generally speaking, we use P (X S ), P (X T ) and P (Y S |X S ), P (Y T |X T ) to represent the marginal distribution and conditional distribution of source domain and target domain respectively.

In Pan et al. FORMULA2 , transfer component analysis(TCA) was proposed to find a mapping which makes P (ϕ(X S )) ≈ P (ϕ(X T )), the MMD between domains in TCA is defined as: DISPLAYFORM0 Long et al. FORMULA2 proposed joint distibution adaptation(JDA) to minimize the differences of marginal distribution and conditinal distribution at the same time, the MMD of conditional distribution is defined as: DISPLAYFORM1 In BID16 , balanced distribution adaptation was proposed, in which algorithm, an extra parameter is introduced to adjust the importance of the distributions, the optimization target is defined by Equation FORMULA13 : DISPLAYFORM2 To solve the nonlinear problem, we could use a kernel matrix defined by: DISPLAYFORM3 , then the optimization proplem can be formalized as: DISPLAYFORM4 Where H is a centering matrix, M 0 and M c represent the MMD matrix of marginal distribution and conditional distribution respectively, A is the mapping matrix.

In domain adaptation, performace is sensitive to the selection of parameters(e.g., kernel type, kernel param or regularization param).

For instance, if we use rbf kernel, as defined in Equation 11, to construct the kernel matrix.

selection of kernel param σ has an influence on the mapping.

In this paper, we use BDA as a base algorithm in the proposed architecture to achieve data diversification by using different kernel types and parameter settings.

DISPLAYFORM5 By taking adavantage of stacking, we could get a better transfer performance than any single algorithm.

Here, we introduce how we choose the kernel parameter in our experiments.

In ensemble learning, it's significant to use unrelated weak classifiers for a better performance Woniak et al.(2014)(i.e.

, learners should have different kinds of classification capabilities).

moreover, performances of learners shouldn't be too different or the poor learners will have an negative effect on ensemble.

In another word, we choose the kernel parameter in a largest range where the performance is acceptable.

We take the following steps to select parameters.

Firstly, search a best value of kernel parameter σ for weak classifier, where the accuracy on validation set is Accuracy max .

secondly, set a threshold parameter λ and find an interval (σ min , σ max ) around σ where the accuracy on validation set satisfy Accuracy max − λ ≤ Accuracy when σ ∈ (σ min , σ max ).

Finally, select N parameters in (σ min , σ max ) by uniformly-spaced sampling.

When multiple type of kernels are utilized, we choose parameter sets for each seperately by repeating the above steps.

In our method, the settings of λ and N should be taken into consideration, if λ is too large, the performance of each learner can't be guaranteed, if λ is too small, training data can't be diversified enough.

Set N to a large number would help to get a better performance in most cases, while the complexity could be high.

Algorithm 2 stacked generalization for feature-based transfer learning.

DISPLAYFORM6 Contruct kernel matrix K t using κ t .

Solve the eigendecomposition problem and use d smallest eigenvectors to build A t .

Train t th leaner on {A DISPLAYFORM0 Call learner t providing DISPLAYFORM1 Construct probability vectors by concatenating DISPLAYFORM2 , where K is the number of classes.

10: end for 11: Construct target feature matrix DISPLAYFORM3 Algorithm 2 presents the detail of our method.

In our algorithm, kernel function κ t can be differentiated by kernel types or kernel params.

In t th iteration, we choose κ t to mapping the feature space, then get the matrix A by BDA, A ⊤ K tar and A ⊤ K src are feed to weak learner.

After that, we concatenate predictions of A ⊤ K tar as features of the super leaner.

In this paper, we assume that there are limited labeled data on target set, so we use modified BDA, which uses real label rather than pseudo label, to adapt conditional distribution.

To evaluate the effectiveness of our method, 6 public datasets and 11 domains as shown in TAB3 are used in our experiment of instance-based transfer learning and feature-based transfer learning.

The detail is described as follow: Figure 3 : Accuracy of instance-based transfer on mnist vs. usps changed with the iterations and ratio of #target.

Figure 3 shows the experiment results of transfer between mnist and USPS under different iterations and ratios of #source and #target.

We observe that stacking achieves much better performance than TrAdaboost.

Firstly, acuuracy of stacking method is higher when ratio changing from 1% to 10%.

Especially, the fewer the labeled target samples are, the more improvement stacking method could achieve.

Secondly, few iterations are required for stacking method to achieve a relatively good performance, when the curve is close to convergence, there's still about 5% improvement compared with TrAdaboost.

Moreover, in both transfer tasks USPS→ mnist and mnist → USPS, stacking method performs significantly better than TrAdaboost.

The reason why stacking performs better is analyzed in section 3.1, we assume that the introduction of source leads to under fitting on target data when hybrid training, to confirm our hypothesis, we made a comparision of training error on source data and target data between TrAdaboost and stacking method, FIG2 shows the result of four of the transfer tasks.

TAB4 shows the results of all the transfer tasks under 20 iterations.

BDA is chosen as the base algorithm to achieve data diversification in our experiment, we mainly test the infulence of different kernel functions has on the perfomance and the effectiveness of the method proposed in section 3.2.

The ratio of #source and #target is set to 5% and we use rbf kernel, poly kernel and sam kernel to conduct our experiment, for the sake of simplicity, kernel function is defined in TAB5 , where γ is variable for different weak learners.

DISPLAYFORM0 To observe how the selection of kernels affects the feature distribution, we visualize the feature representations learned by BDA under different kernel parameters and kernel types when adapting domains A and C. As shown in FIG5 (a), data distribution and similarity between domains change with the kernel parameters, when compared with FIG5 (b), which presents feature distribution of rbf kernel, it's obvious that using different kernel types can provide more diversity.

In this paper, to construct the kernel set by sampling parameters in a range where the performance is not too worse than the best one, we followed the method given in section 3.2 and set threshold λ varies from 5% to 10% for different tasks.

For each kernel type, we select 10 different parameters(i.e., 10 weak classifiers) for stacking.

Tabel 4 shows the comparison between single algorithm and ensemble learning.

For each kernel type, we give the best accuracy, average accuracy of weak learners and accuracy of ensemble learning, Randomforest and LogitRegression as the weak learner and super learner respectively.

Accuracy of integrating all the kernel types is shown in the last column and the best performance of each task is in bold.

We can learn from the table that ensemble learning outperfoms the best single learner in all the tasks, and in most cases, using both of rbf and another type of kernels are able to improve the performance.

However, when should we use multiple kernel types in stacking needs to be further studied.

In summary, the reason why the proposed method can improve the performance of feature-based transfer learning is listed as follows: Firstly, we use super learner to fit the target domain, so the bias of weak learners introduced by hybrid training with source data is reduced.

Secondly, multiple kernels are utilized to achieve data diversification, so we could integrate the classification ability of weak learners trained on diff-distribution data.

In this paper, we proposed a 2-phase transfer learning architecture, which uses the traditional transfer learning algorithm to achieve data diversification in the first stage and the target data is fitted in the second stage by stacking method, so the generalization ability and fitting ability on target data could be satisfied at the same time.

The experiment of instance-based transfer learning and feature-based transfer learning on 11 domains proves the validity of our method.

In summary, this framework has the following advantages:• No matter if source domain and target domain are similar, the training error on target data set can be minimized theoretically.• We reduce the risk of negative transfer in a simple and effective way without a similarity measure.• Introduction of ensemble learning gives a better performance than any single learner.• Most existing transfer learning algorithm can be integrated into this framework.

Moreover, there're still some problems require our further study, some other data diversification method for transfer learning might be useful in our model, such as changing the parameter µ in BDA, integrating multiple kinds of transfer learning algorithms, or even applying this framework for multi-source transfer learning.

<|TLDR|>

@highlight

How to use stacked generalization to improve the performance of existing transfer learning algorithms when limited labeled data is available.