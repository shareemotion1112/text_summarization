Learning domain-invariant representation is a dominant approach for domain generalization.

However, previous methods based on domain invariance overlooked the underlying dependency of classes on domains, which is responsible for the trade-off between classification accuracy and the invariance.

This study proposes a novel method {\em adversarial feature learning under accuracy constraint (AFLAC)}, which maximizes domain invariance within a range that does not interfere with accuracy.

Empirical validations show that the performance of AFLAC is superior to that of baseline methods, supporting the importance of considering the dependency and the efficacy of the proposed method to overcome the problem.

In supervised learning we typically assume that samples are obtained from the same distribution in training and testing; however, because this assumption does not hold in many practical situations it reduces the classification accuracy for the test data BID20 .

One typical situation is domain generalization (DG) BID1 BID18 BID19 BID2 : we have labeled data from several source domains and collectively exploit them such that the trained system generalizes to other unseen, but somewhat similar, target domain(s).

This paper considers DG under the situation where domain d and class y labels are statistically dependent owing to some common latent factor z FIG0 -(c)), which we referred to as domainclass dependency.

For example, the WISDM Activity Prediction dataset (WISDM, BID10 ), where y and d correspond to activities and wearable device users, exhibits this dependency because (1) some activities (e.g., jogging) are strenuous to the extent that some unathletic subjects avoided them (data characteristics), or (2) other activities were added only after the study began and the initial subjects could not perform them (data-collection errors).

The dependency is common in real-world datasets BID23 and a similar setting has been investigated in domain adaptation (DA) studies, but most prior DG studies overlooked the dependency.

Most prior DG methods utilize invariant feature learning (IFL) (e.g., ).

IFL attempts to learn feature representation h from input data x which is invariant to d. When source and target domains have some common structure (see, ), IFL prevents the classifier from overfitting to source domains FIG0 ).

However, under the dependency, merely imposing the domain invariance can adversely affect the classification accuracy as pointed out by BID21 and illustrated in FIG0 .

Although that trade-off occurs in source domains (because DG uses only source data during optimization), it can also negatively affect the classification performance for target domain(s).

For example, if the target domain has characteristics similar (or same as an extreme case) to those of a certain source domain, giving priority to domain invariance obviously interferes with the DG performance ( FIG0 ).In this paper, considering that prioritizing domain invariance under the trade-off can negatively affect the DG performance, we propose a novel method adversarial feature learning under accuracy constraint (AFLAC), which maximizes domain invariance within a range that does not interfere with the classification accuracy FIG0 -(e)) on adversarial training.

Specifically, AFLAC is intended to achieve accuracy-constrained domain invariance, which we define as the maximum H(d|h) (H denotes entropy) value under the condition H(y|x) = H(y|h) (h has as much y information as x).

Empirical validations show that the performance of AFLAC is superior to that of baseline methods, supporting the importance of considering domain-class dependency and the efficacy of the proposed approach for overcoming the issue.

DG has been attracting considerable attention in recent years, and most prior DG methods utilize IFL BID2 BID4 .

In particular, our proposed method is based on Domain Adversarial Nets (DAN), which was originally invented for DA BID3 and BID21 demonstrated its efficacy in DG.

In addition, BID21 intuitively explained the trade-off between classification accuracy and domain invariance, but they did not suggest any solution to the problem except for carefully tuning a weighting parameter.

Several studies that address DG without utilizing IFL have been conducted.

For example, CCSA BID16 , CIDG BID13 , and CIDDG BID14 proposed to make use of semantic alignment, which attempts to make latent representation given class label (p(h|y)) identical within source domains.

This approach was originally proposed by BID6 in the DA context, but its efficacy to overcome the trade-off problem is not obvious.

CrossGrad BID18 , which is one of the recent state-of-the-art DG methods, utilizes data augmentation with adversarial examples.

However, because the method relies on the assumption that y and d are independent, it might not be directly applicable to our setting.

In DA, BID23 ; BID6 address the situation where p(y) changes across the source and target domains by correcting the change of p(y) using unlabeled target data, which is often accomplished at the cost of classification accuracy for the source domain.

However, this approach is not applicable (or necessary) to DG because we are agnostic on target domain(s), and this paper is concerned with the change of p(y) within source domains.

Instead, we propose to maximize the classification accuracy for source domains while improving the domain invariance.

Here we provide the notion of accuracy-constrained domain invariance, which is the maximum domain invariance within a range that does not interfere with the classification accuracy.

The reason for the constraint is that the primary purpose of DG is the classification for unseen domains rather than domain itself, and the improvement of the invariance could detrimentally affect the performance.

Theorem 1 Let h = f (x), i.e., h is a deterministic mapping of x with a function f .

We define accuracy-constrained domain invariance as the maximum H(d|h) value under the constraint that DISPLAYFORM0

Proof 1 Using the properties of entropy, the following inequation holds: DISPLAYFORM0 By assumption, H(y|x) = H(y|h) = 0, and thus the following inequation holds: DISPLAYFORM1 Thus, the maximum H(d|h) value under the constraints is H(d|y).

We propose a novel method named AFLAC, which is designed to achieve accuracy-constraind domain invariance.

Formally, we denote f E (x), q M (y|h), and q D (d|h) (E, M , and D are the parameters) as the deterministic encoder, probabilistic model of the label classifier, and that of domain discriminator, respectively.

Then, the objective function of AFLAC is described as follows: DISPLAYFORM0 Here ?? denotes a weighting parameter.

Note that, although we cannot obtain true distribution p(d|y), we can use the maximum likelihood estimator of it when y and d are discrete, as is usual with DG.Here we formally show that AFLAC is intended to achieve H(d|h) = H(d|y) (accuracy-constrained domain invariance) by a Nash equilibrium analysis similar to BID7 ; BID21 .

We define D * and M * as the solutions to Eq. 3 and Eq. 4 with fixed E. They obviously satisfy q * D = p(d|h), q * M = p(y|h), respectively.

Thus, V in Eq. 3 can be written as follows: DISPLAYFORM1 E * , which we define as the solution to Eq. 5 and in Nash equilibrium, satisfies not only H(y|h) = H(y|x) (optimal classification accuracy) but also E h,y???p (h,y) [D KL [p(d|y)

BMNISTR We created the Biased Rotated MNIST dataset (BMNISTR) by modifying the sample size of the popular benchmark dataset MNISTR BID5 , such that the class distribution differed among the domains.

In MNISTR, each domain was created by rotating images by 15 degree increments: 0, 15, ..., 75 (referred to as M0, ..., M75).

We created four variants of MNISTR that have different types of domain-class dependency, referred to as BMNISTR-1 through BMNISTR-3.

As shown in TAB0 -left, BMNISTR-1, -2 have similar trends but different degrees of dependency, whereas BMNISTR-1 and BMNISTR-3 differ in terms of their trends.

In training, we employed a leave-one-domain-out setting BID5 : we trained the models on five of the six domains and tested them using the remaining one.

WISDM WISDM contains sensor data of accelerometers of six human activities (walking, jogging, upstairs, downstairs, sitting, and standing) performed by 36 users (domains).

WISDM has the dependency for the reason noted in Sec. 1.

we randomly selected <10 / 26> and <26 / 10> users as <source / target> domains, and split the source data into training and validation data.

We compared AFLAC with the following methods.

(1) CNN is a vanilla convolutional networks trained on the aggregation of data from all source domains.

(2) DAN BID21 ) is expected to generalize across domains utilizing domain-invariant representation, but it can be affected by the trade-off as pointed out by BID21 .

(3) CIDDG is our re-implementation of BID14 , which is designed to achieve semantic alignment on adversarial training.

Additionally, we used (4) AFLAC-Abl, which is a version of AFLAC modified for ablation studies.

AFLAC- DISPLAYFORM0

We first investigated the extent to which domain-class dependency affects the performance of domain-invariance-based methods.

In TAB0 -right, we compared the mean F-measures for the classes 0 through 4 and classes 5 through 9 in BMNISTR with the target domain M0.

Recall that the sample sizes for the classes 0???4 are variable across domains, whereas the classes 5???9 have identical sample sizes across domains.

The F-measures show that AFLAC outperformed baselines in most dataset-class pairs, which supports that the dependency reduces the performance of IFL methods and that AFLAC can mitigate the problem.

Further, the relative improvement of AFLAC to AFLAC-Abl is more significant for the classes 0???4 than for 5???9 in BMNISTR-1 and BMNISTR-3, suggesting that AFLAC tends to increase performance more significantly for classes in which the dependency occurs.

Moreover, the improvement is more significant in BMNISTR-1 than in BMNISTR-2, suggesting that the stronger the dependency is, the lower the performance of domain-invariance-based methods becomes.

Finally, although the dependencies of BMNISTR-1 and BMNISTR-3 have different trends, AFLAC improved the F-measures in both datasets.

Next we investigated the relationship between the strength of regularization and performance.

c, d) show that the accuracy gaps of AFLAC-Abl and AFLAC increase with strong regularization (such as when ?? = 10), suggesting that AFLAC, as it was designed, does not tend to reduce accuracy with strong regularizer, and thus AFLAC is robust toward hyperparameter choice.

In this paper, we proposed a novel method AFLAC, which maximizes domain invariance within a range that does not interfere with classification accuracy on adversarial training.

Empirical validations show the superior DG performance of AFLAC to the baseline methods, supporting the importance of the domain-class dependency in domain generalization tasks and the efficacy of the proposed method for overcoming the issue.

<|TLDR|>

@highlight

Address the trade-off caused by the dependency of classes on domains by improving domain adversarial nets