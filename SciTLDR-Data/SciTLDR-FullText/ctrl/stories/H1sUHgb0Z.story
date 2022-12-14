Supervised learning depends on annotated examples, which are taken to be the ground truth.

But these labels often come from noisy crowdsourcing platforms, like Amazon Mechanical Turk.

Practitioners typically collect multiple labels per example and aggregate the results to mitigate noise (the classic crowdsourcing problem).

Given a fixed annotation budget and unlimited unlabeled data, redundant annotation comes at the expense of fewer labeled examples.

This raises two fundamental questions: (1) How can we best learn from noisy workers?

(2) How should we allocate our labeling budget to maximize the performance of a classifier?

We propose a new algorithm for jointly modeling labels and worker quality from noisy crowd-sourced data.

The alternating minimization proceeds in rounds, estimating worker quality from disagreement with the current model and then updating the model by optimizing a loss function that accounts for the current estimate of worker quality.

Unlike previous approaches, even with only one annotation per example, our algorithm can estimate worker quality.

We establish a generalization error bound for models learned with our algorithm and establish theoretically that it's better to label many examples once (vs less multiply) when worker quality exceeds a threshold.

Experiments conducted on both ImageNet (with simulated noisy workers) and MS-COCO (using the real crowdsourced labels) confirm our algorithm's benefits.

Recent advances in supervised learning owe, in part, to the availability of large annotated datasets.

For instance, the performance of modern image classifiers saturates only with millions of labeled examples.

This poses an economic problem: Assembling such datasets typically requires the labor of human annotators.

If we confined the labor pool to experts, this work might be prohibitively expensive.

Therefore, most practitioners turn to crowdsourcing platforms such as Amazon Mechanical Turk (AMT), which connect employers with low-skilled workers who perform simple tasks, such as classifying images, at low cost.

Compared to experts, crowd-workers provide noisier annotations, possibly owing to high variation in worker skill; and a per-answer compensation structure that encourages rapid answers, even at the expense of accuracy.

To address variation in worker skill, practitioners typically collect multiple independent labels for each training example from different workers.

In practice, these labels are often aggregated by applying a simple majority vote.

Academics have proposed many efficient algorithms for estimating the ground truth from noisy annotations.

Research addressing the crowd-sourcing problem goes back to the early 1970s.

BID4 proposed a probabilistic model to jointly estimate worker skills and ground truth labels and used expectation maximization (EM) to estimate the parameters.

BID27 ; ; BID29 proposed generalizations of the Dawid-Skene model, e.g. by estimating the difficulty of each example.

Although the downstream goal of many crowdsourcing projects is to train supervised learning models, research in the two disciplines tends to proceed in isolation.

Crowdsourcing research seldom accounts for the downstream utility of the produced annotations as training data in machine learning (ML) algorithms.

And ML research seldom exploits the noisy labels collected from multiple human workers.

A few recent papers use the original noisy labels and the corresponding worker identities together with the predictions of a supervised learning model trained on those same labels, to estimate the ground truth BID2 BID7 .

However, these papers do not realize the full potential of combining modeling and crowd-sourcing.

In particular, they are unable to estimate worker qualities when there is only one label per training example.

This paper presents a new supervised learning algorithm that alternately models the labels and worker quality.

The EM algorithm bootstraps itself in the following way: Given a trained model, the algorithm estimates worker qualities using the disagreement between workers and the current predictions of the learning algorithm.

Given estimated worker qualities, our algorithm optimizes a suitably modified loss function.

We show that accurate estimates of worker quality can be obtained even when only collecting one label per example provided that each worker labels sufficiently many examples.

An accurate estimate of the worker qualities leads to learning a better model.

This addresses a shortcoming of the prior work and overcomes a significant hurdle to achieving practical crowdsourcing without redundancy.

We give theoretical guarantees on the performance of our algorithm.

We analyze the two alternating steps: (a) estimating worker qualities from disagreement with the model, (b) learning a model by optimizing the modified loss function.

We obtain a bound on the accuracy of the estimated worker qualities and the generalization error of the model.

Through the generalization error bound, we establish that it is better to label many examples once than to label less examples multiply when worker quality is above a threshold.

Empirically, we verify our approach on several multi-class classification datasets: ImageNet and CIFAR10 (with simulated noisy workers), and MS-COCO (using the real noisy annotator labels).

Our experiments validate that when the cost of obtaining unlabeled examples is negligible and the total annotation budget is fixed, it is best to collect a single label per training example for as many examples as possible.

We emphasize that although this paper applies our approach to classification problems, the main ideas of the algorithm can be extended to other tasks in supervised learning.

The traditional crowdsourcing problem addresses the challenge of aggregating multiple noisy labels.

A naive approach is to aggregate the labels based on majority voting.

More sophisticated agreementbased algorithms jointly model worker skills and ground truth labels, estimating both using EM or similar techniques BID4 BID9 BID27 BID30 BID19 BID3 BID19 .

BID28 shows that the EM algorithm with spectral initialization achieves minimax optimal performance under the Dawid-Skene model.

BID12 introduces a message-passing algorithm for estimating binary labels under the Dawid-Skene model, showing that it performs strictly better than majority voting when the number of labels per example exceeds some threshold.

Similar observations are made by BID0 .

A primary criticism of EM-based approaches is that in practice, it's rare to collect more than 3 to 5 labels per example; and with so little redundancy, the small gains achieved by EM over majority voting are not compelling to practitioners.

In contrast, our algorithm performs well in the low-redundancy setting.

Even with just one label per example, we can accurately estimate worker quality.

Several prior crowdsourcing papers incorporate the predictions of a supervised learning model, together with the noisy labels, to estimate the ground truth labels.

consider binary classification and frames the problem as a generative Bayesian model on the features of the examples and the labels.

BID2 consider a generalization of the Dawid-Skene model and estimate its parameters using supervised learning in the loop.

In particular, they consider a joint probability over observed image features, ground truth labels, and the worker labels and compute the maximum likelihood estimate of the true labels using alternating minimization.

We also consider a joint probability model but it is significantly different from theirs as we assume that the optimal labeling function gives the ground truth labels.

We maximize the joint likelihood using a variation of expectation maximization to learn the optimal labeling function and the true labels.

Further, they train the supervised learning model using the intermediate predictions of the labels whereas we train the model by minimizing a weighted loss function where the weights are the intermediate posterior probability distribution of the labels.

Moreover, with only one label per example, their algorithm fails and estimates all the workers to be equally good.

They only consider binary classification, whereas we verify our algorithm on multi-class (ten classes) classification problem.

A rich body of work addresses human-in-loop annotation for computer vision tasks.

However, these works assume that humans are experts, i.e., that they give noiseless annotations BID6 BID24 .

We assume workers are unreliable and have varying skills.

A recent work by BID21 also proposes to use predictions of a supervised learning model to estimate the ground truth.

However, their algorithm is significantly different than ours as it does not use iterative estimation technique, and their approach of incorporating worker quality parameters in the supervised learning model is different.

Their theoretical results are limited to the linear classifiers.

Another line of work employs active learning, iteratively filtering out examples for which aggregated labels have high confidence and collect additional labels for the remaining examples BID27 BID13 .

The underlying modeling assumption in these papers is that the questions have varying levels of difficulty.

At each iteration, these approaches employ an EM-based algorithm to estimate the ground truth label of the remaining unclassified examples.

For simplicity, our paper does not address example difficulties, but we could easily extend our model and algorithm to accommodate this complexity.

Several papers analyze whether repeated labeling is useful.

BID22 analyzed the effect of repeated labeling and showed that it depends upon the relative cost of getting an unlabeled example and the cost of labeling.

BID8 shows that if worker quality is below a threshold then repeated labeling is useful, otherwise not.

BID16 argues that it also depends upon expressiveness of the classifier in addition to the factors considered by others.

However, these works do not exploit predictions of the supervised learning algorithm to estimate the ground truth labels, and hence their findings do not extend to our methodology.

Another body of work that is relevant to our problem is learning with noisy labels where usual assumption is that all the labels are generated through the same noisy rate given their ground truth label.

Recently BID20 proposed a generic unbiased loss function for binary classification with noisy labels.

They employed a modified loss function that can be expressed as a weighted sum of the original loss function, and gave theoretical bounds on the performance.

However, their weights become unstably large when the noise rate is large, and hence the weights need to be tuned.

BID23 BID10 learns noise rate as parameters of the model.

A recent work by BID7 trains an individual softmax layer for each expert and then predicts their weighted sum where weights are also learned by the model.

It is not scalable to crowdsourcing scenario where there are thousands of workers.

There are works that aim to create noise-robust models BID11 BID14 , but they are not relevant to our work.

Let D be the underlying true distribution generating pairs (X, Y ) ??? X ?? K from which n i.i.d.

DISPLAYFORM0 where K denotes the set of possible labels K := {1, 2, ?? ?? ?? , K}, and X ??? R d denotes the set of euclidean features.

We denote the marginal distribution of Y by {q 1 , q 2 , ?? ?? ?? , q K }, which is unknown to us.

Consider a pool of m workers indexed by 1, 2, ?? ?? ?? , m. We use [m] to denote the set {1, 2, ?? ?? ?? , m}. For each i-th sample X i , r DISPLAYFORM1 r are selected randomly, independent of the sample X i .

Each selected worker provides a noisy label Z ij for the sample X i , where the distribution of Z ij depends on the selected worker and the true label Y i .

We call r the redundancy and, for simplicity, assume it to be the same for each sample.

However, our algorithm can also be applied when redundancy varies across the samples.

We use Z (r) i to denote {Z ij } j??? [r] , the set of r labels collected on the i-th example, and w DISPLAYFORM2 Following BID4 , we assume the probability that the a-th worker labels an item in class k ??? K as class s ??? K is independent of any particular chosen item, that is, it is a constant over i ??? [n].

Let us denote this constant by ?? ks ; by definition, s???K ?? ks = 1 for all k ??? K, and we call ?? (a) ??? [0, 1]

K??K the confusion matrix of the a-th worker.

In particular, the distribution of Z is: DISPLAYFORM3 ks .

( 1) The diagonal entries of the confusion matrix correspond to the probabilities of correctly labeling an example.

The off-diagonal entries represent the probability of mislabeling.

We use ?? to denote the collection of confusion matrices {?? (a) } a??? [m] .We assume nr workers w 1,1 , w 1,2 , ?? ?? ?? , w n,r are selected uniformly at random from a pool of m workers with replacement and a batch of r workers are assigned to each of the examples X 1 , X 2 , ?? ?? ?? , X n .

The corrupted labels along with the worker information DISPLAYFORM4 n ) are what the learning algorithm sees.

Let F be the hypothesis class, and f ??? F, f : DISPLAYFORM5 Given the observed samples (X 1 , Z1 , w DISPLAYFORM6 n ), we want to learn a good predictor function f ??? F such that its risk under the true distribution D, R ,D ( f ) is minimal.

Having access to only noisy labels Z (r) by workers w (r) , we compute f as the one which minimizes a suitably modified loss function ??, q (f (X), Z (r) , w (r) ).

Where ?? denote an estimate of confusion matrix ??, and q an estimate of q, the prior distribution on Y .

We define ??, q in the following section.

Assume that there exists a function f DISPLAYFORM0 .

Under the Dawid-Skene model (described in previous section), the joint likelihood of true labeling function f * (X i ) and observed labels {Z ij } i???[n],j??? [r] as a function of confusion matrices of workers ?? can be written as DISPLAYFORM1 q k 's are the marginal distribution of the true labels Y i '

s. We estimate the worker confusion matrices ?? and the true labeling function f * by maximizing the likelihood function L(??; f * (X), Z).

Observe that the likelihood function L(??; f * (X), Z) is different than the standard likelihood function of Dawid-Skene model in that we replace each true hidden labels Y i by f * (X i ).

Like the EM algorithm introduced in BID4 , we propose 'Model Bootstrapped EM' (MBEM) to estimate confusion matrices ?? and the true labeling function f * .

EM converges to the true confusion matrices and the true labels given an appropriate spectral initialization of worker confusion matrices BID28 .

We show in Section 4.4 that MBEM converges under mild conditions when the worker quality is above a threshold and the number of training examples is sufficiently large.

In the following two subsections, we motivate and explain our iterative algorithm to estimate the true labeling function f * given a good estimate of worker confusion matrices ?? and vice-versa.

To begin, we ask, what is the optimal approach to learn the predictor function f when for each worker we have ??, a good estimation of the true confusion matrix ??, and q, an estimate of the prior?

A recent paper, BID20 proposes minimizing an unbiased loss function specifically, a weighted sum of the original loss over each possible ground truth label.

They provide weights for binary classification where each example is labeled by only one worker.

Consider a worker with confusion matrix ??, where ?? y > 1/2 and ?? ???y > 1/2 represent her probability of correctly labeling the examples belonging to class y and ???y respectively.

Then their weights are ?? ???y /(?? y + ?? ???y ??? 1) for class y and ???(1 ??? ?? y )/(?? y + ?? ???y ??? 1) for class ???y.

It is evident that their weights become unstably large when the probabilities of correct classification ?? y and ?? ???y are close to 1/2, limiting the method's usefulness in practice.

As explained below, for the same scenario, our weights would be ?? y /(1 + ?? y ??? ?? ???y ) for class y and (1 ??? ?? ???y )/(1 + ?? y ??? ?? ???y ) for class ???y.

Inspired by their idea, we propose weighing the loss function according to the posterior distribution of the true label given the Z (r) observed labels and an estimate of the confusion matrices of the worker who provided those labels.

In particular, we define ??, q to be DISPLAYFORM0 If the observed label is uniformly random, then all weights are equal and the loss is identical for all predictor functions f .

Absent noise, we recover the original loss function.

Under the Dawid-Skene model, given the observed noisy labels Z (r) , an estimate of confusion matrices ??, and an estimate of prior q, the posterior distribution of the true labels can be computed as follows: DISPLAYFORM1 where I[.] is the indicator function which takes value one if the identity inside it is true, otherwise zero.

We give guarantees on the performance of the proposed loss function in Theorem 4.1.

In practice, it is robust to noise level and significantly outperforms the unbiased loss function.

Given ??, q , we learn the predictor function f by minimizing the empirical risk DISPLAYFORM2

The next question is: how do we get a good estimate ?? of the true confusion matrix ?? for each worker.

If redundancy r is sufficiently large, we can employ the EM algorithm.

However, in practical applications, redundancy is typically three or five.

With so little redundancy, the standard applications of EM are of limited use.

In this paper we look to transcend this problem, posing the question: Can we estimate confusion matrices of workers even when there is only one label per example?

While this isn't possible in the standard approach, we can overcome this obstacle by incorporating a supervised learning model into the process of assessing worker quality.

Under the Dawid-Skene model, the EM algorithm estimates the ground truth labels and the confusion matrices in the following way: It alternately fixes the ground truth labels and the confusion matrices by their estimates and and updates its estimate of the other by maximizing the likelihood of the observed labels.

The alternating maximization begins by initializing the ground truth labels with a majority vote.

With only 1 label per example, EM estimates that all the workers are perfect.

We propose using model predictions as estimates of the ground truth labels.

Our model is initially trained on the majority vote of the labels.

In particular, if the model prediction is {t i } i??? [n] , where t i ??? K, then the maximum likelihood estimate of confusion matrices and the prior distribution is given below.

For the a-th worker, ??ks for k, s ??? K, and q k for k ??? K, we have, DISPLAYFORM0 The estimate is effective when the hypothesis class F is expressive enough and the learner is robust to noise.

Thus the model should, in general, have small training error on correctly labeled examples and large training error on wrongly labeled examples.

Consider the case when there is only one label per example.

The model will be trained on the raw noisy labels given by the workers.

For simplicity, assume that each worker is either a hammer (always correct) or a spammer (chooses labels uniformly random).

By comparing model predictions with the training labels, we can identify which workers are hammers and which are spammers, as long as each worker labels sufficiently many examples.

We expect a hammer to agree with the model more often than a spammer.

Building upon the previous two ideas, we present 'Model Bootstrapped EM', an iterative algorithm for efficient learning from noisy labels with small redundancy.

MBEM takes data, noisy labels, and the corresponding worker IDs, and returns the best predictor function f in the hypothesis class F.In the first round, we compute the weights of the modified loss function ??, q by using the weighted majority vote.

Then we obtain an estimate of the worker confusion matrices ?? using the maximum likelihood estimator by taking the model predictions as the ground truth labels.

In the second round, weights of the loss function are computed as the posterior probability distribution of the ground truth labels conditioned on the noisy labels and the estimate of the confusion matrices obtained in the previous round.

In our experiments, only two rounds are required to achieve substantial improvements over baselines.

, T : number of iterations Output: f : predictor function Initialize posterior distribution using weighted majority vote DISPLAYFORM0 estimate confusion matrices ?? and prior class distribution q given {t i } i???[n] ?? (a) ??? Equation FORMULA14 , for a ??? [m]; q ??? Equation (7) estimate label posterior distribution given ??, q DISPLAYFORM1

The following result gives guarantee on the excess risk for the learned predictor function f in terms of the VC dimension of the hypothesis class F. Recall that risk of a function f w.r.t.

loss function is defined to be R ,D (f ) := E (X,Y )???D [ (f (X), Y )], Equation (2).

We assume that the classification problem is binary, and the distribution q, prior on ground truth labels Y , is uniform and is known to us.

We give guarantees on the excess risk of the predictor function f , and accuracy of ?? estimated in the second round.

For the purpose of analysis, we assume that fresh samples are used in each round for computing function f and estimating ??.

In other words, we assume that f and ?? are each computed using n/4 fresh samples in the first two rounds.

We define ?? and ?? to capture the average worker quality.

Here, we give their concise bound for a special case when all the workers are identical, and their confusion matrix is represented by a single parameter, 0 ??? ?? < 1/2.

Where ?? kk = 1 ??? ??, and ?? ks = ?? for k = s.

Each worker makes a mistake with probability ??.

?? ??? (?? + ) r r u=0 r u (?? u + ?? r???u ) ???1 , where ?? := (?? + )/(1 ??? ?? ??? ).

?? for this special case is ??.

A general definition of ?? and ?? for any confusion matrices ?? is provided in the Appendix.

Theorem 4.1.

Define N := nr to be the number of total annotations collected on n training examples with redundancy r. Suppose min f ???F R ,D (f ) ??? 1/4.

For any hypothesis class F with a finite VC dimension V , and any ?? < 1, there exists a universal constant C such that if N is large enough and satisfies DISPLAYFORM0 then for binary classification with 0-1 loss function , f and ?? returned by Algorithm 1 after T = 2 iterations satisfies DISPLAYFORM1 and DISPLAYFORM2 , with probability at least 1 ??? ??.

Where := 2 4 ?? + 2 8 m log(2 6 m??)/N , and ?? := min f ???F R ,D (f ) + C( ??? V + log(1/??))/((1 ??? 2??) N/r).

1 is defined to be with ?? in it replaced by ?? .The price we pay in generalization error bound on f is (1 ??? 2?? ).

Note that, when n is large, goes to zero, and ?? ??? 2??(1 ??? ??), for r = 1.If min f ???F R ,D (f ) is sufficiently small, VC dimension is finite, and ?? is bounded away from 1/2 then for n = O(m log(m)/r), we get 1 to be sufficiently small.

Therefore, for any redundancy r, error in confusion matrix estimation is small when the number of training examples is sufficiently large.

Hence, for N large enough, using Equation FORMULA19 and the bound on ?? , we get that for fixed total annotation budget, the optimal choice of redundancy r is 1 when the worker quality (1 ??? ??) is above a threshold.

In particular, if (1 ??? ??) ??? 0.825 then label once is the optimal strategy.

However, in experiments we observe that with our algorithm the choice of r = 1 is optimal even for much smaller values of worker quality.

We experimentally investigate our algorithm, MBEM, on multiple large datasets.

On CIFAR-10 ( BID15 ) and ImageNet BID5 ), we draw noisy labels from synthetic worker models.

We confirm our results on multiple worker models.

On the MS-COCO dataset BID18 , we accessed the real raw data that was used to produce this annotation.

We compare MBEM against the following baselines:??? MV: First aggregate labels by performing a majority vote, then train the model.??? weighted-MV: Model learned using weighted loss function with weights set by majority vote.??? EM: First aggregate labels using EM.

Then train model in the standard fashion.

BID4 ??? weighted-EM: Model learned using weighted loss function with weights set by standard EM.??? oracle weighted EM: This model is learned by minimizing ?? , using the true confusion matrices.??? oracle correctly labeled: This baseline is trained using the standard loss function but only using those training examples for which at least one of the r workers has given the true label.

Note that oracle models cannot be deployed in practice.

We show them to build understanding only.

In the plots, the dashed lines correspond to MV and EM algorithm.

The black dashed-dotted line shows generalization error if the model is trained using ground truth labels on all the training examples.

For experiments with synthetic noisy workers, we consider two models of worker skill:??? hammer-spammer: Each worker is either a hammer (always correct) with probability ?? or a spammer (chooses labels uniformly at random).??? class-wise hammer-spammer: Each worker can be a hammer for some subset of classes and a spammer for the others.

The confusion matrix in this case has two types of rows: (a) hammer class: row with all off-diagonal elements being 0.

(b) spammer class: row with all elements being 1/|K|.

A worker is a hammer for any class k ??? K with probability ??.

We sample m confusion matrices {?? (a) } a??? [m] according to the given worker skill distribution for a given ??.

We assign r workers uniformly at random to each example.

Given the ground truth labels, we generate noisy labels according to the probabilities given in a worker's confusion matrix, using Equation (1).

While our synthetic workers are sampled from these specific worker skill models, our algorithms do not use this information to estimate the confusion matrices.

A Python implementation of the MBEM algorithm is available for download at https://github.com/khetan2/MBEM.

This dataset has a total of 60K images belonging to 10 different classes where each class is represented by an equal number of images.

We use 50K images for training the model and 10K images for testing.

We use the ground truth labels to generate noisy labels from synthetic workers.

We choose m = 100, and for each worker, sample confusion matrix of size 10 ?? 10 according to the worker skill distribution.

All our experiments are carried out with a 20-layer ResNet which achieves an accuracy of 91.5%.

With the larger ResNet-200, we can obtain a higher accuracy of 93.5% but to save training time we restrict our attention to ResNet-20.

We run MBEM 1 for T = 2 rounds.

We assume that the prior distribution q is uniform.

We report mean accuracy of 5 runs and its standard error for all the experiments.

FIG0 shows plots for CIFAR-10 dataset under various settings.

The three plots in the first row correspond to "hammer-spammer" worker skill distribution and the plots in the second row correspond to "class-wise hammer-spammer" distribution.

In the first plot, we fix redundancy r = 1, and plot generalization error of the model for varying hammer probability ??.

MBEM significantly outperforms all baselines and closely matches the Oracle weighted EM.

This implies MBEM recovers worker confusion matrices accurately even when we have only one label per example.

When there is only one label per example, MV, weighted-MV, EM, and weighted-EM all reduce learning with the standard loss function .In the second plot, we fix hammer probability ?? = 0.2, and vary redundancy r. This plot shows that weighted-MV and weighted-EM perform significantly better than MV and EM and confirms that our approach of weighing the loss function with posterior probability is effective.

MBEM performs much better than weighted-EM at small redundancy, demonstrating the effect of our bootstrapping idea.

However, when redundancy is large, EM works as good as MBEM.In the third plot, we show that when the total annotation budget is fixed, it is optimal to collect one label per example for as many examples as possible.

We fixed hammer probability ?? = 0.2.

Here, when redundancy is increased from 1 to 2, the number of of available training examples is reduced by 50%, and so on.

Performance of weighted-EM improves when redundancy is increased from 1 to 5, showing that with the standard EM algorithm it might be better to collect redundant annotations for fewer example (as it leads to better estimation of worker qualities) than to singly annotate more examples.

However, MBEM always performs better than the standard EM algorithm, achieving lowest generalization error with many singly annotated examples.

Unlike standard EM, MBEM can estimate worker qualities even with singly annotated examples by comparing them with model predictions.

This corroborates our theoretical result that label-once is the optimal strategy when worker quality is above a threshold.

The plots corresponding to class-wise hammer-spammer workers follow the same trend.

Estimation of confusion matrices in this setting is difficult and hence the gap between MBEM and the baselines is less pronounced.

ImageNet The ImageNet-1K dataset contains 1.2M training examples and 50K validation examples.

We divide test set in two parts: 10K for validation and 40K for test.

Each example belongs to one of the possible 1000 classes.

We implement our algorithms using a ResNet-18 that achieves top- 1 accuracy of 69.5% and top-5 accuracy of 89% on ground truth labels.

We use m = 1000 simulated workers.

Although in general, a worker can mislabel an example to one of the 1000 possible classes, our simulated workers mislabel an example to only one of the 10 possible classes.

This captures the intuition that even with a larger number of classes, perhaps only a small number are easily confused for each other.

Therefore, each workers' confusion matrix is of size 10 ?? 10.

Note that without this assumption, there is little hope of estimating a 1000 ?? 1000 confusion matrix for each worker by collecting only approximately 1200 noisy labels from a worker.

The rest of the settings are the same as in our CIFAR-10 experiments.

In Figure 2 , we fix total annotation budget to be 1.2M and vary redundancy from 1 to 9.

When redundancy is 9, we have only (1.2/9)M training examples, each labeled by 9 workers.

MBEM outperforms baselines in each of the plots, achieving the minimum generalization error with many singly annotated training examples.

These experiments use the real raw annotations collected when MS-COCO was crowdsourced.

Each image in the dataset has multiple objects (approximately 3 on average).

For validation set images (out of 40K), labels were collected from 9 workers on average.

Each worker marks which out of the 80 possible objects are present.

However, on many examples workers disagree.

These annotations were collected to label bounding boxes but we ask a different question: what is the best way to learn a model to perform multi-object classification, using these noisy annotations.

We use 35K images for training the model and 1K for validation and 4K for testing.

We use raw noisy annotations for training the model and the final MS-COCO annotations as the ground truth for the validation and test set.

We use ResNet-98 deep learning model and train independent binary classifier for each of the 80 object classes.

Table in Figure 3 shows generalization F1 score of four different algorithms: majority vote, EM, MBEM using all 9 noisy annotations on each of the training examples, and a model trained using the ground truth labels.

MBEM performs significantly better than the standard majority vote and slightly improves over EM.

In the plot, we fix the total annotation budget to 35K.

We vary redundancy from 1 to 7, and accordingly reduce the number of training examples to keep the total number of annotations fixed.

When redundancy is r < 9 we select uniformly at random r of the original 9 noisy annotations.

Again, we find it best to singly annotate as many examples as possible when the total annotation budget is fixed.

MBEM significantly outperforms majority voting and EM at small redundancy.

We introduced a new algorithm for learning from noisy crowd workers.

We also presented a new theoretical and empirical demonstration of the insight that when examples are cheap and annotations expensive, it's better to label many examples once than to label few multiply when worker quality is above a threshold.

Many avenues seem ripe for future work.

We are especially keen to incorporate our approach into active query schemes, choosing not only which examples to annotate, but which annotator to route them to based on our models current knowledge of both the data and the worker confusion matrices.

Lemma A.2.

Under the assumptions of Theorem 4.1, ??? error in estimated confusion matrices ?? as computed in Equation (7), using n samples and a predictor function f with risk R ,D ??? ??, is bounded by DISPLAYFORM0 with probability at least 1 ??? ?? 1 .First we apply Lemma A.1 with P ?? computed using majority vote.

We get a bound on the risk of function f computed in the first round.

With this f , we apply Lemma A.2.

When n is sufficiently large such that Equation (8) holds, the denominator in Equation FORMULA18 , 1/K ??? ?? ??? 8 m log(4mK 2 /?? 1 )/(nr) ??? 1/8.

Therefore, in the first round, the error in confusion matrix estimation is bounded by , which is defined in the Theorem.

For the second round: we apply Lemma A.1 with P ?? computed as the posterior distribution (5).Where ??? error in ?? is bounded by .

This gives the desired bound in (9).

With this f , we apply Lemma A.2 and obtain ??? error in ?? bounded by 1 , which is defined in the Theorem.

For the given probability of error ?? in the Theorem, we chose ?? 1 in both the lemma to be ??/4 such that with union bound we get the desired probability of ??.

DISPLAYFORM1 For ease of notation, we denote D W,??,r by D ?? .

Similar to R ,D , risk of decision function f with respect to the modified loss function ?? is characterized by the following quantities: DISPLAYFORM2 2.

Empirical ?? -risk on samples: R ?? ,D?? (f ) : DISPLAYFORM3 i , wi ).

With the above definitions, we have the following, DISPLAYFORM4 DISPLAYFORM5 where (19) follows from Equation FORMULA5 .

FORMULA5 follows from the fact that f is the minimizer of R ?? ,D?? as computed in FORMULA12 .

FORMULA5 follows from the basic excess-risk bound.

V is the VC dimension of hypothesis class F, and C is a universal constant.

Following shows the inequality used in Equation (19).

For binary classification, we denote the two classes by Y, ???Y .

DISPLAYFORM6 DISPLAYFORM7 where FORMULA5 follows from Equation FORMULA5 .

FORMULA5 follows from the fact that for 0-1 loss function (f (X), Y ) + (f (X), ???Y ) = 1. (24) follows from the definition of ?? ?? defined in Equation (12).

When ?? is computed using weighted majority vote of the workers then (24) holds with ?? ?? replaced by ??.

?? is defined in (14).Following shows the equality used in Equation FORMULA5 .

Using the notations ?? ?? and ?? ?? , in the following, for any function f ??? F, we compute the excess risk due to the unbiasedness of the modified loss function ?? .

where ?? ?? (Y ) is defined in (11).

Where (25) follows from the definition of ?? given in Equation (4).Observe that when ?? is computed using weighted majority vote of the workers then Equation FORMULA5 holds with ?? ?? (Y ) replaced by ??(y).

??(y) is defined in (13).A.2 PROOF OF LEMMA A.2Recall that we have DISPLAYFORM0 DISPLAYFORM1 Note that A, B, C, D, E depend upon a ??? [m], k, s ??? K. However, for ease of notations, we have not included the subscripts.

We have, ??ks ??? ??

<|TLDR|>

@highlight

A new approach for learning a model from noisy crowdsourced annotations.

@highlight

This paper proposes a method for learning from noisy labels, focusing on the case when data isn't redundantly labeled with theoretical and experimental validation

@highlight

This paper focuses on the learning-from-crowds problem, where jointly updating the classifier weights and the confusion matrices of workers can help on the estimation problem with rare crowdsourced labels.

@highlight

Proposes a supervised learning algorithm for modeling label and worker quality and utilizes algorithm to study how much redundancy is required in crowdsourcing and whether low redundancy with abundant noise examples lead to better labels.