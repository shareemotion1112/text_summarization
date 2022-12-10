Positive-unlabeled (PU) learning addresses the problem of learning a binary classifier from positive (P) and unlabeled (U) data.

It is often applied to situations where negative (N) data are difficult to be fully labeled.

However, collecting a non-representative N set that contains only a small portion of all possible N data can be much easier in many practical situations.

This paper studies a novel classification framework which incorporates such biased N (bN) data in PU learning.

The fact that the training N data are biased also makes our work very different from those of standard semi-supervised learning.

We provide an empirical risk minimization-based method to address this PUbN classification problem.

Our approach can be regarded as a variant of traditional example-reweighting algorithms, with the weight of each example computed through a preliminary step that draws inspiration from PU learning.

We also derive an estimation error bound for the proposed method.

Experimental results demonstrate the effectiveness of our algorithm in not only PUbN learning scenarios but also ordinary PU leaning scenarios on several benchmark datasets.

In conventional binary classification, examples are labeled as either positive (P) or negative (N), and we train a classifier on these labeled examples.

On the contrary, positive-unlabeled (PU) learning addresses the problem of learning a classifier from P and unlabeled (U) data, without need of explicitly identifying N data BID6 BID42 ).PU learning finds its usefulness in many real-world problems.

For example, in one-class remote sensing classification , we seek to extract a specific land-cover class from an image.

While it is easy to label examples of this specific land-cover class of interest, examples not belonging to this class are too diverse to be exhaustively annotated.

The same problem arises in text classification, as it is difficult or even impossible to compile a set of N samples that provides a comprehensive characterization of everything that is not in the P class BID24 BID8 .

Besides, PU learning has also been applied to other domains such as outlier detection BID13 BID36 ), medical diagnosis BID45 , or time series classification BID28 .By carefully examining the above examples, we find out that the most difficult step is often to collect a fully representative N set, whereas only labeling a small portion of all possible N data is relatively easy.

Therefore, in this paper, we propose to study the problem of learning from P, U and biased N (bN) data, which we name PUbN learning hereinafter.

We suppose that in addition to P and U data, we also gather a set of bN samples, governed by a distribution distinct from the true N distribution.

As described previously, this can be viewed as an extension of PU learning, but such bias may also occur naturally in some real-world scenarios.

For instance, let us presume that we would like to judge whether a subject is affected by a particular disease based on the result of a physical examination.

While the data collected from the patients represent rather well the P distribution, healthy subjects that request the examination are in general highly biased with respect to the whole healthy subject population.

We are not the first to be interested in learning with bN data.

In fact, both BID22 and BID7 attempted to solve similar problems in the context of text classification.

BID22 simply discarded negative samples and performed ordinary PU classification.

It was also mentioned in the paper that bN data could be harmful.

BID7 adapted another strategy.

The authors considered even gathering unbiased U data is difficult and learned the classifier from only P and bN data.

However, their method is specific to text classification because it relies on the use of effective similarity measures to evaluate similarity between documents.

Therefore, our work differs from these two in that the classifier is trained simultaneously on P, U and bN data, without resorting to domain-specific knowledge.

The presence of U data allows us to address the problem from a statistical viewpoint, and thus the proposed method can be applied to any PUbN learning problem in principle.

In this paper, we develop an empirical risk minimization-based algorithm that combines both PU learning and importance weighting to solve the PUbN classification problem, We first estimate the probability that an example is sampled into the P or the bN set.

Based on this estimate, we regard bN and U data as N examples with instance-dependent weights.

In particular, we assign larger weights to U examples that we believe to appear less often in the P and bN sets.

P data are treated as P examples with unity weight but also as N examples with usually small or zero weight whose actual value depends on the same estimate.

The contributions of the paper are three-fold:1.

We formulate the PUbN learning problem as an extension of PU learning and propose an empirical risk minimization-based method to address the problem.

We also theoretically establish an estimation error bound for the proposed method.

2.

We experimentally demonstrate that the classification performance can be effectively improved thanks to the use of bN data during training.

In other words, PUbN learning yields better performance than PU learning.

3.

Our method can be easily adapted to ordinary PU learning.

Experimentally we show that the resulting algorithm allows us to obtain new state-of-the-art results on several PU learning tasks.

Relation with Semi-supervised Learning With P, N and U data available for training, our problem setup may seem similar to that of semi-supervised learning BID2 BID29 .

Nonetheless, in our case, N data are biased and often represent only a small portion of the whole N distribution.

Therefore, most of the existing methods designed for the latter cannot be directly applied to the PUbN classification problem.

Furthermore, our focus is on deducing a risk estimator using the three sets of data, whereas in semi-supervised learning the main concern is often how U data can be utilized for regularization BID10 BID1 BID20 BID25 .

The two should be compatible and we believe adding such regularization to our algorithm can be beneficial in many cases.

Relation with Dataset Shift PUbN learning can also be viewed as a special case of dataset shift 1 BID31 ) if we consider that P and bN data are drawn from the training distribution while U data are drawn from the test distribution.

Covariate shift BID38 BID39 ) is another special case of dataset shift that has been studied intensively.

In the covariate shift problem setting, training and test distributions have the same class conditional distribution and only differ in the marginal distribution of the independent variable.

One popular approach to tackle this problem is to reweight each training example according to the ratio of the test density to the training density BID15 .

Nevertheless, simply training a classifier on a reweighted version of the labeled set is not sufficient in our case since there may be examples with zero probability to be labeled.

It is also important to notice that the problem of PUbN learning is intrinsically different from that of covariate shift and neither of the two is a special case of the other.

In this section, we briefly review the formulations of PN, PU and PNU classification and introduce the problem of learning from P, U and bN data.

Let x ∈ R d and y ∈ {+1, −1} be random variables following an unknown probability distribution with density p(x, y).

Let g : R d → R be an arbitrary decision function for binary classification and ℓ : R → R + be a loss function of margin yg(x) that usually takes a small value for a large margin.

The goal of binary classification is to find g that minimizes the classification risk: DISPLAYFORM0 where E (x,y)∼p (x,y) [·] denotes the expectation over the joint distribution p(x, y).

When we care about classification accuracy, ℓ is the zero-one loss ℓ 01 (z) = (1 − sign(z))/2.

However, for ease of optimization, ℓ 01 is often substituted with a surrogate loss such as the sigmoid loss ℓ sig (z) = 1/(1 + exp(z)) or the logistic loss ℓ log (z) = ln(1 + exp(−z)) during learning.

In standard supervised learning scenarios (PN classification), we are given P and N data that are sampled independently from p(x | y = +1) and p(x | y = −1) as X P = {x [ℓ(−g(x) )] partial risks and π = p(y = 1) the P prior.

We have the equality R(g) = πR DISPLAYFORM1 DISPLAYFORM2 The classification risk (1) can then be empirically approximated from data bŷ DISPLAYFORM3 .

By minimizingR PN (g) we obtain the ordinary empirical risk minimizerĝ PN .

In PU classification, instead of N data X N we have only access to X U = {x DISPLAYFORM0 a set of U samples drawn from the marginal density p(x).

Several effective algorithms have been designed to address this problem.

BID23 proposed the S-EM approach that first identifies reliable N data in the U set and then runs the Expectation-Maximization (EM) algorithm to build the final classifier.

The biased support vector machine (Biased SVM) introduced in BID24 regards U samples as N samples with smaller weights.

BID27 solved the PU problem by aggregating classifiers trained to discriminate P data from a small random subsample of U data.

More recently, attention has been paid on the unbiased risk estimator proposed in du BID4 and du BID3 .

The key idea is to use the following equality: DISPLAYFORM1 .

As a result, we can approximate the classification risk (1) bŷ DISPLAYFORM2 DISPLAYFORM3 We then minimizê R PU (g) to obtain another empirical risk minimizerĝ PU .

Note that as the loss is always positive, the classification risk (1) thatR PU (g) approximates is also positive.

However, BID19 pointed out that when the model of g is too flexible, that is, when the function class G is too large,R PU (ĝ PU ) indeed goes negative and the model seriously overfits the training data.

To alleviate overfitting, the authors observed that R DISPLAYFORM4 and proposed the non-negative risk estimator for PU learning: DISPLAYFORM5 In terms of implementation, stochastic optimization was used and when r =R − U (g) − πR − P (g) becomes negative for a mini-batch, they performed a step of gradient ascent along ∇r to make the mini-batch less overfitted.

In semi-supervised learning (PNU classification), P, N and U data are all available.

An abundance of works have been dedicated to solving this problem.

Here we in particular introduce the PNU risk estimator proposed in BID35 .

By directly leveraging U data for risk estimation, it is the most comparable to our method.

The PNU risk is simply defined as a linear combination of PN and PU/NU risks.

Let us just consider the case where PN and PU risks are combined, then for some γ ∈ [0, 1], the PNU risk estimator is expressed aŝ DISPLAYFORM0 We can again consider the non-negative correction by forcing the term DISPLAYFORM1 to be non-negative.

In the rest of the paper, we refer to the resulting algorithm as non-negative PNU (nnPNU) learning (see Appendix D.4 for an alternative definition of nnPNU and the corresponding results).

In this paper, we study the problem of PUbN learning.

It differs from usual semi-supervised learning in the fact that labeled N data are not fully representative of the underlying N distribution p(x | y = −1).

To take this point into account, we introduce a latent random variable s and consider the joint distribution p (x, y, s) DISPLAYFORM0 .

Both π and ρ are assumed known throughout the paper.

In practice they often need to be estimated from data BID17 BID32 .

In place of ordinary N data we collect a set of bN samples DISPLAYFORM1 The goal remains the same: we would like to minimize the classification risk (1).

In this section, we propose a risk estimator for PUbN classification and establish an estimation error bound for the proposed method.

Finally we show how our method can be applied to PU learning as a special case when no bN data are available.

DISPLAYFORM0 The first two terms on the right-hand side of the equation can be approximated directly from data by writingR DISPLAYFORM1 We therefore focus on the third termR DISPLAYFORM2 Our approach is mainly based on the following theorem.

We relegate all proofs to the appendix.

DISPLAYFORM3 In the theorem,R − s=−1 (g) is decomposed into three terms, and when the expectation is substituted with the average over training samples, these three terms are approximated respectively using data from X U , X P and X bN .

The choice of h and η is thus very crucial because it determines what each of the three terms tries to capture in practice.

Ideally, we would like h to be an approximation of σ.

Then, for x such that h(x) is close to 1, σ(x) is close to 1, so the last two terms on the righthand side of the equation can be reasonably evaluated using X P and X bN (i.e., samples drawn from p(x | s = +1)).

On the contrary, if h(x) is small, σ(x) is small and such samples can be hardly found in X P or X bN .

Consequently the first term appeared in the decomposition is approximated with the help of X U .

Finally, in the empirical risk minimization paradigm, η becomes a hyperparameter that controls how important U data is against P and bN data when we evaluateR − s=−1 (g).

The larger η is, the more attention we would pay to U data.

One may be curious about why we do not simply approximate the whole risk using only U samples, that is, set η to 1.

There are two main reasons.

On one hand, if we have a very small U set, which means n U ≪ n P and n U ≪ n bN , approximating a part of the risk with labeled samples should help us reduce the estimation error.

This may seem unrealistic but sometimes unbiased U samples can also be difficult to collect BID16 .

On the other hand, more importantly, we have empirically observed that when the model of g is highly flexible, even a sample regarded as N with small weight gets classified as N in the latter stage of training and performance of the resulting classifier can thus be severely degraded.

Introducing η alleviates this problem by avoiding treating all U data as N samples.

As σ is not available in reality, we propose to replace σ by its estimateσ in (6).

We further substitute h with the same estimate and obtain the following expression: DISPLAYFORM4 ] .We notice thatR s=−1,η,σ depends both on η andσ.

It can be directly approximated from data bŷ DISPLAYFORM5 .We are now able to derive the empirical version of Equation FORMULA14 aŝ DISPLAYFORM6 Estimating σ If we regard s as a class label, the problem of estimating σ is then equivalent to training a probabilistic classifier separating the classes with s = +1 and s = −1.

Observing that DISPLAYFORM7 , it is straightforward to apply nnPU learning with availability of X P , X bN and X U to DISPLAYFORM8 In other words, here we regard X P and X bN as P and X U as U, and attempt to solve a PU learning problem by applying nnPU.

Since we are interested in the classposterior probabilities, we minimize the risk with respect to the logistic loss and apply the sigmoid function to the output of the model to getσ(x).

However, the above risk estimator accepts any reasonableσ and we are not limited to using nnPU for computingσ.

For example, the least-squares fitting approach proposed in BID18 for direct density ratio estimation can also be adapted to solving the problem.

Here we establish an estimation error bound for the proposed method.

Let G be the function class from which we find a function.

The Rademacher complexity of G for the samples of size n drawn from q(x) is defined as DISPLAYFORM0 where X = {x 1 , . . .

, x n } and θ = {θ 1 , . . .

, θ n } with each x i drawn from q(x) and θ i as a Rademacher variable BID26 .

In the following we will assume that R n,q (G) vanishes asymptotically as n → ∞. This holds for most of the common choices of G if proper regularization is considered BID0 BID9 .

Assume additionally the exis- DISPLAYFORM1 We also assume that ℓ is Lipschitz continuous on the interval DISPLAYFORM2 Theorem 2.

Let g * = arg min g∈G R(g) be the true risk minimizer andĝ PUbN,η,σ = arg min g∈GRPUbN,η,σ (g) be the PUbN empirical risk minimizer.

We suppose thatσ is a fixed function independent of data used to computeR PUbN,η,σ DISPLAYFORM3 .

Then for any δ > 0, with probability at least 1 − δ, DISPLAYFORM4 Theorem 2 shows that as DISPLAYFORM5 where O p denotes the order in probability.

As for ϵ, knowing thatσ is also estimated from data in practice 3 , apparently its value depends on both the estimation algorithm and the number of samples that are involved in the estimation process.

For example, in our approach we applied nnPU with the logistic loss to obtainσ, so the excess risk can be written as E x∼p(x) KL(σ(x)∥σ(x)), where by abuse of notation KL(p∥q) = p ln(p/q)+(1−p) ln((1−p)/(1−q)) denotes the KL divergence between two Bernouilli distributions with parameters respectively p and q. It is known that BID44 .

The excess risk itself can be decomposed into the sum of the estimation error and the approximation error.

BID19 showed that under mild assumptions the estimation error part converges to zero when the sample size increases to infinity in nnPU learning.

It is however impossible to get rid of the approximation error part which is fixed 2 For instance, this holds for linear-in-parameter model class DISPLAYFORM6 DISPLAYFORM7 where Cw and C ϕ are positive constants BID26 .3 These data, according to theorem 2, must be different from those used to evaluateR PUbN,η,σ (g) .

This condition is however violated in most of our experiments.

See Appendix D.3 for more discussion.once we fix the function class G. To circumvent this problem, we can either resort to kernel-based methods with universal kernels BID44 or simply enlarge the function class when we get more samples.

In PU learning scenarios, we only have P and U data and bN data are not available.

Nevertheless, if we let y play the role of s and ignore all the terms related to bN data, our algorithm is naturally applicable to PU learning.

Let us name the resulting algorithm PUbN\N, then DISPLAYFORM0 whereσ is an estimate of p(y = +1 | x) and DISPLAYFORM1 ] .PUbN\N can be viewed as a variant of the traditional two-step approach in PU learning which first identifies possible N data in U data and then perform ordinary PN classification to distinguish P data from the identified N data.

However, being based on state-of-the-art nnPU learning, our method is more promising than other similar algorithms.

Moreover, by explicitly considering the posterior p(y = +1 | x), we attempt to correct the bias induced by the fact of only taking into account confident negative samples.

The benefit of using an unbiased risk estimator is that the resulting algorithm is always statistically consistent, i.e., the estimation error converges in probability to zero as the number of samples grows to infinity.

In this section, we experimentally investigate the proposed method and compare its performance against several baseline methods.

We focus on training neural networks with stochastic optimization.

For simplicity, in an experiment, σ and g always use the same model and are trained for the same number of epochs.

All models are learned using AMSGrad BID33 as the optimizer and the logistic loss as the surrogate loss unless otherwise specified.

To determine the value of η, we introduce another hyperparameter τ and choose η such that #{x ∈ X U |σ(x) ≤ η} = τ (1 − π − ρ)n U .

In all the experiments, an additional validation set, equally composed of P, U and bN data, is sampled for both hyperparameter tuning and choosing the model parameters with the lowest validation loss among those obtained after every epoch.

Regarding the computation of the validation loss, we use the PU risk estimator (2) with the sigmoid loss for g and an empirical approximation of DISPLAYFORM0

We assess the performance of the proposed method on three benchmark datasets: MNIST, CIFAR-10 and 20 Newsgroups.

Experimental details are given in Appendix C. In particular, since all the three datasets are originally designed for multiclass classification, we group different categories together to form a binary classification problem.

Baselines.

When X bN is given, two baseline methods are considered.

The first one is nnPNU adapted from (4).

In the second method, named as PU→PN, we train two binary classifiers: one is learned with nnPU while we regard s as the class label, and the other is learned from X P and X bN to separate P samples from bN samples.

A sample is classified in the P class only if it is so classified by the two classifiers.

When X bN is not available, nnPU is compared with the proposed PUbN\N.Sampling bN Data To sample X bN , we suppose that the bias of N data is caused by a latent prior probability change BID40 BID14 in the N class.

Let z ∈ Z := DISPLAYFORM0 In the experiments, the latent categories are the original class labels of the datasets.

Concrete definitions of X bN with experimental results are summarized in TAB0 .Results.

Overall, our proposed method consistently achieves the best or comparable performance in all the scenarios, including those of standard PU learning.

Additionally, using bN data can effectively help improving classification performance.

However, the choice of algorithm is essential.

Both nnPNU and the naive PU→PN are able to leverage bN data to enhance classification accuracy in only relatively few tasks.

In the contrast, the proposed PUbN successfully reduce the misclassification error most of the time.

Clearly, the performance gain that we can benefit from the availability of bN data is case-dependent.

On CIFAR-10, the greatest improvement is achieved when we regard mammals (i.e. cat, deer, dog and horse) as P class and drawn samples from latent categories bird and frog as labeled negative data.

This is not surprising because birds and frogs are more similar to mammals than vehicles, which makes the classification harder specifically for samples from these two latent categories.

By explicitly labeling these samples as N data, we allow the classifier to make better predictions for these difficult samples.

Through experiments we have demonstrated that the presence of bN data effectively helps learning a better classifier.

Here we would like to provide some intuition for the reason behind this.

Let us consider the MNIST learning task where X bN is uniformly sampled from the latent categories 1, 3 and 5.

We project the representations learned by the classifier (i.e., the activation values of the last layer of the neural network) into a 2D plane using PCA for both nnPU and PUbN algorithms.

The results are shown in FIG0 .

Since for both nnPU and PUbN classifiers, the first two principal components account around 90% of variance, we believe that this figure depicts fairly well the learned representations.

Thanks to the use of bN data, in the high-level feature space 1, 3, 5 and P data are further pushed away when we employ the proposed PUbN learning algorithm, and we are always able to separate 7, 9 from P to some extent.

This explains the better performance which is achieved by PUbN learning and the benefit of incorporating bN data into the learning process.

This paper studied the PUbN classification problem, where a binary classifier is trained on P, U and bN data.

The proposed method is a two-step approach inspired from both PU learning and importance weighting.

The key idea is to attribute appropriate weights to each example to evaluate the classification risk using the three sets of data.

We theoretically established an estimation error bound for the proposed risk estimator and experimentally showed that our approach successfully leveraged bN data to improve the classification performance on several real-world datasets.

A variant of our algorithm was able to achieve state-of-the-art results in PU learning.

DISPLAYFORM0

We obtain Equation (6) after replacing p(x, s = +1) by πp(x | y = +1)+ρp(x | y = −1, s = +1).

Forσ and η given, let us define DISPLAYFORM0 The following lemma establishes the uniform deviation bound fromR PUbN,η,σ to R PUbN,η,σ .

DISPLAYFORM1 ] be a fixed function independent of data used to computeR PUbN,η,σ and η ∈ (0, 1].

For any δ > 0, with probability at least 1 − δ, DISPLAYFORM2 Proof.

For ease of notation, let DISPLAYFORM3 ] .From the sub-additivity of the supremum operator, we have DISPLAYFORM4 As a consequence, to conclude the proof, it suffices to prove that with probability at least 1 − δ/3, the following bounds hold separately: DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 Below we prove (8).

FORMULA43 and FORMULA0 are proven similarly.

Let ϕ x : R → R + be the function defined by DISPLAYFORM8 Following the proof of Theorem 3.1 in BID26 , it is then straightforward to show that with probability at least 1 − δ/3, it holds that DISPLAYFORM9 where θ = {θ 1 , . . .

, θ n P } and each θ i is a Rademacher variable.

Also notice that for all x, ϕ x is a (L l /η)-Lipschitz function on the interval [−C g , C g ].

By using a modified version of Talagrad's concentration lemma (specifically, Lemma 26.9 in Shalev-Shwartz & Ben-David (2014)), we can show that, when the set X P is fixed, we have DISPLAYFORM10 After taking expectation over X P ∼ p np P , we obtain the Equation FORMULA42 .However, what we really want to minimize is the true risk R(g).

Therefore, we also need to bound the difference between R PUbN,η,σ (g) and R(g), or equivalently, the difference between DISPLAYFORM11 Proof.

One one hand, we havē DISPLAYFORM12 On the other hand, we can expressR DISPLAYFORM13 The last equality follows from the fact p( DISPLAYFORM14 From the second to the third line we use the Cauchy-Schwarz inequality.

|A 1 − A 2 | ≤ C l √ ζϵ can be proven similarly, which concludes the proof.

Combining lemma 1 and lemma 2, we know that with probability at least 1 − δ, the following holds: DISPLAYFORM15 Finally, with probability at least 1 − δ, DISPLAYFORM16 The first inequality uses the definition ofĝ PUbN,η,σ .

In terms of validation we want to choose the model forσ such that DISPLAYFORM0 The last term does not depend onσ and can be ignored if we want to identifyσ achieving the smallest J(σ).

We denote by J(σ) the sum of the first two terms.

The middle term can be further expanded using DISPLAYFORM1 The validation loss of an estimationσ is then defined aŝ DISPLAYFORM2 It is also possible to minimize this value directly to acquireσ.

In our experiments we decide to learn σ by nnPU for a better comparison between different methods.

In the experiments we work on multiclass classification datasets.

Therefore it is necessary to define the P and N classes ourselves.

MNIST is processed in such a way that pair numbers 0, 2, 4, 6, 8 form the P class and impair numbers 1, 3, 5, 7, 9 form the N class.

Accordingly, π = 0.49.

For CIFAR-10, we consider two definitions of the P class.

The first one corresponds to a quite natural task that aims to distinguish vehicles from animals.

Airplane, automobile, ship and truck are therefore defined to be the P class while the N class is formed by bird, cat, deer, dog, frog and horse.

For the sake of diversity, we also study another task in which we attempt to distinguish the mammals from the non-mammals.

The P class is then formed by cat, deer, dog, and horse while the N class consists of the other six classes.

We have π = 0.4 in the two cases.

As for 20 Newsgroups, alt.

, comp., misc.

and rec. make up the P class whereas sci., soc.

and talk. make up the N class.

This gives π = 0.56.

For the three datasets, we use the standard test examples as a held-out test set.

The test set size is thus of 10000 for MNIST and CIFAR-10, and 7528 for 20 Newsgroups.

Regarding the training set, we sample 500, 500 and 6000 P, bN and U training examples for MNIST and 20 Newsgroups, and 1000, 1000 and 10000 P, bN and U training examples for CIFAR-10.

The validation set is always five times smaller than the training set.

The original 20 Newsgroups dataset contains raw text data and needs to be preprocessed into text feature vectors for classification.

In our experiments we borrow the pre-trained ELMo word embedding BID30 from https://allennlp.org/elmo.

The used 5.5B model was, according to the website, trained on a dataset of 5.5B tokens consisting of Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008 WMT -2012 .

For each word, we concatenate the features from the three layers of the ELMo model, and for each document, as suggested in BID34 , we concatenate the average, minimum, and maximum computed along the word dimension.

This results in a 9216-dimensional feature vector for a single document.

MNIST For MNIST, we use a standard ConvNet with ReLU.

This model contains two 5x5 convolutional layers and one fully-connected layer, with each convolutional layer followed by a 2x2 max pooling.

The channel sizes are 5-10-40.

The model is trained for 100 epochs with a weight decay of 10 −4 .

Each minibatch is made up of 10 P, 10 bN (if available) and 120 U samples.

The learning rate α ∈ {10 −2 , 10 −3 } and τ ∈ {0. 5, 0.7, 0.9}, γ ∈ {0.1, 0.3, 0.5, 0.7, 0 .9} are selected with validation data.

CIFAR-10 For CIFAR-10, we train PreAct ResNet-18 BID11 for 200 epochs and the learning rate is divided by 10 after 80 epochs and 120 epochs.

This is a common practice and similar adjustment can be found in BID11 .

The weight decay is set to 10 −4 .

The minibatch size is 1/100 of the number of training samples, and the initial learning rate is chosen from {10 −2 , 10 −3 }.

We also have τ ∈ {0. 5, 0.7, 0.9} and γ ∈ {0.1, 0.3, 0.5, 0.7, 0.9}. 20 Newsgroups For 20 Newsgroups, with the extracted features, we simply train a multilayer perceptron with two hidden layers of 300 neurons for 50 epochs.

We use basically the same hyperparameters as for MNIST except that the learning rate α is selected from {5 · 10 Our method, specifically designed for PUbN learning, naturally outperforms other baseline methods in this problem.

Nonetheless, Table 1 equally shows that the proposed method when applied to PU learning, achieves significantly better performance than the state-of-the-art nnPU algorithm.

Here we numerically investigate the reason behind this phenomenon.

Besides nnPU and PUbN\N, we compare with unbiased PU (uPU) learning (2).

Both uPU and nnPU are learned with the sigmoid loss, learning rate 10 −3 for MNIST, initial learning rate 10 −4 for CIFAR-10, and learning rate 10 −4 for 20 Newsgroups.

This is because uPU learning is unstable with the logistic loss.

The other parts of the experiments remain unchanged.

On the test sets we compute the false positive rates, false negative rates and misclassification errors for the three methods and plot them in FIG3 .

We first notice that PUbN\N still outperforms nnPU trained with the sigmoid loss.

In fact, the final performance of the nnPU classifier does not change much when we replace the logistic loss with the sigmoid loss.

In BID19 , the authors observed that uPU overfits training data with the risk going to negative.

In other words, a large portion of U samples are classified to the N class.

This is confirmed in our experiments by an increase of false negative rate and decrease of false positive rate.

nnPU remedies the problem by introducing the non-negative risk estimator (3).

While the non-negative correction successfully prevents false negative rate from going up, it also causes more N samples to be classified as P compared to uPU.

However, since the gain in terms of false negative rate is enormous, at the end nnPU achieves a lower misclassification error.

By further identifying possible N samples after nnPU learning, we expect that our algorithm can yield lower false positive rate than nnPU without misclassifying too many P samples as N as in the case of uPU.

FIG3 suggests that this is effectively the case.

In particular, we observe that on MNIST, our method achieves the same false positive rate than uPU whereas its false negative rate is comparable to nnPU.

In the proposed algorithm we introduce η to control howR s=−1 (g) is approximated from data and assume that ρ = p(y = −1, s = +1) is given.

Here we conduct experiments to see how our method is affected by these two factors.

To assess the influence of η, from TAB0 we pick four learning tasks and we choose τ from {0.5, 0.7, 0.9, 2} while all the other hyperparameters are fixed.

Similarly to simulate the case where ρ is misspecified, we replace it by ρ ′ ∈ {0.8ρ, ρ, 1.2ρ} in our learning method and run experiments with all hyperparameters being fixed to a certain value.

However, we still use the true ρ to compute η from τ to ensure that we always use the same number of U samples in the second step of the algorithm independent of the choice of ρ ′ .The results are reported in TAB1 and TAB2 .

We can see that the performance of the algorithm is sensitive to the choice of τ .

With larger value of τ , more U data are treated as N data in PUbN learning, and consequently it often leads to higher false negative rate and lower false positive rate.

The trade-off between these two measures is a classic problem in binary classification.

In particular, when τ = 2, a lot more U samples are involved in the computation of the PUbN risk (7), but this does not allow the classifier to achieve a better performance.

We also observe that there is a positive correlation between the misclassification rate and the validation loss, which confirms that the optimal value of η can be chosen without need of unbiased N data.

TAB2 shows that in general slight misspecification of ρ does not cause obvious degradation of the classification performance.

In fact, misspecification of ρ mainly affect the weights of each sample when we computeR PUbN,η,σ (due to the direct presence of ρ in (7) and influence on estimating σ).

However, as long as the variation of these weights remain in a reasonable range, the learning algorithm should yield classifiers with similar performances.

DISPLAYFORM0 Theorem 2 suggests thatσ should be independent from the data used to computeR PUbN,η,σ .

Therefore, here we investigate the performance of our algorithm whenσ and g are optimized using different sets of data.

We sample two training sets and two validation sets in such a way that they are all disjoint.

The size of a single training set and a single validation set is as indicated in Appendix C.2, except for 20 Newsgroups we reduce the number of examples in a single set by half.

We then use different pairs of training and validation sets to learnσ and g. For 20 Newsgroups we also conduct standard experiments whereσ and g are learned on the same data, whereas for MNIST and CIFAR-10 we resort to TAB0 .The results are presented in TAB3 .

Estimating σ from separate data does not seem to benefit much the final classification performance, despite the fact that it requires collecting twice more samples.

In fact,R − s=−1,η,σ (g) is a good approximation ofR − s=−1,η,σ (g) as long as the functionσ is smooth enough and does not possess abrupt changes between data points.

With the use of non-negative correction, validation data and L2 regularization, the resultingσ does not overfit training data so this should always be the case.

As a consequence, even ifσ and g are learned on the same data, we are still able to achieve small generalization error with sufficient number of samples.

In subsection 2.3, we define the nnPNU algorithm by forcing the estimator of the whole N partial risk to be positive.

However, notice that the term γ(1 − π)R − N (g) is always positive and the chances are that including it simply makes non-negative correction weaker and is thus harmful to the final classification performance.

Therefore, here we consider an alternative definition of nnPNU where we only force the term (1 − γ)(R − U (g) − πR − P (g)) to be positive.

We plug the resulting algorithm in the experiments of subsection 4.2 and summarize the results in TAB4 in which we denote the alternative version of nnPNU by nnPU+PN since it uses the same non-negative correction as nnPU.

The table indicates that neither of the two definitions of nnPNU consistently outperforms the other.

It also ensures that there is always a clear superiority of our proposed PUbN algorithm compared to nnPNU despite its possible variant that is considered here.

17.14 ± 1.87 15.80 ± 0.95 soc.

> talk.

> sci.15.93 ± 1.88 15.80 ± 1.91 sci.

14.69 ± 0.46 14.50 ± 1.32 talk.14.38 ± 0.74 14.71 ± 1.01 soc.

> talk.

> sci.14.41 ± 0.70 13.66 ± 0.72

@highlight

This paper studied the PUbN classification problem, where we incorporate biased negative (bN) data, i.e., negative data that is not fully representative of the true underlying negative distribution, into positive-unlabeled (PU) learning.