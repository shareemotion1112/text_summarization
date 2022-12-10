One of the most prevalent symptoms among the elderly population, dementia, can be detected by classifiers trained on linguistic features extracted from narrative transcripts.

However, these linguistic features are impacted in a similar but different fashion by the normal aging process.

Aging is therefore a confounding factor, whose effects have been hard for machine learning classifiers to isolate.



In this paper, we show that deep neural network (DNN) classifiers can infer ages from linguistic features, which is an entanglement that could lead to unfairness across age groups.

We show this problem is caused by undesired activations of v-structures in causality diagrams, and it could be addressed with fair representation learning.

We build neural network classifiers that learn low-dimensional representations reflecting the impacts of dementia yet discarding the effects of age.

To evaluate these classifiers, we specify a model-agnostic score $\Delta_{eo}^{(N)}$ measuring how classifier results are disentangled from age.

Our best models outperform baseline neural network classifiers in disentanglement, while compromising accuracy by as little as 2.56\% and 2.25\% on DementiaBank and the Famous People dataset respectively.

One in three seniors die of Alzheimer's and other types of dementia in the United States (Association, 2018) .

Although its causes are not yet fully understood, dementia impacts people's cognitive abilities in a detectable manner.

This includes different syntactic distributions in narrative descriptions BID28 , more pausing BID29 , higher levels of difficulty in recalling stories BID21 , and impaired memory generally BID20 .

Fortunately, linguistic features can be used to train classifiers to detect various cognitive impairments.

For example, BID8 detected primary progressive aphasia with up to 100% accuracy, and classified subtypes of primary progressive aphasia with up to 79% accuracy on a set of 40 participants using lexical-syntactic and acoustic features.

BID7 classified dementia from control participants with 82% accuracy on narrative speech.

However, dementia is not the only factor causing such detectable changes in linguistic features of speech.

Aging also impairs cognitive abilities BID11 , but in subtly different ways from dementia.

For example, aging inhibits fluid cognitive abilities (e.g., cognitive processing speed) much more than the consolidated abilities (e.g., those related to cumulative skills and memories) BID4 .

In other words, the detected changes of linguistic features, including more pauses and decreased short-term memories, could attribute to just normal aging process instead of dementia.

Unfortunately, due to the high correlation between dementia and aging, it can be difficult to disentangle symptoms are caused by dementia or aging BID24 .

Age is therefore a confounding factor in detecting dementia.

The effects of confounding factors are hard for traditional machine learning algorithms to isolate, and this is largely due to sampling biases in the data.

For example, some algorithms predict higher risk of criminal recidivism for people with darker skin colors BID15 , others identify images of smiling Asians as blinking BID19 , and GloVe word embeddings can project European-American names significantly closer to the words like 'pleasant' than African-American names BID3 .

It is preferable for classifiers to make decisions without biasing too heavily on demographic factors, and therefore to isolate the effects of confounding factors.

However, as we will show in Experiments, traditional neural network classifiers bias on age to infer dementia; this can lead to otherwise avoidable false positives and false negatives that are especially important to avoid in the medical domain.

Graphically, if both age A and dementia D cause changes in a feature X, the result is a v-structure BID17 A →

X ← D which is activated upon observing X. In other words, the confounder A affects P (D|X) if we train the classifier in traditional ways, which is to collect data points {(X, D) (i) } and to learn an inference model P (D|X) approximating the affected P (D|X).Traditionally, there are several ways to eliminate the effects of confounding factors A.Controlling A gives a posterior distribution P (D|X, A)P (A).

This is unfortunately unrealistic for small, imbalanced clinical datasets, in which sparsity may require stratification.

However, the stratified distributions P (D|X, A) can be far from a meaningful representation of the real world (as we will show, e.g., in FIG3 ).

Moreover, a discrepancy in the sizes of age groups can skew the age prior P (A), which would seriously inhibit the generalizability of a classifier.

Controlling X Conducting a randomized control trial (RCT) on X removes all causal paths leading "towards" the variable X, which gives a de-confounded dataset P (D|do(X)) according to the notation in BID27 .

However, RCTs on X are even less practical because simultaneously controlling multiple features produces exponential number of scenarios, and doing this to more than 400 features require far more data points than any available dataset.

Pre-adjusting X according to a pre-trained model X = f (A) per feature could also approximately generate the dataset P (D|do(X)).

However, such a model should consider participant differences, otherwise interpolating using a fixed age A would give exactly the same features for everybody.

The participant differences, however, are best characterized via X, which are the values you want to predict.

To overcome the various problems with these methods, we let our classifiers be aware of cognitive impairments while actively filtering out any information related to aging.

This is a fair representation learning framework that protects age as a "sensitive attribute".Fair representation learning frameworks can be used to train classifiers to equally consider the subjects with different sensitive attributes.

A sensitive attribute (or "protected attribute") can be race, age, or other variables whose impact should be ignored.

In the framework proposed by BID32 , classifiers were penalized for the differences in classification probabilities among different demographic groups.

After training, the classifiers produced better demographic similarities while compromising only a little overall accuracy.

To push the fair representation learning idea further, adversarial training can be incorporated.

BID9 introduced generative adversarial networks, in which a generator and a discriminator are iteratively optimized against each other.

Incorporating adversarial training, BID22 proposed a framework to learn a latent representation of data in order to limit its adversary's ability to classify based on the sensitive attributes.

However, these approaches to fair representation learning only handle binary attributes.

E.g., BID22 binarized age.

To apply to cognitive impairments detection, we want to represent age on a continuous scale (with some granularity if necessary).

We formulate a fairness metric for evaluating the ability of a classifier to isolate a continuous-valued attribute.

We also propose four models that compress high-dimensional feature vectors into low-dimensional representations which encrypt age from an adversary.

We show empirically that our models achieve better fairness metrics than baseline deep neural network classifiers, while compromising accuracies by as little as 2.56% and 2.25% on our two empirical datasets, respectively.

There are many measures of entanglement between classifier outcomes and specific variables.

We briefly review some relevant metrics, and then propose ours.

Correlation (Pearson, Spearman, etc.) is often used to compare classification outputs with component input features.

To the extent that these variables are stochastic, several information theoretic measures could be applied, including Kullback-Leibler divergence and Jensen-Shannon divergence.

These can be useful to depict characteristics of two distributions when no further information about available data is given.

Mutual information can depict the extent of entanglement of two random variables.

If we treat age (A) and dementia (D) as two random variables, then adopting the approach of BID18 gives an estimation of I(A, D).

However, given the size of clinical datasets, it can be challenging to give precise estimations.

An alternative approach is to assume that these variables fit into some probabilistic models.

For example, we might assume the age variable A, dementia indicator variable D, and multi-dimensional linguistic feature X fit into some a priori model (e.g., the v-structure mentioned above, A → X ← D), then the mutual information between A and D is: DISPLAYFORM0 where the entropy of age H A and of cognitive impairment H D remain constant with respect to the input data X, and DISPLAYFORM1 However, this marginalized probability is difficult to approximate well, because (1) the accuracy of the term p(A|X) relies on the ability of our model to infer age from features, and FORMULA20 it is hard to decide on a good prior distribution on linguistic features p(X).

We want to make the model agnostic to age, leading to a meaningless mutual information in the 'ideal' case.

In our frameworks, we do not assume specific graphical models that correlate confounds and outcomes, and we propose more explainable metrics than the traditional statistical ones.

The literature in fairness representation learning offers several metrics for evaluating the extent of bias in classifiers.

Generally, the fairer the classifier is, the less entangled the results are with respect to some protected features.

Demographic parity BID32 stated that the fairest scenario is reached when the composition of the classifier outcome for the protected group is equal to that of the whole population.

While generally useful, this does not apply to our scenario, in which there really are more elderly people suffering from cognitive impairments than younger people (see FIG3 ).Cross-entropy loss BID5 used the binary classification loss of an adversary that tried to predict sensitive data from latent representations, as a measure of fairness.

This measure can only apply to those models containing an adversary component, not traditional classifiers.

Moreover, this loss also depends on the ability of the adversary network.

For example, a value of this loss could indicate confusing representations (so sensitive information are protected well), but it could also indicate a weak adversary.

Equalized odds BID12 proposed a method in which false positive rates should be equal across groups in the ideal case.

BID22 defined fairness distance as the absolute difference in false positive rates between two groups, plus that of the false negative rates: DISPLAYFORM0 where p a and n a correspond to the false positive rate and false negative rate, respectively, with sensitive attribute a = 0 (a = 1).

We propose an extension of the metric used by BID22 to continuous sensitive attributes, suitable for evaluating an arbitrary two-class classifier. , and classifier C(.).

In age-indep-autoencoder and age-indep-entropy FIG0 , a reconstructor R(.) tries to reconstruct input data from the hidden representation.

In age-indep-consensus-nets FIG0 ), a discriminator D(.) tells apart from which modality the representation originates.

First, groups of age along a scale are divided so that each group has multiple participants with both positive and negative diagnoses, respectively.

Let a be the age group each participant is in.

Then, we aim for the expected false positive (FP) rates of the classifier to be as constant as possible across age groups.

This applies likewise to the false negative (FN) rates.

To measure their variability, we use their sum of differences against the mean.

DISPLAYFORM0 wherex represents the mean of variable x.

Special cases To illustrate the nature of our metric, we apply it to several special cases, i.e.:1.

When there is only one age group, our fairness metric has its best possible value: ∆ eo = 0.

2.

When there are only two age groups, our metric equals that of BID22 .

3.

In the extreme case where there are as many age groups as there are sample points (assuming there are no two people with identical ages but with different diagnoses), our metric becomes less informative, because the empirical expected false positive rates of that group is either 0 or 1.

This is a limitation of our metric, and is the reason that we limit the number of age groups to accommodate the size of the training dataset.

Bounds Our metric is bounded.

The lower bound, 0, is reached when all false positive rates are equal and when all false negative rates are equal across age groups.

Letting N a be the number of age groups divided, an upper bound for ∆ (Na) eo is N a for any better-than-trivial binary classifier.

The detailed proof is included in the Appendix.

Disentanglement Our fairness metric illustrates disentanglement.

A higher ∆ (N ) eo corresponds to a higher variation of incorrect predictions by the classifier across different age groups.

Therefore, a lower value of ∆ (N ) eo is desired for classifiers isolating the effects of age to a better extent.

Throughout this paper, we use the terms 'fairness', 'disentanglement', and 'isolation' interchangeably.

We explain a few design choices here, namely linearity and indirect optimization.

eo to be as linear as possible, for explainability of the fairness score itself.

This eliminates possible scores consisting of higher order terms of FP / FN rates.

eo , we encourage the hidden representations to be age-agnostic (we will explain how to set up age agnostic models in the following section).

On the other hand, FP / FN rates are not differentiable after all.

In this section, we describe four different ways of building representation learning models, which we call age-indep-simple, age-indep-autoencoder, age-indep-consensus-net, and age-indep-entropy.

The simplest model consists of an interpreter network I(.) to compress high-dimensional input data, x, to low-dimensional representations: z = I(x) An adversary A(.) tries to predict the exact age from the representation: DISPLAYFORM0 A classifier C(.) estimated the probability of label (diagnosis) based on the representation: for minibatch x in training data X do 4: DISPLAYFORM1 DISPLAYFORM2 Calculate L a , L c 6: DISPLAYFORM3 For optimization, we set up two losses: the classification negative log likelihood loss L c and the adversarial (L2) loss L a , where: DISPLAYFORM4 We want to train the adversary to minimize the L2 loss, to train the interpreter to maximize it, and to train the classifier (and interpreter) to minimize classification loss.

Overall, DISPLAYFORM5 The training steps are taken iteratively, as in previous work BID9 .

The age-indep-autoencoder structure is similar to BID22 , and can be seen as an extension from the age-indep-simple structure.

Similar to age-indep-simple, there is an interpreter I(.), an adversary A(.), and a classifier C(.) network.

The difference is that there is a reconstructor network R(.) that attempts to recover input data from hidden representations: DISPLAYFORM0 The loss functions are set up as: DISPLAYFORM1 Overall, we want to train both the interpreter and the reconstructor to minimize the reconstruction loss term, in additional to all targets mentioned in the age-indep-simple network.

DISPLAYFORM2 The detailed algorithm is similar to Algorithm 1 and is in the Appendix.

This is another extension from the age-indep-simple structure, borrowing an idea from consensus networks BID33 , i.e., that agreements between multiple modalities can result in representations that are beneficial for classification.

By examining the performance of age-indepconsensus-net, we would like to see whether agreement between multiple modalities of data can be trained to be disentangled from age.

Similar to age-indep-simple structures, there is also an adversary A(.) and a classifier C(.).

The interpreter, however, is replaced with several interpreters I 1..M , each compressing a subset of the input data ("modality") into a low-dimensional representation.

The key of age-indep-consensusnetwork models is that these representations are encouraged to be indistinguishable.

For simplicity, we randomly divide the input features into three modalities (M = 3) with equal (±1) features.

A discriminator D(.) tries to identify the modality from which the representation comes: DISPLAYFORM0 The loss functions are set up as: DISPLAYFORM1 Overall, we want to iteratively optimize the networks: DISPLAYFORM2 L c and max DISPLAYFORM3 The detailed algorithm is in the Appendix.

Note that we do not combine the consensus network with the reconstructor because they do not work well with each other empirically.

In one of the experiments by BID34 , each interpreter I m (.) is paired with a reconstructor R m (.) and the performance decreases dramatically.

The reconstructor encourages hidden representations to retain the fidelity of data, while the consensus networks urges hidden representations to keep only the information common among modalities, which prohibits the reconstructor and consensus mechanism to function together.

The fourth model we apply to fair representation learning is motivated by categorical GANs (Springenberg, 2016), where information theoretic metrics characterizing the confidences of predictions can be optimized.

This motivates an additional loss function term; i.e., we want to encourage the interpreter to increase the uncertainty (i.e., to minimize the entropy) while letting the adversary become more confident in predicting ages from representations.

Age-indep-entropy models have the same network structures as age-indep-autoencoder, except that instead of predicting the exact age, the adversary network outputs the probability of the sample age being larger than the mean: DISPLAYFORM0 This enables us to define the empirical entropy H[p] = E x plog 1 p , which describes the uncertainty of predicting age.

Formally, the loss functions are set up as follows: DISPLAYFORM1 where λ H is a hyper-parameter.

For comparison, we also include two variants, namely the ageindep-entropy (binary) and age-indep-entropy (Honly) variants, each keeping only one of the two terms in L a .

In our experiments, we show that these two terms in L a are better applied together.

Overall, the training procedure is the same as age-indep-autoencoder and algorithm pseudocode is in the Appendix: min DISPLAYFORM2

All models are implemented in PyTorch BID26 , optimized with Adam BID16 with initial learning rate of 3 × 10 −4 , and L2 weight decay 10.

For simplicity, we use fully connected networks with ReLU activations BID25 and batch normalization BID14 before output layers, for all interpreter, adversary, classifier, and discriminator networks.

Our frameworks can be applied to other types of networks in the future.

DementiaBank DementiaBank 1 is the largest available public dataset for assessing cognitive impairments using speech, containing 473 narrative picture descriptions from subjects aged between 45 and 90 BID2 .

In each sample, a participant talks about what is happening in a clinically validated picture.

There is no time limit in each session, but the average description lasts about a minute.

79 samples are excluded due to missing age information.

In the remaining data samples, 182 are labeled 'control', and 213 are labeled 'dementia'.

All participants have mini-mental state estimation (MMSE) scores BID6 ) between 1 and 30 2 .

Of all data samples containing age information, the mean is 68.26 and standard deviation is 9.00.

The Famous People dataset BID1 contains 252 transcripts from 17 people (8 with dementia including Gene Wilder, Ronald Reagan and Glen Campbell, and 9 healthy controls including Michael Bloomberg, Woody Allen, and Tara VanDerveer), collected and transcribed from publicly available speech data (e.g., press conferences, interviews, debatse, talk shows).

Seven data samples are discarded due to missing age information.

Among the remaining samples, there are 121 labeled as control and 124 as impaired.

Note that the data samples were gathered across a wide range of ages (mean 59.25, standard deviation 13.60).

For those people diagnosed with dementia, there are data samples gathered both before and after the diagnosis, and all of which are labeled as 'dementia'.

The Famous People dataset permits for early detection several years before diagnosis, which is a more challenging classification task than DementiaBank.

Older participants in both DementiaBank FIG3 ) and the Famous People dataset FIG3 ) are more likely to have cognitive impairments.

eo and ∆eo ) of several traditional classifiers.

DNN is the baseline used to benchmark our neural network based representation learning models.

We extract 413 linguistic features from the narrative descriptions and their transcripts.

These features were previously identified as the most useful for this task BID28 BID7 BID21 BID13 .

Each feature is z-score normalized.

Acoustic: mean, variance, skewness, and kurtosis of the first 42 cepstral coefficients.

Speech fluency: pause-word ratio, utterance length, number and lengths of filled/unfilled pauses.

Lexical: cosine similarity between pairs of utterances, word lengths, lexical richness (movingaverage type-token ratio, Brunet's index, and Honoré's statistics BID10 ).PoS: Number of occurrences of part-of-speech tags, tagged by SpaCy 3 .Syntactic and semantic: occurrences of context-free grammar phrase types, parsed by Stanford CoreNLP BID23 , and Yngve depth statistics BID31 .

As part of expository data analysis, we show that these linguistic features contain information indicating age.

Simple fully connected neural networks can predict age with mean absolute error of 15.5 ± 1.3 years (on DementiaBank 4 ) and 14.3 ± 2.5 years (on the Famous People dataset 5 ).

This indicates that even simple neural networks are able to infer information about age from linguistic features.

Neural classifiers can therefore also easily bias on age, given the utility of age in downstream tasks.

We first set up benchmarks for our classifiers.

We evaluate several traditional classifiers with our fairness metrics (∆ eo , corresponding to dividing ages into N = 2 and N = 5 groups respectively).

The results 6 are listed in Table 1 .

A DNN is used as the baseline because (1) all our models are based on neural networks, and (2) DNN classifiers have had the best (or statistically indistinguishable from the best) accuracy on the DementiaBank and Famous People datasets.

We evaluate the performances of our four proposed neural networks against the DNN baseline.

As an additional ablation study, two variants of age-indep-entropy are also evaluated.

TAB1 : Evaluation results of our representation learning models.

The "age-indep" prefix are replaced with "*" in model names.

age-indep-simple and age-indep-autoencoder have better disentanglement scores, while the rest two models could have better accuracy.

Accuracy The fair representation learning models compromise accuracy, in comparison to DNN baselines.

This confirms that part of the classification power of DNNs come from biasing with regards to age.

On DementiaBank, the age-indep-autoencoder reduces accuracy the least (only 2.56% in comparison to the DNN baseline).

On the Famous People data, age-indep-consensus and age-indep-entropy models compromise accuracies by only 2.25% and 2.75% respectively, which are not statistically different from the DNN baseline 7 .Disentanglement In comparison to DNN baselines, our fair representation learning models improve disentanglement/fairness 8 , the improvements are mostly significant when measured by the two-group scores ∆eo .

Also, the five-group scores ∆eo are less stable for both datasets, and the scores in the Famous People have higher variances than in DementiaBank.

Following is an explanation.

DementiaBank has ∼400 data samples.

In 5-fold cross validation, each of the five age groups has only ∼16 samples during evaluation.

Famous People data contains ∼250 samples, which increases the variance.

When the number of groups, N of ∆ (N ) eo , is kept small (e.g., ∼100 samples per label per group, as in DementiaBank N = 2), the fairness metrics are stable.

The model age-indep-entropy is best used with a loss function containing both the binary classification term and the uncertainty minimization term.

As shown in TAB1 , although having similar fairness metrics 9 , the two variants with only one term could have lower accuracy than age-indep-entropy.

In general, age-indep-simple and age-indep-autoencoder achieve the best fairness metrics.

Noticeably, the better of them surpass traditional classifiers in both ∆

Here, we identify the problem of entangling age in the detection of cognitive impairments.

After explaining this problem with causality diagrams, we formulate it into a fair representation learning task, and propose a fairness score to measure the extent of disentanglement.

We put forward four fair representation learning models that learn low-dimensional representations of data samples containing as little age information as possible.

Our best model improves upon the DNN baseline in our fairness metrics, while compromising as little accuracy as 2.56% (on DementiaBank) and 2.25% (on the Famous People dataset).7 p = 0.20, 0.16 on 38-DoF one-tailed t-tests, respectively.

8 On DementiaBank, p = 0.01 and 0.03 for age-indep-simple and age-indep-entropy on ∆ (2) eo respectively; these are significant.

p = 0.08 and 0.09 on age-indep-autoencoder and age-indep-consensus-net on ∆ (2) eo respectively; these are marginally significant.

However, these differences are not as significant on ∆ Proof of Theorem For each of the age groups: |p a −p| + |n a −n| ≤ max{|p a − 0| + |n a − 0|, |p a − 0.5| + |n a − 0.5|} ≤ max{0.5, 1} = 1 Summing up the N a age groups results in our upper bound N a for non-trivial classifiers.

Following are the pseudo-code algorithms for our remaining three models; age-indep-AutoEncoder, age-indep-ConsensusNetworks, and age-indep-Entropy.

for minibatch x in training data X do for minibatch x in training data X do

@highlight

Show that age confounds cognitive impairment detection + solve with fair representation learning + propose metrics and models.