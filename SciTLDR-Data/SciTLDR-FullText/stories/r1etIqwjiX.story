Spoken term detection (STD) is the task of determining whether and where a given word or phrase appears in a given segment of speech.

Algorithms for STD are often aimed at maximizing the gap between the scores of positive and negative examples.

As such they are focused on ensuring that utterances where the term appears are ranked higher than utterances where the term does not appear.

However, they do not determine a detection threshold between the two.

In this paper, we propose a new approach for setting an absolute detection threshold for all terms by introducing a new calibrated loss function.

The advantage of minimizing this loss function during training is that it aims at maximizing not only the relative ranking scores, but also adjusts the system to use a fixed threshold and thus enhances system robustness and maximizes the detection accuracy rates.

We use the new loss function in the structured prediction setting and extend the discriminative keyword spotting algorithm for learning the spoken term detector with a single threshold for all terms.

We further demonstrate the effectiveness of the new loss function by applying it on a deep neural Siamese network in a weakly supervised setting for template-based spoken term detection, again with a single fixed threshold.

Experiments with the TIMIT, WSJ and Switchboard corpora showed that our approach not only improved the accuracy rates when a fixed threshold was used but also obtained higher Area Under Curve (AUC).

Spoken term detection (STD) refers to the proper detection of any occurrence of a given word or phrase in a speech signal.

Typically, any such system assigns a confidence score to every term it presumably detects.

A speech signal is called positive or negative, depending on whether or not it contains the desired term.

Ideally, an STD system assigns a positive speech input with a score higher than the score it assigns to a negative speech input.

During inference, a detection threshold is chosen to determine the point from which a score would be considered positive or negative.

The choice of the threshold represents a trade-off between different operational settings, as a high value of the threshold could cause an excessive amount of false negatives (instances incorrectly classified as negative), whereas a low value of the threshold could cause additional false positives (instances incorrectly classified as positive).The performance of STD systems can be measured by the Receiver Operation Characteristics (ROC) curve, that is, a plot of the true positive (spotting a term correctly) rate as a function of the false positive (mis-spotting a term) rate.

Every point on the graph corresponds to a specific threshold value.

The area under the ROC curve (AUC) is the expected performance of the system for all threshold values.

A common practice for finding the threshold is to empirically select the desired value using a cross validation procedure.

In BID2 , the threshold was selected using the ROC curve.

Similarly, in BID7 BID16 and the references therein, the threshold was chosen such that the system maximized the Actual Term Weighted Value (ATWV) score BID14 .

Additionally, BID18 claims that a global threshold that was chosen for all terms was inferior to using a term specific threshold BID17 .In this paper we propose a new method to embed an automatic adjustment of the detection threshold within a learning algorithm, so that it is fixed and known for all terms.

We present two algorithmic implementations of our method: the first is a structured prediction model that is a variant of the discriminative keyword spotting algorithm proposed by BID15 BID20 BID21 , and the second implementation extends the approach used for the structured prediction model on a variant of whole-word Siamese deep network models BID9 BID1 BID13 .

Both of these approaches in their original form aim to assign positive speech inputs with higher scores than those assigned to negative speech inputs, and were shown to have good results on several datasets.

However, maximizing the gap between the scores of the positive and negative examples only ensures the correct relative order between those examples, and does not fix a threshold between them; therefore it cannot guarantee a correct detection for a global threshold.

Our goal is to train a system adjusted to use a global threshold valid for all terms.

In this work, we set the threshold to be a fixed value, and adjust the decoding function accordingly.

To do so, we propose a new loss function that trains the ranking function to separate the positive and negative instances; that is, instead of merely assigning a higher score to the positive examples, it rather fixes the threshold to be a certain constant, and assigns the positive examples with scores greater than the threshold, and the negative examples with scores less than the threshold.

Additionally, this loss function is a surrogate loss function which extends the hinge loss to penalize misdetected instances, thus enhancing the system's robustness.

The new loss function is an upper bound to the ranking loss function, hence minimizing the new loss function can lead to minimization of ranking errors, or equivalently to the maximization of the AUC.

In the STD task, we are provided with a speech utterance and a term and the goal is to decide whether or not the term is uttered.

The term can be provided as a sequence of phonemes or by an acoustic representation given by a speech segment in which the term is known to be uttered.

Throughout the paper, scalars are denoted using lower case Latin letters, e.g., x, and vectors using bold face letters, e.g., x. A sequence of elements is denoted with a bar (x) and its length is written as |x|.Formally, the speech signal is represented by a sequence of acoustic feature vectorsx = (x 1 , . . .

, x T ), where each feature vector is d dimensional x t ??? R d for all 1 ??? t ??? T .

Note that in our setting the number of frames T is not fixed.

We denote by X = (R d ) * the set of all finite length sequences over DISPLAYFORM0 , where p l ??? P for all 1 ??? l ??? L r and P is the set of phoneme symbols.

We denote by P * the set of all finite length sequences over P.A term is a word or a short phrase and is presented to the system either as a sequence of phonemes in the strongly supervised setting or as an acoustic segment containing the term in the weakly supervised setting.

We denote the abstract domain of the term representations (as either a phoneme sequence or an acoustic segment) by R. Our goal is to find a spoken term detector, which takes as input a speech segment and a term and returns a binary output indicating whether the term was pronounced in the acoustic segment or not.

Most often the spoken term detector is a function that returns a real value expressing the confidence that the target term has been uttered.

The confidence score outputted by this function is compared to a threshold, and if the score is above the threshold the term is declared to have been pronounced in the speech segment.

Formally, the detector is a function f from X ?? R to R. The detection threshold is denoted by the scalar ?? ??? R. Usually there is no single threshold for all terms, and it needs to be adjusted after decoding.

Our goal in this work is to propose a new method to learn the spoken term detector from a training set of examples, so that the model is adjusted to use a fixed given threshold for all terms.

The function f is found from a training set of examples, where each example is composed of two speech segments and a representation of a term.

Although the training set contains many different terms, the function f should be able to detect any term, not only those already seen in the training phase.

In this section we describe our main idea, whereas in the next sections we propose two implementations: one with a structured prediction model where the training data is fully supervised and the term is given as a phoneme sequence, and the other with a deep learning model where the training data is weakly supervised and the term is given using a segment of speech.

Recall that during inference the input to the detector is a speech segment and a term and the output is a confidence that the term was pronounced in the speech segment, which is compared to a threshold.

Since the detection threshold is typically not fixed and does depend on the input term, it is often desired to learn the function f such that the confidence of a speech segment that contains the term is higher than the confidence of a speech segment that does not contain the term.

Formally, let us consider two sets of speech segments.

Denote by X r+ a set of speech segments in which the term r is articulated.

Similarly, denote by X r??? a set of speech segments in which the term r is not articulated.

We assume that term r, and two instancesx + ??? X r+ andx ??? ??? X r??? are drawn from a fixed but unknown probability distribution, and we denote by P{??} and E[??] the probability and the expectation of an event ?? under this distribution.

The probability that the confidence ofx + is higher than the confidence ofx ??? is the area under the ROC curve (AUC) BID10 BID0 : DISPLAYFORM0 Instead of keeping a threshold for each term, we adjust f so that the detection threshold will be fixed, and set to a predefined value.

Assume that the predefined threshold is ??, then the accuracy in the prediction can be measured by DISPLAYFORM1 where ??? is the logical conjunction symbol.

Hence our goal is to find the parameters of the function f so as to maximize the accuracy Acc ?? for a given threshold ??.

Equivalently we find the parameters of function f that minimize the error defined as DISPLAYFORM2 where ??? is the logical disjunction symbol, and I{??} is the indicator function, that equals 1 if the predicate ?? holds true and 0 otherwise.

Unfortunately, we cannot minimize the error function (5) directly, since it is a combinatorial quantity.

A common practice is to replace the error function with a surrogate loss function which is easy to minimize.

We suggest to minimize a convex upper-bound to the error function.

Specifically, we replace the last term with the hinge upper bound, DISPLAYFORM3 where [??] + = max{??, 0}. The last upper bound holds true since I{?? < 0} ??? [1 ??? ??] + .

Adding the margin of 1 means that the function f faces a harder problem: not only does it need to have a confidence greater than ?? for a positive speech segment and a confidence lower than ?? for a negative speech segment -the confidence value must be at least ?? + 1 and at most ?? ??? 1 for positive and negative speech segments, respectively.

We now turn to present two algorithmic implementations that are aimed at minimizing the loss function derived from (6), namely, DISPLAYFORM4 Hopefully the minimization of this loss function will lead to the minimization of Err ?? in (6).

Our first construction is based on previous work on discriminative keyword spotting and spoken term detection BID15 BID20 BID21 , where the goal was to maximize the AUC.

In this setting we assume that the term is expressed as a sequence of phonemes denotedp r ??? P * .In this fully-supervised setting we define the alignment between a phoneme sequence and a speech signal.

We denote by y l ??? N the start time of phoneme p l (in frame units), and by e l = y l+1 ??? 1 the end time of phoneme p l , except for the phoneme p L , where the end frame is e L .

The alignment sequence?? r corresponding to the phonemes sequencep r is a sequence of start-times and an end-time, DISPLAYFORM0 , where y l is the start-time of phoneme p l and e L is the end-time of the last phoneme p L .Similar to previous work BID15 BID20 BID21 , our detection function f is composed of a predefined set of n feature functions, {?? j } n j=1 , each of the form ?? j : X * ?? P * ?? N * ??? R. Each feature function takes as input an acoustic representation of a speech utterancex ??? X * , together with the term phoneme sequencep r ??? P * , and a candidate alignment sequence?? r ??? N * , and returns a scalar in R which represents the confidence in the suggested alignment sequence given the term r. For example, one element of the feature function can sum the number of times phoneme p comes after phoneme p , while other elements of the feature function may extract properties of each acoustic feature vector x t provided that phoneme p is pronounced at time t. Our basic set of feature functions is the same as the set used in BID15 .We believe that the threshold value for each term depends on the term's phonetic content and its relative duration.

In order to allow f to learn these subtle differences from the data we introduced an additional set of 4 feature functions: a feature function representing a bias; a feature function that counts the number of occurrences of a phoneme in a term, i.e., |{q|q ???p r }|; a feature function holding the number of phonemes in the term, i.e., |p r |; and a feature function holding the average length of the phonemes in a term, i.e., DISPLAYFORM1 As mentioned above, our goal is to learn a spoken term detector f , which takes as input a sequence of acoustic featuresx, a termp r , and returns a confidence value in R. The form of the function f we use is f (x,p r ) = max DISPLAYFORM2 where w ??? R n is a vector of importance weights that should be learned and ?? ??? R n is a vector function composed out of the feature functions ?? j .

In other words, f returns a confidence prediction about the existence of the term in the utterance by maximizing a weighted sum of the scores returned by the feature function elements over all possible alignment sequences.

If the confidence of the function f is above the threshold ?? then we predict that the term is pronounced in the signal and located in the time span defined by the alignment sequence?? that maximizes (8): DISPLAYFORM3 where the search for the best sequence is practically performed using the Viterbi algorithm as described in BID15 .

Specifically, the algorithm finds the optimal time segment for the keyword r in the speech signalx, and then aligns the phoneme sequencep r within the chosen time segment.

The parameters of the model w are found by minimizing the loss function defined in (7).

In the fully supervised case we use a slightly modified version of it, which is defined as DISPLAYFORM4 This is a convex function in the vector of the parameters w.

We use the Passive-Aggressive (PA) algorithm BID5 BID15 to find the parameters w.

The algorithm receives as input a set of training examples DISPLAYFORM5 and examines each of them sequentially.

Initially, we set w = 0.

At each iteration i, the algorithm updates w according to the current example (p ri ,x DISPLAYFORM6 Denote by w i???1 the value of the weight vector before the ith iteration.

We set the next weight vector w i to be the minimizer of the following optimization problem, DISPLAYFORM7 where C serves as a complexity-accuracy trade-off parameter and ?? is a non-negative slack variable, which indicates the loss of the ith example.

Intuitively, we would like to minimize the loss of the current example, i.e., the slack variable ??, while keeping the weight vector w as close as possible to our previous weight vector w i???1 .

The constraint makes the projection of the utterance in which the term is uttered onto w greater than ?? + 1, and the projection of the utterance in which the term is not uttered onto w less than ?? ??? 1.

The closed form solution to the above optimization problem can be derived using the Karush-Kuhn-Tucker conditions in the same lines of [6, App.

A].The loss in (10) is composed of two hinge functions and therefore introduces a more elaborate solution than the one derived for the ranking loss of BID15 .

We call this algorithm PA-ACC (Passive-Aggressive to maximize Accuracy).

Details about the implementation of the algorithm can be seen in BID8 .

The PA-ACC algorithm is an online algorithm, and deals with drifting hypotheses; therefore, it is highly influenced by the recent examples.

Common methods to convert an online algorithm to a batch algorithm are either by taking the average over all the parameters {w i }, or by taking the best w i over a validation set BID6 BID3 .

We turn to exemplify our idea in the weakly supervised setting using deep networks.

This implementation is based on recent work on whole-word segmental systems BID9 BID1 BID13 .

These works present a Siamese network model trained with a ranking loss function.

Siamese networks BID4 are neural networks with a tied set of parameters which take as input a pair of speech segments and are trained to minimize or maximize the similarity between the segments depending on whether the same term has been pronounced in the pair of segments.

In this setting the term r is represented by two speech segments rather than a phoneme sequence: a speech segment in which the term r is pronounced,x + , and a speech segment, in which the term r is not pronounced,x ??? .

Similar to those works, we assume that each example in the training set is composed of the triplet (x t ,x + ,x ??? ), wherex t ,x + ??? X r+ andx ??? ??? X r??? .

The goal in training the network is that the similarity score betweenx t andx + should be above the similarity score between DISPLAYFORM0 Denote by g u : X * ??? R d a deep network (the specific architecture is discussed in Section 6) with a set of parameters u, where d is the dimension of the output.

Denote by ?? : R d ?? R d ??? R a measure of similarity between two output vectors of size d. The spoken term detector f u : X * ?? X * ??? R is the composition of Siamese networks g u and the similarity function.

Hence an unknown speech segmentx t can be compared to a positive or negative speech segment, as follows: DISPLAYFORM1 The tied parameters u of all the models were found in BID9 BID1 BID13 using the minimization of the ranking loss function DISPLAYFORM2 for different options of the similarity function ??.

In this work we propose to minimize the loss function in FORMULA5 , which is defined for the weakly supervised case as follows: DISPLAYFORM3 when the margin of ?? > 0 is used.

In this case, the parameter ?? is not set to 1, since the function f u is not a linear function and hence is not scale invariant to the margin, as in the structured prediction case.

In the next section we present our empirical comparison for all the loss functions on different speech corpora.

In this section we present experimental results that demonstrate the effectiveness of our proposed calibrated loss function BID6 .

We compared the proposed loss to the standard approach of maximizing AUC using the ranking loss as in BID11 where no fixed threshold can be set.

The experiments on the structured prediction model were conducted using fully supervised training sets of read speech (TIMIT, WSJ).

The experiments on the deep network model performed on a weakly supervised data of conversational speech (Switchboard).

To validate the effectiveness of the proposed approach, we performed experiments with the TIMIT corpus.

The training and validation sets were taken from the TIMIT training set.

The training set was composed from 1,512 randomly chosen terms, corresponding to 11,139 pairs of positive and negative utterances (each term repeated more than once).

Similarly, the validation set was composed from 378 different randomly chosen terms, corresponding to 2,892 pairs.

The validation set was used to tune the algorithm's parameters.

The test set was composed of 80 terms that were suggested as a benchmark in BID15 , and are distinct from the terms used in the training and validation sets.

For each term, we randomly picked at most 20 utterances in which the term was uttered and at most 20 utterances in which it was not uttered.

The utterances were taken from the TIMIT test set.

The number of test utterances in which the term was uttered was not always 20, since some terms were uttered less than 20 times in the whole TIMIT test set.

We measure performance using the AUC defined in (1) and using the accuracy of a fixed threshold ?? denoted Acc ?? .

Specifically, we calculate AUC on the test set of m test examples according to DISPLAYFORM0 and the accuracy by DISPLAYFORM1 We tested the PA-ACC algorithm using two options.

The first was whether the final weight vector was a result of averaging or was the best to perform on the validation set.

The second option was whether the new feature functions we introduced were normalized by the length of the phoneme sequence |p r | or not.

The AUC and Acc ?? rates found on our validation and test set are presented in Table 1 .

In training PA-ACC we chose arbitrarily ?? = 0.

Table 1 : AUC and ACC ?? rates of the PA-ACC algorithm.

The first column indicates whether the new feature functions were normalized or not.

The second column indicates whether the final weight vector was a result of averaging or was the best to perform on the validation set.

We can see from the table that since the TIMIT dataset is very clean the detection rates are very good and the AUC is almost always 1.

The results presented here are improved over the results presented in BID15 due to the introduction of the new feature functions.

It is interesting to note that the best Acc 0 results on the validation set were obtained when the additional features were not normalized and the final weight vector was selected over the validation set, while the best Acc 0 results on the test set were obtained with the opposite conditions: when the final weight vector was the average one and the additional feature functions were normalized.

Further research on feature functions should be conducted and extended to a larger dataset.

We now turn to compare the performance of our algorithm against two other algorithms.

The first is the discriminative keyword spotting algorithm presented in BID15 , which is the Passive-Aggressive algorithm trained with the ranking loss to maximize the AUC.

It is denoted here as PA-AUC.

We introduce two versions of this algorithm: the original version and an extended version with the additional set of feature functions described in Sec. 4.

When using the extended version of PA-AUC, normalizing the features had no affect on our results.

Similarly, a comparison of using the final weight vector versus the best weight vector yielded similar outcomes.

The second algorithm is an HMM-based spoken term detection algorithm presented in BID15 .

BID0 For all the algorithms we report the AUC and Acc ?? in TAB1 .

For the two versions of PA-AUC we selected a single threshold ?? that gave the best Acc ?? on the validation set.

Similarly we selected the best threshold for the HMM algorithm.

For PA-ACC we arbitrarily selected ?? = 0.

It is interesting to see that the AUC of PA-ACC is the same or even higher than that of the PA-AUC.

Since Acc ?? is a lower bound to AUC, the AUC can be thought of as Acc ?? with the best threshold selected for every term in the set.

Indeed from TAB1 we see that the Acc ?? was very close to the AUC but did not reach it.

We evaluate the model trained on TIMIT on the Wall Street Journal (WSJ) corpus BID19 .

This corpus corresponds to read articles of the Wall Street Journal, and hence presents a different linguistic context compared to TIMIT.

Both the discriminative system and the HMM-based system were trained on the TIMIT corpus as described above and evaluated on a different set of 80 keywords from the WSJ corpus.

For each keyword, we randomly picked at most 20 utterances in which the keyword was uttered and at most 20 utterances in which it was not uttered from the si_tr_s portion of the WSJ corpus.

We used the same setting as in BID15 .

As before we arbitrarily chose the threshold ?? = 0.

The results are presented in TAB2 .

Again we see from TAB2 that the model trained with the proposed loss function led to higher accuracy rates with similar AUC rates, meaning a better separation between the positive speech utterances and the negative speech utterances.

Our second set of experiments is focused on deep networks trained on weakly supervised data.

Our model is based on previous work on network training using the ranking loss BID9 BID1 BID13 .

We used the same experimental setup as BID13 BID13 .

The term weak supervision BID13 , refers to the fact that supervision is given in the form of known word pairs, rather than the exact location of the term and its phonetic content as in Subsection 6.1.The data was taken from the Switchboard corpus of English conversational telephone speech.

Melfrequency cepstral coefficients (MFCCs) with first and second order derivatives features were extracted and cepstral mean and variance normalization (CMVN) was applied per conversation side.

The training set consisted of the set of about 10k word tokens from BID11 BID12 ; it consisted of word segments of at least 5 characters and 0.5 seconds in duration extracted from a forced alignment of the transcriptions, and comprises about 105 minutes of speech.

For the Siamese convolutional neural networks (CNNs), this set results in about 100k word segment pairs.

For testing, we used the 11k-token set from BID11 BID12 .The architecture of each network was the same as BID13 BID13 : 1-D convolution with 96 filters over 9 frames; ReLU (Rectified Linear Unit) ; max pooling over 3 units; 1-D convolution with 96 filters over 8 units; ReLU; max pooling over 3 units; 2048-unit fully-connected ReLU; 1024-unit fully-connected linear layer.

All weights were initialized randomly.

Models were trained using ADADELTA BID22 .We reproduced the results in BID13 by training the Siamese network using the ranking loss in (12) with the cosine similarity as a similarity function ??.

The cosine similarity of two vectors v 1 ??? R d and v 2 ??? R d is defined as DISPLAYFORM0 where this function returns a number close to 1 if the two vectors are similar and a number close to -1 if the two vectors are not.

We also train the network using the same similarity function using the Acc ?? loss function with ?? = 0 as in BID12 .

For the ranking loss we used ?? = 0.15 while for the Acc ?? loss we used ?? = 0.10, because the margin ?? is counted twice for Acc ?? loss.

These values were chosen by maximizing Acc ?? on a validation set over 5 epochs.

The AUC and Acc ?? values for training of 5 to 30 epochs are given in TAB3 .

Other training parameters and settings were exactly the same as in BID13 .

We can see in the table that both AUC and Acc ?? are higher when the system is trained with the calibrated ranking loss function.

The reason that the AUC was also improved is most likely because the calibrated ranking loss function is harder to optimize than the ranking loss.

In this work, we introduced a new loss function that can be used to train a spoken term detection system with a fixed desired threshold for all terms.

We introduced a new discriminative structured prediction model that is based on the Passive-Aggressive algorithm.

We show that the new loss can be used in training weakly supervised deep network models as well.

Results suggest that our new loss function yields AUC and accuracy values that are better than previous works' results.

@highlight

Spoken Term Detection, using structured prediction and deep networks, implementing a new loss function that both maximizes AUC and ranks according to a predefined threshold.