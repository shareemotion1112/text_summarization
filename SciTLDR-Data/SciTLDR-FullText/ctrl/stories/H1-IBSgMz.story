Self-normalizing discriminative models approximate the normalized probability of a class without having to compute the partition function.

This property is useful to computationally-intensive neural network classifiers, as the cost of computing the partition function grows linearly with the number of classes and may become prohibitive.

In particular, since neural language models may deal with up to millions of classes, their self-normalization properties received notable attention.

Several recent studies empirically found that language models, trained using Noise Contrastive Estimation (NCE), exhibit self-normalization, but could not explain why.

In this study, we provide a theoretical justification to this property by viewing NCE as a low-rank matrix approximation.

Our empirical investigation compares NCE to the alternative explicit approach for self-normalizing language models.

It also uncovers a surprising negative correlation between self-normalization and perplexity, as well as some regularity in the observed errors that may potentially be used for improving self-normalization algorithms in the future.

The ability of statistical language models (LMs) to estimate the probability of a word given a context of preceding words, plays an important role in many NLP tasks, such as speech recognition and machine translation.

Recurrent Neural Network (RNN) language models have recently become the preferred method of choice, having outperformed traditional n-gram LMs across a range of tasks BID8 ).

Unfortunately however, they suffer from scalability issues incurred by the computation of the softmax normalization term, which is required to guarantee proper probability predictions.

The cost of this computation is linearly proportional to the size of the word vocabulary and has a significant impact on both training and testing.

1 Several methods have been proposed to cope with this scaling issue by replacing the softmax with a more computationally efficient component at train time.

These include importance sampling BID1 ), hierarchical softmax BID13 , BlackOut BID7 ) and Noise Contrastive Estimation (NCE) BID5 ).

NCE has been applied to train neural LMs with large vocabularies BID14 ) and more recently was also successfully used to train LSTM-RNN LMs BID17 ; BID3 ; BID19 ), achieving near state-of-the-art performance on language modeling tasks BID8 ; BID2 ).

All the above works focused on solving the run-time complexity problem at train time.

However, at test time the assumption was that one still needs to explicitly compute the softmax normalization term to obtain a normalized score fit as an estimate for the probability of a word.

Self-normalization was recently proposed as means to address the high run-time complexity associated with predicting normalized probabilities at test time.

A self-normalized discriminative model is trained to produce near-normalized scores in the sense that the sum over the scores of all classes is approximately one.

If this approximation is close enough, the assumption is that the costly exact normalization can be waived at test time without significantly sacrificing prediction accuracy BID4 ).

Two main approaches were proposed to train self-normalizing models.

Explicit selfnormalization is based on using softmax for training and explicitly encouraging the normalization term of the softmax to be as close to one as possible, thus making its computation redundant at test time BID4 ; BID0 ; BID2 ).

The alternative approach is based on NCE.

The original formulation of NCE included a normalization term Z. However, the first work that applied NCE to LM BID14 ) discovered, empirically, that fixing Z to a constant did not affect the performance.

More recent studies BID17 ; BID19 ; BID3 ; BID15 ) empirically found that models trained using NCE with a fixed Z, exhibit self-normalization, but they could not explain this behavior.

To the best of our knowledge, the only theoretical analysis of self-normalization was proposed by BID0 .

This analysis shows that a model trained explicitly to be self-normalizing only on a subset of the training instances, can potentially be self-normalizing on other similar instances as well.

However, their analysis cannot explain how NCE can be self-normalizing without explicitly imposing self-normalization on any of its training instances.

The main contribution of this study is providing a theoretical justification to the self-normalization property of NCE, which was empirically observed in prior work.

We do so by showing that NCE's unnormalized objective can be viewed as finding the best low-rank approximation of the normalized conditional probabilities matrix, without having to explicitly estimate the partition function.

While the said self-normalizing property of NCE is more general, we focus the empirical contribution of the paper on language modeling.

We investigate the self-normalization performance of NCE as well as that of the alternative explicit self-normalization approach over two datasets.

Our results suggest, somewhat surprisingly, that models that achieve better perplexities tend to have worse selfnormalization properties.

We also observe that given a context, the sum of the self-normalized scores is negatively correlated with the entropy of the respective normalized distribution.

In this section, we first review the NCE algorithm for language modeling and then introduce an interpretation of it as a matrix factorization procedure.

Noise Contrastive Estimation (NCE) is a popular algorithm for efficiently training language models.

NCE transforms the parameter learning problem into a binary classifier training problem.

Let p(w|c) be the probability of a word w given a context c that represents its entire preceding context, and let p(w) be a 'noise' word distribution (e.g. a unigram distribution).

The NCE approach assumes that the word w is sampled from a mixture distribution 1 k+1 (p(w|c) + kp(w)) such that the noise samples are k times more frequent than samples from the 'true' distribution p(w|c).

Let y be a binary random variable such that y = 0 and y = 1 correspond to a noise sample and a true sample, respectively, i.e. p(w|c, y = 0) = p(w) and p(w|c, y = 1) = p(w|c).

Assume the distribution p(w|c) has the following parametric form: DISPLAYFORM0 such that w and c are d-dimensional vector representations of the word w and its context c and Z c is a normalization term.

Applying Bayes' rule, it can be easily verified that: DISPLAYFORM1 where σ() is the sigmoid function.

NCE uses Eq. (2) and the following objective function to train a binary classifier that decides which distribution was used to sample w: DISPLAYFORM2 such that w, c go over all the word-context co-occurrences in the learning corpus D and u 1 , ..., u k are 'noise' samples drawn from the word unigram distribution.

The normalization factor Z c is not a free parameter and to obtain its value, one needs to compute DISPLAYFORM3 where V is the word vocabulary.

The original NCE paper BID5 ) proposed to learn the normalization term during training and then use it to normalize the model at test time.

In language modeling applications, this computation is typically not feasible due to the exponentially large number of possible contexts.

Computing the value of Z c at test time is possible, though expensive due to the large vocabulary size.

BID14 found empirically that setting Z c = 1 at train time, which removes the explicit normalization constraint in the NCE formulation (1), didn't affect the performance of the resulting model.

At test time, to compute log p(w|c), they still had to normalize the score w · c + b w , by explicitly computing Z c over all the vocabulary words, in order to obtain a proper distribution.

We next present an alternative interpretation of the NCE language modeling algorithm as a low-rank matrix approximation.

This view of NCE makes the normalization factor redundant during training and explains the self-normalization property, as was empirically observed in later works BID17 ; BID19 ; BID3 ; BID15 ).Definition:

The Pointwise Conditional Entropy (PCE) matrix of a conditional word distribution p(w|c) is: pce(w, c) = log p(w|c)where w goes over the words in the vocabulary and c goes over all the left (preceding) side contexts.

The NCE modeling (1) can also be written as a matrix m(w, c) = log p nce (w|c) = w· c+b w −log Z c with the same dimensions as the PCE matrix.

Assuming that w and c are d-dimensional vectors, the rank of the matrix m is at most d + 2.Let p(w, c) be the joint distribution of words and their left side contexts.

The NCE score (3) can be viewed as a corpus-based approximation of the following expectation based score: DISPLAYFORM4 When we actually compute the NCE score based on a given corpus, we replace the expectation in the first term by averaging over the corpus and the expectation in the second term is replaced by sampling of negative examples from the word unigram distribution.

Theorem 1: The NCE score S nce (m) (4) obtains its global maximum at the PCE matrix.

In other words, S nce (m) ≤ S nce (pce) for every matrix m.

Proof: The claim can be easily verified by computing the derivative and set it to zero.

An alternative proof is based on the fact that the word2vec NEG cost function obtains its global maximum at the Pointwise Mutual Information (PMI) matrix.

BID9 ) showed that the function DISPLAYFORM5 obtains its global maximum when x = pmi k (w, c) = log p(w,c) kp(w)p(c) .

In our case it implies that the global maximum of (4) is obtained when m(w, c) − log(p(w)k) = pmi k (w, c).

Observing that pmi(w, c) = pce(w, c) − log(p(w)k) we complete the proof.

We next show that the value of the function S nce (m) at its maximum point, the PCE matrix, has a concrete interpretation.

The Kullback-Leibler (KL) divergence of a distribution p from a distribution q is defined as follows: KL(p||q) = i∈A p i log pi qi .

The Jensen-Shannon (JS) divergence BID10 ) between distributions p and q is: DISPLAYFORM6 such that 0 < α < 1, r = αp + (1 − α)q.

Unlike KL divergence, JS divergence is bounded from above and 0 ≤ JS α (p, q) ≤ 1.

It can be easily verified that the value of the NCE score with k negative samples (4) at the PCE matrix satisfies: DISPLAYFORM7 where α = 1 k+1 .

In other words the global optimum of the NCE score is the Jensen-Shannon divergence between the joint distribution of words and their left-side contexts and the product of their marginal distributions.

The NCE algorithm finds the best d-dimensional approximation m of the PCE matrix in the sense that it minimizes the difference (k+1) · JS α (p(w, c), p(c)p(w)) − S nce (m).In the traditional derivation of NCE, the parametric model is used to define a proper conditional probability: p nce (w|c) = 1 Zc exp( w· c+b w ).

Hence, to guarantee that we need a normalization factor Z c for each context c. In our NCE interpretation, the training goal is to find the best unnormalized low-rank approximation of the PCE matrix.

Hence, no normalization factor is involved.

Although, normalization is not explicitly included in our view of NCE, we have shown that even so, our model attempts to approximate the true conditional probabilities, which are normalized, and hence we can provide guarantees as to its self-normalization properties as we describe in the next section.

Finally, revisiting prior work, BID14 ; BID17 ; BID19 used Z = 1, while BID3 ; BID15 reported that using log(Z) = 9 at train time gave them the best results with their NCE implementation.

Setting a specific fixed value for Z would alter the mean input to the sigmoid function in the NCE score, which may ensure that it is closer to zero.

This can potentially improve training stability, convergence speed and performance in a way similar to batch normalization BID6 ).

However, we note that it is not related to distribution normalization.

At test time, when we use the model learned by NCE to compute the conditional probability p nce (w|c) (1), we need to compute the normalization factor: DISPLAYFORM0 Note that the NCE language model obtained from the PCE matrix by setting m(w, c) = pce(w, c) = log(w|c), is clearly self normalized: DISPLAYFORM1 Theorem 1 showed that the NCE algorithm finds the best low-rank unnornmalized matrix approximation of the PCE matrix.

Hence, we can assume that the matrix m is close to the PCE matrix and therefore the normalization factors of the LM based on m should be also close to 1: DISPLAYFORM2 We next formally show that if the matrix m is close to the PCE matrix then the NCE model defined by m is approximately self-normalized.

Theorem 2: Assume that for a given context c there is an 0 < such that |m(w, c) − pce(w, c)| ≤ for every w ∈ V .

Then | log Z c | ≤ .Proof: DISPLAYFORM3 Given the assumption that |m(w, c) − pce(w, c)| = | w · c + b w − log p(w|c)| ≤ , we obtain that: DISPLAYFORM4 The concavity of the log function implies that: DISPLAYFORM5 Combining Eq. FORMULA12 and Eq. FORMULA13 we finally obtain that | log Z c | ≤ .In the appendix we show that Theorem 2 remains true if we replace the assumption max w∈V |m(w, c) − pce(w, c)| ≤ by the weaker and more realistic assumption that log w∈V p(w|c) exp(|m(w, c) − pce(w, c)|) ≤ .To summarize, in our analysis we first show (Theorem 1) that the NCE training goal is to make the unnormalized score ( w · c + b w ) close to the normalized log p(w|c).

We then show (Theorem 2) that if the unnormalized score is indeed close to log p(w|c), then log Z c is close to zero.

Combining the two theorems, we obtain that the LM learned by NCE is self-normalized.

In this section we address two related language models and briefly describe how our analysis is related to their training strategies.

The standard LM learning method, which is based on a softmax output layer, is not self-normalized.

BID4 proposed to explicitly encourage self-normalization as part of its training objective function by penalizing deviation from selfnormalizing: DISPLAYFORM0 where Z c = v∈V exp( v · c + b v ) and α is a constant.

The drawback of this approach is that at least at train time you need to explicitly compute the costly Z c .

BID0 proposed an efficiently computed approximation of (10) by eliminating Z c in the first term and computing the second term only on a sampled subset D of the corpus D: DISPLAYFORM1 where γ is an additional constant that determines the sampling rate.

They also provided analysis that justifies computing Z c only on a subset of the corpus by showing that if a given LM is exactly self-normalized on a dense set of contexts (i.e. each context c is close to a context c s.t.

log Z c = 0) then E| log Z c | is small.

This could justifies computing Z c only on a small subset of contexts, but cannot explain why NCE training produces a self-normalized model without computing Z c on any context at all.

Importance sampling (IS) BID1 ) is an efficient alternative to a full softmax layer that is closely related to NCE.

In IS we assume that for each context c we are given k + 1 words w = u 0 , u 1 , ..., u k that were sampled in the following way.

First we sample a uniform random variable y ∈ {0, ..., k}. Then for y = i we sample u i according to p(·|c) and all the other k words are sampled from the unigram distribution: DISPLAYFORM2 Given the observation that for all the contexts y = 0, (i.e. u 1 , ..., u k are sampled from the word unigram distribution and u 0 = w is sampled from p(w|c)), the IS objective function is: DISPLAYFORM3 It can be easily verified that the normalization factor Z c is canceled out in the IS objective function.

Hence, there is no need to estimate it in training.

Unlike NCE, the network learned by IS was not found to be self-normalized.

As a result, explicit computation of the normalization factor is required at test time BID15 ).

In this section, we empirically investigate the self-normalization properties of NCE language modeling and its explicit self-normalization alternative.

We investigated LSTM-based language models with two alternative training schemes: (1) NCE-LM -the NCE language model described in Section 2; and (2) DEV-LM -the softmax language model Table 1 : Self-normalization and perplexity results of the self-normalizing NCE-LM against the nonself-normalizing standard softmax LM (SM-LM).

d denotes the size of the compared models (units).

DISPLAYFORM0 with explicit self-normalization proposed by BID4 .

Following BID4 , to make both DEV-LM and NCE-LM approximately self-normalized at init time, we initialized their output bias terms to b w = − log |V |, where V is the word vocabulary.

We set the negative sampling parameter for NCE-LM to k = 100, following BID19 , who showed highly competitive performance with NCE LMs trained with this number of samples, and BID11 .

Except where noted otherwise, other details of our implementation and choice of hyperparameters follow BID18 who achieved strong perplexity results using standard LSTM-based neural language models.

Specifically, we used a 2-layer LSTM with a 50% dropout ratio to represent the preceding (left-side) context of a predicted word.2 All models were implemented using the Chainer toolkit BID16 ).We used two language modeling datasets in the evaluation.

The first dataset, denoted PTB, is a version of the Penn Tree Bank, commonly used to evaluate language models.3 It consists of 929K/73K/82K training/validation/test words respectively and has a 10K word vocabulary.

The second dataset, denoted WIKI, is the WikiText-2, more recently introduced by BID12 .

This dataset was extracted from Wikipedia articles and is somewhat larger, with 2,088K/217K/245K train/validation/test tokens, respectively, and a vocabulary size of 33K.To evaluate self-normalization, we look at two metrics: (1) µ z = E(log(Z c )), which is the mean log value of the normalization term, across contexts c in a dataset C; and (2) σ z = σ(log(Z c )), which is the corresponding standard deviation.

The closer these two metrics are to zero, the more selfnormalizing the model is considered to be.

We note that a model with |µ z | >> 0 can potentially be 'corrected' (as we show later) by subtracting µ z from the unnormalized score.

However, this is not the case for σ z .

Therefore, from a practical point of view, we consider σ z to be the more important metric of the two.

In addition, we also look at the classic perplexity metric, which is considered a standard measure for the quality of the model predictions.

Importantly, when measuring perplexity, except where noted otherwise, we first perform exact normalization of the models' unnormalized scores by computing the normalization term.

Table 1 shows a range of results that we got on the validation sets when evaluating NCE-LM against standard softmax language model baseline (SM-LM).

4 Looking at the results, we can first see that consistently with previous works, NCE-LM is approximately self-normalized as apparent by relatively low |µ z | and σ z values.

On the other hand, SM-LM, as expected, is far from being selfnormalized.

In terms of perplexity, we see that SM-LM performs a little better when model dimensionality is low, but the gap closes entirely at d = 650.

Curiously, while perplexity improves with Table 2 : Self-normalization and perplexity results of the self-normalizing DEV-LM for different values of the normalization factor α.

d denotes the size of the compared models (units).

Table 3 : Self-normalization and perplexity results on test sets for 'shifted' models with d = 650. 'u-perp' denotes unnormalized perplexity.

higher dimensionality, we see that the quality of NCE-LM's self-normalization, as evident particularly by σ z , actually degrades.

This is surprising, as we would expect that stronger models with more parameters would approximate p(w|c) more closely.

We further investigate this behavior in Section 5.3.Next, Table 2 compares the self-normalization and perplexity performance of DEV-LM for different values of the constant α on the validation sets.

As could be expected, the larger the value of α is, the better the self-normalization becomes, reaching very good self-normalization for α = 10.0.

On the other hand, the improvement in self-normalization seems to occur at the expense of perplexity.

This is particularly true for the smaller models, 5 but is still evident even for d = 650.

Interestingly, as with NCE-LM, we see that σ z is growing (i.e. self-normalization becomes worse), with the size of the model, and is negatively correlated with the improvement in perplexity.

Finally, for the test-set evaluation, we propose a simple technique to center the log(Z) values of a self-normalizing model's scores around zero.

Let µ z be E(log(Z)) observed on the validation set at train time.

The probability estimates of the 'shifted' model are log p(w|c) = w · c + b w − µ z .

Table 3 shows the results that we get when evaluating the shifted NCE-LM and DEV-LM models with d = 650.

For DEV-LM, we chose α = 1.0, which seems to provide an optimal trade-off between self-normalization and perplexity performance on the validation sets.

Following BID15 , in addition to perplexity, we also report 'unnormalized perplexity', which is computed with the unnormalized conditional probability estimates.

When the perplexity measure is close to the unnormalized perplexity, this suggests that the unnormalized estimates are in fact nearly normalized.

As can be seen, with the shifting method, both models achieve near perfect (zero) µ z value, and their unnormalized perplexities are almost identical to their respective real perplexities.

Also, comparing the perplexities of NCE-LM to those of DEV-LM, we see near identical performance.

The standard deviation of the normalization term of DEV-LM is notably better than that of NCE-LM.

However, we note that NCE's advantage is that unlike DEV-LM, it doesn't include the normalization term in its training objective, and therefore its training time does not grow with the size of the vocabulary.

Table 4 : Pearson's correlation between entropy and log(Z) on samples from the validation sets.

DISPLAYFORM0 Figure 1: The normalization term of a predicted distribution as a function of its entropy on a sample from the WIKI validation set.

The entropy of the distributions predicted by a language model is a measure of how uncertain it is regarding the identity of the predicted word.

Low-entropy distributions would be concentrated around few possible words, while high-entropy ones would be much more spread out.

To more carefully analyze the self-normalization properties of NCE-LM, we computed the Pearson's correlation between the entropy of a predicted distribution H c = − v p(v|c) log p(v|c) and its normalization term, log(Z c ).

As can be seen in Table 4 , it appears that a regularity exists, where the value of log(Z c ) is negatively correlated with entropy.

Furthermore, it seems that, to an extent, the correlation is stronger for larger models.

To illustrate this phenomenon, we plot a sample of the predicted distributions in Figure 1 .

We can see there in particular, that low entropy distributions can be associated with very high values of log(Z c ), deviating a lot from the self-normalization objective of log(Z c ) = 0.

Examples for contexts for which NCE-LM predicts such distributions are: "During the American Civil [War]" and "The United [States]", where the actual word following the preceding context appears in parenthesis.

We hypothesize that this observation could be a contributing factor to our earlier finding that larger models have larger variance in their normalization terms, though it seems to account only for some of that at best.

Furthermore, we hope that this regularity could be exploited to improve self-normalization algorithms in the future.

We provided theoretical justification to the empirical observation that NCE is self-normalizing.

Our empirical investigation shows that it performs reasonably well, but not as good as a language model that is explicitly trained to self-normalize.

Accordingly, we believe that an interesting future research direction could be to augment NCE's training objective with some explicit self-normalization component.

In addition, we revealed unexpected correlations between self-normalization and perplexity performance, as well as between the partition function of self-normalized predictions and the entropy of the respective distribution.

We hope that these insights would be useful in improving self-normalizing models in future work.

<|TLDR|>

@highlight

We prove that NCE is self-normalized and demonstrate it on datasets

@highlight

Presents a proof of the self normalization of NCE as a result of being a low-rank matrix approximation of low-rank approximation of the normalized conditional probabilities matrix.

@highlight

This paper considers the problem of self-normalizing models and explains the self-normalizing mechanism by interpreting NCE in terms of matrix factorization.