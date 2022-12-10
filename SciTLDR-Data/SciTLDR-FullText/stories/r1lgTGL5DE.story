REINFORCE can be used to train models in structured prediction settings to directly optimize the test-time objective.

However, the common case of sampling one prediction per datapoint (input) is data-inefficient.

We show that by drawing multiple samples (predictions) per datapoint, we can learn with significantly less data, as we freely obtain a REINFORCE baseline to reduce variance.

Additionally we derive a REINFORCE estimator with baseline, based on sampling without replacement.

Combined with a recent technique to sample sequences without replacement using Stochastic Beam Search, this improves the training procedure for a sequence model that predicts the solution to the Travelling Salesman Problem.

REINFORCE (Williams, 1992 ) is a well known policy optimization algorithm that learns directly from experience.

Variants of it have been used to train models for a wide range of structured prediction tasks, such as Neural Machine Translation BID12 BID0 , Image Captioning (Vinyals et al., 2015b) and predicting solutions (tours) for the Travelling Salesman Problem (TSP) BID1 BID6 .

As opposed to maximum likelihood (supervised) learning, the appeal of using REINFORCE for structured prediction is that it directly optimizes the test-time performance.

When using REINFORCE, often for each datapoint (e.g. a sentence, image or TSP instance) only a single sample/prediction (e.g. a translation, caption or tour) is used to construct a gradient estimate.

From a classic Reinforcement Learning (RL) point of view, this makes sense, as we may not be able to evaluate multiple sampled actions for a state (datapoint).

However, from a data point of view, this is inefficient if we can actually evaluate multiple samples, such as in a structured prediction setting.

Reinforcement Learning with multiple samples/predictions for a single datapoint has been used before (e.g. BID14 ; ), but we use the samples as counterfactual information by constructing a (local, for a single datapoint) REINFORCE baseline.

A similar idea was applied for variational inference by BID10 .Many structured prediction tasks can be formulated in terms of sequence modelling, which is the focus of this paper.

In most sequence modelling tasks, the objective is a deterministic function of the predicted sequence.

As a result, duplicate sampled sequences are uninformative and therefore do not improve the quality of the gradient estimate.

To solve this problem, we propose to use sampling without replacement to construct a better gradient estimate.

This is inspired by recent work by BID7 , who introduce Stochastic Beam Search as a method to sample sequences without replacement, and use this to construct a (normalized) importance-weighted estimator for (sentence level) BLEU score.

We extend this idea to estimate policy gradients using REINFORCE, and we show how to use the same set of samples (without replacement) to construct a baseline.

This way we can leverage sampling without replacement to improve training of sequence models.

In our experiment, we consider the TSP and show that using REINFORCE with multiple samples is beneficial compared to single sample REINFORCE, both computationally and in terms of data-efficiency.

Additionally, for a sample size of 4 − 8 samples per datapoint, sampling without replacement results in slightly faster learning.

The REINFORCE estimator (Williams, 1992) allows to estimate gradients of the expectation E y∼p θ (y) [f (y)] by the relation: DISPLAYFORM0 If we also have a context, or datapoint, x (such as a source sentence), we may write p θ (y|x) and f (y, x), but in this paper, we leave dependence on x implicit.

Extension of the derived estimators to a minibatch of datapoints x is straightforward.

Typically, we estimate the expectation using samples y 1 , ..., y k and we may reduce variance of the estimator by using a baseline B i that is independent of the sample y i (but may depend on the other samples y j , j = i): DISPLAYFORM1 In practice, often a single sample y is used (per datapoint x, as we already have a batch of datapoints) to compute the estimate, e.g. k = 1, but in this paper we consider k > 1.

In this paper, we consider a parametric distribution over discrete structures (sequences).

Enumerating all n possible sequences as y 1 , ..., y n , we indicate with y i the i-th possible outcome, which has log-probability φ i = log p θ (y i ) defined by the model.

We can use the Gumbel-Max trick BID4 BID9 to sample y according to this distribution as follows: let G i ∼ Gumbel (a standard Gumbel distribution) for i = 1, ..., n i.i.d., and let y = y i * , where DISPLAYFORM0 For a proof we refer to BID9 .

In a slight abuse of notation, we write G φi = φ i + G i , and we call G φi the (Gumbel-) perturbed log-probability of y i .The Gumbel-Max trick can be extended to the Gumbel-Top-k trick BID7 to draw an ordered sample without replacement, by taking the top k largest perturbed log-probabilities (instead of just one, the argmax).

The result is equivalent to sequential sampling without replacement, where after an element y is sampled, it is removed from the domain and the remaining probabilities are renormalized.

The Gumbel-Top-k trick is equivalent to Weighted Reservoir Sampling BID3 , as was noted by BID15 .

The ordered sample is also known as a partial ranking according to the Plackett-Luce model BID11 BID8 .For a sequence model with exponentially large domain, naive application of the Gumbel-Top-k trick is infeasible, but an equivalent result can be obtained using Stochastic Beam Search BID7 .

This modification of beam search expands the k partial sequences with maximum (Gumbel) perturbed log-probability, effectively replacing the standard top k operation by sampling without replacement.

The resulting top k completed sequences are a sample without replacement from the sequence model, by the equivalence to the Gumbel-Top-k trick.

For details we refer to BID7 .

For many applications we need to estimate the expectation of a function f (y), where y is the realization of a variable with a discrete probability distribution p θ (y).

When using Monte Carlo (MC) sampling (with replacement), we write y i to indicate the i-th sample in a set of samples.

In contrast, when sampling without replacement we find it convenient to write y i (with superscript i) to refer to the i-th possible value in the domain, so (like we did in Section 2.2) we can enumerate the domain with n possible values as y 1 , ..., y n .

This notation allows us to write out the expectation of f (y): DISPLAYFORM0 Published at the ICLR 2019 workshop: Deep RL Meets Structured Prediction Using MC sampling with replacement, we estimate equation 3 using k samples y 1 , ..., y k : DISPLAYFORM1 When sampling without replacement using the Gumbel-Top-k trick (Section 2.2) we write S as the set of k largest indices of G φi (i.e. S = arg top k{G φi : i ∈ {1, ..., n}}), so the sample (of size k) without replacement is {y i : i ∈ S}. We can use the sample S with the estimator derived by BID16 , based on priority sampling BID2 .

This means that, to correct for the effects of sampling without replacement, we include importance weights DISPLAYFORM2 Here κ is the (k + 1)-th largest value of {G φi : i ∈ {1, ..., n}}, i.e. the (k + 1)-th largest Gumbel perturbed log-probability, and q θ,a (y DISPLAYFORM3 ) is the probability that the perturbed log-probability of y i exceeds a. Then we can use the following estimator: DISPLAYFORM4 This estimator is unbiased, and we include a copy of the proof by BID7 (adapted from the proofs by BID2 and Vieira FORMULA0 ) in Appendix A, as this introduces notation and is the basis for the proof in Appendix C.Intuition behind this estimator comes from the related threshold sampling scenario, where instead of fixing the sample size k, we fix the threshold a and define a variably sized sample S = {i ∈ {1, ..., n} : G φi > a}. With threshold sampling, each element y i in the domain is sampled independently with probability P (G φi > a) = q θ,a (y i ), and DISPLAYFORM5 is a standard importance weight.

As it turns out, instead of having a fixed threshold a, we can fix the sample size k and use κ as empirical threshold (as i ∈ S if G φi > κ), and still obtain an unbiased estimator BID2 BID16 .As was shown by BID7 , in practice it is preferred to normalize the importance weights to reduce variance.

This means that we compute the normalization W (S) = i∈S p θ (y i ) q θ,κ (y i ) and obtain the following (biased) estimator: DISPLAYFORM6

Typically REINFORCE is applied with a single sample y per datapoint x (e.g. one translation per source sentence, or, in our experiment, a single tour per TSP instance).

In some cases, it may be preferred to take multiple samples y per datapoint x as this requires less data.

Taking multiple samples also gives us counterfactual information which can be used to construct a strong (local) baseline.

Additionally, we obtain computational benefits, as for encoder-decoder models we can obtain multiple samples using only a single pass through the encoder.

With replacement, we can use the estimator in equation 2, where we can construct a baseline B i for the i-th term based on the other samples j = i: DISPLAYFORM0 The form in equation 8 is convenient for implementation as it allows to compute a fixed 'baseline' B = 1 k k j=1 f (y j ) once and correct for the bias (as B depends on y i ) by normalizing using DISPLAYFORM1 For details and a proof of unbiasedness we refer to Appendix C.

The basic REINFORCE without replacement estimator follows from combining equation 1 with equation 5 for an unbiased estimator: DISPLAYFORM0 Similar to equation 6, we can compute a lower variance but biased variant by normalizing the importance weights using the normalization W (S) = i∈S DISPLAYFORM1 .

When sampling without replacement, the individual samples are dependent, and therefore we cannot simply define a baseline based on the other samples as we did in Section 3.1.

However, similar to the 'baseline' DISPLAYFORM2 based on the complete sample S (without replacement), using equation 5: DISPLAYFORM3 Using this baseline introduces a bias that we cannot simply correct for by a constant term (as we did in equation 8), as the importance weights depend on y i .

Instead, we weight the individual terms by DISPLAYFORM4 This estimator is unbiased and we give the full proof in Appendix C.For the normalized version, we use the normalization W (S) = i∈S p θ (y i ) q θ,κ (y i ) for the baseline, and DISPLAYFORM5 to normalize the outer terms: DISPLAYFORM6 It seems odd to normalize the terms in the outer sum by

We consider the task of predicting the solution for instances of the Travelling Salesman Problem (TSP) BID17 BID1 BID6 .

The problem is to find the order in which to visit locations (specified by their x, y coordinates) to minimize total travelling distance.

A policy is trained using REINFORCE to minimize the expected length of a tour (sequence of locations) predicted by the model.

The Attention Model by BID6 is a sequence model that considers each instance as a fully connected graph of nodes which are processed by an encoder.

The decoder then produces the tour as a sequence of nodes to visit, one node at a time, where it autoregressively uses as input the node visited in the previous step. .

REINFORCE is used with replacement (WR) and without replacement (WOR) using k = 4 (top row) or k = 8 (bottom row) samples per instance, and a local baseline based on the k samples for each instance.

We compare against REINFORCE using one sample per instance, either with a baseline that is the average of the batch, or the strong greedy rollout baseline by BID6 that requires an additional rollout of the model.

We use the source code by BID6 1 to reproduce their TSP experiment with 20 nodes (as larger instances diminish the benefit of sampling without replacement).

We implement REIN-FORCE estimators based on multiple samples, either sampled with replacement (WR) or without replacement (WOR) using Stochastic Beam Search BID7 .

We compare the following four estimators:• Single sample with a batch baseline.

Here we compute the standard REINFORCE estimator (equation 2) with a single sample (k = 1).

We use a batch of 512 instances (datapoints) and as baseline we take the average of the tour lengths in the batch, hence each instances uses the same baseline.

This is implemented as using the exponential moving average baseline by BID6 with β = 0.• Single sample with a greedy rollout baseline, and batch size 512.

As baseline, we use a greedy rollout: for each instance x we take the length of the tour that is obtained by greedily selecting the next location according to an earlier (frozen) version of the model.

This baseline, similar to self-critical training BID13 , corresponds to the best result found by BID6 , superior to using an exponential moving average or learned value function.

However, the greedy rollout requires an additional forward pass through the model.• Multiple samples with replacement (WR) with a local baseline.

Here we compute the estimator in equation 8 based on k = 4, 8 samples.

We use a batch size of 512 k , so the total number of samples is the same.

The baseline is local as it is different for each datapoint, but it does not require additional model evaluations like the greedy rollout.• Multiple samples without replacement (WOR) with a local baseline.

Here we use the (biased) normalized without replacement estimator in equation 11 with k = 4, 8 samples and batch size 512 k .

Samples are drawn without replacement using Stochastic Beam Search BID7 .

For fair comparison, we do not take a (k + 1)-th sample to compute κ, but sacrifice the k-th sample and compute the summation in equation 11 with the remaining k − 1 (3 or 7) samples.

Note that a single sample with a local baseline is not possible, which is why we use the batch baseline.

The model architecture and training hyperparameters (except batch size) are as in the paper by BID6 .

We present the results in terms of the validation set (not used for additional tuning) optimality gap during training in Figure 1 , using k = 4 (top row) and k = 8 (bottom row).

We found diminishing returns for larger k. The left column presents the results in terms of the number of gradient update steps (minibatches).

We see that sampling without replacement performs on par (k = 8) or slightly better than using the strong but computationally expensive greedy rollout baseline or using multiple samples with replacement.

The standard batch baseline performs significantly worse.

The estimators based on multiple samples do not lose (much) final performance, while using significantly less instances.

In the right column, where results are presented in terms of the number of instances, this effectiveness is confirmed, and we observe that sampling without replacement is preferred to sampling with replacement.

The difference is small, but there is also not much room for improvement as results are close to optimal.

The benefit of learning with less data may be small if data is easily generated (as in our setting), but there is also a significant computational benefit as we need significantly fewer encoder evaluations.

In this paper, we have derived REINFORCE estimators based on drawing multiple samples, with and without replacement, and evaluated the effectiveness of the proposed estimators in a structured prediction setting: the prediction of tours for the TSP.

The derived estimators yield results comparable to recent results using REINFORCE with a strong greedy rollout baseline, at greater data-efficiency and computational efficiency.

These estimators are especially well suited for structured prediction settings, where the domain is too large to compute exact gradients, but we are able to take multiple samples for the same datapoint, and the objective is a deterministic function of the sampled prediction.

We hope the proposed estimators have potential to be used to improve training efficiency in more structured prediction settings, for example in the context of Neural Machine Translation or Image Captioning, where depending on the entropy of the model, sampling without replacement may yield a beneficial improvement.

We include here in full the proof by BID7 , as this introduces necessary notation and helps understanding of the proof in Appendix C.A.1 PROOF OF UNBIASEDNESS OF PRIORITY SAMPLING ESTIMATOR BY KOOL ET AL.The following proof is adapted from the proofs by BID2 and BID16 .

For generality of the proof, we write DISPLAYFORM0 , and we consider general keys h i (not necessarily Gumbel perturbations).We assume we have a probability distribution over a finite domain 1, ..., n with normalized probabilities p i , e.g. n i=1 p i = 1.

For a given function f (i) we want to estimate the expectation DISPLAYFORM1 Each element i has an associated random key h i and we define q i (a) = P (h i > a).

This way, if we know the threshold a it holds that q i (a) = P (i ∈ S) is the probability that element i is in the sample S. As was noted by BID16 , the actual distribution of the key does not influence the unbiasedness of the estimator but does determine the effective sampling scheme.

Using the Gumbel perturbed log-probabilities as keys (e.g. h i = G φi ) is equivalent to the PPSWOR scheme described by BID16 .We define shorthand notation h 1:n = {h 1 , ..., h n }, h −i = {h 1 , ..., h i−1 , h i+1 , ..., h n } = h 1:n \{h i }.

For a given sample size k, let κ be the (k + 1)-th largest element of h 1:n , so κ is the empirical threshold.

Let κ i be the k-th largest element of h −i (the k-th largest of all other elements).Similar to BID2 we will show that every element i in our sample contributes an unbiased estimate of E[f (i)], so that the total estimator is unbiased.

Formally, we will prove that DISPLAYFORM2 from which the result follows: DISPLAYFORM3 To prove equation 12, we make use of the observation (slightly rephrased) by BID2 that conditioning on h −i , we know κ i and the event i ∈ S implies that κ = κ i since i will only be in the sample if h i > κ i which means that κ i is the k + 1-th largest value of h −i ∪ {h i } = h 1:n .

The reverse is also true (if κ = κ i then h i must be larger than κ i since otherwise the k + 1-th largest value of h 1:n will be smaller than κ i ).

DISPLAYFORM4

We will now prove that the REINFORCE estimator based on multiple samples with the sample average as baseline FORMULA10 is unbiased.

Let y 1:k = {y 1 , ..., y k } be the set of independent samples (with replacement) from p θ (y).

First we show that using the batch mean as baseline is equivalent to using the mean of the other elements in the batch, up to a constant DISPLAYFORM0 Note that k−1 k goes to 1 as the batch size k increases and we do not need to include it (and we can simply compute the biased mean) as it can be absorbed into the learning rate.

Since y j is independent of y i , unbiasedness follows: DISPLAYFORM1 The proof that the REINFORCE estimator based on multiple samples without replacement with baseline (equation 10) is unbiased follows from adapting and combining the proofs in Appendix A and B. Additionally to q i (a) = P (h i > a) we define q ij (a) = P (h i > a ∩ h j > a) = P (h i > a)P (h j > a) = q i (a)q j (a) for i = j and q ii (a) = P (h i > a) = q i (a).

For convenience we define shorthand for the conditional q j|i (a) = qij (a) qi(a) , so q j|i (a) = q j (a) for j = i and q i|i (a) = 1.

Furthermore, we define h −ij = h 1:n \ {h i , h j } and define κ ij (i = j) as the (k − 1)-th (not k-th!) largest element of h −ij , and κ ii = κ i , e.g. the k-th largest element of h −i .We denote with with {i, j ∈ S} = {i ∈ S ∩ j ∈ S} the event that both i and j are in the sample, also for i = j which simply means {i ∈ S}. First we generalize P (i ∈ S|h −i ) = q i (κ i ) to the pairwise conditional inclusion probability P (i, j ∈ S|h −ij ).

DISPLAYFORM2 Proof.

For i = j: DISPLAYFORM3 For i = j: Assuming w.l.o.g.

h i < h j there are the following scenario's:• κ ij < h i < h j .

In this case, after adding h i and h j to h −ij , κ ij will be the (k + 1)-th largest element so κ = κ ij and i ∈ S and j ∈ S since h j > h i > κ = κ ij .• h i < κ ij < h j or h i < h j < κ ij .

In both cases, there are at least (k − 1) + 1 = k elements higher than h i so i ∈ S.Therefore it follows that {i, j ∈ S|u −ij } = {h i > κ ij ∩ h j > κ ij |u −ij } and additionally this event implies κ = κ ij .

Now the result follows: DISPLAYFORM4 =P (h i > κ ij |u −ij )P (h j > κ ij |u −ij ) =q i (κ ij )q j (κ ij ) = q ij (κ ij )Using this Lemma we can prove the following Lemma: Lemma 2.

DISPLAYFORM5 Note that the expectation is w.r.t.

the keys h 1:n which define the random variables κ and S = {i : h i > κ}.Proof.

DISPLAYFORM6 u −ij , i, j ∈ S P (i, j ∈ S|u −ij ) + 0 · (1 − P (i, j ∈ S|u −ij )) DISPLAYFORM7 Theorem 1. Let B(S) = j∈S p θ (y j ) qj (κ) f (y j ).

Then the following is an unbiased estimator: DISPLAYFORM8 Proof.

First note that, when i ∈ S, we can rewrite: DISPLAYFORM9 Using equation 16, we can rewrite (similar to equation 13) DISPLAYFORM10 Substituting this into equation 15 and normalizing the outer importance weights by W (S) we see that this term cancels to obtain DISPLAYFORM11

@highlight

We show that by drawing multiple samples (predictions) per input (datapoint), we can learn with less data as we freely obtain a REINFORCE baseline.