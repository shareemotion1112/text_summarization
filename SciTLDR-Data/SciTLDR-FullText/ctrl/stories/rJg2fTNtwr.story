The exposure bias problem refers to the training-inference discrepancy caused by teacher forcing in maximum likelihood estimation (MLE) training for auto-regressive neural network language models (LM).

It has been regarded as a central problem for natural language generation (NLG) model training.

Although a lot of algorithms have been proposed to avoid teacher forcing and therefore to alleviate exposure bias, there is little work showing how serious the exposure bias problem is.

In this work, we first identify the auto-recovery ability of MLE-trained LM, which casts doubt on the seriousness of exposure bias.

We then develop a precise, quantifiable definition for exposure bias.

However, according to our measurements in controlled experiments, there's only around 3% performance gain when the training-inference discrepancy is completely removed.

Our results suggest the exposure bias problem could be much less serious than it is currently assumed to be.

Language model (LM) is a central module for natural language generation (NLG) tasks (Young et al., 2017) such as machine translation (Wu et al., 2017) , dialogue response generation , image captioning (Lin et al., 2014) , etc.

For decades, maximum likelihood estimation (MLE) has been the the most widely used objective for LM training.

However, there is a popular belief in the natural language processing (NLP) community that standard MLE training will cause "exposure bias" and lead to a performance degradation during the test-time language generation.

The exposure bias problem (Bengio et al., 2015; Ranzato et al., 2016) refers to the following discrepancy between MLE training and test-time generation for language models: During training, the language model predicts the next word conditioned on history words sampled from the groundtruth data distribution.

And during generation, the model generates words conditioned on history sequences generated by the model itself.

However, due to the exposure to real data during training, the language model is biased to only perform well on the ground-truth history distribution.

As a result, during generation the errors will accumulate along the generated sequence, and the distribution generated by the model will be distorted.

The forced exposure to ground-truth data during training is also referred to as "teacher forcing".

Given its defintion, the exposure bias problem could rise in the general cases when the model needs to make a sequence of decisions or generations (e.g. music/pixel/speech generation (Lamb et al., 2016) ).

In this work, we focus on the task of language generation, because the exposure bias problem is originally proposed in this field (Bengio et al., 2015) , and has since attracted huge research attention.

In order to avoid teacher forcing, many training algorithms (Bengio et al., 2015; Lamb et al., 2016; Ranzato et al., 2016; Yu et al., 2016; Zhu et al., 2018; Lu et al., 2018; Lin et al., 2017; Guo et al., 2017; Rajeswar et al., 2017; Wiseman & Rush, 2016; Nie et al., 2019; Shi et al., 2018) have been proposed as alternatives to MLE training.

Most of these works utilize techniques from generative adversarial network (GAN) (Goodfellow et al., 2014) or reinforcement learning (RL) (Sutton & Barto, 1998) .

In this paper, we refer to these algorithms as non-MLE methods or text GANs.

Despite the huge research efforts devoted to alleviate exposure bias, surprisingly, its existence or significance is much less studied.

In particular, to the best of our knowledge, no existing work Table 1 : Samples of a MLE-trained STOA transformer LM when fed with different types of length-10 history prefix.

To save space, we omitted the first 7 words of the random history.

attempts to directly show the seriousness of exposure bias in an empirical or theoretical way.

This work is motivated by the belief that a good solution should be built upon a testable and quantifiable problem definition.

In this rest of this paper, we first identify the "self-recovery" ability of popular LM models, which casts doubt on the original claim of exposure bias.

We then develop a precise and quantifiable definition of exposure bias, and validate its seriousness in controlled experiments.

To study the seriousness of exposure bias in standard MLE LM training, we first stress that the following methodology, although tempting, is wrong: If we can rigorously show that the non-MLE methods proposed to avoid teacher forcing do indeed bring solid generation performance gain, then we can conclude exposure bias is a meaningful problem for the original MLE training.

The reason is that we typically do not know the exact underlying reason for the performance gain.

For example, despite the huge success of the batch normalization technique in deep learning, whether "internal covariate shift" (which is the motivation of batch norm) exists in deep neural network training remains a question (Santurkar et al., 2018) .

Therefore, in this work we seek a direct way to validate the seriousness of exposure bias.

We focus on the following informal claim that immediately follows from the original definition of exposure bias: During generation, if we set the history distribution to be the ground-truth data distribution instead of the model's own distribution (now that there is no discrepancy between training and testing), then the model's language generation quality should be much better (we will formalize this notion in Section 4 and 5).

We start with the following qualitative analysis.

We feed a MLE-trained transformer LM on wiki-103 data-set (Baevski & Auli, 2018) with four kinds of prefixes: model's own samples, data samples, shuffled (word-level) data samples or samples from a uniform random distribution.

Then we let the model complete the sentence given these prefixes as history.

We list some samples in Table 1 and more in Appendix A (this experiment is also repeated for a LSTM LM).

Assuming the seriousness of exposure bias, we expect the quality of generated sentence-completion samples with real-data prefixes to be significantly better than the ones from prefixes of model samples.

However, by manual inspection, we do not observe noticeable differences in sample quality.

More surprisingly, the model is still able to generate relevant and fairly high-quality samples from shuffled prefixes.

Even in the extreme case where random sequences are fed, the model is able to generate reasonable sentences.

Due to the recent increasing interest of solving exposure bias in the field of neural machine translation (NMT) (Zhang et al., 2019) , we repeat the above experiment in a standard NMT setting in Appendix A, and get very similar observations.

These experiments clearly show that the MLE-trained auto-regressive LMs have the self-recovery ability, i.e. the model is able to recover from artificially distorted history input, and generate reasonably high-quality samples.

This phenomenon is clearly in contradiction with the popular claim of exposure bias, that the error induced by the mismatch between history and data distribution should accumulate during the generation process.

Motivated by these experiments, in the following sections, we turn to more rigorous methods to quantify the significance of exposure bias.

Note that our quantification approaches will be independent of the training procedure and only require inference from the trained model.

The task of auto-regressive language modelling is to learn the probability distribution of the (l + 1) th word W l+1 in a sentence conditioned on the word history W 1:l := (W 1 , . . .

, W l ).

Here, we use the uppercase W i ∈ V to denote a discrete random variable distributed across the vocabulary V .

The lower-case w is used to denote some particular word in V .

Given a training data-set D consisting of sentences of length L, the standard MLE training minimizes the negative log-likelihood below:

Note that in this work we assume all sentences are of length L for simplicity.

We denote the generation distribution of the trained LM as P M , and the ground-truth data distribution as P D .

Readers can assume P M refers to the generation distribution of a LSTM LM (Hochreiter & Schmidhuber, 1997; Sundermeyer et al., 2012) or a transformer LM (Baevski & Auli, 2018; Dai et al., 2019 ) trained with MLE objective, which is the major subject of this study.

We will mainly present results on LSTM based models to facilitate comparison with text-GAN works (listed in Section 1), which are mostly implemented on LSTM models.

We will also provide results with the transformer model, with very similar observations or measurements.

Our quantification mainly relies on the measurements of the distance from the model's generation distribution to the data distribution.

Hence we define the following notations to simplify expressions.

Let P denote the set of probability distributions on the vocabulary V .

Let d denote a distance measure between distributions (e.g. total variation distance), d : P × P → R ≥0 .

In this section, we propose an intuitive and seemingly correct quantification approach using marginal distributions.

The approach can be applied to real-world text data experiments, but it has some lethal weak point.

The discussion will lead us to our final precise definition of exposure bias in Section 5.

Assuming a given history length l, we consider the marginal distribution of W l+1 from the following three random process:

• Draw word sequences of length L from the data distribution P D .

Denote the marginal distribution of the random variable at position l + 1 (W l+1 ) as P l+1 D|D , where

• Draw word sequences of length L from the model distribution P M .

Denote the marginal distribution of the random variable at position l + 1 as

Under review as a conference paper at ICLR 2020

Denote the marginal distribution of the random variable at position l + 1 as P l+1 M |D , where

By the definition of exposure bias, P l+1 M |M suffers from the training-testing discrepancy, while P l+1 M |D should be closer to the true distribution P l+1 D|D .

To measure this discrepancy, define the marginal generation deviation (MGD) at history length l of history distribution P H with metric d as

where P H ∈ {P M , P D } denotes the history distribution.

MGD measures the deviation of the marginal distribution of W l+1 from ground-truth data distribution.

Finally, we define the rate of exposure bias (EB-M) at history length l of model P M as the ratio (discrepancy) between the MGD measurements when two different history distributions are fed:

For MLE-trained models, EB-M 1 is expected to be larger than 1, and larger EB-M indicates a more serious exposure bias problem for the trained model.

For the metric d, we consider two popular probability metrics: total variation distance (denoted as d T V ), and Jensen-Shannon divergence (denoted as d JS ).

In this section, we focus on answering the following question: "Does the EB-M measurement correctly reflect the significance of exposure bias?"

In short, our answer is not really.

The problem is that the distortion of the marginal P l+1 M |M is not only affected by the presumably existing exposure bias problem alone, but also by the mismatch between the history distribution P M from P D for W 1:l , which grows with the length of the history.

Therefore, even if the measured EB-M is significantly larger than one, we can not conclude that exposure bias causes serious deterioration.

We provide an example to illustrate this argument: Example 1.

Suppose L = 2, and V = {A, B}. P D and P M are crafted as follows:

However, the only problem P M has is the mismatch between the history distributions (P M and P D ) for W 1 .

The next set of experiments also suggest that EB-M does not precisely reflect exposure bias.

On the EMNLP-news data-set (specified in Appendix B), we compare EB-M measurements for several non-MLE training methods with the baseline MLE model.

We include results for Scheduled Sampling (SS) (Bengio et al., 2015) , Cooperative Training (CoT) (Lu et al., 2018) , and Adversarial Ranking (RankGAN) (Lin et al., 2017) .

We provide implementation details for non-MLE methods in Appendix C. Intuitively, these methods will cause the model to be biased to behave well with model samples as history, instead of data samples.

Therefore, we expect EB-M measurement for non-MLE trained models to be smaller than MLE trained models.

However, Figure 1 shows that the measurements for different training frameworks are almost the same.

We believe the reason is that the EB-M measurements are only reflecting the trivial mismatch between the history distributions.

Is it possible that the original definition of exposure bias (Bengio et al., 2015; Ranzato et al., 2016) exactly refers to this mismatch between the model and data history distributions?

However, note that this mismatch is inevitable for any imperfect model, and non-MLE training algorithms can not solve it.

We believe a better, more precise definition is needed to discriminate exposure bias from this trivial mismatch.

Motivated by this view, we propose a second approach in the section below.

Following the discussion in the last section, we wish our measurement to be independent of the quality of the history distribution.

In light of that, we design a quantity to measure the model's conditional generation quality.

Let P H ∈ {P M , P D } denote the history distribution as in the MGD definition (5).

With history length l fixed, we define the conditional generation deviation (CGD) with history distribution P H for P M using metric d as:

where we assume that P D (· | W 1:l )) is computable, and use it to measure the quality of the model's conditional distribution.

For the choice of the distribution distance d, in addition to d T V and d JS , we introduce greedy decoding divergence (d GD ) defined as:

where 1 is the indicator function, and P, Q ∈ P. The distance d GD 2 reflects the model's accuracy during greedy decoding.

Similar to MGD, exposure bias should imply a significant gap between CGD(P M |M , l, d) and CGD(P M |D , l, d).

We again define rate of exposure bias at history length l with metric d to be:

For our definition of EB-C, a natural question is why we only focus on the generation distribution of the very next word.

The reason is we want to precisely measure how the error caused by the history part affect the generation part, by keeping them separate.

If we measure the deviation of, for example, two sampled tokens, the definition will be confusing: Because the second sampled token will be affected not only by the accumulated error induced by the history (sampled from the model), but also by the first generated token as history.

To get a better understanding of the intuition behind the definition of EB-C, we recommend readers to read Appendix A about our NMT experiment.

Since CGD requires inference for ground-truth data distribution P D , we first consider experiments in a synthetic setting.

In text-GAN literature (Yu et al., 2016; Lin et al., 2017) , a randomly-initialized one-layer LSTM model with hidden dimension of 32 is usually used as P D in synthetic experiments (we denote this setting as M random 32

).

However, the model is small-scale and does not reflect any structure existing in real-world text.

To improve upon this approach, we take the MLE baseline model trained on EMNLP-news data (described in Appendix B) as P D in this synthetic setting.

We denote the data model (P D ) as M news 512 .

We then train two LSTM LM (P M ) with different capacities using samples from the data model, with the standard MLE objective.

One is a one-layer LSTM with hidden width of 512 (denoted as LSTM-512), the other one is with hidden width of 32 (denoted as LSTM-32).

We train P M for 100 epochs using the Adam optimizer with learning rate 0.001.

In each epoch, 250k sentences (same to the size of the original EMNLP-news data) of length L = 50 are sampled from M news-512 as training data to avoid over-fitting.

We show perplexity (PPL) results of the trained models in Appendix F. Finally, EB-C is calculated using 100k 3 samples from P M and P D .

In Figure 2 , we show EB-C measurements with different metrics d m , and the two models give similar results.

It is shown that EB-C has a steady but slow increasing trend as history length increases.

This is expected as a consequence of exposure bias, because P M deviates farther from P D as history length increases.

However, the average value of EB-C is less than 1.03 (the largest average value is from d JS for the LSTM-512 experiment), meaning that the gap between CGD(P M |M , l, d) and CGD(P M |D , l, d) is not large.

Also, note that in most NLG applications (such as machine translation or image captioning), the generated sequence typically has short length (less than 20).

In that range of history length, the EB-C measurements that exposure bias only has minimal influence.

In Appendix E, we repeat the experiment for a transformer LM (Dai et al., 2019) , and get very similar EB-C measurements.

These measurements imply a striking conclusion : (Informal) Even if all the bad effects from exposure bias for MLE LM training are removed, the relative performance gain is at most 3%.

If the sequence length is not very long, the gain is less than 1%..

To dive deeper into the cause of the gap in CGD, we experiment with corrupted versions of P M as history distribution.

We first specify a corrupt rate c ∈ [0, 1], and randomly substitute words in a history sample from P M to a "noise" word drawn uniformly from the vocabulary with probability c. Consequently, larger c will cause the history distribution to deviate farther from the groundtruth P D .

In Figure 3 , we show CGD measurement versus the corrupted history P corrupt M .

Large gaps are observed between CGD(P M |M corrupt ) and CGD(P M |D ).

Therefore, the small gap between CGD(P M |M ) and CGD(P M |D ) in Figure 2 results from the small deviation between the history distribution P M and P D .

In other word, P M has learned a "good enough" distribution that is able to keep it in the well-behaving region during sampling.

With these observations, we conclude that, in the synthetic setting considered, exposure bias does exist, but is much less serious than it is presumed to be.

Although there exists mismatch between the history distribution P M and P D , the mismatch is still in the model's "comfortable zone".

In other words, the LSTM LM is more robust than exposure bias claims it to be.

To concretize the this argument, we provide an example LM and show that MLE training is unlikely to generate models with a large EB-C value.

Example 2.

Again suppose L = 2, and V = {A, B}, the ground-truth data distribution is uniform on {AA, AB, BB, BA}. P M is crafted as follows:

.

Note that the model behaves bad when W 1 = A, which is of high probability during sampling.

However, this crafted model is unlikely to be an outcome of MLE training.

The fact that P M (·|W 1 = B) is better modeled indicates that in the training data more sentences begin with W 1 = B than W 1 = A. So MLE training should assign more probability to P M (W 1 = B), not the other way around 4 .

setting 5 , as we find it hard to do a fast implementation of RankGAN for the LSTM-512 setting.

We find that RankGAN and CoT gives lower EB-C measurements than MLE, which is expected, as these methods avoid teacher forcing.

For CoT, at short 4 If we change to PM (W1 = A) = 0.1, then EB-C(PM , 1, dT V ) will be 0.2, meaning that the model has better conditional generation performance during sampling 5 The MLE model is used as the pre-trained model for the RankGAN generator.

The MLE model has an oracle NLL of 8.67, and RankGAN's oracle NLL is 8.55.

Table 2 : An illustration for the next word collection process.

The choices are shuffled.

The first history sample is from real data, and the second history sample is from the trained model.

Table 3 : EB-C measurements with human as P D .

hisotry length, EB-C is even less than 1.

We believe the reason is that CoT trys to make the model be biased to behave better when fed with model samples.

However, SS gives worse EB-C measurements comparing to MLE, which we currently do not have a good explanation.

We refer readers to Huszr (2015) for a discussion about the SS objective.

To the best of our knowledge, this is the first direct empirical evidence that text GAN does indeed alleviate the exposure bias problem.

It also indicates that EB-C correctly reflect the significance of exposure bias.

We believe the reason for why EB-C is still not less than 1 is that, text GANs still rely on MLE pre-training a lot.

In this section, we design experiments to efficiently estimate EB-C for a SOTA transformer LM with real human as P D , by utilizing the Amazon Mechanical Turk (AMT) platform.

Given a MLE-trained LM as P M , by examining the definition of EB-C in Equation 9 and 7, it is clear the only obstacle is that we don't have access to P D (· | W 1:l ) with a given history W 1:l .

So, in this section, we focus on the greedy decoding divergence (d GD ) metric (Equation 8), which only requires the turkers to give the most probable next word prediction, instead of the full distribution (which is clearly intractable).

In our preliminary trials, we find it is still very hard for a person to guess the next word, even with real data history samples.

The reason is that the vocabulary is very big, and the turkers may be not familiar with the context (e.g. wikipedia).

To alleviate that problem, we design the following simplification: For a given history, we let the model output its top-5 next word prediction, then we only ask the turkers to choose among the 5 choices (the turker can also express that he/she thinks none of them is likely).

Finally, we examine whether the turker's choice is indeed the model's top-1 prediction.

We illustrate this process in Table 2 .

We use the code of Transformer-XL (Dai et al., 2019) to train a SOTA transformer LM on the wiki-103 data-set.

We favour the wiki-103 data-set because it is large-scale and has long (over 30 words) paragraphs, which is useful for the measurements of exposure bias.

The model is a 16-layer transformer-xl model with hidden dimension of 410.

Since the estimation of CGD(P M |D , l, d) requires large amounts of unseen real data samples, we use half of the wiki-103 training data (around 900k sentences and 50m words) to train the model P M , and save the other half as samples from P D .

Other training configurations ( learning rate, batch size, etc.) are not changed 6 .

The resulting model P M has a test-set PPL of 27.81 (if trained on full training data, the PPL will be 24.02).

We collect data to estimate EB-C at history length 10, 20, and 30.

For each length and history model (P M or P D ) pair, we collect 10k d GD samples (via next-word prediction) from turkers on the AMT platform.

More details about the AMT setup are provided in Appendix D. The results are shown in Table 3 .

The EB-C measurements are strikingly similar to the results in our synthetic experiments in that, removing the training-testing discrepancy only gives around 2% of relative performance gain.

This result further strengthens our claim that exposure bias is only a minor problem for MLE-based LM training.

Several recent works attempt to carefully evaluate whether the non-MLE training methods (e.g. adversarial training) can give superior NLG performance than standard MLE training for RNN LM.

Caccia et al. (2018) tunes a "temperature" parameter in the softmax output, and evaluate models over the whole quality-diversity spectrum.

Semeniuta et al. (2018) proposes to use "Reverse Language Model score" or "Frechet InferSent Distance" to evaluate the model's generation performance.

Tevet et al. (2018) proposes a method for approximating a distribution over tokens from a GAN, and then evaluate the model with standard LM metrics.

These works arrive at a similar conclusion: The general performance of Text GANs is not convincingly better, or even worse, than standard MLE training.

Hence to some extent, they imply that exposure bias may be not a serious problem in MLE training.

However, as we argued in Section 2, one can not draw direct conclusions about exposure bias with these results.

For example, it is also possible that exposure bias is indeed serious for MLE training, but text GAN does not solve the problem well enough.

In this work, we first identify the self-recovery ability of MLE-trained LM, which casts doubt on the seriousness of exposure bias, which has been regarded as a central problem for MLE training by the LM community.

We then explore two intuitive approaches to quantify the significance of exposure bias for LM training.

The first quantification EB-M relies on the marginal generation distribution and reveals some vagueness in the original definition of exposure bias.

We argue that we should focus on the model's generation performance in terms of its conditional distribution and propose a second quantification EB-C, which we regard as the precise definition for exposure bias.

We design a evaluation of EB-C at different history length with real human (turkers from AMT) as the data model, for a SOTA transformer LM.

It is shown that removing the training-testing discrepancy only gives around 2% of performance gain.

Our synthetic experiments also gives very similar measurements.

By analyzing EB-C measurements with perturbed history samples, we hypothesise that although the mismatch between the data and model distribution for history prefix exists, it is still in the model's "comfortable zone".

With these results, we claim that on the contrary to the popular belief, exposure bias is only a minor problem in MLE-based LM training.

To wrap up, we discuss the fundamental question "Is MLE training really biased?", from the perspective of objective functions.

Note that the MLE objective (1) can be re-written as:

where D KL denotes the Kullback-Leibler divergence, and θ denotes the trainable parameters in P M .

Therefore, MLE training is minizing the divergence from P M , which is exactly the model's sampling distribution, from P D .

While it's true that the training is "exposed" to data samples, we can not simply deduce the objective is "biased".

We want to end our discussion with two remarks.

First, the proposed quantification approaches should not be used as the only metric for NLG.

For example, a position-aware uni-gram LM, which generates words independent of previous context, has no exposure bias problem and can pass our test easily.

Second, the intention of this work is not to discourage researchers from exploring non-MLE training algorithms for LM.

It is completely possible that an training objective different from

, can lead to better generation performance (Lu et al., 2018; Huszr, 2015) .

However, though non-MLE algorithms avoid teacher forcing, these algorithms (using GAN or RL for example) are usually less stable and more difficult to tune.

Given that the quantified measurement of exposure bias is insignificant, we think it should be questioned whether adopting these techniques to avoid exposure bias is a wise trade-off.

In Table 6 , we provide more samples of a MLE-trained transformer LM model (discussed in Section 2) when fed with different kinds of history.

And in Table 7 we repeat the experiment for a LSTM-LM trained on the EMNLP-News data.

In Table 4 we repeat the preliminary experiment in Section 2 for a standard NMT setting.

We train a 6-layer transformer model with hidden dimension 1024 on the IWSLT'14 German to English data set.

We feed the trained model with types of prefix during decoding which represents different level of training-decoding discrepancy.

Note that the source input is kept intact.

The result is very similar (or more striking) to our language model experiment, the data prefix does not seem to help, and in the extreme case of random prefix, the model still generates fairly good translation.

In Section 2 we summarize this observation as the auto-recovery ability.

To interpret the UNREL3 results, we should not directly compare the translation generated from unrelated prefix to the target translation.

In fact, we cannot even compare part of it (e.g. the part after the length-3 prefix).

Instead, we highlight the surprising fact that although the model is forced to begin (conditioned) with a wrong prefix, it still comes up with a reasonable translation.

This is not an Table 4 : A standard NMT transformer model fed with different types of length-3 history prefix.

We did not do any cherry picking.

The "@@" is because BPE tokenization is used.

"DATA" means the first three output tokens are forced to be correct.

"NORMAL" means no prefix is forced during decoding.

"UNREL" means the first three tokens are forced to be from another random unrelated sentence (which is wrong but grammatical).

"RAND" means the first three tokens are completely random words.

easy task even for human translators, yet the model does fairly well.

Again, this contradicts with the "exposure bias" hypothesis that a MLE-trained LM will produce a increasingly deviated sequence when initiated with a non-perfect prefix.

Actually, during generation the model self-corrects the error in the prefix.

It is also the major motivation of our proposed EB-C measurement (Section 5), which is based on the view of measuring distances between conditional distributions.

One problem in the implementation of EB-M is to estimate the described marginal distributions of W l+1 .

We adopt a simple sample-and-count method: P l+1 D|D is estimated by the distribution (histogram) of W l+1 from a number (to be specified below) of sentences sampled from the data distribution.

For P l+1 M |M and P l+1 M |D , we first draw a number of history samples W 1:l from the corresponding history model (model distribution and data distribution respectively).

We then feed sampled history sequences into the trained model and estimate the marginal distribution of the (l + 1) th word by averaging the predicted distribution P M (·|W 1:l ).

We measure EB-M for MLE-trained LSTM LM on two popular data-sets: EMNLP-news (EMNLP 2017 WMT News Section), and wikitext-103 7 .

For EMNLP-news we set L = 20, and only use data samples whose length is longer than L. The resulting training/validation/test set has 268k/10k/10k sentences.

The vocabulary is of size 5k.

We use the 10k samples in the test set for evaluation of EB-M. Note that the EMNLP-news data-set is widely used in text GAN literatures Yu et al. (2016); Lu et al. (2018) .

We train a one-layer LSTM LM (Sundermeyer et al., 2012) of hidden dimension 512 as the MLE baseline model for EMNLP-news.

For wikitext-103, we set L = 50, and regard a paragraph in the original data as a long sentence.

Further, we use half of the data for LM training, and utilize the other half for EB-M evaluation.

The resulting training/validation/test/evaluation set has 300k/1.5k/1.5k/300k sentences.

The vocabulary is of size 50k.

We train a two-layer LSTM LM of hidden dimension 1024 as the MLE baseline model for wikitext-103.

For MLE baseline model training, the Adam optimizer is used with learning rate 0.001, no Dropout (Srivastava et al., 2014) is applied.

The model is trained for 100 epochs.

We first measure EB-M on the wikitext-103 data-set, which has large amount of evaluation data.

The results are shown in Figure 5 .

We provide EB-M measurements with metric d T V in Appendix E, as they are similar to those using metric d JS .

It is shown that the measurements become stable when using 100k data/model samples.

EB-M has an average value of 1.10, indicating a significant gap between the model's MGD when fed with history from P D or P M .

Further, we observe a steady growth of EB-M along the length of history, which is expected as an outcome of exposure bias.

However, as discussed in Section 4.2, these measurements can only show that the LM does have better (marginal) generation quality when fed with data prefixes, but does not provide informative information for the significance of exposure bias.

We implement our MLE baseline and scheduled sampling (SS) in PyTorch.

For SS, we use a linear decay schedule to move from complete teacher forcing to replace-sample rate of 0.1.

We find that larger rate will give worse performance.

For CoT, we use a PyTorch implementation in https://github.com/pclucas14/ GansFallingShort.

We use a mediator model that has twice the size of the generator.

We set M-step to be 4, and G-step to be 1.

For RankGAN, we use a TensorFlow implementation in https://github.com/ desire2020/RankGAN.

Note that in our non-MLE experiments, the generator model is set to be the same size with the baseline MLE model.

We tune the non-MLE methods using the corpus-BLEU metric, which is widely used in text GAN literature.

In this section we provide more details for the AMT evaluation discussed in Section 5.3.

We show the HIT interface in Figure 6 .

Each HIT will include 10 pairs of context and its corresponding choices.

Five of them are history samples from real data, and the other five is from the trained model.

The history samples are mixed, so that the turker doesn't know whether the history sample is from real data or the model.

The next-word choices are also shuffled.

The history length of the context could be 10, 20, or 30.

We collect around 10k HITs for each history length configuration.

The same history sample is not repeated across the HITs.

We limit each turker to do at most 200 HITs.

For all history length configurations, there are around 300 unique turkers.

As shown by Figure 7 , most turkers conduct less than 20 HITs.

In Figure 8 , we show that we are able to get stable measurements of EB-C with 100k samples for the LSTM-512 synthetic experiment.

In Figure 9 and Figure 10 we provide EB-M measurements with metric d T V discussed in Section 4.2, the results are similar to those using metric d JS .

In Figure 11 , we provide EB-C measurements of a 3-layer transformer LM (Dai et al., 2019) with 512 hidden dimension, in the synthetic setting.

We show PPL results for model trained on EMNLP-news data-set in Table 5 .

The MLE model for wiki-103 data-set discussed in Section 4.2 has PPL 84.58.

Note that due to our special setting 8 , our PPL result is not directly comparable to state-of-art LM results on these data-sets.

8 We only keep sentences of length longer than L, and for wiki-103, only half of training data is used.

At the same time, she responded against the package of short-form compatible boats ...

Table 6 : More samples of a STOA MLE-trained transformer LM (on the wiki-103 data-set) when fed with different kinds of history.

To save space, we omitted the first 7 words of the random history.

Model Samples as Hisotry → Model Samples it was only a pieces that had gone up to → the forest and forces the shoppers about their chronic young i mean we didn ' t know what i haven → ' t considered through , " she told bbc radio if he were the president -elect , he was → known that he would run a force in business at but these are not as tired of " the same → message that the harry actor does have been hours in first opinion the agent have taken four seconds , or →

if they don ' t only know anything , were " the economy of the uk is low enough of → people of defending where americans think that " brexit , the economy grew on 1 .

6 % since the → us voted , and when it turned around 200 streets i was able to produce on my own , which → is good ; now that the theatre i ' ve " i ' ve not buying boys i addressed many → nervous times before , as a teenager made me is we think about one -third of the struggles we → actually want to see those very well that even more the story of a album -which made public -→ was still fantastic , and for the second time in " the test comes up before tuesday and when we → ' re feeling ahead again soon , " she posted a year on when he was last seen in his → home and he did not see him , his suffering brady has forced the 9 -known targets to get → all -of -12 gun migration and performing communication i asked if he himself did , i managed to → show all my charges at all , it used to Data Samples as Hisotry → Model Samples what this group does is to take down various different → players in the future and we play in paris we over 1 , 600 a day have reached greece this → gone in 2013 and it planned to allow civilians on " we ' re working through a legacy period , → and i am proud of the experience of the worker ' the first time anyone says you need help , → you don ' t have put accurate press into the out of those who came last year , 69 per → cent of women can really take the drive to avoid he has not played for tottenham ' s first team → this season then and sits down 15 -0 with so you have this man who seems to represent this → bad story , which he plays minutes -because he cnn :

you made that promise , but it wasn → ' t necessarily at all the features he had in this is a part of the population that is unk → lucky to have no fault today , and it would they picked him off three times and kept him out → of the game and was in the field , the the treatment was going to cost $ 12 , 000 → as a result of the request of anyone who was but if black political power is so important , why → doesn ' t we becomes the case that either stands local media reported the group were not looking to hurt → the animals , but would never be seen to say Table 7 : Samples of a MLE-trained LSTM LM (on the EMNLP-news data-set) when fed with different kinds of history.

To save space, we omitted the first 7 words of the random history.

<|TLDR|>

@highlight

We show that exposure bias could be much less serious than it is currently assumed to be for MLE LM training.