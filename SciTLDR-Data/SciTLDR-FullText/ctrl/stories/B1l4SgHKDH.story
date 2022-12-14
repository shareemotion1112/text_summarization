Text generation is ubiquitous in many NLP tasks, from summarization, to dialogue and machine translation.

The dominant parametric approach is based on locally normalized models which predict one word at a time.

While these work remarkably well, they are plagued by exposure bias due to the greedy nature of the generation process.

In this work, we investigate un-normalized energy-based models (EBMs) which operate not at the token but at the sequence level.

In order to make training tractable, we first work in the residual of a pretrained locally normalized language model and second we train using noise contrastive estimation.

Furthermore, since the EBM works at the sequence level, we can leverage pretrained bi-directional contextual representations, such as BERT and RoBERTa.

Our experiments on two large language modeling datasets show that residual EBMs yield lower perplexity compared to locally normalized baselines.

Moreover, generation via importance sampling is very efficient and of higher quality than the baseline models according to human evaluation.

The dominant approach to parametric text generation is based on large neural auto-regressive models (Radford et al., 2019) .

These models can be trained efficiently via maximum likelihood and they can efficiently generate samples of remarkable quality.

Key to their success is local normalization, i.e. they are defined in terms of a product of conditional distributions, one for each token in the sequence.

Such distributions are relatively cheap to compute with modern hardware given the limited vocabulary size of common sub-word units like BPE (Sennrich et al., 2015) .

Unfortunately, local normalization also brings some drawbacks.

First, the designer of the model needs to specify the order in which tokens are generated.

Second, at training time the model is conditioned on ground truth context while at test time it is conditioned on its own generations, a discrepancy referred to as exposure bias (Ranzato et al., 2016) .

Finally, while heuristics like beam search somewhat help rescore at the sequence level, generation generally lacks long-range coherency because it is produced by the greedy selection of one token at the time without lookahead.

Energy-based models (EBMs) (Hinton, 2002; LeCun et al., 2006; Ranzato et al., 2007 ) are a more general framework which potentially address all these issues, as they do not require any local normalization.

They only require the definition of an energy function defined over the whole input sequence.

Training aims at shaping the energy function such that regions of high density of training data points have lower energy than elsewhere.

In principle, EBMs are ideal for modeling text as they can score the whole input at once, they are not prone to label bias (Bottou, 1991) and they may enable generation of large chunks of text, which should help improve coherency.

However, so far EBMs had limited application in text generation, because sampling from the model is intractable, and so is maximum likelihood training.

The problem is that shaping the energy function is accomplished by updating the model parameters such that the energy is decreased at the training data points (a.k.a.

positive examples) and increased at other data points (a.k.a.

negative examples).

In maximum likelihood training negatives are generated from the model, but in text application we cannot use gradient-based MCMC methods (Teh et al., 2003; Du & Mordatch, 2019) and Gibbs sampling (Welling et al., 2005) is too slow to be practical.

Generating negatives by local perturbations of the ground truth would be efficient but hardly useful for generation purposes, when at test time the model needs to generate from scratch.

Recently, Bakhtin et al. (2019) carefully studied the problem of training a discriminator to distinguish human written text from language model generations.

They experimented with different language model and discriminator architectures, training/test time corpora and concluded that the discriminator can generalize rather well to weaker language models when the training/test corpora match.

Bakhtin et al. (2019) found that the learned discriminator is not robust to random perturbations, and argued that the discriminator operates in the "residual" space of the language model.

Concurrently, Grover et al. (2019) proposed a general approach to "de-bias" a generator, by simply training a discriminator and using its output for importance sampling.

In this work, we build upon these two works.

First, we formalize the residual interpretation by Bakhtin et al. (2019) and use a generative model of the form:

where P LM (x) is a locally normalized language model which is fixed during training, and E ?? is the energy function parameterized by ??.

The resulting model P ?? (x) is globally normalized due to the energy term.

Note that the same residual formulation was also used in Rosenfeld et al. (2001) ; Wang & Ou (2018b) ; Parshakova et al. (2019) .

This formulation has multi-fold benefits.

First, by incorporating a locally normalized language model, we can leverage recent advancements in locally normalized language modeling.

Second, the language model provides a natural proposal distribution for training (Bakhtin et al., 2019) as we shall see in ??3.

Third, training can be made efficient by using the conditional noise contrastive estimation objective (Gutmann & Hyv??rinen, 2010) .

Lastly, this formulation also enables efficient evaluation and generation via importance sampling (Horvitz & Thompson, 1952; Grover et al., 2019) .

In some sense, this last point is perhaps the central contribution of the paper, as it allows estimating perplexity of the residual EBM, and thus allows these EBMs to be compared in a standard way to other models.

Indeed, in ??4 we show that our joint model decreases perplexity on two large datasets, when compared to various auto-regressive language model baselines.

Finally, the EBM generations are significantly preferred by humans according to our qualitative evaluation.

To the best of our knowledge, this is the first time that an EBM has demonstrated improved generation ability against very strong auto-regressive baselines, both in terms of estimated perplexity and through human evaluation.

Energy-based models have a long history in machine learning (Hopfield, 1982; Hinton, 2002; LeCun et al., 2006; Ranzato et al., 2007) .

The key challenge of training is mining for good negatives.

This can be accomplished explicitly by fantasizing inputs where the energy should be increased or implicitly via global constraints such as sparsity (Ranzato et al., 2007) .

Methods attempting at maximizing the likelihood of the data require to sample from the distribution induced by the model.

Unfortunately, gradient-based MCMC approaches like Hybrid Monte Carlo (Teh et al., 2003) and Langevyn dynamics (Ranzato et al., 2007; Du & Mordatch, 2019) are not applicable when the input is discrete like in text applications.

Other approaches like Gibbs sampling (Hinton, 2002) were applied to binary inputs but do not scale well to large dictionaries once the energy function is a large bidirectional transformer model like the one used in this work.

Several variants of auto-encoders have also been investigated for representing and generating text (Bowman et al., 2016; Zhao et al., 2018 ), but they have not shown significant improvements in terms of perplexity and they have so far been applied to relatively small datasets only.

Our approach appears similar to discriminative reranking approaches used in the parsing and machine translation community (Shen et al., 2004) .

However, our approach provides a generative model, and parameters/hyper-parameters are directly tuned to close the gap between the model distribution and the data distribution, rather than relying on surrogate ranking losses.

This approach is also related to other sequence level training objectives (Edunov et al., 2018) , with the major difference that in those works training aims at improving the baseline model, but generation at test time is still greedy.

Energy Networks have been used for sequence modeling (Wang et al., 2015; 2018a) .

In particular, Rosenfeld et al. (2001) ; Wang & Ou (2018b) ; Parshakova et al. (2019) used the same residual form as in this work and Wang & Ou (2018b) ; Parshakova et al. (2019) used NCE for training the model.

Wang & Ou (2018b) used an LSTM as the generator and a CNN-LSTM as the energy function, and showed significant gains compared to LSTM baselines in speech recognition.

Our work builds on these prior works and develops new lower and upper bounds for the log-probability under the joint model, which makes it possible to show that the residual EBM approach gets better perplexity.

We also develop an importance weighting sampling scheme used at generation time, which is focused on conditional generation as opposed to rescoring in speech recognition (Wang & Ou, 2018b) .

The residual EBM formalism makes it very natural to use BERT for language modeling, and we show that empirically this type of approach can outperform modern state-of-the-art language modeling baselines, both using our method for estimating perplexity, and through human evaluation.

Generative Adversarial Networks (Goodfellow et al., 2014 ) also relate to EBMs, except that in EBMs the generator is implicit and negatives samples are produced by the discriminator itself.

In our work, the pretrained locally normalized language model can be seen as as fixed generator, like in Bakhtin et al. (2019) .

Azadi et al. (2018) also share our same goal but their generator is not locally normalized and they propose to improve the sampling from the generator by using the discriminator for rejection sampling.

Most similar to our work, Grover et al. (2019) propose to use the discriminator to de-bias the pretrained generator using importance sampling.

We adapt this work to the application of text generation.

In particular, we adopt the conditional noise contrastive estimation (NCE) objective (Ma & Collins, 2018; Gutmann & Hyv??rinen, 2010 ) to our residual model energy function and then sample from the joint model using importance sampling.

We want to note that the same formulation has been proposed in (Wang & Ou, 2018b; Parshakova et al., 2019) .

While Ma & Collins (2018) used conditional NCE to predict the next word in a sequence, we apply it to produce a whole sequence at once with the pretrained auto-regressive language model as the noise distribution.

We study the problem of conditional generation of discrete sequences.

Given a prefix x 1 , ?? ?? ?? , x p with x j ??? V where V is the vocabulary, we want to model the probabilities of generating a sequence of total length T > p 1 .

The generative model is:

where Z ?? (x 1 , ?? ?? ?? , x p ) is a normalizing factor known as partition function.

Computing the partition function is intractable in our case since it involves a sum over |V | T ???p terms which grow exponentially with the sequence length: in our experiments the size of the vocabulary is 50,096 and the length of the generation is 40 tokens.

We call P ?? the joint model, and E ?? the residual energy function since P LM is fixed throughout training.

The goal of training is to learn the parameters of the energy function such that the joint model distribution gets close to the data distribution.

For the sake of reducing clutter in the notation, we will drop the conditioning variables in the following discussion.

When the partition function is intractable, Maximum Likelihood Estimation (MLE) requires samples from the model distribution, which is usually approximated with Monte Carlo sampling or mean field inference (Hinton, 2012; LeCun et al., 2006) for globally normalized models.

Unfortunately, both approaches are too expensive for text applications when using large bidirectional transformer models.

For instance, if we were to employ Gibbs sampling exactly, we would need to perform at every position as many forward passes as words in the dictionary to compute each marginal distribution.

On large datasets where training locally normalized models on multiple machines already takes days, having such additional overhead means that the model would learn from much less data for the same amount of time, and this is seldom a beneficial strategy for learning models that generalize well.

Therefore, we do not use either MCMC nor mean field methods, as the latter would introduce additional variational parameters or an inference network which anyway yields an approximation to MLE learning.

Instead, we train our residual energy function using Noise Contrastive Estimation (NCE) (Gutmann & Hyv??rinen, 2010) , and more specifically its conditional version (Ma & Collins, 2018) .

NCE requires two distributions: The model distribution and a noise distribution.

In our case, the model distribution is the joint model of Eq. 2, P ?? , while the noise distribution is the pretrained language model, P LM .

NCE then trains a binary classifier on the difference of log-probability scores of these two models.

Since our joint model is the product of the energy function (whose parameters we want to learn) with P LM , the difference reduces to: log P ?? ??? log P LM = ???E ?? .

Therefore, under these modeling assumptions of residual learning and noise model, the objective function becomes:

where x + is a positive sequence taken from the human generated training set, and x ??? is a negative sequence drawn from P LM (for a given ground truth prefix).

In other words, training the energy function reduces to training a binary classifier to discriminate between real text and text generated by an auto-regressive language model.

The aim of training is to assign as negative energy as possible to real data, and as positive energy as possible to machine generated data.

Interestingly, the role of positive and negative samples is totally symmetric in this loss function, ??5 will discuss the consequences of this.

With the theoretical guarantee of NCE, we can show that the optimum of the above objective is reached at data distribution with infinite amount of data and model with enough capacity, which is also proved in Ma & Collins (2018) 2 .

Theorem 1.

If P LM has the same support as P data , then the objective function in Eq. 3 reaches its maximum at log P LM (x) ??? E ?? (x) = log P data , if there exists such ??.

Proof.

This theorem directly follows from the proof in Gutmann & Hyv??rinen (2010) .

Note that at optimum,

).

However, we still need to estimate the partition function throughout the rest of this paper, since we cannot guarantee that this optimum can be reached.

A commonly used protocol for evaluating generative sequence models, especially language models, is perplexity (PPL), which is equal to 2 ?????? ,x1) .

PPL can be interpreted as the average number of tokens the model is uncertain of at every time step.

Since the log-likelihood required by PPL relies on estimating the partition function

, we derive two estimators for the log-partition function log Z ?? based on the work of Nowozin (2018) .

Theorem 2.

Denote T n as the empirical estimate of log E x???P LM exp(???E(x)) with n samples

The proof is given in Appendix A.2.

We can use the above two estimators to estimate the lower and upper bounds of the partition function, but we want to emphasize that they are true only asymptotically (when n is sufficiently large).

We also want to note that to get lower variance estimates we use leave-one-out strategy to estimate T n???1 .

See Nowozin (2018) for implementation details and methods to improve numeric stability.

Similarly to locally normalized models, we can also factorize the probabilities of an entire sequence step by step, as P (x) = T t=1 P (x t |x <t ), and evaluate the PPL for each generation step.

By

Algorithm 1: Top-k Joint Sampling Input: number of samples n drawn from P LM , value of k in top-k // Get a set of samples from P LM sample n samples {x 1 , ?? ?? ?? , x n } from P LM with top-k sampling calculate energies

Resample from the set of LM samples sample x = x i with probability

marginalizing over the future, we can derive the following per step probabilities:

The step-wise probabilities in Eq. 5 are an instance of importance sampling (Horvitz & Thompson, 1952) .

The basic P LM distribution is adjusted by the probability assigned to token x t by the energy function (numerator is clamped at x t while denominator sums over all the possible values of the token at position t), with the additional marginalization over all subsequent tokens up to the horizon T .

Since the summation involves exponentially many terms, unless t = T , this is approximated by samples drawn by P LM .

For t = T , we can calculate the log probability by exhaustive enumeration.

This gives us an idea of the true performance of our model at the last step, and it also provides a sanity-check of the tightness of our estimators.

Generating from the joint model is a non-trivial task.

A naive way is to generate from the joint model auto-regressively, by marginalizing the future as in Eq. 5, which we term Top-k auto-regressive sampling.

However, doing so is expensive and impractical, and we only use this method for a qualitative analysis of the joint model in Appendix A.1.

In order to generate efficiently, we use self-normalizing importance sampling (Owen, 2013; Grover et al., 2019) .

Under the assumptions that the model from which we wish to draw samples is the joint model, which is the product of the auto-regressive model and the energy function, and that the proposal distribution is the auto-regressive model itself, sampling proceeds simply by: a) sampling from the auto-regressive language model, followed by b) resampling according to the energy function.

The algorithm is shown in Algorithm 1, where we introduce an optional top-k constraint on the pretrained language model to improve the quality of samples in the set 3 .

Without the top-k constraint, as the number of samples goes to infinity, we would recover exact samples from the joint model distribution.

In this section, we describe the experimental set up and the results we obtained by using the residual EBM for text generation, both in terms of perplexity and generation quality.

Datasets We consider two datasets: the Toronto Book Corpus and CC-News (Bakhtin et al., 2019) .

The former dataset consists of fiction books in 16 different genres, totaling about half a billion words.

The latter is a de-duplicated subset of the English portion of the CommonCrawl news dataset (Nagel, 2016) , which totals around 16 Billion words.

The book corpus is more challenging because the range of style and topics is more diverse than CC-News.

Also, the book corpus is 30 times smaller than CC-News and may pose generalization challenges because of its smaller size.

In all our experiments we use a prefix of size 120 tokens and we generate the following 40 tokens; with the notation of Eq. 2, p = 120 and T = 160.

For training the joint models, for efficiency we generated 16/128 samples per prefix for CC-News/Book Corpus offline, and sample uniformly from those samples at training time.

Baselines We consider as base language model (BASE LM) used to generate negatives for the residual EBM, a transformer language model with 12 layers, h = 16, d model = 1024, d f f = 4096 (we refer to Vaswani et al. (2017) for notations).

This is also our first baseline model.

The joint model has as many parameters as the sum of the number of parameters in the base LM and the number of parameters in the energy network.

To make a fair comparison, we consider two additional baselines that have the same number of parameters as our joint model.

log P RALM (x t |x <t ) = log P LM (x t |x <t ) + log P ?? (x t |x <t ) + const (6) where P ?? takes the form of another auto-regressive language model.

The parameters of P ?? are trained by exact maximum likelihood training of P RALM .

The second baseline is an auto-regressive language model of the same size of our joint model (sum of the base LM and energy function parameters), we dub this model Big Auto-regressive Language Model (BALM).

BALM has 12 layers, h = 16, d model = 1568, d f f = 6272, and is trained by standard token level cross-entropy loss.

Residual EBM Architecture We consider two architectures for our residual EBM, both of them are based on transformers (Vaswani et al., 2017; Devlin et al., 2018) .

The first version uses causal self-attention and is derived from the base LM, a unidirectional transformer (UNIT).

It is of the same architecture as BASE LM, except that in the final layer we project the mean-pooled hidden states to a scalar energy value.

We initialize its parameters with a language model trained on the same dataset.

The second version is instead bi-directional (BIT), and the energy function is computed by projecting the mean-pooled top hidden states down to a single scalar value.

We consider two variants, a BIT-BASE following the architecture of RoBERTa-Base, and a BIT-LARGE * following RoBERTaLarge (Liu et al., 2019) .

We initialize the parameters with a trained BERT, and we use * to mark usage of external data (Liu et al., 2019) , otherwise it means that BERT was trained on our training set.

Notice how our model can be interpreted as a natural way to fine tune large bidirectional pretrained models for the text generation task.

While we expect BIT to yield better results because it can fully leverage context also for intermediate tokens, we also consider UNIT to compare to the RALM baseline, which uses the same architecture and only differs in the way parameters are trained and for the presence of local normalization.

We train our models on 8 DGX nodes, each with 8 Nvidia V100s.

We use the Adam optimizer, with cosine learning rate decay and learning rate warmup.

To stabilize training we used gradient norm clipping.

Detailed hyper-parameter settings can be found in Appendix A.3.

For generation, we use top-k sampling with k = 10 for all human evaluations.

We take 10,000 samples from BASE LM for our joint sampling.

Automatic Evaluation Our main result is reported in Table 1 where we compare models in terms of their perplexity.

We can see that on both datasets, residual EBMs with causal attention JOINT UNITRANSF outperforms the baseline RALM with approximately the same number of parameters.

The non-residual baseline BALM performs similarly to JOINT UNITRANSF, which might be due to the limitation that P LM is not trained jointly with the residual model in both JOINT UNITRANSF and RALM.

However, by using our EBM approach, we can remove the causal attention mask and use bi-directional models, which achieves better performance than baselines and JOINT UNITRANSF: without external data, JOINT BITRANSF-BASE reaches a higher performance than JOINT UNITRANSF with fewer parameters.

By initializing from the state-of-the-art pretrained bi-directional transformers RoBERTa-Base and RoBERTa-Large, JOINT BITRANSF-BASE* and JOINT BITRANSF-LARGE* reach even better performance than JOINT BITRANSF-BASE.

In the lower part of the table, we show that if we make the big language model baseline BALM deeper (BALM-24L) (24 layers instead of 12, for the same number of parameters)

we attain lower perplexity.

However, training the joint model JOINT BITRANSF-BASE on the residual of a deeper language model BASE LM-24L yields even lower perplexity, despite having fewer parameters.

One caveat of our evaluation protocol is that the perplexity bounds are only estimates, which might not reflect the true value, particularly since the number of possible sequences grows exponentially with the number of words that are generated.

We therefore break down perplexity per position in the generated sequences as in Eq. 5, and compare the estimated PPLs to the true enumerated PPLs at the last position, as shown in Figure 1 .

We find that at the final generation step, the estimated bounds agree remarkably well with the exact values, proving that our method at least gets a reasonable PPL estimate at the last generation step.

Human Evaluation Better perplexity results do not necessarily imply better generations.

Besides, since generation from the residual EBM requires approximations as in Algorithm 1, the limited sample size might induce approximation errors compared to truly sampling from the joint distribution.

Therefore, we conducted human evaluations to compare generations from the residual EBM model to generations from the baseline language models.

For each prefix, we present one completion from each model, and ask humans to select the one that is a better continuation.

More details about human evaluation can be found in the Appendix A.4.

The preference rates reported in Table 2 confirm that indeed the generation quality of JOINT BIT-BASE and JOINT BIT-LARGE * is better than both language model baselines.

Depending on the model variant, our joint model is preferred between 56% and almost 60% of the times; interestingly, the preference rate does not change much as we compare against base LM as opposed to BALM.

In fact, humans do not seem to have a preference for BALM over base LM, despite the former scores two perplexity points lower.

Similarly, JOINT UNIT is not preferred over BASE LM despite its lower perplexity score.

We surmise that unidirectional scoring functions and auto-regressive models exhibit generation artifacts which are easily detected by humans, and these may overshadow the improvements brought by perplexity gains.

In this section, we analyze some of the results we obtained.

First, we check whether we used a sufficient number of samples in our perplexity estimates.

Second, we assess whether the joint model produces less repetitions compared to the base language model, and finally we check how well some statistics of the model and data distributions match.

Number of samples.

In Figure 2 , we vary the number of samples we take in order to estimate PPL upper and lower bounds.

Beyond 20,000 samples the upper estimate becomes very stable, although Figure 3 : Density plot of log-probability scores using the base language model (left) or the joint model (right).

The red curve corresponds to real samples, the black curve to samples from BASE LM and the green curve to samples from BIT-BASE.

The joint model provides a much better fit than the base language model.

we have to emphasize that these estimates might be biased even though the gap between lower and upper bound closes as we take more samples.

Repetitions.

A typical artifact of auto-regressive language models is their tendency to repeat phrases.

It is then interesting to check whether the joint model is able to alleviate this artifact.

Fig. 2 shows that indeed the joint model has a slightly higher percentage of unique n-grams compared to the baseline language model with n = 2, 3, 4, although still not as high as the original human generated text.

A necessary condition for the model to match the data distribution.

If the joint model p ?? matches the data distribution p d , then statistics computed on a large population of samples from the two distributions should also match.

In particular, Fig. 3 show the density plots of log-likelihood scores of the baseline language model (left) and joint model (right) when fed with their own samples versus samples from the test set.

We observe that the histogram of samples from the joint model matches the real data distribution more closely: The difference of means in the LM BASE case is 21.64 whereas the difference is 6.20 in the joint approach.

In the previous sections we highlighted the strengths of residual EBMs, namely their simplicity, efficiency both at training and test time, and their improved perplexity scores against strong autoregressive language model baselines.

In this section, we comment on their limitations to caution the reader about when these methods are more likely to succeed and to inform other researchers about what future avenues of research may naturally derive from this work.

In order to make training efficient and side step costly negative mining using the energy function itself, the current approach uses negatives generated from a pretrained auto-regressive language model.

Therefore, our model works as long as the base language model from which we draw samples is strong enough, and as long as the ground truth and other plausible sequences are reachable by the baseline language model.

If the base language model has poor quality, then generation from our joint model is going to be poor as well, as the joint model merely resamples generations from the original language model.

Moreover, training is going to be trivial if the base language model is poor, because the residual energy function merely needs to detect trivial generation artifacts from the base language model.

In fact, observe that the role of positive and negative samples is symmetric in the loss of Eq. 3.

This means that the energy function can choose to minimize the loss by either modeling the true data or the negative samples; since the latter have much simpler structure, it is going to model the negative samples.

Therefore, importance sampling amounts to mostly down-weighing the worst samples from the base language model.

The consequence of this is that search with a poor base language model is going to be catastrophically inefficient, as we would need to sample an impractically large number of negatives in order to find samples that are reasonably close to the true data manifold.

To summarize, this work makes a rather strong implicit assumption on the quality of the base language model, and it is expected to work well only when this is rather strong.

In our application, this assumption is met quite well in practice as large auto-regressive language models trained on large datasets have improved significantly in recent years (Radford et al., 2019) .

In general however, residual learning always carries liability to its base model.

We investigated an EBM trained on the residual of a pretrained autoregressive language model (Wang & Ou, 2018b; Parshakova et al., 2019) .

The resulting joint model scores sequences holistically, thanks to the energy function.

Training is very efficient and consists of a binary classification task between positives from the training set and pregenerated negatives from the fixed language model.

Generation is also very efficient as it amounts to resampling from the large set of negatives produced by the base language model.

Our estimates show that the resulting model has lower perplexity than the base language model.

Finally, this approach may be interpreted as a natural way to finetune a large bidirectional transformer like BERT for text generation applications.

In the future, we plan to investigate other ways to generate negatives that may strike a better tradeoff between the amount of compute each negative requires and their closeness to the joint model distribution.

It would also be interesting to explore other loss functions and the generation of longer pieces of text by using this model auto-regressively at the chunk level, as opposed to the token level.

A APPENDIX

In this subsection, we factorize the joint model BIT-BASE auto-regressively, and compare its differences with BASE LM.

Since even estimating the per step probabilities according to Eq. 5 is too expensive, we further approximate it by only considering the top 128 words predicted by BASE LM, where we sample 10,000 completions for each of them to estimate P (x t |x <t ).

Then we take the top 10 entries and re-normalize, and compare it to the top 10 probabilities of BASE LM.

Our initial explorations suggested that the joint model tends to generate fewer repetitions.

Therefore we picked a few LM samples where there are repetitions at x t , and use the same context x <t to estimate P (x t |x <t ) for the joint model.

Some examples of P (x t |x <t ) of BASE LM and BIT-BASE are presented in Table 3 .

Indeed BASE LM usually assigns lower probabilities to repetitions even though the top k words remain the same, which is not surprising given that the existence of repetition is a strong indicator of coming from the LM, which would lead to a higher energy value hence lower joint probability.

... is aimed at setting common benchmarks for orderly migration practices, thereby reducing irregular flows.

The Global Compact contains ten guiding principles, including that migrants cannot be settled by countries with better integration policies and a fair and sustainable development.

"

For the first time in our history, a legally binding and

Theorem 2.

Denote T n as the empirical estimate of log E x???P LM exp(???E(x)) with n samples x i ??? P LM (i = 1, ?? ?? ?? , n), and let T n = log

Proof.

From Nowozin (2018) Eq. 35, we can write

Therefore,

For the other half part of the proof, using Eq. 8 we have

where c is a constant.

Therefore,

Putting the above together, The optimization settings are presented in Table 4 .

A screenshot of the human evaluation experiments can be found in Fig 4.

Every page asks for 4 comparisons, one of which we know what the ground truth answer is.

We subsampled 333 sentences from the test set of CC-News, and asked 3 Amazon Mechanical turkers to vote.

We consider one continuation better if it gets more votes.

To check the quality of the received ratings, we performed a qualification task beforehand, where one of the continuations is real text, and we kept the top half performing turkers for further evaluation (corresponding to higher than 66.67% accuracy for discriminating real from LM samples -for a total of 26 qualified turkers).

Then in the actual experiment, we use one out of every four comparisons as an attention check and drop responses if the turker did not pass the check.

We present generation examples when our approach BASE LM outperforms baseline BALM in Table 5 , and when our approach underperforms in Table 6 .

Here the judgment is based on human evaluation when all three turkers unanimously voted in favor of one model over the other.

Table 6 : Example generations when BIT-BASE underperforms BALM according to human evaluation.

<|TLDR|>

@highlight

We show that Energy-Based models when trained on the residual of an auto-regressive language model can be used effectively and efficiently to generate text. 

@highlight

A proposed Residual Energy-based Model (EBM) for text generation which operates at the sentence level, and can therefore leverage BERT, and achieves lower perplexity and is preferred by human evaluation.