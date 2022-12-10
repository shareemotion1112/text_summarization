Neural sequence generation is commonly approached by using maximum- likelihood (ML) estimation or reinforcement learning (RL).

However, it is known that they have their own shortcomings; ML presents training/testing discrepancy, whereas RL suffers from sample inefficiency.

We point out that it is difficult to resolve all of the shortcomings simultaneously because of a tradeoff between ML and RL.

In order to counteract these problems, we propose an objective function for sequence generation using α-divergence, which leads to an ML-RL integrated method that exploits better parts of ML and RL.

We demonstrate that the proposed objective function generalizes ML and RL objective functions because it includes both as its special cases (ML corresponds to α → 0 and RL to α → 1).

We provide a proposition stating that the difference between the RL objective function and the proposed one monotonically decreases with increasing α.

Experimental results on machine translation tasks show that minimizing the proposed objective function achieves better sequence generation performance than ML-based methods.

Neural sequence models have been successfully applied to various types of machine learning tasks, such as neural machine translation Sutskever et al., 2014; , caption generation (Xu et al., 2015; BID6 , conversation (Vinyals & Le, 2015) , and speech recognition BID8 BID2 .

Therefore, developing more effective and sophisticated learning algorithms can be beneficial.

Popular objective functions for training neural sequence models include the maximum-likelihood (ML) and reinforcement learning (RL) objective functions.

However, both have limitations, i.e., training/testing discrepancy and sample inefficiency, respectively. indicated that optimizing the ML objective is not equal to optimizing the evaluation metric.

For example, in machine translation, maximizing likelihood is different from optimizing the BLEU score BID19 , which is a popular metric for machine translation tasks.

In addition, during training, ground-truth tokens are used for the predicting the next token; however, during testing, no ground-truth tokens are available and the tokens predicted by the model are used instead.

On the contrary, although the RL-based approach does not suffer from this training/testing discrepancy, it does suffer from sample inefficiency.

Samples generated by the model do not necessarily yield high evaluation scores (i.e., rewards), especially in the early stage of training.

Consequently, RL-based methods are not self-contained, i.e., they require pre-training via ML-based methods.

As discussed in Section 2, since these problems depend on the sampling distributions, it is difficult to resolve them simultaneously.

Our solution to these problems is to integrate these two objective functions.

We propose a new objective function α-DM (α-divergence minimization) for a neural sequence generation, and we demonstrate that it generalizes ML-and RL-based objective functions, i.e., α-DM can represent both functions as its special cases (α → 0 and α → 1).

We also show that, for α ∈ (0, 1), the gradient of the α-DM objective is a combinations of the ML-and RL-based objective gradients.

We apply the same optimization strategy as BID18 , who useed importance sampling, to optimize this proposed objective function.

Consequently, we avoid on-policy RL sampling which suffers from sample inefficiency, and optimize the objective function closer to the desired RL-based objective than the ML-based objective.

The experimental results for a machine translation task indicate that the proposed α-DM objective outperforms the ML baseline and the reward augmented ML method (RAML; BID18 , upon which we build the proposed method.

We compare our results to those reported by BID3 , who proposed an on-policy RL-based method.

We also confirm that α-DM can provide a comparable BLEU score without pre-training.

The contributions of this paper are summarized as follows.• We propose the α-DM objective function using α-divergence and demonstrate that it can be considered a generalization of the ML-and RL-based objective functions (Section 4).• We prove that the α-DM objective function becomes closer to the desired RL-based objectives as α increases in the sense that the upper bound of the maximum discrepancy between ML-and RL-based objective functions monotonically decreases as α increases.• The results of machine translation experiments demonstrate that the proposed α-DM objective outperforms the ML-baseline and RAML (Section 7).

In this section, we introduce ML-based and RL-based objective functions and the problems in association with learning neural sequence models using them.

We also explain why it is difficult to resolve these problems simulataneously.

Maximum-likelihood An ML approach is typically used to train a neural sequence model.

Given a context (or input sequence) x ∈ X and a target sequence y = (y 1 , . . . , y T ) ∈ Y, ML minimizes the negative log-likelihood objective function DISPLAYFORM0 where q(y|x) denotes the true sampling distribution.

Here, we assume that x is uniformly sampled from X and omit the distribution of x from Eq. (1) for simplicity.

For example, in machine translation, if a corpus contains only a single target sentence y * for each input sentence x, then q(y|x) = δ(y − y * ) and the objective becomes L(θ) = − x∈X log p θ (y * |x).ML does not directly optimize the final performance measure; that is, training/testing discrepancy exists.

This arises from at leset these two problems:(i) Objective score discrepancy.

The reward function is not used while training the model; however, it is the performance measure in the testing (evaluation) phase.

For example, in the case of machine translation, the popular evaluation measures such as BLEU or edit rate (Snover et al., 2006) differ from the negative likelihood function. (ii) Sampling distribution discrepancy.

The model is trained with samples from the true sampling distribution q(y|x); however, it is evaluated using samples generated from the learned distribution p θ (y|x).Reinforcement learning In most sequence generation task, the optimization of the final performance measure can be formulated as the minimization of the negative total expected rewards expressed as follows: DISPLAYFORM1 where r(y, y * |x) is a reward function associated with the sequence prediction y, i.e., the BLEU score or the edit rate in machine translation.

RL is an approach to solve the above problems.

The objective function of RL is L * in Eq. (2), which is a reward-based objective function; thus, there is no objective score discrepancy, thereby resolbing problem (i).

Sampling from p θ (y|x) and taking the expectation with p θ (y|x) in Eq. (2) also resolves problem (ii).

BID20 and BID3 directly optimized L * using policy gradient methods (Sutton et al., 2000) .

A sequence prediction task that selects the next token based on an action trajectory (y 1 , . . .

, y t−1 ) can be considered to be an RL problem.

Here the next token selection corresponds to the next action selection in RL.

In addition, the action trajectory and the context x correspond to the current state in RL.RL can suffer from sample inefficiency; thus, it may not generate samples with high rewards, particularly in the early learning stage.

By definition, RL generates training samples from its model distribution.

This means that, if model p θ (y|x) has low predictive ability, only a few samples will exist with high rewards.(iii) Sample inefficiency.

The RL model may rarely draw samples with high rewards, which hinders to find the true gradient to optimize the objective function.

Machine translation suffers from this problem because the action (token) space is vast (typically >10, 000 dimensions) and rewards are sparse, i.e., positive rewards are observed only at the end of a sequence.

Therefore, the RL-based approach usually requires good initialization and thus is not self-contained.

Previous studies have employed pre-training with ML before performing on-policy RL-based sampling BID20 BID3 .Entropy regularized RL To prevent the policy from becoming overly greedy and deterministic, some studies have used the following entropy-regularized version of the policy gradient objective function BID17 : DISPLAYFORM2 Reward augmented ML Norouzi et al. (2016) proposed RAML, which solves problems (i) and (iii) simultaneously.

RAML replaces the sampling distribution of ML, i.e., q(y|x) in Eq.(1), with a reward-based distribution q (τ ) (y|x) ∝ exp {r(y, y * |x)/τ }.

In other words, RAML incorporates the reward information into the ML objective function.

The RAML objective function is expressed as follows: DISPLAYFORM3 However, problem (ii) remains.

Despite these various attempts, a fundamental technical barrier exists.

This barrier prevents solving the three problems using a single method.

The barrier originates from a trade-off between sampling distribution discrepancy (ii) and sample inefficiency (iii), because these issues are related to the sampling distribution.

Thus, our approach is to control the trade-off of the sampling distributions by combining them.

The proposed method utilizes α-divergence DA (p q), which measures the asymmetric distance between two distributions p and q BID0 .

A prominent feature of α-divergence is that it can behave as D KL (p q) or D KL (q p) depending on the value of α, i.e., D(1) DISPLAYFORM0 where DISPLAYFORM1 .

Furthermore, α-divergence becomes a Hellinger distance when α equals to 1/2.

In this section, we describe the proposed objective function α-DM and its gradient.

Furthermore, we demonstrate that it can smoothly bridge both ML-and RL-based objective functions.

We define the α-DM objective function as the α-divergence between p θ and q (τ ) :

This DISPLAYFORM0 DISPLAYFORM1 Figure 1 illustrates how the α-DM objective bridges the ML-and RL-based objective functions.

Although the objectives L * DISPLAYFORM2 , and L (τ ) (θ) have the same global minimizer p θ (y|x) = q (τ ) (y|x), empirical solutions often differ.

To train neural network or other machine learning models via α-divergence minimization, one can use the gradient of α-DM objective function.

The gradient of Eq. (6) can be expressed as DISPLAYFORM0 where p DISPLAYFORM1 is a weight that mixes sampling distributions p θ and q (τ ) .

This weight makes it clear that the α-DM objective can be considered as a mixture of ML-and RL-based objective functions.

See Appendix A for the derivation of this gradient.

It converges to the gradient of entropy regularized RL or RAML by taking α → 1 or α → 0 limits, respectively (up to constant); i.e., DISPLAYFORM2 In Appendix C, we summarize all of the objective functions, gradients, and their connections.

In this section, we characterize the difference between α-DM objective function L (α,τ ) and the desired RL-based objective function L * (τ ) with respect to sup-norm.

Our main claim is that, with respect to sup-norm, the discrepancy between L (α,τ ) and L * (τ ) decreases linearly as α increases to 1.

We utilize this analysis to motivate our α-DM objective function with larger α if there are no concerns about the sampling inefficiency.

Proposition 1 Assume that p θ has the same finite support S as that of q (τ ) , and that for any s ∈ S, there exists δ > 0 such that p θ (s) > δ holds.

For any α ∈ (0, 1), the following holds.

DISPLAYFORM0 whereL (α,τ ) := αL (α,τ ) .

Here, C 1 , C 2 is universal constants irrelevant to α.

The following proposition immediately proves the theorem above.

Proposition 2 Assume that probability distribution p has the same finite support S as that of q, and that for any s ∈ S there exists δ > 0 such that p(s) > δ holds.

For any α ∈ (0, 1), the following holds.

sup DISPLAYFORM1 Here, C = max sup p p log 2 (q/p) , sup p q log 2 (q/p) .For the proof of the Proposition 1 and Proposition 2, see Appendix B.

In this paper, we employed the optimization strategy which is similar to that of RAML.

We sample target sentence y for each x from another data augmentation distribution q 0 (y|x), and then estimate the gradient by importance sampling (IS).

For example, we add some noise to the ground truth target sentence y * by insertion, substitution, or deletion, and the distribution p 0 (y|x) assigns some probability to each modified target sentence.

Given samples from this proposal ditribution p 0 (y|x), we update the parameter using the following IS estimator DISPLAYFORM0 Here, {(x 1 , y 1 ), . . .

, (x N , y N )} are the N samples from the proposal distribution q 0 (y|x), and w i is the importance weight which is proportional to p (α,τ ) θ (y i |x i ): DISPLAYFORM1 Note that the difference betweene RAML and α-DM is only this importance weight w i .

In RAML, w i depends only on q (τ ) (y i |x i ) but not on p θ (y i |x i ).

We normalize w i in each minibatch in order to use same hyperparameter (e.g., learning rate) as ML baseline.

Thus, this estimator becomes a weighted IS estimator.

A weighted IS estimator is not unbiased, yet but it has smaller variance.

Also, we found that normalizing q (τ ) (y i |x i ) and p θ (y i |x i ) in each minibatch leads to good results.

We evaluate the effectiveness of α-DM experimentally using neural machine translation tasks.

We compare the BLEU scores of ML, RAML, and the proposed α-DM on the IWSLT'14 GermanEnglish corpus BID5 .

In order to evaluate the impact of training objective function, we train the same attention-based encoder-decoder model BID16 for each objective function.

Furthermore, we use the same hyperparameter (e.g., learning rate, dropout rate, and temperature τ ) between all the objective functions.

For RAML and α-DM, we employ a data augmentation procedure similar to that of BID18 , and thus we generate samples from a data augmentation distribution q 0 (y|x).

Note that the difference between RAML and α-DM is only the weight w i of Eq. (14).

The details of data augmentation distribution are described in Section 7.2.

BID3 .

Specifically, we trained attention-based encoder-decoder model with the encoder of a bidirectional LSTM with 256 units and the LSTM decoder with the same number of layers and units.

We exponentially decay the learning rate, and the initial learning rate is chosen using grid search to maximize the BLEU performance of ML baseline on development dataset.

The important hyperparameter τ of RAML and α-DM is also determined to maximize the BLEU performance of RAML baseline on development dataset.

As a result, the initial learning rate of 0.5 and τ of 1.0 were used.

Our α-DM used the same hyperparameters as ML and RAML including the initial learning rate, τ , and so on.

Details about the models and parameters are discussed in Section 7.2.

To investigate the impact of hyperparameter α, we train the neural sequence models using α-DM 5 times for each fixed α ∈ {0.0, 0.1, . . .

, 0.9}, and then reported the BLEU score of test dataset.

Moreover, assuming that the underfitted model prevents the gradient from being stable in the early stage of training, we train the same models with α being linearly annealed from 0.0 to larger values; we increase the value of α by adding 0.03 at each epoch.

Here, the beam width k was set to 1 or 10.

All BLEU scores and their averages are plotted in FIG1 .

The results show that for both k = 1, 10, the models performance are better than smaller or larger α when α is around 0.5 (α = 0.5).

However, for larger fixed α, the performance was worse than RAML and ML baselines.

On the other hand, we can see that the annealed versions of α-DM improve the performance of the corresponding fixed versions in relatively larger α.

As a result, in the annealed scenario, α-DM with wide range of α ∈ (0, 1) improves on the performance consistently.

This implies that the underfitted model makes the performance worse.

We summarize the average BLEU scores and their standard deviation of ML, RAML, and α-DM with α ∈ {0.3, 0.4, 0.5} in Table.

1.

The result shows that the BLEU score (k = 10) of our α-DM outperforms ML and RAML baseline.

Furthermore, although the ML baseline performances differ between our results and those of BID3 , the proposed α-DM performance with α = 0.5 without pre-training is comparable with the on-policy RL-based methods BID3 .

We believe that these results come from the fact that α-DM with α > 0 has smaller bias than that of α = 0 (i.e., RAML).

We utilized a stochastic gradient descent with a decaying learning rate.

The learning rate decays from the initial learning rate to 0.05 with dev-decay (Wilson et al., 2017) , i.e., after training each epoch, we monitored the perplexity for the development set and reduced the learning rate by multiplying it with δ = 0.5 only when the perplexity for the development set does not update the best perplexity.

The mini-batch size is 128.

We used the dropout with probability 0.3.

Gradients are rescaled when the norms exceed 5.

In addition, if an unknown token, i.e., a special token representing a word that is not in the vocabulary, is generated in the predicted sentence, it was replaced by the token with the highest attention in the source sentence BID13 .

We implemented our models using a fork from the PyTorch 1 version of the OpenNMT toolkit BID14 .

We calculated the BLEU scores with multi-bleu.perl 2 script for both the development and test sets.

We obtained augmented data in the same manner as the RAML framework BID18 .

For each target sentence, some tokens were replaced by other tokens in the vocabulary and we used the negative Hamming distance as reward.

We assumed that Hamming distance e for each sentence is less than [m × 0.25], where m is the length of the sentence and [a] denotes the maximum integer which is less than or equal to a ∈ R. Moreover, the Hamming distance for a sample is uniformly selected from 0 to [m × 0.25].

One can also use BLEU or another machine translation metric for this reward.

However, we assumed proposal distribution q 0 (y|x) different from that of RAML.

We assumed the simplified proposal distribution q 0 (y|x), which is a discrete uniform distribution over [0, m × 0.25].

This results in hyperparameter τ used in this experiment being different from that of RAML.

We search the τ , which maximize the BLEU score of RAML on the development set.

As a results, τ = 1.0 was chosen, and α-DM also uses this fixed τ in all the experiments.

From the RL literature, reward-based neural sequence model training can be separated into on-policy and off-policy approaches, which differ in the sampling distributions.

The proposed α-DM approach can be considered an off-policy approach with importance sampling.

Recently, on-policy RL-based approaches for neural sequence predictions have been proposed.

BID20 proposed a method that uses the REINFORCE algorithm (Williams, 1992) .

Based on Ranzato et al. FORMULA0 , BID3 proposed a method that estimates a critic network and uses it to reduce the variance of the estimated gradient.

proposed a method that replaces some ground-truth tokens in an output sequence with generated tokens.

Yu et al. FORMULA0 , BID15 Wu et al. (2017) proposed methods based on GAN (generative adversarial net) approaches BID11 .

Note that on-policy RL-based approaches can directly optimize the evaluation metric.

BID10 proposed off-policy gradient methods using importance sampling, and the proposed α-DM off-policy approach utilizes importance sampling to reduce the difference between the objective function and the evaluation measure when α > 0.As mentioned previously, the proposed α-DM can be considered an off-policy RL-based approach in that the sampling distribution differs from the model itself.

Thus, the proposed α-DM approach has the same advantages as off-policy RL methods compared to on-policy RL methods, i.e., computational efficiency during training and learning stability.

On-policy RL approaches must generate samples during training, and immediately utilize these samples.

This property leads to high computational costs during training and if the model falls into a poor local minimum, it is difficult to recover from this failure.

On the other hand, by exploiting data augmentation, the proposed α-DM can collect samples before training.

Moreover, because the sampling distribution is a stationary distribution independent of the model, one can expect that the learning process of α-DM is more stable than that of on-policy RL approaches.

Several other methods that compute rewards before training can be considered off-policy RL-based approaches, e.g., minimum risk training (MRT; BID21 , RANDOMER (Guu et al., 2017 , and Google neural machine translation (GNMT; .While the proposed approach is a mixture of ML-and RL-based approaches, this attempt is not unique.

The sampling distribution of scheduled sampling is also a mixture of ML-and RL-based sampling distributions.

However, the sampling distributions of scheduled sampling can differ even in the same sentence, whereas ours are sampled from a stationary distribution.

To bridge the ML-and RL-based approaches, BID12 considered the weights of the gradients of the ML-and RL-based approaches by directly comparing both gradients.

In contrast, the weights of the proposed α-DM approach are obtained as the results of defining the α-divergence objective function.

GNMT (Wu et al., 2016 ) considered a mixture of ML-and RL-based objective functions by the weighted arithmetic sum of L and L * .

Comparing this weighted mean objective function and α-DM's objective function could be an interesting research direction in future.

In this study, we have proposed a new objective function as α-divergence minimization for neural sequence model training that unifies ML-and RL-based objective functions.

In addition, we proved that the gradient of the objective function is the weighted sum of the gradients of negative loglikelihoods, and that the weights are represented as a mixture of the sampling distributions of the ML-and RL-based objective functions.

We demonstrated that the proposed approach outperforms the ML baseline and RAML in the IWSLT'14 machine translation task.

In this study, we focus our attention on the neural sequence generation problem, but we expect our framework may be useful to broader area of reinforcement learning.

The sample inefficiency is one of major problems in reinforcement learning, and people try to mitigiate this problem by using several type of supervised learning frameworks such as imitation learning or apprenticisip learning.

This alternative approaches bring another problem similar to the neural sequence generaton problem that is originated from the fact that the objective function for training is different from the one for testing.

Since our framework is general and independent from the task, our approach may be useful to combine these approaches.

A GRADIENT OF α-DM OBJECTIVEThe gradient of α-DM can be obtained as follows: DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 where DISPLAYFORM3 In Eq. FORMULA0 , we used the so-called log-trick: ∇ θ p θ (y|x) = p θ (y|x)∇ θ log p θ (y|x).

Proposition 1 Assume that probability distribution p has the same finite support S as that of q, and that for any s ∈ S there exists δ > 0 such that p(s) > δ holds.

For any α ∈ (0, 1) the following holds.

DISPLAYFORM0 A (p q) ≤ C(1 − α).Here, C = max sup p p log 2 (q/p) , sup p q log 2 (q/p) .Proof.

By Taylor's theorem, there is an α ∈ (α, 1) such that DISPLAYFORM1 DISPLAYFORM2 where C 1 = τ max sup θ x∈X y∈Y p θ log 2 (q (τ ) /p θ ) , sup θ x∈X y∈Y q (τ ) log 2 (q (τ ) /p θ ) and C 2 = |Z(τ )|.

In this section, we summarize the objective functions of• ML (Maximum Likelihood),• RL (Reinforcement Learning),• RAML (Reward Augmented Maximum Likelihood; BID18 ,• EnRL (Entropy regularized Reinforcement Learning), and• α-DM (α-Divergence Minimization Training).Objectives.

The objective functions of ML, RL, RAML, EnRL, and α-DM are as follows DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 where q (τ ) (y|x) ∝ exp {r(y, y * |x)/τ }.

Typically, q(y|x) = δ(y, y * |x) where y * is the target with the highest reward.

We can rewrite some of these functions using KL or α-divergences: DISPLAYFORM5 DISPLAYFORM6 DISPLAYFORM7 A (p θ q (τ ) ).In the limits, there are the following connections between the objectives.

DISPLAYFORM8 DISPLAYFORM9 DISPLAYFORM10 DISPLAYFORM11

<|TLDR|>

@highlight

Propose new objective function for neural sequence generation which integrates ML-based and RL-based objective functions.