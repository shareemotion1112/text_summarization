The quality of a machine translation system depends largely on the availability of sizable parallel corpora.

For the recently popular Neural Machine Translation (NMT) framework, data sparsity problem can become even more severe.

With large amount of tunable parameters, the NMT model may overfit to the existing language pairs while failing to understand the general diversity in language.

In this paper, we advocate to broadcast every sentence pair as two groups of similar sentences to incorporate more diversity in language expressions, which we name as parallel cluster.

Then we define a more general cluster-to-cluster correspondence score and train our model to maximize this score.

Since direct maximization is difficult, we derive its lower-bound as our surrogate objective, which is found to generalize point-point Maximum Likelihood Estimation (MLE) and point-to-cluster Reward Augmented Maximum Likelihood (RAML) algorithms as special cases.

Based on this novel objective function, we delineate four potential systems to realize our cluster-to-cluster framework and test their performances in three recognized translation tasks, each task with forward and reverse translation directions.

In each of the six experiments, our proposed four parallel systems have consistently proved to outperform the MLE baseline, RL (Reinforcement Learning) and RAML systems significantly.

Finally, we have performed case study to empirically analyze the strength of the cluster-to-cluster NMT framework.

Recently, an encode-decoder neural architecture has surged and gained its popularity in machine translation.

In this framework, the encoder builds up a representation of the source sentence and the decoder uses its previous RNN hidden state and attention mechanism to generate target translation.

In order to better memorize the input information, an attention mechanism has been exploited to further boost its performance.

In order to train the attentive encoder-decoder architecture, Maximum Likelihood Estimation (MLE) algorithm has been widely used, which aims at maximizing the point-to-point (one sentence to one sentence) log-likelihood of data pairs in a given dataset.

However, this algorithm has severely suffered from data sparsity problem, or in other word, maximizing only likelihood the existing language pairs might make the model blind to all the non-existing similar sentence pairs.

Thus, the large neural model might overfit to certain prototypes existing in the training set while failing to generalize more unseen but similar scenarios in test time.hurting its semantic meaning.

2) Model-Centroid Augmentation (RL), and BID13 leverage model-generated candidates as pseudo training samples, which are weighted with rewards to enhance the model learning.

By exploring self-generated candidates, the model is able to understand the diversity in the output space.

In pseudo-learning algorithms, both RAML and RL can be interpreted as broadcasting a target ground truth as a cluster of analogues while leaving the source input untouched, which though helps the model understand target diversity, fails to capture the input diversity.

In order to explore both sides' diversity, we advocate a novel and general cluster-to-cluster framework of pseudo learning, which first broadcasts both source and target sentence as clusters and then train the model to comprehend their correspondence, as described in FIG0 .In this paper, we first introduce the concept of parallel cluster, then design the cluster-to-cluster correspondence score as our optimization objective, based on which, we derive its lower bound KL-divergence as our surrogate objective for model training.

In order to realize our proposed framework, we design four parallel systems and apply them to three recognized machine translation tasks with both forward and reverse translation directions, these four systems have all demonstrated their advantages over the existing competing algorithms in six translation tasks.

In the appendices, we draw samples from the parallel clusters and further analyze their properties to verify our motivation.

The contributions of our paper can be summarized as follows: 1) We are the first to propose the concept of cluster-to-cluster framework, which provides a novel perspective to current sequence-tosequence learning problems.

2) We delineate the framework and arrive in a novel KL-divergence loss function and generalizes several existing algorithms as special cases, which provides a highlevel understanding about the previous algorithms.2 RELATED LITERATURE

Exposure bias and train-test loss discrepancy are two major issues in the training of sequence prediction models.

Many research works BID16 BID13 BID9 have attempted to tackle these issues by adding reward-weighted samples drawn from model distribution into the training data via a Reinforcement Learning BID17 framework.

By exposing the model to its own distribution, these methods are reported to achieve significant improvements.

, BID13 and BID16 advocate to optimize the sequence model as a stochastic policy to maximize its expected task-level reward.

Though RL is not initially designed to resolve data sparsity problem, the model-centroid training samples can indeed alleviate data sparseness by exposing the sequence-to-sequence model to more unseen scenarios.

One problem of the previous RL works is that, the input information is still restricted to the dataset, which fails to teach model to comprehend source diversity.

The cluster-to-cluster framework augments many similar input sentences to account for source language diversity.

One successful approach for data augmentation in neural machine translation system is Reward Augmented Maximum Likelihood (RAML) , which proposes a novel payoff distribution to augment training samples based on task-level reward (BLEU, Edit Distance, etc) .

In order to sample from this intractable distribution, they further stratify the sampling process as first sampling an edit distance, then performing random substitution/deletion operations.

Following the work of RAML, BID11 introduces a novel softmax Q-Distribution to reveal RAML's relation with Bayes decision rule, and they also propose an alternative sampling strategy -first randomly replacing n-gram of the ground truth sentence and then using payoff distribution to compute corresponding importance weight with local normalization.

These two approaches augment the target-side data by exposing the model to diverse scenarios and improve its robustness.

We draw our inspiration from RAML, but with a difference that, instead of based on task-level reward, a learnable payoff function (cluster distribution) is used in our approach to take more latent structures into account, such as semantic meaning, language fluency, etc.

From the cluster distribution, we can sample semantically and syntactically correct candidates to train the model.

In addition, our more generalized bilateral data augmentation strategy also empowers our model more capability to generalize better.

In order to utilize the large amount of monolingual data in current NMT framework, different strategies have been designed, the most common methods can be concluded into these categories: 1) using large monolingual data to train language model and integrates it to enhance language fluency BID2 .

2) using self-learning method to transform the monolingual data into bilingual form BID14 Zhang & Zong, 2016) .

3) using reconstruction strategy to leverage monolingual data to enhance NMT training BID3 .

Although our motivation to augment training data is aligned with these semi-supervised algorithms, our proposed framework has substantial differences from them: 1) we don't rely on additional monolingual data to boost NMT performance; 2) Though we jointly train forward and backward translation models as advocated in and BID3 , our joint algorithm doesn't involve any interactions between these two models (they can be trained independently).

We define the parallel cluster as two groups of weighted sentences C(Y * ) and C(X * ), whose similarities (BLEU, METEOR, etc) with Y * and X * are above certain threshold M .

DISPLAYFORM0 Every sample X or Y is associated with a normalized weight p(X|X * ) or p(Y |Y * ) to denote how much chance a sentence X or Y is sampled from the corresponding cluster, here we draw a schematic diagram to better visualize the parallel cluster in FIG0 .

We will further talk about how we define and compute the weights in the following sections.

Upon the definition of parallel cluster, we further design a cluster-to-cluster correspondence score CR c???c (X * , Y * ) as the log scaled expectation of likelihood of a random sentence X in source cluster C(X * ) being translated to Y in target cluster C(Y * ), which generally denotes the translatability of two clusters, formally, we define the cluster-to-cluster correspondence score CR c???c (X * , Y * ) as below: The higher correspondence score the more likely these two clusters correspond to each other.

Note that the cluster-to-cluster correspondence score can reflect both NMT's and cluster's quality, assuming the cluster is ideal, then the correspondence score measures the translatability from a source sentence X to a target sentence Y , while assuming the NMT is ideal, then the correspondence score measures the quality of the cluster (the capability to rank paraphrases based on semantically similarity).

DISPLAYFORM0 DISPLAYFORM1

Based on the definition of parallel cluster and cluster-to-cluster correspondence score, we further design the cluster-to-cluster framework's objective function as maximizing the empirical correspondence score CR c???c (X * , Y * ; D) with the regularization of target cluster's entropy H(p(Y |Y * )) in a dataset D, as described below: DISPLAYFORM0 By applying Jensen's inequality to the objective function Obj c???c , we can further derive its lowerbound as: DISPLAYFORM1 From this, we notice that the cluster-to-cluster objective is lower bounded by a negative KL- DISPLAYFORM2 .

Therefore, we can use this lower-bound to maximize correspondence score, by changing the sign of this lower-bound function, we further define the loss function as: DISPLAYFORM3 We theoretically verify that this lower bound KL-divergence can generalize Maximum Likelihood (MLE) and Reward Augmented Maximum Likelihood (RAML) as special cases when we instantiate cluster distribution as Kronecker-Delta function ??(Y |Y * ) and payoff dis- TAB0 .

DISPLAYFORM4

In this section, we try to minimize the proposed KL-divergence KL(p(Y |Y * )||p(Y |X * )) so as to raise the lower bound of the regularized cluster-to-cluster correspondence.

We can write its deriva-tives w.r.t to the NMT parameters in two forms, namely parallel sampling and NMT broadcasting modes, which differ in their Monte-Carlo proposing distribution.??? Parallel Sampling: sampling candidates independently from two clusters and then reweighted pairwise samples with a translation confidence w(Y |X, X * ).

DISPLAYFORM0 ??? Translation Broadcasting: sampling candidates from one cluster and broadcasting them through the NMT to construct its opponents, and re-weighted by cluster confidence c(Y |Y * , X * ).

DISPLAYFORM1 More specifically, translation broadcasting's samples are more NMT-aware in the sense that it incorporates NMT's knowledge to generate correspondents.

The parallel sampling mode works like twosided RAML while translation broadcasting works more like mixed RAML-RL (Williams, 1992).

In this paper, we design cluster distribution in two manners, namely inadaptive (pre-computed without training) and adaptive (trained during optimization) cluster.

Both cluster designs meet the criterion of concentrating around the ground truth according to sentence similarity metric.

In addition, a cutoff criterion is also leveraged to reject samples whose task-level score is lower than certain threshold M value as in Equation 1.??? Inadaptive Cluster: we use two non-parametric distributions q(X|X * ) and q(Y |Y * ) to denote source and target parallel clusters, based on the similarity score between sample X/Y and the ground truth X * /Y * .

We follow the payoff distribution to define our inadaptive cluster: DISPLAYFORM0 where R(Y, Y * ) denotes the task-level reward (BLEU, CIDEr, METEOR, etc) and R(Y, Y * ) denotes its normalization in the whole output space, ?? is the hyper-parameter temperature to control the smoothness of the optimal distribution around correct target Y * .

Since the task-level reward only considers string-level matching (precision, recall, etc) while ignoring semantic coherence, the generated samples though lexically similar, prone to many semantical and syntactical mistakes, which might cause counter-effects to the NMT model.??? Adaptive Cluster: we use two parametric models p(X|X * ) and p(Y |Y * ) to denote the source and target adaptive cluster, which follow encoder-decoder neural architecture but take ground truth X * , Y * as inputs.

Adaptive cluster is designed to fulfill the following two requirements: 1) Proximity to ground truth: the randomly sampled candidates should have high similarity with the ground truth.

2) High correspondence score: parallel cluster should be highly correlated and translatable.

Combining these two goals can guarantee mutual dependence between the source and target clusters and also retain its similarity to the original ground truth.

Formally, we write the optimization target of the target cluster as: DISPLAYFORM1 During optimization, we fix the forward NMT p(Y |X) and target cluster p(X|X * ) to update source cluster p(Y |Y * ), and we fix the parameters of backward NMT p(X|Y ) and source cluster p(Y |Y * ) to update target cluster p(X|X * ).

Here we write target cluster's derivative as following: DISPLAYFORM2 Due to the mutual dependence between adaptive clusters and translation models, we advocate to alternately update the cluster and the translation models.

In this section, we advocate to combine both forward and backward translation directions in a joint system to simultaneously learn four models -forward NMT p(Y |X), backward NMT (X|Y ), source cluster p(X|X * ) and target cluster p(Y |Y * ).

We exploit different scenarios to combine these four models and then design four parallel systems, whose implementations are elaborated in TAB2 .

System-A and B use inadaptive (non-parametric) cluster, thus require optimizing only the two translation systems; system-A applies parallel sampling algorithm while B applies translation broadcasting algorithm.

In contrast, system-C and D apply adaptive (parametric) cluster, thus require simultaneous optimization of both NMT and cluster, system-C applies parallel sampling while system-D applies translation broadcasting algorithm.

These four systems exhibit different characteristics which are shown in details as below: In a slight abuse of notation, we will denote DISPLAYFORM0 System-A For system-A, we use inadaptive cluster with parallel sampling strategy to train the NMT model, and the forward-backward joint objective functions is defined as: DISPLAYFORM1 Formally, the derivative respect to ?? and ?? are shown as: DISPLAYFORM2 Parallel candidates are sampled from source and target cluster distributions are leveraged by scaled translation scores w(X|Y, Y * ), w(Y |X, X * ) during optimization.

System-B With the same loss function in system-A, translation broadcasting is leveraged to compute derivatives in system-B, instead of parallel sampling, and the gradients is shown as: DISPLAYFORM3 This system works similar as reinforcement Learning, where normalized environmental rewards R(X, X * ),R(Y, Y * ) are leveraged to guide the model's policy search, and the gradients is interpreted as a form of Monte-Carlo Policy Gradient BID18 .System-C Unlike System-A and system-B, two adaptive cluster distributions is used in system-C, thus the NMT and cluster are jointly optimized during training, and the loss function is defined as: DISPLAYFORM4 (14) we can get the derivatives as below: DISPLAYFORM5 To train the NMT system, parallel sentence pairs (X, Y ) are firstly sampled from two independent cluster distributions and then translation confidence scores w(Y |X, X * ), w(X|Y, Y * ) are leveraged to guide the training.

The derivatives w.r.t the cluster contain two elements, candidates sampled from translation system, and candidates sampled from cluster itself.

The two components together ensure parallel cluster's translatability and the similarity to the ground truth.

System-D With the same loss function in system-C, translation broadcasting strategy is leveraged to compute derivatives, instead of parallel sampling, and the gradients is shown as: DISPLAYFORM6 System-D works quite similar as system-B but differs in that cluster confidence scores c(X|X * , Y * ), c(Y |Y * , X * ) are leveraged in training NMT, hence it is more abundant than tasklevel rewards (R(X, X * ) andR(Y, Y * )).

System-D adopts the same gradient formulas in system-C to update the clusters.

The details of the training algorithm for system-A,B,C,D are shown in Algorithm 1:

To evaluate our cluster-to-cluster NMT framework on different-sized (small-data, medium-data and large-data) and different-lingual (German-English and Chinese-English) translation tasks, we conduct experiments on three datasets (IWSLT, LDC, WMT).

For more details about the datasets, please refer to Appendix C. For comparability, we follow the existing papers to adopt similar network architectures, and apply learning rate annealing strategy described in to further boost our baseline NMT system.

In our experiments, we design both the NMT and adaptive cluster models based on one-layered encoder-decoder network ) with a maximum sentence length of 62 for both the encoder and decoder.

During training, ADADELTA (Zeiler, 2012 ) is adopted with = 10 ???6 and ?? = 0.95 to separately optimize the NMT's and adaptive cluster's parameters.

During decoding, a beam size of 8 is used to approximate the full search space.

We compute the threshold similarity M via sentence-BLEU, some small-scaled experiments indicate M = 0.5 yields best performance, so we simply stick to this setting throughout all the experiments.

To prevent too much hyper-parameter tuning in building the inadaptive cluster, we follow to select the best temperature ?? = 0.8 in all experiments.

For comparison, RAML and RL systems are also implemented with the same sequence-to-sequence attention model, following and BID18 .

For more details of our RL's and RAML's implementations, please refer to Appendix A.

We can see from TAB5 that our system-D achieves significant improvements on both directions.

Though our baseline system is already extremely strong, using cluster-to-cluster framework can further boost the NMT system by over 1.0 BLEU point.

Baseline Model Baseline Model MIXER BID13 20.10 21.81 --BSO BID19 24.03 26.36 --A-C 27.56 28.53 --Softmax-Q BID11 27.66 28.77 --Our implementation of RL BID18 29.10 29.70 24.40 24.75 Our implementation of RAML Table 4 : Experimental results on NIST Chinese-English Machine Translation Task WMT2014 German-English We can see from Table 5 that system-C achieves the strongest result on both WMT14 EN-DE and DE-EN tasks, which outperforms the baseline system by over 1.1 BLEU points.

It's worth noting that our one-layer RNN model even outperforms the deep multilayer RNN model of Zhou et al. FORMULA0 and , which contain a stack of 4-7 LSTM layers.

By using cluster-to-cluster framework, our one-layer RNN model can fully exploit the dataset and learn to generalize better.

From the above 24 parallel cluster-to-cluster experiments, we observe general improvements over the fine-tuned baseline systems as well as our implemented RL/RAML systems.

To understand the strength of our cluster-to-cluster framework, we give more detailed comparisons with existing competing algorithms as below:Comparison with RAML From the above three tables, we can observe general improvements yielded by RAML algorithm on different tasks (except LDC Chinese-English), but RAML still suffers from two problems: on one hand, RAML's benefits is restricted by its neglect of the input variabilities, and on the other hand, without considering semantic contexts and language fluency, Table 5 : Experimental results on WMT-2014 German-English Machine Translation Task RAML's random replacement strategy may introduce noisy and wrong bilingual pairs to hurt the translation performance (like in LDC Chinese-English translation task).

Our adaptive cluster takes into account more semantic contexts to enclose more rational paraphrases, and the bilateral augmentation also empowers the model more chance to access various inputs.

Comparison with RL We can also observe prevalent improvements yielded by RL algorithm BID13 .

Exposing the model to self-generated translation can improve the performance.

Our methods inherit this merit and further enhance it with source and target clusters, which can improve the model with more sampled bilingual pairs from both source and target sides.

Comparison between four parallel systems Among our proposed four parallel systems, system-C and D achieve better performances than A and B throughout different experiments, which confirms the advantages of the adaptive clusters.

The adaptive cluster is more flexible and target optimized than inadaptive cluster.

Unlike the payoff distribution used in inadaptive cluster which only takes task-level reward into account, the adaptive cluster learns more sophisticated criterion and thus assigns more rational probability to sampled candidates.

We give more detailed analysis and visualization in the appendices to demonstrate how the source and target clusters look like.

We demonstrate the learning curves of four systems and visualize some adaptive clusters in Appendix D and Appendix E, which give a more intuition about cluster-to-cluster learning.

In this paper, we propose a cluster-to-cluster learning framework and incorporate this concept into neural machine translation.

Our designed systems have proved to be efficient in helping current NMT model to generalize in both source and target sides.

In the cluster-to-cluster framework, the cooperation of four agents can augment valuable samples and alleviate data sparsity, and achieve significant improvement compared with strong baseline systems.

We believe the concept of clusterto-cluster learning can be applicable to a wide range of natural language or computer vision tasks, which will be explored in the future.

Appendices A SYSTEM-DESIGN Sequence to sequence problem (machine translation) can be considered to produce an output sequence Y = (y 1 , y 2 , . . .

, y T ), y t ??? A given an input X. Given input-target pairs (X, Y * ), the generated sequence Y on test is evaluated with task-specific score R(Y, Y * ).

Recurrent neural networks have been widely used in sequence to sequence prediction tasks.

As proposed in and , the basic idea is to first encode the input sequence as a variablelength feature vectors, then apply attention mechanism to compute weighted average over the input vectors and summarize a context vector, with which, previous hidden states and previous label are fed into the decoder RNN to predict the next state and its label.

In our approach, attention-based encoder-decoder is leveraged for both the translation and cluster models, shown as: DISPLAYFORM0 A.1 RL NMT In order to train our RL system as well as adaptive cluster, we need to define a task-level reward as driving signal.

Instead of directly applying BLEU or other evaluation metric, we advocate to use a surrogate n-gram match interpolation, as shown as: DISPLAYFORM1 where N n denotes the number of n-gram match between Y and Y * .

In order to alleviate sequencereward sparseness, we further split it as a series of local reward to drive model's policy search at every time step.

Formally, we write the step-wise reward r(y t |y 1:t???1 , Y * ) as following. (22) where N (Y,??? ) represents the occurrence of n-gram??? in sequence Y , specifically, if a certain nsequence y t???n+1:t appears in reference and it's not repeating more than needed, then we assign a corresponding matching score to y t , the policy gradient is described as: DISPLAYFORM2 DISPLAYFORM3 A.2 RAML NMT In order to sample from the intractable payoff distribution for system-A/B as well as our implemented RAML system, we adopt stratified sampling technique described in .

Given a sentence Y * , we first sample an edit distance m, and then randomly select m positions to replace the original labels.

For each sentence, we randomly sample four candidates to perform RAML training.

DISPLAYFORM4 B MATHEMATICAL ANALYSIS We optimize the model parameters of our cluster-to-cluster models by minimizing the lower-bound KL-divergence instead of maximizing the original correspondence score, to characterize the difference between the two objective function, we analyze the relationships between these two functions below: DISPLAYFORM5 which can be further written as: DISPLAYFORM6 therefore, we can derive: DISPLAYFORM7 Since both cluster and translation confidence score c(Y |Y * , X * ) and w(Y |X, X * ) require computing the marginalized probability p(Y |X * ) known to be intractable for variable-length sequences, here we adopt different mechanisms to approximate them.

In system-A and C, we simplify DISPLAYFORM8 p??(Y |X * ) .

In system-B and D, since Y is broadcast through the translation system, the marginalized probabilityp(Y |X * ) is close to one, we discard this factor and approximate c(Y |Y DISPLAYFORM9

IWSLT2014 Dataset The IWSLT2014 German-English training data set contains 153K sentences while the validation data set contains 6,969 sentences pairs.

The test set comprises dev2010, dev2012, tst2010, tst2011 and tst2012, and the total amount are 6,750 sentences.

We adopt 512 as the length of RNN hidden stats and 256 as embedding size.

We use bidirectional encoder and initialize both its own decoder states and coach's hidden state with the learner's last hidden state.

The experimental results for IWSLT2014 German-English and English-German Translation Task are summarized in TAB5 .

In order to give a more intuitive view about what the cluster distribution looks like, we draw some samples from the well-trained cluster distribution in LDC Chinese-English Translation task as shown in TAB9 .

we can observe that most of the paraphrases are based on three types of modification, namely form changing, synonym replacement as well as simplification.

Most of the modifications does not alter the original meaning of the ground truth.

Encompassing more expressions with close meanings can ease the data sparseness problem, and enhance its generalization ability.

We here draw two samples from source and target clusters in FIG3 , which demonstrates how point-point correspondence can be expanded into cluster-to-cluster correspondence.

Reference taihsi natives seeking work in other parts of the country are given a thorough UNK before being hired, and later their colleagues maintain a healthy distance at first .Cluster taihsi natives seeking work in other parts of the country are given a thorough UNK before being employed, and later their colleagues maintain a healthy distance at first .

Property Simplification Reference i once took mr tung chee -hwa to a squatter area where he found beyond imagination that a narrow alley could have accommodated so many people.

Cluster i once took mr tung chee -hwa to a squatter area where he fo und beyond imagination that a narrow alley have a lot of people.

@highlight

We invent a novel cluster-to-cluster framework for NMT training, which can better understand the both source and target language diversity.