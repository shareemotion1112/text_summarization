Machine translation is an important real-world application, and neural network-based AutoRegressive Translation (ART) models have achieved very promising accuracy.

Due to the unparallelizable nature of the autoregressive factorization, ART models have to generate tokens one by one during decoding and thus suffer from high inference latency.

Recently, Non-AutoRegressive Translation (NART) models were proposed to reduce the inference time.

However, they could only achieve inferior accuracy compared with ART models.

To improve the accuracy of NART models, in this paper, we propose to leverage the hints from a well-trained ART model to train the NART model.

We define two hints for the machine translation task: hints from hidden states and hints from word alignments, and use such hints to regularize the optimization of NART models.

Experimental results show that the NART model trained with hints could achieve significantly better translation performance than previous NART models on several tasks.

In particular, for the WMT14 En-De and De-En task, we obtain BLEU scores of 25.20 and 29.52 respectively, which largely outperforms the previous non-autoregressive baselines.

It is even comparable to a strong LSTM-based ART model (24.60 on WMT14 En-De), but one order of magnitude faster in inference.

Neural machine translation has attracted much attention from the research community BID1 BID13 BID5 and has been gradually adopted by industry in the past several years BID22 .

Despite the huge variety of model architectures BID1 BID6 BID19 , given a source sentence x = (x 1 , ..., x Tx ) and a target sentence y = (y 1 , ..., y Ty ), most neural machine translation models decompose and estimate the conditional probability P (y|x) in an universal autoregressive manner: P (y|x) = Π Ty t=1 P (y t |y <t , x),where y <t represents the first t − 1 words of y. During inference, given an input sentence, those models generate the translation results sequentially, token by token from left to right.

We call all such models AutoRegressive neural machine Translation (ART) models.

A state-of-the-art ART model, Transformer BID19 , is shown in the left part of Figure 1 .A well-known limitation of the ART models is that the inference process can hardly be parallelized, and the inference time is linear with respect to the length of the target sequence.

As a result, the ART models suffer from long inference time BID22 , which is sometimes unaffordable for industrial applications.

Consequently, people start to develop Non-AutoRegressive neural machine Translation (NART) models to speed up the inference process BID7 BID15 .

These models use the general encoder-decoder framework: the encoder takes a source sentence x as input and generates a set of contextual embeddings and predicted length T y ; conditioned on the contextual embeddings, the decoder takes a transformed copy of x as input and predicts the target tokens at all the positions independently in parallel according to the following decomposition:P (y|x, T y ) = Π Ty t=1 P (y t |T y , x).While the NART models achieve significant speedup during inference BID7 , their accuracy is considerably lower than their ART counterpart.

Most of the previous works attribute the poor performance to this unavoidable conditional independence assumption of the NART model.

To tackle this issue, they try to improve the expressiveness and accuracy of the decoder input in different ways: BID7 introduce fertilities from statistical machine translation models into the NART models, BID15 base the decoding process of their proposed model on an iterative refinement process, and take a step further to embed an autoregressive submodule that consists of discrete latent variables into their model.

Although such methods provide better expressiveness of decoder inputs and improve the final translation accuracy, the inference speed of these models will be hurt due to the overhead of the introduced modules, which contradicts with the original purpose of introducing the NART models, i.e., to parallelize and speed up neural machine translation models.

Different from previous works that develop new submodules for decoder input, we improve the translation model from another perspective.

We aim to provide more guided signals during optimization.

That is, we do not introduce any new prediction submodule but introduce better regularization.

The reason we tackle the problem from this perspective lies in two points: First, the encoder input (source words) contains all semantic information for translation, and the decoder input in the NART model can be considered as a middle layer between input and output.

It is not clear how much gain can be achieved by developing a sophisticated submodule for a middle layer in a deep neural network.

Second, the encoder-decoder-based NART model is already over-parameterized.

We believe that such neural network still has great ability and space to be better optimized if we can provide it with stronger and richer signals, for example, from a much better ART model: Once we have a well-trained ART model, we actually know rich information about the contexts to make the prediction at each time step and the natural word alignments between bilingual sentences.

All the information could be invaluable towards the improved training of a NART model.

To well leverage an ART model, we use the hint-based training framework BID17 BID3 , in which the information from hidden layers of teacher model (referred as hints) are used to guide the training process of a student model.

However, hint-based training was developed for image classification models and it is challenging to define and use hints for translation.

First, the translation model is composed of stacked encoder layers, attention layers, and stacked decoder layers.

It is not clear how to define hints in such an encoder-decoder framework.

Second, the NART and ART models are of different architectures on the decoding stage.

It is not obvious how to leverage hints from the teacher to the training of student with a different architecture.

We find that directly applying hints used in the classification tasks fails.

In this paper, we first investigate the causes of the bad performance of the NART model, and then define hints targeting to solve the problems.

According to our empirical study, we find that the hidden states of the NART model differ from the ART model: the positions where the NART model outputs incoherent tokens will have very high hidden states similarity.

Also, the attention distributions of the NART model are more ambiguous than those of ART model.

Based on these observations, we design two kinds of hints from the hidden states and attention distributions of the ART model, to help the training of the NART model.

We have conducted experiments on the widely used WMT14 English-to-German/German-toEnglish (En-De/De-En) task and IWSLT14 German-to-English task.

For WMT14 En-De task, our proposed method achieves a BLEU score of 25.20 which significantly outperforms the nonautoregressive baseline models and is even comparable to a strong ART baseline, Google's LSTMbased translation model (24.60 BID22 ).

For WMT14 De-En task, we also achieve significant performance gains, reaching 29.52 in terms of BLEU.2 RELATED WORKS 2.1 AUTOREGRESSIVE TRANSLATION Given a sentence x = (x 1 , . . .

, x Tx ) from the source language, the straight-forward way for translation is to generate the words in the target language y = (y 1 , . . .

, y Ty ) one by one from left to right.

This is also known as the autoregressive factorization in which the joint probability is decomposed into a chain of conditional probabilities, as in the Eqn.

(1).

Deep neural networks are widely used to model such conditional probabilities based on the encoder-decoder framework.

The encoder takes the source tokens (x 1 , . . .

, x Tx ) as input and encodes x into a set of context states c = (c 1 , . . . , c Tx ).

The decoder takes c and subsequence y <t as input and estimates P (y t |y <t , c) according to some parametric function.

Figure 1: Hint-based training from ART model to NART model.

There are many design choices in the encoder-decoder framework based on different types of layers, e.g., recurrent neural network(RNN)-based BID1 , convolution neural network(CNN)-based BID6 and recent self-attention based BID19 approaches.

We show a self-attention based network (Transformer) in the left part of Figure 1 .

While the ART models have achieved great success in terms of translation quality, the time consumption during inference is still far away from satisfactory.

During training, the ground truth pair (x, y) is exposed to the model, and thus the prediction at different positions can be estimated in parallel based on CNN or self-attention networks.

However, during inference, given a source sentence x, the decoder has to generate tokens sequentially, as the decoder inputs y <t must be inferred on the fly.

Such autoregressive behavior becomes the bottleneck of the computational time BID22 .

In order to speed up the inference process, a line of works begin to develop non-autoregressive translation models.

These models follow the encoder-decoder framework and inherit the encoder structure from the autoregressive models.

After generating the context states c by the encoder, a separate module will be used to predict the target sentence length T y and decoder inputs z = (z 1 , . . .

, z Ty ) by a parametric function: (T y , z) ∼ f z (x, c; θ), which is either deterministic or stochastic.

The decoder will then predict y based on following probabilistic decomposition DISPLAYFORM0 Different configurations of T y and z enable the decoder to produce different target sentence y given the same input sentence x, which increases the output diversity of the translation models.

Previous works mainly pay attention to different design choices of f z .

BID7 introduce fertilities, corresponding to the number of target tokens occupied by each of the source tokens, and use a non-uniform copy of encoder inputs as z according to the fertility of each input token.

The prediction of fertilities is done by a separated neural network-based module.

BID15 define z by a sequence of generated target sentences y (0) , . . .

, y (L) , where each y (i) is a refinement of y (i−1) .

use a sequence of autoregressively generated discrete latent variables as inputs of the decoder.

While the expressiveness of z improved by different kinds of design choices, the computational overhead of z will hurt the inference speed of the NART models.

Comparing to the more than 15× speed up in BID7 , which uses a relatively simpler design choice of z, the speedup of is reduced to about 5×, and the speedup of BID15 Figure 2: Case study: the above three figures visualize the hidden state cosine similarities of different models.

The axes correspond to the generated target tokens.

Each pixel shows the cosine similarities cos ij between the last layer hidden states of the i-th and j-th generated tokens, where the diagonal pixel will always be 1.0.2×.

This contradicts with the design goal of the NART models: to parallelize and speed up neural machine translation models.

In this section, we introduce the proposed hint-based training algorithm that leverages a well-trained ART model to train the NART model.

Our model mostly follows Transformer BID19 , with an additional positional attention layer proposed by BID7 , as shown in the right part of Figure 1 .

To avoid overhead, we use simple linear combinations of source token embeddings as z, which has no learnable parameters.

Details about the model can be found in the appendix.

We first describe the observations we find about the ART and NART models, and then discuss what kinds of information can be used as hints and how to use them to help the training of the NART model.

According to the case study in BID7 and the observations based on our trained model, the translations of the NART models contain incoherent phrases and miss meaningful tokens on the source side.

As shown in TAB7 , these patterns do not commonly appear in ART models.

We aim to answer why the NART model tends to produce incoherent phrases (e.g. repetitive words) and miss relevant translations.

To study the first problem, we visualize the cosine similarities between decoder hidden states of a certain layer in both ART and NART models for sampled cases.

Mathematically, for a set of hidden states r 1 , . . . , r T , where T is the number of positions, the pairwise cosine similarity can be derived by cos ij = r i , r j /( r i · r j ).

We then plot the heatmap of the resulting matrix cos, and a typical example is shown in Figure 2 .From the figure, we can see that the cosine similarities between the hidden states at different positions in the NART model are larger than those of the ART model, which indicates that the hidden states across positions in the NART model are "similar".

Positions with highly-correlated hidden states are more likely to generate the same word and make the NART model output repetitive tokens, e.g., the yellow area on the top-left of Figure 2 (b).

However, this problem does not happen in the teacher model.

According to our statistics, 70% of the cosine similarities between hidden states in the teacher model are less than 0.25, and 95% are less than 0.5.To study the second problem, we visualize the encoder-decoder attentions for sampled cases.

Good attentions between the source and target sentences are usually considered to lead to accurate translation while poor ones may cause bad translation with wrong tokens.

As shown in Figure 3 , the attentions of the ART model almost covers all source tokens, while the attentions of the NART model do not cover "farm" but with two "morning".

This directly makes the translation result worse Figure 3: Case study: the above three figures visualize the encoder-decoder attention weights of different models.

The x-axis and y-axis correspond to source and generated target tokens respectively.

The attention distribution is from a single head of the third layer encoder-decoder attention, which is the most informative one according to our observation.

Each pixel shows attention weights α ij between the i-th source token and j-th target token.in the NART model.

These phenomena inspire us to use the intermediate hidden information in the ART model to guide the learning process of the NART model.

The empirical study in the previous section motivates us to leverage intermediate hidden information from a teacher translation model to help the training of a student model, which is usually referred to as hint-based training.

Hint-based training BID17 BID21 BID3 is popularly used to transfer complicated nonlinear mappings from one convolutional neural network to another.

In our scenario, we focus on how to define hints from a well-trained ART teacher model and use it to guide the training process of a NART student model.

We study layer-tolayer hints and assume both the teacher model and the student model have an M -layer encoder and an N -layer decoder, despite that the stacked components are quite different.

Without loss of generality, we discuss our proposed method on a given paired sentence (x, y).

In real experiments, losses are averaged over all training data.

For the teacher model, we use a tr t,l,h as the encoder-to-decoder attention distribution of h-th head in the l-th decoder layer at position t, and use r tr t,l as the output of the l-th decoder layer after feed forward network at position t. Correspondingly, a st t,l,h and r st t,l are used for the student model.

We propose a hint-based training framework that contains two kinds of hints: hints from hidden states and hints from word alignments.

Hints from hidden states The discrepancy of hidden states between ART and NART models motivates us to use hidden states of ART model as a hint for the learning process of the NART model.

One of the straight-forward methods is to regularize the L 1 or L 2 distance between each pair of hidden states in ART and NART models.

However, since the decoder input and network components are completely different in ART and NART models, we find using straight-forward regression method on hidden states hurts the learning of the translation model and fails.

Therefore, we design a more implicit loss to help the student refrain from the incoherent translation results by acting towards the teacher in the hidden-state level.

Specifically, we have where φ is a penalty function.

In particular, we let DISPLAYFORM0 where −1 ≤ γ st , γ tr ≤ 1 are two thresholds controlling whether to penalize or not.

We design this loss since we only want to penalize hidden states that are highly similar in the NART model, but not similar in the ART model.

We have tested several alternative choices of − log(1 − d st ), e.g., exp(d st ), from which we find similar experimental results.

Hints from word alignments Attention mechanism greatly boosts the performance of the ART models BID1 and becomes a crucial building block.

Many papers discover that the attentions provide reasonable word/phrase alignments between the source and target lead to better performance when predicting target tokens.

As we observe that meaningful words in the source sentence are sometimes untranslated by the NART model, and the corresponding positions often suffer from ambiguous attention distributions as shown in Figure 3 , we use the word alignment information from the ART model to help the training of the NART model.

In particular, we minimize KL-divergence between the per-head encoder-to-decoder attention distributions of the teacher and the student to encourage the student to have similar word alignments to the teacher model, i.e. DISPLAYFORM1 Our final training loss L is a weighted sum of two parts stated above and the negative log-likelihood loss L nll defined on bilingual sentence pair (x, y), i.e. DISPLAYFORM2 where λ and µ are hyperparameters controlling the weight of different loss terms.

We evaluate our methods on two widely used public machine translation datasets: IWSLT14 German-to-English (De-En) BID11 BID2 and WMT14 English-toGerman (En-De) dataset BID22 BID6 .

IWSLT14 De-En is a relatively smaller dataset comparing to WMT14 En-De.

To compare with previous works, we also reverse WMT14 English-to-German dataset and obtain WMT14 German-to-English dataset.

We pretrain Transformer BID19 as the autoregressive teacher model on each dataset.

The teacher models achieve 33.26/27.30/31.29 in terms of BLEU in IWSLT14 De-En, WMT14 En-De, De-En test set, respectively.

The student model shares the same number of layers in encoder/decoder, size of hidden states/embeddings and number of heads as the teacher models in each task.

Following BID7 ; BID14 , we replace the target sentences in all datasets by the decoded output of the teacher models.

Hyperparameters for hints based training (γ st , γ tr , λ, µ) are determined to make the scales of three loss components similar after initialization.

We also employ label smoothing of value ls = 0.1 BID18 in all experiments.

We use Adam optimizer and follow the optimizer setting and learning rate schedule in BID19 .

Models for WMT14/IWSLT14 tasks are trained on 8/1 NVIDIA M40 GPUs respectively.

We implement our model based on the open-sourced tensor2tensor and plan to release it in the near future.

More experimental settings can be found in the appendix.

During training, T y does not need to be predicted as the target sentence is given.

During testing, we have to predict the length of the target sentence for each source sentence.

In many languages, the length of the target sentence can be roughly estimated from the length of the source sentence.

For example, if the source sentence is very long, its translation is also a long sentence.

We provide a simple method to avoid the computational overhead, which uses input length to determine target sentence length: T y = T x + C, where C is a constant bias determined by the average length differences between the source and target sentences in the training data.

We can also predict the target length ranging from DISPLAYFORM0 , where B is the halfwidth.

By doing this, we can obtain multiple translation results with different lengths.

BID4 .

Transformer BID19 results are based on our own reproduction, and are used as the teacher models for NART models.

FT: Fertility based NART model by BID7 .

LT: Latent Transformer by .

IR: Iterative Refinement based NART model by BID15 .

En

Once we have multiple translation results, we additionally use our ART teacher model to evaluate each result and select the one that achieves the highest probability.

As the evaluation is fully parallelizable (since it is identical to the parallel training of the ART model), this rescoring operation will not hurt the non-autoregressive property of the NART model.

We use BLEU score BID16 as our evaluation measure.

During inference, we set C to 2, −2, 2 for WMT14 En-De, De-En and IWSLT14 De-En datasets respectively, according to the average lengths of different languages in the training sets.

When using the teacher to rescore, we set B = 4 and thus have 9 candidates in total.

We also evaluate the average per-sentence decoding latencies on one NVIDIA TITAN Xp GPU card by decoding on WMT14 En-De test sets with batch size 1 for our ART teacher model and NART models, and calculate the speedup based on them.

We compare our model with several baselines: LSTM-based, convolution-based, self attentionbased ART models, the fertility based (FT) NART model, the deterministic iterative refinement based (IR) NART model, and the Latent Transformer (LT) which is not fully non-autoregressive by incorporating an autoregressive sub-module in the NART model architecture.

The experimental results are shown in the TAB3 .Across different datasets, our method achieves state-of-the-art performances with significant improvements over previous proposed non-autoregressive models.

Specifically, our method outperforms fertility based NART model with 6.54/7.11 BLEU score improvements on WMT En-De and De-En tasks in similar settings.

Comparing to the ART models, our method achieves comparable results with state-of-the-art LSTM-based sequence-to-sequence model on WMT En-De task.

Apart from the translation accuracy, our model achieves a speedup of 30.2 (output a single sentence) or 17.8 (teacher rescoring) times over the ART counterparts.

Note that our speedups significantly outperform all previous works, because of our lighter design of the NART model: without any computationally expensive module trying to improve the expressiveness.

DISPLAYFORM0 We provide some case studies for the NART models with and without hints in TAB7 .

More cases can be found in the appendix.

From the first case, we can see that the model without hints translates the meaning of "as far as I'm concerned" to a set of meaningless tokens.

In the second case, the model without hints omits the phrase "the farm" and replaces it with a repetitive phrase "every morning".

In the third case, the model without hints mistakenly puts the word "uploaded" to the beginning of the sentence, whereas our model correctly translates the source sentence.

In all cases, hint-based training helps the NART model to generate better target sentences.

Source: ich weiß , dass wir es können , und soweit es mich betrifft ist das etwas , was die welt jetzt braucht .Target: i know that we can , and as far as i &apos;m concerned , that &apos;s something the world needs right now .ART: i know that we can , and as far as i &apos;m concerned , that &apos;s something that the world needs now .NART w/o Hints: i know that we can it , , as as as as it it it is , it &apos;s something that the world needs now .NART w/ Hints: i know that we can do it and as as &apos;s m concerned , that &apos;s something that the world needs now .Source: jeden morgen fliegen sie 240 kilometer zur farm .Target: every morning , they fly 240 miles into the farm .ART: every morning , they fly 240 miles to the farm .NART w/o Hints: every morning , you fly 240 miles to every morning .NART w/ Hints: every morning , they fly 240 miles to the farm .Source:

aber bei youtube werden mehr als 48 stunden video pro minute hochgeladen .Target: but there are over 48 hours of video uploaded to youtube every minute .ART: but on youtube , more than 48 hours of video are uploaded per minute .NART w/o Hints: but on youtube , uploaded than 48 hours hours of video per minute .NART w/ Hints: but on youtube , more than 48 hours video are uploaded per minute .We also visualize the hidden state cosine similarities and attention distributions for the NART model with hint-based training, as shown in Figure 2 (c) and 3(c).

With hints from hidden states, the hidden states similarities of the NART model decrease in general, and especially for the positions where the original NART model outputs incoherent phrases.

The attention distribution of the NART model after hint-based training is more similar to the ART teacher model and less ambiguous comparing with the NART model without hints.

Finally, we study the effectiveness of different parts and compare it with a NART model without hints.

We conduct an ablation study on IWSLT14 De-En task and the results are shown in TAB5 .

The hints from word alignments provide an improvement of about 1.6 BLEU points, and the hints from hidden states improve the results by about 0.8 points in terms of BLEU.

Non-autoregressive translation (NART) models have suffered from low-quality translation results.

In this paper, we proposed to use hints from well-trained autoregressive translation (ART) models to enhance the training of NART models.

Our results on WMT14 En-De and De-En significantly outperform previous NART baselines, and achieve comparable accuracy to an LSTM-based ART model, with one order of magnitude faster in inference.

In the future, we will focus on designing new architectures and new training methods for NART models to achieve comparable accuracy as the state-of-the-art ART models such as Transformer.

Knowledge Distillation (KD) was first proposed by BID9 , which trains a small student network from a large (possibly ensemble) teacher network.

The training objective of the student network contains two parts.

The first part is the standard classification loss, e.g, the cross entropy loss defined on the student network and the training data.

The second part is defined between the output distributions of the student network and the teacher network, e.g, using KL-divergence .

BID14 introduces the KD framework to neural machine translation models.

They replace the ground truth target sentence by the generated sentence from a well-trained teacher model.

Sentencelevel KD is also proved helpful for non-autoregressive translation in multiple previous works BID7 BID15 .However, knowledge distillation only uses the outputs of the teacher model, but ignores the rich hidden information inside a teacher model.

BID17 introduced hint-based training to leverage the intermediate representations learned by the teacher model as hints to improve the training process and final performance of the student model.

BID10 used the attention weights as hints to train a small student network for reading comprehension.

Dataset specifications The training/validation/test sets of the IWSLT14 dataset 3 contain about 153K/7K/7K sentence pairs, respectively.

The training set of the WMT14 dataset 4 contains 4.5M parallel sentence pairs.

Newstest2014 is used as the test set, and Newstest2013 is used as the validation set.

In both datasets, tokens are split into a 32000 word-piece dictionary BID22 which is shared in source and target languages.

Model specifications For the WMT14 dataset, we use the default network architecture of the base Transformer model in BID19 , which consists of a 6-layer encoder and 6-layer decoder.

The size of hidden nodes and embeddings are set to 512.

For the IWSLT14 dataset, we use a smaller architecture, which consists of a 5-layer encoder, and a 5-layer decoder.

The size of hidden states and embeddings are set to 256 and the number of heads is set to 4.Hyperparameter specifications Hyperparameters (τ, γ st , γ tr , λ, µ) are determined to make the scales of three loss components similar after initialization.

Specifically, we use τ = 0.3, γ st = 0.1, γ tr = 0.9, λ = 5.0, µ = 1.0 for IWSLT14 De-En, τ = 0.3, γ st = 0.5, γ tr = 0.9, λ = 5.0, µ = 1.0 for WMT14 De-En and WMT14 En-De.

BLEU scores We use tokenized case-sensitive BLEU BID16 5 for WMT14 En-De and De-En datasets, and use tokenized case-insensitive BLEU for IWSLT14 De-En dataset, which is a common practice in literature.

Repetitive words According to the empirical analysis, the percentage of repetitive words drops from 8.3% to 6.5% by our proposed hint-based training algorithm on the test set of IWSLT14 De-En, which is a more than 20% reduction.

This shows that our proposed method effectively improve the quality of the translation outputs.

Performance on long sentences We select the source sentences whose lengths are at least 40 in the test set of IWSLT14 De-En dataset, and test the models trained with different kinds of hints on this subsampled set.

The results are shown in TAB8 .

As can be seen from the table, our model outperforms the baseline model by more than 3 points in term of BLEU (20.63 v.s. 17.48) .

Note that the incoherent patterns like repetitive words are a common phenomenon among sentences of all lengths, rather than a special problem for long sentences.

Source: klingt verrückt .

aber das geht wieder darauf zurück , dass die eigene vorstellungskraft eine realität schaffen kann .Target: sounds crazy .

but this goes back to that theme about your imagination creating a reality .ART: sounds crazy .

but this goes back to the point that your imagination can create a reality .NART w/o Hints: sounds crazy .

but that back back to that that own imagination can create a reality .NART w/ Hints: sounds crazy .

but this goes back to fact that your imagination can create a reality .Source: vor einem jahr oder so , las ich eine studie , die mich wirklich richtig umgehauen hat .Target: i read a study a year or so ago that really blew my mind wide open .ART: one year ago , or so , i read a study that really blew me up properly .NART w/o Hints: a year year or something , i read a study that really really really me me .NART w/ Hints: a year ago or something , i read a study that really blew me me right .Source:

wenn ich nun hier schaue , sehe ich die athleten , die in dieser ausgabe erscheinen und die sportarten .Target: so i &apos;m looking at this ; i see the athletes that have appeared in this issue , the sports .ART: now , when i look here , i see the athletes that appear in this output and the sports species .NART w/o Hints: now if i look at here , i see the athletes that appear in this this ogand the kinds of sports .NART w/ Hints:

now if i look at here , i see the athlettes that come out in this output and the species of sports .Source: manchmal eilen diese ideen unserem denken voraus , auf ganz wichtige art und weise .Target: sometimes those ideas get ahead of our thinking in ways that are important .ART: sometimes these ideas forecast our thinking in a very important way .NART w/o Hints: sometimes these ideas make of our thinking , in a very important way .NART w/ Hints: sometimes these ideas make ahead of our thinking in a very important way .

@highlight

We develop a training algorithm for non-autoregressive machine translation models, achieving comparable accuracy to strong autoregressive baselines, but one order of magnitude faster in inference.  

@highlight

Distills knowledge from intermediary hidden states and attention weights to improve non-autoregressive neural machine translation.

@highlight

Proposes to leverage well trained autoregressive model to inform the hidden states and the word alignment of non-autoregressive Neural Machine Translation models.