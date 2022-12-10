Simultaneous machine translation models start generating a target sequence before they have encoded or read the source sequence.

Recent approach for this task either apply a fixed policy on transformer, or a learnable monotonic attention on a weaker recurrent neural network based structure.

In this paper, we propose a new attention mechanism, Monotonic Multihead Attention (MMA), which introduced the monotonic attention mechanism to multihead attention.

We also introduced two novel interpretable approaches for latency control that are specifically designed for multiple attentions.

We apply MMA to the simultaneous machine translation task and demonstrate better latency-quality tradeoffs compared to MILk, the previous state-of-the-art approach.

Code will be released upon publication.

Simultaneous machine translation adds the capability of a live interpreter to machine translation: a simultaneous machine translation model starts generating a translation before it has finished reading the entire source sentence.

Such models are useful in any situation where translation needs to be done in real time.

For example, simultaneous models can translate live video captions or facilitate conversations between people speaking different languages.

In a usual neural machine translation model, the encoder first reads the entire sentence, and then the decoder writes the target sentence.

On the other hand, a simultaneous neural machine translation model alternates between reading the input and writing the output using either a fixed or learned policy.

Monotonic attention mechanisms fall into the learned policy category.

Recent work exploring monotonic attention variants for simultaneous translation include: hard monotonic attention (Raffel et al., 2017) , monotonic chunkwise attention (MoChA) and monotonic infinite lookback attention (MILk) (Arivazhagan et al., 2019) .

MILk in particular has shown better quality / latency trade-offs than fixed policy approaches, such as wait-k (Ma et al., 2019) or wait-if-* (Cho & Esipova, 2016) policies.

MILk also outperforms hard monotonic attention and MoChA; while the other two monotonic attention mechanisms only consider a fixed reading window, MILk computes a softmax attention over all previous encoder states, which may be the key to its improved latencyquality tradeoffs.

These monotonic attention approaches also provide a closed form expression for the expected alignment between source and target tokens.

However, monotonic attention-based models, including the state-of-the-art MILk, were built on top of RNN-based models.

RNN-based models have been outperformed by the recent state-of-the-art Transformer model (Vaswani et al., 2017) , which features multiple encoder-decoder attention layers and multihead attention at each layer.

We thus propose monotonic multihead attention (MMA), which combines the strengths of multilayer multihead attention and monotonic attention.

We propose two variants, Hard MMA (MMA-H) and Infinite Lookback MMA (MMA-IL).

MMA-H is designed with streaming systems in mind where the attention span must be limited.

MMA-IL emphasizes the quality of the translation system.

We also propose two novel latency regularization methods.

The first encourages the model to be faster by directly minimizing the average latency.

The second encourages the attention heads to maintain similar positions, preventing the latency from being dominated by a single or a few heads.

The main contributions of this paper are: (1) A novel monotonic attention mechanism, monotonic multihead attention, which enables the Transformer model to perform online decoding.

This model leverages the power of the Transformer and the efficiency of monotonic attention.

(2) Better latencyquality tradeoffs compared to the MILk model, the previous state-of-the-art, on two standard translation benchmarks, IWSLT15 English-Vietnamese (En-Vi) and WMT15 German-English (De-En).

(3) Analyses on how our model is able to control the attention span and on the relationship between the speed of a head and the layer it belongs to.

We motivate the design of our model with an ablation study on the number of decoder layers and the number of decoder heads.

In this section, we review the monotonic attention-based approaches in RNN-based encoder-decoder models.

We then introduce the two types of Monotonic Multihead Attention (MMA) for Transformer models: MMA-H and MMA-IL.

Finally, we introduce strategies to control latency and coverage.

The hard monotonic attention mechanism (Raffel et al., 2017) was first introduced in order to achieve online linear time decoding for RNN-based encoder-decoder models.

We denote the input sequence as x = {x 1 , ..., x T }, and the corresponding encoder states as m = {m 1 , ..., m T }, with T being the length of the source sequence.

The model generates a target sequence y = {y 1 , ..., y U } with U being the length of the target sequence.

At the i-th decoding step, the decoder only attends to one encoder state m ti with t i = j. When generating a new target token y i , the decoder chooses whether to move one step forward or to stay at the current position based on a Bernoulli selection probability p i,j , so that t i ≥ t i−1 .

Denoting the decoder state at the i-th position, starting from j = t i−1 , t i−1 + 1, t i−1 + 2, ..., this process can be calculated as follows:

When z i,j = 1, we set t i = j and start generating a target token y i ; otherwise, we set t i = j + 1 and repeat the process.

During training, an expected alignment α is introduced to replace the softmax attention.

It can be calculated in a recurrent manner, shown in Equation 4: Raffel et al. (2017) also introduce a closed-form parallel solution for the recurrence relation in Equation 5:

where cumprod(x) = [1, x 1 , x 1 x 2 , ...,

In practice, the denominator in Equation 5 is clamped into a range of [ , 1] to avoid numerical instabilities introduced by cumprod.

Although this monotonic attention mechanism achieves online linear time decoding, the decoder can only attend to one encoder state.

This limitation can diminish translation quality as there may be insufficient information for reordering.

Moreover, the model lacks a mechanism to adjust latency based on different requirements at decoding time.

To address these issues, introduce Monotonic Chunkwise Attention (MoChA), which allows the decoder to apply softmax attention to a fixed-length subsequence of encoder states.

Alternatively, Arivazhagan et al. (2019) introduce Monotonic Infinite Lookback Attention (MILk) which allows the decoder to access encoder states from the beginning of the source sequence.

The expected attention for the MILk model is defined in Equation 6.

2.2 MONOTONIC MULTIHEAD ATTENTION Previous monotonic attention approaches are based on RNN encoder-decoder models with a single attention and haven't explored the power of the Transformer model.

2 The Transformer architecture (Vaswani et al., 2017) has recently become the state-of-the-art for machine translation (Barrault et al., 2019 ).

An important feature of the Transformer is the use of a separate multihead attention module at each layer.

Thus, we propose a new approach, Monotonic Multihead Attention (MMA), which combines the expressive power of multihead attention and the low latency of monotonic attention.

Multihead attention allows each decoder layer to have multiple heads, where each head can compute a different attention distribution.

Given queries Q, keys K and values V , multihead attention

The attention function is the scaled dot-product attention, defined in Equation 8:

There are three applications of multihead attention in the Transformer model:

1.

The Encoder contains self-attention layers where all of the queries, keys and values come from previous layers.

2.

The Decoder contains self-attention layers that allow each position in the decoder to attend to all positions in the decoder up to and including that position.

3.

The Encoder-Decoder attention contains multihead attention layers where queries come from the previous decoder layer and the keys and values come from the output of the encoder.

Every decoder layer has a separate encoder-decoder attention.

For MMA, we assign each head to operate as a separate monotonic attention in encoder-decoder attention.

For a transformer with L decoder layers and H attention heads per layer, we define the selection process of the h-th head encoder-decoder attention in the l-th decoder layer as

where W l,h is the input projection matrix, d k is the dimension of the attention head.

We make the selection process independent for each head in each layer.

We then investigate two types of MMA, MMA-H(ard) and MMA-IL(infinite lookback).

For MMA-H, we use Equation 4 in order to calculate the expected alignment for each layer each head, given p l,h i,j .

For MMA-IL, we calculate the softmax energy for each head as follows:

and then use Equation 6 to calculate the expected attention.

Each attention head in MMA-H hardattends to one encoder state.

On the other hand, each attention head in MMA-IL can attend to all previous encoder states.

Thus, MMA-IL allows the model to leverage more information for translation, but MMA-H may be better suited for streaming systems with stricter efficiency requirements.

Finally, our models use unidirectional encoders: the encoder self-attention can only attend to previous states, which is also required for simultaneous translation.

At inference time, our decoding strategy is shown in Algorithm 1.

For each l, h, at decoding step i, we apply the sampling processes discussed in subsection 2.1 individually and set the encoder step at t l,h i .

Then a hard alignment or partial softmax attention from encoder states, shown in Equation 13, will be retrieved to feed into the decoder to generate the i-th token.

The model will write a new target token only after all the attentions have decided to write.

In other words, the heads that have decided to write must wait until the others have finished reading.

Figure 1 illustrates a comparison between our model and the monotonic model with one attention head.

Compared with the monotonic model, the MMA model is able to set attention to different positions so that it can still attend to previous states while reading each new token.

Each head can adjust its speed on-the-fly.

Some heads read new inputs, while the others can stay in the past to retain the source history information.

Even with the hard alignment variant (MMA-H), the model is still able to preserve the history information by setting heads to past states.

In contrast, the hard monotonic model, which only has one head, loses the previous information at the attention layer.

Effective simultaneous machine translation must balance quality and latency.

At a high level, latency measures how many source tokens the model has to read until a translation is generated.

The model we have introduced in subsection 2.2 is not able to control latency on its own.

While MMA allows simultaneous translation by having a read or write schedule for each head, the overall latency is determined by the fastest head, i.e. the head that reads the most.

It is possible that a head always reads new input without producing output, which would result in the maximum possible latency.

Note that the attention behaviors in MMA-H and MMA-IL can be different.

In MMA-IL, a head reaching the end of the sentence will provide the model with maximum information about the source sentence.

On the other hand, in the case of MMA-H, reaching the end of sentence for a head only gives a hard alignment to the end-of-sentence token, which provides very little information to the decoder.

Furthermore, it is possible that an MMA-H attention head stays at the beginning of sentence Algorithm 1 MMA monotonic decoding.

Because each head is independent, we compute line 3 to 16 in parallel Input: x = source tokens, h = encoder states, i = 1, j = 1, t l,h 0 = 1, y 0 = StartOfSequence.

1: while y i−1 = EndOfSequence do 2:

Break 12:

Read token x j

Calculate state h j and append to h 16:

19:

20:

without moving forward.

Such a head would not cause latency issues but would degrade the model quality since the decoder would not have any information about the input.

In addition, this behavior is not suited for streaming systems.

To address these issues, we introduce two latency control methods.

The first one is weighted average latency, shown in Equation 14:

where g l,h i = |x| j=1 jα i,j .

Then we calculate the latency loss with a differentiable latency metric C. Arivazhagan et al. (2019) , we use the Differentiable Average Lagging.

It is important to notice that, unlike the original latency augmented training in Arivazhagan et al. (2019) , Equation 15 is not the expected latency metric given C, but weighted average C on all the attentions.

The real expected latency isĝ = max l,h g l,h instead ofḡ, but using this directly would only affect the speed of the fastest head.

Equation 15, however, can control every head in a way that the faster heads will be automatically assigned to larger weights and slower heads will also be moderately regularized.

heads from getting faster.

However, for MMA-H models, we found that the latency of are mainly due to outliers that skip almost every token.

The weighted average latency loss is not sufficient to control the outliers.

We therefore introduce the head divergence loss, the average variance of expected delays at each step, defined in Equation 16:

:

where λ avg , λ var are hyperparameters that control both losses.

Intuitively, while λ avg controls the overall speed, λ var controls the divergence of the heads.

Combining these two losses, we are able to dynamically control the range of attention heads so that we can control the latency and the reading buffer.

For MMA-IL model, we only use L avg ; for MMA-H we only use L var .

We evaluate our model using quality and latency.

For translation quality, we use tokenized BLEU 3 for IWSLT15 En-Vi and detokenized BLEU with SacreBLEU (Post, 2018) for WMT15 De-En.

For latency, we use three different recent metrics, Average Proportion (AP) (Cho & Esipova, 2016) , Average Lagging (AL) (Ma et al., 2019) and Differentiable Average Lagging (DAL) (Arivazhagan et al., 2019) 4 .

We remind the reader of the metric definitions in Appendix A.2.

Table 3 : Effect of using a unidirectional encoder and greedy decoding to BLEU score.

We evaluate our method on two standard machine translation datasets, IWSLT14 En-Vi and WMT15 De-En.

Statistics of the datasets can be found in Table 1 .

For each dataset, we apply tokenization with the Moses (Koehn et al., 2007) tokenizer and preserve casing.

IWSLT15 English-Vietnamese TED talks from IWSLT 2015 Evaluation Campaign (Cettolo et al., 2016) .

We follow the same settings from and Raffel et al. (2017) .

We replace words with frequency less than 5 by <unk>. We use tst2012 as a validation set tst2013 as a test set.

WMT15 German-English We follow the setting from Arivazhagan et al. (2019) .

We apply byte pair encoding (BPE) (Sennrich et al., 2016) jointly on the source and target to construct a shared vocabulary with 32K symbols.

We use newstest2013 as validation set and newstest2015 as test set.

We evaluate MMA-H and MMA-IL models on both datasets.

The MILK model we evaluate on IWSLT15 En-Vi is based on rather than RNMT+ (Chen et al., 2018) .

In general, our offline models use unidirectional encoders, i.e. the encoder self-attention can only attend to previous states, and greedy decoding.

We report offline model performance in Table 2 and the effect of using unidirectional encoders and greedy decoding in Table 3 .

For MMA models, we replace the encoder-decoder layers with MMA and keep other hyperparameter settings the same as the 3 We acquire the data from https://nlp.stanford.edu/projects/nmt/, which is tokenized.

We do not have the tokenizer which processed this data, thus we report tokenized BLEU for IWSLT15 4 Latency metrics are computed on BPE tokens for WMT15 De-En -consistent with Arivazhagan et al. (2019) -and on word tokens for IWSLT15 En-Vi.

5 Luong & Manning (2015) report a BLEU score of 23.0 but they didn't mention what type of BLEU score they used.

This score is from our implementation on the data aquired from https://nlp.stanford.edu/projects/nmt/ offline model.

Detailed hyperparameter settings can be found in subsection A.1.

We use the Fairseq library (Ott et al., 2019) 6 for our implementation.

Code will be released upon publication.

In this section, we present the main results of our model in terms of latency-quality tradeoffs, ablation studies and analyses.

In the first study, we analyze the effect of the variance loss on the attention span.

Then, we study the effect of the number of decoder layers and decoder heads on quality and latency.

We also provide a case study for the behavior of attention heads in an example.

Finally, we study the relationship between the rank of an attention head and the layer it belongs to.

We plot the quality-latency curves for MMA-H and MMA-IL in Figure 2 .

The BLEU and latency scores on the test sets are generated by setting a latency range and selecting the checkpoint with best BLEU score on the validation set.

We use differentiable average lagging (Arivazhagan et al., 2019) when setting the latency range.

We find that for a given latency, our models obtain a better translation quality.

While MMA-IL tends to have a decrease in quality as the latency decreases, MMA-H has a small gain in quality as latency decreases.

The reason is that a larger latency does not necessarily mean an increase in source information available to the model.

In fact, the large latency is from the outlier attention heads, which skip the entire source sentence and point to the end of the source sentence.

The outliers not only increase the latency but they also do not provide useful information.

We introduce the attention variance loss to eliminate the outliers, as such a loss makes the attention heads focus on the current context for translating the new target token.

It is interesting to observe that even MMA-H has a better latency-quality tradeoff than MILk 7 even though each head only attends to only one state.

Although MMA-H is not yet able to handle an arbitrarily long input (without resorting to segmenting the input), since both the encoder and decoder self-attention have an infinite lookback, that model represents a good step in that direction.

In subsection 2.3, we introduced the attention variance loss to MMA-H in order to prevent outlier attention heads from increasing the latency or increasing the attention span.

We have already evaluated the effectiveness of this method on latency in subsection 4.1.

We also want to measure the difference between the fastest and slowest heads at each decoding step.

We define the average attention span in Equation 18:S

It estimates the reading buffer we need for streaming translation.

We show the relation between the average attention span (averaged over the IWSLT and WMT test sets) versus L var in Figure 3 .

As expected, the average attention span is reduced as we increase L var .

One motivation to introduce MMA is to adapt the Transformer, which is the current state-of-the-art model for machine translation, to online decoding.

Important features of the Transformer architecture include having a separate attention layer for each decoder layer block and multihead attention.

In this section, we test the effect of these two components on the offline, MMA-H, and MMA-IL models from a quality and latency perspective.

We report quality as measured by detokenized BLEU and latency as measured by DAL on the WMT13 validation set in Figure 4 .

We set λ avg = 0.2 for MMA-IL and λ var = 0.2 for MMA-H.

The offline model benefits from having more than one decoder layer.

In the case of 1 decoder layer, increasing the number of attention heads is beneficial but in the case of 3 and 6 decoder layers, we do not see much benefit from using more than 2 heads.

The best performance is obtained for 3 layers and 2 heads (6 effective heads).

The MMA-IL model behaves similarly to the offline model, and the best performance is observed with 6 layers and 4 heads (24 effective heads).

For MMA-H, with 1 layer, performance improves with more heads.

With 3 layers, the single-head setting is the most effective (3 effective heads).

Finally, with 6 layers, the best performance is reached with 16 heads (96 effective heads).

The general trend we observe is that performance improves as we increase the number of effective heads, either from multiple layers or multihead attention, up to a certain point, then either plateaus or degrades.

This motivates the introduction of the MMA model.

We also note that latency increases with the number of effective attention heads.

This is due to having fixed loss weights: when more heads are involved, we should increase λ var or λ avg to better control latency.

We characterize attention behaviors by providing a running example of MMA-H and MMA-IL, shown in Figure 5 .

Each curve represents the path that an attention head goes through at inference time.

For MMA-H, shown in Figure 5a , we found that when the source and target tokens have the same order, the attention heads behave linearly and the distance between fastest head and slowest head is small.

For example, this can be observed from partial sentence pair "I also didn 't know that" and target tokens "Tôi cũng không biết rằng", which have the same order.

However, when the source tokens and target tokens have different orders, such as "the second step" and "bước (step) thứ hai (second)", the model will generate "bước (step)" first and some heads will stay in the past to retain the information for later reordered translation "thứ hai (second)".

We can also see that the attention heads have a near-diagonal trajectory, which is appropriate for streaming inputs.

The behaviors of the heads in MMA-IL models are shown in Figure 5b .

Notice that we remove the partial softmax alignment in this figure.

Because we don't expect streaming capability for MMA-IL, some heads stop at early position of the source sentence to retain the history information.

Moreover, because MMA-IL has more information when generating a new target token, it tends to produce better translation quality.

In this example, the MMA-IL model has better translation on "isolate the victim" than MMA-H ("là cô lập nạn nhân" vs "là tách biệt nạn nhân") In Figure 6 , we calculate the average and standard deviation of rank of each head when generating every target token.

For MMA-IL, we find that heads in lower layers tend to have higher rank and are thus slower.

However, in MMA-H, the difference of the average rank are smaller.

Furthermore, the standard deviation is very large which means that the order of the heads in MMA-H changes frequently over the inference process.

Recent work on simultaneous machine translation falls into three categories.

In the first one, models use a rule-based policy for reading input and writing output.

Cho & Esipova (2016) propose a WaitIf-* policy to enable an offline model to decode simultaneously.

Ma et al. (2019) propose a wait-k policy where the model first reads k tokens, then alternates between read and write actions.

Dalvi et al. (2018) propose an incremental decoding method, also based on a rule-based schedule.

In the second category, a flexible policy is learnt from data.

Grissom II et al. (2014) introduce a Markov chain to phrase-based machine translation models for simultaneous machine translation, in which they apply reinforcement learning to learn the read-write policy based on states.

Gu et al. (2017) introduce an agent which learns to make decisions on when to translate from the interaction with a pre-trained offline neural machine translation model.

Luo et al. (2017) used continuous rewards policy gradient for online alignments for speech recognition.

Lawson et al. (2018) proposed a hard alignment with variational inference for online decoding.

Alinejad et al. (2018) propose a new operation "predict" which predicts future source tokens.

Zheng et al. (2019b) introduce a restricted dynamic oracle and restricted imitation learning for simultaneous translation.

Zheng et al. (2019a) train the agent with an action sequence from labels that are generated based on the rank of the gold target word given partial input.

Models from the last category leverage monotonic attention and replace the softmax attention with an expected attention calculated from a stepwise Bernoulli selection probability.

Raffel et al. (2017) first introduce the concept of monotonic attention for online linear time decoding, where the attention only attends to one encoder state at a time.

extended that work to let the model attend to a chunk of encoder state.

Arivazhagan et al. (2019) also make use of the monotonic attention but introduce an infinite lookback to improve the translation quality.

In this paper, we propose two variants of the monotonic multihead attention model for simultaneous machine translation.

By introducing two new targeted loss terms which allow us to control both latency and attention span, we are able to leverage the power of the Transformer architecture to achieve better quality-latency trade-offs than the previous state-of-the-art model.

We also present detailed ablation studies demonstrating the efficacy and rationale of our approach.

By introducing these stronger simultaneous sequence-to-sequence models, we hope to facilitate important applications, such as high-quality real-time interpretation between human speakers.

Average Proportion 1 |x||y| We provide the detailed results in Figure 2 as Table 6 and Table 7 .

We explore a simple method that can adjust system's latency at inference time without training new models.

In Algorithm 1 line 8, 0.5 was used as an threshold.

One can set different threshold p during the inference time to control the latency.

We run the pilot experiments on IWSLT15 En-Vi dataset and the results are shown as We explore applying a simple average instead of a weighted average loss to MMA-H. The results are shown in Figure 7 and Table 9 .

We find that even with very large weights, we are unable to reduce the overall latency.

In addition, we find that the weighted average loss severely affects the translation quality negatively.

On the other hand, the divergence loss we propose in Equation 16 can efficiently reduce the latency while retaining relatively good translation quality for MMA-H models.

<|TLDR|>

@highlight

Make the transformer streamable with monotonic attention.