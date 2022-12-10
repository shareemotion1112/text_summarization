State of the art sequence-to-sequence models for large scale tasks perform a fixed number of computations for each input sequence regardless of whether it is easy or hard to process.

In this paper, we train Transformer models which can make output predictions at different stages of the network and we investigate different ways to predict how much computation is required for a particular sequence.

Unlike dynamic computation in Universal Transformers, which applies the same set of layers iteratively, we apply different layers at every step to adjust both the amount of computation as well as the model capacity.

On IWSLT German-English translation our approach matches the accuracy of a well tuned baseline Transformer while using less than a quarter of the decoder layers.

The size of modern neural sequence models (Gehring et al., 2017; Vaswani et al., 2017; Devlin et al., 2019) can amount to billions of parameters (Radford et al., 2019) .

For example, the winning entry of the WMT'19 news machine translation task in English-German used an ensemble totaling two billion parameters .

While large models are required to do better on hard examples, small models are likely to perform as well on easy ones, e.g., the aforementioned ensemble is probably not required to translate a short phrase such as "Thank you".

However, current models apply the same amount of computation regardless of whether the input is easy or hard.

In this paper, we propose Transformers which adapt the number of layers to each input in order to achieve a good speed-accuracy trade off at inference time.

We extend Graves (2016; ACT) who introduced dynamic computation to recurrent neural networks in several ways: we apply different layers at each stage, we investigate a range of designs and training targets for the halting module and we explicitly supervise through simple oracles to achieve good performance on large-scale tasks.

Universal Transformers (UT) rely on ACT for dynamic computation and repeatedly apply the same layer (Dehghani et al., 2018) .

Our work considers a variety of mechanisms to estimate the network depth and applies a different layer at each step.

Moreover, Dehghani et al. (2018) fix the number of steps for large-scale machine translation whereas we vary the number of steps to demonstrate substantial improvements in speed at no loss in accuracy.

UT uses a layer which contains as many weights as an entire standard Transformer and this layer is applied several times which impacts speed.

Our approach does not increase the size of individual layers.

We also extend the resource efficient object classification work of Huang et al. (2017) to structured prediction where dynamic computation decisions impact future computation.

Related work from computer vision includes Teerapittayanon et al. (2016) ; Figurnov et al. (2017) and Wang et al. (2018) who explored the idea of dynamic routing either by exiting early or by skipping layers.

We encode the input sequence using a standard Transformer encoder to generate the output sequence with a varying amount of computation in the decoder network.

Dynamic computation poses a challenge for self-attention because omitted layers in prior time-steps may be required in the future.

We experiment with two approaches to address this and show that a simple approach works well ( §2).

Next, we investigate different mechanisms to control the amount of computation in the decoder network, either for the entire sequence or on a per-token basis.

This includes multinomial and binomial classifiers supervised by the model likelihood or whether the argmax is already correct as well as simply thresholding the model score ( §3).

Experiments on IWSLT14 German-English Figure 1: Training regimes for decoder networks able to emit outputs at any layer.

Aligned training optimizes all output classifiers C n simultaneously assuming all previous hidden states for the current layer are available.

Mixed training samples M paths of random exits at which the model is assumed to have exited; missing previous hidden states are copied from below.

translation (Cettolo et al., 2014) as well as WMT'14 English-French translation show that we can match the performance of well tuned baseline models at up to 76% less computation ( §4).

We first present a model that can make predictions at different layers.

This is known as anytime prediction for computer vision models and we extend it to structured prediction (Huang et al., 2017) .

We base our approach on the Transformer sequence-to-sequence model (Vaswani et al., 2017) .

Both encoder and decoder networks contain N stacked blocks where each has several sub-blocks surrounded by residual skip-connections.

The first sub-block is a multi-head dot-product self-attention and the second a position-wise fully connected feed-forward network.

For the decoder, there is an additional sub-block after the self-attention to add source context via another multi-head attention.

Given a pair of source-target sequences (x, y), x is processed with the encoder to give representations s = (s 1 , . . .

, s |x| ).

Next, the decoder generates y step-by-step.

For every new token y t input to the decoder at time t, the N decoder blocks process it to yield hidden states (h

where block n is the mapping associated with the n th block and embed is a lookup table.

The output distribution for predicting the next token is computed by feeding the activations of the last decoder layer h N t into a softmax normalized output classifier W :

Standard Transformers have a single output classifier attached to the top of the decoder network.

However, for dynamic computation we need to be able to make predictions at different stages of the network.

To achieve this, we attach output classifiers C n parameterized by W n to the output h n t of each of the N decoder blocks:

∀n, p(y t+1 |h n t ) = softmax(W n h n t ) (3) The classifiers can be parameterized independently or we can share the weights across the N blocks.

Dynamic computation enables the model to use any of the N exit classifiers instead of just the final one.

Some of our models can choose a different output classifier at each time-step which results in an exponential number of possible output classifier combinations in the sequence length.

We consider two possible ways to train the decoder network (Figure 1) .

Aligned training optimizes all classifiers simultaneously and assumes all previous hidden states required by the self-attention are available.

However, at test time this is often not the case when we choose a different exit for every token which leads to misaligned states.

Instead, mixed training samples several sequences of exits for a given sentence and exposes the model to hidden states from different layers.

Generally, for a given output sequence y, we have a sequence of chosen exits (n 1 , . . .

, n |y| ) and we denote the block at which we exit at time t as n t .

Aligned training assumes all hidden states h n−1 1 , . . .

, h n−1 t are available in order to compute selfattention and it optimizes N loss terms, one for each exit (Figure 1a ):

The compound loss L dec (x, y) is a weighted average of N terms w.r.t.

to (ω 1 , . . .

ω N ).

We found that uniform weights achieve better BLEU compared to other weighing schemes (c.f .

Appendix A).

At inference time, not all time-steps will have hidden states for the current layer since the model exited early.

In this case, we simply copy the last computed state to all upper layers, similar to mixed training ( §2.2.2).

However, we do apply layer-specific key and value projections to the copied state.

Aligned training assumes that all hidden states of the previous time-steps are available but this assumption is unrealistic since an early exit may have been chosen previously.

This creates a mismatch between training and testing.

Mixed training reduces the mismatch by training the model to use hidden states from different blocks of previous time-steps for self-attention.

We sample M different exit sequences (n

and for each one we evaluate the following loss:

When n t < N , we copy the last evaluated hidden state h n t to the subsequent layers so that the self-attention of future time steps can function as usual (see Figure 1b) .

We present a variety of mechanisms to predict the decoder block at which the model will stop and output the next token, or when it should exit to achieve a good speed-accuracy trade-off.

We consider two approaches: sequence-specific depth decodes all output tokens using the same block ( §3.1) while token-specific depth determines a separate exit for each individual token ( §3.2).

We model the distribution of exiting at time-step t with a parametric distribution q t where q t (n) is the probability of computing block 1 , . . . , block n and then emitting a prediction with C n .

The parameters of q t are optimized to match an oracle distribution q * t with cross-entropy:

The exit loss (L exit ) is back-propagated to the encoder-decoder parameters.

We simultaneously optimize the decoding loss (Eq. (4)) and the exit loss (Eq. (6)) balanced by a hyper-parameter α to ensure that the model maintains good generation accuracy.

The final loss takes the form:

In the following we describe for each approach how the exit distribution q t is modeled (illustrated in Figure 2 ) and how the oracle distribution q * t is inferred.

(a) Sequence-specific depth

Decoder depth

Decoder depth

Figure 2: Variants of the adaptive depth prediction classifiers.

Sequence-specific depth uses a multinomial classifier to choose an exit for the entire output sequence based on the encoder output s (2a).

It then outputs a token at this depth with classifier C n .

The token-specific multinomial classifier determines the exit after the first block and proceeds up to the predicted depth before outputting the next token (2b).

The token geometric-like classifier (2c) makes a binary decision after every block to dictate whether to continue (C) to the next block or to stop (S) and emit an output distribution.

For sequence-specific depth, the exit distribution q and the oracle distribution q * are independent of the time-step so we drop subscript t.

We condition the exit on the source sequence by feeding the average s of the encoder outputs to a multinomial classifier:

where W h and b h are the weights and biases of the halting mechanism.

We consider two oracles to determine which of the N blocks should be chosen.

The first is based on the sequence likelihood and the second looks at an aggregate of the correctly predicted tokens at each block.

This oracle is based on the likelihood of the entire sequence after each block and we optimize it with the Dirac delta centered around the exit with the highest sequence likelihood.

We add a regularization term to encourage lower exits that achieve good likelihood:

Correctness-based: Likelihood ignores whether the model already assigns the highest score to the correct target.

Instead, this oracle chooses the lowest block that assigns the largest score to the correct prediction.

For each block, we count the number of correctly predicted tokens over the sequence and choose the block with the most number of correct tokens.

A regularization term controls the trade-off between speed and accuracy.

Oracles based on test metrics such as BLEU are feasible but expensive to compute since we would need to decode every training sentence N times.

We leave this for future work.

The token-specific approach can choose a different exit at every time-step.

We consider two options for the exit distribution q t at time-step t: a multinomial with a classifier conditioned on the first decoder hidden state h Multinomial q t :

The most probable exit arg max q t (n|x, y <t ) is selected at inference.

Geometric-like q t :

where, d is the dimension of the decoder states, W h ∈ R N ×d and w h ∈ R d are the weights of the halting mechanisms, and b h their biases.

During inference the decoder exits when the halting signal χ n t exceeds a threshold τ n which we tune on the valid set to achieve a better accuracy-speed trade-off.

If thresholds (τ n ) 1≤n<N have not been exceeded, then we default to exiting at block N .

The two classifiers are trained to minimize the cross-entropy with respect to either one the following oracle distributions:

Likelihood-based: At each time-step t, we choose the block whose exit classifier has the highest likelihood plus a regularization term weighted by λ to encourage lower exits.

This oracle ignores the impact of the current decision on the future time-steps and we therefore consider smoothing the likelihoods with an RBF kernel.

where we control the size of the surrounding context with σ the kernel width.

We refer to this oracle as LL(σ, λ) including the case where we only look at the likelihood of the current token with σ → 0.

Correctness-based: Similar to the likelihood-based oracle we can look at the correctness of the prediction at time-step t as well as surrounding positions.

We define the target q * t as follows:

Confidence thresholding Finally, we consider thresholding the model predictions ( §2), i.e., exit when the maximum score of the current output classifier p(y t+1 |h n t ) exceeds a hyper-parameter threshold τ n .

This does not require training and the thresholds τ = (τ 1 , . . .

, τ N −1 ) are simply tuned on the valid set to maximize BLEU.

Concretely, for 10k iterations, we sample a sequence of thresholds τ ∼ U(0, 1) N −1 , decode the valid set with the sampled thresholds and then evaluate the BLEU score and computational cost achieved with this choice of τ .

After 10k evaluations we pick the best performing thresholds, that is τ with the highest BLEU in each cost segment.

We evaluate on several benchmarks and measure tokenized BLEU (Papineni et al., 2002) :

IWSLT'14 German to English (De-En).

We use the setup of Edunov et al. (2018) and train on 160K sentence pairs.

We use N = 6 blocks, a feed-forward network (ffn) of intermediate-dimension

Uniform n = 1 n = 2 n = 3 n = 4 n = 5 n = 6 Average WMT'14 English to French (En-Fr).

We also experiment on the much larger WMT'14 EnglishFrench task comprising 35.5m training sentence pairs.

We develop on 26k held out pairs and test on newstest14.

The vocabulary consists of 44k joint BPE types (Sennrich et al., 2016) .

We use a Transformer big architecture and tie the embeddings of the encoder, the decoder and the output classifiers ((W n ) 1≤n≤6 ; §2.1).

We average the last ten checkpoints and use a beam of width 4.

Models are implemented in fairseq and are trained with Adam (Kingma & Ba, 2015) .

We train for 50k updates on 128 GPUs with a batch size of 460k tokens for WMT'14 En-Fr and on 2 GPUs with 8k tokens per batch for IWSLT'14 De-En.

To stabilize training, we re-normalize the gradients if the norm exceeds g clip = 3.

For models with adaptive exits, we first train without exit prediction (α = 0 in Eq. (7)) using the aligned mode (c.f .

§2.2.1) for 50k updates and then continue training with α = 0 until convergence.

The exit prediction classifiers are parameterized by a single linear layer (Eq. (8)) with the same input dimension as the embedding dimension, e.g., 1024 for a big Transformer; the output dimension is N for a multinomial classifier or one for geometric-like.

We exit when χ t,n > 0.5 for geometric-like classifiers.

We first compare the two training regimes for our model ( §2.2).

Aligned training performs selfattention on aligned states ( §2.2.1) and mixed training exposes self-attention to hidden states from different blocks ( §2.2.2).

We compare the two training modes when choosing either a uniformly sampled exit or a fixed exit n = 1, . . .

, 6 at inference time for every time-step.

The sampled exit experiment tests the robustness to mixed hidden states and the fixed exit setup simulates an ideal setting where all previous states are available.

As baselines we show six separate standard Transformers with N ∈ [1..6] decoder blocks.

All models are trained with an equal number of updates and mixed training with M =6 paths is most comparable to aligned training since the number of losses per sample is identical.

Table 1 shows that aligned training outperforms mixed training both for fixed exits as well as for randomly sampled exits.

The latter is surprising since aligned training never exposes the self-attention mechanism to hidden states from other blocks.

We suspect that this is due to the residual connections which copy features from lower blocks to subsequent layers and which are ubiquitous in Transformer models ( §2).

Aligned training also performs very competitively to the individual baseline models.

Aligned training is conceptually simple and fast.

We can process a training example with N exits in a single forward/backward pass while M passes are needed for mixed training.

In the remaining paper, we use the aligned mode to train our models.

Appendix A reports experiments with weighing the various output classifiers differently but we found that a uniform weighting scheme worked well.

On our largest setup, WMT'14 English-French, the training time of an aligned model with six output classifiers increases only marginally by about 1% compared to a baseline with a single output classifier keeping everything else equal.

Next, we train models with aligned states and compare adaptive depth classifiers in terms of BLEU as well as computational effort.

We measure the latter as the average exit per output token (AE).

As baselines we use again six separate standard Transformers with N ∈ [1..6] with a single output classifier.

We also measure the performance of the aligned mode trained model for fixed exits n ∈ [1..6].

For the adaptive depth token-specific models (Tok), we train four combinations: likelihoodbased oracle (LL) + geometric-like, likelihood-based oracle (LL) + multinomial, correctness based oracle (C) + geometric-like and correctness-based oracle (C) + multinomial.

Sequence-specific models (Seq) are trained with the correctness oracle (C) and the likelihood oracle (LL) with different values for the regularization weight λ.

All parameters are tuned on the valid set and we report results on the test set for a range of average exits.

Figure 3 shows that the aligned model (blue line) can match the accuracy of a standard 6-block Transformer (black line) at half the number of layers (n = 3) by always exiting at the third block.

The aligned model outperforms the baseline for n = 2, . . .

, 6.

For token specific halting mechanisms (Figure 3a ) the geometric-like classifiers achieves a better speed-accuracy trade-off than the multinomial classifiers (filled vs. empty triangles).

For geometriclike classifiers, the correctness oracle outperforms the likelihood oracle (Tok-C geometric-like vs. Tok-LL geometric-like) but the trend is less clear for multinomial classifiers.

At the sequence-level, likelihood is the better oracle (Figure 3b ).

The rightmost Tok-C geometric-like point (σ = 0, λ = 0.1) achieves 34.73 BLEU at AE = 1.42 which corresponds to similar accuracy as the N = 6 baseline at 76% fewer decoding blocks.

Figure 3) .

The best accuracy of the aligned model is 34.95 BLEU at exit 5 and the best comparable Tok-C geometric-like configuration achieves 34.99 BLEU at AE = 1.97, or 61% fewer decoding blocks.

When fixing the budget to two decoder blocks, Tok-C geometric-like with AE = 1.97 achieves BLEU 35, a 0.64 BLEU improvement over the baseline (N = 2) and aligned which both achieve BLEU 34.35.

Confidence thresholding (Figure 3c ) performs very well but cannot outperform Tok-C geometriclike.

In this section, we look at the effect of the two main hyperparameters on IWSLT'14 De-En: λ the regularization scale (c.f .

Eq. (9)), and the RBF kernel width σ used to smooth the scores (c.f .

Eq. (15)).

We train Tok-LL Geometric-like models and evaluate them with their default thresholds (exit if χ n t > 0.5).

Figure 4a shows that higher values of λ lead to lower exits.

Figure 4b shows the effect of σ for two values of λ.

In both curves, we see that wider kernels favor higher exits.

Finally, we take the best performing models form the IWSLT benchmark and test them on the large WMT'14 English-French benchmark.

Results on the test set (Figure 5a) show that adaptive depth still shows improvements but that they are diminished in this very large-scale setup.

Confidence thresholding works very well and sequence-specific depth approaches improve only marginally over the baseline.

Tok-LL geometric-like can match the best baseline result of BLEU 43.4 (N = 6) by using only AE = 2.96 which corresponds to less than half the decoder blocks; the best aligned result of BLEU 43.6 can be outmatched at (AE = 3.51, BLEU = 43.71).

In this setup, Tok-LL geometric-like slightly outperforms the Tok-C counterpart.

Confidence thresholding matches the accuracy of the N =6 baseline with AE 2.5 or 59% fewer decoding blocks.

However, confidence thresholding requires computing the output classifier at each block to determine whether to halt or continue.

This is a large overhead since output classifiers predict 44k types for this benchmark ( §4.1).

To better account for this, we measure the average number of FLOPs per output token (details in Appendix B).

Figure 5b shows that the Tok-LL geometric-like approach provides a better trade-off when the overhead of the output classifiers is considered.

Figures 7 and 6 show outputs for examples of the IWSLT'14 De-En and the WMT'14 En-Fr test sets, respectively, together with the exit and model probability for each token.

Less computation is used at the end of the sentence since periods and end of sentence markers (</s>) are easy to predict.

The amount of computation increases when the model is less confident e.g. in Figure 6a , predicting 'présent' (meaning 'present') is hard.

A straightforward translation is 'était là' but the model chooses 'present' which is also appropriate.

In Figure 6b , the model uses more computation to predict the definite article 'les' since the source has omitted the article for 'passengers'.

We extended anytime prediction to the structured prediction setting and introduced simple but effective methods to equip sequence models to make predictions at different points in the network.

We compared a number of different mechanisms to predict the required network depth and find that a simple correctness based geometric-like classifier obtains the best trade-off between speed and accuracy.

Results show that the number of decoder layers can be reduced by more than three quarters at no loss in accuracy compared to a well tuned Transformer baseline.

In this section we experiment with different weights for scaling the output classifier losses.

Instead of uniform weighting, we bias towards specific output classifiers by assigning higher weights to their losses.

Table 2 shows that weighing the classifiers equally provides good results.

Uniform n = 1 n = 2 n = 3 n = 4 n = 5 n = 6 Average Uniform n = 1 n = 2 n = 3 n = 4 n = 5 n = 6 Average Table 2 : Aligned training with different weights (ω n ) on IWSLT De-En.

For each model we report BLEU on the dev set evaluated with a uniformly sampled exit n ∼ U([1..6]) for each token and a fixed exit n ∈ [1..6] throughout the sequence.

The average corresponds to the average BLEU over the fixed exits.

Gradient scaling Adding intermediate supervision at different levels of the decoder results in richer gradients for lower blocks compared to upper blocks.

This is because earlier layers affect more loss terms in the compound loss of Eq. (4).

To balance the gradients of each block in the decoder, we scale up the gradients of each loss term (− LL n ) when it is updating the parameters of its associated block (block n with parameters θ n ) and revert it back to its normal scale before back-propagating it to the previous blocks.

Figure 8 and Algorithm 1 illustrate this gradient scaling procedure.

The θ n are updated with γ n -amplified gradients from the block's supervision and (N −n) gradients from the subsequent blocks.

We choose γ n = γ(N − n) to control the ratio γ:1 as the ratio of the block supervision to the subsequent blocks' supervisions.

Table 3 shows that gradient scaling can benefit the lowest layer at the expense of higher layers.

However, no scaling generally works very well.

Algorithm 1 Pseudo-code for gradient scaling (illustrated for a single step t)

6: end for 7: function SCALE_GRADIENT(Tensor x, scale γ) 8:

STOP_GRADIENT in PyTorch with x.detach().

10: end function Uniform n = 1 n = 2 n = 3 n = 4 n = 5 n = 6 Average Uniform n = 1 n = 2 n = 3 n = 4 n = 5 n = 6 Average Table 3 : Aligned training with different gradient scaling ratios γ : 1 on IWSLT'14 De-En.

For each model we report the BLEU4 score evaluated with a uniformly sampled exit n ∼ U([1..6]) for each token and a fixed exit n ∈ [1..6].

The average corresponds to the average BLEU4 of all fixed exits.

This section details the computation of the FLOPS we report.

The per token FLOPS are for the decoder network only since we use an encoder of the same size for all models.

We breakdown the FLOPS of every operation in Algorithm 2 (blue front of the algorithmic statement).

We omit non-linearities, normalizations and residual connections.

The main operations we account for are dot-products and by extension matrix-vector products since those represent the vast majority of FLOPS (we assume batch size one to simplify the calculation).

Table 4 : FLOPS of basic operations, key parameters and variables for the FLOPS estimation.

With this breakdown, the total computational cost at time-step t of a decoder block that we actually go through, denoted with FC, is:

where the cost of mapping the source' keys and values is incurred the first time the block is called (flagged with FirstCall).

This occurs at t = 1 for the baseline model but it is input-dependent with depth adaptive estimation and may never occur if all tokens exit early.

If skipped, a block still has to compute the keys and value of the self-attention block so the selfattention of future time-steps can function.

We will denote this cost with FS and we have FS = 4d Confidence thresholding: FP(t, q(t)) = 2q(t)V d d

For a set of source sequences {x (i) } i∈I and generated hypotheses {y (i) } i∈I , the average flops per token is: Baseline (N blocks):

Adaptive depth:

q(t)FC(x (i) , t) + (N − q(t))FS + FP(t, q(t)) + 2V d d

In the case of confidence thresholding the final output prediction cost (2V d d ) is already accounted for in the exit prediction cost FP.

<|TLDR|>

@highlight

Sequence model that dynamically adjusts the amount of computation for each input.