Self-attention is a useful mechanism to build generative models for language and images.

It determines the importance of context elements by comparing each element to the current time step.

In this paper, we show that a very lightweight convolution can perform competitively to the best reported self-attention results.

Next, we introduce dynamic convolutions which are simpler and more efficient than self-attention.

We predict separate convolution kernels based solely on the current time-step in order to determine the importance of context elements.

The number of operations required by this approach scales linearly in the input length, whereas self-attention is quadratic.

Experiments on large-scale machine translation, language modeling and abstractive summarization show that dynamic convolutions improve over strong self-attention models.

On the WMT'14 English-German test set dynamic convolutions achieve a new state of the art of 29.7 BLEU.

There has been much recent progress in sequence modeling through recurrent neural networks (RNN; BID54 , convolutional networks (CNN; BID28 BID14 BID7 ) and self-attention models BID40 BID58 .

RNNs integrate context information by updating a hidden state at every time-step, CNNs summarize a fixed size context through multiple layers, while as self-attention directly summarizes all context.

Attention assigns context elements attention weights which define a weighted sum over context representations BID52 BID8 BID36 .

Source-target attention summarizes information from another sequence such as in machine translation while as self-attention operates over the current sequence.

Self-attention has been formulated as content-based where attention weights are computed by comparing the current time-step to all elements in the context FIG0 ).

The ability to compute comparisons over such unrestricted context sizes are seen as a key characteristic of self-attention BID58 .

However, the ability of self-attention to model long-range dependencies has recently come into question BID57 and the unlimited context size is computationally very challenging due to the quadratic complexity in the input length.

Furthermore, in practice long sequences require the introduction of hierarchies .In this paper, we introduce lightweight convolutions which are depth-wise separable BID51 BID7 , softmax-normalized and share weights over the channel dimension.

The result is a convolution with several orders of magnitude fewer weights than a standard nonseparable convolution.

Different to self-attention, lightweight convolutions reuse the same weights for context elements, regardless of the current time-step.

Dynamic convolutions build on lightweight convolutions by predicting a different convolution kernel at every time-step.

The kernel is a function of the current time-step only as opposed to the entire context as in self-attention ( FIG0 .

Dynamic convolutions are similar to locally connected layers in the sense that the weights change at every position, however, the difference is that weights are dynamically generated by the model rather than fixed after training BID30 BID56 BID6 .

Our approach also bears similarity to location-based attention which does not access the context to determine attention weights, however, we do not directly take the attention weights from the previous time-step into account BID8 BID36 .

BID49 reduce complexity by performing attention within blocks of the input sequence and BID48 BID50 perform more fine-grained attention over each feature.

BID47 and BID17 use input-dependent filters for text classification tasks.

Our experiments show that lightweight convolutions perform competitively to strong self-attention results and that dynamic convolutions can perform even better.

On WMT English-German translation dynamic convolutions achieve a new state of the art of 29.7 BLEU, on WMT English-French they match the best reported result in the literature, and on IWSLT German-English dynamic convolutions outperform self-attention by 0.8 BLEU.

Dynamic convolutions achieve 20% faster runtime than a highly-optimized self-attention baseline.

For language modeling on the Billion word benchmark dynamic convolutions perform as well as or better than self-attention and on CNN-DailyMail abstractive document summarization we outperform a strong self-attention model.

We first outline sequence to sequence learning and self-attention.

Our work builds on non-separable convolutions as well as depthwise separable convolutions.

Sequence to sequence learning maps a source sequence to a target sequence via two separate networks such as in machine translation BID54 .

The encoder network computes representations for the source sequence such as an English sentence and the decoder network autoregressively generates a target sequence based on the encoder output.

The self-attention module of BID58 applies three projections to the input X ∈ R n×d to obtain key (K), query (Q), and value (V) representations, where n is the number of time steps, d the input/output dimension ( FIG1 .

It also defines a number of heads H where each head can learn separate attention weights over d k features and attend to different positions.

The module computes dot-products between key/query pairs, scales to stabilize training, and then softmax normalizes the result.

Finally, it computes a weighted sum using the output of the value projection (V): DISPLAYFORM0 Depthwise convolutions perform a convolution independently over every channel.

The number of parameters can be reduced from d 2 k to dk where k is the kernel width.

The output O ∈ R n×d of a depthwise convolution with weight W ∈ R d×k for element i and output dimension c is defined as: DISPLAYFORM1

In this section, we introduce LightConv, a depthwise convolution which shares certain output channels and whose weights are normalized across the temporal dimension using a softmax.

Compared to self-attention, LightConv has a fixed context window and it determines the importance of context elements with a set of weights that do not change over time steps.

We will show that models equipped with lightweight convolutions show better generalization compared to regular convolutions and that they can be competitive to state-of-the-art self-attention models ( §6).

This is surprising because the common belief is that content-based self-attention mechanisms are necessary to obtaining stateof-the-art results in natural language processing applications.

Furthermore, the low computational profile of LightConv enables us to formulate efficient dynamic convolutions ( §4).LightConv computes the following for the i-th element in the sequence and output channel c: DISPLAYFORM0 Weight sharing.

We tie the parameters of every subsequent number of d H channels, which reduces the number of parameters by a factor of d H .

As illustration, a regular convolution requires 7,340,032 (d 2 × k) weights for d = 1024 and k = 7, a depthwise separable convolution has 7,168 weights (d × k), and with weight sharing, H = 16, we have only 112 (H × k) weights.

We will see that this vast reduction in the number of parameters is crucial to make dynamic convolutions possible on current hardware.

BID60 ties the weights of all channels (H = 1).

H×k across the temporal dimension k using a softmax operation: DISPLAYFORM0 Module.

FIG1 shows the architecture of the module where we integrate LightConv.

We first apply an input projection mapping from dimension d to 2d, followed by a gated linear unit (GLU; , and the actual lightweight convolution.

The GLU uses half of the inputs as gates by applying sigmoid units and then computes a pointwise product with the other inputs.

We also apply an output projection of size W O ∈ R d×d to the output of LightConv.

We found DropConnect to be a good regularizer for the LightConv module BID59 .

Specifically, we drop every entry of the normalized weights sof tmax(W ) with probability p and divide it by 1 − p during training.

This amounts to removing some of the temporal information within a channel.

Implementation.

Existing CUDA primitives for convolutions did not perform very well to implement LightConv and we found the following solution faster on short sequences: We copy and expand the normalized weights W ∈ R H×k to a band matrix of size BH × n × n, where B is the batch size.

We then reshape and transpose the inputs to size BH × n × d H , and perform a batch matrix multiplication to get the outputs.

We expect a dedicated CUDA kernel to be much more efficient.

A dynamic convolution has kernels that vary over time as a learned function of the individual time steps.

A dynamic version of standard convolutions would be impractical for current GPUs due to their large memory requirements.

We address this problem by building on LightConv which drastically reduces the number of parameters ( §3).DynamicConv takes the same form as LightConv but uses a time-step dependent kernel that is computed using a function f : DISPLAYFORM0 we model f with a simple linear module with learned weights DISPLAYFORM1 .

Similar to self-attention, DynamicConv changes the weights assigned to context elements over time.

However, the weights of DynamicConv do not depend on the entire context, they are a function of the current time-step only.

Self-attention requires a quadratic number of operations in the sentence length to compute attention weights, while the computation of dynamic kernels for DynamicConv scales linearly in the sequence length.

Our experiments ( §6) show that models using DynamicConv match or exceed the performance of state-of-the-art models that use context-based self-attention.

This challenges the typical intuitions about the importance of content-based self-attention in natural language processing applications.

We use an encoder-decoder architecture for sequence to sequence learning BID54 and we closely follow the architectural choices presented in BID58 .

Our self-attention baseline is the fairseq re-implementation of the Transformer Big architecture .

The encoder and decoder networks have N blocks each.

Encoder blocks contain two sub-blocks: The first is a self-attention module ( §2), a LightConv module (3), or a DynamicConv module ( §4).

The second sub-block is a feed-forward module: DISPLAYFORM0 unless otherwise stated.

Sub-blocks are surrounded by residual connections BID22 and layer normalization BID1 .Decoder blocks are identical except that they have an additional source-target attention sub-block between the self-attention and feed-forward module.

The source-target attention is equivalent to the self-attention module, except that the values and keys are projections over the encoder output for each source word.

Words are fed to the encoder and decoder networks in d dimensional embeddings.

We add sinusoidal position embeddings to encode the absolute position of each word in the sequence BID58 .

The model computes a distribution over vocabulary V by transforming the decoder output via a linear layer with weights W V ∈ R d×V followed by softmax normalization.

LightConv and DynamicConv are identical to Transformer Big, except that self-attention modules are swapped with either fixed or dynamic convolutions.

These models also use fewer parameters per block (cf.

FIG1 and FIG1 ) and we therefore increase the number of blocks to N = 7 for the encoder to roughly match the parameter count of Transformer Big.

We generally set H = 16.

Both LightConv and DynamicConv set the the encoder and decoder kernel sizes to 3, 7, 15, 31x4 for each block respectively; except for the decoder where we have only three top layers with kernel size 31.

To get a thorough understanding of the limitations of LightConv and DynamicConv we evaluate on three different tasks: machine translation, language modeling and abstractive summarization.

Machine Translation.

We report results on four benchmarks: For WMT English to German (EnDe) we replicate the setup of BID58 , based on WMT'16 training data with 4.5M sentence pairs, we validate on newstest2013 and test on newstest2014.

3 The vocabulary is a 32K joint source and target byte pair encoding (BPE; BID44 .

For WMT English to French (EnFr), we borrow the setup of BID15 with 36M training sentence pairs from WMT'14, validate on newstest2012+2013 and test on newstest2014.

The 40K vocabulary is based on a joint source and target BPE factorization.

For WMT English to Chinese (Zh-En), we pre-process the WMT'17 training data following BID21 resulting in 20M sentence pairs.

We develop on devtest2017 and test on newstest2017.

For IWSLT'14 German-English (De-En) we replicate the setup of for 160K training sentence pairs and 10K joint BPE vocabulary.

For this benchmark only, data is lowercased.

For WMT En-De, WMT En-Fr, we measure case-sensitive tokenized BLEU.

4 For WMT En-De only we apply compound splitting similar to BID58 .

For WMT Zh-En we measure detokenized BLEU to be comparable to BID21 .

5 We train three random initializations of a each configuration and report test accuracy of the seed which resulted in the highest validation BLEU.

Ablations are conducted on the validation set and we report the mean BLEU and standard deviation on this set.

WMT En-De, WMT En-Fr are based on beam search with a beam width of 5, IWSLT uses beam 4, and WMT Zh-En beam 8 following BID21 .

For all datasets, we tune a length penalty as well as the number of checkpoints to average on the validation set.

Language Modeling.

We evaluate on the large-scale Billion word dataset BID4 which contains 768M tokens and has a vocabulary of nearly 800K types.

Sentences in this dataset are shuffled and we batch sentences independently of each other.

Models are evaluated in terms of perplexity on the valid and test portions.

Summarization.

We test the model's ability to process long documents on the CNN-DailyMail summarization task BID23 BID37 comprising over 280K news articles paired with multi-sentence summaries.

Articles are truncated to 400 tokens BID43 and we use a BPE vocabulary of 30K types .

We evaluate in terms of F1-Rouge, that is Rouge-1, Rouge-2 and Rouge-L BID33 .

6 When generating summaries, we follow standard practice in tuning the maximum output length, disallowing repeating the same trigram, and we apply a stepwise length penalty BID40 .

Translation.

We use a dropout rate of 0.3 for WMT En-De and IWSLT De-En, 0.1 for WMT EnFr, and 0.25 for WMT Zh-En.

WMT models are optimized with Adam and a cosine learning rate schedule BID29 BID35 where the learning rate is first linearly warmed up for 10K steps from 10 −7 to 10 −3 and then annealed following a cosine rate with a single cycle.

For IWSLT'14 De-En, we use a schedule based on the inverse square root of the current step BID58 .

We train the WMT models on 8 NVIDIA V100 GPUs for a total of 30K steps on WMT En-De, 40K steps for WMT Zh-En and 80K steps for WMT En-Fr.

For IWSLT De-En we train for 50K steps on a single GPU.We use floating point 16 precision and accumulate the gradients for 16 batches before applying an update , except for IWSLT where we do not accumulate gradients.

Batches contain up to 459K source tokens and the same number of target tokens for both WMT En-De and WMT Zh-En, 655K for En-Fr, and 4K for IWSLT De-En.

We use label smoothing with 0.1 weight for the uniform prior distribution over the vocabulary BID55 BID41 .Language Modeling.

We follow the same setup as for translation but remove the encoder module.

For the Billion word benchmark we use an adaptive softmax output layer to reduce the computational burden of the large vocabulary BID18 BID42 We train on 32 GPUs with batches of 65K tokens for 975K updates.

As optimizer we use Nesterov's accelerated gradient method BID53 ) with a momentum value of 0.99 and we renormalize gradients if their norm exceeds 0.1 BID39 .

The learning rate is linearly warmed up from 10 −7 to 1 for 16K steps and then annealed using a cosine learning rate schedule BID35 with one cycle.

Summarization.

We train with Adam using the cosine learning rate schedule with a warmup of 10K steps and a period of 20K updates.

We use weight decay 1e-3 and dropout 0.3.

We first report results on WMT En-De and WMT En-Fr where we compare to the best results in the literature, most of which are based on self-attention.

Table 1 shows that LightConv performs very competitively and only trails the state of the art result by 0.1 BLEU on WMT En-Fr; the state of the art is based on self-attention .

This is despite the simplicity of LightConv which operates with a very small number of fixed weights over all time steps whereas self-attention computes dot-products with all context elements at every time-step.

DynamicConv outperforms the best known result on WMT En-De by 0.4 BLEU and achieves a new state of the art, whereas on WMT En-Fr it matches the state of the art.

This shows that content-based self-attention is not necessary to achieve good accuracy on large translation benchmarks.

IWSLT is a much smaller benchmark and we therefore switch to a smaller architecture: d f f = 1024, d = 512, and H = 4.

The self-attention baseline on this dataset is the best reported result in the literature TAB1 .

Table 3 : Ablation on WMT English-German newstest2013.

(+) indicates that a result includes all preceding features.

Speed results based on beam size 4, batch size 256 on an NVIDIA P100 GPU.

In this section we evaluate the impact of the various choices we made for LightConv ( §3) and DynamicConv ( §4).

We first show that limiting the maximum context size of self-attention has no impact on validation accuracy ( Table 3 ).

Note that our baseline is stronger than the original result of BID58 .

Next, we replace self-attention blocks with non-separable convolutions (CNN) with kernel size 3 and input/output dimension d = 1024.

The CNN block has no input and output projections compared to the baseline and we add one more encoder layer to assimilate the parameter count.

This CNN with a narrow kernel trails self-attention by 1 BLEU.We improve this result by switching to a depthwise separable convolution (CNN Depthwise) with input and output projections of size d = 1024.

When we progressively increase the kernel width from lower to higher layers then this further improves accuracy.

This narrows the gap to self-attention to only 0.5 BLEU.

DropConnect gives a slight performance improvement and weight sharing does not decrease performance.

Adding softmax normalization to the weights is only 0.3 BLEU below the accuracy of the baseline.

This corresponds to LightConv.

In Appendix A we compare softmaxnormalization to various alternatives.

Finally, dynamic convolutions (DynamicConv) achieve the same validation accuracy as self-attention with slightly fewer parameters and at 20% higher inference speed.

Softmax-normalization is important for DynamicConv since training diverged in our experiments when removing it.

To make the models more comparable, we do not introduce GLU after the input projection.

For comparison, we re-implemented averaged attention networks (AAN; ) which compute a uniform average over past model states instead of a weighted average as in self-attention.

Our re-implementation is efficient: we measure 129 sentences/sec for a base transformer-AAN on newstest2014 compared to 20 sentences/sec for .

Table 3 shows that our models outperform this approach.

Note that AANs still use self-attention in the encoder network while as our approach does away with self-attention both in the encoder and decoder.

As second task we consider language modeling on the Billion word benchmark.

The self-attention baseline has N = 16 blocks, each with a self-attention module and a feed-forward module using d f f = 4096 and d = 1024.

DynamicConv uses N = 17 blocks to assimilate the parameter count and we use kernel sizes 15x2, 31x4 and 63x11.

TAB3 shows that DynamicConv achieves slightly better perplexity than our self-attention baseline which is very competitive.

BID58 .

TAB4 shows that LightConv outperforms the self-attention baseline as well as comparable previous work and DynamicConv performs even better.

We also show results for a reinforcement learning approach BID3 and note that RL is equally applicable to our architecture.

We presented lightweight convolutions which perform competitively to the best reported results in the literature despite their simplicity.

They have a very small parameter footprint and the kernel does not change over time-steps.

This demonstrates that self-attention is not critical to achieve good accuracy on the language tasks we considered.

Dynamic convolutions build on lightweight convolutions by predicting a different kernel at every time-step, similar to the attention weights computed by self-attention.

The dynamic weights are a function of the current time-step only rather than the entire context.

Our experiments show that lightweight convolutions can outperform a strong self-attention baseline on WMT'17 Chinese-English translation, IWSLT'14 German-English translation and CNNDailyMail summarization.

Dynamic convolutions improve further and achieve a new state of the art on the test set of WMT'14 English-German.

Both lightweight convolution and dynamic convolution are 20% faster at runtime than self-attention.

On Billion word language modeling we achieve comparable results to self-attention.

We are excited about the future of dynamic convolutions and plan to apply them to other tasks such as question answering and computer vision where inputs are even larger than the tasks we considered in this paper.

We compare our proposed softmax-normalization of weights to other alternatives in Table 6 .

For each setting, we use three seeds and report the mean and the standard deviation of the BLEU score on WMT English-German newstest2013.

The softmax and norms are computed over the kernel dimension.

Simply using the absolute value of the weights or squaring them does not make the training more stable, which shows that having all non-negative weights is not critical.

Dividing the weights by the 2 -norm or bounding the weights with sigmoid or the hyperbolic tangent function also stablizes the training procedure; however, the softmax-normalization performs best.

26.7 ± 0.2 Table 6 : Alternatives to softmax-normalization in DynamicConv on WMT English-German newstest2013 ( = 10 −6 ).

In this section we compare DynamicConv to current non-autoregressive models in the literature.

We measured generation speed for DynamicConv on a P100 GPU using batch size one to be comparable with other results.

Results in the literature are based on either NVIDIA GTX-1080 GPUs or P100 GPUs.

The effects of different GPU types is likely negligible because GPUs are vastly underutilized with batch size one.

Table 7 shows that DynamicConv with a single decoder layer outperforms all previously reported non-autoregressive results both in terms of speed as well as accuracy.

Only two non-autoregressive concurrent efforts BID20 BID32 achieve a speedup over DynamicConv with a small drop in BLEU.

Notably, both BID20 and BID32 distill autoregressive models into non-autoregressive models BID24 , in order to improve their results.

Model (batch size = 1, beam size = 1) Param BLEU Sent/sec Tok/sec NAT (+ FT) (Gu et al., 2018) -17.7 25.6 -NAT (+ FT + NPD=10) BID19 -18.7 12.7 -NAT (+ FT + NPD=100) BID19 (6-decoder layers (k=3,7,15,31,31,31) ) 200M 28.5 3.9 110.9 Table 7 : Inference speed of non-autoregressive models and small decoder versions of DynamicConv on WMT English-German newstest2014.

For some models, the decoding speed (sent/sec) is derived by taking the inverse of the sentence generation latency in the literature.

<|TLDR|>

@highlight

Dynamic lightweight convolutions are competitive to self-attention on language tasks.