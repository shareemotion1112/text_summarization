We propose vq-wav2vec to learn discrete representations of audio segments through a wav2vec-style self-supervised context prediction task.

The algorithm uses either a gumbel softmax or online k-means clustering to quantize the dense representations.

Discretization enables the direct application of algorithms from the NLP community which require discrete inputs.

Experiments show that BERT pre-training achieves a new state of the art on TIMIT phoneme classification and WSJ speech recognition.

Learning discrete representations of speech has gathered much recent interest (Versteegh et al., 2016; Dunbar et al., 2019) .

A popular approach to discover discrete units is via autoencoding (Tjandra et al., 2019; Eloff et al., 2019; Chorowski et al., 2019) sometimes coupled with an autoregressive model .

Another line of research is to learn continuous speech representations in a self-supervised way via predicting context information (Chung & Glass, 2018; van den Oord et al., 2018; Schneider et al., 2019) .

In this paper, we combine these two lines of research by learning discrete representations of speech via a context prediction task instead of reconstructing the input.

This enables us to directly apply well performing NLP algorithms to speech data ( Figure 1a ).

The vq-wav2vec encoder maps raw audio (X ) to a dense representation (Z) which is quantized (q) toẐ and aggregated into context representations (C); training requires future time step prediction.

(b) Acoustic models are trained by quantizing the raw audio with vq-wav2vec, then applying BERT to the discretized sequence and feeding the resulting representations into the acoustic model to output transcriptions.

Our new discretization algorithm, vq-wav2vec, learns discrete representations of fixed length segments of audio signal by utilizing the wav2vec loss and architecture (Schneider et al, 2019; §2) .

To choose the discrete variables, we consider a Gumbel-Softmax approach (Jang et al., 2016) as well as online k-means clustering, similar to VQ-VAE (Oord et al., 2017; Eloff et al., 2019; §3) .

We then train a Deep Bidirectional Transformer (BERT; Devlin et al., 2018; on the discretized unlabeled speech data and input these representations to a standard acoustic model (Figure 1b; §4) .

Our experiments show that BERT representations perform better than log-mel filterbank inputs as well as dense wav2vec representations on both TIMIT and WSJ benchmarks.

Discretization of audio enables the direct application of a whole host of algorithms from the NLP literature to speech data.

For example, we show that a standard sequence to sequence model from the NLP literature can be used to perform speech recognition over discrete audio tokens ( §5, §6).

2.1 WAV2VEC

wav2vec (Schneider et al., 2019) learns representations of audio data by solving a self-supervised context-prediction task with the same loss function as word2vec (Mikolov et al., 2013; van den Oord et al., 2018) .

The model is based on two convolutional neural networks where the the encoder produces a representation z i for each time step i at a rate of 100 Hz and the aggregator combines multiple encoder time steps into a new representation c i for each time step i.

Given an aggregated representation c i , the model is trained to distinguish a sample z i+k that is k steps in the future from distractor samplesz drawn from a distribution p n , by minimizing the contrastive loss for steps k = 1, . . .

, K:

where T is the sequence length, σ(x) = 1/(1 + exp(−x)), and where σ(z i+k h k (c i )) is the probability of z i+k being the true sample.

We consider a step-specific affine transformation et al., 2018) .

We optimize the loss L = K k=1 L k , summing (1) over different step sizes.

After training, the representations produced by the context network c i are input to the acoustic model instead of log-mel filterbank features.

BERT (Devlin et al., 2018 ) is a pre-training approach for NLP tasks, which uses a transformer encoder model to build a representation of text.

Transformers uses self-attention to encode the input sequence as well as an optional source sequence (Vaswani et al., 2017) .

The original BERT model combined two tasks for training: first, masked language modeling randomly removes some of the input tokens and the model has to predict those missing tokens.

Second, next sentence prediction splices two different text passages together into a single example and the model needs to predict whether the passages are from the same document.

Our approach, vq-wav2vec, learns vector quantized (VQ) representations of audio data using a future time-step prediction task.

We follow the same architectual choices as wav2vec ( §2.1) with two convolutional networks f : X → Z and g :Ẑ → C for feature extraction and aggregation, as well as a new quantization module q : Z →Ẑ to build discrete representations ( Figure 1a ).

We first map 30ms segments of raw speech to a dense feature representation z at a stride of 10ms using the encoder network f .

Next, the quantizer (q) turns these dense representations into discrete indices which are mapped to a reconstructionẑ of the original representation z. We feedẑ into the aggregator g and optimize the same context prediction task as wav2vec outlined in §2.1.

The quantization module replaces the original representation z byẑ = e i from a fixed size codebook e ∈ R V ×d which contains V representations of size d.

We consider the Gumbel-Softmax which is a differentiable approximation of the argmax for computing one-hot representations ( §3.1; Figure 2a ) as well as online k-means clustering, similar to the vector quantized variational autoencoder (VQ-VAE; Oord et al., 2017; §3.2; Figure 2b ).

Finally, we perform multiple vector quantizations over different parts of z to mitigate mode collapse ( §3.3).

The Gumbel-Softmax (Gumbel, 1954; Jang et al., 2016; Maddison et al., 2014) enables selecting discrete codebook variables in a fully differentiable way and we use the straight-through estimator of Jang et al. (2016) .

Given the dense representation z, we apply a linear layer, followed by a ReLU and another linear which outputs l ∈ R V logits for the Gumbel-Softmax.

At inference, we simply pick the largest index in l. At training, the output probabilities for choosing the j-th variable are

where v = − log(− log(u)) and u are uniform samples from U(0, 1).

During the forward pass, i = argmax j p j and in the backward pass, the true gradient of the Gumbel-Softmax outputs is used.

The vector quantization approach of van den Oord et al. (2017) is an alternative to making the index selection procedure fully differentiable.

Different to their setup, we optimize a future time step prediction loss instead of the reconstruction loss of an autoencoder.

We choose the codebook variable representation by finding the closest variable to the input features z in terms of the Euclidean distance, yielding i = argmin j z − e j 2 2 .

During the forward pass, we selectẑ = e i by choosing the corresponding variable from the codebook.

We obtain gradients for the encoder network by back-propagating dL wav2vec /dẑ (van den Oord et al., 2017) .

The final loss has two additional terms:

where sg(x) ≡ x, d dx sg(x) ≡ 0 is the stop gradient operator and γ is a hyper-parameter.

The first term is the future prediction task and gradients do not change the codebook because of the straightthrough gradient estimation of mapping z toẑ.

The second term sg(z) −ẑ 2 moves the codebook vectors closer to the encoder output, and the third term z − sg(ẑ) 2 makes sure that the encoder outputs are close to a centroid (codeword).

So far, we considered replacing the encoder feature vector z by a single entry e i in the codebook.

This is prone to mode collapse where only some of the codewords are actually used.

Previously, this problem has been mitigated by workarounds such as re-initializing codewords or applying additional regularizers to the loss function (Caron et al., 2019) .

In the following, we describe another strategy where we independently quantize partitions of z, similar to product quantization (Jegou et al., 2011) .

This results in larger dictionaries and increased downstream performance (Appendix A).

The dense feature vector z ∈ R d is first organized into multiple groups G into the matrix form z ∈ R G×(d/G) .

We then represent each row by an integer index, and hence can represent the full feature vector by the indices i ∈ [V ] G , where V again denotes the possible number of variables for this particular group and each element i j corresponds to a fixed codebook vector.

For each of the G groups, we apply either one of the two VQ approaches ( §3.1 and §3.2).

The codebook itself can be initialized in two possible ways: Codebook variables can be shared across groups, i.e., a particular index in group j would reference the same vector as the same index in group j .

This yields a codebook e ∈ R V ×(G/d) .

In contrast, not sharing the codebook variables yields a codebook of size e ∈ R V ×G×(G/d) .

In practise, we observe that sharing the codebook variables generally yields competitive results to a non-shared representation.

Once we trained a vq-wav2vec model we can discretize audio data and make it applicable to algorithms that require discrete inputs.

One possibility is to use the discretized training data and apply BERT pre-training where the task is to predict masked input tokens based on an encoding of the surrounding context (Devlin et al., 2018) .

Once the BERT model is trained, we can use it to build representations and feed them into an acoustic model to improve speech recognition.

We follow recent advances in BERT training which only use the masked input token prediction .

Since each of the discretized tokens represents around 10 ms of audio it is likely too easy to predict a single masked input token.

We therefore change BERT training by masking spans of consecutive discretized speech tokens, similar to .

To mask the input sequence, we randomly sample p = 0.05 of all tokens to be a starting index, without replacement, and mask M = 10 consecutive tokens from every sampled index; spans may overlap.

This makes the masked token prediction harder and we show later that it improves accuracy over masking individual tokens ( §6.5).

We generally pre-train vq-wav2vec and BERT on the full 960h of Librispeech (Panayotov et al., 2015) and after vq-wav2vec training it is discretized to 345m tokens.

Where indicated we perform ablations on a clean 100h subset which is discretized to 36M tokens.

We evaluate models on two benchmarks: TIMIT (Garofolo et al., 1993b ) is a 5h dataset with phoneme labels and Wall Street Journal (WSJ; Garofolo et al. 1993a ) is a 81h dataset for speech recognition.

For TIMIT, we apply the standard evaluation protocol and consider 39 different phonemes.

For WSJ, we train acoustic models directly on 31 graphemes, including the English alphabet, the apostrophe, the silence token and tokens for repeating characters.

We adapt the fairseq implmentation of wav2vec (Schneider et al., 2019; and use vqwav2vec/wav2vec models with 34 × 10 6 parameters.

The encoder has 8 layers with 512 channels each, kernel sizes (10,8,4,4,4,1,1,1) and strides (5,4,2,2,2,1,1,1), yielding a total stride of 160.

Each layer contains a convolution, followed by dropout, group normalization with a single group (Wu & He, 2018 ) and a ReLU non-linearity.

The aggregator is composed of 12 layers, with 512 channels, stride 1, and kernel sizes starting at 2 and increasing by 1 for every subsequent layer.

The block structure is the same as for the encoder network, except we introduce skip connections between each subsequent block.

We train with the wav2vec context prediction loss (Equation 1) for 400k updates, predicting K = 8 steps into the future and sample 10 negatives from the same audio example.

Training is warmed up for 500 steps where the learning rate is increased from 1 × 10 −7 to 5 × 10 −3 , and then annealed to 1e-06 using a cosine schedule (Loshchilov & Hutter, 2016) .

The batch size is 10, and we crop a random section of 150,000 frames for each example (approximately 9.3 seconds for 16kHz sampling rate).

All models are trained on 8 GPUs.

For ablations and experiments on the 100h Librispeech subset, we use a smaller model with kernels (10,8,4,4,4) and strides (5,4,2,2,2) in the encoder and seven convolutional layers with stride one and kernel size three in the aggregator.

This model is trained for 40k updates.

Gumbel-Softmax Models.

We use G = 2 groups and V = 320 latents per group and the linear layer projects the features produced by the encoder into G · V = 640 logits.

The Gumbel-Softmax produces a one-hot vector for each group G. The temperature τ is linearly annealed from 2 to 0.5 over the first 70% of updates and then kept constant at 0.5.

This enables the model to learn which latents work best for each input before committing to a single latent.

After training this model on 960h of Librispeech and quantizing the training dataset, we are left with 13.5k unique codewords combinations (out of V G = 102k possible codewords).

k-means Models.

We use G = 2 groups and V = 320 variables per group.

vq-wav2vec on full Librispeech yields 23k unique codewords.

Following van den Oord et al. (2017), we found γ = 0.25 to be a robust choice for balancing the VQ auxiliary loss.

BERT base models have 12 layers, model dimension 768, inner dimension (FFN) 3072 and 12 attention heads (Devlin et al., 2018) .

The learning rate is warmed up over the first 10,000 updates to a peak value of 1 × 10 −5 , and then linearly decayed over a total of 250k updates.

We train on 128 GPUs with a batch size of 3072 tokens per GPU giving a total batch size of 393k tokens (Ott et al., 2018) .

Each token represents 10ms of audio data.

BERT small.

For ablations we use a smaller setup with model dimension 512, FFN size 2048, 8 attention heads and dropout 0.05.

Models are trained for 250k updates with a batch size of 2 examples per GPU.

We use wav2letter as accoustic model (Collobert et al., 2016; and train for 1k epochs on 8 GPUs for both TIMIT and WSJ using the auto segmentation criterion.

For decoding the emissions from the acoustic model on WSJ we use a lexicon as well as a separate language model trained on the WSJ language modeling data only.

We consider a 4-gram KenLM language model (Heafield et al., 2013) and a character based convolutional language model (Likhomanenko et al., 2019) and tune the models with the same protocol as Schneider et al. (2019) .

We first evaluate on the WSJ speech recognition benchmark.

We train a vq-wav2vec model on the unlabeled version of Librispeech, then discretize the same data with the resulting model to estimate a BERT model.

Finally, we train a wav2letter acoustic model on WSJ by inputting either the BERT or vq-wav2vec representations instead of log-mel filterbanks.

2 We compare to various results from the literature, including wav2vec (Schneider et al., 2019) and we consider three setups: performance without any language model (No LM), with an n-gram LM (4-gram LM) and with a character convolutional LM (Char ConvLM).

We report the accuracy of wav2letter with log-mel filterbanks as input (Baseline) and wav2vec.

For vq-wav2vec we first experiment with the Gumbel-Softmax, with and without a BERT base model ( §5.3).

Table 1 shows that vq-wav2vec together with BERT training can achieve a new state of the art of 2.34 WER on nov92.

Gains are largest when no language model is used which is the fastest setting.

vq-wav2vec with Gumbel-Softmax uses only 13.5k distinct codewords to represent the audio signal and this limited set of codewords is not sufficient to outperform the baseline.

However, it does enable training BERT models which require a relatively small vocabulary.

Next, we compare Gumbel-Softmax to k-means for vector quantization.

For this experiment we use the faster to train BERT small configuration ( §5.3).

We also train a vq-wav2vec k-means model with a very large number of codewords (39m) to test whether a more expressive model can close dev PER test PER CNN + TD-filterbanks (Zeghidour et al., 2018) 15.6 18.0 Li-GRU + fMLLR (Ravanelli et al., 2018)

-14.9 wav2vec (Schneider et al., 2019) 12.9 14.7

Baseline (log-mel) 16.9 17.6 vq-wav2vec, gumbel 15.34 17.78 + BERT small 9.64 11.64 vq-wav2vec, k-means 15.65 18.73 + BERT small 9.80 11.40 Table 3 : TIMIT phoneme recognition in terms of phoneme error rate (PER).

All our models use the CNN-8L-PReLU-do0.7 architecture (Zeghidour et al., 2018 Table 4 : Librispeech results for a standard sequence to sequence model trained on discretized audio without BERT pre-training and results from the literature.

All results are without a language model.

the gap to wav2vec.

Table 2 shows that Gumbel-Softmax and k-means clustering perform relatively comparably: in the no language model setup without BERT, Gumbel-Softmax is more accurate than k-means but these differences disappear with BERT.

For 4-gram LM setup, k-means is better but those differences disappear again after BERT training.

Finally, the large codeword model can substantially reduce the gap to the original wav2vec model.

Next, we experiment on the much smaller TIMIT phoneme recognition task where we also pre-train vq-wav2vec on the full Librispeech corpus.

Table 3 shows that vq-wav2vec and BERT achieve a new state of the art of 11.67 PER which corresponds to a 21% reduction in error over the previous best result of wav2vec.

So far we used vq-wav2vec to train BERT on discretized speech.

However, once the audio is discretized we can also train a standard sequence to sequence model to perform speech recognition.

In preliminary experiments, we trained an off-the-shelf Big Transformer (Vaswani et al., 2017; on the vq-wav2vec Gumbel-Softmax discretized Librispeech corpus and evaluated on the Librispeech dev/test sets; we use a 4k BPE output vocabulary (Sennrich et al., 2016) .

Table 4 shows that results are promising, even though they are not as good as the state of the art (Park et al., 2019) which depends on data augmentation that we do not use.

Next, we investigate how well vq-wav2vec can compress the audio data.

Specifically, we train models with different numbers of groups G and variables V to vary the size of the possible codebook size V G and measure accuracy on TIMIT phoneme recognition without BERT training.

We measure compression with the bitrate r · G log 2 V at sampling rate r = 100Hz and report the tradeoff between bitrate and accuracy on our phoneme recognition task.

We experiment with vq-wav2vec k-means and train models with 1,2,4,8,16 and 32 groups, using 40,80,160,...,1280 variables, spanning a bitrate range from 0.53 kbit/s (G = 1, V = 40) to 33.03 kbit/s (G = 32, V = 1280).

We place the quantization module after the aggregator module and train all models in the small vq-wav2vec setup ( §5.2) on the 100h clean Librispeech subset.

As baselines, we consider various lossy compression algorithms applied to the TIMIT audio data and train wav2vec models on the resulting audio: Codec2 3 as a low bitrate codec, Opus (Terriberry & Vos, 2012) as a medium bitrate codec and MP3 and Ogg Vorbis (Montgomery, 2004) as high bitrate codecs.

We use the whole spectrum of both variable and constant bitrate settings of the codecs; we encode and decode with ffmpeg (ffmpeg developers, 2016).

Figure 3 shows the trade-off between the bitrate and TIMIT accuracy.

Acoustic models on vq-wav2vec achieve the best results across most bitrate settings.

Table 5a shows that masking entire spans of tokens performs significantly better than individual tokens (M = 1).

Furthermore, BERT training on discretized audio data is fairly robust to masking large parts of the input (Table 5b ).

vq-wav2vec is a self-supervised algorithm that quantizes unlabeled audio data which makes it amenable to algorithms requiring discrete data.

This approach improves the state of the art on the WSJ and TIMIT benchmarks by leveraging BERT pre-training.

In future work, we plan to apply other algorithms requiring discrete inputs to audio data and to explore self-supervised pre-training algorithms which mask part of the continuous audio input.

Another future work avenue is to finetune the pre-trained model to output transcriptions instead of feeding the pre-trained features to a custom ASR model.

We investigate the relationship between number of variables V and groups G. Table 6 shows that multiple groups are beneficial compared to a single group with a large number of variables.

Table 7 shows that with a single group and many variables, only a small number of codewords survive.

Table 6 : PER on TIMIT dev set for vq-wav2vec models trained on Libri100.

Results are based on three random seeds.

<|TLDR|>

@highlight

Learn how to quantize speech signal and apply algorithms requiring discrete inputs to audio data such as BERT.