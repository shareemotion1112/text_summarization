Capturing high-level structure in audio waveforms is challenging because a single second of audio spans tens of thousands of timesteps.

While long-range dependencies are difficult to model directly in the time domain, we show that they can be more tractably modelled in two-dimensional time-frequency representations such as spectrograms.

By leveraging this representational advantage, in conjunction with a highly expressive probabilistic model and a multiscale generation procedure, we design a model capable of generating high-fidelity audio samples which capture structure at timescales which time-domain models have yet to achieve.

We demonstrate that our model captures longer-range dependencies than time-domain models such as WaveNet across a diverse set of unconditional generation tasks, including single-speaker speech generation, multi-speaker speech generation, and music generation.

Audio waveforms have complex structure at drastically varying timescales, which presents a challenge for generative models.

Local structure must be captured to produce high-fidelity audio, while longrange dependencies spanning tens of thousands of timesteps must be captured to generate audio which is globally consistent.

Existing generative models of waveforms such as WaveNet (van den Oord et al., 2016a) and SampleRNN (Mehri et al., 2016) are well-adapted to model local dependencies, but as these models typically only backpropagate through a fraction of a second, they are unable to capture high-level structure that emerges on the scale of several seconds.

We introduce a generative model for audio which captures longer-range dependencies than existing end-to-end models.

We primarily achieve this by modelling 2D time-frequency representations such as spectrograms rather than 1D time-domain waveforms ( Figure 1 ).

The temporal axis of a spectrogram is orders of magnitude more compact than that of a waveform, meaning dependencies that span tens of thousands of timesteps in waveforms only span hundreds of timesteps in spectrograms.

In practice, this enables our spectrogram models to generate unconditional speech and music samples with consistency over multiple seconds whereas time-domain models must be conditioned on intermediate features to capture structure at similar timescales.

Modelling spectrograms can simplify the task of capturing global structure, but can weaken a model's ability to capture local characteristics that correlate with audio fidelity.

Producing high-fidelity audio has been challenging for existing spectrogram models, which we attribute to the lossy nature of spectrograms and oversmoothing artifacts which result from insufficiently expressive models.

To reduce information loss, we model high-resolution spectrograms which have the same dimensionality as their corresponding time-domain signals.

To limit oversmoothing, we use a highly expressive autoregressive model which factorizes the distribution over both the time and frequency dimensions.

Modelling both fine-grained details and high-level structure in high-dimensional distributions is known to be challenging for autoregressive models.

To capture both local and global structure in spectrograms with hundreds of thousands of dimensions, we employ a multiscale approach which generates spectrograms in a coarse-to-fine manner.

A low-resolution, subsampled spectrogram that captures high-level structure is generated initially, followed by an iterative upsampling procedure that adds high-resolution details. (1x, 5x, 25x, 125x) Figure 1: Spectrogram and waveform representations of the same 4 second audio signal.

The waveform spans nearly 100,000 timesteps whereas the temporal axis of the spectrogram spans roughly 400.

Complex structure is nested within the temporal axis of the waveform at various timescales, whereas the spectrogram has structure which is smoothly spread across the time-frequency plane.

Combining these representational and modelling techniques yields a highly expressive and broadly applicable generative model of audio.

Our contributions are are as follows:

• We introduce MelNet, a generative model for spectrograms which couples a fine-grained autoregressive model and a multiscale generation procedure to jointly capture local and global structure.

• We show that MelNet is able to model longer-range dependencies than existing time-domain models.

Additionally, we include an ablation to demonstrate that multiscale modelling is essential for modelling long-range dependencies.

• We demonstrate that MelNet is broadly applicable to a variety of audio generation tasks, including unconditional speech and music generation.

Furthermore, MelNet is able to model highly multimodal data such as multi-speaker and multilingual speech.

We briefly present background regarding spectral representations of audio.

Audio is represented digitally as a one-dimensional, discrete-time signal y = (y 1 , . . .

, y n ).

Existing generative models for audio have predominantly focused on modelling these time-domain signals directly.

We instead model spectrograms, which are two-dimensional time-frequency representations which contain information about how the frequency content of an audio signal varies through time.

Spectrograms are computed by taking the squared magnitude of the short-time Fourier transform (STFT) of a time-domain signal, i.e. x = STFT(y) 2 .

The value of x ij (referred to as amplitude or energy) corresponds to the squared magnitude of the jth element of the frequency response at timestep i.

Each slice x i, * is referred to as a frame.

We assume a time-major ordering, but following convention, all figures are displayed transposed and with the frequency axis inverted.

Time-frequency representations such as spectrograms highlight how the tones and pitches within an audio signal vary through time.

Such representations are closely aligned with how humans perceive audio.

To further align these representations with human perception, we convert the frequency axis to the Mel scale and apply an elementwise logarithmic rescaling of the amplitudes.

Roughly speaking, the Mel transformation aligns the frequency axis with human perception of pitch and the logarithmic rescaling aligns the amplitude axis with human perception of loudness.

Spectrograms are lossy representations of their corresponding time-domain signals.

The Mel transformation discards frequency information and the removal of the STFT phase discards temporal information.

When recovering a time-domain signal from a spectrogram, this information loss manifests as distortion in the recovered signal.

To minimize these artifacts and improve the fidelity of generated audio, we model high-resolution spectrograms.

The temporal resolution of a spectrogram can be increased by decreasing the STFT hop size, and the frequency resolution can be increased by The context x <ij (grey) for the element x ij (black) is encoded using 4 RNNs.

Three of these are used in the time-delayed stack to extract features from preceding frames.

The fourth is used in the frequency-delayed stack to extract features from all preceding elements within the current frame.

Each arrow denotes an individual RNN cell and arrows of the same color use shared parameters.

increasing the number of Mel channels.

Generated spectrograms are converted back to time-domain signals using classical spectrogram inversion algorithms.

We experiment with both Griffin-Lim (Griffin & Lim, 1984 ) and a gradient-based inversion algorithm (Decorsière et al., 2015) , and ultimately use the latter as it generally produced audio with fewer artifacts.

We use an autoregressive model which factorizes the joint distribution over a spectrogram x as a product of conditional distributions.

Given an ordering of the dimensions of x, we define the context x <ij as the elements of x that precede x ij .

We default to a row-major ordering which proceeds through each frame x i, * from low to high frequency, before progressing to the next frame.

The joint density is factorized as

where θ ij parameterizes a univariate density over x ij .

We model each factor distribution as a Gaussian mixture model with K components.

Thus, θ ij consists of 3K parameters corresponding to means

, and mixture coefficients {π ijk } K k=1 .

The resulting factor distribution can then be expressed as

Following the work on Mixture Density Networks (Bishop, 1994) and their application to autoregressive models (Graves, 2013) , θ ij is modelled as the output of a neural network and computed as a function of the context x <ij .

Precisely, for some network f with parameters ψ, we have θ ij = f (x <ij ; ψ).

A maximum-likelihood estimate for the network parameters is computed by minimizing the negative log-likelihood via gradient descent.

To ensure that the network output parameterizes a valid Gaussian mixture model, the network first computes unconstrained parameters {μ ijk ,σ ijk ,π ijk } K k=1 as a vectorθ ij ∈ R 3K , and enforces constraints on θ ij by applying the following transformations:

These transformations ensure the standard deviations σ ijk are positive and the mixture coefficients π ijk sum to one.

To model the distribution in an autoregressive manner, we design a network which computes the distribution over x ij as a function of the context x <ij .

The network architecture draws inspiration from existing autoregressive models for images (Theis & Bethge, 2015; van den Oord et al., 2016c; b; Salimans et al., 2017; Parmar et al., 2018; Child et al., 2019) .

In the same way that these models estimate a distribution pixel-by-pixel over the spatial dimensions of an image, our model estimates a distribution element-by-element over the time and frequency dimensions of a spectrogram.

A noteworthy distinction is that spectrograms are not invariant to translation along the frequency axis, making 2D convolution less desirable than other 2D network primitives which do not assume invariance.

Utilizing multidimensional recurrence instead of 2D convolution has been shown to be beneficial when modelling spectrograms in discriminative settings Sainath & Li, 2016) , which motivates our use of an entirely recurrent architecture.

Similar to Gated PixelCNN (van den Oord et al., 2016b) , the network has multiple stacks of computation.

These stacks extract features from different segments of the input to collectively summarize the full context x <ij :

• The time-delayed stack computes features which aggregate information from all previous frames x <i, * .

• The frequency-delayed stack utilizes all preceding elements within a frame, x i,<j , as well as the outputs of the time-delayed stack, to summarize the full context x <ij .

The stacks are connected at each layer of the network, meaning that the features generated by layer l of the time-delayed stack are used as input to layer l of the frequency-delayed stack.

To facilitate the training of deeper networks, both stacks use residual connections (He et al., 2016) .

The outputs of the final layer of the frequency-delayed stack are used to compute the unconstrained parametersθ.

The time-delayed stack utilizes multiple layers of multidimensional RNNs to extract features from x <i, * , the two-dimensional region consisting of all frames preceding x ij .

Each multidimensional RNN is composed of three one-dimensional RNNs: one which runs forwards along the frequency axis, one which runs backwards along the frequency axis, and one which runs forwards along the time axis.

Each RNN runs along each slice of a given axis, as shown in Figure 2 .

The output of each layer of the time-delayed stack is the concatenation of the three RNN hidden states.

We denote the function computed at layer l of the time-delayed stack (three RNNs followed by concatenation) as F To ensure that h f ij [l] is computed using only elements in the context x <ij , the inputs to the frequencydelayed stack are shifted backwards one step along the frequency axis: h

At the final layer, layer L, a linear map is applied to the output of the frequency-delayed stack to produce the unconstrained Gaussian mixture model parameters, i.e.

To incorporate conditioning information into the model, conditioning features z are simply projected onto the input layer along with the inputs x:

Reshaping, upsampling, and broadcasting can be used as necessary to ensure the conditioning features have the same time and frequency shape as the input spectrogram, e.g. a one-hot vector representation for speaker ID would first be broadcast along both the time and frequency axes.

To improve audio fidelity, we generate high-resolution spectrograms which have the same dimensionality as their corresponding time-domain representations.

Under this regime, a single training example has several hundreds of thousands of dimensions.

Capturing global structure in such high-dimensional distributions is challenging for autoregressive models, which are biased towards capturing local dependencies.

To counteract this, we utilize a multiscale approach which effectively permutes the autoregressive ordering so that a spectrogram is generated in a coarse-to-fine order.

The elements of a spectrogram x are partitioned into G tiers x 1 , . . .

, x G , such that each successive tier contains higher-resolution information.

We define x <g as the union of all tiers which precede x g , i.e. x <g = (x 1 , . . . , x g−1 ).

The distribution is factorized over tiers:

and the distribution of each tier is further factorized element-by-element as described in Section 3.

We explicitly include the parameterization by ψ = (ψ 1 , . . .

, ψ G ) to indicate that each tier is modelled by a separate network.

Figure 5: Schematic showing how tiers of the multiscale model are interleaved and used to condition the distribution for the subsequent tier.

a) The initial tier is generated unconditionally.

b) The second tier is generated conditionally given the the initial tier.

c) The outputs of tiers 1 and 2 are interleaved along the frequency axis and used to condition the generation of tier 3.

d) Tier 3 is interleaved along the time axis with all preceding tiers and used to condition the generation of tier 4.

During training, the tiers are generated by recursively partitioning a spectrogram into alternating rows along either the time or frequency axis.

We define a function split which partitions an input into even and odd rows along a given axis.

The initial step of the recursion applies the split function to a spectrogram x, or equivalently x <G+1 , so that the even-numbered rows are assigned to x G and the odd-numbered rows are assigned to x <G .

Subsequent tiers are defined similarly in a recursive manner:

At each step of the recursion, we model the distribution p(x g | x <g ; ψ g ).

The final step of the recursion models the unconditional distribution over the initial tier p(x 1 ; ψ 1 ).

To model the conditional distribution p(x g | x <g ; ψ g ), the network at each tier needs a mechanism to incorporate information from the preceding tiers x <g .

To this end, we add a feature extraction network which computes features from x <g which are used condition the generation of x g .

We use a multidimensional RNN consisting of four one-dimensional RNNs which run bidirectionally along slices of both axes of the context x <g .

A layer of the feature extraction network is similar to a layer of the time-delayed stack, but since the feature extraction network is not causal, we include an RNN which runs backwards along the time axis and do not shift the inputs.

The hidden states of the RNNs in the feature extraction network are used to condition the generation of x g .

As each tier doubles the resolution, the features extracted from x <g have the same time and frequency shape as x g , allowing the conditioning mechanism described in section 4.3 to be used straightforwardly.

To sample from the multiscale model we iteratively sample a value for x g conditioned on x <g using the learned distributions defined by the estimated network parametersψ = (ψ 1 , . . .

,ψ G ).

The initial tier, x 1 , is generated unconditionally by sampling from p(x 1 ;ψ 1 ) and subsequent tiers are sampled from p(x g | x <g ;ψ g ).

At each tier, the sampled x g is interleaved with the context x <g :

The interleave function is simply the inverse of the split function.

Sampling terminates once a full spectrogram, x <G+1 , has been generated.

A spectrogram generated by a multiscale model is shown in Figure 4 and the sampling procedure is visualized schematically in Figure 5 .

To demonstrate the MelNet is broadly applicable as a generative model for audio, we train the model on a diverse set of audio generation tasks (single-speaker speech generation, multi-speaker speech generation, and music generation) using three publicly available datasets.

Generated audio samples for each task are available on the accompanying web page https://audio-samples.github.io.

We include samples generated using the priming and biasing procedures described by Graves (2013) .

Biasing lowers the temperature of the predictive distribution and priming seeds the model state with a given sequence of audio prior to sampling.

Hyperparameters for all experiments are available in Appendix A.

Speech and music have rich hierarchies of latent structure.

Speech has complex linguistic structure (phonemes, words, syntax, semantics, etc.) and music has highly compositional musical structure (notes, chords, melody and rhythm, etc.).

The presence of these latent structures in generated samples can be used as a proxy for how well a generative model has learned dependencies at various timescales.

As such, a qualitative analysis of unconditional samples is an insightful method of evaluating generative models of audio.

To facilitate such a qualitative evaluation, we train MelNet on each of the three unconditional generation tasks and include samples on the accompanying web page.

For completeness, we briefly provide some of our own qualitative observations regarding the generated samples (Sections 6.1, 6.2, and 6.3).

In addition to qualitative analysis, we conduct a human evaluation experiment to quantitatively compare how well WaveNet and MelNet capture high-level structure (Section 6.4).

Lastly, we ablate the impact of the multiscale generation procedure on MelNet's ability model long-range dependencies (Section 6.5).

To test MelNet's ability to model a single speaker in a controlled environment, we utilize the Blizzard 2013 dataset (King, 2011) , which consists of audiobook narration performed in a highly animated manner by a professional speaker.

We find that MelNet frequently generates samples that contain coherent words and phrases.

Even when the model generates incoherent speech, the intonation, prosody, and speaking style remain consistent throughout the duration of the sample.

Furthermore, the model learns to produce speech using a variety of character voices and learns to generate samples which contain elements of narration and dialogue.

Biased samples tend to contain longer strings of comprehensible words but are read in a less expressive fashion.

When primed with a real sequence of audio, MelNet is able to continue sampling speech which has consistent speaking style and intonation.

Audiobook data is recorded in a highly controlled environment.

To demonstrate MelNet's capacity to model distributions with significantly more variation, we utilize the VoxCeleb2 dataset (Chung et al., 2018) .

The VoxCeleb2 dataset consists of over 2,000 hours of speech data captured with real world noise including laughter, cross-talk, channel effects, music and other sounds.

The dataset is also multilingual, with speech from speakers of 145 different nationalities, covering a wide range of accents, ages, ethnicities and languages.

When trained on the VoxCeleb2 dataset, we find that MelNet is able to generate unconditional samples with significant variation in both speaker characteristics (accent, language, prosody, speaking style) as well as acoustic conditions (background noise and recording quality).

While the generated speech is often not comprehensible, samples can often be identified as belonging to a specific language, indicating that the model has learned distinct modalities for different languages.

Furthermore, it is difficult to distinguish real and fake samples which are spoken in foreign languages.

For foreign languages, semantic structures are not understood by the listener and cannot be used to discriminate between real and fake.

Consequently, the listener must rely largely on phonetic structure, which MelNet is able to realistically model.

To show that MelNet can model audio modalities other than speech, we apply the model to the task of unconditional music generation.

We utilize the MAESTRO dataset , which consists of over 172 hours of solo piano performances.

The samples demonstrate that MelNet learns musical structures such as melody and harmony.

Furthermore, generated samples often maintain consistent tempo and contain interesting variation in volume, timbre, and rhythm.

Making quantitative comparisons with existing generative models such as WaveNet is difficult for various reasons and previous works have ultimately relied on largely empirical evaluations by the reader .

To allow the reader to make these judgements for themselves, we provide samples from both WaveNet and MelNet for each of the tasks described in the previous sections.

Furthermore, in an effort to provide quantitative metrics to support the claim that MelNet generates samples with improved long-range structure in comparison to WaveNet, we conduct a human experiment whereby participants are presented anonymized samples from both models and asked to select which sample exhibits longer-term structure.

We resort to such evaluations since standard metrics for evaluation of generative models such as density estimates cannot be used to compare WaveNet and MelNet as that these models operate on different representations.

The methodology for this experiment is as follows.

For each of the three unconditional audio generation tasks, we generated 50 samples from WaveNet and 50 samples from MelNet.

Participants were shown an anonymized, randomly-drawn sample from each model and instructed to "select the sample which has more coherent long-term structure."

We collected 50 evaluations for each task.

Results, shown in Table 1a , show that evaluators overwhelmingly agreed that samples generated by MelNet had more coherent long-range structure than samples from WaveNet across all tasks.

In addition to comparing MelNet to an unconditional WaveNet model for music generation, we also compare to a two-stage Wave2Midi2Wave model which conditions WaveNet on MIDI generated by a separately-trained Music Transformer .

The two-stage Wave2Midi2Wave model has the advantage of directly modelling labelled musical notes which distill much of the salient, high-level structure in music into a compact symbolic representation.

Despite this, as shown by the results in Table 1b , the two-stage model does not capture long-range structure as well as a MelNet model that is trained without access to any intermediate representations.

To isolate the impact of multiscale modelling procedure described in Section 5, we train models with varying numbers of tiers and evaluate the long-term coherence of their respective samples.

As noted before, long-term coherence is difficult to quantify and we provide samples on the accompanying web page so that the reader can make their own judgements.

We believe the samples clearly demonstrate that increasing the number of tiers results in samples with more coherent high-level structure.

We note that our experiment varies the number of tiers from two to five.

Training a single-tier model on full-resolution spectrograms was prohibitively expensive in terms of memory consumption.

This highlights another benefit of multiscale modelling-large, deep networks can be allocated to learning complex distributional structure in the initial tiers while shallower networks can be used for modelling the relatively simple, low-entropy distributions in the upsampling tiers.

This allows multiscale models to effectively allocate network capacity in proportion to the complexity of the modelling task.

The predominant line of research regarding generative models for audio has been directed towards modelling time-domain waveforms with autoregressive models (van den Oord et al., 2016a; Mehri et al., 2016; .

WaveNet is a competitive baseline for audio generation, and as such, is used for comparison in many of our experiments.

However, we note that the contribution of our work is in many ways complementary to that of WaveNet.

MelNet is more proficient at capturing high-level structure, whereas WaveNet is capable of producing higher-fidelity audio.

Several works have demonstrated that time-domain models can be used to invert spectral representations to highfidelity audio (Shen et al., 2018; Prenger et al., 2019; Arık et al., 2019) , suggesting that MelNet could be used in concert with time-domain models such as WaveNet.

and capture long-range dependencies in waveforms by utilizing a hierarchy of autoencoders.

This approach requires multiple stages of models which must be trained sequentially, whereas the multiscale approach in this work can be parallelized over tiers.

Additionally, these approaches do not directly optimize the data likelihood, nor do they admit tractable marginalization over the latent codes.

We also note that the modelling techniques devised in these works can be broadly applied to autoregressive models such as ours, making their contributions largely complementary to ours.

Recent works have used generative adversarial networks (GANs) (Goodfellow et al., 2014) to model both waveforms and spectral representations .

As with image generation, it remains unclear whether GANs capture all modes of the data distribution.

Furthermore, these approaches are restricted to generating fixed-duration segments of audio, which precludes their usage in many audio generation tasks.

Generating spectral representations is common practice for end-to-end text-to-speech models (Ping et al., 2017; Sotelo et al., 2017; Taigman et al., 2018) .

However, these models use probabilistic models which are much less expressive than the fine-grained autoregressive model used by MelNet.

Consequently, these models are unsuitable for modelling high-entropy, multimodal distributions such as those involved in tasks like unconditional music generation.

The network architecture used for MelNet is heavily influenced by recent advancements in deep autoregressive models for images.

Theis & Bethge (2015) introduced an LSTM architecture for autoregressive modelling of 2D images and van den Oord et al. (2016c) introduced PixelRNN and PixelCNN and scaled up the models to handle the modelling of natural images.

Subsequent works in autoregressive image modelling have steadily improved state-of-the-art for image density estimation (van den Oord et al., 2016b; Salimans et al., 2017; Parmar et al., 2018; Child et al., 2019) .

We draw inspiration from many of these models, and ultimately design a recurrent architecture of our own which is suitable for modelling spectrograms rather than images.

We note that our choice of architecture is not a fundamental contribution of this work.

While we have designed the architecture particularly for modelling spectrograms, we did not experimentally validate whether it outperforms existing architectures and make no such claims to this effect.

We use a multidimensional recurrence in both the time-delayed stack and the upsampling tiers to extract features from two-dimensional inputs.

Our multidimensional recurrence is effectively 'factorized' as it independently applies one-dimensional RNNs across each dimension.

This approach differs from the tightly coupled multidimensional recurrences used by MDRNNs (Graves et al., 2007; Graves & Schmidhuber, 2009 ) and GridLSTMs (Kalchbrenner et al., 2015) and more closely resembles the approach taken by ReNet (Visin et al., 2015) .

Our approach allows for efficient training as we can extract features from an M × N grid in max(M, N ) sequential recurrent steps rather than the M + N sequential steps required for tightly coupled recurrences.

Additionally, our approach enables the use of highly optimized one-dimensional RNN implementations.

Various approaches to image generation have succeeded in generating high-resolution, globally coherent images with hundreds of thousands of dimensions (Karras et al., 2017; Reed et al., 2017; Kingma & Dhariwal, 2018) .

The methods introduced in these works are not directly transferable to waveform generation, as they exploit spatial properties of images which are absent in one-dimensional audio signals.

However, these methods are more straightforwardly applicable to two-dimensional representations such as spectrograms.

Of particular relevance to our work are approaches which combine autoregressive models with multiscale modelling (van den Oord et al., 2016c; Dahl et al., 2017; Reed et al., 2017; Menick & Kalchbrenner, 2018) .

Our work demonstrates that the benefits of a multiscale autoregressive model extend beyond the task of image generation, and can be used to generate high-resolution, globally coherent spectrograms.

We have introduced MelNet, a generative model for spectral representations of audio.

MelNet combines a highly expressive autoregressive model with a multiscale modelling scheme to generate high-resolution spectrograms with realistic structure on both local and global scales.

In comparison to previous works which model time-domain signals directly, MelNet is particularly well-suited to model long-range temporal dependencies.

Experiments show promising results across a diverse set of audio generation tasks.

Furthermore, we believe MelNet provides a foundation for various directions of future work.

Two particularly promising directions are text-to-speech synthesis and representation learning:

• Text-to-Speech Synthesis: MelNet utilizes a more flexible probabilistic model than existing end-to-end text-to-speech models, making it well-suited to model expressive, multi-modal speech data.

• Representation Learning: MelNet is able to uncover salient structure from large quantities of unlabelled audio.

Large-scale, pre-trained autoregressive models for language modelling have demonstrated significant benefits when fine-tuned for downstream tasks.

Likewise, representations learned by MelNet could potentially aid downstream tasks such as speech recognition.

<|TLDR|>

@highlight

We introduce an autoregressive generative model for spectrograms and demonstrate applications to speech and music generation