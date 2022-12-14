We present a hybrid framework that leverages the trade-off between temporal and frequency precision in audio representations to improve the performance of speech enhancement task.

We first show that conventional approaches using specific representations such as raw-audio and spectrograms are each effective at targeting different types of noise.

By integrating both approaches, our model can learn multi-scale and multi-domain features, effectively removing noise existing on different regions on the time-frequency space in a complementary way.

Experimental results show that the proposed hybrid model yields better performance and robustness than using each model individually.

We first describe the objective function and the selected modules that have been reported to show 36 competitive performance using either raw-audio [5] or spectrogram input [3] .

Selected models are 37 each used later as components of our proposed hybrid model.

We employ the energy-conserving loss function proposed in [7] which simultaneously considers 40 speech and noise signals.

Let the noisy input x consist of clean speech s and noise n. The estimated 41 speech by the model is referred to asŝ.

Then, our objective function is defined as follows:

L(x, s, n,ŝ) = s −ŝ 1 + n −n 1 ,wheren = x −ŝ represents the estimated noise signal and · 1 denotes 1 norm.

We construct the time domain network based on TasNet [5] which employs one-dimensional dilated 45 convolution to handle long time sequences of raw-audio.

TasNet has shown competitive sample quality 46 for speech source separation, which is a similar task to speech enhancement.

In our experiments, we used a reduced version of TasNet.

With a slight abuse of notation, we refer to the network as noise from the time-frequency space.

We hybridize both time and T-F domain networks in a cascaded way (Fig. 1 Figure 1 : A schematic illustration of the hybrid system (MDPhD).

Note that the network of the same domain (same color) shares the parameters.

For the time-frequency (T-F) domain network, we convert the time-domain input to a spectrogram using the short time Fourier transform (STFT), whose output is converted back to a waveform using the inverse short time Fourier transform (iSTFT).The final objective of the hybrid model with auxiliary loss becomes DISPLAYFORM0 where θ denotes the network parameter.

that of U-Net with doubled parameter size ( TAB3 ).

Note that, however, TasNet fails to remove high 88 frequency noise, which is supposedly hard to capture in the time domain FIG2 .

the U → D model shares the weakness of U-Net and vice versa.

We conjecture that this happens

Using the test dataset, we compared our results to recent studies of speech enhancement field.

Our model showed the best performance quantitatively and qualitatively among the others under TAB4 : Comparison with other methods.

The predicted rating of speech distortion (CSIG), background distortion (CBAK) and overall quality (COVL) are reported (from 1 to 5, higher is better).

PESQ (from -0.5 to 4.5, higher is better) stands for perceptual evaluation of speech quality and SSNR (higher is better) is segmental SNR.

The best result for each measure is given in bold style.

Table 3 : SNR evaluation of models with various objective functions.

D and U denote the TasNet (reduced) using one-dimensional dilated convolution and U-Net, respectively.

The type of objective functions are noted next to the model name.

1 represents our baseline objective function.

2 represents an objective function that substitutes the 1 term of equation FORMULA0 In this section, we present the detailed configuration of the models we used.

In the following figures,

Figure 3: U-Net (1.5M) architecture.

2D Conv means a two-dimensional convolution block consisting of a two-dimensional convolution operation with filter size F (height, width), stride size S (height, width) and output channel size C followed by batch renormalization and leaky-RELU activation function.

2D t-Conv means a two-dimensional transposed convolution block.

Our baseline models used in experiments process the log-magnitude of the input spectrogram.

<|TLDR|>

@highlight

A hybrid model utilizing both raw-audio and spectrogram information for speech enhancement tasks.