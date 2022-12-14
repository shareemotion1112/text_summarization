Most deep learning-based models for speech enhancement have mainly focused on estimating the magnitude of spectrogram while reusing the phase from noisy speech for reconstruction.

This is due to the difficulty of estimating the phase of clean speech.

To improve speech enhancement performance, we tackle the phase estimation problem in three ways.

First, we propose Deep Complex U-Net, an advanced U-Net structured model incorporating well-defined complex-valued building blocks to deal with complex-valued spectrograms.

Second, we propose a polar coordinate-wise complex-valued masking method to reflect the distribution of complex ideal ratio masks.

Third, we define a novel loss function, weighted source-to-distortion ratio (wSDR) loss, which is designed to directly correlate with a quantitative evaluation measure.

Our model was evaluated on a mixture of the Voice Bank corpus and DEMAND database, which has been widely used by many deep learning models for speech enhancement.

Ablation experiments were conducted on the mixed dataset showing that all three proposed approaches are empirically valid.

Experimental results show that the proposed method achieves state-of-the-art performance in all metrics, outperforming previous approaches by a large margin.

Speech enhancement is one of the most important and challenging tasks in speech applications where the goal is to separate clean speech from noise when noisy speech is given as an input.

As a fundamental component for speech-related systems, the applications of speech enhancement vary from speech recognition front-end modules to hearing aid systems for the hearing-impaired BID36 BID32 .Due to recent advances in deep learning, the speech enhancement task has been able to reach high levels in performance through significant improvements.

When using audio signals with deep learning models, it has been a common practice to transform a time-domain waveform to a time-frequency (TF) representation (i.e. spectrograms) via short-time-Fourier-transform (STFT).

Spectrograms are represented as complex matrices, which are normally decomposed into magnitude and phase components to be used in real-valued networks.

In tasks involving audio signal reconstruction, such as speech enhancement, it is ideal to perform correct estimation of both components.

Unfortunately, complex-valued phase has been often neglected due to the difficulty of its estimation.

This has led to the situation where most approaches focus only on the estimation of a magnitude spectrogram while reusing noisy phase information BID9 BID39 BID7 BID15 BID26 .

However, reusing phase from noisy speech has clear limitations, particularly under extremely noisy conditions, in other words, when signal-to-noise ratio (SNR) is low.

This can be easily verified by simply using the magnitude spectrogram of clean speech with the phase spectrogram of noisy speech to reconstruct clean speech, as illustrated in Fig A popular approach to speech enhancement is to optimize a mask which produces a spectrogram of clean speech when applied to noisy input audio.

One of the first mask-based attempts to perform the task by incorporating phase information was the proposal of the phase-sensitive mask (PSM) .

Since the performance of PSM was limited because of reusing noisy phase, later studies proposed using complex-valued ratio mask (cRM) to directly optimize on complex values BID37 BID3 .

We found this direction promising for phase estimation because it has been shown that a complex ideal ratio mask (cIRM) is guaranteed to give the best oracle performance out of other ideal masks such as ideal binary masks, ideal ratio masks, or PSMs .

Moreover, this approach jointly estimates magnitude and phase, removing the need of separate models.

To estimate a complex-valued mask, a natural desire would be to use an architecture which can handle complex-domain operations.

Recent work gives a solution to this by providing deep learning building blocks adapted to complex arithmetic BID28 .In this paper, we build upon previous studies to design a new complex-valued masking framework, based on a proposed variant of U-Net BID19 , named Deep Complex U-Net (DCUnet).

In our proposed framework, DCUnet is trained to estimate a complex ratio mask represented in polar coordinates with prior knowledge observable from ideal complex-valued masks.

With the complex-valued estimation of clean speech, we can use inverse short-time-Fourier-transform (ISTFT) to convert a spectrogram into a time-domain waveform.

Taking this as an advantage, we introduce a novel loss function which directly optimizes source-to-distortion ratio (SDR) BID31 , a quantitative evaluation measure widely used in many source separation tasks.

Our contributions can be summarized as follows:1.

We propose a new neural architecture, Deep Complex U-Net, which combines the advantages of both deep complex networks and U-Net, yielding state-of-the-art performance.2.

While pointing out limitations of current masking strategies, we design a new complexvalued masking method based on polar coordinates.3.

We propose a new loss function weighted-SDR loss, which directly optimizes a well known quantitative evaluation measure.

Phase estimation for audio signal reconstruction has been a recent major interest within the audio source separation community because of its importance and difficulty.

While iterative methods such as the Griffin-Lim algorithm and its variants BID8 BID17 aimed to address this problem, neural network-based approaches are recently attracting attention as noniterative alternatives.

One major approach is to use an end-to-end model that takes audio as raw waveform inputs without using any explicit time-frequency (TF) representation computed via STFT BID16 BID18 BID23 BID5 .

Since raw waveforms inherently contain phase information, it is expected to achieve phase estimation naturally.

Another method is to estimate magnitude and phase using two separate neural network modules which serially estimate magnitude and phase BID0 BID25 .

In this framework, the phase estimation module uses noisy phase with predicted magnitude to estimate phase of clean speech.

There is also a recent study which proposed to use additional layers with trainable discrete values for phase estimation .A more straightforward method would be to jointly estimate magnitude and phase by using a continuous complex-valued ratio mask (cRM).

Previous studies tried this joint estimation approach bounding the range of the cRM BID37 BID3 .

Despite the advantages of the cRM approach, previously proposed methods had limitations with regard to the loss function and the range of the mask which we will be returning with more details in Section 3 along with our proposed methods to alleviate these issues.

As a natural extension to the works above, some studies have also undergone to examine whether complex-valued networks are useful when dealing with intrinsically complex-valued data.

In the series of two works, complex-valued networks were shown to help singing voice separation performance with both fully connected neural networks and recurrent neural networks BID12 b) .

However, the approaches were limited as it ended up only switching the real-valued network into a complex-valued counterpart and leaving the other deep learning building blocks such as weight initialization and normalization technique in a realvalued manner.

Also, the works do not show whether the phase was actually well estimated either quantitatively or qualitatively, only ending up showing that there was a performance gain.

In this section we will provide details on our approach, starting with our proposed model Deep Complex U-Net, followed by the masking framework based on the model.

Finally, we will introduce a new loss function to optimize our model, which takes a critical role for proper phase estimation.

Before getting into details, here are some notations used throughout the paper.

The input mixture signal x(n) = y(n) + z(n) ??? R is assumed to be a linear sum of the clean speech signal y(n) ??? R and noise z(n) ??? R, where estimated speech is denoted as??(n) ??? R. Each of the corresponding time-frequency (t, f ) representations computed by STFT is denoted as DISPLAYFORM0 The ground truth mask cIRM is denoted as M t,f ??? C and the estimated cRM is denoted asM t,f ??? C, where The U-Net structure is a well known architecture composed as a convolutional autoencoder with skip-connections, originally proposed for medical imaging in computer vision community BID19 .

Furthermore, the use of real-valued U-Net has been shown to be also effective in many recent audio source separation tasks such as music source separation BID10 BID23 BID26 , and speech enhancement BID16 .

Deep Complex U-Net (DCUnet) is an extended U-Net, refined specifically to explicitly handle complex domain operations.

In this section, we will describe how U-Net is modified using the complex building blocks originally proposed by BID28 .

DISPLAYFORM1

Complex-valued Building Blocks.

Given a complex-valued convolutional filter W = A + iB with real-valued matrices A and B, the complex convolution operation on complex vector h = x + iy with W is done by W * h = (A * x ??? B * y) + i(B * x + A * y).

In practice, complex convolutions can be implemented as two different real-valued convolution operations with shared real-valued convolution filters.

Details are illustrated in Appendix A. Activation functions like ReLU were also adapted to the complex domain.

In previous work, CReLU, an activation function which applies ReLU on both real and imaginary values, was shown to produce the best results out of many suggestions.

Details on batch normalization and weight initialization for complex networks can be found in BID28 .Modifying U-Net.

The proposed Deep Complex U-Net is a refined U-Net architecture applied in STFT-domain.

Modifications done to the original U-Net are as follows.

Convolutional layers of UNet are all replaced to complex convolutional layers, initialized to meet the Glorot's criteria BID6 .

Here, the convolution kernels are set to be independent to each other by initializing the weight tensors as unitary matrices for better generalization and fast learning BID2 .

Complex batch normalization is implemented on every convolutional layer except the last layer of the network.

In the encoding stage, max pooling operations are replaced with strided complex convolutional layers to prevent spatial information loss.

In the decoding stage, strided complex deconvolutional operations are used to restore the size of input.

For the activation function, we modified the previously suggested CReLU into leaky CReLU, where we simply replace ReLU into leaky ReLU BID14 , making training more stable.

Note that all experiments performed in Section 4 are done with these modifications.

As our proposed model can handle complex values, we aim to estimate cRM for speech enhancement.

Although it is possible to directly estimate the spectrogram of a clean source signal, it has been shown that better performance can be achieved by applying a weighting mask to the mixture spectrogram BID33 .

One thing to note is that real-valued ratio masks (RM) only change the scale of the magnitude without changing phase, resulting in irreducible errors as illustrated in Appendix D. On the other hand, cRM also perform a rotation on the polar coordinates, allowing to correct phase errors.

In other words, the estimated speech spectrogram?? t,f is computed by multiplying the estimated maskM t,f on the input spectrogram X t,f as follows:Published as a conference paper at ICLR 2019 DISPLAYFORM0 In this state, the real and imaginary values of the estimated cRM is unbounded.

Although estimating an unbounded mask makes the problem well-posed (see Appendix D for more information), we can imagine the difficulty of optimizing from an infinite search space compared to a bounded one.

Therefore, a few techniques have been tried to bound the range of cRM.

For example, Williamson et al. tried to directly optimize a complex mask into a cIRM compressed to a heuristic bound BID37 .

However, this method was limited since it was only able to succeed in training the model by computing the error between cIRM and the predicted cRM which often leads to a degradation of performance BID33 BID41 .

More recently, Ephrat et al.proposed a rectangular coordinate-wise cRM made with sigmoid compressions onto each of the real and imaginary parts of the output of the model BID3 .

After then MSE between clean source Y and estimated source?? was computed in STFT-domain to train the model.

However, the proposed masking method has two main problems regarding phase estimation.

First, it suffers from the inherent problem of not being able to reflect the distribution of cIRM as shown in FIG2 and Appendix E. Second, this approach results in a cRM with a restricted rotation range of 0 ??? to 90??? (only clock-wise), which makes it hard to correct noisy phase.

To alleviate these problems, we propose a polar coordinate-wise cRM method that imposes nonlinearity only on the magnitude part.

More specifically, we use a hyperbolic tangent non-linearity to bound the range of magnitude part of the cRM be [0, 1) which makes the mask bounded in an unit-circle in complex space.

The corresponding phase mask is naturally obtained by dividing the output of the model with the magnitude of it.

More formally, let g(??) be our neural network and the output of it be O t,f = g(X t,f ).

The proposed complex-valued maskM t,f is estimated as follows: DISPLAYFORM1 A summarized illustration of cRM methods is depicted in FIG2 .

A popular loss function for audio source separation is mean squared error (MSE) between clean source Y and estimated source?? on the STFT-domain.

However, it has been reported that optimizing the model with MSE in complex STFT-domain fails in phase estimation due to the randomness in phase structure BID37 .

As an alternative, it is possible to use a loss function defined in the time-domain instead, as raw waveforms contain inherent phase information.

While MSE on waveforms can be an easy solution, we can expect it to be more effective if the loss function is directly correlated with well-known evaluation measures defined in the time-domain.

Here, we propose an improved loss function weighted-SDR loss by building upon a previous work which attempts to optimize a standard quality measure, source-to-distortion ratio (SDR) BID30 .

The original loss function loss V en suggested by Venkataramani et al. is formulated upon the observation from Equation 4, where y is the clean source signal and?? is the estimated source signal.

In practice, the negative reciprocal is optimized as in Equation FORMULA4 .

DISPLAYFORM0 Although using Equation 5 works as a loss function, there are a few critical flaws in the design.

First, the lower bound becomes ??? y 2 , which depends on the value of y causing fluctuation in the loss values when training.

Second, when the target y is empty (i.e., y = 0) the loss becomes zero, preventing the model to learn from noisy-only data due to zero gradients.

Finally, the loss function is not scale sensitive, meaning that the loss value is the same for?? and c??, where c ??? R.To resolve these issues, we redesigned the loss function by giving several modifications to Equation 5.

First, we made the lower bound of the loss function independent to the source y by restoring back the term y 2 and applying square root as in Equation FORMULA5 .

This makes the loss function bounded within the range [-1, 1] and also be more phase sensitive, as inverted phase gets penalized as well.

DISPLAYFORM1 Expecting to be complementary to source prediction and to propagate errors for noise-only samples, we also added a noise prediction term loss SDR (z,???).

To properly balance the contributions of each loss term and solve the scale insensitivity problem, we weighted each term proportional to the energy of each signal.

The final form of the suggested weighted-SDR loss is as follows: DISPLAYFORM2 where,??? = x ????? is estimated noise and ?? = ||y|| 2 /(||y|| 2 + ||z|| 2 ) is the energy ratio between clean speech y and noise z. Note that although weighted SDR loss is a time-domain loss function, it can be backpropagated through our framework.

Specifically, STFT and ISTFT operations are implemented as 1-D convolution and deconvolution layers consisting of fixed filters initialized with the discrete Fourier transform matrix.

The detailed properties of the proposed loss function are in Appendix C.

Dataset.

For all experiments, we used the same experimental setups as previous works in order to perform direct performance comparison BID16 BID18 BID22 BID5 .

Noise and clean speech recordings were provided from the Diverse Environments Multichannel Acoustic Noise Database (DEMAND) BID27 and the Voice Bank corpus BID29 , respectively, each recorded with sampling rate of 48kHz.

Mixed audio inputs used for training were composed by mixing the two datasets with four signalto-noise ratio (SNR) settings (15, 10, 5, and 0 (dB)), using 10 types of noise (2 synthetic + 8 from DEMAND) and 28 speakers from the Voice Bank corpus, creating 40 conditional patterns for each speech sample.

The test set inputs were made with four SNR settings different from the training set (17.5, 12.5, 7.5, and 2.5 (dB)), using the remaining 5 noise types from DEMAND and 2 speakers from the Voice Bank corpus.

Note that the speaker and noise classes were uniquely selected for the training and test sets.

Pre-processing.

The original raw waveforms were first downsampled from 48kHz to 16kHz.

For the actual model input, complex-valued spectrograms were obtained from the downsampled waveforms via STFT with a 64ms sized Hann window and 16ms hop length.

Implementation.

All experiments were implemented and fine-tuned with NAVER Smart Machine Learning (NSML) platform BID24 BID11 .

In this subsection, we compare overall speech enhancement performance of our method with previously proposed algorithms.

As a baseline approach, Wiener filtering (Wiener) with a priori noise SNR estimation was used, along with recent deep-learning based models which are briefly described as the following: SEGAN: a time-domain U-Net model optimized with generative adversarial networks.

Wavenet: a time-domain non-causal dilated wavenet-based network.

MMSE-GAN: a timefrequency masking-based method with modified adversarial training method.

Deep Feature Loss: a time-domain dilated convolution network trained with feature loss from a classifier network.

BID21 3.23 2.68 2.67 2.22 5.07 SEGAN BID16 3 For comparison, we used the configuration of using a 20-layer Deep Complex U-Net (DCUnet-20) to estimate a tanh bounded cRM, optimized with weighted-SDR loss.

As a showcase for the potential of our approach, we also show results from a larger DCUnet-20 (Large-DCUnet-20) which has more channels in each layer.

Both architectures are specified in detail in Appendix B. Results show that our proposed method outperforms the previous state-of-the-art methods with respect to all metrics by a large margin.

Additionally, we can also see that larger models yield better performance.

We see the reason to this significant improvement coming from the phase estimation quality of our method, which we plan to investigate in later sections.

TAB2 shows the jointly combined results on varied masking strategies and loss functions, where three models (DCU-10 (1.4M), DCU-16 (2.3M), and DCU-20 (3.5M)) are investigated to see how architectural differences in the model affect quantitative results.

In terms of masking strategy, the proposed BDT mask mostly yields better results than UBD mask in DCU-10 and DCU-16, implying the importance of limiting optimization space with prior knowledge.

However, in the case of DCU-20, UBD mask was able to frequently surpass the performance of BDT mask.

Intuitively, this indicates that when the number of parameter gets large enough, the model is able to fit the distribution of data well even when the optimization space is not bounded.

In terms of the loss function, almost every result shows that optimizing with wSDR loss gives the best result.

However, we found out that Spc loss often provides better PESQ results than wSDR loss for DCU-10 and DCU-16 except DCU-20 case where Spc and wSDR gave similar PESQ results.

Validation on complex-valued network and mask.

In order to show that complex neural networks are effective, we compare evaluation results of DCUnet (Cn) and its corresponding real-valued UNet setting with the same parameter size (Rn).

For the real-valued network, we tested two settings cRMRn and RMRn to show the effectiveness of phase estimation.

The first setting takes a complexvalued spectrogram as an input, estimating a complex ratio mask (cRM) with a tanh bound.

The second setting takes a magnitude spectrogram as an input, estimating a magnitude ratio mask (RM) with a sigmoid bound.

All models were trained with weighted-SDR loss, where the ground truth phase was given while training RMRn.

Additionally, all models were trained on different number of parameters (20-layer (3.5M), 16-layer (2.3M), and 10-layer (1.4M)) to show that the results are consistent regardless of model capacity.

Detailed network architectures for each model are illustrated in Appendix B.

In TAB3 , evaluation results show that our approach cRMCn makes better results than conventional method RMRn for all cases, showing the effectiveness of phase correction.

Also, cRMCn gives better results than cRMRn, which indicates that using complex-valued networks consistently improve the performance of the network.

Note that these results are consistent through every evaluation measure and model size.

We performed qualitative evaluations by obtaining preference scores between the proposed DCUnet (Large-DCUnet-20) and baseline methods.

15 utterance samples with different noise levels were selected from the test set and used for subjective listening tests.

For each noisy sample, all possible six pairs of denoised audio samples from four different algorithms were presented to the participants in a random order, resulting in 90 pairwise comparisons to be made by each subject.

For each comparison, participants were presented with three audio samples -original noisy speech and two denoised speech samples by two randomly selected algorithms -and instructed to choose either a preferred sample (score 1) or "can't decide" (score 0.5).

A total of 30 subjects participated in the listening test, and the results are presented in TAB4 and in Table 7 . : Scatter plots of estimated cRMs with 9 different mask and loss function configurations for a randomly picked noisy speech signal.

Each scatter plot shows the distribution of complex values from an estimated cRM.

The leftmost plot is from the cIRM for the given input.

We can observe that most real-values are distributed around 0 and 1, while being relatively sparse in between.

The configuration that fits the most to this distribution pattern is observed in the red dotted box which is achieved by the combination of our proposed methods (Bounded (tanh) and weighted-SDR).

TAB4 shows that DCUnet clearly outperforms the other methods in terms of preference scores in every SNR condition.

These differences are statistically significant as confirmed by pairwise one-tailed t-tests.

Furthermore, the difference becomes more obvious as the input SNR condition gets worse, which supports our motivation that accurate phase estimation is even more important under harsh noisy conditions.

This is further confirmed by in-depth quantitative analysis of the phase distance as described in Section 5 and TAB6 .

In this section, we aim to provide constructive insights on phase estimation by analyzing how and why our proposed method is effective.

We first visualized estimated complex masks with scatter plots in FIG3 for each masking method and loss function configuration from scaling the magnitude of noisy speech and fails to correct the phase of noisy speech with rotations (e.g., (X DISPLAYFORM0 .

In order to demonstrate this effect in an alternate perspective, we also plotted estimated waveforms for each loss function in FIG4 .

As one can notice from FIG4 To explicitly support these observations, we would need a quantitative measure for phase estimation.

Here, we define the phase distance between target spectrogram (A) and estimated spectrogram (B) as the weighted average of angle between corresponding complex TF bins, where each bin is weighted by the magnitude of target speech ( A t,f ) to emphasize the relative importance of each TF bin.

Phase distance is formulated as the following: DISPLAYFORM1 where, ???(A t,f , B t,f ) represents the angle between A t,f and B t,f , having a range of [0, 180] .The phase distance between clean and noisy speech (PhaseDist (C, N) ) and the phase distance between clean and estimated speech (PhaseDist(C, E)) are presented in TAB6 .

The results show that the best phase improvement (Phase Improvement = PhaseDist(C, N) ??? PhaseDist(C, E)) is obtained with wSDR loss under every SNR condition.

Also Spc loss gives the worst results, again reinforcing our observation.

Analysis between the phase improvement and performance improvement is further discussed in Appendix G.

In this paper, we proposed Deep Complex U-Net which combines two models to deal with complexvalued spectrograms for speech enhancement.

In doing so, we designed a new complex-valued masking method optimized with a novel loss function, weighted-SDR loss.

Through ablation studies, we showed that the proposed approaches are effective for more precise phase estimation, resulting in state-of-the-art performance for speech enhancement.

Furthermore, we conducted both quantitative and qualitative studies and demonstrated that the proposed method is consistently superior to the previously proposed algorithms.

n the near future, we plan to apply our system to various separation tasks such as speaker separation or music source separation.

Another important direction is to extend the proposed model to deal with multichannel audio since accurate estimation of phase is even more critical in multichannel environments BID34 .

Apart from separation, our approach can be generalized to various audio-related tasks such as dereverberation, bandwidth extension or phase estimation networks for text-to-speech systems.

Taking advantage of sequence modeling, it may also be interesting to find further extensions with complex-valued LSTMs BID1 BID38 .

In this section, we address the difference between the real-valued convolution and the complexvalued convolution.

Given a complex-valued convolution filter W = A + iB with real-valued matrices A and B, the complex-valued convolution can be interpreted as two different real-valued convolution operations with shared parameters, as illustrated in FIG6 (b).

For a fixed number of #Channel product = #Input channel(M ) ?? #Output channel(N ), the number of parameters of the complex-valued convolution becomes double of that of a real-valued convolution.

Considering this fact, we built the pair of a real-valued network and a complex-valued network with the same number of parameters by reducing #Channel product of complex-valued convolution by half for a fair comparison.

The detail of models reflecting this configuration is explained in Appendix B.

In this section, we describe three different model architectures (DCUnet-20 (#params: 3.5M), DCUnet-16 (#params: 2.3M), and DCUnet-10 (#params: 1.4M)) each in complex-valued network setting and real-valued network setting in FIG7 , 8, 9.

Both complex-valued network (C) and realvalued network (R) have the same size of convolution filters with different number of channels to set the parameter equally.

The largest model, Large-DCUnet-20, in TAB0 is also described in FIG12 .

Every convolution operation is followed by batch normalization and an activation function as described in FIG0 .

For the complex-valued network, the complex-valued version of batch normalization and activation function was used following Deep Complex Networks BID28 .

Note that in the very last layer of every model the batch normalization and leaky ReLU activation was not used and non-linearity function for mask was applied instead.

The real-valued network configuration was not considered in the case of largest model.

In this section, we summarize the properties of the proposed weighted-SDR loss.

First, we show that the range of weighted-SDR loss is bounded and explain the conditions under which the minimum value is obtained.

Next, we explain the gradients in the case of noise-only input.

FIG0 : Description of encoder and decoder block.

F f and F t denote the convolution filter size along the frequency and time axis, respectively.

S f and S t denote the stride size of convolution filter along the frequency and time axis, respectively.

O C and O R denote the different number of channels in complex-valued network setting and real-valued network setting, respectively.

The number of channels of O R is set to be roughly ??? 2 times the number of channels of O C so that the number of trainable parameters of real-valued network and complex-valued network becomes approximately the same.

Let x denotes noisy speech with T time step, y denotes target source and?? denotes estimated source.

Then, loss wSDR (x, y,??) is defined as follows: DISPLAYFORM0 where, ?? is the energy ratio between target source and noise, i.e., y 2 /( y 2 + x ??? y 2 ).Proposition 1.

loss wSDR (x, y,??) is bounded on [-1,1] .

Moreover, for fixed y = 0 and x ??? y = 0, the minimum value -1 can only be attained when?? = y, if x = cy for ???c ??? R.Proof.

Cauchy-Schwarz inequality states that for a ??? R T and b ??? R T , ??? a b ??? < a, b > ??? a b .

By this inequality, [-1,1] becomes the range of loss wSDR .

To attain the minimum value, the equality condition of the Cauchy-Schwarz inequality must be satisfied.

This equality condition is equivalent to b = 0 or a = tb, for ???t ??? R. Applying the equality condition with the assumption (y = 0, x ??? y = 0) to Equation 9 leads to?? = t 1 y and x ????? = t 2 (x ??? y), f or ???t 1 ??? R and ???t 2 ??? R. By adding these two equations, we can get (1 ??? t 1 )x = (t 1 ??? t 2 )y. By the assumption x = cy, which is generally satisfied for large T , we can conclude t 1 = 1, t 1 = t 2 must be satisfied when the minimum value is attained.

The following property of the weighted-SDR loss shows that the network can also learn from noiseonly training data.

In experiments, we add small number to denominators of Equation 9.

Thus for the case of y = 0, Equation 9 becomes DISPLAYFORM1 Proposition 2.

When we parameterize?? = g ?? (x), the loss wSDR (x, y, g ?? (x)) has a non-zero gradient with respect to ?? even if the target source y is empty.

Proof.

We can calculate partial derivatives as follows: DISPLAYFORM2 Thus, the non-zero gradients with respect to ?? can be back-propagated.

In this section, we illustrate two possible irreducible errors.

FIG1 (a) shows the irreducible phase error due to lack of phase estimation.

FIG1 (b) shows the irreducible error induced when bounding the range of mask.

Not bounding the range of the mask makes the problem well-posed but it may suffer from the wide range of optimization search space because of the lack of prior knowledge on the distribution of cIRM.

The scatter plots of cIRM from training set is shown in FIG2 .

We show four different scatter plots according to their SNR values of mixture (0, 5, 10, and 15 (dB) ).

Each scattered point of cIRM, M t,f , is defined as follows: DISPLAYFORM0 The scattered points near origin indicate the TF bins where the value of Y t,f is significantly small compared to X t,f .

Therefore, those TF bins can be interpreted as the bins dominated with noise rather than source.

On the other hand, the scattered points near (1,0) indicates the TF bins where the value of Y t,f is almost the same as X t,f .

In this case, those TF bins can be interpreted as the bins dominated with source rather than noise.

Therefore, as SNR becomes higher, the amount of TF bins dominated with clean source becomes larger compared to the lower SNR cases, and consequently the portion of real part close to 1 becomes larger as in FIG2 .

In this section, we show a supplementary visualization of phase of estimated speech.

Although the raw phase information itself does not show a distinctive pattern, the hidden structure can be revealed with group delay, which is the negative derivative of the phase along frequency axis BID40 .

With this technique, the phase information can be explicitly shown as in FIG3 .

FIG3 shows the group delay of clean speech and the corresponding magnitude is shown in FIG3 (a).

The two representations shows that the group delay of phase has a similar structure to that of magnitude spectrogram.

The estimated phase by our model is shown in in FIG3 (c).While the group delay of noisy speech FIG3 ) does not show a distinctive harmonic pattern, our estimation show the harmonic pattern similar to the group delay of clean speech, as shown in the yellow boxes in FIG3

In this section, to show the limitation of conventional approach (without phase estimation), we emphasize that the phase estimation is important, especially under low SNR condition (harsh condition).We first make an assumption that the estimation of phase information becomes more important when the given mixture has low SNR.

Our reasoning behind this assumption is that if the SNR of a given mixture is low, the irreducible phase error is likely to be greater, hence a more room for improvement with phase estimation as illustrated in FIG4 .

This can also be verified in TAB6 columns PhaseDist(C, N) and Phase Improvement where the values of both columns increase as SNR becomes higher.

FIG4 : (a) The case where SNR of given mixture is high.

In this case the source is likely to be dominant in the mixture.

Therefore it is relatively easier to estimate ground truth source with better precision even when the phase is not estimated.

(b) The case where SNR of given mixture is low.

In this case the source is not dominant in the mixture.

Therefore, the irreducible phase error is likely to be higher in low SNR conditions than higher SNR conditions.

Under this circumstance, we assume the lack of phase estimation will result in a particularly bad system performance.

To empirically show the importance of phase estimation, we show correlation between phase improvement and performance difference between the conventional method (without phase estimation) and our proposed method (with phase estimation) in TAB8 .

The performance difference was calculated by simply subtracting the evaluation results of conventional method from the evaluation results of our method with phase estimation.

For fair comparison, both conventional method (RMRn) and proposed method (cRMCn) were set to have the same number of parameters.

Also, both models were trained with weighted-SDR loss.

The results show that when the SNR is low, both the phase improvement and the performance difference are relatively higher than the results from higher SNR conditions.

Furthermore, almost all results show an incremental increase of phase improvement and performance difference as the SNR decreases, which agrees on our assumption.

Therefore we believe that phase estimation is important especially in harsh noisy conditions (low SNR conditions).

Table 7 : Pairwise preference scores of four models including DCUnet.

The scores are obtained by calculating the relative frequency the subjects prefer one method to the other method.

Hard/Medium/Easy denote 2.5/7.5/17.5 SNR conditions in dB, respectively.

Significance for each statistic is also described (n.s.: not significant, * : p<0.05, * * : p<0.01, * * * : p<0.001).

@highlight

This paper proposes a novel complex masking method for speech enhancement along with a loss function for efficient phase estimation.