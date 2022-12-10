We study the use of the Wave-U-Net architecture for speech enhancement, a model introduced by Stoller et al for the separation of music vocals and accompaniment.

This end-to-end learning method for audio source separation operates directly in the time domain, permitting the integrated modelling of phase information and being able to take large  temporal contexts into account.

Our experiments show that the proposed method improves several metrics, namely PESQ, CSIG, CBAK, COVL and SSNR, over the  state-of-the-art with respect to the speech enhancement task on the Voice Bank corpus (VCTK) dataset.

We find that a reduced number of hidden layers is sufficient for speech enhancement in comparison to the original system designed for singing voice separation in music.

We see this initial result as an encouraging signal to further explore speech enhancement in the time-domain, both as an end in itself and as a pre-processing step to speech recognition systems.

The remainder of this paper is structured as follows.

In section 2, we briefly review related work 27 from the literature.

In section 3, we introduce briefly the Wave-U-Net architecture and its application architectures like that of BID0 .

Recently, the U-Net architecture on magnitude spectrograms has for each prediction and is based on the repeated application of dilated convolutions with exponentially increasing dilation factors to factor in context information.

yield an estimate of the target sources, a tanh nonlinearity follows, succeeded by a final LeakyReLU.

In applying the Wave-U-Net architecture to the application of speech enhancement, our objective 58 is to separate a mixture waveform DISPLAYFORM0

To evaluate and compare the quality of the enhanced speech yielded by the Wave-U-Net, we mirror filter sizes, to the task and expanding to multi-channel audio and multi-source-separation.

<|TLDR|>

@highlight

The Wave-U-Net architecture, recently introduced by Stoller et al for music source separation, is highly effective for speech enhancement, beating the state of the art.