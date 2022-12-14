We investigate low-bit quantization to reduce computational cost of deep neural network (DNN) based keyword spotting (KWS).

We propose approaches to further reduce quantization bits via integrating quantization into keyword spotting model training, which we refer to as quantization-aware training.

Our experimental results on large dataset indicate that quantization-aware training can recover performance models quantized to lower bits representations.

By combining quantization-aware training and weight matrix factorization, we are able to significantly reduce model size and computation for small-footprint keyword spotting, while maintaining performance.

by tuning a threshold.

Context information is incorporated by stacking frames in the input.

When 23 deployed on device such KWS are always quantized.

16 bit and 8 bit quantizations are common in 24 the industry [7] [8] [9] [10] [11] [12] [13] .

Since the keyword models in such KWS are usually trained using full-precision 25 arithmetic, quantization degrades their performance on device.

One approach to mitigate that degra-26 dation is by using quantization-aware training.

Quantization-aware training considers the quantized 27 weights in full precision representation in order to inject the quantization error into training.

This 28 method enables the weights to be optimized against quantization errors.

In this work, we use quantization-aware training to build a very small-footprint low-power KWS.

To 30 train the wake word model, we employ quantization-aware training as a final training stage.

We find 31 that 8 bit and 4 bit quantized KWS models can be trained successfully by using quantization-aware 32 training.

The paper is organized as follows: Section 2 introduces keyword spotting system and quantization-

We use dynamic quantization approach, where shifts and scales for quantizing DNN weight matrices 46 are calculated independenlty column-wise.

This is similar to "buketing" BID19 or "per-channel" [18] quantization with technical differences.

Also, the inputs are quantized row-wise on the fly during the The accuracy loss due to quantization is incorporated via quantization-aware training, FIG2 .

The keyword 'Alexa' is chosen for our experiments.

We use an in-house 500 hrs far-field corpus of 69 diverse far-field speech data and a similar composition 100 hrs dataset for evaluation.

We evaluate all 70 models using end-to-end Detection Error Tradeoff (DET) curves, which describe the models' miss 71 rate vs. false accept rate (FAR), as well as DET area under curve (AUC).

For training, we use GPU-72 based distributed DNN training method described in [20] .

The training is organized into 3 stages:

In the 1st stage a small ASR DNN with 3 hidden layers of 128 units is pre-trained from random 74 initialization and using full ASR phone-targets obtained from a large, production ASR system.

In another 20 epochs, using the same exponential learning rate decay schedule.

The performance of the 'naively quantized' models is shown in TAB0 .

We observe that 16 bit Table 2 : AUC improvement of quantized models' performance using quantization-aware training.

Figure 2 : DET for full-precision, quantized, and quantization-aware trained 50k model.

The DET curves for 16 bit and 8 bit quantized-models are not shown due to them not being significantly different from the full-precision model.

@highlight

We investigate quantization-aware training in very low-bit quantized keyword spotters to reduce the cost of on-device keyword spotting.

@highlight

This submission proposes a combination of low-rank decomposition and quanitization approach to compress DNN models for keyword spotting.