Deep neural networks with discrete latent variables offer the promise of better symbolic reasoning, and learning  abstractions that are more useful to new tasks.

There has been a surge in interest in discrete latent variable models,  however, despite several recent improvements, the training of discrete latent variable models has remained  challenging and their performance has mostly failed to match their continuous counterparts.

Recent work on vector quantized autoencoders (VQ-VAE) has made substantial progress in this direction, with its perplexity almost matching that of a VAE on datasets such as CIFAR-10.

In this work, we investigate an alternate training technique for VQ-VAE, inspired by its connection to the Expectation Maximization (EM) algorithm.

Training the discrete autoencoder with EM and combining it with sequence  level knowledge distillation alows us to develop a non-autoregressive machine translation model whose accuracy almost matches a strong greedy autoregressive baseline Transformer, while being 3.3 times faster at inference.

Unsupervised learning of meaningful representations is a fundamental problem in machine learning since obtaining labeled data can often be very expensive.

Continuous representations have largely been the workhorse of unsupervised deep learning models of images BID4 BID16 BID28 BID25 , audio BID27 , and video .

However, it is often the case that datasets are more naturally modeled as a sequence of discrete symbols rather than continuous ones.

For example, language and speech are inherently discrete in nature and images are often concisely described by language, see e.g., .

Improved discrete latent variable models could also prove useful for learning novel data compression algorithms BID32 , while having far more interpretable representations of the data.

We build on Vector Quantized Variational Autoencoder (VQ-VAE) , a recently proposed training technique for learning discrete latent variables.

The method uses a learned code-book combined with nearest neighbor search to train the discrete latent variable model.

The nearest neighbor search is performed between the encoder output and the embedding of the latent code using the 2 distance metric.

VQ-VAE adopts the standard latent variable model generative process, first sampling latent codes from a prior, P (z), which are then consumed by the decoder to generate data from P (x | z).

In van den , the authors use both uniform and autoregressive priors for P (z).

The resulting discrete autoencoder obtains impressive results on unconditional image, speech, and video generation.

In particular, on image generation, VQ-VAE was shown to perform almost on par with continuous VAEs on datasets such as CIFAR-10 (van den ).

An extension of this method to conditional supervised generation, out-performs continuous autoencoders on WMT English-German translation task .The work of introduced the Latent Transformer, which set a new stateof-the-art in non-autoregressive Neural Machine Translation.

However, additional training heuristics, namely, exponential moving averages (EMA) of cluster assignment counts, and product quantization BID24 were essential to achieve competitive results with VQ-VAE.

In this work, we show that tuning for the code-book size can significantly outperform the results presented in .

We also exploit VQ-VAE's connection with the expectation maximization (EM) algorithm BID3 , yielding additional improvements.

With both improvements, we achieve a BLEU score of 22.4 on English to German translation, outperforming by 2.6 BLEU.

Knowledge distillation BID7 BID12 ) provides significant gains with our best models and EM, achieving 26.7 BLEU, which almost matches the autoregressive transformer model with no beam search at 27.0 BLEU, while being 3.3× faster.

Our contributions can be summarized as follows:1.

We show that VQ-VAE from van den can outperform previous state-of-the-art without product quantization.

2.

Inspired by the EM algorithm, we introduce a new training algorithm for training discrete variational autoencoders, that outperforms the previous best result with discrete latent autoencoders for neural machine translation.

3. Using EM training, and combining it sequence level knowledge distillation BID7 BID12 , allows us to develop a non-autoregressive machine translation model whose accuracy almost matches a strong greedy autoregressive baseline Transformer, while being 3.3 times faster at inference.

4.

On the larger English-French dataset, we show that denoising discrete autoencoders gives us a significant improvement (1.0 BLEU) on top of our non-autoregressive baseline (see Section D).

The connection between K-means, and hard EM, or the Viterbi EM algorithm is well known BID1 , where the former can be seen a special case of hard-EM style algorithm with a mixture-of-Gaussians model with identity covariance and uniform prior over cluster probabilities.

In the following sections we briefly explain the VQ-VAE discrete autoencoder for completeness and it's connection to classical EM.

VQ-VAE models the joint distribution P Θ (x, z) where Θ are the model parameters, x is the data point and z is the sequence of discrete latent variables or codes.

Each position in the encoded sequence has its own set of latent codes.

Given a data point, the discrete latent code in each position is selected independently using the encoder output.

For simplicity, we describe the procedure for selecting the discrete latent code (z i ) in one position given the data point (x i ).

The encoder output z e (x i ) ∈ R D is passed through a discretization bottleneck using a nearest-neighbor lookup on embedding vectors e ∈ R K×D .

Here K is the number of latent codes (in a particular position of the discrete latent sequence) in the model.

More specifically, the discrete latent variable assignment is given by, DISPLAYFORM0 The selected latent variable's embedding is passed as input to the decoder, DISPLAYFORM1 The model is trained to minimize: DISPLAYFORM2 where l r is the reconstruction loss of the decoder given z q (x) (e.g., the cross entropy loss), and, sg (.)

is the stop gradient operator defined as follows:sg (x) = x forward pass 0 backward pass To train the embedding vectors e ∈ R K×D , van den proposed using a gradient based loss function DISPLAYFORM3 and also suggested an alternate technique of training the embeddings: by maintaining an exponential moving average (EMA) of all the encoder hidden states that get assigned to it.

It was observed in that the EMA update for training the code-book embedding, results in more stable training than using gradient-based methods.

We analyze this in more detail in Section 5.1.1.Specifically, an exponential moving average is maintained over the following two quantities: 1) the embeddings e j for every j ∈ [1, . . .

, K] and, 2) the count c j measuring the number of encoder hidden states that have e j as it's nearest neighbor.

The counts are updated in a mini-batch of targets as: DISPLAYFORM4 with the embedding e j being subsequently updated as: DISPLAYFORM5 where 1 [.] is the indicator function and λ is a decay parameter which we set to 0.999 in our experiments.

This amounts to doing stochastic gradient in the space of both code-book embeddings and cluster assignments.

These techniques have also been successfully used in minibatch K-means BID30 and online EM BID19 BID29 .The generative process for our latent variable NMT model, P (y, z | x), begins by autoregressively sampling a sequence of discrete latent codes from a model conditioned on the input x, DISPLAYFORM6 which we refer to as the Latent Predictor model .

The decoder then consumes this sequence of discrete latent variables to generate the target y all at once, where DISPLAYFORM7 The autoregressive learned prior prior is fit on the discrete latent variables produced by the autoencoder.

Our goal is to learn a sequence of latents, that is much shorter than the targets, |z| |y|, thereby speeding up decoding significantly with no loss in accuracy.

The architecture of the encoder, the decoder, and the latent predictor model are described in further detail in Section 5.

In this section we briefly recall the hard Expectation maximization (EM) algorithm BID3 .

Given a set of data points (x 1 , . . .

, x N ), the hard EM algorithm approximately solves the following optimization problem: DISPLAYFORM0 Hard EM performs coordinate descent over the following two coordinates: the model parameters Θ, and the hidden variables z 1 , . . .

, z N .

In other words, hard EM consists of repeating the following two steps until convergence: DISPLAYFORM1 A special case of the hard EM algorithm is K-means clustering BID21 BID1 where the likelihood is modelled by a Gaussian with identity covariance matrix.

Here, the means of the K Gaussians are the parameters to be estimated, DISPLAYFORM2 With a uniform prior over the hidden variables (P Θ (z i ) = 1 K ), the marginal is given by DISPLAYFORM3 In this case, equation FORMULA8 is equivalent to: DISPLAYFORM4 Note that optimizing equation FORMULA12 is NP-hard, however one can find a local optima by applying coordinate descent until convergence:1.

E step: Cluster assignment is given by, DISPLAYFORM5 2.

M step: The means of the clusters are updated as, DISPLAYFORM6 We can now easily see the connections between the training updates of VQ-VAE and K-means clustering.

The encoder output z e (x) ∈ R D corresponds to the data point while the discrete latent variables corresponds to clusters.

Given this, Equation 1 is equivalent to the E-step

In this section, we investigate a new training strategy for VQ-VAE using the EM algorithm.

First, we briefly describe the EM algorithm.

While the hard EM procedure selects one cluster or latent variable assignment for a data point, here the data point is assigned to a mixture of clusters.

Now, the optimization objective is given by, DISPLAYFORM0 Coordinate descent algorithm is again used to approximately solve the above optimization algorithm.

The E and M step are given by:1.

E step: DISPLAYFORM1 2.

M step: DISPLAYFORM2

Now, we describe vector quantized autoencoders training using the EM algorithm.

As discussed in the previous section, the encoder output z e (x) ∈ R D corresponds to the data point while the discrete latent variables corresponds to clusters.

The E step instead of hard assignment now produces a probability distribution over the set of discrete latent variables (Equation 12).

Following VQ-VAE, we continue to assume a uniform prior over clusters, since we observe that training the cluster priors seemed to cause the cluster assignments to collapse to only a few clusters.

The probability distribution is modeled as a Gaussian with identity covariance matrix, DISPLAYFORM0 As an alternative to computing the full expectation term in the M step (Equation 13) we perform Monte-Carlo Expectation Maximization BID39 by drawing DISPLAYFORM1 , where Multinomial(l 1 , . . .

, l K ) refers to the K-way multinomial distribution with logits l 1 , . . .

, l K .

This results in a less diffuse target for the autoregressive prior.

Thus, the E step can be finally written as: DISPLAYFORM2 The model parameters Θ are then updated to maximize this Monte-Carlo estimate in the M step given by DISPLAYFORM3 Instead of exactly following the above M step update, we use the EMA version of this update similar to the one described in Section 2.1.When sending the embedding of the discrete latent to the decoder, instead of sending the posterior mode, argmax z P (z | x), similar to hard EM and K-means, we send the average of the embeddings of the sampled latents: DISPLAYFORM4 Since m latent code embeddings are sent to the decoder in the forward pass, all of them are updated in the backward pass for a single training example.

In hard EM training, only one of them is updated during training.

Sending averaged embeddings also results in more stable training using the EM algorithm compared to VQ-VAE as shown in Section 5.To train the latent predictor model (Section 2.1) in this case, we use an approach similar to label smoothing BID26 : the latent predictor model is trained to minimize the cross entropy loss with the labels being the average of the one-hot labels of z

Variational autoencoders were first introduced by BID13 for training continuous representations; unfortunately, training them for discrete latent variable models has proved challenging.

One promising approach has been to use various gradient estimators for discrete latent variable models, starting with the REINFORCE estimator of BID40 , an unbiased, high-variance gradient estimator.

Subsequent work on improving the variance of the REINFORCE estimator are REBAR and RELAX BID5 ).

An alternate approach towards gradient estimators is to use continuous relaxations of categorical distributions, for e.g., the Gumbel-Softmax reparametrization trick BID8 BID22 .

These methods provide biased but low variance gradients for training.

Machine translation using deep neural networks have been shown to achieve impressive results BID31 BID0 Vaswani et al., 2017) .

The state-of-the-art models in Neural Machine Translation are all auto-regressive, which means that during decoding, the model consumes all previously generated tokens to predict the next one.

Recently, there have been multiple efforts to speed-up machine translation decoding.

BID6 attempts to address this issue by using the Transformer model (Vaswani et al., 2017) together with the REINFORCE algorithm BID40 , to model the fertilities of words.

The main drawback of the approach of BID6 is the need for extensive fine-tuning to make policy gradients work, as well as the non-generic nature of the solution.

BID18 propose a non-autoregressive model using iterative refinement.

Here, instead of decoding the target sentence in one-shot, the output is successively refined to produce the final output.

While the output is produced in parallel at each step, the refinement steps happen sequentially.

In this section we report our experiments with VQ-VAE and EM on the English-German translation task, with the aim of improving the decoding speed of autoregressive translation models.

Our model and generative process follows the architecture proposed in and is depicted in Figure 1 .

For all our experiments, we use the Adam BID15 optimizer and decay the learning rate exponentially after initial warm-up steps.

Unless otherwise stated, the dimension of the hidden states of the encoder and the decoder is 512, see TAB4 for a comparison of models with lower dimension.

For all configurations we select the optimal hyperparameters by using WMT'13 English-German as the validation set and reporting the BLEU score on the WMT'14 English-German test set.

Figure 1: VQ-VAE model adapted to conditional supervised translation as described in .

We use x and y to denote the source and target sentence respectively.

The encoder, the decoder and the latent predictor now additionally condition on the source sentence x.

In Neural Machine Translation with latent variables, we model P (y, z | x), where y and x are the target and source sentence respectively.

Our model architecture, depicted in Figure 1 , is similar to the one in .

The encoder function is a series of strided convolutional layers with residual convolutional layers in between and takes target sentence y as input.

The source sentence x is converted to a sequence of hidden states through multiple causal self-attention layers.

In , the encoder of the autoencoder attends additionally to this sequence of continuous representation of the source sentence.

We use VQ-VAE as the discretization algorithm.

The decoders, applied after the bottleneck layer uses transposed convolution layers whose continuous output is fed to a transformer decoder with causal attention, which generates the output.

The results are summarized in Table 1 .

Our implementation of VQ-VAE achieves a significantly better BLEU score and faster decoding speed compared to .

We found that tuning the code-book size (number of clusters) for using 2 12 discrete latents achieves the best accuracy which is 16 times smaller as compared to the code-book size in .

Additionally, we see a large improvement in the performance of the model by using sequence-level distillation BID7 BID12 , as has been observed previously in non-autoregressive models BID6 BID18 .

Our teacher model is a base Transformer (Vaswani et al., 2017 ) that achieves a BLEU score of 28.1 and 27.0 on the WMT'14 test set using beam search decoding and greedy decoding respectively.

The distilled data is decoded from the base Transformer using a beam size of 4.

Our VQ-VAE model trained with soft EM and distillation, achieves a BLEU score of 26.7, without noisy parallel decoding BID6 ).

This perforamce is 1.4 bleu points lower than an autoregressive model decoded with a beam size of 4, while being 4.1× faster.

Importantly, we nearly match the same autoregressive model with beam size 1 (greedy decoding), with a 3.3× speedup.

The length of the sequence of discrete latent variables is shorter than that of target sentence y. Specifically, at each compression step of the encoder we reduce its length by half.

We denote by n c , the compression factor for the latents, i.e. the number of steps for which we do this compression.

In almost all our experiments, we use n c = 3 reducing the length by 8.

We can decrease the decoding time further by increasing the number of compression steps.

As shown in Table 1 , by setting n c to 4, the decoding time drops to 58 milliseconds achieving 25.4 BLEU while a NAT model BID6 with similar decoding speed achieves only 18.7 BLEU.

Note that, all NAT models also train with sequence level knowledge distillation from an autoregressive teacher.

Attention to Source Sentence Encoder: While the encoder of the discrete autoencoder in attends to the output of the encoder of the source sentence, we find that to be unnecessary, with both models achieving the same BLEU score with 2 12 latents.

Removing this attention step results in more stable training (see FIG2 and is the main reason why VQ-VAE works in our setup (see Table 1 ) without the use of Product Quantization (DVQ) .

Note that the decoder of the discrete autoencoder in both and our work does not attend to the source sentence.

TAB2 shows the BLEU score for different code-book sizes for models trained using VQ-VAE without distillation.

While use FORMULA2 16 as their code-book size, we find that 2 12 gives the best performance.

Number of samples in Monte-Carlo EM update: While training with EM, we perform a Monte-Carlo update with a small number of samples (Section 3.2).

TAB3 shows the impact of number of samples on the final BLEU score.

We compare the Gumbel-Softmax of BID8 BID22 and the improved semantic hashing discretization technique proposed in to VQ-VAE.

When trained with sequence level knowledge distillation, the model using Gumbel-Softmax reached 23.2 BLEU, the model using improved semantic hashing reached 24.1 BLEU, and the model using VQ-VAE reached 26.4 BLEU on WMT'14 English-German.

Table 1 : BLEU score and decoding times for different models on the WMT'14 EnglishGerman translation dataset.

The baseline is the autoregressive Transformer of Vaswani et al. (2017) with no beam search, NAT denotes the Non-Autoregressive Transformer of BID6 , and LT + Semhash denotes the Latent Transformer from van den using the improved semantic hashing discretization technique of .

NPD refers to noisy parallel decoding as described in BID6 .

We use the notation n c to denote the compression factor for the latents, and the notation n s to denote the number of samples used to perform the Monte-Carlo approximation of the EM algorithm.

Distillation refers to sequence level knowledge distillation from BID7 ; BID12 .

We used a code-book of size 2 12 for VQ-VAE (for with and without EM) with a hidden dimension of size 512.

Decoding is performed on a single CPU machine with an NVIDIA GeForce GTX 1080 with a batch size of 1 * Speedup reported for these items are compared to the decode time of 408 ms for an autoregressive Transformer from BID6 .

We investigate an alternate training technique for VQ-VAE inspired by its connection to the EM algorithm.

Training the discrete autoencoder with EM and combining it with sequence level knowledge distillation, allows us to develop a non-autoregressive machine translation model whose accuracy almost matches a greedy autoregressive baseline, while being 3.3 times faster at inference.

While sequence distillation is very important for training our best model, we find that the improvements from EM on harder tasks is quite significant.

We hope that our results will inspire further research on using vector quantization for fast decoding of autoregressive sequence models.

Figure 2: VQ-VAE model as described in van den Oord et al. FORMULA0 for image reconstruction.

We use the notation x to denote the input image, with the output of the encoder z e (x) ∈ R D being used to perform nearest neighbor search to select the (sequence of) discrete latent variable.

The selected discrete latent is used to train the latent predictor model, while the embedding z q (x) of the selected discrete latent is passed as input to the decoder.

In this section we report additional experiments we performed using VQ-VAE and EM for the task of image reconstruction.

We train a discrete autoencoder with VQ-VAE (van den and EM on the CIFAR-10 data set, modeling the joint probability P (x, z), where x is the image and z are the discrete latent codes.

We use a field of 8 × 8 × 10 latents with a code-book of size 2 8 each containing 512 dimensions.

We maintain the same encoder and decoder as used in Machine Translation.

For the encoder, we use 4 convolutional layers, with kernel size 5 × 5 and strides 2 × 2, followed by 2 residual layers, and a single dense layer.

For the decoder, we use a single dense layer, 2 residual layers, and 4 deconvolutional layers.

FIG2 shows that our reconstructions are on par with hard EM training.

We also train discrete autoencoders on the SVHN dataset BID23 , with both VQ-VAE (van den and EM.

The autoencoder is similar to our CIFAR-10 model, where each n x = 32 × 32 × 3 image is encoded into 640 discrete latents from a shared codebook of size 256.

By contrasting the reconstructions from several training runs for VQ-VAE (left) and EM (right), we find that training with EM is more reliable and the reconstructions are of high quality (

Gradient based update vs EMA update of code-book: The original VQ-VAE paper proposed a gradient based update rule for learning the code-book where the code-book entries are trained by minimizing sg (z e (x)) − z q (x) 2 .

However, it was found in that the EMA update worked better than this gradient based loss.

Note that if the gradient based loss was minimized using SGD then the update rule for the embeddings is DISPLAYFORM0 for a learning rate η.

This is quite similar to the EMA update rule of Equation 5, with the only difference being that the latter also maintains an EMA over the counts c j .

When using SGD with momentum or Adam, the update rule becomes quite different however, since we now take the moving average of the gradient term itself, before subtracting it from current value of the embedding e j .

This is similar to the issue of using weight decay with Adam, where using the 2 penalty in the loss function results in worse performance BID20 .Model Size: The effect of model size on BLEU score for models trained with EM and distillation is shown in TAB4 .Robustness of EM to Hyperparameters: While EM training gives a small performance improvement, we find that it also leads to more robust training for machine translation.

Our experiments on image reconstruction on SVHN BID23 in section A also highlight the robustness of EM training.

The training approach from van den Oord et al.(2017) exhibits high variance on reconstruction quality, while EM is much more stable, resulting in good reconstructions in almost all training runs.

Figure 5: Comparison of VQ-VAE (green curve) vs EM with different number of samples (yellow and blue curves) on the WMT'14 English-German translation dataset with a codebook size of 2 14 , with the encoder of the discrete autoencoder attending to the output of the encoder of the source sentence as in .

The y-axis denotes the teacher-forced BLEU score on the test set, which is used only for evaluation while training.

Notice that the VQ-VAE run collapsed (green curve), while the EM runs (yellow and blue curves) exhibit more stability.

Emergence of EOS/PAD latent: We observe that all the latent sentences for a specific experiment with VQ-VAE or EM end with a fixed latent indicating the end of the sequence.

Since we always fix the length of the latent sentence to be 2 nc times smaller than the true sentence, the model learns to pad the remainder of the latent sequence with this special code (see Table 5 for examples).

Note that one can speed up decoding even further by stopping the Latent Predictor (LP) model as soon as it outputs this special code.

Table 5 : Example latent codes for sentences from the WMT'14 English-German dataset highlighting the emergence of the EOS/PAD latent (760 in this case).Denoising autoencoder: We also use word dropout with a dropout rate of 0.3 and word permutation with a shuffle rate of 0.5 as in BID17 .

On the WMT EnglishGerman we did not notice any improvement from using these regularization techniques, but on the larger WMT English-French dataset, we observe that using a denoising autoencoder significantly improves performance with a gain of 1.0 BLEU on VQ-VAE and 0.9 BLEU over EM (see Table 6 ).Additional analysis on latents: In order to compute correlations between the discrete latents and n-grams in the original text, we computed Point-wise Mutual Information (PMI) and tf-idf scores where the latents are treated as documents.

However, we were unable to see any semantic patterns that stood out in this analysis.

In this section we report preliminary results on the WMT English-French dataset without using knowledge distillation from an autoregressive teacher BID7 BID12 .

We use a Transformer base model from Vaswani et al. (2017) .

Our best non-autoregressive base model trained on non-distilled targets gets 30.0 BLEU compared to the autoregressive base model with the same choice of hyperparameters, which gets 33.3 BLEU (see Table 6 ).

As in the case of English-German, we anticipate that using knowledge distillation BID7 will likely close this gap.

Table 6 : BLEU score and decoding times for different models on the WMT'13 EnglishFrench translation dataset.

The baseline is the autoregressive Transformer of Vaswani et al. (2017) with no beam search, We use the notation n c to denote the compression factor for the latents, and the notation n s to denote the number of samples used to perform the Monte-Carlo approximation of the EM algorithm.

Reg. refers to word dropout with rate 0.3 and word permutation with shuffle rate 0.5 as described in Section C. The hidden dimension of the codebook is 512.

Decoding is performed on a single CPU machine with an NVIDIA GeForce GTX 1080 with a batch size of 1.

@highlight

Understand the VQ-VAE discrete autoencoder systematically using EM and use it to design non-autogressive translation model matching a strong autoregressive baseline.

@highlight

This paper introduces a new way of interpreting the VQ-VAE and proposes a new training algorithm based on the soft EM clustering.

@highlight

The paper presents an alternative view on the training procedure for the VQ-VAE using the soft EM algorithm