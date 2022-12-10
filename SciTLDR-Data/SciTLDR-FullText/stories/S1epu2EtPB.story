We consider the problem of generating plausible and diverse video sequences, when we are only given a start and an end frame.

This task is also known as inbetweening, and it belongs to the broader area of stochastic video generation, which is generally approached by means of recurrent neural networks (RNN).

In this paper, we propose instead a fully convolutional model to generate video sequences directly in the pixel domain.

We first obtain a latent video representation using a stochastic fusion mechanism that learns how to incorporate information from the start and end frames.

Our model learns to produce such latent representation by progressively increasing the temporal resolution, and then decode in the spatiotemporal domain using 3D convolutions.

The model is trained end-to-end by minimizing an adversarial loss.

Experiments on several widely-used benchmark datasets show that it is able to generate meaningful and diverse in-between video sequences, according to both quantitative and qualitative evaluations.

Imagine if we could teach an intelligent system to automatically turn comic books into animations.

Being able to do so would undoubtedly revolutionize the animation industry.

Although such an immensely labor-saving capability is still beyond the current state-of-the-art, advances in computer vision and machine learning are making it an increasingly more tangible goal.

Situated at the heart of this challenge is video inbetweening, that is, the process of creating intermediate frames between two given key frames.

Recent development in artificial neural network architectures (Simonyan & Zisserman, 2015; He et al., 2016) and the emergence of generative adversarial networks (GAN) (Goodfellow et al., 2014) have led to rapid advancement in image and video synthesis (Aigner & Körner, 2018; Tulyakov et al., 2017) .

At the same time, the problem of inbetweening has received much less attention.

The majority of the existing works focus on two different tasks: i) unconditional video generation, where the model learns the input data distribution during training and generates new plausible videos without receiving further input (Srivastava et al., 2015; Finn et al., 2016; Lotter et al., 2016) ; and ii) video prediction, where the model is given a certain number of past frames and it learns to predict how the video evolves thereafter (Vondrick et al., 2016; Saito et al., 2017; Tulyakov et al., 2017; Denton & Fergus, 2018) .

In most cases, the generative process is modeled as a recurrent neural network (RNN) using either long-short term memory (LSTM) cells (Hochreiter & Schmidhuber, 1997) or gated recurrent units (GRU) (Cho et al., 2014) .

Indeed, it is generally assumed that some form of a recurrent model is necessary to capture long-term dependencies, when the goal is to generate videos over a length that cannot be handled by pure frame-interpolation methods based on optical flow.

In this paper, we show that it is in fact possible to address the problem of video inbetweening using a stateless, fully convolutional model.

A major advantage of this approach is its simplicity.

The absence of recurrent components implies shorter gradient paths, hence allowing for deeper networks and more stable training.

The model is also more easily parallelizable, due to the lack of sequential states.

Moreover, in a convolutional model, it is straightforward to enforce temporal consistency with the start and end frames given as inputs.

Motivated by these observations, we make the following contributions in this paper:

• We propose a fully convolutional model to address the task of video inbetweening.

The proposed model consists of three main components: i) a 2D-convolutional image encoder, which maps the input key frames to a latent space; ii) a 3D-convolutional latent representation generator, which learns how to incorporate the information contained in the input frames with progressively increasing temporal resolution; and iii) a video generator, which uses transposed 3D-convolutions to decode the latent representation into video frames.

• Our key finding is that separating the generation of the latent representation from video decoding is of crucial importance to successfully address video inbetweening.

Indeed, attempting to generate the final video directly from the encoded representations of the start and end frames tends to perform poorly, as further demonstrated in Section 4.

To this end, we carefully design the latent representation generator to stochastically fuse the key frame representations and progressively increase the temporal resolution of the generated video.

• We carried out extensive experiments on several widely used benchmark datasets, and demonstrate that the model is able to produce realistic video sequences, considering key frames that are well over a half second apart from each other.

In addition, we show that it is possible to generate diverse sequences given the same start and end frames, by simply varying the input noise vector driving the generative process.

The rest of the paper is organized as follows: We review the outstanding literature related to our work in Section 2.

Section 3 describes our proposed model in details.

Experimental results, both quantitative and qualitative, are presented in Section 4, followed by our conclusions in Section 5.

Recent advances based on deep networks have led to tremendous progress in three areas related to the current work: i) video prediction, ii) video generation and iii) video interpolation.

Video prediction: Video prediction addresses the problem of producing future frames given one (or more) past frames of a video sequence.

The methods that belong to this group are deterministic, in the sense that always produce the same output for the same input and they are trained to minimize the L2 loss between the ground truth and the predicted future frames.

Most of the early works in this area adopted recurrent neural networks to model the temporal dynamics of video sequences.

In Srivastava et al. (2015) a LSTM encoder-decoder framework is used to learn video representations of image patches.

The work in Finn et al. (2016) extends the prediction to video frames rather than patches, training a convolutional LSTM.

The underlying idea is to compute the next frame by first predicting the motions of either individual pixels or image segments and then merge these predictions via masking.

A multi-layer LSTM is also used in Lotter et al. (2016) , progressively refining the prediction error.

Some methods do not use recurrent networks to address the problem of video prediction.

For example, a 3D convolutional neural network is adopted in Mathieu et al. (2016) .

An adversarial loss is used in addition to the L2 loss to ensure that the predicted frames look realistic.

More recently, Aigner & Körner (2018) proposed a similar approach, though in this case layers are added progressively to increase the image resolution during training (Karras et al., 2017) .

All the aforementioned methods aim at predicting the future frames in the pixel domain directly.

An alternative approach is to first estimate local and global transformations (e.g., affine warping and local filters), and then apply them to each frame to predict the next, by locally warping the image content accordingly (De Brabandere et al., 2016; Chen et al., 2017a; van Amersfoort et al., 2017) .

Video generation: Video generation differs from video prediction in that it aims at modelling future frames in a probabilistic manner, so as to generate diverse and plausible video sequences.

To this end, methods based on generative adversarial networks (GAN) and variational autoencoder networks (VAN) are being currently explored in the literature.

In Vondrick et al. (2016) a GAN architecture is proposed, which consists of two generators (to produce, respectively, foreground and static background pixels), and a discriminator to distinguish between real and generated video sequences.

While Vondrick et al. (2016) generates the whole output video sequence from a single latent vector, in Saito et al. (2017) a temporal generator is first used to produce a sequence of latent vectors that captures the temporal dynamics.

Subsequently an image generator produces the output images from the latent vectors.

Both the generators and the discriminator are based on CNNs.

The model is also able to generate video sequences conditionally on an input label, as well as interpolating between frames by first linearly interpolating the temporal latent vectors.

To address mode collapse in GANs, Denton & Fergus (2018) proposes to use a variational approach.

Each frame is recursively generated combining the previous frame encoding with a latent vector.

This is fed to a LSTM, whose output goes through a decoder.

Similarly to this, Babaeizadeh et al. (2018) samples a latent vector, which is then used as conditioning for the deterministic frame prediction network in Finn et al. (2016) .

A variational approach is used to learn how to sample the latent vector, conditional on the past frames.

Other methods do not attempt to predict the pixels of the future frame directly.

Conversely, a variational autoencoder is trained to generate plausible differences between consecutive frames (Xue et al., 2016) , or motion trajectories (Walker et al., 2016) .

Recently, proposed to use a loss function that combines a variational loss (to produce diverse videos) (Denton & Fergus, 2018) with an adversarial loss (to generate realistic frames) (Saito et al., 2017) .

Video sequences can be modelled as two distinct components: content and motion.

In Tulyakov et al. (2017) the latent vector from which the video is generated is divided in two parts: content and motion.

This leads to improved quality of the generated sequences when compared with previous approaches (Vondrick et al., 2016; Saito et al., 2017) .

A similar idea is explored in Villegas et al. (2017a) , where two encoders, one for motion and one for content, are used to produce hidden representations that are then decoded to a video sequence.

Also explicitly separates motion and content in two streams, which are generated by means of a variational network and then fused to produce the predicted sequence.

An adversarial loss is then used to improve the realism of the generated videos.

All of the aforementioned methods are able to predict or generate just a few video frames into the future.

Long-term video prediction has been originally addressed in Oh et al. (2015) with the goal of predicting up to 100 future frames of an Atari game.

The current frame is encoded using a CNN or LSTM, transformed conditionally on the player action, and decoded into the next frame.

More recently, Villegas et al. (2017b) addressed a similar problem, but for the case of real-world video sequences.

The key idea is to first estimate high-level structures from past frames (e.g., human poses).

Then, a LSTM is used to predict a sequence of future structures, which are decoded to future frames.

One shortcoming of Villegas et al. (2017b) is that it requires ground truth landmarks as supervision.

This is addressed in Wichers et al. (2018) , which proposes a fully unsupervised method that learns to predict a high-level encoding into the future.

Then, a decoder with access to the first frame generates the future frames from the predicted high-level encoding.

Video interpolation: Video interpolation is used to increase the temporal resolution of the input video sequence.

This is addressed with different approaches: optical flow based interpolation (Ilg et al., 2017; , phase-based interpolation (Meyer et al., 2018) , and pixels motion transformation (Niklaus et al., 2017; Jiang et al., 2018) .

These method typically target temporal super-resolution and the frame rate of the input sequence is often already sufficiently high.

Interpolating frames becomes more difficult when the temporal distance between consecutive frames increases.

Long-term video interpolation received far less attention in the past literature.

Deterministic approaches have been explored using either block-based motion estimation/compensation (Ascenso et al., 2005) , or convolutional LSTM models (Kim et al., 2018) .

Our work is closer to those using generative approaches.

In Chen et al. (2017b) two convolutional encoders are used to generate hidden representations of both the first and last frame, which are then fed to a decoder to reconstruct all frames in between.

A variational approach is presented in .

A multi-layer convolutional LSTM is used to interpolate frames given a set of extended reference frames, with the goal of increasing the temporal resolution from 2 fps to 16 fps.

In our experiments, we compare our method with those in Niklaus et al. (2017) , Jiang et al. (2018) , and .

The proposed model receives three inputs: a start frame x s , an end frame x e , and a Gaussian noise vector u ∈ R D .

The output of the model is a video (x s ,x 1 , . . . ,x T −2 , x e ), where different se-

Up-sample (time-wise)

Sigmoid Sigmoid × × Figure 1 : Layout of the model used to generate the latent video representation z. The inputs are the encoded representations of the start and and frames E(x s ) and E(x e ), together with a noise vector u.

quences of plausible in-between frames (x i )

are generated by feeding different instantiations of the noise vector u. In the rest of this paper, we set T = 16 and D = 128.

The model consists of three components: an image encoder, a latent representation generator and a video generator.

In addition, a video discriminator and an image discriminator are added so that the whole model can be trained using adversarial learning (Goodfellow et al., 2014) to produce realistic video sequences.

The image encoder E(x) receives as input a video frame of size H 0 × W 0 and produces a feature map of shape H × W × C, where C is the number of channels.

The encoder architecture consists of six layers, alternating between 4 × 4 convolutions with stride-2 down-sampling and regular 3 × 3 convolutions, followed by a final layer to condense the feature map to the target depth C. This results in spatial dimensions H = H 0 /8 and W = W 0 /8.

We set C = 64 in all our experiments.

The latent representation generator G Z (·) receives as input E(x s ), E(x e ) and u, and produces an output tensor of shape T × H × W × C. Its main function is to gradually fill in the video content between the start and end frames, working directly in the latent space defined by the image encoder.

The model architecture is composed of a series of L residual blocks (He et al., 2016) , each consisting of 3D convolutions and stochastic fusion with the encoded representations of x s and x e .

This way, each block progressively learns a transformation that improves the video content generated by the previous block.

The generic l-th block is represented by the inner rectangle in Figure 1 .

Note that the lengths of the intermediate representations can differ from the final video length T , due to the use of a coarse-to-fine scheme in the time dimension.

To simplify the notation, we defer its description to the end of this section and omit the implied temporal up-sampling from the equations.

Let T (l) denote the representation length within block l. First, we produce a layer-specific noise tensor of shape T (l) × C by applying a linear transformation to the input noise vector u:

where

(l) C , and reshaping the result into a T (l) × C tensor u (l) .

This is used to drive two stochastic "gating" functions for the start and end frames, respectively:

where * denotes convolution along the time dimension, k s , k e are kernels of width 3 and depth C, and σ(·) is the sigmoid activation function.

The gating functions are used to progressively fuse the encoded representations of the start and end frames with the intermediate output of the previous layer z (l−1) , as described by the following equation:

where n

n denotes an additional learned stochastic component added to stimulate diversity in the generative process.

Note that z (l) in has shape T (l) × H × W × C. Therefore, to compute the component-wise multiplication · , E(x s ) and E(x e ) (each of shape S × S × C) are broadcast (i.e., replicated uniformly) T (l) times along the time dimension, while g

e and n (l) (each of shape T (l) × C) are broadcast H × W times over the spatial dimensions.

The idea of the fusion step is similar to that of StyleGAN , albeit with different construction and purposes.

Finally, the fused input is convolved spatially and temporally with 3 × 3 × 3 kernels k

2 in a residual unit (He et al., 2016) :

where h is the leaky ReLU (Maas et al., 2013 ) activation function (with parameter α = 0.2).

given E(x s ) and E(x e ), with A, k, b being its learnable parameters.

The generation of the overall latent video representation z ∈ R T ×S×S×C can be expressed as:

Coarse-to-fine generation: For computational efficiency, we adopt a coarse-to-fine scheme in the time dimension, represented by the outer dashed rectangle in Figure 1 .

More specifically we double the length of z (l) every L/3 generator blocks, i.e., z (1) , . . .

, z (L/3) have length T /4 = 4, z (L/3+1) , . . .

, z (2L/3) have T /2 = 8, and z (2L/3+1) , . . .

, z (L) have the full temporal resolution T = 16.

We initialize z (0) to (E(x s ), E(x e )) (which becomes (E(x s ), E(x s ), E(x e ), E(x e )) after the first up-sampling) and set L = 24, resulting in 8 blocks per granularity level.

The video generator G V produces the output video sequence (x s ,x 1 ,x 2 , . . . , x e ) = G V (z) from the latent video representation z using spatially transposed 3D convolutions.

The generator architecture alternates between 3 × 3 × 3 regular convolutions and transposed 3 × 4 × 4 convolutions with a stride of (1, 2, 2), hence applying only spatial (but not temporal) up-sampling.

Note that it actually generates all T frames including the "reconstructed" start framex 0 and end framex T −1 , though they are not used and are always replaced by the real x s and x e in the output.

We train our model end-to-end by minimizing an adversarial loss function.

To this end, we train two discriminators: a 3D convolutional video discriminator D V and a 2D convolutional image discriminator D I , following the approach of Tulyakov et al. (2017) .

The video discriminator has a similar architecture to Tulyakov et al. (2017) , except that in our case we produce a single output for the entire video rather than for its sub-volumes ("patches").

For the image discriminator, we use a Resnet-based architecture (He et al., 2016) instead of the DCGAN-based architecture (Radford et al., 2016) used in Tulyakov et al. (2017) .

Let X = (x s , x 1 , . . .

, x T −2 , x e ) denote a real video andX = (x s ,x 1 , . . .

,x T −2 , x e ) denote the corresponding generated video conditioned on x s and x e .

Adopting the non-saturating log-loss, training amounts to optimizing the following adversarial objectives:

During optimization we replace the average over the T − 2 intermediate frames with a single uniformly sampled frame to save computation, as is done in Tulyakov et al. (2017) .

This does not change the convergence properties of stochastic gradient descent, since the two quantities have the same expectation.

We regularize the discriminators by penalizing the derivatives of the pre-sigmoid logits with respect to their input videos and images, as is proposed in Roth et al. (2017) to improve GAN stability and prevent mode collapse.

In our case, instead of the adaptive scheme of Roth et al. (2017) , we opt for a constant coefficient of 0.1 for the gradient magnitude, which we found to be more reliable in our experiments.

We use batch normalization (Ioffe & Szegedy, 2015) on all 2D and 3D convolutional layers in the generator and layer normalization (Ba et al., 2016) in the discriminators.

1D convolutions and fully-connected layers are not normalized.

Architectural details of the encoder, decoder, and discriminators are further provided in Appendix A.

We evaluated our approach on three well-known public datasets: BAIR robot pushing (Ebert et al., 2017) , KTH Action Database (Schuldt et al., 2004) , and UCF101 Action Recognition Data Set (Soomro et al., 2012) .

All video frames were down-sampled and cropped to 64×64, and subsequences of 16 frames were used in all the experiments, that is, 14 intermediate frames are generated.

The videos in KTH and UCF101 datasets are 25 fps, translating to key frames 0.6 seconds apart.

The frame rate of BAIR videos is not provided, though visually it appears to be much lower, hence longer time in between key frames.

For all the datasets, we adopted the conventional train/test splits practised in the literature.

A validation set held out from the training set was used for model checkpoint selection.

More details on the exact splits are provided in Appendix B. We did not use any dataset-specific tuning of hyper-parameters, architectural choices, or training schemes.

Our main objective is to generate plausible transition sequences with characteristics similar to real videos, rather than predicting the exact content of the original sequence from which the key frames were extracted.

Therefore we use the recently proposed Fréchet video distance (FVD) (Unterthiner et al., 2018) as our primary evaluation metrics.

The FVD is equivalent to the Fréchet Inception distance (FID) (Heusel et al., 2017) widely used for evaluating image generative models, but revisited in a way that it can be applied to evaluate videos, by adopting a deep neural network architecture that computes video embeddings taking the temporal dimension explicitly into account.

The FVD is a more suitable metrics for evaluating video inbetweening than the widely used structural similarity index (SSIM) (Wang et al., 2004 ).

The latter is suitable when evaluating prediction tasks, as it compares each synthetized frame with the original reference at the pixel level.

Conversely, FVD compares the distributions of generated and ground-truth videos in an embedding space, thus measuring whether the synthesized video belongs to the distribution of realistic videos.

Since the FVD was only recently proposed, we also report the SSIM to be able to compare with the previous literature.

During testing, we ran the model 100 times for each pair of key frames, feeding different instances of the noise vector u to generate different sequences consistent with the given key frames, and computed the FVD for each of these stochastic generations.

This entire procedure was repeated 10 121 [0.112, 0.129] times for each model variant and dataset to account for the randomness in training.

We report the mean over all training runs and stochastic generations as well as the confidence intervals obtained by means of the bootstrap method (Efron & Tibshirani, 1993) .

For training we used the ADAM (Kingma & Ba, 2015) optimizer with β 1 = 0.5, β 2 = 0.999, = 10 −8 , and ran it on batches of 32 samples with a conservative learning rate of 5 × 10 − 5 for 500,000 steps.

A checkpoint was saved every 5000 steps, resulting in 100 checkpoints.

Training took around 5 days on a single Nvidia Tesla V100 GPU.

The checkpoint for evaluation was selected to be the one with the lowest FVD on the validation set.

To assess the impact of the stochastic fusion mechanism as well the importance of having a separate latent video representation generator component, we compare the full model with baselines in which the corresponding components are omitted.

• Baseline without fusion: The gating functions (Equation 2 and 3) are omitted and Equation 4 reduces to z

• Naïve: The entire latent video representation generator described in Section 3.2 is omitted.

Instead, decoding with transposed 3D convolution is performed directly on the (stacked) start/end frame encoded representations z (0) = (E(x 1 ), E(x N )) (which has dimensionality 2×8×8), using a stride of 2 in both spatial and temporal dimensions when up-scaling, to eventually produce 16 64×64 frames.

To maintain stochasticity in the generative process, a spatially uniform noise map is generated by sampling a Gaussian vector u, applying a (learned) linear transform, and adding the result in the latent space before decoding.

The results in Table 1 shows that the dedicated latent video representation generator is indispensable, as the naïve baseline performs rather poorly.

Moreover, stochastic fusion improves the quality of video generation.

Note that the differences are statistically significant at 95% confidence level across all three datasets.

To illustrate the generated videos, Figure 2 shows some exemplary outputs of our full model.

The generated sequence is not expected (or even desired) to reproduce the ground truth, but only needs to be similar in style and consistent with the given start and end frames.

The samples show that the model does well in this area.

For stochastic generation, good models should produce samples that are not only high-quality but also diverse.

Following the approach of , we measure diversity by means of the average pairwise cosine distance (i.e., 1 − cosine similarity) in the FVD embedding space among samples generated from the same start/end frames.

The results Table 2 shows that incorporating fusion increases sample diversity and the difference is statistically significant.

A qualitative illustration of the diversity in the generated videos is further illustrated in Figure 3 , where we take the average of 100 generated videos conditioned on the same start and end frames.

If the robot arm has a very diverse set of trajectories, we should expect to see it "diffuse" into the background due to averaging.

Indeed this is the case, especially near the middle of the sequence.

Finally we computed the average SSIM for our method for each dataset in order to compare our results with those previously reported in the literature, before the FVD metrics was introduced.

The results are shown in Table 3 alongside several existing methods that are capable of video inbetweening, ranging from RNN-based video generation to optical flow-based interpolation (Niklaus et al., 2017; Jiang et al., 2018) 1 .

Note that the competing methods generate 7 frames 1 The numbers for these methods are cited directly from .

Table 3 : Average SSIM of our model using direct 3D convolution and alternative methods based on RNN, such as SDVI , or optical flow, such as SepConv (Niklaus et al., 2017) and SuperSloMo (Jiang et al., 2018) .

Higher is better.

Note the difference in setup: our model spans a time base twice as long as the others.

The SSIM for each test example is computed on the best sequence out of 100 stochastic generations, as per standard practice (Babaeizadeh et al., 2018; Denton & Fergus, 2018; .

We report the mean and the 95%-confidence interval for our model over 10 training runs.

and are conditioned on potentially multiple frames before and after.

In contrast our model generates 14 frames, i.e., over a time base twice as long, and it is conditioned on only one frame before and after.

Consequently, the SSIM figures are not directly comparable.

However it is interesting to see that on UCF101, the most challenging dataset among the three, our model attains higher SSIM than all the other methods despite having to generate much longer sequences.

This demonstrates the potential of the direct convolution approach to outperform existing methods, especially on difficult tasks.

It is also worth noting from Table 3 that purely optical flow-based interpolation methods achieve essentially the same level of SSIM as the sophisticated RNN-based SDVI on BAIR and KTH, which suggests either that a 7-frame time base is insufficient in length to truly test video inbetweening models or that the SSIM is not an ideal metric for this task.

We presented a method for video inbetweening using only direct 3D convolutions.

Despite having no recurrent components, our model produces good performance on most widely-used benchmark datasets.

The key to success for this approach is a dedicated component that learns a latent video representation, decoupled from the final video decoding phase.

A stochastic gating mechanism is used to progressively fuse the information of the given key frames.

The rather surprising fact that video inbetweening can be achieved over such a long time base without sophisticated recurrent models may provide a useful alternative perspective for future research on video generation.

@highlight

This paper presents method for stochastically generating in-between video frames from given key frames, using direct 3D convolutions.