Existing unsupervised video-to-video translation methods fail to produce translated videos which are frame-wise realistic, semantic information preserving and video-level consistent.

In this work, we propose a novel unsupervised video-to-video translation model.

Our model decomposes the style and the content, uses specialized encoder-decoder structure and propagates the inter-frame information through bidirectional recurrent neural network (RNN) units.

The style-content decomposition mechanism enables us to achieve long-term style-consistent video translation results as well as provides us with a good interface for modality flexible translation.

In addition, by changing the input frames and style codes incorporated in our translation, we propose a video interpolation loss, which captures temporal information within the sequence to train our building blocks in a self-supervised manner.

Our model can produce photo-realistic, spatio-temporal consistent translated videos in a multimodal way.

Subjective and objective experimental results validate the superiority of our model over the existing methods.

Recent image-to-image translation (I2I) works have achieved astonishing results by employing Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) .

Most of the GAN-based I2I methods mainly focus on the case where paired data exists (Isola et al. (2017b) , , Wang et al. (2018b) ).

However, with the cycle-consistency loss introduced in CycleGAN , promising performance has been achieved also for the unsupervised image-to-image translation (Huang et al. (2018) , Almahairi et al. (2018) , Liu et al. (2017) , Mo et al. (2018) , Romero et al. (2018) , Gong et al. (2019) ).

While there is an explosion of papers on I2I, its video counterpart is much less explored.

Compared with the I2I task, video-to-video translation (V2V) is more challenging.

Besides the frame-wise realistic and semantic preserving requirements, which is also required in the I2I task, V2V methods additionally need to consider the temporal consistency issue for generating sequence-wise realistic videos.

Consequently, directly applying I2I methods on each frame of the video will not work out as those methods fail to utilize temporal information and can not assure any temporal consistency within the sequence.

In their seminal work, Wang et al. (2018a) combined the optical flow and video-specific constraints and proposed a general solution for V2V in a supervised way.

Their sequential generator can generate long-term high-resolution video sequences.

However, their vid2vid model (Wang et al. (2018a) ) relies heavily on labeled data.

Based on the I2I CycleGAN approach, previous methods on unsupervised V2V proposed to design spatio-temporal translator or loss to achieve more temporally consistent results while preserving semantic information.

In order to generate temporally consistent video sequences, Bashkirova et al. (2018) proposed a 3DCycleGAN method which adopts 3D convolution in the generator and discriminator of the CycleGAN framework to capture temporal information.

However, since the small 3D convolution operator (with temporal dimension 3) only captures dependency between adjacent frames, 3DCycleGAN can not exploit temporal information for generating long-term consistent video sequences.

Furthermore, the vanilla 3D discriminator is also limited in capturing complex temporal relationships between video frames.

As a result, when the gap between input and target domain is large, 3DCycleGAN tends to sacrifice the image-level reality and generates blurry and gray outcomes.

Recently, Bansal et al. (2018) designed a Recycle loss for jointly modeling the spatio-temporal relationship between video frames.

They trained a temporal predictor to predict the next frame based on two past frames, and plugged the temporal predictor in the cycle-loss to impose the spatio-temporal constraint on the traditional image translator.

As the temporal predictors can be trained from video sequences in source and target domain in a self-supervised manner, the recycle-loss is more stable than the 3D discriminator loss proposed by Bashkirova et al. (2018) .

The RecycleGAN method of Bansal et al. (2018) achieved state-of-the-art unsupervised V2V results.

Despite its success in translation scenarios with less variety, such as faceto-face or flower-to-flower, we have experimentally found that applying RecycleGAN to translate videos between domains with a large gap is still challenging.

We think the following two points are major reasons which affect the application of RecycleGAN method in complex scenarios.

On one hand, the translator in Bansal et al. (2018) processes input frames independently, which has limited capacity in exploiting temporal information; and on the other hand, its temporal predictor only imposes the temporal constraint between a few adjacent frames, the generated video content still might shift abnormally: a sunny scene could change to a snowy scene in the following frames.

In a concurrent work, incorporate optical flow to add motion cycle consistency and motion translation constraints.

However, their Motion-guided CycleGAN still suffers from the same two limitations as the RecycleGAN method.

In this paper, we propose UVIT, a novel method for unsupervised video-to-video translation.

We assume that a temporally consistent video sequence should simultaneously be: 1) long-term style consistent and 2) short-term content consistent.

Style consistency requires the whole video sequence to have the same style, it ensures the video frames to be overall realistic; while the content consistency refers to the appearance continuity of contents in adjacent video frames and ensures the video frames to be dynamically vivid.

Compared with previous methods which mainly focused on imposing short-term consistency between frames, we have considered in addition the long-term consistency issue which is crucial to generate visually realistic video sequences.

Figure 1: Overview of our proposed UVIT model: given an input video sequence, we first decompose it to the content by Content Encoder and the style by Style Encoder.

Then the content is processed by special RNN units-TrajGRUs (Shi et al., 2017) to get the content used for translation and interpolation recurrently.

Finally, the translation content and the interpolation content are decoded to the translated video and the interpolated video together with the style latent variable.

We depict here the video translation loss (orange), the cycle consistency loss (violet), the video interpolation loss (green) and the style encoder loss (blue).

To simultaneously impose style and content consistency, we adopt an encoder-decoder architecture as the video translator.

Given an input frame sequence, a content encoder and a style encoder firstly extract its content and style information, respectively.

Then, a bidirectional recurrent network propagates the inter-frame content information.

Updating this information with the single frame content information, we get the spatio-temporal content information.

At last, making use of the conditional instance normalization (Dumoulin et al. (2016) , Perez et al. (2018) ), the decoder takes the style information as the condition and utilizes the spatio-temporal content information to generate the translation result.

An illustration of the proposed architecture can be found in figure 1 .

By applying the same style code to decode the content feature for a specific translated video, we can produce a long-term consistent video sequence, while the recurrent network helps us combine multi-frame content information to achieve content consistent outputs.

The conditional decoder also provides us with a good interface to achieve modality flexible video translation.

Besides using the style dependent content decoder and bidirectional RNNs to ensure long-term and short-term consistency, another advantage of the proposed method lies in our training strategy.

Due to our flexible Encoder-RNN-Decoder architecture, the proposed translator can benefit from the highly structured video data and being trained in a self-supervised manner.

Concretely, by removing content information from frame t and using posterior style information, we use our Encoder-RNNDecoder translator to solve the video interpolation task, which can be trained by video sequences in each domain in a supervised manner.

In the RecycleGAN method, Bansal et al. (2018) proposed to train video predictors and plugged them into the GAN losses to impose spatio-temporal constraints.

They utilize the structured video data in an indirect way: using video predictor trained in a supervised way to provide spatio-temporal loss for training video translator.

In contrast, we use the temporal information within the video data itself, all the components, i.e. Encoders, RNNs and Decoders, can be directly trained with the proposed video interpolation loss.

The processing pipelines of using our Encoder-RNN-Decoder architecture for the video interpolation and translation tasks can be found in figure 2, more details of our video interpolation loss can be found in section 2.

The main contributions of our paper are summarized as follows:

1. a novel Encoder-RNN-Decoder framework which decomposes content and style for temporally consistent and modality flexible unsupervised video-to-video translation; 2.

a novel video interpolation loss that captures the temporal information within the sequence to train translator components in a self-supervised manner; 3.

extensive experiments showing the superiority of our model at both video and image level.

Let A be the video domain A, a 1:T = {a 1 , a 2 , ..., a T } be a sequence of video frames in A, let B be the video domain B, b 1:T = {b 1 , b 2 , ..., b T } be a sequence of video frames in B. For example, they can be sequences of semantic segmentation labels or scene images.

Our general goal of unsupervised video-to-video translation is to train a translator to convert videos between domain A and domain B with many-to-many mappings, so that the distribution of the translated video would be close to that of the real target domain video.

More concretely, to generate the style consistent video sequence, we assume each video frame has a style latent variable z. Let z a ∈ Z A and z b ∈ Z B be the style latent variables in domain A and B, respectively.

Our target is to align the conditional distribution of translated videos and target domain videos, i.e. P (b

The style information can be drawn from the prior or encoded from the style encoder in an example-based way.

In addition, taking the prior subset information (rain, snow, day, night, etc.) as label and incorporating that into the style code, we can also achieve deterministic control for the style of the output.

In this work, we assume a shared content space such that corresponding frames in two domains are mapped to the same latent content code just like UNIT (Liu et al., 2017) .

To achieve the goal of unsupervised video-to-video translation, we propose an Encoder-RNN-Decoder translator which contains the following components:

• Two content encoders CE A and CE B , which extract the frame-wise content information in the common spatial content space (e.g., CE A (a t ) = l t ).

• Two style encoders SE A and SE B , which encode video frames to the respective style domains (e.g., SE A (a 1:

is the posterior style latent variable.

In practice, we usually take the first frame to conduct style encoding(SE A (a 1 ) = z post a ).

• Two Trajectory Gated Recurrent Units (TrajGRUs) (Shi et al., 2017) T rajGRU f orw and T rajGRU back , which propagate the inter-frame content information in the forward and the backward direction to form the forward l f orw t and backward l back t content recurrently.

• One Merge Module M erge, which adaptively combine l f orw t and l back t

.

Without the l t from the current frame, it gets the interpolation content l interp t .

Using l t to update the l interp t , it gets the translation content l trans t .

• Two conditional content decoders CD A and CD B , which take the spatio-temporal content information and the style code to generate the output frame.

It can produce the interpolation frame (e.g., CD A (l is the prior style latent variable of domain A drawn from the prior distribution.

Combining the above components, we achieve two conditional video translation mappings:

In order to achieve the style-consistent translation result, we let all the frames in a video sequence to share the same style code z a (z b ).

Besides imposing long-term style consistency, another benefit of the conditional generator is modality flexible translation.

By assigning partial dimension of the style code to encode subset labels in the training phase, we are able to control the subset style of the translated video in a deterministic way.

As we propose to use the video interpolation loss to train the translator components in a selfsupervised manner, here we also define the video interpolation mappings:

Though the interpolation mapping is conducted within each domain, the interpolation and translation mappings use exactly the same building blocks.

An illustration of the translation and interpolation mappings are provided in figure 2. ) with the content (l t ) from the current frame (a t ).

Video translation loss.

The translated video frames should be similar to the real samples in the target domain.

Both the image-level discriminator and the video-level discriminator are added to ensure the image-level quality and the video-level quality.

Here we adopt relativistic LSGAN loss (Jolicoeur-Martineau (2018), Mao et al. (2017) ).

Such loss for domain B can be listed as: ) is defined in the same way.

Video interpolation loss.

The interpolated video frames should be close to the ground truth frames.

At the same time, they should be realistic compared to other frames in the domain.

This loss term (in domain A) is as follows:

Here, because of the characteristic of bidirectional TrajGRUs, only frames from time 2 to T − 1 are taken to compute the video interpolation loss.

a 2:T −1 are the real frames in domain A, a ) is defined in the same way.

Cycle consistency loss.

This loss is added to ensure semantic consistency.

This loss term (in domain A) is defined as: is defined in the same way.

Style encoder loss.

To train the style encoder, the style reconstruction loss and style adversarial loss are defined as follows:

Here, z

Our objective for the Generator:

Here G are the generator modules, which consist of CE A , CE B , SE A , SE B , T rajGRU f orw , T rajGRU back , M erge, CD A and CD B .

Our objective for the Discriminator:

Here, D are discriminator modules, which consist of

We aim to solve the optimization problem:

Implementation details: Our model is trained with 6 frames per batch, with a resolution of 128 × 128.

This enables us to train our model with a single Titan Xp GPU.

During test time, we follow the experimental setting of Wang et al. (2018a) and load video clips with 30 frames.

These 30 frames are divided into 7 smaller sequences of 6 frames with overlap.

They all share the same style code to be style consistent.

Please note that our model can be easily extended to process video sequences with any lengths.

Details of the network architecture are attached in Appendix A.2.

We use the Viper dataset (Richter et al., 2017) .

Viper has semantic label videos and scene image videos.

There are 5 subsets for the scene videos: day, sunset, rain, snow and night.

The large diversity of scene scenarios makes this dataset a very challenging testing bed for the unsupervised V2V task.

We quantitatively evaluate translation performance by different methods on the imageto-label and the label-to-image mapping tasks.

We further conduct the translation between different subsets of the scene videos for qualitative analysis.

Before comparing the proposed UVIT with state-of-the-art approaches, we first conduct ablation study experiments to emphasize our contributions.

We provide experimental results to show the effect of style-conditioned translation and the effectiveness of the proposed video interpolation loss.

UVIT utilizes an Encoder-RNN-Decoder architecture and adopts a conditional decoder to ensure the generated video sequence to be style consistent.

The conditional decoder also provides us with a good interface to achieve modality flexible video translation.

In our implementation, we use a 21-dimensional vector as the style latent variable to encode the subset label as well as the stochastic part.

By changing the subset label, we are able to control the subset style of the generated video in a deterministic way.

Meanwhile, by changing the stochastic part, we can generate various video sequences in a stochastic way.

In figure 3 , we use the same semantic label sequence to generate video sequences with different sub-domain labels.

In figure 4 , inducing the same subset label -sunset but changing the stochastic part of the style latent variable, we present different sunset videos generated from the same semantic label sequence.

Figure 3 and figure 4 clearly show the effectiveness of the proposed conditional video translation mechanism.

Please note that the training of our method does not rely on the subset labels, we incorporate subset labels for the purpose of a deterministic controllable translation.

Without the subset labels, we can still generate multimodal style consistent results in a stochastic way.

Video Interpolation Loss:

In this part, we provide ablation experiments to show the effectiveness of the proposed video interpolation loss.

We conduct ablation studies on both the image-to-label and the label-to-image tasks.

Besides comparing UVIT with and without video interpolation loss, we also train UVIT with image reconstruction loss (Huang et al., 2018) , which only uses image-level information to train encoder-decoder architectures in a self-supervised manner.

We denote UVIT trained without video interpolation loss as "UVIT w/o vi-loss" and UVIT trained without video interpolation loss but with image reconstruction loss as "UVIT w/o vi w ir loss".

We follow the experimental setting of RecycleGAN (Bansal et al., 2018) and use semantic segmentation metrics to evaluate the image-to-label results quantitatively.

We report the Mean Intersection over Union (mIoU), Average Class Accuracy (AC) and Pixel Accuracy (PA) achieved by different methods in Table 1 .

For the label-to-image task, we use the Fréchet Inception Distance (FID) (Heusel et al., 2017) to evaluate the feature distribution distance between translated videos and ground truth videos.

The same as vid2vid (Wang et al., 2018a) , we use the pretrained I3D (Carreira & Zisserman, 2017) model to extract features from videos.

We use the semantic labels from the respective sub-domains to generate videos and evaluate the FID score on all the subsets of the Viper dataset.

The FID score achieved by the proposed UVIT and its ablations can be found in Table 2 .

On both the image-to-label and label-to-image tasks, the proposed video interpolation loss plays a crucial role for UVIT to achieve good translation results.

In addition, compared with the image-level image reconstruction loss, video interpolation loss could effectively incorporate temporal information, and delivers better video-to-video translation results.

Table 2 : Ablation study: Label-to-image FID.

More details can be found in Section 3.1.

Image-to-label mapping: We use exactly the same setting as our ablation study to compare UVIT with RecycleGAN in the image-to-label mapping task.

The mIoU, AC and PA value by the proposed UVIT and competing methods are listed in Table 3 .

The results clearly validate the advantage of our method over the competing approaches in terms of preserving semantic information.

Table 5 : Label-to-image: Human Preference Score.

Vid2vid is a supervised method and the other methods are unsupervised approaches, more details can be found in Section 3.2.

Label-to-image mapping: In this setting, we compare the quality of the translated video sequence by different methods.

We firstly report the FID score (Heusel et al., 2017) on all the sub-domains of the Viper dataset in the same setting as our ablation experiments.

As the original RecycleGAN method can not produce long-term style consistent video sequences, we also report the results achieved by our improved version of the RecycleGAN.

Concretely, we develop a conditional version which formally controls the style of generated video sequences in a similar way as our UVIT model, and denote the conditional version as improved RecycleGAN.

The FID results by different methods are shown in Table 4 .

The proposed UVIT achieves better FID on all the 5 sub-domains.

To thoroughly evaluate the visual quality of the video translation results, we conduct subjective evaluation on the Amazon Mechanical Turk (AMT) platform.

We compare the proposed UVIT with 3DCycleGAN and RecycleGAN.

The video-level and image-level human preference scores (HPS) are reported in Table 5 .

For reference, we also compare the video-level quality between UVIT and the supervised vid2vid model (Wang et al., 2018a ).

Meanwhile, image-level quality comparison between UVIT and CycleGAN (the image translation baseline) is also included.

demonstrates the effectiveness of our proposed UVIT model.

In the video-level comparison, our unsupervised UVIT model outperforms the competing unsupervised RecycleGAN and 3DCycle-GAN by a large margin, and achieves comparable results with the supervised benchmark.

In the image-level comparison, UVIT achieves better HPS than both the V2V competing approaches and the image-to-image baseline.

A qualitative example in figure 5 also shows that UVIT model produces a more content consistent video sequence.

It could not be achieved by simply introducing the style control without the specialized network structure to record the inter-frame information.

Besides translating video sequences between image and label domains, we also train models to translate video sequences between different image subsets and different video datasets.

In figure 6 , we provide visual examples of video translation from Sunset to Day scenes in the Viper dataset.

More results of translation between Viper and Cityscapes (Cordts et al., 2016) datasets can be found in our Appendix.

In this paper, we have proposed UVIT, a novel method for unsupervised video-to-video translation.

A novel Encoder-RNN-Decoder architecture has been proposed to decompose style and content in the video for temporally consistent and modality flexible video-to-video translation.

In addition, we have designed a video interpolation loss which utilizes highly structured video data to train our translators in a self-supervised manner.

Extensive experiments have been conducted to show the effectiveness of the proposed UVIT model.

Without using any paired training data, the proposed UVIT model is capable of producing excellent multimodal video translation results, which are image-level realistic, semantic information preserving and video-level consistent.

Image level discriminator loss.

This loss term (for D img A in domain A) is defined as follows:

for domain B is defined in the same way.

Video level discriminator loss.

This loss term (for D vid A in domain A) is defined as follows:

for domain B is defined in the same way.

Style latent variable discriminator loss.

This loss term (for D Z A in style domain A) is defined as follows:

for style domain B is defined in the same way.

The Trajectory Gated Recurrent Units (TrajGRUs) (Shi et al., 2017) can actively learn the locationvariant structure in the video data.

It uses the input and hidden state to generate the local neighborhood set for each location at each time, thus warping the previous state to compensate for the motion information.

We take two TrajGRUs to propagate the inter-frame information in both directions in the shared content space.

With the video being in a resolution of 128 × 128, we use a single Titan Xp GPU to train our network for 3 to 4 days to get a mature model.

Due to the GPU memory limitation, the batch size is set to be one.

Currently, the frame per clip is 6.

Feeding more frames per clip may improve the ability of our model to capture the content dependency in a longer range.

However, it requires more GPU memory.

The same requirement holds if we want to achieve a higher resolution and display more details.

An example of style inconsistency of RecyceGAN is shown in figure 7 .

A qualitative example of the mapping between images and labels can be found at figure 8, which shows that our UVIT model can output semantic preserving and consistent segmentation labels.

More results on the label-to-image mapping comparison of UVIT and Improved RecycleGAN are plotted in figure 9 and figure 10.

More results on label sequences to image sequences with multimodality are plotted in figure 11 .

The Cityscapes (Cordts et al. (2016) ) dataset has real-world street scene videos.

As a supplement, we conduct qualitative analysis on the translation between scene videos of Cityscapes and Viper dataset.

The result is organized in figure 12.

@highlight

A temporally consistent and modality flexible unsupervised video-to-video translation framework trained in a self-supervised manner.