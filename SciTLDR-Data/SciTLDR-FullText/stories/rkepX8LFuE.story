In this paper we study image captioning as a conditional GAN training, proposing both a context-aware LSTM captioner and co-attentive discriminator, which enforces semantic alignment between images and captions.

We investigate the viability of two discrete GAN training methods: Self-critical Sequence Training (SCST) and Gumbel Straight-Through (ST) and demonstrate that SCST shows more stable gradient behavior and improved results over Gumbel ST.

Significant progress has been made on the task of generating image descriptions using neural image captioning.

Early systems were traditionally trained using cross-entropy (CE) loss minimization BID6 BID16 .

Later, reinforcement learning techniques BID13 BID9 based on policy gradient methods were introduced to directly optimize metrics such as CIDEr or SPICE BID0 .

Along a similar idea, BID14 introduced Self-critical Sequence Training (SCST), a light-weight variant of REINFORCE, which produced state of the art image captioning results using CIDEr as an optimization metric.

To address the problem of sentence diversity and naturalness, image captioning has been explored in the framework of GANs.

However, due to the discrete nature of text generation, GAN training remains challenging and has been generally tackled either with reinforcement learning techniques BID4 BID12 BID2 or by using Gumbel softmax relaxation BID5 , as in BID15 BID7 .Despite impressive advances, image captioning is far from being a solved task.

It remains a challenge to satisfactorily bridge the semantic gap between image and captions to produce diverse, creative, and "human-like" captions.

Although applying GANs to image captioning for promoting human-like captions is a very promising direction, the discrete nature of the text generation process makes it challenging to train such systems.

The recent work of BID1 showed that the task of text generation for current discrete GAN models is difficult, often producing unsatisfactory results, and requires therefore new approaches and methods.

In this paper, we propose a novel GAN-based framework for image captioning that enables better language composition, more accurate compositional alignment of image and text, and light-weight efficient training of discrete sequence GAN based on SCST.

In this Section we present our novel captioner and discriminator models.

We employ SCST for discrete GAN optimization and compare it to the approach based on the Gumbel relaxation trick.

Context Aware Captioner G θ .

For caption generation we use an LSTM with visual attention BID16 BID14 together with a visual sentinel BID11 to give the LSTM a choice to attend to visual or textual cues.

While BID11 feeds at each step t only an average image feature, we feed a mixture of image and visual sentinel features from t−1 to make the LSTM aware of the last attentional context (called Context Aware attention), as seen in Fig. 1 .

This simple modification gives significant gains, as the captioner is now aware of the visual information used in the past.

As reported in Tab.

1, a captioner with an adaptive visual sentinel BID11 Co-attention Pooling Discriminator D η .

Previous works jointly embed the modalities at the similarity computation level, referred to as Joint-Emb (e.g., BID2 ).

Instead, we propose to jointly embed image and caption in earlier stages using a co-attention model BID10 and compute similarity on the attentive pooled representation.

We call it a Co-attention discriminator, see Fig. 2 .

In Section 3 we compare D η with Joint-Emb of BID2 BID15 , where E I is the average spatial pooling of CNN features and E S the last state of LSTM.

DISPLAYFORM0 Figure 2: Proposed co-attention discriminator (Co-att) architecture.

By jointly embedding image and caption with a co-attention model, the discriminator has the ability to modulate the image features depending on the caption and vice versa.

DISPLAYFORM1 Figure 3: SCST Training of GAN-captioning.

Training D η .

Our discriminator D η is not only trained to distinguish real captions from fake (generated), but also to detect when images are coupled with random unrelated real sentences, thus forcing it to check sentence composition and semantic relationship between image and caption.

We solve the following optimization problem: DISPLAYFORM2 , where w is the real sentence, w s is sampled from generator G θ (fake caption), and w is real but random caption.

DISPLAYFORM3 The main difficulty is the discrete, non-differentiables nature of the problem.

We propose to solve this issue by adopting SCST BID14 and compare it to the Gumbel relaxation approach of BID5 .Training G θ using SCST.

SCST is a REINFORCE variant that uses the reward under the decoding algorithm as baseline.

In this work, the decoding algorithm is a "greedy max", selecting at each step the most probable word from arg max p θ (.|h t ).

For a given image, a single sample w s of the generator is used to estimate the full sequence reward, L I G (θ) = log(D(I, w s )) where w s ∼ p θ (.|I).

Using SCST, the gradient is estimated as follows: DISPLAYFORM4 whereŵ is obtained using greedy max (see Fig. 3 ).

Note that the baseline does not change the expectation of the gradient but reduces the variance of the estimate.

Also, observe that the GAN training can be regularized with any NLP metric r NLP (such as CIDEr) to enforce closeness of the generated captions to the provided ground truth on the n-gram level; the gradient then becomes: DISPLAYFORM5 There are two main advantages of SCST over other policy gradient methods used in the sequential GAN context: 1) The reward in SCST can be global at the sentence level and the training still succeeds.

In other policy gradient methods, e.g., BID2 ; BID9 , the reward needs to be defined at each word generation with the full sentence sampling, so that the discriminator needs to be evaluated T times (sentence length).

2) In BID2 BID9 BID4 , many Monte-Carlo rollouts are needed to reduce variance of gradients, requiring many forward-passes through the generator.

In contrast, due to a strong baseline, only a single sample estimate is enough in SCST.Training G θ using the Gumbel Trick.

An alternative way to deal with the discreteness of the generator is by using Gumbel re-parameterization BID5 .

Define the soft samples y j t , for t = 1, . . .

T (sentence length) and j = 1, . . .

K (vocabulary size) such that: y j t = Softmax 1 τ (logits θ (j|h t , I) + g j ) , where g j are samples from Gumbel distribution, τ is a temperature parameter.

We experiment with Gumbel Soft and Gumbel Straight-Through (Gumbel ST) approach, recently used in BID15 BID7 .For Gumbel soft, we use the soft samples y t as LSTM input w DISPLAYFORM6 where (w * 1:T ) is the ground truth caption corresponding to image I, and E I and E S are co-attention image and sentence embeddings (as defined earlier).

Feature matching enables us to incorporate more granular information from discriminator representations of the ground truth caption, similar to how SCST reward can be regularized with CIDEr.

Experimental Setup.

We evaluate our proposed method and the baselines on COCO dataset BID8 .

Each image is encoded by a resnet-101 BID3 , followed by a spatial adaptive max-pooling to ensure a fixed size of 14×14×2048.

An attention mask is produced over the 14×14 spatial locations, resulting in a spatially averaged 2048-dimension representation.

LSTM hidden state, image, word, and attention embedding dimensions are fixed to 512 for all models.

Before the GAN training, all the models are first pretrained with the cross entropy (CE) loss.

Experimental Results.

Tab.

2 presents results on COCO dataset for context-aware captioner, two discriminator architectures (ours Co-att, and baseline Joint-Emb) and all training algorithms (SCST, Gumbel ST, and Gumbel Soft).

For reference, we also include results for CE (trained only with cross entropy) and CIDEr-RL (pretrained with CE, followed by SCST to optimize CIDEr), as well as results from non-attentional models.

As expected, CIDEr-RL greatly improves the language metrics as compared to CE model (101.6 to 116.1 CIDEr), leading to a significant drop in the vocabulary coverage (from 9.2% to 5.1%).

On the other hand, the underperformance of GANs over CIDEr-RL in terms of CIDEr is also expected since GAN's objective is to make the sentences more descriptive and human-like, deviating from the vanilla ground truth captions.

The results also show the advantage of our Co-att architecture as compared to the Joint-Emb one, showing the importance of the early joint embedding of the image/caption pair for better similarity computation.

Regularizing GANs with CIDEr additionally SCST vs. Gumbel Our experiments also showed that SCST is a more stable approach for training discrete GAN nodels, achieving better results as compared to Gumbel relaxation approaches.

To demonstrate that our experiments fairly compared both approaches, in FIG2 we show training of different Gumbel methods, where we plot the discriminator scores across gradient updates.

As can be seen, at the end of training the generated sentences are scored around 0.5, random near 0.1 and real sentences above 0.7, indicating a properly trained discriminator and a healthy execution of all the Gumbel methods.

FIG3 also compares gradient behaviors during training for SCST and Gumbel, showing that SCST gradients have smaller average norm and variance across minibatches, confirming our conclusion.

In summary, we demonstrated that SCST training for discrete GAN is a promissing new approach that outperforms the Gumbel relaxation in terms of training stability and the overall performance.

Moreover, we showed that our context-aware attention gives larger gains as compared to the adaptive sentinel or the traditional visual attention.

Finally, our co-attention model for discriminator compares favorably against the joint embedding architecture.

@highlight

Image captioning as a conditional GAN training with novel architectures, also study two discrete GAN training methods. 

@highlight

An improved GAN model for image captioning that proposes a context-aware LSTM captioner, introduces a stronger co-attentive discriminator with better performance, and uses SCST for GAN training.