Developing conditional generative models for text-to-video synthesis is an extremely challenging yet an important topic of research in machine learning.

In this work, we address this problem by introducing Text-Filter conditioning Generative Adversarial Network (TFGAN), a GAN model with novel conditioning scheme that aids improving the text-video associations.

With a combination of this conditioning scheme and a deep GAN architecture, TFGAN generates photo-realistic videos from text on very challenging real-world video datasets.

In addition, we construct a benchmark synthetic dataset of moving shapes to systematically evaluate our conditioning scheme.

Extensive experiments demonstrate that TFGAN significantly outperforms the existing approaches, and can also generate videos of novel categories not seen during training.

Generative models have gained much interest in the research community over the last few years as they provide a promise for unsupervised representation learning.

Generative Adversarial Networks (GANs) BID0 have been one of the most successful generative models till date.

Following its introduction in 2014, significant progress has been made towards improving the stability, quality and the diversity of the generated images BID11 BID2 .

While GANs have been successful in the image domain, recent efforts have extended it to other modalities such as texts BID14 , graphs (Wang et al., 2018b) , etc.

In this work, we focus on the less studied domain of videos.

Generating videos are much harder than images because the additional temporal dimension makes generated data extremely high dimensional, and the generated sequences must be both photo-realistically diverse and temporally consistent.

We tackle the problem of text-conditioned video synthesis where the input is a text description and the goal is to synthesize a video corresponding to the input text.

This problem has many potential applications, some of which include producing multimedia special effects, generating synthetic data for model-based Reinforcement Learning systems and domain adaptation, etc.

Two recent works that address the problem of text-conditioned video generation include BID6 and BID9 .

Both these methods are variants of conditional GAN model applied to the video data.

In spite of some successes, they have the following limitations: (1) They employ 3D transposed convolution layers in the generator network, which constrains them to only produce fixed-length videos.

(2) Their models are trained on low-resolution videos -results are shown only at a 64×64 resolution.

(3) Text conditioning is performed using a simple concatenation of video and text features in the discriminator: Such a conditioning scheme may perform well on certain datasets, but has difficulty in capturing rich video-text variations.

In this work, we aim to address all the concerns above.

First, to model videos of varying length, we use a recurrent neural network in the latent space and employ a shared frame generator network similar to BID12 .

Second, we present a model for generating high-resolution videos by using a Resnet-style architecture in the generator and the discriminator network.

Third, we propose a new multi-scale text-conditioning scheme based on convolutional filter generation to strengthen the associations between the conditioned text and the generated video.

We call our model Text-Filter conditioning GAN (TFGAN).

Finally, we construct a benchmark synthetic moving shapes dataset to extensively evaluate the effectiveness of the new conditioning scheme we proposed.

Text representations are extracted from the input text and passed to a GRU network to get a trajectory in the latent space.

These latent vectors are fed to a shared frame generator to produce the video sequence.

The generated videos are then passed to conditional discriminator networks.

The box highlighted in red is where the conditioning is performed and is expanded in FIG1 In summary, our contributions in this work are as follows: (i) A new conditional GAN with an effective multi-scale text-conditioning scheme based on convolutional filter generation is proposed; (ii) A benchmark synthetic dataset for studying text conditioning in video generation is presented; (iii) Photo-realistic video synthesis is achieved using a deeper generator-discriminator architecture.

Two popular approaches to generative modeling include GANs BID0 and Variational Autoencoders(VAEs) BID4 .

GANs are formulated as a 2− player minimax game between a generator and a discriminator network, while VAEs are based on variational inference where a variational lower bound of observed data log-likelihood is optimized.

Among the two approaches, GANs have generated significant interest as they have been shown to produce images of high sample fidelity and diversity BID2 .A variant of GAN models are conditional GANs where the generator network is conditioned on input variables of interest.

Such a conditioning input can be labels BID8 , attributes BID10 ), text (Zhang et al., 2017 Xu et al., 2018) or even images (Zhu et al., 2017) .

We focus on text conditioning since it is relevant to this work.

One of the first works to perform textconditioned image synthesis is BID10 .

Their method was only shown to synthesize low-resolution images.

To improve the resolution, BID3 proposed stacking multiple GAN architectures, each producing images of increasing resolution.

While the above two methods perform conditioning using the global text representation, Xu et al. (2018) adopts an attention mechanism to focus on fine-grained word-level representations to enable improved conditioning.

While image generation is a well studied problem, there has been very little progress in the domain of video generation.

Vondrick et al. (2016) proposed a GAN architecture based on 3D convolutions to generate video sequences, but it can only generate fixed-length videos.

BID12 proposed using a recurrent neural network in the latent space to model videos of varying lengths.

While these models are not designed to handle conditional video generation, BID6 and BID9 perform text-conditioned video synthesis by using the sentence embedding as a conditional input.

However, both of these conditional generative models are based on 3D convolutions, they can only produce fixed-length low-resolution videos.

In this work, we address this issue by developing an architecture capable of producing high-resolution videos of varying length.

We first provide a formal description of the problem being address.

We are given access to n data points DISPLAYFORM0 sampled from an underlying joint distribution p(v, t) in the video-sentence space.

Here, each v i ∈ R T ×C×W ×H is a video clip and t i is a sentence description.

We are interested in learning a model capable of sampling from the unknown conditional distribution p(v|t).

Similar to conditional GANs, we formulate the problem as learning a transformation function G(z, t) from a known prior distribution P z (z) and the conditional input variable t to the unknown conditional distribution p(v|t).

The function G is optimized using an adversarial training procedure.

The framework of our proposed model is shown in FIG0 .

The text description t is passed to a text encoder T to get a frame-level representation t f and a video-level representation t v .

Here, t f is a representation common to all frames, and contains frame-level information like background, objects, etc.

from the text.

The video representation t v extracts the temporal information such as actions, object motion, etc.

The text representation along with a sequence of noise vectors DISPLAYFORM0 are passed to a recurrent neural network to produce a trajectory in the latent space.

Here, l denotes the number of frames in the video sequence.

These sequence of latent vectors are then passed to a shared frame generator model G to produce the video sequence.

The generated video is then fed to two discriminator models -D F and D V .

D F is a frame-level discriminator that classifies if the individual frames in the video are real/fake, whereas the video discriminator D F is trained to classify the entire video as real/fake.

The discriminator models D F and D V also take the text encoding t f and t v respectively as inputs so as to enforce text-conditioning.

To build strong conditional models, it becomes important to learn good video-text associations in the GAN model.

A standard technique is to sample negative (v, t) pairs (wrong associations) and train it as fake class, while the correct (v, t) pairs are trained as real class in the discriminator network.

Since the generator is updated using the gradients from the discriminator network, it becomes important to effectively fuse the video and text representations in the discriminator so as to make the generator condition well on the text.

Previous methods BID6 BID9 use a simple concatenation of text and video features as the feature fusion strategy.

We found that this simple strategy produces poor conditioned models in datasets where there are rich text-video variations (refer to Section.

4 for more details).Our proposed model Text-Filter conditioning GAN (TFGAN) focuses on improving text conditioning.

In TFGAN, we employ a scheme based on generating convolutional filters from the text features.

This scheme, which we call Text-Filter conditioning, is shown in FIG1 .

Let us first divide the discriminator network D (which can be DISPLAYFORM0 .

These sub-networks can be as small as a single layer, or can be a cascade of multiple layers.

Let d (i) denote the output of the i th sub-network of the discriminator.

From the text features, we generate a set of convolution filters DISPLAYFORM1 .

Each filter f i is now convolved with the discriminator response d (i) , and this convolved output is passed through additional convolutions after which they are pooled to get a single video-text representation.

We use this pooled feature vector to classify the (v, t) pairs as real or fake.

Because the generated convolutional filters {f i } are applied to discriminator sub-network outputs {d (i) } from different layers, the resulting text conditioning effectively imposes semantic constraints extracted from input texts to the generated individual frames and video clips at different feature abstraction levels.

The discriminator model D and the generator model G are trained using an adversarial game as done in the standard conditional GANs.

However, since we employ deep Resnet-style architectures for our G − D networks, it was important to stabilize the GAN training.

We use the regularizer as proposed in BID7 where the norm of the discriminator gradients are penalized.

With this regularizer, our optimization objective can be expressed as .

These responses are then passed to additional convolutional layers before they are pooled to get a single video-text representation that is used to classify the input (v, t) pair as real/fake.

DISPLAYFORM0 DISPLAYFORM1 The text encoder T is optimized as follows DISPLAYFORM2 In the above set of equations, p data,real denotes the real data distribution with correct video-text correspondences, whereas p data,f ake refers to the distribution with incorrect video-text correspondences.

Note that we have two discriminator networks -D F , D V in our models, and the above equations have to be repeated for both models.

Eq.1-4 are optimized by alternating between the minimization and maximization problems as done in standard GAN.

Please refer to Appendix A for the detailed training algorithm.

This section discusses the experimental validation of our TFGAN model.

We first describe a benchmark synthetic dataset we created for the task of text-to-video generation, and use it to better analyze our system.

Then, we show results on a challenging real-world video dataset -the Kinetics human action video dataset BID3 .

Finally, we show how our method can be extended to the task of text-to-image synthesis and show results on the CUB birds dataset (Welinder et al., 2010) .

To better understand the task of text-to-video synthesis, we created a dataset of moving shapes where a shape moves along a trajectory as described by the corresponding text description.

This synthetic dataset has 5 control parameters: shape type, size, color, motion type and motion direction.

Of these, the first three parameters are frame-level attributes while the last two are temporal attributes.

all parameters results in 360 unique parameter configurations.

Some samples from this dataset are shown in FIG2 .

We call this dataset Shapes-v1 dataset.

The above dataset contains videos with static background (all black).

While this is a reasonable assumption to make, it is hardly true in practice as many videos have dynamic backgrounds.

So, we create a second dataset called Shapes-v2 dataset which is a version of Moving Shapes dataset with dynamic backgrounds.

To generate the background, we choose images from the Kylberg Texture Dataset BID5 and sample a sequence of patches corresponding to a random trajectory.

Each patch in this sequence forms the background of an individual frame in the video.

These background textures are blended with the moving object resulting in videos as shown in FIG2 .

This dataset is much more challenging than the Shapes-v1 dataset as the generative models should learn to ground the text description to the moving object but not to the randomly moving background.

Both these datasets were created with videos containing 16 frames at a 64 × 64 frame resolution.

Figure 4 : Exploratory experiments on Shapes-v1 dataset.

The image on the top shows the long sequence experiment where we generate 32-length sequence from a model trained for 16 frames.

The top row of this video are the first 16 frames and the bottom row corresponds to the next 16.

The images on the bottom illustrate the interpolation experiments where we generate a video corresponding to a smooth transition between two input sentences.

An important advantage of creating the synthetic dataset is that it provides a framework for quantitative evaluation of the text-conditioning.

First, we train five attribute classifiers (shape, size, color, motion and direction classifiers) on the real data using the ground truth attributes (we have access to ground-truth attributes as we stored them while creating the dataset).

We then use these trained attribute classifiers to verify if the attributes of the generated videos correspond to those described by the input text in the test set.

For each text description in the test set, we generate the video using our TFGAN model and measure the attribute prediction accuracy.

Higher this accuracy, better conditioned is our GAN model.

We experiment with the following models: (1) FeatCat: a conditional GAN model trained using simple text-video feature concatenation in the discriminator network (2) FeatCat branchingD: conditional GAN model with branching D structure where responses at intermediate layers of D are pooled, and this pooled feature is concatenated with text embedding.

This model is essentially TF-GAN but without performing the convolutions from text-filters, and (3) TFGAN with Text-Filter conditioning.

The architecture and hyper-parameter details are described in the Appendix C. Some sample generations of our Text-Filter conditioned GAN model is shown in FIG2 and 3b TAB2 reports the quantitative evaluation of the three conditional GAN models on Shapes dataset.

We observe that TFGAN with text-filter conditioning achieves the best performance among the three models on both the datasets.

An important observation to note is that using a branching architecture in the discriminator network alone does not improve the text conditioning.

This shows that the effectiveness of our method comes not from the branching architecture, but in how text conditioning is applied (using convolutions) at multiple layers of the discriminator network.

In this section, we report some exploratory experiments we perform on the Shapes dataset.

Sentence interpolation In this experiment, we depict conditional interpolation whereby frames in a video transition corresponding to the interpolation between two sentence descriptions.

Let S 1 and S 2 denote the two sentences that are interpolated, and (t S1 f , t DISPLAYFORM0 Instead of using a fixed text representation (t f , t v ) as conditioning argument to all the frames in the Generator network, we use (t i f , t i v ) as input to the frame i.

The resulting interpolated videos are shown in Fig. 4 .

We observe that we are able to obtain smooth transitions.

When interpolating between the blue square and the red square, we obtain some intermediate frames with pink shade.

Interestingly, none of the samples in the dataset contain pink color.

In the second figure, we observe a smooth decrease in the object size while the object continues to move in the specified trajectory.

Generating novel categories To characterize if the model has learned to generalize and not naively memorize the dataset, this experiment aims to study the ability of our TFGAN model to produce videos not seen during training.

Of the 360 unique parameter configurations in the Shapes dataset, we randomly hold out n configurations from the training set.

After training the model on this training set, we feed the text descriptions from the held-out n configurations and measure the attribute classification accuracy in this set.

In this experiment, we choose n = 20.

The results are reported in TAB3 .

We observe that our model achieves good accuracy and this illustrates the ability of our method to generalize.

Long Sequence Generation One of the benefits of using a RNN-based GAN model is that it allows us to model variable-length video sequences.

To demonstrate this, we perform an experiment where we train our TFGAN model on 16-length video sequences and generate 32-length sequences.

This can be performed easily as we could potentially generate a latent trajectory of any length using the RNN model in the latent space, and the videos are generated using a shared generated acting on this latent trajectory.

Fig. 4 shows the output of one such 32-length sequence generated.

We observe that the model is able to clearly perform the zig-zag motion beyond 16 frames.

To demonstrate the practical relevance of our approach, we perform experiments on real-world video datasets.

We use the dataset proposed in BID6 Figure 6 : Sample generations of Text2img synthesis from our model trained on CUB-birds dataset contains videos of human actions, and was curated from YouTube and Kinetics human action video dataset BID3 .

The dataset contains the following action classes -biking in snow, playing hockey, jogging, playing soccer ball, playing football, kite surfing, playing golf, swimming, sailing and water skiing.

This is an extremely challenging dataset for the task of video generation due to the following reasons: (1) videos are extremely diverse, and there are a lot of variations within the video, and (2) some videos have low-resolution and poor-quality video frames.

Some sample videos from the dataset are shown in The results of training our TFGAN model on the Kinetics dataset are shown in FIG3 .

We observe that our model is able to produce videos of much higher quality than the comparison method BID6 .

We are able to generate fine motions like golf swing, while BID6 produces a blobby region.

Also, we train the model to produce 128 × 128 resolution videos, while the method in BID6 was trained only on 64 × 64 videos.

As done in BID6 , we report a simplified version of inception score whereby a video classification model is trained on the real data, and the accuracy on generated data is reported.

We report the performance on the following five categories as done in BID6 : kite surfing, playing golf, biking in snow, sailing, swimming and water skiing.

As can be seen from Table.

3, our methods achieves significantly higher accuracy than the method in BID6 .

In-set refers to the performance obtained on the test set of real videos, thus serves as an upper bound.

We report additional results, architecture and hyper-parameter details in Appendix C. Text-to-image generation is a relatively easier problem than text-to-video generation due to the absence of temporal constraints.

Although the focus of this paper is on text-to-video synthesis, our framework is flexible and can be trivially extended to the problem of text-to-image synthesis.

This can be accomplished by removing the videolevel discriminator D V and the RNN network in the latent space.

We train our GAN model with Text-Filter conditioning on the CUB-Birds dataset Welinder et al. FORMULA4 , a benchmark dataset for text-to-image generation.

Some of the samples from the generated images are shown in Fig. 6 .

We observe that our model is able to produce photo-realistic images.

We also report Inception score as a quantitative metric.

As can be seen from Table.

4, our method achieves higher inception scores than the comparison methods.

In this work, we address the problem of generating videos conditioned on text.

We propose a novel text-conditioning framework whereby conditioning is performed using convolution operations on image feature maps with filters generated from text.

To better understand the text conditioning, we construct a synthetic dataset and show that our conditioning scheme achieves superior performance compared to other techniques.

Finally, by using deeper architectures in the discriminator and generator networks, we generate photo-realistic videos on the challenging Kinetics dataset.

Sample N b real samples with incorrect video-text correspondence DISPLAYFORM0 Update G: DISPLAYFORM1 Update T : DISPLAYFORM2

Both Shapes-v1 and Shapes-v2 dataset contain videos with objects moving along a specific trajectory.

There are 5 control parameters -shape, size and color of object, type and direction of motion.

TAB9 lists the possible values each parameter can take.

The (shape, color, size) tuple describes the structure of the object, while (motion type, direction) tuple dictates how the object moves in the video.

For straight line and zig-zag motion, the object could move in north, south, west and east direction, while for diagonal motion, the possible directions include north-west, north-east, south-west and south-east.

The zig-zag motion was generated using a sinusoidal function.

We first define some basic architecture blocks:• ResnetBlock(x, y): x + Conv2D(x → y, kernel=3, str=2, pad=1) if x == y, else Shortcut(x → y) + Conv(x− >y, 3 × 3, str=2, pad=1).

Here, Shortcut(x → y) is the 1 × 1 convolution that is maps from x filters to y filters.

This is a simplified resnet block that was proposed in BID1 .•

ResnetBlock3D(x, y): x + Conv3D(x → y, kernel=3, str=2, pad=1) if x == y, else Shortcut(x → y) + Conv3D(x− >y, 3 × 3, str=2, pad=1)• We first take the output of D (0) and apply 1D convolution to bring the number of filters to 8.

So, if the output of D (0) was a n×k×w map, this transformation will bring it down to 8×k×w.

Let the text embedding (which is obtained by passing through the text encoder) be a d− dimensional vector.

We first apply a F C(d → 5.5.8.8) to this embedding and reshape it to 8×8×5×5 filter.

This filter is then convolved with the transfomed outputs of D 0 .

These resulting convolved feature maps are passed through two conv-2D layers with Avg-Pool and then reshaped to 128 dimensional vector.

The same procedure is done for D(1) .

For D (2) , we just take the output and pass it through a fully connected layer to get 128 dimensional vector.

These three 128 dimensional vectors are concatenated to get one resulting vector which classifies if the input is real or fake.

The exact sample procedure is repeated for Video discriminators, only difference being the size of Text-Filters: 3D filters of size 3 × 5 × 5 is used instead of 5 × 5 filter .

So, the FC applied on text embedding will be F C(d → 5.5.3.8.8)

For Shapes-v1 and Shapes-v2 datasets, we used an architecture based on ResnetBlock as shown in Tables.

6, 7 and 8.

For the text encoder, we first obtained the GloVE embeddings of individual words, then applied a 1D-CNN based network with the following network architecture: Conv1D(512, kernel=3) → ReLU → MaxPool(2) → Conv1D(512, kernel=3) → ReLU → MaxPool(2) → Conv1D (256) 1D-CNN was sufficient in this case as most sentences were of rougly similar lenghts.

The inputs were zero padded to making every sentence have the dimension.

We tried using a LSTM model and it gave similar performance as 1D CNN.

<|TLDR|>

@highlight

An effective text-conditioning GAN framework for generating videos from text

@highlight

This paper presents a GAN-based method for video generation conditioned on text description, with a new conditioning method that generates convolution filters from the encoded text, and uses them for a convolution in the discriminator.

@highlight

This paper proposes conditional GAN models for text-to-video synthesis: developing text-feature-conditioned CNN filters and constructing moving-shape dataset with improved performance on video/image generation.