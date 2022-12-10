Verifying a person's identity based on their voice is a challenging, real-world problem in biometric security.

A crucial requirement of such speaker verification systems is to be domain robust.

Performance should not degrade even if speakers are talking in languages not seen during training.

To this end, we present a flexible and interpretable framework for learning domain invariant speaker embeddings using Generative Adversarial Networks.

We combine adversarial training with an angular margin loss function, which encourages the speaker embedding model to be discriminative by directly optimizing for cosine similarity between classes.

We are able to beat a strong baseline system using a cosine distance classifier and a simple score-averaging strategy.

Our results also show that models with adversarial adaptation perform significantly better than unadapted models.

In an attempt to better understand this behavior, we quantitatively measure the degree of invariance induced by our proposed methods using Maximum Mean Discrepancy and Frechet distances.

Our analysis shows that our proposed adversarial speaker embedding models significantly reduce the distance between source and target data distributions, while performing similarly on the former and better on the latter.

Generative Adversarial Networks (GAN).

We drawn inspiration from research in computer vision, 48 where GAN based unsupervised domain adaptation methods have been extremely successful BID0 21, 49 18, 19], and adapt these ideas for feature learning in a verification setting.

The basic idea is cast 50 domain adaptation/invariance as an adversarial game -generate features or embeddings such that a 51 discriminator cannot tell if they come from the source or target domain.

Unlike traditional GANs that 52 work in high-dimensional spaces (e.g. natural images,speech), domain adaptation GANs operate in 53 low-dimensional embedding space.

We extend our recent work [2, 4] and propose a novel objective 54 for updating the generator network.

We find that optimizing GAN models with this objective proves 55 to be unstable, and propose to stabilize it by augmenting the discriminator with an auxiliary loss 56 function.

This strategy also helped stabilize training for the conventional generator objective but was 57 not strictly needed.

Additionally, we analyze the transformed source and target data distributions in order to gain further 59 insight regarding the performance of our method.

We measure distances between these distributions 60 using Maximum Mean Discrepancy and Fréchet distances.

From our analysis we see that a good 61 performance in terms of distributional distance corresponds to good verification performance.

Our 62 speaker verification experiments show that the proposed adversarial speaker embedding framework 63 delivers robust performance, significantly outperforming a strong i-vector baseline.

Furthermore, by 64 averaging the scores of our different GAN models, we are able to achieve state-of-the-art results.

The first step for learning discriminative speaker embeddings is to learn a mapping DISPLAYFORM0 D from a sequence of speech frames from speaker s to a D-dimensional feature vector f. F (X) 69 can be implemented using a variety of neural network architectures.

We design our feature extractor 70 using a residual network structure.

We choose to model speech using 1-dimensional convolutional 71 filters, owing to the fact that speech is translation invariant along the time-axis only.

Following the 72 residual blocks we use a combination of self-attention and dense layers in order to represent input 73 audio of arbitrary size by a fixed-size vector, f. Unlike traditional approaches, our proposed feature 74 extractor is updated with an adversarial loss in addition to the standard task loss.

Self-Attention models are an active area of research in the speaker verification community.

Intuitively,

such models allow the network to focus on fragments of speech that are more speaker discriminative.

The attention layers computes a scalar weight corresponding to each time-step t: DISPLAYFORM0 These weights are then normalized, α t = sof tmax(e t ), to give them a probabilistic interpretation.

We use the attention model proposed in [25] , which extends attention to the mean as well as standard DISPLAYFORM0 DISPLAYFORM1 In this work we apply the use of self attention to convolutional feature maps, as indicated in Fig. 1 .

The last residual block outputs a tensor of size n B × n F × T , where n B is the batch size, n F is the 84 number of filters and T is time.

The input to the attention layer, h t , is a n F dimensional vector.

By using a self-attention model, we also equip our network with a more robust framework for computes similarity between classes using cosine, and forces the similarity of the correct class to be 94 greater than that of incorrect classes by a margin m. discriminator D, which is trained using the Binary Cross-Entropy loss (BCE).

DISPLAYFORM0 DISPLAYFORM1 Where X s , X t represent source and target data respectively.

E(.) is the feature extractor/generator.

The adversarial game between D(.) and E(.) is given by: DISPLAYFORM0 Equation FORMULA6 represents the most general form of the GAN game, and can be used to represent 112 different adversarial frameworks depending on the choice of L adv E .

Gradient Reversal: We obtain the gradient reversal framework by setting DISPLAYFORM0

Gradient reversal optimizes the true minmax objective of the adversarial game BID0 .

However, this 115 objective can become problematic, since the discriminator converges early during training and leads 116 to vanishing gradients.

We refer to the model trained with gradient reversal as Domain Adversarial

Neural Speaker Embeddings (DANSE).

GAN: Rather than directly using the minimax loss, the standard way to train the generator is using 119 the inverted label loss.

The generator objective is given by: DISPLAYFORM0 This splits the optimization into two independent objectives, one for the generator and one for the

In a typical GAN setting, the generator is trained only using fake data (with inverted labels).

This 125 structure is also maintained in several adversarial domain adaptation algorithms.

However, in the 126 context of this work we believe that updating the generator using both source and target data can be 127 beneficial.

In this case, the generator loss simply inverts the discriminator loss of eq. (1): DISPLAYFORM0 DISPLAYFORM1 Eq. FORMULA10 DISPLAYFORM2 In order to quantitatively evaluate our models in terms of domain adaptation, we measure the Inception network, we extract embeddings from our gan models from the source and target data.

The

Fréchet Distance between between the Gaussian (m s ,C s ) obtained from the source data distribution 225 p s and the Gaussian (m t ,C t ) from the target data is given by: DISPLAYFORM0 Source Domain Speaker Verification: We use the same source data used to compute the MMD and shows the best performance on this experiment albeit by a small margin.

In this work we we presented a novel framework for learning domain-invariant speaker embeddings

<|TLDR|>

@highlight

Speaker verificaiton performance can be significantly improved by adapting the model to in-domain data using Generative Adversarial Networks. Furthermore, the adaptation can be performed in an unsupervised way.

@highlight

Propose a number of GAN variants on the task of speaker recognition in the domain mismatched condition.