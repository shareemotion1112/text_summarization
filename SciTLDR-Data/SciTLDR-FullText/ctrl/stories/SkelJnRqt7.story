Separating mixed distributions is a long standing challenge for machine learning and signal processing.

Applications include: single-channel multi-speaker separation (cocktail party problem), singing voice separation and separating reflections from images.

Most current methods either rely on making strong assumptions on the source distributions (e.g. sparsity, low rank, repetitiveness) or rely on having training samples of each source in the mixture.

In this work, we tackle the scenario of extracting an unobserved distribution additively mixed with a signal from an observed (arbitrary) distribution.

We introduce a new method: Neural Egg Separation - an iterative method that learns to separate the known distribution from progressively finer estimates of the unknown distribution.

In some settings, Neural Egg Separation is initialization sensitive, we therefore introduce GLO Masking which ensures a good initialization.

Extensive experiments show that our method outperforms current methods that use the same level of supervision and often achieves similar performance to full supervision.

Humans are remarkably good at separating data coming from a mixture of distributions, e.g. hearing a person speaking in a crowded cocktail party.

Artificial intelligence, on the the hand, is far less adept at separating mixed signals.

This is an important ability as signals in nature are typically mixed, e.g. speakers are often mixed with other speakers or environmental sounds, objects in images are typically seen along other objects as well as the background.

Understanding mixed signals is harder than understanding pure sources, making source separation an important research topic.

Mixed signal separation appears in many scenarios corresponding to different degrees of supervision.

Most previous work focused on the following settings:Full supervision: The learner has access to a training set including samples of mixed signals {y i } ∈ Y as well as the ground truth sources of the same signals {b i } ∈ B and {x i } ∈ X (such that y i = x i + b i ).

Having such strong supervision is very potent, allowing the learner to directly learn a mapping from the mixed signal y i to its sources (x i , b i ).

Obtaining such strong supervision is typically unrealistic, as it requires manual separation of mixed signals.

Consider for example a musical performance, humans are often able to separate out the different sounds of the individual instruments, despite never having heard them play in isolation.

The fully supervised setting does not allow the clean extraction of signals that cannot be observed in isolation e.g. music of a street performer, car engine noises or reflections in shop windows.

The learner has access to a training set containing samples from the mixed signal {y i } ∈ Y as well as samples from all source distributions {b j } ∈ B and {x k } ∈ X .

The learner however does not have access to paired sets of the mixed and unmixed signal ground truth (that is for any given y i in the training set, b i and x i are unknown).

This supervision setting is more realistic than the fully supervised case, and occurs when each of the source distributions can be sampled in its pure form (e.g. we can record a violin and piano separately in a studio and can thus obtain unmixed samples of each of their distributions).

It is typically solved by learning to separate synthetic mixtures b j + x k of randomly sampled b j and x k .No supervision: The learner only has access to training samples of the mixed signal Y but not to sources B and X .

Although this settings puts the least requirements on the training dataset, it is a hard problem and can be poorly specified in the absence of strong assumptions and priors.

It is generally necessary to make strong assumptions on the properties of the component signals (e.g. smoothness, low rank, periodicity) in order to make progress in separation.

This unfortunately severely limits the applicability of such methods.

In this work we concentrate on the semi-supervised setting: unmixing of signals in the case where the mixture Y consists of a signal coming from an unobserved distribution X and another signal from an observed distribution B (i.e. the learner has access to a training set of clean samples such that {b j } ∈ B along with different mixed samples {y i } ∈ Y).

One possible way of obtaining such supervision, is to label every signal sample by a label, indicating if the sample comes only from the observed distribution B or if it is a mixture of both distributions B + X .

The task is to learn a parametric function able to separate the mixed signal y i ∈ Y into sources x i ∈ X and b i ∈ B s.t.

y i = b i + x i .

Such supervision is much more generally available than full supervision, while the separation problem becomes much simpler than when fully unsupervised.

We introduce a novel method: Neural Egg Separation (NES) -consisting of i) iterative estimation of samples from the unobserved distribution X ii) synthesis of mixed signals from known samples of B and estimated samples of X iii) training of separation regressor to separate the mixed signal.

Iterative refinement of the estimated samples of X significantly increases the accuracy of the learned masking function.

As an iterative technique, NES can be initialization sensitive.

We therefore introduce another method -GLO Masking (GLOM) -to provide NES with a strong initialization.

Our method trains two deep generators end-to-end using GLO to model the observed and unobserved sources (B and X ).

NES is very effective when X and B are uncorrelated, whereas initialization by GLOM is most important when X and B are strongly correlated such as e.g. separation of musical instruments.

Initialization by GLOM was found to be much more effective than by adversarial methods.

Experiments are conducted across multiple domains (image, music, voice) validating the effectiveness of our method, and its superiority over current methods that use the same level of supervision.

Our semi-supervised method is often competitive with the fully supervised baseline.

It makes few assumptions on the nature of the component signals and requires lightweight supervision.

Source separation: Separation of mixed signals has been extensively researched.

In this work, we focus on single channel separation.

Unsupervised (blind) single-channel methods include: ICA BID3 and RPCA BID10 .

These methods attempt to use coarse priors about the signals such as low rank, sparsity or non-gaussianity.

HMM can be used as a temporal prior for longer clips BID25 , however here we do not assume long clips.

Supervised source separation has also been extensively researched, classic techniques often used learned dictionaries for each source e.g. NMF BID30 .

Recently, neural network-based gained popularity, usually learning a regression between the mixed and unmixed signals either directly BID11 or by regressing the mask BID29 BID32 .

Some methods were devised to exploit the temporal nature of long audio signal by using RNNs BID19 , in this work we concentrate on separation of short audio clips and consider such line of works as orthogonal.

One related direction is Generative Adversarial Source Separation BID27 BID28 ) that uses adversarial training to match the unmixed source distributions.

This is needed to deal with correlated sources for which learning a regressor on synthetic mixtures is less effective.

We present an Adversarial Masking (AM) method that tackles the semi-supervised rather than the fully supervised scenario and overcomes mixture collapse issues not present in the fully supervised case.

We found that non-adversarial methods perform better for the initialization task.

The most related set of works is semi-supervised audio source separation BID26 BID0 , which like our work attempt to separate mixtures Y given only samples from the distribution of one source B. Typically NMF or PLCA (which is a similar algorithm with a probabilistic formulation) are used.

We show experimentally that our method significantly outperforms NMF.Disentanglement: Similarly to source separation, disentanglement also deals with separation in terms of creating a disentangled representation of a source signal, however its aim is to uncover latent factors of variation in the signal such as style and content or shape and color e.g. BID4 ; BID7 .

Differently from disentanglement, our task is separating signals rather than the latent representation.

Generative Models: Generative models learn the distribution of a signal directly.

Classical approaches include: SVD for general signals and NMF BID17 ) for non-negative signals.

Recently several deep learning approaches dominated generative modeling including: GAN BID6 , VAE BID15 and GLO BID1 .

Adversarial training (for GANs) is rather tricky and often leads to mode-collapse.

GLO is non-adversarial and allows for direct latent optimization for each source making it more suitable than VAE and GAN.

In this section we present our method for separating a mixture of sources of known and unknown distributions.

We denote the mixture samples y i , the samples with the observed distribution b i and the samples from the unobserved distribution x i .

Our objective is to learn a parametric function T (), such that b i = T (y i ).Full Supervision: In the fully supervised setting (where pairs of y i and b i are available) this task reduces to a standard supervised regression problem, in which a parametric function T () (typically a deep neural network) is used to directly optimize: DISPLAYFORM0 Where typically is the Euclidean or the L 1 loss.

In this work we use () = L 1 ().Mixed-unmixed pairs are usually unavailable, but in some cases it is possible to obtain a training set which includes unrelated samples x j and b k e.g. BID29 BID32 .

Methods typically randomly sample x j and b k sources and synthetically create mixtures y jk = x j + b k .

The synthetic pairs (b k , y jk ) can then be used to optimize Eq. 1.

Note that in cases where X and B are correlated (e.g. vocals and instrumental accompaniment which are temporally dependent), random synthetic mixtures of x and b might not be representative of y and fail to generalize on real mixtures.

Semi-Supervision: In many scenarios, clean samples of both mixture components are not available.

Consider for example a street musical performance.

Although crowd noises without street performers can be easily observed, street music without crowd noises are much harder to come by.

In this case therefore samples from the distribution of crowd noise B are available, whereas the samples from the distribution of the music X are unobserved.

Samples from the distribution of the mixed signal Y i.e. the crowd noise mixed with the musical performance are also available.

The example above illustrates a class of problems for which the distribution of the mixture and a single source are available, but the distribution of another source is unknown.

In such cases, it is not possible to optimize Eq. 1 directly due to the unavailability of pairs of b and y.

Neural Egg Separation: Fully-supervised optimization (as in Eq. 1) is very effective when pairs of b i and y i are available.

We present a novel algorithm, which iteratively solves the semi-supervised task as a sequence of supervised problems without any clean training examples of X .

We name the method Neural Egg Separation (NES), as it is akin to the technique commonly used for separating egg whites and yolks.

The core idea of our method is that although no clean samples from X are given, it is still possible to learn to separate mixtures of observed samples b j from distribution B combined with some estimates of the unobserved distribution samplesx i .

Synthetic mixtures are created by randomly sampling an approximate samplex i from the unobserved distribution and combining with training sample b j : DISPLAYFORM1 thereby creating pairs (ỹ ij , b j ) for supervised training.

Note that the distribution of synthetic mixturesỹ ij might be different from the real mixture sample distribution y j , but the assumption (which is empirically validated) is that it will eventually converge to the correct distribution.

During each iteration of NES, a neural separation function T () is trained on the created pairs by optimizing the following term: DISPLAYFORM2 At the end of each iteration, the separation function T () can be used to approximately separate the training mixture samples y i into their sources: DISPLAYFORM3 The refined X domain estimatesx i are used for creating synthetic pairs for finetuning T () in the next iteration (as in Eq. 3).The above method relies on having an estimate of the unobserved distribution samples as input to the first iteration.

One simple scheme is to initialize the estimates of the unobserved distribution samples in the first iteration asx i = c · y i , where c is a constant fraction (typically 0.5).

Although this initialization is very naive, we show that it achieves very competitive performance in cases where the sources are independent.

More advanced initializations will be discussed below.

At test time, separation is simply carried out by a single application of the trained separation function T () (exactly as in Eq. 4).

Mixture samples {y i }, Observed source samples {b j } Result: Separation function T () Initialize synthetic unobservable samples withx i ← c · y i or using AM or GLOM; Initialize T () with random weights; DISPLAYFORM0 Optimize separation function for P epochs: DISPLAYFORM1 Update estimates of unobserved distribution samples: DISPLAYFORM2 Algorithm 1: NES Algorithm Our full algorithm is described in Alg.

1.

For optimization, we use SGD using ADAM update with a learning rate of 0.01.

In total we perform N = 10 iterations, each consisting of optimization of T and estimation ofx i , P = 25 epochs are used for each optimization of Eq. 3.GLO Masking: NES is very powerful in practice despite its apparent simplicity.

There are some cases for which it can be improved upon.

As with other synthetic mixture methods, it does not take into account correlation between X and B e.g. vocals and instrumental tracks are highly related, whereas randomly sampling pairs of vocals and instrumental tracks is likely to synthesize mixtures quite different from Y. Another issue is finding a good initialization-this tends to affect performance more strongly when X and B are dependent.

We present our method GLO Masking (GLOM), which separates the mixture by a distributional constraint enforced via GLO generative modeling of the source signals.

GLO BID1 ) learns a generator G(), which takes a latent code z b and attempts to reconstruct an image or a spectrogram: b = G(z b ).

In training, GLO learns end-to-end both the parameters of the generator G() as well as a latent code z b for every training sample b. It trains per-sample latent codes by direct gradient descent over the values of z b (similar to word embeddings), rather than by a feedforward encoder used by autoencoders (e.g. z b = E(b)).

This makes it particularly suitable for our scenario.

Let us define the set of latent codes: DISPLAYFORM3 The optimization is therefore: DISPLAYFORM4 We propose GLO Masking, which jointly trains generators: G B () for B and G X () for X such that their sum results in mixture samples y = G B (z DISPLAYFORM5 We use the supervision of the observed source B to train G B (), while the mixture Y contributes residuals that supervise the training of G X ().

We also jointly train the latent codes for all training images: z b ∈ Z for all b ∈ B, and z B y ∈ Z B , z X y ∈ Z X for all y ∈ Y. The optimization problem is: DISPLAYFORM6 As GLO is able to overfit arbitrary distributions, it was found that constraining each latent code vector z to lie within the unit ball z · z ≤ 1 is required for generalization.

Eq. 6 can either be optimized end-to-end, or the left-hand term can be optimized first to yield Z, G B (), then the right-hand term is optimized to yield Z B , Z X , G X ().

Both optimization procedures yield similar performance (but separate training does not require setting λ).

Once G B () and G X () are trained, for a new mixture sample we infer its latent codes: DISPLAYFORM7 Our estimate for the sources is then: DISPLAYFORM8 Masking Function:

In separation problems, we can exploit the special properties of the task e.g. that the mixed signal y i is the sum of two positive signals x i and b i .

Instead of synthesizing the new sample, we can instead simply learn a separation mask m(), specifying the fraction of the signal which comes from B. The attractive feature of the mask is always being in the range [0, 1] (in the case of positive additive mixtures of signals).

Even a constant mask will preserve all signal gradients (at the cost of introducing spurious gradients too).

Mathematically this can be written as: DISPLAYFORM9 For NES (and baseline AM described below), we implement the mapping function T (y i ) using the product of the masking function y i · m(y i ).

In practice we find that learning a masking function yields much better results than synthesizing the signal directly (in line with other works e.g. BID29 ; BID5 ).GLOM models each source separately and is therefore unable to learn the mask directly.

Instead we refine its estimate by computing an effective mask from the element-wise ratio of estimated sources: DISPLAYFORM10 Initializing Neural Egg Separation by GLOM: Due to the iterative nature of NES, it can be improved by a good initialization.

We therefore devise the following method: i) Train GLOM on the training set and infer the mask for each mixture.

This is operated on images or mel-scale spectrograms at 64 × 64 resolutions ii) For audio: upsample the mask to the resolution of the highresolution linear spectrogram and compute an estimate of the X source linear spectrogram on the training set iii) Run NES on the observed B spectrograms and estimated X spectrograms.

We find experimentally that this initialization scheme improves NES to the point of being competitive with fully-supervised training in most settings.

To evaluate the performance of our method, we conducted experiments on distributions taken from multiple real-world domains: images, speech and music, in cases where the two signals are correlated and uncorrelated.

We evaluated our method against 3 baseline methods:Constant Mask (Const): This baseline uses the original mixture as the estimate.

This baseline method, proposed by BID26 , first trains a set of l bases on the observed distribution samples B by Sparse Adversarial Masking (AM): As an additional contribution, we introduce a new semi-supervised method based on adversarial training, to improve over the shallow NMF baseline.

AM trains a masking function m() so that after masking, the training mixtures are indistinguishable from the distribution of source B under an adversarial discriminator D().

The loss functions (using LS-GAN BID18 ) are given by: DISPLAYFORM0 Differently from CycleGAN BID34 and DiscoGAN BID14 , AM is not bidirectional and cannot use cycle constraints.

We have found that adding magnitude prior L 1 (m(y), 1) improves performance and helps prevent collapse.

To partially alleviate mode collapse, we use Spectral Norm on the discriminator.

We evaluated our proposed methods:GLO Masking (GLOM): GLO Masking on mel-spectrograms or images at 64 × 64 resolution.

The NES method detailed in Sec. 3.

Initializing X estimates using a constant (0.5) mask over Y training samples.

Initializing NES with the X estimates obtained by GLO Masking.

To upper bound the performance of our method, we also compute a fully supervised baseline, for which paired data of b i ∈ B, x i ∈ X and y i ∈ Y are available.

We train a masking function with the same architecture as used by all other regression methods to directly regress synthetic mixtures to unmixed sources.

This method uses more supervision than our method and is an upper bound.

More implementation details can be found in appendix A.

In this section we evaluate the effectiveness of our method on image mixtures.

We conduct experiments both on the simpler MNIST dataset and more complex Shoes and Handbags datasets.

To evaluate the quality of our method on image separation, we design the following experimental protocol.

We split the MNIST dataset BID16 into two classes, the first consisting of the digits 0-4 and the second consisting of the digits 5-9.

We conduct experiments where one source has an observed distribution B while the other source has an unobserved distribution X .

We use 12k B training images as the B training set, while for each of the other 12k B training images, we randomly sample a X image and additively combine the images to create the Y training set.

We evaluate the performance of our method on 5000 Y images similarly created from the test set of X and B. The experiment was repeated for both directions i.e. 0-4 being B while 5-9 in X , as well as 0-4 being X while 5-9 in B.In Tab.

1, we report our results on this task.

For each experiment, the top row presents the results (PSNR and SSIM) on the X test set.

Due to the simplicity of the dataset, NMF achieved reasonable performance on this dataset.

GLOM achieves better SSIM but worse PSNR than NMF while AM performed 1-2dB better.

NES achieves much stronger performance than all other methods, achieving about 1dB worse than the fully supervised performance.

Initializing NES with the masks obtained by GLOM, results in similar performance to the fully-supervised upper bound.

FT from AM (numbers for finetuning from AM were omitted from the tables for clarity, as they were inferior to finetuning from GLOM in all experiments) achieved similar performance (24.0/0.95 and 23.8/0.95) to FT from GLOM.

In order to evaluate our method on more realistic images, we evaluate on separating mixtures consisting of pairs of images sampled from the Handbags BID33 and Shoes BID31 datasets, which are commonly used for evaluation of conditional image generation methods.

To create each Y mixture image, we randomly sample a shoe image from the Shoes dataset and a handbag image from the Handbags dataset and sum them.

For the observed distribution, we sample another 5000 different images from a single dataset.

We evaluate our method both for cases when the X class is Shoes and when it is Handbags.

From the results in Tab.

1, we can observe that NMF failed to preserve fine details, penalizing its performance metrics.

GLOM (which used a VGG perceptual loss) performed much better, due to greater expressiveness.

AM performance was similar to GLOM on this task, as the perceptual loss and stability of training of non-adversarial models helped GLOM greatly.

NES performed much better than all other methods, even when initialized from a constant mask.

Finetuning from GLOM, helped NES achieve stronger performance, nearly identical to the fully-supervised upper bound.

It performed better than finetuning from AM (not shown in table) which achieved 22.5/0.85 and 22.7/0.86 .

Similar conclusions can be drawn from the qualitative comparison in the figure above.

Separating environmental noise from speech is a long standing problem in signal processing.

Although supervision for both human speech and natural noises can generally be obtained, we use this task as a benchmark to evaluate our method's performance on audio signals where X and B are not dependent.

This benchmark is a proxy for tasks for which a clean training set of X sounds cannot be obtained e.g. for animal sounds in the wild, background sounds training without animal noises can easily be obtained, but clean sounds made by the animal with no background sounds are unlikely to be available.

We obtain clean speech segments from the Oxford-BBC Lip Reading in the Wild (LRW) Dataset BID2 , and resample the audio to 16 kHz.

Audio segments from ESC-50 BID21 , a dataset of environmental audio recordings organized into 50 semantic classes, are used as additive noise.

Noisy speech clips are created synthetically by first splitting clean speech into clips with duration of 0.5 seconds, and adding a random noise clip, such that the resulting SNR is zero.

We then compute a mel-scale spectrogram with 64 bins, using STFT with window size of 25 ms, hop length of 10 ms, and FFT size of 512, resulting in an input audio feature of 64 × 64 scalars.

Finally, power-law compression is performed with p = 0.3, i.e. A 0.3 , where A is the input audio feature.

From the results in Tab.

2, we can observe that GLOM, performed better than Semi-Supervised NMF by about 1dB better.

AM training, performed about 2dB better than GLOM.

Due to the independence between the sources in this task, NES performed very well, even when trained from a constant mask initialization.

Performance was less than 1dB lower than the fully supervised result (while not requiring any clean speech samples).

In this setting due to the strong performance of NES, initializing NES with the speech estimates obtained by GLOM (or AM), did not yield improved performance.

Separating vocal music into singing voice and instrumental music as well as instrumental music and drums has been a standard task for the signal processing community.

Here our objective is to understand the behavior of our method in settings where X and B are dependent (which makes synthesis by addition of random X and B training samples a less accurate approximation).For this task we use the MUSDB18 Dataset BID24 , which, for each music track, comprises separate signal streams of the mixture, drums, bass, the rest of the accompaniment, and the vocals.

We convert the audio tracks to mono, resample to 20480 Hz, and then follow the procedure detailed in Sec. 4.2 to obtain input audio features.

From the results in Tab.

3, we can observe that NMF was the worst performer in this setting (as its simple bases do not generalize well between songs).

GLOM was able to do much better than NMF and was even competitive with NES on Vocal-Instrumental separation.

Due to the dependence between the two sources and low SNR, initialization proved important for NES.

Constant initialization NES performed similarly to AM and GLOM.

Finetuning NES from GLOM masks performed much better than all other methods and was competitive with the supervised baseline.

GLOM was much better than AM initialization (not shown in table) that achieved 0.9 and 2.9.

GLO vs. Adversarial Masking: GLO Masking as a stand alone technique usually performed worse than Adversarial Masking.

On the other hand, finetuning from GLO masks was far better than finetuning from adversarial masks.

We speculate that mode collapse, inherent in adversarial training, makes the adversarial masks a lower bound on the X source distribution.

GLOM can result in models that are too loose (i.e. that also encode samples outside of X ).

But as an initialization for NES finetuning, it is better to have a model that is too loose than a model which is too tight.

Supervision Protocol:

Supervision is important for source separation.

Completely blind source separation is not well specified and simply using general signal statistics is generally unlikely to yield competitive results.

Obtaining full supervision by providing a labeled mask for training mixtures is unrealistic but even synthetic supervision in the form of a large training set of clean samples from each source distribution might be unavailable as some sounds are never observed on their own (e.g. sounds of car wheels).

Our setting significantly reduces the required supervision to specifying if a certain sound sample contains or does not contain the unobserved source.

Such supervision can be quite easily and inexpensively provided.

For further sample efficiency increases, we hypothesize that it would be possible to label only a limited set of examples as containing the target sound and not, and to use this seed dataset to finetune a deep sound classifier to extract more examples from an unlabeled dataset.

We leave this investigation to future work.

To showcase the generality of our method, we chose not to encode task specific constraints.

In practical applications of our method however we believe that using signalspecific constraints can increase performance.

Examples of such constraints include: repetitiveness of music BID23 , sparsity of singing voice, smoothness of natural images.

Non-Adversarial Alternatives: The good performance of GLOM vs. AM on the vocals separation task, suggests that non-adversarial generative methods may be superior to adversarial methods for separation.

This has also been observed in other mapping tasks e.g. the improved performance of NAM BID8 over DCGAN .

A perfect signal separation function is a stable global minimum of NES as i) the synthetic mixtures are equal to real mixtures ii) real mixtures are perfectly separated.

In all NES experiments (with constant, AM or GLOM initialization), NES converged after no more than 10 iterations, typically to different local minima.

It is empirically evident that NES is not guaranteed to converge to a global minimum (although it converges to good local minima).

We defer formal convergence analysis of NES to future work.

In this paper we proposed a novel method-Neural Egg Separation-for separating mixtures of observed and unobserved distributions.

We showed that careful initialization using GLO Masking improves results in challenging cases.

Our method achieves much better performance than other methods and was usually competitive with full-supervision.

GLOM and AM use the same generator and discriminator architectures respectively for audio as they do for images.

They operate on mel-scale spectrogram at 64 × 64 resolution.

Masking Network: The generator for AM operates on 64 × 64 mel-scale audio spectrograms.

It consists of 3 convolutional and 3 deconvolutional layers with stride 2 and no pooling.

Outputs of convolutional layers are normalized with BatchNorm and rectified with ReLU activation, except for the last layer where sigmoid is used.

In addition to the LSGAN loss, an additional magnitude loss is used, with relative weight of λ = 1.NES and the supervised method operate on full linear spectrogram of dimensions 257 × 64, without compression.

They use the same DiscoGAN architecture, which contains two additional convolutional and deconvolutional layers.

In this section we describe our implementation of the NMF semi-supervised source separation baseline BID26 .

NMF trains a decomposition: B = W Z where W are the weights and Z = [z 1 , ..., z N ] are the per sample latent codes.

Both W and Z are non-negative.

Regularization is important for the performance of the method.

We follow BID9 BID12 and use L 1 regularization to ensure sparsity of the weights.

The optimization problem therefore becomes: We present a qualitative analysis of the results of GLOM and NES.

To understand the quality of generations of GLO and the effect of the masking function, we present in Fig.2 the results of the GLO generations given different mixtures from the Speech dataset.

We also show the results after the masking operation described in Eq. 10.

It can be observed that GLO captures the general features of the sources, but is not able to exactly capture fine detail.

The masking operation in GLOM helps it recover more fine-grained details, and results in much cleaner separations.

DISPLAYFORM0 We also show in Fig.2 the evolution of NES as a function of iteration for the same examples.

NES(k) denotes the result of NES after k iterations.

It can be seen the NES converges quite quickly, and results improve further with increasing iterations.

In FIG1 , we can observe the performance of NES on the Speech dataset in terms of SDR as a function of iteration.

The results are in line with the qualitative examples presented before, NES converges quickly but makes further gains with increasing iterations.

<|TLDR|>

@highlight

An iterative neural method for extracting signals that are only observed mixed with other signals