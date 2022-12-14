Generative priors have become highly effective in solving inverse problems including denoising, inpainting, and reconstruction from few and noisy measurements.

With a generative model we can represent an image with a much lower dimensional latent codes.

In the context of compressive sensing, if the unknown image belongs to the range of a pretrained generative network, then we can recover the image by estimating the underlying compact latent code from the available measurements.

However, recent studies revealed that even untrained deep neural networks can work as a prior for recovering natural images.

These approaches update the network weights keeping latent codes fixed to reconstruct the target image from the given measurements.

In this paper, we optimize over network weights and latent codes to use untrained generative network as prior for video compressive sensing problem.

We show that by optimizing over latent code, we can additionally get concise representation of the frames which retain the structural similarity of the video frames.

We also apply low-rank constraint on the latent codes to represent the video sequences in even lower dimensional latent space.

We empirically show that our proposed methods provide better or comparable accuracy and low computational complexity compared to the existing methods.

Compressive sensing refers to a broad class of problems in which we aim to recover a signal from a small number of measurements [1] - [3] .

Suppose we are given a sequence of measurements for t = 1, . . .

, T as y t = A t x t + e t ,

where x t denotes the t th frame in the unknown video sequence, y t denotes its observed measurements, A t denotes the respective measurement operator, and e t denotes noise or error in the measurements.

Our goal is to recover the video sequence (x t ) from the available measurements (y t ).

The recovery problem becomes especially challenging as the number of measurements (in y t ) becomes very small compared to the number of unknowns (in x t ).

Classical signal priors exploit sparse and low-rank structures in images and videos for their reconstruction [4] - [16] .

However, the natural images exhibits far richer nonlinear structures than sparsity alone.

We focus on a newly emerging generative priors that learn a function that maps vectors drawn from a certain distribution in a low-dimensional space into images in a highdimensional space.

The generative model and optimization problems we use are inspired by recent work on using generative models for compressive sensing in [17] - [23] .

Compressive sensing using generative models was introduced in [17] , which used a trained deep generative network as a prior for image reconstruction from compressive measurements.

Afterwards deep image prior (DIP) used an untrained convolutional generative model as a prior for solving inverse problems such as inpainting and denoising because of their tendency to generate natural images [22] ; the reconstruction problem involves optimization of generator network parameters.

Inspired by these observations, a number of methods have been proposed for solving compressive sensing problem by optimizing generator network weights while keeping the latent code fixed at a random value [19] , [20] .

Both DIP [22] and deep decoder [20] update the model weights to generate a given image; therefore, the generator can reconstruct wide range of images.

One key difference between the two approaches is that the network used in DIP is highly overparameterized, while the one used in deep decoder is underparameterized.

We observed two main limitations in the DIP and deep decoder-based video recovery that we seek to address in this paper.

(1) The latent codes in DIP and deep decoder methods are initialized at random and stay fixed throughout the recovery process.

Therefore, we cannot infer the structural similarities in the images from the structural similarities in the latent codes.

(2) Both of these methods train one network per image.

A naive approach to train one network per frame in a video will be computationally prohibitive, and if we train a single network to generate the entire video sequence, then their performance degrades.

Therefore, we propose joint optimization over network weights ?? and the latent codes z t to reconstruct video sequence.

Thus we learn a single generator and a set of latent codes to represent a video sequence.

We observe that when we optimize over latent code alongside network weights, the temporal similarity in the video frames is reflected in the latent code representation.

To exploit similarities among the frames in a video sequence, we also include low-rank constraints on the latent codes.

An illustration of different types of representations we use in this paper are shown in Figure 1 .

In this paper, we reconstruct a video sequence from the compressive measurements in (1) by jointly optimizing over the latent codes z t and the network parameters ??.

Since the frames in a video sequence exhibit rich redundancies in their representation, we impose a low-rank constraint on the latent codes to represent the video sequence with a more compact representation of the latent codes.

The key contributions of this paper are as follows.

??? We demonstrate that joint optimization allows us to learn a single generator network for an entire video sequence and corresponding latent codes simultaneously.

We demonstrate that this approach has lower computational complexity and requires less number of parameters to reliably generate the entire video sequence.

Furthermore, joint optimization retains the similarity structure of the video frames in their latent representation which leaves further scope for different tasks which involves latent space manipulation.

??? Consecutive frames in a video sequence share lot of similarities.

To encode similarities among the reconstructed frames, we introduce low-rank constraints on the generator latent codes.

This enables us to represent a video sequence with even smaller number of parameters in the latent space.

We show that, in some cases, the low-rank structure on the latent codes also provides a nice low-dimensional manifold.

For a single image reconstruction, deep image prior solve the following optimization to obtain optimal??, arg min

In this optimization, z is initialized randomly and kept unaltered.

To jointly optimize the latent codes and generator parameters for a video sequence, we use the similar formulation as in (2) but optimize it over the z t and ??.

The resulting optimization problem can be written as

The reconstructed video sequence can be generated using the estimated latent codes (??? 1 , . . .

,??? T ) and generator weights (??) asx t = G??(??? t ).

We initialize latent codes with samples drawn from a Gaussian distribution and normalize them to have unit norm.

We initialize ?? with random weights using the initialization scheme in [24] .

Initilizing the generator with a pretrained set of weights can potentially serve as a good initialization and lead to good and faster convergence.

We tested both variants, but observed little difference in performance; therefore, we use random initialization of parameters in this paper.

Each iteration of joint optimization consists of two steps: 1) latent code optimization and 2) network parameter optimization.

After every gradient descent update of the latent codes, z t , we update the model parameters with stochastic gradient descent.

In all of our experiments with joint optimization, we learned a single set of network weights for the entire sequence.

We note that it is possible to divide a longer video sequences into small segments and learn different sets of network weights for each of them.

At the end of our reconstruction process, we have a single set of trained weights??, reconstructed framesx t and their corresponding optimal latent codes??? t .

As we optimize over the latent codes and the network weights in joint optimization, the latent codes capture the temporal similarity of the video frames.

To further exploit the redundancies in a video sequence, we assume that the variation in the sequence of images are localized and the latent codes sequence can be represented in a low-dimensional space compared to their ambient dimension.

Let us define a matrix Z with all the latent codes as

where z t is the latent code corresponding to t th image of the sequence.

To impose a low-rank constraint, we solve the following constrained optimization:

We solve (4) using a projected gradient descent method in which we project the latent code estimates after every iteration to a manifold of rank-r matrices.

To do that, we compute Z matrix and its rank-r approximation using principal component analysis (PCA) or singular value decomposition (SVD).

In this manner, we can express each of the latent codes in terms of r orthogonal basis vectors vectors u 1 , . . . , u r as

where ?? ij is the weight of the corresponding basis vector.

We can represent a video sequence with T frames with r orthogonal codes, and the lowrank representation of latent codes requires r ?? k + r ?? T parameters compared to T ?? k. This offers r(

k ) times compression to our latent code representation.

As we observe later, we use r = 4 for k = 256 and T = 32 which gives us compression of 0.14 in latent code representation.

In this paper we report the results for one synthetic sequence which we refer to as 'Rotating MNIST'.

In this sequence, we resize one MNIST digit to 64 ?? 64 and rotate by 2

??? per frame for a total of 32 frames.

We experiment on different real video [25] and UCF101 dataset [26] .

In Table I , we report our results for 'Handclapping', 'Handwaving' and 'Walking' video sequences from KTH dataset; 'Archery', 'Apply Eye Makeup' and 'Band Marching' video sequences from UCF101 dataset.

We centered and resized every frame in KTH videos to 64 ?? 64 and UCF101 videos to 256 ?? 256 pixels.

We used the well-known DCGAN architecture [27] for our generators, except that we do not use any batch-normalization layer.

The latent code dimensions for grayscale 64 ?? 64, RGB 64 ?? 64 and RGB 256 ?? 256 video sequences are 64, 256 and 512 respectively.

We use Adam optimizer for generator weights optimization and SGD for latent code optimization.

Unless otherwise mentioned, we use rank=4 constraint as low rank constraint because we empirically found that we need a least rank=4 for a video sequence with 32 frames to get comparable performance.

We show comparison with classical total variation minimization based TVAL3D (3D extension of TVAL3 [28] ) algorithm and state-of-the-art untrained generative prior based deep decoder [20] on denoising, inpainting, and compressive sensing tasks.

We use two different deep decoder settings: underparameterized deep decoder (UP deep decoder) and overparameterized deep decoder (OP deepdecoder).

Although the authors suggested deep decoder to be UP, we report the results for OP deep decoder as well because it shows better performance and its hyperparameters are tuned by the authors of deep decoder.

Other then denoising and inpainting, we performed compressive random projection experiments where we used separable measurements, Y = P T XP , where X, Y are reshaped versions of x, y as 2D matrices, P is a random projection matrix.

We report the results for denoising experiment at 20 dB SNR noise, inpainting experiment for 80% missing pixels and compressive sensing experiments for 20% available measurements in Table I .

From the results, we can observe that joint optimization with/without low-rank constraint outperform TVAL3D algorithm and UP deep decoder.

It performs at par with OP deep decoder.

In Figure 2 , we show reconstruction performance for denoising, inpainting and compressive sensing at different measurement rate or noise level for 'Handwaving' video sequence.

We can get the similar observation from these curves as well.

We report some reconstruction results for 'Handwaving' sequence in Figure 2 .

From the reconstructions, we can say that joint optimization performs at par with the comparing algorithms.

It especially performs well in reconstructing details from masked frames. [28] ) and deep decoder [20] .

The results are average over five experiments with different random measurement matrices (or noise in the case of denoising).

Rotating

The computational complexity of our proposed methods vary with the choice of the generator structure.

We have chosen DCGAN generator structure for our experiments.

We calculate memory requirement for gradient descent using torchsummary package [29] .

For a single 64 ?? 64 RGB image, memory requirement for UP deep decoder, OP deep decoder and joint optimization is 2.75 MB, 66.48 MB and 2.06 MB respectively.

For a single 256 ?? 256 RGB image, memory requirement for UP deep decoder, OP deep decoder and joint optimization is 44.03 MB, 1239.75 MB and 10.88 MB respectively.

For a RGB video seqence with 32 frames, UP deep decoder will require 11, 304 ?? 32 (361,728) parameters while OP deep decoder will have 397, 056 ?? 32 ( 12.7M) parameters.

On the other hand we need 4,852,736 and 6,988,544 network parameters to represent RGB 64 ?? 64 and 256 ?? 256 video sequences, respectively in joint optimization method with DCGAN generator.

Because of the huge memory requirement, OP deep decoder is not suitable for optimization over entire video sequence whereas the low capacity hinders UP deep decoder from generating entire video sequence.

To investigate the similarity structure in the latent codes obtained by joint optimization, we performed another experiment in which we concatenated 16 frames from each of the The cosine similarity matrices for the video frames, compressive measurements, latent codes for with fixed latent codes (random), latent codes for joint optimization, and latent codes for joint optimization with low-rank are presented in Figure 3 (a)-(e).

We can distinguish the video sequences from the pairwise similarity matrices of the latent codes we estimate with joint optimization.

We also observe that the low-rank constraint improves the similarity matrix.

We mentioned that using low rank constraint we can represent the video sequences in much lower dimensional space.

If the generator function is continuous it makes sense that the latent space representation of a video sequence will retain their sequential structure in some low dimensional representation.

We demonstrate one such example using rank=2 constraint on the latent codes while reconstructing 'Rotating MNIST' sequence from its masked version with 80% pixels missing.

As we are enforcing, rank=2 constraint taking mean and first principal component, the latent codes should fall on line.

In Figure 4 , we represent the latent codes in a 2D plane using 2 orthogonal basis vectors.

t th point in Figure 4 , represent latent code of t th frame.

We can observe that latent codes are maintaining sequence in their 2D dimensional representation.

For complex motions, it might take higher dimensional representation to observe such sequential pattern.

<|TLDR|>

@highlight

Recover videos from compressive measurements by learning a low-dimensional (low-rank) representation directly from measurements while training a deep generator. 