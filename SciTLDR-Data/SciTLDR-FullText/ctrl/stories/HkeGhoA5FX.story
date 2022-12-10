In this paper, we propose a residual non-local attention network for high-quality image restoration.

Without considering the uneven distribution of information in the corrupted images, previous methods are restricted by local convolutional operation and equal treatment of spatial- and channel-wise features.

To address this issue, we design local and non-local attention blocks to extract features that capture the long-range dependencies between pixels and pay more attention to the challenging parts.

Specifically, we design trunk branch and (non-)local mask branch in each (non-)local attention block.

The trunk branch is used to extract hierarchical features.

Local and non-local mask branches aim to adaptively rescale these hierarchical features with mixed attentions.

The local mask branch concentrates on more local structures with convolutional operations, while non-local attention considers more about long-range dependencies in the whole feature map.

Furthermore, we propose residual local and non-local attention learning to train the very deep network, which further enhance the representation ability of the network.

Our proposed method can be generalized for various image restoration applications, such as image denoising, demosaicing, compression artifacts reduction, and super-resolution.

Experiments demonstrate that our method obtains comparable or better results compared with recently leading methods quantitatively and visually.

Image restoration aims to recover high-quality (HQ) images from their corrupted low-quality (LQ) observations and plays a fundamental role in various high-level vision tasks.

It is a typical ill-posed problem due to the irreversible nature of the image degradation process.

Some most widely studied image restoration tasks include image denoising, demosaicing, and compression artifacts reduction.

By distinctively modelling the restoration process from LQ observations to HQ objectives, i.e., without assumption for a specific restoration task when modelling, these tasks can be uniformly addressed in the same framework.

Recently, deep convolutional neural network (CNN) has shown extraordinary capability of modelling various vision problems, ranging from low-level (e.g., image denoising (Zhang et al., 2017a) , compression artifacts reduction BID10 , and image super-resolution BID43 BID37 BID40 ) to high-level (e.g., image recognition ) vision applications.

However, there are mainly three issues in the existing CNN based methods above.

First, the receptive field size of these networks is relatively small.

Most of them extract features in a local way with convolutional operation, which fails to capture the long-range dependencies between pixels in the whole image.

A larger receptive field size allows to make better use of training inputs and more context information.

This would be very helpful to capture the latent degradation model of LQ images, especially when the images suffer from heavy corruptions.

Second, distinctive ability of these networks is also limited.

Let's take image denoising as an example.

For a noisy image, the noise may appear in both the plain and textural regions.

Noise removal would be easier in the plain area than that in the textural one.

It is desired to make the denoising model focus on textual area more.

However, most previous denoising methods neglect to consider different contents in the noisy input and treat them equally.

This would result in over-smoothed outputs and some textural details would also fail to be recovered.

Third, all channel-wise features are treated equally in those networks.

This naive treatment lacks flexibility in dealing with different types of information (e.g., low-and high-frequency information).

For a set of features, some contain more information related to HQ image and the others may contain more information related to corruptions.

The interdependencies among channels should be considered for more accurate image restoration.

To address the above issues, we propose the very deep residual non-local attention networks (RNAN) for high-quality image restoration.

We design residual local and non-local attention blocks as the basic building modules for the very deep network.

Each attention block consists of trunk and mask branches.

We introduce residual block for trunk branch and extract hierarchical features.

For mask branch, we conduct feature downscaling and upscaling with largestride convolution and deconvolution to enlarge receptive field size.

Furthermore, we incorporate non-local block in the mask branch to obtain residual non-local mixed attention.

We apply RNAN for various restoration tasks, including image denoising, demosaicing, and compression artifacts reduction.

Extensive experiments show that our proposed RNAN achieves state-of-the-art results compared with other recent leading methods in all tasks.

To the best of our knowledge, this is the first time to consider residual non-local attention for image restoration problems.

The main contributions of this work are three-fold:•

We propose the very deep residual non-local networks for high-quality image restoration.

The powerful networks are based on our proposed residual local and non-local attention blocks, which consist of trunk and mask branches.

The network obtains non-local mixed attention with non-local block in the mask branch.

Such attention mechanis helps to learn local and non-local information from the hierarchical features.• We propose residual non-local attention learning to train very deep networks by preserving more low-level features, being more suitable for image restoration.

Using non-local lowlevel and high-level attention from the very deep network, we can pursue better network representational ability and finally obtain high-quality image restoration results.• We demonstrate with extensive experiments that our RNAN is powerful for various image restoration tasks.

RNAN achieves superior results over leading methods for image denoising, demosaicing, compression artifacts reduction, and super-resolution.

In addition, RNAN achieves superior performance with moderate model size and performs very fast.

Non-local prior.

As a classical filtering algorithm, non-local means BID4 ) is computed as weighted mean of all pixels of an image.

Such operation allows distant pixels to contribute to the response of a position at a time.

It was lately introduced in BM3D BID7 for image denoising.

Recently, BID35 proposed non-local neural network by incorporating non-local operations in deep neural network for video classification.

We can see that those methods mainly introduce non-local information in the trunk pipeline.

BID23 proposed non-local recurrent network for image restoration.

However, in this paper, we mainly focus on learning nonlocal attention to better guide feature extraction in trunk branch.

Attention mechanisms.

Generally, attention can be viewed as a guidance to bias the allocation of available processing resources towards the most informative components of an input BID15 .

Recently, tentative works have been proposed to apply attention into deep neural networks BID15 ).

It's usually combined with a gating function (e.g., sigmoid) to rescale the feature maps.

proposed residual attention network for image classification with a trunk-and-mask attention mechanism.

BID15 proposed squeezeand-excitation (SE) block to model channel-wise relationships to obtain significant performance improvement for image classification.

In all, these works mainly aim to guide the network pay more attention to the regions of interested.

However, few works have been proposed to investigate the effect of attention for image restoration tasks.

Here, we want to enhances the network with distinguished power for noise and image content.

Image restoration architectures.

Stacked denoising auto-encoder BID33 ) is one of the most well-known CNN-based image restoration method.

BID10 proposed AR-CNN for image compression artifact reduction with several stacked convolutional layers.

With the help of residual learning and batch normalization BID17 , Zhang et al. proposed DnCNN (Zhang et al., 2017a) for accurate image restoration and denoiser priors for image restoration in IRCNN (Zhang et al., 2017b) .

Recently, great progresses have been made in image restoration community, where Timofte et al. , BID1 , and Blau et al. BID3 lead the main competitions recently and achieved new research status and records.

For example, Wang et al. BID37 proposed a fully progressive image SR approach.

However, most methods are plain networks and neglect to use non-local information.

Conv RNAB RAB RAB RNAB Conv RAB Figure 1 : The framework of our proposed residual non-local attention network for image restoration.

'Conv', 'RNAB', and 'RAB' denote convolutional layer, residual non-local attention block, and residual local attention block respectively.

Here, we take image denoising as a task of interest.

The framework of our proposed residual non-local attention network (RNAN) is shown in Figure 1 .

Let's denote I L and I H as the low-quality (e.g., noisy, blurred, or compressed images) and highquality images.

The reconstructed image I R can be obtained by DISPLAYFORM0 where H RN AN denotes the function of our proposed RNAN.

With the usage of global residual learning in pixel space, the main part of our network can concentrate on learning the degradation components (e.g., noisy, blurring, or compressed artifacts).The first and last convolutional layers are shallow feature extractor and reconstruction layer respectively.

We propose residual local and non-local attention blocks to extract hierarchical attentionaware features.

In addition to making the main network learn degradation components, we further concentrate on more challenging areas by using local and non-local attention.

We only incorporate residual non-local attention block in low-level and high-level feature space.

This is mainly because a few non-local modules can well offer non-local ability to the network for image restoration.

Then RNAN is optimized with loss function.

Several loss functions have been investigated, such as L 2 BID25 Zhang et al., 2017a; Zhang et al., 2017b) , L 1 BID41 , perceptual and adversarial losses BID21 .

To show the effectiveness of our RNAN, we choose to optimize the same loss function (e.g., L 2 loss function) as previous works.

Given a training set DISPLAYFORM1 , which contains N low-quality inputs and their high-quality counterparts.

The goal of training RNAN is to minimize the L 2 loss function DISPLAYFORM2 where · 2 denotes l 2 norm.

As detailed in Section 4, we use the same loss function as that in other compared methods.

Such choice makes it clearer and more fair to see the effectiveness of our proposed RNAN.

Then we give more details to residual local and non-local attention blocks.

DISPLAYFORM3 Figure 2: Residual (non-)local attention block.

It mainly consists of trunk branch (labelled with gray dashed) and mask branch (labelled with red dashed).

The trunk branch consists of t RBs.

The mask branch is used to learning mixed attention maps in channel-and spatial-wise simultaneously.

Our residual non-local attention network is constructed by stacking several residual local and nonlocal attention blocks shown in Figure 2 .

Each attention block is divided into two parts: q residual blocks (RBs) in the beginning and end of attention block.

Two branches in the middle part: trunk branch and mask branch.

For non-local attention block, we incorporate non-local block (NLB) in the mask branch, resulting non-local attention.

Then we give more details to those components.

As shown in Figure 2 , the trunk branch includes t residual blocks (RBs).

Different from the original residual block in ResNet , we adopt the simplified RB from .

The simplified RB (labelled with blue dashed) only consists of two convolutional layers and one ReLU BID28 , omitting unnecessary components, such as maxpooling and batch normalization BID17 layers.

We find that such simplified RB not only contributes to image super-resolution , but also helps to construct very deep network for other image restoration tasks.

Feature maps from trunk branch of different depths serve as hierarchical features.

If attention mechanism is not considered, the proposed network would become a simplified ResNet.

With mask branch, we can take channel and spatial attention to adaptively rescale hierarchical features.

Then we give more details about local and non-local attention.

As labelled with red dashed in Figure 2 , the mask branches used in our network include local and non-local ones.

Here, we mainly focus on local mask branch, which can become a non-local one by using non-local block (NLB, labelled with green dashed arrow).The key point in mask branch is how to grasp information of larger scope, namely larger receptive field size, so that it's possible to obtain more sophisticated attention map.

One possible solution is to perform maxpooling several times, as used in for image classification.

However, more pixel-level accurate results are desired in image restoration.

Maxpooling would lose lots of details of the image, resulting in bad performance.

To alleviate such drawbacks, we choose to use large-stride convolution and deconvolution to enlarge receptive field size.

Another way is considering non-local information across the whole inputs, which will be discussed in the next subsection.

From the input, large-stride (stride ≥ 2) convolutional layer increases the receptive field size after m RBs.

After additional 2m RBs, the downscaled feature maps are then expanded by a deconvolutional denotes element-wise addition.layer (also known as transposed convolutional layer).

The upscaled features are further forwarded through m RBs and one 1 × 1 convolutional layer.

Then a sigmoid layer normalizes the output values, ranging in [0, 1].

Although the receptive field size of the mask branch is much larger than that of the trunk branch, it cannot cover the whole features at a time.

This can be achieved by using non-local block (NLB), resulting in non-local mixed attention.

As discussed above, convolution operation processes one local neighbourhood at a time.

In order to obtain better attention maps, here we seek to take all the positions into consideration at a time.

Inspired by classical non-local means method BID4 and non-local neural networks BID35 , we incorporate non-local block (NLB) into the mask branch to obtain non-local mixed attention (shown in FIG0 ).

The non-local operation can be defined as DISPLAYFORM0 where i is the output feature position index and j is the index that enumerates all possible positions.

x and y are the input and output of non-local operation.

The pairwise function f (x i , x j ) computes relationship between x i and x j .

The function g(x j ) computes a representation of the input at the position j.

As shown in FIG0 , we use embedded Gaussian function to evaluate the pairwise relationship DISPLAYFORM1 where W u and W v are weight matrices.

As investigated in BID35 , there are several versions of f , such as Gaussian function, dot product similarity, and feature concatenation.

We also consider a linear embedding for g: g(x j ) = W g x j with weight matrix W g .

Then the output z at position i of non-local block (NLB) is calculated as DISPLAYFORM2 where W z is a weight matrix.

For a given i, ∀j f (x i , x j )/ ∀j f (x i , x j ) in Eq. 3 becomes the softmax computation along dimension j.

The residual connection allows us to insert the NLB into pretrained networks BID35 by initializing W z as zero.

With non-local and local attention computation, feature maps in the mask branch are finally mapped by sigmoid function DISPLAYFORM3 where i ranges over spatial positions and c ranges over feature channel positions.

Such simple sigmoid operation is applied to each channel and spatial position, resulting mixed attention .

As a result, the mask branch with non-local block can produce non-local mixed attention.

However, simple multiplication between features from trunk and mask branches is not powerful enough or proper to form very deep trainable network.

We propose residual non-local attention learning to solve those problems.

How to train deep image restoration network with non-local mixed attention remains unclear.

Here we only consider the trunk and mask branches, and residual connection with them ( Figure 2 ).

We focus on obtaining non-local attention information from the input feature x. It should be noted that one form of attention residual learning was proposed in , whose formulation is DISPLAYFORM0 We find that this form of attention learning is not suitable for image restoration tasks.

This is mainly because Eq. 7 is more suitable for high-level vision tasks (e.g., image classification), where low-level features are not preserved too much.

However, low-level features are more important for image restoration.

As a result, we propose a simple yet more suitable residual attention learning method by introducing input feature x directly.

We compute its output DISPLAYFORM1 where H trunk (x) and H mask (x) denote the functions of trunk and mask branches respectively.

Such residual learning tends to preserve more low-level features and allows us to form very deep networks for high-quality image restoration tasks with stronger representation ability.3.4 IMPLEMENTATION DETAILS Now, we specify the implementation details of our proposed RNAN.

We use 10 residual local and non-local attention blocks (2 non-local one).

In each residual (non-)local block, we set q, t, m = 2, 2, 1.

We set 3×3 as the size of all convolutional layers except for those in non-local block and convolutional layer before sigmoid function, where the kernel size is 1×1.

Features in RBs have 64 filters, except for that in the non-local block (see FIG0 , where C = 32.

In each training batch, 16 low-quality (LQ) patches with the size of 48 × 48 are extracted as inputs.

Our model is trained by ADAM optimizer BID19 with β 1 = 0.9, β 2 = 0.999, and = 10 −8 .

The initial learning rate is set to 10 −4 and then decreases to half every 2 × 10 5 iterations of back-propagation.

We use PyTorch BID29 to implement our models with a Titan Xp GPU.

We apply our proposed RNAN to three classical image restoration tasks: image denoising, demosaicing, and compression artifacts reduction.

For image denoising and demosaicing, we follow the same setting as IRCNN (Zhang et al., 2017b) .

For image compression artifacts reduction, we follow the same setting as ARCNN BID10 .

We use 800 training images in DIV2K to train all of our models.

For each task, we use commonly used dataset for testing and report PSNR and/or SSIM BID38 to evaluate the results of each method.

More results are shown in Appendix A. Non-local Mixed Attention.

In cases 1, all the mask branches and non-local blocks are removed.

In case 4, we enable non-local mixed attention with same block number as in case 1.

The positive effect of non-local mixed attention is demonstrated by its obvious performance improvement.

Mask branch.

We also learn that mask branch contributes to performance improvement, no matter non-local blocks are used (cases 3 and 4) or not (cases 1 and 2).

This's mainly because mask branch provides informative attention to the network, gaining better representational ability.

Non-local block.

Non-local block also contributes to the network ability obviously, no matter mask branches are used (cases 2 and 4) or not (cases 1 and 3).

With non-local information from low-level and high-level features, RNAN performs better image restoration.

Block Number.

When comparing cases 2, 4, and 7, we learn that more non-local blocks achieve better results.

However, the introduction of non-local block consumes much time.

So we use 2 non-local blocks by considering low-and high-level features.

When RNAB number is fixed in cases 5 and 7 or cases 6 and 8, performance also benefits from more RABs.

We compare our RNAN with state-of-the-art denoising methods: BM3D BID7 , CBM3D BID6 , TNRD BID5 , RED BID25 , DnCNN (Zhang et al., 2017a) , MemNet , IRCNN (Zhang et al., 2017b) , and FFDNet (Zhang et al., 2017c) .

Kodak24 (http://r0k.us/graphics/kodak/), BSD68 BID26 , and Urban100 BID16 are used for color and gray-scale image denoising.

AWGN noises of different levels (e.g., 10, 30, 50, and 70) are added to clean images.

Quantitative results are shown in TAB1 .

As we can see that our proposed RNAN achieves the best results on all the datasets with all noise levels.

Our proposed non-local attention covers the information from the whole image, which should be effective for heavy image denoising.

To demonstrate this analysis, we take noise level σ = 70 as an example.

We can see that our proposed RNAN achieves 0.48, 0.30, and 1.06 dB PSNR gains over the second best method FFDNet.

This comparison strongly shows the effectiveness of our proposed non-local mixed attention.

We also show visual results in Figures 4 and 5.

With the learned non-local mixed attention, RNAN treats different image parts distinctively, alleviating over-smoothing artifacts obviously.

Following the same setting in IRCNN (Zhang et al., 2017b) , we compare image demosaicing results with those of IRCNN on McMaster (Zhang et al., 2017b) , Kodak24, BSD68, and Urban100.

Since IRCNN has been one of the best methods for image demosaicing and limited space, we only compare with IRCNN in TAB3 .

As we can see, mosaiced images have very poor quality, resulting in very low PSNR and SSIM values.

IRCNN can enhance the low-quality images and achieve relatively high values of PSNR and SSIM.

Our RNAN can still make significant improvements over IRCNN.

Using local and non-local attention, our RNAN can better handle the degradation situation.

Visual results are shown in Figure 6 .

Although IRCNN can remove mosaicing effect greatly, there're still some artifacts in its results (e.g., blocking artifacts in 'img 026').

However, RNAN recovers more faithful color and alliciates blocking artifacts.

We further apply our RNAN to reduce image compression artifacts.

We compare our RNAN with SA-DCT BID12 , ARCNN BID10 , TNRD BID5 , and DnCNN (Zhang et al., 2017a) .

We apply the standard JPEG compression scheme to obtain the compressed images by following BID10 .

Four JPEG quality settings q = 10, 20, 30, 40 are used in Matlab JPEG encoder.

Here, we only focus on the restoration of Y channel (in YCbCr space) to keep fair comparison with other methods.

We use the same datasets LIVE1 BID30 and Classic5 BID12 in ARCNN and report PSNR/SSIM values in TAB4 .

As we can see, our RNAN achieves the best PSNR and SSIM values on LIVE1 and Classic5 with all JPEG qualities.

We further shown visual comparisons in Figure 7 .

We provide comparisons under very low image quality (q=10).

The blocking artifacts can be removed to some degree, but ARCNN, TNRD, and DnCNN would also over-smooth some structures.

RNAN obtains more details with consistent structures by considering non-local mixed attention.

We further compare our RNAN with state-of-the-art SR methods: EDSR , SR-MDNF BID43 , D-DBPN , and RCAN BID40 .

Similar to BID41 , we also introduce self-ensemble strategy to further improve our RNAN and denote the self-ensembled one as RNAN+.As shown in TAB5 , our RNAN+ achieves the second best performance among benchmark datasets: Set5 BID2 ), Set14 (Zeyde et al., 2010 , B100 BID26 ), Urban100 BID16 , and Manga109 BID27 .

Even without self-ensemble, our RNAN achieves third best results in most cases.

Such improvements are notable, because the parameter number of RNAN is 7.5 M, far smaller than 43 M in EDSR and 16 M in RCAN.

The network depth of our RNAN (about 120 convolutional layers) is also far shallower than that of RCAN, which has about 400 convolutional layers.

It indicates that non-local attention make better use of main network, saving much network parameter.

In FIG1 , we conduct image SR (×4) with several state-of-the-art methods.

We can see that our RNAN obtains better visually pleasing results with finer structures.

These comparisons further demonstrate the effectiveness of our proposed RNAN with the usage of non-local mixed attention.

We also compare parameters, running time, and performance based on color image denoising in TAB6 .

PSNR values are tested on Urban100 (σ=50).

RNAN with 10 blocks achieves the best performance with the highest parameter number, which can be reduced to only 2 blocks and obtains second best performance.

Here, we report running time for reference, because the time is related to implementation platform and code.

All the results shown in the main paper are based on DIV2K training data.

Here, we retrain our RNAN on small training sets for 5 tasks.

In TAB7 , for each task, we only refer the second best method from our main paper to compare.

For training data, we use BSD400 BID26 for color/gray-scale image denoising and demosaicing.

We use 91 images in BID39 and 200 images in BID26 (denoted as SR291) for image compression artifacts reduction and super-resolution.

FFDNet used BSD400 BID26 , 400 images from ImageNet BID8 , and 4,744 images in Waterloo Exploration Database BID24 .

Here, 'BSD400+' is used to denote 'BSD400+ImageNet400+WED4744'.

According to TAB7 , we use the same or even smaller training set for our RNAN and obtain better results for 5 tasks.

These experiments demonstrate the effectiveness of our RNAN for general image restoration tasks.

Color and Gray Image Denoising.

We show color and gray-scale image denoising comparisons in Figures 9 and 10 respectively.

We can see that our RNAN recovers shaper edges.

Unlike most of other methods, which over-smooth some details (e.g., tiny lines), RNAN can reduce noise and maintain more details.

With the learned non-local mixed attention, RNAN treats different image parts distinctively, alleviating over-smoothing artifacts obviously.

<|TLDR|>

@highlight

New state-of-the-art framework for image restoration

@highlight

The paper proposes a convolutional neural network architecture that includes blocks for local and non-local attention mechanisms, which are claimed to be responsible for achieving excellent results in four image restoration applications.

@highlight

This paper proposes a residual non-local attention network for image restoration