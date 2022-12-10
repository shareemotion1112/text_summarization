State-of-the-art face super-resolution methods employ deep convolutional neural networks to learn a mapping between low- and high-resolution facial patterns by exploring local appearance knowledge.

However, most of these methods do not well exploit facial structures and identity information, and struggle to deal with facial images that exhibit large pose variation and misalignment.

In this paper, we propose a novel face super-resolution method that explicitly incorporates 3D facial priors which grasp the sharp facial structures.

Firstly, the 3D face rendering branch is set up to obtain 3D priors of salient facial structures and identity knowledge.

Secondly, the Spatial Attention Mechanism is used to better exploit this hierarchical information (i.e. intensity similarity, 3D facial structure, identity content) for the super-resolution problem.

Extensive experiments demonstrate that the proposed algorithm achieves superior face super-resolution results and outperforms the state-of-the-art.

Face images provide crucial clues for human observation as well as computer analysis (Fasel & Luettinb, 2003; Zhao et al., 2003) .

However, the performance of most face image tasks, such as face recognition and facial emotion detection (Han et al., 2018; Thies et al., 2016) , degrades dramatically when the resolution of a facial image is relatively low.

Consequently, face super-resolution, also known as face hallucination, was coined to restore a low-resolution face image to its high-resolution counterpart.

A multitude of deep learning methods (Zhou & Fan, 2015; Yu & Porikli, 2016; Zhu et al., 2016; Cao et al., 2017; Dahl et al., 2017a; Yu et al., 2018b) have been successfully applied in face Super-Resolution (SR) problems and achieve state-of-the-art results.

But super-resolving arbitrary facial images, especially at high magnification factors, is still an open and challenging problem due to the ill-posed nature of the SR problem and the difficulty in learning and integrating strong priors into a face hallucination model.

Some researches (Grm et al., 2018; Yu et al., 2018a; Ren et al., 2019) on exploiting the face priors to assist neural networks to capture more facial details have been proposed recently.

A face hallucination model incorporating identity priors is presented in Grm et al. (2018) .

But the identity prior is extracted only from the multi-scale up-sampling results in the training procedure and therefore cannot provide enough extra priors to guide the network to achieve a better result.

Yu et al. (2018a) employ facial component heatmaps to encourage the upsampling stream to generate super-resolved faces with higher-quality details, especially for large pose variations.

Although heatmaps can provide global component regions, it cannot learn the reconstruction of detailed edges, illumination or expression priors.

Besides, all of these aforementioned face SR approaches ignore facial structure and identity recovery.

In contrast to previous methods, we propose a novel face super-resolution method that embeds 3D face structures and identity priors.

Firstly, a deep 3D face reconstruction branch is set up to explicitly obtain 3D face render priors which facilitate the face super-resolution branch.

Specifically, the 3D face render prior is generated by the ResNet-50 network (He et al., 2016) .

It contains rich hierarchical information, such as low-level (e.g., sharp edge, illumination) and perception level (e.g., identity).

The Spatial Attention Mechanism is proposed here to adaptively integrate the 3D facial prior into the network.

Specifically, we employ the Spatial Feature Transform (SFT) (Wang et al., 2018) to generate affine transformation parameters for spatial feature modulation.

Afterwards, it encourages the network to learn the spatial interdepenencies of features between 3D facial priors and input images after adding the attention module into the network.

The main contributions of this paper are: 1.

A novel face SR model is proposed by explicitly exploiting facial structure in the form of facial-prior estimation.

The estimated 3D facial prior provides not only spatial information of facial components but also their visibility information, which are ignored by the pixel-level content.

2.

We propose a feature-fusion-based network to better extract and integrate the face rendered priors by employing the Spatial Attention Mechanism (SAM).

3.

We qualitatively and quantitatively explore multi-scale face super-resolution, especially at very low input resolutions.

The proposed network achieves better SR criteria and superior visual quality compared to state-of-the-art face SR methods.

Face hallucination relates closely to the natural image super-resolution problem.

Thus, in this section, we discuss recent research on super-resolution and face hallucination to illustrate the necessary context for our work.

Recently, neural networks have demonstrated a remarkable capability to improve SR results.

Since the pioneering network can learn to map the relationship between LR and HR (Dong et al., 2016a) , a lot of CNN architectures have been proposed for SR (Dong et al., 2016b; Shi et al., 2016; Lai et al., 2017; Haris et al., 2018; Tai et al., 2017) .

Most of the existing high-performance SR networks have residual blocks (Jiwon to go deeper in the network architecture, and achieve better performance.

EDSR (Lim et al., 2017) improves the performance by removing unnecessary batch normalization layers in residual blocks.

A residual dense network (RDN) (Zhang et al., 2018a) was proposed to exploit the hierarchical features from all the convolutional layers.

Zhang et al. (2018b) proposed the very deep residual channel attention networks(RCAN) to discard abundant low-frequency information which hinders the representational ability of CNNs.

Wang et al. (2018) used a spatial feature transform layer to introduce the semantic prior as an additional input of SR network.

presented a wavelet-based CNN approach that can ultra-resolve a very low resolution face image in a unified framework.

However, these networks require a lot of time to train the large-scale parameters to obtain good results.

In our work, we largely decrease the training parameters, but still achieve the superior performance in SR criteria (SSIM and PSNR) and visible quality.

Facial Prior Knowledge: Exploiting facial priors in face hallucination, such as spatial configuration of facial components, is the key factor that differentiates it from generic super-resolution tasks.

There are some face SR methods that use facial prior knowledge to better super-resolve LR faces.

Wang & Tang (2005) learned subspaces from LR and HR face images respectively, and then reconstructed an HR output from the PCA coefficients of the LR input.

Liu et al. (2007) set up a Markov Random Field (MRF) to reduce ghosting artifacts because of the misalignments in LR images.

These methods are prone to generate severe artifacts, especially in large pose variations and misalignments in LR images.

Yu & Porikli (2017b) interweaved multiple spatial transformer networks (Jaderberg et al., 2015) with the deconvolutional layers to handle unaligned LR faces.

Dahl et al. (2017b) leveraged the framework of PixelCNN (Van Den Oord et al., 2016) to super-resolve very low-resolution faces.

Zhu et al. (2016) presented a cascade bi-network, dubbed CBN, to localize LR facial components first and then upsample the facial components; however, CBN may produce ghosting faces when localization errors occur.

Recently, Yu et al. (2018a) used a multi-task convolutional neural network (CNN) to incorporate structural information of faces.

Grm et al. (2018) built a face recognition model that acts as identity priors for the super-resolution network during training.

In our paper, we used the 3D face reconstruction branch to extract the facial structure, detailed edges, illumination, and identity priors.

Furthermore, we recover these priors in an explicit way.

The 3D shapes of facial images can be restored from unconstrained 2d images by the 3D face reconstruction.

In this paper, we employ the 3D Morphable Model (3DMM) (Blanz & Vetter, 1999; Booth et al., 2016) based on the fusion of parametric descriptions of face attributes (e.g., gender, identity, and distinctiveness) to reconstruct the 3D facial priors.

The reconstructed face will inherit the facial features and present the clear and sharp facial components.

Given a low-resolution facial image, the 3D rendering branch aims to extract the 3D face coefficients based on the 3D Morphable Model (3DMM).

The high-resolution face rendered image is generated after obtaining the 3D coefficients and regarded as the high-resolution facial priors which facilitate the face super-resolution.

The 3D coefficients contain abundant hierarchical knowledge, such as identity, facial expression, texture, illumination, and face pose.

The proposed face super-resolution framework is presented in Figure 2 , and it consists of two branches: the 3D rendering network to extract the facial prior and the Spatial Attention Mechanism aiming to exploit the prior for the face super-resolution problem.

It is still a challenge for state-of-the-art edge prediction methods to acquire very sharp facial structures from low-resolution images.

Therefore, a 3DMM-based model is proposed to localize the precise facial structure by generating the 3D facial images which are constructed by the 3D coefficient vector.

Besides, there exist large face pose variations, such as inplane and out-of-plane rotations.

A large amount of data is needed to learn the representative features varying with the facial poses.

To address this problem, an inspiration came from the idea that the 3DMM coefficients can analytically model the pose variation with a simple math derivation (Booth et al., 2016; and does not require a large training set, we utilize a face rendering network based on ResNet-50 to regress a face coefficient vector.

The output of the ResNet-50 is the representative feature vector of x = (α, β, δ, γ, ζ) ∈ R 239 , where α ∈ R 80 , β ∈ R 64 , δ ∈ R 80 , γ ∈ R 9 , and ζ ∈ R 6 represent the identity, facial expression, texture, illumination, and face pose , respectively.

According to the Morphable model (Blanz & Vetter, 1999) , we transform the face coefficients to a 3D shape S and texture T of the face image as

and

where S and T are the average values of the S and T. Besides, B t , B id and B exp denote the base vector of texture, identity, and expression calculated by the PCA method.

A modified L 2 based loss function for the 3D face reconstruction is presented based on a paired training set

where j is the paired image index, and L is the total number of training pairs.

i and M denote the pixel index and face region, respectively.

A, I and B represent the skin color based attention mask, the sharp image, and the up-sampling of low-resolution image, respectively.

R(B i j (x)) denotes the reconstructed face image based on the learned face vector by the ResNet-50 network.

The proposed face super-resolution architecture.

Our model consists of two branches: the top block is a ResNet-50 Network to extract the 3D facial coefficients and restore a sharp face rendered structure.

The bottom block is dedicated to face super-resolution guided by the facial coefficients and rendered sharp face structures which are concatenated by the Spatial Feature Transform.

Given the LR images, the generated 3D face rendered reconstructions are shown in Figure 1 .

The rendered face predictions contain the clear spatial knowledge and good visual quality of facial components which are very close to the information of the ground-truths.

The 3D priors grasp very well the pose variations and skin colour, and further embed pose variations into the SR networks which improve the accuracy and stability in face images with large pose variations.

Therefore, we concatenate the reconstructed face image as an additional feature in the SR network.

The face expression, identity, texture, illumination, and face pose are transformed into four feature maps and fed into the spatial feature transform block of the SR network.

As shown in Figure 2 , our Spatial Attention Mechanism aims to exploit the 3D face rendered priors which grasp the precise locations of face components and the facial identity.

In order to explore the interdependence and correlation of priors and input images between channels, the attention block is added into the Spatial Attention Mechanism.

The proposed network, also named the Spatial Attention Mechanism (SAM), consists of three simple parts: a spatial transform block, an attention block, and an upscale module.

We import the 3D face priors into the Spatial Attention Transform Block after a convolutional layer.

The 3D face priors consist of two parts: one directly from the rendered face images (as the RGB input), and the other from the feature transformation of the coefficient parameters.

The feature transformation procedure is described as follows: firstly, the coefficients of (identity, expression, texture, and the fusion of illumination and face pose) are reshaped to a matrix by setting extra elements to zeros.

Afterwards, it is expanded to the same size as the LR images by zero-padding, and then scaled to the interval [0,1].

Finally, the coefficient features are concatenated with the priors from the rendered face images.

The Spatial Feature Transform (SFT) learns a mapping function Θ that provides a modulation parameter pair (µ, ν) according to the priors ψ, such as segmentation probability.

Instead, the 3D face priors are taken as the input.

The outputs of the SFT layer are adaptively controlled by the modulation parameter pair by way of applying an affine transformation spatially to each intermediate feature map.

Specifically, the intermediate transformation parameters (µ, ν) are derived from the priors ψ by a mapping function as:

and then

where N denotes the SR network, and θ represents trainable parameters of the network.

The intermediate feature maps are modified by scaling and shifting feature maps according to the transformation parameters:

where F denotes the feature maps, and ⊗ is referred to element-wise multiplication.

At this step, the SFT layer implements the spatial-wise transformation.

Attention mechanism can be viewed as a guide to bias the allocation of available processing resources towards the most informative components as input (Hu et al., 2017) .

Consequently, the channel module is presented to explore the most informative components and the interdependency between the channels.

The attention module is composed of a series of residual channel attention blocks (RCAB) shown in Figure 2 .

Inspired by the integration of channel attention and residual blocks, we ensemble a series of residual channel attention blocks.

For the b-th block, the output F b of RCAB is obtained by:

where C b denotes the channel attention function.

F b−1 is the block's input, and X b is calculated by two stacked convolutional layers.

In order to evaluate the performance of our priors and algorithms, we compare them with the startof-art methods qualitatively and quantitatively. .

We use open-resource implementations from the authors and train all the networks on the same dataset for a fair comparison.

We propose two models: first is the VDSR+ which is the basic VDSR model embedded with the 3D facial prior as extra RGB channel information and the other is our SR network incorporating facial priors by the Spatial Attention Mechanism (SAM).

The implementation code will be made available to the public.

More results are shown in the supplementary material.

CelebA (Liu et al., 2015) and Menpo (Zafeiriou et al., 2017) dataset are used to verify the performance of the algorithm.

The training phase uses 162,080 images from the CelebA dataset.

In the testing phase, 40,519 images from the CelebA test set are used along with the large-pose-variation test set from the Menpo dataset.

The every facial pose test set of Menpo (left, right and semi-frontal) contains 1000 images, respectively.

The HR ground-truth images are obtained by center-cropping the facial images and then resizing them to the 128×128 pixels.

The LR face images are generated by downsampling HR ground-truths to 32×32 pixels (4 scale) and 16×16 pixels (8 scale).

In our network, the ADAM optimizer is used with a batch size of 64 for training, and input images are center-cropped as RGB channels.

The initial learning rate is 0.0002 and is divided by 2 every 50 epochs.

The whole training process takes 2 days with an NVIDIA Titan X GPU.

Quantitative evaluation of the network using PSNR and the structural similarity (SSIM) scores for the CelebA test set are listed in Table 1 .

Furthermore, in order to analyze the proposed methods' performance and stability regarding to large face pose variations, three case results corresponding to different face poses (left, right, and semifrontal) of the Menpo test data are listed in Table 2 .

CelebA Test: Ours (VDSR+) achieves significantly better results (1 dB higher than the remaining best method and 2 db higher than the basic VDSR method in x8 SR) even for the large-scale parameter methods, such as RDN and RCAN.

But it does perform slightly worse than ours (SAM).

It should be noted that ours (VDSR+) is the same as VDSR except for the extra 3D face priors as the Figure 5: Visual comparison with state-of-the-art methods(×8).

The results from the proposed method have less visual artifacts and more details on key face components (e.g., eyes, mouth, and nose) RGB channel inputs.

It indicates that the 3D priors make a great contribution to the performance improvement (average 1.6 db improvement) of face super-resolution.

Menpo Test: To verify the effectiveness and stability of face priors and our proposed network towards large pose variations, the PSNR and SSIM results of face poses are listed in Table 2 .

While ours (SAM) is the best method superior than others, VDSR+ achieves 1.8db improvement compared with the basic VDSR method in the magnification factors (×4).

Super-resolution: The qualitative results of our methods at different magnifications (×4 and ×8) are shown respectively in Figures 3 and 4 .

It can be observed that our proposed method recovers clearer faces with finer component details (e.g., nose and eyes).

Artifacts:

The outputs of most methods (e.g., RCAN, RDN, and Wavelet-SRNet) contain some artifacts around facial components, such as the eyes, nose, and mouth shown in Figure 5 .

After adding the rendered face priors, ours results show clear and sharp facial structures without any ghosting artifacts.

It illustrates that the proposed 3D priors can help the network understand the spatial location and the entire face structure.

Ablation Study: In this section, we conduct an ablation study to demonstrate the effectiveness of each module.

We compare the proposed network with and without using the rendered 3D face priors and the Spatial Attention Mechanism (SAM) in terms of PSNR and SSIM on the test data.

As shown in Figure 6 (b, f), the baseline method without rendered faces and SAM tends to generate blurry faces that cannot capture sharp edges.

Figure 6 (c and g) shows clearer and sharper facial structures after adding the rendered priors.

By using SAM, the visual quality is further improved in Figure 6 (d and h).

The quantitative comparisons between (VDSR, our VDSR+, and our SAM) in Tables 1 and 2 also illustrate the effectiveness of the rendered priors and the Spatial Attention Mechanism.

Model Size Analysis:

Figure 7 shows comparisons of model size and performance.

Our networks, VDSR+ and SAM, embedded with 3D priors are more lightweight while still achieving the best performance even compared with other state-of-the-art methods (e.g., RCAN and RDN) with a larger scale of parameters.

In this paper, we proposed a novel network that incorporates 3D facial priors of rendered faces and identity knowledge.

The 3D rendered branch utilizes the face rendering loss to encourage a highquality guided image providing clear spatial locations of facial components and other hierarchical information (i.e., expression, illumination, and face pose).

To well exploit 3D priors and consider the channel correlation between priors and inputs, the Spatial Attention Mechanism is presented by employing the Spatial Feature Transform and Attention block.

The comprehensive experimental results have demonstrated that the proposed method can deliver the better performance and largely decrease artifacts in comparison with the state-of-the-art methods by using significantly fewer parameters.

Semi-Frontal Facial Pose Visualization:

For the semi-frontal pose, the SR results of RCAN, RDN and Wavelet-SRNet have a lot of artifacts around facial components (e.g., eyes, teeth, nose, mouth).

Fortunately, after incorporating the rendered face priors, it largely avoids the appearance of ghosting artifacts, seen in Figure.8 Left Facial Pose Visualization:

For the left pose, the high-resolution results of the proposed method perform much better.

Ours (VDSR+) results which exploiting the 3D facial priors can grasp the facial structure knowledge and restore the high-resolution facial components (e.g. mouth) much closer to the ground-truth compared with the basic VDSR method without priors.

Right Facial Pose Visualization:

For the right pose, the high-resolution results of the proposed method are still the best.

Adding the facial structure priors can help network to learn the location of facial components even for the large pose variation.

High Magnification Factor × 8 Visualization: It is still a challenge to generate the sharp superresolution images for a large magnification factor (×8).

The 3D rendered facial priors provide extra facial structure knowledge that are crucial for SR problems.

As shown in Figure 12 and 13, the proposed method generates a high visible quality of SR images even for the large magnification factor.

Learning Curves with Different Ablation Configurations:To verify the effectiveness of 3D facial structure priors, we design the three different configurations (w/o 3D priors, w/o Spatial Attention Mechanism): baseline methods (i.e., VDSR, SRCNN); baseline incorporating 3D facial priors (i.e., VDSR+,SRCNN+); the method using the Spatial Attention Mechanism and 3D priors (our proposed method: +priors and +SAM).

The learning curves of each configuration are plotted to show the effectiveness of the each block.

The priors are easy to insert into any network without increasing any parameters, but largely improve the accuracy and the convergence of the algorithms shown in Figure 14 .

Quantitative Results with Different Ablation Configurations: As shown in Table 3 , each block boosts the accuracy of baseline algorithms: the average performance improvement stemming from 3D facial priors and from Spatial Attention Mechanism are 1.6db and 0.57db, respectively.

Qualitative Evaluation with different ablation configurations: The baseline incorporated with the facial rendered priors tends to avoid some artifacts around the key facial components and generate more sharp edges compared with the basic baseline method without the facial priors.

By adding the Spatial Attention Mechanism, it could help the network better exploit the priors and is easier to generate more sharp facial structures, shown in Figure 15 .

<|TLDR|>

@highlight

We propose a novel face super resolution method that explicitly incorporates 3D facial priors which grasp the sharp facial structures.