Deep image prior (DIP), which utilizes a deep convolutional network (ConvNet) structure itself as an image prior, has attracted huge attentions in computer vision community.

It empirically shows the effectiveness of ConvNet structure for various image restoration applications.

However, why the DIP works so well is still unknown, and why convolution operation is essential for image reconstruction or enhancement is not very clear.

In this study, we tackle these questions.

The proposed approach is dividing the convolution into ``delay-embedding'' and ``transformation (\ie encoder-decoder)'', and proposing a simple, but essential, image/tensor modeling method which is closely related to dynamical systems and self-similarity.

The proposed method named as manifold modeling in embedded space (MMES) is implemented by using a novel denoising-auto-encoder in combination with multi-way delay-embedding transform.

In spite of its simplicity, the image/tensor completion and super-resolution results of MMES are quite similar even competitive to DIP in our extensive experiments, and these results would help us for reinterpreting/characterizing the DIP from a perspective of ``low-dimensional patch-manifold prior''.

The most important piece of information for image/tensor restoration would be the "prior" which usually converts the optimization problems from ill-posed to well-posed, and/or gives some robustness for specific noises and outliers.

Many priors were studied in computer science problems such as low-rank representation (Pearson, 1901; Hotelling, 1933; Hitchcock, 1927; Tucker, 1966) , smoothness (Grimson, 1981; Poggio et al., 1985; Li, 1994) , sparseness (Tibshirani, 1996) , non-negativity (Lee & Seung, 1999; Cichocki et al., 2009) , statistical independence (Hyvarinen et al., 2004) , and so on.

Particularly in today's computer vision problems, total variation (TV) (Guichard & Malgouyres, 1998; Vogel & Oman, 1998) , low-rank representation (Liu et al., 2013; Ji et al., 2010; Zhao et al., 2015; Wang et al., 2017) , and non-local similarity (Buades et al., 2005; Dabov et al., 2007) priors are often used for image modeling.

These priors can be obtained by analyzing basic properties of natural images, and categorized as "unsupervised image modeling".

By contrast, the deep image prior (DIP) (Ulyanov et al., 2018) has been come from a part of "supervised" or "data-driven" image modeling framework (i.e., deep learning) although the DIP itself is one of the state-of-the-art unsupervised image restoration methods.

The method of DIP can be simply explained to only optimize an untrained (i.e., randomly initialized) fully convolutional generator network (ConvNet) for minimizing squares loss between its generated image and an observed image (e.g., noisy image), and stop the optimization before the overfitting.

Ulyanov et al. (2018) explained the reason why a high-capacity ConvNet can be used as a prior by the following statement: Network resists "bad" solutions and descends much more quickly towards naturally-looking images, and its phenomenon of "impedance of ConvNet" was confirmed by toy experiments.

However, most researchers could not be fully convinced from only above explanation because it is just a part of whole.

One of the essential questions is why is it ConvNet? or in more practical perspective, to explain what is "priors in DIP" with simple and clear words (like smoothness, sparseness, low-rank etc) is very important.

In this study, we tackle the question why ConvNet is essential as an image prior, and try to translate the "deep image prior" with words.

For this purpose, we divide the convolution operation into "embedding" and "transformation" (see Fig. 9 in Appendix).

Here, the "embedding" stands for delay/shift-embedding (i.e., Hankelization) which is a copy/duplication operation of image-patches by sliding window of patch size (??, ?? ).

The embedding/Hankelization is a preprocessing to capture the delay/shift-invariant feature (e.g., non-local similarity) of signals/images.

This "transformation" is basically linear transformation in a simple convolution operation, and it also indicates some nonlinear transformation from the ConvNet perspective.

To simplify the complicated "encoder-decoder" structure of ConvNet used in DIP, we consider the following network structure: Embedding H (linear), encoding ?? r (non-linear), decoding ?? r (non-linear), and backward embedding H ??? (linear) (see Fig. 1 ).

Note that its encoder-decoder part (?? r , ?? r ) is just a simple multi-layer perceptron along the filter domain (i.e., manifold learning), and it is sandwitched between forward and backward embedding (H, H ??? ).

Hence, the proposed network can be characterized by Manifold Modeling in Embedded Space (MMES).

The proposed MMES is designed as simple as possible while keeping a essential ConvNet structure.

Some parameters ?? and r in MMES are corresponded with a kernel size and a filter size in ConvNet.

When we set the horizontal dimension of hidden tensor L with r, each ?? 2 -dimensional fiber in H, which is a vectorization of each (??, ?? )-patch of an input image, is encoded into r-dimensional space.

Note that the volume of hidden tensor L looks to be larger than that of input/output image, but representation ability of L is much lower than input/output image space since the first/last tensor (H,H ) must have Hankel structure (i.e., its representation ability is equivalent to image) and the hidden tensor L is reduced to lower dimensions from H. Here, we assume r < ?? 2 , and its lowdimensionality indicates the existence of similar (??, ?? )-patches (i.e., self-similarity) in the image, and it would provide some "impedance" which passes self-similar patches and resist/ignore others.

Each fiber of Hidden tensor L represents a coordinate on the patch-manifold of image.

It should be noted that the MMES network is a special case of deep neural networks.

In fact, the proposed MMES can be considered as a new kind of auto-encoder (AE) in which convolution operations have been replaced by Hankelization in pre-processing and post-processing.

Compared with ConvNet, the forward and backward embedding operations can be implemented by convolution and transposed convolution with one-hot-filters (see Fig. 12 in Appendix for details).

Note that the encoder-decoder part can be implemented by multiple convolution layers with kernel size (1,1) and non-linear activations.

In our model, we do not use convolution explicitly but just do linear transform and non-linear activation for "filter-domain" (i.e., horizontal axis of tensors in Fig. 1 ).

The contributions in this study can be summarized as follow: (1) A new and simple approach of image/tensor modeling is proposed which translates the ConvNet, (2) effectiveness of the proposed method and similarity to the DIP are demonstrated in experiments, and (3) most importantly, there is a prospect for interpreting/characterizing the DIP as "low-dimensional patch-manifold prior".

Note that the idea of low-dimensional patch manifold itself has been proposed by Peyre (2009) and Osher et al. (2017) .

Peyre had firstly formulated the patch manifold model of natural images and solve it by dictionary learning and manifold pursuit.

Osher et al. formulated the regularization function to minimize dimension of patch manifold, and solved Laplace-Beltrami equation by point integral method.

In comparison with these studies, we decrease the dimension of patch-manifold by utilizing AE shown in Fig. 1 .

A related technique, low-rank tensor modeling in embedded space, has been studied recently by Yokota et al. (2018) .

However, the modeling approaches here are different: multi-linear vs nonlinear manifold.

Thus, our study would be interpreted as manifold version of (Yokota et al., 2018) in a perspective of tensor completion methods.

Note that Yokota et al. (2018) applied their model for only tensor completion task.

By contrast, we investigate here tensor completion, super-resolution, and deconvolution tasks.

Another related work is devoted to group sparse representation (GSR) (Zhang et al., 2014a) .

The GSR is roughly characterized as a combination of similar patch-grouping and sparse modeling which is similar to the combination of embedding and manifold-modeling.

However, the computational cost of similar patch-grouping is obviously higher than embedding, and this task is naturally included in manifold learning.

The main difference between above studies and our is the motivation: Essential and simple image modeling which can translate the ConvNet/DIP.

The proposed MMES has many connections with ConvNet/DIP such as embedding, non-linear mapping, and the training with noise.

From a perspective of DIP, there are several related works.

First, the deep geometric prior (Williams et al., 2019 ) utilises a good properties of a multi-layer perceptron for shape reconstruction problem which efficiently learn a smooth function from 2D space to 3D space.

It helps us to understand DIP from a perspective of manifold learning.

For example, it can be used for gray scale image reconstruction if an image is regarded as point could in 3D space (i, j, X ij ).

However, this may not provide the good image reconstruction like DIP, because it just smoothly interpolates a point cloud by surface like a Volonoi interpolation.

Especially it can not provide a property of self-similarity in natural image.

Second, deep decoder (Heckel & Hand, 2018) reconstructs natural images from noises by nonconvolutional networks which consists of linear channel/color transform, ReLU, channel/color normalization, and upsampling layers.

In contrast that DIP uses over-parameterized network, deep decoder uses under-parameterized network and shows its ability of image reconstruction.

Although deep decoder is a non-convolutional network, Authors emphasize the closed relationship between convolutional layers in DIP and upsampling layers in deep decoder.

In this literature, Authors described "If there is no upsampling layer, then there is no notion of locality in the resultant image" in deep decoder.

It implies the "locality" is the essence of image model, and the convolution/upsampling layer provides it.

Furthermore, the deep decoder has a close relationship with our MMES.

Note that the MMES is originally/essentially has only decoder and inverse MDT (see Eq. (3)), and the encoder is just used for satisfying Hankel structure.

The decoder and inverse MDT in our MMES are respectively corresponding linear operation and upsampling layer in deep decoder.

Moreover, concept of under-parameterization is also similar to our MMES.

From this, we can say the essence of image model is the "locality", and its locality can be provided by "convolution", "upsampling", or "delay-embedding".

This is why the image restoration from single image with deep convolutional networks has highly attentions which are called by zero-shot learning, internal learning, or self-supervised learning (Shocher et al., 2018; Lehtinen et al., 2018; Krull et al., 2019; Batson & Royer, 2019; Xu et al., 2019; Cha et al., 2019; Laine et al., 2019) .

Recently, two generative models: SinGAN (Shaham et al., 2019) and InGAN (Shocher et al., 2019) learned from only a single image, have been proposed.

Key concept of both papers is to impose the constraint for local patches of image to be natural.

From a perspective of the constraint for local patches of image, our MMES has closed relationship with these works.

However, we explicitly impose a low-dimensional manifold constraint for local patches rather than adversarial training with patch discriminators.

Here, on the contrary to Section 1, we start to explain the proposed method from the concept of MMES, and we systematically derive the MMES structure from it.

Conceptually, the proposed tensor reconstruction method can be formulated by minimize

where Y ??? R J1??J2??????????J N is an observed corrupted tensor, X ??? R I1??I2??????????I N is an estimated tensor, F : R I1??I2??????????I N ??? R J1??J2??????????J N is a linear operator which represents the observation system, H : R I1??I2??????????I N ??? R D??T is padding and Hankelization operator with sliding window of size (?? 1 , ?? 2 , ..., ?? N ), and we impose each column of matrix H can be sampled from an r-dimensional manifold M r in D-dimensional Euclid space (see Appendix B for details).

We have r ??? D. For simplicity, we putted D := n ?? n and T := n (I n +?? n ???1).

For tensor completion task, F := P ??? is a projection operator onto support set ??? so that the missing elements are set to be zero.

For superresolution task, F is a down-sampling operator of images/tensors.

For deconvolution task, F is a convolution operator with some blur kernels.

Fig. 2 shows the concept of proposed manifold modeling in case of image inpainting (i.e., N = 2).

We minimize the distance between observation Y and reconstruction X with its support ???, and all patches in X should be included in some restricted manifold M r .

In other words, X is represented by the patch-manifold, and the property of the patch-manifold can be image priors.

For example, low dimensionality of patch-manifold restricts the non-local similarity of images/tensors, and it would be related with "impedance" in DIP.

We model X indirectly by designing the properties of patch-manifold M r .

We consider an AE to define the r-dimensional manifold M r in ( n ?? n )-dimensional Euclidean space as follows:

where

.

Note that, in general, the use of AE models is a widely accepted approach for manifold learning (Hinton & Salakhutdinov, 2006) .

The properties of the manifold M r are determined by the properties of ?? r and ?? r .

By employing multi-layer perceptrons (neural networks) for ?? r and ?? r , encoder-decoder may provide a smooth manifold.

In this section, we combine the conceptual formulation (1) and the AE guided manifold constraint to derive a equivalent and more practical optimization problem.

First, we redefine a tensor X as an output of generator:

Algorithm 1 Optimization algorithm for tensor reconstruction input:

where l t ??? R r , and H ??? is a pseudo inverse of H. At this moment, X is a function of {l t } T t=1 , however Hankel structure of matrix H can not be always guaranteed under the unconstrained condition of l t .

For guaranteeing the Hankel structure of matrix H, we further transform it as follow:

where we put A r : R D??T ??? R D??T as an operator which auto-encodes each column of a input matrix with (?? r ,?? r ), and [g 1 , g 2 , ..., g T ] as a matrix, which has Hankel structure and is transformed by Hankelization of some input tensor Z ??? R I1??I2??????????I N .

Note that Z is the most compact representation for Hankel matrix [g 1 , g 2 , ..., g T ].

Eq. (4) describes the MMES network shown in Fig. 1 : H,?? r ,?? r and H ??? are respectively corresponding to forward embedding, encoding, decoding, and backward embedding, where encoder and decoder can be defined e.g. by multi-layer perceptrons (i.e., repetition of linear transformation and non-linear activation).

F , where A r is an AE which defines the manifold M r .

In this study, the AE/manifold is learned from an observed tensor Y itself, thus the optimization problem is finally formulated as

where we refer respectively the first and second terms by a reconstruction loss and an auto-encoding loss, and ?? > 0 is a trade-off parameter for balancing both losses.

Optimization problem (5) consists of two terms: a reconstruction loss, and an auto-encoding loss.

Hyperparameter ?? is set to balance both losses.

Basically, ?? should be large because auto-encoding loss should be zero.

However, very large ?? prohibits minimizing the reconstruction loss, and may lead to local optima.

Therefore, we adjust gradually the value of ?? in the optimization process.

Algorithm 1 shows an optimization algorithm for tensor reconstruction and/or enhancement.

For AE learning, we employs a strategy of denoising-auto-encoder (see Appendix in detail).

Adaptation of ?? is just an example, and it can be modified appropriately with data.

Here, the trade-off parameter ?? is adjusted for keeping L rec > L AE , but for no large gap between both losses.

By exploiting the convolutional structure of H and H ??? (see Appendix B.1), the calculation flow of L rec and L AE can be easily implemented by using neural network libraries such as TensorFlow.

We employed Adam (Kingma & Ba, 2014) optimizer for updating (Z, A r ).

Here, we show the selective experimental results to demonstrate the close similarity and some slight differences between DIP and MMES.

First, toy examples with a time-series signal and a gray-scale image were recovered by the proposed method to show its basic behaviors.

Thereafter, we show the main results by comparison with DIP and other selective methods on color-image inpainting, superresolution, and deconvolution tasks.

Optional results of optimization behavior, hyper-parameter sensitivity, and volumetric/3D image completion are shown in Appendix.

In this section, we apply the proposed method into a toy example of signal recovery.

Fig. 3 shows a result of this experiment.

A one-dimensional time-series signal is generated from Lorentz system, and corrupted by additive Gaussian noise, random missing, and three block occlusions.

The corrupted signal was recovered by the subspace modeling (Yokota et al., 2018) , and the proposed manifold modeling in embedded space.

Window size of delay-embedding was ?? = 64, the lowest dimension of auto-encoder was r = 3, and additive noise standard deviation was set to ?? = 0.05.

Manifold modeling catched the structure of Lorentz attractor much better than subspace modeling.

Similar patches are located near each other, and the smooth change of patterns can be observed.

It implies the relationship between non-local similarity based methods (Buades et al., 2005; Dabov et al., 2007; Gu et al., 2014; Zhang et al., 2014a) , and the manifold modeling (i.e., DAE) plays a key role of "patch-grouping" in the proposed method.

The difference from the non-local similarity based approach is that the manifold modeling is "global" rather than "non-local" which finds similar patches of the target patch from its neighborhood area.

In this section, we compare performance of the proposed method with several selected unsupervised image inpainting methods: low-rank tensor completion (HaLRTC) (Liu et al., 2013) , parallel lowrank matrix factorization (TMac) (Xu et al., 2015) , tubal nuclear norm regularization (tSVD) (Zhang et al., 2014b) , Tucker decomposition with rank increment (Tucker inc.) (Yokota et al., 2018) , lowrank and total-variation (LRTV) regularization (Yokota & Hontani, 2017; , smooth PARAFAC tensor completion (SPC) (Yokota et al., 2016) , GSR (Zhang et al., 2014a) , multi-way delay embedding based Tucker modeling (MDT-Tucker) (Yokota et al., 2018) , and DIP (Ulyanov et al., 2018) .

Implementation and detailed hyper-parameter settings are explained in Appendix.

Basically, we carefully tuned the hyper-parameters for all methods to perform the best scores of peak-signal-tonoise ratio (PSNR) and structural similarity (SSIM).

Fig. 5(a) shows the eight test images and averages of PSNR and SSIM for various missing ratio {50%, 70%, 90%, 95%, 99%} and for selective competitive methods.

The proposed method is quite competitive with DIP.

Fig. 6 shows the illustration of results.

The 99% of randomly selected voxels are removed from 3D (256,256,3)-tensors, and the tensors were recovered by various methods.

Basically low-rank priors (HaLRTC, TMac, tSVD, Tucker) could not recover such highly incomplete image.

In piecewise smoothness prior (LRTV), over-smoothed images were reconstructed since the essential image properties could not be captured.

There was a somewhat jump from them by SPC (i.e., smooth prior of basis functions in low-rank tensor decomposition).

MDT-Tucker further improves it by exploiting the shift-invariant multi-linear basis.

GSR nicely recovered the global pattern of images but details were insufficient.

Finally, the reconstructed images by DIP and MMES recovered both global and local patterns of images.

In this section, we compare the proposed method with selected unsupervised image super-resolution methods: Bicubic interpolation, GSR (Zhang et al., 2014a) , ZSSR (Shocher et al., 2018) and DIP (Ulyanov et al., 2018) .

Implementation and detailed hyper-parameter settings are explained in Appendix.

Basically, we carefully tuned the hyper-parameters for all methods to perform the best scores of PSNR and SSIM.

Fig. 5(b) shows values of PSNR and SSIM of the computer simulation results.

We used three (256,256,3) color images, and six (512,512,3) color images.

Super resolution methods scaling up them from four or eight times down-scaled images of them with Lanczos2 kernels.

According to this quantitative evaluation, bicubic interpolation was clearly worse than others.

ZSSR worked well for up-scaling from (128, 128, 3) , however the performances were substantially decreased for upscaling from (64, 64, 3) .

Basically, GSR, DIP, and MMES were very competitive.

In detail, DIP was slightly better than GSR, and the proposed MMES was slightly better than DIP.

More detailed PSNR/SSIM values are given by Table 3 in Appendix.

Fig. 7 shows selected high resolution images reconstructed by four super-resolution methods.

In general, bicubic method reconstructed blurred images and these were visually worse than others.

GSR results had smooth outlines in all images, but these were slightly blurred.

ZSSR was weak for very low-resolution images.

DIP reconstructed visually sharp images but these images had jagged artifacts along the diagonal lines.

The proposed MMES reconstructed sharp and smooth outlines.

In this section, we compare the proposed method with DIP for image deconvolution/deblurring task.

Three (256,256,3) color images are prepared and blurred by using three different Gaussian filters.

For DIP we choose the best early stopping timing from {1000, 2000, ..., 10000} iterations.

For MMES, we employed the fixed AE structure as [32?? 2 , r, 32?? 2 ], and parameters as ?? = 4, r = 16, and ?? = 0.01 for all nine cases.

Fig. 8 shows the reconstructed deblurring images by DIP and MMES.

Tab.

1 shows the PSNR and SSIM values of these results.

We can see that the similarity of the methods qualitatively and quantitatively.

It is well known that there is no mathematical definition of interpretability in machine learning and there is no one unique definition of interpretation.

We understand the interpretability as a degree to which a human can consistently predict the model's results or performance.

The higher the interpretability of a deep learning model, the easier it is for someone to comprehend why certain performance or predictions or expected output can be achieved.

We think that a model is better interpretable than another model if its performance or behaviors are easier for a human to comprehend than performance of the other models.

The manifold learning and associated auto-encoder (AE) can be viewed as the generalized non-linear version of principal component analysis (PCA).

In fact, manifold learning solves the key problem of dimensionality reduction very efficiently.

In other words, manifold learning (modeling) is an approach to non-linear dimensionality reduction.

Manifold modeling for this task are based on the idea that the dimensionality of many data sets is only artificially high.

Although the patches of images (data points) consist of hundreds/thousands pixels, they may be represented as a function of only a few or quite limited number underlying parameters.

That is, the patches are actually samples from a low-dimensional manifold that is embedded in a high-dimensional space.

Manifold learning algorithms attempt to uncover these parameters in order to find a low dimensional representation of the images.

In our MMES approach to solve the problem we applied original embedding via multi-way delay embedding transform (MDT or Hankelization).

Our algorithm is based on the optimization of cost function and it works towards extracting the low-dimensional manifold that is used to describe the high-dimensional data.

The manifold is described mathematically by Eq. (2) and cost function is formulated by Eq. (5).

As mentioned at introduction, Ulyanov et al. (2018) reported an important phenomenon of noise impedance of ConvNet structures.

Here, we provide a prospect for explaining the noise impedance in DIP through the MMES.

Let us consider the sparse-land model, i.e. noise-free images are distributed along low-dimensional manifolds in the high-dimensional Euclidean space and images perturbed by noises thicken the manifolds (make the dimension of the manifolds higher).

Under this model, the distribution of images can be assumed to be higher along the low-dimensional noise-free image manifolds.

When we assume that the image patches are sampled from low-dimensional manifold like sparse-land model, it is difficult to put noisy patches on the low-dimensional manifold.

Let us consider to fit the network for noisy images.

In such case the fastest way for decreasing squared error (loss function) is to learn "similar patches" which often appear in a large set of image-patches.

Note that finding similar image-patches for denoising is well-known problem solved, e.g., by BM3D algorithm, which find similar image patches by template matching.

In contrast, our auto-encoder automatically maps similar-patches into close points on the low-dimensional manifold.

When similar-patches have some noise, the low-dimensional representation tries to keep the common components of similar patches, while reducing the noise components.

This has been proved by Alain & Bengio (2014) so that a (denoising) auto-encoder maps input image patches toward higher density portions in the image space.

In other words, a (denoising) auto-encoder has kind of a force to reconstruct the low-dimensional patch manifold, and this is our rough explanation of noise impedance phenomenon.

Although the proposed MMES and DIP are not completely equivalent, we see many analogies and similarities and we believe that our MMES model and associated learning algorithm give some new insight for DIP.

A beautiful manifold representation of complicated signals in embedded space has been originally discovered in a study of dynamical system analysis (i.e., chaos analysis) for time-series signals (Packard et al., 1980) .

After this, many signal processing and computer vision applications have been studied but most methods have considered only linear approximation because of the difficulty of non-linear modeling (Van Overschee & De Moor, 1991; Szummer & Picard, 1996; Li et al., 1997; Ding et al., 2007; Markovsky, 2008) .

However nowadays, the study of non-linear/manifold modeling has been well progressed with deep learning, and it was successfully applied in this study.

Interestingly, we could apply this non-linear system analysis not only for time-series signals but also natural color images and tensors (this is an extension from delay-embedding to multi-way shiftembedding).

The best of our knowledge, this is the first study to apply Hankelization with AE into general tensor data reconstruction.

MMES is a novel and simple image reconstruction model based on the low-dimensional patchmanifold prior which has many connections to ConvNet.

We believe it helps us to understand how work ConvNet/DIP through MMES, and support to use DIP for various applications like tensor/image reconstruction or enhancement (Gong et al., 2018; Yokota et al., 2019; Van Veen et al., 2018; Gandelsman et al., 2019) .

Finally, we established bridges between quite different research areas such as the dynamical system analysis, the deep learning, and the tensor modeling.

The proposed method is just a prototype and can be further improved by incorporating other methods such as regularizations, multi-scale extensions, and adversarial training.

We can see the anti-diagonal elements of above matrix are equivalent.

Such matrix is called as "Hankel matrix".

For a two-dimensional array

we consider unfold of it and inverse folding by unfold

, and

The point here is that we scan matrix elements column-wise manner.

Hankelization of this twodimensional array (matrix) with ?? = [2, 2] is given by scanning a matrix with local (2,2)-window column-wise manner, and unfold and stack each local patch left-to-right.

Thus, it is given as

We can see that it is not a Hankel matrix.

However, it is a "block Hankel matrix" in perspective of block matrix, a matrix that its elements are also matrices.

We can see the block matrix itself is a Hankel matrix and all elements are Hankel matrices, too.

Thus, Hankel matrix is a special case of block Hankel matrix in case of that all elements are scalar.

In this paper, we say simply "Hankel structure" for block Hankel structure.

Figure 9 shows an illustrative explanation of valid convolution which is decomposed into delayembedding/Hankelization and linear transformation.

1D valid convolution of f with kernel h = [h 1 , h 2 , h 3 ] can be provided by matrix-vector product of the Hankel matrix and h. In similar way, 2D valid convolution can be provided by matrix-vector product of the block Hankel matrix and unfolded kernel.

Multiway-delay embedding transform (MDT) is a multi-way generalization of Hankelization proposed by Yokota et al. (2018) .

In (Yokota et al., 2018) , MDT is defined by using the multi-linear tensor product with multiple duplication matrices and tensor reshaping.

Basically, we use the same operation, but a padding operation is added.

Thus, the multiway-delay embedding used in this study is defined by where

D??T is an unfolding operator which outputs a matrix from an input N -th order tensor, and S n ??? R

is a duplication matrix.

Fig. 10 shows the duplication matrix with ?? .

For example, our Hankelization with reflection padding of f = [f 1 , f 2 , ..., f 7 ] with ?? = 3 is given by Fig. 11 shows an example of our multiway-delay embedding in case of second order tensors.

The overlapped patch grid is constructed by multi-linear tensor product with S n .

Finally, all patches are splitted, lined up, and vectorized.

The Moore-Penrose pseudo inverse of H is given by

where

, and trim ?? = pad ??? ?? is a trimming operator for removing (?? n ???1) elements at start and end of each mode.

Note that H ??? ???H is an identity map, but H ??? H ??? is not, that is kind of a projection.

Delay embedding and its pseudo inverse can be implemented by using convolution with all onehot-tensor windows of size (?? 1 , ?? 2 , ..., ?? N ).

The one-hot-tensor windows can be given by folding a D-dimensional identity matrix I D ??? R D??D into I D ??? R ??1???????????? N ??D .

Fig. 12 shows a calculation flow of multi-way delay embedding using convolution in a case of N = 2.

Multi-linear tensor product is replaced with convolution with one-hot-tensor windows.

Pseudo inverse of the convolution with padding is given by its adjoint operation, which is called as the "transposed convolution" in some neural network library, with trimming and simple scaling with

Figure 10: Duplication matrix.

In case that we have I columns, it consists of (I ??? ?? + 1) identity matrices of size (??, ?? ).

In this section, we discuss how to design the neural network architecture of auto-encoder for restricting the manifold M r .

The simplest way is controlling the value of r, and it directly restricts the dimensionality of latent space.

There are many other possibilities: Tikhonov regularization (Goodfellow et al., 2016) , drop-out (Gal & Ghahramani, 2016) , denoising auto-encoder (Vincent et al., 2008 ), variational auto-encoder (Diederik P Kingma, 2014 ), adversarial auto-encoder (Makhzani et al., 2015 ), alpha-GAN (Rosca et al., 2017 , and so on.

All methods have some perspective and promise, however the cost is not low.

In this study, we select an attractive and fundamental one: "denoising auto-encoder"(DAE) (Vincent et al., 2008) .

The DAE is attractive because it has a strong relationship with Tikhonov regularization (Bishop, 1995) , and decreases the entropy of data (Sonoda & Murata, 2017) .

Furthermore, learning with noise is also employed in the deep image prior.

Finally, we designed an auto-encoder with controlling the dimension r and the standard deviation ?? of additive zero-mean Gaussian noise.

In case of multi-channel or color image recovery case, we use a special setting of generator network because spacial pattern of individual channels are similar and the patch-manifold can be shared.

Fig. 14 shows an illustration of the auto-encoder shared version of MMES in a case of color image recovery.

In this case, we put three channels of input and each channel input is embedded, independently.

Then, three block Hankel matrices are concatenated, and auto-encoded simultaneously.

Inverted three images are stacked as a color-image (third-order tensor), and finally color-transformed.

The last color-transform can be implemented by convolution layer with kernel size (1,1), and it is also optimized as parameters.

It should be noted that the input three channels are not necessary to correspond to RGB, but it would be optimized as some compact color-representation.

Here, we explain detailed experimental settings in Section 4.2.

In this section, we compared performance of the proposed method with several selected unsupervised image inpainting methods: low-rank tensor completion (HaLRTC) (Liu et al., 2013) , parallel low-rank matrix factorization (TMac) (Xu et al., 2015) , tubal nuclear norm regularization (tSVD) (Zhang et al., 2014b) , Tucker decomposition with rank increment (Tucker inc.) (Yokota et al., 2018) , low-rank and total-variation (LRTV) regularization 2 (Yokota & Hontani, 2017; , smooth PARAFAC tensor completion (SPC) 3 (Yokota et al., 2016) , GSR 4 (Zhang et al., 2014a) , multi-way (Ulyanov et al., 2018) .

For this experiments, hyper-parameters of all methods were tuned manually to perform the best peaksignal-to-noise ratio (PSNR) and for structural similarity (SSIM), although it would not be perfect.

For DIP, we did not try the all network structures with various kernel sizes, filter sizes, and depth.

We just employed "default architecture", which the details are available in supplemental material 7 of (Ulyanov et al., 2018) , and employed the best results at the appropriate intermediate iterations in optimizations based on the value of PSNR.

For the proposed MMES method, we adaptively selected the patch-size ?? , and dimension r. Table 2 shows parameter settings of ?? = [??, ?? ] and r for MMES.

Noise level of denoising auto-encoder was set as ?? = 0.05 for all images.

For auto-encoder, same architecture shown in Fig. 13 was employed.

Initial learning rate of Adam optimizer was 0.01 and we decayed the learning rate with 0.98 every 100 iterations.

The optimization was stopped after 20,000 iterations for each image.

Here, we explain detailed experimental settings in Section 4.3.

In this section, we compare performance of the proposed method with several selected unsupervised image super-resolution methods: bicubic interpolation, GSR 8 (Zhang et al., 2014a) , ZSSR 9 and DIP (Ulyanov et al., 2018) .

In this experiments, DIP was conducted with the best number of iterations from {1000, 2000, 3000, ..., 9000}. For four times (x4) up-scaling in MMES, we set ?? = 6, r = 32, and ?? = 0.1.

For eight times (x8) up-scaling in MMES, we set ?? = 6, r = 16, and ?? = 0.1.

For all images in MMES, the architecture of auto-encoder consists of three hidden layers with sizes of [8?? 2 , r, 8?? 2 ].

We assumed the same Lanczos2 kernel for down-sampling system for all super-resolution methods.

Tab.

3 shows values of PSNR and SSIM of the results.

We used three (256,256,3) color images, and six (512,512,3) color images.

Super resolution methods scaling up them from four or eight times down-scaled images of them.

According to this quantitative evaluation, bicubic interpolation was clearly worse than others.

ZSSR was good for (128,128,3) color images, however the performance were substantially decreased for (64,64,3) color image.

Basically, GSR, DIP, and MMES were very competitive.

In detail, DIP was slightly better than GSR, and the proposed MMES was slightly better than DIP.

For this experiment, we recovered 50% missing gray-scale image of 'Lena'.

We stopped the optimization algorithm after 20,000 iterations.

Learning rate was set as 0.01, and we decayed the learning rate with 0.98 every 100 iterations.

?? was adapted by Algorithm 1 every 10 iterations.

Fig. 15 shows optimization behaviors of reconstructed image, reconstruction loss L rec , auto-encoding loss L DAE , and trade-off coefficient ??.

By using trade-off adjustment, the reconstruction loss and the auto-encoding loss were intersected around 1,500 iterations, and both losses were jointly decreased after the intersection point.

We evaluate the sensitivity of MMES with three hyper-parameters: r, ??, and ?? .

First, we fixed the patch-size as (8, 8) , and dimension r and noise standard deviation ?? were varied.

the reconstruction results of a 99% missing image of 'Lena' by the proposed method with different settings of (r, ??).

The proposed method with very low dimension (r = 1) provided blurred results, and the proposed method with very high dimension (r = 64) provided results which have many peaks.

Furthermore, some appropriate noise level (?? = 0.05) provides sharp and clean results.

For reference, Fig. 16 shows the difference of DIP optimized with and without noise.

From both results, the effects of learning with noise can be confirmed.

Next, we fixed the noise level as ?? = 0.05, and the patch-size were varied with some values of r. Fig. 18 shows the results with various patch-size settings for recovering a 99% missing image.

The patch sizes ?? of (8,8) or (10,10) were appropriate for this case.

Patch size is very important because it depends on the variety of patch patterns.

If patch size is too large, then patch variations might expand and the structure of patch-manifold is complicated.

By contrast, if patch size is too small, then the information obtained from the embedded matrix H is limited and the reconstruction becomes difficult in highly missing cases.

The same problem might be occurred in all patch-based image reconstruction methods (Buades et al., 2005; Dabov et al., 2007; Gu et al., 2014; Zhang et al., 2014a) .

However, good patch sizes would be different for different images and types/levels of corruption, and the estimation of good patch size is an open problem.

Multi-scale approach (Yair & Michaeli, 2018) may reduce a part of this issue but the patch-size is still fixed or tuned as a hyper-parameter.

In this section, we show the results of MR-image/3D-tensor completion problem.

The size of MR image is (109, 91, 91) .

We randomly remove 50%, 70%, and 90% voxels of the original MR-image and recover the missing MR-images by the proposed method and DIP.

For DIP, we implemented the 3D version of default architecture in TensorFlow, but the number of filters of shallow layers were slightly reduced because of the GPU memory constraint.

For the proposed method, 3D patch-size was set as ?? = [4, 4, 4] , the lowest dimension was r = 6, and noise level was ?? = 0.05.

Same architecture shown in Fig. 13 was employed.

Fig. 19 shows reconstruction behavior of PSNR with final value of PSNR/SSIM in this experiment.

From the values of PSNR and SSIM, the proposed MMES outperformed DIP in low-rate missing cases, and it is quite competitive in highly missing cases.

The some degradation of DIP might be occurred by the insufficiency of filter sizes since much more filter sizes would be required for 3D ConvNet than 2D ConvNet.

Moreover, computational times required for our MMES were significantly shorter than that of DIP in this tensor completion problem. (4,4), r=4 (6,6), r=8 (8,8), r=16 (10,10), r=32 (12,12), r=48 (16,16)

@highlight

We propose a new auto-encoder incorporated with multiway delay-embedding transform toward interpreting deep image prior.