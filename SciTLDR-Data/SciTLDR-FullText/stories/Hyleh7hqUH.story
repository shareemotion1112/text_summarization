The use of deep learning models as priors for compressive sensing tasks presents new potential for inexpensive seismic data acquisition.

An appropriately designed Wasserstein generative adversarial network is designed based on a generative adversarial network architecture trained on several historical surveys, capable of learning the statistical properties of the seismic wavelets.

The usage of validating and performance testing of compressive sensing are three steps.

First, the existence of a sparse representation with different compression rates for seismic surveys is studied.

Then, non-uniform samplings are studied, using the proposed methodology.

Finally, recommendations for non-uniform seismic survey grid, based on the evaluation of reconstructed seismic images and metrics, is proposed.

The primary goal of the proposed deep learning model is to provide the foundations of an optimal design for seismic acquisition, with less loss in imaging quality.

Along these lines, a compressive sensing design of a non-uniform grid over an asset in Gulf of Mexico, versus a traditional seismic survey grid which collects data uniformly at every few feet, is suggested, leveraging the proposed method.

Conventional computational recovery is suffered from undesired artifacts such as over-smoothing, image size limitations and high computational cost.

The use of deep generative network (GAN) models offers a very promising alternative approach for inexpensive seismic data acquisition, which improved quality and revealing finer details when compared to conventional approaches or pixel-wise deep learning models.

As one of the pioneers to apply a pixel inpainting GAN on large, real seismic compressed image recovery, we contributes the following points: 1) Introduction of a GAN based inpainting model for compressed image recovery, under uniform or nonuniform sampling, capable to recover the heavily sampled data efficiently and reliably.

2) Superior model for compressive sensing on uniform sampling, that performs better than the originial network and the state-of-the-art interpolation method for uniform sampling.

3) Introduction of an effective, non-uniform, sampling survey recommendation, leveraging the GIN uniform sampling reconstructions and a hierarchical selection scheme.

Compressed image recovery can be stated as a missing pixel inpainting problem: given an incomplete image, filling the missing trace values.

Using historical images of the uncompressed dataset, we train a data-driven deep learning model, utilizing the raw image as ground truth and the binary mask to indicate the locations of the missing pixels.

Once trained, one can test the model's performance on any incomplete image from a different dataset.

We used Compression Rate (CR) to define the proportion between uncompressed data size and compressed data size.

The main challenges in using an inpainting model to solve the seismic image sampling problems are: 1) seismic images have significantly different statistical characteristics, such as texture-based patterns and wide range of frequencies, compared to natural images; 2) the largest number of unknown pixels is only ¼ of the full image in the original network, whereas in our task covers at least ½ of the image (i.e.

CR=2); 3) the known regions in the compressed image are sparsely distributed, contrary to the compact ones in the general inpainting problems.

To address these problems, we will modify the original network and employ it on different experiments.

The Generative Inpainting Network with Contextual Attention (GIN) [1] is a feed-forward GAN, combining and outperforming several state-of-the-art approaches including context encoders [2] , dilated convolutions of inpainting Error!

Reference source not found., Wasserstein GAN [2] and its improvement WGAN-GP [3] .

The architecture of GIN is composed by a coarse network recovering the general features for the missing traces, and a refinement network which further reconstructs the detail.

Especially, the Contextual Attention in the latter, does not only learn from the known pixels surrounding the masked image area, but it also looks for useful patches from other known image locations.

In our Generative Inpainting Network for Compressive Sensing (named as GIN-CS), we replace the single bounding boxes, used in the incomplete image generation, by predefined binary masks.

On the one hand, a binary mask could be regarded as the combination of non-adjacent bounding boxes, so that the multiple edges lose the continuity of original spatial relations.

On the other hand, the maximum width of connected missing traces is generally small, so the edge effect would be ignored over the global image size.

We used a small portion of an internal offshore dataset to train the network, where 5000 of the processed offshore seismic images are cropped into 256×256 and mix the in-line and cross-line cases.

There are two ways to arrange the training masks: random single bounding boxes, as in the original GIN or predefined binary sampling masks, as in our modified GIN-CS.

For testing, both methods use binary masks.

The testing seismic dataset was collected from the Gulf of Mexico (GoM), on the courtesy of TGS.

By comparing the performance of our modified GIN-CS with the original GIN and the conventional biharmonic method [4] in the same CR, our model demonstrates the overall superior performance in terms of Mea Square Error (MSE) and Structural Similarity (SSIM) index [5] in all CR cases (Table 1) .

Also, the GIN related methods are much faster than the traditional method and hardly influenced by the value of CR.

(1)

Focusing on one trace from the testing image, our model's prediction aligns better with the ground truth relatively to the GIN (Figure 1.1) .

Although for CR=8,16, our method does not get better performance in terms of PSNR (Table 1.

3), it still generates closer-to-real seismic images without adding additional artificial noise(GIN) or creating blurry fillings (biharmonic), as seen in Figure 1.

In order to construct a non-uniform optimal sampling survey set-up, we propose a sampling recommendation approach that leverages the fast implementation of image reconstruction with GIN.

This is an efficient non-uniform sampling recommendation method based on hierarchical uniform sampling, which requires only a small number of sampling test cases.

Noted that our recommended sampling method does not consider the connected sampling crossing section width, and the effectiveness highly relies on the performance of GIN.

1) Mask Generation.

For a given uncompressed seismic image, we designed a set of binary masks that complementary to the whole masks with equal bin width b∈{1,2,4,8} of each groups 2) Difference map generation.

The image is tested with all the designed compression cases.

Then, we create the corresponding error matrix by calculating the pixel-wise square error of the reconstructions compared with ground truth.

Then, summing them up to form a complete image difference map and calculate its trace-wise mean vector.

We further split the vector into individual difference values for each trace .

Smaller value indicates better reconstruction at trace .

3) Initial candidate traces generation.

In order to compare for each trace without breaking the unknown connected traces, we introduce the observed interval ∈ {2,4,8} to distinguish from bin width .

The step mean difference is then defined by simply replacing the actual difference value as its mean value over every interval

For an observed interval , the initial recommend candidate traces ′ are the first trace indexes that reach the smallest step mean difference, when = , i.e.

All candidate traces over the interval form ( ) = { ′ ( ) } as a subset of all traces.

One trace might be a candidate for selection from several intervals, or might never become a candidate for selection.

4) Top-to-bottom hierarchical sorting.

To avoid repetitive trace selection, a two-order sorting on all the candidate traces is implemented with the descending order of interval followed by ascending order of step mean.

The traces in the higher orders are selected firstly and all the adjacent traces within all the intervals has been removed corresponding until the total missing trace reaching the limitation of CR or empty traces remains.

The elements in the final set of are the prospective missing traces, which are easy to be recovered by GIN.

We have compared the reconstruction performance result of our recommended sampling survey with an average of 100 random samplings, and report the improvement in Table 2 .

Moreover, we compared the uniform sampling and sampling recommendation separately on the same GoM dataset as the 2D depth view in Figure 1Error !

Reference source not found. shown.

The results of cross-line and in-line is combined by taking the average of the overlapped regions, in order to mimic the actual sampling in both dimensions at the same time.

All the cross-line and in-line sampled images are stacked together to form a 3D reconstruction of the whole block, showing its 2D depth view in the figure.

The recommend sampling points are densely distributed in regions with lithologic features and sparsely distributed in channelized regions.

This successfully captures the heterogenetic of the seismic image.

(2) (1) We designed and implemented a modification of the GIN model, the GIN-CS, and successfully tested its performance on uniform samplings with compression rates ×2, ×4, ×8, ×16.

GIN-CS demonstrates superior reconstruction performance relatively to both the original GIN and the conventional biharmonic method.

More precisely, we show that seismic imaging can be successfully recovered by filling the missing traces, revealing finer details, even in high compression rate cases.

In addition, the proposed method runs approximately 300 times faster than the conventional method.

Finally, a strategy for constructing a recommendation of non-uniform survey is proposed for a field dataset from Gulf of Mexico, based on our results from a combination of limited amount of uniform sampling experiments.

@highlight

Improved a GAN based pixel inpainting network for compressed seismic image recovery andproposed a non-uniform sampling survey recommendatio, which can be easily applied to medical and other domains for compressive sensing technique.