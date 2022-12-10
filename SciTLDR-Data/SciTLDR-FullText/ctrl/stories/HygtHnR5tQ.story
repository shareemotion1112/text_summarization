We propose a framework for extreme learned image compression based on Generative Adversarial Networks (GANs), obtaining visually pleasing images at significantly lower bitrates than previous methods.

This is made possible through our GAN formulation of learned compression combined with a generator/decoder which operates on the full-resolution image and is trained in combination with a multi-scale discriminator.

Additionally, if a semantic label map of the original image is available, our method can fully synthesize unimportant regions in the decoded image such as streets and trees from the label map, therefore only requiring the storage of the preserved region and the semantic label map.

A user study confirms that for low bitrates, our approach is preferred to state-of-the-art methods, even when they use more than double the bits.

Image compression systems based on deep neural networks (DNNs), or deep compression systems for short, have become an active area of research recently.

These systems (e.g. BID6 BID33 ) are often competitive with modern engineered codecs such as WebP (WebP), JPEG2000 BID37 ) and even BPG (Bellard) (the state-of-the-art engineered codec).

Besides achieving competitive compression rates on natural images, they can be easily adapted to specific target domains such as stereo or medical images, and promise efficient processing and indexing directly from compressed representations BID41 .

However, deep compression systems are typically optimized for traditional distortion metrics such as peak signal-to-noise ratio (PSNR) or multi-scale structural similarity (MS-SSIM) BID44 .

For very low bitrates (below 0.1 bits per pixel (bpp)), where preserving the full image content becomes impossible, these distortion metrics lose significance as they favor pixel-wise preservation of local (high-entropy) structure over preserving texture and global structure.

To further advance deep image compression it is therefore of great importance to develop new training objectives beyond PSNR and MS-SSIM.

A promising candidate towards this goal are adversarial losses BID13 which were shown recently to capture global semantic information and local texture, yielding powerful generators that produce visually appealing high-resolution images from semantic label maps BID43 .In this paper, we propose and study a generative adversarial network (GAN)-based framework for extreme image compression, targeting bitrates below 0.1 bpp.

We rely on a principled GAN formulation for deep image compression that allows for different degrees of content generation.

In contrast to prior works on deep image compression which applied adversarial losses to image patches for artifact suppression BID33 BID12 , generation of texture details BID25 , or representation learning for thumbnail images BID35 , our generator/decoder operates on the full-resolution image and is trained with a multi-scale discriminator BID43 .We consider two modes of operation (corresponding to unconditional and conditional GANs BID13 BID31 ), namely• generative compression (GC), preserving the overall image content while generating structure of different scales such as leaves of trees or windows in the facade of buildings, and • selective generative compression (SC), completely generating parts of the image from a semantic label map while preserving user-defined regions with a high degree of detail.

We emphasize that GC does not require semantic label maps (neither for training, nor for deployment).

A typical use case for GC are bandwidth constrained scenarios, where one wants to preserve the full image as well as possible, while falling back to synthesized content instead of blocky/blurry blobs for regions for which not sufficient bits are available to store the original pixels.

SC could be applied in a video call scenario where one wants to fully preserve people in the video stream, but a visually pleasing synthesized background serves the purpose as well as the true background.

In the GC operation mode the image is transformed into a bitstream and encoded using arithmetic coding.

SC requires a semantic/instance label map of the original image which can be obtained using off-the-shelf semantic/instance segmentation networks, e.g., PSPNet and Mask R-CNN BID18 , and which is stored as a vector graphic.

This amounts to a small, image dimension-independent overhead in terms of coding cost.

On the other hand, the size of the compressed image is reduced proportionally to the area which is generated from the semantic label map, typically leading to a significant overall reduction in storage cost.

For GC, a comprehensive user study shows that our compression system yields visually considerably more appealing results than BPG (Bellard) (the current state-of-the-art engineered compression algorithm) and the recently proposed autoencoder-based deep compression (AEDC) system .

In particular, our GC models trained for compression of general natural images are preferred to BPG when BPG uses up to 95% and 124% more bits than those produced by our models on the Kodak (Kodak) and RAISE1K BID11 data set, respectively.

When constraining the target domain to the street scene images of the Cityscapes data set BID9 , the reconstructions of our GC models are preferred to BPG even when the latter uses up to 181% more bits.

To the best of our knowledge, these are the first results showing that a deep compression method outperforms BPG on the Kodak data set in a user study-and by large margins.

In the SC operation mode, our system seamlessly combines preserved image content with synthesized content, even for regions that cross multiple object boundaries, while faithfully preserving the image semantics.

By partially generating image content we achieve bitrate reductions of over 50% without notably degrading image quality.

Deep image compression has recently emerged as an active area of research.

The most popular DNN architectures for this task are to date auto-encoders BID6 BID2 BID41 and recurrent neural networks (RNNs) BID39 .

These DNNs transform the input image into a bit-stream, which is in turn losslessly compressed using entropy coding methods such as Huffman coding or arithmetic coding.

To reduce coding rates, many deep compression systems rely on context models to capture the distribution of the bit stream BID6 BID40 BID33 .

Common loss functions to measure the distortion between the original and decompressed images are the mean-squared error (MSE) BID6 BID2 BID41 , or perceptual metrics such as MS-SSIM BID40 BID33 .

Some authors rely on advanced techniques including multiscale decompositions BID33 , progressive encoding/decoding strategies BID39 , and generalized divisive normalization (GDN) layers BID6 a) .Generative adversarial networks (GANs) BID13 have emerged as a popular technique for learning generative models for intractable distributions in an unsupervised manner.

Despite stability issues BID34 BID3 BID28 , they were shown to be capable of generating more realistic and sharper images than prior approaches and to scale to resolutions of 1024 × 1024px BID22 for some datasets.

Another direction that has shown great progress are conditional GANs BID13 BID31 , obtaining impressive results for image-to-image translation BID43 BID27 on various datasets (e.g. maps to satellite images), reaching resolutions as high as 1024 × 2048px BID43 .Arguably the most closely related work to ours is BID33 , which uses an adversarial loss term to train a deep compression system.

However, this loss term is applied to small image patches and its purpose is to suppress artifacts rather than to generate image content.

Furthermore, it uses a non-standard GAN formulation that does not (to the best of our knowledge) have an interpretation in terms of divergences between probability distributions, as in BID13 BID32 .

We refer to Sec. 6.1 and Appendix A for a more detailed discussion.

BID35 use a GAN framework to learn a generative model over thumbnail images, which is then used as a decoder for thumbnail image compression.

Other works use adversarial training for compression artifact removal (for engineered codecs) BID12 and single image super-resolution BID25 .

Finally, related to our SC mode, spatially allocating bitrate based on saliency of image content has a long history in the context of engineered compression algorithms, see, e.g.,, BID36 BID15 BID16 .

Generative Adversarial Networks: Given a data set X , Generative Adversarial Networks (GANs) can learn to approximate its (unknown) distribution p x through a generator G(z) that tries to map samples z from a fixed prior distribution p z to the distribution p x .

The generator G is trained in parallel with a discriminator D by searching (using stochastic gradient descent (SGD)) for a saddle point of a mini-max objective DISPLAYFORM0 where G and D are DNNs and f and g are scalar functions.

The original paper BID13 uses the "Vanilla GAN" objective with f (y) = log(y) and g(y) = log(1 − y).

This corresponds to G minimizing the Jensen-Shannon (JS) Divergence between the (empirical) distribution of x and G(z).

The JS Divergence is a member of a more generic family of f -divergences, and BID32 show that for suitable choices of f and g, all such divergences can be minimized with (1).

In particular, if one uses f (y) = (y − 1) 2 and g(y) = y 2 , one obtains the LeastSquares GAN BID28 (which corresponds to the Pearson χ 2 divergence), which we adopt in this paper.

We refer to the divergence minimized over G as DISPLAYFORM1 Conditional Generative Adversarial Networks: For conditional GANs (cGANs) BID13 BID31 , each data point x is associated with additional information s, where (x, s) have an unknown joint distribution p x,s .

We now assume that s is given and that we want to use the GAN to model the conditional distribution p x|s .

In this case, both the generator G(z, s) and discriminator D(z, s) have access to the side information s, leading to the divergence DISPLAYFORM2 Deep Image Compression: To compress an image x ∈ X , we follow the formulation of BID2 where one learns an encoder E, a decoder G, and a finite quantizer q.

The encoder E maps the image to a latent feature map w, whose values are then quantized to L levels {c 1 , . . .

, c L } ⊂ R to obtain a representationŵ = q(E(x)) that can be encoded to a bitstream.

The decoder then tries to recover the image by forming a reconstructionx = G(ŵ).

To be able to backpropagate through the non-differentiable q, one can use a differentiable relaxation of q, as in .The average number of bits needed to encodeŵ is measured by the entropy H(ŵ), which can be modeled with a prior BID2 or a conditional probability model .

The trade-off between reconstruction quality and bitrate to be optimized is then DISPLAYFORM3 where d is a loss that measures how perceptually similarx is to x. Given a differentiable estimator of the entropy H(ŵ), the weight β controls the bitrate of the model (large β pushes the bitrate down).

However, since the number of dimensions dim(ŵ) and the number of levels L are finite, the entropy is bounded by (see, e.g., BID10 ) DISPLAYFORM4 It is therefore also valid to set β = 0 and control the maximum bitrate through the bound (5) (i.e., adjusting L and/or dim(ŵ) through the architecture of E).

While potentially leading to suboptimal bitrates, this avoids to model the entropy explicitly as a loss term.

The proposed GAN framework for extreme image compression can be viewed as a combination of (conditional) GANs and learned compression.

With an encoder E and quantizer q, we encode the image x to a compressed representationŵ = q(E(x)).

This representation is optionally concatenated with noise v drawn from a fixed prior p v , to form the latent vector z. The decoder/generator G then tries to generate an imagê x = G(z) that is consistent with the image distribution p x while also recovering the specific encoded image x to a certain degree (see inset Fig.) .

Using z = [ŵ, v] , this can be expressed by our saddle-point objective for (unconditional) generative compression, DISPLAYFORM0 where λ > 0 balances the distortion term against the GAN loss and entropy terms.

Using this formulation, we need to encode a real image,ŵ = E(x), to be able to sample from pŵ.

However, this is not a limitation as our goal is to compress real images and not to generate completely new ones.

Since the last two terms of (6) do not depend on the discriminator D, they do not affect its optimization directly.

This means that the discriminator still computes the same f divergence L GAN as in (2), so we can write (6) as DISPLAYFORM1 We note that equation (6) has completely different dynamics than a normal GAN, because the latent space z containsŵ, which stores information about a real image x. A crucial ingredient is the bitrate limitation on H(ŵ).

If we allowŵ to contain arbitrarily many bits by setting β = 0 and letting L and dim(ŵ) be large enough, E and G could learn to near-losslessly recover x from G(z) = G(q(E(x))), such that the distortion term would vanish.

In this case, the divergence between p x and p G(z) would also vanish and the GAN loss would have no effect.

By constraining the entropy ofŵ, E and G will never be able to make d fully vanish.

In this case, E, G need to balance the GAN objective L GAN and the distortion term λE[d(x, G(z))], which leads to G(z) on one hand looking "realistic", and on the other hand preserving the original image.

For example, if there is a tree for which E cannot afford to store the exact texture (and make d small) G can synthesize it to satisfy L GAN , instead of showing a blurry green blob.

In the extreme case where the bitrate becomes zero (i.e., H(ŵ) → 0, e.g., by setting β = ∞ or dim(ŵ) = 0),ŵ becomes deterministic.

In this setting, z is random and independent of x (through the v component) and the objective reduces to a standard GAN plus the distortion term, which acts as a regularizer.

We refer to the setting in (6) as generative compression (GC), where E, G balance reconstruction and generation automatically over the image.

As for the conditional GANs described in Sec. 3, we can easily extend GC to a conditional case.

Here, we also consider this setting, where the additional information s for an image x is a semantic label map of the scene, but with a twist: Instead of feeding the semantics to E, G and D, we only give them to the discriminator D during training.

This means that no semantics are needed to encode or decode images with the trained models (since E, G do not depend on s).

We refer to this setting as GC (D + ).

For GC and its conditional variant described in the previous section, E, G automatically navigate the trade-off between generation and preservation over the entire image, without any guidance.

Here, we consider a different setting, where we guide the network in terms of which regions should be preserved and which regions should be synthesized.

We refer to this setting as selective generative compression (SC) (an overview of the network structure is given in Fig. 8 in Appendix C).For simplicity, we consider a binary setting, where we construct a single-channel binary heatmap m of the same spatial dimensions asŵ.

Regions of zeros correspond to regions that should be fully synthesized, whereas regions of ones correspond to regions that should be preserved.

However, since our task is compression, we constrain the fully synthesized regions to have the same semantics s as the original image x.

We assume the semantics s are separately stored, and thus feed them through a feature extractor F before feeding them to the generator G. To guide the network with the semantics, we mask the (pixel-wise) distortion d, such that it is only computed over the region to be preserved.

Additionally, we zero out the compressed representationŵ in the regions that should be synthesized.

Provided that the heatmap m is also stored, we then only encode the entries ofŵ corresponding to the preserved regions, greatly reducing the bitrate needed to store it.

At bitrates whereŵ is normally much larger than the storage cost for s and m (about 2kB per image when encoded as a vector graphic), this approach can result in large bitrate savings.

We consider two different training modes: Random instance (RI) which randomly selects 25% of the instances in the semantic label map and preserves these, and random box (RB) which picks an image location uniformly at random and preserves a box of random dimensions.

While the RI mode is appropriate for most use cases, the RB can create more challenging situations for the generator as it needs to integrate the preserved box seamlessly into the generated content.

The architecture for our encoder E and generator G is based on the global generator network proposed in BID43 , which in turn is based on the architecture of BID21 .

We present details in Appendix C.For the entropy term βH(ŵ), we adopt the simplified approach described in Sec. 3, where we set β = 0, use L = 5 centers C = {−2, 1, 0, 1, 2}, and control the bitrate through the upper bound DISPLAYFORM0 For example, for GC, with C = 2 channels, we obtain 0.0181bpp.

2 We note that this is an upper bound; the actual entropy of H(ŵ) is generally smaller, since the learned distribution will neither be uniform nor i.i.d, which would be required for the bound to hold with equality.

When encoding the channels ofŵ to a bit-stream, we use an arithmetic encoder where frequencies are stored for each channel separately and then encode them in a static (non-adaptive) 1 If we assume s is an unknown function of x, another view is that we feed additional features (s) to D. Ours, 0.035bpp, 21.8dB BPG, 0.039bpp, 26.0dB MSE bl.

, 0.035bpp, 24.0dB DISPLAYFORM1 Figure 2: Visual example of images produced by our GC network with C = 4 along with the corresponding results for BPG, and a baseline model with the same architecture (C = 4) but trained for MSE only (MSE bl.), on Cityscapes.

The reconstruction of our GC network is sharper and has more realistic texture than those of BPG and the MSE baseline, even though the latter two have higher PSNR (indicated in dB for each image) than our GC network.

In particular, the MSE baseline produces blurry reconstructions even though it was trained on the Cityscapes data set, demonstrating that domain-specific training alone is not enough to obtain sharp reconstructions at low bitrates.manner, similar to BID2 .

In our experiments, this leads to 8.8% smaller bitrates compared to the upper bound.

By using a context model and adaptive arithmetic encoding, we could reduce the bitrate further, either in a post processing step (as in BID33 BID6 ), or jointly during training (as in )-which led to ≈ 10% savings in these prior works.

For the distortion term we adopt d(x,x) = MSE(x,x) with coefficient λ = 10.

Furthermore, we adopt the feature matching and VGG perceptual losses, L FM and L VGG , as proposed in BID43 with the same weights, which improved the quality for images synthesized from semantic label maps.

These losses can be viewed as a part of d(x,x).

However, we do not mask them in SC, since they also help to stabilize the GAN in this operation mode (as in BID43 ).

We refer to Appendix D for training details.

Data sets: We train GC models (without semantic label maps) for compression of diverse natural images using 188k images from the Open Images data set BID24 and evaluate them on the widely used Kodak image compression data set (Kodak) as well as 20 randomly selected images from the RAISE1K data set BID11 .

To investigate the benefits of having a somewhat constrained application domain and semantic information at training time, we also train GC models with semantic label maps on the Cityscapes data set BID9 , using 20 randomly selected images from the validation set for evaluation.

To evaluate the proposed SC method (which requires semantic label maps for training and deployment) we again rely on the Cityscapes data set.

Cityscapes was previously used to generate images form semantic label maps using GANs .Baselines: We compare our method to the HEVC-based image compression algorithm BPG (Bellard) (in the 4:2:2 chroma format) and to the AEDC network from .

BPG is the current state-of-the-art engineered image compression codec and outperforms other recent codecs such as JPEG2000 and WebP on different data sets in terms of PSNR (see, e.g. ).

We train the AEDC network (with bottleneck depth C = 4) on Cityscapes exactly following the procedure in ) except that we use early stopping to prevent overfitting (note that Cityscapes is much smaller than the ImageNet dataset used in ).

The so-obtained model has a bitrate of 0.07 bpp and gets a slightly better MS-SSIM than BPG at the same bpp on the validation set.

To investigate the effect of the GAN term in our total loss, we train a baseline model with an MSE loss only (with the same architecture as GC and the same training parameters, see Sec. D in the Appendix), referred to as "MSE baseline".Ours 0.0341bpp BPG 0.102bpp Ours 0.0339bpp BPG 0.0382bpp Figure 3 : Visual example of images from RAISE1k produced by our GC network with C = 4 along with the corresponding results for BPG.User study: In the extreme compression regime realized by our GC models, where texture and sometimes even more abstract image content is synthesized, common reconstruction quality measures such as PSNR and MS-SSIM arguably lose significance as they penalize changes in local structure rather than assessing preservation of the global image content (this also becomes apparent by comparing reconstructions produced by our GC model with those obtained by the MSE baseline and BPG, see Fig. 2 ).

Indeed, measuring PSNR between synthesized and real texture patches essentially quantifies the variance of the texture rather than the visual quality of the synthesized texture.

To quantitatively evaluate the perceptual quality of our GC models in comparison with BPG and AEDC (for Cityscapes) we therefore conduct a user study using Amazon Mechanical Turk (AMT).

3 We consider two GC models with C = 4, 8 trained on Open Images, three GC (D + ) models with C = 2, 4, 8 trained on Cityscapes, and BPG at rates ranging from 0.045 to 0.12 bpp.

Questionnaires are composed by combining the reconstructions produced by the selected GC model for all testing images with the corresponding reconstruction produced by the competing baseline model side-byside (presenting the reconstructions in random order).

The original image is shown along with the reconstructions, and the pairwise comparisons are interleaved with 3 probing comparisons of an additional uncompressed image from the respective testing set with an obviously JPEG-compressed version of that image.

20 randomly selected unique users are asked to indicate their preference for each pair of reconstructions in the questionnaire, resulting in a total of 480 ratings per pairing of methods for Kodak, and 400 ratings for RAISE1K and Cityscapes.

For each pairing of methods, we report the mean preference score as well as the standard error (SE) of the per-user mean preference percentages.

Only users correctly identifying the original image in all probing comparisons are taken into account for the mean preference percentage computation.

To facilitate comparisons for future works, we will release all images used in the user studies.

Semantic quality of SC models: The issues with PSNR and MS-SSIM for evaluating the quality of generated content described in the previous paragraph become even more severe for SC models as a large fraction of the image content is generated from a semantic label map.

Following image translation works BID43 , we therefore measure the capacity of our SC models to preserve the image semantics in the synthesized regions and plausibly blend them with the preserved regions-the objective SC models are actually trained for.

Specifically, we use PSPNet BID46 and compute the mean intersection-over-union (IoU) between the label map obtained for the decompressed validation images and the ground truth label map.

For reference we also report this metric for baselines that do not use semantic label maps for training and/or deployment.

6.1 GENERATIVE COMPRESSION Fig. 4 shows the mean preference percentage obtained by our GC models compared to BPG at different rates, on the Kodak and the RAISE1K data set.

In addition, we report the mean preference percentage for GC models compared to BPG and AEDC on Cityscapes.

Example validation images for side-by-side comparison of our method with BPG for images from the Kodak, RAISE1K, and Cityscapes data set can be found in Figs. 1, 3, and 2 , respectively.

Furthermore, we perform extensive visual comparisons of all our methods and the baselines, presented in Appendix F. Figure 4 : User study results evaluating our GC models on Kodak, RAISE1K (top) and Cityscapes (bottom).

For Kodak and RAISE1K, we use GC models trained on Open Images, without any semantic label maps.

For Cityscapes, we used GC (D + ), using semantic label maps only for D and only during training.

The standard error is computed over per-user mean preference percentages.

The blue arrows visualize how many more bits BPG uses when > 50% users still prefer our result.

Our GC models with C = 4 are preferred to BPG even when images produced by BPG use 95% and 124% more bits than those produced by our models for Kodak and RAISE1K, respectively.

Notably this is achieved even though there is a distribution shift between the training and testing set (recall that these GC models are trained on Open Images).

The gains of domain-specificity and semantic label maps (for training) becomes apparent from the results on Cityscapes:

Our GC models with C = 2 are preferred to BPG even when the latter uses 181% more bits.

For C = 4 the gains on Cityscapes are comparable to those obtained for GC on RAISE1K.

For all three data sets, BPG requires between 21 and 49% more bits than our GC models with C = 8.

The GC models produce images with much finer detail than BPG, which suffers from smoothed patches and blocking artifacts.

In particular, the GC models convincingly reconstruct texture in natural objects such as trees, water, and sky, and is most challenged with scenes involving humans.

AEDC and the MSE baseline both produce blurry images.

We see that the gains of our models are maximal at extreme bitrates, with BPG needing 95-181% more bits for the C = 2, 4 models on the three datasets.

For C = 8 gains are smaller but still very large (BPG needing 21-49% more bits).

This is expected, since as the bitrate increases the classical compression measures (PSNR/MS-SSIM) become more meaningful-and our system does not employ the full complexity of current state-of-the-art systems, as discussed next.

State-of-the-art on Kodak: We give an overview of relevant recent learned compression methods and their differences to our GC method and BPG in Table 1 in the Appendix.

BID33 also used GANs (albeit a different formulation) and were state-of-the-art in MS-SSIM in 2017, while the concurrent work of is the current state-of-the-art in image compression in terms of classical metrics (PSNR and MS-SSIM) when measured on the Kodak dataset (Kodak).

Notably, all methods except ours (BPG, Rippel et al., and Minnen et al.) employ adaptive arithmetic coding using context models for improved compression performance.

Such models could also be implemented for our system, and have led to additional savings of 10% in .

Since Rippel et al. and Minnen et al. have only released a selection of their decoded images (for 3 and 4, respectively, out of the 24 Kodak images), and at significantly higher bitrates, a comparison with a user study is not meaningful.

Instead, we try to qualitatively put our results into context with theirs.

In Figs. 12-14 in the Appendix, we compare qualitatively to BID33 .

We can observe that even though BID33 use 29-179% more bits, our models produce images of comparable or better quality.

In FIG1 , we show a qualitative comparison of our results to the images provided by the concurrent work of , as well as to BPG (Bellard) on those images.

First, we see that BPG is still visually competitive with the current state-of-the-art, which is consistent with moderate 8.41% bitrate savings being reported by in terms of PSNR.

Second, even though we use much fewer bits compared to the example images available from , for some of them (Figs. 15 and 16) our method can still produce images of comparable visual quality.

Given the dramatic bitrate savings we achieve according to the user study (BPG needing 21-181% more bits), and the competitiveness of BPG to the most recent state-of-the-art , we conclude that our proposed system presents a significant step forward for visually pleasing compression at extreme bitrates.

Sampling the compressed representations: In FIG1 we explore the representation learned by our GC models (with C = 4), by sampling the (discrete) latent space ofŵ.

When we sample uniformly, and decode with our GC model into images, we obtain a "soup of image patches" which reflects the domain the models were trained on (e.g. street sign and building patches on Cityscapes).

Note that we should not expect these outputs to look like normal images, since nothing forces the encoder outputŵ to be uniformly distributed over the discrete latent space.

However, given the low dimensionality ofŵ (32 × 64 × 4 for 512 × 1024px Cityscape images), it would be interesting to try to learn the true distribution.

To this end, we perform a simple experiment and train an improved Wasserstein GAN (WGAN-GP) BID14 onŵ extracted from Cityscapes, using default parameters and a ResNet architecture.

4 By feeding our GC model with samples from the WGAN-GP generator, we easily obtain a powerful generative model, which generates sharp 1024 × 512px images from scratch.

We think this could be a promising direction for building high-resolution generative models.

In FIG5 in the Appendix, we show more samples, and samples obtained by feeding the MSE baseline with uniform and learned code samples.

The latter yields noisier "patch soups" and much blurrier image samples than our GC network.

Figure 6 : Mean IoU as a function of bpp on the Cityscapes validation set for our GC and SC networks, and for the MSE baseline.

We show both SC modes: RI (inst.), RB (box).

D + annotates models where instance semantic label maps are fed to the discriminator (only during training); EDG + indicates that semantic label maps are used both for training and deployment.

The pix2pixHD baseline BID43 was trained from scratch for 50 epochs, using the same downsampled 1024 × 512px training images as for our method.

The heatmaps in the lower left corners show the synthesized parts in gray.

We show the bpp of each image as well as the relative savings due to the selective generation.

In FIG2 we present example Cityscapes validation images produced by the SC network trained in the RI mode with C = 8, where different semantic classes are preserved.

More visual results for the SC networks trained on Cityscapes can be found in Appendix F.7, including results obtained for the RB operation mode and by using semantic label maps estimated from the input image via PSPNet .Discussion: The quantitative evaluation of the semantic preservation capacity (Fig. 6 ) reveals that the SC networks preserve the semantics somewhat better than pix2pixHD, indicating that the SC networks faithfully generate texture from the label maps and plausibly combine generated with preserved image content.

The mIoU of BPG, AEDC, and the MSE baseline is considerably lower than that obtained by our SC and GC models, which can arguably be attributed to blurring and blocking artifacts.

However, it is not surprising as these baseline methods do not use label maps during training and prediction.

In the SC operation mode, our networks manage to seamlessly merge preserved and generated image content both when preserving object instances and boxes crossing object boundaries (see Appendix F.7).

Further, our networks lead to reductions in bpp of 50% and more compared to the same networks without synthesis, while leaving the visual quality essentially unimpaired, when objects with repetitive structure are synthesized (such as trees, streets, and sky).

In some cases, the visual quality is even better than that of BPG at the same bitrate.

The visual quality of more complex synthesized objects (e.g. buildings, people) is worse.

However, this is a limitation of current GAN technology rather than our approach.

As the visual quality of GANs improves further, SC networks will as well.

Notably, the SC networks can generate entire images from the semantic label map only.

Finally, the semantic label map, which requires 0.036 bpp on average for the downscaled 1024 × 512px Cityscapes images, represents a relatively large overhead compared to the storage cost of the preserved image parts.

This cost vanishes as the image size increases, since the semantic mask can be stored as an image dimension-independent vector graphic.

We proposed and evaluated a GAN-based framework for learned compression that significantly outperforms prior works for low bitrates in terms of visual quality, for compression of natural images.

Furthermore, we demonstrated that constraining the application domain to street scene images leads to additional storage savings, and we explored combining synthesized with preserved image content with the potential to achieve even larger savings.

Interesting directions for future work are to develop a mechanism for controlling spatial allocation of bits for GC (e.g. to achieve better preservation of faces; possibly using semantic label maps), and to combine SC with saliency information to determine what regions to preserve.

In addition, the sampling experiments presented in Sec. 6.1 indicate that combining our GC compression approach with GANs to (unconditionally) generate compressed representations is a promising avenue to learn high-resolution generative models.

When encoding the channels ofŵ to a bit-stream, we use an arithmetic encoder where frequencies are stored for each channel separately and then encode them in a static (non-adaptive) manner, similar to BID2 .

In our experiments, this leads to 8.8% smaller bitrates compared to the upper bound.

We compress the semantic label map for SC by quantizing the coordinates in the vector graphic to the image grid and encoding coordinates relative to preceding coordinates when traversing object boundaries (rather than relative to the image frame).

The so-obtained bitstream is then compressed using arithmetic coding.

To ensure fair comparison, we do not count header sizes for any of the baseline methods throughout.

For the GC, the encoder E convolutionally processes the image x and optionally the label map s, with spatial dimension W × H, into a feature map of size W /16 × H /16 × 960 (with 6 layers, of which four have 2-strided convolutions), which is then projected down to C channels (where C ∈ {2, 4, 8} is much smaller than 960).

This results in a feature map w of dimension W /16 × H /16 × C, which is quantized over L centers to obtain the discreteŵ.

The generator G projectsŵ up to 960 channels, processes these with 9 residual units BID17 at dimension W /16 × H /16 × 960, and then mirrors E by convolutionally processing the features back to spatial dimensions W × H (with transposed convolutions instead of strided ones).Similar to E, the feature extractor F for SC processes the semantic map s down to the spatial dimension ofŵ, which is then concatenated toŵ for generation.

In this case, we consider slightly higher bitrates and downscale by 8× instead of 16× in the encoder E, such that dim(ŵ) = W /8 × H /8 × C. The generator then first processesŵ down to W /16 × H /16 × 960 and then proceeds as for GC.For both GC and SC, we use the multi-scale architecture of BID43 for the discriminator D, which measures the divergence between p x and p G(z) both locally and globally.

We adopt the notation from BID43 to describe our encoder and generator/decoder architectures and additionally use q to denote the quantization layer (see Sec. 3 for details).

The output of q is encoded and stored.• Encoder GC: c7s1-60,d120,d240,d480,d960,c3s1-C,q• Encoders SC:-Semantic label map encoder: c7s1-60,d120,d240,d480,d960-Image encoder: c7s1-60,d120,d240,d480,c3s1-C,q,c3s1-480,d960The outputs of the semantic label map encoder and the image encoder are concatenated and fed to the generator/decoder.• Generator/decoder: c3s1-960,R960,R960,R960,R960,R960,R960,R960, R960, R960,u480,u240,u120,u60,c7s1- DISPLAYFORM0 Figure 8: Structure of the proposed SC network.

E is the encoder for the image x and the semantic label map s. q quantizes the latent code w toŵ.

The subsampled heatmap multipliesŵ (pointwise) for spatial bit allocation.

G is the generator/decoder, producing the decompressed imagex, and D is the discriminator used for adversarial training.

F extracts features from s .

We employ the ADAM optimizer BID23 ) with a learning rate of 0.0002 and set the mini-batch size to 1.

Our networks are trained for 150000 iterations on Cityscapes and for 280000 iterations on Open Images.

For normalization we used instance normalization BID42 , except in the second half of the Open Images training, we train the generator/decoder with fixed batch statistics (as implemented in the test mode of batch normalization BID19 ), since we found this reduced artifacts and color shift.

To train GC models (which do not require semantic label maps, neither during training nor for deployment) for compression of diverse natural images, we use 200k images sampled randomly from the Open Images data set BID24 ) (9M images).

The training images are rescaled so that the longer side has length 768px, and images for which rescaling does not result in at least 1.25× downscaling as well as high saturation images (average S > 0.9 or V > 0.8 in HSV color space) are discarded (resulting in an effective training set size of 188k).

We evaluate these models on the Kodak image compression dataset (Kodak) (24 images, 768 × 512px), which has a long tradition in the image compression literature and is still the most frequently used dataset for comparisons of learned image compression methods.

Additionally, we evaluate our GC models on 20 randomly selected images from the RAISE1K data set BID11 , a real-world image dataset consisting of 8156 high-resolution RAW images (we rescale the images such that the longer side has length 768px).

To investigate the benefits of having a somewhat constrained application domain and semantic labels at training time, we also train GC models with semantic label maps on the Cityscapes data set BID9 (2975 training and 500 validation images, 34 classes, 2048 × 1024px resolution) consisting of street scene images and evaluate it on 20 randomly selected validation images (without semantic labels).

Both training and validation images are rescaled to 1024 × 512px resolution.

To evaluate the proposed SC method (which requires semantic label maps for training and deployment) we again rely on the Cityscapes data set.

Cityscapes was previously used to generate images form semantic label maps using GANs .

The preprocessing for SC is the same as for GC.

In the following Sections, F.1, F.2, F.3, we show the first five images of each of the three datasets we used for the user study, next to the outputs of BPG at similar bitrates.

Secs.

F.4 and F.5 provide visual comparisons of our GC models with BID33 and , respectively, on a subset of images form the Kodak data set.

In Section F.6, we show visualizations of the latent representation of our GC models.

Finally, Section F.7 presents additional visual results for SC.

Figure 12 : Our model loses more texture but has less artifacts on the knob.

Overall, it looks comparable to the output of BID33 , using significantly fewer bits.

Ours, 0.0651bpp Rippel et al., 0.0840bpp (+29%) Figure 13 : Notice that compared to BID33 , our model produces smoother lines at the jaw and a smoother hat, but proides a worse reconstruction of the eye.

Ours, 0.0668bpp Rippel et al., 0.0928bpp (+39%) Figure 14 : Notice that our model produces much better sky and grass textures than BID33 , and also preserves the texture of the light tower more faithfully.

GC (C = 4) MSE (C = 4) Figure 20 : We train the same architecture with C = 4 for MSE and for generative compression on Cityscapes.

When uniformly sampling the (discrete) latent spaceŵ of the models, we see stark differences between the decoded images G(ŵ).

The GC model produces patches that resemble parts of Cityscapes images (street signs, buildings, etc.), whereas the MSE model outputs looks like low-frequency noise.

We experiment with learning the distribution ofŵ = E(x) by training an improved Wasserstein GAN BID14 .

When sampling form the decoder/generator G of our model by feeding it with samples from the improved WGAN generator, we obtain much sharper images than when we do the same with an MSE model.

Reconstructions obtained by our SC network using semantic label maps estimated from the input image via PSPNet .

We collect here additional results for the discussion with the reviewers, so that they are easily found.

We will integrate these results into the paper.

In Table 2 we compute the PSNR on the Cityscapes test set, when varying the entropy constraint (i.e. changing C), and the two extremes (a) when MSE is only optimized and (b) when the GAN loss is only optimized.

The first three rows shows as the entropy constraint is relaxed, the network can more easily optimize the distortion term leading to a higher PSNR.

The fourth row shows that when optimizing for MSE only (see FIG2 for a qualitative example) we obtain superior PSNR (but at the expense of visual quality with blurry images).

The last rows shows that when turning off distortion losses (λ = 0), the network does optimize reconstruction at all.

Here we observe that the GAN "collapses" and outputs repetitive textures (see FIG1 ), suggesting the distortion losses are crucial for stability of training.

In FIG2 we show the loss curves when training our GC, C = 8 model on OpenImages BID24 .

We note that the loss fluctuates heavily across iterations due to the small batch size (one), but the smoothed losses are stable.

For all our experiments, both on Cityscapes and OpenImages, we kept the weights of the losses and ratio between discriminator/generator iterations constant and at point did our (GC and SC) models collapse during training for either dataset.

Our GC, C = 2, H(ŵ) < 0.018bpp 21.46Our GC, C = 4, H(ŵ) < 0.036 23.17Our GC, C = 8, H(ŵ) < 0.072 24.93 MSE bl., C = 4, H(ŵ) < 0.036 25.91 GC, λ = 0, C = 8, H(ŵ) < 0.072 11.65 Table 2 : We consider the effect of the GAN loss, the distortion losses and the entropy constraint on the PSNR of the trained model on the Cityscapes dataset.

Figure 27: We show convergence plots for the distortion losses from training our GC C = 8) channel model on OpenImages.

<|TLDR|>

@highlight

GAN-based extreme image compression method using less than half the bits of the SOTA engineered codec while preserving visual quality