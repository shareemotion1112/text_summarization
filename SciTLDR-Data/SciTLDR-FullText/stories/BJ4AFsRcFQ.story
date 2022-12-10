Recent image style transferring methods achieved arbitrary stylization with input content and style images.

To transfer the style of an arbitrary image to a content image, these methods used a feed-forward network with a lowest-scaled feature transformer or a cascade of the networks with a feature transformer of a corresponding scale.

However, their approaches did not consider either multi-scaled style in their single-scale feature transformer or dependency between the transformed feature statistics across the cascade networks.

This shortcoming resulted in generating partially and inexactly transferred style in the generated images.

To overcome this limitation of partial style transfer, we propose a total style transferring method which transfers multi-scaled feature statistics through a single feed-forward process.

First, our method transforms multi-scaled feature maps of a content image into those of a target style image by considering both inter-channel correlations in each single scaled feature map and inter-scale correlations between multi-scaled feature maps.

Second, each transformed feature map is inserted into the decoder layer of the corresponding scale using skip-connection.

Finally, the skip-connected multi-scaled feature maps are decoded into a stylized image through our trained decoder network.

Recent image style transferring methodsJohnson et al. (2016) ; BID18 improved image generating speed up to sub-realtime processing by learning a feed-forward network of a single style or several fixed stylesDumoulin et al. (2017) .

Huang et al.

Huang & Belongie (2017) proposed an adaptive instance normalization layer (AdaIN) that adaptively transforms the statistics of an encoded content feature into that of a target style feature and they achieved style transferring into arbitrary input target style.

However, they did not consider multi-scaled style characteristics of an imageGatys et al. (2016) but only a single scale feature in differentiating styles inside AdaIN layer.

Li et al.

Li et al. (2017b) proposed to use cascade networks that cumulatively transfer the multi-scaled style characteristics by using a network per scale as shown in FIG0 (a).

They also transformed correlation between channels of feature map by using their whitening and coloring transformer (WCT).

However, their cascade scheme requires multiple feed-forward passes to produce a stylized image and it is not guaranteed that the transferred style through a network is preserved after going through the subsequent networks because of inter-scale dependency in the multi-scaled styles of an image.

Therefore, transferring multi-scaled style without interference between scales is still remained to study.

In this paper, we propose an improved feed-forward network structure ( FIG0 ) and a multi-scaled style transferring method, called total style transfer, to efficiently perform style transfer in all scales of feature maps through a single feed-forward pass.

Our work has the following contributions.• Transforming both intra-scale and inter-scale statistics of multi-scaled feature map: There exist both of inter and intra-correlations in the encoded multi-scaled feature map as shown in fig.2 (b).

Therefore, we match the second-order statistics, i.e., mean and covariance, of the encoded multi-scaled feature map considering the correlations not only between channels in each scale (intra-scale correlation) but also between scales (inter-scale correlation).

Our feature transformer makes the transformed feature map closer to the target style feature map and this results in an output style closer to the target style.

Figure 2: Correlation between channels in the multi-scaled feature map of the input image (a) extracted from the pre-trained VGG16 BID16 .

The area corresponding to each scale of feature map is divided into red lines.

In case of intra-scale feature transform, the diagonal rectangles on the correlation matrix are used.

In case of inter-scale feature transform, entire region of the correlation matrix is considered.• Decoder learning with multi-scaled style loss: we use a multi-scaled style loss consistent to the feature transformer, i.e., mean and covariance loss between the concatenated feature map ( FIG1 ).

Using our multi-scaled style loss allows the decoder network to generate an output image of co-occurring multi-scale patterns which is better style expression than independently occurring scale patterns on the image that the existing methods generated.• Multi-scaled style transfer with a single feed-forward network: we use skip-connections for each decoder layer as shown in FIG0 (b) to consider the transformed feature map as well as the decoded feature map.

By doing this, the style of scale corresponding to the layer and the transferred multi-scaled style so far are optimally merged into the next layer.

Therefore, our method transfers multi-scaled style through a feed-forward pass in a single network instead of multiple feed-forward passes of cascade networks ( FIG0 ) without considering inter-scale correlation.

In the remained of this paper, we review previous works closely related to this work in Sec. 2, our multi-scaled style transforming method is described in Sec. 3, the effectiveness of our method is tested and proven by a bundle of experiments in Sec. 4, and this work is concluded in Sec. 5.

Gatys et al. BID2 represented content and style features of an image using a deep feature map, i.e., the filtered responses of a learned convolutional neural network (CNN).

To stylize an input image, they performed pixel-wise optimization of the image to reduce the feature losses of BID9 interpreted the process of generating a stylized image by matching Gram matrix BID2 as a problem of maximum mean discrepancy (MMD) specifically with a second-order polynomial kernel.

Using a feed-forward neural network BID7 ; BID18 moved the time consuming online optimization process into an offline feed-forward network learning to speed up the image generating speed of the previous method BID2 .

The generated style quality was also improved by using instance normalization (IN) BID19 .

Dumoulin et al. extended the previous single style network to transfer multiple styles.

They used conditional instance normalization (CIN) layers in a single network.

As selecting learnable affine parameters corresponding the specific style in the CIN layers, the feed-forward network transfers the selected style.

This method achieved generation of pre-trained multiple styles with a single network.

To generalize a single network for arbitrary style transfer, Huang et al. BID5 proposed to use a feature transformer called adaptive instance normalization (AdaIN) layer between encoder and decoder networks.

Once feature maps of content and target style images are encoded, AdaIN directly adjusts the mean and standard deviation of a content feature map into those of a target style feature map, and then the adjusted feature map is decoded into an output image of the target style.

Li et al. BID10 further improved the arbitrary stylization by using covariance, instead of standard deviation, which considers the correlation between feature channels.

To transfer multi-scaled style, Li et al. BID10

3.1 MULTI-SCALE FEATURE TRANSFORM As described in BID2 , each scaled feature of CNN represents different style characteristics of an image.

So, we utilize multiple feature transformers for each scale feature to transfer total style characteristics of an image.

In this section, we explain two schemes of our total style transfer, i.e, intra-scale and inter-scale feature transform, with a single feed-forward network.3.1.1 INTRA-SCALE FEATURE TRANSFORM Our intra-scale feature transform is a set of independent single-scale style transforms as an extended multi-scale version of the single-scale correlation alignment of CORAL BID17 or WCT Li et al. (2017b of style image, where C i , H c,i (or H s,i ), and W c,i (or W s,i ) represent the number of channels, spatial height, and width of i-th scale features respectively.

For a single-scale style transform with these features, CORAL or WCT performs (1) style normalization and (2) stylization sequentally.

In the style normalization step, first zero-centered featureF c,i ∈ R Ci×(Hc,i×Wc,i) of the content feature F c,i is calculated and then the content feature F c,i is normalized intoF c,i by using its own covariance matrix cov(F c,i ) ∈ R Ci×Ci as eq.1.

DISPLAYFORM0 In the stylization step, the normalized content featureF c,i is stylized into F cs,i by using the square root of covariance matrix cov(F s,i ) ∈ R Ci×Ci of zero-centered style featureF s,i and spatial mean µ s,i ∈ R Ci×1 of the style feature F s,i as eq.2.

DISPLAYFORM1 Our intra-scale transform method applies the above single-scale transform independently to each scaled feature for i = 1..3 corresponding to {relu_1_2, relu_2_2, relu_3_3} layers.

Then, those transformed features F cs,i , i = 1..3 are inserted into the subsequent decoder through skip-connenction.

More detail about skip-connection will be described in Sec.3.1.3.

As shown in fig.2 (b), there exists not only inter-channel correlation in a certain scale feature but also inter-scale correlation between multi-scale features.

These correlations should be considered in order to transfer total style characteristics of an image.

CORAL BID17 or WCT Li et al. (2017b) did not consider inter-scale correlation but only inter-channel correlation.

Therefore, we propose inter-scale feature transformer which considers more style characteristics of image for style transfer.

To perform feature transform considering both inter-channel and inter-scale correlations, we apply the intra-scale feature transform of Sec.3.1.1 to the concatenated feature F c ∈ R i Ci×(Hc,1×Wc,1) of content image and F s ∈ R i Ci×(Hs,1×Ws,1) of style image (eq.3) instead of independently applying to each scaled features F c,i and F s,i of Sec.3.1.1.

DISPLAYFORM0 As shown in FIG1 , the content features F c,i and style features F s,i for i = 1..3 are spatially upsampled into U (F c,i ) and U (F s,i ) of a common size (we use the largest size of F c,1 or F s,1 corresponding to {relu_1_2}) and concatenated into F c and F s respectively along the channel axis.

After going through a transformer, the transformed feature F cs is split and downsampled into F cs,i ∈ R Ci× (Hc,i×Wc,i) of the original feature size as shown FIG1 (b) and eq.4, DISPLAYFORM1 where, D i (f ) is a function which spatially downsamples f into H c,i × W c,i .

These features are inserted into the subsequent decoder through skip-connenction of Sec.3.1.3.

To utilize the transformed multi-scale features in generating output stylized image, decoding architecture of previous decoder network should be modified because each decoder layer has two input feature maps as FIG0 , one is a decoded feature map from the previous decoder layer, the other is a (intra-scale or inter-scale) transformed feature from the transformer.

We adopt skip-connection, which has been applied to several applications of computer vision field BID14 ; BID13 ; BID6 BID0 but not to image style transfer yet, to merge the two feature maps in decoding process as shown in FIG1 .

Skip-connected two scale features are optimally merged by learnable convolution layer and this improves the quality of the decoded image by considering multi-scale filter responses.

Our method is different from the previous cascade scheme of BID10 because we use a single encoder/decoder network, parallel transformers for each scale feature, and merges multi-scaled styles optimally while the cascade scheme needs several encoder/decoder networks (one network per scale feature) and sequentially transfers scaled styles from large to small scale at the risk of degradation in previously transferred scale of style.

Avatar-Net Sheng et al. (2018) also used a single decoder like ours but it sequentially applied feature transformers from large to small scale without considering possible degradation of the previously transferred scale.

We need an appropriate objective function for decoder network to generate a stylized image from the transformed feature map.

Among the existing losses such as Gram BID2 , Mean-Std Huang & Belongie (2017) , and reconstruction error BID10 , we adopt Mean-Std loss BID5 with some modification because of its consistency with AdaIN transformer.

Instead of using Mean-Std loss as it is, we use Mean-Covariance loss to additionally consider interchannel and inter-scale correlations, which is consistent with our feature transformers described in Sec.3.1.In case of using intra-scale feature transform (Sec.3.1.1), our style loss (eq.5) is calculated as the summation of mean loss BID5 and covariance loss, i.e., square root of Frobenius distance between covariance matrices of feature maps of output and target style images.

In case of using inter-scale feature transform (Sec.3.1.1), the summation of mean and covariance losses of the concatenated features are used as the style loss (eq.6).

DISPLAYFORM0 DISPLAYFORM1 where subscript o represents of output stylized image.

We used VGG16 feature extractor BID16 as the encoder and a mirror-structured network as the decoder of our style transfer network.

Our decoder network has 2 times larger number of channels in the corresponding layer of skipconnections than the previous methods Dumoulin et al. (2017); BID5 .

{relu_1_2, relu_2_2, relu_3_3, relu_4_3} layers were used in calculating style loss and {relu_3_3} layer in calculating content loss.

Here, we used the same content loss of BID2 and our multi-scaled style loss in Sec.3.2.

For training data set, MS-COCO train2014 BID11 and Painter By Numbers BID8 were used as content image set and large style image set respectively.

Each dataset consists of about 80,000 images.

And we used an additional small style image set of 77 style images to verify the effect of our proposed method as the number of training style increases.

Each image was resized into 256 pixels in short side maintaining the original aspect ratio in both training and test phases, and, only for training phase, randomly cropped into (240, 240) pixels to avoid boundary artifact.

We trained networks with 4 batches of random (content, style) image pairs, 4 epochs, and learning rate of 10 .

And all experiments were performed on Pytorch 0.3.1 framework with NVIDIA GTX 1080 Ti GPU card, CUDA 9.0, and CuDNN 7.0.

In order to verify the effect of our multi-scale feature transform for varying number of training style images, we trained two networks, one with the small style image set of 77 images and the other with the large style image set of about 80,000 images.

Then we compared the output stylized images of the networks.

Fig.4 shows an example of the output stylized images using our intra-scale or inter-scale feature transform method.

With the network trained by a small style image set, the result images generated by our intra-scale transform ( fig.4 (c) ) show very similar texture style to the target style images ( fig.4 (b) ).

And those by our inter-scale transform ( fig.4 (d) ) show even better style of texture.

With the network trained by a large style image set ( fig.4 (e,d) ), the result images also show the same tendency that inter-scale is better in expressing the texture of target style.

Because of the existing correlations between scales as shown in fig.2 (b) , inter-scale feature transform which considers interscale correlations shows the better quality of style than intra-scale transform.

To verity the effect of skip-connections in our style transfer network, we trained three different networks.

The first one has the conventional single layer encoder/transformer/decoder architecture Huang & Belongie FORMULA0 ; BID10 which a sinlge feature transformer on {relu_3_3} without skip-connection.

The second one has muti-scale feature transformers on {relu_3_3,relu_2_2} and one skip-connection on {relu_2_2}. The last one has multiscale feature transformers on {relu_3_3,relu_2_2,relu_1_2} and two skip-connections on {relu_2_2,relu_1_2}. Fig.5 shows an example of the output stylized image from the three different networks.

As the number of skip-connections increases from fig.5 (c) to (e), the style loss decreases from 0.535 to 0.497 and, accordingly, the color tone of the stylized image is getting better matched to the target style ( fig.5(b) ) and small patterns appear.

To clarify the contributions of the skip-connected feature from the transformer and the decoded feature from the previous scale of decoder layer, we observed the absolute values of loss gradients with respect to the convolution weights of skip-connected decoder layers during the network learning process.

As shown in fig.6(a) , the gradient values for the skip-connected feature (channel indices from 129 to 256) on {relu_2_2} layer of decoder network are much larger than those for the decoded feature (channel indices from 1 to 128) at the beginning of training.

This means that the skip-connected feature which already has target style through transformer dominantly affected to the decoder learning at the start of training phase.

This happens because the previous decoder layer {relu_3_3} has random initial weights and outputs noisy feature at the start of training phase.

As iteration goes, the gradient values for both features became similar to each other.

This means that both skip-connected feature and decoded feature were equally utilized to generate an image of multi-scaled style.

FIG0 shows that the gradient values for the skip-connected feature (channel indices from 65 to 128) are smaller than those for the decoded feature (channel indices from 1 to 64) at the latter decoder layer.

This means that the decoded feature of the latter decoder layer already has accumulated multi-scaled styles by the previous skip-connection and this resulted in the less impact of the skip-connected feature.

However, using the skip-connection with the stylized feature of smaller scale has a certain effect on the result image in color tone matching as shown in fig.5(d,e) .

We compared the image quality of our method with those of the existing methods BID2 ; BID5 ; BID10 .

We took the output stylized images for BID2 after 700 iterations of Adam optimization with learning rate 10 DISPLAYFORM0 , for BID5 with the same setting of our method except transformer and loss, and for BID10 with style strength α = 0.6 (as mentioned in their work) and 3 cascade networks of VGG16 structure.

It e ra ti o n s( × 5 0 0 ) C h a n n e l s (b) Learning Gradient of relu_1_2 C h a n n e l s It e ra ti o n s( × 5 0 0 ) Figure 6 : Amplitude of loss gradients with resprct to the convolution weights in the skip-connected decoder layers during the learning process: The gradients are drawn every 500 iterations.

The former half of the channels are for decoded feature from the previous scale and the latter are for skip-connected feature from transformer.

Based on the gradients of 1st skip-connected layer (a) and 2nd skip-connected layer (b), the skip-connected (transformed) feature highly seems to affect to the decoder in initial interations but both decoded and transformed features samely affect as iteration goes.

And the latter decoder layer (b) is less affected by the skip-connected feature than the former layer (a).

FIG6 shows the generated images from the existing and our intra/inter-scale feature transform methods.

Compared to the online optimization method BID2 FIG6 ), the other methods based on feed-forward network generated images of somewhat degraded style quality ( FIG6 ).

However, thanks to our muti-scaled style transfer which considering inter-channel or inter-scale correlation, texture detail and color tone of the generated image of our method with inter-scale ( FIG6 ) or intra-scale feature transform ( FIG6 ) are more similar to the target style than those of single-scale style transfer without considering inter-channel correlation BID5 ( FIG6 ) are.

BID2 , (b) is , (c) is BID5 and (d) is BID10 .Compared to BID10 , the generated images of our method present the styles closer to the target styles because our methods trained a decoder network to minimize style loss between target style and output images while BID10 trained its decoder network to minimize reconstruction loss of content images.

We also compared our method to Avatar-Net Sheng et al. (2018) that performs multi-scale feature transform with a single decoder network.

For a fair comparison, as the structure of Avatar-Net Sheng et al. (2018), we used VGG19 network BID16 up to {relu_4_1} layer as the encoder and its mirrored structure as the decoder, and we also used additional style loss (eq.5 or eq.6) on image-level which is corresponding to the image reconstruction loss of Avatar-Net.

As shown in FIG8 , Both our intra (e) and inter (f) methods generated the stylized images with both detailed shapes of content images (a) and multi-scaled strokes of target style images (b).

In contrast, the generated images of Avatar-Net (c, d) show somewhat deemed content shapes and blurred or burnt color patterns without detailed strokes.

And selecting appropriate patch size corresponding to the scale of style pattern was necessary in Avatar-Net (in second row of FIG8 , {patch size=1} (c) did not show large square patterns but {patch size=3} (d) did.) while our method did not require any scale matching parameter due to multi-scaled skip-connections in the decoder network.

For a quantitative comparison, we compared the content and style losses of the generated images of the existing methods and ours.

Table.

1 shows the measured average (standard deviation) losses across 847 tests of style transfer.

Online optimization method BID2 achieved the smallest style loss with a low content loss.

Among the feed-forward networks trained with a small style image set, achieved the lowest style loss (red colored numbers).

This is because it has a learnable transformer and this resulted in a most optimized transformer.

However, it is not extendable to arbitrary style transfer.

Among the arbitrary style transferring methods, our method achieved the lowest style loss (blue colored numbers) with inter-scale feature transformer, and the second lowest style loss (green colored numbers) with intra-scale feature transformer.

Among the feed-forward networks trained with a large style image set, our method shows the lowest style loss with inter-scale feature transformer and the second lowest style loss with intra-scale feature transformer in the same manner of the result with a small style image set.

For the content loss with large style image set, the best method in style loss (our-inter) shows the highest content loss and the second best method (our-intra) shows the lowest content loss.

This interesting result can be interpreted that inter-scale correlation has not only style of an image but also content of the image.

Tasks of transferring style and preserving content are a trade-off in feedforward network methods.

The result of with small style image set also shows the same content/style trade-off.

Therefore, we can select either inter-scale or intra-scale feature transformer according to user preference or purpose of application.

Our method achieved 31% less encoder/decoder feed-forward time (4.4 ms in average of 1000 trials with images of 240 by 240 pixels) and 4% less number of parameters (3,655,296 parameters) than the existing cascade network scheme Li et al. (2017b) (6.4 ms, 3,769,856 parameters) .

In this paper, we proposed a total style transfer network that generates an image through a single feed-forward network by utilizing multi-scale features of content and style images.

Our intra-scale feature transformer transfers multi-scale style characteristics of the target style image and our inter-scale feature transformer transfers even more style characteristics of inter-scale correlation into the content image.

By using our intra/inter scale feature transform, our total style transfer network achieved the lowest style loss among the existing feed-forward network methods.

In addition, we modified the feed-forward network structure by using skip-connections which make our decoder network to utilize all transformed multi-scale features.

This modification allowed a single feed-forward network to generate image of multi-scaled style without using multiple feedforward networks of cascade scheme, and resulted in the reduced test time by 31% and memory consumption by 4% compared to cascade network scheme.

@highlight

A paper suggesting a method to transform the style of images using deep neural networks.