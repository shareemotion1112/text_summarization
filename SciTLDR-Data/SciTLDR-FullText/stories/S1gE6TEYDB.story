Recent image super-resolution(SR) studies leverage very deep convolutional neural networks and the rich hierarchical features they offered, which leads to better reconstruction performance than conventional methods.

However, the small receptive fields in the up-sampling and reconstruction process of those models stop them to take full advantage of global contextual information.

This causes problems for further performance improvement.

In this paper, inspired by image reconstruction principles of human visual system, we propose an image super-resolution global reasoning network (SRGRN) to effectively learn the correlations between different regions of an image, through global reasoning.

Specifically, we propose global reasoning up-sampling module (GRUM) and global reasoning reconstruction block (GRRB).

They construct a graph model to perform relation reasoning on regions of low resolution (LR) images.

They aim to reason the interactions between different regions in the up-sampling and reconstruction process and thus leverage more contextual information to generate accurate details.

Our proposed SRGRN are more robust and can handle low resolution images that are corrupted by multiple types of degradation.

Extensive experiments on different benchmark data-sets show that our model outperforms other state-of-the-art methods.

Also our model is lightweight and consumes less computing power, which makes it very suitable for real life deployment.

Image Super-Resolution (SR) aims to reconstruct an accurate high-resolution (HR) image given its low-resolution (LR) counterpart.

It is a typical ill-posed problem, since the LR to HR mapping is highly uncertain.

In order to solve this problem, a large number of methods have been proposed, including interpolation-based (Zhang & Wu., 2006) , reconstruction-based (Zhang et al., 2012) , and learning-based methods (Timofte et al., 2013; Peleg & Elad., 2014; Schulter et al., 2015; Huang et al., 2015; Tai et al., 2017; Tong et al., 2017; Zhang et al., 2018a; Dong et al., 2016) .

In recent years, deep learning based methods have achieved outstanding performance in superresolution reconstruction.

Some effective residual or dense blocks Zhang et al., 2018b; Lim et al., 2017; Ledig et al., 2017; Ahn et al.; Li et al., 2018) have been proposed to make the network wider and deeper and achieved better results.

However, they only pay close attention to improving the feature extraction module, ignoring that the upsampling process with smaller receptive fields does not make full use of those extracted features.

Small convolution receptive field means that the upsampling process can only perform super-resolution reconstruction based on local feature relationships in LR.

As we all know, different features interact with each other, and features which are in different regions have corresponding effects on upsampling and reconstruction of a certain region.

That is to say that a lot of information is lost in the process of upsampling and reconstruction due to the limitation of the receptive field, although the network extracts a large number of hierarchical features which are from low frequency to high frequency.

Chariker et al. (2016; show that the brain generates the images we see based on a small amount of information observed by the human eye, ranther than acquiring the complete data from the point-by-point scan of the retina.

This process of generating an image is similar to a SR process.

According to their thought, we add global information in SR reconstruction and propose to use relational reasoning to implement the process that the human visual system reconstructs images with observed global information.

In general, extracting global information requires a large receptive field.

A large convolution receptive field usually requires stacking a large number of convolutional layers, but this method does not work in the upsampling and reconstruction process.

Because this will produce a huge number of parameters.

Based on the above analysis, we propose an image super-resolution global reasoning network (SR-GRN) which introduces the global reasoning mechanism to the upsampling module and the reconstruction layer.

The model can capture the relationship between disjoint features of the image with a small respective field, thereby fully exploits global information as a reference for upsampling and reconstruction.

We mainly propose global reasoning upsampling module (GRUM) and global reasoning reconstruction block (GRRB) as the core structure of the network.

GRUM and GRRB first convert the LR feature map into N nodes, each of which not only represents a feature region in the LR image, but also contains the influence of pixels in other regions on this feature.

Then they learn the relationship between the nodes and fuse the information of each node in a global scope.

After that, GRUM learns the relationship between the channels in each node and amplifies the number of channels for the upsampling process.

And then they convert N nodes into pixels with global reasoning information.

Finally, GRUM and GRRB complete the upsampling and reconstruction process respectively.

In general, our work mainly has the following three contributions:

??? We propose an image super-resolution global reasoning network (SRGRN) which draws on the idea of image reconstruction principles of human visual system.

We mainly focus on the upsampling module and the reconstruction module.

The model reconstructs SR images based on relational reasoning in a global scope.

??? We propose a global reasoning upsampling module (GRUM) and global reasoning reconstruction block (GRRB), which construct a graph model to implement the relational reasoning among the feature regions in an image via 1D and 2D convolution, and finally adds the information obtained by global reasoning to each pixel.

It can provide more contextual information to help generate more accurate details.

??? Our proposed GRUM and GRRB are lightweight, which makes it suitable for real life deployment.

More importantly, GRUM and GRRB balance the number of parameters and the reconstruction performance well.

They can be easily inserted into other models.

Deep CNN for SR and upsampling methods.

Deep learning has achieved excellent performance in image super-resolution tasks.

For the first time, Dong et al. applied convolutional neural networks to image SR.

After this, Kim et al. proposed VDSR (Kima et al., 2016) and DRCN (Kim et al., 2016) which introduced residual learning to make the network depth reach 20 layers achieved significant improvement.

And then more and more researchers are starting to pay attention to the improvement of the network feature extraction part.

Lim et al. (2017) proposed EDSR and MDSR, which introduce residual scaling and remove unnecessary modules from the residual block.

Concerned that the previous models only adopt the feather of the last CNN, Zhang et al. (2018b) proposed residual dense network to make full use of hierarchical features from each Conv layer.

The above and most of the subsequent networks implement the upsampling based on either transposed convolution (Zeiler et al., 2010; Zeiler & Fergus., 2014) or sub-pixel convolution (Shi et al., 2016) .

Although these models have achieved good results, there exists a problem that these upsampling methods have only a small receptive field.

This means that upsampling can only take advantage of contextual information within a small area.

Recently, researchers propose some new super-resolution upsampling process.

LapSRN (Lai et al., 2017) allows low-resolution images to be directly input into the network for step-by-step amplification.

Haris et al. (2018) exploit iterative up-and-down sampling layers and propose DBPN.

Li et al. (2019) further explore the application of feedback mechanism (weight sharing) in SR and propose the SRFBN.

These models have achieved a better reconstruction performance.

However, the Conv layers in these upsampling modules still have only a small receptive field.

Global reasoning machansim.

Recently, graph-based deep learning methods have begun to be widely used to solve relation reasoning.

Santoro et al. propose Relation Networks (RN) (Santoro et al., 2017) to solve problems that depend on relational reasoning.

propose SIN, which implement a object detection using a graph model for structure inference.

Furthermore, model a Global Reasoning unit that consists of five convolutions for image classification, semantic segmentation and video action recognition task.

Considering that the human visual system generates images based on the observed global information is also a reasoning process.

Moreover, correlation between feature regions can be obtained through relational reasoning, which makes each pixel in the generated SR image jointly determined by the information in a global scope.

Therefore, we propose a global reasoning network for SR.

We will detail our SRGRN in next section.

According to Chariker et al. (2016; , there is only little information transmitted from the retina to the visual cortex, and then the brain will reconstruct the real-world images based on the information received.

We regard it as a reasoning process in a global scope.

For image SR, the upsampling module constructs SR images base on features in LR images, which is substantially similar to detecting the category of each pixel of a SR image and generating these pixels based on contextual information of corresponding LR image.

Due to the limitation of the convolution receptive field, only a small amount of contextual information can be utilized to generate HR images in most other models.

This leads that many details in the HR image are not fine.

Similarly, the above problem also exists in the reconstruction process.

To solve these problems, we simulate the reasoning process that exists in human visual system, and then propose SRGRN to make full use of the contextual information to recover accurate details, which is achieved by constructing graph model and reasoning the relationship between these regions in an image.

Figure 1 , our SRGRN includes feature extraction part, global reasoning upsample module(GRUM) and global reasoning reconstruction block(GRRB).

Let's denote I LR and I SR as the input and output of SRGRN.

The feature extraction part can use the relevant architecture of most other models.

Here we introduce the feature extraction part of the RDN (Zhang et al., 2018b) as an example.

where H F EX (??) denotes a series of operations of feature extraction part.

As with the previous work (Lim et al., 2017) , the number of GRUM depends on the scaling factor, The GRUM receives F L as input.

F S represents the output of the GRUM.

GRUM can be expressed by the following mathematical formula:

where H GRU M (??)denotes a series of operations of GRUM.

More details about GRUM will be given in Section 3.2.

We further conduct global reasoning reconstruction block(GRRB) to utilize the global contextual information to generate the output image.

GRRB can be expressed by the following mathematical formula:

where H GRRB (??) denotes a series of operations of GRRB.

More details about GRRB will be given in Section 3.3.

After the above operations, we get the corresponding SR image.

In this section, we present details about our proposed global reasoning upsample module(GRUM) in Figure 2 .

In order to help achieve relation reasoning, we map each image to a graph model We first need a function to construct N nodes in the oriented graph, each of which represents a region in the image.

In GRUM, we obtain relationship weights between these pixels through a 2D 1 ?? 1 convolution, and then convert the input F L into N nodes via element-wise product.

The benefits of this approach are mainly reflected in the following aspects: (1) It can not only aggregates a feature region of the input F L into a node, but also dig out the influence of other pixels in the image on this region.

This is equivalent to adding global guidance of the image to each node.

(2) Using convolution means that these relationship weights are trainable.

This process can be expressed by the following mathematical formula:

refers to N nodes with C channels.

After that, we use the 1D Conv -Leaky ReLu -1D Conv (CLC) structure to implement reasoning and interaction between N nodes in the graph.

The parameters in CLC refer to the adjacency matrix of the weighted oriented complete graph, which store the correlations between the nodes.

CLC can learn and reason the complex nonlinear relationship between nodes better than only one 1D Conv.

We use the following formula to describe the reasoning process between nodes:

where Conv(??) and LRelu(??) denote 1D convolution along node-wise and Leaky ReLU (Maas et al., 2013) operation respectively.

And then we use the bottleneck to achieve channel amplification.

The bottleneck receives Y N ??? R N ??C as input and redistributes these channels by modeling the relationship between the channels of each node, amplifying the number of channel to C ?? r 2 , where r is the upscaling factor.

The first convolution in bottleneck makes channel C drop toC = C/??, where ?? represents reduction ratio.

Then the second convolution makes the channel dimensionC grow to C ?? r 2 .

The bottleneck not only fits the complex relationships between channels better and redistributes channels more accurately, but also greatly reduces the number of parameters compared to the method of utilizing a single convolution.

We use the following formula to describe channel amplification:

where Y N C ??? R r 2 C??N refers to the output tensor.

In order to expand the resolution by pixelshuffle like ESPCN (Shi et al., 2016) , we need to retransform the N nodes (r 2 C ?? N ) which have implemented the relational reasoning into a space whose shape is C ?? H ?? W .

As above, we still learn a function to get a weight matrix whose shape is N ??HW through a 1 ?? 1 2D convolution, and then normalize the weight matrix along the column with softmax.

Finally, Y rC and W P can be obtained through:

where

C??H??W is a feature map where each pixel is associated with N nodes.

W P ??? R N ??HW is the normalized weight matrix.

The value of these weights ranges from 0 to 1.

This means that the reconstruction of each pixel is affected by N nodes to varying degrees.

Each pixel in the feature map contains information which is generated by global reasoning.

After pixelshuffle, the output is multiplied by a parameter ?? 1 and added to the upsampling result without global reasoning.

The initial value of ?? 1 is set to 0.

As the global reasoning module trains, the network will gradually learn to assign values to the ?? 1 , thereby fully exploiting global reasoning.

This process can be expressed by the following formula:

where H P S (??) denotes the operations of pixel shuffle and H U P (??) denotes the operations of subpixel convolution.

Finally, F S can be obtained by:

Figure 3: Global reasoning reconstruction block (GRRB) architecture

As shown in Figure 3 , the specific details are similar to GRUM.

We also construct a graph model for reconstruction block.

In GRRB, we first obtain the relationship weights W RN ??? R N ??rHrW between pixels of F S by 2D 1x1 convolution, and then aggregate the regions in F S into N nodes by element-wise product.

The output of this process Y RW ??? R N ??C can be formulated as:

After that, we use CLC to achieve the relationship reasoning between nodes.

Then we exploit the weight matrix W RP ??? R N ??rHrW = Sof tmax(V d (Conv(F S ))) obtained by learning to redistribute the information of N nodes to the pixels.

The output of this process Y RrC ??? R C??rH??rW can be obtained by:

where F CLC refers to the operations of CLC.

In addition, we apply the idea of residual connection in GRRB, which multiplys the information generated via global reasoning by a parameter ?? and then add it to the input feature map.

The output is given by:

The initial value of ?? is set to 0.

As the training progresses, the network assigns more weight to ??.

Finally, we input the feature map with global reasoning into the two Convs for reconstruction.

We can get the final output through:

where H RL (??) denotes the operations of two Convs.

In our proposed SRGRN, like the previous method (Lim et al., 2017) , the number of GRUM depends on the scaling factor.

For Conv layers with kernel size 3 ?? 3, we pad zeros to keep size fixed.

We set the reduction ratio in bottleneck as ??.

The number of nodes in the graph model is set to N. We utilize Leaky ReLu (Maas et al., 2013 ) with a negative slope of 0.2 as non-linear activation function.

The feature extraction part of the network are the same as the RDN (Zhang et al., 2018b) settings.

The final Conv layer has 1 or 3 output channels, as we output gray or color HR images.

4.1 SETTINGS Datasets and Metrics.

We train all our models using 800 training images in the DIV2K (Agustsson & Timofte., 2017) dataset, which contains high-quality 2K images that can be used for image superresolution task.

And We use five standard benchmark datasets to evaluate PSNR and SSIM (Wang et al., 2004 ) metrics: Set5 (Bevilacqua et al., 2012) , Set14 (Zeyde et al., 2010) , B100 (Martin et al., 2001 ), Urban100 (Huang et al., 2015) and Manga109 (Y. Matsui et al., 2017) .

The SR results are evaluated on Y channel of transformed YCbCr space.

Degradation Models.

In order to make a fair comparison with existing models, bicubic dowmsampling(denoted as BI) is regarded as a standard degradation model.

We use it to generate LR images with scaling factor ??2, ??3, and ??4 from ground truth HR images.

To fully demonstrate the effectiveness of our model, we also use two other degradation models and conduct special experiments for them.

Our second model, we defined it as BD, which blurs HR images with a Gaussian kernel of size 7 ?? 7 and a standard deviation of 1.6, and then downsamples the image with scaling factor ??3.

In addition to BI and BD, we also built the DN model, which first performs bicubic downsampling with scaling factor ??3 and then adds Gaussian noise with a noise level of 30.

Training Setting.

In each training batch, 16 LR RGB patches of size 48 ?? 48 are extracted as inputs.

We perform data enhancement on the training images, which are randomly rotated by 90

??? , 180

??? , 270

??? and flipped horizontally.

We use the Adam optimizer to update the parameters of the network with ?? 1 = 0.9, ?? 2 = 0.999, and ?? = 10 ???8 .

For all layers in the network, the initial learning rate is set to 0.0001, and then the learning rate is halved every 200 epochs.

We use the Pytorch framework to implement our model with Tesla P100.

Global reasoning upsampling module.

In order to verify the importance of the GRUM, we remove the GRUM from the network, leaving only the GRRB in the network for relation reasoning.

As shown in Table 1 , after removing GRUM, the performance of the network drops from 32.45 dB to 32.40 dB. When the Case Index is equal to 1, the corresponding model is the baseline model.

We can observe that after GRUM is added to the baseline model, the network performance is improved from 32.31 dB to 32.42 dB. It can be seen that although our baseline model has achieved quite good results, GRUM can still improve the performance by relation reasoning in upsample module.

This also indicates that relation reasoning can indeed result in better prformance.

These comparisons fairly demonstrate the effectiveness of the GRUM for SR tasks.

Global reasoning reconstruction block.

Then, we continue to study the effectiveness of GRRB for the network.

After we add the GRRB to the baseline model, GRRB improves the performance of the model from 32.31 dB to 32.40 dB. Furthermore, the model with GRUM has achieved good performance.

And it is difficult to obtain further improvements.

But when we add the GRRB to it, the network performance shows a significant improvement, and the PSNR value on Set5 increases from 32.42 dB to 32.45 dB. These indicates that it is very essential for our network.

Basic parameters.

Moreover, we also study the effects of two basic parameters N and ?? on the performance of the model.

As shown in Table 1 , we observe that larger N and smaller ?? would lead to higher performance.

Considering that larger N and smaller ?? will also bring more computation, we set 10 and 8 as the value of N and ?? respectively.

Figure 4 .

Although our SRGRN has less parameter number than that of EDSR, MDSR and D-DBPN, our SRGRN and SRGRN+ achieve higher performance, having a better tradeoff between model size and performance.

This demonstrates our method can well balance the number of parameters and the reconstruction performance.

For BI degradation model, we compare our proposed SRGRN and SRGRN+ with other seven stateof-the-art image SR methods in quantitative terms.

Following the previous works (Lim et al., 2017; Zhang et al., 2018b; Li et al., 2019) , we also introduced a self-ensemble strategy to further improve the performance.

We denote the self-ensemble method as SRGRN+.

A quantitative result for ??2, ??3, and ??4 is shown in Table 2 .

We compare our models with other state-of-the-art methods on PSNR and SSIM.

It can be seen that our proposed SRGRN outperforms other methods on all datasets without adding self-ensemble.

After adopting self-ensemble, the performance further improves on the basis of SRGRN, and it achieved the best on all datasets.

It is worth mentioning that SRFBN (Li et al., 2019) uses DIV2K+Flickr2K as their training set, which employs more training images than us.

Previous research has come to a conclusion that more data in training set leads to a better result.

However, their results are still not comparable to ours.

Although RDN (Zhang et al., 2018b ) is a state-of-the-art method, our SRGRN can achieve better performance in all datasets through relational reasoning in upsampling and reconstruction parts.

The quantitative results indicate that our GRUM and GRRB play a vital role in improving network performance. (Dong et al.) and VDSR (Kima et al., 2016) for BD and DN degradation model because of mismatched degradation model.

For BD and DN, there is no doubt that reconstruction has become more difficult.

As shown in Table  3 and Table 4 , in the case of images with a lot of artifacts and noise, our SRGRN can get a excellent performance.

This shows that SRGRN can effectively denoise and alleviate blurring artifacts.

And when added to self-ensemble, SRGRN+ can achieve a better improvement.

To prove that our SRGRN can be widely used in the real world and performs robustly, we also conduct SR experiments on representative real-world images.

We reconstruct some low resolution images in the real world that lack a lot of high frequency information.

Moreover, in this case, the original HR images are not available and the degradation model is unknown either.

Experiments show our SRGRN can recover finer and more faithful real-world images than other state-of-the-art methods under this bad condition.

This further reflects the superiority of relation reasoning.

In this paper, inspired by the process of reconstructing images from the human visual system, we propose an super-resolution global reasoning network (SRGRN) for image SR, which aims at completing the reconstruction of SR images through global reasoning.

We mainly propose global reasoning upsampling module (GRUM) and global reasoning reconstruction block (GRRB) as the core of the network.

The GRUM can give the upsampling module the ability to perform relational reasoning in a global scope, which allows this process to overcome the limitations of the receptive field and recover more faithful details by analyzing more contextual information.

The GRRB also enables the reconstruction block to make full use of the interaction between the regions and pixels to reconstruct SR images.

We exploit SRGRN not only to handle low resolution images that are corrupted by three degradation model, but also to handle real-world images.

Extensive benchmark evaluations demonstrate the importance of GRUM and GRRB.

It also indicates that our SRGRN achieves superiority over state-of-the-art methods through global reasoning.

Visual comparison with BI degradtion model.

As shown in Figure 5 , we show a visual comparison on 4?? SR.

For image "img_078" from Urban100, we observe that most methods, even RDN and SRFBN, cannot recover these lattices and suffer from extremely severe blurring artifacts.

Only our SRGRN can alleviate these blurring artifacts, recovers sharper and clearer edges and finer texture.

For image "MukoukizuNoChonbo" from Manga109, There are heavy blurrings artifacts in all comparison methods, and the outline of some letters are broken.

However, our proposed SRGRN can accurately recover these outlines, more faithful to the ground truth.

The above comparison results are mainly due to the fact that SRGRN can enable upsampling and reconstruction modules to utilize more contextual information through relation reasoning.

Visual comparison with BD and DN degradtion model.

In Figure 6 , we show the comparison of SRGRN with other models in visual results.

For image "img_014", we use bicubic upsampling to recover these images whose HR images are blurred with a Gaussian kernel before bicubic downsampling, then we obtain SR images with a lot of noticeable blurring artifacts.

We have also observed that most methods, including RDN and SRFBN, do not clearly recover the lines around the window.

Only our SRGRN can suppress blurring artifacts and recover these clear enough lines close to the ground truth by relation reasoning.

For image "img_002", a large amount of noise corrupt the LR image and make it loss some detail.

It can be seen that when using bicubic for upsampling, the obtained image not only has a large number of blurring artifacts but also a large amount of noise.

However, we find that our SRGRN has great potential for removing noise efficiently and recover more detail.

This fully demonstrates the effectiveness and robustness of our SRGRN for BD and DN degradation models.

Visual comparison on Real-World Images.

In figure 7 , the resolution of these images is so small that there is a lot of high frequency information missing from them.

Moreover, in this case, the original HR images are not available and the degradation model is unknown either.

For image "window"(with 200 ?? 160 pixels), only our SRGRN is able to recover sharper window edges and produce clearer SR image.

For image "flower"(with 256 ?? 200 pixels), most other methods recover images whose upper left corner produces the edge of the pistil that looks unreal.

And their edges of the petals in the whole image are very blurry.

Our SRGRN can recovers sharper edges and finer details than other state-of-the-art methods.

The above analysis indicate our model perform robustly unknown degradation models.

This further reflects the superiority of relation reasoning.

@highlight

A state-of-the-art model based on global reasoning for image super-resolution