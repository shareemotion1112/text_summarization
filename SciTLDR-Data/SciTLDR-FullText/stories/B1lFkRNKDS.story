As the basic building block of Convolutional Neural Networks (CNNs), the convolutional layer is designed to extract local patterns and lacks the ability to model global context in its nature.

Many efforts have been recently made to complement CNNs with the global modeling ability, especially by a family of works on global feature interaction.

In these works, the global context information is incorporated into local features before they are fed into convolutional layers.

However, research on neuroscience reveals that, besides influences changing the inputs to our neurons, the neurons' ability of modifying their functions dynamically according to context is essential for perceptual tasks, which has been overlooked in most of CNNs.

Motivated by this, we propose one novel Context-Gated Convolution (CGC) to explicitly modify the weights of convolutional layers adaptively under the guidance of global context.

As such, being aware of the global context, the modulated convolution kernel of our proposed CGC can better extract representative local patterns and compose discriminative features.

Moreover, our proposed CGC is lightweight, amenable to modern CNN architectures, and consistently improves the performance of CNNs according to extensive experiments on image classification, action recognition, and machine translation.

Convolutional Neural Networks (CNNs) have achieved significant successes in various tasks, e.g., image classification (He et al., 2016a; Huang et al., 2017) , object detection (Girshick et al., 2014; , image translation , action recognition (Carreira & Zisserman, 2017) , sentence/text classification Kim, 2014) , machine translation (Gehring et al., 2017) , etc.

However, the sliding window mechanism of convolution makes it only capable of capturing local patterns, limiting its ability of utilizing global context.

Taking the 2D convolution on the image as one example, as Figure 1a shows, the standard convolution only operates on the local image patch and thereby composes local features.

According to the recent research on neuroscience (Gilbert & Li, 2013) , neurons' awareness of global context is important for us to better interpret visual scenes, stably perceive objects and effectively process complex perceptual tasks.

Many methods (Vaswani et al., 2017; Wang et al., 2017a; b; Hu et al., 2018; Chen et al., 2019; Cao et al., 2019; Bello et al., 2019) have been proposed recently to introduce global context modeling modules into CNN architectures.

In this paper, such a family of works is named as global feature interaction methods.

As Figure 1b shows, these methods modulate intermediate feature maps by incorporating the global context with the local feature representation.

For example, in Non-local modules (Wang et al., 2017b) , local features are reassembled according to global correspondence, which augments CNNs with the global context modeling ability.

As was discussed by Gilbert & Li (2013) , the global context influences neurons processing information in two distinct ways: "various forms of attention, including spatial, object oriented and feature oriented attention" and "rather than having a fixed functional role, neurons are adaptive processors, changing their function according to behavioral context".

The previous work (Vaswani et al., 2017; Wang et al., 2017a; b; Hu et al., 2018; Chen et al., 2019; Cao et al., 2019; Bello et al., 2019) of global feature interaction methods, shown in Figure 1b , only modifies intermediate features, namely, inputs of neurons, which corresponds to the first way.

However, to the best of our knowledge, the other efficient and intuitive way, i.e., explicitly modulating the convolution kernels according to context, has not been exploited yet.

Motivated by this, we will model convolutional layers as "adaptive processors" and explore how to leverage global context to guide the composition of local features in convolution operations.

In this paper, we propose Context-Gated Convolution (CGC), as Figure 1c shows, a new perspective of complementing CNNs with the awareness of the global context.

Specifically, our proposed CGC learns a series of mappings to generate gates from the global context feature maps to modulate convolution kernels accordingly.

With the modulated kernels, standard convolution is performed on input feature maps, which is enabled to dynamically capture representative local patterns and compose local features of interest under the guidance of global context.

Our contributions are in three-fold.

??? To the best of our knowledge, we make the first attempt of introducing the contextawareness to convolutional layers by modulating the weights of them according to the global context.

??? We propose a novel lightweight CGC to effectively generate gates for convolution kernels to modify the weights with the guidance of global context.

Our CGC consists of a Context Encoding Module that encodes context information into latent representations, a Channel Interacting Module that projects them into the space of output dimension, and a Gate Decoding Module that decodes the latent representations to produce the gate.

??? Our Context-Gated Convolution can better capture local patterns and compose discriminative features, and consistently improve the performance of standard convolution with a negligible complexity increment in various tasks including image classification, action recognition, and machine translation.

2 CONTEXT-GATED CONVOLUTION 2.1 PRELIMINARY Without loss of generality, we consider one sample of 2D case.

The input to a convolutional layer is a feature map X ??? R c??h??w , where c is the number of channels, and h, w are respectively the height and width of the feature map.

In each convolution operation, a local patch of size c ?? k 1 ?? k 2 is collected by the sliding window to multiply with the kernel W ??? R o??c??k1??k2 of this convolutional layer, where o is the number of output channels, and k 1 , k 2 are respectively the height and width of the kernel.

Therefore, only local information within each patch is extracted in one convolution operation.

Although in the training process, the convolution kernels are learned from all the patches from all the images in the training set, the kernels are not adaptive to the current context during inference time.

In order to handle the aforementioned drawback of standard convolution, we propose to incorporate the global context information during the convolution process.

Different from the existing approaches that modify the input features according to the context, e.g., a global correspondence of from C and O to construct the gate G. , denote convolution and element-wise multiplication operations, respectively.

??? is shown in Equation 1.

feature representations, we attempt to directly modulate the convolution kernel under the guidance of the global context information.

One simple and straightforward way of modulating the convolution kernel W with global context information is to directly generate a gate G ??? R o??c??k1??k2 of the same size with W according to global context.

Assuming that we generate the gate from a context vector v ??? R l using a linear layer without the bias term, the number of parameters is l ?? o ?? c ?? k 1 ?? k 2 , which is extremely catastrophic when we modulate the convolution kernel of every convolutional layer.

For modern CNNs, o and c can be easily larger than 100 or even 1,000, which makes o ?? c the dominant term in the complexity.

Inspired by previous works on convolution kernel decomposition (Howard et al., 2017; Chollet, 2017) , we propose to decompose the gate G into two tensors G

(1) ??? R c??k1??k2 and G (2) ??? R o??k1??k2 , so that the complexity of o ?? c can thereby significantly break down.

However, directly generating these two tensors is still not acceptable.

Supposing that we generate them with two linear layers, the number of parameters is l ?? (o + c) ?? k 1 ?? k 2 , which is at the same scale with the number of parameters of the convolution kernel itself.

The bottleneck now is jointly modeling channel-wise and spatial interactions, namely l and (o + c) ?? k 1 ?? k 2 , considering that v ??? R l is encoded from the input feature map X ??? R c??h??w .

Inspired by depth-wise separable convolutions (Howard et al., 2017; Chollet, 2017) , we propose to model the spatial interaction and the channel-wise interaction separately to further reduce complexity.

In this paper, we propose one novel Context-Gated Convolution (CGC) to incorporate the global context information during the convolution process.

Specifically, our proposed CGC consists of three modules: the Context Encoding Module, the Channel Interacting Module, and the Gate Decoding Module.

As Figure 2 shows, the Context Encoding Module encodes global context information in each channel into a latent representation C via spatial interaction; the Channel Interacting Module projects the latent representation to the space of output dimension o via channel-wise interaction; the Gate Decoding Module produces G

(1) and G (2) from the latent representation C and the projected representation O to construct the gate G via spatial interaction.

The details of them are described in the following.

To extract contextual information, we first use a pooling layer to reduce the spatial resolution to h ?? w and then feed the resized feature map to the Context Encoding Module.

It encodes information from all the spatial positions for each channel, and extracts a latent representation of the global context.

We use a linear layer with weight E ??? R h ??w ??d to project the resized feature map in each channel to a latent vector of size d. Inspired by the bottleneck structure from (He et al., 2016a; Hu et al., 2018; Wang et al., 2017b; Vaswani et al., 2017) , we set d = k1??k2 2 to extract informative context, when not specified.

The weight E is shared across different channels.

A normalization layer and an activation function come after the linear layer.

There are c channels, so the output of the Context Encoding Module is C ??? R c??d .

Channel Interacting Module.

It projects the feature representations C ??? R c??d to the space of output dimension o. Inspired by (Ha et al., 2016) , we use a grouped linear layer I ??? R Gate Decoding Module.

It takes both C and O as inputs, and decodes the latent representations to the spatial size of convolution kernels.

We use two linear layers whose weights D c ??? R d??k1??k2 and D o ??? R d??k1??k2 are respectively shared across different channels in C and O. Then each element in the gate is produced by

where ??(??) denotes the sigmoid function.

Now we have G with the same size of the convolution kernel W, which is generated from the global context by our lightweight modules.

Then we can modulate the weight of a convolutional layer by element-wise multiplication to incorporate rich context information:

With the modulated kernel, a standard convolution process is performed on the input feature maps, where the context information can help the kernel capture more representative patterns and also compose features of interest.

Complexity.

The computational complexity of our three modules is

, where h , w can be set independent of h, w. It is

Except the linear time of pooling, the complexity of these three modules is independent of the input's spatial size.

The total number of parameters is

.

Therefore we can easily replace the standard convolution by our proposed CGC with a very limited computation and parameter increment, therefore enabling neurons to be adaptive to global context.

We are aware of previous works on dynamically modifying the convolution operation (Dai et al., 2017; Wu et al., 2019; Jia et al., 2016; Jo et al., 2018; Mildenhall et al., 2018) .

However, two key factors distinguish our approach from those works: whether the information guiding convolution is collected globally and how it changes the parameters of convolution.

Dai et al. (2017) proposed to adaptively set the offset of each element in a convolution kernel, and Wu et al. (2019) proposed to dynamically generate the weights of convolution kernels.

However, in their formulations, the dynamic mechanism for modifying convolution kernels only takes local patches or segments as inputs, so it is only adaptive to local inputs, which limits their ability of leveraging rich information in global context.

According to experiments in Section 3.4, our proposed CGC significantly outperforms Dynamic Convolution (Wu et al., 2019) with the help of global context awareness.

Another family of works on dynamic filters (Jia et al., 2016; Jo et al., 2018; Mildenhall et al., 2018) generates weights of convolution kernels using features extracted from input images by another CNN feature extractor.

The expensive feature extraction process makes it more suitable for generating a few filters, e.g., in the case of low-level image processing.

It is impractical to generate weights for all the layers in a deep CNN model in this manner.

However, our CGC takes input feature maps of a convolutional layer and makes it possible to dynamically modulate the weight of each convolutional layer, which systematically improves CNNs' global context modeling ability.

In this section, we demonstrate the effectiveness of our proposed CGC in incorporating 1D, 2D, and 3D context information in 1D, 2D, and (2+1)D convolutions.

We conduct extensive experiments on image classification, action recognition, and machine translation, and observe that our CGC consistently improves the performance of modern CNNs with negligible parameter increment on four benchmark datasets: ImageNet (Russakovsky et al., 2015) , CIFAR-10 (Krizhevsky et al., 2009), Something-Something (v1) (Goyal et al., 2017) , and IWSLT'14 De-En (Cettolo et al.) .

All of the experiments are based on PyTorch (Paszke et al., 2017) .

All the linear layers are without bias terms.

We follow common practice to use Batch Normalization (Ioffe & Szegedy, 2015) for computer vision tasks, or Layer Normalization (Ba et al., 2016) for natural language processing tasks, and ReLU (Nair & Hinton, 2010) as the activation function.

Note that we learn different sets of scaling and shifting factors for C that is fed to the identity connection and for C that is fed to the Channel Interacting Module.

We use average pooling with h = k 1 and w = k 2 , when not specified.

Note that we only replace convolution kernels with a spatial size larger than 1.

For those Point-wise convolutions, we take them as linear layers and do not modulate them.

To reduce the size of I, we fix c/g = 16 when not specified.

We initialize all these layers as what did for computer vision tasks and as what Glorot & Bengio (2010) did for natural language processing tasks. (He et al., 2016a) on ImageNet, we train models on ImageNet 2012 training set, which contains about 1.28 million images from 1,000 categories, and report the results on its validation set, which contains 50,000 images.

We replace all the 3 ?? 3 convolutions in ResNet-50 (He et al., 2016a) with our CGC and train the network from scratch.

Note that for the first convolutional layer, we use I ??? R 3??64 for the Channel Interacting Module.

We follow common practice (He et al., 2016a) and apply minimum training tricks to isolate the contribution of our CGC.

CIFAR-10 contains 50K training images and 10K testing images in 10 classes.

We follow common practice (He et al., 2016b) to train and evaluate the models.

We take ResNet-110 (He et al., 2016b )(with plain blocks) as the baseline model and test other possibilities of generating the gate G. All the compared methods are trained based on the same training protocol 1 .

The details are provided in appendix.

For evaluation, we report Top-1 and Top-5 accuracy of a single crop with the size 224 ?? 224 for ImageNet and 32 ?? 32 for CIFAR-10, respectively.

Performance Results.

As Table 1 shows, our CGC significantly improves the performance of baseline models on both ImageNet and CIFAR-10.

On ImageNet, our CGC improves the Top-1 accuracy of ResNet-50 by 1.12% with only 0.03M more parameters and 6M more FLOPs, which verifies our CGC's effectiveness of incorporating global context and its efficiency.

We also find that GC-ResNet-50 is hard to train from scratch unless using the fine-tuning protocol reported by Cao et al. (2019) , which indicates that modifying features may be misleading in the early training process.

Although our CGC introduces a few new parameters, our model converges faster and more stably compared to vanilla ResNet-50, as is shown in Figure 3 .

We suppose that this is because the adaptiveness to global context improves the model's generalization ability and the gating mechanism reduces the norm of gradients back-propagated to the convolution kernels, which leads to a smaller Lipschitz constant and thus better training stability (Santurkar et al., 2018; Qiao et al., 2019) .

Ablation Study.

In order to demonstrate the effectiveness of our module design, ablation studies are conducted on CIFAR-10, as illustrated in Table 2 .

Specifically, we ablate many variants of our CGC and find our default setting a good trade-off between parameter increment and performance gain.

The experiments on the combination of G

(1) and G (2) show that our decomposition approach in Equation 1 is a better way to construct the gate.

For channel interacting, we find that using a full linear model with g = 1 achieves better performance with more parameters, as is expected.

We try removing the bottleneck structure and set d = k 1 ?? k 2 , and the performance drops, which validates the necessity of the bottleneck structure.

Shared Norm indicates learning the same set of scaling and shifting factors for C in the following two branches.

Two Es indicates that we learn another E to encode C only for the Channel Interacting Module.

We also try sharing D for generating G (1) and G (2) , using larger resized feature maps and using max pooling instead of average pooling.

All the results support our default setting.

We also test different numbers of layers to replace standard convolutions with our CGC.

The result indicates that the more, the better.

Baseline Methods.

For the action recognition task, we adapt three baselines to evaluate the effectiveness of our CGC: TSN (Wang et al., 2016) , P3D-A (Qiu et al., 2017) (details are in appendix), and TSM .

Because our CGC's effectiveness of introducing 2D spatial context to CNNs has been verified in image classification, in this part, we focus on its ability of incorporating 1D temporal context and 3D spatial-temporal context.

For the 1D case, we apply our CGC to temporal convolutions in every P3D-A block.

For the 3D case, we apply our CGC to spatial convolutions in P3D-A or 2D convolutions in TSN or TSM; the pooling layer produces c ?? k ?? k ?? k cubes, the Context Encoding Module encodes k ?? k ?? k feature maps into a vector of length k 3 /2 and the Gate Decoding Module generates o ?? c ?? t ?? k ?? k gates.

Note that for the first convolutional layer, we use I ??? R 3??64 for the Channel Interacting Module.

Experiment Setting.

The Something-Something (v1) dataset has a training split of 86,017 videos and a validation split of 11,522 videos, with 174 categories.

We follow (Qiao et al., 2019) to train on the training set and report evaluation results on the validation set.

We follow to process videos and augment data.

Since we only use ImageNet for pretraining, we adapt the code base of TSM but the training setting from Qiao et al. (2019) .

We train TSN-and TSM-based models for 45 epochs (50 for P3D-A), start from a learning rate of 0.025 (0.01 for P3D-A), and decrease it by 0.1 at 26 and 36 epochs (30, 40, 45 for P3D-A).

For TSN-and TSM-based models, the batch size is 64 for 8-frame models and 32 for 16-frame models, and the dropout rate is set to 0.5.

P3D-A takes 32 continuously sampled frames as input and the batch size is 64, and the dropout ratio is 0.8.

We use the evaluation setting of for TSN-and TSM-based models and the evaluation settings of Wang et al. (2017b) for P3D-A. All the models are trained with 8-GPU machines.

Performance Comparisons.

As Table 3 shows, our CGC significantly improves the performance of baseline CNN models, compared to Non-local (Wang et al., 2017b) .

As aforementioned, Non- (Vaswani et al., 2017) 39.47M 34.41 DynamicConv (Wu et al., 2019) 38.69M 35.16 LightConv (Wu et al., 2019) 38.14M 34.84 LightConv + Dynamic Encoder (Wu et al., 2019) local modules modify the input feature maps of convolutional layers by reassembling local features according to the global correspondence.

We apply Non-local blocks in the most effective way as is reported by Wang et al. (2017b) .

However, we observe that its performance gain is not consistent when training the model from scratch, which again indicates that modifying features according to the global correspondence may be misleading in the early training process.

When applied to TSM, it even degrades the performance.

Our proposed CGC consistently improves the performance of all the baseline models.

When applied to TSM, our CGC yields the state-of-the-art performance, when without Kinetics (Carreira & Zisserman, 2017) pretraining, with only RGB modality and with negligible parameter increment.

Baseline Methods.

The LightConv proposed by Wu et al. (2019) achieves better performances with a lightweight convolutional model, compared to Transformer (Vaswani et al., 2017) .

We take it as the baseline model and augment their Lightweight Convolution with our CGC.

Note that the Lightweight Convolution is a grouped convolution L ??? R H??k with weight sharing, so we remove the Channel Interacting Module since we do not need it to project latent representations.

We resize the input sequence S ??? R c??L to R H??3k with average pooling.

For those sequences shorter than 3k, we pad them with zeros.

Since the decoder decodes translated words one by one at the inference time, it is unclear how to define global context for it.

Therefore, we only replace the convolutions in the encoder.

Experiment Setting.

We follow Wu et al. (2019) to train all the compared models with 160K sentence pairs and 10K joint BPE vocabulary.

We use the training protocol of DynamicConv (Wu et al., 2019) provided in Ott et al. (2019) .

The widely-used BLEU-4 (Papineni et al., 2002) is reported for evaluation of all the models.

We find that it is necessary to set beam width to 6 to reproduce the results of DynamicConv reported in (Wu et al., 2019) , and we fix it to be 6 for all the models.

Performance Comparisons.

As Table 4 shows, replacing Lightweight Convolutions in the encoder of LightConv with our CGC significantly outperforms LightConv and LightConv + Dynamic Encoder by 0.37 and 0.18 BLEU, and yields the state-of-the-art performance.

As is discussed previously, Dynamic Convolution leverages a linear layer to generate the convolution kernel according to the input segment, which lacks the awareness of global context.

This flaw may lead to sub-optimal encoding of the source sentence and thus the unsatisfied decoded sentence.

However, our CGC incorporates global context of the source sentence and helps significantly improve the quality of the translated sentence.

Moreover, our CGC is much more efficient than Dynamic Convolution because of our module design.

Our CGC only needs 0.01M extra parameters, but Dynamic Convolution needs 30?? more.

There has been much effort in augmenting CNNs with context information.

They can be roughly categorized into three types: first, adding backward connections in CNNs (Stollenga et al., 2014; Zamir et al., 2017; Yang et al., 2018) to model the top-down influence (Gilbert & Li, 2013) like humans' visual processing system; second, modifying intermediate feature representations in CNNs according to attention mechanism (Vaswani et al., 2017; Wang et al., 2017b) ; third, dynamically generating the parameters of convolution according to local or global information (Jia et al., 2016; Noh et al., 2016; Dai et al., 2017; Wu et al., 2019) .

For the first category of works, it is still unclear how the feedback mechanism can be effectively and efficiently modeled in CNNs.

For example, Yang et al. (2018) proposed an Alternately Updated Clique to introduce feedback mechanisms into CNNs.

However, compared to standard CNNs, the complex updating strategy increases the difficulty for training them as well as the latency at the inference time.

The second category of works is the global feature interaction methods.

They (Vaswani et al., 2017; Wang et al., 2017a; b; Hu et al., 2018; Chen et al., 2019; Cao et al., 2019; Bello et al., 2019) were proposed recently to modify local features according to global context information, usually by a global correspondence, i.e. the self-attention mechanism.

There are also works on reducing the complexity of self-attention mechanism (Parmar et al., 2018; Child et al., 2019) .

However, this family of works only considers changing the input feature maps.

The third type of works is more related to our work.

As is discussed before, our approach is distinct from them in two key factors.

In this paper, motivated by neuroscience research on neurons as "adaptive processors", we proposed Context-Gated Convolution (CGC) to incorporate global context information into CNNs.

Different from previous works which usually modifies input feature maps, our CGC directly modulates convolution kernels under the guidance of global context information.

We proposed three modules to efficiently generate a gate to modify the kernel.

As such, our CGC is able to extract representative local patterns according to global context.

The extensive experiment results show consistent performance improvements on various tasks.

There are still a lot of future works that can be done.

For example, ew could design task-specific gating modules to fully uncover the potential of the proposed CGC.

Mohammadreza Zolfaghari, Kamaljeet Singh, and Thomas Brox.

Eco: Efficient convolutional network for online video understanding.

For ImageNet, we use 224 ?? 224 random resized cropping and random horizontal flipping for data augmentation.

Then we standardize the data with mean and variance per channel.

We use a standard cross-entropy loss to train all the networks with a batch size of 256 on 8 GPUs by SGD with a weight decay of 0.0001 and a momentum of 0.9 for 100 epochs.

We start from a learning rate of 0.1 and decrease it by a factor of 10 every 30 epochs.

For CIFAR-10, we use 32 ?? 32 random cropping with a padding of 4 and random horizontal flipping.

We use a batch size of 128 and train on 1 GPU.

We decrease the learning rate at the 81st and 122nd epochs, and ends training after 164 epochs.

Based on ResNet-50, we add a temporal convolution with k = 5, stride = 2 after the first convolutional layer.

For convolutional layers in residual blocks, we follow Wang et al. (2017b) to add 3 ?? 1 ?? 1 convolution (stride is 1) after every two 1 ?? 3 ?? 3 convolutions.

We only inflate the max pooling layer after the first convolutional layer with a temporal kernel size of 3 and a stride of 2 without adding any other temporal pooling layers.

Note that all the aforementioned convolutional layers come with a Batch Normalization layer and a ReLU activation function.

To understand how CGC helps the model capture more informative features under the guidance of context information, we visualize the feature maps of ResNet-50 and our CGC-ResNet-50 by Grad-CAM++ (Chattopadhay et al., 2018) .

As Figure A. 3 shows, overall, the feature maps (After the CGC) produced by our CGC-ResNet-50 cover more informative regions, e.g., more instances or more parts of the ground-truth object, than vanilla ResNet-50.

Specifically, we visualize the feature maps before the last CGC in the model, the context information used by the CGC, and the resulting feature maps after the CGC.

As is clearly shown in Figure A .3, the proposed CGC extracts the context information from representative regions of the target object and successfully refine the feature maps with comprehensive understanding of the whole image and the target object.

For example, in Gold Fish 1, the head of the fishes are partially visible.

Vanilla ResNet-50 mistakes this image as Sea Slug, because it only pays attention to the tails of the fishes, which are similar to sea slugs.

However, our CGC utilizes the context of the whole image and guides the convolution with information from the entire fishes, which helps the model to classify this image correctly.

To further validate that our CGC uses context information of the target objects to guide convolution process, we calculate the average modulated kernel (in the last CGC of the model) for images of each class in the validation set.

Then we calculate inter-class L2 distances between every two average modulated kernels, i.e., class centers, and the intra-class L2 distance (mean distance to the class center) for each class.

As is shown in Figure A .4, we visualize the difference matrix between interclass distances and intra-class distances.

In more than 93.99% of the cases, the inter-class distance is larger than the corresponding intra-class distance, which indicates that there are clear clusters of these modulated kernels and the clusters are aligned very well with the classes.

This observation strongly supports that our CGC successfully extracts class-specific context information and effectively modulates the convolution kernel to extract representative features.

On the other hand, the intra-class variance of the modulated kernels supports that for different images of Figure 5 : Visualization of the difference matrix between inter-class distances and intra-class distances of the last gate in the network on ImageNet validation set. (Best viewed on a monitor when zoomed in) the same class, adjusting the kernels adaptively is beneficial for correct classification, which is consistent with the neuroscience research that motivates our CGC.

@highlight

A novel Context-Gated Convolution which incorporates global context information into CNNs by explicitly modulating convolution kernels, and thus captures more representative local patterns and extract discriminative features.

@highlight

This paper uses global context to modulate the weights of convolutional layers and help CNNs capture more discriminative features with high performance and fewer parameters than feature map modulating.