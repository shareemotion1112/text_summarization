Convolutional Neural Networks (CNNs) are composed of multiple convolution layers and show elegant performance in vision tasks.

The design of the regular convolution is based on the Receptive Field (RF) where the information within a specific region is processed.

In the view of the regular convolution's RF, the outputs of neurons in lower layers with smaller RF are bundled to create neurons in higher layers with larger RF.

As a result, the neurons in high layers are able to capture the global context even though the neurons in low layers only see the local information.

However, in lower layers of the biological brain, the information outside of the RF changes the properties of neurons.

In this work, we extend the regular convolution and propose spatially shuffled convolution (ss convolution).

In ss convolution, the regular convolution is able to use the information outside of its RF by spatial shuffling which is a simple and lightweight operation.

We perform experiments on CIFAR-10 and ImageNet-1k dataset, and show that ss convolution improves the classification performance across various CNNs.

Convolutional Neural Networks (CNNs) and their convolution layers (Fukushima, 1980; Lecun et al., 1998) are inspired by the finding in cat visual cortex (Hubel & Wiesel, 1959) and they show the strong performance in various domains such as image recognition (Krizhevsky et al., 2012; Simonyan & Zisserman, 2015; He et al., 2016) , natural language processing (Gehring et al., 2017) , and speech recognition (Abdel-Hamid et al., 2014; Zhang et al., 2016) .

A notable characteristic of the convolution layer is the Receptive Field (RF), which is the particular input region where a convolutional output is affected by.

The units (or neurons) in higher layers have larger RF by bundling the outputs of the units in lower layers with smaller RF.

Thanks to the hierarchical architectures of CNNs, the units in high layers are able to capture the global context even though the units in low layers only see the local information.

It is known that neurons in the primary visual cortex (i.e., V1 which is low layers) change the selfproperties (e.g., the RF size (Pettet & Gilbert, 1992) and the facilitation effect (Nelson & Frost, 1985) ) based on the information outside of the RF (D. Gilbert, 1992) .

The mechanism is believed to originate from (1) feedbacks from the higher-order area (Iacaruso et al., 2017) and (2) intracortical horizontal connections (D. Gilbert, 1992) .

The feedbacks from the higher-order area convey broader-contextual information than the neurons in V1, which allows the neurons in V1 to use the global context.

For instance, Gilbert & Li (2013) argued that the feedback connections work as attention.

Horizontal connections allow the distanced neurons in the layer to communicate with each other and are believed to play an important role in visual contour integration (Li & Gilbert, 2002) and object grouping (Schmidt et al., 2006) .

Though both horizontal and feedback connections are believed to be important for visual processing in the visual cortex, the regular convolution ignores the properties of these connections.

In this work, we particularly focus on algorithms to introduce the function of horizontal connections for the regular convolution in CNNs.

We propose spatially shuffled convolution (ss convolution), where the information outside of the regular convolution's RF is incorporated by spatial shuffling, which is a simple and lightweight operation.

Our ss convolution is the same operation as the regular convolution except for spatial shuffling and requires no extra learnable parameters.

The design of ss convolution is highly inspired by the function of horizontal connections.

To test the effectiveness of the information outside of the regular convolution's RF in CNNs, we perform experiments on CIFAR-10 (Krizhevsky, 2009) and ImageNet 2012 dataset (Russakovsky et al., 2015) and show that ss convolution improves the classification performance across various CNNs.

These results indicate that the information outside of the RF is useful when processing local information.

In addition, we conduct several analyses to examine why ss convolution improves the classification performance in CNNs and show that spatial shuffling allows the regular convolution to use the information outside of its RF.

There are two types of approaches to improve the Receptive Field (RF) of CNNs with the regular convolution: broadening kernel of convolution layer and modulating activation values by selfattention.

The atrous convolution (Holschneider et al., 1989; Yu & Koltun, 2016) is the convolution with the strided kernel.

The stride is not learnable and given in advance.

The atrous convolution can have larger RF compared to the regular convolution with the same computational complexity and the number of learnable parameters.

The deformable convolution (Dai et al., 2017) is the atrous convolution with learnable kernel stride that depends on inputs and spatial locations.

The stride of the deformable convolution is changed flexibly unlike the atrous convolution, however, the deformable convolution requires extra computations to calculate strides.

Both atrous and deformable convolution contribute to broadening RF, however, it is not plausible to use the pixel information at a distant location when processing local information.

Let us consider the case that the information of p pixels away is useful for processing local information at layer l.

In the simple case, it is known that the size of the RF grows with k √ n where k is the size of the convolution kernel and n is the number of layers (Luo et al., 2016) .

In this case, the size of kernel needs to be p √ n and k is around 45 when p = 100 and l = 5.

If the kernel size is 3 × 3, then the stride needs to be 21 across layers.

Such large stride causes both the atrous and the deformable convolution to have a sparse kernel and it is not suitable for processing local information.

Squeeze and Excitation module (SE module) (Hu et al., 2018 ) is proposed to modulate the activation values by using the global context which is obtained by Global Average Pooling (GAP) (Lin et al., 2014) .

SE module allows CNNs with the regular convolution to use the information outside of its RF as our ss convolution does.

In our experiments, ss convolution gives the marginal improvements on SEResNet50 (Hu et al., 2018) that is ResNet50 (He et al., 2016) with SE module.

This result makes us wonder why ss convolution improves the performance of SEResNet50, thus we conduct the analyses and find that the RF of SEResNet50 is location independent and the RF of ResNet with ss convolution is the location-dependent.

This result is reasonable since the spatial information of activation values is not conserved by GAP in SE module.

We conclude that such a difference may be the reason why ss convolution improves the classification performance on SEResNet50.

Attention Branch Networks (ABN) (Fukui et al., 2019 ) is proposed for a top-down visual explanation by using an attention mechanism.

ABN uses the output of the side branch to modulate activation values of the main branch.

The outputs of the side branch have larger RF than the one of the main branch, thus the main branch is able to modulate the activation values based on the information outside of main branch's RF.

In our experiments, ss convolution improves the performance on ABN and we assume that this is because ABN works as like feedbacks from higher-order areas, unlike ss convolution that is inspired by the function of horizontal connections.

ShuffleNet (Zhang et al., 2017 ) is designed for computation-efficient CNN architecture and the group convolution (Krizhevsky et al., 2012 ) is heavily used.

They shuffle the channel to make cross-group information flow for multiple group convolution layers.

The motivation of using shuffling between ShuffleNet and our ss convolution is different.

On the one hand, our ss convolution uses spatial shuffling to use the information from outside of the regular convolution's RF.

On the other hand, the channel shuffling in ShuffleNet does not broaden RF and not contribute to use the information outside of the RF.

In this section, we introduce spatially shuffled convolution (ss convolution).

Horizontal connections are the mechanism to use information outside of the RF.

We propose ss convolution to incorporate this mechanism into the regular convolution, which consists of two components: spatial shuffling and regular convolution.

The shuffling is based on a permutation matrix that is generated at the initialization.

The permutation matrix is fixed while training and testing.

Our ss convolution is defined as follows:

R represents the offset coordination of the kernel.

For examples, the case of the 3 × 3 kernel is

C×I×J is the input and w ∈ R Cw×Iw×Jw is the kernel weights of the regular convolution.

In Eqn.

(2), the input x is shuffled by P and then the regular convolution is applied.

Fig. 1 -(a) is the visualization of Eqn.

(2).

α ∈ [0, 1] is the hyper-parameter to control how many channels are shuffled.

If αC = 0, then ss convolution is same as the regular convolution.

At the initialization, we randomly generate the permutation matrix π ∈ {0, 1} m×m where

The generated π at the initialization is fixed for training and testing.

The result of CIFAR-10 across various α is shown in Fig. 2 .

The biggest improvement of the classification performance is obtained when α is around 0.06.

The group convolution (Krizhevsky et al., 2012) is the variants of the regular convolution.

We find that the shuffling operation of Eqn.

2 is not suitable for the group convolution.

ResNeXt (Xie et al., 2017 ) is CNN to use heavily group convolutions and Table 1 shows the test error of ResNeXt in CIFAR-10 (Krizhevsky, 2009).

As can be seen in Table 1 , the improvement of the classification performance is marginal with Eqn.

2.

Thus, we propose the spatial shuffling for the (2) 4.2 SS Conv w/ Eqn.

(3) 3.9 group convolution as follows:

Eqn.

3 represents that the shuffled parts are interleaved like the illustration in Fig. 1-(b) .

As can be seen in Table 1 , ss convolution with Eqn.

3 improves the classification performance of ResNeXt.

We use CIFAR-10 (Krizhevsky, 2009) and ImageNet-1k (Russakovsky et al., 2015) for our experiments.

CIFAR-10.

CIFAR-10 is the image classification dataset.

There are 50000 training images and 10000 validation images with 10 classes.

As data augmentation and preprocessing, translation by 4 pixels, stochastic horizontal flipping, and global contrast normalization are applied onto images with 32 × 32 pixels.

We use three types of models of ImageNet-1k.

ImageNet-1k is the large scale dataset for the image classification.

There are 1.28M training images and 50k validation images with 1000 classes.

As data augmentation and preprocessing, resizing images with the scale and aspect ratio augmentation and stochastic horizontal flipping are applied onto images.

Then, global contrast normalization is applied to randomly cropped images with 224 × 224 pixels.

In this work, we use ResNet50 (He et al., 2016) , DenseNet121 (Huang et al., 2017 ), SEResNet50 (Hu et al., 2018 and ResNet50 with ABN (Fukui et al., 2019) for ImageNet-1k experiments.

Implementation Details.

As the optimizer, we use Momentum SGD with momentum of 0.9 and weight decay of 1.0 × 10 −4 .

In CIFAR-10, we train models for 300 epochs with 64 batch size.

In ImageNet, we train models for 100 epochs with 256 batch size.

In CIFAR-10 and ImageNet, the learning rate starts from 0.1 and is divided by 10 at 150, 250 epochs and 30, 60, 90 epochs, respectively.

Table 3 .

In our experiments, we replace all regular convolutions with ss convolutions except for downsampling layers, and use single α across all layers.

We conduct grid search of α ∈ {0.02, 0.04, 0.06} and α is decided according to the classification performance on validation dataset.

We replace all regular convolutions with ss convolutions to investigate whether the information outside of the regular convolution's RF contributes to improving the generalization ability.

The results are shown in Table 2 and 3.

As can be seen in Table 2 and 3, ss convolution contributes to improve the classification performance across various CNNs except for SEResNet50 that shows marginal improvements.

The detailed analysis of the reason why ss convolution gives the marginal improvements in SEResNet50 is shonw in Sec. 5

Since α ∈ {0.02, 0.04, 0.06} is small, the small portion of the input are shuffled, thus ss convolution improves the classification performance with small amount of extra shuffling operations and without extra learnable parameters.

The inference speed is shown in Table 4 and ss convolution make the inference speed 1.15 times slower in exchange for 0.5% improvements in ImageNet-1k dataset.

The more efficient implementation 2 may decrease the gap of the inference speed between the regular convolution and ss convolution.

In this section, we demonstrate two analysis to understand why ss convolution improves the classification performance across various CNNs: the receptive field (RF) analysis and the layer ablation experiment.

Receptive Field Analysis.

We calculate the RF of SEResNet50, ResNet50 with ss convolution and the regular convolution.

The purpose of this analysis is to examine whether ss convolution contributes to use the information outside of the regular convolution's RF.

Layer Ablation Experiment.

The layer ablation experiment is conducted to know which ss convolution influences the model prediction.

In the primary visual cortex, the neurons change selfproperties based on the information outside of RF, thus we would like to investigate whether spatial shuffling in low layers contribute to predictions or not.

Our analyses are based on ImageNet-1k pre-trained model and the structure of ResNet50 (i.e., the base model for analysis) is shown in Table 5 .

In our analysis, we calculate the RF to investigate whether ss convolution uses the information outside of the regular convolution's RF.

The receptive field is obtained by optimization as follows:

x ∈ R C×I×J is input, and R ∈ R C×I×J is the RF to calculate and learnable.

σ is sigmoid function, thus 0 ≤ σ(R) ≤ 1.

φ l is the outpus of the trained model at the layer l.

We call the first term in Eqn.

4 as the local perceptual loss.

It is similar to the perceptual loss (Johnson et al., 2016) , and the difference is the dot product of M ∈ {0, 1}

C×I×J that works as masking.

M is the binary mask and M cij = 1 if 96 ≤ i, j ≤ 128, otherwise M cij = 0 in our analysis.

In other words, the values inside the blue box in Fig 3 are the part of M cij = 1.

The local perceptual loss minimizes the distance of feature on the specific region between σ(R) · x and x. The 2nd term is the penalty to evade the trivial case such as σ(R) = 1.

In layerwise and channelwise RF anlysis, we use β of 1.0 × 10 −6 and 1.0 × 10 −12 , respectively.

We use Adam optimizer (Kingma & Ba, 2015) to calculate R * .

As the hyper-parameter, lr, β 1 , and β 2 are 0.1, 0.9, 0.99, respectively.

The high lr is used since its slow convergence.

The batch size is 32 and we stop the optimization after 10000 iterations.

x is randomly selected from ImageNet-1k training images.

The data augmentation and preprocessing are applied as the same procedure in Sec. 4.1.

The red color indicates that the pixel there changes features inside blue box, and the white color represents that features are invariant.

The name of the layer is described in Table 5 .

The rest of RFs are shown in Appendix A.1.

Layerwise Receptive Field.

We calculate the RF for each model and the results are shown in Fig.  3 .

The top row is the RF of ResNet with the regular convolution, the middle row is the one with ss convolution and the bottom row is the one of SEResNet50.

The red color indicates that the value of the pixel there changes features inside the blue box, and the white color represents that features inside the blue box are invariant even if the value of the pixel there is changed.

In the top row of Fig. 3 , the RFs of ResNet50 with the regular convolution are shown.

The size of RF becomes larger as the layer becomes deeper.

This result is reasonable and obtained RFs are in the classical view.

If the RF of ResNet50 with ss convolution is beyond the one with the regular convolution, it indicates that ss convolution successfully uses the information outside of the regular convolution's RF.

In the middle and bottom row of Fig. 3 are the RF of ResNet50 with ss convolution and SEResNet50, respevtively.

The RFs covers the entire image unlike the RF with the regular convolution.

These results indicate that both SE module and ss convolution contributes to use the information outside of the regular convolution's RF.

is in the range between 0 and 1 and represents the ratio of σ(RF) that is bigger than 0.5.

As can be seen in Fig. 5-(a) , the size of RFs are consistently almost 1 across layers in ResNet50 with ss convolution and SEResNet50.

This result also shows that SE module and the ss convolution contributes to use the information outside of the regular convolution's RF.

This may be the reason why ss convolution improves the classification performance on various CNNs.

However, these results make us wonder why ss convolution improves marginally the performance of SEResNet50.

Further analysis is conducted in channelwise RF analysis.

Channelwise Receptive Field.

Since layerwise RF analysis is based on the RF of the layer, the obtained results have rough directions.

We calculate the channelwise RF for more fine-grained analysis.

Unlike layerwise RF analysis, M becomes different and we minimize the local perceptual loss on the specific channel.

The results are shown in Fig. 3 .

Fig. 4 (a) and ( These results indicate that the information outside of the regular convolution's RF is location-independent in SEResNet 50 and location-dependent in ResNet50 with ss convolution.

This is reasonable since SE module uses the global average pooling and the spatial information is not conserved.

This difference may be the reason why ss convolution marginally improves the classification performance on SEResNet50.

We conduct layer ablation study to investigate which ss convolutions contribute to the generalization ability.

The ablation is done as follows:

Eqn.

5 represents that the activation values of the shuffled parts become 0.

The result of the ablation experiment is shown in Fig. 5-(b) .

Eqn.

5 is applied to all ss convolutions in each block and the biggest drop of the classification performance happens at the ablation of conv4 4.

It indicates that it is useful to use the information outside of the regular convolution's RF between the middle and high layers.

The classification performance is degraded even if the ablation is applied to the first bottleneck (i.e., conv2 1).

This result implies that the information outside of the regular convolution's RF is useful even at low layers.

In this work, we propose spatially shuffled convolution (ss convolution) to incorporate the function of horizontal connections in the regular convolution.

The spatial shuffling is simple, lightweight, and requires no extra learnable parameters.

The experimental results demonstrate that ss convolution captures the information outside of the regular convolution's RF even in lower layers.

The results and our analyses also suggest that using distant information (i.e., non-local) is effective for the regular convolution and improves classification performance across various CNNs.

Figure 6: The receptive field of ImageNet-1k pre-trained ResNet50.

The red color indicates that the pixel there changes features inside the blue box, and the white color represents that features are invariant even if the pixel there changes the value itself.

Those images are the receptive field of all layers and the name of the layer is described in Table 5 (a) conv2 1

Figure 7: The receptive field of ImageNet-1k pre-trained ResNet50 with ss convolutions.

The red color indicates that the pixel there changes features inside the blue box, and the white color represents that features are invariant even if the pixel there changes the value itself.

Those images are the receptive field of all layers and the name of the layer is described in Table 5 (a) conv2 1

Figure 8: The receptive field of ImageNet-1k pre-trained SEResNet50.

The red color indicates that the pixel there changes features inside the blue box, and the white color represents that features are invariant even if the pixel there changes the value itself.

Those images are the receptive field of all layers and the name of the layer is described in Table 5 l e n g t h = i n t ( c h n s / c h s )

@highlight

We propose spatially shuffled convolution that the regular convolution incorporates the information from outside of its receptive field.

@highlight

Proposes SS convulation which uses information outside of its RF, showing improved results when tested on multiple CNN models.

@highlight

The authors proposed a shuffle strategy for convolution layers in convolution layers in convolutional neural networks.