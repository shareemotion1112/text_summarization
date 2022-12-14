Large number of weights in deep neural networks make the models difficult to be deployed in low memory environments such as, mobile phones, IOT edge devices as well as "inferencing as a service" environments on the cloud.

Prior work has considered reduction in the size of the models, through compression techniques like weight pruning, filter pruning, etc.

or through low-rank decomposition of the convolution layers.



In this paper, we demonstrate the use of multiple techniques to achieve not only higher model compression but also reduce the compute resources required during inferencing.

We do filter pruning followed by low-rank decomposition using Tucker decomposition for model compression.



We show that our approach achieves upto 57\% higher model compression when compared to either Tucker Decomposition or Filter pruning alone  at similar accuracy for GoogleNet.

Also, it reduces the Flops by upto 48\% thereby making the inferencing faster.

Deep neural networks are now being used extensively for a variety of artificial intelligence applications 14 ranging from computer vision [19] to speech recognition [11] and natural language processing [5] .

In this paper, we focus particularly on convolutional neural networks (CNNs) which have become 16 ubiquitous in object recognition, image classification, and retrieval (see [17, 8, BID17 29] ).

As datasets 17 increase in size, networks also increase in complexity, number of layers and parameters in order class of approaches, pruning of the weights of a trained CNN BID20 to reduce the compute and memory requirements and to make inferencing faster.

In this paper, we focus on transfer learning.

In such a setting, filter pruning is effective in removing paper, we study these complementary techniques and show that by combining these techniques, we 37 achieve an additional 57% model compression when compared to either filter pruning or Tucker

Decomposition for popular models like GoogleNet.

Also, it reduces the Flops by upto 48% thereby 39 making the inferencing on these networks very fast.

The rest of the paper is organized as follows.

In Section 2, we describe our methodology of combining filter pruning with tensor decomposition.

Our experimental results under different settings are presented in Section 3.

Finally, we present our 42 conclusions in Section 4.

We briefly describe Tucker decomposition and filter pruning approaches from prior work followed by 45 our approach for combining these techniques.

varying the ranks of the output core tensor and factor matrices.

DISPLAYFORM0 (1)

Filter Pruning: Pruning filters from convolution layers is a standard method of compressing the

Models Used.

We demonstrate our results on state-of-the-art deep neural network GoogleNet [27] .

The base model is trained on ImageNet-1K dataset.

The datasets used for transfer learning are We show that our approach of filter pruning followed by low-rank decomposition using Tucker

@highlight

Combining orthogonal model compression techniques to get significant reduction in model size and number of flops required during inferencing.

@highlight

This paper proposes combining Tucker Decomposition with Filter pruning.