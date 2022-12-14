Modern Convolutional Neural Networks (CNNs) are complex, encompassing millions of parameters.

Their deployment exerts computational, storage and energy demands, particularly on embedded platforms.

Existing approaches to prune or sparsify CNNs require retraining to maintain inference accuracy.

Such retraining is not feasible in some contexts.

In this paper, we explore the sparsification of CNNs by proposing three model-independent methods.

Our methods are applied on-the-fly and require no retraining.

We show that the state-of-the-art models' weights can be reduced by up to 73% (compression factor of 3.7x) without incurring more than 5% loss in Top-5 accuracy.

Additional fine-tuning gains only 8% in sparsity, which indicates that our fast on-the-fly methods are effective.

There has been a significant growth in the number of parameters (i.e., layer weights), and the corresponding number of multiply-accumulate operations (MACs), in state-of-the-art CNNs BID14 BID13 BID19 BID23 BID8 BID11 BID24 BID22 .

Thus, it is to no surprise that several techniques exist for "pruning" or "sparsifying" CNNs (i.e., forcing some model weights to 0) to both compress the model and to save computations during inference.

Examples of these techniques include: iterative pruning and retraining BID2 BID7 BID3 BID20 BID17 , Huffman coding BID5 , exploiting granularity BID15 BID4 , structural pruning of network connections BID25 BID16 BID0 BID18 , and Knowledge Distillation (KD) BID9 .A common theme to the aforementioned techniques is that they require a retraining of the model to fine-tune the remaining non-zero weights and maintain inference accuracy.

Such retraining, while feasible in some contexts, is not feasible in others, particularly industrial ones.

For example, for mobile platforms, a machine learning model is typically embedded within an app for the platform that the user directly downloads.

The app utilizes the vendor's platform runtime support (often in the form of a library) to load and use the model.

Thus, the platform vendor must sparsify the model at runtime, i.e., on-the-fly, within the library with no opportunity to retrain the model.

Further, the vendor rarely has access to the labelled data used to train the model.

While techniques such as Knowledge Distillation BID9 can address this lack of access, it is not possible to apply it on-the-fly.

In this paper, we develop fast retraining-free sparsification methods that can be deployed for on-thefly sparsification of CNNs in the contexts described above.

There is an inherent trade-off between sparsity and inference accuracy.

Our goal is to develop model-independent methods that result in large sparsity with little loss to inference accuracy.

We develop three model-independent sparsification methods: flat, triangular, and relative.

We implement these methods in TensorFlow and use the framework to evaluate the sparsification of several pretrained models: Inception-v3, MobileNet-v1, ResNet, VGG, and AlexNet.

Our evaluation shows that up to 81% of layer weights in some models may be forced to 0, incurring only a 5% loss in inference accuracy.

While the relative method appears to be more effective for some models, the triangular method is more effective for others.

Thus, a predictive modeling autotuning BID6 BID1 is needed to identify, at run-time, the optimal choice of method and it hyper-parameters.

Sparsity in a CNN stems from three main sources: (1) weights within convolution (Conv) and fullyconnected (FC) layers (some of these weights may be zero or may be forced to zero); (2) activations of layers, where the often-applied ReLU operation results in many zeros BID21 ; and (3) input data, which may be sparse.

In this paper, we focus on the first source of sparsity, in both Conv and fully connected layers.

This form of sparsity can be determined a priori, which alleviates the need for specialized hardware accelerators.

The input to our framework is a CNN that has L layers, numbered 1 . . .

L. The weights of each layer l are denoted by ?? l .

We sparsify these weights using a sparsification function, S, which takes as input ?? l and a threshold ?? l from a vector of thresholds T .

Each weight i of ?? l is modified by S as follows: DISPLAYFORM0 Conference on Neural Information Processing Systems (NIPS), CDNNRIA Workshop, 2018, Montr??al, Canada.

BID0 The extended version of this paper can be found at: https://arxiv.org/abs/1811.04199 where ?? l = T (l) is the threshold used for layer l. Thus, applying a single threshold ?? l forces weights within ????? l and +?? l in value to become 0.

Our use of thresholds to sparsify layers is motivated by the fact that recent CNNs' weights are distributed around the value 0.The choice of the values of the elements of the vector T defines a sparsification method.

These values impact the resulting sparsity and inference accuracy.

We define and compare three sparsification methods.

The flat method defines a constant threshold ?? for all layers, irrespective of the distribution of their corresponding weights.

The triangular method is inspired by the size variation of layers in some state-of-the-art CNNs, where the early layers have smaller number of parameters than latter layers.

Finally, the relative method defines a unique threshold for each layer that sparsifies a certain percentage of the weights in the layer.

The three methods are depicted graphically in Figure 1a .

The high level work-flow of the sparsification framework is depicted in Figure 1b .Flat Method This method defines a constant threshold ?? for all layers, irrespective of the distribution of their corresponding weights.

It is graphically depicted in the top of Figure 1a .

The weights of the layers are profiled to determine the span ?? min = min ???linL (max(?? l ) ??? min(?? l )).

This span corresponds to the layer k having the smallest range of weights within the pretrained model.

This span is used as an upper-bound value for our flat threshold ?? .

Since using ?? min as a threshold eliminates all the weights in layer k and is likely to adversely affect the accuracy of the sparsified model, we use a fraction ??, 0 ??? ?? ??? 1, of the span ?? l = ?? min ?? ??, where ?? is a parameter of the method that can be varied to achieve different degrees of model sparsity.

The triangular method is defined by two thresholds ?? min and ?? max for respectively the first convolution layer (i.e, layer 1) and the last fully connected layer (i.e., layer L).

They represent the thresholds at the tip and the base of the triangle in middle part of Figure 1a .

These thresholds are determined by the span of the weights in each of the two layers.

Thus, DISPLAYFORM0 where ?? conv is the span of the weights in the first convolution layer, defined in a similar way as for the flat method, and it represents an upper bound on ?? min .

Thus, ?? conv is a fraction that ranges between 0 and 1.

Similarly, ?? f c is the span of the weights in the last fully connected layer and it represents an upper bound on ?? max .

Thus, ?? f c is a fraction that ranges between 0 and 1.

The thresholds of the remaining layers are dictated by the position of these layers in the network.

Relative Method In particular, it uses the ?? th percentile of distribution's weight in layer l denoted by ?? l .

Thus each element of the vector T (l) = ?? l is defined as: DISPLAYFORM1 where 0 ??? ?? l ??? 1 defines the desired percentile of the ?? l of zero weights in each layers.

We evaluate our sparsification methods using TensorFlow v1.4 with CUDA runtime and driver v8.

The evaluation of Top-5 accuracy is done on an NVIDIA's GeForce GTX 1060 with a host running Ubuntu 14.04, kernel v3.19 using ImageNet BID12 .

Figures 2a, 2b, 2c , 2d, and 2e show the inference accuracy as a function of the introduced sparsity by each method.

They reflect that significant sparsity (f) Alexnet Fine-tuning Figure 2 : Sparsity-Accuracy Trade-off for Our Three Proposed Sparsification Methods can be obtained with a small reduction in inference accuracy.

With less than 5% reduction in accuracy, we gain 51% sparsity (2.04?? compression factor), 50% (2??), 62% (2.63??), 70% (3.33??), and 73% (3.7??) for the models.

This validates our approach.

Further, the figures reflect that the relative method outperforms the other two methods for the Inception-v3 BID24 , VGG BID19 and ResNet BID8 , but the triangular one outperforms the other two for MobileNet-v1 BID10 and Alexnet BID13 .

This is likely due to the structure of the models.

MobileNet-v1 and Alexnet have a gradual linear increase in the size of the convolution layers, making the triangular method more effective.

In contrast, the other models have no such increase, making the relative method more effective.

As a case-in-point, ResNet has 152 conv layers with variable sizes, which makes the triangular method less effective, as seen by the drop in accuracy in Figure 2c .An interesting observation is that for Alexnet, introducing the first 50% sparsity suffers little drop in accuracy.

This value is 35%, 30%, 41%, and 42% for the other models and it shows there exists significant redundancy within CNNs.

Han et al. BID5 observe the same with their Caffe implementation of Alexnet.

The work was mainly focused on iteratively pruning and retraining CNNs to compensate the loss of accuracy.

The authors' method with no retraining is not specified and it is unclear if it applies to other CNNs or not.

However, the authors report a gain of around 80% sparsity by pruning (i.e., without retraining) Alexnet with L2 regularization.

Our evaluation validates their result across other models using the proposed on-the-fly methods.

Fine-tuning.

We explore what can be achieved by some fine-tuning of our methods, still with no retraining, in order to gain more sparsity.

We do so to determine the effectiveness of our on-the-fly methods, since the fine-tuning is not likely feasible in out context.

We focus on the relative method and start with a baseline sparsity.

We then vary the degree of sparsity of each layer in turn around the base sparsity, attempting to maintain a no more than 5% drop in inference accuracy.

The results for only AlexNet (due to space limitations) are shown in the table in Figure 2f .

The baseline sparsity is selected as 70%.

It is possible for some layers, particularly larger ones, to have higher sparsity, while smaller/earlier layers are more sensitive to sparsification and must have lower sparsity.

Nonetheless, there is a gain of 8% in overall model sparsity.

This value is 4%, 3%, 2% , and 5% for Inception-v3, Mobilenet-v1, ResNet, and VGG, respectively.

Since this gain comes at the expense of an exploration of different sparsity ratios for the layers and thus more computations, it is not feasible in the contexts we explore.

However, the gain is not significant to render our on-the-fly methods inefficient on their own without further tuning.

In this paper, we proposed three model-independent methods to explore sparsification of CNNs without retraining.

We experimentally evaluated these methods and showed that they can result in up to 73% sparsity with less than 5% drop in inference accuracy.

However, there is no single method that works best for all models.

Further, our evaluation showed that it is possible to fine-tune the methods to further gain sparsity with no significant drop in inference accuracy.

However, such tuning of the methods cannot be employed on-the-fly.

There are two key directions for future work.

The first is to explore heuristics for selecting a sparsification method based on the CNN model and possibly fine tune the parameters of the methods using a predictive modeling.

The second is to realize the benefit of the sparsity in the model's implementation on the NNlib library, which offloads neural networks operations from TensorFlow to Qualcomm's Hexagon-DSP.

@highlight

In this paper, we develop fast retraining-free  sparsification methods that can be deployed for on-the-fly sparsification of CNNs in many industrial contexts.

@highlight

This paper proposes approaches for pruning CNNs without retraining by introducing three schemes to determine the thresholds of pruning weights.

@highlight

This paper describes a method for sparsification of CNNs without retraining.