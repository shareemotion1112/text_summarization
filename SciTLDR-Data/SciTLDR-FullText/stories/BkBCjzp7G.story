We present the iterative two-pass decomposition flow to accelerate existing convolutional neural networks (CNNs).

The proposed rank selection algorithm can effectively determine the proper ranks of the target convolutional layers for the low rank approximation.

Our two-pass CP-decomposition helps prevent from the instability problem.

The iterative flow makes the decomposition of the deeper networks systematic.

The experiment results shows that VGG16 can be accelerated with a 6.2x measured speedup while the accuracy drop remains only 1.2%.

Deep learning has become of vital importance in a variety of artificial intelligence applications.

Recently, convolutional neural networks (CNNs) have been widely applied to have the breakthrough in improving the recognition accuracy for challenging computer vision tasks such as image classification, localization, object detection, and so on BID19 ; BID11 ; BID20 ; BID3 ).

However, those significant achievements using CNNs come with the cost of larger network size and higher computational complexity, which leads to an increasing difficulty for deploying to resource constrained edge devices, or even for the fast computation on the cloud servers.

This paper addresses the acceleration of the existing CNN models to cope with such burden.

There have been research works on accelerating CNNs in many aspects.

Several approaches have been proposed to simplify the convolution operations.

BID21 uses Fast Fourier Transform to accelerate the convolution which is fast for large filters.

In BID12 , Winograd's minimal filtering algorithms have been adopted to speed up small-size filters.

Pruning aims at removing unessential weights in the filters to minimize the computation.

Weights can be removed after the training BID2 ; ; BID24 ) or during the training with the sparsity constraint BID23 ; BID17 ).

Quantization reduces the precision from the 32-bit floating-point computations to the 8-bit fixed-point ones, or even the binary operations BID5 ; BID18 ).

These approaches are efficient for implementing the hardware accelerators of the faster inference.

In addition to the attempts of accelerating pre-trained models, some works focus on designing new models with more efficient computation BID6 ; ; BID4 ; BID26 ).

In general, new models tend to gain more speedup than accelerating pre-trained models.

However, constructing a new CNN from scratch might be difficult without using a large amount of computing resources.

In our approach, we aim at accelerating existing CNNs using the low rank approximation that a specific tensor can be represented with several simpler tensors.

BID7 has shown the redundancies in convolutional layers can be substituted by lower rank filters with the 4.5?? speedup in a text recognition application using a simple four layer CNN.

BID1 has presented a different low-rank decomposition and clustering approach.

They reported the 2?? speedup on the first layer of a 15-layer CNN for image classification.

They also found that the low rank approximation has the potential to improve the generalization ability of the CNN.Our work was inspired by BID13 which adopted the CP-ALS, one of the popular CP decomposition algorithms BID10 ).

However, only one layer in the network was decomposed.

They also observed the instability problem which causes fine-tuning the decomposed networks a difficult problem.

Later, BID9 successfully decomposed the whole network with the Tucker Decomposition BID10 ).

BID25 took the nonlinear unit into account to obtain the decomposed filters and reduced the accumulated error.

Their nonlinear asymmetric 3d reconstruction, combined with the technique in BID7 to further decomposing spatial dimension, achieved the 5?? speedup with the accuracy loss of 2% for VGG16.

An additional fine-tuning was applied to improve the loss to 1%.In BID0 , the iterative fine-tune has been proposed to overcome the CP instability.

Instead of the decomposition of the entire network, each iteration decomposes one layer with the fine-tuning of the whole network.

Their attempt gradually transforms the dense network into decomposed form layer by layer, which achieves the less accuracy drop.

We also adopt and improve this concept to further improve the performance, which will be discussed later in 2.1.3.

Our contributions include the following:??? The proposed two-pass decomposition can prevent from possible instability of CPdecomposition and improve the accuracy effectively.

Training and fine-tuning can be done with vanilla parameter setting (e.g., learning rate).??? Our iterative flow helps the decomposition of a deeper network in a systematic manner.

The rank selection algorithm can be utilized to determine the target ranks for CP-decomposition.

The proposed approach can be applied to different kinds of deep convolutional networks, resulting in a general technique to improve the existing models.

Our experiment shows that we can achieve the 6.2?? speedup for VGG16 BID20 ) while with the classification accuracy loss of only 1.2% on ImageNet 2012 validation set.

The size of the convolutional layers can be reduced by 85% as well.

In addition, our approach can also speed up ResNet50 BID3 ) by 1.35?? faster with the accuracy loss of 1.51%, and the model size reduction of 48%.

In this section, we introduce the low rank approximation to accelerate convolutional layers with the CANDECOMP/PARAFAC (CP) decomposition technique (Kiers FORMULA1 ; BID10 ).

The CP decomposition factorizes the tensors into a sum of series rank-one tensors.

Assume W is a third-order tensor and a r , b r , c r for r = 1, . . .

, R are vectors.

A rank R decomposition can be expressed as DISPLAYFORM0 where W ??? R H??I??J , a r ??? R H , b r ??? R I , c r ??? R J for r = 1, . . .

, R, as shown in FIG0 .In the case of CNNs, the filters are fourth-order tensors in general.

Assume W is a fourth-order tensor and a r , b r , c r , d r for r = 1, . . .

, R are vectors.

A rank R decomposition can be expressed as follows: DISPLAYFORM1 where Let X be the input features, Y be the output features, W be the filter weights and DISPLAYFORM2 DISPLAYFORM3 2 .

Assume W f and H f are both odd numbers.

A convolutional layer with C in input channels, C out output channels and filter shape of W f ?? H f , can be expressed as DISPLAYFORM4 Then the weights in Eq. 3 can be substituted with the decomposed tensors in Eq. 2, i.e., DISPLAYFORM5 Equation FORMULA5 can be rewritten as follows: DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 Y (1) and Y (4) perform the 1??1 convolutions on the input and output channels, respectively.

In addition, Y (2) and Y (3) perform the depth-wise convolutions along the filters' width and height direction, respectively, as shown in FIG2 .

Therefore, the four layers can be applied to substitute the original convolution layer.

DISPLAYFORM9

Assume the size of the input features is W in ?? H in ?? C in , where W in and H in are the width and height of the features, and C in is the number of input channels.

Let the number of output channels be C out , the filter shape be W f ??H f , and the stride be 1.

The complexity of the original convolution DISPLAYFORM0 DISPLAYFORM1 If the convolution layer is decomposed with Eq. 6-9, the complexity becomes DISPLAYFORM2 The speedup of decomposing the single convolutional layer with rank R is DISPLAYFORM3 ).While substituting a layer, the smaller the rank R for the decomposed layers, the higher the speedup.

As BID13 mentioned, training the network with CP decomposition may suffer from the instability problem that leads to the gradient explosion.

The small learning rate is used to cope with the problem.

In addition, part of the decomposed layer is fixed for training.

However, with a small learning rate, the accuracy may improve slowly for decomposing deeper networks.

Fixing decomposed layers also makes the accuracy hard to improve because only a few parameters could be fine-tuned.

In this section, we present the Two-Pass Decomposition to avoid the CP Instability.

In addition, the Iterative Two-Pass Decomposition flow can be applied to improve the accuracy loss when accelerating the deeper networks.

Our two-pass decomposition consists of four steps to compute the decomposed layer.

Step 1: The 1 st Decomposition:

Apply the CP decomposition on the original filter tensor to get the decomposed tensor.

In this work, we adopt the CP-ALS algorithm in BID10 .Step 2: Restoration: Restore the decomposed tensor back to the dense form.

Step 3: Optimization: Replace the original filter tensor with the restored filter tensor in the target convolutional layer.

Optimize the updated networks with fine-tuned filter weights.

Step 4: The 2 nd Decomposition:

Decompose the optimized filter tensor again.

The concept to restore the decomposed tensor back to the dense form is to prevent from the CP instability.

We observed that if the network is trained in the restored dense form, the training result can be more stable because of its smoother convex.

In addition, the structure of the restored dense form tends to be closer to the second low rank form, which also leads to a less decomposition error.

FIG3 compares the accuracy between our two-pass decomposition and the original CP-ALS decomposition of VGG16.

The CNN layers from Conv1 2 to Conv3 1 are evaluated, with Conv1 2 to Conv2 2 decomposed and fixed.

The first layer, Conv1 1, is not computationally intensive and not decomposed.

The ranks of Conv1 2 Conv2 1, and Conv2 2 are 24, 25, and 28, respectively.

In this experiment, different ranks of Conv3 1 are applied to observe the accuracy change.

The comparison shows that the two-pass CP decomposition retains the relatively high accuracy even with smaller ranks, i.e., the ranks of 24 and 32 in the figure.

On the contrary, with the original CP decomposition, the accuracy drops significantly when decreasing the rank.

Such result leaves our two-pass CP decomposition more room to speed up the network with lower ranks while maintaining the accuracy.

Traditionally, the filter weights of the decomposed layers are fixed during the fine-tuned phase to prevent from the gradient explosion.

Our two-pass decomposition provides the better result as compared with the original CP decomposition.

However, the accuracy will still degrade for decomposing the deeper networks, since there will be little room for fine-tuning.

For a DNN with many convolution layers, we propose the Rank Selection algorithm to effectively determine the rank for each layer.

Then our two-pass decomposition technique is applied iteratively to one group of the layers at a time.

The overall flow to decompose the whole CNN is shown in Figure 5 .For the decomposition, target rank of each layer is a hyperparameter to decide.

Our flow takes a given set of ranks as the input to perform the initial decomposition.

After the initial decomposition, the fitness of each decomposed layer can be calculated.

Based on the initial ranks and corresponding fitnesses, we propose the Rank Selection algorithm to optimize the target ranks for the decomposition, which will be discussed in Sec. 2.1.2.The second part of the flow performs the two-pass decomposition iteratively, by grouping the CNN layers.

At the end of each iteration, the whole network will be fine-tuned with the decomposed layers fixed.

The accuracy of the fine-tuning can be further improved with a few more epochs until the loss is converged.

The details will be discussed in Sec. 2.1.3.

Selected Rank

Iterative Two-Pass Decomposition

Figure 5: The proposed iterative two-pass decomposition flow.

Given any initial ranks for the CNN and a target speedup, the proposed Rank Selection algorithm determines the optimized rank configuration.

Let N l denote the operation complexity of the l th layer in the CNN, and N l be its complexity of the decomposed form with the rank of R l .

Therefore, DISPLAYFORM0 The overall theoretical speedup S of the CNN can be defined as DISPLAYFORM1 CP-ALS decomposition in Nickel (2016) is adopted for our approach.

According to BID10 BID16 , the fitness (as the decomposition quality) of a convolutional layer can be defined as DISPLAYFORM2 where X l is the original tensor of the l th layer and X l is the approximated tensor.

The product of the fitness F l and the operation complexity N l is used to estimate the profit to improve a specific layer l. If a layer has higher fitness and complexity, we tend to reduce its rank for the speedup.

Otherwise, its ranks can be increased for the accuracy.

Algorithm 1 shows the proposed Rank Selection to help determine the optimized rank configuration.

Note that the CP decomposition with a large rank is time-consuming.

So a linear approximation is used to predict the fitness in this stage.

Let T be the layer with the largest F T ?? N T among all layers for the speedup.

5: DISPLAYFORM0 Linear approximation of the new fitness.6: DISPLAYFORM1 Let T be the layer with the smallest F T ?? N T among all layers for the accuracy.9: DISPLAYFORM2 Linear approximation of the new fitness.10: DISPLAYFORM3 end if

Update N l , S 13: end while

To prevent from the accuracy degradation, our two-pass decomposition approach can be performed iteratively, based on the concept from BID0 .

FIG4 illustrates the three iterations applied to VGG16 model.

Each iteration consists of five steps.

The first four steps perform the proposed two-pass decomposition of the target group.

E.g., the group of Conv1 2, Conv2 1, Conv2 2 is decomposed and restored back to the dense form in the first iteration (see FIG4 ).

Then the whole network is fine-tuned to optimize the weights.

Afterward, the three target layers are decomposed again.

In the fifth step, the whole network is fine-tuned again with these decomposed layers fixed.

This additional step can improve the degraded accuracy further.

The second iteration deals with the group of Conv3 1, Conv3 2, and Conv3 3.

Note that the previously decomposed layers remain fixed in the Optimization and Fine-tuning steps.

The process continues until all the target layers are decomposed.

The VGG16 model modified from machrisaa (2016) is applied to verify our iterative two-pass decomposition flow.

ImageNet 2012 training set is used for training and the validation set is used for the top-5 single view accuracy measurement.

The accuracy of the pre-trained VGG16 model is 89.9% by using AdaDelta Optimizer with default setting in Tensorflow.

The speedup is measured on a single-thread 2.7GHz Intel Core i5 CPU.All the convolutional layers except the first one, Conv1 1, are decomposed.

We adopt the ranks proposed in BID25 as the Baseline rank configuration (see TAB1 ).

With the Baseline configuration as the initial ranks, The Rank-Selection configuration is obtained with the theoretical speedup of 8.4, which achieves the 6.2?? measured speedup of on the single thread CPU.

Layer Name Baseline RankSelection Layer Name Baseline RankSelection TAB1 To compare the different decomposition sequences, two grouping schemes are evaluated, as shown in Table 2 .

The In-Order scheme groups the convolutional layers based on their connection order.

On the other hand, The Fitness-Based scheme groups the layers based on the sorting order of the fitness, from the smallest to the largest.

In this experiment, each scheme has three groups of layers.

TAB2 shows the accuracy at the optimization step and final fine-tuning step of each iteration for the Baseline configuration and Rank-Selection configuration, respectively.

Two different grouping schemes are also applied for each rank configuration.

For the quick evaluation, one epoch is applied for each fine-tuning.

We can observe that the Rank-Selection configuration improves the accuracy significantly.

In addition, the In-Order grouping scheme is much better than the Fitness-Based scheme for the Baseline rank configuration.

However, their difference is vague when applying the Rank-Selection configuration.

We will discuss it further in Sec. 3.3 Table 2 : Two grouping schemes for the iterative two-pass decomposition.

To compare with previous works BID7 BID25 ), our experiment is extended with additional epochs to fine-tune until the accuracy improvement is smaller than 0.1%.

TAB3 shows the accuracy drop for each iteration.

In this case, the Fitness-Based configuration is better than the In-Order one (see the discussion in Sec. 3.3).

As a result, our method achieves the highest measured speedup with the lowest accuracy drop among the works utilizing decomposition techniques, as shown in TAB4 .

Note that the asymmetric 3d approach in BID25 results in a 2.0% accuracy drop.

An additional fine-tuning is explicitly applied to meet the final 1.0% accuracy drop with 5 more epochs and the learning rate of 10 ???5 .

Our method only relies on vanilla fine-tuning to reach the low accuracy drop.

The learning rate we used is 10 ???3 .The CP decomposition can also compress the filter size effectively.

With the Rank-Selection configuration, our approach reduces 85% of the convolutional layers (from 57MB to 8.3MB).

However, due to the large size of fully connected layers (about 470MB) in VGG16, the overall reduction of the entire networks is 9% (from 528MB to 480MB) after the two-pass decomposition flow.

The deeper CNN, ResNet50, is also evaluated.

The accuracy of the pre-trained model which is modified from ethereon FORMULA0 is 92.02% with the same training environment of the previous experiment, except for the learning rate of 10 ???5 .In this experiment, the 1??1 convolutional layers are excluded for the decomposition.

The initial rank is set to 50 for all target layers.

The result of our iterative decomposition is shown in TAB5 .

Each iteration is trained with one epoch.

After the iterative two-pass decomposition, the measured speedup is 1.35?? and the model size reduces by 48% (98MB???51MB).

BID0 suggests that freezing layers make the fine-tuning greedy, which might cause the optimization to be stuck in a local minimal.

However, our approach works efficiently by iteratively decomposing a group of layers with previously layers frozen.

The attempt also makes the optimization converge faster and prevents from the gradient explosion.

Our first evaluation in TAB2 shows that the In-Order grouping scheme performs better in a typical training and fine-tuning, especially for the Baseline rank configuration.

The In-Order scheme may be suitable for the decomposition with smaller rank configuration.

Because the Fitness-Based approach decomposes those layers of smaller ranks first.

But it also makes the accuracy drop quickly in the former layers of small ranks.

For the Rank-Selection configuration, the Fitness-Based scheme does not lead to a significant accuracy drop due to the relatively higher ranks and fitnesses.

The layers of lower fitness are decomposed first.

Their accuracy drop can be compensated by fine-tuning the latter layers with higher fitness.

The result in TAB3 also shows that a better accuracy drop may be achieved with a proper convergence constraint.

The iterative two-pass decomposition flow has been presented to accelerate existing deep CNNs.

Our two-pass decomposition effectively prevents from the CP instability.

The Rank Selection algorithm provides the fine-grained rank configuration to achieve the target speedup while maintaining the accuracy.

The experiment results show that VGG16 can be accelerated by 6.2 times with the accuracy drop of only 1.20% and the size reduction of 85%.

In addition, ResNet50 can be speeded up by 1.35 times with the accuracy drop of 1.51% and the size reduction of 48%.The future works include the improvement of the grouping scheme and the decomposing order, and a smarter rank selection with non-linear fitness estimation.

In addition, accelerating 1??1 convolutional layers will also be considered for the further improvement on the advanced CNNs.

APPENDIX A FIG6 shows the initial fitnesses of the baseline rank configuration and fitness-based configuration.

Note that the layer order is sorted by the fitnesses of the baseline rank configuration.

Because there is only a slight difference between the sorting of the two configuration, the fitness-based grouping scheme is based on the sorting of the baseline rank configuration.

TAB1

@highlight

We present the iterative two-pass CP decomposition flow to effectively accelerate existing convolutional neural networks (CNNs).

@highlight

The paper proposes a novel workflow for acceleration and compression of CNNs and also proposes a way to determine the target rank of each layer given the target overall acceleration. 

@highlight

This paper addresses the problem of learning a low rank tensor filter operation for filtering layers in deep neural networks (DNNs). 