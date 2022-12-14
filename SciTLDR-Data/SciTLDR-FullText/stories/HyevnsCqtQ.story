With the rapidly scaling up of deep neural networks (DNNs), extensive research studies on network model compression such as weight pruning have been performed for efficient deployment.

This work aims to advance the compression beyond the weights to the activations of DNNs.

We propose the Integral Pruning (IP) technique which integrates the activation pruning with the weight pruning.

Through the learning on the different importance of neuron responses and connections, the generated network, namely IPnet, balances the sparsity between activations and weights and therefore further improves execution efficiency.

The feasibility and effectiveness of IPnet are thoroughly evaluated through various network models with different activation functions and on different datasets.

With <0.5% disturbance on the testing accuracy, IPnet saves 71.1% ~ 96.35% of computation cost, compared to the original dense models with up to 5.8x and 10x reductions in activation and weight numbers, respectively.

Deep neural networks (DNNs) have demonstrated significant advantages in many real-world applications, such as image classification, object detection and speech recognition BID6 BID15 BID16 .

On the one hand, DNNs are developed for improving performance in these applications, which leads to intensive demands in data storage, communication and processing.

On the other hand, the ubiquitous intelligence promotes the deployment of DNNs in light-weight embedded systems that are equipped with only limited memory and computation resource.

To reduce the model size while ensuring the performance quality, DNN pruning is widely explored.

Redundant weight parameters are removed by zeroing-out those in small values BID4 BID13 .

Utilizing the zero-skipping technique BID5 on sparse weight parameters can further save the computation cost.

In addition, many specific DNN accelerator designs BID0 BID14 leveraged the intrinsic zero-activation pattern of the rectified linear unit (ReLU) to realize the activation sparsity.

The approach, however, cannot be directly extended to other activation functions, e.g., leaky ReLU.Although these techniques achieved tremendous success, pruning only the weights or activations cannot lead to the best inference speed, which is a crucial metric in DNN deployment, for the following reasons.

First, the existing weight pruning methods mainly focus on the model size reduction.

However, the most essential challenge of speeding up DNNs is to minimize the computation cost, such as the intensive multiple-and-accumulate operations (MACs).

Particularly, the convolution (conv) layers account for most of the computation cost and dominate the inference time in DNNs BID13 .

Because weights are shared in convolution, the execution speed of conv layers is usually bounded by computation instead of memory accesses BID7 BID21 .

Second, the activation in DNNs is not strictly limited with ReLU.

The intrinsic zeroactivation patterns do not exist in non-ReLU activation functions, such as leaky ReLU and sigmoid.

Third, the weights and activations of a network together determine the network performance.

Our experiment shows that the zero-activation percentage obtained by ReLU decreases after applying the weight pruning BID5 .

Such a deterioration in activation sparsity could potentially eliminate the advantage of the aforementioned accelerator designs.

In this work, we propose the integral pruning (IP) technique to minimize the computation cost of DNNs by pruning both weights and activations.

As the pruning processes for weights and activations are correlated, IP learns dynamic activation masks by attaching activation pruning to weight pruning after static weight masks are well trained.

Through the learning on the different importance of neuron responses and connections, the generated network, namely IPnet, balances the sparsity between activations and weights and therefore further improves execution efficiency.

Moreover, our method not only stretches the intrinsic activation sparsity of ReLU, but also targets as a general approach for other activation functions, such as leaky ReLU.

Our experiments on various network models with different activation functions and on different datasets show substantial reduction in MACs by the proposed IPnet.

Compared to the original dense models, IPnet can obtain up to 5.8?? activation compression rate, 10?? weight compression rate and eliminate 71.1% ??? 96.35% of MACs.

Compared to state-of-the-art weight pruning technique BID4 , IPnet can further reduce the computation cost 1.2?? ??? 2.7??.

Weight Pruning: The weight pruning emerges as an effective compression technique in reducing the model size and computation cost of neural networks.

A common approach of pruning the redundant weights in DNN training is to include an extra regularization term (e.g., the 1 -normalization) in the loss function BID9 BID13 to constrain the weight distribution.

Then the weights below a heuristic threshold will be pruned.

Afterwards, a certain number of finetuning epochs will be applied for recovering the accuracy loss due to the pruning.

In practice, the directpruning and finetuning stages can be carried out iteratively to gradually achieve the optimal trade-off between the model compression rate and accuracy.

Such a weight pruning approach demonstrated very high effectiveness, especially for fully-connected (fc) layers BID4 .

For conv layers, removing the redundant weights in structured forms, e.g., the filters and filter channels, has been widely investigated.

For example, proposed to apply group Lasso regularization on weight groups in a variety of self-defined sizes and shapes to remove redundant groups.

BID12 used the first-order Taylor series expansion of the loss function on feature maps to determine the rankings of filters and those in low ranking will be removed.

The filter ranking can also be represented by the root mean square or the sum of absolute values of filter weights BID11 .Activation Sparsity: The activation sparsity has been widely utilized in various DNN accelerator designs. , BID0 and BID14 accelerated the DNN inference with reduced off-chip memory access and computation cost benefiting from the sparse activations originated from ReLU.

A simple technique to improve activation sparsity by zeroing out small activations was also explored BID0 .

However, the increment of activation sparsity is still limited without accuracy loss.

The biggest issue in the aforementioned works is that they heavily relied on ReLU.

However, zero activations do not exist in non-ReLU activation function.

To regulate and stretch the activation sparsity, many dropout-based methods are proposed.

Adaptive dropout BID1 , for instance, developed a binary belief network overlaid on the original network.

The neurons with larger activation magnitude incur higher probability to be activated.

Although this method achieved a better regularization on DNNs, the inclusion of belief network complicated the training and had no help on inference speedup.

The winners-take-all (WTA) autoencoder was built with a regularization based on activation magnitude to learn deep sparse representations from various datasets BID10 ).As can be seen that the model size compression is the main focus of weight pruning, while the use of activation sparsification focuses more on the intrinsic activation sparsity by ReLU or exploring the virtue of sparse activation in the DNN training for better model generalization.

In contrast, our proposed IP aims for reducing the DNN computation cost and therefore accelerating the inference by integrating and optimizing both weight pruning and activation sparsification.

As depicted in FIG0 , the proposed IP consists of two steps by concatenating the activation pruning to the weight pruning.

Both stages seek for unimportant information (weights and activations, respectively) and mask them off.

We aim to keep only the important connections and activations to minimize the computation cost.

In this section, we will first explain the integration of the two steps.

The technical details in model quality (e.g., accuracy) control will then be introduced.

The prediction method for deriving activation masks is also proposed to speed up the inference of IPnets.

At last, the appropriate settings of dropout layers and training optimizers are discussed.

Weight pruning.

In the weight pruning stage, weight parameters with magnitude under a threshold are masked out, and weight masks will be passed to the following finetuning process.

After the model is finetuned for certain epochs to recover accuracy loss, weight masks need to be updated for the next finetuning round.

There are two crucial techniques to help weight pruning.

1) The threshold used to build weight masks are determined based on the weight distribution of each layer.

Because of different sensitivity for weight pruning, each layer owns a specific weight sparsity pattern.

Basically, the leading several conv layers are more vulnerable to weight pruning.

2) The whole weight pruning stage needs multiple pruning-finetuning recursions to search an optimal weight sparsity.

Weight masks are progressively updated to increase pruning strength.

Finetuning Activation sensitivity analysis Activation pruning.

While weak connections between layers are learned to be pruned, activaitons with small magnitude are taken as unimportant and can be masked out to further minimize interlayer connections, and hence to reduce computation cost.

Notice that, neurons in DNNs are trained to be activated in various patterns according to different input classes, thus dynamic masks should be learned in the activation pruning stage, which are different from the static masks in the weight pruning stage.

The selected activations by the dynamic mask are denoted as winners, and the winner rate is defined as: DISPLAYFORM0 where S winner and S total denote the number of winners and total activation number.

The winner rate per layer is determined by the analysis of activation pruning sensitivity layer-wise on the models obtained after weight pruning.

The winner activation after the pruning mask, A m , obeys the rule: DISPLAYFORM1 where ?? is the threshold derived at run-time from the activation winner rate for each layer, and A orig is the result from original activation function.

Same with weight pruning, the model with dynamic activation masks is finetuned to recover accuracy drop.

No iterative procedure of mask updating and finetuning is required in our activaiton pruning method.

Not all layers share the same winner rate.

Similar to the trend in weight pruning, deeper layers tolerate larger activation pruning strength.

To analyze the activation pruning sensitivity, the model with activation masks is tested on a validation set sampled from the training images with the same size as the testing set.

Accuracy drops are taken as the indicator of pruning sensitivity for different winner rate settings.

Before finetuning, the activation winner rate per layer is set empirically to keep accuracy drop less than 2%.

For the circumstances that model accuracy is resistant to be tuned back, winner rates in the leading several layers should be set smaller.

Examples of sensitivity analysis will be given and discussed in Section 5.

The dynamic activation pruning method increases the activation sparsity and maintains the model accuracy as well.

The solution of determining threshold ?? in Equation (2) for activation masks is actually a canonical argpartion problem to find top-k arguments in an array.

According to the Master Theorem BID2 , argpartition can be fast solved in linear time O(N ) through recursive algorithms, where N is the number of elements to be partitioned.

To further speed up, threshold prediction can be applied on the down-sampled activation set.

An alternate threshold ?? is predicted by selecting top-??k elements from the down-sampled activation set comprising ??N elements with ?? as the down-sampling rate.

?? is applied for the original activation set afterwards.

For DNN training, dropout layer is commonly added after large fc layers to avoid over-fitting problem.

The neuron activations are randomly chosen in the feed-forward phase, and weights updates will be only applied on the neurons associated with the selected activations in the back-propagation phase.

Thus, a random partition of weight parameters are updated in each training iteration.

Although the activation mask only selects a small portion of activated neurons, dropout layer is still needed, for the selected neurons with winner activations are always kept and updated, which makes over-fitting prone to happen.

In fc layers, the remaining activated neurons are reduced to S winner from S total neurons as defined in Equation (1).

The dropout layer connected after the activation mask is suggested to be modified with the setting: DISPLAYFORM0 where 0.5 is the conventionally chosen dropout rate in the training process for original models, and the activation winner rate is introduced to regulate the dropout strength for balancing over-fitting and under-fitting.

The dropout layers will be directly removed in the inference stage.

We find different optimizer requirements for weight pruning and activation pruning.

In the weight pruning stage, it's recommended to adopt the same optimizer used for training the original model.

The learning rate should be properly reduced to 0.1?? ??? 0.01?? of the original learning rate.

In the activation pruning stage, our experiments show that Adadelta (Zeiler, 2012) usually brings the best performance.

Adadelta adapts the learning rate for each individual weight parameter.

Smaller updates are performed on neurons associated with more frequently occurring activations, whereas larger updates will be applied for infrequent activated neurons.

Hence, Adadelta is beneficial for sparse weight updates, which is exactly the common situation in our activation pruning.

During finetuning, only a small portion of weight parameters are updated because of the combination of sparse patterns in weights and activations.

The learning rate for Adadelta is also reduced 0.1?? ??? 0.01?? compared to that used in training the original model.

All of our models and evaluations are implemented in TensorFlow.

IPnets are verified on various models ranging from simple multi-layer perceptron (MLP) to deep convolution neural networks (CNNs) on three datasets, MNIST, CIFAR-10 and ImageNet as in TAB0 .

For AlexNet BID8 and ResNet-32 BID19 , we focus on conv layers because conv layers account for more than 90% computation cost in these two models.

The compression results of IPnets on activations, weights and MACs are summarized in TAB0 compared to the original dense models.

IPnets achieve a 2.3?? ??? 5.8?? activation compression rate and a 2.5?? ??? 10?? weight compression rate.

Benefiting from sparse weights and activations, IPnets only need 3.65% ??? 28.9% of MACs required in dense models.

The accuracy drop is kept less than 0.5%, and for some cases, e.g., MLP-3 and AlexNet in TAB0 , the IPnets achieve a better accuracy.

TAB0 shows that our method can learn both sparser activations and sparse weights and thus save computation.

More importantly, in FIG2 , we will show that our approach is superior to ap-proaches which explore intrinsic sparse ReLU activations and state-of-the-art weight pruning.

The ReLU function brings intrinsic zero activations for MLP-3, ConvNet-5 and AlexNet in our experiments.

However, the non-zero activation percentage increases in weight-pruned (WP) models as depicted in FIG2 (a).

The increment of non-zero activations undermines the effort from weight pruning.

The activation pruning can remedy the activation sparsity loss and prune 7.7% -18.5% more activations even compared to the original dense models.

The largest gain from IP exits in ResNet-32 which uses leaky ReLU as activation function.

Leaky ReLU generates dense activations in the original and WP models.

The IPnet for ResNet-32 realizes a 61.4% activation reduction.

At last, IPnets reduce 4.4% ??? 22.7% more MACs compared to WP models as depicted in FIG2 (b), which means a 1.2?? ??? 2.7?? improvement.

More details on model configuration and analysis are discussed as follows.

The MLP-3 on MNIST has two hidden layers with 300 and 100 neurons respectively, and the model configuration details are summarized in TAB1 .

The amount of MACs is calculated with batch size as 1, and the non-zero activation percentage at the output per layer is averaged from random 1000 samples from the training dataset.

The following discussions on other models obey the same statistics setting.

The model size of MLP-3 is firstly compressed 10?? through weight pruning.

IP further reduces the total number of MACs to 3.65% by keeping only 17.1% activations.

The accuracy of the priginal dense model is 98.41% on MNIST, and the aggressive reduction of MACs (27.4??) doesn't decrease the accuracy.

For digit images in MNIST dataset have specific sparse features, the results on small-footprint MLP-3 are very promising.

IP is further applied for a 5-layers CNN, ConvNet-5, on a more complicated dataset, CIFAR-10.

With two conv layers and three fc layers, the original model has an 86% accuracy.

As shown in TAB2 , the IPnet for ConvNet-5 only needs 27.7% of total MACs compared to the dense model through pruning 59.6% of weights and 56.4% of activations at the same time.

The accuracy only has a marginal 0.06% drop.

The dominant computation cost is from conv layers accounting for more than 4/5 of total MACs for inference.

Although fc layers can generally be pruned in larger strength than conv layers, the computation cost reduction of IPnet is dominated by the pruning results in conv layers.

We push IP onto AlexNet for ImageNet ILSVRC-2012 dataset which consists of about 1.2M training images and 50K validating images.

The ALexNet comprises 5 conv layers and 3 fc layers and achieves 57.22% top-1 accuracy on the validation set.

Similar to ConvNet-5, the computation bottleneck of AlexNet exits in conv layers by consuming more than 9/10 of total MACs.

We focus on conv layers here.

As shown in TAB3 , deeper layers have larger pruning strength on weights and activations because of the sparse high-level feature abstraction of input images.

For example, the MACs of layer conv5 can be reduced 10??, while only a 1.2?? reduction rate is realized in layer conv1.

In total, the needed MACs are reduced 3.5?? using IP with 38.8% weights and 44.2% activations.

TAB4 .

The ResNet-32 consists of 1 conv layer, 3 stacked residual units and 1 fc layer.

Each residual unit contains 5 consecutive residual blocks.

The filter numbers in residual units increase rapidly, and same for weight amount.

An average pooling layer is connected before the last fc layer to reduce feature dimension.

Compared to conv layers, the last fc layer can be neglected in terms of weight volume and computation cost.

The original model has a 95.01% accuracy on CIFAR-10 dataset with 7.34G MACs per image.

Weight and activation pruning strength is designed unit-wise to reduce the exploration space of hyperparameters, i.e., threshold settings.

Notice that leaky ReLU is used as the activation function, thus zero activations are extremely hard to occur in the original and WP model.

Only with IP, the activation percentage can be reduced down to 38.6%.

As shown in TAB4 , the model size is compressed 3.1??, and the final gain is that 86.3% of MACs can be avoided while keeping the accuracy drop less than 0.5%.By randomly selecting 500 images from the training images, the activation distribution of the first residual block in baseline model is depicted in FIG4 (a).

Activations gather near zero with long tails towards both positive and negative directions.

The activation distribution after IP are shown in FIG4 (b) .

Activations near zero are pruned out, and the major contribution comes from removing small negative values.

In addition, the kept activations are trained to be stronger with larger magnitude, which is consistent with the phenomenon that the non-zero activation percentage increases after weight pruning when using ReLU as illustrated in FIG2 (a).

The static activation pruning approach has been widely adopted in efficient DNN accelerator designs BID0 BID14 .

By selecting a proper static threshold ?? in Equation (2), more activations can be pruned with little impact on model accuracy.

For the activation pruning in IP, the threshold is dynamically set according to the winner rate and activation distribution layer-wise.

The comparison between static and dynamic pruning is conducted on ResNet-32 for CIFAR-10 dataset.

For the static pruning setup, the ?? for leaky ReLU is assigned in the range of [0.07, 0.14], which brings different activation sparsity patterns.

40% 45% 50% 55% 60% 65% Non-zero activation percentage As the result of leaky ReLU with static threshold shown in FIG5 , the accuracy starts to drop rapidly when non-zero activation percentage is less than 58.6% (?? = 0.08).

Using dynamic threshold settings according to winner rates, a better accuracy can be obtained under the same activation sparsity constraint.

Finetuning the model using dynamic activation masks will dramatically recover the accuracy loss.

As our experiment in Section 4.4, the IPnet for ResNet-32 can be finetuned to eliminate the 10.4% accuracy drop caused by the static activation pruning.

In weight pruning, the applicable pruning strength is different per layer BID4 BID12 .

Similarly, the pruning sensitivity analysis is required to determine the proper activation pruning strength layer-wise, i.e., the activation winner rate per layer.

FIG7 shows two examples on WP models from AlexNet and ResNet-32.

For AlexNet in FIG7 (a), the accuracy drops sharply when the activation winner rate of layer conv1 is less than 0.3.

Meanwhile, the winner rate of layer conv5 can be set under 0.1 without hurting accuracy.

Deeper conv layers can support sparser activations.

The ResNet-32 in FIG7 (b) has a similar trend of activation pruning sensitivity.

Layer conv1 is most susceptible to the activation pruning.

Verified by thorough experiments in Section 4, the accuracy loss can be well recovered by finetuning with proper activation winner rates.

As discussed in Section 3.3, the process to select activation winners can be accelerated by threshold prediction on down-sampled activation set.

We apply different down-sampling rates on the IPnet for AlexNet.

As can be seen in FIG8 , layer conv1 is most vulnerable to threshold prediction.

From the overall results, it's practical to down-sample 10% (?? = 0.1) of activations by keeping the accuracy drop less than 0.5%.

To minimize the computation cost in DNNs, IP combining weight pruning and activation pruning is proposed in this paper.

The experiment results on various models for MNIST, CIFAR-10 and ImageNet datasets have demonstrated considerable computation cost reduction.

In total, a 2.3?? -5.8?? activation compression rate and a 2.5?? -10?? weight compression rate are obtained.

Only 3.65% -28.9% of MACs are left with marginal effects on model accuracy, which outperforms the weight pruning by 1.2?? -2.7??.

The IPnets are targeted for the dedicated DNN accelerator designs with efficient sparse matrix storage and computation units on chip.

The IPnets featuring compressed model size and reduced computation cost will meet the constraints from memory space and computing resource in embedded systems.

@highlight

This work advances DNN compression beyond the weights to the activations by integrating the activation pruning with the weight pruning. 

@highlight

An integral model compression method that handles both weight and activation pruning, leading to more efficient network computation and effective reduction of the number of multiply-and-accumulate.

@highlight

This article presents a novel approach to reduce the computation cost of deep neural networks by integrating activation pruning along with weight pruning and show that common techniques of exclusive weight pruning  increases the number of non-zero activations after ReLU.