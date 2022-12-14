In this paper, we propose an efficient framework to accelerate convolutional neural networks.

We utilize two types of acceleration methods: pruning and hints.

Pruning can reduce model size by removing channels of layers.

Hints can improve the performance of student model by transferring knowledge from teacher model.

We demonstrate that pruning and hints are complementary to each other.

On one hand, hints can benefit pruning by maintaining similar feature representations.

On the other hand, the model pruned from teacher networks is a good initialization for student model, which increases the transferability between two networks.

Our approach performs pruning stage and hints stage iteratively to further improve the performance.

Furthermore, we propose an algorithm to reconstruct the parameters of hints layer and make the pruned model more suitable for hints.

Experiments were conducted on various tasks including classification and pose estimation.

Results on CIFAR-10, ImageNet and COCO demonstrate the generalization and superiority of our framework.

In recent years, convolutional neural networks (CNN) have been applied in many computer vision tasks, e.g. classification BID21 ; BID6 , objects detection BID8 ; BID30 , and pose estimation BID25 .

The success of CNN drives the development of computer vision.

However, restricted by large model size as well as computation complexity, many CNN models are difficult to be put into practical use directly.

To solve the problem, more and more researches have focused on accelerating models without degradation of performance.

Pruning and knowledge distillation are two of mainstream methods in model acceleration.

The goal of pruning is to remove less important parameters while maintaining similar performance of the original model.

Despite pruning methods' superiority, we notice that for many pruning methods with the increase of pruned channel number, the performance of pruned model drops rapidlly.

Knowledge distillation describes teacher-student framework: use high-level representations from teacher model to supervise student model.

Hints method BID31 shares a similar idea of knowledge distillation, where the feature map of teacher model is used as high-level representations.

According to BID36 , the student network can achieve better performance in knowledge transfer if its initialization can produce similar features as the teacher model.

Inspired by this work, we propose that pruned model outputs similar features with original model's and provide a good initialization for student model, which does help distillation.

And on the other hand, hints can help reconstruct parameters and alleviate degradation of performance caused by pruning operation.

FIG0 illustrates the motivation of our framework.

Based on this analysis, we propose an algorithm: we do pruning and hints operation iteratively.

And for each iteration, we conduct a reconstructing step between pruning and hints operations.

And we demonstrate that this reconstructing operation can provide a better initialization for student model and promote hints step (See FIG1 .

We name our method as PWH Framework.

To our best knowledge, we are the first to combine pruning and hints together as a framework.

Our framework can be applied on different vision tasks.

Experiments on CIFAR- 10 Krizhevsky & Hinton (2009) , ImageNet Deng et al. (2016) and COCO Lin et al. (2014) Hints can help pruned model reconstruct parameters.

And the network pruned from the teacher model can provide a good initialization for student model in hints learning.effectiveness of our framework.

Furthermore, our method is a framework where different pruning and hints methods can be included.

To summarize, the contributions of this paper are as follows: FORMULA0 We analyze the properties of pruning and hints methods and show that these two model acceleration methods are complementary to each other.

(2) To our best knowledge, this is the first work that combines pruning and hints.

Our framework is easy to be extended to different pruning and hints methods.

(3) Sufficient experiments show the effectiveness of our framework on different datasets for different tasks.

Recently, model acceleration has received a great deal of attention.

Quantization methods BID29 BID5 ; BID20 BID39 reduce model size by quantizing float parameters to fixed-point parameters.

And fixed-point networks can be speeded up on special implementation.

Group convolution based methods BID16 BID3 separates a convolution operation into several groups, which can reduce computation complexity.

Several works exploit linear structure of parameters and approach parameters using lowrank way to reduce computational parameters BID7 BID19 ; BID0 .

In our experiments, we use two of current mainstream model acceleration way: pruning and knowledge distillation.

Network pruning has been proposed for a long time, such as BID11 ; BID12 ; BID22 .

Recent pruning methods can be roughly adopted in two levels, i.e. channel-wise BID28 ; BID14 BID32 ; BID17 and parameter-wise BID10 ; BID35 ; BID23 BID34 ; BID27 .In this paper, we use channel-wise approach as our pruning method.

There are many methods in channel-wise family.

He et al. BID14 prune channels in LASSO regression way from sample feature map.

Proposed in BID26 , the scale parameters in Batch Normalization layers are used to evaluate the importance of different channels.

BID28 use taylor formula to analyze each channel's contribution to the loss and prune the lowest contribution channel.

We utilize this method in our framework.

Despite the superiority of pruning methods, we find that the effectiveness of them will observably decrease with the increase of pruned channel numbers.

Knowledge distillation (KD) BID15 is the pioneering work of this field.

Hinton et al. BID15 define soft targets and use it to supervise student networks.

Beyond soft targets, hints are introduced in Fitnets BID31 , which can be explained as whole feature map mimic learning.

Several researches have focused on hints.

BID37 propose atloss to mimic combined output of an ensemble of large networks using student networks.

Furthermore, Li et al. BID24 demonstrate a mimic learning strategy based on region of interests to improve small networks' performance for object detection.

However, most of these works train student model from scratch and ignore the significance of student networks' initialization.

In this section, we will describe our method in details.

First we show hints and pruning methods which have been used in our framework.

Then we introduce reconstructing operation and analyze its effectiveness.

Finally, combining hints, pruning and reconstructing operation, we propose PWH Framework.

The pruning method we use in this paper is based on BID28 .

The algorithm can be described as a combinatorial optimization problem: DISPLAYFORM0 Where C(??) is the cost function of the task, D is the training samples, W and W are the parameters of original and pruning networks.

In the optimization problem, ||W || 0 bounds the number of nonzero parameters in W .

The parameter W i whether to be pruned completely depends on its outputs h i .

And the problem can be redescribed as minimizing ???C( DISPLAYFORM1 However, it's hard to find global optimal solution of the problem.

Using taylor expansion can get approximate formula of the objective function: DISPLAYFORM2 During backpropagation, we can get gradient ??C ??hi and activation h i .

So this criteria can be easily implemented for channel pruning.

Hints can provide an extra supervision for student network, and it usually combines with task loss.

The whole loss function of hints learning is represented as follows: DISPLAYFORM0 Where L t is the task loss (e.g. classification loss) and L h is the hints loss.

?? is hints loss weight which determines the intensity of hints supervision.

There are many types of hints methods.

Different hints methods are suitable for different tasks and network architectures.

To demonstrate the superiority and generalization of our framework, we try three kinds of hints methods in our experiments: L2 hints, normalization hints and adaption hints.

We introduce L2 hints, normalization hints in appendix.

First we slim the original network with reducing certain number of channels.

Then we reconstruct the hints layer of the pruned model to minimize the difference of feature map between teacher and student.

Finally, we start hints step to advance the performance of pruned model.

Adaption Hints demonstrates that it's necessary to add an adaption layer between student and teacher networks.

The adaption layer can help transfer student layer feature space to teacher layer feature space, which will promote hints learning.

The expression of adaption hints is described in equation 4.

DISPLAYFORM1 Where r(??) is adaption layer.

And for convolutional neural networks, adaption layer is 1 ?? 1 convolution layer.

The objective function of reconstructing step is: DISPLAYFORM0 Where Y is the feature map of original (teacher) model, X is the input of hints layer and W is the parameter of hints layer.

The optimization problem has close-form solution using least square method: DISPLAYFORM1 However, because some datasets (e.g. ImageNet BID6 ) have numerous images, X will be high-dimension matrix.

And it's impossible to solve the problem which involves such huge matrix in one time.

So we randomly sample images in dataset and compute X according to these images.

Due to the randomness, the reconstructed weights may be worse than original weights.

Thus, we finally use a switch to select better weights (See equation 6).W f = arg min DISPLAYFORM2 Where W f , W o and W r are the final weights, original weights and reconstructed weights of hints layer.

Y 0 and X 0 are computed from the whole dataset.

The objective function of Normalized L2 loss (See equation 16) is different from L2 loss, but we explain that the reconstruction step is also effective for normalized L2 loss.

The details of proof is available in supplementary material.

Combining pruning step, reconstructing step and hints step, we propose our PWH Framework.

The framework iteratively conducts three steps.

For pruning, we reduce the model size by certain num-bers of channels.

Then the parameters of hints layer will be reconstructed to minimize the difference of feature map between pruned model's and teacher model's.

Finally we use pruned model as student, original model as teacher and conduct hints learning.

And in the next iteration, the original model will be substituted by the student model in this iteration.

After training, the student model becomes the teacher model for the next iteration.

And another hints step is implemented at the ending of the framework where the teacher model will be set as the original model at the beginning of training (the teacher model in the first iteration).

The pseudocode of our approach is provided in appendix.

We demonstrate that compared with the original model in the first iteration, the student model in the current iteration is a better candidate for the teacher model in next hints step.

The reason is that the model before pruned and after pruned have more similar feature map and parameters, which can promote and speed up hints step.

At the end of the whole framework, we do another hints step.

Different from preceding hints step, the teacher model is selected as the original model in the first iteration.

We demonstrate that the final hints step is like the finetune step in pruning methods, which may need long-time training and improves the performance of compressed model.

And original model in the first iteration will be the better teacher.

FIG1 shows the pipeline of our framework.

The hints and pruning in PWH Framework are complementary to each other.

On one hand, pruned model is a good initialization to student model in hints step.

We propose that the feature map of pruned model is similar to original model's compared with random initialization model's.

In this way, proposed in BID36 , the transferability between student and teacher network will increase, which is beneficial for hints learning.

Experiments in ??4.4 demonstrate that the difference between original model's and pruned model's feature map is much smaller than random initialized model's.

On the other hand, hints helps pruning reconstruct parameters.

We demonstrate that when pruned channels number is large, pruning method is inefficient.

The pruning operation will bring large degradation of performance in this case.

We find that pruning numerous channels will destroy the main structure of networks(See 3).

And hints can alleviate this trend and help recover the structure and reconstruct parameters in model(See 4).The motivation of reconstruct step is the generalization of our method.

Our approach is a framework and it should be available for different pruning methods.

However, the criteria of some pruning methods are not based on minimizing the reconstructing error of feature map .

In other words, there is still room to improve the similarity between feature map of original (teacher) and pruned (student) networks, which is beneficial for hints learning.

We only conduct reconstructing operation on hints layer because it can not only reduce the difference of feature map used for hints but also maintain the main structure of the pruned model (See experiments in 4.3.3).

Moreover, for adaption hints methods, it need to initialize adaption layer(hint layer).

Compared with random initialization, reconstruction operation can help to construct this layer and provide more similar features with teacher models'.

We conduct experiments on CIFAR-10 Krizhevsky & Hinton (2009) , ImageNet BID6 and COCO Lin et al. (2014) for classfication and pose estimation tasks to demonstrate the superiority of PWH Framework.

In this section, we first introduce implementation details in different experiments on different datasets.

Then we compare our method with pruning methods and hint methods.

We train networks using PyTorch deep learning framework.

Pruning-only refers to a classical iterative pruning and finetuning operation.

And for hints-only methods, we set original model as teacher model and use the compressed random initialized model as student model.

For fair comparison, the student model in hints-only shares the same network structure with student model in PHW Frame- TAB1 illustrates results.

We can find that PWH Framework outperforms pruning-only method and hints-only method for a large margin on all datasets for different tasks, which verify the effectiveness of our framework and also shows that hints and pruning can be combined to improve the performance.

Results on different tasks and models show that PWH Framework can be implemented without the restriction of tasks and network architectures.

Moreover, illustrated in table 1, our framework can be applied for different pruning ratios, which means that we can adjust pruning ratio in the framework for different tasks to achieve different acceleration ratios.

To further analyze PWH Framework, we do several ablation studies.

All experimetns are conducted on CIFAR-10 dataset using VGG16BN.

The feature map proposed in this section refers to the output of last convolution layer, which is also the feature map used for hints learning.

And in this section, we do ablation study for three different aspects.

First, we do experiments to show iterative operation is an important component of PWH Framework.

Then we study on the selection of teacher model in hints step.

Finally, we validate on the effects of reconstructing step.

In PWH Framework, we implement three steps iteratively.

And in this section we will show the importance and necessity of iterative operation.

We conduct an experiment to compare the effects The relationship between the performance of network and the number of pruned channels using different methods.

We conduct experiment iteratively and for each iteration we prune 256 channels.of doing pruning and hints only once (i.e. First do pruning and then do hints.

Both operations are conducted only once.) and doing pruning and hints iteratively.

TAB2 shows results.

We can see that iterative operation can improve the performance of model dramatically.

To further explain this result, we do another experiment: we analyze the relationship between the performance of pruned model and the number of pruned channels.

Results in FIG3 illustrate that when the number of pruned channels is large, the performance of pruned model will drop rapidlly.

Thus, if we only do pruning and hints once, pruning will bring large degradation of performance and pruned model cannot output the similar feature to original model's.

And in this way, pruned model is not a more resonable initialization to student model.

Pruning step is useless to hints step in this situation.

The teacher model is the pruned model from previous iteration in PWH Framework.

Original model at the beginning of training can be another choice for teacher model in each iteration.

We do an experiment to compare these two set-up for teacher model.

And in this experiment, we prune 256 channels in each iteration.

FIG4 shows results.

We observe that when iteration is small, using original model in the first iteration as teacher model has a comparable performance with using the pruned model in the previous iteration.

However, with the increase of iterations, we can find that superiority of using the pruned model in the previous iteration increases.

The reason is that the original model in the first iteration has higher accuracy so it performs well when iteration is small.

But when iteration becomes large, pruned model's feature map will have large difference with original model's feature map.

And in this situation, there is a gap between pruned model and teacher model in hints step.

On the other hand, using the pruned model in the previous iteration will increase the similarity of feature map between student model's and teacher model's, which will help distillation in hints step.

Proposed in ??3.3, reconstructing step is used to further refine pruned model's feature and make it more similar to teacher's.

We conduct the experiment to validate the effectiveness of reconstructing step.

To fairly compare, we implement PWH Framework with and without reconstructing step.

In each iteration, we prune 256 channels.

We study on the accuracy of compressed model using two different methods.

Furthermore, we also analyze L2 loss between pruned model's and original model's feature map in each iteration.

Figure 5 shows experiment results.

We find that the method with reconstructing step performs better.

We want our framework to be adaptive to different pruning methods but some of the pruning methods' criteria are not minimizing the reconstructing feature map's error.

Reconstructing step can be used to solve this problem and increase the similarity of feature maps between two models.

To further analyze the properties of PWH Framework, we conduct further experiments on our approach.

The experiments results verify our assumptions: pruning and hints are complementary to each other.

All experiments are conducted on CIFAR-10 dataset using VGG16 as the original model.

We conduct experiment to compare the reconstructing feature map error between pruned model and random initial model.

We use the pruning method in ??3.1 to prune certain number of channels from original network and we calculate L2 loss between pruned model's feature map and original model's feature map.

Similarly, we randomly initialize a model whose size is same to the pruned model and calculate L2 loss of feature map between this model and original model.

Then we increase pruned channel number and record these two errors accordingly.

In TAB2 , we notice that in a large range (0-1024 pruned channels) pruned model's feature map is much more similar to original model's.

And this demonstrate that the transferability between pruned model and original model is larger.

And student model, initialized with the weights of pruned model, can perform better in hints learning.

To demonstrate hints method is beneficial to pruning, we first compare experiments between pruning-only with pruning and hints.

Different from PWH Framework, pruning and hints method used in this section doesn't have reconstructing step.

This is because we want to show the effectiveness of hints and reconstructing step is an extraneous variable.

In contrast experiments, we iteratively implement pruning and hints operations .

To fairly compare, in pruning and hints method, we substitute finetune operation for hints operation to get pruning-only method.

In Figure 6 , we observe that pruning and hints method has comparable performance with pruning-only method on small amount of iteration.

However, the margin of two methods becomes larger and larger with the increase of iterations (pruned channels number).

This phenomenon caused by the huge performance degradation in pruning operation when the original model is small.

The small model doesn't have many redundant neurons and the main structure will be broken in pruning.

And hints can alleviate this trend and help reconstruct parameters in pruned model.

In this paper, we propose PWH Framework, an iterative framework for model acceleration.

Our framework takes the advantage of both pruning and hints methods.

To our best knowledge, this is the first work that combine these two model acceleration methods.

Furthermore, we conduct reconstructing operation between hints and pruning steps as a cascader.

We analyze the property of these two methods and show they are complementary to each other: pruning provides a better initialization for student model and hints method helps to adjust parameters in pruned model.

Experiments on CIFAR-10, ImageNet and COCO datasets for classification and pose estimation tasks demonstrate the superiority of PWH Framework.

In this supplementary material, we first provide more implementation details for better illustration of our experiments.

In the second part, we give a proof in ??3.3.

We show that the upper bound of normalized L2 loss will decrease if L2 loss decreases theoretically.

Following section contains more implementation details and We use PyTorch deep learning framework with 4 NVIDIA Titan X GPUs.

A.0.1 CIFAR-10:CIFAR-10 dataset has 10 classes containing 6000 32 ?? 32 color images each.

50000 images are used for training and 10000 for test.

We use top-1 error for evaluation.

For model, we use the VGG-16 BID33 network with BatchNorm BID18 .

In CIFAR-10 finetune (hints) step, we use standard data augmentation with random cropping with 4 pixels, mean substraction of (0.4914, 0.4822, 0.4465) and std to (0.2023, 0.1994, 0.2010) .

A batch size of 128 and learning rate of 1e-3 are used .

We set finetune (hints) epoch as 20.

The hints method utilized on this dataset is L2 hints method with loss weight 10.

We sample 1000 images to reconstruct weights.

ImageNet classification dataset consists of 1000 classes.

We train models on 1.28 million training images and test models on 100k images.

Top-1 error is used to evaluate models.

In this experiment, ResNet18 is our original model and we use PWH Framework to compress it.

During finetune (hints) stage, the batchsize is 256, learning rate is 1e-3.

We set mean substraction to (0.485, 0.456, 0.406) and std to (0.229, 0.224, 0.225).

The loss weight is set as 0.5.

The random cropping is used.

In rescontructing stage, 1000 images are sampled to reconstruct hints layer weights.

We conduct pose estimation experiment on COCO dataset.

In this experiment, we train our models on trainval dataset and evaluate models on minival set.

The evaluation criteria for COCO dataset we use is OKS-based mAP.

We use ResNet18 with FPN Chen et al. (2018) as the original model in the experiment.

And in this experiment, we use random cropping, random rotation and random scale as our data augmentation strategy.

We use a weight decay of 1e-5 and learning rate of 5e-5.

The loss weight is set as 0.5.

A batch size of 96 is used.

The number of sampled images in reconstructing step is 500.

In reconstructing step, we use least square method to reconstruct parameters in hints layer.

The objective function of this step can be described in equation 7.

DISPLAYFORM0 Where Y is the feature map of original (teacher) model, X is the input of hints layer and W is the parameter of hints layer.

However, many hints methods use normalized L2 loss as hints loss (See equation 8).

It's difficult to optimize the problem using common methods if we set normalized L2 loss as objective function.

DISPLAYFORM1 In this section, we will show that the upper bound of normalized L2 loss is related to L2 loss.

In other words, if L2 loss decreases, the upper bound of L2 loss will decrease.

For an image x, y is the feature map of teacher network whose input is x. We suppose that: y = W x+e = y 0 + e DISPLAYFORM2 Where e is the error and it is independent with x. E[??] means the expectation.

We suppose that 1 y can be expressed as the function of y 0 and e using taylor expansion.

Where I is the identity matrix.

For convenience, we assume that K =

<|TLDR|>

@highlight

This is a work aiming for boosting all the existing pruning and mimic method.