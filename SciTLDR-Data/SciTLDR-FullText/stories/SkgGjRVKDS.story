Batch Normalization (BN) is one of the most widely used techniques in Deep Learning field.

But its performance can awfully degrade with insufficient batch size.

This weakness limits the usage of BN on many computer vision tasks like detection or segmentation, where batch size is usually small due to the constraint of memory consumption.

Therefore many modified normalization techniques have been proposed, which either fail to restore the performance of BN completely, or have to introduce additional nonlinear operations in inference procedure and increase huge consumption.

In this paper, we reveal that there are two extra batch statistics involved in backward propagation of BN, on which has never been well discussed before.

The extra batch statistics associated with gradients also can severely affect the training of deep neural network.

Based on our analysis, we propose a novel normalization method, named Moving Average Batch Normalization (MABN).

MABN can completely restore the performance of vanilla BN in small batch cases, without introducing any additional nonlinear operations in inference procedure.

We prove the benefits of MABN by both theoretical analysis and experiments.

Our experiments demonstrate the effectiveness of MABN in multiple computer vision tasks including ImageNet and COCO.

The code has been released in https://github.com/megvii-model/MABN.

Batch Normalization (BN) (Ioffe & Szegedy, 2015) is one of the most popular techniques for training neural networks.

It has been widely proven effective in many applications, and become the indispensable part of many state of the art deep models.

Despite the success of BN, it's still challenging to utilize BN when batch size is extremely small 1 .

The batch statistics with small batch size are highly unstable, leading to slow convergence during training and bad performance during inference.

For example, in detection or segmentation tasks, the batch size is often limited to 1 or 2 per GPU due to the requirement of high resolution inputs or complex structure of the model.

Directly computing batch statistics without any modification on each GPU will make performance of the model severely degrade.

To address such issues, many modified normalization methods have been proposed.

They can be roughly divided into two categories: some of them try to improve vanilla BN by correcting batch statistics (Ioffe, 2017; Singh & Shrivastava, 2019) , but they all fail to completely restore the performance of vanilla BN; Other methods get over the instability of BN by using instance-level normalization (Ulyanov et al., 2016; Ba et al., 2016; Wu & He, 2018) , therefore models can avoid the affect of batch statistics.

This type of methods can restore the performance in small batch cases to some extent.

However, instance-level normalization hardly meet industrial or commercial needs so far, for this type of methods have to compute instance-level statistics both in training and inference, which will introduce additional nonlinear operations in inference procedure and dramatically increase consumption Shao et al. (2019) .

While vanilla BN uses the statistics computed over the whole training data instead of batch of samples when training finished.

Thus BN is a linear operator and can be merged with convolution layer during inference procedure.

Figure 1 (a) shows with ResNet-50 (He et al., 2016) , instance-level normalization almost double the inference time compared with vanilla BN.

Therefore, it's a tough but necessary task to restore the performance of BN in small batch training without introducing any nonlinear operations in inference procedure.

In this paper, we first analysis the formulation of vanilla BN, revealing there are actually not only 2 but 4 batch statistics involved in normalization during forward propagation (FP) as well as backward propagation (BP).

The additional 2 batch statistics involved in BP are associated with gradients of the model, and have never been well discussed before.

They play an important role in regularizing gradients of the model during BP.

In our experiments (see Figure 2) , variance of the batch statistics associated with gradients in BP, due to small batch size, is even larger than that of the widelyknown batch statistics (mean, variance of feature maps).

We believe the instability of batch statistics associated with gradients is one of the key reason why BN performs poorly in small batch cases.

Based on our analysis, we propose a novel normalization method named Moving Average Batch Normalization (MABN).

MABN can completely get over small batch issues without introducing any nonlinear manipulation in inference procedure.

The core idea of MABN is to replace batch statistics with moving average statistics.

We substitute batch statistics involved in BP and FP with different type of moving average statistics respectively, and theoretical analysis is given to prove the benefits.

However, we observed directly using moving average statistics as substitutes for batch statistics can't make training converge in practice.

We think the failure takes place due to the occasional large gradients during training, which has been mentioned in Ioffe (2017) .

To avoid training collapse, we modified the vanilla normalization form by reducing the number of batch statistics, centralizing the weights of convolution kernels, and utilizing renormalizing strategy.

We also theoretically prove the modified normalization form is more stable than vanilla form.

MABN shows its effectiveness in multiple vision public datasets and tasks, including ImageNet (Russakovsky et al., 2015) , COCO (Lin et al., 2014) .

All results of experiments show MABN with small batch size (1 or 2) can achieve comparable performance as BN with regular batch size (see Figure 1(b) ).

Besides, it has same inference consumption as vanilla BN (see Figure 1(a) ).

We also conducted sufficient ablation experiments to verify the effectiveness of MABN further.

Batch normalization (BN) (Ioffe & Szegedy, 2015) normalizes the internal feature maps of deep neural network using channel-wise statistics (mean, standard deviation) along batch dimension.

It has been widely proven effectively in most of tasks.

But the vanilla BN heavily relies on sufficient batch size in practice.

To restore the performance of BN in small batch cases, many normalization techniques have been proposed: Batch Renormalization (BRN) (Ioffe, 2017) introduces renormalizing parameters in BN to correct the batch statistics during training, where the renormalizing parameters are computed using moving average statistics; Unlike BRN, EvalNorm (Singh & Shrivastava, 2019) corrects the batch statistics during inference procedure.

Both BRN and EvalNorm can restore the performance of BN to some extent, but they all fail to get over small batch issues completely.

Instance Normalization (IN) (Ulyanov et al., 2016) , Layer Normalization (LN) (Ba et al., 2016) , and Group normalization (GN) (Wu & He, 2018) (Peng et al., 2018) handle the small batch issues by computing the mean and variance across multiple GPUs.

This method doesn't essentially solve the problem, and requires a lot of resource.

Apart from operating on feature maps, some works exploit to normalize the weights of convolution: Weight Standardization (Qiao et al., 2019) centralizes weight at first before divides weights by its standard deviation.

It still has to combine with GN to handle small batch cases.

First of all, let's review the formulation of batch Normalization (Ioffe & Szegedy, 2015) : assume the input of a BN layer is denoted as X ∈ R B×p , where B denotes the batch size, p denotes number of features.

In training procedure, the normalized feature maps Y at iteration t is computed as:

where batch statistics µ Bt and σ

Bt are the sample mean and sample variance computed over the batch of samples B t at iteration t:

Besides, a pair of parameters γ, β are used to scale and shift normalized value Y :

The scaling and shifting part is added in all normalization form by default, and will be omitted in the following discussion for simplicity.

As Ioffe & Szegedy (2015) demonstrated, the batch statistics µ Bt , σ 2 Bt are both involved in backward propagation (BP).

We can derive the formulation of BP in BN as follows: let L denote the loss, Θ t denote the set of the whole learnable parameters of the model at iteration t. Given the partial

, the partial gradients

is computed as

where · denotes element-wise production, g Bt and Ψ Bt are computed as

It can be seen from (5) that g Bt and Ψ Bt are also batch statistics involved in BN during BP.

But they have never been well discussed before.

According to Ioffe & Szegedy (2015) , the ideal normalization is to normalize feature maps X using expectation and variance computed over the whole training data set:

But it's impractical when using stochastic optimization.

Therefore, Ioffe & Szegedy (2015) uses mini-batches in stochastic gradient training, each mini-batch produces estimates the mean and variance of each activation.

Such simplification makes it possible to involve mean and variance in BP.

From the derivation in section 3.1, we can see batch statistics µ Bt , σ

|Θ t ] are computed over the whole data set.

They contain the information how the mean and the variance of population will change as model updates, so they play an important role to make trade off between the change of individual sample and population.

Therefore, it's crucial to estimate the population statistics precisely, in order to regularize the gradients of the model properly as weights update.

It's well known the variance of MC estimator is inversely proportional to the number of samples, hence the variance of batch statistics dramatically increases when batch size is small.

Figure 2 shows the change of batch statistics from a specific normalization layer of ResNet-50 during training on ImageNet.

Regular batch statistics (orange line) are regarded as a good approximation for population statistics.

We can see small batch statistics (blue line) are highly unstable, and contains notable error compared with regular batch statistics during training.

In fact, the bias of g Bt and Ψ Bt in BP is more serious than that of µ Bt and σ 2 Bt (see Figure 2 (c), 2(d)).

The instability of small batch statistics can worsen the capacity of the models in two aspects: firstly the instability of small batch statistics will make training unstable, resulting in slow convergence; Secondly the instability of small batch can produce huge difference between batch statistics and population statistics.

Since the model is trained using batch statistics while evaluated using population statistics, the difference between batch statistics and population statistics will cause inconsistency between training and inference procedure, leading to bad performance of the model on evaluation data.

Based on the discussion in Section 3.2, the key to restore the performance of BN is to solve the instability of small batch statistics.

Therefore we considered two ways to handle the instability of small batch statistics: using moving average statistics to estimate population statistics, and reducing the number of statistics by modifying the formulation of normalization.

Moving average statistics seem to be a suitable substitute for batch statistics to estimate population statistics when batch is small.

We consider two types of moving average statistics: simple moving average statistics (SMAS) 2 and exponential moving average statistics (EMAS) 3 .

The following theorem shows under mild conditions, SMAS and EMAS are more stable than batch statistics:

Theorem 1 Assume there exists a sequence of random variable (r.v.) {ξ t } ∞ t=1 , which are independent, uniformly bounded, i.e. ∀t, |ξ t | < C, and have uniformly bounded density.

Define:

where

then we have

If the sequence

then we have

The proof of theorem 1 can be seen in appendix A.1.

Theorem 1 not only proves moving average statistics have lower variance compared with batch statistics, but also reveals that with large momentum α, EMAS is better than SMAS with lower variance.

However, using SMAS and EMAS request different conditions: Condition (8)

.

However, under the assumption that learning rate is extremely small, the difference between the distribution of ξ t−1 and ξ t is tiny, thus condition (10) is satisfied, we can use SMAS to replace

.

In a word, we can use EMASμ t ,σ 2 t to replace µ Bt , σ 2 Bt , and use SMASḡ t ,Ψ t to replace g Bt , Ψ Bt in (1) and (4), wherê

Notice neither of SMAS and EMAS is the unbiased substitute for batch statistics, but the bias can be extremely small comparing with expectation and variance of batch statistics, which is proven by equation 11 in theorem 1, our experiments also prove the effectiveness of moving average statistics as substitutes for small batch statistics (see Figure 3 , 4 in appendix B.1).

Relation to Batch Renormalization Essentially, Batch Renormalization (BRN) (Ioffe, 2017) replaces batch statistics µ Bt , σ 2 Bt with EMASμ t ,σ 2 t both in FP (1) and BP (4).

The formulation of BRN during training is written as:

where

).

Based on our analysis, BRN successfully eliminates the effect of small batch statistics µ Bt and σ 2 Bt by EMAS, but the small batch statistics associated with gradients g Bt and Ψ Bt remains during backward propagation, preventing BRN from completely restoring the performance of vanilla BN.

2 The exponential moving average (EMA) for a series {Yt}

To further stabilize training procedure in small batch cases, we consider normalizing feature maps X using EX 2 instead of EX and V ar(X).

The formulation of normalization is modified as:

where

.

Given ∂L ∂Y , the backward propagation is:

The benefits of the modification seems obvious: there's only two batch statistics left during FP and BP, which will introduce less instability into the normalization layer compared with vanilla normalizing form.

In fact we can theoretically prove the benefits of the modification by following theorem:

Theorem 2 If the following assumptions hold:

Then we have:

The proof can be seen in appendix A.2.

According to (17), V ar[∂L/∂X vanilla ] is larger than that of V ar[∂L/∂X modif ied ], the gap is at least V ar[g B ]/σ 2 , which mainly caused by the variance of g B /σ.

So the modification essentially reduces the variance of the gradient by eliminating the batch statistics g B during BP.

Since g Bt is a Monte Carlo estimator, the gap is inversely proportional to batch size.

This can also explain why the improvement of modification is significant in small batch cases, but modified BN shows no superiority to vanilla BN within sufficient batch size (see ablation study in section 5.1).

Centralizing weights of convolution kernel Notice theorem 2 relies on assumption 3.

The vanilla normalization naturally satisfies Ey = 0 by centralizing feature maps, but the modified normalization doesn't necessarily satisfy assumption 3.

To deal with that, inspired by Qiao et al. (2019) , we find centralizing weights W ∈ R q×p of convolution kernels, named as Weight Centralization (WC) can be a compensation for the absence of centralizing feature maps in practice:

where X input , X output are the input and output of the convolution layer respectively.

We conduct further ablation study to clarify the effectiveness of WC (see Table 4 in appendix B.2).

It shows that WC has little benefits to vanilla normalization, but it can significantly improve the performance of modified normalization.

We emphasize that weight centralization is only a practical remedy for the absence of centralizing feature maps.

The theoretical analysis remains as a future work.

Clipping and renormalizing strategy.

In practice, we find directly substituting batch statistics by moving average statistics in normalization layer will meet collapse during training.

Therefore we take use of the clipping and renormalizing strategy from BRN (Ioffe, 2017) .

All in all, the formulation of proposed method MABN is:

where the EMASχ t is computed asχ t = αχ t−1 + (1 − α)χ Bt , SMASΨ t is defined as (13).

The renormalizing parameter is set as r = clip [1/λ,λ] (

This section presents main results of MABN on ImageNet (Russakovsky et al., 2015) , COCO (Lin et al., 2014) .

Further experiment results on ImangeNet, COCO and Cityscapes (Cordts et al., 2016) can be seen in appendix B.2, B.3, B.4 resepectively.

We also evaluate the computational overhead and memory footprint of MABN, the results is shown in appendix B.5.

We evaluate the proposed method on ImageNet (Russakovsky et al., 2015) classification datatsets with 1000 classes.

All classification experiments are conducted with ResNet-50 (He et al., 2016 Comparison with other normalization methods.

Our baseline is BN using small (|B| = 2) or regular (|B| = 32) batch size, and BRN (Ioffe, 2017) with small batch size.

We don't present the performance of instance-level normalization counterpart on ImageNet, because they are not linear-type method during inference time, and they also failed to restore the performance of BN (over +0.5%), according to Wu & He (2018) .

Table 1 shows vanilla BN with small batch size can severely worsen the performance of the model(+11.81%); BRN (Ioffe, 2017) alleviates the issue to some extent, but there's still remaining far from complete recovery(+6.88%); While MABN almost completely restore the performance of vanilla BN(+0.17%).

We also compared the performance of BN, BRN and MABN when varying the batch size (see Figure  1(b) ).

BN and BRN are heavily relies on the batch size of training, though BRN performs better than vanilla BN.

MABN can always retain the best capacity of ResNet-50, regardless of batch size during training.

Ablation study on ImageNet.

We conduct ablation experiments on ImageNet to clarify the contribution of each part of MABN (see table 2 ).

With vanilla normalization form, replacing batch statistics in FP with EMAS (as BRN) will restore the performance to some extents(−4.93%, comparing 3 and 4 ), but there's still a huge gap (+6.88%, comparing 1 and 4 ) from complete restore.

Directly using SMAS in BP with BRN will meet collapse during training ( 5 ), no matter how we tuned hyperparameters.

We think it's due to the instability of vanilla normalization structure in small cases, so we modify the formulation of normalization shown in section 4.2.

The modified normalization even slightly outperforms BRN in small batch cases (comparing 4 and 6 ).

However, modified normalization shows no superiority to vanilla form (comparing 1 and 2 ), which can be interpreted by the result of theorem 2.

With EMAS in FP, modified normalization significantly reduces the error rate further (comparing 6 and 7 ), but still fail to restore the performance completely (+3.62%, comparing 1 and 7 ).

Applying SMAS in BP finally fills the rest of gap, almost completely restore the performance of vanilla BN in small batch cases (+0.17 ,comparing 1 and 8 ).

Table 2 : Ablation study on ImageNet Classification with ResNet-50.

The normalization batch size is 2 in all experiments otherwise stated.

The memory size is 16 and momentum is 0.98 when using SMAS, otherwise the momentum is 0.9. "-" means the training can't converge.

We conduct experiments on Mask R-CNN (He et al., 2017) benchmark using a Feature Pyramid Network(FPN) (Lin et al., 2017a) following the basic setting in He et al. (2017) .

We train the networks from scratch

Since {ξ t } ∞ t=1 are independently, hence we have:

as t → ∞. Hence (9) has been proven.

If the condition (10) is satisfied.

Since {ξ t } ∞ t=1 is uniformly bounded, then ∃C ∈ R + , ∀, |ξ t | < C. As t → ∞, We have

Similarly, we have

Therefore combining (24) and (25), we have

For a fixed memory size m, as t → ∞, we have

Therefore, (11) has been proven.

Without loss of generality, given the backward propagation of two normalizing form of a single input x with batch B:

where g B , Ψ B are the batch statistics, andσ,χ are the EMAS, defined as before.

We omitted the subscript t for simplicity.

Then the variance of partial gradients w.r.t.

inputs x is written as

where (30) is satisfied due to assumption 1.

The variance ofσ is so small thatσ can be regarded as a fixed number; (31) is satisfied because

Due to assumption 2, the correlation between individual sample and batch statistics is close to 0, hence we have

E[yΨ B ] = EyEΨ B (38) Besides, Ey is close to 0 according to assumption 3, hence

(32) is satisfied due to the definition ofχ andσ, we havê χ 2 =σ 2 +μ 2 .

(40) Similar toσ, the variance ofχ is also too small thatχ can be regarded as a fixed number due to assumption 1, so (33) is satisfied.

We analyze the difference between small batch statistics (|B| = 2) and regular batch statistics (|B| = 32) with the modified formulation of normalization (15)

Implementation details.

All experiments on ImageNet are conducted across 8 GPUs.

We train models with a gradient batch size of B g = 32 images per GPU.

To simulate small batch training, we split the samples on each GPU into B g /|B| groups where |B| denotes the normalization batch size.

The batch statistics are computed within each group individually.

All weights from convolutions are initialized as He et al. (2015) .

We use 1 to initialize all γ and 0 to initialize all β in normalization layers.

We use a weight decay of 10 −4 for all weight layers including γ and β (following Wu & He (2018) ).

We train 600, 000 iterations (approximately equal to 120 epoch when gradient batch size is 256) for all models, and divide the learning rate by 10 at 150, 000, 300, 000 and 450, 000 iterations.

The data augmentation follows Gross & Wilber (2016) .

The models are evaluated by top-1 classification error on center crops of 224 × 224 pixels in the validation set.

In vanilla BN or BRN, the momentum α = 0.9, in MABN, the momentum α = 0.98.

Additional ablation studies.

for instance segmentation.

Other basic settings follow He et al. (2017) .

MABN used on heads.

We build mask-rcnn baseline using a Feature Pyramid Network(FPN) (Lin et al., 2017a) backbone.

The base model is ResNet-50.

We train the models for 2× iterations.

We use 4conv1fc instead of 2fc as the box head.

Both backbone and heads contain normalization layers.

We replace all normalization layers in each experiments.

While training models with MABN, we use batch statistics in normalization layers on head during first 10,000 iterations.

Training from scratch for one-stage model.

We also compare MABN and SyncBN based on one-stage pipeline.

We build on retinanet (Lin et al., 2017b) benchmark.

We train the model from scratch for 2× iterations.

The results are shown in Table 7 .

All experiment results shows MABN can get comparable as SyncBN, and significantly outperform BN on COCO.

We evaluate semantic segmentation in Cityscapes (Cordts et al., 2016) .

It contains 5,000 high quality pixel-level finely annotated images collected from 50 cities in different seasons.

We conduct experiments on PSPNET baseline and follow the basic settings mentioned in Zhao et al. (2017) .

For fair comparison, our backbone network is ResNet-101 as in Chen et al. (2017) .

Since we centralize weights of convolutional kernel to use MABN, we have to re-pretrain our backbone model on Imagenet dataset.

During fine-tuning process, we linearly increase the learning rate for 3 epoch (558 iterations) at first.

Then we follow the "poly" learning schedule as Zhao et al. (2017) .

We compare the computational overhead and memory footprint of BN, GN and MABN.

We use maskrcnn with resnet50 and FPN as benchmark.

We compute the theoretical FLOPS during inference and measure the inference speed when a single image (3×224×224) goes through the backbone (resnet50 + FPN).

We assume BN and MABN can be absorbed in convolution layer during inference.

GN can not be absorbed in convolution layer, so its FLOPS is larger than BN and MABN.

Besides GN includes division and sqrt operation during inference, therefore it's much slower than BN and MABN during inference time.

We also monitor the training process of maskrcnn on COCO (8 GPUs, 2 images per GPU), and show its memory footprint and training speed.

Notice we have not optimized the implementation of MABN, so its training speed is a little slower than BN and GN.

@highlight

We propose a novel normalization method to handle small batch size cases.

@highlight

A method to deal with the small batch size problem of BN which applies moving average operation without too much overhead and reduces the number of statistics of BN for better stability.