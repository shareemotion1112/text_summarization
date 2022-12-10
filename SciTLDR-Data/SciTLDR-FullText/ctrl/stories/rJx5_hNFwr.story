Unsupervised domain adaptive object detection aims to learn a robust detector on the domain shift circumstance, where the training (source) domain is label-rich with bounding box annotations, while the testing (target) domain is label-agnostic and the feature distributions between training and testing domains are dissimilar or even totally different.

In this paper, we propose a gradient detach based Stacked Complementary Losses (SCL) method that uses detection objective (cross entropy and smooth l1 regression) as the primary objective, and cuts in several auxiliary losses in different network stages to utilize information from the complement data (target images) that can be effective in adapting model parameters to both source and target domains.

A gradient detach operation is applied between detection and context sub-networks during training to force networks to learn discriminative representations.

We argue that the conventional training with primary objective mainly leverages the information from the source-domain for maximizing likelihood and ignores the complement data in shallow layers of networks, which leads to an insufficient integration within different domains.

Thus, our proposed method is a more syncretic adaptation learning process.

We conduct comprehensive experiments on seven datasets, the results demonstrate that our method performs favorably better than the state-of-the-art methods by a large margin.

For instance, from Cityscapes to FoggyCityscapes, we achieve 37.9% mAP, outperforming the previous art Strong-Weak by 3.6%.

In real world scenarios, generic object detection always faces severe challenges from variations in viewpoint, background, object appearance, illumination, occlusion conditions, scene change, etc.

These unavoidable factors make object detection in domain-shift circumstance becoming a challenging and new rising research topic in the recent years.

Also, domain change is a widely-recognized, intractable problem that urgently needs to break through in reality of detection tasks, like video surveillance, autonomous driving, etc. (see Figure 2 ).

Revisiting Domain-Shift Object Detection.

Common approaches for tackling domain-shift object detection are mainly in two directions: (i) training supervised model then fine-tuning on the target domain; or (ii) unsupervised cross-domain representation learning.

The former requires additional instance-level annotations on target data, which is fairly laborious, expensive and time-consuming.

So most approaches focus on the latter one but still have some challenges.

The first challenge is that the representations of source and target domain data should be embedded into a common space for matching the object, such as the hidden feature space (Saito et al., 2019; Chen et al., 2018) , input space Cai et al., 2019) or both of them (Kim et al., 2019b) .

The second is that a feature alignment/matching operation or mechanism for source/target domains should be further defined, such as subspace alignment (Raj et al., 2015) , H-divergence and adversarial learning (Chen et al., 2018) , MRL (Kim et al., 2019b) , Strong-Weak alignment (Saito et al., 2019) , etc.

In general, our SCL is also a learning-based alignment method across domains with an end-to-end framework.

(a) Non-adapted (b) CVPR'18 (Chen et al., 2018) (c) CVPR'19 (Saito et al., 2019) (d) SCL (Ours) (e) Non-adapted (f) CVPR'18 (Chen et al., 2018) (g) CVPR'19 (Saito et al., 2019) (h) SCL (Ours) Figure 1: Visualization of features from PASCAL to Clipart (first row) and from Cityscapes to FoggyCityscapes (second row) by t-SNE (Maaten & Hinton, 2008) .

Red indicates the source examples and blue is the target one.

If source and target features locate in the same position, it is shown as light blue.

All models are re-trained with a unified setting to ensure fair comparisons.

It can be observed that our feature embedding results are consistently much better than previous approaches on either dissimilar domains (PASCAL and Clipart) or similar domains (Cityscapes and FoggyCityscapes).

Our Key Ideas.

The goal of this paper is to introduce a simple design that is specific to convolutional neural network optimization and improves its training on tasks that adapt on discrepant domains.

Unsupervised domain adaptation for recognition has been widely studied by a large body of previous literature (Ganin et al., 2016; Long et al., 2016; Tzeng et al., 2017; Panareda Busto & Gall, 2017; Hoffman et al., 2018; Murez et al., 2018; Zhao et al., 2019; Wu et al., 2019) , our method more or less draws merits from them, like aligning source and target distributions with adversarial learning (domain-invariant alignment).

However, object detection is a technically different problem from classification, since we would like to focus more on the object of interests (local regions).

Figure 2: Illustration of domain-shift object detection in autonomous driving scenario.

Images are from INIT dataset (Shen et al., 2019) .

Some recent work (Zhu et al., 2019) has proposed to conduct alignment only on local regions so that to improve the efficiency of model learning.

While this operation may cause a deficiency of critical information from context.

Inspired by multi-feature/strong-weak alignment (Saito et al., 2019; Zhang et al., 2018; He & Zhang, 2019) which proposed to align corresponding local-region on shallow layers with small respective field (RF) and align imagelevel features on deep layers with large RF, we extend this idea by studying the stacked complementary objectives and their potential combinations for domain adaptive circumstance.

We observe that domain adaptive object detection is supported dramatically by the deep supervision, however, the diverse supervisions should be applied in a controlled manner, including the cut-in locations, loss types, orders, updating strategy, etc., which is one of the contributions of this paper.

Furthermore, our experiments show that even with the existing objectives, after elaborating the different combinations and training strategy, our method can obtain competitive results.

By pluging-in a new sub-network that learns the context features independently with gradient detach updating strategy in a hierarchical manner, we obtain the best results on several domain adaptive object detection benchmarks.

The Relation to Complement Objective Training (Chen et al., 2019) and Deep Supervision (Lee et al., 2015) .

COL (Chen et al., 2019) proposed to involve additional function that complements the primary objective, and updated the parameters alternately with primary and complement objectives.

Specifically, cross entropy is used as the primary objective H p :

where y i ∈ {0, 1} D is the label of the i-th sample in one-hot representation andŷ i ∈ [0, 1] D is the predicted probabilities.

Th complement entropy H c is defined in COT (Chen et al., 2019) as the average of sample-wise entropies over complement classes in a mini-batch:

where H is the entropy function.ŷ c is the predicted probabilities of complement classes c. The training process is that: for each iteration of training, 1) update parameters by H p first; then 2) update parameters by H c .

In contrast, we don't use the alternate strategy but update the parameters simultaneously using gradient detach strategy with primary and complement objectives.

Since we aim to let the network enable to adapt on both source and target domain data and meanwhile enabling to distinguish objects from them, thus our complement objective design is quite different from COT.

We will describe with details in Section 2.

In essence, our method is more likely to be the deeply supervised formulation (Lee et al., 2015) that backpropagation of error now proceeds not only from the final layer but also simultaneously from our intermediate complementary outputs.

While DSN is basically proposed to alleviate "vanishing" gradient problem, here we focus on how to adopt these auxiliary losses to promote to mix two different domains through domain classifiers for detection.

Interestingly, we observe that diverse objectives can lead to better generalization for network adaptation.

Motivated by this, we propose Stacked Complementary Losses (SCL), a simple yet effective approach for domain-shift object detection.

Our SCL is fairly easy and straight-forward to implement, but can achieve remarkable performance.

We conjecture that previous approaches that focus on conducting domain alignment on high-level layers only (Chen et al., 2018) cannot fully adapt shallow layer parameters to both source and target domains (even local alignment is applied (Saito et al., 2019) ) which restricts the ability of model learning.

Also, gradient detach is a critical part of learning with our complementary losses.

We further visualize the features obtained by non-adapted model, DA (Chen et al., 2018) , Strong-Weak (Saito et al., 2019) and ours, features are from the last layer of backbone before feeding into the Region Proposal Network (RPN).

As shown in Figure 1 , it is obvious that the target features obtained by our model are more compactly matched with the source domain than any other models.

Contributions.

Our contributions in this paper are three-fold.

• We propose an end-to-end learnable framework that adopts complementary losses for domain adaptive object detection.

We study the deep supervisions in this task with a controlled manner.

Our method allows information from source and target domains to be integrated seamlessly.

• We propose a gradient detach learning strategy to enable complementary losses to learn a better representation and boost the performance.

We also provide extensive ablation studies to empirically verify the effectiveness of each component in our framework design.

• To the best of our knowledge, this is a pioneer work to investigate the influence of diverse loss functions and gradient detach for domain adaptive object detection.

Thus, this work gives very good intuition and practical guidance with multi-objective learning for domain adaptive object detection.

More remarkably, our method achieves the highest accuracy on several domain adaptive or cross-domain object detection benchmarks, which are new records on this task.

Following the common formulation of domain adaptive object detection, we define a source domain S where annotated bound-box is available, and a target domain T where only the image can be used in training process without any labels.

Our purpose is to train a robust detector that can adapt well to both source and target domain data, i.e., we aim to learn a domain-invariant feature representation that works well for detection across two different domains.

As shown in Figure 3 , we focus on the complement objective learning and let S = {(x

i is the corresponding bounding box and category labels for sample x i }.

We define a recursive function for layers k = 1, 2, . . .

, K where we cut in complementary losses:

whereΘ k is the feature map produced at layer k, F is the function to generate features at layer k and Z k is input at layer k. We formulate the complement loss of domain classifier k as follows:

where

k denote feature maps from source and target domains respectively.

Following (Chen et al., 2018; Saito et al., 2019) , we also adopt gradient reverse layer (GRL) (Ganin & Lempitsky, 2015) to enable adversarial training where a GRL layer is placed between the domain classifier and the detection backbone network.

During backpropagation, GRL will reverse the gradient that passes through from domain classifier to detection network.

For our instance-context alignment loss L ILoss , we take the instance-level representation and context vector as inputs.

The instance-level vectors are from RoI layer that each vector focuses on the representation of local object only.

The context vector is from our proposed sub-network that combine hierarchical global features.

We concatenate instance features with same context vector.

Since context information is fairly different from objects, joint training detection and context networks will mix the critical information from each part, here we proposed a better solution that uses detach strategy to update the gradients.

We will introduce it with details in the next section.

Aligning instance and context representation simultaneously can help to alleviate the variances of object appearance, part deformation, object size, etc.

in instance vector and illumination, scene, etc.

in context vector.

We define d i as the domain label of i-th training image where d i = 1 for the source and d i = 0 for the target, so the instance-context alignment loss can be further formulated as:

where N s and N t denote the numbers of source and target examples.

P (i,j) is the output probabilities of the instance-context domain classifier for the j-th region proposal in the i-th image.

So our total SCL objective L SCL can be written as:

In this section, we introduce a simple detach strategy which prevents the flow of gradients from context sub-network through the detection backbone path.

We find this can help to obtain more discriminative context and we show empirical evidence (see Figure 6 ) that this path carries information with diversity and hence gradients from this path getting suppressed is superior for such task.

As aforementioned, we define a sub-network to generate the context information from early layers of detection backbone.

Intuitively, instance and context will focus on perceptually different parts of an image, so the representations from either of them should also be discrepant.

However, if we train with the conventional process, the companion sub-network will be updated jointly with the detection backbone, which may lead to an indistinguishable behavior from these two parts.

To this end, in this paper we propose to suppress gradients during backpropagation and force the representation of context sub-network to be dissimilar to the detection network, as shown in Algorithm 1.

To our best knowledge, this may be the first work to show the effectiveness of gradient detach that can help to learn better context representation for domain adaptive object detection.

Although the detach-based method has been adopted in a few work (Arpit et al., 2019) for better optimization on sequential tasks, our design and motivation are quite different from it.

The details of our context sub-network architecture are illustrated in Appendix A.

3.

Update detection net by detection and complementary objectives: L det +L SCL

Our framework is based on the Faster RCNN (Ren et al., 2015) , including the Region Proposal Network (RPN) and other modules.

The objective of the detection loss is summarized as:

where L cls is the classification loss and L reg is the bounding-box regression loss.

To train the whole model using SGD, the overall objective function in the model is:

where λ is the trade-off coefficient between detection loss and our complementary loss.

R denotes the RPN and other modules in Faster RCNN.

Following (Chen et al., 2018; Saito et al., 2019) , we feed one labeled source image and one unlabeled target one in each mini-batch during training.

Datasets.

We evaluate our approach in three different domain shift scenarios: (1) Similar Domains; (2) Discrepant Domains; and (3) From Synthetic to Real Images.

All experiments are conducted on seven domain shift datasets: Cityscapes (Cordts et al., 2016) to FoggyCityscapes , Cityscapes to KITTI (Geiger et al., 2012) , KITTI to Cityscapes, INIT Dataset (Shen et al., 2019) , PASCAL (Everingham et al., 2010) to Clipart (Inoue et al., 2018) , PASCAL to Watercolor (Inoue et al., 2018) , GTA (Sim 10K) (Johnson-Roberson et al., 2016) to Cityscapes.

Implementation Details.

In all experiments, we resize the shorter side of the image to 600 following (Ren et al., 2015; Saito et al., 2019) with ROI-align (He et al., 2017) .

We train the model with SGD optimizer and the initial learning rate is set to 10 −3 , then divided by 10 after every 50,000 iterations.

Unless otherwise stated, we set λ as 1.0 and γ as 5.0, and we use K = 3 in our experiments (the analysis of hyper-parameter K is shown in Table 7 ).

We report mean average precision (mAP) with an IoU threshold of 0.5 for evaluation.

Since there are few pioneer works for exploring the combination of different losses for domain adaptive object detection, here we conduct extensive ablation study for this part to find the best collocation of our SCL method.

We follow some objective design from DA and Weak-Strong (Chen et al., 2018; Saito et al., 2019) which provides guidance for us to utilize these losses.

Cross-entropy (CE) Loss.

CE loss measures the performance of a classification model whose output is a probability value.

It increases as the predicted probability diverges from the actual label:

where p c ∈ [0, 1] is the predicted probability observation of c class.

y c is the c class label.

Least-squares (LS) Loss.

Following (Saito et al., 2019) , we adopt LS loss to stabilize the training of the domain classifier for aligning low-level features.

The loss is designed to align each receptive field of features with the other domain.

The least-squares loss is formulated as:

where D Θ (s) wh denotes the output of the domain classifier in each location (w, h).

Focal Loss (FL).

Focal loss L FL (Lin et al., 2017 ) is adopted to ignore easy-to-classify examples and focus on those hard-to-classify ones during training: -adapted)) 30.2 53.5 DA (Chen et al., 2018) 38.5 64.1 DA (Our impl.) (Chen et al., 2018) 35.6 70.8 SW (Our impl.) (Saito et al., 2019) 37.9 71.0 Ours 41.9 72.7 The results are summarized in Table 1 .

We present several combinations of four complementary objectives with their loss names and performance.

We observe that "LS-CE-F L-F L" obtains the best accuracy with Context and Detach.

It indicates that LS can only be placed on the low-level features (rich spatial information and poor semantic information) and F L should be in the high-level locations (weak spatial information and strong semantic information).

For the middle location, CE will be a good choice.

If you use LS for the middle/high-level features or use F L on the low-level features, it will confuse the network to learn hierarchical semantic outputs, so that ILoss+detach will lose effectiveness under that circumstance.

This verifies that domain adaptive object detection is supported by deep supervision, however, the diverse supervisions should be applied in a controlled manner.

Furthermore, our proposed method performed much better than baseline Strong-Weak (Saito et al., 2019) (37.9% vs.34.3%) and other state-of-the-arts.

Between Cityspaces and KITTI.

In this part, we focus on studying adaptation between two real and similar domains, as we take KITTI and Cityscapes as our training and testing data.

Following (Chen et al., 2018) , we use KITTI training set which contains 7,481 images.

We conduct experiments on both adaptation directions K → C and C → K and evaluate our method using AP of car as in DA.

As shown in Table 2 , our proposed method performed much better than the baseline and other stateof-the-art methods.

Since Strong-Weak (Saito et al., 2019) didn't provide the results on this dataset, we re-implement it and obtain 37.9% AP on K→C and 71.0% AP on C→K. Our method is 4% higher than the former and 1.7% higher than latter.

If comparing to the non-adapted results (source only), our method outperforms it with a huge margin (about 10% and 20% higher, respectively).

INIT Dataset.

INIT Dataset (Shen et al., 2019) contains 132,201 images for training and 23,328 images for testing.

There are four domains: sunny, night, rainy and cloudy, and three instance categories, including: car, person, speed limited sign.

This dataset is first proposed for the instance-level image-to-image translation task, here we use it for the domain adaptive object detection purpose.

Our results are shown in Table 3 .

Following (Shen et al., 2019) , we conduct experiments on three domain pairs: sunny→night (s2n), sunny→rainy (s2r) and sunny→cloudy (s2c).

Since the training images in rainy domain are much fewer than sunny, for s2r experiment we randomly sample the training data in sunny set with the same number of rainy set and then train the detector.

It can be observed that our method is consistently better than the baseline method.

We don't provide the results of s2c (faster) because we found that cloudy images are too similar to sunny in this dataset (nearly the same), thus the non-adapted result is very close to the adapted methods.

In this section, we focus on the dissimilar domains, i.e., adaptation from real images to cartoon/artistic.

Following (Saito et al., 2019) , we use PASCAL VOC dataset (2007+2012 training and validation combination for training) as the source data and the Clipart or Watercolor (Inoue et al., 2018) as the target data.

The backbone network is ImageNet pre-trained ResNet-101.

PASCAL to Clipart.

Clipart dataset contains 1,000 images in total, with the same 20 categories as in PASCAL VOC.

As shown in Table 4 , our proposed SCL outperforms all baselines.

In addition, we observe that replacing F L with CE loss on instance-context classifier can further improve the performance from 40.6% to 41.5%.

More ablation results are shown in our Appendix B.2 (Table 10) .

PASCAL to WaterColor.

Watercolor dataset contains 6 categories in common with PASCAL VOC and has totally 2,000 images (1,000 images are used for training and 1,000 test images for evaluation).

Results are summarized in Table 5 , our SCL consistently outperforms other state-of-the-arts.

Sim10K to Cityscapes.

Sim 10k dataset (Johnson-Roberson et al., 2016) contains 10,000 images for training which are generated by the gaming engine Grand Theft Auto (GTA).

Following (Chen et al., 2018; Saito et al., 2019) , we use Cityscapes as target domain and evaluate our models on Car class.

Our result is shown in Table 6 , which consistently outperforms the baselines.

Hyper-parameter K. Table 7 shows the results for sensitivity of hyper-parameter K in Figure 3 .

This parameter controls the number of SCL losses and context branches.

It can be observed that the proposed method performs best when K = 3 on all three datasets.

Parameter Sensitivity on λ and γ.

Figure 4 shows the results for parameter sensitivity of λ and γ in Eq. 8 and Eq. 11.

λ is the trade-off parameter between SCL and detection objectives and γ controls the strength of hard samples in Focal Loss.

We conduct experiments on two adaptations: Cityscapes → FoggyCityscapes (blue) and Sim10K → Cityscapes (red).

On Cityscapes → FoggyCityscapes, we achieve the best performance when λ = 1.0 and γ = 5.0 and the best accuracy is 37.9%.

On Sim10K → Cityscapes, the best result is obtained when λ = 0.1, γ = 2.0.

indicate values from high to low.

It can be observed that w/ detach training, our models can learn more discriminative representation between object areas and background (context).

Analysis of IoU Threshold.

The IoU threshold is an important indicator to reflect the quality of detection, and a higher threshold means better coverage with ground-truth.

In our previous experiments, we use 0.5 as a threshold suggested by many literature (Ren et al., 2015; Chen et al., 2018) .

In order to explore the influence of IoU threshold with performance, we plot the performance vs. IoU on three datasets.

As shown in Figure 5 , our method is consistently better than the baselines on different threshold by a large margin (in most cases).

Why Gradient Detach Can Help Our Model?

To further explore why gradient detach can help to improve performance vastly and what our model really learned, we visualize the heatmaps on both source and target images from our models w/o and w/ detach training.

As shown in Figure 6 , the visualization is plotted with feature maps after Conv B3 in Figure 3 .

We can observe that the object areas and context from detach-trained models have stronger contrast than w/o detach model (red and blue areas).

This indicates that detach-based model can learn more discriminative features from the target object and context.

More visualizations are shown in Appendix C (Figure 8 ).

Detection Visualization.

Figure 10 shows several qualitative comparisons of detection examples on three test sets with DA (Chen et al., 2018) , Strong-Weak (Saito et al., 2019) and our SCL models.

Our method detects more small and blurry objects in dense scene (FoggyCityscapes) and suppresses more false positives (Clipart and Watercolor) than the other two baselines. (Chen et al., 2018) , Strong-Weak (Saito et al., 2019) and our proposed SCL on three datasets.

For each group, the first row is the result of DA, the second row is from Strong-Weak and the last row is ours.

We show detections with the scores higher than a threshold (0.3 for FoggyCityscapes and 0.5 for other two).

In this paper, we have addressed unsupervised domain adaptive object detection through stacked complementary losses.

One of our key contributions is gradient detach training, enabled by suppressing gradients flowing back to the detection backbone.

In addition, we proposed to use multiple complementary losses for better optimization.

We conduct extensive experiments and ablation studies to verify the effectiveness of each component that we proposed.

Our experimental results outperform the state-of-the-art approaches by a large margin on a variety of benchmarks.

Our future work will focus on exploring the domain-shift detection from scratch, i.e., without the pre-trained models like DSOD (Shen et al., 2017) , to avoid involving bias from the pre-trained dataset.

A CONTEXT NETWORK Our context networks are shown in Table 8 .

We use three branches (forward networks) to deliver the context information and each branch generates a 128-dimension feature vector from the corresponding backbone layers of SCL.

Then we naively concatenate them and obtain the final context feature with a 384-dimension vector.

In this section, we show the adaptation results on source domains in Table 11 , 12, 13 and 14.

Surprisingly, we observe that the best-trained models (on target domains) are not performing best on the source data, e.g., from PASCAL VOC to WaterColor, DA (Chen et al., 2018) obtained the highest results on source domain images (although the gaps with Strong-Weak and ours are marginal).

We conjecture that the adaptation process for target domains will affect the learning and performing on source domains, even we have used the bounding box ground-truth on source data for training.

We will investigate it more thoroughly in our future work and we think the community may also need to rethink whether evaluating on source domain should be a metric for domain adaptive object detection, since it can help to understand the behavior of models on both source and target images.

We provide the detailed results of parameter sensitivity on λ and γ in Table 15 and 16 with the adaptation of from Cityscapes to FoggyCityscapes and from Sim10K to Cityscapes.

Figure 9 , the gradient detach-based models can adapt source and target images to a similar distribution better than w/o detach models.

<|TLDR|>

@highlight

We introduce a new gradient detach based complementary objective training strategy for domain adaptive object detection.