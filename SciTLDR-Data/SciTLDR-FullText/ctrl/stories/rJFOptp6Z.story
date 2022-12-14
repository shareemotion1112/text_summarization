Knowledge distillation is a potential solution for model compression.

The idea is to make a small student network imitate the target of a large teacher network, then the student network can be competitive to the teacher one.

Most previous studies focus on model distillation in the classification task, where they propose different architectures and initializations for the student network.

However, only the classification task is not enough, and other related tasks such as regression and retrieval are barely considered.

To solve the problem, in this paper, we take face recognition as a breaking point and propose model distillation with knowledge transfer from face classification to alignment and verification.

By selecting appropriate initializations and targets in the knowledge transfer, the distillation can be easier in non-classification tasks.

Experiments on the CelebA and CASIA-WebFace datasets demonstrate that the student network can be competitive to the teacher one in alignment and verification, and even surpasses the teacher network under specific compression rates.

In addition, to achieve stronger knowledge transfer, we also use a common initialization trick to improve the distillation performance of classification.

Evaluations on the CASIA-Webface and large-scale MS-Celeb-1M datasets show the effectiveness of this simple trick.

Since the emergence of Alexnet BID12 , larger and deeper networks have shown to be more powerful BID22 .

However, as the network going larger and deeper, it becomes difficult to use it in mobile devices.

Therefore, model compression has become necessary in compressing the large network into a small one.

In recent years, many compression methods have been proposed, including knowledge distillation BID0 BID8 BID19 , weight quantization BID4 BID17 , weight pruning BID6 BID24 and weight decomposition BID2 BID16 .

In this paper, we focus on the knowledge distillation, which is a potential approach for model compression.

In knowledge distillation, there is usually a large teacher network and a small student one, and the objective is to make the student network competitive to the teacher one by learning specific targets of the teacher network.

Previous studies mainly consider the selection of targets in the classification task, e.g., hidden layers BID15 , logits BID0 BID25 BID20 or soft predictions BID8 BID19 .

However, only the distillation of the classification task is not enough, and some common tasks such as regression and retrieval should also be considered.

In this paper, we take face recognition as a breaking point that we start with the knowledge distillation in face classification, and consider the distillation on two domain-similar tasks, including face alignment and verification.

The objective of face alignment is to locate the key-point locations in each image; while in face verification, we have to determine if two images belong to the same identity.

For distillation on non-classification tasks, one intuitive idea is to adopt a similar method as in face classification that trains teacher and student networks from scratch.

In this way, the distillation on all tasks will be independent, and this is a possible solution.

However, this independence cannot give the best distillation performance.

There has been strong evidence that in object detection BID18 , object segmentation BID3 and image retrieval BID30 , they all used the pretrained classification model(on ImageNet) as initialization to boost performance.

This success comes from the fact that their domains are similar, which makes them transfer a lot from low-level to high-level representation BID29 .

Similarly, face classification, alignment and verification also share the similar domain, thus we propose to transfer the distilled knowledge of classification by taking its teacher and student networks to initialize corresponding networks in alignment and verification.

Another problem in knowledge transfer is what targets should be used for distillation?

In face classification, the knowledge is distilled from the teacher network by learning its soft-prediction, which has been proved to work well BID8 BID19 .

However, in face alignment BID27 and verification BID27 , they have additional task-specific targets.

As a result, selecting the classification or task-specific target for distillation remains a problem.

One intuitive idea is to measure the relevance of objectives between non-classification and classification tasks.

For example, it is not obvious to see the relation between face classification and alignment, but the classification can help a lot in verification.

Therefore, it seems reasonable that if the tasks are highly related, the classification target is preferred, or the task-specific target is better.

Inspired by the above thoughts, in this paper, we propose the model distillation in face alignment and verification by transferring the distilled knowledge from face classification.

With appropriate selection of initializations and targets, we show that the distillation performance of alignment and verification on the CelebA and CASIA-WebFace BID28 datasets can be largely improved, and the student network can even exceed the teacher network under specific compression rates.

This knowledge transfer is our main contribution.

In addition, we realize that in the proposed method, the knowledge transfer depends on the distillation of classification, thus we use a common initialization trick to further boost the distillation performance of classification.

Evaluations on the CASIA-WebFace and large-scale MS-Celeb-1M BID5 datasets show that this simple trick can give the best distillation results in the classification task.

In this part, we introduce some previous studies on knowledge distillation.

Particularly, all the following studies focus on the classification task.

BID1 propose to generate synthetic data by a teacher network, then a student network is trained with the data to mimic the identity labels.

However, BID0 observe that these labels have lost the uncertainties of the teacher network, thus they propose to regress the logits (pre-softmax activations) BID8 .

Besides, they prefer the student network to be deep, which is good to mimic complex functions.

To better learn the function, BID25 observe the student network should not only be deep, but also be convolutional, and they get competitive performance to the teacher network in CIFAR BID11 ).

Most methods need multiple teacher networks for better distillation, but this will take a long training and inference time BID20 .

To address the issue, BID20 propose noise-based regularization that can simulate the logits of multiple teacher networks.

However, BID15 observe the values of these logits are unconstrained, and the high dimensionality will cause fitting problem.

As a result, they use hidden layers as they capture as much information as the logits but are more compact.

All these methods only use the targets of the teacher network in distillation, while if the target is not confident, the training will be difficult.

To solve the problem, BID8 propose a multi-task approach which uses identity labels and the target of the teacher network jointly.

Particularly, they use the post-softmax activation with temperature smoothing as the target, which can better represent the label distribution.

One problem is that student networks are mostly trained from scratch.

Given the fact that initialization is important, BID19 propose to initialize the shallow layers of the student network by regressing the mid-level target of the teacher network.

However, these studies only consider knowledge distillation in classification, which largely limits its application in model compression.

In this paper, we consider face recognition as a breaking point and extend knowledge distillation to non-classification tasks.

Due to the proposed knowledge transfer depends on the distillation of classification, improving the classification itself is necessary.

In this part, we first review the idea of distillation for classification, then introduce how to boost it by a simple initialization trick.

We adopt the distillation framework in BID8 , which is summarized as follows.

Let T and S be the teacher and student network, and their post-softmax predictions to be P T = softmax(a T ) and P S =softmax(a S ), where a T and a S are the pre-softmax predictions, also called the logits BID0 .

However, the post-softmax predictions have lost some relative uncertainties that are more informative, thus a temperature parameter ?? is used to smooth predictions P T and P S to be P ?? T and P ?? S , which are denoted as soft predictions: DISPLAYFORM0 Then, consider P ?? T as the target, knowledge distillation optimizes the following loss function DISPLAYFORM1 wherein W cls S is the parameter of the student network, and y cls is the identity label.

For simplicity, we omit min and the number of samples N , and denote the upper right symbol cls as the classification task.

In addition, H(, ) is the cross-entropy, thus the first term is the softmax loss, while the second one is the cross-entropy between the soft predictions of the teacher and student network, with ?? balancing between the two terms.

This multi-task training is advantageous because the target P ?? T cannot be guaranteed to be always correct, and if the target is not confident, the identity label y cls will take over the training of the student work.

It is noticed that in Eqn.(2), the student network is trained from scratch.

As demonstrated in Ba & Caruana (2014) that deeper student networks are better for distillation, initialization thus has become very important BID9 BID10 .

Based on the evidence, Fitnet BID19 first initializes the shallow layers of the student network by regressing the mid-level target of the teacher network, then it follows Eqn.(2) for distillation.

However, only initializing the shallow layers is still difficult to learn high-level representation, which is generated by deep layers.

Furthermore, BID29 shows that the network transferability increases as tasks become more similar.

In our case, the initialization and distillation are both classification tasks with exactly the same data and identity labels, thus more deep layers should be initialized for higher transferability, and we use a simple trick to achieve this.

To obtain an initial student network, we train it with softmax loss: DISPLAYFORM0 wherein the lower right symbol S 0 denotes the initialization for student network S. In this way, the student network is fully initialized.

Then, we modify Eqn.

FORMULA1 as , and the two entropy terms remain the same.

This process is shown in FIG0 .

It can be seen that the only difference with Eqn.(2) is that the student network is trained with the full initialization, and this simple trick has been commonly used, e.g., initializing the VGG-16 model based on a fully pretrained model BID22 .

We later show that this trick can get promising improvements over Eqn.(2) and Fitnet BID19 .

DISPLAYFORM1

In this part, we show how to transfer the distilled knowledge from face classification to face alignment and verification.

The knowledge transfer consists of two steps: transfer initialization and target selection, which are elaborated as follows.

The first step of the transfer is transfer initialization.

The motivation is based on the evidence that in detection, segmentation and retrieval, they have used the pretrained classification model (on ImageNet) as initialization to boost performance BID18 BID3 BID30 .

The availability of this idea comes from the fact that they share the similar domain, which makes them transfer easily from low-level to high-level representation BID29 .

Similarly, the domains of face classification, alignment and verification are also similar, thus we can transfer the distilled knowledge in the same way.

For simplicity, we denote the parameters of teacher and student networks in face classification as W

Based on the initialization, the second step is to select appropriate targets in the teacher network for distillation.

One problem is that non-classification tasks have their own task-specific targets, but given the additional soft predictions P ?? T , which one should we use?

To be clear, we first propose the general distillation for non-classification tasks as follows: DISPLAYFORM0 where W S and y denote the task-specific network parameter and label respectively.

?? (W S , y) is the task-specific loss function, and ?? (K S , K T ) is the task-specific distillation term with the targets selected as K T and K S in teacher and student networks.

Besides, ?? and ?? are the balancing terms between classification and non-classification tasks.

In Eqn.(5), the above problem has become how to set ?? and ?? for a given non-classification task.

In the following two parts, we will give some discussions on two tasks: face alignment and verification.

The task of face alignment is to locate the key-point locations for each image.

Without loss of generality, there is no any identity label, but only the keypoint locations for each image.

Face alignment is usually considered as a regression problem BID27 , thus we train the teacher network with optimizing the Euclidean loss: DISPLAYFORM0 wherein R T is the regression prediction of the teacher network and y ali is the regression label.

In distillation, except for the available soft predictions P ?? T (classification target), another one is the task-specific target that can be the hidden layer K T BID15 , and it satisfies R T = f c (K T ) with f c being a fully-connected mapping.

In face classification, the key in distinguishing different identities is the appearance around the key-points such as shape and color, but the difference of key-point locations for different identities is small.

As a result, face identity is not the main influencing factor for these locations, but it is still related as different identities may have slightly different locations.

Instead, pose and viewpoint variations have a much larger influence.

Therefore, in face alignment, the hidden layer is preferred for distillation, which gives Eqn.(7) by setting ?? < ??, as shown in FIG0 .

DISPLAYFORM1

The task of face verification is to determine if two images belong to the same identity.

In verification, triplet loss BID21 ) is a widely used metric learning method BID21 , and we take it for model distillation.

Without loss of generality, we have the same identity labels as in classification, then the teacher network can be trained as DISPLAYFORM0 where K a T , K p T and K n T are the hidden layers for the anchor, positive and negative samples respectively, i.e., a and p have the same identity, while a and n come from different identities.

Besides, ?? controls the margin between positive and negative pairs.

Similar to face alignment, we consider the hidden layer K T and soft prediction P ?? T as two possible targets in distillation.

In fact, classification focuses on the difference of identities, i.e. the inter-class relation, and this relation can help a lot in telling if two image have the same identity.

As a result, classification can be beneficial to boost the performance of verification.

Therefore, in face verification, the soft prediction is preferred for distillation, which gives the following loss function by setting ?? > ??, as shown in FIG0 .

DISPLAYFORM1 Particularly, some studies show the benefits by using additional softmax loss in Eqn.(8).

For comparison, we also add the softmax loss H(P T , y cls ) and H(P S , y cls ) in Eqn.

FORMULA7 and Eqn.(9) respectively for further enhancement.

As analyzed above, ?? and ?? should be set differently in the distillation of different tasks.

The key is to measure the relevance of objectives between classification and non-classification tasks.

For a given task, if it is highly related to the classification task, then ?? > ?? is necessary, or ?? < ?? should be set.

Though this rule cannot be theoretically guaranteed, it provides some guidelines to use knowledge distillation in more non-classification tasks.

In this section, we give the experimental evaluation of the proposed method.

We first introduce the experimental setup in detail, and then show the results of knowledge distillation in the tasks of face classification, alignment and verification.

Database:

We use three popular datasets for evaluation, including CASIA-WebFace BID28 , CelebA and MS-Celeb-1M BID5 .

CASIA-WebFace contains 10575 people and 494414 images, while CelebA has 10177 people with 202599 images and the label of 5 key-point locations.

Compared to the previous two, MS-Celeb-1M is a large-scale dataset that contains 100K people with 8.4 million images.

In experiments, we use CASIA-WebFace and MS-Celeb-1M for classification, CelebA for alignment and CASIA-WebFace for verification.

Evaluation: In all datasets, we randomly split them into 80% training and 20% testing samples.

In classification, we evaluate the top1 accuracy based on if the identity of the maximum prediction matches the correct identity label BID12 , and the results on the LFW BID13 database (6000 pairs) are also reported by computing the percentage of how many pairs are correctly verified.

In alignment, the Normalized Root Mean Squared Error (NRMSE) is used to evaluate alignment BID27 ; while in verification, we compute the Euclidean distance between each pair in testing samples, and the top1 accuracy is reported based on if a test sample and its nearest sample belong to the same identity.

Particularly, LFW is not used in verification because 6000 pairs are not enough to see the difference obviously for different methods.

Teacher and Student: To learn the large number of identities, we use ResNet-50 as the teacher network, which is deep enough to handle our problem.

For student networks, given the fact that deep student networks are better for knowledge distillation BID0 BID25 BID19 , we remain the same depth but divide the number of convolution kernels in each layer by 2, 4 and 8, which give ResNet-50/2, ResNet-50/4 and ResNet-50/8 respectively.

Pre-processing and Training:

Given an image, we resize it to 256 ?? 256 wherein a sub-image with 224 ?? 224 is randomly cropped and flipped.

Particularly, we use no mean subtraction or image whitening, as we use batch normalization right after the input data.

In training, the batchsize is set to be 256, 64 and 128 for classification, alignment and verification respectively, and the Nesterov Accelerated Gradient(NAG) is adopted for faster convergence.

For the learning rate, if the network is trained from scratch, 0.1 is used; while if the network is initialized, 0.01 is used to continue, and 30 epochs are used in each rate.

Besides, in distillation, student networks are trained with the targets of the teacher network generated online, and the temperature ?? and margin ?? are set to be 3 and 0.4 by cross-validation.

Finally, the balancing terms ?? and ?? have many possible combinations, and we show later how to set them by an experimental trick.

(1)Scratch: student networks are not initialized; (2)P retrain: student networks are trained with the task-specific initialization; (3)Distill: student networks are initialized with W cls S ; (4)Sof t: the soft prediction P ?? T ; (5)Hidden: the hidden layer K T .

In this part, we compare the initialization trick to previous studies in classification.

Table.1 shows the comparison of different targets and initializations.

It can be observed from the first table that without any initialization, soft predictions achieve the best distillation performance, i.e., 61.27%.

Based on the best target, the second table gives the results of different initializations in distillation.

We see that our full initialization obtains the best accuracy of 75.06%, which is much higher than other methods, i.e., 10% and 5% higher than the Scratch and Fitnet BID19 .

These results show that the full initialization of student networks can give the highest transferability in classification, and also demonstrates the effectiveness of this simple trick.

Base on the best initialization and target, Table.

2 shows the distillation results of face classification on CASIA-WebFace and MS-Celeb-1M, and we have three main observations.

Firstly, the student networks trained with full initialization can obtain large improvements over the ones trained from scratch, which further demonstrates the effectiveness of the initialization trick in large-scale cases.

Secondly, some student networks can be competitive to the teacher network or even exceed the teacher one by a large margin, e.g. in the CASIA-WebFace database, ResNet-50/4 can be competitive to the teacher network, while ResNet-50/2 is about 3% higher than the teacher one in the top1 accuracy.

Finally, in the large-scale MS-Celeb-1M, student networks cannot exceed the teacher network but only be competitive, which shows that the knowledge distillation is still challenging for a large number of identities.

In this part, we give the evaluation of distillation in face alignment.

Table.

3 shows the distillation results of ResNet-50/8 with different initializations and targets on CelebA. The reason we only consider ResNet-50/8 is that face alignment is a relatively easy problem and most studies use shallow and small networks, thus a large compression rate is necessary for the deep ResNet-50.

One important thing is how to set ?? and ?? in Eqn.(7).

As there are many possible combinations, we use a simple trick by measuring their individual influence and discard the target with the negative impact by setting ?? = 0 or ?? = 0; while if they both have positive impacts, ?? > 0, ?? > 0 should be set to keep both targets in distillation.

As shown in Table.

3, when the initializations of P retrain and Distill are used, ?? = 1, ?? = 0(soft prediction) always decreases performance, while ?? = 0, ?? = 1(hidden layer) gets consistent improvements, which implies that the hidden layer is preferred in the distillation of face alignment.

It can be observed in Table.

3 that Distill has a lower error rate than P retrain, which shows that W cls S has higher transferability on high-level representation than the task-specific initialization.

Besides, the highest distillation performance 3.21% is obtained with Distill and ?? = 0, ?? = 1, and it can be competitive to the one of the teacher network(3.02%).

In this part, we give the evaluation of distillation in face verification.

Similar to alignment, we select ?? and ?? in the same way.

Table.

4 shows the verification results of different initializations and targets on CASIA-WebFace, and the results are given by Eqn.(9).

It can be observed that no matter which student network or initialization is used, ?? = 0, ?? = 1(hidden layer) always decreases the baseline performance, while ?? = 1, ?? = 0(soft prediction) remains almost the same.

As a result, we discard the hidden layer and only use the soft prediction.

One interesting observation in Table.

4 is that ?? = 0, ?? = 0 always obtains the best performance, and the targets do not work at all.

One possible reason is that the target in classification is not confident, i.e., the top1 accuracy of ResNet-50 in classification is only 88.61%.

To improve the classification ability, we add additional softmax loss in Eqn.

FORMULA7 and Eqn.(9), and the results are shown in Table.

5.

We see that the accuracy of ResNet-50/2 and ResNet-50/4 has obtained remarkable improvements, which implies that the classification targets that are not confident enough cannot help the distillation.

But with the additional softmax loss, the student work can adjust the learning by identity labels.

As a result, ?? = 1, ?? = 0 can get the best performance, which is even much higher than the teacher network, e.g., 79.96% of ResNet-50/2 with Distill and ?? = 1, ?? = 0.

In this paper, we take face recognition as a breaking point, and propose the knowledge distillation on two non-classification tasks, including face alignment and verification.

We extend the previous distillation framework by transferring the distilled knowledge from face classification to face alignment and verification.

By selecting appropriate initializations and targets, the distillation on non-classification tasks can be easier.

Besides, we also give some guidelines for target selection on non-classification tasks, and we hope these guidelines can be helpful for more tasks.

Experiments on the datasets of CASIA-WebFace, CelebA and large-scale MS-Celeb-1M have demonstrated the effectiveness of the proposed method, which gives the student networks that can be competitive or exceed the teacher network under appropriate compression rates.

In addition, we use a common initialization trick to further improve the distillation performance of classification, and this can boost the distillation on non-classification tasks.

Experiments on CASIA-WebFace have demonstrated the effectiveness of this simple trick.

<|TLDR|>

@highlight

We take face recognition as a breaking point and propose model distillation with knowledge transfer from face classification to alignment and verification

@highlight

This paper proposes to transfer the classifier from the model for face classification to the task of alignment and verification.

@highlight

The manuscript presents experiments on distilling knowledge from a face classification model to student models for face alignment and verification.