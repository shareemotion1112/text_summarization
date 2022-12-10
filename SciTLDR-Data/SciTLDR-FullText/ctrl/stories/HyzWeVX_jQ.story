We propose an end-to-end framework for training domain specific models (DSMs) to obtain both high accuracy and computational efficiency for object detection tasks.

DSMs are trained with distillation and focus on achieving high accuracy at a limited domain (e.g. fixed view of an intersection).

We argue that DSMs can capture essential features well even with a small model size, enabling higher accuracy and efficiency than traditional techniques.

In addition, we improve the training efficiency by reducing the dataset size by culling easy to classify images from the training set.

For the limited domain, we observed that compact DSMs significantly surpass the accuracy of COCO trained models of the same size.

By training on a compact dataset, we show that with an accuracy drop of only 3.6%, the training time can be reduced by 93%.

The framework is based on knowledge distillation BID0 [4][5] but targets to reduce the accuracy gap 23 between student and teacher models by training the student using a restricted class of domain specific 24 images.

Since such training may be conducted on edge-devices, we improve the training efficiency 25 by culling easy-to-classify images with small accuracy penalty.

This paper's contribution is summarized below.

• We propose an end-to-end framework for training domain specific models (DSMs) to mit-28 igate the tradeoff between object-detection accuracy and computational efficiency.

To the 29 best of our knowledge, this is the first successful demonstration of training DSMs for object 30 detection tasks.

• By training resnet18-based Faster-RCNN DSMs, we observed a 19.7% accuracy (relative DISPLAYFORM0 compute L train (i) from label(i) and pred(i).

Collect Detection ← DSM.predict(image) Figure 1 : Object detection results of the test image, before and after domain specific training.• Since edge devices will have limited resources, we propose culling the training dataset to 35 significantly reduce the computation resource required for the training.

Only training data 36 that has high utility in training is added.

This filtering allows us to reduce training time by

93% with an accuracy loss of only 3.6%.

DSM framework to train compact models with dataset constructed by domain-specific data.

As illustrated in Algorithm 1, our DSM framework consists of preparation of the data and training of the DSM.

A large challenge when deploying models in surveillance is preparing the training data 48 since manually labelling frames in videos is cumbersome.

To overcome this, we label the dataset 49 used to train the DSM by using the predictions of a much larger teacher model with higher accuracy 50 and treating these predictions as ground truth labels.

Furthermore, we compare the prediction on 51 image x i made by the teacher to that of the DSM; we determine whether to store the x i and label

Teacher.predict(x i ) in our compiled dataset Ω. After the training set is compiled, it is used to train 53 the DSM.

Training a object detection model can take hours even with a GPU and can be challenging for 55 applications requiring frequent retraining.

We exploit the fact that when the DSM is pretrained on 56 large-scale general dataset, it can already provide good predictions for a large chunk of the domain-57 specific data.

This procedure develops a compact dataset Ω that is only composed of data that the 58 DSM finds inconsistent with the prediction made by the teacher model.

Keeping data x j that both 59 the DSM and teacher detections are consistent is computationally redundant because it does not 60 contribute to gradient signals.

We define L train to quantify the consistency between teacher and 61 DSM: images are for training and the later 3600 images for testing.

DISPLAYFORM0

Results.

As shown in table 1, we first train our res18 DSM using the full N train = 3600 training 78 images for 10 epochs using stochastic-gradient descent with a learn rate of 10 we aim to train the models with only the domain specific data.

We show on

<|TLDR|>

@highlight

High object-detection accuracy can be obtained by training domain specific compact models and the training can be very short.