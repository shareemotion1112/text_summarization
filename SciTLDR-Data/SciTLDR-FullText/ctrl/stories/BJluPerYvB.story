Deep neural networks with millions of parameters may suffer from poor generalizations due to overfitting.

To mitigate the issue, we propose a new regularization method that penalizes the predictive distribution between similar samples.

In particular, we distill the predictive distribution between different samples of the same label and augmented samples of the same source during training.

In other words, we regularize the dark knowledge (i.e., the knowledge on wrong predictions) of a single network, i.e., a self-knowledge distillation technique, to force it output more meaningful predictions.

We demonstrate the effectiveness of the proposed method  via  experiments  on  various  image  classification  tasks:  it  improves  not only the generalization ability, but also the calibration accuracy of modern neural networks.

Deep neural networks (DNNs) have achieved state-of-the-art performance on many machine learning applications, e.g., computer vision (He et al., 2016) , natural language processing (Devlin et al., 2019) , and reinforcement learning (Silver et al., 2016) .

As the scale of training dataset increases, the size of DNNs (i.e., the number of parameters) also scales up to handle such a large dataset efficiently.

However, networks with millions of parameters may incur overfitting and suffer from poor generalizations (Pereyra et al., 2017; .

To address the issue, many regularization strategies have been investigated in the literature: early stopping, L 1 /L 2 -regularization (Nowlan & Hinton, 1992) , dropout (Srivastava et al., 2014) , batch normalization (Sergey Ioffe, 2015) and data augmentation (Cubuk et al., 2019) Regularizing the predictive or output distribution of DNNs can be effective because it contains the most succinct knowledge of the model.

On this line, several strategies such as entropy maximization (Pereyra et al., 2017) and angular-margin based methods (Chen et al., 2018; Zhang et al., 2019) have been proposed in the literature.

They can be also influential to solve related problems, e.g., network calibration (Guo et al., 2017) , detection of out-of-distribution samples (Lee et al., 2018) and exploration of the agent in reinforcement learning (Haarnoja et al., 2018) .

In this paper, we focus on developing a new output regularizer for deep models utilizing the concept of dark knowledge (Hinton et al., 2015) , i.e., the knowledge on wrong predictions made by DNN.

Its importance has been first evidenced by the so-called knowledge distillation and investigated in many following works (Romero et al., 2015; Zagoruyko & Komodakis, 2017; Srinivas & Fleuret, 2018; Ahn et al., 2019) .

While the related works (Furlanello et al., 2018; Hessam Bagherinezhad & Farhadi, 2018) use the knowledge distillation (KD; Hinton et al. 2015) to transfer the dark knowledge learned by a teacher network to a student network, we regularize the dark knowledge itself during training a single network, i.e., self-knowledge distillation.

Specifically, we propose a new regularization technique, coined class-wise self-knowledge distillation (CS-KD) that matches or distills the predictive distribution of DNNs between different samples of the same label (class-wise regularization) and augmented samples of the same source (sample-wise regularization) as shown in Figure 1 .

One can expect that the proposed regularization method forces DNNs to produce similar wrong predictions if samples are of the same class, while the conventional cross-entropy loss does not consider such consistency on the wrong predictions.

We demonstrate the effectiveness of our regularization method using deep convolutional neural networks, such as ResNet (He et al., 2016) and DenseNet (Huang et al., 2017) trained for image classification tasks on various datasets including CIFAR-100 (Krizhevsky et al., 2009) , TinyImageNet 1 , CUB-200-2011 (Wah et al., 2011) , Stanford Dogs (Khosla et al., 2011) , and MIT67 (Quattoni & Torralba, 2009 ) datasets.

We compare or combine our method with prior regularizers.

In our experiments, the top-1 error rates of our method are consistently smaller than those of prior output regularization methods such as angular-margin based methods (Chen et al., 2018; Zhang et al., 2019) and entropy regularization (Dubey et al., 2018; Pereyra et al., 2017) .

In particular, the gain tends to be larger in overall for the top-5 error rates and the expected calibration errors (Guo et al., 2017) , which confirms that our method indeed makes predictive distributions more meaningful.

Moreover, we investigate a variant of our method by combining it with other types of regularization method for boosting performance, such as the mixup regularization (Zhang et al., 2018) and the original KD method.

We improve the top-1 error rate of mixup from 37.09% to 31.95% and that of KD from 39.32% to 35.36% under ResNet (He et al., 2016) trained by the CUB-200-2011 dataset.

Our method is very simple to use, and would enjoy a broader usage in the future.

In this section, we introduce a new regularization technique, named class-wise self-knowledge distillation (CS-KD).

Throughout this paper, we focus on fully-supervised or classification tasks, and denote x ∈ X as an input and y ∈ Y = {1, ..., C} as its ground-truth label.

Suppose that a softmax classifier is used to model a posterior distribution, i.e., given the input x, the predictive distribution is as follows:

, where f = [f i ] denotes the logit-vector of DNN, parameterized by θ and T > 0 is the temperature scaling parameter.

We first consider matching the predictive distributions on samples of the same class, which distills their dark knowledge into the model itself.

To this end, we propose a class-wise regularization loss that enforces consistent predictive distributions in the same class.

Formally, given input x and another randomly sampled input x having the same label y, it is defined as follows:

where KL denotes the Kullback-Leibler (KL) divergence and θ is a fixed copy of the parameters θ.

As suggested by (Takeru Miyato & Ishii, 2018) , the gradient is not propagated through θ to avoid Algorithm 1 Class-wise self-knowledge distillation (CS-KD)

Initialize parameters θ.

while θ has not converged do for (x, y) in a sampled batch do g θ ← 0 Get another sample x randomly which has the same label y from the training set.

Generate x aug , x aug using data augmentation methods.

Compute gradient:

Update parameters θ using gradients g θ . end while the model collapsing issue.

Similar to the knowledge distillation method (KD) by Hinton et al. (2015) , L cls matches two predictions.

While the original KD matches predictions of a sample from two networks, we do predictions of different samples from a single network.

Namely, our method performs self-knowledge distillation.

In addition to enforcing the intra-class consistency of predictive distributions, we apply this idea to the single-sample scenario by augmenting the input data.

For a given training sample x, the proposed sample-wise regularization loss L sam is defined as follows:

where x aug is an augmented input that is modified by some data augmentation methods, e.g., resizing, rotating, random cropping (Krizhevsky et al., 2009; Simonyan & Zisserman, 2015; , cutout (DeVries & Taylor, 2017) , and auto-augmentation (Cubuk et al., 2019) .

In our experiments, we use standard augmentation methods for ImageNet (i.e., flipping and random sized cropping) because they make training more stable.

In summary, the total training loss L tot is defined as a weighted sum of the two regularization terms with cross-entropy loss as follows:

where λ cls and λ sam are balancing weights for each regularization, respectively.

Note that the first term is the cross-entropy loss of softmax outputs with temperature T = 1.

In other words, we not only train the true label, but also regularize the wrong labels.

The full training procedure with the proposed loss L tot is summarized in Algorithm 1.

Datasets.

To demonstrate our method under general situations of data diversity, we consider various image classification tasks including conventional classification and fine-grained classification tasks.

We use CIFAR-100 (Krizhevsky et al., 2009) and TinyImageNet 2 datasets for conventional classification tasks, and CUB-200-2011 (Wah et al., 2011) , Stanford Dogs (Khosla et al., 2011) , and MIT67 (Quattoni & Torralba, 2009 ) datasets for fine-grained classification tasks.

Note that fine-grained image classification tasks have visually similar classes and consist of fewer training samples per class compared to conventional classification tasks.

We sample 10% of the training dataset randomly as a validation set for CIFAR-100 and TinyImageNet and report the test accuracy based on the validation accuracy.

For the fine-grained datasets, we report the best validation accuracy.

Network architecture.

We consider two state-of-the-art convolutional neural network architectures: ResNet (He et al., 2016) and DenseNet (Huang et al., 2017) .

We use standard ResNet-18 with 64 filters and DenseNet-121 with growth rate of 32 for image size 224 × 224.

For CIFAR-100 and TinyImageNet, we modify the first convolutional layer 3 with kernel size 3 × 3, strides 1 and padding 1, instead of the kernel size 7 × 7, strides 2 and padding 3, for image size 32 × 32.

Evaluation metric.

For evaluation, we measure the following metrics:

• Top-1 / 5 error rate.

Top-k error rate is the fraction of test samples for which the correct label is amongst the top-k confidences.

We measured top-1 and top-5 error rates to evaluate the generalization performance of the models.

• Expected Calibration Error (ECE).

ECE (Naeini et al., 2015; Guo et al., 2017) approximates the difference in expectation between confidence and accuracy, by partitioning predictions into M equally-spaced bins and taking a weighted average of bins' difference of confidence and accuracy, i.e., ECE = • Recall at k (R@k).

Recall at k is the percentage of test samples that have at least one example from the same class in k nearest neighbors on the feature space.

To measure the distance between two samples, we use L 2 -distance between their average-pooled features in the penultimate layer.

We compare the recall at 1 scores to evaluate intra-class variations of learned features.

Hyper-parameters.

All networks are trained from scratch and optimized by stochastic gradient descent (SGD) with momentum 0.9, weight decay 0.0001 and an initial learning rate of 0.1.

The learning rate is divided by 10 after epochs 100 and 150 for all datasets and total epochs are 200.

We set batch size as 128 for conventional, and 32 for fine-grained classification tasks.

We use standard flips, random resized crops, 32 for conventional and 224 for fine-grained classification tasks, overall experiments.

Furthermore, we set T = 4, λ cls = 1 for all experiments and λ sam = 1 for experiments on fine-grained classification tasks, and λ sam = 0 on conventional classification tasks.

To compute expected calibration error (ECE), we set the number of bins M as 20.

Baselines.

We compare our method with prior regularization methods such as the state-of-the-art angular-margin based methods (Zhang et al., 2019; Chen et al., 2018) and entropy regularization (Dubey et al., 2018; Pereyra et al., 2017) .

They also regularize predictive distributions as like ours.

• AdaCos (Zhang et al., 2019) .

4 AdaCos dynamically scales the cosine similarities between training samples and corresponding class center vectors to maximize angular-margin.

• Virtual-softmax (Chen et al., 2018) .

Virtual-softmax injects an additional virtual class to maximize angular-margin.

• Maximum-entropy (Dubey et al., 2018; Pereyra et al., 2017) .

Maximum-entropy is a typical entropy regularization, which maximizes the entropy of the predictive distribution.

Note that AdaCos and Virtual-softmax regularize the predictive or output distribution of DNN to learn feature representation by reducing intra-class variations and enlarging inter-class margins.

Comparison with output regularization methods.

We measure the top-1 error rates of the proposed method (denoted by CS-KD) by comparing with Virtual-softmax, AdaCos, and Maximumentropy on various image classification tasks.

Table 1 shows that CS-KD outperforms other baselines consistently.

In particular, CS-KD improves the top-1 error rate of cross-entropy loss from 46.00% to 33.50% in the CUB-200-2011 dataset, while the top-1 error rates of other baselines are even worse than the cross-entropy loss (e.g., AdaCos in the CIFAR-100, Virtual-softmax in the MIT67, and Maximum-entropy in the TinyImageNet and the MIT67 under DenseNet).

The results imply that our method is more effective and stable than other baselines.

Compatibility with other types of regularization methods.

We investigate orthogonal usage with other types of regularization methods such as mixup (Zhang et al., 2018) and knowledge distillation (KD).

Mixup utilizes convex combinations of input pairs and corresponding label pairs for training.

We combine our method with mixup regularization by applying the class-wise regularization loss L cls to mixed inputs and mixed labels, instead of standard inputs and labels.

Table 2 shows the effectiveness of our method combined with mixup regularization.

Interestingly, this simple idea significantly improves the performances of fine-grained classification tasks.

In particular, our method improves the top-1 error rate of mixup regularization from 37.09% to 31.95%, where the top-1 error rate of cross-entropy loss is 46.00% in the CUB-200-2011.

KD regularizes predictive distributions of student network to learn the dark knowledge of a teacher network.

We combine our method with KD to learn dark knowledge from the teacher and itself simultaneously.

Table 3 shows that the top-1 error rate under using our method solely is close to that of KD, although ours do not use additional teacher networks.

Besides, learning knowledge from a teacher network improves the top-1 error rate of our method from 39.32% to 35.36% in the CUB-200-2011 dataset.

The results show a wide applicability of our method, compatible to use with other regularization methods.

One can expect that our method forces DNNs to produce meaningful predictions by reducing the intra-class variations.

To verify this, we analyze feature embedding and various evaluation metrics, including the top-1, top-5 error, expected calibration error (Guo et al., 2017) and R@1.

In Figure  2 , we visualize feature embedding of the penultimate layer from ResNet-18 trained with various regularization techniques by t-SNE (Maaten & Hinton, 2008) in the CIFAR-100 dataset.

One can note that intra-class variations are significantly decreased by our method (Figure 2f ), while Virtualsoftmax ( Figure 2b ) and AdaCos (Figure 2c ) only reduce the angular-margin.

We also provide quantitative analysis on the feature embedding by measuring the R@1 values, which are related to intra-class variations.

Note that the larger value of R@1 means the more reduced intra-class variations on the feature embedding (Wengang Zhou, 2017).

As shown in Table 4 , R@1 values can be significantly improved when ResNet-18 is trained with our methods.

In particular, R@1 of our method is 59.22% in the CUB-200-2011 dataset, while R@1 of Virtual-softmax and Adacos are 55.56% and 54.86%, respectively.

Moreover, Table 4 shows the top-5 error rates of our method significantly outperform other regularization methods.

Figure 3 and Table 4 show that our method enhances model calibration significantly, which also confirm that ours forces DNNs to produce more meaningful predictions.

Table 4 : Top-1 / 5 error, ECE, and Recall at 1 rates (%) of ResNet-18.

The arrow on the right side of the evaluation metric indicates ascending or descending order of the value.

We reported the mean and standard deviation over 3 runs with different random seed, and the best results are indicated in bold.

& Fienberg, 1983; Niculescu-Mizil & Caruana, 2005) which show accuracy as a function of confidence, for ResNet-18 trianed on CIFAR-100 using (a) Cross-entropy, (b) Virtual-softmax, (c) AdaCos, and (d) Maximum-entropy.

All methods are compared with our proposed method, CS-KD.

Regularization techniques.

Numerous techniques have been introduced to prevent overfitting of neural networks, including early stopping, weight decay, dropout (Srivastava et al., 2014) , and batch normalization (Sergey Ioffe, 2015) .

Alternatively, regularization methods for the output distribution also have been explored: Szegedy et al. (2016) showed that label-smoothing, which is a mixture of the ground-truth and the uniform distribution, improves generalization of neural networks.

Similarly, Pereyra et al. (2017) proposed penalizing low entropy output distributions, which improves exploration in reinforcement learning and supervised learning.

Zhang et al. (2018) proposed a powerful data augmentation method called mixup, which works as a regularizer that can be utilized with smaller weight decay.

We remark that our method enjoys orthogonal usage with the prior methods, i.e., our methods can be combined with prior methods to further improve the generalization performance.

Knowledge distillation.

Knowledge distillation (Hinton et al., 2015) is an effective learning method to transfer the knowledge from a powerful teacher model to a student.

This pioneering work showed that one can use softmax with temperature scaling to match soft targets for transferring dark knowledge, which contains the information of non-target labels.

There are numerous follow-up studies to distill knowledge in the aforementioned teacher-student framework.

FitNets (Romero et al., 2015) tried to learn features of a thin deep network using a shallow one with linear transform.

Similarly, Zagoruyko & Komodakis (2017) introduced a transfer method that matches attention maps of the intermediate features, and Ahn et al. (2019) tried to maximize the mutual information between intermediate layers of teacher and student for enhanced performance.

Srinivas & Fleuret (2018) proposed a loss function for matching Jacobian of the networks output instead of the feature itself.

We remark that our method and knowledge distillation (Hinton et al., 2015) have a similar component, i.e., using a soft target distribution, but our method utilizes the soft target distribution from itself.

We also remark that joint usage of our method and the prior knowledge distillation methods is effective.

Margin-based softmax losses.

There have been recent efforts toward boosting the recognition performances via enlarging inter-class margins and reducing intra-class variation.

Several approaches utilized metric-based methods that measure similarities between features using Euclidean distances, such as triplet (Weinberger & Saul, 2009 ) and contrastive loss (Chopra et al., 2005) .

To make the model extract discriminative features, center loss and range loss (Xiao Zhang & Qiao, 2017) were proposed to minimize distances between samples belong to the same class.

COCO loss (Liu et al., 2017b) and NormFace (Feng Wang & Yuille, 2017) optimized cosine similarities, by utilizing reformulations of softmax loss and metric learning with feature normalization.

Similarly, Yutong Zheng & Savvides (2018) applied ring loss for soft normalization which uses a convex norm constraint.

More recently, angular-margin based losses were proposed for further improvement.

Lsoftmax (Liu et al., 2016) and A-softmax (Liu et al., 2017a) combined angular margin constraints with softmax loss to encourage the model to generate more discriminative features.

CosFace , AM-softmax (Feng Wang & Cheng, 2018) and ArcFace (Deng et al., 2019) introduced angular margins for a similar purpose, by reformulating softmax loss.

Different from L-Softmax and A-Softmax, Virtual-softmax (Chen et al., 2018) encourages a large margin among classes via injecting additional virtual negative class.

In this paper, we discover a simple regularization method to enhance generalization performance of deep neural networks.

We propose two regularization terms which penalizes the predictive distribution between different samples of the same label and augmented samples of the same source by minimizing the Kullback-Leibler divergence.

We remark that our ideas regularize the dark knowledge (i.e., the knowledge on wrong predictions) itself and encourage the model to produce more meaningful predictions.

Moreover, we demonstrate that our proposed method can be useful for the generalization and calibration of neural networks.

We think that the proposed regularization techniques would enjoy a broader range of applications, e.g., deep reinforcement learning (Haarnoja et al., 2018) and detection of out-of-distribution samples (Lee et al., 2018) .

<|TLDR|>

@highlight

We propose a new regularization technique based on the knowledge distillation.