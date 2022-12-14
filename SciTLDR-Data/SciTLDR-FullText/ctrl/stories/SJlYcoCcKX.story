Although deep neural networks show their extraordinary power in various tasks, they are not feasible for deploying such large models on embedded systems due to high computational cost and storage space limitation.

The recent work knowledge distillation (KD) aims at transferring model knowledge from a well-trained teacher model to a small and fast student model which can significantly help extending the usage of large deep neural networks on portable platform.

In this paper, we show that, by properly defining the neuron manifold of deep neuron network (DNN), we can significantly improve the performance of student DNN networks through approximating neuron manifold of powerful teacher network.

To make this, we propose several novel methods for learning neuron manifold from DNN model.

Empowered with neuron manifold knowledge, our experiments show the great improvement across a variety of DNN architectures and training data.

Compared with other KD methods, our Neuron Manifold Transfer (NMT) has best transfer ability of the learned features.

In recent years, deep neural networks become more and more popular in computer vision and neural language processing.

A well-trained learning model shows its power on tasks such as image classification, object detection, pattern recognizing, live stream analyzing, etc.

We also have the promise that given enough data, deeper and wider neural networks can achieve better performance than the shallow networks BID0 ).

However, these larger but well-trained networks also bring in high computational cost, and leave large amount of memory footprints which make these models very hard to travel and reproduce BID7 ).

Due to this drawback, a massive amount of trainable data gathered by small devices such as mobiles, cameras, smart sensors, etc. is unable to be utilized in the local environment which can cause time-sensitive prediction delay and other impractical issues.

To address the above issues, recently, there are extensive works proposed to mitigate the problem of model compression to reduce the computational burden on embedded system.

Back to the date 2006, Bucilu?? et al. first proposed to train a neural network to mimic the output of a complex and large ensemble.

This method uses ensemble to label the unlabeled data and trains the neural network with the data labeled by the ensemble, thus mimicking the function which learned by the ensemble and achieves similar accuracy.

Based on the idea of (Bucilu??,Geoffrey et al.) originally introduced a student-teacher paradigm in transferring the knowledge from a deeper and wider network (teacher) to a shallow network (student).

They call this student-teacher paradigm as knowledge distillation (KD).

By properly defining the knowledge of teacher as softened softmax (soft target), the student learns to mimic soft target distribution for each class.

Thanks to Hinton's pioneer work, a series of subsequent works have sprung up by utilizing different forms of knowledge.

BID22 regard the spatial attention maps of a convolution neural network as network knowledge.

However, an implicit assumption that they make is that the absolute value of a hidden neuron activation can be used as an indication about the importance of that neuron w.r.t.

the specific input which limited their application only fit for image classification tasks.

Another assumption that has been widely used is from BID0 that deeper networks always learn better representation.

Based on that, FitNets BID13 ) tries to learn a thin deep network using a shallow one with more parameters.

They believe that the convolution regressor is the network knowledge which can inherit from the teacher to its student.

In 2017, researcher from TuSimple ( TuSimple et al. (2015) , using softmax as knowledge, the student network mimics teacher's softmax and minimize the loss on soft target.

Middle one, mentioned in BID22 , named as attention transfer, an additional regularizer has been applied known as attention map, student needs to learn the attention map and soft target.

The right part is our neuron manifold transfer, where we take neuron manifold as knowledge, and it reduces the computational and space cost.(2015)) introduces two new definitions of network knowledge, BID8 takes the advantage of Maximum Mean Discrepancy (MMD) to minimize the distance metric in probability distributions, and they regard network knowledge as class distribution.

BID4 propose to transfer the cross sample similarities between the student and teacher to improve the performance of transferred networks.

We notice that in the DarkRank the network knowledge is defined as cross sample similarities.

By reviewing extensive KD works, we notice that the key point in knowledge transfer is how we define the network knowledge, and in fact, a well-defined network knowledge can greatly improve the performance of the distilled network.

Moreover, in our perspective, a perfect knowledge transfer method must allow us to transfer one neural network architecture into another, while preserving other generalization.

A perfect transfer method, however, would use little observations to train, optimally use the limited samples at its disposal.

Unfortunately, to our best knowledge, due to the complexity of large DNN, simply mimicking the teacher logit or a part of teacher features properties is far away to be benefited.

Therefore, if we look back and consider the essence of DNN training, we notice that, another point of view to look at the distribution of neuron features is the shape of that feature.

Here, the shape of neuron features include the actual value and the relative distance between two features.

That is, during the process of knowledge transfer, student network not only learns the numerical information but also inherits the geometric properties.

Therefore, in order to track the change of large DNN feature knowledge, a manifold approximation technique is vying for our attention.

Manifold learning has been widely used in Topological Data Analysis (TDA) BID3 ), and this technique can project the high dimensional data to a lower dimensional manifold and preserving both numerical feature properties and geometric properties.

In previous works, feature mapping causes computational resource waste, and class distribution matching is limited the usage.

However, using the neuron manifold information, we not only collect as much as possible feature properties which can greatly represent feature, but also preserve inter-neuron characteristics (spatial relation).

Since manifold projection can greatly reduce the dimension of teacher feature, we compress the teacher model and make student model more reliable.

To summarize, the contributions of this paper are three folds:??? We introduce a new type of knowledge that is the d-dimensional smooth sub-manifold of teacher feature maps called neuron manifold.??? We formalize manifold space in feature map, and implement in details.??? We test our proposed method on various metric learning tasks.

Our method can significantly improve the performance of student networks.

And it can be applied jointly with existing methods for a better transferring performance.

We test our method on MNIST, CIFAR-10, and CIFAR-100 datasets and show that our Neuron Manifold Transfer (NMT) improves the students performance notably.

In recent years, there are extensive works proposed to explore the knowledge transfer problems.

As we mentioned the core idea behind knowledge transfer is properly defining the network knowledge and existing efforts can be broadly categorized into following 3 types, each of which is described below.

Soft target knowledge was first proposed by Geoffrey et al., because of their extraordinary pioneer work, network knowledge distillation (KD) becomes more and more popular and is being more practical.

The trick in Hinton's work is that the network knowledge is defined as softened outputs of the teacher network, therefore, student network only need to mimic teacher's softened output to receive a good performance.

The intuition behind is the one-hot labels (hard target) aim to project the samples in each class into one single point in the label space, while the softened labels project the samples into a continuous distribution.

Inspired by Hinton's work, Sau & Balasubramanian (2016) use a perturbation logit to create a multiple teachers environment.

By using noise-based regularizer, in their experiment, they show the reduction of intra-class variation.

A proper noise level can help the student to achieve better performance.

However, soft target knowledge's drawback is also very obvious: it only fits for some classification tasks and relies on the number of classes.

For example, in a binary classification problem, KD could hardly improve the performance since almost no additional supervision could be provided.

Network feature knowledge is proposed to tackle the drawbacks of KD by transferring intermediate features.

Romero et al. proposed FitNet to transfer a wide and shallow network to a thin and deep one.

They think that deep convolution nets are significantly more accurate than shallow convolution models, when given the same parameter budget.

In their adventurous work, FitNet makes the student mimic the full feature maps of the teacher.

The computational cost, therefore, can not be ignored and such settings are too strict since the capacities of the teacher and student may differ greatly.

In certain circumstances, FitNet may adversely affect the performance and convergence BID8 ).

Then Zagoruyko & Komodakis proposed Attention Transfer (AT) by define the network knowledge as spatial attention map of input images.

To reduce the computational cost, they introduce an activation-based mapping function which compresses a 3D tensor to a 2D spatial attention map.

Similarly, they make a questionable assumption that the absolute value of a hidden neuron activation can be used as an indication about the importance of that neuron.

Another network feature knowledge is defined by Huang & Wang, instead of mimic softened output, Zehao aim to minimize the distribution of softened output via Maximum Mean Discrepancy method.

A similar work applied in vision field is proposed by BID20 , they call their knowledge as Flow of Solution Procedure (FSP) which computes the Gram matrix of features from two different layers.

Network feature knowledge provides more supervision than simple KD method.

Jacobian knowledge is quite different from the above two classic type approaches.

Soft target knowledge and network feature knowledge are defined as layer wise consideration, however, Jacobian knowledge generates the full picture of DNN and transfer the knowledge from function perspective.

Sobolev training BID5 ) proposed Jacobian-based regularizer on Sobolev spaces to supervise the higher order derivatives of teacher and student network.

The subsequent work BID17 deals with the problem of knowledge transfer using a first-order approximation of the neural network.

Despite their novelty in knowledge transfer, Jacobian based knowledge is very hard in practical use because large DNNs are complex.

Although the above approaches show their potential power in knowledge distillation, we still think the idea of knowledge distillation should be revisited due to its complexity in both structure and computational property and the fact that deeper networks tend to be more non-linear.

In this section, we brief review the previous knowledge transfer methods.

We also introduce the notations to be used in following sections.

In practical, given a well defined deep neural network, for example, let us consider a Convolution Neural Network (CNN) and refer a teacher network as T and student network as S. FIG0 illustrates three popular knowledge transfer methods and we explain in below.

Assume we have a dataset of elements, with one such element denoted x, where each element has a corresponding one-hot class label: denote the one-hot vector corresponding to x by y. Given x, we have a trained teacher network t = T (x) that outputs the corresponding logits, denoted by t; likewise we have a student network that outputs logits s = S(x).

To perform knowledge distillation we train the student network to minimize the following loss function (averaged across all data items): DISPLAYFORM0 where ??(??) is the softmax function, T is a temperature parameter and ?? is a parameter controlling the ratio of the two terms.

The first term L CE (p, q) is a standard cross entropy loss penalizing the student network for incorrect classifications.

The second term is minimized if the student network produces outputs similar to that of the teacher network.

The idea is from that the outputs of the teacher network contain additional, beneficial information beyond just a class prediction.

Consider a teacher network T has layers i = 1, 2, ?? ?? ?? , L and the corresponding layers in the student network.

At each chosen layer i of the teacher network, we collect the spatial map of the activations for channel j into the vector a DISPLAYFORM0 where ?? is a hyper-parameter.

Zagoruyko & Komodakis (2017) recommended using f (A DISPLAYFORM1 where N Ai is the number of channels at layer i. In other words, the loss targeted the difference in the spatial map of average squared activation, where each spatial map is normalized by the overall activation norm.

In this section, we will illustrate how to approximate neuron manifold from CNN features.

Manifold approximation has been widely used to to avoid the curse of dimensionality, frequently encountered in Big Data analysis BID15 ).

There was a vast development in the field of linear and nonlinear dimension reduction.

This techniques assume that the scattered input data is lying on a lower dimensional manifold, therefore, they aim to harvest this geometrical connection between the points, in order to reduce the effective number of parameters needed to be optimized BID16 ).Determining the neuron manifold of a given feature is not a trivial task.

As we mentioned in section 1, an efficient knowledge can greatly affect the performance of transfer learning.

To our best knowledge, in the recent year, most of manifold approximation are learning based, which is not applied on our case due to high computational costs.

Therefore, a simple but useful manifold approximation method is needed.

Inspired by BID16 , we can approximate the neuron manifold by using Moving Least Squares Projection(MLSP) mentioned in BID15 with O(n) run-time complexity.

In order to use MLSP, the given features should meet some criterion.

Let assume the feature points {f i } I i=1 ??? F are bounded, that is, there exist a distance h such that h = min DISPLAYFORM0 And we also assume the feature points are compact with density ??.

That is DISPLAYFORM1 whereB(m, r) is a closed ball of radius r and centered at m such that ||f i ??? f j || ??? h?? for 1 ??? i ??? j ??? I and ?? > 0.

Once we make the above assumption, according to Theorem 2.3 in BID15 , we can minimize the error bound of our approximation to ||M d ??? m|| < k ?? h m+1 , where M d is our approximated manifold, and m is ground truth sub-manifold of R n , k is some adjust factor.

Now we can approximate the neuron manifold by using Moving Least Squares Projection(MLSP).

Let M ??? R d be the neuron manifold we would like to find, and let {f i } I i=1 be the feature points situated near M. To find the neuron manifold of given feature, two following steps are required, first, we need to find a local d-dimensional affine space H = H(f i ) as our local coordinate system (Algorithm 1 in BID16 ), second, by utilizing the local coordinate defined by H, we project the feature points onto the coordinate system H and minimize the weighted least squares error to retrieve the target points as our neuron manifold features.

Determine local d-dimensional affine space, given the feature map of certain CNN layer, let us say F n ??? R C??W ??H and all feature points f i ??? F n , we would like to find a local d-dimensional affine space H = H(f i ) with a point q = q(f i ) on H, such that the following constrained problem is minimized: DISPLAYFORM2 where d(f i , H) is the Euclidean distance between the point f i and the subspace H. We find the affine space H by an iterative procedure and we initialize the basis vectors of H 1 randomly.

The reason we doing this is because the second term on right side of equation FORMULA7 ??(??) is a weight function such that lim x?????? ??(x) ??? 0.

That is when the feature f i is far away to the affine space H, the influence of this feature to the H is less.

Therefore, this local hyperplane H is passing through the features as much as possible.

Neuron manifold projection.

Then we define the neuron manifold projection function as p : DISPLAYFORM3 So that the approximation of p is performed by a weighted least squares vector valued polynomial function m(x) = (m 1 (f ), ?? ?? ?? m n (f ))T .

Let x i be the orthogonal projections of f i onto H(f i ).

We formulated m(x) as follow: DISPLAYFORM4 ??(s) is a non-negative weight function (rapidly decreasing as s ??? 0), and || ?? || is the Euclidean norm.

Once we solve the above equation, we collect the projected point and mark it as our manifold feature point.

Neuron Manifold Transfer Given a output feature map of a layer in CNN by F ??? R C??W ??H which consists of C feature planes with spatial dimensions H ?? W .

And for each hyperplane feature F, it has a sample set of a lower dimensional manifold M d where d is the intrinsic dimension of M and d C ?? W ?? H. Let F T and F S be the feature maps from certain layers of the teacher and student network, and M F T and M F S be lower dimensional manifold of teacher and student feature map respectively.

Without loss of generality, we assume F T and F S have the same spatial dimensions.

The feature maps can be interpolated if their dimensions do not match.

We can compute teacher network neuron manifold M F T from feature dimension F ??? R C??W ??H by solving equation FORMULA5 .

Then, we train the student network parameters from some selected feature as well as the regressor parameters by minimizing the following loss function: DISPLAYFORM5 where we use a very classical CNN model mentioned in BID7 .

It only has two hidden layers with 1200 rectified linear hidden units refer as Hinton-1200.

We set this model as a pre-trained teacher and a smaller net that has same network architecture but only 800 rectified linear hidden units refer as Hinton-800 are used to be the student model.

On CIFAR datasets, a middle level deep neuron network, ResNet-34 is used to be teacher, and as a result, we transfer the knowledge to a shallow and fast net known as ResNet-18.

We also adopt a pre-activation version of VGG-19 with batch normalization from TorchVision TorchVision (2018) as teacher and create a modified version of AlexNet BID11 who has 8 hidden layers as student.

DISPLAYFORM6 To further validate the effectiveness of our method, we compare our NMT with several state-ofthe-art knowledge transfer methods, including traditional KD Geoffrey et al. (2015) , attention transfer BID22 , HintNet Romero et al. (2014) and Neuron Selectivity Transfer BID8 .

For KD, we set the temperature for softened softmax to 4 and ?? = 16, following BID7 .

For AT, the ?? = 64 and the spatial attention mapping function is defined as sum of absolute values.

It is worth emphasizing that original AT is built on wide-residualnetworks BID21 , therefore we modified the original settings of AT to achieve same results mentioned by Zagoruyko & Komodakis.

As for our NMT, we set manifold approximation function's ??(s) as ??(s) = 1 s 2 and ?? = 22 to achieve best performance.

The number of sample points are various depend on the different network.

We will make our implementation publicly available if the paper is accepted.

We start our toy experiment on MNIST a handwritten digit recognition dataset with 10 classes (0-9) to evaluate our method.

The training set contains 50000 images and validation set contains 10000 images.

All samples are 28 ?? 28 in gray-scale images.

In fact, Hinton-1200 has good performance on MNIST that we train it within 60 epochs and reach 98.6% accuracy on top-1 and 99.9% accuracy on top-5.

Its student model Hinton-800 results show in TAB1 .

We collect 100 handwriting digit not included in original MNIST validation set.

We can clearly see that NMT still has good performance.

What need to be mentioned is that the AT can not be applied here, because AT is based on the attention map of input images, therefore, handwritten digit images with single channel leave zero information to their attention map.

To better understanding, we illustrate the Neuron Manifold Map as FIG3 .

We extract out the fist layer of Hinton-1200, and normalize all value in between 0 to 1.

We use Moving Least Squares method to approximate the true manifold of such layer.

All white dots are feature points and form hyper-ball and the big black dots highlighted indicate the selected representative feature points which can best describe both feature properties and geometric properties(relative position and distance preserved).

The results from experiments on CIFAR dataset is surprising.

CIFAR-10 and CIFAR-100 datasets consist of 50K training images and 10K testing images with 10 and 100 classes, respectively.

We take a 32 ?? 32 random crop from a zero-padded 40 ?? 40 image.

For optimization, we use ADAM Kingma & Ba (2014) with a mini-batch size of 128 on a single GPU.

We train the network in 120 epochs.

FIG4 shows the training and testing curves of all the experiments on CIFR10 and CIFAR100.

CI-FAR10 contains 10 different categories, our NMT achieve most reliable classification result.

Compared to other methods, even training epochs sufficiently large, the top-1 error not converge and being fluctuation.

One possible reason is because AT and FitNet only transfer the knowledge of feature importance, however, NMT also focuses on the relation in between the neuron and NMT transfer the knowledge of inter-class relation.

When training epochs increase, the neuron manifold changes slightly, and is more stable.

Although classical KD has relative good performance, NMT can fast converge, which means using less epochs to have an accurate result.

Another advantage NMT has is that NMT do not rely on soft target.

In FIG4 , we notice that large number of classes hurt the performance of KD.

And we also mentioned that by using NMT the knowledge transferring time remarkably reduces.

This is due to the computational cost of AT and FitNet are much higher than NMT.

FIG5 shows the accuracy on validation set when epochs increase.

One important fact that we can not neglect is that knowledge transfer aims to help the big model travel and deploy the small model on embedded system.

We would like to reduce the time of knowledge transferring process and without accuracy loss.

NMT has great work on training CIFAR10 and has best converge speed.

During the early stage, say epochs between 0 to 40, the training result varies, but the result performance is above the average.

We can clearly see the FitNet is under performance in full transfer period.

VGG training is much more challenging.

The standard VGG-19 is in linear structure, therefore, instead of transferring the neuron knowledge between each group in ResNet, we should be really careful to select the feature blocks for computing the neuron manifold.

We optimize the network using adam with a mini-batch size of 128 on 2 GPUs.

We train the network for 100 epochs.

The initial learning rate is set to 0.1, and then divided by 10 at the 25, 50 and 75 epoch, respectively.

TAB3 summaries the training result on CIFAR-100.

In this section, we mainly focus on the result of system resource usage.

FitNet would match all features between teacher and student, therefore, the matching time is the slowest and we set it as our baseline.

Although there is an overhead due to computing neuron manifold, out NMT still has x6.26 speed up comparing to standard FitNet and AT.

From the training perspective, although KD is the fast one with 143 second per epoch, it has the worst training result.

And both At and FitNet are not as effective as transfer method.

TAB4 summaries the different knowledge transfer methods kernel size.

It is very clearly that FitNet failed in this task because FitNet is trying to match all features and as a consequence it has large kernel run time.

Our NMT automatically chooses the the features to be transferred and result in an acceptable kernel run time.

Compared with AT, computing the neuron manifold introduces overhead.

In this paper, we propose a novel method for knowledge transfer and we define a new type network knowledge named neuron manifold.

By utilizing the state of art technique in Topological Data Analysis, we extract the DNN's feature properties and its geometric properties.

We test our NMT on various dataset and the results are quite promising, thus further confirming that our knowledge transfer method could indeed learn better feature representations.

They can be successfully transferred to high level vision task in the future.

We believe that our novel view will facilitate the further design of knowledge transfer methods.

In our future work, we plan to explore more applications of our NMT methods, especially in various regression problems, such as super resolution and optical flow prediction, etc.

<|TLDR|>

@highlight

A new knowledge distill method for transfer learning

@highlight

The work introduces a knowledge distillation method using the proposed neuron manifold concept. 

@highlight

Proposes a knowledge distilling method in which neural manifold is taken as the transferred knowledge.