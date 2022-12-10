Zero-Shot Learning (ZSL) is a classification task where some classes referred as unseen classes have no labeled training images.

Instead, we only have side information (or description) about seen and unseen classes, often in the form of semantic or descriptive attributes.

Lack of training images from a set of classes restricts the use of standard classification techniques and losses, including the popular cross-entropy loss.

The key step in tackling ZSL problem is bridging visual to semantic space via learning a nonlinear embedding.

A well established approach is to obtain the semantic representation of the visual information and perform classification in the semantic space.

In this paper, we propose a novel architecture of casting ZSL as a fully connected neural-network with cross-entropy loss to embed visual space to semantic space.

During training in order to introduce unseen visual information to the network, we utilize soft-labeling based on semantic similarities between seen and unseen classes.

To the best of our knowledge, such similarity based soft-labeling is not explored for cross-modal transfer and ZSL.

We evaluate the proposed model on five benchmark datasets for zero-shot learning, AwA1, AwA2, aPY, SUN and CUB datasets, and show that, despite the simplicity, our approach achieves the state-of-the-art performance in Generalized-ZSL setting on all of these datasets and outperforms the state-of-the-art for some datasets.

Supervised classifiers, specifically Deep Neural Networks, need a large number of labeled samples to perform well.

Deep learning frameworks are known to have limitations in fine-grained classification regime and detecting object categories with no labeled data Socher et al., 2013; Zhang & Koniusz, 2018) .

On the contrary, humans can recognize new classes using their previous knowledge.

This power is due to the ability of humans to transfer their prior knowledge to recognize new objects (Fu & Sigal, 2016; Lake et al., 2015) .

Zero-shot learning aims to achieve this human-like capability for learning algorithms, which naturally reduces the burden of labeling.

In zero-shot learning problem, there are no training samples available for a set of classes, referred to as unseen classes.

Instead, semantic information (in the form of visual attributes or textual features) is available for unseen classes (Lampert et al., 2009; 2014) .

Besides, we have standard supervised training data for a different set of classes, referred to as seen classes along with the semantic information of seen classes.

The key to solving zero-shot learning problem is to leverage trained classifier on seen classes to predict unseen classes by transferring knowledge analogous to humans.

Early variants of ZSL assume that during inference, samples are only from unseen classes.

Recent observations Scheirer et al., 2013; realize that such an assumption is not realistic.

Generalized ZSL (GZSL) addresses this concern and considers a more practical variant.

In GZSL there is no restriction on seen and unseen classes during inference.

We are required to discriminate between all the classes.

Clearly, GZSL is more challenging because the trained classifier is generally biased toward seen classes.

In order to create a bridge between visual space and semantic attribute space, some methods utilize embedding techniques (Palatucci et al., 2009; Romera-Paredes & Torr, 2015; Socher et al., 2013; Bucher et al., 2016; Xu et al., 2017; Zhang et al., 2017; Simonyan & Zisserman, 2014; Xian et al., 2016; Zhang & Saligrama, 2016; Al-Halah et al., 2016; Zhang & Shi, 2019; Atzmon & Chechik, 2019) and the others use semantic similarity between seen and unseen classes (Zhang & Saligrama, 2015; Mensink et al., 2014) .

Semantic similarity based models represent each unseen class as a mixture of seen classes.

While the embedding based models follow three various directions; mapping visual space to semantic space (Palatucci et al., 2009; Romera-Paredes & Torr, 2015; Socher et al., 2013; Bucher et al., 2016; Xu et al., 2017; Socher et al., 2013) , mapping semantic space to the visual space (Zhang et al., 2017; Shojaee & Baghshah, 2016; Ye & Guo, 2017) , and finding a latent space then mapping both visual and semantic space into the joint embedding space Simonyan & Zisserman, 2014; Xian et al., 2016; Zhang & Saligrama, 2016; Al-Halah et al., 2016) .

The loss functions in embedding based models have training samples only from the seen classes.

For unseen classes, we do not have any samples.

It is not difficult to see that this lack of training samples biases the learning process towards seen classes only.

One of the recently proposed techniques to address this issue is augmenting the loss function with some unsupervised regularization such as entropy minimization over the unseen classes .

Another recent methodology which follows a different perspective is deploying Generative Adversarial Network (GAN) to generate synthetic samples for unseen classes by utilizing their attribute information Zhu et al., 2018; Xian et al., 2018) .

Although generative models boost the results significantly, it is difficult to train these models.

Furthermore, the training requires generation of large number of samples followed by training on a much larger augmented data which hurts their scalability.

The two most recent state-of-the-art GZSL methods, CRnet (Zhang & Shi, 2019) and COSMO (Atzmon & Chechik, 2019) , both employ a complex mixture of experts approach.

CRnet is based on k-means clustering with an expert module on each cluster (seen class) to map semantic space to visual space.

The output of experts (cooperation modules) are integrated and finally sent to a complex loss (relation module) to make a decision.

CRnet is a multi-module (multi-network) method that needs end-to-end training with many hyperparameters.

Also COSMO is a complex gating model with three modules: a seen/unseen classifier and two expert classifiers over seen and unseen classes.

Both of these methods have many modules, and hence, several hyperparameters; architectural, and learning decisions.

A complex pipeline is susceptible to errors, for example, CRnet uses k-means clustering for training and determining the number of experts and a weak clustering will lead to bad results.

Our Contribution: We propose a simple fully connected neural network architecture with unified (both seen and unseen classes together) cross-entropy loss along with soft-labeling.

Soft-labeling is the key novelty of our approach which enables the training data from the seen classes to also train the unseen class.

We directly use attribute similarity information between the correct seen class and the unseen classes to create a soft unseen label for each training data.

As a result of soft labeling, training instances for seen classes also serve as soft training instance for the unseen class without increasing the training corpus.

This soft labeling leads to implicit supervision for the unseen classes that eliminates the need for any unsupervised regularization such as entropy loss in .

Soft-labeling along with crossentropy loss enables a simple MLP network to tackle GZSL problem.

Our proposed model, which we call Soft-labeled ZSL (SZSL), is simple (unlike GANs) and efficient (unlike visual-semantic pairwise embedding models) approach which achieves the state-of-the-art performance in Generalized-ZSL setting on all five ZSL benchmark datasets and outperforms the state-of-the-art for some of them.

In zero-shot learning problem, a set of training data on seen classes and a set of semantic information (attributes) on both seen and unseen classes are given.

The training dataset

includes n samples where x i is the visual feature vector of the i-th image and y i is the class label.

All samples in D belong to seen classes S and during training there is no sample available from unseen classes U. The total number of classes is C = |S| + |U|.

Semantic information or attributes a k ∈ R a , are given for all C classes and the collection of all attributes are represented by attribute matrix A ∈ R a×C .

In the inference phase, our objective is to predict the correct classes (either seen or unseen) of the test dataset D .

The classic ZSL setting assumes that all test samples in D belong to unseen classes U and tries to classify test samples only to unseen classes U.

While in a more realistic setting i.e. GZSL, there is no such an assumption and we aim at classifying samples in D to either seen or unseen classes S ∪ U.

3 PROPOSED METHODOLOGY 3.1 NETWORK ARCHITECTURE As Figure 1 illustrates our architecture, We map visual space to semantic space, then compute the similarity score (dot-product) between true attributes and the attribute/semantic representation of the input (x).

Finally, the similarity score is fed into a Softmax, and the probability of all classes are computed.

For the visual features as the input, in all five benchmark datasets, we use the extracted visual features by a pre-trained ResNet-101 on ImageNet provided by .

We do not fine-tune the CNN that generates the visual features unlike model in .

In this sense, our proposed model is also fast and straightforward to train.

In ZSL problem, we do not have any training instance from unseen classes, so the output nodes corresponding to unseen classes are always inactive during learning.

Standard supervised training with cross entropy loss biases the network towards seen classes only.

The true labels (hard labels) used for training only represent seen classes so the cross entropy cannot penalize unseen classes.

Moreover, the available similarity information between the seen and unseen attributed is never utilized.

We propose soft labeling based on the similarity between semantic attributes.

For each seen sample, we represent its relationship to unseen categories by obtaining semantic similarity (dot-product) using the seen class attribute and all the unseen class attributes.

In the simplest form, for every training data, we can find the nearest unseen class to the correct seen class label and assign a small probability q (partial membership or soft label) of this instance to be from the closest unseen class.

Note, each training sample only contains a label which comes from the set of seen classes.

With soft labeling, we enrich the label with partial assignments to unseen classes and as Hinton et al. (2015) shows, soft labels act as a regularizer which allows each training case to enforce much more constraint on weights.

In a more general soft labeling approach, we propose assigning a probability to all the unseen classes.

A natural choice is to transform seen-to-unseen similarities to probabilities (soft labels) shown in Equation (1).

The unseen distribution is obtained for each seen class by calculating dot-product of seen class attribute and all unseen classes attributes and squashing all these dot-product values by Softmax to acquire probabilities.

In this case, we distribute the probability q among all unseen classes based on the obtained unseen distribution.

This proposed strategy results in a soft label for each seen image during training, which as we show later helps the network to learn unseen categories.

In order to control the flatness of the unseen distribution, we utilize temperature parameter τ .

Higher temperature results in flatter distribution over unseen categories and lower temperature creates a more ragged distribution with peaks on nearest unseen classes.

A small enough temperature basically results in the nearest unseen approach.

The Impact of temperature τ on unseen distribution is depicted in Figure 3 .a for a particular seen class.

Soft labeling implicitly introduces unseen visual features into the network without generating fake unseen samples as in generative methods Zhu et al., 2018; Xian et al., 2018) .

Hence our proposed approach is able to reproduce same effect as in generative models without the need to create fake samples and train generative models that are known to be difficult to train.

Below is the formal description of temperature Softmax:

where a i is the i-th column of attribute matrix A ∈ R a×C which includes both seen and unseen class attributes:

And s i,j is the true similarity score between two classes i, j based on their attributes.

τ and q are temperature parameter and total probability assigned to unseen distribution, respectively.

Also y u i,k is the soft label (probability) of unseen class k for seen class i.

It should be noted that q is the sum of all unseen soft labels i.e. k∈U y u i,k = q.

The proposed method is a multi-class probabilistic classifier that produces a C-dimensional vector of class probabilities p for each sample x i as p( C-dimensional vector of all similarity scores of an input sample.

The predicted similarity score between semantic representation of sample x i and attribute a k isŝ i,k g w (x i ) , a k .

Each element of vector p, represents an individual class probability that can be shown below:

This Softmax as the activation function of the last layer of the network is calculated on all classes.

An established choice to train a multi-class probabilistic classifier is the cross-entropy loss which we later show naturally integrates our idea of soft labeling.

Inspired by Hinton et al. (2015) , in addition to the cross-entropy loss over soft targets, we also consider cross entropy-loss over true labels (hard labels) to improve the performance.

During training, we aim at learning the nonlinear mapping g w (.) i.e. obtaining network weights W through:

where λ and γ are regularization factors which are obtained through hyperparameter tuning, and L(x i ) is the weighted sum of cross-entropy loss over soft labels (L soft ) and cross-entropy loss over hard labels (L hard ) for each sample as shown below:

where α ∈ [0, 1] is a hyperparameter.

For better understanding, the hard-loss and soft-loss terms for each sample x i (or x for simplicity) are expanded and elaborated.

The hard-loss term is a conventional cross-entropy loss L hard (x) = − C k=1 z k log(p k ), where z k is the hard label.

Clearly, hard-loss term alone does not work in ZSL regime since it does not penalize unseen classes.

The soft-loss term is expanded to seen and unseen terms as follows:

Utilizing Equation (1)

Hence the first two terms of L soft (x) is the weighted sum of cross-entropy of seen classes and crossentropy of unseen classes.

In particular, first term penalizes and controls the relative (normalized) probabilities within all seen classes and the second term acts similarly within unseen classes.

We also require to penalize the total probability of all seen classes (1 −q) and total probability of all unseen classes (q).

This is accomplished through the last two terms of Equation (7) which is basically a binary cross entropy loss.

Intuitively soft-loss in Equation (7) works by controlling the balance within seen/unseen classes (first two terms) as well as the balance between seen and unseen classes (last two terms).

As we have shown in Equation (7), soft-loss enables the classifier to learn unseen classes by only being exposed to samples from seen classes.

Hyperparameter q acts as a trade-off coefficient between seen and unseen cross-entropy losses (Figure 2 ).

We can see that the regularizer is a weighted cross entropy on unseen class, which leverages similarity structure between attributes compared to uniform entropy function of DCN .

DCN and all prior works use uniform entropy as regularizer, which does not capitalize on the known semantic similarity information between seen and unseen class attributes.

At the inference time, our proposed SZSL method works the same as a conventional classifier, we only need to provide the test image and the network will produce class probabilities for all seen and unseen classes.

We conduct comprehensive comparison of our proposed SZSL model with the state-of-the-art methods for GZSL settings on five benchmark datasets (Table 1) .

We present the detailed description of datasets in Appendix A. Our model outperforms the state-of-the-art methods on GZSL setting for all benchmark datasets.

For the purpose of validation, we employ the validation splits provided along with the Proposed Split (PS) to perform cross-validation for hyper-parameter tuning.

The main objective of GZSL is to simultaneously improve seen samples accuracy and unseen samples accuracy i.e. imposing a trade-off between these two metrics.

As the result, the standard GZSL evaluation metric is harmonic average of seen and unseen accuracy.

This metric is chosen to encourage the network not be biased toward seen classes.

Harmonic average of accuracies is defined as

where A S and A U are seen and unseen accuracies, respectively. (Akata et al., 2015) 11.3 74.6 19.6 8.0 73.9 14.4 3.7 55.7 6.9 23.5 59.2 33.6 14.7 30.5 19.8 ConSE (Norouzi et al., 2013) 0.4 88.6 0.8 0.5 90.6 1.0 0.0 91.2 0.0 1.6 72.2 3.1 6.8 39.9 11.6 Sync 8.9 87.3 16.2 10.0 90.5 18.0 7.4 66.3 13.3 11.5 70.9 19.8 7.9 43.3 13.4 DeViSE 13.4 68.7 22.4 17.1 74.7 27.8 4.9 76.9 9.2 23.8 53.0 32.8 16.9 27.4 20.9 CMT (Socher et al., 2013) 0.9 87.6 1.8 8.7 89.0 15.9 1.4 85.2 2.8 7.2 49.8 12.6 8.1 21.8 11.8

Generative Models f-CLSWGAN (Xian et al., 2018) 57.9 61.4 59.6 ------43.7 57.7 49.7 42.6 36.6 39.4 SP-AEN (Chen et al., 2018) 23.3 90.9 37.1 ---13.7 63.4 13.7 34.7 70.6 46.6 24.9 38.6 30.3 cycle-UWGAN (Felix et al., 2018) 59.6 63.4 59.

To evaluate SZSL, we follow the popular experimental framework and the Proposed Split (PS) in for splitting classes into seen and unseen classes to compare GZSL/ZSL methods.

Utilizing PS ensures that none of the unseen classes have been used in the training of ResNet-101 on ImageNet.

The input to the model is the visual features of each image sample extracted by a pre-trained ResNet-101 (He et al., 2016) on ImageNet provided by .

The dimension of visual features is 2048.

We utilized Keras (Chollet, 2015) with TensorFlow back-end (Abadi et al., 2016 ) to implement our model.

We used proposed unseen classes for validation (3-fold CV) and added 20% of train samples (seen classes) as seen validation samples to obtain GZSL validation sets.

We crossvalidate τ ∈ [10 −2 , 10], mini-batch size ∈ {64, 128, 256, 512, 1024}, q ∈ [0, 1], α ∈ [0, 1], hidden layer size ∈ {128, 256, 512, 1024, 1500} and activation function ∈{tanh, sigmoid, hard-sigmoid, relu} to tune our model.

To obtain statistically consistent results, the reported accuracies are averaged over 5 trials (using different initialization) after tuning hyper-parameters with cross-validation.

Also we ran our experiments on a machine with 56 vCPU cores, Intel(R) Xeon(R) CPU E5-2660 v4 @ 2.00GHZ and 2 NVIDIA-Tesla P100 GPUs each with 16GB memory.

The code is provided in the supplementary material.

To demonstrate the effectiveness of SZSL model in GZSL setting, we comprehensively compare our proposed method with state-of-the-art GZSL models in Table 2 .

Since we use the standard proposed split, the published results of other GZSL models are directly comparable.

As reported in Table 2 , accuracies of our model achieves the state-of-the-art GZSL performance on all five benchmark datasets and outperforms the state-of-the-art on AwA2 and aPY datasets.

It is exciting and motivating while our architecture is much simpler compared to recently proposed CRnet and COSMO, yet, we achieve similar or better accuracies compared to them.

We have only one simple fully connected neural network with 2 trainable layers, compared to CRnet with K mixture of experts followed by relation module with complex loss functions (pairwsie).

Soft labeling employed in SZSL gives the model new flexibility to trade-off between seen and unseen accuracies during training and attain a higher value of harmonic accuracy A H , which is the standard metric for GZSL.

Assigned unseen soft labels (unseen probability q) enables the classifier to gain more confidence in recognizing unseen classes, which in turn results in considerably higher unseen accuracy A U .

As the classifier is now discriminating between more classes we get marginally lower seen accuracy A S .

However, balancing A S and A U with the cost of deteriorating A S leads to much higher A H .

This trade-off phenomenon is depicted in Figure 2 for all datasets.

The flexibility provided by soft labeling is examined by obtaining accuracies for different values of q. In Figure 2 .a and 2.b, by increasing total unseen probability q, A U increases and A S decreases as expected.

From the trade-off curves, there is an optimal q where A H takes its maximum value as shown in Figure  2 .

Maximizing A H is the primary objective in a GZSL problem that can be achieved by semantic similarity based soft labeling and the trade-off knob, q.

It should be noted that both AwA and aPY datasets (Figure 2 .a and 2.b) are coarse-grained class datasets.

In contrast, CUB and SUN datasets are fine-grained with hundreds of classes and highly unbalanced seen-unseen split, and hence their accuracies have different behavior concerning q, as shown in Figure 2 .c and 2.d.

However, harmonic average curve still has the same behavior and possesses a maximum value at an optimal q.

We illustrate the intuition with AwA dataset (Lampert et al., 2009 ), a ZSL benchmark dataset.

Consider a seen class squirrel.

We compute closest unseen classes to the class squirrel in terms of attributes.

We naturally find that the closest class is rat and the second closest is bat, while other classes such as horse, dolphin, sheep, etc.

are not close (Figure 3.a) .

This is not surprising as squirrel and rat share several attribute.

It is naturally desirable to have a classifier that gives rat higher probability than other classes.

If we force this softly, we can ensure that classifier is not blind towards unseen classes due to lack of any training example.

From a learning perspective, without any regularization, we cannot hope classifier to classify unseen classes accurately.

This problem was identified in , where they proposed entropy- based regularization in the form of Deep Calibration Network (DCN).

DCN uses cross-entropy loss for seen classes, and regularize the model with entropy loss on unseen classes to train the network.

Authors in DCN postulate that minimizing the uncertainty (entropy) of predicted unseen distribution of training samples, enables the network to become aware of unseen visual features.

While minimizing uncertainty is a good choice of regularization, it does not eliminate the possibility of being confident about the wrong unseen class.

Clearly in DCN's approach, for the above squirrel example, the uncertainty can be minimized even when the classifier gives high confidence to a wrong unseen class dolphin on an image of seen class squirrel.

Utilizing similarity based soft-labeling implicitly regularizes the model in a supervised fashion.

The similarity values naturally has information of how much certainty we want for specific unseen class.

We believe that this supervised regularization is the critical difference why our model outperforms DCN with a significant margin.

Figure 3 shows the effect of τ and the consequent assigned unseen distribution on accuracies for AwA dataset.

Small τ enforces q to be concentrated on nearest unseen class, while large τ spread q over all the unseen classes and basically does not introduce helpful unseen class information to the classifier.

The optimal value for τ is 0.2 for AwA dataset as depicted in Figure 3 .b.

The impact of τ on the assigned distribution for unseen classes is shown in Figure 3 .a when seen class is squirrel in AwA dataset.

Unseen distribution with τ = 0.2, well represents the similarities between seen class (squirrel) and similar unseen classes (rat, bat, bobcat) and basically verifies the result of Figure 3 .b where τ = 0.2 is the optimal temperature.

While in the extreme cases, when τ = 0.01, distribution on unseen classes in mostly focused on the nearest unseen class, rat, and consequently the other unseen classes' similarities are ignored.

Also τ = 10 flattens the unseen distribution which results in high uncertainty and does not contribute helpful unseen class information to the learning.

We proposed a discriminative GZSL classifier with visual-to-semantic mapping and cross-entropy loss.

During training, while SZSL is trained on a seen class, it simultaneously learns similar unseen classes through soft labels based on semantic class attributes.

We deploy similarity based soft labeling on unseen classes that allows us to learn both seen and unseen signatures simultaneously via a simple architecture.

Our proposed soft-labeling strategy along with cross-entropy loss leads to a novel regularization via generalized similarity-based weighted cross-entropy loss that can successfully tackle GZSL problem.

Soft-labeling offers a trade-off between seen and unseen accuracies and provides the capability to adjust these accuracies based on the particular application.

We achieve state-of-the-art performance, in GZSL setting, on all five ZSL benchmark datasets while keeping the model simple, efficient and easy to train.

@highlight

How to use cross-entropy loss for zero shot learning with soft labeling on unseen classes : a simple and effective solution that achieves state-of-the-art performance on five ZSL benchmark datasets.