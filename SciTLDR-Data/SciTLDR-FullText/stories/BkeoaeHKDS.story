We address the challenging problem of deep representation learning--the efficient adaption of a pre-trained deep network to different tasks.

Specifically, we propose to explore gradient-based features.

These features are gradients of the model parameters with respect to a task-specific loss given an input sample.

Our key innovation is the design of a linear model that incorporates both gradient features and the activation of the network.

We show that our model provides a local linear approximation to a underlying deep model, and discuss important theoretical insight.

Moreover, we present an efficient algorithm for the training and inference of our model without computing the actual gradients.

Our method is evaluated across a number of representation learning tasks on several datasets and using different network architectures.

We demonstrate strong results in all settings.

And our results are well-aligned with our theoretical insight.

Despite tremendous success of deep models, training deep neural networks requires a massive amount of labeled data and computing resources.

The recent development of representation learning holds great promises for improving data efficiency of training, and enables an easy adaption to different tasks using the same feature representation.

These features can be learned via either unsupervised learning using deep generative models (Kingma & Welling, 2013; Dumoulin et al., 2016) , or self-supervised learning with "pretext" tasks and pseudo labels (Noroozi & Favaro, 2016; Zhang et al., 2016; Gidaris et al., 2018) , or transfer learning from another large-scale dataset (Yosinski et al., 2014; Oquab et al., 2014; Girshick et al., 2014) .

After learning, the activations of the deep network are considered as generic features.

By leveraging these features, simple classifiers, e.g., linear models, can be build for different tasks.

However, given sufficient amount of training data, the performance of representation learning methods lack behind fully-supervised deep models.

As a step to bridge this gap, we propose to make use of gradient-based features from a pre-trained network, i.e., gradients of the model parameters relative to a task-specific loss given an input sample.

Our key intuition is that these per-sample gradients contain task-relevant discriminative information.

More importantly, we design a novel linear model that accounts for both gradient-based and activation-based features.

The design of our linear model stems from the recent advances in the theoretical analysis of deep models.

Specifically, our gradient-based features are inspired by the neural tangent kernel (Jacot et al., 2018; Arora et al., 2019b ) modified for finite-width networks.

Therefore, our model provides a local approximation of fine-tuning a underlying deep model, and the accuracy of the approximation is controlled by the semantic gap between the representation learning and the target tasks.

Finally, the specific structure of the gradient-based features and the linear model allows us to derive an efficient and scalable algorithm for training the linear model with these features.

To evaluate our method, we focus on visual representation learning in this paper, although our model can be easily modified for natural language processing or speech recognition.

To this end, we consider a number of learning tasks in vision, including unsupervised, self-supervised and transfer learning.

Our method was evaluated across tasks, datasets and architectures and compared against a set of baseline methods.

We observe empirically that our model with gradient-based features outperforms the traditional activation-based features by a significant margin in all settings.

Moreover, our results compare favorably against those produced by fine-tuning of network parameters.

Our main contributions are thus summarized as follows.

• We propose a novel representation learning method.

At the core of our method lies in a linear model that builds on gradients of model parameters as the feature representation.

• From a theoretical perspective, we show that our linear model provides a local approximation of fine-tuning an underlying deep model.

From a practical perspective, we devise an efficient and scalable algorithm for the training and inference of our method.

• We demonstrate strong results of our method across various representation learning tasks, different network architectures and several datasets.

Furthermore, these empirical results are well-aligned with our theoretical insight.

Representation Learning.

Learning good representation of data without expensive supervision remains a challenging problem.

Representation learning using deep models has been recently explored.

For example, different types of deep latent variable models (Kingma & Welling, 2013; Higgins et al., 2017; Berthelot et al., 2018; Dumoulin et al., 2016; Donahue et al., 2016; Dinh et al., 2016; Kingma & Dhariwal, 2018; Grathwohl et al., 2018) were considered for representation learning.

These models were designed to fit to the distribution of data, yet their intermediate responses were found useful for discriminative tasks.

Another example is self-supervised learning.

This paradigm seeks to learn from a discriminative pretext task whose supervision comes almost for free.

These pretext tasks for images include predicting rotation angles (Gidaris et al., 2018) , solving jigsaw puzzles (Noroozi & Favaro, 2016) and colorizing grayscale images (Zhang et al., 2016) .

Finally, the idea of transfer learning hinges on the assumption that feature maps learned from a large and generic dataset can be shared across closely related tasks and datasets (Girshick et al., 2014; Sharif Razavian et al., 2014; Oquab et al., 2014) .

The most successful models for transfer learning so far are those pre-trained on the ImageNet classification task (Yosinski et al., 2014) .

As opposed to proposing new representation learning tasks, our work primarily studies how to get the most out of the existing tasks.

Hence, our method is broadly applicable -it offers a generic framework that can be readily combined with any representation learning paradigm.

Gradients of Deep Networks.

Our method makes use of the Jacobian matrix of a deep network as feature representation for a downstream task.

Gradient information is traditionally employed for visualizing and interpreting convolutional networks (Simonyan et al., 2013) , and more recently for generating adversarial samples (Szegedy et al., 2013) , crafting defense strategies (Goodfellow et al., 2014) , facilitating network compression (Sinha et al., 2018) , knowledge distillation (Srinivas & Fleuret, 2018) , and boosting multi-task and meta learning (Sinha et al., 2018; Achille et al., 2019) .

Our work draws inspiration from Fisher vectors (FVs) (Jaakkola & Haussler, 1999 )-gradient-based features from a probabilistic model (e.g. GMM).

FVs have demonstrated its success for visual recognition using hand-crafted features (Perronnin & Dance, 2007) .

More recently, FVs have shown promising results with deep models, first as an ingredient of a hybrid system (Perronnin & Larlus, 2015) , and then as task embeddings for meta-learning (Achille et al., 2019) .

Our method differs from the FV approaches in two folds.

First, it is not built around a probabilistic model, hence has distinct theoretical motivations as we describe later.

Second, our method enjoys exact gradient computation with respect to network parameters and allows scalable training, whereas Perronnin & Larlus (2015) extracts FVs from a probabilistic module posterior to the network, and Achille et al. (2019) employs heuristics in their method to aggressively approximate the computation of FVs.

Neural Tangent Kernel (NTK) for Wide Networks.

Jacot et al. (2018) established the connection between deep networks and kernel methods by introducing the neural tangent kernel (NTK).

further showed that a network evolves as a linear model in the infinite width limit when trained on certain losses under gradient descent.

Similar ideas have been used to analyze wide deep neural networks, e.g., (Arora et al., 2019b; a; Li & Liang, 2018; Allen-Zhu et al., 2019a; Du et al., 2019; Allen-Zhu et al., 2019b; Cao & Gu, 2019; Mei et al., 2019) .

Our method is, to our best knowledge, the first attempt to port the theory into the regime of practical networks.

In the case of binary classification, our linear model reduces to a kernel machine equipped with NTK.

Instead of assuming random initialization of network parameters as all the prior works do, we for the first time evaluate the implication of pre-training on the linear approximation theory.

We consider a feed-forward deep neural network F (x; θ, ω) ω T f θ (x) that consists of a backbone f (x; θ) f θ (x) with its vectorized parameters θ and a linear model defined by ω (italic for vectors and bold for matrix).

Specifically, f θ encodes the input x into a vector representation f θ (x) ∈ R d .

ω ∈ R d×c are thus linear classifiers that map a feature vector into c output dimensions.

For this work, we focus on convolutional networks (ConvNets) for classification tasks.

With trivial modifications, our method can easily extend beyond ConvNets and classification, e.g., for a recurrent network as the backbone and/or for a regression task.

Following the setting of representation learning, we assume that a pre-trained fθ is given withθ as the learned weights.

The term representation learning refers to a set of learning methods that do not make use of discriminative signals from the task of interest.

For example, f can be the encoder of a deep generative model (Kingma & Welling, 2013; Dumoulin et al., 2016; Donahue et al., 2016) , or a ConvNet learned by using proxy tasks (self-supervised learning) (Goyal et al., 2019; Kolesnikov et al., 2019) or from another large-scale labeled dataset such as ImageNet (Deng et al., 2009 ).

Given a target task, it is a common practice to regard fθ(x) as a fixed feature extractor (activation-based features) and train a set of linear classifiers, given by

We omit the bias term for clarity.

Note thatω andθ are instantiations of ω and θ, whereω is the solution of the linear model andθ is given by representation learning.

Based on this setup, we now describe our method, discuss the theoretic implications and present an efficient training scheme.

Our method assumes a partition of θ (θ 1 , θ 2 ), where θ 1 and θ 2 parameterize the bottom and top layers of the ConvNet f (see Figure 1 (a) for an illustration).

Importantly, we propose to use gradient-based features ∇θ

is the Jacobian matrix of fθ w.r.t.

the pre-trained parametersθ 2 from the top layers of f .

Given the features (fθ(x), ω T Jθ 2 (x)) for x, our linear modelĝ, hereby considered as a classifier for concreteness, takes the form

where w 1 ∈ R d×c are linear classifiers initialized fromω, w 2 ∈ R |θ2| are shared linear weights for gradient features, and |θ 2 | is the size of the parameters θ 2 .

Both w 1 and w 2 are our model parameters that need to be learned from a target task.

An overview of the model is shown in Figure 1 (b).

Our model subsumes the linear model in Eq. (1) as the first term, and includes a second term that is linear in the gradient-based features.

We note that this extra linear term is different from traditional linear classifiers as in Eq. (1).

In this case, the gradient-based features form a matrix and the linear weight w 2 is multiplied to each row of the feature matrix.

Therefore, w 2 is shared for all output dimensions.

Similar to linear classifiers, the output ofĝ is further normalized by a softmax function and trained with a cross-entropy loss using labeled data from the target dataset.

Conceptually, our method can be summarized into three steps.

• Pre-train the ConvNet fθ.

This is accomplished by substituting in any existing representation learning algorithm.

• Train linear classifiersω using fθ(x).

This is a standard step in representation learning.

• Learn the linear modelĝ w1,w2 (x).

A linear model of special form (in Eq. (2)) is learned using gradient-based and activation-based features.

Note that our features are obtained when θ =θ is kept fixed, hence requires no extra tuning of the parametersθ.

The key insight is that our model provides a local linear approximation to F (x; θ 2 , ω).

This approximation comes from Eq. (2)-the crux of our approach.

Importantly, our linear model is mathematically well motivated -it can be interpreted as the 1st-order Taylor expansion of F θ,ω w.r.t.

its parameters (θ 2 , ω) around the point of (θ 2 ,ω).

More formally, we note that

With ω = w 1 and θ 2 −θ 2 = w 2 , Eq. (2) provides a linear approximation of the deep model F θ2,ω (x) around the initialization (θ 2 ,ω).

F θ2,ω can be considered as fine-tuning both θ 2 and ω for the target task.

Our key intuition is that given a sufficiently good base network, our model will provide a linear approximation to the underlying model F θ2,ω , and our training approximates fine-tuning F θ2,ω .

The quality of the linear approximation can be theoretically analyzed when the base network ω T fθ(x) is sufficiently wide and at random initialization.

This has been done via the recent neural tangent kernel approach (Jacot et al., 2018; Arora et al., 2019b) or some related ideas (Arora et al., 2019a; Li & Liang, 2018; Allen-Zhu et al., 2019a; Du et al., 2019; Allen-Zhu et al., 2019b; Cao & Gu, 2019; Mei et al., 2019) .

In fact, these studies are the theoretical inspiration for our approach.

However, while their approximation was developed for networks with infinite or sufficiently large width at random initialization, we apply the linear approximation on pre-trained models of practical sizes.

We argue that such an approximation is useful in practice for the following two critical and natural reasons:

The network f from representation learning provides a strong starting point.

Thus, the pretrained network parameterθ is close to a good solution for the downstream task.

Note that the key for a good linear approximation is that the output of the network is stable w.r.t.

small changes in the network parameter and activation.

In the existing analysis, this is proved under the conditions of large width and random base networks (see, e.g., Section 7 in (Allen-Zhu et al., 2019b) or Section B.1 in (Cao & Gu, 2019) ).

The pre-trained base network also has such stability properties, which are supported by empirical observations.

For example, the pre-trained network has similar predictions for a significant fraction of data in the downstream task as a fine-tuned network.

The network width required for the linearization to hold decreases as data becomes more structured.

An assumption made in existing analysis is that the network is sufficiently or even infinitely wide compared to the size of the dataset, so that the approximation can hold for any dataset.

We argue that this is not necessary in practice, since the practical datasets are well-structured, and theoretically it has been shown that as long as the trained network is sufficiently wide compared to the effective complexity determined by the structure of the data, then the approximation can hold (Li & Liang, 2018; Allen-Zhu et al., 2019a) .

Our approach thus takes advantage of the bottom layers to reduce data complexity in the hope that linearization of the top (and often the widest) layers can be sufficiently accurate.

Compared to fine-tuning F θ2,ω , our method only needs to learn a linear classifier, which is efficient and straightforward.

In particular, it is efficient for training and inference using our scalable training technique described below, while achieving much better performance than using activation-based features.

In the most interesting setting where the tasks for pre-training are similar to the target tasks, the linear approximation works well, and our method achieves comparable or even better performance than fine-tuning; see the experimental section for details.

Moving beyond the theoretic aspects, a practical challenge of our method lies in the scalable training ofĝ.

A naïve approach requires evaluating and storingω T Jθ 2 (x) for all x, which is computationally expensive and can become infeasible as the output dimension c and the number of parameters |θ 2 | grow.

Inspired by Pearlmutter (1994), we design an efficient training and inference scheme forĝ.

Thanks to this scheme, the complexity of training our model using gradient-based features is on the same magnitude as training a linear classifier on activation-based features.

Central to our scalable approach is the inexpensive evaluation of the Jacobian-vector product (JVP) ω T Jθ 2 (x)w 2 , whose size is the same as the output dimension c. First, we note that

by 1st-order Taylor expansion around a scalar r = 0.

Rearrange and take the limit of r to 0, we get

which can be conveniently evaluated via forward-mode automatic differentiation.

More precisely, let us consider the basic building block of f -convolutional layers.

These layers are defined as a linear function h(z c ; w c , b c ) = w

where ∂zc ∂r is the JVP coming from the upstream layer.

When a nonlinearity is encountered, we have, using the ReLU function as an example,

where is the element-wise product, and 1 is the element-wise indicator function and z c is the input to the layer.

Other activation functions as well as average/max pooling layers can be handled in the same spirit.

For batch normalization, we fold them into their corresponding convolutions.

Importantly, Eq. (6) and (7) provide an efficient approach to computing the desired JVP in Eq. (5) by successively evaluating a set of JVPs on the fly.

This process starts with the seed ∂z0 ∂r = 0, where z 0 is the output of the last layer in f parameterized by θ 1 and can be pre-computed.ω T Jθ 2 (x)w 2 can be computed along with the standard forward propagation through f .

Moreover, during the training ofĝ, its parameters w 1 and w 2 can be updated via standard back-propagation.

In summary, our approach only requires a single forward pass through the fixed f for evaluatingĝ, and a single backward pass for updating the parameters w 1 and w 2 .

Complexity Analysis.

We further discuss the complexity of our method in training and inference, and contrast our method to the fine-tuning of network parameters θ 2 .

Our forward pass, as demonstrated by Eq. (6) and (7), is a chain of linear operations intertwined by element-wise multiplications.

The second term in Eq. (6) forms the "main stream" of computation, while the first, "branch" term merges into the main stream at every layer of the ConvNet f .

The same reasoning holds for the backward pass.

Overall, our method requires twice as many linear operations as fine-tuning θ 2 of the ConvNet.

Note, however, that half of the linear operations by our method are slightly cheaper due to removal of the bias term.

Moreover, in the special case where θ 2 only includes the very top layer of f , our method carries out the same number of operations as fine-tuning since the second term in Eq. (6) can be dropped.

For memory consumption, our method requires to store an additional "copy" (linear weights) of the model parameters in comparison to fine-tuning.

As the size of θ 2 is small, this minor increase of computing and memory cost puts our method on the same page as fine-tuning.

We now describe our experiments and results.

Our main results are organized into two parts.

First, we conduct an ablation study of our method on CIFAR-10 dataset (Krizhevsky et al., 2009 ).

Our study is to dissect how different approaches of computing the gradient features influence their representational power.

Moreover, we evaluate our method on three representation learning tasks: unsupervised learning using deep generative models, self-supervised learning using a pretext task, and transfer learning from large-scale datasets.

We report results on several datasets and network architectures and demonstrate the strength of our method.

Implementation Details.

In the rest our experiments, we use the same settings for training and evaluating our models and baseline methods, unless otherwise noticed.

Concretely, we adopt the NTK parametrization (Jacot et al., 2018) for θ 2 and fold batch normalization into their preceding convolutional layers, piror to our training.

For training of our method, we use a batch size of 128, learning rate of 0.001, and train until convergence using the Adam optimizer (Kingma & Ba, 2014) .

No weight decay or dropout is used for our method.

For inference, we evaluate on a single center crop of the image for all datasets.

We often refer to "baseline" as a linear classifier (logistic regression) trained on top of the last activation from the network.

In this study, we probe the design of our model on a set of ablations on CIFAR-10 by using a pretrained encoder ConvNet from BiGAN (Dumoulin et al., 2016) (trained on CIFAR-10).

Specifically, our gradient feature ω T Jθ 2 (x) is a function of three parameters θ 1 , θ 2 and ω, which can take on either random or pre-trained values.

We probe the representational power of ω T Jθ 2 (x) by feeding it all possible configurations of the three parameters.

Moreover, we compare our results to (1) a linear classifier baseline, (2) the fine-tuning of the network and (3) our method starting from a finetuned model.

Our results, summarized in Table 1, highlight that the success of our method crucially depends on pre-training all three parameters.

We now provide a detailed discussion of our findings.

Random vs. pre-trainedθ.

Pre-training f plays a central role in the materialization of our theoretical insight.

As is evidenced by our results, pre-trained θ 1 and θ 2 each injects substantial amount of information into ω T Jθ 2 (x).

Computing ω T Jθ 2 (x) from random θ 2 leads to visible performance drop, while our method becomes virtually indistinguishable from the baseline when θ 1 is set to be random.

Random vs. pre-trainedω.

We obtain better results by first training the baseline classifier Eq.

(1), feeding its learned weightsω into the gradient operator ∇θ 2 F for computing ω T Jθ 2 (x), and then seamlessly transitioning to training our proposed classifier Eq. (2) by adding on the gradient term.

Hence, the best training procedure for our model can be effectively broken down into a "pretraining" phase and a "residual learning" phase.

Optimal size of θ 2 .

As the size of ω T Jθ 2 (x) grows, our model's performance improves slightly albeit at the expense of extra computational overhead.

Fortunately, our results suggest that it suffices to set the very top layer as θ 2 to enjoy a reasonably large performance gain.

Our method vs. fine-tuning of θ 2 and ω.

We highlight two observations hint by our theoretical insight.

First, our model, initialized with pre-trained parameters, stays close to the fine-tuned network in terms of performance, when the gradient features come from a top layer with moderate width.

Second, we observe an enlarging performance gap between our model and the fine-tuned network as we gather gradient features from more narrower layers towards the bottom.

Furthermore, our model using gradient features w.r.t.

fine-tuned parameters stays almost the same as the fine-tuned network.

Remarks.

Our ablation study shows that the best practice for our method is to start with pretrainedθ 2 (from representation learning) andω (baseline classifier for the target task).

Moreover, the performance of our method increases slightly as the size of θ 2 grows, i.e., linearization of more layers in f .

This performance boost, however, comes with an increased computational cost, as well as an increased performance gap to the fine-tuned model.

A small sized θ 2 , e.g., from the parameters of last few convolutional layers seems to be sufficient.

We present results of our method on three different representation learning tasks: unsupervised learning, self-supervised learning and transfer learning.

For all experiments in this section, we contrast at least three sets of results: a baseline linear classifier on activation-based features, our proposed linear classifier that makes use of gradient-based features w.r.t.

θ 2 , and a network whose θ 2 is fine-tuned for the target dataset.

We compare our model against the baseline to demonstrate the advantage of our gradient-based features, and against the fine-tuned network (our theoretical upper bound) to illustrate the various factors that support or break down our theoretical insight.

Unsupervised Learning using Deep Generative Models.

We consider BiGAN and VAE training as the representation-learning tasks and use their encoders as the pre-trained ConvNet f .

Our BiGAN and VAE models strictly follow the architecture and training setup from (Dumoulin et al., 2016) and (Berthelot et al., 2018) .

We average-pool the output of the ConvNet for activation-based features, and compute our gradient-based features from one, two or all of the top-3 convolutional layers.

Training of our linear model follows the two-step process as described before.

We train on the train split of CIFAR-10/100 and the extra split of SVHN, and report the top-1 classification accuracy on the test splits of both datasets.

Results.

We summarize our results in Table 2 .

Our models consistently outperform the baseline across three datasets with relative improvement over 10%.

Moreover, we observe good agreement of performance between our models and the fine-tuned networks.

In the BiGAN case, performance of our method saturates with gradient-based features only from the very top layer.

Self-supervised Learning.

We experiment with a ResNet50 pre-trained on the Jigsaw pretext task available from (Goyal et al., 2019) .

We refer the reader to Noroozi & Favaro (2016) for technical details.

We average-pool the output of the ConvNet for activation-based features, and compute our Table 3 : Self-supervise Learning Results: Evaluation of our method using ConvNet f pre-trained via self-supervised learning.

All results are reported using mean average precision (mAP) score.

Our f is a ResNet-50 pre-trained on the Jigsaw pretext task.

Our gradient-based features are computed w.r.t.

to the three residual blocks in the last stage.

The three conv layers in each block have widths 2048, 512 and 2048.

Unless specified, the cited experiments are AlexNet-based. (2015) Logistic regressor 44.7 n/a gradient-based features from the three residual blocks in the last stage.

Unlike Goyal et al. (2019) which uses linear SVM as their baseline classifier, we use a standard linear logistic regressor.

We train on the trainval split of VOC07 and the train split of COCO2014 for image classification, and report the mean average precision (mAP) scores on their test and val splits respectively.

Results.

We summarize our experimental results in Table 3 .

Again, we observe a large performance boost in comparison to the linear classifier baseline (over 5% on VOC07 and COCO2014).

Our method also outperforms a large set of self-supervised learning methods.

We note that the gap between our method and the fine-tuned network is quite large in this setting (more than 15%).

We conjecture that this is due to the large semantic gap between the representation learning task (jigsaw on ImageNet) and the target task (classification on VOC07 and COCO2014).

Table 4 : Transfer Learning Results: Evaluation of our method using ImageNet pre-trained models f .

All results are reported using mAP score.

Our f is conv1-5 with or without fc6-7 of a pretrained AlexNet.

Our gradient-based features are from the last two layers of f .

We use a SGD optimizer with a learning rate of 1e-3 for VOC07 (1e-2 for COCO2014), a weight decay of 1e-6 and a momentum of 0.9.

We run for a total of 80000 iterations for VOC07 (20000 iterations for COCO2014), reducing the learning rate by a factor of 2 eight times evenly throughout training.

Transfer Learning from ImageNet.

We start with the PyTorch distribution of ImageNet pre-trained AlexNet (Paszke et al., 2017) .

In our first setting (conv1-5), we remove the fully connected layers so that the remaining convolutional block becomes our f .

To obtain activation-based features, we downsample the output of f to form a 1024-dimensional feature vector.

Our gradient features are computed w.r.t.

conv4 and conv5 (width 256).

Our second setting (conv1-5 + fc6-7) treats all except the last layer as f , and computes gradient features from fc6 and fc7 (width 4096).

We follow the weight rescaling technique and the evaluation framework proposed by Krähenbühl et al. (2015) .

Results.

Our results are presented in Table 4 .

Our method again demonstrates strong performance on both datasets, with a notable 2% (5%) and 5% (2%) improvements on VOC07 and COCO2014 datasets in comparison to the baseline.

More importantly, our method slightly outperforms the finetuned networks on both datasets in the first setting.

This is a very interesting result.

We argue that it is due to the significant overlapping between the pre-training and target tasks.

It would be interesting to identify other representation learning scenarios where our method is particularly beneficial.

In this paper, we presented a novel method for deep representation learning.

Specifically, given a pre-trained deep model, we explored the per-sample gradients of the model parameters relative to a task-specific loss, and constructed a linear model that combines gradients of model parameters and the activation of the model.

We showed that our model can be very efficient in training and inference, and provides a local linear approximation to an underlying deep model.

Through a set of experiments, we demonstrated that these gradient-based features are highly discriminative for the target task, and our method can significantly improve over the baseline method of representation learning across tasks, datasets and network architectures.

We believe that our work provides a step forward towards deep representation learning.

Our ablation studies seek to address two important questions:

1.

Does pre-training encourage more powerful gradient feature?

2.

What is the optimal size of the gradient feature?

We answer the first question by introducing different combinations of random and pre-trained parameters in the base network, and the second by varying the number of layers contributing to the gradient feature.

We compare our linear model against two baselines.

The first baseline, g w1 (x), is the first term of our model.

It is the standard logistic regressor on network activation and is used for representation learning in practice.

The second baseline,ω T Jθ 2 (x)w 2 , is the second term of our model.

It is linear in the gradient feature and can serve as a linear classifier on its own.

We hereafter call the two baselines the activation model and the gradient model, and our proposed model the full model for convenience.

Moreover, we compare our method against fine-tuning, which helps us assess the validity of our theoretical insight.

We train a BiGAN on CIFAR-10 and use its encoder as the base network for classification of the same dataset.

The base network has five conv layers with width 32, 64, 128, 256 and 512, and a final fc layer.

The results are shown in Table 5 .

To help the reader better digest the results, we explain what the hyperparameter setting entails in Table 5 using two examples.

We first consider the two top entries under the column θ 2 : conv5, which shows that the gradient model and the full model achieve an accuracy of 23.09% and 62.83% respectively.

In this scenario, θ 1 parametrizes conv1-4, θ 2 parametrizes conv5, and ω parametrizes the fc layer.

The gradient feature (i.e. the Jacobian of network activation w.r.t.

θ 2 ) is computed using a completely random network (i.e., θ 1 , θ 2 and ω are all randomly initialized).

Now consider the two bottom entries from the last column.

Here, θ 1 and θ 2 parametrize conv1-2 and conv3-5 respectively, hence the size of θ 2 grows compared to the first scenario.

In addition, the gradient feature is now computed using a network whose parameters are all pre-trained.

Importantly, the activation feature in all scenarios is computed using the pre-trained network, which ensures that variation in performance of the full model can solely be attributed to the gradient feature.

It should now be straightforward to interpret the rest of the table.

We use the PyTorch distribution of ImageNet pre-trained ResNet-18 as the base network for VOC07 object classification.

The main section of the network has four layers, each containing two residual blocks.

Our results are shown in Table 6 and can be interpreted as in the unsupervised setting.

We summarize our main conclusions from the ablation studies.

1.

Pre-training is required for the representation power of the gradient feature.

This holds for both the gradient model and the full model.

The full model's improvement over the standard logistic regressor is not a consequence of an increase in the number of model parameters.

The full model supplied with gradient from a random network is no better than the simple activation baseline.

3.

Our method is more successful as the dataset and the base network grow in complexity.

The full model consistently outperforms the baselines and fine-tuning with VOC07 and ResNet-18 as the target dataset and the base network, while it is indistinguishable from the gradient baseline in the case of toy datasets (CIFAR-10) and toy networks (BiGAN encoder for CIFAR-10).

This conclusion is further justified by later results on self-supervised and transfer learning.

4.

Gradient from the topmost layer (or residual block) of a pre-trained network suffices to ensure a reasonably large performance gain.

Further inflating the gradient feature introduces little extra gain and can sometimes hurt performance.

every 20K iterations or 20 epochs.

We set β 1 to 0.5 and weight decay to 1e-6.

For finetuning, we also apply the SGD optimizer with the same learning rate scheduling, a weight decay of 5e-5 and a momentum of 0.9.

We report the better result between the two runs.

4.

We only use gradient feature from the last conv layer or residual block of the base network so as to respect our conclusion from the ablation studies.

5.

For VOC07 and COCO2014 datasets, predictions are averaged over ten random crops at test time.

This is in contrast to our past practice, where prediction is based on the center crop.

Note that we still use the center-crop for evaluation in the ablation study for simplicity.

6.

In the transfer learning setting, we use ResNet-18 instead of AlexNet as the base network.

This same network was used in our ablation study.

Please see Tables 7, 8 and 9 for the results.

Our model outperforms the standard logistic regressor on network activation by a large margin in all scenarios.

On the challenging VOC07 and COCO2014 datasets, our model outperforms the gradient baseline and even fine-tuning.

Moreover, we observe that our model is as good as or better than fine-tuning when either the dataset has low complexity or there is significant overlapping between the pre-training and the target tasks.

It would be interesting to identify other representation learning scenarios where our method is particularly beneficial.

@highlight

Given a pre-trained model, we explored the per-sample gradients of the model parameters relative to a task-specific loss, and constructed a linear model that combines gradients of model parameters and the activation of the model.

@highlight

This paper proposes to use the gradients of specific layers of convolutional networks as features in a linearized model for transfer learning and fast adaptation.