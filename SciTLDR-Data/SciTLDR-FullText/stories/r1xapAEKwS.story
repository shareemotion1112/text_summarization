In probabilistic classification, a discriminative model based on Gaussian mixture exhibits flexible fitting capability.

Nevertheless, it is difficult to determine the number of components.

We propose a sparse classifier based on a discriminative Gaussian mixture model (GMM), which is named sparse discriminative Gaussian mixture (SDGM).

In the SDGM, a GMM-based discriminative model is trained by sparse Bayesian learning.

This learning algorithm improves the generalization capability by obtaining a sparse solution and automatically determines the number of components by removing redundant components.

The SDGM can be embedded into neural networks (NNs) such as convolutional NNs and can be trained in an end-to-end manner.

Experimental results indicated that the proposed method prevented overfitting by obtaining sparsity.

Furthermore, we demonstrated that the proposed method outperformed a fully connected layer with the softmax function in certain cases when it was used as the last layer of a deep NN.

In supervised classification, probabilistic classification is an approach that assigns a class label c to an input sample x by estimating the posterior probability P (c|x).

This approach is primarily categorized into two types of models: discriminative model and generative model.

The former optimizes the posterior distribution P (c|x) directly on a training set, whereas the latter finds the class conditional distribution P (x|c) and class prior P (c) and subsequently derives the posterior distribution P (c|x) using Bayes' rule.

The discriminative model and generative model are mutually related (Lasserre et al., 2006; Minka, 2005) .

According to Lasserre et al. (2006) , the only difference between these models is their statistical parameter constraints.

Therefore, given a certain generative model, we can derive a corresponding discriminative model.

For example, the discriminative model corresponding to a unimodal Gaussian distribution is logistic regression (see Appendix A for derivation).

Several discriminative models corresponding to the Gaussian mixture model (GMM) have been proposed (Axelrod et al., 2006; Bahl et al., 1996; Klautau et al., 2003; Tsai & Chang, 2002; Tsuji et al., 1999; Tüske et al., 2015; Wang, 2007) .

They indicate more flexible fitting capability than the generative GMM and have been applied successfully in fields such as speech recognition (Axelrod et al., 2006; Tüske et al., 2015; Wang, 2007) .

The problem to address in mixture models such as the GMM is the determination of the number of components M .

Classically, Akaike's information criterion and the Bayesian information criterion have been used; nevertheless, they require a considerable computational cost because a likelihood must be calculated for every candidate component number.

In the generative GMM, methods that optimize M during learning exist (Crouse et al., 2011; Štepánová & Vavrečka, 2018) .

However, in a discriminative GMM, a method to optimize M simultaneously during learning has not been clearly formulated.

In this paper, we propose a novel GMM having two important properties: sparsity and discriminability, which is named sparse discriminative Gaussian mixture (SDGM).

In the SDGM, a GMM-based discriminative model is trained by sparse Bayesian learning.

This learning algorithm improves the generalization capability by obtaining a sparse solution and determines the number of components automatically by removing redundant components.

Furthermore, the SDGM can be embedded into neural networks (NNs) such as convolutional NNs and trained in an end-to-end manner with an NN.

To the authors best knowledge, there is no GMM that has both of sparsity and discriminability.

The contributions of this study are as follows:

• We propose a novel sparse classifier based on a discriminative GMM.

The proposed SDGM has both sparsity and discriminability, and determines the number of components automatically.

The SDGM can be considered as the theoretical extension of the discriminative GMM and the relevance vector machine (RVM) (Tipping, 2001 ).

• This study attempts to connect both fields of probabilistic models and NNs.

From the equivalence of a discriminative model based on Gaussian distribution to a fully connected layer, we demonstrate that the SDGM can be used as a module of a deep NN.

We also show that the SDGM can show superior performance than the fully connected layer with a softmax function via an end-to-end learning with an NN on the image recognition task.

An SDGM takes a continuous variable x ∈ R D as its input and outputs its posterior probability P (c|x) for each class c ∈ {1, . . .

, C}. An SDGM acquires a sparse structure by removing redundant components via sparse Bayesian learning.

Figure 1 shows how the SDGM is trained by removing unnecessary components while keeping discriminability.

The two-class training data are from Ripley's synthetic data (Ripley, 2006) , where a Gaussian mixture model with two components is used for generating data of each class.

In this training, we set the initial number of components to three for each class.

As the training progresses, one of the components for each class becomes small gradually and is removed.

The posterior probabilities for each class c is calculated as follows:

where M c is the number of components for class c and π cm is the mixture weight that is equivalent to the prior of each component P (c, m).

It should be noted that we use w cm ∈ R H , which is the weight vector representing the m-th Gaussian component of class c. The dimension of w cm , i.e., H, is the same as that of φ; namely, H = 1 + D(D + 3)/2.

Algorithm 1: Weight updating Input: Training data set X and teacher vector T. Output: Trained weight w obtained by maximizing (11).

Initialize the weights w, hyperparameters α, mixture coefficients π, and posterior probabilities r; while α have not converged do Calculate J using (9); while r have not converged do while w have not converged do Calculate gradients using (12); Calculate Hessian (13); Maximize (11) w.r.t.

w; Calculate P (c, m|x n ) and P (c|x n ); end r ncm = P (c, m|x n )/P (c|x n ); end Calculate Λ using (16); Update α using (17); Update π using (18); end Derivation.

Utilizing Gaussian distribution as a conditional distribution of x given c and m, P (x|c, m), the posterior probability of c given x, P (c|x), is calculated as follows:

where µ cm ∈ R D and Σ cm ∈ R D×D are the mean vector and the covariance matrix for component m in class c. Since the calculation inside an exponential function in (5) is quadratic form, the conditional distributions can be transformed as follows:

where

Here, s cmij is the (i, j)-th element of Σ −1 cm .

Algorithm 1 shows the training of the SDGM.

In this algorithm, the optimal weight is obtained as maximum a posteriori solution.

We can obtain a sparse solution by optimizing the prior distribution set to each weight simultaneously with weight optimization.

A set of training data and target value {x n , t nc } (n = 1, · · · , N ) is given.

The target t nc is coded in a one-of-K form, where t nc = 1 if the n-th sample belongs to class c, t nc = 0 otherwise.

A binary random variable z ncm is introduced.

The variable z ncm = 1 when the n-th sample from class c belongs to the m-th component.

Otherwise, z ncm = 0.

This variable is required for the optimization of the mixture weight π cm .

We also define π and z as vectors that comprise π cm and z ncm as their elements, respectively.

As the prior distribution of the weight w cmh , we employ a Gaussian distribution with a mean of zero.

Using a different precision parameter (inverse of the variance) α cmh for each weight w cmh , the joint probability of all the weights is represented as follows:

where w and α are vectors with w cmh and α cmh as their elements, respectively.

During learning, we update not only w but also α.

If α cmh → ∞, the prior (8) is 0; hence a sparse solution is obtained by optimizing α.

Using these variables, the expectation of the log-likelihood function over z, J, is defined as follows:

where T is a matrix with t nc as its element.

The training data matrix X contains x T n in the n-th row.

The variable r ncm in the right-hand side corresponds to P (m|c, x n ) and can be calculated as r ncm = P (c, m|x n )/P (c|x n ).

The posterior probability of the weight vector w is described as follows:

An optimal w is obtained as the point where (10) is maximized.

The denominator of the right-hand side in (10) is called the evidence term, and we maximize it with respect to α.

However, this maximization problem cannot be solved analytically; therefore we introduce the Laplace approximation described as the following procedure.

With α fixed, we obtain the mode of the posterior distribution of w. The solution is given by the point where the following equation is maximized:

where A = diag α cmh .

We obtain the mode of (11) via Newton's method.

The gradient and Hessian required for this estimation can be calculated as follows:

(13) Each element of ∇J and ∇∇J is calculated as follows:

where δ cc mm is a variable that takes 1 if both c = c and m = m , 0 otherwise.

Hence, the posterior distribution of w can be approximated by a Gaussian distribution with a mean ofŵ and a covariance matrix of Λ, where

Because the evidence term can be represented using the normalization term of this Gaussian distribution, we obtain the following updating rule by calculating its derivative with respect to α cmh .

where λ cmh is the diagonal component of Λ. The mixture weight π cm can be estimated using r ncm as follows:

where N c is the number of training samples belonging to class c. As described above, we obtain a sparse solution by alternately repeating the update of hyper-parameters, as described in (17) and (18) and the posterior distribution estimation of w using the Laplace approximation.

During the procedure, the {c, m}-th component is eliminated if π cm becomes 0 or all the weights w cmh corresponding to the component become 0.

To evaluate the characteristics of the SDGM, we conducted classification experiments using synthetic data.

The dataset comprises two classes.

The data were sampled from a Gaussian mixture model with eight components for each class.

The numbers of training data and test data were 320 and 1,600, respectively.

The scatter plot of this dataset is shown in Figure 2 .

In the evaluation, we calculated the error rates for the training data and the test data, the number of components after training, the number of nonzero weights after training, and the weight reduction ratio (the ratio of the number of the nonzero weights to the number of initial weights), by varying the number of initial components as 2, 4, 8, . . .

, 20.

Figure 2 displays the changes in the learned class boundaries according to the number of initial components.

When the number of components is small, such as that shown in Figure 2(a) , the decision boundary is simple; therefore, the classification performance is insufficient.

However, according to the increase in the number of components, the decision boundary fits the actual class boundaries.

It is noteworthy that the SDGM learns the GMM as a discriminative model instead of a generative model; an appropriate decision boundary was obtained even if the number of components for the model is less than the actual number (e.g., 2(c)).

Figure 3 shows the evaluation results of the characteristics.

Figures 3(a), (b) , (c), and (d) show the recognition error rate, number of components after training, number of nonzero weights after training, and weight reduction ratio, respectively.

The horizontal axis shows the number of initial components in all the graphs.

In Figure 3(a) , the recognition error rates for the training data and test data are almost the same with the few number of components, and decrease according to the increase in the number of initial components while it is 2 to 6.

This implied that the representation capability was insufficient when the number of components was small, and that the network could not accurately separate the classes.

Meanwhile, changes in the training and test error rates were both flat when the number of initial components exceeded eight, even though the test error rates were slightly higher than the training error rate.

In general, the training error decreases and the test error increases when the complexity of the classifier is increased.

However, the SDGM suppresses the increase in complexity using sparse Bayesian learning, thereby preventing overfitting.

In Figure 3(b) , the number of components after training corresponds to the number of initial components until the number of initial components is eight.

When the number of initial components exceeds ten, the number of components after training tends to be reduced.

In particular, eight components are reduced when the number of initial components is 20.

The results above indicate the SDGM can reduce unnecessary components.

From the results in Figure 3 (c), we confirm that the number of nonzero weights after training increases according to the increase in the number of initial components.

This implies that the complexity of the trained model depends on the number of initial components, and that the minimum number of components is not always obtained.

Meanwhile, in Figure 3 (d), the weight reduction ratio increases according to the increase in the number of initial components.

This result suggests that the larger the number of initial weights, the more weights were reduced.

Moreover, the weight reduction ratio is greater than 99 % in any case.

The results above indicate that the SDGM can prevent overfitting by obtaining high sparsity and can reduce unnecessary components.

To evaluate the capability of the SDGM quantitatively, we conducted a classification experiment using benchmark datasets.

The datasets used in this experiment were Ripley's synthetic data (Ripley, 2006) (Ripley hereinafter) and four datasets cited from Rätsch et al. (2001) ; Banana, Waveform, Titanic, and Breast Cancer.

Ripley is a synthetic dataset that is generated from a two-dimensional (D = 2) Gaussian mixture model, and 250 and 1,000 samples are provided for training and test, respectively.

The number of classes is two (C = 2), and each class comprises two components.

The remaining four datasets are all two-class (C = 2) datasets, which comprise different data size and dimensionality.

Since they contain 100 training/test splits, we repeated experiments for 100 times and then calculated average statistics.

For comparison, we used three classifiers that can obtain a sparse solution: a linear logistic regression (LR) with l 1 constraint, a support vector machine (SVM) (Cortes & Vapnik, 1995) and a relevance vector machine (RVM) (Tipping, 2001) .

In the evaluation, we compared the recognition error rates for discriminability and number of nonzero weights for sparsity on the test data.

The results of SVM and RVM were cited from Tipping (2001).

For ablation study, we also tested our SDGM without sparse learning by omitting the update of α.

By way of summary, the statistics were normalized by those of the SDGM and the overall mean was shown.

Table 1 shows the recognition error rates and number of nonzero weights for each method.

The results in Table 1 show that the SDGM achieved an equivalent or greater accuracy compared with the SVM and RVM on average.

The SDGM is developed based a Gaussian mixture model and is particularly effective for data where a Gaussian distribution can be assumed, such as the Ripley dataset.

On the number of nonzero weights, understandably, the LR showed the smallest number since it is a linear model.

Among the remaining nonlinear classifiers, the SDGM achieved relatively small number of nonzero weights thanks to its sparse Bayesian learning.

The results above indicated that the SDGM demonstrated generalization capability and a sparsity simultaneously.

In this experiment, the SDGM is embedded into a deep neural network.

Since the SDGM is differentiable with respect to the weights, SDGM can be embedded into a deep NN as a module and is trained in an end-to-end manner.

In particular, the SDGM plays the same role as the softmax function since the SDGM calculates the posterior probability of each class given an input vector.

We can show that a fully connected layer with the softmax is equivalent to the discriminative model based on a single Gaussian distribution for each class by applying a simple transformation (see Appendix A), whereas the SDGM is based on the Gaussian mixture model.

To verify the difference between them, we conducted image classification experiments.

Using a CNN with a softmax function as a baseline, we evaluated the capability of SDGM by replacing softmax with the SDGM.

We used the following datasets and experimental settings in this experiment.

MNIST: This dataset includes 10 classes of handwritten binary digit images of size 28 × 28 (LeCun et al., 1998) .

We used 60,000 images as training data and 10,000 images as testing data.

As a feature extractor, we used a simple CNN that consists of five convolutional layers with four max pooling layers between them and a fully connected layer.

To visualize the learned CNN features, we first set the output dimension of the fully connected layer of the baseline CNN as two (D = 2).

Furthermore, we tested by increasing the output dimension of the fully connected layer from two to ten (D = 10).

Fashion-MNIST: Fashion-MNIST (Xiao et al., 2017) includes 10 classes of binary fashion images with a size of 28 × 28.

It includes 60,000 images for training data and 10,000 images for testing data.

We used the same CNN as in MNIST with 10 as the output dimension.

CIFAR-10: CIFAR-10 ( Krizhevsky & Hinton, 2009 ) is the labeled subsets of an 80 million tiny image dataset.

This dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

There are 50,000 training images and 10,000 test images.

For CIFAR-10, we trained DenseNet (Huang et al., 2017 ) with a depth of 40 and a growth rate of 12.

For each dataset, the network was trained with a batch size of 64 for 100 epochs with a learning rate of 0.01 We used a weight decay of 1.0 × 10 −5 and the Nesterov optimization algorithm (Sutskever et al., 2013 ) with a momentum of 0.9.

The network weights were initialized using the Glorot uniform (Glorot & Bengio, 2010) .

Figure 4 shows the two-dimensional feature embeddings on the MNIST dataset.

Different feature embeddings were acquired for each method.

When softmax was used, the features spread in a fan shape and some part of the distribution overlapped around the origin.

However, when the SDGM was used, the distribution for each class exhibited an ellipse shape and margins appeared between the class distributions.

This is because the SDGM is based on a Gaussian mixture model and functions to push the samples into a Gaussian shape.

Table 2 shows the recognition error rates on each dataset.

SDGM achieved better performance than softmax.

As shown in Figure 4 , SDGM can create margins between classes by pushing the features into a Gaussian shape.

This phenomenon positively affected the classification capability.

Figure 5 illustrates the relationship of our study with other studies.

This study is primarily consists of three factors: discriminative model, Gaussian mixture model, and Sparse Bayesian learning.

This study is the first that combines these three factors and expands the body of knowledge in these fields.

Sparse GMM (Gaiffas, 2014) Discriminative GMM (Klautau, 2003) Mixture model Discriminative Sparse Bayes Figure 5 : Relationship of our study with other studies.

the logistic regression is equivalent to the discriminative model of a unimodal Gaussian model, the SDGM can be considered as an extended RVM using a GMM.

furthermore, from the perspective of the probabilistic model, the SDGM is considered as the an extended discriminative GMM (Klautau et al., 2003) using sparse Bayesian learning, and an extended sparse GMM (Gaiffas & Michel, 2014) using the discriminative model.

Sparse methods have often been used in machine learning.

Three primary merits of using sparse learning are as follows: improvements in generalization capability, memory reduction, and interpretability.

Several attempts have been conducted to adapt sparse learning to deep NNs.

Graham (2014) proposed a spatially-sparse convolutional neural network.

Liu et al. (2015) proposed a sparse convolution neural network.

Additionally, sparse Bayesian learning has been applied in many fields.

For example, an application to EEG classification has been reported (Zhang et al., 2017) .

In this paper, we proposed a sparse classifier based on a GMM, which is named SDGM.

In the SDGM, a GMM-based discriminative model was trained by sparse Bayesian learning.

This learning algorithm improved the generalization capability by obtaining a sparse solution and automatically determined the number of components by removing redundant components.

The SDGM could be embedded into NNs such as convolutional NNs and could be trained in an end-to-end manner.

In the experiments, we demonstrated that the SDGM could reduce the amount of weights via sparse Bayesian learning, thereby improving its generalization capability.

The comparison using benchmark datasets suggested that SDGM outperforms the conventional sparse classifiers.

We also demonstrated that SDGM outperformed the fully connected layer with the softmax function when it was used as the last layer of a deep NN.

One of the limitations of this study is that sparse Bayesian learning was applied only when the SDGM was trained stand-alone.

In future work, we will develop a sparse learning algorithm for a whole deep NN structure including the feature extraction part.

This will improve the ability of the CNN for larger data classification.

@highlight

A sparse classifier based on a discriminative Gaussian mixture model, which can also be embedded into a neural network.

@highlight

The paper presents a Gaussian mixture model trained via gradient descent arguments which allows for inducing sparsity and reducing the trainable model layer parameters.

@highlight

This paper proposes a classifier, called SDGM, based on discriminative Gaussian mixture and its sparse parameter estimation.