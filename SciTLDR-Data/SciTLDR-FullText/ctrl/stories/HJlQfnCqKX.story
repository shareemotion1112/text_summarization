As shown in recent research, deep neural networks can perfectly fit randomly labeled data, but with very poor accuracy on held out data.

This phenomenon indicates that loss functions such as cross-entropy are not a reliable indicator of generalization.

This leads to the crucial question of how generalization gap should be predicted from the training data and network parameters.

In this paper, we propose such a measure, and conduct extensive empirical studies on how well it can predict the generalization gap.

Our measure is based on the concept of margin distribution, which are the distances of training points to the decision boundary.

We find that it is necessary to use margin distributions at multiple layers of a deep network.

On the CIFAR-10 and the CIFAR-100 datasets, our proposed measure correlates very strongly with the generalization gap.

In addition, we find the following other factors to be of importance: normalizing margin values for scale independence, using characterizations of margin distribution rather than just the margin (closest distance to decision boundary), and working in log space instead of linear space (effectively using a product of margins rather than a sum).

Our measure can be easily applied to feedforward deep networks with any architecture and may point towards new training loss functions that could enable better generalization.

Generalization, the ability of a classifier to perform well on unseen examples, is a desideratum for progress towards real-world deployment of deep neural networks in domains such as autonomous cars and healthcare.

Until recently, it was commonly believed that deep networks generalize well to unseen examples.

This was based on empirical evidence about performance on held-out dataset.

However, new research has started to question this assumption.

Adversarial examples cause networks to misclassify even slightly perturbed images at very high rates BID9 BID22 .

In addition, deep networks can overfit to arbitrarily corrupted data BID10 , and they are sensitive to small geometric transformations BID2 BID6 .

These results have led to the important question about how the generalization gap (difference between train and test accuracy) of a deep network can be predicted using the training data and network parameters.

Since in all of the above cases, the training loss is usually very small, it is clear that existing losses such as cross-entropy cannot serve that purpose.

It has also been shown (e.g. in BID10 ) that regularizers such as weight decay cannot solve this problem either.

Consequently, a number of recent works BID21 BID4 BID23 BID1 have started to address this question, proposing generalization bounds based on analyses of network complexity or noise stability properties.

However, a thorough empirical assessment of these bounds in terms of how accurately they can predict the generalization gap across various practical settings is not yet available.

In this work, we propose a new quantity for predicting generalization gap of a feedforward neural network.

Using the notion of margin in support vector machines (Vapnik, 1995) and extension to deep networks BID5 , we develop a measure that shows a strong correlation with generalization gap and significantly outperforms recently developed theoretical bounds on Test Acc.: 55.2% Test Acc.: 70.6% Test Acc.: 85.1% Figure 1 : (Best seen as PDF) Density plots (top) and box plots (bottom) of normalized margin of three convolutional networks trained with cross-entropy loss on CIFAR-10 with varying test accuracy: left: 55.2%, middle: 70.6%, right: 85.1%.

The left network was trained with 20% corrupted labels.

Train accuracy of all above networks are close to 100%, and training losses close to zero.

The densities and box plots are computed on the training set.

Normalized margin distributions are strongly correlated with test accuracy (moving to the right as accuracy increases).

This motivates our use of normalized margins at all layers.

The (Tukey) box plots show the median and other order statistics (see section 3.2 for details), and motivates their use as features to summarize the distributions.

generalization 2 .

This is empirically shown by studying a wide range of deep networks trained on the CIFAR-10 and CIFAR-100 datasets.

The measure presented in this paper may be useful for a constructing new loss functions with better generalization.

Besides improvement in the prediction of the generalization gap, our work is distinct from recently developed bounds and margin definitions in a number of ways:1.

These recently developed bounds are typically functions of weight norms (such as the spectral, Frobenius or various mixed norms).

Consequently, they cannot capture variations in network topology that are not reflected in the weight norms, e.g. adding residual connections BID10 without careful additional engineering based on the topology changes.

Furthermore, some of the bounds require specific treatment for nonlinear activations.

Our proposed measure can handle any feedforward deep network.

2.

Although some of these bounds involve margin, the margin is only defined and measured at the output layer BID4 BID21 .

For a deep network, however, margin can be defined at any layer BID5 .

We show that measuring margin at a single layer does not suffice to capture generalization gap.

We argue that it is crucial to use margin information across layers and show that this significantly improves generalization gap prediction.

3.

The common definition of margin, as used in the recent bounds e.g. BID21 , or as extended to deep networks, is based on the closest distance of the training points to the decision boundary.

However, this notion is brittle and sensitive to outliers.

In contrast, we adopt margin distribution BID7 BID13 Zhang & Zhou, 2017; by looking at the entire distribution of distances.

This is shown to have far better prediction power.

4.

We argue that the direct extension of margin definition to deep networks BID5 , although allowing margin to be defined on all layers of the model, is unable to capture generalization gap without proper normalization.

We propose a simple normalization scheme that significantly boosts prediction accuracy.

The recent seminal work of BID10 has brought into focus the question of how generalization can be measured from training data.

They showed that deep networks can easily learn to fit randomly labeled data with extremely high accuracy, but with arbitrarily low generalization capability.

This overfitting is not countered by deploying commonly used regularizers.

The work of BID4 proposes a measure based on the ratio of two quantities: the margin distribution measured at the output layer of the network; and a spectral complexity measure related to the network's Lipschitz constant.

Their normalized margin distribution provides a strong indication of the complexity of the learning task, e.g. the distribution is skewed towards the origin (lower normalized margin) for training with random labels.

BID21 a) also develop bounds based on the product of norms of the weights across layers.

BID1 develop bounds based on noise stability properties of networks: more stability implies better generalization.

Using these criteria, they are able to derive stronger generalization bounds than previous works.

The margin distribution (specifically, boosting of margins across the training set) has been shown to correspond to generalization properties in the literature on linear models BID25 : they used this connection to explain the effectiveness of boosting and bagging techniques.

BID24 showed that it was important to control the complexity of a classifier when measuring margin, which calls for some type of normalization.

In the linear case (SVM), margin is naturally defined as a function of norm of the weights Vapnik (1995) .

In the case of deep networks, true margin is intractable.

Recent work BID5 proposed a linearization to approximate the margin, and defined the margin at any layer of the network.

BID26 provide another approximation to the margin based on the norm of the Jacobian with respect to the input layer.

They show that maximizing their approximations to the margin leads to improved generalization.

However, their analysis was restricted to margin at the input layer.

BID23 and BID15 propose a normalized cross-entropy measure that correlates well with test loss.

Their proposed normalized loss trades off confidence of predictions with stability, which leads to better correlation with test accuracy, leading to a significant lowering of output margin.

In this section, we introduce our margin-based measure.

We first explain the construction scheme for obtaining the margin distribution.

We then squeeze the distributional information of the margin to a small number of statistics.

Finally, we regress these statistics to the value of the generalization gap.

We assess prediction quality by applying the learned regression coefficients to predict the generalization gap of unseen models.

We will start with providing a motivation for using the margins at the hidden layers which is supported by our empirical findings.

SVM owes a large part of its success to the kernel that allows for inner product in a higher and richer feature space.

At its crux, the primal kernel SVM problem is separated into the feature extractor and the classifier on the extracted features.

We can separate any feed forward network at any given hidden layer and treat the hidden representation as a feature map.

From this view, the layers that precede this hidden layer can be treated as a learned feature extractor and then the layers that come after are naturally the classifier.

If the margins at the input layers or the output layers play important roles in generalization of the classifier, it is a natural conjecture that the margins at these hidden representations are also important in generalization.

In fact, if we ignore the optimization procedure and focus on a converged network, generalization theories developed on the input such as BID18 can be easily extended to the hidden layers or the extracted features.

First, we establish some notation.

Consider a classification problem with n classes.

We assume a classifier f consists of non-linear functions f i : X → R, for i = 1, . . .

, n that generate a prediction score for classifying the input vector x ∈ X to class i.

The predicted label is decided by the class with maximal score, i.e. i * = arg max i f i (x).

Define the decision boundary for each class pair (i, j)as: DISPLAYFORM0 Under this definition, the l p distance of a point x to the decision boundary D (i,j) can be expressed as the smallest displacement of the point that results in a score tie: DISPLAYFORM1 Unlike an SVM, computing the "exact" distance of a point to the decision boundary (Eq. 2) for a deep network is intractable BID31 .

In this work, we adopt the approximation scheme from Elsayed et al. FORMULA0 to capture the distance of a point to the decision boundary.

This a first-order Taylor approximation to the true distance Eq. 2.

Formally, given an input x to a network, denote its representation at the l th layer (the layer activation vector) by x l .

For the input layer, let l = 0 and thus x 0 = x. Then for p = 2, the distance of the representation vector x l to the decision boundary for class pair (i, j) is given by the following approximation: DISPLAYFORM2 Here f i (x l ) represents the output (logit) of the network logit i given x l .

Note that this distance can be positive or negative, denoting whether the training sample is on the "correct" or "wrong" side of the decision boundary respectively.

This distance is well defined for all (i, j) pairs, but in this work we assume that i always refers to the ground truth label and j refers to the second highest or highest class (if the point is misclassified).

The training data x induces a distribution of distances at each layer l which, following earlier naming convention BID7 BID13 , we refer to as margin distribution (at layer l).

For margin distribution, we only consider distances with positive sign (we ignore all misclassified training points).

Such design choice facilitates our empirical analysis when we transform our features (e.g. log transform); further, it has also been suggested that it may be possible to obtain a better generalization bound by only considering the correct examples when the classifier classifies a significant proportion of the training examples correctly, which is usually the case for neural networks BID3 .

For completeness, the results with negative margins are included in appendix Sec. 7.A problem with plain distances and their associated distribution is that they can be trivially boosted without any significant change in the way classifier separates the classes.

For example, consider multiplying weights at a layer by a constant and dividing weights in the following layer by the same constant.

In a ReLU network, due to positive homogeneity property BID15 , this operation does not affect how the network classifies a point, but it changes the distances to the decision boundary 4 .

To offset the scaling effect, we normalize the margin distribution.

Consider margin distribution at some layer l, and let x l k be the representation vector for training sample k. We compute the variance of each coordinate of {x l k } separately, and then sum these individual variances.

This quantity is called total variation of x l .

The square root of this quantity relates to the scale of the distribution.

That is, if x l is scaled by a factor, so is the square root of the total variation.

Thus, by dividing distances by the square root of total variation, we can construct a margin distribution invariant to scaling.

More concretely, the total variation is computed as: DISPLAYFORM3 i.e. the trace of the empirical covariance matrix of activations.

Using the total variation, the normalized margin is specified by:d DISPLAYFORM4 While the quantity is relatively primitive and easy to compute, Fig. 1 (top) shows that the normalizedmargin distributions based on Eq. 5 have the desirable effect of becoming heavier tailed and shifting to the right (increasing margin) as generalization gap decreases.

We find that this effect holds across a range of networks trained with different hyper-parameters.

Instead of working directly with the (normalized) margin distribution, it is easier to analyze a compact signature of that.

The moments of a distribution are a natural criterion for this purpose.

Perhaps the most standard way of doing this is computing the empirical moments from the samples and then take the n th root of the n th moment.

In our experiments, we used the first five moments.

However, it is a well-known phenomenon that the estimation of higher order moments based on samples can be unreliable.

Therefore, we also consider an alternate way to construct the distribution's signature.

Given a set of distances D = {d m } n m=1 , which constitute the margin distribution.

We use the median Q 2 , first quartile Q 1 and third quartile Q 3 of the normalized margin distribution, along with the two fences that indicate variability outside the upper and lower quartiles.

There are many variations for fences, but in this work, with IQR = Q 3 − Q 1 , we define the upper fence to be max({d m :d m ∈ D ∧d m ≤ Q 3 + 1.5IQR}) and the lower fence to be min( et al., 1978) .

These 5 statistics form the quartile description that summarizes the normalized margin distribution at a specific layer, as shown in the box plots of Fig. 1 .

We will later see that both signature representations are able to predict the generalization gap, with the second signature working slightly better.

DISPLAYFORM0 A number of prior works such as BID4 , BID21 , BID17 , Sun et al. (2015) , BID26 , and BID14 have focused on analyzing or maximizing the margin at either the input or the output layer of a deep network.

Since a deep network has many hidden layers with evolving representations, it is not immediately clear which of the layer margins is of importance for improving generalization.

Our experiments reveal that margin distribution from all of the layers of the network contribute to prediction of generalization gap.

This is also clear from Fig. 1 (top) : comparing the input layer (layer 0) margin distributions between the left and right plots, the input layer distribution shifts slightly left, but the other layer distributions shift the other way.

For example, if we use quartile signature, we have 5L components in this vector, where L is the total number of layers in the network.

We incorporate dependence on all layers simply by concatenating margin signatures of all layers into a single combined vector θ that we refer to as total signature.

Empirically, we found constructing the total signature based on four evenly-spaced layers (input, and 3 hidden layers) sufficiently captures the variation in the distributions and generalization gap, and also makes the signature agnostic to the depth of the network.

Our goal is to predict the generalization gap, i.e. the difference between training and test accuracy at the end of training, based on total signature θ of a trained model.

We use the simplest prediction model, i.e. a linear formĝ = a T φ(θ) + b, where a ∈ R dim(θ) and b ∈ R are parameters of the predictor, and φ : R → R is a function applied element-wise to θ.

Specifically, we will explore two choices of φ: the identity φ(x) = x and entry-wise log transform φ(x) = log(x), which correspond to additive and multiplicative combination of margin statistics respectively.

We do not claim this model is the true relation, but rather it is a simple model for prediction; and our results suggest that it is a surprisingly good approximation.

In order to estimate predictor parameters a, b, we generate a pool of n pretrained models (covering different datasets, architectures, regularization schemes, etc.

as explained in Sec. 4) each of which gives one instance of the pair θ, g (g being the generalization gap for that model).

We then find a, b by minimizing mean squared error: (a * , b DISPLAYFORM0 , where i indexes the i th model in the pool.

The next step is to assess the prediction quality.

We consider two metrics for this.

The first metric examines quality of predictions on unseen models.

For that, we consider a held-out pool of m models, different from those used to estimate (a, b), and compute the value of g on them viaĝ = a T φ(θ) + b. In order to quantify the discrepancy between predicted gapĝ and ground truth gap g we use the notion of coefficient of determination (R 2 ) BID8 : DISPLAYFORM1 R 2 measures what fraction of data variance can be explained by the linear model 5 (it ranges from 0 to 1 on training points but can be outside that range on unseen points).

To be precise, we use k-fold validation to study how the predictor can perform on held out pool of trained deep networks.

We use 90/10 split, fit the linear model with the training pool, and measure R 2 on the held out pool.

The performance is averaged over the 10 splits.

Since R 2 is now not measured on the training pool, it does not suffer from high data dimension and can be negative.

In all of our experiments, we use k = 10.

We provide a subset of residual plots and corresponding univariate F-Test for the experiments in the appendix (Sec. 8).

The F-score also indicates how important each individual variable is.

The second metric examines how well the model fits based on the provided training pool; it does not require a test pool.

To characterize this, we use adjustedR 2 (Glantz et al., 1990) defined as: DISPLAYFORM2 TheR 2 can be negative when the data is non-linear.

Note thatR 2 is always smaller than R 2 .

Intuitively,R 2 penalizes the model if the number of features is high relative to the available data points.

The closerR 2 is to 1, the better the model fits.

UsingR 2 is a simple yet effective method to test the fitness of linear model and is independent of the scale of the target, making it a more illustrative metric than residuals.

We tested our measure of generalization gapĝ, along with baseline measures, on a number of deep networks and architectures: nine-layer convolutional networks on CIFAR-10 (10 with input layer), and 32-layer residual networks on both CIFAR-10 and CIFAR-100 datasets.

The trained models and relevant Tensorflow BID0 code to compute margin distributions are released at https://github.com/google-research/google-research/tree/master/demogen

Using the CIFAR-10 dataset, we train 216 nine-layer convolutional networks with different settings of hyperparameters and training techniques.

We apply weight decay and dropout with different strengths; we use networks with and without batch norm and data augmentation; we change the number of hidden units in the hidden layers.

Finally, we also include training with and without corrupted labels, as introduced in Zhang et al. FORMULA0 ; we use a fixed amount of 20% corruption of the true labels.

The accuracy on the test set ranges from 60% to 90.5% and the generalization gap ranges from 1% to 35%.

In standard settings, creating neural network models with small generalization gap is difficult; in order to create sufficiently diverse generalization behaviors, we limit some models' capacities by large weight regularization which decreases generalization gap by lowering the training accuracy.

All networks are trained by SGD with momentum.

Further details are provided in the supplementary material (Sec. 6).For each trained network, we compute the depth-agnostic signature of the normalized margin distribution (see Sec. 3) .

This results in a 20-dimensional signature vector.

We estimate the parameters of the linear predictor (a, b) with the log transform φ(x) = log(x) and using the 20-dimensional signature vector θ.

FIG0 shows the resulting scatter plot of the predicted generalization gapĝ and the true generalization gap g. As it can be seen, it is very close to being linear across the range of generalization gaps, and this is also supported by theR 2 of the model, which is 0.96 (max is 1).As a first baseline method, we compare against the work of BID4 which provides one of the best generalization bounds currently known for deep networks.

This work also constructs a margin distribution for the network, but in a different way.

To make a fair comparison, we extract the same signature θ from their margin distribution.

Since their margin distribution can only be defined for the output layer, their θ is 5-dimensional for any network.

The resulting fit is shown in FIG0 .

It is clearly a poorer fit than that of our signature, with a significantly lowerR 2 of 0.72.For a fairer comparison, we also reduced our signature θ from 20 dimensions to the best performing 4 dimensions (even one dimension less than what we used for Bartlett's) by dropping 16 components in our θ.

This is shown in FIG0 (middle) and has aR 2 of 0.89, which is poorer than our complete 5 A simple manipulation shows that the prediction residual BID4 BID26 BID5 .

A indicates adjusted; kf indicates k-fold; mse indicates mean squared error in 10 −3 ; N indicates negative.

DISPLAYFORM0 θ but still significantly higher than that of BID4 .

In addition, we considered two other baseline comparisons: BID26 , where margin at input is defined as a function of the Jacobian of output (logits) with respect to input; and Elsayed et al. FORMULA0 where the linearized approximation to margin is derived (for the same layers where we use our normalized margin approximation).

To quantify the effect of the normalization, different layers, feature transformation etc., we conduct a number of ablation experiments with the following configuration: 1.

linear/log: Use signature transform of φ(x) = x or φ(x) = log(x); 2.

sl: Use signature from the single best layer (θ ∈ R 5 ); 3. sf: Use only the single best statistic from the total signature for all the layers (θ ∈ R 4 , individual layer result can be found in Sec. 7); 4. moment: Use the first 5 moments of the normalized margin distribution as signature instead of quartile statistics θ ∈ R 20 (Sec. 3); 5. spectral: Use signature of spectrally normalized margins from BID4 (θ ∈ R 5 ); 6. qrt:

Use all the quartile statistics as total signature θ ∈ R 20 (Sec. 3); 7.

best4: Use the 4 best statistics from the total signature (θ ∈ R 4 ); 8.

Jacobian: Use the Jacobian-based margin defined in Eq (39) of BID26 (θ ∈ R 5 ); 9.

LM: Use the large margin loss from BID5 at the same four layers where the statistics are measured (θ ∈ R 4 ); 10.

unnorm indicates no normalization.

In Table 1 , we list theR 2 from fitting models based on each of these scenarios.

We see that, both quartile and moment signatures perform similarly, lending support to our thesis that the margin distribution, rather than the smallest or largest margin, is of importance in the context of generalization.

On the CIFAR-10 dataset, we train 216 convolutional networks with residual connections; these networks are 32 layers deep with standard ResNet 32 topology BID10 .

Since it is difficult to train ResNet without activation normalization, we created generalization gap variation with batch normalization BID11 and group normalization (Wu & He, 2018) .

We further use different initial learning rates.

The range of accuracy on the test set ranges from 83% to 93.5% and generalization gap from 6% to 13.5%.

The residual networks were much deeper, and so we only chose 4 layers for feature-length compatibility with the shallower convoluational networks.

This design choice also facilitates ease of analysis and circumvents the dependency on depth of the models.

Table 1 shows theR 2 .

Note in the presence of residual connections that use convolution instead of identity and identity blocks that span more than one convolutional layers, it is not immediately clear how to properly apply the bounds of BID4 (third from last row) without morphing the topology of the architecture and careful design of reference matrices.

As such, we omit them for ResNet.

Fig. 3 (left) shows the fit for the resnet models, withR 2 = 0.87.

Fig. 3 (middle) and Fig. 3 (right) compare the log normalized density plots of a CIFAR-10 resnet and CIFAR-10 CNN.

The plots show that the Resnet achieves a better margin distribution, correlated with greater test accuracy, even though it was trained without data augmentation.

Log density (ResNet-32) Log Density (CNN) Figure 3 : (Best seen as PDF) Left: Regression model fit in log space for the full 20-dimensional feature space for 216 residual networks (R 2 = 0.87) on CIFAR-10; Middle: Log density plot of normalized margins of a particular residual network that achieves 91.7% test accuracy without data augmentation; Right: Log density plot of normalized margins of a CNN that achieves 87.2% with data augmentation.

We see that the resnet achieves larger margins, especially at the hidden layers, and this is reflected in the higher test accuracy.

On the CIFAR-100 dataset, we trained 324 ResNet-32 with the same variation in hyperparameter settings as for the networks for CIFAR-10 with one additional initial learning rate.

The range of accuracy on the test set ranges from 12% to 73% and the generalization gap ranges from 1% to 75%.

Table 1 showsR 2 for a number of ablation experiments and the full feature set.

FIG1 shows the fit of predicted and true generalization gaps over the networks (R 2 = 0.97).

FIG1 (middle) and FIG1 (right) compare a CIFAR-100 residual network and a CIFAR-10 residual network with the same architecture and hyperparameters.

Under these settings, the CIFAR-100 network achieves 44% test accuracy, whereas CIFAR-10 achieves 61%.

The resulting normalized margin density plots clearly reflect the better generalization achieved by CIFAR-10: the densities at all layers are wider and shifted to the right.

Thus, the normalized margin distributions reflect the relative "difficulty" of a particular dataset for a given architecture.

We have presented a predictor for generalization gap based on margin distribution in deep networks and conducted extensive experiments to assess it.

Our results show that our scheme achieves a high adjusted coefficient of determination (a linear regression predicts generalization gap accurately).

Specifically, the predictor uses normalized margin distribution across multiple layers of the network.

The best predictor uses quartiles of the distribution combined in multiplicative way (additive in log transform).

Compared to the strong baseline of spectral complexity normalized output margin BID4 , our scheme exhibits much higher predictive power and can be applied to any feedforward network (including ResNets, unlike generalization bounds such as BID4 BID21 BID1 ).

We also find that using hidden layers is crucial for the predictive power.

Our findings could be a stepping stone for studying new generalization theories and We use an architecture very similar to Network in Network BID16 ), but we remove all dropout and max pool from the network.

Layer Index Layer Type Output Shape 0 Input 32 × 32 × 3 1 3 × 3 convolution + stride 2 16 × 16 × 192 Residual plots for all explanatory variables, row: h0, h1, h2, h3, column: lower fence, Q 1 , Q 2 , Q 3 , upper fence.

lower fence is clipped because distance cannot be smaller than 0.

The residual is less evenly distributed as are in other two settings; this fact is well reflected in the cluster along the x axis and in theR 2 ; we speculate that this is due to not having diverse enough generalization gap in the models trained to cover the entire space of the "model" unlike in the other two settings.

3.45e-13 3.04e-16 9.21e-9 4.07e-4 6.59e-3 h3 4.14e-13 0.60 0.27 7.14e-3 2.4e-10 Table 6 : F score (top) and p-values (bottom) for all 20 variables.

Using p = 0.05, we see that the null hypotheses are not rejected for 4 of the variables.

We believe having a more diverse generalization behavior in the study will solve this problem.

Residual plots for all explanatory variables, row: h0, h1, h2, h3, column: lower fence, Q 1 , Q 2 , Q 3 , upper fence.

lower fence is clipped because distance cannot be smaller than 0.

The residual is fairly evenly distributed around 0.

There is one outlier in this experimental setting as shown in the plots.

9 APPENDIX: SOME OBSERVATIONS AND CONJECTURES Everythig here uses the full quartile description.

DISPLAYFORM0

We perform regression analysis with both base CNN and ResNet32 on CIFAR-10.

The resultinḡ R 2 = 0.91 and the k-fold R 2 = 0.88.

This suggests that the same coefficient works generally well across architectures provided they are trained on the same data.

Somehow, the distribution at the 3 locations of the networks are comparable even though the depths are vastly different.

We join all our experiment data and the resulting The resultingR 2 = 0.93 and the k-fold R 2 = 0.93.

It is perhaps surprising that a set of coefficient exists across both datasets and architectures.

We believe that the method developed here can be used in complementary with existing generalization bound; more sophisticated engineering of the predictor may be used to actually verify what kind of function the generalization bound should look like up to constant factor or exponents; it may be helpful for developing generalization bound tighter than the existing ones.

<|TLDR|>

@highlight

We develop a new scheme to predict the generalization gap in deep networks with high accuracy.

@highlight

Authors suggest using a geometric margin and layer-wise margin distribution for predicting generalization gap.

@highlight

Empirically shows an interesting connection between the proposed margin statistics and the generalization gap, which can be used to provide some prescriptive insights towards understanding generalization in deep neural nets. 