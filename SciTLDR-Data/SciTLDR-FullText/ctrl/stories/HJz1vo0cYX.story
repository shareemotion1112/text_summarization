We propose a generic framework to calibrate accuracy and confidence (score) of a prediction through stochastic inferences in deep neural networks.

We first analyze relation between variation of multiple model parameters for a single example inference and variance of the corresponding prediction scores by Bayesian modeling of stochastic regularization.

Our empirical observation shows that accuracy and score of a prediction are highly correlated with variance of multiple stochastic inferences given by stochastic depth or dropout.

Motivated by these facts, we design a novel variance-weighted confidence-integrated loss function that is composed of two cross-entropy loss terms with respect to ground-truth and uniform distribution, which are balanced by variance of stochastic prediction scores.

The proposed loss function enables us to learn deep neural networks that predict confidence calibrated scores using a single inference.

Our algorithm presents outstanding confidence calibration performance and improves classification accuracy with two popular stochastic regularization techniques---stochastic depth and dropout---in multiple models and datasets; it alleviates overconfidence issue in deep neural networks significantly by training networks to achieve prediction accuracy proportional to confidence of prediction.

Deep neural networks have achieved remarkable performance in various tasks, but have critical limitations in reliability of their predictions.

One example is that inference results are often overly confident even for unseen or tricky examples; the maximum scores of individual predictions are very high even for out-of-distribution examples and consequently distort interpretation about the predictions.

Since many practical applications including autonomous driving, medical diagnosis, and machine inspection require accurate uncertainty estimation as well as high prediction accuracy for each inference, such an overconfidence issue makes deep neural networks inappropriate to be deployed for real-world problems in spite of their impressive accuracy.

Regularization is a common technique in training deep neural networks to avoid overfitting problems and improve generalization accuracy BID18 ; BID6 ; BID7 .

However, their objectives are not directly related to generating score distributions aligned with uncertainty of individual predictions.

In other words, existing deep neural networks are inherently poor at calibrating prediction accuracy and confidence.

Our goal is to learn deep neural networks that are able to estimate accuracy and uncertainty of each prediction at the same time.

Hence, we propose a generic framework to calibrate prediction score (confidence) with accuracy in deep neural networks.

Our algorithm starts with an observation that variance of prediction scores measured from multiple stochastic inferences is highly correlated with accuracy and confidence of the prediction based on the average score, where we employ stochastic regularization techniques such as stochastic depth or dropout to obtain multiple stochastic inference results.

We also interpret stochastic regularization as a Bayesian model, which shows relation between stochastic modeling and stochastic inferences of deep neural networks.

By exploiting these properties, we design a loss function to enable deep neural network to predict confidence-calibrated scores based only on a single prediction, without stochastic inferences.

Our contribution is summarized below:• We provide a generic framework to estimate uncertainty of a prediction based on stochastic inferences in deep neural networks, which is motivated by empirical observation and theoretical analysis.• We design a variance-weighted confidence-integrated loss function in a principled way without hyper-parameters, which enables deep neural networks to produce confidencecalibrated predictions even without stochastic inferences.• The proposed framework presents outstanding performance to reduce overconfidence issue and estimate accurate uncertainty in various architectures and datasets.

The rest of the paper is organized as follows.

We first discuss prior research related to our algorithm, and describe theoretical background for Bayesian interpretation of our approach in Section 2 and 3, respectively.

Section 4 presents our confidence calibration algorithm through stochastic inferences, and Section 5 illustrates experimental results.

Uncertainty estimation is a critical problem in deep neural networks and receives growing attention from machine learning community.

Bayesian approach is a common tool to provide a mathematical framework for uncertainty estimation in deep neural networks.

However, exact Bayesian inference is not tractable in deep neural networks due to its high computational cost, and various approximate inference techniques-MCMC Neal (1996), Laplace approximation MacKay (1992) and variational inference BID0 BID3 and deep ensembles BID8 .

All the post-processing methods require a hold-out validation set to adjust prediction scores after training, and the ensemble-based technique employs multiple models to estimate uncertainty.

Stochastic regularization is a common technique to improve generalization performance by injecting random noise to deep neural networks.

The most notable method is BID18 , which randomly drops their hidden units by multiplying Bernoulli random noise.

There exist several variants, for example, dropping weights BID21 or skipping layers BID6 .

Most stochastic regularization methods exploit stochastic inferences during training, but perform deterministic inferences using the whole network during testing.

On the contrary, we also use stochastic inferences to obtain diverse and reliable outputs during testing.

Although the following works do not address uncertainty estimation, their main idea is relevant to our objective.

Label smoothing BID19 encourages models to be less confident, by preventing a network from assigning the full probability to a single class.

The same loss function is discussed to train confidence-calibrated classifiers in BID9 , but it focuses on how to discriminate in-distribution and out-of-distribution examples, rather than estimating uncertainty or alleviating miscalibration of in-distribution examples.

On the other hand, BID15 claims that blind label smoothing and penalizing entropy enhances accuracy by integrating loss functions with the same concept with BID19 ; BID9 , but improvement is marginal in practice.

This section describes Bayesian interpretation of stochastic regularization in deep neural networks, and discusses relation between stochastic regularization and uncertainty modeling.

Deep neural networks are prone to overfit due to their large number of parameters, and various regularization techniques including weight decay, dropout BID18 , and batch normalization BID7 have been employed to alleviate the issue.

One popular class of regularization techniques is stochastic regularization, which introduces random noise to a network for perturbing its inputs or weights.

We focus on the multiplicative binary noise injection, where random binary noise is applied to the inputs or weights by elementwise multiplication since such stochastic regularization techniques are widely used BID18 ; BID21 ; BID6 .

Note that input perturbation can be reformulated as weight perturbation.

For example, dropout-binary noise injection to activations-is intertpretable as weight perturbation that masks out all the weights associated with the dropped inputs.

Therefore, if a classification network modeling p(y|x, θ) with parameters θ is trained with stochastic regularization methods by minimizing the cross entropy loss, its objective can be defined by DISPLAYFORM0 whereω i = θ i is a set of perturbed parameters by elementwise multiplication with random noise sample i ∼ p( ), and (x i , y i ) ∈ D is a pair of input and output in training dataset D. Note thatω i is a random sample from p(ω) given by the product of the deterministic parameter θ and a random noise i .At inference time, the network is parameterized by the expectation of the perturbed parameters, DISPLAYFORM1

Given the dataset D with N examples, Bayesian objective is to estimate the posterior distribution of the model parameter, denoted by p(ω|D), to predict a label y for an input x, which is given by DISPLAYFORM0 A common technique for the posterior estimation is variational approximation, which introduces an approximate distribution q θ (ω) and minimizes Kullback-Leibler (KL) divergence with the true posterior D KL (q θ (ω)||p(ω|D)) as follows: DISPLAYFORM1 The intractable integral and summation over the entire dataset in Equation 4 is approximated by Monte Carlo method and mini-batch optimization resulting in DISPLAYFORM2 DISPLAYFORM3 is a sample from the approximate distribution, S is the number of samples, and M is the size of a mini-batch.

Note that the first term is data likelihood and the second term is divergence of the approximate distribution with respect to the prior distribution.

Suppose that we train a classifier with 2 regularization by a stochastic gradient descent method.

Then, the loss function in Equation 1 is rewritten aŝ DISPLAYFORM0 where 2 regularization is applied to the deterministic parameters θ with weight λ.

Optimizing this loss function is equivalent to optimizing Equation 5 if there exists a proper prior p(ω) and q θ (ω) is approximated as a Gaussian mixture distribution BID1 .

Note that Gal & DISPLAYFORM1 Following BID1 and BID20 , we estimate the predictive mean and uncertainty by Monte Carlo approximation by drawing binary noise samples DISPLAYFORM2 as DISPLAYFORM3 where y = (y 1 , . . .

, y C ) denotes a vector of C class labels.

Note that the binary noise samples realize stochastic inferences such as stochastic depth and dropout by elementwise multiplication with model parameter θ.

Equation 8 means that the average prediction and its variance can be computed directly from multiple stochastic inferences.

We present a novel confidence calibration technique for prediction in deep neural networks, which is given by a variance-weighted confidence-integrated loss function.

We present our observation that variance of multiple stochastic inferences is closely related to accuracy and confidence of predictions, and provide an end-to-end training framework for confidence self-calibration.

Then, prediction accuracy and uncertainty are directly accessible from the predicted scores obtained from a single forward pass.

This section presents our observation from stochastic inferences and technical details about our confidence calibration technique.

Equation FORMULA9 suggests that variation of models 1 is correlated to variance of multiple stochastic predictions for a single example.

In other words, by observing variation of multiple stochastic inferences, we can estimate accuracy and uncertainty of the prediction given by average of the stochastic inferences corresponding to an example.

FIG0 presents how variance of multiple stochastic inferences given by stochastic depth or dropout is related to accuracy and confidence of the corresponding average prediction, where the confidence is measured by the maximum score of the average prediction.

In the figure, accuracy and score of each bin are computed with the examples belonging to the corresponding bin of the normalized variance.

We present results from CIFAR-100 with ResNet-34 and VGGNet with 16 layers.

The histograms illustrate the strong correlation between the predicted variance and the reliability-accuracy and confidence-of a prediction, and between accuracy and prediction.

These results suggest that one can disregard examples based on their prediction variances.

Note that variance computation with more stochastic inferences provides more reliable estimation of accuracy and confidence.

We first design a simple loss function for accuracy-score calibration by augmenting a confidenceintegrated loss L U to the standard cross-entropy loss term, which is given by DISPLAYFORM0 where H is the cross entropy loss function, p GT is the ground-truth distribution, p(y|x i , θ) is the predicted distribution with model parameter θ, U(y) is the uniform distribution, and ξ is a constant.

The loss denoted by L 1 (·) is determined based on cross-entropy with the ground-truths and KLdivergence from the uniform distribution.

The main idea of this loss function is to regularize with the uniform distribution by expecting the score distributions of uncertain examples to be flattened first while the distributions of confident ones remain intact, where the impact of the confidenceintegrated loss term is controlled by a global hyper-parameter β.

The proposed loss function is also employed in BID15 to regularize deep neural networks and improve classification accuracy.

However, BID15 does not discuss confidence calibration issues.

On the other hand, BID9 discusses the same loss function but focuses on differentiating between in-distribution and out-of-distribution examples by measuring loss of each example based only on one of the two loss terms depending on its origin.

Contrary to these approaches, we employ the loss function in Equation 9 for estimating prediction confidence in deep neural networks.

Although the proposed loss makes sense intuitively, blind selection of a constant β limits its generality.

Hence, we propose a more sophisticated confidence loss term by leveraging variance of multiple stochastic inferences.

The strong correlation of accuracy and confidence with predicted variance observed in FIG0 shows great potential to make confidence-calibrated prediction by stochastic inferences.

However, variance computation involves multiple stochastic inferences by executing multiple forward passes.

Note that this property incurs additional computational cost and may produce inconsistent results.

To overcome these limitations, we propose a generic framework for training accuracy-score calibration networks whose prediction score from a single forward pass directly provides confidence of the prediction.

In this framework, we combine two complementary loss terms as in Equation 9, but they are balanced by the variance measured by multiple stochastic inferences.

Our variance-weighted confidence-integrated loss L(·) for the whole training data (x i , y i ) ∈ D is defined by a linear interpolation of the standard cross-entropy loss with ground-truth L GT (·) and the cross-entropy with the uniform distribution L U (·), which is formally given by DISPLAYFORM0 where α i ∈ [0, 1] is a normalized variance,ω i,j (= θ i,j ) is a sampled model parameter with binary noise for stochastic prediction, T is the number of stochastic inferences, and ξ i is a constant.

The two terms in our variance-weighted confidence-integrated loss pushes the network toward opposite directions; the first term encourages the network to produce a high score for the ground truth label while the second term forces the network to predict the uniform distribution.

These terms are linearly interpolated by a balancing coefficient α i , which is the normalized variance of individual example obtained by multiple stochastic inferences.

Note that the normalized variance α i is unique for each training example and is used to measure model uncertainty.

Therefore, optimizing our loss function produces gradient signals, forcing the prediction toward the uniform distribution for the examples with high uncertainty derived by high variance while intensifying prediction confidence of the examples with low variance.

After training models in our framework, prediction of each testing example is made by a single forward pass.

Unlike the ordinary models, however, a prediction score of our model is well-calibrated and represents confidence of the prediction, which means that we can rely more on the predictions with high scores.

There are several score calibration techniques BID3 ; BID23 ; Naeini et al. FORMULA0 ; BID14 by adjusting confidence scores through postprocessing, among which BID3 proposes a method to calibrate confidence of predictions by scaling logits of a network using a global temperature τ .

The scaling is performed before applying the softmax function, and τ is trained with validation dataset.

As discussed in BID3 , this simple technique is equivalent to maximize entropy of the output distribution p(y i |x i ).

It is also identical to minimize KL-divergence D KL (p(y i |x i )||U(y)) because DISPLAYFORM0 where ξ c is a constant.

We can formulate another confidence-integrated loss with the entropy as DISPLAYFORM1 where γ is a constant.

Equation 12 suggests that temperature scaling in BID3 is closely related to our framework.

We choose two most widely adapted deep neural network architectures: ResNet and VGGNet.

VGG architecture follows BID17 , where we employ dropout BID18 before every fc layer except for the classification layer.

In ResNet, instead of stacking conv layers directly, outputs of residual blocks are added to the input feature representation by residual connections as proposed in BID4 .

Stochastic depth BID6 is used for stochastic regularization in ResNet.

Note that, as discussed in Section 3.3, both dropout and stochastic depth inject multiplicative binary noise to within-layer activations or residual blocks, they are equivalent to noise injection into network weights.

Hence, training with 2 regularization term enables us to interpret stochastic depth and dropout by Bayesian models.

We evaluate the proposed framework on two benchmarks, Tiny ImageNet and CIFAR-100.

Tiny ImageNet contains 64 × 64 images with 200 object labels whereas CIFAR-100 has 32 × 32 images of 100 objects.

There are 500 training images per class in both datasets.

For testing, we use the validation set of Tiny ImageNet and the test set of CIFAR-100, which contain 50 and 100 images per class, respectively.

To use the same network for two benchmarks, we resize images in Tiny ImageNet into 32 × 32.All networks are trained with stochastic gradient decent with the momentum of 0.9 for 300 epochs.

We set the initial learning rate to 0.1 and exponentially decay it with factor of 0.2 at epoch 60, 120, 160, 200 and 250.

Each batch consists of 64 and 256 training examples for ResNet and VGG architectures, respectively.

To train networks with the proposed variance-weighted confidence-integrated loss, we draw T samples for each input image by default, and compute the normalized variance α by running T forward passes.

The number of samples T is set to 5.

The normalized variance is estimated based on the variance of Bhattacharyya coefficients between individual predictions and the average prediction.

The trained models with the variance-weighted confidence-integrated (VWCI) loss are compared to the models with confidence-integrated (CI) losses for several different constant β's.

We measure classification accuracy and expected calibration error (ECE) of the trained models.

While classification accuracy shows regularization effect of the confidence-integrated loss term, ECE summarizes miscalibration of a model by measuring discrepancy between confidence and accuracy.

Specifically, let B m be a set of indices of test examples whose scores for the ground-truth labels fall into the score interval ( DISPLAYFORM0 , where M is the number of bins.

Then, ECE is formally defined by DISPLAYFORM1 where N is the number of the test samples.

Also, accuracy and confidence of each bin are given by DISPLAYFORM2 whereŷ i and y i are predicted and true label of the i-th example and p i is its predicted confidence.

TAB1 presents results of ResNet-34 and VGG-16 on both datasets.

We observe that baseline methods with stochastic inferences reduce calibration error and the reduction becomes more significant in proportion to number of inferences.

These results imply benefit of stochastic inference for confidence calibration, and reflect performance of methods by multiplicative noise in BID1 ; BID11 .

The models trained with VWCI loss consistently outperform baselines and is competitive to models with CI loss on both classification accuracy and confidence calibration.

Stochastic inference and variance-driven weight allow us to measure uncertainty for each instance, and enable two oppositional terms to be well balanced by the measured uncertainty.

The confidence loss term regularizes the network by forcing predictions to uniform distribution, and the proper estimation of its coefficient leads to accuracy gain and confidence calibration.

Note that, by assigning a low coefficient to the loss term for a confident example, the network allows the example to remain confident whereas a high weight for an uncertain example reduces confidence of the prediction.

The CI loss has a confidence loss term with fixed coefficient β.

The networks trained with proper β show impressive improvement on both criteria, but their performance is sensitive to choice of β as this strategy ignores predictive uncertainty for confidence loss; an inappropriate choice of β even worsens accuracy and calibration error, e.g., ResNet-34 trained with CI[β = 0.01] on Tiny ImageNet.

Also, there seems to be no single β that is globally optimal across architectures and benchmark datasets.

For instance, training the network with CI[β = 0.01] on Tiny ImageNet gives the worst accuracy with ResNet-34 and the best accuracy with VGG-16.

In the experiments, the CI loss often works well on CIFAR-100 due to high accuracy.

The majority of examples are classified correctly and the overconfident property of deep neural networks do little harm for confidence calibration.

Specifically, CI loss sometimes achieves slightly better performance than VWCI with a certain fixed coefficient β because the measured normalized variance by stochastic inferences and its range are small.

On the other hand, in Tiny ImageNet dataset, performance of VMCI is consistently better than CI because Tiny ImageNet is substantially more challenging than CIFAR-100.A critical benefit of our variance-driven weight in the VWCI loss is the capability to maintain examples with high accuracy and high confidence.

This is an important property for building real-world decision making systems with confidence interval, where the decisions should be both highly accurate and confident.

FIG1 illustrates coverage of test examples varying the confidence threshold, and VWCI shows better coverage than CI because CI pushes all instances to uniform with the same strength β regardless of their uncertainty unlike VWCI.

It is also notable that β for the best coverage is different from that for the best accuracy and ECE whereas VWCI balances these based on the predictive uncertainty.

These results suggest that using the predictive uncertainty for balancing the terms is preferable over setting a constant coefficient in our loss function.

More experimental results are presented in the supplementary document.

We presented a generic framework for uncertainty estimation of a prediction in deep neural networks by calibrating accuracy and score based on stochastic inferences.

Based on Bayesian interpretation of stochastic regularization and our empirical observation results, we claim that variation of multiple stochastic inferences for a single example is a crucial factor to estimate uncertainty of the average prediction.

Motivated by this fact, we design the variance-weighted confidence-integrated loss to learn confidence-calibrated networks and enable uncertainty to be estimated by a single prediction.

The proposed algorithm is also useful to understand existing confidence calibration methods in a unified way, and we compared our algorithm with other variations within our framework to analyze their characteristics.

<|TLDR|>

@highlight

We propose a framework to learn confidence-calibrated networks by designing a novel loss function that incorporates predictive uncertainty estimated through stochastic inferences.