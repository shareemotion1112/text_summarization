In order for machine learning to be deployed and trusted in many applications, it is crucial to be able to reliably explain why the machine learning algorithm makes certain predictions.

For example, if an algorithm classifies a given pathology image to be a malignant tumor, then the doctor may need to know which parts of the image led the algorithm to this classification.

How to interpret black-box predictors is thus an important and active area of research.

A fundamental question is: how much can we trust the interpretation itself?

In this paper, we show that interpretation of deep learning predictions is extremely fragile in the following sense:  two perceptively indistinguishable inputs with the same predicted label can be assigned very different}interpretations.

We systematically characterize the fragility of the interpretations generated by several widely-used feature-importance interpretation methods (saliency maps, integrated gradient, and DeepLIFT) on ImageNet and CIFAR-10.

Our experiments show that even small random perturbation can change the feature importance and new systematic perturbations can lead to dramatically different interpretations without changing the label.

We extend these results to show that interpretations based on exemplars (e.g. influence functions) are similarly fragile.

Our analysis of the geometry of the Hessian matrix gives insight on why fragility could be a fundamental challenge to the current interpretation approaches.

Predictions made by machine learning algorithms play an important role in our everyday lives and can affect decisions in technology, medicine, and even the legal system (Rich, 2015; Obermeyer & Emanuel, 2016) .

As the algorithms become increasingly complex, explanations for why an algorithm makes certain decisions are ever more crucial.

For example, if an AI system predicts a given pathology image to be malignant, then the doctor would want to know what features in the image led the algorithm to this classification.

Similarly, if an algorithm predicts an individual to be a credit risk, then the lender (and the borrower) might want to know why.

Therefore having interpretations for why certain predictions are made is critical for establishing trust and transparency between the users and the algorithm (Lipton, 2016) .Having an interpretation is not enough, however.

The explanation itself must be robust in order to establish human trust.

Take the pathology predictor; an interpretation method might suggest that a particular section in an image is important for the malignant classification (e.g. that section could have high scores in saliency map).

The clinician might then focus on that section for investigation, treatment or even look for similar features in other patients.

It would be highly disconcerting if in an extremely similar image, visually indistinguishable from the original and also classified as malignant, a very different section is interpreted as being salient for the prediction.

Thus, even if the predictor is robust (both images are correctly labeled as malignant), that the interpretation is fragile would still be highly problematic in deployment.

Our contributions.

The fragility of prediction in deep neural networks against adversarial attacks is an active area of research BID4 Kurakin et al., 2016; Papernot et al., 2016; Moosavi-Dezfooli et al., 2016) .

In that setting, fragility is exhibited when two perceptively indistinguishable images are assigned different labels by the neural network.

In this paper, we extend the definition of fragility to neural network interpretation.

More precisely, we define the interpretation of neural network to be fragile if perceptively indistinguishable images that have the same prediction label by the neural network are given substantially different interpretations.

We systematically The fragility of feature-importance maps.

We generate feature-importance scores, also called saliency maps, using three popular interpretation methods: simple gradient (a), DeepLIFT (b) and integrated gradient (c).

The top row shows the the original images and their saliency maps and the bottom row shows the perturbed images (using the center attack with = 8, as described in Section 3) and the corresponding saliency maps.

In all three images, the predicted label has not changed due to perturbation; in fact the network's (SqueezeNet) confidence in the prediction has actually increased.

However, the saliency maps of the perturbed images are meaningless.investigate two classes of interpretation methods: methods that assign importance scores to each feature (this includes simple gradient (Simonyan et al., 2013) , DeepLift (Shrikumar et al., 2017) , and integrated gradient (Sundararajan et al., 2017) ), as well as a method that assigns importances to each training example: influence functions (Koh & Liang, 2017) .

For both classes of interpretations, we show that targeted perturbations can lead to dramatically different interpretations ( FIG0 ).Our findings highlight the fragility of interpretations of neural networks, which has not been carefully considered in literature.

Fragility directly limits how much we can trust and learn from the interpretations.

It also raises a significant new security concern.

Especially in medical or economic applications, users often take the interpretation of a prediction as containing causal insight ("this image is a malignant tumor likely because of the section with a high saliency score").

An adversary could minutely manipulate the input to draw attention away from relevant features or onto his/her desired features.

Such attacks might be especially hard to detect as the actual labels have not changed.

While we focus on image data here because most of the interpretation methods have been motivated by images, the fragility of neural network interpretation could be a much broader problem.

Fig. 2 illustrates the intuition that when the decision boundary in the input feature space is complex, as is the case with deep nets, a small perturbation in the input can push the example into a region with very different loss contours.

Because the feature importance is closely related to the gradient which is perpendicular to the loss contours, the importance scores can also be dramatically different.

We provide additional analysis of this in Section 5.

This first class of methods explains predictions in terms of the relative importance of features in a test input sample.

Given the sample x t ∈ R d and the network's prediction l, we define the score of the predicted class S l (x t ) to be the value of the l-th output neuron right before the softmax operation.

We take l to be the class with the max score; i.e. the predicted class.

Feature-importance methods seek to find the dimensions of input data point that most strongly affect the score, and in doing so, these methods assign an absolute saliency score to each input feature.

Here we normalize the scores for each image by the sum of the saliency scores across the features.

This ensures that any perturbations that we design change not the absolute feature saliencies (which may still preserve DISPLAYFORM0 This training point has a large influence on the loss at +

This training point has a large influence on the loss at Figure 2 : Intuition for why interpretation is fragile.

Consider a test example x t ∈ R 2 (black dot) that is slightly perturbed to a new position x t + δ in input space (gray dot).

The contours and decision boundary corresponding to a loss function (L) for a two-class classification task are also shown, allowing one to see the direction of the gradient of the loss with respect to the input space.

Neural networks with many parameters have decision boundaries that are roughly piecewise linear with many transitions.

We illustrate that points near the transitions are especially fragile to interpretability-based analysis.

A small perturbation to the input changes the direction of ∇ x L from being in the direction of x 1 to being in the direction of x 2 , directly affecting feature-importance analyses.

Similarly, a small perturbation to the test image changes which training image, when up-weighted, has the largest influence on L, directly affecting exemplar-based analysis.the ranking of different features), but their relative values.

We summarize three different methods to calculate the normalized saliency score, denoted by R(x t ).Simple gradient method Introduced in BID2 and applied to deep neural networks in Simonyan et al. (2013) , the simple gradient method applies a local linear approximation of the model to detect the sensitivity of the score to perturbing each of the input dimensions.

Given input x t ∈ R d , the score is defined as: DISPLAYFORM0 Integrated gradients A significant drawback of the simple gradient method is the saturation problem discussed by Shrikumar et al. (2017); Sundararajan et al. (2017) .

Consequently, Sundararajan et al. (2017) introduced the integrated gradients method where the gradients of the score with respect to M scaled versions of the input are summed and then multiplied by the input.

Letting x 0 be the reference point and ∆x t = x t − x 0 , the feature importance vector is calculated by: DISPLAYFORM1 which is then normalized for our analysis.

Here the absolute value is taken for each dimension.

DeepLIFT DeepLIFT is an improved version of layer-wise relevance propagation (LRP) method BID1 .

LRP methods decompose the score S l (x t ) backwards through the neural network.

In each step, the score from the last layer is propagated to the previous layer, with the score being divided proportionally to magnitude of the activations of the neurons in the previous layer.

The scores are propagated to the input layer, and the result is a relevance score assigned to each of the input dimensions.

DeepLIFT (Shrikumar et al., 2017 ) defines a reference point in the input space and propagates relevance scores proportionally to the changes in the neuronal activations from the reference.

We use DeepLIFT with the Rescale rule; see Shrikumar et al. (2017) for details.

A complementary approach to interpreting the results of a neural network is to explain the prediction of the network in terms of its training examples, {(x i , y i )}.

DISPLAYFORM0 where z i def = (x i , y i ) and z t is defined analogously.

L(z,θ) is the loss of the network with parameters set toθ for the (training or test) data point z. Hθ DISPLAYFORM1 is the empirical Hessian of the network calculated over the training examples.

The training examples with the highest influence are understood as explaining why a network made a particular prediction for a test example.

We consider two natural metrics for quantifying the similarity between interpretations for two different images.

As shown in Fig. 3 , these metrics can be used to evaluate the effectiveness of a targeted attack on interpretability.• Spearman's rank order correlation: Because interpretation methods rank all of the features or training examples in order of importance, it is natural to use the rank correlation (Spearman, 1904) to compare the similarity between interpretations.• Top-k intersection:

In many settings, only the most important features or interpretations are of interest.

In these settings, we can compute the size of the intersection of the k most important features before and after perturbation.

Problem statement For a given fixed neural network N and input data point x t , the feature importance and influence function methods that we have described produce an interpretation I(x t ; N ).

For feature importance, I(x t ; N ) is a vector of feature scores; for influence function I(x t ; N ) is a vector of scores for training examples.

We would like to devise efficient perturbations to change the interpretability of a test image.

Yet, the perturbations should be visually imperceptible and should not change the label of the prediction.

Formally, we define the problem as: DISPLAYFORM0 where D(·) measures the change in interpretation (e.g. how many of the top-k pixels are no longer the top-k pixels of the saliency map after the perturbation) and > 0 constrains the norm of the perturbation.

In this paper, we carry out three kinds of input perturbations.

Random sign perturbation As a baseline, we generate random perturbations in which each pixel is randomly perturbed by ± .

This is used to measure robustness against untargeted perturbations.

Iterative attacks against feature-importance methods In Algorithm 1 we define two adversarial attacks against feature-importance methods, each of which consists of taking a series of steps in the direction that maximizes a differentiable dissimilarity function between the original and perturbed interpretation.(1) The top-k attack seeks to perturb the saliency map by decreasing the relative importance of the k most important features of the original image.

(2) When the input data are images, the center of mass of the saliency map often captures the user's attention.

The mass-center attack is designed to result in the maximum spatial displacement of the center of mass of the saliency scores.

Both of these attacks can be applied to any of the three feature-importance methods.

We can obtain effective adversarial images for influence functions without resorting to interative procedures.

We linearize FORMULA3 around the values of the current inputs and parameters.

If we further constrain the L ∞ norm of the perturbation to , we obtain an optimal single-step perturbation: DISPLAYFORM0 Algorithm 1 Iterative Feature-Importance Attacks Input: test image x t , maximum norm of perturbation , normalized feature importance function R(·), number of iterations P , step size α Define a dissimilarity function D to measure the change between interpretations of two images: DISPLAYFORM1 where B is the set of the k largest dimensions a of R(x t ), and C(·) is the center of saliency mass b .

DISPLAYFORM2 Perturb the test image in the direction of signed gradient c of the dissimilarity function: DISPLAYFORM3 If needed, clip the perturbed input to satisfy the norm constraint: DISPLAYFORM4 . .

, x P }, return the element with the largest value for the dissimilarity function and the same prediction as the original test image.a The goal is to damp the saliency scores of the k features originally identified as the most important.

b The center of mass is defined for a W × H image as: DISPLAYFORM5 In some networks, such as those with ReLUs, this gradient is always 0.

To attack interpretability in such networks, we replace the ReLU activations with their smooth approximation (softplus) when calculating the gradient and generate the perturbed image using this approximation.

The perturbed images that result are effective adversarial attacks against the original ReLU network, as discussed in Section 4.The attack we use consists of applying the negative of the perturbation in (4) to decrease the influence of the 3 most influential training images of the original test image 1 .

Of course, this affects the influence of all of the other training images as well.

We follow the same setup for computing the influence function as was done by the authors of Koh & Liang (2017) .

Because the influence is only calculated with respect to the parameters that change during training, we calculate the gradients only with respect to parameters in the final layer of our network (InceptionNet, see Section 4).

This makes it feasible for us to compute (4) exactly, but it gives us the perturbation of the input into the final layer, not the first layer.

So, we use standard back-propagation to calculate the corresponding gradient for the input test image.

We then take the sign of this gradient as the perturbation and clip the image to produce the adversarial test image.

Data sets and models To evaluate the robustness of feature-importance methods, we used two image classification data sets: ILSVRC2012 (ImageNet classification challenge data set) (Russakovsky et al., 2015) and CIFAR-10 (Krizhevsky, 2009) .

For the ImageNet classification data set, we used a pre-trained SqueezeNet 2 model introduced by BID5 .

For the CIFAR-10 data set we trained our own convolutional network, whose architecture is presented in Appendix A.1 In other words, we generate the perturbation given by: − sign( DISPLAYFORM0 where z (i) is the i th most influential training image of the original test image.

2 https://github.com/rcmalli/keras-squeezenet For both data sets, the results are examined using simple gradient, integrated gradients, and DeepLIFT feature importance methods.

For DeepLIFT, we used the pixel-wise and the channelwise mean images as the CIFAR-10 and ImageNet reference points respectively.

For the integrated gradients method, the same references were used with parameter M=100.

We ran all iterative attack algorithms for P = 300 iterations with step size α = 0.5.To evaluate the robustness of influence functions, we followed a similar experimental setup to that of the original authors: we trained an InceptionNet v3 with all but the last layer frozen (the weights were pre-trained on ImageNet and obtained from Keras 3 ).

The last layer was trained on a binary flower classification task (roses vs. sunflowers), using a data set consisting of 1,000 training images 4 .

This data set was chosen because it consisted of images that the network had not seen during pre-training on ImageNet.

The network achieved a validation accuracy of 97.5% on this task.

Results for feature-importance methods From the ImageNet test set, 512 correctly-classified images were randomly sampled for evaluation purposes.

Examples of the mass-center attack against three feature importance methods were presented in FIG0 Figure 3: Evaluation metrics vs subjective change We generate snapshots of the perturbed image and its simple gradient saliency maps along with iterations of mass-center attack to visualize the gradual change in saliency map with its corresponding the rank-correlation and top-1000 intersection metrics.

In FIG1 , we present results aggregated over all 512 images.

We compare different attack methods using top-1000 intersection and rank correlation methods.

In all the images, the attacks does not change the original predicted label of the image.

Random sign perturbation already causes significant changes in both top-1000 intersection and rank order correlation.

For example, with L ∞ = 8, on average, there is less than 30% overlap in the top 1000 most salient pixels between the original and the randomly perturbed images across all three of interpretation methods.

This suggests that the saliency of individual or small groups of pixels can be extremely fragile to the input and should be interpreted with caution.

With targeted perturbations, we observe more dramatic fragility.

Even with a perturbation of L ∞ = 2, the interpretations change significantly.

Both iterative attack algorithms have similar effects on feature importance of test images when measured on the basis of rank correlation or top-1000 intersection.

In Appendix D, we show an additional metric: the displacement of the center of mass between the original and perturbed saliency maps.

Empirically, we find this metric to correspond most strongly with intuitive perceptions of the similarity between two saliency maps.

Not surprisingly, we found that the center attack method was more effective than the top-k attack at moving the center of mass of the saliency maps.

Comparing the fragility of neural network interpretation among the three different methods, we found that the integrated gradients method was Across 512 correctly-classified ImageNet images, we find that the top-k and center attacks perform similarly in top-1000 intersection and rank correlation measures, and are far more effective than the random sign perturbation at demonstrating the fragility of interpretability, as characterized through top-1000 intersection (top) as well as rank order correlation (bottom).

This is true for (a) the simple gradient method, (b) DeepLift, and (c) the integrated gradients method.the most robust to both random and adversarial attacks.

Similar results for CIFAR-10 can be found in Appendix D.Results for influence functions We evaluate the robustness of influence functions on a test data set consisting of 200 images of roses and sunflowers.

Fig. 5 shows a representative test image to which we have applied the gradient sign attack.

Although the prediction of the image does not change, the most influential training examples selected according to (3), as explanation for the prediction, change entirely from images of sunflowers and yellow petals that resemble the input image to those of red and pink roses that do not.

Additional examples can be found in Appendix E.In Fig. 6 , we compare the random perturbations and gradient sign attacks across all of the test images.

We find that the gradient sign-based attacks are significantly more effective at decreasing the rank correlation of the influence of the training images, as well as distorting the top-5 influential images.

For example, on average, with a targeted perturbation of magnitude = 8, only 2 of the top 5 most influential training images remain as the top 5 most influential images after the visually imperceptible perturbation.

The influences of the training images before and after an adversarial attack are essentially uncorrelated.

However, we find that even random attacks can have a nonnegligible effect on influence functions, on average reducing the rank correlation to 0.8 ( ≈ 10).

In this section, we try to understand the source of interpretation fragility.

The question is whether fragility a consequence of the complex non-linearities of a deep network or a characteristic present even in high-dimensional linear models, as is the case for adversarial examples for prediction BID4 .

To gain more insight into the fragility of gradient based interpretations, let S(x; W ) denote the score function of interest; x ∈ R d is an input vector and W is the weights of the neural network, which is fixed since the network has finished training.

We are interested in the Figure 5 : Gradient sign attack on influence functions.

An imperceptible perturbation to a test image can significantly affect exemplar-based interpretability.

The original test image is that of a sunflower that is classified correctly in a rose vs. sunflower classification task.

The top 3 training images identified by influence functions are shown in the top row.

Using the gradient sign attack, we perturb the test image (with = 8) to produce the leftmost image in the second row.

Although the image is even more confidently predicted as a sunflower, influence functions suggest very different training images by means of explanation: instead of the sunflowers and yellow petals that resemble the input image, the most influential images are pink/red roses.

The plot on the right shows the influence of each training image before and after perturbation.

The 3 most influential images (targeted by the attack) have decreased in influence, but the influences of other images have also changed.

Figure 6: Comparison of random and targeted perturbations on influence functions.

Here, we show the averaged results of applying random (green) and gradient sign-based (orange) perturbations to 200 test images on the flower classification task.

While random attacks affect interpretability, the effect is small and generally doesn't affect the most influential images.

On the other hard, a targeted attack can significantly affect (a) the rank correlation and (b) even change the make-up of the 5 most influential images.

Even at the maximal level of noise, the changes to the perturbed images were visually imperceptible, and prediction confidence was not significantly changed (the mean change was < 1% for random attacks and < 5% for targeted attacks at the highest level of noise).Hessian H whose entries are H i,j = ∂S ∂xi∂xj .

The reason is that the first order approximation of gradient for some input perturbation direction δ ∈ R d is: DISPLAYFORM0 First, consider a linear model whose score for an input x is S = w x. Here, ∇ x S = w and ∇ 2 x S = 0; the feature-importance vector w is robust, because it is completely independent of x. Thus, some non-linearity is required for interpretation fragility.

A simple network that is susceptible to adversarial attacks on interpretations consists of a set of weights connecting the input to a single neuron followed by a non-linearity (e.g. softmax): S = g(w x).We can calculate the change in saliency map due to a small perturbation in x → x + δ.

The first-order approximation for the change in saliency map will be equal to : H · δ = ∇ 2 x S · δ.

In particular, the saliency of the i th feature changes by (∇ 2 x S · δ) i and furthermore, the relative change DISPLAYFORM1 For the simple network, this relative change is: DISPLAYFORM2 where we have used g (·) and g (·) to refer to the first and second derivatives of g(·).

Note that g (w x) and g (w x) do not scale with the dimensionality of x because in general, independent from the dimensionality, x and w are 2 -normalized or have fixed 2 -norm due to data preprocessing and weight decay regularization.

However, if we choose δ = sign(w), then the relative change in the saliency grows with the dimension, since it is proportional to the 1 -norm of w. When the input is high-dimensional-which is the case with images-the relative effect of the perturbation can be substantial.

Note also that this perturbation is exactly the sign of the first right singular vector of the Hessian ∇ 2 x S, which is appropriate since that is the vector that has the maximum effect on the gradient of S. A similar analysis can be carried out for influence functions (see Appendix F).For this simple network, the direction of adversarial attack on interpretability, sign(w) is the same as the adversarial attack on prediction.

This means that we cannot perturb interpretability independently of prediction.

For more complex networks, this is not the case and in Appendix G we show this analytically for a simple case of a two-layer network.

As an empirical test, in FIG3 , we plot the distribution of the angle between ∇ x S and v 1 (the first right singular vector of H which is the most fragile direction of feature importance) for 1000 CIFAR10 images (Details of the network in Appendix A).

In FIG3 , we plot the equivalent distribution for influence functions, computed across all 200 test images.

The result confirms that the steepest direction of change in interpretation and prediction are generally orthogonal, justifying how the perturbations can change the interpretation without changing the prediction.

Related works To the best of our knowledge, the notion of adversarial examples has not previously been studied in the context of interpretation of neural networks.

Adversarial attacks to the input that changes the prediction of a network have been actively studied.

Szegedy et al. (2013) demonstrated that it is relatively easy to fool neural networks into making very different predictions for test images that are visually very similar to each other.

BID4 introduced the Fast Gradient Sign Method (FGSM) as a one-step prediction attack.

This was followed by more effective iterative attacks (Kurakin et al., 2016) Interpretation of neural network predictions is also an active research area.

Post-hoc interpretability (Lipton, 2016) is one family of methods that seek to "explain" the prediction without talking about the details of black-box model's hidden mechanisms.

These included tools to explain predictions by networks in terms of the features of the test example (Simonyan et al., 2013; Shrikumar et al., 2017; Sundararajan et al., 2017; Zhou et al., 2016) , as well as in terms of contribution of training examples to the prediction at test time (Koh & Liang, 2017) .

These interpretations have gained increasing popularity, as they confer a degree of insight to human users of what the neural network might be doing (Lipton, 2016) .Conclusion This paper demonstrates that interpretation of neural networks can be fragile in the specific sense that two similar inputs with the same predicted label can be given very different interpretations.

We develop new perturbations to illustrate this fragility and propose evaluation metrics as well as insights on why fragility occurs.

Fragility of neural network interpretation is orthogonal to fragility of the prediction-we demonstrate how perturbations can substantially change the interpretation without changing the predicted label.

The two types of fragility do arise from similar factors, as we discuss in Section 5.

Our focus is on the interpretation method, rather than on the original network, and as such we do not explore how interpretable is the original predictor.

There is a separately line of research that tries to design simpler and more interpretable prediction models BID0 .Our main message is that robustness of the interpretation of a prediction is an important and challenging problem, especially as in many applications (e.g. many biomedical and social settings) users are as interested in the interpretation as in the prediction itself.

Our results raise concerns on how interpretations of neural networks are sensitive to noise and can be manipulated.

Especially in settings where the importance of individual or a small subset of features are interpreted, we show that these importance scores can be sensitive to even random perturbation.

More dramatic manipulations of interpretations can be achieved with our targeted perturbations, which raise security concerns.

We do not suggest that interpretations are meaningless, just as adversarial attacks on predictions do not imply that neural networks are useless.

Interpretation methods do need to be used and evaluated with caution while applied to neural networks, as they can be fooled into identifying features that would not be considered salient by human perception.

Our results demonstrate that the interpretations (e.g. saliency maps) are vulnerable to perturbations, but this does not imply that the interpretation methods are broken by the perturbations.

This is a subtle but important distinction.

Methods such as saliency measure the infinitesimal sensitivity of the neural network at a particular input x. After a perturbation, the input has changed tox = x + δ, and the salency now measures the sensitivity at the perturbed input.

The saliency correctly captures the infinitesimal sensitivity at the two inputs; it's doing what it is supposed to do.

The fact that the two resulting saliency maps are very different is fundamentally due to the network itself being fragile to such perturbations, as we illustrate with Fig. 2 .While we focus on image data (ImageNet and CIFAR-10), because these are the standard benchmarks for popular interpretation tools, this fragility issue can be wide-spread in biomedical, economic and other settings where neural networks are increasingly used.

Understanding interpretation fragility in these applications and develop more robust methods are important agendas of research.

We trained the following structure using ADAM optimizer (Kingma & Ba, 2014) with default parameters.

The resulting test accuracy using ReLU activation was 73%.

For the experiment in FIG3 , we replaced ReLU activation with Softplus and retrained the network (with the ReLU network weights as initial weights).

The resulting accuracy was 73%.

3 × 3 conv.

96 ReLU 3 × 3 conv.

96 ReLU 3 × 3 conv.

96 Relu Stride 2 3 × 3 conv.

192 ReLU 3 × 3 conv.

192 ReLU 3 × 3 conv.

192 Relu Stride 2 1024 hidden sized feed forward

Here we provide three more examples from ImageNet.

For each example, all three methods of feature importance are attacked by random sign noise and our two targeted adversarial algorithms.

Figure 12: Center-shift results for three feature importance methods on ImageNet: As discussed in the paper, among our three measurements, center-shift measure was the most correlated measure with the subjective perception of change in saliency maps.

The results in Appendix B also show that the center attack which resulted in largest average center-shift, also results in the most significant subjective change in saliency maps.

Random sign perturbations, on the other side, did not substantially change the global shape of the saliency maps, though local pockets of saliency are sensitive.

Just like rank correlation and top-1000 intersection measures, the integrated gradients method is the most robust method against adversarial attacks in the center-shift measure .

FIG0 : Results for adversarial attacks against CIFAR10 feature importance methods: For CIFAR10 the mass-center attack and top-k attack with k=100 achieve similar results for rank correlation and top-100 intersection measurements and both are stronger than random perturbations.

Mass-center attack moves the center of mass more than two other perturbations.

Among different feature importance methods, integrated gradients is more robust than the two other methods.

Additionally, results for CIFAR10 show that images in this data set are more robust against adversarial attack compared to ImageNet images which agrees with our analysis that higher dimensional inputs are tend to be more fragile.

In this appendix, we provide additional examples of the fragility of influence functions, analogous to Fig. 5 .

Here, we demonstrate that increasing the dimension of the input of a simple neural network increases the fragility of that network with respect to influence functions, analogous to the calculations carried out for importance-feature methods in Section 5.

Recall that the influence of a training image z i = (x i , y i ) on a test image z = (x, y) is given by: DISPLAYFORM0 We restrict our attention to the term in (6) that is dependent on x, and denote it by J def = ∇ θ L. J represents the infinitesimal effect of each of the parameters in the network on the loss function evaluated at the test image.

Now, let us calculate the change in this term due to a small perturbation in x → x + δ.

The firstorder approximation for the change in J is equal to: DISPLAYFORM1 For the simple network defined in Section 5, this evaluates to (replacing θ with w for consistency of notation): DISPLAYFORM2 where for simplicity, we have taken the loss to be L = |y − g(w x)|, making the derivatives easier to calculate.

Furthermore, we have used g (·) and g (·) to refer to the first and second derivatives of g(·).

Note that g (w x) and g (w x) do not scale with the dimensionality of x because x and w are generalized L 2 -normalized due to data preprocessing and weight decay regularization.

However, if we choose δ = sign(w), then the relative change in the saliency grows with the dimension, since it is proportional to the L 1 -norm of w.

Consider a two layer neural network with activation function g(·), input x ∈ R d , hidden vector u ∈ R h , and score function S), we have: DISPLAYFORM0 where w j = ||w j || 2ŵj .

We have: DISPLAYFORM1 Now for an input sample x perturbation δ, for the change in feature importance: DISPLAYFORM2 which is equal to: DISPLAYFORM3 We further assume that the input is high-dimensional so that h < d and for i = j we have w j · w i = 0.

For maximizing the 2 norm of saliency difference we have the following perturbation direction:δ m = argmax ||δ||=1 ||∇ x S(x + δ) − ∇ x S(x)|| =ŵ k where: k = argmax|v j g (w j .x)| × ||w k || 2 2 comparing which to the direction of feature importance: DISPLAYFORM4 we conclude that the two directions are not parallel unless g (.) = g (.) which is not the case for many activation functions like Softplus, Sigmoid, etc.

The analyses and experiments in this paper have demonstrated that small perturbations in the input layers of deep neural networks can have large changes in the interpretations.

This is analogous to classical adversarial examples, whereby small perturbations in the input produce large changes in the prediction.

In that setting, it has been proposed that the Lipschitz constant of the network be constrained during training to limit the effect of adversarial perturbations (Szegedy et al., 2013) .

This has found some empirical success BID3 .Here, we propose an analogous method to upper-bound the change in interpretability of a neural network as a result of perturbations to the input.

Specifically, consider a network with K layers, which takes as input a data point we denote as y 0 .

The output of the i th layer is given by y i+1 = f i (y i ) for i = 0, 1 . . .

K − 1.

We define S def = f K−1 (f K−2 (. . .

f 0 (y 0 ) . . .)) to be the output (e.g. score for the correct class) of our network, and we are interested in designing a network whose gradient S = ∇ y0 S is relatively insensitive to perturbations in the input, as this corresponds to a network whose feature importances are robust.

A natural quantity to consider is the Lipschitz constant of S with respect to y 0 .

By the chain rule, the Lipschitz constant of S is DISPLAYFORM0 Now consider the function f i (·), which maps y i to y i+1 .

In the simple case of the fully-connected network, which we consider here, f i (y i ) = g i (W i y i ), where g i is a non-linearity and W i are the trained weights for that layer.

Thus, the Lipschitz constant of the i th partial derivative in FORMULA24 is the Lipschitz constant of DISPLAYFORM1 which is upper-bounded by ||W i || 2 · L(g i (·)), where ||W || denotes the operator norm of W (its largest singular value) 5 .

This suggests that a conservative upper ceiling for (8) is DISPLAYFORM2 Because the Lipschitz constant of the non-linearities g i (·) are fixed, this result suggests that a regularization based on the operator norms of the weights W i may allow us to train networks that are robust to attacks on feature importance.

The calculations in this Appendix section is meant to be suggestive rather than conclusive, since in practice the Lipschitz bounds are rarely tight.

<|TLDR|>

@highlight

Can we trust a neural network's explanation for its prediction? We examine the robustness of several popular notions of interpretability of neural networks including saliency maps and influence functions and design adversarial examples against them.