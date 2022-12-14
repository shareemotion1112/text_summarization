It is well known that it is possible to construct "adversarial examples" for neural networks: inputs which are misclassified by the network yet indistinguishable from true data.

We propose a simple modification to standard neural network architectures, thermometer encoding, which significantly increases the robustness of the network to adversarial examples.

We demonstrate this robustness with experiments on the MNIST, CIFAR-10, CIFAR-100, and SVHN datasets, and show that models with thermometer-encoded inputs consistently have higher accuracy on adversarial examples, without decreasing generalization.

State-of-the-art accuracy under the strongest known white-box attack was  increased from 93.20% to 94.30% on MNIST and 50.00% to 79.16% on CIFAR-10.

We explore the properties of these networks, providing evidence that thermometer encodings help neural networks to find more-non-linear decision boundaries.

Adversarial examples are inputs to machine learning models that are intentionally designed to cause the model to produce an incorrect output.

The term was introduced by Szegedy et al. (2014) in the context of neural networks for computer vision.

In the context of spam and malware detection, such inputs have been studied earlier under the name evasion attacks BID0 .

Adversarial examples are interesting from a scientific perspective, because they demonstrate that even machine learning models that have superhuman performance on I.I.D. test sets fail catastrophically on inputs that are modified even slightly by an adversary.

Adversarial examples also raise concerns in the emerging field of machine learning security because malicious attackers could use adversarial examples to cause undesired behavior BID15 .Unfortunately, there is not yet any known strong defense against adversarial examples.

Adversarial examples that fool one model often fool another model, even if the two models are trained on different training examples (corresponding to the same task) or have different architectures (Szegedy et al., 2014) , so an attacker can fool a model without access to it.

Attackers can improve their success rate by sending inputs to a model, observing its output, and fitting their own own copy of the model to the observed input-output pairs BID15 .

Attackers can also improve their success rate by searching for adversarial examples that fool multiple different models-such adversarial examples are then much more likely to fool the unknown target model .

Szegedy et al. (2014) proposed to defend the model using adversarial training (training on adversarial examples as well as regular examples) but it was not feasible to generate enough adversarial examples in the inner loop of the training process for the method to be effective at the time.

Szegedy et al. (2014) used a large number of iterations of L-BFGS to produce their adversarial examples.

BID3 developed the fast gradient sign method (FGSM) of generating adversarial examples and demonstrated that adversarial training is effective for reducing the error rate on adversarial examples.

A major difficulty of adversarial training is that it tends to overfit to the method of adversarial example generation used at training time.

For example, models trained to resist FGSM adversarial examples usually fail to resist L-BFGS adversarial examples.

BID9 introduced the basic iterative method (BIM) which lies between FGSM and L-BFGS on a curve trading speed for effectiveness (the BIM consists of running FGSM for a medium number of iterations).

Adversarial training using BIM still overfits to the BIM, unfortunately, and different iterative methods can still successfully attack the model.

Recently, BID13 showed that adversarial training using adversarial examples created by adding random noise before running BIM results in a model that is highly robust against all known attacks on the MNIST dataset.

However, it is less effective on more complex datasets, such as CIFAR.

A strategy for training networks which are robust to adversarial attacks across all contexts is still unknown.

In this work, we demonstrate that thermometer code discretization and one-hot code discretization of real-valued inputs to a model significantly improves its robustness to adversarial attack, advancing the state of the art in this field.

We propose to break the linear extrapolation behavior of machine learning models by preprocessing the input with an extremely nonlinear function.

This function must still permit the machine learning model to function successfully on naturally occurring inputs.

The recent success of the PixelRNN model BID14 has demonstrated that one-hot discrete codes for 256 possible values of color pixels are effective representations for input data.

Other extremely nonlinear functions may also defend against adversarial examples, but we focused attention on vector-valued discrete encoding as our nonlinear function because of the evidence from PixelRNN that it would support successful machine learning.

Images are often encoded as a 3D tensor of integers in the range [0, 255] .

The tensor's three dimensions correspond to the image's height, width, and color channels (e.g. three for RGB, one for greyscale).

Each value represents an intensity value for a given color at a given horizontal/vertical position.

For classification tasks, these values are typically normalized to floating-point approximations in the range (0, 1).

Input discretization refers to the process of separating these continuousvalued pixel inputs into a set of non-overlapping buckets, which are each mapped to a fixed binary vector.

Past work, for example depth-color-squeezing (Xu et al., 2017) , has explored what we will refer to as quantization of inputs as a potential defense against adversarial examples.

In that approach, each pixel value is mapped to the low-bit version of its original value, which is a fixed scalar.

The key novel aspect of our approach is that rather than replacing a real number with a number of low bit depth, we replace each real number with a binary vector.

Different values of the real number activate different bits of the input vector.

Multiplying the input vector by the network's weights thus performs an operation similar to an embedding lookup in a language model, so different input values actually use different parameters of the network.

To avoid confusion, we will consistently refer to scalar-toscalar precision reduction as quantization and scalar-to-vector encoding schemes as discretization throughout this work.

A comparison of these techniques can be seen in Table 1 .

Note that, unlike depth-color-squeezing, discretization makes a meaningful change to the model, even when it is configured to use enough discretization levels to avoid losing any information from a traditionally formatted computer vision training set; discretizing each pixel to 256 levels will preserve all of the information contained in the original image.

Discretization defends against adversarial examples by changing which parameters of the model are used, and may also discard information if the number of discretization levels is low; quantization can only defend the model by discarding information.

Table 1 : Examples mapping from continuous-valued inputs to quantized inputs, one-hot codes, and thermometer codes, with ten evenly-spaced levels.

In BID3 , the authors provide evidence that several network architectures, including LSTMs BID6 , sigmoid networks BID4 , and maxout networks BID2 , are vulnerable to adversarial examples due to the empirical fact that, when trained, the loss function of these networks tends to be highly linear with respect to its inputs.

We briefly recall the reasoning of BID3 .

Assume that we have a logistic regressor with weight matrix w. Consider an image x ??? R n which is perturbed into x = x + ?? by some noise ?? such that ?? ??? ??? ?? for some ??.

The probability that the model assigns to the true class is equal to: DISPLAYFORM0 If the perturbation ?? is adversarial, such as in the case where DISPLAYFORM1 , then the input to the sigmoid is increased by ?? ?? n. If n is large, as is typically the case in images and other highdimensional spaces of interest, this linearity implies that even imperceptibly small values of ?? can have a large effect on the model's prediction, making the model vulnerable to adversarial attacks.

Though neural networks in principle have the capacity to represent highly nonlinear functions, networks trained via stochastic gradient descent on real-world datasets tend to converge to mostly-linear solutions.

This is illustrated in the empirical studies conducted by BID3 .

One hypothesis proposed to explain this phenomenon is that the nonlinearities typically used in networks are either piecewise linear, like ReLUs, or approximately linear in the parts of their domain in which training takes place, like the sigmoid function.

One potential solution to this problem is to use more non-linear activation functions, such as quadratic or RBF units.

Indeed, it was shown by BID3 that such units were more resistant to adversarial perturbations of their inputs.

However, these units are difficult to train, and the resulting models do not generalize very well BID3 , sacrificing accuracy on clean examples.

As an alternative to introducing highly non-linear activation functions in the network, we propose applying a non-differentiable and non-linear transformation (discretization) to the input, before passing it into the model.

A comparison of the input to the model under various regimes can be seen in FIG1 , highlighting the strong non-linearity of discretization techniques.

Comparison of regular inputs, quantized inputs, and discretized inputs (16 levels, projected to one dimension) on MNIST, adversarially trained with ?? = 0.3.

The x-axis represents the true pixel value of the image, and the y-axis represents the value that is passed as input to the network after the input transformation has been applied.

For real-valued inputs, the inputs to the network are affected linearly by perturbations to the input.

Quantized inputs are also affected approximately linearly by perturbations where ?? is greater than the bucket width.

Discretizing the input, and then using learned weights to project the discretized value back to a single scalar, we see that the model has learned a highly non-linear function to represent the input in a fashion that is effective for resisting the adversarial perturbations it has seen.

When starting at the most common pixel-values for MNIST, 0 and 1, any perturbation of the pixels (where ?? ??? 0.3) has barely any effect on the input to the network.

In this work we consider two approaches to constructing discretized representations f (x) of the input image x. Assume for the sake of simplicity that the entries of x take values in the continuous DISPLAYFORM0 We first describe a quantization function b. Choose 0 < b 1 < b 2 < ?? ?? ?? < b k = 1 in some fashion.(In this work, we simply divide the domain evenly, i.e. b i = i k .)

For a real number ?? ??? [0, 1] define b(??) to be the largest index ?? ??? {1, . . .

, k} such that ?? ??? b ?? .

For an index j ??? {1, . . . , k} let ??(j) ??? R k be the indicator or one-hot vector of j, i.e., DISPLAYFORM0 The discretization function is defined pixel-wise for a pixel i ??? {1, . . .

, n} as: DISPLAYFORM1 One-hot encodings are simple to compute and understand, and are often used when it is necessary to represent a categorical variable in a neural network.

However, one-hot encodings are not well suited for representing categorical variables with an interpretation of ordering between them.

Note that the ordering information between two pixels x i and x j is lost by applying the transformation f onehot ; for a pair of pixels i, j whenever b(x i ) = b(x j )), we see: DISPLAYFORM2 In the case of pixel values, this is not a good inductive bias, as there is a clear reason to believe that neighboring buckets are more similar to each other than distant buckets.

In order to discretize the input image x without losing the relative distance information, we propose thermometer encodings.

For an index j ??? {1, . . .

, k}, let ?? (j) ??? R k be the thermometer vector defined as DISPLAYFORM0 Then the discretization function f is defined pixel-wise for a pixel i ??? {1, . . .

, n} as: DISPLAYFORM1 where C is the cumulative sum function, DISPLAYFORM2 Note that the thermometer encoding preserves pairwise ordering information, i.e., for pixels i, j if DISPLAYFORM3

Discretizing the input makes it difficult to attack the model with standard white-box attack algorithms, such as FGSM BID3 and PGD BID13 , since it is impossible to backpropagate through our discretization function to determine how to adversarially modify the model's input.

In this section, we describe two novel iterative attacks which allow us to construct adversarial examples for networks trained on discretized inputs.

Constructing white-box attacks on discretized inputs serves two primary purposes.

First, it allows us to more completely evaluate whether the model is robust to all adversarial attacks, as white-box attacks are typically more powerful than their black-box counterparts.

Secondly, adversarial training is typically performed in a white-box fashion, and so in order to utilize and properly compare against the adversarial training techniques of BID13 , it is important to have strong white-box attacks.

For ease of presentation, we will describe the attacks assuming that f : R ??? R k discretizes inputs into thermometer encodings; in order to attack one-hot encodings, simply replace all instances of f therm with f onehot , ?? with ??, and C with the identity function I. We represent the adversarial image after t steps of the attack as z t , where the value of the ith pixel is z t i .

The first attack, Discrete Gradient Ascent (DGA), follows the direction of the gradient of the loss with respect to f (x), but is constrained at every step to be a discretized vector.

If we have discretized the input image into k-dimensional vectors using the one-hot encoding, this corresponds to moving to a vertex of the simplex (??? k ) n at every step.

The second attack, Logit-Space Projected Gradient Ascent (LS-PGA), relaxes this assumption, allowing intermediate iterates to be in the interior of the simplex.

The final adversarial image is obtained by projecting the final point back to the nearest vertex of the simplex.

Note that if the number of attack steps is 1, then the two attacks are equivalent; however, for larger numbers of attack steps, LS-PGA is a generalization of DGA.

Following PGD BID13 , we initialize DGA by placing each pixel into a random bucket that is within ?? of the pixel's true value.

At each step of the attack, we look at all buckets that are within ?? of the true value, and select the bucket that is likely to do the most 'harm', as estimated by the gradient of setting that bucket's indicator variable to 1, with respect to the model's loss at the previous step.

DISPLAYFORM0 Because the outcome of this optimization procedure will vary depending on the initial random perturbation, we suggest strengthening the attack by re-running it several times and using the perturbation with the greatest loss.

The pseudo-code for the DGA attack is given in Section B of the appendix.

To perform LS-PGA, we soften the discrete encodings into continuous relaxations, and then perform standard Projected Gradient Ascent (PGA) on these relaxed values.

We represent the distribution over embeddings as a softmax over logits u, each corresponding to the unnormalized log-weight of a specific bucket's embedding.

To improve the attack, we scale the logits with temperature T , allowing us to trade off between how closely our softmax approximates a true one-hot distribution as in the Gumbel-softmax trick BID8 BID12 , and how much gradient signal the logits receive.

At each step of a multi-step attack, we anneal this value via exponential decay with rate ??.

DISPLAYFORM0 We initialize each of the logits randomly with values sampled from a standard normal distribution.

At each step, we ensure that the model does not assign any probability to buckets which are not within ?? of the true value by fixing the logits to be ??????. The model's loss is a continuous function of the logits, so we can simply utilize attacks designed for continuous-valued inputs, in this case PGA with step-size ??.

DISPLAYFORM1 Because the outcome of this optimization procedure will vary depending on the initial perturbation, we suggest strengthening the attack by re-running it several times and using the perturbation with the greatest loss.

The pseudo-code for the LS-PGA attack is given in Section B of the appendix.

We compare models trained with input discretization to state-of-the-art adversarial defenses on a variety of datasets.

We match the experimental setup of the prior literature as closely as possible.

Rows labeled with "Vanilla (Madry)" give the numbers reported in BID13 ; other rows contain results of our own experiments, with "Vanilla" containing a direct replication.

For our MNIST experiments, we use a convolutional network; for CIFAR-10, CIFAR-100, and SVHN we use a Wide ResNet (Zagoruyko & Komodakis, 2016) .

We use a network of depth 30 for the CIFAR-10 and CIFAR-100 datasets, while for SVHN we use a network of depth 15.

The width factor of all the Wide ResNets is set to k = 4.

1 Unless otherwise specified, all quantized and discretized models use 16 levels.

We found that in all cases, LS-PGA was strictly more powerful than DGA, so all attacks on discretized models use LS-PGA with ?? = 0.01, ?? = 1.2, and 1 random restart.

To be consistent with BID13 , we describe attacks in terms of the maximum ??? -norm of the attack, ??.

All MNIST experiments used ?? = 0.3 and 40 steps for iterative attacks; experiments on CIFAR used ?? = 0.031 and 7 steps for iterative attacks; experiments on SVHN used ?? = 0.047 and 10 steps for iterative attacks.

These settings were used for adversarial training, white-box attacks, and blackbox attacks.

Figure 3 plots the effectiveness of the iterated PGD/LS-PGA attacks on vanilla and discretized models for MNIST and shows that increasing the number of iterations beyond 40 would have no effect on the performance of the model on ??? -bounded adversarial examples for MNIST.In BID13 , adversarially-trained models are trained using exclusively adversarial inputs.

This led to a small but noticeable loss in accuracy on clean examples, dropping from 99.2% to 98.8% on MNIST and from 95.2% to 87.3% on CIFAR-10 in return for more robustness towards adversarial examples.

Past work has also sometimes performed adversarial training on batches composed of half clean examples and half adversarial examples BID3 BID1 .

To be consistent with BID13 , we list experiments on models trained only on adversarial inputs in the main paper; additional experiments on a mix of clean and adversarial inputs can be found in the appendix.

We also run experiments exploring the model's relationship with the number of distinct levels to which we quantize the input before discretizing it, and exploring various settings of hyperparameters for LS-PGA.

Our adversarially-trained baseline models were able to approximately replicate the results of BID13 .

On all datasets, discretizing the inputs of the network dramatically improves resistance to adversarial examples, while barely sacrificing any accuracy on clean examples.

Quantized models also beat the baseline, but with lower accuracy on clean examples.

Discretization via thermometer encodings outperformed one-hot encodings in most settings.

See Tables 2,3,4 and 5 for results on MNIST and CIFAR-10.

Additional results on CIFAR-100 and SVHN are included in the appendix.

In Figures 2 and 5 (located in appendix) , we plot the test-set accuracy across training timesteps for various adversarially trained models on the SVHN and CIFAR-10 datasets, and observe that the discretized models become robust against adversarial examples more quickly.

Clean FGSM PGD/LS-PGA

In BID3 , the seeming linearity of deep neural networks was shown by visualizing the networks in several different ways.

To test our hypothesis that discretization breaks some of this linearity, we replicate these visualizations and contrast them to visualizations of discretized models.

See Appendix G for an illustration of these properties.

For non-discretized, clean trained models, test-set examples always yield a linear boundary between correct and incorrect classification; in contrast, non-adversarially-trained models have a more interesting parabolic shape (see Figure 9 ).

Loss for iterated white-box attacks on various models on a randomly chosen data point from MNIST.

By step 40, which is where we evaluate, the loss of the point found by iterative attacks has converged.

DISPLAYFORM0 When discretizing the input, we introduce C w ?? C h ?? C o ?? c ?? (k ??? 1) extra parameters, where c is the number of channels in the image, k is the number of levels of discretization, and C w , C h , C o are the width, height, and output channels of the first convolutional layer.

Discretizing using 16 levels introduced 0.03% extra parameters for MNIST, 0.08% for CIFAR-10 and CIFAR-100, and 2.3% for SVHN.

This increase is negligible, so it is likely that the robustness comes from the input discretization, and is not merely a byproduct of having a slightly higher-capacity model.

In this section, we describe the hyperparameters used in our experiments.

For CIFAR-10 and CIFAR-100 we follow the standard data augmenting scheme as in BID10 BID5 BID7 Zagoruyko & Komodakis, 2016) : each training image is zero-padded with 4 pixels on each side and randomly cropped to a new 32 ?? 32 image.

The resulting image is randomly flipped with probability 0.5, it's brightness is adjusted with a delta chosen uniformly at random in the interval [???63, 63) and it's contrast is adjusted using a random contrast factor in the interval [0.2, 1.8].

For MNIST we use the Adam optimizer with a fixed learning rate of 1e???4 as in BID13 .

For CIFAR-10 and CIFAR-100 we use the Momentum optimizer with momentum 0.9, 2 weight decay of ?? = 0.0005 and an initial learning rate of 0.1 which is annealed by a factor of 0.2 after epochs 60, 120 and 160 respectively as in Zagoruyko & Komodakis (2016) .

For SVHN we use the same optimizer with initial learning rate of 1e???2 which is annealed by a factor of 0.1 after epochs 80 and 120 respectively.

We also use a dropout of 0.3 for CIFAR-10, CIFAR-100 and SVHN.

The DGA attack is described in Algorithm 2 and the LS-PGA attack is described in Algorithm 3.

Both these algorithms make use of a getM ask() sub-routine which is described in Algorithm 1.

DISPLAYFORM0 Algorithm 1: Sub-routine for getting an ??-discretized mask of an image.

In this section we list the additional experiments we performed using discretized models on MNIST.The main hyperparameters of Algorithm 3 are the step size ?? used to perform the projected gradient ascent, and the annealing rate of ??.

We found that the choice of these hyperparameters was not critical to the robustness of the model.

In particular, we performed experiments with ?? = 1.0 and ?? = 0.001, and both achieved similar accuracies as in TAB2 .

Additionally, we found that without annealing, i.e., ?? = 1.0, the performance was only slightly worse than with ?? = 1.2.We also experimented with discretizing by using percentile information per color channel instead of using uniformly distributed buckets.

This did not result in any significant changes in robustness or accuracy for the MNIST dataset.

Input: Image x, label y, discretization function f , loss L(?? ?? ??, f (x), y), l attack steps, parameter ?? Output: Adversarial input to the network TAB7 : Comparison of adversarial robustness to white-box attacks on MNIST using 16 levels and with various choices of the hyperparameters ?? and ?? for Algorithm 3.

The models are evaluated on white-box attacks and on black-box attacks using a vanilla, clean trained model; both use LS-PGA.

DISPLAYFORM0 Finally, we also trained on a mix of clean and adversarial examples: this resulted in significantly higher accuracy on clean examples, but decreased accuracy on white-box and black-box attacks compared to TAB2 .

Table 9 : Comparison of adversarial robustness on MNIST as the number of levels of discretization is varied.

All models are trained mix of adversarial examples and clean examples.

In this section we list the additional experiments we performed on CIFAR-10.

Firstly, we trained models on a mix of both clean and adversarial examples.

The results for mixed training are listed in TAB12 ; as expected it has lower accuracy on adversarial examples, but higher accuracy on clean examples, compared to training on only adversarial examples TAB4 ).

Table 11 : Comparison of adversarial robustness to black-box attacks on CIFAR-10 of various models using a mix of clean and adversarial examples.

In order to explore whether the number of levels of discretization affected the performance of the model, we trained several models which varied this number.

As expected, we found that models with fewer levels had worse accuracy on clean examples, likely because there was not enough information to correctly classify the image, but greater robustness to adversarial examples, likely because larger buckets mean a greater chance that a given perturbation will not yield any change in input to the network (Xu et al., 2017) .

Results can be seen in Tables 12, and are visualized in FIG3 .

TAB2 : Comparison of adversarial robustness on CIFAR-10 as the number of levels of discretization is varied.

All models are trained only on adversarial examples.

We list the experimental results on CIFAR-100 in TAB3 .

We choose ?? = 0.01 and ?? = 1.2 for the LS-PGA attack hyperparameters.

For the discretized models, we used 16 levels.

All adversarially trained models were trained on a mix of clean and adversarial examples.

DISPLAYFORM0

In FIG3 we plot the effect of increasing the levels of discretization for the MNIST and CIFAR-10 datasets.

In Figure 5 we plot the convergence rate of clean trained and adversarially trained models on the CIFAR-10 dataset.

Note that thermometer encoded inputs converge much faster in accuracy on both clean and adversarial inputs.

Figure 6 plots the norm of the gradient as a function of the number of iterations of the attack on MNIST.

Note that the gradient vanishes at around 40 iterations, which coincides with the loss stabilizing in Figure 3 .In FIG5 , we create a linear interpolation between a clean image and an adversarial example, and then continue to extrapolate along this line, evaluating probability of each class at each point.

In models trained on unquantized inputs, the class probabilities are all mostly piecewise linear in both the positive and negative directions.

In contrast, the discretized model has a much more jagged and irregular shape.

In FIG6 , we plot the error for different models on various values of ??.

The discretized models are extremely robust to all values less-than-or-equal-to the values that they have been exposed to during training.

However, beyond this threshold, discretized models collapse immediately, while real-valued models still maintain some semblance of robustness.

This exposes a weakness of the discretization approach; the same nonlinearity that helps it learn to become robust to all attacks it sees during training-time causes its behavior is unpredictable beyond that.

However, we believe that the fact that the performance of thermometer-encoded models degrades more quickly than that of vanilla models beyond the training epsilon is a significant weakness in practice, but no worse than other defenses.

The "standard setting" for the adversarial example problem (in which we constrain the L-infinity norm of the perturbed image to an epsilon ball around the original image) was designed to ensure that any adversarially-perturbed image is still recognizable as its original image by a human.

However, this artificial constraint excludes many other potential attacks that also result in human-recognizable images.

State-of-the art defenses in the standard setting can still be easily defeated by non-standard attacks; for an example of this, see appendix A of ICLR submission "Adversarial Spheres".

A "larger epsilon" attack is just one special case of a "non-standard" attack.

If we permit non-standard attacks, a fair comparison would show that all current approaches are easily breakable.

There is nothing special about the "larger epsilon" attack that makes a vulnerability to this non-standard attack in particular more problematic than vulnerabilities to other non-standard attacks.

In FIG1 , 11, 12 , 13 and 14 we plot several examples of church-window plots for MNIST BID3 .

Each plot is crafted by taking several test-set images, calculating the vector corresponding to an adversarial attack on each image, and then choosing an additional random orthogonal direction.

In each plot, the clean image is at the center and corresponds to the color white, the x-axis corresponds to the magnitude of a perturbation in the adversarial direction, and the y-axis corresponds to the magnitude of a perturbation in the orthogonal direction.

Note that we use the same random seed to generate the test set examples and the adversarial directions across different church-window plots.

Figure 11: Church-window plots of adversarially-trained models on MNIST, trained using a mix of clean and adversarial examples.

The x-axis of each sub-plot represents the adversarial direction, while the y-axis represents a random orthogonal direction.

The correct class is represented by white.

Every row in the plot contains a training data point chosen uniformly at random, while each column uses a different random orthogonal vector for the y-axis.

The ?? bound for both axes is [???1.0, 1.0].

Notice the almost-linear decision boundaries on non-discretized models.

Figure 14: Church-window plots of adversarially-trained models on CIFAR-10, trained using a mix of clean and adversarial examples.

The x-axis of each sub-plot represents the adversarial direction, while the y-axis represents a random orthogonal direction.

The correct class is represented by white.

Every row in the plot contains a training data point chosen uniformly at random, while each column uses a different random orthogonal vector for the y-axis.

The ?? bound for both axes is [???1.0, 1.0].

Notice the almost-linear decision boundaries on non-discretized models.

@highlight

Input discretization leads to robustness against adversarial examples

@highlight

The authors present an in-depth study of discretizing / quantizing the input as a defense against adversarial examples