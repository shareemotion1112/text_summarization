While deep neural networks have proven to be a powerful tool for many recognition and classification tasks, their stability properties are still not well understood.

In the past, image classifiers have been shown to be vulnerable to so-called adversarial attacks, which are created by additively perturbing the correctly classified image.

In this paper, we propose the ADef algorithm to construct a different kind of adversarial attack created by iteratively applying small deformations to the image, found through a gradient descent step.

We demonstrate our results on MNIST with convolutional neural networks and on ImageNet with Inception-v3 and ResNet-101.

In a first observation in BID25 it was found that deep neural networks exhibit unstable behavior to small perturbations in the input.

For the task of image classification this means that two visually indistinguishable images may have very different outputs, resulting in one of them being misclassified even if the other one is correctly classified with high confidence.

Since then, a lot of research has been done to investigate this issue through the construction of adversarial examples: given a correctly classified image x, we look for an image y which is visually indistinguishable from x but is misclassified by the network.

Typically, the image y is constructed as y = x + r, where r is an adversarial perturbation that is supposed to be small in a suitable sense (normally, with respect to an p norm).

Several algorithms have been developed to construct adversarial perturbations, see BID9 BID18 ; BID14 ; BID16 ; BID5 and the review paper BID0 .Even though such pathological cases are very unlikely to occur in practice, their existence is relevant since malicious attackers may exploit this drawback to fool classifiers or other automatic systems.

Further, adversarial perturbations may be constructed in a black-box setting (i.e., without knowing the architecture of the DNN but only its outputs) BID19 BID17 and also in the physical world BID14 BID1 BID3 BID24 .

This has motivated the investigation of defenses, i.e., how to make the network invulnerable to such attacks, see BID13 ; BID4 ; BID16 ; BID27 ; BID28 ; BID20 ; BID2 ; BID12 .

In most cases, adversarial examples are artificially created and then used to retrain the network, which becomes more stable under these types of perturbations.

Most of the work on the construction of adversarial examples and on the design of defense strategies has been conducted in the context of small perturbations r measured in the ∞ norm.

However, this is not necessarily a good measure of image similarity: e.g., for two translated images x and y, the norm of x−y is not small in general, even though x and y will look indistinguishable if the translation is small.

Several papers have investigated the construction of adversarial perturbations not designed for norm proximity BID21 BID24 BID3 BID6 BID29 .In this work, we build up on these ideas and investigate the construction of adversarial deformations.

In other words, the misclassified image y is not constructed as an additive perturbation y = x + r, but as a deformation y = x • (id + τ ), where τ is a vector field defining the transformation.

In this case, the similarity is not measured through a norm of y − x, but instead through a norm of τ , which quantifies the deformation between y and x.

We develop an efficient algorithm for the construction of adversarial deformations, which we call ADef.

It is based on the main ideas of DeepFool BID18 , and iteratively constructs the smallest deformation to misclassify the image.

We test the procedure on MNIST (LeCun) (with convolutional neural networks) and on ImageNet (Russakovsky et al., 2015) (with Inception-v3 BID26 and ResNet-101 BID10 ).

The results show that ADef can succesfully fool the classifiers in the vast majority of cases (around 99%) by using very small and imperceptible deformations.

We also test our adversarial attacks on adversarially trained networks for MNIST.

Our implementation of the algorithm can be found at https://gitlab.math.

ethz.ch/tandrig/ADef.The results of this work have initially appeared in the master's thesis BID8 , to which we refer for additional details on the mathematical aspects of this construction.

While writing this paper, we have come across BID29 , in which a similar problem is considered and solved with a different algorithm.

Whereas in BID29 the authors use a second order solver to find a deforming vector field, we show how a first order method can be formulated efficiently and justify a smoothing operation, independent of the optimization step.

We report, for the first time, success rates for adversarial attacks with deformations on ImageNet.

The topic of deformations has also come up in BID11 , in which the authors introduce a class of learnable modules that deform inputs in order to increase the performance of existing DNNs, and BID7 , in which the authors introduce a method to measure the invariance of classifiers to geometric transformations.

Let K be a classifier of images consisting of P pixels into L ≥ 2 categories, i.e. a function from the space of images X = R cP , where c = 1 (for grayscale images) or c = 3 (for color images), and into the set of labels L = {1, . . .

, L}. Suppose x ∈ X is an image that is correctly classified by K and suppose y ∈ X is another image that is imperceptible from x and such that K(y) = K(x), then y is said to be an adversarial example.

The meaning of imperceptibility varies, but generally, proximity in p -norm (with 1 ≤ p ≤ ∞) is considered to be a sufficient substitute.

Thus, an adversarial perturbation for an image x ∈ X is a vector r ∈ X such that K(x + r) = K(x) and r p is small, where DISPLAYFORM0 Given such a classifier K and an image x, an adversary may attempt to find an adversarial example y by minimizing x − y p subject to K(y) = K(x), or even subject to K(y) = k for some target label k = K(x).

Different methods for finding minimal adversarial perturbations have been proposed, most notably FGSM BID9 and PGD BID16 for ∞ , and the DeepFool algorithm BID18 DISPLAYFORM1 extending ξ by zero outside of [0, 1] 2 .

Deformations capture many natural image transformations.

For example, a translation of the image ξ by a vector v ∈ R 2 is a deformation with respect to the constant vector field τ = v. If v is small, the images ξ and ξ v may look similar, but the corresponding perturbation ρ = ξ v − ξ may be arbitrarily large in the aforementioned L p -norms.

Figure 1 shows three minor deformations, all of which yield large L ∞ -norms.

In the discrete setting, deformations are implemented as follows.

We consider square images of W × W pixels and define the space of images to be DISPLAYFORM2 In what follows we will only consider the set T of vector fields that do not move points on the grid {1, . . .

, W } 2 outside of [1, W ] 2 .

More precisely, DISPLAYFORM3 An image x ∈ X can be viewed as the collection of values of a function ξ : DISPLAYFORM4 for s, t = 1, . . .

, W .

Such a function ξ can be computed by interpolating from x. Thus, the deformation of an image x with respect to the discrete vector field τ can be defined as the discrete deformed image x τ in X by DISPLAYFORM5 It is not straightforward to measure the size of a deformation such that it captures the visual difference between the original image x and its deformed counterpart x τ .

We will use the size of the corresponding vector field, τ , in the norm defined by DISPLAYFORM6 as a proxy.

The p -norms defined in (1), adapted to vector fields, can be used as well.

(We remark, however, that none of these norms define a distance between x and x τ , since two vector fields τ, σ ∈ T with τ T = σ T may produce the same deformed image x τ = x σ .)

We will now describe our procedure for finding deformations that will lead a classifier to yield an output different from the original label.

DISPLAYFORM0 Let x ∈ X be the image of interest and fix ξ : [0, 1] 2 → R c obtained by interpolation from x. Let l = K(x) denote the true label of x, let k ∈ L be a target label and set f = F k − F l .

We assume that x does not lie on a decision boundary, so that we have f (x) < 0.We define the function g : DISPLAYFORM1 We can use a linear approximation of g around the zero vector field as a guide: DISPLAYFORM2 for small enough τ ∈ T and D 0 g : T → R the derivative of g at τ = 0.

Hence, if τ is a vector field such that DISPLAYFORM3 and τ T is small, then the classifier K has approximately equal confidence for the deformed image x τ to have either label l or k. This is a scalar equation with unknown in T , and so has infinitely many solutions.

In order to select τ with small norm, we solve it in the least-squares sense.

In view of (2), we have DISPLAYFORM4 .

Thus, by applying the chain rule to g(τ ) = f (x τ ), we obtain that its derivative at τ = 0 can, with a slight abuse of notation, be identified with the vector field DISPLAYFORM5 where ∇f (x) s,t ∈ R 1×c is the derivative of f in x calculated at (s, t).

With this, DISPLAYFORM6 , and the solution to (5) in the least-square sense is given by DISPLAYFORM7 Finally, we define the deformed image x τ ∈ X according to (2).One might like to impose some degree of smoothness on the deforming vector field.

In fact, it suffices to search in the range of a smoothing operator S : T → T .

However, this essentially amounts to applying S to the solution from the larger search space DISPLAYFORM8 where S denotes the componentwise application of a two-dimensional Gaussian filter ϕ (of any standard deviation).

Then the vector field DISPLAYFORM9 also satisfies (5), since S is self-adjoint.

We can hence replace τ byτ to obtain a smooth deformation of the image x.

We iterate the deformation process until the deformed image is misclassified.

More explicitly, let x (0) = x and for n ≥ 1 let τ (n) be given by (7) for x (n−1) .

Then we can define the iteration as DISPLAYFORM10 ).

The algorithm terminates and outputs an adversarial example y = x DISPLAYFORM11 The iteration also terminates if x (n) lies on a decision boundary of K, in which case we propose to introduce an overshoot factor 1 + η on the total deforming vector field.

Provided that the number of iterations is moderate, the total vector field can be well approximated by τ * = τ(1) +· · ·+τ (n) and the process can be altered to output the deformed image DISPLAYFORM12 The target label k may be chosen in each iteration to minimize the vector field to obtain a better approximation in the linearization (4).

More precisely, for a candidate set of labels k 1 , . . .

, k m , we compute the corresponding vectors fields τ 1 , . . .

, τ m and select DISPLAYFORM13 The candidate set consists of the labels corresponding to the indices of the m smallest entries of F − F l , in absolute value.

DISPLAYFORM14 By equation (6), provided that ∇f is moderate, the deforming vector field takes small values wherever ξ has a small derivative.

This means that the vector field will be concentrated on the edges in the image x (see e.g. the first row of figure 2).

Further, note that the result of a deformation is always a valid image in the sense that it does not violate the pixel value bounds.

This is not guaranteed for the perturbations computed with DeepFool.

We evaluate the performance of ADef by applying the algorithm to classifiers trained on the MNIST (LeCun) and ImageNet (Russakovsky et al., 2015) datasets.

Below, we briefly describe the setup of the experiments and in tables 1 and 2 we summarize their results.

MNIST: We train two convolutional neural networks based on architectures that appear in BID16 and BID27 respectively.

The network MNIST-A consists of two convolutional layers of sizes 32 × 5 × 5 and 64 × 5 × 5, each followed by 2 × 2 max-pooling and a rectifier activation function, a fully connected layer into dimension 1024 with a rectifier activation function, and a final linear layer with output dimension 10.

The network MNIST-B consists of two convolutional layers of sizes 128 × 3 × 3 and 64 × 3 × 3 with a rectifier activation function, a fully connected layer into dimension 128 with a rectifier activation function, and a final linear layer with output dimension 10.

During training, the latter convolutional layer and the former fully connected layer of MNIST-B are subject to dropout of drop probabilities 1 /4 and 1 /2.

We use ADef to produce adversarial deformations of the images in the test set.

The algorithm is configured to pursue any label different from the correct label (all incorrect labels are candidate labels).

It performs smoothing by a Gaussian filter of standard deviation 1 /2, uses bilinear interpolation to obtain intermediate pixel intensities, and it overshoots by η = 2 /10 whenever it converges to a decision boundary.

An image from the ILSVRC2012 validation set, the output of ADef with a Gaussian filter of standard deviation 1, the corresponding vector field and perturbation.

The rightmost image is a close-up of the vector field around the nose of the ape.

Second row: A larger deformation of the same image, obtained by using a wider Gaussian filter (standard deviation 6) for smoothing.

We apply ADef to pretrained Inception-v3 BID26 and ResNet-101 BID10 ) models to generate adversarial deformations for the images in the ILSVRC2012 validation set.

The images are preprocessed by first scaling so that the smaller axis has 299 pixels for the Inception model and 224 pixels for ResNet, and then they are center-cropped to a square image.

The algorithm is set to focus only on the label of second highest probability.

It employs a Gaussian filter of standard deviation 1, bilinear interpolation, and an overshoot factor η = 1 /10.We only consider inputs that are correctly classified by the model in question, and, since τ * = τ(1) +· · ·+τ (n) approximates the total deforming vector field, we declare ADef to be successful if its output is misclassified and τ * T ≤ ε, where we choose ε = 3.

Observe that, by (3), a deformation with respect to a vector field τ does not displace any pixel further away from its original position than τ T .

Hence, for high resolution images, the choice ε = 3 indeed produces small deformations if the vector fields are smooth.

In appendix A, we illustrate how the success rate of ADef depends on the choice of ε.

When searching for an adversarial example, one usually searches for a perturbation with ∞ -norm smaller than some small number ε > 0.

Common choices of ε range from 1 /10 to 3 /10 for MNIST classifiers BID9 BID16 BID28 BID27 BID12 and 2 /255 to 16 /255 for ImageNet classifiers BID9 BID13 BID27 BID12 .

TAB1 shows that on average, the perturbations obtained by ADef are quite large compared to those constraints.

However, as can be seen in FIG0 , the relatively high resolution images of the ImageNet dataset can be deformed into adversarial examples that, while corresponding to large perturbations, are not visibly different from the original images.

In appendices B and C, we give more examples of adversarially deformed images.

In addition to training MNIST-A and MNIST-B on the original MNIST data, we train independent copies of the networks using the adversarial training procedure described by BID16 .

That is, before each step of the training process, the input images are adversarially perturbed using the PGD algorithm.

This manner of training provides increased robustness against adversarial perturbations of low ∞ -norm.

Moreover, we train networks using ADef instead of PGD as an adversary.

In table 2 we show the results of attacking these adversarially trained networks, using ADef on the one hand, and PGD on the other.

We use the same configuration for ADef as above, and for PGD we use 40 iterations, step size 1 /100 and 3 /10 as the maximum ∞ -norm of the perturbation.

Interestingly, using these configurations, the networks trained against PGD attacks are more resistant to adversarial deformations than those trained against ADef.

ADef can also be used for targeted adversarial attacks, by restricting the deformed image to have a particular target label instead of any label which yields the optimal deformation.

FIG1 demonstrates the effect of choosing different target labels for a given MNIST image, and FIG2 shows the result of targeting the label of lowest probability for an image from the ImageNet dataset.

In this work, we proposed a new efficient algorithm, ADef, to construct a new type of adversarial attacks for DNN image classifiers.

The procedure is iterative and in each iteration takes a gradient descent step to deform the previous iterate in order to push to a decision boundary.

We demonstrated that with almost imperceptible deformations, state-of-the art classifiers can be fooled to misclassify with a high success rate of ADef.

This suggests that networks are vulnerable to different types of attacks and that simply training the network on a specific class of adversarial examples might not form a sufficient defense strategy.

Given this vulnerability of neural networks to deformations, we wish to study in future work how ADef can help for designing possible defense strategies.

Furthermore, we also showed initial results on fooling adversarially trained networks.

Remarkably, PGD trained networks on MNIST are more resistant to adversarial deformations than ADef trained networks.

However, for this result to be more conclusive, similar tests on ImageNet will have to be conducted.

We wish to study this in future work.

T from the MNIST experiments.

Deformations that fall to the left of the vertical line at ε = 3 are considered successful.

The networks in the first column were trained using the original MNIST data, and the networks in the second and third columns were adversarially trained using ADef and PGD, respectively.

Figures 5 and 6 show the distribution of the norms of the total deforming vector fields, τ * , from the experiments in section 3.

For networks that have not been adversarially trained, most deformations fall well below the threshold of = 3.

Out of the adversarially trained networks, only MNIST-A trained against PGD is truly robust against ADef.

Further, a comparison between the first column of figure 5 and figure 6 indicates that ImageNet is much more vulnerable to adversarial deformations than MNIST, also considering the much higher resolution of the images in ImageNet.

Thus, it would be very interesting to study the performance of ADef with adversarially trained network for ImageNet, as mentioned in the Conclusion.

The standard deviation of the Gaussian filter used for smoothing in the update step of ADef has significant impact on the resulting vector field.

To explore this aspect of the algorithm, we repeat the experiment from section 3 on the Inception-v3 model, using standard deviations σ = 0, 1, 2, 4, 8 (where σ = 0 stands for no smoothing).

The results are shown in table 3, and the effect of varying σ is illustrated in figures 7 and 8.

We observe that as σ increases, the adversarial distortion steadily increases both in terms of vector field norm and perturbation norm.

Likewise, the success rate of ADef decreases with larger σ.

However, from figure 8 we see that the constraint τ * T ≤ 3 on the total vector field may provide a rather conservative measure of the effectiveness of ADef in the case of smooth high dimensional vector fields.

Figures 9 and 10 show adversarial deformations for the models MNIST-A and MNIST-B, respectively.

The attacks are performed using the same configuration as in the experiments in section 3.

Observe that in some cases, features resembling the target class have appeared in the deformed image.

For example, the top part of the 4 in the fifth column of figure 10 has been curved slightly to more resemble a 9.

Figures 11 -15 show additional deformed images resulting from attacking the Inception-v3 model using the same configuration as in the experiments in section 3.

Similarly, figures 16 -20 show deformed images resulting from attacking the ResNet-10 model.

However, in order to increase variability in the output labels, we perform a targeted attack, targeting the label of 50th highest probability.

Deformed: hartebeest

@highlight

We propose a new, efficient algorithm to construct adversarial examples by means of deformations, rather than additive perturbations.