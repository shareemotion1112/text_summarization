We improve the robustness of deep neural nets  to adversarial attacks by using an interpolating function as the output activation.

This data-dependent activation function remarkably improves both classification accuracy and stability to adversarial perturbations.

Together with the total variation minimization of adversarial images and augmented training, under the strongest attack, we achieve up to 20.6%, 50.7%, and 68.7% accuracy improvement w.r.t.

the fast gradient sign method, iterative fast gradient sign method, and Carlini-WagnerL2attacks, respectively.

Our defense strategy is additive to many of the existing methods.

We give an intuitive explanation of our defense strategy via analyzing the geometry of the feature space.

For reproducibility, the code will be available on GitHub.

The adversarial vulnerability BID26 of deep neural nets (DNNs) threatens their applicability in security critical tasks, e.g., autonomous cars BID0 , robotics BID8 , DNN-based malware detection systems BID20 BID7 .

Since the pioneering work by BID26 , many advanced adversarial attack schemes have been devised to generate imperceptible perturbations to sufficiently fool the DNNs BID6 BID19 BID5 BID29 BID11 BID2 .

And not only are adversarial attacks successful in white-box attacks, i.e. when the adversary has access to the DNN parameters, but they are also successful in black-box attacks, i.e. it has no access to the parameters.

Black-box attacks are successful because one can perturb an image so it misclassifies on one DNN, and the same perturbed image also has a significant chance to be misclassified by another DNN; this is known as transferability of adversarial examples BID22 ).

Due to this transferability, it is very easy to attack neural nets in a blackbox fashion BID14 BID4 .

In fact, there exist universal perturbations that can imperceptibly perturb any image and cause misclassification for any given network (MoosaviDezfooli et al. (2017) ).

There is much recent research on designing advanced adversarial attacks and defending against adversarial perturbation.

In this work, we propose to defend against adversarial attacks by changing the DNNs' output activation function to a manifold-interpolating function, in order to seamlessly utilize the training data's information when performing inference.

Together with the total variation minimization (TVM) and augmented training, we show state-of-the-art defense results on the CIFAR-10 benchmark.

Moreover, we show that adversarial images generated from attacking the DNNs with an interpolating function are more transferable to other DNNs, than those resulting from attacking standard DNNs.

Defensive distillation was recently proposed to increase the stability of DNNs which dramatically reduces the success rate of adversarial attacks BID21 , and a related approach BID27 ) cleverly modifies the training data to increase robustness against black-box attacks, and adversarial attacks in general.

To counter the adversarial perturbations, BID9 proposed to use image transformation, e.g., bit-depth reduction, JPEG compression, TVM, and image quilting.

A similar idea of denoising the input was later explored by BID17 , where they divide the input into patches, denoise each patch, and then reconstruct the image.

These input transformations are intended to be non-differentiable, thus making adversarial attacks more difficult, especially for gradient-based attacks.

BID25 noticed that small adversarial perturbations shift the distribution of adversarial images far from the distribution of clean images.

Therefore they proposed to purify the adversarial images by PixelDefend.

Adversarial training is another family of defense methods to improve the stability of DNNs BID6 BID15 BID18 .

And GANs are also employed for adversarial defense BID24 .

In BID1 , the authors proposed a straight-through estimation of the gradient to attack the defense methods that is based on the obfuscated gradient.

Meanwhile, many advanced attack methods have been proposed to attack the DNNs BID29 BID11 .Instead of using softmax functions as the DNNs' output activation, BID28 utilized a class of non-parametric interpolating functions.

This is a combination of both deep and manifold learning which causes the DNNs to sufficiently utilize the geometric information of the training data.

The authors show a significant amount of generalization accuracy improvement, and the results are more stable when one only has a limited amount of training data.

In this section, we summarize the architecture, training, and testing procedures of the DNNs with the data-dependent activation BID28 ).

An overview of training and testing of the standard DNNs with softmax output activation is shown in FIG0 and (b), respectively.

In the kth iteration of training, given a mini-batch of training data X, Y, the procedure is:Forward propagation: Transform X into features by a DNN block (ensemble of convolutional layers, nonlinearities and others), and then pass this output through the softmax activation to obtain the predictionsỸ: DISPLAYFORM0 Then the loss is computed (e.g., cross entropy) between Y andỸ: L = Loss(Y,Ỹ).Backpropagation: Update weights (Θ k−1 , W k−1 ) by gradient descent (learning rate γ): DISPLAYFORM1 Once the model is optimized, the predicted labels for testing data X are: BID28 proposed to replace the data-agnostic softmax activation by a data-dependent interpolating function, defined in the next section.

Under review as a conference paper at ICLR 2019 DISPLAYFORM2 DISPLAYFORM3 DISPLAYFORM4 te m } be a subset of X which are labeled with label function g(x).

We want to interpolate a function u that is defined on the entire manifold and can be used to label the entire dataset X. The harmonic extension is a natural and elegant approach to find such an interpolating function, which is defined by minimizing the Dirichlet energy functional: DISPLAYFORM5 with the boundary condition: DISPLAYFORM6 where w(x, y) is a weight function, typically chosen to be Gaussian: DISPLAYFORM7 ) with σ being a scaling parameter.

The Euler-Lagrange equation for Eq. FORMULA5 is: DISPLAYFORM8 By solving the linear system (Eq. FORMULA8 ), we obtain labels u(x) for unlabeled data x ∈ X/X te .

This interpolation becomes invalid when the labeled data is tiny, i.e., |X te | |X/X te |.

To resolve this issue, the weights of the labeled data is increased in the Euler-Lagrange equation, which gives: DISPLAYFORM9 The solution u(x) to Eq. (3) is named weighted nonlocal Laplacian (WNLL), denoted as DISPLAYFORM10 is the one-hot labels for the example x.

In both training and testing of the WNLL-activated DNNs, we need to reserve a small portion of data/label pairs, denoted as (X te , Y te ), to interpolate the label for new data Y. We name the reserved data (X te , Y te ) as the template.

Directly replacing softmax by WNLL has difficulties in back propagation, namely the true gradient ∂L ∂Θ is difficult to compute since WNLL defines a very complex implicit function.

Instead, to train WNLL-activated DNNs, a proxy via an auxiliary neural net ( FIG0 ) is employed.

On top of the original DNNs, we add a buffer block (a fully connected layer followed by a ReLU), and followed by two parallel branches, WNLL and the linear (fully connected) layers.

The auxiliary DNNs can be trained by alternating between training DNNs with linear and WNLL activations, respectively.

The training loss of the WNLL activation function is backpropped via a straight-through estimation approach BID1 BID3 .

At test time, we remove the linear classifier from the neural nets and use the DNN and buffer blocks together with WNLL to predict new data ( FIG0 ; here for simplicity, we merge the buffer block to the DNN block.

For a given set of testing data X, and the labeled template {(X te , Y te )}, the predicted labels for X is given bỹ DISPLAYFORM0

We consider three benchmark attack methods in this work, namely, the fast gradient sign method (FGSM) BID6 , iterative FGSM (IFGSM) BID13 , and CarliniWagner's L 2 (CW-L2) BID5 attacks.

We denote the classifier defined by the DNNs with softmax activation asỹ = f (θ, x) for a given instance (x, y).

FGSM finds the adversarial image x by maximizing the loss L(x , y), subject to the l ∞ perturbation ||x −x|| ∞ ≤ with as the attack strength.

Under the first order approximation i.e., DISPLAYFORM0 , the optimal perturbation is given by DISPLAYFORM1 IFGSM iterates FGSM to generate enhanced adversarial images, i.e., DISPLAYFORM2 where m = 1, · · · , M , x (0) = x and x = x (M ) , with M be the number of iterations.

The CW-L2 attack is proposed to circumvent defensive distillation.

For a given image-label pair (x, y), and ∀t = y, CW-L2 searches the adversarial image that will be classified to class t by solving the optimization problem: min DISPLAYFORM3 where δ is the adversarial perturbation (for simplicity, we ignore the dependence of θ in f ).The equality constraint in Eq. FORMULA15 is hard to satisfy, so instead Carlini et al. consider the surrogate DISPLAYFORM4 where Z(x) is the logit vector for an input x, i.e., output of the neural net before the softmax layer.

Z(x) i is the logit value corresponding to class i.

It is easy to see that f (x + δ) = t is equivalent to g(x + δ) ≤ 0.

Therefore, the problem in Eq. (6) can be reformulated as DISPLAYFORM5 where c ≥ 0 is the Lagrangian multiplier.

By letting δ = This unconstrained optimization problem can be solved efficiently by the Adam optimizer BID12 .

All three of the attacks clip the values of the adversarial image x to between 0 and 1.

In this work, we focus on untargeted attacks and defend against them.

For a given small batch of testing images (X, Y) and template (X te , Y te ), we denote the DNNs modified with WNLL as output activation asỸ = WNLL(Z({X, X te }), Y te ), where Z({X, X te }) is the composition of the DNN and buffer blocks as shown in FIG0 .

By ignoring dependence of the loss function on the parameters, the loss function for DNNs with WNLL activation can be written asL(X, Y, X te , Y te ).

The above attacks for DNNs with WNLL activation on the batch of images, X, are formulated below.

DISPLAYFORM0 • IFGSM DISPLAYFORM1 where m = 1, 2, · · · , N ; X (0) = X and X = X (M ) .•

CW-L2 DISPLAYFORM2 where i is the logit values of the input images X.Based on our numerical experiments, the batch size of X has minimal influence on the adversarial attack and defense.

In all of our experiments we choose the batch size of X to be 500.

Similar to BID28 , we choose the size of the template to be 500.We apply the above attack methods to ResNet-56 with either softmax or WNLL as the output activation function.

For IFGSM, we run 10 iterations of Eqs. FORMULA14 and FORMULA5 to attack DNNs with two different output activations, respectively.

For CW-L2 attacks (Eqs. (9, 12)) in both scenarios, we set the parameters c = 10 and κ = 0.

FIG2 depicts three randomly selected images (horse, automobile, airplane) from the CIFAR-10 dataset, their adversarial versions by different attack methods on ResNet-56 with two kinds of activation functions, and the TV minimized images.

All attacks successfully fool the classifiers to classify any of them correctly.

FIG2 shows that FGSM and IFGSM with perturbation = 0.02 changes the contrast of the images, while it is still easy for humans to correctly classify them.

The adversarial images of the CW-L2 attacks are imperceptible, however they are extremely strong in fooling DNNs.

FIG2 shows the images of (a) with a stronger attack, = 0.08.

With a larger , the adversarial images become more noisy.

The TV minimized images of FIG2 (a) and (b) are shown in FIG2 and FORMULA3 , respectively.

The TVM removes a significant amount of detailed information from the original and adversarial images, meanwhile it also makes it harder for humans to classify both the TV-minimized version of the original and adversarial images.

Visually, it is hard to discern the adversarial images resulting from attacking the DNNs with two types of output layers.

We consider the geometry of features of the original and adversarial images.

We randomly select 1000 training and 100 testing images from the airplane and automobile classes, respectively.

We consider two visualization strategies for ResNet-56 with softmax activation: (1) extract the original 64D features output from the layer before the softmax, and (2) apply the principle component analysis (PCA) to reduce them to 2D.

However, the principle components (PCs) do not encode the entire geometric information of the features.

Alternatively, we add a 2 by 2 fully connected (FC) layer before the softmax, then utilize the 2D features output from this newly added layer.

We verify that the newly added layer does not change the performance of ResNet-56 as shown in FIG3 , and that the training and testing performance remains essentially the same for these two cases.

Figure 4 (a) and (b) show the 2D features generated by ResNet-56 with additional FC layer for the original and adversarial testing images, respectively, where we generate the adversarial images by using FGSM ( = 0.02).

Before adversarial perturbation FIG4 ), there is a straight line that can easily separate the two classes.

The small perturbation causes the features to overlap and there is no linear classifier that can easily separate these two classes FIG4 ).

The first two PCs of the 64D features of the clean and adversarial images are shown in FIG4 and FORMULA3 , respectively.

Again, the PCs are well separated for clean images, while adversarial perturbation causes overlap and concentration.

The bottom charts of FIG4 depict the first two PCs of the 64D features output from the layer before the WNLL.

The distributions of the unperturbed training and testing data are the same, as illustrated in panels (e) and (f).

The new features are better separated which indicates that DNNs with WNLL are more robust to small random perturbation.

Panels (g) and (h) plot the features of the adversarial and TV minimized adversarial images in the test set.

The adversarial attacks move the automobiles' features to the airplanes' region and TVM helps to eliminate the outliers.

Based on our computation, most of the adversarial images of the airplane classes can be correctly classified with the interpolating function.

The training data guides the interpolating function to classify adversarial images correctly.

The fact that the adversarial perturbations change the features' distribution was also noticed in BID25 .

DISPLAYFORM0

To defend against adversarials, we combine the ideas of data-dependent activation, input transformation, and training data augmentation.

We train ResNet-56, respectively, on the original training data, the TV minimized training data, and a combination of the previous two.

On top of the datadependent activation output and augmented training, we further apply the TVM BID23 used by BID9 to transform the adversarial images to boost defensive performance.

The basic idea is to reconstruct the simplest image z from the sub-sampled image, X x, with X the mask filled by a Bernoulli binary random variable, by solving the following TVM problem where λ T V > 0 is the regularization constant.

DISPLAYFORM0 7 NUMERICAL RESULTS

To verify the efficacy of attack methods for DNNs with WNLL output activation, we consider the transferability of adversarial images.

We train ResNet-56 on the aforementioned three types of training data with either softmax or WNLL activation.

After the DNNs are trained, we attack them by FGSM, IFGSM, and CW-L2 with different .

Finally, we classify the adversarial images by using ResNet-56 with the opponent activation.

We list the mutual classification accuracy on adversarial images in Table.

1.

The adversarial images resulting from attacking DNNs with two types of activation functions are both transferable, as the mutual classification accuracy is significantly lower than testing on the clean images.

Overall we see a remarkably higher accuracy when applying ResNet-56 with WNLL activation to classify the adversarial images resulting from attacking ResNet-56 with softmax activation.

For instance, for DNNs that are trained on the original images and attacked by FGSM, DNNs with the WNLL classifier have at least 5.4% higher accuracy (56.3% v.s. 61.7% ( = 0.08)).

The accuracy improvement is more significant in many other scenarios.7.2 ADVERSARIAL DEFENSE FIG5 plots the result of adversarial defense by combining the WNLL activation, TVM, and training data augmentation.

Panels (a), (b) and (c) show the testing accuracy of ResNet-56 with and without defense on CIFAR-10 data for FGSM, IFGSM, and CW-L2, respectively.

It can be observed that with increasing attack strength, , the testing accuracy decreases rapidly.

FGSM is a relatively weak attack method, as the accuracy remains above 53.5% ( = 0.1) even with the strongest attack.

Meanwhile, the defense maintains accuracy above 71.8% ( = 0.02).

FIG5 (b) and (c) show that both IFGSM and CW-L2 can fool ResNet-56 near completely even with small .

The defense maintains the accuracy above 68.0%, 57.2%, respectively, under the CW-L2 and IFGSM attacks.

Compared to state-of-the-art defensive methods on CIFAR-10, PixelDefend, our method is much simpler and faster.

Without adversarial training, we have shown our defense is more stable to IFGSM, and more stable to all three attacks under the strongest attack than PixelDefend BID25 .

Moreover, our defense strategy is additive to adversarial training and many other defenses including PixelDefend.

To analyze the defensive contribution from each component of the defensive strategy, we separate the three parts and list the testing accuracy in Table.

2.

Simple TVM cannot defend FGSM attacks except when the DNNs are trained on the augmented data, as shown in the first and fourth horizontal blocks of the table.

WNLL activation improves the testing accuracy of adversarial attacks significantly and persistently.

Augmented training can improve the stability consistently as well.

In this paper, by analyzing the influence of adversarial perturbations on the geometric structure of the DNNs' features, we propose to defend against adversarial attack by applying a data-dependent activation function, total variation minimization on the adversarial images, and training data augmentation.

Results on ResNet-56 with CIFAR-10 benchmark reveal that the defense improves robustness to adversarial perturbation significantly.

Total variation minimization simplifies the adversarial images, which is very useful in removing adversarial perturbation.

Another interesting direction to explore is to apply other denoising methods to remove adversarial perturbation.

<|TLDR|>

@highlight

We proposal strategies for adversarial defense based on data dependent activation function, total variation minimization, and training data augmentation