Design of reliable systems must guarantee stability against input perturbations.

In machine learning, such guarantee entails preventing overfitting and ensuring robustness of models against corruption of input data.

In order to maximize stability, we analyze and develop a computationally efficient implementation of Jacobian regularization that increases classification margins of neural networks.

The stabilizing effect of the Jacobian regularizer leads to significant improvements in robustness, as measured against both random and adversarial input perturbations, without severely degrading generalization properties on clean data.

Stability analysis lies at the heart of many scientific and engineering disciplines.

In an unstable system, infinitesimal perturbations amplify and have substantial impacts on the performance of the system.

It is especially critical to perform a thorough stability analysis on complex engineered systems deployed in practice, or else what may seem like innocuous perturbations can lead to catastrophic consequences such as the Tacoma Narrows Bridge collapse (Amman et al., 1941) and the Space Shuttle Challenger disaster (Feynman and Leighton, 2001) .

As a rule of thumb, well-engineered systems should be robust against any input shifts -expected or unexpected.

Most models in machine learning are complex nonlinear systems and thus no exception to this rule.

For instance, a reliable model must withstand shifts from training data to unseen test data, bridging the so-called generalization gap.

This problem is severe especially when training data are strongly biased with respect to test data, as in domain-adaptation tasks, or when only sparse sampling of a true underlying distribution is available, as in few-shot learning.

Any instability in the system can further be exploited by adversaries to render trained models utterly useless (Szegedy et al., 2013; Goodfellow et al., 2014; Moosavi-Dezfooli et al., 2016; Papernot et al., 2016a; Kurakin et al., 2016; Madry et al., 2017; Carlini and Wagner, 2017; Gilmer et al., 2018) .

It is thus of utmost importance to ensure that models be stable against perturbations in the input space.

Various regularization schemes have been proposed to improve the stability of models.

For linear classifiers and support vector machines (Cortes and Vapnik, 1995) , this goal is attained via an L 2 regularization which maximizes classification margins and reduces overfitting to the training data.

This regularization technique has been widely used for neural networks as well and shown to promote generalization (Hinton, 1987; Krogh and Hertz, 1992; Zhang et al., 2018) .

However, it remains unclear whether or not L 2 regularization increases classification margins and stability of a network, especially for deep architectures with intertwining nonlinearity.

In this paper, we suggest ensuring robustness of nonlinear models via a Jacobian regularization scheme.

We illustrate the intuition behind our regularization approach by visualizing the classification margins of a simple MNIST digit classifier in Figure 1 (see Appendix A for more).

Decision cells of a neural network, trained without regularization, are very rugged and can be unpredictably unstable ( Figure 1a ).

On average, L 2 regularization smooths out these rugged boundaries but does not necessarily increase the size of decision cells, i.e., does not increase classification margins (Figure 1b) .

In contrast, Jacobian regularization pushes decision boundaries farther away from each training data point, enlarging decision cells and reducing instability (Figure 1c ).

The goal of the paper is to promote Jacobian regularization as a generic scheme for increasing robustness while also being agnostic to the architecture, domain, or task to which it is applied.

In support of this, after presenting the Jacobian regularizer, we evaluate its effect both in isolation as well as in combination with multiple existing approaches that are intended to promote robustness and generalization.

Our intention is to showcase the ease of use and complimentary nature of our proposed regularization.

Domain experts in each field should be able to quickly incorporate our regularizer into their learning pipeline as a simple way of improving the performance of their state-of-the-art system.

The rest of the paper is structured as follows.

In Section 2 we motivate the usage of Jacobian regularization and develop a computationally efficient algorithm for its implementation.

Next, the effectiveness of this regularizer is empirically studied in Section 3.

As regularlizers constrain the learning problem, we first verify that the introduction of our regularizer does not adversely affect learning in the case when input data remain unperturbed.

Robustness against both random and adversarial perturbations is then evaluated and shown to receive significant improvements from the Jacobian regularizer.

We contrast our work with the literature in Section 4 and conclude in Section 5.

Here we introduce a scheme for minimizing the norm of an input-output Jacobian matrix as a technique for regularizing learning with stochastic gradient descent (SGD).

We begin by formally defining the input-output Jacobian and then explain an efficient algorithm for computing the Jacobian regularizer using standard machine learning frameworks.

Let us consider the set of classification functions, f , which take a vectorized sensory signal, x ∈ R I , as input and outputs a score vector, z = f (x) ∈ R C , where each element, z c , is associated with likelihood that the input is from category, c.

1 In this work, we focus on learning this classification function as a neural network with model parameters θ, though our findings should generalize to any parameterized function.

Our goal is to learn the model parameters that minimize the classification objective on the available training data while also being stable against perturbations in the input space so as to increase classification margins.

1 Throughout the paper, the vector z denotes the logit before applying a softmax layer.

The probabilistic output of the softmax pc relates to zc via pc ≡ The input-output Jacobian matrix naturally emerges in the stability analysis of the model predictions against input perturbations.

Let us consider a small perturbation vector, ∈ R I , of the same dimension as the input.

For a perturbed input x = x + , the corresponding output values shift to

where in the second equality the function was Taylor-expanded with respect to the input perturbation and in the third equality the input-output Jacobian matrix,

was introduced.

As the function f is typically almost everywhere analytic, for sufficiently small perturbations the higher-order terms can be neglected and the stability of the prediction is governed by the input-output Jacobian.

From Equation (1), it is straightforward to see that the larger the components of the Jacobian are, the more unstable the model prediction is with respect to input perturbations.

A natural way to reduce this instability then is to decrease the magnitude for each component of the Jacobian matrix, which can be realized by minimizing the square of the Frobenius norm of the input-output Jacobian,

For linear models, this reduces exactly to L 2 regularization that increases classification margins of these models.

For nonlinear models, however, Jacobian regularization does not equate to L 2 regularization, and we expect these schemes to affect models differently.

In particular, predictions made by models trained with the Jacobian regularization do not vary much as inputs get perturbed and hence decision cells enlarge on average.

This increase in stability granted by the Jacobian regularization is visualized in Figure 1 , which depicts a cross section of the decision cells for the MNIST digit classification problem using a nonlinear neural network (LeCun et al., 1998) .

The Jacobian regularizer in Equation (3) can be combined with any loss objective used for training parameterized models.

Concretely, consider a supervised learning problem modeled by a neural network and optimized with SGD.

At each iteration, a mini-batch B consists of a set of labeled examples, {x α , y α } α∈B , and a supervised loss function, L super , is optimized possibly together with some other regularizer R(θ) -such as L 2 regularizer λWD 2 θ 2 -over the function parameter space, by minimizing the following bare loss function

To integrate our Jacobian regularizer into training, one instead optimizes the following joint loss

where λ JR is a hyperparameter that determines the relative importance of the Jacobian regularizer.

By minimizing this joint loss with sufficient training data and a properly chosen λ JR , we expect models to learn both correctly and robustly.

2 Minimizing the Frobenius norm will also reduce the L 1 -norm, since these norms satisfy the inequalities ||J(x)||F ≤ i,c Jc;i (x) ≤ √ IC||J(x)||F.

We prefer to minimize the Frobenius norm over the L 1 -norm because the ability to express the former as a trace leads to an efficient algorithm [see Equations (6) through (8)].

In the previous section we have argued for minimizing the Frobenius norm of the input-output Jacobian to improve robustness during learning.

The main question that follows is how to efficiently compute and implement this regularizer in such a way that its optimization can seamlessly be incorporated into any existing learning paradigm.

Recently, Sokolić et al. (2017) also explored the idea of regularizing the Jacobian matrix during learning, but only provided an inefficient algorithm requiring an increase in computational cost that scales linearly with the number of output classes, C, compared to the bare optimization problem (see explanation below).

In practice, such an overhead will be prohibitively expensive for many large-scale learning problems, e.g. ImageNet classification has C = 1000 target classes (Deng et al., 2009 ). (Our scheme, in contrast, can be used for ImageNet: see Appendix H.)

Here, we offer a different solution that makes use of random projections to efficiently approximate the Frobenius norm of the Jacobian.

3 This only introduces a constant time overhead and can be made very small in practice.

When considering such an approximate algorithm, one naively must trade off efficiency against accuracy for computing the Jacobian, which ultimately trades computation time for robustness.

Prior work by Varga et al. (2017) briefly considers an approach based on random projection, but without providing any analysis on the quality of the Jacobian approximation.

Here, we describe our algorithm, analyze theoretical convergence guarantees, and verify empirically that there is only a negligible difference in model solution quality between training with the exact computation of the Jacobian as compared to training with the approximate algorithm, even when using a single random projection (see Figure 2 ).

Given that optimization is commonly gradient based, it is essential to efficiently compute gradients of the joint loss in Equation (5) and in particular of the squared Frobenius norm of the Jacobian.

First, we note that automatic differentiation systems implement a function that computes the derivative of a vector such as z with respect to any variables on which it depends, if the vector is first contracted with another fixed vector.

To take advantage of this functionality, we rewrite the squared Frobienus norm as

where a constant orthonormal basis, {e}, of the C-dimensional output space was inserted in the second equality and the last equality follows from definition (2) and moving the constant vector inside the derivative.

For each basis vector e, the quantity in the last parenthesis can then be efficiently computed by differentiating the product, e · z, with respect to input parameters, x. Recycling that computational graph, the derivative of the squared Frobenius norm with respect to the model parameters, θ, can be computed through backpropagation with any use of automatic differentiation.

Sokolić et al. (2017) essentially considers this exact computation, which requires backpropagating gradients through the model C times to iterate over the C orthonormal basis vectors {e}. Ultimately, this incurs computational overhead that scales linearly with the output dimension C.

Instead, we further rewrite Equation (6) in terms of the expectation of an unbiased estimator

where the random vectorv is drawn from the (C − 1)-dimensional unit sphere S C−1 .

Using this relationship, we can use samples of n proj random vectorsv µ to estimate the square of the norm as

which converges to the true value as O(n −1/2 proj ).

The derivation of Equation (7) and the calculation of its convergence make use of random-matrix techniques and are provided in Appendix B.

Finally, we expect that the fluctuations of our estimator can be suppressed by cancellations within a mini-batch.

With nearly independent and identically distributed samples in a mini-batch of size The difference between the exact method (cyan) and the random projection method with n proj = 1 (blue) and n proj = 3 (red orange) is negligible both in terms of accuracy (a) and the norm of the input-output Jacobian (b) on the test set for LeNet' models trained on MNIST with λ JR = 0.01.

Shading indicates the standard deviation estimated over 5 distinct runs and dashed vertical lines signify the learning rate quenches.

Algorithm 1 Efficient computation of the approximate gradient of the Jacobian regularizer.

Inputs: mini-batch of |B| examples x α , model outputs z α , and number of projections n proj .

Outputs: Square of the Frobenius norm of the Jacobian J F and its gradient ∇ θ J F .

Uniform sampling from the unit sphere for each α.

1, we expect the error in our estimate to be of order (n proj |B|) −1/2 .

In fact, as shown in Figure 2 , with a mini-batch size of |B| = 100, single projection yields model performance that is nearly identical to the exact method, with computational cost being reduced by orders of magnitude.

The complete algorithm is presented in Algorithm 1.

With a straightforward implementation in PyTorch (Paszke et al., 2017) and n proj = 1, we observed the computational cost of the training with the Jacobian regularization to be only ≈ 1.3 times that of the standard SGD computation cost, while retaining all the practical benefits of the expensive exact method.

In this section, we evaluate the effectiveness of Jacobian regularization on robustness.

As all regularizers constrain the learning problem, we begin by confirming that our regularizer effectively reduces the value of the Frobenius norm of the Jacobian while simultaneously maintaining or improving generalization to an unseen test set.

We then present our core result, that Jacobian regularization provides significant robustness against corruption of input data from both random and adversarial perturbations (Section 3.2).

In the main text we present results mostly with the MNIST dataset; the corresponding experiments for the CIFAR-10 ( Krizhevsky and Hinton, 2009) and ImageNet (Deng et al., 2009 ) datasets are relegated to Appendices E and H. The following specifications apply throughout our experiments:

Datasets: The MNIST data consist of black-white images of hand-written digits with 28-by-28 pixels, partitioned into 60,000 training and 10,000 test samples (LeCun et al., 1998) .

We preprocess the data by subtracting the mean (0.1307) and dividing by the variance (0.3081) of the training data.

No regularization 49.2 ± 1.9 67.0 ± 1.7 83.3 ± 0.7 90.4 ± 0.5 98.9 ± 0.1 32.9 ± 3.3 L 2 49.9 ± 2.1 68.1 ± 1.9 84.3 ± 0.8 91.2 ± 0.5 99.2 ± 0.1 4.6 ± 0.2 Dropout 49.7 ± 1.7 67.4 ± 1.7 83.9 ± 1.8 91.6 ± 0.5 98.6 ± 0.1 21.5 ± 2.3 Jacobian 49.3 ± 2.1 68.2 ± 1.9 84.5 ± 0.9 91.3 ± 0.4 99.0 ± 0.0 1.1 ± 0.1 All Combined 51.7 ± 2.1 69.7 ± 1.9 86.3 ± 0.9 92.7 ± 0.4 99.1 ± 0.1 1.2 ± 0.0 Implementation Details:

For the MNIST dataset, we use the modernized version of LeNet-5 (LeCun et al., 1998) , henceforth denoted LeNet' (see Appendix D for full details).

We optimize using SGD with momentum, ρ = 0.9, and our supervised loss equals the standard cross-entropy with one-hot targets.

The model parameters θ are initialized at iteration t = 0 by the Xavier method (Glorot and Bengio, 2010) and the initial descent value is set to 0.

The hyperparameters for all models are chosen to match reference implementations: the L 2 regularization coefficient (weight decay) is set to λ WD = 5 · 10 −4 and the dropout rate is set to p drop = 0.5.

The Jacobian regularization coefficient λ JR = 0.01, is chosen by optimizing for clean performance and robustness on the white noise perturbation.

(See Appendix G for performance dependence on the coefficient λ JR .)

The main goal of supervised learning involves generalizing from a training set to unseen test set.

In dealing with such a distributional shift, overfitting to the training set and concomitant degradation in test performance is the central concern.

For neural networks one of the most standard antidotes to this overfitting instability is L 2 reguralization (Hinton, 1987; Krogh and Hertz, 1992; Zhang et al., 2018) .

More recently, dropout regularization has been proposed as another way to circumvent overfitting (Srivastava et al., 2014) .

Here we show how Jacobian regualarization can serve as yet another solution.

This is also in line with the observed correlation between the input-output Jacobian and generalization performance (Novak et al., 2018) .

We first verify that in the clean case, where the test set is composed of unseen samples drawn from the same distribution as the training data, the Jacobian regularizer does not adversely affect classification accuracy.

Table 1 reports performance on the MNIST test set for the LeNet' model trained on either a subsample or all of the MNIST train set, as indicated.

When learning using all 60,000 training examples, the learning rate is initially set to η 0 = 0.1 with mini-batch size |B| = 100 and then decayed ten-fold after each 50,000 SGD iterations; each simulation is run for 150,000 SGD iterations in total.

When learning using a small subsample of the full training set, training is carried out using SGD with full batch and a constant learning rate η = 0.01, and the model performance is evaluated after 10,000 iterations.

The main observation is that optimizing with the proposed Jacobian regularizer or the commonly used L 2 and dropout regularizers does not change performance on clean data within domain test samples in any statistically significant way.

Notably, when few samples are available during learning, performance improved with increased regularization in the form of jointly optimizing over all criteria.

Finally, in the right most column of Table 1 , we confirm that the model trained with all data and regularized with the Jacobian minimization objective has an order of magnitude smaller Jacobian norm than models trained without Jacobian regularization.

This indicates that while the model continues to make the same predictions on clean data, the margins around each prediction has increased as desired.

We test the limits of the generalization provided by Jacobian regularization by evaluating an MNIST learned model on data drawn from a new target domain distribution -the USPS (Hull, 1994) test set.

Here, models are trained on the MNIST data as above, and the USPS test dataset consists of 2007 black-white images of hand-written digits with Table 2 : Generalization on clean test data from an unseen domain.

LeNet' models learned with all MNIST training data are evaluated for accuracy on data from the novel input domain of USPS test set.

Here, each regularizer, including Jacobian, increases accuracy over an unregularized model.

In addition, the regularizers may be combined for the strongest generalization effects.

Averages and 95% confidence intervals are estimated over 5 distinct runs.

No regularization L 16-by-16 pixels; images are upsampled to 28-by-28 pixels using bilinear interpolation and then preprocessed following the MNIST protocol stipulated above.

Table 2 offers preliminary evidence that regularization, of each of the three forms studied, can be used to learn a source model which better generalizes to an unseen target domain.

We again find that the regularizers may be combined to increase the generalization property of the model.

Such a regularization technique can be immediately combined with state-of-the-art domain adaptation techniques to achieve further gains.

This section showcases the main robustness results of the Jacobian regularizer, highlighted in the case of both random and adversarial input perturbations.

The real world can differ from idealized experimental setups and input data can become corrupted by various natural causes such as random noise and occlusion.

Robust models should minimize the impact of such corruption.

As one evaluation of stability to natural corruption, we perturb each test input image x to x = x + crop where each component of the perturbation vector is drawn from the normal distribution with variance σ noise as

and the perturbed image is then clipped to fit into the range [0, 1] before preprocessing.

As in the domain-adaptation experiment above, models are trained on the clean MNIST training data and then tested on corrupted test data.

Results in Figure 3a show that models trained with the Jacobian regularization is more robust against white noise than others.

This is in line with -and indeed quantitatively validates -the embiggening of decision cells as shown in Figure 1 .

Adversarial Perturbations: The world is not only imperfect but also possibly filled with evil agents that can deliberately attack models.

Such adversaries seek a small perturbation to each input example that changes the model predictions while also being imperceptible to humans.

Obtaining the actual smallest perturbation is likely computationally intractable, but there exist many tractable approxima-tions.

The simplest attack is the white-box untargeted fast gradient sign method (FGSM) (Goodfellow et al., 2014) , which distorts the image as x = x + crop with

This attack aggregates nonzero components of the input-output Jacobian to a substantial effect by adding them up with a consistent sign.

In Figure 3b we consider a stronger attack, projected gradient descent (PGD) method (Kurakin et al., 2016; Madry et al., 2017) , which iterates the FGSM attack in Equation (10) k times with fixed amplitude ε FGSM = 1/255 while also requiring each pixel value to be within 32/255 away from the original value.

Even stronger is the Carlini-Wagner (CW) attack (Carlini and Wagner, 2017 ) presented in Figure 3c , which yields more reliable estimates of distance to the closest decision boundary (see Appendix F).

Results unequivocally show that models trained with the Jacobian regularization is again more resilient than others.

As a baseline defense benchmark, we implemented adversarial training, where the training image is corrupted through the FGSM attack with uniformly drawn amplitude ε FGSM ∈ [0, 0.01]; the Jacobian regularization can be combined with this defense mechanism to further improve the robustness.

5 Appendix A additionally depicts decision cells in adversarial directions, further illustrating the stabilizing effect of the Jacobian regularizer.

To our knowledge, double backpropagation (Drucker and LeCun, 1991; is the earliest attempt to penalize large derivatives with respect to input data, in which (∂L super /∂x) 2 is added to the loss in order to reduce the generalization gap.

6 Different incarnations of a similar idea have appeared in the following decades (Simard et al., 1992; Mitchell and Thrun, 1993; Aires et al., 1999; Rifai et al., 2011; Gulrajani et al., 2017; Yoshida and Miyato, 2017; Czarnecki et al., 2017; Jakubovitz and Giryes, 2018) .

Among them, Jacobian regularization as formulated herein was proposed by Gu and Rigazio (2014) to combat against adversarial attacks.

However, the authors did not implement it due to a computational concern -resolved by us in Section 2 -and instead layer-wise Jacobians were penalized.

Unfortunately, minimizing layer-wise Jacobians puts a stronger constraint on model capacity than minimizing the input-output Jacobian.

In fact, several authors subsequently claimed that the layer-wise regularization degrades test performance on clean data (Goodfellow et al., 2014; Papernot et al., 2016b) and results in marginal improvement of robustness (Carlini and Wagner, 2017) .

Very recently, full Jacobian regularization was implemented in Sokolić et al. (2017) , but in an inefficient manner whose computational overhead for computing gradients scales linearly with the number of output classes C compared to unregularized optimization, and thus they had to resort back to the layer-wise approximation above for the task with a large number of output classes.

This computational problem was resolved by Varga et al. (2017) in exactly the same way as our approach (referred to as spherical SpectReg in Varga et al. (2017) ).

As emphasized in Section 2, we performed more thorough theoretical and empirical convergence analysis and showed that there is practically no difference in model solution quality between the exact and random projection method in terms of test accuracy and stability.

Further, both of these two references deal only with the generalization property and did not fully explore strong distributional shifts and noise/adversarial defense.

In particular, we have visualized (Figure 1 ) and quantitatively borne out (Section 3) the stabilizing effect of Jacobian regularization on classification margins of a nonlinear neural network.

In this paper, we motivated Jacobian regularization as a task-agnostic method to improve stability of models against perturbations to input data.

Our method is simply implementable in any open source automatic differentiation system, and additionally we have carefully shown that the approximate nature of the random projection is virtually negligible.

Furthermore, we have shown that Jacobian regularization enlarges the size of decision cells and is practically effective in improving the generalization property and robustness of the models, which is especially useful for defense against input-data corruption.

We hope practitioners will combine our Jacobian regularization scheme with the arsenal of other tricks in machine learning and prove it useful in pushing the (decision) boundary of the field and ensuring stable deployment of models in everyday life.

We show in Figure S1 plots similar to the ones shown in Figure 1 in the main text, but with different seeds for training models and around different test data points.

Additionally, shown in Figure S2 are similar plots but with different scheme for hyperplane slicing, based on adversarial directions.

Interestingly, the adversarial examples constructed with unprotected model do not fool the model trained with Jacobian regularization.

Figure S2 : Cross sections of decision cells in the input space for LeNet' models trained on the MNIST dataset along adversarial hyperplanes.

Namely, given a test sample (black dot), the hyperplane through it is spanned by two adversarial examples identified through FGSM, one for the model trained with L 2 regularization λ WD = 0.0005 and dropout rate 0.5 but no defense (dark-grey dot; left figure) and the other for the model with the same standard regularization methods plus Jacobian regularization λ JR = 0.01 and adversarial training (white-grey dot; right figure) .

Let us denote by Ev ∼S C−1 [F (v) ] the average of the arbitrary function F over C-dimensional vectorsv sampled uniformly from the unit sphere S C−1 .

As in Algorithm 1, such a unit vector can be sampled by first sampling each component v c from the standard normal distribution N (0, 1) and then normalizing it asv ≡ v/||v||.

In our derivation, the following formula proves useful:

where e is an arbitrary C-dimensional unit vector and dµ (O) First, let us derive Equation (7).

Using Equation (11), the square of the Frobenius norm can then be written as

where in the second line we insert the identity matrix in the form I = O T O and make use of the cyclicity of the trace; in the third line we rewrite the trace as a sum over an orthonormal basis {e} of the C-dimensional output space; in the forth line Equation (11) was used; and in the last line we note that the expectation no longer depends on the basis vectors e and perform the trivial sum.

This completes the derivation of Equation (7).

Next, let us compute the variance of our estimator.

Using tricks as before, but in reverse order, yields

In this form, we use the following formula (Collins andŚniady, 2006; Collins and Matsumoto, 2009) to evaluate the first term

After the dust settles with various cancellations, the expression for the variance simplifies to

We can strengthen our claim by using the relation ||AB||

The right-hand side is independent of J and thus independent of the details of model architecture and particular data set considered.

In the end, the relative error of the random-projection estimate for ||J(x)|| 2 F with n proj random vectors will diminish as some order-one number divided by n −1/2 proj .

In addition, upon averaging ||J(x)|| 2 F over a mini-batch of samples of size |B|, we expect the relative error of the Jacobian regularization term to be additionally suppressed by ∼ 1/ |B|.

Finally, we speculate that in the large-C limit -possibly relevant for large-class datasets such as the ImageNet (Deng et al., 2009 ) -there might be additional structure in the Jacobian traces (e.g. the central-limit concentration) that leads to further suppression of the variance.

It is also possible to derive a closed-form expression for the derivative of the Jacobian regularizer, thus bypassing any need for random projections while maintaining computational efficiency.

The expression is here derived for multilayer perceptron, though we expect similar computations may be done for other models of interest.

We provide full details in case one may find it practically useful to implement explicitly in any open-source packages or generalize it to other models.

Let us denote the input x i and the output z c = z

Defining the layer-wise Jacobian as

the total input-output Jacobian is given by

The Jacobian regularizer of interest is defined as (up to the magnitude coefficient λ JR )

Its derivatives with respect to biases and weights are denoted as

Some straightforward algebra then yields

and

where we have set B

Algorithmically, we can iterate the following steps for = L, L − 1, . . .

, 1:

2.

Compute

Note that the layer-wise Jacobians, J ( ) 's, are calculated within the standard backpropagation algorithm.

The core of the algorithm is in the computation of Ω ( ) j −1 ,j in Equation (28).

It is obtained by first backpropagating from − 1 to 1, then forwardpropagating from 1 to L, and finally backpropagating from L to + 1.

It thus makes the cycle around , hence the name cyclopropagation.

In order to describe architectures of our convolutional neural networks in detail, let us associate a tuple [F, C in → C out , S, P ; M ] to a convolutional layer with filter width F , number of in-channels C in and out-channels C out , stride S, and padding P , followed by nonlinear activations and then a max-pooling layer of width M (note that M = 1 corresponds to no pooling).

Let us also associate a pair [N in → N out ] to a fully-connected layer passing N in inputs into N out units with activations and possibly dropout.

With these notations, our LeNet' model used for the MNIST experiments consists of a (28, 28, 1) input followed by a convolutional layer with [5, 1 → 6, 1, 2; 2], another one with [5, 6 → 16, 1, 0; 2], a fully-connected layer with [2100 → 120] and dropout rate p drop , another fully-connected layer with [120 → 84] and dropout rate p drop , and finally a fully-connected layer with [84 → 10], yielding 10-dimensional output logits.

For our nonlinear activations, we use the hyperbolic tangent.

For the CIFAR-10 dataset, we use the model architecture specified in the paper on defensive distillation (Papernot et al., 2016b) , abbreviated as DDNet.

Specifically, the model consists of a (32, 32, 3) input followed by convolutional layers with In addition, we experiment with a version of ResNet-18 (He et al., 2016) modified for the 32-by-32 input size of CIFAR-10 and shown to achieve strong performance on clean image recognition.

9 For this architecture, we use the standard PyTorch initialization of the parameters.

Data preproceessing and optimization hyperparameters for both architectures are specified in the next section.

For our ImageNet experiments, we use the standard ResNet-18 model available within PyTorch (torchvision.models.resnet) together with standard weight initialization.

Note that there is typically no dropout regularization in the ResNet models but we still examine the effect of L 2 regularization in addition to Jacobian regularization.

is vacuous.

9 Model available at: https://github.com/kuangliu/pytorch-cifar.

No regularization 12.9 ± 0.7 15.5 ± 0.7 20.5 ± 1.3 26.6 ± 1.0 76.8 ± 0.4 115.1 ± 1.8 L • Datasets: the CIFAR-10 dataset consists of color images of objects -divided into ten categories -with 32-by-32 pixels in each of 3 color channels, each pixel ranging in [0, 1], partitioned into 50,000 training and 10,000 test samples (Krizhevsky and Hinton, 2009 ).

The images are preprocessed by uniformly subtracting 0.5 and multiplying by 2 so that each pixel ranges in [−1, 1].

• Optimization: essentially same as for the LeNet' on MNIST, except the initial learning rate for full training.

Namely, model parameters θ are initialized at iteration t = 0 by the Xavier method (Glorot and Bengio, 2010) for DDNet and standard PyTorch initialization for ResNet-18, along with the zero initial velocity v(t = 0) = 0.

They evolve under the SGD dynamics with momentum ρ = 0.9, and for the supervised loss we use cross-entropy with one-hot targets.

For training with the full training set, mini-batch size is set as |B| = 100, and the learning rate η is initially set to η 0 = 0.01 for the DDNet and η 0 = 0.1 for the ResNet-18 and in both cases quenched ten-fold after each 50,000 SGD iterations; each simulation is run for 150,000 SGD iterations in total.

For few-shot learning, training is carried out using full-batch SGD with a constant learning rate η = 0.01, and model performance is evaluated after 10,000 iterations.

• Hyperparameters: the same values are inherited from the experiments for LeNet' on the MNIST and no tuning was performed.

Namely, the weight decay coefficient λ WD = 5·10 −4 ; the dropout rate p drop = 0.5; the Jacobian regularization coefficient λ JR = 0.01; and adversarial training with uniformly drawn FGSM amplitude ε FGSM ∈ [0, 0.01].

The results relevant for generalization properties are shown in Table S3 .

One difference from the MNIST counterparts in the main text is that dropout improves test accuracy more than L 2 regularization.

Meanwhile, for both setups the order of stability measured by ||J|| F on the test set more or less stays the same.

Most importantly, turning on the Jacobian regularizer improves the stability by orders of magnitude, and combining it with other regularizers do not compromise this effect.

The results relevant for robustness against input-data corruption are plotted in Figures S3 and S4 .

The success of the Jacobian regularizer is retained for the white-noise and CW adversarial attack.

For the PGD attack results are mixed at high degradation level when Jacobian regularization is combined with adversarial training.

This might be an artifact stemming from the simplicity of the PGD search algorithm, which overestimates the shortest distance to adversarial examples in comparison to the CW attack (see Appendix F), combined with Jacobian regularization's effect on simplifying the loss landscape with respect to the input space that the attack methods explore.

In Figure S5 , we compare the effects of various input perturbations on changing model's decision.

For each attack method, fooling L 2 distance in the original input space -before preprocessing -is measured between the original image and the fooling image as follows (for all attacks, cropping is performed to put pixels in the range [0, 1] in the orignal space): (i) for the white noise attack, a random direction in the input space is chosen and the magnitude of the noise is cranked up until the model yields wrong prediction; (ii) for the FGSM attack, the gradient is computed at a clean sample and then the magnitude ε FGSM is cranked up until the model is fooled; (iii) for the PGD attack, the attack step with ε FGSM = 1/255 is iterated until the model is fooled [as is customary for PGD and described in the main text, there is saturation constraint that demands each pixel value to be within 32/255 (MNIST) and 16/255 (CIFAR-10) away from the original clean value]; and (iv) the CW attack halts when fooling is deemed successful.

Here, for the CW attack (see Carlini and Wagner (2017) for details of the algorithm) the Adam optimizer on the logits loss (their f 6 ) is used with the learning rate 0.005, and the initial value of the conjugate variable, c, is set to be 0.01 and binary-searched for 10 iterations.

For each model and attack method, the shortest distance is evaluated for 1,000 test samples, and the test error (= 100% − test accuracy) at a given distance indicates the amount of test examples misclassified with the fooling distance below that given distance.

Below, we highlight various notable features.

• The most important highlight is that, in terms of effectiveness of attacks, CW > PGD > FGSM > white noise, duly respecting the complexity of the search methods for finding adversarial examples.

Compared to CW attack, the simple methods such as FGSM and PGD attacks could sometime yield erroneous picture for the geometry of the decision cells, especially regarding the closest decision boundary.

, we used 10,000 test examples (rather than 1,000 used for other figures) to compensate for the lack of multiple runs.

@highlight

We analyze and develop a computationally efficient implementation of Jacobian regularization that increases the classification margins of neural networks.