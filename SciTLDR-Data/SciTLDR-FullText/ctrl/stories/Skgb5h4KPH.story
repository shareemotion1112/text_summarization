We study the training process of Deep Neural Networks (DNNs) from the Fourier analysis perspective.

We demonstrate a very universal Frequency Principle (F-Principle) --- DNNs often fit target functions from low to high frequencies --- on high-dimensional benchmark datasets, such as MNIST/CIFAR10, and deep networks, such as VGG16.

This F-Principle of DNNs is opposite to the learning behavior of most conventional iterative numerical schemes (e.g., Jacobi method), which exhibits faster convergence for higher frequencies, for various scientific computing problems.

With a naive theory, we illustrate that this F-Principle results from the regularity of the commonly used activation functions.

The F-Principle implies an implicit bias that DNNs tend to fit training data by a low-frequency function.

This understanding provides an explanation of good generalization of DNNs on most real datasets and bad generalization of DNNs on parity function or randomized dataset.

Understanding the training process of Deep Neural Networks (DNNs) is a fundamental problem in the area of deep learning.

We find a common behavior of the gradient-based training process of DNNs, that is, a Frequency Principle (F-Principle):

DNNs often fit target functions from low to high frequencies during the training process.

In another word, at the early stage of training, the low-frequencies are fitted and as iteration steps of training increase, the high-frequencies are fitted.

For example, when a DNN is trained to fit y = sin(x) + sin(2x), its output would be close to sin(x) at early stage and as training goes on, its output would be close to sin(x) + sin(2x).

F-Principle was observed empirically in synthetic low-dimensional data with MSE loss during DNN training (Xu et al., 2018; Rahaman et al., 2018) .

However, in deep learning, empirical phenomena could vary from one network structure to another, from one dataset to another and could exhibit significant difference between synthetic data and highdimensional real data.

Therefore, the universality of the F-Principle remains an important problem for further study.

Especially for high-dimensional real problems, because the computational cost of high-dimensional Fourier transform is prohibitive in practice, it is of great challenge to demonstrate the F-Principle.

On the other hand, the mechanism underlying the F-Principle and its implication to the application of DNNs, e.g., design of DNN-based PDE solver, as well as their generalization ability are also important open problems to be addressed.

In this work, we design two methods, i.e., projection and filtering methods, to show that the FPrinciple exists in the training process of DNNs for high-dimensional benchmarks, i.e., MNIST (LeCun, 1998) , CIFAR10 (Krizhevsky et al., 2010) .

The settings we have considered are i) different DNN architectures, e.g., fully-connected network, convolutional neural network (CNN), and VGG16 (Simonyan & Zisserman, 2014) ; ii) different activation functions, e.g., tanh and rectified linear unit (ReLU); iii) different loss functions, e.g., cross entropy, mean squared error (MSE), and loss energy functional in variational problems.

These results demonstrate the universality of the F-Principle.

To facilitate the designs and applications of DNN-based schemes, we characterize a stark difference between DNNs and conventional numerical schemes on various scientific computing problems, where most of the conventional methods (e.g., Jacobi method) exhibit the opposite convergence behavior -faster convergence for higher frequencies.

This difference implies that DNN can be adopted to accelerate the convergence of low frequencies for computational problems.

We also intuitively explain with theories under an idealized setting how the smoothness/regularity of commonly used activation functions contributes to the F-Principle.

Note that this mechanism is rigorously demonstrated for DNNs of general settings in a subsequent work (Luo et al., 2019) .

Finally, we discuss that the F-Principle provides an understanding of good generalization of DNNs in many real datasets (Zhang et al., 2016) and poor generalization in learning the parity function (Shalev-Shwartz et al., 2017; Nye & Saxe, 2018) , that is, the F-Principle which implies that DNNs prefer low frequencies, is consistent with the property of low frequencies dominance in many real datasets, e.g., MNIST/CIFAR10, but is different from the parity function whose spectrum concentrates on high frequencies.

Compared with previous studies, our main contributions are as follows:

1.

By designing both the projection and filtering methods, we consistently demonstrate the F-Principle for MNIST/CIFAR10 over various architectures such as VGG16 and various loss functions.

2.

For the application of solving differential equations, we show that (i) conventional numerical schemes learn higher frequencies faster whereas DNNs learn lower frequencies faster by the FPrinciple, (ii) convergence of low frequencies can be greatly accelerated with DNN-based schemes.

3.

We present theories under an idealized setting to illustrate how smoothness/regularity of activation function contributes to the F-Principle.

4.

We discuss in detail the implication of the F-Principle to the generalization of DNNs that DNNs are implicitly biased towards a low frequency function and provide an explanation of good and poor generalization of DNNs for low and high frequency dominant target functions, respectively.

The concept of "frequency" is central to the understanding of F-Principle.

In this paper, the "frequency" means response frequency NOT image (or input) frequency as explained in the following.

Image (or input) frequency (NOT used in the paper): Frequency of 2-d function I : R 2 → R representing the intensity of an image over pixels at different locations.

This frequency corresponds to the rate of change of intensity across neighbouring pixels.

For example, an image of constant intensity possesses only the zero frequency, i.e., the lowest frequency, while a sharp edge contributes to high frequencies of the image.

Response frequency (used in the paper): Frequency of a general Input-Output mapping f .

For example, consider a simplified classification problem of partial MNIST data using only the data with label 0 and 1, f (x 1 , x 2 , · · · , x 784 ) : R 784 → {0, 1} mapping 784-d space of pixel values to 1-d space, where x j is the intensity of the j-th pixel.

Denote the mapping's Fourier transform asf (k 1 , k 2 , · · · , k 784 ).

The frequency in the coordinate k j measures the rate of change of f (x 1 , x 2 , · · · , x 784 ) with respect to x j , i.e., the intensity of the j-th pixel.

If f possesses significant high frequencies for large k j , then a small change of x j in the image might induce a large change of the output (e.g., adversarial example).

For a dataset with multiple classes, we can similarly define frequency for each output dimension.

For real data, the response frequency is rigorously defined via the standard nonuniform discrete Fourier transform (NUDFT), see Appendix A.

Frequency Principle: DNNs often fit target functions from low to high (response) frequencies during the training process.

An illustration of F-Principle using a function of 1-d input is in Appendix B. The F-Principle is rigorously defined through the frequency defined by the Fourier transform (Appendix A, Bracewell & Bracewell (1986) ) and the converging speed defined by the relative error.

By using high-dimensional real datasets, we then experimentally demonstrate F-Principle at the levels of both individual frequencies (projection method) and coarse-grained frequencies (filtering method).

Real datasets are very different from synthetic data used in previous studies.

In order to utilize the F-Principle to understand and better use DNNs in real datasets, it is important to verify whether the F-Principle also holds in high-dimensional real datasets.

In the following experiments, we examine the F-Principle in a training dataset of {(

where n is the size of dataset.

x i ∈ R d is a vector representing the image and y i ∈ {0, 1} 10 is the output (a one-hot vector indicating the label for the dataset of image classification).

d is the dimension of the input (d = 784 for MNIST and d = 32 × 32 × 3 for CIFAR10).

Since the high dimensional discrete Fourier transform (DFT) requires prohibitively high computational cost, in this section, we only consider one direction in the Fourier space through a projection method for each examination.

For a dataset {(x i , y i )} n−1 i=0 we consider one entry of 10-d output, denoted by y i ∈ R. The high dimensional discrete non-uniform Fourier transform of {(

The number of all possible k grows exponentially on dimension d. For illustration, in each examination, we consider a direction of k in the Fourier space, i.e., k = kp 1 , p 1 is a chosen and fixed unit vector, hence |k| = k.

Then we haveŷ k =

, where x p1,i = p 1 · x i is the projection of x i on the direction p 1 (Bracewell & Bracewell, 1986) .

For each training dataset, p 1 is chosen as the first principle component of the input space.

To examine the convergence behavior of different frequency components during the training, we compute the relative difference between the DNN output and the target function for selected important frequencies k's at each recording step, that is,

i=0 and the corresponding DNN output{h i } n−1 i=0 , respectively, along p 1 .

Note that each response frequency component,ĥ k , of DNN output evolves as the training goes.

In the following, we show empirically that the F-Principle is exhibited in the selected direction during the training process of DNNs when applied to MNIST/CIFAR10 with cross-entropy loss.

The network for MNIST is a fully-connected tanh DNN (784-400-200-10) and for CIFAR10 is two ReLU convolutional layers followed by a fully-connected DNN (800-400-400-400-10).

All experimental details of this paper can be found in Appendix C. We consider one of the 10-d outputs in each case using non-uniform Fourier transform.

As shown in Fig. 1(a) and 1(c), low frequencies dominate in both real datasets.

During the training, the evolution of relative errors of certain selected frequencies (marked by black squares in Fig. 1(a) and 1(c)) is shown in Fig. 1 (b) and 1(d).

One can easily observe that DNNs capture low frequencies first and gradually capture higher frequencies.

Clearly, this behavior is consistent with the F-Principle.

For other components of the output vector and other directions of p, similar phenomena are also observed.

The projection method in the previous section enables us to visualize the F-Principle in one direction for each examination at the level of individual frequency components.

However, demonstration by this method alone is insufficient because it is impossible to verify the F-Principle at all potentially informative directions for high-dimensional data.

To compensate the projection method, in this section, we consider a coarse-grained filtering method which is able to unravel whether, in the radially averaged sense, low frequencies converge faster than high frequencies.

The idea of the filtering method is as follows.

We split the frequency domain into two parts, i.e., a low-frequency part with |k| ≤ k 0 and a high-frequency part with |k| > k 0 , where | · | is the length of a vector.

The DNN is trained as usual by the original dataset {(x i , y i )} n−1 i=0 , such as MNIST or CIFAR10.

The DNN output is denoted as h. During the training, we can examine the convergence of relative errors of low-and high-frequency part, using the two measures below

, respectively, where· indicates Fourier transform, 1 k≤k0 is an indicator function, i.e.,

If we consistently observe e low < e high for different k 0 's during the training, then in a mean sense, lower frequencies are first captured by the DNN, i.e., F-Principle.

However, because it is almost impossible to compute above quantities numerically due to high computational cost of high-dimensional Fourier transform, we alternatively use the Fourier transform of a Gaussian functionĜ δ (k), where δ is the variance of the Gaussian function G, to approximate 1 |k|>k0 .

This is reasonable due to the following two reasons.

First, the Fourier transform of a Gaussian is still a Gaussian, i.e.,Ĝ δ (k) decays exponentially as |k| increases, therefore, it can approximate 1 |k|≤k0 byĜ δ (k) with a proper δ(k 0 ) (referred to as δ for simplicity).

Second, the computation of e low and e high contains the multiplication of Fourier transforms in the frequency domain, which is equivalent to the Fourier transform of a convolution in the spatial domain.

We can equivalently perform the examination in the spatial domain so as to avoid the almost impossible high-dimensional Fourier transform.

The low frequency part can be derived by y

where * indicates convolution operator, and the high frequency part can be derived by y high,δ i

(2) Then, we can examine

where h low,δ and h high,δ are obtained from the DNN output h, which evolves as a function of training epoch, through the same decomposition.

If e low < e high for different δ's during the training, F-Principle holds; otherwise, it is falsified.

Next, we introduce the experimental procedure.

Step One: Training.

Train the DNN by the original dataset {(x i , y i )} n−1 i=0 , such as MNIST or CIFAR10.

x i is an image vector, y i is a one-hot vector.

Step Two: Filtering.

The low frequency part can be derived by

where

is a normalization factor and

The high frequency part can be derived by y high,δ i

.

We also compute h low,δ i and h high,δ i for each DNN output h i .

Step Three: Examination.

To quantify the convergence of h low,δ and h high,δ , we compute the relative error e low and e high at each training epoch through Eq. (3).

With the filtering method, we show the F-Principle in the DNN training process of real datasets for commonly used large networks.

For MNIST, we use a fully-connected tanh-DNN (no softmax) with MSE loss; for CIFAR10, we use cross-entropy loss and two structures, one is small ReLU-CNN network, i.e., two convolutional layers, followed by a fully-connected multi-layer neural network with a softmax; the other is VGG16 (Simonyan & Zisserman, 2014) equipped with a 1024 fully-connected layer.

These three structures are denoted as "DNN", "CNN" and "VGG" in Fig. 2 , respectively.

All are trained by SGD from scratch.

More details are in Appendix C.

We scan a large range of δ for both datasets.

As an example, results of each dataset for several δ's are shown in Fig. 2 , respectively.

Red color indicates small relative error.

In all cases, the relative error of the low-frequency part, i.e., e low , decreases (turns red) much faster than that of the high-frequency part, i.e., e high .

Therefore, as analyzed above, the low-frequency part converges faster than the high-frequency part.

We also remark that, based on the above results on cross-entropy loss, the F-Principle is not limited to MSE loss, which possesses a natural Fourier domain interpretation by the Parseval's theorem.

Note that the above results holds for both SGD and GD.

Recently, DNN-based approaches have been actively explored for a variety of scientific computing problems, e.g., solving high-dimensional partial differential equations (E et al., 2017; Khoo et al., 2017; He et al., 2018; Fan et al., 2018) and molecular dynamics (MD) simulations .

However, the behaviors of DNNs applied to these problems are not well-understood.

To facilitate the designs and applications of DNN-based schemes, it is important to characterize the difference between DNNs and conventional numerical schemes on various scientific computing problems.

In this section, focusing on solving Poisson's equation, which has broad applications in mechanical engineering and theoretical physics (Evans, 2010), we highlight a stark difference between a DNN-based solver and the Jacobi method during the training/iteration, which can be explained by the F-Principle.

Consider a 1-d Poisson's equation:

We consider the example with g(x) = sin(x)+4 sin(4x)−8 sin ( (Fig. 3(a) ).

A DNN-based scheme is proposed by considering the following empirical loss function (E & Yu, 2018) ,

The second term in I emp (h) is a penalty, with constant β, arising from the Dirichlet boundary condition (7).

After training, the DNN output well matches the analytical solution u ref .

Focusing on the convergence of three peaks (inset of Fig. 3(a) ) in the Fourier transform of u ref , as shown in Fig. 3(b) , low frequencies converge faster than high frequencies as predicted by the F-Principle.

For comparison, we also use the Jacobi method to solve problem (6).

High frequencies converge faster in the Jacobi method (Details can be found in Appendix D), as shown in Fig. 3(c) .

As a demonstration, we further propose that DNN can be combined with conventional numerical schemes to accelerate the convergence of low frequencies for computational problems.

First, we solve the Poisson's equation in Eq. (6) by DNN with M optimization steps (or epochs), which needs to be chosen carefully, to get a good initial guess in the sense that this solution has already learned the low frequencies (large eigenvalues) part.

Then, we use the Jacobi method with the new initial data for the further iterations.

We use h − u ref ∞ max x∈Ω |h(x) − u ref (x)| to quantify the learning result.

As shown by green stars in Fig. 3(d) , h − u ref ∞ fluctuates after some running time using DNN only.

Dashed lines indicate the evolution of the Jacobi method with initial data set to the DNN output at the corresponding steps.

If M is too small (stop too early) (left dashed line), which is equivalent to only using Jacobi, it would take long time to converge to a small error, because low frequencies converges slowly, yet.

If M is too big (stop too late) (right dashed line), which is equivalent to using DNN only, much time would be wasted for the slow convergence of high frequencies.

A proper choice of M is indicated by the initial point of orange dashed line, in which low frequencies are quickly captured by the DNN, followed by fast convergence in high frequencies of the Jacobi method.

This example illustrates a cautionary tale that, although DNNs has clear advantage, using DNNs alone may not be the best option because of its limitation of slow convergence at high frequencies.

Taking advantage of both DNNs and conventional methods to design faster schemes could be a promising direction in scientific computing problems.

A subsequent theoretical work (Luo et al., 2019) provides a rigorous mathematical study of the FPrinciple at different frequencies for general DNNs (e.g., multiple hidden layers, different activation functions, high-dimensional inputs).

The key insight is that the regularity of DNN converts into the decay rate of a loss function in the frequency domain.

For an intuitive understanding of this key insight, we present theories under an idealized setting, which connect the smoothness/regularity of the activation function with different gradient and convergence priorities in frequency domain.

The activation function we consider is σ(x) = tanh(x), which is smooth in spatial domain and its derivative decays exponentially with respect to frequency in the Fourier domain.

For a DNN of one hidden layer with m nodes, 1-d input x and 1-d output:

We also use the notation θ = {θ lj } with θ 1j = a j , θ 2j = w j , and

,· is the Fourier transform, f is the target function.

The total loss function is defined as: L = +∞ −∞ L(k) dk.

Note that according to Parseval's theorem, this loss function in the Fourier domain is equal to the commonly used MSE loss.

We have the following theorems (The proofs are at Appendix E.).

Define W = (w 1 , w 2 , · · · , w m ) T ∈ R m .

Theorem 1.

Considering a DNN of one hidden layer with activation function σ(x) = tanh(x), for any frequencies k 1 and k 2 such that |f (k 1 )| > 0, |f (k 2 )| > 0, and |k 2 | > |k 1 | > 0, there exist positive constants c and C such that for sufficiently small δ, we have µ W :

where B δ ⊂ R m is a ball with radius δ centered at the origin and µ(·) is the Lebesgue measure.

Theorem 1 indicates that for any two non-converged frequencies, with small weights, the lowerfrequency gradient exponentially dominates over the higher-frequency ones.

Due to Parseval's theorem, the MSE loss in the spatial domain is equivalent to the L2 loss in the Fourier domain.

To intuitively understand the higher decay rate of a lower-frequency loss function, we consider the training in the Fourier domain with loss function of only two non-zero frequencies.

Theorem 2.

Considering a DNN of one hidden layer with activation function σ(x) = tanh(x).

Suppose the target function has only two non-zero frequencies k 1 and k 2 , that is,

that is, L(k 1 ) decreases faster than L(k 2 ).

There exist positive constants c and C such that for sufficiently small δ, we have

where B δ ⊂ R m is a ball with radius δ centered at the origin and µ(·) is the Lebesgue measure.

DNNs often generalize well for real problems (Zhang et al., 2016) but poorly for problems like fitting a parity function (Shalev-Shwartz et al., 2017; Nye & Saxe, 2018) despite excellent training accuracy for all problems.

Understanding the differences between above two types of problems, i.e., good and bad generalization performance of DNN, is critical.

In the following, we show a qualitative difference between these two types of problems through Fourier analysis and use the F-Principle to provide an explanation different generalization performances of DNNs.

For MNIST/CIFAR10, we examineŷ total,k = 1 n total n total −1 i=0

consists of both the training and test datasets with certain selected output component, at different directions of k in the Fourier space.

We find thatŷ total,k concentrates on the low frequencies along those examined directions.

For illustration,ŷ total,k 's along the first principle component are shown by green lines in Fig. 4(a, b) for MNIST/CIFAR10, respectively.

When only the training dataset is used,ŷ train,k well overlaps withŷ total,k at the dominant low frequencies.

For the parity function x j e −i2πk·x .

As shown in Fig. 4(c) ,

By experiments, the generalization ability of DNNs can be well reflected by the Fourier analysis.

For the MNIST/CIFAR10, we observed the Fourier transform of the output of a well-trained DNN on

faithfully recovers the dominant low frequencies, as illustrated in Fig. 4 (a) and 4(b), respectively, indicating a good generalization performance as observed in experiments.

However, for the parity function, we observed that the Fourier transform of the output of a well-trained DNN on {x i } i∈S significantly deviates fromf (k) at almost all frequencies, as illustrated in Fig. 4(c) , indicating a bad generalization performance as observed in experiments.

The F-Principle implicates that among all the functions that can fit the training data, a DNN is implicitly biased during the training towards a function with more power at low frequencies.

If the target function has significant high-frequency components, insufficient training samples will lead to artificial low frequencies in training dataset (see red line in Fig. 4(c) ), which is the wellknown aliasing effect.

Based on the F-Principle, as demonstrated in Fig. 4 (c), these artificial low frequency components will be first captured to explain the training samples, whereas the high frequency components will be compromised by DNN.

For MNIST/CIFAR10, since the power of high frequencies is much smaller than that of low frequencies, artificial low frequencies caused by aliasing can be neglected.

To conclude, the distribution of power in Fourier domain of above two types of problems exhibits significant differences, which result in different generalization performances of DNNs according to the F-Principle.

There are different approaches attempting to explain why DNNs often generalize well.

For example, generalization error is related to various complexity measures (Bartlett et al., 1999; Neyshabur et al., 2017; , local properties (sharpness/flatness) of loss functions at minima (Keskar et al., 2016; Wu et al., 2017) , stability of optimization algorithms (Hardt et al., 2015) , and implicit bias of the training process (Soudry et al., 2018; Arpit et al., 2017; Xu et al., 2018) .

On the other hand, several works focus on the failure of DNNs (Shalev-Shwartz et al., 2017; Nye & Saxe, 2018), e.g., fitting the parity function, in which a well-trained DNN possesses no generalization ability.

We propose that the Fourier analysis can provide insights into both success and failure of DNNs.

F-Principle was first discovered in (Xu et al., 2018; Rahaman et al., 2018) simultaneously through simple synthetic data and not very deep networks.

In the revised version, Rahaman et al. (2018) examines the F-Principle in the MNIST dataset.

However, they add noise to MNIST, which contaminates the labels and damages the structure of real data.

They only examine not very deep (6-layer) fully connected ReLU network with MSE loss, while cross-entropy loss is widely used.

This paper verified that F-Principle holds in the training process of MNIST and CIFAR10, both CNN and fully connected networks, very deep networks (VGG16) and various loss functions, e.g., MSE Loss, cross-entropy loss and variational loss function.

In the aspect of theoretical study, based on the key mechanism found by the theoretical study in this paper, Luo et al. (2019) shows a rigorous proof of the F-Principle for general DNNs.

The theoretical study of the gradient of tanh(x) in the Fourier domain is adopted by Rahaman et al. (2018) , in which they generalize the analysis to ReLU and show similar results.

Thm 1 is also used to analyze a nonlinear collaborative scheme for deep network training (Zhen et al., 2018) .

In the aspect of application, based on the study of the F-Principle in this paper, and Cai & Xu (2019) In all our experiments, we consistently consider the response frequency defined for the mapping function g between inputs and outputs, say R d → R and any k ∈ R d via the standard nonuniform discrete Fourier transform (NUDFT)

which is a natural estimator of frequency composition of g. (More details can be found in https: //en.wikipedia.org/wiki/Non-uniform_discrete_Fourier_transform.)

As n → ∞,ĝ k → g(x)e −i2πk·x ν(x) dx, where ν(x) is the data distribution.

We restrict all the evaluation of Fourier transform in our experiments to NUDFT of {y i } (ii) It allows us to perform the convergence analysis.

As t → ∞, in general, h(x i , t) →

y i for any i (h(x i , t) is the DNN output), leading toĥ k →ŷ k for any k. Therefore, we can analyze the convergence at different k by evaluating ∆ F (k) = |ĥ k −ŷ k |/|ŷ k | during the training.

If we use a different set of data points for frequency evaluation of DNN output, then ∆ F (k) may not converge to 0 at the end of training.

(iii)ŷ k faithfully reflect the frequency structure of training data {x i , y i } n−1 i=0 .

Intuitively, high frequencies ofŷ k correspond to sharp changes of output for some nearby points in the training data.

Then, by applying a Gaussian filter and evaluating still at {x i } n−1 i=0 , we obtain the low frequency part of training data with these sharp changes (high frequencies) well suppressed.

In practice, it is impossible to evaluate and compare the convergence of all k ∈ R d even with a proper cutoff frequency for a very large d of O(10 2 ) (MNIST) or O(10 3 ) (CIFAR10) due to curse of dimensionality.

Therefore, we propose the projection approach, i.e., fixing k at a specific direction and the filtering approach as detailed in Section 3 and 4, respectively.

To illustrate the phenomenon of F-Principle, we use 1-d synthetic data to show the evolution of relative training error at different frequencies during the training of DNN.

we train a DNN to fit a 1-d target function f (x) = sin(x) + sin(3x) + sin(5x) of three frequency components.

On n = 201 evenly spaced training samples, i.e., {x i } n−1 i=0 in [−3.14, 3.14], the discrete Fourier transform (DFT) of f (x) or the DNN output (denoted by h(x)) is computed byf k = 1 n n−1 i=0 f (x i )e −i2πik/n and

/n , where k is the frequency.

As shown in Fig. 5(a) , the target function has three important frequencies as we design (black dots at the inset in Fig. 5(a) ).

To examine the convergence behavior of different frequency components during the training with MSE, we compute the relative difference between the DNN output and the target function for the three important frequencies k's at each recording step, that is, ∆ F (k) = |ĥ k −f k |/|f k |, where | · | denotes the norm of a complex number.

As shown in Fig. 5(b) , the DNN converges the first frequency peak very fast, while converging the second frequency peak much slower, followed by the third frequency peak.

Next, we investigate the F-Principle on real datasets with more general loss functions other than MSE which was the only loss studied in the previous works (Xu et al., 2018; Rahaman et al., 2018) .

All experimental details can be found in Appendix.

C.

In Fig. 5 , the parameters of the DNN is initialized by a Gaussian distribution with mean 0 and standard deviation 0.1.

We use a tanh-DNN with widths 1-8000-1 with full batch training.

The learning rate is 0.0002.

The DNN is trained by Adam optimizer (Kingma & Ba, 2014) with the MSE loss function.

In Fig. 1 , for MNIST dataset, the training process of a tanh-DNN with widths 784-400-200-10 is shown in Fig. 1(a) and 1(b) .

For CIFAR10 dataset, results are shown in Fig. 1(c) and 1(d) of a ReLU-CNN, which consists of one convolution layer of 3 × 3 × 64, a max pooling of 2 × 2, one convolution layer of 3 × 3 × 128, a max pooling of 2 × 2, followed by a fully-connected DNN with widths 800-400-400-400-10.

For both cases, the output layer of the network is equipped with a softmax.

The network output is a 10-d vector.

The DNNs are trained with cross entropy loss by Adam optimizer (Kingma & Ba, 2014) . (a, b) are for MNIST with a tanh-DNN.

The learning rate is 0.001 with batch size 10000.

After training, the training accuracy is 0.951 and test accuracy is 0.963.

The amplitude of the Fourier coefficient with respect to the fourth output component at each frequency is shown in (a), in which the red dots are computed using the training data.

Selected frequencies are marked by black squares.

(b) ∆ F (k) at different training epochs for the selected frequencies. (c, d) are for CIFAR10 dataset.

We use a ReLU network of a CNN followed by a fully-connected DNN.

The learning rate is 0.003 with batch size 512.

(c) and (d) are the results with respect to the ninth output component.

After training, the training accuracy is 0.98 and test accuracy is 0.72.

In Fig. 2 , for MNIST, we use a fully-connected tanh-DNN with widths 784-400-200-10 and MSE loss; for CIFAR10, we use cross-entropy loss and a ReLU-CNN, which consists of one convolution layer of 3 × 3 × 32, a max pooling of 2 × 2, one convolution layer of 3 × 3 × 64, a max pooling of 2 × 2, followed by a fully-connected DNN with widths 400-10 and the output layer of the network is equipped with a softmax.

The learning rate for MNIST and CIFAR10 is 0.015 and 0.003, respectively.

The networks are trained by Adam optimizer (Kingma & Ba, 2014) with batch size 10000.

For VGG16, the learning rate is 10 −5 .

The network is trained by Adam optimizer (Kingma & Ba, 2014) with batch size 500.

In Fig. 3 , the samples are evenly spaced in [0, 1] with sample size 1001.

We use a DNN with widths 1-4000-500-400-1 and full batch training by Adam optimizer (Kingma & Ba, 2014) .

The learning rate is 0.0005.

β is 10.

The parameters of the DNN are initialized following a Gaussian distribution with mean 0 and standard deviation 0.02.

In Fig. 4 , the settings of (a) and (b) are the same as the ones in Fig. 1 .

For (c), we use a tanh-DNN with widths 10-500-100-1, learning rate 0.0005 under full batch-size training by Adam optimizer (Kingma & Ba, 2014) .

The parameters of the DNN are initialized by a Gaussian distribution with mean 0 and standard deviation 0.05.

Consider a one-dimensional (1-d) Poisson's equation:

u(x) = 0, x = −1, 1.

[−1, 1] is uniformly discretized into n + 1 points with grid size h = 2/n.

The Poisson's equation in Eq. (9) can be solved by the central difference scheme,

resulting a linear system Au = g,

where

A class of methods to solve this linear system is iterative schemes, for example, the Jacobi method.

Let A = D − L − U , where D is the diagonal of A, and L and U are the strictly lower and upper triangular parts of −A, respectively.

Then, we obtain

At step t ∈ N, the Jacobi iteration reads as

We perform the standard error analysis of the above iteration process.

Denote u * as the true value obtained by directly performing inverse of A in Eq. (11).

The error at step t + 1 is e t+1 = u t+1 − u * .

Then, e t+1 = R J e t , where

The converging speed of e t is determined by the eigenvalues of R J , that is,

and the corresponding eigenvector v k 's entry is

So we can write

where α t k can be understood as the magnitude of e t in the direction of v k .

Then,

.

Therefore, the converging rate of e t in the direction of v k is controlled by λ k .

Since

the frequencies k and (n − k) are closely related and converge with the same rate.

Consider the frequency k < n/2, λ k is larger for lower frequency.

Therefore, lower frequency converges slower in the Jacobi method.

The activation function we consider is σ(x) = tanh(x).

For a DNN of one hidden layer with m nodes, 1-d input x and 1-d output:

where w j , a j , and b j are called parameters, in particular, w j and a j are called weights, and b j is also known as a bias.

In the sequel, we will also use the notation θ = {θ lj } with θ 1j = a j , θ 2j = w j , and

where the Fourier transformation and its inverse transformation are defined as follows:

The Fourier transform of σ(w j x + b j ) with w j , b j ∈ R, j = 1, · · · , m reads as

Thusĥ

We define the amplitude deviation between DNN output and the target function f (x) at frequency k as

For readers' reference, we list the partial derivatives of L(k) with respect to parameters

where

The descent increment at any direction, say, with respect to parameter θ lj , is

The absolute contribution from frequency k to this total amount at θ lj is

where θ j {w j , b j , a j }, θ lj ∈ θ j , F lj (θ j , k) is a function with respect to θ j and k, which can be found in one of Eqs. (24, 25, 26) .

When the component at frequency k whereĥ(k) is not close enough tof (k), exp (−|πk/2w j |) would dominate G lj (θ j , k) for a small w j .

Through the above framework of analysis, we have the following theorem.

Define

Theorem.

Consider a one hidden layer DNN with activation function σ(x) = tanh x. For any frequencies k 1 and k 2 such that |f (k 1 )| > 0, |f (k 2 )| > 0, and |k 2 | > |k 1 | > 0, there exist positive constants c and C such that for sufficiently small δ, we have µ W :

where B δ ⊂ R m is a ball with radius δ centered at the origin and µ(·) is the Lebesgue measure.

We remark that c and C depend on k 1 , k 2 , |f (k 1 )|, |f (k 2 )|, sup |a i |, sup |b i |, and m.

Proof.

To prove the statement, it is sufficient to show that µ(S lj,δ )/µ(B δ ) ≤ C exp(−c/δ) for each l, j, where

We prove this for S 1j,δ , that is, θ lj = a j .

The proofs for θ lj = w j and b j are similar.

Without loss of generality, we assume that k 1 , k 2 > 0, b j > 0, and w j = 0, j = 1, · · · , m. According to Eq. (24), the inequality |

Therefore, lim

For W ∈ B δ with sufficiently small δ, A(

wj − φ(k 2 ))| ≤ 1 and that for sufficiently small δ,

Thus, inequality (32) implies that

Noticing that 2 π |x| ≤ | sin x| (|x| ≤ π 2 ) and Eq. (34), we have for W ∈ S lj,δ , for some q ∈ Z,

that is,

where

and c 2 = π(k 2 − k 1 ).

Define I := I + ∪ I − where

For w j > 0, we have for some q ∈ Z,

Since W ∈ B δ and c 1 exp(−c 2 /δ) + arg(f (k 1 )) ≤ 2π, we have bj k1

2π+qπ ≤ w j ≤ δ.

Then Eq. (40) only holds for some large q, more precisely, q ≥ q 0 := bj k πδ − 2.

Thus we obtain the estimate for the (one-dimensional) Lebesgue measure of I

The similar estimate holds for µ(I − ), and hence

T is in a ball with radius δ in R m−1 .

Therefore, we final arrive at the desired estimate

where ω m is the volume of a unit ball in R m .

Theorem.

Considering a DNN of one hidden layer with activation function σ(x) = tanh(x).

Suppose the target function has only two non-zero frequencies k 1 and k 2 , that is, |f (k 1 )| > 0, |f (k 2 )| > 0, and |k 2 | > |k 1 | > 0, and |f (k)| = 0 for k = k 1 , k 2 .

Consider the loss function of L = L(k 1 ) + L(k 2 ) with gradient descent training.

Denote

that is, L(k 1 ) decreases faster than L(k 2 ).

There exist positive constants c and C such that for sufficiently small δ, we have

where B δ ⊂ R m is a ball with radius δ centered at the origin and µ(·) is the Lebesgue measure.

Proof.

By gradient descent algorithm, we obtain

To obtain

it is sufficient to have ∂L(k 1 ) ∂θ lj > ∂L(k 2 ) ∂θ lj .

Eqs. (43, 44) also yield to ∂L(k 1 ) ∂t < 0.

Therefore, Eq. (45) is a sufficient condition for S. Based on the theorem 1, we have proved the theorem 2.

We train a DNN to fit a natural image (See Fig. 6(a) ), a mapping from coordinate (x, y) to gray scale strength, where the latter is subtracted by its mean and then normalized by the maximal absolute value.

First, we initialize DNN parameters by a Gaussian distribution with mean 0 and standard deviation 0.08 (initialization with small parameters).

From the snapshots during the training process, we can see that the DNN captures the image from coarse-grained low frequencies to detailed high frequencies ( Fig. 6(b) ).

As an illustration of the F-Principle, we study the Fourier transform of the image with respect to x for a fixed y (red dashed line in Fig. 6(a) , denoted as the target function f (x) in the spatial domain).

The DNN can well capture this 1-d slice after training as shown in Fig. 6(c) .

Fig. 6 (d) displays the amplitudes |f (k)| of the first 40 frequency components.

Due to the small initial parameters, as an example in Fig. 6(d) , when the DNN is fitting low-frequency components, high frequencies stay relatively small.

As the relative error shown in Fig. 6 (e), the first five frequency peaks converge from low to high in order.

Next, we initialize DNN parameters by a Gaussian distribution with mean 0 and standard deviation 1 (initialization with large parameters).

After training, the DNN can well capture the training data, as shown in the left in Fig. 6(f) .

However, the DNN output at the test pixels are very noisy, as shown in the right in Fig. 6(f) .

For the pixels at the red dashed lines in Fig. 6(a) , as shown in Fig. 6(g) , the DNN output fluctuates a lot.

Compared with the case of small initial parameters, as shown in Fig. 6(h) , the convergence order of the first five frequency peaks do not have a clear order.

<|TLDR|>

@highlight

In real problems, we found that DNNs often fit target functions from low to high frequencies during the training process.

@highlight

This paper analyzes the loss of neural networks in the Fourier domain and finds that DNNs tend to learn low-frequency components before high-frequency ones.

@highlight

The paper studies the training process of NNs through Fourier analysis, concluding that NNs learn low frequency components before high frequency components.