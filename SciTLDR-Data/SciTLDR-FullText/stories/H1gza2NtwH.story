The geometric properties of loss surfaces, such as the local flatness of a solution, are associated with generalization in deep learning.

The Hessian is often used to understand these geometric properties.

We investigate the differences between the eigenvalues of the neural network Hessian evaluated over the empirical dataset, the Empirical Hessian, and the eigenvalues of the Hessian under the data generating distribution, which we term the True Hessian.

Under mild assumptions, we use random matrix theory to show that the True Hessian has eigenvalues of smaller absolute value than the Empirical Hessian.

We support these results for different SGD schedules on both a 110-Layer ResNet and VGG-16.

To perform these experiments we propose a framework for spectral visualization, based on GPU accelerated stochastic Lanczos quadrature.

This approach is an order of magnitude faster than state-of-the-art methods for spectral visualization, and can be generically used to investigate the spectral properties of matrices in deep learning.

The extraordinary success of deep learning in computer vision and natural language processing has been accompanied by an explosion of theoretical (Choromanska et al., 2015a; b; Pennington & Bahri, 2017) and empirical interest in their loss surfaces, typically through the study of the Hessian and its eigenspectrum (Ghorbani et al., 2019; Li et al., 2017; Sagun et al., 2016; Wu et al., 2017) .

Exploratory work on the Hessian, and its evolution during training (e.g., Jastrzębski et al., 2018) , attempts to understand why optimization procedures such as SGD can discover good solutions for training neural networks, given complex non-convex loss surfaces.

For example, the ratio of the largest to smallest eigenvalues, known as the condition number, determines the convergence rate for first-order optimization methods on convex objectives (Nesterov, 2013) .

The presence of negative eigenvalues indicates non-convexity even at a local scale.

Hessian analysis has also been a primary tool in further explaining the difference in generalization of solutions obtained, where under Bayesian complexity frameworks, flatter minima, which require less information to store, generalize better than sharp minima (Hochreiter & Schmidhuber, 1997) .

Further work has considered how large batch vs small batch stochastic gradient descent (SGD) alters the sharpness of solutions (Keskar et al., 2016) , with smaller batches leading to convergence to flatter solutions, leading to better generalization.

These geometrical insights have led to generalization procedures, such as taking the Cesàro mean of the weights along the SGD trajectory , and algorithms that optimize the model to select for local flatness (Chaudhari et al., 2016) .

Flat regions of weight space are more robust under adversarial attack (Yao et al., 2018) .

Moreover, the Hessian defines the curvature of the posterior over weights in the Laplace approximation for Bayesian neural networks (MacKay, 1992; 2003) , and thus crucially determines its performance.

In this paper we use random matrix theory to analyze the spectral differences between the Empirical Hessian, evaluated via a finite data sample (hence related to the empirical risk) and what we term the True Hessian, given under the expectation of the true data generating distribution.

1 1 We consider loss surfaces that correspond to risk surfaces in statistical learning theory terminology.

In particular, we show that the differences in extremal eigenvalues between the True Hessian and the Empirical Hessian depend on the ratio of model parameters to dataset size and the variance per element of the Hessian.

Moreover, we show that that the Empirical Hessian spectrum, relative to that of the True Hessian, is broadened; i.e. the largest eigenvalues are larger and the smallest smaller.

We support this theory with experiments on the CIFAR-10 and CIFAR-100 datasets for different learning rate schedules using a large modern neural network, the 110 Layer PreResNet.

It is not currently known if key results, such as (1) the flatness or sharpness of good and bad optima, (2) local non-convexity at the end of training, or (3) rank degeneracy hold for the True Hessian in the same way as for the Empirical Hessian.

We hence provide an investigation of these foundational questions.

Previous work has used random matrix theory to study the spectra of neural network Hessians under assumptions such as normality of inputs and weights, to show results such as the decreasing difference in loss value between the local and global minima (Choromanska et al., 2015a) and the fraction of negative eigenvalues at final points in training (Pennington & Bahri, 2017) .

To the authors knowledge no theoretical work has been done on the nature of the difference in spectra between the True Hessian and Empirical Hessian in machine learning.

Previous empirical work on neural network loss surfaces (Ghorbani et al., 2019; Papyan, 2018; Sagun et al., 2017; Jastrzębski et al., 2018) has also exclusively focused on the Empirical Hessian or a sub-sample thereof.

The work closest in spirit to our work is the spiked covariance literature, which studies the problem of learning the true covariance matrix given by the generating distribution, from the noisy sample covariance matrix.

This problem is studied in mathematics and physics (Baik & Silverstein, 2004; Bloemendal et al., 2016b; a) with applications leading to extensively improved results in both sparse Principal Component Analysis (Johnstone & Lu, 2004) , portfolio theory (Laloux et al., 1999) and Bayesian covariance matrix spectrum estimation (Everson & Roberts, 2000) .

A key concept from this literature, is the ratio of parameters to data samples and its effect on the observed spectrum; this ratio is mentioned in (Pennington & Bahri, 2017) , but it is not used to determine the perturbation between the Empirical Hessian and True Hessian. (Kunstner et al., 2019) consider circumstances under which the Empirical Fisher information matrix is a bad proxy to the True Fisher matrix, but do not use a random matrix theory to characterise the spectral perturbations between the two.

We list our main contributions below.

• We introduce the concept of the True Hessian, discuss its importance and investigate it both theoretically and empirically.

• We use random matrix theory to analytically derive the eigenvalue perturbations between the True Hessian and the Empirical Hessian, showing the spectrum of the Empirical Hessian to be broadened.

• We visualize the True Hessian spectrum by combining a GPU accelerated stochastic Lanczos quadrature (Gardner et al., 2018) with data-augmentation.

Our spectral visualization technique is an order of magnitude faster than recent iterative methods (Ghorbani et al., 2019; Papyan, 2018) , requires one less hand-tuned hyper-parameter, and is consistent with the observed moment information.

This procedure can be generically used to compute the spectra of large matrices (e.g., Hessians, Fisher information matrices) in deep learning.

For an input x ∈ R dx and output y ∈ R dy we have a given prediction function h(·; ·) : R dx × R P → R dy , we consider the family of prediction functions parameterised by a weight vector w, i.e., H := {h(·; w) : w ∈ R P } with a given loss (h(x; w), y) :

R dx × R dy → R.

Ideally we would vary w such that we minimize the loss over our data generating distribution ψ(x, y), known as the true risk.

with corresponding gradient g true (w) = ∇R true (w) and Hessian H true (w) = ∇∇R true (w).

If the loss is a negative log likelihood, then the true risk is the expected negative log likelihood per data point under the data generating distribution.

However, given a dataset of size N , we only have access to the empirical or full risk

and the gradients g emp (w) and Hessians H emp (w) thereof.

The Hessian describes the curvature at that point in weight space w and hence the risk surface can be studied through the Hessian.

Weight vectors which achieve low values of equation 2 do not necessarily achieve low values of equation 1.

Their difference is known as the generalization gap.

We rewrite our Empirical Hessian as

where 2 ε(w) ≡ H emp (w) − H true (w).

A symmetric matrix with independent normally distributed elements of equal variance is known as the Gaussian Orthogonal Ensemble (GOE) and a matrix with independent non-Gaussian elements of equal variance is known as the Wigner ensemble.

By the spectral theorem, we can rewrite H emp in terms of its eigenvalue, eigenvector pairs, [λ i , φ i ]:

The magnitude of the eigenvalues represent the magnitude of the curvature in the direction of the corresponding eigenvectors.

Often the magnitude of the largest eigenvalue λ 1 , or the Frobenius norm (given by the square root of the sum of all the eigenvalues squared (

, is used to define the sharpness of an optimum (e.g., Jastrzębski et al., 2018; .

The normalized mean value of the spectrum, also known as the trace P −1 P i λ 2 i , has also been used (Dinh et al., 2017) .

In all of these cases a larger value indicates greater sharpness.

Other definitions of flatness have looked at counting the number of 0 (Sagun et al., 2016) , or close to 0 (Chaudhari et al., 2016), eigenvalues.

It is often argued that flat solutions provide better generalization because they are more robust to shifts between the train and test loss surfaces (e.g., Keskar et al., 2016 ) that exist because of statistical differences between the training and test samples.

It is then compelling to understand whether flat solutions associated with the True Hessian would also provide better generalization, since the True Hessian is formed from the full data generating distribution.

If so, there may be other important reasons why flat solutions provide better generalization.

Reparametrization can give the appearance of flatness or sharpness from the perspective of the Hessian.

However, it is common practice to hold the parametrization of the model fixed when comparing the Hessian at different parameter settings, or evaluated over different datasets.

We show that the elementwise difference between the True and Empirical Hessian converges to a zero mean normal random variable, assuming a Lipshitz bounded gradient.

Moreover, we derive a precise analytic relationship between the values of the extremal values for both the Empirical and True Hessians, showing the eigenvalues for the Empirical Hessian to be larger in magnitude.

This result indicates that the true risk surface is flatter than its empirical counterpart.

We establish some weak regularity conditions on the loss function and data, under which the elements of ε(w) converge to zero mean normal random variables.

Lemma 1.

For an L-Lipshitz-continuous-gradient and almost everywhere twice differentiable loss function (h(x; w), y), the True Hessian elements are strictly bounded in the range

Proof.

By the fundamental theorem of calculus and the definition of Lipshitz continuity λ max ≤ L

Lemma 2.

For unbiased independent samples drawn from the data generating distribution and an L-Lipshitz loss the difference between the True Hessian and Empirical Hessian converges element-wise to a zero mean, normal random variable with variance ∝ 1/N .

Proof.

The difference between the Empirical and True Hessian ε(w) is given as

By Lemma 1, the Hessian elements are bounded, hence the moments are bounded and using independence and the central limit theorem, equation 6 converges almost surely to a normal random variable P(µ jk , σ 2 jk /N ).

Remark.

For finite P and N → ∞, i.e. q = P/N → 0, |ε(w)| → 0 we recover the True Hessian.

Similarly in this limit our empirical risk converges almost surely to our true risk, i.e. we eliminate the generalization gap.

However, in deep learning typically the network size eclipses the dataset size by orders of magnitude.

In order to derive analytic results, we move to the large Dimension limit, where P, N → ∞ but P/N = q > 0 and employ the machinery of random matrix theory to derive results for the perturbations on the eigenspectrum between the True Hessian and Empirical Hessian.

This differs from the classical statistical regime where q → 0.

We primarily focus on the regime when q 1.

In order to make the analysis tractable, we introduce two further assumptions on the nature of the elements ε(w).

Assumption 1.

The elements of ε(w) are identically and independently Gaussian distributed N (0, σ 2 ) up to the Hermitian condition.

Assumption 2.

H true is of low rank r P .

Under assumption 1, ε(w) becomes a Gaussian Orthogonal Ensemble (GOE) and for the GOE we can prove the following Lemma 3.

We discuss the necessity of the assumptions in Section 4.3.

Lemma 3.

The extremal eigenvalues [λ 1 , λ P ] of the matrix sum A + B/ √ P , where A ∈ R P ×P is a matrix of finite rank r with extremal eigenvalues [λ 1 , λ P ] and B ∈ R P ×P is a GOE matrix with element variance σ 2 are given by

Proof.

See Appendix.

Proof.

The result follows directly from Lemmas 2 and 3.

Remark.

This result shows that the extremal eigenvalues of the Empirical Hessian are larger in magnitude than those of the True Hessian: Although we only state this result for the extremal eigenvalues, it holds for any number of well separated outlier eigenvalues.

This spectral broadening effect has already been observed empirically when moving from (larger) training to the (smaller) test set (Papyan, 2018) .

Intuitively variance of the Empirical Hessian eigenvalues can be seen as the variance of the True Hessian eigenvalues plus the variance of the Hessian, as we show in Appendix F. We also note that if the condition |λ i | > P/N σ is not met, the value of the positive and negative extremal eigenvalues are completely determined by the noise matrix.

For Assumption 1, we note that the results for the Wigner ensemble (of which the GOE is a special case) can be extended to non-identical element variances (Tao, 2012) and element dependence (Götze et al., 2012; Schenker & Schulz-Baldes, 2005) .

Hence, similarly to Pennington & Bahri (2017) , under extended technical conditions we expect our results to hold more generally.

Assumption 2 can be relaxed if the number of extremal eigenvalues is much smaller than P , and the bulk of H true (w)'s eigenspectra also follows a Wigner ensemble which is mutually free with that of ε(w).

Furthermore, corrections to our results for finite P scale as P −1/4 for matrices with finite 4'th moments and P

for all finite moments (Bai, 2008).

In order to perform spectral analysis on the Hessian of typical neural networks, with tens of millions of parameters, we avoid the infeasible O(P 3 ) eigen-decomposition and use the stochastic Lanczos quadrature (SLQ) algorithm, in conjunction with GPU acceleration.

Our procedure in this section is a general-purpose approach for efficiently computing the spectra of large matrices in deep learning.

The Lanczos algorithm (Meurant & Strakoš, 2006 ) is a power iteration algorithm variant which by enforcing orthogonality and storing the Krylov subspace, K m+1 (H, v) = {v, Hv, .., H m v}, optimally approximates the extremal and interior eigenvalues (known as Ritz values).

It requires Hessian vector products, for which we use the Pearlmutter trick (Pearlmutter, 1994) with computational cost O(N P ), where N is the dataset size and P is the number of parameters.

Hence for m steps the total computational complexity including re-orthogonalisation is O(N P m) and memory cost of O(P m).

In order to obtain accurate spectral density estimates we re-orthogonalise at every step (Meurant & Strakoš, 2006) .

We exploit the relationship between the Lanczos method and Gaussian quadrature, using random vectors to allow us to learn a discrete approximation of the spectral density.

A quadrature rule is a relation of the form,

for a function f , such that its Riemann-Stieltjes integral and all the moments exist on the measure dµ(λ), on the interval [a, b] and where R[f ] denotes the unknown remainder.

The nodes t j of the Gauss quadrature rule are given by the Ritz values and the weights (or mass) ρ j by the squares of the first elements of the normalized eigenvectors of the Lanczos tri-diagonal matrix (Golub & Meurant, 1994) .

For zero mean, unit variance random vectors, using the linearity of trace and expectation

hence the measure on the LHS of equation 9 corresponds to that of the underlying spectral density.

The error between the expectation over the set of all zero mean, unit variance vectors v and the Monte Carlo sum used in practice can be bounded (Hutchinson, 1990; Roosta-Khorasani & Ascher, 2015) , but the bounds are too loose to be of any practical value (Granziol & Roberts, 2017) .

However, in the high dimensional regime N → ∞, we expect the squared overlap of each random vector with an eigenvector of H, |v T φ i | 2 ≈ 1 P ∀i, with high probability (Cai et al., 2013) .

This result can be seen intuitively by looking at the normalized overlap squared between two random Rademacher vectors, where each element is drawn from the distribution ±1 with equal probability, which gives 1/P .

The Lanczos algorithm can also be easily parallelized, since it largely involves matrix multiplicationsa major advantage for computing eigenvalues of large matrices in deep learning.

We exploit GPU parallelization in performing Lanczos (Gardner et al., 2018) , for significant acceleration.

In recent work using SLQ for evaluating neural network spectra (Papyan, 2018; Ghorbani et al., 2019) , the Hessian vector product is averaged over a set of random vectors, typically ≈ 10.

The resulting discrete moment matched approximation to the spectra in equation 9 is smoothed using a Gaussian kernel N (λ i , σ

2 ).

The use of multiple random vectors can be seen as reducing the variance of the vector overlap in a geometric equivalent of the central limit theorem.

Whilst this may be desirable for general spectra, if we believe the spectra to be composed of outliers and a bulk as is empirically observed (Sagun et al., 2016; , then the same self-averaging procedure happens with the bulk.

We converge to the spectral outliers quickly using the Lanczos algorithm, as shown in the Appendix by the known convergence theorem 3.

Hence all the spectral information we need can be gleaned from a single random vector.

Furthermore, it has been proven that kernel smoothing biases the moment information obtained by the Lanczos method (Granziol et al., 2018) .

We hence avoid smoothing (and the problem of choosing σ), along with the ≈ 10× increased computational cost, by plotting the discrete spectral density implied by equation 9 for a single random vector, in the form of a stem plot.

It is possible to determine what fraction of eigenvalues is near the origin by evaluating the weight of the Ritz value(s) closest to the origin.

But due to the discrete nature of the Ritz values, which give the nodes of the associated moment-matched spectral density in equation 9, it is not possible to see exactly whether the Hessian of neural networks is rank degenerate.

It is also not possible to exactly determine what fraction of eigenvalues are negative.

Furthermore, smoothing the density and using quadrature has the problems discussed in Section 5.2.

Instead, we determine the fraction of negative eigenvalues as the weight of negative Ritz values which are well separated from the smallest magnitude Ritz value(s), to avoid counting a split of the degenerate mass.

In order to test the validity of our theoretical results in Section 4.2, we control the parameter q = P/N by applying data-augmentation and then running GPU powered Lanczos method (Gardner et al., 2018) on the augmented dataset at the final set of weights w f inal at the end of the training procedure.

We use random horizontal flips, 4 × 4 padding with zeros and random 32 × 32 crops.

We neglect the dependence of the augmented dataset on the input dataset in our analysis, but we would expect the effective number of data points N ef f < N due to degeneracy in the samples.

For the Emprical Hessian, we use the full training and test dataset N = 50, 000 with no augmentation.

We use PyTorch (Paszke et al., 2017) .

We train a 110 layer pre-activated ResNet (He et al., 2016) neural network architecture on CIFAR-10 and CIFAR-100 using SGD with momentum set at ρ = 0.9 and data-augmentation.

We include further plots and detail the training procedure for the VGG-16 (Simonyan & Zisserman, 2014) in Appendix I. In order to investigate whether empirically observed phenomena for the Empirical Hessian such as sharp minima generalizing more poorly than their flatter counterparts also hold for the True Hessian, we run two slightly different SGD decayed learning rate schedules (equation 11) on a 110-Layer ResNet .

For both schedules we use r = 0.01.

For the Normal schedule we use α 0 = 0.1 and T = 225 whereas for the Overfit schedule we use α 0 = 0.001 and T = 1000.

We plot the training curves in the Appendices B, C and include the best training and test accuracies and losses, along with the extremal eigenvalues of the Empirical Hessian and Augmented Hessian in Table 1 for CIFAR-100 and Table 2 We then run SLQ with m = 100 on the final set of training weights to compute an approximation to the spectrum for both schedules on the full 50, 000 training set (i.e the Empirical Hessian) and with m = 80 on an augmented 1, 500, 000 data-set as a proxy for the True Hessian, which we denote the Augmented Hessian.

The 110-Layer ResNet has 1, 169, 972/1, 146, 842 parameters for CIFAR-100/CIFAR-10 respectively, hence for both we have q < 1.

We primarily comment on the extremal eigenvalues of the Augmented Hessian and their deviation from those of the Empirical Hessian.

We also investigate the spectral density at the origin and the change in negative spectral mass.

A sharp drop in negative spectral mass, indicates due to Theorem 1 that most of the negative spectral mass is due to the perturbing GOE.

Figure 4 : Augmented 1, 500, 000 data Hessian spectrum for CIFAR-100 SGD.

We plot the Empirical Hessian spectrum for the end of training for CIFAR-100 in Figure 1 and the Augmented Hessian in Figure 4 .

We plot the same discrete spectral density plots for the Empirical Hessian for CIFAR-10 in Figure 2 and the Augmented Hessian in Figure 3 .

We note that the effects predicted by Theorem 1 are empirically supported by the differences between the Empirical and Augmented Hessian.

For CIFAR-100 the extremal eigenvalues shrink from [−6.54, 20 .56] (Figure 1 −4 ] respectively for CIFAR-100/CIFAR-10.

We note that despite reasonably extreme spectral shrinkage, the sharpest values of the Overfit schedule augmented Hessians are still significantly larger than those of the Normal schedule, hence the intuition of sharp minima generalizing poorly still holds for the True Hessian.

The fraction of negative Ritz values for the Empirical Hessian is 43/100, with relative negative spectral mass of 0.032.

For the Augmented Hessian, the number of negative Ritz values is 0.

For the CIFAR-10 Overfit Augmented Hessian, there is one negative Ritz value very close to the origin with a large relative spectral mass 0.086, hence it could also be part of the spectral peak at the origin.

The vast majority of eigen-directions of the Augmented Hessian are, like those of the Empirical Hessian, locally extremely close to flat.

We show this by looking at the 3 Ritz values of largest weight for all learning rate schedules and datasets, shown in Table 3 .

The majority of the spectral bulk is carried by the 3 closest weights (≈ 0.99) to the origin for all learning rate schedules and datasets.

In instances (CIFAR-100 SGD OverFit) where it looks like the largest weight has been reduced, the second largest weight (which is close in spectral distance) is amplified massively.

We further note that, as we move from the q 1 regime to q < 1, we move from having a symmetric to right skew bulk.

Both have a significant spike near the origin.

This experimentally corroborates the result derived in (Pennington & Bahri, 2017) , where under normality of weights, inputs and further assumptions the symmetric bulk is given by the Wigner semi-circle law and the right skew bulk is given by the Marcenko-Pastur.

The huge spectral peak near the origin, indicates that the result lies in a low effective dimension.

For all Augmented spectra, the negative spectral mass shrinks drastically and we see for the Overfit Schedule for CIFAR-100 that there is no spectral mass below 0 and for CIFAR-10 there is one negative Ritz value (and 79 positive).

This Ritz value λ n = −0.00093 has a weight of ρ = 0.085 compared to the largest spike at λ = 0.0001 with weight 0.889, hence we cannot rule out that both values belong to a true spectral peak at the origin.

The geometric properties of loss landscapes in deep learning have a profound effect on generalization performance.

We introduced the True Hessian to investigate the difference between the landscapes for the true and empirical loss surfaces.

We derived analytic forms for the perturbation between the extremal eigenvalues of the True and Empirical Hessians, modelling the difference between the two as a Gaussian Orthogonal Ensemble.

Moreover, we developed a method for fast eigenvalue computation and visualization, which we used in conjunction with data augmentation to approximate the True Hessian spectrum.

We show both theoretically and empirically that the True Hessian has smaller variation in eigenvalues and that its extremal eigenvalues are smaller in magnitude than the Empirical Hessian.

We also show under our framework that we expect the Empirical Hessian to have a greater negative spectral density than the True Hessian and our experiments support this conclusion.

This result may provide some insight as to why first order (curvature blind) methods perform so well on neural networks.

Reported non-convexity and pathological curvature is far worse for the empirical risk than the true risk, which is what we wish to descend.

The shape of the true risk is particularly crucial for understanding how to develop effective procedures for Bayesian deep learning.

With a Bayesian approach, we not only want to find a single point that optimizes a risk, but rather to integrate over a loss surface to form a Bayesian model average.

The geometric properties of the loss surface, rather than the specific location of optima, therefore greatly influences the predictive distribution in a Bayesian procedure.

Furthermore, the posterior representation for neural network weights with popular approaches such as the Laplace approximation has curvature directly defined by the Hessian.

In future work, one could also replace the GOE noise matrix ε(w) with a positive semi-definite white Wishart kernel in order to derive results for the empirical Gauss-Newton and Fisher information matrices, which are by definition positive semi-definite and are commonly employed in second order deep learning (Martens & Grosse, 2015) .

Our approach to efficient eigenvalue computation and visualization can be used as a general-purpose tool to empirically investigate spectral properties of large matrices in deep learning, such as the Fisher information matrix.

Following the notation of (Bun et al., 2017 ) the resolvent of a matrix H is defined as

with z = x + iη ∈ C. The normalised trace operator of the resolvent, in the N → ∞ limit

is known as the Stieltjes transform of ρ.

The functional inverse of the Siteltjes transform, is denoted the blue function B(S(z)) = z. The R transform is defined as

crucially for our calculations, it is known that the R transform of the Wigner ensemble is

Consider an n × n symettric matrix M n , whose entries are given by

The Matrix M n is known as a real symmetric Wigner matrix.

Theorem 2.

Let {M n } ∞ n=1 be a sequence of Wigner matrices, and for each n denote X n = M n / √ n. Then µ Xn , converges weakly, almost surely to the semi circle distribution,

the property of freeness for non commutative random matrices can be considered analogously to the moment factorisation property of independent random variables.

The normalized trace operator, which is equal to the first moment of the spectral density

We say matrices A&B for which ψ(A) = ψ(B) = 0 4 are free if they satisfy for any integers n 1 ..n k

E DERIVATION

The Stijeles transform of Wigners semi circle law, can be written as (Tao, 2012)

from the definition of the Blue transform, we hence have

Computing the R transform of the rank 1 matrix H true , with largest non-trivial eigenvalue β, on the effect of the spectrum of a matrix A, using the Stieltjes transform we easily find following (Bun et al., 2017) that

We can use perturbation theory similar to in equation equation 22 to find the blue transform which to leading order gives

setting ω = S M (z)

using the ansatz of

we find that S 0 (z) = S (w) (z) and using that B M (z) = 1/g (z) , we conclude that

and hence

and hence in the large N limit the correction only survives if S (w) (z) = 1/β

clearly for β → −β we have

An extensive linear algebraic and geometric discussion about spectral broadening for sample covariance matrices can be found in (Ledoit & Wolf, 2004) and whilst the Hessian is not a covariance matrix (the generalized Gauss-Newton or Fisher matrices can be seen as a co-variance of gradients), identical arguments will hold here 5 .

The dispersion of the Empirical Hessian eigenvalues around their mean will equal the dispersion of the True Hessian eigenvalues around their mean plus the variance of the Empirical Hessian (from its true value), which is in general > 0.

Our theoretical assumptions in sections 4.1 and 4.2 allow us to derive analytic results for this broadening.

Our assumptions are significantly stricter than those in (Ledoit & Wolf, 2004) and hence automatically fulfil their requirements.

In order to empirically analyse properties of modern neural network spectra with tens of millions of parameters N = O(10 7 ), we use the Lanczos algorithm (Meurant & Strakoš, 2006) with Hessian vector products using the Pearlmutter trick (Pearlmutter, 1994) with computational cost O(N T m), where N is the dataset size and m is the number of Lanczos steps.

The main properties of the Lanczos algorithm are summarized in the theorems 3,4 Theorem 3.

Let H N ×N be a symmetric matrix with eigenvalues λ 1 ≥ .. ≥ λ n and corresponding orthonormal eigenvectors z 1 , ..z n .

If θ 1 ≥ .. ≥ θ m are the eigenvalues of the matrix T m obtained after m Lanczos steps and q 1 , ...q k the corresponding Ritz eigenvectors then

where c k is the chebyshev polyomial of order k

Proof: see (Golub & Van Loan, 2012) .

Theorem 4.

The eigenvalues of T k are the nodes t j of the Gauss quadrature rule, the weights w j are the squares of the first elements of the normalized eigenvectors of T k

Proof: See (Golub & Meurant, 1994) .

The first term on the RHS of equation 9 using Theorem 4 can be seen as a discrete approximation to the spectral density matching the first m moments v T H m v (Golub & Meurant, 1994; Golub & Van Loan, 2012) , where v is the initial seed vector.

Using the expectation of quadratic forms, for zero mean, unit variance random vectors, using the linearity of trace and expectation

The error between the expectation over the set of all zero mean, unit variance vectors v and the monte carlo sum used in practice can be bounded (Hutchinson, 1990; Roosta-Khorasani & Ascher, 2015) .

However in the high dimensional regime N → ∞, we expect the squared overlap of each random vector with an eigenvector of H, |v

N ∀i, with high probability.

This result can be seen by computing the moments of the overlap between Rademacher vectors, containing elements P (v j = ±1) = 0.5.

Further analytical results for Gaussian vectors have been obtained (Cai et al., 2013 The number of Parameters in the V GG − 16 network is P = 15291300 so our augmentation procedure is unable to probe the limit q < 1.

The training procedure is identical to the PreResNet except for an initial learning rate of 0.05 and T = 300 epochs.

Here we also see a reduction in both extremal eigenvalues.

@highlight

Understanding the neural network Hessian eigenvalues under the data generating distribution.

@highlight

This paper analyzes the spectrum of the Hessian matrix of large neural networks, with an analysis of max/min eigenvalues and visualization of spectra using a Lanczos quadrature approach.

@highlight

This paper uses the random matrix theory to study the spectrum distribution of the empirical Hessian and true Hessian for deep learning, and proposes an efficient spectrum visualization methods.