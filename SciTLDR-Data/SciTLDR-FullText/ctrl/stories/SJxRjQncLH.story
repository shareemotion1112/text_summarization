Neural networks have reached outstanding performance for solving various ill-posed inverse problems in imaging.

However, drawbacks of end-to-end learning approaches in comparison to classical variational methods are the requirement of expensive retraining for even slightly different problem statements and the lack of provable error bounds during inference.

Recent works tackled the first problem by using networks trained for Gaussian image denoising as generic plug-and-play regularizers in energy minimization algorithms.

Even though this obtains state-of-the-art results on many tasks, heavy restrictions on the network architecture have to be made if provable convergence of the underlying fixed point iteration is a requirement.

More recent work has proposed to train networks to output descent directions with respect to a given energy function with a provable guarantee of convergence to a minimizer of that energy.

However, each problem and energy requires the training of a separate network.

In this paper we consider the combination of both approaches by projecting the outputs of a plug-and-play denoising network onto the cone of descent directions to a given energy.

This way, a single pre-trained network can be used for a wide variety of reconstruction tasks.

Our results show improvements compared to classical energy minimization methods while still having provable convergence guarantees.

In many image processing tasks an observed image f is modeled as the result of the transformation of a clean imageû under a known (linear) operator A and unknown noise ξ, f = Aû + ξ.

(

In most cases, the problem of reconstructingû from f and A is ill-posed and can thus not be solved by a simple inversion of A, giving rise to the field of regularization theory with iterative or variational methods, see e.g. [2] for an overview.

In recent years neural networks were very successful in learning a direct mapping G(f ) ≈û for a variety of problems such as deblurring [32, 28] , denoising [34] , super-resolution [8] , demosaicing [9] and MRI-or CT-reconstruction [33, 14] .

Even though this works well in practice, there are rarely any guarantees on the behaviour of neural networks on unseen data, making them difficult to use in safety-critical applications.

Moreover, for each problem and type of noise a separate network has to be trained.

In contrast, classical variational methods try to find the solution by the minimization of a suitable energy function of the formû

where H f is a data fidelity term, for example commonly chosen as H f (u) = 1 2 ||Au − f || 2 , and R is a regularization function that models prior knowledge about the solution, e.g. the popular total variation (TV) regularization, R(u) = ∇u 1 , [24] .

While minimizers of (2) come with many desirable theoretical guarantees, regularizations like the TV often cannot perfectly capture the complex structure of the space of natural images.

To combine the advantages of powerful feed-forward networks and model-based approaches like (2), authors have considered various hybrid models like learning regularizers (e.g. [23, 1, 11, 5] ), designing networks architectures that resemble the structure of minimization algorithms or differential equations, e.g. [25, 36, 15, 6] , interleaving networks with classical optimization steps [16, 17] , or using the parametrization of networks as a regularization for (2), see e.g. [29, 12] .

A particularly flexible approach arises from [7, 37, 30, 13] , where proximal operators with respect to the regularizer are replaced by arbitrary denoising operators, with recent works focusing on the use of denoising networks [18, 4, 35] .

While such approaches allow to tackle different inverse problems with the same neural network, the derivation of theoretical guarantees -even in terms of the convergence of the resulting algorithmic scheme -remains difficult, see [3, 27] or some discussion in [20] , unless the denoiser satisfies particular properties [22] .

The starting point of the above-mentioned algorithmic schemes that utilize denoising networks to regularize model-based inverse problems are methods for the minimization of (2).

While most works focus on primal-dual / ADMM approaches, their convergence analysis is quite delicate even in a setting in which one still minimizes (nonconvex) energies, such that we turn to two simpler methods, gradient descent and proximal gradient methods,

where

Following the idea of [7, 37, 30, 13] , considering either a gradient descent or a proximal step on the regularization as a generic denoising operation gives rise to the following two algorithmic schemes,

where G denotes any kind of denoiser, e.g. a convolutional neural network, and we define ρ(u

for the sake of brevity of notation.

We refer to [20] for a more detailed derivation.

Algorithmic schemes like (5) or (6) combine the model-based flexibility of energy minimization methods (i.e. explicit modelling of H f ) with the expressive power of deep neural networks G.

Unfortunately -despite their success in various practical applications -schemes like (5) or (6) remain dangerous to be used: Figure 1 shows the result of running the iteration (6) with H f = 0 on a noisy input image f = u 0 for 100 and 800 iterations using a DnCNN [34] preimplemented in Matlab as the denoiser G. As we can see the image gets completely distorted.

Even more strikingly, the range of the image increased from values in [0, 1] to an interval of [−185, 218] within the first 1000 iterations.

Clearly, the algorithmic scheme diverges.

A natural condition for the provable convergence of a scheme like (6) (at least along subsequences) would be a 1-Lipschitz continuous operator G. There has been previous work on computing upper bounds for the best Lipschitz constant of a network and using it to enforce a user defined Lipschitz constant L during training time [10, 26, 19] but we found that enforcing non-expansiveness drastically decreased the denoising performance.

The problem of computing the best Lipschitz constant, in hope of improving those results, was recently proved to be NP-hard [31] and thus is infeasible.

Therefore, we adapt the recent idea proposed in [21] to safeguard neural networks by forcing them to predict a descent direction to a given model-based energy, such that it can be used within a line search algorithm to guarantee convergence.

More precisely, at any given estimate u and model-based energy E the authors use the Euclidean projection onto the half space

as the last layer of their network.

Even though the resulting algorithm converges to the minimizer of E, experiments showed significantly higher peaks of the PSNR value in early iterations compared to classical gradient descent on E. Intuitively, the descent direction proposed by the network pushes the iteration closer towards the distribution of the training data than a usual gradient descent step.

While the approach of [21] has to train a separate network for each inverse problem and each type of noise, we investigate the combination of the flexible algorithmic schemes (5) and (6) with the idea from [21] to project onto the half-space of descent directions to safeguard the underlying algorithm.

In the following G will always refer to a generic denoising network, like DnCNN [34] .

We assume that E(u) = H f (u) + R(u) is a continuously differentiable, strictly convex and coercive energy function.

As a first step, we simply rewrite the algorithmic schemes (5) and (6) in such a way that they resemble a gradient descent iteration, i.e.,

such that we can interpret

as "update directions" of the respective algorithmic schemes.

Because the plain iterations (8) and (9) can easily be divergent, we safeguard them by projecting them onto the half-space of descent directions C(γ, ∇E(u k )), i.e.,

Note that we replaced the averaging of the gradient descent and denoising step in (8) by an abitrary convex combination using a parameter α to determine the respective influence of the data term and the denoising more flexibly.

After computing the above directions d k , we update our iterates using

with a step size t k chosen based on a backtracking line-search mechanism similar to [21] .

Under weak additional conditions, the latter guarantees the convergence of the proposed scheme to the minimizer of E. Such a minimizer could of course be determined by any classical algorithm, but we hope for (10) to yield a better path towards the true minimizer, and consider a discrepancy principle for stopping the iteration before convergence.

More precisely, we terminate (10) as soon as

for H f (û) being an estimate on the (data-term-dependent measure of the) noise level of the considered problem, and β being a scaling factor (typically close to 1).

We tested our implementation with the image reconstruction tasks of Gaussian deblurring with standard deviation 1 and 4× single image super resolution.

In both cases we added Gaussian noise with standard deviation 0.02 to the corrupted image.

We chose the PyTorch implementation 1 of DnCNN [34] pre-trained on a noise level of 0.1 as our denoising network.

Our surrogate energy uses a TV regularization with Huber-norm instead of the 1 -norm.

As our data term we choose H f (u) = operator.

The best hyperparameters for all methods were found with a grid search.

In all experiments, for scheme (conv), α = 0 was the best choice for any τ , indicating that the gradient descent step on the data term does not yield much additional information, assumably because the projection onto C(γ, ∇E(u)) which depends on the gradient of the data term anyway.

When using (prox), we empirically found τ = 30 to be the best choice.

For the projection onto the half-space of descent directions, we we used γ = 5 for both methods for deblurring, and γ = 50 in (conv) and γ = 1 in (prox) for super resolution.

For fairness, the classical gradient descent was also implemented using backtracking line search.

Figure 2 shows the reconstruction quality of the current iterate compared to ground truth over a span of 500 iterations.

The PSNR quickly peaks before slowly converging to the fixed point of the surrogate energy which is consistent with the results of [21] .

Notably, the convex combination method peaks earlier but not as high as the prox method.

Tables 1 and 2 show results using early stopping using a discrepancy principle.

On all test images our prox scheme beats gradient descent.

Table 1 : PSNR values for deblurring for varying images and stopping criteria.

The algorithm was stopped when H f (u k ) < βH f (û).

best refers to the highest PSNR over 500 iterations and a "*" means that the stopping criterion was not triggered such that the last iteration was used instead.

Table 2 : PSNR values for super resolution for varying images and stopping criteria.

The algorithm was stopped when H f (u k ) < βH f (û).

best refers to the highest PSNR over 500 iterations.

We combine deep learning and energy minimization methods for solving inverse problems in image reconstruction into a provably convergent algorithmic scheme.

Still, our approach is able to generalize to different problems with a single denoising network and without the need to retrain if that problem changes.

We were able to reach better results than the energy minimization baseline in our experiments, and are happy to elaborate on the above aspects in the NeurIPS workshop.

<|TLDR|>

@highlight

We use neural networks trained for image denoising as plug-and-play priors in energy minimization algorithms for image reconstruction problems with provable convergence.