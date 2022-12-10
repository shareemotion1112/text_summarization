We propose a robust Bayesian deep learning algorithm to infer complex posteriors with latent variables.

Inspired by dropout, a popular tool for regularization and model ensemble, we assign sparse priors to the weights in deep neural networks (DNN) in order to achieve automatic “dropout” and avoid over-fitting.

By alternatively sampling from posterior distribution through stochastic gradient Markov Chain Monte Carlo (SG-MCMC) and optimizing latent variables via stochastic approximation (SA), the trajectory of the target weights is proved to converge to the true posterior distribution conditioned on optimal latent variables.

This ensures a stronger regularization on the over-fitted parameter space and more accurate uncertainty quantification on the decisive variables.

Simulations from large-p-small-n regressions showcase the robustness of this method when applied to models with latent variables.

Additionally, its application on the convolutional neural networks (CNN) leads to state-of-the-art performance on MNIST and Fashion MNIST datasets and improved resistance to adversarial attacks.

Bayesian deep learning, which evolved from Bayesian neural networks (Neal, 1996; BID4 , provides an alternative to point estimation due to its close connection to both Bayesian probability theory and cutting-edge deep learning models.

It has been shown of the merit to quantify uncertainty BID6 , which not only increases the predictive power of DNN, but also further provides a more robust estimation to enhance AI safety.

Particularly, BID5 BID3 described dropout (Srivastava et al., 2014) as a variational Bayesian approximation.

Through enabling dropout in the testing period, the randomly dropped neurons generate some amount of uncertainty with almost no added cost.

However, the dropout Bayesian approximation is variational inference (VI) based thus it is vulnerable to underestimating uncertainty.

MCMC, known for its asymptotically accurate posterior inference, has not been fully investigated in DNN due to its unscalability in dealing with big data and large models.

Stochastic gradient Langevin dynamics (SGLD) (Welling and Teh, 2011) , the first SG-MCMC algorithm, tackled this issue by adding noise to a standard stochastic gradient optimization, smoothing the transition between optimization and sampling.

Considering the pathological curvature that causes the SGLD methods inefficient in DNN models, BID15 proposed combining adaptive preconditioners with SGLD (pSGLD) to adapt to the local geometry and obtained state-of-the-art performance on MNIST dataset.

To avoid SGLD's random-walk behavior, BID3 proposed using stochastic gradient Hamiltonian Monte Carlo (SGHMC), a second-order Langevin dynamics with a large friction term, which was shown to have lower autocorrelation time and faster convergence BID2 .

Saatci and Wilson (2017) used SGHMC with GANs BID8 ) to achieve a fully probabilistic inference and showed the Bayesian GAN model with only 100 labeled images was able to achieve 99.3% testing accuracy in MNIST dataset.

Raginsky et al. (2017) ; Zhang et al. (2017) ; Xu et al. (2018) provided theoretical interpretations of SGLD from the perspective of non-convex optimization, echoing the empirical fact that SGLD works well in practice.

When the number of predictors exceeds the number of observations, applying the spike-and-slab priors is particularly powerful and efficient to avoid over-fitting by assigning less probability mass on

We denote the decaying learning rate at time k by (k) , the entire data by DISPLAYFORM0 , where d i = (x i , y i ), the log of posterior by L(β), ∇ as the gradient of any function in terms of β.

The minibatch of data B is of size n with indices S = {s 1 , s 2 , ..., s n }, where s i ∈ {1, 2, ..., N }.

Stochastic gradient ∇L(β) from a mini-batch of data B randomly sampled from D is used to approximate the true gradient ∇L(β):∇L(β) = ∇ log P(β) + N n i∈S ∇ log P(d i |β).The stochastic gradient Langevin dynamics (no momentum) is formed as follows: DISPLAYFORM1 where τ > 0 denotes the temperature, G is a positive definite matrix to precondition the dynamics.

It has been shown that SGLD asymptotically converges to a stationary distribution π(β|D) ∝ e τ L(β) (Teh et al., 2015; Zhang et al., 2017) .

As τ increases, the algorithm tends towards optimization with underestimated uncertainty.

Another variant of SG-MCMC, SGHMC BID3 Ma et al., 2015) , proposes to use second-order Langevin dynamics to generate samples: DISPLAYFORM2 where r is the momentum item, M is the mass,B is an estimate of the error variance from the stochastic gradient, C is a user-specified friction term to counteracts the noisy gradient.

Dropout has been proven successful, as it alleviates over-fitting and provides an efficient way of making bagging practical for ensembles of countless sub-networks.

Dropout can be interpreted as assigning the Gaussian mixture priors on the neurons BID7 .

To mimic Dropout in our Bayesian CNN models, we assign the spike-and-slab priors on the most fat-tailed weights in FC1 FIG0 .

From the Bayesian perspective, the proposed robust algorithm distinguishes itself from the dropout approach in treating the priors: our algorithm keeps updating the priors during posterior inference, rather than fix it.

The inclusion of scaled mixture priors in deep learning models were also studied in BID1 ; BID15 with encouraging results.

However, to the best of our knowledge, none of the existing SG-MCMC methods could deal with complex posterior with latent variables.

Intuitively, the Bayesian formulation with model averaging and the spike-and-slab priors is expected to obtain better predictive performance through averages from a "selective" group of "good" submodels, rather than averaging exponentially many posterior probabilities roughly.

For the weight priors of the rest layers (dimension u), we just assume they follow the standard Gaussian distribution, while the biases follow improper uniform priors.

Similarly to the hierarchical prior in the EM approach to variable selection (EMVS) (Rořková and George, 2014), we assume the weights β ∈ R p in FC1 follow the spike-and-slab mixture prior DISPLAYFORM0 where DISPLAYFORM1 for each j and 0 < v 0 < v 1 .

By introducing the latent variable γ j = 0 or 1, the mixture prior is represented as DISPLAYFORM2 The interpretation is: if γ j = 0, then β j is close to 0; if γ j = 1, the effect of β j on the model is intuitively large.

The likelihood of this model given a mini-batch of data {( DISPLAYFORM3 where ψ(x i ; β) can be a mapping for logistic regression or linear regression, or a mapping based on a series of nonlinearities and affine transformations in the deep neural network.

In the classification formulation, y i ∈ {1, . . .

, K} is the response value of the i-th example.

In addition, the variance σ 2 follows an inverse gamma prior DISPLAYFORM4 The i.i.d.

Bernoulli prior is used since there is no structural information in the same layer.

DISPLAYFORM5 Finally, our posterior density follows DISPLAYFORM6 The EMVS approach is efficient in identifying potential sparse high posterior probability submodels on high-dimensional regression (Rořková and George, 2014) and classification problem (McDermott et al., 2016) .

These characteristics are helpful for large neural network computation, thus we refer the stochastic version of the EMVS algorithm as Expectation Stochastic-Maximization (ESM).

Due to the existence of latent variables, optimizing π(β, σ 2 , δ|B) directly is difficult.

We instead iteratively optimize the "complete-data" posterior log π(β, σ 2 , δ, γ|B), where the latent indicator γ is treated as "missing data".More precisely, the ESM algorithm is implemented by iteratively increasing the objective function DISPLAYFORM0 where E γ|· denotes the conditional expectation DISPLAYFORM1 ) at the k-th iteration, we first compute the expectation of Q, then alter (β, σ, δ) to optimize it.

For the conjugate spike-slab hierarchical prior formulation, the objective function Q is of the form DISPLAYFORM2 where DISPLAYFORM3 spike-and-slab priors in the "sparse" layer, e.g. FC1 DISPLAYFORM4 Gaussian priors in other layers DISPLAYFORM5 and DISPLAYFORM6

The physical meaning of E γ|· γ j in Q 2 is the probability ρ j , where ρ ∈ R p , of β j having a large effect on the model.

Formally, we have DISPLAYFORM0 where DISPLAYFORM1 Bernoulli prior enables us to use P(γ j = 1|δ DISPLAYFORM2 The other conditional expectation comes from a weighted average κ j , where κ ∈ R p .

DISPLAYFORM3

Since there is no closed-form optimal solution for β here, to optimize Q 1 with respect to β, we use Adam (Kingma and Ba, 2014), a popular algorithm with adaptive learning rates, to train the model.

In order to optimize Q 1 with respect to σ, by denoting DISPLAYFORM0 as V, following the formulation in McDermott et al. (2016) and Rořková and George (2014) we have: DISPLAYFORM1 To optimize Q 2 , a closed-form solution can be derived from Eq.(12) and Eq.(13).

DISPLAYFORM2 Algorithm 1 SGLD-SA with spike and slab priors DISPLAYFORM3 Initialize: DISPLAYFORM4 end for

The EMVS algorithm is designed for linear regression models, although the idea can be extended to nonlinear models.

However, when extending to nonlinear models, such as DNNs, the M-step will not have a closed-form update anymore.

A trivial implementation of the M-step will likely cause a local-trap problem.

To tackle this issue, we replace the E-step and the M-step by SG-MCMC with the prior hyperparameters tuned via stochastic approximation BID0 : DISPLAYFORM0 where DISPLAYFORM1 is the gradient-like function in stochastic approximation (see details in Appendix A.2), g θ (·) is the mapping detailed in Eq.(13), Eq.(14), Eq.(15) and Eq.(16) to derive the optimal θ based on the current β, the step size ω (k) can be set as A(k + Z) −α with α ∈ (0.5, 1].

The interpretation of this algorithm is that we sample β (k+1) fromL(β (k) , θ (k) ) and adaptively optimize θ (k+1) from the mapping g θ (k) .

We expect to obtain an augmented sequence as follows: DISPLAYFORM2 We show the (local) L 2 convergence rate of SGLD-SA below and present the details in Appendix B.Theorem 1 (L 2 convergence rate).

For any α ∈ (0, 1] and any compact subset Ω ∈ R 2p+2 , under assumptions in Appendix B.1, the algorithm satisfies: there exists a constant λ such that DISPLAYFORM3 where t(Ω) = inf{k : DISPLAYFORM4 Corollary 1.

For any α ∈ (0, 1] and any compact subset Ω ∈ R 2p+2 , under assumptions in Appendix B.2, the distribution of β (k) in FORMULA1 converges weakly to the invariant distribution e L(β,θ * ) as → 0.The key to guaranteeing the consistency of the latent variable estimators is from stochastic approximation and the fact of DISPLAYFORM5 The non-convex optimization of SGLD (Raginsky et al., 2017; Xu et al., 2018) , in particular the ability to escape shallow local optima (Zhang et al., 2017) , ensures the robust optimization of the latent variables.

Furthermore, the mappings of Eq.(13), Eq.(14), Eq.(15) and Eq.(16) all satisfy the assumptions on g in a compact subset Ω (Appendix B), which enable us to apply theorem 1 to SGLD-SA.

Because SGHMC proposed by BID3 is essentially a second-order Langevin dynamics and yields a stationary distribution given an accurate estimation of the error variance from the stochastic gradient, the property of SGLD-SA also applies to SGHMC-SA.

Corollary 2.

For any α ∈ (0, 1] and any compact subset Ω ∈ R 2p+2 , under assumptions in Appendix B.2, the distribution of β (k) from SG-MCMC-SA converges weakly to the invariant distribution π(β, ρ * , κ * , σ * , δ * |D) as → 0.

The posterior average given decreasing learning rates can be approximated through the weighted DISPLAYFORM0 (Welling and Teh, 2011) to avoid over-emphasizing the tail end of the sequence and reduce the variance of the estimator.

Teh et al. (2015) ; BID2 showed a theoretical optimal learning rate (k) ∝ k −1/3 for SGLD and k −1/5 for SGHMC to achieve faster convergence for posterior average, which are used in Sec. 4.1 and Sec. 4.2 respectively.

4.1 SIMULATION OF LARGE-P-SMALL-N REGRESSION SGLD-SA can be applied to the (logistic) linear regression cases, as long as u = 0 in Eq.(11).

We conduct the linear regression experiments with a dataset containing n = 100 observations and p = 1000 predictors.

N p (0, Σ) is chosen to simulate the predictor values X (training set) where DISPLAYFORM0 Response values y are generated from Xβ + η, where β = (β 1 , β 2 , β 3 , 0, 0, ..., 0) and η ∼ N n (0, 3I n ).

To make the simulation in Rořková and George (2014) more challenging, we assume DISPLAYFORM1 We introduce some hyperparameters, but most of them are uninformative, e.g. ν ∈ {10 −3 , 1, 10 3 } makes little difference in the test set performance.

Sensitivity analysis shows that three hyperparameters are important: v 0 , a and σ, which are used to identify and regularize the over-fitted space.

We fix τ = 1, λ = 1, ν = 1, v 1 = 100, δ = 0.5, b = p and set a = 1.

The learning rates for SGLD-SA and SGLD are set to (k) = 0.001 × k To implement SGLD-SA, we perform stochastic gradient optimization by randomly selecting 50 observations and calculating the corresponding gradient in each iteration.

We simulate 500, 000 samples from the posterior distribution and at the same time keep optimizing the latent variables.

EMVS is implemented with β directly optimized each time.

We also simulate a group of the test set with 1000 observations (display 50 in FIG5 (e)) in the same way as generating the training set to evaluate the generalizability of our algorithm.

Tab.1 shows that EMVS frequently fails given bad initializations, while SGLD-SA is fairly robust to the hyperparameters.

In addition, from , we can see SGLD-SA is the only algorithm among the three that quantifies the uncertainties of β 1 , β 2 and β 3 reasonably and always gives more accurate posterior average ( FIG5 ); by contrast, the estimated response values y from SGLD is close to the true values in the training set ( FIG5 ), but are far away from them in the testing set (2(e)), indicating the over-fitting problem of SGLD without proper regularization.

For the simulation of SGLD-SA in logistic regression to demonstrate the advantage of SGLD-SA over SGLD and ESM, we leave the results in Appendix C.

We implement all the algorithms in Pytorch (Paszke et al., 2017) and run the experiments on GeForce GTX 1080 GPUs.

The first DNN we use is a standard 2-Conv-2-FC CNN model ( FIG0 ) of 670K parameters (see details in Appendix D.1).

The first set of experiments is to compare methods on the same model without using other complication, such as data augmentation (DA) or batch normalization (BN) BID12 .

We refer to the general CNN without dropout as Vanilla, with 50% dropout rate applied to the green neurons ( FIG0 as Dropout.

Vanilla and Dropout models are trained q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q with Adam (Kingma and Ba, 2014) with Pytorch default parameters (with learning rate 0.001).

We use SGHMC as a benchmark method as it is also sampling-based and has a close relationship with the popular momentum based optimization approaches in DNN.

SGHMC-SA differs from SGHMC in that SGHMC-SA applies the spike-and-slab priors to the FC1 layer while SGHMC just uses the standard normal priors.

The hyperparameters v 1 = 1, v 0 = 1 × 10 −3 and σ = 0.1 in SGHMC-SA are used to regularize the over-fitted space, and a, b are set to p to obtain a moderate "sparsity" to resemble dropout, the step size is ω (k) = 0.1 × (k + 1000) DISPLAYFORM0 .

We use training batch size 1000 and a thinning factor 500 to avoid a cumbersome system, and the posterior average is applied to each Bayesian model.

Temperatures are tuned to achieve better results (Appendix D.2).The four CNN models are tested on the classical MNIST and the newly introduced Fashion MNIST (FMNIST) (Xiao et al., 2017) dataset.

Performance of these models is shown in Tab.2.

Compared with SGHMC, our SGHMC-SA outperforms SGHMC on both datasets.

We notice the posterior averages from SGHMC-SA and SGHMC obtain much better performance than Vanilla and Dropout.

Without using either DA or BN, SGHMC-SA achieves 99.60% which even outperforms some state-of-the-art models, such as stochastic pooling (99.53%) (Zeiler and Fergus, 2013) , Maxout Network (99.55%) BID9 and pSGLD (99.55%) BID15 .

In F-MNIST, SGHMC-SA obtains 93.01% accuracy, outperforming all other competing models.

To further test the maximum performance of SGHMC-SA, we apply DA and BN to the following experiments (see details in Appendix D.3) and refer the datasets with DA as aMNIST and aFMNIST.

All the experiments are conducted using a 2-Conv-BN-3-FC CNN of 490K parameters.

Using this model, we obtain 99.75% on aMNIST (300 epochs) and 94.38% on aFMNIST (1000 epochs).

The results are noticeable because posterior model averaging is essentially conducted on a single Bayesian neural network.

We also conduct the experiments based on the ensemble of five networks and refer them as aMNIST-5 and aFMNIST-5 in Tab.

2.

We achieve 99.79% on aMNIST-5 using 5 small Bayesian neural networks each with 2 thinning samples (4 thinning samples in aFMNIST-5), which is comparable with the state-of-the-art performance (Wan et al., 2013) .

Continuing with the setup in Sec. 4.2, the third set of experiments focus on evaluating model robustness.

We expect less robust models perform considerably well on a certain dataset due to over-tuning; however, as the degree of adversarial attacks increases, the performance decreases sharply.

In contrast, more robust models should be less affected by these adversarial attacks.

We apply the Fast Gradient Sign method BID10 to generate the adversarial examples with one single gradient step as in Papernot et al. (2016)'s study: DISPLAYFORM0 where ζ ranges from 0.1, 0.2, . . .

, 0.5 to control the different levels of adversarial attacks.

Similar to the setup in the adversarial experiments by BID16 , we normalize the adversarial images by clipping to the range [0, 1].

As shown in FIG6 and FIG6 , there is no significant difference among all the four models in the early phase.

As the degree of adversarial attacks arises, the images become vaguer as shown in FIG6 and FIG6 .

In this scenario the performance of Vanilla decreases rapidly, reflecting its poor defense against adversarial attacks, while Dropout performs better than Vanilla.

But Dropout is still significantly worse than the sampling based methods SGHMC-SA and SGHMC.

The advantage of SGHMC-SA over SGHMC becomes more significant when ζ > 0.25.In the case of ζ = 0.5 in MNIST where the images are hardly recognizable, both Vanilla and Dropout models fail to identify the right images and their predictions are as worse as random guesses.

However, SGHMC-SA model achieves roughly 11% higher than these two models and 1% higher than SGHMC, which demonstrates the strong robustness of our proposed SGHMC-SA.

Overall, SGHMC-SA always yields the most robust performance.

We propose a mixed sampling-optimization method called SG-MCMC-SA to efficiently sample from complex DNN posteriors with latent variables and prove its convergence.

By adaptively searching and penalizing the over-fitted parameter space, the proposed method improves the generalizability of deep neural networks.

This method is less affected by the hyperparameters, achieves higher prediction accuracy over the traditional SG-MCMC methods in both simulated examples and real applications and shows more robustness towards adversarial attacks.

Interesting future directions include applying SG-MCMC-SA towards popular large deep learning models such as the residual network BID11 on CIFAR-10 and CIFAR-100, combining active learning and uncertainty quantification to learn from datasets of smaller size and proving posterior consistency and the consistency of variable selection under various shrinkage priors concretely.

The Fokker-Planck equation (FPE) can be formulated from the time evolution of the conditional distribution for a stationary random process.

Denoting the probability density function of the random process at time t by q(t, β), where β is the parameter, the stochastic dynamics is given by DISPLAYFORM0 Let q(β) = lim t→∞ q(t, β).

If lim t→∞ ∂ t q(t, β) = 0, then DISPLAYFORM1 .

In other words, lim t→∞ q(t, β) ∝ e L(β) , i.e. q(t, β) gradually converges to the Bayesian posterior e L(β) .

Robbins-Monro algorithm is the first stochastic approximation algorithm to deal with the root finding problem.

Given the random output of H(θ, β) with respect to β, our goal is to find θ * such that DISPLAYFORM0 where E θ * denotes the expectation with respect to the distribution of β given parameter θ * .

To implement the Robbins-Monro Algorithm, we can generate iterates of the form as follows : DISPLAYFORM1 Note that in this algorithm, H(θ, β) is the unbiased estimator of h(θ), that is DISPLAYFORM2 If there exists an antiderivative Q(θ, β) that satisfies H(θ, β) =

and E θ [Q(θ, β)] is concave, it is equivalent to solving the stochastic optimization problem max θ∈Θ E θ [Q(θ, β)].

In contrast to Eq.(21) and Eq.(22), the general stochastic approximation algorithm is intent on solving the root of the integration equation DISPLAYFORM0 where θ ∈ Θ, β ∈ B, for a subset B ∈ B, the transition kernel Π θ (β (k) , B), which converges to the invariant distribution f θ (β), satisfies that DISPLAYFORM1 .

The stochastic approximation algorithm is an iterative recursive algorithm consisting of two steps: DISPLAYFORM2 is not equal to 0 but decays to 0 as k → ∞.To summarise, H(θ, β) is a biased estimator of h(θ) in finite steps, but as k → ∞, the bias decreases to 0.

In the SG-MCMC-SA algorithm FORMULA1 , DISPLAYFORM3 B CONVERGENCE ANALYSIS

The stochastic gradient Langevin Dynamics with a stochastic approximation adaptation (SGLD-SA) is a mixed half-optimization-half-sampling algorithm to handle complex Bayesian posterior with latent variables, e.g. the conjugate spike-slab hierarchical prior formulation.

Each iteration of the algorithm consists of the following steps:(1) Sample β (k+1) using SGLD based on the current θ (k) , i.e. DISPLAYFORM0 (2) Optimize θ (k+1) from the following recursion DISPLAYFORM1 where g θ (k) (·) is some mapping to derive the optimal θ based on the current β.

DISPLAYFORM2 .

In this formulation, our target is to find θ * such that h(θ DISPLAYFORM3 as k → ∞ , this algorithm falls to the category of the general stochastic approximation.

To provide the local L 2 upper bound for SGLD-SA, we first lay out the following assumptions:Assumption 1 (Step size and Convexity).

{ω (k) } k∈N is a positive decreasing sequence of real numbers such that DISPLAYFORM0 There exist constant δ > 0 and θ DISPLAYFORM1 with additionally DISPLAYFORM2 Then for any α ∈ (0, 1] and suitable A and B, a practical ω (k) can be set as DISPLAYFORM3 Assumption 2 (Existence of Markov transition kernel).

For any θ ∈ Θ, in the mini-batch sampling, there exists a family of noisy Kolmogorov operators {Π θ } approximating the operator (infinitesimal) of the Ito diffusion, such that every Kolmogorov operator Π θ corresponds to a single stationary distribution f θ , and for any Borel subset A of β, we have DISPLAYFORM4 Assumption 3 (Compactness).

For any compact subset Ω of Θ, we only consider θ ∈ Ω such that DISPLAYFORM5 Note that the compactness assumption of the latent variable θ is not essential, the assumption that the variable is in the compact domain is not only reasonable, but also simplifies our proof.

In addition, there exists constants C 1 (Ω) and C 2 (Ω) so that DISPLAYFORM6 Assumption 4 (Solution of Poisson equation).

For all θ ∈ Θ, there exists a function µ θ on β that solves the Poisson equation DISPLAYFORM7 For any compact subset Ω of Θ, there exist constants C 3 (β, Ω) and C 4 (β, Ω) such that for all θ, θ ∈ Ω, DISPLAYFORM8 Remark: For notation simplicity, we write C 1 (Ω) as C 1 , C 2 (Ω) as C 2 , . . .

in the following context.

Lemma 1 is a restatement of Lemma 25 (page 447) from BID0 .

Lemma 1.

Suppose k 0 is an integer which satisfies with DISPLAYFORM9 Then for any k > k 0 , the sequence {Λ K k } k=k0,...,K defined below is increasing and bounded by 2ω DISPLAYFORM10 Lemma 2 is an extension of Lemma 23 (page 245) from BID0 Lemma 2.

There exist λ 0 and k 0 such that for all λ ≥ λ 0 and k ≥ k 0 , the sequence u DISPLAYFORM11 From FORMULA2 , we can denote a positive constant ∆ + as lim k→∞ inf 2δω DISPLAYFORM12 .

Then (36) can be simplified as DISPLAYFORM13 There exist λ 0 and k 0 such that for all λ > λ 0 and k > k 0 , (37) holds.

Note that in practical case when C 1 is small, finding a suitable λ 0 will not be a problem.

Theorem 1 (L 2 convergence rate).

Suppose that Assumptions 1-4 hold, for any compact subset Ω ∈ Θ, the algorithm satisfies: there exists a constant λ such that DISPLAYFORM14 where t(Ω) = inf{k : DISPLAYFORM15 Proof.

Denote T (k) = θ (k) − θ * , with the help of FORMULA2 and Poisson equation FORMULA2 , we deduce that DISPLAYFORM16 First of all, according to FORMULA1 and FORMULA2 , we have DISPLAYFORM17 DISPLAYFORM18 Conduct the decomposition of D3 similar to Theorem 24 (p.g.

246) from BID0 and Lemma A.5 (Liang, 2010) .

DISPLAYFORM19 (ii) From FORMULA3 and FORMULA3 respectively, we deduce that DISPLAYFORM20 Thus there exists C 4 = 2C 4 C 0 such that DISPLAYFORM21 (iii) D3-3 can be further decomposed to D3-3a and D3-3b DISPLAYFORM22 ) with a constant C 3 = 4C 0 C 3 which satisfies that DISPLAYFORM23 Finally, add all the items D1, D2 and D3 together, for some C 1 = C 3 + C 4 , we have DISPLAYFORM24 Moreover, from FORMULA3 and FORMULA3 , there exists a constant C 5 such that DISPLAYFORM25 Lemma 3 is an extension of Lemma 26 (page 248) from BID0 .Lemma 3.

Let {u (k) } k≥k0 as a sequence of real numbers such that for all k ≥ k 0 , some suitable constants C 1 and C 2 DISPLAYFORM26 and assume there exists such k 0 that DISPLAYFORM27 Then for all k > k 0 , we have DISPLAYFORM28 Proof of Theorem 1 (Continued).

From Lemma 2, we can choose λ 0 and k 0 which satisfy the conditions (39) and (40) DISPLAYFORM29 From Lemma 3, it follows that for all k > k 0 DISPLAYFORM30 From (38) and the increasing property of Λ k j in Lemma 1, we have DISPLAYFORM31 Therefore, given the sequence u (k) = λ 0 ω (k) that satisfies conditions FORMULA3 , FORMULA5 and Lemma 3, for any k > k 0 , from FORMULA1 and FORMULA2 , we have DISPLAYFORM32 where λ = λ 0 + 6C 5 .

In addition to the previous assumptions, we make one more assumption on the stochastic gradients to guarantee that the samples converge to the posterior conditioned on the optimal latent variables: Assumption 5 (Gradient Unbiasedness and Smoothness).

For all β ∈ B and θ ∈ Θ, the mini-batch of data B, the stochastic noise ξ, which comes from∇L(β, θ) − ∇L(β, θ), is a white noise and independent with each other.

In addition, there exists a constant l ≥ 2 such that the following conditions hold: DISPLAYFORM0 For all θ, θ ∈ Θ, there exists a constant M > 0 such that the gradient is M-smooth: DISPLAYFORM1 Corollary 1.

For all α ∈ (0, 1], under assumptions 1-5, the distribution of β (k) converges weakly to the invariant distribution e L(β,θ * ) as → 0.Proof.

The proof framework follows from section 4 of Sato and Nakagawa (2014).

In the context of stochastic noise ξ (k) , we ignore the subscript of and only consider the case of τ = 1.

Since θ DISPLAYFORM2 converges to θ * in SGLD-SA and the gradient is M-smooth FORMULA5 , we transform the stochastic gradient from ∇L( DISPLAYFORM3 , therefore Eq. FORMULA2 can be written as DISPLAYFORM4 Using Eq. FORMULA3 , the characteristic function of DISPLAYFORM5 Then the characteristic function of DISPLAYFORM6 Rewrite β (k+1) as β (k+ ) , the characteristic function of β (t+ ) is the characteristic function of DISPLAYFORM7 With the fact exp(x) = 1 + x + O(x 2 ), we can get DISPLAYFORM8 Therefore, For any integrable function f , set F as the Fourier transform defined by DISPLAYFORM9 DISPLAYFORM10 The inverse Fourier transform of F[f (x)] and the l-th order derivatives of f (l) (x) is DISPLAYFORM11 Combine Eq. FORMULA5 , Eq. FORMULA6 and Eq. FORMULA1 , we arrive at the following simplified equation: DISPLAYFORM12 Since DISPLAYFORM13 Finally, we have proved that the distribution of β (k) converges weakly to the invariant distribution e L(β,θ * ) as → 0.

Now we conduct the experiments on binary logistic regression.

The setup is similar as before, except n is set to 500, Σ i,j = 0.3 |i−j| and η ∼ N (0, I/2).

We set the learning rates in SGLD-SA and SGLD to 0.01 × k FIG11 , FIG11 and FIG11 demonstrate the posterior distribution of SGLD-SA is significantly better than that of SGLD.

As shown in FIG11 , SGLD-SA is the best method to regulate the over-fitting space and provides the most reasonable posterior mean.

Table.

3 illustrates the predictive power of SGLD-SA is overall better than the other methods and robust to different initializations.

FIG11 and FIG11 show that the over-fitting problem of SGLD when p > n in logistic regression and the algorithm fails to regulate the over-fitting space; We observe SGLD-SA is able to resist over-fitting and always yields reproducible results.

q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q q The first DNN we use is a standard 2-Conv-2-FC CNN: it has two convolutional layers with a 2 × 2 max pooling after each layer and two fully-connected layers.

The filter size in the convolutional layers is 5 × 5 and the feature maps are set to be 32 and 64, respectively BID13 ).

The fully-connected layers (FC) have 200 hidden nodes and 10 outputs.

We use the rectified linear unit (ReLU) as activation function between layers and employ a cross-entropy loss.

The second DNN is a 2-Conv-BN-3-FC CNN: it has two convolutional layers with a 2 × 2 max pooling after each layer and three fully-connected layers with batch normalization applied to the first FC layer.

The filter size in the convolutional layers is 4 × 4 and the feature maps are both set to 64.

We use 256 × 64 × 10 fully-connected layers.

In practice, we observe a suitable temperature setting is helpful to improve the classification accuracy.

For example, by setting τ = 100 in the second DNN (see Appendix D.1) we obtain 99.70% on aMNIST.

To account for the scale difference of weights in different layers, we apply different temperatures to different layers based on different standard deviations of the gradients in each layer and obtain the results in Tab.

2.

The MNIST dataset is augmented by (1) randomCrop: randomly crop each image with size 28 and padding 4, (2) random rotation: randomly rotate each image by a degree in [−15• , +15• ], (3) normalization: normalize each image with empirical mean 0.1307 and standard deviation 0.3081.The FMNIST dataset is augmented by (1) randomCrop: same as MNIST, (2) randomHorizontalFlip: randomly flip each image horizontally, (3) normalization: same as MNIST.

<|TLDR|>

@highlight

a robust Bayesian deep learning algorithm to infer complex posteriors with latent variables