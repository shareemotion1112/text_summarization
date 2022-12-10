Generative Adversarial Networks (GANs) are a very powerful framework for generative modeling.

However, they are often hard to train, and learning of GANs often becomes unstable.

Wasserstein GAN (WGAN) is a promising framework to deal with the instability problem as it has a good convergence property.

One drawback of the WGAN is that it evaluates the Wasserstein distance in the dual domain, which requires some approximation, so that it may fail to optimize the true Wasserstein distance.

In this paper, we propose evaluating the exact empirical optimal transport cost efficiently in the primal domain and performing gradient descent with respect to its derivative to train the generator network.

Experiments on the MNIST dataset show that our method is significantly stable to converge, and achieves the lowest Wasserstein distance among the WGAN variants at the cost of some sharpness of generated images.

Experiments on the 8-Gaussian toy dataset show that better gradients for the generator are obtained in our method.

In addition, the proposed method enables more flexible generative modeling than WGAN.

Generative Adversarial Networks (GANs) BID2 are a powerful framework of generative modeling which is formulated as a minimax game between two networks: A generator network generates fake-data from some noise source and a discriminator network discriminates between fake-data and real-data.

GANs can generate much more realistic images than other generative models like variational autoencoder BID10 or autoregressive models BID14 , and have been widely used in high-resolution image generation BID8 , image inpainting BID18 , image-to-image translation BID7 , to mention a few.

However, GANs are often hard to train, and various ways to stabilize training have been proposed by many recent works.

Nonetheless, consistently stable training of GANs remains an open problem.

GANs employ the Jensen-Shannon (JS) divergence to measure the distance between the distributions of real-data and fake-data BID2 . provided an analysis of various distances and divergence measures between two probability distributions in view of their use as loss functions of GANs, and proposed Wasserstein GAN (WGAN) which has better theoretical properties than the original GANs.

WGAN requires that the discriminator (called the critic in ) must lie within the space of 1-Lipschitz functions to evaluate the Wasserstein distance via the Kantorovich-Rubinstein dual formulation.

further proposed implementing the critic with a deep neural network and applied weight clipping in order to ensure that the critic satisfies the Lipschitz condition.

However, weight clipping limits the critic's function space and can cause gradients in the critic to explode or vanish if the clipping parameters are not carefully chosen BID3 .

WGAN-GP BID3 and Spectral Normalization (SN) BID12 apply regularization and normalization, respectively, on the critic trying to make the critic 1-Lipschitz, but they fail to optimize the true Wasserstein distance.

In the latest work, BID11 proposed a new WGAN variant to evaluate the exact empirical Wasserstein distance.

They evaluate the empirical Wasserstein distance between the empirical distributions of real-data and fake-data in the discrete case of the Kantorovich-Rubinstein dual for-mulation, which can be solved efficiently because the dual problem becomes a finite-dimensional linear-programming problem.

The generator network is trained using the critic network learnt to approximate the solution of the dual problem.

However, the problem of approximation error by the critic network remains.

In this paper, we propose a new generative model without the critic, which learns by directly evaluating gradient of the exact empirical optimal transport cost in the primal domain.

The proposed method corresponds to stochastic gradient descent of the optimal transport cost. argued that JS divergences are potentially not continuous with respect to the generator's parameters, leading to GANs training difficulty.

They proposed instead using the Wasserstein-1 distance W 1 (q, p), which is defined as the minimum cost of transporting mass in order to transform the distribution q into the distribution p.

Under mild assumptions, W 1 (q, p) is continuous everywhere and differentiable almost everywhere.

The WGAN objective function is constructed using the Kantorovich-Rubinstein duality (Villani, 2009, Chapter 5) as DISPLAYFORM0 to obtain min DISPLAYFORM1 where D is the set of all 1-Lipschitz functions, where P r is the real-data distribution, and where P g is the generator distribution implicitly defined by y = G(z), z ∼ p(z).

Minimization of this objective function with respect to G with optimal D is equivalent to minimizing W 1 (P r , P g ).

et al. (2017) further proposed implementing the critic D in terms of a deep neural network with weight clipping.

Weight clipping keeps the weight parameter of the network lying in a compact space, thereby ensuring the desired Lipschitz condition.

For a fixed network architecture, however, weight clipping may significantly limit the function space to a quite small fraction of all possible 1-Lipschitz functions representable by networks with the prescribed architecture.

BID3 proposed introduction of gradient penalty (GP) to the WGAN objective function in place of the 1-Lipschitz condition in the Kantorovich-Rubinstein dual formulation, in order to explicitly encourage the critic to have gradients with magnitude equal to 1.

Since enforcing the constraint of unit-norm gradient everywhere is intractable, they proposed enforcing the constraint only along straight line segments, each connecting a real-data point and a fake-data point.

The resulting learning scheme, which is called the WGAN-GP, was shown to perform well experimentally.

It was pointed out, however BID12 , that WGAN-GP is susceptible to destabilization due to gradual changes of the support of the generator distribution as learning progresses.

Furthermore, the critic can easily violate the Lipschitz condition in practice, so that there is no guarantee that WGAN-GP optimizes the true Wasserstein distance.

SN, proposed by BID12 , is based on the observation that the Lipschitz norm of a critic represented by a multilayer neural network is bounded from above by the product, across all layers, of the Lipschitz norms of the activation functions and the spectral norms of the weight matrices, and normalizes each of the weight matrices with its spectral norm to ensure the resulting critic to satisfy the desired Lipschitz condition.

It is well known that, for any m × n matrix W = (w ij ), the max norm W max = max{|w ij |} and the spectral norm σ(W ) satisfy the inequality DISPLAYFORM0

The proposed method in this paper is based on the fact that the optimal transport cost between two probability distributions can be evaluated efficiently when the distributions are uniform over finite sets of the same cardinality.

Our proposal is to evaluate empirical optimal transport costs on the basis of equal-size sample datasets of real-and fake-data points.

The optimal transport cost between the real-data distribution P r and the generator distribution P g is defined as DISPLAYFORM0 where c(x, y) is the cost of transporting one unit mass from x to y, assumed differentiable with respect to its arguments almost everywhere, and where Π(P r , P g ) denotes the set of all couplings between P r and P g , that is, all joint probability distributions that have marginals P r and P g .Let D = {x j |x j ∼ P r (x)} be a dataset consisting of independent and identically-distributed (iid) real-data points, and F = {y i |y i ∼ P g (y)} be a dataset consisting of iid fake-data points sampled from the generator.

Let P D and P F be the empirical distributions defined by the datasets D and F , respectively.

We further assume in the following that |D| = |F | = N holds.

The empirical optimal transport costĈ(D, F ) = C(P D , P F ) between the two datasets D and F is formulated as a linear-programming problem, aŝ DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 It is known BID15 that the linear-programming problem (4)- FORMULA6 admits solutions which are permutation matrices.

One can then replace the constraints M i,j ≥ 0 in FORMULA6 with M i,j ∈ {0, 1} without affecting the optimality.

The resulting optimization problem is what is called the linear sum assignment problem, which can be solved more efficiently than the original linear-programming problem.

As far as the authors' knowledge, the most efficient algorithm to date for solving a linear sum assignment problem has time complexity of O(N 2.5 log(N C)), where C = max i,j c(x j , y i ) when one scales up the costs {c(x j , y i )|x j ∈ D, y i ∈ F } to integers (Burkard et al., 2012, Chapter 4) .

This is a problem to find the optimal transport plan, where M i,j = 1 is corresponding to transporting fake-data point y i ∈ F to real-data point x j ∈ D, and where the objective is to minimize the average FIG0 shows a two-dimensional example of this problem and its solution.

(4) - FORMULA6 with N = 8.

Circles • represent real-data points in D and triangles represent fake-data points in F .

Arrows between circles • and filled triangles show the optimal transport plan M * with c(x, y) = x − y 2 , which is an identity matrix.

Arrows between open △ and filled triangles show small perturbations of F , which do not change M * .

DISPLAYFORM4 One requires evaluations not only of the optimal transport cost C(P r , P g ) but also of its derivative in order to perform learning of the generator with backpropagation.

Let θ denote the parameter of the generator, and let ∂ θ C denote the derivative of the optimal transport cost C with respect to θ.

Conditional on z, the generator output G(z) is a function of θ.

Hence, in order to estimate ∂ θ C, in our framework one has to evaluate ∂ θĈ .

In general, it is difficult to differentiate (4) with respect to generator output y i , as the optimal transport plan M * can be highly dependent on y i .

Under the assumption |D| = |F | = N which we adopt here, however, the feasible set for M is the set of all permutation matrices and is a finite set.

It then follows that, as a generic property, the optimal transport plan M * is unchanged under small enough perturbations of F (see FIG0 .

We take advantage of this fact and regard M * as independent of y i .

Now that differentiation of (4) becomes tractable, we use (4) as the loss function of the generator and update the generator with the direct gradient of the empirical optimal transport cost, as DISPLAYFORM5 Although the framework described so far is applicable to any optimal transport cost, several desirable properties can be stated if one specializes in the Wasserstein distance.

Assume, for a given p ≥ 1, that the real-data distribution P r and the generator distribution P g have finite moments of order p.

The Wasserstein-p distance between P r and P g is defined in terms of the optimal transport cost with c(x, y) = x − y p as DISPLAYFORM6 Due to the law of large numbers, the empirical distributions P D and P F converge weakly to P r and P g , respectively, as N → ∞. It is also known (Villani, 2009, Theorem 6.9 ) that the Wasserstein-p distance W p metrizes the space of probability measures with finite moments of order p. Consequently, the empirical Wasserstein distanceŴ p (D, F ) is a consistent estimator of the true Wasserstein distance W p (P r , P g ).

Furthermore, with the upper bound of the error of the estimator DISPLAYFORM7 which is derived on the basis of the triangle inequality, as well as with the upper bounds available for expectations of W p (P D , P r ) and W p (P F , P g ) under mild conditions BID17 , one can see thatŴ p (D, F ) is an asymptotically unbiased estimator of W p (P r , P g ).Note that our method can directly evaluate the empirical Wasserstein distance without recourse to the Kantorovich-Rubinstein dual.

Hence, our method does not use a critic and is therefore no longer a GAN.

It is also applicable to any optimal transport cost.

We summarize the proposed method in Algorithm 1.

We first show experimental results on the MNIST dataset of handwritten digits.

In this experiment, we resized the images to resolution 64 × 64 so that we can use the convolutional neural networks Sample {x i } i∈{1,...,N } ∼ X real from real-data.

Sample {z j } j∈{1,...,N } ∼ p(z) from random noises.

Let y j = G θ (z j ), ∀j ∈ {1, . . .

, N }.

Solve (4)-7 to obtain M * .9: DISPLAYFORM0 θ ← Adam(g θ , θ, α, β 1 , β 2 ) 11: end while described in Appendix A.1 as the critic and the generator.

In all methods, the batch size was set to 64 and the prior noise distribution was the 100-dimensional standard normal distribution.

The maximum number of iterations in training of the generator was set to 30,000.

The Wasserstein-1 distance with c(x, y) = x − y 1 was used.

More detailed settings are described in Appendix B.1.Although several performance metrics have been proposed and are commonly used to evaluate variants of WGAN, we have decided to use the empirical Wasserstein distance (EWD) to compare performance of all methods.

It is because all the methods adopt objective functions that are based on the Wasserstein distance, and because EWD is a consistent and asymptotically unbiased estimator of the Wasserstein distance and can efficiently be evaluated, as discussed in Section 4.

TAB1 shows EWD evaluated with 256 samples and computation time per generator update for each method.

For reference, performance comparison with the Fréchet Inception Distance BID4 and the Inception Score BID13 , which are commonly used as performance measures to evaluate GANs using feature space embedding with an inception model, is shown in Appendix C. The proposed method achieved a remarkably small EWD and computational cost compared with the variants of WGAN.

Our method required the lowest computational cost in this experimental setting mainly because it does not use the critic.

Although we think that the batch size used in the experiment of the proposed method was appropriate since the proposed method achieved lower EWD, if a larger batch size would be required in training, it will take much longer time to solve the linear sum assignment problem (4)-(7).We further investigated behaviors of the methods compared in more detail, on the basis of EWD.

WGAN-SN failed to learn.

The loss function of the critic showed divergent movement toward −∞, and the behaviors of EWD in different trials were different even though the behaviors of the critic loss were the same (Figure 2 (a) and (b)).

WGAN training never failed in 5 trials, and EWD improved stably without sudden deterioration.

Although training with WGAN-GP proceeded favorably in initial stages, at certain points the gradient penalty term started to increase, causing EWD to deteriorate (Figure 2 (c) ).

This happened in all 5 trials.

Since gradient penalty is a weaker restriction than weight clipping, the critic may be more likely to cause extreme behaviors.

We examined both WGAN-TS with and without weight scaling.

Whereas WGAN-TS with weight scaling did not fail in training but achieved higher EWD than WGAN, WGAN-TS without weight scaling achieved lower EWD than WGAN at the cost of the stability of training (Figure 3 ).

The proposed method was trained stably and never failed in 5 trials.

As mentioned in Section 3, the critic in WGAN-TS simply regresses the optimizer of the empirical version of the Kantorovich-Rubinstein dual.

Thus, there is no guarantee that the critic will satisfy the 1-Lipschitz condition.

BID11 pointed out that it is indeed practically problematic with WGAN-TS, and proposed weight scaling to ensure that the critic satisfies the desired condition.

We have empirically found, however, that weight scaling exhibited the following trade-off (Figure 3) .

Without weight scaling, training of WGAN-TS suddenly deteriorated in some trials because the critic came to not satisfy the Lipschitz condition.

With weight scaling, on the other hand, the regression error of the critic with respect to the solution increased and the EWD became worse.

The proposed method directly solves the empirical version of the optimal transport problem in the primal domain, so that it is free from such trade-off.

Figure 4 shows fake-data images generated by the generators trained with WGAN, WGAN-GP, WGAN-TS, and the proposed method.

Although one can identify the digits for the generated images with the proposed method most easily, these images are less sharp.

Among the generated images with the other methods, one can notice several images which have almost the same appearance as real-data images, whereas in the proposed method, such fake-data images are not seen and images that seem averaged real-data images belonging to the same class often appear.

This might imply that merely minimizing the Wasserstein distance between the real-data distribution and the generator distribution in the raw-image space may not necessarily produce realistic images.

We next observed how the generator distribution is updated in order to compare the proposed method with variants of WGAN in terms of the gradients provided.

Figure 5 shows typical behavior of the generator distribution trained with the proposed method on the 8-Gaussian toy dataset.

The 8-Gaussian toy dataset and experimental settings are described in Appendix B.2.

One can observe that, as training progresses, the generator distribution comes closer to the real-data distribution.

FIG4 shows comparison of the behaviors of the proposed method, WGAN-GP, and WGAN-TS.

We excluded WGAN and WGAN-SN from this comparison: WGAN tended to yield generator distributions that concentrated around a single Gaussian component, and hence training did not progress well.

WGAN-SN could not correctly evaluate the Wasserstein distance as in the experiment on the MNIST dataset.

One can observe in FIG4 that directions of sample updates are diverse in the proposed method, especially in later stages of training, and that the sample update directions tend to be aligned with the optimal gradient directions.

These behaviors will be helpful for the generator to learn the realdata distribution efficiently.

In WGAN-GP and WGAN-TS, on the other hand, the sample update directions exhibit less diversity and less alignment with the optimal gradient directions, which would make the generator distribution difficult to spread and would slow training.

One would be able to ascribe such behaviors to poor quality of the critic: Those behaviors would arise when the generator learns on the basis of unreliable gradient information provided by the critic without learning sufficiently to accurately evaluate the Wasserstein distance.

If one would increase the number n c of critic iterations per generator iteration in order to train the critic better, the total computational cost of training would increase.

In fact, n c = 5 is recommended in practice and has commonly been used in WGAN and its variants because the improvement in learning of the critic is thought to be small relative to increase in computational cost.

In reality, however, 5 iterations would not be sufficient for the critic to learn, and this might be a principal reason for the critic to provide poor gradient information to the generator in the variants of WGAN.

We have proposed a new generative model that learns by directly minimizing exact empirical Wasserstein distance between the real-data distribution and the generator distribution.

Since the proposed method does not suffer from the constraints on the transport cost and the 1-Lipschitz constraint imposed on WGAN by solving the optimal transport problem in the primal domain instead of the dual domain, one can construct more flexible generative modeling.

The proposed method provides the generator with better gradient information to minimize the Wasserstein distance (Section 5.2) and achieved smaller empirical Wasserstein distance with lower computational cost (Section 5.1) than any other compared variants of WGAN.

In the future work, we would like to investigate the behavior of the proposed method when transport cost is defined in the feature space embedded by an appropriate inception model.

A NETWORK ARCHITECTURES A.1 CONVOLUTIONAL NEURAL NETWORKS We show in TAB2 the network architecture used in the experiment on the MNIST dataset in Section 5.1.

The generator network receives a 100-dimensional noise vector generated from the standard normal distribution as an input.

The noise vector is passed through the fully-connected layer and reshaped to 4 × 4 feature maps.

Then they are passed through four transposed convolution layers with 5 × 5 kernels, stride 2 and no biases (since performance was empirically almost the same with or without biases, we took the simpler option of not considering biases), where the resolution of feature maps is doubled and the number of them is halved except for the last layer.

The critic network is basically the reverse of the generator network.

A convolution layer is used instead of a transposed convolution layer in the critic.

After the last convolution layer, the feature maps are flattened into a vector and passed through the fully-connected layer.

We employed batch normalization BID6 in all intermediate layers in both of the generator and the critic.

Rectified linear unit (ReLU) was used as the activation function in all but the last layers.

As the activation function in the last layer, the hyperbolic tangent function and the identity function were used for the generator and for the critic, respectively.

We show in Table 3 the network architecture used in the experiment on the 8-Gaussian toy dataset in Section 5.2.

The generator network architecture receives a 100-dimensional noise vector as in the experiment on the MNIST dataset.

The noise vector is passed through the four fully-connected layers with biases and mapped to a two-dimensional space.

The critic network is likewise the reverse of the generator network.

The MNIST dataset of handwritten digits used in the experiment in Section 5.1 contains 60,000 two-dimensional images of handwritten digits with resolution 28 × 28.We used default parameter settings decided by the proposers of the respective methods.

We used RMSProp BID5 with learning rate 5e−5 for the critic and the generator in WGAN.

The weight clipping parameter c was set to 0.01.

We used Adam BID9 with learning rate 1e−4, β 1 = 0.5, β 2 = 0.999 in the other methods.

λ gp in WGAN-GP was set to 10.

In the methods with the critic, the number n c of critic iterations per generator iteration was set to 5.

The 8-Gaussian toy dataset used in the experiment in Section 5.2 is a two-dimensional synthetic dataset, which contains real-data sampled from the Gaussian mixture distribution with 8 centers equally distant from the origin and unit variance as the real-data distribution.

The centers of the 8 Gaussian component distributions are (±10, 0), (0, ±10), and (±10/ √ 2, ±10/ √ 2).

30, 000 samples were generated in advance before training and were used as the real-data samples.

In all methods, the batch size was set to 64 and the maximum number of iterations in training the generator was set to 1,000.

WGAN and WGAN-SN could not learn well with this dataset, even though we considered several parameter sets.

We used Adam with learning rate 1e−3, β 1 = 0.5, β 2 = 0.999 for WGAN-GP, WGAN-TS and the proposed method.

λ gp in WGAN-GP was set to 10.

In the methods with the critic, the number n c of critic iterations was set to 5.

All the numerical experiments in this paper were executed on a computer with an Intel Core i7-6850K CPU (3.60 GHz, 6 cores) and 32 GB RAM, and with four GeForce GTX 1080 graphics cards installed.

Linear sum assignment problems were solved using the Hungarian algorithm, which has time complexity of O(N 3 ).

Codes used in the experiments were written in tensorflow 1.10.1 on python 3.6.0, with eager execution enabled.

We show the result of evaluation of the experimented methods with FID and IS in TAB3 .

Both FID and IS are commonly used to evaluate GANs.

FID calculates the distance between the set of real-data points and that of fake-data points.

The smaller the distance is, the better the fake-data points are judged.

Assuming that the vector obtained from a fake-or real-data point through the inception model follows a multivariate Gaussian distribution, FID is defined by the following equation: DISPLAYFORM0 where (µ i , Σ i ) is the mean vector and the covariance matrix for dataset i, evaluated in the feature space embedded with inception scores.

It is nothing but the square of the Wasserstein-2 distance between two multivariate Gaussian distributions with parameters (µ 1 , Σ 1 ) and (µ 2 , Σ 2 ), respectively.

IS is a metric to evaluate only the set of fake-data points.

Let x i be a data point, y be the label of x i in the data identification task for which the inception model was trained, p(y|x i ) be the probability of label y obtained by inputting x i to the inception model.

Letting X be the set of all data points used for calculating the score, the marginal probability of label y is p(y) = 1 |X| xi∈X p(y|x i ).

IS is defined by the following equation: DISPLAYFORM1 where KL is Kullback-Leibler divergence.

IS is designed to be high as the data points are easy to identify by the inception model and variation of labels identified from the data points is abundant.

In WGAN-GP, WGAN-SN and WGAN-TS*, we observed that training suddenly deteriorated in some trials.

We thus used early stopping on the basis of EWD, and the results of these methods shown in TAB3 are with early stopping.

The proposed method marked the worst in FID and the best in IS among all the methods compared.

Certainly, the fake-data generated by the proposed method are non-sharp and do not resemble realdata points, but it seems that it is easy to distinguish them and they have diversity as digit images.

If one wishes to produce higher FID results using the proposed method, transport cost should be considered in the desired space corresponding to FID.

@highlight

We have proposed a flexible generative model that learns stably by directly minimizing exact empirical Wasserstein distance.