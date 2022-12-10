We introduce NAMSG, an adaptive first-order algorithm for training neural networks.

The method is efficient in computation and memory, and is straightforward to implement.

It computes the gradients at configurable remote observation points, in order to expedite the convergence by adjusting the step size for directions with different curvatures in the stochastic setting.

It also scales the updating vector elementwise by a nonincreasing preconditioner to take the advantages of AMSGRAD.

We analyze the convergence properties for both convex and nonconvex problems by modeling the training process as a dynamic system, and provide a strategy to select the observation factor without grid search.

A data-dependent regret bound is proposed to guarantee the convergence in the convex setting.

The method can further achieve a O(log(T)) regret bound for strongly convex functions.

Experiments demonstrate that NAMSG works well in practical problems and compares favorably to popular adaptive methods, such as ADAM, NADAM, and AMSGRAD.

Training deep neural networks (Collobert et al., 2011; Hinton et al., 2012; Amodei et al., 2016; He et al., 2016) with large datasets costs a huge amount of time and computational resources.

Efficient optimization methods are urgently required to accelerate the training process.

First-order optimization methods (Robbins & Monro, 1951; Polyak, 1964; Bottou, 2010; Sutskever et al., 2013; Kingma & Ba, 2015; Bottou et al., 2018) are currently the most popular for training neural networks.

They are easy to implement since only first-order gradients are introduced as input.

Besides, they require low computation overheads except for computing gradients, which is of the same computational complexity as just evaluating the function.

Compared with second-order methods (Nocedal, 1980; Martens, 2010; Byrd et al., 2016) , they are more effective to handle gradient noise.

Sutskever et al. (2013) show that the momentum is crucial to improve the performance of SGD.

Momentum methods, such as HB Polyak (1964) , can amplify steps in low-curvature eigen-directions of the Hessian through accumulation, although careful tuning is required to ensure fine convergence along the high-curvature directions.

Sutskever et al. (2013) also rewrite the Nesterov's Accelerated Gradient (NAG) (Nesterov, 1983) in a momentum form, and show the performance improvement over HB.

The method computes the gradient at a observation point ahead of the current point along the last updating direction.

They illustrate that NAG suppresses the step along high curvature eigen-directions in order to prevent oscillations.

However, all these approaches are approximation of their original forms derived for exact gradients, without fully study on gradient noise.

show the insufficiency of HB and NAG in stochastic optimization, especially for small minibatches.

They further present ASGD and show significant improvements.

However, the method requires tuning of 3 parameters, leading to huge costs that impede its practical applications.

Among variants of SGD methods, adaptive methods that scale the gradient elementwise by some form of averaging of the past gradients are particularly successful.

ADAGRAD (Duchi et al., 2011) is the first popular method in this line.

It is well-suited for sparse gradients since it uses all the past gradients to scale the update.

Nevertheless, it suffers from rapid decay of step sizes, in cases of nonconvex loss functions or dense gradients.

Subsequent adaptive methods, such as RMSPROP (Tieleman & Hinton., 2012) , ADADELTA (Zeiler, 2012) , ADAM (Kingma & Ba, 2015) , and NADAM (Dozat, 2016) , mitigate this problem by using the exponential moving averages of squared past gradients.

However, Reddi et al. (2018) show that ADAM does not converge to optimal solutions in some convex problems, and the analysis extends to RMSPROP, ADADELTA, and NADAM.

They propose AMSGRAD, which fixes the problem and shows improvements in experiments.

In this paper, we propose NAMSG, that is an efficient first-order method for training neural networks.

The name is derived from combining a configurable NAG method (CNAG) and AMSGRAD.

NAMSG computes the stochastic gradients at configurable observation points ahead of the current parameters along the last updating direction.

Nevertheless, instead of approximating NAG for exact gradients, it adjusts the learning rates for eigen-directions with different curvatures to expedite convergence in the stochastic setting, by selecting the observation distance.

It also scales the update vector elementwisely using the nonincreasing preconditioner of AMSGRAD.

We analyze the convergence properties by modeling the training process as a dynamic system, reveal the benefits of remote gradient observations and provide a strategy to select the observation factor without grid search.

A regret bound is introduced in the convex setting, and it is further improved for strongly convex functions.

Finally, we present experiments to demonstrate the efficiency of NAMSG in real problems.

2 THE NAMSG SCHEME Before further description, we introduce the notations following Reddi et al. (2018) , with slight abuse of notation.

The letter t denotes iteration number, d denotes the dimension of vectors and matrices, denotes a predefined positive small value, and S d + denotes the set of all positive definite d × d matrix.

For a vector a ∈ R d and a matrices M ∈ R d × R d , we use a/M to denote M −1 a, diag(a) to denote a square diagonal matrix with the elements of a on the main diagonal, M i to denote the i th row of M , and

, we use √ a for elementwise square root, a 2 for elementwise square, a/b for elementwise division, and max(a, b) to denote elementwise maximum.

For any vector θ i ∈ R d , θ i,j denotes its j th coordinate where j ∈ {1, 2, . . .

, d}. We define F ⊂ R d as the feasible set of points.

Assume that F has bounded diameter D ∞ , i.e. x − y ≤ D ∞ for any x, y ∈ F, and ∇f t (x) ∞ ≤G ∞ , ∇f t (x) 1 ≤G 1 for all x ∈ F. The projection operation is defined as Π F ,A (y) = arg min x∈F A 1/2 (x − y) for A ∈ S d + and y ∈ R d .

In the context of machine learning, we consider the minimization problem of a stochastic function,

where x is a d dimensional vector consisting of the parameters of the model, and ξ is a random datum consisting of an input-output pair.

Since the distribution of ξ is generally unavailable, the optimizing problem (1) is approximated by minimizing the empirical risk on the training set {ζ 1 , ζ 2 , ..., ζ N }, as

In order to save computation and avoid overfitting, it is common to estimate the objective function and its gradient with a minibatch of training data, as

where the minibatch S t ⊂ {1, 2, ..., N }, and b = |S t | is the size of S t .

Firstly, we propose a configurable NAG method (CNAG).

Since the updating directions are partially maintained in momentum methods, gradients computed at observation points, which lie ahead of the current point along the last updating direction, contain the predictive information of the forthcoming update.

The remote observation points are defined asẋ t = x t − η t u t−1 where u t−1 is the updating vector, andẋ 1 = x 1 .

By computing the gradient at a configurable observation pointẋ t , and substituting the gradient with the observation gradient in the HB update, we obtain the original form of CNAG, as

where α t , β t , η t are configurable coefficients, and m 0 = 0.

The observation distance η t can be configured to accommodate gradient noise, instead of η t = β t in NAG (Sutskever et al., 2013) .

Both x t andẋ t are required to update in (4).

To make the method more efficient, we simplify the update by approximation.

Assume that the coefficients α t , β 1t , and η t , change very slowly between adjacent iterations.

Substituting x t byẋ t + η t−1 α t−1 m t−1 , we obtain the concise form of CNAG, as

where the observation factor µ t = η t (1 − β t )/β t , and we use x instead ofẋ for simplicity.

In practical computation of CNAG, we further rearrange the update form as

where only 3 scalar vector multiplications and 3 vector additions are required per iteration besides the gradient computation.

Hereinafter, we still use (5) for simplicity in expressions.

Then, we study the relation of CNAG and ASGD, that guides the selection of the momentum coefficient. shows that ASGD improves on SGD in any information-theoretically admissible regime.

By taking a long step as well as short step and an appropriate average of both of them, ASGD tries to make similar progress on different eigen-directions.

It takes 3 hyper-parameters: short stepα, long step parameterκ, and statistical advantage parameterξ.α is the same as the step size in SGD.

For convex functions,κ is an estimation of the condition number.

The statistical advantage parameterξ ≤ √κ captures trade off between statistical and computational condition numbers, andξ √κ in high stochasticity regimes.

These hyper-parameters vary in large ranges, and are difficult to estimate.

The huge costs in tuning limits the application of ASGD.

The appendix shows that CNAG is a more efficient equivalent form of ASGD.

For CNAG with constant hyper-parameters, the momentum coefficient β t = β = (κ − 0.49ξ)/(κ + 0.7ξ).

Since the condition number is generally large in real high dimension problems, and the statistical advantage parameterξ ≤ √κ , β is close to 1.

To sum up, the equivalence of CNAG and ASGD shows that in order to narrow the gap between the step sizes on eigen-directions with different curvatures, the momentum coefficient β should be close to 1.

Finally, we form NAMSG by equipping CNAG with the nonincreasing preconditioner of AMSGRAD, and project the parameter vector x into the feasible set F. Algorithm 1 shows the pseudo code of NAMSG.

Compared with AMSGRAD, NAMSG requires low computation overheads, as a scalar vector multiplication and a vector addiction per iteration, which are much cheaper than the gradient estimation.

Almost no more memory is needed if the vector operations are run by pipelines.

In most cases, especially when weight decay is applied for regularization, which limits the norm of the parameter vectors, the projection can also be omitted in implementation to save computation.

In Algorithm 1, the observation factor µ t is configurable to accelerate convergence.

However, it is costly to select it by grid search.

In this section we analyze the convergence rate in a local stochastic quadratic optimization setting by investigating the optimizing process as a dynamic system, and reveal the effect of remote gradient observation for both convex and non-convex problems.

Based on the analysis, we provide the default values and a practical strategy to set the observation factor without grid search.

The problem (1) can be approximated locally as a stochastic quadratic optimization problem, as

whereF is a local set of feasible parameter points.

In the problem, the gradient observation is noisy as ∇f t (x) = ∇Φ(x) +ǵ t , whereǵ t is the gradient noise.

4:

5:

8:

Consider the optimization process of NAMSG, and ignore the projections for simplicity.

Sincev t varies slowly when t is large, we can ignore the change ofv t between recent iterations.

The operation of dividing the update by √v t can be approximated by solving a preconditioned problem, as

wherex

, which is supposed to have improved condition number compared withĤ in the convex setting.

Then, we model the optimization process as a dynamic system.

Solving the quadratic problem (7) by NAMSG is equal to solving the preconditioned problem (8) by CNAG, aš

where the preconditioned stochastic functionf t (x) = f t (V −1/4 tx ), the initial momentumm 0 = 0, the coefficients α = (1 − β 1t )α t , β = β 1t , and µ = µ t are considered as constants.

We use ν to denote a unit eigenvector of the Hessian H, and the corresponding eigenvalue is λ.

We define the coefficients asṡ t = ν,x t ,v t = ν,m t .

According to (9), the coefficients are updated as

where the gradient error coefficient δ t = V −1/4 tǵt , ν /λ.

Substitutingv t byṽ t = αv t , and denote τ = αλ, we rewrite the update (10) into a dynamic system as

where A is the gain matrix.

The eigenvalues of A are

where ρ = 1 + β − τ (1 − β(1 − µ)).

Denote the corresponding unit eigenvectors as w 1 and w 2 , that are solved numerically since the expressions are too complicated.

Define the coefficients c 1 , c 2 , d 1 , d 2 satisfying

From (11), (12) and (13), we obtain

Assume that δ t = σδ, where δ obeys the standard normal distribution, and σ is the standard deviation of δ t .

From (14), we obtain E(ṡ t+1 ) = r

According to the analysis in Section 2, we recommend the momentum factor β = 0.999.

Figure  1 presents the gain factor g f ac = max(|r 1 |, |r 2 |) and the stand deviation limit lim t→+∞ Std(ṡ t ) of CNAG.

It is shown that compared with HB (µ = 0), a proper observation factor µ improves the convergence rate significantly, and also accelerates the divergence in nonconvex problems where τ = αλ < 0.

When the step size α is constant, compared with large curvatures, a small curvature λ converges much slower, forming the bottleneck of the whole training process.

The problem can be alleviated by using a large µ. However, the noise level also increases along with µ when α is constant, that prohibits too large µ. Consequently, we recommend µ = 0.1 to achieve fast convergence speed, and µ = 0.2 to improve generalization at the cost of more iterations, since higher noise level is beneficial for expanding the range of exploration.

For NAMSG, experiments shows that when β 2 is close to 1, its variation does not effect the results significantly.

We recommend β 2 = 0.99.

Only the step size α is left for grid search.

Figure 1 also shows that a large β and a proper µ ensures a large convergence domain, while 0 < τ < 2 is required for convergence in SGD (β = 0).

Since the range of eigenvalue λ is problemdependent, a large maximum τ (denoted by τ max ) allows large step sizes.

As shown in Figure 1 (a), µ does not effect g fac significantly for a tiny range of τ close to 0.

Then, g fac decreases almost linearly according to τ to the minimum.

Consequently, training with a large step size α and small µ is beneficial for both the convergence of tiny positive λ, and the divergence of tiny negative λ in nonconvex settings.

While selecting a smaller µ and scaling α proportional to argmin τ g fac , the λ to minimize g fac is unchanged, and the convergence rate for 0 < λ < τ max /α is generally improved according to Figure 1 .

However, the noise level also increases, that prohibits too large α.

We propose a hyper-parameter policy named observation boost (OBSB).

The policy performs grid search for a small portion of iterations using a small µ to select an optimal initial α.

In training, when the loss flattens, it doubles µ, and scales α proportional to argmin τ g fac .

The recommend initial µ is 0.05.

In this section, we provide a data dependent regret bound of NAMSG in the convex setting, and further improve the bound for strongly convex functions.

Since the sequence of cost functions f t (x) are stochastic, we evaluate the convergence property of our algorithm by regret, which is the sum of all the previous difference between the online prediction f t (x t ) and the best fixed point parameter f t (x * ) for all the previous steps, defined as

When the regret of an algorithm satisfies R T = o(T ), the algorithm converges to the optimal parameters on average.

The positive definiteness of Γ t results in a nonincreasing step size and avoids the non-convergence of ADAM.

Following Reddi et al. (2018), we derive the following key results for NAMSG.

Theorem 1.

Let {x t }, {v t } and {v t } be the sequences obtained from Algorithm 1,

, · · · , T }, and x ∈ F.

We have the following bound on the regret

By compared with the regret bound of AMSGRAD (Reddi et al., 2018), we find that the regret bounds of the two methods have the similar form.

However, when β 1 and γ are close to 1, which is the typical situation, NAMSG has lower coefficients on all of the 3 terms.

From Theorem 1, we can immediately obtain the following corollary.

Corollary 1.

Suppose β 1t = β 1 /t, then we have

The bound in Corollary 1 is considerably better than O( Duchi et al., 2011) .

For strongly convex functions, NAMSG further achieves a O(log(T )) regret bound with O(1/t) step size (Bottou et al., 2018; Wang et al., 2019) under certain assumptions.

, where λ is a positive constant.

Let {x t }, {v t } and {v t } be the sequences obtained from Algorithm 1.

The initial step size α ≥ max i∈{1,··· ,d} tv 1/2

, · · · , T }, and x ∈ F.

We have the following bound on the regret

When the gradients are sparse, satisfying

The proof of theorems are given in the appendix.

It should be noted that although the proof requires a decreasing schedule of α t and β 1t to ensure convergence, numerical experiments show that piecewise constant α t and constant β 1t provide fast convergence speed in practice.

In this section, we present experiments to evaluate the performance of NAMSG and the OBSB policy for NAMSG, compared with SGD with momentum (Polyak, 1964) , CNAG, and popular adaptive stochastic optimization methods, such as ADAM (Kingma & Ba, 2015) , NADAM (Dozat, 2016) , and AMSGRAD 1 (Reddi et al., 2018) .

We study logistic regression and neural networks for multiclass classification, representing convex and nonconvex settings, respectively.

The experiments are carried out with MXNET (Chen et al., 2015) .

We compare the performance of SGD, ADAM, NADAM, CNAG, AMSGRAD, NAMSG and OBSB, for training logistic regression and neural network on the MNIST dataset (LeCun et al., 1998) .

The dataset consists of 60k training images and 10k testing images in 10 classes.

The image size is 28 × 28.

Logistic regression:In the experiment, the minibatch size is 256.

The hyper-parameters for all the methods except NAMSG and OBSB are chosen by grid search (see appendix), and the best results in training are reported.

In NAMSG and OBSB, only the step size α is chosen by grid search, and the other hyper-parameters are set according to the default values.

We report the train and test results in Figure 2 , which are the average of 5 runs.

It is observed that OBSB perform the best with respect to train loss, and NAMSG also converges faster than other methods.

The test accuracy is roughly consistent with the train loss in the initial epochs, after which they fluctuate for overfitting.

The experiment shows that NAMSG and OBSB achieves fast convergence in the convex setting.

In the experiment, we train a simple convolutional neural network (CNN) for the multiclass classification problem on MNIST.

The architecture has two 5 × 5 convolutional layers, with 20 and 50 outputs.

Each convolutional layer is followed by Batch Normalization (BN) (Ioffe & Szegedy, 2015) and a 2×2 max pooling.

The network ends with a 500-way fully-connected layer with BN and ReLU (Nair & Hinton, 2010) , a 10-way fully-connected layer, and softmax.

The hyper-parameters are set in a way similar to the previous experiment.

The results are also reported in Figure 2 , which are the average of 5 runs.

We can see that NAMSG has the lowest train loss, which translates to good generalization performance.

OBSB also converges faster than other methods.

The experiment shows that NAMSG and OBSB are efficient in non-convex problems.

In the experiment, we train Resnet-20 (He et al., 2016) on the CIFAR-10 dataset (Krizhevsky, 2009) , that consists of 50k training images and 10k testing images in 10 classes.

The image size is 32 × 32.

The architecture of the network is as follows: In training, the network inputs are 28 × 28 images randomly cropped from the original images or their horizontal flips to save computation.

The inputs are subtracted by the global mean and divided by the standard deviation.

The first layer is 3 × 3 convolutions.

Then we use a stack of 18 layers with 3 × 3 convolutions on the feature maps of sizes {28, 14, 7} respectively, with 6 layers for each feature map size.

The numbers of filters are {16, 32, 64} respectively.

A shortcut connection is added to each pair of 3×3 filters.

The subsampling is performed by convolutions with a stride of 2.

Batch normalization is adopted right after each convolution and before the ReLU activation.

The network ends with a global average pooling, a 10-way fully-connected layer, and softmax.

In testing, the original 32 × 32 images are used as inputs.

We train Resnet-20 on CIFAR-10 using SGD, ADAM, NADAM, CNAG, AMSGRAD, NAMSG, and OBSB.

The training for each network runs for 75 epochs.

The hyper-parameters are selected in a way similar to the previous experiments, excepting that we divide the constant step size by 10 at the 12000 th iteration (in the 62 th epoch).

A weight decay of 0.001 is used for regularization.

Two group of hyper-parameters are obtained for each method, one of which minimizes the train loss before the dropping of step size, and the other maximizes the mean test accuracy of the last 5 epoches.

Figure 3 shows the average results of 5 runs.

In experiments to achieve the fastest training speed (Figure 3 (a),(b) ), OBSB converges the fastest, and NAMSG is also faster than other methods.

Compares with ADAM, OBSB is more than 1 time faster, and NAMSG is roughly 1 time faster to reach the train loss before the dropping of step size.

OBSB has the best test accuracy, and NAMSG is better than other methods.

CNAG achieves significant acceleration upon SGD, and is also faster than ADAM, NADAM, and AMSGRAD.

In experiments to achieve the best generalization (Figure 3 (c) , (d)), OBSB still converges the fastest, NAMSG and CNAG converge at almost the same speed, which is faster than other methods.

The mean best generalization accuracy of SGD, ADAM, NADAM, CNAG, AMSGRAD, NAMSG, and OBSB are 0.9129, 0.9065, 0.9066, 0.9177, 0.9047, 0.9138, and 0.9132, respectively.

CNAG achieves the highest test accuracy.

OBSB, NAMSG, and SGD obtains almost the same final test accuracy, which is much higher than ADAM, NADAM, and AMSGRAD.

It should be noted that CNAG achieves the best test accuracy at the cost of grid search for 3 parameters, while NAMSG and OBSB only search for the step size.

The experiments show that in the machine learning problems tested, NAMSG and OBSB converges faster compared with other popular adaptive methods, such as ADAM, NADAM, and AMSGRAD.

The acceleration is achieved with low computational overheads and almost no more memory.

We present the NAMSG method, which computes the gradients at configurable remote observation points, and scales the update vector elementwise by a nonincreasing preconditioner.

It is efficient in computation and memory, and is straightforward to implement.

A data-dependent regret bound is proposed to guarantee the convergence in the convex setting.

The bound is further improved to O(log(T )) for strongly convex functions.

The analysis of the optimizing process provides a hyperparameter policy (OBSB) which leaves only the step size for grid search.

Numerical experiments demonstrate that NAMSG and OBSB converge faster than ADAM, NADAM, and AMSGRAD, for the tested problems.

A.1 PROOF OF THEOREM 1

In this proof, we use y i to denote the i th coordinate of a vector y.

From Algorithm 1,

we have

Since 0 ≤ β 1t < 1, from the assumptions,

Rearrange the inequity (A2), we obtain

For simplicity, denote

Because of the convexity of the objective function, the regret satisfies

The first inequity follows from the convexity of function f t .

The second inequality is due to (A4).

We now bound the term

We have

In (A7), the second inequity is follows from the definition of v t , the fifth inequality is due to Cauchy-Schwarz inequality.

The final inequality is due to the following bound on harmonic sum:

From (A7), and Lemma A2, which bound

, we further bound the term P 2 as

The third inequity is due to β 1t ≥ β 1t+1 andv 1/2 t,i /α t ≥v 1/2 t−1,i /α t−1 by definition.

We also have

In (A9), the second inequity follows from the assumption β 1t < β 1t−1 , the third and the last inequality is due tov 1/2 t,i /α t ≥v 1/2 t−1,i /α t−1 by definition and the assumption α t = α/ √ t.

Combining (A6), (A8), and (A9), we obtain

The proof is complete.

The Lemmas used in the proof are as follows:

Lemma A2. (Reddi et al., 2018) For the parameter settings and conditions assumed in Theorem 1, which is the same as Theorem 4 in Reddi et al. (2018), we have

The proofs of Lemma A1 and A2 are described in Reddi et al. (2018) .

Because of the objective function is strongly convex, from (A3) and (A4) the regret satisfies

We divide the righthand side of (A11) to three parts, as

Firstly, we bound the term Q 1 .

≤ 0 (A13) The first inequity follows from β t is nonincreasing, the second equity follows from α t = α/t.

The last inequity is because of the assumption α ≥ max i∈{1,··· ,d} tv 1/2

Algorithm A1 ASGD Algorithm Input: initial parameter vector x 1 , short stepα, long step parameterκ ≥ 1, statistical advantage parameterξ ≤ √κ , iteration number T Output: parameter vector x T 1: Setx 1 = x 1 ,β = 1 − 0.7 2ξ /κ.

2: for t = 1 to T − 1 do 3: g t = ∇f t (x t ).

4:x t+1 =βx t + (1 −β) x t −κα 0.7 g t .

5:

Finally, we bound the term Q 3 .

Both the first equity and the first inequity follow from the assumptions α t = α/t and β 1t = β 1 /t 2 .

The last inequity is due tov t,i is nondecreasing by definition.

Combining (A11), (A13), (A17), and (A18), we obtain

A.3 EQUIVALENCE OF CNAG AND ASGD

The pseudo code of ASGD ) is shown in Algorithm A1.

ASGD maintains two iterates: descent iterate x t and a running averagex t .

The running average is a weighted average of the previous average and a long gradient step from the descent iterate, while the descent iterate is updated as a convex combination of short gradient step from the descent iterate and the running average.

We rewrite the update of Algorithm A1 as

Define the variable transform as m t x t =T x t x t ,T = lkl 0 1 ,

wherek arel are adjustable coefficients.

Combining (A20) and (A21), we obtain m t+1 x t+1 =T m t x t +Tbg t ,T =TÄT −1 .

In order to minimize the number of vector computations, we solve the adjustable coefficientsk andl by assigningT 1,2 = 0,T 2,1 = 1.

We choose the solution as

Combining (A22) and (A23), we obtain m t+1 x t+1 =T m t x t +Tbg t ,T = 0.7β (1−β)+0.7 0 1 1 .

The update (A24) is identical to the practical form of CNAG update (6) with constant hyperparameters.

The momentum coefficient of CNAG is β t = β = 0.7β (1 −β) + 0.7 = (κ − 0.49ξ)/(κ + 0.7ξ),

where the second equity follows from the definition ofβ in Algorithm A1.

It should be noted that the computational overheads of ASGD besides the gradient computation is 6 scalar vector multiplications and 4 vector additions per iteration, while CNAG reduces the costs to 3 scalar vector multiplications and 3 vector additions.

We use constant hyper-parameters in the experiments.

For ADAM, NADAM, and AMSGRAD, the hyper-parameters (α, β 1 , β 2 ) are selected from {0.0005, 0.001, 0.002, 0.005, 0.01, 0.02} × {0, 0.9, 0.99, 0.999, 0.9999} × {0.99, 0.999} by grid search.

For SGD, the hyperparameters (α, β) are selected from {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} × {0, 0.9, 0.99, 0.999, 0.9999} by grid search.

For CNAG, the hyper-parameters (α, β, µ) are selected from {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0} × {0, 0.9, 0.99, 0.999, 0.9999} ×{0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.9} by grid search.

For NAMSG and OBSB, the hyperparameters (α) is selected from {0.0005, 0.001, 0.002, 0.005, 0.01, 0.02} by grid search, (β 1 , β 2 , µ) are set according to the default values.

In OBSB, the grid search runs for 5 epochs in the experiments on MNIST, and 20 epochs on CIFAR10.

The average convergence rate is computed each 2 epoches on MNIST, and 10 epochs on CIFAR10.

α and µ are scaled when the converging rate is halved to achieve fast convergence, and at the 50th epoch (when the loss flattens) to maximize generalization.

The experiments are carried out on a workstation with an Intel Xeon E5-2680 v3 CPU and a NVIDIA K40 GPU.

The source code of NAMSG can be downloaded at https://github.com/rationalspark/NAMSG/blob/master/Namsg.py, and the hyper-parameters can be downloaded at https://github.com/rationalspark/NAMSG/ blob/master/hyperparamters.txt.

The simulation environment is MXNET, which can be downloaded at http://mxnet.incubator.apache.org.

The MNIST dataset can be downloaded at http://yann.lecun.com/exdb/mnist; the CIFAR-10 dataset can be downloaded at http://www.cs.toronto.edu/~kriz/cifar.html.

<|TLDR|>

@highlight

A new algorithm for training neural networks that compares favorably to popular adaptive methods.