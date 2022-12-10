Momentum is a simple and widely used trick which allows gradient-based optimizers to pick up speed along low curvature directions.

Its performance depends crucially on a damping coefficient.

Largecamping  coefficients can potentially deliver much larger speedups, but are prone to oscillations and instability; hence one typically resorts to small values such as 0.5 or 0.9.

We propose Aggregated Momentum (AggMo), a variant of momentum which combines multiple velocity vectors with different damping coefficients.

AggMo is trivial to implement, but significantly dampens oscillations, enabling it to remain stable even for aggressive damping coefficients such as 0.999.

We reinterpret Nesterov's accelerated gradient descent as a special case of AggMo and analyze rates of convergence for quadratic objectives.

Empirically, we find that AggMo is a suitable drop-in replacement for other momentum methods, and frequently delivers faster convergence with little to no tuning.

In spite of a wide range of modern optimization research, gradient descent with momentum and its variants remain the tool of choice in machine learning.

Momentum methods can help the optimizer pick up speed along low curvature directions without becoming unstable in high-curvature directions.

The simplest of these methods, classical momentum BID24 , has an associated damping coefficient, 0 ≤ β < 1, which controls how quickly the momentum vector decays.

The choice of β imposes a tradoff between speed and stability: in directions where the gradient is small but consistent, the terminal velocity is proportional to 1/(1 − β), suggesting that β slightly less than 1 could deliver much improved optimization performance.

However, large β values are prone to oscillations and instability BID22 BID3 , requiring a smaller learning rate and hence slower convergence.

Finding a way to dampen the oscillations while preserving the high terminal velocity of large beta values could dramatically speed up optimization.

BID29 found that Nesterov accelerated gradient descent BID20 , which they reinterpreted as a momentum method, was more stable than classical momentum for large β values and gave substantial speedups for training neural networks.

However, the reasons for the improved performance remain somewhat mysterious.

O' Donoghue & Candes (2015) proposed to detect oscillations and eliminate them by resetting the velocity vector to zero.

But in practice it is difficult to determine an appropriate restart condition.

In this work, we introduce Aggregated Momentum (AggMo), a variant of classical momentum which maintains several velocity vectors with different β parameters.

AggMo averages the velocity vectors when updating the parameters.

We find that this combines the advantages of both small and large β values: the large values allow significant buildup of velocity along low curvature directions, while the small values dampen the oscillations, hence stabilizing the algorithm.

AggMo is trivial to implement and incurs almost no computational overhead.

We draw inspiration from the physics literature when we refer to our method as a form of passive damping.

Resonance occurs when a system is driven at specific frequencies but may be prevented through careful design BID4 .

Passive damping can address this in structures by making use of different materials with unique resonant frequencies.

This prevents any single frequency from producing catastrophic resonance.

By combining several momentum velocities together we achieve a similar effect -no single frequency is driving the system and so oscillation is prevented.

In this paper we analyze rates of convergence on quadratic functions.

We also provide theoretical convergence analysis showing that AggMo achieves converging average regret in online convex programming BID37 .

To evaluate AggMo empirically we compare against other commonly used optimizers on a range of deep learning architectures: deep autoencoders, convolutional networks, and long-term short-term memory (LSTM).In all of these cases, we find that AggMo works as a drop-in replacement for classical momentum, in the sense that it works at least as well for a given β parameter.

But due to its stability at higher β values, it often delivers substantially faster convergence than both classical and Nesterov momentum when its maximum β value is tuned.2 Background: momentum-based optimization Classical momentum We consider a function f : R d → R to be minimized with respect to some variable θ.

Classical momentum (CM) minimizes this function by taking some initial point θ 0 and running the following iterative scheme, v t = βv t−1 − ∇ θ f (θ t−1 ), DISPLAYFORM0 where γ t denotes a learning rate schedule, β is the damping coefficient and we set v 0 = 0.

Momentum can speed up convergence but it is often difficult to choose the right damping coefficient, β.

Even with momentum, progress in a low curvature direction may be very slow.

If the damping coefficient is increased to overcome this then high curvature directions may cause instability and oscillations.

Nesterov momentum Nesterov's Accelerated Gradient BID20 BID21 ) is a modified version of the gradient descent algorithm with improved convergence and stability.

It can be written as a momentum-based method BID29 , DISPLAYFORM1 Nesterov momentum seeks to solve stability issues by correcting the error made after moving in the direction of the velocity, v.

In fact, it can be shown that for a quadratic function Nesterov momentum adapts to the curvature by effectively rescaling the damping coefficients by the eigenvalues of the quadratic BID29 .

We begin by studying convergence on quadratic functions, which have been an important test case for analyzing convergence behavior BID29 BID22 BID3 , and which can be considered a proxy for optimization behavior near a local minimum BID22 .We analyze the behavior of these optimizers along the eigenvectors of a quadratic function in FIG0 .

In the legend, λ denotes the corresponding eigenvalue.

In (a) we use a low damping coefficient (β = 0.9) while (b) shows a high damping coefficient (β = 0.999).

When using a low damping coefficient it takes many iterations to find the optimal solution.

On the other hand, increasing the damping coefficient from 0.9 to 0.999 causes oscillations which prevent convergence.

When using CM in practice we seek the critical damping coefficient which allows us to rapidly approach the optimum without becoming unstable BID3 .

On the other hand, Nesterov momentum with β = 0.999 is able to converge more quickly within high curvature regions than CM but retains oscillations for the quadratics exhibiting lower curvature.

Aggregated Momentum We propose Aggregated Momentum (AggMo), a variant of gradient descent which aims to improve stability while providing the convergence benefits of larger damping coefficients.

We modify the gradient descent algorithm by including several velocity vectors each with their own damping coefficient.

At each optimization step these velocities are updated and then averaged to produce the final velocity used to update the parameters.

This updated iterative procedure can be written as follows, The arrows show the direction and relative amplitude of the velocities at various points in time.

We discuss points (1) and (2) in Section 3. DISPLAYFORM0 where v (i) 0 = 0 for each i. We refer to the vector β = [β (1) , . . .

, β (K) ] as the damping vector.

By taking advantage of several damping coefficients, AggMo is able to optimize well over illconditioned curvature.

FIG0 shows the optimization along the eigenvectors of a quadratic function using AggMo.

AggMo dampens oscillations quickly for all eigenvalues and converges faster than CM and Nesterov in this case.

In FIG1 we display the AggMo velocities during optimization.

At point (1) the velocities are aligned towards the minima, with the β = 0.999 velocity contributing substantially more to each update.

By point (2) the system has begun to oscillate.

While the β = 0.999 velocity is still pointed away from the minima, the β = 0.9 velocity has changed direction and is damping the system.

Combining the velocities allows AggMo to achieve fast convergence while reducing the impact of oscillations caused by large β values.

Choosing the damping vector Recall that in a direction with small but steady gradient, the terminal velocity is proportional to 1/(1 − β).

We found that a good choice of damping vectors was therefore to space the terminal velocities exponentially.

To do so, we specify an exponential scale-factor, a, and a count K. The damping vector is then constructed as β (i) = 1 − a i−1 , for i = 1 . . .

K. We fix a = 0.1 throughout and vary only K. A good default choice is K = 3 which corresponds to β = [0, 0.9, 0.99].

We found this setting to be both stable and effective in all of our experiments.

Computational/Memory overhead There is very little additional computational overhead when using AggMo compared to CM, as it only requires a handful of extra addition and multipliciation operations on top of the single gradient evaluation.

There is some memory overhead due to storing the K velocity vectors, which are each the same size as the parameter vector.

However, for most modern deep learning applications, the memory cost at training time is dominated by the activations rather than the parameters BID5 BID1 BID31 BID7 , so the overhead will generally be small.

In this section we show that we can recover Nesterov Momentum (Equation 2) using a simple generalization of Aggregated Momentum (Equation 3).

We now introduce separate learning rates for each velocity, γ (i) , so that the iterate update step from Equation 3 is replaced with, DISPLAYFORM0 with each velocity updated as in Equation 3.

To recover Nesterov momentum we consider the special case of β = [0, β] and γ DISPLAYFORM1 t = 2βγ.

The AggMo update rule can now be written as, DISPLAYFORM2 Similarly, we may write the Nesterov momentum update with constant learning rate γ t = γ as, DISPLAYFORM3 Now we consider Equation 6 when using the reparameterization given by φ t = θ t + γβv t , DISPLAYFORM4 It follows that the update to φ from Nesterov is identical to the AggMo update to θ, and we have φ 0 = θ 0 .

We can think of the φ reparameterization as taking a half-step forward in the Nesterov optimization allowing us to directly compare the iterates at each time step.

We note also that if γ DISPLAYFORM5 (2) t = 2γ then the equivalence holds approximately when β is sufficiently close to 1.

We demonstrate this equivalence empirically in Appendix B.This formulation allows us to reinterpret Nesterov momentum as a weighted average of a gradient update and a momentum update.

Moreover, by showing that AggMo recovers Nesterov momentum we gain access to the same theoretical convergence results that Nesterov momentum achieves.5 Convergence analysis

We can learn a great deal about optimizers by carefully reasoning about their convergence on quadratic functions.

BID22 point out that in practice we do not know the condition number of the function to be optimized and so we aim to design algorithms which work well over a large possible range.

Sharing this motivation, we consider the convergence behaviour of momentum optimizers on quadratic functions with fixed hyperparameters over a range of condition numbers.

To compute the convergence rate, ||θ t −θ * || 2 , we model each optimizer as a linear dynamical systems as in BID13 .

The convergence rate is then determined by the eigenvalues of this system.

We leave details of this computation to appendix B. FIG2 displays the convergence rate of each optimizer for quadratics with condition numbers (κ) from 10 1 to 10 7 .

The blue dashed line displays the optimal convergence rate achievable by CM with knowledge of the condition number -an unrealistic scenario in practice.

The two curves corresponding to CM (red and purple) each meet the optimal convergence rate when the condition number is such that β is critical.

On the left of this critical point, where the convergence rates for CM are flat, the system is "under-damped" meaning there are complex eigenvalues corresponding to oscillations.

We observe that the convergence rate of AggMo interpolates smoothly between the convergence rates of CM with β = 0.9 and β = 0.99 as the condition number varies.

AggMo's ability to quickly kill oscillations leads to an approximately three-times faster convergence rate than Nesterov momentum in the under-damped regime without sacrificing performance on larger condition numbers.

We evaluate the convergence rate of AggMo in the setting of online convex programming, as proposed in BID37 .

This is an increasingly common setting to analyze optimization algorithms tailored to machine learning BID2 BID10 BID26 .

Notably, this is equivalent to analyzing the convergence rate in the setting of stochastic convex optimization.

We consider a sequence of unknown convex cost functions, f 1 (θ), . . .

, f T (θ).

At each time t, our goal is to predict the parameter θ t which minimizes the regret, DISPLAYFORM0 where θ * is the fixed point parameter minimizing T t=1 f t (θ * ).

We are able to show that AggMo has regret bounded by O( √ T ) -a result asymptotically comparable to the best known bound BID2 .

We adopt the following definitions from BID2 to simplify the notation.

We write g t = ∇f t (θ t ) with g t,i as the i th element of this vector.

Additionally, we write g 1:t,i ∈ R t as the vector containing the i th element of the gradient over the first t iterations; g 1:t,i = [g 1,i , . . .

, g t,i ].

Then the following theorem holds, Theorem 1.

Assume that f t has bounded gradients, DISPLAYFORM1 .

Then AggMo achieves the following regret bound, for all T ≥ 1.

DISPLAYFORM2 It immediately follows that the average regret of AggMo converges, i.e. that R(T )/T → 0, by observing that ||g 1: DISPLAYFORM3 The full proof is given in Appendix C alongside some open questions on the convergence of AggMo.

While the statement of Theorem 1 requires strict assumptions we note that this result is certainly non-trivial.

BID26 showed that the average regret of Adam BID10 is not guaranteed to converge under the same assumptions.

The convergence of momentum methods has been studied extensively, both theoretically and empirically BID32 BID33 BID9 .

By analyzing the failure modes of existing methods these works motivate successful momentum schemes.

BID29 explored the effect of momentum on the optimization of neural networks and introduced the momentum view of Nesterov's accelerated gradient.

They focused on producing good momentum schedules during optimization to adapt to ill-conditioned curvature.

Despite strong evidence that this approach works well, practitioners today still typically opt for a fixed momentum schedule and vary the learning rate instead.

In Appendix C.1 we show that AggMo evolves as a (K+1)-th order finite difference equation, enabling AggMo to utilize greater expressiveness over the gradient history.

BID14 also introduce dependence on a larger gradient history by adding lagged momentum terms.

However, in doing so the authors introduce many new hyperparameters to be tuned.

Adaptive gradient methods have been introduced to deal with the ill-conditioned curvature that we often observe in deep learning BID2 BID10 BID36 BID30 .

These methods typically approximate the local curvature of the objective to adapt to the geometry of the data.

Natural gradient descent (Amari, 1998) preconditions by the Fisher information matrix, which can be shown to approximate the Hessian under certain assumptions BID17 .

Several methods have been proposed to reduce the computational and memory cost of this approach BID18 BID16 ) but these are difficult to implement and introduce additional hyperparameters and computational overhead compared to SGD.Another line of adaptive methods seeks to detect when oscillations occur during optimization.

O' Donoghue & Candes (2015) proposed using an adaptive restarting scheme to remove oscillations whenever they are detected.

In its simplest form, this is achieved by setting the momentum velocity to zero whenever the loss increases.

Further work has suggested using an adaptive momentum schedule instead of zeroing BID27 .

Although this technique works well for well-conditioned convex problems it is difficult to find an appropriate restart condition for stochastic optimization where we do not have an accurate computation of the loss.

On the other hand, AggMo's passive damping approach addresses the oscillation problem without the need to detect its occurrence.

We evaluated the AggMo optimizer on the following deep learning architectures; deep autoencoders, convolutional networks, and LSTMs.

To do so we used four datasets: MNIST (LeCun et al., 1998), CIFAR-10, CIFAR-100 BID11 BID11 Penn Treebank (Marcus et al., 1993) .

In each experiment we compared AggMo to classical momentum, Nesterov momentum, and Adam.

These optimizers are by far the most commonly used and even today remain very difficult to outperform in a wide range of tasks.

For each method, we performed a grid search over the learning rate and the damping coefficient.

For AggMo, we keep the scale a = 0.1 fixed and vary K as discussed in Section 3.1.

Full details of the experimental set up for each task can be found in Appendix D with additional results given in Appendix E.For each of the following experiments we choose to report the validation and test performance of the network in addition to the final training loss when it is meaningful to do so.

We include these generalization results because recent work has shown that the choice of optimizer may have a significant effect on the generalization error of the network in practice BID35

We trained fully-connected autoencoders on the MNIST dataset using a set-up similar to that of BID29 .

While their work focused on finding an optimal momentum schedule we instead kept the momentum fixed and applied a simple learning rate decay schedule.

For CM and Nesterov we evaluated damping coefficients in the range: {0.0, 0.9, 0.99, 0.999}. For Adam, it is standard to use β 1 = 0.9 and β 2 = 0.999.

Since β 1 is analogous to the momentum damping parameter, we considered β 1 ∈ {0.9, 0.99, 0.999} and kept β 2 = 0.999.

For AggMo, we explored K in { 2,3,4 }.

Each model was trained for 1000 epochs.

We report the training, validation, and test errors in Table 1 .

Results are displayed for the hyperparameters that achieved the best training loss and also for those that achieved the best validation loss.

While Adam is able to perform well on the training objective it is unable to match the performance of AggMo or Nesterov on the validation/test sets.

AggMo achieves the best performance in all cases.

In these experiments the optimal damping coefficient for both CM and Nesterov was β = 0.99 while the optimal damping vector for AggMo was β = [0.0, 0.9, 0.99, 0.999], given by K = 4.

In FIG3 we compare the convergence of each of the optimizers under the optimal hyperparameters for the training loss.

Increasing damping coefficients During our experiments we observed that AggMo remains stable during optimization for learning rates an order of magnitude (or more) larger than is possible for CM and Nesterov with β equal to the max damping coefficient used in AggMo.

We further investigated the effect of increasing the maximum damping coefficient of AggMo in FIG4 .

The learning rate is fixed at 0.1 and we vary K from 2 to 5.

We compared to Nesterov with damping coefficients in the same range (max of 0.9999) and a fixed learning rate of 0.05 (to be consistent with our analysis in Section 4).

We do not include the curves for which training is unstable:

Nesterov with β ∈ {0.999, 0.9999} and AggMo with K = 5.

AggMo is able to take advantage of the larger damping coefficient of 0.999 and achieves the fastest overall convergence.

For the following experiments we evaluated AggMo using two network architectures: a neural network with 5 convolutional layers (CNN-5) and the ResNet-32 architecture BID6 .

We use data augmentation and regularization only for the latter.

Each model was trained for 400 epochs.

Table 2 : Classification accuracy on CIFAR-10 and CIFAR-100 We display results using the optimal hyperparameters for CM, Nesterov, Adam and AggMo on the validation set and also with default settings for CM, Nesterov and AggMo.

For each optimizer we report the accuracy on a randomly held out validation set and the test set.

All of the models achieve near-perfect accuracy on the training set and so we do not report this.

The results are displayed in Table 2 .

On the small convolutional network without regularization, AggMo significantly out performed the other methods.

For both of the ResNet-32 experiments we observed the best validation accuracy with CM.

This is perhaps expected as the model architecture and hyperparameters were likely to have been tuned using CM.

Despite this, we observed that AggMo performed consistently well and had the fastest overall convergence.

We found that our proposed default hyperparameters for AggMo (a = 0.1, K = 3) led to much faster convergence than CM and Nesterov with β = 0.9, a common default choice.

FIG5 shows the training loss and validation accuracy during training for each optimizer used to train the ResNet-32 model.

The hyperparameters used for each plot are those which obtained the best validation accuracy.

AggMo converged most quickly on the training objective without sacrificing final validation performance.

Surprisingly, we found that using AggMo we were also able to train the ResNet-32 architecture on CIFAR-100 without using batch normalization.

With a limited search over learning rates we achieved 69.32% test error compared to a best value of 67.26% using CM.

We also found that, with batch normalization removed, optimization with AggMo remained stable at larger learning rates than with CM.We note that the additional network hyperparameters (e.g. weight decay) are defaults which were likely picked as they work well with classical momentum.

This may disadvantage the other optimizers, including our own.

Despite this, we found that we are able to outperform CM with the AggMo and Nesterov optimizers without additional tuning of any of these hyperparameters.

We trained LSTM Language Models on the Penn Treebank dataset.

We followed the experimental setup of BID19 and made use of the code provided by the authors.

We used the optimal hyperparameter settings described by the authors and vary only the learning rate, momentum and whether gradient clipping is used.

The network hyperparameters were tuned using SGD and may not be optimal for the other optimizers we evaluate (including our own).

We followed only the base model training used in BID19 and do not include the fine-tuning and continuous cache optimization steps.

Each model was trained for 750 epochs.

As noted in BID19 , it is typically observed that SGD without momentum performs better than momentum-based methods in language modeling tasks.

However, in our experiments we Table 3 : Penn Treebank LSTM Perplexity across different optimizers.

We display the train, validation, and test error for the optimization run that produced the best validation loss.

* uses ASGD BID25 and corresponds to the base model reported in BID19 observed all momentum-based optimizers but CM outperform SGD without momentum.

Surprisingly, we found that Adam is well-suited to this task and achieves the best training, validation, and test performance.

We believe that the heavy regularization used when training the network makes Adam a good choice.

AggMo is very close in terms of final performance to Adam.

Table 3 contains the results for the hyperparameter settings which achieved the best validation error for each optimizer.

The first row (denoted *) uses the scheme suggested in BID19 : once the validation loss plateaus we switch to the ASGD (Polyak & Juditsky, 1992) optimizer.

The other rows instead decay the learning rate when the validation loss plateaus.

Figure 7 compares the convergence of the training and validation perplexity of each optimizer.

While the momentum methods converge after 300 epochs, the momentum-free methods converged much more slowly.

Surprisingly, we found that SGD worked best without any learning rate decay.

Adam converged most quickly and achieved a validation perplexity which is comparable to that of AggMo.

While gradient clipping is critical for SGD without momentum, which utilizes a large learning rate, we found that all of the momentum methods perform better without gradient clipping.

In short, while existing work encourages practitioners to avoid classical momentum we found that using other momentum methods may significantly improve convergence rates and final performance.

AggMo worked especially well on this task over a large range of damping coefficients and learning rates.

Aggregated Momentum is a simple extension to classical momentum which is easy to implement and has negligible computational overhead on modern deep learning tasks.

We showed empirically that AggMo is able to remain stable even with large damping coefficients and enjoys faster convergence rates as a consequence of this.

Nesterov momentum can be viewed as a special case of AggMo.(Incidentally, we found that despite its lack of adoption by deep learning practitioners, Nesterov momentum also showed substantial advantages compared to classical momentum.)

On the tasks we explored, AggMo could be used as a drop-in replacement for existing optimizers with little-to-no additional hyperparameter tuning.

But due to its stability at higher β values, it often delivered substantially faster convergence than both classical and Nesterov momentum.

In this section we demonstrate this equivalence on two toy problems.

In each of the figures included here we take β = 0.999.We first consider a 2D quadratic function, f (x) = x T Ax, where A has eigenvalues 1.0 and 0.001.The learning rates for each optimizer are set as described in Section 4.

Each optimizer is initialized at the same position.

FIG7 shows both optimizers following the same optimization trajectories.

In this setting, the two paths are also visually indistinguishable with γ We now optimize the Rosenbrock function, given by, DISPLAYFORM0 This function has a global minimum at (x, y) = 1.

Once again the optimizers are initialized at the same point but for this example we take γ DISPLAYFORM1 for AggMo.

FIG8 shows the optimization trajectories of both algorithms.

In this case we see that the updates are initially indistinguishable but begin to differ as the algorithms approach the origin.

In this section we present details of the convergence rate computations in FIG2 .

We also present some additional supporting results.

We first note that for quadratic functions of the form f (x) = 1 2 x T Ax + b T x we can write the AggMo optimization procedure as a linear dynamical systems in K + 1 variables: DISPLAYFORM0 . . .

The spectral norm of the matrix B determines the rate at which the linear dynamical system converges and thus bounds ||x t − x * || 2 BID13 .

We can write down the exact form of B as follows, DISPLAYFORM1 DISPLAYFORM2 We note in particular that in the special case of K = 1 (CM) we recover the characteristic equation of O'Donoghue & Candes FORMULA0 : DISPLAYFORM3 Which in turn yields the critical damping coefficient and optimal rate, with DISPLAYFORM4 When β < β * the system is over-damped and exhibits slow monotone convergence FIG0 ).

When β > β * the system is under-damped and the characteristic equation yields imaginary solutions that correspond to oscillations FIG0 ) with convergence rate equal to 1 − |β|.

At the critical damping coefficient the convergence is optimal at 1.0 DISPLAYFORM5 We can combine this analysis with Theorem 2 from BID29 to recover similar convergence bounds for Nesterov momentum.

Producing FIG2 To produce the curves in FIG2 we compute the eigenvalues directly from the matrix B for matrices A with varying condition numbers.

While we can find the optimal learning rate for CM and Nesterov momentum in closed form we have been unable to do so for AggMo.

Therefore, we instead perform a fine-grained grid search to approximate the optimal learning rate for each condition number.

Studying Velocity We now present a brief study illustrating how using multiple velocities can break oscillations during optimization.

FIG0 shows the optimization of a 1-D quadratic function with CM, Nesterov, and AggMo.

The shaded region around each curve represents the direction and relative magnitude of the velocities term during optimization.

CM (a) has a single velocity and oscillates at a near-constant amplitude.

For Nesterov momentum (b) we display the velocity and the "error-correcting" term.

AggMo (c) has shaded regions for each velocity.

For AggMo, the velocity with β = 0.9 oscillates at a higher frequency and thus damps the whole system.

Here we present the proof of Theorem 5 1.

We introduce some simplifying notation used in BID2 .

We write g t = ∇f (θ t ), with g t,i denoting the i th element of the vector g t .

We further write g 1:t,i ∈ R t for the i th dimension of gradients up to iteration t.

We begin with the following lemma, Lemma 1.

We write v i t,j to indicate the j th element of the i th velocity at time t. Assume g t is bounded, then the following holds for all j, DISPLAYFORM0 Proof We begin by expanding the last term in the sum using the update equations, DISPLAYFORM1 The first inequality is obtained via Cauchy-Schwarz and by noting that β (i)t ≤ β for all t. The second inequality follows directly from the fact that DISPLAYFORM2 .

We can apply this upper bound to each term of the sum over t, DISPLAYFORM3 Under equality we swap the order of sums and collect terms under g t .

The third inequality follows from DISPLAYFORM4 The fourth inequality is an application of Cauchy-Schwarz.

The final inequality is from the harmonic sum bound:T t=1 1/t ≤ 1 + log(T ).

This completes the proof.

From the update equations we may write, DISPLAYFORM0 We now shift focus to only the j th dimension.

We subtract θ * j from both sides and square, DISPLAYFORM1 We can rearrange this expression and bound as follows, DISPLAYFORM2 The first inequality is an application of Young's inequality.

For the second inequality we use the sum-of-squares inequality.

We now make use of convexity, and take the sum over dimensions and time, DISPLAYFORM3 We now make use of the bounding assumptions, DISPLAYFORM4 The first two terms are collapsed using a telescoping sum.

Using t λ t−1 √ t ≤ 1/(1 − λ) 2 , we achieve the following bound, DISPLAYFORM5

While studying the convergence properties of AggMo we made several interesting observations which presented theoretical challenges.

We present some of these observations here to shed light on key differences between AggMo and existing momentum methods.

We hope that these will provoke further study.

Further reduction of B In Appendix B we derived the matrix B in order to get bounds on the convergence.

We can further reduce B to block diagonal form, where the j th block takes the form, DISPLAYFORM0 From this relatively simple form we may be able to derive a closed-form solution for the eigenvalues which would allow us to reason theoretically about the quadratic convergence properties of AggMo.

An easier goal would be finding suitable conditions under which the eigenvalues are complex and the system is under-damped.

In this section we demonstrate that the dynamics of AggMo can be written as a (K + 1)-th order finite difference equation.

While most momentum methods can be viewed as the discretization of second order ODEs it seems that AggMo does not fall into this class of algorithms.

As a consequence, it becomes difficult to apply existing convergence proof techniques to AggMo.

For simplicity, we assume a fixed learning rate γ for all time steps.

We will first tackle the special case K = 2.

From the AggMo update rule, we have DISPLAYFORM0 Denoting the matrices as symbols correspondingly, it becomes v = Bv − g, therefore DISPLAYFORM1 t+1 ), plugging Eq 10 into it, we have DISPLAYFORM2 Which reduces to the following finite difference equation, DISPLAYFORM3 For K ≥ 2, we only need to change Eq 9 accordingly, follow the remaining derivations, and recover a (K +1)-th order difference equation.

We could also derive the same result using sequence elimination, made simpler with some sensible variable substitutions.

This result is of considerable importance.

Existing momentum methods can generally be rewritten as a second order difference equation (Section 2 in O'Donoghue & Candes FORMULA0 ) which then induce a second order ODE BID28 BID32 .

The momentum optimization procedure can then be thought of as a discretization of a Hamiltonian flow.

On the other hand, AggMo does not obviously lend itself to the analytical tools developed in this setting -it is not obvious whether the form in AggMo is indeed a discretization of a Hamiltonian flow.

All of our experiments are conducted using the pytorch library BID23 .

In each experiment we make use of early stopping to determine the run with the best validation performance.

For the autoencoders we train fully connected networks with encoders using the following architecture: 784-1000-500-250-30.

The decoder reverses this architecture.

We use relu activations throughout the network.

We train for a total of 1000 epochs using a multiplicative learning rate decay of 0.1 at 200, 400, and 800 epochs.

We train using batch sizes of 200.For these experiments the training set consists of 90% of the training data with the remaining 10% being used for validation.

For each optimizer we searched over the following range of learning rates: { 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001}.

For each of the classification tasks we train for a total of 400 epochs using batchsizes of 128.

We make use of a multiplicative learning rate decay of 0.1 at 150 and 250 epochs.

For each of these experiments we use 80% of the training data for training and use the remaining 20% as validation.

In these experiments we searched over the following learning rates for all optimizers: { 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001 }.

We searched over the same damping coefficients as in the autoencoder experiments.

Each model was trained for a total of 500 epochs.

When training without batch normalization we explored a smaller range of learning rates for both CM and AggMo: { 0.1, 0.05, 0.01, 0.005 }.

The CNN-5 model uses relu activations throughout and 2x2 max pooling with stride 2.

The first convolutional layer uses an 11x11 kernel with a stride of 4.

This is followed by a max pooling layer.

There is then a 5x5 convolutional kernel followed by max pooling.

The network then uses three 3x3 convolutional layers and a final max pooling layer before feeding into a fully connected output layer.

We do not use any regularization when training this model.

We use the ResNet-32 architecture on both CIFAR-10 and CIFAR-100.

We make use of a weight decay of 0.0005 and use batch normalization BID8 .

We introduce data augmentation by using random crops with a padding of 4 and use random horizontal flips with probability 0.5.

We train LSTMs with 3-layers containing 1150 hidden units per layer, and a 400 embedding size.

Within the network we use dropout on the layers with probability 0.4.

The hidden layers use dropout with probability 0.3 and the input embedding layers use dropout with probability 0.65 while the embedding layer itself uses dropout with probability 0.1.

We also apply the weight drop method proposed in BID19 with probability 0.5.

L2 regularization is applied on the RNN activations with a scaling of 2.0, we also use temporal activation regularization (slowness regularization) with scaling 1.0.

Finally, all weights receive a weight decay of 1.2e-6.We train the model using variable sequence lengths and batch sizes of 80.

We measure the validation loss during training and decrease the learning rate if the validation loss has not decreased for 15 epochs.

We found that a learning rate decay of 0.5 worked best for all optimizers except for SGD which achieved best performance with a fixed learning rate.

For SGD, CM, AggMo and Nesterov we searched over learning rates in the range {50, 30, 10, 5, 2.5, 1, 0.1, 0.01}. We found that Adam required much smaller learning rates in this setting and so searched over values in the range {0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001}. We searched over the damping coefficients as in the previous experiments.

Each model was trained for 750 epochs, as in BID19 .

In this section we display some of the experimental results which we are unable to fit in the main paper.

To better understand how AggMo is able to help in non-convex settings we explore its effectiveness on a simple non-convex toy problem.

The function we aim to optimize is defined as follows, where a and b are constants which may be varied.

We choose this function because it features flat regions and a series of non-convex funnels with varied curvature.

The optimizer must traverse the flat regions quickly whilst remaining stable within the funnels.

This function has an optimal value at (x, y) = (0, 0).

FIG0 compares the performance of classical momentum and aggregated momentum when optimizing Equation 13 with a = 8, b = 10.

We see that GD with β = 0 and β = 0.9 are unable to leave the flat region around x < −1.

For GD with β = 0.999 the optimizer enters the funnels but frequently becomes unstable with oscillations and finally overshoots the optimum.

Compared to GD, AggMo is able to quickly traverse both the flat region and the funnels while remaining stable.

AggMo also successfully slows down quickly once reaching the optimum.

DISPLAYFORM0

In this section we present results using the default damping coefficient settings for the autoencoder and LSTM experiments.

The default settings for CM, Nesterov, and AggMo are compared in TAB4 .

The default settings of AggMo outperform both CM and Nesterov significantly.

Moreover, while the AggMo default settings perform similarly to the best results in Table 1 there is a large gap for the CM and Nesterov defaults.

This suggests that for this task AggMo is less sensitive to hyperparameter tuning than the other methods.

For the LSTM experiments we found that all methods worked best with their default damping coefficients except for Nesterov momentum which used β = 0.99.

For Nesterov momentum with β = 0.9 the validation perplexity was 63.67 and the test perplexity was 61.45.

AggMo with default settings achieved better training, validation and test perplexity than both the CM and Nesterov defaults.

In this section we present a continuous analog of AggMo which provides additional insight into its effectiveness.

The AggMo update rule features the average of several velocities with some chosen damping coefficients, β.

A natural extension to this formulation instead considers a mapping from beta values to velocities with the space of velocities being integrated over instead of summed.

Explicitly, we write this update rule as, DISPLAYFORM0 Where π(b) is a probability density defined on [0, 1].

We can link this back to aggregated momentum in the following way.

If we sampled b (i) under the density π for i = 1 : M then the procedure described by Equation 3 is approximating Equation 14 via Monte Carlo Integration.

Although this seems like a reasonable idea, it is not obvious whether we can compute this integral in closed form.

We can understand this update rule by expanding v t recursively, DISPLAYFORM1 Thus we can write the update rule for x t as, DISPLAYFORM2 Thus to compute the update rule we must compute the raw moments of b. Fortunately, for the special case where π is the density function of a Beta distribution then we have closed form solutions for the raw moments of b ∼ Beta(α, β) (note that β here is not referring to a damping coefficient) then these raw moments have a closed form: DISPLAYFORM3 This provides a closed form solution to compute θ t given θ t−1 and the history of all previous gradients.

We refer to this update scheme as Beta-Averaged Momentum.

Unfortunately, each update requires the history of all previous gradients to be computed.

We may find some reasonable approximation to the update rule.

For example, we could keep only the T most recently computed gradients.

FIG0 shows the optimization of 1D quadratics using Beta-Averaged Momentum.

The trajectories are similar to those achieved using the original AggMo formulation.

@highlight

We introduce a simple variant of momentum optimization which is able to outperform classical momentum, Nesterov, and Adam on deep learning tasks with minimal hyperparameter tuning.

@highlight

Introduces a variant of momentum that aggregates several velocities with different dampening coefficients that significantly decreases oscillation

@highlight

Proposed an aggregated momentum methods for gradient based optimization by using multiple velocity vectors with different damping factors instead of a single velocity vector to improve stability

@highlight

The authors combine several update steps together to achieve aggregated momentum also demonstrating it is more stable than the other momentum methods