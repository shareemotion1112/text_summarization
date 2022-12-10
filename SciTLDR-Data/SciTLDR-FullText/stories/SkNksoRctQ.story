The notion of the stationary equilibrium ensemble has played a central role in statistical mechanics.

In machine learning as well, training serves as generalized equilibration that drives the probability distribution of model parameters toward stationarity.

Here, we derive stationary fluctuation-dissipation relations that link measurable quantities and hyperparameters in the stochastic gradient descent algorithm.

These relations hold exactly for any stationary state and can in particular be used to adaptively set training schedule.

We can further use the relations to efficiently extract information pertaining to a loss-function landscape such as the magnitudes of its Hessian and anharmonicity.

Our claims are empirically verified.

Equilibration rules the long-term fate of many macroscopic dynamical systems.

For instance, as we pour water into a glass and let it be, the stationary state of tranquility is eventually attained.

Zooming into the tranquil water with a microscope would reveal, however, a turmoil of stochastic fluctuations that maintain the apparent stationarity in balance.

This is vividly exemplified by the Brownian motion BID3 : a pollen immersed in water is constantly bombarded by jittery molecular movements, resulting in the macroscopically observable diffusive motion of the solute.

Out of the effort in bridging microscopic and macroscopic realms through the Brownian movement came a prototype of fluctuation-dissipation relations BID6 BID37 .

These relations quantitatively link degrees of noisy microscopic fluctuations to smooth macroscopic dissipative phenomena and have since been codified in the linear response theory for physical systems BID28 BID9 BID16 , a cornerstone of statistical mechanics.

Machine learning begets another form of equilibration.

As a model learns patterns in data, its performance first improves and then plateaus, again reaching apparent stationarity.

This dynamical process naturally comes equipped with stochastic fluctuations as well: often given data too gigantic to consume at once, training proceeds in small batches and random selections of these mini-batches consequently give rise to the noisy dynamical excursion of the model parameters in the loss-function landscape, reminiscent of the Brownian motion.

It is thus natural to wonder if there exist analogous fluctuation-dissipation relations that quantitatively link the noise in mini-batched data to the observable evolution of the model performance and that in turn facilitate the learning process.

Here, we derive such fluctuation-dissipation relations for the stochastic gradient descent algorithm.

The only assumption made is stationarity of the probability distribution that governs the model parameters at sufficiently long time.

Our results thus apply to generic cases with non-Gaussian mini-batch noises and nonconvex loss-function landscapes.

Practically, the first relation (FDR1) offers the metric for assessing equilibration and yields an adaptive algorithm that sets learning-rate schedule on the fly.

The second relation (FDR2) further helps us determine the properties of the lossfunction landscape, including the strength of its Hessian and the degree of anharmonicity, i.e., the deviation from the idealized harmonic limit of a quadratic loss surface and a constant noise matrix.

Our approach should be contrasted with recent attempts to import the machinery of stochastic differential calculus into the study of the stochastic gradient descent algorithm BID21 BID20 BID22 BID19 BID33 BID4 BID12 BID39 .

This line of work all assumes Gaussian noises and sometimes additionally employs the quadratic harmonic approximation for loss-function landscapes.

The more severe drawback, however, is the usage of the analogy with continuous-time stochastic differential equations, which is inconsistent in general (see Section 2.3.3).

Instead, the stochastic gradient descent algorithm can be properly treated within the framework of the KramersMoyal expansion BID36 BID7 BID30 BID29 BID18 .The paper is organized as follows.

In Section 2, after setting up notations and deriving a stationary fluctuation-dissipation theorem (FDT), we derive two specific fluctuation-dissipation relations.

The first relation (FDR1) can be used to check stationarity and the second relation (FDR2) to delineate the shape of the loss-function landscape, as empirically borne out in Section 3.

An adaptive scheduling method is proposed and tested in Section 3.3.

We conclude in Section 4 with future outlooks.

A model is parametrized by a weight coordinate, θ = {θ i } i=1,...,P .

The training set of N s examples is utilized by the model to learn patterns in the data and the model's overall performance is evaluated by a full-batch loss function, f (θ) ≡ 1 Ns Ns α=1 f α (θ), with f α (θ) quantifying the performance of the model on a particular sample α: the smaller the loss is, the better the model is expected to perform.

The learning process can thus be cast as an optimization problem of minimizing the loss function.

One of the most commonly used optimization schemes is the stochastic gradient descent (SGD) algorithm BID31 in which a mini-batch B ⊂ {1, 2, . . .

, N s } of size |B| is stochastically chosen for training at each time step.

Specifically, the update equation is given by DISPLAYFORM0 where η > 0 is a learning rate and a mini-batch loss f DISPLAYFORM1 with . . .

m.b. denoting the average over mini-batch realizations.

For later purposes, it is convenient to define a full two-point noise matrix C through DISPLAYFORM2 and, more generally, higher-point noise tensors DISPLAYFORM3 Below, we shall not make any assumptions on the distribution of the noise vector ∇f B -other than that a mini-batch is independent and identically distributed from the N s training samples at each time step -and the noise distribution is therefore allowed to have nontrivial higher connected moments indicative of non-Gaussianity.

It is empirically often observed that the performance of the model plateaus after some training through SGD.

It is thus natural to hypothesize the existence of a stationary-state distribution, p ss (θ), that dictates the SGD sampling at long time (see Section 2.3.4 for discussion on this assumption).

For any observable quantity, O (θ), -something that can be measured during training such as θ 2 and f (θ) -its stationary-state average is then defined as In general the probability distribution of the model parameters evolves as p(θ, t DISPLAYFORM4 DISPLAYFORM5 and in particular for the stationary state DISPLAYFORM6 Thus follows the master equation DISPLAYFORM7 In the next two subsections, we apply this general formula to simple observables in order to derive various stationary fluctuation-dissipation relations.

Incidentally, the discrete version of the FokkerPlanck equation can be derived through the Kramers-Moyal expansion, considering the more general nonstationary version of the above equation and performing the Taylor expansion in η and repeated integrations by parts BID36 BID7 BID30 BID29 BID18 .

Applying the master equation (FDT) to the linear observable, DISPLAYFORM0 We thus have ∇f = 0 .This is natural because there is no particular direction that the gradient picks on average as the model parameter stochastically bounces around the local minimum or, more generally, wanders around the loss-function landscape according to the stationary distribution.

Performing similar algebra for the quadratic observable θ i θ j yields DISPLAYFORM1 In particular, taking the trace of this matrix-form relation, we obtain DISPLAYFORM2 More generally, in the case of SGD with momentum µ and dampening ν, whose update equation is given by DISPLAYFORM3 DISPLAYFORM4 a similar derivation yields (see Appendix A) DISPLAYFORM5 The last equation reduces to the equation (FDR1) when µ = ν = 0 with v = −∇f B .

Also note that θ · (∇f ) = (θ − θ c ) · (∇f ) for an arbitrary constant vector θ c because of the equation (8).This first fluctuation-dissipation relation is easy to evaluate on the fly during training, exactly holds without any approximation if sampled well from the stationary distribution, and can thus be used as the standard metric to check if learning has plateaued, just as similar relations can be used to check equilibration in Monte Carlo simulations of physical systems BID32 .

[It should be cautioned, however, that the fluctuation-dissipation relations are necessary but not sufficient to ensure stationarity BID27 .]

Such a metric can in turn be used to schedule changes in hyperparameters, as shall be demonstrated in Section 3.3.

Applying the master equation (FDT) on the full-batch loss function and Taylor-expanding it in the learning rate η yields the closed-form expression DISPLAYFORM0 where we recalled the equation FORMULA3 and introduced DISPLAYFORM1 In particular, DISPLAYFORM2 is the Hessian matrix.

Reorganizing terms, we obtain DISPLAYFORM3 (FDR2) In the case of SGD with momentum and dampening, the left-hand side is replaced by (1 − ν) (∇f ) 2 − µ v · ∇f and C i1,i2,...,i k by more hideous expressions (see Appendix A).We can extract at least two types of information on the loss-function landscape by evaluating the dependence of the left-hand side, G(η) ≡ (∇f ) 2 , on the learning rate η.

First, in the small learning rate regime, the value of 2G(η)/η approximates Tr H C around a local ravine.

Second, nonlinearity of G(η) at higher η indicates discernible effects of anharmonicity.

In such a regime, the Hessian matrix H cannot be approximated as constant (which also implies that {F i1,i2,...,i k } k>2 are nontrivial) and/or the noise two-point matrix C cannot be regarded as constant.

Such nonlinearity especially indicates the breakdown of the harmonic approximation, that is, the quadratic truncation of the loss-function landscape, often used to analyze the regime explored at small learning rates.

In order to gain some intuition about the fluctuation-dissipation relations, let us momentarily employ the harmonic approximation, i.e., assume that there is a local minimum of the loss function at θ = θ and retain only up to quadratic terms of the Taylor expansions around it: DISPLAYFORM0 η Tr C , linking the height of the noise ball to the noise amplitude.

This is in line with, for instance, the theorem 4.6 of the reference BID2 and substantiates the analogy between SGD and simulated annealing, with the learning rate η -multiplied by Tr C -playing the role of temperature BID1 .

Additional relations can be derived by repeating similar calculations for higher-order observables.

For example, at the cubic order, DISPLAYFORM0 The systematic investigation of higher-order relations is relegated to future work.

There is no limit in which SGD asymptotically reduces to the stochastic differential equation (SDE).

In order to take such a limit with continuous time differential dt → 0 + , each SGD update must become infinitesimal.

One may thus try dt ≡ η → 0 + , as in recent work adapting the view that SGD=SDE BID21 BID20 BID22 BID19 BID33 BID4 BID12 BID39 .

But this in turn forces the noise vector with zero mean, ∇f B −∇f , to be multiplied by dt.

This is in contrast to the scaling √ dt needed for the standard machinery of SDE -Itô-Stratonovich calculus and all that -to apply; the additional factor of dt 1/2 makes the effective noise covariance be suppressed by dt and the resulting equation in the continuous-time limit, if anything, would just be an ordinary differential equation without noise 2 [unless noise with the proper scaling is explicitly added as in stochastic gradient Langevin dynamics BID38 BID35 and natural Langevin dynamics BID23 BID24 ].In short, the recent work views η = √ η √ dt and sends dt → 0 + while pretending that η is finite, which is inconsistent.

This is not just a technical subtlety.

When unjustifiably passing onto the continuous-time Fokker-Planck equation, the diffusive term is incorrectly governed by the connected two-point noise matrix DISPLAYFORM0 rather than the full two-point noise matrix C i,j (θ) that appears herein.

3 We must instead employ the discrete-time version of the Fokker-Planck equation derived in references Van Kampen (1992); BID7 BID30 ; BID29 ; BID18 , as has been followed in the equation (6).

In contrast to statistical mechanics where an equilibrium state is dictated by a handful of thermodynamic variables, in machine learning a stationary state generically depends not only on hyperparameters but also on a part of its learning history.

The stationarity assumption made herein, which is codified in the equation (6) , is weaker than the typicality assumption underlying statistical mechanics and can hold even in the presence of lingering memory.

In the full-batch limit |B| = N s , for instance, any distribution delta-peaked at a local minimum is stationary.

For sufficiently small learning rates η as well, it is natural to expect multiple stationary distributions that form disconnected ponds around these minima, which merge upon increasing η and fragment upon decreasing η.

It is beyond the scope of the present paper to formulate conditions under which stationary distributions exist.

Indeed, if the formulation were too generic, there could be counterexamples to such a putative existence statement.

A case in point is a model with the unregularized cross entropy loss, whose model parameters keep cascading toward infinity in order to sharpen its softmax output BID25 with logarithmically diverging θ 2 BID34 .

It would be interesting to see if there are any other nontrivial caveats.

In this section we empirically bear out our theoretical claims in the last section.

To this end, two simple models of supervised learning are used (see Appendix B for full specifications): a multilayer perceptron (MLP) learning patterns in the MNIST training data (LeCun et al., 1998) through SGD without momentum and a convolutional neural network (CNN) learning patterns in the CIFAR-10 training data BID15 ) through SGD with momentum µ = 0.9.

For both models, the mini-batch size is set to be |B| = 100, and the training data are shuffled at each epoch t = Ns |B|t epoch witht epoch ∈ N. In order to avoid the overfitting cascade mentioned in Section 2.3.4, the L 2 -regularization term 1 2 λθ 2 with the weight decay λ = 0.01 is included in the loss function f .2 One may try to evade this by employing the 1/ |B|-scaling of the connected noise covariant matrix, but that would then enforces |B| → 0 + as dt → 0 + , which is unphysical.

3 Heuristically, (∇f ) 2 ∼ ηH C for small η due to the relation FDR2, and one may thus neglect the difference between C and C, and hence justify the naive use of SDE, when ηH 1 and the Gaussian-noise assumption holds.

In the similar vein, the reference BID20 proves faster convergence between SGD and SDE when the term proportional to η∇ (∇f ) 2 is added to the gradient.

Before proceeding further, let us define the half-running average of an observable O as DISPLAYFORM0 This is the average of the observable up to the time step t, with the initial half discarded as containing transient.

If SGD drives the distribution of the model parameters to stationarity at long time, then lim DISPLAYFORM1

In order to assess the proximity to stationarity, define DISPLAYFORM0 (with v replaced by −∇f B for SGD without momentum).

4 Both of these observables can easily be measured on the fly at each time step during training and, according to the relation (FDR1'), the running averages of these two observables should converge to each other upon equilibration.

Figure 1: Approaches toward stationarity during the initial trainings for the MLP on the MNIST data (a) and for the CNN on the CIFAR-10 data (b).

Top panels depict the half-running average f B (t) (dark green) and the instantaneous value f B (t) (light green) of the mini-batch loss.

Bottom panels depict the convergence of the half-running averages of the observables O L = θ · ∇f B and O R = (1+µ) 2(1−ν) ηv 2 , whose stationary-state averages should agree according to the relation (FDR1').In order to verify this claim, we first train the model with the learning rate η = 0.1 fort total epoch = 100 epochs, that is, for t total = Ns |B|t total epoch = 100Ns |B| time steps.

As shown in the figure 1, the observables O L (t) and O R (t) converge to each other.

We then take the model at the end of the initial 100-epoch training and sequentially train it further at various learning rates η (see Appendix B).

The observables O L (t) and O R (t) again converge to each other, as plotted in the figure 2.

Note that the smaller the learning rate is, the longer it takes to equilibrate.

In order to assess the loss-function landscape information from the relation (FDR2), define DISPLAYFORM0 4 If the model parameter θ happens to fluctuate around large values, for numerical accuracy, one may want to replace OL = θ · ∇f B by (θ − θc) · ∇f B where a constant vector θc approximates the vector around which θ fluctuates at long time. (dotted light-colored).

They agree at sufficiently long times but the relaxation time to reach such a stationary regime increases as the learning rate η decreases.(with the second term nonexistent for SGD without momentum).

5 Note that (∇f ) 2 is a full-batchnot mini-batch -quantity.

Given its computational cost, here we measure this first term only at the end of each epoch and take the half-running average over these sparse sample points, discarding the initial half of the run.

The half-running average of the full-batch observable O FB at the end of sufficiently long training, which is a good proxy for O FB , is plotted in the figure 3 as a function of the learning rate η.

As predicted by the relation (FDR2), at small learning rates η, the observable O FB approaches zero; its slope -divided by Tr C if preferred -measures the magnitude of the Hessian matrix, component-wise averaged over directions in which the noise preferentially fluctuates.

Meanwhile, nonlinearity at higher learning rates η measures the degree of anharmonicity experienced over the distribution p ss (θ).

We see that anharmonic effects are pronounced especially for the CNN on the CIFAR-10 data even at moderately small learning rates.

This invalidates the use of the quadratic harmonic approximation for the loss-function landscape and/or the assumption of the constant noise matrix for this model except at very small learning rates.

Saturation of the relation (FDR1) suggests the learning stationarity, at which point it might be wise to decrease the learning rate η.

Such scheduling is often carried out in an ad hoc manner but we can now algorithmize this procedure as follows:1.

Evaluate the half-running averages O L (t) and O R (t) at the end of each epoch.

OL(t) OR(t) − 1 < X, then decrease the learning rate as η → (1 − Y )η and also set t = 0 for the purpose of evaluating half-running averages.

Here, two scheduling hyperparameters X and Y are introduced, which control the threshold for saturation of the relation (FDR1) and the amount of decrease in the learning rate, respectively.

Plotted in the figure 4 are results for SGD without momentum, with the Xavier initialization BID8 and training through (i) preset training schedule with decrease of the learning rate by a factor of 10 for each 100 epochs, (ii) an adaptive scheduler with X = 0.01 (1% threshold) and 5 For the second term, in order to ensure that limt→∞ v · ∇f B (t) = limt→∞ v · ∇f (t), we measure the half-running average of v (t) · .

From top to bottom, plotted are the learning rate η, the full-batch training loss f , and prediction accuracies on the training-set images (solid) and the 10000 test-set images (dashed).These two scheduling methods span different subspaces of all the possible schedules.

The adaptive scheduling method proposed herein has a theoretical grounding and in practice much less dimensionality for tuning of scheduling hyperparameters than the presetting method, thus ameliorating the optimization of scheduling hyperparameters.

The systematic comparison between the two scheduling methods for state-of-the-arts architectures, and also the comparison with the AMSGrad algorithm for natural language processing tasks, could be a worthwhile avenue to pursue in the future.

In this paper, we have derived the fluctuation-dissipation relations with no assumptions other than stationarity of the probability distribution.

These relations hold exactly even when the noise is nonGaussian and the loss function is nonconvex.

The relations have been empirically verified and used to probe the properties of the loss-function landscapes for the simple models.

The relations further have resulted in the algorithm to adaptively set learning-rate schedule on the fly rather than presetting it in an ad hoc manner.

In addition to systematically testing the performance of this adaptive scheduling algorithm, it would be interesting to investigate non-Gaussianity and noncovexity in more details through higher-point observables, both analytically and numerically.

It would also be interesting to further elucidate the physics of machine learning by extending our formalism to incorporate nonstationary dynamics, linearly away from stationarity BID28 BID9 BID16 and beyond BID11 BID5 , so that it can in particular properly treat overfitting cascading dynamics and time-dependent sample distributions.

The author thanks Ludovic Berthier, Léon Bottou, Guy Gur-Ari, Kunihiko Kaneko, Ari Morcos, Dheevatsa Mudigere, Yann Ollivier, Yuandong Tian, and Mark Tygert for discussions.

Special thanks go to Daniel Adam Roberts who prompted the practical application of the fluctuationdissipation relations, leading to the adaptive method in Section 3.3.

For SGD with momentum µ and dampening ν, the update equation is given by DISPLAYFORM0 Here v = {v i } i=1,...

,P is the velocity and η > 0 the learning rate; SGD without momentum is the special case with µ = 0.

Again hypothesizing the existence of a stationary-state distribution p ss (θ, v), the stationary-state average of an observable O (θ, v) is defined as DISPLAYFORM1 Just as in the main text, from the assumed stationarity follows the master equation for SGD with momentum and dampening DISPLAYFORM2 For the linear observables, DISPLAYFORM3 and DISPLAYFORM4 thus v = 0 and ∇f = 0 .For the quadratic observables DISPLAYFORM5 DISPLAYFORM6 and DISPLAYFORM7 Note that the relations (26) and (27) are trivially satisfied at each time step if the left-hand side observables are evaluated at one step ahead and thus their being satisfied for running averages has nothing to do with equilibration [the same can be said about the relation (23)]; the only nontrivial relation is the equation FORMULA1 , which is a consequence of setting θ i θ j constant of time.

After taking traces and some rearrangement, we obtain the relation (FDR1') in the main text.

For the full-batch loss function, the algebra similar to the one in the main text yields DISPLAYFORM8

The MNIST training data consist of N s = 60000 black-white images of hand-written digits with 28-by-28 pixels BID17 .

We preprocess the data through an affine transformation such that their mean and variance (over both the training data and pixels) are zero and one, respectively.

Our multilayer perceptron (MLP) consists of a 784-dimensional input layer followed by a hidden layer of 200 neurons with ReLU activations, another hidden layer of 200 neurons with ReLU activations, and a 10-dimensional output layer with the softmax activation.

The model performance is evaluated by the cross-entropy loss supplemented by the L 2 -regularization term 1 2 λθ 2 with the weight decay λ = 0.01.Throughout the paper, the MLP is trained on the MNIST data through SGD without momentum.

The data are shuffled at each epoch with the mini-batch size |B| = 100.The MLP is initialized through the Xavier method BID8 and trained for t total epoch = 100 epochs with the learning rate η = 0.1.

We then sequentially train it with (η,t total epoch ) = (0.05, 500) → (0.02, 500) → (0.01, 500) → (0.005, 1000) → (0.003, 1000).

This sequential-run protocol is carried out with 4 distinct seeds for the random-number generator used in data shuffling, all starting from the common model parameter attained at the end of the initial 100-epoch run.

The figure 2 depicts trajectories for one particular seed, while the figure 3 plots means and error bars over these distinct seeds.

The CIFAR-10 training data consist of N s = 50000 color images of objects -divided into ten categories -with 32-by-32 pixels in each of 3 color channels, each pixel ranging in [0, 1] BID15 ).

We preprocess the data through uniformly subtracting 0.5 and multiplying by 2 so that each pixel ranges in [−1, 1].In order to describe the architecture of our convolutional neural network (CNN) in detail, let us associate a tuple [F, C, S, P ; M ] to a convolutional layer with filter width F , a number of channels C, stride S, and padding P , followed by ReLU activations and a max-pooling layer of width M .

Then, as in the demo at BID13 , our CNN consists of a (32, 32, 3) input layer followed by a convolutional layer with [5, 16, 1, 2; 2] , another convolutional layer with [5, 20, 1, 2; 2] , yet another convolutional layer with [5, 20, 1, 2; 2] , and finally a fully-connected 10-dimensional output layer with the softmax activation.

The model performance is evaluated by the cross-entropy loss supplemented by the L 2 -regularization term 1 2 λθ 2 with the weight decay λ = 0.01.Throughout the paper (except in Section 3.3 where the adaptive scheduling method is tested for SGD without momentum), the CNN is trained on the CIFAR-10 data through SGD with momentum µ = 0.9 and dampening ν = 0.

The data are shuffled at each epoch with the mini-batch size |B| = 100.The CNN is initialized through the Xavier method BID8 and trained for t total epoch = 100 epochs with the learning rate η = 0.1.

We then sequentially train it with (η,t

In the figure 4(a) for the MNIST classification task with the MLP, the proposed adaptive method with the scheduling hyperparameters X = 0.01 and Y = 0.1 outperforms the AMSGrad algorithm in terms of accuracy attained at long time and also exhibits a quick initial convergence.

In the figure 4(b) for the CIFAR-10 classification task with the CNN, however, while the proposed adaptive method attains better accuracy at long time, its initial accuracy gain is visibly slower than the AMSGrad algorithm.

This lag in initial accuracy gain can be ameliorated by choosing another combination of the scheduling hyperparameters, e.g., X = 0.1 and Y = 0.3, at the expense of degradation in generalization accuracy with respect to the original choice X = 0.01 and Y = 0.1.

See the FIG2 . .

From top to bottom, plotted are the learning rate η, the full-batch training loss f , and prediction accuracies on the training-set images (solid) and the 10000 test-set images (dashed).

@highlight

We prove fluctuation-dissipation relations for SGD, which can be used to (i) adaptively set learning rates and (ii) probe loss surfaces.

@highlight

Paper's concepts work in the discrete-time formalism, use the master equation, and  remove reliance on a locally quadratic approximation of the loss function or on any Gaussian asumptions of the SGD noise. 

@highlight

The authors derive the stationary fluctuation-dissipation relations that link measurable quantities and hyperparameters in SGD and use the relations to set training schedule adaptively and analyze the loss-function landscape.