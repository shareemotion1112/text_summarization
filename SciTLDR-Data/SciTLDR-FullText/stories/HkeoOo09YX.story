Stochastic gradient Markov chain Monte Carlo (SG-MCMC) has become increasingly popular for simulating posterior samples in large-scale Bayesian modeling.

However, existing SG-MCMC schemes are not tailored to any specific probabilistic model, even a simple modification of the underlying dynamical system requires significant physical intuition.

This paper presents the first meta-learning algorithm that allows automated design for the underlying continuous dynamics of an SG-MCMC sampler.

The learned sampler generalizes Hamiltonian dynamics with state-dependent drift and diffusion, enabling fast traversal and efficient exploration of energy landscapes.

Experiments validate the proposed approach on Bayesian fully connected neural network, Bayesian convolutional neural network and Bayesian recurrent neural network tasks, showing that the learned sampler outperforms generic, hand-designed SG-MCMC algorithms, and generalizes to different datasets and larger architectures.

There is a resurgence of research interests in Bayesian deep learning BID8 Blundell et al., 2015; BID10 BID9 BID4 BID33 , which applies Bayesian inference to neural networks for better uncertainty estimation.

It is crucial for e.g. better exploration in reinforcement learning (Deisenroth & Rasmussen, 2011; Depeweg et al., 2017) , resisting adversarial attacks BID2 BID19 BID24 and continual learning BID28 .

A popular approach to performing Bayesian inference on neural networks is stochastic gradient Markov chain Monte Carlo (SG-MCMC), which adds properly scaled Gaussian noise to a stochastic gradient ascent procedure BID46 .

Recent advances in this area further introduced optimization techniques such as pre-conditioning BID1 BID30 , annealing (Ding et al., 2014 ) and adaptive learning rates BID16 Chen et al., 2016) .

All these efforts have made SG-MCMC highly scalable to many deep learning tasks, including shape and texture modeling in computer vision BID17 and language modeling with recurrent neural networks BID5 .

However, inventing novel dynamics for SG-MCMC requires significant mathematical work to ensure the sampler's stationary distribution is the target distribution, which is less friendly to practitioners.

Furthermore, many of these algorithms are designed as a generic sampling procedure, and the associated physical mechanism might not be best suited for sampling neural network weights.

This paper aims to automate the SG-MCMC proposal design by introducing meta-learning techniques BID36 Bengio et al., 1992; BID26 BID43 .

The general idea is to train a learner on one or multiple tasks in order to acquire common knowledge that generalizes to future tasks.

Recent applications of meta-learning include learning to transfer knowledge to unseen few-shot learning tasks BID35 BID32 BID3 , and learning algorithms such as gradient descent (Andrychowicz et al., 2016; BID18 BID47 , Bayesian optimization BID5 and reinforcement learning (Duan et al., 2016; .

Unfortunately, these advances cannot be directly transferred to the world of MCMC samplers, as a naive neural network parameterization of the transition kernel does not guarantee the posterior distribution to be the stationary distribution of the sampler.??? An SG-MCMC sampler that extends Hamiltonian dynamics with learnable diffusion and curl matrices.

Once trained, the sampler can generalize to different datasets and architectures.??? Extensive evaluation of the proposed sampler on Bayesian fully connected neural networks, Bayesian convolutional neural networks and Bayesian recurrent neural networks, with comparisons to popular SG-MCMC schemes based on e.g. Hamiltonian Monte Carlo (Chen et al., 2014) and pre-conditioned Langevin dynamics BID16 .

Consider sampling from a target density ??(?? ?? ??) that is defined by an energy function: U (?? ?? ??), ?? ?? ?? ??? R D , ??(?? ?? ??) ??? exp(???U (?? ?? ??)).

In this paper, we focus on this sampling task with a Bayesian modeling set-up, i.e. given observed data D = {o o o n } N n=1 , we define a probabilistic model p(D, ?? ?? ??) = N n=1 p(o o o n |?? ?? ??)p(?? ?? ??), and we want samples from the target density defined as posterior distribution ??(?? ?? ??) = p(?? ?? ??|D).

We use Bayesian neural networks as an illustrating example, in this case, o o o n = (x x x n , y y y n ), the prior p(?? ?? ??) is a Gaussian N (?? ?? ??; 0 0 0, ?? ???1 I I I), and the energy function is defined as DISPLAYFORM0 log p(y y y n |x x x n , ?? ?? ??) ??? log p(?? ?? ??) = N n=1 (y y y n , NN ?? ?? ?? (x x x n )) + ??||?? ?? ??|| DISPLAYFORM1 with (y y y,?? y y) usually defined as the 2 loss for regression or the cross-entropy loss for classification.

A typical MCMC sampler constructs a Markov chain with a transition kernel, and corrects the proposed samples with Metropolis-Hastings (MH) rejection steps.

Some of these methods, e.g. Hamiltonian Monte Carlo (HMC) (Duane et al., 1987; BID27 , further augment the state space as z z z = (?? ?? ??, r r r) with auxiliary variables r r r, and sample from the augmented distribution ??(z z z) ??? exp (???H(z z z)), with the Hamiltonian H(z z z) = U (?? ?? ??) + g(r r r) such that exp(???g(r r r))dr r r = C. Thus, marginalizing out the auxiliary variable r r r will not affect the stationary distribution ??(?? ?? ??) ??? exp(???U (?? ?? ??)).For deep learning tasks, the observed dataset D often contains thousands, if not millions, of instances, making MH rejection steps computationally prohibitive.

Fortunately this is mitigated by SG-MCMC, whose transition kernel is implicitly defined by a stochastic differential equation (SDE) that leaves the target density invariant BID46 BID1 BID30 Chen et al., 2014; Ding et al., 2014) .

Such a Markov process is called It?? diffusion governed by the continuous-time SDEs: DISPLAYFORM2 with f f f (z z z) the deterministic drift, W W W (t) the Wiener process, and D D D(z z z) the diffusion matrix.

As a simple example, Langevin dynamics considers z z z = ?? ?? ??, f f f (?? ?? ??) = ?????? ?? ?? ?? U (?? ?? ??) and D D D(?? ?? ??) = I. Then using forward Euler discretization with step-size ?? the update rule of the parameters is DISPLAYFORM3 Stochastic gradient Langevin dynamics (SGLD, Welling & Teh, 2011) proposed an approximation to (3), by replacing the exact gradient ??? ?? ?? ?? U (?? ?? ??) with an estimate using a mini-batch of datapoints: DISPLAYFORM4 Therefore SGLD can be viewed as a stochastic gradient descent (SGD) that adds in a properly scaled Gaussian noise term.

Similarly, SGHMC (Chen et al., 2014) is closely related to momentum SGD (see appendix).

Furthermore, MH rejection steps are usually dropped in SG-MCMC when a carefully selected discretization step-size is in use.

Therefore SG-MCMC has the same computational complexity as many stochastic optimization algorithms, making it highly scalable for sampling posterior distributions of neural network weights conditioned on big datasets.

BID25 derived a framework of SG-MCMC samplers using advanced statistical mechanics BID50 BID37 , which explicitly parameterizes the drift f f f (z z z) : DISPLAYFORM5 with Q Q Q(z z z) the curl matrix, D D D(z z z) the diffusion matrix and ?? ?? ??(z z z) a correction term.

Remarkably BID25 showed the completeness of their framework:1.

??(z z z) ??? exp(???H(z z z)) is a stationary distribution of the SDE (2)+(5) for any pair of positive semi-definite matrix D D D(z z z) and skew-symmetric matrix Q Q Q(z z z); 2. for any It?? diffusion process that has the unique stationary distribution ??(z z z), under mild conditions there exist D D D(z z z) and Q Q Q(z z z) matrices such that the process is governed by (2)+(5).As a consequence, the construction of an SG-MCMC algorithm reduces to defining the state-space z z z and the D D D(z z z), Q Q Q(z z z) matrices.

Indeed BID25 also cast existing SG-MCMC samplers within the framework, and proposed an improved version of SG-Riemannian-HMC.

In general, an appropriate design of these two matrices leads to significant improvements in mixing as well as reduction of sample bias BID16 BID25 .

However, this design has been historically based on strong physical intuitions from statistical mechanics (Duane et al., 1987; BID27 Ding et al., 2014) .

Therefore, it can still be difficult for practitioners to understand and engineer the sampling method that is best suited to their machine learning tasks.

In the next section, we will describe our recipe on meta-learning an SG-MCMC sampler of the form (2)+(5).

Before the presentation, we emphasize that the completeness result of the framework is beneficial for our meta-learning task.

On the one hand, as meta-learning searches the best algorithm for a given set of tasks, it is crucial that the search space is large enough to contain many useful candidates.

On the other hand, some form of "correctness" guarantee is often required to achieve better generalization to test tasks that might not be very similar to the training tasks.

BID25 's completeness result indicates that our proposed method searches SG-MCMC samplers in the biggest subset of all It?? diffusion processes such that each instance is a valid posterior sampler.

Therefore, the proposed meta-learning algorithm has the best from both worlds, indeed our experiments show that the learned sampler is superior to a number of other baseline SG-MCMC methods.

This section presents a meta-learning approach to learn an SG-MCMC proposal from data.

Our aim is to design an appropriate parameterization of D D D(z z z) and Q Q Q(z z z), so that the sampler can be trained on simple tasks with a meta-learning procedure, and generalize to more complicated densities.

For simplicity, we only augment the state-space by one extra variable p p p called momentum (Duane et al., 1987; BID27 , although generalization to e.g. thermostat variable (Ding et al., 2014 ) is straightforward.

Thus, the augmented state-space is z z z = (?? ?? ??, p p p) (i.e. r r r = p p p), and the Hamiltonian is defined as H(z z z) = U (?? ?? ??) + 1 2 p p p T p p p with identity mass matrix.

For neural networks, the dimensionality of ?? ?? ?? can be at least tens of thousands.

Thus, training and applying full D D D(z z z) and Q Q Q(z z z) matrices can cause a huge computational burden, let alone gradient computations required by ?? ?? ??(z z z).

To address this, we define the preconditioning matrices as follows: DISPLAYFORM0 Here f f f ?? D and f f f ?? Q are neural network parameterized functions that will be detailed in section 3.2, and c is a small positive constant.

We choose D D D f and Q Q Q f to be diagonal for fast computation, although future work can explore low-rank matrix solutions.

From BID25 , our design has the unique stationary distribution ??(?? ?? ??) ??? exp(???U (?? ?? ??)) if f f f ?? D is non-negative for all z z z.

We discuss the role of each precondition matrix for better intuition.

The curl matrix Q Q Q(z z z) in (2) mainly controls the deterministic drift forces introduced by the energy gradient ??? ?? ?? ?? U (?? ?? ??) (as seen in many HMC-like procedures and in eq. FORMULA5 ).

Usually, we only have access to the stochastic gradient DISPLAYFORM1 should also account for the pre-conditioning effect introduced by Q Q Q f (z z z), e.g, when the magnitude of Q Q Q f (z z z) is large, we need higher friction correspondingly.

This explains the squared term DISPLAYFORM2 The positive scaling constant ?? is heuristically selected following (Chen et al., 2014; BID25 (see appendix).

Finally, the extra term DISPLAYFORM3 is responsible for compensating the changes introduced by preconditioning matrices Q Q Q(z z z) and D D D(z z z).The discretized dynamics of the state z z z = (?? ?? ??, p p p) with step-size ?? and stochastic gradient ??? ?? ?? ???? (?? ?? ??) are DISPLAYFORM4 We use a modified forward Euler discretization BID27 here, and the computation graph of eq. FORMULA10 is visualized in the right part of FIG0 (see appendix for SGHMC discretized updates).Again we see that Q Q Q f (z z z) is responsible for the acceleration of ?? ?? ??, and from the DISPLAYFORM5 controls the friction introduced to the momentum.

Note that in the big-data setting, the noisy gradient is approximately Gaussian distributed with mean 0 0 0 and variance V V V (?? ?? ??).

Observing this, BID25 further suggested a correction scheme to counter for stochastic gradient noise, which samples the Gaussian noise DISPLAYFORM6 instead.

These corrections can be dropped when the discretization step-size ?? is small, therefore, we do not consider them in our experiments.

We now present detailed functional forms for f f f ?? Q and f f f ?? D .

When designing these, our goal was to achieve a good balance between generalization power and computational efficiency.

Recall that the curl matrix Q Q Q(z z z) mainly controls the drift of the dynamics, and the desired behavior is the fast traverse through low-density regions.

One useful source of information to identify this is the energy function U (?? ?? ?? ?? ?? ?? ?? ?? ??).

1 We also include the momentum p i to the inputs of f f f ?? Q , allowing the Q Q Q(z z z) matrix to observe the velocity information of the ?? ?? ?? i .

We further add an offset ?? to Q Q Q(z z z) to prevent the vanishing of this matrix.

Putting all of them together, we define the i DISPLAYFORM0 The corresponding ?? ?? ??(z z z) term requires both DISPLAYFORM1 The energy gradient ??? ??i U (?? ?? ??) also appears in (7), so it remains to compute ??? U f f f ?? Q , which, along with ??? pi f ?? Q (U (?? ?? ??), p i ), can be obtained by automatic differentiation BID0 .Matrix D D D(z z z) is responsible for the friction and the stochastic gradient noise, which are crucial for better exploration around high-density regions.

Therefore, we also add the energy gradient ??? ??i U (?? ?? ??) to the inputs, meaning that the i DISPLAYFORM2 By the construction of the D D D(z z z) matrix, the ?? ?? ?? vector only requires ??? p p p D D D f without computing any higher order information.

In practice, both U (?? ?? ??) and ??? ??i U (?? ?? ??) are replaced by their stochastic estimates?? (?? ?? ??) and ??? ??i?? (?? ?? ??), respectively.

To keep the scale of the inputs roughly the same across tasks, we rescale all the inputs using statistics computed by simulating the sampler with randomly initialized f f f ?? D and f f f ?? Q .

When the computational budget is limited, we replace the exact gradient computation required by ?? ?? ??(z z z) with finite difference approximations.

We refer the reader to the appendix for details.

Another challenge is to design a meta-learning procedure for the sampler to encourage faster convergence and low bias on test tasks.

To achieve these goals we propose two loss functions that we named as the cross-chain loss and the in-chain loss.

From now on we consider the discretized dynamics and define q t (?? ?? ??|D) as the marginal distribution of the random variable ?? ?? ?? at time t.

Cross-chain loss We introduce cross-chain loss that encourages the sampler to converge faster.

Since the sampler is guaranteed to have the unique stationary distribution ??(?? ?? ??) ??? exp(???U (?? ?? ??)), fast convergence means that KL[q t ||??] is close to zero when t is small.

Therefore this KL-divergence becomes a sensible objective to minimize, which is equivalent to maximizing the variational lower- BID11 Beal, 2003) .

We further make the objective doubly stochastic: (1) the energy term is further approximated by its stochastic estimates?? (?? ?? ??); (2) we use Monte Carlo variational inference (MCVI, BID31 Blundell et al., 2015) which estimates the lower-bound with samples ?? ?? ?? DISPLAYFORM0 DISPLAYFORM1 k=1,t=1 are obtained by simulating K parallel Markov chains with the sampler, and the cross-chain loss is defined by accumulating the lower-bounds through time: DISPLAYFORM2 By minimizing this objective, we can improve the convergence of the sampler, especially at the early times of the Markov chain.

The objective also takes the sampler bias into account because the two distributions will match when the KL-divergence is minimized.

In-chain loss For very big neural networks, simulating multiple Markov chains is prohibitively expensive.

The issue is mitigated by thinning that collects samples for every ?? step (after burn-in), which effectively draws samples from the averaged distributionq(?? ?? ??|D) = 1 T /?? T /?? s=1 q s?? (?? ?? ??).

The in-chain loss is, therefore, defined as the ELBO evaluated at the averaged distributionq, which is then approximated by Monte Carlo with samples DISPLAYFORM3 obtained by thinning: DISPLAYFORM4 Gradient approximation We leverage the recently proposed Stein gradient estimator to estimate the intractable gradients ??? ?? log q t (?? ?? ??) for cross-chain loss and ??? ?? logq(?? ?? ??) for in-chain loss.

Precisely, by the chain rule, we have ??? ?? log q t (?? ?? ??) = ??? ?? ?? ?? ????? ?? ?? ?? log q t (?? ?? ??), so it remains to estimate the gradients G G G = ( extensions to Riemannian Langevin dynamics and HMC BID6 have also been proposed BID30 BID25 .

Our proposed sampler architecture further generalizes SG-Riemannian-HMC as it decouples the design of D D D(z z z) and Q Q Q(z z z) matrices, and the detailed functional form of these two matrices are also learned from data.

We do not use RNNs in our approach as it cannot be represented within the framework of BID25 .

We leave the combination of learnable RNN proposals to future work.

Also presented an initial attempt to meta-learn an approximate inference algorithm, which simply combined the stochastic gradient and the Gaussian noise with a neural network.

Thus the stationary distribution of that sampler (if it exists) is only an approximation to the exact posterior.

On the other hand, the proposed sampler (with ?? ??? 0) is guaranteed to be correct by the complete framework BID25 .

Very recently BID49 discussed that short-horizon meta-objectives for learning optimizers can cause a serious issue for long-time generalization.

We found this bias is less severe in our approach, again due to the fact that the learned sampler is provably correct.

Recent research also considered improving HMC with a trainable transition kernel.

BID34 improved upon vanilla HMC by introducing a trainable re-sampling distribution for the momentum.

BID39 parameterized the HMC transition kernel with a trainable invertible transformation called non-linear independent components estimation (NICE, Dinh et al., 2014) , and train it with Wasserstein adversarial training (Arjovsky et al., 2017) .

BID14 generalized HMC by augmenting the state space with a binary direction variable, and they parameterized the transition kernel with a non-volume preserving invertible transformation that is inspired by the real-valued non-volume preserving (RealNVP) flows (Dinh et al., 2017) .

The sampler is trained with the expected squared jump distance BID29 .

We note that adversarial training is less reliable for high dimensional data, thus it is not considered in this paper.

Also, the jump distance does not explicitly take the sampling bias and convergence speed into account.

More importantly, the purpose of these approaches is to directly improve the HMC-like sampler on the target distribution, and with NICE/RealNVP parametrization it is difficult to generalize the sampler to densities of different dimensions.

In contrast, our goal is to learn an SG-MCMC sampler that can later be transferred to sample from different Bayesian neural network posterior distributions, which will typically have different dimensionality and include tens of thousands of random variables.

We evaluate the meta-learned SG-MCMC sampler, which is referred to as NNSGHMC or the meta sampler in the following.

Detailed test set-ups are reported in the appendix.

The code is available at https://github.com/WenboGong/MetaSGMCMC.

We first consider sampling Gaussian variables to demonstrate fast convergence and low bias of the meta sampler.

To mimic stochastic gradient settings, we manually inject Gaussian noise with unit variance to the gradient as suggested by (Chen et al., 2014) .

The training density is a 10D Gaussian with randomly generated diagonal covariance matrix, and the test density is a 20D Gaussian.

For evaluation, we simulate K = 50 parallel chains for T = 12, 000 steps.

Then we follow BID25 to evaluate the sampler's bias measured by the KL divergence from the empirical estimate to the ground truth.

Results are visualized on the left panel of FIG1 , showing that the meta sampler both converges much faster and achieves lower bias compared to SGHMC.

The effective sample size 2 for SGHMC and NNSGHMC are 22 and 59, again indicating better efficiency of the meta sampler.

For illustration purposes, we also plot in the other two panels the trajectory of samples by simulating NNSGHMC (middle) and SGHMC (right) on a 2D Gaussian for a fixed amount of time ??T .

This confirms that the meta sampler explores more efficiently and is less affected by the injected noise.

Next, we consider Bayesian neural network classification on MNIST data with three generalization tests: network architecture generalization (NT), activation function generalization (AF) and dataset generalization (Data).

In all tests, the sampler is trained with a 1-hidden layer multi-layer perceptron (MLP) (20 units, ReLU activation) as the underlying model for the target distribution ??(?? ?? ??).

We also report long-time horizon generalization results, meaning that the simulation time steps in test time are much longer than that of training (cf.

Andrychowicz et al., 2016) .

Algorithms in comparison include SGLD BID46 ), SGHMC (Chen et al., 2014 and preconditioned SGLD (PSGLD, BID16 .

Note that PSGLD uses RMSprop-like preconditioning techniques BID44 ) that require moving average estimates of the gradient's second moments.

Therefore the underlying dynamics of PSGLD cannot be represented within our framework (6).

Thus we mainly focus on comparisons with SGLD and SGHMC, and leave the PSGLD results as reference.

The discretization step-sizes for the samplers are tuned on the validation dataset for each task.

In this test we use the trained sampler to draw samples from the posterior distribution of a 2-hidden layer MLP with 40 units and ReLU activations.

FIG3 shows the learning curves of test error and negative test log-likelihood (NLL) for 100 epochs, where the final performance is reported in TAB0 .

Overall NNSGHMC achieves the fastest convergence even when compared with PSGLD.

It has the lowest test error compared to SGLD and SGHMC.

NNSGHMC's final test LL is on par with SGLD and slightly worse than PSGLD, but it is still better than SGHMC.

Architecture + Activation function generalization (NT+AF) Next we replace NT's test network's activation function with sigmoid and re-run the same test as before.

Again results in FIG3 and TAB0 show that NNSGHMC converges faster than others for both test error and NLL.

It also achieves the best NLL results among all samplers, and the same test error as SGHMC.Architecture + Dataset generalization (NT+Data) In this test we split MNIST into training task (classifying digits 0-4) and test task (digits 5-9).

The meta sampler is trained with the smaller MLP, and it is evaluated on the task with the larger MLP with NT's architecture.

Thus, the meta sampler is trained without any knowledge of the test task's training and test data.

From FIG3 , we see that NNSGHMC, although a bit slower at the start, catches up quickly and proceeds to lower error.

The difference between these samplers NLL results is marginal, and NNSGHMC is on par with PSGLD.

Following the setup of BNN MNIST experiments, we also test our algorithm on convolutional neural networks (CNNs) for CIFAR-10 (Krizhevsky, 2009) classification, again with three generalization tasks (NT, AF and Data).

The meta sampler is trained using a smaller CNN with two convolutional layers (3 ?? 3 ?? 3 ?? 8 and 3 ?? 3 ?? 8 ?? 8) and one fully connected (fc) layer (50 hidden units).

ReLU activations, and max-pooling operators of size 2 are applied after each convolutional layer.

The meta sampler is trained using 100 "meta-epochs", where each "meta-epoch" has 5 data epochs.

At the beginning of each "meta-epoch", a "replay" technique inspired by experience replay BID21 is utilized (see appendix).

The discretization step-sizes are tuned on a validation dataset for each task.

Architecture generalization (NT) The test CNN has two convolutional layers (3 ?? 3 ?? 3 ?? 16 and 3 ?? 3 ?? 16 ?? 16) and one fc layer (100 hidden units), resulting in roughly 4?? dimenality of ?? ?? ??.

FIG6 shows that the meta sampler achieves the fastest learning at the first 10 epochs, and continues to have better performance in both test accuracy and NLL.

Interestingly, PSGLD slows down quickly after 3 epochs, and it converges to a worse answer.

The best performance over 200 epochs is shown in TAB1 , where the meta sampler is a clear winner in both accuracy and NLL.

This demonstrates that our sampler indeed converges faster and has found a better posterior mode.

Architecture + Activation function generalization (NT+AF) We use the same CNN architecture as in NT but replace the ReLU activations with sigmoid.

FIG6 and TAB1 show that the meta sampler again has better convergence speed and the best final performance.

Architecture + Dataset generalization (NT+Data) We split CIFAR-10 according to labels 0-4 as the training task and 5-9 as the test task.

We also used the same CNN architecture as in NT.

From FIG6 and TAB1 , the meta sampler consistently achieves the fastest convergence speed.

It also achieves similar accuracy as SGHMC, but it has slightly worse test NLL compared to SGHMC.

Lastly, we consider a more challenging setup: sequence modeling with Bayesian RNNs.

Here a single datum is a sequence o o o n = {x x x 1 n , ..., x x x T n } and the log-likelihood is defined as log p(o o o n |?? ?? ??) = T t=1 log p(x x x n t |x x x n 1 , . . .

, x x x n t???1 , ?? ?? ??), with each of the conditional densities produced by a gated recurrent unit (GRU) network (Cho et al., 2014) .

We consider four polyphonic music datasets for this task: Piano-midi (Piano) as training data, and Nottingham (Nott), MuseData (Muse) and JSB chorales (JSB) for evaluation.

The meta sampler is trained on a small GRU with 100 hidden states.

At test time, we follow Chen et al. FORMULA1 and set the step-size to ?? = 0.001.

We found SGLD significantly under-performs, so instead, we report the performances of two optimizers, Adam (Kingma & Ba, 2014) and Santa (Chen et al., 2016), taken from Chen et al. (2016) .

Again, these two optimizers use moving average schemes which are out of the scope of our framework, so we mainly compare the meta sampler with SGHMC and leave the others as references.

The meta sampler is tested on the four datasets using 200 unit GRU.

So for Piano this corresponds to architecture generalization only.

From Figure 6 we see that the meta sampler achieves faster convergence compared to SGHMC, at the same time it achieves similar speed as Santa at early stages.

All the samplers achieve best results close to Santa on Piano.

The meta sampler successfully generalizes to the other three datasets, demonstrating faster convergence than SGHMC consistently, and better final performance on Muse.

Interestingly, the meta sampler's final results on Nott and JSB are slightly worse than other samplers.

Presumably, these two datasets are very different from Muse and Piano, therefore, the energy landscape is less similar to the training density (see appendix).

Specifically, JSB is a dataset with much shorter sequences.

And in this case, SGHMC also exhibits over-fitting but to a smaller degree.

Therefore, we further test the meta sampler on JSB without the offset ?? in f f f ?? Q to reduce the acceleration (denoted as NNSGHMC-s).

Surprisingly, NNSGHMC-s

We have presented a meta-learning algorithm that can learn an SG-MCMC sampler on simpler tasks and generalizes to more complicated densities in high dimensions.

Experiments on Bayesian MLPs, Bayesian CNNs and Bayesian RNNs confirmed the strong generalization of the trained sampler to the long-time horizon as well as across datasets and network architectures.

Future work will focus on better designs for both the sampler and the meta-learning procedure.

For the former, temperature variable augmentation as well as moving average estimation will be explored.

For the latter, better loss functions will be proposed for faster training, e.g. by reducing the unrolling steps of the sampler during training.

Finally, the automated design of generic MCMC algorithms that might not be derived from continuous Markov processes remains an open challenge.

A COMPARING MOMENTUM SGD AND SGHMC Similar to the relationship between SGLD and SGD, SGHMC is closely related SGD with momentum (SGD-M).

First in HMC, the state space is augmented with an additional momentum variable denoted as p p p ??? R D .

We assume an identity mass matrix associated with that momentum term.

Then the corresponding drift f f f (?? ?? ??, p p p) and diffusion matrix D D D are: DISPLAYFORM0 where C C C is a positive definite matrix called friction coefficient.

Thus, HMC's continuous-time dynamics is governed by the following SDE: DISPLAYFORM1 The discretized update rule (with simple Euler discretization) of HMC with step-size ?? is DISPLAYFORM2 If stochastic gradient ????? (?? ?? ??) is used, we need to replace the covariance matrix of with 2??(C C C ???B B B) whereB B B is the variance estimation of the gradients.

On the other hand, the update equations of SGD with momentum (SGD-M) are the following: DISPLAYFORM3 where k and l are called momentum discount factor and learning rate, respectively.

Also we can rewrite the SGHMC update equations by setting ??p p DISPLAYFORM4 Thus, the discretized SGHMC updates can be viewed as the SGD-M update injected with carefully controlled Gaussian noise.

Therefore, the hyperparameter of SGHMC can be heuristically chosen based on the experience of SGD-M and vice versa.

BID27 showed that in practice, simple Euler discretization for HMC simulation might cause divergence, therefore advanced discretization schemes such as Leapfrog and modified Euler are recommended.

We use modified Euler discretization in our implementation of SGHMC and the meta sampler, resulting in the following update: DISPLAYFORM5 DISPLAYFORM6 Due to the two-stage update of Euler integrator, at time t, we have f DISPLAYFORM7 ), which is not exactly the history from the previous time.

Therefore we further approximate it using delayed estimate: DISPLAYFORM8 Similarly, the ?? ?? ?? p p p term expands as DISPLAYFORM9 We further approximate DISPLAYFORM10 ???U (?? ?? ??) by the following DISPLAYFORM11 This only requires the storage of previous Q Q Q matrix.

However, DISPLAYFORM12 ???pi requires one further forward pass to obtainf DISPLAYFORM13 Therefore the proposed finite difference method only requires one more forward passes to comput?? f f f t???1 ?? D and instead, save 3 back-propagations.

As back-propagation is typically more expensive than forward pass, our approach reduces running time drastically, especially when the sampler are applied to large neural network.

Time complexity figures Every SG-MCMC method (including the meta sampler) requires ??? ?? ?? ???? (?? ?? ??).

The main burden is the forward pass and back-propagation through the D D D(z z z) and Q Q Q(z z z) matrices, where the latter one has been replaced by the proposed finite difference scheme.

The time complexity is O(HD) for both forward pass and finite difference with H the number of hidden units in the neural network of the meta sampler.

Parallel computation with GPUs improves real-time speed, indeed in our MNIST experiment the meta sampler spends roughly 1.5x time when compared with SGHMC.

For a distribution q(?? ?? ??) that is implicitly defined by a generative procedure, the density q(?? ?? ??) is often intractable. derived the Stein gradient estimator that estimates G = (??? ?? ?? ?? 1 log q(?? ?? ?? 1 ), ?? ?? ?? ??? ?? ?? ?? K log q(?? ?? ?? K )) T on samples ?? ?? ?? 1 , ..., ?? ?? ?? K ??? q(?? ?? ??).

There are two different ways to derive this gradient estimator, here we briefly introduce one of them, and refer the readers to for details.

We start by introducing Stein's identity BID41 BID42 BID7 .

DISPLAYFORM0 be a differentiable multivariate test function which maps ?? ?? ?? to a column vector DISPLAYFORM1 T .

One can use integration by parts to show the following Stein's identity when a boundary condition lim ||?? ?? ??||?????? q(?? ?? ??)h(?? ?? ??) = 0 is assumed for the test function: DISPLAYFORM2 This boundary condition holds for almost any test function if q has sufficiently fast-decaying tails (e.g. Gaussian tails).

proposed the Stein gradient estimator for ??? ?? ?? ?? log q(?? ?? ??) by inverting a Monte Carlo (MC) version of Stein's identity (23): DISPLAYFORM3 Then G is obtained by ridge regression (with || ?? || F the Frobenius norm of a matrix) DISPLAYFORM4 which has an analytical solution?? DISPLAYFORM5 where DISPLAYFORM6 Here ?? ?? ?? k (j) denotes the j th element of vector ?? ?? ?? k .

One can show that the RBF kernel satisfies Stein's identity .

In this case h(?? ?? ??) = K(?? ?? ??, ??), d = +??? and by the reproducing kernel property, h(?? ?? ??) FORMULA1 also show that the Stein gradient estimator can be obtained by minimizing a Monte Carlo estimate of the kernelized Stein discrepancy (Chwialkowski et al., 2016; .

DISPLAYFORM7 The kernel choice It is well-known for kernel methods that a better choice of the kernel can greatly improve the performance.

However, optimal kernels are often problem specific, and they are generally difficult to obtain.

Recently, a popular approach for kernel design is to compose a simple kernel (e.g. RBF kernel) on features extracted from a deep neural network.

Representative work include deep kernel learning for Gaussian processes BID48 , and adversarial approaches to learn kernel parameters Bi??kowski et al., 2018) .

Unfortunately, both approaches do not scale very well to our application as ?? ?? ?? has at least tens of thousands of dimensions.

Furthermore, they both considered kernel learning for observed data, while in our case ?? ?? ?? is a latent variable to be inferred.

Therefore it remains a research question on how to learn kernels on latent variables efficiently, and addressing this question is out of the scope of the paper.

Instead, we follow BID22 ; to use RBF kernel for the gradient estimator.

Other kernels can be trivially adapted to our method.

We expect even better performance if an optimal kernel is in use, but we leave the investigation to future work.

Time complexity figures During meta sampler training, the Stein gradient estimator requires the kernel matrix inversion which is O(K 3 ) for cross-chain training.

In practice, we only run a few parallel Markov chains K = 20 ??? 50, thus, this will not incur huge computation cost.

For in-chain loss the computation can also be reduced with proper thinning schemes.

We visualize on the left panel of FIG9 the unrolled computation scheme.

We apply truncated back-propagate through time (BPTT) to train the sampler.

Specifically, we manually stop the gradient flow through the input of D D D and Q Q Q matrices to avoid computing higher order gradients.

We also illustrate cross-chain in-chain training on the right panel of FIG9 .

Cross-chain training encourages both fast convergence and low bias, provided that the samples are taken from parallel chains.

On the other hand, in-chain training encourages sample diversity inside a chain.

In practice, we might consider thinning the chains when performing in-chain training.

Empirically this improves the Stein gradient estimator's accuracy as the samples are spread out.

Computationally, this also prevents inverting big matrices for the Stein gradient estimator, and reduces the number of backpropagation operations.

Another trick we applied is parallel chain sub-sampling: if all the chains are used, then there is less encouragement of singe chain mixing, since the parallel chain samples can be diverse enough already to give reasonable gradient estimate.

One potential challenge is that for different tasks and problem dimensions, the energy function, momentum and energy gradient can have very different scales and magnitudes.

This affects the meta sampler's generalization, for example, if training and test densities have completely different energy scales, then the meta sampler is likely to produce wrong strategies.

This is especially the case when the meta sampler is generalized to much bigger networks or to very different datasets.

To mediate this issue, we propose to pre-process the inputs to both f f f ?? D and f f f ?? Q networks to make it at similar scale as those in training task.

Recall that the energy function is U (?? ?? ??)

= ??? N n=1 log p(y y y n |x x x n , ?? ?? ??) ??? log p(?? ?? ??) where the prior log p(?? ?? ??) is often an isotropic Gaussian distribution.

Thus the energy function scale linearly w.r.t both the dimensionality of ?? ?? ?? and the total number of observations N .

Often the energy function is further approximated using mini-batches of M datapoints.

Putting them together, we propose pre-processing the energy as DISPLAYFORM0 where D train and D test are the dimensionality of ?? ?? ?? in the training task and the test task, respectively.

Importantly, for RNNs N represents the total sequence length, namely N = N data n=1 T n , where N data is the total number of sequences and T n is the sequence length for a datum x x x n .

We also define M accordingly.

The momentum and energy gradient magnitudes are estimated by simulating a randomly initialized meta sampler for short iterations.

With these statistics we normalize both the momentum and the energy gradient to have roughly zero mean and unit variance.

We train our meta sampler on a 10D uncorrelated Gaussian with mean (3, ..., 3) and randomly generated covariance matrix.

We do not set any offset and additional frictions, i.e. ?? = 0 and ?? = 0.

The noise estimation matrixB B B are set to be 0 for both meta sampler and SGHMC.

To mimic stochastic gradient, we manually inject Gaussian noise with zero mean and unit variance into ??? ?? ?? ???? (?? ?? ??) = ??? ?? ?? ?? U (?? ?? ??) + , ??? N (0 0 0, I I I).

The functions f f f ?? D and f f f ?? Q are represented by 1-hidden-layer MLPs with 40 hidden units.

For training task, the meta sampler step size is 0.01.

The initial positions are drawn from Uniform ([0, 6] D ).

We train our sampler for 100 epochs and each epochs consists 4 x 100 steps.

For every 100 steps, we updates the Q Q Q and D D D matrices using Adam optimizer with learning rate 0.0005.

Then we continue the updated sampler with last position and momentum until 4 sub-epochs are finished.

We re-initialize the momentum and position.

We use both cross-chain and in-chain losses.

The Stein Gradient estimator uses RBF kernel with bandwidth chosen to be 0.5 times the median-heuristic estimated value.

We unroll the Markov Chain for 20 steps before we manually stop the gradient.

For cross-chain training, we take sampler across chain for each 2 time steps.

For in-Chain, we discard initial 50 points for burn-in and sub-sample the chain with batch size 5.

We thin the samples for every 3 steps.

For both training and evaluation, we run 50 parallel Markov Chains.

The test task is to draw samples from a 20D correlated Gaussian with with mean (3, ..., 3) and randomly generated covariance matrix.

The step size is 0.025 for both meta sampler and SGHMC.

To stabilize the meta sampler we also clamp the output values of f f f ?? Q within [???5, 5] .

The friction matrix for SGHMC is selected as I I I.

In MNIST experiment, we apply input pre-processing on energy function as in FORMULA2 and scale energy gradient by 70.

Also, we scale up f f f ?? D by 50 to account for sum of stochastic noise.

The offset ?? is selected as 0.01 ?? as suggested by Chen et al. (2014) , where ?? = lr N with lr the per-batch learning rate.

We also turn off the off-set and noise estimation, i.e. ?? = 0 andB B B = 0.

We run 20 parallel chains for both training and evaluation.

We only adopt the cross chain training with thinning samplers of 5 times step.

We also use the finite difference technique during evaluation to speed-up computations.

We train the meta sampler on a smaller BNN with architecture 784-20-10 and ReLU activation function, then test it on a larger one with architecture 784-40-40-10.

In both cases the batch size is 500 following Chen et al. (2014) .

Both f f f ?? D and f f f ?? Q are parameterized by 1-hidden-layer MLPs with 10 units.

The per-batch learning rate is 0.007.

We train the sampler for 100 epochs and each one consists of 7 sub-epochs.

For each sub-epoch, we run the sampler for 100 steps.

We re-initialize ?? ?? ?? and momentum after each epoch.

To stabilize the meta sampler in evaluation, we first run the meta sampler with small per-batch learning rate 0.0085 for 3 data epochs and clamp the Q Q Q values.

After, we increase the per-batch learning rate to 0.018 with clipped f f f ?? Q .

The learning rate for SGHMC is 0.01 for all times.

For SGLD and PSGLD, they are 0.2 and 1.4 ?? 10 ???3 respectively.

These step-sizes are tuned on MNIST validation data.

We modify the test network's activation function to sigmoid.

We use almost the same settings as in network generalization tests, except that the per-batch learning rates are tuned again on validation data.

For the meta sampler and SGHMC, they are 0.18 and 0.15.

For SGLD and PSGLD, they are 1 and 1.3 ?? 10 ???2 .

We train the meta sampler on ReLU network with architecture 784-20-5 to classify images 0-4, and test the sampler on ReLU network 784-40-40-5 to classify images 5-9.

The settings are mostly the same as in network architecture generalization for both training and evaluation.

One exception is again the per-batch learning rate for PSGLD, which is tuned as 1.3 ?? 10 ???3 .

Note that even though we use the same per-batch learning rate as before, the discretization step-size is now different due to smaller training dataset, thus, ?? will be automatically adjusted accordingly.

CIFAR-10 dataset contains 50,000 training images with 10 labels and 10,000 test images.

We train our meta sampler using smaller CNN classifier with two convolutional layer (3 ?? 3 ?? 3 ?? 8 and 3 ?? 3 ?? 8 ?? 8, no padding) and one fc layer of 50 hidden units.

Therefore the dimensionality of ?? ?? ?? is 15, 768.

The training sampler discretization step-size ?? is 0.0007 50000. and scaling term is ?? = 0.005 ?? .

To make it analogous to optimization methods, we call 0.0007 as per-batch learning rate and 0.005 as friction coefficient.

The f f f ?? Q and f f f ?? D are defined by 2-layer MLPs with 10 hidden units.

We set the offset values to 0 for both Q Q Q and D D D. Further, we scale up the output of D D D f (z z z) by 10 and its gradient input ???U (?? ?? ??) by 100.

We scale up the energy input U (?? ?? ??) to both f f f ?? Q and f f f ?? D by 5.

We train our meta sampler using 100 "meta epoch" with 5 data epoch and 500 batch size.

Within each "meta epoch", we repeat the following computation for 10 times: we run 50 parallel chains using the meta sampler for 50 iterations (0.5 dataset epoch), compute the loss function, and update the meta sampler's parameters using Adam.

We manually stop the gradient after 20 iterations.

Then we start the next sub-epoch using the last ?? ?? ?? and p p p.

After we finish all sub-epoch, we re-initialize the ?? ?? ?? and p p p using replay techniques with probability 0.15.

The sub-sample chain number for in-chain loss is set to 5.

Experience replay BID21 ) is a technique broadly used in reinforcement learning literature.

Inspired by this, in Bayesian CNN experiments we train the meta sampler in a similar way, and we found this replay technique particularly useful for more complicated dataset like CIFAR-10.At the beginning of each "meta epoch", each chain is initialized either with a specific state randomly chosen from a replay pool, or with a random state sampled from a Gaussian distribution.

We use a pre-defined replay probability to control the replay strategy.

The replay pool is updated after each sub-epoch, and it has a queue-like data structure of constant size, so that the old states are replaced by the new ones.

Therefore, this replay technique is useful for both short-time and long-time horizon generalization.

On one hand, the meta sampler can continue with previous states, allowing it to accommodate long-time horizon behavior.

On the other hand, due to non-zero probability of random restart, the meta sampler can learn a better strategy for fast convergence.

Therefore with this replay technique, the sampler can observe both burn-in and roughly-converged behavior, and this balance is controlled by the replay probability.

For architecture generalization, the test CNN has two convolutional layer (3 ?? 3 ?? 3 ?? 16 and 3 ?? 3 ?? 16 ?? 16, no padding) and one fully connected layer with 100 hidden units.

Thus, the dimensionality of ?? ?? ?? is 61,478, roughly 4 times of the training dimension.

We run 20 parallel chains in test time.

We split the 50,000 training images into 45,000 training and 5,000 validation images, and tune the discretization step-size of each sampling and optimization methods on the validation set for 80 epochs.

For test, we run the tuned samplers/optimizers for 200 data epoch (roughly 40 times longer than training) to ensure convergence.

For the meta sampler, the per-batch rate is 0.003.

For SGHMC, the per-batch is also 0.003 with friction coefficient 0.01.

For SGLD, the per-batch learning rate is 0.15.

PSGLD uses 1.3 ?? 10 ???3 as learning rate and 0.99 as moving average term.

For optimization methods, we use learning rate 0.002 for Adam and 0.003 for SGD-M. The momentum term is 0.9.

To prevent overfitting, we use weight penalty with coefficient 0.001.

The test CNN has same architecture as in NT, except that it replaces all ReLU activation functions with sigmoid activations.

We fix all other parameters for sampling method and only re-tune the step sizes using same setup as in NT.

The per-batch rate for meta sampler, SGHMC, SGLD and PSGLD are 0.1, 0.03, 0.5 and 0.005 respectively.

For optimization methods, the step size for Adam and SGD-M are 0.002 and 0.03 respectively.

We split the CIFAR-10 training and test dataset according to the labels.

We use training data with labels 0-4 for meta sampler training, training data with labels 5-9 for test CNN training, and test data with labels 5-9 for test CNN evaluation.

Thus, the meta sampler has no access to the test task's training and test data during sampler training.

We train our sampler using the same scaling terms as in NT but reduce the discretization step-size to 0.0005.

The rest setup is the same as in NT.We use the same test CNN architecture and ReLU activation as in NT, and tune the learning rate using validation data.

The step size for the meta sampler, SGHMC, SGLD and PSGLD are 0.0015, 0.005,

The Piano data is selected as the training task, which is further split into training, validation and test subsets.

We use batch-size 1, meaning that the energy and the gradient are estimated on a single sequence.

The meta sampler uses similar neural network architectures as in MNIST tests.

The training and evaluation per-batch learning rate for all the samplers is set to be 0.001 following Chen et al. (2016) .

We train the meta sampler for 40 epochs with 7 sub-epochs with only cross chain loss.

Each sub-epochs consists 70 iterations.

We scale the D D D output by 20 and set ?? =

?? , where ?? is defined in the same way as before.

We use zero offset during training, i.e. ?? = 0.

We apply input pre-processing for both f f f ?? D and f f f ?? Q .

To prevent divergence of the meta sampler at early training stage.

We also set the constant of c = 100 to the f ?? D .

For dataset generalization, we tune the off-set value based on Piano validation set and transfer the tuned setting ?? = ???1.5 to the other three datasets.

For Piano architecture generalization, we do not tune any hyper-parameters including ?? and use exactly same settings as training.

Exact gradient is used in RNN experiments instead of computing finite differences.

We list some data statistics in TAB3 which roughly indicates the similarity between datasets.

Piano dataset is the smallest in terms of data number, however, the averaged sequence length is the largest.

Muse dataset is similar to Piano in sequence length and energy scale but much larger in terms of data number.

On the other hand, Nott dataset has very different energy scale compared to the other three.

This potentially makes the generalization much harder due to inconsistent energy scale fed into f f f ?? Q and f f f ?? D .

For JSB, we notice a very short sequence length on average, therefore the GRU model is more likely to over-fit.

Indeed, some algorithms exhibits significant over-fitting behavior on JSB dataset compared to other data (Santa is particularly severe).

We also run the samplers using the same settings as in MNIST experiments for a short period of time (500 iterations).

We also compare to other optimization methods including momentum SGD (SGD-M) and Adam.

We use the same per-batch learning rate for SGD-M and SGHMC as in MNIST experiment.

For Adam, we use 0.002 for ReLU and 0.01 for Sigmoid network.

The results are shown in Figure 8 .

Meta sampler and Adam achieves the fastest convergence speed.

This again confirms the faster convergence of the meta sampler especially at initial stages.

We also provide additional contour plots (Figure 9) for MNIST experiments to demonstrate the strategy learned by f f f ?? D for reference.

@highlight

This paper proposes a method to automate the design of stochastic gradient MCMC proposal using meta learning approach. 

@highlight

Prsents a meta-learning approach to automatically design MCMC sampler based on Hamiltonian dynamics to mix faster on problems similar to training problems

@highlight

Parameterizes diffusion and curl matrices by neural networks and meta-learn and optimize an sg-mcmc algorithm. 