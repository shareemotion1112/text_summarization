The training of stochastic neural network models with binary ($\pm1$) weights and activations via continuous surrogate networks is investigated.

We derive, using mean field theory, a set of scalar equations describing how input signals propagate through surrogate networks.

The equations reveal that depending on the choice of surrogate model, the networks may or may not exhibit an order to chaos transition, and the presence of depth scales that limit the maximum trainable depth.

Specifically, in solving the equations for edge of chaos conditions, we show that surrogates derived using the Gaussian local reparameterisation trick have no critical initialisation, whereas a deterministic surrogates based on analytic Gaussian integration do.

The theory is applied to a range of binary neuron and weight design choices, such as different neuron noise models, allowing the categorisation of algorithms in terms of their behaviour at initialisation.

Moreover, we predict theoretically and confirm numerically, that common weight initialization schemes used in standard continuous networks, when applied to the mean values of the stochastic binary weights, yield poor training performance.

This study shows that, contrary to common intuition, the means of the stochastic binary weights should be initialised close to close to $\pm 1$ for deeper networks to be trainable.

The problem of learning with low-precision neural networks has seen renewed interest in recent years, in part due to the deployment of neural networks on low-power devices.

Currently, deep neural networks are trained and deployed on GPUs, without the memory or power constraints of such devices.

Binary neural networks are a promising solution to these problems.

If one is interested in addressing memory usage, the precision of the weights of the network should be reduced, with the binary case being the most extreme.

In order to address power consumption, networks with both binary weights and neurons can deliver significant gains in processing speed, even making it feasible to run the neural networks on CPUs Rastegari et al. (2016) .

Of course, introducing discrete variables creates challenges for optimisation, since the networks are not continuous and differentiable.

Recent work has opted to train binary neural networks directly via backpropagation on a differentiable surrogate network, thus leveraging automatic differentiation libraries and GPUs.

A key to this approach is in defining an appropriate differentiable surrogate network as an approximation to the discrete model.

A principled approach is to consider binary stochastic variables and use this stochasticity to "smooth out" the non-differentiable network.

This includes the cases when (i) only weights, and (ii) both weights and neurons are stochastic and binary.

In this work we study two classes of surrogates, both of which make use of the Gaussian central limit theorem (CLT) at the receptive fields of each neuron.

In either case, the surrogates are written as differentiable functions of the continuous means of stochastic binary weights, but with more complicated expressions than for standard continuous networks.

One approximation, based on analytic integration, yields a class of deterministic surrogates Soudry et al. (2014) .

The other approximation is based on the local reparameterisation trick (LRT) Kingma & Welling (2013) , which yields a class of stochastic surrogates Shayer et al. (2017) .

Previous works have relied on heuristics to deal with binary neurons Peters & Welling (2018) , or not backpropagated gradients correctly.

Moreover, none of these works considered the question of initialisation, potentially limiting performance.

The seminal papers of Saxe et al. (2013) , Poole et al. (2016) , Schoenholz et al. (2016) used a mean field formalism to explain the empirically well known impact of initialization on the dynamics of learning in standard networks.

From one perspective the formalism studies how signals propagate forward and backward in wide, random neural networks, by measuring how the variance and correlation of input signals evolve from layer to layer, knowing the distributions of the weights and biases of the network.

By studying these moments the authors in Schoenholz et al. (2016) were able to explain how heuristic initialization schemes avoid the "vanishing and exploding gradients problem" Glorot & Bengio (2010) , establishing that for neural networks of arbirary depth to be trainable they must be initialised at "criticality", which corresponds to initial correlation being preserved to any depth.

The paper makes three contributions.

The first contribution is the presentation of new algorithms, with a new derivation able to encompass both surrogates, and all choices of stochastic binary weights, or neurons.

The derivation is based on representing the stochastic neural network as a Markov chain, a simplifying and useful development.

As an example, using this representation we are easily able to extend the LRT to the case of stochastic binary neurons, which is new.

This was not possible in Shayer et al. (2017) , who only considered stochastic binary weights.

As a second example, the deterministic surrogate of Soudry et al. (2014) is easily derived, without the need for Bayesian message passing arguments.

Moreover, unlike Soudry et al. (2014) we correctly backpropagate through variance terms, as we discuss.

The second contribution is the theoretical analysis of both classes of surrogate at initialisation, through the prism of signal propagation theory Poole et al. (2016) , Schoenholz et al. (2016) .

This analysis is achieved through novel derivations of the dynamic mean field equations, which hinges on the use of self-averaging arguments Mezard et al. (1987) .

The results of the theoretical study, which are supported by numerical simulations and experiment, establish that for a surrogate of arbitrary depth to be trainable, it must be randomly initialised at "criticality".

In practical terms, criticality corresponds to using initialisations that avoid the "vanishing and exploding gradients problem" Glorot & Bengio (2010) .

We establish the following key results:

• For networks with stochastic binary weights and neurons, the deterministic surrogate can achieve criticality, while the LRT cannot.

• For networks with stochastic binary weights and continuous neurons, the LRT surrogate can achieve criticality (no deterministic surrogate exists for this case)

In both cases, the critical initialisation corresponds to randomly initialising the means of the binary weights close to ±1, a counter intuitive result.

A third contribution is the consideration of the signal propagation properties of random binary networks, in the context of training a differentiable surrogate network.

We derive these results, which are partially known, and in order to inform our discussion of the experiments.

This paper provides insights into the dynamics and training of the class of binary neural network models.

To date, the initialisation of any binary neural network algorithm has not been studied, although the effect of quantization levels has been explored through this perspective Blumenfeld et al. (2019) .

Currently, the most popular surrogates are based on the so-called "Straight-Through" estimator Bengio et al. (2013) , which relies on heuristic definitions of derivatives in order to define a gradient.

However, this surrogate typically requires the use of batch normalization, and other heuristics.

The contributions in this paper may help shed light on what is holding back the more principled algorithms, by suggesting practical advice on how to initialise, and what to expect during training.

Paper outline: In section 2 we present the binary neural network algorithms considered.

In subsection 2.1 we define binary neural networks and subsection 2.2 their stochastic counterparts.

In subsection 2.3 we use these definitions to present new and existing surrogates in a coherent framework, using the Markov chain representation of a neural network to derive variants of both the deterministic surrogate, and the LRT-based surrogates.

We derive the LRT for the case of stochastic binary weights, and both LRT and deterministic surrogates for the case of stochastic binary weights and neurons.

In section 3 we derive the signal propagation equations for both the deterministic and stochastic LRT surrogates.

This includes deriving the explicit depth scales for trainability, and solving the equations to find the critical initialisations for each surrogate, if they exist.

In section 4 we present the numerical simulations of wide random networks, to validate the mean field description, and experimental results to test the trainability claims.

Finally in section 5 we summarize the key results, and provide a discussion of the insights they provide.

A neural network model is typically defined as a deterministic non-linear function.

We consider a fully connected feedforward model, which is composed of N × N −1 weight matrices W and bias vectors b in each layer ∈ {1, . . .

, L}, with elements W ij ∈ R and b i ∈ R. Given an input vector x 0 ∈ R N0 , the network is defined in terms of the following recursion,

where the pointwise non-linearity is, for example, φ (·) = max(0, ·).

We refer to the input to a neuron, such as h , as the pre-activation field.

A deterministic binary neural network simply has weights W ij ∈ {±1} and φ (·) = sign(·), and otherwise the same propagation equations.

Of course, this is not differentiable, thus we define stochastic binary variables in order to smooth out the non-differentiable network.

The product of training a surrogate of a stochastic binary network is ideally a deterministic binary network that is able to generalise from its training set.

We could also use the stochastic binary network, but this is not as computationally advantageous in standard hardware.

In stochastic binary neural networks we denote the matrices as S with all weights 1 S ij ∈ {±1} being independently sampled binary variables with probability is controlled by the mean M ij = ES ij .

Neuron activation in this model are also binary random variables, due to pre-activation stochasticity and to inherent noise.

We consider parameterised neurons such that the mean activation conditioned on the pre-activation is given by some function taking values in

We write the propagation rules for the stochastic network as follows:

Notice that the distribution of x factorizes when conditioning on x −1 .

The form of the neuron's mean function φ(·) depends on the underlying noise model.

We can express a binary random variable x ∈ {±1} with x ∼ p(x; θ) via its latent variable formulation x = sign(θ + αL).

In this form θ is referred to as a "natural" parameter, and the term L is a latent random noise, whose cumulative distribution function σ(·) determines the form of the non-linearity since φ(·) = 2σ(·)−1.

In general the form of φ(·) will impact on the surrogates' performance, including within and beyond the mean field description presented here.

However, a result from the analysis in Section 3 is that choosing a deterministic binary neuron, ie.

the sign(·) function, or a stochastic binary neuron, produces the same signal propagation equations, up to a scaling constant.

The idea behind several recent papers Soudry et al. (2014) , Baldassi et al. (2018) , Shayer et al. (2017) , Peters & Welling (2018) is to adapt the mean of the binary stochastic weights, with the stochastic model essentially used to "smooth out" the discrete variables and arrive at a differentiable function, open to the application of continuous optimisation techniques.

We now derive both the deterministic surrogate and LRT-based surrogates, in a common framework.

We consider a supervised classification task, with training set D = {x µ , y µ } P µ=1 , with y µ the label.

we define a loss function for our surrogate model via

For a given input x µ and a realization of weights, neuron activations and biases in all layers, denoted by (S, x, b), the stochastic neural network produces a probability distribution over the classes.

Expectations over weights and activations are given by the mean values, ES = M and E[x |h ] = φ(h ).

This objective can be recognised as a (minus) marginal likelihood, thus this method could be described as Type II maximum likelihood, or empirical Bayes.

The starting point for our derivations comes from rewriting the expectation equation 3 as the marginalization of a Markov chain, with layers indexes corresponding to time indices ∈ {1, . . .

, L}.

Markov chain representation of stochastic neural network:

where in the second line we dropped from the notation p(S ; M ) the dependence on M for brevity.

Therefore, for a stochastic network the forward pass consists in the propagation of the joint distribution of layer activations, p(x |x µ ), according to the Markov chain.

We drop the explicit dependence on the initial input x µ from now on.

In what follows we will denote with φ(h ) the average value of x according to p(x ).

The first step to obtaining a differentiable surrogate is to introduce continuous random variables.

We take the limit of large layer width and appeal to the central limit theorem to model the field h as Gaussian, with meanh and covariance matrix Σ .

Assumption 1: (CLT for stochastic binary networks) In the large N limit, under the Lyapunov central limit theorem, the field h = 1 √ N −1 S x −1 + b converges to a Gaussian random variable

While this assumption holds true for large enough networks, due to S and x −1 independency, the Assumption 2 below, is stronger and tipically holds only at initialization.

Assumption 2: (correlations are zero) We assume the independence of the pre-activation field h between any two dimensions.

Specifically, we assume the covariance Σ = Cov(h , h ) to be well approximated by Σ M F (φ(h −1 )), with MF denoting the mean field (factorized) assumption, where

This assumption approximately holds assuming the neurons in each layer are not strongly correlated.

In the first layer this is certainly true, since the input neurons are not random variables 2 .

In subsequent layers, since the fields h i and h j share stochastic neurons from the previous layer, this cannot be assumed to be true.

We expect this correlation to not play a significant role, since the weights act to decorrelate the fields, and the neurons are independently sampled.

However, the choice of surrogate influences the level of dependence.

The sampling procedure used within the local reparametrization trick reduces correlations since variables are sampled, while the deterministic surrogate entirely discards them.

We obtain either surrogate model by successively approximating the marginal distributions, p(x ) = dh p(x |h ) ≈p(x ), starting from the first layer.

We can do this by either (i) marginalising over 2 In this case the variance is actually

the Gaussian field using analytic integration, or (ii) sampling from the Gaussian.

After this, we use the approximationp(x i ) to form the Gaussian approximation for the next layer, and so on.

Deterministic surrogate: We perform the analytic integration based on the analytic form of p(x +1 i |h ) = σ(x i h i ), with σ(·) a sigmoidal function.

In the case that σ(·) is the Gaussian CDF, we obtainp(x i ) exactly 3 by the Gaussian integral of the Gaussian cumulative distribution function,

Since we start from the first layer, all random variables are marginalised out, and thush i has no dependence on random h −1 j via the neuron means φ(h ) as in Assumption 1.

Instead, we have

In the case that σ(·) is the Gaussian CDF, then ϕ (·) is the error function.

Finally, the forward pass can be expressed as

This is a more general formulation than that in Soudry et al. (2014) , which considered sign activations, which we obtain in the appendices as a special case.

Furthermore, in all implementations we backpropagate through the variance terms Σ

M F , which were ignored in the previous work of Soudry et al. (2014) .

Note that the derivation here is simpler as well, not requiring complicated Bayesian message passing arguments, and approximations therein.

LRT surrogate: The basic idea here is to rewrite the incoming Gaussian field h ∼ N (µ, Σ) as

Thus expectations over h can be written as expectations over and approximated by sampling.

The resulting network is thus differentiable, albeit not deterministic.

The forward propagation equations for this surrogate are

The local reparameterisation trick (LRT) Kingma & Welling (2013) has been previously used to obtain differentiable surrogates for binary networks.

The authors of Shayer et al. (2017) considered only the case of stochastic binary weights, since they did not write the network as a Markov chain.

Peters & Welling (2018) considered stochastic binary weights and neurons, but relied on other approximations to deal with the neurons, having not used the Markov chain representation.

The result of each approximation, applied successively from layer to layer by either propagating means and variances or by, produces a differentiable function of the parameters M ij .

It is then possible to perform gradient descent with respect to the M and b. Ideally, at the end of training we obtain a binary network that attains good performance.

This network could be a stochastic network, where we sample all weights and neurons, or a deterministic binary network.

A deterministic network might be chosen taking the most likely weights, therefore setting W ij = sign(M ij ), and replacing the stochastic neurons with sign(·) activations.

We first recount the formalism developed in Poole et al. (2016) .

Assume the weights of a standard continuous network are initialised with ab .

The mean field approximation used here replaces each element in the pre-activation field h i by a Gaussian random variable whose moments are matched.

Assuming also independence within a layer; Eh i;a h j;a = q aa δ ij and Eh i;a h j;b = q ab δ ij , one can derive recurrence relations from layer to layer,

2 the standard Gaussian measure.

The recursion for the covariance is given by

where

ab ) 2 z 2 , and we identify c ab as the correlation in layer .

The other important quantity is the slope of the correlation recursion equation or mapping from layer to layer, denoted as χ, which is given by:

We denote χ at the fixed point c * = 1 as χ 1 .

As discussed Poole et al. (2016) , when χ 1 = 1, correlations can propagate to arbitrary depth. , as explained in Poole et al. (2016) .

Therefore controlling χ 1 will prevent the gradients from either vanishing or growing exponentially with depth.

We thus define critical initialisations as follows.

This definition also holds for the surrogates which we now study.

For the deterministic surrogate model we assume at initialization that the binary weight means M ij are drawn independently and identically from a distribution P (M ), with mean zero and variance of the means given by σ 2 m .

For instance, a valid distribution could be a clipped Gaussian 4 , or another stochastic binary variable, for example

We show in Appendix B that the stochastic and deterministic binary neuron cases reduce to the same signal propagation equations, up to scaling constants.

In light of this, we consider the deterministic sign(·) neuron case, since equation for the field is slightly simpler:

which we can be read from the Eq. 7.

As in the continuous case we are interested in computing the variance q aa = 1 N i (h i;a ) 2 and covariance Eh i;a h j;b = q ab δ ij , via recursive formulae.

The key to the derivation is recognising that the denominator Σ M F,ii is a self-averaging quantity Mezard et al. (1987) .

This means it concentrates in probability to its expected value for large N .

Therefore we can safely replace it with its expectation.

Following this self-averaging argument, we can take expectations more readily as shown in the appendices.

We find the variance recursion to be

Based on this expression, and assuming q aa = q bb , the correlation recursion can be written as

The slope of the correlation mapping from layer to layer, when the normalized length of each input is at its fixed point q aa = q bb = q * (σ m , σ b ), denoted as χ, is given by:

where u a and u b are defined exactly as in the continuous case.

Refer to the appendices for full details of the derivation.

The condition for critical initialisation is χ 1 = 1, since this determines the stability of the correlation map fixed point c * = 1.

Note that for the deterministic surrogate this is always a fixed point.

We can solve for the hyper-parameters (σ

This can be established by rearranging Equations 13 and 15.

We solve for σ 2 b numerically, as shown in Figure 4 , for different neuron noise models and hence non-linearities ϕ(·).

We find that the critical initialisation for any of these design choices is close to the point (σ 2 m , σ 2 b ) = (1, 0).

However, it is not just the singleton point, as for example in Hayou et al. (2019) for the ReLu case for standard networks.

We plot the solutions in the Appendix.

The depth scales, as derived in Schoenholz et al. (2016) provide a quantitative indicator to the number of layers correlations will survive for, and thus how trainable a network is.

Similar depth scales can be derived for these deterministic surrogates.

Asymptotically in network depth , we expect that |q aa − q * | ∼ exp(− ξq ) and |c ab − c * | ∼ exp(− ξc ), where the terms ξ q and ξ c define the depth scales over which the variance and correlations of signals may propagate.

We are most interested in the correlation depth scale, since it relates to χ.

The derivation is identical to that of Schoenholz et al. (2016) .

One can expand the correlation c ab = c * + , and assuming q aa = q * , it is possible to write

The depth scale ξ −1 c are given by the log ratio log

We plot this depth scale in Figure 2 .

We derive the variance depth scale in the appendices, since it is different to the standard continuous case, but not of prime practical importance.

From Equation 8, the pre-activation field for the perturbed surrogate with both stochastic binary weights and neurons is given by,

where we recall that ∼ N (0, 1).

The non-linearity φ(·) can of course be derived from any valid binary stochastic neuron model.

Appealing to the same self-averaging arguments used in the previous section, we find the variance map to be

(19) Interestingly, we see that the variance map does not depend on the variance of the means of the binary weights.

This is not immediately obvious from the pre-activation field definition.

In the covariance map we do not have such a simplification since the perturbation i,a is uncorrelated between inputs a and b. Thus the correlation map is given by

We first verify that the theory accurately predicts the average behaviour of randomly initialised networks.

We present simulations for the deterministic surrogate in Figure 1 .

We see that the average behaviour of random networks are well predicted by the mean field theory.

Estimates of the variance and correlation are plotted, with dotted lines corresponding to empirical means and the shaded area corresponding to one standard deviation.

Theoretical predictions are given by solid lines, with strong agreement for even finite networks.

Similar plots can be produced for the LRT surrogate.

In Appendix D we present the variance and correlation depth scales as a function of σ m , and different curves corresponding to different bias variance values σ b .

Here we test experimentally the predictions of the mean field theory by training networks to overfit a dataset in the supervised learning setting, having arbitrary depth and different initialisations.

We consider first the performance of the deterministic and LRT surrogates, not their corresponding binary networks.

We use the MNIST dataset with reduced training set size (50%) and record the training performance (percentage of the training set correctly labeled) after 10 epochs of gradient descent over the training set, for various network depths L < 70 and different mean variances σ 2 m ∈ [0, 1).

The optimizer used was Adam Kingma & Ba (2014) with learning rate of 2 × 10 −4 chosen after simple grid search, and a batch size of 64.

We see that the experimental results match the correlation depth scale derived, which are overlaid as dotted curves.

A proportion of 6ξ c was found to indicate the maximum attenuation in signal strength before trainability becomes difficult, for continuous networks.

The deterministic surrogate appears to share this, but a different scaling was found for the LRT surrogate.

The reason we see the trainability not diverging in Figure 2 is that training time increases with depth, on top of requiring smaller learning rates for deeper networks, as described in detail in Saxe et al. (2013) .

The experiment here used the same number of epochs regardless of depth, meaning shallower networks actually had an advantage over deeper networks.

Note that this theory does not specify for how many steps of training the effects of critical initialisation will persist.

Therefore, the number of steps we trained the network for is an arbitrary choice, and thus the experiments validate the theory in a more qualitative than quantitative way.

Results were similar for other optimizers, including SGD, SGD with momentum, and RMSprop.

Note that these networks were trained without dropout, batchnorm or any other heuristics.

In Figure 3 we present the training performance for the deterministic surrogate and its stochastic binary counterpart.

The results for a deterministic binary network were similar to a single Monte Carlo sample.

Once again, we test our algorithms on the MNIST dataset and plot results after 5 epochs.

We see that the performance of the stochastic network matches more closely the performance of the continuous surrogate as the number of samples increases, from N = 5 to N = 100 samples.

We can report that the number of samples necessary to achieve better classification, at least for more shallow networks, appears to depends on the number of training epochs.

In some way, this is a sensible relationship, since during the course of training we might expect the means of the weights to polarise, moving closer to the bounds ±1.

Likewise, from experience continuous with neural networks, the neurons, which initially have zero mean pre-activations, are expected to "saturate" during training, that is, they become either always "on" (+1) or "off" (−1).

A stochastic network being "closer" to deterministic would require fewer samples overall.

We can again report that this phenomena was observed.

This first study of two classes of surrogate networks, and the derivation of their initialisation theories has yielded results of practical significance.

Based on the results of Section 3, in particular Claims 1-3, we can offer the following advice.

If a practitioner is interested in training networks with binary weights and neurons, one should use the deterministic surrogate, not the LRT surrogate, since the latter has no critical initialisation.

If a practitioner is interested in binary weights only,the LRT in this case does have a critical initialisation (and is the only choice from amongst these two classes of surrogate).

Furthermore, both networks are critically initialised when σ 2 b → 0 and by setting the means of the weights to ±1.

Interesting results were uncovered for the binary neural networks corresponding to the trained surrogate.

It was seen that during training, when evaluating the stochastic binary counterparts concurrently with the surrogate, the performance of binary networks is worse than the continuous model, especially as depth increases.

We reported that the stochastic binary network, with more samples, outperformed the deterministic binary network.

This makes sense since the objective optimised is the expectation over an ensemble of stochastic binary networks.

A study of random binary networks, included in the Appendices, and published recently Blumenfeld et al. (2019) for a different problem, showed that binary networks are always in a chaotic phase.

Of course, when evaluating any binary network which is trained via gradient descent on a given surrogate model, signals have different average behaviour through the corresponding binary network.

It makes sense that the closer one is to the early stages of the training process, the closer the signal propagation behaviour is to the randomly initialised case.

It is likely that as training progresses the behaviour of the binary counterparts approaches that of the trained surrogate.

Any such difference would not be observed for a heuristic surrogate as used in Courbariaux & Bengio (2016) or Rastegari et al. (2016) , which has no continuous forward propagation equations.

The form of each neuron's probability distribution depends on the underlying noise model.

We can express a stochastic binary random variable S ∈ {±1} with S ∼ p(S; θ) via its latent variable formulation,

In this form θ is referred to as a "natural" parameter, from the statistics literature on exponential families.

The term L is a latent random noise, which determines the form of the probability distribution.

We also introduce a scaling α to control the variance of the noise, so that as α → 0 the neuron becomes a deterministic sign function.

Letting α = 1 for simplicity, we see that the probability of the binary variable taking a positive value is

where p(L) is the known probability density function for the noise L. The two common choices of noise models are Gaussian or logistic noise.

The Gaussian of course has shifted and scaled erf(·) function as its cumulative distribution.

The logistic random variable has the classic "sigmoid" or logistic function as its CDF, σ(z) = 1 1+e −z .

Thus, the probability of a the variable being positive is a function of the CDF.

In the Gaussian case, this is Φ(θ).

By symmetry, the probability of p(S = −1) = Φ(−θ).

Thus, we see the probability distribution for the binary random variable in general is the CDF of the noise L, and we write p(S) = Φ(Sθ).

In the logistic noise case, we have p(S) = σ(Sθ)

For the stochastic neurons, the natural parameter is the incoming field h i = j S i,j x −1 j + b i .

Assuming this is approximately Gaussian in the large layer width limit, we can successively marginalise over the stochastic inputs to each neuron, calculating an approximation of each neuron's probability distribution,p(x i ).

This approximation is then used in the central limit theorem for the next layer, and so on.

For the case of neurons with latent Gaussian noise as part of the binary random variable model, the integration over the pre-activation field (assumed to be Gaussian) is exact.

Explicitly,

where Φ(·) is the CDF of the Gaussian distribution.

We have again Σ M F denoting the mean field approximation to the covariance between the stochastic binary pre-activations.

The Gaussian expectation of the Gaussian CDF is a known identity, which we state in more generality in the next section, where we also consider neurons with logistic noise.

This new approximate probability distributionp(x i ) can then used as part of the Gaussian CLT applied at the next layer, since it determines the means of the neurons in the next layer,

If we follow these setps from layer to layer, we see that we are actually propagating approximate means for the neurons, combined non-linearly with the means of the weights.

Given the approximately analytically integrated loss function, it is possible to perform gradient descent with respect to the means and biases, M ij and b i .

In the case of deterministic sign() neurons we obtain particularly simple expressions.

In this case the "probability" of a neuron taking, for instance, positive is just Heaviside step function of the incoming field.

Denoting the Heaviside with Θ(·), we have

We can write out the network forward equations for the case of deterministic binary neurons, since it is a particularly elegant result.

In general we havē

where φ(·) = erf(·) is the mean of the next layer of neurons, being a scaled and shifted version of the neuron's noise model CDF.

The constant is η = 1 √ 2

, standard for the Gaussian CDF to error functin conversion.

We now present the integration of stochastic neurons with logistic as well as Gaussian noise as part of their latent variable models.

The logistic case is an approximation built on the Gaussian case, motivated by approximating the logistic CDF with the Gaussian CDF.

The reason we may be interested in using logistic CDFs, rather than just considering latent Gaussian noise models which integrate exactly, is not justified in any rigorous or experimental way.

Any such analysis would likely consider the effect of the tails of the logistic versus the Gaussian distributions, where the logistic tails are much heavier than those of the Gaussian.

One historic reason for considering the logistic function, we note, is the prevalence of logistic-type functions (such as tanh(·)) in the neural network literature.

The computational cost of evaluating either logistic or error functions is similar, so there is no motivation from the efficiency side.

Instead it seems a historic preference to have logistic type functions used with neural networks.

As we saw in the previous subsection, the integration over the analytic probability distribution for each neuron gave a function which allows us to calculate the means of the neurons in the next layer.

Therefore, we directly calculate the expression for the means.

The Gaussian integral of the Gaussian CDF was used in the previous section to derive the exact probability distribution for the stochastic binary neuron in the next layer.

The result is well known, and can be stated in generality as follows,

We can integrate a logistic noise binary neuron using this result as well.

The idea is to approximate the logistic noise with a suitably scaled Gaussian noise.

However, since the overall network approximation results in propagating means from layer to layer, we can equivalently need to approximate the tanh(·) with the with the erf.

Specifically, if we have f (x; α) = tanh(

, by requiring equality of derivatives at the origin.

In order to establish this, consider

and

Equating these, gives

The approximate integral over the stochastic binary neuron mean is then

If we so desire, we can approximate this again with a tanh(·) using the tanh(·) to erf(·) approximation in reverse.

The scale parameter of this tanh(·) will be α 2 = π 4αγ .

If α = 1 as is standard, then

Assume a stochastic neuron with some latent noise, as per the previous appendix, with

).

The field is given by

We see that the expression for the variance of the field simplifies as follows,

By similar steps, we find that in the deterministic binary neuron case, we would obtain the same expression, albeit with a different scaling constant.

This is easily seen by inspection of the field term in the deterministic neuron case,

which again was derived in the previous appendix.

Here we present the derivations for the signal propagation in the continuous network models studied in the paper.

We first calculate the variance given a signal:

Where for us:

and

Where, Eφ 2 h l−1 j,a can be written explicitly, taking into account that h l−1 j,a ∼ N (0, q aa ):

We can now perform the following change of variable:

Then:

In the first layer, input neurons are not stochastic: they are samples drawn from the Gaussian distribution x 0 ∼ N 0, q 0 :

To determine the correlation recursion we start from its definition:

where q l ab represents the covariance of the pre-activations h l i,a and h l i,b , related to two distinct input signals and therefore defined as:

Replacing the pre-activations with their expressions provided in eq. (41) and taking advantage of the self-averaging argument, we can then write:

At this point, given that q

To check the stability at the fixed point, we need to compute the slope of the correlations mapping from layer to layer at the fixed point:

where we get rid of σ b because independent from c l−1 ab .

Replacing the definition of u a and u b provided in the continuous model, we can explicitly compute the derivative with respect to c l−1 ab :

where we have defined A and B as:

An alternative perspective on critical initialisation, to be contrasted with the forward signal propagation theory, is that we are simply attempting to control the mean squared singular value of the input-output Jacobain matrix of the entire network, which we can decompose into the product of single layer Jacobian matrices.

In standard networks, the single layer Jacobian mean squared singular value is equal to the derivative of the correlation mapping χ as established in Poole et al. (2016) .

For the Gaussian model studied here this is not true, and corrections must be made to calculate the true mean squared singular value.

This can be seen by observing the terms arising from denominator of the pre-activation field,

Since Σ ii is a quantity that scales with the layer width N , it is clear that when we consider squared quantities, such as the mean squared singular value, the second term, from the derivative of the denominator, will vanish in the large layer width limit.

Thus the mean squared singular value of the single layer Jacobian approaches χ.

We will proceed as if χ is the exact quantity we are interested in controlling.

The analysis involved in determining whether the mean squared singular value is well approximated by χ essentially takes us through the mean field gradient backpropagation theory as described in Schoenholz et al. (2016) .

This idea provides complementary depth scales for gradient signals travelling backwards.

We present, in slightly more detail, the signal propagation equations for the case of continuous neurons and stochastic binary weights yields the variance map,

Thus, once again, the variance map does not depend on the variance of the means of the binary weights.

The covariance map however does retain a dependence on σ and we have the derivative of the correlation map given by

We recount the argument from the paper here.

Since the mean variance σ 2 m does not appear in the variance map, we must once again consider different conditions for critical initialisation.

Specifically, from the correlation map we have a fixed point c * = 1 if and only if

In turn, the condition χ 1 = 1 holds if

Thus, to find the critical initialisation, we need to find a value of q aa = Eφ 2 (h .

This is confirmed by experiment, as we reported in the paper.

It is of course possible to investigate this perturbed surrogate for different noise models.

For example, given different noise scaling κ, as in the previous chapter, there will be a corresponding σ 2 b that satisfy the critical initialisation condition.

We leave such an investigation to future work, given the case of binary weights and continuous neurons does not appear to be of a particular interest over the binary neuron case.

In this neural network, it should be understood that all neurons are simply sign(·) functions of their input, and all weights W ij ∈ {±1} are randomly distributed according to

thus maintaining a zero mean.

The pre-activation field is given by

So, the length map is:

Interestingly, this is the same value as for the perturbed Gaussian with stochastic binary weights and neurons.

The covariance evolves as

we again have a correlation map: We can find this correlation in closed form.

First we rewrite our integral with h, for a joint density p(h a , h b ), and then rescale the h a such that the variance is 1, so that dh a = √ q aa dv a

where p(v a , v b ) is a joint with the same correlation c ab (which is now equal to its covariance), and the capital P (v 1 , v 2 ) corresponds to the (cumulative) distribution function.

A standard result for standard bivariate normal distributions with correlation ρ,

Recall that sin −1 (1) = π 2 , so we have that c * = 1 is a fixed point always.

We will now derive its slope, denoted as χ =

We can see that the derivative χ diverges at c ab = 1, meaning that there is no critical initialisation for this system.

This of course means that correlations will not propagate to arbitrary depth in deterministic binary networks, as one might have expected.

We begin again with the variance map, q

where x h l−1 j,a denotes a stochastic binary neuron whose natural parameter is the pre-activation from the previous layer.

The expectation for the length map is defined in terms of nested conditional expectations, since we wish to average over all random elements in the forward pass, q aa = E h E x|h x h (93) Once again, this is the same value as for the perturbed Gaussian with stochastic binary weights and neurons.

Similarly, the covariance map gives us,

Similar arguments to the above show that the equations for this case are exactly equivalent to the perturbed surrogate model.

This means that no critical initialisation exists in this case either.

A legitimate immediate concern with initialisations that send σ 2 m → 1 may be that the binary stochastic weights S ij are no longer stochastic, and that the variance of the Gaussian under the central limit theorem would no longer be correct.

First recall the CLT's variance is given by Var(h ) = j (1−m 2 j x 2 j ).

If the means m j → ±1 then variance is equal in value to j m 2 j (1−x 2 j ), which is the central limit variance in the case of only stochastic binary neurons at initialisation.

Therefore, the applicability of the CLT is invariant to the stochasticity of the weights.

This is not so of course if both neurons and weights are deterministic, for example if neurons are just tanh() functions.

@highlight

signal propagation theory applied to continuous surrogates of binary nets;  counter intuitive initialisation; reparameterisation trick not helpful

@highlight

The authors investigate the training dynamics of binary neural networks when using continuous surrogates, study what properties networks should have at initialization to best train, and provide concrete advice about stochastic weights at initialization.

@highlight

An in-depth exploration of stochastic binary networks, continuous surrogates, and their training dynamics, with insights on how to initialize weights for best performance.