We propose a framework to understand the unprecedented performance and robustness of deep neural networks using field theory.

Correlations between the weights within the same layer can be described by symmetries in that layer, and networks generalize better if such symmetries are broken to reduce the redundancies of the weights.

Using a two parameter field theory, we find that the network can break such symmetries itself towards the end of training in a process commonly known in physics as spontaneous symmetry breaking.

This corresponds to a network generalizing itself without any user input layers to break the symmetry, but by communication with adjacent layers.

In the layer decoupling limit applicable to residual networks (He et al., 2015), we show that the remnant symmetries that survive the non-linear layers are spontaneously broken based on empirical results.

The Lagrangian for the non-linear and weight layers together has striking similarities with the one in quantum field theory of a scalar.

Using results from quantum field theory we show that our framework is able to explain many experimentally observed phenomena, such as training on random labels with zero error (Zhang et al., 2017), the information bottleneck and the phase transition out of it (Shwartz-Ziv & Tishby, 2017), shattered gradients (Balduzzi et al., 2017), and many more.

Deep neural networks have been used in image recognition tasks with great success.

The first of its kind, AlexNet BID8 , led to many other neural architectures have been proposed to achieve start-of-the-art results in image processing at the time.

Some of the notable architectures include, VGG BID12 , Inception BID14 and Residual networks (ResNet) BID3 .Understanding the inner workings of deep neural networks remains a difficult task.

It has been discovered that the training process ceases when it goes through an information bottleneck (ShwartzZiv & Tishby, 2017 ) until learning rate is decreased to a suitable amount then the network under goes a phase transition.

Deep networks appear to be able to regularize themselves and able to train on randomly labeled data BID18 with zero training error.

The gradients in deep neural networks behaves as white noise over the layers BID1 .

And many other unexplained phenomena.

A recent work BID0 showed that the ensemble behavior and binomial path lengths BID15 of ResNets can be explained by just a Taylor series expansion to first order in the decoupling limit.

They found that the series approximation generates a symmetry breaking layer that reduces the redundancy of weights, leading to a better generalization.

Because the ResNet does not contain such symmetry breaking layers in the architecture.

They suggest that ResNets are able to break the symmetry by the communication between the layers.

Another recent work also employed the Taylor expansion to investigate ResNets BID6 .In statistical terms, a quantum theory describes errors from the mean of random variables.

We wish to study how error propagate through each layer in the network, layer by layer.

In the limit of a continuous sample space, the quantum theory becomes a quantum field theory.

The effects of sampling error and labelling error can then be investigated.

It is well known in physics that a scalar field can drive a phase transition.

Using a scalar field theory we show that a phase transition must exist towards the end of training based on empirical results.

It is also responsible for the remarkable performance of deep networks compared to other classical models.

In Appendix D, We explain that quantum field theory is likely one of the simplest model that can describe a deep network layer by layer in the decoupling limit.

Much of the literature on neural network design focuses on different neural architecture that breaks symmetry explicitly, rather than spontaneously.

For instance, non-linear layers explicitly breaks the symmetry of affine transformations.

There is little discussion on spontaneous symmetry breaking.

In neural networks, the Goldstone theorem in field theory states that for every continuous symmetry that is spontaneously broken, there exists a weight with zero Hessian eigenvalue at the loss minimum.

No such weights would appear if the symmetries are explicitly broken.

It turns out that many seemingly different experimental results can be explained by the presence of these zero eigenvalue weights.

In this work, we exploit the layer decoupling limit applicable to ResNets to approximate the loss functions with a power series in symmetry invariant quantities and illustrate that spontaneous symmetry breaking of affine symmetries is the sufficient and necessary condition for a deep network to attain its unprecedented power.

The organization of this paper is as follows.

The background on deep neural networks and field theory is given in Section 2.

Section 3 shows that remnant symmetries can exist in a neural network and that the weights can be approximated by a scalar field.

Experimental results that confirm our theory is given in Section 4.

We summarize more evidence from other experiments in Appendix A. A review of field theory is given in Appendix B. An explicit example of spontaneous symmetry breaking is shown in Appendix C.

In this section we introduce our frame work using a field theory based on Lagrangian mechanics.

A deep neural network consists of layers of neurons.

Suppose that the first layer of neurons with weight matrix W 1 and bias b 1 takes input x 1 and outputs y 1 DISPLAYFORM0 where x = (x , 1) and W 1 = (W 1 , b 1 ), where W 1 and b 1 are real valued.

Now suppose that R 1 denotes a nonlinear operator corresponding to a sigmoid or ReLU layer located after the weight layer, so that DISPLAYFORM1 For a neural network with T repeating units, the output for layer t is DISPLAYFORM2

We now show the necessary and sufficient conditions of preserving symmetry.

We explicitly include symmetry transformations in Equation (1) and investigate the effects caused by a symmetry transformation of the input in subsequent layers.

Suppose Q t ∈ G is a transformation matrix in some Lie group G for all t. Note that the Q t are not parameters to be estimated.

We write y t = y t (Q t ), where the dependence on Q t is obtained from a transformation on the input, x t (Q t ) = Q t x t , and the weights, DISPLAYFORM0 If G is a symmetry group, then y t is covariant with x t , such that y t (Q t ) = Q t y t .

This requires two conditions to be satisfied.

First, DISPLAYFORM1 t , where Q −1 t Q t = I and the existence of the inverse is trivial because G is a group and Q t ∈ G. The second is the commutativity between R t and Q t , such that R t Q t = Q t R t .

For example, if g t ∈ Aff(D), the group of affine transformations, R t may not commute with g t .

However, commutativity is satisfied when the transformation corresponds to the 2D rotation of feature maps.

Including transformation matrices, the output at layer t is DISPLAYFORM2

Statistical learning requires the loss function to be minimized.

It can be written in the form of a mutual information, training error, or the Kullback-Leibler divergence.

In this section we approximate the loss function in the continuum limit of samples and layers.

Then we define the loss functional to transition into Lagrangian mechanics and field theory.

Let z i = (X i , Y i ) ∈ X be the i-th input sample in data set X , (X i , Y i ) are the features and the desired outputs, respectively, and i ∈ {1, . . .

, N }.

The loss function is DISPLAYFORM0 where W = (W 1 , . . .

, W T ), and Q = (Q 1 , . . .

, Q T ), Q t ∈ G where G is a Lie group, and T is the depth of the network.

Taking the continuum limit, DISPLAYFORM1 where p(X, Y) is the joint distribution of X and Y. Using the first fundamental theorem of calculus and taking the continuous layers (t) limit, we write DISPLAYFORM2 where L x (t = 0) is the value of the loss before training.

We let L x,t = dL x /dt be the loss rate per layer.

The loss rate L x,t is bounded from below.

Therefore DISPLAYFORM3 Minimizing the loss rate guarantees the minimization of the total loss.

We require L x,t to be invariant under symmetry transformations.

That is, if DISPLAYFORM4 However if Q 1 (t) and Q 2 (t) do not belong in the same symmetry group, the above equality does not necessarily hold.

Now we define the loss functional for a deep neural network DISPLAYFORM5

Having defined the loss functional, we can transition into Lagrangian dynamics to give a description of the feature map flow at each layer.

Let the minimizer of the loss rate be DISPLAYFORM0 From now on, we combine z = (X, Y) as Y only appears in W * in this formalism, each Y determines a trajactory for the representation flow determined by Lagrangian mechanics.

Now we define, for each i-th element of W(t), and a non-linear operator R(t) acting on W(t) such that the loss minimum is centered at the origin, DISPLAYFORM1 We now define the Lagrangian density, DISPLAYFORM2 and L = T − V, where T is the kinetic energy and V is the potential energy.

We define the potential energy to be DISPLAYFORM3 The probability density p(z) and the loss rate L x,t are invariant under symmetry transformations.

Therefore V is an invariant as well.

DISPLAYFORM4 We now set up the conditions to obtain a series expansion of V around the minimum w i (Q t ) = 0.

First, since V is an invariant.

Each term in the series expansion must be an invariant such that DISPLAYFORM5 , the orthogonal group and that w T (Q t ) = w T Q T and w(Q t ) = Q t w.

So w i w i is an invariant.

Then f = w i w i is invariant for all Q t where the Einstein summation convention was used DISPLAYFORM6 Now we perform a Taylor series expansion about the minimum w i = 0 of the potential, DISPLAYFORM7 where H i j = ∂ w i ∂

w j V is the Hessian matrix, and similarly for Λ ij mn .

The overall constant C can be ignored without loss of generality.

Because V is an even function in w i around the minimum, we must have DISPLAYFORM8 The O(D) symmetry enforces that all weight Hessian eigenvalues to be H i i = m 2 /2 for some constant m 2 .

This can be seen in the O(2) case, with constants a, b, a = b, Q ∈ O(2) such that w 1 (Q) = w 2 and w 2 (Q) = w 1 , DISPLAYFORM9 , so the O(2) symmetry implies a = b. This can be generalized to the O(D) case.

For the quartic term, the requirement that V be even around the minimum gives DISPLAYFORM10

ii ii = λ/4 for some constant λ and zero for any other elements, the potential is DISPLAYFORM0 where the numerical factors were added for convention.

The power series is a good approximation in the decoupling limit which may be applicable for Residual Networks.

1 For the kinetic term T , we expand in power series of the derivatives, DISPLAYFORM1 where the coefficient for (∂ t w) 2 is fixed by the Hamiltonian kinetic energy 1 2 (∂ t w) 2 .

Higher order terms in (∂ t w) 2 are negligible in the decoupling limit.

If the model is robust, then higher order terms in (∂ z w) 2 can be neglected as well.

2 The Lagrangian density is DISPLAYFORM2 where we have set w 2 = w i w i and absorbed c into z without loss of generality.

This is precisely the Lagrangian for a scalar field in field theory.

Standard results for a scalar field theory can be found in Appendix B. To account for the effect of the learning rate, we employ results from thermal field theory BID7 and we identify the temperature with the learning rate η.

So that now DISPLAYFORM3

Spontaneous symmetry breaking describes a phase transition of a deep neural network.

Consider the following scalar field potential invariant under O(D ) transformations, DISPLAYFORM0 where m 2 (η) = −µ 2 + 1 4 λη 2 , µ 2 > 0 and learning rate η.

There exists a value of η = η c such that m 2 = 0.

In the first phase, η > η c , the loss minimum is at w * 0i = 0, where DISPLAYFORM1 When the learning rate η drops sufficiently low, the symmetry is spontaneously broken and the phase transition begins.

The loss minimum bifurcates at η = η c into DISPLAYFORM2 This occurs when the Hessian eigenvalue becomes negative, m 2 (η) < 0, when η < η c .This phenomenon has profound implications.

It is responsible for phase transition in neural networks and generates long range correlation between representations and the desired output.

Details from field theory can be found in Appendix C. FIG0 depicts the shape of the loss rate during spontaneous symmetry breaking with a single weight w, and the orthogonal group O(D ) is reduced to a reflection symmetry O(1) = {1, −1} such that w(Q) = ±w.

At η > η c , the loss rate has a loss minima at point A. When the learning rate decreases, such that η < η c , the critical point at A becomes unstable and new minima with equal loss rate are generated.

The weight must go through B to get to the new minimum C. If the learning rate is too small, the weight will be stuck near A. This explains why a cyclical learning rate can outperform a monotonic decreasing learning rate BID13 .Because the loss rate is invariant still to the sponteneously broken symmetry, any new minima generated from spontaneous symmetry breaking must have the same loss rate.

If there is a unbroken continuous symmetry remaining, there would be a connected loss rate surface corresponding to the new minima generated by the unbroken symmetry.

Spontaneous symmetry breaking splits the weights into two sets, w → (π, σ).

The direction along this degenerate minima in weight space corresponds to π.

And the direction in weight space orthogonal to π is σ.

This has been shown experimentally by BID2 in FIG0 .

We show the case for the breaking of O(3) to O(2) in FIG2 .2 The kinetic term T is not invariant under transformation Q(t).

To obtain invariance ∂tw i is to be replaced by the covariant derivative Dtw i so that (Dtw i ) 2 is invariant under Q(t) BID10 .

The covariant derivative is DISPLAYFORM3 with B(z, t, Qt) = Q(t)B(z, t)Q(t) −1 .

The new fields B introduced for invariance is not responsible for spontaneous symmetry breaking, the focus of this paper.

So we will not consider them further.3 Formally, the ∂zw term should be part of the potential V, as T contains only ∂tw terms.

However we adhere to the field theory literature and put the ∂zw term in T with a minus sign.

In this section we show that spontaneous symmetry breaking occurs in neural networks.

First, we show that learning by deep neural networks can be considered solely as breaking the symmetries in the weights.

Then we show that some non-linear layers can preserve symmetries across the nonlinear layers.

Then we show that weight pairs in adjacent layers, but not within the same layer, is approximately an invariant under the remnant symmetry leftover by the non-linearities.

We assume that the weights are scalar fields invariant under the affine Aff(D ) group for some D and find that experimental results show that deep neural networks undergo spontaneous symmetry breaking.

Theorem 1: Deep feedforward networks learn by breaking symmetries Proof: Let A i be an operator representing any sequence of layers, and let a network formed by applying A i repeatedly such that DISPLAYFORM0 .

Then x out = Lx in for some L ∈ Aff(D) and x out can be computed by a single affine transformation L. When A i contains a nonlinearity for some i, this symmetry is explicitly broken by the nonlinearity and the layers learn a more generalized representation of the input.

Now we show that ReLU preserves some continuous symmetries.

Theorem 2: ReLU reduces the symmetry of an Aff(D) invariant to some subgroup Aff(D ), where D < D. Proof: Suppose R denotes the ReLU operator with output y t and Q t ∈ Aff(D) acts on the input x t , where R(x) = max(0, x).

Let x T x be an invariant under Aff(D) and let DISPLAYFORM1 Note that γ i can be transformed into a negative value as it has passed the ReLU already.

Corollary If there exists a group G that commutes with a nonlinear operator R, such that QR = RQ, for all Q ∈ G, then R preserves the symmetry G.Definition: Remnant Symmetry If Q t ∈ G commutes with a non-linear operator R t for all Q t , then G is a remnant symmetry at layer t.

For the loss function L i (X i , Y i , W, Q) to be invariant, we need the predicted output y T to be covariant with x i .

Similarly for an invariant loss rate L x,t we require y t to be covariant with x t .

The following theorem shows that a pair of weights in adjacent layers can be considered an invariant for power series expansion.

Theorem 3: Neural network weights in adjacent layers form an approximate invariant Suppose a neural network consists of affine layers followed by a continuous non-linearity, R t , and that the weights at layer t, W t (Q t ) = Q t W t Q −1 t , and that Q t ∈ H is a remnant symmetry such that Q t R t = R t Q t .

Then w t w t−1 can be considered as an invariant for the loss rate.

Proof: Consider x(Q t ) = Q t x t , then DISPLAYFORM2 where in the last line Q t R t = R t Q t was used, so y t (Q t ) = Q t y t is covariant with x t .

Now, x t = R t−1 W t−1 x t−1 , so that DISPLAYFORM3 The pair (R t W t )(R t−1 W t−1 ) can be considered an invariant under the ramnant symmetry at layer t.

Let w t = R t W t − R t W * t .

Therefore w t w t−1 is an invariant.

In the continuous layer limit, w t w t−1 tends to w(t)T w(t) such that w(t) is the first layer and w(t)T corresponds to the one after.

Therefore w(t) can be considered as D scalar fields under the remnant symmetry.

The remnant symmetry is not exact in general.

For sigmoid functions it is only an approximation.

The crucial feature for the remnant symmetry is that it is continuous so that strong correlation between inputs and outputs can be generated from spontaneous symmetry breaking.

In the following we will only consider exact remnant symmetries.

We will state the Goldstone Theorem from field theory without proof.

Theorem (Goldstone) For every spontaneously broken continuous symmetry, there exist a weight π with zero eigenvalue in the Hessian m 2 π = 0.

In any case, we will adhere to the case where the remnant symmetry is an orthogonal group O(D ).

Note that W is a D × D matrix and D < D. We choose a subset Γ ∈ R D of W such that Γ T Γ is invariant under Aff(D ).

Now that we have an invariant, we can write down the Lagrangian for a deep feedforward network for the weights responsible for spontaneous symmetry breaking.

Now we can use standard field theory results and apply it to deep neural networks.

A review for field theory is given in Appendix B. The formalism for spontaneous symmetry breaking is given in Appendix C.

In this section we assume that the non-linear operator is a piecewise linear function such as ReLU and set R = I to be the identity and restrict our attention to the symmetry preserving part of R (see theorem 2).

Our discussion also applies to other piecewise-linear activation functions.

According to the Goldstone theorem, spontaneous symmetry breaking splits the set of weight deviations γ into two sets (σ, π) with different behaviors.

Weights π with zero eigenvalues and a spectrum dominated by small frequencies k in its correlation function.

4 The other weights σ, have Hessian eigenvalues µ 2 as the weights before the symmetry is broken.

In Appendix C, a standard calculation in field theory shows that the correlation functions of the weights have the form Spontaneous symmetry breaking and the information bottleneck The neural network undergoes a phase transition out of the information bottleneck via spontaneous symmetry breaking described in Section 2.5.

Before the phase transition, the weights γ have positive Hessian eigenvalues m 2 .

After the phase transition, weights π with zero Hessian eigenvalues are generated by spontaneous symmetry breaking.

The correlation function for the π weights is concentrated around small values of |k|, see Equation FORMULA36 , with ω 0 = |k| for any t. This corresponds to a highly correlated representations across the sample (input) space and layers.

Because the loss is minimized, the feature maps across the network is highly correlated with the desired output.

And a large correlation across the sample space means that the representations are independent of the input.

This is shown in FIG2 of BID11 .

After phase transition, I(Y ; T ) 1 bit for all layers T , and I(X; T ) is small even for representations in early layers.

DISPLAYFORM0 Gradient variance explosion It has been shown that the variance in weight gradients in the same layer grow by an order of magnitude during the end of training BID11 .

We also connect this to spontaneous symmetry breaking.

As two sets of weights, (σ, π) are generated with different distributions.

Considering them as the same object would result in a larger variance.

We find that neural networks are resilient to overfitting.

Recall that the fluctuation in the weights can arise from sampling noise.

Then (∂ z w i ) 2 can be a measure of model robustness.

A small value denotes the weights' resistance to sampling noise.

If the network were to overfit, the weights would be very sensitive to sampling error.

After spontaneous symmetry breaking, weights at the loss minimum with zero eigenvalues obey the Klein-Gordon equation with m DISPLAYFORM0 The singularity in the correlation function suggests |k| 2 0.

The zero eigenvalue weights provide robustness to the model.

BID18 referred to this phenomenon as implicit regularization.

In this work we solved one of the most puzzling mysteries of deep learning by showing that deep neural networks undergo spontaneous symmetry breaking.

This is a first attempt to describe a neural network with a scalar quantum field theory.

We have shed light on many unexplained phenomenon observed in experiments, summarized in Appendix A.One may wonder why our theoretical model works so well explaining the experimental results with just two parameters.

It is due to the decoupling limit such that a power series in the loss function is a good approximation to the network.

In our case, the two expansion coefficients are the lowest number of possible parameters that is able to describe the phase transition observed near the end of training, where the performance of the deep network improves drastically.

It is no coincidence that our model can explain the empirical observations after the phase transition.

In fact, our model can describe, at least qualitatively, the behaviors of phase transition in networks that the decoupling limit may not apply to.

This suggests that the interactions with nearby layers are responsible for the phase transition.

In this section we summarize other experimental findings that can be explained by the proposed field theory and the perspective of symmetry breaking.

Here Q ∈ G acts on the the input and hidden variables x, h, as Qx, Qh.• The shape of the loss function after spontaneous symmetry breaking has the same shape observed by BID2 towards the end of training, see FIG0 .•

The training error typically drops drastically when learning rate is decreased.

This occurs when the learning rate drops below η c , forcing a phase transition so that new minima develop.

See FIG0 • A cyclical learning rate BID13 helps to get to the new minimum faster, see Section 2.5.• Stochasticity in gradient descent juggles the loss function such that the weights are no longer at the local maximum of FIG0 .

A gradient descent step is taken to further take the weights towards the local minimum.

Stochasticity helps the network to generalize better.• When the learning rate is too small to move away from A in FIG0 .

PReLU's BID4 could move the weight away from A through the training of the non-linearity.

This corresponds to breaking the symmetry explicitly in Theorem 1.• Results from Shwartz-Ziv & Tishby (2017) are due to spontaneous symmetry breaking, see Section 4.• Deep neural networks can train on random labels with low training loss as feature maps are highly correlated with their respective desired output.

BID18 observed that a deep neural network can achieve zero training error on random labels.

This shows that small Hessian eigenvalues is not the only condition that determines robustness.• Identity mapping outperforms other skip connections BID5 ) is a result of the residual unit's output being small.

Then the residual units can be decoupled leading to a small λ and so it is easier for spontaneous symmetry breaking to occur, from m 2 = −µ 2 + 1 4 λη 2 .• Skip connection across residual units breaks additional symmetry.

Suppose now an identity skip connection connects x 1 and the output of F 2 .

Now perform a symmetry transformation on x 1 and x 2 , Q 1 and Q 2 ∈ G , respectively.

Then the output after two residual untis is Qx 3 = Q 1 x 1 + Q 2 x 2 + Q 2 F 2 .

Neither Q = Q 1 nor Q = Q 2 can satisfy the covariance under G. This is observed by BID9 .•

The shattered gradient problem BID1 .

It is observed that the gradient in deep (non-residual) networks is very close to white noise.

This is reflected in the exponential in Equation FORMULA56 .

This effect on ResNet is reduced because of the decoupling limit λ → 0.

This leads to the weight eigenvalues m 2 being larger in non-residual networks owing to m 2 = −µ 2 + 1 4 λη 2 .

And so a higher oscillation frequency in the correlation function.• In recurrent neural networks, multiplicative gating BID16 combines the input x and the hidden state h by an element-wise product.

Their method outperforms the method with an addition x+h because the multiplication gating breaks the covariance of the output.

A transformation Qx * Qh = Q(x * h), whereas for addition the output remains covariant Qx + Qh = Q(x + h).

In this section we state the relevant results in field theory without proof.

We use Lagrangian mechanics for fields w(x, t).

Equations of motion for fields are the solution to the Euler-Lagrange equation, which is a result from the principle of least action.

The action, S, is DISPLAYFORM0 where L is the Lagrangian.

Define the Lagrangian density DISPLAYFORM1 The action in term of the Lagrangian density is DISPLAYFORM2 The Lagrangian can be written as a kinetic term T , and a potential term V (loss function), DISPLAYFORM3 For a real scalar field w(x, t), DISPLAYFORM4 where we have set the constant c 2 = 1 without loss of generality.

The potential for a scalar field that allows spontaneous symmetry breaking has the form DISPLAYFORM5 In the decoupling limit, λ → 0, the equation of motion for w is the Klein-Gordon Equation DISPLAYFORM6 In the limit of m 2 → 0, the Klein-Gordon Equation reduces to the wave equation with solution w(z, t) = e i(ωt−k·z) , DISPLAYFORM7 One can treat w as a random variable such that the probability distribution (a functional) of the scalar field w(z, t) is p[w] = exp(−S[w])/Z, where Z is some normalizing factor.

The distribution peaks at the solution of the Klein-Gordon equation since it minimizes the action S. Now we can define the correlation function between w(z 1 , t 1 ) and w(z 2 , t 2 ), DISPLAYFORM8 where Dw denotes the integral over all paths from (z 1 , t 1 ) to (z 2 , t 2 ).

In the decoupling limit λ → 0, it can be shown that DISPLAYFORM9 where Stokes theorem was used and the term on the boundary of (sample) space is set to zero.

The above integral in the exponent is quadratic in w and the integral over Dw can be done in a similar manner to Gaussian integrals.

The correlation function of the fields across two points in space and time is w(z 1 , t 1 )w(z 2 , t 2 ) = G(z 1 , t 1 , z 2 , t 2 ), where G(z 1 , t 1 , z 2 , t 2 ) is the Green's function to the Klein-Gordon equation, satisfying DISPLAYFORM10 The Fourier transformation of the correlation function is DISPLAYFORM11 An inverse transform over ω gives DISPLAYFORM12 with ω DISPLAYFORM13

In this section we show that weights π with small, near zero, eigenvalues m 2 π = 1 4 λη 2 are generated by spontaneous symmetry breaking.

Note that we can write the Lagrangian in Equation (3) as L = T − V. Consider weights γ that transforms under O(D ), from Equation (3) DISPLAYFORM0 When m 2 = −µ 2 + 1 4 λη 2 < 0, it can be shown that in this case the loss minimum is no longer at γ i = 0, but it has a degenerate minima on the surface such that i (γ i ) 2 = v, where v = −m 2 /λ.

Now we pick a point on this loss minima and expand around it.

Write γ i = (π k , v + σ), where k ∈ {1, . . .

, D − 1}. Intuitively, the π k fields are in the subspace of degenerate minima and σ is the field orthogonal to π.

Then it can be shown that the Lagrangian can be written as DISPLAYFORM1 where, in the weak coupling limit λ → 0, DISPLAYFORM2 V π = O(λ), DISPLAYFORM3 the fields π and σ decouple from each other and can be treated separately.

The σ fields satisfy the Klein-Gordon Equation ( − m 2 )σ = 0, with = ∂ 2 t − ∂ 2 z .

The π fields satisfy the waveequation, π = 0.

The correlation functions of the weights across sample space and layers, P σ = σ(z , t )σ(z, t) and P π = π(z , t )π(z, t) are the Green's functions of the respective equations of motion.

Fourier transforming the correlation functions give P σ,π (t, k) = i 2ω 0 exp − iω 0 t ,where ω 0 = |k| 2 + |m 2 σ,π |, and m Even though we formulated our field theory based on the decoupling limit of ResNets, the result of infinite correlation is very general and can be applied even if the decoupling limit is not valid.

It is a direct result of spontaneous symmetry breaking.

We state the Goldstone Theorem without proof.

Theorem (Goldstone): For every continuous symmetry that is spontaneously broken, a weight π with zero Hessian eigenvalue is generated at zero temperature (learning rate η).

In brief, the formalism for spontaneous symmetry breaking is mostly done in quantum field theory.

In terms of statistics, quantum mechanics is the study of errors.

We also believe that it is a good approximation to deep neural networks in the presence of the non-linear operators.

The non-linear operators quantizes the input.

Let R denotes the opertor corresponding to a sigmoid, say, then the output is R(W) {0, +1} for the most part.

And the negative end of ReLU is zero.

Let us take a step back and go through the logical steps to understand that a scalar quantum field theory is perhaps one of the simplest model one can consider to describe a neural network layer by layer, in the decoupling limit.

We wish to formulate a dynamical model to describe the weights layer by layer,

@highlight

Closed form results for deep learning in the layer decoupling limit applicable to Residual Networks