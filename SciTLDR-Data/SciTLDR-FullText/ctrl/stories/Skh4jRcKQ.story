Training activation quantized neural networks involves minimizing a piecewise constant training loss whose gradient vanishes almost everywhere, which is undesirable for the standard back-propagation or chain rule.

An empirical way around this issue is to use a straight-through estimator (STE) (Bengio et al., 2013) in the backward pass only, so that the "gradient" through the modified chain rule becomes non-trivial.

Since this unusual "gradient" is certainly not the gradient of loss function, the following question arises: why searching in its negative direction minimizes the training loss?

In this paper, we provide the theoretical justification of the concept of STE by answering this question.

We consider the problem of learning a two-linear-layer network with binarized ReLU activation and Gaussian input data.

We shall refer to the unusual "gradient" given by the STE-modifed chain rule as coarse gradient.

The choice of STE is not unique.

We prove that if the STE is properly chosen, the expected coarse gradient correlates positively with the population gradient (not available for the training), and its negation is a descent direction for minimizing the population loss.

We further show the associated coarse gradient descent algorithm converges to a critical point of the population loss minimization problem.

Moreover, we show that a poor choice of STE leads to instability of the training algorithm near certain local minima, which is verified with CIFAR-10 experiments.

Deep neural networks (DNN) have achieved the remarkable success in many machine learning applications such as computer vision (Krizhevsky et al., 2012; Ren et al., 2015) , natural language processing (Collobert & Weston, 2008) and reinforcement learning (Mnih et al., 2015; Silver et al., 2016) .

However, the deployment of DNN typically require hundreds of megabytes of memory storage for the trainable full-precision floating-point parameters, and billions of floating-point operations to make a single inference.

To achieve substantial memory savings and energy efficiency at inference time, many recent efforts have been made to the training of coarsely quantized DNN, meanwhile maintaining the performance of their float counterparts (Courbariaux et al., 2015; Rastegari et al., 2016; Cai et al., 2017; Hubara et al., 2018; Yin et al., 2018b) .Training fully quantized DNN amounts to solving a very challenging optimization problem.

It calls for minimizing a piecewise constant and highly nonconvex empirical risk function f (w) subject to a discrete set-constraint w ∈ Q that characterizes the quantized weights.

In particular, weight quantization of DNN have been extensively studied in the literature; see for examples (Li et al., 2016; Zhu et al., 2016; Li et al., 2017; Yin et al., 2016; 2018a; Hou & Kwok, 2018; He et al., 2018; Li & Hao, 2018) .

On the other hand, the gradient ∇f (w) in training activation quantized DNN is almost everywhere (a.e.) zero, which makes the standard back-propagation inapplicable.

The arguably most effective way around this issue is nothing but to construct a non-trivial search direction by properly modifying the chain rule.

Specifically, one can replace the a.e.

zero derivative of quantized activation function composited in the chain rule with a related surrogate.

This proxy derivative used in the backward pass only is referred as the straight-through estimator (STE) (Bengio et al., 2013) .

In the same paper, Bengio et al. (2013) proposed an alternative approach based on stochastic neurons.

In addition, Friesen & Domingos (2017) proposed the feasible target propagation algorithm for learning hard-threshold (or binary activated) networks (Lee et al., 2015) via convex combinatorial optimization.

The idea of STE originates to the celebrated perceptron algorithm (Rosenblatt, 1957; 1962) in 1950s for learning single-layer perceptrons.

The perceptron algorithm essentially does not calculate the "gradient" through the standard chain rule, but instead through a modified chain rule in which the derivative of identity function serves as the proxy of the original derivative of binary output function 1 {x>0} .

Its convergence has been extensive discussed in the literature; see for examples, (Widrow & Lehr, 1990; Freund & Schapire, 1999) and the references therein.

Hinton (2012) extended this idea to train multi-layer networks with binary activations (a.k.a.

binary neuron), namely, to backpropagate as if the activation had been the identity function.

Bengio et al. (2013) proposed a STE variant which uses the derivative of the sigmoid function instead.

In the training of DNN with weights and activations constrained to ±1, (Hubara et al., 2016) substituted the derivative of the signum activation function with 1 {|x|≤1} in the backward pass, known as the saturated STE.

Later the idea of STE was readily employed to the training of DNN with general quantized ReLU activations (Hubara et al., 2018; Zhou et al., 2016; Cai et al., 2017; Choi et al., 2018; Yin et al., 2018b) , where some other proxies took place including the derivatives of vanilla ReLU and clipped ReLU.

Despite all the empirical success of STE, there is very limited theoretical understanding of it in training DNN with stair-case activations.

Goel et al. (2018) considers leaky ReLU activation of a one-hidden-layer network.

They showed the convergence of the so-called Convertron algorithm, which uses the identity STE in the backward pass through the leaky ReLU layer.

Other similar scenarios, where certain layers are not desirable for back-propagation, have been brought up recently by (Wang et al., 2018) and (Athalye et al., 2018) .

The former proposed an implicit weighted nonlocal Laplacian layer as the classifier to improve the generalization accuracy of DNN.

In the backward pass, the derivative of a pre-trained fullyconnected layer was used as a surrogate.

To circumvent adversarial defense (Szegedy et al., 2013) , (Athalye et al., 2018) introduced the backward pass differentiable approximation, which shares the same spirit as STE, and successfully broke defenses at ICLR 2018 that rely on obfuscated gradients.

Throughout this paper, we shall refer to the "gradient" of loss function w.r.t.

the weight variables through the STE-modified chain rule as coarse gradient.

Since the backward and forward passes do not match, the coarse gradient is certainly not the gradient of loss function, and it is generally not the gradient of any function.

Why searching in its negative direction minimizes the training loss, as this is not the standard gradient descent algorithm?

Apparently, the choice of STE is non-unique, then what makes a good STE?

From the optimization perspective, we take a step towards understanding STE in training quantized ReLU nets by attempting these questions.

On the theoretical side, we consider three representative STEs for learning a two-linear-layer network with binary activation and Gaussian data: the derivatives of the identity function (Rosenblatt, 1957; Hinton, 2012; Goel et al., 2018) , vanilla ReLU and the clipped ReLUs (Cai et al., 2017; Hubara et al., 2016) .

We adopt the model of population loss minimization (Brutzkus & Globerson, 2017; Tian, 2017; Li & Yuan, 2017; Du et al., 2018) .

For the first time, we prove that proper choices of STE give rise to training algorithms that are descent.

Specifically, the negative expected coarse gradients based on STEs of the vanilla and clipped ReLUs are provably descent directions for the minimizing the population loss, which yield monotonically decreasing energy in the training.

In contrast, this is not true for the identity STE.

We further prove that the corresponding training algorithm can be unstable near certain local minima, because the coarse gradient may simply not vanish there.

Complementary to the analysis, we examine the empirical performances of the three STEs on MNIST and CIFAR-10 classifications with general quantized ReLU.

While both vanilla and clipped ReLUs work very well on the relatively shallow LeNet-5, clipped ReLU STE is arguably the best for the deeper VGG-11 and ResNet-20.

In our CIFAR experiments in section 4.2, we observe that the training using identity or ReLU STE can be unstable at good minima and repelled to an inferior one with substantially higher training loss and decreased generalization accuracy.

This is an implication that poor STEs generate coarse gradients incompatible with the energy landscape, which is consistent with our theoretical finding about the identity STE.To our knowledge, convergence guarantees of perceptron algorithm (Rosenblatt, 1957; 1962) and Convertron algorithm (Goel et al., 2018) were proved for the identity STE.

It is worth noting that Convertron (Goel et al., 2018) makes weaker assumptions than in this paper.

These results, however, do not generalize to the network with two trainable layers studied here.

As aforementioned, the identity STE is actually a poor choice in our case.

Moreover, it is not clear if their analyses can be extended to other STEs.

Similar to Convertron with leaky ReLU, the monotonicity of quantized activation function plays a role in coarse gradient descent.

Indeed, all three STEs considered here exploit this property.

But this is not the whole story.

A great STE like the clipped ReLU matches quantized ReLU at the extrema, otherwise the instability/incompatibility issue may arise.

Organization.

In section 2, we study the energy landscape of a two-linear-layer network with binary activation and Gaussian data.

We present the main results and sketch the mathematical analysis for STE in section 3.

In section 4, we compare the empirical performances of different STEs in 2-bit and 4-bit activation quantization, and report the instability phenomena of the training algorithms associated with poor STEs observed in CIFAR experiments.

Due to space limitation, all the technical proofs as well as some figures are deferred to the appendix.

Notations.

· denotes the Euclidean norm of a vector or the spectral norm of a matrix.

0 n ∈ R n represents the vector of all zeros, whereas 1 n ∈ R n the vector of all ones.

I n is the identity matrix of order n.

For any w, z ∈ R n , w z = w, z = i w i z i is their inner product.

w z denotes the Hadamard product whose i th entry is given by (w z) i = w i z i .

We consider a model similar to (Du et al., 2018) that outputs the prediction DISPLAYFORM0 for some input Z ∈ R m×n .

Here w ∈ R n and v ∈ R m are the trainable weights in the first and second linear layer, respectively; Z i denotes the ith row vector of Z; the activation function σ acts component-wise on the vector Zw, i.e., σ(Zw) i = σ((Zw) i ) = σ(Z i w).

The first layer serves as a convolutional layer, where each row Z i can be viewed as a patch sampled from Z and the weight filter w is shared among all patches, and the second linear layer is the classifier.

The label is generated according to y * (Z) = (v * ) σ(Zw * ) for some true (non-zero) parameters v * and w * .

Moreover, we use the following squared sample loss DISPLAYFORM1 Unlike in (Du et al., 2018) , the activation function σ here is not ReLU, but the binary function σ(x) = 1 {x>0} .We assume that the entries of Z ∈ R m×n are i.i.d.

sampled from the Gaussian distribution N (0, 1) (Zhong et al., 2017; Brutzkus & Globerson, 2017) .

Since (v, w; Z) = (v, w/c; Z) for any scalar c > 0, without loss of generality, we take w * = 1 and cast the learning task as the following population loss minimization problem: DISPLAYFORM2 where the sample loss (v, w; Z) is given by (1).

With the Gaussian assumption on Z, as will be shown in section 2.2, it is possible to find the analytic expressions of f (v, w) and its gradient DISPLAYFORM0 .The gradient of objective function, however, is not available for the network training.

In fact, we can only access the expected sample gradient, namely, DISPLAYFORM1 .

By the standard back-propagation or chain rule, we readily check that DISPLAYFORM2 and DISPLAYFORM3 Note that σ is zero a.e.

, which makes (4) inapplicable to the training.

The idea of STE is to simply replace the a.e.

zero component σ in (4) with a related non-trivial function µ (Hinton, 2012; Bengio et al., 2013; Hubara et al., 2016; Cai et al., 2017) , which is the derivative of some (sub)differentiable function µ. More precisely, back-propagation using the STE µ gives the following non-trivial surrogate of ∂ ∂w (v, w; Z), to which we refer as the coarse (partial) gradient DISPLAYFORM4 Using the STE µ to train the two-linear-layer convolutional neural network (CNN) with binary activation gives rise to the (full-batch) coarse gradient descent described in Algorithm 1.Algorithm 1 Coarse gradient descent for learning two-linear-layer CNN with STE µ .

DISPLAYFORM5

Let us present some preliminaries about the landscape of the population loss function f (v, w).To this end, we define the angle between w and w * as θ(w, w * ) := arccos w w * w w * for any w = 0 n .

Recall that the label is given by y * (Z) = (v * ) Zw * from (1), we elaborate on the analytic expressions of f (v, w) and ∇f (v, w).

Lemma 1.

If w = 0 n , the population loss f (v, w) is given by DISPLAYFORM0 Lemma 2.

If w = 0 n and θ(w, w * ) ∈ (0, π), the partial gradients of f (v, w) w.r.t.

v and w are DISPLAYFORM1 respectively.

For any v ∈ R m , (v, 0 m ) is impossible to be a local minimizer.

The only possible (local) minimizers of the model (2) are located at 1.

Stationary points where the gradients given by (6) and (7) vanish simultaneously (which may not be possible), i.e., DISPLAYFORM2 2.

Non-differentiable points where θ(w, w * ) = 0 and v = v * , or θ(w, w * ) = π and v = DISPLAYFORM3 are obviously the global minimizers of (2).

We show that the stationary points, if exist, can only be saddle points, and DISPLAYFORM4 are the only potential spurious local minimizers.

DISPLAYFORM5 give the saddle points obeying (8), and DISPLAYFORM6 are the spurious local minimizers.

Otherwise, the model (2) has no saddle points or spurious local minimizers.

We further prove that the population gradient ∇f (v, w) given by (6) and (7), is Lipschitz continuous when restricted to bounded domains.

Lemma 3.

For any differentiable points (v, w) and (ṽ,w) with min{ w , w } = c w > 0 and max{ v , ṽ } = C v , there exists a Lipschitz constant L > 0 depending on C v and c w , such that DISPLAYFORM7

We are most interested in the complex case where both the saddle points and spurious local minimizers are present.

Our main results are concerned with the behaviors of the coarse gradient descent summarized in Algorithm 1 when the derivatives of the vanilla and clipped ReLUs as well as the identity function serve as the STE, respectively.

We shall prove that Algorithm 1 using the derivative of vanilla or clipped ReLU converges to a critical point, whereas that with the identity STE does not.

Theorem 1 (Convergence).

Let {(v t , w t )} be the sequence generated by Algorithm 1 with ReLU µ(x) = max{x, 0} or clipped ReLU µ(x) = min {max{x, 0}, 1}. Suppose w t ≥ c w for all t with some c w > 0.

Then if the learning rate η > 0 is sufficiently small, for any initialization (v 0 , w 0 ), the objective sequence {f (v t , w t )} is monotonically decreasing, and {(v t , w t )} converges to a saddle point or a (local) minimizer of the population loss minimization (2).

In addition, if 1 m v * = 0 and m > 1, the descent and convergence properties do not hold for Algorithm 1 with the identity function µ(x) = x near the local minimizers satisfying θ(w, w * ) = π and DISPLAYFORM0 The convergence guarantee for the coarse gradient descent is established under the assumption that there are infinite training samples.

When there are only a few data, in a coarse scale, the empirical loss roughly descends along the direction of negative coarse gradient, as illustrated by Figure 1 .

As the sample size increases, the empirical loss gains monotonicity and smoothness.

This explains why (proper) STE works so well with massive amounts of data as in deep learning.

Remark 2.

The same results hold, if the Gaussian assumption on the input data is weakened to that their rows i.i.d. follow some rotation-invariant distribution.

The proof will be substantially similar.

In the rest of this section, we sketch the mathematical analysis for the main results.sample size = 10 sample size = 50 sample size = 1000Figure 1: The plots of the empirical loss moving by one step in the direction of negative coarse gradient v.s. the learning rate (step size) η for different sample sizes.

If we choose the derivative of ReLU µ(x) = max{x, 0} as the STE in FORMULA7 , it is easy to see µ (x) = σ(x), and we have the following expressions of DISPLAYFORM0 Let µ(x) = max{x, 0} in (5).

The expected coarse gradient w.r.t.

w is DISPLAYFORM1 where DISPLAYFORM2 As stated in Lemma 5 below, the key observation is that the coarse partial gradient E Z g relu (v, w; Z) has non-negative correlation with the population partial gradient ∂f ∂w (v, w), and −E Z g relu (v, w; Z) together with −E Z ∂ ∂v (v, w; Z) form a descent direction for minimizing the population loss.

Lemma 5.

If w = 0 n and θ(w, w * ) ∈ (0, π), then the inner product between the expected coarse and population gradients w.r.t.

w is DISPLAYFORM3 Moreover, if further v ≤ C v and w ≥ c w , there exists a constant A relu > 0 depending on C v and c w , such that DISPLAYFORM4 Clearly, when DISPLAYFORM5 We redefine the second term as 0n in the case θ(w, w * ) = π, or equivalently, DISPLAYFORM6 that the coarse gradient descent behaves like the gradient descent directly on f (v, w).

Here we would like to highlight the significance of the estimate (12) in guaranteeing the descent property of Algorithm 1.

By the Lipschitz continuity of ∇f specified in Lemma 3, it holds that DISPLAYFORM7 where a) is due to (12).

Therefore, if η is small enough, we have monotonically decreasing energy until convergence.

Lemma 6.

When Algorithm 1 converges, E Z ∂ ∂v (v, w; Z) and E Z g relu (v, w; Z) vanish simultaneously, which only occurs at the 1.

Saddle points where (8) is satisfied according to Proposition 1.

Lemma 6 states that when Algorithm 1 using ReLU STE converges, it can only converge to a critical point of the population loss function.

For the STE using clipped ReLU, µ(x) = min {max{x, 0}, 1} and µ (x) = 1 {0<x<1} (x).

We have results similar to Lemmas 5 and 6.

That is, the coarse partial gradient using clipped ReLU STE E Z g crelu (v, w; Z) generally has positive correlation with the true partial gradient of the population loss ∂f ∂w (v, w) (Lemma 7)).

Moreover, the coarse gradient vanishes and only vanishes at the critical points (Lemma 8).

Lemma 7.

If w = 0 n and θ(w, w * ) ∈ (0, π), then DISPLAYFORM0 * same as in Lemma 5, and DISPLAYFORM1 2 )dr.

The inner product between the expected coarse and true gradients w.r.t.

w DISPLAYFORM2 Moreover, if further v ≤ C v and w ≥ c w , there exists a constant A crelu > 0 depending on C v and c w , such that DISPLAYFORM3 Lemma 8.

When Algorithm 1 converges, E Z ∂ ∂v (v, w; Z) and E Z g crelu (v, w; Z) vanish simultaneously, which only occurs at the 1.

Saddle points where (8) is satisfied according to Proposition 1.

Now we consider the derivative of identity function.

Similar results to Lemmas 5 and 6 are not valid anymore.

It happens that the coarse gradient derived from the identity STE does not vanish at local minima, and Algorithm 1 may never converge there.

Lemma 9.

Let µ(x) = x in (5).

Then the expected coarse partial gradient w.r.t.

w is DISPLAYFORM0 If θ(w, w * ) = π and DISPLAYFORM1 i.e., E Z g id (v, w; Z) does not vanish at the local minimizers if 1 m v * = 0 and m > 1.Lemma 10.

If w = 0 n and θ(w, w * ) ∈ (0, π), then the inner product between the expected coarse and true gradients w.r.t.

w is DISPLAYFORM2 Lemma 9 suggests that if 1 m v * = 0, the coarse gradient descent will never converge near the spurious minimizers with θ(w, w * ) = π and DISPLAYFORM3 does not vanish there.

By the positive correlation implied by (15) of Lemma 10, for some proper (v 0 , w 0 ), the iterates {(v t , w t )} may move towards a local minimizer in the beginning.

But when {(v t , w t )} approaches it, the descent property (13) does not hold for E Z [g id (v, w; Z)] because of (16), hence the training loss begins to increase and instability arises.

While our theory implies that both vanilla and clipped ReLUs learn a two-linear-layer CNN, their empirical performances on deeper nets are different.

In this section, we compare the performances of the identity, ReLU and clipped ReLU STEs on MNIST (LeCun et al., 1998) and CIFAR-10 (Krizhevsky, 2009) benchmarks for 2-bit or 4-bit quantized activations.

As an illustration, we plot the 2-bit quantized ReLU and its associated clipped ReLU in Figure 3 in the appendix.

Intuitively, the clipped ReLU should be the best performer, as it best approximates the original quantized ReLU.

We also report the instability issue of the training algorithm when using an improper STE in section 4.2.

In all experiments, the weights are kept float.

The resolution α for the quantized ReLU needs to be carefully chosen to maintain the full-precision level accuracy.

To this end, we follow (Cai et al., 2017 ) and resort to a modified batch normalization layer (Ioffe & Szegedy, 2015) without the scale and shift, whose output components approximately follow a unit Gaussian distribution.

Then the α that fits the input of activation layer the best can be pre-computed by a variant of Lloyd's algorithm (Lloyd, 1982; Yin et al., 2018a) applied to a set of simulated 1-D half-Gaussian data.

After determining the α, it will be fixed during the whole training process.

Since the original LeNet-5 does not have batch normalization, we add one prior to each activation layer.

We emphasize that we are not claiming the superiority of the quantization approach used here, as it is nothing but the HWGQ (Cai et al., 2017), except we consider the uniform quantization.

The optimizer we use is the stochastic (coarse) gradient descent with momentum = 0.9 for all experiments.

We train 50 epochs for LeNet-5 (LeCun et al., 1998) on MNIST, and 200 epochs for VGG-11 (Simonyan & Zisserman, 2014) and ResNet-20 (He et al., 2016) on CIFAR-10.

The parameters/weights are initialized with those from their pre-trained full-precision counterparts.

The schedule of the learning rate is specified in TAB2 in the appendix.

The experimental results are summarized in Table 1 , where we record both the training losses and validation accuracies.

Among the three STEs, the derivative of clipped ReLU gives the best overall performance, followed by vanilla ReLU and then by the identity function.

For deeper networks, clipped ReLU is the best performer.

But on the relatively shallow LeNet-5 network, vanilla ReLU exhibits comparable performance to the clipped ReLU, which is somewhat in line with our theoretical finding that ReLU is a great STE for learning the two-linear-layer (shallow) CNN.

We report the phenomenon of being repelled from a good minimum on ResNet-20 with 4-bit activations when using the identity STE, to demonstrate the instability issue as predicted in Theorem 1.

By Table 1 , the coarse gradient descent algorithms using the vanilla and clipped ReLUs converge to the neighborhoods of the minima with validation accuracies (training losses) of 86.59% (0.25) and 91.24% (0.04), respectively, whereas that using the identity STE gives 54.16% (1.38).

Note that the landscape of the empirical loss function does not depend on which STE is used in the training.

Then we initialize training with the two improved minima and use the identity STE.

To see if the algorithm is stable there, we start the training with a tiny learning rate of 10 −5 .

For both initializations, the training loss and validation error significantly increase within the first 20 epochs; see Figure 4 .2.

To speedup training, at epoch 20, we switch to the normal schedule of learning rate specified in TAB2 and run 200 additional epochs.

The training using the identity STE ends up with a much worse minimum.

This is because the coarse gradient with identity STE does not vanish at the good minima in this case (Lemma 9).

Similarly, the poor performance of ReLU STE on 2-bit activated ResNet-20 is also due to the instability of the corresponding training algorithm at good minima, as illustrated by Figure 4 in Appendix C, although it diverges much slower.

Figure 2: When initialized with weights (good minima) produced by the vanilla (orange) and clipped (blue) ReLUs on ResNet-20 with 4-bit activations, the coarse gradient descent using the identity STE ends up being repelled from there.

The learning rate is set to 10 −5 until epoch 20.

We provided the first theoretical justification for the concept of STE that it gives rise to descent training algorithm.

We considered three STEs: the derivatives of the identity function, vanilla ReLU and clipped ReLU, for learning a two-linear-layer CNN with binary activation.

We derived the explicit formulas of the expected coarse gradients corresponding to the STEs, and showed that the negative expected coarse gradients based on vanilla and clipped ReLUs are descent directions for minimizing the population loss, whereas the identity STE is not since it generates a coarse gradient incompatible with the energy landscape.

The instability/incompatibility issue was confirmed in CIFAR experiments for improper choices of STE.

In the future work, we aim further understanding of coarse gradient descent for large-scale optimization problems with intractable gradients.

Figure 4 : When initialized with the weights produced by the clipped ReLU STE on ResNet-20 with 2-bit activations (88.38% validation accuracy), the coarse gradient descent using the ReLU STE with 10 −5 learning rate is not stable there, and both classification and training errors begin to increase.

Lemma 11.

Let z ∈ R n be a Gaussian random vector with entries i.i.d.

sampled from N (0, 1).

Given nonzero vectors w,w ∈ R n with the angle θ, we have DISPLAYFORM0 Proof of Lemma 11.

The third identity was proved in Lemma A.1 of (Du et al., 2018) .

To show the first one, without loss of generality we assume w = [w 1 , 0 n−1 ] with w 1 > 0, then E 1 {z w>0} = P(z 1 > 0) = 1 2 .

We further assumew = [w 1 ,w 2 , 0 n−2 ] .

It is easy to see that DISPLAYFORM1 To prove the last identity, we use polar representation of two-dimensional Gaussian random variables, where r is the radius and φ is the angle with dP r = r exp(−r 2 /2)dr and dP φ = 1 2π dφ.

Then E z i 1 {z w>0, z w>0} = 0 for i ≥ 3.

Moreover, DISPLAYFORM2 Therefore, Lemma 12.

Let z ∈ R n be a Gaussian random vector with entries i.i.d.

sampled from N (0, 1).

Given nonzero vectors w,w ∈ R n with the angle θ, we have Moreover, E z i 1 {0<z w<1} = 0 for i ≥ 3.

So the first identity holds.

For the second one, we have DISPLAYFORM3 DISPLAYFORM4 DISPLAYFORM5 and similarly, E z 2 1 {0<z w<1,z w>0} = q(θ, w).

Therefore, DISPLAYFORM6 DISPLAYFORM7 Proof of Lemma 13.

DISPLAYFORM8 where the last inequality is due to the rearrangement inequality since both sin(φ) and ξ DISPLAYFORM9 2.

Since cos(φ)ξ DISPLAYFORM10 is even, we have DISPLAYFORM11 The first inequality is due to part 1 which gives p(π/2, w) ≤ q(π/2, w), whereas the second one holds because sin(φ)ξ Proof of Lemma 14.

1.

Since by Cauchy-Schwarz inequality, DISPLAYFORM12 we have DISPLAYFORM13 Therefore, DISPLAYFORM14 where we used the fact sin(x) ≥ 2.

Since I n − ww w 2 w * is the projection of w * onto the complement space of w, and likewise for I n −ww w 2 w * , the angle between I n − ww w 2 w * and I n −ww w 2 w * is equal to the angle between w andw.

Therefore, DISPLAYFORM15

Lemma 1.

If w = 0 n , the population loss f (v, w) is given by DISPLAYFORM0 Proof of Lemma 1.

We notice that DISPLAYFORM1 Let Z i be the i th row vector of Z. Since w = 0 n , using Lemma 11, we have DISPLAYFORM2 and for i = j, DISPLAYFORM3 , DISPLAYFORM4 Then it is easy to validate the first claim.

Moreover, if w = 0 n , then DISPLAYFORM5 Lemma 2.

If w = 0 n and θ(w, w * ) ∈ (0, π), the partial gradients of f (v, w) w.r.t.

v and w are DISPLAYFORM6 Proof of Lemma 2.

The first claim is trivial, and we only show the second one.

Since θ(w, w DISPLAYFORM7 give the saddle points obeying FORMULA11 DISPLAYFORM8 From FORMULA1 it follows that DISPLAYFORM9 On the other hand, from (18) it also follows that DISPLAYFORM10 where we used (I m + 1 m 1 m )1 m = (m + 1)1 m .

Taking the difference of the two equalities above gives DISPLAYFORM11 By (19), we have θ(w, w DISPLAYFORM12 Furthermore, since ∂f ∂v (v, w) = 0, we have DISPLAYFORM13 Next, we check the local optimality of the stationary points.

By ignoring the scaling and constant terms, we rewrite the objective function as DISPLAYFORM14 It is easy to check that its Hessian matrix DISPLAYFORM15 is indefinite.

Therefore, the stationary points are saddle points.

DISPLAYFORM16 where we used (20) in the last identity.

We consider an arbitrary point (v + ∆v, π + ∆θ) in the neighborhood of (v, π) with ∆θ ≤ 0.

The perturbed objective value is DISPLAYFORM17 On the right hand side, since v = (I m + 1 m 1 m ) −1 (1 m 1 m − I m )v * is the unique minimizer to the quadratic functionf (v, π), we have if ∆v = 0 m , DISPLAYFORM18 Moreover, for sufficiently small ∆v , it holds that ∆θ · (v + ∆v) v * > 0 for ∆θ < 0 because of (21).

Therefore,f (v + ∆v, π + ∆θ) >f (v, π) whenever (∆v, ∆θ) is small and non-zero, and DISPLAYFORM19 To prove the second claim, suppose Lemma 3.

For any differentiable points (v, w) and (ṽ,w) with min{ w , w } = c w > 0 and max{ v , ṽ } = C v , there exists a Lipschitz constant L > 0 depending on C v and c w , such that DISPLAYFORM20 DISPLAYFORM21 DISPLAYFORM22

where the last inequality is due to Lemma 14.1.

DISPLAYFORM0 where the second last inequality is to due to Lemma 14.2.

Combining the two inequalities above validates the claim.

Lemma 4.

The expected partial gradient of (v, w; Z) w.r.t.

v is DISPLAYFORM1 Let µ(x) = max{x, 0} in (5).

The expected coarse gradient w.r.t.

w is DISPLAYFORM2 Proof of Lemma 4.

The first claim is true because DISPLAYFORM3 Using the fact that µ = σ = 1 {x>0} , we have DISPLAYFORM4 Invoking Lemma 11, we have DISPLAYFORM5 and DISPLAYFORM6 3 We redefine the second term as 0n in the case θ(w, w * ) = π, or equivalently, DISPLAYFORM7 Therefore, DISPLAYFORM8 and the result follows.

Lemma 5.

If w = 0 n and θ(w, w * ) ∈ (0, π), then the inner product between the expected coarse and true gradients w.r.t.

w is DISPLAYFORM9 Moreover, if further v ≤ C v and w ≥ c w , there exists a constant A relu > 0 depending on C v and c w , such that DISPLAYFORM10 Proof of Lemma 5.

By Lemmas 2 and 4, we have DISPLAYFORM11 Notice that I n − ww w 2 w = 0 n and w * = 1, if θ(w, w * ) = 0, π, then we have DISPLAYFORM12 To show the second claim, without loss of generality, we assume w = 1.

Denote θ := θ(w, w * ).

By Lemma 1, we have DISPLAYFORM13 By Lemma 4, DISPLAYFORM14 where DISPLAYFORM15 and by the first claim, DISPLAYFORM16 Hence, for some A relu depending only on C v and c w , we have DISPLAYFORM17 Saddle points where (8) is satisfied according to Proposition 1.

Proof of Lemma 6.

By Lemma 4, suppose we have DISPLAYFORM0 and DISPLAYFORM1 where DISPLAYFORM2 If θ(w, w * ) = 0, then by (24), v = v * , and FORMULA2 If v v * = 0, then by (24), we have the expressions for v and θ(w, w * ) from Proposition 1, and (25) is satisfied.

Lemma 7.

If w = 0 n and θ(w, w * ) ∈ (0, π), then DISPLAYFORM3 where DISPLAYFORM4 * same as in Lemma 5, and DISPLAYFORM5 2 )dr.

The inner product between the expected coarse and true gradients w.r.t.

w DISPLAYFORM6 Moreover, if further v ≤ C v and w ≥ c w , there exists a constant A crelu > 0 depending on C v and c w , such that DISPLAYFORM7 Proof of Lemma 7.

Denote θ := θ(w, w * ).

We first compute E Z g crelu (v, w; Z) .

By (5), DISPLAYFORM8 Since µ = 1 {0<x<1} and σ = 1 {x>0} , we have In the last equality above, we called Lemma 12.

DISPLAYFORM9 Notice that I n − ww w 2 w = 0 n and w * = 1.

If θ(w, w * ) = 0, π, then the inner product between E Z g crelu (v, w; Z) and Combining the above estimate together with FORMULA2 , FORMULA2 and FORMULA2 , and using Cauchy-Schwarz inequality, we have DISPLAYFORM10 where p(0, w) and q(θ, w) are uniformly bounded.

This completes the proof.

Proof of Lemma 8.

The proof of Lemma 8 is similar to that of Lemma 6, and we omit it here.

The core part is that q(θ, w) defined in Lemma 12 is non-negative and equals 0 only at θ = 0, π, as well as p(0, w) ≥ p(θ, w) ≥ p(π, w) = 0.Lemma 9.

Let µ(x) = x in (5).

Then the expected coarse partial gradient w.r.t.

w is Proof of Lemma 9.

By (5), DISPLAYFORM11 Using the facts that µ = 1 and σ = 1 {x>0} , we have DISPLAYFORM12 In the last equality above, we called the third identity In the third equality, we used the identity (I m + 1 m 1 m )1 m = (m + 1)1 m twice.

Lemma 10.

If w = 0 n and θ(w, w * ) ∈ (0, π), then the inner product between the expected coarse and true gradients w.r.t.

w is

<|TLDR|>

@highlight

We make theoretical justification for the concept of straight-through estimator.