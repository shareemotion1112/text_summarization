The novel \emph{Unbiased Online Recurrent Optimization} (UORO) algorithm allows for online learning of general recurrent computational graphs such as recurrent network models.

It works in a streaming fashion and avoids backtracking through past activations and inputs.

UORO is computationally as costly as \emph{Truncated Backpropagation Through Time} (truncated BPTT), a widespread algorithm for online learning of recurrent networks \cite{jaeger2002tutorial}.  UORO is a modification of \emph{NoBackTrack} \cite{DBLP:journals/corr/OllivierC15} that bypasses the need for model sparsity and makes implementation easy in current deep learning frameworks, even for complex models.

Like NoBackTrack, UORO provides unbiased gradient estimates; unbiasedness is the core hypothesis in stochastic gradient descent theory, without which convergence to a local optimum is not guaranteed.

On the contrary, truncated BPTT does not provide this property, leading to possible divergence.

On synthetic tasks where truncated BPTT is shown to diverge, UORO converges.

For instance, when a parameter has a positive short-term but negative long-term influence, truncated BPTT diverges unless the truncation span is very significantly longer than the intrinsic temporal range of the interactions, while UORO performs well thanks to the unbiasedness of its gradients.

Current recurrent network learning algorithms are ill-suited to online learning via a single pass through long sequences of temporal data.

Backpropagation Through Time (BPTT Jaeger (2002) ), the current standard for training recurrent architectures, is well suited to many short training sequences.

Treating long sequences with BPTT requires either storing all past inputs in memory and waiting for a long time between each learning step, or arbitrarily splitting the input sequence into smaller sequences, and applying BPTT to each of those short sequences, at the cost of losing long term dependencies.

This paper introduces Unbiased Online Recurrent Optimization (UORO), an online and memoryless learning algorithm for recurrent architectures: UORO processes and learns from data samples sequentially, one sample at a time.

Contrary to BPTT, UORO does not maintain a history of previous inputs and activations.

Moreover, UORO is scalable: processing data samples with UORO comes at a similar computational and memory cost as just running the recurrent model on those data.

Like most neural network training algorithms, UORO relies on stochastic gradient optimization.

The theory of stochastic gradient crucially relies on the unbiasedness of gradient estimates to provide convergence to a local optimum.

To this end, in the footsteps of NoBackTrack (NBT) BID11 , UORO provides provably unbiased gradient estimates, in a scalable, streaming fashion.

Unlike NBT, though, UORO can be easily implemented in a black-box fashion on top of an existing recurrent model in current machine learning software, without delving into the structure and code of the model.

The framework for recurrent optimization and UORO is introduced in Section 2.

The final algorithm is reasonably simple (Alg.

1), but its derivation (Section 3) is more complex.

In Section 6, UORO is shown to provide convergence on a set of synthetic experiments where truncated BPTT fails to display reliable convergence.

An implementation of UORO is provided as supplementary material.

A widespread approach to online learning of recurrent neural networks is Truncated Backpropagation Through Time (truncated BPTT) BID5 , which mimics Backpropagation Through Time, but zeroes gradient flows after a fixed number of timesteps.

This truncation makes gradient estimates biased; consequently, truncated BPTT does not provide any convergence guarantee.

Learning is biased towards short-time dependencies.1 .

Storage of some past inputs and states is required.

Online, exact gradient computation methods have long been known (Real Time Recurrent Learning (RTRL) BID15 BID12 ), but their computational cost discards them for reasonably-sized networks.

NoBackTrack (NBT) BID11 also provides unbiased gradient estimates for recurrent neural networks.

However, contrary to UORO, NBT cannot be applied in a blackbox fashion, making it extremely tedious to implement for complex architectures.

Other previous attempts to introduce generic online learning algorithms with a reasonable computational cost all result in biased gradient estimates.

Echo State Networks (ESNs) Jaeger FORMULA0 ; BID6 simply set to 0 the gradients of recurrent parameters.

Others, e.g., BID8 BID14 , introduce approaches resembling ESNs, but keep a partial estimate of the recurrent gradients.

The original Long Short Term Memory algorithm BID3 (LSTM now refers to a particular architecture) cuts gradient flows going out of gating units to make gradient computation tractable.

Decoupled Neural Interfaces BID4 bootstrap truncated gradient estimates using synthetic gradients generated by feedforward neural networks.

The algorithm in BID10 provides zeroth-order estimates of recurrent gradients via diffusion networks; it could arguably be turned online by running randomized alternative trajectories.

Generally these approaches lack a strong theoretical backing, except arguably ESNs.

UORO is a learning algorithm for recurrent computational graphs.

Formally, the aim is to optimize θ, a parameter controlling the evolution of a dynamical system DISPLAYFORM0 in order to minimize a total loss L : DISPLAYFORM1 , where o * t is a target output at time t. For instance, a standard recurrent neural network, with hidden state s t (preactivation values) and output o t at time t, is described with the update equations F state (x t+1 , s t , θ) : DISPLAYFORM2 Optimization by gradient descent is standard for neural networks.

In the spirit of stochastic gradient descent, we can optimize the total loss L = 0≤t≤T t (o t , o * t ) one term at a time and update the parameter online at each time step via DISPLAYFORM3 where η t is a scalar learning rate at time t. (Other gradient-based optimizers can also be used, once ∂ t ∂θ is known.)

The focus is then to compute, or approximate, DISPLAYFORM4 ∂θ by unfolding the network through time, and backpropagating through the unfolded network, each timestep corresponding to a layer.

BPTT thus requires maintaining the full unfolded network, or, equivalently, the history of past inputs and activations.2 Truncated BPTT only unfolds the network for a fixed number of timesteps, reducing computational cost in online settings BID5 .

This comes at the cost of biased gradients, and can prevent convergence of the gradient descent even for large truncations, as clearly exemplified in FIG0 .

Unbiased Online Recurrent Optimization is built on top of a forward computation of the gradients, rather than backpropagation.

Forward gradient computation for neural networks (RTRL) is described in BID15 and we review it in Section 3.1.

The derivation of UORO follows in Section 3.2.

Implementation details are given in Section 3.3.

UORO's derivation is strongly connected to BID11 but differs in one critical aspect: the sparsity hypothesis made in the latter is relieved, resulting in reduced implementation complexity without any model restriction.

The proof of UORO's convergence to a local optimum can be found in BID9 .

Forward computation of the gradient for a recurrent model (RTRL) is directly obtained by applying the chain rule to both the loss function and the state equation (1) , as follows.

Direct differentiation and application of the chain rule to t+1 yields DISPLAYFORM0 Here, the term ∂s t /∂θ represents the effect on the state at time t of a change of parameter during the whole past trajectory.

This term can be computed inductively from time t to t + 1.

Intuitively, looking at the update equation (1) , there are two contributions to ∂s t+1 /∂θ:•

The direct effect of a change of θ on the computation of s t+1 , given s t .•

The past effect of θ on s t via the whole past trajectory.

With this in mind, differentiating (1) with respect to θ yields DISPLAYFORM1 This gives a way to compute the derivative of the instantaneous loss without storing past history: at each time step, update ∂s t /∂θ from ∂s t−1 /∂θ, then use this quantity to directly compute ∂ t+1 /∂θ.

This is how RTRL Williams & Zipser (1989) proceeds.

A huge disadvantage of RTRL is that ∂s t /∂θ is of size dim(state) × dim(params).

For instance, with a fully connected standard recurrent network with n units, ∂s t /∂θ scales as n 3 .

This makes RTRL impractical for reasonably sized networks.

UORO modifies RTRL by only maintaining a scalable, rank-one, provably unbiased approximation of ∂s t /∂θ, to reduce the memory and computational cost.

This approximation takes the forms t ⊗θ t , wheres t is a column vector of the same dimension as s t ,θ t is a row vector of the same dimension as θ , and ⊗ denotes the outer product.

The resulting quantity is thus a matrix of the same size as ∂s t /∂θ.

The memory cost of storings t andθ t scales as dim(state) + dim(params).

Thus UORO is as memory costly as simply running the network itself (which indeed requires to store the current state and parameters).

The following section details hows t andθ t are built to provide unbiasedness.

Given an unbiased estimation of ∂s t /∂θ, namely, a stochastic matrixG t such that EG t = ∂s t /∂θ, unbiased estimates of ∂ t+1 /∂θ and ∂s t+1 /∂θ can be derived by pluggingG t in (4) and (5).

Unbiasedness is preserved thanks to linearity of the mean, because both (4) and (5) are affine in ∂s t /∂θ.

Thus, assuming the existence of a rank-one unbiased approximationG t =s t ⊗θ t at time t, we can plug it in (5) to obtain an unbiased approximationĜ t+1 at time t + 1 DISPLAYFORM0 However, in general this is no longer rank-one.

To transformĜ t+1 intoG t+1 , a rank-one unbiased approximation, the following rank-one trick, introduced in Ollivier et al. FORMULA0 is used: Proposition 1.

Let A be a real matrix that decomposes as DISPLAYFORM1 Let ν be a vector of k independent random signs, and ρ a vector of k positive numbers.

Consider the rank-one matrixÃ DISPLAYFORM2 ThenÃ is an unbiased rank-one approximation of A: DISPLAYFORM3 The rank-one trick can be applied for any ρ.

The choice of ρ influences the variance of the approximation; choosing DISPLAYFORM4 minimizes the variance of the approximation, E A −Ã The UORO update is obtained by applying the rank-one trick twice to (6).

First, ∂Fstate ∂θ (x t+1 , s t , θ) is reduced to a rank one matrix, without variance minimization.3 Namely, let ν be a vector of independant random signs; then, DISPLAYFORM5 This results in a rank-two, unbiased estimate of ∂s t+1 /∂θ by substituting (10) into (6) DISPLAYFORM6 Applying Prop.

1 again to this rank-two estimate, with variance minimization, yields UORO's estimateG t+1 DISPLAYFORM7 which satisfies that E νGt+1 is equal to (6).

(By elementary algebra, some random signs that should appear in (12) cancel out.)

Here DISPLAYFORM8 minimizes variance of the second reduction.

The unbiased estimation (12) is rank-one and can be rewritten asG t+1 =s t+1 ⊗θ t+1 with the updates DISPLAYFORM9 3 Variance minimization is not used at this step, since computing DISPLAYFORM10 for every i is not scalable.

Initially, ∂s 0 /∂θ = 0, thuss 0 = 0,θ 0 = 0 yield an unbiased estimate at time 0.

Using this initial estimate and the update rules FORMULA5 - FORMULA6 , an estimate of ∂s t /∂θ is obtained at all subsequent times, allowing for online estimation of ∂ t /∂θ.

Thanks to the construction above, by induction all these estimates are unbiased.

4 We are left to demonstrate that these update rules are scalably implementable.

Implementing UORO requires maintaining the rank-one approximation and the corresponding gradient loss estimate.

UORO's estimate of the loss gradient ∂ t+1 /∂ θ at time t + 1 is expressed by plugging into (4) the rank-one approximation ∂s t /∂θ ≈s t ⊗θ t , which results in DISPLAYFORM0 , thus providing all necessary terms to compute (16).Updatings andθ requires applying FORMULA5 - FORMULA6 at each step.

Backpropagating the vector of random signs ν once through F state returns _, _, ν ∂F state (x t+1 , s t , θ)/∂θ , providing for (15).Updatings via (14) requires computing (∂F state /∂s t ) ·s t .

This is computable numerically through DISPLAYFORM1 computable through two applications of F state .

This operation is referred to as tangent forward propagation BID13 and can also often be computed algebraically.

This allows for complete implementation of one step of UORO (Alg.

1).

The cost of UORO (including running the model itself) is three applications of F state , one application of F out , one backpropagation through F out and F state , and a few elementwise operations on vectors and scalar products.

The resulting algorithm is detailed in Alg.

1.

F.forward(v) denotes pointwise application of F at point v, F.backprop(v, δo) backpropagation of row vector δo through F at point v, and F.forwarddiff(v, δv) tangent forward propagation of column vector δv through F at point v. Notably, F.backprop(v, δo) has the same dimension as v , e.g. F out .backprop((x t+1 , s t , θ), δo t+1 ) has three components, of the same dimensions as x t+1 , s t and θ .The proposed update rule for stochastic gradient descent (3) can be directly adapted to other optimizers, e.g. Adaptative Momentum (Adam) BID7 or Adaptative Gradient BID0 .

Vanilla stochastic gradient descent (SGD) and Adam are used hereafter.

In Alg.

1, such optimizers are denoted by SGDOpt and the corresponding parameter update given current parameter θ, gradient estimate g t and learning rate η t is denoted SGDOpt.update(g t , η t , θ).

The unbiased gradient estimates of UORO injects noise via ν, thus requiring smaller learning rates.

To reduce noise, UORO can be used on top of truncated BPTT so that recent gradients are computed exactly.

Formally, this just requires applying Algorithm 1 to a new transition function F T which is just T consecutive steps of the original model F .

Then the backpropagation operation in Algorithm 1 becomes a backpropagation over the last T steps, as in truncated BPTT.

The loss of one step of F T is the sum of the losses of the last T steps of F , namely T .

This way, we obtain an unbiased gradient estimate in which the gradients from the last T steps are computed exactly and incur no noise.

The resulting algorithm is referred to as memory-T UORO.

Its scaling in T is similar to T -truncated BPTT, both in Algorithm 1 -One step of UORO (from time t to t + 1)

-x t+1 , o * t+1 , s t and θ: input, target, previous recurrent state, and parameters -s t column vector of size state,θ t row vector of size params such that Es t ⊗θ t = ∂s t /∂θ -SGDOpt and η t+1 : stochastic optimizer and its learning rate Outputs: -t+1 , s t+1 and θ: loss, new recurrent state, and updated parameters -s t+1 andθ t+1 such that Es t+1 ⊗θ t+1 = ∂s t+1 /∂θ -g t+1 such that Eg t+1 = ∂ t+1 /∂θ /* compute next state and loss */ DISPLAYFORM0 /* compute gradient estimate */ (_, δs, δθ) ← F out .backprop (x t+1 , s t , θ), ∂ t+1 ∂o t+1 g t+1 ← (δs ·s t )θ t + δθ /* prepare for reduction */ Draw ν, column vector of random signs ±1 of size statẽ DISPLAYFORM1 terms of memory and computation.

In the experiments below, memory-T UORO reduced variance early on, but did not significantly impact later performance.

The noise in UORO can also be reduced by using higher-rank gradient estimates (rank-r instead of rank-1), which amounts to maintaining r distinct values ofs andθ in Algorithm 1 and averaging the resulting values ofg.

We did not exploit this possibility in the experiments below, although r = 2 visibly reduced variance in preliminary tests.

Gradient-based sequential learning on an unbounded data stream requires that the variance of the gradient estimate does not explode through time.

UORO is specifically built to provide an unbiased estimate whose variance does not explode over time.

A precise statement regarding UORO's convergence and boundedness of the variance of gradients is provided in BID9 .

Informally, when the largest eigenvalue of the differential transition operator ∂F state /∂s is uniformly bounded by a constant δ < 1 (which characterizes stable dynamical systems), the normalizing factors in FORMULA5 and FORMULA6 enforce that the influence of previous ν's decrease exponentially with time.

We hereby provide an experimental validation of the boundedness of UORO's variance in FIG2 .

To monitor the variance of UORO's estimate over time, a 64-unit GRU recurrent network is trained on the first 10 7 characters of the full works of Shakespeare using UORO.

The network is then rerun 100 times on the 10000 first characters of the text, and gradients estimates at each time steps are computed, but not applied.

The gradient relative variance, that is DISPLAYFORM0 is computed, where the average is taken with respect to runs.

This quantity appears to be stationary over time FIG2 .

As the number of hidden units in the recurrent network increases, the rank one approximation that is used to provide an unbiased gradient estimate becomes coarser.

Consequently, the relative variance, as defined in FORMULA22 , should increase as the number of hidden units increases.

This increase is experimentally verified in FIG2 .

Untrained GRU networks with various number of units are run for 10 timesteps, 100 times for each size, and the UORO gradient estimate after these 10 timesteps is computed (but not applied).

The relative variance of these gradients over the 100 runs is evaluated, for each network size.

As shown in the figure, the relative variance increases with the number of units.

Note the horizontal log scale.

The increase of the variance of the estimate with network size underlines the need for smaller learning rates when training large networks with UORO, compared to truncated backpropagation.

This can imply slower learning for the kind of dependencies that truncated backpropagation can learn.

The need for lower learning rates with larger networks is exemplified in FIG2 .

GRU networks of various hidden sizes are trained with UORO on a simple copy task, as presented in BID3 , with a lag of T = 5.

The networks are all trained with the same decreasing learning rate, η t = 10 −41+3·10 −3 t .

For all network sizes except the largest, the error decreases slowly but steadily.

For the largest network, the variance is too large compared to the learning rate, and the error jumps sharply midway through.

The set of experiments below aims at displaying specific cases where the biases from truncated BPTT are likely to prevent convergence of learning.

On this test set, UORO's unbiasedness provides steady convergence, highlighting the importance of unbiased estimates for general recurrent learning.

Influence balancing.

The first test case exemplifies learning of a scalar parameter θ which has a positive influence in the short term, but a negative one in the long run.

Short-sightedness of truncated algorithms results in abrupt failure, with the parameter exploding in the wrong direction, even with truncation lengths exceeding the temporal dependency range by a factor of 10 or so.

Consider the linear dynamics DISPLAYFORM0 with A a square matrix of size n with A i,i = 1/2, A i,i+1 = 1/2, and 0 elsewhere; θ ∈ R is a scalar parameter.

The second term has p positive-θ entries and n − p negative-θ entries.

Intuitively, the effect of θ on a unit diffuses to shallower units over time FIG6 .

Unit i only feels the effect of θ from unit i + n after n time steps, so the intrinsic time scale of the system is ≈ n. The loss considered is a target on the shallowest unit s 1 , DISPLAYFORM1 Learning is performed online with vanilla SGD, using gradient estimates either from UORO or Ttruncated BPTT with various T .

Learning rates are of the form DISPLAYFORM2 for suitable values of η.

As shown in FIG0 , UORO solves the problem while T -truncated BPTT fails to converge for any learning rate, even for truncations T largely above n. Failure is caused by ill balancing of time dependencies: the influence of θ on the loss is estimated with the wrong sign due to truncation.

For n = 23 units, with 13 minus signs, truncated BPTT requires a truncation T ≥ 200 to converge.

Next-character prediction.

The next experiment is character-level synthetic text prediction: the goal is to train a recurrent model to predict the t + 1-th character of a text given the first t online, with a single pass on the data sequence.

A single layer of 64 units, either GRU or LSTM, is used to output a probability vector for the next character.

The cross entropy criterion is used to compute the loss.

At each time t we plot the cumulated loss per character on the first t characters, 1 t t s=1 s .

(Losses for individual characters are quite noisy, as not all characters in the sequence are equally difficult to predict.)

This would be the compression rate in bits per character if the models were used as online compression algorithms on the first t characters.

In addition, in TAB1 we report a "recent" loss on the last 100, 000 characters, which is more representative of the model at the end of learning.

Optimization was performed using Adam with the default setting β 1 = 0.9 and β 2 = 0.999, and a decreasing learning rate η t = γ 1+α √ t , with t the number of characters processed.

As convergence of UORO requires smaller learning rates than truncated BPTT, this favors UORO.

Indeed UORO can fail to converge with non-decreasing learning rates, due to its stochastic nature.

DISTANT BRACKETS DATASET (s, k, a).

The distant brackets dataset is generated by repeatedly outputting a left bracket, generating s random characters from an alphabet of size a, outputting a right bracket, generating k random characters from the same alphabet, repeating the same first s characters between brackets and finally outputting a line break.

A sample is shown in FIG6 .UORO is compared to 4-truncated BPTT.

Truncation is deliberately shorter than the inherent time range of the data, to illustrate how bias can penalize learning if the inherent time range is unknown a priori.

The results are given in FIG0 (with learning rates using α = 0.015 and γ = 10 −3 ).

UORO beats 4-truncated BPTT in the long run, and succeeds in reaching near optimal behaviour both with GRUs and LSTMs.

Truncated BPTT remains stuck near a memoryless optimum with LSTMs; with GRUs it keeps learning, but at a slow rate.

Still, truncated BPTT displays faster early convergence.

a n b n (k, l) DATASET.

The a n b n (k, l) dataset tests memory and counting BID1 ; it is generated by repeatedly picking a random number n between k and l, outputting a string of n a's, a line break, n b's, and a line break (see FIG6 ).

The difficulty lies in matching the number of a's and b's.

Plots for a few setups are given in Fig. 4 .

The learning rates used α = 0.03 and γ = 10 −3 .

Numerical results at the end of training are given in TAB1 .

For reference, the true entropy rate is 0.14 bpc, while the entropy rate of a model that does not understand that the numbers of a's and b's coincide is double, 0.28 bpc.

Here, in every setup, UORO reliably converges and reaches near optimal performance.

Increasing UORO's range does not significantly improve results: providing an unbiased estimate is enough to provide reliable convergence in this case.

Meanwhile, truncated BPTT performs inconsistently.

Notably, with GRUs, it either converges to a poor local optimum corresponding to no understanding of the temporal structure, or exhibits gradient reascent in the long run.

Remarkably, with LSTMs rather than GRUs, 16-truncated BPTT reliably reaches optimal behavior on this problem even with biased gradient estimates.

We introduced UORO, an algorithm for training recurrent neural networks in a streaming, memoryless fashion.

UORO is easy to implement, and requires as little computation time as truncated BPTT, at the cost of noise injection.

Importantly, contrary to most other approaches, UORO scalably provides unbiasedness of gradient estimates.

Unbiasedness is of paramount importance in the current theory of stochastic gradient descent.

Furthermore, UORO is experimentally shown to benefit from its unbiasedness, converging even in cases where truncated BPTT fails to reliably achieve good results or diverges pathologically.

@highlight

Introduces an online, unbiased and easily implementable gradient estimate for recurrent models.

@highlight

The authors introduce a novel approach to online learning of the parameters of recurrent neural networks from long sequences that overcomes the imitation of truncated backpropagation through time

@highlight

This paper approaches online training of RNNs in a principled way, and proposes a modification to RTRL and to use forward approach for gradient calculation.