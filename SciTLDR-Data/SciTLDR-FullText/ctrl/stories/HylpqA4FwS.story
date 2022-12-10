Recurrent neural networks (RNNs) are particularly well-suited for modeling long-term dependencies in sequential data, but are notoriously hard to train because the error backpropagated in time either vanishes or explodes at an exponential rate.

While a number of works attempt to mitigate this effect through gated recurrent units, skip-connections, parametric constraints and design choices, we propose a novel incremental RNN (iRNN), where hidden state vectors keep track of incremental changes, and as such approximate state-vector increments of Rosenblatt's (1962) continuous-time RNNs.

iRNN exhibits identity gradients and is able to account for long-term dependencies (LTD).

We show that our method is computationally efficient overcoming overheads of many existing methods that attempt to improve RNN training, while suffering no performance degradation.

We demonstrate the utility of our approach with extensive experiments and show competitive performance against standard LSTMs on LTD and other non-LTD tasks.

Recurrent neural networks (RNNs) in each round store a hidden state vector, h m ∈ R D , and upon receiving the input vector, x m+1 ∈ R d , linearly transform the tuple (h m , x m+1 ) and pass it through a memoryless non-linearity to update the state over T rounds.

Subsequently, RNNs output an affine function of the hidden states as its prediction.

The model parameters (state/input/prediction parameters) are learnt by minimizing an empirical loss.

This seemingly simple update rule has had significant success in learning complex patterns for sequential input data.

Nevertheless, that training RNNs can be challenging, and that performance can be uneven on tasks that require long-term-dependency (LTD), was first noted by Hochreiter (1991) , Bengio et al. (1994) and later by other researchers.

Pascanu et al. (2013b) attributed this to the fact that the error gradient back-propagated in time (BPTT), for the time-step m, is dominated by product of partials of hiddenstate vectors, T −1 j=m ∂hj+1 ∂hj , and these products typically exhibit exponentially vanishing decay or explosion, resulting in incorrect credit assignment during training and test-time.

Rosenblatt (1962) , on whose work we draw inspiration from, introduced continuous-time RNN (CTRNN) to mimic activation propagation in neural circuitry.

CTRNN dynamics evolves as follows: τġ(t) = −αg(t) + φ(U g(t) + W x(t) + b), t ≥ t 0 .

(

Here, x(t) ∈ R d is the input signal, g(t) ∈ R D is the hidden state vector of D neurons,ġ i (t) is the rate of change of the i-th state component; τ, α ∈ R + , referred to as the post-synaptic time-constant, impacts the rate of a neuron's response to the instantaneous activation φ(U g(t) + W x(t) + b); and U ∈ R D×D , W ∈ R D×d , b ∈ R D are model parameters.

In passing, note that recent RNN works that draw inspiration from ODE's (Chang et al., 2019) are special cases of CTRNN (τ = 1, α = 0).

Vanishing Gradients.

The qualitative aspects of the CTRNN dynamics is transparent in its integral form:

This integral form reveals that the partials of hidden-state vector with respect to the initial condition, ∂g(t) ∂g(t0) , gets attenuated rapidly (first term in RHS), and so we face a vanishing gradient problem.

We will address this issue later but we note that this is not an artifact of CTRNN but is exhibited by ODEs that have motivated other RNNs (see Sec. 2).

Shannon-Nyquist Sampling.

A key property of CTRNN is that the time-constant τ together with the first term −g(t), is in effect a low-pass filter with bandwidth ατ −1 suppressing high frequency components of the activation signal, φ((U g(s)) + (W x(s)) + b).

This is good, because, by virtue of the Shannon-Nyquist sampling theorem, we can now maintain fidelity of discrete samples with respect to continuous time dynamics, in contrast to conventional ODEs (α = 0).

Additionally, since high-frequencies are already suppressed, in effect we may assume that the input signal x(t) is slowly varying relative to the post-synaptic time constant τ .

Equilibrium.

The combination of low pass filtering and slowly time varying input has a significant bearing.

The state vector as well as the discrete samples evolve close to the equilibrium state, i.e., g(t) ≈ φ(U g(t) + W x(t) + b) under general conditions (Sec. 3).

Incremental Updates.

Whether or not system is in equilibrium, the integral form in Eq. 2 points to gradient attenuation as a fundamental issue.

To overcome this situation, we store and process increments rather than the cumulative values g(t) and propose dynamic evolution in terms of increments.

Let us denote hidden state sequence as h m ∈ R D and input sequence x m ∈ R d .

For m = 1, 2, . . .

, T , and a suitable β > 0 τġ(t) = −α(g(t) ± h m−1 ) + φ(U (g(t) ± h m−1 ) + W x m + b), g(0) = 0, t ≥ 0 (3)

Intuitively, say system is in equilibrium and −α(µ(x m , h m−1 ))+φ(U µ(x m , h m−1 )+W x m +b) = 0.

We note state transitions are marginal changes from previous states, namely, h m = µ(x m , h m−1 ) − h m−1 .

Now for a fixed input x m , as to which equilibrium is reached depends on h m−1 , but are nevertheless finitely many.

So encoding marginal changes as states leads to "identity" gradient.

Incremental RNN (iRNN) achieves Identity Gradient.

We propose to discretize Eq. 3 to realize iRNN (see Sec. 3).

At time m, it takes the previous state h m−1 ∈ R D and input x m ∈ R d and outputs h m ∈ R D after simulating the CTRNN evolution in discrete-time, for a suitable number of discrete steps.

We show that the proposed RNN approximates the continuous dynamics and solves the vanishing/exploding gradient issue by ensuring identity gradientIn general, we consider two options, SiRNN, whose state is updated with a single CTRNN sample, similar to vanilla RNNs, and, iRNN, with many intermediate samples.

SiRNN is well-suited for slowly varying inputs.

Contributions.

To summarize, we list our main contributions: (A) iRNN converges to equilibrium for typical activation functions.

The partial gradients of hiddenstate vectors for iRNNs converge to identity, thus solving vanishing/exploding gradient problem!

(B) iRNN converges rapidly, at an exponential rate in the number of discrete samplings of Eq. 1.

SiRNN, the single-step iRNN, is efficient and can be leveraged for slowly varying input sequences.

It exhibits fast training time, has fewer parameters and better accuracy relative to standard LSTMs.

(C) Extensive experiments on LTD datasets show that we improve upon standard LSTM accuracy as well as other recent proposals that are based on designing transition matrices and/or skip connections.

iRNNs/SiRNNs are robust to time-series distortions such as noise paddings (D) While our method extends directly (see Appendix A.1) to Deep RNNs, we deem these extensions complementary, and focus on single-layer to highlight our incremental perspective.

Gated Architectures.

Long short-term memory (LSTM) (Hochreiter & Schmidhuber, 1997 ) is widely used in RNNs to model long-term dependency in sequential data.

Gated recurrent unit (GRU) (Cho et al., 2014 ) is another gating mechanism that has been demonstrated to achieve similar performance of LSTM with fewer parameters.

Some recent gated RNNs include UGRNN (Collins et al., 2016) , and FastGRNN (Kusupati et al., 2018) .

While mitigating vanishing/exploding gradients, they do not eliminate it.

Often, these models incur increased inference, training costs, and model size.

Unitary RNNs.

Arjovsky et al. (2016); Jing et al. (2017) ; ; Mhammedi et al. (2016) focus on designing well-conditioned state transition matrices, attempting to enforce unitary-property, during training.

Unitary property does not generally circumvent vanishing gradient (Pennington et al. (2017) ).

Also, it limits expressive power and prediction accuracy while also increasing training time.

Deep RNNs.

These are nonlinear transition functions incorporated into RNNs for performance improvement.

For instance, Pascanu et al. (2013a) empirically analyzed the problem of how to construct deep RNNs.

Zilly et al. (2017) proposed extending the LSTM architecture to allow stepto-step transition depths larger than one.

Mujika et al. (2017) proposed incorporating the strengths of both multiscale RNNs and deep transition RNNs to learn complex transition functions from one timestep to the next.

While Deep RNNs offer richer representations relative to single-layers, it is complementary to iRNNs.

Residual/Skip Connections.

Jaeger et al. (2007) ; Bengio et al. (2013); Campos et al. (2017) ; Kusupati et al. (2018) feed-forward state vectors to induce skip or residual connections, to serve as a middle ground between feed-forward and recurrent models, and to mitigate gradient decay.

Nevertheless, these connections, cannot entirely eliminate gradient explosion/decay.

For instance, Kusupati et al. (2018) suggest h m = α m h m−1 + β m φ(U h m−1 + W x m + b), and learn parameters so that α m ≈ 1 and β m ≈ 0.

Evidently, this setting can lead to identity gradient, observe that setting β m ≈ 0, implies little contribution from the inputs and can conflict with good accuracy, as also observed in our experiments.

Linear RNNs. (Bradbury et al., 2016; Balduzzi & Ghifary, 2016) have focused on speeding up recurrent neural networks by replacing recurrent connections, such as hidden-to-hidden interactions, with light weight linear components.

While this has led to reduced training time, it has resulted in significantly increasing model size.

For example, typically requires twice the number of cells for LSTM level performance.

ODE/Dynamical Perspective.

There are a few works that are inspired by ODEs, and attempt to address stability, but do not end up eliminating vanishing/exploding gradients.

Talathi & Vartak (2015) proposed a modified weight initialization strategy based on a dynamical system perspective on weight initialization process that leads to successfully training RNNs composed of ReLUs.

Niu et al. (2019) analyzed RNN architectures using numerical methods of ODE and propose a family of ODE-RNNs.

Chang et al. (2019) , propose Antisymmetric-RNN.

Their key idea is to express the transition matrix in Eq. 1, for the special case α = 0, τ = 1, as a difference: U = V − V T and note that the eigenspectrum is imaginary.

Nevertheless, Euler discretization, in this context leads to instability, necessitating damping of the system.

As such vanishing gradient cannot be completely eliminated.

Its behavior is analogous to FastRNN Kusupati et al. (2018) , in that, identity gradient conflicts with high accuracy.

In summary, we are the first to propose evolution over the equilibrium manifold, and demonstrating identity gradients.

Neural ODEs (Chen et al., 2018; Rubanova et al., 2019) have also been proposed for time-series prediction to deal with irregularly sampled inputs.

To do this they parameterize the derivative of the hidden-state in terms of an autonomous differential equation and let the ODE evolve in continuous time until the next input arrives.

As such, this is not our goal, our ODE explicitly depends on the input, and evolves until equilibrium for that input is reached.

We introduce incremental updates to bypass vanishing/exploding gradient issues, which is not of specific concern for these works.

We use Euler's method to discretize Eq. 3 in steps δ = ητ .

Denoting the kth step as g k = g(kδ)

Rearranging terms we get a compact form for iRNN (see Fig. 1 ).

In addition we introduce a learnable parameter η k m and let it be a function of time m and the recursion-step k.

with some suitable initial condition.

This could be g 0 = 0 or initialized to the previous state, i.e., g 0 = h m−1 at time m. In many of our examples, we find the input sequence is slowly varying, and K = 1 can also realize good empirical performance.

We refer to this as single-step-incremental-RNN (SiRNN).

For both iRNN and SiRNN we drop the superscript whenever it is clear from the context.

Figure 1 : iRNN depicted by unfolding into K recursions for one transition from

.

See Sec. A.2 for implementation and pseudo-code.

This resembles Graves (2016) , who propose to vary K with m as a way to attend to important input transitions.

However, the transition functions used are gated units, unlike our conventional ungated functions.

As such, while this is not their concern, equilibrium may not even exist and identity gradients are not guaranteed in their setup.

Root Finding and Transitions.

The two indices k and m should not be confused.

The index m ∈ [T ] refers to the time index, and indexes input, x m and hidden state h m over time horizon T .

The index k ∈ [K] is a fixed-point recursion for converging to the equilibrium solution at each time m, given input x m and the hidden state h m−1 .

We iterate over k so that at k = K, g K satisfies,

The recursion (Eq. 5) at time m runs for K rounds, terminates, and recursion is reset for the new input, x m+1 .

Indeed, Eq. 5 is a standard root-finding recursion, with g k−1 serving as the previous solution, plus a correction term, which is the error,

.

If the sequence converges, the resulting solution is the equilibrium point.

Proposition 2 guarantees a geometric rate of convergence.

Identity Gradient.

We will informally (see Theorem 1) show here that partial gradients are identity.

Say we have for sufficiently large K, h m = g K is the equilibrium solution.

It follows that,

Taking derivatives, we have,

Thus if the matrix (∇φ(·)U − αI) is not singular, it follows that ( Residual Connections vs. iRNN/SiRNN.

As such, our architecture is a special case of skip/residual connections.

Nevertheless, unlike skip connections, our connections are structured, and the dynamics driven by the error term ensures that the hidden state is associated with equilibrium and leads to identity gradient.

No such guarantees are possible with unstructured skip connections.

Note that for slowly varying inputs, after a certain transition-time period, we should expect SiRNN to be close to equilibrium as well.

Without this imposed structure, general residual architectures can learn patterns that can be dramatically different (see Fig. 2 ).

Let us now collect a few properties of Eq. 3 and Eq. 5.

First, denote the equilibrium solutions for an arbirary input x ∈ R d , arbitrary state-vector ν ∈ R D , in an arbitrary round:

Whenever the equilibrium set is a singleton, we denote it as a function h eq (x, ν).

For simplicity, we assume below that η i k is a positive constant independent of k and i. Proposition 1.

Suppose, φ(·) is a 1-Lipshitz function in the norm induced by · , and U < α, then for any x m ∈ R d and h m−1 ∈ R D , it follows that M eq (x, ν) is a singleton and as K → ∞, the iRNN recursions converge to this solution, namely,

We now invoke the Banach fixed point theorem, which asserts that a contractive operator on a complete metric space converges to a unique fixed point, namely, T K (g) → g * .

Upon substitution, we see that this point g * must be such that, φ(U (g * + h m−1 ) + W x m + b) − (g * + h m−1 ) = 0.

Thus equilibrium point exists and is unique.

Result follows by setting h m h eq (x m , h m−1 ).

Handling U ≤ α.

In experiments, we set α = 1, and do not enforce U ≤ α constraint.

Instead, we initialize U as a Gaussian matrix with IID mean zero, small variance components.

As such, the matrix norm is smaller than 1.

Evidently, the resulting learnt U matrix does not violate this condition.

Next we show for η > 0, iRNN converges at a linear rate, which follows directly from Proposition 1.

Proposition 2.

Under the setup in Proposition 1, it follows that,

Remark.

Proposition 1 accounts for typical activation functions ReLU, tanh, sigmoids as well as deep RNNs (appendix A.1).

In passing we point out that, in our experiments, we learn parameters η k m , and a result that accounts for this case is desirable.

We describe this case in Appendix A.3.

A fundamental result we describe below is that partials of hidden-state vectors, on the equilibrium surface is unity.

For technical simplicity, we assume a continuously differentiable activation, which appears to exclude ReLU activations.

Nevertheless, we can overcome this issue, but requires more technical arguments.

The main difficulty stems from ensuring that derivatives along the equilibrium surface exist, and this can be realized by invoking the implicit function theorem (IFT).

IFT requires continuous differentiability, which ReLUs violate.

Nevertheless, recent results 1 suggests that one can state implicit function theorem for everywhere differentiable functions, which includes ReLUs.

Theorem 1.

Suppose φ(·) is a continuously differentiable, 1-Lipshitz function, with U < α.

Then as K → ∞, = −I. Furthermore, as K → ∞ the partial gradients over arbitrary number of rounds for iRNN is identity.

.

We overload notation and view the equilibrium point as a function of h m−1 , i.e., g * (h m−1 ) = h eq (x m , h m−1 ).

Invoking standard results 2 in ODE's, it follows that g * (h m−1 ) is a smooth function, so long as the Jacobian, ∇ g ψ(g * , h m−1 ) with respect to the first coordinate, g * , is non-singular.

Upon computation, we see that,

It follows that we can take partials of the state-vectors.

By taking the partial derivatives w.r.t.

h m−1 in Eq. 5, at the equilibrium points we have [∇φ(g * , h m−1 )U − αI][ ∂g * ∂hm−1 + I] = 0 (see Eq. 7).

The rest of the proof follows by observing that the first term is non-singular.

Remark.

We notice that replacing h m−1 with −h m−1 in Eq. 12 will lead to ∂heq ∂hm−1 = I, which also has no impact on magnitudes of gradients.

As a result, both choices are suitable for circumventing vanishing or exploding gradients during training, but still may converge to different local minima and thus result in different test-time performance.

Furthermore, notice that the norm preserving property is somewhat insensitive to choices of α, so long as the non-singular condition is satisfied.

Approximate Identity.

Theorem 1 guarantees identity gradients on the equilibrium surface.

However, Proposition 2 only guarantees we are close to equilibrium surface for finite K. Consequently we may still have a vanishing gradient problem for sufficiently large T .

Nevertheless, this can be readily handled by suitable choices of K. This is because, while larger T degrades gradients, it is compensated by larger values of K. To ensure that end-to-end partials are no worse than 1± , we need K = O(log(T / )), thanks to the geometric rate of convergence of partials, which can be established under additional conditions on transition function φ.

As such log(T / ) is small for most problems we encountered.

Fig. 2 depicts phase portrait and illustrates salient differences between RNN, FastRNN (RNN with skip connection), and iRNN (K=5).

RNN and FastRNN exhibit complex trajectories, while iRNN trajectory is smooth, projecting initial point (black circle) onto the equilibrium surface (blue) and moving within it (green).

This suggests that iRNN trajectory belongs to a low-dimensional manifold.

Variation of Equilibrium w.r.t.

Input.

As before, h eq be an equilibrium solution for some tuple (h m−1 , x m ).

It follows that,

This suggests that, whenever the input undergoes a slow variation, we expect that the equilibrium point moves in such a way that U ∂h eq must lie in a transformed span of

Low Rank Matrix Parameterization.

For typical activation functions, note that whenever the argument is in the unsaturated regime, ∇φ(·) ≈ I. We then approximately get span(αI − U ) ≈ span(W ).

We can express these constraints as U = αI + V H with low-rank matrices V ∈ R D×d1 , H ∈ R d1×D , and further map both U h m and W x m onto a shared space.

Since in our experiments the signal vectors we encounter are low-dimensional, and sequential inputs vary slowly over time, we enforce this restriction in all our experiments.

In particular, we consider,

The parameter matrix P ∈ R D×D maps the contributions from input and hidden states onto the same space.

To decrease model-size we let P = U = (I + V H) learn these parameters.

We organize this section as follows.

First, the experimental setup, competing algorithms will be described.

Then we present an ablative analysis to highlight salient aspects of iRNN and justify some of our experimental choices.

We then plot and tabulate experimental results on benchmark datasets.

Choice of Competing Methods: We choose competing methods based on the following criteria: (a) methods that are devoid of additional application or dataset-specific heuristics, (b) methods that leverage only single cell/block/layer, and (c) methods without the benefit of complementary add-ons (such as gating, advanced regularization, model compression, etc.).

Requiring (a) is not controversial since our goal is methodological.

Conditions (b),(c) are justifiable since we could also leverage these add-ons and are not germane to any particular method 3 .

We benchmark iRNN against standard RNN, LSTM (Hochreiter & Schmidhuber, 1997) , (ungated) AntisymmetricRNN (Chang et al., 2019) , (ungated) FastRNN (Kusupati et al., 2018) .

Vorontsov et al. (2017) attributes it to modReLU or leaky ReLU activations.

Leaky ReLUs allow for linear transitions, and copy task being a memory task benefits from it.

With hard non-linear activation, unitary RNN variants can take up to 1000's of epochs for even 100-length sequences (Vorontsov et al. (2017) ).

Implementation.

For all our experiments, we used the parametrized update formulation in Eq. 9 for iRNN .

We used tensorflow framework for our experiments.

For most competing methods apart from AntisymmetricRNN, which we implemented, code is publicly available.

All the experiments were run on an Nvidia GTX 1080 GPU with CUDA 9 and cuDNN 7.0 on a machine with Intel Xeon 2.60 GHz CPU with 20 cores.

Datasets.

Pre-processing and feature extraction details for all publicly available datasets are in the appendix A.4.

We replicate benchmark test/train split with 20% of training data for validation to tune hyperparameters.

Reported results are based on the full training set, and performance achieved on the publicly available test set.

Table 4 (Appendix) and A.4 describes details for all the data sets.

Hyper Parameters We used grid search and fine-grained validation wherever possible to set the hyper-parameters of each algorithm, or according to the settings published in (Kusupati et al., 2018; Arjovsky et al., 2016 ) (e.g. number of hidden states).

Both the learning rate and η's were initialized to 10 −2 .

The batch size of 128 seems to work well across all the data sets.

We used ReLU as the non-linearity and Adam (Kingma & Ba (2015) ) as the optimizer for all the experiments.

We perform ablative analysis on the benchmark add-task (Sec 4.3.1) for sequence length 200 for 1000 iterations and explore mean-squared error as a metric.

Fig. 3 depicts salient results.

(a) Identity Gradients & Accuracy: iRNN accuracy is correlated with identity gradients.

Increasing K improves gradients, and correlates with increased accuracy (Fig. 3) .

While other models h t = αh t−1 + βφ((U − γI)h t−1 + W x t ), can realize identity gradients for suitable choices; linear (α = 1, β = 1, γ = 0, U = 0), FastRNN (α ≈ 1, β ≈ 0, γ = 0) and Antisymmetric (α = 1, β = 1, U = V − V T , U ≤ γ), this goal may not be correlated with improved test accuracy.

FastRNN(η = 0.001), Antisymmetric (γ = 0.01, = 0.001) have good gradients but poorer test accuracy relative to FastRNN(η = 0.01), Antisymmetric(γ = 0.01, = 0.1), with poorer gradients.

(b) Identity gradient implies faster convergence: Identity gradient, whenever effective, must be capable of assigning credit to the informative parts, which in turn results in larger loss gradients, and significantly faster convergence with number of iterations.

This is borne out in figure 3(a) .

iRNN for larger K is closer to identity gradient with fewer (unstable) spikes (K = 1, 5, 10).

With K = 10, iRNN converges within 300 iterations while competing methods take about twice this time (other baselines not included here exhibited poorer performance than the once plotted).

(c) SiRNN (iRNN with K = 1 delivers good performane in some cases.

Fig. 3(a) illustrates that iRNN K = {5, 10} achieves faster convergence than SiRNN, but the computational overhead per iteration roughly doubles or triples in comparison.

SiRNN is faster relative to competitors.

For this reason, we sometimes tabulate only SiRNN, whenever it is SOTA in benchmark experiments, since accuracy improves with K but requires higher overhead.

We list five types of datasets, all of which in some way require effective gradient propagation: (1) Conventional Benchmark LTD tasks (Add & Copy tasks) that illustrate that iRNN can rapidly learn long-term dependence; (2) Benchmark vision tasks (pixel MNIST, perm-MNIST) that may not require long-term, but nevertheless, demonstrates that iRNN achieves SOTA for short term dependencies but with less resources.

(3) Noise Padded (LTD) Vision tasks (Noisy MNIST, Noisy CIFAR), where a large noise time segment separates information segments and the terminal state, and so the learner must extract information parts while rejecting the noisy parts; (4) short duration activity embedded in a larger time-window (HAR-2, Google-30 in Appendix Table 4 and many others A.7), that usually arise in the context of smart IoT applications and require a small model-size footprint.

Chang et al. (2019) further justify (3) and (4) as LTD, because for these datasets where only a smaller unknown segment(s) of a longer sequence is informative.

(5) Sequence-sequence prediction tasks (PTB language modeling) that are different from terminal prediction (reported in appendix A.7).

Addition and Copy tasks (Hochreiter & Schmidhuber, 1997) have long been used as benchmarks in the literature to evaluate LTD (Hori et al., 2017; Arjovsky et al., 2016; Martens & Sutskever, 2011) .

We follow the setup described in Arjovsky et al. (2016) to create the adding and copying tasks.

See appendix A.4 for detailed description.

For both tasks we run iRNN with K = 5.

Figure 4 show the average performance of various methods on these tasks.

For the copying task we observe that iRNN converges rapidly to the naive baseline and is the only method to achieve zero average cross entropy.

For the addition task, both FastRNN and iRNN solves the addition task but FastRNN takes twice the number of iterations to reach desired 0 MSE.

4 In both the tasks, iRNN performance is much more stable across number of online training samples.

In contrast, other methods either takes a lot of samples to match iRNN 's performance or depict high variance in the evaluation metric.

This shows that iRNN converges faster than the baselines (to the desired error).

These results demonstrate that iRNN easily and quickly learns the long term dependencies .

We omitted reporting unitary RNN variants for Add and Copy task.

See Sec. 4.1 for copy task.

On Add-task we point out that our performance is superior.

In particular, for the longer T = 750 length, Arjovsky et al. (2016) , points out that MSE does not reach zero, and uRNN is noisy.

Others either (Wisdom et al., 2016) do not report add-task or report only for shorter lengths .

Next, we perform experiments on the sequential vision tasks: (a) classification of MNIST images on a pixel-by-pixel sequence; (b) a fixed random permuted MNIST sequence (Lecun et al., 1998) .

These tasks typically do not fall in the LTD categories (Chang et al., 2019) , but are useful to demonstrate faster training, which can be attributed to better gradients.

For the pixel-MNIST task, Kusupati et al. (2018) reports that it takes significantly longer time for existing (LSTMs, Unitary, Gated, Spectral) RNNs to converge to reasonable performance.

In contrast, FastRNN trains at least 2x faster than LSTMs.

Our results (table 1) for iRNN shows a 9x speedup relative LSTMs, and 2x speedup in comparison to Antisymmetric.

In terms of test accuracy, iRNN matches the performance of Antisymmetric, but with at least 3x fewer parameters.

We did not gain much with increased K values 5 .

For the permuted version of this task, we seem to outperform the existing baselines 6 .

In both tasks, iRNN trained at least 2x faster than the strongest baselines.

These results demonstrate that iRNN converges much faster than the baselines with fewer parameters.

Additionally, as in Chang et al. (2019) , we induce LTD by padding CIFAR-10 with noise exactly replicating their setup, resulting in Noisy-CIFAR.

We extend this setting to MNIST dataset resulting in Noisy-MNIST.

Intuitively we expect our model to be resilient to such perturbations.

We attribute iRNN's superior performance to the fact that it is capable of suppressing noise.

For example, say noise is padded at t > τ and this results in W x t being zero on average.

For iRNN the resulting states ceases to be updated.

So iRNN recalls last informative state h τ (modulo const) unlike RNNs/variants!

Thus information from signal component is possibly better preserved.

Results for Noisy-MNIST and Noisy-CIFAR are shown in Table 2 .

Note that almost all timesteps contain noise in these datasets.

LSTMs perform poorly on these tasks due to vanishing gradients.

This is consistent with the earlier observations (Chang et al., 2019) .

iRNN outperforms the baselines very comprehensively on CIFAR-10, while on MNIST the gains are smaller, as it's a relatively easier task.

These results show that iRNN is more resilient to noise and can account for longer dependencies.

We are interested in detecting activity embedded in a longer sequence with small footprint RNNs (Kusupati et al. (2018) ): (a) Google-30 (Warden, 2018) , i.e. detection of utterances of 30 commands plus background noise and silence, and (b) HAR-2 (Anguita et al., 2012), i.e. Human Activity Recognition from an accelerometer and gyroscope on a Samsung Galaxy S3 smartphone.

Table 3 shows accuracy, training time, number of parameters and prediction time.

Even with K = 1, we compare well against competing methods, and iRNN accuracy improves with larger K. Interestingly, higher K yields faster training as well as moderate prediction time, despite the overhead of additional recursions.

These results show that iRNN outperforms baselines on activity recognition tasks, and fits within IoT/edge-device budgets.

We point out in passing that our framework readily admits deep multi-layered networks within a single time-step.

Indeed our setup is general; it applies to shallow and deep nets; small and large time steps.

As a case in point, the Deep Transition RNN Pascanu et al. (2013c) :

is readily accounted by Theorem 1 in an implicit form:

So is Deep-RNN Hermans & Schrauwen (2013) .

The trick is to transform h m → h m + h m+1 and h m+1 → h m + h m+1 .

As such, all we need is smoothness of f h , which has no restriction on # layers.

On the other hand, that we do not have to limit the number of time steps is the point of Theorem 1, which asserts that the partial differential of hidden states (which is primarily why vanishing/exploding gradient arises Pascanu et al. (2013b) in the first place) is identity!!

Given an input sequence and iRNN model parameters, the hidden states can be generated with the help of subroutine 1.

This routine can be plugged into standard deep learning frameworks such as Tensorflow/PyTorch to learn the model parameters via back-propagation.

Then there exists > 0 such that if g 0 − h eq ≤ where h eq denotes the fixed point, the sequence g i generated by the Euler method converges to the equilibrium solution in M eq (h k−1 , x k ) locally with linear rate.

The proof is based on drawing a connection between the Euler method and inexact Newton methods, and leverages Thm.

2.3 in Dembo et al. (1982) .

See appendix Sec. A.8.1 Thm.

3 and Sec. A.7.5 (for proof, empirical verification).

k ∇F (g i ) < 1, ∀k, ∀i, the forward propagation (Eq. 13) is stable and the sequence {g i } converges locally at a linear rate.

Table 4 and table 6 lists the statistics of all the datasets described below.

Google-12 & Google-30: Google Speech Commands dataset contains 1 second long utterances of 30 short words (30 classes) sampled at 16KHz.

Standard log Mel-filter-bank featurization with 32 filters over a window size of 25ms and stride of 10ms gave 99 timesteps of 32 filter responses for a 1-second audio clip.

For the 12 class version, 10 classes used in Kaggle's Tensorflow Speech Recognition challenge 7 were used and remaining two classes were noise and background sounds (taken randomly from remaining 20 short word utterances).

Both the datasets were zero mean -unit variance normalized during training and prediction.

8 : Human Activity Recognition (HAR) dataset was collected from an accelerometer and gyroscope on a Samsung Galaxy S3 smartphone.

The features available on the repository were directly used for experiments.

The 6 activities were merged to get the binarized version.

The classes Sitting, Laying, Walking_Upstairs and Standing, Walking, Walking_Downstairs were merged to obtain the two classes.

The dataset was zero mean -unit variance normalized during training and prediction.

Penn Treebank: 300 length word sequences were used for word level language modeling task using Penn Treebank (PTB) corpus.

The vocabulary consisted of 10,000 words and the size of trainable word embeddings was kept the same as the number of hidden units of architecture.

This is the setup used in (Kusupati et al., 2018; .

Pixel-by-pixel version of the standard MNIST-10 dataset 9 .

The dataset was zero mean -unit variance normalized during training and prediction.

Permuted-MNIST: This is similar to Pixel-MNIST, except its made harder by shuffling the pixels with a fixed permutation.

We keep the random seed as 42 to generate the permutation of 784 pixels.

To introduce more long-range dependencies to the Pixel-MNIST task, we define a more challenging task called the Noisy-MNIST, inspired by the noise padded experiments in Chang et al. (2019) .

Instead of feeding in one pixel at one time, we input each row of a MNIST image at every time step.

After the first 28 time steps, we input independent standard Gaussian noise for the remaining time steps.

Since a MNIST image is of size 28 with 1 RGB channels, the input dimension is m = 28.

The total number of time steps is set to T = 1000.

In other words, only the first 28 time steps of input contain salient information, all remaining 972 time steps are merely random noise.

For a model to correctly classify an input image, it has to remember the information from a long time ago.

This task is conceptually more difficult than the pixel-by-pixel MNIST, although the total amount of signal in the input sequence is the same.

This is exactly replica of the noise paded CIFAR task mentioned in Chang et al. (2019) .

Instead of feeding in one pixel at one time, we input each row of a CIFAR-10 image at every time step.

After the first 32 time steps, we input independent standard Gaussian noise for the remaining time steps.

Since a CIFAR-10 image is of size 32 with three RGB channels, the input dimension is m = 96.

The total number of time steps is set to T = 1000.

In other words, only the first 32 time steps of input contain salient information, all remaining 968 time steps are merely random noise.

For a model to correctly classify an input image, it has to remember the information from a long time ago.

This task is conceptually more difficult than the pixel-by-pixel CIFAR-10, although the total amount of signal in the input sequence is the same.

Addition Task: We closely follow the adding problem defined in (Arjovsky et al., 2016; Hochreiter & Schmidhuber, 1997) to explain the task at hand.

Each input consists of two sequences of length T. The first sequence, which we denote x, consists of numbers sampled uniformly at random U[0, 1].

The second sequence is an indicator sequence consisting of exactly two entries of 1 and remaining entries 0.

The first 1 entry is located uniformly at random in the first half of the sequence, whilst the second 1 entry is located uniformly at random in the second half.

The output is the sum of the two entries of the first sequence, corresponding to where the 1 entries are located in the second sequence.

A naive strategy of predicting 1 as the output regardless of the input sequence gives an expected mean squared error of 0.167, the variance of the sum of two independent uniform distributions.

Copying Task: Following a similar setup to (Arjovsky et al., 2016; Hochreiter & Schmidhuber, 1997) , we outline the copy memory task.

Consider 10 categories, {a i } 9 i=0 .

The input takes the form of a T + 20 length vector of categories, where we test over a range of values of T. The first 10 entries are sampled uniformly, independently and with replacement from {a i } 7 i=0 , and represent the sequence which will need to be remembered.

The next T − 1 entries are set to a 8 , which can be thought of as the 'blank' category.

The next single entry is a 9 , which represents a delimiter, which should indicate to the algorithm that it is now required to reproduce the initial 10 categories in the output.

The remaining 10 entries are set to a 8 .

The required output sequence consists of T + 10 repeated entries of a 8 , followed by the first 10 categories of the input sequence in exactly the same order.

The goal is to minimize the average cross entropy of category predictions at each time step of the sequence.

The task amounts to having to remember a categorical sequence of length 10, for T time steps.

A simple baseline can be established by considering an optimal strategy when no memory is available, which we deem the memoryless strategy.

The memoryless strategy would be to predict a 8 for T + 10 entries and then predict each of the final 10 categories from the set {a i } 7 i=0 i=0 independently and uniformly at random.

The categorical cross entropy of this strategy is 10 log (8) T +20

10 : This dataset is based on Daily and Sports Activity (DSA) detection from a resourceconstrained IoT wearable device with 5 Xsens MTx sensors having accelerometers, gyroscopes and magnetometers on the torso and four limbs.

The features available on the repository were used for experiments.

The dataset was zero mean -unit variance normalized during training and prediction.

Yelp-5: Sentiment Classification dataset based on the text reviews 11 .

The data consists of 500,000 train points and 500,000 test points from the first 1 million reviews.

Each review was clipped or padded to be 300 words long.

The vocabulary consisted of 20000 words and 128 dimensional word embeddings were jointly trained with the network.

In our experiments section, we stated that some of the potential baselines were removed due to experimental conditions enforced in the setup.

Here we clearly justify our choice.

Mostly the reasoning is to avoid comparing complementary add-ons and compare the bare-bone cells.

• Cooijmans et al. (2016) is removed since its an add-on and can be applied to any method.

Besides its pixel-mnist results involve dataset specific heuristics.

• Gong et al. (2018) is also an add-on and hence can be applied to any method.

• Zilly et al. (2017); Pascanu et al. (2013a); Mujika et al. (2017) denote deep transitioning methods.

They are add-ons for any single recurrent block and hence can be applied to any recurrent cell.

• Gating variants of single recurrent cells (Chang et al., 2019; Kusupati et al., 2018) have also been removed.

Since iRNN can be extended to a gating variant and hence its just an add-on.

Figure 5 shows the results for remaining experiments for the addition task for length 100, 400.

Table 7 shows the results including left out baselines for Pixel-MNIST and permute-MNIST task.

Here we also include star rating prediction on a scale of 1 to 5 of Yelp reviews Yelp (2017) .

Table 8 shows the results for this dataset.

We also include activity recognition tasks: (a) Google-12 Warden (2018) , i.e. detection of utterances of 10 commands plus background noise and silence and (b) DSA-19 Altun et al. (2010) , Daily and Sports Activity (DSA) detection from a resource-constrained IoT wearable device with 5 Xsens MTx sensors having accelerometers, gyroscopes and magnetometers on the torso and four limbs.

Table  9 shows results for these activities along with some other baselines for activity recognition tasks mentioned in Sec. 4.3.4 and described in Sec. A.4.

We follow (Kusupati et al., 2018; to setup our PTB experiments.

We only pursue one layer language modelling, but with more difficult sequence length (300).

Table 10 reports all the evaluation metrics for the PTB Language modelling task with 1 layer as setup by Kusupati et al. (2018) , including test time and number of parameters (which we omitted from the main paper due to lack of space).

A.7.5 LINEAR RATE OF CONVERGENCE TO FIXED POINT Empirically we verify the local convergence to a fixed point with linear rate by comparing the Euclidean distance between the approximate solutions, h (k) t , using Eq. 11 with g 0 = 0 and the fixed points, h t , computed using FSOLVE from SCIPY.

The learnable parameters are initialized suitably and then fixed.

We illustrate our results in Fig. 6 , which clearly demonstrates that the approximate solutions tend to converge with linear rate.

Here we include some experiments to show that our theoretical assumptions hold true.

Non-Singularity of the matrix D For our iRNN parametrization to satisfy the conditions of having equillibrium points to be locally asymptotically stable, the eigen values of the matrix D = (∇φ(·)U − γI) should be negative.

We plot a histogram of the eigenvalues of D for all the points in the HAR-2 dataset.

As illustrated in the figure 7, all the eigenvalues are negative.

A.7.7 IDENTITY GRADIENT COMPARISON iRNN VS RNN To verify Theorem.

1 empirically, we train RNN and iRNN on the HAR-2 data set (see more details in Sec. 4), respectively, and plot in Fig. 8 layer h 1 in log scale to confirm that our approach leads to no vanishing or exploding gradients when the error is back-propagated through time.

We also conducted experiments to verify that the gradient of iRNN is norm preserving (see Sec. A.7.8 and Figure .

3).

As we see clearly, RNN suffers from serious vanishing gradient issue in training, while iRNN's backpropagated gradients is close to 1, and the variance arises mainly our approximation of fixed points and stochastic behavior in training networks, demonstrating much better training stability of iRNN.

A.7.8 GRADIENT NORM W.R.T. LOSS

In addition to the gradient ratio we plot in Sec.4.2, we also show in figure 9 , the more popular quantity captured in earlier works (Arjovsky et al., 2016; could become zero by the virtue that the loss is nearly zero.

This happens in our addition task experiment, because MSE is close to zero, we experience nearly 0 value for this quantity.

But this is clearly because the MSE is 0.

Also note that none of our graphs have log scale, which is not the case in earlier works.

The conclusion that can be drawn from the loss-gradient is that it is somewhat stable, and can inform us about quality of convergence.

We also plot Recall that we rewrite the fixed-point constraints in our iRNN as the following ODE:

Then based on the Euler method, we have the following update rule for solving fixed-points:

Inexact Newton methods Dembo et al. (1982) refer to a family of algorithms that aim to solve the equation system F (z) = 0 approximately at each iteration using the following rule:

where ∇F denotes the (sub)gradient of function F , and r i denotes the error at the i-th iteration between F (z i ) and 0.

By drawing the connection between Eq. 13 and Eq. 15, we can set z i ≡ g i and s i ≡ η

Then based on Eq. 15 we have

(16) Lemma 1 (Thm. 2.3 in Dembo et al. (1982) ).

Assume that

where · denotes an arbitrary norm and the induced operator norm.

There exists ε > 0 such that, if z 0 − z * ≤ ε, then the sequence of inexact Newton iterates {z i } converges to z * .

Moreover, the convergence is linear in the sense that z i+1 − z * * ≤ τ z i − z * * , where y * = ∇F (z * )y . .

This together with Figure 3 shows that the gradients are identity everywhere for K = 10 21

Theorem 3 (Local Convergence with Linear Rate).

Assume that the function F in Eq. 12 and the parameter η (i) k in Eq. 13 satisfy

Then there exists > 0 such that if g 0 − h eq ≤ where h eq denotes the fixed point, the sequence {g i } generated by the Euler method converges to the equilibrium solution in M eq (h k−1 , x k ) locally with linear rate.

Proof.

By substituting Eq. 16 into Eq. 17, to prove local convergence we need to guarantee

By taking the square of both sides in Eq. 19, we can show that Eq. 19 is equivalent to Eq. 18.

We then complete our proof.

Corollary 2.

Assume that I + η (i) k ∇F (g i ) < 1, ∀i, ∀k holds.

Then the forward propagation using Eq. 13 is stable and our sequence {g i } converges locally with linear rate.

Proof.

By substituting Eq. 16 into Eq. 17 and based on the assumption in the corollary, we have

Further based on Prop.

2 in Chang et al. (2019) and Thm.

2, we then complete our proof.

<|TLDR|>

@highlight

Incremental-RNNs resolves exploding/vanishing gradient problem by updating state vectors based on difference between previous state and that predicted by an ODE.

@highlight

The authors address the problem of signal propagation in recurrent neural networks by building an attractor system for the signal transition and checking whether it converges to an equilibrium. 