Designing neural networks for continuous-time stochastic processes is challenging, especially when observations are made irregularly.

In this article, we analyze neural networks from a frame theoretic perspective to identify the sufficient conditions that enable smoothly recoverable representations of signals in L^2(R).

Moreover, we show that, under certain assumptions, these properties hold even when signals are irregularly observed.

As we converge to the family of (convolutional) neural networks that satisfy these conditions, we show that we can optimize our convolution filters while constraining them so that they effectively compute a Discrete Wavelet Transform.

Such a neural network can efficiently divide the time-axis of a signal into orthogonal sub-spaces of different temporal scale and localization.

We evaluate the resulting neural network on an assortment of synthetic and real-world tasks: parsimonious auto-encoding, video classification, and financial forecasting.

The predominant assumption made in deep learning for time series analysis is that observations are made regularly, with the same duration of time separating each successive timestamps BID10 BID14 BID27 BID20 BID29 BID3 .

However, this assumption is often inappropriate, as many real-world time series are observed irregularly and are, occasionally, event-driven (e.g., financial data, social networks, internet-of-things).One common approach in working with irregularly observed time series is to interpolate the observations to realign them to a regular time-grid.

However, interpolation schemes may result in spurious statistical artifacts, as shown in BID17 BID4 .

Fortunately, procedures for working with irregularly observed time series in their unaltered form have been devised, notably in the field of Gaussian-processes and kernel-learning BID17 BID4 and more recently in deep learning BID24 .In this article, we investigate the underlying representation of time series data as it is processed by a neural network.

Our objective is to identify a class of neural networks that provably guarantee information preservation for certain irregularly observed signals.

In doing so, we must analyze neural networks from a frame theoretic perspective, which has enabled a clear understanding of the impact discrete sampling has on representations of continuous-time signals BID7 BID5 BID6 BID13 BID15 BID22 .Although frame theory has historically been studied in the linear setting, recent work by BID26 has related frames with non-linear operators in Banach space, to what can be interpreted as non-linear frames.

Here, we extend this generalization of frames to characterize entire families of neural networks.

In doing so, we can show that the composition of certain non-linear neural layers (i.e., convolutions and fully-connected layers) form non-linear frames in L 2 (R), while others do not (i.e., recurrent layers).Moreover, frame theory can be used to analyze randomly-observed time series.

In particular, when observations are made according to a family of self-exciting point processes known as Hawkes processes BID12 .

We prove that such processes, under certain assumptions of stability, almost surely yield non-linear frames on a class of band-limited functions.

That is to say, that despite having discrete and irregular observations, the signal of interest can still be smoothly recovered.

As we obtain a family of convolutional neural networks that constitute non-linear frames, we show that under certain conditions, such networks can efficiently divide the time-axis of a time series into orthogonal sub-spaces of different temporal scale and localization.

Namely, we optimize the weights of our convolution filters while constraining them so that they effectively compute a Discrete Wavelet Transform BID23 .

Our numerical experiments on synthetic data highlight this unique capacity that allows neural networks to learn sparse representations of signals in L 2 (R), and how such a property is particularly powerful when training parsimoniously parameterized auto-encoders.

Such auto-encoders learn optimal ways of compressing certain classes of input signals.

Finally, we show that the ability of these networks to divide time series into a set sub-spaces, corresponding to different temporal scales and localization, can be composed with existing predictive frameworks to improve both accuracy and efficiency.

This is demonstrated on real-world video classification and financial forecasting tasks.

DISPLAYFORM0 We introduce the article with a theoretical analysis of the sufficient conditions on neural networks that enable smoothly recoverable representations of signals in L 2 (R) and prove that, under certain assumptions, this property holds true in the irregularly observed setting.

We proceed to show that by enforcing certain constraints on convolutional filters, we can guarantee that the representation that the neural network produces only depends on the coordinates of the input signal in an learned orthonormal basis.3.

Numerical experiments: Finally, we evaluate the resulting constrained convolutional neural network on an assortment of synthetic and real-world tasks: parsimonious auto-encoding, video classification, and financial forecasting.

• L 2 (R) is the space of square-integrable real-valued functions defined on R and equipped with the norm induced by the inner product f, g ∈ L 2 (R) → ∫ t∈R f (t)g(t)dt.• L 2 d (R) is the space of square-integrable d-dimensional vector-valued functions defined on R and equipped with the norm induced by the inner product f, g ∈ L DISPLAYFORM0 • l 2 (Z) is the space of square-integrable real-valued sequences indexed by Z and equipped with the norm induced by the inner product (x), (y) ∈ l 2 (Z) → ∑ n∈Z x n y n .• l 2 d (Z) is the space of square-integrable d-dimensional vector-valued sequences indexed by Z and equipped with the norm induced by the inner product (x), (y) ∈ l DISPLAYFORM1 Note that the inner products of the spaces we consider are Hilbert spaces on the classes of equivalent functions for the Lebesgue measure.• F T [⋅] denotes the Fourier transform and z denotes the complex conjugate of z ∈ C. Recall the Fourier transform of a sequence (x n ) ∈ Z is given at any frequency ω by F T [(x n )](ω) = ∑ n∈Z e −2πiωn x n .•

For a function DISPLAYFORM2 • For a set A, (ξ) denotes the sequence (ξ n ) ∈ A Z .• (ξ)[∶∶ 2] denotes (ξ 2n ) n∈Z .

That is, the dilation of a sequence by a factor of 2.• For two vectors DISPLAYFORM3

We begin by investigating sufficient conditions on composite functions that guarantee such functions produce discretized representations of continuous-time signals that can be smoothly reconstructed.

To do so, we must leverage frame theory BID6 , a theory developed to precisely to characterize the suitable properties for linear representations of irregularly observed signals.

Intuitively, a frame is a representation of a signal that enables signal recovery in a smooth manner (i.e., suitable for the representation to be homeomorphic).Formally BID6 , we define a frame as an operator from L 2 (R) to l 2 (Z) that is characterized by a family of functions (S n ) n∈Z in L 2 (R) (i.e., the atoms of the frame).

DISPLAYFORM0 is a frame of L 2 (R) if and only if there exist two real-valued constants 0 DISPLAYFORM1 Representations provided by frames depend smoothly on their inputs.

Moreover, a direct consequence of the definition above is that a frame is invertible in a smooth manner on its image.

There are many examples of frames.

For now, we provide two concrete examples from BID6 BID22 .

Recall the definition of the Haar function as .

In both cases, the atoms of the frame are orthonormal families of functions -it is trivial to prove that A = B = 1.

While the first frame works for the entire space of square integrable functions, the second only applies to the sub-space of band-limited signals.

DISPLAYFORM2 DISPLAYFORM3 This fundamental proposition is proven in BID6 BID22 .

As our goal is to find the conditions for non-linear representations of L 2 (R) to be homeomorphic, we unfortunately can not leverage properties in the linear setting.

Therefore, we must adopt an alternative definition.

Let a non-linear frame be an operator from L 2 (R) to l 2 (Z) that is characterized by a family of functions (S n ) n∈Z in L 2 (R) and a family of non-linear real valued functions (ψ n ) n∈Z defined over DISPLAYFORM4 A non-linear discrete representation scheme DISPLAYFORM5 It is worth noting that a linear frame (in the standard definition of the term) is still a frame in this non-linear setting.

Proposition 1.2 Smoothness of signal recovery: A non-linear frame is invertible on its image of L 2 (R) and the inverse is DISPLAYFORM6 and therefore f and g are in same equivalence class in L 2 (R).

Therefore, F (S),(ψ) is injective and if we consider (x), (y) ∈ F (S),(ψ) (L 2 (R)) and if we denote by f x the only element in DISPLAYFORM7 In a later section, we will show that smooth signal recovery is crucial for non-linear signal approximations (consisting of a finite number of coefficients) to remain stable during reconstruction.

However in order to show this, we must first explore the sufficient conditions on non-linear operators to produce non-linear frames.

We start by introducing several definitions on multivariate real-valued functions.

DISPLAYFORM8 , is a BLI operator and we refer to (A, B) as some framing constants of Φ. Theorem 1.1 BLI operators and linear frames: Let (Φ l ) l=1...

L be a collection of BLI operators with framing constants ((A l , B l )) l=1...

L and F a frame on L 2 (R) with framing constants A 0 and DISPLAYFORM9 The proof of the theorem is immediate but we use it to expose how our careful choice of the definition of non-linear frames is leveraged.

First let us recall that injectivity is preserved by composition.

Then, we initiate an immediate proof by induction with a simple remark: consider two functions f, g ∈ L 2 (R) DISPLAYFORM10 To conclude the proof, a similar statement can then be made if we compose DISPLAYFORM11 This proof allows us to make guarantees about operator pipelines while relying on conditions that are simple to verify.

We can now use the theory we have established to analyze the representational properties of neural networks; in particular, convolutional neural networks (CNN) and recurrent neural networks (RNN).

Here, we study representational properties of recent CNN architectures BID11 BID18 BID19 ) that rely on depth-wise separable convolutions.

We show that by enforcing certain constraints on the structure of temporal filters, we obtain a network that is, provably, a non-linear frame.

Here we trade off expressiveness for representational guarantees as we impose constraints on network parameters.

In depth-wise separated convolution stacks BID11 BID18 ) a temporal convolution is applied before a depth-wise linear transform and finally a leaky ReLU layer.

We assume that the depth-wise linear operators being learned are all full rank (or full column rank if they increase the number of dimensions of the representation).

Such an assumption makes sense for CNNs being trained by a stochastic optimization method with non-pathological data-sets.

Inspired by the multi-scale parsing enabled by the discrete wavelet transform or dyadic wavelet transform we employ time domain convolutions that are conjugate mirror filters BID22 .

Such time domain filters constitute a decomposition filter bank consisting of cascading convolutions.

The decomposition filter bank admits a dual reconstruction filter bank thereby guaranteeing injectivity.

Definition 1.4 Element-wise Leaky ReLU (LReLU): Consider 0 < α << 1, LReLU applies a piecewise linear function element-wise as DISPLAYFORM0 Definition 1.5 Depth-wise fully connected layer (DFC): Consider two integers DISPLAYFORM1 Lemma 1.1 Full column rank DFC (FDFC) layers are BLI: The function DISPLAYFORM2 A is full column rank is left invertible.

Also, the left inverse is Lipschitz as DISPLAYFORM3 Proof 1.3 As LReLU is strictly increasing and continuous therefore it is invertible and as A is full column rank it admits a left inverse which proves the first part of the lemma.

We finish the proof by using the fact that linear functions in vector spaces of finite dimensions are Lipschitz, the fact that LReLU and its inverse are Lipschitz, and the fact that Lipschitz-ness is preserved by composition.

∎ Let us now study the representational properties of time domain convolution layers whose filters are constrained in the Frequency domain.

Definition 1.6 Reconstructible convolution layer (RConv): Consider two convolution filters h, g ∈ l 2 (Z) such that there existh,g ∈ l 2 (Z) and DISPLAYFORM4 The following convolution is a Reconstructible convolution layer: DISPLAYFORM5 Later on, we show that entire families of suchh,g ∈ l 2 (Z) exist under some conditions on h, g. In particular Eq. (4) will provide simple sufficient conditions on h and g for Eq.(1) to hold.

Lemma 1.2 Temporal convolutions allowing reconstruction: Consider four temporal convolution filters h,h, g,g such that their Fourier transforms satisfy (1).

DISPLAYFORM6 (where [∶∶ −1] means that we iterate in reverse order on the filter weights) which proves the pair of convolution filters constitutes an invertible operator.

Also, DISPLAYFORM7 , therefore, if we recall that the Fourier Transform diagonalizes convolutions and turns time reversal into complex conjugacy, we have DISPLAYFORM8 DISPLAYFORM9 is BLI.

Proof 1.5 Let us recall again that injectivity is stable by composition of operators.

It is also clear that non-linear framing conditions remain true as composite bi-Lipschitz functions are also bi-Lipschitz.

∎ With the proposition above it is now trivial to prove the theorem below.

is a non-linear frame.

Now, we expose the framing properties of RNNs (for an introduction on RNNs we refer the reader to BID9 ).

For the vast majority of popular recurrent architectures (for instance, LSTMs, GRUs BID16 BID10 ) the use of bounded output layers leads to saturation and vanishing gradients.

With such vanishing gradients BID8 , it is possible to find series of input sequences that diverge in l 2 (Z) while their outputs through the RNN are a Cauchy sequence.

Proposition 1.4 Saturating RNNs do not provide non-linear frames: Let us consider a RNN DISPLAYFORM0 where DISPLAYFORM1 is not a linear frame.

as the sequence (v k ) would then be Cauchy and therefore converge as l 2 (Z) is complete for the l 2 norm.

∎ Such a proposition highlights a key difference between the representational ability of RNNs and CNNs.

We explore representations of irregularly sampled data through the lens of non-linear frames.

We now show that even when signals are irregularly observed by a random sampling process, that particular neural networks can still, almost surely produce a homeomorphic representation.

Sampling by Hawkes processes is a common assumption in finance, seismology, and social media analytics BID25 BID2 BID4 BID30 BID21 BID31 .

We use (I N t ) to denote the canonical filtration associated with the stochastic process (N t ) t∈R .

We recommend BID12 ) for a more thorough introduction to this concept.

As a simplification, we denote I N t to be the information generated by (N s ) s<t .

DISPLAYFORM0 For a Hawkes process characterized by φ ∶ t ∈ R → φ(t) ≥ 0, ∀t < 0, φ(t) = 0, µ ≥ 0, we assume DISPLAYFORM1 In other words, λ t is the number of observations per unit of time expected given the events that occurred until time t. Intuitively, if λ t is higher, then it is more likely for observations to be available shortly after the time t.

As in BID4 BID2 , Hawkes processes can be used to model the random observation time of a continuous-time series in a setting where information is observed asynchronously in an event-driven manner across multiple channels (the extension to multi-variate point processes is immediate).Proposition 1.5 Sampling density of stable Hawkes processes: If the Hawkes process (2) is stable (i.e. ∫ t∈R φ(t)dt < 1), then almost surely DISPLAYFORM2 A complete proof of the ergodic behavior of stable Hawkes processes is provided in BID12 .

Now, given an asymptotic Nyquist sampling rate for a random sampling scheme, the following lemma delineates which frames can still be used for signal recovery.

In particular, we can no longer recover all signals in an unambiguous manner.

Hence, exact recovery is only possible for band-limited functions (i.e. functions whose Fourier transform has bounded support).

DISPLAYFORM3 (where the real axis represents sampling frequencies) with left inverse F + .

Considering R < R 1 and S ∈ L 2 (R) such that DISPLAYFORM4 Complete proof is given in BID7 BID5 ; the theorem is regarded as the fundamental theorem of frame analysis for irregularly observed signals.

We now leverage the fundamental properties that were obtained in the deterministic setting and extend them to provide guarantees under random sampling schemes.

Proposition 1.6 Under Hawkes process random sampling, framing is preserved almost surely: Let (t n ) n∈Z be a family of sampling time-stamps generated by a stable Hawkes process whose intensity follows the dynamics described in (2), denote DISPLAYFORM5 and let S be a frame operator abiding by conditions (3), then almost surely the frame is injective on DISPLAYFORM6 when translated by the irregular time-stamps (t n ) n∈Z .

The proposition is a direct consequence of Prop.

1.5 and Theorem 1.3.

∎ Theorem 1.4 Recovery of randomly observed band-limited signals: Let (t n ) n∈Z be a family of sampling time-stamps generated by a stable Hawkes process whose intensity follows the dynamics described in (2) Consider R = 1 2 µ 1− ∫t∈R φ(t)dt , let F (S) be a frame operator with atoms (S(⋅ − t n )) n∈Z DISPLAYFORM0 is almost surely a non-linear frame over the set of functions in L 2 (R) whose Fourier Transform has its support included in [−R, R].

In particular such a representation is invertible on its image by a Lipschitz inverse.

Proof 1.8 Previously, we proved Theorem 1.2 on the preservation of framing properties by composition with FDFC and RConv layers.

In Prop.

1.6 we proved that F (S(⋅−tn)) n∈Z is almost surely a frame of the subset of L 2 (R) of functions with band-limit [−R, R].

∎ One concern, however, is that the theorems we developed assumed observations on the entire real axis are available as well as infinite representations indexed by Z. In particular, a theory of framing for band-limited functions is useful but only applies to periodic functions.

Bounded support function of L 2 (R) are part of the many examples that are not band-limited BID22 ; Benedetto & Ferreira (2003).

In our objective to develop theoretical statements that can be leveraged in practice (i.e., when computing with finite time and memory), we must now extend our analysis to (1) functions observed on compact intervals and (2) finite approximations of signals.

The following statements show how the requirements of Lipschitz-ness in non-linear frames provide guarantees on the impact of approximation errors associated with finite representation of continuous-time signals.

The theorems above can be employed for irregularly observed functions that are periodic and band-limited BID6 BID5 BID22 .

However, since we hope to develop a representational theory that is applicable to non-stationary signals, we must also consider non-periodic functions.

In the appendix, we show how wavelet decomposition can efficiently approximate certain classes of functions that are smooth and not band-limited.

With ⌊log 2 (N )⌋ scales of decomposition and O(N ) scalars representing the approximation of f as P DISPLAYFORM0 norm for the space of α Lipschitz functions (see Definition 4.1 in appendix).

As we employ functions that are BLI the impact of the approximation error remains controlled.

DISPLAYFORM1 In other words, the numerical representation can be arbitrarily close to the true representation of smooth, continuous-time functions with compact support.

Indeed, if W is a wavelet basis, then Proj +∞ W (f ) = f .

The argument stresses the critical role of our assumption of the Lipschitz-ness of frames and the BLI functions which guarantees that representations based on approximations can be arbitrarily accurate.

So far, we have focused on sufficient conditions to make accurate representation of continuous-time signals possible as they are observed randomly and as the corresponding observation are processed non-linearly.

We now show that additional conditions on time-domain convolutional filter banks further guarantee that the representation is minimal (i.e., produces orthogonal outputs).

As our goal is to obtain different representations of a time series while avoiding redundancy, let us introduce multi-resolution approximations BID22 .

DISPLAYFORM0 .

In addition we require that (H 0 ) there exists an orthonormal family (S(⋅ − n)) n∈Z such that span((S(⋅) n∈Z )) = H l , i.e. S(⋅ − n) is a Riesz basis of H 0 with scaling function S.Such Riesz basis is proven to exist in BID22 ; the family of Haar wavelets is merely an example.

General conditions for a function S ∈ L 2 (R) to be a scaling function are given by the following theorem.

Theorem 2.1 Conjugate mirror temporal convolution layer (CMConv): BID23 Let κ S and κ W in l 2 (Z) be two convolution filters such that DISPLAYFORM1 DISPLAYFORM2 DISPLAYFORM3 We further assume that inf DISPLAYFORM4 is a scaling function S of L 2 (R) for a multi-resolution approximation.

Moreover, the Wavelet function W defined as the inverse Fourier transform of DISPLAYFORM5 n ∈ Z is an orthonormal basis W l defined as the orthogonal complement of H l in H l+1 .

In particular, (W l,n ) l∈Z,n∈Z is an orthonormal basis of L 2 (R).

We now show how depth-wise separable convolutions with scaling and wavelet filters quickly come with guarantees of orthogonality.

In the following we consider an input space with d input channels and a series of affine operators with increasing output dimensions (d l ) L l=1 .

We denote F S(⋅−tn) n∈Z by F to simplify notations.

DISPLAYFORM0 The representation (θ l (f )) l=1...

L is a non-linear frame that only depends on the coordinates of f in an orthonormal basis of L 2 (R).Proof 2.1 We start the proof by showing that ( DISPLAYFORM1 ) and then as ∀ω ∈ [−1 2, DISPLAYFORM2 1 2) 2 = 2 the first part of the proof is concluded.

The second part of the proof utilizes the fact that the cascading convolutions above compute a Discrete Wavelet Transform BID22 DISPLAYFORM3 The cascaded time domain convolutions being computed yield the coordinates of f in an orthonormal basis.

Therefore, as the orthogonal CNN grows deeper it can only yield novel orthogonal information about the input signal that is informative of its properties on a particular temporal scale.

Such is the nature of our efficiency claim for the neural networks we consider.

A key point here is that the 1x1 convolutions operate in depth and not along the axis of time which preserves the temporal scaling properties of the Discrete Wavelet Transform.

As noted in Mallat FORMULA32 FORMULA32 DISPLAYFORM0 In our implementation we approximate the constraint by computing the Fast Fourier Transform of the filter, since it is defined discretely in time by a finite set of weights.

Therefore, we interleave the normal training step of κ S with solving the following following minimization problem DISPLAYFORM1 where H is the number of free parameters we allow in our temporal convolution filter.

Such an optimization problem can be rewritten as a difference of convex (DC) functions (as DISPLAYFORM2 2 − 2 is clearly convex and convexity is stable by composition by a non-decreasing function) and an adapted solver (Tao & An, 1998) can then take advantage of the particular structure of the problem to find an optimal solution rapidly.

Here, we show that the sufficient conditions for neural networks to yield non-linear frames are computationally tractable.

The following experiments explore the empirical properties of such neural networks compared to various baselines.

In our first numerical experiment, we generate regularly sampled non-stationary stochastic processes, characterized by a random mixture of Gabor functions BID22 and step functions.

As shown in FIG5 , the resulting signals are highly irregular, lack permanent seasonality, and have compact support.

The objective here is to devise a procedure to train conjugate mirror (convolutional) filters with stochastic optimization methods to progressively improve representational power.

We train a 16 parameter filter κ S to optimally conduct the following compression (i.e., auto-encoding) task.

The pair of filters specified in Eq. (4) are employed as in Theorem 2.2 to produce the coordinates of the input signal in the wavelet basis corresponding to the (learned) filters κ S and κ W .

The input signals are uni-variate with 128 observations each.

The encoding, therefore, initially consists of 128 scalar values, of which, only the 64 with higher magnitude are selected -all other values are set to 0.An inverse Discrete Wavelet Transform is then employed to reconstruct the input signal.

The quality of this reconstruction is measured by the squared L 2 loss, which penalizes discrepancies between the input signal and its reconstruction.

To train this model, we use a stochastic optimization algorithm, RMSProp, to minimize the aforementioned loss.

We train for 2,500 iterations with a learning rate of 10 −3 .

This optimization is interleaved with a constraint enforcing program that enforces Eq. (4) every 100 iterations.

FIG5 shows that this procedure progressively improves the randomly-initialized filters and significantly out-performs an LSTM-based auto-encoder model.

We further show that a wavelet representation can be composed with classical recurrent architecture (in regularly observed settings) to mitigate the effect of noisy data.

This is particularly useful for LSTM networks BID16 , since hyperbolic tangent layers tend to saturate in the presence of high-magnitude perturbations.

The YouTube-8M data-set contains millions of YouTube videos and their associated genres (AbuEl-Haija et al., 2016) .

Because the frames in each video are pre-featurized (i.e., a time series of featurized frames), models designed for this data-set must solely leverage the temporal structure in the data.

In particular, the raw video feed is not available.

A thorough description of the baselines we employ is available in BID1 .

This has enabled the authors of the paper to achieve state-of-the-art results in video classifications using a 2-layer LSTM model.

In our experiment, we train a similar model to learn on a multi-scale wavelet representation of data.

This representation separates the original time series into d scales, varying from fine to coarse.

Each of the d time series in this multi-scale representation are fed into a similar 2-layer LSTMs with d 2 times fewer parameters which results in a decrease of the total number of parameters in the recurrent layers by a factor of d. The outputs of each LSTM, are then concatenated before the final soft-max layer.

We provide a model diagram detailing these components in the appendix.

Our experimental results in FIG3 indicate that this multi-scale representation greatly improves the performance of recurrent neural networks while using far fewer parameters.

In 2015, an astounding medium volume of 40 million shares of AAPL (Apple Inc.) were traded each day.

With the price of each share at approximately 100 USD, each 15-minute trading period represents an exchange of 142 million USD.

Trades are highly irregular events characterized by an instantaneous exchange of shares between actors.

Forecasting trade volume at a very fine resolution is essential in leveraging arbitrage opportunities.

However, the noisy nature of financial markets makes this task incredibly challenging BID0 elapsed.

On average, the duration between time-stamps was 907ms (25th percentile: 200ms, median: 220ms, 75th percentile: 1800ms).After the first scale projection onto a Haar wavelet basis BID22 is produced (with a characteristic resolution τ = 8 seconds), both the wavelet transform network (with M = 8) and the LSTM make predictions with this first scale as input.

Each model is evaluated by the L 2 loss against a baseline predicting a constant trading volume equal to the average over the previous 15 observed minutes.

Notice that in FIG6 , the LSTM struggles with the noisiness of the data, whereas the wavelet transform network is robust, and manages to improve the prediction performance by a half-percent.

This half-percent represents 50 thousand USD of exchanged volume over a 15 minute period.

In this article, we analyze neural networks from a frame theoretic perspective.

In doing so, we come to the conclusion that by considering time series as an irregularly observed continuous-time stochastic processes, we are better able to devise robust and efficient convolutional neural networks.

By leveraging recent contributions to frame theory, we prove properties about non-linear frames that allow us to make guarantees over an entire class of convolutional neural networks.

Particularly regarding their capacity to produce discrete representations of continuous time signals that are both injective and bi-Lipschitz.

Moreover, we show that, under certain conditions, these properties almost certainly hold, even when the signal is irregularly observed in an event-driven manner.

Finally, we show that bounded-output recurrent neural networks do not satisfy the sufficient conditions to yield non-linear frames.

This article is not limited to the theoretical statements it makes.

In particular, we show that we can build a convolutional neural network that effectively computes a Discrete Wavelet Transform.

The network's filters are dynamically learned while being constrained to produce outputs that preserve both orthogonality and the properties associated with non-linear frames.

Our numerical experiments on real-world prediction tasks further demonstrate the benefits of such neural networks.

Notably, their ability to produce compact representations that allow for efficient learning on latent continuous-time stochastic processes.

We rely on Wavelet approximations BID22 Under some conditions on W BID22 , the family W l,τ = 1 √ 2 l W ( ⋅−τ 2 l ) can be orthonormal and every function f ∈ L 2 (R) can be written in the limit as f = ∑ l∈Z ∑ τ ∈Z < f, W l,τ > W l,τ .

A Wavelet function is defined as the high frequency mirror of a low frequency Scale function whose unit translations constitute a set of orthonormal atoms for a frame of L 2 (R) (i.e. a Riesz basis of L 2 (R)).In the following we consider functions with bounded support and restrict our study to functions defined on the interval [0, 1] to simplify notations.

A change of variable can immediately be employed to generalize the statements below to any bounded support function.

In other words we consider functions defined on compacts that can be well approximated by polynomial splines and therefore have a certain degree of smoothness.

The proposition above, proven in BID22 helps us examine how such an approximation affects the representations we employ.

Figure 6: The architecture we propose for the Youtube video classification task that leverages a multi-resolution approximation computed by a wavelet convolution stack.

@highlight

Neural architectures providing representations of irregularly observed signals that provably enable signal reconstruction.

@highlight

Proves that convolutional neural networks with Leaky ReLU activation function are nonlinear frames, with similar results for non-uniformly sampled time-series

@highlight

This article considers neural networks over time-series and show that the first convolutional filters can be chosen to represent a discrete wavelet transform.