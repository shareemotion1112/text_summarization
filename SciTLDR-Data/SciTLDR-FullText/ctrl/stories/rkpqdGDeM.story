In this work, we propose the Sparse Deep Scattering Croisé Network (SDCSN) a novel architecture based on the Deep Scattering Network (DSN).

The DSN is achieved by cascading  wavelet transform convolutions with a complex modulus and a time-invariant operator.

We extend this work by first, crossing multiple wavelet family transforms to increase the feature diversity while avoiding any learning.

Thus providing a more informative latent representation and benefit from the development of highly specialized wavelet filters over the last decades.

Beside, by combining all the different wavelet representations, we reduce the amount of prior information needed regarding the signals at hand.

Secondly, we develop an optimal thresholding strategy for over-complete filter banks that regularizes the network and controls instabilities such as inherent non-stationary noise in the signal.

Our systematic and principled solution sparsifies the latent representation of the network by acting as a local mask distinguishing between activity and noise.

Thus, we propose to enhance the DSN by increasing the variance of the scattering coefficients representation as well as improve its robustness with respect to non-stationary noise.

We show that our new approach is more robust and outperforms the DSN on a bird detection task.

Modern Machine Learning focuses on developing algorithms to tackle natural machine perception tasks such as speech recognition, computer vision, recommendation among others.

Historically, some of the proposed models were based on well-justified mathematical tools from signal processing such as Fourier analysis.

Hand-crafted features were then computed based on those tools and a classifier was trained supervised for the task of interest.

However, such theory-guided approaches have become almost obsolete with the growth of computational power and the advent of high-capacity models.

As such, over the past decade the standard solution evolved around deep neural networks (DNNs).

While providing state-of-the-art performance on many benchmarks, at least two pernicious problems still plague DNNs: First, the absence of stability in the DNN's input-output mapping.

This has famously led to adversarial attacks where small perturbations of the input lead to dramatically different outputs.

In addition, this lack of control manifests in the detection thresholds (i.e: ReLU bias) of DNNs, rendering them prone to instabilities when their inputs exhibit non-stationary noise and discontinuities.

Second, when inputs have low SNR, or classes are unbalanced, the stability of DNNs is cantilevered.

A common approach to tackle this difficulty is to increase both the size of the training set and the number of parameters of the network resulting in a longer training time and a costly labeling process.

In order to alleviate these issues we propose the use of the DSN by creating a new non-linearity based on continuous wavelet thresholding.

Thus our model, inherits the mathematical guarantees intrinsic to the DSN regarding the stability, and improves the control via wavelet thresholding method.

Then, in order to produce time-frequency representation that are not biased toward a single wavelet family, we propose to combine diverse wavelet families throughout the network.

Increasing the variability of the scattering coefficient, we improve the linearization capability of the DSN and reduce the need of an expert knowledge regarding the choice of specific filter bank with respect to each input signal.

The paper is organized as follows: 1.1 and 1.2 are devoted to the related work and contribution of the paper, the section 2 shows the theoretical results, where 2.1 is dedicated to the network architecture and its properties, and 2.2 provides the milestone of our thresholding method, then section 2.3 shows the characterization, via latent representation, of our network on different events by on the Freefield1010 1 audio scenes dataset.

Finally, we evaluate our architecture and compare it to the DSN on a bird detection task are shown in 2.4.

The appendix in divided into three parts, Appendix A provides both, the pre-requisite and details about building the wavelets dictionary to create our architecture; Appendix B shows additional results on the sparsity of the SDCSN latent representations; Appendix C shows mathematical details and proofs for the over-complete thresholding non-linearity.

We extend the Deep Scattering Network, first developed in BID29 and first successfully applied in BID9 ; Andén & Mallat.

The Scattering Network (SN) is a cascade of linear and non-linear operators on the input signal.

The linear transformation is a wavelet transform, and the nonlinear transformation is a complex modulus.

For each layer, the scattering coefficients are computed according to the application of the scaling function on the representation.

This network is stable (Lipschitz-continuous) and suitable for machine learning tasks as it removes spatiotemporal nuisances by building space/time-invariant features.

The translation invariant property is provided by the scaling function that acts as an averaging operator on each layer of the transform leading to an exponential decay of the scattering coefficients Waldspurger (2017).

Since the continuous wavelet transform increases the number of features, the complex modulus is used as its contractive property reduces the variance of the projected space BID30 .

Two extensions of this architecture have been already developed: the Joint Scattering Network BID2 and the time-chromafrequency scattering BID27 .

They introduced extra parameterization of the wavelets coefficient in the second layer of the network to capture frequency correlations allowing the scattering coefficient to represent the transient structure of harmonic sounds.

Thresholding in the wavelet domain remains a powerful approach for signal denoising as it exploits the edge-detector property of wavelets, providing a sparse representation of the input signal in the time-frequency plane.

This property is characterized for each wavelet by its vanishing moments expressing the orthogonality of the wavelet with respect to a given order of smoothness in the input signal.

We base our approach on the theories relating the thresholding of signal in the wavelet basis and the evaluation of the best basis.

Both are realized via a risk evaluation that arose from different perspectives: statistical signal processing BID11 ; BID23 , information theory BID12 ; Wijaya et al. (2017) ; BID13 , and signal processing BID31 BID28 .

However, to the best of our knowledge, there is no thresholding method developed for continuous wavelet transform.

We will thus extend the work of Berkner (1998) on thresholding over-complete dictionnary in the case of TIDWT and Biorthogonal-DWT to build a risk evaluation in the case of over-complete continuous wavelet transform.

As opposed to the chroma-time-frequency scattering, using one wavelet family filter bank but deriving many symmetries of the latter, we propose to use multiple wavelet families having complementary properties (described in A.2) within a unified network yielding cross connections.

It helps the architecture to provide higher dimensional and uncorrelated features, reducing the need of an expert to hand-choose the DSN wavelet filters, and also enables any downstream classifier to have greater orbit learning capacity.

Therefore our architecture, the Deep Croisé Scattering Network (DCSN), leverages the simultaneous decomposition of complementary filter-banks as well as their crossed decomposition, hence the term "croisé."

Then, endowing this architecture with our novel thresholding operator, we build the SDSCN providing new features based on the reconstruction risk of each wavelet dictionary.

This method based on empirical risk minimization will bring several advantages.

First, it enables us to insure and control the stability of the input-output mapping via thresholding the coefficients.

Second, the model has sparse latent representations that ease the learning of decision boundaries as well as increases generalization performance.

Finally, the risk associated with each wavelet family provides a characterization of the time-frequency components of the analyzed signal, that, when combines with scattering features enhances the high linearization capacity of DSN.

As opposed to ReLU-based nonlinearities that impose sparsity by thresholding coefficients based on a fixed learned scalar threshold, we propose an input-dependant locally adaptive thresholding method.

Therefore, our contribution leading to the Sparse Deep Croisé Network is twofold:• Deep Croisé Scattering Network: a natural extension of the DSN allowing the use of multiple wavelet families and their crossed representations.• Derivation of optimal non-orthogonal thresholding for overcomplete dictionaries: empirical risk minimization leads to an analytical solution for the denoising mask, allowing deterministic per-input solutions, and endowing the DSN with sparse latent representations.

The Deep Croisé Scattering Network (DCSN) is a tree architecture (2 layers of such model are shown in FIG0 ) based on the Deep Scattering Network (DSN).

The first layer of a scattering transform corresponding to standard scalogram is now replaced with a 3D tensor by adding the wavelet family dimension.

Hence, it can be seen as a stacked version of multiple scalograms, one per wavelet family.

The second layer of the DCSN brings inter and intra wavelet family decompositions.

In fact, each wavelet family of the second layer will be applied on all the first layer scalograms, the same process is successively applied for building deeper model.

We first proceed by describing the formalism of the DCSN ∀x ∈ L 2 , details on wavelets and filter bank design are provided in Appendix A. We denote by DISPLAYFORM0 the collection of B (1) mother wavelets for the first layer.

We also denote by, DISPLAYFORM1 the resolution coefficients for this first layer with J (1) representing the number of octave to decompose and Q(1) the quality coefficients a.k.a the number of wavelets per octave.

Based on those configuration coefficients, the filter-banks can be derived by scaling of the mother wavelets through the resolution coefficients.

We thus denote the filter-bank creation operator W by DISPLAYFORM2 To avoid redundant notation, we thus denote this filter-bank as W (1,b) with implicit parameters Ψ(1) and Λ (1) .

We now developed of the needed tools to explicit define the filter layer of the DCSN.

We denote by U(1) the output of this first layer and as previously mentioned it consist of a 3D tensor of shape (B (1) , J (1) Q (1) , N ) with N the length of the input signal denoted as x. We omit here boundary conditions, sub-sampling, and consider a constant shape of N throughout the representations.

We thus obtain DISPLAYFORM3 where |.| operator corresponds to an element-wise complex modulus application.

We define the convolution operation between those two objects as DISPLAYFORM4 From this, the second layer we present below will introduce the cross family representations.

First, we denote by λ (2) and Ψ (2) the internal parameters of layer 2 analogous to the first layer definition.

We now denote the second layer representation as DISPLAYFORM5 .

This object is a 5D tensor introduced 2 extra dimension on the previous tensor shape.

In fact, is it defined as DISPLAYFORM6 from this, we denote by croisé representation all the terms in DISPLAYFORM7 .

Based on those notations it is straightforward to extend those representation to layer as U b1→···→b j2,...,j[x].

We however limit ourselves in practice to 2 layers as usually done with the standard scattering networks.

Given those representations, the scattering coefficients, the features per say, are defined as follows: S b1→···→b j2,...,j DISPLAYFORM8 with φ is a scaling function.

This application of a low frequency band-pass filter allows for symmetries invariances, inversely proportional to the cut-off frequency of ψ.

We present an illustration of the network computation in Fig. 2 .As can be seen in the proposed example, while the first layer provides time-frequency information, the second layer characterizes transients as demonstrated in .

With this extended framework, we now dive into the problem of thresholding over complete basis, cases where the quality factor, Q, is greater than 1 which are in practice needed to bring enough frequency precision.

Sparsity in the latent representation of connectivists models have been praised many times Narang et al. FORMULA0 ; BID24 ; BID34 .

It represents the fitness of the internal parameters of a model needed with only few nonzeros coefficients to perform the task at hand.

Furthermore, sparsity is synonym of simpler models as directly related with the Minimum Description Length Dhillon et al. FORMULA0 guaranteeing increased generalization performances.

In addition of those concepts, thresholding brings in practice robustness to noise.

In particular, as we will demonstrate, even in large scale configuration, non-stationnary noise can not be completely handled by common ML approaches on their own.

To do so, we propose to extend standard wavelet thresholding techniques for non-orthogonal filter-banks.

Our approach aims at minimizing the reconstruction error of the thresholded signal in the wavelet domain via an oracle decision.

Through this formulation, we are able to derive an analytical thresholding based on the input representation and the filter-bank redundancy.

We now propose to derive this scheme and then provide interpretation on its underlying tests and computations.

As the decomposition is not orthogonal, the first point to be tackle is the difference of the L 2 approximation errors in between the original basis and the over-complete wavelet basis as Parseval equality does not hold.

Beside, the transpose of the change of basis matrix is not anymore the inverse transform.

Berknet et.

al. in Berkner (1998) proposed the use of the Moore pseudo inverse to build the reconstruction dictionary.

In the following we develop an upper bound to the ideal risk such that we benefit an explicit equation for the thresholding operator that is adapted to any over-complete transformation.

Let's assume the observed signal, denoted by y, is corrupted with white noise such that y = x+ where x is the signal of interest and ∼ N (0, σ 2 ).

We now denote by W ∈ C N (J * Q+1)×n the matrix composed by the the wavelets at each time and scale (i.e: explicit convolution version of W) such that ∀x ∈ R N , W x is the wavelet transform.

We denote by W † ∈ C n×n(J * Q+1) the generalized inverse such that DISPLAYFORM0 Because of the correlation implied by the redundant information contained in the filter banks, the ideal risk is now dependent on all the possible pairs in the frequency axis.

However,the independence in time remains.

Since this optimization problem does not have an analytic expression, we propose the following upper bound explicitly derived in Appendix C.1.

The upper-bound on the optimal risk is denoted by R up and defined as, DISPLAYFORM1 where we denote by R U up the upper bound error term corresponding to unselected coefficients: DISPLAYFORM2 and by R S up the upper bound error term corresponding to the selected coefficients: DISPLAYFORM3 Now, one way to evaluate this upper-bound is to assume an orthogonal basis, and to compare it with the optimal risk in the orthogonal case which leads to the following proposition.

Proposition 1.

Assuming orthogonal filter matrix W O , the upper bound ideal risk coincides with the orthogonal ideal risk: DISPLAYFORM4 the proof is derived in C.2 In order to apply the ideal risk derive, ones needs an oracle decision regarding the signal of interest.

In real application, the signal of interest x is unknown.

We thus propose the following empirical risk: DISPLAYFORM5 This risk corresponds to the empirical version of the ideal risk where the observed signal y is evaluate in the left part of the minimization function.

In order to compare this empirical risk with the ideal version, we propose their comparison the following extreme cases: Proposition 2.

In the case where D S = I, the empirical risk coincides with the upper bound ideal risk:R (y, W ) = R up (x, W ).

Proposition 3.

In the case where D U = I, the following bound shows the distance between the empirical risk and the upper bound ideal risk: DISPLAYFORM6 where, DISPLAYFORM7 Refer to C.3 for proofs.

As the empirical risk introduces the noise in the left part of the risk expression, this term represents the propagation of this noise throughout the decomposition.

We provided a generic development of the risk minimization process.

When applied to a particular path of the scattering network, it is denoted as, DISPLAYFORM8 with DISPLAYFORM9 ( ) ×N and R representing the risk minimization operator based on a given representation and the associated filter-bank.

We define by T the tresholding operator minimizing the the empirical risk, DISPLAYFORM0 In particular when applied to a specific path of the tree, this thresholding operator is denoted as DISPLAYFORM1 We provide in Fig. 2 illustration showing the effect of this thresholding operator at each layer of the network.

We demonstrated in the last section the important of the risk in the optimal thresholding optimization.

This empirical version of this risk represents the ability of the induced representation to perform efficient denoising and signal reconstruction.

This concept is identical to the one of function fitness when considering the denoised ideal signal x and the thresholded reconstruction.

As a result, it is clear that the optimal basis given a signal is the one with minimal empirical risk.

We thus propose here simple visualization and discussion on this concept and motivate the need to use the optimal empirical risk as part of the features characterizing the input signal y along all the representations.

In Fig. 3 , we provide two samples from the dataset corresponding to very different acoustic scene.

One represents transients on the right while the left one provides mixture of natural sounds.

Risk based analysis of the filter-banks fitness provide consistent information with the specificities of the selected wavelets.

In fact, Paul family is known to be very adapted for transient characterization via its high time localization.

On the other hand, the Morlet wavelet is optimal in term of Heisenberg principle and thus suitable for natural sounds such as bird songs, speech, music.

We propose to validate the two contributions over a large-scale audio dataset.

As we will demonstrate below, our method as well as each contribution taken independently and jointly lead to significant increase in the final performance.

We compare our results against the standard SN.

In all cases, the scattering coefficients are then fed into a random forest Breiman FORMULA0 that can be formally defined as a binary classification task, where each label corresponds to the presence or absence of birds.

Each signal is 10sec.

long, and has been sampled at 44.1Khz.

The evaluation of the results is performed via the Area Under Curve metric on 33% of the data.

The experiments are repeated 50 times.

The total audio length of this dataset is thus of slightly more than 11 hours of audio recordings.

To put in comparison, it is about 10× larger than CIFAR10 in term of numbers of scalar values in the dataset.

The results comparing our algorithm to the DSN with each of the wavelet family used in both SDCSN and DCSN are in TAB0 .

Both the SDCSN and DCSN outperform from at least 20% accuracy of any DSN proving the enhancement of the scattering feature by including the crossed latent representations.

For all the architectures, the octave and quality parameters of the layers are J1 = 5, Q1 = 8, J2 = 4, Q2 = 1.

As the feature of interests are birds songs, only high frequency content requires high resolution, the thresholding is applied per window of 2 16 representing ≈ 1.5sec.

When considering different dataset sizes, the impact of denoising can be analyzed in details in Fig. 4a .

As the dataset becomes smaller, the thresholding operator removing the noise perturbations becomes mandatory..

With infinite data and very high capacity classifier, a priori denoising becomes redundant as it is possible for the classifier to leverage the variance of the data to adjust correctly the hyperplanes delimiting the class regions.

However, doing such learning is not possible with small scale dataset hence requiring a priori and deterministic denoising.

Another experiment highlighting the need for denoising in practical application comes from the possibility to have different noise levels from the training set to the test set.

Thus we propose in Fig. 4b the following experiment.

For both the SDCSN and DCSN models, training is achieved done on the denoised dataset.

Then, the testing phase is performed on the raw dataset.

Clearly, performances degrade strongly for the DCSN showing the inhability of the classifier, even though after standard scattering network transform, to be robust to noise level changes during and after training.

This shows empirically the need of a thresholding non-linearity to provide more robustness to Scattering networks.

We now propose to visualize the sparsity induced via our thresholding technique FIG0 ).

To do so we present histograms of the representation with and without thresholding.

Greater sparsity coupled with better performances and closely related to better linearization capacities, which benefits greatly the classifier as the size of the data is small 4a.

We presented an extension of the scattering network so that one can leverage multiple wavelet families simultaneously.

Via a specific topology, cross family representations are performed carrying crucial information, as we demonstrated experimentally, allowing to significantly outperform standard scattering networks.

We then motivated and proposed analytical derivation of an optimal overcomplete basis threhsolding being input adaptive.

By providing greater sparsity in the representation but also a measure of filter-bank fitness.

Again, we provided experimental validation of the use of our thresholding technique proving the robustness implied by such non-linearity.

Finally, the ability to perform active denoising has been demonstrated crucial as we demonstrated that even in large scale setting, standard machine learning approach coupled with the SN fail to discard non-stationary noise.

This coupled with the denoising ability of our approach should provide real world application the stability needed for consistent results and prediction control.

Among the possible extensions is the one adapting the technique to convolutional neural networks such that it provides robustness with respect to adversarial attacks.

Furthermore, using a joint scattering and DNN will inherit the benefits presented with our technique as our layers are the ones closer to the input.

Hence, denoising will benefit the inner layers, the unconstrained standard DNN layers.

Finally, it is possible to perform more consistent best basis selection a la maxout network.

In fact, our thresholding technique can be linked to an optimised ReLU based thresholding.

In this scheme, applying best basis selection based on the empirical risk would thus become equivalent to the pooling operator of a maxout network.

A BUILDING A DEEP CROISÉ SCATTERING NETWORK A.1 CONTINOUS WAVELET TRANSFORM "By oscillating it resembles a wave, but by being localized it is a wavelet".

Yves MeyerWavelets were first introduced for high resolution seismology BID21 and then developed theoretically by Meyer et al. BID14 .

Formally, wavelet is a function ψ ∈ L 2 such that: DISPLAYFORM0 it is normalized such that ψ L 2 = 1.

There exist two categories of wavelets, the discrete wavelets and the continuous ones.

The discrete wavelets transform are constructed based on a system of linear equation.

These equations represent the atom's property.

These wavelet when scaled in a dyadic fashion form an orthonormal atom dictionary.

Withal, the continuous wavelets have an explicit formulation and build an over-complete dictionary when successively scaled.

In this work, we will focus on the continuous wavelets as they provide a more complete tool for analysis of signals.

In order to perform a time-frequency transform of a signal, we first build a filter bank based on the mother wavelet.

This wavelet is names the mother wavelet since it will be dilated and translated in order to create the filters that will constitute the filter bank.

Notice that wavelets have a constant-Q property, thereby the ratio bandwidth to center frequency of the children wavelets are identical to the one of the mother.

Then, the more the wavelet atom is high frequency the more it will be localized in time.

The usual dilation parameters follows a geometric progression and belongs to the following set: DISPLAYFORM1 .

Where the integers J and Q denote respectively the number of octaves, and the number of wavelets per octave.

In order to develop a systematic and general principle to develop a filter bank for any wavelet family, we will consider the weighted version of the geometric progression mentioned above, that is: DISPLAYFORM2 .

In fact, the implementation of wavelet filter bank can be delicate since the mother wavelet has to be define at a proper center frequency such that no artifact or redundant information will appear in the final representation.

Thus, in the section A.3 we propose a principled approach that allows the computation of the filter bank of any continuous wavelet.

Beside, this re-normalized scaled is crucial to the comparison between different continuous wavelet.

Having selected a geometric progression ensemble, the dilated version of the mother wavelet in the time are computed as follows: DISPLAYFORM3 , and can be calculated in the Fourier domain as follows: DISPLAYFORM4 Notice that in practice the wavelets are computed in the Fourier domain as the wavelet transform will be based on a convolution operation which can be achieved with more efficiency.

By construction the children wavelets have the same properties than the mother one.

As a result, in the Fourier domain:ψ λ = 0, ∀λ ∈ Λ .

Thus, to create a filter bank that cover all the frequency support, one needs a function that captures the low frequencies contents.

The function is called the scaling function and satisfies the following criteria: DISPLAYFORM5 Finally, we denote by W x, where W ∈ C N * (J * Q)×N is a block matrix such that each block corresponds to the filters at all scales for a given time.

Also, we denote by S(W x)(λ, t) the reshape operator such that, DISPLAYFORM6 where ψ is the complex conjugate of ψ λ .

Among the continuous wavelets, different selection of mother wavelet is possible.

Each one posses different properties, such as bandwidth, center frequency.

This section is dedicated to the development of the families that are important for the analysis of diverse signals.

The Morlet wavelet FIG3 is built by modulating a complex exponential and a Gaussian window defined in the time domain by, DISPLAYFORM0 where ω 0 defines the frequency plane.

In the frequency domain, we denote it byψ M (t), DISPLAYFORM1 thus, it is clear that ω 0 defines the center frequency of the mother wavelet.

With associated frequency center and standard deviation denoted respectively by ω λi c and ∆ λi ω, ∀j ∈ {0, ..., J * Q − 1} are: DISPLAYFORM2 Notice that for the admissibility criteria ω 0 = 6, however one can impose that zeros-mean condition facilely in the Fourier domain.

Usually, this parameter is assign to the control of the center frequency of the mother wavelet, however in our case, we will see in the section A.3 a simple way to select a mother wavelet close enough to the Nyquist frequency such that all its contracted versions are properly defined.

Then, we are able to vary the parameter ω 0 in order to have different support of Morlet wavelet.

The Morlet wavelet, is optimal from the uncertainty principle point of view BID28 .

The uncertainty principle, when given a time-frequency atoms, is the area of the rectangle of its joint time-frequency resolution.

In the case of wavelet, given the fact that their ratio bandwidth to center frequency is equal implies that this area is equal for the mother wavelets and its scaled versions.

As a result, because of its time-frequency versatility this wavelet is wildly used for biological signals such as bio-acoustic BID5 , seismic traces Chopra* & Marfurt FORMULA0 , EEG DAvanzoa et al. data.

The Gammatone wavelet is a complex-valued wavelet that has been developed by BID36 via a transformation of the real-valued Gammatone auditory filter which provides a good approximation of the basilar membrane filter BID20 .

Because of its origin and properties, this wavelet has been successfully applied for classification of acoustic scene Lostanlen & Andén.

The Gammatone wavelet FIG4 is defined in the time domain by, DISPLAYFORM0 and in the frequency domain by,ψ DISPLAYFORM1 A precise work on this wavelet achieved by V. Lostalnen in BID25 allows us to have an explicit formulation of the parameter σ such that the wavelet can be scaled while respecting the admissibility criteria: DISPLAYFORM2 where ξ is the center frequency and B is the bandwidth parameter.

Notice that B = (1 − 2 DISPLAYFORM3 induce a quasi orthogonal filter bank.

The associated frequency center and standard deviation denoted respectively by ω λi c and ∆ λi ω, ∀j ∈ {0, ..., J * Q − 1} are thus: DISPLAYFORM4 For this wavelet, thanks to the derivation in BID25 , we can manually select for each order m the center frequency and bandwidth of the mother wavelet, which ease the filter bank design.

An important property that is directly related to the auditory response system is the asymmetric envelop, thereby the Gammatone wavelet is not invariant to time reversal to the contrary of the Morlet wavelet that behaves as a Gaussian function.

Thus, for task such as sound classifications, this wavelet provides an efficient filter that will be prone to perceive the sound attack's.

Beside this suitable property for specific analysis, this wavelet is near optimal with respect to the uncertainty principle.

Notice that, when m → ∞ it yields the Gabor wavelet BID11 .

Another interesting property of this wavelet is the causality, by taking into account only the previous and present information, there is no bias implied by some future information and thus it is suitable for real time signal analysis.

The Paul wavelet is a complex-valued wavelet which is highly localized in time, thereby has a poor frequency resolution.

Because of its precision in the time domain, this wavelet is an ideal candidate to perform transient detection.

The Paul wavelet of order m FIG5 is defined in the time domain by, DISPLAYFORM0 and in the frequency domain by, DISPLAYFORM1 With associated frequency center and standard deviation denoted respectively by ω λi c and ∆ λi ω , ∀j ∈ {0, ..., J * Q − 1} are: DISPLAYFORM2 In BID35 they provide a clear and explicit formulation of some wavelet families applied the Paul wavelet in order to capture irregularly periodical variation in winds and sea surface temperatures over the tropical eastern Pacific Ocean .

In addition, it directly represents the phase gradient from a single fringe pattern, yet providing a powerful tool in order to perform optical phase evaluation BID0 .

In the previous section, we defined and develop the properties of several families of wavelets.

Thereby, we can now consider the creation of the filter bank by means of these wavelets.

Notice that we propose a simple manner to obtain the filter bank in the Fourier domain.

Two reasons are at the origin of this choice: first, the wavelet transform is often computed in the Fourier domain because of its efficiency, secondly the wavelets are derived according to geometric progression scales, these scales can directly be represented in the frequency domain, thereby it provided us a way of knowing the position of the wavelet.

However, in the time domain they are not directly quantifiable.

Our systematic framework is based on the intuitive consideration of the problem: we have to select a wavelet, named mother wavelet, that when contracted will create the filter bank derived from the selected scales.

Assuming that the signals we will use are real valued, then the information represented in [−π, 0] and [0, π] are the same if extracted with a symmetric atom.

Now, two kind of wavelets are considered, if the wavelet is complex-valued then its support is in [0, π] , thus the choice of the mother wavelet should be around π and the contracted all along the frequency axis until the total number of octave are covered.

In the case of real-valued wavelet, if the wavelet is not symmetric then it will capture other phase information in the frequency band: [−π, 0] .

Still, the mother wavelet can be selected to be close to π for its positive part, and −π for its negative one.

After defining the routine in order to select the mother wavelet, we propose a simple way to set the position of the mother wavelet.

For each family, the center frequency and standard deviation are derived by finding α such that: DISPLAYFORM0 where λ 0 = α * 2 0/Q denotes the first wavelet position.

Given this equation, one create the mother wavelet such that it avoids capturing elements after the Nyquist frequency and avert the spectral mirror effect and artifacts.

Given the value of α for a wavelet family, one can derive the wavelet filter bank according to the Algorithm 1.

The wavelet filter banks generated by this algorithm for the different families aforementioned can be seen in FIG6 .

Notice that for sake of clarity, the scaling functions are not shown in FIG6 .

Finally, in order to guarantee the admissibility criterion one has to verify that all the wavelets are zeros-mean and square norm one.

The first one is easily imposed by setting the wavelet to be null around ω = 0 as it has been done to efficiently use the Morlet wavelet by Antoine et.

al Farge (1992) ; BID3 .

Then, because of Parseval equality and the energy conservation principle, the second one can be achieved by a re-normalization in the frequency domain of each atom. , where U and S denote respectively the set of selected and unselected wavelet coefficients.

We also define D U such that I = D U + D S .

This estimate corresponds to a thresholding operation in the new basis and the inverse transform of this truncated representation.

<|TLDR|>

@highlight

We propose to enhance the Deep Scattering Network  in order to improve control and stability of any given machine learning pipeline by proposing a continuous wavelet thresholding scheme