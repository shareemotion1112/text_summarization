A key problem in neuroscience and life sciences more generally is that the data generation process is often best thought of as a hierarchy of dynamic systems.

One example of this is in-vivo calcium imaging data, where observed calcium transients are driven by a combination of electro-chemical kinetics where hypothesized trajectories around manifolds determining the frequency of these transients.

A recent approach using sequential variational auto-encoders demonstrated it was possible to learn the latent dynamic structure of reaching behaviour from spiking data modelled as a Poisson process.

Here we extend this approach using a ladder method to infer the spiking events driving calcium transients along with the deeper latent dynamic system.

We show strong performance of this approach on a benchmark synthetic dataset against a number of alternatives.

In-vivo two-photon calcium imaging provides systems neuroscientists with the ability to observe the activity of hundreds of neurons simultaneously during behavioural experiments.

Such highdimensional data is ripe for techniques identifying low-dimensional latent factors driving neural dynamics.

The most common methods, such as principal components analysis, ignore non-linearity and temporal dynamics in brain activity.

Pandarinath et al. (2018) [1] developed a new technique using deep, recurrent, variational auto-encoders which they named Latent Factor Analysis via Dynamical Systems (LFADS).

Using LFADS they found non-linear, dynamic latent variables describing highdimensional activity in the motor cortex that can decode reaching behaviour with much higher fidelity than other methods.

However, LFADS was designed for application to spiking data recorded from extracellular electrodes, not for two-photon calcium imaging data.

Two-photon calcium imaging poses the additional problem of identifying latent spike trains in fluorescence traces.

If we continue to model the frequency of events as being generated by a Poisson process, this can be seen as hierarchy of dynamic systems (Fig 1A) , in which low dimensional dynamics generate spike train probabilities that drive fluctuations in biophysical dynamics of calcium activity (Fig 1B.

Here we propose a method that extends LFADS to accommodate calcium activity using this hierarchical dynamic systems approach.

The model is a variational ladder autoencoder (VLAE) [2] with recurrent neural networks (RNNs) that supports uncovering latent dynamic systems (Fig 1C) .

It can be seen as a unification of two recent applications of variational autoencoders (VAEs) in neuroscience: 1) Latent Factor Analysis for Dynamic Systems (LFADS) [1] and 2) DeepSpike, a VAE approach to inferring spike counts from calcium imaging data [3] .

We choose the VLAE approach since it has been shown to learn disentangled hierarchical features, in contrast to stacked VAEs or ladder VAEs [2, 4] .

The inferred dynamic system underlying the frequency of calcium events in the data is identical to that of LFADS (Fig 1C, blue modules) .

The prior distribution of initial conditions g 0 and external inputs u t are modelled as Gaussian distributions P (g 0 ) = N (µ g0 , σ 2 g0 ), and P (u t ) = N (µ ut , σ 2 ut ).

The underlying dynamic system G(g t , u t ) is modelled by a Gated Recurrent Unit (GRU) taking the initial hidden state g 0 and inputs u t .

Low dimensional factors f t are calculated as a linear transformation of the generator hidden state f t = W f ac g t .

These factors are used to reconstruct the Poisson process intensity function with a fully connected layer and exponential non-linearity

Inferred spike counts s t are generated by sampling z t from Gaussian distributions P (z t ) = N (µ zt , σ 2 zt ) and projecting these through an affine transformation and non-linearity along with the factors from the deeper layer, i.e., Figure 1C blue modules) .

We assume a simple model of calcium dynamics: y t = −y t /τ y +α y s t +β y where the parameters τ y , α y , β y are measured from the data, however it is a topic for future research to fit the calcium dynamics simultaneously.

In our synthetic data, these are valued at 0.4 s, 1, and 0 respectively.

The value of τ y is chosen as it is the known decay time constant of GCamP6, a commonly used calcium fluorescence indicator used in calcium imaging experiments.

The variational posterior distributions Q(z t |x), Q(g 0 |x), Q(u t |x) are modelled as Gaussian distributions, with the mean and standard deviations parameterised by a stack of bidirectional GRUs, E and C 2 are concatenated at each time step t with s t−1 and f t−1 .

Subsequently these concatenated activities are mapped onto the parameters of Q(z(t)|x) and Q(u(t)|x) with fully connected layers.

One of the advantages of using VLAEs is that the evidence lower bound (ELBO) formulation is the same as for VAEs despite the hierarchical latent space [2] .

As such, our cost function remains very similar to that of LFADS.

The likelihood function P (x t |y t ) is modelled as a Gaussian distribution x t ∼ N (y t , σ 2 y ), where σ 2 y is learned.

Although s t is not discrete, P (s t |λ t ) is treated as an approximate Poisson process s t ∼ P oisson(λ t ) = s λt t exp(−λ t )/Γ(s t + 1).

Parameters of our model were optimized with ADAM, with an initial learning rate of 0.01, which decreased by a factor of 0.95 whenever plateaus in training error were detected.

As in LFADS training, KL and L2 terms in the cost function were 'warmed up', i.e., had a scaling factor being 0 and 1 applied that gradually increased.

Warmup for the deeper parameters (blue modules in Figure 1 ) was delayed until warmup for shallower parameters was completed (red modules in Figure 1 ).

The model was tested on synthetic data with Lorenz dynamics embedded in the frequency of calcium fluorescence transients, as described in [5] , where generated spikes were convolved with an exponential kernel with a time constant of 0.4 ms, and white noise added to the resulting traces.

We measure the performance of the model in three ways: 1) uncovering the underlying Lorenz dynamics, 2) reconstructing the rates of calcium transients an inhomogenous Poisson intensity functions, 3) reconstructing the spike counts contributing to increases in the calcium fluorescence signal.

The model was compared against a ground-truth where the spike counts are known, and LFADS is used to reconstruct the latent dynamics and intensity function, and against a model where spike counts are extracted using a deconvolution algorithm [6] before using LFADS to reconstruct the rates and intensity function (OASIS + LFADS).

It was also tested against a model that used a 1-D convolution of the intensity function to reconstruct either the first two (Gaussian-LFADS) or four (Edgeworth-LFADS) time-varying moments of fluorescence, as used previously in estimating the intensity functions of filtered Poisson processes in neuroscience [7] . (Fig 2B) , spikes ( Fig 2C) and Lorenz dynamics (Fig 2D) .

Visually, the model provides very close fit to the fluorescence traces, intensity functions, and Lorenz dynamics.

The model also captures spike-timing, although these spike trains appear smoothed.

Table 1 compares the R 2 goodness-of-fit on reconstructing held-out validation data with ground-truth latent dynamic structure.

Of all approaches, our model easily performs best at reconstructing fluorescence traces, and almost performs as well as LFADS in reconstructing the Lorenz dynamics.

It is to be expected that LFADS performs better, since there is an additional source of observation noise in our synthetic dataset generating fluorescence transients from spikes.

Notably, our model does not perform as well as the deconvolution method OASIS in reconstructing spike trains, however this does not impact the ability of our model to reconstruct the latent dynamics.

In fact, constraining the reconstructed spikes by the latent dynamics may mitigate any errors in spike train reconstruction that occur by deconvolution, since the deconvolution algorithm may erroneously drop spikes during high rates, whereas our model should be less likely to do so.

It will be necessary to assess this possibility further.

It should be noted that the deconvolution algorithm performs much better at reconstructing spike trains in our synthetic dataset than in real datasets where ground-truth spiking is known [8] .

To our knowledge, there are no known dual recordings of population 2-photon calcium imaging with ground-truth electrophysiology in a subpopulation of neurons in-vivo during behaviour driven by hypothesized low-dimensional dynamics that we would be able to validate this with.

Nevertheless, since the relationship between calcium dynamics and somatic spiking is highly non-linear, especially in dendrites, it remains to be seen how useful it is to faithfully reproduce unseen spike trains in calcium fluorescence activity.

<|TLDR|>

@highlight

We propose an extension to LFADS capable of inferring spike trains to reconstruct calcium fluorescence traces using hierarchical VAEs.