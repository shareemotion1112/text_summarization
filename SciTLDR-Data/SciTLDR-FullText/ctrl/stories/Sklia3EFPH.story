Anatomical studies demonstrate that brain reformats input information to generate reliable responses for performing computations.

However, it remains unclear how neural circuits encode complex spatio-temporal patterns.

We show that neural dynamics are strongly influenced by the phase alignment between the input and the spontaneous chaotic activity.

Input alignment along the dominant chaotic projections causes the chaotic trajectories to become stable channels (or attractors), hence, improving the computational capability of a recurrent network.

Using mean field analysis, we derive the impact of input alignment on the overall stability of attractors formed.

Our results indicate that input alignment determines the extent of intrinsic noise suppression and hence, alters the attractor state stability, thereby controlling the network's inference ability.

Brain actively untangles the input sensory data and fits them in behaviorally relevant dimensions that enables an organism to perform recognition effortlessly, in spite of variations DiCarlo et al. (2012) ; Thorpe et al. (1996) ; DiCarlo & Cox (2007) .

For instance, in visual data, object translation, rotation, lighting changes and so forth cause complex nonlinear changes in the original input space.

However, the brain still extracts high-level behaviorally relevant constructs from these varying input conditions and recognizes the objects accurately.

What remains unknown is how brain accomplishes this untangling.

Here, we introduce the concept of chaos-guided input alignment in a recurrent network (specifically, reservoir computing model) that provides an avenue to untangle stimuli in the input space and improve the ability of a stimulus to entrain neural dynamics.

Specifically, we show that the complex dynamics arising from the recurrent structure of a randomly connected reservoir Rajan & Abbott (2006) ; Kadmon & Sompolinsky (2015) ; Stern et al. (2014) can be used to extract an explicit phase relationship between the input stimulus and the spontaneous chaotic neuronal response.

Then, aligning the input phase along the dominant projections determining the intrinsic chaotic activity, causes the random chaotic fluctuations or trajectories of the network to become locally stable channels or dynamic attractor states that, in turn, improve its' inference capability.

In fact, using mean field analysis, we derive the effect of introducing varying phase association between the input and the network's spontaneous chaotic activity.

Our results demonstrate that successful formation of stable attractors is strongly determined from the input alignment.

We also illustrate the effectiveness of input alignment on a complex motor pattern generation task with reliable generation of learnt patterns over multiple trials, even in presence of external perturbations.

We describe the effect of chaos guided input alignment on a standard firing-rate based reservoir model of N interconnected neurons.

Specifically, each neuron in the network is described by an

where r i (t) = φ(x i (t)) represents the firing rate of each neuron characterized by the nonlinear response function, φ(x) = tanh(x) and τ = 10ms is the neuron time constant.

W represents a sparse N × N recurrent weight matrix (with W ij equal to the strength of the synapse connecting unit j to unit i) chosen randomly and independently from a Gaussian distribution with 0 mean and variance, g 2 /p c N Van Vreeswijk et al. (1996); van Vreeswijk & Sompolinsky (1998) , where g is the synaptic gain parameter and p c is the connection probability between units.

The output unit z reads out the activity of the network through the connectivity matrix, W Out , with initial values drawn from a Gaussian distribution with 0 mean and variance 1/N .

The readout weights are trained using Recursive Least Square (RLS) algorithm Laje & Buonomano (2013) ; Sussillo & Abbott (2009); Jaeger & Haas (2004) .

The input weight matrix, W Input , is drawn from a Gaussian distribution with zero mean and unit variance.

The external input, I, is an oscillatory sinusoidal signal, I = I 0 cos(2πf t + χ), with amplitude I 0 , frequency f , that is the same for each unit i.

Here, we use a phase factor χ chosen randomly and independently from a uniform distribution between 0 and 2π.

This ensures that the spatial pattern of input is not correlated with the recurrent connectivity initially.

Through input alignment analysis, we then obtain the optimal phases to project the inputs in the preferred direction of the network's spontaneous or chaotic activity.

In all our simulations (without loss of generality), throughout the paper we have assumed, p c = 0.1, N = 800, g = 1.5, f = 10Hz, unless, specified otherwise.

First, we ask the question how is the subspace of input driven activity aligned with respect to the subspace of spontaneous or chaotic activity of a recurrent network.

Using Principal Component Analysis (PCA), we observed that the input-driven trajectory converges to a uniform shape becoming more circular with increasing input amplitude (See Appendix A, Fig. A1 (a,b) ).

We utilize the concept of principal angles, introduced in Rajan et al. (2010a); Ipsen & Meyer (1995) , to visualize the relationship between the chaotic and input driven (circular) subspace.

Specifically, for two subspaces of dimension D 1 and D 2 defined by unit principal component vectors (that are mutually

Fig. 1 (a) schematically represents the angle between the circular input driven network activity and the irregular spontaneous chaotic activity.

Here, θ chaos (and θ driven ) refers to the subspace defined by the first two Principal Components (PCs) of the intrinsic chaotic activity (and input driven activity).

It is evident that rotating the circular orbit by θ rotate will align it along the chaotic trajectory projection.

We observe that aligning the inputs in directions (along dominant PCs) that account for maximal variance in the chaotic spontaneous activity facilitates intrinsic noise suppression at relatively low input amplitudes, thereby, allowing the network to produce stable trajectories.

For instance, instead of using random phase input, we set I = I 0 cos(2πf t + Θ) and visualize the network activity as shown in Fig. 1 (b) .

Even at lower amplitude of I 0 = 1.5, we observe a uniform circular orbit (in the PC subspace) for the network activity that is characteristic of reduction in intrinsic noise and input sensitization.

In fact, even after the input is turned off after t = 50ms, the neural units yield stable and synchronized trajectories with minimal variation across different trials ( Fig.  1 (b, Right)) in comparison to the random phase input driven network (of higher amplitude).

Note, Appendix A ( Fig. A1 (b) ) shows an example of PC activity for random phase input driven reservoir driven with high input amplitude I 0 = 5.

This shows the effectiveness of subspace alignment for intrinsic noise suppression.

In addition, working in low input-amplitude regimes offers an additional advantage of higher network dimensionality (refer to Appendix A Fig. A1 (c) for dimensionality discussion), that in turn improves the overall discriminative ability of the network.

Note, previous work Rajan et al. (2010a; b) have shown that spatial structure of the input does not have a keen influence on the spatial structure of the network response.

Here, we bring in this association explicitly with subspace alignment.

Θ, in the above analysis, is the input phase that corresponds to a subspace rotation of driven activity toward spontaneous chaotic activity.

We observe that the temporal phase of the input contributes to the neuronal activity in a recurrent network.

Fig. 1 (c) illustrates this correlation wherein the input phase determines the orientation of the input-driven circular orbit with respect to the dominant sub-space of intrinsic chaotic activity.

For a given input frequency (f = 10Hz), input phase, Θ = 83.2

• , aligns the driven activity (θ driven ) along the chaotic activity (θ chaos ) resulting in θ rotate = 0

• for varying input amplitude (I 0 = 1.5, 3).

An interesting observation here is that the frequency of the input modifies the orientation of the evoked response that yields different input phases at which θ chaos and θ driven are aligned (refer to Fig. 1 (c, Right) ).

We also observe that the subspace alignment is extremely sensitive toward the input phase in certain regions with abrupt jumps and non-smooth correlation.

This non-linear behavior is a consequence of the recurrent connectivity that overall shapes the complex interaction between the driving input and the intrinsic dynamics.

While this correlation yields several important implications for neuro-biological computational experiments DiCarlo et al. (2012) ; Thorpe et al. (1996) ; DiCarlo & Cox (2007) , we utilize this behavior for subspace alignment.

Consequently, in all our experiments, for a given θ rotate , we find a corresponding input phase Θ that approximately aligns the input in the preferred direction.

Next, we describe the implication of input alignment along the chaotic projections on the overall learning ability of the network.

First, we trained a recurrent network ( Fig. 2 (a)) with two output units to generate a timed response at t = 1s as shown in Fig. 2 (a, Right).

Two distinct and brief sinusoidal inputs (of 50 ms duration and amplitude I 0 = 1.5) were used to stimulate the recurrent network.

The network trajectories produced were then mapped to the output units using RLS training (to learn the weights W Out ).

Here, the network (after readout training) is expected to produce timed output dynamics at readout unit 1 or 2 in response to input I 1 or I 2 , respectively.

The network is reliable if it generates consistent response at the readout units across repeated presentations of the inputs during testing, across different trials.

This simple experiment utilizes the fact that neural dynamics in a recurrent network implicitly encode timing that is fundamental to the processing and generation of complex spatio-temporal patterns.

Note, in such cases of multiple inputs, values of both inputs are zero, except for a timing window during which one input is briefly turned on in a given trial.

Since both the inputs, in the above experiment, have same amplitude and frequency dynamics, the circular orbit describing the network activity in the input-driven state (for both inputs) is almost similar giving rise to one principal angle (θ driven1,2 in Fig. 2 (b) ) for the input subspace.

To discriminate between the output responses for the two inputs, it is apparent that the inputs have to be aligned in different directions.

One obvious approach is to align each input along two different principal angles defining the chaotic spontaneous activity (i.e. I 1 along ∠P C1.P C2 and I 2 along ∠P C3.P C4).

Note, ∠P C1.P C2 denotes the angle θ calculated using Eqn.

2.

Another approach is to align I 1 along ∠P C1.P C2 ≡ θ chaos and I 2 along ∠P C1.P C2 + 90

• ≡ θ chaos,90 • as shown in Fig. 2 (b) .

We analyze the latter approach in detail as it involves input phase rotation in one subspace that makes it easier for formal theoretical analysis.

To characterize the discriminative performance of the network, we evaluated the Euclidean distances to measure the inter-and intra-input trajectories.

Inter-input trajectory distance is measured in response to different inputs (I 1 , I 2 ).

Intra-input trajectory distance is measured in response to the clean input (say, I 1 = I 0 cos(2πf t + Θ 1 )) and a slightly varied version of the same input (for instance, I 1,δ = (I 0 + )cos(2πf t + Θ 1 ).

Here, is a random number between [0, 0.5]) and Θ 1 (Θ 2 ) is the input phase that aligns I 1 (I 2 ) along θ chaos (θ chaos,90 • ).

The Euclidean distance is calculated as

, where r 1 (t) (r 2 (t)) is the firing rate activity of the network corresponding to I 1 (I 2 ).

Note, for intra-input evaluation (say, for I 1 ), the distance is measured between firing rates corresponding to inputs I 1 and I 1,δ .

The inter-/intra-input trajectory distances are plotted in Fig. 2 (c) for scenarios-with and without input alignment.

It is desirable to have larger inter-trajectory distance and small intra-trajectory distance such that the network easily distinguishes between two inputs while being able to reproduce the required output response even when a particular input is slightly perturbed.

We observe that aligning the inputs in direction parallel and perpendicular to the dominant projections (i.e. I 1 along θ chaos , I 2 along θ chaos,90 • as in Fig. 2 (c, Middle) ) increases the inter-trajectory distance compared to the non-aligned case (Fig. 2 (c, Left) ) while decreasing the intra-input trajectory separation.

This further ascertains the fact that subspace alignment reduces intrinsic fluctuations within a network thereby enhancing its discrimination capability.

Note, without input alignment, the intrinsic fluctua-tions cannot be overcome with low-amplitude inputs (I 0 = 1.5).

Hence, for fair comparison and to obtain stable readout-trainable trajectory in the non-aligned case, we use a higher input amplitude of I 0 = 3.

We hypothesize that intrinsic noise suppression occurs as input subspace alignment along dominant projections (that account for maximal variance such as P C1, P C2) causes chaotic trajectories along different directions (in this case, along θ chaos , θ chaos,90 • ) to become locally stable channels or attractor states.

These attractors behave as potential wells (or local minima from an optimization standpoint) toward which the network activity converges for different inputs.

Thus, the successful formation of stable yet distinctive attractors for different inputs are strongly influenced by the orientation along which the inputs are aligned.

As a consequence of our hypothesis, depending upon the orientation of the input with respect to the dominant chaotic activity (θ chaos in Fig. 2 (b) ), the extent of noise suppression will vary that will eventually alter the stability of the attractor states.

To test this, we rotated I 2 (from θ chaos,90 • ) further by 90

• (θ chaos,180 • in Fig. 2 (b) ) and monitored the intra-trajectory distance.

In Fig. 2 (c, Middle) corresponding to 90

• phase difference between I 1 , I 2 (I 1 along θ chaos , I 2 along θ chaos,90 • ), I 2 corresponds to a more stable attractor since its intradistance is lower than I 1 .

In contrast, in Fig. 2 (c, Right) corresponding to 180

• phase difference (I 1 along θ chaos , I 2 along θ chaos,180 • ), I 1 turns out be more stable than I 2 .

Note, the 90

• , 180

• phase difference between I 1 , I 2 (mentioned above and in the remainder of the paper) refers to the phase difference between the inputs in the chaotic subspace after subspace alignment using Θ. For our analysis,

• phase between I 1 , I 2 in chaotic subspace, while

In addition to the trajectory distance, visualizing the network activity in the 3-D PC space ( Fig. 2  (d) ), also, shows the influence of input orientation (and hence the phase correlation) toward formation of distinct attractor states.

Since I 1 , I 2 are aligned in the subspace defined by ∠P C1.P C2, the 2D projection of the circular orbit onto PC1 and PC2 in both input aligned scenarios (90 • , 180

• phase) are comparable.

However, the third dimension, PC3, marks the difference between the two input projections.

In fact, the progress of the network activity as time evolves (shown by dashed arrows in Fig. 2 (d) ) follows a completely different cycle for the input aligned scenarios.

The change in the overall rotation cycle from anti-clockwise (I 2 with 90 • phase, Fig. 2 (d, Middle) ) to clockwise (I 2 with 180

• phase, Fig. 2 (d, Right) ) can be viewed as an indication toward the altering of the attractor state stability.

On the other hand, the non-aligned case with I 0 = 3 yields incoherent and more random trajectory (Fig. 2 (d, Left) representative of intrinsic noise.

In order to get more coherent activity and to suppress the noise further, we need to increase the input amplitude to I 0 ≥ 5 as shown in Appendix A (Fig. A1 (a,b) ).

To explain the above results analytically, we use mean-field methods developed to evaluate the properties of random network models in the limit N → ∞ Rajan et al. (2010b) ; Sompolinsky et al. (1988) .

A key quantity in Mean Field Theory (MFT) is the average autocorrelation function that characterizes the interaction within the network as

where <> denotes the time average.

The main idea of MFT is to replace the network interaction term in Eqn.

1 by Gaussian noise η such that

x i 0 (t) = Acos(2πf t + ζ) with A = I 0 / 1 + (2πf t) 2 .

Here, ζ incorporates the averaged temporal phase relationship between the reservoir neurons and the input induced by input subspace alignment, ζ(θ rotate ) = Θ. The temporal correlation of η is calculated self-consistently from C(τ ).

For selfconsistence, the first and second moment of η must match the moments of the network interaction term.

Thus, we get < η i (t) >= 0 as mean of the recurrent synaptic matrix < W ij >= 0.

For calculating the second moment , we use the identity < W ij W kl >= g 2 δ ij δ kl /N and obtain < η i (t)η j (t + τ ) >= g 2 C(τ ).

Combining this result with the MFT noise-interaction based network equation yields where ∆(τ ) =< x i 1 (t)x i 1 (t+τ ) >.

Eqn.

4 resembles the Newtonian motion equation of a classical particle moving under the influence of force given by the right hand side of the equation.

This force depends on C that, in turn, depends on the input subspace alignment (ζ) which directs the initial position of the particle (or state of the network ∆(0)).

From this analogy, it is evident that analyzing the overall potential energy function of the particle (or network) will be equivalent to visualizing the different attractor states formed in a network in response to a particular input stimulus.

Thus, we formulated an expression for the correlation function (with certain constraints) using Taylor series expansion, that allows us to derive the force and hence the dynamics of the network under various input alignment conditions.

The non-linear firing rate function r(x) = φ(x) = tanh(gx) can be expanded with Taylor series for small values of g, i.e. g = 1 + δ, where δ denotes a small increment in g beyond 1.

Note, g = 1 + δ satisfies the criterion, g > 1 Rajan & Abbott (2006); Sompolinsky et al. (1988) , to operate the networks in chaotic regime.

Also, the overall network statistics does not change with g being expressed as a gain factor in the firing-rate function instead of overall synaptic strength.

Using tanh(gx) gx − 1/3g 3 x 3 + 2/15g 5 x 5 , we can express C(τ ) from Eqn.

3 as

Now, we can express Eqn.

4 as

Writing l = kδ due to the small limit of g, Eqn.

4 simplifies to

where G = g 2 A 2 cos(ωτ + 2ζ)/(2δ 3 ) and n is a parameter defined in terms of m, δ.

Appendix B provides a detailed derivation of Eqn.

5 and comments about the assumptions on initial conditions.

Note, Eqn.

5 is an approximate version of Eqn.

4 that depicts network activity in the manner of Newtonian motion independent of all intrinsic time (or averaging parameters) while taking into account the influence of input alignment.

Now, we can express the potential of the network driven by a force, F , equivalent to the right hand side of Eqn.

5 as

We solve Eqn.

5, 6 with initial conditions k(0) = 1,k(0) = 0 and monitor the change in force, F , and potential, V , for different values of G. First, let us examine the attractor state formation when there is no input stimulus (i.e. G = 0) by visualizing the potential V .

For G = 0, the expressions for force and potential become

Fig .

3 shows the evolution of potential energy as k varies for different G. When input G = 0, the network dynamics is chaotic that results in the formation of potential wells that are both equally stable.

The network activity will thus converge to any one of these wells (that can be interpreted as attractor states) depending upon the initial state or starting point.

This supports the observation that a network with no input yields chaotic activity with incoherent and irregular trajectory for every trial (see Fig. A1 (a) in Appendix A for reference).

For nonzero G, the force (and potential) equation will be dependent on ζ since G cos(ωτ + 2ζ).

For different values of ζ, we solved for V (Eqn.

6) numerically and plotted the potential evolution as shown in Fig. 3 .

For ζ = π/4, the potential well is more attractive on the left end.

This validates the fact that intrinsic fluctuations are suppressed in the presence of an input.

For ζ = π/2, the left attractor becomes more stable.

Changing ζ further shows that the potential well on the right end becomes more stable.

This result confirms that input subspace alignment with respect to the initial chaotic state influences the overall stability and convergence capability of a recurrent network.

The fact that stability corresponding to different attractor states (ζ = π/2, π) arises, qualifies our earlier hypothesis that input orientation with respect to the chaotic subspace alters the attractor state stability, corroborating the result of Fig. 2 (c) .

Note, we solved Eqn.

6 by setting some initial and boundary value conditions on k and by iterating over different n until we reached a steady state solution.

Changing these conditions will result in a completely new set of ζ values (different from those in Fig. 3 ).

Nevertheless, we will observe a similar evolution of the potential well and change in attractor state stability as Fig. 3 .

Furthermore, the MFT calculations use ζ to denote a functional relationship between subspace alignment and input phase that eventually affects the attractor state stability.

In the future, we will examine the real-time evaluation of ζ and its' impact on the analytical studies.

Finally, the constraint under which we derive the potential energy functions and show the altering of attractor state is g = 1 + δ.

We expect all our results to be valid for large g as well since Eqn.

4 (that was simplified with Taylor expansion) still remains unchanged.

Finally, we illustrate the effectiveness of input alignment on a complex motor pattern generation task Laje & Buonomano (2013) .

We trained a recurrent network to generate the handwritten words "chaos" and "neuron" in response to two different inputs 1 Laje & Buonomano (2013) .

After obtaining the principal angle of the chaotic spontaneous activity, we aligned the input I 1 corresponding to "chaos" along ∠P C1.P C2 using optimal input phase Θ. Then, we monitored the output activity for different orientation (i.e. 90

• , 180 • ) of input I 2 , corresponding to "neuron", with respect to I 1 in the chaotic subspace.

The two output units (representing the x and y axes) were trained using RLS to trace the original target locations (x(t), y(t)) of the handwritten patterns at each time instant.

Fig.  4 (a) shows the handwritten patterns generated by the network across 10 test trials for the scenario when inputs are aligned at 90

• in the chaotic subspace.

We observe similar robust patterns generated for 180

• phase as well.

The notable feature of input alignment is that the chaotic trajectories become locally stable channels and function as dynamic attractor states.

However, external perturbation can induce more chaos in the reservoir that will overwhelm the stable patterns of activity.

To test the susceptibility of the dynamic attractor states formed with input alignment to external perturbation, we introduced random Gaussian noise onto a trained model along with the standard intrinsic chaos-aligned inputs during testing.

The injection of noise alters the net external current received by the neuronal units (I = Σ i [W Input i I + N 0 rand(i)], where N 0 is the noise amplitude, i denotes a neural unit in the reservoir and rand is the random Gaussian distribution).

Fig. 4  (b) shows the mean squared error (calculated as the average Euclidean distance between the target (x, y) and the actual output produced at different time instants, averaged across 20 test trials) of the network for varying levels of noise.

As N 0 increases, we observe a steady increase in the error value implying degradation in the prediction capability of the network.

However, for moderate noise (with N 0 < 0.01), the network exhibits high robustness with negligible degradation in prediction capability for both the words.

Interestingly, for 90

• phase difference, "neuron" is more stable than "chaos" with increased reproducibility across different trials even with more noise (N 0 = 0.2).

In contrast, for 180

• phase, "chaos" is less sensitive to noise (Fig. 4 (b) ).

On the other hand, for 45

• phase alignment between I 1 , I 2 in the chaotic subspace, we observe that the network is sensitive even toward slight perturbation (N 0 = 0.001).

This implies that the attractor states formed, in this case, are very unstable.

This further corroborates the fact that the extent of noise suppression and hence the attractor state stability varies based upon the input alignment.

Fig. 4(c) shows the handwritten pattern generated in one test trial for different phase alignment between I 1 , I 2 , when I 1 is aligned along the principal angle defining the spontaneous chaotic activity of the network.

It is noteworthy to mention that the neural trajectories of the recurrent units corresponding to both cases are stable.

In fact, we observe in the 90

• case, the trajectories of neurons responding to I 1 that corresponds to output "chaos" become slightly divergent and incoherent beyond 1000ms.

In contrast, the trajectories of units responding to the word "neuron" are more synergized and coherent throughout the 1500ms time period of simulation.

This indicates that the network activity for "neuron" converges to a more stable attractor state than "chaos".

As a result, we see that the network is more robust while reproducing "neuron" even in presence of external perturbation (N 0 , noise amplitude is 0.2).

In the 180

• phase difference case, we see exactly opposite stability phenomena with "chaos" converging to more stable attractor.

Models of cortical networks often use diverse plasticity mechanisms for effective tuning of recurrent connections to suppress the intrinsic chaos (or fluctuations) Laje & Buonomano (2013) ; Panda & Roy (2017) .

We show that input alignment alone produces stable and repeatable trajectories, even, in presence of variable internal neuronal dynamics for dynamical computations.

Combining input alignment with recurrent synaptic plasticity mechanism can further enable learning of stable correlated network activity at the output (or readout layer) that is resistant to external perturbation to a large extent.

Furthermore, since input subspace alignment allows us to operate networks at low amplitude while maintaining a stable network activity, it provides an additional advantage of higher dimensionality.

A network of higher dimensionality offers larger number of disassociated principal chaotic projections along which different inputs can be aligned (see Appendix A, Fig. A1(c) ).

Thus, for a classification task, wherein the network has to discriminate between 10 different inputs (of varying frequencies and underlying statistics), our notion of untangling with chaos-guided input alignment can, thus, serve as a foundation for building robust recurrent networks with improved inference ability.

Further investigation is required to examine which orientations specifically improve the discrimination capability of the network and the impact of a given alignment on the stability of the readout dynamics around an output target.

In summary, the analyses we present suggest that input alignment in the chaotic subspace has a large impact on the network dynamics and eventually determines the stability of an attractor state.

In fact, we can control the network's convergence toward different stable attractor channels during its voyage in the neural state space by regulating the input orientation.

This indicates that, besides synaptic strength variance Rajan & Abbott (2006) , a critical quantity that might be modified by modulatory and plasticity mechanisms controlling neural circuit dynamics is the input stimulus alignment.

To examine the structure of the recurrent network's representations, we visualize and compare the neural trajectories in response to varying inputs using Principal Component Analysis (PCA) Rajan et al. (2010a) .

The network state at any given time instant can be described by a point in the Ndimensional space with coordinates corresponding to the firing rates of the N neuronal units.

With time, the network activity traverses a trajectory in this N-dimensional space and we use PCA to outline the subspace in which this trajectory lies.

To conduct PCA, we diagonalize the equal-time cross-correlation matrix of the firing rates of the N units as

where the angle brackets, <>, denote time average and r(t) denotes the firing rate activity of the neuron.

The eigenvalues of the matrix D (specifically, λ a / N a=1 λ a , where λ a is the eigenvalue corresponding to principal component a) indicate the contribution of different Principal Components (PCs) toward the fluctuations/total variance in the spontaneous activity of the network.

Fig. A1 shows the impact of varying input amplitude (I 0 ) on the spontaneous chaotic activity of the network.

For I 0 = 0, the network is completely chaotic as is evident from the highly variable projections of the network activity onto different Principal Components (PCs) (see Fig. A1 (a, Left) ).Generally, the leading 10−15% (depending upon the value of g) of the PCs account for ∼ 95% of the network's chaotic activity Rajan et al. (2010a) .

Visualizing the network activity in a 3D space composed of the dominant principal components (PC1, 2, 3) shows a random and irregular trajectory characteristic of chaos (Fig. A1 (a, Middle) ).

In fact, plotting the trajectories (firing rate r(t) of the neuron as time evolves) of 5 recurrent units in the network (Fig. A1 (a, Right) ) shows diverging and incoherent activity across 10 different trials, also, representative of intrinsic chaos.

In addition, the projections of the network activity onto components with smaller variances, such as PC50, fluctuate more rapidly and irregularly ( Fig. A1 (a, Left) ).

This further corroborates the fact that the leading PCs (such as, PC1-PC15) define a network's spontaneous chaotic activity.

Driving the recurrent network with a sinusoidal input of high amplitude (Fig. A1 (b) ) sensitizes the network toward the input, thereby, suppressing the intrinsic chaotic fluctuations.

The PC projections of the network activity are relatively periodic.

A noteworthy observation here is that the trajectories of the recurrent units (Fig. A1 (b, Right) become more stable and consistent across 10 different presentations of the input pattern with increasing amplitude.

A readout layer appended to a recurrent network can be easily trained on these stable trajectories for a particular task.

Thus, the input amplitude determines the network's encoding trajectories and in turn, its' inference ability.

In fact, the chaotic intrinsic activity is completely suppressed for larger inputs.

However, this is not preferred as input dominance drastically declines the discriminative ability of a network that can be justified by dimensionality measurements.

The effective dimensionality of a reservoir is calculated as

−1 that provides a measure of the effective number of PCs describing a network's activity for a given input stimulus condition.

Fig. A1 (c) illustrates how the effective dimensionality decreases with increasing input amplitude for different g values.

It is, hence, critical that input drive be strong enough to influence network activity while not overriding the intrinsic chaotic dynamics to enable the network to operate at the edge of chaos.

Note, higher g in Fig. A1 (c) yields a larger dimensionality due to richer chaotic activity.

In our simulations in Fig. A1 (b) , the input is shown for 50ms starting at t = 0.

Thus, we observe that the trajectories of the recurrent units are chaotic until the input is turned on.

Although the network returns to spontaneous chaotic fluctuations when the input is turned off (at t = 50ms), we observe that the network trajectories are stable and non-chaotic that is in coherence with the previous findings from Bertschinger & Natschläger (2004) ; Rajan et al. (2010b) .

From the visualization of network activity in the dominant PC space, we see that the input-driven trajectory converges to a uniform shape becoming more circular (along PC1 and PC2 dimensions) with higher input amplitude (Fig. A1 (b, Middle) ).

This informs us that the orbit describing the network activity in the input-driven state consists of a circle in a two-dimensional subspace of the full N-dimensional hyperspace of the neuronal activities (supporting the schematic depiction of driven and chaotic subspaces in Fig. 1 (a) ).

Note, all simulations in Appendix are conducted with similar parameters mentioned in the main text, i.e., N = 800, f = 10Hz, p c = 0.1.

First, let us solve for x i 1 such that we can get an expression for the correlation function, C(τ ) in Eqn.

3 of main text.

Noting that, x i 1 is driven by Gaussian noise (as indicated by the MFT noiseinteraction equation:

, we can assume their moments as < x i 1 (t) >=< x i 1 (t + τ ) >= 0, < x i 1 (t)x i 1 (t) >=< x i 1 (t + τ )x i 1 (t + τ ) >= ∆(0) and < x i 1 (t)x i 1 (t + τ ) >= ∆(τ ).

x 1 (t) (dropping index i as all neuronal variables have similar statistics) can then be written as x 1 (t) = αz 1 + βz 3 ; x 1 (t + τ ) = αz 2 + γz 3

where z 1 , z 2 , z 3 are Gaussian random variables with 0 mean/unit variance and α = ∆(0) − |∆(τ )|, β = sgn(∆(τ )) |∆(τ )|, γ = |∆(τ )|.

Now, writing x = x 0 + x 1 , C is computed by integrating over z 1 , z 2 , z 3 as

<< φ(x i 0 (t) + αz 1 + βz 3 ) > z1 < φ(x i 0 (t + τ )) + αz 2 + γz 3 > z2 > z3 (10)

where < f (z)

for z = z 1 , z 2 , z 3 .

Now, x i 0 (t) = Acos(2πf t + ζ),

where A = I 0 / 1 + (2πf t) 2 (solve dxi 0 dt = −x i 0 + I 0 cos(2πf t + ζ) for x i 0 ) and ζ incorporates the averaged temporal phase relationship between the individual neurons and the input induced by input subspace alignment.

Replacing the value of x i 0 in Eqn.

10, we get

<<< φ(Acos(ζ) + αz 1 + βz 3 ) > z1 ) < φ(Acos(ωτ + ζ) + αz 2 + γz 3 > z2 > z3 ) > ζ

The above correlation function also satisfies Eqn.

4 of main text.

Note, ω = 2πf in Eqn.

11.

Now we solve Eqn.

11 using Taylor series approximation for tanh(gx) = gx−1/3g 3 x 3 +2/15g 5 x 5 .

We have φ(Acos(ζ) + αz 1 + βz 3 ) = g(Acos(ζ) + αz 1 + βz 3 ) − 1/3g 3 (αz 1 + βz 3 ) 3 + 2/15g 5 (αz 1 + βz 3 )

<|TLDR|>

@highlight

Input Structuring along Chaos for Stability