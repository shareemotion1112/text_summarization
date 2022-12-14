Equilibrium Propagation (EP) is a learning algorithm that bridges Machine Learning and Neuroscience, by computing gradients closely matching those of Backpropagation Through Time (BPTT), but with a learning rule local in space.

Given an input x and associated target y, EP proceeds in two phases: in the first phase neurons evolve freely towards a first steady state; in the second phase output neurons are nudged towards y until they reach a second steady state.

However, in existing implementations of EP, the learning rule is not local in time: the weight update is performed after the dynamics of the second phase have converged and requires information of the first phase that is no longer available physically.

This is a major impediment to the biological plausibility of EP and its efficient hardware implementation.

In this work, we propose a version of EP named Continual Equilibrium Propagation (C-EP) where neuron and synapse dynamics occur simultaneously throughout the second phase, so that the weight update becomes local in time.

We prove theoretically that, provided the learning rates are sufficiently small, at each time step of the second phase the dynamics of neurons and synapses follow the gradients of the loss given by BPTT (Theorem 1).

We demonstrate training with C-EP on MNIST and generalize C-EP to neural networks where neurons are connected by asymmetric connections.

We show through experiments that the more the network updates follows the gradients of BPTT, the best it performs in terms of training.

These results bring EP a step closer to biology while maintaining its intimate link with backpropagation.

A motivation for deep learning is that a few simple principles may explain animal intelligence and allow us to build intelligent machines, and learning paradigms must be at the heart of such principles, creating a synergy between neuroscience and Artificial Intelligence (AI) research.

In the deep learning approach to AI (LeCun et al., 2015) , backpropagation thrives as the most powerful algorithm for training artificial neural networks.

Unfortunately, its implementation on conventional computer or dedicated hardware consumes more energy than the brain by several orders of magnitude (Strubell et al., 2019) .

One path towards reducing the gap between brains and machines in terms of power consumption is by investigating alternative learning paradigms relying on locally available information, which would allow radically different hardware implementations: such local learning rules could be used for the development of extremely energy efficient learning-capable hardware.

Investigating such bioplausible learning schemes with real-world applicability is therefore of interest not only for neuroscience, but also for developing neuromorphic computing hardware that takes inspiration from information-encoding, dynamics and topology of the brain to reach fast and energy efficient AI (Ambrogio et al., 2018; Romera et al., 2018) .

In these regards, Equilibrium Propagation (EP) is an alternative style of computation for estimating error gradients that presents significant advantages (Scellier and Bengio, 2017) .

EP belongs to the family of contrastive Hebbian learning (CHL) algorithms (Ackley et al., 1985; Movellan, 1991; Hinton, 2002) and therefore benefits from an important feature of these algorithms: neural dynamics and synaptic updates depend solely on information that is locally available.

As a CHL algorithm, EP applies to convergent RNNs, i.e. RNNs that are fed by a static input and converge to a steady state.

Training such a convergent RNN consists in adjusting the weights so that the steady state corresponding to an input x produces output values close to associated targets y. CHL algorithms proceed in two phases: in the first phase, neurons evolve freely without external influence and settle to a (first) steady state; in the second phase, the values of output neurons are influenced by the target y and the neurons settle to a second steady state.

CHL weight updates consist in a Hebbian rule strengthening the connections between co-activated neurons at the first steady state, and an anti-Hebbian rule with opposite effect at the second steady state.

A difference between Equilibrium Propagation and standard CHL algorithms is that output neurons are not clamped in the second phase but elastically pulled towards the target y.

A second key property of EP is that, unlike CHL and other related algorithms, it is intimately linked to backpropagation.

It has been shown that synaptic updates in EP follow gradients of recurrent backpropagation (RBP) and backpropagation through time (BPTT) (Ernoult et al., 2019) .

This makes it especially attractive to bridge the gap between neural networks developed by neuroscientists, neuromorphic researchers and deep learning researchers.

Nevertheless, the bioplausibility of EP still undergoes two major limitations.

First, although EP is local in space, it is non-local in time.

In all existing implementations of EP the weight update is performed after the dynamics of the second phase have converged, when the first steady state is no longer physically available.

Thus the first steady state has to be artificially stored.

Second, the network dynamics have to derive from a primitive function, which is equivalent to the requirement of symmetric weights in the Hopfield model.

These two requirements are biologically unrealistic and also hinder the development of efficient EP computing hardware.

In this work, we propose an alternative implementation of EP (called C-EP) which features temporal locality, by enabling synaptic dynamics to occur throughout the second phase, simultaneously with neural dynamics.

We then address the second issue by adapting C-EP to systems having asymmetric synaptic connections, taking inspiration from Scellier et al. (2018) ; we call this modified version C-VF.

More specifically, the contributions of the current paper are the following:

??? We introduce Continual Equilibrium Propagation (C-EP, Section 3.1-3.2), a new version of EP with continual weight updates: the weights of the network are adjusted continually in the second phase of training using local information in space and time.

Neuron steady states do not need to be stored after the first phase, in contrast with standard EP where a global weight update is performed at the end of the second phase.

Like standard EP, the C-EP algorithm applies to networks whose synaptic connections between neurons are assumed to be symmetric and tied.

??? We show mathematically that, provided that the changes in synaptic strengths are sufficiently slow (i.e. the learning rates are sufficiently small), at each time step of the second phase the dynamics of neurons and synapses follow the gradients of the loss obtained with BPTT (Theorem 1 and Fig. 2 , Section 3.3).

We call this property the Gradient Descending Dynamics (GDD) property, for consistency with the terminology used in Ernoult et al. (2019) .

??? We demonstrate training with C-EP on MNIST, with accuracy approaching the one obtained with standard EP (Section 4.2).

??? Finally, we adapt our C-EP algorithm to the more bio-realistic situation of a neural network with asymmetric connections between neurons.

We call this modified version C-VF as it is inspired by the Vector Field method proposed in Scellier et al. (2018) .

We demonstrate this approach on MNIST, and show numerically that the training performance is correlated with the satisfaction of Gradient Descending Dynamics (Section 4.3).

For completeness, we also show how the Recurrent Backpropagation (RBP) algorithm of Almeida (1987) ; Pineda (1987) relates to C-EP, EP and BPTT.

We illustrate the equivalence of these four algorithms on a simple analytical model ( Fig. 3 ) and we develop their relationship in Appendix A.

Convergent RNNs With Static Input.

We consider the supervised setting, where we want to predict a target y given an input x. The model is a recurrent neural network (RNN) parametrized by ?? and evolving according to the dynamics:

F is the transition function of the system.

Assuming convergence of the dynamics before time step T , we have s T = s * where s * is the steady state of the network characterized by

The number of timesteps T is a hyperparameter that we choose large enough so that s T = s * for the current value of ??.

The goal is to optimize the parameter ?? in order to minimize a loss:

Algorithms that optimize the loss L * for RNNs include Backpropagation Through Time (BPTT) and the Recurrent Backpropagation (RBP) algorithm of Almeida (1987); Pineda (1987) , presented in Appendix B.

Equilibrium Propagation (EP).

EP (Scellier and Bengio, 2017 ) is a learning algorithm that computes the gradient of L * in the particular case where the transition function F derives from a scalar function ??, i.e. with F of the form F (x, s, ??) = ????? ???s (x, s, ??).

The algorithm consists in two phases (see Alg.

1 of Fig. 1 Scellier and Bengio (2017) have shown that the gradient of the loss L * can be estimated based on the two steady states s * and s ?? * .

Specifically, in the limit

This section presents the main theoretical contributions of this paper.

We introduce a new algorithm to optimize L * (Eq. 3): a new version of EP with continual parameter updates that we call C-EP.

Unlike typical machine learning algorithms (such as BPTT, RBP and EP) in which the weight updates occur after all the other computations in the system are performed, our algorithm offers a mechanism in which the weights are updated continuously as the states of the neurons change.

The key idea to understand how to go from EP to C-EP is that the gradient of EP appearing in Eq. (4) reads as the following telescopic sum:

In Eq. (5) we have used that s ?? 0 = s * and s ?? t ??? s ?? * as t ??? ???. Here lies the very intuition of continual updates motivating this work; instead of keeping the weights fixed throughout the second phase and updating them at the end of the second phase based on the steady states s * and s ?? * , as in EP (Alg.

1 of Fig. 1 ), the idea of the C-EP algorithm is to update the weights at each time t of the second phase between two consecutive states s ?? t???1 and s ?? t (Alg.

2 of Fig. 1 ).

One key difference in C-EP compared to EP though, is that, in the second phase, the weight update at time step t influences the neural states at time step t + 1 in a nontrivial way, as illustrated in the computational graph of Fig. 2 .

In the next subsection we define C-EP using notations that explicitly show this dependency.

Left.

Pseudo-code of EP.

This is the version of EP for discrete-time dynamics introduced in Ernoult et al. (2019) .

Right.

Pseudo-code of C-EP with simplified notations (see section 3.2 for a formal definition of C-EP).

Difference between EP and C-EP.

In EP, one global parameter update is performed at the end of the second phase ; in C-EP, parameter updates are performed throughout the second phase.

Eq. 5 shows that the continual updates of C-EP add up to the global update of EP.

The first phase of C-EP is the same as that of EP (see Fig. 1 ).

In the second phase of C-EP the parameter variable is regarded as another dynamic variable ?? t that evolves with time t along with s t .

The dynamics of s t and ?? t in the second phase of C-EP depend on the values of the two hyperparameters ?? (the hyperparameter of influence) and ?? (the learning rate), therefore we write s ??,?? t and ?? ??,?? t to show explicitly this dependence.

With now both the neurons and the synapses evolving in the second phase, the dynamic variables s

The difference in C-EP compared to EP is that the value of the parameter used to update s ??,?? t+1 in Eq. (6) is the current ?? ??,?? t , not ??.

Provided the learning rate ?? is small enough, i.e. the synapses are slow compared to the neurons, this effect is weak.

Intuitively, in the limit ?? ??? 0, the parameter changes are negligible so that ?? ??,?? t can be approximated by its initial value ?? ??,?? 0 = ??.

Under this approximation, the dynamics of s ??,?? t in C-EP and the dynamics of s ?? t in EP are the same.

See Fig. 3 for a simple example, and Appendix A.3 for a proof in the general case.

Now we prove that, provided the hyperparameter ?? and the learning rate ?? are small enough, the dynamics of the neurons and the weights given by Eq. (6) follow the gradients of BPTT (Theorem 1 and Fig. 2 ).

For a formal statement of this property, we define the normalized (continual) updates of C-EP, as well as the gradients of the loss L = (s T , y) after T time steps, computed with BPTT:

which corresponds to the parameter gradient at time t, defined informally in Eq. (5).

The following result makes this statement more formal.

Theorem 1 (GDD Property).

Let s 0 , s 1 , . . . , s T be the convergent sequence of states and denote s * = s T the steady state.

Further assume that there exists some step K where 0 < K ??? T such that s * = s T = s T ???1 = . . .

s T ???K .

Then, in the limit ?? ??? 0 and ?? ??? 0, the first K normalized updates in the second phase of C-EP are equal to the negatives of the first K gradients of BPTT, i.e.

Theorem 1 rewrites s

, showing that in the second p??ase of C-EP, neurons and synapses descend the gradients of the loss L obtained with BPTT, with the hyperparameters ?? and ?? playing the role of learning rates for s ??,?? t and ?? ??,?? t , respectively.

Fig. 3 illustrates Theorem 1 with a simple dynamical system for which the normalized updates ??? C???EP and the gradients ??? BPTT are analytically tractable -see Appendix C for derivation details.

In this section, we validate our continual version of Equilibrium Propagation against training on the MNIST data set with two models.

The first model is a vanilla RNN with tied and symmetric weights: the dynamics of this model approximately derive from a primitive function, which allows training with C-EP.

The second model is a Discrete-Time RNN with untied and asymmetric weights, which is therefore closer to biology.

We train this second model with a modified version of C-EP which we call C-VF (Continual Vector Field) as it is inspired from the algorithm with Vector-Field dynamics of Scellier et al. (2018) .

Ernoult et al. (2019) showed with simulations the intuitive result that, if a model is such that the normalized updates of EP 'match' the gradients of BPTT (i.e. if they are approximately equal), then the model trained with EP performs as well as the model trained with BPTT.

Along the same lines, we show in this work that the more the EP normalized updates follow the gradients of BPTT before training, the best is the resulting training performance.

We choose to implement C-EP and C-VF on vanilla RNNs to accelerate simulations (Ernoult et al., 2019) .

Vanilla RNN with symmetric weights trained by C-EP.

The first phase dynamics is defined as:

where ?? is an activation function, W is a symmetric weight matrix connecting the layers s and W x is a matrix connecting the input x to the layers s. Although the dynamics are not directly defined in terms of a primitive function, note that s t+1 ??? ????? ???s (s t , W ) with ??(s, W ) = 1 2 s ?? W ?? s if we ignore the activation function ??.

Following Eq. (6) and Eq. (7), we define the normalized updates of this model as:

Note that this model applies to any topology as long as existing connections have symmetric values: this includes deep networks with any number of layers -see Appendix E for detailed descriptions of the models used in the experiments.

More explicitly, for a network whose layers of neurons are s 0 , s 1 , ..., s N , with W n,n+1 connecting the layers s n+1 and s n in both directions, the corresponding

Vanilla RNN with asymmetric weights trained by C-VF.

In this model, the dynamics in the first phase is the same as Eq. (10) but now the weight matrix W is no longer assumed to be symmetric, i.e. the reciprocal connections between neurons are not constrained.

In this setting the weight dynamics in the second phase is replaced by a version for asymmetric weights:

, so that the normalized updates are equal to:

Like the previous model, the vanilla RNN with asymmetric weights also applies to deep networks with any number of layers.

Although in C-VF the dynamics of the weights is not one of the form of Eq. (6) that derives from a primitive function, the (bioplausible) normalized weight updates of Eq. (12) can approximately follow the gradients of BPTT, provided that the values of reciprocal connections are not too dissimilar: this is illustrated in Fig. 5 (as well as in Fig. 12 and Fig. 13 of Appendix E.6) and proved in Appendix D.2.

This property motivates the following training experiments.

Training results on MNIST with EP, C-EP and C-VF.

"#h" stands for the number of hidden layers.

We indicate over 5 trials the mean and standard deviation for the test error (mean train error in parenthesis).

T (resp.

K) is the number of iterations in the 1 st (resp.

2 nd ) phase.

For C-VF results, the initial angle between forward (?? f ) and backward (?? b ) weights is ??(?? f , ?? b ) = 0

??? .

Right: Test error rate on MNIST achieved by C-VF as a function of the initial ??(?? f , ?? b ).

Experiments are performed with multi-layered vanilla RNNs (with symmetric weights) on MNIST.

The table of Fig. 4 .1 presents the results obtained with C-EP training benchmarked against standard EP training (Ernoult et al., 2019) -see Appendix E for model details and Appendix F.1 for training conditions.

Although the test error of C-EP approaches that of EP, we observe a degradation in accuracy.

This is because although Theorem 1 guarantees Gradient Descending Dynamics (GDD) in the limit of infinitely small learning rates, in practice we have to strike a balance between having a learning rate that is small enough to ensure this condition but not too small to observe convergence within a reasonable number of epochs.

As seen on Fig. 5 (b) , the finite learning rate ?? of continual updates leads to ??? C???EP (??, ??, t) curves splitting apart from the ?????? BPTT (t) curves.

As seen per Fig. 5 (a), this effect is emphasized with the depth: before training, angles between the normalized updates of C-EP and the gradients of BPTT reach 50 degrees for two hidden layers.

The deeper the network, the more difficult it is for the C-EP dynamics to follow the gradients provided by BPTT.

As an evidence, we show in Appendix F.2 that when we use extremely small learning rates throughout the second phase (?? ??? ?? + ?? tiny ??? C???EP ?? ) and rescale up the resulting total weight update (?? ??? ?? ??? ????? tot + ?? ??tiny ????? tot ), we recover standard EP results.

Depending on whether the updates occur continuously during the second phase and the system obey general dynamics with untied forward and backward weights, we can span a large range of deviations from the ideal conditions of Theorem 1.

Fig. 5 (b) qualitatively depicts these deviations with a model for which the normalized updates of EP match the gradients of BPTT (EP) ; with continual weight updates, the normalized updates and gradients start splitting apart (C-EP), and even more so if the weights are untied (C-VF).

Protocol.

In order to create these deviations from Theorem 1 and study the consequences in terms of training, we proceed as follows.

For each C-VF simulations, we tune the initial angle between forward weights (?? f ) and backward weights (?? b ) between 0 and 180

??? .

We denote this angle ??(?? f , ?? b ) -see Appendix F.1 for the angle definition and the angle tuning technique employed.

For each of these weight initialization, we compute the angle between the total normalized update provided by C-VF, i.e. BPTT (tot) before training.

This graphical representation spreads the algorithms between EP which best satisfies the GDD property (leftmost point in green at ??? 20

??? ) to C-VF which satisfies the less the GDD property (rightmost points in red and orange at ??? 100

??? ).

As expected, high angles between gradients of C-VF and BPTT lead to high error rates that can reach 90% for ?? ??? C???VF (tot), ?????? BPTT (tot) over 100

??? .

More precisely, the inset of Fig. 5 shows the same data but focusing only on results generated by initial weight angles lying below 90

From standard EP with one hidden layer to C-VF with two hidden layers, the test error increases monotonically with ?? ???(tot), ?????? BPTT (tot) but does not exceed 5.05% on average.

This result confirms the importance of proper weight initialization when weights are untied, also discussed in other context (Lillicrap et al., 2016) .

When the initial weight angle is of 0

??? , the impact of untying the weights on classification accuracy remains constrained, as shown in table of Fig. 4 .1.

Upon untying the forward and backward weights, the test error increases by ??? 0.2% with one hidden layer and by ??? 0.5% with two hidden layers compared to standard C-EP.

Equilibrium Propagation is an algorithm that leverages the dynamical nature of neurons to compute weight gradients through the physics of the neural network.

C-EP embraces simultaneous synapse and neuron dynamics, resolving the initial need of artificial memory units for storing the neuron values between different phases.

The C-EP framework preserves the equivalence with Backpropagation Through Time: in the limit of sufficiently slow synaptic dynamics (i.e. small learning rates), the system satisfies Gradient Descending Dynamics (Theorem 1).

Our experimental results confirm this theorem.

When training our vanilla RNN with symmetric weights with C-EP while ensuring convergence in 100 epochs, a modest reduction in MNIST accuracy is seen with regards to standard EP.

This accuracy reduction can be eliminated by using smaller learning rates and rescaling up the total weight update at the end of the second phase (Appendix F.2).

On top of extending the theory of Ernoult et al. (2019) , Theorem 1 also appears to provide a statistically robust tool for C-EP based learning.

Our experimental results show as in Ernoult et al. (2019) that, for a given network with specified neuron and synapse dynamics, the more the updates of Equilibrium Propagation follow the gradients provided by Backpropagation Through Time before training (in terms of angle in this work), the better this network can learn.

Our C-EP and C-VF algorithms exhibit features reminiscent of biology.

C-VF extends C-EP training to RNNs with asymmetric weights between neurons, as is the case in biology.

Its learning rule, local in space and time, is furthermore closely acquainted to Spike Timing Dependent Plasticity (STDP), a learning rule widely studied in neuroscience, inferred in vitro and in vivo from neural recordings in the hippocampus (Dan and Poo, 2004) .

In STDP, the synaptic strength is modulated by the relative timings of pre and post synaptic spikes within a precise time window (Bi and Poo, 1998; 2001) .

Each randomly selected synapse corresponds to one color.

While dashed and continuous lines coincide for standard EP, they split apart upon untying the weights and using continual updates.

Strikingly, the same rule that we use for C-VF learning can approximate STDP correlations in a rate-based formulation, as shown through numerical experiments by .

From this viewpoint our work brings EP a step closer to biology.

However, C-EP and C-VF do not aim at being models of biological learning per se, in that it would account for how the brain works or how animals learn, for which Reinforcement Learning might be a more suited learning paradigm.

The core motivation of this work is to propose a fully local implementation of EP, in particular to foster its hardware implementation.

When computed on a standard computer, due to the use of small learning rates to mimic analog dynamics within a finite number of epochs, training our models with C-EP and C-VF entail long simulation times.

With a Titan RTX GPU, training a fully connected architecture on MNIST takes 2 hours 39 mins with 1 hidden layer and 10 hours 49 mins with 2 hidden layers.

On the other hand, C-EP and C-VF might be particularly efficient in terms of speed and energy consumption when operated on neuromorphic hardware that employs analog device physics (Ambrogio et al., 2018; Romera et al., 2018) .

To this purpose, our work can provide an engineering guidance to map our algorithm onto a neuromorphic system.

Fig. 5 (a) shows that hyperparameters should be tuned so that before training, C-EP updates stay within 90

??? of the gradients provided by BPTT.

More concretely in practice, it amounts to tune the degree of symmetry of the dynamics, for instance the angle between forward and backward weights -see Fig. 4 .1.

Our work is one step towards bridging Equilibrium Propagation with neuromorphic computing and thereby energy efficient implementations of gradient-based learning algorithms.

A PROOF OF THEOREM 1

In this appendix, we prove Theorem 1, which we recall here.

Theorem 1 (GDD Property).

Let s 0 , s 1 , . . . , s T be the convergent sequence of states and denote s * = s T the steady state.

Further assume that there exists some step K where 0 < K ??? T such that s * = s T = s T ???1 = . . .

s T ???K .

Then, in the limit ?? ??? 0 and ?? ??? 0, the first K normalized updates in the second phase of C-EP are equal to the negatives of the first K gradients of BPTT, i.e.

A.1 A SPECTRUM OF FOUR COMPUTATIONALLY EQUIVALENT LEARNING ALGORITHMS Proving Theorem 1 amounts to prove the equivalence of C-EP and BPTT.

In fact we can prove the equivalence of four algorithms, which all compute the gradient of the loss:

1.

Backpropagation Through Time (BPTT), presented in Section B.2, 2.

Recurrent Backpropagation (RBP), presented in Section B.3, 3.

Equilibrium Propagation (EP), presented in Section 2, 4.

Equilibrium Propagation with Continual Weight Updates (C-EP), introduced in Section 3.

In this spectrum of algorithms, BPTT is the most practical algorithm to date from the point of view of machine learning, but also the less biologically realistic.

In contrast, C-EP is the most realistic in terms of implementation in biological systems, while it is to date the least practical and least efficient for conventional machine learning (computations on standard Von-Neumann hardware are considerably slower due to repeated parameter updates, requiring memory access at each time-step of the second phase).

Theorem 1 can be proved in three phases, using the following three lemmas.

Lemma 2 (Equivalence of C-EP and EP).

In the limit of small learning rate, i.e. ?? ??? 0, the (normalized) updates of C-EP are equal to those of EP:

Lemma 3 (Equivalence of EP and RBP).

Assume that the transition function derives from a primitive function, i.e. that F is of the form F (x, s, ??) = ????? ???s (x, s, ??).

Then, in the limit of small hyperparameter ??, the normalized updates of EP are equal to the gradients of RBP:

Lemma 4 (Equivalence of BPTT and RBP).

In the setting with static input x, suppose that the network has reached the steady state s * after T ??? K steps, i.e.

Then the first K gradients of BPTT are equal to the first K gradient of RBP, i.e.

Proofs of the Lemmas can be found in the following places:

??? The link between BPTT and RBP (Lemma 2) is known since the late 1980s and can be found e.g. in Hertz (2018) .

We also prove it here in Appendix B.

??? Lemma 3 was proved in in the setting of real-time dynamics.

??? Lemma 4 is the new ingredient contributed here, and we prove it in Appendix A.3.

Also a direct proof of the equivalence of EP and BPTT was derived in Ernoult et al. (2019) .

First, recall the dynamics of C-EP in the second phase: starting from s ??,?? 0 = s * and ?? ??,?? 0 = ?? we have ???t ??? 0 :

We have also defined the normalized updates of C-EP:

We also recall the dynamics of EP in the second phase:

as well as the normalized updates of EP, as defined in Ernoult et al. (2019) :

Lemma 2 (Equivalence of C-EP and EP).

In the limit of small learning rate, i.e. ?? ??? 0, the (normalized) updates of C-EP are equal to those of EP:

Proof of Lemma 2.

We want to compute the limits of ??? C???EP s (??, ??, t) and ??? C???EP ?? (??, ??, t) as ?? ??? 0 with ?? > 0.

First of all, note that under mild assumptions -which we made here -of regularity on the functions ?? and (e.g. continuous differentiability), for fixed t and ??, the quantities s

It follows from Eq. (20) and Eq. (21) that

Now let us compute lim ?????0 (??>0) ??? C???EP s (??, ??, t).

Using Eq. (16), we have

Similarly as before, for fixed t,

is a continuous function of ??.

Therefore

A consequence of Lemma 2 is that the total update of C-EP matches the total update of EP in the limit of small ??, so that we retrieve the standard EP learning rule of Eq. (4).

More explicitly, after K steps in the second phase and starting from ??

, Eq. (7))

In this section, we recall Backprop Through Time (BPTT) and the Almeida-Pineda Recurrent Backprop (RBP) algorithm, which can both be used to optimize the loss L * of Eq. 3.

Historically, BPTT and RBP were invented separately around the same time.

RBP was introduced at a time when convergent RNNs (such as the one studied in this paper) were popular.

Nowadays, convergent RNNs are less popular ; in the field of deep learning, RNNs are almost exclusively used for tasks that deal with sequential data and BPTT is the algorithm of choice to train such RNNs.

Here, we present RBP in a way that it can be seen as a particular case of BPTT.

Lemma 4, which we recall here, is a consequence of Proposition 5 and Definition 6 below.

Lemma 4 (Equivalence of BPTT and RBP).

In the setting with static input x, suppose that the network has reached the steady state s * after T ??? K steps, i.e.

Then the first K gradients of BPTT are equal to the first K gradient of RBP, i.e.

However, in general, the gradients ??? BPTT (t) of BPTT and the gradients ??? RBP (t) of RBP are not equal for t > K. This is because BPTT and RBP compute the gradients of different loss functions:

??? BPTT computes the gradient of the loss after T time steps, i.e. L = (s T , y),

??? RBP computes the gradients of the loss at the steady state, i.e. L * = (s * , y).

Backpropagation Through Time (BPTT) is the standard method to train RNNs and can also be used to train the kind of convergent RNNs that we study in this paper.

To this end, we consider the cost of the state s T after T time steps, denoted L = (s T , y), and we substitute the loss after T time steps L as a proxy for the loss at the steady state L * = (s * , y).

The gradients of L can then be computed with BPTT.

To do this, we recall some of the inner working mechanisms of BPTT.

Eq. (1) rewrites in the form s t+1 = F (x, s t , ?? t+1 = ??), where ?? t denotes the parameter of the model at time step t, the value ?? being shared across all time steps.

This way of rewriting Eq. (1) enables us to define the partial derivative ???L ?????t as the sensitivity of the loss L with respect to ?? t when ?? 1 , . . .

?? t???1 , ?? t+1 , . . .

?? T remain fixed (set to the value ??).

With these notations, the gradient ???L ????? reads as the sum:

BPTT computes the 'full' gradient ???L ????? by first computing the partial derivatives ???L ???st and ???L ?????t iteratively, backward in time, using the chain rule of differentiation.

In this work, we denote the gradients that BPTT computes:

Proposition 5 (Gradients of BPTT).

The gradients ??? BPTT s (t) and ??? BPTT ?? (t) satisfy the recurrence relationship

B.3 FROM BACKPROP THROUGH TIME (BPTT) TO RECURRENT BACKPROP (RBP)

In general, to apply BPTT, it is necessary to store in memory the history of past hidden states s 1 , s 2 , . . .

, s T in order to compute the gradients ??? BPTT s (t) and ??? BPTT ?? (t) as in Eq. 30-31.

However, in our specific setting with static input x, if the network has reached the steady state s * after T ??? K steps, i.e. if s T ???K = s T ???K+1 = ?? ?? ?? = s T ???1 = s T = s * , then we see that, in order to compute the first K gradients of BPTT, all one needs to know is ???F ???s (x, s * , ??) and ???F ????? (x, s * , ??).

To this end, all one needs to keep in memory is the steady state s * .

In this particular setting, it is not necessary to store the past hidden states s T , s T ???1 , . . .

, s T ???K since they are all equal to s * .

The Almeida-Pineda algorithm (a.k.a.

Recurrent Backpropagation, or RBP for short), which was invented independently by Almeida (1987) and Pineda (1987) , relies on this property to compute the gradients of the loss L * using only the steady state s * .

Similarly to BPTT, it computes quantities ??? RBP s (t) and ??? RBP ?? (t), which we call 'gradients of RBP', iteratively for t = 0, 1, 2, . . .

RBP s (t) and ??? RBP ?? (t) are defined and computed iteratively as follows:

Unlike in BPTT where keeping the history of past hidden states is necessary to compute (or 'backpropagate') the gradients, in RBP Eq. 33-34 show that it is sufficient to keep in memory the steady state s * only in order to iterate the computation of the gradients.

RBP is more memory efficient than BPTT.

Input: x, y, ??.

Output: ??.

1: s 0 ??? 0 2: for t = 0 to T ??? 1 do 3:

Algorithm 4 RBP Input: x, y, ??.

Output: ??.

1: s 0 ??? 0 2: repeat 3: Figure 6 : Left.

Pseudo-code of BPTT.

The gradients ???(t) denote the gradients ??? BPTT (t) of BPTT.

Right.

Pseudo-code of RBP.

Difference between BPTT and RBP.

In BPTT, the state s T ???t is required to compute ???F ???s (x, s T ???t , ??) and ???F ????? (x, s T ???t , ??) ; thus it is necessary to store in memory the sequence of states s 1 , s 2 , . . .

, s T .

In contrast, in RBP, only the steady state s * is required to compute ???F ???s (x, s * , ??) and ???F ????? (x, s * , ??) ; it is not necessary to store the past states of the network.

In this subsection we motivate the name of 'gradients' for the quantities ??? RBP s (t) and ??? RBP ?? (t) by proving that they are the gradients of L * in the sense of Proposition 7 below.

They are also the gradients of what we call the 'projected cost function' (Proposition 8), using the terminology of .

Proposition 7 (RBP Optimizes L * ).

The total gradient computed by the RBP algorithm is the gradient of the loss L * = (s * , y), i.e.

??? RBP s (t) and ??? RBP ?? (t) can also be expressed as gradients of L t = (s t , y), the cost after t time steps.

In the terminology of , L t was named the projected cost.

For t = 0, L 0 is simply the cost of the initial state s 0 .

For t > 0, L t is the cost of the state projected a duration t in the future.

Proposition 8 (Gradients of RBP are Gradients of the Projected Cost).

The 'RBP gradients' ??? RBP s (t) and ??? RBP ?? (t) can be expressed as gradients of the projected cost:

where the initial state s 0 is the steady state s * .

Proof of Proposition 7.

First of all, by Definition 6 (Eq. 32-34)

it is straightforward to see that

Second, recall that the loss L * is

where

By the chain rule of differentiation, the gradient of L * (Eq. 39) is

In order to compute ???s * ????? , we differentiate the steady state condition (Eq. 40) with respect to ??, which yields

Rearranging the terms, and using the Taylor expansion (Id ??? A)

Therefore

Proof of Proposition 8.

By the chain rule of differentiation we have

Evaluation this expression for s 0 = s * we get

Model.

To illustrate the equivalence of the four algorithms (BPTT, RBP, EP and CEP), we study a simple model with scalar variable s and scalar parameter ??:

where s * is the steady state of the dynamics (it is easy to see that the solution is s * = ??).

The dynamics rewrites s t+1 = F (s t , ??) with the transition function F (s, ??) = 1 2 (s + ??), and the loss rewrites L * = (s * ) with the cost function (s) = 1 2 s 2 .

Furthermore, a primitive function of the system 1 is ??(s, ??) = 1 4 (s + ??) 2 .

This model has no practical application ; it is only meant for pedagogical purpose.

With BPTT, an important point is that we approximate the steady state s * by the state after T time steps s T , and we approximate L * (the loss at the steady state) by the loss after T time steps L = (s T ).

In order to compute (i.e. 'backpropagate') the gradients of BPTT, Proposition 5 tells us that we need to compute

The state after T time steps in BPTT converges to the steady state s * as T ??? ???, therefore the gradients of BPTT converge to the gradients of RBP.

Also notice that the steady state of the dynamics is s * = ??.

Following the equations governing the second phase of EP (Fig. 1) , we have:

This linear dynamical system can be solved analytically:

Notice that s ?? t ??? ?? as ?? ??? 0 ; for small values of the hyperparameter ??, the trajectory in the second phase is close to the steady state s * = ??.

Using Eq. 19, it follows that the normalized updates of EP are

Notice again that the normalized updates of EP converge to the gradients of RBP as ?? ??? 0.

The system of equations governing the system is:

1 The primitive function ?? is determined up to a constant.

First, rearranging the terms in the second equation, we get

It follows that ???

Therefore, all we need to do is to compute ??? C???EP s (??, ??, t).

Second, by iterating the second equation over all indices from t = 0 to t ??? 1 we get

Using s * = ?? and plugging this into the first equation we get

Solving this linear dynamical system, and using the initial condition s

Finally:

Step-by-step equivalence of the dynamics of EP and gradient computation in BPTT was shown in Ernoult et al. (2019) and was refered to as the Gradient-Descending Updates (GDU) property.

In this appendix, we first explain the connection between the GDD property of this work and the GDU property of Ernoult et al. (2019) .

Then we prove another version of the GDD property (Theorem 9 below), more general than Theorem 1.

The GDU property of Ernoult et al. (2019) states that the (normalized) updates of EP are equal to the gradients of BPTT.

Similarly, the Gradient-Descending Dynamics (GDD) property of this work states that the normalized updates of C-EP are equal to the gradients of BPTT.

The difference between the GDU property and the GDD property is that the term 'update' has slightly different meanings in the contexts of EP and C-EP.

In C-EP, the 'updates' are the effective updates by which the neuron and synapses are being dynamically updated throughout the second phase.

In contrast in EP, the 'updates' are effectively performed at the end of the second phase.

The Gradient Descending Dynamics property (GDD, Theorem 1) states that, when the system dynamics derive from a primitive function, i.e. when the transition function F is of the form F = ????? ???s , then the normalized updates of C-EP match the gradients provided by BPTT.

Remarkably, even in the case of the C-VF dynamics that do not derive from a primitive function ??, Fig. 5 shows that the biologically realistic update rule of C-VF follows well the gradients of BPTT.

More illustrations of this property are shown on Fig. 12 and Fig. 13 .

In this section we give a theoretical justification for this fact by proving a more general result than Theorem 1.

First, recall the dynamics of the C-VF model.

In the first phase:

where ?? is an activation function and W is a square weight matrix.

In the second phase, starting from s ??,?? 0 = s * and W ??,?? 0 = W , the dynamics read:

Now let us define the transition function F (s, W ) = ??(W ?? s), so that the dynamics of the first phase rewrites

As for the second phase, notice that

Now, recall the definition of the normalized updates of C-VF, as well as the gradients of the loss L = (s T , y) after T time steps, computed with BPTT:

The loss L and the gradients ??? ) .

Then, in the limit ?? ??? 0 and ?? ??? 0, the first K normalized updates of C-VF follow the the first K gradients of BPTT, i.e. ???t = 0, 1, . . .

, K :

A few remarks need to be made:

Ignoring the factor ?? (W ?? s), we see that if W is symmetric then the Jacobian of F is also symmetric, in which case the conditions of Theorem 9 are met.

2.

Theorem 1 is a special case of Theorem 9.

To see why, notice that if the transition function F is of the form

In this case the extra assumption in Theorem 9 is automatically satisfied.

Theorem 9 is a consequence of Proposition 5 (Appendix B.2), which we recall here, and Lemma 10 below.

BPTT s (t) and ??? BPTT ?? (t) satisfy the recurrence relationship

Lemma 10 (Updates of C-VF).

Define the (normalized) neural and weight updates of C-VF in the limit ?? ??? 0 and ?? ??? 0:

They satisfy the recurrence relationship

The proof of Lemma 10 is similar to the one provided in Ernoult et al. (2019) .

In this section, we describe the C-EP and C-VF algorithms when implemented on multi-layered models, with tied weights and untied weights respectively.

In the fully connected layered architecture model, the neurons are only connected between two consecutive layers (no skip-layer connections and no lateral connections within a layer).

We denote neurons of the n-th layer as s n with n ??? [0, N ??? 1], where N is the number of hidden layers.

Layers are labelled in a backward fashion: n = 0 labels the output layer, n = 1 the first hidden layer starting from the output layer, and n = N ??? 1 the last hidden layer (before the input layer).

Thus, there are N hidden layers in total.

Fig. 7 shows this architecture with N = 2.

Each model are presented here in a "real-time" and "discrete-time" settings For each model we lay out the equations of the neuron and synapse dynamics, we demonstrate the GDD property and we specify in which part of the main text they are used.

We present in this order: Demonstrating the Gradient Descending Dynamics (GDD) property (Theorem 1) on MNIST.

For this experiment, we consider the 784-512-. . .

-512-10 network architecture, with 784 input neurons, 10 ouput neurons, and 512 neurons per hidden layer.

The activation function used is ??(x) = tanh(x).

The experiment consists of the following: we take a random MNIST sample (of size 1 ?? 784) and its associated target (of size 1 ?? 10).

For a given value of the time-discretization parameter , we perform the first phase for T steps.

Then, we perform on the one hand BPTT over K steps (to compute the gradients ??? BPTT ), on the other hand C-EP (or C-VF) over K steps for given values of ?? and ?? (to compute the normalized updates ??? C???EP or ??? C???VF ) and compare the gradients and normalized updates provided by the two algorithms.

Precise values of the hyperparameters , T , K, ?? and ?? are given in Tab.

E.6.

Equations with N = 2.

We consider the layered architecture of Fig. 7 , where s 0 denotes the output layer, and the feedback connections are constrained to be the transpose of the feedforward connections, i.e. W nn???1 = W n???1n .

In the discrete-time setting of EP, the dynamics of the first phase are defined as:

In the second phase the dynamics reads:

As usual, y denotes the target.

Consider the function:

We can compute, for example:

Comparing Eq. (76) and Eq. (79), and ignoring the activation function ??, we can see that

And similarly for the layers s 0 and s 2 .

According to the definition of ??? C???EP ?? in Eq. (19), for every layer and every t ??? [0, K]:

Simplifying the equations with N = 2.

To go from our multi-layered architecture to the more general model presented in section 4.1.

we define the state s of the network as the concatenation of all the layers' states, i.e. s = (s 2 , s 1 , s 0 ) and we define the weight matrices W and W x as:

Note that Eq. (76) and Eq. (78) can be vectorized into:

Generalizing the equations for any N .

For a general architecture with a given N , the dynamics of the first phase are defined as:

and those of the second phase as:

where y denotes the target.

Defining:

ignoring the activation function ??, Eq. (85) rewrites:

According to the definition of ??? C???EP ?? in Eq. (19), for every layer W nn+1 and every t ??? [0, K]:

Defining s = (s N , s N ???1 , . . . , s 0 ) and:

Eq. (85) and Eq. (87) can also be vectorized into:

Thereafter we introduce the other models in this general case.

Context of use.

This model is used for training experiments in Section 4.2 and Table 4 .1.

Equations.

Recall that we consider the layered architecture of Fig. 7 , where s 0 denotes the output layer.

Just like in the discrete-time setting of EP, the dynamics of the first phase are defined as:

Again, as in EP, the feedback connections are constrained to be the transpose of the feedforward connections, i.e. W nn???1 = W n???1n .

In the second phase the dynamics reads:

(??, ??, t) ????? ??? {W nn+1 } (94) As usual, y denotes the target.

Since Eq. (93) and Eq. (85) are the same, the equations describing the C-EP model can also be written in a vectorized block-wise fashion, as in Eq. (91) and Eq. (92).

We can consequently define the C-EP model in Section 4.1 per Eq. (10).

According to the definitions of Eq. (6) and Eq. (7), for every layer W nn+1 and every t ??? [0, K]:

Context of use.

This model has not been used in this work.

We only introduce it for completeness with respect to Ernoult et al. (2019) .

Equations.

For this model, the primitive function is defined as:

so that the equations of motion read:

In the second phase:

????? ??? {W nn+1 } (97) where is a time-discretization parameter and y denotes the target.

According the definition of the C-EP dynamics (Eq. (6)), the definition of ??? C???EP ?? (Eq. (7)) and the explicit form of ?? (Eq. 96), for all time step t ??? [0, K], we have: Under review as a conference paper at ICLR 2020

Equations.

Recall that we consider the layered architecture of Fig. 7 , where s 0 denotes the output layer.

The dynamics of the first phase in C-VF are defined as:

Here, note the difference with EP and C-EP: the feedforward and feedback connections are unconstrained.

In the second phase of C-VF:

As usual y denotes the target.

Note that Eq. (98) can also be in a vectorized block-wise fashion as Eq. (91) with s = (s 0 , s 1 , . . .

, s N ???1 ) and provided that we define W and W x as:

For all layers W nn+1 and W n+1n , and every t ??? [0, K], we define: Table E .6 for precise hyperparameters.

Equations.

For this model, the dynamics of the first phase are defined as:

where is the time-discretization parameter.

Again, as in the discre-time version of C-VF, the feedforward and feedback connections W nn???1 and W n???1n are unconstrained.

In the second phase, the dynamics reads:

where y denotes the target, as usual.

For every feedforward connection matrix W nn+1 and every feedback connection matrix W n+1n , and for every time step t ??? [0, K] in the second phase, we define

In the following figures, we show the effect of using continual updates with a finite learning rate in terms of the ??? C???EP and ?????? BPTT processes on different models introduced above.

These figures have been realized either in the discrete-time or continuous-time setting with the fully connected layered architecture with one hidden layer on MNIST.

Dashed an continuous lines respectively represent the normalized updates ??? and the gradients ??? BPTT .

Each randomly selected synapse or neuron correspond to one color.

We add an s or ?? index to specify whether we analyse neuron or synapse updates and gradients.

Each C-VF simulation has been realized with an angle between forward and backward weights of 0 degrees (i.e. ??(?? f , ?? b ) = 0

??? ).

For each figure, left panels demonstrate the GDD property with C-EP with ?? = 0 and the right panels show that, upon using ?? > 0, dashed and continuous lines start to split appart.

Simulation framework.

Simulations have been carried out in Pytorch.

The code has been attached to the supplementary materials upon submitting this work on OpenReview.

We have also attached a readme.txt with a specification of all dependencies, packages, descriptions of the python files as well as the commands to reproduce all the results presented in this paper.

Data set.

Training experiments were carried out on the MNIST data set.

Training set and test set include 60000 and 10000 samples respectively.

Optimization.

Optimization was performed using stochastic gradient descent with mini-batches of size 20.

For each simulation, weights were Glorot-initialized.

No regularization technique was used and we did not use the persistent trick of caching and reusing converged states for each data sample between epochs as in Scellier and Bengio (2017) .

Activation function.

For training, we used the activation function

Although it is a shifted and rescaled sigmoid function, we shall refer to this activation function as 'sigmoid'.

Use of a randomized ??.

The option 'Random ??' appearing in the detailed table of results (Table 3) refers to the following procedure.

During training, instead of using the same ?? accross mini-batches, we only keep the same absolute value of ?? and sample its sign from a Bernoulli distribution of probability 1 2 at each mini-batch iteration.

This procedure was hinted at by Scellier and Bengio (2017) to improve test error, and is used in our context to improve the model convergence for Continual Equilibrium Propagation -appearing as C-EP and C-VF in Table 4 .1 -training simulations.

Tuning the angle between forward and backward weights.

In Table 4 .1, we investigate C-VF initialized with different angles between the forward and backward weights -denoted as ?? in Table 4 .1.

Denoting them respectively ?? f and ?? b , the angle ?? between them is defined here as:

where Tr denotes the trace, i.e. Tr(A) = i A ii for any squared matrix A. To tune arbitrarily well enough ??(?? f , ?? b ), the procedure is the following: starting from ?? b = ?? f , i.e. ??(?? f , ?? b ) = 0, we can gradually increase the angle between ?? f and ?? b by flipping the sign of an arbitrary proportion of components of ?? b .

The more components have their sign flipped, the larger is the angle.

More formally, we write ?? b in the form ?? b = M (p) ?? f and we define:

where M (p) is a mask of binary random values {+1, -1} of the same dimension of ?? f : M (p) = ???1 with probability p and M (p) = +1 with probability 1 ???

p. Taking the cosine and the expectation of Eq. (103), we obtain:

Thus, the angle ?? between ?? f and ?? f M (p) can be tuned by the choice of p through:

Hyperparameter search for EP.

We distinguish between two kinds of hyperparameters: the recurrent hyperparameters -i.e.

T , K and ?? -and the learning rates.

A first guess of the recurrent hyperparameters T and ?? is found by plotting the ??? C???EP and ??? BPTT processes associated to synapses and neurons to see qualitatively whether the theorem is approximately satisfied, and by conjointly computing the proportions of synapses whose ??? C???EP W processes have the same sign as its ??? BPTT W processes.

K can also be found out of the plots as the number of steps which are required for the gradients to converge.

Morever, plotting these processes reveal that gradients are vanishing when going away from the output layer, i.e. they lose up to 10 ???1 in magnitude when going from a layer to the previous (i.e. upstream) layer.

We subsequently initialized the learning rates with increasing values going from the output layer to upstreams layers.

The typical range of learning rates is [10 ???3 , 10 ???1 ], [10, 1000] for T , [2, 100] for K and [0.01, 1] for ??.

Hyperparameters where adjusted until having a train error the closest to zero.

Finally, in order to obtain minimal recurrent hyperparameters -i.e.

smallest T and K possible -we progressively decreased T and K until the train error increases again.

Table 2 : Table of hyperparameters used for training.

"C" and "VF" respectively denote "continual" and "vector-field", "-#h" stands for the number of hidden layers.

The sigmoid activation is defined by Eq. Table 3 : Training results on MNIST with EP, C-EP and C-VF.

"#h" stands for the number of hidden layers.

We indicate over five trials the mean and standard deviation for the test error, the mean error in parenthesis for the train error.

T (resp.

K) is the number of iterations in the first (resp.

second) phase.

Full

<|TLDR|>

@highlight

We propose a continual version of Equilibrium Propagation, where neuron and synapse dynamics occur simultaneously throughout the second phase, with theoretical guarantees and numerical simulations.