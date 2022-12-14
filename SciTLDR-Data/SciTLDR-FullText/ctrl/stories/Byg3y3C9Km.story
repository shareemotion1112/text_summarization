The Boltzmann distribution is a natural model for many systems, from brains to materials and biomolecules, but is often of limited utility for fitting data because Monte Carlo algorithms are unable to simulate it in available time.

This gap between the expressive capabilities and sampling practicalities of energy-based models is exemplified by the protein folding problem, since energy landscapes underlie contemporary knowledge of protein biophysics but computer simulations are challenged to fold all but the smallest proteins from first principles.

In this work we aim to bridge the gap between the expressive capacity of energy functions and the practical capabilities of their simulators by using an unrolled Monte Carlo simulation as a model for data.

We compose a neural energy function with a novel and efficient simulator based on Langevin dynamics to build an end-to-end-differentiable model of atomic protein structure given amino acid sequence information.

We introduce techniques for stabilizing backpropagation under long roll-outs and demonstrate the model's capacity to make multimodal predictions and to, in some cases, generalize to unobserved protein fold types when trained on a large corpus of protein structures.

Many natural systems, such as cells in a tissue or atoms in a protein, organize into complex structures from simple underlying interactions.

Explaining and predicting how macroscopic structures such as these arise from simple interactions is a major goal of science and, increasingly, machine learning.

The Boltzmann distribution is a foundational model for relating local interactions to system behavior, but can be difficult to fit to data.

Given an energy function U ✓ [x] , the probability of a system configuration x scales exponentially with energy as DISPLAYFORM0 where the (typically intractable) constant Z normalizes the distribution.

Importantly, simple energy functions U ✓ [x] consisting of weak, local interactions can collectively encode complex system behaviors, such as the structures of materials and molecules or, when endowed with latent variables, the statistics of images, sound, and text BID0 BID17 .

Unfortunately, learning model parameters✓ and generating samples x ⇠ p ✓ (x) of the Boltzmann distribution is difficult in practice, as these procedures depend on expensive Monte Carlo simulations that may struggle to mix effectively.

These difficulties have driven a shift towards generative models that are easier to learn and sample from, such as directed latent variable models and autoregressive models (Goodfellow et al., 2016) .The protein folding problem provides a prime example of both the power of energy-based models at describing complex relationships in data as well as the challenge of generating samples from them.

Decades of research in biochemistry and biophysics support an energy landscape theory of An unrolled simulator as a model for protein structure.

NEMO combines a neural energy function for coarse protein structure, a stochastic simulator based on Langevin dynamics with learned (amortized) initialization, and an atomic imputation network to build atomic coordinate output from sequence information.

It is trained end-to-end by backpropagating through the unrolled folding simulation.protein folding (Dill et al., 2017) , in which the folds that natural protein sequences adopt are those that minimize free energy.

Without the availability of external information such as coevolutionary information (Marks et al., 2012) or homologous structures (Martí-Renom et al., 2000) to constrain the energy function, however, contemporary simulations are challenged to generate globally favorable low-energy structures in available time.

How can we get the representational benefits of energy-based models with the sampling efficiency of directed models?

Here we explore a potential solution of directly training an unrolled simulator of an energy function as a model for data.

By directly training the sampling process, we eschew the question 'when has the simulator converged' and instead demand that it produce a useful answer in a fixed amount of time.

Leveraging this idea, we construct an end-to-end differentiable model of protein structure that is trained by backpropagtion through folding ( FIG0 ).

NEMO (Neural energy modeling and optimization) can learn at scale to generate 3D protein structures consisting of hundreds of points directly from sequence information.

Our main contributions are:• Neural energy simulator model for protein structure that composes a deep energy function, unrolled Langevin dynamics, and an atomic imputation network for an end-to-end differentiable model of protein structure given sequence information• Efficient sampling algorithm that is based on a transform integrator for efficient sampling in transformed coordinate systems• Stabilization techniques for long roll-outs of simulators that can exhibit chaotic dynamics and, in turn, exploding gradients during backpropagation• Systematic analysis of combinatorial generalization with a new dataset of protein sequence and structure

Protein modeling Our model builds on a long history of coarse-grained modeling of protein structure (Kolinski et al., 1998; Kmiecik et al., 2016 Figure 2 : A neural energy function models coarse grained structure and is sampled by internal coordinate dynamics.

(A) The energy function is formulated as a Markov Random Field with structure-based features and sequence-based weights computed by neural networks FIG2 ).

(B) To rapidly sample low-energy configurations, the Langevin dynamics simulator leverages both (i) an internal coordinate parameterization, which is more effective for global rearrangements, and (ii) a Cartesian parameterization, which is more effective for localized structural refinement.

(C) The base features of the structure network are rotationally and translationally invariant internal coordinates (not shown), pairwise distances, and pairwise orientations.2016; BID2 BID5 .

Structured Prediction Energy Networks (SPENs) with unrolled optimization BID6 are a highly similar approach to ours, differing in terms of the use of optimization rather than sampling.

Additional methodologically related work includes approaches to learn energy functions and samplers simultaneously (Kim & Bengio, 2016; BID25 Dai et al., 2017; BID20 BID8 , to learn efficient MCMC operators BID20 Levy et al., 2018) , to build expressive approximating distributions with unrolled Monte Carlo simulations BID18 BID23 , and to learn the parameters of simulators with implicitly defined likelihoods 1 BID10 BID24 .

Overview NEMO is an end-to-end differentiable model of protein structure X conditioned on sequence information s consisting of three components ( FIG0 ): (i) a neural energy function U ✓ [x; s] for coarse grained structure x given sequence, (ii) an unrolled simulator that generates approximate samples from U via internal coordinate Langevin dynamics ( § 2.3), and (iii) an imputation network that generates an atomic model X from the final coarse-grained sample x (T ) ( § 2.4).

All components are trained simultaneously via backpropagation through the unrolled process.

Proteins Proteins are linear polymers (sequences) of amino acids that fold into defined 3D structures.

The 20 natural amino acids have a common monomer structure [-(N-H)-(C-R)-(C=O)-] with variable side-chain R groups that can differ in properties such as hydrophobicity, charge, and ability to form hydrogen bonds.

When placed in solvent (such as water or a lipid membrane), interactions between the side-chains, backbone, and solvent drive proteins into particular 3D configurations ('folds'), which are the basis for understanding protein properties such as biochemical activity, ligand binding, and interactions with drugs.

Coordinate representations We predict protein structure X in terms of 5 positions per amino acid: the four heavy atoms of the backbone (N, C ↵ , and carbonyl C=O) and the center of mass of the side chain R group.

While it is well-established that the locations of C ↵ carbons are sufficient to reconstruct a full atomic structure (Kmiecik et al., 2016) , we include these additional positions for evaluating backbone hydrogen bonding (secondary structure) and coarse side-chain placement.

Internally, the differentiable simulator generates an initial coarse-grained structure (1-position-peramino-acid) with the loss function targeted to the midpoint of the C ↵ carbon and the side chain center of mass.

Sequence conditioning We consider two modes for conditioning our model on sequence information: (1) 1-seq, in which s is an L ⇥ 20 matrix containing a one-hot encoding of the amino acid sequence, and (2) Profile, in which s is an L ⇥ 40 matrix encoding both the amino acid sequence and a profile of evolutionarily related sequences ( § B.7).Internal coordinates In contrast to Cartesian coordinates x, which parameterize structure in terms of absolute positions of points x i 2 R 3 , internal coordinates z parameterize structure in terms of relative distances and angles between points.

We adopt a standard convention for internal coordinates of chains BID13 where each point x i is placed in a spherical coordinate system defined by the three preceding points x i 1 , x i 2 , x i 3 in terms of a radius (bond length 2 ) b i 2 (0, 1), a polar angle (bond angle) a i 2 [0, ⇡), and an azimuthal angle (dihedral angle) d i 2 [0, 2⇡) ( Figure 2B ).

We define z i = {b i ,ã i , d i }, whereb i ,ã i are unconstrained parameterizations of b i and a i ( § A.1).

The transformation x = F(z) from internal coordinates to Cartesian is then defined by the recurrence DISPLAYFORM0 ||ûi 1 ⇥ûi|| is a unit vector normal to each bond plane.

The inverse transformation z = F 1 (x) is simpler to compute, as it only involves local (and fully parallelizable) calculations of distances and angles ( § A.1).

Deep Markov Random Field We model the distribution of a structure x conditioned on a sequence s with the Boltzmann distribution, p ✓ (x|s) = 1.

Internal coordinates z All internal coordinates except 6 are invariant to rotation and translation 3 and we mask these in the energy loss.2.

Distances D ij = kx i x j k between all pairs of points.

We further process these by 4 radial basis functions with (learned) Gaussian kernels.3.

Orientation vectorsv ij , which are unit vectors encoding the relative position of point x j in a local coordinate system of x i with base vectorsû DISPLAYFORM0 kûi û i+1 k ,n i+1 , and the cross product thereof.

Langevin dynamics The Langevin dynamics is a stochastic differential equation that aymptotically samples from the Boltzmann distribution (Equation 1).

It is typically simulated by a first-order discretization as DISPLAYFORM0 Internal coordinate dynamics The efficiency with which Langevin dynamics explores conformational space is highly dependent on the geometry (and thus parameterization) of the energy landscape U (x).

While Cartesian dynamics are efficient at local structural rearrangement, internal coordinate dynamics much more efficiently sample global, coherent changes to the topology of the fold (Figure 2B) .

We interleave the Cartesian Langevin dynamics with preconditioned Internal Coordinate dynamics, DISPLAYFORM1 where C is a preconditioning matrix that sets the relative scaling of changes to each degree of freedom.

For all simulations we unroll T = 250 time steps, each of which is comprised of one Cartesian step followed by one internal coordinate step (Equation 9, § A.3).Transform integrator Simulating internal coordinate dynamics is often computationally intensive as it requires rebuilding Cartesian geometry x from internal coordinates z with F(z) BID13 which is an intrinsically sequential process.

Here we bypass the need for recomputing coordinate transformations at every step by instead computing on-the-fly transformation integration (Figure 3 ).

The idea is to directly apply coordinate updates in one coordinate system to another by numerically integrating the Jacobian.

This can be favorable when the Jacobian has a simple structure, such as in our case where it requires only distributed cross products.

Local reference frame reconstruction The imputation network builds an atomic model X from the final coarse coordinates x (T ) .

Each atomic coordinate X i,j of atom type j at position i is placed in a local reference frame as DISPLAYFORM0 where e i,j (z; ✓) and r i,j (z; ✓) are computed by a 1D convolutional neural network ( FIG2 ).

We train and evaluate the model on a set of ⇠67,000 protein structures (domains) that are hierarchically and temporally split.

The model is trained by gradient descent using a composite loss that combines terms from likelihood-based and empirical-risk minimization-based training.

Compute forces fz = @x @z DISPLAYFORM0 Compute forces fz = @x @z DISPLAYFORM1 Figure 3: A transform integrator simulates Langevin dynamics in a more favorable coordinate system (e.g. internal coordinates z) directly in terms of the untransformed state variables (e.g. Cartesian x).

This exchanges the cost of an inner-loop transformation step (e.g. geometry construction F(z)) for an extra Jacobian evaluation, which is fully parallelizable on modern hardware (e.g. GPUs).

Structural stratification There are several scales of generalization in protein structure prediction, which range from predicting the structure of a sequence that differs from the training set at a few positions to predicting a 3D fold topology that is absent from training set.

To test these various levels of generalization systematically across many different protein families, we built a dataset on top of the CATH hierarchical classification of protein folds BID11 .

CATH hierarchically organizes proteins from the Protein Data Bank BID7 into domains (individual folds) that are classified at the levels of Class, Architecture, Topology, and Homologous superfamily (from general to specific).

We collected protein domains from CATH releases 4.1 and 4.2 up to length 200 and hierarchically and temporally split this set ( § B.1) into training (⇠35k folds), validation (⇠21k folds), and test sets (⇠10k folds).

The final test set is subdivided into four subsets: C, A, T, and H, based on the level of maximal similarity between a given test domain and domains in the training set.

For example, domains in the C or A sets may share class and potentially architecture classifications with train but will not share topology (i.e. fold).

Likelihood The gradient of the data-averaged log likelihood of the Boltzmann distribution is DISPLAYFORM0 which, when ascended, will minimize the average energy of samples from the data relative to samples from the model.

In an automatic differentiation setting, we implement a Monte Carlo estimator for (the negative of) this gradient by adding the energy gap, DISPLAYFORM1 to the loss, where ? is an identity operator that sets the gradient to zero 4 .Empirical Risk In addition to the likelihood loss, which backpropagates through the energy function but not the whole simulation, we developed an empirical risk loss composing several measures of protein model quality.

It takes the form schematized in FIG2 .

Our combined loss sums all of the terms L = L ER + L ML without weighting.

DISPLAYFORM2

We found that the long roll-outs of our simulator were prone to chaotic dynamics and exploding gradients, as seen in other work (Maclaurin et al., 2015; BID12 .

Unfortunately, when chaotic dynamics do occur, it is typical for all gradients to explode (across learning steps) and standard techniques such as gradient clipping BID14 are unable to rescue learning ( § B.5).

To stabilize training, we developed two complimentary techniques that regularize against chaotic simulator dynamics while still facilitating learning when they arise.

They are• Lyapunov regularization We regularize the simulator time-step function (rather than the energy function) to be approximately 1-Lipschitz. (If exactly satisfied, this eliminates the possibility of chaotic dynamics.)• Damped backpropagation through time We exponentially decay gradient accumulation on the backwards pass of automatic differentiation by multiplying each backwards iteration by a damping factor .

We adaptively tune to cancel the scale of the exploding gradients.

This can be thought of as a continuous relaxation of and a quantitatively tunable alternative to truncated backpropagation through time.

Figure 5 : Examples of fold generalization at topology and architecture level.

These predicted structures show a range of prediction accuracy for structural generalization (C and A) tasks, with the TM-score comparing the top ranked 3D-Jury pick against the target.

The largest clusters are the three most-populated clusters derived from 100 models per domain with a within-cluster cutoff of TM > 0.5.

CATH IDs: 2oy8A03; 5c3uA02; 2y6xA00; 3cimB00; 4ykaC00; 2f09A00; 3i5qA02; 2ayxA01.

For each of the 10,381 protein structures in our test set, we sampled 100 models from NEMO, clustered them by structural similarity, and selected a representative structure by a standard consensus algorithm (Ginalski et al., 2003) .

For evaluation of performance we focus on the TM-Score BID27 , a measure of structural similarity between 0 and 1 for which TM > 0.5 is typically considered an approximate reconstruction of a fold.

Calibrated uncertainty We find that, when the model is confident (i.e. the number of distinct structural clusters is low ⇠1-3), it is also accurate with some predictions having average TM > 0.5 FIG1 .

Unsurprisingly, the confidence of the model tends to go with the difficulty of generalization, with the most confident predictions from the H test set and the least confident from C. Structural generalization However, even when sequence identity is low and generalization difficulty is high FIG1 , center), the model is still able to make some accurate predictions of 3D structures.

Figure 5 illustrates some of these successful predictions at C and A levels, specifically 4ykaC00, 5c3uA02 and beta sheet formation in 2oy8A03.

We observe that the predictive distribution is multimodal with non-trivial differences between the clusters representing alternate packing of the chain.

In some of the models there is uneven distribition of uncertainty along the chain, which sometimes corresponded to loosely packed regions of the protein.

Comparison to an end-to-end baseline We constructed a baseline model that is a non-iterative replica of NEMO which replaces the coarse-grained simulator module (and energy function) with a two-layer bidirectional LSTM that directly predicts coarse internal coordinates z (0) (followed by transformation to Cartesian coordinates with F).

We trained this baseline across a range of hyperparameter values and found that for difficult C, A, and T tasks, NEMO generalized more effectively than the RNNs TAB1 .

For the best performing 2x300 architecture, we trained two additional replicates and report the averaged perfomance in FIG1 (right).Additionally, we report the results of a sequence-only NEMO model in TAB1 .

Paralleling secondary structure prediction BID16 McGuffin et al., 2000) , we find that the availability of evolutionary information has significant impact on prediction quality.

This work presents a novel approach for protein structure prediction that combines the inductive bias of simulators with the speed of directed models.

A major advantage of the approach is that model sampling (inference) times can be considerably faster than conventional approaches to protein structure prediction TAB5 ).

There are two major disadvantages.

First, the computational cost of training and sampling is higher than that of angle-predicting RNNs FIG0 ) such as our baseline or AlQuraishi (2018).

Consequently, those methods have been scaled to larger datasets than ours (in protein length and diversity) which are more relevant to protein structure prediction tasks.

Second, the instability of backpropagating through long simulations is unavoidable and only partially remedied by our approaches of Lipschitz regularization and gradient damping.

These approaches can also lead to slower learning and less expressive energy functions.

Methods for efficient (i.e. subquadratic) N -body simulations and for more principled stabilization of deep networks may be relevant to addressing both of these challenges in the future.

We described a model for protein structure given sequence information that combines a coarse-grained neural energy function and an unrolled simulation into an end-to-end differentiable model.

To realize this idea at the scale of real proteins, we introduced an efficient simulator for Langevin dynamics in transformed coordinate systems and stabilization techniques for backpropagating through long simulator roll-outs.

We find that that model is able to predict the structures of protein molecules with hundreds of atoms while capturing structural uncertainty, and that the model can structurally generalize to distant fold classifications more effectively than a strong baseline. (MPNN, bottom left) , and outputs energy function weights l as well as simulator hyperparameters (top center).

Second, the simulator iteratively modifies the structure via Langevin dynamics based on the gradient of the energy landscape (Forces, bottom center).

Third, the imputation network constructs predicted atomic coordinates X from the final simulator time step x (T ) .

During training, the true atomic coordinates X (Data) , predicted atomic coordinates X, simulator trajectory x (1) , . . . , x (T ) , and secondary structure predictions SS (Model) feed into a composite loss function (Loss, bottom right), which is then optimized via backpropagation.

Inverse transformation The inverse transformation z = F 1 (x) involves fully local computations of bong lengths and angles.

DISPLAYFORM0 Jacobian The Jacobian @x @z defines the infinitesimal response of the Cartesian coordinates x to perturbations of the internal coordinates z. It will be important for both converting Cartesian forces into angular torques and bond forces as well as the development of our transform integrator.

It is defined element-wise as DISPLAYFORM1 The Jacobian has a simple form that can be understood by imagining the protein backbone as a robot arm that is planted at x 0 ( Figure 2B ).

Increasing or decreasing the bond length b i extends or retracts all downstream coordinates along the bonds axis, moving a bond angle a i drives circular motion of all downstream coordinates around the bond normal vectorn i centered at x i 1 , and moving a dihedral angle d i drives circular motion of downstream coordinate x j around bond vectorû i 1 centered at x i 1 .Unconstrained representations Bond lengths and angles are subject to the constraints b i > 0 and 0 < a i < ⇡.

We enforce these constraints by representing these degrees of freedom in terms of fully unconstrained variablesb i andã i via the transformations b i = log ⇣ 1 + eb i ⌘ and a i = ⇡ 1+e ã i .

All references to the internal coordinates z and Jacobians @x @z will refer to the use of fully unconstrained representations TAB3 .

FIG2 provides an overall schematic of the model, including the components of the energy function.

CNN primitives All convolutional neural network primitives in the model schematic ( FIG2 ) follow a common structure consisting of stacks of residual blocks.

Each residual block includes DISPLAYFORM0 consists of a layer of channel mixing (1x1 convolution), a variable-sized convolution layer, and a second layer of channel mixing.

We use dropout with p = 0.9 and Batch Renormalization (Ioffe, 2017) on all convolutional layers.

Batch Renormalization rather than Normalization was necessary rather owing to the large variation in sizes of the structures of the proteins and resulting large variation in mini-batch statistics.

Why sampling vs. optimization Deterministic methods for optimizing the energy U (x; s) such as gradient descent or quasi-Newton methods can effectively seek local minima of the energy surface, but are challenged to optimize globally and completely ignore the contribution of the widths of energy minima (entropy) to their probability.

We prefer sampling to optimization for three reasons: (i) noise in sampling algorithms can facilitate faster global conformational exploration by overcoming local minima and saddle points, (ii) sampling generates populations of states that respect the width (entropy) of wells in U and can be used for uncertainty quantification, and (iii) sampling allows training with an approximate Maximum Likelihood objective (Equation 5).Langevin Dynamics The Langevin dynamics are a stochastic dynamics that sample from the canonical ensemble.

They are defined as a continuous-time stochastic differential equation, and are simulated in discrete time with the first order discretization DISPLAYFORM0 Each time step of ✏ involves a descent step down the energy gradient plus a perturbation of Gaussian noise.

Importantly, as time tends toward to infinity, the time-distribution of the Langevin dynamics converges to the canonical ensemble.

Our goal is to design a dynamics that converge to an approximate sample in a very short period of time.

The comparison of this algorithm with naive integration is given in Figure 8 .

The corrector step is important for eliminating the large second-order errors that arise in curvilinear motions caused by angle changes ( Figure 2B and Figure 8 ).

In principle higher-order numerical integration methods or more time steps could increase accuracy at the cost of more evaluations of the Jacobian, but we found that second-order effects seemed to be the most relevant on our timescales.

Mixed integrator Cartesian dynamics favor local structural rearrangements, such as the transitioning from a helical to an extended conformation, while internal coordinate dynamics favor global motions such as the change of the overall fold topology.

Since both kinds of structural rearrangements are important to the folding process, we form a hybrid integrator (Algorithm 3) by taking one step with each integrator per force evaluation.

Translational and rotational detrending Both Cartesian and Internal coordinates are overparameterized with 3L degrees of freedom, since only 3L 6 degrees of freedom are necessary to encode a centered and un-oriented structure 5 .

As a consequence, a significant fraction of the per time-step changes x can be explained by rigid translational and rotational motions of the entire structure.

We isolate and remove these components of motion by treating the system {x 1 , . . . , x L } as a set of particles with unit mass, and computing effective structural translational and rotational velocities by summing point-wise momenta.

The translational component of motion is simply the average displacement across positions x Trans i = h x i i.

For rotational motion around the center of mass, it is convenient to define the non-translational motion as x i = x i x Trans i and the centered Cartesian coordinates asx i = x i hx i i.

The point-wise angular momentum is then l i =x i ⇥ x i and we define a total angular velocity of the structure !

by summing these and dividing by the moment of inertia as ! = ( DISPLAYFORM1

Input :Initial state DISPLAYFORM0 Speed clipping We found it helpful to stabilize the model by enforcing a speed limit on overall structural motions for the internal coordinate steps.

This prevents small changes to the energy function during learning from causing extreme dynamics that in turn produce a non-informative learning signal.

To accomplish this, we translationally and rotationally detrend the update of the predictor step x and compute a hypothetical time step✏ z that would limit the fastest motion to 2 Angstroms per iteration.

We then compute modified predictor and corrector steps subject to this new, potentially slower, time step.

While this breaks the asymptotics of Langevin dynamics, (i) it is unlikely on our timescales that we achieve stationarity and (ii) it can be avoided by regularizing the dynamics away from situations where clipping is necessary.

In the future, considering non-Gaussian perturbations with kinetic energies similar to Relativistic Monte Carlo (Lu et al., 2017) might accomplish a similar goal in a more principled manner.

The final integrator combining these ideas is presented in Figure 3 .

B APPENDIX B: TRAINING B.1 DATA For a training and validation set, we downloaded all protein domains of length L  200 from Classes ↵, , and ↵/ in CATH release 4.1 (2015), and then hierarchically purged a randomly selected set of A, T, and H categories.

This created three validation sets of increasing levels of difficulty: H, which contains domains with superfamilies that are excluded from train (but fold topologies may be present), T, which contains fold topologies that were excluded from train (fold generalization), and A which contains secondary structure architectures that were excluded from train.

For a test set, we downloaded all folds that were new to CATH release 4.2 (2017), which (due to a propensity of structural biology to make new structures of previously solved folds), provided 10,381 test domains.

We further stratified this test set into C, A, T, and H categories based on their nearest CATH classification in the training set.

We also analyzed test set stratifications based on nearest neighbors in both training and validation in FIG0 .

We note that the validation set was not explicitly used to tune hyperparameters due to the large cost of training ( 2 months on 2 M40 GPUs), but we did keep track of validation statistics during training.

We optimized all models for 200,000 iterations with Adam (Kingma & Ba, 2014).

We optimize the model using a composite loss containing several terms, which are detailed as follows.

Distance loss We score distances in the model with a contact-focused distance loss DISPLAYFORM0 where the contact-focusing weights are DISPLAYFORM1 is the sigmoid function.

Angle loss We use the loss DISPLAYFORM2 where DISPLAYFORM3 T are unit length feature vectors that map the angles {a i , d i } to the unit sphere.

Other angular losses, such as the negative log probability of a Von-Mises Fisher distribution, are based on the inner product of the feature vectors H(z a ) · H(z b ) rather than the Euclidean distance ||H(z a ) H(z b )|| between them.

It is worth noting that these two quantities are directly related DISPLAYFORM4 Taking z a as fixed and z b as the argument, the Euclidean loss has a cusp at z a whereas the Von-Mises Fisher loss is smooth around z a .

This is analogous to the difference between L 1 and L 2 losses, where the cusped L 1 loss favors median behavior while the smooth L 2 loss favors average behavior.

Trajectory loss In a further analogy to reinforcement learning, damped backpropation through time necessitates an intermediate loss function that can criticize transient states of the simulator.

We compute this by featurizing the per time step coordinates as the product D ijvij ( Figure 2C ) and doing the same contact-weighted averaging as the distance loss.

Template Modelling (TM) Score The TM-score BID27 , DISPLAYFORM5 is a measure of superposition quality between two protein structures on [0, 1] that was presented as an approximately length-independent alternative to RMSD.

The TM-score is the best attainable value of the preceding quantity for all possible superpositions of two structures, where DISPLAYFORM6 ||.

This requires iterative optimization, which we implemented with a sign gradient descent with 100 iterations to optimally superimpose the model and target structure.

We backpropagate through this unrolled optimization process as well as that of the simulator.

Hydrogen bond loss We determine intra-backbone hydrogen bonds using the electrostatic model of DSSP (Kabsch & Sander, 1983) .

First, we place virtual hydrogens at 1 Angstroms along the negative angle bisector of the C i 1 N i C↵ i bond angle.

Second, we compute a putative energy U h-bond ij (in kcal/mol) for each potential hydrogen bond from an amide donor at i to a carbonyl acceptor at j as DISPLAYFORM7 = 0.084 DISPLAYFORM8 where D ab = ||X i,a X j,b || is the Euclidean distance between atom a of residue i and atom b of residue j.

We then make hard assignments of hydrogen bonds for the data with DISPLAYFORM9 Published as a conference paper at ICLR 2019We 'predict' the probabilities of hydrogen bonds of the data given the model via logisitic regression of soft model assignments as DISPLAYFORM10 where a, b, c are learned parameters with the softplus parameterizations enforcing a, b > 0 and (u) = 1/(1 + exp( u) is the sigmoid function.

The final hydrogen bond loss is the cross-entropy between these predictions and the data, DISPLAYFORM11 Secondary Structure Prediction We output standard 8-class predictions of secondary structure and score them with a cross-entropy loss.

The combination of energy function, simulator, and refinement network can build an atomic level model of protein structure from sequence, and our goal is to optimize (meta-learn) this entire procedure by gradient descent.

Before going into specifics of the loss function, however, we will discuss a challenges and solutions for computing gradients of unrolled simulations in the face of chaos.

Gradient-based learning of iterative computational procedures such as Recurrent Neural Networks (RNNs) is well known to be subject to the problems of exploding and vanishing gradients BID14 .

Informally, these occur when the sensitivities of model outputs to inputs become either extremely large or extremely small and the gradient is no longer an informative signal for optimization.

We find that backpropagation through unrolled simulations such as those presented is no exception to this rule.

Often we observed that a model would productively learn for tens of thousands of iterations, only to suddenly and catastrophically exhibit diverging gradients from which the optimizer could not recover -even when the observed simulation dynamics exhibited no obvious qualitative changes to behavior and the standard solutions of gradient clipping BID14 were in effect.

Similar phenomena have been observed previously in the context of meta-learning (Maclaurin et al., 2015) and are explored in detail in a concurrent work BID12 .In FIG5 , we furnish a minimal example that illustrates how chaos can lead to irrevocable loss of learning.

We see that for even a simple particle-in-a-well, some choices of system parameters (such as too large a time step) can lead to chaotic dynamics which are synonymous with explosive gradients.

This example is hardly contrived, and is in fact a simple model of the distance potentials between coordinates in our simulations.

Moreover, it is important to note that chaos may not be easy to diagnose: for learning rates ↵ 2 [1.7, 1.8] the position of the particle x remains more or less confined in the well while the sensitivities diverge to 10 200 .

It seems unlikely that meta-learning would be able to recover after descending into chaos.

The view per time step Exploding gradients and chaotic dynamics involve the same mechanism: a multiplicative accumulation of sensitivities.

In dynamical systems this is frequently phrased as 'exponentially diverging sensitivity to initial conditions'.

Intuitively, this can be understood by examining how the Jacobian of an entire trajectory decomposes into a product of Jacobians as DISPLAYFORM0 When the norms of the per time-step Jacobians DISPLAYFORM1 @x (t 1) are typically larger than 1, the sensitivity || @x (T ) @x (0) || will grow exponentially with T .

Ideally, we would keep these norms well-behaved which is the rationale recent work on stabilization of RNNs (Henaff et al., 2016; BID9 .

Next we will offer a general-purpose regularizer to approximately enforce this goal for any differentiable computational iteration with continuous state.

dx (0) (bottom).

When the step size ↵ is small, these dynamics converge to a periodic orbit over 2 k values where 0  k < 1.

After some critical step size, the dynamics undergo a period-doubling bifurcation (Strogatz, 2018), become chaotic, and the gradients regularly diverge to huge numbers.

Approximate Lipschitz conditions One condition that guarantees that a deterministic map F : R N !

R N , x t = F (x t 1 , ✓) cannot exhibit exponential sensitivity to initial conditions is the condition of being non-expansive (also known as 1-Lipschitz or Metric).

That is, for any two input points x a , x b 2 R N , iterating the map cannot increase the distance between them as |F (x a , ✓) DISPLAYFORM2 Repplying the map to the bound immediately implies DISPLAYFORM3 for any number of iterations t. Thus, two initially close trajectories iterated through a non-expansive mapping must remain at least that close for arbitrary time.

We approximately enforce non-expansivity by performing an online sensitivity analysis within simulations.

At randomly selected time-steps, the current time step x (t) is rolled back to the preceding state and re-executed with small Gaussian perturbations to the state ⇠ N (0, 10 4 I) 6 .

We regularize the sensitivity by adding DISPLAYFORM4 to the loss.

Interestingly, the stochastic nature of this approximate regularizer is likely a good thing -a truly non-expansive map is quite limited in what it can model.

However, being 'almost' non-expansive seems to be incredibly helpful for learning.

Damped Backpropagation through Time The approximate Lipschitz conditions (or Lyapunov regularization) encourage but do not guarantee stable backpropagation.

When chaotic phasetransitions or otherwise occur we need a fall-back plan to be able to continue learning.

At the same time, we would like gradient descent to proceed in the usual manner when simulator dynamics To reduce our reliance on alignments and the generation of profiles for inference of new sequences while still leveraging evolutionary sequence data, we augmented our training set by dynamically spiking in diverse, related sequence into the model during training.

Given a set of M sequences in the alignment we sample a sequence t based on its normalized Hamming distance d t with probability DISPLAYFORM5 where EDA is a scaling parameter that we set to 5.

When the alternate sequence contains gaps, we construct a chimeric sequence that substitutes those sites with the query.

This strategy increased the number of available sequence-structure pairs by several orders of magnitude, and we used it for both profile and 1-seq based training.

C APPENDIX C: RESULTS

For each sequence from the CATH release 4.2 dataset, 100 structures were generated from both the profile and sequence-only models, while a single structure was generated from the RNN baseline models.

The reported TM-scores were calculated using Maxcluster BID19 .

A single representative structure was chosen from the ensemble of 100 structures using 3D-Jury (Ginalski et al., 2003) .

A pairwise distance matrix of TM-scores was calculated for all of the 100 structures in the ensemble.

Clusters were determined by agglomerative hierarchical clustering with complete linkage using a TM-score threshold of 0.5 to determine cluster membership.

NEMO RNN, 300 hidden units FIG0 : Sampling speed.

Per-protein sampling times for various batch sizes across NEMO and one of the RNN baselines on a single Tesla M40 GPU with 12GB memory and 20 cores.

For all results in the main paper, 100 models were sampled per protein followed by consensus clustering with 3D-jury, adding an additional factor of 10 2 cost between NEMO and the RNN.

<|TLDR|>

@highlight

We use an unrolled simulator as an end-to-end differentiable model of protein structure and show it can (sometimes) hierarchically generalize to unseen fold topologies.