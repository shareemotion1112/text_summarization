Graph neural networks have recently achieved great successes in predicting quantum mechanical properties of molecules.

These models represent a molecule as a graph using only the distance between atoms (nodes) and not the spatial direction from one atom to another.

However, directional information plays a central role in empirical potentials for molecules, e.g. in angular potentials.

To alleviate this limitation we propose directional message passing, in which we embed the messages passed between atoms instead of the atoms themselves.

Each message is associated with a direction in coordinate space.

These directional message embeddings are rotationally equivariant since the associated directions rotate with the molecule.

We propose a message passing scheme analogous to belief propagation, which uses the directional information by transforming messages based on the angle between them.

Additionally, we use spherical Bessel functions to construct a theoretically well-founded, orthogonal radial basis that achieves better performance than the currently prevalent Gaussian radial basis functions while using more than 4x fewer parameters.

We leverage these innovations to construct the directional message passing neural network (DimeNet).

DimeNet outperforms previous GNNs on average by 77% on MD17 and by 41% on QM9.

In recent years scientists have started leveraging machine learning to reduce the computation time required for predicting molecular properties from a matter of hours and days to mere milliseconds.

With the advent of graph neural networks (GNNs) this approach has recently experienced a small revolution, since they do not require any form of manual feature engineering and significantly outperform previous models .

GNNs model the complex interactions between atoms by embedding each atom in a high-dimensional space and updating these embeddings by passing messages between atoms.

By predicting the potential energy these models effectively learn an empirical potential function.

Classically, these functions have been modeled as the sum of four parts: (Leach, 2001 )

where E bonds models the dependency on bond lengths, E angle on the angles between bonds, E torsion on bond rotations, i.e. the dihedral angle between two planes defined by pairs of bonds, and E non-bonded models interactions between unconnected atoms, e.g. via electrostatic or van der Waals interactions.

The update messages in GNNs, however, only depend on the previous atom embeddings and the pairwise distances between atoms -not on directional information such as bond angles and rotations.

Thus, GNNs lack the second and third terms of this equation and can only model them via complex higher-order interactions of messages.

Extending GNNs to model them directly is not straightforward since GNNs solely rely on pairwise distances, which ensures their invariance to translation, rotation, and inversion of the molecule, which are important physical requirements.

In this paper, we propose to resolve this restriction by using embeddings associated with the directions to neighboring atoms, i.e. by embedding atoms as a set of messages.

These directional message embeddings are equivariant with respect to the above transformations since the directions move with the molecule.

Hence, they preserve the relative directional information between neighboring atoms.

We propose to let message embeddings interact based on the distance between atoms and the angle between directions.

Both distances and angles are invariant to translation, rotation, and inversion of the molecule, as required.

Additionally, we show that the distance and angle can be jointly represented in a principled and effective manner by using spherical Bessel functions and spherical harmonics.

We leverage these innovations to construct the directional message passing neural network (DimeNet).

DimeNet can learn both molecular properties and atomic forces.

It is twice continuously differentiable and solely based on the atom types and coordinates, which are essential properties for performing molecular dynamics simulations.

DimeNet outperforms previous GNNs on average by 76 % on MD17 and by 31 % on QM9.

Our paper's main contributions are:

1.

Directional message passing, which allows GNNs to incorporate directional information by connecting recent advances in the fields of equivariance and graph neural networks as well as ideas from belief propagation and empirical potential functions such as Eq. 1.

2.

Theoretically principled orthogonal basis representations based on spherical Bessel functions and spherical harmonics.

Bessel functions achieve better performance than Gaussian radial basis functions while reducing the radial basis dimensionality by 4x or more.

3.

The Directional Message Passing Neural Network (DimeNet): A novel GNN that leverages these innovations to set the new state of the art for molecular predictions and is suitable both for predicting molecular properties and for molecular dynamics simulations.

ML for molecules.

The classical way of using machine learning for predicting molecular properties is combining an expressive, hand-crafted representation of the atomic neighborhood (Bart??k et al., 2013) with Gaussian processes (Bart??k et al., 2010; or neural networks (Behler & Parrinello, 2007) .

Recently, these methods have largely been superseded by graph neural networks, which do not require any hand-crafted features but learn representations solely based on the atom types and coordinates molecules (Duvenaud et al., 2015; Unke & Meuwly, 2019) .

Our proposed message embeddings can also be interpreted as directed edge embeddings. (Undirected) edge embeddings have already been used in previous GNNs (J??rgensen et al., 2018; Chen et al., 2019) .

However, these GNNs use both node and edge embeddings and do not leverage any directional information.

Graph neural networks.

GNNs were first proposed in the 90s (Baskin et al., 1997; Sperduti & Starita, 1997) and 00s (Gori et al., 2005; Scarselli et al., 2009 ).

General GNNs have been largely inspired by their application to molecular graphs and have started to achieve breakthrough performance in various tasks at around the same time the molecular variants did (Kipf & Welling, 2017; Klicpera et al., 2019; Zambaldi et al., 2019) .

Some recent progress has been focused on GNNs that are more powerful than the 1-Weisfeiler-Lehman test of isomorphism (Morris et al., 2019; Maron et al., 2019) .

However, for molecular predictions these models are significantly outperformed by GNNs focused on molecules (see Sec. 7).

Some recent GNNs have incorporated directional information by considering the change in local coordinate systems per atom (Ingraham et al., 2019) .

However, this approach breaks permutation invariance and is therefore only applicable to chain-like molecules (e.g. proteins).

Equivariant neural networks.

Group equivariance as a principle of modern machine learning was first proposed by Cohen & Welling (2016) .

Following work has generalized this principle to spheres , molecules (Thomas et al., 2018) , volumetric data (Weiler et al., 2018) , and general manifolds (Cohen et al., 2019) .

Equivariance with respect to continuous rotations has been achieved so far by switching back and forth between Fourier and coordinate space in each layer or by using a fully Fourier space model (Kondor et al., 2018; Anderson et al., 2019) .

The former introduces major computational overhead and the latter imposes significant constraints on model construction, such as the inability of using non-linearities.

Our proposed solution does not suffer from either of those limitations.

In recent years machine learning has been used to predict a wide variety of molecular properties, both low-level quantum mechanical properties such as potential energy, energy of the highest occupied molecular orbital (HOMO), and the dipole moment and high-level properties such as toxicity, permeability, and adverse drug reactions (Wu et al., 2018) .

In this work we will focus on scalar regression targets, i.e. targets t ??? R. A molecule is uniquely defined by the atomic numbers z = {z 1 , . . .

, z N } and positions X = {x 1 , . . .

, x N }.

Some models additionally use auxiliary information ?? such as bond types or electronegativity of the atoms.

We do not include auxiliary features in this work since they are hand-engineered and non-essential.

In summary, we define an ML model for molecular prediction with parameters ?? via f ?? : {X, z} ??? R.

Symmetries and invariances.

All molecular predictions must obey some basic laws of physics, either explicitly or implicitly.

One important example of such are the fundamental symmetries of physics and their associated invariances.

In principle, these invariances can be learned by any neural network via corresponding weight matrix symmetries (Ravanbakhsh et al., 2017) .

However, not explicitly incorporating them into the model introduces duplicate weights and increases training time and complexity.

The most essential symmetries are translational and rotational invariance (follows from homogeneity and isotropy), permutation invariance (follows from the indistinguishability of particles), and symmetry under parity, i.e. under sign flips of single spatial coordinates.

Molecular dynamics.

Additional requirements arise when the model should be suitable for molecular dynamics (MD) simulations and predict the forces F i acting on each atom.

The force field is a conservative vector field since it must satisfy conservation of energy (the necessity of which follows from homogeneity of time (Noether, 1918) ).

The easiest way of defining a conservative vector field is via the gradient of a potential function.

We can leverage this fact by predicting a potential instead of the forces and then obtaining the forces via backpropagation to the atom coordinates, i.e.

We can even directly incorporate the forces in the training loss and directly train a model for MD simulations (Pukrittayakamee et al., 2009) :

where the targett =?? is the ground-truth energy (usually available as well),F are the ground-truth forces, and the hyperparameter ?? sets the forces' loss weight.

For stable simulations F i must be continuously differentiable and the model f ?? itself therefore twice continuously differentiable.

We hence cannot use discontinuous transformations such as ReLU non-linearities.

Furthermore, since the atom positions X can change arbitrarily we cannot use pre-computed auxiliary information ?? such as bond types.

Graph neural networks.

Graph neural networks treat the molecule as a graph, in which the nodes are atoms and edges are defined either via a predefined molecular graph or simply by connecting atoms that lie within a cutoff distance c. Each edge is associated with a pairwise distance between atoms d ij = x i ??? x j 2 .

GNNs implement all of the above physical invariances by construction since they only use pairwise distances and not the full atom coordinates.

However, note that a predefined molecular graph or a step function-like cutoff cannot be used for MD simulations since this would introduce discontinuities in the energy landscape.

GNNs represent each atom i via an atom embedding h i ??? R H .

The atom embeddings are updated in each layer by passing messages along the molecular edges.

Messages are usually transformed based on an edge embedding e (ij) ??? R He and summed over the atom's neighbors N i , i.e. the embeddings are updated in layer l via

with the update function f update and the interaction function f int , which are both commonly implemented using neural networks.

The edge embeddings e (l) (ij) usually only depend on the interatomic distances, but can also incorporate additional bond information or be recursively updated in each layer using the neighboring atom embeddings (J??rgensen et al., 2018) .

Directionality.

In principle, the pairwise distance matrix contains the full geometrical information of the molecule.

However, GNNs do not use the full distance matrix since this would mean passing messages globally between all pairs of atoms, which increases computational complexity and can lead to overfitting.

Instead, they usually use a cutoff distance c, which means they cannot distinguish between certain molecules (Xu et al., 2019) .

E.g. at a cutoff of roughly 2 ?? a regular GNN would not be able to distinguish between a hexagonal (e.g. Cyclohexane) and two triangular molecules (e.g. Cyclopropane) with the same bond lengths since the neighborhoods of each atom are exactly the same for both (see Appendix, Fig. 6 ).

This problem can be solved by modeling the directions to neighboring atoms instead of just their distances.

A principled way of doing so while staying invariant to a transformation group G (such as described in Sec. 3) is via group-equivariance (Cohen & Welling, 2016)

with the group action in the input and output space ?? X g and ?? Y g .

However, equivariant CNNs only achieve equivariance with respect to a discrete set of rotations (Cohen & Welling, 2016) .

For a precise prediction of molecular properties we need continuous equivariance with respect to rotations, i.e. to the SO(3) group.

Directional embeddings.

We solve this problem by noting that an atom by itself is rotationally invariant.

This invariance is only broken by neighboring atoms that interact with it, i.e. those inside the cutoff c. Since each neighbor breaks up to one rotational invariance they also introduce additional degrees of freedom, which we need to represent in our model.

We can do so by generating a separate embedding m ji for each atom i and neighbor j by applying the same learned filter in the direction of each neighboring atom (in contrast to equivariant CNNs, which apply filters in fixed, global directions).

These directional embeddings are equivariant with respect to global rotations since the associated directions rotate with the molecule and hence conserve the relative directional information between neighbors.

Representation via joint 2D basis.

We use the directional information associated with each embedding by leveraging the angle ?? (kj,ji) = ???x k x j x i when aggregating the neighboring embeddings m kj of m ji .

We combine the angle with the interatomic distance d kj associated with the incoming message m kj and jointly represent both in a (kj,ji) SBF ??? R NSHBF??NSRBF using a 2D representation based on spherical Bessel functions and spherical harmonics, as explained in Sec. 5.

We empirically found that this basis representation provides a better inductive bias than the raw angle alone.

Message embeddings.

The directional embedding m ji associated with the atom pair ji can be thought of as a message being sent from atom j to atom i. Hence, in analogy to belief propagation, we embed each atom i using a set of incoming messages m ji , i.e. h i = j???Ni m ji , and update the message m ji based on the incoming messages m kj (Yedidia et al., 2003) .

Hence, as illustrated in Fig. 1 , we define the update function and aggregation scheme for message embeddings as

where e (ji)

RBF denotes the radial basis function representation of the interatomic distance d ji , which will be discussed in Sec. 5.

We found this aggregation scheme to not only have a nice analogy to belief propagation, but also to empirically perform better than alternatives.

Note that since f int now incorporates the angle between atom pairs, or bonds, we have enabled our model to directly learn the angular potential E angle , the second term in Eq. 1.

Moreover, the message embeddings are essentially embeddings of atom pairs, as used by the provably more powerful GNNs based on higher-order Weisfeiler-Lehman tests of isomorphism.

Our model can therefore provably distinguish molecules that a regular GNN cannot (e.g. the previous example of a hexagonal and two triangular molecules) (Morris et al., 2019) .

Representing distances and angles.

For the interaction function f int in Eq. 4 we use a joint representation a (kj,ji) SBF of the angles ?? (kj,ji) between message embeddings and the interatomic distances d kj = x k ??? x j 2 , as well as a representation e (ji) RBF of the distances d ji .

Earlier works have used a set of Gaussian radial basis functions to represent interatomic distances, with tightly spaced means that are distributed e.g. uniformly or exponentially (Unke & Meuwly, 2019) .

Similar in spirit to the functional bases used by steerable CNNs (Cohen & Welling, 2017; Cheng et al., 2019) we propose to use an orthogonal basis instead, which reduces redundancy and thus improves parameter efficiency.

Furthermore, a basis chosen according to the properties of the modeled system can even provide a helpful inductive bias.

We therefore derive a proper basis representation for quantum systems next.

with the spherical Bessel functions of the first and second kind j l and y l and the spherical harmonics Y m l .

As common in physics we only use the regular solutions, i.e. those that do not approach ?????? at the origin, and hence set b lm = 0.

Recall that our first goal is to construct a joint 2D basis for d kj and ?? (kj,ji) , i.e. a function that depends on d and a single angle ??.

To achieve this we set m = 0 and obtain

.

The boundary conditions are satisfied by setting k = z ln c , where z ln is the n-th root of the l-order Bessel function, which are precomputed numerically.

Normalizing ?? SBF inside the cutoff distance c yields the 2D spherical Fourier-Bessel basis?? (kj,ji) SBF ??? R NSHBF??NSRBF , which is illustrated in Fig. 2 and defined by??

with n ??? [1 . .

N RBF ].

Both of these bases are purely real-valued and orthogonal in the domain of interest.

They furthermore enable us to bound the highest-frequency components by ?? ?? ??? NSHBF 2?? , ?? d kj ??? NSRBF c , and ?? dji ??? NRBF c .

This restriction is an effective way of regularizing the model and ensures that predictions are stable to small perturbations.

We found N SRBF = 6 and N RBF = 16 radial basis functions to be more than sufficient.

Note that N RBF is 4x lower than PhysNet's 64 (Unke & Meuwly, 2019) and 20x lower than SchNet's 300 radial basis functions .

and their first and second derivatives to go to 0 at the cutoff.

We achieve this with the polynomial where p ??? N 0 .

We did not find the model to be sensitive to different choices of envelope functions and choose p = 3.

Note that using an envelope function causes the bases to lose their orthonormality, which we did not find to be a problem in practice.

We furthermore fine-tune the Bessel wave numbers k n = n?? c used in??? RBF ??? R NRBF via backpropagation after initializing them to these values, which we found to give a small boost in prediction accuracy.

The Directional Message Passing Neural Network's (DimeNet) design is based on a streamlined version of the PhysNet architecture (Unke & Meuwly, 2019) , in which we have integrated directional message passing and spherical Fourier-Bessel representations.

DimeNet generates predictions that are invariant to atom permutations and translation, rotation and inversion of the molecule.

DimeNet is suitable both for the prediction of various molecular properties and for molecular dynamics (MD) simulations.

It is twice continuously differentiable and able to learn and predict atomic forces via backpropagation, as described in Sec. 3.

The predicted forces fulfill energy conservation by construction and are equivariant with respect to permutation and rotation.

Model differentiability in combination with basis representations that have bounded maximum frequencies furthermore guarantees smooth predictions that are stable to small deformations.

Fig. 4 gives an overview of the architecture.

Embedding block.

Atomic numbers are represented by learnable, randomly initialized atom type embeddings h (0) i ??? R F that are shared across molecules.

The first layer generates message embeddings from these and the distance between atoms via

where denotes concatenation and the weight matrix W and bias b are learnable.

Interaction block.

The embedding block is followed by multiple stacked interaction blocks.

This block implements f int and f update of Eq. 4 as shown in Fig. 4 .

Note that the 2D representation a (kj,ji) SBF is first transformed into an N tensor -dimensional representation via a linear layer.

The main purpose of this is to make the dimensionality of a (kj,ji) SBF independent of the subsequent bilinear layer, which uses a comparatively large N tensor ?? F ?? F -dimensional weight tensor.

We have also experimented with using a bilinear layer for the radial basis representation, but found that the element-wise multiplication e

RBF W m kj performs better, which suggests that the 2D representations require more complex transformations than radial information alone.

The interaction block transforms each message embedding m ji using multiple residual blocks, which are inspired by ResNet (He et al., 2016) and consist of two stacked dense layers and a skip connection.

Output block.

The message embeddings after each block (including the embedding block) are passed to an output block.

The output block transforms each message embedding m ji using the radial basis e (ji) RBF , which ensures continuous differentiability and slightly improves performance.

Afterwards the incoming messages are summed up per atom i to obtain h i = j m ji , which is then transformed using multiple dense layers to generate the atom-wise output t (l) i .

These outputs are then summed up to obtain the final prediction t = i l t (l) i .

Continuous differentiability.

Multiple model choices were necessary to achieve twice continuous model differentiability.

First, DimeNet uses the self-gated Swish activation function ??(x) = x ?? sigmoid(x) (Ramachandran et al., 2018) instead of a regular ReLU activation function.

Second, we multiply the radial basis functions??? RBF (d) with an envelope function u(d) that has a root of multiplicity 3 at the cutoff c. Finally, DimeNet does not use any auxiliary data but relies on atom types and positions alone.

Models.

For hyperparameter choices and training setup see Appendix B.

We use 6 state-of-the-art models for comparison: SchNet , PhysNet (whose results we have generated ourselves using the reference implementation) (Unke & Meuwly, 2019), provably powerful graph networks (PPGN) (Maron et al., 2019) , MEGNet-simple (the variant without auxiliary information) (Chen et al., 2019) , Cormorant (Anderson et al., 2019) , and symmetrized gradient-domain machine learning (sGDML) (Chmiela et al., 2018) .

Note that sGDML cannot be used for QM9 since it can only be trained on a single molecule.

We test DimeNet's performance for predicting molecular properties using the common QM9 benchmark (Ramakrishnan et al., 2014) .

It consists of roughly 130 000 molecules in equilibrium with up to 9 heavy C, O, N, and F atoms.

We use 110 000 molecules in the training, 10 000 in the validation and 13 885 in test set.

We only use the atomization energy for U 0 , U , H, and G, i.e. subtract the atomic reference energies, which are constant per atom type.

In Table 1 we report the mean absolute error (MAE) of each target and the overall mean standardized MAE (std.

MAE) and mean standardized logMAE (for details see Appendix C).

We predict ??? simply by taking

.

We use MD17 to test model performance in molecular dynamics simulations.

The goal of this benchmark is predicting both the energy and atomic forces of eight small organic molecules, given the atom coordinates of the thermalized (i.e. non-equilibrium, slightly moving) system.

The ground truth data is computed via molecular dynamics simulations using DFT.

A separate model is trained for each molecule, with the goal of providing highly accurate individual predictions.

This dataset is commonly used with 50 000 training and 10 000 validation and test samples.

We found that DimeNet can match state-of-the-art performance in this setup.

E.g. for Benzene, depending on the force weight ??, DimeNet achieves 0.035 kcal mol ???1 MAE for the energy or 0.07 kcal mol ???1 and 0.17 kcal mol

for energy and forces, matching the results reported by Anderson et al. (2019) and Unke & Meuwly (2019) .

However, this accuracy is two orders of magnitude below the DFT calculation's accuracy (approx.

2.3 kcal mol ???1 for energy (Faber et al., 2017) ), so any remaining difference to real-world data is almost exclusively due to errors in the DFT simulation.

Truly reaching better accuracy can therefore only be achieved with more precise ground-truth data, which requires far more expensive methods (e.g. CCSD(T)) and thus ML models that are more sample-efficient (Chmiela et al., 2018) .

We therefore instead test our model on the harder task of using only 1000 training samples.

As shown in Table 2 DimeNet outperforms SchNet by a large margin and performs roughly on par with sGDML.

However, sGDML uses hand-engineered descriptors that provide a strong advantage for small datasets, can only be trained on a single molecule (a fixed set of atoms), and does not scale well with the number of atoms or training samples.

Ablation studies.

To test whether directional message passing and the Fourier-Bessel basis are the actual reason for DimeNet's improved performance, we ablate them individually and compare the mean standardized MAE and logMAE for multi-task learning on QM9.

Table 3 shows that both of our contributions have a significant impact on the model's performance.

Using 64 Gaussian RBFs instead of 16 and 6 Bessel basis functions to represent d ji and d kj increases the error by 10 %, which shows that this basis does not only reduce the number of parameters but additionally provides a helpful inductive bias.

DimeNet's error increases by around 26 % when we ignore the angles between messages by setting N SHBF = 1, showing that directly incorporating directional information does indeed improve performance.

Using node embeddings instead of message embeddings (and hence also ignoring directional information) has the largest impact and increases MAE by 68 %, at which point DimeNet performs worse than SchNet.

Furthermore, Fig. 5 shows that the filters exhibit a structurally meaningful dependence on both the distance and angle.

For example, some of these filters are clearly being activated by benzene rings (120

??? angle, 1.39 ?? distance).

This further demonstrates that the model learns to leverage directional information.

In this work we have introduced directional message passing, a more powerful and expressive interaction scheme for molecular predictions.

Directional message passing enables graph neural networks to leverage directional information in addition to the interatomic distances that are used by normal GNNs.

We have shown that interatomic distances can be represented in a principled and effective manner using spherical Bessel functions.

We have furthermore shown that this representation can be extended to directional information by leveraging 2D spherical Fourier-Bessel basis functions.

We have leveraged these innovations to construct DimeNet, a GNN suitable both for predicting molecular properties and for use in molecular dynamics simulations.

We have demonstrated DimeNet's performance on QM9 and MD17 and shown that our contributions are the essential ingredients that enable DimeNet's state-of-the-art performance.

DimeNet directly models the first two terms in Eq. 1, which are known as the important "hard" degrees of freedom in molecules (Leach, 2001) .

Future work should aim at also incorporating the third and fourth terms of this equation.

This could improve predictions even further and enable the application to molecules much larger than those used in common benchmarks like QM9.

Figure 6 : A standard non-directional GNN cannot distinguish between a hexagonal (left) and two triangular molecules (right) with the same bond lengths, since the neighborhood of each atom is exactly the same.

An example of this would be Cyclohexane and two Cyclopropane molecules with slightly stretched bonds, when the GNN either uses the molecular graph or a cutoff distance of c ??? 2.5 ??. Directional message passing solves this problem by considering the direction of each bond.

The model architecture and hyperparameters were optimized using the QM9 validation set.

We use 6 stacked interaction blocks and embeddings of size F = 128 throughout the model.

For the basis functions we choose N SHBF = 7, N SRBF = 6, and N RBF = 16 and N tensor = 12 for the weight tensor in the interaction block.

We did not find the model to be very sensitive to these values as long as they were chosen large enough (i.e. at least 8).

To illustrate the filters learned by DimeNet we separate the spatial dependency in the interaction function f int via f int (m, d ji , d kj , ?? (kj,ji) ) = n [??(W m + b)]

n f filter1,n (d ji )f filter2,n (d kj , ?? (kj,ji) ).

The filters f filter1,n : R + ??? R and f filter2,n : R + ?? [0, 2??] ??? R F are given by

where W RBF , W SBF , and W are learned weight matrices/tensors, e RBF (d) is the radial basis representation, and a SBF (d, ??) is the 2D spherical Fourier-Bessel representation.

Fig. 5 shows how the first 15 elements of f filter2,n (d, ??) vary with d and ?? when choosing the tensor slice n = 1 (with ?? = 0 at the top of the figure) .

E MULTI-TARGET RESULTS

@highlight

Directional message passing incorporates spatial directional information to improve graph neural networks.