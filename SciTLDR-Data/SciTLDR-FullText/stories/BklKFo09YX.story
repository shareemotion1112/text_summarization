Designing a molecule with desired properties is one of the biggest challenges in drug development, as it requires optimization of chemical compound structures with respect to many complex properties.

To augment the compound design process we introduce Mol-CycleGAN -- a CycleGAN-based model that generates optimized compounds with a chemical scaffold of interest.

Namely, given a molecule our model generates a structurally similar one with an optimized value of the considered property.

We evaluate the performance of the model on selected optimization objectives related to structural properties (presence of halogen groups, number of aromatic rings) and to a physicochemical property (penalized logP).

In the task of optimization of penalized logP of drug-like molecules our model significantly outperforms previous results.

The principal goal of the drug design process is to find new chemical compounds that are able to modulate the activity of a given target (typically a protein) in a desired way BID27 ).

However, finding such molecules in high-dimensional chemical space of all molecules without any prior knowledge is nearly impossible.

In silico methods have been introduced to leverage the existing chemical, pharmacological and biological knowledge, thus forming a new branch of science -computer-aided drug design (CADD) BID26 BID0 .

In particular, the recent advancements in deep learning encouraged its application to CADD BID4 .

Computer methods are nowadays applied at every stage of drug design pipelines BID26 from the search of new, potentially active compounds BID22 , through optimization of their activity and physicochemical profile BID15 and simulating their scheme of interaction with the target protein BID9 , to assisting in planning the synthesis and evaluation of its difficulty BID30 .In the center of our interest are the hit-to-lead and lead optimization phases of the compound design process.

Their goals are to optimize drug-like molecules identified in previous steps in terms of, respectively, the desired activity profile (increased potency towards given target protein and provision of inactivity towards undesired proteins) and the physicochemical and pharmacokinetic properties.

The challenge here is to optimize a molecule with respect to multiple properties simultaneously BID15 .Our principal contribution is the introduction of Mol-CycleGAN, a generative model based on CycleGAN BID36 with the goal to augment the compound design process.

We show that our model can generate molecules that possess desired properties 1 while retaining their chemical scaffolds.

Given a starting molecule, the model generates a similar one but with a desired characteristics.

The similarity between the two molecules is important in the context of multiparameter optimization, as it makes it easier to optimize the selected property without spoiling the previously optimized ones.

To the best of our knowledge, this is the first approach to molecule generation that uses the CycleGAN architecture.

We evaluate our model on its ability to perform structural transformations and molecular optimization.

The former indicates that the model is able to do simple structural modifications such as a change in the presence of halogen groups or number of aromatic rings.

In the latter, we aim to maximize penalized logP to assess the model's utility for compound design.

Penalized logP is a physicochemical property often selected as a testing ground for molecule optimization models BID18 BID35 , as it is relevant in the drug design process.

In the optimization of penalized logP for drug-like molecules our model significantly outperforms previous results.

There are two main approaches of applying deep learning in drug design (see BID4 for a recent review): paq use discriminative models to screen commercial databases and classify molecules as likely active or likely inactive (virtual screening); pbq use generative models to propose novel molecules that likely possess the desired properties.

The former application already proved to give outstanding results BID10 BID17 BID6 BID25 .

The latter use case is rapidly emerging.

Many generative deep learning models have already been applied in the compound design context.

Initial molecule generation models mostly operate on SMILES strings BID33 .

Long short-term memory (LSTM) network architecture is applied in BID29 ; Bjerrum & Threlfall (2017); BID34 ; BID14 .

Variational Autoencoder (VAE) BID20 ) is used by BID11 to generate SMILES of new molecules.

Unfortunately, these models can generate invalid SMILES that do not correspond to any molecules.

Introduction of grammars into the model improved the success rate of valid SMILES generation BID21 BID7 .

Maintaining chemical validity within a generative process became possible through VAEs realized directly on molecular graphs BID31 BID18 .Generative Adversarial Networks (GANs) BID12 are an alternative architecture that has been applied to de novo drug design.

BID13 propose GANs and Reinforcement Learning (RL) model (based on SMILES), which generates samples that fulfill desired objectives while promoting diversity.

BID8 use GANs and RL, together with graph representation (adjacency and annotation matrices) to generate new molecules with the given properties.

BID35 train convolutional GANs on molecular graphs and use RL to ensure that the proposed molecules have logP and molecular weight in the desired range.

Indeed, some of these models can be used to effectively search through the chemical space.

Nevertheless, these approaches are not without flaws.

The generated compounds can be, e.g., difficult or impossible to synthesize.

We address this issue by proposing Mol-CycleGAN, a generative model designed to generate molecules with the desired properties while retaining their chemical scaffolds.

Such a model can prove to be very useful for optimizing active molecules towards a given property, which is essential in compound design.

We introduce Mol-CycleGAN to perform optimization by learning from the sets of molecules with or without the desired molecular property (denoted by the sets X and Y ).

Our approach is to train a model to perform the transformation G : X Ñ Y and then use this model to perform optimization of molecules.

In the context of compound design X (Y ) can be, e.g., the set of inactive (active) molecules.

To represent the sets X and Y our approach requires an embedding of molecules, from which it should be possible to decode the coordinates back into some complete representation (e.g., the SMILES representation).

Here, the latent space of variational autoencoders can be used.

This has the added benefit that the distance between molecules (required to calculate the loss function) can be defined in the latent space.

Essential chemical properties are easier to express on graphs rather than linear SMILES representations BID33 .

This is why for molecule representation we use the latent space obtained from Junction Tree Variational Autoencoder (JT-VAE) BID18 .

JT-VAE is based on a graph structure of molecules and shows superior properties compared to SMILES-based VAEs (cf. also the discussion in Section 2).

One could also try formulating the CycleGAN on the SMILES representation directly, but this would raise the problem of defining the intermolecular distance, as the standard manners of measuring similarity between molecules (Tanimoto similarity) are non-differentiable.

Our approach extends the CycleGAN framework BID36 to molecular embeddings, created by JT-VAE BID18 .

We represent each molecule with a latent vector, given by the mean of the variational encoding distribution.

The inclusion of the cyclic component acts as a regularization and may also help in the regime of low data, as the model can learn from both directions of the transformation.

With the cyclic component the resulting model is more robust (cf. e.g. the comparison of non-cyclic IcGAN BID24 vs CycleGAN in BID5 ).

Our model works as follows (cf.

FIG0 ): (i) we start by defining the sets X and Y (e.g., active/inactive molecules); (ii) we introduce the mapping functions G : X Ñ Y and F : Y Ñ X; (iii) we introduce discriminators D X (and D Y ) which force the generator F (and G) to generate samples from a distribution close to the distribution of X (or Y ).

The components F , G, D X , and D Y are implemented with neural networks acting in the latent space (see Appendix A for technical details).After training the model we perform optimization of a given molecule by: (i) computing its latent space embedding, x; (ii) using the generating function to compute Gpxq; (iii) decoding the latent space coordinates given by Gpxq to obtain the SMILES representation of the optimized molecule.

Thereby, any molecule can be cast onto the set of molecules with the desired property, Y .For training the Mol-CycleGAN we use the following loss function: DISPLAYFORM0 and aim to solve G˚, F˚" arg min DISPLAYFORM1 We use the adversarial loss introduced in LS-GAN BID23 ) DISPLAYFORM2 which ensures that the generator G (and F ) generates samples from a distribution close to the distribution of Y (or X).

The cycle consistency loss DISPLAYFORM3 is responsible for reducing the space of possible mapping functions, such that for a molecule x from set X, the GAN cycle brings it back to a molecule similar to x, i.e. F pGpxqq is close to x (and analogously GpF pyqq is close to y).

Finally, to ensure that the optimized molecule is close to the starting one we use the identity mapping loss BID36 L identity pG, F q " E y"p data pyq r}F pyq´y} 1 s`E x"p data pxq r}Gpxq´x} 1 s, which further reduces the space of possible mapping functions and prevents the model from generating molecules that lay far away from the starting molecule in the JT-VAE latent space.

In all our experiments we use the values of hyperparameters λ 1 " 0.3 and λ 2 " 0.1, which were chosen by checking a couple of combinations (for structural tasks) and verifying that our optimization process: (i) improves the studied property and (ii) generates molecules similar to the starting ones.

We have not performed a grid search for optimal values of λ 1 and λ 2 , and hence there could be space for improvement here.

Note that these parameters control the balance between improvement in the optimized property and similarity between the generated and the starting molecule.

Both the improvement and the similarity can be obtained with our model, as we show in the next section.

We conduct experiments to test whether the proposed model is able to generate molecules that possess desired properties and are close to the starting molecules.

Namely, we evaluate the model on tasks related to structural modifications, as well as on tasks related to molecule optimization.

For testing molecule optimization we select the octanol-water partition coefficient (logP) penalized by the synthetic accessibility (SA) score.

logP describes lipophilicity -a parameter influencing a whole set of other characteristics of compounds such as solubility, permeability through biological membranes, ADME (absorption, distribution, metabolism, and excretion) properties, and toxicity.

We use the formulation as in BID18 (see Appendix D therein).

Explicitly, for molecule m the penalized logP is given as logP pmq´SApmq.

We use the ZINC-250K dataset used earlier by Kusner et al. (2017) ; BID18 which contains 250 000 drug-like molecules extracted from the ZINC database BID32 .

Molecular similarity and drug-likeness are achieved in all experiments.

The detailed formulation of the tasks is the following:• Structural transformations We test the model's ability to perform simple structural transformations of the molecules: • Constrained molecule optimization We optimize penalized logP, while constraining the degree of deviation from the original molecule.

The similarity between molecules is measured with Tanimoto similarity on Morgan Fingerprints BID28 .

The set X (Y ) is a random sample from ZINC-250K of the compounds with penalized logP below (above) median.

Here we follow the task previously proposed in BID18 .

DISPLAYFORM0 • Unconstrained molecule optimization We perform unconstrained optimization of penalized logP. The set X is a random sample from ZINC-250K and the set Y is a random sample from the top-20% molecules with the highest penalized logP in ZINC-250K.

In each structural experiment, we test the model's ability to perform simple transformations of molecules in both directions X Ñ Y and Y Ñ X. Here, X and Y are non-overlapping sets of molecules with a specific structural property.

We start with experiments on structural properties because they are easier to interpret and the rules related to transforming between X and Y are well defined.

Hence, the present task should be easier for the model, as compared to the optimization of complex molecular properties, for which there are no simple rules connecting X and Y .

Table 1 : Evaluation of models modifying the presence of halogen moieties and the number of aromatic rings.

Success rate is the fraction of times when a desired modification occurs.

Non-identity is the fraction of times when the generated molecule is different from the starting one.

Uniqueness is the fraction of unique molecules in the set of generated molecules.

In Table 1 we show the success rates for the tasks of performing structural transformations of molecules.

The task of changing the number of aromatic rings is more difficult than changing the presence of halogen moieties.

In the former the transition between X (with 2 rings) and Y (with 1 or 3 rings, cf.

FIG2 ) is more than a simple addition/removal as it is in the other case.

This is reflected in the success rates which are higher for the halogen moieties task.

In the dataset used to construct the latent space (ZINC-250K) 64.9 % molecules do not contain any halogen moiety, whereas the remaining 35.1 % contain one or more halogen moieties.

This imbalance might be the reason for the higher success rate in the task of removing halogen moieties (Y Ñ F pY q).

To confirm that the generated molecules are close to the starting ones, we show in Fig. 3 distributions of their Tanimoto similarities (using Morgan fingerprints).

For comparison we also include distributions of the Tanimoto similarities between the starting molecule and a random molecule from the ZINC-250K dataset.

The high similarities between the generated and the starting molecules show that our procedure is neither a random sampling from the latent space, nor a memorization of the manifold in the latent space with the desired property value.

We also visualize the molecules, which after transformation are the most similar to the starting molecules in Fig. 4 .

As our main task we optimize the desired property under the constraint that the similarity between the original and the generated molecule is higher than some fixed threshold.

This is a more realistic scenario in drug discovery, where the development of new drugs usually starts with known molecules such as existing drugs .

Here, we maximize the penalized logP coefficient and use the Tanimoto similarity with the Morgan fingerprint BID28 to define the threshold of similarity, simpm, m 1 q ě δ.

We compare our results with BID18 and BID35 .In our optimization procedure each molecule (given by the latent space coordinates x) is fed into the generator to obtain the 'optimized' molecule Gpxq.

The pair px, Gpxqq defines what we call 'optimization path' in the JT-VAE latent space.

To be able to make a comparison with BID18 we start the procedure from the 800 molecules with the lowest values of penalized logP in From the resulting set of K molecules we report the molecule with the highest penalized logP score that satisfies the similarity constraint.

A modification succeeds if one of the decoded molecules satisfies the constraint and is distinct from the starting one.

We show the results in TAB1 .

In the task of optimizing penalized logP of drug-like molecules, our method significantly outperforms the previous results in the mean improvement of the property.

It achieves a comparable mean similarity in the constrained scenario (for δ ą 0).

The success rates are comparable for δ " 0, 0.2, whereas for the more stringent constraints (δ " 0.4, 0.6) our model has lower success rates.

Note that comparably high improvements of penalized logP can be obtained using reinforcement learning BID35 .

However, the resulting optimized molecules are not druglike, e.g., they have a very low quantitative estimate of drug-likeness scores (Bickerton et al., Figure 5 : Molecules with the highest improvement of the penalized logP for δ ě 0.6.

In the top row we show the starting molecules, whereas in the bottom row we show the generated molecules.

Upper row numbers indicate Tanimoto similarities between the starting and the generated molecule.

Improvement in the score is given in at the bottom.2012) even in the constrained optimization scenario.

In our method (as well as in JT-VAE) druglikeness is achieved 'by construction' and is a feature of the latent space obtained by training the variational autoencoder on molecules from ZINC (which are druglike).

Our architecture is tailor made for the scenario of constrained molecule optimization.

However, as an additional task, we check what happens when we iteratively use the generator on the molecules being optimized, which leads to diminishing of similarity between the starting molecules and those in consecutive iterations.

For the present task the set X needs to be a sample from the entire ZINC-250K, whereas the set Y is chosen as a sample from the top-20% of molecules with the highest value of penalized logP. Each molecule is fed into the generator and the corresponding 'optimized' molecule is obtained.

The generated molecule is then treated as the new input for the generator.

The process is repeated K times and the resulting set of molecules is tGpxq, GpGpxqq, ... }.

Here, as in the previous task and as in BID18 we start the procedure from the 800 molecules with the lowest values of penalized logP in ZINC-250K.The results of our unconstrained molecule optimization are shown in FIG4 .

In FIG4 and (c) we observe that consecutive iterations keep shifting the distribution of the objective (penalized logP) towards higher values.

However, the improvement from further iterations is decreasing.

Interestingly, the maximum of the distribution keeps increasing (although in somewhat random fashion).

After 10-20 iterations it reaches the high values observed from molecules which are not druglike in BID35 (obtained with RL).

In our case the molecules with the highest penalized logP after many iterations also become non-druglike -see Appendix D for a list of compounds with the max- imum values of penalized logP in our iterative optimization procedure.

This lack of drug-likeness is related to the fact that after performing many iterations, the distribution of coordinates of our set of molecules in the latent space goes far away from the prior distribution (multivariate normal) used when training the JT-VAE on ZINC-250K.

In FIG4 (b) we show the evolution of the distribution of Tanimoto similarities between the starting molecules and those obtained after K " 1, 2, 5, 10 iterations.

We also show the similarity between the starting molecules and random molecules from ZINC-250K.

We observe that after 10 iterations the similarity between the starting molecules and the optimized ones is comparable to the similarity to random molecules from ZINC-250K.

After around 20 iterations the optimized molecules become less similar to the starting ones than random molecules from ZINC-250K.

In this work, we introduced Mol-CycleGAN -a new model based on CycleGAN that can be used for the de novo generation of molecules.

The advantage of the proposed model is the ability to learn transformation rules from the sets of compounds with desired and undesired values of the considered property.

The model operates in the latent space trained by another model -in our work we use the latent space of JT-VAE.

The model can generate molecules with desired properties -both structural and physicochemical.

The generated molecules are close to the starting ones and the degree of similarity can be controlled via a hyperparameter.

In the task of constrained optimization of druglike molecules our model significantly outperforms previous results.

In future work we will extend the approach to multi-parameter optimization of molecules using StarGAN BID5 .

It would also be interesting to test the model on cases where a small structural change leads to a drastic change in the property (e.g. on the so-called activity cliffs), which are hard for other approaches.

Another interesting direction is the application of the model to working on text embeddings, where the X and Y sets could be characterized, e.g., by different sentiment.

All networks are trained using the Adam optimizer BID19 with learning rate 0.0001.

During training we use batch normalization BID16 .

As the activation function we use leaky-ReLU with α " 0.1.

In experiments from sections 4.1, 4.2 the models are trained for 100 epochs and in experiments from 4.3 for 300 epochs.

A.1 FOR EXPERIMENTS IN SECTIONS 4.1, 4.2• Generators are built of one fully connected residual layer, followed by one dense layer.

All layers contain 56 units.• Discriminators are build of 6 dense layers of the following sizes: 56, 42, 28, 14, 7, 1 units.

• Generators are built of four fully connected residual layers.

All layers contain 56 units.• In the experiment on halogen moieties, the set X always (i.e., both in train-and test-time) contains molecules without halogen moieties, and the set Y always contains molecules with halogen moieties.

In the dataset used to construct the latent space (ZINC-250K) 64.9 % molecules do not contain any halogen moiety, whereas the remaining 35.1 % contain one or more halogen moieties.

In the experiment on aromatic rings, the set X always (i.e., both in train-and test-time) contains molecules with 2 rings, and the set Y always contains molecules with 1 or 3 rings.

The distribution of the number of aromatic rings in the dataset used to construct the latent space (ZINC-250K) is shown in FIG5 along with the distribution for X and Y .

For the molecule optimization tasks we plot the distribution of the property being optimized (penalized logP) in Figs. 8 (constrained optimization) and 9 (unconstrained optimization).Figure 8: Distribution of penalized logP in ZINC-250K and in the sets used in the task of constrained molecule optimization (Section 4.2).

Note that the sets X train and Y train are non-overlapping (they are a random sample from ZINC-250K split by the median).

X test is the set of 800 molecules from ZINC-250K with the lowest values of penalized logP.

@highlight

We introduce Mol-CycleGAN - a new generative model for optimization of molecules to augment drug design.

@highlight

The paper presents an approach for optimizing molecular properties based on the application of CycleGANs to variational autoencoders for molecules and employs a domain-specific VAE called Junction Tree VAE (JT-VAE).

@highlight

This paper uses a variational autoencoders to learn a translation function, from the set of molecules without the interested property to the set of molecules with the property. 