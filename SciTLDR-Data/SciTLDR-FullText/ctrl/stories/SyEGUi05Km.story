Our main motivation is to propose an efficient approach to generate novel multi-element stable chemical compounds that can be used in real world applications.

This task can be formulated as a combinatorial problem, and it takes many hours of human experts to construct, and to evaluate new data.

Unsupervised learning methods such as Generative Adversarial Networks (GANs) can be efficiently used to produce new data.

Cross-domain Generative Adversarial Networks were reported to achieve exciting results in image processing applications.

However, in the domain of materials science, there is a need to synthesize data with higher order complexity compared to observed samples, and the state-of-the-art cross-domain GANs can not be adapted directly.



In this contribution, we propose a novel GAN called CrystalGAN which generates new chemically stable crystallographic structures with increased domain complexity.

We introduce an original architecture, we provide the corresponding loss functions, and we show that the CrystalGAN generates very reasonable data.

We illustrate the efficiency of the proposed method on a real original problem of novel hydrides discovery that can be further used in development of hydrogen storage materials.

In modern society, a big variety of inorganic compositions are used for hydrogen storage owing to its favorable cost BID4 .

A vast number of organic molecules are applied in solar cells, organic light-emitting diodes, conductors, and sensors BID25 .

Synthesis of new organic and inorganic compounds is a challenge in physics, chemistry and in materials science.

Design of new structures aims to find the best solution in a big chemical space, and it is in fact a combinatorial optimization problem.

In this work, we focus on applications of hydrogen storage, and in particular, we challenge the problem to investigate novel chemical compositions with stable crystals.

Traditionally, density functional theory (DFT) plays a central role in prediction of chemically relevant compositions with stable crystals BID22 .

However, the DFT calculations are computationally expensive, and it is not acceptable to apply it to test all possible randomly generated structures.

A number of machine learning approaches were proposed to facilitate the search for novel stable compositions BID3 .

There was an attempt to find new compositions using an inorganic crystal structure database, and to estimate the probabilities of new candidates based on compositional similarities.

These methods to generate relevant chemical compositions are based on recommender systems BID10 .

The output of the recommender systems applied in the crystallographic field is a rating or preference for a structure.

A recent approach based on a combination of machine learning methods and the high-throughput DFT calculations allowed to explore ternary chemical compounds BID21 , and it was shown that statistical methods can be of a big help to identify stable structures, and that they do it much faster than standard methods.

Recently, support vector machines were tested to predict crystal structures BID16 showing that the method can reliably predict the crystal structure given its composition.

It is worth mentioning that data representation of observations to be passed to a learner, is critical, and data representations which are the most suitable for learning algorithms, are not necessarily scientifically intuitive BID23 .Deep learning methods were reported to learn rich hierarchical models over all kind of data, and the GANs BID8 ) is a state-of-the-art model to synthesize data.

Moreover, deep networks were reported to learn transferable representations BID18 .

The GANs were already exploited with success in cross-domain learning applications for image processing BID13 BID12 .Our goal is to develop a competitive approach to identify stable ternary chemical compounds, i.e., compounds containing three different elements, from observations of binary compounds.

Nowadays, there does not exist any approach that can be applied directly to such an important task of materials science.

The state-of-the-art GANs are limited in the sense that they do not generate samples in domains with increased complexity, e.g., the application where we aim to construct crystals with three elements from observations containing two chemical elements only.

An attempt to learn many-to-many mappings was recently introduced by BID0 , however, this promising approach does not allow to generate data of a higher-order dimension.

Our contribution is multi-fold:• To our knowledge, we are the first to introduce a GAN to solve the scientific problem of discovery of novel crystal structures, and we introduce an original methodology to generate new stable chemical compositions; • The proposed method is called CrystalGAN, and it consists of two cross-domain GAN blocks with constraints integrating prior knowledge including a feature transfer step; • The proposed model generates data with increased complexity with respect to observed samples; • We demonstrate by numerical experiments on a real challenge of chemistry and materials science that our approach is competitive compared to existing methods; • The proposed algorithm is efficiently implemented in Python, and it will be publicly available shortly, as soon as the contribution is de-anonymized.

This paper is organized as follows.

We discuss the related work in Section 2.

In Section 3, we provide the formalisation of the problem, and introduce the CrystalGAN.

The results of our numerical experiments are discussed in Section 4.

Concluding remarks and perspectives close the paper.

Our contribution is closely related to the problems of unsupervised learning and cross-domain learning, since our aim is to synthesize novel data, and the new samples are supposed to belong to an unobserved domain with an augmented complexity.

In the adversarial nets framework, the deep generative models compete with an adversary which is a discriminative model learning to identify whether an observation comes from the model distribution or from the data distribution BID7 .

A classical GAN consists of two models, a generator G whose objective is to synthesize data and a discriminator D whose aim is to distinguish between real and generated data.

The generator and the discriminator are trained simultaneously, and the training problem is formulated as a two-player minimax game.

A number of techniques to improve training of GANs were proposed by ; BID9 ; BID19 .Learning cross domain relations is an active research direction in image processing.

Several recent papers BID13 BID0 discuss an idea to capture some particular characteristics of one image and to translate them into another image.

A conditional GAN for image-to-image translation is considered by .

An advantage of the conditional model is that it allows to integrate underlying structure into the model.

The conditional GANs were also used for multi-model tasks BID15 ).

An idea to combine observed data to produce new data was proposed in BID26 , e.g., an artist can mix existing pieces of music to create a new one.

First domain, H is hydrogen, and A is a metal BH:Second domain, H is hydrogen, and B is another metal DISPLAYFORM0 Generator function that translates input features xAH from (domain) AH to BH GBHA 1 :Generator function that translates input features xBH from (domain) BH to AH DAH and DBH:Discriminator functions of AH domain and BH domain, respectively AHB1:xAHB 1 is a sample generated by generator function GAHB 1 BHA1:yBHA 1 is a sample produced by generator function GBHA 1 AHBA1 and BHAB1 : Data reconstructed after two generator translations AHBg and BHAg:Data obtained after feature transfer step from domain AH to domain BH, and from domain BH to domain AH, respectively Input data for the second step of CrystalGAN GAHB 2 :Generator function that translates xAHB g Features generated in the first step from AHBg to AHB2 GBHA 2 :Generator function that translates yBHA g Data generated in first step from BHAg to BHA2 DAHB and DBHA:The discriminator functions of domain AHBg and domain BHAg, respectively AHB2:xAHB 2 is a sample generated by the generator function GAHB 2 BHA2:yBHA 2 is a sample produced by the generator function GBHA 2 AHBA2 and BHAB2:Data reconstructed as a result of two generators translations AHB2 and BHA2:Final new data (to be explored by human experts) An approach to learn high-level semantic features, and to train a model for more than a single task, was introduced by BID18 .

In particular, it was proposed to train a model to jointly learn several complementary tasks.

This method is expected to overcome the problem of overfitting to a single task.

An idea to introduce multiple discriminators whose role varies from formidable adversary to forgiving teacher was discussed by BID5 .Several GANs were adapted to some materials science and chemical applications.

So, ObjectiveReinforced GANs that perform molecular generation of carbon-chain sequence taking into consideration some desired properties, were introduced in BID20 , and the method was shown to be efficient for drug discovery.

Another avenue is to integrate rule-based knowledge, e.g., molecular descriptors with the deep learning.

ChemNet BID6 ) is a deep neural network pre-trained with chemistry-relevant representations obtained from prior knowledge.

The model can be used to predict new chemical properties.

However, as we have already mentioned before, none of these methods generates crystal data of augmented complexity.

In this section, we introduce our approach.

The CrystalGAN consists of three procedures:1.

First step GAN which is closely related to the cross-domain GANs, and that generates pseudo-binary samples where the domains are mixed.

2.

Feature transfer procedure constructs higher order complexity data from the samples generated at the previous step, and where components from all domains are well-separated.

3.

Second step GAN synthesizes, under geometric constraints, novel ternary stable chemical structures.

First, we describe a cross-domain GAN, and then, we provide all the details on the proposed CrystalGAN.

We provide all notations used by the CrystalGAN in TAB0 .

The GANs architectures for the first and the second steps are shown on FIG0 .

We now propose a novel architecture based on the cross-domain GAN algorithms with constraint learning to discover higher order complexity crystallographic systems.

We introduce a GAN model to find relations between different crystallographic domains, and to generate new materials.

To make the paper easier to follow, without loss of generality, we will present our method providing a specific example of generating ternary hydride compounds of the form "A (a metal) -H (hydrogen) -B (a metal)".Our observations are stable binary compounds containing chemical elements A+H which is a composition of some metal A and the hydrogen H, and B+H which is a mixture of another metal B with the hydrogen.

So, a machine learning algorithm has access to observations {(x AHi )} DISPLAYFORM0 Our goal is to generate novel ternary, i.e. more complex, stable data x AHB (or y BHA ) based on the properties learned from the observed binary structures.

Below we describe the architecture of the CrystalGAN.

Our approach consists of two consecutive steps with a feature transfer procedure inbetween.

The first step of CrystalGAN generates new data with increased complexity.

The adversarial network takes DISPLAYFORM0 , and synthesizes DISPLAYFORM1 and FIG0 summarizes the first step of CrystalGAN.

DISPLAYFORM2 The reconstruction loss functions take the following form: DISPLAYFORM3 Ideally, L R AH = 0, L R BH = 0, and x AHBA1 = x AH , y BHAB1 = y BH , and we minimize the distances d(x AHBA1 , x AH ) and d(y BHAB1 , y BH ).The generative adversarial loss functions of the first step of CrystalGAN aim to control that the original observations are reconstructed as accurate as possible: DISPLAYFORM4 DISPLAYFORM5 The generative loss functions contain the two terms defined above: DISPLAYFORM6 The discriminative loss functions aim to discriminate the samples coming from AH and BH: DISPLAYFORM7 DISPLAYFORM8 Now, we have all elements to define the full generative loss function of the first step: DISPLAYFORM9 where λ 1 , λ 2 , λ 3 , and λ 4 are real-valued hyper-parameters that control the ratio between the corresponding terms, and the hyper-parameters are to be fixed by cross-validation.

The full discriminator loss function of this step L D1 is defined as follows: DISPLAYFORM10

The first step generates pseudo-binary samples M H, where M is a new discovered domain merging A and B properties.

Although these results can be interesting for human experts, the samples generated by the first step are not easy to interpret, since the domains A and B are completely mixed in these samples, and there is no way to deduce characteristics of two separate elements coming from these domains.

So, we need a second step which will generate data of a higher order complexity from two given domains.

We transfer the attributes of A and B elements, this procedure is also shown on FIG0 , in order to construct a new dataset that will be used as a training set in the second step of the CrystalGAN.In order to prepare the datasets to generate higher order complexity samples, we add a placeholder.

(E.g., for domain AH, the fourth matrix is empty, and for domain BH, the third matrix is empty.)

The second step GAN takes as input the data generated by the first step GAN and modified by the feature transfer procedure.

The results of the second step are samples which describe ternary chemical compounds that are supposed to be stable from chemical viewpoint.

The geometric constraints control the quality of generated data.

A crystallographic structure is fully described by a local distribution.

This distribution is determined by distances to all nearest neighbors of each atom in a given crystallographic structure.

We enforce the second step GAN with the following geometric constraints which satisfy the geometric conditions of our scientific domain application.

The implemented constraints are also shown on FIG0 .

DISPLAYFORM0 be the set of distances of the first neighbors of all atoms in a crystallographic structure.

There are two geometric constraints to be considered while generating new data.

The first geometric (geo) constraint is defined as follows: DISPLAYFORM1 where d 1 is the minimal distance between two first nearest neighbors in a given crystallographic structure.

The second geometric constraint takes the following form: DISPLAYFORM2 where d 2 is the maximal distance between two first nearest neighbors.

The loss function of the second step GAN is augmented by the following geometric constraints: DISPLAYFORM3 Given x AHBg and y BHAg from the previous step, we generate: DISPLAYFORM4 The reconstruction loss functions are given: DISPLAYFORM5 DISPLAYFORM6 The generative adversarial loss functions are given by: DISPLAYFORM7 DISPLAYFORM8 The generative loss functions of the this step are defined as follows: DISPLAYFORM9 DISPLAYFORM10 The losses of the discriminator of the second step can be defined: DISPLAYFORM11 DISPLAYFORM12 Now, we have all elements to define the full generative loss function: DISPLAYFORM13 where λ 1 , λ 2 , λ 3 , λ 4 , λ 5 , and λ 6 are real-valued hyper-parameters that control the influence of the terms.

The full discriminative loss function of the second step L D2 takes the form: DISPLAYFORM14 To summarise, in the second step, we use the dataset issued from the feature transfer as an input containing two domains x AHBg and y BHAg .

We train the cross-domain GAN taking into consideration constraints of the crystallographic environment.

We integrated geometric constraints proposed by crystallographic and materials science experts to satisfy environmental constraints, and to increase the rate of synthesized stable ternary compounds.

The second step is drafted on FIG0 .

Crystallographic structures can be represented using the POSCAR files which are input files for the DFT calculations under the VASP code BID14 .

These are coordinate files, they contain the lattice geometry and the atomic positions, as well as the number (or the composition) and the nature of atoms in the crystal unit cell.

We use a dataset constructed from BID2 Villars & Cenzual, 2017) by experts in materials science.

Our training data set contains the POSCAR files, and the proposed CrystalGAN generates also POSCAR files.

Such a file contains three matrices: the first one is abc matrix, corresponding to the three lattice vectors defining the unit cell of the system, the second matrix contains atomic positions of H atom, and the third matrix contains coordinates of metallic atom A (or B).The information from the files is fed into 4-dimensional tensors.

An example of a POSCAR file, and its corresponding representation for the GANs is shown on FIG1 .

On the same figure on the right we show the corresponding structure in 3D.

Note that we increase the data complexity by the feature transfer procedure by adding placeholders.

Our training dataset includes 1,416 POSCAR files of binary hydrides divided into 63 classes where each class is represented as a 4-dimensional tensor.

Each class of binary M H hydride contains two elements: the hydrogen H and another element M from the periodic table.

In our experiments, after discussions with materials science researchers, we focused on exploration of ternary compositions "Palladium -Hydrogen -Nickel" from the binary systems observations of "Palladium -Hydrogen" and "Nickel -Hydrogen".

So, AH = PdH, and BH = NiH. We also considered another task to generate ternary compounds "Magnesium -Hydrogen -Titanium".In the CrystalGAN, we need to compute all the distances of the nearest neighbors for each generated POSCAR file.

The distances between hydrogen atoms H in a given crystallographic structure should respect some geometric rules, as well as the distances between the atoms A−B, A−A', and B −B .

We applied the geometric constraints on the distances between the neighbors (for each atom in a crystallographic structure) introduced in the previous section.

Note that the distances A−H and B−H are not penalized by the constraints.

In order to compute the distances between all nearest neighbors in the generated data, we used the pythonic library Pymatgen BID17 specifically developed for material analysis.

For all experiments in this paper, the distances are fixed by our colleagues in crystallographic and materials science to d 1 = 1.8Å (angstrom, 10 −10 meter) and d 2 = 3Å.

We set all the hyperparameters by cross validation, however, we found that a reasonable performance is reached when all λ i have similar values, and are quite close to 1.

We use the standard AdamOptimizer with learning rate α = 0.0001, and β 1 = 0.5.

The number of epochs is set to 1000 (we verified that the functions converge).

The mini-batch size equals 35.

DISPLAYFORM0 without constraints with constraints Pd -Ni -H 0 0 4 9 Mg -Ti -H 0 0 2 8 Table 2 : Number of ternary compositions of good quality generated by the tested methods.

Each block of the CrystalGAN architecture (the generators and the discriminators) is a multi-layer neural network with 5 hidden layers.

Each layer contains 100 units.

We use the rectified linear unit (ReLU) as an activation function of the neural network.

All these parameters were fixed by cross-validation (for both chosen domains "Palladium -Hydrogen" and "Nickel -Hydrogen").Our code is implemented in Python (TensorFlow).

We run the experiments using GPU with graphics card NVIDIA Quadro M5000.

In our numerical experiments, we compare the proposed CrystalGAN with a classical GAN, the DiscoGAN BID13 , and the CrystalGAN but without the geometric constraints.

All these GANs generate POSCAR files, and we evaluate the performance of the models by the number of generated ternary structures which satisfy the geometric crystallographic environment.

Table 2 shows the number of successes for the considered methods.

The classical GAN which takes Gaussian noise as an input, does not generate acceptable chemical structures.

The DiscoGAN approach performs quite well if we use it to generate novel pseudo-binary structures, however, it is not adapted to synthesize ternary compositions.

We observed that the CrystalGAN (with the geometric constraints) outperforms all tested methods.

From multiple discussions with experts in materials science and chemistry, first, we know that the number of novel stable compounds can not be very high, and it is already considered as a success if we synthesize several stable structures which satisfy the constraints.

Hence, we can not really reason in terms of accuracy or error rate which are widely used metrics in machine learning and data mining.

Second, evaluation of a stable structure is not straightforward.

Given a new composition, only the result of density functional theory (DFT) calculations can provide a conclusion whether this composition is stable enough, and whether it can be used in practice.

However, the DFT calculations are computationally too expensive, and it is out of question to run them on all data we generated using the CrystalGAN.

It is planned to run the DFT calculations on some pre-selected generated ternary compositions to take a final decision on practical utility of the chemical compounds.

Our goal was to develop a principled approach to generate new ternary stable crystallographic structures from observed binary, i.e. containing two chemical elements only.

We propose a learning method called CrystalGAN to discover cross-domain relations in real data, and to generate novel structures.

The proposed approach can efficiently integrate, in form of constraints, prior knowledge provided by human experts.

CrystalGAN is the first GAN developed to generate scientific data in the field of materials science.

To our knowledge, it is also the first approach which generates data of a higher-order complexity, i.e., ternary structures where the domains are well-separated from observed binary compounds.

The CrystalGAN was, in particular, successfully tested to tackle the challenge to discover new materials for hydrogen storage.

Currently, we investigate different GANs architectures, also including elements of reinforcement learning, to produce data even of a higher complexity, e.g., compounds containing four or five chemical elements.

Note that although the CrystalGAN was developed and tested for applications in materials science, it is a general method where the constraints can be easily adapted to any scientific problem.

<|TLDR|>

@highlight

"Generating new chemical materials using novel cross-domain GANs."