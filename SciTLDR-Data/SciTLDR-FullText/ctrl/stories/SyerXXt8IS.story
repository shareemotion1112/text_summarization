We seek to auto-generate stronger  input features  for ML methods faced with limited training data.

Biological neural nets (BNNs) excel at fast learning, implying that they extract highly informative features.

In particular, the insect olfactory network  learns new odors very rapidly, by means of three key elements: A competitive inhibition layer; randomized, sparse connectivity into a high-dimensional sparse plastic layer; and Hebbian updates of synaptic weights.

In this work we deploy MothNet, a computational model of the moth olfactory network, as an automatic feature generator.

Attached as a front-end pre-processor, MothNet's readout neurons provide new features, derived from the original features, for use by standard ML classifiers.

These ``insect cyborgs'' (part BNN and part ML method) have significantly better performance than baseline ML methods alone on vectorized MNIST and Omniglot data sets, reducing test set error averages 20% to 55%.

The MothNet feature generator also substantially out-performs other feature generating methods including PCA, PLS, and NNs.

These results highlight the potential value of BNN-inspired feature generators in the ML context.

Machine learning (ML) methods, especially neural nets (NNs) with backprop, often require large amounts of training data to attain their high performance.

This creates bottlenecks to deployment, and constrains the types of problems that can be addressed [1] .

The limited-data constraint is common for ML targets that use medical, scientific, or field-collected data, as well as AI efforts focused on rapid learning.

We seek to improve ML methods' ability to learn from limited data by means of an architecure that automatically generates, from existing features, a new set of class-separating features.

Biological neural nets (BNNs) are able to learn rapidly, even from just a few samples.

Assuming that rapid learning requires effective ways to separate classes, we may look to BNNs for effective feature-generators [2] .

One of the simplest BNNs that can learn is the insect olfactory network [3] , containing the Antennal Lobe (AL) [4] and Mushroom Body(MB) [5] , which can learn a new odor given just a few exposures.

This simple but effective feedforward network contains three key elements that are ubiquitous in BNN designs: Competitive inhibition [6] , high-dimensional sparse layers [7; 8] , and a Hebbian update mechanism [9] .

Synaptic connections are largely random [10] .

MothNet is a computational model of the M. sexta moth AL-MB that demonstrated rapid learning of vectorized MNIST digits, with performance superior to standard ML methods given N ≤ 10 training samples per class [11] .

The MothNet model includes three key elements, as follows.

(i) Competitive inhibition in the AL: Each neural unit in the AL receives input from one feature, and outputs not only a feedforward excitatory signal to the MB, but also an inhibitory signal to other neural units in the AL that tries to dampen other features' presence in the sample's output AL signature. (ii) Sparsity in the MB, of two types: The projections from the AL to the MB are non-dense (≈ 15% non-zero), and the MB neurons fire sparsely in the sense that only the strongest 5% to 15% of the total population are allowed to fire (through a mechanism of global inhibition). (iii) Weight updates affect only MB→Readout connections (AL connections are not plastic).

Hebbian updates occur as: ∆w ij = αf i f j if f i f j > 0 (growth), and ∆w ij = −δw ij if f i f j = 0 (decay), where f i , f j are two neural firing rates (f i ∈ MB, f j ∈ Readouts) with connection weight w ij .

In this work we tested whether the MothNet architecture can usefully serve as a front-end feature generator for an ML classifier (our thanks to Blake Richards for this suggestion).

We combined MothNet with a downstream ML module, so that the Readouts of the trained AL-MB model were fed into the ML module as additional features.

From the ML perspective, the AL-MB acted as an automatic feature generator; from the biological perspective, the ML module stood in for the downstream processing in more complex BNNs.

Our Test Case was a non-spatial, 85-feature, 10-class task derived from the downsampled, vectorized MNIST data set (hereafter "vMNIST").

On this non-spatial dataset, CNNs or other spatial methods were not applicable.

The trained Mothnet Readouts, used as features, significantly improved the accuracies of ML methods (NN, SVM, and Nearest Neighbors) on the test set in almost every case.

That is, the original input features (pixels) contained class-relevant information unavailable to the ML methods alone, but which the AL-MB network encoded in a form that enabled the ML methods to access it.

MothNet-generated features also significantly out-performed features generated by PCA (Principal Components Analysis), PLS (Partial Least Squares), NNs, and transfer learning (weight pretraining) in terms of their ability to improve ML accuracy.

These results indicate that the insect-derived network generated significantly stronger features than these other methods.

To generate vMNIST, we downsampled, preprocessed, and vectorized the MNIST data set to give samples with 85 pixels-as-features.

vMNIST has the advantage that our baseline ML methods (Nearest Neighbors, SVM, and Neural Net) do not attain full accuracy at low N. Trained accuracy of baseline ML methods was controlled by restricting training data.

Full network architecture details of the AL-MB model (MothNet) are given in [11] .

Full Matlab code for these cyborg experiments including comparison methods, all details re ML methods and hyperparameters, and code for MothNet simulations, can be found at [12] .

MothNet instances were generated randomly from templates that specified connectivity parameters.

We ran two sets of experiments:

Cyborg vs baseline ML methods on vMNIST Experiments were structured as follows:

1.

A random set of N training samples per class was drawn from vMNIST.

2.

The ML methods trained on these samples, to provide a baseline.

3.

MothNet was trained on these same samples, using time-evolved stochastic differential equation simulations and Hebbian updates, as in [11] .

4.

The ML methods were then retrained from scratch, with the Readout Neuron outputs from the trained MothNet instance fed in as additional features.

These were the "insect cyborgs", i.e. an AL-MB feature generator joined to a ML classifier.

5.

Trained ML accuracies of the baselines and cyborgs were compared to assess gains.

To compare the effectiveness of MothNet features vs features generated by conventional ML methods, we ran vMNIST experiments structured as as above, but with MothNet replaced by one of the following feature generators:

1.

PCA applied to the vMNIST training samples.

The new features were the projections onto each of the top 10 modes.

2.

PLS applied to the vMNIST training samples.

The new features were the projections onto each of the top 10 modes.

Since PLS incorporates class information, we expected it to out-perform PCA.

3.

NN pre-trained on the vMNIST training samples.

The new features were the (logs of the) 10 output units.

This feature generator was used as a front end to SVM and Nearest Neighbors only.

Since vMNIST has no spatial content, CNNs were not used.

4. NN with weights initialized by training on an 85-feature vectorized Omniglot data set [13] , then trained on the vMNIST data as usual (transfer learning, applied to the NN baseline only).

Omniglot is an MNIST-like thumbnail collection of 1623 characters with 20 samples each.

(5.)

For the baseline NN method, we used one hidden layer.

Including two hidden layers did not improve baseline performance.

This was an implicit control, showing that MothNet features were not equivalent to just adding an extra layer to a NN.

MothNet readouts as features significantly improved accuracy of ML methods, demonstrating that the MothNet architecture effectively captured new class-relevant features.

We also tested a non-spatial, 10-class task derived from the Omniglot data set and found similar gains.

MothNet-generated features were also far more effective than the comparison feature generators (PCA, PLS, and NN).

Gains due to MothNet features on vMNIST ML baseline test set accuracies ranged from 10% to 88%, depending on method and on N (we stopped our sweep at N = 100).

This baseline accuracy is marked by the lower colored circles in Fig 1.

Cyborg test set accuracy is marked by the upper colored circles in Fig 1, and the raw gains in accuracy due to MothNet features are marked by thick vertical bars.

MothNet features increased raw accuracy across all ML models.

Relative reduction in test set error, as a percentage of baseline error, was 20% to 55%, with high baseline accuracies seeing the most benefit (Fig 2) .

NN models saw the greatest benefits, with 40% to 55% relative reduction in test error.

Remarkably, a MothNet front-end improved ML accuracy even in cases where the ML baseline already exceeded the ≈ 75% accuracy ceiling of MothNet (e.g. NNs at N = 15 to 100 samples per class): the MothNet readouts contained clustering information which ML methods leveraged more effectively than MothNet itself.

Gains were significant in almost all cases with N > 3.

Table 1 gives p-values of the gains due to MothNet.

Table 1 ).

We ran the cyborg framework on vMNIST using PCA (projections onto top 10 modes), PLS (projection onto top 10 modes), and NN (logs of the 10 output units) as feature generators.

Each feature generator was trained (e.g. PCA projections were defined) using the training samples.

Table 2 gives the relative increase in mean accuracy due to the various feature generators (or to pre-training) for NN models (13 runs per data point).

Results for Nearest Neighbors and SVM were similar.

MothNet features were far more effective than these other methods.

Effect of pass-through AL The MothNet architecture has two main layers: a competitive inhibition layer (AL) and a highdimensional, sparse layer (MB).

To test the effectiveness the MB alone, we ran the vMNIST experiments, but using a pass-through (identity) AL layer for MothNet.

Cyborgs with a pass-through AL still posted significant improvements in accuracy over baseline ML methods.

The gains of cyborgs with pass-through ALs were generally between 60% and 100% of the gains posted by cyborgs with normal ALs (see Table 3 ), suggesting that the high-dimensional, trainable layer (the MB) was most important.

However, the competitive inhibition of the AL layer clearly added value in terms of generating strong features, up to 40% of the total gain.

NNs benefitted most from the AL layer.

We deployed an automated feature generator based on a very simple BNN, containing three key elements rare in engineered NNs but endemic in BNNs of all complexity levels: (i) competitive inhibition; (ii) sparse projection into a high-dimensional sparse layer; and (iii) Hebbian weight updates for training.

This bio-mimetic feature generator significantly improved the learning abilities of standard ML methods on both vMNIST and vOmniglot.

Class-relevant information in the raw feature distributions, not extracted by the ML methods alone, was evidently made accessible by MothNet's pre-processing.

In addition, MothNet features were consistently much more useful than features generated by standard methods such as PCA, PLS, NNs, and pre-training.

The competitive inhibition layer may enhance classification by creating several attractor basins for inputs, each focused on the features that present most strongly for a given class.

This may push otherwise similar samples (of different classes) away from each other, towards their respective class attractors, increasing the effective distance between the samples.

The sparse connectivity from AL to MB has been analysed as an additive function, which has computational and anti-noise benefits [14] .

The insect MB brings to mind sparse autoencoders (SAs) e.g. [15] .

However, there are several differences: MBs do not seek to match the identity function; the sparse layers of SAs have fewer active neurons than the input dimension, while in the MB the number of active neurons is much greater than the input dimension; MBs have no pre-training step; and the MB needs very few samples to bake in structure that improves classification.

The MB differs from Reservoir Networks [16] in that MB neurons have no recurrent connections.

Finally, the Hebbian update mechanism appears to be quite distinct from backprop.

It has no objective function or output-based loss that is pushed back through the network, and Hebbian weight updates, either growth or decay, occur on a local "use it or lose it" basis.

We suspect that the dissimilarity of the optimizers (MothNet vs ML) was an asset in terms of increasing total encoded information.

<|TLDR|>

@highlight

Features auto-generated by the bio-mimetic MothNet model significantly improve the test accuracy of standard ML methods on vectorized MNIST. The MothNet-generated features also outperform standard feature generators.