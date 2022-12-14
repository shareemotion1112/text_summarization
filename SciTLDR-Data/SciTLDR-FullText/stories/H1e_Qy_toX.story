Deep learning has demonstrated abilities to learn complex structures, but they can be restricted by available data.

Recently, Consensus Networks (CNs) were proposed to alleviate data sparsity by utilizing features from multiple modalities, but they too have been limited by the size of labeled data.

In this paper, we extend CN to Transductive Consensus Networks (TCNs), suitable for semi-supervised learning.

In TCNs, different modalities of input are compressed into latent representations, which we encourage to become indistinguishable during iterative adversarial training.

To understand TCNs two mechanisms, consensus and classification, we put forward its three variants in ablation studies on these mechanisms.

To further investigate TCN models, we treat the latent representations as probability distributions and measure their similarities as the negative relative Jensen-Shannon divergences.

We show that a consensus state beneficial for classification desires a stable but imperfect similarity between the representations.

Overall, TCNs outperform or align with the best benchmark algorithms given 20 to 200 labeled samples on the Bank Marketing and the DementiaBank datasets.

Deep learning has demonstrated impressive capacities to learn complicated structures from massive data sets.

However, acquiring sufficient labeled data can be expensive or difficult (e.g., for specific pathological populations BID10 ).

Transductive learning (a set of semi-supervised algorithms) uses intrinsic structures among unlabeled data to boost classifier performance.

In the real world, data can spread across multiple modalities (e.g., visual, acoustic, and text) in typical tasks, although many existing transductive algorithms do not exploit the structure across these modalities.

Co-training [3] and tri-training BID23 use one classifier per modality to supervise each other, but they can only apply to two and three modalities respectively.

Recently, Consensus Networks (CNs) BID24 incorporated the idea of co-training.

Not limited by the number of modalities, CNs showed promising results on detecting cognitive impairments from multi-modal datasets of speech.

A consensus network contains several interpreters (one per modality), a discriminator, and a classifier.

The interpreters try to produce low-dimensional representations of input data that are indistinguishable by the discriminator.

The classifier makes predictions based on these representation vectors.

Despite promising results, CN is limited by the amount of available training data.

This motivates our extension into semi-supervised learning with our Transductive Consensus Network (TCN).TCNs operate in two mechanisms: as consensus or classifier.

The consensus mechanism urges the modality representations to resemble each other (trained on the whole dataset without using labels), and the classifier mechanism optimizes the networks to retain information useful for classification (trained on the labeled dataset).

To illustrate the importance of these two mechanisms in an ablation study, we also put forward its three variants: TCN-embed, TCN-svm, and TCN-AE in ??3.

By this ablation study, we show that both mechanisms should function together via iterative training.

To further reveal the mechanisms of TCN, we formulate in ??3.5 the similarity between latent representations using negative Jensen-Shannon divergences.

By monitoring their similarities, we show that a meaningful consensus state prefers representations to have suboptimal similarities.

In experiments ( ??4), we compare TCN to its three variants, TCN's multimodal supervised learning counterpart (CN), and several other semi-supervised learning benchmark algorithms on two datasets: Bank Marketing (from the UCI repository) and DementiaBank (a dataset of pathological speech in multiple modalities).

On both datasets, the F-scores of TCN align with the best benchmark models when there are more labeled data available, and outperform benchmarks (including tri-training) given as few as 20 labeled points.

Transductive SVMs BID8 were an early attempt in transductive semi-supervised learning.

In addition to the SVM objective, TSVMs minimize the hinge loss on unlabeled data.

TSVMs have yielded good performance on our datasets, so we include them for completeness.

Later, many semi-supervised learning algorithms took either autoencoding or GAN approaches.

In autoencoding, a model learns a low-dimensional representation and a reconstruction for each data sample.

Usually, noise is added in generating the low-dimensional representation.

By trying to minimize the difference between reconstructed and original data, the model learns (i) an encoder capturing low-dimensional hidden information and (ii) a decoder, which is a generative model able to recover data.

This is the approach of the denoising autoencoder BID22 BID11 .

An extension is Ladder network BID19 , which stacks denoising autoencoders and adds layer-wise reconstruction losses.

Ladder networks are often more computationally efficient than stacked denoise autoencoders.

In generative adversarial networks (GANs) BID6 , a generator tries to produce data that are indistinguishable from true data, given a discriminator which itself learns to tell them apart.

This adversarial training procedure could proceed with few labeled data points.

For example, Feature-matching GANs BID20 add generated ("synthetic") samples into the training data of the discriminator as an additional class.

Another example is Categorical GANs BID21 which optimize uncertainty (measured by entropy of predictions) in the absence of labels.

Noticeably, BID3 showed that a discriminator performing well on a training set might not benefit the whole dataset.

CNs and TCNs, despite not containing generative components, are built with the adversarial principles inspired from GANs.

The idea to make multiple components in the network agree with each other has been adopted by several previous models.

For example, [2] proposed Parallel Consensus Networks, where multiple networks classify by majority voting.

Each of the networks is trained on features after a unique transform.

BID13 proposed consensus optimization in GAN in which a term is added to the utility functions of both the generator and discriminator to alleviate the adversity between generator and discriminator.

However, neither they nor semi-supervised learning utilized the multiple modalities.

Multi-modal learning is also referred to as multi-view learning.

BID18 computed multiple viewpoints from speech samples and classified cognitive impairments.

By contrast, our multi-view learning is semi-supervised, and can involve non-overlapping subsets of features.

In domain adaptation, some work has been applied to find a unified representation between domains, for example, by applying domain invariant training BID5 and semantic similarity loss BID15 .

However, our approach does not involve multiple domains -we only handle data from one domain.

Here, the term 'domain' refers to how the data are naturally generated, whereas the term 'modality' refers to how different aspects of data are observed.

In previous work, Consensus Networks BID24 was proposed for multimodal supervised learning.

We extend the model to be suitable for semi-supervised learning, resulting in Transductive Consensus Networks (TCNs).

This section also presents three variants: TCN-embed, TCN-svm, and TCN-AE.

Given labeled data, DISPLAYFORM0 , and unlabeled data, {x (i) } (where x (i) ??? X U ), we want to learn a model that reaches high accuracies in predicting labels in unlabeled data.

In the semisupervised learning setting, there are many more unlabeled data points than labeled: DISPLAYFORM1 Each data point x contains feature values from multiple modalities (i.e., 'views').

If M be the total number of modalities, then DISPLAYFORM2 m is consistent throughout the dataset.

E.g., there may be 200 (100) acoustic (semantic) features for each data point.

Here we briefly review the structures of CNs.

In a CN model, M interpreter networks I 1,...,M (.) each compress the corresponding modality for a data sample into a low-dimensional vector v. DISPLAYFORM0 We call these networks interpreters, because they interpret the feature spaces with representations.

In TCNs, a consensus is expected to be reached given representations from multiple views of the same data sample.

A discriminator network D(.) tries to identify the originating modality of each representation vector.

If we write the m th modality of the dataset as a set M m of vectors, then the discriminator function D(.) can be expressed as: DISPLAYFORM1 To prevent the discriminator from looking at only superficial aspects for each data sample in the forward pass, Consensus Networks BID24 include an additional 'noise modality' representation sampled from a normal distribution, with mean and variance determined by the 'non-noise' representations: DISPLAYFORM2 The model's discrimination loss L D is therefore defined as the cross entropy loss across all modalities (plus the noise modality), averaged across both labeled and unlabeled datasets X : DISPLAYFORM3 Finally, a classifier network C(.) predicts the probability of class assignment (y) from the combined representation vectors, given model parameters: DISPLAYFORM4 The classification loss L C is just the cross entropy loss across the labeled data: DISPLAYFORM5 The overall optimization goals for CN can therefore be described as: DISPLAYFORM6

Consensus Networks (CNs), as a supervised learning framework, are limited by the amount of labeled data.

This motivates us to generalize the approach to semi-supervised scenarios.

There are two mechanisms in the CN training procedures, namely the classifier mechanism and consensus mechanism.

The classifier mechanism requires labeled data, but the consensus mechanism does not explicitly require these labels.

We let the consensus mechanism handle both labeled and unlabeled data.

This results in Transductive Consensus Networks.

Formally, the loss functions are rewritten as: DISPLAYFORM0 where X consists of both labeled data X L and unlabeled data X U .

Overall, the optimization goal can still be written as: min DISPLAYFORM1 These goals set up a complex nonlinear optimization problem.

To figure out a solution, we break down the goals into three iterative steps, similar to GAN BID6 :??? The 'I' step encourages interpreters to produce indistinguishable representations: max DISPLAYFORM2 ??? The 'D' step encourages discriminators to recognize modal-specific information retained in representations: min DISPLAYFORM3 ??? The 'CI' step trains the networks to make a correct decision: min DISPLAYFORM4

The consensus mechanism builds a low-dimensional latent representation of each (labeled and unlabeled) data sample containing common knowledge across different modalities, and the classifier mechanism tries to make these representations meaningful.

Three modifications are made to our base TCN model, resulting in the following models:TCN-embed consists of the same networks as TCN but is trained slightly differently.

Before the I-D-CI optimization cycle, we add a pretraining phase with I-D iterations, which emphasizes the consensus mechanism.

TCN-svm removes the classifier network from TCN-embed.

After the pretraining phase across the whole dataset, we extract the representations of those labeled data samples to train a supervised learning classifier (i.e., an SVM).

TCN-svm discards the classifier mechanism, which results in deteriorations to model performance ( ??5).TCN-AE provides insights from another perspective.

In contrast to TCN, TCN-AE contains several additional reconstructor networks, R 1..M (.) (one per modality).

Each reconstructor network tries to recover the input modality from the corresponding low-dimensional representations (plus a small noise ): DISPLAYFORM0 Defining reconstruction loss as L R = E x???X E m |x m ??? x m | 2 , the optimization target in TCN-AE can be expressed as: DISPLAYFORM1 L C , and max DISPLAYFORM2 and min DISPLAYFORM3 TCN-AE is inspired by denoising autoencoder BID22 , where the existence of reconstructor networks encourage the latent variables to preserve realistic information.

This somehow works against the consensus mechanism, which according to BID24 tries to agree on simple representations.

TCN-AE therefore weakens the consensus mechanism.

We will show in ??5 that an inhibited consensus mechanism results in inferior model performance.

We want to quantitatively measure the effects of the consensus and the classification mechanisms.

To evaluate the similarities of representations, we treat the hidden dimensions of each representation DISPLAYFORM0 ..] (after normalization) as discrete values of a probability mass function 1 , which we write as p m .

The M modalities for each data point are therefore approximated by M probability distributions.

Now we can measure the relative JS divergences between each pair of representations v m and v n derived from the same data sample (D(p m ||p n )).

To acquire the relative value, we normalize the JS divergence by the total entropy in p m and p n : DISPLAYFORM1 where DISPLAYFORM2 where v m,j and v n,j are the j th component of v m and v n respectively.

In total, for each data sample with M modalities, DISPLAYFORM3 pairs of relative divergences are calculated.

We average the negative of these divergences to get the similarity: DISPLAYFORM4 Note that by our definition the maximum value of the "similarity" value is 0 (where there is no JS divergence between any pair of the representation vectors), and it has no theoretical lower bound.

FIG3 shows several 2D visualizations of representation vectors drawn from an arbitrary run.

In Figure 5 , we illustrate how the similarities between modality representations evolve during training.

We compare TCN and its variants on two benchmark datasets: Bank Marketing, and DementiaBank.

The full list of features, by modalities, are provided in the Supplementary Material.

The Bank Marketing dataset is from the UCI machine learning repository BID4 .

used for predicting whether the customer will subscribe a term deposit in a bank marketing campaign via telephone BID14 .

There are originally 4,640 positive samples (subscribed) and 36,548 negative ones (did not subscribe).

Since consensus network models do not work well on imbalanced datasets, we randomly sample 5,000 negative samples to create an (almost) balanced dataset.

We also convert the categorical raw features 2 into one-hot representations.

We then divide the features into three modalities: basic information, statistical data, and employment-related features.

1 There is a ReLU layer at output of each interpreter network, so the probability mass will be non-negative.

2 https://archive.ics.uci.edu/ml/datasets/bank+marketing DementiaBank 3 contains 473 spoken picture descriptions of the clinical "cookie-theft picture" [1] , containing 240 positive samples (the Dementia class) and 233 negative samples (the Control class).

We extract 413 linguistic features from each speech sample and their transcriptions, including acoustic (e.g., pause durations), lexical & semantic (e.g., average cosine similarities between words in sentences) and syntactic (e.g., complexity of the syntactic parse structures) modalities.

Table 1 : Basic information about the datasets (after preprocessing).

In the Bank Marketing dataset, the three modalities correspond to basic information, statistical data, and employment-related features.

In DementiaBank, the three modalities correspond to acoustic, syntactic, and lexical&semantic features.

Detailed descriptions of the features are included in supplementary materials.

We evaluate TCN and its variants against several benchmarks, including:1.

Multimodal semi-supervised learning benchmark: Tri-training BID23 4 .2.

TCN's supervised counterpart: Consensus Network (CN).3.

Unimodal semi-supervised learning: TSVM BID8 , ladder network BID19 , CatGAN BID21 .

For simplicity, we use fully connected networks for all of I 1..M , D, C, and R 1..

M in this paper.

To enable faster convergence, all fully connected networks have a batch normalization layer BID7 .

For training, the batch size is set to 10.

The neural network models are implemented using PyTorch BID16 , and supervised learning benchmark algorithms (SVM, MLP) use scikit-learn BID17 .We use the Adam optimizer BID9 with an initial learning rate of 0.001.

In training TCN, TCN-embed, and TCN-AE, optimizations are stopped when the classification loss does not change by more than 10 ???5 in comparison to the previous step, or when the step count reaches 100.

In the pre-training phase of TCN-embed and TCN-svm, training is stopped when the discrimination loss changes by less than 10 ???5 , or when pretraining step count reaches 20.Sometimes, the iterative optimization (i.e., the I-D-CI cycle for TCN / TCN-embed, and the I-D-RI-CI cycle for the TCN-AE variant) is trapped in local saddle points -the training classification loss does not change while the training classification loss is higher than log 2 ??? 0.693.

This is the expected loss of a binary classifier with zero knowledge.

If the training classification loss is higher than log 2, the model is re-initialized with a new random seed and the training is restarted.

Empirically, this re-initialization happens no more than once per ten runs, but the underlying cause needs to be examined further.5 Results and discussion

As shown in FIG1 , TCN outperforms or matches the best benchmarks.

On the Bank Marketing dataset, TCN, CN, and TSVM clearly outperform the rest.

On DementiaBank, Tri-train, TCN, and TSVM form the "first tier".Also, semi-supervised learning does not always outperform those supervised algorithms.

For example, on the Bank Marketing dataset, CN (i.e., TCN's supervised learning counterpart) holds the second best performance.

BID23 ), uni-modal semi-supervised (TSVM BID8 , Ladder BID19 , CatGAN BID21 ), and multi-modal supervised (CN BID24 ).

As shown in FIG2 , TCN aligns with or outperforms TCN-embed.

Both of these approaches significantly outperform TCN-AE.

On the other hand, TCN-svm produces almost trivial classifiers.

There are several points worth noting.??? Both the consensus and the classification mechanisms are beneficial to classifier performance.

The classification mechanism can be beneficial even with as few as 20 labeled data samples.??? Iterative optimization is crucial.

Without classification mechanisms, the consensus mechanism by itself fails to derive good reprensentations.

Without consensus mechanisms (i.e., when the reconstructors hinder the consensus mechanisms), accuracies drop significantly.

To understand more about TCN, we visualize them with T-SNE BID12 in FIG3 , and plot the similarity values in Figure 5 .???

The higher similarity values corresponds to a state where the distributions contain higher symmetry in aggregate manner.??? Measured from the similarity values, TCN models reach a consensus state where the similarities are stable.

TCN-svm reaches agreements quickly but are close to trivial.

TCN-AE, with the autoencoder blocking the consensus mechanism, fails to reach a state of agreement. .

The three colors represent three modalities.

At step 2, the representations are distributed randomly.

At step 110, they become mixed evenly.

The most interesting embedding happens at step 30, when representations of the three modalities form three 'drumstick' shapes.

With the highest visual symmetry, this configuration also has the highest similarity among the three.

Figure 5 : Examples of similarity plots against the number of steps taken, for DementiaBank using 80 labeled samples ("DB80", blue) and Bank Marketing using 20 labeled samples ("BM20", green).

The y axis are scaled to (???0.035, 0) except TCN-AE, where the relative JS divergences "explode".

Note that we stop the training procedure when losses converge (as detailed in ??4.3), so the trials may stop at different steps.

In this paper, we present Transductive Consensus Networks (TCNs) that extend consensus networks with semi-supervised learning.

We identify two mechanisms in which TCNs function, i.e., the consensus and classifier mechanisms.

With three TCN variants in an ablation study, we show the importance of both mechanisms.

Moreover, by treating the representations as probability distributions and defining their similarity as negative relative JS divergences, we show that although the consensus mechanism urges high similarities, a good consensus state might not need perfect similarities between modality representations.

In the future, several avenues may be considered.

To start with, building consensus networks using other types of neural networks may be considered.

In addition, more exploration could be done to find a more explainable metric to describe the extent of agreement.

Currently, we use ???

H1+H2 , but this requires some approximations.

Optimizing against the similarity metric, instead of setting up a discriminator, may be worth examining as well.

@highlight

TCN for multimodal semi-supervised learning + ablation study of its mechanisms + interpretations of latent representations