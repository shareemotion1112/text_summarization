We propose RaPP, a new methodology for novelty detection by utilizing hidden space activation values obtained from a deep autoencoder.

Precisely, RaPP compares input and its autoencoder reconstruction not only in the input space but also in the hidden spaces.

We show that if we feed a reconstructed input to the same autoencoder again, its activated values in a hidden space are equivalent to the corresponding reconstruction in that hidden space given the original input.

In order to aggregate the hidden space activation values, we propose two metrics, which enhance the novelty detection performance.

Through extensive experiments using diverse datasets, we validate that RaPP improves novelty detection performances of autoencoder-based approaches.

Besides, we show that RaPP outperforms recent novelty detection methods evaluated on popular benchmarks.

How can we characterize novelty when only normality information is given?

Novelty detection is the mechanism to decide whether a data sample is an outlier with respect to the training data.

This mechanism is especially useful in situations where a proportion of detection targets is inherently small.

Examples are fraudulent transaction detection (Pawar et al., 2014; Porwal & Mukund, 2018) , intrusion detection (Lee, 2017; Aoudi et al., 2018) , video surveillance (Ravanbakhsh et al., 2017; Xu et al., 2015b) , medical diagnosis (Schlegl et al., 2017; Baur et al., 2018) and equipment failure detection (Kuzin & Borovicka, 2016; Zhao et al., 2017; Beghi et al., 2014) .

Recently, deep autoencoders and their variants have shown outstanding performances in finding compact representations from complex data, and the reconstruction error has been chosen as a popular metric for detecting novelty (An & Cho, 2015; Vasilev et al., 2018) .

However, this approach has a limitation of measuring reconstruction quality only in an input space, which does not fully utilize hierarchical representations in hidden spaces identified by the deep autoencoder.

In this paper, we propose RAPP, a new method of detecting novelty samples exploiting hidden activation values in addition to the input values and their autoencoder reconstruction values.

While ordinary reconstruction-based methods carry out novelty detection by comparing differences between input data before the input layer and reconstructed data at the output layer, RAPP extends these comparisons to hidden spaces.

We first collect a set of hidden activation values by feeding the original input to the autoencoder.

Subsequently, we feed the autoencoder reconstructed input to the autoencoder to calculate another set of activation values in the hidden layers.

This procedure does not need additional training of the autoencoder.

In turn, we quantify the novelty of the input by aggregating these two sets of hidden activation values.

To this end, we devise two metrics.

The first metric measures the total amount of reconstruction errors in input and hidden spaces.

The second metric normalizes the reconstruction errors before summing up.

Note that RAPP falls back to the ordinary reconstruction-based method if we only aggregate input values before the input layer and the reconstructed values at the output layer.

Also, we explain the motivations that facilitated the development of RAPP.

We show that activation values in a hidden space obtained by feeding a reconstructed input to the autoencoder are equivalent to the corresponding reconstruction in that hidden space for the original input.

We refer the latter quantity as a hidden reconstruction of the input.

Note that this is a natural extension of the reconstruction to the hidden space.

Unfortunately, we cannot directly compute the hidden reconstruction as in the computation of the ordinary reconstruction because the autoencoder does not impose any correspondence between encoding-decoding pairs of hidden layers during the training.

Nevertheless, we show that it can be computed by feeding a reconstructed input to the autoencoder again.

Consequently, RAPP incorporates hidden reconstruction errors as well as the ordinary reconstruction error in detecting novelty.

With extensive experiments, we demonstrate using diverse datasets that our method effectively improves autoencoder-based novelty detection methods.

In addition, we show by evaluating on popular benchmark datasets that RAPP outperforms competing methods recently developed.

Our contributions are summarized as follows.

• We propose a new novelty detection method by utilizing hidden activation values of an input and its autoencoder reconstruction, and provide aggregation functions for them to quantify novelty of the input.

• We provide motivation that RAPP extends the reconstruction concept in the input space into the hidden spaces.

Precisely, we show that hidden activation values of a reconstructed input are equivalent to the corresponding hidden reconstruction of the original input.

• We demonstrate that RAPP improves autoencoder-based novelty detection methods in diverse datasets.

Moreover, we validate that RAPP outperforms recent novelty detection methods on popular benchmark datasets.

Various novelty detection methods with deep neural networks rely on the reconstruction error (Sakurada & Yairi, 2014; Hoffmann, 2007; An & Cho, 2015) , because discriminative learning schemes are not suitable for highly class-imbalanced data which is common in practice.

Unsupervised and semi-supervised learning approaches handle such imbalance by focusing on the characterization of normality and detecting samples out of the normality.

Variational Autoencoders (VAE) (Kingma & Welling, 2014) were reported to outperform vanilla autoencoders for novelty detection based on reconstruction error (An & Cho, 2015) .

To carry out the novelty detection outlined in this approach, an autoencoder needs to be trained only with normal data.

The autoencoder encodes the training data, which comprises of only normal data in this case, into a lower-dimensional space and decodes them to the input space.

To test novelty, an input value is fed to the autoencoder to produce a reconstructed value and calculate the distance between the input and reconstructed values.

This distance is the reconstruction error.

A higher reconstruction error means that the input value cannot be encoded onto the lower-dimensional space that represents normal data.

Therefore, the input value can be marked as a novelty if its reconstruction error exceeds a certain threshold.

Instead of autoencoders, Generative Adversarial Networks (GAN) have been also suggested to model a distribution of normal data (Sabokrou et al., 2018; Schlegl et al., 2017) .

Despite the same purpose of discovering a simpler, lower-dimensional representation, the training criterion for GAN is focusing on the quality of data generation rather than the reconstruction quality of training data.

Recently, several pieces of research have combined autoencoders and adversarial learning to meet both criteria in dimension reduction and data generation (Haloui et al., 2018; Pidhorskyi et al., 2018; Zenati et al., 2018) .

One limitation of these methods based on the ordinary reconstruction error is that they do not exploit all the information available along the projection pathway of deep autoencoders.

We will explain how to leverage this information for novelty detection in the next section.

From the viewpoint of the diversity and ratio of the normal data in novelty detection, there are two cases available.

The first case is when a small fraction of classes are normal.

This case has been studied in a one-class classification context, and usually evaluated by organizing training data into a collection of samples belonging to a small number of normal classes (Ruff et al., 2018; Perera & Patel, 2018; Sabokrou et al., 2018; Golan & El-Yaniv, 2018) .

The second case is when a majority of classes are assigned as normal (An & Cho, 2015; Schlegl et al., 2017; Haloui et al., 2018; Zenati et al., 2018) .

In this case, normal data is more diverse, and the training data is consist of samples of a relatively large number of normal classes: e.g., nine digits of MNIST.

One setup does not dominate the other, but depending on applications, either can be more suitable than the other.

Different methods may perform differently in both cases.

In this paper, we evaluate RAPP and other competing methods with experiments in both setups.

In this section, we describe the proposed novelty detection method RAPP based on an autoencoder.

The main idea is to compare hidden activations of an input and its hidden reconstructions along the projection pathway of the autoencoder.

To be precise, we project the input and its autoencoder reconstruction onto the hidden spaces to obtain pairs of activation values, and aggregate them to quantify the novelty of the input.

For the aggregation, we present two metrics to measure the total amount of difference within each pair.

An autoencoder A is a neural network consisting of an encoder g and a decoder f , responsible for dimension reduction and its inverse mapping to the original input space, respectively: i.e. A = f • g. For the purpose, training the autoencoder aims to minimize difference between its input x and output A(x).

The space that the encoder g constitutes is called the latent space, and provides more concise representation for data than the input space.

Due to this unsupervised representation learning property, the autoencoder has been widely used for novelty detection.

Specifically, training an autoencoder on normal data samples, novelty of a test sample x is measured by the following reconstruction error :

The test sample x is more likely to be novel as the error (x) becomes larger, because it means that x is farther from the manifold that the autoencoder describes.

Although this approach has shown promising results in novelty detection, the reconstruction error alone does not fully exploit information provided by a trained autoencoder especially when its architecture is deep.

In other words, hierarchical information identified by the deep architecture is being ignored.

This is rather unfortunate because hierarchical representation learning is one of the most successfully proven capabilities of deep neural networks.

To fully leverage that capability, below we will describe the way to exploit hidden spaces to capture the difference between normal and novel samples in more detail.

Let A = f • g be a trained autoencoder where g and f are an encoder and a decoder, and be the number of hidden layers of g. Namely, g = g • · · · • g 1 .

We define partial computation of g as follows:

Let x be an input vector, andx be its reconstruction by A: i.e.,x = A(x).

In addition to comparing x andx in the input space, as the ordinary approach does, we examine them in hidden spaces along a projection pathway of A. More precisely, feeding x andx into A, we obtain pairs (h i ,ĥ i ) of their hidden representations where Figure 1a illustrates the procedure of computing h i andĥ i .

As a result, novelty of the sample x is quantified by aggregating

The overall procedure of RAPP is summarized in Algorithm 1.

To clearly state the required variables to construct H, we write the algorithm with the for loop in Lines 3-5, but in practice, all of them can be computed by feed-forwarding one time each of x andx to g. Note that RAPP is indeed a generalization of the ordinary reconstruction method with defining g 0 as the identity function and s ord as follows.

The quantity that RAPP computes, the hidden activation of the reconstruction input, is equivalent to the hidden reconstruction of the input.

Sincef = f , computingĥ 2 =ĥ 2 does not require explicitly evaluatingf i but only g i and f =f .

Algorithm 1: RAPP to compute a novelty score.

Input : Sample x, trained autoencoder A = f • g, the number of layers , and aggregation s. Output: Novelty score S.

In this paper, we provide two metrics s SAP and s N AP which more extensively utilize H than s ord .

Those are especially suited when no prior knowledge exists for the selection of layers to derive a novelty metric, which commonly happens when modeling with deep neural networks.

Note that, however, more elaborate metrics can be designed if we have knowledge on or can characterize the spaces.

This is the most straightforward metric that one can define on H. For a data sample x, SAP is defined by summing the square of Euclidean distances for all pairs in H:

Although SAP is intuitive, it does not consider properties of hidden spaces; distance distributions of pairs in H may be different depending on the individual hidden spaces.

For instance, the magnitude of distances can depend on layers, or there may exist correlated neurons even across layers which are unintentionally emphasized in SAP.

To capture clearer patterns, we propose to normalize the distances via two steps: orthogonalization and scaling.

; given a training set X, let D be a matrix whose i-th row corresponds to d(x i ) for x i ∈ X, andD be the column-wise centered matrix of D. For the normalization, we computeD = U ΣV , SVD ofD, to obtain its singular values Σ and right singular vectors V .

For a given data sample x, we define s N AP as follows:

where µ X is the column-wise mean of D, and d(x) is expressed as a column vector.

Note that s N AP is equal to the Mahalanobis distance with the covariance matrix V ΣΣV .

Although SVD computation time is quadratic in the number of columns of the target matrix, we observe that its impact is relatively small in practical setups.

See Appendix A for more details.

One natural question in using the ordinary reconstruction method is as follows: why do we investigate only the input space?

Or, why do we not use information in hidden spaces?

While the reconstruction error in the input space is extensively employed, any similar concept does not exist in hidden spaces.

One reason is that the corresponding encoding and decoding layers are not guaranteed to express the same space: e.g. permuted dimensions.

This is because the autoencoder objective does not have any term involving activations from intermediate hidden layers.

As a result, f :i+1 (g(x)) cannot be considered a reconstruction of g :i (x), except for i = 0 with which they become the ordinary reconstruction of and input to an autoencoder, respectively.

Nevertheless, in this section, we will show that there is an indirect way to compute the hidden reconstruction.

Precisely, we will show thatĥ i (x) = g :i (A(x)) is indeed equivalent to a reconstruction of g :i (x).

The overall mechanism is depicted in Figure 1b.

Let A = f • g be a trained autoencoder, and M 0 = {A(x) : x ∈ R n } be the low dimensional manifold that A describes (Pidhorskyi et al., 2018) : i.e., ∀x ∈ M 0 , x = A(x).

Defining M i = {g :i (x) : x ∈ M 0 }, which is the low dimensional image of M 0 defined by g :i , g and f restricted on M 0 and M , respectively, are inverse functions of each other.

Quantifying Hidden Reconstruction We first assume that there exists a decoderf

The second condition makesf :i+1 a proper decoder corresponding to g i+1: , and thus,f enables to define the i-th hidden reconstructionĥ i (x) as follows:

Finally, we conclude thatĥ i (x) is equal toĥ i (x) for x ∈ M 0 as follows.

where we do not needf i for computingĥ i (x), but only g i and f .

Note that for x ∈ M 0 already on the manifold, its i-th hidden reconstructionĥ i (x) becomes equal to its corresponding hidden input

.

For x / ∈ M 0 , its hidden reconstructionĥ i (x) will differ from the input h i (x).

Existence off Since x = A(x) for x ∈ M 0 , g i and f i are one-to-one functions from M i−1 and M i , respectively.

Let us definef i = g

for M i ; then it also holdsf = g −1 for M .

This implies x = (f • g)(x) for x ∈ M 0 , and consequently,f = f on M .

This definition off i satisfies the two conditions above, and as discussed, we are able to compute hidden reconstructions given an input x, through computing the i-th hidden activation of the reconstructed input: i.e.ĥ i (x) = (

Existence off with Neural Networks Given g i , if the symmetric architecture forf i is used, we may not be able to learnf i = g −1 i .

Neural networks are, however, highly flexible frameworks in which we can deal with models of arbitrary function forms by adjusting network architecture.

This property enables us to design a layer capable of representingf i .

For instance, even iff i is too complicated to be represented with a single fully connected layer, we can still approximatef i by stacking multiple layers.

Hence, given g i ,f i can be represented by neural networks.

In this section, we evaluate RAPP in comparison to existing methods.

To this end, we tested the methods on several benchmarks and diverse datasets collected from Kaggle and the UCI repository which are suitable for evaluating novelty detection methods.

The datasets from Kaggle and the UCI repository are chosen from problem sets of anomaly detection and multi-class classification, summarized in Table 1 .

We note that MI-F and MI-V share the same feature matrix, but are considered to be different datasets because their labels normal and abnormal are assigned by different columns: i.e. machine completed and pass visual inspection, respectively.

We use these datasets to compare RAPP with standard autoencoder-based methods described in Section 5.2.

To compare RAPP with novelty detection methods in recent literatures, we also use popular benchmark datasets for evaluating deep learning techniques: MNIST (LeCun & Cortes, 2010) and F-MNIST (Xiao et al., 2017) .

For theses datasets, we do not take pre-split training and test sets, but instead merge them for post-processing.

Novelty detection detects novel patterns by focusing on deviations from model-learned normal patterns.

Thus, training sets contain only normal samples and test sets contain both normal and anomaly samples in our evaluation setups.

Precisely, if a dataset contains an anomaly label, we assign all samples with that label to the test set for detection.

If a dataset does not have any anomaly labels, we consider the following two setups.

• Multimodal Normality: A single class is chosen to be the novelty class and the remaining classes are assigned as the normal class.

This setup is repeated to produce sub-datasets with all possible novelty assignments.

For instance, MNIST results in a set of datasets with 10 different novelty classes.

• Unimodal Normality:

In contrast to the multimodal normality setup, we take one class for normality, and the others for novelty.

For instance, MNIST results in a set of datasets with 10 different normal classes.

We applied these two setups to STL, OTTO, SNSR, MNIST, and F-MNIST datasets.

We compare RAPP and the other methods using Area Under Receiver Operating Characteristic (AUROC).

Note that we do not employ thresholding-based metrics such as F 1 score because access to abnormal samples is only allowed in testing time.

Hence, we focus on the separability of models for novelty with AUROC.

For the datasets in Table 1 , we compare the effectiveness of the reconstruction error, SAP and NAP for three models: Autoencoder (AE), Variational Autoencoder (VAE), Adversarial Autoencoder (AAE) (Makhzani et al., 2016) .

For the benchmark datasets, recent approaches including OCNN (Chalapathy et al., 2018) , GPND (Pidhorskyi et al., 2018) , DSVDD (Ruff et al., 2018) and GT (Golan & El-Yaniv, 2018 ) are available.

To obtain the performances of the existing approaches, we downloaded their codes and applied against our problem setups.

Given novelty classes, we create the test sets by randomly selecting samples while maintaining novelty ratios to 35% for the multimodal and 50% for the unimodal normality setups, respectively.

Note that the expectation value of AUROC is invariant to the novelty ratio.

We use symmetric architecture with fully-connected layers for the three base models, AE, VAE, and AAE.

Each encoder and decoder has 10 layers with different bottleneck size.

For the Kaggle and UCI datasets, we carry out PCA for each dataset first.

The minimum number of principal components that explain at least 90% of the variance is selected as the bottleneck size of the autoencoders.

We set bottleneck size to 20 for benchmark datasets.

Leaky-ReLU (Xu et al., 2015a) activation and batch normalization (Ioffe & Szegedy, 2015) layers are appended to all layers except the last layer.

We train AE, VAE and AAE with Adam optimizer (Kingma & Ba, 2015) , and select the model with the lowest validation loss as the best model.

For training stability of VAE, 10 Monte Carlo samples were averaged in the reparamterization trick (Kingma & Welling, 2014 ) to obtain reconstruction from the decoder.

In the calculation of SAP and NAP, we excluded reconstructions in the input space for MNIST and F-MNIST.

Each AUROC score is obtained by averaging AUROC scores from five trials to reduce the random errors in training neural networks.

More results are provided in Appendix: standard deviations in Appendix B, comparison to baselines other than autoencoder variants C, and the effect of varying hidden layers involved in RAPP computation in Appendix D. Table 2 summarizes the result of our performance evaluation; the best score for each model is in bold, and the best score for each dataset with an underline.

Since STL, OTTO, SNSR, MNIST, and F-MNIST do not have anomaly labels, their scores are averaged over all possible anomaly class assignments.

For instance, the AUROC value for OTTO in the unimodal normality setup is the average of 9 AUROC values with different anomaly class assignments.

In Table 2 , RAPP shows the highest AUROC scores for most of the cases.

If we examine the performance for each dataset, RAPP achieves the best for 13 cases out of 15 (see the underlines).

Table 3 summarizes the comparison of RAPP to recent novelty detection methods.

As in Table 2 , AUROC values are calculated by averaging results from 10 cases with different anomaly class assignments for both datasets.

Except for the unimodal F-MNIST setup, NAP outperforms all competing methods regardless of base model choice.

Notably, NAP combined with VAE always shows the best performance, which is even higher than that of GT relying on image-specific data transformations for all cases.

In this paper, we propose a novelty detection method which utilizes hidden reconstructions along a projection pathway of deep autoencoders.

To this end, we extend the concept of reconstruction in the input space to hidden spaces found by an autoencoder and present a tractable way to compute the hidden reconstructions, which requires neither modifying nor retraining the autoencoder.

Our experimental results show that the proposed method outperforms other competing methods in terms of AUROC for diverse datasets including popular benchmarks.

A SVD COMPUTATION TIME

We compare running times of training an autoencoder and computing SVD for NAP.

We choose two packages for the SVD computation: Pytorch SVD and fbpca provided in https://fbpca.

readthedocs.io/en/latest/.

Since the time complexity of SVD is linear in the number of data samples 1 , we mainly focus on the performance of SVD with varying the number of columns of the input matrix that SVD is applied.

To obtain variable sizes of the columns, we vary the depth and bottleneck size of autoencoders.

The result is shown below.

Notably, Pytorch SVD utilizing GPU is at least 47x faster than training neural networks.

Even, fbpca running only on CPU achieves at least 2.4x speedup.

The detailed setups to obtain the matrices for the experiment are given in the 1  20  100  20  40  20  90  2  18  100  18  40  18  90  3  16  100  16  40  16  90  4  14  100  14  40  14  90  5  12  100  12  40  12  90  6  10  100  10  40  10  90  7  8  100  8  40  8  90  8  6  100  6  40  6  90  9  4  100  4  40  4  90  10  2  100  2  40  2  90  11  2  80  2  30  2  70  12  2  60  2  20  2  50  13  2  40  2  10  2  30  14  2  20  2  10 B STANDARD DEVIATIONS OF EXPERIMENTAL RESULTS

We provide the standard deviations of the result in We investigate the performance of NAP while increasing the number of hidden layers involved in the NAP computation.

Specifically, we consider two ways for the increment: 1) adding hidden layers one by one from the input layer (forward addition), and 2) adding hidden layers one by one from the bottleneck layer (backward addition).

Experimental results on two datasets are shown below.

For most cases, more hidden layers tend to result in higher performance.

The values are obtained from one trial, not averaged over 5 trials as done in Section 5.

@highlight

A new methodology for novelty detection by utilizing hidden space activation values obtained from a deep autoencoder.