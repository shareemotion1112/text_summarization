Unsupervised anomaly detection on multi- or high-dimensional data is of great importance in both fundamental machine learning research and industrial applications, for which density estimation lies at the core.

Although previous approaches based on dimensionality reduction followed by density estimation have made fruitful progress, they mainly suffer from decoupled model learning with inconsistent optimization goals and incapability of preserving essential information in the low-dimensional space.

In this paper, we present a Deep Autoencoding Gaussian Mixture Model (DAGMM) for unsupervised anomaly detection.

Our model utilizes a deep autoencoder to generate a low-dimensional representation and reconstruction error for each input data point, which is further fed into a Gaussian Mixture Model (GMM).

Instead of using decoupled two-stage training and the standard Expectation-Maximization (EM) algorithm, DAGMM jointly optimizes the parameters of the deep autoencoder and the mixture model simultaneously in an end-to-end fashion, leveraging a separate estimation network to facilitate the parameter learning of the mixture model.

The joint optimization, which well balances autoencoding reconstruction, density estimation of latent representation, and regularization, helps the autoencoder escape from less attractive local optima and further reduce reconstruction errors, avoiding the need of pre-training.

Experimental results on several public benchmark datasets show that, DAGMM significantly outperforms state-of-the-art anomaly detection techniques, and achieves up to 14% improvement based on the standard F1 score.

Unsupervised anomaly detection is a fundamental problem in machine learning, with critical applications in many areas, such as cybersecurity BID18 ), complex system management BID14 ), medical care BID10 ), and so on.

At the core of anomaly detection is density estimation: given a lot of input samples, anomalies are those ones residing in low probability density areas.

Although fruitful progress has been made in the last several years, conducting robust anomaly detection on multi-or high-dimensional data without human supervision remains a challenging task.

Especially, when the dimensionality of input data becomes higher, it is more difficult to perform density estimation in the original feature space, as any input sample could be a rare event with low probability to observe BID3 ).

To address this issue caused by the curse of dimensionality, two-step approaches are widely adopted BID2 ), in which dimensionality reduction is first conducted, and then density estimation is performed in the latent low-dimensional space.

However, these approaches could easily lead to suboptimal performance, because dimensionality reduction in the first step is unaware of the subsequent density estimation task, and the key information for anomaly detection could be removed in the first place.

Therefore, it is desirable to combine the force of dimensionality reduction and density estimation, although a joint optimization accounting for these two components is usually computationally difficult.

Several recent works BID29 ; BID26 ; BID24 ) explored this direction by utilizing the strong modeling capacity of deep networks, but the resulting performance is limited either by a reduced low-dimensional space that is unable to preserve essential information of input samples, an over-simplified density estimation model without enough capacity, or a training strategy that does not fit density estimation tasks.

Figure 1: Low-dimensional representations for samples from a private cybersecurity dataset: (1) each sample denotes a network flow that originally has 20 dimensions, (2) red/blue points are abnormal/normal samples, (3) the horizontal axis denotes the reduced 1-dimensional space learned by a deep autoencoder, and (4) the vertical axis denotes the reconstruction error induced by the 1-dimensional representation.

In this paper, we propose Deep Autoencoding Gaussian Mixture Model (DAGMM), a deep learning framework that addresses the aforementioned challenges in unsupervised anomaly detection from several aspects.

First, DAGMM preserves the key information of an input sample in a low-dimensional space that includes features from both the reduced dimensions discovered by dimensionality reduction and the induced reconstruction error.

From the example shown in Figure 1 , we can see that anomalies differ from normal samples in two aspects: (1) anomalies can be significantly deviated in the reduced dimensions where their features are correlated in a different way; and (2) anomalies are harder to reconstruct, compared with normal samples.

Unlike existing methods that only involve one of the aspects BID32 ; BID29 ) with sub-optimal performance, DAGMM utilizes a sub-network called compression network to perform dimensionality reduction by an autoencoder, which prepares a low-dimensional representation for an input sample by concatenating reduced low-dimensional features from encoding and the reconstruction error from decoding.

Second, DAGMM leverages a Gaussian Mixture Model (GMM) over the learned low-dimensional space to deal with density estimation tasks for input data with complex structures, which are yet rather difficult for simple models used in existing works BID29 ).

While GMM has strong capability, it also introduces new challenges in model learning.

As GMM is usually learned by alternating algorithms such as Expectation-Maximization (EM) (Huber (2011)), it is hard to perform joint optimization of dimensionality reduction and density estimation favoring GMM learning, which is often degenerated into a conventional two-step approach.

To address this training challenge, DAGMM utilizes a sub-network called estimation network that takes the low-dimensional input from the compression network and outputs mixture membership prediction for each sample.

With the predicted sample membership, we can directly estimate the parameters of GMM, facilitating the evaluation of the energy/likelihood of input samples.

By simultaneously minimizing reconstruction error from compression network and sample energy from estimation network, we can jointly train a dimensionality reduction component that directly helps the targeted density estimation task.

Finally, DAGMM is friendly to end-to-end training.

Usually, it is hard to learn deep autoencoders by end-to-end training, as they can be easily stuck in less attractive local optima, so pre-training is widely adopted BID22 ; BID26 ; BID24 ).

However, pre-training limits the potential to adjust the dimensionality reduction behavior because it is hard to make any significant change to a well-trained autoencoder via fine-tuning.

Our empirical study demonstrates that, DAGMM is well-learned by the end-to-end training, as the regularization introduced by the estimation network greatly helps the autoencoder in the compression network escape from less attractive local optima.

Experiments on several public benchmark datasets demonstrate that, DAGMM has superior performance over state-of-the-art techniques, with up to 14% improvement of F1 score for anomaly detection.

Moreover, we observe that the reconstruction error from the autoencoder in DAGMM by the end-to-end training is as low as the one made by its pre-trained counterpart, while the reconstruction error from an autoencoder without the regularization from the estimation network stays high.

In addition, the end-to-end trained DAGMM significantly outperforms all the baseline methods that rely on pre-trained autoencoders.

Tremendous effort has been devoted to unsupervised anomaly detection BID3 , and the existing methods can be grouped into three categories.

Reconstruction based methods assume that anomalies are incompressible and thus cannot be effectively reconstructed from low-dimensional projections.

Conventional methods in this category include Principal Component Analysis (PCA) BID8 ) with explicit linear projections, kernel PCA with implicit non-linear projections induced by specific kernels (G??nter et al.) , and Robust PCA (RPCA) (Huber (2011); BID2 ) that makes PCA less sensitive to noise by enforcing sparse structures.

In addition, multiple recent works propose to analyze the reconstruction error induced by deep autoencoders, and demonstrate promising results BID31 ; BID29 ).

However, the performance of reconstruction based methods is limited by the fact that they only conduct anomaly analysis from a single aspect, that is, reconstruction error.

Although the compression on anomalous samples could be different from the compression on normal samples and some of them do demonstrate unusually high reconstruction errors, a significant amount of anomalous samples could also lurk with a normal level of error, which usually happens when the underlying dimensionality reduction methods have high model complexity or the samples of interest are noisy with complex structures.

Even in these cases, we still have the hope to detect such "lurking" anomalies, as they still reside in low-density areas in the reduced low-dimensional space.

Unlike the existing reconstruction based methods, DAGMM considers the both aspects, and performs density estimation in a low-dimensional space derived from the reduced representation and the reconstruction error caused by the dimensionality reduction, for a comprehensive view.

Clustering analysis is another popular category of methods used for density estimation and anomaly detection, such as multivariate Gaussian Models, Gaussian Mixture Models, k-means, and so on BID1 ; BID32 ; BID11 BID25 ).

Because of the curse of dimensionality, it is difficult to directly apply such methods to multi-or high-dimensional data.

Traditional techniques adopt a two-step approach BID3 ), where dimensionality reduction is conducted first, then clustering analysis is performed, and the two steps are separately learned.

One of the drawbacks in the two-step approach is that dimensionality reduction is trained without the guidance from the subsequent clustering analysis, thus the key information for clustering analysis could be lost during dimensionality reduction.

To address this issue, recent works propose deep autoencoder based methods in order to jointly learn dimensionality reduction and clustering components.

However, the performance of the state-of-the-art methods is limited by over-simplified clustering models that are unable to handle clustering or density estimation tasks for data of complex structures, or the pre-trained dimensionality reduction component (i.e., autoencoder) has little potential to accommodate further adjustment by the subsequent fine-tuning for anomaly detection.

DAGMM explicitly addresses these issues by a sub-network called estimation network that evaluates sample density in the low-dimensional space produced by its compression network.

By predicting sample mixture membership, we are able to estimate the parameters of GMM without EM-like alternating procedures.

Moreover, DAGMM is friendly to end-to-end training so that we can unleash the full potential of adjusting dimensionality reduction components and jointly improve the quality of clustering analysis/density estimation.

In addition, one-class classification approaches are also widely used for anomaly detection.

Under this framework, a discriminative boundary surrounding the normal instances is learned by algorithms, such as one-class SVM BID4 ; BID17 ; BID23 ).

When the number of dimensions grows higher, such techniques usually suffer from suboptimal performance due to the curse of dimensionality.

Unlike these methods, DAGMM estimates data density in a jointly learned low-dimensional space for more robust anomaly detection.

There has been growing interest in joint learning of dimensionality reduction (feature selection) and Gaussian mixture modeling.

BID27 BID28 propose a method that jointly learns linear dimensionality reduction and GMM.

BID16 studies how to perform better feature selection with a pre-trained GMM as a regularizer.

BID21 and BID30 propose joint learning frameworks, where the parameters of GMM are directly estimated through supervision information in speech recognition applications.

BID19 b) investigate how to use log-linear mixture models to approximate GMM posterior under the conditions that a class/mixture prior distribution is given and a covariance matrix is globally shared.

Unlike the existing works, we focus on unsupervised settings: DAGMM extracts useful features for anomaly detection through non-linear dimensionality reduction realized by a deep autoencoder, and jointly learns their density under the GMM framework by mixture membership estimation, for which DAGMM can be viewed as a more powerful deep unsupervised version of adaptive mixture of experts BID7 ) in combination with a deep autoencoder.

More importantly, DAGMM combines induced reconstruction error and learned latent representation for unsupervised anomaly detection.

Deep Autoencoding Gaussian Mixture Model (DAGMM) consists of two major components: a compression network and an estimation network.

As shown in FIG0 , DAGMM works as follows: (1) the compression network performs dimensionality reduction for input samples by a deep autoencoder, prepares their low-dimensional representations from both the reduced space and the reconstruction error features, and feeds the representations to the subsequent estimation network; (2) the estimation network takes the feed, and predicts their likelihood/energy in the framework of Gaussian Mixture Model (GMM).

The low-dimensional representations provided by the compression network contains two sources of features: (1) the reduced low-dimensional representations learned by a deep autoencoder; and (2) the features derived from reconstruction error.

Given a sample x, the compression network computes its low-dimensional representation z as follows.

DISPLAYFORM0 DISPLAYFORM1 where z c is the reduced low-dimensional representation learned by the deep autoencoder, z r includes the features derived from the reconstruction error, ?? e and ?? d are the parameters of the deep autoencoder, x is the reconstructed counterpart of x, h(??) denotes the encoding function, g(??) denotes the decoding function, and f (??) denotes the function of calculating reconstruction error features.

In particular, z r can be multi-dimensional, considering multiple distance metrics such as absolute Euclidean distance, relative Euclidean distance, cosine similarity, and so on.

In the end, the compression network feeds z to the subsequent estimation network.

Given the low-dimensional representations for input samples, the estimation network performs density estimation under the framework of GMM.In the training phase with unknown mixture-component distribution ??, mixture means ??, and mixture covariance ??, the estimation network estimates the parameters of GMM and evaluates the likelihood/energy for samples without alternating procedures such as EM BID32 ).

The estimation network achieves this by utilizing a multi-layer neural network to predict the mixture membership for each sample.

Given the low-dimensional representations z and an integer K as the number of mixture components, the estimation network makes membership prediction as follows.

DISPLAYFORM0 where?? is a K-dimensional vector for the soft mixture-component membership prediction, and p is the output of a multi-layer network parameterized by ?? m .

Given a batch of N samples and their membership prediction, ???1 ??? k ??? K, we can further estimate the parameters in GMM as follows.

DISPLAYFORM1 where?? i is the membership prediction for the low-dimensional representation z i , and?? k ,?? k ,?? k are mixture probability, mean, covariance for component k in GMM, respectively.

With the estimated parameters, sample energy can be further inferred by DISPLAYFORM2 where | ?? | denotes the determinant of a matrix.

In addition, during the testing phase with the learned GMM parameters, it is straightforward to estimate sample energy, and predict samples of high energy as anomalies by a pre-chosen threshold.

Given a dataset of N samples, the objective function that guides DAGMM training is constructed as follows.

DISPLAYFORM0 This objective function includes three components.??? L(x i , x i ) is the loss function that characterizes the reconstruction error caused by the deep autoencoder in the compression network.

Intuitively, if the compression network could make the reconstruction error low, the low-dimensional representation could better preserve the key information of input samples.

Therefore, a compression network of lower reconstruction error is always desired.

In practice, L 2 -norm usually gives desirable results, as L( DISPLAYFORM1 ??? E(z i ) models the probabilities that we could observe the input samples.

By minimizing the sample energy, we look for the best combination of compression and estimation networks that maximize the likelihood to observe input samples.??? DAGMM also has the singularity problem as in GMM: trivial solutions are triggered when the diagonal entries in covariance matrices degenerate to 0.

To avoid this issue, we penalize small values on the diagonal entries by DISPLAYFORM2 , where d is the number of dimensions in the low-dimensional representations provided by the compression network.??? ?? 1 and ?? 2 are the meta parameters in DAGMM.

In practice, ?? 1 = 0.1 and ?? 2 = 0.005 usually render desirable results.

In DAGMM, we leverage the estimation network to make membership prediction for each sample.

From the view of probabilistic graphical models, the estimation network plays an analogous role of latent variable (i.e., sample membership) inference.

Recently, neural variational inference BID15 ) has been proposed to employ deep neural networks to tackle difficult latent variable inference problems, where exact model inference is intractable and conventional approximate methods cannot scale well.

Theoretically, we can also adapt the membership prediction task of DAGMM into the framework of neural variational inference.

For sample x i , the contribution of its compressed representation z i to the energy function can be upper-bounded as follows BID9 ), DISPLAYFORM0 (10) where Q ??m (k | z i ) is the estimation network that predicts the membership of z i , KL(??||??) is the Kullback-Leibler divergence between two input distributions, p(k) = ?? k is the mixing coefficient to be estimated, and p(k | z i ) is the posterior probability distribution of mixture component k given z i .By minimizing the negative evidence lower bound in Equation (8) , we can make the estimation network approximate the true posterior and tighten the bound of energy function.

In DAGMM, we use Equation (6) as a part of the objective function instead of its upper bound in Equation FORMULA0 simply because the energy function of DAGMM is tractable and efficient to evaluate.

Unlike neural variational inference that uses the deep estimation network to define a variational posterior distribution as described above, DAGMM explicitly employs the deep estimation network to parametrize a sampledependent prior distribution.

In the history of machine learning research, there were research efforts towards utilizing neural networks to calculate sample membership in mixture models, such as adaptive mixture of experts BID7 ).

From this perspective, DAGMM can be viewed as a powerful deep unsupervised version of adaptive mixture of experts in combination with a deep autoencoder.

Unlike existing deep autoencoder based methods BID26 ; BID24 ) that rely on pre-training, DAGMM employs end-to-end training.

First, in our study, we find that pre-trained compression networks suffer from limited anomaly detection performance, as it is difficult to make significant changes in the well-trained deep autoencoder to favor the subsequent density estimation tasks.

Second, we also find that the compression network and estimation network could mutually boost each others' performance.

On one hand, with the regularization introduced by the estimation network, the deep autoencoder in the compression network learned by end-to-end training can reduce reconstruction error as low as the error from its pre-trained counterpart, which meanwhile cannot be achieved by simply performing end-to-end training with the deep autoencoder alone.

On the other hand, with the well-learned low-dimensional representations from the compression network, the estimation network is able to make meaningful density estimations.

In Section 4.5, we employ an example from a public benchmark dataset to discuss the choice between pre-training and end-to-end training in DAGMM.

In this section, we use public benchmark datasets to demonstrate the effectiveness of DAGMM in unsupervised anomaly detection.

We employ four benchmark datasets: KDDCUP, Thyroid, Arrhythmia, and KDDCUP-Rev.??? KDDCUP.

The KDDCUP99 10 percent dataset from the UCI repository BID13 ) originally contains samples of 41 dimensions, where 34 of them are continuous and 7 are categorical.

For categorical features, we further use one-hot representation to encode them, and eventually we obtain a dataset of 120 dimensions.

As 20% of data samples are labeled as "normal" and the rest are labeled as "attack", "normal" samples are in a minority group; therefore, "normal" ones are treated as anomalies in this task.??? Thyroid.

The Thyroid BID13 ) dataset is obtained from the ODDS repository 1 .

There are 3 classes in the original dataset.

In this task, the hyperfunction class is treated as anomaly class and the other two classes are treated as normal class, because hyperfunction is a clear minority class.??? Arrhythmia.

The Arrhythmia BID13 ) dataset is also obtained from the ODDS repository.

The smallest classes, including 3, 4, 5, 7, 8, 9, 14, and 15 , are combined to form the anomaly class, and the rest of the classes are combined to form the normal class.??? KDDCUP-Rev.

This dataset is derived from KDDCUP.

We keep all the data samples labeled as "normal" and randomly draw samples labeled as "attack" so that the ratio between "normal" and "attack" is 4 : 1.

In this way, we obtain a dataset with anomaly ratio 0.2, where "attack" samples are in a minority group and treated as anomalies.

Note that "attack" samples are not fixed, and we randomly draw "attack" samples in every single run.

Detailed information about the datasets is shown in Table 1 .

We consider both traditional and state-of-the-art deep learning methods as baselines.??? OC-SVM.

One-class support vector machine BID4 ) is a popular kernel-based method used in anomaly detection.

In the experiment, we employ the widely adopted radial basis function (RBF) kernel in all the tasks.??? DSEBM-e.

Deep structured energy based model (DSEBM) BID29 ) is a state-ofthe-art deep learning method for unsupervised anomaly detection.

In DSEBM-e, sample energy is leveraged as the criterion to detect anomalies.??? DSEBM-r.

DSEBM-e and DSEBM-r BID29 ) share the same core technique, but reconstruction error is used as the criterion in DSEBM-r for anomaly detection.??? DCN.

Deep clustering network (DCN) BID26 ) is a state-of-the-art clustering algorithm that regulates autoencoder performance by k-means.

We adapt this technique to anomaly detection tasks.

In particular, the distance between a sample and its cluster center is taken as the criterion for anomaly detection: samples that are farther from their cluster centers are more likely to be anomalies.

Moreover, we include the following DAGMM variants as baselines to demonstrate the importance of individual components in DAGMM.??? GMM-EN.

In this variant, we remove the reconstruction error component from the objective function of DAGMM.

In other words, the estimation network in DAGMM performs membership estimation without the constraints from the compression network.

With the learned membership estimation, we infer sample energy by Equation FORMULA3 and FORMULA4 under the GMM framework.

Sample energy is used as the criterion for anomaly detection.??? PAE.

We obtain this variant by removing the energy function from the objective function of DAGMM, and this DAGMM variant is equivalent to a deep autoenoder.

To ensure the compression network is well trained, we adopt the pre-training strategy BID22 ).

Sample reconstruction error is the criterion for anomaly detection.??? E2E-AE.

This variant shares the same setting with PAE, but the deep autoencoder is learned by end-to-end training.

Sample reconstruction error is the criterion for anomaly detection??? PAE-GMM-EM.

This variant adopts a two-step approach.

At step one, we learn the compression network by pre-training deep autoencoder.

At step two, we use the output from the compression network to train the GMM by a traditional EM algorithm.

The training procedures in the two steps are separated.

Sample energy is used as the criterion for anomaly detection.??? PAE-GMM.

This variant also adopts a two-step approach.

At step one, we learn the compression network by pre-training deep autoencoder.

At step two, we use the output from the compression network to train the estimation network.

The training procedures in the two steps are separated.

Sample energy is used as the criterion for anomaly detection.??? DAGMM-p.

This variant is a compromise between DAGMM and PAE-GMM: we first train the compression network by pre-training, and then fine-tune DAGMM by end-to-end training.

Sample energy is the criterion for anomaly detection.??? DAGMM-NVI.

The only difference between this variant and DAGMM is that this variant adopts the framework of neural variational inference BID15 ) and replaces Equation (6) with the upper bound in Equation (10) as a part of the objective function.

In all the experiment, we consider two reconstruction features from the compression network: relative Euclidean distance and cosine similarity.

Given a sample x and its reconstructed counterpart x , their relative Euclidean distance is defined as DISPLAYFORM0 , and the cosine similarity is derived by x??x x 2 x 2 .

In Appendix D, for readers of interest, we discuss why reconstruction features are important to DAGMM and how to select reconstruction features in practice.

The network structures of DAGMM used on individual datasets are summarized as follows.??? KDDCUP.

For this dataset, its compression network provides 3 dimensional input to the estimation network, where one is the reduced dimension and the other two are from the reconstruction error.

The estimation network considers a GMM with 4 mixture components for the best performance.

In particular, the compression network runs with FC(120, 60, ??? Thyroid.

The compression network for this dataset also provides 3 dimensional input to the estimation network, and the estimation network employs 2 mixture components for the best performance.

In particular, the compression network runs with FC(6, 12, tanh)-FC(12, 4, where FC(a, b, f ) means a fully-connected layer with a input neurons and b output neurons activated by function f (none means no activation function is used), and Drop(p) denotes a dropout layer with keep probability p during training.

All the DAGMM instances are implemented by tensorflow BID0 ) and trained by Adam BID12 ) algorithm with learning rate 0.0001.

For KDDCUP, Thyroid, Arrhythmia, and KDDCUP-Rev, the number of training epochs are 200, 20000, 10000, and 400, respectively.

For the sizes of mini-batches, they are set as 1024, 1024, 128, and 1024, respectively.

Moreover, in all the DAGMM instances, we set ?? 1 as 0.1 and ?? 2 as 0.005.

For readers of interest, we discuss how ?? 1 and ?? 2 impact DAGMM in Appendix F.For the baseline methods, we conduct exhaustive search to find the optimal meta parameters for them in order to achieve the best performance.

We detail their exact configuration in Appendix A.

Metric.

We consider average precision, recall, and F 1 score as intuitive ways to compare anomaly detection performance.

In particular, based on the anomaly ratio suggested in Table 1 , we select the threshold to identify anomalous samples.

For example, when DAGMM performs on KDDCUP, the top 20% samples of the highest energy will be marked as anomalies.

We take anomaly class as positive, and define precision, recall, and F 1 score accordingly.

In the first set of experiment, we follow the setting in BID29 ) with completely clean training data: in each run, we take 50% of data by random sampling for training with the rest 50% reserved for testing, and only data samples from the normal class are used for training models.

Table 2 reports the average precision, recall, and F 1 score after 20 runs for DAGMM and its baselines.

In general, DAGMM demonstrates superior performance over the baseline methods in terms of F 1 score on all the datasets.

Especially on KDDCUP and KDDCUP-Rev, DAGMM achieves 14% and 10% improvement at F 1 score, compared with the existing methods.

For OC-SVM, the curse of dimensionality could be the main reason that limits its performance.

For DSEBM, while it works reasonably well on multiple datasets, DAGMM outperforms as both latent representation and reconstruction error are jointly considered in energy modeling.

For DCN, PAE-GMM, and DAGMM-p, their performance could be limited by the pre-trained deep autoencoders.

When a deep autoencoder is well-trained, it is hard to make any significant change on the reduced dimensions and favor the subsequent density estimation tasks.

For GMM-EN, without the reconstruction constraints, it seems difficult to perform reasonable density estimation.

In terms of PAE, the single view of reconstruction error may not be sufficient for anomaly detection tasks.

For E2E-AE, we observe that it is unable to reduce reconstruction error as low as PAE and DAGMM do on KDDCUP, KDDCUP-Rev, and Thyroid.

As the key information of data could be lost during dimensionality reduction, E2E-AE suffers poor performance on KDDCUP and Thyroid.

In addition, the performance of DAGMM and DAGMM-NVI is quite similar.

As GMM is a fairly simple graphical model, we cannot spot significant improvement brought by neural variational inference in DAGMM.

In Appendix B, for readers of interest, we show the cumulative distribution functions of the energy function learned by DAGMM for all the datasets under the setting of clean training data.

Table 2 : Average precision, recall, and F 1 from DAGMM and the baseline methods.

For each metric, the best result is shown in bold.

In the second set of experiment, we investigate how DAGMM responds to contaminated training data.

In each run, we reserve 50% of data by random sampling for testing.

For the rest 50%, we take all samples from the normal class mixed with c% of samples from the anomaly class for model training.

Table 3 : Anomaly detection results on contaminated training data from KDDCUP Table 3 reports the average precision, recall, and F 1 score after 20 runs of DAGMM, DCN, DSEBMe, and OC-SVM on the KDDCUP dataset, respectively.

As expected, contaminated training data negatively affect detection accuracy.

When contamination ratio c increases from 1% to 5%, average precision, recall, and F 1 score decrease for all the methods.

Meanwhile, we notice that DAGMM is able to maintain good detection accuracy with 5% contaminated data.

For OC-SVM, we adopt the same parameter setting used in the experiment with clean training data, and observe that OC-SVM is more sensitive to contamination ratio.

In order to receive better detection accuracy, it is important to train a model with high-quality data (i.e., clean or keeping contamination ratio as low as possible).In sum, the DAGMM learned by end-to-end training achieves the state-of-the-art accuracy on the public benchmark datasets, and provides a promising alternative for unsupervised anomaly detection.

In this section, we use an example to demonstrate the advantage of DAGMM learned by end-to-end training, compared with the baselines that rely on pre-trained deep autoencoders.

DISPLAYFORM0 Figure 3: KDDCUP samples in the learned 3-dimensional space by DAGMM, PAE, DAGMM-p, and DCN, where red points are samples from anomaly class and blue ones are samples from normal class Figure 3 shows the low-dimensional representation learned by DAGMM, PAE, DAGMM-p, and DCN, from one of the experiment runs on the KDDCUP dataset.

First, we can see from Figure 3a that DAGMM can better separate anomalous samples from normal samples in the learned low-dimensional space, while anomalies overlap more with normal samples in the low-dimensional space learned by PAE, DAGMM-p, or DCN.

Second, Even if DAGMM-p and DCN take effort to fine-tune the pre-trained deep autoencoder by its estimation network or k-means regularization, one could barely see significant change among Figure 3b , Figure 3c , and Figure 3d , where many anomalous samples are still mixed with normal samples.

Indeed, when a deep autoencoder is pre-trained, it tends to be stuck in a good local optima for the purpose of reconstruction only, but it could be suboptimal for the subsequent density estimation tasks.

In addition, in our study, we find that the reconstruction error in a trained DAGMM is as low as the error received from a pre-trained deep autoencoder (e.g., around 0.26 in terms of per-sample reconstruction error for KDDCUP).

Meanwhile, we also observe that it is difficult to reduce the reconstruction error for a deep autoencoder of the identical structure by endto-end training (e.g., around 1.13 in terms of per-sample reconstruction error for KDDCUP).

In other words, the compression network and estimation network mutually boost each others' performance during end-to-end training: the regularization introduced by the estimation network helps the deep autoencoder escape from less attractive local optima for better compression, while the compression network feeds more meaningful low-dimensional representations to estimation network for robust density estimation.

In Appendix C, for readers of interest, we show the visualization of the latent representation learned by DSEBM.In summary, our experimental results show that DAGMM suggests a promising direction for density estimation and anomaly detection, where one can combine the forces of dimensionality reduction and density estimation by end-to-end training.

In Appendix E, we provide another case study to discuss which kind of samples benefit more from joint training in DAGMM for readers of interest.

In this paper, we propose the Deep Autoencoding Gaussian Mixture Model (DAGMM) for unsupervised anomaly detection.

DAGMM consists of two major components: compression network and estimation network, where the compression network projects samples into a low-dimensional space that preserves the key information for anomaly detection, and the estimation network evaluates sample energy in the low-dimensional space under the framework of Gaussian Mixture Modeling.

DAGMM is friendly to end-to-end training: (1) the estimation network predicts sample mixture membership so that the parameters in GMM can be estimated without alternating procedures; and (2) the regularization introduced by the estimation network helps the compression network escape from less attractive local optima and achieve low reconstruction error by end-to-end training.

Compared with the pre-training strategy, the end-to-end training could be more beneficial for density estimation tasks, as we can have more freedom to adjust dimensionality reduction processes to favor the subsequent density estimation tasks.

In the experimental study, DAGMM demonstrates superior performance over state-of-the-art techniques on public benchmark datasets with up to 14% improvement on the standard F 1 score, and suggests a promising direction for unsupervised anomaly detection on multior high-dimensional data.

A BASELINE CONFIGURATION OC-SVM.

Unlike other baselines that only need decision thresholds in the testing phase, OC-SVM needs parameter ?? be set in the training phase.

Although ?? intuitively means anomaly ratio in training data, it is non-trivial to set a reasonable ?? in the case where training data are all normal samples and anomaly ratio in the testing phase could be arbitrary.

In this study, we simply perform exhaustive search to find the optimal ?? that renders the highest F 1 score on individual datasets.

In particular, ?? is set to be 0.1, 0.02, 0.04, and 0.1 for KDDCUP, Thyroid, Arrhythmia, and KDDCUP-Rev, respectively.

DSEBM.

We use the network structure for the encoding in DAGMM as guidelines to set up DSEBM instances.

For KDDCUP and KDDCUP-Rev, it is configured as FC FORMULA0

In this section, we detail the discussion on reconstruction features.

We realize the importance of reconstruction features from our investigation on a private network security dataset.

In this dataset, normal samples are normal network flows, and anomalies are network flows with spoofing attack.

As it is difficult to analyze the samples from their original space with 20 dimensions, we utilize deep autoencoders to perform dimension reduction.

In this case, we are a little bit ambitious, and reduce dimensions from 20 to 1.

In the reduced 1-dimensional space, for some of the anomalies, we are able to easily separate them from normal samples.

However, for the rest, their latent representations are quite similar to the representations of the normal samples.

Meanwhile, in the original space, they are actually quite different from the normal ones.

Inspired by this observation, we investigate their L 2 reconstruction error, and obtain the plot shown in Figure 1 .

In Figure 1 , the red points in the top-right corner are the anomalies sharing similar representations with the normal samples in the reduced space.

With the additional view from reconstruction error, it becomes easier to separate these anomalies from the normal samples.

In our study, this concrete example motivates us to include reconstruction features into DAGMM.What are the guidelines for reconstruction feature selection?

In practice, one can select reconstruction features by the following rules.

First, for an error metric used to derive a reconstruction feature, its analytical form should be continuous and differentiable.

Second, the output of an error metric should be in a range of relatively small values for the ease of training the estimation network in DAGMM.

In the experiment of this paper, we select cosine similarity and relative Euclidean distance based on these two rules.

For cosine similarity, it is continuous and differentiable, and the range of its output is [???1, 1].

For relative Euclidean distance, it is also continuous and differentiable.

Theoretically, the range of its output is [0, +???).

On the datasets considered in the experiment, we observe that its output is usually a small positive value; therefore, we include this metric as one of the reconstruction features.

In sum, as long as an error metric meets the above two rules, it could serve as a candidate metric to derive a reconstruction feature for DAGMM.

In this section, we perform a case study to investigate what kind of samples benefit more from the joint training applied in DAGMM over decoupled training.

In the evaluation, we employ PAE-GMM as a representative for the methods that leverage decoupled training, and the following results are generated from one run on the KDDCUP dataset.

FIG5 and 6d, we observe that the anomalies of low cosine similarity and high relative Euclidean distance could be the easy ones that are captured by both techniques.

For the difficult ones shown in FIG5 and 6f, we observe that they usually have medium level of relative Euclidean distance (in the range of [1.0, 1.2] for both cases) with larger than 0.6 cosine similarity.

For such anomalous samples, the model learned by PAE-GMM has difficult time to separate them from the normal samples.

In addition, we also observe that the model learned by DAGMM tends to assign lower cosine similarity to such anomalies than PAE-GMM does, which also makes it easier to differentiate the anomalies from the normal samples.

As shown in Equation FORMULA5 , the objective function of DAGMM includes three components: the loss function from deep autoencoder, the energy function from estimation network, and the penalty function for covariance matrices.

The coefficient ratio among the three components can be characterized as 1 : ?? 1 : ?? 2 .

In terms of ?? 1 , a large value could make the loss function of deep autoencoder play little role in optimization so that we are unable to obtain a good reduced representation for input samples, while a small value could lead to ineffective estimation network so that GMM is not well trained.

For ?? 2 of a large value, DAGMM tends to find GMM with large covariance, which is less desirable as many samples will have high energy as rare events.

For ?? 2 of a small value, the regularization may not be strong enough to counter the singularity effect.

In our exploration, we find the ratio 1 : 0.1 : 0.005 consistently delivers expected results across all the datasets in the experiment.

To investigate the sensitivity of this ratio, we vary its base and see how different bases affect anomaly detection accuracy.

For example, when the base is set to 2, ?? 1 and ?? 2 are adjusted to 0.2 and 0.01, respectively.

TAB6 shows the average precision, recall, and F 1 score after 20 runs of DAGMM on the KDDCUP dataset.

As we vary the base from 1 to 9 with step 2, DAGMM performs in a consistent way, and ?? 1 , ?? 2 are not sensitive to the changes on the base.

<|TLDR|>

@highlight

An end-to-end trained deep neural network that leverages Gaussian Mixture Modeling to perform density estimation and unsupervised anomaly detection in a low-dimensional space learned by deep autoencoder.

@highlight

The paper presents a joint deep learning framework for dimension reduction-clustering, leads to competitive anomaly detection.

@highlight

A new technique for anomaly detection where the dimension reduction and density estimation steps are jointly optimized.