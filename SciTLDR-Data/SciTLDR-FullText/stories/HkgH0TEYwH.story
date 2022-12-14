Deep approaches to anomaly detection have recently shown promising results over shallow methods on large and complex datasets.

Typically anomaly detection is treated as an unsupervised learning problem.

In practice however, one may have---in addition to a large set of unlabeled samples---access to a small pool of labeled samples, e.g. a subset verified by some domain expert as being normal or anomalous.

Semi-supervised approaches to anomaly detection aim to utilize such labeled samples, but most proposed methods are limited to merely including labeled normal samples.

Only a few methods take advantage of labeled anomalies, with existing deep approaches being domain-specific.

In this work we present Deep SAD, an end-to-end deep methodology for general semi-supervised anomaly detection.

Using an information-theoretic perspective on anomaly detection, we derive a loss motivated by the idea that the entropy of the latent distribution for normal data should be lower than the entropy of the anomalous distribution.

We demonstrate in extensive experiments on MNIST, Fashion-MNIST, and CIFAR-10, along with other anomaly detection benchmark datasets, that our method is on par or outperforms shallow, hybrid, and deep competitors, yielding appreciable performance improvements even when provided with only little labeled data.

Anomaly detection (AD) (Chandola et al., 2009; Pimentel et al., 2014) is the task of identifying unusual samples in data.

Typically AD methods attempt to learn a "compact" description of the data in an unsupervised manner assuming that most of the samples are normal (i.e., not anomalous).

For example, in one-class classification (Moya et al., 1993; Schölkopf et al., 2001 ) the objective is to find a set of small measure which contains most of the data and samples not contained in that set are deemed anomalous.

Shallow unsupervised AD methods such as the One-Class SVM (Schölkopf et al., 2001; Tax & Duin, 2004) , Kernel Density Estimation (Parzen, 1962; Kim & Scott, 2012; Vandermeulen & Scott, 2013 ), or Isolation Forest (Liu et al., 2008 often require manual feature engineering to be effective on high-dimensional data and are limited in their scalability to large datasets.

These limitations have sparked great interest in developing novel deep approaches to unsupervised AD (Erfani et al., 2016; Zhai et al., 2016; Chen et al., 2017; Ruff et al., 2018; Deecke et al., 2018; Golan & El-Yaniv, 2018; Hendrycks et al., 2019) .

Unlike the standard unsupervised AD setting, in many real-world applications one may also have access to some verified (i.e., labeled) normal or anomalous samples in addition to the unlabeled data.

Such samples could be hand labeled by a domain expert, for instance.

This leads to a semisupervised AD problem: Given n (mostly normal but possibly containing some anomalous contamination) unlabeled samples x 1 , . . .

, x n and m labeled samples (x 1 ,ỹ 1 ), . . . , (x m ,ỹ m ), wherẽ y = +1 andỹ = −1 denote normal and anomalous samples respectively, the task is to learn a model that compactly characterizes the "normal class."

The term semi-supervised anomaly detection has been used to describe two different AD settings.

Most existing "semi-supervised" AD methods, both shallow (Muñoz-Marí et al., 2010; Blanchard et al., 2010; Chandola et al., 2009 ) and deep Akcay et al., 2018; Chalapathy & Chawla, 2019) , only incorporate the use of labeled normal samples but not labeled anomalies, i.e. they are more precisely instances of Learning from Positive (i.e., normal) and Unlabeled Examples (LPUE) (Zhang & Zuo, 2008) .

A few works (Wang et al., 2005; Liu & Zheng, 2006; Gör-nitz et al., 2013) have investigated the general semi-supervised AD setting where one also utilizes labeled anomalies, however existing deep approaches are domain or data-type specific (Ergen et al., 2017; Kiran et al., 2018; Min et al., 2018) .

Research on deep semi-supervised learning has almost exclusively focused on classification as the downstream task (Kingma et al., 2014; Rasmus et al., 2015; Odena, 2016; Dai et al., 2017; Oliver et al., 2018) .

Such semi-supervised classifiers typically assume that similar points are likely to be of the same class, this is known as the cluster assumption (Zhu, 2005; Chapelle et al., 2009 ).

This assumption, however, only holds for the "normal class" in AD, but is crucially invalid for the "anomaly class" since anomalies are not necessarily similar to one another.

Instead, semi-supervised AD approaches must find a compact description of the normal class while also correctly discriminating the labeled anomalies (Görnitz et al., 2013) .

Figure 1 illustrates the differences between various learning paradigms applied to AD on a toy example.

We introduce Deep SAD (Deep Semi-supervised Anomaly Detection) in this work, an end-to-end deep method for general semi-supervised AD.

Our main contributions are the following:

• We introduce an information-theoretic framework for deep AD based on the Infomax principle (Linsker, 1988) .

• Using this framework, we derive Deep SAD as a generalization of the unsupervised Deep SVDD method (Ruff et al., 2018) to the general semi-supervised setting.

• We conduct extensive experiments in which we establish experimental scenarios for the general semi-supervised AD problem where we also introduce novel baselines.

The study of the theoretical foundations of deep learning is an active and ongoing research effort (Montavon et al., 2011; Tishby & Zaslavsky, 2015; Cohen et al., 2016; Eldan & Shamir, 2016; Neyshabur et al., 2017; Raghu et al., 2017; Zhang et al., 2017; Achille & Soatto, 2018; Arora et al., 2018; Belkin et al., 2018; Wiatowski & Bölcskei, 2018; Lapuschkin et al., 2019) .

One important line of research that has emerged is rooted in information theory (Shannon, 1948) .

In the supervised classification setting where one has input variable X, latent variable Z (e.g., the final layer of a deep network), and output variable Y (i.e., the label), the well-known Information Bottleneck principle (Tishby et al., 1999; Tishby & Zaslavsky, 2015; Shwartz-Ziv & Tishby, 2017; Alemi et al., 2017; Saxe et al., 2018) provides an explanation for representation learning as the trade-off between finding a minimal compression Z of the input X while retaining the informativeness of Z for predicting the label Y .

Put formally, supervised deep learning seeks to minimize the mutual information I(X; Z) between the input X and the latent representation Z while maximizing the mutual information I(Z; Y ) between Z and the classification task Y , i.e. min p(z|x)

where p(z|x) is modeled by a deep network and the hyperparameter α > 0 controls the trade-off between compression (i.e., complexity) and classification accuracy.

For unsupervised deep learning, due to the absence of labels Y and thus the lack of a clear task, other information-theoretic learning principles have been formulated.

Of these, the Infomax principle (Linsker, 1988; Bell & Sejnowski, 1995; Hjelm et al., 2019 ) is one of the most prevalent and widely used principles.

In contrast to (1), the objective of Infomax is to maximize the mutual information I(X; Z) between the data X and its latent representation Z: max

This is typically done under some additional constraint or regularization R(Z) on the representation Z with hyperparameter β > 0 to obtain statistical properties desired for some specific downstream task.

Examples where the Infomax principle has been applied include tasks such as independent component analysis (Bell & Sejnowski, 1995) , clustering (Slonim et al., 2005; Ji et al., 2018) , generative modeling (Chen et al., 2016; Hoffman & Johnson, 2016; Zhao et al., 2017; Alemi et al., 2018) , and unsupervised representation learning in general (Hjelm et al., 2019) .

We observe that the Infomax principle has also been applied in previous deep representations for AD.

Most notably autoencoders (Rumelhart et al., 1986; Hinton & Salakhutdinov, 2006) , which are the predominant approach to deep AD (Hawkins et al., 2002; Sakurada & Yairi, 2014; Andrews et al., 2016; Erfani et al., 2016; Zhai et al., 2016; Chen et al., 2017; Chalapathy & Chawla, 2019) , can be understood as implicitly maximizing the mutual information I(X; Z) via the reconstruction objective (Vincent et al., 2008) under some regularization of the latent code Z. Choices for regularization include sparsity (Makhzani & Frey, 2014) , the distance to some latent prior distribution, e.g. measured via the KL divergence (Kingma & Welling, 2013; Rezende et al., 2014) , an adversarial loss (Makhzani et al., 2015) , or simply a bottleneck in dimensionality.

Such restrictions for AD share the idea that the latent representation of the normal data should be in some sense "compact."

As illustrated in Figure 1 , a supervised (or semi-supervised) classification approach to AD only learns to recognize anomalies similar to those seen during training, due to the class cluster assumption (Chapelle et al., 2009 ).

However, anything not normal is by definition an anomaly and thus anomalies do not have to be similar.

This makes supervised (or semi-supervised) classification learning principles such as (1) ill-defined for AD.

We instead build upon principle (2) to derive a deep method for general semi-supervised AD, where we include the label information Y through a novel representation learning regularization objective R(Z) = R(Z; Y ) that is based on entropy.

In the following, we introduce Deep SAD, a deep method for general semi-supervised AD.

To formulate our objective, we first show that the unsupervised Deep SVDD method (Ruff et al., 2018) can be interpreted in terms of an entropy minimization objective on the latent representation.

We then generalize the method to the semi-supervised AD setting.

The objective of Deep SVDD is to train the neural network φ to learn a transformation that minimizes the volume of a data-enclosing hypersphere in output space Z centered on a predetermined point c. Given n (unlabeled) training samples x 1 , . . .

, x n ∈ X , the One-Class Deep SVDD objective is

Penalizing the mean squared distance of the mapped samples to the hypersphere center c forces the network to extract those common factors of variation which are most stable within the dataset.

As a consequence normal data points tend to get mapped near the hypersphere center, whereas anomalies are mapped further away (Ruff et al., 2018) .

The second term is a standard weight decay regularizer.

Deep SVDD is optimized via SGD using backpropagation.

For initialization, Ruff et al. (2018) first pre-train an autoencoder and then initialize the weights W of the network φ with the converged weights of the encoder.

After initialization, the hypersphere center c is set as the mean of the network outputs obtained from an initial forward pass of the data.

Once the network is trained, the anomaly score for a test point x is given by the distance from φ(x; W) to the center of the hypersphere:

We now show that Deep SVDD may not only be interpreted in geometric terms as minimum volume estimation (Scott & Nowak, 2006) , but also in probabilistic terms as entropy minimization over the latent distribution.

For a latent random variable Z with covariance Σ, pdf p(z), and support Z ⊆ R d , we have the following bound on entropy

which holds with equality iff Z is jointly Gaussian (Cover & Thomas, 2012) .

Assuming the latent distribution Z follows an isotropic Gaussian, Z ∼ N (µ, σ 2 I) with σ > 0, we get

i.e. for a fixed dimensionality d, the entropy of Z is proportional to its log-variance.

Now observe that the Deep SVDD objective (3) (disregarding weight decay regularization) is equivalent to minimizing the empirical variance and thus minimizes an upper bound on the entropy of a latent Gaussian.

Since the Deep SVDD network is pre-trained on an autoencoding objective that implicitly maximizes the mutual information I(X; Z) (Vincent et al., 2008) , we may interpret Deep SVDD as following the Infomax principle (2) with the additional "compactness" objective that the latent distribution should have minimal entropy.

We are happy to now introduce Deep SAD.

Assume that, in addition to the n unlabeled samples x 1 , . . .

, x n ∈ X with X ⊆ R D , we also have access to m labeled samples (x 1 ,ỹ 1 ), . . .

, (x m ,ỹ m ) ∈ X × Y with Y = {−1, +1} whereỹ = +1 denotes known normal samples andỹ = −1 known anomalies.

Following our insights above, we formulate the Deep SAD objective under the assumption that the latent distribution of the normal data, Z + = Z|{Y =+1}, should have low entropy, whereas the latent distribution of anomalies, Z − = Z|{Y =−1}, should have high entropy.

We argue that such a model better captures the nature of anomalies, which can be thought of as being generated from an infinite mixture of distributions that are different from the normal data distribution, indubitably a distribution that has high entropy.

Our objective notably does not impose any cluster assumption on the anomaly-generating distribution X|{Y =−1} as is typically made in supervised or semisupervised classification approaches (Zhu, 2005; Chapelle et al., 2009) .

We can express this idea in terms of principle (2) with an entropy regularization objective on the latent distribution:

Based on the connection between Deep SVDD and entropy minimization we have shown above, we now define our Deep SAD objective as follows:

We employ the same loss term as Deep SVDD for the unlabeled data in our Deep SAD objective and thus recover Deep SVDD (3) as the special case when there is no labeled training data available (m = 0).

In doing this we also incorporate the assumption that most of the unlabeled data is normal.

For the labeled data, we introduce a new loss term that is weighted via the hyperparameter η > 0 which controls the balance between the labeled and the unlabeled term.

Setting η > 1 puts more emphasis on the labeled data whereas η < 1 emphasizes the unlabeled data.

For the labeled normal samples (ỹ = +1), we also impose a quadratic loss on the distances of the mapped points to the center c, thus intending to overall learn a latent distribution with low entropy for the normal data.

Again, one might consider η > 1 to emphasize labeled normal over unlabeled samples.

For the labeled anomalies (ỹ = −1) in contrast, we penalize the inverse of the distances such that anomalies must be mapped further away from the center.

That is, we penalize low variance and thus the network must attempt to map known anomalies to a heavy-tailed distribution that has high entropy.

To maximize the mutual information I(X; Z) in (7), we also rely on autoencoder pretraining (Vincent et al., 2008; Ruff et al., 2018) .

We found that simply setting η = 1 yields a consistent and substantial performance improvement.

A sensitivity analysis on η is in Section 4.3.

In addition to the inverse squared norm loss we experimented with several other losses including the negative squared norm loss, negative robust losses, and the hinge loss.

The negative squared norm loss, which is unbounded from below, resulted in an ill-posed optimization problem and caused optimization to diverge.

Negative robust losses, such as the Hampel loss, introduce one or more scale parameters which are difficult to select or optimize in conjunction with the changing representation learned by the network.

Like Ruff et al. (2018), we observed that the hinge loss was difficult to optimize and resulted in poorer performance.

The inverse squared norm loss instead is bounded from below and smooth, which are crucial properties for losses used in deep learning (Goodfellow et al., 2016) , and ultimately performed the best while remaining conceptually simple.

We define the Deep SAD anomaly score again by the distance of the mapped point to the center c as given in Eq. (4) and optimize our Deep SAD objective (8) via SGD using backpropagation.

We provide a summary of the Deep SAD optimization procedure and further details in Appendix C.

We evaluate Deep SAD on MNIST, Fashion-MNIST, and CIFAR-10 as well as on classic AD benchmark datasets.

We compare to shallow, hybrid, as well as deep unsupervised, semi-supervised and supervised competitors.

We refer to other recent works (Ruff et al., 2018; Golan & El-Yaniv, 2018; Hendrycks et al., 2019) for further comparisons between unsupervised deep AD methods.

We consider the OC-SVM (Schölkopf et al., 2001 ) and SVDD (Tax & Duin, 2004) with Gaussian kernel (which in this case are equivalent), Isolation Forest (Liu et al., 2008), and KDE (Parzen, 1962) for shallow unsupervised baselines.

For deep unsupervised competitors, we consider wellestablished (convolutional) autoencoders and the state-of-the-art unsupervised Deep SVDD method (Ruff et al., 2018) .

To avoid confusion, we note again that some literature Chalapathy & Chawla, 2019) refer to the methods above as being "semi-supervised" if they are trained on only labeled normal samples.

For general semi-supervised AD approaches that also take advantage of labeled anomalies, we consider the state-of-the-art shallow SSAD method (Görnitz et al., 2013) with Gaussian kernel.

As mentioned earlier, there are no deep competitors for general semisupervised AD that are applicable to general data types.

To get a comprehensive comparison we therefore introduce a novel hybrid SSAD baseline that applies SSAD to the latent codes of autoencoder models.

Such hybrid methods have demonstrated solid performance improvements over their raw feature counterparts on high-dimensional data (Erfani et al., 2016; Nicolau et al., 2016) .

We also include such hybrid variants for all unsupervised shallow competitors.

To also compare to a deep semi-supervised learning method that targets classification as the downstream task, we add the well-known Semi-Supervised Deep Generative Model (SS-DGM) (Kingma et al., 2014) where we use the latent class probability estimate (normal vs. anomalous) as the anomaly score.

To complete the full learning spectrum, we also include a fully supervised deep classifier trained on the binary cross-entropy loss.

1 To ensure numerical stability, we add a machine epsilon (eps ∼ 10 −6 ) to the denominator of the inverse.

2 Our code is available at: https://tinyurl.com/y6rwhn5r (public repository in the final version)

In our experiments we deliberately grant the shallow and hybrid methods an unfair advantage by selecting their hyperparameters to maximize AUC on a subset (10%) of the test set to minimize hyperparameter selection issues.

To control for architectural effects between the deep methods, we always use the same (LeNet-type) deep networks.

Full details on network architectures and hyperparameter selection can be found in Appendices D and E. Due to space constraints, in the main text we only report results for methods which showed competitive performance and defer results for the underperforming methods in Appendix F.

Semi-supervised anomaly detection setup MNIST, Fashion-MNIST, and CIFAR-10 all have ten classes from which we derive ten AD setups on each dataset following previous works (Ruff et al., 2018; Chalapathy et al., 2018; Golan & El-Yaniv, 2018) .

In every setup, we set one of the ten classes to be the normal class and let the remaining nine classes represent anomalies.

We use the original training data of the respective normal class as the unlabeled part of our training set.

Thus we start with a clean AD setting that fulfills the assumption that most (in this case all) unlabeled samples are normal.

The training data of the respective nine anomaly classes then forms the data pool from which we draw anomalies for training to create different scenarios.

We compute the commonly used AUC measure on the original respective test sets using ground truth labels to make a quantitative comparison, i.e.ỹ = +1 for the normal class andỹ = −1 for the respective nine anomaly classes.

We rescale pixels to [0, 1] via min-max feature scaling as the only data pre-processing step.

Experimental scenarios We examine three scenarios in which we vary the following three experimental parameters: (i) the ratio of labeled training data γ l , (ii) the ratio of pollution γ p in the unlabeled training data with (unknown) anomalies, and (iii) the number of anomaly classes k l included in the labeled training data.

(i) Adding labeled anomalies In this scenario, we investigate the effect that including labeled anomalies during training has on detection performance to see the benefit of a general semisupervised AD approach over other paradigms.

To do this we increase the ratio of labeled training data γ l = m/(n+m) by adding more and more known anomaliesx 1 , . . .

,x m withỹ j = −1 to the training set.

The labeled anomalies are sampled from one of the nine anomaly classes (k l = 1).

For testing, we then consider all nine remaining classes as anomalies, i.e. there are eight novel classes at testing time.

We do this to simulate the unpredictable nature of anomalies.

For the unlabeled part of the training set, we keep the training data of the respective normal class, which we leave unpolluted in this experimental setup, i.e. γ p = 0.

We iterate this training set generation process per AD setup always over all the nine respective anomaly classes and report the average results over the ten AD setups × nine anomaly classes, i.e. over 90 experiments per labeled ratio γ l .

(ii) Polluted training data Here we investigate the robustness of the different methods to an increasing pollution ratio γ p of the training set with unlabeled anomalies.

To do so we pollute the unlabeled part of the training set with anomalies drawn from all nine respective anomaly classes in each AD setup.

We fix the ratio of labeled training samples at γ l = 0.05 where we again draw samples only from k l = 1 anomaly class in this scenario.

We repeat this training set generation process per AD setup over all the nine respective anomaly classes and report the average results over the resulting 90 experiments per pollution ratio γ p .

We hypothesize that learning from labeled anomalies in a semi-supervised AD approach alleviates the negative impact pollution has on detection performance since similar unknown anomalies in the unlabeled data might be detected.

(iii) Number of known anomaly classes In the last scenario, we compare the detection performance at various numbers of known anomaly classes.

In scenarios (i) and (ii), we always sample labeled anomalies only from one out of the nine anomaly classes (k l = 1).

In this scenario, we now increase the number of anomaly classes k l included in the labeled part of the training set.

Since we have a limited number of anomaly classes (nine) in each AD setup, we expect the supervised classifier to catch up at some point.

We fix the overall ratio of labeled training examples again at γ l = 0.05 and consider a pollution ratio of γ p = 0.1 for the unlabeled training data in this scenario.

We repeat this training set generation process for ten seeds in each of the ten AD setups and report the average results over the resulting 100 experiments per number k l .

For each seed, the k l classes are drawn uniformly at random from the nine respective anomaly classes.

Results The results of scenarios (i)-(iii) are shown in Figures 2-4 .

In addition to the avg.

AUC with st.

dev., we report the outcome of Wilcoxon signed-rank tests (Wilcoxon, 1945) applied to the first and second best performing method to indicate statistically significant (α = 0.05) differences in performance.

Figure 2 demonstrates the benefit of our semi-supervised approach to AD especially on the most complex CIFAR-10 dataset, where Deep SAD performs best.

Figure 2 moreover confirms that a supervised classification approach is vulnerable to novel anomalies at testing time when only little labeled training data is available.

In comparison, Deep SAD generalizes to novel anomalies while also taking advantage of the labeled examples.

Note that our novel hybrid SSAD baseline also performs well.

Figure 3 shows that the detection performance of all methods decreases with increasing data pollution.

Deep SAD proves to be most robust again especially on CIFAR-10.

Finally, Figure 4 shows that the more diverse the labeled anomalies in the training set, the better the detection performance becomes.

We can again see that the supervised method is very sensitive to the number of anomaly classes but catches up at some point as suspected.

This does not occur with CIFAR-10, however, where γ l = 0.05 labeled training samples seems to be insufficient for classification.

Overall, we see that Deep SAD is particularly beneficial on the more complex data.

We run Deep SAD experiments on the ten AD setups described above on each dataset for η ∈ {10 −2 , . . .

, 10 2 } to analyze the sensitivity of Deep SAD with respect to the hyperparameter η > 0.

In this analysis, we set the experimental parameters to their default, γ l = 0.05, γ p = 0.1, and k l = 1, and again iterate over all nine anomaly classes in every AD setup.

The results shown in Figure 5 suggest that Deep SAD is fairly robust against changes of the hyperparameter η.

In addition, we run experiments under the same experimental settings while varying the dimension d ∈ {2 4 , . . .

, 2 9 } of the output space

to infer the sensitivity of Deep SAD with respect to the representation dimensionality, where we keep η = 1.

The results are given in Figure 6 in Appendix A. There we also compare to our hybrid SSAD baseline, which was the strongest competitor.

Interestingly we observe that detection performance increases with dimension d, converging to an upper bound in performance.

This suggests that one would want to set d large enough to have sufficiently high mutual information I(X; Z) before compressing to a compact characterization.

In a final experiment, we also examine the detection performance of the various methods on some well-established AD benchmark datasets (Rayana, 2016) .

We run these experiments to evaluate the deep versus the shallow approaches on non-image datasets that are rarely considered in deep AD literature.

Here we observe that the shallow kernel methods seem to have a slight edge on the relatively small, low-dimensional benchmarks.

Nonetheless, Deep SAD proves competitive and the small differences observed might be explained by the advantage we grant the shallow methods in their hyperparameter selection.

We give the full details and results in Appendix B.

Our results and other recent works (Ruff et al., 2018; Golan & El-Yaniv, 2018; Hendrycks et al., 2019) overall demonstrate that deep methods are especially superior on complex data with hierarchical structure.

Unlike other deep approaches (Ergen et al., 2017; Kiran et al., 2018; Min et al., 2018; Deecke et al., 2018; Golan & El-Yaniv, 2018) , however, our Deep SAD method is not domain or data-type specific.

Due to its good performance using both deep and shallow networks we expect Deep SAD to extend well to other data types.

We introduced Deep SAD, a deep method for general semi-supervised anomaly detection.

Our method is based on an information-theoretic framework we formulated for deep anomaly detection based on the Infomax principle.

This framework can form the basis for rigorous theoretical analyses, e.g. studying the problem under the rate-distortion curve (Alemi et al., 2018) and new methods in the future.

Our results suggest that general semi-supervised anomaly detection should always be preferred whenever some labeled information on both normal samples or anomalies is available. ) performing methods in the experimental scenarios (i)-(iii) on the most complex CIFAR-10 dataset.

If most points fall above the identity line, this is a very strong indication that the best method indeed significantly outperforms the second best, which often is the case for our Deep SAD method.

In this experiment, we examine the detection performance on some well-established AD benchmark datasets (Rayana, 2016) listed in Table 1 .

We do this to evaluate the deep against the shallow approaches also on non-image, tabular datasets that are rarely considered in the deep AD literature.

For the evaluation, we consider random train-to-test set splits of 60:40 while maintaining the original proportion of anomalies in each set.

We then run experiments for 10 seeds with γ l = 0.01 and γ p = 0, i.e. 1% of the training set are labeled anomalies and the unlabeled training data is unpolluted.

Since there are no specific different anomaly classes in these datasets, we have k l = 1.

We standardize features to have zero mean and unit variance as the only pre-processing step.

Table 2 shows the results of the competitive methods.

We observe that the shallow kernel methods seem to perform slightly better on the rather small, low-dimensional benchmarks.

Deep SAD proves competitive though and the small differences might be explained by the strong advantage we grant the shallow methods in the selection of their hyperparameters.

We provide the complete table with the results from all methods in Appendix F for each mini-batch do

end for 7: end for Using SGD allows Deep SAD to scale with large datasets as the computational complexity scales linearly in the number of training batches and computations in each batch can be parallelized (e.g., by training on GPUs).

Moreover, Deep SAD has low memory complexity as a trained model is fully characterized by the final network parameters W * and no data must be saved or referenced for prediction.

Instead, the prediction only requires a forward pass on the network which usually is just a concatenation of simple functions.

This enables fast predictions for Deep SAD.

Initialization of the network weights W We establish an autoencoder pre-training routine for initialization.

That is, we first train an autoencoder that has an encoder with the same architecture as network φ on the reconstruction loss (mean squared error or cross-entropy).

After training, we then initialize W with the converged parameters of the encoder.

Note that this is in line with the Infomax principle (2) for unsupervised representation learning (Vincent et al., 2008) .

Initialization of the center c After initializing the network weights W, we fix the hypersphere center c as the mean of the network representations that we obtain from an initial forward pass on the data (excluding labeled anomalies).

We found SGD convergence to be smoother and faster by fixing center c in the neighborhood of the initial data representations as also observed by Ruff et al. (2018) .

If sufficiently many labeled normal examples are available, using only those examples for a mean initialization would be another strategy to minimize possible distortions from polluted unlabeled training data.

Adding center c as a free optimization variable would allow a trivial "hypersphere collapse" solution for the fully unlabeled setting, i.e. for unsupervised Deep SVDD.

Preventing a hypersphere collapse A "hypersphere collapse" describes the trivial solution that neural network φ converges to the constant function φ ≡ c, i.e. the hypersphere collapses to a single point.

Ruff et al. (2018) demonstrate theoretical network properties that prevent such a collapse which we adopt for Deep SAD.

Most importantly, network φ must have no bias terms and no bounded activation functions.

We refer to Ruff et al. (2018) for further details.

If there are sufficiently many labeled anomalies available for training, however, hypersphere collapse is not a problem for Deep SAD due to the opposing labeled and unlabeled objectives.

We employ LeNet-type convolutional neural networks (CNNs) on MNIST, Fashion-MNIST, and CIFAR-10, where each convolutional module consists of a convolutional layer followed by leaky ReLU activations with leakiness α = 0.1 and (2×2)-max-pooling.

On MNIST, we employ a CNN with two modules, 8×(5×5)-filters followed by 4×(5×5)-filters, and a final dense layer of 32 units.

On Fashion-MNIST, we employ a CNN also with two modules, 16×(5×5)-filters and 32×(5×5)-filters, followed by two dense layers of 64 and 32 units respectively.

On CIFAR-10, we employ a CNN with three modules, 32×(5×5)-filters, 64×(5×5)-filters, and 128×(5×5)-filters, followed by a final dense layer of 128 units.

On the classic AD benchmark datasets, we employ standard MLP feed-forward architectures.

On arrhythmia, a 3-layer MLP with 128-64-32 units.

On cardio, satellite, satimage-2, and shuttle a 3-layer MLP with 32-16-8 units.

On thyroid a 3-layer MLP with 32-16-4 units.

For the (convolutional) autoencoders, we always employ the above architectures for the encoder networks and then construct the decoder networks symmetrically, where we replace max-pooling with simple upsampling and convolutions with deconvolutions.

OC-SVM/SVDD The OC-SVM and SVDD are equivalent for the Gaussian/RBF kernel we employ.

As mentioned in the main paper, we deliberately grant the OC-SVM/SVDD an unfair advantage by selecting its hyperparameters to maximize AUC on a subset (10%) of the test set to establish a strong baseline.

To do this, we consider the RBF scale parameter γ ∈ {2 −7 , 2 −6 , . . .

2 2 } and select the best performing one.

Moreover, we always repeat this over ν-parameter ν ∈ {0.01, 0.05, 0.1, 0.2, 0.5} and then report the best final result.

We set the number of trees to t = 100 and the sub-sampling size to ψ = 256, as recommended in the original work (Liu et al., 2008) .

Kernel Density Estimator (KDE) We select the bandwidth h of the Gaussian kernel from h ∈ {2 0.5 , 2 1 , . . .

, 2 5 } via 5-fold cross-validation using the log-likelihood score following (Ruff et al., 2018) .

SSAD We also deliberately grant the state-of-the-art semi-supervised AD kernel method SSAD the unfair advantage of selecting its hyperparameters optimally to maximize AUC on a subset (10%) of the test set.

To do this, we again select the scale parameter γ of the RBF kernel we use from γ ∈ {2 −7 , 2 −6 , . . .

2 2 } and select the best performing one.

Otherwise we set the hyperparameters as recommend by the original authors to κ = 1, κ = 1, η u = 1, and η l = 1 (Görnitz et al., 2013) .

(Convolutional) Autoencoder ((C)AE) To create the (convolutional) autoencoders, we symmetrically construct the decoders w.r.t.

the architectures reported in Appenidx D, which make up the encoder parts of the autoencoders.

Here, we replace max-pooling with simple upsampling and convolutions with deconvolutions.

We train the autoencoders on the MSE reconstruction loss that also serves as the anomaly score.

Hybrid Variants To establish hybrid methods, we apply the OC-SVM, IF, KDE, and SSAD as outlined above to the resulting bottleneck representations given by the respective converged autoencoders.

Unsupervised Deep SVDD We consider both variants, Soft-Boundary Deep SVDD and One-Class Deep SVDD as unsupervised baselines and always report the better performance as the unsupervised result.

For Soft-Boundary Deep SVDD, we optimally solve for the radius R on every mini-batch and run experiments for ν ∈ {0.01, 0.1}. We set the weight decay hyperparameter to λ = 10 −6

.

For Deep SVDD, we always remove all the bias terms from a network to prevent a hypersphere collapse as recommended by the authors in the original work (Ruff et al., 2018) .

Deep SAD We set λ = 10

and equally weight the unlabeled and labeled examples by setting η = 1 if not reported otherwise.

We consider both the M2 and M1+M2 model and always report the better performing result.

Otherwise we follow the settings as recommended in the original work (Kingma et al., 2014) .

Note that we use the latent class probability estimate (normal vs. anomalous) of semi-supervised DGM as a natural choice for the anomaly score, and not the reconstruction error as used for unsupervised autoencoding models such as the (convolutional) autoencoder we consider.

Such deep semi-supervised models designed for classification as the downstream task have no notion of outof-distribution and again implicitly make the cluster assumption (Zhu, 2005; Chapelle et al., 2009) we refer to.

Thus, semi-supervised DGM also suffers from overfitting to previously seen anomalies at training similar to the supervised model which explains its bad AD performance.

Supervised Deep Binary Classifier To interpret AD as a binary classification problem, we rely on the typical assumption that most of the unlabeled training data is normal by assigning y = +1 to all unlabeled examples.

Already labeled normal examples and labeled anomalies retain their assigned labels ofỹ = +1 andỹ = −1 respectively.

We train the supervised classifier on the binary crossentropy loss.

Note that in scenario (i), in particular, the supervised classifier has perfect, unpolluted label information but still fails to generalize as there are novel anomaly classes at testing.

We use the Adam optimizer with recommended default hyperparameters (Kingma & Ba, 2014) and apply Batch Normalization (Ioffe & Szegedy, 2015) in SGD optimization.

For all deep approaches and on all datasets, we employ a two-phase ("searching" and "fine-tuning") learning rate schedule.

In the searching phase we first train with a learning rate ε = 10 −4 for 50 epochs.

In the fine-tuning phase we train with ε = 10 −5 for another 100 epochs.

We always use a batch size of 200.

For the autoencoder, SS-DGM, and the supervised classifier, we initialize the network with uniform Glorot weights (Glorot & Bengio, 2010) .

For Deep SVDD and Deep SAD, we establish an unsupervised pre-training routine via autoencoder as explained in Appendix C, where we set the network φ to be the encoder of the autoencoder that we train beforehand.

The following Tables 3-6 list the complete experimental results of all the methods in all our experiments.

Under review as a conference paper at ICLR 2020 Table 6 : Complete results on classic AD benchmark datasets in the setting with no pollution γ p = 0 and a ratio of labeled anomalies of γ l = 0.01 in the training set.

We report the avg.

AUC with st.

dev. computed over 10 seeds.

@highlight

We introduce Deep SAD, a deep method for general semi-supervised anomaly detection that especially takes advantage of labeled anomalies.

@highlight

A new method to find anomaly data, when some labeled anomalies are given, that applies information theory-derived loss based on normal data usuallly having lower entropy than abnormal data.

@highlight

Proposal for an abnormal detection framework under settings where unlabeled data, labeled positive data, and labeled negative data are available, and proposal to approach semi-supervised AD from an information theoretic perspective.