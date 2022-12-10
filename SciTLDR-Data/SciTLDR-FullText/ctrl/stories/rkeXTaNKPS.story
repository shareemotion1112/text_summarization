Semi-Supervised Learning (SSL) approaches have been an influential framework for the usage of unlabeled data when there is not a sufficient amount of labeled data available over the course of training.

SSL methods based on Convolutional Neural Networks (CNNs) have recently provided successful results on standard benchmark tasks such as image classification.

In this work, we consider the general setting of  SSL problem where the labeled and unlabeled data  come from the same underlying probability distribution.

We  propose a new approach that adopts  an Optimal Transport (OT) technique serving as a metric of similarity between discrete empirical probability measures to  provide pseudo-labels for the unlabeled data, which can then be used in conjunction with the initial labeled data to train the CNN model in an SSL manner.

We have evaluated and compared our proposed method with state-of-the-art SSL algorithms on standard datasets to demonstrate the superiority and effectiveness of our  SSL algorithm.

Recent developments in CNNs have provided promising results for many applications in machine learning and computer vision Krizhevsky et al. (2012) ; Zagoruyko & Komodakis (2016) .

However, the success of CNN models requires a vast amount of well-annotated training data, which is not always feasible to perform manually Krizhevsky et al. (2012) .

There are essentially two different solutions that are usually used to deal with this problem: 1) Transfer Learning (TL) and 2) SemiSupervised Learning (SSL).

In TL methods Tan et al. (2018) , the learning of a new task is improved by transferring knowledge from a related task which has already been learned.

SSL methods Oliver et al. (2018) , however, tend to learn discriminative models that can make use of the information from an input distribution that is given by a large amount of unlabeled data.

To make use of unlabeled data, it is presumed that the underlying distribution of data has some structure.

SSL algorithms make use of at least one of the following structural assumptions: continuity, cluster, or manifold Chapelle et al. (2009) .

In the continuity assumption, data which are close to each other are more likely to belong to the same class.

In the cluster assumption, data tends to form discrete clusters, and data in the same cluster are more likely to share the same label.

In the manifold assumption, data lies approximately on a manifold of much lower dimension than the input space which can be classified by using distances and densities defined on the manifold.

Thus, to define a natural similarity distance or divergence between probability measures on a manifold, it is important to consider the geometrical structures of the metric space in which the manifold exists Bronstein et al. (2017) .

There are two principal directions that model geometrical structures underlying the manifold on which the discrete probability measures lie.

The first direction is based on the principal of invariance, which relies on the criterion that the geometry between probability measures should be invariant under invertible transformations of random variables.

This perspective is the foundation of the theory of information geometry, which operates as a base for the statistical inference Amari (2016) .

The second direction is established by the theory of Optimal Transport (OT), which exploits prior geometric knowledge on the base space in which random variables are valued Villani (2008) .

Computing OT or Wasserstein distance between two random variables equals to achieving a coupling between these two variables that is optimal in the sense that the expectation of the transportation cost between the first and second variables is minimal.

The Wasserstein distance between two probability measures considers the metric properties of the base space on which a structure or a pattern is defined.

However, traditional information-theoretic divergences such as the Hellinger divergence and the Kullback-Leibler (KL) divergence are not able to properly capture the geometry of the base space.

Thus, the Wasserstein distance is useful for the applications where the structure or geometry of the base space plays a significant role Amari & Nagaoka (2007) .

In this work, similar to other SSL methods, we make a structural assumption about the data in which the data are represented by a CNN model.

Inspired by the Wasserstein distance, which exploits properly the geometry of the base space to provide a natural notion of similarity between the discrete empirical measures, we use it to provide pseudo-labels for the unlabeled data to train a CNN model in an SSL fashion.

Specifically, in our SSL method, labeled data belonging to each class is a discrete measure.

Thus, all the labeled data create a measure of measures and similarly, the pool of unlabeled data is also a measure of measures constructed by data belonging to different classes.

Thus, we design a measure of measures OT plan serving as a similarity metric between discrete empirical measures to map the unlabeled measures to the labeled measures based on which, the pseudo-labels for the unlabeled data are inferred.

Our SSL method is based on the role of Wasserstein distances in the hierarchical modeling Nguyen et al. (2016) .

It stems from the fact that the labeled and unlabeled datasets hierarchically create a measure of measures in which each measure is constructed by the data belonging to the same class.

Computing the exact Wasserstein distance, however, is computationally expensive and usually is solved by a linear program (Appendix A and D ).

Cuturi (2013) introduced an interesting method which relaxes the OT problem using the entropy of the solution as a strong convex regularizer.

The entropic regularization provides two main advantageous: 1) The regularized OT problem relies on Sinkhorns algorithm Sinkhorn (1964) that is faster by several orders of magnitude than the exact solution of the linear program.

2) In contrast to exact OT, the regularized OT is a differentiable function of their inputs, even when the OT problem is used for discrete measures.

These advantages have caused that the regularized OT to receive a lot of attention in machine learning applications such as generating data ; Gulrajani et al. (2017) , designing loss function Frogner et al. (2015) , domain adaptation Damodaran et al. (2018) ; Courty et al. (2017) , clustering Cuturi & Doucet (2014) ; Mi et al. (2018) and low-rank approximation Seguy & Cuturi (2015) .

Pseudo-Labeling is a simple approach whereby a model incorporates it's own predictions on unlabeled data to obtain additional information during the training Rosenberg et al. (2005) ; Lee (2013) ; Rasmus et al. (2015) .

The main downside of these methods is that they are unable to correct their own mistakes where predictions of the model on unlabeled data are confident but incorrect.

In such a case, the erroneous data not only can not contribute to the training, but the error of the models is amplified during the training as well.

This effect is aggravated where the domain of the unlabeled data is different from that of labeled data.

Note that pseudo-labeling in Lee (2013) is similar to entropy regularization Pereyra et al. (2017) , in the sense that it forces the model to provide higher confidence predictions for unlabeled data.

However, it differs because it only forces these criteria on data which have a low entropy prediction due to the threshold of confidence.

Consistency Regularization can be considered as a way of using unlabeled data to explore a smooth manifold on which all of the data points are embedded Belkin et al. (2006) .

This simple criterion has provided a set of methods that are currently considered as state of the art for the SSL challenge.

Some of these methods are stochastic perturbations Sajjadi et al. (2016b) , π-model Laine & Aila (2016) , mean teacher Tarvainen & Valpola (2017) , and Virtual Adversarial Training (VAT) Miyato et al. (2018) .

The original idea behind stochastic perturbations and π-model was first introduced in Bachman et al. (2014) and has been referred to as pseudo-ensembles.

The pseudo-ensembles regularization techniques are usually designed such that the prediction of the model ideally should not change significantly if the data given to the model is perturbed; in other words, under realistic perturbations of a data point x (x → x ), output of the model f θ (x) should not change significantly.

This goal is achieved by adding a weighted loss term such as d(f θ (x), f θ (x )) to the total loss of the model f θ (x), where d(., .) is mean squared error or Kullback-Leibler divergence which measures a distance between outputs of the prediction function.

The main problem of pseudo-ensemble methods, including π-model is that they rely on a potentially unstable target prediction, which can immediately change during the training.

To address this problem, two methods, including temporal ensembling Laine & Aila (2016) and mean teacher Tarvainen & Valpola (2017) , were proposed to obtain a more stable target output f θ (x).

Specifically, temporal ensembling uses an exponentially accumulated average of outputs, f θ (x), to make the target output smooth and consistent.

Inspired by this method, mean teacher instead uses a prediction function which is parametrized by an exponentially accumulated average of θ during the training.

Like the π-model, mean teacher adds a mean squared error loss d(f θ (x), f θ (x)) as a regularization term to the total loss function for training the network.

It has been shown that mean teacher outperforms temporal ensembling in practice Tarvainen & Valpola (2017) .

Contrary to stochastic perturbation methods which rely on constructing f θ (x) stochastically, VAT in the first step approximates a small perturbation r to add it to x which significantly changes the prediction of the model f θ (x).

In the next step, a consistency regularization technique is applied to minimize d(f θ (x), f θ (x + r)) with respect to θ which is the parameters of the model.

Entropy Minimization methods use a loss term which is applied on the unlabeled data to force the model f θ (x) to produce confident predictions (i.e., low-entropy) for all of the samples, regardless of what the actual labels are Grandvalet & Bengio (2005) .

For example, by assuming the softmax layer of a CNN has c outputs, the loss term applied on unlabeled data is as follows:

Ideally, this class of methods penalizes the decision boundary that passes near the data points, while they instead force the model to provide a high-confidence prediction Grandvalet & Bengio (2005) .

It has been shown that entropy minimization on its own, can not produce competitive results Sajjadi et al. (2016a) .

However, entropy minimization can be used in conjunction with VAT (i.e., EntMin VAT) to provide state of the art results in which VAT assumes a fixed virtual label prediction in the regularization d(f θ (x), f θ (x + r)) Miyato et al. (2018) .

For any subset θ ⊂ R c , assume that S(θ) represents the space of Borel probability measures on θ.

The Wasserstein space of order k ∈ [1, ∞) of probability measures on θ is defined as follows:

where, ||.|| is the Euclidean distance in R c .

Let Π(P, Q) denote the set of all probability measures on θ × θ which have marginals P and Q; then the k-th Wasserstein distance between P and Q in S k (θ), is defined as follows Villani (2008) :

where x ∼ P, x ∼ Q and k ≥ 1.

Explicitly, W k (P, Q) is the optimal cost of moving mass from P to Q, where the cost of moving mass is proportional to the Euclidean distance raised to the power k.

In Eq.

(1), the Wasserstein between two probability measures was defined.

However, using a recursion of concepts, we can talk about measure of measures in which a cloud of measures (M ) is transported to another cloud of measures (M).

We define a relevant distance metric on this abstract space as follows: let the space of Borel measures on S k (θ) be represented by S k (S k (θ)); this space is also a Polish, complete and separable metric space as S k (θ) is a Polish space (cf. section.

3 in Nguyen et al. (2016) ).

It will be endowed with a Wasserstein metric W k (.) of order k that is induced by a metric

where, Q ∼ M , P ∼ M, and Π(M , M) is the set of all probability measures on S k (θ) × S k (θ) that have marginals M and M. Note that the existence of an optimal solution, π ∈ Π(M , M), is always guaranteed (Appendix E).

In words, W k (M , M) corresponds to the optimal cost of transporting mass from M to M , where the cost of moving unit mass in its space of support, S k (θ), is proportional to the power k of the Wasserstein distance W k (.) in S k (θ).

The goal of our algorithm is to use OT to provide pseudo-labels for the unlabeled data to train a CNN model in an SSL manner.

The basic premise in our algorithm is that the discrepancy between two discrete empirical measures which come from the same underlying distribution is expected to be less than the case where these measures come from two different distributions.

In this work, since we make a structural assumption about the data and assume that the labeled and unlabeled data belonging to the same class come from the same distribution (i.e., general setting in SSL), we leverage OT metric to map similar measures from two measure of measures.

This is because OT exploits well the structure or geometry of the underlying metric space to provide a natural notion of similarity between empirical measures in the metric space.

Here, labeled data belonging to the same class is a measure.

Thus, all the initially labeled data construct a measure of measures and similarly, all the unlabeled data is also a measure of measures constructed by data from different classes.

Thus, we design a measure of measures OT plan to map the unlabeled measures to the similar labeled measures based on which, pseudo-labels for the unlabeled data in each measure are inferred.

The mapping between the labeled and unlabeled measures based on the measure of measures OT is formulated as follows:

Given an image z i ∈ R m×n from the either labeled or unlabeled dataset, the CNN acts as a function f (w, z i ) : R m×n → R c with the parameters w that maps z i to a c-dimensional representation, where c is number of the classes.

Assume that X = {x 1 , ..., x m } and X = {x 1 , ..., x m } are the sets of c-dimensional outputs represented by the CNN for the labeled and unlabeled images, respectively.

Let P i = 1/n i ni j=1 δ xj denote a discrete measure constructed by the labeled data belonging to the i-th class, where δ xj is a Dirac unit mass on x j and n i is number of the data within the i-th class.

Thus, all the labeled data construct a measure of measures M = c i=1 α i δ Pi , where α i = n i /m represents amount of the mass in the measure P i and δ Pi is a Dirac unit mass on the measure P i .

Similarly unlabeled data construct a measure of measures M = c j=1 β j δ Qj in that each measure Q i , is created by the unlabeled data belonging to the unknown but the same class, where β j = n j /m is amount of the mass in the measure Q j and δ Qi is a Dirac unit mass on Q j .

The goal of our SSL method is to use the OT to find a coupling between the measures in M and M that is optimal in the sense that it has a minimal expected transportation cost.

This is because the transportation cost between two empirical measures which come from the same distribution (data from the same class) is expected to be less than the case where these measures come from two different distributions (data from different classes).

Thus, we design an OT cost function defined in Eq. (3) to obtain an optimal coupling between measures in M and M based on which the labels of data in the unlabeled measures are inferred:

where T is the optimal coupling matrix in which T (i, j) indicates amount of the mass that should be moved from Q i to P j to provide an OT plan between M and M. Thus, if highest amount of the mass from Q i is transported to P k (i.e., Q i is mapped to P k ); the data belonging to the measure Q i are annotated by k which is the label of the measure P k .

Variable X is the pairwise similarity matrix between measures within M and M in which X(i, j) = W k (Q i , P j ) which is the Wasserstein distance between two clouds of data points Q i and P j .

Note that the ground metric used for computing W k (Q i , P j ) is the Euclidean distance.

Moreover, T, M denotes the Frobenius dot-product between T and X matrices, and T is transportation polytope defined as follows:

where 1 c is a c-dimensional vector with all elements equal to one.

Finally, E(T ) is entropy of the optimal coupling matrix T which is used for regularizing the OT, and λ is a hyperparameter that balances between two terms in Eq. (3).

The optimal coupling solution for the regularized OT defined in Eq. (3) is obtained by an iterative algorithm relied on Sinkhorn algorithm (Appendix D).

In Sec. 4, we represented the pool of unlabeled data as a measure of measures M = c j=1 β j δ Qj in which each measure is constructed by data that belong to the same class.

However, label of the unlabeled data is unknown to allow us to identify these unlabeled measures.

Moreover, CNN as a classifier trained on a limited amount of the labeled data simply miss-classifies these unlabeled data.

In such a case, there is little option other than to use unsupervised methods, such as the clustering to explore the unlabeled data belonging to the same class.

This is because in structural assumption based on the clustering, it is assumed that the data within the same cluster are more likely to share the same label.

Here, we leverage the Wasserstein metric to explore these unknown measures underlying the unlabeled data.

Specifically, we relate the clustering algorithm to the problem of exploring Wasserstein barycenter of the unlabeled data.

Wasserstein barycenter was initially introduced by Agueh & Carlier (2011) .

Given probability measures R 1 , ..., R l ∈ S 2 (θ) for l ≥ 1, their Wasserstein barycenterR l,µ is defined as follows:

where µ i is the weight associated with R i .

In the case where R 1 , ..., R l are discrete measures with finite number of elements and the weights in µ are uniform, it is shown by Anderes et al. (2016) that the problem of exploring Wasserstein barycenterR l,µ on the space of S 2 (θ) in (4) is recast to search only on O r (θ) denoting as a set of probability measures with at most r support points in θ, where r = l i=1 e i − l + 1 and e i is the number of elements in R i for all 1 ≤ i ≤ l. Moreover, an efficient algorithm for exploring local solutions of the Wasserstein barycenter problem over O r (θ) for some r ≥ 1 has been studied by Cuturi & Doucet (2014) .

Beside, the popular K-means clustering can be considered as solving an optimization problem that comes up in the quantization problem, a simple but very practical connection Pollard (1982) ; Graf & Luschgy (2007) .

The connection is as follows: Given m unlabeled data x 1 , ..., x m ∈ θ.

Suppose that these data are related to at most k clusters where k ≥ 1 is a given number.

The K-means problem finds the set Z containing at most k atoms θ 1 , ..., θ k ∈ θ that minimizes:

is equivalent to explore a discrete measure H including finite number of support points and minimizing the following objective:

.

This problem can also be thought of as a Wasserstein barycenter problem when l = 1.

From this prospective, as denoted by Cuturi & Doucet (2014) , the algorithm for finding the Wasserstein barycenters is an alternative for the popular Loyds algorithm to find local minimum of the K-means objective.

Thus, we adopt the algorithm introduced in Cuturi & Doucet (2014) used for computing the Wasserstein barycenters of empirical probability measures to explore the clusters underlying the unlabeled data (Appendix B).

Our SSL method finally leverages the unlabeled image data annotated by pseudo-labels obtained from the OT in conjunction with the supervision signals of the initial labeled image data to train the CNN classifier.

Thus, we use the generic cross entropy as our discriminative loss function to train the parameters of our CNN as follows: Let X l be all of the labeled training data annotated by true labels Y, and X u be the unlabeled training data annotated by pseudo-labels Y , then the total loss function L(.), used to train our CNN in an SSL fashion is as follows:

where w is parameters of the CNN, and L c (.) denotes cross entropy loss function, and α is a hyperparameter that balances between two losses obtained from the labeled and unlabeled data.

For training, we initially train the CNN using the labeled data as a warm up step, and then use OT to provide pseudo-labels for the unlabeled data to train the CNN in conjunction with the initial labeled data for the next epochs.

Specifically, after training the CNN using the labeled data, in each epoch, we select the same amount of initial labeled data from the pool of unlabeled data and then use OT to compute their pseudo-labels; then, we train the CNN in a mini-batch mode.

Our overall SSL method is described in Algorithm 2 (Appendix C).

For evaluating our SSL technique and comparing it with the other SSL algorithms, we follow the concrete suggestions and criteria which are provided in Oliver et al. (2018) .

Some of these recommendations are as follows: 1) we use a common CNN architecture and training procedure to conduct a comparative analysis, because differences in CNN architecture or even implementation details can influence the results.

2) We report the performance of a fully-supervised case as a baseline because the goal of SSL is to greatly outperform the fully-supervised settings.

3) We change the amount of labeled and unlabeled data when reporting the performance of our SSL algorithm because an ideal SSL method should remain efficient even with the small amount of labeled and additional unlabeled data.

4) We also perform an analysis on realistic small validation sets.

This is because, in real-world applications, the large validation set is instead used as the training, therefore, an SSL algorithm which needs heavy tuning on a per-task or per-model basis to perform well would not be applicable if the validation sets are realistically small (This analysis is done in Appendix F).

For the first criterion, we have used the 'WRN-28-2' model (i.e., ResNet with depth 28 and width 2) Zagoruyko & Komodakis (2016) , including batch normalization Ioffe & Szegedy (2015) and leaky ReLU nonlinearities Maas et al. (2013) .

We conducted our experiments on the widely used CIFAR-10 Krizhevsky & Hinton (2009), and SVHN Netzer et al. (2011) datasets.

Note that in our experiments, we tackle the general SSL challenge where the labeled and unlabeled data come from the same underlying distribution, and a given unlabeled data belongs to one of the classes in the labeled set and therefor, there is no class distribution mismatch.

Moreover, for each of these datasets, we split the training set into two different sets of labeled and unlabeled data.

For training, we use the well-known Adam optimizer Kingma & Ba (2014) with the default hyperparameters values and a learning rate of 3 × 10 −3 in our experiments, and all the experiments have been done on a NVIDIA TITAN X GPU.

The batch size in our experiments is set to 100.

We have not used any form of early stopping; however, we have consistently monitored the performance of the validation set and reported test error at the point of lowest validation error.

The stopping criteria for the Sinkhorn algorithm is either maxIter = 10,000 or tolerance = 10 −8 , where maxIter is the maximum number of iterations and tolerance is a threshold for the integrated stopping criterion based on the marginal differences.

In experiments, we followed the data augmentation and standard data normalization used in Oliver et al. (2018) .

Specifically, for SVHN, we converted pixel intensity values of the images to floating point values in the range of [-1, 1] .

For the data augmentation, we only applied random translation by up to 2 pixels.

We used the standard training and validation split, with 65,932 images for the training set and 7,325 for the validation set.

For CIFAR-10, we applied global contrast normalization.

The data augmentation on CIFAR-10 are random translation by up to 2 pixels, random horizontal flipping, and Gaussian input noise with standard deviation 0.15.

We used the standard training and validation split, with 45,000 images for the training set and 5,000 images for the validation set.

Here, we consider the second criterion for evaluation of our SSL method.

The purpose of SSL is mainly to achieve a better performance when it uses the unlabeled data than the case where using the labeled data alone.

To ensure that our SSL model benefits from the unlabeled data during the training, we report the error rate of the WRN model for both cases where we only use the labeled data (i.e., Supervised in Table.

1), and the case where we leverage the unlabeled data by using the OT technique during the training (i.e., ROT in Table.

1).

Moreover, we have reported the performance of other SSL algorithms in Table.

1 which also leverage the unlabeled data during the training.

All of the compared SSL methods use the common CNN model (i.e., 'WRN-28-2') and training procedure as suggested in the first criterion for the realistic evaluation of SSL models.

The result of all SSL methods reported in Table.

1 is the test error at the point of lowest validation error for tuning their hyperparameters.

For a fair evaluation with other SSL algorithms, we selected 4,000 samples of the training set as the labeled data and the remaining as the unlabeled data for the CIFAR-10 dataset, and we chose 1,000 samples of the training set as the labeled data and the rest as the unlabeled data for the SVHN dataset.

We ran our SSL algorithm over five times with different random splits of labeled and unlabeled sets for each dataset, and we reported the mean and standard deviation of the test error rate in Table.

1.

The results in Table.

1 indicates that on both CIFAR-10 and SVHN, the gap between the fully-supervised baseline and ROT is bigger than this gap for the other SSL methods.

This indicates the potential of our model for leveraging the unlabeled data in comparison to other methods that also use the unlabeled data to improve the classification performance of a CNN model in SSL fashion.

Moreover, we trained our baseline WRN on the entire training set of CIFAR-10 and SVHN and the test error over five runs are 4.23(±0.18) and 2.56(±0.04), respectively.

Besides the particular manner in which we choose the one particular pseudo-label, we also use "soft pseudo-labels".

Essentially, instead of having the one-hot target in the usual classification loss (i.e., cross-entropy), we can have the row of the transport plan corresponding to the unlabeled data points as the target.

We used the soft pseudo-labels produced by OT to train the CNN.

The comparison of results in Table.

1 show that one-hot targets used in ROT outperforms the soft pseudo-labels used in ROT.

Why this is happening can be supported by SSL methods based on the entropy minimization criterion.

This set of methods force the model to produce confident predictions (i.e., low entropy for output of the model).

Similarly here, once we use one-hot targets, we encourage the network to produce more confident predictions than when using soft-pseudo labels.

In this section, we compare ROT which is based on the measure of measure OT with two other baselines.

Both the baselines assign pseudo-labels for the unlabeled samples based on the greedy nearest neighbor (GNN) search.

The first baseline is sample to sample (S-S-GNN) case, where pseudolabels for the unlabeled data are obtained by GNN on the outputs of softmax layer.

Specifically, for each of the unlabeled sample, we annotate it with the label of the closest labeled sample in the training set.

The second baseline is sample to measure (S-M-GNN) case where, pseudo-labels of the unlabeled samples are obtained based on the GNN between the unlabeled samples and the probability measures constructed by initial labeled data in the training set.

When transporting from a Dirac to a probability measure, the OT problem (regularized or not) has a closed form.

Essentially, there is only one admissible coupling.

Thus, in such a case, the Wasserstein distance between a sample to a probability measure is simply computed as follows:

Given an unlabeled Dirac δ x i and a labeled measure

The comparison of results between ROT, and these baselines on the SVHN and CIFAR-10 in Table.

2 shows the benefit of measure of measure OT for training a CNN in an SSL manner.

Instead of using the CNN as a classifier to produce pseudo-labels for the unlabeled data, we used the Wasserstein barycenters to cluster the unlabeled data.

This allowed us to explore the unlabeled measures that we could then match them with the labeled measures for pseudo-labeling.

This was because the CNN, as a classifier trained on a limited amount of the labeled data, simply miss-classifies the unlabeled data.

To compare these two different strategies for producing the pseudo-labels to train the CNN classifier in an SSL fashion, we experimentally show how the clustering-based method (i.e., ROT) can have a greater positive influence on the training of our CNN classifier.

We report the number of pseudo-labels which are accurately predicted by ROT.

This result allows us to know the level of accuracy of the pseudo-label obtained for the unlabeled data, which the CNN can then benefit from during the training.

We also report these results with that of predicted labels achieved by the baseline CNN classifier (i.e., WRN) on the unlabeled training data.

This comparison also allows us to know whether or not the CNN classifier can benefit from our strategy for providing pseudo-labels during the training, because, otherwise, the WRN can simply use its own predicted labels on unlabeled training data over the course of training.

To indicate the efficiency of our method during the training of the CNN, we changed the number of initial labeled data in the training set and reported the number of accurately predicted pseudo-labels by the baseline WRN, and ROT on the remaining unlabeled training data.

Fig. 2 (c) and Fig. 1(d) show that, for both CIFAR and SVHN datasets, the labels predicted by ROT on the unlabeled training data are more accurate than the WRN, which means that the entire CNN network can better benefit from the ROT strategy than the case where it is trained solely by its own predicted labels.

Moreover, we monitored the trend of transportation cost between the labeled and unlabeled measures obtained by Eq. 3 during the training.

Fig. 2(a) and Fig. 2(b) show that the transportation cost is reduced as the images fed into the CNN are represented by a better feature set during the training.

In Table.

2, we evaluated ROT for the case where we only use 4,000 and 1,000 initial labeled data for the CIFAR-10 and SVHN, respectively.

However, here, we explore that how varying the amount of initial labeled data decreases the performance of ROT in the very limited label regime, and also at which point our SSL method can recover the performance of training when using all of the labeled data in the dataset.

To do this evaluation, we gradually increase the number of labeled data during the training and report the performance of our SSL method on the testing set.

In this experiment, we ran our SSL method over five times with different random splits of labeled and unlabeled sets for each dataset, and reported the mean and standard deviation of the error rate in Fig. 2(a) and Fig. 2(b) .

The results show that the performance of ROT tends to converge as the number of labels increases.

Another possibility for evaluating the performance of our SSL method is to change the number of unlabeled data during the training.

However, using the CIFAR-10 and SVHN datasets in isolation puts an upper limit on the amount of available unlabeled data.

Fortunately, in contrast to CIFAR-10, SVHN has been distributed with the SVHN-extra dataset, which includes 531,131 additional digit images and has also been previously used as unlabeled data for evaluation of different SSL methods in Oliver et al. (2018) .

These additional data come from the same distribution as SVHN does, which allows us to use them in our SSL framework.

Fig. 2(c) shows the trend of test error for our SSL algorithm on SVHN with 1,000 labels and changing amounts of unlabeled images from SVHN-extra dataset.

The results shows that, increasing the amount of unlabeled data improves the performance of our SSL method, but this improvement is not significant when we provide 40k unlabeled data.

We proposed a new SSL method based on the optimal transportation technique in which unlabeled data masses are transported to a set of labeled data masses, each of which is constructed by data belonging to the same class.

In this method, we found a mapping between the labeled and unlabeled masses which was used to infer pseudo-labels for the unlabeled data so that we could use them to train our CNN model.

Finally, we experimentally evaluated our SSL method to indicate its potential and effectiveness for leveraging the unlabeled data when labels are limited during the training.

Discrete Optimal Transport: For any r ≥ 1, let the probability simplex be denoted by

, and also assume that U = {u 1 , ..., u n } and V = {v 1 , ..., v m } are two sets of data points in

between two discrete measures U and V is the k-th root of the optimum of a network flow problem known as the transportation problem Bertsimas & Tsitsiklis (1997) .

Note that δ ui is the Dirac unit mass located on point u i , a and b are the weighting vectors which belong to the probability simplex ∆ n and ∆ m , respectively.

The transportation problem depends on the two following components: 1) matrix M ∈ R n×m + which encodes the geometry of the data points by measuring the pairwise distance between elements in U and V increased to the power k, 2) the transportation polytope P (a, b) ∈ R n×m + which acts as a feasible set, characterized as a set of n × m non-negative matrices such that their row and column marginals are a and b, respectively.

This means that the transportation plan should satisfy the marginal constraints.

In other words, let 1 m be an m-dimensional vector with all elements equal to one, then the transportation polytope is represented as follows: P (a, b) = {T ∈ R n×m + |T 1 n = b, T 1 m = a}. Essentially, each element T (i, j) indicates the amount of mass which is transported from i to j. Note that in the transportation problem, the matrix M is also considered as a cost parameter such that

Let T, M denote the Frobenius dot-product between T and M matrices.

Then the discrete Wasserstein distance W k (U, V) is formulated by an optimum of a parametric linear program g(.) on a cost matrix M , and n × m number of variables parameterized by the marginals a and b as follows:

The Wasserstein distance in (6) is a Linear Program (LP) and a subgradient of its solution can be calculated using Lagrange duality.

The dual LP of (6) is formulated as follows:

where the polyhedron C M of dual variables is as follows:

Considering LP duality, the following equality is established Tsitsiklis (1997) .

Computing the exact Wasserstein distance in (6) is time consuming.

To alleviate this problem, Cuturi (2013) has introduced an interesting method that regularizes (6) using the entropy of the solution matrix H(T ), (i.e., min T, M + γH(T )).

It has been shown that if T γ is the solution of the regularized version of (6) and α γ is its dual solution in (7), then ∃!u ∈ R n + , v ∈ R m + such that the solution matrix is T γ = diag(u)Kdiag(v) and α γ = − log(u)/γ + (log(u) 1 n )/(γn))1 n where, K = exp(−M/γ).

The vectors u and v are updated iteratively between step 1 and 2 by using the well-known Sinkhorn algorithm as follows: step 1)u = a/Kv and step 2)v = b/K u, where/ denotes element-wise division operator Cuturi (2013) .

Given an image x n ∈ R m×n from the either labeled or the unlabeled set, the CNN acts as a function f n : R m×n → R c with the parameters θ n that maps x n to a c-dimensional representation, where c is the number of classes.

Assume that X u = {x 1 , ..., x n } is the set of CNN outputs extracted from the unlabeled data.

As noted in Cuturi & Doucet (2014) , the Wasserstein barycenter of the unlabeled set X u is equivalent to Lloyd's algorithm, where the maximization step (i.e., the assignment of the weight of each data point to its closest centroid) is equivalent to the computation of α in dual form, while the expectation step (i.e., the re-centering step) is equivalent to the update for centers Y using the optimal transport, which in this case is equivalent to the trivial transportation plan that assigns the weight (divided by n) of each unlabeled data in X u to its closest neighbor in centers Y .

Algorithm 1 shows the Wasserstein barycenter of the unlabeled data for clustering.

while not converged do 6:

, t ← t + 1 10:

end while 11:

a ←â 12:

Expectation Step:

T ← optimal coupling of p(a, b, M XuY )

14:

, balancing coefficients: α, λ, learning rate: β, batch size: b, distance matrix: X, 1: train CNN parameters initially using the labeled data, 2: repeat 3:

: Softmax layer outputs on Z l and Z u , 4:

{Q 1 , ..., Q c } ← cluster on X u using Algorithm.

1,

{P 1 , ..., P c } ← labeled data grouped to c classes, 6: compute α, β based on amount of the mass in measures Q and P,

for each Q i and P j do 8:

end for 10:

T ← optimal coupling of p(α, β, X),

{y u } n u=1 ← pseudo-label data in each cluster Q i with the highest amount of mass transport toward the labeled measure (i.e., argmax T (i, :)), 12: The regular OT problem defined in (6) can be solved by an effective linear programming method in the order of O(n 3 log(n)) time complexity, where n is number of the points in each probability measures.

Cuturi Cuturi (2013) has introduced an interesting approach which relaxes the OT problem by adding a strong convex regularizer to the OT cost function to reduce the time complexity to O(n 2 ).

Specifically, this approach asks for a solution T with more entropy, instead of computing the exact Wasserstein distance.

In other words, the regularized OT distances can interpolate the solution, depending on the regularization strength γ, between exact OT (γ = 0 ), and Maximum Mean Discrepancy, MMD, (γ = ∞).

In this work, we use the regularized OT not only for the matter of time complexity, but also it has been shown that the sample complexity of exact Wasserstein distance is O(1/n 1/d ), while the regularized Wasserstein distance depending on γ value, is between O(1/ √ n) and O(1/n 1/d ), where d is dimension of the samples Genevay et al. (2019); .

This means that the entropic regularization reduces the chance of over-fitting for our SSL model when it computes the Wasserstein distance between output of the CNN obtained from the labeled and unlabeled data.

Hence, our OT problem in the regularized form is recast as follows:

where γ is a hyperparameter that balances two terms in (9), and E(T ) = − mn ij T ij (log(T ij − 1) is the entropy of the solution matrix T .

It has been shown that if T γ is the solution of the optimization (9), then ∃!u ∈ R n + , v ∈ R m + such that the solution matrix for (9) is T γ = diag(u)Kdiag(v) where, K = exp(−X/γ) Cuturi (2013) .

The vectors u and v are updated iteratively between step 1 and 2 by using the well-known Sinkhorn algorithm as follows: step 1)u = a/Kv and step 2)v = b/K u, where/ denotes element-wise division operator Cuturi (2013) .

It can be simply shown that there always exists an optimal coupling, π ∈ Π(M, M ), that achieves infimum of Eq. (2) in the paper.

This is because the cost function ||x − y|| in Eq. (1) is continuous, and based on Theorem 4.1, the existence of an optimal coupling π ∈ Π(R, S) which obtains the infimum is guaranteed due to the tightness of Π(R, S).

Furthermore, based on Corollary 6.11, the term W k (x, x ) used in Eq. (2) is a continuous function and Π(M, M ) is tight again, so the existence of an optimal coupling in Π(M, M ) is also guaranteed.

Let L 1 be the Lebesgue space of exponent 1, and (X , µ) and (Y, ν) be two Polish probability spaces; let a : X → R ∪ {−∞} and b : Y → R ∪ {−∞} be two upper semi-continuous functions such that

Then there is a coupling of (µ, ν) which minimizes the total cost Ec(X, Y ) among all possible couplings (X, Y ).

Lemma 1: Let X and Y be two Polish spaces.

Let R ⊂ P(X ) and S ⊂ P(Y) be tight subsets of P(X ) and P(Y) respectively.

Then, the set Π(R, S) of all transference plans whose marginals lie in R and S respectively, is itself tight in P(X × Y).

Proof of Lemma: Let µ ∈ R, ν ∈ S, and π ∈ Π(µ, ν).

By assuming that, for any > 0 there is a compact set K ⊂ X , independent of the choice of µ in R, such that µ[X nK ] ≤ ; and similarly there is a compact set L ⊂ Y, independent of the choice of ν in S, such that ν[YnL ] ≤ .

Then, for any coupling (X, Y ) of (µ, ν),

The desired result follows because this bound is independent of the coupling, and K × L is compact in X × Y.

Lemma 2: Let X and Y be two Polish spaces, and c : X × Y → R ∪ {+∞} a lower semi-continuous cost function.

Let h : X × Y → R ∪ {−∞} be an upper semi-continuous function such that c ≥ h. Let (π k ) k ∈ N be a sequence of probability measures on X × Y, converging weakly to some

Therefore,

In particular, if c is non-negative, then F : π → cdπ is lower semi-continuous on P(X × Y), equipped with the topology of weak convergence.

Replacing c by c−h, we may assume that c is a non-negative lower semi-continuous function.

Then c can be written as the point-wise limit of a non-decreasing family (c ) ∈ N of continuous real-valued functions.

By monotone convergence, Prokhorovs Theorem Billingsley (2013): If X is a Polish space, then a set R ⊂ P(X ) is precompact for the weak topology if and only if it is tight, i.e. for any > 0 there is a compact set K such that µ[X nK ] ≤ for all µ ∈ R.

Proof of Theorem 4.1: Since X is Polish, {µ} is tight in P(X ); similarly, {ν} is tight in P(Y).

By using the Lemma 1, Π(µ, ν) is tight in P(X × Y), and by using Prokhorovs theorem, this set has a compact closure.

By passing to the limit in the equation for marginals, we see that Π(µ, ν) is closed, so it is in fact compact.

Then let (π k ) k ∈ N be a sequence of probability measures on X × Y, such that cdπ k converges to the infimum transport cost.

Extracting a sub-sequence if necessary, we may assume that π k converges to some π ∈ Π(µ, ν).

, and c ≥ h by assumption; moreover, hdπ k = hdπ = adµ + bdν; so Lemma 2 implies:

Therefore, π is minimizing.

Note that further details of the proof of Theorem 4.1 are also available in Villani's book Villani (2008) .

Corollary 6.11 in Villanis book Villani (2008) :

) is a Polish space, and p ∈ [1, ∞), then W p is continuous on P p (X ).

More explicitly, if µ k (resp.

ν k ) converges to µ (resp.

ν) weakly in P p (X ) as k → ∞, then

One of the interesting arguments presented in Oliver et al. (2018) for a standard evaluation of different SSL models is that it may not be feasible to perform model selection for an SSL challenge if the hyperparameters of the model are tuned on the realistically small validation sets.

On the other hand, most of the SSL datasets in the literature are designed in such a way that the validation set, which is used for tuning the hyperparameters but not for parameters of the model, is much larger than the training set.

For example, the standard SVHN dataset used in our work has about 7000 labeled data in the validation set.

Hence, the validation set is seven times larger than the training set of the SSL methods which evaluate their performance by using only 1,000 labeled data during the training.

However, this is not a practical choice for a real-world application.

This is because, this large validation set will be used as the training set instead of validation set for tuning the hyperparameters.

Using small validation sets, however, causes an issue in that the evaluation metric, such as the accuracy for tuning the hyperparameters will be unstable and noisy across the different runs.

Although the fact that small validation sets limit the ability for model selection has been discussed in Chapelle et al. (2009) , the work presented in Oliver et al. (2018) has used the Hoeffding inequality Hoeffding (1994) to directly analyze the relationship between the size of validation set and the variance in estimation of a models accuracy: P(|V − E(V )| < p) > 1 − 2 exp(−2np 2 ).

In this inequality, V denotes the empirical estimate of the validation error, E[V ] is its hypothetical true value, p is the desired maximum deviation between the estimation and the true value, and n represents the number of samples in the validation set.

Based on this inequality, the number of samples in the validation set should be very large.

For example, we will require about 20,000 samples in the validation set if we want to be 95% confident in estimation of validation error that differes less than 1% from the absolute true value.

Note that in this analysis, validation error is computed as the average of independent binary indicator variables representing if a given sample in the validation set is classified correctly or not.

This analysis may be unrealistic because of the assumption that the validation accuracy is the average of independent variables.

To address this problem, Oliver et al. Oliver et al. (2018) measure this phenomenon empirically, and train the SSL methods using 1,000 labels in the training set from SVHN dataset and then evaluate them on the validation sets with different sizes.

Note that these small synthetic validation sets are generated by different randomly sampled sets without overlapping from the full SVHN validation set.

Following the same setting for evaluation of our SSL algorithm (ROT) in a real world scenario, in Fig. 3(a) and Fig. 4(a) , we reported the mean and standard deviation of validation errors over five times randomly non-overlapping splitting the SVHN and CIFAR validation sets with varying sizes.

The results in Fig. 3(a) and Fig. 4(a) indicate that as we increase the size of validation set, the ROT algorithm will be more confident and stable to select its hyperparameters than the case where we use small-size validation set.

For a fair comparison between our method and the other SSL methods in Table.

1 of the paper, we have been consistent with other methods in the size of the training and validation sets as it is designed in standard SVHN and CIFAR-10 datasets.

Specifically, for SVHN, we used 65,932 images for the training set and 7,325 for the validation set, and for CIFAR-10 dataset, we used 45,000 images for the training set and 5,000 images for the validation set.

Fig. 3(b) and Fig. 4(b) indicate the error rate of the ROT algorithm on the SVHN and CIFAR validation sets for different values of λ in our transportation plan.

Note that during the tuning of λ, we fixed α in Eq. (5)

<|TLDR|>

@highlight

We propose a new algorithm based on the optimal transport to train a CNN in an SSL fashion.