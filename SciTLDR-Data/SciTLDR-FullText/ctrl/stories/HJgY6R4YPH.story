The difficulty of obtaining sufficient labeled data for supervised learning has motivated domain adaptation, in which a classifier is trained in one domain, source domain, but operates in another, target domain.

Reducing domain discrepancy has improved the performance, but it is hampered by the embedded features that do not form clearly separable and aligned clusters.

We address this issue by propagating labels using a manifold structure, and by enforcing cycle consistency to align the clusters of features in each domain more closely.

Specifically, we prove that cycle consistency leads the embedded features distant from all but one clusters if the source domain is ideally clustered.

We additionally utilize more information from approximated local manifold and pursue local manifold consistency for more improvement.

Results for various domain adaptation scenarios show tighter clustering and an improvement in classification accuracy.

Classifiers trained through supervised learning have many applications (Bahdanau et al., 2015; Redmon et al., 2016) , but it requires a great deal of labeled data, which may be impractical or too costly to collect.

Domain adaptation circumvents this problem by exploiting the labeled data available in a closely related domain.

We call the domain where the classifier will be used at, the target domain, and assume that it only contains unlabeled data {x t }; and we call the closely related domain the source domain and assume that it contains a significant amount of labeled data {x s , y s }.

Domain adaptation requires the source domain data to share discriminative features with the target data (Pan et al., 2010) .

In spite of the common features, a classifier trained using only the source data is unlikely to give satisfactory results in the target domain because of the difference between two domains' data distributions, called domain shift (Pan et al., 2010) .

This may be addressed by fine-tuning on the target domain with a small set of labeled target data, but it tends to overfit to the small labeled dataset (Csurka, 2017) .

Another approach is to find discriminative features which are invariant between two domains by reducing the distance between the feature distributions.

For example, domain-adversarial neural network (DANN) (Ganin et al., 2016) achieved remarkable result using generative adversarial networks (GANs) (Goodfellow et al., 2014) .

However, this approach still has room to be improved.

Because the classifier is trained using labels from the source domain, the source features become clustered, and they determine the decision boundary.

It would be better if the embedded features from the target domain formed similar clusters to the source features in class-level so that the decision boundary does not cross the target features.

Methods which only reduce the distance between two marginal distributions bring the features into general alignment, but clusters do not match satisfactorily, as shown in Fig. 1(a) .

As a consequence, the decision boundary is likely to cross the target features, impairing accuracy.

In this work, we propose a novel domain adaptation method to align the manifolds of the source and the target features in class-level, as shown in Fig. 1(b) .

We first employ label propagation to evaluate the relation between manifolds.

Then, to align them, we reinforce the cycle consistency that is the correspondence between the original labels in the source domain and the labels that are propagated from the source to the target and back to the source domain.

The cycle consistency draws features from both domains that are near to each other to converge, and those that are far apart to diverge.

The proposed method exploits manifold information using label propagation which had not been taken into account in other cycle consistency based methods.

As a result, our approach outperforms other baselines on various scenarios as demonstrated in Sec. 4.

Moreover, the role of cycle consistency is theoretically explained in Sec. 3.2 that it leads to aligned manifolds in class-level.

To acquire more manifold information within the limited number of mini-batch samples, we utilize local manifold approximation and pursue local manifold consistency.

In summary, our contributions are as follows:

??? We propose a novel domain adaptation method which exploits global and local manifold information to align class-level distributions of the source and the target.

??? We analyze and demonstrate the benefit of the proposed method over the most similar baseline, Associative domain adaptation (AssocDA) (Haeusser et al., 2017) .

??? We present the theoretical background on why the proposed cycle consistency leads to class-level manifold alignment, bringing better result in domain adaptation.

??? We conduct extensive experiments on various scenarios and achieve the state-of-the-art performance.

Unsupervised Domain Adaptation It has been shown (Ben-David et al., 2010) that the classification error in the target domain is bounded by that in the source domain, the discrepancy between the domains and the difference in labeling functions.

Based on this analysis, a number of works have endeavored to train domain-confusing features to minimize the discrepancy between the domains (Ganin et al., 2016; Long et al., 2013; Tzeng et al., 2014; 2017) .

Maximum mean discrepancy can be used (Long et al., 2015; Tzeng et al., 2014) as a measure of domain discrepancy.

In an approach inspired by GANs, a domain confusion can be converted (Ganin et al., 2016; Tzeng et al., 2017) into a minmax optimization.

While minimization of domain discrepancy can be effective in reducing the upper bound on the error, it does not guarantee that the feature representation in the target domain is sufficiently discriminative.

To address this issue, several techniques had been proposed.

Explicit separation of the shared representation from the individual characteristics of each domain may enhance the accuracy of the model (Bousmalis et al., 2016) .

This approach has been implemented as a network with private and shared encoders and a shared decoder.

The centroid and prototype of each category can be used for class-level alignment (Pinheiro, 2018; Xie et al., 2018 ).

An alternative to such featurespace adaptation techniques is the direct conversion of target data to source data (Bousmalis et al., 2017; Hoffman et al., 2018; Yoo et al., 2017) .

Those proposed methods intend to transfer the style of images to another domain while preserving the content.

This performs well on datasets containing Figure 2: Overview of our method.

The feature generator G projects the input data into the feature space.

The dashed line means weight sharing.

The embedded source features f s and the target features f t are organized into a graph and then used together to evaluate cycle consistency through label propagation.

The embedding classifier C learns from the source ground-truth labels.

The discriminator D determines whether features originated in the source or the target domain.

images that are similar at the pixel-level; they are problematic when the mapping between high-level features and images is complicated (Tzeng et al., 2017) .

Metric Learning Metric learning is learning an appropriate metric distance to measure the similarity or dissimilarity between data (Bellet et al., 2013) .

Reducing the distances between similar data and increasing the distances between distinct data has shown (Schroff et al., 2015) to improve the accuracy of a classifier.

Metric learning is particularly beneficial when very little labeled data is available, which is the situation for domain adaptation.

Sener et al. (2016) combined metric learning and unsupervised domain adaptation with the enforcement of cycle consistency.

In particular, the inner products of source features and target features with the same label are maximized, and minimized between features with different labels.

AssocDA (Haeusser et al., 2017) enforces the feature alignment between the source and target by forcing the two step round trip probability to be uniform in the same class and to vanish between different classes.

Graph-based learning is closely related to metric learning, in that it achieves clustering using distance information.

Label consistency (Zhou et al., 2004 ) is usually assumed, meaning that adjacent data tend to have the same labels (Wang et al., 2009) .

Label propagation (Zhou et al., 2004) has improved the performance of semi-supervised learning by enforcing label consistency by propagating labels from labeled to unlabeled data.

To overcome need for fixed graphs to be provided in advance, the distances between each node can be adaptively learned (Liu et al., 2019; Oshiba et al., 2019) , as in metric learning, and this increases accuracy in both semi-supervised and few-shot learning.

Our algorithm, shown in Fig. 2 , uses label propagation and cycle consistency to learn features from the source and the target domains which are both 1) indistinguishable each other and 2) close when placed within the same class, but distant when placed in different classes.

The details are as follows.

Manifold learning (Nie et al., 2010) extracts intrinsic structures from both unlabeled and labeled data.

We obtain these structures by constructing a graph whose vertexes are the embedded features and whose edges are the relations between data.

We first embed the input data in the feature space, using the feature generator composed of convolutional layers following previous work (Liu et al., 2019; Oshiba et al., 2019) .

Subsequently, a fully connected graph is constructed according to the distances between the features.

The edge weights W ij between the input data x i , x j are determined from the feature vectors using Gaussian similarity,

), where f i , f j are the embedded feature vectors of x i , x j , and ?? is a scale parameter.

It is known (Liu et al., 2019 ) that graph-based methods are sensitive to the scale parameter ??.

A large ?? results in an uniformly connected graph that disregards the latent structure of the data, while a small ?? produces a sparse graph which fails to express all the relationship between the data.

To adapt ?? according to the embedded features, we take ?? as a trainable variable to be learned during training.

Label propagation (Zhou et al., 2004 ) is a method of manifold regularization, which in turn produces a classifier that is robust against small perturbations.

Label propagation can be seen as a repeated random walk through the graph of features using an affinity matrix to assign the labels of target data (Xiaojin & Zoubin, 2002) .

A label matrix y n ??? R (Ns+Nt)??C refers to the labels assigned to data in both domains at the n-th step random walk.

The dimension of y n is determined by N s , N t , and C which are the numbers of source and target data points and the number of classes, respectively.

The first N s rows of y n contain the labels of the source data, and the remaining N t rows contain the labels of the target data.

The initial label vector y 0 contains y s for the source data, which is one-hot coded ground-truth labels and zero vectors for the target data.

The one step of the random walk transforms the label vector as follows:

where,

.

W ts is a similarity matrix between the target and source data, and W tt is a similarity matrix which represents the interrelations in the target data.

These are described in the Sec. 3.1.

The normalization operation normalize(??) transforms the sum of each row to 1.

The identity matrix in the normalized transition matrix T signifies that the labels of source data do not change because its labels are already known.

In graph theory, these source data points would be called absorbing nodes.

In label propagation, the labels of the target domain is assigned to the propagated labels?? t by infinite transition, formulated as??

tt T ts y s , which converges as follows (Xiaojin & Zoubin, 2002)

In our method,?? t is used to obtain the propagated labels of the source data in the same way a?? y s = (I ??? T ss ) ???1 T st?? t where T ss and T st are defined analogous to T tt and T ts , so that we can learn the features of which clusters match each other.

We then refer to the property that?? s should be the same as the original label y s as cycle consistency.

Pursuing cycle consistency forces not perfectly aligned features to move toward the nearest cluster, as shown in Fig. 3 .

The following theorem shows that enforcing cycle consistency on ideally clustered source data will segregate different classes of the source and the target data and gather the same classes.

Theorem 1.

Let {e i |1 ??? i ??? C} be the standard bases of C-dimensional Euclidean space.

For the sake of simplicity, source data x 1 , x 2 , ?? ?? ?? , x Ns are assumed to be arranged so that the first n 1 data belong to class 1, the n 2 data to class 2, and so forth.

Assume that 1) the source data is ideally clustered, in the sense that T ss has positive values if the row and the column are the same class and zero otherwise, i.e., T ss = diag(T 1 , T 2 , ?? ?? ?? , T C ), the block diagonal where T i is a n i ?? n i positive matrix for i = 1, 2, ?? ?? ?? , C and 2)?? s = y s .

Then for all 1 ??? j ??? C, there exists a nonnegative vector v j ??? R Ns such that 1) the part where source data belongs to j th class (from

th element to [n 1 + n 2 + ?? ?? ?? + n j ] th element) are positive and the other elements are all zero and 2) v j T st?? t e i = 0 for all 1 ??? i ??? C, i = j.

Proof.

The illustration and the proof is given in Appx.

A.

In Thm.

1,?? t e i refers to the assigned probability as i th class to the target data.

The conclusion implies that if a target data is enough to be predicted as i th class through label propagation, i.e., i th elements of the row in?? t corresponding to the target data is nonzero, then the elements of T st which represent the transitions from source data of all but i th class to the target data should vanish, i.e., the target data is segregated from the source data in different classes.

As described in Sec. 3.4, we employed DANN to prevent the target data distribution to be distinct from the source data distribution.

If a column of T st is a zero vector, the feature of the corresponding target data for the column is considerably distant from all source data features.

However, minimizing the DANN loss makes target features lie around source features, and thus each column of T st is not likely to be a zero vector.

Combining this conjecture with Thm.

1, each row of?? t has only one nonzero value, i.e., every target data belongs to only one cluster.

We thus argue that by pursuing this property, generator can learn more discriminative shared features, and classification performance may improve.

Cycle consistency is enforced by minimizing the l 1 loss L cycle between?? s and y s :

Comparison with AssocDA The proposed method has some resemblance with AssocDA in that they both consider the similarities and transitions between data.

However, we argue that AssocDA is a special case of our method.

First, our method exploits manifold over each domain by taking relations within the same domain into account through label propagation, whereas AssocDA only considers relations across the domains.

Specifically, in Eq. 1, our method utilizes both T ts and T tt , but AssocDA ignores T tt which often has useful information about the target data manifold.

Second, AssocDA forces the two-step transition to be uniform within the same class.

This strict condition may drive the source features of each class to collapse to one mode and can cause overfitting.

On the contrary, our method only constrains source data to preserve its original labels after the label propagation.

Thus, it does not require all source data be close to each other within the same class; it allows moderate intra-class variance.

The experiment in Sec. 4.1 and Fig. 4 support these arguments and visualize the effect of the differences.

As shown in Thm.

1, the introduced cycle consistency utilizes graph based global manifold information and enforces the source and target features to be aligned in class-level.

However, in practice, the limited size of mini-batch may restrict the available information of graph.

The knowledge from the local manifold of each sample, in this case, can complement the global manifold information.

In this regard, we additionally pursue local manifold consistency that the output should not be sensitive to small perturbations in the local manifold, as suggested elsewhere (Simard et al., 1992; Kumar et al., 2017; Qi et al., 2018) .

Concretely, localized GAN (LGAN) (Qi et al., 2018 ) is employed to approximate the local manifold of each data and sample a marginally perturbed image along the local manifold from the given data.

LGAN allows it as LGAN focuses on learning and linking patches of local manifolds in its training procedure.

The difference between the predicted label of the perturbed image and that of the original image is minimized to impose local manifold consistency of the classifier as follows:

where, C, G and G L are the embedding classifier, the feature generator and the LGAN generator, respectively.

LGAN generator, G L (x, z), takes an image x and noise z to generate locally perturbed image along the approximated local manifold.

H(??, ??) denotes cross entropy.

?? and ?? are coefficients for the source and the target local manifold consistency loss, respectively.

Our method learns a clustered feature representation that is indistinguishable across the source and target domains through the training process as follows:

where, D is the discriminator.

?? and ?? are coefficients for the last two terms and ?? is a scheduling parameter described in Appx B.1.

L class is a widely used cross-entropy loss for labeled source data and L dann is a GAN loss (Ganin et al., 2016; Goodfellow et al., 2014) :

where discriminator's output D(??) is the probability that the input originated from the source domain.

From the metric learning perspective, L class serves to separate the source features according to their ground-truth labels, which supports the assumption in Thm.

1, the ideally clustered source features.

Subsequently, L dann takes a role in moving the target features toward the source features, but it is insufficient to produce perfectly aligned clusters.

Our cycle loss L cycle and local loss L local facilitate clustering by enforcing cycle consistency and local manifold consistency.

We present a toy example to empirically demonstrate the effect of our proposed cycle loss using manifold information compared to the most similar method, AssocDA.

We designed synthetic dataset in 2-dimensional feature space with two classes as illustrated in the leftmost of Fig. 4 .

The source data lie vertically and the target data are slightly tilted and translated.

The second column shows the negative gradients of AssocDA loss and our cycle loss with respect to each data.

Negative gradients can be interpreted as the movement of features at each iteration.

The third and fourth are the updated features using gradient descent in the middle and at the end of feature updates 1 .

As argued in Sec. 3.2, AssocDA does not consider the transition within the same domain and thus target data which are close to source data with different label (points inside red circles in the second column) are strongly attracted to them.

On the other hand, the gradients of the cycle loss are much smaller than AssocDA.

We speculate that it is because the attractions from source data in the same class are propagated through target data manifold.

As a result, AssocDA leads some data to move in wrong direction, being misclassified, while cycle loss brought correctly aligned manifolds.

In addition, AssocDA attracts all features too close at the end of updates, which may cause overfitting.

Last but not least, our cycle loss aligned source and target clusters correctly without the aid of dann loss.

We thus argue that our method is complementary to DANN rather than an extension.

We show the performance of the proposed method on two real visual dataset.

First dataset, which we call by Digit & Object dataset, includes digit dataset such as SVHN and Synthetic Digits (DIGITS),

Gradients of loss Progress at 150 steps Progress at 600 steps Figure 5 : Visualization of learned features using t-SNE.

Circles and x markers respectively indicate the source and target features.

Colors correspond to labels.

In all cases, the features from two domains form similar and tight clusters, which is the key objective of our method.

and object dataset such as STL and CIFAR.

We used ImageCLEF-DA as second dataset for more challenging benchmark.

We employed three networks as previous work (Shu et al., 2018; Xie et al., 2018; .

A network with two convolutional layers and two fully connected layers for digit dataset and a network with nine convolutional layers and one fully connected layer for object dataset were implemented.

Pretrained ResNet (He et al., 2016) was used for ImageCLEF-DA dataset.

More details on training settings, adaptation scenarios and an experiment on non-visual dataset are provided in Appx.

B.1, B.2 and E.

Tab.

1 compares the accuracy of our method on Digit & Object dataset with that of other approaches.

For our method, we reported the results of three models, one with local loss (L), another with cycle loss (C) and the other with both losses (C+L).

Our algorithm outperformed the others on most of the tasks.

In the most experiments, the performance of the proposed method was better than the state-ofthe-art.

This suggests that enforcing alignment in addition to domain-invariant embedding reduces the error-rate.

PixelDA (Bousmalis et al., 2017) showed superior performance on MNIST???MNIST-M, but it is attributable to the fact that PixelDA learns transferring the style of images at a pixel level which is similar (Pinheiro, 2018) to the way MNIST-M is generated from MNIST.

T-SNE embeddings in Fig. 5 indicates that the learned features are well aligned and clustered.

Tab.

2 reports the results on ImageCLEF-DA dataset experiments.

The performance of our method was better than or comparable to those of other baselines.

Especially, our method outperforms CAT Deng et al. (2019) which also aims to learn clustered and aligned features.

Although the objectives are related, the approaches are quite different.

Our method utilizes the manifolds of the source and the target domain through label propagation and cycle consistency, whereas CAT considers the distance between two samples for clustering and the distance between the first-order statistics of distributions for alignment.

We argue that the better performance is attributed to utilizing manifold information beyond one to one relations of which benefits are explained in Sec. 4.1.

Throughout ImageCLEF-DA experiments, the proposed method without the local loss achieved better accuracy compared to that with the local loss.

Approximation of the local manifold on ImageCLEF-DA generated by LGAN was slightly worse than that on Digit & Object dataset; perturbed image was blurred and semantically invariant with the original image.

Hence, we speculate that the performance of the proposed method may be improved with better local manifold approximation.

In this paper, we proposed a novel domain adaptation which stems from the objective to correctly align manifolds which might result in better performance.

Our method achieved it, which was supported by intuition, theory and experiments.

In addition, its superior performance was demonstrated on various benchmark dataset.

Based on graph, our method depends on how to construct the graph.

Pruning the graph or defining a similarity matrix considering underlying geometry may improve the performance.

Our method also can be applied to semi supervised learning only with slight modification.

We leave them as future work.

A PROOF OF THEOREM 1 Theorem 1.

Let {e i |1 ??? i ??? C} be the standard bases of C-dimensional Euclidean space.

For the sake of simplicity, source data x 1 , x 2 , ?? ?? ?? , x Ns are assumed to be arranged so that the first n 1 data belong to class 1, the n 2 data to class 2, and so forth.

Assume that 1) the source data is ideally clustered, in the sense that T ss has positive values if the row and the column are the same class and zero otherwise, i.e., T ss = diag(T 1 , T 2 , ?? ?? ?? , T C ), the block diagonal where T i is a n i ?? n i positive matrix for i = 1, 2, ?? ?? ?? , C and 2)?? s = y s .

Then for all 1 ??? j ??? C, there exists a nonnegative vector v j ??? R Ns such that 1) the part where source data belongs to j th class (from [n 1 + n 2 + ?? ?? ?? + n j???1 + 1] th element to [n 1 + n 2 + ?? ?? ?? + n j ] th element) are positive and the other elements are all zero and 2) v j T st?? t e i = 0 for all 1 ??? i ??? C, i = j. From the assumption, T ss is a block diagonal matrix of which block elements are T 1 ,T 2 ,?? ?? ?? ,T C .

v j is all zero except n j elements in the middle of v j .

The n j elements are all positive and their indices correspond to those of T j in T ss .

In the proof, the left eigenvector u j of T j will be substituted to this part.

Proof.

From the Perron-Frobenius Theorem (Frobenius et al., 1912; Perron, 1907) that positive matrix has a real and positive eigenvalue with positive left and right eigenvectors, T j , the block diagonal element of T ss , has a positive left eigenvector u j with eigenvalue ?? j for all j = 1, 2, ?? ?? ?? C. Then, as shown below, v j = ( 0 0 ?????? 0 u j 0 ?????? 0 ) where n 1 + n 2 + ?? ?? ?? + n j???1 zeros, u j and n j+1 + n j+2 + ?? ?? ?? + n C zeros are concatenated, is a left eigenvector of T ss with eigenvalue ?? j by the definition of eigenvector.

From the label propagation, we have,??

By multiplying v j (I ??? T ss ) on the left and e i on the right to the both sides in Equation 13 and combining with the assumption?? s = y s , we have,

The last zero comes from the definition of v j .

In this subsection, we offer the modified version of Thm.

1 when the source features are slightly perturbed from the ideally clustered condition and the other assumption y s =?? s holds.

We start from representing T ss as follows to indicate the perturbation.

where, ??T ss is assumed to be sufficiently small under infinite norm and T

ss is a block diagonal transition matrix when the source features are ideally clustered as stated in Thm.

1.

In the proof above, we showed eigenvalue ?? j and its corresponding eigenvector v j of T j .

According to perturbation theory of eigenvalue and eigenvector (Greenbaum et al., 2019) , the eigenvector can be approximated by first order when the perturbation is small.

More generally and precisely,

where, the norm is vector or matrix 2-norm and m j is determined by T

ss .

For the sake of simplicity, we use Big-O notation in Eq. 19 and Eq. 20.

Now, we reuse Eq. 16 from the proof of Theorem 1 since it is still valid under the modified condition.

We apply Eq. 19 to the right hand side as follows,

where i = j. Eq. 24 holds because only j th block elements of v (0) j are nonzero.

We also used the fact that y s is bounded by 0 and 1.

Similarly, the left hand side of Eq. 22 can be transformed as follows,

The second term of Eq. 26 holds because T st and?? t are bounded by 0 and 1.

Finally, by combining Eq. 24 and Eq. 26, we have,

Eq. 27 implies that if the perturbation is sufficiently small i.e., ||??T ss || ??? << 1 and a target data is enough to be predicted as i th class through label propagation, then the transitions from source data of all but i th class to the target data is negligible because v (0) j is positive for j th block and zero for others.

It is the same with the conclusion of Theorem 1.

In addition, the more strongly the target data is classified as i th class i.e., the corresponding element of?? t becomes greater, the smaller the transitions from source data in the other classes are, indicating the segregation against the other classes.

Practically, the coefficients for L cycle and L cycle are scheduled to facilitate the clustering of source features correctly at the early stage of training.

Thus we may assume that T ss is marginally perturbed around the ideally clustered one when our cycle loss takes effect.

Scheduling the effect of losses To reduce the effect of noisy signal from L dann and L cycle during the early stages of training, a weight balance factor ?? = 2 1+exp(???????p) ??? 1 is applied in Eq. 6.

A constant ?? determines the rate of increase of ??; p is the progress of training, which proceeds from 0 to 1.

The parameter was introduced (Ganin et al., 2016) to make a classifier less sensitive to the erroneous signals from the discriminator in the beginning.

Throughout the experiments, ?? was set to 10.

Hyperparameter Although it would be ideal to avoid utilizing labels from the target domain in the hyperparameter optimization, it seems that no globally acceptable method exists for this.

One possibility (Ganin et al., 2016) is reverse validation scheme but this may not be accurate enough to estimate test accuracy (Bousmalis et al., 2016) .

In addition (Bousmalis et al., 2016) , applications exist where the labeled target domain data is available at the test phase but not at the training phase.

Hence, we adopted the protocol of (Bousmalis et al., 2016) that exploits a small set of labeled target domain data as a validation set; 256 samples for the Amazon review experiment and 1,000 samples for the other experiments (Bousmalis et al., 2016; 2017; Saito et al., 2017) .

During training, Adam optimizer (Kingma & Ba, 2015) with learning rate of 10 ???3 was utilized.

Exponential moving averaging was applied to the optimization trajectory.

It is an inherent characteristic of our method that each data sample affects the graph structure.

So it is important for each class sample in each batch to represent its classes accurately.

In other words, the transition matrix can be corrupted by biases in the samples.

Therefore, the number of data samples in each class in a batch should be sufficient to avoid any likely bias.

To address this problem, we performed experiments with batch size of up to 384 and observed very little improvement beyond a batch size of 128.

So we fixed the batch size to 128 for Digit & Object dataset.

For the ImageCLEF-DA dataset, we set the batch size to 36 because of limited computing resource.

MNIST ???

MNIST-M The MNIST database of hand-written digits (LeCun et al., 1998) consists of digit images with 10 classes and MNIST-M (Ganin et al., 2016) consists of MNIST digits blended with natural color patches from the BSDS500 dataset (Arbelaez et al., 2011) .

In addition, following other work (Pinheiro, 2018) the colors of the MNIST images were inverted randomly, because their colors are always white on black, whereas the MNIST-M images exhibit various colors.

MNIST ??? USPS USPS (Denker et al., 1989 ) is another dataset of hand-written images of digits, with 10 classes.

USPS contains 16??16 images and the size of the USPS image is upscaled to 28??28, which is the size of the MNIST image in our experiment.

The evaluation protocol of CYCADA (Hoffman et al., 2018 ) is adopted.

SVHN ??? MNIST The Street View House Numbers (SVHN) (Netzer et al., 2011) dataset consists of images of house numbers acquired by Google Street View.

The natural images that it contains, are substantially different from the line drawings in the MNIST dataset.

The size of each MNIST image is upscaled to 32??32, which is the size of SVHN images.

SYN DIGITS ??? SVHN SYN DIGITS dataset is synthetic number dataset which is similar to the SVHN dataset (Ganin et al., 2016) .

The most significant difference between the SYN DIGITS dataset and the SVHN dataset is the untidiness (Ganin et al., 2016) in the background of real images.

CIFAR ??? STL Both CIFAR dataset (Krizhevsky & Hinton, 2009 ) and STL dataset are 10-class datasets that contain images of animals and vehicles.

Not overlapped classes are removed to make a 9-class domain adaptation task (Shu et al., 2018) .

We used the larger network only for this experiment.

2 The twelve common classes of three publicly available dataset (Caltech-256, ImageNet ILSVRC2012, and PASCAL VOC2012) are selected to form visual domain adaptation tasks.

We perform all six possible adaptation scenarios among these three dataset.

We searched hyperparameters within ?? = {0, 0.01, 0.1, 1}, ?? = {0.01, 0.1, 1}, ?? = {0, 0.01} and ?? = {0, 0.1}. Perturbation to the LGAN generator, i.e. z, is fixed to 0.5 for all experiments.

The best hyperparameters for each task is shown in Table.

3.

Setting an appropriate value for the scale parameter, ??, is important because it has a substantial role in determining the transition matrix, T. Therefore, we conducted several experiments with fixing ?? to various values.

For these experiments, we excluded L local to observe the effect of ??. '

Adapt' means that the ?? is learned to adapt according to the embedded features.

For four out of five scenarios, fixing ?? to 1 performed better than fixing it to 0.1 or 10.

With this observation, we initialized ?? to 1 took it as a trainable variable.

The result of adaptively learning ?? is reported at the bottom row of the table.

Compared to fixing ?? to 1, adaptively learning ?? achieved better accuracy and had a lower standard deviation range which means that it is more stable.

We also would like to highlight that our model is robust to the initial value of ??.

We conducted extensive experiments with initializing ?? to 0.1, 1 and 10 and taking it as a trainable variable.

Except for SVHN ??? MNIST transfer task with setting initial ?? value to 10, the initial value of ?? has a minute influence to the accuracy.

We believe that adaptively learning the scale parameter can be usefully employed in any other graph-based method.

The learned ?? values for various scenarios are as follows.

It seems that ?? adaptively learns its value according to the transfer task, regardless of its initialization.

We tried l 2 loss and cross entropy loss to enforce cycle consistency as well.

We excluded L local to compare the effectiveness of these functions.

For all Digit dataset adaptation experiments, evaluating cycle consistency with l 1 norm achieved the highest accuracy.

We speculate that l 1 norm is more numerically stable or provides more effective gradients than other functions in this case.

The Amazon Reviews (Blitzer et al., 2007) dataset provides a non-visual domain for domain adaptation experiments.

It contains reviews of books, DVDs, electronics, and kitchen appliances encoded as 5,000-dimensional feature vectors containing unigrams and bigrams of the texts with binary labels.

Four-and five-star reviews are labeled 'positive'; reviews with fewer stars are labeled 'negative'.

We used 2,000 labeled source data and 2,000 unlabeled target data for training, and between 3,000 to 6,000 target data for testing.

Tab.

8 shows that our method performs better than DANN (Ganin et al., 2016) , VFAE (Louizos et al., 2016) and ATT (Saito et al., 2017) on the Amazon Reviews data in six out of twelve experiments.

Our method was more accurate than DANN in nine out of twelve settings, showing approximately 2.0% higher classification accuracy on average.

<|TLDR|>

@highlight

A novel domain adaptation method to align manifolds from source and target domains using label propagation for better accuracy.