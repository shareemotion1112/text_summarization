We propose order learning to determine the order graph of classes, representing ranks or priorities, and classify an object instance into one of the classes.

To this end, we design a pairwise comparator to categorize the relationship between two instances into one of three cases: one instance is `greater than,' `similar to,' or `smaller than' the other.

Then, by comparing an input instance with reference instances and maximizing the consistency among the comparison results, the class of the input can be estimated reliably.

We apply order learning to develop a facial age estimator, which provides the state-of-the-art performance.

Moreover, the performance is further improved when the order graph is divided into disjoint chains using gender and ethnic group information or even in an unsupervised manner.

To measure the quality of something, we often compare it with other things of a similar kind.

Before assigning 4 stars to a film, a critic would have thought, "It is better than 3-star films but worse than 5-stars."

This ranking through pairwise comparisons is done in various decision processes (Saaty, 1977) .

It is easier to tell the nearer one between two objects in a picture than to estimate the distance of each object directly (Chen et al., 2016; Lee & Kim, 2019a) .

Also, it is easy to tell a higher pitch between two notes, but absolute pitch is a rare ability (Bachem, 1955) .

Ranking through comparisons has been investigated for machine learning.

In learning to rank (LTR), the pairwise approach learns, between two documents, which one is more relevant to a query (Liu, 2009) .

Also, in ordinal regression (Frank & Hall, 2001; Li & Lin, 2007) , to predict the rank of an object, binary classifications are performed to tell whether the rank is higher than a series of thresholds or not.

In this paper, we propose order learning to learn ordering relationship between objects.

Thus, order learning is related to LTR and ordinal regression.

However, whereas LTR and ordinal regression assume that ranks form a total order (Hrbacek & Jech, 1984) , order learning can be used for a partial order as well.

Order learning is also related to metric learning (Xing et al., 2003) .

While metric learning is about whether an object is 'similar to or dissimilar from' another object, order learning is about 'greater than or smaller than.'

Section 2 reviews this related work.

In order learning, a set of classes, Θ = {θ 1 , θ 2 , · · · , θ n }, is ordered, where each class θ i represents one or more object instances.

Between two classes θ i and θ j , there are three possibilities: θ i > θ j or θ i < θ j or neither (i.e. incomparable).

These relationships are represented by the order graph.

The goal of order learning is to determine the order graph and then classify an instance into one of the classes in Θ. To achieve this, we develop a pairwise comparator that determines ordering relationship between two instances x and y into one of three categories: x is 'greater than,' 'similar to,' or 'smaller than'

y.

Then, we use the comparator to measure an input instance against multiple reference instances in known classes.

Finally, we estimate the class of the input to maximize the consistency among the comparison results.

It is noted that the parameter optimization of the pairwise comparator, the selection of the references, and the discovery of the order graph are jointly performed to minimize a common loss function.

Section 3 proposes this order learning.

We apply order learning to facial age estimation.

Order learning matches age estimation well, since it is easier to tell a younger one between two people than to estimate each person's age directly (Chang et al., 2010; Zhang et al., 2017a) .

Even when we assume that age classes are linearly ordered, the proposed age estimator performs well.

The performance is further improved, when classes are divided into disjoint chains in a supervised manner using gender and ethnic group information or even in an unsupervised manner.

Section 4 describes this age estimator and discusses its results.

Finally, Section 5 concludes this work.

Pairwise comparison:

It is a fundamental problem to estimate the priorities (or ranks) of objects through pairwise comparison.

In the classic paper, Saaty (1977) noted that, even when direct estimates of certain quantities are unavailable, rough ratios between them are easily obtained in many cases.

Thus, he proposed the scaling method to reconstruct absolute priorities using only relative priorities.

The scaling method was applied to monocular depth estimation (Lee & Kim, 2019a) and aesthetic assessment (Lee & Kim, 2019b) .

Ranking from a pairwise comparison matrix has been studied to handle cases, in which the matrix is huge or some elements are noisy (Braverman & Mossel, 2008; Jamieson & Nowak, 2011; Negahban et al., 2012; Wauthier et al., 2013) .

On the other hand, the pairwise approach to LTR learns, between two documents, which one is more relevant to a query (Liu, 2009; Herbrich et al., 1999; Burges et al., 2005; Tsai et al., 2007) .

The proposed order learning is related to LTR, since it also predicts the order between objects.

But, while LTR sorts multiple objects with unknown ranks and focuses on the sorting quality, order learning compares a single object x with optimally selected references with known ranks to estimate the rank of x.

Ordinal regression: Ordinal regression predicts an ordinal variable (or rank) of an instance.

Suppose that a 20-year-old is misclassified as a 50-year old and a 25-year old, respectively.

The former error should be more penalized than the latter.

Ordinal regression exploits this characteristic in the design of a classifier or a regressor.

In Frank & Hall (2001) and Li & Lin (2007) , a conversion scheme was proposed to transform an ordinal regression problem into multiple binary classification problems.

Ordinal regression based on this conversion scheme has been used in various applications, including age estimation (Chang et al., 2010; Niu et al., 2016; Chen et al., 2017) and monocular depth estimation (Fu et al., 2018) .

Note that order learning is different from ordinal regression.

Order learning performs pairwise comparison between objects, instead of directly estimating the rank of each object.

In age estimation, ordinal regression based on the conversion scheme is concerned with the problem, "Is a person's age bigger than a threshold θ?" for each θ.

In contrast, order learning concerns "Between two people, who is older?" Conceptually, order learning is easier.

Technically, if there are N ranks, the conversion scheme requires N − 1 binary classifiers, but order learning needs only a single ternary classifier.

Moreover, whereas ordinal regression assumes that ranks form a total order, order learning can be used even in the case of a partial order (Hrbacek & Jech, 1984) .

Metric learning: A distance metric can be learned from examples of similar pairs of points and those of dissimilar pairs (Xing et al., 2003) .

The similarity depends on an application and is implicitly defined by user-provided examples.

If a learned metric generalizes well to unseen data, it can be used to enforce the desired similarity criterion in clustering (Xing et al., 2003) , classification (Weinberger et al., 2006) , or information retrieval (McFee & Lanckriet, 2010) .

Both metric learning and order learning learn important binary relations in mathematics: metric and order (Hrbacek & Jech, 1984) .

However, a metric decides whether an object x is similar to or dissimilar from another object y, whereas an order tells whether x is greater than or smaller than y. Thus, a learned metric is useful for grouping similar data, whereas a learned order is suitable for processing ordered data.

Age estimation: Human ages can be estimated from facial appearance (Kwon & da Vitoria Lobo, 1994) .

Geng et al. (2007) proposed the aging pattern subspace, and Guo et al. (2009) introduced biologically inspired features to age estimation.

Recently, deep learning has been adopted for age estimation.

Niu et al. (2016) proposed OR-CNN for age estimation, which is an ordinal regressor using the conversion scheme.

Chen et al. (2017) proposed Ranking-CNN, which is another ordinal regressor.

While OR-CNN uses a common feature for multiple binary classifiers, Ranking-CNN employs a separate CNN to extract a feature for each binary classifier.

Tan et al. (2018) grouped adjacent ages via the group-n encoding, determined whether a face belongs to each group, and combined the results to predict the age.

Pan et al. (2018) proposed the mean-variance loss to train a CNN classifier for age estimation.

Shen et al. (2018) proposed the deep regression forests for age estimation.

Zhang et al. (2019) developed a compact age estimator using the two-points representation.

Also, Li et al. (2019) proposed a continuity-aware probabilistic network for age estimation.

Figure 1: Examples of order graphs, in which node n precedes node m (n → m), if n divides m. For clarity, self-loops for reflexivity and edges deducible from transitivity are omitted from the graphs.

3 ORDER LEARNING

Let us first review mathematical definitions and concepts related to order.

An order (Hrbacek & Jech, 1984; Bartle, 1976) , often denoted by ≤, is a binary relation on a set Θ = {θ 1 , θ 2 , · · · , θ n } that satisfies the three properties of

In real-world problems, an order describes ranks or priorities of objects.

For example, in age estimation, θ i ≤ θ j means that people in age class θ i look younger than those in θ j .

We may use the symbol →, instead of ≤, to denote an order on a finite set Θ. Then, the order can be represented by a directed graph (Gross & Yellen, 2006 ) using elements in Θ as nodes.

If θ i → θ j , there is a directed edge from node θ i to node θ j .

The order graph is acyclic because of antisymmetry and transitivity.

For example, for n, m ∈ N, let n → m denote that m is a multiple of n. Note that it is an order on any subset of N. Figure 1(a) is the graph representing this order on {1, . . .

, 9}.

Elements θ i and θ j are comparable if θ i → θ j or θ j → θ i , or incomparable otherwise.

In Figure 1(a), 6 and 8 are incomparable.

In age estimation, it is difficult to compare apparent ages of people in different ethnic groups or of different genders.

An order on a set Θ is total (or linear) if all elements in Θ are comparable to one another.

In such a case, Θ is called a linearly ordered set.

In some real-world problems, orders are not linear.

In this work, a subset Θ c of Θ is referred to as a chain, if Θ c is linearly ordered and also maximal, i.e. there is no proper superset of Θ c that is linearly ordered.

In Figure 1 (a), nodes 1, 2, 4, and 8 form a chain.

In Figure 1 (b), the entire set is composed of three disjoint chains.

Let Θ = {θ 1 , θ 2 , · · · , θ n } be an ordered set of classes, where each class θ i represents one or more object instances.

For example, in age estimation, age class 11 is the set of 11-year-olds.

The objective of order learning is to determine the order graph, such as Figure 1 (a) or (b), and categorize an object instance into one of the classes.

However, in many cases, order graphs are given explicitly or obvious from the contexts.

For example, in quality assessment, there are typically five classes (poor → satisfactory →

good → very good → excellent), forming a single chain.

Also, in age estimation, suppose that an algorithm first classifies a person's gender into female or male and then estimates the age differently according to the gender.

In this case, implicitly, there are separate age classes for each gender, and the age classes compose two disjoint chains similarly to Figure 1 (b) .

Thus, in this subsection, we assume that the order graph is already known.

Also, given an object instance, we assume that the chain to which the instance belongs is known.

Then, we attempt to categorize the instance into one of the classes in the chain.

Section 3.4 will propose the order learning in the case of an unknown order graph, composed of disjoint chains.

Instead of directly estimating the class of each instance, we learn pairwise ordering relationship between two instances.

Let Θ c = {0, 1, . . .

, N − 1} be a chain, where N is the number of classes.

Let x and y be two instances belonging to classes in Θ c .

Let θ(·) denote the class of an instance.

Then, x and y are compared and their ordering relationship is defined according to their class difference as

where τ is a threshold.

To avoid confusion, we use ' , ≈, ≺' for the instance ordering, while '>, =, <' for the class order.

In practice, the categorization in (1)∼(3) is performed by a pairwise comparator in Figure 2 , which consists of a Siamese network and a ternary classifier (Lee & Kim, 2019b) .

To train the comparator, only comparable instance pairs are employed.

We estimate the class θ(x) of a test instance x by comparing it with reference instances y m , 0 ≤ m ≤ M − 1, where M is the number of references.

The references are selected from training data such that they are from the same chain as x. Given x and y m , the comparator provides one of three categories ' , ≈, ≺' as a result.

Let θ be an estimate of the true class θ(x).

Then, the consistency between the comparator result and the estimate is defined as

is the indicator function.

The function φ con (x, y m , θ ) returns either 0 for an inconsistent case or 1 for a consistent case.

For example, suppose that the pairwise comparator declares x ≺ y m but θ − θ(y m ) > τ .

Then, φ con (x, y m , θ ) = 0 · 1 + 0 · 0 + 1 · 0 = 0.

Due to a possible classification error of the comparator, this inconsistency may occur even when the estimate θ equals the true class θ(x).

To maximize the consistency with all references, we estimate the class of x bŷ

which is called the maximum consistency (MC) rule.

Figure 3 illustrates this MC rule.

It is noted that ' , ≈, ≺' is not an mathematical order.

For example, if θ(x) + 3 4 τ = θ(y) = θ(z) − 3 4 τ , then x ≈ y and y ≈ z but x ≺ z. This is impossible in an order.

More precisely, due to the quantization effect of the ternary classifier in (1)∼(3), ' , ≈, ≺' is quasi-transitive (Sen, 1969) , and '≈' is symmetric but intransitive.

We use this quasi-transitive relation to categorize an instance into one of the classes, on which a mathematical order is well defined.

In the simplest case of 1CH, all classes form a single chain Θ c = {0, 1, . . .

, N − 1}. For example, in 1CH age estimation, people's ages are estimated regardless of their ethnic groups or genders.

We implement the comparator in Figure 2 (5) is written within the box.

For θ = 7, there are six inconsistent boxes.

For θ = 9, there are 24 such boxes.

In this example, θ = 7 minimizes the inconsistency, or equivalently maximizes the consistency.

Therefore,θ MC (x) = 7.

where T is the set of all training instances and R ⊂ T is the set of reference instances.

First, we initialize R = T and minimize co via the stochastic gradient descent.

Then, we reduce the reference set R by sampling references from T .

Specifically, for each class in Θ c , we choose M/N reference images to minimize the same loss co , where M is the number of all references and N is the number of classes.

In other words, the reliability score of a reference candidate y is defined as

and the M/N candidates with the highest reliability scores are selected.

Next, after fixing the reference set R, the comparator is trained to minimize the loss co .

Then, after fixing the comparator parameters, the reference set R is updated to minimize the same loss co , and so forth.

In the test phase, an input instance is compared with the M references and its class is estimated using the MC rule in (5).

In KCH, we assume that classes form K disjoint chains, as in Figure 1 (b).

For example, in the supervised 6CH for age estimation, we predict a person's age according to the gender in {female, male} and the ethnic group in {African, Asian, European}. Thus, there are 6 chains in total.

In this case, people in different chains are assumed to be incomparable for age estimation.

It is supervised, since gender and ethnic group annotations are used to separate the chains.

The supervised 2CH or 3CH also can be implemented by dividing chains by genders only or ethnic groups only.

The comparator is trained similarly to 1CH.

However, in computing the comparator loss in (6), a training instance x and a reference y are constrained to be from the same chain.

Also, during the test, the type (or chain) of a test instance should be determined.

Therefore, a K-way type classifier is trained, which shares the feature extractor with the comparator in Figure 2 and uses additional fully-connected (FC) layers.

Thus, the overall loss is given by

where co is the comparator loss and ty is the type classifier loss.

The comparator and the type classifier are jointly trained to minimize this overall loss .

During the test, given an input instance, we determine its chain using the type classifier, and compare it with the references from the same chain, and then estimate its class using the MC rule in (5).

This subsection proposes an algorithm to separate classes into K disjoint chains when there are no supervision or annotation data available for the separation.

First, we randomly partition the training set

Input: T = training set of ordinal data, K = # of chains, N = # of classes in each chain, and M = # of references in each chain 1: Partition T randomly into T0, . . .

, TK−1 and train a pairwise comparator 2:

From T k , select M/N references y with the highest reliability scores α k (y) 4: end for 5: repeat 6:

for each instance x do Membership Update (T k ) 7:

Assign it to T k * , where k * = arg max k β k (x) subject to the regularization constraint 8: end for 9:

Fine-tune the comparator and train a type classifier using T0, . . .

, TK−1 to minimize = co + ty 10:

Assign it to T k where k is its type classification result 12: end for 13:

From T k , select M/N references y with the highest reliability scores α k (y) 15:

end for 16: until convergence or predefined number of iterations Output: Pairwise comparator, type classifier, reference sets R0, . . .

, RK−1 to (6), the comparator loss co can be written as

where R k ⊂ T k is the set of references for the kth chain, α k (y) = x∈T k j q xy j log p xy j is the reliability of a reference y in the kth chain, and β k (x) = y∈R k j q xy j log p xy j is the affinity of an instance x to the references in the kth chain.

Note that β k (x) = − y∈R k D(q xy p xy ) where D is the Kullback-Leibler distance (Cover & Thomas, 2006) .

Second, after fixing the chain membership T k for each chain k, we select references y to maximize the reliability scores α k (y).

These references form R k .

Third, after fixing R 0 , . . .

, R K−1 , we update the chain membership T 0 , . . .

, T K−1 , by assigning each training instance x to the kth chain that maximizes the affinity score β k (x).

The second and third steps are iteratively repeated.

Both steps decrease the same loss co in (9).

The second and third steps are analogous to the centroid rule and the nearest neighbor rule in the Kmeans clustering (Gersho & Gray, 1991) , respectively.

The second step determines representatives in each chain (or cluster), while the third step assigns each instance to an optimal chain according to the affinity.

Furthermore, both steps decrease the same loss alternately.

However, as described in Algorithm 1, we modify this iterative algorithm by including the membership refinement step in lines 10 ∼ 12.

Specifically, we train a K-way type classifier using T 0 , . . .

, T K−1 .

Then, we accept the type classification results to refine T 0 , . . .

, T K−1 .

This refinement is necessary because the type classifier should be used in the test phase to determine the chain of an unseen instance.

Therefore, it is desirable to select the references also after refining the chain membership.

Also, in line 7, if we assign an instance x to maximize β k (x) only, some classes may be assigned too few training instances, leading to data imbalance.

To avoid this, we enforce the regularization constraint so that every class is assigned at least a predefined number of instances.

This regularized membership update is described in Appendix A.

We develop an age estimator based on the proposed order learning.

Order learning is suitable for age estimation, since telling the older one between two people is easier than estimating each person's age directly (Chang et al., 2010; Zhang et al., 2017a) .

It is less difficult to distinguish between a 5-year-old and a 10-year-old than between a 65-yearold and a 70-year-old.

Therefore, in age estimation, we replace the categorization based on the arithmetic difference in (1)∼(3) with that based on the geometric ratio as follows.

which represent 'older,' 'similar,' and 'younger.'

The consistency in (4) is also modified accordingly.

There are 5 reference images for each age class within range [15, 80] in this work (M = 330, N = 66).

Thus, a test image should be compared with 330 references.

However, we develop a twostep approach, which does at most 130 comparisons but performs as good as the method using 330 comparisons.

The two-step estimation is employed in all experiments.

It is described in Appendix B.

We align all facial images using SeetaFaceEngine (Zhang et al., 2014) and resize them into 256 × 256 × 3.

Then, we crop a resized image into 224 × 224 × 3.

For the feature extractors in Figure 2 , we use VGG16 without the FC layers (Simonyan & Zisserman, 2014) .

They yield 512-channel feature vectors.

Then, the vectors are concatenated and input to the ternary classifier, which has three FC layers, yielding 512-, 512-, and 3-channel vectors sequentially.

The 3-channel vector is normalized to the softmax probabilities of the three categories ' , ≈, ≺.' In (10)∼(12), τ age is set to 0.1.

In KCH with K ≥ 2, the type (or chain) of a test image should be determined.

Thus, we design a type classifier, which shares the feature extractor with the comparator.

Similarly to the ternary classifier, the type classifier uses three FC layers, yielding 512-, 512-, and K-channel vectors sequentially.

The comparator and the type classifier are jointly trained.

To initialize the feature extractors, we adopt the VGG16 parameters pre-trained on ImageNet (Deng et al., 2009) .

We randomly initialize all the other layers.

We update the parameters using the Adam optimizer (Kingma & Ba, 2014) .

We set the learning rate to 10 −4 for the first 70 epochs.

Then, we select 5 references for each age class.

Using the selected references, we fine-tune the network with a learning rate of 10 −5 .

We repeat the reference selection and the parameter fine-tuning up to 3 times.

In the case of unsupervised chains, we enforce the regularization constraint (line 7 in Algorithm 1).

By default, for each age, all chains are constrained to be assigned the same number of training images.

If there are L training images of θ-year-olds, the age classes θ in the K chains are assigned L/K images, respectively, according to the affinity scores β k (x) by Algorithm 2 in Appendix A.

MORPH II (Ricanek & Tesafaye, 2006 ) is the most popular age estimation benchmark, containing about 55,000 facial images in the age range [16, 77] .

IMDB-WIKI (Rothe et al., 2018) is another dataset containing about 500,000 celebrity images obtained from IMDB and Wikipedia.

It is sometimes used to pre-train age estimation networks.

Optionally, we also select 150,000 clean data from IMDB-WIKI to pre-train the proposed pairwise comparator.

Although several facial age datasets are available, most are biased to specific ethnic groups or genders.

Data unbalance restricts the usability and degrades the generalization performance.

Thus, we form a 'balanced dataset' from MORPH II, AFAD (Niu et al., 2016) , and UTK (Zhang et al., 2017b) .

Table 1 shows how the balanced dataset is organized.

Before sampling images from MORPH II, AFAD, and UTK, we rectify inconsistent labels by following the strategy in Yip et al. (2018) .

For each combination of gender in {female, male} and ethnic group in {African, Asian, European}, we sample about 6,000 images.

Also, during the sampling, we attempt to make the age distribution as For performance assessment, we calculate the mean absolute error (MAE) (Lanitis et al., 2004 ) and the cumulative score (CS) (Geng et al., 2006) .

MAE is the average absolute error between predicted and ground-truth ages.

Given a tolerance level l, CS computes the percentage of test images whose absolute errors are less than or equal to l. In this work, l is fixed to 5, as done in Chang et al. Table 2 compares the proposed algorithm (1CH) with conventional algorithms on MORPH II.

As evaluation protocols for MORPH II, we use four different settings, including the 5-fold subjectexclusive (SE) and the 5-fold random split (RS) (Chang et al., 2010; Guo & Wang, 2012) .

Appendix C.1 describes these four settings in detail and provides an extended version of Table 2 .

OHRank, OR-CNN, and Ranking-CNN are all based on ordinal regression.

OHRank uses traditional features, yielding relatively poor performances, whereas OR-CNN and Ranking-CNN use CNN features.

DEX, DRFs, MO-CNN, MV, and BridgeNet employ VGG16 as backbone networks.

Among them, MV and BridgeNet achieve the state-of-the-art results, by employing the mean-variance loss and the gating networks, respectively.

The proposed algorithm outperforms these algorithms in setting C, which is the most challenging task.

Furthermore, in terms of CS, the proposed algorithm yields the best performances in all four settings.

These outstanding performances indicate that order learning is an effective approach to age estimation.

In Table 3 , we analyze the performances of the proposed algorithm on the balanced dataset according to the number of hypothesized chains.

We also implement and train the state-of-the-art MV on the balanced dataset and provide its results using supervised chains.

Let us first analyze the performances of the proposed algorithm using 'supervised' chains.

The MAE and CS scores on the balanced dataset are worse than those on MORPH II, since the balanced dataset contains more diverse data and thus is more challenging.

By processing facial images separately according to the genders (2CH), the proposed algorithm reduces MAE by 0.05 and improves CS by 0.2% in comparison with 1CH.

Similar improvements are obtained by 3CH or 6CH, which consider the ethnic groups only or both gender and ethnic groups, respectively.

In contrast, in the case of MV, multi-chain hypotheses sometimes degrade the performances; e.g., MV (6CH) yields a lower CS than MV (1CH).

Regardless of the number of chains, the proposed algorithm trains a single comparator but uses a different set of references for each chain.

The comparator is a ternary classifier.

In contrast, MV (6CH) should train six different age estimators, each of which is a 66-way classifier, to handle different chains.

Thus, their training is more challenging than that of the single ternary classifier.

Note that, for the multi-chain hypotheses, the proposed algorithm first identifies the chain of a test image using the type classifiers, whose accuracies are about 98%.

In Table 3 , these Comparison results are color-coded.

Cyan, yellow, and magenta mean that the test subject is older than ( ), similar to (≈), and younger than (≺) a reference.

The age is estimated correctly as 22.

type classifiers are used to obtain the results of the proposed algorithm, whereas the ground-truth gender and ethnic group of each test image are used for MV.

Figure 4 shows how to estimate an age in 6CH.

In this test, the subject is a 22-year-old Asian male.

He is compared with the references who are also Asian males.

Using the comparison results, the age is correctly estimated as 22 by the MC rule in (5).

Table 4 lists the MAE results for each test chain.

Europeans yield poorer MAEs than Africans or Asians.

However, this is not due to inherent differences between ethnic groups.

It is rather caused by differences in image qualities.

As listed in Table 1 , more European faces are sampled from UTK.

The UTK faces were crawled from the Internet and their qualities are relatively low.

Also, from the cross-chain test results using 6CH, some observations can be made:

• Except for the As-F test chain, the lowest MAE is achieved by the references in the same chain.

• Eu-M and Eu-F are mutually compatible.

For Eu-M, the second best performance is obtained by the Eu-F references, and vice versa.

On the other hand, some chains, such as Af-M and Eu-F, are less compatible for the purpose of the proposed age estimation.

Table 3 also includes the performances of the proposed algorithm using 'unsupervised' chains.

The unsupervised algorithm outperforms the supervised one, which indicates that the gender or ethnic group is not the best information to divide data for age estimation.

As in the supervised case, 2CH, 3CH, and 6CH yield similar performances, which means that two chains are enough for the balanced set.

Compared with MV (1CH), the unsupervised algorithm (2CH) improves the performances significantly, by 0.33 in terms of MAE and 4.1% in terms of CS.

Figure 5 shows how training images are divided into two chains in the unsupervised 2CH.

During the membership update, for each age, each chain is regularized to include at least a certain percentage (κ) of the training images.

In the default mode, the two chains are assigned the same number of images with κ = 50%.

However, Appendix C.3 shows that the performance is not very sensitive to κ.

At κ = 10%, MAE = 4.17 and CS = 73.7%.

From Figure 5 , we observe • The division of the chains is not clearly related to genders or ethnic groups.

Regardless of genders or ethnic groups, about half of the images are assigned to chain 1 and the others to chain 2.

• At κ = 10%, chain 1 mostly consists of middle ages, while chain 2 of 10s, 20s, 60s, and 70s.

• At κ = 50%, there is no such strong age-dependent tendency.

But, for some combinations of gender, ethnic group, and age band, it is not equal division.

For example, for Asian females, a majority of 40s are assigned to chain 1 but a majority of 50s and 60s are assigned to chain 2.

The unsupervised algorithm is designed to divide instances into multiple clusters when gender and ethnic group information is unavailable.

As shown in Appendix C.3, different κ's yield various clustering results.

Surprisingly, these different clusters still outperform the supervised algorithm.

For example, at κ = 10%, let us consider the age band of 20s and 30s.

If the references in chain 2 are used to estimate the ages of people in chain 1, the average error is 4.6 years.

On the contrary, if the references in chain 1 are used for chain 2, the average error is −5.4 years.

These opposite biases mean that people in chain 1 tend to look older than those in chain 2.

These 'looking-older' people in 20s and 30s compose the blue cluster (chain 1) together with most people in 40s and 50s in Figure 5 .

In this case, 'looking-older' people in 20s and 30s are separated from 'looking-younger' ones by the unsupervised algorithm.

This is more effective than the gender-based or ethnic-group-based division of the supervised algorithm.

Appendix C presents more results on age estimation.

Order learning was proposed in this work.

In order learning, classes form an ordered set, and each class represents object instances of the same rank.

Its goal is to determine the order graph of classes and classify a test instance into one of the classes.

To this end, we designed the pairwise comparator to learn ordering relationships between instances.

We then decided the class of an instance by comparing it with reference instances in the same chain and maximizing the consistency among the comparison results.

For age estimation, it was shown that the proposed algorithm yields the stateof-the-art performance even in the case of the single-chain hypothesis.

The performance is further improved when the order graph is divided into multiple disjoint chains.

In this paper, we assumed that the order graph is composed of disjoint chains.

However, there are more complicated graphs, e.g. Figure 1 (a), than disjoint chains.

For example, it is hard to recognize an infant's sex from its facial image (Porter et al., 1984) .

But, after puberty, male and female take divergent paths.

This can be reflected by an order graph, which consists of two chains sharing common nodes up to a certain age.

It is an open problem to generalize order learning to find an optimal order graph, which is not restricted to disjoint chains.

During the chain membership update in Algorithm 1, we assign an instance x to chain k to maximize β k (x) subject to the regularization constraint.

As mentioned in Section 4.1, in age estimation, this regularization is enforced for each age.

Let X denote the set of θ-year-olds for a certain θ.

Also, let K = {0, 1, . . .

, K − 1} be the set of chains.

Suppose that we should assign at least a certain number (L) of instances in X to each chain.

This is done by calling RegularAssign(K, X , L) in Algorithm 2, which is a recursive function.

Algorithm 2 yields the membership function c(x) as output.

For example, c(x) = 1 means that x belongs to chain 1.

Input: K = set of chains, X = set of instances, and L = minimum number 1: for each k ∈ K do Initialize chains 2:

X k = ∅ 3: end for 4: for each x ∈ X do Irregular partitioning 5:

c(x) = arg max k∈K β k (x) 6: X c(x) = X c(x) ∪ {x} 7: end for 8: km = arg min k∈K |X k | Chain of the minimum size 9: if |X km | ≥ L then 10: return 11: else 12:

while |X km | < L do Increase X km 14:

x = maxx∈X β km (x) 15:

X km = X km ∪ {x } 17:

end while 18:

B TWO-STEP ESTIMATION There are 5 reference images for each age within range [15, 80] in this work.

Thus, for the age estimation of a test image using the MC rule in (5), the test image should be compared with M = 330 reference images.

However, we reduce the number of comparisons using a two-step approach.

First, the test image is compared with the 35 references of ages 15, 25, . . .

, 75 only, and a rough age estimateθ 1 is obtained using the MC rule.

Second, it is compared with the 105 references of all ages within [θ 1 − 10,θ 1 + 10], and the final estimateθ 2 is obtained.

Since there are at least 10 common references in the first and second steps, the two-step estimation requires at most 130 comparisons.

• Setting A: 5,492 images of Europeans are randomly selected and then divided into training and testing sets with ratio 8:2 (Chang et al., 2011 ).

• Setting B: About 21,000 images are randomly selected, while restricting the ratio between Africans and Europeans to 1:1 and that between females and males to 1:3.

They are divided into three subsets (S1, S2, S3).

The training and testing are done under two subsettings (Guo & Mu, 2011) .

-(B1) training on S1, testing on S2 + S3 -(B2) training on S2, testing on S1 + S3

• Setting C (SE): The entire dataset is randomly split into five folds, subject to the constraint that the same person's images should belong to only one fold, and the 5-fold crossvalidation is performed.

• Setting D (RS): The entire dataset is randomly split into five folds without any constraint, and the 5-fold cross-validation is performed.

Table 5 is an extended version of Table 2 .

It includes the results of more conventional algorithms.

We assess the proposed age estimator (1CH) on the FG-NET database (Panis et al., 2016) .

FG-NET is a relatively small dataset, composed of 1,002 facial images of 82 subjects.

Ages range from 0 to 69.

For FG-NET, the leave one person out (LOPO) approach is often used for evaluation.

In other words, to perform tests on each subject, an estimator is trained using the remaining 81 subjects.

Then, the results are averaged over all 82 subjects.

In order to assess the generalization performance, we do not retrain the comparator on the FG-NET data.

Instead, we fix the comparator trained on the balanced dataset and just select references from the remaining subjects' faces in each LOPO test.

For the comparator, the arithmetic scheme in (1)∼(3) is tested as well as the default geometric scheme in (10)∼(12).

For comparison, MV (Pan et al., 2018 ) is tested, but it is trained for each LOPO test. [15, 69] , the proposed algorithm outperforms MV, even though the comparator is not retrained.

These results indicate that the comparator generalizes well to unseen data, as long as the training images cover a desired age range.

Also, note that the geometric scheme provides better performances than the arithmetic scheme.

Figure 6 compares MAEs according to a test age.

Again, within the covered range [15, 69] , the proposed algorithm significantly outperforms MV especially when test subjects are older than 45.

The ordering relationship between two instances can be categorized via the arithmetic scheme in (1)∼(3) using a threshold τ or the geometric scheme in (10)∼(12) using a threshold τ age .

Table 8 lists the performances of the proposed algorithm (1CH) according to these thresholds.

We see that the geometric scheme outperforms the arithmetic scheme in general.

The best performance is achieved with τ age = 0.1, which is used in all experiments in the main paper.

Note that the scores are poorer than those in Table 3 , since the comparator is trained for a smaller number of epochs to facilitate this test.

At τ age = 0.1, two teenagers are declared to be not 'similar to' each other if their age difference is larger than about 1.

Also, two forties are not 'similar' if the age difference is larger than about 5.

C.5 PERFORMANCE ACCORDING TO NUMBER OF REFERENCES Table 9 : The performances of the proposed algorithm (supervised) on the balanced dataset according to the number of references for each age class (M/N ).

In general, the performances get better with more references.

However, the performances are not very sensitive to M/N .

They saturate when M/N ≥ 5.

Therefore, we set M/N = 5 in this work.

Figure 8 : All reference images in the supervised 6CH.

For some ages in certain chains, the balanced dataset includes less than 5 faces.

In such cases, there are less than 5 references.

<|TLDR|>

@highlight

The notion of order learning is proposed and it is applied to regression problems in computer vision