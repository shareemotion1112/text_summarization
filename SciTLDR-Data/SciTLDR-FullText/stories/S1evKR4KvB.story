Extreme Classification Methods have become of paramount importance, particularly for Information Retrieval (IR) problems, owing to the development of smart algorithms that are scalable to industry challenges.

One of the prime class of models that aim to solve the memory and speed challenge of extreme multi-label learning is Group Testing.

Multi-label Group Testing (MLGT) methods construct label groups by grouping original labels either randomly or based on some similarity and then train smaller classifiers to first predict the groups and then recover the original label vectors.

Recently, a novel approach called MACH (Merged Average Classifiers via Hashing) was proposed which projects the huge label vectors to a small and manageable count-min sketch (CMS) matrix and then learns to predict this matrix to recover the original prediction probabilities.

Thereby, the model memory scales O(logK) for K classes.

MACH is a simple algorithm which works exceptionally well in practice.

Despite this simplicity of MACH, there is a big gap between the theoretical understanding of the trade-offs with MACH.

In this paper we fill this gap.

Leveraging the theory of count-min sketch we provide precise quantification of the memory-identifiablity tradeoffs.

We extend the theory to the case of multi-label classification, where the dependencies make the estimators hard to calculate in closed forms.

To mitigate this issue, we propose novel quadratic approximation using the Inclusion-Exclusion Principle.

Our estimator has significantly lower reconstruction error than the typical CMS estimator across various values of number of classes K, label sparsity and compression ratio.

Extreme Classification has taken center-stage of Data Mining and Information Retrieval research in the past few years (Zamani et al., 2018; Prabhu et al., 2018b; Jain et al., 2019; Choromanska & Langford, 2015) .

It refers to the vanilla multiclass and multilabel classification problems where the number of classes K is significantly large.

A large number of classes K brings a new set of computational and memory challenges in training and deploying classifiers.

There have been several paradigms of models that tackle the scale challenge of Extreme Classification like 1-vs-all methods (Prabhu et al., 2018b; Jain et al., 2019; Babbar & Sch??lkopf, 2017) , tree based methods (Prabhu et al., 2018a; Jain et al., 2016) , embedding models (Nigam et al., 2019; Bhatia et al., 2015) , etc. (as noted on the popular Extreme Classification Repository).

One of the recent approaches proposed to alleviate the scale challenge of Multilabel Classification is Group Testing (Ubaru & Mazumdar, 2017; Ubaru et al., 2016; Vem et al., 2017) .

In this method, all labels are grouped randomly into m groups/clusters.

Each label may go into more than one group.

We first train a classifier that predicts which of these clusters the input belongs to (treating each cluster as a separate label in a multilabel setting).

For any given input, we first predict the clusters into which the true labels of the input may have been pooled.

We can then identify all the true labels by taking an intersection over the inverted clusters.

This approach suffers from a critical problem that even tree based approaches have, i.e., hard assignment of clusters.

Since the recovery of true labels depends solely on hard-prediction of clusters, a mistake in the cluster prediction can cost us dearly in the final label prediction.

Also, since the labels are pooled randomly, each individual meta-classifier is a weak and noisy one.

In a recent development, Merged Average Classifiers via Hashing (MACH) (Medini et al., 2019) was proposed that alleviates the hard-prediction problem in Group Testing methods by identifying the best labels based on the sum of prediction probabilities of the respective groups for a given input.

In the hindsight, MACH subtly learns to predict a count-min sketch (CMS) (Cormode & Muthukrishnan, 2005 ) matrix of the original probability vector.

For the case of multiclass classification (every input having just a single label unlike multilabel), MACH proposes an unbiased estimator to recover the original K dimensional probability vector from the predicted CMS matrix.

Multiclass classification naturally fits into the count-min sketch setting as no two labels can appear simultaneously for a given input.

But the proposed theory does not naturally extend to multilabel learning.

Further, the variance and error bounds for multiclass classification rely heavily on the choice of number of hash tables and the size of each hash table.

That aspect has not been explored in prior work.

Our Contributions: In this work we broadly make the following contributions: 1) We revisit MACH with a thorough analysis of proposed reconstruction estimator for multiclass learning.

In particular, we prove that the variance of estimation is inversely proportional to the product of product of number of hash tables and size of each hash table (in theorem 2).

2) We also obtain a lower bound on hash table hyperparametrs given a tolerance to prediction error (in Theorems 4 and 5).

3) We propose a novel reconstruction estimator for the case of multilabel learning using InclusionExclusion principle (in theorem 6).

This estimator comes out as a solution to a quadratic equation (hence we code-name it as 'quadratic estimator').

4) We simulate multilabel learning setting by generating K dimensional probability vectors and their proxy CMS measurements.

We then reconstruct the probability vector using both the mean estimator and the quadratic estimator and show that the reconstruction Mean-Squared Error (MSE) is significantly lower for the new estimator.

Count-Min Sketch: Count-Min Sketch (CMS) (Cormode & Muthukrishnan, 2005) was proposed to solve the frequency counting problem in large streaming setting.

Assume that we have an infinite stream of elements e 1 , e 2 , e 3 , ... coming in.

Each of these elements can take any value between K distinct ones.

Here, K is very large and we cannot afford to store an array of counts to store every element's frequency (limited memory setting).

We need a sub-linear efficient data structure from which we can retrieve the frequency of every element.

In Count-Min Sketch (Cormode & Muthukrishnan, 2005) , we basically assign O(log K) 'signatures' to each class using 2-universal hash functions.

We use O(log K) different hash functions H 1 , H 2 , H 3 , ..., H O(log K) , each mapping any class i to a small range of buckets B << K, i.e., H j (i) ??? {0, 1, 2, ..., B}. We maintain a counting-matrix C of order O(log K) * B. If we encounter class i in the stream of classes, we increment the counts in cells H 1 (i), H 2 (i)....., H O(log K) (i).

It is easy to notice that there will be collisions of classes into these counting cells.

Hence, the counts for a class in respective cells could be over-estimates of the true count.

During inference, we want to know the frequency of a particular element say a 1 .

We simply go to all the cells where a 1 is mapped to.

Each cell gives and over-estimated value of the original frequency of a 1 .

To reduce the offset of estimation, the algorithm proposes to take the minimum of all the estimates as the approximate frequency, i.e., n approx (

An example illustration of CMS is shown in figure 1.

Connecting CMS and Extreme Classification:

Given a data instance x, a vanilla classifier outputs the probabilities p i , i ??? {1, 2, ..., K}. We want to essentially compress the information of these K numbers to log K,i.e., we can only keep track of log K = BR measurements.

Ideally, without any assumption, we cannot compress the information in K numbers to anything less than O(K), if we want to retain all information.

However, in classification, the most informative quantity is the identity of arg max p i .

If we can identify a scheme that can recover the high probability classes from smaller measurement vector, we can train a small-classifier to map an input to these measurements instead of the big classifier.

The foremost class of models to accomplish this task are Encoder and Decoder based models like Compressive Sensing (Baraniuk, 2007) .

The connection between compressed sensing and extreme classification was identified in prior works (Hsu et al., 2009; Dietterich & Bakiri, 1995) .

We provide an intuitive explanation of why compressed sensing or any other sketching algorithm does work like count-min sketch in the appendix A.

Figure 2: Schematic diagram of MACH.

Both the input and the label vector are independently hashed R times (label vector is hashed from K to B, K being number of classes and B being number of buckets in each of the R hash tables).

Small models are then trained in parallel.

MACH (Medini et al., 2019 ) is a new paradigm for extreme classification that uses universal hashing to reduce memory and computations.

MACH randomly merges K classes into B meta-classes or buckets (B K).

We then runs any offthe shelf classifier (typically simple feed forward neural networks) to predict the meta classes.

This process is repeated R number of times, changing the hash function each time (or by simply changing the random seed of the same hash function, to induce a different random pooling each time).

During prediction, MACH aggregates the output from each of the R small meta classifiers to retrieve the best class.

In the schema shown in figure 2, the input is assumed to be a large dimensional sparse vector.

In order to reduce model size from both ends (input and output), the sparse input can also be feature hashed (Weinberger et al., 2009 ) to a manageable dimension.

Please note that the theoretical analysis of MACH is agnostic to the input feature hashing.

We are only concerned with retrieving the most relevant labels from the meta-class predictions.

The subsequent sections formalize the algorithm and quantify the mean, variance, error bounds and hyper-parameter bounds.

We begin with emphasizing that MACH does not assume any dependence among the classes.

This is a fairly strong assumption because often in extreme classification, the labels have strong correlations.

More so, this assumption is intrinsically violated in the case of multilabel learning.

Nevertheless, MACH works extremely well in practice, particularly at industry scale challenges.

Let there be K classes originally.

We'll hash them to B meta-classes using a universal hash function.

We repeat this process R times each with a different hash function (can be obtained by simply changing the random seed each time).

We only have an R * B matrix that holds all information about the original probability vector of K dimensions (R * B K).

Typical classification algorithms model the probability P r(y = i|x) = p i where i ??? {0, 1, 2...K ??? 1} .

With MACH, we bypass the hassle of training a huge last layer by instead modelling P r(y = b|x) = P j b for every hash function h j , where b ??? {0, 1, 2, ..., B ??? 1} and j ??? {0, 1, 2, ..., R ??? 1}. During prediction, we sought to recover the K vector from P j b matrix using an unbiased estimator as shown in subsequent sections.

P j hj (i) stands for the probability of the bin (meta-class) that i th class is hashed into in j th repetition.

Our goal is to obtain an unbiased estimator of p i in terms of {P

From here on, the analysis diverges between Multiclass and Multilabel classification problems.

We have

With the above equations, given the R classifier models, an unbiased estimator of p i is: Theorem 1.

Proof:

Proof for this theorem has been given in (Medini et al., 2019) .

For clarity and coherence, we show the proof again here.

For any j, we can always write

where 1 hj (k)=hj (i) is an indicator random variable (generically denoted by I k from here on) suggesting whether class k has been hashed into the same bin as class i using hash function j. Since the hash function is universal, the expected value of the indicator is 1 B (each class will uniformly be binned into one of the B buckets).

Thus

This is because the expression k =i p k = 1???p i as the total probability sum up to one.

Simplifying, we get

.

Using linearity of expectation and the fact that E(P j hj (i) ) = E(P k hj (i) ) for any j = k, it is not difficult to see that this value is also equal to

Proof: Using the known result V ar(aX + b) = a 2 V ar(X) and the fact that variance accumulates over sum of i.i.d random variables, we can write

We first need to get V ar(P j hj (i) ).

From eqn.

3,

Hence,

It's easy to see

Hence, by merging eqns.

5 and 6, we get

We can observe that larger the original probability p i , lower the variance of estimation which suggests that the higher probabilities are retained with high certainty and the lower probabilities are prone to noise.

Since we only care for the correct prediction of the best class, we can offset the noise by increasing R.

is also the computational complexity of prediction.

With MACH, the memory complexity is O(BRd) and the computational complexity is O(BRd + KR) (including inference).

To obtain significant savings, we want BR to be significantly smaller than K. We next show that BR ??? O(log K) is sufficient for uniquely identifying the final class with high probability.

Also, we need to tune the two knobs R and B for optimal performance on recovering the original probabilities.

The subsequent theorems facilitate the prior knowledge of reasonable values of R and B based on our reconstruction error tolerance.

In (Medini et al., 2019) , the following theorem has been proven Theorem 3.

For any B, R = log

, guarantees that all pairs of classes c i and c j are distinguishable from each other with probability greater than 1 ??? ?? 1 .

The above theorem specifies a bound such that no two pair of classes end up in the same bucket on all R hash functions.

While this is simple and intuitive, it does not take into account the ease of classification.

To be precise, when the difference between the probability of best class and the 2 nd best class is low (predictions are spurious), it is much harder to identify the best class as oppposed to when the difference is higher.

Theorem 3 is completely agnostic to such considerations.

Hence, the next theorems quantifies the requirements on R, B based on our tolerance to recovery error between p i andp i and also the ease of prediction (given by the difference between the p i and p j where i and j are the two best classes respectively).

for any random variable X. For our proposed unbiased estimator in theorem 1, we have

For a large enough B, B???1 B ??? 1.

Hence, we get the desired result

If the best class i * has p i * > ?? and we primarily care for recovering p i * with high probability, then we have RB > and classes i and j are the first and second best respectively (p i > p j > p k f or every k = i, j),

Hence, based on the previous two theorems, we can get a reasonable estimate of what bucket size B should we choose and how many models that we need to train in parallel.

The major difference between multi-class and multi-label classification from an analysis perspective is that eqn.

1 does not apply anymore.

Hence, all the subsequent derivations do not apply in the case of multi-label classification.

In the following theorems, we'll derive an approximate estimator using inclusion-exclusion principle to recover original probability vectors from MACH measurements for the case of multi-label classification.

Each p i independently takes a value in [0, 1].

If we do not assume any relation between p i , it would be very difficult to derive an estimator.

The most realistic assumption on the probability vectors is sparsity.

Most real datasets have only few labels per sample even when the number of classes K is huge.

For the purpose of analysis, we will assume that

where V is the average of number of active labels per input.

Theorem 6.

Proof: P j hj (i) is the probability of union of all classes that have been hashed to bin h j (i) in j th hash function.

Hence, using inclusion-exclusion principle, it can be written as

Since all classes are independent of each other, we have

Aggregating similar terms, we get

In typical multilabel dataset, K runs into the order of millions where B is a few thousands.

If we ignore all terms with B in denominator, we essentially end up with a plain mean estimator

hj (i) ).

We ideally want to use all terms but it is very cumbersome to analyze the summation (please note that the summation doesn't simplify to exponential as we have the clause k j = k l in each summation).

In our case, we empirically show later on that even by limiting the expression to first order summation (ignore all terms B 2 or higher powers of B in denominator), we get a much better estimator for true probability.

We can simplify the above expression into

Solving for p i , we get our desired result

Unfortunately, proposing an unbiased estimator using the above result is hard.

One intuitive estimator that can potentially work isp i =

Using Jensen's inequality (specifically, E[

Hence, E[p i ]

??? p i and we do not have an unbiased estimator.

Nevertheless, the next section details simulation experiments that corroborate that our proposed estimator for multilabel classification has much lower mean-squared-error (MSE) than a plain mean estimator.

To simulate the setup for multi-label MACH, we perform the following steps:

??? Choose a base prob ??? (0, 1] which says how confident the prediction in the original probability vector is.

??? Initialize a K dimensional vector p orig = (p 1 , p 2 , ...., p K ) with all zeros.

We then implant the value base prob in int( V base prob ) number of random locations.

We now have a vector p orig which obeys p i = V .

??? Generate 1000 samples of K dimensional label vectors where each dimension i is a Bernoulli random variable with probability p i .

These sample labels are realizations of p orig.

??? Merge each sample label vector into B dimensional binary labels where a bucket b is an OR over the constituent classes {i : h j (i) = b}. We repeat this step for R different hash functions ,i.e., for all j ??? 1, 2, ..., R.

??? For each of R repetitions, calculate the mean of the respective B dimensional labels to get

??? Reconstruct p approx using theorem 6 and {P j : j = 1, 2, .., R}.

??? Calculate L2-norm of p orig ??? p approx ??? Repeat all above steps for 10000 times (generating a different p orig each time) and report the average L2-norm from the last step (it serves as the reconstruction MSE, lower the better).

Following the above steps, we show the comparison of our proposed quadratic estimator in theorem 6 against the plain mean estimator by varying the values of K, B, V and base prob in figure 3 .

We can infer the following insights from the plots :

??? As K increases, the MSE grows.

This is expected because the reconstructed vector has a small non-zero probability for many of the K classes and this induces noise and hence MSE grows.

But the top classes are still retrieved with high certainty.

??? For any K, V, base prob, the MSE decreases when B increases which is expected (fewer collisions of classes and hence less noisier predictions).

As the MSE gets lower, the gains from the square-root estimator are also low.

This is good because in scenarios where B and R are small, we can do much better recovery using the proposed estimator.

??? For any K, B, base prob the MSE increases with V .

This is again natural because larger V induces more 'true' class collisions and hence the retrieval becomes fuzzy.

??? For any K, B, V the MSE decreases with base prob, albeit with much little difference than previous cases.

This is interesting because a high base prob means that we have few but highly confident 'true' classes among K. On the other hand, lower base prob indicates that 'true' classes are scattered among a larger subset among K classes.

Yet, MACH recovers the original probabilities with commendably low MSE.

Varying B for K = 10000

Varying base prob for K = 10000

Varying B for K = 100000 Varying V for K = 100000 Varying base prob for K = 100000

Varying B for K = 1000000 Varying B for K = 1000000 Varying base prob for K = 1000000

Figure 3: Reconstruction Error (MSE) comparison between 1) vanilla mean estimator (plotted in magenta) and 2) proposed square-root estimator (plotted in green); for various configurations of K,B and V. The value of K varies as 10000, 100000, 1000000 for the 1 st , 2 nd and 3 rd rows respectively.

In each row, the first plot fixes V, base prob and compares various values of B. The 2 nd plot fixes B, base prob and compares different values of B. The 3 rd one fixes B, V and compares different values of base prob.

In all cases, we notice that the square-root estimator is consistently and significantly lower in MSE than the corresponding mean estimator.

We perform a rigorous theoretical analysis of using Count-Min-Sketch for Extreme Classification and come up with error bounds and hyper-parameter constraints.

We identify a critical shortcoming of reconstruction estimators proposed in prior research.

We overcome the shortcoming by treating each bucket in a hash table as a union of merged original classes.

Using inclusion-exclusion principle and a controlled label sparsity assumption, we come up with an approximate estimator to reconstruct original probability vector from the predicted Count-Min Sketch measurements.

Our new estimator has significantly lower reconstruction MSE than the prior estimator.

Why not Compressive Sensing or Count-Sketch?

The measurements in Compressive Sensing are not a probability distribution but rather a few linear combinations of original probabilities.

Imagine a set of classes {cats, dogs, cars, trucks}. Suppose we want to train a classifier that predicts a compressed distribution of classes like {0.6 * cars + 0.4 * cats, 0.5 * dogs + 0.5 * trucks}.

There is no intuitive sense to these classes and we cannot train a model using softmax-loss which has been proven to work the best for classification.

We can only attempt to train a regression model to minimize the norm(like L 1 -norm or L 2 -norm) between the projections of true K-vector and the predicted K-vectors(like in the case of (Hsu et al., 2009)

@highlight

How to estimate original probability vector for millions of classes from count-min sketch measurements -  a theoretical and practical setup.