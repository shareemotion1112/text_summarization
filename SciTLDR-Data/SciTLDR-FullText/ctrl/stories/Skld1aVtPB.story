This work views neural networks as data generating systems and applies anomalous pattern detection techniques on that data in order to detect when a network is processing a group of anomalous inputs.

Detecting anomalies is a critical component for multiple machine learning problems including detecting the presence of adversarial noise added to inputs.

More broadly, this work is a step towards giving neural networks the ability to detect groups of out-of-distribution samples.

This work introduces ``Subset Scanning methods from the anomalous pattern detection domain to the task of detecting anomalous inputs to neural networks.

Subset Scanning allows us to answer the question: "``Which subset of inputs have larger-than-expected activations at which subset of nodes?"  Framing the adversarial detection problem this way allows us to identify systematic patterns in the activation space that span multiple adversarially noised images.

Such images are ``"weird together".

Leveraging this common anomalous pattern, we show increased detection power as the proportion of noised images increases in a test set.

Detection power and accuracy results are provided for targeted adversarial noise added to CIFAR-10 images on a 20-layer ResNet using the Basic Iterative Method attack.

The vast majority of data in the world can be thought of as created by unknown, and possibly complex, normal behavior of data generating systems.

But what happens when data is generated by an alternative system instead?

Fraudulent records, disease outbreaks, cancerous cells on pathology slides, or adversarial noised images are all examples of data that does not come from the original, normal system.

These are the interesting data points worth studying.

The goal of anomalous pattern detection is to quantify, detect, and characterize the data that are generated under these alternative systems.

Furthermore, subset scanning extends these ideas to consider groups of data records that may only appear anomalous when viewed together (as a subset) due to the assumption that they were generated by the same alternative system.

Neural networks may be viewed as one of these data generating systems.

The activations are a source of high-dimensional data that can be mined to discover anomalous patterns.

Mining activation data has implications for interpretable machine learning as well as more objective tasks such as detecting groups of out-of-distribution samples.

This paper addresses the question: "Which of the exponentially many subset of inputs (images) have higher-than-expected activations at which of the exponentially many subset of nodes in a hidden layer of a neural network?" We treat this scenario as a search problem with the goal of finding a "high-scoring" subset of images × nodes by efficiently maximizing nonparametric scan statistics in the activation space of neural networks.

The primary contribution of this work is to demonstrate that nonparametric scan statistics, efficiently optimized over node-activations × multiple inputs (images), are able to quantify the anomalousness of a subset of those inputs (images) into a real-valued "score".

This definition of anomalousness is with respect to a set of clean "background" inputs (images) that are assumed to generate normal or expected patterns in the activation space of the network.

Our method measures the deviance between the activations of a subset of inputs (images) under evaluation and the activations generated by the background inputs.

The challenging aspect of measuring deviances in the activation space of neural networks is dealing with high-dimensional data, on the order of the number of nodes in a hidden layer × the number of inputs (images) under consideration.

Therefore, the measure of anomalousness must be effective in capturing systematic (yet potentially subtle) deviances in a high-dimensional subspace and be computationally tractable.

Subset scanning meets both of these requirements (see Section 2).

The reward for addressing this difficult problem is an unsupervised, anomalous-input detector that can be applied to any input and to any type of neural network architecture.

Neural networks universally rely on their activation space to encode the features of their inputs and therefore quantifying deviations from expected behavior in the activation space has broad appeal and potential beyond detecting anomalous patterns in groups of images.

Furthermore, an additional output of subset scanning not fully explored in this paper is the subset of nodes at which the subset of inputs (images) had the higher-than-expected activations.

These may be used to characterize the anomalous pattern that is affecting the inputs.

The second contribution of this work focuses on detection of targeted adversarial noise added to inputs in order to change the labels to a target class Szegedy et al. (2013) ; Goodfellow et al. (2014) ; .

Our critical insight to this problem is the ability to detect the presence of noise (i.e. an anomalous pattern) across multiple images simultaneously.

This view is grounded by the idea that targeted attacks will create a subtle, but systematic, anomalous pattern of activations across multiple noised images.

Therefore, during a realistic attack on a machine learning system, we expect a subset of the inputs to be anomalous together by sharing higher-than-expected activations at similar nodes.

Empirical results show that detection power drastically increases when targeted images compose 8%-10% of the data under evaluation.

Detection power is near 1 when the proportion reaches 14%.

In summary, this is the first work to apply subset scanning techniques to data generated from neural networks in order to detect anomalous patterns of activations that span multiple inputs (images).

To the best of our knowledge, this is the first topic to address adversarial noise detection by considering images as a group rather than individually.

Subset scanning treats pattern detection as a search for the "most anomalous" subset of observations in the data where anomalousness is quantified by a scoring function, F (S) (typically a loglikelihood ratio).

Therefore, we wish to efficiently identify S * = arg max S F (S) over all relevant subsets of the data S. Subset scanning has been shown to succeed where other heuristic approaches may fail (Neill, 2012) .

"Top-down" methods look for globally interesting patterns and then identifies sub-partitions to find smaller anomalous groups of records.

These approaches may fail when the true anomaly is not evident from global aggregates.

Similarly, "Bottom-up" methods look for individually anomalous data points and attempt to aggregate them into clusters.

These methods may fail when the pattern is only evident by evaluating a group of data points collectively.

Treating the detection problem as a subset scan has desirable statistical properties for maximizing detection power but the exhaustive search is infeasible for even moderately sized data sets.

However, a large class of scoring functions satisfy the "Linear Time Subset Scanning" (LTSS) property which allows for exact, efficient maximization over all subsets of data without requiring an exhaustive search Neill (2012) .

The following sub-sections highlight a class of functions that satisfy LTSS and describe how the efficient maximization process works for scanning over activations.

This work uses nonparametric scan statistics (NPSS) that have been used in other pattern detection methods Neill & Lingwall (2007); McFowland III et al. (2013); McFowland et al. (2018) ; Chen & Neill (2014) .

These scoring functions make no parametric assumptions about how activations are distributed at a node in a neural network.

They do, however, require baseline or background data to inform their data distribution under the null hypothesis H 0 of no anomaly present.

The first step of NPSS is to compute empirical p-values for the evaluation input (e.g. images potentially affected with adversarial noise) by comparing it to the empirical baseline distribution generated from the background inputs that are "natural" inputs known to be free of an anomalous pattern.

NPSS then searches for subsets of data S in the evaluation inputs that contain the most evidence for Given D, our N ×J matrix of activations for each evaluation image at each network node, and D H0 , our corresponding M × J matrix of activations for each of the background images, we can obtain an empirical p-value for each A ij : a means to measure of how anomalous the activation value of (potentially contaminated) image X i is at node O j .

This p-value p ij is the proportion of activations from the background images, A H0 zj , that are larger than the activation from the evaluation images A ij at node O j .

We note that McFowland III et al. (2013) extend this notion to p-value ranges such that p ij is uniformly distributed between p min ij and p max ij .

This current work makes a simplifying assumption here to only consider a range by its upper bound, p max ij

The matrix of activations A ij is now converted into a matrix of p-values P ij .

Intuitively, if an evaluation image X i is "natural" (its activations are drawn from the same distribution as the baseline images) then few of the p-values generated by image X i across the network nodes-will be extreme.

The key assumption for subset scanning approaches is that under the alternative hypothesis of an anomaly present in the activation data then at least some subset of the activations, for the effected subset of images, will systematically appear extreme.

The goal is to identify this "high-scoring" subset through an efficient search procedure that maximizes a nonparametric scan statistic.

The matrix of p-values P ij from evaluation images is processed by a nonparametric scan statistic in order to identify the subset of evaluation images S X ⊆ D whose activations at some subset of nodes S O ⊆ O maximizes the scoring function max S=S X ×S O F (S), where S = S X × S O represents a submatrix of P ij , as this is the subset with the most statistical evidence for having been effected by an anomalous pattern.

The general form of the NPSS score function is

where N (S) represents the number of empirical p-values contained in subset S and N α (S) is the number of p-values less than (significance level) α contained in subset S.

This generalizes to a submatrix, S = S X × S O , intuitively.

There are well-known goodness-of-fit statistics that can be utilized in NPSS McFowland et al. (2018) , the most popular is the Kolmogorov-Smirnov test Kolmogorov (1933) .

Another option is Higher-Criticism Donoho & Jin (2004) .

In this work we use the Berk-Jones test statistic Berk & Jones (1979) : φ BJ (α, N α , N ) = N * KL Nα N , α , where KL is the Kullback-Liebler divergence KL(x, y) = x log x y + (1 − x) log 1−x 1−y between the observed and expected proportions of significant p-values.

Berk-Jones can be interpreted as the log-likelihood ratio for testing whether the p-values are uniformly distributed on [0, 1] as compared to following a piece-wise constant step function alternative distribution, and has been shown to fulfill several optimality properties and has greater power than any weighted Kolmogorov statistic.

Although NPSS provides a means to evaluate the anomalousness of a subset of node activations S O for a given subset of evaluation images S X , discovering which of the 2 N × 2 J possible subsets (S = S X × S O ) provides the most evidence of an anomalous pattern is computationally infeasible for moderatly sized subsets of images and nodes.

However, NPSS has been shown to satisfy the linear-time subset scanning (LTSS) property Neill (2012) , which allows for an efficient and exact maximization over subsets of data.

For a pair of functions F (S) and G(X i ) representing the score of a given subset S of data and the "priority" of a data record X i respectively, we have a guarantee that the subset maximizing the score will be one consisting only of the top-k highest priority records, for some k between 1 and N .

If we consider a data record to be an image X i , then our goal is to max S X ⊆{S1,...,S N } F (S X × S O ), for a given subset of nodes S O .

The corresponding G(X i ) function to measure the priority of an image, is the proportion of its p-values that are less than α:

Thus far we have described how to find the most anomalous subset of images for a given subset of nodes.

Because F (S) operates on a submatrix of p-value ranges, we can reorient the same process to identify an anomalous subset of nodes S O for a given subset of images.

The goal is then to

, for a given subset of images S X .

The corresponding G(O j ) function to measure the priority of an node, is the proportion of its p-value ranges that are less than α:

Given the two efficient optimization steps described above (optimizing over all subsets of images for a given subset of nodes, and optimizing over all subsets of nodes for a given subset of images), we are able to compute an efficient local maximum of max S X ⊆{X1,...,

via an iterative ascent procedure.

To do so, we first choose a subset of attributes S O ⊆ {O 1 ...O J } uniformly at random.

We then iterate between the two LTSS-enabled optimization steps described above, until convergence, at which point we have reached a conditional maximum of the score function (S X is conditionally optimal given S O , and S O is conditionally optimal given S X ).

Moreover, we can perform multiple random restarts to approach the global optimum.

LTSS enabled efficient optimization of NPSS has been shown to reach the global maximum with high probability empirically and theoretical conditions have been provided the guarantee exact identification of the truly affected subset of data McFowland III et al. (2013) .

Machine Learning models are susceptible to adversarial perturbations of their input data that can cause the input to be misclassified Szegedy et al. (2013) ; Goodfellow et al. (2014) ; Kurakin et al. (2016a) ; Dalvi et al. (2004) .

There are a variety of methods to make neural networks more robust to adversarial noise.

Some require retraining with altered loss functions so that adversarial images must have a higher perturbation in order to be successful Papernot et al. (2015) ; .

Other detection methods rely on a supervised approach and treat the problem as classification by training on labeled noised examples Grosse et al. (2017) ; Gong et al. (2017); Huang et al. (2015) .

Another supervised approach is to use activations from hidden layers as features used by the detector.

Metzen et al. (2017) In contrast, our work treats the problem as anomalous pattern detection and operates in an unsupervised manner without apriori knowledge of the attack or labeled examples.

We also do not rely on training data augmentation or specialized training techniques.

Furthermore, our work is complimentary to many of the defenses mentioned above.

For example, if one defense type requires the noising process to makes more extreme perturbations to change the class label, then those patterns should be more easily detected by subset scanning methods.

A defense in Feinman et al. (2017) is more similar to our work.

They build a kernel density estimate over background activations from the nodes in only the last hidden layer and report when an image falls in a low density part of the density estimate.

This works well on MNIST, but performs poorly on CIFAR-10 Carlini & Wagner (2017).

Our novel subset scanning approach looks at anomalousness at the node-level and across multiple inputs (images) simultaneously in order to detect patterns that span altered inputs (images).

We trained a ResNet20 (v1) residual neural network He et al. (2015) on 50,000 CIFAR-10 training images that had their mean pixel values subtracted.

The test accuracy of the model was 0.9183.

Future work will explore the effect of model classification accuracy on the ability to detect anomalous patterns within its activations.

For this paper we chose Resnet20 for its relatively small size, popularity, and classification accuracy.

We focus our subset scanning methods on the final convolutional layer of the last residual block.

This layer contains 64 filters each containing 8x8 nodes.

Therefore our analysis is on the activations produced at these 4096 nodes.

We do not use the spatial locations or filter membership of the nodes in the scanning process, however both of these could be useful extensions.

For our adversarial experiments, we took M = |D H0 | = 9000 of the 10000 validation images and used them to generate the background activation distribution (D H0 ) at each of the 4096 nodes.

These images form our expectation of "normal' activation behavior for the network.

The remaining 1000 images were used to form groups: "Clean" (C) and "Adversarial" (A t ) with Adversarial containing targeted noised versions of the 1000 images in Clean, repeated for each targeted class label t ∈ (0 . . .

9).

We used the Basic Iterative Method (BIM) adversarial attack Kurakin et al. (2016b) which is an improved version of the original Fast Gradient Sign Method Goodfellow et al. (2014) .

BIM has an parameter which controls how far a pixel is allowed to change from its original value when noise is being added to the image.

We used a value of = 0.02 in the scaled [0,1] pixel space.

We also allowed the method to reach its final noised state over 10 steps with each of size 0.002.

Smaller values of make the pattern subtler and harder to detect, but also less likely for the attacks to succeed in changing the class label to the target.

All attacks were generated with the CleverHans package .

The images in A t were additionally reduced to remove any images that were originally classified as t (and therefore the noise had no effect).

The attacks had near 100% success rates.

Target classes 1, 4, and 7 each had 1 failed attack that were also removed from their respective group, A t .

The end result is each A t group contains approximately 900 images that were successfully noised from an original predicted label to label t. The 1000 images in group C are natural and have all class labels represented (nearly) equally.

We now proceed to generating tests sets that are composed of 500 images with varying percentages for clean images and adversarial images drawn randomly from group C and A t without replacement, respectively.

The percentages of adversarial images used in our test sets are 6%, 8%, 10%, 12%, and 14% for each of the target class labels.

We additionally consider a special case where instead of drawing from a single group A t , we uniformly draw from all A t groups.

This is a particularly difficult test case because we do not expect there to be a common pattern consistent within images adversarialy targeted to separate classes.

Each case is then repeated 100 times.

Using t = 0 and p = 0.10 as an example, there would be 100 test sets of size 500 images composed of randomly selected 450 clean images (from all classes) and randomly selected 50 images noised to class 0.

Table 1 .

The tests sets are scanned by our method in order to determine the highest-scoring (most anomlaous) subset of the 500 images.

Each test set was evaluated with 5 random restarts where a random subset of nodes was generated as the seed for the iterative ascent.

The most anomlaous subset returned by the scan is compared with the true state of the image (clean or noised).

We then report Detection Accuracy as measured by precision and recall of the most anomalous subset.

These results are shown in Table 1 .

In order to measure Detection Power we must model the null hypothesis of no adversarial noise present.

To do so we also create 100 test sets containing 500 clean images only; no noised images.

Scanning these sets generates a distribution of scores under the null hypothesis.

We then compare the distribution of scores generated when noised samples were present in the test set to the distribution of scores when scanning over a subset of 500 clean images.

This analysis results in an ROC curve for each case of target t and proportion p.

The area under the ROC curve then measures detection power of our method for that particular case.

Figure 2 demonstrates this process for target class 0 and proportions 6% and 12%.

Detection Power results are reported in Table 1 .

Finally, we consider experiments where images are scanned individually instead of as a larger group of 500 images.

The process is identical to the one describe above except the test set size is 1 image instead of 500.

We do not consider varying proportions in the individual case as each set is either p = 0 or p = 1.

We also do not report precision or recall as there is no subset returned in this restricted case.

We are able to report detection power by comparing the distribution of scores of an individual anomalous image to the distribution of scores created by scanning clean images individually.

The top panel of Table 1 provides detection power for our experiments.

Detection power is measured by Area-Under-ROC curves as demonstrated in Figure 2 .

The ability to detect targeted noise on individual images varies by class with moderate results and can be viewed as a performance floor.

The focus of this work however, is the ability to detect adversarial noise across multiple images simultaneously.

To that end, we show how detection power increases as the proportion of noised images increases in the test sets.

At a proportion of 10% detection power is higher as a group than it is for an individual image across all target classes.

Detection is nearly perfect for all classes at 12% and above.

This suggests our scanning method is identifying a subtle anomalous pattern of activations that persists across multiple noised images targeting a single class.

We now focus on the "All" category which considers the test sets containing targeted examples from each of the 10 class labels.

Detection power lags behind any single target class.

This is because in the single target cases, our scanning method is exploiting an anomalous activation pattern that is consistent across multiple images.

This pattern is less consistent when targeting different class Table 1 : Detection Power and Accuracy for targeted adversarial noise added to CIFAR-10 images by the Basic Iterative Method attack.

Results are provided for detecting individual images and subsets of 500 images where the number of noised images varies from 6% to 14%.

labels in the same test set.

This suggests that targetted noise is activating the same set of nodes despite the original images coming from different classes.

In addition to Detection Power, Table 1 provides precision and recall measurements for the subsets of images identified by our scanning method.

Precision is consistently lower than recall.

We attribute this to two reasons.

The first is that the 500 image test set contains targeted noised examples of a single class label, as well as natural images of that same class.

Therefore, we believe the subset of anomalous images is likely to include the noised images and the natural images belonging to the target class, which decreases precision.

Another reason for a relatively low precision is due to a static setting of a parameter to the scanning function, α max .

For simplicity, this value was set to 0.5 for all runs and may be interpreted as assuming up to half of the data may be affected by the anomalous pattern.

This is an inflated value which can be lowered if investigators had an apriori belief on the prevalence of the affected subsets in their data (i.e. the 6%-14% used in our experiments).

Lowering this value would almost certainly increase precision (and lower recall).

We now consider recall measurements located in the bottom panel of Table 1 .

Recall is exceptionally high in our experiments.

Similar to the argument for low precision, the high recall values are due to a large, static α max value.

A hyper-parameter search is feasible in supervised settings.

Instead these experiments were conducted in an unsupervised form with α max set arbitrarily at 0.5.

We now highlight a more subtle strength of our method with regards to recall.

All things being equal, increasing the number of the noised images should decrease the recall rate as there are more noised images to miss.

However, in almost all target classes we observe steady trend or an increase.

This demonstrates subset scanning's innate adaptability by maintaining strong recall despite the number of noised images more than doubling (from 6% to 14%).

This work uses the Adversarial Noise domain as an effective narrative device to demonstrate that anomalous patterns in the activation space of neural networks can be efficiently quantified and detected across a subset of inputs (images).

The primary contribution of this work to the data mining and deep learning literature is a novel, unsupervised anomaly detector that can be applied to any pre-trained, off-the-shelf neural network model.

The method is based on subset scanning which treats the detection problem as a search for the highest scoring (most anomalous) subset of node activations × inputs (images) as measured by nonparametric scan statistics.

This is the first work to apply subset scanning methods to neural network activations and represents a novel contribution to both domains.

Nonparametric scan statistics applied to neural network activations operate on three levels of anomalousness.

The first level is at a single activation generated by an input at a node.

The anomalousness of this activation is quantified by its empirical p-value that reflects how large this activation is compared to a "background" of activations from known, natural images at the same node.

Of course, not every input that has a large activation at this node is anomalous.

We therefore must consider the second level measured by NPSS: the anomalousness of a subset of activations for a single image (or equivalently, a single node).

This level identifies the most anomalous subset of empirical p-values from a single image (or a single node).

Despite the exponentially many subsets to consider, this optimization can be done exactly by only considering a linearly-many number of subsets of activations Neill (2012) .

An image (or node) that has a large number of small p−values is considered to be more anomalous.

However, the large activations that make one image anomalous may occur at different nodes than an image that is equally anomalous with high activations at a different subet of nodes.

This consideration brings us to the third and highest level of anomalousness for NPSS applied to neural network activations: identifying a subset of inputs (images) that have higher-than-expected activations (i.e. large number of low empirical p−values) at a subset of nodes.

This search procedure uses the same efficient optimization step to iteratively ascend between identifying the most anomalous subset of nodes (for a given, fixed subset of images) and identifying the most anomalous subset of images (for a given, fixed subset of nodes).

In practice, this scanning method is able to identify a high scoring subset of images × nodes from a search space of 500 images and 4096 nodes in 3.8 seconds on average.

Efficient optimization is important in the large search space of neural network activations.

However, this work also demonstrated that the subset identified by the scanning procedure is relevant for an anomalous pattern of interest.

The second contribution of this paper is providing empirical results that subset scanning can detect the presence of targeted adversarial noise.

Furthermore the detection power, precision, and recall increase when images with targeted noise are considered together as a group.

To the best of our knowledge, this is the first work to consider adversarial noise detection across a group of images.

Most adversarial noise defenses only provide detection results for individual images as they fail to scale to detecting at the group level.

In practical settings, if a neural network is under a targeted attack there will be systematic differences across the affected images.

Our method is capable of detecting these subtle, but systematic, patterns at node activations across multiple images.

We also highlight that the adversarial noise detection task was performed completely unsupervised and orthogonal to the original goal of the trained ResNet: to attain high classification accuracy.

This suggests that subset scanning over neural network activations will be relevant in a broad range of neural network applications.

This appendix is meant to provide implementation details that are relevant to readers wishing to implement their own version of our experiments.

Direct code is provided anonymously on Github https://github.com/hikayifix/adversarialdetector.

We also provide code examples and psuedo code here as well.

This paper has a focus on reproducible results and has used as many vanilla settings as possible.

Our resnet training was done in Keras and was taken directly from https://github.

com/keras-team/keras/blob/master/examples/cifar10_resnet.py We had to change steps per epoch to The activations from the 9000 background images were sorted offline.

Then at runtime we use np.searchsorted to determine where an evaluation activation would fall among the sorted background activations at each node.

This was used to efficiently calculate p-value ranges.

<|TLDR|>

@highlight

We efficiently find a subset of images that have higher than expected activations for some subset of nodes.  These images appear more anomalous and easier to detect when viewed as a group. 

@highlight

The paper proposed a scheme to detect the presence of anomalous inputs based on a "subset scanning" approach to detect anomalous activations in the deep learning network.