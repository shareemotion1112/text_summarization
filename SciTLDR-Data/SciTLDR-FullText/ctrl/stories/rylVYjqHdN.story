The goal of multi-label learning (MLL) is to associate a given instance with its relevant labels from a set of concepts.

Previous works of MLL mainly focused on the setting where the concept set is assumed to be fixed, while many real-world applications require introducing new concepts into the set to meet new demands.

One common need is to refine the original coarse concepts and split them into finer-grained ones, where the refinement process typically begins with limited labeled data for the finer-grained concepts.

To address the need, we propose a special weakly supervised MLL problem that not only focuses on the situation of limited fine-grained supervision but also leverages the hierarchical relationship between the coarse concepts and the fine-grained ones.

The problem can be reduced to a multi-label version of negative-unlabeled learning problem using the hierarchical relationship.

We tackle the reduced problem with a meta-learning approach that learns to assign pseudo-labels to the unlabeled entries.

Experimental results demonstrate that our proposed method is able to assign accurate pseudo-labels, and in turn achieves superior classification performance when compared with other existing methods.

Multi-label learning (MLL) is an important learning problem with a wide range of applications BID2 BID0 BID11 .

While traditional setting focuses on the scenario where the label classes are fixed before learning, many real-world applications face different situations.

One scenario that is common in many applications is the growing number of classes BID13 , where the growth splits high-level concepts to finer-grained ones BID1 .

For example, the set of classes might start from high-level concepts such as {Animal, . . .

, Food }, and then grow to include finer-grained concepts like {Cat, . . .

, Dog, . . .

, Apple, . . .

, Banana}. Typical applications may have collected sufficient number of labeled data for learning the high-level concepts in a fully supervised manner, but it can be challenging for the applications to efficiently adapt the classifier from the high-level (coarse-grained) concepts to the finer-grained ones.

Conquering the challenge calls for two components: one is a strategic algorithm to actively collect a few fine-grained and informative labels, and the other is an effective learning model to exploit the fine-grained labels that have been partially collected.

This work focuses on the design of the second component-learning an accurate fine-grained classifier with only limited supervision.

In particular, we assume that the model receives a data set that contains all the coarse-grained labels and a few fine-grained ones, as shown in FIG0 .

Then, the problem of constructing a predictive fine-grained model with the presented data set falls under the big umbrella of weakly supervised learning.

Specifically, when we focus on leveraging the coarse-grained labels to build a fine-grained classifier, the problem resembles learning with inexact supervision considered by BID12 , where the coarse-grained labels are not in the exact form for the desired output and could only provide weak information about the target fine-grained labels.

On the other hand, if we focus on using the fine-grained part of the labels to train the classifier, the problem can be viewed as a multi-label variant of learning with incomplete supervision as some instances receive their exact fine-grained ground-truth labels whereas some do not have labels at all BID12 .

While both the aforementioned problems have attracted much research attention, the combination of them (inexact and incomplete supervision) which our problem of interest can be cast as, has not yet been carefully investigated to the best of our knowledge.

Organization In this work, we start from a formal definition of our problem of interest.

We then demonstrate a simple way to reduce the original problem into a special form of negative-unlabeled learning problem BID7 leveraging the label hierarchy.

To tackle the reduced problem, we begin with a discussion on the caveats carried by some possible existing approaches, and propose a new model that undertakes the challenges posed by inexact and incomplete supervision through a novel learning to learn method which jointly exploits the hierarchical relationship between the coarse-and fine-grained labels, as well as the benefits of all available data in hand.

The key idea within our model is to take into account all available information to learn the labeling assignments for the unlabeled entries, called pseudo-labels, and use them to guide the decent direction of the parameter updates on the underlying classifier.

Finally, we experimentally demonstrate that the proposed method not only assigns accurate pseudo-labels to the unknown entries but also enjoys significantly better performance than other methods for learning fine-grained classifiers under the limited supervision setting.

Formally, we denote an instance by a feature vector x ∈ R d , and its relevant labels by a bit vector y ∈ {−1, 1}K to indicate whether the labels in a pre-defined set Y = {y 1 , ..., y K } are relevant, i.e., y[k] = 1 if and only if y k is relevant.

In this work, rather than assuming that the set Y is fixed, we consider the problem of splitting the original high-level concepts into finer-grained ones, refining the label set of interest from Y c = {y 1 , ..., y C } into Y f = {y 11 , y 12 , ..., y C1 , y C2 , ...} as shown in FIG2 .

Let y c and y f be the corresponding label vectors for Y c and Y f respectively.

Assume that we receive a data set DISPLAYFORM0 consisting of N examples that are annotated only with the high-level (coarse-grained) labels, and an additional small set DISPLAYFORM1 of M examples with their fine-grained labels annotated, our goal is to leverage these examples to learn an accurate fine-grained classifier Φ(θ, x) : DISPLAYFORM2 K where θ is the model parameter and K is the total number of fine-grained classes.

Fully-Supervised Learning A straightforward way to learn a fine-grained classifier is to utilize only the fully-annotated training examples in D tr through standard supervised approaches.

Nevertheless, the small number of examples in this set might be unable to train a strong classifier.

Moreover, it completely ignores the (weak) supervision provided by the abundant coarse labels.

Multi-Label Learning with Missing Labels One way to make use of the higher-level supervision on learning the fine-grained concepts is to leverage the hierarchical relationship where the TAB0 To tackle the reduced problem, one way is to treat the unknown fine-grained labels as missing entries, and apply MLL algorithms that can learn with the presence of missing labels BID4 BID8 BID9 .

BID9 proposed a classic empirical risk minimization styled method LEML, attempting to solve the optimization problem that arrives at the model parameters which can most accurately recover the observed training labels.

Roughly, their objective is formulated as: DISPLAYFORM0 where Ω is the set of indices of the observed entries in Y f and L is a loss function that measures the discrepancy between the predicted and ground-truth labels.

From the objective, however, we note that only the observed entries contribute to the learning of model parameters, and the unobserved ones are basically ignored in the model training process.

Unfortunately, in our setting, as the observed fine-grained labels are mostly deduced from the irrelevance of their parent labels, LEML is thus unable to exploit the weak supervision provided by the relevant coarse labels.

One-Class Multi-Label Learning Another plausible direction to approach the reduced problem is through one-class multi-label learning methods BID10 .

A common approach took in these methods is to assume the values of the unobserved entries to be the opposite class of the observed ones, and train a cost-sensitive classifier with different weights given to the observed and unobserved entries.

Nonetheless, as the underlying ground truths for the missing entries are not necessarily the presumed class, without careful algorithm redesign or label distribution estimation, these methods may suffer from the introduced label bias that results in suboptimal performances.

While existing solutions have developed different ways of treating the unknown entries during the learning process, they somehow do not delicately exploit the benefits of the unlabeled entries as mentioned in the previous section.

In light of this, we seek for a method that could more properly leverage the missing entries with a key assumption that: When the missing entries are all correctly recovered and used in the training process, the classifier learned could achieve the best performance.

Based on the assumption, we attempt to find the best labeling assignment to the unknown entries, called pseudo-labels, which when the model is trained accordingly, can lead to best classification performance on the fine-grained concepts.

Towards this goal, we propose to use the few examples that receive their fully-annotated fine-grained labels in D tr as a validation set to evaluate the classifier's performance on the fine-grained concepts.

Formally, we aim to train our fine-grained classifier using the examples in D tr with a pseudo fine-grained label matrix Y pseudo where: DISPLAYFORM0 where every p ij is a pseudo label to be determined and the objective is: DISPLAYFORM1 Note that with different pseudo-labels assigned to the missing entries, we arrive at different optimal model parameter θ * (Y pseudo ).

And the optimal assignment of the pseudo-labels should be based on the validation performance of the resulting classifier: DISPLAYFORM2 However, solving Eq. 4 to find the optimal pseudo-label assignment requires a computationally prohibiting two-loop optimization procedure.

To conquer the optimization challenge, inspired by recent works in meta-learning literature BID3 BID6 , we attempt to tackle the problem with an iterative approach which dynamically find the best pseudo-label assignments locally at each optimization step.

Specifically, consider a typical gradient descent update step: DISPLAYFORM3 where α is the step size and t is the current timestep.

Then, at each iteration t, we aim to learn the pseudo-label assignment which leads to the model parameters that minimize the validation loss after a single update step: DISPLAYFORM4 Solving Eq. 6 at each timestep t could, nevertheless, still be very expensive.

As a result, we propose a simple approximation of (Y pseudo ) * t by looking at the gradient direction (sign) of the validation loss wrt.

the pseudo-labels.

Particularly, we assign pseudo-labels at timestep t by: DISPLAYFORM5 4 EXPERIMENTS AND DISCUSSIONTo justify the effectiveness of the proposed method, we test our method on a multi-label image dataset MS COCO BID5 .

We compare our method with three baseline methods, namely, (1) a standard fully-supervised (FS) learning model, (2) LEML, a classic approach in handling typical missing-label setup BID9 , and (3) a representative method that tackles the problem of one-class classification (OCC) BID10 .

We deploy a fully-connected neural network as our underlying base model.

In TAB0 , we show the results of different methods with varying size of D tr .

It can be seen that our method consistently achieves the best performances across different settings.

It is worthwhile to note that although the standard fully-supervised approach does not leverage the partially labeled examples in D tr , it surprisingly outperforms the other two baseline methods in many cases.

To investigate the reasons for so, we plot the learning curves of different methods in Figure 3 .

From the figure, we see that although LEML and OCC achieve comparable, or even better, performances than the fully-supervised approach at the very beginning of the learning process, the two approaches then quickly suffers from overfitting that results in the performance drop.

For LEML, we conjecture that the performance degrading comes from the overwhelming number of negative entries dominating the learning dynamic.

And arguably, the severe overfitting of OCC results from the over-simple assumption on the missing entries which brings label bias into the learning objective.

Recover rate of our method To understand the benefits of the pseudo-labels learned in our approach, we show how our method is capable of correctly recovering the missing entries, as well as the correlation between the recover rate and model performance.

In Figure 4 , we plot the recover performance of the learned pseudo-labels measured by F1-loss (1 − F1-score), and the horizontal bars are the corresponding F1-loss by simply treating all missing entries as ones and assigning them random labels.

We can see from the figure that the pseudo-labels learned from our method could much more correctly recover the missing entries than the two naive baselines.

In addition, there is a strong correlation between the recover rate and model classification performance.

With more accurate assignment of pseudo-labels on the unknown entries, the trained model is able to achieve stronger classification performance.

We design a tailored method through a meta-learning strategy, which learns to accurately assign pseudo-labels to the unknown entries of a special weakly supervised MLL problem.

Experimental results show that our proposed method not only assigns accurate pseudo-labels, but also enable the underlying classifier learned to perform better than other possible existing solutions.

<|TLDR|>

@highlight

We propose a special weakly-supervised multi-label learning problem along with a newly tailored algorithm that learns the underlying classifier by learning to assign pseudo-labels.