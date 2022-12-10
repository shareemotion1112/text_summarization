We study the problem of multiset prediction.

The goal of multiset prediction is to train a predictor that maps an input to a multiset consisting of multiple items.

Unlike existing problems in supervised learning, such as classification, ranking and sequence generation, there is no known order among items in a target multiset, and each item in the multiset may appear more than once, making this problem extremely challenging.

In this paper, we propose a novel multiset loss function by viewing this problem from the perspective of sequential decision making.

The proposed multiset loss function is empirically evaluated on two families of datasets, one synthetic and the other real, with varying levels of difficulty, against various baseline loss functions including reinforcement learning, sequence, and aggregated distribution matching loss functions.

The experiments reveal the effectiveness of the proposed loss function over the others.

A relatively less studied problem in machine learning, particularly supervised learning, is the problem of multiset prediction.

The goal of this problem is to learn a mapping from an arbitrary input to a multiset 1 of items.

This problem appears in a variety of contexts.

For instance, in the context of high-energy physics, one of the important problems in a particle physics data analysis is to count how many physics objects, such as electrons, muons, photons, taus, and jets, are in a collision event BID4 .

In computer vision, automatic alt-text, such as the one available on Facebook, 2 is a representative example of multiset prediction BID16 BID9 .

3 In multiset prediction, a learner is presented with an arbitrary input and the associated multiset of items.

It is assumed that there is no predefined order among the items, and that there are no further annotations containing information about the relationship between the input and each of the items in the multiset.

These properties make the problem of multiset prediction unique from other wellstudied problems.

It is different from sequence prediction, because there is no known order among the items.

It is not a ranking problem, since each item may appear more than once.

It cannot be transformed into classification, because the number of possible multisets grows exponentially with respect to the maximum multiset size.

In this paper, we view multiset prediction as a sequential decision making process.

Under this view, the problem reduces to finding a policy that sequentially predicts one item at a time, while the outcome is still evaluated based on the aggregate multiset of the predicted items.

We first propose an oracle policy that assigns non-zero probabilities only to prediction sequences that result exactly in the target, ground-truth multiset given an input.

This oracle is optimal in the sense that its prediction never decreases the precision and recall regardless of previous predictions.

That is, its decision is optimal in any state (i.e., prediction prefix).

We then propose a novel multiset loss which minimizes the KL divergence between the oracle policy and a parametrized policy at every point in a decision trajectory of the parametrized policy.

1 A set that allows multiple instances, e.g. {x, y, x}. See Appendix A for a detailed definition.

https://newsroom.fb.com/news/2016/04/using-artificial-intelligenceto-help-blind-people-see-facebook/ 3 We however note that such a multiset prediction problem in computer vision can also be solved as segmentation, if fine-grained annotation is available.

See, e.g., BID6 .We compare the proposed multiset loss against an extensive set of baselines.

They include a sequential loss with an arbitrary rank function, sequential loss with an input-dependent rank function, and an aggregated distribution matching loss and its one-step variant.

We also test policy gradient, as was done by BID16 recently for multiset prediction.

Our evaluation is conducted on two sets of datasets with varying difficulties and properties.

According to the experiments, we find that the proposed multiset loss outperforms all the other loss functions.

The paper is structured as follows.

We first define multiset prediction at the beginning of Section 2, and compare it to existing problems in supervised learning in 2.1.

Then we propose the multiset loss in Section 2.2, followed by alternative baseline losses in Section 3.

The multiset loss and baselines are then empirically evaluated in Section 4.

A multiset prediction problem is a generalization of classification, where a target is not a single class but a multiset of classes.

The goal is to find a mapping from an input x to a multiset Y = y 1 , . . .

, y |Y| , where y k ∈ C. Some of the core properties of multiset prediction are 1.

the input x is an arbitrary vector.2.

there is no predefined order among the items y i in the target multiset Y.3.

the size of Y may vary depending on the input x.4.

each item in the class set C may appear more than once in Y.Refer to Appendix A for definitions related to multiset prediction.

As is typical in supervised learning, in multiset prediction a model DISPLAYFORM0 and computing evaluation metrics m(·) that compare the predicted and target multisets, DISPLAYFORM1 Here, F1 score and exact match (defined in Appendix A), are used as evaluation metrics.

Variants of this multiset prediction problem have been extensively studied.

However, they differ from our definition of the problem.

Here, we go over each variant and discuss how it differs from our definition of multiset prediction.

Power Multiset Classification Perhaps the most naive approach to multiset prediction is to transform the class set C into a set M (C) of all possible multisets.

This transformation, or the size of M (C), is not well defined unless some constraints are put in place.

If the maximum size of a target multiset is set to K, the number of all possible multisets is DISPLAYFORM0 With some constant |C|

, we notice that this grows exponentially in the maximum size of the target multiset.

Once the class set C is transformed, we can train a multi-class classifier π that maps an input x to one of the elements in M (C).

However, this is infeasible in practice and generally intractable.

For instance, for the COCO Medium dataset used later in the experiments (see section 4.1), M (C) has roughly 20 thousand elements while the dataset only contains roughly 40 thousand training examples.

For the full MS COCO dataset, |M (C)| is on the order of 10 49 , making it infeasible to learn a classifier using this method.

Ranking A ranking problem can be considered as learning a mapping from a pair of input x and one of the items c ∈ C to its score s(x, c).

All the items in the class set are then sorted according to the score, and this sorted order determines the rank of each item.

By taking the top-K items from this sorted list, we can turn this problem of ranking into set prediction.

Similarly to multiset prediction, the input x is arbitrary, and the target is a set without any prespecific order.

However, ranking differs from multiset prediction in that it is unable to handle multiple occurrences of a single item in the target set.

Aggregated Distribution Matching Instead of considering the target multiset as an actual multiset, one can convert it into a distribution by computing the frequency of each item from the class set in the target multiset.

That is, DISPLAYFORM1 where I · is an indicator function.

Then, we can simply minimize a divergence between this distribution and the predictive distribution from a model.

This loss function works only when the conditional distribution p(y|x) substantially differs from the marginal distribution p(y), since the model would resort to a trivial solution of predicting the marginal distribution regardless of the input x.

We describe this approach in more detail in Sec. 3.1, and test it against our proposal in the experiments.

Sequence prediction A sequence prediction problem is characterized as finding a mapping from an input x to a sequence of classes Y = y 1 , . . .

, y |Y| .

Representative examples of sequence prediction include machine translation, automatic speech recognition and other tagging problems, such as part-of-speech tagging, in natural language processing.

Similarly to multiset prediction, the input x is arbitrary, and an item in the class set C may appear more than once in the target sequence.

It is, however, different from multiset prediction in that there is a clear, predetermined order of items in the target sequence.

We detail this sequence prediction approach later in Sec. 3.2.

In this paper, we propose a novel loss function, called multiset loss, for the problem of multiset prediction.

This loss function is best motivated by treating the multiset prediction problem as a sequential decision making process with a model being considered a policy π.

This policy takes as input the input x and all the previously predicted classesŷ <t at time t, and outputs the distribution over the next class to be predicted.

That is, π θ (y t |ŷ <t , x).

This policy is parametrized with a set θ of parameters.

We first define a free label multiset at time t as Definition 1 (Free Label Multiset).Y t ← Y t−1 \ {ŷ t−1 } y t−1 is the prediction made by the policy at time t − 1.This free label multiset Y t contains all the items that remain to be predicted after t − 1 predictions by the policy.

We then construct an oracle policy π * .

This oracle policy takes as input a sequence of predicted labelsŷ <t , the input x, and the free label multiset with respect to its predictions, Y t = Y\ {ŷ <t }.

It outputs a distribution whose entire probability (1) is evenly distributed over all the items in the free label multiset Y t .

In other words, Definition 2 (Oracle).

DISPLAYFORM0 An interesting and important property of this oracle is that it is optimal given any prefixŷ <t with respect to both precision and recall.

This is intuitively clear by noticing that the oracle policy allows only a correct item to be selected.

We call this property the optimality of the oracle.

Remark 1.

Given an arbitrary prefixŷ <t , DISPLAYFORM1 The proof is given in Appendix B. See Appendix A for definitions of precision and recall for multisets.

From the remark above, it follows that the oracle policy is an optimal solution to the problem of multiset prediction in terms of precision and recall.

Remark 2.

DISPLAYFORM2 The proof can be found in Appendix C.It is trivial to show that sampling from such an oracle policy would never result in an incorrect prediction.

That is, this oracle policy assigns zero probability to any sequence of predictions that is not a permutation of the target multiset.

Remark 3.

DISPLAYFORM3 where multiset equality refers to exact match, defined in Appendix A. In short, this oracle policy tells us at each time step t which of all the items in the class set C must be selected.

This optimality allows us to consider a step-wise loss between a parametrized policy π θ and the oracle policy π * , because the oracle policy provides us with an optimal decision regardless of the quality of the prefix generated so far.

We thus propose to minimize the KL divergence from the oracle policy to the parametrized policy at each step separately.

This divergence is defined as DISPLAYFORM4 where Y t is formed using predictionsŷ <t from π θ , and H(π t * ) is the entropy of the oracle policy at time step t. This entropy term can be safely ignored when learning π θ , since it is constant with respect to θ.

We define DISPLAYFORM5 and call it a per-step loss function.

We note that it is indeed possible to use another divergence in the place of the KL divergence.

It is intractable to minimize the per-step loss from Eq. (2) for every possible state (ŷ <t , x), since the size of the state space grows exponentially with respect to the size of a target multiset.

We thus propose here to minimize the per-step loss only for the state, defined as a pair of the input x and the prefixŷ <t , visited by the parametrized policy π θ .

That is, we generate an entire trajectory (ŷ 1 , . . .

,ŷ T ) by executing the parametrized policy until either all the items in the target multiset have been predicted or the predefined maximum number of steps have passed.

Then, we compute the loss function at each time t based on (x,ŷ <t ), for all t = 1, . . .

, T .

The final loss function is then the sum of all these per-step loss functions.

DISPLAYFORM6 where T is the smaller of the smallest t for which Y t = ∅ and the predefined maximum number of steps allowed.

Note that as a consequence of Remarks 2 and 3, minimizing the multiset loss function results in maximizing F1 and exact match.

As was shown by BID12 , the use of the parametrized policy π θ instead of the oracle policy π * allows the upper bound on the learned policy's error to be linear with respect to the size of the target multiset.

If the oracle policy had been used, the upper bound would have grown quadratically with respect to the size of the target multiset.

To confirm this empirically, we test the following three alternative strategies for executing the parametrized policy π θ in the experiments:1.

Greedy search:ŷ t = arg max y log π θ (y|ŷ <t , x) 2.

Stochastic sampling: DISPLAYFORM7 Once the proposed multiset loss is minimized, we evaluate the learned policy by greedily selecting each item from the policy.

We have defined the proposed loss function for multiset prediction while assuming that the size of the target multiset was known.

However, this is a major limitation, and we introduce two different methods for relaxing this constraint.

Termination Policy The termination policy π s outputs a stop distribution given the predicted sequence of itemsŷ <t and the input x. Because the size of the target multiset is known during training, we simply train this termination policy in a supervised way using a binary cross-entropy loss.

At evaluation time, we simply threshold the predicted stop probability at a predefined threshold (0.5).Special Class An alternative strategy is to introduce a special item to the class set, called END , and add it to the final free label multiset Y |Y|+1 = { END }.

Thus, the parametrized policy is trained to predict this special item END once all the items in the target multiset have been predicted.

This is analogous to NLP sequence models which predict an end of sentence token BID14 BID0 , and was used in BID16 to predict variable-sized multisets.

In addition to the proposed multiset loss function, we propose three more loss functions for multiset prediction.

They serve as baselines in our experiments later.

In the case of distribution matching, we consider the target multiset Y as a set of samples from a single, underlying distribution q * over the class set C. This underlying distribution can be empirically estimated by counting the number of occurrences of each item c ∈ C in Y. That is, DISPLAYFORM0 where I is the indicator function as before.

Similarly, we can construct an aggregated distribution computed by the parametrized policy π θ .

As with the proposed multiset loss in Def.

3, we first execute π θ to predict a multisetŶ. This is converted into an aggregated distribution q θ in the same way as we turned the target multiset into the oracle aggregate distribution.

Learning is equivalent to minimizing the divergence between these two distributions.

In this paper, we test two types of divergences.

The first one is from a family of L p distances defined as DISPLAYFORM1 where q * and q are the vectors representing the corresponding categorical distributions.

The other is a usual KL divergence defined earlier in Eq. (1): DISPLAYFORM2 One major issue with this approach is that minimizing the divergence between the aggregated distributions does not necessarily result in the optimal policy (see the oracle policy in Def.

2.)

That is, a policy that minimizes this loss function may assign non-zero probability to an incorrect sequence of predictions, unlike the oracle policy.

This is due to the invariance of the aggregated distribution to the order of predictions.

Later when analyzing this loss function, we empirically notice that a learned policy often has a different behaviour from the oracle policy, for instance, reflected by the increasing entropy of the action distribution over time.

We can train an one-step predictor with this aggregate distribution matching criterion, instead of learning a policy π θ .

That is, a predictor outputs both a point q θ (·|x) in a |C|-dimensional simplex and the sizel θ (x) of the target multiset.

Then, for each unique item c ∈ C, the number of its occurrences in the predicted multisetŶ is DISPLAYFORM0 where λ > 0 is a coefficient for balancing the contributions from the two terms.

A major weakness of this one-step variant, compared to the approaches based on sequential decision making, is the lack of modelling dependencies among the items in the predicted multiset.

We test this approach in the experiments later and observe this lack of output dependency modelling results in substantially worse prediction accuracy.

All the loss functions defined so far have not relied on the availability of an existing order of items in a target multiset.

However, by turning the problem of multiset prediction into sequential decision making, minimizing such a loss function is equivalent to capturing an order of items in the target multiset implicitly.

Here, we instead describe an approach based on explicitly defining an order in advance.

This will serve as a baseline later in the experiments.

We first define a rank function r that maps from one of the unique items in the class set c ∈ C to a unique integer.

That is, r : C → Z. This function assigns the rank of each item and is used to order items y i in a target multiset Y. This results in a sequence S = (s 1 , . . .

, s |Y| ), where r(s i ) ≥ r(s j ) for all j > i, and s i ∈ Y. With this target sequence S created from Y using the rank function r, we define a sequence loss function as DISPLAYFORM0 Minimizing this loss function is equivalent to maximizing the conditional log-probability of the sequence S given x.

This sequence loss function has two clear disadvantages.

First, it does not take into account the actual behaviour of the policy π θ (see, e.g., BID1 BID2 BID12 .

This makes a learned policy potentially vulnerable to cascading error at test time.

Second and more importantly, this loss function requires a pre-specified rank function r. Because multiset prediction does not come with such a rank function by definition, we must design an arbitrary rank function, and the final performance varies significantly based on the choice.

We demonstrate this variation in section 4.3.Input-Dependent Rank Function When the input x has a well-known structure, and an object within the input for each item in the target multiset is annotated, it is possible to devise a rank function per input.

A representative example is an image input with bounding box annotations.

Here, we present two input-dependent rank functions in such a case.

First, a spatial rank function r spatial assigns an integer rank to each item in a given target multiset Y such that where x i and x j are the objects corresponding to the items y i and y j .Second, an area rank function r area decides the rank of each label in a target multiset according to the size of the corresponding object inside the input image: DISPLAYFORM1 The area may be determined based on the size of a bounding box or the number of pixels, depending on the level of annotation.

We test these two image-specific input-dependent rank functions against a random rank function in the experiments.

In BID16 , an approach based on reinforcement learning was proposed for multiset prediction.

Instead of assuming the existence of an oracle policy, this approach solely relies on a reward function r designed specifically for multiset prediction.

The reward function is defined as DISPLAYFORM0 The goal is then to maximize the sum of rewards over a trajectory of predictions from a parametrized policy π θ .

The final loss function is DISPLAYFORM1 where the second term inside the expectation is the negative entropy multiplied with a regularization coefficient λ.

The second term encourages the exploration during training.

As in BID16 , we use REINFORCE (Williams, 1992) to stochastically minimize the loss function above with respect to π θ .This loss function is optimal in that the return, i.e., the sum of the step-wise rewards, is maximized when both the precision and recall are maximal (= 1).

In other words, the oracle policy, defined in Def.

2, maximizes the expected return.

However, this approach of reinforcement learning is known to be difficult, with a high variance BID11 .

This is especially true here, as the size of the state space grows exponentially with respect to the size of the target multiset, and the action space of each step is as large as the number of unique items in the class set.

In this section, we extensively evaluate the proposed multiset loss function against various baseline loss functions presented throughout this paper.

More specifically, we focus on its applicability and performance on image-based multiset prediction.

MNIST Multi MNIST Multi is a class of synthetic datasets.

Each dataset consists of multiple 100x100 images, each of which contains a varying number of digits from the original MNIST (LeCun et al., 1998).

We vary the size of each digit and also add clutters.

In the experiments, we consider the following variants of MNIST Multi:• MNIST Multi (4): |Y| = 4, 20-50 pixel digits • MNIST Multi (1-4): |Y| ∈ {1, . . .

, 4}, 20-50 pixel digits • MNIST Multi (10): |Y| = 10, 20 pixel digits Each dataset has a training set with 70,000 examples and a test set with 10,000 examples.

We randomly sample 7,000 examples from the training set to use as a validation set, and train with the remaining 63,000 examples.

MS COCO As a real-world dataset, we use Microsoft COCO BID10 which includes natural images with multiple objects.

Compared to MNIST Multi, each image in MS COCO has objects of more varying sizes and shapes, and there is a large variation in the number of object instances per image which spans from 1 to 91.

The problem is made even more challenging with many overlapping and occluded objects.

To control the difficulty in order to better study the loss functions, we create the following two variants:• COCO Easy: |Y| = 2, 10,230 training examples, 24 classes• COCO Medium: |Y| ∈ {1, . . .

, 4}, 44,121 training examples, 23 classesIn both of the variants, we only include images whose |Y| objects are large and of common classes.

An object is defined to be large if the object's area is above the 40-th percentile across the train set of MS COCO.

After reducing the dataset to have |Y| large objects per image, we remove images containing only objects of rare classes.

A class is considered rare if its frequency is less than 1 |C| , where C is the class set.

These two stages ensure that only images with a proper number of large objects are kept.

We do not use fine-grained annotation (pixel-level segmentation and bounding boxes) except for creating input-dependent rank functions from Sec. 3.2.For each variant, we hold out a randomly sampled 15% of the training examples as a validation set.

We form separate test sets by applying the same filters to the COCO validation set.

The test set sizes are 5,107 for COCO Easy and 21,944 for COCO Medium.

MNIST Multi We use three convolutional layers of channel sizes 10, 10 and 32, followed by a convolutional long short-term memory (LSTM) layer BID18 .

At each step, the feature map from the convolutional LSTM layer is average-pooled spatially and fed to a softmax classifier.

In the case of the one-step variant of aggregate distribution matching, the LSTM layer is skipped.

MS COCO We use a ResNet-34 BID5 ) pretrained on ImageNet BID3 ) as a feature extractor.

The final feature map from this ResNet-34 is fed to a convolutional LSTM layer, as described for MNIST Multi above.

We do not finetune the ResNet-34 based feature extractor.

In all experiments, for predicting variable-sized multisets we use the termination policy approach since it is easily applicable to all of the baselines, thus ensuring a fair comparison.

Conversely, it is unclear how to extend the special class approach to the distribution matching baselines.

When evaluating a trained policy, we use greedy decoding and the termination policy for determining the size of a predicted multiset.

Each predicted multiset is compared against the ground-truth target multiset, and we report both the accuracy based on the exact match (EM) and F-1 score (F1), as defined in Appendix A.More details about the model architectures and training are in Appendix D.

We test three alternatives: a random rank function 4 r and two input-dependent rank functions r spatial and r area .

We compare these rank functions on MNIST Multi (4) and COCO Easy validation sets.

We present the results in TAB0 .

It is clear from the results that the performance of the sequence prediction loss function is dependent on the choice of a rank function.

In the case of MNIST Multi, the area-based rank function was far worse than the other choices.

However, this was not true on COCO Easy, where the spatial rank function was worst among the three.

In both cases, we have observed that the random rank function performed best, and from here on, we use the random rank function in the remaining experiments.

This set of experiments firmly suggests the need of an order-invariant multiset loss function, such as the multiset loss function proposed in this paper.

In this set of experiments, we compare the three execution strategies for the proposed multiset loss function, illustrated in Sec. 3.

They are greedy decoding, stochastic sampling and oracle sampling.

We test them on MNIST Multi (10) and COCO Easy.

As shown in TAB2 , greedy decoding and stochastic sampling, both of which consider states that are likely to be visited by the parametrized policy, outperform the oracle sampling.

This is consistent with the theory by BID12 .

Although the first two strategies perform comparably to each other, across both of the datasets and the two evaluation metrics, greedy decoding tends to outperform stochastic sampling.

We conjecture this is due to better matching between training and testing in the case of greedy decoding.

Thus, from here on, we use greedy decoding when training a model with the proposed multiset loss function.

We now compare the proposed multiset loss function against the five baseline loss functions: reinforcement learning L RL , aggregate distribution matching-L 1 dm and L KL dm -, its one-step variant L 1-step , and sequence prediction L seq .MNIST Multi We present the results on the MNIST Multi variants in TAB1 .

On all three variants and according to both metrics, the proposed multiset loss function outperforms all the others.

The reinforcement learning based approach closely follows behind.

Its performance, however, drops as the number of items in a target multiset increases.

This is understandable, as the variance of policy gradient grows as the length of an episode grows.

A similar behaviour was observed with sequence prediction as well as aggregate distribution matching.

We were not able to train any decent models with the one-step variant of aggregate distribution matching.

This was true especially in terms of exact match (EM), which we attribute to the one-step variant not being capable of modelling dependencies among the predicted items.

TAB3 .

On COCO Easy, with only two objects to predict per example, both aggregated distribution matching (with KL divergence) and the sequence loss functions are as competitive as the proposed multiset loss.

The other loss functions significantly underperform these three loss functions, as they did on MNIST Multi.

The performance gap between the proposed loss and the others, however, grows substantially on the more challenging COCO Medium, which has more objects per example.

The proposed multiset loss outperforms the aggregated distribution matching with KL divergence by 3.7 percentage points on exact match and 4.8 on F1.

This is analogous to the experiments on the MNIST Multi variants, where the performance gap increased when moving from four to ten digits.

One property of the oracle policy defined in Sec. 2.2 is that the entropy of the predictive distribution strictly decreases over time, i.e., H π * (y|ŷ <t , x) > H π * (y|ŷ ≤t , x).

This is a natural consequence from the fact that there is no pre-specified rank function, because the oracle policy cannot prefer any item from the others in a free label multiset.

Hence, we examine here how the policy learned based on each loss function compares to the oracle policy in terms of per-step entropy.

We consider the policies trained on MNIST Multi (10), where the differences among them were most clear.

As shown in FIG1 , the policy trained on MNIST Multi (10) using the proposed multiset loss closely follows the oracle policy.

The entropy decreases as the predictions are made.

The decreases can be interpreted as concentrating probability mass on progressively smaller free labels sets.

The variance is quite small, indicating that this strategy is uniformly applied for any input.

The policy trained with reinforcement learning retains a relatively low entropy across steps, with a decreasing trend in the second half.

We carefully suspect the low entropy in the earlier steps is due to the greedy nature of policy gradient.

The policy receives a high reward more easily by choosing one of many possible choices in an earlier step than in a later step.

This effectively discourages the policy from exploring all possible trajectories during training.

On the other hand, the policy found by aggregated distribution matching (L KL dm ) has the opposite behaviour.

The entropy in general grows as more predictions are made.

To see why this is suboptimal, consider the final (10th) step.

Assuming the first nine predictions {ŷ 1 , ...,ŷ 9 } were correct (i.e. they form a subset of Y), there is only one correct class left for the final predictionŷ 10 .

The high entropy, however, indicates that the model is placing a significant amount of probability on incorrect sequences.

We believe such a policy is found by minimizing the aggregated distribution matching loss function because it cannot properly distinguish between policies with increasing and decreasing entropies.

The increasing entropy also indicates that the policy has learned a rank function implicitly and is fully relying on it.

Given some unknown free label multiset, inferred from the input, this policy uses the implicitly learned rank function to choose one item from this set.

We conjecture this reliance on an inferred rank function, which is by definition sub-optimal, 5 resulted in lower performance of aggregate distribution matching.

We have extensively investigated the problem of multiset prediction in this paper.

We rigorously defined the problem, and proposed to approach it from the perspective of sequential decision making.

In doing so, an oracle policy was defined and shown to be optimal, and a new loss function, called multiset loss, was introduced as a means to train a parametrized policy for multiset prediction.

The experiments on two families of datasets, MNIST Multi variants and MS COCO variants, have revealed the effectiveness of the proposed loss function over other loss functions including reinforcement learning, sequence, and aggregated distribution matching loss functions.

The success of the proposed multiset loss brings in new opportunities of applying machine learning to various new domains, including high-energy physics.

Precision Precision gives the ratio of correctly predicted elements to the number of predicted elements.

Specifically, letŶ = (C, µŶ ), Y = (C, µ Y ) be multisets.

Then DISPLAYFORM0 The summation and membership are done by enumerating the multiset.

For example, the multisetŝ Y = {a, a, b} and Y = {a, b} are enumerated asŶ = {a DISPLAYFORM1 Formally, precision can be defined as DISPLAYFORM2 where the summation is now over the ground set C. Intuitively, precision decreases by 1 |Ŷ| each time an extra class label is predicted.

Recall Recall gives the ratio of correctly predicted elements to the number of ground-truth elements.

Recall is defined analogously to precision, as: Similarly, we start with the definition of the recall: DISPLAYFORM3 Rec(ŷ <t , Y) = y∈ŷ<t I y∈Y |Y| .turned into a conditional distribution over the next item after affine transformation followed by a softmax function.

When the one-step variant of aggregated distribution matching is used, we skip the convolutional LSTM layers, i.e., c = DISPLAYFORM4 See Fig. 2 for the graphical illustration of the entire network.

See TAB4 for the details of the network for each dataset.

conv 5 × 5 max-pool 2 × 2 feat 10 81 conv 3 × 3 feat 32 conv 5 × 5 max-pool 2 × 2 feat 10 conv 3 × 3 feat 32 conv 5 × 5 max-pool 2 × 2 feat 32

ResNet-34 361 conv 3 × 3 feat 512 conv 3 × 3 feat 512Preprocessing For MNIST Multi, we do not preprocess the input at all.

In the case of MS COCO, input images are of different sizes.

Each image is first resized so that its larger dimension has 600 pixels, then along its other dimension is zero-padded to 600 pixels and centered, resulting in a 600x600 image.

Training The model is trained end-to-end, except ResNet-34 which remains fixed after being pretrained on ImageNet.

For all the experiments, we train a neural network using Adam BID7 ) with a fixed learning rate of 0.001, β of (0.9, 0.999) and of 1e-8.

The learning rate was selected based on the validation performance during the preliminary experiments, and the other parameters are the default values.

For MNIST Multi, the batch size was 64, and for COCO was 32.Feedforward Alternative While we use a recurrent model in the experiments, the multiset loss can be used with a feedforward model as follows.

A key use of the recurrent hidden state is to retain the previously predicted labels, i.e. to remember the full conditioning setŷ 1 , ...,ŷ t−1 in p(y t |ŷ 1 , ...,ŷ t−1 ).

Therefore, the proposed loss can be used in a feedforward model by encodinĝ y 1 , ...,ŷ t−1 in the input x t , and running the feedforward model for |Ŷ| steps, where |Ŷ| is determined with a method from section 2.3.

Note that compared to the recurrent model, this approach involves additional feature engineering.

@highlight

We study the problem of multiset prediction and propose a novel multiset loss function, providing analysis and empirical evidence that demonstrates its effectiveness.