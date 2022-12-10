Few-shot classiﬁcation aims to learn a classiﬁer to recognize unseen classes during training with limited labeled examples.

While signiﬁcant progress has been made, the growing complexity of network designs, meta-learning algorithms, and differences in implementation details make a fair comparison difﬁcult.

In this paper, we present 1) a consistent comparative analysis of several representative few-shot classiﬁcation algorithms, with results showing that deeper backbones signiﬁcantly reduce the gap across methods including the baseline, 2) a slightly modiﬁed baseline method that surprisingly achieves competitive performance when compared with the state-of-the-art on both the mini-ImageNet and the CUB datasets, and 3) a new experimental setting for evaluating the cross-domain generalization ability for few-shot classiﬁcation algorithms.

Our results reveal that reducing intra-class variation is an important factor when the feature backbone is shallow, but not as critical when using deeper backbones.

In a realistic, cross-domain evaluation setting, we show that a baseline method with a standard ﬁne-tuning practice compares favorably against other state-of-the-art few-shot learning algorithms.

Deep learning models have achieved state-of-the-art performance on visual recognition tasks such as image classification.

The strong performance, however, heavily relies on training a network with abundant labeled instances with diverse visual variations (e.g., thousands of examples for each new class even with pre-training on large-scale dataset with base classes).

The human annotation cost as well as the scarcity of data in some classes (e.g., rare species) significantly limit the applicability of current vision systems to learn new visual concepts efficiently.

In contrast, the human visual systems can recognize new classes with extremely few labeled examples.

It is thus of great interest to learn to generalize to new classes with a limited amount of labeled examples for each novel class.

The problem of learning to generalize to unseen classes during training, known as few-shot classification, has attracted considerable attention BID29 ; BID27 ; BID6 ; BID25 ; BID28 ; BID9 ; BID24 .

One promising direction to few-shot classification is the meta-learning paradigm where transferable knowledge is extracted and propagated from a collection of tasks to prevent overfitting and improve generalization.

Examples include model initialization based methods BID25 ; BID6 , metric learning methods BID29 ; BID27 ; BID28 , and hallucination based methods BID0 ; BID11 ; BID31 .

Another line of work BID10 ; BID24 also demonstrates promising results by directly predicting the weights of the classifiers for novel classes.

Limitations.

While many few-shot classification algorithms have reported improved performance over the state-of-the-art, there are two main challenges that prevent us from making a fair comparison and measuring the actual progress.

First, the discrepancy of the implementation details among multiple few-shot learning algorithms obscures the relative performance gain.

The performance of baseline approaches can also be significantly under-estimated (e.g., training without data augmentation).

Second, while the current evaluation focuses on recognizing novel class with limited training examples, these novel classes are sampled from the same dataset.

The lack of domain shift between the base and novel classes makes the evaluation scenarios unrealistic.

Our work.

In this paper, we present a detailed empirical study to shed new light on the few-shot classification problem.

First, we conduct consistent comparative experiments to compare several representative few-shot classification methods on common ground.

Our results show that using a deep backbone shrinks the performance gap between different methods in the setting of limited domain differences between base and novel classes.

Second, by replacing the linear classifier with a distance-based classifier as used in BID10 ; BID24 , the baseline method is surprisingly competitive to current state-of-art meta-learning algorithms.

Third, we introduce a practical evaluation setting where there exists domain shift between base and novel classes (e.g., sampling base classes from generic object categories and novel classes from fine-grained categories).

Our results show that sophisticated few-shot learning algorithms do not provide performance improvement over the baseline under this setting.

Through making the source code and model implementations with a consistent evaluation setting publicly available, we hope to foster future progress in the field.

1 Our contributions.1.

We provide a unified testbed for several different few-shot classification algorithms for a fair comparison.

Our empirical evaluation results reveal that the use of a shallow backbone commonly used in existing work leads to favorable results for methods that explicitly reduce intra-class variation.

Increasing the model capacity of the feature backbone reduces the performance gap between different methods when domain differences are limited.2.

We show that a baseline method with a distance-based classifier surprisingly achieves competitive performance with the state-of-the-art meta-learning methods on both mini-ImageNet and CUB datasets.3.

We investigate a practical evaluation setting where base and novel classes are sampled from different domains.

We show that current few-shot classification algorithms fail to address such domain shifts and are inferior even to the baseline method, highlighting the importance of learning to adapt to domain differences in few-shot learning.

Given abundant training examples for the base classes, few-shot learning algorithms aim to learn to recognizing novel classes with a limited amount of labeled examples.

Much efforts have been devoted to overcome the data efficiency issue.

In the following, we discuss representative few-shot learning algorithms organized into three main categories: initialization based, metric learning based, and hallucination based methods.

Initialization based methods tackle the few-shot learning problem by "learning to fine-tune".

One approach aims to learn good model initialization (i.e., the parameters of a network) so that the classifiers for novel classes can be learned with a limited number of labeled examples and a small number of gradient update steps BID6 BID22 BID26 .

Another line of work focuses on learning an optimizer.

Examples include the LSTM-based meta-learner for replacing the stochastic gradient decent optimizer BID25 and the weight-update mechanism with an external memory BID21 .

While these initialization based methods are capable of achieving rapid adaption with a limited number of training examples for novel classes, our experiments show that these methods have difficulty in handling domain shifts between base and novel classes.

Distance metric learning based methods address the few-shot classification problem by "learning to compare".

The intuition is that if a model can determine the similarity of two images, it can classify an unseen input image with the labeled instances BID16 .

To learn a sophisticated comparison models, meta-learning based methods make their prediction conditioned on distance or metric to few labeled instances during the training process.

Examples of distance metrics include cosine similarity BID29 , Euclidean distance to class-mean representation BID27 , CNN-based relation module BID28 , ridge regression BID1 , and graph neural network BID9 .

In this paper, we compare the performance of three distance metric learning methods.

Our results show that a simple baseline method with a distancebased classifier (without training over a collection of tasks/episodes as in meta-learning) achieves competitive performance with respect to other sophisticated algorithms.

Besides meta-learning methods, both BID10 and BID24 develop a similar method to our Baseline++ (described later in Section 3.2).

The method in BID10 learns a weight generator to predict the novel class classifier using an attentionbased mechanism (cosine similarity), and the BID24 directly use novel class features as their weights.

Our Baseline++ can be viewed as a simplified architecture of these methods.

Our focus, however, is to show that simply reducing intra-class variation in a baseline method using the base class data leads to competitive performance.

Hallucination based methods directly deal with data deficiency by "learning to augment".

This class of methods learns a generator from data in the base classes and use the learned generator to hallucinate new novel class data for data augmentation.

One type of generator aims at transferring appearance variations exhibited in the base classes.

These generators either transfer variance in base class data to novel classes BID11 , or use GAN models BID0 to transfer the style.

Another type of generators does not explicitly specify what to transfer, but directly integrate the generator into a meta-learning algorithm for improving the classification accuracy BID31 .

Since hallucination based methods often work with other few-shot methods together (e.g. use hallucination based and metric learning based methods together) and lead to complicated comparison, we do not include these methods in our comparative study and leave it for future work.

Domain adaptation techniques aim to reduce the domain shifts between source and target domain BID23 ; BID8 , as well as novel tasks in a different domain BID14 .

Similar to domain adaptation, we also investigate the impact of domain difference on fewshot classification algorithms in Section 4.5.

In contrast to most domain adaptation problems where a large amount of data is available in the target domain (either labeled or unlabeled), our problem setting differs because we only have very few examples in the new domain.

Very recently, the method in BID5 addresses the one-shot novel category domain adaptation problem, where in the testing stage both the domain and the category to classify are changed.

Similarly, our work highlights the limitations of existing few-shot classification algorithms problem in handling domain shift.

To put these problem settings in context, we provided a detailed comparison of setting difference in the appendix A1.

In this section, we first outline the details of the baseline model (Section 3.1) and its variant (Section 3.2), followed by describing representative meta-learning algorithms (Section 3.3) studied in our experiments.

Given abundant base class labeled data X b and a small amount of novel class labeled data X n , the goal of few-shot classification algorithms is to train classifiers for novel classes (unseen during training) with few labeled examples.

Our baseline model follows the standard transfer learning procedure of network pre-training and fine-tuning.

FIG0 illustrates the overall procedure.

Training stage.

We train a feature extractor f θ (parametrized by the network parameters θ ) and the classifier C(·|W b ) (parametrized by the weight matrix W b ∈ R d×c ) from scratch by minimizing a standard cross-entropy classification loss L pred using the training examples in the base classes Fine-tuning stage.

To adapt the model to recognize novel classes in the fine-tuning stage, we fix the pre-trained network parameter θ in our feature extractor f θ and train a new classifier C(.|W n ) (parametrized by the weight matrix W n ) by minimizing L pred using the few labeled of examples (i.e., the support set) in the novel classes X n .

DISPLAYFORM0

In addition to the baseline model, we also implement a variant of the baseline model, denoted as Baseline++, which explicitly reduces intra-class variation among features during training.

The importance of reducing intra-class variations of features has been highlighted in deep metric learning BID15 and few-shot classification methods BID10 .The training procedure of Baseline++ is the same as the original Baseline model except for the classifier design.

As shown in FIG0 , we still have a weight matrix W b ∈ R d×c of the classifier in the training stage and a W n in the fine-tuning stage in Baseline++.

The classifier design, however, is different from the linear classifier used in the Baseline.

Take the weight matrix W b as an example.

We can write the weight matrix W b as [w 1 , w 2 , ...w c ], where each class has a d-dimensional weight vector.

In the training stage, for an input feature f θ (x i ) where x i ∈ X b , we compute its cosine similarity to each weight vector [w 1 , · · · , w c ] and obtain the similarity scores DISPLAYFORM0 We can then obtain the prediction probability for each class by normalizing these similarity scores with a softmax function.

Here, the classifier makes a prediction based on the cosine distance between the input feature and the learned weight vectors representing each class.

Consequently, training the model with this distance-based classifier explicitly reduce intra-class variations.

Intuitively, the learned weight vectors [w 1 , · · · , w c ] can be interpreted as prototypes (similar to BID27 ; BID29 ) for each class and the classification is based on the distance of the input feature to these learned prototypes.

The softmax function prevents the learned weight vectors collapsing to zeros.

We clarify that the network design in Baseline++ is not our contribution.

The concept of distancebased classification has been extensively studied in BID18 and recently has been revisited in the few-shot classification setting BID10 ; BID24 .

Here we describe the formulations of meta-learning methods used in our study.

We consider three distance metric learning based methods (MatchingNet Vinyals et al. (2016) , ProtoNet Snell et al. (2017) ).

While meta-learning is not a clearly defined, BID29 considers a few-shot classification method as meta-learning if the prediction is conditioned on a small support set S, because it makes the training procedure explicitly learn to learn from a given small support set.

As shown in FIG1 , meta-learning algorithms consist of a meta-training and a meta-testing stage.

In the meta-training stage, the algorithm first randomly select N classes, and sample small base support set S b and a base query set Q b from data samples within these classes.

The objective is to train a classification model M that minimizes N-way prediction loss L N−way of the samples in the query set Q b .

Here, the classifier M is conditioned on provided support set S b .

By making prediction conditioned on the given support set, a meta-learning method can learn how to learn from limited labeled data through training from a collection of tasks (episodes).

In the meta-testing stage, all novel class data X n are considered as the support set for novel classes S n , and the classification model M can be adapted to predict novel classes with the new support set S n .Different meta-learning methods differ in their strategies to make prediction conditioned on support set (see FIG1 ).

For both MatchingNet Vinyals et al. (2016) and ProtoNet Snell et al. (2017) , the prediction of the examples in a query set Q is based on comparing the distance between the query feature and the support feature from each class.

MatchingNet compares cosine distance between the query feature and each support feature, and computes average cosine distance for each class, while ProtoNet compares the Euclidean distance between query features and the class mean of support features.

RelationNet BID28 shares a similar idea, but it replaces distance with a learnable relation module.

The MAML method BID6 is an initialization based meta-learning algorithm, where each support set is used to adapt the initial model parameters using few gradient updates.

As different support sets have different gradient updates, the adapted model is conditioned on the support set.

Note that when the query set instances are predicted by the adapted model in the meta-training stage, the loss of the query set is used to update the initial model, not the adapted model.

Datasets and scenarios.

We address the few-shot classification problem under three scenarios: 1) generic object recognition, 2) fine-grained image classification, and 3) cross-domain adaptation.

For object recognition, we use the mini-ImageNet dataset commonly used in evaluating few-shot classification algorithms.

The mini-ImageNet dataset consists of a subset of 100 classes from the ImageNet dataset BID4 and contains 600 images for each class.

The dataset was first proposed by BID29 , but recent works use the follow-up setting provided by BID25 , which is composed of randomly selected 64 base, 16 validation, and 20 novel classes.

For fine-grained classification, we use CUB-200-2011 dataset BID30 (referred to as the CUB hereafter).

The CUB dataset contains 200 classes and 11,788 images in total.

Following the evaluation protocol of BID13 , we randomly split the dataset into 100 base, 50 validation, and 50 novel classes.

For the cross-domain scenario (mini-ImageNet →CUB), we use mini-ImageNet as our base class and the 50 validation and 50 novel class from CUB.

Evaluating the cross-domain scenario allows us to understand the effects of domain shifts to existing few-shot classification approaches.

Implementation details.

In the training stage for the Baseline and the Baseline++ methods, we train 400 epochs with a batch size of 16.

In the meta-training stage for meta-learning methods, we train 60,000 episodes for 1-shot and 40,000 episodes for 5-shot tasks.

We use the validation set to select the training episodes with the best accuracy.

2 In each episode, we sample N classes to form N-way classification (N is 5 in both meta-training and meta-testing stages unless otherwise mentioned).

For each class, we pick k labeled instances as our support set and 16 instances for the query set for a k-shot task.

In the fine-tuning or meta-testing stage for all methods, we average the results over 600 experiments.

In each experiment, we randomly sample 5 classes from novel classes, and in each class, we also pick k instances for the support set and 16 for the query set.

For Baseline and Baseline++, we use the entire support set to train a new classifier for 100 iterations with a batch size of 4.

For meta-learning methods, we obtain the classification model conditioned on the support set as in Section 3.3.All methods are trained from scratch and use the Adam optimizer with initial learning rate 10 −3 .

We apply standard data augmentation including random crop, left-right flip, and color jitter in both the training or meta-training stage.

Some implementation details have been adjusted individually for each method.

For Baseline++, we multiply the cosine similarity by a constant scalar 2 to adjust original value range [-1,1] to be more appropriate for subsequent softmax layer.

For MatchingNet, we use an FCE classification layer without fine-tuning in all experiments and also multiply cosine similarity by a constant scalar.

For RelationNet, we replace the L2 norm with a softmax layer to expedite training.

For MAML, we use a first-order approximation in the gradient for memory efficiency.

The approximation has been shown in the original paper and in our appendix to have nearly identical performance as the full version.

We choose the first-order approximation for its efficiency.

We now conduct experiments on the most common setting in few-shot classification, 1-shot and 5-shot classification, i.e., 1 or 5 labeled instances are available from each novel class.

We use a four-layer convolution backbone (Conv-4) with an input size of 84x84 as in BID27 and perform 5-way classification for only novel classes during the fine-tuning or meta-testing stage.

To validate the correctness of our implementation, we first compare our results to the reported numbers for the mini-ImageNet dataset in Table 1 .

Note that we have a ProtoNet # , as we use 5-way classification in the meta-training and meta-testing stages for all meta-learning methods as mentioned in Section 4.1; however, the official reported results from ProtoNet uses 30-way for one shot and 20-way for five shot in the meta-training stage in spite of using 5-way in the meta-testing stage.

We report this result for completeness.

From Table 1 , we can observe that all of our re-implementation for meta-learning methods do not fall more than 2% behind reported performance.

These minor differences can be attributed to our Table 1 : Validating our re-implementation.

We validate our few-shot classification implementation on the mini-ImageNet dataset using a Conv-4 backbone.

We report the mean of 600 randomly generated test episodes as well as the 95% confidence intervals.

Our reproduced results to all few-shot methods do not fall behind by more than 2% to the reported results in the literature.

We attribute the slight discrepancy to different random seeds and minor implementation differences in each method.

"Baseline * " denotes the results without applying data augmentation during training.

ProtoNet # indicates performing 30-way classification in 1-shot and 20-way in 5-shot during the meta-training stage.

modifications of some implementation details to ensure a fair comparison among all methods, such as using the same optimizer for all methods.

Moreover, our implementation of existing work also improves the performance of some of the methods.

For example, our results show that the Baseline approach under 5-shot setting can be improved by a large margin since previous implementations of the Baseline do not include data augmentation in their training stage, thereby leads to over-fitting.

While our Baseline * is not as good as reported in 1-shot, our Baseline with augmentation still improves on it, and could be even higher if our reproduced Baseline * matches the reported statistics.

In either case, the performance of the Baseline method is severely underestimated.

We also improve the results of MatchingNet by adjusting the input score to the softmax layer to a more appropriate range as stated in Section 4.1.

On the other hand, while ProtoNet # is not as good as ProtoNet, as mentioned in the original paper a more challenging setting in the meta-training stage leads to better accuracy.

We choose to use a consistent 5-way classification setting in subsequent experiments to have a fair comparison to other methods.

This issue can be resolved by using a deeper backbone as shown in Section 4.3.After validating our re-implementation, we now report the accuracy in TAB1 .

Besides additionally reporting results on the CUB dataset, we also compare Baseline++ to other methods.

Here, we find that Baseline++ improves the Baseline by a large margin and becomes competitive even when compared with other meta-learning methods.

The results demonstrate that reducing intra-class variation is an important factor in the current few-shot classification problem setting.

FIG2 and TAB9 for larger figure and detailed statistics.)However, note that our current setting only uses a 4-layer backbone, while a deeper backbone can inherently reduce intra-class variation.

Thus, we conduct experiments to investigate the effects of backbone depth in the next section.

In this section, we change the depth of the feature backbone to reduce intra-class variation for all methods.

See appendix for statistics on how network depth correlates with intra-class variation.

Starting from Conv-4, we gradually increase the feature backbone to Conv-6, ResNet-10, 18 and 34, where Conv-6 have two additional convolution blocks without pooling after Conv-4.

ResNet-18 and 34 are the same as described in BID12 with an input size of 224×224, while ResNet-10 is a simplified version of ResNet-18 where only one residual building block is used in each layer.

The statistics of this experiment would also be helpful to other works to make a fair comparison under different feature backbones.

Results of the CUB dataset shows a clearer tendency in FIG2 .

As the backbone gets deeper, the gap among different methods drastically reduces.

Another observation is how ProtoNet improves rapidly as the backbone gets deeper.

While using a consistent 5-way classification as discussed in Section 4.2 degrades the accuracy of ProtoNet with Conv-4, it works well with a deeper backbone.

Thus, the two observations above demonstrate that in the CUB dataset, the gap among existing methods would be reduced if their intra-class variation are all reduced by a deeper backbone.

However, the result of mini-ImageNet in FIG2 is much more complicated.

In the 5-shot setting, both Baseline and Baseline++ achieve good performance with a deeper backbone, but some metalearning methods become worse relative to them.

Thus, other than intra-class variation, we can assume that the dataset is also important in few-shot classification.

One difference between CUB and mini-ImageNet is their domain difference in base and novel classes since classes in mini-ImageNet have a larger divergence than CUB in a word-net hierarchy BID19 .

To better understand the effect, below we discuss how domain differences between base and novel classes impact few-shot classification results.

To further dig into the issue of domain difference, we design scenarios that provide such domain shifts.

Besides the fine-grained classification and object recognition scenarios, we propose a new cross-domain scenario: mini-ImageNet →CUB as mentioned in Section 4.1.

We believe that this is practical scenario since collecting images from a general class may be relatively easy (e.g. due to increased availability) but collecting images from fine-grained classes might be more difficult.

We conduct the experiments with a ResNet-18 feature backbone.

As shown in TAB4 , the Baseline outperforms all meta-learning methods under this scenario.

While meta-learning methods learn to learn from the support set during the meta-training stage, they are not able to adapt to novel classes that are too different since all of the base support sets are within the same dataset.

A similar concept is also mentioned in BID29 .

In contrast, the Baseline simply replaces and trains a new classifier based on the few given novel class data, which allows it to quickly adapt to a novel class and is less affected by domain shift between the source and target domains.

The Baseline also performs better than the Baseline++ method, possibly because additionally reducing intra-class variation compromises adaptability.

In Figure 4 , we can further observe how Baseline accuracy becomes relatively higher as the domain difference gets larger.

That is, as the domain difference grows larger, the adaptation based on a few novel class instances becomes more important.

To further adapt meta-learning methods as in the Baseline method, an intuitive way is to fix the features and train a new softmax classifier.

We apply this simple adaptation scheme to MatchingNet and ProtoNet.

For MAML, it is not feasible to fix the feature as it is an initialization method.

In contrast, since it updates the model with the support set for only a few iterations, we can adapt further by updating for as many iterations as is required to train a new classification layer, which is 100 updates as mentioned in Section 4.1.

For RelationNet, the features are convolution maps rather than the feature vectors, so we are not able to replace it with a softmax.

As an alternative, we randomly split the few training data in novel class into 3 support and 2 query data to finetune the relation module for 100 epochs.

The results of further adaptation are shown in FIG3 ; we can observe that the performance of MatchingNet and MAML improves significantly after further adaptation, particularly in the miniImageNet →CUB scenario.

The results demonstrate that lack of adaptation is the reason they fall behind the Baseline.

However, changing the setting in the meta-testing stage can lead to inconsistency with the meta-training stage.

The ProtoNet result shows that performance can degrade in sce-narios with less domain difference.

Thus, we believe that learning how to adapt in the meta-training stage is important future direction.

In summary, as domain differences are likely to exist in many real-world applications, we consider that learning to learn adaptation in the meta-training stage would be an important direction for future meta-learning research in few-shot classification.

In this paper, we have investigated the limits of the standard evaluation setting for few-shot classification.

Through comparing methods on a common ground, our results show that the Baseline++ model is competitive to state of art under standard conditions, and the Baseline model achieves competitive performance with recent state-of-the-art meta-learning algorithms on both CUB and mini-ImageNet benchmark datasets when using a deeper feature backbone.

Surprisingly, the Baseline compares favorably against all the evaluated meta-learning algorithms under a realistic scenario where there exists domain shift between the base and novel classes.

By making our source code publicly available, we believe that community can benefit from the consistent comparative experiments and move forward to tackle the challenge of potential domain shifts in the context of few-shot learning.

As mentioned in Section 2, here we discuss the relationship between domain adaptation and fewshot classification to clarify different experimental settings.

As shown in Table A1 , in general, domain adaptation aims at adapting source dataset knowledge to the same class in target dataset.

On the other hand, the goal of few-shot classification is to learn from base classes to classify novel classes in the same dataset.

Several recent work tackle the problem at the intersection of the two fields of study.

For example, cross-task domain adaptation BID14 also discuss novel classes in the target dataset.

In contrast, while BID20 has "few-shot" in the title, their evaluation setting focuses on classifying the same class in the target dataset.

If base and novel classes are both drawn from the same dataset, minor domain shift exists between the base and novel classes, as we demonstrated in Section 4.4.

To highlight the impact of domain shift, we further propose the mini-ImageNet →CUB setting.

The domain shift in few-shot classification is also discussed in BID5 .

Different meta-learning works use different terminology in their works.

We highlight their differences in appendix TAB1 to clarify the inconsistency.

For character recognition, we use the Omniglot dataset BID17 commonly used in evaluating few-shot classification algorithms.

Omniglot contains 1,623 characters from 50 languages, and we follow the evaluation protocol of BID29 to first augment the classes by rotations in 90, 180, 270 degrees, resulting in 6492 classes.

We then follow BID27 to split these classes into 4112 base, 688 validation, and 1692 novel classes.

Unlike BID27 , our validation classes are only used to monitor the performance during meta-training.

For cross-domain character recognition (Omniglot→EMNIST), we follow the setting of BID5 to use Omniglot without Latin characters and without rotation augmentation as base classes, so there are 1597 base classes.

On the other hand, EMNIST dataset BID2 contains 10-digits and upper and lower case alphabets in English, so there are 62 classes in total.

We split these classes into 31 validation and 31 novel classes, and invert the white-on-black characters to black-on-white as in Omniglot.

We use a Conv-4 backbone with input size 28x28 for both settings.

As Omniglot characters are black-and-white, center-aligned and rotation sensitive, we do not use data augmentation in this experiment.

To reduce the risk of over-fitting, we use the validation set to select the epoch or episode with the best accuracy for all methods, including baseline and baseline++.

4 As shown in TAB4 , in both Omniglot and Omniglot→EMNIST settings, meta-learning methods outperform baseline and baseline++ in 1-shot.

However, all methods reach comparable performance in the 5-shot classification setting.

We attribute this to the lack of data augmentation for the baseline and baseline++ methods as they tend to over-fit base classes.

When sufficient examples in novel classes are available, the negative impact of over-fitting is reduced.

BID29 ) apply a Baseline with 1-NN classifier in the test stage.

We include our result as in TAB8 .

The result shows that using 1-NN classifier has better performance than that of using the softmax classifier in 1-shot setting, but softmax classifier performs better in 5-shot setting.

We note that the number here are not directly comparable to results in BID29 because we use a different mini-ImageNet as in BID25 .

FIG0 .

We observe that while the full version MAML converge faster, both versions reach similar accuracy in the end.

This phenomena is consistent with the difference of first-order (e.g. gradient descent) and secondorder methods (e.g. Newton) in convex optimization problems.

Second-order methods converge faster at the cost of memory, but they both converge to similar objective value.

As mentioned in Section 4.3, here we demonstrate decreased intra-class variation as the network depth gets deeper as in FIG1 .

We use the Davies-Bouldin index BID3 to measure intra-class variation.

The Davies-Bouldin index is a metric to evaluate the tightness in a cluster (or class, in our case).

Our results show that both intra-class variation in the base and novel class feature decrease using deeper backbones.

Here we use Davies-Bouldin index to represent intra-class variation, which is a metric to evaluate the tightness in a cluster (or class, in our case).

The statistics are Davies-Bouldin index for all base and novel class feature (extracted by feature extractor learned after training or meta-training stage) for CUB dataset under different backbone.

Here we show a high-resolution version of FIG2 in FIG2 and show detailed statistics in TAB9 for easier comparison.

We experiment with a practical setting that handles different testing scenarios.

Specifically, we conduct the experiments of 5-way meta-training and N-way meta-testing (where N = 5, 10, 20) to examine the effect of testing scenarios that are different from training.

As in Table A6 , we compare the methods Baseline, Baseline++, MatchingNet, ProtoNet, and RelationNet.

Note that we are unable to apply the MAML method as MAML learns the initialization for the classifier and can thus only be updated to classify the same number of classes.

Our results show that for classification with a larger N-way in the meta-testing stage, the proposed Baseline++ compares favorably against other methods in both shallow or deeper backbone settings.

We attribute the results to two reasons.

First, to perform well in a larger N-way classification setting, one needs to further reduce the intra-class variation to avoid misclassification.

Thus, Baseline++ has better performance than Baseline in both backbone settings.

Second, as meta-learning algorithms were trained to perform 5-way classification in the meta-training stage, the performance of these algorithms may drop significantly when increasing the N-way in the meta-testing stage because the tasks of 10-way or 20-way classification are harder than that of 5-way one.

One may address this issue by performing a larger N-way classification in the meta-training stage (as suggested in BID27 ).

However, it may encounter the issue of memory constraint.

For example, to perform a 20-way classification with 5 support images and 15 query images in each class, we need to fit a batch size of 400 (20 x (5 + 15)) that must fit into the GPUs.

Without special hardware parallelization, the large batch size may prevent us from training models with deeper backbones such as ResNet.

Table A6 : 5-way meta-training and N-way meta-testing experiment.

The experimental results are on mini-ImageNet with 5-shot.

We could see Baseline++ compares favorably against other methods in both shallow or deeper backbone settings.

@highlight

 A detailed empirical study in few-shot classification that revealing challenges in standard evaluation setting and showing a new direction.