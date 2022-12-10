Object recognition in real-world requires handling long-tailed or even open-ended data.

An ideal visual system needs to reliably recognize the populated visual concepts and meanwhile efficiently learn about emerging new categories with a few training instances.

Class-balanced many-shot learning and few-shot learning tackle one side of this problem, via either learning strong classifiers for populated categories or learning to learn few-shot classifiers for the tail classes.

In this paper, we investigate the problem of generalized few-shot learning (GFSL) -- a model during the deployment is required to not only learn about "tail" categories with few shots, but simultaneously classify the "head" and "tail" categories.

We propose the Classifier Synthesis Learning (CASTLE), a learning framework that learns how to synthesize calibrated few-shot classifiers in addition to the multi-class classifiers of ``head'' classes, leveraging a shared neural dictionary.

CASTLE sheds light upon the inductive GFSL through optimizing one clean and effective GFSL learning objective.

It demonstrates superior performances than existing GFSL algorithms and strong baselines on MiniImageNet and TieredImageNet data sets.

More interestingly, it outperforms previous state-of-the-art methods when evaluated on standard few-shot learning.

Visual recognition for objects in the "long tail" has been an important challenge to address (Wang et al., 2017; Liu et al., 2019) .

We often have a very limited amount of data on those objects as they are infrequently observed and/or visual exemplars of them are hard to collect.

As such, state-of-the-art methods (e.g deep learning) can not be directly applied due to their notorious demand of a large number of annotated data (Krizhevsky et al., 2017; Simonyan & Zisserman, 2014; He et al., 2016) .

Few-shot learning (FSL) (Vinyals et al., 2016; Snell et al., 2017; Finn et al., 2017 ) is mindful of the limited instances (i.e, shots) per "tail" concept, which attempts to address this challenging problem by distinguishing between the data-rich "head" categories as SEEN classes and data-scarce "tail" categories as UNSEEN classes.

While it is difficult to build classifiers with data from UNSEEN classes, FSL leverages data from SEEN classes to extract inductive biases for effective classifiers acquisition on UNSEEN ones.

We refer to (Larochelle, 2018) for an up-to-date survey in few-shot learning.

This type of learning, however, creates a chasm in object recognition.

Classifiers from many-shot learning for SEEN classes and those from few-shot learning for UNSEEN classes do not mix -they cannot be combined directly to recognize all object categories at the same time.

In this paper, we study the problem of Generalized Few-Shot Learning (GFSL), which focuses on the joint classification of both data-rich and data-poor categories.

In particular, our goal is for the model trained on the SEEN categories to be capable of incorporating limited UNSEEN class instances, and make predictions for test instances in both the "head" and "tail" of the entire distribution of categories.

Figure 1 illustrates the high-level idea of our proposal, contrasting the standard few-shot learning.

In contrast to prior works (Hariharan & Girshick, 2017; Wang et al., 2017; Liu et al., 2019 ) that focus on learning "head" and "tail" concepts in a transductive manner, our learning setup requires inductive modeling of the"tail", which is therefore more challenging as we assume no knowledge about the UNSEEN "tail" categories is available during the model learning phase. (GFSL) .

GFSL requires to extract inductive bias from SEEN categories to facilitate efficiently learning on few-shot UNSEEN "tail" categories, while maintaining discernability on "head" classes.

To this end, we propose Classifier Synthesis Learning (CASTLE), where the few-shot classifiers are synthesized based on a shared neural dictionary across classes.

Such synthesized few-shot classifiers are then used together with the many-shot classifiers.

To this purpose, we create a scenario, via sampling a set of instances from SEEN categories and pretend that they come from UNSEEN, and apply the synthesized classifiers (based on the instances) as if they are many-shot classifiers to optimize multi-class classification together with the remaining many-shot SEEN classifiers.

In other words, we construct few-shot classifiers to not only perform well on the few-shot classes but also to be competitive when used in conjunction with many-shot classifiers of populated classes.

We argue that such highly contrastive learning can benefit few-shot classification with high discernibility in its learned visual embeddings (cf.

Section 4.2 and Section 4.4).

We empirically validate our approach on two standard benchmark data sets -MiniImageNet and TieredImageNet.

The proposed approach retains competitive "head" concept recognition performances while outperforming existing approaches on few-shot learning and generalized few-shot learning.

We highlight that CASTLE has learned a better calibration between many-shot SEEN classifiers and synthesized UNSEEN classifiers, which naturally addresses the confidence mismatch phenomena , i.e, SEEN and UNSEEN classifiers have different confidence ranges.

We define a K-shot N -way classification task to be one with N classes to make prediction and K training examples per class for learning.

The training set (i.e, support set) is represented as

, where x i ∈ R D is an instance and y i ∈ {0, 1} N (i.e, one-hot vector) is its label.

Similarly, the test set is D test and contains i.i.d.

samples from the same distribution as D train .

From few-shot learning to generalized few-shot learning.

In many-shot learning, where K is large, a classification model f : R D → {0, 1} N is learned by optimizing E (xi,yi)∈D train (f (x i ), y i ).

Here f is often instantiated as an embedding function φ(·) and a linear classifier Θ: f (x i ) = φ(x i ) Θ. The loss function (·, ·) measures the discrepancy between the prediction and the true label.

On the other hand, Few-shot learning (FSL) faces the challenge in transferring knowledge across learning visual concepts.

It assumes two non-overlapping sets of SEEN (S) and UNSEEN (U) classes.

During training, it has access to all SEEN classes for learning an inductive bias, which is then transferred to learn good classifiers on U rapidly with a small K. Generalized Few-Shot Learning (GFSL), different from FSL which neglects classification of the S classes, aims at building models that simultaneously predicts over S ∪ U categories.

As a result, such a model needs to deal with many-shot classification from |S| SEEN classes along side with learning |U| emerging UNSEEN classes 1 .

Meta-learning for few-shot learning.

Meta-learning has been an effective framework for FSL (Vinyals et al., 2016; Finn et al., 2017; Snell et al., 2017) in the recent years.

The main idea is to mimic the future few-shot learning scenario by optimizing a shared f across K-shot N -way tasks drawn from the SEEN class sets S.

In particular, a K-shot N -way task D S train sampled from S is constructed by randomly choosing N categories from S and K examples in each of them.

A corresponding test set D S test (a.k.a.

query set) is sampled from S to evaluate the resulting few-shot classifier f .

Therefore, we expect the learned classifier f "generalizes" well on the training few-shot tasks sampled from SEEN classes, to "generalize" well on few-shot tasks drawn from UNSEEN class set U.

In this paper, we focus on the methods described in (Vinyals et al., 2016; Snell et al., 2017) .

Specifically, the classifier f is based on an embedding function, f = φ : R D → R d , which transforms input examples into a latent space with d dimensions.

φ is learned to pull similar objects close while pushing dissimilar ones far away (Koch et al., 2015) .

For a test instance x j , the embedding function φ makes a prediction based on a soft nearest neighbor classifier:

sim(φ(x j ), φ(x i )) measures the similarity between the test instance φ(x j ) and each training instance φ(x i ).

When there is more than one instance per class, i.e, K > 1, instances in the same class can be averaged to assist make a final decision.

By learning a good φ, important visual features for few-shot classification are distilled, which will be used for few-shot tasks from the UNSEEN classes.

The main idea of CASTLE includes a classifier composition model for synthesizing classifiers with the few-shot training data, and an effective learning algorithm that learns many-shot classifiers and few-shot classifiers (together with its composition model end-to-end) at the same time.

In Section 3.1, we introduce the classifier composition model uses a few-shot training data to query a common set of neural bases, and then assemble the target "synthesized classifiers".

In Section 3.2, we propose a unified learning objective that directly contrasts many-shot classifiers with few-shot classifiers, via constructing classification tasks over U ∪ S categories.

It enforces the few-shot classifiers to explicitly compete against the many-shot classifiers in the model learning, which leads to more discriminative few-shot classifiers in the GFSL setting.

We base our classifier composition model on 2018) .

Different from their approach with a pre-fixed feature embedding, we use a learned embedding function and a neural dictionary.

Here we define a dictionary as pairs of "key" and "value" embeddings, where each "key" and "value" is associated with a neural base, which is designed to encode shared primitives for composing classifiers of S ∪ U. Formally, the neural dictionary contains a set of |B| learnable bases B = {b 1 , b 2 , . . .

, b |B| }, and b k ∈ B ∈ R d .

The key and value for the dictionary are generated based on two linear projections U and V of elements in B. For instance, Ub i and Vb i represent the generated key and value embeddings.

Denote I [ y i = c ] as an indicator that selects instances in the class c. To synthesize a classifier for a class c, we first compute the class signature as the embedding prototype, defined as the average embedding of all K shots of instances (in a K-shot N -way task):

We then compute the coefficients α c for assembling the classifier of class c, via measuring the compatibility score between the class signature and the key embeddings of the neural dictionary,

The coefficient α k c is then normalized with the sum of compatibility scores over all |B| bases, which then is used to convexly combine the value embeddings and synthesize the classifier,

We formulate the classifier composition as a summation of the initial prototype embedding p c and the residual component

classifier is then 2 -normalized and used for (generalized) few-shot classification.

Since both the embedding "key" and classifier "value" are generated based on the same set of neural bases, it encodes a compact set of latent features for a wide range of classes.

We hope the learned neural bases contain a rich set of classifier primitives to be transferred to novel compositions of emerging visual categories.

In addition to transferring knowledge from SEEN to UNSEEN classes as in FSL, in generalized fewshot learning, the few-shot classifiers is required to do well when used in conjunction with many-shot classifiers.

Therefore, a GFSL classifier f should have a low expected error as what follows:

Suppose we have sampled a K-shot N -way few-shot learning task D U train , which contains |U| visual UNSEEN categories.

For each task, the classifier f predicts a test instance in D S∪U test towards both tail classes U and head classes S. In other words, based on D U train and the many-shot classifiers Θ S , a randomly sampled instance in S ∪ U should be effectively predicted.

In summary, a GFSL classifier generalizes its joint prediction ability to S ∪ U given D U train and Θ S during inference.

Unified learning objective.

CASTLE learns a generalizable GFSL classifier via training on the SEEN class set S. For each class in s ∈ S, it keeps many-shot classifiers (i.e, liner classifier over the embedding function φ(·)) Θ s .

Next, we sample a "fake" K-shot N -way few-shot task from S, which contains C categories.

For each classes in C, we synthesize their classifiers by W C = { w c | c ∈ C } as in Eq. 5.

We treat the remaining S − C classes as the "fake" head classes, and use their corresponding many-shot classifiers Θ S−C .

They are combined with the synthesized classifiers W C (from the few-shot classes C) to form the set of joint classifiersŴ = W C ∪ Θ S−C , over all classes in S. Finally, we optimize the learning objective as what follows:

Despite that few-shot classifiers W C are synthesized using with K training instances (cf.

Eq. 3), they are optimized to jointly classify instances from all SEEN categories S. After minimizing the accumulated loss in Eq. 7 over multiple GFSL tasks, the learned model extends its discerning ability to UNSEEN classes so as has low error in Eq. 6.

During inference, CASTLE synthesizes the classifiers for UNSEEN classes based on the neural dictionary with their few-shot training examples, and makes a joint prediction over S ∪ U with the help of many-shot classifier Θ S .

Multi-classifier learning.

A natural way to minimize Eq. 7 implements a stochastic gradient descent step in each mini-batch by sampling one GFSL task, which contains a K-shot N -way training set together with a set of test instances (x j , y j ) from S. It is clear that increasing the number of GFSL tasks per gradient step can improve the optimization stability.

Therefore, we propose an efficient implementation that utilizes a large number of GFSL tasks to compute gradients.

with different sets of C, which is then applied to compute the averaged loss over z using Eq. 7.

In the scope of this paper, CASTLE always uses multi-classifier learning unless it is explicitly mentioned.

With this, we observed a significant speed-up in terms of convergence (cf.

Section C.1 in the appendix for an ablation study).

In this section, we design experiments to validate the effectiveness of the CASTLE in GFSL (cf.

Section 4.2).

We first introduce the training and evaluation protocol of Ren et al. (2018a) and compare CASTLE with existing methods.

Next, we provide an analysis over algorithms with alternative protocols that measures different aspects of GFSL (cf.

Section 4.3).

We verify that CASTLE is advantageous as it learns a better calibration between SEEN and UNSEEN classifiers.

Finally, we show that CASTLE also benefit standard FSL performances (cf.

Section 4.4).

Data sets.

We consider two benchmark data sets derived from ILSVRC-12 dataset (Russakovsky et al., 2015) .

The miniImageNet dataset (Vinyals et al., 2016) has 100 classes and 600 examples per class.

For evaluation, we follow the split of (Ravi & Larochelle, 2017) Figure A5 of the Appendix provides an illustration of how data are split.

Baselines and prior methods.

We explore several (strong) choices in deriving classifiers for the SEEN and UNSEEN classes: (1) Multiclass Classifier (MC) + kNN.

A multi-class classifier is trained on the SEEN classes as standard many-shot classification (He et al., 2016) .

When evaluated on UNSEEN classes for few-shot tasks, we apply the learned feature embedding with a nearest neighbor classifier.

(2) ProtoNet + ProtoNet.

We train Prototypical Network (Snell et al., 2017 ) (a.k.a ProtoNet) on SEEN classes, pretending they were few-shot.

When evaluated on the SEEN categories, we randomly sample 100 training instances per category to compute the class prototypes.

We use the MC classifier's feature mapping to initialize the embedding function, and use the final embedding function for UNSEEN classes.

The prediction is straightforward as both sets of classes are generated with ProtoNet.

(3) MC + ProtoNet.

We combine the learning objective of (1) and (2) to jointly learn the MC classifier and feature embedding, which trades off between few-shot and many-shot learning.

Besides IFSL (Ren et al., 2018a) , we also re-implemented existing approaches (or adapted the original release if available), i.e, L2ML (Wang et al., 2017) and DFSL (Gidaris & Komodakis, 2018) to compare with CASTLE.

Note that L2ML is originally designed in the transductive setting, which we made some adaption for inductive prediction.

Please refer to original papers for details.

For CASTLE, we use the {Θ S } (i.e, the multiclass classifiers, cf.

Section 3.2) for the SEEN classes and the synthesized classifiers for the UNSEEN classes to classify an instance into all classes, and then select the prediction with the highest confidence score.

Evaluation measures.

Mean accuracy over all SEEN and 5 sampled UNSEEN classes is the main measurement to evaluate a GFSL method (Gidaris & Komodakis, 2018; Wang et al., 2018) .

We sample 10,000 1-shot or 5-shot GFSL tasks to evaluate this for the sake of reliability.

Besides the few-shot training examples, an equal number of test instances sampled from all head and 5 tail categories are used during the evaluation.

The mean and 95% confidence interval are reported.

In addition to accuracy, Ren et al. (2018a) also use ∆-value, a measure of average accuracy drop between predicting specific (SEEN or UNSEEN) class and predicting all categories jointly.

Methods balance the prediction of SEEN and UNSEEN classes well can receive a low accuracy drop.

In the later sections, we introduce two other GFSL measures --the harmonic mean accuracy and the area under SEEN-UNSEEN curve (AUSUC).

Please refer to the Section A of the Appendix for more details about experimental setups, implementation details, model optimization, and evaluation measures 3 .

The main results of all methods on miniImageNet is shown in Table 1 .

We found that CASTLE outperforms all the existing methods as well as our proposed baseline systems in terms of the mean accuracy.

Meanwhile, when looked at the ∆-value, CASTLE is least affected between predicting for SEEN/USSEEN classes separately and predicting over all classes jointly.

However, we argue that either mean accuracy or ∆-value is not informative enough to tell about a GFSL algorithm's performances.

For example, a baseline system, i.e, ProtoNet + ProtoNet perform better than IFSL in terms of 5-shot mean accuracy but not ∆-value.

In this case, how shall we rank these two systems?

To answer this question, we propose to use another evaluation measure, harmonic mean of the mean accuracy for each SEEN and UNSEEN category, when they are classified jointly.

Harmonic mean is a better GFSL performance measure.

Since the number of SEEN and UNSEEN classes are most likely to be not equal, e.g, 64 vs. 5 in our cases, directly computing the mean accuracy over all classes is almost always biased.

For example, a many-shot classifier that only classifies samples into SEEN classes can receive a good performance than one that recognizes both SEEN and UNSEEN.

Therefore, we argue that harmonic mean over the mean accuracy can better assess a classifier's performance, as now the performances are negatively affected when a classifier ignores classes (e.g, MC classifier get 0% harmonic mean).

Specifically, we compute the top-1 accuracy for instances from SEEN and UNSEEN classes, and take their harmonic mean as the performance measure.

The results are included in the right side of the Table 1 .

Now we observe that the many-shot baseline MC+kNN has extremely low performance as it tends to ignore UNSEEN categories.

Meanwhile, CASTLE remains the best when ranked by the harmonic mean accuracy against others.

Evaluate GFSL beyond 5 UNSEEN categories.

Besides using harmonic mean accuracy, we argue that another important aspect in evaluating GFSL is to go beyond the 5 sampled UNSEEN categories, as it is never the case in real-world.

On the contrary, we care most about the GFSL with a large number of UNSEEN classes.

To this end, we evaluate GFSL with all available SEEN and UNSEEN categories over both MiniImageNet and TieredImageNet, and report their results in Table 2 and  Table 3 .

We report the mean accuracy over SEEN and UNSEEN categories, as well as the harmonic mean accuracy of all categories.

We observe that CASTLE outperforms all approaches in the UNSEEN and more importantly, the ALL categories section, across two data sets.

On the SEEN categories, CASTLE remains competitive against the ad hoc many-shot classifier (MC).

In this section, we do analyses to show (1) tuning a great confidence calibration factor significantly improves GFSL performance of baseline models, (2) CASTLE has balanced the confidence score of SEEN and UNSEEN predictions, requiring no explicit calibration, and (3) CASTLE is consistently better than other approaches across an increasing number of "tail" categories.

For more ablation studies about CASTLE, we refer readers to the Appendix (cf.

Section C.1).

Confidence calibration matters in GFSL.

In generalized zero-shot learning, has identified a significant prediction bias between classification confidence of SEEN and UNSEEN classifiers.

We find a similar phenomena in GFSL.

For instance, ProtoNet + ProtoNet baseline has a very confident classifier on SEEN categories than UNSEEN categories (The scale of confidence is on average 2.1 times higher).

To address this issue, we compute a calibration factor based on the validation set of UNSEEN categories, such that the prediction logits are calibrated by subtracting this factor out from the confidence of SEEN categories' predictions.

The results of all methods after calibration is shown in Figure 2 .

We observe a consistent improvement over the harmonic mean of accuracy for all methods, while CASTLE is the least affected.

This suggests that CASTLE, learned with the unified GFSL objective, has a well-calibrated classification confidence and does not require additional data and extra learning phase to search this calibration factor.

Moreover, we use area under SEEN-UNSEEN curve (AUSUC) as a measure of different GFSL algorithms.

Here, AUSUC is a performance measure that takes the effects of calibration factor out.

To do so, we enumerate through a large range of calibration factors, and subtract it from the confidence score of SEEN classifiers.

Through this process, the joint prediction performances over SEEN and UNSEEN categories, denoted as S → S ∪ U and U → S ∪ U, shall vary as the calibration factor changes.

For instance, when calibration factor is infinite large, we are measuring a classifier that only predicts UNSEEN categories.

We denote this as the SEEN-UNSEEN curve.

The results is shown in Figure 3 .

As a result, we observe that CASTLE archives the largest area under curve, which indicates that CASTLE is in general a better algorithm over others among different calibration factors.

Robust evaluation of GFSL.

Other than the harmonic mean accuracy of all SEEN and UNSEEN categories shown in cf.

Table 2 and 3, we study the dynamic of how harmonic mean accuracy changes with an incremental number of UNSEEN "tail" concepts.

In other words, we show the GFSL performances w.r.t.

different numbers of "tail" concepts.

We use this as a robust evaluation of each system's GFSL capability.

The 1-shot learning result is shown as Figure 4 .

We observe that CASTLE consistently outperforms other baselines by a clear margin.

Finally, we also evaluate our proposed approach's performance on two standard few-shot learning benchmarks, i.e, miniImageNet and TieredImageNet data set.

The results are shown in the Table 4  and Table 5 .

We compare our approach to previous state-of-the-art methods and found CASTLE ProtoNet (Snell et al., 2017) 61.40 ± 0.02 76.56 ± 0.02 LEO (Rusu et al., 2018) 61.76 ± 0.08 77.59 ± 0.12 OptNet (Lee et al., 2019) 62 Sung et al. (2018) 54.48 ± 0.93 71.32 ± 0.78 LEO (Rusu et al., 2018) 66.33 ± 0.05 81.44 ± 0.09 OptNet (Lee et al., 2019) 65 outperforming all of them, in both 1-shot 5-way and 5-shot 5-way accuracy.

This supports our hypothesis that jointly learning with many-shot classification forces few-shot classifiers to be discriminative.

Please refer to the Appendix for details about task setups, performance measures, and visualizations.

Building a high-quality visual system usually requires to have a large scale annotated training set with many shots per categories.

Many large-scale datasets such as ImageNet have an ample number of instances for popular classes (Russakovsky et al., 2015; Krizhevsky et al., 2017) .

However, the data-scarce "tail" of the category distribution matters.

For example, a visual search engine needs to deal with the rare object of interests (e.g endangered species) or newly defined items (e.g new smartphone models), which only possess a few data instances.

Directly training a system over all classes is prone to over-fit and can be biased towards the data-rich categories.

Few-shot learning (FSL) is proposed to tackle this problem, via meta-learning an inductive bias from the SEEN classes, such that it transfers to the learning process of UNSEEN classes with few training data during the model deployment.

For example, one line of works uses meta-learned discriminative feature embeddings (Snell et al., 2017; Oreshkin et al., 2018; Rusu et al., 2018; Scott et al., 2018; Ye et al., 2018; Lee et al., 2019) together with non-parametric nearest neighbor classifiers, to recognize novel classes given a few exemplars.

Another line of works (Finn et al., 2017; Nichol et al., 2018; Lee & Choi, 2018; Antoniou et al., 2018; Vuorio et al., 2018) chooses to learn a common initialization to a pre-specified model configuration and adapt rapidly using fixed steps of gradient descents over the few-shot training data from UNSEEN categories.

FSL emphasizes on building models of the UNSEEN classes and ignore its real-world use case of assisting the many-shot recognition of the "'head" categories.

A more realistic setting, i.e, low-shot learning, has been studied before (Hariharan & Girshick, 2017; Wang et al., 2018; Gao et al., 2018; Ye et al., 2018; Liu et al., 2019) .

The main aim is to recognize the entire set of concepts in a transductive learning framework -during the training of the target model, you have access to both the SEEN and UNSEEN categories.

The key difference to our proposed GFSL is that we assume no access to UNSEEN classes in the learning phase, which requires the model to inductively transfer knowledge from SEEN classes to UNSEEN ones during the evaluation.

Previous approaches mostly focus on the transductive setup of GFSL.

Some of them (Hariharan & Girshick, 2017; Wang et al., 2018; Gao et al., 2018) apply the exemplar-based classification paradigms on both SEEN and UNSEEN categories to resolve the transductive learning problem.

Others (Wang et al., 2017; Schönfeld et al., 2018; Liu et al., 2019) usually ignore the explicit relationship between SEEN and UNSEEN categories, and learn separate classifiers.

Ren et al. (2018a) ; Gidaris & Komodakis (2018) propose to solve inductive GFSL via either composing UNSEEN with SEEN classifiers or meta-leaning with recurrent back-propagation procedure.

Gidaris & Komodakis (2018) is the most related work to CASTLE, where we differ in how we compose classifiers and the unified learning objective, i.e, we used a learned neural dictionary instead of using MC classifiers as bases.

In summary, CASTLE learns both many-shot classifiers and synthesized classifiers via optimizing a single unified objective function, where a classifier composition model with a neural dictionary is leveraged for assembling few-shot classifiers.

Our experiments highlight that CASTLE not only outperforms existing methods in terms of GFSL performances from many different aspects, but more interestingly, also improves the classifier's discernibility over standard FSL.

Following the recent methods (Qiao et al., 2017; Rusu et al., 2018; Ye et al., 2018) , we use a residual network (He et al., 2016 ) (ResNet) to implement the embedding backbone φ.

We first pre-train this backbone network (also explored by (Qiao et al., 2017; Rusu et al., 2018; Ye et al., 2018; Lee et al., 2019) ) and perform model selection strategy similar to (Ye et al., 2018) .

To learn our methods as well as baseline systems, we then use Momentum SGD with an initial learning rate 1e-4.

In the rest of this section, we explain each of the above with complete details.

A.1 DATA SET DETAILS.

Two benchmark data sets are used in our experiments.

The MiniImageNet dataset (Vinyals et al., 2016 ) is a subset of the ILSVRC-12 dataset (Russakovsky et al., 2015) .

There are totally 100 classes and 600 examples in each class.

For evaluation, we follow the split of (Ravi & Larochelle, 2017) and use 64 of 100 classes for meta-training, 16 for validation, and 20 for meta-test (model evaluation).

In other words, a model is trained on few-shot tasks sampled from the 64 SEEN classes set during meta-training, and the best model is selected based on the few-shot classification performance over the 16 class set.

The final model is evaluated based on few-shot tasks sampled from the 20 UNSEEN classes.

The TieredImageNet (Ren et al., 2018b ) is a more complicated version compared with the miniImageNet.

It contains 34 super-categories in total, with 20 for meta-training, 6 for validation, and 8 for model testing (meta-test).

Each of the super-category has 10 to 30 classes.

In detail, there are 351, 97, and 160 classes for meta-training, meta-validation, and meta-test, respectively.

The divergence of the super-concept leads to a more difficult few-shot classification problem.

Since both data sets are constructed by images from ILSVRC-12, we augment the meta-train set of each data set by sampling non-overlapping images from the corresponding classes in ILSVRC-12.

The auxiliary meta-train set is used to measure the generalized few-shot learning classification performance on the SEEN class set.

For example, for each of the 64 SEEN classes in the MiniImageNet, we collect 200 more non-overlapping images per class from ILSVRC-12 as the test set for many-shot classification.

An illustration of the data set split is shown in Figure A5 .

Figure A5 : The split of data in the generalized few-shot classification scenario.

In addition to the standard data set like MiniImagetnet (blue part), we collect non-overlapping augmented "head" class instances from the corresponding categories in the ImageNet (red part), to measure the classification ability on the seen classes.

Then in the generalized few-shot classification task, few-shot instances are sampled from each of the unseen classes, while the model should have the ability to predict instances from both the "head" and "tail" classes.

Following the setting of most recent methods (Qiao et al., 2017; Rusu et al., 2018; Ye et al., 2018) , we use the residual network (He et al., 2016) to implement the embedding backbone φ.

Different from the standard configuration, the literature (Qiao et al., 2017; Rusu et al., 2018; Ye et al., 2018) resize the input image to 80 × 80 × 3 for MiniImageNet (while 84 × 84 × 3 for TieredImageNet) and remove the first two down-sampling layers in the network.

In concrete words, three residual blocks are used after an initial convolutional layer (with stride 1 and padding 1) over the image, which have channels 160/320/640, stride 2, and padding 2.

After a global average pooling layer, it leads to a 640 dimensional embedding.

The concrete architecture is visualized as Figure A15 .

Please refer to Pytorch documentation 4 for complete references of each building blocks.

Before the meta-training stage, we try to find a good initialization for the embedding φ.

In particular, on MiniImageNet we add a linear layer on the backbone output and optimize a 64-way (while 351-way for TieredImageNet) classification problem on the meta-training set with the cross-entropy loss function.

Stochastic gradient descent with initial learning rate 0.1 and momentum 0.9 is used to complete such optimization.

The 16 classes in MiniImageNet (resp.

97 classes in TieredImageNet) for model selection also assist the choice of the pre-trained model.

After each epoch, we use the current embedding and measures the nearest neighbor based few-shot classification performance on the sampled few-shot tasks from these 16 (resp.

97) classes.

The most suitable embedding function is recorded.

After that, such learned backbone is used to initialize the embedding part φ of the whole model.

In later sections, we will show the effect of pre-training strategy on both few-shot and generalized few-shot classification measures.

We use the pre-trained backbone to initialize the embedding part φ of a model for CASTLE and our re-implemented comparison methods such as MC+kNN, ProtoNet+ProtoNet, MC+ProtoNet, L2ML (Wang et al., 2017) , and DFSL (Gidaris & Komodakis, 2018) .

When there exists a backbone initialization, we set the initial learning rate as 1e-4 and optimize the model with Momentum SGD.

The learning rate will be halved after optimizing 2,000 mini-batches.

During meta-learning, all methods are optimized over 5-way few-shot tasks, where the number of shots in a task is consistent with the inference (meta-test) stage.

For example, if the goal is a 1-shot 5-way model, we sample 1-shot 5-way D test .

An illustration of the architecture of CASTLE is shown in Figure A6 .

For CASTLE, we randomly sample a 24-way task from S in each mini-batch, and re-sample 64 5-way tasks from it.

It is notable that all instances in the 24-way task are encoded by the ResNet backbone with same parameters in advance.

Therefore, by embedding the synthesized 5-way few-shot classifiers into the global many-shot classifier, it results in 64 different configurations of the generalized few-shot classifiers.

To evaluate which we randomly sample instances with batch size 128 from S and compute the GFSL objective in Eq. 7.

In this section, we provide details about the training and evaluation setups for the generalized few-shot learning, followed by concrete descriptions for comparison methods.

Setup.

We train a multi-class classifier on the populated SEEN classes following practices of training Residual Networks (He et al., 2016) .

Here a ResNet backbone network is used, identical to the ones described in Section A.2.

During the training |S|-way classifiers are trained in a supervised learning manner.

Training details.

During the inference, test examples of S categories are evaluated based on the |S|-way classifiers and |U| categories are evaluated using the support embeddings from D U train with a nearest neighbor classifier.

To evaluate the generalized few-shot classification task, we take the union of multi-class classifiers' confidence and ProtoNet confidence as joint classification scores on S ∪ U.

Setup.

We train a few-shot classifier (initialized by the MC classifier's feature mapping) using the Prototypical Network (Snell et al., 2017 ) (a.k.a ProtoNet).

The backbone network is the same ResNet as before.

Training and inference.

During the inference, we compute the class prototypes of SEEN classes via using 100 training instances per category.

The class prototypes of UNSEEN classes are computed based on the sampled few-shot training set.

During the inference of generalized few-shot learning, the confidence of a test instances is jointly determined by its (negative) distance to both SEEN and UNSEEN class prototypes.

Setup.

We combine the learning objective of the previous two baselines to jointly learn the MC classifier and feature embedding.

Since there are two objectives for many-shot (cross-entropy loss on all SEEN classes) and few-shot (ProtoNet meta-learning objective) classification respectively, it trades off between many-shot and few-shot learning.

Therefore, this learned model can be used as multi-class linear classifiers on the "head" categories, and used as ProtoNet on the "tail" categories.

Training and inference.

During the inference, the model predicts instances from SEEN class S with the MC classifier, while takes advantage of the few-shot prototypes to discern UNSEEN class instances.

Figure A7 : An illustration of the harmonic mean based GFSL evaluation.

S and U denotes the SEEN and UNSEEN instances (x) and labels (y) respectively.

S ∪ U is the joint set of S and U. The notation X → Y, X, Y ∈ {S, U, S ∪ U} means computing prediction results with instances from X to labels of Y .

By computing a performance measure (like accuracy) on the joint label space prediction of SEEN and UNSEEN instances separately, a harmonic mean is computed to obtain the final measure.

To evaluate the generalized few-shot classification task, we take the union of multi-class classifiers' confidence and ProtoNet confidence as joint classification scores on S ∪ U.

Setup.

Wang et al. (2017) propose learning to model the "tail" (L2ML) by connecting a few-shot classifier with the corresponding many-shot classifier.

The method is designed to learn classifier dynamics from data-poor "tail" classes to the data-rich "head" classes.

Since L2ML is originally designed to learn with both SEEN and UNSEEN classes in a transductive manner, in our experiment, we adaptive it to out setting.

Therefore, we learn a classifier mapping based on the sampled few-shot tasks from SEEN class set S, which transforms a few-shot classifier in UNSEEN class set U inductively.

Training and inference.

Following (Wang et al., 2017) , we first train a many-shot classifier W upon the ResNet backbone on the SEEN class set S. We use the same residual architecture as in (Wang et al., 2017) to implement the classifier mapping f , which transforms a few-shot classifier to a many-shot classifier.

During the meta-learning stage, a S-way few-shot task is sampled in each mini-batch, which produces a S-way linear few-shot classifierŴ based on the fixed pre-trained embedding.

The objective of L2ML not only regresses the mapped few-shot classifier f (Ŵ ) close to the many-shot one W measured by square loss, but also minimize the classification loss of f (Ŵ ) over a randomly sampled instances from S. Therefore, this learned model uses a pre-trained multi-class classifier W for those "head" categories, and used the predicted few-shot classifiers with f for the "tail" categories.

Setup.

Dynamic Few-Shot Learning without forgetting (DFSL) (Gidaris & Komodakis, 2018 ) also adopts a generalized few-shot learning objective.

It decomposes the GFSL learning with two stages.

A cosine classifier together with the backbone is learned at first.

The pre-trained cosine classifier is regarded as bases.

Based on the fixed backbone, another attention-based network constructs the classifier for a particular class by a linear combination of the elements in the bases.

Training and inference.

We follow the strategy in (Gidaris & Komodakis, 2018) to train the DFSL model.

Based on the pre-trained backbone and cosine classifier, we construct a dictionary with size |S| whose elements correspond to each category in S. In each mini-batch of meta-training, we sample a few-shot task from the SEEN class set whose classes construct the set C. Then, an attention model composes the classifier for the few-shot task by weighting the |S| − |C| elements in the dictionary not corresponding to C. To evaluate the composed classifier, DFSL samples an equal number of instances from C and S − C for a test.

For inference, we use the cosine classifier for "head" classes and composed few-shot classifier for "tail" classes.

We take advantage of the auxiliary meta-train set from the benchmark data sets during GFSL evaluations, and an illustration of the data set construction can be found in Figure A5 .

The notation X → Y with X, Y ∈ {S, U, S ∪ U} means computing prediction results with instances from X to labels of Y .

For example, S → S ∪ U means we first filter instances come from the SEEN class set (x ∈ S), and predict them into the joint label space (y ∈ S ∪ U).

For a GFSL model, we consider its performance with different measurements.

An illustration of some criteria is shown in Figure A7 .

Many-shot accuracy.

A model is required to predict the auxiliary SEEN class instances towards all SEEN classes (S → S).

This is the same criterion with the standard supervised learning.

Few-shot accuracy.

Following the standard protocol (Vinyals et al., 2016; Finn et al., 2017; Snell et al., 2017; Ye et al., 2018) , we sample 10,000 K-shot N -way tasks from U during inference.

In detail, we first sample N classes from U, and then sample K + 15 instances for each class.

The first N K labeled instances (K instances from each of the N classes) are used to build the few-shot classifier, and the remaining 15N (15 instances from each of the N classes) are used to evaluate the quality of such few-shot classifier.

During our test, we consider K = 1 and K = 5 as in the literature, and change N ranges from {5, 10, 15, . . .

, |U|} as a more robust measure.

It is noteworthy that in this test stage, all the instances come from U and are predicted to classes in U (U → U).

Generalized few-shot accuracy.

Different from many-shot and few-shot evaluations, the generalized few-shot learning takes the joint instance and label spaces into consideration.

In other words, the instances come from S ∪ U and their predicted labels also in S ∪ U (S ∪ U → S ∪ U).

This is obviously more difficult than the previous many-shot (S → S) and few-shot (U → U) tasks.

During the test, with a bit abuse of notations, we sample K-shot S + N -way tasks from S ∪ U. Concretely, we first sample a K-shot N -way task from U, with N K training and 15N test instances respectively.

Then, we randomly sample 15N instances from S. Thus in a GFSL evaluation task, there are N K labeled instances from U, and 30N test instances from S ∪ U.

We compute the accuracy of S ∪ U as the final measure.

Generalized few-shot ∆-value.

Since the problem becomes difficult when the predicted label space expands from S → S to S → S ∪ U (and also U → U to U → S ∪ U), the accuracy of a model will have a drop.

To measure how the classification ability of a GFSL model changes when working in a GFSL scenario, Ren et al. (2018a) propose the ∆-Value to measure the average accuracy drop.

In detail, for each sampled GFSL task, we first compute its many-shot accuracy (S → S) and few-shot accuracy (U → U).

Then we calculate the corresponding accuracy of SEEN and UNSEEN instances in the joint label space, i.e, S → S ∪ U and U → S ∪ U.

The ∆-Value is the average decrease of accuracy in these two cases.

Generalized few-shot harmonic mean.

Directly computing the accuracy still gets biased towards the populated classes, so we also consider the harmonic mean as a more balanced measure (Xian et al., 2017) .

By computing performance measurement such as top-1 accuracy and sample-wise Mean Average Precision (MAP) for S → S ∪ U and U → S ∪ U, the harmonic mean is used to average the performance in these two cases as the final measure.

An illustration is in Figure A7 .

Generalized few-shot AUSUC. propose a calibration-agnostic criterion for generalized zero-shot learning.

To avoid evaluating a model influenced by a calibration factor between SEEN and UNSEEN classes, they propose to determine the range of the calibration factor for all instances at first, and then plot the SEEN-UNSEEN accuracy curve based on different configurations of the calibration values.

Finally, the area under the SEEN-UNSEEN curve is used as a more robust criterion.

We follow to compute the AUSUC value for sampled GFSL tasks.

In this section, we first do ablation studies on the proposed CASTLE approach, and then provide additional results for comparison methods in the GFSL evaluations.

In this section, we aim to study the ablated variant of our approach and perform in-depth analyses.

Effects on the neural dictionary size |B|.

We show the effects of the dictionary size (as the ratio of SEEN class size) for the generalized few-shot learning (measured by harmonic mean accuracy when there are 5 UNSEEN classes) in Figure A8 .

We observe that the neural dictionary with a ratio of 2 or 3 works best amongst all other dictionary sizes.

Therefore, we fix the dictionary size as 128 across all experiments.

Note that when |B| = 0, our method degenerates to case optimizing the unified objective in Eq. 7 without using the neural dictionary.

How well is synthesized classifiers comparing multi-class classifiers?

To assess the quality of synthesized classifier, we made a comparison against ProtoNet and also the Multi-class Classifier on the "head" SEEN concepts.

To do so, we sample few-shot training instances on each SEEN category to synthesize classifiers (or compute class prototypes for ProtoNet), and then use solely the synthesized classifiers/class prototypes to evaluate multi-class accuracy.

The results are shown in the Figure A9 .

We observe that the learned synthesized classifier outperforms over ProtoNet by a large margin.

Also, the model trained with unified learning objective (ULO) improves over the vanilla synthesized classifiers.

Note that there is still a significant gap left against multi-class classifiers trained on the entire data set.

It suggests that the classifier synthesis we learned is effective against using sole instance embeddings while still far from the many-shot multi-class classifiers.

Different choices of the classifier synthesis.

As in Eq. 3, when there are more than one instance per class in a few-shot task (i.e K > 1), CASTLE compute the averaged embeddings first, and then use the prototype of each class as the input of the neural dictionary to synthesize their corresponding classifiers.

Here we explore another choice to deal with multiple instances in each class.

We synthesize classifiers based on each instance first, and then average the corresponding synthesized classifiers for each class.

This option equals an ensemble strategy to average the prediction results of each instance's synthesized classifier.

We denote the pre-average strategy (the one used in CASTLE) as "Pre-AVG", and the post-average strategy as "Post-AVG".

The 5-Shot 5-way classification results on MiniImageNet for these two strategies are shown in Table A6 .

From the results, "Post-AVG" does not improve the FSL and GFSL performance obviously.

Since averaging the synthesized classifiers in a hindsight way costs more memory during meta-training, we choose the "Pre-AVG" option to synthesize classifiers when there are more than 1 shot in each class.

What is the performance when evaluated with more UNSEEN classes?

As mentioned in the analysis of the main text, we now give additional five-shot learning results for the incremental evaluation of the generalized few-shot learning (together with one-shot learning results).

In addition to the test instances from the "head" 64 classes in MiniImageNet, 5 to 20 novel classes are included to compose the generalized few-shot tasks.

Concretely, 1 or 5 instances per novel class are used to construct the "tail" classifier, combined with which the model is asked to do a joint classification of both SEEN and UNSEEN classes.

Figure A10 and Figure A11 record the change of generalized few-shot learning performance (harmonic mean) when more UNSEEN classes emerge.

We observe that CASTLE consistently outperforms all baseline approaches in each evaluation setup, with a clear margin.

How is multiple classifiers learning's impact over the training? (cf.

Section 3) CASTLE adopts a multi-classifier training strategy, i.e considering multiple GFSL tasks with different combinations of classifiers in a single mini-batch.

Here we show the influence of the multi-classifier training method based on their FSL and GFSL performance.

Figure A12 and Figure A13 show the change of loss and harmonic mean accuracy (with 5 UNSEEN tasks) when training CASTLE with different number of classifiers based on a pre-trained backbone, respectively.

It is obvious that training with multiple classifiers converges faster and generalizes better than the vanilla model, without increasing the computational burden a lot.

A more detailed comparison for training with different numbers of classifiers is listed in Table A7 , which verifies the effectiveness of the multi-classifier training strategy.

In this subsection, we provide concrete values for the GFSL measurements on MiniImageNet.

To avoid repetition, only the results of 1-Shot GFSL tasks are listed.

From Table A8 to Table A11 , the number of ways of UNSEEN classes in an inference GFSL task varies from 5 to 20.

In addition to the top-1 accuracy, the sample-wise mean average precision (MAP) is also calculated as a basic measure before harmonic mean.

As shown in Figure A7 , the harmonic mean is the harmonic average of the joint prediction performance of SEEN (S → S ∪ U) and UNSEEN (U → S ∪ U) instances.

Although CASTLE cannot achieve high joint label space prediction on SEEN class instances (S → S ∪ U), its high harmonic mean performance results from its competitive discerning ability on the joint prediction of UNSEEN instances (S → S ∪ U).

Table A9 : Concrete evaluation criteria for generalized few-shot classification measurements on MiniImageNet.

The GFSL tasks are composed by 1-shot 10-Way UNSEEN class.

"HM" denotes the harmonic mean.

As mentioned before, to obtain better generalized few-shot learning performances, a confidence calibration procedure between predictions for S and U is necessary.

We therefore tune this factor based on the validation UNSEEN classes (e.g in the MiniImageNet cases, we use 16 validation classes to compute this value) and then applied to the evaluation on test UNSEEN classes (e.g corresponding to the 20 test categories in MiniImageNet ).

Table A12 : Concrete evaluation criteria for generalized few-shot classification measurements on MiniImageNet.

The GFSL tasks are composed by 1-shot 5-Way UNSEEN class, and the harmonic mean is computed with a calibration factor.

"HM" denotes the harmonic mean.

As mentioned in the main text, now we show the complete details and more results of the study with regard to the effects of calibration factors.

The importance of the calibration factor has already been validated in Wang et al., 2018) .

We exactly follow the strategy in to complete the calibration by subtracting a bias on the prediction logits of all SEEN classes.

In other words, different from the vanilla prediction, a calibration bias is subtracted from the confidence for SEEN classes, to make it balanced with the predictions for the unseen parts.

In detail, we choose the range of the bias by sampling 200 generalized few-shot tasks composed by validation instances and record the difference between the maximum value of SEEN and UNSEEN logits.

The averaged difference value is used as the range of the bias selection.

30 equally split calibration bias values are used as candidates, and the best one is chosen based on 500 generalized few-shot tasks sampled from the meta-validation set.

As a result, we observe that calibrated methods can have a consistent improvement over the harmonic mean of accuracy.

The results are listed from Table A12 to Table A15 , and the number of UNSEEN classes in a GFSL task changes from 5 to 20.

Comparing with the results without calibration factor in Table A8 -A11, the additional calibration step increases the joint prediction ability of UNSEEN instances a lot, so as to improve the final harmonic mean measurement.

Our CASTLE get similar results after using the calibration bias, especially when there are 5 UNSEEN classes.

Therefore, CASTLE fits the generalized few-shot learning task, and does not require additional calibration step to balance the SEEN and UNSEEN predictions.

To show the discriminative ability of the learned embedding, we visualize the embedding of 6 randomly selected UNSEEN classes with 50 instances per class from MiniImageNet in Figure A14 .

The embedding results of four baseline approaches, namely MC + kNN, ProtoNet + ProtoNet, MC + ProtoNet, and CASTLE are shown.

It can be found that CASTLE grasps the instance relationship of UNSEEN classes better than others.

<|TLDR|>

@highlight

We propose to learn synthesizing few-shot classifiers and many-shot classifiers using one single objective function for GFSL.