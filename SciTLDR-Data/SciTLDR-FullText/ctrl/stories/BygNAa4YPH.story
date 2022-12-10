In many real-world settings, a learning model must perform few-shot classification: learn to classify examples from unseen classes using only a few labeled examples per class.

Additionally, to be safely deployed, it should have the ability to detect out-of-distribution inputs: examples that do not belong to any of the classes.

While both few-shot classification and out-of-distribution detection are popular topics, their combination has not been studied.

In this work, we propose tasks for out-of-distribution detection in the few-shot setting and establish benchmark datasets, based on four popular few-shot classification datasets.

Then, we propose two new methods for this task and investigate their performance.

In sum, we establish baseline out-of-distribution detection results using standard metrics on new benchmark datasets and show improved results with our proposed methods.

Few-shot learning, at a high-level, is the paradigm of learning where a model is asked to learn about new concepts from only a few examples (Fei-Fei et al., 2006; Lake et al., 2015) .

In the case of fewshot classification, a model must classify examples from novel classes, based on only a few labelled examples from each class.

The model has to quickly learn (or adapt) a classifier given this very limited amount of learning signal.

This paradigm of learning is attractive for the fundamental reason that it resembles how an intelligent system in the real-world has to behave.

Unlike the traditional supervised setting, in most real-world settings we would not have access to millions of labelled examples, but would benefit if a few-shot classifier could be deployed, for example, to recognize the facial gestures of a new user, in order to improve human-computer interaction for individuals with motor disabilities (Wang et al., 2019) .

For an intelligent system to be deployed in the real-world, not only does it have to do well on the designated task, but perhaps more importantly it should defer its actions when faced with unforeseen situations.

In particular, when an input is invalid, or does not belong to any of the target classes, the system should identify the input as out-of-distribution.

Successfully detecting out-of-distribution examples is crucial in a safety critical environment.

In the supervised setting, out-of-distribution detection has been studied from many different angles (Hendrycks & Gimpel, 2016; Nalisnick et al., 2018) , but this task has not been investigated in the few-shot setting.

Worryingly, the current state-of-the-art learning systems, deep neural networks, are known to be unreasonably confident about inputs unrecognizable to humans (Nguyen et al., 2015) , and their predictions can be manipulated with imperceptible changes in input space (Szegedy et al., 2013) .

In general, the behavior of deep nets is not well specified when the test queries are out-of-distribution.

A standard practice when studying out-of-distribution detection is to evaluate the detection performance when examples from other datasets are mixed into the test set (Hendrycks & Gimpel, 2016) .

Here we refer to this type of out-of-distribution input as out-of-dataset (OOS) 1 inputs.

In the few-shot setting, within each episode, what is in-distribution is specified based on a few labeled examples, known as the support set.

Hence, there naturally exists another type of out-of-distribution input, the inputs that belong to the same dataset but come from classes not represented by the support set.

We refer to these as out-of-episode (OOE) examples.

These different types of out-of-distribution examples are illustrated in Figure 1 .

Out-of-Episode Out-of-Dataset Figure 1 : Examples of the support set, indistribution, OOE and OOS inputs in one episode.

Being able to detect out-of-distribution examples is critical for improvements in many other important applications, including semisupervised learning and continual learning.

In the case of semi-supervised learning methods, it was shown that if the unlabelled set is polluted with only 25% out-of-distribution examples, then using the unlabeled data actually has a negative effect on performance (Oliver et al., 2018) .

In the natural continual learning framework, where a model has to learn new concepts while not forgetting old ones, detecting when examples do not belong to any previously-learned class is a fundamental problem.

Hence, in this work, we focus on this core problem of out-of-distribution detection in the few-shot setting.

• We develop benchmark datasets for out-of-distribution detection, both OOE and OOS, based on four standard benchmark datasets for few-shot classification: Omniglot, CIFAR100, miniImageNet, and tieredImageNet.

• We establish baseline results for both the OOS and OOE tasks for two popular few-shot classifiers-Prototypical Networks and MAML-on these datasets.

• We show that a simple distance metric-based approach dramatically improves the performance on both tasks.

• Finally, we propose a learned scoring function which further improves both tasks on the most challenging new benchmark datasets.

Many systems not only evaluate on episodes but also train episodically, i.e., looping over episodes as opposed to over typical mini-batches.

In few-shot classification, a model is tasked to classify unlabeled 'queries' Q = {x i } N Q i=1 into one of N C classes from a set C test .

This setup differs from standard 'supervised' classification in that only a few labeled examples are available from each class c ∈ C test , referred to as that class' support set

.

Following the standard terminology, we refer to the number of classes N C as the 'way' of the task and the number of support examples per class N S as the 'shot' of the task.

We also use the term episode to refer to a classification task defined by a support and a query set.

While we assume that little data is available for each such test classification episode, the model has access to a (possibly large) training set beforehand that contains examples from a different set of classes C train , disjoint from C test .

The key is therefore to figure out how to exploit this seemingly-irrelevant data at training time in order to obtain a model that is capable of learning a new episode at test time using only its small support set in a way that performs well on classifying its corresponding query set.

Most recent approaches for this adopt the design choice of creating episodes from the training set of classes too, and expressing the training loss for each episode in terms of performing well on its query examples, after having 'learned' on its small support set.

The intuition is to practice learning on episodes that have the same structure as those that will be encountered at test time.

At training time, these episodes are created by randomly sampling N C classes (from the training set of classes), N S examples of each of those classes to form the support set, and some different examples of each of them to form the query set.

We refer to this type of training as 'episodic training' (see Figure 2) .

Different methods are distinguished by the manner in which learning is performed on the support set.

We now give an overview of two popular approaches to few-shot learning: Prototypical Networks (Snell et al., 2017) and MAML (Finn et al., 2017) .

Prototypical Networks (Snell et al., 2017) are a simple but effective instance of the above framework where the 'learning procedure' that the model undergoes based on the support set has a closed form.

More concretely, it consists of a parameterized embedding function f φ (typically a deep net) and a distance metric d(·, ·) on the embedding space.

Given support sets of the chosen classes, the Prototypical Network computes the prototype µ c of each class c in the embedding space:

Then a query x in is classified based on its distance to the class prototypes:

During training episodes, the parameters of f φ are updated according to the Prototypical Network loss:

Algorithm 2 (in Appendix B) is a description of standard episodic training of a Prototypical Network.

Meta-learning.

MAML (Finn et al., 2017 ) is another popular model of this episodic family that is parameterized by a representation function and a linear classification layer on top, where jointly we denote the weights as ψ.

Training unfolds over a sequence of training episodes, as usual.

In each episode, the weights ψ are adapted via a few steps of gradient descent (denoted as SGD parameters (L)) to minimize the cross entropy loss over the N C -way classification on the support set, resulting in updated weights φ which are then used to classify the queries in the given episode.

Over a number of episodes, the aggregated loss is then used to update ψ again with gradient descent.

The model is thus encouraged to learn a global initialization ψ of weights such that a few steps of adaptation on a new episode's support set suffice for performing well on its query set.

The term "out-of-distribution" refers to input data that is drawn from a different generative process than that of the training data.

Hendrycks & Gimpel (2016) used different benchmark datasets as sources of out-of-distribution examples.

For example, when a network is trained on MNIST, the out-of-distribution examples can come from Omniglot, black-and-white CIFAR10, etc.

Another common evaluation setup is to treat data from the same dataset-but from different classes than those under consideration-as out-of-distribution.

These have been referred to as same manifold (Liang et al., 2017) , or unobserved class (Louizos & Welling, 2017) out-of-distribution examples.

Problem Set-up.

Out-of-distribution detection is a binary detection problem.

At test-time, the model is required to produce a score, s θ (x) ∈ R, where x is the query, and θ is the set of learnable parameters for the detection task.

We desire s θ (x in ) > s θ (x out ), i.e, the scores for in-distribution examples are higher than that of out-of-distribution examples.

Typically for quantitative evaluation, threshold-free metrics are used, e.g., the area under the receiver-operating curve (AUROC) (see Section 5 for details).

Approaches.

The main approaches to out-ofdistribution detection can be categorized into one of the three families: 1) scores based on the predictive probability of a classifier; 2) scores based on fitting a density model to the inputs directly; and 3) scores based on fitting a density model to representations of a pretrained model (e.g., a classifier).

These are illustrated in Figure 3.

1.

Predictive probability -Recall that classification of the in-distribution data is done using p φ (y = c|x in ) where φ represents the classifier parameters.

Commonly used scores include softmax prediction probability (SPP), s(x in ; φ) = max c p φ (y = c |x in ) (Hendrycks & Gimpel, 2016) and negative predictive entropy (NPE), s(

Note that we use the notation s(·; φ) to emphasize that these scores operate on top of pretrained classifiers.

A popular extension is to use Bayesian classifiers, i.e., Bayesian Neural Networks (BNNs), and improve the scores by looking at the aggregated score based on the model posterior.

2.

Input density -Another natural approach to detecting out-of-distribution examples is to fit a density model on the data and consider examples with low likelihood under that model to be OOD.

However, this approach is not as competitive when the input domain is high-dimensional images.

Nalisnick et al. (2018) showed that deep generative models (e.g., flow-based models (Kingma & Dhariwal, 2018) or auto-regressive models (Salimans et al., 2017) ) can assign higher densities to out-of-distribution examples than in-distribution examples.

3.

Representation density -While fitting a density model on the inputs directly has not proven useful for OOD detection, fitting simple density models on learned classifier representations has.

Lee et al. (2018b) fit a Mixture-of-Gaussian (MoG) density with shared diagonal covariance on the classifier activations of the training set.

Intuitively, this approach fits a density model in a space where much of the variation in the input has been filtered out, which makes it an easier problem than learning a density model in the input space.

In this study, we focus on two types of out-of-distribution detection problems, described below.

In both cases, we denote the set of in-distribution and out-of-distribution examples by Q = {x

where N Q is the number of examples.

Note that we use N Q to denote the number of queries in an episode, and the number of in-distribution/out-of-distribution examples, as they mean the same thing depending on context.

One could consider different numbers of OOD examples from in-distribution ones, but this is omitted for presentation clarity.

Out-of-Episode (OOE).

OOE examples come from the same dataset, but from classes not in the current episode.

In other words, if the current episode consists of classes in C episode , we sample OOE examples R as follows:

Here, D C denotes the set of all examples of classes in set C , \ is the set difference.

This type of out-of-distribution detection is easily motivated.

Taking the example where we want to build a Algorithm 1 Episodic training with OOE inputs.

Modified steps are highlighted in blue.

1: while not converged do 2:

for c in C episode do for each class 4:

end for 7:

{S, Q, R}) 10: end while customized facial gesture recognizer for a user, when the system sees the user's face performing a gesture that is not registered (i.e., not in the support set), we would like the system to know that the gesture is out-of-distribution, and not perform an inappropriate action.

OOS examples come from a completely different dataset.

For example, if the in-distribution set is Omniglot, then the OOS examples can come from black-and-white CIFAR10.

The motivation for this type of out-of-distribution example is also straightforward: a system should defer its actions when faced with something completely different from what it was trained on.

Generally, we use s(·) to denote the scoring function for out-of-distribution detection, which expresses the model's 'confidence' that an example is in-distribution.

Hence, we desire that s(x in ) > s(x out ) for any in-distribution query x in and out-of-distribution example x out .

In what follows, we propose two novel methods: 1) a parameter-free method that measures the distance in the learned embedding of a few-shot classifier, and 2) a learned scoring function on top of the embedding of a few-shot classifier.

(1).

Minimum Distance Confidence Score (-MinDist).

To illustrate why standard softmax prediction probability (SPP) fails in the few-shot setting, consider the classifier learned by Prototypical Network.

The original Prototypical Network formulation makes decisions based on a softmax over the negative distances in the embedding space.

However, when a query embedding is far away from all prototypes (as we may expect for OOS examples), converting distances to probabilities can yield arbitrarily confident predictions (for details see Appendix E).

This makes SPP unsuitable for OOS detection.

We propose an alternative confidence score, based on the negative minimum distance from a query to any of the prototypes:

Episodic Optimization with OOE Inputs.

When training our backbone, we can also add a term to our loss to encourage it to accurately detect OOE examples, in addition to accurately performing the episode's classification task.

Intuitively, adding this term changes the embedding in such a way that the optimized confidence score performs well on the OOE task.

This new term is the following:

where s(·; φ) here can be any of the parameter-free scores, and σ(·) is the logistic function.

Algorithm 1 is a description of episodic training with OOE examples.

(2).

Learnable Class BOundary (LCBO) Network.

We introduce a parametric, class-conditional confidence score that takes a query x and a class c, and yields a score indicating whether x belongs to class c.

The LCBO takes as input: 1) the support embeddings for a particular class, and 2) a query embedding.

The LCBO outputs a real-valued score representing the confidence that the query belongs to the corresponding class.s

Aggregation.

The LCBO outputs class-conditional confidence scores (e.g., the confidence that a query belongs to a specific class).

To obtain a final score for in-distribution vs OOS for each query, we aggregate the class-conditional scores.

We take the maximum confidence of all the classes:

Intuitively,s θ (x in , S c ) computes the distance between a query embedding and a prototype, and the max() aggregation function says that a query is an inlier if it belongs to at least one class.

By design, this is strictly more powerful than -MinDist since it is parameterized by a new set of weights θ, but could also recover simple distance between x in and µ, i.e., -MinDist.

The difficulty of designing a good uncertainty estimate based on a trained classifier leads us to believe that adding capacity to the confidence score using learnable parameters can be beneficial.

Implementation Details.

We parameterize the learned confidence score s θ by an MLP with two hidden layers of dimension 100, that takes as input the concatenation [µ c ; f φ (x in )] where µ c is the class prototype and f φ (x in ) is the query embedding.

Note that LCBO always operates on top of the backbone f φ (·), so this dependency is omitted for notational simplicity.

Training the LCBO.

We train the LCBO episodically.

However, instead of training the aggregated score, we use the following binary cross-entropy objective on the score before aggregation:

For the OOE queries x out , we assigned them a label drawn from the uniform distribution of the in-distribution classes.

In this section, we: 1) establish the OOE and OOS detection performance of standard few-shot methods as well as a novel variant, and 2) show that both our proposed methods improve substantially over these baseline approaches.

To enable fair comparisons, for the experiments in this section we use the same network configuration, a standard 4-layer ConvNet architecture that is well-established in the few-shot literature (Snell et al., 2017) .

None of the methods discussed here sacrifice in-distribution classification accuracy.

Evaluation Metrics.

We evaluate the OOE and OOS detection performance using the area under the receiver-operating curve (AUROC).

This is a simple metric that circumvents the need to set a threshold for the score.

The base-rate (i.e., a completely naïve scoring function) for all of our experiments is 50%.

A scoring function that can completely separate s(x in ) from s(x out ) would achieve an AUROC score of 100%.

Following standard practice (Hendrycks & Gimpel, 2016; Liang et al., 2017; Lee et al., 2018a) , we also report scores for area under the precision and recall curve (AUPR), and false positive rate (FPR) ( Table 1 ).

All results are evaluated using 1000 test episodes, i.e., episodes that contain classes never seen during training.

Please refer to Appendix C for descriptions of the in-distribution and OOS datasets.

We first evaluate the out-of-episode and out-of-distribution detection performance of three few-shot classifiers, using the standard SPP confidence score.

The results are summarized in Table 1 .

We note that not only are these classifiers similar in their distribution classification accuracy , but their ability to detect out-of-distribution examples is also similar.

Amit & Meir (2018) .

However, ABML did not significantly improve over MAML, at least according to our implementation (since Ravi & Beatson (2019) did not release code, in App.

F we discuss details of our best effort to reproduce this method).

Next we show that out-of-distribution performance can be greatly improved.

Few-shot classification can be evaluated in many different (way, shot) settings, e.g., 5-way 5-shot, 10-way 1-shot, etc.

Due to lack of space, we report only 5-shot 5-way results in this section.

Full results for {5, 10}-way × {1, 5}-shot settings on CIFAR-100 are provided in Appendix I. Table 2 shows the results of SPP, -MinDist, and the learned LCBO score on all four of the datasets.

Across the board, on both OOE and OOS tasks, either -MinDist or LCBO outperformed the baseline method.

Interestingly, it seems to confirm our hypothesis that -MinDist might not be the most suitable confidence score for all embedding spaces.

For a more detailed discussion of -MinDist and its connection to a similar method proposed in the supervised setting (Lee et al., 2018b) , please see Appendix E. On the largest datasets, i.e., both versions of the ImageNet dataset, LCBO outperformed -MinDist on both OOE and OOS tasks.

This was somewhat surprising, since one might expect that parameter-free functions like -MinDist can generalize better to OOS datasets that are very different from the in-distribution data.

This was still true on CIFAR100, but not on ImageNet datasets.

One major difference between CIFAR100 and ImageNet was the image size (32 × 32 vs 84 × 84), which resulted in different embedding dimensions (256 vs 1600).

This suggests that as we scale up the dimensionality of the embedding space, it becomes increasingly difficult to design a suitable parameter-free confidence score.

Hence, a learnable score such as LCBO becomes critical.

Effect of different backbones.

We also investigated the effect of the backbone network, f φ , on OOE and OOS detection.

Recently, trained larger backbones like ResNet without using episodic training.

We include results with these larger backbones in Appendix I.

Ren et al. (2018) proposed to study few-shot semi-supervised learning (FS-SSL), where each episode is augmented with an unlabelled set.

To make it more realistic, there are also 'distractors' present.

In previous FS-SSL studies, only OOE examples are considered for both training and testing phases.

This is somewhat unrealistic, as there can be unforeseen distractors in the test episodes.

In Table 3 we show that when evaluated in this more realistic setting, the method of Ren et al. (2018) suffers.

Here, we do not claim that LCBO improves upon semi-supervised learning methods.

Still, especially in the case when distractor inputs are OOS instead of only OOE examples, baseline semi-supervised methods significantly degrade the classification accuracy (see Appendix G for more details on this task).

As few-shot out-of-distribution detection is a new problem, here we discuss recent attempts to study uncertainty in the few-shot setting, and previous approaches that worked well for out-of-distribution detection in the supervised setting.

Other Out-of-Distribution Approaches.

ODIN (Liang et al., 2017) consists of 2 innovations: 1) it performs temperature scaling to calibrate the predicted probability (Guo et al., 2017) ; and 2) when doing out-of-distribution detection, it adds virtual adversarial perturbations (VAP) to the input.

Intuitively, VAP will have a larger effect on the in-distribution input compared to the out-ofdistribution input.

Lee et al. (2018b) showed that this approach can be complementary to fitting a Gaussian density to the activations of the network.

Our preliminary experiments showed that ODIN did not have a big impact in the few-shot setting.

Outlier exposure, another recent method (Hendrycks et al., 2019) , also did not show a significant effect.

We included these results in Appendix J.

For a while, methods using the predictive probability were the dominant approach in out-ofdistribution detection.

Nalisnick et al. (2018) pointed out that the community had been using the learned density model incorrectly by directly looking at the p(x) scores, and instead should use a measure of typicality (Nalisnick et al., 2019) .

Ren et al. (2019) proposed to train a separate "background" model and use the likelihood ratio as the score.

Generative/density models have not been extensively studied in the few-shot setting.

We believe that this is due to the lack of a good task/quantitative evaluation, and that the tasks we study might facilitate research done on such models.

Another topic similar to ours is generalized zero-shot recognition (Mandal et al., 2019) .

The main difference in our setting is that not only are the OOD examples unseen, but the in-distribution examples/classes at evaluation time are also unseen, and only defined by a support set.

To the best of our knowledge, this is the first study to investigate both OOS and OOE tasks and report results using commonly-used metrics in the few-shot setting.

We showed that existing confidence scores developed in the supervised setting (i.e., setting with a fixed number of classes) are not suitable when used with popular few-shot classifiers.

Our proposed confidence scores, -MinDist and LCBO, substantially outperformed the baselines on both tasks across four staple few-shot classification datasets.

We hope that our work encourages future studies on quantitative evaluation of out-of-distribution detection and uncertainty in the few-shot setting.

Training.

Algorithm 2 is a description of episodic training of a classifier.

Here, D C denotes the set of all examples of classes in set C .

RANDOMSAMPLE(s, n) randomly selects n elements from the set s.

Algorithm 2 Episodic training.

1: while not converged do 2:

for c in C episode do for each class φ ← φ − α(∇ φ L P N (φ; {S, Q}) 8: end while Evaluation.

Algorithm 3 is a description of episodic evaluation of the OOE task.

Note that now both the in-distribution and out-of-episode classes are drawn from unseen classes.

Both the indistribution and out-of-episodes scores are accumulated over 1000 episodes, and evaluated using a metric such as AUROC.

In the OOS setting, one would modify lines 9 to 12 to sample from the OOS set, e.g., R ← RANDOMSAMPLE(D OOS , N C * N Q ).

Omniglot.

The Omniglot dataset (Lake et al., 2011) contains 28 × 28 greyscale images of handwritten characters.

This is the most widely adopted benchmark dataset for few-shot classification.

We use the same splits as in (Snell et al., 2017) .

Each class has 20 samples, and there are a total of 1200 × 4 training classes and 423 × 4 unseen classes.

CIFAR100.

The CIFAR100 dataset (Krizhevsky, 2009) contains 32 × 32 color images.

It is similar to the CIFAR10 dataset, but has 100 classes of 600 images each.

We used 64 classes for training, 16 for validation, and 20 for test.

miniImageNet.

The miniImageNet dataset is another commonly used few-shot benchmark (Snell et al., 2017; Vinyals et al., 2016) .

It consists of 84 × 84 colored images.

It also has 100 classes, and 600 examples each.

Similarly, we used 64 classes for training, 16 for validation, and 20 for test.

1: S in ← ∅ 2: S out ← ∅ 3: for 1000 do 4:

for c in C episode do 6:

end for 9:

for c in C ooe do 11:

end for

for x in in Q do 14:

end for

for x out in R do 17:

end for 19: end for 20: Metric(S in , S out ) tieredImagenet.

The tieredImageNet dataset is very similar to the miniImageNet dataset.

Proposed by Ren et al. (2018) , it has 608 classes instead of 100.

Out-of-Dataset.

The OOS datasets were adopted from previous studies including those by Hendrycks et al. (2019); Liang et al. (2017) .

Since we experimented with in-distribution datasets of different scales, the OOS inputs were scaled accordingly.

• Noise: We used uniform, Gaussian, and Rademacher noise of the same dimensionality as the in-distribution data (e.g., 3 × 32 × 32 uniform noise as OOS data for CIFAR-100).

• notMNIST consists of 28 × 28 grayscale images of alphabetic characters from several typefaces.

• CIFAR10bw is simply a grayscale version of CIFAR10.

• LSUN is a large-scale scene understanding dataset (Yu et al., 2015) .

• iSUN is a subset of SUN consisting of 8925 images (Xu et al., 2015) .

• Texture is a dataset with different real world patterns (Cimpoi et al., 2014 ).

• Places is another large scale scene understanding dataset (Zhou et al., 2017) .

• SVHN refers to the Google Street View House Numbers dataset (Netzer et al., 2011) .

• TinyImagenet consists of 64 × 64 color images from 200 ImageNet classes, with 600 examples of each class.

All the results in this section are in the 5-way, 5-shot setting, and were obtained using the 4-layer convolutional backbone.

In Tables 5 and 6 , we show that -MinDist improved both OOE and OOS detection results under all metrics.

The improvement on the OOS task was very pronounced due to the fact that baseline scoring functions based on p(y|x in ) behaved erratically for x in far away from the empirical distribution of the in-distribution embedding.

For example, when the embedding network is trained on CIFAR100, an embedded point based on image of Gaussian noise has an L 2 -norm 10× larger than the average embedding of an in-distribution input.

This resulted in a very confident SPP score (see paragraph below).

This effect was eliminated by using -MinDist, and any embedded point far away from the class prototypes was assigned low confidence.

This intuition seemed to apply to most of the OOS tasks.

On the more challenging task of OOE detection, -MinDist improved over the baselines, but not by as large a margin when the in-distribution dataset was easy (e.g., Omniglot).

The improvement on the OOE task was more substantial when the in-distribution dataset was CIFAR100.

Toy example of when 'softmax of distance' breaks down.

Note that when the input to the softmax, or our logits, are the negative distances to each of the prototypes:

the softmax function is invariant to a constant additive bias in the logits.

This makes anything outside of the convex hull formed by the prototypes equivalent to being right on the boundary of the convex hull.

In the case that we have a 1-dimensional embedding, and only 2 prototypes located at 0 and 1, anything within the range of 0 and 1 would give reasonable probability, and the point 0.5 would give maximum entropy.

However, our intuition says that anything that is very far away from both prototypes, say the point of 100, should also have maximum entropy.

Yet, due to the invariant to constant additive bias, anything outside of the range 0 and 1 would have the undesirable behavior that as one moves away from this range, the output of the softmax decreases in entropy while we desire it to increase in entropy.

In higher dimensions, a similar phenomenon happens, hence confidence functions that operate in the predicted probability space are not suitable for the out-of-distribution data.

A good connection between -MinDist and the method in (Lee et al., 2018b) can be made.

However, Lee et al. (2018b) fit a full covariance Gaussian to each of the classes, and use the Mahalanobis distance as score, which requires computing the inverse covariance of the support embeddings.

This approach faces a fundamental difficulty in the few-shot setting: because the number of training examples (i.e., 25 for the 5-way 5-shot setting) Figure 4 : The toy example in PyTorch.

a is our embedded query, and we have a prototype at 0, and another at 1.

When a = .5, SPP is 0.5.

When a = 100, SPP is 1, which is undesirable.

is smaller than the dimension of the embedding space (i.e. 256), the covariance matrix is singular.

Early in our project we found that the most natural adaptation of (Lee et al., 2018b) , which learns a Gaussian with diagonal covariance per class, performed worse than -MinDist (Table 9 ).

The setup of few-shot ABML consists of a prior p(ψ) on the global initialization ψ, and a prior p(φ|ψ) on episode-specific model weights for each episode.

The training objective is to learn a posterior distribution of ψ which maximizes a variational lower bound of the likelihood of the data.

In each episode, with model weights prior p(φ|ψ), the ABML algorithm performs standard Bayes by Backprop (Blundell et al., 2015) on the support set to obtain the variational posterior distribution for φ.

In practice, the initial variational parameter for φ is set to ψ to reduce the total number of parameters, while the performance did not seem to be negatively affected empirically (Ravi & Beatson, 2019) .

Furthermore, based on the assumption that the variance in ψ should be low due to training over a large number of episodes, Ravi & Beatson (2019) simplify the inference of ψ to a point estimate, and update ψ by the usual gradient descent with gradients aggregated over a sequence of episodes, analogous to the MAML setting.

Following the description in (Ravi & Beatson, 2019), we implemented ABML based on the MAML implementation we got from https://github.com/wyharveychen/ CloserLookFewShot.

Since in general it is difficult to measure how properly Bayesian a method is, we also performed the calibration experiment from the original paper, and found a similar trend (see Figure 5 ).

Combined with similar classification accuracy, we believe that we have a somewhat meaningful implementation of ABML.

MAML ABML (1 posterior sample) ABML (10 posterior sample) Figure 5 : Calibration results.

ABML with 10 posterior samples (ECE=0.40%) have better calibration than ABML with 1 posterior sample (ECE=1.16%), and MAML (ECE=3.61%).

ECE is the expected calibration error (Guo et al., 2017) .

First studied by Ren et al. (2018) , there has been a recent surge of interest in semi-supervised few-shot learning.

Each episode has an additional unlabeled set U = {u} Nu i .

Examples from this set act as additional learning signals in each episode, much like the role of the support set.

However, there are two differences: 1) we are not given label information, and 2) it contains 'distractor' classes, i.e., data that do not come from target classes of interest.

In (Ren et al., 2018) , their 'distractor' inputs are exactly what we refer to as OOE inputs here.

It is known, at least in the supervised setting, that when the unlabelled dataset is polluted with out-of-distribution examples, semi-supervised methods can sometimes even degrade the classifier accuracy (Oliver et al., 2018) .

Similarly, in (Ren et al., 2018) , without the more sophisticated methods that implicitly mask out distractors, soft k-Means with the unlabelled dataset barely has an effect.

Here, we propose a simple semi-supervised inference method with Prototypical Networks based on LCBO.

Since naturally, we can think of p i,c σ(s θ (u i , S c )) as the probability of an unknown input u i belonging to class c, we simply perform soft k-Means to obtain our new prototypes usingp i,c as the responsibilities:μ

and classification in this semi-supervised setting is done based on these updated prototypesμ c .

We usep i,c instead of p i,c because p i,c was optimized so that a point on the boundary of being out-of-distribution would have a p i,c of .5, whereas in this soft clustering scheme, we want those points to have 0 weight.

We also investigated the effect of outlier exposure (OE) (Hendrycks et al., 2019) for training the LCBO network.

We denote LCBO trained with OE as LCBO+OE.

Note, however, that this setup differs from that studied by Hendrycks et al. (2019) .

They do not have a learnable confidence score like LCBO.

They simply have a regularization term to encourage the backbone network to output a uniform distribution for OE inputs.

We do not train the backbone with OE as they do, but use the OE inputs as additional out-of-distribution examples to train our LCBO network.

To train LCBO+OE, we modify the second term in Equation 9 to include queries from the auxiliary dataset, D, along with the usual OOE queries R:

The test-time aggregation for LCBO+OE is identical to that described in Section 4.1.

We investigated two auxiliary dataset settings D for LCBO+OE: 1) using the TinyImages dataset as suggested by Hendrycks et al. (2019) ; and 2) using a combination of TinyImages and the three OOS noise distributions we consider (Gaussian, uniform, and Rademacher noise).

<|TLDR|>

@highlight

We quantitatively study out-of-distribution detection in few-shot setting, establish baseline results with ProtoNet, MAML, ABML, and improved upon them.

@highlight

The paper proposes two new confidence scores which are more suitable for out-of-distribution detection of few-shot classification and shows that a distance metric-based approach improves performance.