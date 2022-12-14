In most real-world scenarios, training datasets are highly class-imbalanced, where deep neural networks suffer from generalizing to a balanced testing criterion.

In this paper, we explore a novel yet simple way to alleviate this issue via synthesizing less-frequent classes with adversarial examples of other classes.

Surprisingly, we found this counter-intuitive method can effectively learn generalizable features of minority classes by transferring and leveraging the diversity of the majority information.

Our experimental results on various types of class-imbalanced datasets in image classification and natural language processing show that the proposed method not only improves the generalization of minority classes significantly compared to other re-sampling or re-weighting methods, but also surpasses other methods of state-of-art level for the class-imbalanced classification.

Deep neural networks (DNNs) trained by large-scale datasets have enabled many breakthroughs in machine learning, especially in various classification tasks such as image classification (He et al., 2016a) , object detection (Redmon & Farhadi, 2017) , and speech recognition (Park et al., 2019) .

Here, a practical issue in this large-scale training regime, however, is at the difficulty in data acquisition process across labels, e.g. some labels are more abundant and easier to collect (Mahajan et al., 2018) .

This often leads a dataset to have "long-tailed" label distribution, as frequently found in modern real-world large-scale datasets.

Such class-imbalanced datasets make the standard training of DNN harder to generalize (Wang et al., 2017; Ren et al., 2018; Dong et al., 2018) , particularly if one requires a class-balanced performance metric for a practical reason.

A natural approach in attempt to bypass this class-imbalance problem is to re-balance the training objective artificially in class-wise with respect to their numbers of samples.

Two of such methods are representative: (a) "re-weighting" the given loss function by a factor inversely proportional to the sample frequency in class-wise (Huang et al., 2016; Khan et al., 2017) , and (b) "re-sampling" the given dataset so that the expected sampling distribution during training can be balanced, either by "over-sampling" the minority classes (Japkowicz, 2000; Cui et al., 2018) or "under-sampling" the majority classes (He & Garcia, 2008) .

The methods on this line, however, usually result in harsh over-fitting to minority classes, since in essence, they cannot handle the lack of information on minority data.

Several attempts have been made to alleviate this over-fitting issue: Cui et al. (2019) proposed the concept of "effective number" of samples as alternative weights in the re-weighting method.

In the context of re-sampling, on the other hand, SMOTE (Chawla et al., 2002 ) is a widely-used variant of the over-sampling method that mitigates the over-fitting via data augmentation, but generally this direction has not been much explored recently.

Cao et al. (2019) found that both re-weighting and re-sampling can be much more effective when applied at the later stage of training, in case of neural networks.

Another line of the research attempts to prevent the over-fitting with a new regularization scheme that minority classes are more regularized, where the margin-based approaches generally suit well as a form of data-dependent regularizer (Zhang et al., 2017; Dong et al., 2018; Khan et al., 2019; Cao et al., 2019) .

There have also been works that view the class-imbalance problem in the framework of active learning (Ertekin et al., 2007; Attenberg & Ertekin, 2013) or meta-learning (Wang et al., 2017; Ren et al., 2018; Shu et al., 2019; Liu et al., 2019) .

Contribution.

In this paper, we revisit the over-sampling framework and propose a new way of generating minority samples, coined Adversarial Minority Over-sampling (AMO).

In contrast to other over-sampling methods, e.g. SMOTE (Chawla et al., 2002) that applies data augmentation to minority samples to mitigate the over-fitting issue, we attempt to generate minority samples in a completely different way: AMO does not use the existing minority samples for synthesis, but use adversarial examples (Szegedy et al., 2014; Goodfellow et al., 2015) of non-minority samples made from another, baseline classifier (potentially, over-fitted to minority classes) independently trained using the given imbalanced dataset.

This motivation leads us to a very counter-intuitive method at a first glance: it results in labeling minority class on an adversarial example of a majority class at last.

Our key finding is that, this method actually can be very effective on learning generalizable features in the imbalanced learning: it does not overly use the minority samples, and leverages the richer information of the majority samples simultaneously.

Our minority over-sampling method consists of three components to improve the sampling quality.

First, we propose an optimization objective for generating synthetic samples, so that a majority input can be translated into a synthetic minority sample via optimizing it, while not affecting the performance of the majority class (even the sample is labeled to the minority class).

Second, we design a sample rejection criteria based on the observation that generation from more majority class is more preferable.

Third, based on the proposed rejection criteria, we suggest an optimal distribution for sampling the initial seed points of the generation.

We evaluate our method on various imbalanced classification problems, including synthetically imbalanced CIFAR-10/100 (Krizhevsky, 2009) , and real-world imbalanced datasets including Twitter dataset (Gimpel et al., 2011) and Reuters dataset (Lewis et al., 2004) in natural language processing.

Despite its simplicity, our method of adversarial minority over-sampling significantly improves the balanced test accuracy compared to previous re-sampling or re-weighting methods across all the tested datasets.

These results even surpass the results from state-of-the-art margin-based method (LDAM; Cao et al. 2019) .

We also highlight that our method is fairly orthogonal to the regularization-based methods, by showing that joint training of our method with LDAM could further improve the balanced test accuracy as well.

Despite the great generalization ability of DNNs, they are known to be susceptible to adversarial examples, which makes it difficult to deploy them in real-world safety-critical applications (Szegedy et al., 2014; Goodfellow et al., 2015) .

The broad existence of adversarial examples in DNNs is still a mysterious phenomenon (Gilmer et al., 2019; Galloway et al., 2019; Ilyas et al., 2019) , and we think our results can be of independent interest to shed new insight on understanding their property.

We consider a classification problem with K classes from a dataset

, where x ??? R d and y ??? {1, ?? ?? ?? , K} denote an input and the corresponding class label, respectively.

Let f : R d ??? R K be a classifier designed to output K logits, which we want to train against the classimbalanced dataset D. We denote N := k N k to be the total sample size of D, where N k is that of class k. Without loss of generality, we assume N 1 ??? N 2 ??? ?? ?? ?? ??? N K .

In the class-imbalanced classification, the class-conditional data distributions P k := p(x | y = k) are assumed to be invariant across training and test time, but they have different prior distributions, say p train (y) and p test (y), respectively: p train (y) is highly imbalanced while p test (y) is usually assumed to be the uniform distribution.

The primary goal of the class-imbalanced learning is to train f from D ??? P train that generalizes well under P test compared to the standard training, e.g., empirical risk minimization (ERM) with an appropriate loss function L(f ):

Our method is primarily based on over-sampling technique (Japkowicz, 2000) , a traditional and principled way to balance the class-imbalanced training objective via sampling minority classes more frequently.

In other words, we assume a "virtually balanced" training dataset D bal made from D such that the class k has N 1 ??? N k more samples, and f is trained on D bal instead of D.

A key difficulty in over-sampling is to prevent over-fitting on minority classes, as the objective modified is essentially much biased to a few samples of minority classes.

In contrast to prior work that focuses on applying data augmentation to minority samples to mitigate this issue (Chawla et al., 2002; Liu et al., 2019) , we attempt to synthesize minority samples in a completely different way: our method does not use the minority samples for synthesis, but use adversarial examples of nonminority samples made from another classifier g :

Consider a scenario of training a neural network f on a class-imbalanced dataset D. The proposed Adversarial Minority Over-sampling (AMO) attempts to construct a new balanced dataset D bal for training of f , by adding adversarial examples (Szegedy et al., 2014) of another classifier g. Here, we assume the classifier g is a pre-trained neural network on D so that performs well (at least) on the training imbalanced dataset, e.g., via standard ERM training.

Therefore, g may be over-fitted to minority classes and perform badly under the balanced testing dataset.

On the other hand, f is the target network we aim to train to perform well on the balanced testing criterion.

During the training f , AMO utilizes the classifier g to generate new minority samples, and the resulting samples are added to D to construct D bal on the fly.

To obtain a single synthetic minority point x * of class k, our method solves an optimization problem starting from another training sample x 0 of a (relatively) major class k 0 < k: where L CE denotes the standard cross entropy loss and ?? > 0 is a hyperparameter.

In other words, our method "translates" a seed point x 0 into x * , so that g confidently classifies it as class k. It is not required for f to classifies x * to k as well, but the optimization objective restricts that f to have lower confidence at the original class k 0 .

The generated sample x * is then labeled to class k, and fed into f for training to perform better on D bal .

Here, the regularization term ?? ?? f k0 (x) on logit reduces the risk when x * is labeled to k, whereas it may contain significant features of x 0 in the viewpoint of f .

Intuitively, one can regard the overall process as teaching the minority classifiers of f to learn new features which g considers it significant, i.e., via extension of the decision boundary from the knowledge g. Figure 1 illustrates the basic idea of our method.

One may understand our method better by considering the case when g is an "oracle" (possibly the Bayes optimal) classifier, e.g., (roughly) humans.

Here, solving (2) essentially requires a transition of the original input x 0 of class k 0 with 100% confidence to another class k with respect to g: this would let g "erase and add" the features related to the class k 0 and k, respectively.

Hence, in this case, our process corresponds to collecting more in-distribution minority data, which may be argued as the best way one could do to resolve the class-imbalance problem.

An intriguing point here is, however, that neural network models are very far from this ideal behavior, even for that achieves super-human performance.

Instead, when f and g are neural networks, (2) often finds x * at very close to x 0 , i.e., similar to the phenomenon of adversarial examples (Szegedy et al., 2014; Goodfellow et al., 2015) .

Nevertheless, we found our method still effectively improves the generalization of minority classes even in such cases.

This observation is, in some sense, aligned to a recent claim that adversarial perturbation is not a "bug" in neural networks, but a "generalizable" feature (Ilyas et al., 2019) .

Sample rejection criteria.

An important factor that affects the quality of the synthetic minority samples in our method is the quality of g, especially for g k0 : a better g k0 would more effectively "erase" important features of x 0 during the generation, thereby making the resulting minority samples more reliable.

In practice, however, g is not that perfect so the synthetic samples still contain some discriminative features of the original class k 0 , in which it may even harm the performance of f .

This risk of "unreliable" generation becomes more harsh when N k0 is small, as we assume that g is also trained on the given imbalanced data D.

Algorithm 1 Adversarial Minority Over-sampling (AMO)

x 0 ??? A randomly-chosen sample of class k 0 in D 7:

if L(g; x * , k) > ?? or R = 1 then 10:

end if 12:

end for 14: end for

To alleviate this risk, we consider a simple criteria for rejecting each of the synthetic samples randomly with probability depending on k 0 and k:

where (??) + := max(??, 0), and ?? ??? [0, 1) is a hyperparameter which controls the reliability of g: the smaller ??, the more reliable g. For example, if ?? = 0.999, the synthetic samples are accepted with probability more than 99% if N k0 ??? N k > 4602.

When ?? = 0.9999, on the other hand, it requires N k0 ??? N k > 46049 to achieve the same goal.

This exponential modeling of rejection probability is motivated by the effective number of samples (Cui et al., 2019) , a heuristic recently proposed to model the observation that the impact of adding a single data point exponentially decreases at larger datasets.

When a synthetic sample is rejected, we simply replace it with another minority point over-sampled from the original D to maintain the loss balance.

Optimal seed-point sampling.

Another design choice of our method is how to choose an initial seed point x 0 for each generation in (2).

This is important since it also affects the final quality of the generation, as the choice of x 0 corresponds to the sampling distribution of k 0 .

Based on the rejection policy proposed in (3), we design a sampling distribution for selecting the class of initial point x 0 given target class k, namely Q(k 0 |k), considering two aspects: (a) Q maximizes the acceptance rate under our rejection policy, and at the same time (b) Q chooses diverse classes as much as possible, i.e., the entropy H(Q) is maximized.

In our over-sampling scenario, i.e., the marginal sampling distribution is uniform in class-wise, these objectives lead Q to be equal to the distribution P (k 0 |k) such that each class is sampled proportional to its acceptance rate:

as it maximizes a joint objective of (a) and (b) above, which turns out to be equivalent to the KLdivergence of P and Q when (a) is formulated to E Q [log P ], i.e., the expected value of the logprobability of P :

where D KL (?? ??) denotes the KL-divergence.

Therefore, we use (4) to sample a seed point for each generation, as the sample-wise re-weighting factor with respect to its class and the given target minority class.

Practical implementation via re-sampling.

In practice of training a neural network f , e.g., stochastic gradient descent (SGD) with mini-batch sampling, AMO is implemented using batch-wise resampling: more precisely, in order to simulate the generation of N 1 ??? N k samples for the class k, we first obtain a balanced mini-batch

via standard re-sampling, and randomly select the indices i to perform the generation with probability

The generation is only performed for the selected indices, where each y i acts as the target class k. For a single generation, we select a seed image x 0 inside the given mini-batch following (4): we found sampling seed images per each mini-batch does not degrades the effectiveness of AMO.

Starting from the selected x 0 , we solve the optimization (2) by performing gradient descent for a fixed number of iterations T .

We only accept the result sample x * only if L(g; x * , k) is less than ?? > 0 for stability.

The overall procedure of AMO is summarized in Algorithm 1.

We evaluate our method on various class-imbalanced classification tasks in visual recognition and natural language processing: synthetically-imbalanced CIFAR-10/100 (Krizhevsky, 2009) , Twitter (Gimpel et al., 2011), and Reuters (Lewis et al., 2004) datasets.

Figure 2 illustrates the class-wise sample distributions for each dataset considered in our experiments.

In overall, our results clearly demonstrate that minority synthesis via adversarial examples consistently improves the efficiency of over-sampling, in terms of the significant improvement of the generalization in minority classes compared to other re-sampling baselines, across all the tested datasets.

We also perform an ablation study to verify the effectiveness of our main ideas.

Throughout this section, we divide the classes in a given dataset into "majority" and "minority" classes, so that the majority classes consist of top-k frequent classes with respect to the training sample size where k is the minimum number that k N k exceeds 50% of the total.

We denote the minority classes as the remaining classes.

Here, we use ?? = 0.9999; (f ) deferred re-sampling (DRS; Cui et al. 2019) : re-sampling is deferred until the later stage of the training; (g) focal loss (Focal; Lin et al. 2017 ): the objective is up-weighted for relatively hard examples to focus more on the minority; (h) label-distribution-aware margin (LDAM; Lin et al. 2017 ): the classifier is trained to impose larger margin to minority classes.

Roughly, the considered baselines can be classified into three categories: (i) "re-sampling" based methods -(b, c, f ), (ii) "re-weighting" based methods -(d, e), and (iii) different loss functions -(a, g, h).

Training details.

We train every model via stochastic gradient descent (SGD) with momentum of weight 0.9.

For CIFAR-10/100 datasets, we train ResNet-32 (He et al., 2016b) for 200 epochs with mini-batch size 128, and set a weight decay of 2 ?? 10 ???4 .

We follow the learning rate schedule used by Cui et al. (2019) for fair comparison: the initial learning rate is set to 0.1, and we decay it by a factor of 100 at 160-th and 180-th epoch.

Although it did not affect much to our method, we also adopt the linear warm-up strategy on the learning rate in the first 5 epochs, as some of the baseline methods, e.g. re-weighting, highly depend on this strategy.

For Twitter and Reuters datasets, on the other hand, we train a 2-layer fully connected network for 15 epochs with mini-batch 64, with a weight decay of 5 ?? 10 ???5 .

The initial learning rate is also set to 0.1, but we decay it by a factor of 10 at 10-th epoch.

Details on AMO.

When our method is applied in the experiments, we use another classifier g of the same architecture that is pre-trained on the given (imbalanced) dataset via standard ERM training.

Also, in a similar manner to that of Cao et al. (2019) , we use the deferred scheduling to our method, i.e., we start to apply our method after the standard ERM training of 160 epochs.

We choose hyperparameters in our method from a fixed set of candidates, namely ?? ??? {0.99, 0.999}, ?? ??? {0.1, 0.5} and ?? ??? {0.9, 0.99}, based on its validation accuracy.

CIFAR-10/100 datasets (Krizhevsky, 2009 ) consist of 60,000 images of size 32 ?? 32, 50,000 for training and 10,000 for test.

Although the original datasets are balanced across 10 and 100 classes, respectively, we consider some "long-tailed" variants of CIFAR-10/100 (CIFAR-LT-10/100), in order to evaluate our method on various levels of imbalance.

To simulate the long-tailed distribution frequently appeared in imbalanced datasets, we control the imbalance ratio ?? > 1 and artificially reduce the training sample sizes of each class except the first class, so that: (a) N 1 /N K equals to ??, and (b) N k in between N 1 and N K follows an exponential decay across k. We keep the test dataset unchanged during this process, thereby the evaluation can be done in the balanced setting.

We compare the (balanced) test accuracy of various training methods (including ours) on CIFAR-LT-10 and 100, considering two imbalance ratios ?? ??? {100, 10} for each (See Figure 2 (a) and 2(b) for an illustration of the sample distribution).

For all the tested methods, we also report the test accuracies computed only on major and minor classes, to identify the relative impacts of each method on the major and minor classes, respectively.

Table 1 and 2 summarize the main results.

In overall, the results show that our method consistently improves the test accuracy by a large margin, across all the tested baselines.

For example, in the case when N 1 /N K = 100 on CIFAR-10, our adversarial minority over-sampling method applied on the baseline ERM improves the test accuracy by 14.0% in the relative gain.

This result even surpasses the "LDAM+DRW" baseline (Cao et al., 2019) , which is known to be a state-of-the-art to the best of our knowledge.

Moreover, we point out, in most cases, our method could further improve the overall test accuracy when applied upon the LDAM training scheme (see "LDAM+AMO"): this indicates that the accuracy gain from our method is fairly orthogonal to that of LDAM, i.e., the margin-based approach, which suggests a new promising direction of improving the generalization when a neural network suffers from a problem of small data.

Next, we further verify the effectiveness of AMO on real-world imbalanced dataset, especially focusing on two natural language processing (NLP) tasks: Twitter (Gimpel et al., 2011) and Reuters (Lewis et al., 2004) datasets.

Twitter dataset is for a part-of-speech (POS) tagging task.

There are 14,614 training examples with 23 classes, and the imbalance ratio, i.e., N 1 /N k , naturally made is about 150 (see Figure 2 (c) for the details).

Reuters dataset, on the other hand, is for a text categorization task which is originally composed of 52 classes.

For a reliable evaluation, we discarded the classes that have less than 5 test examples, and obtained a subset of the full dataset of 36 classes with 6436 training samples.

Nevertheless, the distribution of the resulting dataset is still extremely imbalanced, e.g. N 1 /N k = 710 (see Figure 2 (d) for the details).

Unlike CIFAR-10/100, we found that the two datasets have imbalance issue even in the test samples.

Therefore, we report the averaged value of the class-wise accuracy instead of the standard test accuracy.

Table 3 demonstrates the results.

Again, AMO performed best amongst other baseline methods, demonstrating a wider applicability of our algorithm beyond image classification.

Remarkably, the results on Reuters dataset suggest our method can be even more effective under regime of extremely imbalanced datasets, as the Reuters dataset has much larger imbalance ratio than the others.

We conduct an ablation study on the proposed method, investigating the detailed analysis on it.

All the experiments throughout this study are performed with ResNet-32 models, trained on CIFAR-LT-10 with the imbalance ratio 100.

The use of adversarial examples.

The most intriguing component that consists our method would be the use of "adversarial examples", i.e., to label an adversarial example of majority class to a minority class, e.g. as illustrated in Figure 4 .

To understand more on how the adversarial perturbations affect our method, we consider a simple ablation, which we call "AMO-Clean": recall that our algorithm synthesizes a minority sample x * from a seed image x 0 .

Instead of using x * , this ablation uses the "clean" initial point x 0 as the synthesized minority when accepted.

Under the identical training setup, we notice a significant reduction in the overall accuracy of AMO-Clean compared to the original AMO (see Table 4 ).

This observation reveals that the adversarial perturbations ablated are extremely crucial to make our algorithm to work, regardless of how the noise is small.

The effect of ??.

In the optimization objective (2) for the synthesis in AMO, we impose a regularization term ?? ?? f k0 (x) to improve the quality of synthetic samples as they might confuse f if it still contains important features of the original class in a viewpoint of f .

To verify the effect of this term, we consider an ablation that ?? is set to 0, and compare the performance to the original method.

As reported in Table 4 , we found a certain level of degradation in test accuracy at this ablation, which shows the effectiveness of the proposed regularization.

Comparison of t-SNE embeddings.

To further validate the effectiveness of our method, we visualize and compare the penultimate features learned from various training methods (including ours) using t-SNE (Maaten & Hinton, 2008) .

Each embedding is computed from a randomly-chosen subset of training samples in the CIFAR-LT-10 (?? = 100), so that it consists of 50 samples per each class.

Figure 3 illustrates the results, and shows that the embedding from our training method (AMO) is of much separable features compared to other methods: one could successfully distinguish each cluster under the AMO embedding (even though they are from minority classes), while others have some obscure region.

We propose a new over-sampling method for imbalanced classification, called Advserarial Minority Over-sampling (AMO).

The problems we explored in this paper lead us to an essential question that whether an adversarial perturbation could be a good feature.

Our findings suggest that it could be at least to improve imbalanced learning, where the minority classes suffer over-fitting due to insufficient data.

We believe our method could open a new direction of research both in imbalanced learning and adversarial examples.

<|TLDR|>

@highlight

We develop a new method for imbalanced classification using adversarial examples

@highlight

Proposes a new optimization objective that generates synthetic samples by over-sampling the majority classes instead of minority classes, solving the problem of overfitting minority classes.

@highlight

The authors propose to tackle imbalance classification using re-sampling methods, showing that adversarial examples in the minority class would help to train a new model that generalizes better.