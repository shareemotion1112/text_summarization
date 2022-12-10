Ensembles, where multiple neural networks are trained individually and their predictions are averaged, have been shown to be widely successful for improving both the accuracy and predictive uncertainty of single neural networks.

However, an ensemble's cost for both training and testing increases linearly with the number of networks.

In this paper, we propose BatchEnsemble, an ensemble method whose computational and memory costs are significantly lower than typical ensembles.

BatchEnsemble achieves this by defining each weight matrix to be the Hadamard product of a shared weight among all ensemble members and a rank-one matrix per member.

Unlike ensembles, BatchEnsemble is not only parallelizable across devices, where one device trains one member, but also parallelizable within a device, where multiple ensemble members are updated simultaneously for a given mini-batch.

Across CIFAR-10, CIFAR-100, WMT14 EN-DE/EN-FR translation, and contextual bandits tasks, BatchEnsemble yields competitive accuracy and uncertainties as typical ensembles; the speedup at test time is 3X and memory reduction is 3X at an ensemble of size 4.

We also apply BatchEnsemble to lifelong learning, where on Split-CIFAR-100, BatchEnsemble yields comparable performance to progressive neural networks while having a much lower computational and memory costs.

We further show that BatchEnsemble can easily scale up to lifelong learning on Split-ImageNet which involves 100 sequential learning tasks.

Ensembling is one of the oldest tricks in machine learning literature (Hansen & Salamon, 1990) .

By combining the outputs of several models, an ensemble can achieve better performance than any of its members.

Many researchers demonstrate that a good ensemble is one where the ensemble's members are both accurate and make independent errors (Perrone & Cooper, 1992; Maclin & Opitz, 1999) .

In neural networks, SGD (Bottou, 2003) and its variants (Kingma & Ba, 2014) are the most common optimization algorithm.

The random noise from sampling mini-batches of data in SGD-like algorithms and random initialization of the deep neural networks, combined with the fact that there is a wide variety of local minima solutions in high dimensional optimization problem (Kawaguchi, 2016; Ge et al., 2015) , results in the following observation: deep neural networks trained with different random seeds can converge to very different local minima although they share similar error rates.

One of the consequence is that neural networks trained with different random seeds will usually not make all the same errors on the test set, i.e. they may disagree on a prediction given the same input even if the model has converged.

Ensembles of neural networks benefit from the above observation to achieve better performance by averaging or majority voting on the output of each ensemble member (Xie et al., 2013; Huang et al., 2017) .

It is shown that ensembles of models perform at least as well as its individual members and diverse ensemble members lead to better performance (Krogh & Vedelsby, 1995) .

More recently, Lakshminarayanan et al. (2017) showed that deep ensembles give reliable predictive uncertainty estimates while remaining simple and scalable.

A further study confirms that deep ensembles generally achieves the best performance on out-of-distribution uncertainty benchmarks (Ovadia et al., 2019) compared to other methods such as MC-dropout (Gal & Ghahramani, 2015) .

Despite their success on benchmarks, ensembles in practice are limited due to their expensive computational and memory costs, which increase linearly with the ensemble size in both training and testing.

Computation-wise, each ensemble member requires a separate neural network forward pass of its inputs.

Memory-wise, each ensemble member requires an independent copy of neural network weights, each up to millions (sometimes billions) of parameters.

This memory requirement also makes many tasks beyond supervised learning prohibitive.

For example, in lifelong learning, a natural idea is to use a separate ensemble member for each task, adaptively growing the total number of parameters by creating a new independent set of weights for each new task.

No previous work achieves competitive performance on lifelong learning via ensemble methods, as memory is a major bottleneck.

Our contribution: In this paper, we aim to address the computational and memory bottleneck by building a more parameter efficient ensemble model: BatchEnsemble.

We achieve this goal by exploiting a novel ensemble weight generation mechanism: the weight of each ensemble member is generated by the Hadamard product between: a. one shared weight among all ensemble members.

b. one rank-one matrix that varies among all members, which we refer to as fast weight in the following sections.

Figure 1 compares testing and memory cost between BatchEnsemble and naive ensemble.

Unlike typical ensembles, BatchEnsemble is mini-batch friendly, where it is not only parallelizable across devices like typical ensembles but also parallelizable within a device.

Moreover, it incurs only minor memory overhead because a large number of weights are shared across ensemble members.

Empirically, we show that BatchEnsemble has the best trade-off among accuracy, running time, and memory on several deep learning architectures and learning tasks: CIFAR-10/100 classification with ResNet32 (He et al., 2016) and WMT14 EN-DE/EN-FR machine translation with Transformer (Vaswani et al., 2017) .

Additionally, we show that BatchEnsemble is also effective in uncertainty evaluation on contextual bandits.

Finally, we show that BatchEnsemble can be successfully applied in lifelong learning and scale up to 100 sequential learning tasks without catastrophic forgetting and the need of memory buffer.

In this section, we describe relevant background about ensembles, uncertainty evaluation, and lifelong learning for our proposed method, BatchEnsemble.

Bagging, also called boostrap aggregating, is an algorithm to improve the total generalization performance by combining several different models (Breiman, 1996) .

The strategy to combine those models such as averaging and majority voting are known as ensemble methods.

It is shown that ensembles of models perform at least as well as each of its ensemble member (Krogh & Vedelsby, 1995) .

Moreover, ensembles achieve the best performance when each of their members makes independent errors (Goodfellow et al., 2015; Hansen & Salamon, 1990) .

Related work on ensembles: Ensembles have been studied extensively for improving model performance (Hansen & Salamon, 1990; Perrone & Cooper, 1992; Dietterich, 2000; Maclin & Opitz, 1999) .

One major direction in ensemble research is how to reduce their cost at test time.

Bucila et al. (2006) developed a method to compress large, complex ensembles into smaller and faster models which achieve faster test time prediction.

developed the above approach further by distilling the knowledge in an ensemble of models into one single neural network.

Another major direction in ensemble research is how to reduce their cost at training time.

Xie et al. (2013) forms ensembles by combining the output of networks within a number of training checkpoints, named Horizontal Voting Vertical Voting and Horizontal Stacked Ensemble.

Additionally, models trained with different regularization and augmentation can be used as ensemble to achieve better performance in semi-supervised learning (Laine & Aila, 2017) .

More recently, Huang et al. (2017) proposed Snapshot ensemble, in which a single model is trained by cyclic learning rates (Loshchilov & Hutter, 2016; Smith, 2015) so that it is encouraged to visit multiple local minima.

Those local minima solutions are then used as ensemble members.

Garipov et al. (2018) proposed fast geometric ensemble where it finds modes that can be connected by simple curves.

Each mode can taken as one ensemble member.

Explicit ensembles are expensive so another line of work lies on what so-called "implicit" ensembles.

For example, Dropout (Srivastava et al., 2014) can be interpreted as creating an exponential number of weight-sharing sub-networks, which are implicitly ensembled in test time prediction (Warde-Farley et al., 2014) .

MC-dropout can be used for uncertainty estimates (Gal & Ghahramani, 2015) .

Although deep neural networks achieve state-of-the-art performance on a variety of benchmarks, their predictions are often poorly calibrated.

Bayesian neural networks (Hinton & Neal, 1995) , which fit a distribution to the weights rather than a point estimate, are often used to model uncertainty.

However, they requires modifications to the traditional neural network training scheme.

Deep ensembles have been proposed as a simple and scalable alternative, and have been shown to make well-calibrated uncertainty estimates (Lakshminarayanan et al., 2017) .

Several metrics had been proposed to measure the quality of uncertainty estimates.

In Section 4.4, we use the contextual bandits benchmark (Riquelme et al., 2018) , where maximizing reward is of direct interest; this requires good uncertainty estimates in order to balance exploration and exploitation.

Appendix D also uses Expected Calibrated Error (ECE) (Guo et al., 2017; Naeini et al., 2015) as an uncertainty metric.

In lifelong learning, the model trains on a number of tasks in a sequential (online) order, without access to entire previous tasks' data (Thrun, 1998; Zhao & Schmidhuber, 1996) .

One core difficulty of lifelong learning is "catastrophic forgetting": neural networks tend to forget what it has learnt after training on the subsequent tasks (McCloskey, 1989; French, 1999) .

Previous work on alleviating catastrophic forgetting can be divided into two categories.

In the first category, updates on the current task are regularized so that the neural network does not forget previous tasks.

Elastic weight consolidation (EWC) applies a penalty on the parameter update based on the distance between the parameters for the new and the old task evaluated by Fisher information metric .

Other methods maintain a memory buffer that stores a number of data points from previous tasks.

For example, gradient episodic memory approach penalizes the gradient on the current task so that it does not increase the loss of examples in the memory buffer (Lopez-Paz & Ranzato, 2017; Chaudhry et al., 2018) .

Another approach focuses on combining existing experience replay algorithms with lifelong learning (Rolnick et al., 2018; Riemer et al., 2018) .

In the second category, one increases model capacity as new tasks are added.

For example, progressive neural networks (PNN) copy the entire network for the previous task and add new hidden units when adopting to a new task .

This prevents forgetting on previous tasks by construction (the network on previous tasks remains the same).

However, it leads to significant memory consumption when faced with a large number of lifelong learning tasks.

Some following methods expand the model in a more parameter efficient way at the cost of introducing an extra learning task and not entirely preventing forgetting.

Yoon et al. (2017) applies group sparsity regularization to efficiently expand model capacity; Xu & Zhu (2018) Figure 2 : An illustration on how to generate the ensemble weights for two ensemble members.

For each training example in the mini-batch, it receives an ensemble weight W i by elementwise multiplying W , which we refer to as "slow weights", with a rank-one matrix F i , which we refer to as "fast weights."

The subscript i represents the selection of ensemble member.

Since W is shared across ensemble members, we term it as "shared weight" in the following paper.

Vectorization: We show how to make the above ensemble weight generation mechanism parallelizable within a device, i.e., where one computes a forward pass with respect to multiple ensemble members in parallel.

This is achieved by the fact that manipulating the matrix computations for a mini-batch.

Let x denote the activations of the incoming neurons in a neural network layer.

The next layer's activations are given by:

where φ denotes the activation function and the subscript n represents the index in the mini-batch.

The output represents next layer's activations from the i th ensemble member.

To vectorize these computations, we define matrices R and S whose rows consist of the vectors r i and s i for all examples in the mini-batch.

The above equation is vectorized as:

where X is the mini-batch input.

By computing Eqn.

5, we can obtain the next layer's activations for each ensemble member in a mini-batch friendly way.

This allows us to take the full advantage of GPU parallelism to implement ensemble efficiently.

To match the input and the ensemble weight, we can divide the input mini-batch into M sub-batches and each sub-batch receives ensemble weight

Ensembling During Testing:

In our experiments, we take the average of predictions of each ensemble member.

Suppose the test batch size is B and there are M ensemble members.

To achieve an efficient implementation, one repeats the input mini-batch M times, which leads to an effective batch size B · M .

This enables all ensemble members to compute the output of the same B input data points in a single forward pass.

It eliminates the need to calculate the output of each ensemble member sequentially and therefore reduces the ensemble's computational cost.

The only extra computation in BatchEnsemble over a single neural network is the Hadamard product, which is cheap compared to matrix multiplication.

Thus, BatchEnsemble incurs almost no additional computational overhead ( Figure 1 ).

1 One limitation of BatchEnsemble is that if we keep the minibatch size the same as single model training, each ensemble member gets only a portion of input data.

In practice, the above issue can be remedied by increasing the batch size so that each ensemble member receives the same amount of data as ordinary single model training.

Since BatchEnsemble is parallelizable within a device, increasing the batch size incurs almost no computational overhead in both training and testing stages on the hardware that can fully utilize large batch size.

Moreover, when increasing the batch size reaches its diminishing return regime, BatchEnsemble can still take advantage from even larger batch size by increasing the ensemble size.

The only memory overhead in BatchEnsemble is the set of vectors, {r 1 , . . .

, r m } and {s 1 , . . . , s m }, which are cheap to store compared to the weight matrices.

By eliminating the need to store full weight matrices of each ensemble member, BatchEnsemble has almost no additional memory cost.

For example, BatchEnsemble of ResNet-32 of size 4 incurs 10% more parameters while naive ensemble incurs 4X more.

The significant memory cost of ensemble methods limits its application to many real world learning scenarios such as multi-task learning and lifelong learning, where one might apply an independent copy of the model for each task.

This is not the case with BatchEnsemble.

Specifically, consider a total of T tasks arriving in sequential order.

Denote D t = (x i , y i , t) as the training data in task t where t ∈ {1, 2, . . . , T } and i is the index of the data point.

Similarly, denote the test data set as T t = (x i , y i , t).

At test time, we compute the average performance on T t across all tasks seen so far as the evaluation metric.

To extend BatchEnsemble to lifelong learning, we compute the neural network prediction in task t with weight W t = W • (r t s t ) in task t. In other words, each ensemble member is in charge of one lifelong learning task.

For the training protocol, we train the shared weight W and two fast weights r 1 , s 1 on the first task, min

where L 1 is the objective function in the first task such as cross-entropy in image classification.

On a subsequent task t, we only train the relevant fast weights r t , s t .

min

BatchEnsemble shares similar advantages as progressive neural networks (PNN): it entirely prevents catastrophic forgetting as the model for previously seen tasks remains the same.

This removes the need of storing any data from previous task.

In addition, BatchEnsemble has significantly less memory consumption than PNN as only fast weights are trained to adapt to a new task.

Therefore, BatchEnsemble can easily scale to up to 100 tasks as we showed in Section 4.1 on split ImageNet.

Another benefit of BatchEnsemble is that if future tasks arrive in parallel rather than sequential order, one can train on all the tasks at once (see Section 3.1).

We are not aware of any other lifelong learning methods can achieve this.

Limitations: BatchEnsemble is one step toward toward a full lifelong learning agent that is both immune to catastrophic forgetting and parameter-efficient.

On existing benchmarks like split-CIFAR and split-ImageNet, Section 4.1 shows that BatchEnsemble's rank-1 perturbation per layer provides enough expressiveness for competitive state-of-the-art accuracies.

However, one limitation of BatchEnsemble is that only rank-1 perturbations are fit to each lifelong learning task and thus the model's expressiveness is a valid concern when each task is significantly varied.

Another limitation is that the shared weight is only trained on the first task.

This implies that only information learnt for the first task can transfer to subsequent tasks.

There is no explicit transfer, for example, between the second and third tasks.

One solution is to enable lateral connections to features extracted by the weights of previously learned tasks, as done in PNN.

However, we found that no lateral connections were needed for Split-CIFAR100 and Split-ImageNet.

Therefore we leave the above solution to future work to further improve BatchEnsemble for lifelong learning.

Section 4.1 firsts demonstrate the BatchEnsemble's effectiveness as an alternative approach to lifelong learning on Split-CIFAR and Split-ImageNet.

We next evaluate BatchEnsemble on several benchmark datasets with common deep learning architectures in Section 4.2 and Section 4.3, including classification task with ResNet (He et al., 2016) , neural machine translation with Transformer (Vaswani et al., 2017) .

Then, we demonstrate that BatchEnsemble can be used for uncertainty modelling in Section 4.4.

Detailed description of datasets we used is in Appendix A. .

PNN: Progressive neural network .

BN-Tuned: Fine tuning Batch Norm layer per subsequent tasks.

BatchE: BatchEnsemble.

Upperbound: Individual ResNet-50 per task.

We showcase BatchEnsemble for lifelong learning on Split-CIFAR100 and Split-ImageNet.

Split-CIFAR100 proposed in Rebuffi et al. (2016) is a harder lifelong learning task than MNIST permutations and MNIST rotations , where one introduces a new set of classes upon the arrival of a new task.

Each task consists of examples from a disjoint set of 100/T classes assuming T tasks in total.

To show that BatchEnsemble is able to scale to 100 sequential tasks, we also build our own Split-ImageNet dataset which shares the same property as Split-CIFAR100 except more classes (and thus more tasks) and higher image resolutions are involved.

More details about these two lifelong learning datasets are provided in Appendix A.

We consider T = 20 tasks on Split-CIFAR100, following the setup of Lopez-Paz & Ranzato (2017) .

We used ResNet-18 with slightly fewer number of filters across all convolutional layers.

Noted that for the purpose of making use of the task descriptor, we build a different final dense layer per task.

We compare BatchEnsemle to progressive neural networks (PNN) , vanilla neural networks, and elastic weight consolidation (EWC) on Split-CIFAR100.

Xu & Zhu (2018) reported similar accuracies among DEN (Yoon et al., 2017) , RCL (Xu & Zhu, 2018) and PNN.

Therefore we compare accuracy only to PNN which has an official implementation and only compare computational and memory costs to DEN and RCL in Appendix C. Figure 3b displays results on Split-CIFAR100 over three metrics including accuracy, forgetting, and cost.

The accuracy measures the average validation accuracy over total 20 tasks after lifelong learning ends.

Average forgetting over all tasks is also presented in Figure 3b .

Forgetting on task t is measured by the difference between accuracy of task t right after training on it and at the end of lifelong learning.

It measures the degree of catastrophic forgetting.

As showed in Figure 3b , BatchEnsemble achieves comparable accuracy as PNN while has 4X speed-up and 50X less memory consumption.

It also preserves the no-forgetting property of PNN.

Therefore BatchEnsemble has the best trade-off among all compared methods.

For Split-ImageNet, we consider T = 100 tasks and apply ResNet-50 followed by a final linear classifier per task.

The parameter overhead of BatchEnsemble on Split-ImageNet over 100 sequential tasks is 20%: the total number of parameters is 30M v.s. 25M (vanilla ResNet-50).

PNN is not capable of learning 100 sequential tasks due to the significant memory consumption; other methods noted above have also not shown results at ImageNet scale.

Therefore we adopt two of our baselines.

The first baseline is "BN-Tuned", which fine-tunes batch normalization parameters per task and which has previously shown strong performance for multi-task learning (Mudrakarta et al., 2018) .

To make a fair comparison, we augment the number of filters in BN-Tuned so that both methods have the same number of parameters.

The second baseline is a naive ensemble which trains an individual ResNet-50 per task.

This provides a rough upper bound on the BatchEnsemble's expressiveness per task.

Note BatchEnsemble and both baselines are immune to catastrophic forgetting.

So we consider validation accuracy on each subsequent task as evaluation metric.

Figure 3a shows that In this section, we evaluate BatchEnsemble on the Transformer (Vaswani et al., 2017) and the large-scale machine translation tasks WMT14 EN-DE/EN-FR.

We apply BatchEnsemble to all self-attention layers with an ensemble size of 4.

The ensemble in a selfattention layer can be interpreted as each ensemble member keeps their own attention mechanism and makes independent decisions.

We conduct our experiments on WMT16 English-German dataset and WMT14 English-French dataset with Transformer base (65M parameters) and Transformer big (213M parameters).

We maintain exactly the same training scheme and hyper-parameters between single Transformer model and BatchEnsemble Transformer model.

As the result shown in Figure 4 , BatchEnsemble achieves a much faster convergence than a single model.

Big BatchEnsemble Transformer is roughly 1.5X faster than single big Transformer on WMT16 English-German.

In addition, the BatchEnsemble Transformer also gives a lower validation perplexity than big Transformer (Table 1 ).

This suggests that BatchEnsemble is promising for even larger Transformers.

We also compared BatchEnsemble to dropout ensemble (MC-drop in Table 1 ).

Transformer single model itself has dropout layer.

We run multiple forward passes with dropout mask during testing.

The sample size is 16 which is already 16X more expensive than BatchEnsmeble.

As Table 1 showed, dropout ensemble doesn't give better performance than single model.

However, note Appendix B shows that while BatchEnemble's test BLEU score increases faster over the course of training, BatchEnsemble which gives lower validation loss does not necessarily improve BLEU score over a single model which is trained for long enough timesteps.

We evaluate BatchEnsemble on classification tasks with CIFAR-10/100 dataset (Krizhevsky, 2009) .

We run our evaluation on ResNet32 (He et al., 2016) .

To achieve 100% training accuracy on CIFAR100, we use 4X more filters than the standard ResNet-32.

In this section, we compare to MC-dropout (Gal & Ghahramani, 2015) which is also a memory efficient ensemble method.

We add one more dense layer followed by dropout before the final linear classifier so that the number of parameters of MC-dropout are the same as BatchEnsemble.

Most hyper-parameters are shared across the single model, BatchEnsemble, and MC-dropout.

More details about hyper-parameters are in Appendix B. Note we increase the training iterations for BatchEnsemble to reach its best performance because each ensemble member gets only a portion of input data.

We train both BatchEnsemble model and MC-dropout with 375 epochs on CIFAR-10/100, which is 50% more iterations than single model.

Although the training duration is longer, BatchEnsemble is still significantly faster than training individual model sequentially.

Another implementation that leads to the same performance is to increase the mini-batch size.

For example, if we use 4X large minibatch size then there is no need to increase the training iterations.

Table 2 shows that BatchEnsemble reaches better accuracy than single model and MC-dropout.

We also calculate the accuracy of naive ensemble, whose members consist of individually trained single models.

Its accuracy can be viewed as the upper bound of Ensemble methods.

We also compare BatchEnsemble to naive ensemble of small models in Appendix F.

In this section, we conduct analysis beyond accuracy, where we show that BatchEnsemble can be used for uncertainty modelling in contextual bandits.

Appendix D evaluates the predictive uncertainty of BatchEnsemble on out-of-distribution tasks and ECE loss.

We also show that BatchEnsemble preserves diversity among ensemble members in predictive distribution just like naive ensemble in Appendix E.

For uncertainty modelling, we evaluate our BatchEnsemble method on the recently proposed bandits benchmark (Riquelme et al., 2018) .

Bandit data comes from different empirical problems that highlight several aspects of decision making.

No single algorithm can outperform every other algorithm on every bandit problem.

Thus, average performance of the algorithm over different problems is used to evaluate the quality of uncertainty estimation.

The key factor to achieve good performance in contextual bandits is to learn a reliable uncertainty model.

In our experiment, Thompson sampling samples from the policy given by one of the ensemble members.

The fact that Dropout which is an implicit ensemble method achieves competitive performance on bandits problem suggests that ensemble can be used as uncertainty modelling.

Indeed, Table 3 shows that BatchEnsemble with an ensemble size 8 achieves the best mean value on the bandits task.

Both BatchEnsemble with ensemble size 4 and 8 outperform Dropout in terms of average performance.

We also evaluate BatchEnsemble on CIFAR-10 corrupted dataset (Hendrycks & Dietterich, 2019) in Appendix D. Figure 7 shows that BatchEnsemble achieves promising accuracy, uncertainty and cost trade-off among all methods we compared.

Moreover, combining BatchEnsemble and dropout ensemble leads to better uncertainty prediction.

We introduced BatchEnsemble, an efficient method for ensembling and lifelong learning.

BatchEnsemble can be used to improve the accuracy and uncertainty of any neural network like typical ensemble methods.

More importantly, BatchEnsemble removes the computation and memory bottleneck of typical ensemble methods, enabling its successful application to not only faster ensembles but also lifelong learning on up to 100 tasks.

We believe BatchEnsemble has great potential to improve in lifelong learning.

Our work may serve as a starting point for a new research area.

CIFAR: We consider two CIFAR datasets, CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) .

Each consists of a training set of size 50K and a test set of size 10K.

They are natural images with 32x32 pixels.

In our experiments, we follow the standard data pre-processing schemes including zero-padding with 4 pixels on each sise, random crop and horizon flip (Romero et al., 2015; Huang et al., 2016; .

In machine translation tasks, we consider the standard training datasets WMT16 EnglishGerman and WMT14 English-French.

WMT16 English-German dataset consists of roughly 4.5M sentence pairs.

We follow the same pre-processing schemes in (Vaswani et al., 2017) .Source and target tokens are processed into 37K shared sub-word units based on byte-pair encoding (BPE) (Britz et al., 2017) .

Newstest2013 and Newstest2014 are used as validation set and test set respectively.

WMT14 English-French consists of a much larger dataset sized at 36M sentences pairs.

We split the tokens into a 32K word-piece vocabulary (Wu et al., 2016) .

The dataset has the same set of images as CIFAR-100 dataset (Krizhevsky, 2009) .

It randomly splits the entire dataset into T tasks so each task consists of 100/T classes of images.

To leverage the task descriptor in the data, different final linear classifier is trained on top of feature extractor per task.

This simplifies the task to be a 100/T class classification problem in each task.

i.e. random prediction has accuracy T /100.

Notice that since we are not under the setting of single epoch training, standard data pre-processing including padding, random crop and random horizontal flip are applied to the training set.

The dataset has the same set of images as ImageNet dataset (Deng et al., 2009) .

It randomly splits the entire dataset into T tasks so each task consists of 1000/T classes of images.

Same as Split-CIFAR100, each task has its own final linear classifier.

Data preprocessing (He et al., 2016 ) is applied to the training data.

In this section, we discuss some implementation details of BatchEnsemble.

Weight Decay:

In the BatchEnsemble, the weight of each ensemble member is never explicitly calculated because we obtain the activations directly by computing Eqn.

5.

To maintain the goal of no additional computational cost, we can instead regularize the mean weight W over ensemble members, which can be efficiently calculated as

where W is the shared weight among ensemble members, S and R are the matrices in Eqn.

5.

We can also only regularize the shared weight and leave the fast weights unregularized because it only accounts for a small portion of model parameters.

In practice, we find the above two schemes work equally.

Diversity Encouragement: Additional loss term such as KL divergence among ensemble members can be added to encourage diversity.

However, we find it sufficient for BatchEnsemble to have desired diversity by initializing the fast weight (s i and r i in Eqn.

1) to be random sign vectors.

Also note that the scheme that each ensemble member is trained with different sub-batch of input can encourage diversity as well.

The diversity analysis is provided in Appendix E.

Machine Translation: The Transformer base is trained for 100K steps and the Transformer big is trained for 180K steps.

The training steps of big model are shorter than Vaswani et al. (2017) because we terminate the training when it reaches the targeted perplexity on validation set.

Experiments are run on 4 NVIDIA P100 GPUs.

The BLEU score of Big Transformer on English-German task is in Figure 5 .

Although BatchEnsemble has lower perplexity as we showed in Section 4.2, we didn't observe a better BLEU score.

Noted that the BLEU score in Figure 5 is lower than what Vaswani et al. (2017) reported.

It is because in order to correctly evaluate model performance at a given timestep, we didn't use the averaging checkpoint trick.

The dropout rate of Transformer base is 0.1 and 0.3 for Transformer big on English-German while remaining 0.1 on English-French.

For dropout ensemble, we ran a grid search between 0.05 and 0.3 in the testing time and report the best validation perplexity.

Classification: We train the model with mini-batch size 128.

We also keep the standard learning rate schedule for ResNet.

The learning rate decreases from 0.1 to 0.01, from 0.01 to 0.001 at halfway of training and 75% of training.

The weight decay coefficient is set to be 10 −4 .

We use an ensemble size of 4, which means each ensemble member receives 32 training examples if we maintain the mini-batch size of 128.

It is because Batch Normalization (Ioffe & Szegedy, 2015) requires at least 32 examples to be effective on CIFAR dataset.

As for the training budget, we train the single model for 250 epochs.

Dynamically expandable networks (Yoon et al., 2017) and Reinforced continual learning (Xu & Zhu, 2018) are two recently proposed lifelong learning methods that achieve competitive performance.

As discussed in Section 4.1, these two methods can be seen as an improved version progressive neural network (PNN) in terms of memory efficiency.

As shown in Xu & Zhu (2018) , all three methods result to similar accuracy measure in Split-CIFAR100 task.

Therefore, among three evaluation metrics (accuracy, forgetting and cost), we only compare the accuracy of BatchEnsemble to PNN in Section 4.1 and compare the cost in this section.

We first compute the cost relative to PNN on Split-CIFAR100 on LeNet and then compute the rest of the numbers base on what were reported in Xu & Zhu (2018) .

Notice that PNN has no much computational overhead on Split-CIFAR100 because the number of total tasks is limited to 10.

Even on the simple setup above, BatchEnsemble gives the best computational and memory efficiency.

The advantage on large lifelong learning task such as Split-ImageNet would be even obvious.

MC-drop BatchE NaiveE Single C10 2.89% 2.37% 2.32% 3.27% C100 8.99% 8.89% 6.82% 9.28% Similar to Lakshminarayanan et al. (2017) , we first evaluate BatchEnsemble on out-of-distribution examples from unseen classes.

It is known that deep neural network tends to make over-confident predictions even if the prediction is wrong or the input comes from unseen classes.

Ensembles of models can give better uncertainty prediction when the test data is out of the distribution of training data.

To measure the uncertainty on the prediction, we calculate the predictive entropy of Single neural network, naive ensemble and BatchEnsemble.

The result is presented in Figure 6a .

As we expected, single model produces over-confident predictions on unseen examples, whereas ensemble methods exhibit higher uncertainty on unseen classes, including both BatchEnsemble and naive ensemble.

It suggests our ensemble weight generation mechanism doesn't degrade uncertainty modelling.

Additionally, we calculate the Expected Calibration Error (Naeini et al., 2015) (ECE) of single model, naive ensemble and BatchEnsemble on both CIFAR-10 and CIFAR-100 in Table 6b .

To calculate ECE, we group model predictions into M interval

where n is the number of samples.

ECE as a criteria of model calibration, measures the difference in expectation between confidence and accuracy (Guo et al., 2017) .

It shows that BatchEnsemble makes more calibrated prediction compared to single neural networks.

Additionally, we evaluate the calibration of different mehtods on recently proposed CIFAR-10 corruption dataset (Hendrycks & Dietterich, 2019) .

The dataset consists of over 30 types of corruptions to the images.

It is commonly used to benchmark a wide range of methods on calibrated prediction (Ovadia et al., 2019) .

To the best of our knowledge, dropout ensemble is the state-of-the-art memory efficient ensemble method.

Thus, in our paper, we compare BatchEnsemble to dropout ensemble in this section.

Naive ensemble is also plotted as an upper bound of our method.

As showed in Figure 7 , BatchEnsemble achieves better calibration than dropout as the skew intensity increases.

Moreover, dropout ensemble requires multiple forward passes to get the best performance.

Ovadia et al. (2019) used sample size 128 while we found no significant difference between sample size 128 and 8.

Note that even the sample size is 8, it is 8X more expensive than BatchEnsemble in the testing time cost.

Finally, we showed that combining BatchEnsemble and dropout ensemble leads to better calibration.

It is competitive to naive ensemble while keeping memory consumption efficient.

It is also an evidence that BatchEnsemble is an orthogonal method to dropout ensemble.

As we discussed in Section 2, ensemble benefits from the diversity among its members.

We focus on the set of test examples on CIFAR-10 where single model makes confident incorrect predictions while ensemble model predicts correctly.

We used the final models we reported in Section 4.3.

In Figure 8 , we randomly select examples from the above set and plot the prediction map of single model, each ensemble member and mean ensemble.

As we can see, although some of the ensemble members make mistakes on thoes examples, the mean prediction takes the advantage of the model averaging and achieves better accuracy on CIFAR-10 classification task.

We notice that BatchEnsemble preserves the diversity among ensemble members as naive ensemble.

In this section, we compare BatchEnsemble to naive ensemble of small models on CIFAR-10/100 dataset.

To maintain the same memory consumption as BatchEnsemble, we trained 4 independent ResNet14x4 models and evaluate the naive ensemble on these 4 models.

This setup of naive ensemble still has roughly 10% memory overhead to BatchEnsemble.

The results are reported in Table 5 .

It shows that naive ensemble of small models achieves lower accuracy than BatchEnsemble.

It illustrates that given the same memory budget, BatchEnsemble is a better choice over naive ensemble.

<|TLDR|>

@highlight

We introduced BatchEnsemble, an efficient method for ensembling and lifelong learning which can be used to improve the accuracy and uncertainty of any neural network like typical ensemble methods.