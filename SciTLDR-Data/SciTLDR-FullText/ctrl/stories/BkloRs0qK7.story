We present a large-scale empirical study of catastrophic forgetting (CF) in modern Deep Neural Network (DNN) models that perform sequential (or: incremental) learning.

A new experimental protocol is proposed that takes into account typical constraints encountered in application scenarios.

As the investigation is empirical, we evaluate CF behavior on the hitherto largest number of visual classification datasets, from each of which we construct a representative number of Sequential Learning Tasks (SLTs) in close alignment to previous works on CF.

Our results clearly indicate that there is no model that avoids CF for all investigated datasets and SLTs under application conditions.

We conclude with a discussion of potential solutions and workarounds to CF, notably for the EWC and IMM models.

This article is in the context of sequential or incremental learning in Deep Neural Networks (DNNs).

Essentially, this means that a DNN is not trained once, on a single task D, but successively on two or more sub-tasks D 1 , . . .

, D n , one after another.

Learning tasks of this type, which we term Sequential Learning Tasks (SLTs) (see FIG0 ), are potentially very common in real-world applications.

They occur wherever DNNs need to update their capabilities on-site and over time: gesture recognition, network traffic analysis, or face and object recognition in mobile robots.

In such scenarios, neural networks have long been known to suffer from a problem termed "catastrophic forgetting"(CF) (e.g., BID7 ) which denotes the abrupt and near-complete loss of knowledge from previous subtasks D 1 , . . .

, D k−1 after only a few training iterations on the current sub-task D k (see FIG0 compared to FIG0 ).

We focus on SLTs from the visual domain with two sub-tasks each, as DNNs show pronounced CF behavior even when only two sub-tasks are involved.

The sequential learning tasks used in this study only have two sub-tasks: D1 and D2.

During training (white background) and re-training (gray background), test accuracy is measured on D1 (blue, ), D2 (green, ) and D1 ∪ D2 (red, ).

The blue curve allows to determine the presence of CF by simple visual inspection: if there is significant degradation w.r.t.

the red curve, then CF has occurred.

DISPLAYFORM0

The field of incremental learning is large, e.g., BID20 and BID8 .

Recent systematic comparisons between different DNN approaches to avoid CF are performed in, e.g., BID23 or .

Principal recent approaches to avoid CF include ensemble methods BID22 BID6 , dual-memory systems BID24 BID11 BID21 BID9 and regularization approaches.

Whereas BID10 suggest Dropout for alleviating CF, the EWC method BID14 proposes to add a term to the energy function that protects weights that are important for the previous sub-task (s) .

Importance is determined by approximating the Fisher information matrix of the DNN.

A related approach is pursued by the Incremental Moment Matching technique (IMM) (see ), where weights from DNNs trained on a current and a past sub-tasks are "merged" using the Fisher information matrix.

Other regularization-oriented approaches are proposed in BID2 ; BID25 and BID13 which focus on enforcing sparsity of neural activities by lateral interactions within a layer.

Number of tested datasets In general, most methods referenced here are evaluated only on a few datasets, usually on MNIST BID16 and various derivations thereof (permutation, rotation, class separation).

Some studies make limited use of CIFAR10, SVHN, the Amazon sentiment analysis problem, and non-visual problems such as data from Q-learning of Atari games.

A largescale evaluation on a huge number of qualitatively different datasets is still missing 1 .

Model selection and prescience Model selection (i.e., selecting DNN topology and hyperparameters) is addressed in some approaches BID10 but on the basis of a "prescient" evaluation where the best model is selected after all tasks have been processed, an approach which is replicated in BID14 .

This amounts to a knowledge of future sub-tasks which is problematic in applications.

Most approaches ignore model selection BID25 BID2 BID13 , and thus implicitly violate causality.

Storage of data from previous sub-tasks From a technical point of view, DNNs can be retrained without storing training data from previous sub-tasks, which is done in BID10 and BID25 .

For regularization approaches, however, there are regularization parameters that control the retention of previous knowledge, and thus must be chosen with care.

In BID14 , this is λ, whereas two such quantities occur in : the "balancing" parameter α and the regularization parameter λ for L2-transfer.

The only study where regularization parameters are obtained through cross-validation (which is avoided in other studies) is BID2 (for λ SN I and λ Ω ) but this requires to store all previous training data.

This review shows that enormous progress has been made, but that there are shortcomings tied to applied scenarios which need to be addressed.

We will formalize this in Sec. 1.2 and propose an evaluation strategy that takes these formal constraints into account when testing CF in DNNs.

When training a DNN model on SLTs, first of all the model must be able to be retrained at any time by new classes (class-incremental learning).

Secondly, it must exhibit retention, or at least graceful decay, of performance on previously trained classes.

Some forgetting is probably unavoidable, but it should be gradual and not immediate, i.e., catastrophic.

However, if a DNN is operating in, e.g., embedded devices or autonomous robots, additional conditions may be applicable: Low memory footprint Data from past sub-tasks cannot be stored and used for re-training, or else to determine when to stop re-training.

Causality Data from future sub-tasks, which are often known in academic studies but not in applications, must not be utilized in any way, especially not for DNN model selection.

This point might seem trivial, but a number of studies such as BID14 ; BID10 and BID25 perform model selection in hindsight, after having processed all sub-tasks.

Constant update complexity Re-training complexity (time and memory) must not depend on the number of previous sub-tasks, thus more or less excluding replay-based schemes such as BID24 .

Clearly, even if update complexity is constant w.r.t.

the number of previous sub-tasks, it should not be too high in absolute terms either.

The original contributions of our work can be summarized as follows:• We propose a training and evaluation paradigm for incremental learning in DNNs that enforces typical application constraints, see Sec. 1.2.

The importance of such an applicationoriented paradigm is underlined by the fact that taking application constraints into account leads to radically different conclusions about CF than those obtained by other recent studies on CF (see Sec. 1.1).•

We investigate the incremental learning capacity of various DNN approaches (Dropout, LWTA, EWC and IMM) using the largest number of qualitatively different classification datasets so far described.

We find that all investigated models are afflicted by catastrophic forgetting, or else in violation of application constraints and discuss potential workarounds.• We establish that the "permuted" type of SLTs (e.g., "permuted MNIST") should be used with caution when testing for CF.• We do not propose a method for avoiding CF in this article.

This is because avoiding CF requires a consensus on how to actually measure this effect: our novel contribution is a proposal how to do just that.

We collect a large number of visual classification datasets, from each of which we construct SLTs according to a common scheme, and compare several recent DNN models using these SLTs.

The experimental protocol is such that application constraints, see Sec. 1.2, are enforced.

For all tested DNN models (see below), we use a TensorFlow (v1.7) implementation under Python (v3.4 and later).

The source code for all processed models, the experiment-generator and evaluation routine can be found on our public available repository 2 .FC A normal, fully-connected (FC) feed-forward DNN with a variable number and size of hidden layers, each followed by ReLU, and a softmax readout layer minimizing cross-entropy.

CONV A convolutional neural network (CNN) based on the work of BID4 .

It is optimized to perform well on image classification problems like MNIST.

We use a fixed topology: two conv-layers with 32 and 64 filters of size 5 × 5 plus ReLU and 2 × 2 max-pooling, followed by a fc-layer with 1024 neurons and softmax readout layer minimizing a cross-entropy energy function.

EWC The Elastic Weight Consolidation (EWC) model presented by BID14 .

LWTA A fully-connected DNN with a variable number and size of hidden layers, each followed by a Local Winner Takes All (LWTA) transfer function as proposed in BID25 .

IMM The Incremental Moment Matching model as presented by .

We examine the weight-transfer techniques in our experiments, using the provided implementation.

D-FC and D-CONV Motivated by BID10 we combine the FC and CONV models with Dropout as an approach to solve the CF problem.

Only FC and CONV are eligible for this, as EWC and IMM include dropout by default, and LWTA is incompatible with Dropout.

We perform model selection in all our experiments by a combinatorial hyper-parameter optimization, whose limits are imposed by the computational resources available for this study.

In particular, we vary the number of hidden layers L ∈ {2, 3} and their size S ∈ {200, 400, 800} (CNNs excluded), the learning rate 1 ∈ {0.01, 0.001} for sub-task D 1 , and the re-training learning rate 2 ∈ {0.001, 0.0001, 0.00001} for sub-task D 2 .

The batch size (batch size ) is fixed to 100 for all experiments, and is used for both training and testing.

As in other studies, we do not use a fixed number of training iterations, but specify the number of training epochs (i.e., passes through the whole dataset) as E = 10 for each processed dataset (see Sec. 2.2), which allows an approximate comparison of different datasets.

The number of training/testing batches per epoch, B, can be calculated from the batch size and the currently used dataset size.

The set of all hyper-parameters for a certain model, denoted P, is formed as a Cartesian product from the allowed values of the hyperparameters L, S, 1 , 2 and complemented by hyper-parameters that remain fixed (E, batch size ) or are particular to a certain model.

For all models that use dropout, the dropout rate for the input layer is fixed to 0.2, and to 0.5 for all hidden layers.

For CNNs, the dropout rate is set to 0.5 for both input and hidden layers.

All other hyper-parameters for CNNs are fixed, e.g., number and size of layers, the max-pooling and filter sizes and the strides (2 × 2) for each channel.

These decisions were made based on the work of BID10 .

The LWTA block size is fixed to 2, based on the work of BID25 .

The model parameter λ for EWC is set to λ1/ 2 (set but not described in the source code of BID14 ).

For all models except IMM, the momentum parameter for the optimizer is set to µ = 0.99 BID26 .

For the IMM models, the SGD optimizer is used, and the regularizer value for the L2-regularization is set to 0.01 for L2-transfer and to 0.0 for weight transfer.

We select the following datasets (see Tab.

1).

In order to construct SLTs uniformly across datasets, we choose the 10 best-represented classes (or random classes if balanced) if more are present.

MNIST FORMULA0 BID16 ) is the common benchmark for computer vision systems and classification problems.

It consist of gray scale images of handwritten digits (0-9).EMNIST BID5 is an extended version of MNIST with additional classes of handwritten letters.

There are different variations of this dataset: we extract the ten best-represented classes from the By Class variation containing 62 classes.

Fruits 360 BID18 ) is a dataset comprising fruit color images from different rotation angles spread over 75 classes, from which we extract the ten best-represented ones.

Devanagari BID1 contains gray scale images of Devanagari handwritten letters.

From the 46 character classes (1.700 images per class) we extract 10 random classes.

FashionMNIST BID27 consists of images of clothes in 10 classes and is structured like the MNIST dataset.

We use this dataset for our investigations because it is a "more challenging classification task than the simple MNIST digits data BID27 ".

SVHN BID19 ) is a 10-class dataset based on photos of house numbers (0-9).

We use the cropped digit format, where the number is centered in the color image.

CIFAR10 BID15 ) contains color images of real-world objects e.g, dogs, airplanes etc.

NotMNIST (Bulatov Yaroslav) contains grayscale images of the 10 letter classes from "A" to "J", taken from different publicly available fonts.

DISPLAYFORM0 is a modified version of the "Arabic Digits dataBase", containing grayscale images of handwritten digits written by 700 different persons.

Tab.

2).

For the latter case, we include SLTs where the second sub-task adds only 1 class (D9-1 type SLTs) or 5 classes (D5-5 type SLTs), since CF may be tied to how much newness is introduced.

We include permutations (DP10-10) since we suspected that this type of SLT is somehow much easier than others, and therefore not a good incremental learning benchmark.

As there are far more ways to create D5-5 type SLTs than D9-1 type SLTs, we create more of the former (8-vs-3) in order to avoid misleading results due to a particular choice of subdivision, whereas we create only a single permutation-type SLT.

Table 2 : Overview of all SLTs.

The assignment of classes to sub-tasks D1 and D2 are disjunct, except for DP10-10 where two different seeded random image permutations are applied.

SLT → D5-5a D5-5b D5-5c D5-5d D5-5e D5-5f D5-5g D5-5h D9-1a D9-1b D9-1c DP10-10 D 1 0-4 0 2 4 6 8 3 4 6 8 9 0 2 5 6 7 0 1 3 4 5 0 3 4 8 9 0 5 6 7 8 0 2 3 6 8 0-8 1-9 0 2-9 0-9 D 2 5-9 1 3 5 7 9 0 1 2 5 7 1 3 4 8 9 2 6 7 8 9 1 2 5 6 7 1 2 3 4 9 1 4 5 7 9 9 0 1 0-9

This study presents just one, albeit very large, experiment, whose experimental protocol implements the constraints from Sec. 1.2.Every DNN model from Sec. 2 is applied to each SLT as defined in Sec. 2.3 while taking into account model selection, see Sec. 2.1.

A precise definition of our application-oriented experimental protocol is given in Alg.

1.

For a given model m and an SLT (D 1 and D 2 ), the first step is to determine the best hyper-parameter vector p * for sub-task D 1 only (see lines 1-4), which determines the model m p * used for re-training.

In a second step, m p * (from line 5) is used for re-training on D 2 , with a different learning rate 2 which is varied separately.

We introduce two criteria for determining the ( 2 -dependent) quality of a re-training phase (lines 6-10): "best", defined by the highest test accuracy on D 1 ∪ D 2 , and "last", defined by the test accuracy on D 1 ∪ D 2 at the end of re-training.

Although the "best" criterion violates the application constraints of Sec. 1.2 (requires D 1 ), we include it for comparison purposes.

Finally, the result is computed as the highest 2 -dependent quality (line 11).

Independently of the second step, another training of m p * is conducted using D 1 ∪ D 2 , resulting in what we term baseline accuracy.

The results of the experiment described in Sec. 3 are summarized in Tab.

3, and in Tab.

4 for IMM.

They lead us to the following principal conclusions:Permutation-based SLTs should be used with caution We find that DP10-10, the SLT based on permutation, does not show CF for any model and dataset, which is exemplary visualized for the FC model in FIG2 which fails completely for all other SLTs.

While we show this only for SLTs with two sub-tasks, and it is unknown how experiments with more sub-tasks would turn out, we nevertheless suggest caution when intepreting results on permutation-based SLTs.

All examined models exhibit CF While this is not surprising for FC and CONV, D-FC as proposed in BID10 performs poorly (see FIG3 ), as does LWTA BID25 .

For EWC and IMM, the story is slightly more complex and will be discussed below.

EWC is mildly effective against CF for simple SLTs.

Our experiments shows that EWC is effective against CF for D9-1 type SLTs, at least when the "best" evaluation criterion is used, which makes use of D 1 .

This, in turn, violates the application requirements of Sec. 1.2.

For the "last" criterion not making use of D 1 , EWC performance, though still significant, is much less impressive.

We can see the origins of this difference illustrated in FIG4 .

Illustrating the difference between the "best" and "last" criterion for EWC.

Shown is the accuracy over time for the best model on SLT D9-1c using EMNIST (a), D9-1a using EMNIST (b) and D9-1b using Devanagari (c).

The blue curve ( ) measures the accuracy on D1, green ( ) only on D2 and red ( ) the D1 ∪ D2 during the training (white) and the re-training phase (gray).

Additionally, the baseline (dashed line) is indicated.

In all three experiments, the "best" strategy results in approximately 90% accuracy, occurring at the beginning of re-training when D2 has not been learned yet.

Here, the magnitude of the best/last difference is a good indicator of CF which clearly happens in (c), partly in (b) and slightly or not at all in (a).EWC is ineffective against CF for more complex problems.

Tab.

3 shows that EWC cannot prevent CF for D5-5 type SLTs, see FIG6 .

Apparently, the EWC mechanism cannot protect all the weights relevant for D 1 here, which is likely to be connected to the fact that the number of samples in both sub-tasks is similar.

This is not the case for D9-1 type tasks where EWC does better and where D 2 has about 10% of the samples in D 1 .

Best EWC experiments for SLT D5-5d constructed from all datasets, to be read as FIG2 .

We observe that CF happens for all datasets.

See also Appendix B for 2D plots.

IMM is effective for all SLTs but unfeasible in practice.

As we can see from Tab.

4, wtIMM clearly outperforms all other models compared in Tab.

3.

Especially for the D5-5 type SLTs, a modest incremental learning quality is attained, which is however quite far away from the baseline accuracy, even for MNIST-derived SLTs.

This is in contrast to the results reported in for MNIST: we attribute this discrepancy to the application-oriented model selection procedure using only D 1 that we perform.

In contrast, in , a model with 800/800/800 neurons, for which good results on MNIST are well-established, is chosen beforehand, thus arguably making implicit use of D 2 .

A significant problem of IMM is the determination of the balancing parameter α, exemplarily illustrated in FIG7 .

Our results show that the optimal value cannot simply be guessed from the relative sizes of D 1 and D 2 , as it is done in , but must be determined by cross-validation, thereby requiring knowledge of D 1 (violates constraints).

Apart from these conceptual issues, we find that the repeated calculation of the Fisher matrices is quite time and memory-consuming (>4h and >8GB), to the point that the treatment of SLTs from certain datasets becomes impossible even on high-end machine/GPU combinations when using complex models.

This is why we can evaluate IMM only for a few datasets.

It is possible that this is an artifact of the TensorFlow implementation, but in the present state IMM nevertheless violates not one but two application constraints from Sec. 1.2.

FIG8 and FIG9 give a visual impression of training an IMM model on D9-1 and D5-5 type SLTs, again illustrating basic feasibility, but also the variability of the "tuning curves" we use to determine the optimal balancing parameter α.

Best wtIMM experiments for SLT D5-5b constructed from datasets we were able to test.

The blue surfaces (epochs 0-10) represent the test accuracy during training on D1, the green surfaces the test accuracy on D2 during training on D2 (epochs 10-20).

The white bars in the middle represent baseline accuracy, whereas the right part shows accuracies on D1 ∪ D2 for different α values, computed for mean-IMM (orange surfaces) and mode-IMM (red surfaces).

See also Appendix B for 2D plots.

The primary conclusion from the results in Sec. 4 is that CF still represents a major problem when training DNNs.

This is particularly true if DNN training happens under application constraints as outlined in Sec. 1.2.

Some of these constraints may be relaxed depending on the concrete application: if some prior knowledge about future sub-task exists, it can be used to simplify model selection and improve results.

If sufficient resources are available, a subset of previously seen data may be kept in memory and thus allow a "best" type evaluation/stopping criterion for re-training, see Alg.

1.Our evaluation approach is similar to , and we adopt some measures for CF proposed there.

A difference is the setting of up to 10 sub-tasks, whereas we consider only two of them since we focus less on the degree but mainly on presence or absence of CF.

Although comparable both in the number of tested models and benchmarks, BID23 uses a different evaluation methodology imposing softer constraints than ours, which is strongly focused on application scenarios.

This is, to our mind, the reason why those results differ significantly from ours and underscores the need for a consensus of how to measure CF.In general application scenarios without prior knowledge or extra resources, however, an essential conclusion we draw from Sec. 4 is that model selection must form an integral part of training a DNN on SLTs.

Thus, a wrong choice of hyper-parameters based on D 1 can be disastrous for the remaining sub-tasks, which is why application scenarios require DNN variants that do not have extreme dependencies on hyper-parameters such as layer number and layer sizes.

Lastly, our findings indicate workarounds that would make EWC or IMM practicable in at least some application scenarios.

If model selection is addressed, a small subset of D 1 may be kept in memory for both methods: to determine optimal values of α for IMM and to determine when to stop re-training for EWC.

FIG7 shows that small changes to α do not dramatically impact final accuracy for IMM, and FIG4 indicates that accuracy loss as a function of re-training time is gradual in most cases for EWC.

The inaccuracies introduced by using only a subset of D 1 would therefore not be very large for both algorithms.

To conclude, this study shows that the consideration of applied scenarios significantly changes the procedures to determine CF behavior, as well as the conclusions as to its presence in latestgeneration DNN models.

We propose and implement such a procedure, and as a consequence claim that CF is still very much of a problem for DNNs.

More research, either on generic solutions, or on workarounds for specific situations, needs to be conducted before the CF problem can be said to be solved.

A minor but important conclusion is that results obtained on permutation-type SLTs should be treated with caution in future studies on CF.

We gratefully acknowledge the support of NVIDIA Corporation with the donation of the Titan Xp GPU used for this research.

BID11 .

This is achieved by dividing the "best" measure from Tab.

3 by the baseline performance.

Each table entry contains two numbers: the baseline performance and Ω all , and cell coloring (indicating presence or absence of CF) is performed based on Ω all .

The overall picture is similar to the one from Tab.

3, as indicated by the cell coloring.

A notable exception is the performance of the CONV and D-CONV models on the SVHN dataset, where Ω all shows an increase, but we do not consider this significant since the already the baseline performance is at chance level here.

That is, this problem is too hard for the simple architectures we use, in which case a small fluctuation due to initial conditions will exceed baseline performance.

We therefore conclude that Ω all is an important measure whenever baseline performance is better than random, in which case is it not meaningful.

On the other hand, our measure works well for random baselines but is less insightful for the opposite case (as the presence of CF is not immediately observable from the raw performances.

A combination of both measures might be interesting to cover both cases.

Here, we present the best results of all algorithms on the MNIST, EMNIST and Devanagari datasets (according to the "best" criterion) for the D9-1b SLT, and the best EWC results on the D5-5d SLT (qualitatively identical to the other D5-5 type SLTs).

Such 2D representations of some experimental results, to be just as FIG4 , may give more clear insights into the details of each experiment.

Here we can observe CF behavior for all algorithms except EWC and IMM for D9-1b.

We can infer that there was no discernible dependency between the occurrence of CF and particular hyperparameter settings (number and size of layers, in particular) since these are already the best experiments for each algorithm and dataset: if these show CF, this means that non of the settings we sampled were able to prevent CF.

EWC shows clear CF for the Devanagari dataset, but might conceivably do better on EMNIST given a little more time for learning D 2 (this will be investigated).

For D5-5d, clear CF occurs even for EWC.

IMM does not exhibit CF for D9-1b (at enormous computations cost, though), and we observe that the value for the balancing parameter cannot simply be set to 0.9 respectively 0.1, as it has its argmax elsewhere.

Mean-IMM;test:All Mode-IMM;test:All

<|TLDR|>

@highlight

We check DNN models for catastrophic forgetting using a new evaluation scheme that reflects typical application conditions, with surprising results.