An important research direction in machine learning has centered around developing meta-learning algorithms to tackle few-shot learning.

An especially successful algorithm has been Model Agnostic Meta-Learning (MAML), a method that consists of two optimization loops, with the outer loop finding a meta-initialization, from which the inner loop can efficiently learn new tasks.

Despite MAML's popularity, a fundamental open question remains -- is the effectiveness of MAML due to the meta-initialization being primed for rapid learning (large, efficient changes in the representations) or due to feature reuse,  with the meta initialization already containing high quality features?

We investigate this question, via ablation studies and analysis of the latent representations, finding that feature reuse is the dominant factor.

This leads to the ANIL (Almost No Inner Loop) algorithm, a simplification of MAML where we remove the inner loop for all but the (task-specific) head of the underlying neural network.

ANIL matches MAML's performance on benchmark few-shot image classification and RL and offers computational improvements over MAML.

We further study the precise contributions of the head and body of the network, showing that performance on the test tasks is entirely determined by the quality of the learned features, and we can remove even the head of the network (the NIL algorithm).

We conclude with a discussion of the rapid learning vs feature reuse question for meta-learning algorithms more broadly.

A central problem in machine learning is few-shot learning, where new tasks must be learned with a very limited number of labelled datapoints.

A significant body of work has looked at tackling this challenge using meta-learning approaches (16; 37; 32; 6; 30; 28; 24) .

Broadly speaking, these approaches define a family of tasks, some of which are used for training and others solely for evaluation.

A proposed meta-learning algorithm then looks at learning properties that generalize across the different training tasks, and result in fast and efficient learning of the evaluation tasks.

One highly successful meta-learning algorithm has been Model Agnostic Meta-Learning (MAML) (6) .

At a high level, the MAML algorithm is comprised of two optimization loops.

The outer loop (in the spirit of meta-learning) aims to find an effective meta-initialization, from which the inner loop can perform efficient adaptation -optimize parameters to solve new tasks with very few labelled examples.

This algorithm, with deep neural networks as the underlying model, has been highly influential, with significant follow on work, such as first order variants (24) , probabilistic extensions (8) , augmentation with generative modelling (29) , and many others (15; 7; 12; 35) .

Despite the popularity of MAML, and the numerous followups and extensions, there remains a fundamental open question on the basic algorithm.

Does the meta-initialization learned by the outer loop result in rapid learning on unseen test tasks (efficient but significant changes in the representations) or is the success primarily due to feature reuse (with the meta-initialization already providing high quality representations)?

In this paper, we explore this question and its many surprising consequences.

Our main contributions are:

??? We perform layer freezing experiments and latent representational analysis of MAML, finding that feature reuse is the predominant reason for efficient learning.

??? Based on these results, we propose the ANIL (Almost No Inner Loop) algorithm, a significant simplification to MAML that removes the inner loop updates for all but the head (final layer) of a neural network during training and inference.

ANIL performs identically to MAML on standard benchmark few-shot classification and RL tasks and offers computational benefits over MAML.

??? We study the effect of the head of the network, finding that once training is complete, the head can be removed, and the representations can be used without adaptation to perform unseen tasks, which we call the No Inner Loop (NIL) algorithm.

??? We study different training regimes, e.g. multiclass classification, multitask learning, etc, and find that the task specificity of MAML/ANIL at training facilitate the learning of better features.

We also find that multitask training, a popular baseline with no task specificity, performs worse than random features.

??? We discuss rapid learning and feature reuse in the context of other meta-learning approaches.

MAML (6) is a highly popular meta-learning algorithm for few-shot learning, achieving competitive performance on several benchmark few-shot learning problems (16; 37; 32; 30; 28; 24) .

It is part of the family of optimization-based meta-learning algorithms, with other members of this family presenting variations around how to learn the weights of the task-specific classifier.

For example (19; 10; 4; 18; 39) , first learn functions to embed the support set and target examples of a few-shot learning task, before using the test support set to learn task specific weights to use on the embedded target examples. (14) also proceeds similarly, using a Bayesian approach.

Of these optimization-based meta-learning algorithms, MAML has been especially influential, inspiring numerous direct extensions in recent literature (1; 8; 12; 29) .

Most of these extensions critically rely on the core structure of the MAML algorithm, incorporating an outer loop (for meta-training), and an inner loop (for task-specific adaptation), and there is little prior work analyzing why this central part of the MAML algorithm is practically successful.

In this work, we focus on this foundational question, examining how and why MAML leads to effective few-shot learning.

To do this, we utilize analytical tools such as Canonical Correlation Analysis (CCA) (26; 23) and Centered Kernel Alignment (CKA) (17) to study the neural network representations learned with the MAML algorithm, which also demonstrates MAML's ability to learn effective features for few-shot learning.

Insights from this analysis lead to a simplification that almost completely removes the inner optimization loop (the ANIL algorithm) with no reduction in performance.

Other work has looked at having outer/inner loop specific parameters (40) , but does this in a more complex fashion, partitioning parameters within each layer, and for specific layers, contrasting with the simple head/body separation in ANIL.

Our work is complementary to methods extending MAML, and our simplification and insights could be applied to such extensions also.

Our goal is to understand whether the MAML algorithm efficiently solves new tasks due to rapid learning or feature reuse.

In rapid learning, large representational and parameter changes occur during adaptation to each new task as a result of favorable weight conditioning from the meta-initialization.

In feature reuse, the meta-initialization already contains highly useful features that can mostly be reused as is for new tasks, so little task-specific adaptation occurs.

Figure 1 shows a schematic of these two hypotheses.

We start off by overviewing the details of the MAML algorithm, and then we study the rapid learning vs feature reuse questions via layer freezing experiments and analyzing latent representations of models trained with MAML.

The results strongly support feature reuse as the predominant factor behind MAML's success.

In Section 4, we explore the consequences of this, providing a significant simplification of MAML -the ANIL algorithm, and in Section 6, we outline the connections to meta-learning more broadly.

parameter setting that is well-conditioned for fast learning, and inner loop updates result in significant task specialization.

In Feature Reuse, the outer loop leads to parameter values corresponding to reusable features, from which the parameters do not move significantly in the inner loop.

Images from (13; 9; 36; 2; 22; 34).

The MAML algorithm finds an initialization for a neural network so that new tasks can be learnt with very few examples (k examples from each class for k-shot learning) via two optimization loops:

??? Outer Loop: Updates the initialization of the neural network parameters (often called the meta-initialization) to a setting that enables fast adaptation to new tasks.

??? Inner Loop: Performs adaptation: takes the outer loop initialization, and, separately for each task, performs a few gradient updates over the k labelled examples (the support set) provided for adaptation.

More formally, we first define our base model to be neural network with meta-initialization parameters ??; let this be represented by f ?? .

We have have a distribution D over tasks, and draw a batch {T 1 , ..., T B } of B tasks from D. For each task T b , we have a support set of examples S T b , which are used for inner loop updates, and a target set of examples Z T b , which are used for outer loop updates.

Let ??

i signify ?? after i gradient updates for task T b , and let ?? (b) 0 = ??.

In the inner loop, during each update, we compute ??

for m fixed across all tasks, where

) is the loss on the support set of T b after m ??? 1 inner loop updates.

We then define the meta loss as

where

) is the loss on the target set of T b after m inner loop updates, making clear the dependence of f ?? (b) m on ??.

The outer optimization loop then updates ?? as

At test time, we draw unseen tasks {T

} from the task distribution, and evaluate the loss and accuracy on Z T (test) i after inner loop adaptation using S T

).

We now turn our attention to the key question:

Is MAML's efficacy predominantly due to rapid learning or feature reuse?

In investigating this question, there is an important distinction between the head (final layer) of the network and the earlier layers (the body of the network).

In each few-shot We find that freezing even all four convolutional layers of the network (all layers except the network head) hardly affects accuracy.

This strongly supports the feature reuse hypothesis: layers don't have to change rapidly at adaptation time; they already contain good features from the meta-initialization.

learning task, there is a different alignment between the output neurons and classes.

For instance, in task T 1 , the (wlog) five output neurons might correspond, in order, to the classes (dog, cat, frog, cupcake, phone), while for a different task, T 2 , they might correspond, in order, to (airplane, frog, boat, car, pumpkin).

This means that the head must necessarily change for each task to learn the new alignment, and for the rapid learning vs feature reuse question, we are primarily interested in the behavior of the body of the network.

We return to this in more detail in Section 5, where we present an algorithm (NIL) that does not use a head at test time.

To study rapid learning vs feature reuse in the network body, we perform two sets of experiments:

(1) We evaluate few-shot learning performance when freezing parameters after MAML training, without test time inner loop adaptation; (2) We use representational similarity tools to directly analyze how much the network features and representations change through the inner loop.

We use the MiniImageNet dataset, a popular standard benchmark for few-shot learning, and with the standard convolutional architecture in (6) .

Results are averaged over three random seeds.

Full implementation details are in Appendix B.

To study the impact of the inner loop adaptation, we freeze a contiguous subset of layers of the network, during the inner loop at test time (after using the standard MAML algorithm, incorporating both optimization loops, for training).

In particular, the frozen layers are not updated at all to the test time task, and must reuse the features learned by the meta-initialization that the outer loop converges to.

We compare the few-shot learning accuracy when freezing to the accuracy when allowing inner loop adaptation.

Results are shown in Table 1 .

We observe that even when freezing all layers in the network body, performance hardly changes.

This suggests that the meta-initialization has already learned good enough features that can be reused as is, without needing to perform any rapid learning for each test time task.

We next study how much the latent representations (the latent functions) learned by the neural network change during the inner loop adaptation phase.

Following several recent works (26; 31; 23; 21; 27; 11; 3) we measure this by applying Canonical Correlation Analysis (CCA) to the latent representations of the network.

CCA provides a way to the compare representations of two (latent) layers L 1 , L 2 of a neural network, outputting a similarity score between 0 (not similar at all) and 1 (identical).

For full details, see (26; 23) .

In our analysis, we take L 1 to be a layer before the inner loop adaptation steps, and L 2 after the inner loop adaptation steps.

We compute CCA similarity between L 1 , L 2 , averaging the similarity score across different random seeds of the model and different test time tasks.

Full details are in Appendix B.2

The result is shown in Figure 2 , left pane.

Representations in the body of the network (the convolutional layers) are highly similar, with CCA similarity scores of > 0.9, indicating that the inner loop induces little to no functional change.

By contrast, the head of the network, which does change significantly in the inner loop, has a CCA similarity of less than 0.5.

To further validate this, we also compute CKA (Centered Kernel Alignment) (17) (Figure 2 except the head.

We compute CCA/CKA similarity between the representation of a layer before the inner loop adaptation and after adaptation.

We observe that for all layers except the head, the CCA/CKA similarity is almost 1, indicating perfect similarity.

This suggests that these layers do not change much during adaptation, but mostly perform feature reuse.

Note that there is a slight dip in similarity in the higher conv layers (e.g. conv3, conv4); this is likely because the slight representational differences in conv1, conv2 have a compounding effect on the representations of conv3, conv4.

The head of the network must change significantly during adaptation, and this is reflected in the much lower CCA/CKA similarity.

Having observed that the inner loop does not significantly affect the learned representations with a fully trained model, we extend our analysis to see whether the inner loop affects representations and features earlier on in training.

We take MAML models at 10000, 20000, and 30000 iterations into training, perform freezing experiments (as in Section 3.2.1) and representational similarity experiments (as in Section 3.2.2).

Results in Figure 3 show the same patterns from early in training, with CCA similarity between activations pre and post inner loop update on MiniImageNet-5way-5shot being very high for the body (just like Figure 2 ), and similar to Table 1 , test accuracy remaining approximately the same when freezing contiguous subsets of layers, even when freezing all layers of the network body.

This shows that even early on in training, significant feature reuse is taking place, with the inner loop having minimal effect on learned representations and features.

Results for 1shot MiniImageNet are in Appendix B.5, and show very similar trends.

Table 2 : ANIL matches the performance of MAML on few-shot image classification and RL.

On benchmark few-shot classification tasks MAML and ANIL have comparable accuracy, and also comparable average return (the higher the better) on standard RL tasks (6).

In the previous section we saw that for all layers except the head of the neural network, the metainitialization learned by the outer loop of MAML results in very good features that can be reused as is on new tasks.

Inner loop adaptation does not significantly change the representations of these layers, even from early on in training.

This suggests a natural simplification of the MAML algorithm: the ANIL (Almost No Inner Loop) algorithm.

In ANIL, during training and testing, we remove the inner loop updates for the network body, and apply inner loop adaptation only to the head.

The head requires the inner loop to allow it to align to the different classes in each task.

In Section 5.1 we consider another variant, the NIL (No Inner Loop) algorithm, that removes the head entirely at test time, and uses learned features and cosine similarity to perform effective classification, thus avoiding inner loop updates altogether.

For the ANIL algorithm, mathematically, let ?? = (?? 1 , ..., ?? l ) be the (meta-initialization) parameters for the l layers of the network.

Following the notation of Section 3.1, let ??

m be the parameters after m inner gradient updates for task T b .

In ANIL, we have that:

i.e. only the final layer gets the inner loop updates.

As before, we then define the meta-loss, and compute the outer loop gradient update.

The intuition for ANIL arises from Figure 3 , where we observe that inner loop updates have little effect on the network body even early in training, suggesting the possibility of removing them entirely.

Note that this is distinct to the freezing experiments, where we only removed the inner loop at inference time.

Figure 4 presents the difference between MAML and ANIL, and Appendix C.1 considers a simple example of the gradient update in ANIL, showing how the ANIL update differs from MAML.

Computational benefit of ANIL: As ANIL almost has no inner loop, it significantly speeds up both training and inference.

We found an average speedup of 1.7x per training iteration over MAML and an average speedup of 4.1x per inference iteration.

In Appendix C.5 we provide the full results.

Results of ANIL on Standard Benchmarks: We evaluate ANIL on few-shot image classification and RL benchmarks, using the same model architectures as the original MAML authors, for both supervised learning and RL.

Further implementation details are in Appendix C.4.

The results in Table  2 (mean and standard deviation of performance over three random initializations) show that ANIL matches the performance of MAML on both few-shot classification (accuracy) and RL (average return, the higher the better), demonstrating that the inner loop adaptation of the body is unnecessary for learning good features.

MAML and ANIL Models Show Similar Behavior: MAML and ANIL perform equally well on few-shot learning benchmarks, illustrating that removing the inner loop during training does not hinder performance.

To study the behavior of MAML and ANIL models further, we plot learning curves for both algorithms on MiniImageNet-5way-5shot, Figure 5 .

We see that loss and accuracy for both algorithms look very similar throughout training.

We also look at CCA and CKA scores of the representations learned by both algorithms, Table 3 .

We observe that MAML-ANIL representations have the same average similarity scores as MAML-MAML and ANIL-ANIL representations, suggesting both algorithms learn comparable features (removing the inner loop doesn't change the kinds of features learned.)

Further learning curves and representational similarity results are presented in Appendices C.2 and C.3.

So far, we have seen that MAML predominantly relies on feature reuse, with the network body (all layers except the last layer) already containing good features at meta-initialization.

We also observe that such features can be learned even without inner loop adaptation during training (ANIL algorithm).

The head, however, requires inner loop adaptation to enable task specificity.

In this section, we explore the contributions of the network head and body.

We first ask: How important is the head at test time, when good features have already been learned?

Motivating this question is that these features needed no adaptation at inference time, so perhaps they are themselves sufficient to perform classification, with no head.

In Section 5.1, we find that test time performance is entirely determined by the quality of these representations, and we can use similarity of the frozen meta-initialization representations to perform unseen tasks, removing the head entirely.

We call this the NIL (No Inner Loop) algorithm.

Given this result, we next study how useful the head is at training (in ensuring the network body learns good features).

We look at multiple different training regimes (some without the head) for the network body, and evaluate the quality of the representations.

We find that MAML/ANIL result in the best representations, demonstrating the importance of the head during training for feature learning.

Here, we study how important the head (and task specific alignment) are, when good features have already been learned (through training) by the meta-initialization.

At test time, we find that the representations can be used directly, with no adaptation, which leads to the No Inner Loop (NIL) algorithm:

1 Train a few-shot learning model with ANIL/MAML algorithm as standard.

We use ANIL training.

2 At test time, remove the head of the trained model.

For each task, first pass the k labelled examples (support set) through the body of the network, to get their penultimate layer representations.

Then, for a test example, compute cosine similarities between its penultimate layer representation and those of the support set, using these similarities to weight the support set labels, as in (37).

The results for the NIL algorithm, following ANIL training, on few-shot classification benchmarks are given in Table 4 .

Despite having no network head and no task specific adaptation, NIL performs comparably to MAML and ANIL.

This demonstrates that the features learned by the network body when training with MAML/ANIL (and reused at test time) are the critical component in tackling these benchmarks.

The NIL algorithm and results of Section 5.1, lead to the question of how important task alignment and the head are during training to ensure good features.

Here, we study this question by examining the quality of features arising from different training regimes for the body.

We look at (i) MAML and ANIL training; (ii) multiclass classification, where all of the training data and classes (from which training tasks are drawn) are used to perform standard classification; (iii) multitask training, a standard baseline, where no inner loop or task specific head is used, but the network is trained on all the tasks at the same time; (iv) random features, where the network is not trained at all, and features are frozen after random initialization; (v) NIL at training time, where there is no head and cosine distance on the representations is used to get the label.

After training, we apply the NIL algorithm to evaluate test performance, and quality of features learned at training.

The results are shown in Table 5 .

MAML and ANIL training performs best.

Multitask training, which has no task specific head, performs the worst, even worse than random features (adding evidence for the need for task specificity at training to facilitate feature learning.)

Up till now, we have closely examined the MAML algorithm, and have demonstrated empirically that the algorithm's success is primarily due to feature reuse, rather than rapid learning.

We now discuss rapid learning vs feature reuse more broadly in meta-learning.

By combining our results with an analysis of evidence reported in prior work, we find support for many meta-learning algorithms succeeding via feature reuse, identifying a common theme characterizing the operating regime of much of current meta-learning.

MAML falls within the broader class of optimization based meta-learning algorithms, which at inference time, directly optimize model parameters for a new task using the support set.

MAML has inspired many other optimization-based algorithms, which utilize the same two-loop structure (19; 29; 8) .

Our analysis so far has thus yielded insights into the feature reuse vs rapid learning question for this class of algorithms.

Another broad class of meta-learning consists of model based algorithms, which also have notions of rapid learning and feature reuse.

In the model-based setting, the meta-learning model's parameters are not directly optimized for the specific task on the support set.

Instead, the model typically conditions its output on some representation of the task definition.

One way to achieve this conditioning is to jointly encode the entire support set in the model's latent representation (37; 33), enabling it to adapt to the characteristics of each task.

This constitutes rapid learning for model based meta-learning algorithms.

An alternative to joint encoding would be to encode each member of the support set independently, and apply a cosine similarity rule (as in (37) ) to classify an unlabelled example.

This mode of operation is purely feature reuse -we do not use information defining the task to directly influence the decision function.

If joint encoding gave significant test-time improvement over non-joint encoding, then this would suggest that rapid learning of the test-time task is taking place, as task specific information is being utilized to influence the model's decision function.

However, on analyzing results in prior literature, this improvement appears to be minimal.

Indeed, in e.g. Matching Networks (37), using joint encoding one reaches 44.2% accuracy on MiniImageNet-5way-1shot, whereas with independent encoding one obtains 41.2%: a small difference.

More refined models suggest the gap is even smaller.

For instance, in (5), many methods for one shot learning were re-implemented and studied, and baselines without joint encoding achieved 48.24% accuracy in MiniImageNet-5way-1shot, whilst other models using joint encoding such as Relation Net (33) achieves very similar accuracy of 49.31%

(they also report MAML, at 46.47%).

As a result, we believe that the dominant mode of "feature reuse" rather than "rapid learning" is what has currently dominated both MAML-styled optimization based meta-learning and model based meta-learning.

In this paper, we studied a fundamental question on whether the highly successful MAML algorithm relies on rapid learning or feature reuse.

Through a series of experiments, we found that feature reuse is the dominant component in MAML's efficacy.

This insight led to the ANIL (Almost No Inner Loop) algorithm, a simplification of MAML that has identical performance on standard image classification and reinforcement learning benchmarks, and provides computational benefits.

We further study the importance of the head (final layer) of a neural network trained with MAML, discovering that the body (lower layers) of a network is sufficient for few-shot classification at test time, allowing us to remove the network head for testing (NIL) and still match performance.

We connected our results to the broader literature in meta-learning, identifying feature reuse to be a common mode of operation for other meta-learning algorithms also.

Based off of our conclusions, future work could look at developing and analyzing new meta-learning algorithms that perform more rapid learning, which may expand the datasets and problems amenable to these techniques.

We consider the few-shot learning paradigm for image classification to evaluate MAML and ANIL.

We evaluate using two datasets often used for few-shot multiclass classification -the Omniglot dataset and the MiniImageNet dataset.

Omniglot: The Omniglot dataset consists of over 1600 different handwritten character classes from 23 alphabets.

The dataset is split on a character-level, so that certain characters are in the training set, and others in the validation set.

We consider the 20-way 1-shot and 20-way 5-shot tasks on this dataset, where at test time, we wish our classifier to discriminate between 20 randomly chosen character classes from the held-out set, given only 1 or 5 labelled example(s) from each class from this set of 20 testing classes respectively.

The model architecture used is identical to that in the original MAML paper, namely: 4 modules with a 3 x 3 convolutions and 64 filters with a stride of 2, followed by batch normalization, and a ReLU nonlinearity.

The Omniglot images are downsampled to 28 x 28, so the dimensionality of the last hidden layer is 64.

The last layer is fed into a 20-way softmax.

Our models are trained using a batch size of 16, 5 inner loop updates, and an inner learning rate of 0.1.

The MiniImagenet dataset was proposed by (28) , and consists of 64 training classes, 12 validation classes, and 24 test classes.

We consider the 5-way 1-shot and 5-way 5-shot tasks on this dataset, where the test-time task is to classify among 5 different randomly chosen validation classes, given only 1 and 5 labelled examples respectively.

The model architecture is again identical to that in the original paper: 4 modules with a 3 x 3 convolutions and 32 filters, followed by batch normalization, ReLU nonlinearity, and 2 x 2 max pooling.

Our models are trained using a batch size of 4.

5 inner loop update steps, and an inner learning rate of 0.01 are used.

10 inner gradient steps are used for evaluation at test time.

In this section, we provide further experimental details and results from freezing and representational similarity experiments.

We concentrate on MiniImageNet for our freezing and representational similarity experiments in Section 3.2, as it is more complex than Omniglot.

The model architecture used for our experiments is identical to that in the original paper: 4 modules with a 3 x 3 convolutions and 32 filters, followed by batch normalization, ReLU nonlinearity, and 2 x 2 max pooling.

Our models are trained using a batch size of 4, 5 inner loop update steps, and an inner learning rate of 0.01.

10 inner gradient steps are used for evaluation at test time.

We train models 3 times with different random seeds.

Models were trained for 30000 iterations.

CCA takes in as inputs L 1 = {z

2 , ..., z

2 , ..., z

n }, where L 1 , L 2 are layers, and z We apply this to compare corresponding layers of two networks, net1 and net2, where net1 and net2 might differ due to training step, training method (ANIL vs MAML) or the random seed.

When comparing convolutional layers, as described in (25), we perform the comparison over channels, flattening out over all of the spatial dimensions, and then taking the mean CCA coefficient.

We average over three random repeats.

In addition to assessing representational similarity with CCA/CKA, we also consider the simpler measure of Euclidean distance, capturing how much weights of the network change during the inner loop update (task-specific finetuning).

We note that this experiment does not assess functional changes on inner loop updates as well as the CCA experiments do; however, they serve to provide useful intuition.

We plot the per-layer average Euclidean distance between the initialization ?? and the finetuned weights ??

across different layers l, for MiniImageNet in Figure 6 .

We observe that very quickly after the start of training, all layers except for the last layer have small Euclidean distance difference before and after finetuning, suggesting significant feature reuse.

(Note that this is despite the fact that these layers have more parameters than the final layer.)

MiniImageNet-5way-5shot: Weight differences after finetune conv1 conv2 conv3 conv4 w5 Figure 6 : Euclidean distance before and after finetuning for MiniImageNet.

We compute the average (across tasks) Euclidean distance between the weights before and after inner loop adaptation, separately for different layers.

We observe that all layers except for the final layer show very little difference before and after inner loop adaptation, suggesting significant feature reuse.

The experiment in Section 3.2.2 compared representational similarity of L 1 and L 2 at different points in training (before/after inner loop adaptation) but corresponding to the same random seed.

To complete the picture, it is useful to study whether representational similarity across different random seeds is also mostly unaffected by the inner loop adaptation.

This motivates four natural comparisons: assume layer L 1 is from the first seed, and layer L 2 is from the second seed.

Then we can compute the representational similarity between (L 1 pre,

, where pre/post signify whether we take the representation before or after adaptation.

Prior work has shown that neural network representations may vary across different random seeds (26; 23; 20; 38) , organically resulting in CCA similarity scores much less than 1.

So to identify the effect of the inner loop on the representation, we plot the CCA similarities of (i) (

separately across the different random seeds and different layers.

We then compute the line of best fit for each plot.

If the line of best fit fits the data and is close to y = x, this suggests that the inner loop adaptation doesn't affect the features much -the similarity before adaptation is very close to the similarity after adaptation.

The results are shown in Figure 7 .

In all of the plots, we see that the line of best fit is almost exactly y = x (even for the pre/pre vs post/post plot, which could conceivably be more different as both seeds change) and a computation of the coefficient of determination R 2 gives R 2 ??? 1 for all three plots.

Putting this together with Figure 2 , we can conclude that the inner loop adaptation step doesn't Figure 7: Computing CCA similarity pre/post adaptation across different random seeds further demonstrates that the inner loop doesn't change representations significantly.

We compute CCA similarity of L1 from seed 1 and L2 from seed 2, varying whether we take the representation pre (before) adaptation or post (after) adaptation.

To isolate the effect of adaptation from inherent variation in the network representation across seeds, we plot CCA similarity of of the representations before adaptation against representations after adaptation in three different combinations: (i) (L1 pre, L2 pre) against (L1 pre, L1 post), (ii) (L1 pre, L2 pre) against (L1 pre, L1 post) (iii) (L1 pre, L2 pre) against (L1 post, L2 post).

We do this separately across different random seeds and different layers.

Then, we compute a line of best fit, finding that in all three plots, it is almost identical to y = x, demonstrating that the representation does not change significantly pre/post adaptation.

Furthermore a computation of the coefficient of determination R 2 gives R 2 ??? 1, illustrating that the data is well explained by this relation.

In Figure 8 , we perform this comparison with CKA, observing the same high level conclusions.

We consider freezing and representational similarity experiments for MiniImageNet-5way-1shot.

We see that early on in training (from as few as 10k iterations in), the inner loop updates have little effect on the learned representations and features, and that removing the inner loop updates for all layers but the head have little-to-no impact on the validation set accuracy.

affect the representation learned by any layer except the head, and that the learned representations and features are mostly reused as is for the different tasks.

B.5 MINIIMAGENET-5WAY-1SHOT FREEZING AND CCA OVER TRAINING Figure 9 shows that from early on in training, on MiniImageNet-5way-1shot, that the CCA similarity between activations pre and post inner loop update is very high for all layers but the head.

We further see that the validation set accuracy suffers almost no decrease if we remove the inner loop updates and freeze all layers but the head.

This shows that even early on in training, the inner loop appears to have minimal effect on learned representations and features.

This supplements the results seen in Figure 3 on MiniImageNet-5way-5shot.

In this section, we provide more details about the ANIL algorithm, including an example of the ANIL update, implementation details, and further experimental results.

Consider a simple, two layer linear network with a single hidden unit in each layer:??(x; ??) = ?? 2 (?? 1 x).

In this example, ?? 2 is the head.

Consider the 1-shot regression problem, where we have access to examples (x

2 ) for tasks t = 1, . . .

, T .

Note that (x

1 ) is the (example, label) pair in the meta-training set (used for inner loop adaptation -support set), and (x

2 ) is the pair in the meta-validation set (used for the outer loop update -target set).

In the few-shot learning setting, we firstly draw a set of N tasks and labelled examples from our meta-training set: (x .

Assume for simplicity that we only apply one gradient step in the inner loop.

The inner loop updates for each task are thus defined as follows:

where L(??, ??) is the loss function, (e.g. mean squared error) and ??

i refers to a parameter after inner loop update for task t.

The task-adapted parameters for MAML and ANIL are as follows.

Note how only the head parameters change per-task in ANIL:

In the outer loop update, we then perform the following operations using the data from the metavalidation set:

2 ) ????? 1 (5)

Considering the update for ?? 1 in more detail for our simple, two layer, linear network (the case for ?? 2 is analogous), we have the following update for MAML:

2 ) ????? 1

y(x

For ANIL, on the other hand, the update will be:

2 ) ????? 1 (9)

Note the lack of inner loop update for ?? 1 , and how we do not remove second order terms in ANIL (unlike in first-order MAML); second order terms still persist through the derivative of the inner loop update for the head parameters.

We implement ANIL on MiniImageNet and Omniglot, and generate learning curves for both algorithms in Figure 10 .

We find that learning proceeds almost identically for ANIL and MAML, showing that removing the inner loop has little effect on the learning dynamics.

We compute CCA similarities across representations in a MAML seed and an ANIL seed, and then plot these against the same MAML seed representation compared to a different MAML seed (and similarly for ANIL).

We find a strong correlation between these similarities (Figure 11 ), which suggests that MAML and ANIL are learning similar representations, despite their algorithmic differences.

Supervised Learning Implementation: We used the TensorFlow MAML implementation opensourced by the original authors (6) .

We used the same model architectures as in the original MAML paper for our experiments, and train models 3 times with different random seeds.

All models were trained for 30000 iterations, with a batch size of 4, 5 inner loop update steps, and an inner learning rate of 0.01.

10 inner gradient steps were used for evaluation at test time.

Table 6 : ANIL offers significant computational speedup over MAML, during both training and inference. not exactly match those in the original paper; this may be due to large variance in results, depending on the random initialization.

We used the same model architecture as the original paper (two layer MLP with 100 hidden units in each layer), a batch size of 40, 1 inner loop update step with an inner learning rate of 0.1 and 20 trajectories for inner loop adaptation.

We trained three MAML and ANIL models with different random initialization, and quote the mean and standard deviation of the results.

As in the original MAML paper, for RL experiments, we select the best performing model over 500 iterations of training and evaluate this model at test time on a new set of tasks.

C.5 ANIL IS COMPUTATIONALLY SIMPLER THAN MAML Table 6 shows results from a comparison of the computation time for MAML, First Order MAML, and ANIL, during training and inference, with the TensorFlow implementation described previously, on both MiniImageNet domains.

These results are average time for executing forward and backward passes during training (above) and a forward pass during inference (bottom), for a task batch size of 1, and a target set size of 1.

More specifically, we do the following:

??? We train MAML/ANIL networks as standard, and do standard test time adaptation.

??? For multiclass training, we first (pre)train with multiclass classification, then throw away the head and freeze the body.

We initialize a new e.g. 5-class head, and train that (on top of the frozen multiclass pretrained features) with MAML.

At test time we perform standard adaptation.

??? The same process is applied to multitask training.

??? A similar process is applied to random features, except the network is initialized and then frozen.

The results of this, along with the results from Table 5 in the main text is shown in Table 7 .

We observe very little performance difference between using a MAML/ANIL head and a NIL head for each training regime.

Specifically, task performance is purely determined by the quality of the features and representations learned during training, with task-specific alignment at test time being (i) unnecessary (ii) unable to influence the final performance of the model (e.g. multitask training performance is equally with a MAML head as it is with a NIL-head.)

Here we include results on using CCA and CKA on the representations learned by the different training methods.

Specifically, we studied how similar representations of different training methods were to MAML training, finding a direct correlation with performance -training schemes learning representations most similar to MAML also performed the best.

We computed similarity scores by averaging the scores over the first three conv layers in the body of the network.

@highlight

The success of MAML relies on feature reuse from the meta-initialization, which also yields a natural simplification of the algorithm, with the inner loop removed for the network body, as well as other insights on the head and body.

@highlight

The paper finds that feature reuse is the dominant factor in the success of MAML, and propose new algorithms which spend much less computation than MAML.