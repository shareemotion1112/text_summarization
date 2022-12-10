Lifelong machine learning focuses on adapting to novel tasks without forgetting the old tasks, whereas few-shot learning strives to learn a single task given a small amount of data.

These two different research areas are crucial for artificial general intelligence, however, their existing studies have somehow assumed some impractical settings when training the models.

For lifelong learning, the nature (or the quantity) of incoming tasks during inference time is assumed to be known at training time.

As for few-shot learning, it is commonly assumed that a large number of tasks is available during training.

Humans, on the other hand, can perform these learning tasks without regard to the aforementioned assumptions.

Inspired by how the human brain works, we propose a novel model, called the Slow Thinking to Learn (STL), that makes sophisticated (and slightly slower) predictions by iteratively considering interactions between current and previously seen tasks at runtime.

Having conducted experiments, the results empirically demonstrate the effectiveness of STL for more realistic lifelong and few-shot learning settings.

Deep Learning has been successful in various applications.

However, it still has a lot of areas to improve on to reach human's lifelong learning ability.

As one of its drawbacks, neural networks (NNs) need to be trained on large datasets before giving satisfactory performance.

Additionally, they usually suffer from the problem of catastrophic forgetting (McCloskey & Cohen (1989); French (1999) )-a neural network performs poorly on old tasks after learning a novel task.

In contrast, humans are able to incorporate new knowledge even from few examples, and continually throughout much of their lifetime.

To bridge this gap between machine and human abilities, effort has been made to study few-shot learning (Fei-Fei et al. (2006) ; Lake et al. (2011); Santoro et al. (2016) ; Vinyals et al. (2016) ; Snell et al. (2017) ; Ravi & Larochelle (2017b) ; Finn et al. (2017) ; ; Garcia & Bruna (2018) ; Qi et al. (2018) ), lifelong learning (Gepperth & Karaoguz (2016) ; Rusu et al. (2016) ; Kirkpatrick et al. (2017) ; Yoon et al. (2018) ; ; ; SerrÃ et al. (2018) ; Schwarz et al. (2018) ; Sprechmann et al. (2018) ; Riemer et al. (2019) ), and both (Kaiser et al. (2017) ).

The learning tasks performed by humans are, however, more complicated than the settings used by existing lifelong and few-shot learning works.

Task uncertainty: currently, lifelong learning models are usually trained with hyperparameters (e.g., number of model weights) optimized for a sequence of tasks arriving at test time.

The knowledge about future tasks (even their quantity) may be a too strong assumption in many real-world applications, yet without this knowledge, it is hard to decide the appropriate model architecture and capacity when training the models.

Sequential few-shot tasks: existing few-shot learning models are usually (meta-)trained using a large collection of tasks.

1 Unfortunately, this collection is not available in the lifelong learning scenarios where tasks come in sequentially.

Without seeing many tasks at training time, it is hard for an existing few-shot model to learn the shared knowledge behind the tasks and use the knowledge to speed up the learning of a novel task at test time.

Humans, on the other hand, are capable of learning well despite having only limited information and/or even when not purposely preparing for a particular set of future tasks.

Comparing how humans learn and think to how the current machine learning models are trained to learn and make predictions, we observe that the key difference lies on the part of thinking, which is the decision-making counterpart of models when making predictions.

While most NN-based supervised learning models use a single forward pass to predict, humans make careful and less error-prone decisions in a more sophisticated manner.

Studies in biology, psychology, and economics (Parisi et al. (2019) ; Kahneman & Egan (2011) ) have shown that, while humans make fast predictions (like machines) when dealing with daily familiar tasks, they tend to rely on a slow-thinking system that deliberately and iteratively considers interactions between current and previously learned knowledge in order to make correct decisions when facing unfamiliar or uncertain tasks.

We hypothesize that this slow, effortful, and less error-prone decision-making process can help bridge the gap of learning abilities between humans and machines.

We propose a novel brain-inspired model, called the Slow Thinking to Learn (STL), for taskuncertain lifelong and sequential few-shot machine learning tasks.

STL has two specialized but dependent modules, the cross-task Slow Predictor (SP) and per-task Fast Learners (FLs), that output lifelong and few-shot predictions, respectively.

We show that, by making the prediction process of SP more sophisticated (and slightly slower) at runtime, the learning process of all modules can be made easy at training time, eliminating the need to fulfill the aforementioned impractical settings.

Note that the techniques for slow predictions (Finn et al. (2017) ; Ravi & Larochelle (2017b) ; Nichol & Schulman (2018) ; Sprechmann et al. (2018) ) and fast learning (McClelland et al. (1995) ; Kumaran et al. (2016) ; Kaiser et al. (2017) ) have already been proposed in the literature.

Our contributions lie in that we 1) explicitly model and study the interactions between these two techniques, and 2) demonstrate, for the first time, how such interactions can greatly improve machine capability to solve the joint lifelong and few-shot learning problems encountered by humans everyday.

2 Slow Thinking to Learn (STL)

Figure 1: The Slow Thinking to Learn (STL) model.

To model the interactions between the shared SP f and per-task FLs {(g (t) , M (t) )} t , we feed the output of FLs into the SP while simultaneously letting the FLs learn from the feedback given by SP.

We focus on a practical lifelong and fewshot learning set-up:

, · · · arriving in sequence and the labeled examples

also coming in sequence, the goal is to design a model such that it can be properly trained by data

) collected up to any given time point s, and then make correct predictions for unlabeled data X (t) = {x (t,i) } i in any of the seen tasks, t ≤ s.

Note that, at training time s, the future tasks To solve Problem 1, we propose the Slow Thinking to Learn (STL) model, whose architecture is shown in Figure 1 .

The STL is a cascade where the shared Slow Predictor (SP) network f parameterized by θ takes the output of multiple task-specific Fast Learners (FLs) {(g (t) , M (t) )} t , t ≤ s, as input.

An FL for task T (t) consists of an embedding network g (t)2 parameterized by φ (t) and augmented with an external, episodic, non-parametric memory

Here, we use the Memory Module (Kaiser et al. (2017) ) as the external memory which saves the clusters of seen examples {(x (t,i) , y (t,i) )} i to achieve better storage efficiency-the h (t,j) of an entry (h (t,j) , v (t,j) ) denotes the embedding of a cluster of x (t,i) 's with the same label while the v (t,j) denotes the shared label.

We use the FL (g (t) , M (t) ) and SP f to make few-shot and lifelong predictions for task T (t) , respectively.

We let the number of FLs grow with the number of seen tasks in order to ensure that the entire STL model will have enough complexity to learn from possibly endless tasks in lifelong.

This does not imply that the SP will consume unbounded memory space to make predictions at runtime, as the FL for a specific task can be stored on a hard disk and loaded into the main memory only when necessary.

Slow Predictions.

The FL predicts the label of a test instance x using a single feedforward pass just like most existing machine learning models.

As shown in Figure 2 (a), the FL for task T (t) first embed the instance to get h = g (t) (x ) and then predicts the labelŷ FL of x by averaging the cluster labels

where KNN(h ) is the set of K nearest neighboring embeddings of h .

We havê

where h, h denotes the cosine similarity between h (t,j) and h .

On the other hand, the SP predicts the label of x with a slower, iterative process, which is shown in Figure 2 (b).

The SP first adapts (i.e., fine-tunes) its weights θ to KNN(h ) and their corresponding values stored in M (t) to getθ by solving

where loss(·) denotes a loss function.

Then, the SP makes a prediction byŷ SP = f (h ;θ ).

The adapted network fθ is discarded after making the prediction.

The slower decision-making process of SP may seem unnecessary and wasteful of computing resources at first glance.

Next, we explain why it is actually a good bargain.

Life-Long Learning with Task Uncertainty.

Since the SP makes predictions after runtime adaptation, we define the training objective of θ for task T (s) such that it minimizes the losses after being adapted for each seen task

The term loss(f (h;θ * ), v) denotes the empirical slow-prediction loss of the adapted SP on an example (x, y) in M (t) , whereθ * denotes the weights of the adapted SP for x following Eq. (1):

requires recursively solvingθ * for each (x, y) remembered by the FLs.

We use an efficient gradient-based approach proposed by Finn et al. (2017) ) to solve Eq. (2).

Please refer to Section 2.1 of the Appendix for more details.

Since the SP learns from the output of FLs, theθ * in Eq. (2) approximates a hypothesis used by an FL to predict the label of x. The θ, after being trained, will be close to everyθ * and can be fine-tuned to become a hypothesis, meaning that θ encodes the invariant principles 3 underlying the hypotheses for different tasks.

(a) (b) (c) Figure 3 : The relative positions between the invariant representations θ and the approximate hypothesesθ (t) 's of FLs for different tasks T (t) 's on the loss surface defined by FLs after seeing the (a) first, (b) second, and (c) third task.

Since θ−θ (t) ≤ R for any t in Eq. (2), the effective capacity of SP (at runtime) is the union of the capacity of all possible points within the dashed R-circle centered at θ.

Furthermore, after being sequentially trained by two tasks using Eq. (3), the θ will easily get stuck in the middle ofθ

(1) andθ (2) .

To solve the third task, the third FL needs to change its embedding function (and therefore the loss surface) such thatθ (3) falls into the R-circle centered at θ.

Recall that in Problem 1, the nature of tasks arriving after a training process is unknown, thus, it is hard to decide the right model capacity at training time.

A solution to this problem is to use an expandable network (Rusu et al. (2016) ; Yoon et al. (2018) ) and expand the network when training it for a new task, but the number of units to add during each expansion remains unclear.

Our STL walks around this problem by not letting the SP learn the tasks directly but making it learn the invariant principles behind the tasks.

Assuming that the underlying principles of the learned hypotheses for different tasks are universal and relatively simple, 4 one only needs to choose a model architecture with capacity that is enough to learn the shared principles in lifelong manner.

Note that limiting the capacity of SP at training time does not imply underfitting.

As shown in Figure 3 , the postadaptation capacity of SP at runtime can be much larger than the capacity decided during training.

Sequential Few-Shot Learning.

Although each FL is augmented with an external memory that has been shown to improve learning efficiency by the theory of complementary learning systems (McClelland et al. (1995) ; Kumaran et al. (2016) ), it is not sufficient for FLs to perform few-shot predictions.

Normally, these models need to be trained on many existing few-shot tasks in order to obtain good performance at test time.

Without assuming s in Problem 1 to be a large number, the STL takes a different approach that fast stabilizes θ and then let the FL for a new incoming task learn a good hypothesis by extrapolating from θ.

We define the training objective of g (s) , which is parameterized by φ (s) and augmented with memory M (s) , for the current task T (s) as follows:

where

) is the empirical loss term whose specific form depends on the type of external memory used (see Section 2.2 of the Appendix for more details), and

) is a regularization term, which we call the feedback term, whose inverse value denotes the usefulness of the FL in helping SP (f parameterized by θ) adapt.

Specifically, it is written as

The feedback term encourages each FL to learn unique and salient features for the respective task so the SP will not be confused by two tasks having similar embeddings.

As shown in Figure 3 (b), the relative position of θ gets "stuck" easily after seeing a few of previous tasks.

To solve the current task, g (s) needs to change the loss surface for θ such thatθ (s) falls into the R-circle centered at θ (Figure 3(c) ).

This makes θ an efficient guide (through the feedback term) to finding g (s) when there are only few examples and also few previous tasks.

We use an alternate training procedure to train the SP and FLs.

Please see Section 2.3 of the Appendix for more details.

Note that when sequentially training STL for task T (s) in lifelong, we can safely discard the data

in the previous tasks because the FLs are task-specific (see Eq. (3)) and the SP does not require raw examples to train (see Eq. (2)).

In this section, we discuss related works that are not mentioned in Sections 1 and 2.

For a complete discussion, please refer to Section 1 of the Appendix.

Runtime Adaptation.

Our idea of adapting SP at runtime is similar to that of MbPA (Sprechmann et al. (2018) ), which is a method proposed for lifelong learning only.

In MbPA, the embedder and output networks are trained together, in a traditional approach, as one network for the current task.

Its output network adapts to examples stored in an external memory for a previous task before making lifelong predictions.

Nevertheless, there is no discussion of how the runtime adaptation could improve the learning ability of a model, which is the main focus of this paper.

Meta-Learning.

The idea of learning the invariant representations in SP is similar to meta-learning (Finn et al. (2017) ; Ravi & Larochelle (2017b) ; Nichol & Schulman (2018) ), where a model (meta-)learns good initial weights that can speed up the training of the model for a new task using possibly only few shots of data.

To learn the initial weights (which correspond to the invariant representations in our work), existing studies usually assume that the model can sample tasks, including training data, following the task distribution of the ground truth.

However, the Problem 1 studied in this paper does not provide such a luxury.

Memory-Augmented Networks.

An FL is equipped with an external episodic memory module, which is shown to have fast-learning capability (McClelland et al. (1995) ; Kumaran et al. (2016) ) due to its nonparametric nature.

Although we use the Memory Module (Kaiser et al. (2017) ) in this work, our model can integrate with other types of external memory modules, such as Gepperth & Karaoguz (2016) ; Pritzel et al. (2017) ; Santoro et al. (2016) .

This is left as our future work.

FewShot Learning without Forgetting.

Recently, Gidaris & Komodakis (2018) proposed a new few-shot learning approach that does not forget previous tasks when trained on a new one.

However, it still needs to be trained on a large number of existing tasks in order to make few-shot predictions and therefore cannot be applied to Problem 1.

In this section, we evaluate our model in different aspects.

We implement STL and the following baselines using TensorFlow (Abadi et al. (2016) ): Vanilla NN.

A neural network without any technique for preventing catastrophic forgetting or preparation for few-shot tasks.

EWC.

A regularization technique (Kirkpatrick et al. (2017) ) protecting the weights that are important to previous tasks in order to mitigate catastrophic forgetting.

Memory Module.

An external memory module (Kaiser et al. (2017) ) that can make predictions (using KNNs) by itself.

It learns to cluster rows to improve prediction accuracy and space efficiency.

MbPA+.

A memory-augmented model (Sprechmann et al. (2018) trained (FLs first, and then SP), and we use MAML (Finn et al. (2017) ) to solve Eq.

(2) when training the SP.

Separate-MbPA.

This is similar to Separate-MAML, except that the SP is not trained to prepare for run-time adaptation, but it still applies run-time adaptation at test time.

Next, we evaluate the abilities of the models to fight against catastrophic forgetting using the permuted MNIST (LeCun et al. (1998) ) and CIFAR-100 (Krizhevsky & Hinton (2009) ) datasets.

Then, we investigate the impact of task-uncertainty on model performance.

Permuted MNIST.

We create a sequence of 10 tasks, where each task contains MNIST images whose pixels are randomly permuted using the same seed.

The seeds are different across tasks.

We train models for one task at a time, and test their performance on all tasks seen so far.

We first use a setting where all memory-augmented models can save raw examples in their external memory.

This eliminates the need for an embedding network, and, following the settings in Kirkpatrick et al. (2017) , we use a 2-layer MLP with 400 units for all models.

We trained all models using the Adam optimizer for 10,000 iterations per task, with their best-tuned hyperparameters.

Figure 4(a) shows the average performance of models for all tasks seen so far.

The memory-augmented models outperform the Vanilla NN and EWC and do not suffer from forgetting.

This is consistent with previous findings (Sprechmann et al. (2018) ; Kaiser et al. (2017) ).

However, saving raw examples for a potentially infinite number of tasks may be infeasible as it consumes a lot of space.

We therefore use another setting where memory-augmented models save only the embedded examples.

This time, we let both the embedder and the output network (in STL, it is SP) consist of 1-layer MLP with 400 units.

Figure 4(b) shows that the memory-augmented models do not forget even when saving the embeddings.

The only exception is MbPA+, because it uses the same embedder network for all tasks, the embedder network is prone to forgetting.

CIFAR-100.

Here, we design more difficult lifelong learning tasks using the CIFAR-100 dataset.

The CIFAR-100 dataset consists of 100 classes.

Each class belongs to a superclass, and each superclass has 5 classes.

We create a sequence of tasks, called CIFAR-100 Normal, where the class labels in one task belong to different superclasses, but the labels across different tasks are from the same superclass.

This ensures that there is transferable knowledge between tasks in the ground truth.

We also create another sequence of tasks, called CIFAR-100 Hard, where the class labels in one task belong to the same superclasses, while the labels across different tasks are from different superclass.

The tasks in CIFAR-100 Hard share less information, making the lifelong learning more difficult.

For CIFAR-100 tasks, we let the memory-augmented models store embeddings in external memory.

The embedding networks of all models consist of 4 convolutional layers followed by one fully connected layer, and all output networks (SP in STL) are a 2-layer MLP with 400 units.

We search for the best hyperparameters for each model but limit the memory size to 100 embeddings, apply early stopping during training, and use Adam as the optimizer.

As shown in Figures  5(a)(b) , our SP clearly outperforms the baseline models for both the Normal and Hard tasks.

Task Uncertainty and Hyperparameters.

To understand why the SP outperforms other baselines, we study how the performance of each model changes with model capacity.

Figure 4 (c) shows the performance of different models on the permuted MNIST dataset when we deliberately limit the size of external memory to 10 embeddings.

Only the SP performs well in this case.

We also vary the size of external memory used by our FLs and find out that the performance of SP does not drastically change like the other baselines, except when the memory size is extremely small, as shown in Figure 5 (c).

The above results justify that our STL can avoid the customization of memory size (a hyperparameter) to be specifically catered to expected future tasks, whose precise characteristics may not be known at training time.

In addition to memory size, we also conduct experiments with models whose architectures of the output networks (SP in our STL) are changed based on LeNet (LeCun et al. (1998) ).

We consider two model capacities.

The larger model has 4 convolutional layers with 128, 128, 256, and 256 filters followed by 3 fully-connected layers with 256, 256, and 128 units; whereas the small model has 4 convolutional layers with 16, 16, 32, and 32 filters followed by 3 fully-connected layers with 64, 32, 16 units.

Figure 6 compares the performance of different parametric models for the current and previous CIFAR-100 Normal tasks.

We can see that the performance of EWC on current task is heavily affected by model capacity.

EWC with small model size can learn well at first, but struggles to fit the following tasks, which was not a problem when it has larger model size.

MbPA has good performance on current task but forgets the previous tasks no matter how large the model is.

On the other hand, STL is able to perform well on both the previous and current tasks regardless of model size.

This proves the advantage of SP's runtime adaptation ability, that is, it mitigates the need for a model that is carefully sized to the incoming uncertain lifelong tasks.

CIFAR-100.

Existing few-shot learning models are usually trained using a large collection of tasks as training batches.

However, in sequential continual learning settings, collections of these tasks are not available.

Here we designed an experiment setting that simulates an incoming few-shot task during lifelong learning.

We modified the CIFAR Normal and CIFAR Hard sequential tasks, where we trained the models with sequential tasks just like conventional lifelong learning set-up, except that the last task is a "few-shot" task.

5 In this experiment, we assume that the input domains are the same, which means we can use the network parameters (e.g. embedder's weights) learned from previous tasks as initial weights.

We consider three baselines, namely the Memory Module, Separate-MAML, and Vanilla NN.

The Memory Module is the only known model designed for both lifelong and few-shot learning.

We use Separate-MAML to simulate the STL without the feedback term in Eq (3) and the Vanilla NN to indicate "default" fine-tuning performance for each task.

Figure 7 shows the performance on the few-shot task with different number of batches of available training data.

Each batch contains 16 examples, and the memory size for each task is 20 in all memory-augmented models.

We can see that both the FLs and SP in our model outperform other baselines.

The Memory Module cannot learn well without seeing a large collection of tasks at training time, and the Separate-MAML gives unstable performance due to the lack of feedback from the SP (MAML).

Interestingly, these two sophisticated models perform worse than the Vanilla NN sometimes, justifying that the interactions between the fast-leaning and slow-thinking modules are crucial to the joint lifelong and few-shot learning.

Our above observations still hold on the even more challenging dataset, CIFAR Hard.

Please refer to Section 3.3 of the Appendix for more details.

Comparing the results of FLs and SP, we can see that an FL gives better performance when the training data is small.

This justifies that the invariant representations learned by the SP can indeed guide an FL to better learn from the few shots.

Interestingly, the intersection of the predictive ability of an FL and the SP seem to be stable across tasks and usually falls within the range of 48 to 192 examples.

In Section 3.4 of the Appendix, we visualize the embeddings stored in the FLs and the Memory Module to understand how the feedback from SP guide the representation learning of FLs.

Inference Time.

The SP makes "slow" predictions because of runtime adaptation.

Here, we study the time required by the SP to make a single prediction.

We run trained models on a machine with a commodity NVIDIA GTX-1070 GPU.

The number of adaptation steps used is 3 as in previous experiments.

For an FL, SP, and a non-adaptive Vanilla NN trained for the CIFAR-100 Normal tasks, we get 0.24 ms, 2.62 ms, and 0.79 ms per-example inference time on average.

We believe that trading delay of a few milliseconds at runtime for a great improvement on lifelong and few-shot learning abilities is a good bargain in many applications.

Space Efficiency.

The STL also has an advantage in space efficiency.

Please see Section 3 of the Appendix for more details.

Inspired by the thinking process that humans undergo when making decisions, we propose STL, a cascade of per-task FLs and shared SP.

To the best of our knowledge, this is the first work that studies the interactions between the fast-learning and slow-prediction techniques and shows how such interactions can greatly improve machine capability to solve the joint lifelong and few-shot learning problems under challenging settings.

For future works, we will focus on integrating the STL with different types of external memory and studying the performance of STL in real-world deployments.

Memory-Augmented Neural Network (MANN, Santoro et al. (2016) ) bridged the gap of leveraging an external memory for one-shot learning.

MANN updates its external memory by learning a content-based memory writer.

The Memory Module (Kaiser et al. (2017) ) learns a Matching Network (Vinyals et al. (2016) ) but includes an external memory that retains previously seen examples or their representatives (cluster of embeddings).

Unlike MANN, the Memory Module has a deterministic way of updating its memory.

Memory Module has the ability for providing ease and efficiency in grouping of incoming data and selecting class representations.

However, Memory Module encounters limitation in learning and making precise predictions when the given memory space becomes extremely small.

Our proposed STL focuses on the interaction between the per-task memory-augmented Fast Learners (FLs) and the Slow Predictor (SP) to optimize the data usage stored in the memory.

This interaction allows an FL to learn better representations for a better lifelong and few-shot predictions.

It is common for lifelong learning algorithms to store a form of knowledge from previously learned tasks to overcome forgetting.

Some remember the task specific models (Lee et al. (2017) ), while some store raw data, the hessian of the task, or the attention mask of the network for the task (Kirkpatrick et al. (2017) ; Lopez-Paz & Ranzato (2017); SerrÃ et al. (2018) ).

Some approaches such as Yoon et al. (2018) not only attempts to consolidate the model but also expands the network size.

Other works like Hu et al. (2019) ; Schwarz et al. (2018) tried to solve the problem with fixed storage consumption.

Except for Yoon et al. (2018) , the previously mentioned works need to predefine the model capacity, and lacks the flexibility to unknown number of future tasks.

Although Yoon et al. (2018) can expand its capacity when training for a new task, the challenge of deciding how many number of units to add during each expansion still remains.

Some of the recent models (Li & Hoiem (2016); Gepperth & Karaoguz (2016) ; Kirkpatrick et al. (2017) ; He & Jaeger (2018) ; ; Sprechmann et al. (2018) ; Zenke et al. (2017) ) in lifelong learning have taken inspiration on how the brain works (Parisi et al. (2019) ).

Our proposed framework is closely related to other dual-memory systems that are inspired by the complementary learning systems (CLS) theory, which defines the contribution of the hippocampus for quick learning and the neocortex for memory consolidation.

A version of GeppNet (Gepperth & Karaoguz (2016) ) that is augmented with external memory stores some of its training data for rehearsal after each new class is trained.

FearNet ) is composed of three networks for quick recall, memory consolidation, and network selection.

Both GeppNet and FearNet have dedicated sleep phases for memory replay, a mechanism to mitigate catastrophic forgetting.

STL, however, does not require a dedicated sleep or shutdown to consolidate the memory.

This choice is based on considering that there are cases wherein a dedicated sleep time is not feasible, such as when using a machine learning model to provide a frontline service that needs to be up and running all the time and cannot be interrupted by a regular sleep schedule.

In this section, we discuss more technical details about the design and training of STL.

There are different ways to solve Eq. (2).

One can use either the gradient-based MAML (Finn et al. (2017) ) or Reptile (Nichol & Schulman (2018) ) to get an approximated solution efficiently.

The constraint θ − θ ≤ R can be implemented by either adding a Lagrange multiplier in the objective or limiting the number of gradient steps in MAML/Reptile.

In this paper, we use MAML due to its simplicity, ease of implementation, and efficiency, and we enforce the constraint θ − θ ≤ R by limiting the number of adaptation steps of SP at runtime.

An FL in STL is compatible with different types of external memory modules, such as Santoro et al. (2016); Sprechmann et al. (2018) ; Vinyals et al. (2016) .

We choose the Memory Module (Kaiser et al. (2017) ) in this paper due to its clustering capabilities, which increase space efficiency.

For completeness, we briefly discuss how an FL based on the Memory Module are optimized.

) be the sorted K nearest neighbors (from the closest to the farthest) of the embedding of x , and

where · , · denotes the cosine similarity between two vectors, and p and b are the smallest indices such that v (p) = y and v (b) = y, respectively; h (p) and h (b) are the closest positive and negative neighbors.

As this loss is minimized, g (s) maximizes the similarity of embedding of training data points to their positive neighbors, while minimizing the similarity to the negative neighbors by a margin of ε.

The Memory Module Kaiser et al. (2017) also has deterministic update and replacement rules for records in M (s) .

In effect, an h represents the embedding of a cluster of data points, and its value v denotes the shared label of points in that group.

We sequentially train the STL for tasks T

(1) , T (2) , · · · coming in lifelong.

For the current task T (s) , we train the STL using an alternate training approach.

First, the weights φ (s) of g (s) for T (t) is updated by taking some descent steps following Eq. (3) in the main paper.

Next, the θ that parametrizes f is updated following Eq. (2) in the main paper.

One alternate training iteration involves training the FL for the current task for a steps, and then the SP for b steps.

We set the alternate ratio a : b to 1:1 by default.

The pseudo-code of STL's training procedure is shown in Algorithms 1, 2, and 3.

One important hyperparameter to decide before training the STL is R, which affects the number of adaptation steps used by SP.

A larger R allows the adapted weightsθ's to move farther from θ, which may lead the SP to better lifelong predictions but will result in higher computation cost at runtime.

A smaller R helps stabilize θ after the model is trained on previous tasks and enables θ to guide the FLs for new incoming tasks sooner.

We experimented on different values of R by adjusting the number of adaptation steps of SP, and found out that it does not need to be large to achieve good performance.

Normally, it suffices to have less than 5 adaptation steps.

The SP in STL can work with FTs having very small external memory.

Figure 8(a) shows the trade-off between the average all-task performance at the 10-th sequential task on the permuted MNIST dataset and the size (in number of embedded examples) of external memory.

While the performance of most memory-augmented models drops when the memory size is 10, the SP can still perform well.

This justifies that the invariant principles learned by the SP can indeed guide FLs to find better representations that effectively "bring back" the knowledge of SP for a particular task.

Figure 8(b) shows the memory space required by different models in order to achieve at least 0.9 average all-task accuracy.

The STL consumes less than 1% space as compared to MbPA and Memory Module.

The SP also has high adaptation efficiency.

Figure 8 (c) shows the performance gain of different adaptive models after runtime adaptation.

When the memory size is 1000, all models can adapt for the current task to give improved performance.

However, the adaptation efficiency of the baseline models drops when the memory size is 10.

The SP, on the other hand, achieves good performance in this case even after being adapted for just one step thanks to 1) the SP is trained to be ready for adaptation and 2) the invariant principles learned by the SP are useful by themselves and require only few examples (embeddings) to transform to a good hypothesis.

3.3 Sequential Few-shot Learning on CIFAR Hard Figure 9 shows the sequential few-shot learning results of different models on the CIFAR Hard dataset.

In this dataset, the labels of different tasks come from different superclasses in the original CIFAR 100 dataset.

So, it is very challenging for a model to learn from only a few examples in a new task without being able to see a lot of previous tasks.

As we can see, our STL model still outperforms other baselines.

In particular, Figure 9 (a) shows that the STL is able to perform few-shot learning after seeing just one previous task.

Again, the above demonstrates that the interactions between the FLs and SP are a key to improving machine ability for the joint lifelong and few-shot learning.

In order to understand how the feedback from SP guide the representation learning of FLs, we visualize the embeddings stored in the FLs and the Memory Module.

We sample 300 testing examples per task, get their embeddings from the two models, and then project them onto a 2D space using the t-SNE algorithm (Maaten & Hinton (2008)).

The results are shown in Figure 10 .

We can see that the embeddings produced by different FLs for different tasks are more distant from each other than those output by the Memory Module.

Recall that the feedback term in Eq. (3) encourages each FL to learn features that help the SP adapt.

Therefore, each FL learns more salient features for the respective task so the SP will not be confused by two tasks having similar embeddings.

This, in turn, quickly stabilizes SP and makes it an efficient guide (through the feedback term) to learning the FL for a new task when there are only few examples and also few previous tasks, as Figure 3 shows.

@highlight

This paper studies the interactions between the fast-learning and slow-prediction models and demonstrate how such interactions can improve machine capability to solve the joint lifelong and few-shot learning problems.