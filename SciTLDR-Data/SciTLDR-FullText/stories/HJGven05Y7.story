The field of few-shot learning has recently seen substantial advancements.

Most of these advancements came from casting few-shot learning as a meta-learning problem.

Model Agnostic Meta Learning or MAML is currently one of the best approaches for few-shot learning via meta-learning.

MAML is simple, elegant and very powerful, however, it has a variety of issues, such as being very sensitive to neural network architectures, often leading to instability during training, requiring arduous hyperparameter searches to stabilize training and achieve high generalization and being very computationally expensive at both training and inference times.

In this paper, we propose various modifications to MAML that not only stabilize the system, but also substantially improve the generalization performance, convergence speed and computational overhead of MAML, which we call MAML++.

The human capacity to learn new concepts using only a handful of samples is immense.

In stark contrast, modern deep neural networks need, at a minimum, thousands of samples before they begin to learn representations that can generalize well to unseen data-points BID11 BID9 , and mostly fail when the data available is scarce.

The fact that standard deep neural networks fail in the small data regime can provide hints about some of their potential shortcomings.

Solving those shortcomings has the potential to open the door to understanding intelligence and advancing Artificial Intelligence.

Few-shot learning encapsulates a family of methods that can learn new concepts with only a handful of data-points (usually 1-5 samples per concept).

This possibility is attractive for a number of reasons.

First, few-shot learning would reduce the need for data collection and labelling, thus reducing the time and resources needed to build robust machine learning models.

Second, it would potentially reduce training and fine-tuning times for adapting systems to newly acquired data.

Third, in many real-world problems there are only a few samples available per class and the collection of additional data is either remarkably time-consuming and costly or altogether impossible, thus necessitating the need to learn from the available few samples.

The nature of few-shot learning makes it a very hard problem if no prior knowledge exists about the task at hand.

For a model to be able to learn a robust model from a few samples, knowledge transfer (see e.g. BID4 from other similar tasks is key.

However, manual knowledge transfer from one task to another for the purpose of fine-tuning on a new task can be a time consuming and ultimately inefficient process.

Meta-learning BID22 BID24 , or learning to learn BID23 , can instead be used to automatically learn across-task knowledge usually referred to as across-task (or sometimes slow) knowledge such that our model can, at inference time, quickly acquire task-specific (or fast) knowledge from new tasks using only a few samples.

Meta-learning can be broadly defined as a class of machine learning models that become more proficient at learning with more experience, thus learning how to learn.

More specifically meta-learning involves learning at two levels.

At the task-level, where the base-model is required to acquire task-specific (fast) knowledge rapidly, and at the meta-level, where the meta-model is required to slowly learn across-task (slow) knowledge.

Recent work in meta-learning has produced One can see that 2 out of 3 seeds with the original strided MAML seem to become unstable and erratic, whereas all 3 of the strided MAML++ models seem to consistently converge very fast, to much higher generalization accuracy without any stability issues.state of the art results in a variety of settings BID26 BID2 BID27 BID0 BID25 BID14 BID15 BID1 BID3 BID17 .

The application of meta-learning in the fewshot learning setting has enabled the overwhelming majority of the current state of the art few-shot learning methods BID25 BID19 BID6 BID7 BID15 BID17 .

One such method, known for its simplicity and state of the art performance, is Model Agnostic Meta-Learning (MAML) BID7 .

In MAML, the authors propose learning an initialization for a base-model such that after applying a very small number of gradient steps with respect to a training set on the base-model, the adapted model can achieve strong generalization performance on a validation set (the validation set consists of new samples from the same classes as the training set).

Relating back to the definitions of metamodel and base-model, in MAML the meta-model is effectively the initialization parameters.

These parameters are used to initialize the base-model, which is then used for task-specific learning on a support set, which is then evaluated on a target set.

MAML is a simple yet elegant meta-learning framework that has achieved state of the art results in a number of settings.

However, MAML suffers from a variety of problems which: 1) cause instability during training, 2) restrict the model's generalization performance, 3) reduce the framework's flexibility, 4) increase the system's computational overhead and 5) require that the model goes through a costly (in terms of time and computation needed) hyperparameter tuning before it can work robustly on a new task.

In this paper we propose MAML++, an improved variant of the MAML framework that offers the flexibility of MAML along with many improvements, such as robust and stable training, automatic learning of the inner loop hyperparameters, greatly improved computational efficiency both during inference and training and significantly improved generalization performance.

MAML++ is evaluated in the few-shot learning setting where the system is able to set a new state of the art across all established few-shot learning tasks on both Omniglot and Mini-Imagenet, performing as well as or better than all established meta learning methods on both tasks.

The set-to-set few-shot learning setting BID25 , neatly casts few-shot learning as a meta-learning problem.

In set-to-set few-shot learning we have a number of tasks, each task is composed by a support set which is used for task-level learning, and a target set which is used for evaluating the base-model on a certain task after it has acquired task-specific (or fast) knowledge.

Furthermore, all available tasks are split into 3 sets, the meta-training set, the meta-validation set and the meta-test set, used for training, validating and testing our meta-learning model respectively.

Once meta-learning was shown to be an effective framework for few-shot learning and the set to set approach was introduced, further developments in few-shot learning were made in quick succession.

One contribution was Matching Networks BID25 .

Matching networks achieve few-shot learning by learning to match target set items to support set items.

More specifically, a matching network learns to match the target set items to the support set items using cosine distance and a fully differentiable embedding function.

First, the support set embedding function g, parameterized as a deep neural network, embeds the support set items into embedding vectors, then the target set embedding function f embeds the target set items.

Once all data-item embeddings are available, cosine distance is computed for all target set embeddings when compared to all support set embeddings.

As a result, for each target set item, a vector of cosine distances with respect to all support set items will be generated (with each distance's column tied to the respective support set class).

Then, the softmax function is applied on the generated distance vectors, to convert them into probability distributions over the support set classes.

Another notable advancement was the gradient-conditional meta-learner LSTM BID19 that learns how to update a base-learner model.

At inference time, the meta-learner model applies a single update on the base-learner given gradients with respect to the support set.

The fully updated base-model then computes predictions on the target set.

The target set predictions are then used to compute a task loss.

Furthermore they jointly learn the meta-learner's parameters as well as the base-learners initializations such that after a small number of steps it can do very well on a given task.

The authors ran experiments on Mini-Imagenet where they exceed the performance of Matching Networks.

In Model Agnostic meta-learning (MAML) BID7 ) the authors proposed increasing the gradient update steps on the base-model and replacing the meta-learner LSTM with Batch Stochastic Gradient Descent BID11 , which as a result speeds up the process of learning and interestingly improves generalization performance and achieves state of the art performance in both Omniglot and Mini-Imagenet.

In Meta-SGD BID15 ) the authors proposed learning a static learning rate and an update direction for each parameter in the base-model, in addition to learning the initialization parameters of the base-model.

Meta-SGD showcases significantly improved generalization performance (when compared to MAML) across all few-shot learning tasks, whilst only requiring a single inner loop update step.

However this practice effectively doubles the model parameters and computational overheads of the system.

Model Agnostic Meta-Learning (MAML) BID7 ) is a meta-learning framework for fewshot learning.

MAML is elegantly simple yet can produce state of the art results in few-shot regression/classification and reinforcement learning problems.

In a sentence, MAML learns good initialization parameters for a network, such that after a few steps of standard training on a few-shot dataset, the network will perform well on that few shot task.

More formally, we define the base model to be a neural network f θ with meta-parameters θ.

We want to learn an initial θ = θ 0 that, after a small number N of gradient update steps on data from a support set S b to obtain θ N , the network performs well on that task's target set T b .

Here b is the index of a particular support set task in a batch of support set tasks.

This set of N updates steps is called the inner-loop update process.

The updated base-network parameters after i steps on data from the support task S b can be expressed as: DISPLAYFORM0 where α is the learning rate, θ b i are the base-network weights after i steps towards task b, DISPLAYFORM1 ) is the loss on the support set of task b after (i − 1) (i.e. the previous step) update steps.

Assuming that our task batch size is B we can define a meta-objective, which can be expressed as: DISPLAYFORM2 where we have explicitly denoted the dependence of θ b N on θ 0 , given by unrolling (1).

The objective (2) measures the quality of an initialization θ 0 in terms of the total loss of using that initialization across all tasks.

This meta objective is now minimized to optimize the initial parameter value θ 0 .

It is this initial θ 0 that contains the across-task knowledge.

The optimization of this meta-objective is called the outer-loop update process.

The resulting update for the meta-parameters θ 0 can be expressed as: DISPLAYFORM3 where β is a learning rate and L T b denotes the loss on the target set for task b.

In this paper we use the cross-entropy BID5 BID20 loss throughout.

The simplicity, elegance and high performance of MAML make it a very powerful framework for meta-learning.

However, MAML has also many issues that make it problematic to use.

Training Instability: Depending on the neural network architecture and the overall hyperparameter setup, MAML can be very unstable during training as illustrated in FIG0 .

Optimizing the outer loop involved backpropagating derivatives through an unfolded inner loop consisting of the same network multiple times.

This alone could be cause for gradient issues.

However, the gradient issues are further compounded by the model architecture, which is a standard 4-layer convolutional network without skip-connections.

The lack of any skip-connections means that every gradient must be passed through each convolutional layer many times; effectively the gradients will be multiplied by the same sets of parameters multiple times.

After multiple back-propagation passes, the large depth structure of the unfolded network and lack of skip connections can cause gradient explosions and diminishing gradient problems respectively.

Second Order Derivative Cost: Optimization through gradient update steps requires the computation of second order gradients which are very expensive to compute.

The authors of MAML proposed using first-order approximations to speed up the process by a factor of three, however using these approximations can have a negative impact on the final generalization error.

Further attempts at using first order methods have been attempted in Reptile BID18 where the authors apply standard SGD on a base-model and then take a step from their initialization parameters towards the parameters of the base-model after N steps.

The results of Reptile vary, in some cases exceeding MAML, and in others producing results inferior to MAML.

Approaches to reduce computation time while not sacrificing generalization performance have yet to be proposed.

Absence of Batch Normalization Statistic Accumulation: A further issue that affects the generalization performance is the way that batch normalization is used in the experiments in the original MAML paper.

Instead of accumulating running statistics, the statistics of the current batch were used for batch normalization.

This results in batch normalization being less effective, since the biases learned have to accommodate for a variety of different means and standard deviations instead of a single mean and standard deviation.

On the other hand, if batch normalization uses accumulated running statistics it will eventually converge to some global mean and standard deviation.

This leaves only a single mean and standard deviation to learn biases for.

Using running statistics instead of batch statistics, can greatly increase convergence speed, stability and generalization performance as the normalized features will result in smoother optimization landscape BID21 .Shared (across step) Batch Normalization Bias: An additional problem with batch normalization in MAML stems from the fact that batch normalization biases are not updated in the inner-loop; instead the same biases are used throughout all iterations of base-models.

Doing this implicitly assumes that all base-models are the same throughout the inner loop updates and hence have the same distribution of features passing through them.

This is a false assumption to make, since, with each inner loop update, a new base-model is instantiated that is different enough from the previous one to be considered a new model from a bias estimation point of view.

Thus learning a single set of biases for all iterations of the base-model can restrict performance.

Shared Inner Loop (across step and across parameter) Learning Rate: One issue that affects both generalization and convergence speed (in terms of training iterations) is the issue of using a shared learning rate for all parameters and all update-steps.

Doing so introduces two major problems.

Having a fixed learning rate requires doing multiple hyperparameter searches to find the correct learning rate for a specific dataset; this process can be very computationally costly, depending on how search is done.

The authors in BID15 propose to learn a learning rate and update direction for each parameter of the network.

Doing so solves the issue of manually having to search for the right learning rate, and also allows individual parameters to have smaller or larger learning rates.

However this approach brings its own problems.

Learning a learning rate for each network parameter means increased computational effort and increased memory usage since the network contains between 40K and 50K parameters depending on the dimensionality of the data-points.

Fixed Outer Loop Learning Rate:

In MAML the authors use Adam with a fixed learning rate to optimize the meta-objective.

Annealing the learning rate using either step or cosine functions has proven crucial to achieving state of the art generalization performance in a multitude of settings BID16 BID8 BID13 BID9 .

Thus, we theorize that using a static learning rate reduces MAML's generalization performance and might also be a reason for slower optimization.

Furthermore, having a fixed learning rate might mean that one has to spend more (computational) time tuning the learning rate.

In this section we propose methods for solving the issues with the MAML framework, described in Section 3.1.

Each solution has a reference identical to the reference of the issue it is attempting to solve.

Gradient Instability → Multi-Step Loss Optimization (MSL): MAML works by minimizing the target set loss computed by the base-network after it has completed all of its inner-loop updates towards a support set task.

Instead we propose minimizing the target set loss computed by the base-network after every step towards a support set task.

More specifically, we propose that the loss minimized is a weighted sum of the target set losses after every support set loss update.

More formally: DISPLAYFORM0 Where β is a learning rate, DISPLAYFORM1 ) denotes the target set loss of task b when using the base-network weights after i steps towards minimizing the support set task and v i denotes the importance weight of the target set loss at step i, which is used to compute the weighted sum.

By using the multi-step loss proposed above we improve gradient propagation, since now the basenetwork weights at every step receive gradients both directly (for the current step loss) and indirectly (from losses coming from subsequent steps).

With the original methodology described in Section 3 the base-network weights at every step except the last one were optimized implicitly as a result of backpropagation, which caused many of the instability issues MAML had.

However using the multi-step loss alleviates this issue as illustrated in FIG0 .

Furthermore, we employ an annealed weighting for the per step losses.

Initially all losses have equal contributions towards the loss, but as iterations increase, we decrease the contributions from earlier steps and slowly increase the contribution of later steps.

This is done to ensure that as training progresses the final step loss receives more attention from the optimizer thus ensuring it reaches the lowest possible loss.

If the annealing is not used, we found that the final loss might be higher than with the original formulation.

Second Order Derivative Cost →

Derivative-Order Annealing (DA): One way of making MAML more computationally efficient is reducing the number of inner-loop updates needed, which can be achieved with some of the methods described in subsequent sections of this report.

However, in this paragraph, we propose a method that reduces the per-step computational overhead directly.

The authors of MAML proposed the usage of first-order approximations of the gradient derivatives.

However they applied the first-order approximation throughout the whole of the training phase.

Instead, we propose to anneal the derivative-order as training progresses.

More specifically, we propose to use first-order gradients for the first 50 epochs of the training phase, and to then switch to second-order gradients for the remainder of the training phase.

We empirically demonstrate that doing so greatly speeds up the first 50 epochs, while allowing the second-order training needed to achieve the strong generalization performance the second-order gradients provide to the model.

An additional interesting observation is that derivative-order annealing experiments showed no incidents of exploding or diminishing gradients, contrary to second-order only experiments which were more unstable.

Using first-order before starting to use second-order derivatives can be used as a strong pretraining method that learns parameters less likely to produce gradient explosion/diminishment issues.

Absence of Batch Normalization Statistic Accumulation → Per-Step Batch Normalization Running Statistics (BNRS): In the original implementation of MAML Finn et al. (2017) the authors used only the current batch statistics as the batch normalization statistics.

This, we argue, caused a variety of undesirable effects described in Section 3.1.

To alleviate the issues we propose using running batch statistics for batch normalization.

A naive implementation of batch normalization in the context of MAML would require sharing running batch statistics across all update steps of the inner-loop fast-knowledge acquisition process.

However doing so would cause the undesirable consequence that the statistics stored be shared across all inner loop updates of the network.

This would cause optimization issues and potentially slow down or altogether halt optimization, due to the increasing complexity of learning parameters that can work across various updates of the network parameters.

A better alternative would be to collect statistics in a per-step regime.

To collect running statistics per-step, one needs to instantiate N (where N is the total number of inner-loop update steps) sets of running mean and running standard deviation for each batch normalization layer in the network and update the running statistics respectively with the steps being taken during the optimization.

The per-step batch normalization methodology should speed up optimization of MAML whilst potentially improving generalization performance.

In the MAML paper the authors trained their model to learn a single set of biases for each layer.

Doing so assumes that the distributions of features passing through the network are similar.

However, this is a false assumption since the base-model is updated for a number of times, thus making the feature distributions increasingly dissimilar from each other.

To fix this problem we propose learning a set of biases per-step within the inner-loop update process.

Doing so, means that batch normalization will learn biases specific to the feature distributions seen at each set, which should increase convergence speed, stability and generalization performance.

Shared Inner Loop Learning Rate (across step and across parameter) →

Learning Per-Layer Per-Step Learning Rates and Gradient Directions (LSLR): Previous work in BID15 demonstrated that learning a learning rate and gradient direction for each parameter in the basenetwork improved the generalization performance of the system.

However, that had the consequence of increased number of parameters and increased computational overhead.

So instead, we propose, learning a learning rate and direction for each layer in the network as well as learning different learning rates for each adaptation of the base-network as it takes steps.

Learning a learning rate and direction for each layer instead for each parameter should reduce memory and computation needed whilst providing additional flexibility in the update steps.

Furthermore, for each learning rate learned, there will be N instances of that learning rate, one for each step to be taken.

By doing this, the parameters are free to learn to decrease the learning rates at each step which may help alleviate overfitting.

Fixed Outer Loop Learning Rate → Cosine Annealing of Meta-Optimizer Learning Rate (CA): In MAML the authors use a static learning rate for the optimizer of the meta-model.

Annealing the learning rate, either by using step-functions BID8 or cosine functions BID16 has proved vital in learning models with higher generalization power.

The cosine annealing scheduling has been especially effective in producing state of the art results whilst removing the need for any hyper-parameter searching on the learning rate space.

Thus, we propose applying the cosine annealing scheduling on the meta-model's optimizer (i.e. the meta-optimizer).

Annealing the learning rate allows the model to fit the training set more effectively and as a result might produce higher generalization performance.

The datasets used to evaluate our methods were the Omniglot (Lake et al., 2015) and Mini-Imagenet BID25 BID19 datasets.

Each dataset is split into 3 sets, a training, validation and test set.

The Omniglot dataset is composed of 1623 characters classes from various alphabets.

There exist 20 instances of each class in the dataset.

For Omniglot we shuffle all character classes and randomly select 1150 for the training set and from the remaining classes we use 50 for validation and 423 for testing.

In most few-shot learning papers the first 1200 classes are used for training and the remaining for testing.

However, having a small validation set to choose the best model is crucial, so we choose to use a small set of 50 classes as validation set.

For each class we use all available 20 samples in the sets.

Furthermore for the Omniglot dataset, data augmentation is used on the images in the form of rotations of 90 degree increments.

Class samples that are rotated are considered new classes, e.g. a 180 degree rotated character C is considered a different class from a non rotated C, thus effectively having 1623 x 4 classes in total.

However the rotated classes are generated dynamically after the character classes have been split into the sets such that rotated samples from a class reside in the same set (i.e. the training, validation or test set).

The Mini-Imagenet dataset was proposed in BID19 , it consists of 600 instances of 100 classes from the ImageNet dataset, scaled down to 84x84.

We use the split proposed in BID19 , which consists of 64 classes for training, 12 classes for validation and 24 classes for testing.

To evaluate our methods we adopted a hierarchical hyperparameter search methodology.

First we began with the baseline MAML experiments, which were ran on the 5/20-way and 1/5-shot settings on the Omniglot dataset and the 5-way 1/5-shot setting on the Mini-Imagenet dataset.

Then we added each one of our 6 methodologies on top of the default MAML and ran experiments for each one separately.

Once this stage was completed we combined the approaches that showed improvements in either generalization performance or convergence speed (both in terms of number of epochs and clock-time) and ran a final experiment to establish any potential gains from the combination of the techniques.

An experiment consisted of training for 150 epochs, each epoch consisting of 500 iterations.

At the end of each epoch, we evaluated the performance of the model on the validation set.

Upon completion of all epochs, an ensemble of the top 3 performing per-epoch-models on the validation set were applied on the test set, thus producing the final test performance of the model.

An evaluation ran consisted of inference on 600 unique tasks.

A distinction between the training and evaluation tasks, was that the training tasks were generated dynamically continually without repeating previously sampled tasks, whilst the 600 evaluation tasks generated were identical across epochs.

Thus ensuring that the comparison between models was fair, from an evaluation set viewpoint.

Every experiment was repeated for 3 independent runs.

The models were trained using the Adam optimizer with a learning rate of 0.001, β 1 = 0.9 and β 2 = 0.99.

Furthermore, all Omniglot experiments used a task batch size of 16, whereas for the Mini-Imagenet experiments we used a task batch size of 4 and 2 for the 5-way 1-shot and 5-way 5-shot experiments respectively.

Our proposed methodologies are empirically shown to improve the original MAML framework.

In TAB0 one can see how our proposed approach performs on Omniglot.

Each proposed methodology can individually outperform MAML, however, the most notable improvements come from the learned per-step per-layer learning rates and the per-step batch normalization methodology.

In the 5-way 1-shot tasks it achieves 99.47% and in the 20-way Omniglot tasks MAML++ achieves 97.76% and 99.33% in the 1-shot and 5-shot tasks respectively.

MAML++ also showcases improved convergence speed in terms of training iterations required to reach the best validation performance.

Furthermore, the multi-step loss optimization technique substantially improves the training stability of the model as illustrated in FIG0 .

In TAB0 we also include the results of our own implementation of MAML, which reproduces all results except the 20-way 1-shot Omniglot case.

Difficulty in replicating the specific result has also been noted before in BID10 .

We base our conclusions on the relative performance between our own MAML implementation and the proposed methodologies.

TAB1 showcases MAML++on Mini-Imagenet tasks, where MAML++ sets a new state of the art in both the 5-way 1-shot and 5-shot cases where the method achieves 52.15% and 68.32% respectively.

More notably, MAML++ can achieve very strong 1-shot results of 51.05% with only a single inner loop step required.

Not only is MAML++ cheaper due to the usage of derivative order annealing, but also because of the much reduced inner loop steps.

Another notable observation is that MAML++converges to its best generalization performance much faster (in terms of iterations required) when compared to MAML as shown in FIG0 .

In this paper we delve deep into what makes or breaks the MAML framework and propose multiple ways to reduce the inner loop hyperparameter sensitivity, improve the generalization error, stabilize and speed up MAML.

The resulting approach, called MAML++sets a new state of the art across all few-shot tasks, across Omniglot and Mini-Imagenet.

The results of the approach indicate that learning per-step learning rates, batch normalization parameters and optimizing on per-step target losses appears to be key for fast, highly automatic and strongly generalizable few-shot learning.

@highlight

MAML is great, but it has many problems, we solve many of those problems and as a result we learn most hyper parameters end to end, speed-up training and inference and set a new SOTA in few-shot learning