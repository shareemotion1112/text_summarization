Model distillation aims to distill the knowledge of a complex model into a simpler one.

In this paper, we consider an alternative formulation called dataset distillation: we keep the model fixed and instead attempt to distill the knowledge from a large training dataset into a small one.

The idea is to synthesize a small number of data points that do not need to come from the correct data distribution, but will, when given to the learning algorithm as training data, approximate the model trained on the original data.

For example, we show that it is possible to compress 60,000 MNIST training images into just 10 synthetic distilled images (one per class) and achieve close to the original performance, given a fixed network initialization.

We evaluate our method in various initialization settings.

Experiments on multiple datasets, MNIST, CIFAR10, PASCAL-VOC, and CUB-200, demonstrate the ad-vantage of our approach compared to alternative methods.

Finally, we include a real-world application of dataset distillation to the continual learning setting: we show that storing distilled images as episodic memory of previous tasks can alleviate forgetting more effectively than real images.

proposed network distillation as a way to transfer the knowledge from an ensemble of many separately-trained networks into a single, typically compact network, performing a type of model compression.

In this paper, we are considering a related but orthogonal task: rather than distilling the model, we propose to distill the dataset.

Unlike network distillation, we keep the model fixed but encapsulate the knowledge of the entire training dataset, which typically contains thousands to millions of images, into a small number of synthetic training images.

We show that we can go as low as one synthetic image per category, training the same model to reach surprisingly good performance on these synthetic images.

For example, in Figure 1a , we compress 60, 000 training images of MNIST digit dataset into only 10 synthetic images (one per category), given a fixed network initialization.

Training the standard LENET on these 10 images yields test-time MNIST recognition performance of 94%, compared to 99% for the original dataset.

For networks with unknown random weights, 100 synthetic images train to 89%.

We name our method Dataset Distillation and these images distilled images.

But why is dataset distillation interesting?

First, there is the purely scientific question of how much data is encoded in a given training set and how compressible it is?

Second, we wish to know whether it is possible to "load up" a given network with an entire dataset-worth of knowledge by a handful of images.

This is in contrast to traditional training that often requires tens of thousands of data samples.

Finally, on the practical side, dataset distillation enables applications that require compressing data with its task.

We demonstrate that under the continual learning setting, storing distilled images as memory of past task and data can alleviate catastrophic forgetting (McCloskey and Cohen, 1989) .

A key question is whether it is even possible to compress a dataset into a small set of synthetic data samples.

For example, is it possible to train an image classification model on synthetic images that are not on the manifold of natural images?

Conventional wisdom would suggest that the answer is no, as the synthetic training data may not follow the same distribution of the real test data.

Yet, in this work, we show that this is indeed possible.

We present an optimization algorithm for synthesizing a small number of synthetic data samples not only capturing much of the original training data but also tailored explicitly for fast model training with only a few data point.

To achieve our goal, we first derive the network weights as a We distill the knowledge of tens of thousands of images into a few synthetic training images called distilled images.

On MNIST, 100 distilled images can train a standard LENET with a random initialization to 89% test accuracy, compared to 99% when fully trained.

On CIFAR10, 100 distilled images can train a network with a random initialization to 41% test accuracy, compared to 80% when fully trained.

In Section 3.6, we show that these distilled images can efficiently store knowledge of previous tasks for continual learning.

differentiable function of our synthetic training data.

Given this connection, instead of optimizing the network weights for a particular training objective, we optimize the pixel values of our distilled images.

However, this formulation requires access to the initial weights of the network.

To relax this assumption, we develop a method for generating distilled images for randomly initialized networks.

To further boost performance, we propose an iterative version, where the same distilled images are reused over multiple gradient descent steps so that the knowledge can be fully transferred into the model.

Finally, we study a simple linear model, deriving a lower bound on the size of distilled data required to achieve the same performance as training on the full dataset.

We demonstrate that a handful of distilled images can be used to train a model with a fixed initialization to achieve surprisingly high performance.

For networks pre-trained on other tasks, our method can find distilled images for fast model fine-tuning.

We test our method on several initialization settings: fixed initialization, random initialization, fixed pre-trained weights, and random pre-trained weights.

Extensive experiments on four publicly available datasets, MNIST, CIFAR10, PASCAL-VOC, and CUB-200, show that our approach often outperforms existing methods.

Finally, we demonstrate that for continual learning methods that store limited-size past data samples as episodic memory (Lopez-Paz and Ranzato, 2017; Kirkpatrick et al., 2017) , storing our distilled data instead is much more effective.

Our distilled images contain richer information about the past data and tasks, and we show experimental evidence on standard continual learning benchmarks.

Our code, data, and models will be available upon publication.

Knowledge distillation.

The main inspiration for this paper is network distillation (Hinton et al., 2015) , a widely used technique in ensemble learning (Radosavovic et al., 2018) and model compression (Ba and Caruana, 2014; Romero et al., 2015; Howard et al., 2017) .

While network distillation aims to distill the knowledge of multiple networks into a single model, our goal is to compress the knowledge of an entire dataset into a few synthetic data.

Our method is also related to the theoretical concept of teaching dimension, which specifies the minimal size of data needed to teach a target model to a learner (Shinohara and Miyano, 1991; Goldman and Kearns, 1995) .

However, methods (Zhu, 2013; 2015) inspired by this concept require the existence of target models, which our method does not.

Dataset pruning, core-set construction, and instance selection.

Another way to distill knowledge is to summarize the entire dataset by a small subset, either by only using the "valuable" data for model training (Angelova et al., 2005; Felzenszwalb et al., 2010; Lapedriza et al., 2013) or by only labeling the "valuable" data via active learning (Cohn et al., 1996; Tong and Koller, 2001) .

Similarly, core-set construction (Tsang et al., 2005; Har-Peled and Kushal, 2007; Bachem et al., 2017; Sener and Savarese, 2018) and instance selection (Olvera-López et al., 2010) methods aim to select a subset of the entire training data, such that models trained on the subset will perform as well as the model trained on the full dataset.

For example, solutions to many classical linear learning algorithms, e.g., Perceptron (Rosenblatt, 1957) and SVMs (Hearst et al., 1998) , are weighted sums of subsets of training examples, which can be viewed as core-sets.

However, algorithms constructing these subsets require many more training examples per category than we do, in part because their "valuable" images have to be real, whereas our distilled images are exempt from this constraint.

Gradient-based hyperparameter optimization.

Our work bears similarity with gradient-based hyperparameter optimization techniques, which compute the gradient of hyperparameter w.r.t.

the final validation loss by reversing the entire training procedure (Bengio, 2000; Domke, 2012; Maclaurin et al., 2015; Pedregosa, 2016) .

We also backpropagate errors through optimization steps.

However, we use only training set data and focus more heavily on learning synthetic training data rather than tuning hyperparameters.

To our knowledge, this direction has only been slightly touched on previously (Maclaurin et al., 2015) .

We explore it in greater depth and demonstrate the idea of dataset distillation in various settings.

More crucially, our distilled images work well across random initialization weights, not possible by prior work.

Understanding datasets.

Researchers have presented various approaches for understanding and visualizing learned models (Zeiler and Fergus, 2014; Zhou et al., 2015; Mahendran and Vedaldi, 2015; Bau et al., 2017; Koh and Liang, 2017) .

Unlike these approaches, we are interested in understanding the intrinsic properties of the training data rather than a specific trained model.

Analyzing training datasets has, in the past, been mainly focused on the investigation of bias in datasets (Ponce et al., 2006; Torralba and Efros, 2011) .

For example, Torralba and Efros (2011) proposed to quantify the "value" of dataset samples using cross-dataset generalization.

Our method offers a different perspective for understanding datasets by distilling full datasets into a few synthetic samples.

Given a model and a dataset, we aim to obtain a new, much-reduced synthetic dataset which performs almost as well as the original dataset.

We first present our main optimization algorithm for training a network with a fixed initialization with one gradient descent (GD) step (Section 3.1).

In Section 3.2, we derive the resolution to a more challenging case, where initial weights are random rather than fixed.

In Section 3.3, we further study a linear network case to help readers understand both the properties and limitations of our method.

We also discuss the distribution of initial weights with which our method can work well.

In Section 3.4, we extend our approach to reuse the same distilled images over 2, 000 gradient descent steps and largely improve the performance.

Finally, Section 3.5 discusses dataset distillation for different initialization distributions.

Finally, in Section 3.6, we show that our distilled images can be used as effective episodic memory for continual learning tasks.

, we parameterize our neural network as θ and denote (x i , θ) as the loss function that represents the loss of this network on a data point x i .

Our task is to find the minimizer of the empirical error over entire training data:

where for notation simplicity we overload the (·) notation so that (x, θ) represents the average error of θ over the entire dataset.

We make the mild assumption that is twice-differentiable, which holds true for the majority of modern machine learning models and tasks.

Standard training usually applies minibatch stochastic gradient descent or its variants.

At each step t, a minibatch of training data x t = {x t,j } n j=1 is sampled to update the current parameters as

where η is the learning rate.

Such a training process often takes tens of thousands or even millions of update steps to converge.

Instead, we learn a tiny set of synthetic distilled training datax

with M N and a corresponding learning rateη so that a single GD step such as

Input: p(θ0): distribution of initial weights; M : the number of distilled data Input: α: step size; n: batch size; T : the number of optimization iterations;η0: initial value forη

either from N (0, I) or from real training images.

Initializeη ←η0 2: for each training step t = 1 to T do 3:

Get a minibatch of real training data xt = {xt,j} n j=1

4: Sample a batch of initial weights θ

for each sampled θ

Compute updated parameter with GD: θ

Evaluate the objective function on real training data:

10: end for Output: distilled datax and optimized learning rateη using these learned synthetic datax can greatly boost the performance on the real test set.

Given an initial θ 0 , we obtain these synthetic datax and learning rateη by minimizing the objective below L:

where we derive the new weights θ 1 as a function of distilled datax and learning rateη using Equation 2 and then evaluate the new weights over all the real training data x. The loss L(x,η; θ 0 ) is differentiable w.r.t.x andη, and can thus be optimized using standard gradient-based methods.

In many classification tasks, the data x may contain discrete parts, e.g., class labels in data-label pairs.

For such cases, we fix the discrete parts rather than learn them.

Unfortunately, the above distilled data is optimized for a given initialization, and does not generalize well to other initializations, as it encodes the information of both the training dataset x and a particular network initialization θ 0 .

To address this issue, we turn to calculate a small number of distilled data that can work for networks with random initializations from a specific distribution.

We formulate the optimization problem as follows:

where the network initialization θ 0 is randomly sampled from a distribution p(θ 0 ).

During our optimization, the distilled data are optimized to work well for randomly initialized networks.

In practice, we observe that the final distilled data generalize well to unseen initializations.

In addition, these distilled images often look quite informative, encoding the discriminative features of each category (e.g., in Figure 2 ).

Algorithm 1 illustrates our main method.

As the optimization (Equation 4) is highly non-linear and complex, the initialization ofx plays a critical role in the final performance.

We experiment with different initialization strategies and observe that using random real images as initialization often produces better distilled images compared to random initialization, e.g., N (0, I).

For a compact set distilled data to be properly learned, it turns out having only one GD step is far from sufficient.

Next, we derive a lower bound on the size of distilled data needed for a simple model with arbitrary initial θ 0 in one GD step, and discuss its implications on our algorithm.

This section studies our formulation in a simple linear regression problem with quadratic loss.

We derive a lower bound of the size of distilled data needed to achieve the same performance as training on the full dataset for arbitrary initialization with one GD step.

Consider a dataset x containing N data-target pairs

, where d i ∈ R D and t i ∈ R, which we represent as two matrices: an N × D data matrix d and an N × 1 target matrix t. Given the mean squared error metric and a D × 1 weight matrix θ, we have

We aim to learn M synthetic data-target pairsx = (d,t), whered is an M × D matrix,t an M × 1 matrix (M N ), andη the learning rate, to minimize (x, θ 0 −η∇ θ0 (x, θ 0 )).

The updated weight matrix after one GD step with these distilled data is

For the quadratic loss, there always exists distilled datax that can achieve the same performance as training on the full dataset x (i.e., attaining the global minimum) for any initialization θ 0 .

For example, given any global minimum solution θ * , we can choosed = N · I andt = N · θ * .

But how small can the size of the distilled data be?

For such models, the global minimum is attained at any θ *

in the condition above, we have

Here we make the mild assumption that the feature columns of the data matrix d are independent (i.e., d T d has full rank).

For ax = (d,t) to satisfy the above equation for any θ 0 , we must have

which implies thatd Td has full rank and M ≥ D.

Discussion.

The analysis above only considers a simple case but suggests that any small number of distilled data fail to generalize to arbitrary initial θ 0 .

This is intuitively expected as the optimization target (x, θ 1 ) = (x, θ 0 −η∇ θ0 (x, θ 0 )) depends on the local behavior of (x, ·) around θ 0 (e.g., gradient magnitude), which can be drastically different across various initializations θ 0 .

The lower bound M ≥ D is a quite restricting one, considering that real datasets often have thousands to even hundreds of thousands of dimensions (e.g., images).

This analysis motivates us to avoid the limitation of using one GD step by extending to multiple steps in the next section.

We extend Algorithm 1 to more than one gradient descent steps by changing Line 6 to multiple sequential GD steps on the same batch of distilled data, i.e., each step i performs

and changing Line 9 to backpropagate through all steps.

We do not share the same learning rates across steps as later steps often require lower learning rates.

Naively computing gradients is memory and computationally intensive.

Therefore, we exploit a recent technique called back-gradient optimization, which allows for significantly faster gradient calculation in reverse-mode differentiation (Domke, 2012; Maclaurin et al., 2015) .

Specifically, back-gradient optimization formulates the necessary second-order terms into efficient Hessian-vector products (Pearlmutter, 1994) , which can be easily calculated with modern automatic differentiation systems such as PyTorch (Paszke et al., 2017) .

There is freedom in choosing the distribution of initial weights p(θ 0 ).

In this work, we explore the following four practical choices in the experiments: • Random initialization: Distribution over random initial weights, e.g., He Initialization (He et al., 2015) and Xavier Initialization (Glorot and Bengio, 2010) for neural networks.

• Fixed initialization: A particular fixed network initialized by the method above.

• Random pre-trained weights: Distribution over models pre-trained on other tasks or datasets, e.g., ALEXNET networks trained on ImageNet (Deng et al., 2009 ).

• Fixed pre-trained weights: A particular fixed network pre-trained on other tasks and datasets.

Distillation with pre-trained weights.

Such learned distilled data essentially fine-tune weights pre-trained on one dataset to perform well for a new dataset, thus bridging the gap between the two domains.

Domain mismatch and dataset bias represent a challenging problem in machine learning (Torralba and Efros, 2011; Daume III, 2007; Saenko et al., 2010) .

In this work, we characterize the domain mismatch via distilled data.

In Section 4.1.2, we show that a small number of distilled images are sufficient to quickly adapt convolutional neural network (CNN) models to new datasets and tasks.

To guard against domain shift, several continual learning methods store a subset of training samples in a small memory buffer, and restrict future updates to maintain reasonable performance on these stored samples (Rebuffi et al., 2017; Kirkpatrick et al., 2017; Lopez-Paz and Ranzato, 2017; Nguyen et al., 2018) .

As our distilled images contain rich information about the past training data and task, they could naturally serve as a compressed memory of the past.

To test this, we modify a recent continual learning method called Gradient Episodic Memory (GEM) (Lopez-Paz and Ranzato, 2017).

GEM enforces inequality constraints such that the new model, after being trained on the new data and task, should perform at least as well as the old model on the previously stored data and tasks.

Here, we store our distilled data for each task instead of randomly drawn training samples as used in GEM.

We use the distilled data to construct inequality constraints, and solve the optimization using quadratic programming, same as in GEM.

As shown in Section 4.2, our method compares favorably against several baselines that rely on real images.

In this section, we report experiments of regular image classifications on MNIST (LeCun, 1998) and CIFAR10 (Krizhevsky and Hinton, 2009) , adaptation from ImageNet (Deng et al., 2009 ) to PASCAL-VOC (Everingham et al., 2010) and CUB-200 (Wah et al., 2011) , and continual learning on permuted MNIST and CIFAR100.

Baselines.

For each experiment, in addition to baselines specific to the setting, we generally compare our method against baselines trained with data derived or selected from real training images:

• Random real images: We randomly sample the same number of real images per category.

• Optimized real images: We sample different sets of random real images as above, and choose the top 20% best performing sets.

• k-means++:

We apply k-means++ (Arthur and Vassilvitskii, 2007) clustering to each category, and extract the cluster centroids.

Table 1 : Comparison between our method and various baselines.

All methods use ten images per category (100 in total), except for the average real images baseline, which reuses the same images in different GD steps.

For MNIST, our method uses 2000 GD steps, and baselines use the best among #steps ∈ {1, 100, 500, 1000, 2000}. For CIFAR10, our method uses 50 GD steps, and baselines use the best among #steps ∈ {1, 5, 10, 20, 500}. In addition, we include a K-nearest neighbors (KNN) baseline, and report best results among all combinations of distance metric ∈ {l1, l2} and one or three neighbors.

• Average real images: We compute the average image for each category.

Please see the appendix for more details about training and baselines, and additional results.

We first present experimental results on training classifiers either from scratch or adapting from pre-trained weights.

For MNIST, the distilled images are trained with LENET (LeCun et al., 1998) , which achieves about 99% test accuracy if conventionally trained.

For CIFAR10, we use a network architecture (Krizhevsky, 2012) that achieves around 80% test accuracy if conventionally trained.

For ImageNet adaptations, we use an ALEXNET .

We use 2000 GD steps for MNIST and 50 GD steps for CIFAR10.

For random initializations and random pre-trained weights, we report means and standard deviations over 200 held-out models, unless otherwise stated.

For baselines, we perform each evaluation on 200 held-out models using all possible combinations of learning rate ∈ {distilled learning ratesη * , 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1} and several choices of numbers of training GD steps (see table captions for details), and report results with the best performing combination.

Fixed initialization.

With access to initial network weights, distilled images can directly train a fixed network to reach high performance.

Experiment results show that just 10 distilled images (one per class) can boost the performance of a LENET with an initial accuracy 8.25% to a final accuracy of 93.82% on MNIST in 2000 GD steps.

Using 100 distilled images (ten per class) can raise the final accuracy can be raised to 94.41%, as shown in the first column of Table 1 .

Similarly, 100 distilled images can train a network with an initial accuracy 10.75% to test accuracy of 45.15% on CIFAR10 in 50 GD steps.

Figure 2 distilled images trained with randomly sampled initializations using Xavier Initialization (Glorot and Bengio, 2010) .

While the resulting average test accuracy from these images are not as high as those for fixed initialization, these distilled images crucially do not require a specific initial point, and thus could potentially generalize to a much wider range of starting points.

In Section 4.2 below, we present preliminary results of achieving nontrivial gains from applying such distilled images to classifier networks during a continual learning training process.

Table 2 : Adapting models among MNIST (M), USPS (U), and SVHN (S) using 100 distilled images.

Our method outperforms few-shot domain adaptation (Motiian et al., 2017) and other baselines in most settings.

Due to computation limitations, the 100 distilled images are split into 10 minibatches applied in 10 sequential GD steps, and the entire set of 100 distilled images is iterated through 3 times (30 GD steps in total).

For baselines, we train the model using the same number of images with {1, 3, 5} times and report the best result.

Table 3 : Adapting an ALEXNET pre-trained on ImageNet to PASCAL-VOC and CUB-200.

We use one distilled image per category, repeatedly applied via three GD steps.

Our method significantly outperforms the baselines.

For baselines, we train the model with {1, 3, 5} GD steps and report the best.

Results are over 10 runs.

Multiple gradient descent steps.

Section 3.3 has shown theoretical limitations of using only one step in a simple linear case.

In Figure 3 , we empirically verify for deep networks that using multiple steps drastically outperforms the single step method, given the same number of distilled images.

Table 1 summarizes the results of our method and all baselines.

Our method with both fixed and random initializations outperforms all the baselines on CIFAR10 and most of the baselines on MNIST.

Next, we show the extended setting of our algorithm discussed in Section 3.5, where the weights are not randomly initialized but pre-trained on a particular dataset.

In this section, for random initial weights, we train the distilled images on 2000 pre-trained models and evaluate them on 200 unseen models.

Fixed and random pre-trained weights on digits.

As shown in Section 3.5, we can optimize distilled images to quickly fine-tune pre-trained models on a new dataset.

Table 2 shows that our method is more effective than various baselines on adaptation between three digits datasets: MNIST, USPS (Hull, 1994) , and SVHN (Netzer et al., 2011) .

We also compare our method against a stateof-the-art few-shot domain adaptation method (Motiian et al., 2017) .

Although our method uses the entire training set to compute the distilled images, both methods use the same number of images to distill the knowledge of target dataset.

Prior work (Motiian et al., 2017 ) is outperformed by our method with fixed pre-trained weights on all the tasks, and by our method with random pre-trained weights on two of the three tasks.

This result shows that our distilled images effectively compress the information of target datasets.

Fixed pre-trained ALEXNET to PASCAL-VOC and CUB-200.

In Table 3 , we adapt a widely used ALEXNET model pre-trained on ImageNet to image classification on PASCAL-VOC and CUB-200 datasets.

Given only one distilled image per category, our method outperforms various baselines significantly.

Our method is on par with fine-tuning on the full datasets with thousands of images.

We modify Gradient Episodic Memory (GEM) (Lopez-Paz and Ranzato, 2017) to store distilled data for each task rather than real training images.

Experiments in Lopez-Paz and Ranzato (2017) use large memory buffers, up to 25% of the training set.

Instead, we focus on a more realistic scenario where the buffer is rather small (≤ 1% of the training set).

Following the experiment settings and architecture choices from Lopez-Paz and Ranzato (2017), we consider two continual learning tasks:

Permuted MNIST CIFAR100

Memory size per task = 10 iCaRL (Rebuffi et al., 2017) -42.4 GEM (Lopez-Paz and Ranzato, 2017) 67 No memory buffer EWC (Kirkpatrick et al., 2017) 63.5 45.6 Table 4 : Continual learning results.

Distilled images are trained with random Xavier Initialization distribution.

For permuted MNIST, they are trained with 2000 GD steps.

For CIFAR100, they are trained for 200 GD steps.

• Permuted MNIST: 20 classification tasks each formed by using a different permutation to arrange pixels from MNIST images.

Each task contains 1, 000 training images.

The classifier used has 2 hidden layers each with 100 neurons.

• CIFAR100: 20 classification tasks formed by splitting the 100 classes into 20 equal subsets of 5 classes.

Each task contains 2, 500 training images.

The classifier used is RESNET18 (He et al., 2016) .

Table 4 shows that using distilled data drastically improves final overall accuracy on all tasks, and reduces buffer size by up to 5× compared to the original GEM that uses real images.

We only report the basic iCaRL (Rebuffi et al., 2017) setting on CIFAR100 because it requires similar input distributions across all tasks, and it is unclear how to properly inject distilled images into its specialized examplar selection procedure.

The appendix details the hyper-parameters tested for each continual learning algorithm.

In this paper, we have presented dataset distillation for compressing the knowledge of entire training data into a few synthetic training images.

We demonstrate how to train a network to reach surprisingly good performance with only a small number of distilled images.

Finally, the distilled images can efficiently store the memory of previous tasks in the continual learning setting.

Many challenges remain for knowledge distillation of data.

Although our method generalizes well to random initializations, it is still limited to a particular network architecture.

Since loss surfaces for different architectures might be drastically different, a more flexible method of applying the distilled data may overcome this difficulty.

Another limitation is the increasing computation and memory requirements for finding the distilled data as the number of images and steps increases.

To compress large-scale datasets such as ImageNet, we may need first-order gradient approximations to make the optimization computationally feasible.

Nonetheless, we are encouraged by the findings in this paper on the possibilities of training large models with a few distilled data, leading to potential applications such as accelerating network evaluation in neural architecture search (Zoph and Le, 2017) .

We believe that the ideas developed in this work might give new insights into the quantity and type of data that deep networks are able to process, and hopefully inspire others to think along this direction.

In our experiments, we disable dropout layers in the networks due to the randomness and computational cost they introduce in distillation.

Moreover, we initialize the distilled learning rates with a constant between 0.001 and 0.02 depending on the task, and use the Adam solver (Kingma and Ba, 2015) with a learning rate of 0.001.

For random initialization and random pre-trained weights, we sample 4 to 16 initial weights in each optimization step.

We run all the experiments on NVIDIA 1080 Ti, 2080 Ti, Titan Xp, and V100 GPUs.

We use one GPU for fixed initial weights and up to four GPUs for random initial weights.

Each training typically takes 1 to 6 hours.

Below we describe the details of our baselines using real training images.

• Random real images: We randomly sample the same number of real images per category.

We evaluate the performance over 10 randomly sampled sets.

• Optimized real images: We sample 50 sets of real images using the procedure above, pick 10 sets that achieve the best performance on 20 held-out models and 1024 randomly chosen training images, and evaluate the performance of these 10 sets.

• k-means++:

For each category, we use k-means++ (Arthur and Vassilvitskii, 2007) clustering to extract the same number of cluster centroids as the number of distilled images in our method.

We evaluate the method over 10 runs.

• Average real images: We compute the average image of all the images in each category, which is repeated to match the same total number of images.

We evaluate the model only once because average images are deterministic.

To enforce our optimized learning rate to be positive, we apply softplus to a scalar trained parameter.

For continual learning experiment on CIFAR10 dataset, to compare with GEM (Lopez-Paz and Ranzato, 2017), we replace the Batch normalization (Wu and He, 2018) with Group normalization (Ioffe and Szegedy, 2015) in RESNET18 (He et al., 2016) , as it is difficult to run back-gradient optimization through batch norm running statistics.

For a fair comparison, we use the same architecture for our method and other baselines.

For dataset distillation experiments with pre-trained initial weights, distilled images are initialized with N (0, 1) at the beginning of training.

For other experiments, distilled images are initialized with random real samples, unless otherwise stated.

For the compared continual learning methods, we report the best report from the following combinations of hyper-parameters:

• GEM: -γ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1}.

-learning rate = 0.1.

• iCARL: -regularization ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}.

-learning rate = 0.1.

• Figures 4 and 5 show distilled images trained for random initializations on MNIST and CI-FAR10.

• Figures 6, 7 , and 8 show distilled images trained for adapting random pre-trained models on digits datasets including MNIST, USPS, and SVHN.

Step: 0 LRs: 0.0073, 0.0072, 0.0153 0 1 2 3 4 5 6 7 8 9

Step: 1 LRs: 0.0156, 0.0144, 0.0246

Step: 2 LRs: 0.0114, 0.0543, 0.0371

Step: 3 LRs: 0.0151, 0.0564, 0.0631

Step: 4 LRs: 0.0161, 0.0437, 0.0441

Step: 5 LRs: 0.0538, 0.1200, 0.0960

Step: 6 LRs: 0.0324, 0.0490, 0.0362

Step: 7 LRs: 0.1045, 0.0609, 0.0532

Step: 8 LRs: 0.0375, 0.0465, 0.0437

Step: 9 LRs: 0.1236, 0.1507, 0.0439 Figure 6 : Dataset distillation for adapting random pretrained models from USPS to MNIST.

100 distilled images are split into 10 GD steps, shown as 10 rows here.

Top row is the earliest GD step, and bottom row is the last.

The 10 steps are iterated over three times to finish adaptation, leading to a total of 30 GD steps.

These images train average test accuracy on 200 held out models from 67.54% ± 3.91% to 92.74% ± 1.38%.

Step: 0 LRs: 0.0038, 0.0027, 0.0063 0 1 2 3 4 5 6 7 8 9

Step: 1 LRs: 0.0035, 0.0044, 0.0030

Step: 2 LRs: 0.0039, 0.0040, 0.0047

Step: 3 LRs: 0.0035, 0.0034, 0.0026

Step: 4 LRs: 0.0029, 0.0040, 0.0050

Step: 5 LRs: 0.0022, 0.0032, 0.0027

Step: 6 LRs: 0.0031, 0.0039, 0.0019

Step: 7 LRs: 0.0005, 0.0005, 0.0024

Step: 8 LRs: 0.0008, 0.0005, 0.0018

Step: 9 LRs: 0.0018, 0.0010, 0.0009 Figure 7 : Dataset distillation for adapting random pretrained models from MNIST to USPS.

100 distilled images are split into 10 GD steps, shown as 10 rows here.

Top row is the earliest GD step, and bottom row is the last.

The 10 steps are iterated over three times to finish adaptation, leading to a total of 30 GD steps.

These images train average test accuracy on 200 held out models from 90.43% ± 2.97% to 95.38% ± 1.81%.

Step: 0 LRs: 0.0050, 0.0105, 0.0119 0 1 2 3 4 5 6 7 8 9

Step: 1 LRs: 0.0099, 0.0269, 0.0162

Step: 2 LRs: 0.0049, 0.0232, 0.0160

Step: 3 LRs: 0.0143, 0.0532, 0.0438

Step: 4 LRs: 0.0072, 0.0195, 0.0389

Step: 5 LRs: 0.0228, 0.0540, 0.0382

Step: 6 LRs: 0.0392, 0.0347, 0.0489

Step: 7 LRs: 0.0277, 0.0373, 0.0308

Step: 8 LRs: 0.0525, 0.0225, 0.0192

Step: 9 LRs: 0.0321, 0.0707, 0.0250 Figure 8 : Dataset distillation for adapting random pretrained models from SVHN to MNIST.

100 distilled images are split into 10 GD steps, shown as 10 rows here.

Top row is the earliest GD step, and bottom row is the last.

The 10 steps are iterated over three times to finish adaptation, leading to a total of 30 GD steps.

These images train average test accuracy on 200 held out models from 51.64% ± 2.77% to 85.21% ± 4.73%.

<|TLDR|>

@highlight

We propose to distill a large dataset into a small set of synthetic data that can train networks close to original performance. 