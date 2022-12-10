We introduce a novel method that enables parameter-efficient transfer and multi-task learning with deep neural networks.

The basic approach is to learn a model patch - a small set of parameters - that will specialize to each task, instead of fine-tuning the last layer or the entire network.

For instance, we show that learning a set of scales and biases is sufficient to convert a pretrained network to perform well on qualitatively different problems (e.g. converting a Single Shot MultiBox Detection (SSD) model into a 1000-class image classification model while reusing 98% of parameters of the SSD feature extractor).

Similarly, we show that re-learning existing low-parameter layers (such as depth-wise convolutions) while keeping the rest of the network frozen also improves transfer-learning accuracy significantly.

Our approach allows both simultaneous (multi-task) as well as sequential transfer learning.

In several multi-task learning problems, despite using much fewer parameters than traditional logits-only fine-tuning, we match single-task performance.

Deep neural networks have revolutionized many areas of machine intelligence and are now used for many vision tasks that even few years ago were considered nearly impenetrable BID15 BID26 .

Advances in neural networks and hardware is resulting in much of the computation being shifted to consumer devices, delivering faster response, and better security and privacy guarantees BID11 BID8 .As the space of deep learning applications expands and starts to personalize, there is a growing need for the ability to quickly build and customize models.

While model sizes have dropped dramatically from >50M parameters of the pioneering work of AlexNet BID15 and VGG BID26 to <5M of the recent Mobilenet BID25 BID8 and ShuffleNet BID30 BID19 , the accuracy of models has been improving.

However, delivering, maintaining and updating hundreds of models on the embedded device is still a significant expense in terms of bandwidth, energy and storage costs.

While there still might be space for improvement in designing smaller models, in this paper we explore a different angle: we would like to be able to build models that require only a few parameters to be trained in order to be re-purposed to a different task, with minimal loss in accuracy compared to a model trained from scratch.

While there is ample existing work on compressing models and learning as few weights as possible BID24 BID25 BID8 to solve a single task, to the best of our awareness, there is no prior work that tries to minimize the number of model parameters when solving many tasks together.

Our contribution is a novel learning paradigm in which each task carries its own model patcha small set of parameters -that, along with a shared set of parameters constitutes the model for that task (for a visual description of the idea, see FIG0 , left side).

We put this idea to use in two scenarios: a) in transfer learning, by fine-tuning only the model patch for new tasks, and b) in multi-task learning, where each task performs gradient updates to both its own model patch, and the shared parameters.

In our experiments (Section 5), the largest patch that we used is smaller than 10% of the size of the entire model.

We now describe our contribution in detail.

Transfer learning We demonstrate that, by fine-tuning less than 35K parameters in MobilenetV2 BID25 and InceptionV3 , our method leads to significant accuracy improvements over fine-tuning only the last layer (102K-1.2M parameters, depending on the number of classes) on multiple transfer learning tasks.

When combined with fine-tuning the last layer, we train less than 10% of the model's parameters in total.

We also show the effectiveness of our method over last-layer-based fine-tuning on transfer learning between completely different problems, namely COCO-trained SSD model to classification over ImageNet BID4 .Multi-task learning We explore a multi-task learning paradigm wherein multiple models that share most of the parameters are trained simultaneously (see FIG0 , right side).

Each model has a task-specific model patch.

Training is done in a distributed manner; each task is assigned a subset of available workers that send independent gradient updates to both shared and task-specific parameters using standard optimization algorithms.

Our results show that simultaneously training two such MobilenetV2 BID25 ) models on ImageNet BID4 ) and Places-365 reach accuracies comparable to, and sometimes higher than individually trained models.

We apply our multi-task learning paradigm to domain adaptation.

For ImageNet BID4 , we show that we can simultaneously train MobilenetV2 BID25 ) models operating at 5 different resolution scales, 224, 192, 160, 128 and 96 , while sharing more than 98% of the parameters and resulting in the same or higher accuracy as individually trained models.

This has direct practical benefit in power-constrained operation, where an application can switch to a lower resolution to save on latency/power, without needing to ship separate models and having to make that trade-off decision at the application design time.

The cascade algorithm from BID27 can further be used to reduce the average running time by about 15% without loss in accuracy.

The rest of the paper is organized as follows: we describe our method in Section 2 and discuss related work in Section 3.

In Section 4, we present simple mathematical intuition that contrasts the expressiveness of logit-only fine-tuning and that of our method.

Finally, in Section 5, we present detailed experimental results.

The central concept in our method is that of a model patch.

It is essentially a small set of perchannel transformations that are dispersed throughout the network resulting in only a tiny increase in the number of model parameters.

Suppose a deep network M is a sequence of layers represented by their parameters (weights, biases), W 1 , . . .

, W n .

We ignore non-trainable layers (e.g., some kinds of activations) in this formulation.

A model patch P is a set of parameters W i1 , . . .

, W i k , 1 ≤ i 1 , . . . , i k ≤ n that, when applied to M, adds layers at positions i 1 , . . .

, i n .

Thus, a patched model In this paper, we introduce two kinds of patches.

We will see below that they can be folded with the other layers in the network, eliminating the need to perform any explicit addition of layers.

In Section 5, we shed some light on why the particular choice of these patches is important.

Scale-and-bias patch This patch applies per-channel scale and bias to every layer in the network.

In practice this transformations can often be absorbed into normalization layer such as Batch Normalization BID9 .

Let X be an activation tensor.

Then, the batch-normalized version of X DISPLAYFORM0 where µ(X), σ(X) are mean and standard deviation computed per minibatch, and γ, β are learned via backpropagation.

These statistics are computed as mini-batch average, while during inference they are computed using global averages.

The scale-and-bias patch corresponds to all the γ, β, µ, σ in the network.

Using BN as the model patch also satisfies the criterion that the patch size should be small.

For instance, the BN parameters in both MobilenetV2 BID25 and InceptionV3 network performing classification on ImageNet amounts to less than 40K parameters, of about 1% for MobilenetV2 that has 3.5 million Parameters, and less than 0.2% for Inception V3 that has 25 million parameters.

While we utilize batch normalization in this paper, we note that this is merely an implementation detail and we can use explicit biases and scales with similar results.

Depthwise-convolution patch The purpose of this patch is to re-learn spatial convolution filters in a network.

Depth-wise separable convolutions were introduced in deep neural networks as way to reduce number of parameters without losing much accuracy BID8 BID2 .

They were further developed in BID25 by adding linear bottlenecks and expansions.

In depthwise separable convolutions, a standard convolution is decomposed into two layers: a depthwise convolution layer, that applies one convolutional filter per input channel, and a pointwise layer that computes the final convolutional features by linearly combining the depthwise convolutional layers' output across channels.

We find that the set of depthwise convolution layers can be repurposed as a model patch.

They are also lightweight -for instance, they account for less than 3% of MobilenetV2's parameters when training on ImageNet.

Next, we describe how model patches can be used in transfer and multi-task learning.

Transfer learning In transfer learning, the task is to adapt a pretrained model to a new task.

Since the output space of the new task is different, it necessitates re-learning the last layer.

Following our approach, we apply a model patch and train the patched parameters, optionally also the last layer.

The rest of the parameters are left unchanged.

In Section 5, we discuss the inclusion/exclusion of the last layer.

When the last layer is not trained, it is fixed to its random initial value.

Multitask learning We aim to simultaneously, but independently, train multiple neural networks that share most weights.

Unlike in transfer learning, where a large fraction of the weights are kept frozen, here we learn all the weights.

However, each task carries its own model patch, and trains a patched model.

By training all the parameters, this setting offers more adaptability to tasks while not compromising on the total number of parameters.

To implement multi-task learning, we use the distributed TensorFlow paradigm 1 : a central parameter server receives gradient updates from each of the workers and updates the weights.

Each worker reads the input, computes the loss and sends gradients to the parameter server.

We allow subsets of workers to train different tasks; workers thus may have different computational graphs, and taskspecific input pipelines and loss functions.

A visual depiction of this setting is shown in FIG0 .

One family of approaches BID29 BID5 widely used by practitioners for domain adaptation and transfer learning is based on fine-tuning only the last layer (or sometimes several last layers) of a neural network to solve a new task.

Fine-tuning the last layer is equivalent to training a linear classifier on top of existing features.

This is typically done by running SGD while keeping the rest of the network fixed, however other methods such as SVM has been explored as well BID10 .

It has been repeatedly shown that this approach often works best for similar tasks (for example, see BID5 ).Another frequently used approach is to use full fine-tuning BID3 ) where a pretrained model is simply used as a warm start for the training process.

While this often leads to significantly improved accuracy over last-layer fine-tuning, downsides are that 1) it requires one to create and store a full model for each new task, and 2)

it may lead to overfitting when there is limited data.

In this work, we are primarily interested in approaches that allow one to produce highly accurate models while reusing a large fraction of the weights of the original model, which also addresses the overfitting issue.

While the core idea of our method is based on learning small model patches, we see significant boost in performance when we fine-tune the patch along with last layer (Section 5).

This result is somewhat in contrast with BID7 , where the authors show that the linear classifier (last layer) does not matter when training full networks.

Mapping out the conditions of when a linear classifier can be replaced with a random embedding is an important open question.

BID16 show that re-computing batch normalization statistics for different domains helps to improve accuracy.

In BID24 it was suggested that learning batch normalization layers in an otherwise randomly initialized network is sufficient to build non-trivial models.

Recomputing batch normalization statistics is also frequently used for model quantization where it prevents the model activation space from drifting BID13 .

In the present work, we significantly broaden and unify the scope of the idea and scale up the approach by performing transfer and multi-task learning across completely different tasks, providing a powerful tool for many practical applications.

Our work has interesting connections to meta-learning BID22 BID6 .

For instance, when training data is not small, one can allow each task to carry a small model patch in the Reptile algorithm of BID22 in order to increase expressivity at low cost.

Experiments (Section 5) show that model-patch based fine-tuning, especially with the scale-andbias patch, is comparable and sometimes better than last-layer-based fine-tuning, despite utilizing a significantly smaller set of parameters.

At a high level, our intuition is based on the observation that individual channels of hidden layers of neural network form an embedding space, rather than correspond to high-level features.

Therefore, even simple transformations to the space could result in significant changes in the target classification of the network.

In this section (and in Appendix A), we attempt to gain some insight into this phenomenon by taking a closer look at the properties of the last layer and studying low-dimensional models.

A deep neural network performing classification can be understood as two parts:1.

a network base corresponding to a function F : R d → R n mapping d-dimensional input space X into an n-dimensional embedding space G, and 2.

a linear transformation s : R n → R k mapping embeddings to logits with each output component corresponding to an individual class.

DISPLAYFORM0 We compare fine-tuning model patches with fine-tuning only the final layer s. Fine-tuning only the last layer has a severe limitation caused by the fact that linear transformations preserve convexity.

It is easy to see that, regardless of the details of s, the mapping from embeddings to logits is such that if both ξ a , ξ b ∈ G are assigned label c, the same label is assigned to every DISPLAYFORM1 Thus, if the model assigns inputs {x i |i = 1, . . .

, n c } some class c, then the same class will also be assigned to any point in the preimage of the convex hull of {F (x i )|i = 1, . . .

, n c }.This property of the linear transformation s limits one's capability to tune the model given a new input space manifold.

For instance, if the input space is "folded" by F and the neighborhoods of very different areas of the input space X are mapped to roughly the same neighborhood of the embedding space, the final layer cannot disentangle them while operating on the embedding space alone (should some new task require differentiating between such "folded" regions).We illustrate the difference in expressivity between model-patch-based fine-tuning and last-layerbased fine-tuning in the cases of 1D (below) and 2D (Appendix A) inputs and outputs.

Despite the simplicity, our analysis provides useful insights into how by simply adjusting biases and scales of a neural network, one can change which regions of the input space are folded and ultimately the learned classification function.

In what follows, we will work with a construct introduced by BID21 that demonstrates how neural networks can "fold" the input space X a number of times that grows exponentially with the neural network depth 2 .

We consider a simple neural network with one-dimensional inputs and outputs and demonstrate that a single bias can be sufficient to alter the number of "folds", the topology of the X → G mapping.

More specifically, we illustrate how the number of connected components in the preimage of a one-dimensional segment [ξ a , ξ b ] can vary depending on a value of a single bias variable.

As in BID21 , consider the following function: DISPLAYFORM2 p is an even number, and b = (b 0 , . . .

, b p ) is a (p + 1)-dimensional vector of tunable parameters characterizing q. Function q(x; b) can be represented as a two-layer neural network with ReLU activations.

Set p = 2.

Then, this network has 2 hidden units and a single output value, and is capable of "folding" the input space twice.

Defining F to be a composition of k such functions DISPLAYFORM3 we construct a neural network with 2k layers that can fold input domain R up to 2 k times.

By (1) 0 , the number of "folds" can vary from 2 k to 0.

We evaluate the performance of our method in both transfer and multi-task learning using the image recognition networks MobilenetV2 BID25 and InceptionV3 et al., 2013), Aircraft BID20 , Flowers-102 BID23 and Places-365 ).

An overview of these datasets can be found in TAB0 .

We also show preliminary results on transfer learning across completely different types of tasks using MobilenetV2 and Single-Shot Multibox Detector (SSD) networks.

We use both scale-and-bias (S/B) and depthwise-convolution patches (DW) in our experiments.

Both MobilenetV2 and InceptionV3 have batch normalization -we use those parameters as the S/B patch.

MobilenetV2 has depthwise-convolutions from which we construct the DW patch.

In our experiments, we also explore the effect of fine-tuning the patches along with the last layer of the network.

We compare with two scenarios: 1) only fine-tuning the last layer, and 2) fine-tuning the entire network.

We use TensorFlow BID0 , and NVIDIA P100 and V100 GPUs for our experiments.

Following the standard setup of Mobilenet and Inception we use 224 × 224 images for MobilenetV2 and 299 × 299 for InceptionV3.

As a special-case, for Places-365 dataset, we use 256 × 256 images.

We use RMSProp optimizer with a learning rate of 0.045 and decay factor 0.98 per 2.5 epochs.

To demonstrate the expressivity of the biases and scales, we perform an experiment on MobilenetV2, where we learn only the scale-and-bias patch while keeping the rest of the parameters frozen at their initial random state.

The results are shown in TAB2 (right side).

It is quite striking that simply adjusting biases and scales of random embeddings provides features powerful enough that even a linear classifier can achieve a non-trivial accuracy.

Furthermore, the synergy exhibited by the combination of the last layer and the scale-and-bias patch is remarkable.

We take MobileNetV2 and InceptionV3 models pretrained on ImageNet (Top1 accuracies 71.8% and 76.6% respective), and fine-tune various model patches for other datasets.

Results on InceptionV3 are shown in TAB1 .

We see that fine-tuning only the scale-and-bias patch (using a fixed, random last layer) results in comparable accuracies as fine-tuning only the last layer while using fewer parameters.

Compared to full fine-tuning BID3 , we use orders of magnitude fewer parameters while achieving nontrivial performance.

Our results using MobilenetV2 are similar (more on this later).In the next experiment, we do transfer learning between completely different tasks.

We take an 18-category object detection (SSD) model ) pretrained on COCO images (Lin et al., TAB2 .

Again, we see the effectiveness of training the model patch along with the last layer -a 2% increase in the parameters translates to 19.4% increase in accuracy.

Next, we discuss the effect of learning rate.

It is common practice to use a small learning rate when fine-tuning the entire network.

The intuition is that, when all parameters are trained, a large learning rate results in network essentially forgetting its initial starting point.

Therefore, the choice of learning rate is a crucial factor in the performance of transfer learning.

In our experiments (Appendix B.2, FIG11 ) we observed the opposite behavior when fine-tuning only small model patches: the accuracy grows as learning rate increases.

In practice, fine-tuning a patch that includes the last layer is more stable w.r.t.

the learning rate than full fine-tuning or fine-tuning only the scale-and-bias patch.

Finally, an overview of results on MobilenetV2 with different learning rates and model patches is shown in FIG4 .

The effectiveness of small model patches over fine-tuning only the last layer is again clear.

Combining model patches and fine-tuning results in a synergistic effect.

In Appendix B, we show additional experiments comparing the importance of learning custom bias/scale with simply updating batch-norm statistics (as suggested by BID16 ).

In this section we show that, when using model-specific patches during multi-task training, it leads to performance comparable to that of independently trained models, while essentially using a single model.

We simultaneously train MobilenetV2 BID25 on two large datasets: ImageNet and Places365.

Although the network architecture is the same for both datasets, each model has its own private patch that, along with the rest of the model weights constitutes the model for that dataset.

We choose a combination of the scale-and-bias patch, and the last layer as the private model patch in this experiment.

The rest of the weights are shared and receive gradient updates from all tasks.

In order to inhibit one task from dominating the learning of the weights, we ensure that the learning rates for different tasks are comparable at any given point in time.

This is achieved by setting hyperparameters such that the ratio of dataset size and the number of epochs per learning rate decay step is the same for all tasks.

We assign the same number of workers for each task in the distributed learning environment.

The results are shown in TAB3 .

Multi-task validation accuracy using a separate S/B patch for each model, is comparable to singletask accuracy, while considerably better than the setup that only uses separate logit-layer for each task, while using only using 1% more parameters (and 50% less than the independently trained setup).

In this experiment, each task corresponds to performing classification of ImageNet images at a different resolution.

This problem is of great practical importance because it allows one to build very compact set of models that can operate at different speeds that can be chosen at inference time depending on power and latency requirements.

Unlike in Section 5.3, we only have the scaleand-bias patch private to each task; the last layer weights are shared.

We use bilinear interpolation to scale images before feeding them to the model.

The learning rate schedule is the same as in Section 5.3.The results are shown in TAB4 .

We compare our approach with S/B patch only against two baseline setups.

All shared is where all parameters are shared across all models and individually trained is a much more expensive setup where each resolution has its own model.

As can be seen from the table, scale-and-bias patch allows to close the accuracy gap between these two setups and even leads to a slight increase of accuracy for a couple of the models at the cost of 1% of extra parameters per each resolution.

We introduced a new way of performing transfer and multi-task learning where we patch only a very small fraction of model parameters, that leads to high accuracy on very different tasks, compared to traditional methods.

This enables practitioners to build a large number of models with small incremental cost per model.

We have demonstrated that using biases and scales alone allows pretrained neural networks to solve very different problems.

While we see that model patches can adapt to a fixed, random last layer (also noted in Hoffer et al. FORMULA4 ), we see a significant accuracy boost when we allow the last layer also to be trained.

It is important to close this gap in our understanding of when the linear classifier is important for the final performance.

From an analytical perspective, while we demonstrated that biases alone maintain high expressiveness, more rigorous analysis that would allow us to predict which parameters are important, is still a subject of future work.

From practical perspective, cross-domain multi-task learning (such as segmentation and classification) is a promising direction to pursue.

Finally our approach provides for an interesting extension to the federated learning approach proposed in BID11 , where individual devices ship their gradient updates to the central server.

In this extension we envision user devices keeping their local private patch to maintain personalized model while sending common updates to the server.

Here we show an example of a simple network that "folds" input space in the process of training and associates identical embeddings to different points of the input space.

As a result, fine-tuning the final linear layer is shown to be insufficient to perform transfer learning to a new dataset.

We also show that the same network can learn alternative embedding that avoids input space folding and permits transfer learning.

Consider a deep neural network mapping a 2D input into 2D logits via a set of 5 ReLU hidden layers: 2D input → 8D state → 16D state → 16D state → 8D state → m-D embedding (no ReLU) → 2D logits (no ReLU).

Since the embedding dimension is typically smaller than the input space dimension, but larger than the number of categories, we first choose the embedding dimension m to be 2.

This network is trained (applying sigmoid to the logits and using cross entropy loss function) to map (x, y) pairs to two classes according to the groundtruth dependence depicted in FIG6 .

Learned function is shown in FIG6 (c).

The model is then fine-tuned to approximate categories shown in FIG6 .

Fine-tuning all variables, the model can perfectly fit this new data as shown in FIG6 .Once the set of trainable parameters is restricted, model fine-tuning becomes less efficient.

Interestingly, poor performance of logit fine-tuning seen in figure 4(E) extends to higher embedding dimensions as well.

Plots similar to those in figure 4, but generated for the model with the embedding dimension m of 4 are shown in FIG7 .

In this case, we can see that the final layer fine-tuning is again insufficient to achieve successful transfer learning.

As the embedding dimension goes higher, last layer fine-tuning eventually reaches acceptable results (see FIG8 showing results for m = 8).The explanation behind poor logit fine-tuning results can be seen by plotting the embedding space of the original model with m = 2 (see FIG9 ).

Both circular regions are assigned the same embedding and the final layer is incapable of disentangling them.

But it turns out that the same network could have learned a different embedding that would make last layer fine-tuning much more efficient.

We show this by training the network on the classes shown in figure 7(b).

This class assignment breaks the symmetry and the new learned embedding shown in figure 7(c) can now be used to adjust to new class assignments shown in figure 7(d), (e) and (f) by fine-tuning the final layer alone.

Published as a conference paper at ICLR 2019 The results of BID16 suggested that adjusting Batch Normalization statistics helps with domain adaption.

Interestingly we found that it significantly worsens results for transfer learning, unless bias and scales are allows to learn.

We find that fine-tuning on last layer with batch-norm statistics readjusted to keep activation space at mean 0/variance 1, makes the network to significantly under-perform compared to fine-tuning with frozen statistics.

Even though adding learned bias/scales signifcanty outperforms logit-only based fine-tuning.

We summarize our experiments in TAB5

An application of domain adaptation using model patches is cost-efficient model cascades.

We employ the algorithm from BID27 which takes several models (of varying costs) performing the same task, and determines a cascaded model with the same accuracy as the best task but lower average cost.

Applying it to MobilenetV2 models on multiple resolutions that we trained via multitask learning, we are able to lower the average cost of MobilenetV2 inference by 15.2%.

Note that, in order to achieve this, we only need to store 5% more model parameters than for a single model.

Generally, we did not see a large variation in training speed.

All fine-tuning approaches needed 50-200K steps depending on the learning rate and the training method.

While different approaches definitely differ in the number of steps necessary for convergence, we find these changes to be comparable to changes in other hyperparameters such as learning rate.

@highlight

A novel and practically effective method to adapt pretrained neural networks to new tasks by retraining a minimal (e.g., less than 2%) number of parameters