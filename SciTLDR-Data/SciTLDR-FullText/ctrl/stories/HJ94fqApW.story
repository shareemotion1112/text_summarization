Model pruning has become a useful technique that improves the computational efficiency of deep learning, making it possible to deploy solutions in resource-limited scenarios.

A widely-used practice in relevant work assumes that a smaller-norm parameter or feature plays a less informative role at the inference time.

In this paper, we propose a channel pruning technique for accelerating the computations of deep convolutional neural networks (CNNs) that does not critically rely on this assumption.

Instead, it focuses on direct simplification of the channel-to-channel computation graph of a CNN without the need of performing a computationally difficult and not-always-useful task of making high-dimensional tensors of CNN structured sparse.

Our approach takes two stages: first to adopt an end-to-end stochastic training method that eventually forces the outputs of some channels to be constant, and then to prune those constant channels from the original neural network by adjusting the biases of their impacting layers such that the resulting compact model can be quickly fine-tuned.

Our approach is mathematically appealing from an optimization perspective and easy to reproduce.

We experimented our approach through several image learning benchmarks and demonstrate its interest- ing aspects and competitive performance.

Not all computations in a deep neural network are of equal importance.

In a typical deep learning pipeline, an expert crafts a neural architecture, which is trained using a prepared dataset.

The success of training a deep model often requires trial and error, and such loop usually has little control on prioritizing the computations happening in the neural network.

Recently researchers started to develop model-simplification methods for convolutional neural networks (CNNs), bearing in mind that some computations are indeed non-critical or redundant and hence can be safely removed from a trained model without substantially degrading the model's performance.

Such methods not only accelerate computational efficiency but also possibly alleviate the model's overfitting effects.

Discovering which subsets of the computations of a trained CNN are more reasonable to prune, however, is nontrivial.

Existing methods can be categorized from either the learning perspective or from the computational perspective.

From the learning perspective, some methods use a dataindependent approach where the training data does not assist in determining which part of a trained CNN should be pruned, e.g. BID7 and , while others use a datadependent approach through typically a joint optimization in generating pruning decisions, e.g., BID4 and BID1 .

From the computational perspective, while most approaches focus on setting the dense weights of convolutions or linear maps to be structured sparse, we propose here a method adopting a new conception to achieve in effect the same goal.

Instead of regarding the computations of a CNN as a collection of separate computations sitting at different layers, we view it as a network flow that delivers information from the input to the output through different channels across different layers.

We believe saving computations of a CNN is not only about reducing what are calculated in an individual layer, but perhaps more importantly also about understanding how each channel is contributing to the entire information flow in the underlying passing graph as well as removing channels that are less responsible to such process.

Inspired by this new conception, we propose to design a "gate" at each channel of a CNN, controlling whether its received information is actually sent out to other channels after processing.

If a channel "gate" closes, its output will always be a constant.

In fact, each designed "gate" will have a prior intention to close, unless it has a "strong" duty in sending some of its received information from the input to subsequent layers.

We find that implementing this idea in pruning CNNs is unsophisticated, as will be detailed in Sec 4.Our method neither introduces any extra parameters to the existing CNN, nor changes its computation graph.

In fact, it only introduces marginal overheads to existing gradient training of CNNs.

It also possess an attractive feature that one can successively build multiple compact models with different inference performances in a single round of resource-intensive training (as in our experiments).

This eases the process to choose a balanced model to deploy in production.

Probably, the only applicability constraint of our method is that all convolutional layers and fully-connected layer (except the last layer) in the CNN should be batch normalized BID9 .

Given batch normalization has becomes a widely adopted ingredient in designing state-of-the-art deep learning models, and many successful CNN models are using it, we believe our approach has a wide scope of potential impacts.

In this paper, we start from rethinking a basic assumption widely explored in existing channel pruning work.

We point out several issues and gaps in realizing this assumption successfully.

Then, we propose our alternative approach, which works around several numerical difficulties.

Finally, we experiment our method across different benchmarks and validate its usefulness and strengths.

Reducing the size of neural network for speeding up its computational performance at inference time has been a long-studied topic in the communities of neural network and deep learning.

Pioneer works include Optimal Brain Damage BID11 and Optimal Brain Surgeon BID5 .

More recent developments focused on either reducing the structural complexity of a provided network or training a compact or simplified network from scratch.

Our work can be categorized into the former, thus the literature review below revolves around reducing the structural complexity.

To reduce the structural complexity of deep learning models, previous work have largely focused on sparsifying the weights of convolutional kernels or the feature maps across multiple layers in a network BID1 BID4 .

Some recent efforts proposed to impose structured sparsity on those vector components motivated from the implementation perspective on specialized hardware BID18 BID21 BID0 BID10 ).

Yet as argued by BID16 , regularization-based pruning techniques require per layer sensitivity analysis which adds extra computations.

Their method relies on global rescaling of criteria for all layers and does not require sensitivity estimation, a beneficial feature that our approach also has.

To our knowledge, it is also unclear how widely useful those works are in deep learning.

In Section 3, we discuss in details the potential issues in regularization-based pruning techniques potentially hurting them being widely applicable, especially for those that regularize high-dimensional tensor parameters or use magnitude-based pruning methods.

Our approach works around the mentioned issues by constraining the anticipated pruning operations only to batchnormalized convolutional layers.

Instead of posing structured sparsity on kernels or feature maps, we enforce sparsity on the scaling parameter γ in batch normalization operator.

This blocks the sample-wise information passing through part of the channels in convolution layer, and in effect implies one can safely remove those channels.

A recent work by BID8 used a similar technique as ours to remove unimportant residual modules in ResNet by introducing extra scaling factors to the original network.

However, some optimization subtleties as to be pointed out in our paper were not well explained.

Another recent work called Network-Slimming BID15 ) also aims to sparsify the scaling parameters of batch normalization.

But instead of using off-the-shelf gradient learning like theirs, we propose a new algorithmic approach based on ISTA and rescaling trick, improving robustness and speed of the undergoing optimization.

In particular, the work of BID15 was able to prune VGG-A model on ImageNet.

It is unclear how their work would deal with the γ-W rescaling effect and whether their approach can be adopted to large pre-trained models, such as ResNets and Inceptions.

We experimented with the pre-trained ResNet-101 and compared to most recent work that were shown to work well with large CNNs.

We also experimented with an image segmentation model which has an inception-like module (pre-trained on ImageNet) to locate foreground objects.

In most regularized linear regressions, a large-norm coefficient is often a strong indicator of a highly informative feature.

This has been widely perceived in statistics and machine learning communities.

Removing features which have a small coefficient does not substantially affect the regression errors.

Therefore, it has been an established practice to use tractable norm to regularize the parameters in optimizing a model and pick the important ones by comparing their norms after training.

However, this assumption is not unconditional.

By using Lasso or ridge regression to select important predictors in linear models, one always has to first normalize each predictor variable.

Otherwise, the result might not be explanatory.

For example, ridge regression penalizes more the predictors which has low variance, and Lasso regression enforces sparsity of coefficients which are already small in OLS.

Such normalization condition for the right use of regularization is often unsatisfied for nonconvex learning.

For example, one has to carefully consider two issues outlined below.

We provides these two cases to exemplify how regularization could fail or be of limited usage.

There definitely exist ways to avoid the described failures.

Model Reparameterization.

In the first case, we show that it is not easy to have fine-grained control of the weights' norms across different layers.

One has to either choose a uniform penalty in all layers or struggle with the reparameterization patterns.

Consider to find a deep linear (convolutional) network subject to a least square with Lasso: for λ > 0, DISPLAYFORM0 The above formulation is not a well-defined problem because for any parameter set DISPLAYFORM1 , one can always find another parameter set {W i } 2n i=1 such that it achieves a smaller total loss while keeping the corresponding l 0 norm unchanged by actually setting DISPLAYFORM2 where α > 1.

In another word, for any > 0, one can always find a parameter set DISPLAYFORM3 (which is usually non-sparse) that minimizes the first least square loss while having its second Lasso term less than .We note that gradient-based learning is highly inefficient in exploring such model reparameterization patterns.

In fact, there are some recent discussions around this BID3 .

If one adopts a pre-trained model, and augments its original objective with a new norm-based parameter regularization, the new gradient updates may just increase rapidly or it may take a very long time for the variables traveling along the model's reparameterization trajectory.

This highlights a theoretical gap questioning existing sparsity-inducing formulation and actual computational algorithms whether they can achieve widely satisfactory parameter sparsification for deep learning models.

Transform Invariance.

In the second case, we show that batch normalization is not compatible with weight regularization.

The example is penalizing l 1 -or l 2 -norms of filters in convolution layer which is then followed by a batch normalization: at the l-th layer, we let DISPLAYFORM4 where γ and β are vectors whose length is the number of channels.

Likewise, one can clearly see that any uniform scaling of W l which changes its l 1 -and l 2 -norms would have no effects on the output x l+1 .

Alternatively speaking, if one is interested in minimizing the weight norms of multiple layers together, it becomes unclear how to choose proper penalty for each layer.

Theoretically, there always exists an optimizer that can change the weight to one with infinitesimal magnitude without hurting any inference performance.

As pointed by one of the reviewers, one can tentatively avoid this issue by projecting the weights to the surface of unit ball.

Then one has to deal with a non-convex feasible set of parameters, causing extra difficulties in developing optimization for data-dependent pruning methods.

It is also worth noting that some existing work used such strategy in a layer-by-layer greedy way BID7 .Based on this discussion, many existing works which claim to use Lasso, group Lasso (e.g. BID18 ; BID1 ), or thresholding (e.g. BID16 ) to enforce parameter sparsity have some theoretical gaps to bridge.

In fact, many heuristic algorithms in neural net pruning actually do not naturally generate a sparse parameterized solution.

More often, thresholding is used to directly set certain subset of the parameters in the network to zeros, which can be problematic.

The reason is in essence around two questions.

First, by setting parameters less than a threshold to zeros, will the functionality of neural net be preserved approximately with certain guarantees?

If yes, then under what conditions?

Second, how should one set those thresholds for weights across different layers?

Not every layer contributes equally in a neural net.

It is expected that some layers act critically for the performance but only use a small computation and memory budget, while some other layers help marginally for the performance but consume a lot resources.

It is naturally more desirable to prune calculations in the latter kind of layers than the former.

In contrast with these existing approaches, we focus on enforcing sparsity of a tiny set of parameters in CNN -scale parameter γs in all batch normalization.

Not only placing sparse constraints on γ is simpler and easier to monitor, but more importantly, we have two strong reasons:1.

Every γ always multiplies a normalized random variable, thus the channel importance becomes comparable across different layers by measuring the magnitude values of γ; 2.

The reparameterization effect across different layers is avoided if its subsequent convolution layer is also batch-normalized.

In other words, the impacts from the scale changes of γ parameter are independent across different layers.

Nevertheless, our current work still falls short of a strong theoretical guarantee.

We believe by working with normalized feature inputs and their regularized coefficients together, one is closer to a more robust and meaningful approach.

Sparsity is not the goal, but to find less important channels using sparsity inducing formulation is.

We describe the basic principle and algorithm of our channel pruning technique.

Pruning constant channels.

Consider convolution with batch normalization: DISPLAYFORM0 For the ease of notation, we let γ = γ l .

Note that if some element in γ is set to zero, say, γ[k] = 0, its output image x l+1 :,:,:,k becomes a constant β k , and a convolution of a constant image channel is almost everywhere constant (except for padding regions, an issue to be discussed later).

Therefore, we show those constant image channels can be pruned while the same functionality of network is approximately kept:• If the subsequent convolution layer does not have batch normalization, DISPLAYFORM1 its values (a.k.a.

elements in β) is absorbed into the bias term by the following equation DISPLAYFORM2 new , 0 , where * γ denotes the convolution operator which is only calculated along channels indexed by non-zeros of γ.

Remark that DISPLAYFORM3 • If the subsequent convolution layer has batch normalization, DISPLAYFORM4 instead its moving average is updated as DISPLAYFORM5 Remark that the approximation (≈) is strictly equivalence (=) if no padding is used in the convolution operator * , a feature that the parallel work Liu et al. FORMULA11 does not possess.

When the original model uses padding in computing convolution layers, the network function is not strictly preserved after pruning.

In our practice, we fine-tune the pruned network to fix such performance degradation at last.

In short, we formulate the network pruning problem as simple as to set more elements in γ to zero.

It is also much easier to deploy the pruned model, because no extra parameters or layers are introduced into the original model.

To better understand how it works in an entire CNN, imagine a channel-to-channel computation graph formed by the connections between layers.

In this graph, each channel is a node, their inference dependencies are represented by directed edges.

The γ parameter serves as a "dam" at each node, deciding whether let the received information "flood" through to other nodes following the graph.

An end-to-end training of channel pruning is essentially like a flood control system.

There suppose to be rich information of the input distribution, and in two ways, much of the original input information is lost along the way of CNN inference, and the useful part -that is supposed to be preserved by the network inference -should be label sensitive.

Conventional CNN has one way to reduce information: transforming feature maps (non-invertible) via forward propagation.

Our approach introduces the other way: block information at each channel by forcing its output being constant using ISTA.ISTA.

Despite the gap between Lasso and sparsity in the non-convex settings, we found that ISTA BID2 ) is still a useful sparse promoting method.

But we just need to use it more carefully.

Specifically, we adopt ISTA in the updates of γs.

The basic idea is to project the parameter at every step of gradient descent to a potentially more sparse one subject to a proxy problem: let l denote the training loss of interest, at the (t + 1)-th step, we set DISPLAYFORM6 where ∇ γ l t is the derivative with respect to γ computed at step t, µ t is the learning rate, λ is the penalty.

In the stochastic learning, ∇ γ l t is estimated from a mini-batch at each step.

Eq. (1) has closed form solution as DISPLAYFORM7 where prox η (x) = max{|x| − η, 0} · sgn(x).

The ISTA method essentially serves as a "flood control system" in our end-to-end learning, where the functionality of each γ is like that of a dam.

When γ is zero, the information flood is totally blocked, while γ = 0, the same amount of information is passed through in form of geometric quantities whose magnitudes are proportional to γ.

Scaling effect.

One can also see that if γ is scaled by α meanwhile W l+1 is scaled by 1/α, that is, DISPLAYFORM8 the output x l+2 is unchanged for the same input x l .

Despite not changing the output, scaling of γ and W l+1 also scales the gradients ∇ γ l and ∇ W l+1 l by 1/α and α, respectively.

As we observed, the parameter dynamics of gradient learning with ISTA depends on the scaling factor α if one decides to choose it other than 1.0.

Intuitively, if α is large, the optimization of W l+1 is progressed much slower than that of γ.

We describe our algorithm below.

The following method applies to both training from scratch or re-training from a pre-trained model.

Given a training loss l, a convolutional neural net N , and hyper-parameters ρ, α, µ 0 , our method proceeds as follows:1.

Computation of sparse penalty for each layer.

Compute the memory cost per channel for each layer denoted by λ l and set the ISTA penalty for layer l to ρλ l .

Here DISPLAYFORM0 where DISPLAYFORM1 h is the size of input image of the neural network.

3. End-to-End training with ISTA on γ.

Train N by the regular SGD, with the exception that γ l s are updated by ISTA, where the initial learning rate is µ 0 .

Train N until the loss l plateaus, the total sparsity of γ l s converges, and Lasso ρ l λ l γ l 1 converges.

4.

Post-process to remove constant channels.

Prune channels in layer l whose elements in γ l are zero and output the pruned model N by absorbing all constant channels into subsequent layers (as described in the earlier section.).

DISPLAYFORM2 5.

γ-W rescaling trick.

For γ l s and weights in N which were scaled in Step 2 before training, scale them by 1/α and α respectively (scaling back).6.

Fine-tune N using regular stochastic gradient learning.

Remark that choosing a proper α as used in Steps 2 and 5 is necessary for using a large µ t · ρ in ISTA, which makes the sparsification progress of γ l s faster.

We summarize the sensitivity of hyper-parameters and their impacts for optimization below:• µ (learning rate): larger µ leads to fewer iterations for convergence and faster progress of sparsity.

But if if µ too large, the SGD approach wouldn't converge.• ρ (sparse penalty): larger ρ leads to more sparse model at convergence.

If trained with a very large ρ, all channels will be eventually pruned.• α (rescaling): we use α other than 1.

only for pretrained models, we typically choose α from {0.001, 0.01, 0.1, 1} and smaller α warms up the progress of sparsity.

We recommend the following parameter tuning strategy.

First, check the cross-entropy loss and the regularization loss, select ρ such that these two quantities are comparable at the beginning.

Second, choose a reasonable learning rate.

Third, if the model is pretrained, check the average magnitude of γs in the network, choose α such that the magnitude of rescaled γ l is around 100µλ l ρ.

We found as long as one choose those parameters in the right range of magnitudes, the optimization progress is enough robust.

Again one can monitor the mentioned three quantities during the training and terminate the iterations when all three quantities plateaus.

There are several patterns we found during experiments that may suggest the parameter tuning has not been successful.

If during the first few epochs the Lasso-based regularization loss keeps decreasing linearly while the sparsity of γs stays near zero, one may decrease α and restart.

If during the first few epochs the sparsity of γs quickly raise up to 100%, one may decrease ρ and restart.

If during the first few epochs the cross-entropy loss keeps at or increases dramatically to a non-informative level, one may decrease µ or ρ and restart.

We experiment with the standard image classification benchmark CIFAR-10 with two different network architectures: ConvNet and ResNet-20 .

We resize images to 32 × 32 and zero-pad them to 40 × 40.

We pre-process the padded images by randomly cropping with size 32 × 32, randomly flipping, randomly adjusting brightness and contrast, and standardizing them such that their pixel values have zero mean and one variance.

ConvNet For reducing the channels in ConvNet, we are interested in studying whether one can easily convert a over-parameterized network into a compact one.

We start with a standard 4-layer convolutional neural network whose network attributes are specified in Table 1 .

We use a fixed learning rate µ t = 0.01, scaling parameter α = 1.0, and set batch size to 125.Model A is trained from scratch using the base model with an initial warm-up ρ = 0.0002 for 30k steps, and then is trained by raising up ρ to 0.001.

After the termination criterion are met, we prune the channels of the base model to generate a smaller network called model A. We evaluate the classification performance of model A with the running exponential average of its parameters.

It is found that the test accuracy of model A is even better than the base model.

Next, we start from the pre-trained model A to create model B by raising ρ up to 0.002.

We end up with a smaller network called model B, which is about 1% worse than model A, but saves about one third parameters.

Likewise, we start from the pre-trained model B to create model C. The detailed statistics and its pruned channel size are reported in Table 1 .

We also train a reference ConvNet from scratch whose channel sizes are 32-64-64-128 with totally 224,008 parameters and test accuracy being 86.3%.

The referenced model is not as good as Model B, which has smaller number of parameters and higher accuracy.

We have two major observations from the experiment: (1) When the base network is overparameterized, our approach not only significantly reduces the number of channels of the base model but also improves its generalization performance on the test set.

(2) Performance degradation seems unavoidable when the channels in a network are saturated, and our approach gives satisfactory tradeoff between test accuracy and model efficiency.

We also want to verify our second observation with the state-of-art models.

We choose the popular ResNet-20 as our base model for the CIFAR-10 benchmark, whose test accuracy is 92%.

We focus on pruning the channels in the residual modules in ResNet-20, which has 9 convolutions in total.

As detailed in Table 2 , model A is trained from scratch using ResNet-20's network structure as its base model.

We use a warm-up ρ = 0.001 for 30k steps and then train with ρ = 0.005.

We are able to remove 37% parameters from ResNet-20 with only about 1 percent accuracy loss.

Likewise, Model B is created from model A with a higher penalty ρ = 0.01.

Table 2 : Comparisons between ResNet-20 and its two pruned versions.

The last columns are the number of channels of each residual modules after pruning.

We experiment our approach with the pre-trained ResNet-101 on ILSVRC2012 image classification dataset .

ResNet-101 is one of the state-of-the-art network architecture in ImageNet Challenge.

We follow the standard pipeline to pre-process images to 224×224 for training ResNets.

We adopt the pre-trained TensorFlow ResNet-101 model whose single crop error rate is 23.6% with about 4.47 × 10 7 parameters.

2 We set the scaling parameter α = 0.01, the initial learning rate µ t = 0.001, the sparsity penalty ρ = 0.1, and the batch size = 128 (across 4 GPUs).

The learning rate is decayed every four epochs with rate 0.86.

We create two pruned models from the different iterations of training ResNet-101: one has 2.36 × 10 7 parameters and the other has 1.73 × 10 7 parameters.

We then fine-tune these two models using the standard way for training ResNet-101, and report their error rates.

The Top-5 error rate increases of both models are less than 0.5%.

The Top-1 error rates are summarized in TAB2 .

To our knowledge, only a few works have reported their performance on this very large-scale benchmark w.r.t.

the Top-1 errors.

We compare our approach with some recent works in terms of model parameter size, flops, and error rates.

As shown in TAB2 , our model v2 has achieved a compression ratio more than 2.5 while maintaining more than 1% lower error rates than that of other state-of-the-art models at comparable size of parameters.

In the first experiment (CIFAR-10), we train the network from scratch and allocate enough steps for both γ and W adjusting their own scales.

Thus, initialization of an improper scale of γ-W is not really an issue given we optimize with enough steps.

But for the pre-trained models which were originally optimized without any constraints of γ, the γs scales are often unanticipated.

It actually takes as many steps as that of training from scratch for γ to warm up.

By adopting the rescaling trick setting α to a smaller value, we are able to skip the warm-up stage and quick start to sparsify γs.

For example, it might take more than a hundred epoch to train ResNet-101, but it only takes about 5-10 epochs to complete the pruning and a few more epochs to fine-tune.

network param size.

flops error (%) ratio resnet-50 pruned BID8

As we have discussed about the two major observations in Section 5.1, a more appealing scenario is to apply our approach in pruning channels of over-parameterized model.

It often happens when one adopts a pre-trained network on a large task (such as ImageNet classification) and fine-tunes the model to a different and smaller task BID16 .

In this case, one might expect that some channels that have been useful in the first pre-training task are not quite contributing to the outputs of the second task.

We describe an image segmentation experiment whose neural network model is composed from an inception-like network branch and a densenet network branch.

The entire network takes a 224 × 224 image and outputs binary mask at the same size.

The inception branch is mainly used for locating the foreground objects while the densenet network branch is used to refine the boundaries around the segmented objects.

This model was originally trained on multiple datasets.

In our experiment, we attempt to prune channels in both the inception branch and densenet branch.

We set α = 0.01, ρ = 0.5, µ t = 2 × 10 −5 , and batch size = 24.

We train the pre-trained base model until all termination criterion are met, and build the pruned model for fine-tuning.

The pruned model saves 86% parameters and 81% flops of the base model.

We also compare the fine-tuned pruned model with the pre-trained base model across different test benchmark.

Mean IOU is used as the evaluation metric.3 It shows that pruned model actually improves over the base model on four of the five test datasets with about 2% ∼ 5%, while it performs worse than the base model on the most challenged dataset DUT-Omron, whose foregrounds might contain multiple objects.

base model pruned model test dataset (#images) mIOU mIOU MSRA10K (Liu et al., 2011) (2,500) 83.4% 85.5% DUT-Omron (Yang et al., 2013) (1,292) 83.2% 79.1% Adobe Flickr-portrait BID17 FORMULA11 88.6% 93.3% Adobe Flickr-hp BID17 (300) 84.5% 89.5% COCO-person BID13 (50) 84.1% 87.5% param.

size 1.02 × 10 Table 4 : mIOU reported on different test datasets for the base model and the pruned model.

We proposed a model pruning technique that focuses on simplifying the computation graph of a deep convolutional neural network.

Our approach adopts ISTA to update the γ parameter in batch normalization operator embedded in each convolution.

To accelerate the progress of model pruning, we use a γ-W rescaling trick before and after stochastic training.

Our method cleverly avoids some possible numerical difficulties such as mentioned in other regularization-based related work, hence is easier to apply for practitioners.

We empirically validated our method through several benchmarks and showed its usefulness and competitiveness in building compact CNN models.

Figure 1 : Visualization of the number of pruned channels at each convolution in the inception branch.

Colored regions represents the number of channels kept.

The height of each bar represents the size of feature map, and the width of each bar represents the size of channels.

It is observed that most of channels in the bottom layers are kept while most of channels in the top layers are pruned.

<|TLDR|>

@highlight

A CNN model pruning method using ISTA and rescaling trick to enforce sparsity of scaling parameters in batch normalization.