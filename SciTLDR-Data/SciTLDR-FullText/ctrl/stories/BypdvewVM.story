The backpropagation of error algorithm (BP) is often said to be impossible to implement in a real brain.

The recent success of deep networks in machine learning and AI, however, has inspired a number of proposals for understanding how the brain might learn across multiple layers, and hence how it might implement or approximate BP.

As of yet, none of these proposals have been rigorously evaluated on tasks where BP-guided deep learning has proved critical, or in architectures more structured than simple fully-connected networks.

Here we present the first results on scaling up a biologically motivated model of deep learning to datasets which need deep networks with  appropriate architectures to achieve good performance.

We present results on CIFAR-10 and ImageNet.

For CIFAR-10 we show that our algorithm, a straightforward, weight-transport-free variant of difference target-propagation (DTP) modified to remove backpropagation from the penultimate layer, is competitive with BP in training deep networks with locally defined receptive fields that have untied weights.

For ImageNet we find that both DTP and our algorithm perform significantly worse than BP, opening questions about whether different architectures or algorithms are required to scale these approaches.

Our results and implementation details help establish baselines for biologically motivated deep learning schemes going forward.

The suitability of the backpropagation of error (BP) algorithm BID27 for explaining learning in the brain was questioned soon after its popularization BID8 BID5 .

Weaker objections included undesirable characteristics of artificial networks in general, such as their violation of Dale's Law, their lack of cell-type variability, and the need for the gradient signals to be both positive and negative.

Much more serious objections were: (1) The need for the feedback connections carrying the gradient to have the same weights as the corresponding feedforward connections and (2) The need for a distinct form of information propagation (error propagation) that does not influence neural activity, and hence does not conform to known biological feedback mechanisms underlying neural communication.

Researchers have long sought biologically plausible and empirically powerful learning algorithms that avoid some of these flaws BID1 BID25 BID0 BID23 Xie & Seung, 2003; BID12 BID14 BID10 BID21 .

A common theme of some of the most promising approaches -such as Contrastive Hebbian Learning BID22 , and Generalized Recirculation BID23 -is to use feedback connections to influence neural activity, and to use differences in feedfoward-driven and feedback-driven activities or products of activities to locally approximate gradients BID0 BID26 BID23 Xie & Seung, 2003; BID30 Whittington & Bogacz, 2017) .

Since these activity propagation methods don't require explicit propagation of gradients through the network, they go a long way towards answering the second serious objection noted above.

However, many of these methods require long "positive" and "negative" settling phases for computing the activities or activity products whose differences provide the learning signal.

Proposals for shortening the phases BID11 BID4 are not entirely satisfactory as they still fundamentally depend on a settling process, and, in general, any settling process will likely be too slow for a brain that needs to quickly compute hidden activities.

Indeed, for the same reason, only a handful of the algorithms that require settling have ever been used on large scale problems in machine learning.

Perhaps the most practical among this family of "activity propagation" algorithms is target propagation (TP) and its variants BID17 BID11 BID2 .

The intuition for TP is as follows: Suppose you have a feedforward neural network and have the capacity to compute perfect inverses backwards through the network (i.e., given the activities in layer h l+1 , we can compute h l = f −1 (h l+1 ; θ l+1 )).

If we impose an output target (for a given input) on the output layer, then we can propagate activity backwards through the network to infer what the activities should be to produce the output target.

These backwards propagated activities are denoted the layer targets, or simply targets.

Then, when computing a feedfoward propagation through the network given some input, we can layer-wise compare the feedforward activations to what they should have been (i.e., the targets), and use the differences to compute weight changes.

TP algorithms do not require settling dynamics, and thus can compute forward passes and updates quickly.

As well, for one TP variant , it has been shown that weight changes that cause future feedforward activity to be nudged towards their targets approximate the weight changes computed by BP.While TP and its variants are promising as biologically-plausible algorithms, there are some lingering questions about their applicability to the brain.

First, the only variant explored empirically -difference target propagation (DTP) -still depends on explicit gradient computation via backpropagation for learning the penultimate layer's outgoing synaptic weights (see Algorithm Box 1 in ).

Second, they have not been tested on datasets more difficult than MNIST.

And third, they have not been incorporated into architectures more complicated than simple multi-layer perceptrons (MLPs).In this work we address each of these issues.

Our contribution is threefold: (1) We examine the learning and performance of a biologically-motivated algorithm, Difference Target-propagation (DTP), on MNIST, CIFAR, and ImageNet, (2) We develop a variant of DTP called Simplified Difference Target Propagation (SDTP), which eliminates significant lingering biologically implausible features from DTP, and (3) We investigate the role of weight-sharing convolutions, which are key to performance on difficult datasets in artificial neural networks, by testing the effectiveness of locally connected architectures trained with BP, DTP, and SDTP.Sharing the weights of locally connected units greatly reduces the number of free parameters and this has several very beneficial effects on computer simulations of large neural nets.

It improves generalization and it drastically reduces both the amount of memory needed to store the parameters and the amount of communication required between replicas of the same model running on different subsets of the data on different processors.

From a biological perspective we are interested in how STDP compares with BP without using weight sharing, so both our BP results and our SDTP results are considerably worse than convolutional neural nets and take far longer to produce.

Consider the case of a feed-forward neural network with L layers {h l } L l=1 , whose activations h l are computed by elementwise-applying a non-linear function σ l to an affine transformation of previous layer activations h l−1 : DISPLAYFORM0 with input to the network denoted as h 0 = x and the last layer h L used as output.

For example, in classification problems the output layer h L parametrizes a predicted distribution over possible labels p(y|h L ), usually using the softmax function.

The learning signal is then provided as a loss L(h L ) incurred by making a prediction for an input x, which in the classification case can be cross-entropy between the ground-truth label distribution q(y|x) and the predicted one: DISPLAYFORM1 The goal of training is then to adjust the parameters Θ = {θ l } L l=1 in order to minimize a given loss over the training set of inputs.

In BP and DTP, the final layer target is used to compute a loss, and the gradients from this loss are shuttled backwards (through all layers, in BP, or just one layer, in DTP) in error propagation steps that do not influence actual neural activity.

SDTP never transports gradients using error propagation steps, unlike DTP and BP.

Backpropagation BID27 ) was popularized as a method for learning in neural networks by computing gradients with respect to layer parameters using the chain rule: DISPLAYFORM0 Thus, gradients are obtained by first propagating activations forward to the output layer and then recursively applying these equations.

These equations imply that gradients are propagated backwards through the network using weights symmetric to their feedforward counterparts.

This is biologically problematic because it implies a mode of information propagation (error propagation) that does not influence neural activity, and that depends on an implausible network architecture (symmetric weight connectivity for feedforward and feedback directions, which is called the weight transport problem).

In target propagation BID17 BID2 backwards communication induces neural activity, unlike in BP where backwards communication passes on gradients without inducing neural activity.

The induced activities are those that layers should strive to match so as to produce the target output.

After feedforward propagation given some input, the final output layer h L is trained directly to minimize the loss L, while all other layers are trained so as to match their associated targets.

In general, good targets are those that minimize the loss computed in the output layer if they were actually realized in feedforward propagation.

In networks with invertible layers one could generate such targets by first finding a loss-optimal output activationĥ L (e.g. the correct label distribution) and then propagating it back using inverse transformationsĥ l = f −1 (ĥ l+1 ; θ l+1 ).

Since it is hard to maintain invertibility in a network, approximate inverse transformations (or decoders) can be learned DISPLAYFORM0 Note that this learning obviates the need for symmetric weight connectivity.

The generic form of target propagation algorithms we consider in this paper can be summarized as a scheduled minimization of two kinds of losses for each layer: DISPLAYFORM1 2 2 used to train the approximate inverse that is parametrized similarly to the forward computation g( DISPLAYFORM2 where activations h l−1 are assumed to be propagated from the input.

One can imagine other learning rules for the inverse, for example, the original DTP algorithm trained inverses on noise-corrupted versions of activations with the purpose of improved generalization.

In our implementation we instead used the denoising criterion which we find more biologically plausible, see the appendix for details.

The loss is applied for every layer except the first, since the first layer does not need to propagate target inverses backwards.

2 2 penalizes the layer parameters for producing activations different from their targets.

Parameters of the last layer are trained to minimize the task's loss L directly.

Under this framework both losses are local and involve only single layer's parameters, and implicit dependencies on other layer's parameters are ignored.

Variants differ in the way targetsĥ l are computed.

Target propagation "Vanilla" target propagation (TP) computes targets by propagating the higher layers' targets backwards through layer-wise inverses; i.e.ĥ l = g(ĥ l+1 ; λ l+1 ).

For traditional categorization tasks the same 1-hot vector in the output will always map back to precisely the same hidden unit activities in a given layer.

Thus, this kind of naive TP may have difficulties when different instances of the same class have very different appearances since it will be trying to make their representations identical even in the early layers.

Also, there are no guarantees about how TP will behave when the inverses are imperfect.

Difference target propagation Difference target propagation updates the output weights and biases using the standard gradient rule, but this is biologically unproblematic because it does not require weight transport BID23 BID21 .

For most other layers in the network, difference target propagation (DTP) computes targets asĥ l = h l + g(ĥ l+1 ; λ l+1 ) − g(h l+1 ; λ l+1 ).

The extra terms provide a stabilizing linear correction for imprecise inverse functions.

However, in the original work by the penultimate layer target, h L−1 , was computed using gradients from the network's loss, rather than by target propagation.

That is, DISPLAYFORM0 Though not stated explicitly, this approach was presumably taken to insure that the penultimate layer received reasonable and diverse targets despite the low-dimensional 1-hot targets at the output layer.

When there are a small number of 1-hot targets (e.g. 10 classes), learning a good inverse mapping from these vectors back to the hidden activity of the penultimate hidden layer (e.g. 1000 units) might be problematic, since the inverse mapping cannot provide information that is both useful and unique to a particular sample.

Using BP in the penultimate layer sidesteps this concern, but deviates from the intent of using these algorithms to avoid gradient computation and delivery.

Simplified difference target propagation We introduce SDTP as a simple modification to DTP.

In SDTP we compute the target for the penultimate layer DISPLAYFORM1 .

This completely removes biologically infeasible gradient communication (and hence weight-transport) from the algorithm.

However, it is not clear whether targets for the penultimate layer will be diverse enough (given low entropy classification targets) or precise enough (given the inevitable poor performance of the learned inverse for this layer).

This is a non-trivial change that requires empirical validation.

Parallel and alternating training of inverses In the original implementation of DTP 1 , the authors trained forward and inverse model parameters by alternating between their optimizations; in practice they trained one loss for one full epoch of the training set before switching to training the other loss.

We considered a variant that simply optimizes both losses in parallel, which seems nominally more plausible in the brain since both forward and feedback connections are thought to undergo plasticity changes simultaneously.

Though, it is possible that a kind of alternating learning schedule for forward and backward connections could be tied to wake/sleep cycles.

Noise-preserving versus de-noising autoencoder training In the original DTP algorithm, autoencoder training is done via a noise-preserving loss, which may be a principled choice for the algorithm on a computer .

But in the brain, autoencoder training is de-noising, since uncontrolled noise is necessarily added downstream of a given layer (e.g. by subsequent spiking activity and stochastic vesicle release).

Therefore, in our experiments with TP we use de-noising autoencoder training.

We also compared noise-preserving and de-noising losses in the context of DTP and SDTP and found that they performed roughly equivalently (see Appendix 4).

Propagate activity forward: DISPLAYFORM0 Compute targets for lower layers: DISPLAYFORM1

Convolution-based architectures are critical for achieving state of the art in image recognition BID16 .

These architectures are biologically implausible, however, because of their extensive weight sharing.

To implement convolutions in biology, many neurons would need to share the values of their weights precisely, which is unlikely.

In the absence of weight sharing, the "locally connected" receptive field structure of convolutional neural networks is in fact very biologically realistic and may still offer a useful prior.

Under this prior, neurons in the brain could sample from small areas of visual space, then pool together to create spatial maps of feature detectors.

We assess the the degree to which BP-guided learning is enhanced by convolutions, and not BP per se, by evaluating learning methods (including BP) on networks with locally connected layers.

Since the purpose of our study was not to establish state of the art results, but rather to assess the limitations of biologically-motivated learning methods, we focused on evaluating architectures that were considered reasonable for a particular task or dataset.

Thus, we did not perform an exhaustive architecture search beyond adjusting total number of training parameters to prevent overfitting.

All experiments share the same straightforward methodology: a hyperparameter search was performed for a fixed architecture, for each learning algorithm.

We then selected the best run from each hyperparameter search based on validation set accuracy across 5 consecutive training epochs (i.e. passes over training set) at the end of which we also measured accuracy on the test set.

All locally-connected architectures consist of a stack of locally-connected layers specified as (receptive field size, number of output channels, stride, padding) followed by an output softmax layer.

For padding, SAME denotes padding with zeros to ensure unchanged shape of the output with stride = 1 and VALID padding denotes no padding.

For optimization we use Adam BID13 , with different hyper-parameters for forward and inverse models in the case of target propagation.

All layers are initialized using the method of BID7 .

In all networks we used the hyperbolic tangent as a nonlinearity between layers as it was previously found to work better with DTP than ReLUs .

To compare to previously reported results we began with the MNIST dataset, consisting of 60000 train and 10000 test 28 × 28 gray-scale images of hand-drawn digits, with 10000 images from the train test reserved for validation.

For the evaluation of fully-connected architectures we chose a network from the original DTP paper , consisting of 7 hidden layers with 240 units per layer.

While 7 hidden layers provide arguably excessive capacity for this task, this setup is well-suited for understanding how suitable the considered methods are for learning in relatively deep networks which are known to be prone to exploding or vanishing learning signals.

The locally-connected architecture consisted of 4 hidden layers and has the following structure: (3 × 3, 32, 2, SAME), (3 × 3, 16, 1, SAME), (3 × 3, 16, 1, SAME), (3 × 3, 10, 1, VALID).Results are reported in table 1 and the learning dynamics is plotted on figure 4.

Quite surprisingly, SDTP performed competitively with respect to both DTP and BP, even though it didn't use gradient propagation to assign targets for the penultimate hidden layer.

This suggests that, at least for relatively simple learning tasks, the problem of finite number of targets may not be as serious as one might expect.

Locally connected architectures performed well with all variants of target propagation, and about as well as with BP.

Still, the resulting test accuracy did not match previous known results obtained with convolutional networks, which can produce less than 1% test error, see, e.g. BID19 .

However, the observed improvement in generalization in our experiments must have been solely caused by locally-connected layers, as none of the fully-connected networks with smaller number of hidden layers (and hence with less excessive numbers of parameters) performed similarly.

We noticed that target propagation showed noisier and slower learning comparing to BP (see FIG2 ).

Yet, with early stopping and hyper-parameter tuning it performed competitively.

One can also see that with a fully-connected architecture BP achieved worse test error selected by our methodology.

This is likely explained by the fact that BP overfits to the training set faster (in contrast, none of target propagation variants achieved 0% train error).

These same phenomena were also observed in the locally-connected network.

CIFAR-10 is a more challenging dataset introduced by BID15 .

It consists of 32 × 32 RGB images of 10 categories of objects in natural scenes, split into 50000 train and 10000 test images, where we also reserve 10000 train images for validation.

In contrast to MNIST, classes in CIFAR-10 do not have a "canonical appearance" such as a "prototypical bird" or "prototypical truck" as opposed to "prototypical 7" or "prototypical 9".

This makes them harder to classify with simple template matching, making depth imperative for achieving good performance.

To our best knowledge, this is the first empirical study of biologically-motivated learning methods without weight transport on this dataset.

We considered a fully-connected network with 3 hidden layers of 1024 units and a 5-layer network with locally-connected layers having the following structure: (3 × 3, 32, 2, SAME), (3 × 3, 32, 2, SAME), (3 × 3, 16, 1, SAME), (3 × 3, 16, 2, SAME), (1 × 1, 10, 1, SAME).Final results can be found in table 2.

One can see that with even on a more complex dataset different TP variants, including the most biologically-feasible SDTP performed similarly to BP.

Clearly, the data augmentation employed (random crops and left-right flips) has been necessary for the locallyconnected network to demonstrate a significant improvement over the fully-connected network, otherwise LC models begin to overfit (see FIG2 ).

At the same time, convolutional analog of the LC network has achieved 31.23% and 34.37% of train and test error correspondingly, without use of data augmentation.

This quantitatively demonstrates the need of further advances in biologically-plausible architectures in order to match performance of modern convolutional networks.

Table 3 : Top-1 test error on ImageNet after 18 epochs.

Finally, we assessed performance of the methods on the ImageNet dataset BID28 , a large-scale benchmark that has propelled recent progress in deep learning.

Again, to the best of our knowledge, this is the first empirical study of biologically-motivated methods and architectures conducted on a dataset of such scale and difficulty.

ImageNet consists of 1271167 training examples from which 10000 were reserved for validation and 50000 for testing.

It has 1000 object classes appearing in a variety of natural scenes and captured in high-resolution images (resized to 224 × 224).The locally-connected architecture we considered for this experiment was inspired by the ImageNet architecture used in BID31 .

Unfortunately, the naive replacement of convolutional layers with locally-connected layers would result into a computationally-prohibitive architecture, so we decreased number of output channels in the layers and also removed layers with 1 × 1 filters.

We also slightly decreased filters in the first layer, from 11 × 11 to 9 × 9.

The resulting network had the following architecture: (9 × 9, 48, 4, SAME), pooling, (5 × 5, 64, 1, SAME), pooling, (3 × 3, 96, 1, SAME), pooling, (3 × 3, 128, 1, SAME), spatial 6 × 6 average.

Here every pooling layer is an average pooling with 3 × 3 receptive field.

See the appendix for details of implementing locally-connected networks.

To further reduce the amount of required computation, we included only parallel variants of DTP and SDTP in the evaluation, as these methods are more representative of the biological constraints, and are more straightforward to implement given the size of this dataset's epochs.

Models were trained for 5 days, resulting in 18 passes over training set.

The final results can be found in table 3.

Unlike on MNIST and CIFAR, on ImageNet all variants performed quite poorly.

Additionally, it is on this dataset where we first observed a striking difference between BP and the TP variants.

A number of factors could contribute to this result.

One factor may be that deeper networks might require more careful hyperparameter tuning when using TP; for example, different learning rates or amount of noise injected for each layer.

A second factor may be the difficulty with learning in the output layer, where a 1000-dimensional vector is predicted from just a 128-dimensional output from the final spatial average layer.

Moreover, the inverse computation involves non-compressing learning, which has not been well studied in the context of TP.

Unfortunately, preserving the original 1920 channels in the layer certainly presents a computational challenge.

Addressing both of these factors could help improve performance, so it would be untimely to conclude on any principal inefficiencies of TP.

Therefore, we leave the challenge of matching performance of BP on ImageNet to the future work.

Historically, there has been significant disagreement about whether BP can tell us anything interesting about learning in the brain BID5 BID8 .

Indeed, from the mid 1990s to 2010, work on applying BP to the brain all but disappeared.

Recent progress in machine learning has prompted a revival of this debate; where other approaches have failed, deep networks trained via BP have been key to achieving impressive performance on difficult datasets such as ImageNet.

It is once again natural to wonder whether some approximation of BP might underlie learning in the brain.

However, none of the algorithms proposed as approximations of BP have been tested on the datasets that were instrumental in convincing the machine learning and neuroscience communities to revisit these questions.

Here we introduced a straightforward variant of difference target-propogation that completely removed gradient propagation and weight transport and tested it on the challenging task of classifying CIFAR and ImageNet images.

We also investigated and reported results on the use of local connectivity.

We demonstrated that networks trained with SDTP without any weight sharing (i.e. weight transport in the backward pass or weight tying in convolutions) are generally able to compete with those trained with BP on difficult tasks such as CIFAR.

However, BP significantly outperforms both DTP and SDTP on ImageNet, and more work is required to understand why this issue arises at scale.

We note that although activity-propagation-based algorithms go a long way towards biological plausibility, there are still many biological constraints that we did not address here.

For example, we've set aside the question of spiking neurons entirely to focus on asking whether variants of TP can scale up to solve difficult problems at all.

The question of spiking networks is an important one BID29 BID9 ), but it is nevertheless possible to gain algorithmic insight to the brain without tackling all of the elements of biological complexity simultaneously.

Similarly, we also ignore Dale's law in all of our experiments BID24 .

In general, we've aimed at the simplest models that allow us to address questions around (1) weight sharing, and (2) the form and function of feedback communication.

Algorithms that contend for a role in helping us understand learning in cortex should be able to perform well on difficult domains without relying on weight transport or tying.

Thus, our results offer a new benchmark for future work looking to evaluate the effectiveness of potential biologically plausible algorithms in more powerful architectures and on more difficult datasets.

Although locally-connected layers can be seen as a simple generalization of convolution layers, their implementation is not entirely straightforward.

First, a locally-connected layer has many more trainable parameters than a convolutional layer with an equivalent specification (i.e. receptive field size, stride and number of output channels).

This means that a simple replacement of every convolutional layer with a locally-connected layer can be computationally prohibitive for larger networks.

Thus, one has to decrease the number of parameters in some way to run experiments using a reasonable amount of memory and compute.

In our experiments we opted to decrease the number of output channels in each layer by a given factor.

Obviously, this can have a negative effect on the resulting performance and more work needs to be done to scale locally-connected architectures.

Inverse operations When training locally-connected layers with target propagation, one also needs to implement the inverse computation in order to train the feedback weights.

As in fully-connected layers, the forward computation implemented by both locally-connected and convolutional layers can be seen as a linear transformation y = W x + b, where the matrix W has a special, sparse structure (i.e., has a block of non-zero elements, and zero-elements elsewhere), and the dimensionality of y is not more than x.

The inverse operation requires computation of the form x = V y + c, where matrix V has a similar sparse structure as W T .

However, given this sparsity of V , computing the inverse of y using V would be highly inefficient BID6 .

We instead use an implementation trick often used in deconvolutional architectures.

First, we define a forward computation z = Ax, where z and A are dummy activities and weights.

We then define a transpose matrix as the gradient of this feedforward operation: DISPLAYFORM0 and thus DISPLAYFORM1 The gradient dz dx (and its multiplication with y) can be very quickly computed by the means of automatic differentiation in many popular deep learning frameworks.

Note that this is strictly an implementation detail and does not introduce any additional use of gradients or weight sharing in learning.

For DTP and SDTP we optimized over parameters of the model and inverse Adam optimizers, learning rate α used to compute targets for h L−1 in DTP, and the Gaussian noise magnitude σ used to train inverses.

For backprop we optimized only the model Adam optimizer parameters.

For all experiments the best hyperparameters were found by random searches over 60 random configurations drawn from the relevant ranges specified in table 4.

As we mention in section 2.1.1, in our implementation of TP algorithms we use denoising training of model inverses which we find more biologically motivated than noise-preserving training used by .

In particular, because downstream activity will always have noise applied to it (e.g., given that downstream neurons spike stochastically), one is always fundamentally in the denoising case in the brain.

We did not observe a significant empirical difference between these two methods in practice for either DTP and SDTP.

FIG6 shows the learning dynamics for parallel versions of DTP and SDTP with noise-preserving inverse losses to compare with figure 2c with denoising inverse loss.

One can see that the considered methods converge to roughly same train and test errors with similar speed.

<|TLDR|>

@highlight

Benchmarks for biologically plausible learning algorithms on complex datasets and architectures