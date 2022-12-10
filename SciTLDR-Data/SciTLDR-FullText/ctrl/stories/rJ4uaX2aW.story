A common way to speed up training of large convolutional networks is to add  computational units.

Training is then performed using data-parallel synchronous Stochastic Gradient Descent (SGD) with a mini-batch divided between computational units.

With an increase in the number of nodes, the batch size grows.

However,  training with a large batch  often results in lower model accuracy.

We argue that the current recipe for large batch training (linear learning rate scaling with warm-up) is not general enough and training may diverge.

To overcome these optimization difficulties, we propose a new training algorithm based on Layer-wise Adaptive Rate Scaling (LARS).

Using LARS, we scaled AlexNet  and ResNet-50 to a batch size of 16K.

Training of large Convolutional Neural Networks (CNN) takes a lot of time.

The brute-force way to speed up CNN training is to add more computational power (e.g. more GPU nodes) and train network using data-parallel Stochastic Gradient Descent, where each worker receives some chunk of global mini-batch (see e.g. BID10 or BID4 ).

The size of a chunk should be large enough to utilize the computational resources of the worker.

So scaling up the number of workers results in the increase of batch size.

But using large batch may negatively impact the model accuracy, as was observed in BID10 , BID14 , BID8 , BID6 .Increasing the global batch while keeping the same number of epochs means that you have fewer iterations to update weights.

The straight-forward way to compensate for a smaller number of iterations is to do larger steps by increasing the learning rate (LR).

For example, BID10 suggests to linearly scale up LR with batch size.

However using a larger LR makes optimization more difficult, and networks may diverge especially during the initial phase.

To overcome this difficulty, BID4 suggested a "learning rate warm-up": training starts with a small LR, which is slowly increased to the target "base" LR.

With a LR warm-up and a linear scaling rule, BID4 successfully trained ResNet-50 BID5 ] with batch B=8K, see also BID1 ].

Linear scaling of LR with a warm-up is the "state-of-the art" recipe for large batch training.

We tried to apply this linear scaling and warm-up scheme to train AlexNet BID11 ] on ImageNet BID3 ], but scaling stopped after B=2K since training diverged for large LR-s.

For B=4K the accuracy dropped from the baseline 57.6% (B=512) to 53.1%, and for B=8K the accuracy decreased to 44.8%.

To enable training with a large LR, we replaced Local Response Normalization layers in AlexNet with Batch Normalization (BN) BID7 ].

We will refer to this models AlexNet-BN.

BN improved model convergence for large LRs as well as accuracy: for B=8K the accuracy gap decreased from 14% to 2.2%.To analyze the training stability with large LRs we measured the ratio between the norm of the layer weights and norm of gradients update.

We observed that if this ratio is too high, the training becomes unstable.

On other hand, if the ratio is too small, then weights don't change fast enough.

The layer with largest ||∇W || ||W || defines the global limit on the learning rate.

Since this ratio varies a lot between different layers, we can speed-up training by using a separate LR for each layer.

Thus we propose a novel Layer-wise Adaptive Rate Scaling (LARS) algorithm.

There are two notable differences between LARS and other adaptive algorithms such as ADAM BID9 ) or RMSProp BID16 ): first, LARS uses a separate learning rate for each layer and not for each weight, which leads to better stability.

And second, the magnitude of the update is defined with respect to the weight norm for better control of training speed.

With LARS we trained AlexNet-BN and ResNet-50 with B=16K without accuracy loss.

The training of CNN is done using Stochastic Gradient (SG) based methods.

At each step t a minibatch of B samples x i is selected from the training set.

The gradients of loss function ∇L(x i , w) are computed for this subset, and networks weights w are updated based on this stochastic gradient: DISPLAYFORM0 The computation of SG can be done in parallel by N units, where each unit processes a chunk of the mini-batch with B N samples.

Increasing the mini-batch permits scaling to more nodes without reducing the workload on each unit.

However, it was observed that training with a large batch is difficult.

To maintain the network accuracy, it is necessary to carefully adjust training hyper-parameters (learning rate, momentum etc).

BID10 suggested the following rules for training with large batches: when you increase the batch B by k times, you should also increase LR by k times while keeping other hyper-parameters (momentum, weight decay, etc) unchanged.

The logic behind linear LR scaling is straight-forward: if you increase B by k times while keeping the number of epochs unchanged, you will do k times fewer steps.

So it seems natural to increase the step size by k times.

For example, let's take k = 2.

The weight updates for batch size B after 2 iterations would be: DISPLAYFORM1 The weight update for the batch B 2 = 2 * B with learning rate λ 2 : DISPLAYFORM2 will be similar if you take λ 2 = 2 * λ, assuming that ∇L(x j , w t+1 ) ≈ ∇L(x j , w t ) .Using the "linear LR scaling" BID10 trained AlexNet with batch B=1K with minor (≈ 1%) accuracy loss.

The scaling of AlexNet above 2K is difficult, since the training diverges for larger LRs.

It was observed that linear scaling works much better for networks with Batch Normalization (e.g. BID2 ).

For example BID0 trained the Inception model with batch B=6400, and Li (2017) trained ResNet-152 for B=5K.The main obstacle for scaling up batch is the instability of training with high LR.

BID6 tried to use less aggressive "square root scaling" of LR with special form of Batch Normalization ("Ghost Batch Normalization") to train AlexNet with B=8K, but still the accuracy (53.93%) was much worse than baseline 58%.

To overcome the instability during initial phase, BID4 proposed to use LR warm-up: training starts with small LR, and then LR is gradually increased to the target.

After the warm-up period (usually a few epochs), you switch to the regular LR policy ("multi-steps", polynomial decay etc).

Using LR warm-up and linear scaling BID4 trained ResNet-50 with batch B=8K without loss in accuracy.

These recipes constitute the current state-of-the-art for large batch training, and we used them as the starting point of our experiments.

Another problem related to large batch training is so called "generalization gap", observed by BID8 .

They came to conclusion that "the lack of generalization ability is due to the fact that large-batch methods tend to converge to sharp minimizers of the training function."

They tried a few methods to improve the generalization with data augmentation and warm-starting with small batch, but they did not find a working solution.

We used BVLC 1 AlexNet with batch B=512 as baseline.

Model was trained using SGD with momentum 0.9 with initial LR=0.02 and the polynomial (power=2) decay LR policy for 100 epochs.

The baseline accuracy is 58.8% (averaged over last 5 epochs).

Next we tried to train AlexNet with B=4K by using larger LR.

In our experiments we changed the base LR from 0.01 to 0.08, but training diverged with LR > 0.06 even with warm-up 2 .

The best accuracy for B=4K is 53.1%, achieved for LR=0.05.

For B=8K we couldn't scale-up LR either, and the best accuracy is 44.8% , achieved for LR=0.03 (see TAB0 (a) ).To stabilize the initial training phase we replaced Local Response Normalization layers with Batch Normalization (BN).

We will refer to this model as AlexNet-BN.3 .

AlexNet-BN model was trained using SGD with momentum=0.9, weight decay=0.0005 for 128 epochs.

We used polynomial (power 2) decay LR policy with base LR=0.02.

The baseline accuracy for B=512 is 60.2%.

With BN we could use large LR-s even without warm-up.

For B=4K the best accuracy 58.9% was achieved for LR=0.18, and for B=8K the best accuracy 58% was achieved for LR=0.3.

We also observed that BN significantly widens the range of LRs with good accuracy.

Still there is a 2.2% accuracy loss for B=8K.

To check if it is related to the "generalization gap" (Keskar et al. FORMULA0 ), we looked at the loss gap between training and testing (see FIG0 ).

We did not find the significant difference in the loss gap between B=512 and B=8K.

We conclude that in this case the accuracy loss was mostly caused by the slow training and was not related to a generalization gap.

The standard SGD uses the same LR λ for all layers: w t+1 = w t − λ∇L(w t ).

When λ is large, the update ||λ * ∇L(w t )|| can become larger than ||w||, and this can cause the divergence.

This makes the initial phase of training highly sensitive to the weight initialization and to initial LR.

We found that the ratio of the L2-norm of weights and gradients ||w||/||∇L(w t )|| varies significantly between weights and biases, and between different layers.

For example, let's take AlexNet after one iteration TAB1 , "*.w" means layer weights, and "*.b" -biases).

The ratio ||w||/||∇L(w)|| for the 1st convolutional layer ("conv1.w") is 5.76, and for the last fully connected layer ("fc6.w") -1345.

The ratio is high during the initial phase, and it is rapidly decreasing after few epochs (see Figure 2 ).

If LR is large comparing to the ratio for some layer, then training may becomes unstable.

The LR "warm-up" attempts to overcome this difficulty by starting from small LR, which can be safely used for all layers, and then slowly increasing it until weights will grow up enough to use larger LRs.

We would like to use different approach.

We want to make sure that weights update is small comparing to the norm of weights to stabilize training DISPLAYFORM0 where η < 1 control the magnitude of update with respect to weights.

The coefficient η defines how much we "trust" that the value of stochastic gradient ∇L(w l t ) is close to true gradient.

The η depends on the batch size.

"Trust" η is monotonically increasing with batch size: for example for Alexnet for batch B = 1K the optimal η = 0.0002, for batch B = 4K -η = 0.005, and for B = 8K -η = 0.008.

We implemented this idea through defining local LR λ l for each layer l: DISPLAYFORM1 where γ defines a global LR policy (e.g. steps, or exponential decay), and local LR λ l is defined for each layer through "trust" coefficient η < 1 4 : DISPLAYFORM2 Note that now the magnitude of the update for each layer doesn't depend on the magnitude of the gradient anymore, so it helps to partially eliminate vanishing and exploding gradient problems.

The network training for SGD with LARS are summarized in the Algorithm 1 5 .LARS was designed to solve the optimization difficulties, and it does not replace standard regularization methods (weight decay, batch norm, or data augmentation).

But we found that with LARS we can use larger weight decay, since LARS automatically controls the norm of layer weights: DISPLAYFORM3 where B is min-batch size, N -number of training epochs, and S -number of samples in the training set.

Here we assumed that global rate policy starts from 1 and decrease during training over training interval [0, N * S/B].

BID4 and BID1 we used the second setup with an extended augmentation with variable image scale and aspect ratio similar to ] .

The baseline top-1 accuracy for this setup is 75.4%.

The accuracy with B=16K is 0.7-1.4% less than baseline.

This gap is related to smaller number of steps.

We will show in the next section that one can recover the accuracy by training for more epochs.

When batch becomes large (32K), even models trained with LARS and large LR don't reach the baseline accuracy.

One way to recover the lost accuracy is to train longer (see BID6 ).

Note that when batch becomes large, the number of iteration decrease.

So one way to try to improve the accuracy, would be train longer.

For example for Alexnet and Alexnet-BN with B=16K, when we double the number of iterations from 7800 (100 epochs) to 15600 (200 epochs) the accuracy improved by 2-3% (see TAB4 ).

The same effect we observed for Resnet-50: training for additional 100 epochs recovered the top-1 accuracy to the baseline 75.5%.

In general we found that we have to increase the training duration to keep the accuracy.

Consider for example Googlenet ].

As a baseline we trained BVLC googlenet 8 with batch=256 for 100 epoch.

The top-1 accuracy of this model is 69.2%.

Googlenet is deep, so in original paper authors used auxiliary losses to accelerate SGD.

We used LARS to solve optimization difficulties so we don't need these auxiliary losses.

The original model also has no Batch Normalization, so we used data augmentation for better regularization.

The baseline accuracy for B=256 is 70.3% with extended augmentation and LARS.

We found that Googlenet is very difficult to train with large batch even with LARS: we needed both large number of epoch and longer ramp-up to scale learning rate up (see TAB5 ).

Large batch is a key for scaling up training of convolutional networks.

The existing approach for large-batch training, based on using large learning rates, leads to divergence, especially during the initial phase, even with learning rate warm-up.

To solve these difficulties we proposed the new optimization algorithm, which adapts the learning rate for each layer (LARS) proportional to the ratio between the norm of weights and norm of gradients.

With LARS the magnitude of the update for each layer doesn't depend on the magnitude of the gradient anymore, so it helps with vanishing and exploding gradients.

But even with LARS and warm-up we couldn't increase LR farther for very large batches, and to keep the accuracy we have to increase the number of epochs and use extensive data augmentation to prevent over-fitting.

<|TLDR|>

@highlight

A new large batch training algorithm  based on Layer-wise Adaptive Rate Scaling (LARS); using LARS, we scaled AlexNet  and ResNet-50 to a batch of 16K.