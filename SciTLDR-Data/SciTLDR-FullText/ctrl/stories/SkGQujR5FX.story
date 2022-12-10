Distributed computing can significantly reduce the training time of neural networks.

Despite its potential, however, distributed training has not been widely adopted: scaling the training process is difficult, and existing SGD methods require substantial tuning of hyperparameters and learning schedules to achieve sufficient accuracy when increasing the number of workers.

In practice, such tuning can be prohibitively expensive given the huge number of potential hyperparameter configurations and the effort required to test each one.

We propose DANA, a novel approach that scales out-of-the-box to large clusters using the same hyperparameters and learning schedule optimized for training on a single worker, while maintaining similar final accuracy without additional overhead.

DANA estimates the future value of model parameters by adapting Nesterov Accelerated Gradient to a distributed setting, and so mitigates the effect of gradient staleness, one of the main difficulties in scaling SGD to more workers.



Evaluation on three state-of-the-art network architectures and three datasets shows that DANA scales as well as or better than existing work without having to tune any hyperparameters or tweak the learning schedule.

For example, DANA achieves 75.73% accuracy on ImageNet when training ResNet-50 with 16 workers, similar to the non-distributed baseline.

Modern deep neural networks are comprised of millions of parameters, which require massive amounts of data and time to learn.

Steady growth of these networks over the years has made it impractical to train them from scratch on a single GPU.

Distributing the computations over several GPUs can drastically reduce this training time.

Unfortunately, stochastic gradient descent (SGD), typically used to train these networks, is an inherently sequential algorithm.

As a result, training deep neural networks on multiple workers (computational devices) is difficult, especially when trying to maintain high efficiency, scalability and final accuracy.

Data Parallelism is a common practice for distributing computation: data is split across multiple workers and each worker computes over its own data.

Synchronous SGD (SSGD) is the most straightforward method to distribute the training process of neural networks: each worker computes the gradients over its own separate mini-batches, which are then aggregated to update a single model.

The result is identical to multiplying the batch size B by the number of workers N , so the effective batch size is B · N .

This severely limits scalability and reduces the model accuracy if not handled carefully BID25 BID6 BID8 .

Furthermore, synchronization limits SSGD progress to the slowest worker: all workers must finish their current mini-batch and update the parameter server before any can proceed to the next mini-batch.

Asynchronous SGD (ASGD) addresses these drawbacks by removing synchronization between the workers BID5 .

Unfortunately, it suffers from gradient staleness: gradients sent by workers are often based on parameters that are older than the master's (parameter server) current parameters.

Hence, distributed ASGD suffers from slow convergence and reduced final accuracy, and may not converge at all if the number of workers is high BID34 .

Several works attempt to address these issues BID35 BID33 BID34 BID5 , but none has managed to overcome these problems when scaling to a large number of workers.

More crucially, many ASGD algorithms require re-tuning of hyperparameters when scaling to different numbers of workers, and several even introduce new hyperparameters that must also be tuned BID35 BID33 BID34 .

In practice, the vast number of potential hyperparameter configurations means that tuning is often done in parallel, with each worker independently evaluating a single configuration using standard SGD.

Once the optimal hyperparameters are selected, training is completed on larger clusters of workers.

Any additional tuning for ASGD can thus be computationally expensive and time-consuming.

Though many algorithms have been proposed to reduce the cost of tuning BID2 BID13 BID12 BID9 BID26 , hyperparameter search remains a significant obstacle, and many practitioners cannot afford to re-tune hyperparameters for distributed training.

Our contribution: We propose Distributed Accelerated Nesterov ASGD (DANA), a new distributed ASGD algorithm that works out of the box: it achieves state-of-the-art accuracy on existing architectures without any additional hyperparameter tuning or changes to the training schedule, while scaling as well or better than existing ASGD approaches, and without any additional overhead.

Our DANA implementation achieves state-of-the-art accuracy on ImageNet when training ResNet-50 with 16 and even 32 workers, as well as on CIFAR-10 and CIFAR-100.

The goal of SGD is to minimize an optimization problem J(θ) where J is the objective function (i.e., loss) and the vector θ ∈ R k is the model's parameters.

Let ∇J be the gradient of J with respect to its argument θ.

Then the update rule of SGD for the given problem with learning rate η is: DISPLAYFORM0 Momentum Momentum BID22 has been demonstrated to accelerate SGD convergence and reduce oscillation BID27 .

Momentum can be compared to a heavy ball rolling downhill that accumulates speed on its way towards the minima.

Mathematically, the momentum update rule is obtained by adding a fraction γ of the previous update vector v t−1 to the current update vector v t : DISPLAYFORM1 When successive gradients have similar direction, momentum results in larger update steps (higher speed), yielding up to quadratic speedup in convergence rate for stochastic and standard gradient descent BID19 a) .

Receive parameters θ t from master Compute gradients g t ← ∇J(θ t ) Send g t to master Algorithm 2 ASGD: master Receive gradients g t from worker i (at iteration t + τ ) θ t+τ +1 ← θ t+τ − ηg t Send parameters θ t+τ +1 to worker i Nesterov Continuing the analogy of a heavy ball rolling downhill, higher speed might make the heavy ball overshoot the bottom of the valley (the local or global minima) if it does not slow down in time.

BID20 proposed Nesterov Accelerated Gradient (NAG), which gives the ball a "sense" of where it is going, allowing it to slow down in advance.

Formally, NAG approximateŝ θ t , the future value of θ t , using the previous update vector v t :θ t = θ t − ηγv t , and computes the gradients on the parameters' approximated future valueθ instead of their current value θ.

This allows NAG to slow down near the minima before overshooting the goal and climbing back up the hill.

We call this look-ahead since it allows us a peek at θ's future position.

The NAG update rule is identical to Equation 2, except that the gradient g t is computed on the approximated future parametersθ t instead of θ t : g t = ∇J(θ t ).

It is then applied to the original parameters θ t via v t as in Equation 2.

Equation 3 shows that the difference between the updated parameters θ t+1 and the approximated future positionθ t is only affected by the newly computed gradients g t , and not by v t .

Hence, NAG can accurately estimate future gradients even when the update vector v t is large.

DISPLAYFORM0 3 GRADIENT STALENESS AND MOMENTUMIn ASGD training, each worker i pulls up-to-date parameters θ t from the master and computes gradients of a single batch (Algorithm 1).

Once computation has finished, worker i sends the computed gradient g t back to the master.

The master (Algorithm 2) then applies the gradient g t to its current set of parameters θ t+τ , where τ is the lag: the number of updates the master has received from other workers while worker i was computing g t .In other words, gradient g t is stale: it was computed from parameters θ t but applied to θ t+τ .

This gradient staleness is major obstacle to scaling ASGD: the lag τ increases as the number of workers N grows, decreasing gradient accuracy, and ultimately reducing the accuracy of the trained model.

We denote by ∆ θ = θ t − θ t+τ the difference between the master and worker parameters, and define the gap as the sum of layer-wise RMSE: G(∆ θ ) = ψ∈layers RMSE(ψ), where for each model layer ψ with m parameters, RMSE(ψ) = ψ / √ m. Ideally, there should be no difference between θ t and θ t+τ : when ∆ θ = 0, gradients are computed on the same parameters they will be applied to.

This is the case for sequential and synchronous methods such as SGD and SSGD.

In asynchronous methods, however, more workers result in an increased lag τ and thus a larger gap, as demonstrated by Figure 1(a) .

A larger gap means less accurate gradients, since they are computed on parameters that differ significantly from those they will be applied to.

Conversely, a smaller gap means that gradients are likely to be more accurate.

The Effect of Momentum While momentum and Nesterov methods improve SGD convergence and accuracy of trained models, they make scaling to more workers more difficult.

As Figure 1 (b) shows, adding NAG to ASGD exacerbates gradient staleness, even though the lag τ is unchanged.

Put differently, NAG and momentum increase the gap G(∆ θ ).

Let x i be the variable x for worker i (for the master, i = 0) and x i t be the value of that variable at the worker's t iteration.

For ASGD without momentum or NAG, ∆ θ is the sum of gradients 1 , ∆ DISPLAYFORM0 whereas in the case of ASGD with NAG, ∆ θ is the sum of update vectors: Figure 1: The gap between θ t and θ t+τ while training ResNet-20 on the CIFAR-10 dataset with (a) different numbers of workers, and (b) different asynchronous algorithms.

Adding workers or using momentum increases the effect of the lag τ on the gap.

The large drops in G(θ t − θ t+τ ) are caused by learning rate decay.

DISPLAYFORM1 by design, momentum and NAG increase the magnitude of updates to θ t : v t ≥ g t .

Moreover, if the distribution of training data in each worker is the same (e.g., the common case of assigning data to workers uniformly at random), then the directions of updates v i t are approximately similar, since the loss surfaces are similar.

Applying the identity a + b 2 = a 2 + b 2 + 2 a, b and the triangle inequality, it follows that in general the gap with NAG is larger than the gap without it: DISPLAYFORM2 Figure 1(b) shows that the gap for ASGD with NAG is substantially larger than for ASGD without it.

Conversely, DANA-Zero, detailed in the next section, maintains a low gap throughout training even though it also uses NAG.

DANA is a distributed optimizer that converges without hyperparameter tuning even when training with momentum on large clusters.

It reduces the gap G(∆ θ ) by computing the worker's gradients on parameters that more closely resemble the master's future position θ t+τ .

We extend NAG to the common distributed setting with N workers and one master, obtaining similar look-ahead to the traditional method with a single worker.

This means that for the same lag τ , DANA suffers from a reduced gap and therefore suffers less from gradient staleness.

In DANA-Zero, the master maintains a separate update vector v i for each worker, which is updated with the worker's gradients g i using the same update rule as in classic SGD with momentum (Equation 2).

Since the master updates each v i only with the gradients from worker i, we can apply look-ahead using the most recent update vectors of the other workers.

We know that v i t−1 will move the master's parameters θ 0 on iteration t of worker i by ηγv DISPLAYFORM0 gives us an approximation of the next position of the master's parameters after worker i has sent its gradients.

Instead of sending the master's current parameters θ 0 to the worker, DANA-Zero sends the estimated future position of the master's parameters after N updates, one for each worker: DISPLAYFORM1 where prev(i) denotes the last iteration where worker i sent gradients to the master.

Algorithm 3 shows the DANA-Zero master algorithm; the worker code is the same as in ASGD (Algorithm 1).Given the update rule, we calculate the gap of DANA-Zero, G(∆ DANA θ ), similarly to Equation 3: DISPLAYFORM2 Algorithm 3 DANA-Zero master.

DANA-Zero uses the standard ASGD worker (Algorithm 1).Receive gradients g i from worker i Update worker's momentum DISPLAYFORM3 Algorithm 4 DANA worker i. DANA uses the standard ASGD master (Algorithm 2).Receive parameters DISPLAYFORM4 Equation FORMULA8 shows that DANA-Zero has the same gap as ASGD without momentum.

Figure 1 (b) demonstrates this empirically: ASGD with momentum has a larger gap than ASGD throughout the training process, whereas DANA-Zero's gap is similar to ASGD despite also using momentum.

Additionally, when running with one worker (N = 1), DANA-Zero reduces to a single standard NAG optimizer: with one worker, θ 1 t = θ 0 t − ηγ, so merging the master and the worker algorithms yields the Nesterov update rule (see Appendix A for more details).

In DANA-Zero, the master maintains an update vector for every worker, and must also computeθ at each iteration.

This adds a computation and memory overhead to the master.

DANA is a variation of DANA-Zero that obtains the same look-ahead as DANA-Zero but without any additional memory or computation overhead.

BID1 proposed a simplified Nesterov update rule, known as Bengio-Nesterov Momentum.

This variation of the classic Nesterov is occasionally used in deep learning frameworks BID21 since it simplifies the implementation.

BengioNesterov Momentum works by defining a new variable Θ to represent θ after the momentum update:

Substituting θ t with Θ t in the NAG update rule (Section 2) yields the Bengio-Nesterov update rule: DISPLAYFORM0 Using Equation 7, an implementation need only store one set of parameters in memory (Θ) since gradients are both computed from and applied to Θ, rather than computed onθ but applied to θ.

The DANA Update Rule We leverage the ideas of Bengio-Nesterov Momentum to optimize DANA.

As we did in Equation 6, we define a new variable Θ that represents θ after the momentum update from all workers: DISPLAYFORM1 We define Θ t+1 as Θ t after applying worker i's update v t+1 = γv t + ∇J(Θ t ): DISPLAYFORM2 Substituting θ t with Θ t yields the DANA update rule (Equation 9): DISPLAYFORM3 Algorithm 4 shows DANA: a variation of DANA-Zero that uses Bengio-Nesterov to eliminate the overhead at the master.

DANA only changes the worker side and uses the same master algorithm as in ASGD (Algorithm 2); hence, it eliminates any additional overhead at the master.

DANA is equivalent to DANA-Zero in all other ways, and provides the same benefits: it works out-of-the-box, provides look-ahead to reduce the gap 2 and achieves the same fast convergence and high accuracy.

We implemented DANA using PyTorch BID21 and mpi4py BID4 and evaluated it by: (a) simulating multiple distributed workers 3 on a single machine to focus on accuracy rather than communication overheads and update scheduling; and (b) running the distributed algorithm on multiple machines, where we measure run time speedups and confirm simulation accuracy.

We simulate two modes.

In block-random scheduling every block of N updates contains one update from each worker and order is shuffled between blocks, which simulates the common case where distributed workers have very similar computational power.

In the gamma-distributed model the execution time for each individual batch is drawn from a gamma distribution BID0 .

The gamma distribution is a well-accepted model for task execution time, and gives rise to stragglers naturally.

We use the formulation proposed by BID0 and set V = 0.1 and µ = B * V 2 , where B is the chosen batch size, yielding a mean execution time of B simulated time units.

Our main evaluation metric is final test error: the error achieved by a trained model after training using the baseline training schedule.

We also measure improvement in training time (speedup) using the distributed DANA implementation.

Algorithms As we are interested in out-of-the-box performance, we compare DANA to algorithms that do not introduce new parameters and require no re-tuning (see TAB0 for comparison to non-OOTB methods).

All runs use the same hyperparameters, training schedule and data augmentation from the original paper where the network architectures are proposed.1.

Baseline: single worker with the same hyperparameters as in the respective NN paper.

2. SSGD: similar to BID8 with the linear scaling rule.

3. ASGD: standard asynchronous SGD without momentum (momentum parameter set to 0).

4. NAG-ASGD: asynchronous SGD which uses a single NAG optimizer for all workers.

5. Multi-ASGD: asynchronous SGD which holds a separate NAG optimizer for each worker.

6. DANA: DANA as described in Section 4.2.In the early stages of training, the network changes rapidly, which can lead to training error spikes.

For all algorithms, we follow the gradual warm-up approach proposed by BID8 to overcome this problem: we divide the initial training rate by the number of workers N and ramp it up linearly until it reaches its original value after 5 epochs.

We also use momentum correction BID8 in all algorithms to stabilize training when the learning rate changes.

Datasets We evaluated DANA on CIFAR-10, CIFAR-100 (Hinton, 2007) and ImageNet BID24 .

The CIFAR-10 Hinton (2007) dataset is comprised of 60k RGB images partitioned into 50k training images and 10k test images.

Each image contains 32x32 RGB pixels and belongs to one of ten equal-sized classes.

CIFAR-100 is similar but has 100 classes.

The ImageNet dataset BID24 , known as ILSVRC2012, consists of RGB images, each labeled as one of 1000 classes.

Images are partitioned to 1.28 million training images and 50k validation images, and each image is randomly cropped and re-sized to 224x224 (1-crop validation).

Out-of-the-box, DANA's final test error remains similar to the baseline error with up to 24 workers in Figure 2 (a) and 12 workers in Figures 2(b) and 2(c).

Moreover, DANA's final error is lower than the other algorithms when using up to 24-32 workers -all without any tuning.

Above that point, DANA is no longer the superior algorithm because of the smaller size of CIFAR-10 and CIFAR-100: with so many workers the amount of data per worker is so small that gradients from different workers become dissimilar, and DANA is no longer able to mitigate the effects of momentum.

ImageNet results TAB2 show that DANA easily scales to 32 workers when there is enough data per worker.

NAG-ASGD demonstrates the detrimental effect of momentum on gradient staleness: it yields good accuracy with few workers, but test error climbs sharply and sometimes even fails to converge when used with more than 16 workers.

On the other hand, even though ASGD without NAG appears to be the most scalable algorithm, its test error is unacceptably high even with 2 workers.

While SSGD appears to offer a middle ground of reasonable accuracy with good scalability, in practice speedup is limited by synchronization and the increase in effective batch size means tuning is required to achieve good accuracy.

DANA provides a way out of this dilemma: by mitigating gradient staleness, it achieves the best final accuracy while scaling to many workers, and works without changing any hyperparameter or changing the learning schedule.

Finally, Multi-ASGD serves as an ablation study: its poor scalability demonstrates that it is not sufficient to simply maintain update vectors for every worker.

The DANA update rules (Section 4) are also required to achieve a high test accuracy.

TAB2 lists out-of-the-box test errors when training the ResNet-50 architecture BID10 on ImageNet.

Due to the long training time of ImageNet, we only conducted experiments on ImageNet with SSGD, ASGD and DANA.

DANA consistently outperforms all other out-of-the-box algorithms.

Similar test-errors to TAB2 were achieved when training DANA with the gammadistributed model on 32 and 64 workers, yielding a final test-error of +0.54% and +5.84% respectively.

TAB0 compares DANA to reported results from state-of-the-art asynchronous algorithms that rely on tuning or changes to the learning rate schedule, while DANA converges to the ImageNet's baseline test accuracy with 16 and 32 workers, matching or exceeding recent state-of-the-art algorithms (AD-PSGD and DC-ASGD), despite making no changes to any hyperparameter.

While this work focuses on improving out-of-the-box ASGD accuracy without adding overhead, we also measured speedup, defined as the runtime for DANA with N workers divided by the runtime for the single worker baseline.

FIG2 shows the speedup and final test error when running DANA on the Google Cloud Platform with a single parameter server (master) and one Nvidia Tesla V100 GPU per machine, when training ResNet-20 on the CIFAR-10 dataset.

It shows speedup of up to ×16 when training with N = 24 workers, and as before, its final test error remains close to the baseline up to N = 24 workers.

At 24 workers, the parameter server becomes a bottleneck.

This phenomenon is consistent with literature BID29 on ASGD, and is well-studied.

Since the DANA master is unchanged from the ASGD algorithm (Algorithm 2), existing techniques, such as sharding the parameter server BID5 , improving network utilization BID14 , lock-free synchronizations BID23 BID32 , and gradient compression BID17 BID28 BID3 , are fully compatible with DANA but are beyond the scope of this work.

we increase the number of workers, since it must wait after each iteration until all workers complete their batch.

FIG4 (b) shows that DANA (or any ASGD variant) is up to 21% faster than SSGD.

This speedup is an underestimate, since our simulation only includes batch execution times, and does not model execution time of barriers, all-gather operations, and other overheads.

DANA achieves out-of-the-box scaling by explicitly mitigating the effects of gradient staleness.

Other approaches to mitigating staleness include DC-ASGD BID35 , which uses a Taylor expansion to approximate the gradients as if they were calculated on the master's recent parameters.

DC-ASGD requires substantial tuning of several hyperparameters, introduces additional hyperparameters that must also be tuned, and requires additional computation at the master to approximate the Hessian.

Elastic Averaging SGD (EASGD) BID33 is an ASGD algorithm that uses a center force to pull the workers' parameters towards the master's parameters.

This allows each worker to train asynchronously and synchronize with the master once every few batches.

However, EASGD introduces three new hyperparameters that must be tuned.

BID34 proposed Staleness-aware ASGD: worker gradients are weighted by the lag between two successive updates, so stale gradients have lower impact.

This method adds one new hyperparameter, and achieves lower or equivalent final accuracy compared to SSGD.

DANA scales without adding hyperparameters or tuning, and achieves final accuracy comparable to that of a single worker.

Other approaches to scaling are SSGD learning rate schedulers.

BID8 introduced a linear scaling rule and warmup epochs to help increase the mini-batch size, which is key to scaling the number of workers in a synchronous environment.

BID30 further generalize that work and introduce LARS, a method that changes the learning rate independently for each layer, according to the ratio between norm of the layer's weights and the norm of the layer's current gradient, whose parameters need to be tuned.

BID25 suggest increasing the batch size instead of decaying the learning rate.

These approaches are compatible with (and indeed orthogonal to) DANA.Finally, decentralized approaches to scaling SGD eliminate the parameter server entirely.

In D-PSGD BID15 , workers first compute and apply gradients locally and then synchronously average models with their neighbors.

Very recently, BID16 proposed AD-PSGD, which operates asynchronously.

While they demonstrate impressive scaling, these works focus on different communication topologies and use other learning schedules and batch sizes than the baselines.

DANA is a new asynchronous SGD algorithm for training of neural networks.

By mitigating the effect of gradient staleness, DANA scales out-of-the-box to large clusters using the same hyperparameters and learning schedule optimized for training on a single worker, while maintaining similar final accuracy, without adding any overhead at the master.

DANA could be used to extend other non-distributed optimization procedures (e.g., Nadam BID7 ) to a distributed setting without adding parameters.

Integrating DANA with DC-ASGD could further mitigate gradient staleness, though without eliminating tuning.

Finally, we are working to extend DANA with separate, selfadjusting weights per worker to address settings with heterogeneous workers while avoiding tuning.

A.1 DANA-ZERO EQUIVALENCE TO NESTEROV When running with one worker (N = 1) DANA-Zero reduces to a single NAG optimizer.

This can be shown by merging the worker and master (Algorithms 1 and 3) into a single algorithm: since at all times θ 1 t = θ 0 t − ηγv t−1 , the resulting algorithm trains one set of parameters θ, which is exactly the Nesterov update rule.

Algorithm 5 shows the fused algorithm, equivalent to standard NAG optimizer.

Algorithm 5 Fused DANA-Zero worker/master (when N = 1)Compute gradients g t ← ∇J(θ t − ηγv t−1 ) Update momentum v t ← γv t−1 + g t Update weights θ t+1 ← θ t − ηv t

This section shows the results of the ResNet-20 BID10 and Wide ResNet 16-4 (16 depth and 4 width) network architectures on the CIFAR-10 and CIFAR-100 datasets.

We ran each experiment five times to show the mean and standard deviation of the final test error, which are shown in the tables below.

The experiments were executed using two types of worker update orders:• Round Robin: Every worker updates the master in a sequential order.

For example, if N = 4, the order of updates is 1, 2, 3, 4, 1, 2, 3, 4 . . . .

• Block Random: Every worker updates the master in a random order.

However, every N updates, it is guaranteed that every worker has updated the master exactly once.• Gamma Distribution: Every worker updates the master in a gamma distribution order BID0 .

The gamma distribution is a well-accepted model for task execution time, and gives rise to stragglers naturally.

We use the formulation proposed by BID0 and set V = 0.1 and µ = B * V 2 , where B is the chosen batch size.

When the batch size is 128 (as in both CIFAR datasets for example) this yields the distribution Γ(100, 1.28) with mean execution time of 128 simulated time units.

Tables 3 and 4 show the final test error of the ResNet-20 architecture whose training schedule BID10 starts with an initial learning rate of 0.1, which decays by a factor of ten on epochs 80 and 120.

The batch size is 128, momentum is 0.9, and the baseline uses NAG.

BID31 starts with an initial learning rate of 0.1, which decays by a factor of five on epochs 60, 120 and 160.

The batch size is 128, momentum is 0.9, and the baseline uses NAG.

<|TLDR|>

@highlight

A new distributed asynchronous SGD algorithm that achieves state-of-the-art accuracy on existing architectures without any additional tuning or overhead.

@highlight

Proposes an improvement to existing ASGD approaches at mid-size scaling using momentum with SGD for asynchronous training across a distributed worker pool.

@highlight

This paper addresses the gradient staleness vs parallel performance problem in distributed deep learning training, and proposes an approach to estimate future model parameters at the slaves to reduce communication latency effects.