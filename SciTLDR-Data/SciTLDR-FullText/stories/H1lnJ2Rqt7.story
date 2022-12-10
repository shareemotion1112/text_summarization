Stochastic Gradient Descent (SGD) methods using randomly selected batches are widely-used to train neural network (NN) models.

Performing design exploration to find the best NN for a particular task often requires extensive training with different models on a large dataset,  which is very computationally expensive.

The most straightforward method to accelerate this computation is to distribute the batch of SGD over multiple processors.

However, large batch training often times leads to degradation in accuracy, poor generalization, and even poor robustness to adversarial attacks.

Existing solutions for large batch training either do not work or require massive hyper-parameter tuning.

To address this issue, we propose a novel large batch training method which combines recent results in adversarial training (to regularize against ``sharp minima'') and second order optimization (to use curvature information to change batch size adaptively during training).

We extensively evaluate our method on Cifar-10/100, SVHN, TinyImageNet, and ImageNet datasets, using multiple NNs, including residual networks as well as compressed networks such as SqueezeNext.

Our new approach exceeds the performance of the existing solutions in terms of both accuracy and the number of SGD iterations (up to 1\% and $3\times$, respectively).

We emphasize that this is achieved without any additional hyper-parameter tuning to tailor our method to any of these experiments.

Finding the right NN architecture for a particular application requires extensive hyper-parameter tuning and architecture search, often times on a very large dataset.

The delays associated with training NNs is often the main bottleneck in the design process.

One of the ways to address this issue to use large distributed processor clusters; however, to efficiently utilize each processor, the portion of the batch associated with each processor (sometimes called the mini-batch) must grow correspondingly.

In the ideal case, the hope is to decrease the computational time proportional to the increase in batch size, without any drop in generalization quality.

However, large batch training has a number of well known draw backs.

These include degradation of accuracy, poor generalization, and poor robustness to adversarial perturbations BID17 BID36 .In order to address these drawbacks, many solutions have been proposed BID14 BID37 BID7 BID29 BID16 .

However, these methods either work only for particular models on a particular dataset, or they require massive hyperparameter tuning, which is often times not discussed in the presentation of results.

Note that while extensive hyper-parameter turning may result in good result tables, it is antithetical to the original motivation of using large batch sizes to reduce training time.

One solution to reduce the brittleness of SGD to hyper-parameter tuning is to use second-order methods.

Full Newton method with line search is parameter-free, and it does not require a learning rate.

This is achieved by using a second-order Taylor series approximation to the loss function, instead of a first-order one as in SGD, to obtain curvature information.

BID25 ; BID34 BID2 show that Newton/quasi-Newton methods outperform SGD for training NNs.

However, their re-sults only consider simple fully connected NNs and auto-encoders.

A problem with second-order methods is that they can exacerbate the large batch problem, as by construction they have a higher tendency to get attracted to local minima as compared to SGD.

For these reasons, early attempts at using second-order methods for training convolutional NNs have so far not been successful.

Ideally, if we could find a regularization scheme to avoid local/bad minima during training, this could resolve many of these issues.

In the seminal works of El Ghaoui & BID9 ; BID33 , a very interesting connection was made between robust optimization and regularization.

It was shown that the solution to a robust optimization problem for least squares is the same as the solution of a Tikhonov regularized problem BID9 .

This was also extended to the Lasso problem in BID33 .

Adversarial learning/training methods, which are a special case of robust optimization methods, are usually described as a min-max optimization procedure to make the model more robust.

Recent studies with NNs have empirically found that robust optimization usually converges to points in the optimization landscape that are flatter and are more robust to adversarial perturbation BID36 .Inspired by these results, we explore whether second order information regularized by robust optimization can be used to do large batch size training of NNs.

We show that both classes of methods have properties that can be exploited in the context of large batch training to help reduce the brittleness of SGD with large batch size training, thereby leading to significantly improved results.

In more detail, we propose an adaptive batch size method based on curvature information extracted from the Hessian, combined with a robust optimization method.

The latter helps regularize against sharp minima, especially during early stages of training.

We show that this combination leads to superior testing performance, as compared to the proposed methods for large batch size training.

Furthermore, in addition to achieving better testing performance, we show that the total number of SGD updates of our method is significantly lower than state-of-the-art methods for large batch size training.

We achieve these results without any additional hyper-parameter tuning of our algorithm (which would, of course, have helped us to tailor our solution to these experiments).

Here is a more detailed itemization of the main contributions of this work:• We propose an Adaptive Batch Size method for SGD training that is based on second order information, computed by backpropagating the Hessian operator.

Our method automatically changes the batch size and learning rate based on Hessian information.

We state and prove a result that this method is convergent for a convex problem.

More importantly, we empirically test the algorithm for important non-convex problems in deep learning and show that it achieves equal or better test performance, as compared to small batch SGD (We refer to this method as ABS).• We propose a regularization method using robust training by solving a min-max optimization problem.

We combine the second order adaptive batch size method with recent results of BID36 , which show that robust training can be used to regularize against sharp minima.

We show that this combination of Hessian-based adaptive batch size and robust optimization achieves significantly better test performance with little computational overhead (we refer to this Adaptive Batch Size Adversarial method as ABSA).• We test the proposed strategies extensively on a wide range of datasets (Cifar-10/100, SVHN, TinyImageNet, and ImageNet), using different NNs, including residual networks.

Importantly, we use the same hyper-parameters for all of the experiments, and we do not perform any kind of tuning of our hyper-parameters to tailor our results.

The empirical results show the clear benefit of our proposed method, as compared to the state-of-the-art.

The proposed algorithm achieves equal or better test accuracy (up to 1%) and requires significantly fewer SGD updates (up to 5×).• We empirically show that we can use a block approximation of the Hessian operator (i.e. the Hessian of the last fewer layers) to reduce the computational overhead of backpropagating the second order information.

This approximation is especially effective for deep NNs.

While a number of recent works have discussed adaptive batch size or increasing batch size during training BID7 BID29 BID10 BID1 , to the best of our knowledge this is the first paper to introduce Hessian information and adversarial training in adaptive batch size training, with extensive testing on many datasets.

We believe that it is important for every work to state its limitations (in general, but in particular in this area).

We were particularly careful to perform extensive experiments and repeated all the reported tests multiple times.

We test the algorithm on models ranging from a few layers to hundreds of layers, including residual networks as well as smaller networks such as SqueezeNext.

An important limitation is that second order methods have additional overhead for backpropagating the Hessian.

Currently, most of the existing frameworks do not support (memory) efficient backpropagation of the Hessian (thus providing a structural bias against these powerful methods).

However, the complexity of each Hessian matvec is the same as a gradient computation BID21 .

Our method requires Hessian spectrum, which typically needs ten Hessian matvecs (for power method iterations to reach a tolerance of 1e-2).

Thus, the benefits that we show in terms of testing accuracy and reduced number of updates do come at a cost (see Table 3 for details).

We measure this additional overhead and report it in terms of wall clock time.

Furthermore, we (empirically) show that this power iteration needs to be done only at the end of every epoch, thus significantly reducing the additional overhead.

Another limitation is that our theory only holds for convex problems (under certain smoothness assumptions).

Proving convergence for non-convex setting requires more involved analysis.

Recently, BID32 has provided interesting theoretical guarantees for AdaGrad in the non-convex setting.

Exploring a similar direction for our method is of interest for future work.

Another point is that adaptive batch size, prevents one from utilizing all of the processes, as compared to using large batch throughout the training.

However, a large data center can handle and accommodate a growing number of requests for processor resources, which could alleviate this.

Optimization methods based on SGD are currently the most effective techniques for training NNs, and this is commonly attributed to SGD's ability to escape saddle-points and "bad" local minima BID5 .The sequential nature of weight updates in synchronous SGD limits possibilities for parallel computing.

In recent years, there has been considerable effort on breaking this sequential nature, through asynchronous methods (Zhang et al., 2015) or symbolic execution techniques BID20 .

A main problem with asynchronous methods is reproducibility, which, in this case, depends on the number of processes used (Zheng et al., 2016; BID0 .

Due to this issue, recently there have been attempts to increase parallelization opportunities in synchronous SGD by using large batch size training.

With large batches, it is possible to distribute more efficiently the computations to parallel compute nodes BID11 , thus reducing the total training time.

However, large batch training often leads to sub-optimal test performance BID17 BID36 .

This has been attributed to the observation that large batch size training tends to get attracted to local minima or sharp curvature directions, which are not robust to (possible) mismatch between training and testing curves BID17 .

A full understanding of this, however, remains elusive.

There have been several solutions proposed for alleviating the problem with large batch size training.

The first notable work here is BID14 , where it was shown that by scaling the learning rate, it is possible to achieve the same testing accuracy for large batches.

In particular, ResNet-50 model was tested on ImageNet dataset, and it was shown that the baseline accuracy could be recovered up to a batch size of 8192.

However, this approach does not generalize to other networks such as AlexNet BID37 , or other tasks such as NLP.

In BID37 , an adaptive learning rate method (called LARS) was proposed which allowed scaling training to a much larger batch size of 32K with more hyper-parameter tuning.

Another notable work is Smith et al. FORMULA0 (and also BID7 ), which proposed a hybrid increase of batch size and learning rate to accelerate training.

In this approach, one would select a strategy to "anneal" the batch size during the training.

This is based on the idea that large batches contain less "noise," and that could be used much the same way as reducing learning rate during training.

More recent work BID16 ; BID24 proposed mix-precision method to further explore the limit of large batch training.

A recent study has shown that anisotropic noise injection could also help in escaping sharp minima (Zhu et al., 2018) .

The authors showed that the noise from SGD could be viewed as anisotropic, with the Hessian as its covariance matrix.

Injecting random noise using the Hessian as covariance was proposed as a method to avoid sharp minima.

Another recent work by BID36 has shown that adversarial training (or robust optimization) could be used to "regularize" against these sharp minima, with preliminary results showing superior testing performance as compared to other methods.

The link between robust optimization and regularization is a very interesting observation that has been theoretically proved in the case of Ridge regression (El Ghaoui & BID9 , and Lasso BID2 .

BID26 ; BID27 used adversarial training and showed that the model training using robust optimization is often times more robust to perturbations, as compared to normal SGD training.

Similar observations have been made by others BID30 BID13 .

We consider a supervised learning framework where the goal is to minimize a loss function L(θ): DISPLAYFORM0 where θ are the model weight parameters, Z = X × Y is the training dataset, and l(z, θ) is the loss for a datum z ∈ Z. Here, X is the input, Y is the corresponding label, and N = |Z| is the cardinality of the training set.

SGD is typically used to optimize Eqn.

FORMULA0 by taking steps of the form: DISPLAYFORM1 where B is a mini-batch of examples drawn randomly from Z, and η t is the step size (learning rate) at iteration t. In the case of large batch size training, the batch size is increased to large values.

Smith & Le (2018) views the learning rate and batch size as noise injected during optimization.

Both a large learning rate as well as a small batch size can be considered to be equivalent to high noise injection.

This is explained by modeling the behavior of NNs as a stochastic differential equation (SDE) of the following form: DISPLAYFORM2 where (t) is the noise injected by SGD (see BID28 for details).

The authors then argue that the noise magnitude is proportional to g = η t ( |Z| |B| − 1).

For mini-batch |B| |Z|, the noise magnitude can be estimated as g ≈ η t |Z| |B| .

Hence, in order to achieve the benefits from small batch size training, i.e., the noise generated by small batch training, the learning rate η t should increase proportionally to the batch size, and vice versa.

That is, the same annealing behavior could be achieved by increasing the batch size, which is the method used by BID29 .The need for annealing can be understood by considering a convex problem.

When we get closer to a local minimum, a more accurate descent direction with less noise is preferable to a more noisy direction, since less noise helps converge to rather than oscillate around the local minimum.

This explains the manual batch size and learning rate changes proposed in BID29 BID7 .

Ideally, we would like to have an automatic method that could provide us with such information and regularize against local minima with poor generalization.

As we show next, this is possible through the use of second order information combined with robust optimization.

In this section, we propose a method for utilizing second order information to adaptively change the batch size.

We refer to this as the Adaptive Batch Size (ABS) method; see Alg.

1.

Intuitively, using a larger batch size in regions where the loss has a "flatter" landscape, and using a smaller batch size in regions with a "sharper" loss landscape, could help to avoid attraction to local minima with poor generalization.

This information can be obtained through the lens of the Hessian operator.

-Learning rate lr, learning rate decay steps A, learning rate decay ratio ρ -Initial Batch Size B, minimum batch size Bmin, maximum batch size Bmax, input x, label y. -Eigenvalue decreasing ratio α, eigenvalue computation frequency n, i.e., after training n samples compute eigenvalue, batch increasing ratio β, duration factor κ, i.e., if we compute κ times Hessian but eigenvalue does not decrease, we would increase the batch size -If adversarial training is used, perturbation magnitude adv , perturbation ratio γ (γmax) of training data, decay ratio ω, vanishing step τ 2: Initialization: Eig = None, Visiting Sample = 0 DISPLAYFORM0 DISPLAYFORM1 We adaptively increase the batch size as the Hessian eigenvalue decreases or stays stable for several epochs (fixed to be ten in all of the experiments).The second component of our framework is robust optimization.

In the seminal works of (El Ghaoui & BID9 BID33 ), a connection between robust optimization and regularization was proved in the context of ridge and lasso regression.

In BID36 , the authors empirically showed that adversarial training leads to more robust models with respect to adversarial perturbation.

An interesting corollary was that, after adversarial training, the model converges to regions that are considerably flatter, as compared to the baseline.

Thus, we can combine our ABS algorithm with adversarial training as a form of regularization against "sharp" minima.

We refer to this as the Adaptive Batch Size Adversarial (ABSA) method; see Alg.

1.

In practice, ABSA is often more stable than ABS.

This corresponds to solving a minmax problem instead of a normal minimization problem BID17 BID36 .

Solving this min-max problem for NNs is an intractable problem, and thus we approximately solve the maximization problem through the Fast Gradient Sign Method (FGSM) proposed by BID13 .

This basically corresponds to generating adversarial inputs using one gradient ascent step (i.e., the perturbation is computed by ∆x = ∇ x l(z, θ)).

Other possible choices are proposed by BID31 BID4 BID22 .1 FIG1 illustrates our ABS schedule as compared to a normal training strategy and the increasing batch size strategy of Smith et al. FORMULA0 ; BID7 .

Note that our learning rate adaptively changes based on the Hessian eigenvalue in order to keep the same noise level as in the baseline SGD training.

As we show in section 4, our combined approach (second order and robust optimization) not only achieves better accuracy, but it also requires significantly fewer SGD updates, as compared to Smith et al. FORMULA0 ; BID7 .

Before discussing the empirical results, an important question is whether using ABS is a convergent algorithm for even a convex problem.

Here, we show that our ABS algorithm does converge for strongly convex problems.

Based on an assumption about the loss (Assumption 2 in Appendix A), it is not hard to prove the following theorem.

Theorem 1.

Under Assumption 2, let assume at step t, the batch size used for parameter update is b t , the step size is b t η 0 , where η 0 is fixed and satisfies, DISPLAYFORM0 where B max is the maximum batch size during training.

Then, with θ 0 as the initilization, the expected optimality gap satisfies the following inequality, DISPLAYFORM1 From Theorem 1, if b t ≡ 1, the convergence rate for t steps, based on equation 5, is (1 − η 0 c s ).

However, the convergence rate of Alg.

1 becomes DISPLAYFORM2 With an adaptive b t , Alg.

1 can converge faster than basic SGD.

We show empirical results for a logistic regression problem in the Appendix A, which is a simple convex problem.

We evaluate the performance of our ABS and ABSA methods on different datasets (ranging from O(1E4) to O(1E7) training examples) and multiple NN models.

We compare the baseline performance (i.e., small batch size), along with other state-of-the-art methods proposed for large batch training BID29 BID14 .

The two main metrics for comparison are (1) the final accuracy and (2) the total number of updates.

Preferably we would want a higher testing accuracy along with fewer SGD updates.

We emphasize that, for all of the datasets and models we tested, we do not change any of the hyper-parameters in our algorithm.

We use the exact same parameters used in the baseline model, and we do not tailor any parameters to suit our algorithm.

A detailed explanation of the different NN models, and the datasets is given in Appendix B.Section 4.1 shows the result of ABS (ABSA) compared to BaseLine (BL), FB BID14 and GG BID29 .

Section 4.2 presents the results on more challenging datasets of TinyImageNet and ImageNet.

The superior performance of our method does come at the cost of backpropagating the Hessian.

Thus, in section 4.3, we discuss how approximate Hessian informatino could be used to alleviate teh costs.

We first start by discussing the results of ABS and ABSA on SVHN and Cifar-10/100 datasets.

Notice that GG and our ABS (ABSA) have different batch sizes during training.

Hence the batch size reported in our results represents the maximum batch size during training.

To allow for a direct comparison we also report the number of weight updates in our results (lower is better).

It should be mentioned that the number of SGD updates is not necessarily the same as the wall-clock time.

Therefore, we also report a simulated training time of I3 model in Appendix C. TAB1 report the test accuracy and the number of parameter updates for different datasets and models.

First, note the drop in BL accuracy for large batch confirming the accuracy degradation problem.

Moreover, note that the FB strategy only works well for moderate batch sizes (it diverges for large batch).

However, the GG method has a very consistent performance, but its number of parameter updates are usually greater than our method.

Looking at the last two major columns of TAB1 -7, the test performances ABS achieves are similar accuracy as BL.

Overall, the number of updates of ABS is 3-10 times smaller than BL with batch size 128.

However, for most cases, ABSA achieves superior results.

This confirms the effectiveness of adversarial training combined with the second order information.

SVHN is a very simple dataset, and Cifar-10/100 are relatively small datasets, and one might wonder whether the improvements we reported in section 4.1 hold for more complex problems.

Here, we report the ABSA method on more challenging datasets, i.e., TinyImageNet and ImageNet.

We use the exact same hyper-parameters in our algorithm, even though tuning them could potentially be preferable for us.

TinyImageNet is an image classification problem, with 200 classes and only 500 images per class.

Thus it is easy to overfit the training data.

The results for I1 model is reported in TAB2 .

Note that with fewer SGD iterations, ABSA can achieve better test accuracy than other methods.

The performance of ABSA is actually about 1% higher ( the training loss and test performance of I1 on TinyImagenet is shown in FIG4 in appendix).

Note that we do not tune the hyper-parameters, e.g., α, β, and perhaps one could close the gap between 70.24% and 70.4% with fine tuning of our hyper-parameters.

However, from a practical point of view such tuning is antithetical to the goal of large batch size training as it would increase the total training time, and we specifically did not want to tailor any new parameters for a particular model/dataset.

One of the limitations of our ABS (ABSA) method is the additional computational cost for computing the top Hessian eigenvalue.

If we use the full Hessian operator, the second backpropagation needs to be done all the way to the first layer of NN.

For deep networks this could lead to high cost.

Here, we empirically explore whether we could use approximate second order information, and in particular we test a block Hessian approximation Figure 6 .

The block approximation corresponds to only analyzing the Hessian of the last few layers.

In Figure 6 (see Appendix D), we plot the trace of top eigenvalues of full Hessian and block Hessian for C1 model.

Although the top eigenvalue of block Hessian has more variance than that of full Hessian, the overall trends are similar for C1.

The test performance of C1 on Cifar-10 with block Hessian is 84.82% with 4600 parameter updates (as compared to 84.42% for full Hessian ABSA).

The test performance of C4 on Cifar-100 with block Hessian is 68.01% with 12500 parameter updates (as compared to 68.43% for full Hessian ABSA).

These results suggest that using a block Hessian to estimate the trend of the full Hessian might be a good choice to overcome computation cost, but a more detailed analysis is needed.

We introduce an adaptive batch size algorithm based on Hessian information to speed up the training process of NNs, and we combine this approach with adversarial training (which is a form of robust optimization, and which could be viewed as a regularization term for large batch training).

We extensively test our method on multiple datasets (SVHN, Cifar-10/100, TinyImageNet and ImageNet) with multiple NN models (AlexNet, ResNet, Wide ResNet and SqueezeNext).

As the goal of large batch is to reduce training time, we did not perform any hyper-parameter tuning to tailor our method for any of these tests.

Our method allows one to increase batch size and learning rate automatically, based on Hessian information.

This helps significantly reduce the number of parameter updates, and it achieves superior generalization performance, without the need to tune any of the additional hyper-parameters.

Finally, we show that a block Hessian can be used to approximate the trend of the full Hessian to reduce the overhead of using second-order information.

These improvements are useful to reduce NN training time in practice.

• L(θ) is continuously differentiable and the gradient function of L is Lipschitz continuous with Lipschitz constant L g , i.e. DISPLAYFORM0 for all θ 1 and θ 2 .Also, the global minima of L(θ) is achieved at θ * and L(θ * ) = L * .•

Each gradient of each individual l i (z i ) is an unbiased estimation of the true gradient, i.e. DISPLAYFORM1 where V(·) is the variance operator, i.e. DISPLAYFORM2 From the Assumption 2, it is not hard to get, DISPLAYFORM3 DISPLAYFORM4 With Assumption 2, the following two lemmas could be found in any optimization reference, e.g. .

We give the proofs here for completeness.

Lemma 3.

Under Assumption 2, after one iteration of stochastic gradient update with step size η t at θ t , we have DISPLAYFORM5 where DISPLAYFORM6 Proof.

With the L g smooth of L(θ), we have DISPLAYFORM7 From above, the result follows.

Lemma 4.

Under Assumption 2, for any θ, we have DISPLAYFORM8 Proof.

Let DISPLAYFORM9 Then h(θ) has a unique global minima atθ DISPLAYFORM10 The following lemma is trivial, we omit the proof here.

DISPLAYFORM11 PROOF OF THEOREM 1Given these lemmas, we now proceed with the proof of Theorem 1.Proof.

Assume the batch used at step t is b t , according to Lemma 3 and 5, DISPLAYFORM12 where the last inequality is from Lemma 4.

This yields DISPLAYFORM13 It is not hard to see, DISPLAYFORM14 which concludes DISPLAYFORM15 Therefore, DISPLAYFORM16 We show a toy example of binary logistic regression on mushroom classification dataset 2 .

We split the whole dataset to 6905 for training and 1819 for validation.

η 0 = 1.2 for SGD with batch size 100 and full gradient descent.

We set 100 ≤ b t ≤ 3200 for our algorithm, i.e. ABS.

Here we mainly focus on the training losses of different optimization algorithms.

The results are shown in FIG3 .

In order to see if η 0 is not an optimal step size of full gradient descent, we vary η 0 for full gradient descent; see results in FIG3 .

In this section, we give the detailed outline of our training datasets, models, strategy as well as hyper-parameter used in Alg 1.Dataset.

We consider the following datasets.• SVHN.

The original SVHN BID23 dataset is small.

However, in this paper, we choose the additional dataset, which contains more than 500k samples, as our training dataset.

• Cifar.

The two Cifar (i.e., Cifar-10 and Cifar-100) datasets BID18 ) have same number of images but different number of classes.• TinyImageNet.

TinyImageNet consists of a subset of ImangeNet images BID6 , which contains 200 classes.

Each of the class has 500 training and 50 validation images.

3 The size of each image is 64 × 64.• ImageNet.

The ILSVRC 2012 classification dataset BID6 ) consists of 1000 images classes, with a total of 1.2 million training images and 50,000 validation images.

During training, we crop the image to 224 × 224.Model Architecture.

We implement the following convolution NNs.

When we use data augmentation, it is exactly same the standard data augmentation scheme as in the corresponding model.• S1.

AlexNet like model on SVHN as same as BID36 [C1].

We training it for 20 epochs with initial learning rate 0.01, and decay a factor of 5 at epoch 5, 10 and 15.

There is no data augmentation.• C1.

ResNet18 on Cifar-10 dataset BID15 .

We training it for 90 epochs with initial learning rate 0.1, and decay a factor of 5 at epoch 30, 60, 80.

There is no data augmentation.

• C2.

WResNet 16-4 on Cifar-10 dataset (Zagoruyko & Komodakis, 2016) .

We training it for 90 epochs with initial learning rate 0.1, and decay a factor of 5 at epoch 30, 60, 80.

There is no data augmentation.• C3.

SqueezeNext on Cifar-10 dataset BID12 .

We training it for 200 epochs with initial learning rate 0.1, and decay a factor of 5 at epoch 60, 120, 160.

Data augmentation is implemented.• C4.

ResNet18 on Cifar-100 dataset BID15 .

We training it for 160 epochs with initial learning rate 0.1, and decay a factor of 10 at epoch 80, 120.

Data augmentation is implemented.

• I1.

ResNet50 on TinyImageNet dataset BID15 .

We training it for 120 epochs with initial learning rate 0.1, and decay a factor of 10 at epoch 60, 90.

Data augmentation is implemented.

• I2.

AlexNet on ImageNet dataset BID19 .

We training it for 90 epochs with initial learning rate 0.01, and decay it to 0.0001 quadratically at epoch 60, then keeps it as 0.0001 for the rest 30 epochs.

Data augmentation is implemented.• I3 ResNet18 on ImageNet dataset BID15 .

We training it for 90 epochs with initial learning rate 0.1, and decay a factor of 10 at epoch 30, 60 and 80.

Data augmentation is implemented.

Training Strategy.

We use the following training strategies• BL.

Use the standard training procedure.• FB.

Use linear scaling rule BID14 with warm-up stage.• GG.

Use increasing batch size instead of decay learning rate BID29 .• ABS.

Use our adaptive batch size strategy without adversarial training.• ABSA.

Use our adaptive batch size strategy with adversarial training.

For adversarial training, the adversarial data are generated using Fast Gradient Sign Method (FGSM) BID13 .

The hyper-parameters in Alg.

1 (α and β) are chosen to be 2, κ = 10, adv = 0.005, γ = 20%, and ω = 2 for all the experiments.

The only change is that for SVHN, the frequency to compute Hessian information is 65536 training examples as compared to one epoch, due to the small number of total training epochs (only 20).C SIMULATED TRAINING TIMEAs discussed above, the number of SGD updates does not necessarily correlate with wall-clock time, and this is particularly the case because our method require Hessian backpropagation.

Here, we use the method suggested in BID11 , to approximate the wall-clock time of our algorithm when utilizing p parallel processes.

For the ring algorithm BID31 , the communication time per SGD iteration for p processes is: DISPLAYFORM0 where α latency is the network latency, β bandwidth is the inverse bandwidth, and |θ| is the size number of model parameters measured in terms of Bits.

Moreover, we manually measure the wall-clock time of computing the Hessian information using our in-house code, as well as the cost of forward/backward calculations on a V100 GPU.

The total time will consists of this computation time and the communication one along with Hessian computation overhead (if any).

Therefore we have: DISPLAYFORM1 where T compute is the time to compute forward and backward propagation, T communication is the time to communicate between different machine, and T Hessian is the time to compute top eigenvalues.

We use the latency and bandwidth values of α latency = 2 µs, and β bandwidth = 1 6 Gb/s based on NERSC's Cori2 supercomputing platform.

Based on above formulas, we give an example of simulated computation time cost of I3 on ImageNet.

Note that for large processes and small latency terms, the communication time formula is simplified as, T comm = 2β bandwidth |θ|.In Table 3 we report the simulation time of I3 on ImageNet on 512 processes.

For GG, we assume it increases batch size by a factor of 10 at epoch 30, 60 and 80.

The batch size per GPU core is set to 16 for SGD (and 8 for Hessian computation due to memory limit) and the total batch size used for Hessian computation is set to 4096 images.

The T comp and T comm is for one SGD update and T Hessian is for one complete Hessian eigenvalue computation (including communication for Hessian computation).

Note that the total Hessian computation time for ABS/ABSA is only 1.15 × 90 = 103.5 s even though the Hessian computation is not efficiently implemented in the existing frameworks.

Note that even with the additional Hessian overhead ABS/ABSA is still much faster than BL (and these numbers are with an in-house and not highly optimized code for Hessian computations).

We furthermore note that we have added the additional computational overhead of adversarial computations to the ABSA method.

Table 3 : Below we present the breakdown of one SGD update training time in terms of forward/backwards computation (T comp ), one step communication time (T comm ), one total Hessian spectrum computation (if any T Hess ), and the total training time.

The results correspond to I3 model on ImageNet (for accuracy results please see FIG2 ).

In this section, we present additional empirical results.

TAB2 for details).

As one can see, from epoch 60 to 80, the test performance drops due to overfitting.

However, ABSA achieves the best performance with apparently less overfitting (it has higher training loss).

@highlight

Large batch size training using adversarial training and second order information