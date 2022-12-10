Training large deep neural networks on massive datasets is  computationally very challenging.

There has been recent surge in interest in using large batch stochastic optimization methods to tackle this issue.

The most prominent algorithm in this line of research is LARS, which by  employing layerwise adaptive learning rates trains ResNet on ImageNet in a few minutes.

However, LARS performs poorly for attention models like BERT, indicating that its performance gains are not consistent across tasks.

In this paper, we first study a principled layerwise adaptation strategy to accelerate training of deep neural networks using large mini-batches.

Using this strategy, we develop a new layerwise adaptive large batch optimization technique called LAMB; we then provide convergence analysis of LAMB as well as LARS, showing convergence to a stationary point in general nonconvex settings.

Our empirical results demonstrate the superior performance of LAMB across various tasks such as BERT and ResNet-50 training with very little hyperparameter tuning.

In particular, for BERT training, our optimizer enables use of very large batch sizes of 32868 without any degradation of performance.

By increasing the batch size to the memory limit of a TPUv3 Pod, BERT training time can be reduced from 3 days to just 76 minutes (Table 1).

With the advent of large scale datasets, training large deep neural networks, even using computationally efficient optimization methods like Stochastic gradient descent (SGD), has become particularly challenging.

For instance, training state-of-the-art deep learning models like BERT and ResNet-50 takes 3 days on 16 TPUv3 chips and 29 hours on 8 Tesla P100 gpus respectively (Devlin et al., 2018; He et al., 2016) .

Thus, there is a growing interest to develop optimization solutions to tackle this critical issue.

The goal of this paper is to investigate and develop optimization techniques to accelerate training large deep neural networks, mostly focusing on approaches based on variants of SGD.

Methods based on SGD iteratively update the parameters of the model by moving them in a scaled (negative) direction of the gradient calculated on a minibatch.

However, SGD's scalability is limited by its inherent sequential nature.

Owing to this limitation, traditional approaches to improve SGD training time in the context of deep learning largely resort to distributed asynchronous setup (Dean et al., 2012; Recht et al., 2011) .

However, the implicit staleness introduced due to the asynchrony limits the parallelization of the approach, often leading to degraded performance.

The feasibility of computing gradient on large minibatches in parallel due to recent hardware advances has seen the resurgence of simply using synchronous SGD with large minibatches as an alternative to asynchronous SGD.

However, naïvely increasing the batch size typically results in degradation of generalization performance and reduces computational benefits (Goyal et al., 2017) .

Synchronous SGD on large minibatches benefits from reduced variance of the stochastic gradients used in SGD.

This allows one to use much larger learning rates in SGD, typically of the order square root of the minibatch size.

Surprisingly, recent works have demonstrated that up to certain minibatch sizes, linear scaling of the learning rate with minibatch size can be used to further speed up the training Goyal et al. (2017) .

These works also elucidate two interesting aspects to enable the use of linear scaling in large batch synchronous SGD: (i) linear scaling of learning rate is harmful during the initial phase; thus, a hand-tuned warmup strategy of slowly increasing the learning rate needs to be used initially, and (ii) linear scaling of learning rate can be detrimental beyond a certain batch size.

Using these tricks, Goyal et al. (2017) was able to drastically reduce the training time of ResNet-50 model from 29 hours to 1 hour using a batch size of 8192.

While these works demonstrate the feasibility of this strategy for reducing the wall time for training large deep neural networks, they also highlight the need for an adaptive learning rate mechanism for large batch learning.

Variants of SGD using layerwise adaptive learning rates have been recently proposed to address this problem.

The most successful in this line of research is the LARS algorithm (You et al., 2017) , which was initially proposed for training RESNET.

Using LARS, ResNet-50 can be trained on ImageNet in just a few minutes!

However, it has been observed that its performance gains are not consistent across tasks.

For instance, LARS performs poorly for attention models like BERT.

Furthermore, theoretical understanding of the adaptation employed in LARS is largely missing.

To this end, we study and develop new approaches specially catered to the large batch setting of our interest.

Contributions.

More specifically, we make the following main contributions in this paper.

• Inspired by LARS, we investigate a general adaptation strategy specially catered to large batch learning and provide intuition for the strategy.

• Based on the adaptation strategy, we develop a new optimization algorithm (LAMB) for achieving adaptivity of learning rate in SGD.

Furthermore, we provide convergence analysis for both LARS and LAMB to achieve a stationary point in nonconvex settings.

We highlight the benefits of using these methods for large batch settings.

• We demonstrate the strong empirical performance of LAMB across several challenging tasks.

Using LAMB we scale the batch size in training BERT to more than 32k without degrading the performance; thereby, cutting the time down from 3 days to 76 minutes.

Ours is the first work to reduce BERT training wall time to less than couple of hours.

• We also demonstrate the efficiency of LAMB for training state-of-the-art image classification models like RESNET.

To the best of our knowledge, ours is first adaptive solver that can achieve state-of-the-art accuracy for RESNET-50 as adaptive solvers like Adam fail to obtain the accuracy of SGD with momentum for these tasks.

The literature on optimization for machine learning is vast and hence, we restrict our attention to the most relevant works here.

Earlier works on large batch optimization for machine learning mostly focused on convex models, benefiting by a factor of square root of batch size using appropriately large learning rate.

Similar results can be shown for nonconvex settings wherein using larger minibatches improves the convergence to stationary points; albeit at the cost of extra computation.

However, several important concerns were raised with respect to generalization and computational performance in large batch nonconvex settings.

It was observed that training with extremely large batch was difficult (Keskar et al., 2016; Hoffer et al., 2017) .

Thus, several prior works carefully hand-tune training hyper-parameters, like learning rate and momentum, to avoid degradation of generalization performance (Goyal et al., 2017; Li, 2017; You et al., 2018; Shallue et al., 2018) .

(Krizhevsky, 2014) empirically found that simply scaling the learning rate linearly with respect to batch size works better up to certain batch sizes.

To avoid optimization instability due to linear scaling of learning rate, Goyal et al. (2017) proposed a highly hand-tuned learning rate which involves a warm-up strategy that gradually increases the LR to a larger value and then switching to the regular LR policy (e.g. exponential or polynomial decay).

Using LR warm-up and linear scaling, Goyal et al. (2017) managed to train RESNET-50 with batch size 8192 without loss in generalization performance.

However, empirical study (Shallue et al., 2018) shows that learning rate scaling heuristics with the batch size do not hold across all problems or across all batch sizes.

More recently, to reduce hand-tuning of hyperparameters, adaptive learning rates for large batch training garnered significant interests.

Several recent works successfully scaled the batch size to large values using adaptive learning rates without degrading the performance, thereby, finishing RESNET-50 training on ImageNet in a few minutes (You et al., 2018; Iandola et al., 2016; Codreanu et al., 2017; Akiba et al., 2017; Jia et al., 2018; Smith et al., 2017; Martens & Grosse, 2015; Devarakonda et al., 2017; Mikami et al., 2018; Osawa et al., 2018; You et al., 2019; Yamazaki et al., 2019) .

To the best of our knowledge, the fastest training result for RESNET-50 on ImageNet is due to Ying et al. (2018) , who achieve 76+% top-1 accuracy.

By using the LARS optimizer and scaling the batch size to 32K on a TPUv3 Pod, Ying et al. (2018) was able to train RESNET-50 on ImageNet in 2.2 minutes.

However, it was empirically observed that none of these performance gains hold in other tasks such as BERT training (see Section 4).

Notation.

For any vector x t ∈ R d , either x t,j or [x t ] j are used to denote its j th coordinate where j ∈ [d].

Let I be the d × d identity matrix, and let I = [I 1 , I 2 , ..., I h ] be its decomposition into column submatrices

be the block of variables corresponding to the columns of I i i.e.,

to denote the gradient with respect to x (i) .

For any vectors u, v ∈ R d , we use u 2 and u/v to denote elementwise square and division operators respectively.

We use .

and .

1 to denote l 2 -norm and l 1 -norm of a vector respectively.

We start our discussion by formally stating the problem setup.

In this paper, we study nonconvex stochastic optimization problems of the form

where is a smooth (possibly nonconvex) function and P is a probability distribution on the domain S ⊂ R k .

Here, x corresponds to model parameters, is the loss function and P is an unknown data distribution.

We assume function (x) is L i -smooth with respect to x (i) , i.e., there exists a constant L i such that

We use L ∞ and L avg to denote max i L i and i Li h respectively.

We assume the following bound on the variance in stochastic gradients:

to denote the vectors of standard deviations of stochastic gradient per layer and per dimension respectively.

Finally, we assume that the gradients are bounded i.e., [∇l(x, s)

Note that such assumptions are typical in the analysis of stochastic first-order methods (cf. (Ghadimi & Lan, 2013a; Ghadimi et al., 2014) ).

Stochastic gradient descent (SGD) is one of the simplest first-order algorithms for solving problem in Equation 1.

The update at the t th iteration of SGD is of the following form:

where S t is set of b random samples drawn from the distribution P. For very large batch settings, the following is a well-known result for SGD.

Theorem 1 ((Ghadimi & Lan, 2013b) ).

With large batch b = T and using appropriate learning rate, we have the following for the iterates of SGD:

where x * is an optimal solution to the problem in equation 1 and x a is an iterate uniformly randomly chosen from {x 1 , · · · , x T }.

However, tuning the learning rate η t in SGD, especially in large batch settings, is difficult in practice.

Furthermore, the dependence on L ∞ (the maximum of smoothness across dimension) can lead to significantly slow convergence.

In the next section, we discuss algorithms to circumvent this issue.

In this section, we first discuss a general strategy to adapt the learning rate in large batch settings.

Using this strategy, we discuss two specific algorithms in the later part of the section.

Since our primary focus is on deep learning, our discussion is centered around training a h-layer neural network.

General Strategy.

Suppose we use an iterative base algorithm A (e.g. SGD or ADAM) in the small batch setting with the following layerwise update rule:

x t+1 = x t + η t u t , where u t is the update made by A at time step t. We propose the following two changes to the update for large batch settings:

1.

The update is normalized to unit l 2 -norm.

This is ensured by modifying the update to the form u t / u t .

Throughout this paper, such a normalization is done layerwise i.e., the update for each layer is ensured to be unit l 2 -norm.

2.

The learning rate is scaled by φ( x t ) for some function φ : R + → R + .

Similar to the normalization, such a scaling is done layerwise.

Suppose the base algorithm A is SGD, then the modification results in the following update rule:

for all layers i ∈ [h] and where x (i) t and g

t are the parameters and the gradients of the i th layer at time step t. The normalization modification is similar to one typically used in normalized gradient descent except that it is done layerwise.

Note that the modification leads to a biased gradient update; however, in large-batch settings, it can be shown that this bias is small.

It is intuitive that such a normalization provides robustness to exploding gradients (where the gradient can be arbitrarily large) and plateaus (where the gradient can be arbitrarily small).

Normalization of this form essentially ignores the size of the gradient and is particularly useful in large batch settings where the direction of the gradient is largely preserved.

The scaling term involving φ ensures that the norm of the update is of the same order as that of the parameter.

We found that this typically ensures faster convergence in deep neural networks.

In practice, we observed that a simple function of φ(z) = min{max{z, γ l }, γ u } works well.

It is instructive to consider the case where φ(z) = z. In this scenario, the overall change in the learning rate is

, which can also be interpreted as an estimate on the inverse of Lipschitz constant of the gradient (see equation 2).

We now discuss different instantiations of the strategy discussed above.

In particular, we focus on two algorithms: LARS (3.1) and the proposed method, LAMB (3.2).

The first instantiation of the general strategy is LARS algorithm (You et al., 2017) , which is obtained by using momentum optimizer as the base algorithm A in the framework.

LARS was earlier proposed for large batch learning for RESNET on ImageNet.

In general, it is observed that the using (heavy-ball) momentum, one can reduce the variance in the stochastic gradients at the cost of little bias.

The pseudocode for LARS is provide in Algorithm 1.

We now provide convergence analysis for LARS in general nonconvex setting stated in this paper.

For the sake of simplicity, we analyze the case where β 1 = 0 and λ = 0 in Algorithm 1.

However, our analysis should extend to the general case as well.

We will defer all discussions about the convergence rate to the end of the section.

Then for x t generated using LARS (Algorithm 1), we have the following bound

where x * is an optimal solution to the problem in equation 1 and x a is an iterate uniformly randomly chosen from {x 1 , · · · , x T }.

The second instantiation of the general strategy is obtained by using ADAM as the base algorithm A. ADAM optimizer is popular in deep learning community and has shown to have good performance for training state-of-the-art language models like BERT.

Unlike LARS, the adaptivity of LAMB is two-fold: (i) per dimension normalization with respect to the square root of the second moment used in ADAM and (ii) layerwise normalization obtained due to layerwise adaptivity.

The pseudocode for LAMB is provided in Algorithm 2.

When β 1 = 0 and β 2 = 0, the algorithm reduces to be Sign SGD where the learning rate is scaled by square root of the layer dimension (Bernstein et al., 2018) .

The following result provides convergence rate for LAMB in general nonconvex settings.

Similar to the previous case, we focus on the setting where β 1 = 0 and λ = 0.

As before, our analysis extends to the general case; however, the calculations become messy.

, and α l ≤ φ(v) ≤ α u for all v > 0 where α l , α u > 0.

Then for x t generated using LAMB (Algorithm 2), we have the following bounds:

where x * is an optimal solution to the problem in equation 1 and x a is an iterate uniformly randomly chosen from {x 1 , · · · , x T }.

Discussion on convergence rates.

We first start our discussion with the comparison of convergence rate of LARS with that of SGD (Theorem 1).

The convergence rates of LARS and SGD differ in two ways: (1) the convergence criterion is (E[

2 as opposed to E[ ∇f 2 ] in SGD and (2) the dependence on L and σ in the convergence rate.

Briefly, the convergence rate of LARS is better than SGD when the gradient is denser than curvature and stochasticity.

This convergence rate comparison is similar in spirit to the one obtained in (Bernstein et al., 2018) .

Assuming that the convergence criterion in Theorem 1 and Theorem 2 is of similar order (which happens when gradients are fairly dense), convergence rate of LARS and LAMB depend on L avg instead of L ∞ and are thus, significantly better than that of SGD.

A more quantitative comparison is provided in Section C of the Appendix.

The comparison of LAMB (with β 2 = 0) with SGD is along similar lines.

We obtain slightly worse rates for the case where β 2 > 0; although, we believe that its behavior should be better than the case β 2 = 0.

We leave this investigation to future work.

We now present empirical results comparing LAMB with existing optimizers on two important large batch training tasks: BERT and RESNET-50 training.

We also compare LAMB with existing optimizers for small batch size (< 1K) and small dataset (e.g. CIFAR, MNIST) (see Appendix).

Experimental Setup.

To demonstrate its robustness, we use very minimal hyperparameter tuning for the LAMB optimizer.

Thus, it is possible to achieve better results by further tuning the hyperparameters.

The parameters β 1 and β 2 in Algorithm 2 are set to 0.9 and 0.999 respectively in all our experiments; we only tune the learning rate.

We use a polynomially decaying learning rate of η t = η 0 ×(1−t/T ) in Algorithm 2), which is the same as in BERT baseline.

This setting also works for all other applications in this paper.

Furthermore, for BERT and RESNET-50 training, we did not tune the hyperparameters of LAMB while increasing the batch size.

We use the square root of LR scaling rule to automatically adjust learning rate and linear-epoch warmup scheduling.

We use TPUv3 in all the experiments.

A TPUv3 Pod has 1024 chips and can provide more than 100 petaflops performance for mixed precision computing.

To make sure we are comparing with solid baselines, we use grid search to tune the hyper-parameters for ADAM, ADAGRAD, ADAMW (ADAM with weight decay), and LARS.

We also tune weight decay for ADAMW.

All the hyperparameter tuning settings are reported in the Appendix.

Due to space constraints, several experimental details are relegated to the Appendix.

We first discuss empirical results for speeding up BERT training.

For this experiment, we use the same dataset as Devlin et al. (2018) , which is a concatenation of Wikipedia and BooksCorpus with 2.5B and 800M words respectively.

We specifically focus on the SQuAD task 2 in this paper.

The F1 score on SQuAD-v1 is used as the accuracy metric in our experiments.

All our comparisons are with respect to the baseline BERT model by Devlin et al. (2018) .

To train BERT, Devlin et al. (2018) first train the model for 900k iterations using a sequence length of 128 and then switch to a sequence length of 512 for the last 100k iterations.

This results in a training time of around 3 days on 16 TPUv3 chips.

The baseline BERT model 3 achieves a F1 score of 90.395.

To ensure a fair comparison, we follow the same SQuAD fine-tune procedure of Devlin et al. (2018) without modifying any configuration (including number of epochs and hyperparameters).

As noted earlier, we could get even better results by changing the fine-tune configuration.

For instance, by just slightly changing the learning rate in the fine-tune stage, we can obtain a higher F1 score of 91.688 for the batch size of 16K using LAMB.

We report a F1 score of 91.345 in Table 1 , which is the score obtained for the untuned version.

Below we describe two different training choices for training BERT and discuss the corresponding speedups.

For the first choice, we maintain the same training procedure as the baseline except for changing the training optimizer to LAMB.

We run with the same number of epochs as the baseline but with batch size scaled from 512 to 32K.

The choice of 32K batch size (with sequence length 512) is mainly due to memory limits of TPU Pod.

Our results are shown in Table 1 .

By using the LAMB optimizer, we are able to achieve a F1 score of 91.460 in 15625 iterations for a batch size of 32768 (14063 iterations for sequence length 128 and 1562 iterations for sequence length 512).

With 32K batch size, we reduce BERT training time from 3 days to around 100 minutes.

We achieved 49.1 times speedup by 64 times computational resources (76.7% efficiency).

We consider the speedup is great because we use the synchronous data-parallelism.

There is a communication overhead coming from transferring of the gradients over the interconnect.

For RESNET-50, researchers are able to achieve 90% scaling efficiency because RESNET-50 has much fewer parameters (# parameters is equal to #gradients) than BERT (25 million versus 300 million).

To obtain further improvements, we use the Mixed-Batch Training procedure with LAMB.

Recall that BERT training involves two stages: the first 9/10 of the total epochs use a sequence length of 128, while the last 1/10 of the total epochs use a sequence length of 512.

For the second stage training, which involves a longer sequence length, due to memory limits, a maximum batch size of only 32768 can be used on a TPUv3 Pod.

However, we can potentially use a larger batch size for the first stage because of a shorter sequence length.

In particular, the batch size can be increased to 131072 for the first stage.

However, we did not observe any speedup by increasing the batch size from 65536 to 131072 for the first stage, thus, we restrict the batch size to 65536 for this stage.

By using this strategy, we are able to make full utilization of the hardware resources throughout the training Table 1 : We use the F1 score on SQuAD-v1 as the accuracy metric.

The baseline F1 score is the score obtained by the pre-trained model (BERT-Large) provided on BERT's public repository (as of February 1st, 2019).

We use TPUv3s in our experiments.

We use the same setting as the baseline: the first 9/10 of the total epochs used a sequence length of 128 and the last 1/10 of the total epochs used a sequence length of 512.

All the experiments run the same number of epochs.

Dev set means the test data.

It is worth noting that we can achieve better results by manually tuning the hyperparameters.

The data in this procedure.

Increasing the batch size is able to warm-up and stabilize the optimization process (Smith et al., 2017) , but decreasing the batch size brings chaos to the optimization process and can cause divergence.

In our experiments, we found a technique that is useful to stabilize the second stage optimization.

Because we switched to a different optimization problem, it is necessary to re-warm-up the optimization.

Instead of decaying the learning rate at the second stage, we ramp up the learning rate from zero again in the second stage (re-warm-up).

As with the first stage, we decay the learning rate after the re-warm-up phase.

With this method, we only need 8599 iterations and finish BERT training in 76 minutes (100.2% efficiency).

Comparison with ADAMW and LARS.

To ensure that our approach is compared to a solid baseline for the BERT training, we tried three different strategies for tuning ADAMW: (1) ADAMW with default hyperparameters (see Devlin et al. (2018) ) (2) ADAMW with the same hyperparameters as LAMB, and (3) ADAMW with tuned hyperparameters.

ADAMW stops scaling at the batch size of 16K because it is not able to achieve the target F1 score (88.1 vs 90.4).

The tuning information of ADAMW is shown in the Appendix.

For 64K/32K mixed-batch training, even after extensive tuning of the hyperparameters, we fail to get any reasonable result with ADAMW optimizer.

We conclude that ADAMW does not work well in large-batch BERT training or is at least hard to tune.

We also observe that LAMB performs better than LARS for all batch sizes (see Table 2 ).

ImageNet training with ResNet-50 is an industry standard metric that is being used in MLPerf 4 .

The baseline can get 76.3% top-1 accuracy in 90 epochs (Goyal et al., 2017) .

All the successful implementations are based on momentum SGD (He et al., 2016; Goyal et al., 2017) or LARS optimizer (Ying et al., 2018; Jia et al., 2018; Mikami et al., 2018; You et al., 2018; Yamazaki et al., 2019) .

Before our study, we did not find any paper reporting a state-of-the-art accuracy achieved by ADAM, ADAGRAD, or ADAMW optimizer.

In our experiments, even with comprehensive hyper-parameter tuning, ADAGRAD/ADAM/ADAMW (with batch size 16K) only achieves 55.38%/66.04%/67.27% top-1 accuracy.

After adding learning rate scheme of Goyal et al. (2017) , the top-1 accuracy of ADAGRAD/ADAM/ADAMW was improved to 72.0%/73.48%/73.07%.

However, they are still much lower than 76.3%.

The details of the tuning information are in the Appendix.

Table 3 shows that LAMB can achieve the target accuracy.

Beyond a batch size of 8K, LAMB's accuracy is higher than the momentum.

LAMB's accuracy is also slightly better than LARS.

At a batch size of 32K, LAMB achieves 76.4% top-1 accuracy while LARS achieves 76.3%.

At a batch size of 2K, LAMB is able to achieve 77.11% top-1 accuracy while LARS achieves 76.6%.

Table 3 : Top-1 validation accuracy of ImageNet/RESNET-50 training at the batch size of 16K (90 epochs).

The performance of momentum was reported by (Goyal et al., 2017) .

+ means adding the learning rate scheme of Goyal et al. (2017) to the optimizer: (1) 5-epoch warmup to stablize the initial stage; and (2) multiply the learning rate by 0.1 at 30th, 60th, and 80th epoch.

The target accuracy is around 0.763 (Goyal et al., 2017) .

All the adaptive solvers were comprehensively tuned.

Proof.

We analyze the convergence of LARS for general minibatch size here.

Recall that the update of LARS is the following

For simplicity of notation, we reason the Since the function f is L-smooth, we have the following:

The first inequality follows from the lipschitz continuous nature of the gradient.

Let ∆

t − ∇ i f (x t ).

Then the above inequality can be rewritten in the following manner:

Using Cauchy-Schwarz inequality in the above inequality, we have:

Taking expectation, we obtain the following:

Summing the above inequality for t = 1 to T and using telescoping sum, we have the following inequality:

Rearranging the terms of the above inequality, and dividing by ηT α l , we have:

Proof.

We analyze the convergence of LAMB for general minibatch size here.

Recall that the update of LAMB is the following

For simplicity of notation, we reason the Since the function f is L-smooth, we have the following:

The above inequality simply follows from the lipschitz continuous nature of the gradient.

We bound term T 1 in the following manner:

This follows from the fact that r

If β 2 = 0, then T 1 can be bounded as follows:

The rest of the proof for β 2 = 0 is similar to argument for the case β 2 > 0, which is shown below.

Taking expectation, we have the following:

Using the bound on the probability that the signs differ, we get:

Substituting the above bound on T 1 in equation 7, we have the following bound:

Input: x1 ∈ R d , learning rate {ηt} T t=1 , parameters 0 < β1, β2 < 1, scaling function φ, > 0, parameters 0 < {β

Compute ratio rt =m √v

Compute ratio rt =m √v

Summing the above inequality for t = 1 to T and using telescoping sum, we have the following inequality:

Rearranging the terms of the above inequality, and dividing by ηT α l , we have:

Inspired by the comparison used by (Bernstein et al., 2018) for comparing SIGN SGD with SGD, we define the following quantities:

Then LARS convergence rate can be written in the following manner:

If ψ L ψ 2 g and ψ σ ψ 2 g then LARS (i.e., gradient is more denser than curvature or stochasticity), we gain over SGD.

Otherwise, SGD's upper bound on convergence rate is better.

Figure 1: This figure shows N-LAMB and NN-LAMB can achieve a comparable accuracy compared to LAMB optimizer.

Their performances are much better than momentum solver.

The result of momentum optimizer was reported by Goyal et al. (2017) .

For Nadam, we use the learning rate recipe of (Goyal et al., 2017) : (1) 5-epoch warmup to stablize the initial stage; and (2) multiply the learning rate by 0.1 at 30th, 60th, and 80th epoch.

The target accuracy is around 0.763 (Goyal et al., 2017) .

We also tuned the learning rate of Nadam in {1e-4, 2e-4, ..., 9e-4, 1e-3, 2e-3, ..., 9e-3, 1e-2}. Sutskever et al. (2013) report that Nesterov's accelerated gradient (NAG) proposed by Nesterov (1983) is conceptually and empirically better than the regular momentum method for convex, non-stochastic objectives.

Dozat (2016) incorporated Nesterov's momentum into Adam optimizer and proposed the Nadam optimizer.

Specifically, only the first moment of Adam was modified and the second moment of Adam was unchanged.

The results on several applications (Word2Vec, Image Recognition, and LSTM Language Model) showed that Nadam optimizer improves the speed of convergence and the quality of the learned models.

We also tried using Nesterov's momentum to replace the regular momentum of LAMB optimizer's first moment.

In this way, we got a new algorithm named as N-LAMB (Nesterov LAMB).

The complete algorithm is in Algorithm 3.

We can also Nesterov's momentum to replace the regular momentum of LAMB optimizer's second moment.

We refer to this algorithm as NN-LAMB (Nesterov's momentum for both the first moment and the second moment).

The details of NN-LAMB were shown in Algorithm 4.

Dozat (2016) suggested the best performance of Nadam was achieved by β 1 = 0.975, β 2 = 0.999, and = 1e-8.

We used the same settings for N-LAMB and NN-LAMB.

We scaled the batch size to 32K for ImageNet training with ResNet-50.

Our experimental results show that N-LAMB and NN-LAMB can achieve a comparable accuracy compared to LAMB optimizer.

Their performances are much better than momentum solver (Figure 1 ).

There are two operations at each iteration in original Adam optimizer (let us call it adam-correction):

It has an impact on the learning rate by η t := η t * (1 − β t 2 )/(1 − β t 1 ).

According to our experimental results, adam-correction essentially has the same effect as learning rate warmup (see Figure 2) .

The warmup function often was implemented in the modern deep learning system.

Thus, we can remove adam-correction from the LAMB optimizer.

We did not observe any drop in the test or validation accuracy for BERT and ImageNet training.

We need to compute the matrix/tensor norm for each layer when we do the parameter updating in the LAMB optimizer.

We tried different norms in LAMB optimizer.

However, we did not observe a significant difference in the validation accuracy of ImageNet training with ResNet-50.

In our experiments, the difference in validation accuracy is less than 0.1 percent (Figure 3) .

We use L2 norm as the default.

According to DAWNBench, DavidNet (a custom 9-layer Residual ConvNet) is the fastest model for CIFAR-10 dataset (as of April 1st, 2019) 5 .

The baseline uses the momentum SGD optimizer.

Table 6 and Figure 4 show the test accuracy of CIFAR-10 training with DavidNet.

The PyTorch implementation (momentum SGD optimizer) on GPUs was reported on Standford DAWNBench's website, which achieves 94.06% in 24 epochs.

The Tensorflow implementation (momentum SGD optimizer) on TPU achieves a 93.72% accuracy in 24 epochs 6 .

We use the implementation of TensorFlow on TPUs.

LAMB optimizer is able to achieve 94.08% test accuracy in 24 epochs, which is better than other adaptive optimizers and momentum SGD.

Even on the smaller tasks like MNIST training with LeNet, LAMB is able to achieve a better accuracy than existing solvers (Table 7 ).

Figure 4: LAMB is better than the existing solvers (batch size = 512).

We make sure all the solvers are carefully tuned.

The learning rate tuning space of Adam, AdamW, Adagrad and LAMB is {0. 0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50} .

The momentum optimizer was tuned by the baseline implementer.

The weight decay term of AdamW was tuned by {0.0001, 0.001, 0.01, 0.1, 1.0}. Table 6 : CIFAR-10 training with DavidNet (batch size = 512).

All of them run 24 epochs and finish the training under one minute on one cloud TPU.

We make sure all the solvers are carefully tuned.

The learning rate tuning space of Adam, AdamW, Adagrad and LAMB is {0. 0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45

There are several hyper-parameters in LAMB optimizer.

Although users do not need to tune them, we explain them to help users to have a better understanding.

β 1 is used for decaying the running average of the gradient.

β 2 is used for decaying the running average of the square of gradient.

The default setting for other parameters: weight decay rate λ=0.01, β 1 =0.9, β 2 =0.999, =1e-6.

We did not tune β 1 and β 2 .

However, our experiments show that tuning them may get a higher accuracy.

Based on our experience, learning rate is the most important hyper-parameter that affects the learning efficiency and final accuracy.

Bengio (2012) suggests that it is often the single most important hyper-parameter and that it always should be tuned.

Thus, to make sure we have a solid baseline, we carefully tune the learning rate of ADAM, ADAMW, ADAGRAD, and momentum SGD

In our experiments, we found that the validation loss is not reliable for large-batch training.

A lower validation loss does not necessarily lead to a higher validation accuracy ( Figure 5 ).

Thus, we use the test/val accuracy or F1 score on dev set to evaluate the optimizers.

H.0.1 BERT Table 8 shows some of the tuning information from BERT training with ADAMW optimizer.

ADAMW stops scaling at the batch size of 16K.

The target F1 score is 90.5.

LAMB achieves a F1 score of 91.345.

The table shows the tuning information of ADAMW.

In Table 8 , we report the best F1 score we observed from our experiments.

The loss curves of BERT training by LAMB for different batch sizes are shown in Figure 6 .

We observe that the loss curves are almost identical to each other, which means our optimizer scales well with the batch size.

The training loss curve of BERT mixed-batch pre-training with LAMB is shown in Figure 7 .

This figure shows that LAMB can make the training converge smoothly at the batch size of 64K.

Figure 8 shows that we can achieve 76.8% scaling efficiency by scaling the batch size (49.1 times speedup by 64 times computational resources) and 101.8% scaling efficiency with mixed-batch (65.2 times speedup by 64 times computational resources) From these figures we can see that these ratios are very different from each other for different layers.

LAMB uses the trust ratio to help the slow learners to train faster.

If you are not interested in the baseline tuning details, please skip this section.

Goyal et al. (2017) suggested a proper learning rate warmup and decay scheme may help improve the ImageNet classification accuracy.

We included these techniques in Adam/AdamW/AdaGrad tuning.

Specifically, we use the learning rate recipe of Goyal et al. (2017): (1) 5-epoch warmup to stablize the initial stage; and (2) multiply the learning rate by 0.1 at 30th, 60th, and 80th epoch.

The target accuracy is around 76.3% (Goyal et al., 2017) .

There techniques help to improve the accuracy of Adam/AdamW/AdaGrad to around 73%.

However, even with these techniques, Adam/AdamW/AdaGrad stil can not achieve the target validation accuracy.

To make sure our baseline is solid, we carefully tuned the hyper-parameters.

Table 9 shows the tuning information of standard Adagrad.

Table 10 shows the tuning information of adding the learning rate scheme of Goyal et al. (2017) to standard Adagrad.

Table 11 shows the tuning information of standard Adam.

Table shows the tuning information of adding the learning rate scheme of Goyal et al. (2017) to standard Adam.

It is tricky to tune the AdamW optimizer since both the L2 regularization and weight decay have the effect on the performance.

Thus we have four tuning sets.

The first tuning set is based on AdamW with default L2 regularization.

We tune the learning rate and weight decay.

The tuning information is in Figures 13, 14, 15 , and 16.

The second tuning set is based on AdamW with disabled L2 regularization.

We tune the learning rate and weight decay.

The tuning information is in Figures 17, 18 , 19, and 20.

Figure 6: This figure shows the training loss curve of LAMB optimizer.

We just want to use this figure to show that LAMB can make the training converge smoothly.

Even if we scale the batch size to the extremely large cases, the loss curves are almost identical to each other.

Then we add the learning rate scheme of Goyal et al. (2017) to AdamW and refer to it as AdamW+.

The third tuning set is based on AdamW+ with default L2 regularization.

We tune the learning rate and weight decay.

The tuning information is Figure 21 and 22.

The fourth tuning set is based on AdamW+ with disabled L2 regularization.

We tune the learning rate and weight decay.

The tuning information is in Figures 23, 24 , 25.

Based on our comprehensive tuning results, we conclude the existing adaptive solvers do not perform well on ImageNet training or at least it is hard to tune them.

Published as a conference paper at ICLR 2020 Published as a conference paper at ICLR 2020

@highlight

A fast optimizer for general applications and large-batch training.

@highlight

In this paper, the authors made a study on large-batch training for the BERT, and successfully trained a BERT model in 76 minutes.

@highlight

This paper develops a layerwise adaptation strategy that allows training BERT models with large 32k mini-batches vs baseline 512.