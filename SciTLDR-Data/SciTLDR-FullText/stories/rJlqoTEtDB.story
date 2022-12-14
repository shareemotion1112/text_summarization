In this paper, we propose a novel technique for improving the stochastic gradient descent (SGD) method to train deep networks, which we term \emph{PowerSGD}. The proposed PowerSGD method simply raises the stochastic gradient to a certain power $\gamma\in[0,1]$ during iterations and introduces only one additional parameter, namely, the power exponent $\gamma$ (when $\gamma=1$, PowerSGD reduces to SGD).

We further propose PowerSGD with momentum, which we term \emph{PowerSGDM}, and provide convergence rate analysis on both PowerSGD and PowerSGDM methods.

Experiments are conducted on popular deep learning models and benchmark datasets.

Empirical results show that the proposed PowerSGD and PowerSGDM obtain faster initial training speed than adaptive gradient methods,  comparable generalization ability with SGD, and improved robustness to hyper-parameter selection and vanishing gradients.

PowerSGD is essentially a gradient modifier via a nonlinear transformation.

As such, it is orthogonal and complementary to other techniques for accelerating gradient-based optimization.

Stochastic optimization as an essential part of deep learning has received much attention from both the research and industry communities.

High-dimensional parameter spaces and stochastic objective functions make the training of deep neural network (DNN) extremely challenging.

Stochastic gradient descent (SGD) (Robbins & Monro, 1951 ) is the first widely used method in this field.

It iteratively updates the parameters of a model by moving them in the direction of the negative gradient of the objective evaluated on a mini-batch.

Based on SGD, other stochastic optimization algorithms, e.g., SGD with Momentum (SGDM) (Qian, 1999) , AdaGrad (Duchi et al., 2011) , RMSProp (Tieleman & Hinton, 2012) , Adam (Kingma & Ba, 2015) are proposed to train DNN more efficiently.

Despite the popularity of Adam, its generalization performance as an adaptive method has been demonstrated to be worse than the non-adaptive ones.

Adaptive methods (like AdaGrad, RMSProp and Adam) often obtain faster convergence rates in the initial iterations of training process.

Their performance, however, quickly plateaus on the testing data (Wilson et al., 2017) .

In Reddi et al. (2018) , the authors provided a convex optimization example to demonstrate that the exponential moving average technique can cause non-convergence in the RMSProp and Adam, and they proposed a variant of Adam called AMSGrad, hoping to solve this problem.

The authors provide a theoretical guarantee of convergence but only illustrate its better performance on training data.

However, the generalization ability of AMSGrad on test data is found to be similar to that of Adam, and a considerable performance gap still exists between AMSGrad and SGD (Keskar & Socher, 2017; Chen et al., 2018) .

Indeed, the optimizer is chosen as SGD (or with Momentum) in several recent state-of-the-art works in natural language processing and computer vision (Luo et al., 2018; Wu & He, 2018) , where in these instances SGD does perform better than adaptive methods.

Despite the practical success of SGD, obtaining sharp convergence results in the non-convex setting for SGD to efficiently escape saddle points (i.e., convergence to second-order stationary points) remains a topic of active research (Jin et al., 2019; Fang et al., 2019) .

Related Works: SGD, as the first efficient stochastic optimizer for training deep networks, iteratively updates the parameters of a model by moving them in the direction of the negative gradient of the objective function evaluated on a mini-batch.

SGDM brings a Momentum term from the physical perspective, which obtains faster convergence speed than SGD.

The Momentum idea can be seen as a particular case of exponential moving average (EMA).

Then the adaptive learning rate (ALR) technique is widely adopted but also disputed in deep learning, which is first introduced by AdaGrad.

Contrast to the SGD, AdaGrad updates the parameters according to the square roots of the sum of squared coordinates in all the past gradients.

AdaGrad can potentially lead to huge gains in terms of convergence (Duchi et al., 2011) when the gradients are sparse.

However, it will also lead to rapid learning rate decay when the gradients are dense.

RMSProp, which first appeared in an unpublished work (Tieleman & Hinton, 2012) , was proposed to handle the aggressive, rapidly decreasing learning rate in AdaGrad.

It computes the exponential moving average of the past squared gradients, instead of computing the sum of the squares of all the past gradients in AdaGrad.

The idea of AdaGrad and RMSProp propelled another representative algorithm: Adam, which updates the weights according to the mean divided by the root mean square of recent gradients, and has achieved enormous success.

Recently, research to link discrete gradient-based optimization to continuous dynamic system theory has received much attention (Yuan et al., 2016; Mazumdar & Ratliff, 2018) .

While the proposed optimizer excels at improving initial training, it is completely complementary to the use of learning rate schedules (Smith & Topin, 2019; Loshchilov & Hutter, 2016) .

We will explore how to combine learning rate schedules with the PoweredSGD optimizer in future work.

While other popular techniques focus on modifying the learning rates and/or adopting momentum terms in the iterations, we propose to modify the gradient terms via a nonlinear function called the Powerball function by the authors of Yuan et al. (2016) .

In Yuan et al. (2016) , the authors presented the basic idea of applying the Powerball function in gradient descent methods.

In this paper, we 1) systematically present the methods for stochastic optimization with and without momentum; 2) provide convergence proofs; 3) include experiments using popular deep learning models and benchmark datasets.

Another related work was presented in Bernstein et al. (2018) , where the authors presented a version of stochastic gradient descent which uses only the signs of gradients.

This essentially corresponds to the special case of PoweredSGD (or PoweredSGDM) when the power exponential ?? is set to 0.

We also point out that despite the name resemblance, the power PowerSign optimizer proposed in Bello et al. (2017) is a conditional scaling of the gradient, whereas the proposed PoweredSGD optimizer applies a component-wise trasformation to the gradient.

Inspired by the Powerball method in Yuan et al. (2016) , this paper uses Powerballbased stochastic optimizers for the training of deep networks.

In particular, we make the following major contributions:

1.

We propose the PoweredSGD, which is the first systematic application of the Powerball function technique in stochastic optimization.

PoweredSGD simply applies the Powerball function (with only one additional parameter ??) on the stochastic gradient term in SGD.

Hence, it is easy to implement and requires no extra memory.

We also propose the PoweredSGDM as a variant of PoweredSGD with momentum to further improve its convergence and generalization abilities.

2.

We have proved the convergence rates of the proposed PoweredSGD and PoweredSGDM.

It has been shown that both the proposed PoweredSGD and PoweredSGDM attain the best known rates of convergence for SGD and SGDM on non-convex functions.

In fact, to the knowledge of the authors, the bounds we proved for SGD and SGDM (as special cases of PoweredSGD and PoweredSGDM when ?? = 1) provide the currently best convergence bounds for SGD and SGDM in the non-convex setting in terms of both the constants and rates of convergence (see, e.g. Yan et al. (2018) ).

3.

Experimental studies are conducted on multiple popular deep learning tasks and benchmark datasets.

The results empirically demonstrate that our methods gain faster convergence rate especially in the early train process compared with the adaptive gradient methods.

Meanwhile, the proposed methods show comparable generalization ability compared with SGD and SGDM.

Outline: The remainder of the paper is organized as below.

Section 2 proposes the PoweredSGD and PoweredSGDM algorithms.

Section 3 provides convergence results of the proposed algorithms for non-convex optimization.

Section 4 gives the experiment results of the proposed algorithms on a variety of models and datasets to empirically demonstrate their superiority to other optimizers.

Finally, conclusions are drawn in section 5.

Notation:

Given a vector a ??? R n , we denote its i-th coordinate by a i ; we use a to denote its 2-norm (Euclidean norm) and a p to denote its p-norm for p ??? 1.

Given two vectors a, b ??? R n , we use a ?? b to denote their inner product.

We denote by E[??] the expectation with respect to the underlying probability space.

In this section, we present the main algorithms proposed in this paper: PoweredSGD and PoweredS-GDM.

PoweredSGD combines the Powerball function technique with stochastic gradient descent, and PoweredSGDM is an extension of PoweredSGD to include a momentum term.

We shall prove in Section 3 that both methods converge and attain at least the best known rates of convergence for SGD and SGDM on non-convex functions, and demonstrate in Section 4 the advantages of using PoweredSGD and PoweredSGDM compared to other popular stochastic optimizers for train deep networks.

Train a DNN with n free parameters can be formulated as an unconstrained optimization problem

where f (??) : R n ??? R is a function bounded from below.

SGD proved itself an efficient and effective solution for high-dimensional optimization problems.

It optimizes f by iteratively updating the parameter vector x t ??? R n at step t, in the opposite direction of a stochastic gradient g(x t , ?? t ) (where ?? t denotes a random variable), which is calculated on t-th mini-batch of train dataset.

The update rule of SGD for solving problem (1) is

starting from an arbitrary initial point x 1 , where ?? t is known as the learning rate at step t. In the rest of the article, let g t = g(x t , ?? t ) for the sake of notation.

We then introduce a nonlinear transformation ?? ?? (z) = sign(z)|z| ?? named as the Powerball function where sign(z) returns the sign of z, or 0 if z = 0.

For any vector z = (z 1 , . . . , z n ) T , the Powerball function ?? ?? (z) is applied to all elements of z. A parameter ?? ??? R is introduced to adjust the mechanism and intensity of the Powerball function.

Applying the Powerball function to the stochastic gradient term in the update rule (2) gives the proposed PoweredSGD algorithm:

where ?? ??? [0, 1] is an additional parameter.

Clearly, when ?? = 1, we obtain the vanilla SGD (2).

The detailed pseudo-code of the proposed PoweredSGD is presented in Algorithm 1.

The momentum trick inspired by physical processes Polyak (1964); Nesterov (1983) has been successfully combined with SGD to give SGDM, which almost always gives better convergence rates on train deep networks.

We hereby follow this line to propose the PoweredSGD with Momentum (PoweredSGDM), whose update rule is

Clearly, when ?? = 0, PowerSDGM (4) reduces to PoweredSGD (3).

Pseudo-code of the proposed PoweredSGDM is detailed in Algorithm 2.

In this section, we present convergence results of PoweredGD and PoweredSGDM in the non-convex setting.

We start with some standard technical assumptions.

First, we assume that the gradient of the objective function f is L-Lipschitz.

We then assume that a stochastic first-order black-box oracle is accessible as a noisy estimate of the gradient of f at any point x ??? R n , and the variance of the noise is bounded.

Assumption 3.2 The stochastic gradient oracle gives independent and unbiased estimate of the gradient and satisfies:

where?? ??? 0 is a constant.

We will be working with a mini-batch size in the proposed PoweredSGD and PoweredSGDM.

Let n t be the mini-batch size at the t-th iteration and the corresponding mini-batch stochastic gradient be given by the average of n t calls to the above oracle.

Then by Assumption 3.2 we can show that

In other words, we can reduce variance by choosing a larger mini-batch size (see Supplementary Material A.2).

We now state the main convergence result for the proposed PoweredSGD.

Theorem 3.1 Suppose that Assumptions 3.1 and 3.2 hold.

Let T be the number of iterations.

PoweredSGD (3) with an adaptive learning rate and mini-batch size B t = T (independent of a particular step t) can lead to

where ?? ??? (0, 1), p = 1+?? 1????? for any ?? ??? [0, 1) and p = ??? for ?? = 1.

The proof of Theorem 3.1 can be found in the Supplementary Material A.2.

Remark 3.1 The proposed PoweredSGD and PoweredSGDM have the potential to outperform popular stochastic optimizers by allowing the additional parameter ?? that can be tuned for different training cases, and they always reduce to other optimizers when setting ?? = 1.

Remark 3.2 We leave ?? ??? (0, 1) to be a free parameter in the bound to provide trade-offs between bounds given by the curvature L and stochasticity?? .

If?? = 0, we can choose ?? ??? 0 and recover the convergence bound for PoweredGD (see Supplementary Material A.1).

The above theorem provides a sharp estimate of the convergence of PoweredSGD in the following sense.

When ?? = 1, the convergence bound reduces to the best known convergence rate for SGD.

Note that, because of the choice of batch size, it requires T 2 gradient evaluations in T iterations.

So the convergence rate is effectively O(1/ ??? T ).

This is the best known rate of convergence for SGD Ge et al. (2015) .

When?? = 0 (i.e., exact gradients are used and B t = 1), PoweredSGD can attain convergence in the order O(1/T ), which is consistent with the convergence rate of gradient descent.

We now present convergence analysis for PoweredSGDM.

The proof is again included in the Supplementary Material B.2 due to the space limit.

Theorem 3.2 Suppose that Assumptions 3.1 and 3.2 hold.

Let T be the number of iterations.

For any ?? ??? [0, 1), PoweredSGDM (4) with an adaptive learning rate and mini-batch size B t = T (independent of a particular step t) can lead to

where ?? ??? (0, 1), p = 1+?? 1????? for any ?? ??? [0, 1) and p = ??? for ?? = 1.

Remark 3.4 Convergence analysis of stochastic momentum methods for non-convex optimization is an important but under-explored topic.

While our results on convergence analysis do not improve the rate of convergence for stochastic momentum methods in a non-convex setting, it does match the currently best known rate of convergence (Yan et al., 2018; Bernstein et al., 2018) in special cases (?? = 0, 1) and offers very concise upper bounds in terms of the constants.

The upper bound continuously interpolates the convergence rate for ?? varying in [0, 1] and ?? varying in [0, 1).

The key technical result that made the results of Theorems 3.1 and 3.2 possible is Lemma B.1 in the Supplementary Material, which provide a tight estimate of accumulated momentum terms.

We also note that the convergence rates for ?? ??? (0, 1) are entirely new and not reported elsewhere before.

Even for the special case of ?? = 0, 1, our proof differs from that of (Yan et al., 2018; Bernstein et al., 2018) and seems more transparent.

Remark 3.5 A large mini-batch (B t = T ) is assumed for the convergence results to hold.

This is consistent with the convergence analysis in Bernstein et al. (2018) for the special case ?? = 0.

We assume this because it enables us to put analysis of PoweredGD and PoweredSGD in a unified framework so that we can obtain tighter bounds.

In the stochastic setting, similar to Remark 3.3, we note that our proof requires T 2 gradient calls in T iterations and hence the effective convergence rate is O(1/ ??? T ), which is consistent with the known rate of convergence for SGD (Ge et al., 2015) .

The propose of this section is to demonstrate the efficiency and effectiveness of the proposed PoweredSGD and PoweredSGDM algorithms.

We conduct experiments of different model architectures on datasets in comparison with widely used optimization methods including the non-adaptive method SGDM and three popular adaptive methods: AdaGrad, RMSprop and Adam.

This section is mainly composed of two parts: (1) the convergence and generalization experiments and (2) the Powerball feature experiments.

The setup for each experiment is detailed in Table 1 1 .

In the first part, we present empirical study of different deep neural network architectures to see how the proposed methods behave in terms of convergence speed and generalization.

In the second part, the experiments are conducted to explore the potential features of PoweredSGD and PoweredSGDM.

To ensure stability and reproducibility, we conduct each experiment at least 5 times from randomly initializations and the average results are shown.

The settings of hyper-parameters of a specific optimization method that can achieve the best performance on the test set are chosen for comparisons.

When two settings achieve similar test performance, the setting which converges faster is adopted.

We can have the following findings from our experiments: (1) The proposed PoweredSGD and PoweredSGDM methods exhibit better convergence rate than other adaptive methods such as Adam and RMSprop.

(2) Our proposed methods achieve better generalization performance than adaptive methods although slightly worse than SGDM.

Table 1 : Summaries of the models and datasets in our experiments.

Since the initial learning rate has a large impact on the performances of optimizers, we implement a logarithmically-spaced grid search strategy around the default learning rate for each optimization method, and leave the other hyper-parameters to their default settings.

The default learning rate for SGDM is 0.01.

We tune the learning rate on a logarithmic scale from {1, 0.1, 0.01, 0.001, 0.0001}. The momentum value in all experiments is set to default value 0.9.

PoweredSGD, PoweredSGDM: The learning rates for PoweredSGD and PoweredSGDM are chosen from the same range {1, 0.1, 0.01, 0.001, 0.0001} as SGDM.

The momentum value for PoweredS-GDM is also 0.9.

Note that ?? = 1 in Powerball function corresponds to the SGD or SGDM.

Based on extensive experiments, we empirically tune ?? from {0.5, 0.6, 0.7, 0.8, 0.9}.

AdaGrad: The learning rates for AdaGrad are {1e-1, 5e-2, 1e-2, 5e-3, 1e-3} and we choose 0 for the initial accumulator value.

RMSprop, Adam:

Both have the default learning rate 1e-3 and their learning rates are searched from {1e-2, 5e-3, 1e-3, 5e-4, 1e-4}. The parameters ?? 1 , ?? 2 and the perturbation value ?? are set to default.

As previous findings Wilson et al. (2017) show, adaptive methods generalize worse than non-adaptive methods and carefully tuning the initial learning rate yields significant improvements for them.

To better compare with adaptive methods, once we have found the value that was best performing in adaptive methods, we would try the learning rate between the best learning rate and its closest neighbor.

For example, if we tried learning rates {1e-2, 5e-3, 1e-3, 5e-4, 1e-4} and 1e-4 was best performing, we would try the learning rate 2e-4 to see if performance was improved.

We iteratively update the learning rate until performance could not be improved any more.

For all experiments, we used a mini-batch size of 128.

Fig. 1 shows the learning curves of three experiments we have conducted to observe the performance of PoweredSGD and PoweredSGDM in comparison with other widely-used optimization methods.

ResNet-50 on CIFAR-10: We trained a ResNet-50 model on CIFAR-10 and our results are shown in Fig. 1(a) and Fig. 1(b) .

We ran each experiment for a fixed budget of 160 epochs and reduced the learning rate by a factor of 10 after every 60 epochs Wilson et al. (2017) .

As the figure shows, the adaptive methods converged fast and appeared to be performing better than the non-adaptive method SGDM as expected.

WideResNet on CIFAR-100: Next, we conducted experiments on the CIFAR-100 dataset using WideResNet model.

The fixed budget here is 120 epochs and the learning rate reduces by a factor of 10 after every 60 epochs.

The results are shown in Fig. 1(e) and Fig. 1(f) .

The performance of the PoweredSGD and PoweredSGDM are still promising in both the train set and test set.

PoweredSGD, PoweredSGDM and AdaGrad had the fastest initial progress.

In the test set, PoweredSGD and PoweredSGDM had much better test accuracy than all other adaptive methods.

ResNet-50 on ImageNet: Finally, we conducted experiments on the ImageNet dataset using ResNet-50 model.

The fixed budget here is 120 epochs and the learning rate reduces by a factor of 10 after every 30 epochs.

The results are shown in Fig. 1(i) and Fig. 1(j) .

We observed that PoweredSGD and PoweredSGDM gave better convergence rates than adaptive methods while AdaGrad quickly plateaus due to too many parameter updates.

For test set, we can notice that although SGDM achieved the best test accuracy of 76.27%, PoweredSGD and PoweredSGDM gave the results of 73.71% and 73.96%, which were better than those of adaptive methods.

Additional experiments (DenseNet-121 on CIFAR-10 and ResNeXt on CIFAR100) are shown in Fig. 1(c In deep learning, the phenomenon of gradient vanishing poses difficulties in training very deep neural networks by SGD.

During the training process, the stochastic gradients in early layers can be extremely small due to the chain rule, and this can even completely stop the networks from being trained.

Our proposed PoweredSGD method can relieve the phenomenon of gradient vanishing by effectively rescaling the stochastic gradient vectors.

To validate this, we conduct experiments on the MNIST dataset by using a 13-layer fully-connected neural network with ReLU activation functions.

The SGD and proposed PoweredSGD are compared in terms of train accuracy and 1-norm of gradient vector.

As can be observed in Fig. 2 (2016) proposed the so-called Powerball accelerated gradient descent algorithm, which was updated as follows,

The authors of Yuan et al. (2016) Theorem A.1 Suppose that Assumption 3.1 holds.

The PoweredGD scheme (6) can lead to

where T is the number of iterations and p = 1+?? 1????? for any ?? ??? [0, 1) and p = ??? for ?? = 1.

Proof: Denote by x the minimizer and f = f (x ).

Then, by the L-Lipschitz continuity of ??? f and (6),

Let

By H??lder's inequality, for ?? ??? (0, 1) and with p = 1+?? 1????? and q = 1+?? 2?? , we have

It follows that

which, by a telescoping sum, gives

where 1 is vector with entries all given by 1.

It is easy to see that the estimate is also valid for ?? = 1 with p = ??? and for ?? = 0.

The proof is complete.

To analyze the convergence of PoweredSGD, we need some preliminary on the relation between mini-batch size and variance reduction of SGD.

Let ??? f (x) be the gradient of f at x ??? R n .

Suppose that we use the average of m calls to the stochastic gradient oracle, denoted by g(x, ?? i ) (i = 1, ?? ?? ?? , m), to estimate ??? f (x).

By Assumption 3.2, we have

where in the second equality we used the fact that g(x, ?? i ) (i = 1, ?? ?? ?? , m) are drawn independently and all give unbiased estimate of ??? f (x) (provided by Assumption 3.2).

Now we are ready to present the proof of Theorem 3.1.

Proof: By the L-Lipschitz continuity of ??? f and (6),

Fix any iteration number T > 1 and let ?? ??? (0, 1) to be chosen.

We can estimate

where the last inequality followed from the elementary inequality 2ab ??? ??a 2 + 1 ?? b 2 for any positive real number ?? and real numbers a, b. Substituting this into (9) gives

By the same argument in the proof for Theorem A.1, we can derive

Taking conditional expectation from both sizes gives

where ?? 2 t is the variance of the t-th stochastic gradient approximation computed using the chosen mini-batch size B t = T , which therefore satisfies ?? 2 t ????? 2 T .

Taking expectation from both sides and performing a telescoping sum give

The proof is complete.

We first analyze the deterministic version of PoweredSGDM (denoted by PoweredGDM).

The update rule for PoweredGDM is

where ?? ??? [0, 1) is a momentum constant and v 0 = 0.

Clearly, when ?? = 0, the scheme also reduces to PoweredGD.

Theorem B.1 Suppose that Assumption 3.1 holds.

For any ?? ??? [0, 1), the PoweredGDM scheme (11) with an adaptive learning rate can lead to

where T is the number of iterations and p = 1+?? 1????? for any ?? ??? [0, 1) and p = ??? for ?? = 1.

Proof: Let z t = x t + ?? 1????? v t .

It can be verified that the PoweredGDM scheme satisfies

By the L-Lipschitz continuity of ??? f and (12),

We can estimate

where ?? > 0 is to be chosen.

By the L-Lipschitz continuity of ??? f ,

Lemma B.1 For T ??? 1, we have

Proof:

It is easy to show by induction that, for t ??? 1,

Indeed, we have v 1 = 0 and v 2 = ????? 1 ?? (??? f (x 1 )).

Suppose that the above holds for t ??? 1.

Then

By Lemma B.1, inequalities (14), (15), and a telescoping sum on (13), we get

It is clear that ?? =

(1????? ) 2 L?? would minimize the bound on the right-hand side (among different choices of ?? > 0) and give

For any ?? ??? [0, 1), we can choose

(1????? ) 2 1+?? so that the bound reduces to

which immediately gives the bound in the theorem by noting z 1 = x 1 .

Proof: The proof is built on that for Theorem B.1.

With z t = x t + ?? 1????? v t , it can be verified that the PoweredSGDM scheme satisfies

By the L-Lipschitz continuity of ??? f and (4),

Similar to the proof of Theorem B.1, we can estimate

where ?? 1 > 0 is to be chosen.

By the L-Lipschitz continuity of ??? f ,

Similar to Lemma B.1, we obtain

We can also bound

where ?? > 0.

By inequalities (18)- (21), and a telescoping sum on (17), we get

Setting ?? 1 =

(1????? ) 2 L?? and choosing

(1????? ) 2 1+?? lead to

which, by taking expectation from both sides and by the same argument in the proof for Theorem A.1, leads to

which immediately gives the bound in the theorem by noting z 1 = x 1 .

Remark B.1 Clearly, Theorem 3.2 exactly reduces to Theorem B.1 when?? = 0 and ?? ??? 0.

Moreover, when ?? = 0, Theorem 3.2 reduces exactly to Theorem 3.1.

This in a sense shows that our estimates are sharp.

A careful reader will notice that in Theorems 3.1 and 3.2, our estimates of convergence rates for PoweredSGD and PoweredSGDM, respectively, are in terms of the stochastic gradients g t .

We now show that this is without loss of generality in view of Assumption 3.2.

When ?? = 1, we have

where in the last equality, we used Assumption 3.2.

This would imply

].

When ?? ??? [0, 1), by the equivalence of norm in R n , there exist positive constants C ?? and D ?? such that

for all x ??? R n .

Hence

which implies that

].

In other words, the estimates are equivalent (modulo a constant factor).

We prefer the versions in Theorems 3.1 and 3.2, because the bounds are more elegant.

The vanishing gradient problem is quite common when training deep neural networks using gradientbased methods and backpropagation.

The gradients can become too small for updating the weight values.

Eventually, this may stop the networks from further training.

The Powerball function can help amplify the gradients especially when they approach zero.

We visualized the amplification effects of Powerball function in Fig. 3 .

Thus, the attributes of PoweredSGD can help alleviate the vanishing gradient problem to some extent.

We investigated the actual performance of PoweredSGD and SGD when dealing with very deep networks.

We trained deep networks on the MNIST dataset using PoweredSGD with ?? chosen from {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} and learning rate ?? chosen from {1.0, 0.1, 0.01, 0.001}. When ?? = 1.0, the PoweredSGD becomes the vanilla SGD.

The architecture of network depth which ranges from 12 to 15 with ReLU as the activation function is shown in Table 2 .

The results are visualized using heatmaps in Fig. 4 .

784 ??? 256 Hidde layers (??10/11/12/13) 256 ??? 256 Output layer 256 ??? 10 Table 2 : The architecture of MLP in vanishing gradient experiments.

As we can observe in the visualisation, when the network depth is more than 13 layers, increasing or decreasing the learning rate of SGD could not solve the vanishing gradient problem.

For PoweredSGD, the usage of the Powerball function enables it to amplify the gradients and thus allows to further train deep networks with proper ?? settings.

This confirms our hypothesis that PoweredSGD helps alleviate the vanishing gradient problem to some extent.

We also note that, when the network increases to 15 layers, both SGD and PoweredSGD could not train the network further.

We speculate that this is due to the ratio of amplified gradients to the original gradients becomes too large (see Fig. 3 ) and a much smaller learning rate is needed (this is also consistent with the change of theoretical learning rates suggested in the convergence proofs as the gradient size decreases).

Since PoweredSGD is essentially a gradient modifier, it would also be interesting to see how to combine it with other techniques for dealing with the vanishing gradient problem.

Since PoweredSGD also reduces the gradient when the gradient size is large, it may also help alleviate the exploding gradient problem.

This gives another interesting direction for future research.

The Powerball function is a nonlinear function with a tunable hyper-parameter ?? applied to gradients, which is introduced to accelerate optimization.

To test the robustness of different ??, we trained ResNet-50 and DenseNet-121 on the CIFAR-10 dataset with PoweredSGD and SGDM.

The parameter ?? is chosen from {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0} and the learning rate is chosen from {1.0, 0.1, 0.01, 0.001}. The PoweredSGD becomes the vanilla SGD when ?? = 1.

The maximum test accuracy is recorded and the results are visualized in Fig. 5 .

Although the ?? that gets the best test accuracy depends on the choice of learning rates, we can observe that ?? can be selected within a wide range from 0.5 to 1.0 without much loss in test accuracy.

Moreover, the Powerball function with a hyper-parameter ?? could help regularize the test performance while the learning rate decreases.

For example, when ?? = 0.001 and ?? = 0.6, PoweredSGD get the best test accuracy of 90.06% compared with 79.87% accuracy of SGD.

We also compare the convergence performance of different ?? choice in Fig. 6 .

The training loss is recorded when training ResNet-50 on CIFAR-10 dataset.

As the initial learning rate decreases, the range from which the hyper-parameter ?? can be selected to accelerate training becomes wider.

As a practical guide, ?? = 0.8 seems a proper setting in most cases.

It is again observed that the choice of ?? in the range of 0.4-0.8 seems to provide improved robustness to the change of learning rates.

Figure 5 : Effects of different ?? on test accuracy.

We show the best Top-1 accuracy on CIFAR-10 dataset of ResNet-50 and DenseNet121 trained with PoweredSGD.

Although the best choice of ?? depends on learning rates, the selections can be quite robust considering the test accuracy.

In the main part of the paper, we demonstrated through multiple experiments that PoweredSGD can achieve faster initial training.

In this section we demonstrate that PoweredSGD as a gradient modifier is orthogonal and complementary to other techniques for improved learning.

The learning rate is the most important hyper-parameter to tune for deep neural networks.

Motivated by recent advancement in designing learning rate schedules such as CLR policies (Smith, 2015) and SGDR (Loshchilov & Hutter, 2016) , we conducted some preliminary experiments on combining learning rate schedules with PoweredSGD to improve its performance.

The results are shown in Fig.  7 .

The selected learning rate schedule is warm restarts introduced in (Loshchilov & Hutter, 2016) , which reset the learning rate to the initial value after a cycle of decaying the learning rate with a Figure 6 : Effects of different ?? on convergence.

We show the best train loss on CIFAR-10 dataset of ResNet-50 trained with PoweredSGD.

While the ?? which achieves the best convergence performance is closely related to the choice of learning rates, a ?? chosen in the range of 0.4-0.6 seem to provide better robustness to change of learning rates.

cosine annealing for each batch.

In Fig. 7 , SGD with momentum combined with warm restarts policy is named as SGDR.

Similarly, PoweredSGDR indicates PoweredSGD combined with a warm restarts policy.

The hyper-parameter setting is T 0 = 10 and T mult = 2 for warm restarts.

We test their performance on CIFAR-10 dataset with ResNet-50.

The results showed that the learning rate policy can improve both the convergence and test accuracy of PoweredSGD.

Indeed, PoweredSGDR achieved the lowest training error compared with SGDM and SGDR.

The test accuracy for PoweredSGDR was also improved from the 94.12% accuracy of PoweredSGD to 94.64%.

The results demonstrate that the nonlinear transformation of gradients given by the Powerball function is orthogonal and complementary to existing methods.

As such, its combination with other techniques could potentially further improve the performance.

Figure 8 below, in which the hyper-parameters that lead to the best test accuracy are chosen and can be found in Table 3 .

Table 3 : Hyper-parameter settings for experiments shown in Figure 8 .

@highlight

We propose a new class of optimizers for accelerated non-convex optimization via a nonlinear gradient transformation. 