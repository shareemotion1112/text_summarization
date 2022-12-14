Adam is shown not being able to converge to the optimal solution in certain cases.

Researchers recently propose several algorithms to avoid the issue of non-convergence of Adam, but their efficiency turns out to be unsatisfactory in practice.

In this paper, we provide a new insight into the non-convergence issue of Adam as well as other adaptive learning rate methods.

We argue that there exists an inappropriate correlation between gradient $g_t$ and the second moment term $v_t$ in Adam ($t$ is the timestep), which results in that a large gradient is likely to have small step size while a small gradient may have a large step size.

We demonstrate that such unbalanced step sizes are the fundamental cause of non-convergence of Adam, and we further prove that decorrelating $v_t$ and $g_t$ will lead to unbiased step size for each gradient, thus solving the non-convergence problem of Adam.

Finally, we propose AdaShift, a novel adaptive learning rate method that decorrelates $v_t$ and $g_t$ by temporal shifting, i.e., using temporally shifted gradient $g_{t-n}$ to calculate $v_t$. The experiment results demonstrate that AdaShift is able to address the non-convergence issue of Adam, while still maintaining a competitive performance with Adam in terms of both training speed and generalization.

First-order optimization algorithms with adaptive learning rate play an important role in deep learning due to their efficiency in solving large-scale optimization problems.

Denote g t ∈ R n as the gradient of loss function f with respect to its parameters θ ∈ R n at timestep t, then the general updating rule of these algorithms can be written as follows (Reddi et al., 2018) : DISPLAYFORM0 In the above equation, m t φ(g 1 , . . .

, g t ) ∈ R n is a function of the historical gradients; v t ψ(g 1 , . . .

, g t ) ∈ R n + is an n-dimension vector with non-negative elements, which adapts the learning rate for the n elements in g t respectively; α t is the base learning rate; and αt √ vt is the adaptive step size for m t .One common choice of φ(g 1 , . . .

, g t ) is the exponential moving average of the gradients used in Momentum (Qian, 1999) and Adam (Kingma & Ba, 2014) , which helps alleviate gradient oscillations.

The commonly-used ψ(g 1 , . . .

, g t ) in deep learning community is the exponential moving average of squared gradients, such as Adadelta (Zeiler, 2012) , RMSProp (Tieleman & Hinton, 2012) , Adam (Kingma & Ba, 2014) and Nadam (Dozat, 2016) .Adam (Kingma & Ba, 2014 ) is a typical adaptive learning rate method, which assembles the idea of using exponential moving average of first and second moments and bias correction.

In general, Adam is robust and efficient in both dense and sparse gradient cases, and is popular in deep learning research.

However, Adam is shown not being able to converge to optimal solution in certain cases.

Reddi et al. (2018) point out that the key issue in the convergence proof of Adam lies in the quantity DISPLAYFORM1 which is assumed to be positive, but unfortunately, such an assumption does not always hold in Adam.

They provide a set of counterexamples and demonstrate that the violation of positiveness of Γ t will lead to undesirable convergence behavior in Adam.

Reddi et al. (2018) then propose two variants, AMSGrad and AdamNC, to address the issue by keeping Γ t positive.

Specifically, AMSGrad definesv t as the historical maximum of v t , i.e.,v t = max {v i } t i=1 , and replaces v t withv t to keep v t non-decreasing and therefore forces Γ t to be positive; while AdamNC forces v t to have "long-term memory" of past gradients and calculates v t as their average to make it stable.

Though these two algorithms solve the non-convergence problem of Adam to a certain extent, they turn out to be inefficient in practice: they have to maintain a very large v t once a large gradient appears, and a large v t decreases the adaptive learning rate αt √ vt and slows down the training process.

In this paper, we provide a new insight into adaptive learning rate methods, which brings a new perspective on solving the non-convergence issue of Adam.

Specifically, in Section 3, we study the counterexamples provided by Reddi et al. (2018) via analyzing the accumulated step size of each gradient g t .

We observe that in the common adaptive learning rate methods, a large gradient tends to have a relatively small step size, while a small gradient is likely to have a relatively large step size.

We show that the unbalanced step sizes stem from the inappropriate positive correlation between v t and g t , and we argue that this is the fundamental cause of the non-convergence issue of Adam.

In Section 4, we further prove that decorrelating v t and g t leads to equal and unbiased expected step size for each gradient, thus solving the non-convergence issue of Adam.

We subsequently propose AdaShift, a decorrelated variant of adaptive learning rate methods, which achieves decorrelation between v t and g t by calculating v t using temporally shifted gradients.

Finally, in Section 5, we study the performance of our proposed AdaShift, and demonstrate that it solves the non-convergence issue of Adam, while still maintaining a decent performance compared with Adam in terms of both training speed and generalization.

Adam.

In Adam, m t and v t are defined as the exponential moving average of g t and g 2 t : m t = β 1 m t−1 + (1 − β 1 )g t and v t = β 2 v t−1 + (1 − β 2 )g 2 t ,where β 1 ∈ [0, 1) and β 2 ∈ [0, 1) are the exponential decay rates for m t and v t , respectively, with m 0 = 0 and v 0 = 0.

They can also be written as: DISPLAYFORM0 To avoid the bias in the estimation of the expected value at the initial timesteps, Kingma & Ba (2014) propose to apply bias correction to m t and v t .

Using m t as instance, it works as follows: DISPLAYFORM1 Online optimization problem.

An online optimization problem consists of a sequence of cost functions f 1 (θ), . . .

, f t (θ), . . . , f T (θ), where the optimizer predicts the parameter θ t at each timestep t and evaluate it on an unknown cost function f t (θ).

The performance of the optimizer is usually evaluated by regret R(T ) DISPLAYFORM2 , which is the sum of the difference between the online prediction f t (θ t ) and the best fixed-point parameter prediction f t (θ * ) for all the previous steps, where θ * = arg min θ∈ϑ T t=1 f t (θ) is the best fixed-point parameter from a feasible set ϑ.Counterexamples.

Reddi et al. (2018) highlight that for any fixed β 1 and β 2 , there exists an online optimization problem where Adam has non-zero average regret, i.e., Adam does not converge to optimal solution .

The counterexamples in the sequential version are given as follows: DISPLAYFORM3 where C is a relatively large constant and d is the length of an epoch.

In Equation 6, most gradients of f t (θ) with respect to θ are −1, but the large positive gradient C at the beginning of each epoch makes the overall gradient of each epoch positive, which means that one should decrease θ t to minimize the loss.

However, according to (Reddi et al., 2018) , the accumulated update of θ in Adam under some circumstance is opposite (i.e., θ t is increased), thus Adam cannot converge in such case.

Reddi et al. (2018) argue that the reason of the non-convergence of Adam lies in that the positive assumption of Γ t ( √ v t /α t − √ v t−1 /α t−1 ) does not always hold in Adam.

The counterexamples are also extended to stochastic cases in (Reddi et al., 2018) , where a finite set of cost functions appear in a stochastic order.

Compared with sequential online optimization counterexample, the stochastic version is more general and closer to the practical situation.

For the simplest one dimensional case, at each timestep t, the function f t (θ) is chosen as i.i.d.: DISPLAYFORM4 where δ is a small positive constant that is smaller than C. The expected cost function of the above problem is F (θ) = 1+δ C+1 Cθ − C−δ C+1 θ = δθ, therefore, one should decrease θ to minimize the loss.

Reddi et al. (2018) prove that when C is large enough, the expectation of accumulated parameter update in Adam is positive and results in increasing θ.

Reddi et al. (2018) propose maintaining the strict positiveness of Γ t as solution, for example, keeping v t non-decreasing or using increasing β 2 .

In fact, keeping Γ t positive is not the only way to guarantee the convergence of Adam.

Another important observation is that for any fixed sequential online optimization problem with infinitely repeating epochs (e.g., Equation 6), Adam will converge as long as β 1 is large enough.

Formally, we have the following theorem:

Theorem 1 (The influence of β 1 ).

For any fixed sequential online convex optimization problem with infinitely repeating of finite length epochs (d is the length of an epoch), if ∃G ∈ R such that DISPLAYFORM0 for any fixed β 2 ∈ [0, 1), there exists a β 1 ∈ [0, 1) such that Adam has average regret ≤ 2 ;The intuition behind Theorem 1 is that, if DISPLAYFORM1

In this section, we study the non-convergence issue by analyzing the counterexamples provided by Reddi et al. (2018) .

We show that the fundamental problem of common adaptive learning rate methods is that: v t is positively correlated to the scale of gradient g t , which results in a small step size α t / √ v t for a large gradient, and a large step size for a small gradient.

We argue that such an unbalanced step size is the cause of non-convergence.

We will first define net update factor for the analysis of the accumulated influence of each gradient g t , then apply the net update factor to study the behaviors of Adam using Equation 6 as an example.

The argument will be extended to the stochastic online optimization problem and general cases.

When β 1 = 0, due to the exponential moving effect of m t , the influence of g t exists in all of its following timesteps.

For timestep i (i ≥ t), the weight of g t is (1 − β 1 )β i−t 1 .

We accordingly define a new tool for our analysis: the net update net(g t ) of each gradient g t , which is its accumulated influence on the entire optimization process: DISPLAYFORM0 and we call k(g t ) the net update factor of g t , which is the equivalent accumulated step size for gradient g t .

Note that k(g t ) depends on {v i } It is worth noticing that in Momentum method, v t is equivalently set as 1.

Therefore, we have k(g t ) = α t and net(g t ) = α t g t , which means that the accumulated influence of each gradient g t in Momentum is the same as vanilla SGD (Stochastic Gradient Decent).

Hence, the convergence of Momentum is similar to vanilla SGD.

However, in adaptive learning rate methods, v t is function over the past gradients, which makes its convergence nontrivial.

Note that v t exists in the definition of net update factor (Equation 8).

Before further analyzing the convergence of Adam using the net update factor, we first study the pattern of v t in the sequential online optimization problem in Equation 6.

Since Equation 6 is deterministic, we can derive the formula of v t as follows:Lemma 2.

In the sequential online optimization problem in Equation 6, denote β 1 , β 2 ∈ [0, 1) as the decay rates, d ∈ N as the length of an epoch, n ∈ N as the index of epoch, and i ∈ {1, 2, ..., d} as the index of timestep in one epoch.

Then the limit of v nd+i when n → ∞ is: DISPLAYFORM0 Given the formula of v t in Equation 9, we now study the net update factor of each gradient.

We start with a simple case where β 1 = 0.

In this case we have DISPLAYFORM1 Since the limit of v nd+i in each epoch monotonically decreases with the increase of index i according to Equation 9, the limit of k(g nd+i ) monotonically increases in each epoch.

Specifically, the first gradient g nd+1 = C in epoch n represents the correct updating direction, but its influence is the smallest in this epoch.

In contrast, the net update factor of the subsequent gradients −1 are relatively larger, though they indicate a wrong updating direction.

We further consider the general case where β 1 = 0.

The result is presented in the following lemma:Lemma 3.

In the sequential online optimization problem in Equation 6, when n → ∞, the limit of net update factor k(g nd+i ) of epoch n satisfies: DISPLAYFORM2 and lim DISPLAYFORM3 where k(C) denotes the net update factor for gradient g i = C.Lemma 3 tells us that, in sequential online optimization problem in Equation 6, the net update factors are unbalanced.

Specifically, the net update factor for the large gradient C is the smallest in the entire epoch, while all gradients −1 have larger net update factors.

Such unbalanced net update factors will possibly lead Adam to a wrong accumulated update direction.

Similar conclusion also holds in the stochastic online optimization problem in Equation 7.

We derive the expectation of the net update factor for each gradient in the following lemma:Lemma 4.

In the stochastic online optimization problem in Equation 7, assuming α t = 1, it holds that k(C) < k(−1), where k(C) denote the expectation net update factor for g i = C and k(−1) denote the expectation net update factor for g i = −1.Though the formulas of net update factors in the stochastic case are more complicated than those in deterministic case, the analysis is actually more easier: the gradients with the same scale share the same expected net update factor, so we only need to analyze k(C) and k(−1).

From Lemma 4, we can see that in terms of the expectation net update factor, k(C) is smaller than k(−1), which means the accumulated influence of gradient C is smaller than gradient −1.

As we have observed in the previous section, a common characteristic of these counterexamples is that the net update factor for the gradient with large magnitude is smaller than these with small magnitude.

The above observation can also be interpreted as a direct consequence of inappropriate correlation between v t and g t .

Recall that v t = β 2 v t−1 + (1 − β 2 )g 2 t .

Assuming v t−1 is independent of g t , then: when a new gradient g t arrives, if g t is large, v t is likely to be larger; and if g t is small, v t is also likely to be smaller.

If β 1 = 0, then k(g t ) = α t / √ v t .

As a result, a large gradient is likely to have a small net update factor, while a small gradient is likely to have a large net update factor in Adam.

When it comes to the scenario where β 1 > 0, the arguments are actually quite similar.

Given DISPLAYFORM0 are independent from g t , then: not only does v t positively correlate with the magnitude of g t , but also the entire infinite sequence {v i } ∞ i=t positively correlates with the magnitude of g t .

Since the net update factor k( DISPLAYFORM1 , it is thus negatively correlated with the magnitude of g t .

That is, k(g t ) for a large gradient is likely to be smaller, while k(g t ) for a small gradient is likely to be larger.

The unbalanced net update factors cause the non-convergence problem of Adam as well as all other adaptive learning rate methods where v t correlates with g t .

To construct a counterexample, the same pattern is that: the large gradient is along the "correct" direction, while the small gradient is along the opposite direction.

Due to the fact that the accumulated influence of a large gradient is small while the accumulated influence of a small gradient is large, Adam may update parameters along the wrong direction.

Finally, we would like to emphasize that even if Adam updates parameters along the right direction in general, the unbalanced net update factors are still unfavorable since they slow down the convergence.

According to the previous discussion, we conclude that the main cause of the non-convergence of Adam is the inappropriate correlation between v t and g t .

Currently we have two possible solutions: (1) making v t act like a constant, which declines the correlation, e.g., using a large β 2 or keep v t nondecreasing (Reddi et al., 2018) ; (2) using a large β 1 (Theorem 1), where the aggressive momentum term helps to mitigate the impact of unbalanced net update factors.

However, neither of them solves the problem fundamentally.

The dilemma caused by v t enforces us to rethink its role.

In adaptive learning rate methods, v t plays the role of estimating the second moments of gradients, which reflects the scale of gradient on average.

With the adaptive learning rate α t / √ v t , the update step of g t is scaled down by √ v t and achieves rescaling invariance with respect to the scale of g t , which is practically useful to make the training process easy to control and the training system robust.

However, the current scheme of v t , i.e., v t = β 2 v t−1 + (1 − β 2 )g 2 t , brings a positive correlation between v t and g t , which results in reducing the effect of large gradients and increasing the effect of small gradients, and finally causes the non-convergence problem.

Therefore, the key is to let v t be a quantity that reflects the scale of the gradients, while at the same time, be decorrelated with current gradient g t .

Formally, we have the following theorem:Theorem 5 (Decorrelation leads to convergence).

For any fixed online optimization problem with infinitely repeating of a finite set of cost functions {f 1 (θ), . . .

, f t (θ), . . .

f n (θ)}, assuming β 1 = 0 and α t is fixed, we have, if v t follows a fixed distribution and is independent of the current gradient g t , then the expected net update factor for each gradient is identical.

Let P v denote the distribution of v t .

In the infinitely repeating online optimization scheme, the expectation of net update factor for each gradient g t is DISPLAYFORM0 Given P v is independent of g t , the expectation of the net update factor E[k(g t )] is independent of g t and remains the same for different gradients.

With the expected net update factor being a fixed constant, the convergence of the adaptive learning rate method reduces to vanilla SGD.Momentum (Qian, 1999) can be viewed as setting v t as a constant, which makes v t and g t independent.

Furthermore, in our view, using an increasing β 2 (AdamNC) or keepingv t as the largest v t (AMSGrad) is also to make v t almost fixed.

However, fixing v t is not a desirable solution, because it damages the adaptability of Adam with respect to the adapting of step size.

We next introduce the proposed solution to make v t independent of g t , which is based on temporal independent assumption among gradients.

We first introduce the idea of temporal decorrelation, then extend our solution to make use of the spatial information of gradients.

Finally, we incorporate first moment estimation.

The pseudo code of the proposed algorithm is presented as follows.

Algorithm 1 AdaShift: Temporal Shifting with Block-wise Spatial Operation DISPLAYFORM1 5: DISPLAYFORM2 end for 9: end for 10: //

We ignore the bias-correction, epsilon and other misc for the sake of clarity

In practical setting, f t (θ) usually involves different mini-batches x t , i.e., f t (θ) = f (θ; x t ).

Given the randomness of mini-batch, we assume that the mini-batch x t is independent of each other and further assume that f (θ; x) keeps unchanged over time, then the gradient g t = ∇f (θ; x t ) of each mini-batch is independent of each other.

Therefore, we could change the update rule for v t to involve g t−n instead of g t , which makes v t and g t temporally shifted and hence decorrelated: DISPLAYFORM0 Note that in the sequential online optimization problem, the assumption "g t is independent of each other" does not hold.

However, in the stochastic online optimization problem and practical neural network settings, our assumption generally holds.

Most optimization schemes involve a great many parameters.

The dimension of θ is high, thus g t and v t are also of high dimension.

However, v t is element-wisely computed in Equation 14.

Specifically, we only use the i-th dimension of g t−n to calculate the i-th dimension of v t .

In other words, it only makes use of the independence between g t−n [i] and g t [i], where g t [i] denotes the i-th element of g t .

Actually, in the case of high-dimensional g t and v t , we can further assume that all elements of gradient g t−n at previous timesteps are independent with the i-th dimension of g t .

Therefore, all elements in g t−n can be used to compute v t without introducing correlation.

To this end, we propose introducing a function φ over all elements of g 2 t−n , i.e., DISPLAYFORM0 For easy reference, we name the elements of g t−n other than g t−n [i] as the spatial elements of g t−n and name φ the spatial function or spatial operation.

There is no restriction on the choice of φ, and we use φ(x) = max i x[i] for most of our experiments, which is shown to be a good choice.

The max i x[i] operation has a side effect that turns the adaptive learning rate v t into a shared scalar.

An important thing here is that, we no longer interpret v t as the second moment of g t .

It is merely a random variable that is independent of g t , while at the same time, reflects the overall gradient scale.

We leave further investigations on φ as future work.

In practical setting, e.g., deep neural network, θ usually consists of many parameter blocks, e.g., the weight and bias for each layer.

In deep neural network, the gradient scales (i.e., the variance) for different layers tend to be different (Glorot & Bengio, 2010; He et al., 2015) .

Different gradient scales make it hard to find a learning rate that is suitable for all layers, when using SGD and Momentum methods.

In traditional adaptive learning rate methods, they apply element-wise rescaling for each gradient dimension, which achieves rescaling-invariance and somehow solves the above problem.

However, Adam sometimes does not generalize better than SGD (Wilson et al., 2017; Keskar & Socher, 2017) , which might relate to the excessive learning rate adaptation in Adam.

In our temporal decorrelation with spatial operation scheme, we can solve the "different gradient scales" issue more naturally, by applying φ block-wisely and outputs a shared adaptive learning rate scalar v t [i] for each block: DISPLAYFORM0 It makes the algorithm work like an adaptive learning rate SGD, where each block has an adaptive learning rate α t / v t [i] while the relative gradient scale among in-block elements keep unchanged.

As illustrated in Algorithm 1, the parameters θ t including the related g t and v t are divided into M blocks.

Every block contains the parameters of the same type or same layer in neural network.

First moment estimation, i.e., defining m t as a moving average of g t , is an important technique of modern first order optimization algorithms, which alleviates mini-batch oscillations.

In this section, we extend our algorithm to incorporate first moment estimation.

We have argued that v t needs to be decorrelated with g t .

Analogously, when introducing the first moment estimation, we need to make v t and m t independent to make the expected net update factor unbiased.

Based on our assumption of temporal independence, we further keep out the latest n gradients {g t−i } n−1 i=0 , and update v t and m t via DISPLAYFORM0 In Equation 17, β 1 ∈ [0, 1] plays the role of decay rate for temporal elements.

It can be viewed as a truncated version of exponential moving average that only applied to the latest few elements.

Since we use truncating, it is feasible to use large β 1 without taking the risk of using too old gradients.

In the extreme case where β 1 = 1, it becomes vanilla averaging.

The pseudo code of the algorithm that unifies all proposed techniques is presented in Algorithm 1 and a more detailed version can be found in the Appendix.

It has the following parameters: spatial operation φ, n ∈ N + , β 1 ∈ [0, 1], β 2 ∈ [0, 1) and α t .

The key difference between Adam and the proposed method is that the latter temporally shifts the gradient g t for n-step, i.e., using g t−n for calculating v t and using the kept-out n gradients to evaluate m t (Equation 17), which makes v t and m t decorrelated and consequently solves the nonconvergence issue.

In addition, based on our new perspective on adaptive learning rate methods, v t is not necessarily the second moment and it is valid to further involve the calculation of v t with the spatial elements of previous gradients.

We thus proposed to introduce the spatial operation φ that outputs a shared scalar for each block.

The resulting algorithm turns out to be closely related to SGD, where each block has an overall adaptive learning rate and the relative gradient scale in each block is maintained.

We name the proposed method that makes use of temporal-shifting to decorrelated v t and m t AdaShift, which means "ADAptive learning rate method with temporal SHIFTing".

In this section, we empirically study the proposed method and compare them with Adam, AMSGrad and SGD, on various tasks in terms of training performance and generalization.

Without additional declaration, the reported result for each algorithm is the best we have found via parameter grid search.

The anonymous code is provided at http://bit.ly/2NDXX6x.

Firstly, we verify our analysis on the stochastic online optimization problem in Equation 7, where we set C = 101 and δ = 0.02.

We compare Adam, AMSGrad and AdaShift in this experiment.

For fair comparison, we set α = 0.001, β 1 = 0 and β 2 = 0.999 for all these methods.

The results are shown in FIG1 .

We can see that Adam tends to increase θ, that is, the accumulate update of θ in Adam is along the wrong direction, while AMSGrad and AdaShift update θ in the correct direction.

Furthermore, given the same learning rate, AdaShift decreases θ faster than AMSGrad, which validates our argument that AMSGrad has a relatively higher v t that slows down the training.

In this experiment, we also verify Theorem 1.

As shown in FIG1 , Adam is also able to converge to the correct direction with a sufficiently large β 1 and β 2 .

Note that (1) AdaShift still converges with the fastest speed; (2) a small β 1 (e.g., β 1 = 0.9, the light-blue line in FIG1 ) does not make Adam converge to the correct direction.

We do not conduct the experiments on the sequential online optimization problem in Equation 6, because it does not fit our temporal independence assumption.

To make it converge, one can use a large β 1 or β 2 , or set v t as a constant.

We further compare the proposed method with Adam, AMSGrad and SGD by using Logistic Regression and Multilayer Perceptron on MNIST, where the Multilayer Perceptron has two hidden layers and each has 256 hidden units with no internal activation.

The results are shown in Figure 2 and Figure 3 , respectively.

We find that in Logistic Regression, these learning algorithms achieve very similar final results in terms of both training speed and generalization.

In Multilayer Perceptron, we compare Adam, AMSGrad and AdaShift with reduce-max spatial operation (max-AdaShift) and without spatial operation (non-AdaShift).

We observe that max-AdaShift achieves the lowest training loss, while non-AdaShift has mild training loss oscillation and at the same time achieves better generalization.

The worse generalization of max-AdaShift may be due to overfitting in this task, and the better generalization of non-AdaShift may stem from the regularization effect of its relatively unstable step size.

ResNet (He et al., 2016) and DenseNet (Huang et al., 2017) are two typical modern neural networks, which are efficient and widely-used.

We test our algorithm with ResNet and DenseNet on CIFAR-10 datasets.

We use a 18-layer ResNet and 100-layer DenseNet in our experiments.

We plot the best results of Adam, AMSGrad and AdaShift in Figure 4 and Figure 5 for ResNet and DenseNet, respectively.

We can see that AMSGrad is relatively worse in terms of both training speed and generalization.

Adam and AdaShift share competitive results, while AdaShift is generally slightly better, especially the test accuracy of ResNet and the training loss of DenseNet.

We further increase the complexity of dataset, switching from CIFAR-10 to Tiny-ImageNet, and compare the performance of Adam, AMSGrad and AdaShift with DenseNet.

The results are shown in FIG3 , from which we can see that the training curves of Adam and AdaShift are basically overlapped, but AdaShift achieves higher test accuracy than Adam.

AMSGrad has relatively higher training loss, and its test accuracy is relatively lower at the initial stage.

We also test our algorithm on the training of generative model and recurrent model.

We choose WGAN-GP (Gulrajani et al., 2017 ) that involves Lipschitz continuity condition (which is hard to optimize), and Neural Machine Translation (NMT) (Luong et al., 2017 ) that involves typical recurrent unit LSTM, respectively.

In FIG4 , we compare the performance of Adam, AMSGrad and AdaShift in the training of WGAN-GP discriminator, given a fixed generator.

We notice that AdaShift is significantly better than Adam, while the performance of AMSGrad is relatively unsatisfactory.

The test performance in terms of BLEU of NMT is shown in FIG4 , where AdaShift achieves a higher BLEU than Adam and AMSGrad.

In this paper, we study the non-convergence issue of adaptive learning rate methods from the perspective of the equivalent accumulated step size of each gradient, i.e., the net update factor defined in this paper.

We show that there exists an inappropriate correlation between v t and g t , which leads to unbalanced net update factor for each gradient.

We demonstrate that such unbalanced step sizes are the fundamental cause of non-convergence of Adam, and we further prove that decorrelating v t and g t will lead to unbiased expected step size for each gradient, thus solving the non-convergence problem of Adam.

Finally, we propose AdaShift, a novel adaptive learning rate method that decorrelates v t and g t via calculating v t using temporally shifted gradient g t−n .In addition, based on our new perspective on adaptive learning rate methods, v t is no longer necessarily the second moment of g t , but a random variable that is independent of g t and reflects the overall gradient scale.

Thus, it is valid to calculate v t with the spatial elements of previous gradients.

We further found that when the spatial operation φ outputs a shared scalar for each block, the resulting algorithm turns out to be closely related to SGD, where each block has an overall adaptive learning rate and the relative gradient scale in each block is maintained.

The experiment results demonstrate that AdaShift is able to solve the non-convergence issue of Adam.

In the meantime, AdaShift achieves competitive and even better training and testing performance when compared with Adam.

FIG7 .

It suggests that for a fixed sequential online optimization problem, both of β 1 and β 2 determine the direction and speed of Adam optimization process.

Furthermore, we also study the threshold point of C and d, under which Adam will change to the incorrect direction, for each fixed β 1 and β 2 that vary among [0, 1).

To simplify the experiments, we keep d = C such that the overall gradient of each epoch being +1.

The result is shown in FIG7 , which suggests, at the condition of larger β 1 or larger β 2 , it needs a larger C to make Adam stride on the opposite direction.

In other words, large β 1 and β 2 will make the non-convergence rare to happen.

We also conduct the experiment in the stochastic problem to analyze the relation among C, β 1 , β 2 and the convergence behavior of Adam.

Results are shown in the FIG7 and FIG7 and the observations are similar to the previous: larger C will cause non-convergence more easily and a larger β 1 or β 2 somehow help to resolve non-convergence issue.

In this experiment, we set δ = 1.Lemma 6 (Critical condition).

In the sequential online optimization problem Equation 6, let α t being fixed, define S(β 1 , β 2 , C, d) to be the sum of the limits of step updates in a d-step epoch: DISPLAYFORM0 Let S(β 1 , β 2 , C) = 0, assuming β 2 and C are large enough such that v t 1, we get the equation: DISPLAYFORM1 Equation FORMULA0 , though being quite complex, tells that both β 1 and β 2 are closely related to the counterexamples, and there exists a critical condition among these parameters.

Algorithm 2 AdaShift: We use a first-in-first-out queue Q to denote the averaging window with the length of n. P ush(Q, g t ) denotes pushing vector g t to the tail of Q, while P op(Q) pops and returns the head vector of Q. And W is the weight vector calculated via β 1 .

DISPLAYFORM0 3: for t = 1 to T do 4: DISPLAYFORM1 if t ≤ n then 6: DISPLAYFORM2 else 8: DISPLAYFORM3 P ush(Q, g t )10: DISPLAYFORM4 p t = p t−1 β 2

for i = 1 to M do 13: DISPLAYFORM0 14: DISPLAYFORM1 DISPLAYFORM2 To verify the temporal correlation between g t [i] and g t−n [i], we range n from 1 to 10 and calculate the average temporal correlation coefficient of all variables i. Results are shown in TAB3 .

To verify the spatial correlation between g t [i] and g t−n [j], we again range n from 1 to 10 and randomly sample some pairs of i and j and calculate the average spatial correlation coefficient of all the selected pairs.

Results are shown in [i] and v t within max-AdaShift, we range the keep number n from 1 to 10 to calculate v t and the average correlation coefficient of all variables i.

The result is shown in Table 3 and Table 4 .

With bias correction, the formulation of m t is written as follows DISPLAYFORM3 According to L'Hospitals rule, we can draw the following: DISPLAYFORM4 Thus, DISPLAYFORM5 According to the definition of limitation, let g DISPLAYFORM6 2 So, m t shares the same sign with g * in every dimension.

Given it is a convex optimization problem, let the optimal parameter be θ * , and the maximum step size is DISPLAYFORM7 Given ∇f t (θ) ∞ ≤ G, we have f t (θ) − f t (θ * ) < 2 , which implies the average regret DISPLAYFORM8 E PROOF OF LEMMA 2 DISPLAYFORM9 For a fixed d, as n approach infinity, we get the limit of m nd+i as: DISPLAYFORM10 Similarly, for v nd+i : DISPLAYFORM11 For a fixed d, as n approach infinity, we get the limit of v nd+i as: DISPLAYFORM12

Proof.

First, we define V i as: DISPLAYFORM0 where 1 ≤ i ≤ d and i ∈ N. And V i has a period of d. Let t = t − nd, then we can draw: DISPLAYFORM1 Thus, we can get the forward difference of k(g nd+i ) as: DISPLAYFORM2 V nd+1i monotonically increases within one period, when 1 ≤ i ≤ d and i ∈ N. And the weigh β j 1 for every difference term V j+i+1 − V i is fixed when i varies.

Thus, the weighted summation DISPLAYFORM3 is monotonically decreasing from positive to negative.

In other words, the forward difference is monotonically decreasing, such that there exists j, 1 ≤ j ≤ d and lim nd→∞ k(g nd+1 ) is the maximum among all net updates.

Moreover, it is obvious that lim DISPLAYFORM4 is the minimum.

Hence, we can draw the conclusion: DISPLAYFORM5 and lim DISPLAYFORM6 where K(C) is the net update factor for gradient g i = C.

Lemma 7.

1 For a bounded random variable X and a differentiable function f (x), the expectation of f (X) is as follows: DISPLAYFORM0 where D(X) is variance of X, and R 3 is as follows: DISPLAYFORM1 F (x) is the distribution function of X. R 3 is a small quantity under some condition.

And c is large enough, such that: for any > 0, DISPLAYFORM2 Proof. (Proof of Lemma 4 ) In the stochastic online optimization problem equation 7, the gradient subjects the distribution as: DISPLAYFORM3 Then we can get the expectation of g i : DISPLAYFORM4 Meanwhile, under the assumption that gradients are i.i.d., the expectation and variance of v i are as following when nd → ∞: DISPLAYFORM5 Then, for the gradient g i , the net update factor is as follows: DISPLAYFORM6 It should to be clarified that we define t j=1 β t−j 2 g 2 i+j equal to zero when t = 0.

Then we define X t as: DISPLAYFORM7 DISPLAYFORM8 DISPLAYFORM9 = β 2(t+1) 2 DISPLAYFORM10 DISPLAYFORM11 f (x) = 3 · x −5/2 8 According to lemma 7, we can the expectation of f (X t ) as follows: DISPLAYFORM12 DISPLAYFORM13 where DISPLAYFORM14 Then for gradient C and −1, the net update factor is as follows: DISPLAYFORM15 and DISPLAYFORM16 We can see that each term in the infinite series of k(C) is smaller than the corresponding one in k(−1).

Thus, k(C) < k(−1).

Proof.

From Lemma 2, we can get: DISPLAYFORM0 We sum up all updates in an epoch, and define the summation as S(β 1 , β 2 , C).

DISPLAYFORM1 Assume β 2 and C are large enough such that v t 1, we get the approximation of limit of v nd+i as: DISPLAYFORM2 Then we can draw the expression of S(β 1 , β 2 , C) as: DISPLAYFORM3 Let S(β 1 , β 2 , C) = 0, we get the equation about critical condition: DISPLAYFORM4 I HYPER-PARAMETERS INVESTIGATION

Here, we list all hyper-parameter setting of all above experiments.

In this section, we discuss the learning rate α t sensitivity of AdaShift.

We set α t ∈ {0.1, 0.01, 0.001} and let n = 10, β 1 = 0.9 and β 2 = 0.999.

The results are shown in Figure 9 and FIG1 .

Empirically, we found that when using the max spatial operation, the best learning rate for AdaShift is around ten times of Adam.

In this section, we discuss the β 1 and β 2 sensitivity of AdaShift.

We set α = 0.01, n = 10 and let β 1 ∈ {0, 0.9} and β 2 ∈ {0.9, 0.99, 0.999}. The results are shown in FIG1 and FIG1 .

According to the results, AdaShift holds a low sensitivity to β 1 and β 2 .

In some tasks, using the first moment estimation (with β 1 = 0.9 and n = 10) or using a large β 2 , e.g., 0.999 can attain better performance.

The suggested parameters setting is n = 10, β 1 = 0.9, β 2 = 0.999.

In this section, we discuss the n sensitivity of AdaShift.

Here we also test a extended version of first moment estimation where it only uses the latest m gradients (m ≤ n): DISPLAYFORM0 We set β 1 = 0.9, β 2 = 0.999.

The results are shown in FIG1 , FIG1 and FIG1 .

In these experiments, AdaShift is fairly stable when changing n and m. We have not find a clear pattern on the performance change with respect to n and m. J TEMPORAL-ONLY AND SPATIAL-ONLYIn our proposed algorithm, we apply a spatial operation on the temporally shifted gradient g t−n to update v t : DISPLAYFORM1 It is based on the temporal independent assumption, i.e., g t−n is independent of g t .

And according to our argument in Section 4.2, one can further assume every element in g t−n is independent of the i-th dimension of g t .We purposely avoid involving the spatial elements of the current gradient g t , where the independence might not holds: when a sample which is rare and has a large gradient appear in the mini-batch x t , the overall scale of gradient g t might increase.

However, for the temporally already decorrelation g t−i , further taking the advantage of the spatial irrelevance will not suffer from this problem.

We here provide extended experiments on two variants of AdaShift: (i) AdaShift (temporal-only), which only uses the vanilla temporal independent assumption and evaluate v t with: v t = β 2 v t−1 + (1 − β 2 )g 2 t−n ; (ii) AdaShift (spatial-only), which directly uses the spatial elements without temporal shifting.

According to our experiments, AdaShift (temporal-only), i.e., without the spatial operation, is less stable than AdaShift.

In some tasks, AdaShift (temporal-only) works just fine; while in some other cases, AdaShift (temporal-only) suffers from explosive gradient and requires a relatively small learning rate.

The performance of AdaShift (spatial-only) is close to Adam.

More experiments for AdaShift (spatial-only) are included in the next section.

In this section, we extend the experiments and add the comparisons with Nadam and AdaShift (spatial-only).

The results are shown in FIG1 , Figure19 and Figure20.

According to these experiments, Nadam and AdaShift (spatial-only) share similar performence as Adam.

Rahimi & Recht raise the point, at test of time talk at NIPS 2017, that it is suspicious that gradient descent (aka back-propagation) is ultimate solution for optimization.

A ill-conditioned quadratic problem with Two Layer Linear Net is showed to be challenging for gradient descent based methods, while alternative solutions, e.g., Levenberg-Marquardt, may converge faster and better.

The problem is defined as follows: DISPLAYFORM0 where A is some known badly conditioned matrix (k = 10 20 or 10 5 ), and W 1 and W 2 are the trainable parameters.

We test SGD, Adam and AdaShift with this problem, the results are shown in FIG1 , Figure 24 .

It turns out as long as the training goes enough long, SGD, Adam, AdaShift all basically converge in this problem.

Though SGD is significantly better than Adam and AdaShift.

We would tend to believe this is a general issue of adaptive learning rate method when comparing with vanilla SGD.

Because these adaptive learning rate methods generally are scale-invariance, i.e., the step-size in terms of g t /sqrt(v t ) is basically around one, which makes it hard to converge very well in such a ill-conditioning quadratic problem.

SGD, in contrast, has a step-size g t ; as the training converges SGD would have a decreasing step-size, makes it much easier to converge better.

The above analysis is confirmed with Figure 22 and Figure 23 , with a decreasing learning rate, Adam and AdaShfit both converge very good.

@highlight

We analysis and solve the non-convergence issue of Adam.