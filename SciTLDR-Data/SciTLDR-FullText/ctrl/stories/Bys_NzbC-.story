L1 and L2 regularizers are critical tools in machine learning due to their ability to simplify solutions.

However, imposing strong L1 or L2 regularization with gradient descent method easily fails, and this limits the generalization ability of the underlying neural networks.

To understand this phenomenon, we investigate how and why training fails for strong regularization.

Specifically, we examine how gradients change over time for different regularization strengths and provide an analysis why the gradients diminish so fast.

We find that there exists a tolerance level of regularization strength, where the learning completely fails if the regularization strength goes beyond it.

We propose a simple but novel method, Delayed Strong Regularization, in order to moderate the tolerance level.

Experiment results show that our proposed approach indeed achieves strong regularization for both L1 and L2 regularizers and improves both accuracy and sparsity on public data sets.

Our source code is published.

Regularization has been very common for machine learning to prevent over-fitting and to obtain sparse solutions.

Deep neural networks (DNNs), which have shown huge success in many tasks such as computer vision BID9 BID15 BID5 and speech recognition , often contain a number of parameters in multiple layers with non-linear activation functions, in order to gain enough expressive power.

However, DNNs with many parameters are often prone to over-fitting, so the need for regularization has been emphasized.

While new regularization techniques such as dropout BID16 and pruning BID2 have been proposed to solve the problem, the traditional regularization techniques using L1 or L2 norms have cooperated with them to further improve the performance significantly.

L1 regularization, often called Lasso BID17 , obtains sparse solutions so that the required memory and power consumption are reduced while keeping reasonable accuracy.

On the other hand, L2 regularization smooths the parameter distribution and reduces the magnitude of parameters, so the resulting solution is simple (i.e., less prone to over-fitting) and effective.

Indeed, our empirical results show that applying strong L2 regularization to the deep neural networks that already has dropout layers can reduce the error rate by up to 24% on a public data set.

Strong regularization is especially desired when the model contains too many parameters for the given amount of training data.

This is often the case for deep learning tasks in practice because DNNs often contain millions of parameters while labeled training data set is limited and expensive.

However, imposing strong L1 or L2 regularization on DNNs is difficult for gradient descent method due to the vanishing gradient problem.

If we impose too strong regularization, the gradient from regularization becomes dominant, and DNNs stop learning.

In this paper, we first study the interesting phenomenon that strong regularization fails in learning.

We also provide an analysis why the gradients diminish so quickly that learning completely fails.

Then, we propose a simple yet effective solution, Delayed Strong Regularization, which carries a time-dependent schedule of regularization strength.

We find that we can overcome the failure in learning by waiting for the model to reach an "active learning" phase, where the gradients' magnitudes are significant, and then enforcing strong regularization.

Delayed Strong Regularization enables us to obtain the superior performance that is otherwise hidden by learning failure in deep networks.

The proposed approach is general and does not require any additional computation.

The experiment results indicate that the proposed approach indeed achieves strong regularization, consistently yielding even higher accuracy and higher compression rate that could not be achieved.

2.1 BACKGROUND Let us denote a generic DNN by y = f (x; w) where x ∈ R d is an input vector, w ∈ R n is a flattened vector of all parameters in the network f , and y ∈ R c is an output vector after feed-forwarding x through multiple layers in f .

The network f is trained by finding optimal set of w by minimizing the cost function within the training data DISPLAYFORM0 as follows.

DISPLAYFORM1 where L is the loss function, which is usually cross-entropy loss for classification tasks.

Here, the regularization term λΩ(w) is added to simplify the solution, and λ is set to zero for non-regularized cost function.

A higher value of λ means that stronger regularization is imposed.

The most commonly used regularization function is a squared L2 norm: Ω(w) = ||w|| 2 2 , which is also called as weight decay in deep learning literature.

This L2 regularizer has an effect of reducing the magnitude of the parameters w, and the simpler solution becomes less prone to over-fitting.

On the other hand, the L1 regularizer Ω(w) = ||w|| 1 is often employed to induce sparsity in the model (i.e., make a portion of w zero).

The sparse solution is often preferred to reduce computation time and memory consumption for deep learning since DNNs often require heavy computation and big memory space.

With the gradient descent method, each model parameter at time t, w (t) i , is updated with the following formula: DISPLAYFORM2 where α is a learning rate.

As L1 norm is not differentiable at 0, the formula doesn't have value when w (t) i = 0, but in practice, the subgradient 0 is often used.

Please see Section 2.3 for more details.

From the formula, we can see that L2 regularizer continuously reduces the magnitude of a parameter proportionally to it while L1 regularizer reduces the magnitude by a constant.

In both regularizers, strong regularization thus means greatly reducing the magnitude of parameters.

Strong regularization is especially useful for deep learning because the DNNs often contain a large number of parameters while the training data is limited in practice.

However, we have observed a phenomenon where learning suddenly fails when strong regularization is imposed for gradient descent method, which is the most commonly used solver for deep learning.

The example of the phenomenon is depicted in Figure 1 .

In the example, the architectures VGG-16 BID15 and AlexNet BID9 were employed for the data set CIFAR-100 BID8 .2 As shown, the accuracy increases as we enforce more regularization.

However, it suddenly drops to 1.0% after enforcing a little more regularization, which means that the model entirely fails to learn.

The depicted training loss also indicates that it indeed learns faster with stronger regularization (λ = 1 × 10 −3 ), but the training loss does not improve at all when even stronger regularization is imposed (λ = 2 × 10 −3 ).In order to look at this phenomenon in more detail, we show how gradients and their proportion change in Figure 2 .

As depicted in Figure 2a

−3 ) follows a path that has a relatively steep slope during the first 150 epoch, and then it converges with a gentle slope.

However, a model with a little stronger L2 regularization (λ = 2 × 10 −3 ) does not follow a path that has a good slope, so it does not really have a chance to learn from gradients.

A close-up view of this in the first 20 epochs is depicted in Figure 2b .

The models with moderate L1 and L2 regularization seem to follow a good path in a couple of epochs.

Through following the good path, the models keep the proportion of the gradients from L to all gradients dominant, especially for the first 150 epochs (Figure 2c ).

On the other hand, the models with a little stronger regularization fail to follow such path and the gradients from L decrease exponentially (Figure 2b ).

Since the magnitude of gradients from L decreases faster than that from Ω, the proportion of the latter to all gradients becomes dominant (Figure 2c ), and it results in failure in learning.

From this observation, we can see that there exists a tolerance level of regularization strength, which decides success or failure of entire learning.

Why does the magnitude of the gradient from L decrease so fast?

It is not difficult to see why the magnitude of ∂L ∂wi decreases so fast when the regularization is strong.

In deep neural networks, the gradients are dictated by back-propagation.

It is well known that the gradients at the l th layer are given by DISPLAYFORM0 where a (l−1) is the output of the neurons at the (l − 1) th layer and δ (l) is the l th -layer residual which follows the recursive relation DISPLAYFORM1 where ⊙ and a ′ denote the element-wise multiplications and derivatives of the activation function respectively.

Using the recursive relation, we obtain DISPLAYFORM2 If the regularization is too strong, the weights would be significantly suppressed as shown in Figure 5b .

From (5), since the gradients are proportional to the product of the weights at later layers (whose magnitudes are typically much less than 1 for strong regularization), they are even more suppressed.

In fact, the suppression is more severe than what we have deduced above.

The factor a (l−1) in (5) could actually lead to further suppression to the gradients when the weights are very small, for the following reasons.

First of all, we use ReLU as the activation function and it could be written as DISPLAYFORM3 where Θ(x) is the Heaviside step function.

Using this, we could write DISPLAYFORM4 Applying FORMULA7 recursively, we can see that a (l−1) is proportional to the product of the weights at previous layers.

Again, when the weights are suppressed by strong regularization, a (l−1) would be suppressed correspondingly.

Putting everything together, we can conclude that in the presence of strong regularization, the gradients are far more suppressed than the weights.

Strictly speaking, the derivations above are valid only for fully-connected layers.

For convolutional layers, the derivations are more complicated but similar.

Our conclusions above would still be valid.

Normalization Normalization techniques such as batch normalization BID7 and weight normalization BID13 can be possible approaches to prevent the L gradients from diminishing quickly.

However, it has been shown that L2 regularization has no regularizing effect when combined with normalization but only influences on the effective learning rate, resulting in good performance BID18 .

In other words, the normalization techniques do not really simplify the solution as the decrease of parameter magnitude is canceled by normalization.

This does not meet our goal, which is to heavily simplify solutions to reduce over-fitting, so we propose an approach that meets our goal.

Since we have seen that stronger regularization can result in better performance in Figure 1 , we propose a method that is able to accommodate strong regularization.

Specifically, we introduce a time-dependent regularization strength, λ t , to the equation FORMULA2 , and it is defined as DISPLAYFORM0 where epoch(t) gets the epoch number of the time step t, and γ is a hyper-parameter that is set through cross-validation.

The formula means that we do not impose any regularization until γ th epoch, and then impose the strong regularization in each training step.

The underlying hypothesis is that once the model follows a good learning path, i.e., the gradient from L is big enough, it won't easily change its direction because of the steep slope, and thus, it can learn without failure.

We empirically verify our hypothesis in the experiment section.

The hyper-parameter γ is relatively easy to set because the models often follow the good path in a couple of epochs, and once they follow such path, learning does not fail.

We recommend using 2 ≤ γ ≤ 20.

Please note that our approach is different from imposing a slightly weaker regularization throughout the whole training.

The reduced amount by not skipping regularization for the first few epochs is negligible compared to the total reduced amount by regularization.

In addition, we empirically show that our approach can achieve a much higher sparsity than the baseline in the parameter space.

The proposed method is easy to implement, and the hyper-parameter is easy to set.

Also, the method is very close to the traditional regularization method so that it inherits the traditional one's good performance for non-strong regularization while it also achieves strong regularization.

Although the method is very simple, we found that it shows the best accuracy among the approaches we tried in our preliminary experiments while it is the simplest.

The preliminary experiments are further discussed in Appendix B.Proximal gradient algorithm for L1 regularizer Meanwhile, since L1 norm is not differentiable at zero, we employ the proximal gradient algorithm BID11 , which enables us to obtain proper sparsity (i.e., guaranteed convergence) for non-smooth regularizers.

We use the following update formulae: DISPLAYFORM1 where S is a soft-thresholding operator.

Basically, the algorithm assigns zero to a parameter if its next updated value is smaller than αλ.

In other cases, it just decreases the magnitude of the parameter as usual.

We first evaluate the effectiveness of our proposed method with popular architectures, AlexNet BID9 and VGG-16 BID15 on the public data sets CIFAR-10 and CIFAR-100 BID8 ).

Then, we employ variations of VGG on another public data set SVHN BID10 , in order to see the effect of the number of hidden layers on the tolerance level.

Please note that we do not employ architectures that contain normalization techniques such as batch normalization BID7 , for the reason described in Section 2.2.

The data set statistics are described in TAB0 .

VGG-11 and VGG-19 for SVHN contain 9.8 and 20.6 millions of parameters.

Regularization is applied to all network parameters except bias terms.

We use PyTorch 3 framework for all experiments, and we use its official computer vision library 4 for the implementations of the networks.

In order to accommodate the data sets, we made some modifications to the networks.

The kernel size of AlexNet's max-pooling layers is changed from 3 to 2, and the first convolution layer's padding size is changed from 2 to 5.

All of its fully connected layers are modified to have 256 neurons.

For VGG, we modified the fully connected layers to have 512 neurons.

The output layers of both networks have 10 neurons for CIFAR-10 and SVHN, and 100 neurons for CIFAR-100.

The networks are learned by stochastic gradient descent algorithm with momentum of 0.9.

The parameters are initialized according to BID4 .

The batch size is set to 128, and the initial learning rate is set to 0.05 and decays by a factor of 2 every 30 epochs during the whole 300-epoch training.

In all experiments, we set γ = 5.

We did not find significantly different results for 2 ≤ γ ≤ 20.

Please note that we still use drop out layers (with drop probability 0.5) and pre- AlexNet and VGG-16 are experimented for different regularization methods (L1 and L2) and different data sets (CIFAR-10 and CIFAR-100), yielding 8 combinations of experiment sets.

Then, VGG-11, VGG-16, and VGG-19 are experimented for L1 and L2 regularization methods on SVHN, yielding 6 experiment sets.

For each experiment set, we set the baseline method as the one with well-tuned L1 or L2 regularization but without our time-dependent regularization strength.

For each regularization, we try more than 10 different values of λ, and for each value, we report average accuracy of three independent runs and report 95% confidence interval.

We perform statistical significance test (t-test) for the improvement over the baseline method and report the p-value.

We also report sparsity of each trained model, which is the proportion of the number of zero-valued parameters to the number of all parameters.

Please note that we mean the sparsity by the one actually derived by the models, not by pruning parameters with threshold after training.

The experiment results by VGG-16 are depicted in FIG1 .

As we investigated in Section 2.2, the baseline method suddenly fails beyond certain values of tolerance level.

However, our proposed method does not fail for higher values of λ.

As a result, our model can achieve higher accuracy as well as higher sparsity.

In practice, L2 regularization is used more often than L1 regularization due to its superior performance, and this is true for our VGG-16 experiments too.

Using L2 regularization, our model improves the model without L1 or L2 regularization but with dropout, by 14.4% in accuracy, which is about 24% of error rate improvement.

Tuning L2 regularization parameter is difficult as the curves have somewhat sharp peak, but our proposed method ease the problem to some extent by preventing the sharp drop.

Our L1 regularizer obtains much better sparsity for the similar level of accuracy FIG1 ), which means that strong regularization plays an important role in compressing neural networks.

The improvement is more prominent on CIFAR-100 than on CIFAR-10, and we think this is because over-fitting can more likely occur on CIFAR-100 as there are less images per class than on CIFAR-10.The experiment results by AlexNet are depicted in FIG2 .

Again, our proposed method achieves higher accuracy and sparsity in general.

Unlike VGG-16, we obtain more improvement over baseline with L1 regularization than with L2 regularization.

In addition, the curves make sharper peaks than those by VGG-16 especially for the sparsity regularizer (L1).

Interestingly, our proposed method often obtains higher accuracy even when the baseline does not fail on CIFAR-10, and this is only prominent when the regularization strength is relatively strong (better shown in FIG1 ).

This may be because avoiding strong regularization in the early stage of training can help the model to explore more spaces freely, and the better exploration results in finding superior local optima.

The overall experiment results are shown in TAB1 .

It shows that there is more performance improvement by L1/L2 regularization on VGG-16 than on AlexNet, which is reasonable since VGG-16 contains about 6 times more parameters so that it is more prone to over-fitting.

Our proposed model always improves the baselines by up to 3.89%, except AlexNet with L1 regularization on CIFAR-10, and most (6 out of 7) improvements are statistically significant (p¡0.05).

Our L1 regularization models always obtain higher sparsity with compression rate up to 4.2× than baselines, meaning that our model is promising for compressing neural networks.

We also show in Figure 5 how gradients and weights change when our method and the baseline are applied.

We hypothesized that if the model reaches an "active learning" phase with an elevated gradient amount, it does not suffer from vanishing gradients any more even when strong regularization is enforced.

The Figure 5a shows that our model indeed reaches there by skipping strong regularization for the first five epochs, and and it keeps learning even after strong regularization is enforced.

In Figure 5b , although the same strong regularization is enforced since epoch 5, the magnitude of weights in our model stops decreasing around epoch 20, while that in baseline (green dotted line) keeps decreasing towards zero.

This means that our model can cope with strong regularization, and it maintains its equilibrium between gradients from L and those from regularization.

The analysis in Section 2.2 implies that the number of hidden layers would affect the tolerance level when strong regularization is imposed.

That is, if there are more hidden layers in the neural network architecture, the learning will fail more easily by strong regularization.

In order to check the hypothesis empirically, we employ variations of the VGG architecture, i.e., which contain 11, 16 , and 19 hidden layers, respectively.

We experiment them on the SVHN data set.

The results by L2 regularization are depicted in Figure 6 .

As shown, the peak of our method's performance is formed around λ = 1 × 10 −3 .

As more hidden layers are added to the network, the tolerance level where the performance suddenly drops by the baseline is shifted to left, as hypothesized by our analysis.

The results by L1 regularization are in Appendix A, and it is shown that VGG-19 more easily fails as the parameters become more sparse.

The overall experiment results are shown in TAB2 .

As the method without L1/L2 regularization already performs well on this data set and there are relatively many training images per class, the improvement by L1/L2 regularization is not big.

Our method still outperforms the baseline in all experiments (6 out of 6), but the improvement is less statistically significant compared to CIFAR-10 and CIFAR-100 data sets.

The compression rate is especially good for VGG-19 mainly because its tolerance level is low so that the baseline can only achieve low sparsity.

The related work is partially covered in Section 1, and we extend other related work here.

It has been shown that L2 regularization is important for training DNNs BID9 BID1 .

Although there has been a new regularization method such as dropout, L2 regularization has been shown to reduce the test error effectively when combined with dropout BID16 .

Meanwhile, L1 regularization has also been used often in order to obtain sparse solutions.

To reduce computation and power consumption, L1 regularization and its variations such as group sparsity regularization has been promising for deep neural networks BID19 BID14 Yoon & Hwang, 2017) .

However, for both L1 and L2 regularization, the phenomenon that learning fails with strong regularization has not been emphasized previously.

BID0 showed that tuning hyper-parameters such as L2 regularization strength can be effectively done through random search instead of grid search, but they did not study how and why learning fails or how strong regularization can be successfully achieved.

Yosinski et al. (2015) visualized activations to understand deep neural networks and showed that strong L2 regularization fails to learn.

However, it was still not shown how and why learning fails and how strong regularization can be achieved.

To the best of our knowledge, there is no existing work that is dedicated to studying the phenomenon that learning fails with strong regularization and to proposing a method that can avoid the failure.

In this work, we studied the problem of achieving strong regularization for deep neural networks.

Strong regularization with gradient descent algorithm easily fails for deep neural networks, but few work addressed this phenomenon in detail.

We provided investigation and analysis of the phenomenon, and we found that there is a strict tolerance level of regularization strength.

To avoid this problem, we proposed a novel but simple method: Delayed Strong Regularization.

We performed experiments with fine tuning of regularization strength.

Evaluation results show that (1) our model successfully achieves strong regularization on deep neural networks, verifying our hypothesis that the model will keep learning once it reaches an "active learning" phase, (2) with strong regularization, our model obtains higher accuracy and sparsity, (3) the number of hidden layers in neural networks affects the tolerance level, and (4) L1/L2 regularization is difficult to tune, but it can yield great performance boost when tuned well.

There are limitations in this work.

Our proposed method can be especially useful when strong regularization is desired.

For example, deep learning projects that cannot afford a huge labeled data set can benefit from our method.

However, strong regularization may not be necessary in some other cases where the large labeled data set is available or the networks do not contain many parameters.

In addition, our experiments were not performed on a bigger data set such as ImageNet data set.

We need to fine-tune the models with different regularization parameters, and we also need multiple training sessions of each model to obtain confidence interval.

For example, the experiment results in FIG1 and 4 include 750 training sessions in total.

This is something we cannot afford with ImageNet data set, which requires several weeks of training for EACH session (unless we have GPU clusters).

Our approach cannot be applied to architectures containing normalization techniques for the reason in Section 2.2.

We actually tried to intentionally exclude normalization part from Residual Networks BID5 ) and train the model to see if we can apply our method to non-normalized Residual Networks.

However, we could not control the exploding gradients caused by the exclusion of normalization.

Our work can be further extended in several ways.

Since our model can achieve strong regularization, it will be interesting to see how the strongly regularized model performs if combined with pruning-related methods BID2 .

We applied our approach to only L1 and L2 regularizers, but applying it to other regularizers such as group sparsity regularizers will be promising as they are often employed for DNNs to compress networks.

Lastly, our proposed Delayed Strong Regularization is very simple, so one can easily extend it to more complicated methods.

All these directions are left as our future work.

To empirically check the effect of the number of hidden layers on the tolerance level, we experimented variations of VGG on SVHN, and we showed the results by L2 regularizer in Section 3.2.

Here, we show the results by L1 regularizer in FIG4 .

As more hidden layers are included to the network, the tolerance level where the baseline method suddenly fails is shifted to left.

Such pattern in baseline method is more clearly shown in the accuracy vs. sparsity plots.

VGG-19 fails to learn even when it loses only 27% of its parameters, whereas VGG-11 can still learn after losing 84% of its parameters.

The reason why we proposed a very simple method is that it is effective while it is simple to implement.

The only additional hyper-parameter, which is the number of initial epochs to skip regularization, is also not difficult to set.

We think that the proposed method is very similar to the traditional regularization method so that it inherits the traditional one's good performance for non-strong regularization while it also achieves strong regularization.

We actually tried a couple more approaches other than the proposed one in our preliminary experiments.

We found that the proposed one shows the best accuracy among the approaches we tried while it is the simplest.

For example, we tried an approach that can be regarded as a warm-start strategy.

It starts with the regularization parameter λ t = 0, and then it gradually increases λ t to λ for γ epochs, where γ >= 0 and it is empirically set.

We found that it can achieve strong regularization, but its best accuracy is similar to or slightly lower than that of our proposed approach.

We also tried a method that is similar to Ivanov regularization BID12 .

In this method, the regularization term is applied only when the L1 norm of the weights is greater than a certain threshold.

To enforce strong regularization, we set λ just above the tolerance level that is found by the baseline method.

However, this method did not accomplish any learning.

The reason is that, to reach the level of L1 norm that is low enough, the model needs to go through the strong regularization for the first few epochs, and the neurons already lose its learning ability during this period like the baseline method.

If we set λ below the tolerance level, it cannot reach the desired L1 norm without strong regularization, and thus the performance is inferior to our proposed method.

Meanwhile, an approach that applies strong regularization first and then continuously reduces the regularization strength is used in sparse learning for convex optimization.

This approach is opposite to our approach in that ours avoids strong regularization for the first few epochs and then apply strong regularization afterwards.

We performed a simple experiment with VGG-16 on CIFAR-100 to see if the approach can perform well for deep neural networks.

We set the initial regularization parameter λ = 2 × 10 −3 and λ = 6 × 10 −5 for L2 and L1 regularization, respectively, which are just above the "tolerance level".

Then, we continuously reduced λ t to zero throughout the training session.

The trained models didn't show any improvement over "random guess", which means that they were not able to learn.

Once the strong regularization is enforced in the beginning, the magnitudes of weights decrease quickly.

This in turn drives the magnitudes of gradients to diminish exponentially in deep neural networks as explained in Section 2.2, and thus, the model loses its ability to learn after a short period of strong regularization.

<|TLDR|>

@highlight

We investigate how and why strong L1/L2 regularization fails and propose a method than can achieve strong regularization.