We consider new variants of optimization algorithms.

Our algorithms are based on the observation that mini-batch of stochastic gradients in consecutive iterations do not change drastically and consequently may be predictable.

Inspired by the similar setting in online learning literature called Optimistic Online learning, we propose two new optimistic algorithms for AMSGrad and Adam, respectively, by  exploiting the predictability of gradients.

The new algorithms combine the idea of momentum method, adaptive gradient method, and algorithms in Optimistic Online learning, which leads to speed up in training deep neural nets in practice.

Nowadays deep learning has been shown to be very effective in several tasks, from robotics (e.g. BID15 ), computer vision (e.g. BID12 ; BID9 ), reinforcement learning (e.g. BID18 , to natural language processing (e.g. ).

Typically, the model parameters of a state-of-the-art deep neural net is very high-dimensional and the required training data is also in huge size.

Therefore, fast algorithms are necessary for training a deep neural net.

To achieve this, there are number of algorithms proposed in recent years, such as AMSGRAD (Reddi et al. (2018) ), ADAM BID13 ), RMSPROP (Tieleman & Hinton (2012) ), ADADELTA (Zeiler (2012) ), and NADAM BID6 ), etc.

All the prevalent algorithms for training deep nets mentioned above combines two ideas: the idea of adaptivity in ADAGRAD BID7 BID17 ) and the idea of momentum as NESTEROV'S METHOD BID19 ) or the HEAVY BALL method BID20 ).

ADAGRAD is an online learning algorithm that works well compared to the standard online gradient descent when the gradient is sparse.

The update of ADAGRAD has a notable feature: the learning rate is different for different dimensions, depending on the magnitude of gradient in each dimension, which might help in exploiting the geometry of data and leading to a better update.

On the other hand, NESTEROV'S METHOD or the Momentum Method BID20 ) is an accelerated optimization algorithm whose update not only depends on the current iterate and current gradient but also depends on the past gradients (i.e. momentum).

State-of-the-art algorithms like AMSGRAD (Reddi et al. (2018) ) and ADAM BID13 ) leverages these two ideas to get fast training for neural nets.

In this paper, we propose an algorithm that goes further than the hybrid of the adaptivity and momentum approach.

Our algorithm is inspired by OPTIMISTIC ONLINE LEARNING BID4 ; Rakhlin & Sridharan (2013) ; Syrgkanis et al. (2015) ; BID0 ).

OPTIMISTIC ONLINE LEARNING considers that a good guess of the loss function in the current round of online learning is available and plays an action by exploiting the good guess.

By exploiting the guess, those algorithms in OPTIMISTIC ONLINE LEARNING have regret in the form of O( T t=1 g t − m t ), where g t is the gradient of loss function in round t and m t is the "guess" of g t before seeing the loss function in round t (i.e. before getting g t ).

This kind of regret can be much smaller than O( √ T ) when one has a good guess m t of g t .

We combine the OPTIMISTIC ONLINE LEARNING idea with the adaptivity and the momentum ideas to design new algorithms in training deep neural nets, which leads to NEW-OPTIMISTIC-AMSGRAD and NEW-OPTIMISTIC-ADAM.

We also provide theoretical analysis of NEW-OPTIMISTIC-AMSGRAD.

The proposed OPTIMISTIC-algorithms not only adapt to the informative dimensions and exhibit momentums but also take advantage of a good guess of the next gradient to facilitate acceleration.

We evaluate our algorithms with BID13 ), (Reddi et al. (2018) ) and BID5 ).

Experiments show that our OPTIMISTIC-algorithms are faster than the baselines.

We should explain that BID5 proposed another version of optimistic algorithm for ADAM, which is referred to as ADAM-DISZ in this paper.

We apply the idea of BID5 ) on AMSGRAD, which leads to AMSGRAD-DISZ.

Both ADAM-DISZ and AMSGRAD-DISZ are used as baselines.

Both AMSGRAD (Reddi et al. (2018) ) and ADAM BID13 ) are actually ONLINE LEARNING algorithms.

They use REGRET ANALYSIS to provide some theoretical guarantees of the algorithms.

Since one can convert an online learning algorithm to an offline optimization algorithm by online-to-batch conversion BID3 ), one can design an offline optimization algorithm by designing and analyzing its counterpart in online learning.

Therefore, we would like to give a brief review of ONLINE LEARNING and OPTIMISTIC-ONLINE LEARNING.

In the typical setting of online learning, there is a LEARNER playing an action and then receiving a loss function in each round t.

Specifically, the learner plays an action w t ∈ K in round t, where w t is chosen in a compact and convex set K ⊆ R n , known as the DECISION SPACE.

Then, the learner sees the LOSS FUNCTION t (·) and suffers loss t (w t ) for the choice.

No distributional assumption is made on the loss functions sequence { 1 (·), 2 (·), . . .

, T (·)} in ONLINE LEARNING.

Namely, the loss functions can be adversarial.

The goal of an online learner is minimizing its REGRET, which is DISPLAYFORM0 We can also define AVERAGE REGRET as Regret T := DISPLAYFORM1 , which is REGRET divided by number of rounds T .

In ONLINE LEARNING literature, NO-REGRET ALGORITHMS means online learning algorithms satisfying Regret T → 0 as T → ∞.

In recent years, there is a branch of works in the paradigm of OPTIMISTIC ONLINE LEARNING (e.g. BID4 ; Rakhlin & Sridharan (2013) ; Syrgkanis et al. (2015) ; BID0 ).

The idea of OPTIMISTIC ONLINE LEARNING is as follows.

Suppose that, in each round t, the learner has a good guess m t (·) of the loss function t (·) before playing an action w t . (Recall that the learner receives the loss function after the learner commits an action!) Then, the learner should exploit the guess m t (·) to choose an action w t , as m t (·) is close to the true loss function t (·).

DISPLAYFORM0 where R(·) is a 1-strong convex function with respect to a norm ( · ) on the constraint set K, L t−1 := t−1 s=1 g s is the cumulative sum of gradient vectors of the convex loss functions (i.e. g s := ∇ s (w s ) ) up to but not including t, and η is a parameter.

The OPTIMISTIC-FTRL of (Syrgkanis et al. (2015) ) has update DISPLAYFORM1 where m t is the learner's guess of the gradient vector g t := ∇ t (w t ).

Under the assumption that loss functions are convex, the regret of OPTIMISTIC-FTRL satisfies Regret T ≤ O( T t=1 g t − m t * ), which can be much smaller than O( √ T ) of FTRL if m t is close to g t .

Consequently, OPTIMISTIC-FTRL will have much smaller regret than FTRL.

On the other hand, if m t is far from g t , then the regret of OPTIMISTIC-FTRL would be a constant factor worse than that of FTRL without optimistic update.

In the later section, we provide a way to get m t .

Here, we just want to emphasize the importance of leveraging a good guess m t for updating w t to get a fast convergence rate (or equivalently, small regret).

We also note that the works of OPTIMISTIC ONLINE LEARNING Chiang et al. (2012) ; Rakhlin & Sridharan (2013); Syrgkanis et al. (2015) ) has been shown to accelerate the convergence of some zero-sum games.

ADAM BID13 ) is a very popular algorithm for training deep nets.

It combines the momentum idea BID20 with the idea of ADAGRAD BID7 ), which has individual learning rate for different dimensions.

The learning rate of ADAGRAD in iteration t for a dimension j is proportional to the inverse of Σ t s=1 g s [j] 2 , where g s [j] is the j th element of the gradient vector g s in time s. This adaptive learning rate may help for accelerating the convergence when the gradient vector is sparse BID7 ).

However, when applying ADAGRAD to train deep nets, it is observed that the learning rate might decay too fast BID13 Get mini-batch stochastic gradient vector g t ∈ R d at w t .5: DISPLAYFORM0 6: DISPLAYFORM1 8: DISPLAYFORM2 (element-wise division) 9: end for Ba (2015)) proposes using a moving average of gradients (element-wise) divided by the root of the second moment of the moving average to update model parameter w (i.e. line 5,6 and line 8 of Algorithm 1).

Yet, ADAM BID13 ) fails at some online convex optmization problems.

AMSGRAD (Reddi et al. (2018) ) fixes the issue.

The algorithm of AMSGRAD is shown in Algorithm 1.

The difference between ADAM and AMSGRAD lies on line 7 of Algorithm 1.

ADAM does not have the update of line 7. (Reddi et al. (2018) ) adds the step to guarantee a nonincreasing learning rate, ηt √v t .

which helps for the convergence (i.e. average regret Regret T → 0.)

For the parameters of AMSGRAD, it is suggested that β 1 = 0.9, β 2 = 0.99 and η t = η/ √ t for a number η.

As mentioned in the introduction, BID5 proposed one version of optimistic algorithm for ADAM, which is referred to as ADAM-DISZ in this paper.

BID5 did not propose an optimistic algorithm for AMSGRAD but such an extension is straightforward which is referred to as AMSGRAD-DISZ.In this section, we propose a new algorithm for training deep nets: NEW-OPTIMISTIC-AMSGRAD, shown in Algorithm 2.

NEW-OPTIMISTIC-AMSGRAD has an optimistic update, which is line 9 of Algorithm 2.

It exploits the guess m t+1 of g t+1 to get w t+1 , since the vector h t+1 uses m t+1 .

Notice that the gradient vector is computed at w t instead of w t− 1 2 and the moving average of gradients is used to update w t+ 1 2.

One might want to combining line 8 and line 9and getting a single line: DISPLAYFORM0 .

From this, we see that w t+1 is updated from w t− 1 2 instead of w t .

Therefore, while NEW-OPTIMISTIC-AMSGRAD looks like just doing an additional update compared to AMSGRAD, the difference of update is subtle.

We also want to emphasize that although the learning rate on line 9 contains 4 1−β1 factor.

We suspect that it is due to the artifact of our theoretical analysis.

In our experiments, the learning rate on line 9 does not have the factor of 4 1−β1 .

That is, in practice, we implement line 9 as w t+1 = w t+ DISPLAYFORM1 We leave closing the gap between theory and practice as a future work.

We see that NEW-OPTIMISTIC-AMSGRAD inherits three properties• Adaptive learning rate of each dimension as ADAGRAD BID7 ). (line 8)• Exponentially moving average of the past gradients as NESTEROV'S METHOD BID19 ) and the HEAVY-BALL method BID20 The first property helps acceleration when the gradient has sparse structure.

The second one is the well-recognized idea of momentum which can achieve acceleration.

The last one, perhaps less known outside the ONLINE LEARNING community, can actually achieve acceleration when the prediction of the next gradient is good.

We are going to elaborate this property in the later section where we give the theoretical analysis of NEW-OPTIMISTIC-AMSGRAD.To obtain m t , we use the extrapolation algorithm of (Scieur et al. (2016) ).

Extrapolation studies estimating the limit of sequence using the last few iterates BID1 ).

Some classical works include Anderson acceleration (Walker & Ni. (2011) ), minimal polynomial extrapolation BID2 ), reduced rank extrapolation BID8 ).

These method typically assumes that the sequence {x t } ∈ R d has a linear relation DISPLAYFORM2 for an unknown matrix A ∈ R d×d (not necessarily symmetric).

The goal is to use the last few iterates {x t } to estimate the fixed point x * on (4). (Scieur et al. (2016) ) adapt the classical extrapolation methods to the iterates/updates of Get mini-batch stochastic gradient vector g t ∈ R d at w t .5: DISPLAYFORM3 6: DISPLAYFORM4 DISPLAYFORM5 Obtain z by solving (U U + λI)z = 1.

Get c = z/(z 1) 5: Output: Σ r−1 i=0 c i x i , the approximation of the fixed point x * .an optimization algorithm and propose an algorithm that produces a solution that is better than the last iterate of the underlying optimization algorithm in practice.

The algorithm of (Scieur et al. FORMULA0 ) (shown in Algorithm 3) allows the iterates {x t } to be nonlinear DISPLAYFORM0 where e t is a second order term (namely, satisfying e t 2 = O( x t−1 − x * 2 2 ).

Some theoretical guarantees regarding the distance between the output and x * is provided in (Scieur et al. (2016) ).In NEW-OPTIMISTIC-AMSGRAD, we use Algorithm 3 to get m t .

Specifically, m t is obtained by• Call Algorithm 3 with input being a sequence of some past r + 1 gradients, {g t , g t−1 , g t−2 , . . .

, g t−r } to obtain m t , where r is a parameter.• Set m t := Σ r−1 i=0 c i g t−r+i from the output of Algorithm 3.If the past few gradients can be modeled by FORMULA9 approximately, the extrapolation method should be expected to work well in predicting the gradient.

In practice, it helps to achieve faster convergence.

NEW-OPTIMISTIC-ADAM By removing line 7 in Algorithm 2, the step of making monotone weighted second moment, we obtain an algorithm which we call it NEW-OPTIMISTIC-ADAM, as the resulting algorithm can be viewed as an OPTIMISTIC-variant of ADAM.

We provide the regret analysis here.

We denote the Mahalanobis norm · H = ·, H· for some PSD matrix H. For the PSD matrix diag{v t }, where diag{v t } represents the diagonal matrix such that its i th diagonal element isv t [i] in Algorithm 2, we define the the corresponding Mahalanobis norm · ψt := ·, diag{v t } 1/2 · , where we use the notation ψ t to represent the matrix diag{v t } 1/2 .

We can also define the the corresponding dual norm DISPLAYFORM0 We assume that the model parameter w is in d-dimensional space.

That is, w ∈ R d .

Also, the analysis of NEW-OPTIMISTIC-AMSGRAD is for unconstrained optimization.

Thus, we assume that the constraint K of the benchmark in the regret definition, min w∈K T t=1 t (w), is a finite norm ball that contains the optimal solutions to the underlying offline unconstrained optimization problem.

Now we can conduct our analysis.

First of all, we can decompose the regret as follows.

DISPLAYFORM1 where the first inequality is by assuming that the loss function t (·) is convex and that we use the notation g t := ∇ t (w t ) which we adopt throughout the following proof for brevity.

Given the decomposition, let us analyze the first term DISPLAYFORM2 Astute readers may realize that the bound in Lemma 1 is actually the bound of AMSGRAD.

Indeed, since in online learning setting the loss vectors g t come adversarially, it does matter how g t is generated.

Therefore, the regret of DISPLAYFORM3 − w * , g t can be bounded in the same way as AMSGRAD.

In Appendix B, we provide the detail proof of Lemma 1.

Now we switch to bound the other sums DISPLAYFORM4 , h t in (6).

The proof is available in Appendix C. DISPLAYFORM5 Combining (6) and Lemma 1 and 2 leads to DISPLAYFORM6 Now we can conclude the following theorem.

The proof is in Appendix D. DISPLAYFORM7 One should compare the bound with that of AMSGRAD (Reddi et al. (2018) ), which is DISPLAYFORM8 where η t = η/ √ t in their setting.

We need to compare the last two terms in (8) with the last term in (9).

We are going to show that, under certain conditions, the bound is smaller than that of AMSGRAD.

Let us suppose that g t is close to m t so that DISPLAYFORM9 is much smaller than the last term of (8).

Yet, the last term of (8), DISPLAYFORM10 ψt−1) * , might be actually o( √ T ) and consequently might also be smaller than the last term of (9).

To see this, let us rewrite DISPLAYFORM11 Assume that if each DISPLAYFORM12 in the inner sum is bounded by a constant c, we will get DISPLAYFORM13 Yet, the denominator v t−1 is non-decreasing so that we can actually have a smaller bound.

That is, in practice, v t−1 [i] might grow over time, and the growth rate is different for different dimension i. DISPLAYFORM14 then the last term is just O(log T ), which might be better than that of the last term on (9).

One can also get a data dependent bound of the last term of (8).

To summarize, when m t is close to g t , NEW-OPTIMISTIC-AMSGRAD can have a smaller regret (thus better convergence rate) than ADAM (Kingma & Ba FORMULA0 ) and AMSGRAD (Reddi et al. (2018) ).

Algorithm 4 ADAM-DISZ BID5 1: Required: parameter β 1 , β 2 , and η t .

2: Init: DISPLAYFORM0 Get mini-batch stochastic gradient vector g t ∈ R d at w t .5: DISPLAYFORM1 6: DISPLAYFORM2 7: DISPLAYFORM3 We are aware of Algorithm 1 in BID5 , which was also motivated by OPTIMISTIC ONLINE LEARN-ING 2 .

For comparison, we replicate ADAM-DISZ in Algorithm 4.

We are going to describe the differences of the algorithms and the differences of the contributions between our work and their work.

FORMULA0 ) showing that if both players use some kinds of OPTIMISTIC-update, then acceleration to the convergence of the minimax value of the game is possible.

BID5 was inspired by these related works and showed that OPTIMISTIC-MIRROR-DESCENT can avoid the cycle behavior in a bilinear zero-sum game, which accelerates the convergence.

Our work is about solving min x f (x) (e.g. empirical risk) quickly.

We also show that ADAM-DISZ suffers the non-convergence issue as ADAM.

The proof is available in Appendix E.Theorem 2.

There exists a convex online learning problem such that ADAM-DISZ has nonzero average regret (i.e.

One might wonder if the non-convergence issue can be avoided if one let the weighted second moment of ADAM-DISZ be monotone by adding the stepv t = max(v t−1 , v t ) as AMSGRAD, which we call it AMSGRAD-DISZ.

Unfortunately, we are unable to prove if the step guarantees convergence or not.

To demonstrate the effectiveness of our proposed method, we test its performance with various neural network architectures, including fully-connected neural networks, convolutional neural networks (CNN's) and recurrent neural networks (RNN's).

The results illustrate that NEW-OPTIMISTIC-AMSGRAD is able to speed up the convergence of state-of-art AMSGRAD algorithm, making the learning process more efficient.

For AMSGRAD algorithm, we set the parameter β 1 and β 2 , respectively, to be 0.9 and 0.999, as recommended in Reddi et al. (2018) .

We tune the learning rate η over a fine grid and report the results under best-tuned parameter setting.

For NEW-OPTIMISTIC-AMSGRAD and AMSGRAD-DISZ, we use same β 1 , β 2 and learning rate as those for AMSGRAD to make a fair comparison of the enhancement brought by the optimistic step.

We use the same weight initialization for all algorithms.

The remaining tuning parameter of NEW-OPTIMISTIC-AMSGRAD is r, the number of previous gradients that we use to predict the next move.

We conduct NEW-OPTIMISTIC-AMSGRAD with different values of r and observe similar performance (See Appendix A).

Hence, we report r = 15 for all experiments for tidiness of plots.

To follow previous works of Reddi et al. (2018) and BID13 , we compare different methods on MNIST, CIFAR10 and IMDB datasets in our experiment.

For MNIST, we use a noisy version named as MNIST-back-rand in BID14 to increase the training difficulty.

Our experiments start with fully connected neural network for multi-class classification problems.

MNIST-back-rand dataset consists of 12000 training samples and 50000 test samples, where random background is inserted in original MNIST hand written digit images.

The input dimension is 784 (28×28) and the number of classes is 10.

We investigate a multi-layer neural networks with input layer followed by a hidden layer with 200 cells, which is then connected to a layer with 100 neurons before the output layer.

All hidden layer cells are rectifier linear units (ReLu's).

We use mini-batch size 128 to calculate stochastic gradient in each iteration.

Model performance is evaluated by multi-class cross entropy loss.

The training loss with respect to number of iterations is reported in FIG0 .

On this dataset, we empirically observe obvious improvement of NEW-OPTIMISTIC-AMSGRAD in terms of both convergence speed and training loss.

On the other hand, AMSGRAD-DISZ performs similarly to AMSGRAD in general.

Convolutional Neural Networks (CNN) have been widely studied and is important in various deep learning applications such as computer vision and natural language processing.

We test the effectiveness of NEW-OPTIMISTIC-AMSGRAD in deep CNN's with dropout.

We use the CIFAR10 dataset, which includes 60,000 images (50,000 for training and 10,000 for testing) of size 32 × 32 in 10 different classes.

ALL-CNN architecture proposed in Springenberg et al. FORMULA0 is implemented with two blocks of 3 × 3 convolutional filter, 3 × 3 convolutional layer with stride 2 and dropout layer with keep probability 0.5.

Another block of 3 × 3, 1 × 1 convolutional layer and a 6 × 6 global averaging pooling is added before the output layer.

We apply another dropout with keep probability 0.8 on the input layer.

The cost function is multi-class cross entropy.

The batch size is 128.

The images are all whitened.

The training loss is provided in FIG5 .

The result shows that NEW-OPTIMISTIC-AMSGRAD accelerates the learning process significantly and gives lowest training cost after 10000 iterations.

For this dataset, the performance of AMSGRAD-DISZ is worse than original AMSGRAD.

As another important application of deep learning, natural language processing tasks often benefit from considering sequence dependency in the models.

Recurrent Neural Networks (RNN's) achieves this goal by adding hidden state units that act as "memory".

Long-Short Term Memory (LSTM) is the most popular structure in building RNN's.

We use IMDB movie review dataset from BID16 to test the performance of NEW-OPTIMISTIC-AMSGRAD in RNN's under the circumstance of high data sparsity.

IMDB is a binary classification dataset with 25000 training and test samples respectively.

Our model includes a word embedding layer with 5000 input entries representing most frequent words in the dataset and each word is embedded into a 32 dimensional space.

The output of embedding layer is passed to 100 LSTM units, which is then connected to 100 fully connected ReLu's before reaching the output layer.

Binary cross-entropy loss is used and the batch size is 128.

We provide the results in figure 3.

We observe a considerable improvement in convergence speed.

In the first epoch, the result is more exciting.

At epoch 0.5, NEW-OPTIMISTIC-AMSGRAD already achieves the training loss that vanilla AMSGRAD can produce with more than 1 epoch.

The sample efficiency is significantly improved.

On this dataset, AMSGRAD-DISZ performs less effectively and may be trapped in local minimum.

We remark that in each iteration, only a small portion of gradients in embedding layer is non-zero.

Thus, this experiment demonstrates that NEW-OPTIMISTIC-AMSGRAD could also perform well with sparse gradient. (See additional experiments in Appendix A.)

In this paper, we propose NEW-OPTIMISTIC-AMSGRAD, which combines optimistic learning and AMSGRAD to strengthen the learning process of optimization problems, in particular, deep neural networks.

The idea of adding optimistic step can be easily extended to other optimization algorithms, e.g ADAM and ADAGRAD.

We provide OPTIMISTIC-ADAGRAD algorithm and theoretical results in Appendix F. A potential direction based on this work is to improve the method for predicting next gradient.

We expect that optimistic acceleration strategy could be widely used in various optimization problems.

In NEW-OPTIMISTIC-AMSGRAD, parameter r, the number of previous gradients used, may affect the performance dramatically.

If we choose r too small (e.g r < 10), the optimistic updates will start at very early stage, but the information we collect from the past is limited.

This may make our training process off-track at first several iterations.

On the other hand, if r is chosen to be too large (e.g r >= 30), although we may get a better prediction of next gradient, the optimistic step will start late so NEW-OPTIMISTIC-AMSGRAD may miss great chances to improve the learning performance at early stages.

Additionally, we need more room to store past gradients, hence the operational time will increase.

The empirical impact of different r value is reported in FIG7 .

We suggest 10 ≤ r ≤ 20 as an ideal range, and r = 15 tend to perform well in most training tasks.

Actually, we may make the algorithm more flexible by "early start".

For example, when we set r = 20, we can instead start adding optimistic step at iteration 10, and gradually increase the number of past gradients we use to predict next move from 10 to 20 in next 10 iterations.

After iteration 21, we fix the moving window size for optimistic prediction as 20.

This may bring enhancement to NEW-OPTIMISTIC-AMSGRAD because it can seek opportunities to accelerate learning in first several iterations, which is critical for indicating a better direction towards the minimum loss.

We also conduct experiments on NEW-OPTIMISTIC-ADAM and ADAM-DISZ.

The experiment setting is similar to NEW-OPTIMISTIC-AMSGRAD, where we fix β 1 , β 2 and compare the performance using the best learning rate with respect to ADAM.

We provide a brief summary of the results in FIG8 , with r = 15.

The improvement brought by NEW-OPTIMISTIC-ADAM is obvious, indicating that adding optimistic step could also enhance the performance of ADAM optimizer.

B PROOF OF LEMMA 1 DISPLAYFORM0 Proof.

The proof of this lemma basically follows that of (Reddi et al. (2018) ).

We have that DISPLAYFORM1 By rearranging the terms above and summing it from t = 1, . . .

, T , DISPLAYFORM2 where the first inequality is due to Young's inequality and the second one is due to the constraint 0 < β 1 ≤ 1 so that DISPLAYFORM3 where the first inequality is due to (12).Continue the analysis, we have DISPLAYFORM4 where the last equality is by telescoping sum.

DISPLAYFORM5 Proof.

From the update, w t = w t− 1 2 DISPLAYFORM6 Therefore, we have that DISPLAYFORM7 where (a) is by Hölder's inequality and (b) is by Young's inequality.

D PROOF OF THEOREM 1 DISPLAYFORM8 Proof.

Recall that from FORMULA20 , we show that DISPLAYFORM9 To proceed, let us analyze DISPLAYFORM10 above.

Notice that DISPLAYFORM11 and DISPLAYFORM12 where we use the update rule, w t = w t− 1 2 DISPLAYFORM13 .

Therefore, we have that DISPLAYFORM14 where the second inequality is due to that the sequence {v t [i]} is non-decreasing and the last equality is because DISPLAYFORM15 To summarize, DISPLAYFORM16 E PROOF OF THEOREM 2Theorem 2 There exists a convex online learning problem such that ADAM-DISZ has nonzero average regret (i.e.

We basically follow the same setting as Theorem 1 of Reddi et al. (2018) .

In each round, the loss function t (w) is linear and the learner's decision space is DISPLAYFORM0 where C ≥ 4.

For this loss function sequences, the point w = −1 achieves the minimum regret, i.e. − 1 = arg min w∈K T t=1 f t (w) when T → ∞. Consider the execution of ADAM-DISZ with DISPLAYFORM1 β 2 < 1 in this case, which satisfies the conditions of BID13 .The goal is to show that for all t, w 3t+1 = 1, and w t > 0.

Let us denoteŵ t+1 : DISPLAYFORM2 Assume the initial point is 1.

As pointed it out by Reddi et al. (2018) , the assumption of the initial point is without loss of generality.

If the initial point is not 1, then one can translate the coordinate system to a new coordinate system so that the initial point is 1 and then choose the loss function sequences for the new coordinate system.

Therefore, the base case x 1 = 1 is true.

Now assume that for some t > 0, x 3t+1 = 1 and x s > 0 for all s ≤ 3t + 1.

Observe that.∇f t (w) = C , for t mod 3 = 1 −1 , otherwise , According to the update of ADAM-DISẐ DISPLAYFORM3 So,ŵ 3t+2 = w 3t+2 .

DISPLAYFORM4 Forŵ 3t+4 , let us first consider the case thatŵ 3t+3 < 1 DISPLAYFORM5 (3t + 1)(1 − β 2 )) + η (3t + 1)(β 2 C 2 + (1 − β 2 )) + 2η (3t + 3)(β 2 C 2 + (1 − β 2 )) is a decreasing function of t and has its largest value at t = 1 for any t > 0, which is 1 − ( DISPLAYFORM6 ) ≤ 0.12.

To summarize, we have that w 3t+4 = 1.

Now if it is the case thatŵ 3t+3 ≥ 1 w 3t+4 = min(ŵ 3t+3 , 1) + 2η (3t + 3)(β 2 v 3t+2 + (1 − β 2 )) − η (3t + 3)(β 2 v 3t+1 + (1 − β 2 )) = 1 + 2η (3t + 3)(β 2 v 3t+2 + (1 − β 2 )) − η (3t + 3)(β 2 v 3t+1 + (1 − β 2 )) > 1where the last inequality is because 2 β 2 v 3t+1 + (1 − β 2 ) ≥ β 2 v 3t+2 + (1 − β 2 )To see this, β 2 v 3t+2 + (1 − β 2 ) = β 2 (β 2 v 3t+1 + (1 − β 2 )) + (1 − β 2 ) = (β 2 2 )v 3t+1 + 1 − β 2 2 (28) So, the above inequality is equivalent to DISPLAYFORM7 which is true.

This means that w 3t+4 = 1.

Therefore, we have completed the induction.

To summarize, we have 3t+1 (w 3t+1 ) + 3t+2 (w 3t+2 ) + 3t+3 (w 3t+3 ) − 3t+1 (−1) − 3t+2 (w −1 ) − 3t+3 (−1) ≥ 2C − 4.

That is, for every 3 steps, the algorithms suffers regret at least 2C − 4, this means that the regret over T rounds would be (2C − 4)T /3, which is not sublinear to T .

Now we have completed the proof.

One may follow the analysis of Theorem 2 of Reddi et al. (2018) to generalize the result so that ADAM-DISZ does not converge for any β 1 , β 2 ∈ [0, 1) such that β 1 < √ β 2 .

The optimistic update step can also be extended to other algorithms as well.

For example, based on ADAGRAD, we propose OPTIMISTIC-ADAGRAD by including the optimistic update step.

Algorithm 5 OPTIMISTIC-ADAGRAD (UNCONSTRAINED) Get current gradient g t at w t .

− ηdiag(G t ) −1/2 m t+1 , where m t+1 is the guess of g t+1 .

7: end for

We provide the regret analysis here.

Let us recall the notations and assumptions first.

We denote the Mahalanobis norm · H = ·, H· for some PSD matrix H. We let ψ t (x) := x, diag{v t } 1/2 x for the PSD matrix diag{v t }, where diag{v t } represents the diagonal matrix such that its i th diagonal element isv t [i] in Algorithm 2.

Consequently, ψ t (·) is 1-strongly convex with respect to the norm · ψt := ·, diag{v t } 1/2 · .

Namely, ψ t (·) satisfies ψ t (u) ≥ ψ t (v) + ψ t (v), u − v + (u, v) is defined as B ψt (u, v) := ψ t (u)−ψ t (v)− ψ t (v), u−v and ψ t is called the distance generating function.

We can also define the associate dual norm as ψ * t (x) := x, diag{v t } −1/2 x .

We assume that the model parameter w is in d-dimensional space.

That is, w ∈ Rd .

It suffices to analyze Algorithm 6, which holds for any convex set K. The algorithm reduces to Algorithm 5 when K = R d .

@highlight

We consider new variants of optimization algorithms for training deep nets.