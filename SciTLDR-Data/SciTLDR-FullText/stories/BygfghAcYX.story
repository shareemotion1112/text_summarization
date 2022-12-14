Despite existing work on ensuring generalization of neural networks in terms of scale sensitive complexity measures, such as norms, margin and sharpness, these complexity measures do not offer an explanation of why neural networks generalize better with over-parametrization.

In this work we suggest a novel complexity measure based on unit-wise capacities resulting in a tighter generalization bound for two layer ReLU networks.

Our capacity bound correlates with the behavior of test error with increasing network sizes (within the range reported in the experiments), and could partly explain the improvement in generalization with over-parametrization.

We further present a matching lower bound for the Rademacher complexity that improves over previous capacity lower bounds for neural networks.

Deep neural networks have enjoyed great success in learning across a wide variety of tasks.

They played a crucial role in the seminal work of Krizhevsky et al. (2012) , starting an arms race of training larger networks with more hidden units, in pursuit of better test performance (He et al., 2016) .

In fact the networks used in practice are over-parametrized to the extent that they can easily fit random labels to the data (Zhang et al., 2017) .

Even though they have such a high capacity, when trained with real labels they achieve smaller generalization error.

Traditional wisdom in learning suggests that using models with increasing capacity will result in overfitting to the training data.

Hence capacity of the models is generally controlled either by limiting the size of the model (number of parameters) or by adding an explicit regularization, to prevent from overfitting to the training data.

Surprisingly, in the case of neural networks we notice that increasing the model size only helps in improving the generalization error, even when the networks are trained without any explicit regularization -weight decay or early stopping (Lawrence et al., 1998; Srivastava et al., 2014; Neyshabur et al., 2015c) .

In particular, Neyshabur et al. (2015c) observed that training on models with increasing number of hidden units lead to decrease in the test error for image classification on MNIST and CIFAR-10.

Similar empirical observations have been made over a wide range of architectural and hyper-parameter choices (Liang et al., 2017; Novak et al., 2018; Lee et al., 2018) .

What explains this improvement in generalization with over-parametrization?

What is the right measure of complexity of neural networks that captures this generalization phenomenon?Complexity measures that depend on the total number of parameters of the network, such as VC bounds, do not capture this behavior as they increase with the size of the network.

Existing works suggested different norm, margin and sharpness based measures, to measure the capacity of neural networks, in an attempt to explain the generalization behavior observed in practice (Neyshabur et al., 2015b; Keskar et al., 2017; Dziugaite & Roy, 2017; Neyshabur et al., 2017; Bartlett et al., 2017; We observe that even when after network is large enough to completely fit the training data(reference line), the test error continues to decrease for larger networks.

Middle panel: Training fully connected feedforward network with single hidden layer on CIFAR-10.

We observe the same phenomena as the one observed in ResNet18 architecture.

Right panel: Unit capacity captures the complexity of a hidden unit and unit impact captures the impact of a hidden unit on the output of the network, and are important factors in our capacity bound (Theorem 1).

We observe empirically that both average unit capacity and average unit impact shrink with a rate faster than 1/ ??? h where h is the number of hidden units.

Please see Supplementary Section A for experiments settings.

BID0 Golowich et al., 2018; BID0 .

In particular, Bartlett et al. (2017) showed a margin based generalization bound that depends on the spectral norm and 1,2 norm of the layers of a network.

However, as shown in Neyshabur et al. (2017) and in FIG6 , these complexity measures fail to explain why over-parametrization helps, and in fact increase with the size of the network.

Dziugaite & Roy (2017) numerically evaluated a generalization bound based on PAC-Bayes.

Their reported numerical generalization bounds also increase with the increasing network size.

These existing complexity measures increase with the size of the network, even for two layer networks, as they depend on the number of hidden units either explicitly, or the norms in their measures implicitly depend on the number of hidden units for the networks used in practice (Neyshabur et al., 2017)

To study and analyze this phenomenon more carefully, we need to simplify the architecture making sure that the property of interest is preserved after the simplification.

We therefore chose two layer ReLU networks since as shown in the left and middle panel of FIG0 , it exhibits the same behavior with over-parametrization as the more complex pre-activation ResNet18 architecture.

In this paper we prove a tighter generalization bound (Theorem 2) for two layer ReLU networks.

Our capacity bound, unlike existing bounds, correlates with the test error and decreases with the increasing number of hidden units, in the experimental range considered.

Our key insight is to characterize complexity at a unit level, and as we see in the right panel in FIG0 these unit level measures shrink at a rate faster than 1/ ??? h for each hidden unit, decreasing the overall measure as the network size increases.

When measured in terms of layer norms, our generalization bound depends on the Frobenius norm of the top layer and the Frobenius norm of the difference of the hidden layer weights with the initialization, which decreases with increasing network size (see FIG1 ).The closeness of learned weights to initialization in the over-parametrized setting can be understood by considering the limiting case as the number of hidden units go to infinity, as considered in Bengio et al. (2006) and BID1 .

In this extreme setting, just training the top layer of the network, which is a convex optimization problem for convex losses, will result in minimizing the training error, as the randomly initialized hidden layer has all possible features.

Intuitively, the large number of hidden units here represent all possible features and hence the optimization problem involves just picking the right features that will minimize the training loss.

This suggests that as we over-parametrize the networks, the optimization algorithms need to do less work in tuning the weights of the hidden units to find the right solution.

Dziugaite & Roy (2017) indeed have numerically evaluated a PAC-Bayes measure from the initialization used by the algorithms and state that the Euclidean distance to the initialization is smaller than the Frobenius norm of the parameters.

Nagarajan & Kolter (2017) also make a similar empirical observation on the significant role of initialization, and in fact prove an initialization dependent generalization bound for linear networks.

However they do not prove a similar generalization bound for neural networks.

Alternatively, Liang et al. (2017) suggested a Fisher-Rao metric based complexity measure that correlates with generalization behavior in larger networks, but they also prove the capacity bound only for linear networks.

Contributions: Our contributions in this paper are as follows.??? We empirically investigate the role of over-parametrization in generalization of neural networks on 3 different datasets (MNIST, CIFAR10 and SVHN) , and show that the existing complexity measures increase with the number of hidden units -hence do not explain the generalization behavior with over-parametrization.??? We prove tighter generalization bounds (Theorem 2) for two layer ReLU networks, improving over previous results.

Our proposed complexity measure for neural networks decreases with the increasing number of hidden units, in the experimental range considered (see Section 2), and can potentially explain the effect of over-parametrization on generalization of neural networks.??? We provide a matching lower bound for the Rademacher complexity of two layer ReLU networks with a scalar output.

Our lower bound considerably improves over the best known bound given in Bartlett et al. (2017) , and to our knowledge is the first such lower bound that is bigger than the Lipschitz constant of the network class.

We consider two layer fully connected ReLU networks with input dimension d, output dimension c, and the number of hidden units h. Output of a network is DISPLAYFORM0 h??d and V ??? R c??h .

We denote the incoming weights to the hidden unit i by u i and the outgoing weights from hidden unit i by v i .

Therefore u i corresponds to row i of matrix U and v i corresponds to the column i of matrix V.We consider the c-class classification task where the label with maximum output score will be selected as the prediction.

Following Bartlett et al. (2017) , we define the margin operator ?? : R c ?? [c] ??? R as a function that given the scores f (x) ??? R c for each label and the correct label y ??? [c], it returns the difference between the score of the correct label and the maximum score among other labels, i.e. DISPLAYFORM1 .

We now define the ramp loss as follows: DISPLAYFORM2 For any distribution D and margin ?? > 0, we define the expected margin loss of a predictor f (.) as DISPLAYFORM3 The loss L ?? (.) defined this way is bounded between 0 and 1.

We useL ?? (f ) to denote the empirical estimate of the above expected margin loss.

As setting ?? = 0 reduces the above to classification loss, we will use L 0 (f ) andL 0 (f ) to refer to the expected risk and the training error respectively.

For any function class F, let ?? ??? F denote the function class corresponding to the composition of the loss function and functions from class F. With probability 1 ??? ?? over the choice of the training set of size m, the following generalization bound holds for any function f ??? F (Mohri et al., 2012, Theorem 3.1): DISPLAYFORM0 where R S (H) is the Rademacher complexity of a class H of functions with respect to the training set S which is defined as: Rademacher complexity is a capacity measure that captures the ability of functions in a function class to fit random labels which increases with the complexity of the class.

DISPLAYFORM1

We will bound the Rademacher complexity of neural networks to get a bound on the generalization error.

Since the Rademacher complexity depends on the function class considered, we need to choose the right function class that only captures the real trained networks, which is potentially much smaller than networks with all possible weights, to get a complexity measure that explains the decrease in generalization error with increasing width.

Choosing a bigger function class can result in weaker capacity bounds that do not capture this phenomenon.

Towards that we first investigate the behavior of different measures of network layers with increasing number of hidden units.

The experiments discussed below are done on the CIFAR-10 dataset.

Please see Section A for similar observations on SVHN and MNIST datasets.

First layer: As we see in the second panel in FIG1 even though the spectral and Frobenius norms of the learned layer decrease initially, they eventually increase with h, with Frobenius norm increasing at a faster rate.

However the distance Frobenius norm, measured w.r.t.

initialization ( U ??? U 0 F ), decreases.

This suggests that the increase in the Frobenius norm of the weights in larger networks is due to the increase in the Frobenius norm of the random initialization.

To understand this behavior in more detail we also plot the distance to initialization per unit and the distribution of angles between learned weights and initial weights in the last two panels of FIG1 .

We indeed observe that per unit distance to initialization decreases with increasing h, and a significant shift in the distribution of angles to initial points, from being almost orthogonal in small networks to almost aligned in large networks.

This per unit distance to initialization is a key quantity that appears in our capacity bounds and we refer to it as unit capacity in the remainder of the paper.

Unit capacity.

We define ?? i = u i ??? u 0 i 2 as the unit capacity of the hidden unit i.

Second layer: Similar to first layer, we look at the behavior of different measures of the second layer of the trained networks with increasing h in the first panel of FIG1 .

Here, unlike the first layer, we notice that Frobenius norm and distance to initialization both decrease and are quite close suggesting a limited role of initialization for this layer.

Moreover, as the size grows, since the Frobenius norm V F of the second layer slightly decreases, we can argue that the norm of outgoing weights v i from a hidden unit i decreases with a rate faster than 1/ ??? h. If we think of each hidden unit as a linear separator and the top layer as an ensemble over classifiers, this means the impact of each classifier on the final decision is shrinking with a rate faster than 1/ ??? h. This per unit measure again plays an important role and we define it as unit impact for the remainder of this paper.

Unit impact.

We define ?? i = v i 2 as the unit impact, which is the magnitude of the outgoing weights from the unit i.

Motivated by our empirical observations we consider the following class of two layer neural networks that depend on the capacity and impact of the hidden units of a network.

Let W be the following restricted set of parameters: DISPLAYFORM0 We now consider the hypothesis class of neural networks represented using parameters in the set W: DISPLAYFORM1 (5) Our empirical observations indicate that networks we learn from real data have bounded unit capacity and unit impact and therefore studying the generalization behavior of the above function class can potentially provide us with a better understanding of these networks.

Given the above function class, we will now study its generalization properties.

In this section we prove a generalization bound for two layer ReLU networks.

We first bound the Rademacher complexity of the class F W in terms of the sum over hidden units of the product of unit capacity and unit impact.

Combining this with the equation FORMULA4 will give us the generalization bound.

Theorem 1.

Given a training set S = {x i } m i=1 and ?? > 0, Rademacher complexity of the composition of loss function ?? over the class F W defined in equations FORMULA6 and (5) is bounded as follows: DISPLAYFORM0 The proof is given in the supplementary Section C. The main idea behind the proof is a new technique to decompose the complexity of the network into complexity of the hidden units.

To our knowledge, all previous works decompose the complexity to that of layers and use Lipschitz property of the network to bound the generalization error.

However, Lipschitzness of the layer is a rather weak property that ignores the linear structure of each individual layer.

Instead, by decomposing the complexity across the hidden units, we get the above tighter bound on the Rademacher complexity of the two layer neural networks.

The generalization bound in Theorem 1 is for any function in the function class defined by a specific choice of ?? and ?? fixed before the training procedure.

To get a generalization bound that holds for all networks, it suffices to cover the space of possible values for ?? and ?? and take a union bound over it.

The following theorem states the generalization bound for any two layer ReLU network 2 .Theorem 2.

For any h ??? 2, ?? > 0, ?? ??? (0, 1) and U 0 ??? R h??d , with probability 1 ??? ?? over the choice of the training set DISPLAYFORM1 and U ??? R h??d , the generalization error is bounded as follows: DISPLAYFORM2 The above generalization bound empirically improves over the existing bounds, and decreases with increasing width for networks learned in practice (see Section 2.3).

We also show an explicit lower bound for the Rademacher complexity (Theorem 3), matching the first term in the above generalization bound, thereby showing its tightness.

The additive factor??( h/m) in the above bound is the result of taking the union bound over the cover of ?? and ??.

As we see in FIG6 , in the regimes of interest this additive term is small and does not dominate the first term, resulting in an overall decrease in capacity with over-parametrization.

In Appendix Section B, we further extend the generalization bound in Theorem 2 to p norms, presenting a finer tradeoff between the two terms.

In TAB1 with respect to the size of the network trained on CIFAR-10.

BID1 Golowich et al., 2018) : The first term in their bound U 2 V ??? V 0 1,2 is of smaller magnitude and behaves roughly similar to the first term in our bound U 0 2 V F (see FIG2 last two panels).

The key complexity term in their bound is U ??? U 0 1,2 V 2 , and in our bound is U ??? U 0 F V F , for the range of h considered.

V 2 and V F differ by number of classes, a small constant, and hence behave similarly.

However, U ??? U 0 1,2 can be as big as DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 when most hidden units have similar capacity.

Infact their bound increases with h mainly because of this term U ??? U 0 1,2 .

As we see in the first and second panels of Experimental comparison.

We train two layer ReLU networks of size h on CIFAR-10 and SVHN datasets with values of h ranging from 2 6 to 2 15 .

The training and test error for CIFAR-10 are shown in the first panel of FIG0 , and for SVHN in the left panel of FIG4 .

We observe for both datasets that even though a network of size 128 is enough to get to zero training error, networks with sizes well beyond 128 can still get better generalization, even when trained without any regularization.

We further measure the unit-wise properties introduce in the paper, namely unit capacity and unit impact.

These quantities decrease with increasing h, and are reported in the right panel of FIG0 and second panel of FIG4 .

Also notice that the number of epochs required for each network size to get 0.01 cross-entropy loss decreases for larger networks as shown in the third panel of FIG4 .For the same experimental setup, FIG6 compares the behavior of different capacity bounds over networks of increasing sizes.

Generalization bounds typically scale as C/m where C is the effective capacity of the function class.

The left panel reports the effective capacity C based on different measures calculated with all the terms and constants.

We can see that our bound is the only that decreases with h and is consistently lower that other norm-based data-independent bounds.

Our bound even improves over VC-dimension for networks with size larger than 1024.

While the actual numerical values are very loose, we believe they are useful tools to understand the relative generalization behavior with respect to different complexity measures, and in many cases applying a set of data-dependent techniques, one can improve the numerical values of these bounds significantly (Dziugaite & Roy, 2017; BID0 each capacity bound normalized by its maximum in the range of the study for networks trained on CIFAR-10 and SVHN respectively.

For both datasets, our capacity bound is the only one that decreases with the size even for networks with about 100 million parameters.

All other existing norm-based bounds initially decrease for smaller networks but then increase significantly for larger networks.

Our capacity bound therefore could potentially point to the right properties that allow the over-parametrized networks to generalize.

Finally we check the behavior of our complexity measure under a different setting where we compare this measure between networks trained on real and random labels (Neyshabur et al., 2017; Bartlett et al., 2017) .

We plot the distribution of margin normalized by our measure, computed on networks trained with true and random labels in the last panel of FIG4 -and as expected they correlate well with the generalization behavior.

In this section we will prove a lower bound for the Rademacher complexity of neural networks, that matches the dominant term in the upper bound of Theorem 1.

We will show our lower bound on a smaller function class than F W , with an additional constraint on spectral norm of the hidden layer.

This allows for comparison with the existing results, and also extends the lower bound to the bigger class F W .

Theorem 3.

Define the parameter set DISPLAYFORM0 and let F W be the function class defined on W by equation (5).

Then, for any DISPLAYFORM1 Clearly, W ??? W, since it has an extra constraint.

The complete proof is given in the supplementary Section C.3.Published as a conference paper at ICLR 2019The above complexity lower bound matches the first term, DISPLAYFORM2 , in the upper bound of Theorem 1, upto 1 ?? , which comes from the 1 ?? -Lipschitz constant of the ramp loss l ?? .

To match the second term in the upper bound for Theorem 1, consider the setting with c = 1 and ?? = 0, resulting in, DISPLAYFORM3 where DISPLAYFORM4 In other words, when ?? = 0, the function class DISPLAYFORM5 , and therefore we have the above lower bound, showing that the upper bound provided in Theorem 1 is tight.

It also indicates that even if we have more information, such as bounded spectral norm with respect to the reference matrix is small (which effectively bounds the Lipschitz of the network), we still cannot improve our upper bound.

To our knowledge, all the previous capacity lower bounds for spectral norm bounded classes of neural networks with a scalar output and element-wise activation functions correspond to the Lipschitz constant of the network.

Our lower bound strictly improves over this, and shows a gap between the Lipschitz constant of the network (which can be achieved by even linear models), and the capacity of neural networks.

This lower bound is non-trivial, in the sense that the smaller function class excludes the neural networks with all rank-1 matrices as weights, and thus shows a ??( ??? h)-capacity gap between the neural networks with ReLU activations and linear networks.

The lower bound therefore does not hold for linear networks.

Finally, one can extend the construction in this bound to more layers by setting all the weight matrices in the intermediate layers to be the Identity matrix.

for the function class defined by the parameter set: DISPLAYFORM6 Note that s 1 s 2 is the Lipschitz bound of the function class F Wspec .

Given W spec with bounds s 1 and s 2 , choosing ?? and ?? such that ?? 2 = s 1 and max i???[h]

?? i = s 2 results in W ??? W spec .

Hence we get the following result from Theorem 3, showing a stronger lower bound for this function class as well.

DISPLAYFORM7 Hence our result improves the lower bound in Bartlett et al. (2017) by a factor of ??? h. Theorem 7 in Golowich et al. (2018) also gives a ???(s 1 s 2 ??? c) lower bound, c is the number of outputs of the network, for the composition of 1-Lipschitz loss function and neural networks with bounded spectral norm, or ???-Schatten norm.

Our above result even improves on this lower bound.

In this paper we present a new capacity bound for neural networks that empirically decreases with the increasing number of hidden units, and could potentially explain the better generalization performance of larger networks.

In particular, we focused on understanding the role of width in the generalization behavior of two layer networks.

More generally, understanding the role of depth and the interplay between depth and width in controlling capacity of networks, remain interesting directions for future study.

We also provided a matching lower bound for the capacity improving on the current lower bounds for neural networks.

While these bounds are useful for relative comparison between networks of different size, their absolute values still remain larger than the number of training samples, and it is of interest to get bounds with numerically smaller values.

In this paper we do not address the question of whether optimization algorithms converge to low complexity networks in the function class considered in this paper, or in general how does different hyper parameter choices affect the complexity of the recovered solutions.

It is interesting to understand the implicit regularization effects of the optimization algorithms (Neyshabur et al., 2015a; Gunasekar et al., 2017; Soudry et al., 2018) for neural networks, which we leave for future work.

Below we describe the setting for each reported experiment.

In this experiment, we trained a pre-activation ResNet18 architecture on CIFAR-10 dataset.

The architecture consists of a convolution layer followed by 8 residual blocks (each of which consist of two convolution) and a linear layer on the top.

Let k be the number of channels in the first convolution layer.

The number of output channels and strides in residual blocks is then [k, k, 2k, 2k, 4k, 4k, 8k, 8k] and [1, 1, 1, 2, 1, 2, 1, 2] respectively.

Finally, we use the kernel sizes 3 in all convolutional layers.

We train 11 architectures where for architecture i we set k = 2 2+i/2 .

In each experiment we train using SGD with mini-batch size 64, momentum 0.9 and initial learning rate 0.1 where we reduce the learning rate to 0.01 when the cross-entropy loss reaches 0.01 and stop when the loss reaches 0.001 or if the number of epochs reaches 1000.

We use the reference line in the plots to differentiate the architectures that achieved 0.001 loss.

We do not use weight decay or dropout but perform data augmentation by random horizontal flip of the image and random crop of size 28 ?? 28 followed by zero padding.

We trained fully connected feedforward networks on CIFAR-10, SVHN and MNIST datasets.

For each data set, we trained 13 architectures with sizes from 2 3 to 2 15 each time increasing the number of hidden units by factor 2.

For each experiment, we trained the network using SGD with mini-batch size 64, momentum 0.9 and fixed step size 0.01 for MNIST and 0.001 for CIFAR-10 and SVHN.

We did not use weight decay, dropout or batch normalization.

For experiment, we stopped the training when the cross-entropy reached 0.01 or when the number of epochs reached 1000.

We use the reference line in the plots to differentiate the architectures that achieved 0.01 loss.

Evaluations For each generalization bound, we have calculated the exact bound including the logterms and constants.

We set the margin to 5th percentile of the margin of data points.

Since bounds in BID2 and Neyshabur et al. (2015c) are given for binary classification, we multiplied BID2 by factor c and Neyshabur et al. (2015c) by factor ??? c to make sure that the bound increases linearly with the number of classes (assuming that all output units have the same norm).

Furthermore, since the reference matrices can be used in the bounds given in Bartlett et al. (2017) and BID0 , we used random initialization as the reference matrix.

When plotting distributions, we estimate the distribution using standard Gaussian kernel density estimation.

Figures 6 and 7 show the behavior of several measures on networks with different sizes trained on SVHN and MNIST datasets respectively.

The left panel of FIG10 shows the over-parametrization phenomenon in MNSIT dataset and the middle and right panels compare our generalization bound to others.

In this section we generalize the Theorem 2 to p norm.

The main new ingredient in the proof is the Lemma 11, in which we construct a cover for the p ball with entry-wise dominance.

Theorem 5.

For any h, p ??? 2, ?? > 0, ?? ??? (0, 1) and U 0 ??? R h??d , with probability 1 ??? ?? over the choice of the training set DISPLAYFORM0 and U ??? R h??d , the generalization error is bounded as follows: DISPLAYFORM1 where .

p,2 is the p norm of the row 2 norms.

For p of order ln h, h e ???p ??? constant improves on the ??? h additive term in Theorem 2 and DISPLAYFORM2 which is a tight upper bound for V F and is of the same order if all rows of V have the same norm -hence giving a tighter bound that decreases with h for larger values.

In particular for p = ln h we get the following bound.

Corollary 6.

Under the settings of Theorem 5, with probability 1 ??? ?? over the choice of the training set S = {x i } m i=1 , for any function f (x) = V[Ux] + , the generalization error is bounded as follows: DISPLAYFORM3 We start by stating a simple lemma which is a vector-contraction inequality for Rademacher complexities and relates the norm of a vector to the expected magnitude of its inner product with a vector of Rademacher random variables.

We use the following technical result from Maurer (2016) in our proof.

Lemma 7 (Propostion 6 of Maurer FORMULA2 ).

Let ?? i be the Rademacher random variables.

For any vector v ??? R d , the following holds: DISPLAYFORM4 The above DISPLAYFORM5 Proof.

DISPLAYFORM6 (i) follows from the Jensen's inequality.

We next show that the Rademacher complexity of the class of networks defined in (5) and FORMULA6 can be decomposed to that of hidden units.

Lemma 9 (Rademacher Decomposition).

Given a training set S = {x i } m i=1 and ?? > 0, Rademacher complexity of the class F W defined in equations (5) and (4) is bounded as follows: DISPLAYFORM7 Proof.

We prove the inequality in the lemma statement using induction on t in the following inequality: DISPLAYFORM8 where for simplicity of the notation, we let ?? DISPLAYFORM9 The above statement holds trivially for the base case of t = 1 by the definition of the Rademacher complexity (3).

We now assume that it is true for any t ??? t and prove it is true for t = t + 1.

DISPLAYFORM10 The last inequality follows from the ??? 2 ?? Lipschitzness of the ramp loss.

The ramp loss is 1/?? Lipschitz with respect to each dimension but since the loss at each point only depends on score of the correct labels and the maximum score among other labels, it is ??? 2 ?? -Lipschitz.

By Lemma 7, the right hand side of the above inequality can be bounded as follows: DISPLAYFORM11 This completes the induction proof.

Lemma 10 (Ledoux-Talagrand contraction, Ledoux & Talagrand (1991) ).

Let f : R + ??? R + be convex and increasing.

Let ?? i : R ??? R satisfy ?? i (0) = 0 and be L-Lipschitz.

Let ?? i be independent Rademacher random variables.

For any T ??? R n , DISPLAYFORM12 The above lemma will be used in the following proof of Theorem 1.Proof of Theorem 1.

By Lemma 9, we have: DISPLAYFORM13 Now we can apply Lemma 10 with n = m ?? c, f DISPLAYFORM14 , and we get DISPLAYFORM15 The proof is completed by taking sum of above inequality over j from 1 to h.

We start by the following covering lemma which allows us to prove the generalization bound in Theorem 5 without assuming the knowledge of the norms of the network parameters.

The following lemma shows how to cover an p ball with a set that dominates the elements entry-wise, and bounds the size of a one such cover.

Lemma 11 ( p covering lemma).

Given any , D, ?? > 0, p ??? 2, consider the set S DISPLAYFORM0 Proof.

We prove the lemma by construction.

DISPLAYFORM1 By Lemma 11, picking = ((1 + ??) 1/p ??? 1), we can find a set of vectors, DISPLAYFORM2 Lemma 13.

For any h, p ??? 2, c, d, ??, ?? > 0, ?? ??? (0, 1) and U 0 ??? R h??d , with probability 1 ??? ?? over the choice of the training set DISPLAYFORM3 c??h and U ??? R h??d , the generalization error is bounded as follows: DISPLAYFORM4 where DISPLAYFORM5 and .

p,2 is the p norm of the column 2 norms.

Proof.

This lemma can be proved by directly applying union bound on Lemma 12 with for every C 1 ??? .

For V p,2 ??? 1 h 1/2???1/p , we can use the bound where C 1 = 1, and the additional constant 1 in Eq. 12 will cover that.

The same is true for the case of U p,2 ??? i h 1/2???1/p X F .

When any of h 1/2???1/p V p,2 and h 1/2???1/p X F U p,2 is larger than DISPLAYFORM6 , the second term in Eq. 12 is larger than 1 thus holds trivially.

For the rest of the case, there exists (C 1 , C 2 ) such that h 1/2???1/p C 1 ??? h 1/2???1/p V p,2 + 1 and h 1/2???1/p C 2 ??? h 1/2???1/p X F X F U p,2 + 1.

Finally, we have ?? ??? m 4 ??? 1 otherwise the second term in Eq. 12 is larger than 1.

Therefore, DISPLAYFORM7 We next use the general results in Lemma 13 to give specific results for the case p = 2.Lemma 14.

For any h ??? 2, c, d, ?? > 0, ?? ??? (0, 1) and U 0 ??? R h??d , with probability 1 ??? ?? over the choice of the training set S = {x i } m i=1 ??? R d , for any function f (x) = V[Ux] + such that V ??? R c??h and U ??? R h??d , the generalization error is bounded as follows: DISPLAYFORM8 Proof.

To prove the lemma, we directly upper bound the generalization bound given in Lemma 13 for p = 2 and ?? = 3 ??? 2 4 ??? 1.

For this choice of ?? and p, we have 4(?? + 1) 2/p ??? 3 ??? 2 and ln N p,h is bounded as follows:ln N p,h = ln h/?? + h ??? 2 h ??? 1 ??? ln e h/?? + h ??? 2 h ??? 1 h???1 = (h ??? 1) ln e + e h/?? ??? 1 h ??? 1 ??? (h ??? 1) ln e + e h/?? h ??? 1 ??? h ln(e + 2e/??) ??? 5hProof of Theorem 2.

The proof directly follows from Lemma 14 and using?? notation to hide the constants and logarithmic factors.

Next lemma states a generalization bound for any p ??? 2, which is looser than 14 for p = 2 due to extra constants and logarithmic factors.

Lemma 15.

For any h, p ??? 2, c, d, ?? > 0, ?? ??? (0, 1) and U 0 ??? R h??d , with probability 1 ??? ?? over the choice of the training set S = {x i } m i=1 ??? R d , for any function f (x) = V[Ux] + such that V ??? R c??h and U ??? R h??d , the generalization error is bounded as follows: DISPLAYFORM9 .

p,2 is the p norm of the column 2 norms.

Proof.

To prove the lemma, we directly upper bound the generalization bound given in Lemma 13 for ?? = e p ??? 1.

For this choice of ?? and p, we have (?? + 1) 2/p = e 2 .

Furthermore, if ?? ??? h, N p,h = 0, otherwise ln N p,h is bounded as follows: DISPLAYFORM10 = ( h/(e p ??? 1) ??? 1) ln e + e h ??? 1 h/(e p ??? 1) ??? 1 ??? ( e 1???p h ??? 1) ln (eh)Since the right hand side of the above inequality is greater than zero for ?? ??? h, it is true for every ?? > 0.Proof of Theorem 5.

The proof directly follows from Lemma 15 and using?? notation to hide the constants and logarithmic factors.

Proof of Theorem 3.

We will start with the case h = d = 2 k , m = n2 k for some k, n ??? N.We will pick V = ?? = [?? 1 . . .

?? 2 k ] for every ??, and DISPLAYFORM0 , where x i := e i n .

That is, the whole dataset are divides into 2 k groups, while each group has n copies of a different element in standard orthonormal basis.

We further define j (??) = For any ?? ??? {???1, 1} n , let Diag(??) be the square diagonal matrix with its diagonal equal to ?? and F(??) be the following: F(??) := [ f 1 , f 2 , . . .

, f 2 k ] such that if i (??) ??? 0, f i = f i , and if i (??) < 0, f i = 0, and we will choose U(??) as Diag(??) ?? F(??).Since F is orthonormal, by the definition of F(??), we have F(??) 2 ??? 1 and the 2-norm of each row of F is upper bounded by 1.

Therefore, we have U(??) 2 ??? Diag(??) 2 F(??) 2 ??? max i ?? i , and

@highlight

We suggest a generalization bound that could partly explain the improvement in generalization with over-parametrization.