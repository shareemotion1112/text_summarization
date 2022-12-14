We study the convergence of gradient descent (GD) and stochastic gradient descent (SGD) for training $L$-hidden-layer linear residual networks (ResNets).

We prove that for training deep residual networks with certain linear transformations at input and output layers, which are fixed throughout training, both GD and SGD with zero initialization on all hidden weights can converge to the global minimum of the training loss.

Moreover, when specializing to appropriate Gaussian random linear transformations, GD and SGD provably optimize wide enough deep linear ResNets.

Compared with the global convergence result of GD for training standard deep linear networks \citep{du2019width}, our condition on the neural network width is sharper by a factor of $O(\kappa L)$, where $\kappa$ denotes the condition number of the covariance matrix of the training data.

In addition, for the first time we establish the global convergence of SGD for training deep linear ResNets and prove a linear convergence rate when the global minimum is $0$.

Despite the remarkable power of deep neural networks (DNNs) trained using stochastic gradient descent (SGD) in many machine learning applications, theoretical understanding of the properties of this algorithm, or even plain gradient descent (GD), remains limited.

Many key properties of the learning process for such systems are also present in the idealized case of deep linear networks.

For example, (a) the objective function is not convex; (b) errors back-propagate; and (c) there is potential for exploding and vanishing gradients.

In addition to enabling study of systems with these properties in a relatively simple setting, analysis of deep linear networks also facilitates the scientific understanding of deep learning because using linear networks can control for the effect of architecture choices on the expressiveness of networks (Arora et al., 2018; Du & Hu, 2019) .

For these reasons, deep linear networks have received extensive attention in recent years.

One important line of theoretical investigation of deep linear networks concerns optimization landscape analysis (Kawaguchi, 2016; Hardt & Ma, 2016; Freeman & Bruna, 2016; Lu & Kawaguchi, 2017; Yun et al., 2018; Zhou & Liang, 2018) , where major findings include that any critical point of a deep linear network with square loss function is either a global minimum or a saddle point, and identifying conditions on the weight matrices that exclude saddle points.

Beyond landscape analysis, another research direction aims to establish convergence guarantees for optimization algorithms (e.g. GD, SGD) for training deep linear networks.

Arora et al. (2018) studied the trajectory of gradient flow and showed that depth can help accelerate the optimization of deep linear networks.

Ji & Telgarsky (2019) ; Gunasekar et al. (2018) investigated the implicit bias of GD for training deep linear networks and deep linear convolutional networks respectively.

More recently, Bartlett et al. (2019) ; Arora et al. (2019a) ; Shamir (2018) ; Du & Hu (2019) analyzed the optimization trajectory of GD for training deep linear networks and proved global convergence rates under certain assumptions on the training data, initialization, and neural network structure.

Inspired by the great empirical success of residual networks (ResNets), Hardt & Ma (2016) considered identity parameterizations in deep linear networks, i.e., parameterizing each layer's weight matrix as I`W, which leads to the so-called deep linear ResNets.

In particular, Hardt & Ma (2016) established the existence of small norm solutions for deep residual networks with sufficiently large depth L, and proved that there are no critical points other than the global minimum when the maximum spectral norm among all weight matrices is smaller than Op1{Lq.

Motivated by this intriguing finding, Bartlett et al. (2019) studied the convergence rate of GD for training deep linear networks with identity initialization, which is equivalent to zero initialization in deep linear ResNets.

They assumed whitened data and showed that GD can converge to the global minimum if (i) the training loss at the initialization is very close to optimal or (ii) the regression matrix ?? is symmetric and positive definite.

(In fact, they proved that, when ?? is symmetric and has negative eigenvalues, GD for linear ResNets with zero-initialization does not converge.)

Arora et al. (2019a) showed that GD converges under substantially weaker conditions, which can be satisfied by random initialization schemes.

The convergence theory of stochastic gradient descent for training deep linear ResNets is largely missing; it remains unclear under which conditions SGD can be guaranteed to find the global minimum.

In this paper, we establish the global convergence of both GD and SGD for training deep linear ResNets without any condition on the training data.

More specifically, we consider the training of L-hidden-layer deep linear ResNets with fixed linear transformations at input and output layers.

We prove that under certain conditions on the input and output linear transformations, GD and SGD can converge to the global minimum of the training loss function.

Moreover, when specializing to appropriate Gaussian random linear transformations, we show that, as long as the neural network is wide enough, both GD and SGD with zero initialization on all hidden weights can find the global minimum.

There are two main ingredients of our proof: (i) establishing restricted gradient bounds and a smoothness property; and (ii) proving that these properties hold along the optimization trajectory and further lead to global convergence.

We point out the second aspect is challenging especially for SGD due to the uncertainty of its optimization trajectory caused by stochastic gradients.

We summarize our main contributions as follows:

??? We prove the global convergence of GD and SGD for training deep linear ResNets.

Specifically, we derive a generic condition on the input and output linear transformations, under which both GD and SGD with zero initialization on all hidden weights can find global minima.

Based on this condition, one can design a variety of input and output transformations for training deep linear ResNets.

??? When applying appropriate Gaussian random linear transformations, we show that as long as the neural network width satisfies m " ???pkr?? 2 q, with high probability, GD can converge to the global minimum up to an -error within Op?? logp1{ qq iterations, where k, r are the output dimension and the rank of training data matrix X respectively, and ?? " }X} 2 2 {?? 2 r pXq denotes the condition number of the covariance matrix of the training data.

Compared with previous convergence results for training deep linear networks from Du & Hu (2019) , our condition on the neural network width is independent of the neural network depth L, and is strictly better by a factor of OpL??q.

??? Using the same Gaussian random linear transformations, we also establish the convergence guarantee of SGD for training deep linear ResNets.

We show that if the neural network width satisfies m " r ???`kr?? 2 log 2 p1{ q??n 2 {B 2??, with constant probability, SGD can converge to the global minimum up to an -error within r O`?? 2 ??1 logp1{ q??n{B??iterations, where n is the training sample size and B is the minibatch size of stochastic gradient.

This is the first global convergence rate of SGD for training deep linear networks.

Moreover, when the global minimum of the training loss is 0, we prove that SGD can further achieve linear rate of global convergence, and the condition on the neural network width does not depend on the target error .

As alluded to above, we analyze networks with d inputs, k outputs, and m ?? maxtd, ku nodes in each hidden layer.

Linear transformations that are fixed throughout training map the inputs to the first hidden layer, and the last hidden layer to the outputs.

We prove that our bounds hold with high probability when these input and output transformations are randomly generated by Gaussian distributions.

If, instead, the input transformation simply copies the inputs onto the first d components of the first hidden layer, and the output transformation takes the first k components of the last hidden layer, then our analysis does not provide a guarantee.

There is a good reason for this: a slight modification of a lower bound argument from Bartlett et al. (2019) demonstrates that GD may fail to converge in this case.

However, we describe a similarly simple, deterministic, choice of input and output transformations such that wide enough networks always converge.

The resulting condition on the network width is weaker than that for Gaussian random transformations, and thus improves on the corresponding convergence guarantee for linear networks, which, in addition to requiring wider networks, only hold with high probability for random transformations.

In addition to what we discussed above, a large bunch of work focusing on the optimization of neural networks with nonlinear activation functions has emerged.

We will briefly review them in this subsection.

It is widely believed that the training loss landscape of nonlinear neural networks is highly nonconvex and nonsmooth (e.g., neural networks with ReLU/LeakyReLU activation), thus it is fundamentally difficult to characterize the optimization trajectory and convergence performance of GD and SGD.

Some early work (Andoni et al., 2014; Daniely, 2017) showed that wide enough (polynomial in sample size n) neural networks trained by GD/SGD can learn a class of continuous functions (e.g., polynomial functions) in polynomial time.

However, those works only consider training some of the neural network weights rather than all of them (e.g., the input and output layers)

1 .

In addition, a series of papers investigated the convergence of gradient descent for training shallow networks (typically 2-layer networks) under certain assumptions on the training data and initialization scheme (Tian, 2017; Du et al., 2018b; Brutzkus et al., 2018; Zhong et al., 2017; Li & Yuan, 2017; Zhang et al., 2018) .

However, the assumptions made in these works are rather strong and not consistent with practice.

For example, Tian (2017); Du et al. (2018b); Zhong et al. (2017); Li & Yuan (2017); Zhang et al. (2018) assumed that the label of each training data is generated by a teacher network, which has the same architecture as the learned network.

Brutzkus et al. (2018) assumed that the training data is linearly separable.

Li & Liang (2018) addressed this drawback; they proved that for two-layer ReLU network with cross-entropy loss, as long as the neural network is sufficiently wide, under mild assumptions on the training data SGD with commonly-used Gaussian random initialization can achieve nearly zero expected error.

Du et al. (2018c) proved the similar results of GD for training two-layer ReLU networks with square loss.

Beyond shallow neural networks, Allen-Zhu et al. (2019); Du et al. (2019); Zou et al. (2019) generalized the global convergence results to multi-layer over-parameterized ReLU networks.

Chizat et al. (2019) showed that training over-parameterized neural networks actually belongs to a so-called "lazy training" regime, in which the model behaves like its linearization around the initialization.

Furthermore, the parameter scaling is more essential than over-paramterization to make the model learning within the "lazy training" regime.

Along this line of research, several follow up works have been conducted.

Oymak & Soltanolkotabi (2019); Zou & Gu (2019); Su & Yang (2019) ; Kawaguchi & Huang (2019) improved the convergence rate and over-parameterization condition for both shallow and deep networks.

Arora et al. (2019b) showed that training a sufficiently wide deep neural network is almost equivalent to kernel regression using neural tangent kernel (NTK), proposed in Jacot et al. (2018) .

Allen-Zhu et al. (2019); Du et al. (2019); Zhang et al. (2019) proved the global convergence for training deep ReLU ResNets.

Frei et al. (2019) proved the convergence of GD for training deep ReLU ResNets under an over-parameterization condition that is only logarithmic in the depth of the network, which partially explains why deep residual networks are preferable to fully connected ones.

However, all the results in Allen-Zhu et al. (2019) also require that all data points are separated by a positive distance and have unit norm.

As shown in Du & Hu (2019) and will be proved in this paper, for deep linear (residual) networks, there is no assumption on the training data, and the condition on the network width is significantly milder, which is independent of the sample size n. While achieving a stronger result for linear networks than for nonlinear ones is not surprising, we believe that our analysis, conducted in the idealized deep linear case, can provide useful insights to understand optimization in the nonlinear case.

We use lower case, lower case bold face, and upper case bold face letters to denote scalars, vectors and matrices respectively.

For a positive integer, we denote the set t1, . . .

, ku by rks.

Given a vector x, we use }x} 2 to denote its 2 norm.

We use N p??, ?? 2 q to denote the Gaussian distribution with mean ?? and variance ?? 2 .

Given a matrix X, we denote }X} F , }X} 2 and }X} 2,8 as its Frobenious norm, spectral norm and 2,8 norm (maximum 2 norm over its columns), respectively.

In addition, we denote by ?? min pXq, ?? max pXq and ?? r pXq the smallest, largest and r-th largest singular values of X respectively.

For a square matrix A, we denote by ?? min pAq and ?? max pAq the smallest and largest eigenvalues of A respectively.

For two sequences ta k u k??0 and tb k u k??0 , we say a k " Opb k q if a k ?? C 1 b k for some absolute constant C 1 , and use a k " ???pb k q if a k ?? C 2 b k for some absolute constant C 2 .

Except the target error , we use r Op??q and r ???p??q to hide the logarithmic factors in Op??q and ???p??q respectively.

Model.

In this work, we consider deep linear ResNets defined as follows:

where x P R d is the input, f W pxq P R k is the corresponding output, ?? ?? 0 is a scaling parameter, A P R m??d , B P R k??m denote the weight matrices of input and output layers respectively, and W 1 , . . .

, W L P R m??m denote the weight matrices of all hidden layers.

It is worth noting that the formulation of ResNets in our paper is different from that in Hardt & Ma (2016) ; Bartlett et al. (2019) , where the hidden layers have the same width as the input and output layers.

In our formulation, we allow the hidden layers to be wider by choosing the dimensions of A and B appropriately.

Loss Function.

Let tpx i , y i qu i"1,...,n be the training dataset, X " px 1 , . . .

, x n q P R d??n be the input data matrix and Y " py 1 , . . .

, y n q P R k??n be the corresponding output label matrix.

We assume the data matrix X is of rank r, where r can be smaller than d. Let W " tW 1 , . . .

, W L u be the collection of weight matrices of all hidden layers.

For an example px, yq, we consider square loss defined by

Then the training loss over the training dataset takes the following form

Algorithm.

Similarly to Allen-Zhu et al. (2019); Zhang et al. (2019) , we consider algorithms that only train the weights W for hidden layers while fixing the input and output weights A and B unchanged throughout training.

For hidden weights, we follow the similar idea in Bartlett et al. (2019) and adopt zero initialization (which is equivalent to identity initialization for standard linear network).

We would also like to point out that at the initialization, all the hidden layers automatically satisfy the so-called balancedness condition (Arora et al., 2018; 2019a; Du et al., 2018a Remark 3.2.

Theorem 3.1 can imply the convergence result in Bartlett et al. (2019) .

Specifically, in order to turn into the setting considered in Bartlett et al. (2019) , we choose m " d " k, A " I, B " I, LpW??q " 0 and XX J " I.

Then it can be easily observed that the condition in Theorem 3.1 becomes LpW p0q q??LpW??q ?? C??2.

This implies that the global convergence can be established as long as LpW p0q q??LpW??q is smaller than some constant, which is equivalent to the condition proved in Bartlett et al. (2019) .

In general, LpW p0q q??LpW??q can be large and thus the setting considered in Bartlett et al. (2019) may not be able to guarantee global convergence.

Therefore, it is natural to ask in which setting the condition on A and B in Theorem 3.1 can be satisfied.

Here we provide one possible choice which is commonly used in practice (another viable choices can be found in Section 4).

We use Gaussian random input and output transformations, i.e., each entry in A is independently generated from N p0, 1{mq and each entry in B is generated from N p0, 1{kq.

Based on this choice of transformations, we have the following proposition that characterizes the quantity of the largest and smallest singular values of A and B, and the training loss at the initialization (i.e., LpW p0q q).

The following proposition is proved in Section A.2.

Proposition 3.3.

In Algorithm 1, if each entry in A is independently generated from N p0, 1{mq and each entry in B is independently generated from N p0, 1{kq, then if m ?? C??pd`k`logp1{??qq for some absolute constant C, with probability at least 1????, it holds that

and

Then based on Theorem 3.1 and Proposition 3.3, we provide the following corollary which shows that GD is able to achieve global convergence if the neural network is wide enough.

Corollary 3.4.

Let ?? " 1{L, suppose }Y} F " Op}X} F q. Then using Gaussian random input and output transformations in Proposition 3.3, if the neural network width satisfies m " r ???pkr???? 2 q, with high probability, the output of GD in Algorithm 1 achieves training loss at most LpW??q` within T " r O`????logp1{ q??iterations, where ?? " }X} 2 2 {?? 2 r pXq denotes the condition number of the covariance matrix of training data.

Remark 3.5.

For standard deep linear networks, Du & Hu (2019) proved that GD with Gaussian random initialization can converge to a -suboptimal global minima within T " Op????logp1{ qq iterations if the neural network width satisfies m " OpLkr???? 3 q. In stark contrast, training deep linear ResNets achieves the same convergence rate as training deep linear networks and linear regression, while the condition on the neural network width is strictly milder than that for training standard deep linear networks by a factor of OpL??q.

This improvement may in part validate the empirical advantage of deep ResNets.

The following theorem establishes the global convergence of SGD for training deep linear ResNets.

Theorem 3.6.

Let ?? " 1{L. There are absolute constants C, C 1 and C 2 , such for any 0 ?? ?? ?? 1{6 and ?? 0, if the input and output weight matrices satisfy

LpW p0q q, and the step size and maximum iteration number are set as

then with probability 2 at least 1{2 (with respect to the random choices of mini batches), SGD in Algorithm 1 can find a network that achieves training loss at most LpW??q` .

By combining Theorem 3.6 and Proposition 3.3, we can show that as along as the neural network is wide enough, SGD can achieve global convergence.

Specifically, we provide the condition on the neural network width and the iteration complexity of SGD in the following corollary.

Corollary 3.7.

Let ?? " 1{L, suppose }Y} F " Op}X} F q. Then using Gaussian random input and output transformations in Proposition 3.3, for sufficiently small ?? 0, if the neural network width satisfies m " r ???`kr?? 2 log 2 p1{ q??n 2 {B 2??, with constant probability, SGD in Algorithm 1 can find a point that achieves training loss at most LpW??q` within T " r O`?? 2 ??1 logp1{ q??n{B??iterations.

From Corollaries 3.7 and 3.4, we can see that compared with the convergence guarantee of GD, the condition on the neural network width for SGD is worse by a factor of r Opn 2 log 2 p1{ q{B 2 q and the iteration complexity is higher by a factor of r Op?? ??1??n {Bq.

This is because for SGD, its trajectory length contains high uncertainty, and thus we need stronger conditions on the neural network in order to fully control it.

We further consider the special case that LpW??q " 0, which implies that there exists a ground truth matrix ?? such that for each training data point px i , y i q we have y i " ??x i .

In this case, we have the following theorem, which shows that SGD can attain a linear rate to converge to the global minimum.

Theorem 3.8.

Let ?? " 1{L. There are absolute constants C, and C 1 such that for any 0 ?? ?? ?? 1, if the input and output weight matrices satisfy

LpW p0q q, 2 One can boost this probability to 1???? by independently running logp1{??q copies of SGD in Algorithm 1.

and the step size is set as

for some maximum iteration number T , then with probability at least 1????, the following holds for all t ?? T ,

Similarly, using Gaussian random transformations in Proposition 3.3, we show that SGD can achieve global convergence for wide enough deep linear ResNets in the following corollary.

Corollary 3.9.

Let ?? " 1{L, suppose }Y} F " Op}X} F q.

In this section, we will discuss several different choices of linear transformations at input and output layers and their effects to the convergence performance.

For simplicity, we will only consider the condition for GD.

As we stated in Subsection 3.1, GD converges if the input and output weight matrices A and B

Then it is interesting to figure out what kind of choice of A and B can satisfy this condition.

In Proposition 3.3, we showed that Gaussian random transformations (i.e., each entry of A and B is generated from certain Gaussian distribution) satisfy this condition with high probability, so that GD converges.

Here we will discuss the following two other transformations.

Identity transformations.

We first consider the transformations that A " rI d??d , 0 d??pm??dq s J and B " a m{k??rI k??k , 0 k??pm??kq s. which is equivalent to the setting in Bartlett et al. (2019) when m " k " d. Then it is clear that ?? min pBq " ?? max pBq " a m{k and ?? min pAq " ?? max pAq " 1.

Now let us consider LpW p0q q. By our choices of B and A and zero initialization on weight matrices in hidden layers, in the case that d " k, we have

{2 could be as big as

F??( for example, when X and Y are orthogonal).

Then plugging these results into (4.1), the condition on A and B becomes

where the second inequality is due to the fact that LpW??q ?? }Y} 2 F {2.

Then it is clear if }X} F ?? ?

2{C, the above inequality cannot be satisfied for any choice of m, since it will be cancelled out on both sides of the inequality.

Therefore, in such cases, our bound does not guarantee that GD achieves global convergence.

Thus, it is consistent with the non-convergence results in (Bartlett et al., 2019) .

Note that replacing the scaling factor a m{k in the definition of B with any other function of d, k and m would not help.

Gaussian random initialization on hidden weights, where the input and output weights are generated by random initialization, and remain fixed throughout the training.

Modified identity transformations.

In fact, we show that a different type of identity transformations of A and B can satisfy the condition (4.1).

Here we provide one such example.

Assuming m ?? d`k, we can construct two sets S 1 , S 2 ?? rms satisfying

Then we construct matrices A and B as follows:

where ?? is a parameter which will be specified later.

In this way, it can be verified that BA " 0, ?? min pAq " ?? max pAq " 1, and ?? min pBq " ?? max pBq " ??.

Thus it is clear that the initial training loss satisfies LpW p0q q " }Y} 2 F {2.

Then plugging these results into (4.1), the condition on A and B can be rewritten as

The R.H.S. of the above inequality does not depend on ??, which implies that we can choose sufficiently large ?? to make this inequality hold.

Thus, GD can be guaranteed to achieve the global convergence.

Moreover, it is worth noting that using modified identity transformation, a neural network with m " d`k suffices to guarantee the global convergence of GD.

We further remark that similar analysis can be extended to SGD.

In this section, we conduct various experiments to verify our theory on synthetic data, including i) comparison between different input and output transformations and ii) comparison between training deep linear ResNets and standard linear networks.

To validate our theory, we performed simple experiment on 10-d synthetic data.

Specifically, we randomly generate X P R 10??1000 from a standard normal distribution and set Y "??X`0.1??E, where each entry in E is independently generated from standard normal distribution.

Consider 10-hidden-layer linear ResNets, we apply three input and output transformations including identity transformations, modified identity transformations and random transformations.

We evaluate the convergence performances for these three choices of transformations and report the results in Figures  1(a)-1(b) , where we consider two cases m " 40 and m " 200.

It can be clearly observed that gradient descent with identity initialization gets stuck, but gradient descent with modified identity initialization or random initialization converges well.

This verifies our theory.

It can be also observed that modified identity initialization can lead to slightly faster convergence rate as its initial training loss can be smaller.

In fact, with identity transformations in this setting, only the first 10 entries of the m hidden variables in each layer ever take a non-zero value, so that, no matter how large m is, effectively, m " 10, and the lower bound of Bartlett et al. (2019) applies.

Then we compare the convergence performances with that of training standard deep linear networks.

Specifically, we adopt the same training data generated in Section 5.1 and consider training Lhidden-layer neural network with fixed width m. The convergence results are displayed in Figures 1(c)-1(d) , where we consider different choices of L. For training linear ResNets, we found that the convergence performances are quite similar for different L, thus we only plot the convergence result for the largest one (e.g., L " 20 for m " 40 and L " 100 for m " 200).

However, it can be observed that for training standard linear networks, the convergence performance becomes worse as the depth increases.

This is consistent with the theory as our condition on the neural network width is m " Opkr?? 2 q (please refer to Corollary 3.4), which has no dependency in L, while the condition for training standard linear network is m " OpLkr?? 3 q (Du & Hu, 2019), which is linear in L.

In this paper, we proved the global convergence of GD and SGD for training deep linear ResNets with square loss.

More specifically, we considered fixed linear transformations at both input and output layers, and proved that under certain conditions on the transformations, GD and SGD with zero initialization on all hidden weights can converge to the global minimum.

In addition, we further proved that when specializing to appropriate Gaussian random linear transformations, GD and SGD can converge as long as the neural network is wide enough.

when W is staying inside a certain region.

Its proof is in Section B.1.

Lemma A.1.

Let ?? " 1{L, then for any weight matrices satisfying max lPrLs }W l } 2 ?? 0.5, it holds that,

In addition, , the stochastic gradient G l in Algorithm 1 satisfies

where B is the minibatch size.

The gradient lower bound can be also interpreted as the Polyak-??ojasiewicz condition, which is essential to the linear convergence rate.

The gradient upper bound is crucial to bound the trajectory length, since this lemma requires that max lPrLs }W l } ?? 0.5.

The following lemma proves the smoothness property of the training loss function LpWq when W is staying inside a certain region.

Its proof is in Section B.2.

Lemma A.2.

Let ?? " 1{L. Then for any two collections of weight matrices, denoted by

Based on these two lemmas, we are able to complete the proof of all theorems, which are provided as follows.

Proof of Theorem 3.1.

In order to simplify the proof, we use the short-hand notations ?? A , ?? A , ?? B and ?? B to denote }A} 2 , ?? min pAq, }B} 2 and ?? min pBq respectively.

Specifically, we rewrite the condition on A and B as follows

We prove the theorem by induction on the update number s, using the following two-part inductive hypothesis:

First, it can be easily verified that this holds for s " 0.

Now, assume that the inductive hypothesis holds for s ?? t.

We first prove that max lPrLs }W ptq l } F ?? 0.5.

By triangle inequality and the update rule of gradient descent, we have

where the second inequality follows from Lemma A.1, and the third inequality follows from the inductive hypothesis.

Since ?

1??x ?? 1??x{2 for any x P r0, 1s, we further have

Under the condition that ??

{2 {?? 2 r pXq, it can be readily verified that }W ptq l } F ?? 0.5.

Since this holds for all l P rLs, we have proved Part (i) of the inductive step, i.e., max lPrLs }W ptq l } F ?? 0.5.

Induction for Part (ii): Now we prove Part (ii) of the inductive step, bounding the improvement in the objective function.

Note that we have already shown that W ptq satisfies max lPrLs }W ptq l } F ?? 0.5, thus by Lemma A.2 we have

where we use the fact that W ptq l??W pt??1q l "??????? W l LpW pl??1q q. Note that LpW pt??1q q ?? LpW p0q q and the step size is set to be

so that we have

where the second inequality is by Lemma A.1.

Applying the inductive hypothesis, we get

which completes the proof of the inductive step of Part (ii).

Thus we are able to complete the proof.

Proof of Proposition 3.3.

We prove the bounds on the singular values and initial training loss separately.

Bounds on the singular values: Specifically, we set the neural network width as m ?? 100??`amaxtd, ku`a2 logp12{??q??2

By Corollary 5.35 in Vershynin (2010), we know that for a matrix U P R d1??d2 (d 1 ?? d 2 ) with entries independently generated by standard normal distribution, with probability at least 1??2 expp??t 2 {2q, its singular values satisfy

Based on our constructions of A and B, we know that each entry of ? kB and ?

mA follows standard Gaussian distribution.

Therefore, set t " 2 a logp12{??q and apply union bound, with probability at least 1????{3, the following holds,

where we use the facts that ?? min p??Uq " ???? min pUq and ?? max p??Uq " ???? max pUq for any scalar ?? and matrix U. Then applying our choice of m, we have with probability at least 1????{3, 0.9 ?? ?? min pAq ?? ?? max pAq ?? 1.1 and 0.9 a m{k ?? ?? min pBq ?? ?? max pBq ?? 1.1 a m{k.

This completes the proof of the bounds on the singular values of A and B.

Bounds on the initial training loss: The proof in this part is similar to the proof of Proposition 6.5 in Du & Hu (2019) .

Since we apply zero initialization on all hidden layers, by Young's inequality, we have the following for any px, yq,

Since each entry of B is generated from N p0, 1{kq, conditioned on A, each component of BAx is distributed according to N p0, }Ax} Note that by our bounds of the singular values, if m ?? 100??`amaxtd, ku`a2 logp8{??q??2, we have with probability at least 1????{3, }A} 2 ?? 1.1, thus, it follows that with probability at least 1???? 1???? , }BAx} 2 2 ?? 1.21

Then by union bound, it is evident that with probability 1??n?? 1???? {3,

Set ?? 1 " ??{p3nq, suppose logp1{?? 1 q ?? 1, we have with probability at least 1??2??{3,

This completes the proof of the bounds on the initial training loss.

Applying a union bound on these two parts, we are able to complete the proof.

A.3 PROOF OF COROLLARY 3.4

Proof of Corollary 3.4.

Recall the condition in Theorem 3.1:

Then by Proposition 3.3, we know that

Note that }X} F ?? ? r}X} 2 , thus the condition (A.3) can be satisfied if m " Opkr?? 2 q, where ?? " }X} 2 2 {?? 2 r pXq.

In addition, based on the results in Proposition 3.3, it can be computed that ?? " O`kL{pm}X} 2 2 q??. Then in order to achieve -suboptimal training loss, the iteration complexity is

This completes the proof.

A.4 PROOF OF THEOREM 3.6

Proof of Theorem 3.6.

The guarantee is already achieved by W p0q if ?? LpW p0q q??LpW??q, so we may assume without loss of generality that ?? LpW p0q q??LpW??q.

Similar to the proof of Theorem 3.1, we use the short-hand notations ?? A , ?? A , ?? B and ?? B to denote }A} 2 , ?? min pAq, }B} 2 and ?? min pBq respectively.

Then we rewrite the condition on A and B, and our choices of ?? and T as follows

where we set 1 " {3 for the proof purpose.

We first prove the convergence guarantees on expectation, and then apply the Markov's inequality.

For SGD, our guarantee is not made on the last iterate but the best one.

Define E t to be the event that there is no s ?? t such that LpW ptq q??LpW??q ?? 1 .

If 1pE t q " 0, then there is an iterate W s with s ?? t that achieves training loss within 1 of optimal.

Similar to the proof of Theorem 3.1, we prove the theorem by induction on the update number s, using the following inductive hypothesis: either 1pE s q " 0 or the following three inequalities hold,

where the expectation in Part (ii) is with respect to all of the random choices of minibatches.

Clearly, if 1pE s q " 0, we have already finished the proof since there is an iterate that achieves training loss within 1 of optimal.

Recalling that ?? LpW p0q q??LpW??q, it is easy to verify that the inductive hypothesis holds when s " 0.

For the inductive step, we will prove that if the inductive hypothesis holds for s ?? t, then it holds for s " t.

When 1pE t??1 q " 0, then 1pE t q is also 0 and we are done.

Therefore, the remaining part is to prove the inductive hypothesis for s " t under the assumption that 1pE t??1 q " 1, which implies that (i), (ii) and (iii) hold for all s ?? t??1.

For Parts (i) and (ii), we will directly prove that the corresponding two inequalities hold.

For Part (iii), we will prove that either this inequality holds or 1pE t q " 0.

As we mentioned, this part will be proved under the assumption 1pE t??1 q " 1.

Besides, combining Part (i) for s " t??1 and our choice of ?? and T implies that max lPrLs }W pt??1q l } F ?? 0.5.

Then by triangle inequality, we have the following for }W

By Lemma A.1, we have

Then we have

By Part (iii) for s " t??1, we know that LpW pt??1q q ?? 2LpW p0q q. Then by Part (i) for s " t??1, it is evident that

This completes the proof of the inductive step of Part (i).

Induction for Part (ii): As we previously mentioned, we will prove this part under the assumption 1pE t??1 q " 1.

Thus, as mentioned earlier, the inductive hypothesis implies that max lPrLs }W pt??1q l } F ?? 0.5.

By Part (i) for s " t, which has been verified in (A.5), it can be proved that max lPrLs }W ptq l } F ?? 0.5, then we have the following by Lemma A.2,

(A.6) By our condition on A and B, it is easy to verify that

Then by Part (iii) for s " t??1 (A.6) yields

Taking expectation conditioning on W pt??1q gives

Note that, for i sampled uniformly from t1, ..., nu, the expectation Er}G

By Lemma A.1, we have

Plugging the above inequality into (A.9) and (A.8), we get

Recalling that ?? ?? L{p6e?? Therefore, setting the step size as

we further have (A.14) where the second inequality is by (A.13) and the last inequality is by the fact that we assume 1pE t??1 q " 1, which implies that LpW pt??1q q??LpW??q ?? 1 ?? 4?? 1 ??{?? 0 .

Further taking expectation over W pt??1q , we get

where the second inequality follows from Part (ii) for s " t??1 and the assumption that 1pE 0 q " 1.

Plugging the definition of ?? 0 , we are able to complete the proof of the inductive step of Part (ii).

Recalling that for this part, we are going to prove that either LpW ptq q ?? 2LpW p0q q or 1pE t q " 0, which is equivalent to LpW ptq q??1pE t q ?? 2LpW p0q q since LpW p0q q and LpW ptq q are both positive.

We will prove this by martingale inequality.

Let F t " ??tW p0q ,??????, W ptq u be a ??-algebra, and F " tF t u t??1 be a filtration.

We first prove that ErLpW ptq q 1pE t q|F t??1 s ?? LpW pt??1q q 1pE t??1 q. Apparently, this inequality holds when 1pE t??1 q " 0 since both sides will be zero.

Then if 1pE t??1 q " 1, by (A.14) we have ErLpW ptq q|W pt??1q s ?? LpW pt??1q q since LpW??q is the global minimum.

Therefore, ErLpW ptq q 1pE t q|F t??1 , 1pE t??1 q " 1s ?? ErLpW ptq q|F t??1 , 1pE t??1 q " 1s ?? LpW pt??1q q.

Combining these two cases, by Jensen's inequality, we further have E " log`LpW ptq q 1pE t q??|F t??1 ??? ?? log`ErLpW ptq q 1pE t q|F t??1 s?? log`LpW pt??1q q 1pE t??1 q??, which implies that tlog`LpW ptq q??1pE t q??u t??0 is a super-martingale.

Then we will upper bound the martingale difference log`LpW ptq q??1pE t q????log`LpW pt??1q q??1pE t??1 q??. Clearly this quantity would be zero if 1pE t??1 q " 0.

Then if 1pE t??1 q " 1, by (A.7) we have

By Part (i) for s " t??1, Lemma A.1, we further have (A.15) where the second inequality follows from the choice of ?? that

Using the fact that 1pE t q ?? 1 and 1pE t??1 q " 1, we further have

which also holds for the case 1pE t??1 q " 0.

Recall that tlog`LpW ptq q??1pE t q??u t??0 is a supermartingale, thus by one-side Azuma's inequality, we have with probability at least 1???? 1 , log`LpW ptq q??1pE t q???? log`LpW p0q q??`3 e??n??

BL??a 2t logp1{?? 1 q.

Setting ?? 1 " ??{T , using the fact that t ?? T and leveraging our choice of T and ??, we have with probability at least 1????{T ,

, which implies that (A.16) This completes the proof of the inductive step of Part (iii).

Note that this result holds with probability at least 1????{T .

Thus applying union bound over all iterates tW ptq u t"0,...,T yields that all induction arguments hold for all t ?? T with probability at least 1????.

Moreover, plugging our choice of T and ?? into Part (ii) gives

By Markov's inequality, we further have with probability at least 2{3, it holds that rLpW pT q q?? pW??qs??1pE t q ?? 3 1 " .

Therefore, by union bound (together with the high probability arguments of (A.16)) and assuming ?? ?? 1{6, we have with probability at least 2{3???? ?? 1{2, one of the iterates of SGD can achieve training loss within 1 of optimal.

This completes the proof.

A.5 PROOF OF COROLLARY 3.7

Proof of Corollary 3.7.

Recall the condition in Theorem 3.6: This completes the proof.

A.6 PROOF OF THEOREM 3.8

Proof of Theorem 3.8.

Similar to the proof of Theorem 3.6, we set the neural network width and step size as follows,

where ?? A , ?? A , ?? B and ?? B denote }A} 2 , ?? min pAq, }B} 2 and ?? min pBq respectively.

Different from the proof of Theorem 3.6, the convergence guarantee established in this regime is made on the last iterate of SGD, rather than the best one.

Besides, we will prove the theorem by induction on the update parameter t, using the following two-part inductive hypothesis:

Induction for Part (i) We first prove that max lPrLs }W ptq l } F ?? 0.5.

By triangle inequality and the update rule of SGD, we have

where the second inequality is by Lemma A.1, the third inequality follows from Part (ii) for all s ?? t and the fact that p1??xq 1{2 ?? 1??x{2 for all x P r0, 1s.

Then applying our choice of m implies that }W ptq l } F ?? 0.5.

Induction for Part (ii) Similar to Part (ii) and (iii) of the induction step in the proof of Theorem 3.6, we first prove the convergence in expectation, and then use Azuma's inequality to get the highprobability based results.

It can be simply verfied that where the second inequality is by logp1`xq ?? x. Then similar to the proof of Theorem 3.6, we are going to apply martingale inequality to prove this part.

Let F t " ??tW p0q ,??????, W ptq u be a ??-algebra, and F " tF t u t??1 be a filtration, the above inequality implies that where the second inequality is by (A.18), the third inequality follows from the fact that??at`b ?

t ?? b 2 {a, and the last inequality is by our choice of ?? that

Then it is clear that with probability at least 1???? 1 , (A.19) which completes the induction for Part (ii).

Similar to the proof of Theorem 3.6, (A.19) holds with probability at least 1???? 1 for a given t. Then we can set ?? 1 " ??{T and apply union bound such that with probability at least 1????, (A.19) holds for all t ?? T .

This completes the proof. (1994) ).

Let U, V P R d??d be two positive definite matrices, then it holds that ?? min pUqTrpVq ?? TrpUVq ?? ?? max pUqTrpVq.

The following Lemma is proved in Section B.3.

Lemma B.3.

Let U P R d??r be a rank-r matrix.

Then for any V P R r??k , it holds that

Proof of Lemma A.1.

Proof of gradient lower bound: We first prove the gradient lower bound.

Let

, by Lemma B.1 and the definition of LpW??q, we know that there exist a matrix ?? P R k??d such that

Therefore, based on the assumption that max lPrLs }W l } F ?? 0.5, we have

pW??LpW??q??. Note that we set ?? " 1{L. Then using the inequality p1`0.5{Lq 2L??2 ?? p1`0.5{Lq 2L ?? e, we are able to complete the proof of gradient upper bound.

Proof of the upper bound of }??? W l pW; x i , y i q} where the second inequality is by our choice ?? " 1{L, and the last inequality is by the fact that p1`0.5{Lq 2L??2 ?? e.

Proof of the upper bound of stochastic gradient: Define by B the set of training data points used to compute the stochastic gradient, then define byX and?? the stacking of tx i u iPB and ty i u iPB respectively.

Let U " BpI`?? W L q??????pI`?? W 1 qA, the minibatch stochastic gradient takes form where the second inequality is by the assumptions that ?? " 1{L and max lPrLs }W l } F ?? 0.5, and the last inequality follows from the the fact that p1`0.5{Lq 2L??2 ?? p1`0.5{Lq 2L ?? e. Note thatX and?? are constructed by stacking B columns from X and Y respectively, thus we have }X} We begin by working on the first term.

Let V " pI`?? W L q??????pI`?? W 1 q and r V " pI?? ?? W L q??????pI`?? ?? W 1 q, we have r U??U " Bp r V??VqA. Breaking down the effect of transforming V " ?? 1 j"L pI`?? W j q into r V " ?? 1 j"L pI`?? ?? W j q into the effects of replacing one layer at a time, we get r V??V "

pI`?? ?? W j q??ff and, for each l, pulling out a common factor of????

(B.6)

It can be derived that the first term V 1 satisfies,

where the first equality is by the definition of V 1 .

Now we focus on the second term V 2 of (B.6),

pI`?? W l??1 q??????pI`?? W s`1 qp ?? W s??Ws qpI`?? ?? W s??1 q??????pI`?? ?? W 1 q.

Recalling that }W l } F , } ?? W l } F ?? 0.5 for all l P rLs, by triangle inequality we have }V 2 } F ?? ?? 2 p1`0.5?? q

where the third inequality follows from the fact that p1`0.5?? q L " p1`0.5{Lq L ?? ?

e and the last inequality is by Jensen's inequality.

Next, we are going to upper bound the second term of (B. 9) where the last inequality is by Jensen's inequality and the fact that ?? " 1{L. Plugging (B.7), (B.8) and (B.9) into (B.5), we have

@highlight

Under certain condition on the input and output linear transformations, both GD and SGD can achieve global convergence for training deep linear ResNets.

@highlight

The authors study the convergence of gradient descent in training deep linear residual networks, and establish a global convergence of GD/SGD and linear convergence rates of SG/SGD.

@highlight

Study of convergence properties of GD and SGD on deep linear resnets, and proof that under certain conditions on the input and output transformations and with zero initialization, GD and SGD converges to global minima.