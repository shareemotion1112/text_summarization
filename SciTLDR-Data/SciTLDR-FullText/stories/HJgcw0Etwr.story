To analyze deep ReLU network, we adopt a student-teacher setting in which an over-parameterized student network learns from the output of a fixed teacher network of the same depth, with Stochastic Gradient Descent (SGD).

Our contributions are two-fold.

First, we prove that when the gradient is zero (or bounded above by a small constant) at every data point in training, a situation called  \emph{interpolation setting}, there exists many-to-one \emph{alignment} between student and teacher nodes in the lowest layer under mild conditions.

This suggests that generalization in unseen dataset is achievable, even the same condition often leads to zero training error.

Second, analysis of noisy recovery and training dynamics in 2-layer network shows that strong teacher nodes (with large fan-out weights) are learned first and subtle teacher nodes are left unlearned until late stage of training.

As a result, it could take a long time to converge into these small-gradient critical points.

Our analysis shows that over-parameterization plays two roles: (1) it is a necessary condition for alignment to happen at the critical points, and (2) in training dynamics, it helps student nodes cover more teacher nodes with fewer iterations.

Both improve generalization.

Experiments justify our finding.

Deep Learning has achieved great success in the recent years (Silver et al., 2016; He et al., 2016; Devlin et al., 2018) .

Although networks with even one-hidden layer can fit any function (Hornik et al., 1989) , it remains an open question how such networks can generalize to new data.

Different from what traditional machine learning theory predicts, empirical evidence (Zhang et al., 2017) shows more parameters in neural network lead to better generalization.

How over-parameterization yields strong generalization is an important question for understanding how deep learning works.

In this paper, we analyze deep ReLU networks with teacher-student setting: a fixed teacher network provides the output for a student to learn via SGD.

Both teacher and student are deep ReLU networks.

Similar to (Goldt et al., 2019) , the student is over-realized compared to the teacher: at each layer l, the number n l of student nodes is larger than the number m l of teacher (n l > m l ).

Although over-realization is different from over-parameterization, i.e., the total number of parameters in the student model is larger than the training set size N , over-realization directly correlates with the width of networks and is a measure of over-parameterization.

The student-teacher setting has a long history (Saad & Solla, 1996; 1995; Freeman & Saad, 1997; Mace & Coolen, 1998) and recently gains increasing interest (Goldt et al., 2019; Aubin et al., 2018) in analyzing 2-layered network.

While worst-case performance on arbitrary data distributions may not be a good model for real structured dataset and can be hard to analyze, using a teacher network implicitly enforces an inductive bias and could potentially lead to better generalization bound.

Specialization, that is, a student node becomes increasingly correlated with a teacher node during training (Saad & Solla, 1996) , is one of the important topic in this setup.

If all student nodes are specialized to the teacher, then student tends to output the same as the teacher and generalization performance can be expected.

Empirically, it has been observed in 2-layer networks (Saad & Solla, 1996; Goldt et al., 2019) and multi-layer networks (Tian et al., 2019; Li et al., 2016) , in both synthetic and real dataset.

In contrast, theoretical analysis is limited with strong assumptions (e.g., Gaussian inputs, infinite input dimension, local convergence, 2-layer setting, small number of hidden nodes).

In this paper, with arbitrary training distribution and finite input dimension, we show rigorously that when gradient at each training sample is small (i.e., the interpolation setting as suggested in (Ma

Student-teacher setting.

This setting has a long history (Engel & Van den Broeck, 2001; Gardner & Derrida, 1989) .

The seminar works (Saad & Solla, 1996; 1995) studies 1-hidden layer case from statistical mechanics point of view in which the input dimension goes to infinity, or so-called thermodynamics limits.

They study symmetric solutions and locally analyze the symmetric breaking behavior and onset of specialization of the student nodes towards the teacher.

Recent follow-up works (Goldt et al., 2019) makes the analysis rigorous and empirically shows that random initialization and training with SGD indeed gives student specialization in 1-hidden layer case, which is consistent with our experiments.

With the same assumption, (Aubin et al., 2018 ) studies phase transition property of specialization in 2-layer networks with small number of hidden nodes using replica formula.

In these works, inputs are assumed to be Gaussian and step or Gauss error function is used as nonlinearity.

Few works study teacher-student setting with more than two layers. (Allen-Zhu et al., 2019a) shows the recovery results for 2 and 3 layer networks, with modified SGD, batchsize 1 and heavy over-parameterization.

In comparison, our work shows that specialization happens around the SGD critical points in the lowest layer for deep ReLU networks, without any parametric assumptions of input distribution.

Local minima is Global.

While in deep linear network, all local minima are global (Laurent & Brecht, 2018; Kawaguchi, 2016) , situations are quite complicated with nonlinear activations.

While local minima is global when the network has invertible activation function and distinct training samples (Nguyen & Hein, 2017; Yun et al., 2018) or Leaky ReLU with linear separate input data (Laurent & von Brecht, 2017) , multiple works (Du et al., 2018a; Ge et al., 2017; Safran & Shamir, 2017; Yun et al., 2019) show that in GD case with population or empirical loss, spurious local minima can happen even in two-layered network.

Many are specific to two-layer and hard to generalize to multi-layer setting.

In contrast, our work brings about a generic formulation for deep ReLU network and gives recovery properties in the student-teacher setting.

Learning wild networks.

Recent works on Neural Tangent Kernel (Jacot et al., 2018; Du et al., 2018b; Allen-Zhu et al., 2019b) show the global convergence of GD for multi-layer networks with infinite width. (Li & Liang, 2018) shows the convergence in one-hidden layer ReLU network using GD/SGD to solution with good generalization, when the input data are assumed to be clustered into classes.

Both lines of work assume heavily over-parameterized network, requiring polynomial growth of number of nodes with respect to the number of samples. (Chizat & Bach, 2018) shows global convergence of over-parameterized network with optimal transport. (Tian et al., 2019) assumes mild over-realization and gives convergence results for 2-layer network when a subset of the student network is close to the teacher.

Our work extends it with much weaker assumptions.

Deep Linear networks.

For deep linear networks, multiple works (Lampinen & Ganguli, 2019; Saxe et al., 2013; Arora et al., 2019; Advani & Saxe, 2017) have shown interesting training dynamics.

One common assumption is that the singular spaces of weights at nearby layers are aligned at initialization, which decouples the training dynamics.

Such a nice property would not hold for nonlinear network. (Lampinen & Ganguli, 2019) shows that under this assumption, weight components with large singular value are learned first, while we analyze and observe empirically similar behaviors on the student node level.

Generalization property of linear networks can also be analyzed in the limit of infinite input dimension with teacher-student setting (Lampinen & Ganguli, 2019) .

However, deep linear networks lack specialization which plays a crucial role in the nonlinear case.

To our knowledge, we are the first to analyze specialization rigorously in deep ReLU networks.

Notation.

Consider a student network and its associated teacher network ( Fig. 1(a) ).

Denote the input as x. We focus on multi-layered networks with ??(??) as ReLU nonlinearity.

We use the following equality extensively: ??(x) = I[x > 0]x, where I[??] is the indicator function.

For node j, f j (x), z j (x) and g j (x) are its activation, gating function and backpropagated gradient after the gating.

Both teacher and student networks have L layers.

The input layer is layer 0 and the topmost layer (layer that is closest to the output) is layer L.

For layer l, let m l be the number of teacher node while n l be the number of student node.

The weights W l ??? R n l???1 ??n l refers to the weight matrix that connects layer l ??? 1 to layer l on the student side.

W l = [w l,1 , w l,2 , . . . , w l,n l ] where each w ??? R n l???1 is the weight vector.

Similarly we have teacher weight

n l ??n l be the diagonal matrix of gating function (for ReLU it is either 0 or 1), and g l (x) = [g l,1 (x), . . .

, g l,n l (x)]

??? R n l be the backpropated gradient vector.

By definition, the input layer has f 0 (x) = x ??? R n0 and m 0 = n 0 .

Note that f l (x), g l (x) and D l (x) are all dependent on W. For brevity, we often use f l (x) rather than f l (x; W).

* are from the teacher, only dependent on the teacher and remains the same throughout the training.

D * L (x) = D L (x) ??? I C??C since there is no ReLU gating.

Note that C is the dimension of output for both teacher and student.

With the notation, gradient descent is:

In SGD, the expectation E x [??] is taken over a batch.

In GD, it is over the entire dataset.

Bias term.

With the same notation we can also include the bias term.

In this case,

Objective.

We assume that both the teacher and the student output a vector.

We use the output of teacher as the input of the student and the objective is:

We want to ask the following qeustion:

Are student nodes specialized to teacher nodes at the same layers after training?

One might wonder this is hard since the student's intermediate layer receives no direct supervision from the corresponding teacher layer, but relies only on backpropagated gradient.

Surprisingly, the following theorem shows that it is possible for every intermediate layer:

Lemma 1 (Recursive Gradient Rule).

At layer l, the backpropagated g l (x) satisfies

where the mixture coefficient

C??m l are defined in a top-down manner:

For convenience, we can write

Let R 0 = {x : ??(x) > 0} be the infinite training set, where ??(x) is the input data distribution.

Let R l = {f l (x) : x ??? R 0 }, which is the image of the training set at the output of layer l, and also a convex polytope.

Then the mixture coefficient A l (x) and B l (x) have the following property: Corollary 1 (Piecewise constant).

R 0 can be decomposed into a finite (but potentially exponential) set of regions

We first show that due to property of ReLU node and subset sampling in SGD, at SGD critical point, under mild condition, the teacher node aligns with at least one student node and the goal (*) can be reached in the lowest layer.

Definition 1 (SGD critical point).?? is a SGD critical point if for any batch,

Such critical points exist since over-realized student can mimic teacher perfectly.

Note that critical points in SGD is much stronger than those in GD, where the gradient is always averaged at a fixed data distribution.

If f l???1 has a bias term (and

which is global optimum with zero training loss.

In the following, we want to check whether this condition leads to specialization, i.e., whether the teacher's weights are recovered/aligned by the student, i.e., whether for teacher j, there exists a student k at the same layer so that w k = ??w j for some ?? > 0.

Note that g l (x i ;??) = 0 might be a strong assumption since in practice the gradient is small but never zero.

A weaker assumption is that g l (x i ;??) ??? ??? or even E t g l (x i ;??) ??? ??? .

For this, Theorem 5 shows (approximate) alignment/specialization still holds for noisy case.

Obviously, an arbitrary teacher network won't be reconstructed.

A trivial example is that a teacher network always output 0 since all the training samples lie in the inactive halfspace of its ReLU nodes.

Therefore, we need to impose condition on the teacher network.

Let E j = {x : f j (x) > 0} be the activation region of node j. Note that the halfspace E j is an open set.

Let ???E j = {x : f j (x) = 0} be the decision boundary of node j. Definition 2 (Observer).

Node k is an observer of node j if E k ??? ???E j = ???. Assumption 1 (Teacher Network).

For each layer l, we require that (1) the teacher weights w * l,j

are not co-linear.

and (2) the boundary of w * l,j is visible in the training set:

Assumption 1 is our assumption of the teacher.

The first requirement is trivial.

The second one is reasonable since two teacher nodes who behaves linearly in the training set are indistinguishable.

We first start with 2-layer case, in which A 1 (x) and B 1 (x) are constant with respect to x, since there is no ReLU gating at the top layer l = 2.

In this case, from the SGD critical point at l = 1,

The intuition is that if the input x takes sufficiently diverse values, ReLU activations ??(w k x) can be proven to be mutually linear independent.

On the other hand, the gradient of each student node k when active, is ?? k f 1 (x) ??? b k f 1 (x) = 0, a linear combination of teacher and student nodes (note ?? k and ?? k are k-th rows of A 1 and B 1 ).

Therefore, zero gradient means that the summation of coefficients of co-linear ReLU nodes is zero.

Since teachers are not co-linear, any teacher node is co-linear with at least one student node.

Alignment with multiple student nodes is also possible.

If there is no nonlinearity (e.g., deep linear models), alignment won't happen since a linear subspace has many representations.

Note that a necessary condition of a reconstructed teacher node is that its boundary is in the active region of student, or is observed (Definition 2).

This is intuitive since a teacher node which behaves like a linear node is partly indistinguishable from a bias term.

This also suggests that overparameterization (more student nodes) are important.

More student nodes mean more observers, and the existence argument in Theorem 4 is more likely to happen and more teacher nodes can be covered by student, yielding better generalization.

For student nodes that are not aligned with the teacher, if they are observed by other student nodes, then following a similar logic, we have the following: Theorem 3 (Prunable Un-specialized Student Nodes).

With Assumption 1, at SGD critical point, if an unaligned student k has C independent observers (concatenating v yields a full rank matrix),

If node k is not co-linear with any other student, then v k = 0.

Corollary 2.

With sufficient observers, the contribution of all unaligned student nodes is zero.

LeCun et al., 1990; Hassibi et al., 1993; Hu et al., 2016) .

This is consistent with Theorem 5 in (Tian et al., 2019) which also shows the fan-out weights are zero up on convergence in 2-layer networks, if the initialization is close.

In contrast, Theorem 3 analyzes the critical point rather than the dynamics.

Note that a relate theorem (Theorem 6) in (Laurent & von Brecht, 2017 ) studies 2-layer network with scalar output and linear separable input, and discusses characteristics of individual data point contributing loss in a local minima of GD.

Here no linear separable condition is imposed.

Thanks to Lemma 1 which holds for deep ReLU networks, we can use similar intuition to analyze the behavior of the lowest layer (l = 1) in the multiple layer case.

The difference here is that A 1 (x) and B 1 (x) are no longer constant over x. Fortunately, using Corollary 1, we know that A 1 (x) and B 1 (x) are piece-wise constant that separate the input region R 0 into a finite (but potentially exponential) set of constant regions

} plus a zero-measure set.

This suggests that we could check each region separately.

If the boundary of a teacher j and a student k lies in the region, similar logic applies (here ?? kj is the (k, j) entry of A 1 (x) and is constant in a region R ??? R 0 ).

Theorem 4 (Student-teacher Alignment, Multiple Layers).

With Assumption 1, at SGD critical points, for any teacher node j at l = 1, if there exists a region R ??? R and a student observer k so that ???E * j ??? E k ??? R = ??? and ?? kj (R) = 0, then node j aligns with at least one student node k .

Note that even with random V 1 (x) (e.g., at initialization), Theorem 4 still holds with high probability (when ?? kj = 0) and teacher f * 1 (x) can still align with student f 1 (x).

This suggests a picture of bottom-up training in backpropagation: After the alignment of activations at layer 1, we just treat layer 1 as the low-level features and the procedure repeats until the student matches with the teacher at all layers.

This is consistent with many previous works that empirically show the network is learned in a bottom-up manner .

Note that the alignment may happen concurrently across layers: if the activations of layer 1 start to align, then activations of layer 2, which depends on activations of layer 1, will also start to align since there now exists a W 2 that yields strong alignments, and so on.

This creates a critical path from important student nodes at the lowest layer all the way to the output, and this critical path accelerates the convergence of that student node.

We leave a formal analysis to the future work.

Small Gradient Case.

In practice, stochastic gradient (or its expectation over time) fluctuates around zero (

, but never zero.

In this case, Theorem 5 shows that a rough specialization still follows.

The ratio of recovery is also shown for weights/biases separately, as a function of .

Note?? jj is the angle of two weightsw j andw j .

for any x ??? R 0 with ??? 0 , then for any teacher j at l = 1, if there exists a region R ??? R and a student observer k so that ???E * j ???E k ???R = ???, and ?? kj (R) = 0, then j is roughly aligned with a student k :

|?? kj | for any ?? > 0.

The hidden constants depends on ??, 0 and the size of region ???E * j ???E k ???R.

Note that E t [ g 1 (x) ??? ] ??? leads to g 1 (x) ??? ??? at least for some iteration t.

Therefore, Theorem 5 still applies since it does not rely on past history of the weight/gradient.

Note that Theorem 5 assumes infinite number of data points and leave finite sample case to future work.

Our analysis so far shows student specialization happens at SGD critical points under mild conditions.

A natural question arises: is running SGD long enough sufficient to achieve these critical points?

Some previous works (Ge et al., 2017; Livni et al., 2014) show that empirically SGD does not recover the parameters of a teacher network up to permutation, while other works (Saad & Solla, 1996; Goldt et al., 2019) show specialization happens.

Why there is a discrepancy?

There are several reasons.

First, from Theorem 3, there exist un-specialized student nodes, so a simple permutation test on student weights might fail.

Second, as suggested by Theorem 5, it can take a long time to recover a teacher node k with small v * k (since ?? kj = v * k v j ).

In fact, if v * k = 0 then it has no contribution to the output and recovery never happens.

This is particularly problematic if the output dimension is 1 (scalar output), since a single small teacher weight v * k would block the recovery of the entire teacher node k. Previous works (Lampinen & Ganguli, 2019) shows similar behaviors in the dynamics of singular values in deep linear networks in teacher-student setting, which lack student specialization.

Here we study these behaviors in deep ReLU networks.

In the following, we analyze various local dynamic behaviors of 2-layer ReLU network.

Due to the complexity, we leave a formal characterization of the entire training procedure for future work.

Definition 3.

A teacher node j is strong (or weak), if v * j is large (or small).

In this case, the dynamics can be written as the following:

where

5.1 WEIGHT MAGNITUDE From Eqn.

5, we know that for both ReLU and linear network (since

When there is only a single output, r l is a scalar and Eqn.

6 is simply an inner product between the residue and the activation of node k, over the batch.

So if the node k has activation which aligns well with the residual, the inner product is larger and w k grows faster.

Note that Eqn.

6 only tell that the weight norm would increase, but didn't tell whether w k converges to any teacher node w * j .

It could be the case that w k goes up but doesn't move towards the teacher.

To see that, let's check the quantity:

where

Putting it in another way, we want to check the spectrum property of the PSD matrix G kj .

Intuitively, the direction of E x f l???1 z k f * j should lie between w k and w * j , and the magnitude is large when w k and w * j are close to each other.

This means that if r is dominated by a teacher j (i.e., v * j is large), then??? k would push w k towards w * j .

This also shows that SGD will first try fitting strong teacher nodes, then weak teacher nodes.

Theorem 6 confirms this intuition if f l???1 follows spherical symmetric distribution (e.g., N (0, I)).

where ?? is the angle between w * j and w k .

As a result, for all ?? ??? [0, ??], E x f l???1 z k f * j is always between w * j and w k since ?? ??? ?? and sin ?? are always non-negative.

Without such symmetry, we assume the following holds:

Note that critical point analysis is applicable to any batch size, including 1.

On the other hand, Assumption 2 holds when a moderately large batchsize leads to a decent estimation of the terms.

With this assumption, we can write the dynamics as??? k = w k r k , where the time-varying residue r k of node k is defined as the following (?? is a scalar related to ?? ): Figure 3: Student specialization of a 2-layered network with 10 teacher nodes and 1x/2x/5x/10x student nodes.

p is teacher polarity factor (Eqn. 9).

For a student node k, we plot its normalized correlation (in terms of activation vector evaluated in a separate evaluation set) to its best correlated teacher as the x coordinate and the fan-out weight norm v k as the y coordinate.

We plot results from 32 random seed.

Student nodes of different seeds are in different color.

An un-specialized student node has low fan-out weight norm (Theorem 3).

.

We consider a special (and symmetric) case: r k = r = w * ??? k a k w k with all a k > 0, where w * is a joint contribution of all teacher nodes.

In this case, we could show that whenw

dt (w k r k ???w k r k ) < 0 and vice versa.

So the system provides negative feedback untilw k =w k and according to Eqn.

7, the ratio between w k and w k remains constant, after initial transition.

We can also show thatw k will align with w * and every student node goes to w * .

However, due to Theorem 6, the net effect w * can be different for different students and thus r k are different.

This opens the door for complicated dynamic behavior of neural network training.

Symmetry breaking.

As one example, if we add a very small delta to some node, say k = 1 so that r 1 = r + w * .

Then to make d dt (w k r k ???w k r k ) = 0, we havew k r k >w k r k and thus according to Theorem 7, w k / w k grows exponentially.

This symmetric breaking behavior provides a potential winners-take-all mechanism, since according to Theorem 6, the coefficient of w * depends critically on the initial angle between w k and w * .

Strong teacher nodes are learned first.

If v * j is the largest among teacher nodes, then the joint w * heavily biases towards teacher j and all student nodes move towards teacher j. As a result, strong teacher learns first and is often covered by multiple co-linear students (Fig. 6, teacher-0) .

Focus shifting to weak teacher nodes.

The process above continues until residual along the direction of w * j quickly shrinks and residual corresponding to other teacher node (e.g., w * j for j = j) becomes dominant.

Since each r k is different, student node k whose direction is closer to w * j (j = j) will shift their focus towards w * j , as shown in the green (shift to teacher-2) and magenta (shift to teacher-5) curves in Fig. 6 .

Possible slow convergence to weak teacher nodes.

While expected angle between two weights from initialization is ??/2, shifting a student node w k from chasing after a strong teacher node w * j to a weaker one w * j could yield a large initial angle (e.g., close to ??) between w k and w j .

For example, all student nodes have been attracted to the opposite direction of a weak teacher node.

In this case, the convergence can be arbitrarily slow.

In fact, if there is only one teacher node and ?? is the angle between teacher and student, then from Eqn.

8 we arrive at?? ??? ?????(??) sin ??.

Since ??(??) sin ?? ??? (?? ??? ??) 2 around ?? = ??, the time spent from ?? = ?? ??? to some ?? 0 is t 0 ??? 1 ??? 1 ???????0 ??? +??? when ??? 0.

In this case, over-realization helps by having more student nodes that are possibly ready for shifting towards weaker teachers, and thus accelerate convergence (Fig. 7) .

Alternatively, we could reinitialize those student nodes (Prakash et al., 2019) .

We first verify our theoretical finding on synthetic dataset.

We generate the input using N (0, ?? 2 I) with ?? = 10 and we sample 10k as training and another 10k as evaluation.

For deep ReLU networks, we regenerate the dataset after every epoch to mimic infinite sample setting.

The details of teacher/student construction is in Appendix (Sec. 8.16).

The normalized correlation between nodes is computed in terms of activation vectors evaluated on a separate evaluation set.

Two layer networks.

First we verify Theorem 2 and Theorem 3 in the 2-layer setting.

Fig. 6 shows student nodes correlate with different teacher nodes over time.

Fig. 3 shows for different degrees of Number of hidden teacher nodes is 50-75-100-125.

Student is 10x over-realized.

The dataset is regenerated with the input distribution after every epoch.

For node k, y-axis is Ex [?? kk (x)], equivalent to the fan-out weight norm v k in 2-layer case, and x-axis is its max correlation to the teachers.

The lower layer learns first.

over-realization (1??/2??/5??/10??), for nodes with weak specialization (i.e., its normalized correlation to the most correlated teacher is low), their magnitudes of fan-out weights are small.

Otherwise the nodes with strong specialization have high fan-out weights.

Deep Networks.

For deep ReLU networks, we observe specialization not only at the lowest layer, as suggested by Theorem 4, but also at multiple hidden layers.

This is shown in Fig. 4 .

For each student node k, the x-axis is its best normalized correlation to teacher nodes, and y-axis is E x [?? kk (x)], which is equivalent to v k in 2-layer case.

In the plot, we can also see the lowest layer learns first (the "L-shape" curve was established at epoch 10), then the top layers follow.

Ablation on the effect of over-realization.

To further understand the role of over-realization, we plot the average rate of a teacher node that is matched with at least one student node successfully (i.e., correlation > 0.95).

Fig. 5 shows that stronger teacher nodes are more likely to be matched, while weaker ones may not be explained well, in particular when the strength of the teacher nodes are polarized (p is large).

Over-realized student can explain more teacher nodes, while a student with 1?? nodes has sufficient capacity to fit the teacher perfectly, it gets stuck despite long training.

In addition, the evaluation loss (Appendix Fig. 11) shows that over-realization yields better generalization, in particular with large teacher node polarity (p is large), where weak teacher nodes are hard to capture.

For good performance on real datasets, getting weak teacher nodes can be important.

Training Dynamics.

We set up a diverse strength of teacher node by constructing the fanout weights of teacher node j as follows:

where p is the teacher polarity factor that controls how strong the energy decays across different teacher nodes.

p = 0 means all teacher nodes are symmetric, and large p means that the strength of teacher nodes are more polarized.

Figure 6: Student specialization with teacher polarity p = 1 (Eqn. 9).

Same students are represented by the same color across plots.

Three rows represent three different random seeds.

We can see more students nodes specialize to teacher-1 first.

In contrast, teacher-5 was not specialized until later by a node (e.g., magenta in the first row) that first chases after teacher-1 then shifts its focus.

Figure 7: Evolution of best student correlation to teacher over iterations.

Each rainbow color represents one of the 20 teachers (blue: strongest, red: weakest).

5x over-parameterization on 2-layer network.

Fig. 6 and Fig. 7 show that many student nodes specialize to a strong teacher node first.

Once the strong teacher node was covered well, weaker teacher nodes are covered after many epochs.

We also experiment on CIFAR-10.

We first pre-train a teacher network with 64-64-64-64 ConvNet (64 are channel sizes of the hidden layers, L = 5) on CIFAR-10 training set.

Then the teacher network is pruned in a structured manner to keep strong teacher nodes.

The student is over-realized based on teacher's remaining channels.

The convergence and specialization behaviors of student network is shown in Fig. 8 .

Specialization happens at all layers for different degree of over-realization.

Over-realization boosts student specialization, measured by mean of maximal normalized correlation ?? mean = mean j??? teacher max j ??? studentf * jf j at each layer (f j is the normalized activation of node j over N evaluation samples), and improves generalization, evaluated on CIFAR-10 evaluation set.

In this paper, we use student-teacher setting to analyze how an (over-parameterized) deep ReLU student network trained with SGD learns from the output of a teacher.

When the magnitude of gradient per sample is small (student weights are near the critical points), the teacher can be proven to be covered by (possibly multiple) students and thus the teacher network is recovered in the lowest layer.

By analyzing training dynamics, we also show that strong teacher node with large v * is reconstructed first, while weak teacher node is reconstructed slowly.

This reveals one important reason why the training takes long to reconstruct all teacher weights and why generalization improves with more training.

As the next step, we would like to extend our analysis to finite sample case, and analyze the training dynamics in a more formal way.

Verifying the insights from theoretical analysis on a large dataset (e.g., ImageNet) is also the next step.

Figure 8: Mean of the max teacher correlation ??mean with student nodes over epochs in CIFAR10.

More over-realization gives better student specialization across all layers and achieves strong generalization (higher evaluation accuracy on CIFAR-10 evaluation set).

the fact that D L (x) = I C??C (no ReLU gating in the last layer), the condition holds.

Now suppose for layer l, we have:

Using

we have:

Proof.

By definition of SGD critical point, we know that for any batch B j , Eqn.

1 vanishes:

where

.

Note that B j can be any subset of samples from the data distribution.

Therefore, for a dataset of size N , Eqn.

20 holds for all N |B| batches, but there are only N data samples.

With simple Gaussian elimination we know that for any i 1 = i 2 , U i1 = U i2 = U .

Plug that into Eqn.

20 we know U = 0 and thus for any i, U i = 0.

Since U i is an outer product, the theorem follows.

Note that if ??? l ??? ??? , which is i???Bj U i ??? ??? , then with simple Gaussian elimination for two batches B 1 and B 2 with only two sample difference, we will have for any

Plug things back in and we have |B| U i ??? ??? [2(|B| ??? 1) + 1] , which is U i ??? ??? 2 .

If f l???1 (x;??) has the bias term, then immediately we have g l (x;??) ??? ??? .

Proof.

The base case is that V L (x) = V * L (x) = I C??C , which is constant (and thus piece-wise constant) over the entire input space.

If for layer l, V l (x) and V * l (x) are piece-wise constant, then by Eqn.

4 (rewrite it here):

since D l (x) and D * l (x) are piece-wise constant and W l and W * l are constant, we know that for layer l ??? 1, V l???1 (x) and V * l???1 (x) are piece-wise constant.

Therefore, for all l = 1, . . .

L, V l (x) and V * l (x) are piece-wise constant.

Therefore, A l (x) and B l (x) are piece-wise constant with respect to input x. They separate the region R 0 into constant regions with boundary points in a zero-measured set.

Lemma 2.

Consider K ReLU activation functions f j (x) = ??(w j x) for j = 1 . . .

K. If w j = 0 and no two weights are co-linear, then j c j f j (x) = 0 for all x ??? R d+1 suggests that all c j = 0.

Proof.

Suppose there exists some c j = 0 so that j c j f j (x) = 0 for all x. Pick a point x 0 ??? ???E j so that w j x 0 = 0 but all w j x 0 = 0 for j = j, which is possible due to the distinct weight conditions.

Consider an -ball B x0, = {x : x ??? x 0 ??? }.

We pick so that sign(w j x) for all j = j remains the same within B x0, (Fig. 9(a) ).

Denote [j + ] as the indices of activated ReLU functions in B x0, except j.

Then for all x ??? B x0, ??? E j , we have:

Since B x0, is a d-dimensional object rather than a subspace, for x 0 and x 0 + e k ??? B(x 0 , ), we have

where e k is axis-aligned unit vector (1 ??? k ??? d).

This yields

Plug it back to Eqn.

22 yields

where means that for the (augmented) d + 1 dimensional weight:

However, if we pick

which is a contradiction.

Lemma 3 (Local ReLU Independence).

Let R be an open set.

Consider K ReLU nodes f j (x) = ??(w j x), j = 1, . . .

, K. w j = 0, w j = ??w j for j = j with any ?? > 0.

If there exists c 1 , . . .

, c K , c??? so that the following is true:

and for node j, ???E j ??? R = ???, then c j = 0.

Proof.

We can apply the same logic as Lemma 2 to the region R ( Fig. 9(b) ).

For any node j, since its boundary ???E j is in R, we can find a similar x 0 so that x 0 ??? ???E j ??? R and x 0 / ??? ???E j for any j = j. We construct B x0, .

Since R is an open set, we can always find > 0 so that B x0, ??? R and no other boundary is in this -ball.

Following similar logic of Lemma 2, c j = 0.

Lemma 4 (Relation between Hyperplanes).

Let w j and w j two distinct hyperplanes with w j = w j = 1.

Denote ?? jj as the angle between the two vectors w j and w j .

Then there exists u j ???w j and w j ?? j = sin ?? jj .

Proof.

Note that the projection ofw j ontow j is:

It is easy to verify that ?? j = 1 and w j ?? j = sin ?? jj .

Lemma 5 (Evidence of Data points on Misalignment).

Let R ??? R d be an open set.

Consider K ReLU nodes f j (x) = ??(w j x), j = 1, . . .

, K. w j = 1, w j are not co-linear.

Then for a node j with ???E j ??? R = ???, and ??? 0 , either of the conditions holds:

(1) There exists node j = j so that sin

(2) There exists x j ??? ???E j ??? R so that for any j = j,

where ?? jj is the angle betweenw j andw j , ?? > 0, r is the radius of a d ??? 1 dimensional ball

Proof.

Define q j = 5 /|c j |.

For each j = j, define I j = {x : |w j x| ??? q j , x ??? ???E j }.

We prove by contradiction.

Suppose for any j = j, sin ?? jj > KM 1????? /|c j | or |b j ??? b j | > M 2 1???2?? /|c j |.

Otherwise the theorem already holds.

From Lemma 4, we know that for any x ??? ???E j , if w j x = ???q j , with a j ??? 2q j |c j |/M K 1????? = Consider a d ??? 1-dimensional sphere B ??? ??? j and its intersection of I j ??? B for j = j. Suppose the sphere has radius r.

For each I j ??? B, its d ??? 1-dimensional volume is upper bounded by:

where V d???2 (r) is the d ??? 2-dimensional volume of a sphere of radius r. Intuitively, the intersection between w j x = ???q j and B is at most a d ??? 2-dimensional sphere of radius r, and the "height" is at most a j .

In this case, we want to show that for any x ??? ??? j , |w j x| > q j and thus I j ??? B = ???. If this is not the case, then there exists x ??? ??? j so that |w j x| ??? q j .

Then since x ??? ???E j , we have:

Therefore, from Cauchy inequality and triangle inequality, we have:

From the condition, we have w j ???w j = 2 sin

which is equivalent to:

(35) for ??? 0 .

This is a contradiction.

Therefore, I j ??? B = ??? and thus V (I j ??? B) = 0.

Volume argument.

Therefore, from the definition of M , we have

, then for ??? 0 , we have:

This means that there exists x j ??? B ??? ??? j so that x j / ??? I j ??? B for any j = j and j in case 1.

That is,

On the other hand, for j in case 2, the above condition holds for entire ??? j , and thus hold for the chosen x j .

Lemma 6 (Local ReLU Independence, Noisy case).

Let R be an open set.

Consider K ReLU nodes f j (x) = ??(w j x), j = 1, . . .

, K. w j = 1, w j are not co-linear.

If there exists c 1 , . . .

, c K , c??? and ??? 0 so that the following is true:

and for a node j, ???E j ??? R = ???.

Then there exists node j = j so that sin ?? jj ??? M K 1????? /|c j | and |b j ??? b j | ??? M 2 1???2?? /|c j |, where r, ??, M, M 2 are defined in Lemma 5 but with r = r ??? 5 /|c j |.

Proof.

Let q j = 5 /|c j | and ??? j = {x : x ??? ???E j ??? R, B(x, q j ) ??? R}. If situation (1) in Lemma 5 happens then the theorem holds.

Otherwise, applying Lemma 5 with R = {x : x ??? R, B(x, q j ) ??? R} and there exists x j ??? ??? j so that Let two points x ?? j = x j ?? q jwj ??? R. In the following we show that the three points x j and x ?? j are on the same side of ???E j for any j = j.

This can be achieved by checking whether (w j x j )(w j x ?? j ) ??? 0 (Fig. 10) :

Since |w j w j | ??? 1, it is clear that (w j x j )(w j x ?? j ) ??? 0.

Therefore the three points x j and x ?? j are on the same side of ???E j for any j = j.

we know that all terms related to w??? and w j with j = j will cancel out (they are in the same side of the boundary ???E j ) and thus:

which is a contradiction.

Proof.

In this situation, because D 2 (x) = D * 2 (x) = I, according to Eqn.

4, V 1 (x) = W 1 and V * 1 (x) = W * 1 are independent of input x. Therefore, both A 1 and B 1 are independent of input x. From Assumption 1, since ??(x) > 0 in R 0 , from Theorem.

1 we know that the SGD critical points gives

Picking node k, the following holds for every node k and every

Here ?? k is the k-th row of A 1 , A 1 = [?? 1 , . . .

, ?? n1 ] and similarly for ?? k .

Note here layer index l = 1 is omitted for brevity.

For teacher j, suppose it is observed by student k, i.e., ???E * j ??? E k = ???. Given all teacher and student nodes, note that co-linearity is a equivalent relation, we could partition these nodes into disjoint groups.

Suppose node j is in group s.

In Eqn.

44, if we combine all coefficients in group s together into one term c s w * j (with w * j = 1), we have:

"At most" because from Assumption 1, all teacher weights are not co-linear.

Note that co-linear(j) might be an empty set.

By Assumption 1, ???E * j ??? R 0 = ??? and by observation property, ???E * j ??? E k = ???, we know that for R = R 0 ??? E k , ???E * j ??? R = ???. Applying Lemma 3, we know that c s = 0.

Since ?? kj = 0, we know co-linear(j) = ??? and there exists at least one student k that is aligned with the teacher j.

Proof.

We basically apply the same logic as in Theorem 2.

Consider the colinear group co-linear(k).

If for all k ??? co-linear(k), ?? k k ??? v k 2 = 0, then v k = 0 and the proof is complete.

Otherwise, if there exists some student k so that v k = 0.

By the condition, it is observed by some student node k o , then with the same logic we will have

Since k is observed by C students k Proof.

We can write the contribution of all student nodes which are not aligned with any teacher nodes as follows:

where w s is the unit vector that represents the common direction of the co-linear group s. From Theorem 3, for group s that is not aligned with any teacher, k???co-linear(s) v k w k = 0 and thus the net contribution is zero.

Proof.

In multi-layer case, A l (x) and B l (x) are no longer constant over input x. Fortunately, thanks to the recursive definition (Eqn.

4) which only contains input-independent terms (weights) and gating function, A l (x) and B l (x) are piece-wise constant function over the input R 0 .

Note that R 0 can be partitioned into R = {R From the condition, there exists open set R ??? R and a student observer node k so that ???E * j ??? E k ??? R = ??? (( Fig. 9(c) ).

Let H R and similarly H * R be the student and teacher nodes whose boundary where ?? = 2?? + ?? and

We construct teacher networks in the following manner.

For two-layered network, the output dimension C = 50 and input dimension d = m 0 = n 0 = 100.

For multi-layered network, we use 50-75-100-125 (i.e, m 1 = 50, m 2 = 75, m 3 = 100, m 4 = 125, L = 5, d = m 0 = n 0 = 100 and C = m 5 = n 5 = 50).

The teacher network is constructed to satisfy Assumption 1: at each layer, teacher filters are distinct from each other and their bias is set so that ??? 50% of the input data activate the nodes.

This makes their boundary (maximally) visible in the dataset.

To train the model, we use vanilla SGD with learning rate 0.01 and batchsize 16.

Fig. 11 shows how the loss changes over iterations.

With high teacher polarity (Eqn.

9), it becomes harder to learn the weak teacher nodes and over-realization helps in getting low evaluation loss (in particular for p = 2.5).

Besides Gaussian distribution we also test on uniform distribution x ??? U [???15, 15] .

For training, we sample 100k data points in each epoch.

Fig. 12 shows that the results on 4 layer ReLU network (50-75-100-125) are similar.

Note that in multi-layer setting, Theorem 3 might not hold since it is for 2-layer so there could be un-specialized student nodes with large ?? kk (x).

???15, 15] .

For all teacher nodes, the normalized correlations are all close to 1.0 (?? mean ??? 0.998 at all layers).

@highlight

This paper analyzes training dynamics and critical points of training deep ReLU network via SGD in the teacher-student setting. 

@highlight

Study of over-parametrization in student-teacher multilayer ReLU networks, a theoretical part about SGD critical points for the teacher-student setting, and a heuristic and empirical part on dynamics of the SDG algorithm as a function of teacher networks.