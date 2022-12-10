Much attention has been devoted recently to the generalization puzzle in deep learning: large, deep networks can generalize well, but existing theories bounding generalization error are exceedingly loose, and thus cannot explain this striking performance.

Furthermore, a major hope is that knowledge may transfer across tasks, so that multi-task learning can improve generalization on individual tasks.

However we lack analytic theories that can quantitatively predict how the degree of knowledge transfer depends on the relationship between the tasks.

We develop an analytic theory of the nonlinear dynamics of generalization in deep linear networks, both within and across tasks.

In particular, our theory provides analytic solutions to the training and testing error of deep networks as a function of training time, number of examples, network size and initialization, and the task structure and SNR.

Our theory reveals that deep networks progressively learn the most important task structure first, so that generalization error at the early stopping time primarily depends on task structure and is independent of network size.

This suggests any tight bound on generalization error must take into account task structure, and explains observations about real data being learned faster than random data.

Intriguingly our theory also reveals the existence of a learning algorithm that proveably out-performs neural network training through gradient descent.

Finally, for transfer learning, our theory reveals that knowledge transfer depends sensitively, but computably, on the SNRs and input feature alignments of pairs of tasks.

Many deep learning practitioners closely monitor both training and test errors, hoping to achieve both a small training error and a small generalization error, or gap between testing and training errors.

Training is usually stopped early, before overfitting sets in and increases the test error.

This procedure often results in large networks that generalize well on structured tasks, raising an important generalization puzzle BID23 : many existing theories that upper bound generalization error BID4 BID14 BID7 BID8 BID15 BID3 Arora et al., 2018, e.g ) in terms of various measures of network complexity yield very loose bounds.

Therefore they cannot explain the impressive generalization capabilities of deep nets.

In the absence of any such tight and computable theory of deep network generalization error, we develop an analytic theory of generalization error for deep linear networks.

Such networks exhibit highly nonlinear learning dynamics BID19 b) including many prominent phenomena like learning plateaus, saddle points, and sudden drops in training error.

Moreover, theory developed for the learning dynamics of deep linear networks directly inspired better initialization schemes for nonlinear networks BID21 BID16 .

Here we show that deep linear networks also provide a good theoretical model for generalization dynamics.

In particular we develop an analytic theory for both the training and test error of a deep linear network as a function of training time, number of training examples, network architecture, initialization, and task structure and SNR.

Our theory matches simulations and reveals that deep networks with small weight initialization learn the most important aspects of a task first.

Thus the optimal test error at the early stopping time depends largely on task structure and SNR, and not on network architecture, as long as the architecture is expressive enough to attain small training error.

Thus our exact analysis of generalization dynamics reveals the important lesson that any theory that seeks to upper bound generalization error based only on network architecture, and not on task structure, is likely to yield exceedingly loose upper bounds.

Intriguingly our theory also reveals a non-gradient-descent learning algorithm that proveably out-performs neural network training through gradient descent.

We also apply our theory to multi-task learning, which enables knowledge transfer from one task to another, thereby further lowering generalization error BID6 Luong et al., 2016, e.g.) .

Moreover, knowledge transfer across tasks may be key to human generalization capabilities BID10 .

We provide an analytic theory for how much knowledge is transferred between pairs of tasks, and we find that it displays a sensitive but computable dependence on the relationship between pairs of tasks, in particular, their SNRs and feature space alignments.

We note that a related prior work BID0 ) studied generalization in shallow and deep linear networks, but that work was limited to networks with a single output, thereby precluding the possibility of addressing the issue of transfer learning.

Moreover, analyzing networks with a single output also precludes the possibility of addressing interesting tasks that require higher dimensional outputs, for example in language (Dong et al., 2015, e.g.) , generative models (Goodfellow et al., 2014, e.g ), and reinforcement learning Silver et al., 2016, e.g ).

We work in a student-teacher scenario in which we consider an ensemble of low rank, noisy teacher networks that generate training data for a potentially more complex student network, and define the training and test errors whose dynamics we wish to understand.

We first consider an ensemble of 3-layer linear teacher networks with N i units in layer i, and weight matrices W 21 ∈ R N2×N 1 and W 32 ∈ R N3×N2 between the input to hidden, and hidden to output layers, respectively.

The teacher network thus computes the composite map y = Wx, where W ≡ W 32 W 21 .

Of critical importance is the singular value decomposition (SVD) of W: DISPLAYFORM0 Where U ∈ R N3×N 2 and V ∈ R N1×N 2 are both matrices with orthonormal columns and S is an N 2 × N 2 diagonal matrix.

We construct a random teacher by picking U and V to be random matrices with orthonormal columns and choosing O(1) values for the diagonal elements of S. We work in the limit N 1 , N 3 → ∞ with an O(1) aspect ratio A = N 3 /N 1 ∈ (0, 1] so that the teacher has fewer outputs than inputs.

Also, we hold N 2 ∼ O(1), so the teacher has a low, finite rank, and we study generalization performance as a function of the N 2 teacher singular values.

We further assume the teacher generates noisy outputs from a set of N 1 orthonormal inputs: DISPLAYFORM1 This training set yields important second-order training statistics that will guide student learning: DISPLAYFORM2 Here the input covariance Σ 11 is assumed to be white (a common pre-processing step), the inputoutput covariance Σ 31 is simplified using (2), and Z ∈ R N 3×N 1 is the noise matrix, whose µ'th column is z µ .

Its matrix elements z µ i are drawn iid.

from a Gaussian with zero mean and variance σ 2 z /N 1 .

The noise scaling is chosen so the singular values of the teacher W and the noise Z are both O(1), leading to non-trivial generalization effects.

As generalization performance will depend on the ratio of teacher singular values to the noise variance parameter σ 2 z , we simply set σ z = 1 in the following.

Thus we can think of teacher singular values as signal to noise ratios (SNRs).Finally, we note that while we focus for ease of exposition in the main paper on the case of one hidden layer networks and a full orthonormal basis of P = N 1 training inputs in the main paper, neither of these assumptions are essential to our theory.

Indeed in Section 3.4 and App.

A we extend our theory to networks of arbitrary depth, and in App.

G we extend our theory to the case of white inputs with P = N 1 , obtaining a good match between theory and experiment in both cases.

Now consider a student network with N i units in each layer.

We assume the first and last layers match the teacher (i.e. N 1 = N 1 and N 3 = N 3 ) but N 2 ≥ N 2 , allowing the student to have more hidden units than the teacher.

We also consider deeper students (see below and App.

A).

Now consider any student whose input-output map is given by y = W 32 W 21 ≡ Wx.

Its training error on the teacher dataset in (2) and its test error over a distribution of new inputs are given by DISPLAYFORM0 respectively.

Herex µ andŷ µ are the noisy training set inputs and outputs in (2), whereas x denotes a random test input drawn from zero mean Gaussian with identity covariance, y µ = Wx µ is noise free teacher output, and · denotes an average w.r.t the distribution of the test input x. Due to the orthonormality of the training and isotropy of the test inputs, both ε train and ε test can be expressed as DISPLAYFORM1 (5) Both ε train and ε test can be further expressed in terms of the student, training data and teacher SVDs, which we denote by W = USV T , Σ 31 =ÛŜV T , and W = U S V T respectively.

Specifically, DISPLAYFORM2 DISPLAYFORM3 Thus as the student learns, its training and test error dynamics depends on the alignment of the time-evolving student singular modes {s α , u α , v α } with the fixed training data {ŝ α ,û α ,v α } and teacher {s α , u α , v α } singular modes respectively.

Here we derive and numerically test analytic formulas for both the training and test errors of a student network as it learns from training data generated from a teacher network.

We explore the dependence of these quantitites on the student network size, student initialization, teacher SNR, and training time.

We assume the student weights undergo batch gradient descent with learning rate λ on the training error µ ||ŷ µ −W 32 W 21xµ || 2 2 , which for small λ is well approximated by the differential equations: DISPLAYFORM0 (where τ ≡ 1/λ), which must be solved from an initial set of student weights at time t = 0 BID19 .

We consider two classes of student initializations.

The first initialization corresponds to a random student where the weights W 21 and W 32 are chosen such that the composite map W = W 32 W 21 has an SVD W = UV T , where U and V are random singular vector matrices and all student singular values are .

As such a random student learns, the composite map undergoes a time dependent evolution DISPLAYFORM1 For white inputs, as t → ∞, W → Σ 31 , and so the time-dependent student singular modes {s α (t), u α (t), v ( t)} converge to the training data singular modes {ŝ α ,û α ,v α }.

However, the explicit dynamics of the student singular modes can be difficult to obtain analytically from random initial conditions.

Thus we also consider a special class of training aligned (TA) initial conditions in which W 21 and W 32 are chosen such that the composite map W = W 32 W 21 has an SVD W = ÛV T .

That is, the TA network (henceforth referred to simply as the TA) has the same singular vectors as the training data covariance Σ 31 , but has all singular values equal to .

As shown in BID19 , as the TA learns according to (8), the singular vectors of its composite map W remain unchanged, while the singular values evolve as s α (t) = s(t,ŝ α ), where the learning curve function s(t,ŝ) as well as its functional inverse t(s,ŝ) is given by DISPLAYFORM2 Here the function s(t,ŝ) describes analytically how each training set singular valueŝ drives the dynamics of the corresponding TA singular value s, and for notational simplicity, we have suppressed the dependence of s(t,ŝ) on τ and the initial condition .

As shown in FIG0 , for eachŝ, s(t,ŝ) is a sigmoidal learning curve that undergoes a sharp transition around time t/τ = 1 2ŝ ln (ŝ/ − 1), at which it rises from its small initial value of at t = 0 to its asymptotic value ofŝ as t/τ → ∞. Alternatively, we can plot s(t,ŝ)/ŝ as a function ofŝ for different training times t/τ , as in FIG0 .

This shows that TA learning corresponds to a singular mode detection wave which progressively sweeps from large to small singular values.

At any given training time t, training data modes with singular valuesŝ > t/τ have been learned, while those with singular valuesŝ < t/τ have not.

While the TA is more sophisticated than the random student, since it already knows the singular vectors of the training data before learning, we will see that the analytic solution for the TA learning dynamics provides a good approximation to the student learning dynamics, not only for the training error, as shown in BID19 , but also for the generalization error as shown below.

The results in this section assume a single hidden layer, but BID19 derived t(s,ŝ) for networks of arbitrary depth and we apply our theory to some deeper networks.

The general differential equation and derivations for deeper networks can be found in Appendix A.

In the previous section, we reviewed an exact analytic solution for the composite map of a TA network, namely that its singular modes are related to those of the training data through the relation the relation (3).

Since the input matrixX is orthonormal, Σ 31 is simply a perturbation of the low rank teacher W by a high dimensional noise matrix Z. The relation between the singular modes of a low rank matrix and its noise perturbed version has been studied extensively in BID5 , in the high dimensional limit we are working in, namely N 1 , N 3 → ∞ with the aspect ratio A = N 3 /N 1 ∈ (0, 1], and N 2 ∼ O(1).

DISPLAYFORM0 In this limit, the top N 2 singular values and vectors of Σ 31 converge toŝ(s α ), where the transfer function from a teacher singular value s to a training data singular valueŝ is given by the function DISPLAYFORM1 The associated top N 2 singular vectors of Σ 31 can also acquire a nontrivial overlap with the N 2 modes of the teacher through the relation |û DISPLAYFORM2 , where the singular vector overlap function is given by DISPLAYFORM3 The rest of the N 3 − N 2 singular vectors of Σ 31 are orthogonal to the top N 2 ones, and their singular values are distributed according to the the Marchenko-Pastur (MP) distribution: DISPLAYFORM4 Overall, these equations describe a singular vector phase transition in the training data, as illustrated in FIG1 .

For example in the case of no teacher, the training data is simply noise and the singular values of Σ 31 are distributed as an MP sea spread between 1 ± √ A. When one adds a teacher, how each teacher singular mode is imprinted on the training data depends crucially on the teacher singular value s, and the nature of this imprinting undergoes a phase transition at s = A 1/4 .

For s ≤ A 1/4 , the teacher mode SNR is too low and this mode is not imprinted in the noisy training data; the associated training data singular valueŝ remains at the edge of the MP sea at 1 + √ A, and the overlap O(s) between training and teacher singular vectors remains zero.

However, when s > A 1/4 , this teacher mode is imprinted in the training data; there is an associated training data singular valueŝ that pops out of the MP sea FIG1 .

However, the training data singular value emerges at a positionŝ > s that is inflated by the noise, though the inflation effect decreases at larger s, with the ratioŝ/s approaching the unity line as s becomes large FIG1 .

Similarly, the corresponding training data singular vectors acquire a non-trivial overlap with the teacher singular vectors when s > A 1/4 , and the alignment approaches unity as s increases FIG1 ).

Based on an analytic understanding of how the singular mode structure {s α , u α , v α } of the teacher W is imprinted in the modes {ŝ α ,û α ,v α } of the training data covariance Σ 31 through FORMULA0 , FORMULA0 and FORMULA0 , and in turn how this training data singular structure drives the time evolving singular modes of a Figure 3 : Match between theory and experiment for rank 1 (row 1, a-d) and rank 3 (row 2, e-h) teachers with single-hidden-layer students: (a-b, e-f) log train and test error, respectively, showing very close match between theory and experiment for TA, and close match for the random student. (c,g) comparing TA and randomly initialized students minimum generalization errors, showing almost perfect match. (d,h) comparing TA and randomly initialized students optimal stopping times, showing small lag due to alignment.

(N 1 = 100, N 2 = 50, N 3 = 50.) FORMULA9 , we can now derive analytic expressions for ε train and ε test in (6) and FORMULA6 , for a TA network.

We will also show that these learning curves closely approximate those of a random student with time-evolving singular vectors {u α (t), v α (t)}, and match on several key aspects.

First, inserting the TA dynamics in (10) into ε train in (6), we obtain DISPLAYFORM0 DISPLAYFORM1 Here, s α (t) = s(ŝ α , t) as defined in (9) are the TA singular values, andŝ α =ŝ(s α ) as defined in (11) are the training data singular values associated with the teacher singular values s α .

Also · R denotes an average with respect to the MP distribution in (13) over a region R. Two distinct regions contribute to training error.

First R in contains those top N 2 − N 2 training data singular values that do not correspond to the N 2 singular values of the teacher but will be learned by a rank N 2 student.

Second, R out corresponds to the remaining N 3 − N 2 lowest training data singular values that cannot be learned by a rank N 2 student.

In terms of the MP distribution, DISPLAYFORM2 , where f is the point at which the MP density has 1 − N 2 /N 3 of its mass to the left and N 2 /N 3 of its mass to the right.

In the simple case of a full rank student, f = 1 − √ A, and one need only integrate over R in which is the entire range.

Equation FORMULA0 for ε train makes it manifest that it will go to zero for a full rank student as its singular values approach those of the training data.

Of course the test error can behave very differently.

Inserting the TA training dynamics in (10) into ε test in (7), and using (11), (12) and (13) to relate training data to the teacher, we find FORMULA0 and FORMULA0 constitute a complete theory of generalization dynamics in terms of the structure of the data distribution (i.e. the teacher rank N 2 , teacher SNRs {s α }, and the teacher aspect ratio A = N 3 /N 1 ), the architectural complexity of the student (i.e. its rank N 2 , its number of layers N l , and the norm of its initialization), and the training time t. They yield considerable insight into the dynamics of good generalization early in learning and overfitting later, as we show below.

Fig. 3 demonstrates an excellent match between the theory and simulations for the TA, and a close match for random students, for single-hidden-layer students and various teacher ranks N 2 .

Intuitively, as time t proceeds, learning corresponds to singular mode detection wave sweeping from large to small training data singular values (i.e. the wave in FIG0 sweeps across the training data spectrum in FIG1 .

Initially, strong singular values associated with large SNR teacher modes are learned and both ε train and ε test drop.

Fig. 3A -D are for a rank 1 teacher, and so in Fig 3AB we see a single sharp drop early on, if the teacher SNR is sufficiently high.

By contrast, with a rank 3 teacher in Fig. 3E -H, there are several early drops as the three modes are picked up.

However, as time progresses, the singular mode detection wave penetrates the MP sea, and the student picks up noise structure in the data, so ε train drops but ε test rises, indicating the onset of overfitting.

DISPLAYFORM3

The main difference between the random student and TA learning curves is that the random student learning is slightly delayed relative to the TA, especially late in training.

This is understandable because the TA already knows the singular vectors of the training data, while the random student must learn them.

Nevertheless, two of the most important aspects of learning, namely the optimal early stopping time t opt gradient ≡ argmin t ε test (t) and the minimal test error achieved at this time ε opt gradient ≡ min t ε test (t), match well between TA and random student, as shown in Fig. 3CD .

At low teacher SNRs, the student takes a little longer to learn than the TA, but their optimal test errors match.

Our theory can also be easily extended to describe the learning dynamics deeper networks.

BID19 derived t(s,ŝ) for networks of arbitrary depth, so we only need to adjust this factor in our formulas, see App.

A for details.

In FIG2 we show that again there is an excellent match between TA networks and theory for student networks with N l = 5 layers (i.e. 3 hidden layers).

Randomly-initialized networks show a much longer alignment lag for deeper networks (see App.

B for details), but the curves are qualitatively similar and optimal stopping errors match.

We also demonstrate extensions of our theory to different numbers of training examples (App.

G).Importantly, many of the phenomena we observe in linear networks are qualitatively replicated in nonlinear networks (Fig. 5) , suggesting that our theory may help guide understanding of the nonlinear case.

In particular, features such as stage-like initial learning, followed by a plateau if SNR is high, and finally followed by overfitting, are replicated.

However, there are some discrepancies, in particular nonlinear networks (especially deeper ones) begin overfitting earlier than linear networks.

This is likely because a mode in a non-linear network can be co-opted by an orthogonal mode, while in a linear network it cannot.

Thus noise modes are able to "stow away" on the strong signal modes once they are learned.

However, overall learning patterns are similar, and we show below that many interesting phenomena in nonlinear networks are understandable in the linear case, such as the (non-)effects of overparameterization, the dynamics of memorization, and the benefits of transfer.

DISPLAYFORM0 Figure 5: Train (first row, A-D) and test (second row, E-H) error for nonlinear networks (leaky relu at all hidden layers) with one hidden layer (first two columns) or three hidden layers (last two columns) trained on the tasks above, with a rank 1 teacher (first and third columns) or a rank 3 teacher (second and fourth columns).

Note that many of the qualitative phenomena observed in linear networks, such as stage-like improvement in the errors, followed by a plateau, followed by overfitting, also appear in nonlinear networks.

Compare the first column to Fig. 3AB , the second column to Fig. 3EF , the third to FIG2 , and the fourth to An intriguing observation that resurrected the generalization puzzle in deep learning was the observation by BID23 that deep networks can memorize data with the labels randomly permuted.

However, as BID2 pointed out, the learning dynamics of training error for randomized labels can be slower than than for structured data.

This phenomenon also arises in deep linear networks, and our theory yields an analytic explanation for why.

We randomize data by choosing orthonormal inputsx µ as in the structured case, but we choose the outputsŷ µ to be i.i.d.

Gaussian with zero mean and the same diagonal variance as the structured training data generated by the teacher.

For structured data generated by a low rank teacher with singular values s α , the diagonal output variance is given by σ modes, yielding this stretched MP distribution (compare 6A top and bottom).

However, even on this stretched MP distribution, the right edge will be much smaller than the signal singular values, since the signal variance will be diluted by spreading it out over many more modes in the randomized data.

Thus the randomized data will lead to slower initial training error drops relative to the structured data (Fig. 6B) since the singular mode detection wave encounters the first signal singular values in structured data earlier than it encounters the edge of the stretched MP sea in randomized data.

For the case of a rank 1 teacher, it is straightforward to derive a good analytic approximation to the important quantities ε opt gradient and t opt gradient .

We assume the teacher SNR is beyond the phase transition point so its unique singular value s 1 > A 1/4 , yielding a separation between the training data singular valueŝ 1 in (11) and the edge of the MP sea.

In this scenario, optimal early stopping will occur at a time before the detection wave in FIG0 penetrates the MP sea, so to minimize test error, we can neglect the first term in (15).

Then optimizing the second term yields the optimal student singular value s 1 = s 1 O(s 1 ).

Inserting this value into (15) yields ε opt gradient = 1 − O(s 1 ) 2 , and inserting it into (9) yields t opt gradient .

Thus the optimal generalization error with a rank 1 teacher is very simply related to the alignment of the top training data singular vectors with the teacher singular vectors, and it decreases as this alignment increases.

In App.

E, we show this match in the rank 1 case.

With higher rank teachers, ε opt gradient and t opt gradient must negotiate a more complex trade-off between teacher modes with different SNRs.

For example, as the singular mode detection wave passes the top training data singular value, s 1 (t) →ŝ 1 which is greater than the optimal s 1 = s 1 O(s 1 ) for mode 1.

Thus as learning progresses, the student overfits on the first mode but learns lower modes.

However, this neural generalization dynamics suggests a superior non-gradient training algorithm that simply optimally sets each s α to s α O(s α ) in (15), yielding an optimal generalization error: DISPLAYFORM0 Standard gradient descent learning cannot achieve this low generalization error because it cannot independently adjust all student singular values.

A simple algorithm that achieves ε opt non-gradient is as follows.

From the training data covariance Σ 31 , extract the top singular valuesŝ α that pop-out of the MP sea, use the functional inverse of (11) to compute s a (ŝ α ), use (12) to compute the optimal s α , and then construct a matrix W with the same top singular vectors as Σ 31 , but with the outlier singular values shrunk fromŝ α to s α and the rest set to zero.

This non-gradient singular value shrinkage algorithm provably outperforms neural network training with ε

Consider two tasks A and B, described by N 3 by N 1 teacher maps W A and W B , of ranks N A 2 and N B 2 , respectively.

Now two student networks can learn from the two teacher networks separately, each achieving optimal early stopping test errors ε opt A and ε opt B .

Alternatively, one could construct a composite teacher (and student) that concatenates the hidden and output units, but shares the same input units (Fig. 7) .

The composite student and teacher each have two heads, one for each task, with N 3 neurons per head.

Optimal early stopping on each head of the student yields test errors ε opt A←B and ε opt B←A .

We define the transfer benefit that task B confers on task A to be T A←B ≡ ε opt A − ε opt A←B .

A postive (negative) transfer benefit implies learning tasks A and B simultaneously yields a lower (higher) optimal test error on task A compared to just learning task A alone.

as spanning a low dimensional feature space in N 1 dimensional input space that is important for each task, then Q reflects the input feature subspace similarity matrix.

Interestingly, the transfer benefit is independent of output singular vectors U A and U B .

What matters for knowledge transfer in this setting are the relevant input features, not how you must respond to them.

We describe the transfer benefit for the simple case of two rank one teachers.

Then S A , S B , and Q are simply scalars s A , s B and q, and we explore the function T A←B (s A , s B , q) in Fig. 5ABC , which reveals several interesting features.

First, knowledge can be transferred from a high SNR task to a low SNR task (Fig. 5A ) and the degree of transfer increases with task alignment q. This can make it possible to capture signals from task A which would otherwise sink into the MP sea by learning jointly with a related task, even if the tasks are only weakly aligned (Fig. 5A ).

However, if task A already has a high SNR, task B must be very well aligned to it for transfer to be beneficial -otherwise there will be interference.

The degree of alignment required increases as the task A SNR increases, but the quantity of benefit or interference decreases correspondingly (Fig. 5BC) .

In Appendix D we explain why our theory predicts these results.

Furthermore, in Appendix F we demonstrate these phenomena are qualitatively recapitulated in nonlinear networks, which suggests that our theory may give insight into how to choose auxiliary tasks.

structured data, and provides a non-gradient based learning method that out-performs gradient descent learning in the linear case.

Finally, we provide an analytic theory of how knowledge is transferred from one task to another, demonstrating that the degree of alignment of input features important for each task, but not how one must respond to these features, is critical for facilitating knowledge transfer.

We think these analytic results provide useful insight into the similar generalization and transfer phenomena observed in the nonlinear case.

Among other things, we hope our work will motivate and enable: (1) the search for tighter upper bounds on generalization error that take into account task structure; (2) the design of non gradient based training algorithms that outperform gradient-based learning; and (3) the theory-driven selection of auxiliary tasks that maximize knowledge transfer.

In the main text, we described the dynamics of how a single-hidden-layer network converges toward the training data singular modes {ŝ α ,û α ,v α }, which were originally derived in BID19 .

There it was also proven that for a network with N l layers (i.e. N l − 2 hidden layers), the strength of the mode obeys the differential equation:

DISPLAYFORM0 This equation is separable and can be integrated for any integer number of layers.

In particular, we consider the case of 5 layers (3 hidden), in which case: DISPLAYFORM1 This expression cannot be analytically inverted to find s(t,ŝ), so we numerically invert it where necessary.

As noted in the main text, the randomly-initialized networks behave quite similarly to the TA networks, except that the randomly-initialized networks show a lag due to the time it takes for the network's modes to align with the data modes.

In fig. 9 we explore this lag by plotting the alignment of the modes and the increase in the singular value for several randomly initialized networks.

Notice that stronger modes align more quickly.

Furthermore, the mode alignment is relatively independent -whether the teacher is rank 1 or rank 3, the alignment of the modes is similar for the mode of singular value 2.

Most importantly, note how the deeper networks show substantially slower mode alignment, with alignment not completed until around when the singular value increases.

This explains why deeper networks show a larger lag between randomly-initialized and TA networks -the alignment process is much slower for deeper networks.

In the case of transfer learning, or more generally when we want to evaluate a network's loss on a subset of its outputs, we need to use a slight generalization of the train and test error formulas given in the main text.

Suppose we are interested in the train and test errors after applying a projection operator P: DISPLAYFORM0 respectively.

As in the main text, we can rexpress these as: DISPLAYFORM1 DISPLAYFORM2 Using the cyclic property of the trace, we can modify these to get: DISPLAYFORM3 DISPLAYFORM4 As before, we express these in terms of the student, training data and teacher SVDs, W = USV T , Σ 31 =ÛŜV T , and W = U S V T respectively.

Specifically, DISPLAYFORM5 DISPLAYFORM6

Thm 1 (Transfer theorem) The transfer benefit T A←B :• Is unaffected by the U A and U B .• Is completely determined by only σ Figure 9: Alignment of randomly-initialized network modes to data modes and growth of singular values, plotted for 1 hidden layer (first two rows, a-d) and 3 hidden layers (last two rows, e-h), and for a rank 1 teacher (first and third rows, a & e), or a rank 3 teacher (second and fourth rows, b-d & f-h).

The columns are the different modes, with respective singular values of 6, 4, and 2.

σ z was set to 1.

The deeper networks show substantially slower mode alignment, with alignment not completed until around when the singular value increases.

Proof: We define DISPLAYFORM0 Because of the 0 blocks in U AB , the vectors in blocks corresponding to task A and task B are completely orthogonal, so U AB remains orthonormal.

Thus the relationship between the U A and U A is irrelevant to the transfer.

(In our simulations we use arbitrary orthonormal matrices for U A and U B .)

Therefore the transfer effects will be entirely driven by the relationship between the matrices V A and V B and the singular values.

We define N DISPLAYFORM1 Now if c is an eigenvector of this matrix: DISPLAYFORM2 This implies that DISPLAYFORM3 , with the mapping between the eigenvectors given by V AB .

Furthermore, this mapping must be a bijection for eigenvectors with non-zero eigenvalues, since the matrices have the same rank (the rank of V AB ).

To see this, note that S AB 2 is full rank.

From this, it is clear that DISPLAYFORM4 is positive definite, so DISPLAYFORM5 Now that we know the eigenvectors of these matrices are in bijection, note that: DISPLAYFORM6 Because the output modes don't matter (as noted above), the alignment between the eigenvectors of DISPLAYFORM7 and V A , weighted by their respective eigenvalues, gives the transfer benefit.

For any given tasks, the transfer benefit can be calculated using our theory.

However, in certain special cases, we can give exact answers.

For example, in the rank one case with equal singular values between the tasks (s A = s B = s), the matrix DISPLAYFORM8 with eigenvalues s √ 1 ± q and eigenvectors DISPLAYFORM9 Corresponding to the shared structure between the tasks and the differences between them.

We note that the sign of the alignment q is irrelevant as a special case of the fact (noted above) that any orthogonal transformation on the output modes does not affect transfer.

Why is there interference between tasks which are not well aligned?

In the rank one case, we are effectively changing the (input) singular dimensions of Y A from V A to V AB .

The two singular modes of V AB correspond to the shared structure between the tasks (weighted by the relative signal strengths), and the differences between them, respectively.

Although we may be improving our estimates of the shared mode if q > 0 (by increasing its singular value relative to s A ), we are actually decreasing its alignment with V A unless q = 1.

This misalignment is captured by the second mode of V AB , but the increase in the singular value of the first mode must come at the cost of a decrease in the singular value of the second mode.

See FIG0 for a conceptual illustration of this.

This means that the multi-task setting allows the distinctions between the tasks to sink towards the sea of noise, while pulling out the common structure.

In other words, transferring knowledge from one task always comes at the cost of ignoring differences between the tasks.

Furthermore, incorporating a task B allows its noise to seep into the task A signal.

Together, these two effects help to explain why transfer can be sometimes beneficial but sometimes detrimental.

Figure 10: Conceptual cartoon of how T A←B , the transfer benefit (or cost) arises from alignment between the task's input modes.

In FIG0 we show the match between the error achieved by training the student by gradient descent and the optimal stopping error predicted by the non-gradient shrinkage algorithm in the case of a rank-1 teacher.

FIG0 : Match between optimal stopping error prediction from non-gradient training algorithm and empirical optimal stopping error for a rank-1 teacher.

Since most deep learning practitioners do not train linear networks, it is important that our theoretical insights generalize beyond this simple case.

In this section we show that the transfer patterns qualitatively generalize to non-linear networks.

Here, we show results from teacher networks with N 1 = 100 N 3 = 50, N 2 = 4 (thus the task is higher rank) and leaky relu non-linearities at the hidden and output layers.

We train a student with leaky relu units and N 2 = N 3 to solve this task.

Results qualitatively look quite similar to those in A. With support from another aligned task, especially one with moderately higher SNR, performance on a low SNR task will improve.

(b) s A = 3.

Tasks with modest signals will face interference from poorly aligned tasks, but benefits from well aligned tasks.

These effects are amplified by SNR.

(c) s A = 100.

Tasks with very strong signals will show little effect from other tasks (note y-axis scale), but any impact will be negative unless the tasks are very well aligned.

In the main text, we focused on the test error dynamics in the case in which the number of examples equalled the number of inputs.

Here we show how the formula for test error curves is modified as the number of training examples P is varied.

For simplicity, when P = N 1 , we focus on the case of a full rank student with aspect ratio A = 1 (so that N 1 = N 2 = N 3 ).

The more general case of lower rank students with non-unity aspect ratios can be easily found from this case, but with some additional bookkeeping.

As before, we assume the teacher generates noisy outputs from a set of P inputs: DISPLAYFORM0 Figure FORMULA0 The effects of varying the number of training examples P .

(a) Test error for a student learning from a rank-1 teacher with an SNR of 3, with different numbers of inputs. (b,c) Minimum generalization error plotted against P/N 1 and SNR · P/N 1 , respectively, at different SNRs.

When P ≥ N 1 , the minimum generalization error is simply determined by SNR P/N 1 , so all curves converge to a single asymptotic line in (c) as P increases.

When P < N 1 , however, the curves for different SNRs separate because the projection and noise effects depend on initial SNR.

This training set yields important second-order training statistics that will guide student learning: DISPLAYFORM1 HereX,Ŷ, and Z are each N 1 by P , N 3 by P , and N 3 by P matrices respectively, whose µ'th columns arex µ ,ŷ µ , andẑ µ , respectively.

Σ 11 is an N 1 by N 1 input correlation matrix, and Σ 31 is an N 3 by N 1 the input-output correlation matrix.

We choose the matrix elements z µ i of the noise matrix Z to be drawn iid from a Gaussian with zero mean and variance σ 2 z /N 1 .

The noise scaling is chosen so the singular values of the teacher W and the noise Z are both O(1), leading to non-trivial generalization effects.

Furthermore, we chose training inputs to be close to unit-norm, and make the input covariance matrix Σ 11 as white as possible (whitening is a common pre-processing step for inputs).

When P > N 1 , this can be done by choosing the rows ofX to be orthonormal and then scaling up by P/N 1 , so the columns are approximately unit norm.

Then Σ 11 = P/N 1 I is proportional to the identity.

On the otherhand, if P < N 1 , we choose the columns ofX to be orthonormal, so that Σ 11 = P || , where P || is a projection operator onto the P dimensional column space ofX spanned by the input examples.

Both these choices are intended to approximate the situation in which the columns ofX are chosen to be iid unit-norm vectors.

Finally, as generalization performance will depend on the ratio of teacher singular values to the noise variance parameter σ 2 z , we simply set σ z = 1 as in the main text.

Thus, given the unit-norm inputs, we can think of the teacher singular values as signal to noise ratios (SNRs).

We now examine how the dynamics of the test error evolves as we vary the number of training examples P .

We split our analyses into two distinct regimes: (1) the oversampled regime in which the data density D ≡ P/N 1 > 1, and (2) the undersampled regime in which D < 1.

The oversampled regime (D > 1) is relatively simple.

For the undersampled regime (D < 1), we must account for the fact that the P training inputs do not span the full N 1 dimensional space of all inputs.

Thus the projection operator P || onto the P dimensional column space ofX plays a crucial role.

Indeed the input-correlation Σ 11 = P || .And Σ 31 = WP || + ZX T .

This implies that the learning dynamics only transforms the composite student map W from the P dimensional subspace spanned by the inputs to the N 3 dimensional output space.

In contrast, the student map from the N 1 − P dimensional subspace orthogonal to the image of P || remains frozen.

Tracing through the equations of the main paper and accounting for the projection operator P || , we find the effective aspect ratio for this undersampled learning problem (when N 3 = N 2 = N 1 ) is no longer A = N 3 /N 1 but rather D = P/N 1 .

Furthermore, in the limit This equation has several modifications compared to the case P = N 1 in (15).

First the term in the numerator involving N 3 − P reflects generalization error due to the N 3 − P dimensional frozen subspace, and the initial weight variance 2 contributes to this generalization error.

The second term in the numerator involves all the P − N 2 training modes which cannot be correlated with the teacher, and the average · is over a Marcenko-Pasteur distribution of singular values (see (13)) except with the aspect ratio A replaced by D. The third term accounts for learned correlations between the student and teacher.

It involves the transformation from teacher singular values s to training data singular valuesŝ through the formula (11) except with the aspect ratio replacement A → D, and the effective teacher singular value attenuation s → √ Ds.

Similarly, the computation of the singular vector overlap is done through (12) also with the replacements A → D and s → √ Ds.

In FIG0 , we show an excellent match between our theory and empirical simulations for varying values of P , both in the oversampled and undersampled measurement regimes.

There are a number of interesting features to note.

First, although the minimum generalization error improves monotonically with P , the asymptotic (t → ∞) generalization error does not, because of a frozen subspace (Advani & Saxe, 2017) of the modes that are not overfit when P < N 1 , because the training data rank is ≤ P .

Second, when P ≥ N 1 , the minimum generalization error is simply determined by SNR P/N 1 , so all curves converge to a single asymptotic line as P increases.

When P < N 1 , however, the curves for different SNRs separate because the projection and noise effects depend on initial SNR.

Finally, in FIG0 we show that approximately unit norm i.i.d.

gaussian inputs yield similar results to the orthogonalized data matrices we employed in the theory, although the gaussian inputs do result in slightly higher optimal stopping error.

Although we generally assumed students were full rank in the main text to simplify the calculations, our theory remains exact for TA networks of any rank.

Furthermore, as shown in FIG0 , the TA and random networks again show very similar optimal stopping generalization error, but with the optimal stopping time of the random networks lagging behind that of the TA networks.

Furthermore, this lag increases as the rank of the random network decreases (because a low rank network will have less initial projection onto the random modes, there is is more alignment to be done).

However, reducing the student rank does not change the optimal stopping error (as long as it is still greater than the teacher rank).(a) Best generalization error is quite similar between aligned and random initializations The optimal stopping time in the randomly initialized networks consistently lags behind the aligned networks, because it takes time for the alignment to occur.

This lag increases as the students rank decreases.

(c) Randomly initialized networks of varying ranks obey qualitatively similar trends of increase in optimal stopping error and optimal stopping time as SNR decreases.

(d) The theory predicts the aligned networks trends of increase in optimal stopping error and optimal stopping time with decreasing SNR almost perfectly. (All plots are made with a rank 1 teacher and N 1 = N 3 = 100)

<|TLDR|>

@highlight

We provide many insights into neural network generalization from the theoretically tractable linear case.

@highlight

The authors study a simple model of linear networks towards understanding generalization and transfer learning