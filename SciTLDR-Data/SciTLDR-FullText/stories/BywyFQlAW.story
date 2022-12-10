We introduce and study minimax curriculum learning (MCL), a new method for adaptively selecting a sequence of training subsets for a succession of stages in machine learning.

The subsets are encouraged to be small and diverse early on, and then larger, harder, and allowably more homogeneous in later stages.

At each stage, model weights and training sets are chosen by solving a joint continuous-discrete minimax optimization, whose objective is composed of a continuous loss (reflecting training set hardness) and a discrete submodular promoter of diversity for the chosen subset.

MCL repeatedly solves a sequence of such optimizations with a schedule of increasing training set size and decreasing pressure on diversity encouragement.

We reduce MCL to the minimization of a surrogate function handled by submodular maximization and continuous gradient methods.

We show that MCL achieves better performance and, with a clustering trick, uses fewer labeled samples for both shallow and deep models while achieving the same performance.

Our method involves repeatedly solving constrained submodular maximization of an only slowly varying function on the same ground set.

Therefore, we develop a heuristic method that utilizes the previous submodular maximization solution as a warm start for the current submodular maximization process to reduce computation while still yielding a guarantee.

Inspired by the human interaction between teacher and student, recent studies BID28 BID2 BID56 ) support that learning algorithms can be improved by updating a model on a designed sequence of training sets, i.e., a curriculum.

This problem is addressed in curriculum learning (CL) BID6 , where the sequence is designed by a human expert or heuristic before training begins.

Instead of relying on a teacher to provide the curriculum, self-paced learning (SPL) BID31 BID58 BID57 BID59 chooses the curriculum during the training process.

It does so by letting the student (i.e., the algorithm) determine which samples to learn from based on their hardness.

Given a training set D = {(x 1 , y 1 ), . . .

, (x n , y n )} of n samples and loss function L(y i , f (x i , w)), where x i ∈ R m represents the feature vector for the i th sample, y i is its label, and f (x i , w) is the predicted label provided by a model with weight w, SPL performs the following: DISPLAYFORM0 SPL jointly learns the model weights w and sample weights ν, which end up being 0-1 indicators of selected samples, and it does so via alternating minimization.

Fixing w, minimization w.r.t.

ν selects samples with loss L(y i , f (x i , w)) < λ, where λ is a "hardness parameter" as it corresponds to the hardness as measure by the current loss (since with large λ, samples with greater loss are allowed in).

Self-paced curriculum learning BID27 introduces a blending of "teacher mode" in CL and "student mode" in SPL, where the teacher can define a region of ν by attaching a linear constraint a T ν ≤ c to Eq. (1).

SPL with diversity (SPLD) BID26 , adds to Eq. (1) a negative group sparse regularization term −γ ν 2,1 −γ b j=1 ν (j) 2 , where the samples are divided into b groups beforehand and ν (j) is the weight vector for the j th group.

Samples coming from different groups are thus preferred, to the extent that γ > 0 is large.

CL, SPL, and SPLD can be seen as a form of continuation scheme BID1 ) that handles a hard task by solving a sequence of tasks moving from easy to hard; the solution to each task is the warm start for the next slightly harder task.

That is, each task, in the present case, is determined by the training data subset and other training hyperparameters, and the resulting parameters at the end of a training round are used as the initial parameters for the next training round.

Such continuation schemes can reduce the impact of local minima within neural networks BID7 BID5 .

With SPL, after each round of alternating minimization to optimize Eq. (1), λ is increased so that the next round selects samples that have a larger loss, a process BID28 BID59 BID2 ) that can both help avoid local minima and reduce generalization error.

In SPLD, γ is also increased between training rounds, increasingly preferring diversity.

In each case, each round results in a fully trained model for the currently selected training samples.

Selection of training samples has been studied in other settings as well, often with a different motivation.

In active learning (AL) BID53 and experimental design BID43 , the learner can actively query labels of samples from an unlabeled pool during the training process, and the goal is to reduce annotation costs.

The aim is to achieve the same or better performance using fewer labeled samples by ruling out uninformative ones.

Diversity modeling was introduced to AL in BID62 .

It uses submodular maximization to select diverse training batches from the most uncertain samples.

However, changing the diversity during the learning process has not been investigated as far as we know.

In boosting BID51 BID19 , the goal is to learn an ensemble of weak classifiers sequentially; it does this by assigning weights to all samples, with larger weights given to samples having larger loss measured by an aggregation of previously trained models.

Both active learning and boosting favor samples that are difficult to predict, since they are the most informative to learn.

For example, uncertainty sampling BID13 BID52 BID14 BID15 selects samples that are most uncertain, while query by committee BID54 BID14 BID0 selects the ones that multiple models most disagree on.

With machine teaching BID28 BID66 BID49 BID65 , a separate teacher helps the training procedure find a good model.

The SPL approach starts with a smaller set of easy samples and gradually increases the difficulty of the chosen samples as measured by the sample loss of the model produced by previous round's training.

One of the difficulties of this approach is the following: since for any given value of λ the relatively easiest samples are chosen, there is a good chance that the process can repeatedly select a similar training set over multiple rounds and therefore can learn slowly.

This is precisely the problem that SPLD address -by concomitantly increasing the desired diversity over rounds, the sample selection procedure chooses from an increasingly diverse set of different groups, as measured by ν 2,1 .

Therefore, in SPLD, early stages train on easier not necessarily diverse samples and later stages train on harder more diverse samples.

There are several challenges remaining with SPLD, however.

One is that in early stages, it is still possible to repeatedly select a similar training set over multiple rounds since diversity might not increase dramatically between successive rounds.

Potentially more problematically, it is not clear that having a large diversity selection weight in late stages is desirable.

For example, with a reasonably trained model, it might be best to select primarily the hardest samples in the part of the space near the difficult regions of the decision boundaries.

With a high diversity weight, samples in these difficult decision boundary regions might be avoided in favor of other samples perhaps already well learnt and having a large margin only because they are diverse, thereby leading to wasted effort.

At such point, it would be beneficial to choose points having small margin from the same region but that might not have the greatest diversity, especially when using only a simple notion of diversity such as the group sparse norm v 2,1 .

Also, it is possible that late stages of learning can select outliers only because they are both hard and diverse.

Lastly, the SPL/SPLD min-min optimization involves minimizing a lower bound of the loss, while normally one would, if anything, wish to minimize the loss directly or at least an upper bound.

Motivated by these issues, we introduce a new form of CL that chooses the hardest diverse samples in early rounds of training and then actually decreases, rather than increases, diversity as training rounds proceed.

Our contention is that diversity is more important during the early phases of training when only relatively few samples are selected.

Later rounds of training will naturally have more diversity opportunity simply because the size of the selected samples is much larger.

Also, to avoid successive rounds selecting similar sets of samples, our approach selects the hardest, rather than the easiest, samples at each round.

Hence, if a set of samples is learnt well during one training round, those samples will tend to be ill-favored in the next round because they become easier.

We also measure hardness via the loss function, but the selection is always based on the hardest and most diverse samples of a given size k, where the degree of diversity is controlled by a parameter λ, and where diversity is measured by an arbitrary non-monotone submodular function.

In fact, for binary variables the group sparse norm is also submodular where ν 2,1 = b j=1|C j ∩ A| = F (A) where A is the set for which ν is the characteristic vector, and C j is the set of samples in the j th group.

Our approach allows the full expressive class of submodular functions to be used to measure diversity since the selection phases is based on submodular optimization.

Evidence for the naturalness of such hardness and diversity adjustment in a curriculum can also be found in human education.

For example, courses in primary school usually cover a broad, small, and relatively easy range of topics, in order to expose the young learner to a diversity of knowledge early on.

In college and graduate school, by contrast, students focus on advanced deeper knowledge within their majors.

As another example, studies of bilingualism BID8 BID35 BID39 BID29 show that learning multiple languages in childhood is beneficial for future brain development, but early-age multi-lingual learning is usually not advanced or concentrated linguistically for any of the languages involved.

Still other studies argue that difficulty can be desired at early human learning stages BID10 BID38 ).

We introduce a new form of curriculum learning called minimax curriculum learning (MCL).

MCL increases desired hardness and reduces diversity encouragement over rounds of training.

This is accomplished by solving a sequence of minimax optimizations, each of which having the form: min DISPLAYFORM0 The objective is composed of the loss on a subset A of samples evaluating their hardness and a normalized monotone non-decreasing submodular function F : 2 V → R + measuring A's diversity, where V is the ground set of all available samples.

A larger loss implies that the subset A has been found harder to learn, while a larger F (A) indicates greater diversity.

The weight λ controls the trade-off between hardness and diversity, while k, the size of the resulting A, determines the number of samples to simultaneously learn and hence is a hardness parameter.

It is important to realize that F (A) is not a parameter regularizer (e.g., 1 or 2 regularization on the parameters w) but rather an expression of preference for a diversity of training samples.

In practice, one would add to Eq. (2) an appropriate parameter regularizer as we do in our experiments (Section 3).Like SPL/SPLD, learning rounds are scheduled, here each round with increasing k and decreasing λ.

Unlike SPL/SPLD, we explicitly schedule the number of selected samples via k rather than indirectly via a hardness parameter.

This makes sense since we are always choosing the hardest k samples at a given λ diversity preference, so there is no need for an explicit real-valued hardness parameter as in SPL/SPLD.

Also, the MCL optimization minimizes an upper bound of the loss on any size k subset of training samples.

The function F (·) may be chosen from the large expressive family of submodular functions, all of which are natural for measuring diversity, and all having the following diminishing returns property: given a finite ground set V , and any A ⊆ B ⊆ V and a v / ∈ B, DISPLAYFORM1 This implies v is no less valuable to the smaller set A than to the larger set B. The marginal gain of v conditioned on A is denoted f (v|A) f (v ∪ A) − f (A) and reflects the importance of v to A. Submodular functions BID20 have been widely used for diversity models BID37 BID36 BID3 BID50 BID21 BID9 .

Although Eq. (2) is a hybrid optimization involving both continuous variables w and discrete variables A, it can be reduced to the minimization of a piecewise function, where each piece is defined by a subset A achieving the maximum in a region around w.

Each piece is convex when the loss is convex, so various off-the-shelf algorithms can be applied once A has been computed.

However, the number of possible sets A is n k , and enumerating them all to find the maximum is intractable.

Thanks to submodularity, fast approximate algorithms BID45 BID40 BID41 exist to find an approximately optimal A. Therefore, the outer optimization over w will need to minimize an approximation of the piecewise function defined by an approximate A computed via submodular maximization.

The minimax problem in Eq. (2) can be seen as a two-person zero-sum game between a teacher (the maximizer) and a student (the minimizer): the teacher chooses training set A based on the student's feedback about the hardness (i.e., the loss achieved by current model w) and how diverse according to the teacher (λF (A)), while the student updates w to reduce the loss on training set A (i.e., learn A) given by the teacher.

Similar teacher-student interaction also exist in real life.

In addition, the teacher usually introduces concepts at the beginning and asks a small number of easy questions from a diverse range of topics and receives feedback from the student, and then further trains the student on the topics the student finds difficult while eschewing topics the student has mastered.

MCL's minimax formulation is different from the min-min formulation used in SPL/SPLD.

For certain losses and models, L(y i , f (x i , w)) is convex in w. The min-min formulation, however, is only bi-convex and requires procedures such as alternative convex search (ACS) as in BID4 .

Furthermore, diversity regularization of ν in SPLD leads to the loss of bi-convexity altogether.

Minimizing the worst case loss, as in MCL, is a widely used strategy in machine learning BID32 BID17 BID55 ) to achieve better generalization performance and model robustness, especially when strong assumptions cannot be made about the data distribution.

Compared to SPL/SPLD, MCL is also better in that the outer minimization over w in Eq. (2) is a convex program, and corresponds to minimizing the objective g(w) in Eq. (4).

On the other hand, querying g(w) requires submodular maximization which can only be solved approximately.

The goal of this section, therefore, is to address the minimax problem in Eq. (2), i.e., the minimization min w∈R m g(w) of the following objective g(w).

DISPLAYFORM0 If the loss function L(y i , f (x i , w)) is convex w.r.t.

w, then g(w) is convex but, as mentioned above, enumerating all subsets is intractable.

Defining the discrete objective G w : 2 V → R + where DISPLAYFORM1 shows that computing g(w) in involves a discrete optimization over G w (A), a problem that is submodular since G w (A) is weighted sum of a non-negative (since loss is non-negative) modular and a submodular function, and thus G w is monotone non-decreasing submodular.

Thus, the fast greedy procedure mentioned earlier can be used to approximately optimizes G w (A) for any w.

LetÂ w ⊆ V be the k-constrained greedy approximation to maximizing G w (A).

We define the following approximate objective: DISPLAYFORM2 and note that it satisfies αg(w) ≤ĝ(w) ≤ g(w) where α is the approximation factor of submodular optimization.

Forw within a region around w,ĝ(w) will utilize the same setÂ w .

Therefore,ĝ(w) is piecewise convex, if the loss function L(y i , f (x i , w)) is convex w.r.t.

w, and different regions of within R m are associated with differentÂ although not necessarily the same regions or sets that define g(w).

We show in Section 2.2 that minimizingĝ(w) offers an approximate solution to Eq. (2).Withĝ(w) given, our algorithm is simply gradient descent for minimizingĝ(w), where many off-the-shelf methods can be invoked, e.g., SGD, momentum methods, Nesterov's accelerated gradient BID47 , Adagrad BID16 , etc.

The key problem is how to obtainĝ(w), which depends on suboptimal solutions in different regions of w. It is not necessary, however, to run submodular maximization for every region of w. Since we use gradient descent, we only need to knowĝ(w) for w on the optimization path.

At the beginning of each iteration, we fix w and use submodular maximization to achieve theÂ w that definesĝ(w).

Then a gradient update step is applied toĝ(w).

Let A * w represent the optimal solution to Eq. (5), thenÂ w satisfies G(Â) ≥ αG(A * ).Algorithm 1 Minimax Curriculum Learning (MCL) DISPLAYFORM3 for t ∈ {0, · · · , p} do 6: DISPLAYFORM4 DISPLAYFORM5 12: end while Algorithm 1 details MCL.

Lines 5-10 solve the optimization in Eq. (2) with λ and k scheduled in line 11.

Lines 6-7 finds an approximateÂ via submodular maximization, discussed further in Section 2.1.

Lines 8-9 update w for the currentÂ by gradient descent π(·, η) with learning rate η.

The inner optimization stops after p steps and then λ is reduced by factor 1 − γ where γ ∈ [0, 1] and k is increased by ∆. The outer optimization stops after T steps when a form of "convergence", described below, is achieved.

GivenÂ w ,ĝ(w) has gradient DISPLAYFORM6 and thus gradient descent method can update w. For example, we can treatÂ as a batch if k is small, and update w by w ← w − η∇ĝ(w) with learning rate η.

For largeÂ w , we can use SGD that applies an update rule to mini-batches withinÂ w .

More complex gradient descent rules π(·, η) can take historical gradients and w t τ 's into account leading to w t+1 ← w t + π {w 1:t }, {∇ĝ(w 1:t )}, η .Considering the outer loop as well, the algorithm approximately solves a sequence of Eq. (2)s with decreasing λ and increasing k, where the previous solutions act as a warm start for the next iterations.

This corresponds to repeatedly updating the model w on a sequence of training setsÂ that changes from small, diverse, and hard to large.

Although solving Eq. FORMULA4 exactly is NP-hard, a near-optimal solution can be achieved by the greedy algorithm, which offers a worst-case approximation factor of α = 1 − e −1 BID45 .

The algorithm starts with A ← ∅, and selects next the element with the largest marginal gain f (v|A) from V \A, i.e., A ← A ∪ {v * } where v * ∈ argmax v∈V \A f (v|A), and this repeats until |A| = k. It is simple to implement, fast, and usually outperforms other methods, e.g., those based on integer linear programming.

It requires O(nk) function evaluations for ground set size |V | = n. Since Algorithm 1 runs greedy T p times, it is useful for the greedy procedure to be as fast as possible.

The accelerated, or lazy, greedy algorithm BID40 reduces the number of evaluations per step by updating a priority queue of marginal gains, while having the same output and guarantee as the original (thanks to submodularity) and offers significant speedups.

Still faster variants are also available BID41 .

Our own implementation takes advantage of the fact that line 7 of Algorithm 1 repeatedly solves submodular maximization over a sequence of submodular functions that are changing only slowly, and hence the previous set solution can be used as a warm start for the current algorithm, a process we call WS-SUBMODULARMAX outlined in Algorithm 2.The greedy procedure offers much better approximation factors than 1 − e −1 when the objective G(A) is close to modular.

Specifically, the approximation factor becomes α = (1 − e −κ G )/κ G BID12 , which depends on the curvature κ G ∈ [0, 1] of G(A) defined as DISPLAYFORM0 When κ G = 0, G is modular, and when κ G = 1, G is fully curved and the above bound recovers 1 − e −1 .

G(A) becomes more modular as the outer loop proceeds since λ decreases.

Therefore, the approximation improves with the number of outer loops.

In fact, we have: DISPLAYFORM1 where F is a monotone non-decreasing submodular function with curvature κ F , L is a non-negative modular function, and DISPLAYFORM2 The proof is given in Appendix 4.1.

In MCL, therefore, the submodular approximation improves (α → 1) as λ grows, and the surrogate functionĝ(w) correspondingly approaches the true convex objective g(w).

In this section, we study how close the solutionŵ is of applying gradient descent toĝ(w), where we assume p is large enough so that a form of convergence occurs.

Specifically, in Theorem 1, we analyze the upper bound on ŵ − w * 2 2 based on two assumptions: 1) the loss L (y i , f (x i , w)) being β-strongly convex w.r.t.

w; and 2)ŵ is achieved by running gradient descent in lines 6-9 of Algorithm 1 until convergence, defined as the gradient reaching zero.

In case the loss L (y i , f (x i , w)) is convex but not β-strongly convex, a commonly used trick to modify it to β-strongly convex is to add an 2 regularization (β/2) w 2 2 .

In addition, for non-convex L (y i , f (x i , w)), it is possible to prove that with high probability, a noise perturbed SGD onĝ(w) can hit an -optimal local solution of g(w) in polynomial time -we leave this for future work.

In our empirical study (Section 3), MCL achieves good performance even when applied to non-convex deep neural networks.

The following theorem relies on the fact that the maximum of multiple β-strongly convex functions is also β-strongly convex, shown in Appendix 4.2.Theorem 1 (Inner-loop convergence).

For the minimax problem in Eq. (2) with ground set of samples V and λ ≥ 0, if the loss function L (y i , f (x i , w)) is β-strongly convex and |V | ≥ k, running lines 6-9 of Algorithm 1 until convergence (defined as the gradient reaching zero) yields a solutionŵ satisfying DISPLAYFORM0 w is the solution achieved at convergence, w * is the optimal solution of the minimax problem in Eq.(2), g(w * ) is the objective value achieved on w * , and α is the approximation factor that submodular maximization can guarantee for G(A).The proof is given in Appendix 4.3.It is interesting to note that the bound depends both on the strong convexity parameter β and on the submodular maximization approximation α.

As mentioned in Lemma 1, as λ gets smaller, the approximation factor α approaches 1 meaning that the bound in Equation FORMULA13 improves.

We mention the convergence criteria where the gradient reaches zero.

While it is possible, in theory, for lines 6-9 of Algorithm 1 to oscillate amongst the non-differentiable boundaries between the convex pieces, with most damped learning rates, this will eventually subside and the algorithm will remain within one convex piece.

The reason for this is line 7 of the algorithm always chooses onê A thereby selecting one convex piece associated with the region around w t τ , and with only small subsequent adjustments to w t τ , the sameÂ will continue to be selected.

Hence, the algorithm will, in such case, reach the minimum of that convex piece where the gradient is zero.

We can restate and then simplify the above bound in terms of the resulting parameters, and corresponding λ, k values, used at a particular iteration τ of the outer loop.

In the following,ŵ τ is the solution achieved by Algorithm 1 at the iteration τ of the outer loop, and the optimal solution of the minimax problem in Eq.(2) with λ, k set as in iteration τ is denoted w * T .

Corollary 1.

If the loss function L (y i , f (x i , w)) is β-strongly convex, the submodular function F (·) has curvature κ F , and if each inner-loop in Algorithm 1 runs until convergence, then the solutionŵ τ at the end of the τ th iteration of the outer-loop fulfills: DISPLAYFORM1 where w * τ is the optimal solution of the minimax problem in Eq. (2) with λ set as in the τ th outer loop iteration.

Thus, if k starts from k 0 and linearly increases via k ← k + ∆ (as in line 11 of Algorithm alg:mcl), DISPLAYFORM2 Otherwise, if k increases exponentially, i.e., k ← (1 + ∆) · k, DISPLAYFORM3 In the above, λ 0 and k 0 are the initial values for λ and k, c DISPLAYFORM4 The proof can be found in Appendix 4.5.

On the one hand, the upper bound above is in terms of the ratio λ/k which improves with larger subset sizes.

On the other hand, submodular maximization becomes more expensive with k. Hence, Algorithm 1 chooses a schedule to decrease λ exponentially and increase k only linearly.

Also, we see that the bound is dependent on the submodular curvature κ F , the strongly-convex constant β, and c 1 which relates the submodular and modular terms (similar to as in Lemma 1).

These quantities (κ F /β and c 1 ) might be relevant for other convex-submodular optimization schemes.

There are several heuristic improvements we employ that are described next.

Algorithm 1 stops gradient descent after p steps.

A reason for doing this is thatŵ p can be sufficient as a warm-start for the next iteration if p is large enough.

We also have not observed any benefit for larger p, although we do eventually observe convergence empirically when the average loss no longer change appreciably between stages.

Also, lines 6-7 of Algorithm 1 require computing the loss on all the samples, and each step of the greedy algorithm needs to, in the worst case, evaluate the marginal gains of all of the unselected samples.

Moreover, this is done repeatedly in the inner-most block of two nested loops.

Therefore, we use two heuristic tricks to improve efficiency.

Fist, rather than selecting individual samples, we first cluster the data and then select clusters, thereby reducing the ground set size from the number of samples to the number of clusters.

We replace the per-sample loss L (y i , f (x i , w)) with a per-cluster loss L Y (i) , f (X (i) , w) that we approximate by the loss of the sample closest to the centroid within each cluster: DISPLAYFORM0 where C (i) is the set of indices of the samples in the i th cluster, and x (i) with label y (i) is the sample closest to the cluster centroid.

We find that the loss on x (i) is sufficiently representative to approximately indicate the hardness of the cluster.

The set V becomes the set of clusters and A ⊆ V is a set of clusters, and hence the ground set size is reduced speeding up the greedy algorithm.

When computing F (A), the diversity of selected clusters, cluster centroids again represent the cluster.

In line 8, the gradient is computed on all the samples in the selected clusters rather than on only x (i) at which point the labels of all the samples in the selected clusters are used.

Otherwise, when selecting clusters via submodular maximization, the labels of only the centroid samples are needed.

Thus, we need only annotate and compute the loss for samples in the selected clusters and the representative centroid samples x (i) of other clusters.

This also reduces the need to label all samples up front as only the labels of the selected clusters, and centroid samples of each cluster, are used (i.e., the clustering process itself does not use the labels).We can further reduce the ground set to save computation during submodular maximization via prefiltering methods that lead either to no BID60 or little BID64 BID41 reduction in approximation quality.

Moreover, as λ decreases in the MCL objective and G(A) becomes more modular, pruning method become more effective.

More details are given in Section 4.6.

In this section, we apply different curriculum learning methods to train logistic regression models on 20newsgroups BID33 , LeNet5 models on MNIST BID34 , convolutional neural nets (CNNs) with three convolutional layers 1 on CIFAR10 BID30 , CNNs with two convolutional layers 2 on Fashion-MNIST ("Fashion" in all tables) BID63 , CNNs with six convolutional layers on STL10 , and CNNs with seven convolutional layers on SVHN BID48 3 .

Details on the datasets can be found in TAB5 of the appendix.

In all cases, we also use 2 parameter regularization on w with weight 1e − 4 (i.e., the weight decay factor of the optimizer).

We compare MCL and its variants to SPL BID31 , SPLD BID26 and SGD with a random curriculum (i.e., with random batches).

Each method uses mini-batch SGD for π(·, η) with the same learning rate strategy to update w. The methods, therefore, differ only in the curriculum (i.e., the sequence of training sets).For SGD, in each iteration, we randomly select 4000 samples (20newsgroups) or 5000 samples (other datasets) and apply mini-batch SGD to the selected samples.

In SPL and SPLD, the training set starts from a fixed size k (4000 samples for 20newsgroups, 5000 samples for other datasets), and increases by a factor of 1 + µ (where µ = 0.1) per round of alternating minimization (i.e., per iteration of the outer loop) 4 .

We use ρ to denote the number of iterations of the inner loop, which aims to minimize the loss w.r.t.

the model w on the selected training set.

In SPLD, we also have a weight for the negative group sparsity: it starts from ξ and increases by a factor of 1.1 at each round of alternating minimization (i.e., per iteration of the outer loop).

We test five different combinations of {ρ, µ} and {ρ, ξ} for SPL and SPLD respectively.

The best combination with the smallest test error rate is what we report.

Neither SPL nor SPLD uses the clustering trick we applied to MCL: they compute the exact loss on each sample in each iteration.

Hence, they have more accurate estimation of the hardness on each sample, and require knowing the labels of all samples (selected and unselected) and cannot reduce annotation costs.

Note SPLD still needs to run clustering and use the resulted clusters as groups in the group sparsity (which measures diversity in SPLD).

We did not select samples with SPL/SPLD as we do with MCL since we wanted to test SPL/SPLD as originally presented -intuitively, SPL/SPLD should if anything only do better without such clustering due to the more accurate sample-specific hardness estimation.

The actual clustering, however, used for SPLD's diversity term is the same as that used for MCL's cluster samples.

We apply the mini-batch k-means algorithm to the features detailed in the next paragraph to get the clusters used in MCL and SPLD.

Although both SPL and SPLD can be reduced to SGD when λ → ∞ (i.e., all samples always selected), we do not include this special case because SGD is already a baseline.

For SGD with a random curriculum, results of 10 independent trials are reported.

In our MCL experiments, we use a simple "feature based" submodular function BID61 where F (A) = u∈U ω u c u (A) and where U is a set of features.

For a subset A of clusters, c u (A) = i∈A c u (i), where c u (i) is the nonnegative feature u of the centroid for cluster i, and can be interpreted as a nonnegative score for cluster i. We use TF-IDF features for 20newsgroup.

For the other datasets, we train a corresponding neural networks on a small random subset of training data (e.g., hundreds of samples) for one epoch, and use the inputs to the last fully connected layer (whose outputs are processed by softmax to generate class probabilities) as features.

Because we always use ReLU activations between layers, the features are all nonnegative and the submodularity of F (A) follows as a consequence.

These features are also used by mini-batch k-means to generate clusters for MCL and SPLD.For MCL, we set the number of inner loop iterations to p ≤ 50.

For each dataset, we choose p as the number among {10, 20, 50} that reduces the training loss the most in the first few iterations of the outer loop, and then use that p for the remaining iterations.

As shown in TAB6 , we use p = 50 for 20newsgroups, MNIST and Fashion-MNIST, and p = 20 for the other three datasets.

We consider five variants of MCL: 1) MCL(∆ = 0, λ = 0, γ = 0) having neither submodular regularization that promotes diversity nor scheduling of k that increases hardness; 2) MCL(∆ = 0, λ > 0, γ > 0), which decreases diversity by exponentially reducing the weight λ of the submodular regularization, but does not have any scheduling of k, i.e., k is fixed during the algorithm; 3) MCL(∆ > 0, λ > 0, γ = 0), which only uses the scheduling of k shown in Algorithm 1, but the diversity weight λ is positive and fixed during the algorithm, i.e., with γ = 0; 4) MCL-RAND(r,q), which randomly samples r clusters as a training setÂ after every q rounds of the outer loop in Algorithm 1, and thus combines both MCL and SGD; 5) MCL(∆ > 0, λ > 0, γ > 0), which uses the scheduling of both λ and k shown in Algorithm 1.

We tried five different combinations of {q, r} for MCL-RAND(r,q) and five different ∆ values for MCL(∆ > 0, λ > 0, γ > 0), and report the one with the smallest test error.

Other parameters, such as the initial values for λ and k, the values for γ and p, and the total number of clusters are the same for different variants (the exact values of these quantities are given in TAB6 of the Appendix).

In MCL, running greedy is the only extra computation comparing to normal SGD.

To show that in our implementation (see Section 4.6) its additional time cost is negligible, we report in TAB2 the total time cost for MCL(∆ > 0, λ > 0, γ > 0) and the time spent on our implementation WS-SUBMODULARMAX.We summarize the main results in FIG1 -8.

More results are given at the end of the appendix (Section 4.7).

In all figures, grey curves correspond to the ten trials of SGD under a random curriculum.

The legend in all figures gives the parameters used for the different methods using the following labels: 1) SPL (ρ, µ); 2) SPLD(ρ, ξ); and 3) MCL-RAND(q, r).

corresponding to training time.

Note only MCL and its variants use the clustering trick, while SPL/SPLD need to compute loss on every sample and thus require knowledge of the labels of all samples.

The left plot shows only the number of loss gradient calculations needed -1) in MCL, for those clusters never selected in the curriculum, the loss (and hence the label) of only the centroid sample is needed; 2) in SPL/SPLD, for those samples never selected in the curriculum, their labels are needed only to compute the loss but not the gradient, so they are not reflected in the left plots of all figures because their labels are not used to compute a gradient.

Therefore, thanks to the clustering trick, MCL and its variants can train without needing all labels, similar to semi-supervised learning methods.

This can help to reduce the annotation costs, if an MCL process is done in tandem with a labeling procedure analogous to active learning.

The right plots very roughly indicate convergence rate, namely how the test error decreases as a function of the amount of training.

On all datasets, MCL and most of its variants outperform SPL and SPLD in terms of final test accuracy (shown in Table 1 ) with comparable efficiency (shown in the right plots of all figures).

MCL is slightly slower than SGD to converge in early stages but it can achieve a much smaller error when using the same number of labeled samples for loss gradients.

Moreover, when using the same learning rate strategy, they can be more robust to overfitting, as shown in MCL(λ > 0, γ > 0, ∆ > 0) always outperforms MCL(∆ > 0, λ > 0, γ = 0), which supports our claim that it is better to decrease the diversity as training proceeds rather than keeping it fixed.

In particular, MCL(∆ > 0, λ > 0, γ = 0) shows slower convergence than other MCL variants in later stages.

In our experiments in the MCL(∆ > 0, λ > 0,γ = 0) case, we needed to carefully choose λ and use a relatively large ∆ for it to work at all, as otherwise it would repeatedly choose the same subset (with small ∆, the loss term decreases as training proceeds, so with fixed λ the diversity term comes to dominate the objective).

This suggests that a large diversity encouragement is neither necessary nor beneficial when the model matures, possibly since k is large at that point and there is ample opportunity for a diversity of samples to be selected just because k is large, and also since encouraging too much loss-unspecific diversity at that point might only select outliers.

The combination of MCL and random curriculum (MCL-RAND) speeds up convergence, and sometimes (e.g., on MNIST, SVHN and Fashion-MNIST) leads to a good final test accuracy, but requires more labeled samples for gradient computation and still cannot outperform MCL(λ > 0, γ > 0, ∆ > 0).

These results indicate that the diversity introduced by submodular regularization does yield improvements, and changing both hardness and diversity improves performance.

Proof.

We have DISPLAYFORM0

Proposition 1.

The maximum of multiple β-strongly convex functions is β-strongly convex as well.

Proof.

Let g(x) = max i g i (x), where g i (x) is β-strongly convex for any i. According to a definition of strongly convex function given in Theorem 2.1.9 (page 64) of BID46 , ∀λ ∈ [0, 1], we have DISPLAYFORM0 The following proves that g(x) is also β-strongly convex: DISPLAYFORM1

Proof.

The objective g(w) of the minimax problem in Eq. (2) after eliminating A is given in Eq. (4).

Since G(A) in Eq. FORMULA4 is monotone non-decreasing submodular, the optimal subset A when defining g(w) in Eq. (4) always has size k if |V | ≥ k. In addition, because the loss function L (y i , f (x i , w)) is β-strongly convex, g(w) in Eq. FORMULA3 is the maximum over multiple kβ-strongly convex functions with different A. According to Proposition 1, g(w) is also kβ-strongly convex, i.e., DISPLAYFORM0 Since the convex function g(w) achieves minimum on w * , it is valid to substitute ∇g(w * ) = 0 ∈ ∂g(w * ) into Eq. (14).

After rearrangement, we have DISPLAYFORM1 In the following, we will prove g(w * ) ≥ α · g(ŵ), which together with Eq. (15) will lead to the final bound showing how closeŵ is to w * .Noteĝ(w) (Eq. FORMULA5 ) is a piecewise function, each piece of which is convex and associated with differentÂ achieved by a submodular maximization algorithm of approximation factor α.

Sincê A is not guaranteed to be a global maxima, unlike g(w), the wholeĝ(w) cannot be written as the maximum of multiple convex functions and thus can be non-convex.

Therefore, gradient descent in lines 6-9 of Algorithm 1 can lead to either: 1)ŵ is a global minima ofĝ(w); or 2)ŵ is a local minima ofĝ(w).

Saddle points do not exist onĝ(w) because each piece of it is convex.

We are also assuming other issues associated with the boundaries between convex pieces do not repeatedly occur.

(16) The first inequality is due to g(·) ≥ĝ(·).

The second inequality is due to the global optimality ofŵ.

The third inequality is due to the approximation boundĝ(·) ≥ α · g(·) guaranteed by the submodular maximization in Step 7 of Algorithm 1.2) Whenŵ is a local minima ofĝ(w), we have ∇ĝ(ŵ) = 0.

Let h(w) be the piece ofĝ(w) whereŵ is located, thenŵ has to be a global minima of h(w) due to the convexity of h(w).

Let A denote the ground set ofÂ on all pieces ofĝ(w), we define an auxiliary convex functiong(w) as DISPLAYFORM0 It is convex because it is defined as the maximum of multiple convex function.

So we havê DISPLAYFORM1 The first inequality is due to the definition of A, and the second inequality is a result of A ⊆ 2 V by comparing g(w) in Eq. (4) withg(w) in Eq. (17).

Letw denote a global minima ofg(w), we have DISPLAYFORM2 The first inequality is due to Eq. (18), the second inequality is due to the global optimality ofw oñ g(w), the third inequality is due to the definition ofg(w) in Eq. FORMULA0 is the maximum of all pieces ofĝ(w) and h(w) is one piece of them), the fourth inequality is due to the global optimality of w on h(w), the last inequality is due to the approximation boundĝ(·) ≥ α · g(·) guaranteed by the submodular maximization in Step 7 of Algorithm 1.Therefore, in both cases we have g(w * ) ≥ α · g(ŵ).

Applying it to Eq. (15) results in DISPLAYFORM3

Proposition 2.

If x ∈ [0, 1], the following inequality holds true.

DISPLAYFORM0 Proof.

Due to two inequalities e x ≤ 1 + x + x 2 /2 for x ≤ 0 and 1 − e −x ≥ x/2 for x ∈ [0, 1], DISPLAYFORM1 4.5 PROOF OF COROLLARY 1Proof.

Applying the inequality in Proposition 2 and the approximation factor of lazy greedy α = (1 − e −κ G )/κ G to the right hand side of Eq. (9) from Theorem 1 yields DISPLAYFORM2 where κ G is the curvature of submodular function G(·) defined in Eq. (5).

Substituting the inequality about κ G from Lemma 1 into Eq. (23) results in DISPLAYFORM3 We use subscript as the index for iterations in the outer-loop, e.g.,ŵ T is the model weights w after the T th iteration of outer-loop.

If we decrease λ exponentially from λ = λ 0 and increase k linearly from k = k 0 , as Step 11 in Algorithm 1, we have DISPLAYFORM4 According to the definition of g(·) in Eq. (4), for g(w * T ) we have g(w * T ) = min DISPLAYFORM5 DISPLAYFORM6 Substituting Eq. FORMULA1 to Eq. (25) yields DISPLAYFORM7 If we can tolerate more expensive computational cost for running submodular maximization with larger budget k, and increase k exponentially, i.e., k ← (1 + ∆) · k, we have DISPLAYFORM8 This completes the proof.

Algorithm 1 repeatedly runs a greedy procedure to solve submodular maximization, and this occurs two nested loops deep.

In this section we describe how we speed this process up.

Our first strategy reduces the size of the ground set before starting a more expensive submodular maximization procedure.

We use a method described in BID60 where we sort the elements of V non-increasingly by G(i|V \ i) and then remove any element i from V having G(i) < G(δ(k)|V \ δ(k)) where δ(k) is k th element in the sorted permutation.

Any such element will never be chosen by the k-cardinality constrained greedy procedure because for any ∈ {1, 2, . . .

, k}, and any set A, we have DISPLAYFORM0 and thus greedy would always be able to choose an element better than i. This method results in no reduction in approximation quality, although it might not yield any speedup at all.

But with a decreasing λ, G(A) becomes more modular, and the filtering method can become more effective.

Other methods we can employ are those such as BID64 BID41 , resulting in small reduction in approximation quality, but we do not describe these further.

The key contribution of this section is a method exploiting a potential warm start set that might already achieve a sufficient approximation quality.

Normally, the greedy procedure starts with the empty set and adds elements greedily until a set of size k is reached.

In Algorithm 1, by contrast, a previous iteration has already solved a size-k constrained submodular maximization problem for the previous submodular function, the solution to which is one that could very nearly already satisfy a desired approximation bound for the current submodular function.

The reason for this is that, depending on the weight update method in line 9 of Algorithm 1 between inner loop iterations, and the changes to parameters ∆ and γ between outer iterations, the succession of submodular functions might not change very quickly.

For example, when the learning rate η is small, theÂ from the previous iteration could still be valued highly by the current iteration's function, so running a greedy procedure from scratch is unnecessary.

Our method warm-starts a submodular maximization process with a previously computed set, and offers a bound that trades off speed and approximation quality.

The approach is given in Algorithm 2, which (after the aforementioned filtering in line 3 BID60 ) tests in linear time if the warm start setÂ already achieves a sufficient approximation quality, and if so, possibly improves it further with an additional linear or quasilinear time computation.

To test approximation quality ofÂ, our approach uses a simple modular function upper bound, in line 4, to compute an upper bound on the global maximum value.

For the subsequent improvement ofÂ, our approach utilizes a submodular semigradient approach BID24 ) (specifically subgradients BID20 in this case).

If the warm-start setÂ does not achieve sufficient approximation quality in line 5, the algorithm backs off to standard submodular maximization in line 11 (we use the accelerated/lazy greedy procedure BID40 here although other methods, e.g., BID41 , can be used as well).Algorithm 2 Warm Start (WS) WS-SUBMODULARMAX(G, k,Â,α ∈ [0, 1)) 1: Input: G(·), k,Â,α 2: Output:Ã 3: Reduce ground set size: arrange V non-increasingly in terms of G(i|V \i) in a permutation δ where δ(k) is the k th element, set V ← {i ∈ V |G(i) ≥ G(δ(k)|V \δ(k))}; 4: Compute upper bound to maximum of Eq. FORMULA4 : DISPLAYFORM1 Permutation σ of V : the first k elements have S σ k =Â and are chosen ordered non-increasing by κ G (v); the remaining n − k elements V \Â for σ are chosen non-increasing by κ G (v).

Define modular function hÂ(A) i∈A hÂ(i) with hÂ(σ(i)) = G(S DISPLAYFORM0 Line 4 computes the upper bound τ ≥ max A∈V,|A|≤k G(A) which holds due to submodularity, requiring only a modular maximization problem (which can be done in O(|V |) time, independent of k, to select the top k elements).

Line 5 checks if anα approximation to this upper bound is achieved by the warm-start setÂ, and if not we back off to a standard submodular maximization procedure in line 11.IfÂ is anα approximation to the upper bound τ , then lines 6-9 runs a subgradient optimization procedure, a process that can potentially improve it further.

The approach selects a subgradient defined by a permutation σ = (σ(1), σ(2), . . .

, σ(n)) of the elements.

The algorithm then defines a modular function L(A), tight atÂ and a lower bound everywhere else, i.e., L(Â) = G(Â), and ∀A, L(A) ≤ G(A).

Any permutation will achieve this as long asÂ = {σ(1), σ(2), . . .

, σ(k)}. The specific permutation we use is described below.

Once we have the modular lower bound, we can do simple and fast modular maximization.

Lines 6-9 of Algorithm 2 offer a heuristic that can only improve the objective -lettingÃ be the solution after line 9, we have DISPLAYFORM1 The first inequality follows since L(·) is a lower bound of G(·); the second inequality follows from the optimality ofÂ + ; the equality follows since L is tight atÂ.The approximation factorα is distinct from the submodular maximization approximation factor α achieved by the greedy algorithm.

Setting, for exampleα = 1 − 1/e would ask for the previous solution to be this good relative to τ , the upper bound on the global maximum, and the algorithm would almost always immediately jump to line 11 since achieving such approximation quality might not even be possible in polynomial time BID18 .

Withα large, we recover the approximation factor of the greedy algorithm but ignore the warm start.

Ifα is small, many iterations might use the warm start from the previous iteration, updating it only via one step of subgradient optimization, but with a worse approximation factor.

In practice, therefore, we use a more lenient bound (often we set α = 1/2) which is a good practical tradeoff between approximation accuracy and speed (meaning lines 6-9 execute a reasonable fraction of the time leading to a good speedup, i.e., in our experiments, the time cost for WS-SUBMODULARMAX increases if α = 1 by a factor ranging from about 3 to 5).

In general, we have the following final bound based on the smaller ofα and α.

@highlight

Minimax Curriculum Learning is a machine teaching method involving increasing desirable hardness and scheduled reducing diversity.

@highlight

 A curriculum learning approach using a submodular set function that captures the diversity of examples chosen during training. 

@highlight

The paper introduces MiniMax Curriculum learning as an approach for adaptively training models by providing it different subsets of data. 