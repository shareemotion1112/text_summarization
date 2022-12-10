We introduce dynamic instance hardness (DIH) to facilitate the training of machine learning models.

DIH is a property of each training sample and is computed as the running mean of the sample's instantaneous hardness as measured over the training history.

We use DIH to evaluate how well a model retains knowledge about each training sample over time.

We find that for deep neural nets (DNNs), the DIH of a sample in relatively early training stages reflects its DIH in later stages and as a result, DIH can be effectively used to reduce the set of training samples in future epochs.

Specifically, during each epoch, only samples with high DIH are trained (since they are historically hard) while samples with low DIH can be safely ignored.

DIH is updated each epoch only for the selected samples, so it does not require additional computation.

Hence, using DIH during training leads to an appreciable speedup.

Also, since the model is focused on the historically more challenging samples, resultant models are more accurate.

The above, when formulated as an algorithm, can be seen as a form of curriculum learning, so we call our framework DIH curriculum learning (or DIHCL).

The advantages of DIHCL, compared to other curriculum learning approaches, are: (1) DIHCL does not require additional inference steps over the data not selected by DIHCL in each epoch, (2) the dynamic instance hardness, compared to static instance hardness (e.g., instantaneous loss), is more stable as it integrates information over the entire training history up to the present time.

Making certain mathematical assumptions, we formulate the problem of DIHCL as finding a curriculum that maximizes a multi-set function $f(\cdot)$, and derive an approximation bound for a DIH-produced curriculum relative to the optimal curriculum.

Empirically, DIHCL-trained DNNs significantly outperform random mini-batch SGD and other recently developed curriculum learning methods in terms of efficiency, early-stage convergence, and final performance, and this is shown in training several state-of-the-art DNNs on 11 modern datasets.

We study the dynamics of training a machine learning model, and in particular, the difficulty a model has over time (i.e., training epochs) in learning each sample from a training set.

To this end, we introduce the concept of "dynamic instance hardness" (DIH) and propose several metrics to measure DIH, all of which share the same form as a running mean over different instantaneous sample hardness measures.

Let a t (i) be a measure of instantaneous (i.e., at time t) hardness of a sample, where i is a sample index and t is a time iteration index (typically a count of mini-batches that have been processed).

In previous work, a t (i) has been called the "instance hardness" (Smith et al., 2014) corresponding to 1−p w (y i |x i ), i.e., the complement of the posterior probability of label y i given input x i for the i th sample under model w.

We introduce three different notions of instantaneous instance hardness in this work: (A) the loss (y i , F (x i ; w t )), where (·, ·) is the loss function and F (·; w) is the model with parameters w, (B) the loss change | (y i , F (x i ; w t )) − (y i , F (x i ; w t−1 ))| between two consecutive time steps, and (C) the prediction flip 1[ŷ t i =ŷ t−1 i ], whereŷ t i is the prediction of sample i in step t, e.g., argmax j F (x i ; w t )[j] for classification.

Our (A) corresponds closely to the "instance hardness" of Smith et al. (2014) .

However, our (B) and (C) require information from previous time steps.

Nevertheless, we consider (A), (B) , and (C) all variations of instantaneous instance hardness since they use information from only a very local time window around training iteration t. Dynamics is achieved when we compute a running average over instantaneous instance hardness, computed recursively as follows:

where γ ∈ [0, 1] is a discount factor, S t ⊆ V , and V = [n] is the set of all n training sample indices.

S t is the set of sample selected for training at time t by some method (e.g., a DIH-based curriculum learning (DIHCL) algorithm we introduce and study below) or simply a random batch.

In general, S t should be large early during training, but as r t (i) decreases to small values for many samples, choosing significantly smaller S t is possible to result in faster training and more accurate models.

We find that r t (i) can vary dramatically between different samples since very early stage (with small t).

One can think of this as some samples being more memorable and are retained more easily, while other samples are harder to learn and retain.

In addition, the predictions of the hard samples are less stable under changes in optimization parameters (such as the learning rate).

More importantly, once a sample's r t (i) is established (i.e., once t is sufficiently but not unreasonably large) each sample tends to maintain its DIH properties.

That is, a sample's DIH value converges relatively quickly to its final relative position amongst all of the samples DIH values.

For example, if a sample's DIH becomes small (i.e., meaning the sample is easily learned), it stays small relative to the other samples, or if it becomes large DIH (i.e., the sample is difficult to learn), it stays there.

I.e., once r t (i) for a sample has converged, its DIH status is retained throughout the remainder training.

We can therefore accurately identify categories of sample hardness relatively early in the course of training.

This suggests a natural curriculum learning strategy where S t corresponds mostly to those samples that are hard according to r t−1 (i).

In other words, the model concentrates on that which it finds difficult.

This is similar to strategies that improve human learning, such as the Leitner system for spaced repetition (Leitner, 1970) .

This is also analogous to boosting (Schapire, 1990 ) -in boosting, however, we average the instantaneous sample performance of multiple weak learners at the current time, while in DIHCL we average the instantaneous sample performance of one strong learner over the training history.

As mentioned above, instance hardness has been studied before (Smith et al., 2014; Prudencio et al., 2015; Smith & Martinez, 2016) where it corresponds to the complement posterior probability.

More recently, instance hardness has also been studied as an average over training steps in Toneva et al. (2019) where the mean of prediction flips over the entire training history is computed.

We note that Toneva et al. (2019) is a special case of DIH in Eq.

(1) with γ = 1 /t+1 and t = T , where T is the total number of training steps.

Our study generalizes Toneva et al. (2019) to the running dynamics computed during training.

This therefore leads to a novel curriculum learning strategy and also steps towards a better theoretical understanding of curriculum learning.

Also, in Toneva et al. (2019) , a small neural net is trained beforehand to determine the hard samples, and this is then used to train large neural nets.

In our approach, we take the average over time of a t (i), which requires no additional model or inference steps and hence is computationally trivial.

Another observation we find is that r t (i), for any sample, tends to monotonically decrease with t for any i. This means, not surprisingly, that during training samples become easier in terms of small DIH (i.e., they are better learned).

This also means that easy samples stay easy throughout training, and hard samples also become easier the more we train on them.

If we also make (admittedly) a mathematical leap, and assume that r t (i) is generated by the marginal gain of an unknown diminishing returns function f (·) that measures the quality of any curriculum, we can formulate DIHCL as an online learning problem that maximizes the unknown f (·) by observing its partial observation r t (i) over time for each i. Here, f is defined over an integer lattice and has a diminishing returns property, although the function is accessible only via the gains of every element.

This formulation provides a setting where the quality of the learnt curriculum is provably approximately good.

As will be shown below, DIHCL performs optimization in a greedy manner.

At each time step t, DIHCL selects a subset S t of samples using r t (i) where the hard samples have higher probabilities of being selected relative to the easy samples.

The model is then updated based only on the selected subset S t rather than V , which requires performing inference (e.g., a forward pass of a DNN) only on S t .

This therefore leads to a speedup to the extent that |S t | |V |.

The inference produces new instantaneous instance hardness a t (i) that is then used to update r t+1 (i) as in Equation 1.

To encourage exploration, improve stability, and get an initial estimate of r t (i) for all i ∈ V , during the first few epochs, DIHCL sweeps through the entire training set.

We provide several options for DIH-weighted subset sampling, which introduces different types of randomness in the selection since randomness is essential in optimizing non-convex problems.

Under certain additional mathematical assumptions, we also give theoretical bounds on the curriculum achieved by DIHCL compared to the optimal curriculum.

We empirically evaluate several variants of DIHCL and compare them against random mini-batch SGD as well as against recent curriculum learning algorithms, and test on 11 datasets including CIFAR10, CIFAR100, STL10, SVHN, Fashion-MNIST, Kuzushiji-MNIST, Food-101, Birdsnap, FGVC Aircraft, Stanford Cars and ImageNet.

DIHCL shows an advantage over other baselines in terms both of time/sample efficiency and test set accuracy.

Early curriculum learning (CL) (Khan et al., 2011; Basu & Christensen, 2013; Spitkovsky et al., 2009) work shows that feeding an optimized sequence of training sets (i.e., a curriculum), that can be designed by a human expert (Bengio et al., 2009) , into the training algorithms can improve the models' performance.

Self-paced learning (SPL) Tang et al., 2012a; Supancic III & Ramanan, 2013; Tang et al., 2012b) chooses the curriculum adaptive to some instance hardness (e.g., per-sample loss) during training.

SPL selects samples with smaller losses, and gradually increases the subset size over time to cover all the training data.

Self-paced curriculum learning (Jiang et al., 2015) combines the human expert in CL and loss-adaptation in SPL.

SPL with diversity (SPLD) (Jiang et al., 2014 ) adds a negative group sparse regularization term to SPL and increases its weight to increase selection diversity.

Machine teaching (Khan et al., 2011; Zhu, 2015; Patil et al., 2014) aims to find the optimal and smallest training subset leading to similar performance as training on all the data.

Minimax curriculum learning (MCL) (Zhou & Bilmes, 2018) argues that the diversity of samples is more critical in early learning since it encourages exploration, while difficulty becomes more useful later.

It also uses a form of instantaneous instance hardness (the loss) but is not dynamic like DIH, and formulates optimization as a minimax problem.

Compared to the above methods, DIHCL has the following advantages: (1) DIHCL improves the efficiency of CL since extra inference on the entire training set per step is not required; and (2) DIHCL uses DIH as the metric for hardness which is a more stable measure than instantaneous hardness.

Our paper is also related to Zhang et al. (2017) , which refers to overfitting in noisy data.

Our observations suggest that the learning of simple patterns (Arpit et al., 2017) happen mainly amongst the easy memorable early during in training (additional discussion is given in the Appendix and Figure 5 ).

Our paper is also distinct from catastrophic forgetting (Kirkpatrick et al., 2017) , which considers sequential learning of multiple tasks, where later learned tasks make the model forget what has been learned from earlier tasks.

In our work, we consider single task learning and show that easy samples remain easy.

If we make certain additional mathematical assumptions (as we do in our theoretical discussion below), our work is related to online submodular function optimization.

Specific forms have been studied including maximization (Streeter & Golovin, 2009; Chen & Krause, 2013) , maximization in the bandit setting with noisy feedback (Chen et al., 2017) , and continuous submodular function maximization (Chen et al., 2018b; a) .

The work most related to ours, perhaps, is the work on instance hardness (Smith et al., 2014; Prudencio et al., 2015; Smith & Martinez, 2016) , where the hardness of a sample corresponds to the complement posterior probability, as discussed above.

Also, a special case of DIH was studied in Toneva et al. (2019) : they compute DIH after training completes, and show that removing the easy samples (those having the smallest DIH over training set) leads to less degradation on generalization performance than removing random samples.

By contrast, our study of DIH focuses on its dynamics during training.

We start out by conducting an empirical study of DIH in DNN training.

We train a WideResNet with depth of 28 and width factor 10 on CIFAR10 dataset, and apply a modified cosine annealing learning rate schedule (Loshchilov & Hutter, 2017) for multiple episodes of increasing length (300 epochs in total) and target learning rate decaying.

We contend that a cyclic learning rate suits our study because: (1) it includes the most commonly used monotone decreasing schedule since the learning rate in each cycle is decreasing; (2) compared to monotone decreasing schedule, it can uncover the properties of DIH in more scenarios such as increasing learning rate and different decaying speeds of the learning rate.

In the study, we test two type of instantaneous instance hardness, where a t (i) is either prediction flips or loss (i.e., cases (A) and (C) in the previous section).

We visualize r t (i) for all i ∈ [50000] training samples, but divide [50000] them into three groups according to r t (i) (with a t (i) as prediction flips), and we do this at epoch either 10, 40, or 210.

For example, at epoch 40, the 10,000 samples with the largest r 40 (i) comprise the first group, the 10,000 samples ones with the smallest r 40 (i) comprise the next group, and the remaining 30,000 samples comprise the final group.

In Figure 1 , we plot the dynamics for the average prediction flips over each group (left plot) and the mean/standard deviation of loss in each group (right plot).

We observe that samples with small r t (i) are learned quickly in the early epochs and their losses remain small and predictions almost unchanged thereafter, indicating they are easy over time.

By contrast, the samples with large r t (i) have large variance, i.e., their losses oscillate rapidly between small and large values, and their predictions frequently change, indicating difficulty.

The quickly identified easy samples are never unlearnt, and do not suffer from any large loss later in the training.

The hard samples are also quickly identified, and remain difficult to learn even after training for many epochs.

On average, the dynamics on the hard samples is more consistent with the learning rate schedule, which implies that doing well on these samples can only be achieved at a shared sharp local minima.

This suggests that the model generalizes better in the regions containing the easy samples than those containing the hard ones.

Similar to human learning, a natural resolution is to learn the hard samples more frequently and reduce the learning on the easy, already learnt, ones.

That is, based on r t (i) (even during the early stages when t is not large), it is prudent to apply additional training effort on hard samples and begin to ignore already learnt easy samples.

Further empirical verification can be found in experiments and Figure 4 in the Appendix.

Figure 2 shows that all three types (base on (A), (B), and (C)) of DIH metrics decrease during training for both easy and hard samples.

This indicates that as learning continues, samples become less informative as more training takes place.

If we make additional mathematical assumptions, we can model the curriculum learning procedure as a scheme that maximizes an unknown diminishing return function f (·) via only observing its marginal gains r t (i).

A curriculum is a sequence of selected subsets, where each subset is a mini-batch of data points, i.e. (S 1 , S 2 , . . .

, S T ).

We define a function f :

is an non-negative integer valued vector of length |V | where z(i) counts how many times sample i has been selected in the T training data subsets.

Any subset S ⊆ V has a characteristic vector e S where e S (i) = 1 if i ∈ S and otherwise s S (i) = 0 if i / ∈ S. We also define S 1:t t =1 e St as a multi-set input to function f and S 1:t ∈ Z V ≥0 .

Ideally, our goal becomes finding the best curriculum (S 1 , S 2 , . . .

, S T ), i.e., one that maximizes f (S 1:T ), as in: max

where k t is a limit on the size of the set of samples selected at time t. However, f (·) can be an arbitrary unknown function in practice.

It is also intractable to estimate since it measures the utility of all possible training sequences (in exponential number), so it is inaccessible for optimization.

As a surrogate, information about f (·) might be available at each step in the form of the DIH values r t (i).

That is, if we make the mathematical assumption that there is some function f such that r t (i) = f (i|S 1:t−1 ) f (e i + S 1:t−1 ) − f (S 1:t−1 ), then r t (i) may be used in f 's stead and we can optimize the unknown f (·) only based on its partial observation r t (i).

In such case, DIHCL can be seen as a form of online optimization problem whose goal is to find a curriculum that maximizes f : at every step t, we select a subset of samples (e.g., a mini-batch) S t ⊆ V of size |S t | = k t to train the model, observing only marginal gains r t (i) of f (·) for each i. We can therefore define the following objective: max

For simplicity, we slightly overload the notation of S 1:T for function g(·) so that we retain information about the subset selected at every time step (i.e., we can extract S t for 1 ≤ t ≤ T from S 1:T ).

In practice, we update r t (i) only for samples in S t since it is a byproduct of training (i.e., the information needed to update r t (i) requires no more computation than what needed to train the model on S t ).

Hence, for i / ∈ S t , r t (i) = r t−1 (i).

Although our solution is an approximate solution to the ideal but intractable optimization in Eq. (2), we give its approximation bound in Corollary 1 to the global optimal solution of Eq. (2), and the bound is tight (the best we can get) up to a constant factor.

We arrive at a curriculum learning strategy that increases the probability learning on hard samples, and reduces learning on easy ones.

However, directly solving Eq. (3) requires costly inference over all the n training samples before selecting every subset S t , as most previous curriculum learning methods do.

Algorithm 1 DIH Curriculum Learning (DIHCL-Greedy) Let S t = argmax S:|S|=kt i∈S r t−1 (i); Apply optimization π(·; η) and record F (x i ; w t−1 ) for i ∈ S t ;

Compute normalized a t (i) for i ∈ S t using Eq. (4);

Update dynamic instance hardness r t+1 (i) using Eq. (1); 12:

Most optimization algorithms require inference on the training samples before updating the model parameters, which generates the predictions and losses for the samples used for training.

In step t, after the model gets trained on S t , the feedback a t (i) for i ∈ S t is already available.

However, for i / ∈ S t , extra inference is inevitable if the curriculum design requires instantaneous instance hardness on the remaining samples to select next subset S t+1 .

By contrast, DIHCL relies on r t (i) which is a running mean of a t (i), and it only updates r t (i) for i ∈ S t and keeps r t (i) for i / ∈ S t unchanged, thereby saves extra computation.

At step t of DIHCL, we select subset S t ⊆

[n] with large r t−1 (i) and then update the model by training on S t .

We then update r t (i) via Equation (1).

Since the learning rate can change over different steps, and large learning rates means greater model change, we normalize a t (i) by the learning rate η t−1 1 .

Specifically, we use one of the following depending on if we're in case (A), (B), or (C):

is the training data, π(·; η) is an optimization method such as SGD, η 1:T are the learning rates of steps 1 to T and γ k is the reduction factor for subset sizes k t .

DIHCL trains on more samples early on to produce an initial more accurate estimate of r t (i).

This is indicated by T 0 , the number of warm start epochs over the whole training set at the start.

After this, we start by selecting larger subsets each step and gradually reduce k t down to the most difficult samples as training proceeds.

A simple method to further reduce training time in the earlier stages is to extract a small subset of S t by encouraging the diversity of the selected samples.

We gradually reduce the diversity preference as training switching to the exploitation stage (reduce λ t by 0 ≤ γ λ ≤ 1 for every step t).

Inspired by MCL (Zhou & Bilmes, 2018) , after line 7, we reduce S t to a subset of size

by (approximately) solving the following submodular maximization.

max S⊆St,|S|≤k t i∈S

The function G : 2 St → R + is may be any submodular function (Fujishige, 2005) , and hence we can exploit fast greedy algorithms (Nemhauser et al., 1978; Minoux, 1978; Mirzasoleiman et al., 2015) to solve Eq. (5) with an approximation guarantee.

If in addition to the assumption that there exists some function f :

, we also assume that f has certain properties, then an approximation bound is achievable.

The diminishing return (DR) property for f can be defined if,

Recall that e i is a one-hot vector with all zeros except for a single one at the ith position.

We also assume f is normalized and monotone, i.e., f (0) = 0 and f (x) ≤ f (y), ∀ 0 ≤ x ≤ y. W.l.o.g.

we assume the max singleton gain is bounded by 1 (max i f (e i ) ≤ 1).

With such an assumption, we see that r t (i) is monotonically decreasing with increasing t. That is, r t (i) monotonically decreasing is a necessary, but not sufficient, condition for the DR property on f to hold.

Empirically we observe, in Figure 2 that r t (i) is indeed decreasing, meaning this evidence does not rule out there being a DR function governing r t (i)'s behavior.

On the other hand, this of course does not guarantee the DR property.

Nevertheless, if it is the case that r t (i) is produced as above from some DR function, it enables the following analysis to proceed.

Under the above assumptions, we may derive bounds of DIHCL-Greedy(Alg.

1) when k t = k ∀t ∈ [T ].

For simplicity, assume n mod k = 0 and let m n k .

We first show the bound on function g of observed gains, and then connect it to the unknown function f .

on ground set V with DR property, compared to any solution S * 1:T , S 1:T , the solution of DIHCL-Greedy, achieves

Where C f,m m min A1:m g(A 1:m ) such that

The proofs are given the Appendix.

The C f,m term in the bound reflects our loss during the warm start phase, where we cannot estimate the gain of each sample unless we select each sample at least once, which is independent of T and vanishes in the long run.

The 1 − e −1 comes from the DR property and our greedy procedure.

For the 1/k factor and the k/n factor of the bound on g, we give hard cases in the Appendix so our bound is tight to constant factors.

These factors result from our assumption about the function f , which may have arbitrary interactions among data points.

In practice, similar data points tend to have similar DIH, and we can incorporate such information by adding an additional term of submodular function G to the DIH value to model data point interactions.

In line 7 of Alg.

1, we select S t with the highest r t−1 (i) values.

In practice, we find adding randomness to the selection procedure gives better performance as (1) exploration on samples with small r t (i) is necessary for accurate estimate to r t (i), and (2) randomness of training samples is essential to achieve a good quality solution w for non-convex models such as DNNs.

Instead of choosing greedily the top k t samples, we perform random sampling with probability p t,i ∝ h(r t−1 (i)), where h(·) is a monotone non-decreasing function, and we still prefer data points with high DIH.

An ideal choice of h(·) should balance between the exploration of data with poorly estimated DIH and exploitation of data with well estimated DIH.

We propose the following three sampling methods to replace line 7 of Alg.

1, and give extensive evaluations in the experiment section.

Let h(r t (i)) = r t (i).

We sample data points weighted by their DIH values.

We trade-off exploration and exploitation similarly to Exp3 (Auer et al., 2003) , which samples based on the softmax value and reweigh the observation by the selection probability to encourage exploration:

DIHCL-Beta: We utilize the idea of Thompson sampling (Thompson, 1933) and use a Beta distribution prior to balance exploration and exploitation, i.e., h(r t (i)) ∼ Beta(r t (i), c − r t (i)), where c is a sufficiently large constant that c ≥ r t (i), e.g., c = 1 when a t (i) is prediction flip.

The Beta distribution encourages exploration when the difference between r t (i) and c − r t (i) is small.

and SVHN (Netzer et al., 2011) .

We use mini-batch SGD with momentum of 0.9 and cyclic cosine annealing learning rate schedule (Loshchilov & Hutter, 2017 ) (multiple episodes with starting/target learning rate decayed by a multiplicative factor 0.85).

We use T 0 = 5, γ = 0.95, γ k = 0.85 for all DIHCL variants, and gradually reduce k from n to 0.2n.

On each dataset, we apply each method to train the same model for the same number of epochs, but each method may select different amount of samples per epoch.

More details about the datasets and settings can be found in the Appendix.

For DIHCL variants that further reduce S t by solving Eq. (5), we use λ 1 = 1.0, γ λ = 0.8, γ k = 0.4 and employ the "facility location" submodular function (Cornuéjols et al., 1977 ) G(S) = j∈St max i∈S ω i,j where ω i,j represents the similarity between sample x i and x j .

We utilize a Gaussian kernel for similarity using neural net features (e.g., the inputs to the last fully connected layer in our experiments) z(x) for each x, i.e.,

2 , where σ is the mean value of all the k(k−1) /2 pairwise distances.

In Figure 3 , we show how the test set accuracy changes when increasing the number of training batches in each curriculum learning method on 3 datasets.

The results for other 8 datasets can be found in the Appendix, together with the wall-clock time for (1) the entire training and (2) the submodular maximization part in DIHCL with diversity and MCL.

The final test accuracy achieved by each method is reported in Table 1 .

DIHCL and its variants show significantly faster and smoother gains on test accuracy than baselines during training especially at earlier stages.

They also achieve higher final accuracy and show improvements in sample efficiency (meaning they reach their best performance sooner, after less computation has taken place).

MCL can reach similar performance as DIHCL on some datasets but it shows less stability and requires more computation for submodular maximization.

We also observe a similar instability of SPL.

The reason is that, compared to the methods that use DIH, both MCL and SPL deploy instantaneous instance hardness (i.e., current loss) as the score to select samples, a measure that is more sensitive to randomness and perturbation that occurs during training.

Compared to MCL and DIHCL, SPL and the random mini-batch curriculum method requires more epochs to reach their best accuracy, since they spend training effort on the easier and memorable samples but lack sufficient repeated-learning of the forgettable ones.

Although every variant of DIHCL achieves the best accuracy among all the evaluated methods on some datasets, DIHCL-Exp using loss and DIHCL-Beta using prediction flip, as the instantaneous hardness, exhibit advantages over the other DIHCL variants.

One possible explanation is that the running mean computed on the loss and prediction flips are more stable along the training trajectory as shown in Figure 2 , or perhaps they are more in line with our assumption in Section 3.2 about the diminishing return property of f (·).

A PROOFS f : Z V ≥0 → R ≥0 on ground set V is defined over an integer lattice.

The diminishing return (DR) property of f is the following inequality 0 ≤ ∀x ≤ y:

Where e i is a one-hot vector with all zeros except for a single one at the ith position.

We assume f is normalized and monotone, i.e., f (0) = 0 and f (x) ≤ f (y), ∀0 ≤ x ≤ y. W.l.o.g.

we also assume the max singleton gain is bounded by 1, i.e., max i f (e i ) ≤ 1.

We can think that f takes input as a multi-set, and the gain of an item diminishes as its counter increases in the multi-set.

In the setting of selecting mini-batches for training machine learning models, suppose the mini-batch size is k, the training set is V , and at every time step t, we select S t ⊆ V with |S t | = k, and only observe the gains on the selected subset (e.g., for neural networks, we update the running mean of training losses during the forward pass of the chosen mini-batch, or DIH type (A)).

At every time step of selecting a mini-batch, we observe f (i|S 1:t−1 ) ∀i ∈ S t .

Let n = |V |, m = n k , and for simplicity assume n mod k = 0.

We define function g to reflect the observed gains from f as we select data samples at each training step:

For simplicity, we slightly overload the notation of S 1:T for function g(·) so that we retain information about the subset selected at every time step (i.e., we can extract S t for 1 ≤ t ≤ T from S 1:T ).

Note that g is permutation-variant for k > 1, i.e., for different ordering in S 1:t , g gives different values.

on ground set V with DR property, compared to any solution S * 1:T , S 1:T , the solution of DIHCL-Greedy, achieves

Where C f,m m min A1:m g(A 1:m ) such that m i=1 A i = V, and |A i | = k.

To bridge S 1:T with S * 1:T , we first connect S 1:T to the greedy solution with singleton gain oracle, but uses the history of sequence of (S 1 , S 2 , . . .

, S T −1 ), which we denote by (Ŝ 1 ,Ŝ 2 , . . .

,Ŝ T ):

S t = argmax S⊆V,|S|=k i∈S f (i|S 1:t−1 ).

Note we denote any set with subscript 0 (at time step 0) as an empty set, i.e. S 0 = ∅,Ŝ 0 = ∅, and etc..

We define the observed gain values on the singleton gain oracle with history of (S 1 , S 2 , . . .

, S T −1 ) as:

Firstly, we derive a lower bound of g(S 1:T ) in terms of g(Ŝ 1:T |S 1:T −1 ).

Proof.

Define ζ(i, A 1:t ) to return the subsequence of A 1:t that starts from A 1 and ends at A t where A t is the last set in the whole sequence that contains the element i, i.e., ζ(i, A 1:t ) = argmax A 1:t t 1 i∈A t .

When i is not present in the whole sequence A 1:t , ζ(i, A 1:t ) returns ∅.

By definitions of C f,m and g(Ŝ 1:m |S 1:m−1 ), we have C f,m ≥ mf (V ) ≥ g(Ŝ 1:m |S 1:m−1 ) due to the diminishing return (DR) property.

For T ≤ m, Lemma 1 is true because of the above inequality.

For T ≥ m + 1, we compare the previous gains of elements in S t to the current gains of elements in S t :

Eq. 13 and Eq. 16 hold due to the diminishing return property and Eq. 15 is a result of the greedy step (i.e., S t is optimal when conditioning on ζ(i, S 1:t−1 )).

Note that we are guaranteed to find an element in the sequence history (|ζ(i, S 1:t−1 )| > 0 in Eq. 14 and Eq. 15) since we sweep the ground set V in the first m steps of solution S 1:m .

Remarks.

In the proof, we ignore the gain at the T step, i.e., i∈S T f (i|S 1:T −1 ) as such gain can potentially be zero.

In other words, g(S 1:T −1 ) + C f,m ≥ g(Ŝ 1:T |S 1:T −1 ).

For the case that f is modular, i.e., f (x + e i ) = f (x) + f (e i ), and for only k elements in V , the function evaluations are non-zero, the bound meets in equality: g(S 1:T −1 ) + C f,m = g(Ŝ 1:T |S 1:T −1 ).

The idea is that we have to sweep all elements in the ground set before we identify the non-zero-valued elements.

Next, we find a lower bound of g(Ŝ 1:T |S 1:T −1 ) in terms of g(S * 1:T ).

Proof.

For T < T , we compare g(S * 1:T ) with g(Ŝ 1:T |S 1:T −1 ):

From Eq. 19 to Eq. 20, we use t=1:T max i∈S * t f (i|S * 1:t−1 ) ≤ t=1:T f (S * t |S * 1:t−1 ) = f (S * 1:T ).

Eq. 22 is due to DR property and Eq. 23 is a result of greedy selection.

Also note that for Eq. 21,

By rearranging Eq. 25, we have

, every time step, we reduce the gap to 1/k of the of optimal solution by at least

Remarks.

We will show that there is a hard case with 1/k factor.

Suppose f is a set cover function (f (i|A) = 0 if i ∈ A) and |V | = k 2 .

The ground set V is partitioned into

For the first time step, g(Ŝ 1 |∅) gets a gain of k which is equal to g(S * 1 ).

However, S 1 may select one element from each of the group since we are doing the ground set sweeping exploration, and all the rest gains will be zero conditioned on S 1 .

The optimal solution, on the other hand, can select all k elements from one group at a time, and get a value of k 2 in the end.

Combine Lemma 1 and Lemma 2, we get the first factor

for the bound in Theorem 1.

Proof.

We will first connect g(Ŝ 1:T |S 1:T −1 ) with the solution that selects the entire ground set V at every step, i.e., g(V 1:T ) = g((V, V, . . .

, V )).

For Eq. 26, we use the fact thatŜ 1:T achieve the top k gains selected by the greedy process in each step.

Next, we will bound any solution g(S * 1:T ) by g(V 1:T |V 1:T −1 ).

Firstly, we will need to partition S * 1:T into two parts: (1) for the first part, we collect all the new elements introduced at every time step t that do not exist in S * 1:t−1 , i.e.,S * 1:T = (S * 1 \ ∅, S * 2 \ ∪(S * 1:1 ), S * 3 \ ∪(S * 1:2 ), . . .

, S * T \ ∪(S * 1:T −1 )), whereS * t S * t \ ∪(S * 1:t−1 ) and ∪(S 1:t ) i=1:t S i , which is the set union on all elements in the multiset (you can think it sets all the counters in the multiset with values ≥ 1 to ones), and "\" is the set minus operation.

Therefore,S * 1:T contains every element in S * 1:T exactly once, i.e., every element in S * 1:T only appears once inS * 1:T , and at many time steps,S * t might be empty; (2) the other part contains all the rest elements, i.e., S * 1:T −S * 1:T = (S * 1 \S * 1 , S * 2 \S * 2 , . . .

, S * T \S * T ).

We bound the two parts as follows:

From Eq. 31 to Eq. 32, we use the fact t=1:T i∈S * t f (i|S *

T contains one instance of every element in S * 1:T and removing the conditioning part would make the gains larger (guaranteed by diminishing return property).

To get Eq. 33, we reduce the conditioning part of f (i|S * 1:t−1 ) in Eq. 32 by using the following inequality: for

According to the pre-defined partition, we pick out the first occurrence of every element intoS * 1:T , every remaining element i ∈ (S * t \S * t ) is guaranteed to find itself in its conditioning history S * 1:t−1

and therefore, we may use the inequality described in Eq. 35 to bound the second term in Eq. 32 by f (S * 1:T ) (letting A 1 = S * t \S * t and A 2 = S * 1:t−1 and applying the inequality from t = 1 to T sequentially).

To make it more concrete, for example, at step t = 2, by using Eq. 35, we have:

At time step t = 3, we have:

(39) Hence, we have the inequality between Eq. 32 and Eq. 33.

To get Eq. 34 from Eq. 33, we use the fact i∈V f (i) ≤ g(V 1:T |V 1:T −1 ) because g(V 1:T |V 1:T −1 ) contains i∈V f (i) at step t = 1, and the second term in Eq. 34 is due to the fact that f (·) is monotone non-decreasing.

Finally, we combine Eq. 29 and Eq. 34, we get 2g(Ŝ 1:T |S 1:

By combining Lemma 1 and Lemma 3, we get the second factor k 2n for the bound in Theorem 1.

1−e −1 k dominates when k is relatively small compared to n. Recall the hard case example above on the 1/k factor.

We can generalize it to any k < n by (almost) equally distribute the n elements into the k groups described in the hard case.

Then, for n < k 2 , the optimal solution gets n in the end while the greedy solution gets k, so the ratio is k n .

For n ≥ k 2 , the optimal solution still gets k 2 while the greedy solution gets k, so the ratio is 1 k .

In both scenarios, our bounds match the hard example up to constant factors.

Remarks.

We will show there is a hard case with the 1/k 2 factor.

The same as the hard case mentioned above for the set cover function, f (Ŝ 1 ) gets a gain of 1 since the selected items can be totally redundant, and the future gains are all zeros since S 1 select one element from each group.

However, the optimal solution can still achieve an evaluation of k 2 in the end.

Also, note that Theorem 1 is true for any solution S * 1:T and the optimal solution for g and the optimal solution for f can be different.

We mentioned a few weighted sampling method to replace the greedy step.

Here, we apply a random sampling procedure similar to the lazier-than-lazy approach Mirzasoleiman et al. (2015): we sample a subset R j ⊆

V \ S t,j−1 of size n k log 1 , and then choose the top-gain element from R j and add it to S t,j−1 to from S t,j .

We denote such sampling based greedy as DIHCL-Greedy-random.

Theorem 2.

For f : Z V ≥0 → R ≥0 on ground set V with DR property, compared to any solution S * 1:T , S 1:T , the solution of DIHCL-Greedy-random, achieves

Proof.

We can think the selection of every S t is a greedy process of k steps, with S t as the optimal solution.

Suppose up to step j, we select the set S t,j .

We first bound the probability that the sampled set has some intersection with the optimal set S t .

(47)

(49) In step j, we denote the selected item by v j .

We can then get the expected gain given the probability that there is some intersection:

Again, we get the argument that we are reducing the gap to the optimal solution by (1 − )/k for every selected item v j on expectation.

We can then apply Eq. 52 in the Eq. 14 of Lemma 1, and get

Combine with Lemma 2 we get the bound in Theorem 2.

Remarks.

When n is large and n k, we can approximate the sample without replacement using sample with replacement, and we can independently sample k subsets each of size |R| at every time step to generate S k .

In such a case, the bound becomes E[g(S 1:

.

Similarly, we can also get the expectation bound on f :

Proof.

We can extend the setting so that we get noisy feedback from the gains of function f : f (i|S 1:t−1 )+α t , and the problem becomes a multi-armed bandit problem.

Specifically if we assume the noise a t form a martingale difference sequence, i.e. E[α t |α 1 , α 2 , . . .

, α t−1 ] = 0 and all α t are bounded α t ≤ σ and if we make further assumption about the smoothness of the f and g function (assume the gains of f and g have RKHS-norm bounded by value B with some kernel k), we can utilize the contextual bandit UCB algorithm proposed in Krause & Ong (2011) to get a √ T dependent regret.

Also, under the noise setting, the contextual information becomes crucial, as the function has DR-property, and without an estimate of how much the gain decreases, we cannot have a better estimate of the upper bound on the noise term.

However, we note that utilizing the contextual information involves calculating large kernel matrices, which is not feasible for our purpose of efficient curriculum learning.

We include the following result for completeness.

Theorem 3.

For f : Z V ≥0 → R ≥0 on ground set V with DR property, suppose the gain of function g has RKHS-norm bounded by value B with some kernel k), and the noise α t 's from a martingale difference sequence: E[α t |α 1 , α 2 , . . .

, α t−1 ] = 0 and all α t are bounded |α t | ≤ σ.

We define the maximum information gain if we have the perfect information about f , ρ T = max A 1:T H(y A 1:T ) − H(y A 1:T |f ), where H is the Shannon entropy, and y A 1:T = {f (i|A 1:t−1 ) + a t |i ∈ A t , t = 1 : T } denotes the collection of gain values we get from the sequence of A 1:T .

We get the following regret bound:

Proof.

The proof directly utilizes the third case of Theorem 1 in Krause & Ong (2011) , using the history sequence (S 1 , S 2 , . . .

, S t ) as the context:

Combine with Lemma 1, we have:

B DYNAMIC INSTANCE HARDNESS (CONT.) Firstly, we present a quantitative verification of the second observation in Section 2, i.e., dynamic instance hardness in early stages can predict later dynamics.

It tries to predict the samples with large/small DIH values in the future by only using the DIH computed on early training history.

In Figure 4 , we show two upper triangle matrices quantitatively verifying the above statement.

They are computed based on the results of the CIFAR10 training experiment in Section 2.

Take the matrix A in the left plot for example, given U i , the top-10k samples with the largest DIH values computed in epoch 15i, and U j for any j > i, the entry A i,j = |Ui∩Uj | /10000.

Similarly, the matrix in the right plot measures the same overlapping percentage for the top-10k samples with the smallest DIH values between epoch 15i and epoch 15j.

They show that after a few first epochs, DIH can accurately predict the forgettable and memorable samples in the future.

This verifies the second statement we made in Section 2.

In addition, they also show that |Ui∩Uj | /10000 between consecutive epochs 15i and 15j is close to 100%, which indicates that DIH is a stable and smoothly changed metric with high consistency across training trajectory.

partitioned by a DIH metric (i.e.,running mean of prediction-flip) computed at epoch 10,40 and 60 during training WideResNet-28-10 on CIFAR10 with random labels.

In this setting, the random (but wrong) labels will be remembered very well after some training, and DIH in early stages loses the capability to predict the future DIH, i.e., they can only reflect the history but not the future.

This characteristic of DIH might be helpful to detect noisy data.

Secondly, we conduct an empirical study of dynamic instance hardness during training a neural net on very noisy data, as studied in (Zhang et al., 2017) and (Arpit et al., 2017) .

In particular, we replace the ground truth labels of the training samples by random labels, and apply the same training setting used in Section 2.

Then, we compute the running mean of prediction-flip for each sample at some epoch (i.e., 10, 40, 60) , and partition the training samples into three groups, as we did to generate Figure 1 .

The result is shown in Figure 5 .

It shows 1) the group with the smallest prediction flip over history (left plot) is possible to have large but unchanging loss as shown in the right plot; and 2) the DIH in this case can only reflect the history but cannot predict the future.

However, it also indicates that the capability of DIH to predict the future is potential to be an effective metric to distinguish noisy data or adversarial attack from real data.

We will discuss it in our future work.

Thirdly, we change the WideResNet to a much smaller CNN architecture with three convolutional layers 2 .

We apply the same training setting used in Section 2.

Then, we compute the running mean of prediction-flip for each sample at some epoch (i.e., 10, 40, 140, 210) , and partition the training samples into three groups, as we did to generate Figure 1 .

The result is shown in Figure 6 .

Compared to DIH of training deeper and wider neural nets shown in Figure 1 , the memorable and forgettable samples are indistinguishable until very late stages, e.g., Epoch-140.

This indicates that using DIH in earlier stage to select forgettable samples into curriculum might not be reliable when training small neural nets.

We will leave explanation of this phenomenon to our future works.

Moreover, we provide a comparison of the smoothness between DIH and instantaneous loss on individual samples in Figure 7 .

It shows that the DIH is a smooth and consistent measure of the learning/memorization progress on individual samples.

In contrast, the frequently used instantaneous loss is much noisier, so selecting training samples according to it will lead to unstable behaviors during training.

In Figure 8 , we also provide a comparison of DIH and instantaneous loss on the two groups of samples in Figure 2 , which shows a similar phenomenon.

We use cosine annealing learning rate schedule for multiple episodes.

The switching epoch between each two consecutive episode for different datasets are listed below.

• CIFAR10, CIFAR100: (5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300); • Food-101, Birdsnap, FGVCaircraft, StanfordCars: (10, 20, 30, 40, 60, 90, 150, 240, 400); • ImageNet: (5, 10, 15, 20, 30, 45, 75, 120, 200); • STL10: (20, 40, 60, 80, 120, 160, 240, 360, 560, 840, 1200); • SVHN: (5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300) ; • KMNIST, FMNIST: (5, 10, 15, 20, 30, 40, 60, 90, 140, 210, 300) ;

We report how the test accuracy changes with the number of training batches for each method, and the wall-clock time for all the 11 datasets in Figure 9 -12.

<|TLDR|>

@highlight

New understanding of training dynamics and metrics of memorization hardness lead to efficient and provable curriculum learning.

@highlight

This paper formulates DIH as a curriculum leaning problem that can more effectively utilize the data to train DNNs, and derives theory on the approximation bound.