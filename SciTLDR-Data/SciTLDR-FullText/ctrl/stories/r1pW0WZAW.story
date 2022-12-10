Recurrent neural networks (RNNs) have achieved state-of-the-art performance on many diverse tasks, from machine translation to surgical activity recognition, yet training RNNs to capture long-term dependencies remains difficult.

To date, the vast majority of successful RNN architectures alleviate this problem using nearly-additive connections between states, as introduced by long short-term memory (LSTM).

We take an orthogonal approach and introduce MIST RNNs, a NARX RNN architecture that allows direct connections from the very distant past.

We show that MIST RNNs 1) exhibit superior vanishing-gradient properties in comparison to LSTM and previously-proposed NARX RNNs; 2) are far more efficient than previously-proposed NARX RNN architectures, requiring even fewer computations than LSTM; and 3) improve performance substantially over LSTM and Clockwork RNNs on tasks requiring very long-term dependencies.

Recurrent neural networks BID33 Werbos, 1988; BID35 ) are a powerful class of neural networks that are naturally suited to modeling sequential data.

For example, in recent years alone, RNNs have achieved state-of-the-art performance on tasks as diverse as machine translation , speech recognition BID29 , generative image modeling BID30 , and surgical activity recognition BID8 .These successes, and the vast majority of other RNN successes, rely on a mechanism introduced by long short-term memory BID20 BID14 , which was designed to alleviate the so called vanishing gradient problem (Hochreiter, 1991; BID3 .

The problem is that gradient contributions from events at time t − τ to a loss at time t diminish exponentially fast with τ , thus making it extremely difficult to learn from distant events (see FIG0 .

LSTM alleviates the problem using nearly-additive connections between adjacent states, which help push the base of the exponential decay toward 1.

However LSTM in no way solves the problem, and in many cases still fails to learn long-term dependencies (see, e.g., BID0 ).

1 RNNs BID27 offer an orthogonal mechanism for dealing with the vanishing gradient problem, by allowing direct connections, or delays, from the distant past.

However NARX RNNs have received much less attention in literature than LSTM, which we believe is for two reasons.

First, as previously introduced, NARX RNNs have only a small effect on vanishing gradients, as they reduce the exponent of the decay by only a factor of n d , the number of delays.

Second, as previously introduced, NARX RNNs are extremely inefficient, as both parameter counts and computation counts grow by the same factor n d .In this paper, we introduce MIxed hiSTory RNNs (MIST RNNs), a new NARX RNN architecture which 1) exhibits superior vanishing-gradient properties in comparison to LSTM and previouslyproposed NARX RNNs; 2) improves performance substantially over LSTM on tasks requiring very long-term dependencies; and 3) remains efficient in parameters and computation, requiring even fewer than LSTM for a fixed number of hidden units.

Importantly, MIST RNNs reduce the decay's exponent by a factor of 2 n d −1 ; see FIG1 .

2 BACKGROUND AND RELATED WORK Recurrent neural networks, as commonly described in literature, take on the general form DISPLAYFORM0 which compute a new state h t in terms of the previous state h t−1 , the current input x t , and some parameters θ (which are shared over time).One of the earliest variants, now known to be especially vulnerable to the vanishing gradient problem, is that of simple RNNs (Elman, 1990), described by DISPLAYFORM1 In this equation and elsewhere in this paper, all weight matrices W and biases b collectively form the parameters θ to be learned, and tanh is always written explicitly 2 .Long short-term memory BID20 BID14 , the most widelyused RNN architecture to date, was specifically introduced to address the vanishing gradient problem.

The term LSTM is often overloaded; we refer to the variant with forget gates and without peephole connections, which performs similarly to more complex variants BID16 : DISPLAYFORM2 Here σ(·) denotes the element-wise sigmoid function and denotes element-wise multiplication.

f t , i t , and o t are referred as the forget, input, and output gates, which can be interpreted as controlling how much we reset, write to, and read from the memory cell c t .

LSTM has better gradient properties than simple RNNs (see FIG1 ) because of the mechanism in Equation 7, which introduces a path between c t−1 and c t which is modulated only by the forget gate.

We also remark that gated recurrent units BID5 alleviate the vanishing gradient problem using this exact same idea.

NARX RNNs BID27 also address the vanishing gradient problem, but using a mechanism that is orthogonal to (and possibly complementary to) that of LSTM.

This is done by allowing delays, or direct connections from the past.

NARX RNNs in their general form are described by but literature typically assumes the specific variant explored in BID27 , DISPLAYFORM3 DISPLAYFORM4 which we refer to as simple NARX RNNs.

Note that simple NARX RNNs require approximately n d as much computation and n d as many parameters as their simple-RNN counterpart (with n d = 1), which greatly hinders their applicability in practice.

To our knowledge, this drawback holds for all NARX RNN variants before MIST RNNs.

For example, in (Soltani & Jiang, 2016) , higher-order recurrent neural networks (HORNNs) are defined precisely as simple NARX RNNs, and every variant in the paper suffers from this exact same problem.

And, in BID37 , a simple NARX RNN architecture is defined that is limited to having precisely two delays with non-zero weights.

This way, at the expense of having fewer, longer paths to the past, parameter and computation counts are only doubled.

The previous work that is most similar to ours is that of Clockwork RNNs BID22 , which split weights and hidden units into partitions, each with a distinct period.

When it's not a partition's time to tick, its hidden units are passed through unchanged, and so Clockwork RNNs in some ways mimic NARX RNNs.

However Clockwork RNNs differ in two key ways.

First, Clockwork RNNs sever high-frequency-to-low-frequency paths, thus making it difficult to learn long-term behavior that must be detected at high frequency (for example, learning to depend on quick motions from the past for activity recognition).

Second, Clockwork RNNs require hidden units to be partitioned a priori, which in practice is difficult to do in any meaningful way.

NARX RNNs (and in particular MIST RNNs) suffer from neither of these drawbacks.

Many other approaches have also been proposed to capture long-term dependencies.

Notable approaches include maintaining a generative model over inputs and learning to process only unexpected inputs (Schmidhuber, 1992) , operating explicitly at multiple time scales BID9 , Hessian-free optimization BID28 , using associative or explicit memory BID32 BID7 BID15 BID34 , and initializing or restricting weight matrices to be orthogonal BID0 BID18 .

In BID3 BID31 , gradient decompositions and sufficient conditions for vanishing gradients are presented for simple RNNs, which contain one path between times t − τ and t. Here, we use the chain rule for ordered derivatives (Werbos, 1990) to connect gradient components to paths and edges, which in turn provides a simple extension of the results from BID3 BID31 to general NARX RNNs.

We remark that we rely on slightly overloaded notation for clarity, as otherwise notation becomes cumbersome (see (Werbos, 1989) ).We begin by disambiguating notation, as the symbol ∂f ∂x is routinely overloaded in literature.

Consider the Jacobian of f (x, u(x)) with respect to x. We let DISPLAYFORM0 ∂x , a collection of full derivatives, and we let DISPLAYFORM1 ∂x , a collection of partial derivatives.

This lets us write the ordinary chain rule as DISPLAYFORM2 ∂x .

Note that this notation is consistent with (Werbos, 1989; BID10 , but is the exact opposite of the convention used in BID31 .

Consider an ordered system of n vectors v 1 , v 2 , . . .

, v n , where each is a function of all previous: DISPLAYFORM0 The chain rule for ordered derivatives expresses the full derivatives DISPLAYFORM1 ∂vj for any j < i in terms of the full derivatives that relate v i to all previous v k : DISPLAYFORM2

Consider NARX RNNs in their general form FORMULA3 ), which we remark encompasses other RNNs such as LSTM as special cases.

Also, for simplicity, consider the situation that is most often encountered in practice, where the loss at time t is defined in terms of the current state h t and its own parameters θ l (which are independent of θ).

DISPLAYFORM0 (This is in not necessary, but we proceed this way to make the connection with RNNs in practice evident.

For example, f l may be a linear transformation with parameters θ l followed by squarederror loss.)

Then the Jacobian (or transposed gradient) with respect to θ can be written as DISPLAYFORM1 because the additional term DISPLAYFORM2 , and so on in Equations 11 and 12, we immediately obtain DISPLAYFORM3 because all of the partials ∂xt−τ ∂θ are 0.

Equations 14 and 15 extend Equations 3 and 4 of BID31 to general NARX RNNs, which encompass simple RNNs, LSTM, etc., as special cases.

This decomposition breaks ∂ + ht ∂θ into its temporal components, making it clear that the spectral norm of DISPLAYFORM4 ∂ht−τ plays a major role in how h t−τ affects the final gradient DISPLAYFORM5 In particular, if the norm of DISPLAYFORM6 ∂ht−τ is extremely small, then h t−τ has only a negligible effect on the final gradient, which in turn makes it extremely difficult to learn from events that occurred at t − τ .

Equations 14 and 15, along with the chain rule for ordered derivatives, let us connect gradient components to paths and edges, which is useful for a) gaining insights into various architectures and b) solidifying intuitions from backpropagation through time which suggest that short paths between t − τ and t facilitate gradient flow.

Here we provide an overview of the main idea; please see the appendix for a full derivation.

By applying the chain rule for ordered derivatives to expand DISPLAYFORM0 , we obtain a sum over τ terms.

However each term involves a partial derivative between h t and a prior hidden state, and thus all of these terms are 0 with the exception of those states that share an edge with h t .

Now, for each term, we can repeat this process.

This then yields non-zero terms only for hidden states which can be connected to h t through two edges.

We can then continue to apply the chain rule for ordered derivatives repeatedly, until only partial derivatives remain.

Upon completion, we have a sum over gradient components, with each component corresponding to exactly one path from t − τ to t and being a product over its path's edges.

The spectral norm corresponding to any particular path (t − τ → t → t → · · · → t) can then bounded as DISPLAYFORM1 where λ is the maximum spectral norm of any factor and n e is the number of edges on the path.

Terms with λ < 1 diminish exponentially fast, and when all λ < 1, shortest paths dominate 3 .

Viewing gradient components as paths, with each component being a product with one factor per edge along the path, gives us useful insight into various RNN architectures.

When relating a loss at time t to events at time t − τ , simple RNNs and LSTM contain shortest paths of length τ , while simple NARX RNNs contain shortest paths of length τ /n d , where n d is the number of delays.

One can envision many NARX RNN architectures with non-contiguous delays that reduce these shortest paths further.

In this section we introduce one such architecture using base-2 exponential delays.

In this case, for all τ ≤ 2 n d −1 , shortest paths exist with only log 2 τ edges; and for τ > 2 n d −1 , shortest paths exist with only τ /2 n d −1 edges (see FIG0 ).

Finally we must avoid the parameter and computation growth of simple NARX RNNs.

We achieve this by sharing weights over delays, instead using an attention-like mechanism BID2 over delays and a reset mechanism from gated recurrent units BID5 ).The proposed architecture, which we call mixed history RNNs (MIST RNNs), is described by DISPLAYFORM0 DISPLAYFORM1 Here, a t is a learned vector of n d convex-combination coefficients and r t is a reset gate.

At each time step, a convex combination of delayed states is formed according to a t ; units of this combination are reset according to r t ; and finally the typical linear layer and nonlinearity are applied.

Here we compare MIST RNNs to simple RNNs, LSTM, and Clockwork RNNs.

We begin with the sequential permuted MNIST task and the copy problem, synthetic tasks that were introduced to explicitly test RNNs for their ability to learn long-term dependencies BID20 BID28 BID24 BID0 BID18 BID7 .

Next we move on to 3 tasks for which it is plausible that very long-term dependencies play a role: recognizing surgical maneuvers from robot kinematics, recognizing phonemes from speech, and classifying activities from smartphone motion data.

We note that for all architectures involved, many variations can be applied (variational dropout, layer normalization, zoneout, etc.).

We keep experiments manageable by comparing architectures without such variations.

The sequential MNIST task BID24 consists of classifying 28x28 MNIST images BID25 as one of 10 digits, by scanning pixel by pixel -left to right, top to bottom -and emitting FORMULA0 is a challenging variant where a random permutation of pixels is chosen and applied to all images before classification.

LSTM with 100 hidden units is used as a baseline, with hidden unit counts for other architectures chosen to match the number of parameters.

Means and standard deviations are computed using the top 5 randomized trials out of 50 (ranked according to performance on the validation set), with random learning rates and initializations.

Additional experimental details can be found in the appendix.

Test error rates are shown in TAB0 .

Here, MIST RNNs outperform simple RNNs, LSTM, and Clockwork RNNs by a large margin.

We remark that our LSTM error rates are consistent with best previously-reported values, such as the error rates of 9.8% in BID6 and 12% in BID0 , which also use 100 hidden units.

One may also wonder if the difference in performance is due to hidden-unit counts.

To test this we also increased the LSTM hidden unit count to 139 (to match MIST RNNs), and continued to increase the capacity of each model further.

MIST RNNs significantly outperform LSTM in all cases.

We also used this task to visualize gradient magnitudes as a function of τ (the distance from the loss which occurs at time t = 784).

Gradient norms for all methods were averaged over a batch of 100 random examples early in training; see FIG1 .

Here we can see that simple RNNs and LSTM capture essentially no learning signal from steps that are far from the loss.

To validate this claim further, we repeated the 512-unit LSTM and MIST RNN experiments, but using only the last 200 permuted pixels (rather than all 784).

LSTM performance remains the same (7.4% error, within 1 standard deviation) whereas MIST RNN performance drops by 15 standard deviations (6.0% error).

The copy problem is a synthetic task that explicitly challenges a network to store and reproduce information from the past.

Our setup follows BID0 , which is in turn based on BID20 ).

An input sequence begins with L relevant symbols to be copied, is followed by a delay of D − 1 special blank symbols and 1 special go symbol, and ends with L additional blank symbols.

The corresponding target sequence begins with L + D blank symbols and ends with a copy of the relevant symbols from the inputs (in the same order).

We run experiments with copy delays of D = 50, 100, 200, and 400.

LSTM with 100 hidden units is used as a baseline, with hidden unit counts for other architectures chosen to match the number of parameters.

Additional experimental details can be found in the appendix.

Results are shown in FIG2 , showing validation curves of the top 5 randomized trials out of 50, with random learning rates and initializations.

With a short copy delay of D = 50, we can see that all methods other than Clockwork RNNs can solve the task in a reasonable amount of time.

However, as the copy delay D is increased, we can see that simple RNNs and LSTM become unable to learn a solution, whereas MIST RNNs are relatively unaffected.

We also note that our LSTM results are consistent with those in BID0 BID18 .Note that Clockwork RNNs are expected to fail for large delays (for example, the second symbol can only be seen by the highest-frequency partition, so learning to copy this symbol will fail for precisely the same reason that simple RNNs fail).

However, here they also fail for short delays, which is surprising because the high-speed partition resembles a simple RNN.

We hypothesized that this failure is due to hidden unit counts / parameter counts: here, the high-frequency partition is allocated only 256 / 8 = 32 hidden units.

To test this hypothesis, we reran the Clockwork RNN experiments with 1024 hidden units, so that 128 are allocated to the high-frequency partition.

Indeed, under this configuration (with 10x as many parameters), Clockwork RNNs do solve the task for a delay of D = 50 and fail to solve the task for all higher delays, thus behaving like simple RNNs.

Here we consider the task of online surgical maneuver recognition using the MISTIC-SL dataset BID12 BID8 .

Maneuvers are fairly long, high-level activities; examples include suture throw and knot tying.

The dataset was collected using a da Vinci, and the goal is to map robot kinematics over time (e.g., x, y, z) to gestures over time (which are densely labeled as 1 of 4 maneuvers on a per-frame basis).

We follow BID8 , which achieves state-ofthe-art performance on this task, as closely as possible, using the same kinematic inputs, test setup, and hyperparameters; details can be found in the original work or in the appendix.

The primary difference is that we replace their LSTM layer with our layers.

Results are shown in TAB1 .

Here MIST RNNs match LSTM performance (with half the number of parameters).

Here we consider the task of online framewise phoneme recognition using the TIMIT corpus BID13 .

Each frame is originally labeled as 1 of 61 phonemes.

We follow common practice and collapse these into a smaller set of 39 phonemes BID26 , and we include glottal stops to yield 40 classes in total.

We follow BID16 for data preprocessing and (Halberstadt, 1998) for training, validation, and test splits.

LSTM with 100 hidden units is used as a baseline, with hidden unit counts for other architectures chosen to match the number of parameters.

Means and standard deviations are computed using the top 5 randomized trials out of 50 (ranked according to performance on the validation set), with random learning rates and initializations.

Other experimental details can be found in the appendix.

TAB2 shows that LSTM and MIST RNNs perform nearly identically, which both outperform simple RNNs and Clockwork RNNs.

Here we consider the task of sequence classification from smartphones using the MobiAct (v2.0) dataset BID4 .

The goal is to classify each sequence as jogging, running, sitting down, etc., using smartphone motion data over time.

Approximately 3,200 sequences were collected from 67 different subjects.

We use the first 47 subjects for training, the next 10 for validation, and the final 10 for testing.

Means and standard deviations are computed using the top 5 randomized trials out of 50 (ranked according to performance on the validation set), with random learning rates and initializations.

Other experimental details can be found in the appendix.

Results are shown in TAB3 .

Here, MIST RNNs outperform all other methods, including LSTM and LSTM + , a variant with the same number of hidden units and twice as many parameters.

In this work we analyzed NARX RNNs and introduced a variant which we call MIST RNNs, which 1) exhibit superior vanishing-gradient properties in comparison to LSTM and previously-proposed NARX RNNs; 2) improve performance substantially over LSTM on tasks requiring very long-term dependencies; and 3) require even fewer parameters and computation than LSTM.

One obvious direction for future work is the exploration of other NARX RNN architectures with non-contiguous delays.

In addition, many recent techniques that have focused on LSTM are immediately transferable to NARX RNNs, such as variational dropout BID11 , layer normalization BID1 , and zoneout BID23 , and it will be interesting to see if such enhancements can improve MIST RNN performance further.

Removed for anonymity.

∂ht−τ are 0 except for the one satisfying t = t − τ + 1.

This yields DISPLAYFORM0 Now, by applying Equation 12 again to DISPLAYFORM1 , and then to DISPLAYFORM2 , and so on, we trace out a path from t − τ to t, as shown in FIG0 , finally resulting the single term DISPLAYFORM3 which is associated with the only path from t − τ to t, with one factor for each edge that is encountered along the path.

Next we consider simple NARX RNNs, again by expanding Equation 15.

From Equation 10, we can see that up to n d partials are now nonzero, and that any particular partial ∂h t ∂ht−τ is nonzero if and only if t > t − τ and t and t − τ share an edge.

Collecting these t as the set V t−τ = {t : t > t − τ and (t − τ, t ) ∈ E}, we can write DISPLAYFORM0 We can then apply this exact same process to each DISPLAYFORM1 ; by defining V t = {t : t > t and (t , t ) ∈ E} for all t , we can write DISPLAYFORM2 By continuing this process until only partials remain, we obtain a summation over all possible paths from t − τ to t. Each term in the sum is a product over factors, one per edge: DISPLAYFORM3 The analysis is nearly identical for general NARX RNNs, with the only difference being the specific sets of edges that are considered.8 APPENDIX: EXPERIMENTAL DETAILS 8.1 GENERAL EXPERIMENTAL SETUP Everything in this section holds for all experiments except surgical maneuver recognition, as in that case we mimicked BID8 as closely as possible, as described above.

All weight matrices are initialized using a normal distribution with a mean of 0 and a standard deviation of 1/ √ n h , where n h is the number of hidden units.

All initial hidden states (for t < 1) are initialized to 0.

For optimization, gradients are computed using full backpropagation through time, and we use stochastic gradient descent with a momentum of 0.9, with gradient clipping as described by BID31 at 1, and with a minibatch size of 100.

Biases are generally initialized to 0, but we follow best practice for LSTM by initializing the forget-gate bias to 1 Gers et al. FORMULA1 ; BID21 .

For Clockwork RNNs, 8 exponential periods are used, as in the original paper.

For MIST RNNs, 8 delays are used.

We avoid manual learning-rate tuning in its entirety.

Instead we run 50 trials for each experimental configuration.

In each trial, the learning rate is drawn uniformly at random in log space between 10 −4 and 10 1 , and initial weight matrices are also redrawn at random.

We report results over the top 10% of trials according to validationset error.

(An alternative option is to report results over all trials.

However, because the majority of trials yields bad performance for all methods, this simply blurs comparisons.

See for example FIG2 of BID16 , which compares these two options.)

Data preprocessing is kept minimal, with each input image individually shifted and scaled to have mean 0 and variance 1.

We split the official training set into two parts, the first 58,000 used for training and the last 2,000 used for validation.

Our test set is the same as the official test set, consisting of 10,000 images.

Training is carried out by minimizing cross-entropy loss.

In our experiments, the L relevant symbols are drawn at random (with replacement) from the set {0, 1, . . . , 9}; D is always a multiple of 10; and L is chosen to be D/10.

This way the simplest baseline of always predicting the blank symbol yields a constant error rate for all experiments.

No input preprocessing of any kind is performed.

In each case, we generate 100,000 examples for training and 1,000 examples for validation.

Training is carried out by minimizing cross-entropy loss.

We use the same experimental setup as BID8 , which currently holds state-of-the-art performance on these tasks.

For kinematic inputs we use positions, velocities, and gripper angles for both hands.

We also use their leave-one-user-out teset setup, with 8 users in the case of JIGSAWS and 15 users in the case of MISTIC-SL.

Finally we use the same hyperparameters: 1 hidden layer of 1024 units; dropout with p = 0.5; 80 epochs of training with a learning rate of 1.0 for the first 40 epochs and having the learning rate every 5 epochs for the rest of training.

As mentioned in the main paper, the primary difference is that we replaced their LSTM layer with our simple RNN, LSTM, or MIST RNN layer.

Training is carried out by minimizing cross-entropy loss.

We follow BID16 and extract 12 mel frequency cepstral coefficients plus energy every 10ms using 25ms Hamming windows and a pre-emphasis coefficient of 0.97.

However we do not use derivatives, resulting in 13 inputs per frame.

Each input sequence is individually shifted and scaled to have mean 0 and variance 1 over each dimension.

We form our splits according to BID17 , resulting in 3696 sequences for training, 400 sequences for validation, and 192 sequences for testing.

Training is carried out by minimizing cross-entropy loss.

Means and standard deviations are computed using the top 5 randomized trials out of 50 (ranked according to performance on the validation set).

In BID4 , emphasis was placed on hand-crafted features, and each subject was included during both training and testing (with no official test set defined).

We instead operate on

<|TLDR|>

@highlight

We introduce MIST RNNs, which a) exhibit superior vanishing-gradient properties in comparison to LSTM; b) improve performance substantially over LSTM and Clockwork RNNs on tasks requiring very long-term dependencies; and c) are much more efficient than previously-proposed NARX RNNs, with even fewer parameters and operations than LSTM.