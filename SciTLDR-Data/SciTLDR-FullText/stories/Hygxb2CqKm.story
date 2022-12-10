Stability is a fundamental property of dynamical systems, yet to this date it has had little bearing on the practice of recurrent neural networks.

In this work, we conduct a thorough investigation of stable recurrent models.

Theoretically, we prove stable recurrent neural networks are well approximated by feed-forward networks for the purpose of both inference and training by gradient descent.

Empirically, we demonstrate stable recurrent models often perform as well as their unstable counterparts on benchmark sequence tasks.

Taken together, these findings shed light on the effective power of recurrent networks and suggest much of sequence learning happens, or can be made to happen, in the stable regime.

Moreover, our results help to explain why in many cases practitioners succeed in replacing recurrent models by feed-forward models.

Recurrent neural networks are a popular modeling choice for solving sequence learning problems arising in domains such as speech recognition and natural language processing.

At the outset, recurrent neural networks are non-linear dynamical systems commonly trained to fit sequence data via some variant of gradient descent.

Stability is of fundamental importance in the study of dynamical system.

Surprisingly, however, stability has had little impact on the practice of recurrent neural networks.

Recurrent models trained in practice do not satisfy stability in an obvious manner, suggesting that perhaps training happens in a chaotic regime.

The difficulty of training recurrent models has compelled practitioners to successfully replace recurrent models with non-recurrent, feed-forward architectures.

This state of affairs raises important unresolved questions.

Is sequence modeling in practice inherently unstable?

When and why are recurrent models really needed?In this work, we shed light on both of these questions through a theoretical and empirical investigation of stability in recurrent models.

We first prove stable recurrent models can be approximated by feed-forward networks.

In particular, not only are the models equivalent for inference, they are also equivalent for training via gradient descent.

While it is easy to contrive non-linear recurrent models that on some input sequence cannot be approximated by feed-forward models, our result implies such models are inevitably unstable.

This means in particular they must have exploding gradients, which is in general an impediment to learnibility via gradient descent.

Second, across a variety of different sequence tasks, we show how recurrent models can often be made stable without loss in performance.

We also show models that are nominally unstable often operate in the stable regime on the data distribution.

Combined with our first result, these observation helps to explain why an increasingly large body of empirical research succeeds in replacing recurrent models with feed-forward models in important applications, including translation BID25 BID5 , speech synthesis BID24 , and language modeling .

While stability does not always hold in practice to begin with, it is often possible to generate a high-performing stable model by imposing stability during training.

Our results also shed light on the effective representational properties of recurrent networks trained in practice.

In particular, stable models cannot have long-term memory.

Therefore, when stable and unstable models achieve similar results, either the task does not require long-term memory, or the unstable model does not have it.

In this work, we make the following contributions.1.

We present a generic definition of stable recurrent models in terms of non-linear dynamical systems and show how to ensure stability of several commonly used models.

Previous work establishes stability for vanilla recurrent neural networks.

We give new sufficient conditions for stability of long short-term memory (LSTM) networks.

These sufficient conditions come with an efficient projection operator that can be used at training time to enforce stability.

2.

We prove, under the stability assumption, feed-forward networks can approximate recurrent networks for purposes of both inference and training by gradient descent.

While simple in the case of inference, the training result relies on non-trivial stability properties of gradient descent.

3.

We conduct extensive experimentation on a variety of sequence benchmarks, show stable models often have comparable performance with their unstable counterparts, and discuss when, if ever, there is an intrinsic performance price to using stable models.

In this section, we define stable recurrent models and illustrate the concept for various popular model classes.

From a pragmatic perspective, stability roughly corresponds to the criterion that the gradients of the training objective do not explode over time.

Common recurrent models can operate in both the stable and unstable regimes, depending on their parameters.

To study stable variants of common architectures, we give sufficient conditions to ensure stability and describe how to efficiently enforce these conditions during training.

A recurrent model is a non-linear dynamical system given by a differentiable state-transition map φ w : R n × R d → R n , parameterized by w ∈ R m .

The hidden state h t ∈ R n evolves in discrete time steps according to the update rule DISPLAYFORM0 where the vector x t ∈ R d is an arbitrary input provided to the system at time t. This general formulation allows us to unify many examples of interest.

For instance, for a recurrent neural network, given weight matrices W and U , the state evolves according to DISPLAYFORM1 Recurrent models are typically trained using some variant of gradient descent.

One natural-even if not strictly necessary-requirement for gradient descent to work is that the gradients of the training objective do not explode over time.

Stable recurrent models are precisely the class of models where the gradients cannot explode.

They thus constitute a natural class of models where gradient descent can be expected to work.

In general, we define a stable recurrent model as follows.

Definition 1.

A recurrent model φ w is stable if there exists some λ < 1 such that, for any weights w ∈ R m , states h, h ∈ R n , and input DISPLAYFORM2 Equivalently, a recurrent model is stable if the map φ w is λ-contractive in h. If φ w is λ-stable, then ∇ h φ w (h, x) < λ, and for Lipschitz loss p, ∇ w p is always bounded BID20 .Stable models are particularly well-behaved and well-justified from a theoretical perspective.

For instance, at present, only stable linear dynamical systems are known to be learnable via gradient descent BID7 .

In unstable models, the gradients of the objective can explode, and it is a delicate matter to even show that gradient descent converges to a stationary point.

The following proposition offers one such example.

The proof is provided in the appendix.

Proposition 1.

There exists an unstable system φ w where gradient descent does not converge to a stationary point, and ∇ w p → ∞ as the number of iterations N → ∞.

In this section, we provide sufficient conditions to ensure stability for several common recurrent models.

These conditions offer a way to require learning happens in the stable regime-after each iteration of gradient descent, one imposes the corresponding stability condition via projection.

Linear dynamical systems and recurrent neural networks.

Given a Lipschitz, point-wise nonlinearity ρ and matrices W ∈ R n×n and U ∈ R n×d , the state-transition map for a recurrent neural network (RNN) is DISPLAYFORM0 If ρ is the identity, then the system is a linear dynamical system.

BID9 show if ρ is L ρ -Lipschitz, then the model is stable provided W < 1 Lρ .

Indeed, for any states h, h , and any x, DISPLAYFORM1 In the case of a linear dynamical system, the model is stable provided W < 1.

Similarly, for the 1-Lipschitz tanh-nonlinearity, stability obtains provided W < 1.

In the appendix, we verify the assumptions required by the theorems given in the next section for this example.

Imposing this condition during training corresponds to projecting onto the spectral norm ball.

Long short-term memory networks.

Long Short-Term Memory (LSTM) networks are another commonly used class of sequence models BID8 .

The state is a pair of vectors s = (c, h) ∈ R 2d , and the model is parameterized by eight matrices, W ∈ R d×d and U ∈ R d×n , for ∈ {i, f, o, z}. The state-transition map φ LSTM is given by DISPLAYFORM2 where • denotes elementwise multiplication, and σ is the logistic function.

We provide conditions under which the iterated system φ DISPLAYFORM3 If the weights W f , U f and inputs x t are bounded, then f ∞ < 1 since |σ| < 1 for any finite input.

This means the next state c t must "forget" a non-trivial portion of c t−1 .

We leverage this phenomenon to give sufficient conditions for φ LSTM to be contractive in the ∞ norm, which in turn implies the iterated system φ r LSTM is contractive in the 2 norm for r = O(log(d)).

Let W ∞ denote the induced ∞ matrix norm, which corresponds to the maximum absolute row sum max i j |W ij |.

DISPLAYFORM4 2 , and r = O(log(d)), then the iterated system φ r LSTM is stable.

The proof is given in the appendix.

The conditions given in Proposition 2 are fairly restrictive.

Somewhat surprisingly we show in the experiments models satisfying these stability conditions still achieve good performance on a number of tasks.

We leave it as an open problem to find different parameter regimes where the system is stable, as well as resolve whether the original system φ LSTM is stable.

Imposing these conditions during training and corresponds to simple row-wise normalization of the weight matrices and inputs.

More details are provided in Section 4 and the appendix.

In this section, we prove stable recurrent models can be well-approximated by feed-forward networks for the purposes of both inference and training by gradient descent.

From a memory perspective, stable recurrent models are equivalent to feed-forward networks-both models use the same amount of context to make predictions.

This equivalence has important consequences for sequence modeling in practice.

When a stable recurrent model achieves satisfactory performance on some task, a feed-forward network can achieve similar performance.

Consequently, if sequence learning in practice is inherently stable, then recurrent models may not be necessary.

Conversely, if feed-forward models cannot match the performance of recurrent models, then sequence learning in practice is in the unstable regime.

For our purposes, the salient distinction between a recurrent and feed-forward model is the latter has finite-context.

Therefore, we say a model is feed-forward if the prediction made by the model at step t is a function only of the inputs x t−k , . . .

, x t for some finite k.

While there are many choices for a feed-forward approximation, we consider the simplest onetruncation of the system to some finite context k. In other words, the feed-forward approximation moves over the input sequence with a sliding window of length k producing an output every time the sliding window advances by one step.

Formally, for context length k chosen in advance, we define the truncated model via the update rule DISPLAYFORM0 Note that h k t is a function only of the previous k inputs x t−k , . . .

, x t .

While this definition is perhaps an abuse of the term "feed-forward", the truncated model can be implemented as a standard autoregressive, depth-k feed-forward network, albeit with significant weight sharing.

Let f denote a prediction function that maps a state h t to outputs f (h t ) = y t .

Let y k t denote the predictions from the truncated model.

To simplify the presentation, the prediction function f is not parameterized.

This is without loss of generality because it is always possible to fold the parameters into the system φ w itself.

In the sequel, we study y t − y k t both during and after training.

Suppose we train a full recurrent model φ w and obtain a prediction y t .

For an appropriate choice of context k, the truncated model makes essentially the same prediction y k t as the full recurrent model.

To show this result, we first control the difference between the hidden states of both models.

Lemma 1.

Assume φ w is λ-contractive in h and L x -Lipschitz in x. Assume the input sequence x t ≤ B x for all t. If the truncation length k ≥ log 1/λ DISPLAYFORM0 Lemma 1 effectively says stable models do not have long-term memory-distant inputs do not change the states of the system.

A proof is given in the appendix.

If the prediction function is Lipschitz, Lemma 1 immediately implies the recurrent and truncated model make nearly identical predictions.

Proposition 3.

If φ w is a L x -Lipschitz and λ-contractive map, and f is L f Lipschitz, and the DISPLAYFORM1

Equipped with our inference result, we turn towards optimization.

We show gradient descent for stable recurrent models finds essentially the same solutions as gradient descent for truncated models.

Consequently, both the recurrent and truncated models found by gradient descent make essentially the same predictions.

Our proof technique is to initialize both the recurrent and truncated models at the same point and track the divergence in weights throughout the course of gradient descent.

Roughly, we show if k ≈ O(log(N/ε)), then after N steps of gradient descent, the difference in the weights between the recurrent and truncated models is at most ε.

Even if the gradients are similar for both models at the same point, it is a priori possible that slight differences in the gradients accumulate over time and lead to divergent weights where no meaningful comparison is possible.

Building on similar techniques as BID6 , we show that gradient descent itself is stable, and this type of divergence cannot occur.

Our gradient descent result requires two essential lemmas.

The first bounds the difference in gradient between the full and the truncated model.

The second establishes the gradient map of both the full and truncated models is Lipschitz.

We defer proofs of both lemmas to the appendix.

Let p T denote the loss function evaluated on recurrent model after T time steps, and define p k T similarly for the truncated model.

Assume there some compact, convex domain Θ ⊂ R m so that the map φ w is stable for all choices of parameters w ∈ Θ. Lemma 2.

Assume p (and therefore p k ) is Lipschitz and smooth.

Assume φ w is smooth, λ-contractive, and Lipschitz in x and w. Assume the inputs satisfy x t ≤ B x , then DISPLAYFORM0 , suppressing dependence on the Lipschitz and smoothness parameters.

Lemma 3.

For any w, w ∈ Θ, suppose φ w is smooth, λ-contractive, and Lipschitz in w. If p is Lipschitz and smooth, then trunc is small.

Lemma 3 then guarantees that this small difference in weights does not lead to large differences in the gradient on the subsequent time step.

For an appropriate choice of learning rate, formalizing this argument leads to the following proposition.

Proposition 4.

Under the assumptions of Lemmas 2 and 3, for compact, convex Θ, after N steps of projected gradient descent with step size α t = α/t, w DISPLAYFORM1 DISPLAYFORM2 The decaying step size in our theorem is consistent with the regime in which gradient descent is known to be stable for non-convex training objectives BID6 .

While the decay is faster than many learning rates encountered in practice, classical results nonetheless show that with this learning rate gradient descent still converges to a stationary point; see p. 119 in BID3 and references there.

In the appendix, we give empirical evidence the O(1/t) rate is necessary for our theorem and show examples of stable systems trained with constant or O(1/ √ t) rates that do not satisfy our bound.

Critically, the bound in Proposition 4 goes to 0 as k → ∞. In particular, if we take α = 1 and k ≥ Ω(log(γN β /ε)), then after N steps of projected gradient descent, w DISPLAYFORM3 For this choice of k, we obtain the main theorem.

The proof is left to the appendix.

Theorem 1.

Let p be Lipschitz and smooth.

Assume φ w is smooth, λ-contractive, Lipschitz in x and w. Assume the inputs are bounded, and the prediction function f is L f -Lipschitz.

If k ≥ Ω(log(γN β /ε)), then after N steps of projected gradient descent with step size α t = 1/t, DISPLAYFORM4

In the experiments, we show stable recurrent models can achieve solid performance on several benchmark sequence tasks.

Namely, we show unstable recurrent models can often be made stable without a loss in performance.

In some cases, there is a small gap between the performance between unstable and stable models.

We analyze whether this gap is indicative of a "price of stability" and show the unstable models involved are stable in a data-dependent sense.

We consider four benchmark sequence problems-word-level language modeling, character-level language modeling, polyphonic music modeling, and slot-filling.

Language modeling.

In language modeling, given a sequence of words or characters, the model must predict the next word or character.

For character-level language modeling, we train and evaluate models on Penn Treebank BID14 .

To increase the coverage of our experiments, we train and evaluate the word-level language models on the Wikitext-2 dataset, which is twice as large as Penn Treebank and features a larger vocabulary BID15 .

Performance is reported using bits-per-character for character-level models and perplexity for word-level models.

Polyphonic music modeling.

In polyphonic music modeling, a piece is represented as a sequence of 88-bit binary codes corresponding to the 88 keys on a piano, with a 1 indicating a key that is pressed at a given time.

Given a sequence of codes, the task is to predict the next code.

We evaluate our models on JSB Chorales, a polyphonic music dataset consisting of 382 harmonized chorales by J.S. Bach BID0 .

Performance is measured using negative log-likelihood.

Slot-filling.

In slot filling, the model takes as input a query like "I want to Boston on Monday" and outputs a class label for each word in the input, e.g. Boston maps to Departure City and Monday maps to Departure Time.

We use the Airline Travel Information Systems (ATIS) benchmark and report the F1 score for each model (Price, 1990).

For each task, we first train an unconstrained RNN and an unconstrained LSTM.

All the hyperparameters are chosen via grid-search to maximize the performance of the unconstrained model.

For consistency with our theoretical results in Section 3 and stability conditions in Section 2.2, both models have a single recurrent layer and are trained using plain SGD.

In each case, the resulting model is unstable.

However, we then retrain the best models using projected gradient descent to enforce stability without retuning the hyperparameters.

In the RNN case, we constrain W < 1.

After each gradient update, we project the W onto the spectral norm ball by computing the SVD and thresholding the singular values to lie in [0, 1).

In the LSTM case, after each gradient update, we normalize each row of the weight matrices to satisfy the sufficient conditions for stability given in Section 2.2.

Further details are given in the appendix.

Stable and unstable models achieve similar performance.

Across all the tasks we considered, stable and unstable RNNs have roughly the same performance.

Stable RNNs and LSTMs achieve results comparable to published baselines on slot-filling BID17 and polyphonic music modeling BID2 ).

On word and character level language modeling, both stable and unstable RNNs achieve comparable results to BID2 .On the language modeling tasks, however, there is a gap between stable and unstable LSTM models.

Given the restrictive conditions we place on the LSTM to ensure stability, it is surprising they work as well as they do.

Weaker conditions ensuring stability of the LSTM could reduce this gap.

It is also possible imposing stability comes at a cost in representational capacity required for some tasks.

The gap between stable and unstable LSTMs on language modeling raises the question of whether there is an intrinsic performance cost for using stable models on some tasks.

If we measure stability in a data-dependent fashion, then the unstable LSTM language models are stable, indicating this gap is illusory.

However, in some cases with short sequences, instability can offer modeling benefits.

LSTM language models are stable in a "data-dependent" way.

Our notion of stability is conservative and requires stability to hold for every input and pair of hidden states.

If we instead consider a weaker, data-dependent notion of stability, the word and character-level LSTM models are stable (in the iterated sense of Proposition 2).

In particular, we compute the stability parameter only using input sequences from the data.

Furthermore, we only evaluate stability on hidden states reachable via gradient descent.

More precisely, to estimate λ, we run gradient ascent to find worst-case hidden states h, h to maximize DISPLAYFORM0 .

More details are provided in the appendix.

The data-dependent definition given above is a useful diagnostic-when the sufficient stability conditions fail to hold, the data-dependent condition addresses whether the model is still operating in the stable regime.

Moreover, when the input representation is fixed during training, our theoretical results go through without modification when using the data-dependent definition.

Using the data-dependent measure, in FIG3 (a), we show the iterated character-level LSTM, φ r LSTM , is stable for r ≈ 80 iterations.

A similar result holds for the word-level language model for r ≈ 100.

These findings are consistent with experiments in BID13 which find LSTM trajectories converge after approximately 70 steps only when evaluated on sequences from the data.

For language models, the "price of stability" is therefore much smaller than the gap in TAB1 suggests-even the "unstable" models are operating in the stable regime on the data distribution.

Unstable systems can offer performance improvements for short-time horizons.

When sequences are short, training unstable models is less difficult because exploding gradients are less of an issue.

In these case, unstable models can offer performance gains.

To demonstrate this, we train truncated unstable models on the polyphonic music task for various values of the truncation parameter k. In FIG3 (b), we simultaneously plot the performance of the unstable model and the stability parameter λ for the converged model for each k. For short-sequences, the final model is more unstable (λ ≈ 3.5) and achieves a better test-likelihood.

For longer sequence lengths, λ decreases closer to the stable regime (λ ≈ 1.5), and this improved test-likelihood performance disappears.

What is the intrinsic "price of stability"?

For language modeling, we show the unstable LSTMs are actually stable in weaker, data-dependent sense.

On the other hand, for polyphonic music modeling with short sequences, instability can improve model performance.

In the previous section, we showed nominally unstable models often satisfy a data-dependent notion of stability.

In this section, we offer further evidence unstable models are operating in the stable regime.

These results further help explain why stable and unstable models perform comparably in experiments.

Vanishing gradients.

Stable models necessarily have vanishing gradients, and indeed this ingredient is a key ingredient in the proof of our training-time approximation result.

For both word and character-level language models, we find both unstable RNNs and LSTMs also exhibit vanishing gradients.

In Figures 3(a) and 3(b) , we plot the average gradient of the loss at time t + i with respect to the input at time t, ∇ xt p t+i as t ranges over the training set.

For either language modeling task, the LSTM and the RNN suffer from limited sensitivity to distant inputs at initialization and throughout training.

The gradients of the LSTM vanish more slowly than those of the RNN, but both models exhibit the same qualitative behavior.

Figure 4 : Effect of truncating unstable models.

On both language and music modeling, RNNs and LSTMs exhibit diminishing returns for large values of the truncation parameter k. In LSTMs, larger k doesn't affect performance, whereas for unstable RNNs, large k slightly decreases performance Proposition (4) holds for unstable models.

In stable models, Proposition (4) in Section 3 ensures the distance between the weight matrices w recurr − w trunc grows slowly as training progresses, and this rate decreases as k becomes large.

In Figures 5(a) and 5(b), we show a similar result holds empirically for unstable word-level language models.

All the models are initialized at the same point, and we track the distance between the hidden-to-hidden matrices W as training progresses.

Training the full recurrent model is impractical, and we assume k = 65 well captures the fullrecurrent model.

In Figures 5(a) and 5(b), we plot W k − W 65 for k ∈ {5, 10, 15, 25, 35, 50, 64} throughout training.

As suggested by Proposition (4), after an initial rapid increase in distance, W k − W 65 grows slowly, as suggested by Proposition 4.

Moreover, there is a diminishing return to choosing larger values of the truncation parameter k in terms of the accuracy of the approximation.

Figure 5: Qualitative version of Proposition 4 for unstable, word-level language models.

We assume k = 65 well-captures the full-recurrent model and plot w trunc − w recurr = W k − W 65 as training proceeds, where W denotes the recurrent weights.

As Proposition 4 suggests, this quantity grows slowly as training proceeds, and the rate of growth decreases as k increases.

Our experiments show recurrent models trained in practice operate in the stable regime, and our theoretical results show stable recurrent models are approximable by feed-forward networks, As a consequence, we conjecture recurrent networks trained in practice are always approximable by feed-forward networks.

Even with this conjecture, we cannot yet conclude recurrent models as commonly conceived are unnecessary.

First, our present proof techniques rely on truncated versions of recurrent models, and truncated recurrent architectures like LSTMs may provide useful inductive bias on some problems.

Moreover, implementing the truncated approximation as a feed-forward network increases the number of weights by a factor of k over the original recurrent model.

Declaring recurrent models truly superfluous would require both finding more parsimonious feed-forward approximations and proving natural feed-forward models, e.g. fully connected networks or CNNs, can approximate stable recurrent models during training.

This remains an important question for future work.

Learning dynamical systems with gradient descent has been a recent topic of interest in the machine learning community.

BID7 show gradient descent can efficiently learn a class of stable, linear dynamical systems, BID19 shows gradient descent learns a class of stable, non-linear dynamical systems.

Work by BID22 gives a moment-based approach for learning some classes of stable non-linear recurrent neural networks.

Our work explores the theoretical and empirical consequences of the stability assumption made in these works.

In particular, our empirical results show models trained in practice can be made closer to those currently being analyzed theoretically without large performance penalties.

For linear dynamical systems, BID23 exploit the connection between stability and truncation to learn a truncated approximation to the full stable system.

Their approximation result is the same as our inference result for linear dynamical systems, and we extend this result to the non-linear setting.

We also analyze the impact of truncation on training with gradient descent.

Our training time analysis builds on the stability analysis of gradient descent in BID6 , but interestingly uses it for an entirely different purpose.

Results of this kind are completely new to our knowledge.

For RNNs, the link between vanishing and exploding gradients and W was identified in BID20 .

For 1-layer RNNs, BID9 give sufficient conditions for stability in terms of the norm W and the Lipschitz constant of the non-linearity.

Our work additionally considers LSTMs and provides new sufficient conditions for stability.

Moreover, we study the consequences of stability in terms of feed-forward approximation.

A number of recent works have sought to avoid vanishing and exploding gradients by ensuring the system is an isometry, i.e. λ = 1.

In the RNN case, this amounts to constraining W = 1 BID1 BID27 BID10 BID18 BID11 .

BID26 observes strictly requiring W = 1 reduces performance on several tasks, and instead proposes maintaining W ∈ [1 − ε, 1 + ε].

BID28 maintains this "soft-isometry" constraint using a parameterization based on the SVD that obviates the need for the projection step used in our stable-RNN experiments.

BID12 sidestep these issues and stabilizes training using a residual parameterization of the model.

At present, these unitary models have not yet seen widespread use, and our work shows much of the sequence learning in practice, even with nominally unstable models, actually occurs in the stable regime.

From an empirical perspective, BID13 introduce a non-chaotic recurrent architecture and demonstrate it can perform as well more complex models like LSTMs.

BID2 conduct a detailed evaluation of recurrent and convolutional, feed-forward models on a variety of sequence modeling tasks.

In diverse settings, they find feed-forward models outperform their recurrent counterparts.

Their experiments are complimentary to ours; we find recurrent models can often be replaced with stable recurrent models, which we show are equivalent to feed-forward networks.

Proof of Proposition 1.

Consider a scalar linear dynamical system DISPLAYFORM0 where h 0 = 0, a, b ∈ R are parameters, and x t , y t ∈ R are elements the input-output sequence DISPLAYFORM1 , where L is the sequence length, andŷ t is the prediction at time t. Stability of the above system corresponds to |a| < 1.Suppose (x t , y t ) = (1, 1) for t = 1, . . .

, L. Then the desired system (4) simply computes the identity mapping.

Suppose we use the squared-loss (y t ,ŷ t ) = (1/2)(y t −ŷ t ) 2 , and suppose further b = 1, so the problem reduces to learning a = 0.

We first compute the gradient.

Compactly write DISPLAYFORM2 .Plugging in y t = 1, this becomes DISPLAYFORM3 For large T , if |a| > 1, then a L grows exponentially with T and the gradient is approximately DISPLAYFORM4 Therefore, if a 0 is initialized outside of [−1, 1], the iterates a i from gradient descent with step size α i = (1/i) diverge, i.e. a i → ∞, and from equation FORMULA20 , it is clear that such a i are not stationary points.

DISPLAYFORM5 2 , so since tanh(x) ∈ [−1, 1], tanh(x) is 1-Lipschitz and 2-smooth.

We previously showed the system is stable since, for any states h, h , DISPLAYFORM6 (1−λ) for all t.

Therefore, for any W, W , U, U , DISPLAYFORM7 so the model is Lipschitz in U, W .

We can similarly argue the model is B U Lipschitz in x. For smoothness, the partial derivative with respect to h is DISPLAYFORM8 so for any h, h , bounding the ∞ norm with the 2 norm, DISPLAYFORM9 For any W, W , U, U satisfying our assumptions, DISPLAYFORM10

is Lipschitz in h and w.

Similar to the previous sections, we assume s 0 = 0.The state-transition map is not Lipschitz in s, much less stable, unless c is bounded.

However, assuming the weights are bounded, we first prove this is always the case.

DISPLAYFORM0 Proof of Lemma 4.

Note |tanh(x)| , |σ(x)| ≤ 1 for all x. Therefore, for any t, h t ∞ = o t • tanh(c t ) ∞ ≤ 1.

Since σ(x) < 1 for x < ∞ and σ is monotonically increasing DISPLAYFORM1 Using the trivial bound, i t ∞ ≤ 1 and z t ∞ ≤ 1, so DISPLAYFORM2 Unrolling this recursion, we obtain a geometric series DISPLAYFORM3 Proof of Proposition 2.

We show φ LSTM is λ-contractive in the ∞ -norm for some λ < 1.

For r ≥ log 1/λ ( √ d), this in turn implies the iterated system φ r LSTM is contractive is the 2 -norm.

Consider the pair of reachable hidden states s = (c, h), s = (c , h ).

By Lemma 4, c, c are bounded.

Analogous to the recurrent network case above, since σ is (1/4)-Lipschitz and tanh is 1-Lipschitz, DISPLAYFORM4 Both z ∞ , i ∞ ≤ 1 since they're the output of a sigmoid.

Letting c + and c + denote the state on the next time step, applying the triangle inequality, DISPLAYFORM5 A similar argument shows DISPLAYFORM6 By assumption, DISPLAYFORM7 and so DISPLAYFORM8 as well as DISPLAYFORM9 which together imply DISPLAYFORM10 establishing φ LSTM is contractive in the ∞ norm.

Throughout this section, we assume the initial state h 0 = 0.

Without loss of generality, we also assume φ w (0, 0) = 0 for all w. Otherwise, we can reparameterize φ w (h, x) → φ w (h, x) − φ w (0, 0) without affecting expressivity of φ w .

For stable models, we also assume there some compact, convex domain Θ ⊂ R m so that the map φ w is stable for all choices of parameters w ∈ Θ.Proof of Lemma 1.

For any t ≥ 1, by triangle inequality, DISPLAYFORM0 Applying the stability and Lipschitz assumptions and then summing a geometric series, DISPLAYFORM1 Now, consider the difference between hidden states at time step t. Unrolling the iterates k steps and then using the previous display yields DISPLAYFORM2 and solving for k gives the result.

Before proceeding, we introduce notation for our smoothness assumption.

We assume the map φ w satisfies four smoothness conditions: for any reachable states h, h , and any weights w, w ∈ Θ, there are some scalars β ww , β wh , β hw , β hh such that 1.

DISPLAYFORM0

In the section, we argue the difference in gradient with respect to the weights between the recurrent and truncated models is O(kλ k ).

For sufficiently large k (independent of the sequence length), the impact of truncation is therefore negligible.

The proof leverages the "vanishing-gradient" phenomenon-the long-term components of the gradient of the full recurrent model quickly vanish.

The remaining challenge is to show the short-term components of the gradient are similar for the full and recurrent models.

Proof of Lemma 2.

The Jacobian of the loss with respect to the weights is DISPLAYFORM0 where ∂ht ∂w is the partial derivative of h t with respect to w, assuming h t−1 is constant with respect to w. Expanding the expression for the gradient, we wish to bound DISPLAYFORM1 The first term consists of the "long-term components" of the gradient for the recurrent model.

The second term is the difference in the "short-term components" of the gradients between the recurrent and truncated models.

We bound each of these terms separately.

For the first term, by the Lipschitz assumptions, DISPLAYFORM2 Using submultiplicavity of the spectral norm, DISPLAYFORM3 Focusing on the second term, by triangle inequality and smoothness, DISPLAYFORM4 .Using Lemma 1 to upper bound (a), DISPLAYFORM5 Using the triangle inequality, Lipschitz and smoothness, (b) is bounded by DISPLAYFORM6 In this section, we prove that the gradient map ∇ w p T is Lipschitz.

First, we show on the forward pass, the difference between hidden states h t (w) and h t (w ) obtained by running the model with weights w and w , respectively, is bounded in terms of w − w .

Using smoothness of φ, the difference in gradients can be written in terms of h t (w) − h t (w ) , which in turn can be bounded in terms of w − w .

We repeatedly leverage this fact to conclude the total difference in gradients must be similarly bounded.

We first show small differences in weights don't significantly change the trajectory of the recurrent model.

Lemma 5.

For some w, w , suppose φ w , φ w are λ-contractive and L w Lipschitz in w. Let h t (w), h t (w ) be the hidden state at time t obtain from running the model with weights w, w on common inputs {x t }.

If h 0 (w) = h 0 (w ), then DISPLAYFORM7 Proof.

By triangle inequality, followed by the Lipschitz and contractivity assumptions, DISPLAYFORM8 Iterating this argument and then using h 0 (w) = h 0 (w ), we obtain a geometric series in λ.

DISPLAYFORM9 The proof of Lemma 3 is similar in structure to Lemma 2, and follows from repeatedly using smoothness of φ and Lemma 5.Proof of Lemma 3.

Let h t = h t (w ).

Expanding the gradients and using DISPLAYFORM10 from Lemma 5.

DISPLAYFORM11 .

Focusing on term (a), DISPLAYFORM0 where the penultimate line used, DISPLAYFORM1 To bound (b), we peel off terms one by one using the triangle inequality, DISPLAYFORM2 Supressing Lipschitz and smoothness constants, we've shown the entire sum is O(1/(1 − λ) 3 ), as required.

Equipped with the smoothness and truncation lemmas (Lemmas 2 and 3), we turn towards proving the main gradient descent result.

Proof of Proposition 4.

Let Π Θ denote the Euclidean projection onto Θ, and let δ i = w i recurr − w i trunc .

Initially δ 0 = 0, and on step i + 1, we have the following recurrence rela-tion for δ i+1 , DISPLAYFORM0 the penultimate line applied lemmas 2 and 3, and the last line used 1 + x ≤ e x for all x. Unwinding the recurrence relation at step N , DISPLAYFORM1 Bounding the inner summation via an integral, N j=i+1 1 j ≤ log(N/i) and simplifying the resulting expression, DISPLAYFORM2 exp(αβ log(N/i)) αγkλ DISPLAYFORM3

Proof of Theorem 1.

Using f is L f -Lipschitz and the triangle inequality, DISPLAYFORM0 By Lemma 5, the first term is bounded by , and by Lemma 1, the second term is bounded by λ k LxBx(1−λ) .

Using Proposition 4, after N steps of gradient descent, we have DISPLAYFORM1 and solving for k such that both terms are less than ε/2 gives the result.

The O(1/t) rate may be necessary.

The key result underlying Theorem 1 is the bound on the parameter difference w trunc − w recurr while running gradient descent obtained in Proposition 4.

We show this bound has the correct qualitative scaling using random instances and training randomly initialized, stable linear dynamical systems and tanh-RNNs.

In FIG6 , we plot the parameter error w t trunc − w t recurr as training progresses for both models (averaged over 10 runs).

The error scales comparably with the bound given in Proposition 4.

We also find for larger step-sizes like α/ √ t or constant α, the bound fails to hold, suggesting the O(1/t) condition is necessary.

∼ N (0, I 32 ).

We fix the truncation length to k = 35, set the learning rate to α t = α/t for α = 0.01, and take N = 200 gradient steps.

These parameters are chosen so that the γkλ k N αβ+1 bound from Proposition 4 does not become vacuous -by triangle inequality, we always have w trunc − w recurr ≤ 2λ.

Stable vs. unstable models.

The word and character level language modeling experiments are based on publically available code from BID16 .

The polyphonic music modeling code is based on the code in BID2 , and the slot-filling model is a reimplementation of BID17 1 Since the sufficient conditions for stability derived in Section 2.2 only apply for networks with a single layer, we use a single layer RNN or LSTM for all experiments.

Further, our theoretical results are only applicable for vanilla SGD, and not adaptive gradient methods, so all models are trained with SGD.

Table 2 contains a summary of all the hyperparameters for each experiment.

All hyperparameters are shared between the stable and unstable variants of both models.

In the RNN case, enforcing stability is conceptually simple, though computationally expensive.

Since tanh is 1-Lipschitz, the RNN is stable as long as W < 1.

Therefore, after each gradient update, we project W onto the spectral norm ball by taking the SVD and thresholding the singular values to lie in [0, 1).

In the LSTM case, enforcing stability is conceptually more difficult, but computationally simple.

To ensure the LSTM is stable, we appeal to Proposition 2.

We enforce the following inequalities after each gradient update

@highlight

Stable recurrent models can be approximated by feed-forward networks and empirically perform as well as unstable models on benchmark tasks.

@highlight

Studies the stability of RNNs and investigation of spectral normalization to sequential predictions.