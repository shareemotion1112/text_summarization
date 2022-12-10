Given a large database of concepts but only one or a few examples of each, can we learn models for each concept that are not only generalisable, but interpretable?

In this work, we aim to tackle this problem through hierarchical Bayesian program induction.

We present a novel learning algorithm which can infer concepts as short, generative, stochastic programs, while learning a global prior over programs to improve generalisation and a recognition network for efficient inference.

Our algorithm, Wake-Sleep-Remember (WSR), combines gradient learning for continuous parameters with neurally-guided search over programs.

We show that WSR learns compelling latent programs in two tough symbolic domains: cellular automata and Gaussian process kernels.

We also collect and evaluate on a new dataset, Text-Concepts, for discovering structured patterns in natural text data.

A grand challenge for building more flexible AI is developing learning algorithms which quickly pick up a concept from just one or a few examples, yet still generalise well to new instances of that concept.

In order to instill algorithms with the correct inductive biases, research in few-shot learning usually falls on a continuum between model-driven and data-driven approaches.

Model-driven approaches place explicit domain-knowledge directly into the learner, often as a stochastic program describing how concepts and their instances are produced.

For example, we can model handwritten characters with a motor program that composes distinct pen strokes BID13 , or spoken words as sequences of phonemes which obey particular phonotactic constraints.

Such representationally explicit models are highly interpretable and natural to compose together into larger systems, although it may be difficult to completely pre-specify the required inductive biases.

By contrast, data-driven approaches start with only minimal assumptions about a domain, and instead acquire the inductive biases themselves from a large background dataset.

This is typified by recent work in deep meta-learning, such as the Neural Statistian BID5 ; see also BID9 ), MAML BID6 ; see also BID14 ) and Prototypical Networks BID15 .

Crucially, these models rely on stochastic gradient descent (SGD) for the meta-learning phase, as it is a highly scalable algorithm that applies easily to datasets with thousands of classes.

Ideally these approaches would not be exclusive -for many domains of AI we have access to large volumes of data and also rich domain knowledge, so we would like to utilise both.

In practice, however, different algorithms are suited to each end of the continuum: SGD requires objectives to be differentiable, but explicit domain knowledge often introduces discrete latent variables, or programs.

Thus, meta-learning from large datasets is often challenging in more explicit models.

In this work, we aim to bridge these two extremes: we learn concepts represented explicitly as stochastic programs, while meta-learning generative parameters and an inductive bias over programs from a large unlabelled dataset.

We introduce a simple learning algorithm, Wake-Sleep-Remember (WSR), which combines SGD over continuous parameters with neurally-guided search over latent programs to maximize a variational objective, the evidence lower bound (ELBo).In evaluating our algorithm, we also release a new dataset for few-shot concept learning in a highlystructured natural domain of short text patterns (see TAB0 ).

This dataset contains 1500 concepts such as phone numbers, dates, email addresses and serial numbers, crawled from public GitHub repositories.

Such concepts are easy for humans to learn using only a few examples, and are well described as short programs which compose discrete, interpretable parts.

Thus, we see this as an excellent challenge domain for structured meta-learning and explainable AI.

2 BACKGROUND: HELMHOLTZ MACHINES AND VARIATIONAL BAYES Suppose we wish to learn generative models of spoken words unsupervised, using a large set of audio recordings.

We may aim to include domain knowledge that words are built up from different short phonemes, without defining in advance exactly what the kinds of phoneme are, or exactly which phonemes occur in each recording.

This means that, in order to learn a good model of words in general, we must also infer the particular latent phoneme sequence that generated each recording.

This latent sequence must be re-estimated whenever the global model is updated, which itself can be a hard computational problem.

To avoid a costly learning 'inner-loop', a longstanding idea in machine learning is to train two distinct models simultaneously: a generative model which describes the joint distribution of latent phonemes and sounds, and a recognition model which allows phonemes to be inferred quickly from data.

These two models together are often called a Helmholtz Machine BID2 .Formally, algorithms for training a Helmholtz Machine are typically motivated by Variational Bayes.

Suppose we wish to learn a generative model p(z, x), which is a joint distribution over latent variables z and observations x, alongside a recognition model q(z; x), which is a distribution over latent variables conditional on observations.

It can be shown that the marginal likelihood of each observation is bounded below by DISPLAYFORM0 where D KL [q(z; x)||p(z|x)] is the KL divergence from the true posterior p(z|x) to the recognition model's approximate posterior q(z; x).

Learning a Helmholtz machine is then framed as maximisation of this evidence lower bound (or ELBo), which provides the shared basis for two historically distinct approaches to learning.

The first method, proposed by Hinton et al., is an alternating optimisation algorithm: alternate between updates to the generative model p and recognition model q. The update for p(x|z), called the 'wake' phase, can be derived simply from Eq. 2 as:Wake phase: Maximise E[log p(x|z)] of observed data x using inferred latent variables z ∼ q(z; x)Unfortunately, the exact update for q(z; x), which is minimisation of D KL [q(z; x)

More recently BID12 proposed the Variational Autoencoder (VAE).

This offers an alternative solution to the problem of training q without relying on the above KL-divergence approximation.

Instead, the authors note that it is possible to construct an unbiased approximation to the ELBo (Eq. 2) using only a single sample from q.

Under specific assumptions about the form of q -specifically, that it is a continuous distribution which can be reparametrised by transforming a fixed auxiliary distribution -they use this to construct a low variance estimate for the gradient of the ELBo.

As it is unbiased, this gradient estimate can used in SGD to train both q and p, typically neural networks, simultaneously towards the ELBo VAE Update: Sample z ∼ q(z; x) and take a gradient step on log DISPLAYFORM0 When z are discrete, VAEs cannot be trained through the use of reparameterisation but instead rely on the policy gradient (otherwise called REINFORCE, Williams (1992)) estimator from reinforcement learning.

This estimator is notoriously high variance, in many cases rendering SGD ineffectual.

This difficulty has motivated a wide literature on variance reduction techniques, BID8 BID11 BID16 BID7 ), yet training VAEs with discrete latent variables remains a challenging open research problem.

The above description highlights a bias-variance tension between these two approaches ( TAB2 ).

The wake-sleep algorithm applies well to a wide variety of models, including structured models with discrete latent variables, but relies on an approximate update for q which may be heavily biased.

By contrast, VAEs are proven to converge to a local optimum of the evidence lower bound (and so are often seen as more 'principled') but require much stronger assumptions on the form of the model in order for learning to be practical.

Additionally, both VAEs and Wake-sleep rely on the ability of the recognition model, q(z; x), to learn to carry out posterior inference accurately; any departure from this changes the optimal p.

This strong constraint is often unrealistic and unnecessary: on hard problems, a recognition model may still be useful if only one in a hundred samples are of high quality.

Recent work aims to address this in both VAEs BID1 and Wake-sleep BID0 ) by using importance weighting over many samples from q. This solution is well suited when fully amortised inference is just out of reach of the recognition model, but is bottlenecked by how many samples it is practical to evaluate per gradient step.

The next section describes our alternative approach, motivated by the idea that good explanations needn't be forgotten.

Simply put, we mitigate the difficulties of discrete inference by introducing a separate 'memory' into the Helmholtz Machine, explicitly keeping track of the best discovered latent explanations, or programs z i , for each observation x i .

Comparison of VAE and Wake-Sleep algorithms for training Helmholtz machines.

Wakesleep uses an approximation to the correct update for q, which may be heavily biased.

VAE updates are unbiased, but for discrete variables they are often too high variance for learning to succeed.

Figure 1: For VAEs and Wake-sleep, the recognition model q also serves as the variational distribution that trains p. WSR distinguishes these, learning a recognition model r and a categorical variational posterior q which is separate from r. This means that like VAEs, WSR jointly trains p and q using an unbiased estimate of the variational objective (blue).

Like wake-sleep, the recognition model can train self-supervised (green), allowing WSR to handle discrete latent variables.

To optimise the finite support of q, WSR incorporates a memory module M that remembers the best values of z i found by r(z i ; x i ) across iterations.

In this work we start from a different set of modelling assumptions to those typical of VAE-family models.

Rather than describe each observation with a latent vector z which lacks explicit structure, we assume each observation is generated by an explicit latent program, and wish to learn:1.

A posterior distribution over the latent program q i (z i ) for each instance x i 2.

A prior p(z) that captures a global inductive bias over programs.

4.

An approximate recognition network r(z; x) which helps infer programs for novel data.

Using programs as a latent representation makes this setup challenging for two reasons.

First, as seen in TAB2 , training discrete Helmholtz machines usually requires accepting either high bias or high variance in the learning objective.

Second, by assumption, inferring programs from data is a hard problem, so performing highly accurate amortised inference may be overly ambitious.

We therefore desire a learning algorithm for which weaker recognition models may reduce the speed of learning but will not change the set of stable solutions to the learning problem.

To achieve this, we depart from the usual Helmholtz machines formulation by separating the recognition model from the variational distribution ( Figure 1 ).

As in Wake-sleep, we train the recognition model r(z; x) self supervised using samples from the prior -an effective strategy when z is discrete.

Figure 2 : What trains what?

r is a recognition network trained self-supervised on samples from the p.

The memory M i for each task i is a set of the 'best' z values ever sampled by r, selected according their joint probability p(z, x i ).

p is then trained using samples from M .However, unlike Wake-sleep we do not use samples from r directly to train the generative model p.

Instead, they are routed through a separate Memory module, M , which maintains a set of the best values of z found for each x, across training iterations, weighted by their joint probability p(z, x).

Then, we simply resample from M in order to train p.

By weighting each z proportional to its join probability, we guarantee that every update to M decreases the KL divergence between M i and the true posterior p(z i |x i ).

Thus, we may view each M i as a truncated variational posterior over z i , which is optimised towards the ELBo using samples from r as proposals.

Our full training procedure is detailed in Algorithm 1.Algorithm 1: Basic WSR training procedure (batching omitted for notational simplicity).

In practice, we avoid evaluation of p θ in the each wake phase by maintainging a cache of p θ (z M , x) in the sleep phase.

We re-calculate each p θ (z, x) only as a final correctness check before modifying DISPLAYFORM0 1.

Update memory with sample from recognition network DISPLAYFORM1 2.

Train generative model with sample from memory DISPLAYFORM2 3.

Train recognition network with sample from generative model DISPLAYFORM3 4.

Train prior with sample from reference distribution g = ∇ θ s sleep + s wake (+s hyper ) θ = θ + λg Gradient step (e.g. SGD, Adam) until convergence

In the above problem setup, we assumed that the prior p(z) either was fixed, or was learned to maximise the ELBo training objective.

However, for many modelling problems neither is adequate: we often have some idea about the global distribution over of latent programs, but deciding on the exact p(z) in advance would be too strong a commitment.

In these cases we would rather provide a reference distribution p (z) as a first approximation to the global distribution, while but still allow the model to update its prior p(z) to move away from p as it learns from data.

In this situation we may place a hyperprior over p(z), defined with respect to the reference distribution as: DISPLAYFORM0 where α is a concentration parameter controlling the level of confidence in the reference distribution p .

This form of hyperprior can be integrated into the training objective simply by addition of an extra term: Ez∼p α log p(z), estimated by sampling from p .

Algorithm 1 includes this as an optional variant, which corresponds to maximum a posteriori estimation over p.

We first test our algorithm at learning the rules for noisy 1-dimensional cellular automata, from the images they generate.

We create 64 × 64 binary images generated row by row, sampling each pixel using a rule that depends only on its 'neighbours' in the row above.

Specifically, given a 'neighbourhood size' D and 'corruption probability' , we generate images by the following procedure:• Choose a binary vector z ∈ {0, 1} 2 D to represent the update rule for the cellular automaton.• Sample the first generation uniformly at random, as g 1 ∈ {0, 1}

•

For each subsequent row i = 2, . . .

, 64 and each cell (i, j):1. read the neighbouring D cells from the previous row: DISPLAYFORM0 2.

sample g ij according to: p(g ij = z sij ) = 1 − Figure 3 : One-shot generalisations produced by each algorithm on each the cellular automata datasets.

For each input image we sample a program z from the variational distribution q, then synthesize a new image in the same style from p (z|x) using the learned .We create easy, medium and hard datasets corresponding to increasingly complex cellular automaton rules, with neighbourhood sizes of D = 3, 4, 5 respectively (easy corresponds to the full set of 256 elementary automata studied by BID18 .

For medium and hard, we sample 10,000 of the 65,000 and 4 billion available rules).

All datasets share a noise parameter = 0.01.Our goal is discover a latent rule, or program, z i corresponding to each image in the dataset, while also learning the global noise .

Thus, we learn a p (z, x) with same structure as the true generative process, and use a CNN with independent Bernoulli outputs as the recognition network r(z; x).

Fixing this architecture, we train WSR using k = 5 as the memory size, and compare performance of for the against three baseline algorithms:• VAE.

We use policy-gradient (REINFORCE) for discete choices, and additionally reduce variance by subtracting a learned baseline for each task.• Wake-Sleep.

We perform gradient descent on the recognition model q and generative model p together, using samples from the p to train q, and samples from q to train p.• No Recognition.

We evaluate a lesioned Algorithm 1 in which no recognition model is learned.

We instead propose updates to M i using samples from the prior z p ∼ p(z).Our results highlight clear differences between these approaches.

Despite our efforts at variance reduction, a VAE reliably struggles to get off the ground on any dataset, and instead learns quickly to model all instances as noise ( Figure 3 and 4 bottom).

Wake-sleep is able to learn accurate rules for images from the easiest datasets, but on the most challenging dataset but its performance appears to asymptote prematurely.

By contrast, WSR reliably learns accurate programs that can be used to classify unseen images 100-way with > 99% accuracy, even on the hard dataset.

Figure 4 : Quantitative results on all variants of the cellular automata dataset.

In all cases WSR learns programs which generalise to unseen images of the same concepts, achieving > 99% accuracy on a 100-way classification task (second row).

WSR also best recovers the true noise parameter = 0.01 (third row).

Note: x-axis is wallclock time on a single Titan-X GPU to allow a fair comparison, as WSR requires several times more computation per iteration.

Next, we evaluate our algorithm on the the task of finding explainable models for time-series data.

We draw inspiration from BID4 , who frame this problem as Gaussian process (GP) kernel learning.

They describe a grammar for building kernels compositionally, and demonstrate that inference in this grammar can produce highly interpretable and generalisable descriptions of the structure in a time series.

Inference is achieved on a single time-series through a custom greedy search algorithm, requiring a costly inner loop that approximately marginalises over kernel parameters.

Here, we follow a similar approach embedded within a larger goal: we use a dataset of many different time-series, and learn a hierarchical model over time-series.

That is, we learn a separate GP kernel for each series in the dataset while also learning an inductive bias over kernels themselves.

We start with time series data provided by the UCR Time Series Classification Archive.

This dataset contains 1-dimensional times series data from a variety of sources (such as household electricity usage, apparent brightness of stars, and seismometer readings).

In this work, we use 1000 time series randomly drawn from this archive, and normalise each to zero mean and unit variance.

For our model, we define the following simple grammar over kernels: DISPLAYFORM0 • WN is the White Noise kernel, DISPLAYFORM1 We wish to learn a prior distribution over both the symbolic structure of a kernel and its continuous variables (σ, l, etc.) .

Rather than describe a prior over kernel structures directly, we define the latent program to z to be a symbolic kernel 'expression': a string over the characters {(, ), +, * , WN, SE, Per, C} We define an LSTM prior p θ (z) over these kernel expressions, alongside parametric prior distributions over continuous latent variables (p θσ (σ), p θ l (l), . . .).

As in previous work, exact evaluation of the marginal likelihood p(x|z) of a kernel expression z is intractable and so requires an approximation.

For this we use a simple variational inference scheme which cycles through coordinate updates to each continuous latent variable (up to 100 steps), and estimates a lowerbound on p(x|z) using 10 samples from the variational distribution.

Finally, following section 3.1, we place a hyperprior on the distribution over kernel expressions, using the grammar above (Eq. 3) as a reference distribution.

Examples of latent programs discovered by our model are displayed in Figure 5 .

These programs describe meaningful compositional structure in the time series data, and can also be used to make highly plausible extrapolations.

Figure 5: Kernels inferred by the WSR for various real time series in the UCR dataset.

Blue (left) is a 256-timepoint observation, and orange (right) is a sampled extrapolation using the inferred kernel (top, simplified where possible).

The explicit compositional structure of this latent representation allows each discovered concept to be easily translated into natural language.

Finally, we test our model on the task of learning short text concepts, such as 'phone number' or 'email address', from a few examples.

For this task we created a new dataset, Text-Concepts comprising 1500 of such concepts with 10 examples of each (Figure 1 ).To collect the data, we first crawled several thousand randomly chosen spreadsheets from online public repositories on GitHub.

We then automatically selected a subset of 1500 the columns from this set, filtered to remove columns that contain only numbers, English words longer than 5 characters, or common first names and surnames.

To promote diversity in the dataset, we also filtered so that no two columns originated from the same spreadsheet, and no more than 3 columns share the same column header.

This us to capture a wide variety of concept types (e.g. 'date', 'time') while maintaining variation that exists within each type.

Common to most patterns in the Text-Concepts dataset is that they can be well described by concatenative structure, as they usually involve the composition of discrete parts.

With this in mind, we aim to model of this dataset using the language of Regular Expressions (regex).We first define a grammar over regular expressions as follows, borrowing the standard syntax that is common to many programming languages: DISPLAYFORM0 where Character can produce any printable ASCII character, and is the empty string.

We assume that each class x i in the Text-Conceptsdataset can be described by a latent regular expression z i from this grammar.

However, for our purposes, we endow each regex z with probabilistic, generative semantics.

We define a likelihood (decoder) p θ (x|z) by placing probability distributions over every random choice involved in generating a string from regex, as given in TAB3 .To evaluate the probability of a regex z generating a set of strings i, we use dynamic programming to efficiently calculate the exact probability of the most probable parse for each string in the set, and multiply these to serve as our likelihood p θ (x|z).

As in the Gaussian Process example, our p θ (z) is parametrised as a simple LSTM, and we define a hyperprior over this by using the above grammar (Eq. 4) the reference grammar distribution.

For the recognition model we require a network which is able to generate a sequence of tokens (the regex) taking a set of strings as an input.

We achieve this using a variant of the RobustFill architecture, introduced in BID3 .

We pass each string in the set individually through an LSTM, and then attend to these while decoding a regular expression character by character.

Given this problem setup, our goal is to learn a regex z corresponding to each set of strings, x in the dataset, while also learning a global distribution p(z) and a recognition model r(z; x) to guide inference on novel sets.

For any regex expression e: e * evaluates to e+ with probability θ+, and with otherwise.

e+ evaluates to ee * .

e|e2 evaluates to e with probability θ | , and e2 otherwise.

e? evaluates to e with probability θ ? , and otherwise. .

evaluates to any character, with probabilities θ.

w evaluates to any alphanumeric character, with probabilities θw d evaluates to any digit, with probabilities θ d u evaluates to any uppercase character, with probabilities θu l evaluates to any lowercase character, with probabilities θ l s evaluates to any whitespace character, with probabilities θs where θ are parameters to be learned Quantitative results from training the above model using the WSR algorithm (with k = 5), are shown in FIG0 .

From five examples of each concept, WSR learns a regular expression that generalises well to new examples, achieving over 75% accuracy in a challenging 100-way classification task.

Comparing to Wake-Sleep and No-Recognition baselines, we find that WSR crucially utilises on both its recognition model and its memory in order to achieve this result -neither are sufficient alone.

The VAE algorithm was unable to learn effectively in our regex model, even when using control variates to reduce variance.

For a more fair comparison, we also provide results from training a VAE using a different model architecture to which is better suited: for VAE-LSTM we use a 32-dimensional vector for the latent representation, with a fixed Gaussian prior p(z), and LSTM networks for both p(x|z) and q(z|x).

While this model is able to optimise its training objective effectively, it instead suffers from the lack of domain knowledge built into its structure.

The latent representations it infers for concepts are not only less explicit but also generalise less effectively to new examples of a given concept.

EA., SD., CSB., . . .

TAB5 .3 ordered by WSR ELBo (descending).Finally, investigate whether WSR learns a realistic inductive bias over concepts, by sampling new concepts from the learned prior p θ (z) and then for each of these sampling a set of instances from p θ (x|z).

In TAB5 .3, we see that our model generalises meaningfully from the training data, learning higher level part structure that is common in the dataset (e.g. strings of uppercase characters) and then composing these parts in new ways.

In this paper, we consider learning interpretable concepts from one or a few examples: a difficult task which gives rise to both inductive and computational challenges.

Inductively, we aim to achieve strong generalisation by starting with rich domain knowledge and then 'filling in the gaps', using a large amount of background data.

Computationally, we aim to tackle the challenge of finding high-probability programs by using a neural recognition model to guide search.

Putting these pieces together we propose the Wake-Sleep-Remember algorithm, in which a Helmholtz machine is augmented with an persistent memory of discovered latent programs -optimised as a finite variational posterior.

We demonstrate on several domains that our algorithm can learn generalisable concepts, and comparison with baseline models shows that WSR (a) utilises both its recognition model and its memory in order to search for programs effectively, and (b) utilises both domain knowledge and extensive background data in order to make strong generalisations.

@highlight

We extend the wake-sleep algorithm and use it to learn to learn structured models from few examples, 