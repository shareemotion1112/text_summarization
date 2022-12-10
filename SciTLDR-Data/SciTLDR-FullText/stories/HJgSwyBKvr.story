Learning disentangled representations that correspond to factors of variation in real-world data is critical to interpretable and human-controllable machine learning.

Recently, concerns about the viability of learning disentangled representations in a purely unsupervised manner has spurred a shift toward the incorporation of weak supervision.

However, there is currently no formalism that identifies when and how weak supervision will guarantee disentanglement.

To address this issue, we provide a theoretical framework—including a calculus of disentanglement— to assist in analyzing the disentanglement guarantees (or lack thereof) conferred by weak supervision when coupled with learning algorithms based on distribution matching.

We empirically verify the guarantees and limitations of several weak supervision methods (restricted labeling, match-pairing, and rank-pairing), demonstrating the predictive power and usefulness of our theoretical framework.

Many real-world data can be intuitively described via a data-generating process that first samples an underlying set of interpretable factors, and then-conditional on those factors-generates an observed data point.

For example, in image generation, one might first generate the object identity and pose, and then build an image of this object accordingly.

The goal of disentangled representation learning is to learn a representation where each dimensions of the representation measures a distinct factor of variation in the dataset (Bengio et al., 2013) .

Learning such representations that align with the underlying factors of variation may be critical to the development of machine learning models that are explainable or human-controllable (Gilpin et al., 2018; Lee et al., 2019; Klys et al., 2018) .

In recent years, disentanglement research has focused on the learning of such representations in an unsupervised fashion, using only independent samples from the data distribution without access to the true factors of variation (Higgins et al., 2017; Chen et al., 2018a; Kim & Mnih, 2018; Esmaeili et al., 2018) .

However, Locatello et al. (2019) demonstrated that many existing methods for the unsupervised learning of disentangled representations are brittle, requiring careful supervision-based hyperparameter tuning.

To build robust disentangled representation learning methods that do not require large amounts of supervised data, recent work has turned to forms of weak supervision (Chen & Batmanghelich, 2019; Gabbay & Hoshen, 2019) .

Weak supervision can allow one to build models that have interpretable representations even when human labeling is challenging (e.g., hair style in face generation, or style in music generation).

While existing methods based on weaklysupervised learning demonstrate empirical gains, there is no existing formalism for describing the theoretical guarantees conferred by different forms of weak supervision (Kulkarni et al., 2015; Reed et al., 2015; Bouchacourt et al., 2018) .

In this paper, we present a comprehensive theoretical framework for weakly supervised disentanglement, and evaluate our framework on several datasets.

Our contributions are several-fold.

2.

We propose a set of definitions for disentanglement that can handle correlated factors and are inspired by many existing definitions in the literature (Higgins et al., 2018; Suter et al., 2018; Ridgeway & Mozer, 2018) .

3. Using these definitions, we provide a conceptually useful and theoretically rigorous calculus of disentanglement.

4.

We apply our theoretical framework of disentanglement to analyze the theoretical guarantees of three notable weak supervision methods (restricted labeling, match pairing, and rank pairing) and experimentally verify these guarantees.

Our goal in disentangled representation learning is to identify a latent-variable generative model whose latent variables correspond to ground truth factors of variation in the data.

To identify the role that weak supervision plays in providing guarantees on disentanglement, we first formalize the model families we are considering, the forms of weak supervision, and finally the metrics we will use to evaluate and prove components of disentanglement.

We consider data-generating processes where S ∈ R n are the factors of variation, with distribution p * (s), and X ∈ R m is the observed data point which is a deterministic function of S, i.e., X = g * (S).

Many existing algorithms in unsupervised learning of disentangled representations aim to learn a latent-variable model with prior p(z) and generator g, where g(Z) d = g * (S).

However, simply matching the marginal distribution over data is not enough: the learned latent variables Z and the true generating factors S could still be entangled with each other (Locatello et al., 2019) .

To address the failures of unsupervised learning of disentangled representations, we leverage weak supervision, where information about the data-generating process is conveyed through additional observations.

By performing distribution matching on an augmented space (instead of just on the observation of X), we can provide guarantees on learned representations.

We consider three practical forms of weak supervision: restricted labeling, match pairing, and rank pairing.

All of these forms of supervision can be thought of as augmented forms of the original joint distribution, where we partition the latent variables in two S = (S I , S \I ), and either observe a subset of the latent variables or share latents between multiple samples.

A visualization of these augmented distributions is presented in Figure 1 , and below we detail each form of weak supervision.

In restricted labeling, we observe a subset of the ground truth factors, S I in addition to X. This allows us to perform distribution matching on p * (s I , x), the joint distribution over data and observed factors, instead of just the data, p * (x), as in unsupervised learning.

This form of supervision is often leveraged in style-content disentanglement, where labels are available for content but not style (Kingma et al., 2014; Narayanaswamy et al., 2017; Chen et al., 2018b; Gabbay & Hoshen, 2019) .

Match Pairing uses paired data, (x, x ) that share values for a known subset of factors, s I .

In many data modalities, certain factors of variation may be difficult to prescribe as an explicit label, but it is easier to collect pairs of samples that share the same underlying factor (e.g., it may be easier to collect pairs of images of different people wearing the same glasses, than to explicitly define a label for glasses style).

Match pairing is a weaker form of supervision than restricted labeling, as the learning algorithm no longer depends on the underlying value s I .

Several variants of match pairing have appeared in the literature (Kulkarni et al., 2015; Bouchacourt et al., 2018; Ridgeway & Mozer, 2018) , but typically focus on groups of observations in contrast to the paired setting we consider in this paper.

Rank Pairing is another form of paired data generation where the pairs (x, x ) are generated in an i.i.d.

fashion, and an additional indicator variable y is observed that determines whether the corresponding latent s i is greater than s i : y = 1 {s i ≥ s i }.

Such a form of supervision is effective when it is easier to compare two samples with respect to an underlying factor than to directly collect labels (e.g., comparing two object sizes versus providing a ruler measurement of an object).

Although supervision via ranking features prominently in the metric learning literature (McFee & Lanckriet, 2010; Wang et al., 2014) , our focus in this paper will be on rank pairing in the context of disentanglement guarantees.

For each form of weak supervision, we can train generative models with the same structure as in Figure 1 , using data sampled from the ground truth model and a distribution matching objective.

For example, for match pairing, we train a generative model (p(z), g) such that the paired random variable (g(Z I , Z \I ), g(Z I , Z \I )) from the generator matches the distribution of the corresponding paired random variable (g * (S I , S \I ), g * (S I , S \I )) from the augmented data distribution.

To identify the role that weak supervision plays in providing guarantees on disentanglement, we introduce a set of definitions that are consistent with our intuitions about what constitutes "disentanglement" and amenable to theoretical analysis.

Our new definitions decompose disentanglement into two distinct concepts: consistency and restrictiveness.

Different forms of weak supervision can enable consistency or restrictiveness on subsets of factors, and in Section 4 we build up a calculus of disentanglement from these primitives.

We discuss the relationship to prior definitions of disentanglement in Appendix A.

Figure 2: Illustration of disentanglement, consistency, and restrictiveness of z 1 with respect to the factor of variation size.

Each image of a shape represents the decoding g(z 1:3 ) by the generative model.

Each column denotes a fixed choice of z 1 .

Each row denotes a fixed choice of (z 2 , z 3 ).

A demonstration of consistency versus restrictiveness on models from disentanglement lib is available in Appendix B.

To ground our discussion of disentanglement in a concrete example, we shall consider an oracle that generates shapes, with the underlying factors of variation size (S 1 ), shape (S 2 ), and color (S 3 ).

We now wish to determine whether Z 1 of our generative model disentangles the concept of size.

Intuitively, one way to check whether Z 1 of the generative model disentangles size (S 1 ) is to visually inspect what happens as we vary Z 1 , Z 2 , and Z 3 , and see whether the resulting visualizations are consistent with Figure 2a .

In doing so, our visual inspection checks for two properties:

1.

When Z 1 is fixed, the size (S 1 ) of the generated object never changes.

2.

When Z 1 is changed, the change is restricted to the size (S 1 ) of the generated object.

We thus argue that disentanglement decomposes into these two properties, which we refer to as generator consistency and generator restrictiveness.

We shall now formalize these two properties.

Let H be a hypothesis class of generative models from which we assume the true data-generating function is drawn.

Each element of the hypothesis class H is a tuple (p(s), g, e), where p(s) describes the distribution over factors of variation, the generator g is a function that maps from the factor space S ∈ R n to the observation space X ∈ R m , and the encoder e is a function that maps from X → S. S and X can consist of both discrete and continuous random variables.

We impose a few mild assumptions on H (see Appendix I.1).

Notably, we assume every factor of variation is exactly recoverable from the observation X, i.e. e(g(S)) = S.

Given an oracle model h * = (p * , g * , e * ) ∈ H, we would like to learn a model h = (p, g, e) ∈ H whose latent variables disentangle the latent variables in h * .

We refer to the latent-variables in the oracle h * as S and the alternative model h's latent variables as Z. If we further restrict h to only those models where g(Z) d = g * (S) are equal in distribution, it is natural to align Z and S via S = e * •g (Z) .

Under this relation between Z and S, our goal is to construct definitions that describe whether the latent code Z i disentangles the corresponding factor S i .

Generator Consistency.

Let I denote a set of indices and p I denote the generating process

This generating process samples Z I once and then conditionally samples Z I twice in an i.i.d.

fashion.

We say that Z I is consistent with S I if

where e * I is the oracle encoder restricted to the indices I. Intuitively, Equation (3) states that, for any fixed choice of Z I , resampling of Z \I will not influence the oracle's measurement of the factors S I .

In other words, S I is invariant to changes in Z \I .

An illustration of a generative model where Z 1 is consistent with size (S 1 ) is provided in Figure 2b .

A notable property of our definition is that the prescribed sampling process p I does not require the underlying factors of variation to be statistically independent.

We characterize this property in contrast to previous definitions of disentanglement in Appendix A.

Generator Restrictiveness.

Let p \I denote the generating process

We say that Z I is restricted to S I if

Equation (6) states that, for any fixed choice of Z \I , resampling of Z I will not influence the oracle's measurement of the factors S \I .

In other words, S \I is invariant to changes in Z I .

Thus, changing Z I is restricted to modifying only S I .

An illustration of a generative model where Z 1 is restricted to size (S 1 ) is provided in Figure 2c .

Generator Disentanglement.

We now say that Z I disentangles S I if Z I is consistent with and restricted to S I .

If we denote consistency and restrictiveness via Boolean functions C(I) and R(I), we can now concisely state that

where D(I) denotes whether Z I disentangles S I .

An illustration of a generative model where Z 1 disentangles size (S 1 ) is provided in Figure 2a .

Note that while size increases monotonically with Z 1 in the figure for convenience of illustration, we wish to clarify that monotonicity is orthogonal to the concepts of consistency and restrictiveness.

Under our mild assumptions on H, distribution matching on g(Z) d = g(S) combined with generator disentanglement on factor I implies the existence of two invertible functions f I and f \I such that the alignment via S = e * • g(Z) decomposes into

This expression highlights the connection between disentanglement and invariance, whereby S I is only influenced by Z I , and S \I is only influenced by Z \I .

However, such a bijectivity-based definition of disentanglement does not naturally expose the underlying primitives of consistency and restrictiveness, which we shall demonstrate in our theory and experiments to be valuable concepts for describing disentanglement guarantees under weak supervision.

Our proposed definitions are asymmetric-measuring the behavior of a generative model against an encoder.

So far, we have chosen to present the definitions from the perspective of a learned generator (p, g) measured against an oracle encoder e * .

In this sense, they are generator-based definitions.

We can also develop a parallel set of definitions for encoder-based consistency, restrictiveness, and disentanglement within our framework simply by using an oracle generator (p * , g * ) measured against a learned encoder e. We only present consistency for brevity.

Encoder Consistency.

Let p * I denote the generating process

This generating process samples S I once and then conditionally samples S I twice in an i.i.d.

fashion.

We say that S I is consistent with Z I if

We now make two important observations.

First, a valuable trait of our encoder-based definitions is that one can check for encoder consistency / restrictiveness / disentanglement as long as one has access to match pairing data from the oracle generator.

This is in contrast to the existing disentanglement definitions and metrics, which require access to the ground truth factors (Higgins et al., 2017; Kumar et al., 2018; Kim & Mnih, 2018; Chen et al., 2018a; Suter et al., 2018; Ridgeway & Mozer, 2018; Eastwood & Williams, 2018) .

The ability to check for our definitions in a weakly supervised fashion is the key to why we can develop a theoretical framework using the language of consistency and restrictiveness.

Second, encoder-based definitions are tractable to measure when testing on synthetic data, since the synthetic data directly serves the role of the oracle generator.

As such, while we develop our theory to guarantee both generator-based and the encoder-based disentanglement, all of our measurements in the experiments will be conducted with respect to a learned encoder.

We make three remarks on notations.

First, based) .

Where important, we shall make this dependency explicit (e.g., let D(I ; p, g, e * ) denote generator-based disentanglement).

We apply these conventions to C and R analogously.

We note several interesting relationships between restrictiveness and consistency.

First, by definition, C(I) is equivalent to R(\I).

Second, we can see from Figures 2b and 2c that C(I) and R(I) do not imply each other.

Based on these observations and given that consistency and restrictiveness operate over subsets of the random variables, a natural question that arises is whether consistency or restrictiveness over certain sets of variables imply additional properties over other sets of variables.

We develop a calculus for discovering implied relationships between learned latent variables Z and ground truth factors of variation S given known relationships as follows.

Our calculus provides a theoretically rigorous procedure for reasoning about disentanglement.

In particular, it is no longer necessary to prove whether the supervision method of interest satisfies consistency and restrictiveness for each and every factor.

Instead, it suffices to show that a supervision method guarantees consistency or restrictiveness for a subset of factors, and then combine multiple supervision methods via the calculus to guarantee full disentanglement.

We can additionally use the calculus to uncover consistency or restrictiveness on individual factors when weak supervision is available only for sets of variables.

For example, achieving consistency on S 1,2 and S 2,3 implies consistency on the intersection S 2 .

Furthermore, we note that these rules are agnostic to using generator or encoder-based definitions.

We defer the complete proofs to Appendix I.2.

In this section, we address the important question of how to distinguish when disentanglement arises from the supervision method and when it comes from model inductive bias.

This challenge was first put forth by Locatello et al. (2019) , which noted that unsupervised disentanglement is heavily reliant on model inductive bias.

As we transition toward supervised approaches, it is crucial that we formalize what it means for disentanglement to be guaranteed by weak supervision.

Sufficiency for Disentanglement.

Let P denote a family of augmented distributions.

We say that a weak supervision method S : H → P is sufficient for learning a generator whose latent codes Z I disentangle the factors S I if there exists a learning algorithm A : P → H such that for any choice of (p * (s), g * , e * ) ∈ H, the procedure A • S(p * (s), g * , e * ) returns a model (p(z), g, e) for which both D(I ; p, g, e * ) and D(I ; p * , g * , e) hold, and g(Z)

The key insight of this definition is that we force the strategy and learning algorithm pair (S, A) to handle all possible oracles drawn from the hypothesis class H. This prevents the exploitation of model inductive bias, since any bias from the learning algorithm A toward a reduced hypothesis classĤ ⊂ H will result in failure to handle oracles in the complementary hypothesis class H \Ĥ.

The distribution matching requirement g(Z) d = g * (S) ensures latent code informativeness, i.e., preventing trivial solutions where the latent code is uninformative (see Proposition 6 for formal statement).

Intuitively, distribution matching paired with a deterministic generator guarantees invertibility of the learned generator and encoder, enforcing that Z I cannot encode less information than S I (e.g., only encoding age group instead of numerical age) and vice versa.

We now apply our theoretical framework to three practical weak supervision methods: restricted labeling, match pairing, and rank pairing.

Our main theoretical findings are that: (1) These methods can be applied in a targeted manner to provide single factor consistency or restrictiveness guarantees.

(2) By enforcing consistency (or restrictiveness) on all factors, we can learn models with strong disentanglement performance.

Correspondingly, Figure 3 and Figure 5 are our main experimental results, demonstrating that these theoretical guarantees have predictive power in practice.

We prove that if a training algorithm successfully matches the generated distribution to data distribution generated via restricted labeling, match pairing, or rank pairing of factors S I , then Z I is guaranteed to be consistent with S I : Theorem 1.

Given any oracle (p * (s), g * , e * ) ∈ H, consider the distribution-matching algorithm A that selects a model (p(z), g, e) ∈ H such that:

Then (p, g) satisfies C(I ; p, g, e * ) and e satisfies C(I ; p * , g * , e).

Theorem 1 states that distribution-matching under restricted labeling, match pairing, or rank pairing of S I guarantees both generator and encoder consistency for the learned generator and encoder respectively.

We note that while the complement rule C(I) =⇒ R(\I) further guarantees that Z \I is restricted to S \I , we can prove that the same supervision does not guarantee that Z I is restricted to S I (Theorem 2).

However, if we additionally have restricted labeling for S \I , or match pairing for S \I , then we can see from the calculus that we will have guaranteed R(I) ∧ C(I), thus implying disentanglement of factor I. We also note that while restricted labeling and match pairing can be applied on a set of factors at once (i.e. |I| ≥ 1), rank pairing is restricted to one-dimensional factors for which an ordering exists.

In the experiments below, we empirically verify the theoretical guarantees provided in Theorem 1.

We conducted experiments on five prominent datasets in the disentanglement literature: Shapes3D Reed et al., 2015) .

Since some of the underlying factors are treated as nuisance variables in SmallNORB and Scream-dSprites, we show in Appendix C that our theoretical framework can be easily adapted accordingly to handle such situations.

We use generative adversarial networks (GANs, Goodfellow et al. (2014) ) for learning (p, g) but any distribution matching algorithm (e.g., maximum likelihood training in tractable models, or VI in latent-variable models) could be applied.

Our results are collected over a broad range of hyperparameter configurations (see Appendix H for details).

Since existing quantitative metrics of disentanglement all measure the performance of an encoder with respect to the true data generator, we trained an encoder post-hoc to approximately invert the learned generator, and measured all quantitative metrics (e.g., mutual information gap) on the encoder.

Our theory assumes that the learned generator must be invertible.

While this is not true for conventional GANs, our empirical results show that this is not an issue in practice (see Appendix G).

We present three sets of experimental results: (1) Single-factor experiments, where we show that our theory can be applied in a targeted fashion to guarantee consistency or restrictiveness of a single factor.

(2) Consistency versus restrictiveness experiments, where we show the extent to which single-factor consistency and restrictiveness are correlated even when the models are only trained to maximize one or the other.

(3) Full disentanglement experiments, where we apply our theory to fully disentangle all factors.

A more extensive set of experiments can be found in the Appendix.

We empirically verify that single-factor consistency or restrictiveness can be achieved with the supervision methods of interest.

Note there are two special cases of match pairing: one where S i is the only factor that is shared between x and x and one where S i is the only factor that is changed.

We distinguish these two conditions as share pairing and change pairing, respectively.

Figure 3 : Heatmap visualization of ablation studies that measure either single-factor consistency or single-factor restrictiveness as a function of various supervision methods, conducted on Shapes3D.

Our theory predicts the diagonal components to achieve the highest scores.

Note that share pairing, change pairing, and change pair intersection are special cases of match pairing.

shows that restricted labeling, share pairing, and rank pairing of the i th factor are each sufficient supervision strategies for guaranteeing consistency on S i .

Change pairing at S i is equivalent to share pairing at S \i ; the complement rule C(I) ⇐⇒ R(\I) allows us to conclude that change pairing guarantees restrictiveness.

The first four heatmaps in Figure 3 show the results for restricted labeling, share pairing, change pairing, and rank pairing.

The numbers shown in the heatmap are the normalized consistency and restrictiveness scores.

We define the normalized consistency score as

This score is bounded on the interval [0, 1] (a consequence of Lemma 1) and is maximal when C(I ;p * , g * , e) is satisfied.

This normalization procedure is similar in spirit to that used in Suter et al. (2018)'s Interventional Robustness Score.

The normalized restrictiveness scorer can be analogously defined.

In practice, we estimate this score via Monte Carlo estimation.

The final heatmap in Figure 3 demonstrates the calculus of intersection.

In practice, it may be easier to acquire paired data where multiple factors change simultaneously.

If we have access to two kinds of datasets, one where S I are changed and one where S J are changed, our calculus predicts that training on both datasets will guarantee restrictiveness on S I∩J .

The final heatmap shows six such intersection settings and measures the normalized restrictiveness score; in all but one setting, the results are consistent with our theory.

We show in Figure 7 that this inconsistency is attributable to the failure of the GAN to distribution-match due to sensitivity to a specific hyperparameter.

We now determine the extent to which consistency and restrictiveness are correlated in practice.

In Figure 4 , we collected all 864 Shapes3D models that we trained in Section 6.2.1 and measured the consistency and restrictiveness of each model on each factor, providing both the correlation plot and scatterplots ofc(i) versusr(i).

Since the models trained in Section 6.2.1 only ever targeted the consistency or restrictiveness of a single factor, and since our calculus demonstrates that consistency and restrictiveness do not imply each other, one might a priori expect to find no correlation in Figure 4 .

Our results show that the correlation is actually quite strong.

Since this correlation is not guaranteed by our choice of weak supervision, it is necessarily a consequence of model inductive bias.

We believe this correlation between consistency and restrictiveness to have been a general source of confusion in the disentanglement literature, causing many to either observe or believe that restricted labeling or share pairing on S i (which only guarantees consistency) is sufficient for disentangling S i (Kingma et al., 2014; Chen & Batmanghelich, 2019; Gabbay & Hoshen, 2019; Narayanaswamy et al., 2017) .

It remains an open question why consistency and restrictiveness are so strongly correlated when training existing models on real-world data.

Figure 5: Disentanglement performance of a vanilla GAN, share pairing GAN, change pairing GAN, rank pairing GAN, and fully-labeled GAN, as measured by the mutual information gap across several datasets.

A comprehensive set of performance evaluations on existing disentanglement metrics is available in Figure 13 .

If we have access to share / change / rank-pairing data for each factor, our calculus states that it is possible to guarantee full disentanglement.

We trained our generative model on either complete share pairing, complete change pairing, or complete rank pairing, and measured disentanglement performance via the discretized mutual information gap (Chen et al., 2018a; Locatello et al., 2019) .

As negative and positive controls, we also show the performance of an unsupervised GAN and a fully-supervised GAN where the latents are fixed to the ground truth factors of variation.

Our results in Figure 5 empirically verify that combining single-factor weak supervision datasets leads to consistently high disentanglement scores.

In this work, we construct a theoretical framework to rigorously analyze the disentanglement guarantees of weak supervision algorithms.

Our paper clarifies several important concepts, such as consistency and restrictiveness, that have been hitherto confused or overlooked in the existing literature, and provides a formalism that precisely distinguishes when disentanglement arises from supervision versus model inductive bias.

Through our theory and a comprehensive set of experiments, we demonstrated the conditions under which various supervision strategies guarantee disentanglement.

Our work establishes several promising directions for future research.

First, we hope that our formalism and experiments inspire greater theoretical and scientific scrutiny of the inductive biases present in existing models.

Second, we encourage the search for other learning algorithms (besides distribution-matching) that may have theoretical guarantees when paired with the right form of supervision.

Finally, we hope that our framework enables the theoretical analysis of other promising weak supervision methods.

Our appendix consists of nine sections.

We provide a brief summary of each section below.

Appendix A: We elaborate on the connections between existing definitions of disentanglement and our definitions of consistency / restrictiveness / disentanglement.

In particular, we highlight three notable properties of our definitions not present in many existing definitions.

Appendix B: We evaluate our consistency and restrictiveness metrics on the 10800 models in the disentanglement lib, and identify models where consistency and restrictiveness are not correlated.

Appendix C: We adapt our definitions to be able to handle nuisance variables.

We do so through a simple modification of the definition of restrictiveness.

Appendix D: We show several additional single-factor experiments.

We first address one of the results in the main text that is not consistent with our theory, and explain why it can be attributed to hyperparameter sensitivity.

We next unwrap the heatmaps into more informative boxplots.

Appendix E: We provide an additional suite of consistency versus restrictiveness experiments by comparing the effects of training with share pairing (which guarantees consistency), change pairing (which guarantees restrictiveness), and both.

Appendix F: We provide full disentanglement results on all five datasets as measured according to six different metrics of disentanglement found in the literature.

Appendix G: We show visualizations of a weakly supervised generative model trained to achieve full disentanglement.

Appendix H: We describe the set of hyperparameter configurations used in all our experiments.

Appendix I: We provide the complete set of assumptions and proofs for our theoretical framework.

Numerous definitions of disentanglement are present in the literature (Higgins et al., 2017; Kim & Mnih, 2018; Suter et al., 2018; Ridgeway & Mozer, 2018; Eastwood & Williams, 2018; Chen et al., 2018a) .

We mostly defer to the terminology suggested by Ridgeway & Mozer (2018) , which decomposes disentanglement into modularity, compactness, and explicitness.

Modularity means a latent code Z i is predictive of at most one factor of variation S j .

Compactness means a factor of variation S i is predicted by at most one latent code Z j .

And explicitness means a factor of variation S j is predicted by the latent codes via a simple transformation (e.g. linear).

Similar to Eastwood & Williams (2018); Higgins et al. (2018) , we suggest a further decomposition of Ridgeway & Mozer (2018)'s explicitness into latent code informativeness and latent code simplicity.

In this paper, we omit latent code simplicity from consideration.

Since informativeness of the latent code is already enforced by our requirement that g(Z) is equal in distribution to g * (S) (see Proposition 6), we focus on comparing our proposed concepts of consistency and restrictiveness to modularity and compactness.

We make note of three important distinctions.

Restrictiveness is not synonymous with either modularity or compactness.

In Figure 2c , it is evident the factor of variation size is not predictable any individual Z i (conversely, Z 1 is not predictable from any individual factor S i ).

As such, Z 1 is neither a modular nor compact representation of size, despite being restricted to size.

To our knowledge, no existing quantitative definition of disentanglement (or its decomposition) specifically measures restrictiveness.

Consistency and restrictiveness are invariant to statistically dependent factors of variation.

Many existing definitions of disentanglement are instantiated by measuring the mutual information between Z and S. For example, Ridgeway & Mozer (2018) defines that a latent code Z i to be "ideally modular" if it has high mutual information with a single factor S j and zero mutual information with all other factors S \j .

This presents a issue when the true factors of variation themselves are statistically dependent; even if Z 1 = S 1 , the latent code Z 1 would violate modularity if S 1 itself has positive mutual information with S 2 .

Consistency and restrictiveness circumvent this issue by relying on conditional resampling.

Consistency, for example, only measures the extent to which S I is invariant to resampling of Z \I when conditioned on Z I and is thus achieved as long as s I is a function of only z I -irrespective of whether s I and s \I are statistically dependent.

In this regard, our definitions draw inspiration from Suter et al. (2018)'s intervention-based definition but replaces the need for counterfactual reasoning with the simpler conditional sampling.

Consistency and restrictiveness arise in weak supervision guarantees.

One of our goals is to propose definitions that are amenable to theoretical analysis.

As we can see in Section 4, consistency and restrictiveness serve as the core primitive concepts that we use to describe disentanglement guarantees conferred by various forms of weak supervision.

To better understand the empirical relationship between consistency and restrictiveness, we calculated the normalized consistency and restrictiveness scores on the suite of 12800 models from disentanglement lib for each ground-truth factor.

By using the normalized consistency and restrictiveness scores as probes, we were able to identify models that achieve high consistency but low restrictiveness (and vice versa).

In Fig. 6 , we highlight two models that are either consistent or restrictive for object color on the Shapes3D dataset.

In Fig. 6a , we can see that this factor consistenly represents object color, i.e. each column of images has the same object color, but as we move along rows we see that other factors change as well, e.g. object type, thus this factor is not restricted to object color.

In Fig. 6b , we see that varying the factor along each row results in changes to object color but to no other attributes.

However if we look across columns, we see that the representation of color changes depending on the setting of other factors, thus this factor is not consistent for object color.

Our theoretical framework can handle nuisance variables, i.e., variables we cannot measure or perform weak supervision on.

It may be impossible to label, or provide match-pairing on that factor of variation.

For example, while many features of an image are measurable (such as brightness and coloration), we may not be able to measure certain factors of variation or generate data pairs where these factors are kept constant.

In this case, we can let one additional variable η act as nuisance variable that captures all additional sources of variation / stochasticity.

Formally, suppose the full set of true factors is S ∪ {η} ∈ R n+1 .

We define η-consistency C η (I) = C(I) and η-restrictiveness R η (I) = R(I ∪ {η}).

This captures our intuition that, with nuisance variable, for consistency, we still want changes to Z \I ∪ {η} to not modify S I ; for restrictiveness, we want changes to Z I ∪ {η} to only modify S I ∪ {η}. We define η-disentanglement as D η (I) = C η (I) ∧ R η (I).

All of our calculus still holds where we substitute C η (I), R η (I), D η (I) for C(I), R(I), D(I); we prove one of the new full disentanglement rule as an illustration: Proposition 1.

Proof.

On the one hand,

In (Locatello et al., 2019), the "instance" factor in SmallNORB and the background image factor in Scream-dSprites are treated as nuisance variables.

By Proposition 1, as long as we perform weak supervision on all of the non-nuisance variables (via sharing-pairing, say) to guarantee their consistency with respect to the corresponding true factor of variation, we still have guaranteed full disentanglement despite the existence of nuisance variable and the fact that we cannot measure or perform weak supervision on nuisance variable.

Figure 7 : This is the same plot as Figure 7 , but where we restrict our hyperparameter sweep to always set extra dense = False.

See Appendix H for details about hyperparameter sweep.

Figure 12: Normalized consistency vs. restrictiveness score of different models on each factor (row) across different datasets (columns).

In many of the plots, we see that models trained via changesharing (blue) achieve higher restrictiveness; models trained via share-sharing (orange) achieve higher consistency; models trained via both techniques (green) simultaneously achieve restrictiveness and consistency in most cases.

, and fully-labeled GAN (purple), as measured by normalized consistency score of each factor (rows) across multiple datasets (columns).

Factors {3, 4, 5} in the first column shows that distribution matching to all six change / share pairing datasets is particularly challenging for the models when trained on certain hyperparameter choices.

However, since consistency and restrictiveness can be measured in weakly supervised settings, it suffices to use these metrics for hyperparameter selection.

We see in Figure 16 and Appendix G that using consistency and restrictiveness for hyperparameter selection serves as a viable weakly-supervised surrogate for existing fully-supervised disentanglement metrics.

Figure 15: Performance of a vanilla GAN (blue), share pairing GAN (orange), change pairing GAN (green), rank pairing GAN (red), and fully-labeled GAN (purple), as measured by normalized restrictiveness score of each factor (rows) across multiple datasets (columns).

Since restrictiveness and consistency are complementary, we see that the anomalies in Figure 14 are reflected in the complementary factors in this figure.

Figure 16: Scatterplot of existing disentanglement metrics versus average normalized consistency and restrictiveness.

Whereas existing disentanglement metrics are fully-supervised, it is possible to measure average normalized consistency and restrictiveness with weakly supervised data (sharepairing and match-pairing respectively), making it viable to perform hyperparameter tuning under weakly supervised conditions.

As a demonstration of the weakly-supervised generative models, we visualize our best-performing match-pairing generative models (as selected according to the normalized consistency score averaged across all the factors).

Recall from Figures 2a to 2c that, to visually check for consistency and restrictiveness, it is important that we not only ablate a single factor (across the column), but also show that the factor stays consistent (down the row).

Each block of 3 × 12 images in Figures 17 to 21 checks for disentanglement of the corresponding factor.

Each row is constructed by random sampling of Z \i and then ablating Z i .

Table 1 : We trained a probablistic Gaussian encoder to approximately invert the generative model.

The encoder is not trained jointly with the generator, but instead trained separately from the generative model (i.e. encoder gradient does not backpropagate to generative model).

During training, the encoder is only exposed to data generated by the learned generative model.

4 × 4 spectral norm conv.

32.

lReLU 4 × 4 spectral norm conv.

32.

lReLU 4 × 4 spectral norm conv.

64.

lReLU 4 × 4 spectral norm conv.

64.

lReLU flatten 128 spectral norm dense.

lReLU 2 × z-dim spectral norm dense Table 5 : Discriminator used for rank pairing.

For rank-pairing, we use a special variant of the projection discriminator, where the conditional logit is computed via taking the difference between the two pairs and multiplying by y ∈ {−1, +1}. The discriminator is thus implicitly taking on the role of an adversarially trained encoder that checks for violations of the ranking rule in the embedding space.

Parts in red are part of hyperparameter search.

Discriminator Body Applied Separately to x and x 4 × 4 spectral norm conv.

32 × width.

lReLU 4 × 4 spectral norm conv.

32 × width.

lReLU 4 × 4 spectral norm conv.

64 × width.

lReLU 4 × 4 spectral norm conv.

64 × width.

lReLU flatten If extra dense: 128 × width spectral norm dense.

lReLU concatenate the pair.

Unconditional Head Applied Separately to x and x 1 spectral norm dense with bias.

Conditional Head Applied Separately to x and x y-dim spectral norm dense.

Intuitively, this assumption allows transition from s 1:n to s 1:n via a series of modifications that are only in I or only in J. Note that zig-zag connectedness is necessary for restrictiveness union (Proposition 3) and consistency intersection (Proposition 4).

Fig. 22 gives examples where restrictiveness union is not satisfied when zig-zag connectedness is violated.

Assumption 3.

For arbitrary coordinate j ∈ [m] of g that maps to a continuous variable X j , we assume that g j (s) is continuous at s, ∀s ∈ B(S); For arbitrary coordinate j ∈ [m] of g that maps to a discrete variable X j , ∀s D where p(s D ) > 0, we assume that g j (s) is constant over each connected component of int(supp(p(s C | s D )).

Define B(X) analogously to B(S).

Symmetrically, for arbitrary coordinate i ∈ [n] of e that maps to a continuous variable S i , we assume that e i (x) is continuous at x, ∀x ∈ B(X); For arbitrary coordinate i ∈ [n] of e that maps to a discrete S i , ∀x D where p(x D ) > 0, we assume that e i (x) is constant over each connected component of int(supp(p(x C | x D )).

Assumption 4.

Assume that every factor of variation is recoverable from the observation X .

Formally, (p, g, e) satisfies the following property E p(s1:n) e • g(s 1:n ) − s 1:n 2 = 0.

I.2 CALCULUS OF DISENTANGLEMENT I.2.1 EXPECTED-NORM REDUCTION LEMMA Lemma 1.

Let x, y be two random variables with distribution p, f (x, y) be arbitrary function.

Then E x∼p(x) E y,y ∼p(y|x) f (x, y) − f (x, y ) 2 ≤ E (x,y),(x ,y )∼p(x,y) f (x, y) − f (x , y ) 2 .

Proof.

Assume w.l.o.g that E (x,y)∼p(x,y) f (x, y) = 0.

LHS = 2E (x,y)∼p(x,y) f (x, y) 2 − 2E x∼p(x) E y,y ∼p(y|x) f (x, y) T f (x, y )

= 2E (x,y)∼p(x,y) f (x, y) 2 − 2E x∼p(x) E y∼p(y|x) f (x, y) T E y ∼p(y|x) f (x, y )

= 2E (x,y)∼p(x,y) f (x, y) 2 − 2E x∼p(x) E y∼p(y|x) f (x, y)

= 2E (x,y)∼p(x,y) f (x, y) 2 − 2E (x,y),(x ,y )∼p(x,y) f (x, y) T f (x , y ) (22) = RHS.

Now we prove the forward direction:

Assume for the sake of contradiction that ∃(z I , z \I ), (z I , z \I ) ∈ B(Z) such that f (z I , z \I ) < f (z I , z \I ).

Denote U = I ∩ D, V = I ∩ C, W = \I ∩ D, Q = \I ∩ C.

We have f (z U , z V , z W , z Q ) < f (z U , z V , z W , z Q ).

Since f is continuous (or constant) at (z U , z V , z W , z Q ) in the interior of B([z U , z W ]), and f is also continuous (or constant) at (z U , z V , z W , z Q ) in the interior of B([z U , z W ]), we can draw open balls of radius r > 0 around each point, i.e., B r (z V , z Q ) ⊂ B([z U , z W ]) and B r (z V , z Q ) ⊂ B([z U , z W ]), where

When we draw z \I ∼ p(z \I ), z I , z I ∼ p(z I |z \I ), let C denote the event that (z I , z \I ) = (z * V , z U , z 2 > 0 whenever event C happens, which contradicts R(I).

Therefore ∀(z I , z \I ), (z I , z \I ) ∈ B(Z), f (z I , z \I ) = f (z I , z \I ).

We have shown that

Similarly

Let the zig-zag path between

.

Repeatedly applying the equivalent conditions of R(I) and R(J) gives us Proof.

C(I) ∧ C(J) =⇒ R(\I) ∧ R(\J) (37) =⇒ R(\I ∪ \J) (38) =⇒ C(\(\I ∪ \J)) (39) =⇒ C(I ∩ J).

Proposition 5.

R(I) ∧ R(J) =⇒ R(I ∩ J).

Proof is analogous to Proposition 4.

Proposition 6.

If (p * , g * , e * ) ∈ H, and (p, g, e) ∈ H, and g * (S)

, then there exists a continuous function r such that E p(s1:n) r • e • g * (s) − s = 0.

@highlight

We construct a theoretical framework for weakly supervised disentanglement and conducted lots of experiments to back up the theory.