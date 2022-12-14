Disentangled representations, where the higher level data generative factors are reflected in disjoint latent dimensions, offer several benefits such as ease of deriving invariant representations, transferability to other tasks, interpretability, etc.

We consider the problem of unsupervised learning of disentangled representations from large pool of unlabeled observations, and propose a variational inference based approach to infer disentangled latent factors.

We introduce a regularizer on the expectation of the approximate posterior over observed data that encourages the disentanglement.

We also propose a new disentanglement metric which is better aligned with the qualitative disentanglement observed in the decoder's output.

We empirically observe significant improvement over existing methods in terms of both disentanglement and data likelihood (reconstruction quality).

Feature representations of the observed raw data play a crucial role in the success of machine learning algorithms.

Effective representations should be able to capture the underlying (abstract or high-level) latent generative factors that are relevant for the end task while ignoring the inconsequential or nuisance factors.

Disentangled feature representations have the property that the generative factors are revealed in disjoint subsets of the feature dimensions, such that a change in a single generative factor causes a highly sparse change in the representation.

Disentangled representations offer several advantages -(i) Invariance: it is easier to derive representations that are invariant to nuisance factors by simply marginalizing over the corresponding dimensions, (ii) Transferability: they are arguably more suitable for transfer learning as most of the key underlying generative factors appear segregated along feature dimensions, (iii) Interpretability: a human expert may be able to assign meanings to the dimensions, (iv) Conditioning and intervention: they allow for interpretable conditioning and/or intervention over a subset of the latents and observe the effects on other nodes in the graph.

Indeed, the importance of learning disentangled representations has been argued in several recent works BID5 BID37 BID50 .Recognizing the significance of disentangled representations, several attempts have been made in this direction in the past BID50 .

Much of the earlier work assumes some sort of supervision in terms of: (i) partial or full access to the generative factors per instance BID48 BID58 BID35 BID33 , (ii) knowledge about the nature of generative factors (e.g, translation, rotation, etc.) BID29 BID11 , (iii) knowledge about the changes in the generative factors across observations (e.g., sparse changes in consecutive frames of a Video) BID25 BID57 BID21 BID14 BID32 , (iv) knowledge of a complementary signal to infer representations that are conditionally independent of it 1 BID10 BID41 BID53 .

However, in most real scenarios, we only have access to raw observations without any supervision about the generative factors.

It is a challenging problem and many of the earlier attempts have not been able to scale well for realistic settings BID51 BID15 BID13 ) (see also, ).Recently, BID9 proposed an approach to learn a generative model with disentangled factors based on Generative Adversarial Networks (GAN) BID24 , however implicit generative models like GANs lack an effective inference mechanism 2 , which hinders its applicability to the problem of learning disentangled representations.

More recently, proposed an approach based on Variational AutoEncoder (VAE) BID34 for inferring disentangled factors.

The inferred latents using their method (termed as ??-VAE ) are empirically shown to have better disentangling properties, however the method deviates from the basic principles of variational inference, creating increased tension between observed data likelihood and disentanglement.

This in turn leads to poor quality of generated samples as observed in .In this work, we propose a principled approach for inference of disentangled latent factors based on the popular and scalable framework of amortized variational inference BID34 BID55 BID23 BID49 powered by stochastic optimization BID30 BID34 BID49 .

Disentanglement is encouraged by introducing a regularizer over the induced inferred prior.

Unlike ??-VAE , our approach does not introduce any extra conflict between disentanglement of the latents and the observed data likelihood, which is reflected in the overall quality of the generated samples that matches the VAE and is much better than ??-VAE.

This does not come at the cost of higher entanglement and our approach also outperforms ??-VAE in disentangling the latents as measured by various quantitative metrics.

We also propose a new disentanglement metric, called Separated Attribute Predictability or SAP, which is better aligned with the qualitative disentanglement observed in the decoder's output compared to the existing metrics.

We start with a generative model of the observed data that first samples a latent variable z ??? p(z), and an observation is generated by sampling from p ?? (x|z).

The joint density of latents and observations is denoted as p ?? (x, z) = p(z)p ?? (x|z).

The problem of inference is to compute the posterior of the latents conditioned on the observations, i.e., p ?? (z|x) = DISPLAYFORM0 .

We assume that we are given a finite set of samples (observations) from the true data distribution p(x).

In most practical scenarios involving high dimensional and complex data, this computation is intractable and calls for approximate inference.

Variational inference takes an optimization based approach to this, positing a family D of approximate densities over the latents and reducing the approximate inference problem to finding a member density that minimizes the Kullback-Leibler divergence to the true posterior, i.e., q * x = min q???D KL(q(z) p ?? (z|x)) BID6 .

The idea of amortized inference BID34 BID55 BID23 BID49 is to explicitly share information across inferences made for each observation.

One successful way of achieving this for variational inference is to have a so-called recognition model, parameterized by ??, that encodes an inverse map from the observations to the approximate posteriors (also referred as variational autoencoder or VAE) BID34 BID49 .

The recognition model parameters are learned by optimizing the problem min ?? E x KL(q ?? (z|x) p ?? (z|x)), where the outer expectation is over the true data distribution p(x) which we have samples from.

This can be shown as equivalent to maximizing what is termed as evidence lower bound (ELBO): DISPLAYFORM1 The ELBO (the objective at the right side of Eq. 1) lower bounds the log-likelihood of observed data, and the gap vanishes at the global optimum.

Often, the density forms of p(z) and q ?? (z|x) are chosen such that their KL-divergence can be written analytically in a closed-form expression (e.g., p(z) is N (0, I) and q ?? (z|x) is N (?? ?? (x), ?? ?? (x))) BID34 .

In such cases, the ELBO can be efficiently optimized (to a stationary point) using stochastic first order methods where both expectations are estimated using mini-batches.

Further, in cases when q ?? (??) can be written as a continuous transformation of a fixed base distribution (e.g., the standard normal distribution), a low variance estimate of the gradient over ?? can be obtained by coordinate transformation (also referred as reparametrization) BID22 BID34 BID49 .

Most VAE based generative models for real datasets (e.g., text, images, etc.) already work with a relatively simple and disentangled prior p(z) having no interaction among the latent dimensions (e.g., the standard Gaussian N (0, I)) BID7 BID43 BID31 BID59 .

The complexity of the observed data is absorbed in the conditional distribution p ?? (x|z) which encodes the interactions among the latents.

Hence, as far as the generative modeling is concerned, disentangled prior sets us in the right direction.

Although the generative model starts with a disentangled prior, our main objective is to infer disentangled latents which are potentially conducive for various goals mentioned in Sec. 1 (e.g., invariance, transferability, interpretability).

To this end, we consider the density over the inferred latents induced by the approximate posterior inference mechanism, DISPLAYFORM0 which we will subsequently refer to as the inferred prior or expected variational posterior (p(x) is the true data distribution that we have only samples from).

For inferring disentangled factors, this should be factorizable along the dimensions, i.e., DISPLAYFORM1 This can be achieved by minimizing a suitable distance between the inferred prior q ?? (z) and the disentangled generative prior p(z).

We can also define expected posterior as DISPLAYFORM2 If we take KL-divergence as our choice of distance, by relying on its pairwise convexity (i.e., KL( BID56 , we can show that the distance between q ?? (z) and p ?? (z) is bounded by the objective of the variational inference: DISPLAYFORM3 DISPLAYFORM4 In general, the prior p(z) and expected posterior p ?? (z) will be different, although they may be close (they will be same when p ?? (x) = p ?? (x|z)p(z)dz is equal to p(x)).

Hence, variational posterior inference of latent variables with disentangled prior naturally encourages inferring factors that are close to being disentangled.

We think this is the reason that the original VAE (Eq. (1) ) has also been observed to exhibit some disentangling behavior on simple datasets such as MNIST BID34 .

However, this behavior does not carry over to more complex datasets BID4 BID39 , unless extra supervision on the generative factors is provided BID35 BID33 .

This can be due to: (i) p(x) and p ?? (x) being far apart which in turn causes p(z) and p ?? (z) being far apart, and (ii) the non-convexity of the ELBO objective which prevents us from achieving the global minimum of E x KL(q ?? (z|x) p ?? (z|x)) (which is 0 and implies KL(q ?? (z) p ?? (z)) = 0).

In other words, maximizing the ELBO (Eq. (1)) might also result in reducing the value of KL(q ?? (z) p(z)), however, due to the aforementioned reasons, the gap between KL(q ?? (z) p(z)) and E x KL(q ?? (z|x) p ?? (z|x)) could be large at the stationary point of convergence.

Hence, minimizing KL(q ?? (z) p(z)) or any other suitable distance D(q ?? (z), p(z)) explicitly will give us better control on the disentanglement.

This motivates us to add D(q ?? (z) p(z)) as part of the objective to encourage disentanglement during inference, i.e., DISPLAYFORM5 where ?? controls its contribution to the overall objective.

We refer to this as DIP-VAE (for Disentangled Inferred Prior) subsequently.

Optimizing FORMULA7 directly is not tractable if D(??, ??) is taken to be the KL-divergence KL(q ?? (z) p(z)), which does not have a closed-form expression.

One possibility is use the variational formulation of the KL-divergence BID45 BID46 ) that needs only samples from q ?? (z) and p(z) to estimate a lower bound to KL(q ?? (z) p(z)).

However, this would involve optimizing for a third set of parameters ?? for the KL-divergence estimator, and would also change the optimization to a saddle-point (min-max) problem which has its own optimization challenges (e.g., gradient vanishing as encountered in training generative adversarial networks with KL or Jensen-Shannon (JS) divergences BID24 BID2 ).

Taking D to be another suitable distance between q ?? (z) and p(z) (e.g., integral probability metrics like Wasserstein distance BID54 ) might alleviate some of these issues but will still involve complicating the optimization to a saddle point problem in three set of parameters 3 .

It should also be noted that using these variational forms of the distances will still leave us with an approximation to the actual distance.

We adopt a simpler yet effective alternative of matching the moments of the two distributions.

Matching the covariance of the two distributions will amount to decorrelating the dimensions of DISPLAYFORM6 By the law of total covariance, the covariance of z ??? q ?? (z) is given by DISPLAYFORM7 where E q ?? (z|x) [z] and Cov q ?? (z|x) [z] are random variables that are functions of the random variable x (z is marginalized over).

Most existing work on the VAE models uses q ?? (z|x) having the form DISPLAYFORM8 , where ?? ?? (x) and ?? ?? (x) are the outputs of a deep neural net parameterized by ??.

In this case Eq. (5) reduces to Cov DISPLAYFORM9 , which we want to be close to the Identity matrix.

For simplicity, we choose entry-wise squared 2 -norm as the measure of proximity.

Further, ?? ?? (x) is commonly taken to be a diagonal matrix which means that cross-correlations (off-diagonals) between the latents are due to only DISPLAYFORM10 .

This suggests two possible options for the disentangling regularizer: DISPLAYFORM11 which we refer as DIP-VAE-II.

Penalizing just the off-diagonals in both cases will lead to lowering the diagonal entries of Cov p(x) [?? ?? (x)] as the ij'th off-diagonal is really a derived attribute obtained by multiplying the square-roots of i'th and j'th diagonals (for each example x ??? p(x), followed by averaging over all examples).

This can be compensated in DIP-VAE-I by a regularizer on the diagonal entries of Cov p(x) [?? ?? (x)] which pulls these towards 1.

We opt for two separate hyperparameters controlling the relative importance of the loss on the diagonal and off-diagonal entries as follows: DISPLAYFORM12 The regularization terms involving Cov p(x) [?? ?? (x)] in the above objective (6) can be efficiently optimized using SGD, where Cov p(x) [?? ?? (x)] can be estimated using the current minibatch 4 .For DIP-VAE-II, we have the following optimization problem: DISPLAYFORM13 As discussed earlier, the term DISPLAYFORM14 Penalizing the off-diagonals of Cov p(x) [?? ?? (x)] in the Objective (7) will contribute to reduction in the magnitude of its diagonals as discussed earlier.

As the regularizer on the diagonals is not directly on DISPLAYFORM15 , unlike DIP-VAE-I, it will be not be able to keep DISPLAYFORM16 ii such that their sum remains close to 1.

In datasets where the number of generative factors is less than the latent dimension, DIP-VAE-II is more suitable than DIP-VAE-I as keeping all dimensions active might result in splitting of an attribute across multiple dimensions, hurting the goal of disentanglement.

It is also possible to match higher order central moments of q ?? (z) and the prior p(z).

In particular, third order central moments (and moments) of the zero mean Gaussian prior are zero, hence 2 norm of third order central moments of q ?? (z) can be penalized.

Recently proposed ??-VAE proposes to modify the ELBO by upweighting the KL(q ?? (z|x) p(z)) term in order to encourage the inference of disentangled factors: DISPLAYFORM0 where ?? is taken to be great than 1.

Higher ?? is argued to encourage disentanglement at the cost of reconstruction error (the likelihood term in the ELBO).

Authors report empirical results with ?? ranging from 4 to 250 depending on the dataset.

As already mentioned, most VAE models proposed in the literature, including ??-VAE, work with N (0, I) as the prior p(z) and N (?? ?? (x), ?? ?? (x)) with diagonal ?? ?? (x) as the approximate posterior q ?? (z|x).

This reduces the objective (8) to DISPLAYFORM1 For high values of ??, ??-VAE would try to pull ?? ?? (x) towards zero and ?? ?? (x) towards the identity matrix (as the minimum of x ??? ln x for x > 0 is at x = 1), thus making the approximate posterior q ?? (z|x) insensitive to the observations.

This is also reflected in the quality of the reconstructed samples which is worse than VAE (?? = 1), particularly for high values of ??.

Our proposed method does not have such increased tension between the likelihood term and the disentanglement objective, and the sample quality with our method is on par with the VAE.Finally, we note that both ??-VAE and our proposed method encourage disentanglement of inferred factors by pulling Cov q ?? (z) (z) in Eq. (5) towards the identity matrix: ??-VAE attempts to do it by making Cov q ?? (z|x) (z) close to I and E q ?? (z|x) (z) close to 0 individually for all observations x, while the proposed method directly works on Cov q ?? (z) (z) (marginalizing over the observations x) which retains the sensitivity of q ?? (z|x) to the conditioned-upon observation.3 QUANTIFYING DISENTANGLEMENT: SAP SCORE propose a metric to evaluate the disentanglement performance of the inference mechanism, assuming that the ground truth generative factors are available.

It works by first sampling a generative factor y, followed by sampling L pairs of examples such that for each pair, the sampled generative factor takes the same value.

Given the inferred z x := ?? ?? (x) for each example x, they compute the absolute difference of these vectors for each pair, followed by averaging these difference vectors.

This average difference vector is assigned the label of y. By sampling n such minibatches of L pairs, we get n such averaged difference vectors for the factor y. This process is repeated for all generative factors.

A low capacity multiclass classifier is then trained on these vectors to predict the identities of the corresponding generative factors.

Accuracy of this classifier on the difference vectors for test set is taken to be a measure of disentanglement.

We evaluate the proposed method on this metric and refer to this as Z-diff score subsequently.

We observe in our experiments that the Z-diff score is not correlated well with the qualitative disentanglement at the decoder's output as seen in the latent traversal plots (obtained by varying only one latent while keeping the other latents fixed).

It also depends on the multiclass classifier used to obtain the score.

We propose a new metric, referred as Separated Attribute Predictability (SAP) score, that is better aligned with the qualitative disentanglement observed in the latent traversals and also does not involve training any classifier.

It is computed as follows: (i) We first construct a d ?? k score matrix S (for d latents and k generative factors) whose ij'th entry is the linear regression or classification score (depending on the generative factor type) of predicting j'th factor using only i'th latent [?? ?? (x)] i .

For regression, we take this to be the R 2 score obtained with fitting a line (slope and intercept) that minimizes the linear regression error (for the test examples).

The R 2 score is given by DISPLAYFORM2 and ranges from 0 to 1, with a score of 1 indicating that a linear function of the i'th inferred latent explains all variability in the j'th generative factor.

For classification, we fit one or more thresholds (real numbers) directly on i'th inferred latents for the test examples that minimize the balanced classification errors, and take S i j to be the balanced classification accuracy of the j'th generative factor.

For inactive latent Table 1 : Z-diff score , the proposed SAP score and reconstruction error (per pixel) on the test sets for 2D Shapes and CelebA (?? 1 = 4, ?? 2 = 60, ?? = 10, ?? 1 = 5, ?? 2 = 500 for 2D Shapes; ?? 1 = 4, ?? 2 = 32, ?? = 2, ?? 1 = 1, ?? 2 = 80 for CelebA).

For the results on a wider range of hyperparameter values, refer to Fig. 1 DISPLAYFORM3 ] ii close to 0), we take S ij to be 0.(ii) For each column of the score matrix S which corresponds to a generative factor, we take the difference of top two entries (corresponding to top two most predictive latent dimensions), and then take the mean of these differences as the final SAP score.

Considering just the top scoring latent dimension for each generative factor is not enough as it does not rule out the possibility of the factor being captured by other latents.

A high SAP score indicates that each generative factor is primarily captured in only one latent dimension.

Note that a high SAP score does not rule out one latent dimension capturing two or more generative factors well, however in many cases this would be due to the generative factors themselves being correlated with each other, which can be verified empirically using ground truth values of the generative factors (when available).

Further, a low SAP score does not rule out good disentanglement in cases when two (or more) latent dimensions might be correlated strongly with the same generative factor and poorly with other generative factors.

The generated examples using single latent traversals may not be realistic for such models, and DIP-VAE discourages this from happening by enforcing decorrelation of the latents.

However, the SAP score computation can be adapted to such cases by grouping the latent dimensions based on correlations and getting the score matrix at group level, which can be fed as input to the second step to get the final SAP score.

We evaluate our proposed method, DIP-VAE, on three datasets -(i) CelebA BID39 : It consists of 202, 599 RGB face images of celebrities.

We use 64 ?? 64 ?? 3 cropped images as used in several earlier works, using 90% for training and 10% for test. (ii) 3D Chairs BID4 : It consists of 1393 chair CAD models, with each model rendered from 31 azimuth angles and 2 elevation angles.

Following earlier work BID58 BID18 that ignores near-duplicates, we use a subset of 809 chair models in our experiments.

We use the binary masks of the chairs as the observed data in our experiments following .

First 80% of the models are used for training and the rest are used for test. (iii) 2D Shapes : This is a synthetic dataset of binary 2D shapes generated from the Cartesian product of the shape (heart, oval and square), x-position (32 values), y-position (32 values), scale (6 values) and rotation (40 values).

We consider two baselines for the task of unsupervised inference of disentangled factors: (i) VAE BID34 BID49 , and (ii) the recently proposed ??-VAE .

To be consistent with the evaluations in , we use the same CNN network architectures (for our encoder and decoder), and same latent dimensions as used in for CelebA, 3D Chairs, 2D Shapes datasets.

Hyperparameters.

For the proposed DIP-VAE-I, in all our experiments we vary ?? od in the set {1, 2, 5, 10, 20, 50, 100, 500} while fixing ?? d = 10?? od for 2D Shapes and 3D Chairs, and ?? d = 50?? od for CelebA. For DIP-VAE-II, we fix ?? od = ?? d for 2D Shapes, and ?? od = 2?? d for CelebA. Additionally, for DIP-VAE-II we also penalize the 2 -norm of third order central moments of q ?? (z) with hyperparameter ?? 3 = 200 for 2D Shapes data (?? 3 = 0 for CelebA).

For ??-VAE, we experiment with ?? = {1, 2, 4, 8, 16, 25, 32, 64, 100, 128 , 200 , 256} (where ?? = 1 corresponds to the VAE).

We used a batch size of 400 for all 2D Shapes experiments and 100 for all CelebA experiments.

For both CelebA and 2D Shapes, we show the results in terms of the Z-diff score , the Figure 1 : Proposed Separated Atomic Predictability (SAP) score and the Z-diff disentanglement score as a function of average reconstruction error (per pixel) on the test set of 2D Shapes data for ??-VAE and the proposed DIP-VAE.

The plots are generated by varying ?? for ??-VAE, and ?? od for DIP-VAE-I and DIP-VAE-II (the number next to each point is the value of these hyperparameters, respectively).

DISPLAYFORM0 is computed for every attribute k using the training set and a bias is learned by minimizing the hinge loss.

Accuracy on other attributes stays about same across all methods.

Arched proposed SAP score, and reconstruction error.

For 3D Chairs data, only two ground truth generative factors are available and the quantitative scores for these are saturated near the peak values, hence we show only the latent traversal plots which we based on our subjective evaluation of the reconstruction quality and disentanglement (shown in Appendix).Disentanglement scores and reconstruction error.

For the Z-diff score , in all our experiments we use a one-vs-rest linear SVM with weight on the hinge loss C set to 0.01 and weight on the regularizer set to 1.

Table 1 shows the Z-diff scores and the proposed SAP Figure 2 : The proposed SAP score and the Z-diff score as a function of average reconstruction error (per pixel) on the test set of CelebA data for ??-VAE and the proposed DIP-VAE.

The plots are generated by varying ?? for ??-VAE, and ?? od for DIP-VAE-I and DIP-VAE-II (the number next to each point is the value of these hyperparameters, respectively).scores along with reconstruction error (which directly corresponds to the data likelihood) for the test sets of CelebA and 2D Shapes data.

Further we also show the plots of how the Z-diff score and the proposed SAP score change with the reconstruction error as we vary the hyperparameter for both methods (?? and ?? od , respectively) in Fig. 1 (for 2D Shapes data) and Fig. 2 (for CelebA data).

The proposed DIP-VAE-I gives much higher Z-diff score at little to no cost on the reconstruction error when compared with VAE (?? = 1) and ??-VAE, for both 2D Shapes and CelebA datasets.

However, we observe in the decoder's output for single latent traversals (varying a single latent while keeping others fixed, shown in FIG0 and FIG1 ) that a high Z-diff score is not necessarily a good indicator of disentanglement.

Indeed, for 2D Shapes data, DIP-VAE-I has a higher Z-diff score (98.7) and almost an order of magnitude lower reconstruction error than ??-VAE for ?? = 60, however comparing the latent traversals of ??-VAE in FIG0 and DIP-VAE-I in FIG1 indicate a better disentanglement for ??-VAE for ?? = 60 (though at the cost of much worse reconstruction where every generated sample looks like a hazy blob).

On the other hand, we find the proposed SAP score to be correlated well with the qualitative disentanglement seen in the latent traversal plots.

This is reflected in the higher SAP score of ??-VAE for ?? = 60 than DIP-VAE-I. We also observe that for 2D Shapes data, DIP-VAE-II gives a much better trade-off between disentanglement (measured by the SAP score) and reconstruction error than both DIP-VAE-I and ??-VAE, as shown quantitatively in Fig. 1 and qualitatively in the latent traversal plots in FIG0 .

The reason is that DIP-VAE-I enforces [Cov p(x) [?? ?? (x)]] ii to be close to 1 and this may affect the disentanglement adversely by splitting a generative factor across multiple latents for 2D Shapes where the generative factors are much less than the latent dimension.

For real datasets having lots of factors with complex generative processes, such as CelebA, DIP-VAE-I is expected to work well which can be seen in Fig. 2 where DIP-AVE-I yields a much lower reconstruction error with a higher SAP score (as well as higher Z-diff scores).Binary attribute classification for CelebA. We also experiment with predicting the binary attribute values for each test example in CelebA from the inferred ?? ?? (x).

For each attribute k, we compute the attribute vector set, and project the ?? ?? (x) along these vectors.

A bias is learned on these scalars (by minimizing hinge loss) which is then used for classifying the test examples.

TAB1 shows the results for the attribute which show the highest change across various methods (most other attribute accuracies do not change).

The proposed DIP-VAE outperforms both VAE and ??-VAE for most attributes.

The performance of ??-VAE gets worse as ?? is increased further.

DISPLAYFORM0

Adversarial autoencoder BID40 also matches q ?? (z) (which is referred as aggregated posterior in their work) to the prior p(z).

However, adversarial autoencoder does not have the goal of minimizing KL(q ?? (z|x)||p ?? (z|x)) which is the primary goal of variational inference.

It DISPLAYFORM0 , where D is the distance induced by a discriminator that tries to classify z ??? q ?? (z) from z ??? p(z) by optimizing a cross-entropy loss (which induces JS-divergence as D).

This can be contrasted with the objective in (4).Invariance and Equivariance.

Disentanglement is closely connected to invariance and equivariance of representations.

If R : x ??? z is a function that maps the observations to the feature representions, equivariance (with respect to T ) implies that a primitive transformation T of the input results in a corresponding transformation T of the feature, i.e., R(T (x)) = T (R(x)).

Disentanglement requires that T acts only on a small subset of dimensions of R(x) (a sparse action).

In this sense, equivariance is a more general notion encompassing disentanglement as a special case, however this special case carries additional benefits of interpretability, ease of transferrability, etc.

Invariance is also a special case of equivariance which requires T to be identity for R to be invariant to the action of T on the input observations.

However, invariance can obtained more easily from disentangled representations than from equivariant representations by simply marginalizing the appropriate subset of dimensions.

There exists a lot of prior work in the literature on equivariant and invariant feature learning, mostly under the supervised setting which assumes the knowledge about the nature of input transformations (e.g., rotations, translations, scaling for images, etc.)

BID52 BID8 BID0 BID50 BID12 BID16 BID27 BID44 BID47 .

We proposed a principled variational framework to infer disentangled latents from unlabeled observations.

Unlike ??-VAE, our variational objective does not have any conflict between the data log-likelihood and the disentanglement of the inferred latents, which is reflected in the empirical results.

We also proposed the SAP disentanglement metric that is much better correlated with the qualitative disentanglement seen in the latent traversals than the Z-diff score .

An interesting direction for future work is to take into account the sampling biases in the generative process, both natural (e.g., sampling the female gender makes it unlikely to sample beard for face images in CelebA) as well as artificial (e.g., a collection of face images that contain much more smiling faces for males than females misleading us to believe p(gender,smile) = p(gender)p(smile)), which makes the problem challenging and also somewhat less well defined (at least in the case of natural biases).

Effective use of disentangled representations for transfer learning is another interesting direction for future work.

<|TLDR|>

@highlight

We propose a variational inference based approach for encouraging the inference of disentangled latents. We also propose a new metric for quantifying disentanglement. 