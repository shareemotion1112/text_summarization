Variational auto-encoders  (VAEs) offer a tractable approach when performing approximate inference in otherwise intractable generative models.

However, standard VAEs often produce latent codes that are disperse and lack interpretability, thus making the resulting representations unsuitable for auxiliary tasks (e.g. classiﬁcation) and human interpretation.

We address these issues by merging ideas from variational auto-encoders and sparse coding, and propose to explicitly model sparsity in the latent space of a VAE with a Spike and Slab prior distribution.

We derive the evidence lower bound using a discrete mixture recognition function thereby making approximate posterior inference as computational efﬁcient as in the standard VAE case.

With the new approach, we are able to infer truly sparse representations with generally intractable non-linear probabilistic models.

We show that these sparse representations are advantageous over standard VAE representations on two benchmark classiﬁcation tasks (MNIST and Fashion-MNIST) by demonstrating improved classiﬁcation accuracy and signiﬁcantly increased robustness to the number of latent dimensions.

Furthermore, we demonstrate qualitatively that the sparse elements capture subjectively understandable sources of variation.

Variational auto-encoders (VAEs) offer an efficient way of performing approximate posterior inference with otherwise intractable generative models and yield probabilistic encoding functions that can map complicated high-dimensional data to lower dimensional representations BID11 BID26 BID31 BID25 .

Making such representations meaningful and efficient, however, is a particularly difficult task and currently a major challenge in representation learning BID8 BID2 BID9 BID33 .

Large latent spaces often give rise to many latent dimensions that do not carry any information, and obtaining codes that properly capture the complexity of the observed data is generally problematic BID33 BID6 BID2 .In the case of linear mappings, sparse coding offers an elegant solution to the aforementioned problem; the representation space is induced to be sparse.

In such a way, the encoding function is encouraged to use the minimum number of non-zero elements necessary to describe each observation and condense information in few active variables, different for each sample BID22 BID7 .

In fact, due to their efficiency of representation, sparse codes have been used in many learning and recognition systems, as they provide easier interpretation BID14 BID1 BID18 BID0 and increased efficiency in, for example, classification, clustering, and transmission tasks when used as learning inputs BID38 BID35 BID12 .In this work, we aim to extent the aforementioned capability of linear sparse coding to non-linear probabilistic generative models thus allowing efficient, informative and interpretable representations in the general case.

To this end we formulate a new variation of the classical VAE in which we employ a sparsity inducing prior in the latent space based on the Spike and Slab distribution.

We match this by a discrete mixture recognition function that can map observations to sparse latent vectors.

Efficient inference, comparable in complexity to that of standard VAEs, is achieved by deriving an evidence lower bound (ELBO) for the new model which is optimized using standard gradient methods to recover the encoding and decoding functions.

In our experiments, we consider two benchmark dataset (MNIST and Fashion-MNIST) and show how the resulting ELBO is able to recover sparse, informative and interpretable representations regardless of the predefined number of latent dimensions.

The ability to adjust to data complexity allows to automatically discover the sources of variation in given observations, without the need to carefully adjust the architecture of a model to the given representation task.

We demonstrate these properties by first performing classification experiments using latent vectors as inputs, where we demonstrate that VSC representations marginally outperform VAE ones and display greatly improved robustness over large variations in latent space dimensionality.

Secondly we show that many sparse elements in retrieved codes control subjectively recognisable features in the generated observations.

2 BACKGROUND AND RELATED WORK 2.1 SPARSE CODING Sparse coding aims to approximately represent input vectors x i with a weighted linear combination of few unknown basis vectors b j BID14 BID1 BID15 .

The problem of determining the optimal basis and weights is generally formulated as the minimisation of an objective function of the following form arg min DISPLAYFORM0 where X ∈ R M ×N is the matrix of data, having as columns the input vectors x i ∈ R M ×1 , B ∈ R M ×J is the matrix having as columns the basis vectors b j ∈ R M ×1 , Z ∈ R J×N is the sparse codes matrix, having as columns the sparse codes z i ∈ R J×1 corresponding to the inputs x i , λ is a real positive parameter and φ(z i ) is a sparsity inducing function.

Sparse coding can be probabilistically interpreted as a generative model, where the observed vectors x i are generated from the unobserved latent variables z i through the linear process x i = Bz i + , where is the observation noise and is drawn from an isotropic normal distribution with zero mean BID14 BID1 .

The model can then be described with the following prior and likelihood distributions DISPLAYFORM1 where β is a real positive parameter, σ is the standard deviation of the observation noise and I is the identity matrix.

Performing maximum a posteriori (MAP) estimation with this model results in the minimisation shown in equation 1 with λ = σ 2 β.

In contrast to the MAP formulation, we are interested in maximising the marginal likelihood p(x) = p(x i ) and being able to perform such optimisation for arbitrarily complicated likelihood functions p(x|z).Previous work has demonstrated variational EM inference for such maximisation in the linear generative model case, with a particular choice of sparsity inducing prior BID32 BID4 .

However, EM inference becomes intractable for more complicated non-linear posteriors and a large number of input vectors BID11 , making such an approach unsuitable to scale to our desired model.

Conversely, some work has been done in generalising sparse coding to non-linear transformations, by defining sparsity on Riemannian manifolds BID7 BID3 .

These generalisations, however, perform MAP inference as they define a non-linear equivalent of the objective function in equation 1 and are limited to simple manifolds due to the need to compute the manifold's logarithmic map.

Variational auto-encoders (VAEs) are models for unsupervised efficient coding that aim to maximise the marginal likelihood p(x) = p(x i ) with respect to some decoding parameters θ of the likelihood function p θ (x|z) and encoding parameters φ of a recognition model q φ (z|x) BID11 ; BID26 ; BID24 Figure 1: Schematic representation of the variational sparse coding model (right) compared to a standard VAE (left).

In both cases an observed variable x i is assumed to be generated from an unobserved variable z i .

Variational sparse coding, however, models sparsity in the latent space with the Spike and Slab prior distribution.

One example is shown for each prior with a sample from the MNIST dataset.

variation in the observations.

Common choices are a Gaussian or a Bernoulli distributions.

The parameters of p θ (x|z) are the output of a neural network having as input a latent variable z i ∈ R J×1 .

The latent variable is assumed to be drawn from a prior p(z) which can be chosen to take different parametric forms.

In the most common VAE implementations, the prior takes the form of a multivariate Gaussian with identity covariance N (z; 0, I) BID11 BID26 BID6 BID2 BID39 .The aim is then to maximise a joint posterior distribution of the form p(x) = p θ (x i |z)p(z)dz, which for an arbitrarily complicated conditional p(x|z) is intractable.

To address this intractability, VAEs introduce a recognition model q φ (z|x) and define an evidence lower bound to be estimated in place of the true posterior.

The recognition function is a chosen to be a parametric distributions, where the parameters are the output of a neural network having as input a data point x i .

The ELBO can, due to Jensen's inequality, be formulated as DISPLAYFORM0 The ELBO is composed of two terms; a prior term, which encourages minimisation of the KL divergence between the encoding distributions and the prior, and a reconstruction term, which maximises the expectation of the data likelihood under the recognition function.

The VLB is then maximised with respect to the model's parameters θ and φ.

The prior term can be defined analytically, while the reconstruction term is optimised stochastically through a reparameterization trick BID11 .

Figure 1 (left) schematically depicts the model with an example of data and corresponding latent variable.

Discrete latent distributions are a closely related theme to sparsity, as exactly sparse PDFs involve sampling from some discrete variables.

BID21 and BID30 model VAEs with a Stick-Breaking Process and an Indian Buffet Process priors respectively in order to allow for stochastic dimensionality in the latent space.

In such a way, the prior can set to zero unused dimensions.

However, the resulting representations are not truly sparse; the same elements are set to zero for every encoded observation.

The scope of these works is dimensionality selection rather than sparsification.

Other models which present discrete variables in their latent space have been proposed in order to capture discrete features in natural observations.

Rolfe (2016) model a discrete latent space composed of continuous variables conditioned on discrete ones in order to capture both discrete and continuous sources of variation in observations.

Similarly motivated, van den BID34 perform variational inference with a learned discrete prior and recognition function.

The resulting latent spaces can present sparsity, depending on the choice of prior.

However, they do not induce directly sparse statistics in the latent space.

Perhaps the most closely related work to our own is the Epitomic VAE BID39 .

In this work, the authors propose to learn a deterministic selection variable that dictates which latent dimensions the recognition function should exploit in the latent space.

In such a way, different embeddings can exploit different combinations of variables, which achieves the goal of counteracting over-pruning.

This approach does result into sparse latent variables.

However, the method can be considered variational only in the continuous variables and the samples are not induced to present the statistics of a given sparse prior, but rather activate a constant number of elements in the latent vectors.

Differently from the aforementioned prior work, we aim to directly induce sparsity in a continuous latent space through a sparse PDF and find a suitable evidence lower bound to perform approximate variational inference.

We propose to use the framework of VAEs to perform approximate variational inference with neural network sparse coding architectures.

With this approach, we aim to discover and discern the nonlinear features that constitute variability in data and represent them as few non-zero elements in sparse vectors.

We model sparsity in the latent space with a Spike and Slab probability density prior.

The Spike and Slab PDF is a discrete mixture model which assigns point mass to null elements and therefore probabilistically models sparsity BID4 BID32 BID19 .

Because of this characteristic, this distribution has been used in various Bayesian sparse inference models BID28 BID20 BID29 BID5 .The Spike and Slab distribution is defined over two variables; a binary spike variable s j and a continuous slab variable z j BID19 .

The spike variable is either one or zero with defined probabilities α and (1 − α) respectively and the slab variable has a distribution which is either a Gaussian or a Delta function centered at zero, conditioned on whether the spike variable is one or zero respectively.

The prior probability density over the latent variable z we are interested in is then DISPLAYFORM0 where δ(·) indicates the Dirac delta function centered at zero.

This choice of prior leads to the assumption that observed data is generated from sparse vectors in the latent space.

The recognition function q φ (z|x) is chosen to be a discrete mixture model of the form DISPLAYFORM1 where the distribution parameters µ z,i,j , σ 2 z,i,j and γ i,j are the outputs of a neural network having parameters φ and input x i .

A description of the recognition function neural network can be found in appendix A.2.

Similarly to the standard Spike and Slab distribution of equation 4, the distribution of equation 5 can be described with Spike variables, having probabilities of being one γ i,j , and Slab variables having Gaussian distributions N (z i,j ; µ z,i,j , σ 2 z,i,j ).

On one side, this choice of recognition function allows for the posterior to match the prior, while on the other, the freedom to control the Gaussian moments and the Spike probabilities independently enables the model to encode information in the latent space.

Figure 1 (right) schematically depicts the model with an example of an observation and corresponding latent sparse vector.

A more detailed description of the model can be found in appendix A.As in the standard VAE setting, we aim to perform approximate variational inference by maximising a lower bound.

The ELBO we aim to maximise during training is of the form detailed in equation 3, with the Spike and Slab probability density function p s (z) of equation 4 as prior and the discrete mixture distribution of of equation 5 as recognition function q φ (z|x i ).

In the following subsections we derive the prior and reconstruction terms of the VSC lower bound under these conditions.

In this section we derive in closed form the regularisation component of the lower bound for our model, corresponding to the negative of the KL divergence between the discrete mixture of equation 5 and the Spike and Slab PDF.As both p s (z j ) and q φ (z j |x) are mixtures of Dirac Delta functions and Gaussians, the regularisation term can be split in four cross entropy component in each latent dimension; two Gaussian-discrete mixture components and two Dirac Delta-discrete mixture components: DISPLAYFORM0 The first and third term have the form of a cross entropy between a Gaussian and a discrete mixture distribution.

These components reduce to the corresponding weighted Gaussian-Gaussian entropy terms, as the point mass contributions vanish.

In fact, for any finite density distributions f (z j ) and g(z j ), the point mass contribution to the cross entropy between f (z j ) and a discrete mixture h(z j ) = αg(z j ) + (1 − α)δ(z j − c) is infinitesimal.

The proof is as follows: DISPLAYFORM1 ).Where the last term vanishes.

Applying this result to the first and third cross entropy terms gives the corresponding standard weighted Gaussian-Gaussian result, plus the normalisation constant γ i,j log(α/γ i,j ).

The second and fourth terms take the form of the cross entropy between a Dirac Delta function and a discrete mixture distribution.

In this case, instead, the continuous density contributions vanish: DISPLAYFORM2 Combining the two results, we obtain the prior term of the VSC lower bound DISPLAYFORM3 A more detailed derivation is provided in appendix B. This prior term naturally presents two components.

The first is the negative KL divergence between the distributions of the Slab variables, multiplied by the probability of z i,j being non-zero γ i,j .

This component gives a similar regularisation to that of the standard VAE and encourages the Gaussian components of the recognition function to match those of the prior, proportionally to the Spike probabilities γ i,j .

The second term is the negative KL divergence between the distributions of the Spike variables.

This term encourages the probabilities of the latent variables being non-zero γ i,j to match the prior Spike probability α.

Similarly to the standard VAE, the reconstruction term of the lower bound is estimated and maximised stochastically as follows DISPLAYFORM0 where the samples z i,l are drawn from the recognition function q φ (z|x i ).

As in the standard VAE, to make the reconstruction term differentiable with respect to the encoding parameters φ, we employ a reparameterization trick to draw from q φ (z|x i ).

To parametrise samples from the discrete binary component of q φ (z|x i ) we use a continuous relaxation of binary variables analogous to that presented in BID17 and BID27 .

We make use of two auxiliary noise variables and η, normally and uniformly distributed respectively. is used to draw from the Slab distributions, resulting in a reparametrisation analogous to the standard VAE BID11 .

η is used to parametrise draws of the Spike variables through a non-linear binary selection function T (y i,l ).

The two variables are then multiplied together to obtain the parametrised draw from q φ (z|x i ).

A more detailed description of the reparametrisation of sparse samples is reported in appendix C

Combining the prior and reconstruction terms from section 3.1 and 3.2, we obtain the estimation of the VSC lower bound DISPLAYFORM0 The final ELBO is relatively simple and of easy interpretation; the prior term is composed of the negative Spike and weighted Slab KL divergences, while the reconstruction term is the expectation of the likelihood under the recognition function PDF, estimated stochastically.

We also point out that for γ i,j = α = 1 we recover the lower bound of the standard VAE BID11 as expected from the definition of the model.

To train the VSC model, we maximise the ELBO in the form of equation 11 with respect to the encoding and decoding parameters φ and θ through gradient ascent.

We test the VSC model on two image datasets commonly used to benchmark learning performance; the hand written digits dataset MNIST BID13 and the more recent fashion items dataset fashion-MNIST BID36 , both composed of 28×28 grey scale images of handwritten digits and pieces of clothing respectively.

We also make use of the CelebA faces dataset BID16 to illustrate more qualitative results.

Details of these datasets are given in appendix D.2.

Various examples of latent sparse codes and corresponding reconstructions are shown in appendix E.1, while measurements of the latent space sparsity are presented in appendix E.2.In the following subsections we test the VSC model in different settings.

First, we evaluate the ELBO at varying prior sparsity and number of latent space dimensions.

Secondly, to evaluate quantitatively representation efficiency in the latent space, we test classification using latent variables as inputs.

Lastly, we qualitatively assess latent space interpretation by examining the effect of altering individual non-zero elements in the sparse codes.

Details of the experimental conditions can be found in appendix D.

We evaluate the ELBO at varying numbers of latent dimensions and different levels of prior sparsity α.

We first train a standard VAE at a varying number of latent dimensions imposing a limit of 20, 000 iterations.

For each dimensionality, we find the best performing initial step size for the Adam optimiser BID10 .

We then use identical settings in each condition to test VSCs lower bound with different prior sparsity.

Our evaluation performed on the test sets is shown in figure 2 .

Results for different iteration limits are included in appendix E.3.

DISPLAYFORM0 Figure 2: Test set ELBO for the VSC at varying number of latent dimensions.

The standard VAE reaches high ELBO for a correct choice of latent space dimensions, but drops rapidly for larger latent spaces.

With increasing sparsity in the latent space, the VSC drops in performance at the optimal VAE dimensionality, but remains more stable with larger latent spaces.

The standard VAE achieves high ELBO values provided that the size of its latent space is chosen correctly, but for spaces which are too large its performance rapidly drops, as encoding in many latent variables becomes increasingly difficult.

Conversely, the VSC reaches a lower maximum ELBO, but remain significantly more stable with more latent dimensions.

With few latent dimensions available, the sparsity imposed by the prior, controlled by the parameter α, is too restrictive to allow rich descriptions of the observations and matching of the prior simultaneously.

In this limit the ELBO of the VSC is comparable to that of a VAE, but slightly under-performs it.

With more latent dimensions, only a subset of the available elements is used to encode each observation, making learning efficiency more stable as the latent space grows in size.

An important focus of this work is the ability of VSC to recover latent codes which carry a high level of information about the input.

To test this aspect, we compare the representation efficiency of VAE and VSC by performing a standard classification experiment using the latent variables as input features.

In order to encourage information rich codes in the VSC, we set the prior Spike probability α to a low value of 0.01.

With this very sparse prior, the recognition function activates non-zero elements only when needed to reconstruct an observation, while the prior induces the remaining elements to be mostly null.

We train VAEs and VSCs at varying number of latent dimensions for 20, 000 iterations.

In each case, we use 5, 000 encoded labelled examples from the training sets to train a simple one layer fully DISPLAYFORM0 Figure 3: Classification performance of VSC and standard VAE at varying number of latent space dimensions.

The VAE reaches its peak performance for optimal choice of latent space dimensions, but yields inefficient codes if the latent space is too large.

VSC recovers efficient codes for arbitrarily large latent spaces which outperform the VAE ones as classification inputs.connected classifiers using the latent codes as inputs.

Figure 3 shows the classification performance obtained on the test set.

VSC is able to reliably recover efficient codes without the need to specify an optimal latent space size and also marginally outperforms the best VAE.

This is because the recognition function activates only the subset of variables it needs to describe each observation, regardless of the latent space dimensionality.

In such a way, the sources of variations in the observed data are automatically discovered and encoded into few non-zero elements.

The peak performance for the VSCs occurs at larger latent spaces than for the standard VAEs, indicating that there is a representation advantage in encoding to larger spaces with sparser solutions than into smaller dense codes.

Additional evaluations of classification accuracy at varying sparsity and number of labelled examples are shown in appendix E.4.

Lastly, we qualitatively examine the interpretation of the non-zero elements in the sparse codes recovered with the VSC model.

To this end, we encode several examples from the test sets of the Fashion-MNIST and CelebA datasets with VSCs trained with prior spike probability α = 0.01.

The Fashion-MNIST and CelebA examples were encoded in 200 and 800 latent dimensions respectively.

We then show the effect of altering individual non-zero components on the reconstructed observations.

Examples are shown in figure 4.We find that many of the non-zero elements in the latent codes control interpretable features of the generated observations, as shown in figure 4.

We further note that these results are not obtained through interpolation of many labelled examples, but simply by altering individually some of the few components activated by the recognition function.

Though we are not directly inducing interpretation in the latent space, sparsity does lead to a higher expectation of interpretability due to the conditional activation of only certain dimensions.

For a particular observation, the recognition function defines a low dimensional sub-space by activating only few non-zero elements that control the features necessary to describe such observation and similar ones, thereby defining a sort of sub-generative model for this type of objects (see appendix E.5 for examples of sampling in the sub-spaces defined by sparse encodings).

For different observations, the model can activate different subsets of non-zero elements, exploiting a larger space for the aggregate posterior.

In such a way, a particular example is described by a small subset of variables which are easier to manually explore, while the model can adjust its capacity to represent large and varied datasets.

It is also interesting to consider interpolation between different objects in the VSC latent space; as representations are sparse, so are interpolation vectors between them and we can examine their nonzero elements individually.

We show an example considering the interpolation between one image of a shirt and one of a t-shirt in the Fashion-MNIST dataset.

FIG1 shows the effect of altering individually the two largest interpolation vector elements for each example.

The first and largest of the two non-zero elements considered controls the sleeves, which can be added to the t-shirt and subtracted from the shirt by altering this element alone.

The second ele-

In this paper, we lay the general framework to induce sparsity in the latent space of VAEs, allowing approximate variational inference with arbitrarily complicated and probabilistic sparse coding models.

We derived a lower bound which is of clear interpretation and efficient to estimate and optimise, as the ELBO of a standard VAE.

With the resulting encoders, we recovered efficient sparse codes, which proved to be optimal learning inputs in standard classification benchmarks and exhibit good interpretation in many of their non-zero components.

We conclude that inducing sparsity in the latent space of generative models appears to be a promising route to obtaining useful codes, interpretable representations and controlled data synthesis, which are all outstanding challenges in VAEs and representation learning in general.

In future work, we aim to further study the properties of a sparse latent space with respect to its interpretation and features disentanglement capability.

We expect VSC to be able to model huge ensembles of varied data by sparsely populating large latent spaces, hence isolating the features that govern variability among similar objects in widely diverse aggregates of data.

We describe here the details of the VSC model and the architecture of the neural networks we employed as likelihood and recognition functions.

The likelihood function p(x|z i ) is composed of a neural network which takes as input a latent variable z i ∈ R J×1 and outputs the mean µ i ∈ R M ×1 and log variance log(σ DISPLAYFORM0 The log likelihood of a sample x i is then computed evaluating the log probability density assigned to x i by a Gaussian having mean µ i and standard deviation σ i .In our experiments we use a one hidden layer fully connected neural network for the VSCs trained with the MNIST and Fashion-MNIST datasets and a two hidden layers network for the VSCs trained with the CelebA dataset.

The recognition function p(z|x i ) is composed of a neural network which takes as input an observation x i ∈ R M ×1 and outputs the mean µ z,i ∈ R J×1 , the log variance log(σ 2 z,i ) ∈ R J×1 and the log Spike probabilities vector log(γ i ) ∈ R J×1 .

The elements of γ i need to be constrained between 0 and 1, therefore, other than using log(γ i ) as output, which ensures γ i > 0, we employ a ReLU non-linearity at this output of the neural network as follows DISPLAYFORM0 Where v out,i is output to the same standard neural network that outputs µ z,i and log(σ 2 z,i ).

This ensures that γ i < 1.

Samples in the latent space z i,l can then be drawn as detailed in equation 24.

As for the likelihood function, we use a one hidden layer fully connected neural network for the VSCs trained with the MNIST and Fashion-MNIST datasets and a two hidden layers network for the VSCs trained with the CelebA dataset.

We report here a detailed derivation of the VSC lower bound prior term shown in equation 9.

As described in section 3, the lower bound we aim to maximise has the same form as the standard VAE one of equation 3, with the Spike and Slab probability density function p s (z) of equation 4 as prior and the discrete mixture distribution of of equation 5 as recognition function q φ (z|x i ).

The VSC lower bound prior term is therefore obtained by substituting these distribution in the negative KL divergence term of equation 3.

By doing so, we obtain four cross entropy components in each latent dimension DISPLAYFORM0 1 and 3 are of a similar form; the cross entropy between a Gaussian and a discrete mixture distributions.

These components reduce to the corresponding Gaussian-Gaussian entropy terms, as the point mass contributions vanish.

In fact, for any finite density distributions f (z j ) and g(z j ), the point mass contribution to the cross entropy between f (z j ) and a discrete mixture h(z j ) = αg(z j ) + (1 − α)δ(z j − c) is infinitesimal.

The proof is as follows: The cross entropy between the functions f (z j ) and h(z j ) is DISPLAYFORM1 We can split this integral in two components over two different domains, the first in the region where z j = c and the second in the region where z j = c. By using a Dirac Delta function, the first component can be expressed as follows DISPLAYFORM2 where from the first to the second line we can ignore the component containing δ(z j − c), as the domain does not include z j = c. We then use a coefficient which is zero at z j = c and one otherwise to write the integral over the whole domain of z j .

Similarly, we can write the term in the domain z j = c as DISPLAYFORM3 Now combining the two terms we obtain DISPLAYFORM4 Rearranging to gather the terms in δ(z j − c)/δ(0)

we get DISPLAYFORM5 Simplifying the argument of the second logarithm and solving the second integral we get DISPLAYFORM6 where the second term tends to zero, leaving the cross entropy between f (z j ) and αg(z j ).

Applying this result to 1 and 3 we obtain the following DISPLAYFORM7 The KL divergence D KL N (z i,j ; µ z,i,j , σ 2 z,i,j ) || N (z j ; 0, 1) is analogous to that of the standard VAE and has a simple analytic form BID11 : DISPLAYFORM8 2 and 4 take the form of the cross entropy between a Dirac Delta function and a discrete mixture distribution.

In this case, instead, the continuous density contributions vanish: DISPLAYFORM9 Substituting the results of equations 20, 21 and 22 into equation 13, we obtain the prior term of the VSC lower bound DISPLAYFORM10 Negative Slab KL Divergence DISPLAYFORM11 This prior term presents two components.

The first is the negative KL divergence between the distributions of the Slab variables, multiplied by the probability of z i,j being non-zero γ i,j .

The second term is the negative KL divergence between the distributions of the Spike variables.

We find of particular interest that by computing the KL divergence analytically we recover a linear combination of the Spike and Slab components divergences.

The draws z i,l are computed as follows DISPLAYFORM0 where indicates an element wise product.

The function T (y i,l ) is in principle a step function centered at zero, however, in order to maintain differentiability, we employ a scaled Sigmoid function T (y) = S(cy).

In the limit c → ∞, S(cy) tends to the true binary mapping.

In practice, the value of c needs to be small enough to provide stability of the gradient ascent.

In our implementation we employ a warm-up strategy to gradually increase the value of c during training.

We report here a detailed description of the Spike variable reparametrisation, similar to the relaxation of discrete variables in BID17 and BID27 .

Our aim is to find a function f (η l,j , γ i,j ) such that a binary variable w i,l,j ∼ p(w i,l,j ) drawn from the discrete distribution p(w i,l,j = 1) = γ i,j , p(w i,l,j = 0) = (1 − γ i,j ) can be expressed as w i,l,j = f (η l,j , γ l,j ), where η l,j is some noise variable drawn from a distribution which does not depend on γ i,j .The function of choice f (η l,j , γ i,j ) should ideally only take values 1 and 0, as these are the only values of w i,l,j permitted by p(w i,l,j ).

Furthermore, the probabilities of w i,l,j being 1 or 0 are linear in γ i,j , therefore the distribution of the noise variable η i,j should have evenly distributed mass.

The simplest function which satisfy these conditions and yields our reparametrisation is then DISPLAYFORM0 where η l,j is uniformly distributed and T (y)

is the following step function DISPLAYFORM1 An illustration of this reparametrisation is shown in figure 6 .

P(η l,j ) P(w=0) = (1-γ i,j ) P(w=1) = 1 P(w=1) = γ i,j(1-γ i,j )Figure 6: Schematic representation of the reparametrisation of the Spike variable.

The variable y i,l,j is drawn in the range covered by the grey square with probability proportional to its height.

On the left, for a spike probability γ i,j = 1, the variable y i,l,j is drawn to always be greater than zero and the Spike variable w i,l,j is always one.

On the right, for an arbitrary γ i,j , the probability density of y i,l,j is displaced to the left by 1 − γ i,j and y i,l,j has probability γ i,j of being ≥ 0, in which case w i,l,j is one, and probability 1 − γ i,j of being < 0, in which case w i,l,j is zero.

As described in section 3.2, the function T (y i,l,j ) is not differentiable, therefore we approximate it with a scaled Sigmoid S(cy i,l,j ), where c is a real positive constant.

In our implementation, we gradually increase c from 50 to 200 during training to achieve good approximations without making convergence unstable.

In our experiments, we use VAEs and VSCs having one 400-dimensional hidden layer between the observations and latent variables, both for encoders q φ (z|x) and decoders p θ (x|z).

The only exception is the VSC used to obtain the qualitative results with the CelebA dataset, which was composed of two hidden layers with 2, 000 dimensions between the observations and latent variables.

We trained all auto-encoders with the ADAM optimiser, where the initial training rate was chosen according to best VLB performance of the standard VAE and kept the same for the corresponding VSC we compare to it.

All training rates used were between 0.001 and 0.01.

MNIST and Fashion-MNIST are composed of 28 × 28 grey-scale images of hand-written digits and pieces of clothing respectively.

Both sets contain ten different classes, which is the categories in which we classify in section 4.2.

CelebA is a dataset of 200, 000 examples of colour images of celebrity faces.

We

We measure the latent space posterior sparsity at varying prior sparsity α.

We encode both the MNIST and Fashion-MNIST datasets in 200-dimensional spaces with different values of the prior Spike probability α.

In each case, we measure the aggregate posterior sparsity.

Results are shown in figure 8.

Measured Sparsity Perfect Prior Match Figure 8 : Measured sparsity at varying prior Spike probability α.

For larger values of α the resulting codes retain approximately the sparsity induced by the prior as expected.

At lower values of α the latent codes sparsity increasingly departs from the value induced by the prior.

This is expected since below a certain sparsity value, the recognition function is induced to activate a certain number of latent dimensions in order to satisfy reconstruction.

We report on the ELBO evaluation results.

Next, we show the behaviour of the lower bound at varying prior sparsity α for high dimensional latent spaces.

We encode both the MNIST and Fashion-MNIST datasets in 200-dimensional spaces with different values of α and measure the training and test sets ELBO in each case.

The results are shown in figure 10 .By making the Prior increasingly sparser (i.e. α going from 1 to 0) the ELBO increases thanks to the smaller sub-spaces needed to represent each observation.

At very low α, the lower bound decreases again, as the number of dimensions activated by the recognition function in order to describe the observations is too high to match the prior.

Test Set Figure 10 : Training and test sets ELBO at varying prior Spike probability α.

We show the classification performance at varying prior sparsity α for high dimensional latent spaces and various limits of available number of training examples.

We encode both the MNIST and Fashion-MNIST datasets in 200-dimensional spaces with different values of α and measure the classification accuracy when classifying with a one layer network as described in 4.2.

Figure 11 displays the results.20,000 Labelled Examples 5,000 Labelled Examples 2,000 Labelled Examples 20,000 Labelled Examples 5,000 Labelled Examples 2,000 Labelled Examples Figure 11 : Classification performance at varying prior Spike probability α.

As the prior Spike probability is decreased, the recovered codes are increasingly more efficient.

VAEs are attractive for their ability to produce arguably realistic samples from the prior through ancestral sampling.

Though VSC can be used to perform the same generation task, samples directly from the sparse prior do not give as realistic synthetic observations, as not just any combination of sparse features is a feasible one (see figure 12 ).However, VSC is capable of generating good synthetic samples conditioned on the combination of features identified in a particular observation.

The recognition function from a certain observation defines a sub-space over certain active dimensions.

If we sample from the Gaussian prior only along these dimensions we can generate objects that express variability only in the features recognised in the original observation.

Examples are shown in figure 13 .

Figure 12: Ancestral sampling with a VAE and a VSC trained on the Fashion-MNIST dataset.

The samples generated by the VSC sometimes result into unnatural combinations of features, such as asymmetric t-shirts and bags handles on pieces of clothing.

VSC Conditional Sampling (α = 0.1) Figure 13 : Conditional sampling in a VSC trained on the Fashion-MNIST dataset.

Samples are drawn from the prior Gaussian component, but only along the dimensions activated by the recognition function, which are different for different observations.

As a result, we obtain different subgenerative models that can generate different distinct types of objects in the aggregate used to train the model.

<|TLDR|>

@highlight

We explore the intersection of VAEs and sparse coding.

@highlight

This paper proposes an extension of VAEs with sparse priors and posteriors to learn sparse interpretable representations.