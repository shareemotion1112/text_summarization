Machine learning systems often encounter Out-of-Distribution (OoD) errors when dealing with testing data coming from a different distribution from the one used for training.

With their growing use in critical applications, it becomes important to develop systems that are able to accurately quantify its predictive uncertainty and screen out these anomalous inputs.

However, unlike standard learning tasks, there is currently no well established guiding principle for designing architectures that can accurately quantify uncertainty.

Moreover, commonly used OoD detection approaches are prone to errors and even sometimes assign higher likelihoods to OoD samples.

To address these problems, we first seek to identify guiding principles for designing uncertainty-aware architectures, by proposing Neural Architecture Distribution Search (NADS).

Unlike standard neural architecture search methods which seek for a single best performing architecture, NADS searches for a distribution of architectures that perform well on a given task, allowing us to identify building blocks common among all uncertainty aware architectures.

With this formulation, we are able to optimize a stochastic outlier detection objective and construct an ensemble of models to perform OoD detection.

We perform multiple OoD detection experiments and observe that our NADS performs favorably compared to state-of-the-art OoD detection methods.

Detecting anomalous data is crucial for safely applying machine learning in autonomous systems for critical applications and for AI safety (Amodei et al., 2016) .

Such anomalous data can come in settings such as in autonomous driving NHTSA, 2017) , disease monitoring (Hendrycks & Gimpel, 2016) , and fault detection (Hendrycks et al., 2019b) .

In these situations, it is important for these systems to reliably detect abnormal inputs so that their occurrence can be overseen by a human, or the system can proceed using a more conservative policy.

The widespread use of deep learning models within these autonomous systems have aggravated this issue.

Despite having high performance in many predictive tasks, deep networks tend to give high confidence predictions on Out-of-Distribution (OoD) data (Goodfellow et al., 2015; Nguyen et al., 2015) .

Moreover, commonly used OoD detection approaches are prone to errors and even assign higher likelihoods to samples from other datasets (Lee et al., 2018; Hendrycks & Gimpel, 2016) .

Unlike common machine learning tasks such as image classification, segmentation, and speech recognition, there are currently no well established guidelines for designing architectures that can accurately screen out OoD data and quantify its uncertainty.

Such a gap in our knowledge makes Neural Architecture Search (NAS) a promising option to explore the better design of uncertaintyaware models (Elsken et al., 2018) .

NAS algorithms attempt to find an optimal neural network architecture for a specific task.

Existing efforts have primarily focused on searching for architectures that perform well on image classification or segmentation.

However, it is unclear whether architecture components that are beneficial for image classification and segmentation models would also lead to better uncertainty quantification and thereafter be effective for OoD detection.

Moreover, previous work on deep uncertainty quantification shows that ensembles can help calibrate OoD classifier based methods, as well as improve OoD detection performance of likelihood estimation models (Lakshminarayanan et al., 2017; Choi & Jang, 2018) .

Because of this, instead of a single best performing architecture for uncertainty awareness, one might consider a distribution of wellperforming architectures.

Along this direction, designing an optimization objective which leads to uncertainty-aware models is also not straightforward.

With no access to labels, unsupervised/self-supervised generative models which maximize the likelihood of in-distribution data become the primary tools for uncertainty quantification (Hendrycks et al., 2019a) .

However, these models counter-intuitively assign high likelihoods to OoD data (Nalisnick et al., 2019a; Choi & Jang, 2018; Hendrycks et al., 2019a; Shafaei et al.) .

Because of this, maximizing the log-likelihood is inadequate for OoD detection.

On the other hand, Choi & Jang (2018) proposed using the Widely Applicable Information Criterion (WAIC) (Watanabe, 2013) , a penalized log-likelihood score, as the OoD detection criterion.

However, the score was approximated using an ensemble of models that was trained on maximizing the likelihood and did not directly optimize the WAIC score.

To this end, we propose a novel Neural Architecture Distribution Search (NADS) framework to identify common building blocks that naturally incorporate model uncertainty quantification and compose good OoD detection models.

NADS is an architecture search method designed to search for a distribution of well-performing architectures, instead of a single best architecture by formulating the architecture search problem as a stochastic optimization problem.

Using NADS, we optimize the WAIC score of the architecture distribution, a score that was shown to be robust towards model uncertainty.

Such an optimization problem with a stochastic objective over a probability distribution of architectures is unamenable to traditional NAS optimization strategies.

We make this optimization problem tractable by taking advantage of weight sharing between different architectures, as well as through a parameterization of the architecture distribution, which allows for a continuous relaxation of the discrete search problem.

Using the learned posterior architecture distribution, we construct a Bayesian ensemble of deep models to perform OoD detection.

Finally, we perform multiple OoD detection experiments to show the efficacy of our proposed method.

Neural Architecture Search (NAS) algorithms aim to automatically discover an optimal neural network architecture instead of using a hand-crafted one for a specific task.

Previous work on NAS has achieved successes in image classification (Pham et al., 2018) , image segmentation (Liu et al., 2019) , object detection (Ghiasi et al., 2019) , structured prediction (Chen et al., 2018) , and generative adversarial networks (Gong et al., 2019) .

However, there has been no NAS algorithm developed for uncertainty quantificaton and OoD detection.

NAS consists of three components: the proxy task, the search space, and the optimization algorithm.

Prior work in specifying the search space either searches for an entire architecture directly, or searches for small cells and arrange them in a pre-defined way.

Optimization algorithms that have been used for NAS include reinforcement learning (Baker et al., 2017; Zhong et al., 2018; Zoph & Le, 2016) , Bayesian optimization (Jin et al., 2018) , random search (Chen et al., 2018) , Monte Carlo tree search (Negrinho & Gordon, 2017) , and gradient-based optimization methods (Liu et al., 2018b; Ahmed & Torresani, 2018) .

To efficiently evaluate the performance of discovered architectures and guide the search, the design of the proxy task is critical.

Existing proxy tasks include leveraging shared parameters (Pham et al., 2018) , predicting performance using a surrogate model (Liu et al., 2018a) , and early stopping Chen et al., 2018) .

To our best knowledge, all existing NAS algorithms seek a single best performing architecture.

In comparison, searching for a distribution of architectures allows us to analyze the common building blocks that all of the candidate architectures have.

Moreover, this technique can also complement ensemble methods by creating a more diverse set of models for the ensemble decision, an important ingredient for deep uncertainty quantification (Lakshminarayanan et al., 2017) .

Prior work on uncertainty quantification and OoD detection for deep models can be divided into model-dependent (Lakshminarayanan et al., 2017; Gal & Ghahramani, 2016; Liang et al., 2017) , and model-independent techniques (Dinh et al., 2016; Germain et al., 2015; Oord et al., 2016) .

Model-dependent techniques aim to yield confidence measures p(y|x) for a model's prediction y when given input data x. However, a limitation of model-dependent OoD detection is that they may discard information regarding the data distribution p(x) when learning the task specific model p(y|x).

This could happen when certain features of the data are irrelevant for the predictive task, causing information loss regarding the data distribution p(x).

Moreover, existing methods to calibrate model uncertainty estimates assume access to OoD data during training (Lee et al., 2018; Hendrycks et al., 2019b) .

Although the OoD data may not come from the testing distribution, this assumes that the structure of OoD data is known ahead of time, which can be incorrect in settings such as active/online learning where new training distributions are regularly encountered.

On the other hand, model-independent techniques seek to estimate the likelihood of the data distribution p(x).

These techniques include Variational Autoencoders (VAEs) (Kingma & Welling, 2013) , generative adversarial networks (GANs) (Goodfellow et al., 2014) , autoregressive models (Germain et al., 2015; Oord et al., 2016) , and invertible flow-based models (Dinh et al., 2016; Kingma & Dhariwal, 2018) .

Among these techniques, invertible models offer exact computation of the data likelihood, making them attractive for likelihood estimation.

Moreover, they do not require OoD samples during training, making them applicable to any OoD detection scenario.

Thus in this paper, we focus on searching for invertible flow-based architectures, though the presented techniques are also applicable to other likelihood estimation models.

Along this direction, recent work has discovered that likelihood-based models can assign higher likelihoods to OoD data compared to in-distribution data (Nalisnick et al., 2019a; Choi & Jang, 2018 ) (see Figure 13 for an example).

One hypothesis for such a phenomenon is that most data points lie within the typical set of a distribution, instead of the region of high likelihood (Nalisnick et al., 2019b) .

Thus, Nalisnick et al. (2019b) recommend to estimate the entropy using multiple data samples to screen out OoD data instead of using the likelihood.

Other uncertainty quantification formulations can also be related to entropy estimation (Choi & Jang, 2018; Lakshminarayanan et al., 2017) .

However, it is not always realistic to test multiple data points in practical data streams, as testing data often come one sample at a time and are never well-organized into in-distribution or out-of-distribution groups.

With this in mind, model ensembling becomes a natural consideration to formulate entropy estimation.

Instead of averaging the entropy over multiple data points, model ensembles produce multiple estimates of the data likelihood, thus "augmenting" one data point into as many data points as needed to reliably estimate the entropy.

However, care must be taken to ensure that the model ensemble produces likelihood estimates that agree with one another on in-distribution data, while also being diverse enough to discriminate OoD data likelihoods.

In what follows, we propose NADS as a method that can identify distributions of architectures for uncertainty quantification.

Using a loss function that accounts for the diversity of architectures within the distribution, NADS allows us to construct an ensemble of models that can reliably detect OoD data.

Putting Neural Architecture Distribution Search (NADS) under a common NAS framework (Elsken et al., 2018) , we break down our search formulation into three main components: the proxy task, the search space, and the optimization method.

Specifying these components for NADS with the ultimate goal of uncertainty quantification for OoD detection is not immediately obvious.

For example, naively using data likelihood maximization as a proxy task would run into the issue pointed out by Nalisnick et al. (2019a) , with models assigning higher likelihoods to OoD data.

On the other hand, the search space needs to be large enough to include a diverse range of architectures, yet still allowing a search algorithm to traverse it efficiently.

In the following sections, we motivate our decision on these three choices and describe these components for NADS in detail.

The first component of NADS is the training objective that guides the neural architecture search.

Different from existing NAS methods, our aim is to derive an ensemble of deep models to improve model uncertainty quantification and OoD detection.

To this end, instead of searching for architectures which maximize the likelihood of in-distribution data, which may cause our model to incorrectly assign high likelihoods to OoD data, we instead seek architectures that can perform entropy estimation by maximizing the Widely Applicable Information Criteria (WAIC) of the training data.

The WAIC score is a Bayesian adjusted metric to calculate the marginal likelihood (Watanabe, 2013) .

This metric has been shown by Choi & Jang (2018) to be robust towards the pitfall causing likelihood estimation models to assign high likelihoods to OoD data.

The score is defined as follows:

Here, E[·] and V[·] denote expectation and variance respectively, which are taken over all architectures α sampled from the posterior architecture distribution p(α).

Such a strategy captures model uncertainty in a Bayesian fashion, improving OoD detection.

Intuitively, minimizing the variance of training data likelihoods allows its likelihood distribution to remain tight which, by proxy, minimizes the overlap of in-distribution and out-of-distribution likelihoods, thus making them separable.

Under this objective function, we search for an optimal distribution of network architectures p(α) by deriving the corresponding parameters that characterize p(α).

Because the score requires aggregating the results from multiple architectures α, optimizing such a score using existing search methods can be intractable, as they typically only consider a single architecture at a time.

Later, we will show how to circumvent this problem in our optimization formulation.

NADS constructs a layer-wise search space with a pre-defined macro-architecture, where each layer can have a different architecture component.

Such a search space has been studied by (Zoph & Le, 2016; Liu et al., 2018b; Real et al., 2019) , where it shows to be both expressive and scalable/efficient.

The macro-architecture closely follows the Glow architecture presented in Kingma & Dhariwal (2018) .

Here, each layer consists of an actnorm, an invertible 1 × 1 convolution, and an affine coupling layer.

Instead of pre-defining the affine coupling layer, we allow it to be optimized by our architecture search.

The search space can be viewed in Figure 1 .

Here, each operational block of the affine coupling layer is selected from a list of candidate operations that include 3 × 3 average pooling, 3 × 3 max pooling, skip-connections, 3 × 3 and 5 × 5 separable convolutions, 3 × 3 and 5 × 5 dilated convolutions, identity, and zero.

We choose this search space to answer the following questions towards better architectures for OoD detection:

• What topology of connections between layers is best for uncertainty quantification?

Traditional likelihood estimation architectures focus only on feedforward connections without adding any skip-connection structures.

However, adding skip-connections may improve optimization speed and stability.

• Are more features/filters better for OoD detection?

More feature outputs of each layer should lead to a more expressive model.

However, if many of those features are redundant, it may slow down learning, overfitting nuisances and resulting in sub-optimal models.

• Which operations are best for OoD detection?

Intuitively, operations such as max/average pooling should not be preferred, as they discard information of the original data point "too aggressively".

However, this intuition remains to be confirmed.

Having specified our proxy task and search space, we now describe our optimization method for NADS.

Several difficulties arise when attempting to optimize this setup.

First, optimizing p(α), a distribution over high-dimensional discrete random variables α, jointly with the network parameters is intractable as, at worst, each network's optimal parameters would need to be individually identified.

Second, even if we relax the discrete search space, the objective function involves computing an expectation and variance over all possible discrete architectures.

To alleviate these problems, we first introduce a continuous relaxation for the discrete search space, allowing us to approximately optimize the discrete architectures through backpropagation and weight sharing between common architecture blocks.

We then approximate the stochastic objective by using Monte Carlo samples to estimate the expectation and variance.

Specifically, let A denote our discrete architecture search space and α ∈ A be an architecture in this space.

Let l θ * (α) be the loss function of architecture α with its parameters set to θ * such that it satisfies θ * = arg min θ l(θ|α) for some loss function l(·).

We are interested in finding a distribution p φ (α) parameterized by φ that minimizes the expected loss of an architecture α sampled from it.

We denote this loss function as

Solving L(φ) for arbitrary parameterizations of p φ (α) can be intractable, as the inner loss function l θ * (α) involves searching for the optimal parameters θ * of a neural network architecture α.

Moreover, the outer expectation causes backpropagation to be inapplicable due to the discrete random architecture variable α.

We adopt a tractable optimization paradigm to circumvent this problem through a specific reparameterization of the architecture distribution p φ (α), allowing us to backpropagate through the outer expectation and jointly optimize φ and θ.

For clarity of exposition, we first focus on sampling an architecture with a single hidden layer.

In this setting, we intend to find a probability vector φ = [φ 1 , . . .

, φ K ] with which we randomly pick a single operation from a list of

the random categorical indicator vector sampled from φ, where b i is 1 if the i th operation is chosen, and zero otherwise.

Note that b is equivalent to the discrete architecture variable α in this setting.

With this, we can write the random output y of the hidden layer given input x as

To make optimization tractable, we relax the discrete mask b to be a continuous random variableb using the Gumbel-Softmax reparameterization (Gumbel, 1954; Maddison et al., 2014) as follows:

Here, g 1 . . .

g k ∼ − log(− log(u)) where u ∼ Unif(0, 1), and τ is a temperature parameter.

For low values of τ ,b approaches a sample of a categorical random variable, recovering the original discrete problem.

While for high values,b will equally weigh the K operations (Jang et al., 2016) .

Using this, we can compute backpropagation by approximating the gradient of the discrete architecture α with the gradient of the continuously relaxed categorical random variableb, as

With this backpropagation gradient defined, generalizing the above setting to architectures with multiple layers simply involves recursively applying the above gradient relaxation to each layer.

With this formulation, we can gradually remove the continuous relaxation and sample discrete architectures by annealing the temperature parameter τ .

With this, we are able to optimize the architecture distribution p φ (α) and sample candidate architectures for further retraining, finetuning, or evaluation.

By sampling M architectures from the distribution, we are able to approximate the WAIC score expectation and variance terms as: Figure 2 : Summary of our architecture search findings: the most likely architecture structure for each block K found by NADS.

We applied our architecture search on five datasets: CelebA (Liu et al.) , CIFAR-10, CIFAR-100, (Krizhevsky et al., 2009) , SVHN (Netzer et al., 2011) , and MNIST (LeCun).

In all experiments, we used the Adam optimizer with a fixed learning rate of 1 × 10 −5 with a batch size of 4 for 10000 iterations.

We approximate the WAIC score using M = 4 architecture samples, and set the temperature parameter τ = 1.5 .

The number of layers and latent dimensions is the same as in the original Glow architecture (Kingma & Dhariwal, 2018) , with 4 blocks and 32 flows per block.

Images were resized to 64 × 64 as inputs to the model.

With this setup, we found that we are able to identify neural architectures in less than 1 GPU day.

Our findings are summarized in Figure 2 , while more samples from our architecture search can be seen in Appendix C. Observing the most likely architecture components found on all of the datasets, a number of notable observations can be made:

• The first few layers have a simple feedforward structure, with either only a few convolutional operations or average pooling operations.

On the other hand, more complicated structures with skip connections are preferred in the deeper layers of the network.

We hypothesize that in the first few layers, simple feature extractors are sufficient to represent the data well.

Indeed, recent work on analyzing neural networks for image data have shown that the first few layers have filters that are very similar to SIFT features or wavelet bases (Zeiler & Fergus, 2014; Lowe, 1999 ).

•

The max pooling operation is almost never selected by the architecture search.

This confirms our hypothesis that operations that discard information about the data is unsuitable for OoD detection.

However, to our surprise, average pooling is preferred in the first layers of the network.

We hypothesize that average pooling has a less severe effect in discarding information, as it can be thought of as a convolutional filter with uniform weights.

• The deeper layers prefer a more complicated structure, with some components recovering the skip connection structure of ResNets (He et al., 2016) .

We hypothesize that deeper layers may require more skip connections in order to feed a strong signal for the first few layers.

This increases the speed and stability of training.

Moreover, a larger number of features can be extracted using the more complicated architecture.

Interestingly enough, we found that the architectures that we sample from our NADS perform well in image generation without further retraining, as shown in Appendix D.

Using the architectures sampled from our search, we create a Bayesian ensemble of models to estimate the WAIC score.

Each model of our ensemble is weighted according to its probability, as in Hoeting et al. (1999) .

The log-likelihood estimate as well as the variance of this model ensemble Here, 'Baseline' denotes the method proposed by Hendrycks & Gimpel (2016) .

Subcaptions denote training-testing set pairs.

Additional figures are provided in Appendix G. is given as follows:

(6) Intuitively, we are weighing each member of the ensemble by their posterior architecture distribution p φ (α), a measure of how likely each architecture is in optimizing the WAIC score.

We note that for our setup, V[log p αi (x)] is zero for each model in our ensemble; however, for models which do have variance estimates, such as models that incorporate variational dropout Kingma et al., 2015; Gal & Ghahramani, 2016) , this term may be nonzero.

Using these estimates, we are able to approximate the WAIC score in equation (1).

We trained our proposed method on 4 datasets: CIFAR-10, CIFAR-100 (Krizhevsky et al., 2009) , SVHN (Netzer et al., 2011), and MNIST (LeCun) .

In all experiments, we randomly sampled an (Hendrycks & Gimpel, 2016) , and Outlier Exposure (OE) (Hendrycks et al., 2019b ensemble of M = 5 models from the posterior architecture distribution p φ * (α) found by NADS.

Although these models can sufficiently perform image synthesis without retraining as shown in Appendix D, we observed that further retraining these architectures led to a significant improvement in OoD detection.

Because of this, we retrained each architecture on data likelihood maximization for 150000 iterations using Adam with a learning rate of 1 × 10 −5 .

We first show the effects of increasing the ensemble size in Figure 3 and Appendix F. Here, we can see that increasing the ensemble size causes the OoD WAIC scores to decrease as their corresponding histograms shift away from the training data WAIC scores, thus improving OoD detection performance.

Next, we compare our ensemble search method against a traditional ensembling method that uses a single Glow architecture trained with multiple random initializations.

As shown in Table 2 , we find that our method is superior compared to the traditional ensembling method when compared on OoD detection using CIFAR-10 as the training distribution.

We then compared our NADS ensemble OoD detection method for screening out samples from datasets that the original model was not trained on.

For SVHN, we used the Texture, Places, LSUN, and CIFAR-10 as the OoD dataset.

For CIFAR-10 and CIFAR-100, we used the SVHN, Texture, Places, LSUN, CIFAR-100 (CIFAR-10 for CIFAR-100) datasets, as well as the Gaussian and Rademacher distributions as the OoD dataset.

Finally, for MNIST, we used the not-MNIST, F-MNIST, and K-MNIST datasets.

We compared our method against a baseline method that uses maximum softmax probability (MSP) (Hendrycks & Gimpel, 2016) , as well as two popular OoD detection methods: ODIN (Liang et al., 2017) and Outlier Exposure (OE) (Hendrycks et al., 2019b) .

ODIN attempts to calibrate the uncertainty estimates of an existing model by reweighing its output softmax score using a temperature parameter and through random perturbations of the input data.

For this, we use DenseNet as the base model as described in (Liang et al., 2017) .

On the other hand, OE models are trained to minimize a loss regularized by an outlier exposure loss term, a loss term that requires access to OoD samples.

As shown in Table 1 and Table 3 , our method outperforms the baseline MSP and ODIN significantly while performing better or comparably with OE, which requires OoD data during training, albeit not from the testing distribution.

We plot Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves in Figure 4 and Appendix G for more comprehensive comparison.

In particular, our method consistently achieves high area under PR curve (AUPR%), showing that we are especially capable of screening out OoD data in settings where their occurrence is rare.

Such a feature is important in situations where anomalies are sparse, yet have disastrous consequences.

Notably, ODIN underperforms in screening out many OoD datasets, despite being able to reach the original reported performance when testing on LSUN using a CIFAR10 trained model.

This suggests that ODIN may not be stable for use on different anomalous distributions.

Unlike NAS for common learning tasks, specifying a model and an objective to optimize for uncertainty estimation and outlier detection is not straightforward.

Moreover, using a single model may not be sufficient to accurately quantify uncertainty and successfully screen out OoD data.

We developed a novel neural architecture distribution search (NADS) formulation to identify a random ensemble of architectures that perform well on a given task.

Instead of seeking to maximize the likelihood of in-distribution data which may cause OoD samples to be mistakenly given a higher likelihood, we developed a search algorithm to optimize the WAIC score, a Bayesian adjusted estimation of the data entropy.

Using this formulation, we have identified several key features that make up good uncertainty quantification architectures, namely a simple structure in the shallower layers, use of information preserving operations, and a larger, more expressive structure with skip connections for deeper layers to ensure optimization stability.

Using the architecture distribution learned by NADS, we then constructed an ensemble of models to estimate the data entropy using the WAIC score.

We demonstrated the superiority of our method to existing OoD detection methods and showed that our method has highly competitive performance without requiring access to OoD samples.

Overall, NADS as a new uncertainty-aware architecture search strategy enables model uncertainty quantification that is critical for more robust and generalizable deep learning, a crucial step in safely applying deep learning to healthcare, autonomous driving, and disaster response.

A FIXED MODEL ABLATION STUDY

@highlight

We propose an architecture search method to identify a distribution of architectures and use it to construct a Bayesian ensemble for outlier detection.