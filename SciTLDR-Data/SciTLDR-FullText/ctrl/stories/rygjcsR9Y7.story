High-dimensional time series are common in many domains.

Since human cognition is not optimized to work well in high-dimensional spaces, these areas could benefit from interpretable low-dimensional representations.

However, most representation learning algorithms for time series data are difficult to interpret.

This is due to non-intuitive mappings from data features to salient properties of the representation and non-smoothness over time.

To address this problem, we propose a new representation learning framework building on ideas from interpretable discrete dimensionality reduction and deep generative modeling.

This framework allows us to learn discrete representations of time series, which give rise to smooth and interpretable embeddings with superior clustering performance.

We introduce a new way to overcome the non-differentiability in discrete representation learning and present a gradient-based version of the traditional self-organizing map algorithm that is more performant than the original.

Furthermore, to allow for a probabilistic interpretation of our method, we integrate a Markov model in the representation space.

This model uncovers the temporal transition structure, improves clustering performance even further and provides additional explanatory insights as well as a natural representation of uncertainty.

We evaluate our model in terms of clustering performance and interpretability on static (Fashion-)MNIST data, a time series of linearly interpolated (Fashion-)MNIST images, a chaotic Lorenz attractor system with two macro states, as well as on a challenging real world medical time series application on the eICU data set.

Our learned representations compare favorably with competitor methods and facilitate downstream tasks on the real world data.

Interpretable representation learning on time series is a seminal problem for uncovering the latent structure in complex systems, such as chaotic dynamical systems or medical time series.

In areas where humans have to make decisions based on large amounts of data, interpretability is fundamental to ease the human task.

Especially when decisions have to be made in a timely manner and rely on observing some chaotic external process over time, such as in finance or medicine, the need for intuitive interpretations is even stronger.

However, many unsupervised methods, such as clustering, make misleading i.i.d.

assumptions about the data, neglecting their rich temporal structure and smooth behaviour over time.

This poses the need for a method of clustering, where the clusters assume a topological structure in a lower dimensional space, such that the representations of the time series retain their smoothness in that space.

In this work, we present a method with these properties.

We choose to employ deep neural networks, because they have a very successful tradition in representation learning BID5 .

In recent years, they have increasingly been combined with generative modeling through the advent of generative adversarial networks (GANs) BID13 and variational autoencoders (VAEs) BID18 .

However, the representations learned by these models are often considered cryptic and do not offer the necessary interpretability .

A lot of work has been done to improve them in this regard, in GANs as well as VAEs BID16 BID9 .

Alas, these works have focused entirely on continuous representations, while discrete ones are still underexplored.

In order to define temporal smoothness in a discrete representation space, the space has to be equipped with a topological neighborhood relationship.

One type of representation space with such a structure is induced by the self-organizing map (SOM) BID21 .

The SOM allows to map states from an uninterpretable continuous space to a lower-dimensional space with a predefined topologically interpretable structure, such as an easily visualizable two-dimensional grid.

However, while yielding promising results in visualizing static state spaces, such as static patient states BID27 , the classical SOM formulation does not offer a notion of time.

The time component can be incorporated using a probabilistic transition model, e.g. a Markov model, such that the representations of a single time point are enriched with information from the adjacent time points in the series.

It is therefore potentially fruitful to apply the approaches of probabilistic modeling alongside representation learning and discrete dimensionality reduction in an end-to-end model.

In this work, we propose a novel deep architecture that learns topologically interpretable discrete representations in a probabilistic fashion.

Moreover, we introduce a new method to overcome the non-differentiability in discrete representation learning architectures and develop a gradient-based version of the classical selforganizing map algorithm with improved performance.

We present extensive empirical evidence for the model's performance on synthetic and real world time series from benchmark data sets, a synthetic dynamical system with chaotic behavior and real world medical data.

• Devise a novel framework for interpretable discrete representation learning on time series.• Show that the latent probabilistic model in the representation learning architecture improves clustering and interpretability of the representations on time series.• Show superior clustering performance of the model on benchmark data and a real world medical data set, on which it also facilitates downstream tasks.

Our proposed model combines ideas from self-organizing maps BID21 , variational autoencoders BID18 ) and probabilistic models.

In the following, we will lay out the different components of the model and their interactions.

A schematic overview of our proposed model is depicted in FIG0 .

An input x ∈ R d is mapped to a latent encoding z e ∈ R m (usually m < d) by computing z e = f θ (x), where f θ (·) is parameterized by the encoder neural network.

The encoding is then assigned to an embedding z q ∈ R m in the dictionary of embeddings E = {e 1 , . . .

, e k | e i ∈ R m } by sampling z q ∼ p(z q |z e ).

The form of this distribution is flexible and can be a design choice.

In order for the model to behave similarly to the original SOM algorithm (see below), in our experiments we choose the distribution to be categorical with probability mass 1 on the closest embedding to z e , i.e. p(z q |z e ) = 1[z q = arg min e∈E z e − e 2 ], where 1[·] is the indicator function.

A reconstructionx of the input can then be computed asx = g φ (z), where g φ (·) is parameterized by the decoder neural network.

Since the encodings and embeddings live in the same space, one can compute two different reconstructions, namelyx e = g φ (z e ) andx q = g φ (z q ).To achieve a topologically interpretable neighborhood structure, the embeddings are connected to form a self-organizing map.

A self-organizing map consists of k nodes V = {v 1 , . . .

, v k }, where every node corresponds to an embedding in the data space e v ∈ R d and a representation in a lower-dimensional discrete space m v ∈ M , where usually M ⊂ N 2 .

During training on a data set D = {x 1 , . . .

, x n }, a winner nodẽ v is chosen for every point x i according toṽ = arg min v∈V e v − x i 2 .

The embedding vector for every [red] .

In order to achieve a discrete representation, every latent data point (z e ) is mapped to its closest node in the SOM (z q ).

A Markov transition model [blue] is learned to predict the next discrete representation (z t+1 q ) given the current one (z t q ).

The discrete representations can then be decoded by another neural network back into the original data space.

node u ∈ V is then updated according to e u ← e u + N (m u , mṽ)η(x i − e u ), where η is the learning rate and N (m u , mṽ) is a neighborhood function between the nodes defined on the representation space M .

There can be different design choices for N (m u , mṽ).

A more thorough review of the self-organizing map algorithm is deferred to the appendix (Sec. A).We choose to use a two-dimensional SOM because it facilitates visualization similar to BID27 .

Since we want the architecture to be trainable end-to-end, we cannot use the standard SOM training algorithm described above.

Instead, we devise a loss function term whose gradient corresponds to a weighted version of the original SOM update rule (see below).

We implement it in such a way that any time an embedding e i,j at position (i, j) in the map gets updated, it also updates all the embeddings in its immediate neighborhood N (e i,j ).

The neighborhood is defined as N (e i,j ) = {e i−1,j , e i+1,j , e i,j−1 , e i,j+1 } for a two-dimensional map.

The loss function for a single input x looks like DISPLAYFORM0 where x, z e , z q ,x e andx q are defined as above and α and β are weighting hyperparameters.

Every term in this function is specifically designed to optimize a different model component.

The first term is the reconstruction loss L reconstruction (x,x q ,x e ) = x−x q 2 + x−x e 2 .

The first subterm of this is the discrete reconstruction loss, which encourages the assigned SOM node z q (x) to be an informative representation of the input.

The second subterm encourages the encoding z e (x) to also be an informative representation.

This ensures that all parts of the model have a fully differentiable credit assignment path to the loss function, which facilitates training.

Note that the reconstruction loss corresponds to the evidence lower bound (ELBO) of the VAE part of our model BID18 .

Since we assume a uniform prior over z q , the KL-term in the ELBO is constant w.r.t.

the parameters and can be ignored during optimization.

The term L commitment encourages the encodings and assigned SOM nodes to be close to each other and is defined as DISPLAYFORM1 2 .

Closeness of encodings and embeddings should be expected to already follow from the L reconstruction term in a fully differentiable architecture.

However, due to the nondifferentiability of the embedding assignment in our model, the L commitment term has to be explicitly added to the objective in order for the encoder to get gradient information about z q .

DISPLAYFORM2 2 , where N (·) is the set of neighbors in the discrete space as defined above and sg [·] is the gradient stopping operator that does not change the outputs during the forward pass, but sets the gradients to 0 during the backward pass.

It encourages the neighbors of the assigned SOM node z q to also be close to z e , thus enabling the embeddings to exhibit a self-organizing map property, while stopping the gradients on z e such that the encoding is not pulled in the direction of the neighbors.

This term enforces a neighborhood relation between the discrete codes and encourages all SOM nodes to ultimately receive gradient information from the data.

The gradient stopping in this term is motivated by the observation that the data points themselves do not get moved in the direction of their assigned SOM node's neighbors in the original SOM algorithm either (see above).

We want to optimize the embeddings based on their neighbors, but not the respective encodings, since any single encoding should be as close as possible to its assigned embedding and not receive gradient information from any other embeddings that it is not assigned to.

Note that the gradient update of a specific SOM node in this formulation depends on its distance to the encoding, while the step size in the original SOM algorithm is constant.

It will be seen that this offers some benefits in terms of optimization and convergence (see Sec. 4.1).

The main challenge in optimizing our architecture is the non-differentiability of the discrete cluster assignment step.

Due to this, the gradients from the reconstruction loss cannot flow back into the encoder.

A model with a similar problem is the recently proposed vector-quantized VAE (VQ-VAE) BID29 .

It can be seen as being similar to a special case of our SOM-VAE model, where one sets β = 0, i.e. disables the SOM structure.

In order to mitigate the non-differentiability, the authors of the VQ-VAE propose to copy the gradients from z q to z e .

They acknowledge that this is an ad hoc approximation, but observed that it works well in their experiments.

Due to our smaller number of embeddings compared to the VQ-VAE setup, the average distance between an encoding and its closest embedding is much larger in our case.

The gradient copying (see above) thus ceases to be a feasible approximation, because the true gradients at points in the latent space which are farther apart will likely be very different.

In order to still overcome the non-differentiability issue, we propose to add the second reconstruction subterm to L reconstruction , where the reconstructionx e is decoded directly from the encoding z e .

This adds a fully differentiable credit assignment path from the loss to the encoder and encourages z e to also be an informative representation of the input, which is a desirable model feature.

Most importantly, it works well in practice (see Sec. 4.1).Note that since z e is continuous and therefore much less constrained than z q , this term is optimized easily and becomes small early in training.

After that, mostly the z q -term contributes to L reconstruction .

One could therefore view the z e -term as an initial encouragement to place the data encodings at sensible positions in the latent space, after which the actual clustering task dominates the training objective.

Our ultimate goal is to predict the development of time series in an interpretable way.

This means that not only the state representations should be interpretable, but so should be the prediction as well.

To this end, we use a temporal probabilistic model.

Learning a probabilistic model in a high-dimensional continuous space can be challenging.

Thus, we exploit the low-dimensional discrete space induced by our SOM to learn a temporal model.

For that, we define a system state as the assigned node in the SOM and then learn a Markov model for the transitions between those states.

The model is learned jointly with the SOM-VAE, where the loss function becomes DISPLAYFORM0 with weighting hyperparameters γ and τ .The term L transitions encourages the probabilities of actually observed transitions to be high.

It is defined as DISPLAYFORM1 ) being the probability of a transition from state z q ( DISPLAYFORM2 The term L smoothness encourages the probabilities for transitions to nodes that are far away from the current data point to be low or respectively the nodes with high transition probabilities to be proximal.

It achieves this by taking large values only for transitions to far away nodes that have a high probability under the model.

It is defined as L smoothness ( DISPLAYFORM3 2 .

The probabilistic model can inform the evolution of the SOM through this term which encodes our prior belief that transitions in natural data happen smoothly and that future time points will therefore mostly be found in the neighborhood of previous ones.

In a setting where the data measurements are noisy, this improves the clustering by acting as a temporal smoother.

From the early inception of the k-means algorithm for clustering BID24 , there has been much methodological improvement on this unsupervised task.

This includes methods that perform clustering in the latent space of (variational) autoencoders BID1 or use a mixture of autoencoders for the clustering BID32 Locatello et al., 2018) .

The method most related to our work is the VQ-VAE (van den BID29 , which can be seen as a special case of our framework (see above).

Its authors have put a stronger focus on the discrete representation as a form of compression instead of clustering.

Hence, our model and theirs differ in certain implementation considerations (see Sec. 2.2).

All these methods have in common that they only yield a single number as a cluster assignment and provide no interpretable structure of relationships between clusters.

The self-organizing map (SOM) BID21 , however, is an algorithm that provides such an interpretable structure.

It maps the data manifold to a lower-dimensional discrete space, which can be easily visualized in the 2D case.

It has been extended to model dynamical systems BID4 and combined with probabilistic models for time series BID25 , although without using learned representations.

There are approaches to turn the SOM into a "deeper" model BID8 , combine it with multi-layer perceptrons BID11 or with metric learning (Płoński and Zaremba, 2012).

However, it has (to the best of our knowledge) not been proposed to use SOMs in the latent space of (variational) autoencoders or any other form of unsupervised deep learning model.

Interpretable models for clustering and temporal predictions are especially crucial in fields where humans have to take responsibility for the model's predictions, such as in health care.

The prediction of a patient's future state is an important problem, particularly on the intensive care unit (ICU) BID15 BID3 .

Probabilistic models, such as Gaussian processes, have been successfully applied in this domain BID7 BID26 .

Recently, deep generative models have been proposed BID10 , sometimes even in combination with probabilistic modeling BID23 .

To the best of our knowledge, SOMs have only been used to learn interpretable static representations of patients BID27 , but not dynamic ones.

We performed experiments on MNIST handwritten digits BID22 , Fashion-MNIST images of clothing BID31 , synthetic time series of linear interpolations of those images, time series from a chaotic dynamical system and real world medical data from the eICU Collaborative Research Database BID12 .

If not otherwise noted, we use the same architecture for all experiments, sometimes including the latent probabilistic model (SOM-VAE_prob) and sometimes excluding it (SOM-VAE).

For model implementation details, we refer to the appendix (Sec. B) 1 .We found that our method achieves a superior clustering performance compared to other methods.

We also show that we can learn a temporal probabilistic model concurrently with the clustering, which is on par with the maximum likelihood solution, while improving the clustering performance.

Moreover, we can learn interpretable state representations of a chaotic dynamical system and discover patterns in real medical data.

In order to test the clustering component of the SOM-VAE, we performed experiments on MNIST and Fashion-MNIST.

We compare our model (including different adjustments to the loss function) against k-means (Lloyd, 1982) (sklearn-package (Pedregosa et al., 2011)), the VQ-VAE (van den Oord et al., 2017), a standard implementation of a SOM (minisom-package BID30 ) and our version of a GB-SOM (gradient-based SOM), which is a SOM-VAE where the encoder and decoder are set to be identity functions.

The k-means algorithm was initialized using k-means++ BID2 .

To ensure comparability of the performance measures, we used the same number of clusters (i.e. the same k) for all the methods.

The results of the experiment in terms of purity and normalized mutual information (NMI) are shown in Table 1 .

The SOM-VAE outperforms the other methods w.r.t.

the clustering performance measures.

It should be noted here that while k-means is a strong baseline, it is not density matching, i.e. the density of cluster centers is not proportional to the density of data points.

Hence, the representation of data in a space induced by the k-means clusters can be misleading.

As argued in the appendix (Sec. C), NMI is a more balanced measure for clustering performance than purity.

If one uses 512 embeddings in the SOM, one gets a lower NMI due to the penalty term for the number of FIG1 : Images generated from a section of the SOM-VAE's latent space with 512 embeddings trained on MNIST.

It yields a discrete two-dimensional representation of the data manifold in the higher-dimensional latent space.

clusters, but it yields an interpretable two-dimensional representation of the manifolds of MNIST FIG1 , Supp.

FIG3 ) and Fashion-MNIST (Supp.

Fig. S5 ).The experiment shows that the SOM in our architecture improves the clustering (SOM-VAE vs. VQ-VAE) and that the VAE does so as well (SOM-VAE vs. GB-SOM).

Both parts of the model therefore seem to be beneficial for our task.

It also becomes apparent that our reconstruction loss term on z e works better in practice than the gradient copying trick from the VQ-VAE (SOM-VAE vs. gradcopy), due to the reasons described in Section 2.2.

If one removes the z e reconstruction loss and does not copy the gradients, the encoder network does not receive any gradient information any more and the learning fails completely (no_grads).

Another interesting observation is that stochastically optimizing our SOM loss using Adam (Kingma and Ba, 2014) seems to discover a more performant solution than the classical SOM algorithm (GB-SOM vs. minisom).

This could be due to the dependency of the step size on the distance between embeddings and encodings, as described in Section 2.1.

Since k-means seems to be the strongest competitor, we are including it as a reference baseline in the following experiments as well.

In order to test the probabilistic model in our architecture and its effect on the clustering, we generated synthetic time series data sets of (Fashion-)MNIST images being linearly interpolated into each other.

Each time series consists of 64 frames, starting with one image from (Fashion-)MNIST and smoothly changing sequentially into four other images over the length of the time course.

After training the model on these data, we constructed the maximum likelihood estimate (MLE) for the Markov model's transition matrix by fixing all the weights in the SOM-VAE and making another pass over the training set, counting all the observed transitions.

This MLE transition matrix reaches a negative log likelihood of 0.24, while our transition matrix, which is learned concurrently with the architecture, yields 0.25.

Our model is therefore on par with the MLE solution.

Comparing these results with the clustering performance on the standard MNIST and Fashion-MNIST test sets, we observe that the performance in terms of NMI is not impaired by the inclusion of the probabilistic model into the architecture (Tab.

2).

On the contrary, the probabilistic model even slightly increases the performance on Fashion-MNIST.

Note that we are using 64 embeddings in this experiment instead of 16, leading to a higher clustering performance in terms of purity, but a slightly lower performance in terms of NMI compared to Table 1 .

This shows again that the measure of purity has to be interpreted with care when comparing This experiment shows that we can indeed fit a valid probabilistic transition model concurrently with the SOM-VAE training, while at the same time not hurting the clustering performance.

It also shows that for certain types of data the clustering performance can even be improved by the probabilistic model (see Sec. 2.3).

In order to assess whether our model can learn an interpretable representation of more realistic chaotic time series, we train it on synthetic trajectories simulated from the famous Lorenz system (Lorenz, 1963) .

The Lorenz system is a good example for this assessment, since it offers two well defined macro-states (given by the attractor basins) which are occluded by some chaotic noise in the form of periodic fluctuations around the attractors.

A good interpretable representation should therefore learn to largely ignore the noise and model the changes between attractor basins.

For a review of the Lorenz system and details about the simulations and the performance measure, we refer to the appendix (Sec. D.2).In order to compare the interpretability of the learned representations, we computed entropy distributions over simulated subtrajectories in the real system space, the attractor assignment space and the representation spaces for k-means and our model.

The computed entropy distributions over all subtrajectories in the test set are depicted in FIG2 .

The experiment shows that the SOM-VAE representations FIG2 are much closer in entropy to the groundtruth attractor basin assignments FIG2 than the k-means representations FIG2 .

For most of the subtrajectories without attractor basin change they assign a very low entropy, effectively ignoring the noise, while the k-means representations partially assign very high entropies to those trajectories.

In total, the k-means representations' entropy distribution is similar to the entropy distribution in the noisy system space FIG2 .

The representations learned by the SOM-VAE are therefore more interpretable than the k-means representations with regard to this interpretability measure.

As could be expected from these figures, the SOM-VAE representation is also superior to the k-means one in terms of purity with respect to the attractor assignment (0.979 vs. 0.956) as well as NMI (0.577 vs. 0.249).Finally, we use the learned probabilistic model on our SOM-VAE representations to sample new latent system trajectories and compute their entropies.

The distribution looks qualitatively similar to the one over real Table 3 : Performance comparison of our method with and without probabilistic model (SOM-VAE-prob and SOM-VAE) against k-means in terms of normalized mutual information on a challenging unsupervised prediction task on real eICU data.

The dynamic endpoints are the maximum of the physiology score within the next 6, 12 or 24 hours (physiology_6_hours, physiology_12_hours, physiology_24_hours).

The values are the means of 10 runs and the respective standard errors.

Each method is used to fit 64 embeddings/clusters. .

It can be seen that our model is the only one that learns a topologically interpretable structure.trajectories FIG2 ), but our model slightly overestimates the attractor basin change probabilities, leading to a heavier tail of the distribution.

In order to demonstrate interpretable representation learning on a complex real world task, we trained our model on vital sign time series measurements of intensive care unit (ICU) patients.

We analyze the performance of the resulting clustering w.r.t.

the patients' future physiology states in Table 3 .

This can be seen as a way to assess the representations' informativeness for a downstream prediction task.

For details regarding the data selection and processing, we refer to the appendix (Sec. D.3).Our full model (including the latent Markov model) performs best on the given tasks, i.e. better than k-means and also better than the SOM-VAE without probabilistic model.

This could be due to the noisiness of the medical data and the probabilistic model's smoothing tendency (see Sec. 2.3).In order to qualitatively assess the interpretability of the probabilistic SOM-VAE, we analyzed the average future physiology score per cluster FIG3 .

Our model exhibits clusters where higher scores are enriched compared to the background level.

Moreover, these clusters form compact structures, facilitating interpretability.

We do not observe such interpretable structures in the other methods.

For full results on acute physiology scores, an analogue experiment showing the future mortality risk associated with different regions of the map, and an analysis of enrichment for particular physiological abnormalities, we refer to the appendix (Sec. D.4).As an illustrative example for data visualization using our method, we show the trajectories of two patients that start in the same state FIG3 .

The trajectories are plotted in the representation space of the probabilistic SOM-VAE and should thus be compared to the visualization in FIG3 .

One patient (green) stays in the regions of the map with low average physiology score and eventually gets discharged from the hospital healthily.

The other one (red) moves into map regions with high average physiology score and ultimately dies.

Such knowledge could be helpful for doctors, who could determine the risk of a patient for certain deterioration scenarios from a glance at their trajectory in the SOM-VAE representation.

The SOM-VAE can recover topologically interpretable state representations on time series and static data.

It provides an improvement to standard methods in terms of clustering performance and offers a way to learn discrete two-dimensional representations of the data manifold in concurrence with the reconstruction task.

It introduces a new way of overcoming the non-differentiability of the discrete representation assignment and contains a gradient-based variant of the traditional self-organizing map that is more performant than the original one.

On a challenging real world medical data set, our model learns more informative representations with respect to medically relevant prediction targets than competitor methods.

The learned representations can be visualized in an interpretable way and could be helpful for clinicians to understand patients' health states and trajectories more intuitively.

It will be interesting to see in future work whether the probabilistic component can be extended to not just improve the clustering and interpretability of the whole model, but also enable us to make predictions.

Promising avenues in that direction could be to increase the complexity by applying a higher order Markov Model, a Hidden Markov Model or a Gaussian Process.

Another fruitful avenue of research could be to find more theoretically principled ways to overcome the non-differentiability and compare them with the empirically motivated ones.

Lastly, one could explore deviating from the original SOM idea of fixing a latent space structure, such as a 2D grid, and learn the neighborhood structure as a graph directly from data.

The general idea of a self-organizing map (SOM) is to approximate a data manifold in a high-dimensional continuous space with a lower dimensional discrete one BID21 .

It can therefore be seen as a nonlinear discrete dimensionality reduction.

The mapping is achieved by a procedure in which this discrete representation (the map) is randomly embedded into the data space and then iteratively optimized to approach the data manifold more closely.

The map consists of k nodes V = {v 1 , . . .

, v k }, where every node corresponds to an embedding in the data space e v ∈ R d and a representation in the lower-dimensional discrete space m v ∈ M , where usually M ⊂ N 2 .

There are two different geometrical measures that have to be considered during training: the neighborhood function N (m u , mṽ) that is defined on the low-dimensional map space and the Euclidean distance D(e u , eṽ) = e u − eṽ 2 in the high-dimensional data space.

The SOM optimization tries to induce a coupling between these two properties, such that the topological structure of the representation reflects the geometrical structure of the data.

Require: DISPLAYFORM0 for all x i ∈ D do find the closest SOM nodeṽ := arg min v∈V x i − e v 2 update node embedding eṽ ← eṽ + η (x i − eṽ) for all u ∈ V \ṽ do update neighbor embedding e u ← e u + η N (mṽ, m u )(x i − e u ) end for end for end whileThe SOM training procedure is described in Algorithm 1.

During training on a data set D, a winner nodeṽ is chosen for every point x i according to the Euclidean distance of the point and the node's embedding in the data space.

The embedding vector for the winner node is then updated by pulling it into the direction of the data point with some step size η.

The embedding vectors of the other nodes are also updated -potentially with a smaller step size -depending on whether they are neighbors of the winner node in the map space M .The neighborhood is defined by the neighborhood function N (m u , mṽ).

There can be different design choices for the neighborhood function, e.g. rectangular grids, hexagonal grids or Gaussian neighborhoods.

For simplicity and ease of visualization, we usually choose a two-dimensional rectangular grid neighborhood in this paper.

In this original formulation of the SOM training, the nodes are updated one by one with a fixed step size.

In our model, however, we use a gradient-based optimization of their distances to the data points and update them in minibatches.

This leads to larger step sizes when they are farther away from the data and smaller step sizes when they are close.

Overall, our gradient-based SOM training seems to perform better than the original formulation (see Tab.

1).It also becomes evident from this procedure that it will be very hard for the map to fit disjoint manifolds in the data space.

Since the nodes of the SOM form a fully connected graph, they do not possess the ability to model spatial gaps in the data.

We overcome this problem in our work by mapping the data manifold with a variational autoencoder into a lower-dimensional latent space.

The VAE can then learn to close the aforementioned gaps and map the data onto a compact latent manifold, which can be more easily modeled with the SOM.

The hyperparameters of our model were optimized using Robust Bayesian Optimization with the packages sacred and labwatch BID14 for the parameter handling and RoBo for the optimization, using the mean squared reconstruction error as the optimization criterion.

Especially the weighting hyperparameters α, β, γ and τ (see Eq. (1) and Eq. (2)) have to be tuned carefully, such that the different parts of the model converge at roughly the same rate.

We found that 2000 steps of Bayesian optimization sufficed to yield a performant hyperparameter assignment.

Since our model defines a general framework, some competitor models can be seen as special cases of our model, where certain parts of the loss function are set to zero or parts of the architecture are omitted.

We used the same hyperparameters for those models.

For external competitor methods, we used the hyperparameters from the respective publications where applicable and otherwise the default parameters from their packages.

The models were implemented in TensorFlow BID0 and optimized using Adam (Kingma and Ba, 2014).

Given that one of our most interesting tasks at hand is the clustering of data, we need some performance measures to objectively compare the quality of this clustering with other methods.

The measures that we decided to use and that have been used extensively in the literature are purity and normalized mutual information (NMI) (Manning et al., 2008) .

We briefly review them in the following.

Let the set of ground truth classes in the data be C = {c 1 , c 2 , . . .

, c J } and the set of clusters that result from the algorithm Ω = {ω 1 , ω 2 , . . . , ω K }.

The purity π is then defined as π(C, Ω) = DISPLAYFORM0 where N is the total number of data points.

Intuitively, the purity is the accuracy of the classifier that assigns the most prominent class label in each cluster to all of its respective data points.

While the purity has a very simple interpretation, it also has some shortcomings.

One can for instance easily observe that a clustering with K = N , i.e. one cluster for every single data point, will yield a purity of 1.0 but still probably not be very informative for most tasks.

It would therefore be more sensible to have another measure that penalizes the number of clusters.

The normalized mutual information is one such measure.

The NMI is defined as NMI(C, Ω) = Table 1 , we performed experiments to assess the influence of the number of clusters k on the clustering performance of our method.

We chose different values for k between 4 and 64 and tested the clustering performance on MNIST and Fashion-MNIST (Tab.

S1).It can be seen that the purity increases monotonically with k, since it does not penalize larger numbers of clusters (see Sec. C).

The NMI, however, includes an automatic penalty for misspecifying the model with too many clusters.

It therefore increases first, but then decreases again for too large values of k. The optimal k according to the NMI seems to lie between 16 and 36.

The Lorenz system is the system of coupled ordinary differential equations defined by , the system shows chaotic behavior by forming a strange attractor BID28 with the two attractor points being given by DISPLAYFORM0 DISPLAYFORM1 We simulated 100 trajectories of 10,000 time steps each from the chaotic system and trained the SOM-VAE as well as k-means on it with 64 clusters/embeddings respectively.

The system chaotically switches back and forth between the two attractor basins.

By computing the Euclidian distance between the current system state and each of the attractor points p 1,2 , we can identify the current attractor basin at each time point.

In order to assess the interpretability of the learned representations, we have to define an objective measure of interpretability.

We define interpretability as the similarity between the representation and the system's ground truth macro-state.

Since representations at single time points are meaningless with respect to this measure, we compare the evolution of representations and system state over time in terms of their entropy.

We divided the simulated trajectories from our test set into spans of 100 time steps each.

For every subtrajectory, we computed the entropies of those subtrajectories in the real system space (macro-state and noise), the assigned attractor basin space (noise-free ground-truth macro-state), the SOM-VAE representation and the k-means representation.

We also observed for every subtrajectory whether or not a change between attractor basins has taken place.

Note that the attractor assignments and representations are discrete, while the real system space is continuous.

In order to make the entropies comparable, we discretize the system space into unit hypercubes for the entropy computation.

For a representation R with assignments R t at time t and starting time t start of the subtrajectory, the entropies are defined as DISPLAYFORM2 with H(·) being the Shannon information entropy of a discrete set.

All experiments were performed on dynamic data extracted from the eICU Collaborative Research Database BID12 .

Irregularly sampled time series data were extracted from the raw tables and then resampled to a regular time grid using a combination of forward filling and missing value imputation using global population statistics.

We chose a grid interval of one hour to capture the rapid dynamics of patients in the ICU.Each sample in the time-grid was then labeled using a dynamic variant of the APACHE score BID20 , which is a proxy for the instantaneous physiological state of a patient in the ICU.

Specifically, the variables MAP, Temperature, Respiratory rate, HCO3, Sodium, Potassium, and Creatinine were selected from the score definition, because they could be easily defined for each sample in the eICU time series.

The value range of each variable was binned into ranges of normal and abnormal values, in line with the definition of the APACHE score, where a higher score for a variable is obtained for abnormally high or low values.

The scores were then summed up, and we define the predictive score as the worst (highest) score in the next t hours, for t ∈ {6, 12, 24}. Patients are thus stratified by their expected pathology in the near future, which corresponds closely to how a physician would perceive the state of a patient.

The training set consisted of 7000 unique patient stays, while the test set contained 3600 unique stays.

As mentioned in the main text (see FIG3 the SOMVAEProb is able to uncover compact and interpretable structures in the latent space with respect to future physiology scores.

In this section we show results for acute physiology scores in greater detail, analyze enrichment for future mortality risk, arguably the most important severity indicator in the ICU, and explore phenotypes for particular physiological abnormalities.

FIG0 : (a) shows the difference in distribution of the acute physiology score in the next 24 hours, between time-points assigned to the most abnormal cell in the SOMVAEprob map with coordinates [2,0] vs. a normal cell chosen from the middle of the map with coordinates [4, 3] .

It is apparent that the distributions are largely disjoint, which means that the representation induced by SOMVAEprob clearly distinguishes these risk profiles.

Statistical tests for difference in distribution and location parameter are highly significant at p-values of p ≤ 10 −3 , as we have validated using a 2-sample t-test and Kolmogorov-Smirnov test.

In (b-c) the enrichment of the map for the mean acute physiology score in the next 6 and 12 hours is shown, for completeness.

The enrichment patterns on the 3 maps, for the future horizons {6, 12, 24}, are almost identical, which provides empirical evidence for the temporal stability of the SOMVAEProb embedding. hours.

We observe that the left-edge and right-edge regions of the SOMVAEprob map which are enriched for higher acute physiology scores (see FIG3 also exhibit elevated mortality rates over the baseline.

Interestingly, according to future mortality risk, which is an important severity indicator, patients on the left-edge are significantly more sick on average than those on the right edge, which is less visible from the enrichment for acute physiology scores.

PATIENT STATE PHENOTYPES ON THE SOMVAEP R O B MAP Low sodium and high potassium states are enriched near the left edge, and near the right edge, respectively, which could represent sub-types of the high-risk phenotype found in these regions (compare FIG3 the distribution of the acute physiology score).

Elevated creatinine is a trait that occurs in both these regions.

A compact structure associated with elevated HCO3 can be found in the center of the map, which could represent a distinct phenotype with lower mortality risk in our cohort.

In all phenotypes, the tendency of SOMVAEprob to recover compact structures is exemplified.

FIG3 : Images generated from the SOM-VAE's latent space with 512 embeddings trained on MNIST.

It yields an interpretable discrete two-dimensional representation of the data manifold in the higher-dimensional latent space.

Figure S5 : Images generated from the SOM-VAE's latent space with 512 embeddings trained on Fashion-MNIST.

It yields an interpretable discrete two-dimensional representation of the data manifold in the higherdimensional latent space.

<|TLDR|>

@highlight

We present a method to learn interpretable representations on time series using ideas from variational autoencoders, self-organizing maps and probabilistic models.