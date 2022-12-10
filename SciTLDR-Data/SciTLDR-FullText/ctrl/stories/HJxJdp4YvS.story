Generating visualizations and interpretations from high-dimensional data is a common problem in many fields.

Two key approaches for tackling this problem  are clustering and representation learning.

There are very performant deep clustering models on the one hand and interpretable representation learning techniques,  often relying on latent topological structures such as self-organizing maps, on the other hand.

However, current methods do not yet successfully combine these two approaches.

We present a new deep architecture for probabilistic clustering,  VarPSOM, and its extension to time series data, VarTPSOM, composed of VarPSOM  modules connected by LSTM cells.

We show that they achieve superior  clustering performance compared to current deep clustering methods on static  MNIST/Fashion-MNIST data as well as medical time series, while inducing an interpretable representation.

Moreover, on the medical time series, VarTPSOM successfully predicts future trajectories in the original data space.

Given a set of data samples {x i } i=1,...,n , where x i ∈ R d , the goal is to partition the data into a set 102 of clusters {S i } i=1,...,K , while retaining a topological structure over the cluster centroids.

The proposed architecture for static data is presented in Figure 1a .

The input vector x i is embedded 104 into a latent representation z i using a VAE.

This latent vector is then clustered using PSOM, a 112 DECODER ENCODER (a) VarPSOM architecture for clustering of static data.

Data points xi are mapped to a continuous embedding zi using a VAE (parameterized by Φ).

The loss function is the sum of a SOMbased clustering loss and the ELBO.

(b) VarTPSOM architecture, composed of VarPSOM modules connected by LSTMs across the time axis, which predict the continuous embedding zt+1 of the next time step.

This architecture allows to unroll future trajectories in the latent space as well as the original data space by reconstructing the xt using the VAE.

A Self-Organizing Map is comprised of K nodes connected to form a grid M ⊆ N 2 , where the 114 node m i,j , at position (i, j) of the grid, corresponds to a centroid vector, µ i,j in the input space.

The centroids are tied by a neighborhood relation N (µ i,j ) = {µ i−1,j , µ i+1,j , µ i,j−1 , µ i,j+1 }.

Given a 116 random initialization of the centroids, the SOM algorithm randomly selects an input x i and updates 117 both its closest centroid µ i,j and its neighbors N (µ i,j ) to move them closer to x i .

For a complete 118 description of the SOM algorithm, we refer to the appendix (A).

The Clustering Assignment Hardening method has been recently introduced by the DEC model (Xie et al., 2015) and was shown to perform well in the latent space of AEs (Aljalbout et al., 2018) .

Given an embedding function z i = f (x i ), it uses a Student's t-distribution (S) as a kernel to measure the similarity between an embedded data point z i , and a centroid µ j :

It improves the cluster purity by enforcing the distribution S to approach a target distribution, T :

By taking the original distribution to the power of γ and normalizing it, the target distribution puts more emphasis on data points that are assigned a high confidence.

We follow (Xie et al., 2015) in choosing γ=2, which leads to larger gradient contributions of points close to cluster centers, as they show empirically.

The resulting clustering loss is defined as:

Our proposed clustering method, called PSOM, expands Clustering Assignment Hardening to include a SOM neighborhood structure over the centroids.

We add an additional loss to (1) to achieve an interpretable representation.

This loss term maximizes the similarity between each data point and the neighbors of the closest centroids.

For each embedded data point z i and each centroid µ j the loss is defined as the negative sum of all the neighbors of µ j , {e : µ e ∈ N (µ j (x i ))}, of the probability that z i is assigned to e, defined as s ie .

This sum is weighted by the similarity s ij between z i and the centroid µ j :

The complete PSOM clustering loss is then: L PSOM = KL(T S) + βL SOM .

We note that for β = 0 it becomes equivalent to Clustering Assignment Hardening.

In our method, the nonlinear mapping between the input x i and embedding z i is realized by a VAE.

Instead of directly embedding the input x i into a latent embedding z i , the VAE learns a probability distribution q φ (z | x i ) parametrized as a multivariate normal distribution whose mean and variance are (µ φ , Σ φ ) = f φ (x i ).

Similarly, it also learns the probability distribution of the reconstructed output given a sampled latent embedding, p θ (x i | z) where (µ θ , Σ θ ) = f θ (z i ).

Both f φ and f θ are neural networks, respectively called encoder and decoder.

The ELBO loss is:

where p(z) is an isotropic Gaussian prior over the latent embeddings.

The second term can be interpreted as a form of regularization, which encourages the latent space to be compact.

For each data point x i the latent embedding z i is sampled from q φ (z | x i ).

Adding the ELBO loss to the PSOM loss from the previous subsection, we get the overall loss function of VarPSOM:

To the best of our knowledge, no previous SOM methods attempted to use a VAE to embed the 123 inputs into a latent space.

There are many advantages of a VAE over an AE for realizing our goals.

Its prior on the latent space encourages structured and disentangled factors (Higgins et al., 2016) To extend our proposed model to time series data, we add a temporal component to the architecture.

Given a set of N time series of length T , {x t,i } t=1,...,T ;i=1,...,N , the goal is to learn interpretable trajectories on the SOM grid.

To do so, the VarPSOM could be used directly but it would treat each time step t of the time series independently, which is undesirable.

To exploit temporal information and enforce smoothness in the trajectories, we add an additional loss to (3):

where u it,it+1 = g(z i,t , z i,t+1 ) is the similarity between z i,t and z i,t+1 using a Student's t- between time points are discouraged.

One of the main goals in time series modeling is to predict future data points, or alternatively, future embeddings.

This can be achieved by adding a long short-term memory network (LSTM) across the latent embeddings of the time series, as shown in Fig 1b.

Each cell of the LSTM takes as input the latent embedding z t at time step t, and predicts a probability distribution over the next latent embedding, p ω (z t+1 | z t ).

We parametrize this distribution as a Multivariate Normal Distribution whose mean and variance are learnt by the LSTM.

The prediction loss is the log-likelihood between the learned distribution and a sample of the next embedding z t+1 :

The final loss of VarTPSOM, which is trainable in a fully end-to-end fashion, is configurations we refer to the appendix, (B.3).

Implementation In implementing our models we focused on retaining a fair comparison with the 158 baselines.

Hence we decided to use a standard network structure, with fully connected layers of 159 dimensions d − 500 − 500 − 2000 − l, to implement both the VAE of our models and the AE of the 160 baselines.

The latent dimension, l, is set to 100 for the VAE, and to 10 for the AEs.

Since the prior 161 in the VAE enforces the latent embeddings to be compact, it also requires more dimensions to learn 162 performance.

We suspect this is due to the regularization effect of the SOM's topological structure.

Overall, VarPSOM outperforms both DEC and IDEC.

Improvement over Training After obtaining the initial configuration of the SOM structure, both 187 clustering and feature extraction using the VAE are trained jointly.

To illustrate that our architecture 188 improves clustering performance over the initial configuration, we plotted NMI and Purity against 189 the number of training iterations in Figure 2 .

We observe that the performance is stable when 190 increasing the number of epochs and no overfitting is visible.

as this is the only method among the baselines that is suited for temporal data.

We presented two novel methods for interpretable unsupervised clustering, VarPSOM and VarTP-

SOM.

Both models make use of a VAE and a novel clustering method, PSOM, that extends the 229 classical SOM algorithm to include a centroid-based probability distribution.

Our models achieve are tied by a neighborhood relation, here defined as N (µ i,j ) = {µ i−1,j , µ i+1,j , µ i,j−1 , µ i,j+1 }.

Given a random initialization of the centroids, the SOM algorithm randomly selects an input x i and 335 updates both its closest centroid µ i,j and its neighbors N (µ i,j ) to move them closer to x i .

The 336 algorithm (1) then iterates these steps until convergence.

Algorithm 1 Self-Organizing Maps

At each time t, present an input x(t) and select the winner,

Update the weights of the winner and its neighbours,

until the map converges score in the next 6 and 12 hours (APACHE-6/12), and the mortality in the next 24 hours.

Only those variables from the APACHE score definition which are recorded in the eICU 360 database were taken into account.

Each dataset is divided into training, validation and test sets for both our models and the baselines.

We evaluate the DEC model for different latent space dimensions.

Table S1 shows that the AE, used 364 in the DEC model, performs better when a lower dimensional latent space is used.

Figure S3: Randomly sampled VarTPSOM trajectories, from patients expired at the end of the ICU stay, as well as healthily dispatched patients.

Superimposed is a heatmap which displays the cluster enrichment in the current APACHE score, from this model run.

We observe that trajectories of dying patients are often in different locations of the map as healthy patients, in particular in those regions enriched for high APACHE scores, which corresponds with clinical intuition.

assignments of data points to clusters which results in a better ability to quantify uncertainty in the 392 data.

For visualizing health states in the ICU, this property is very important.

In Fig S4 we plot an 393 example patient trajectory, where 6 different time-steps (in temporal order) of the trajectory were 394 chosen.

Our model yields a soft centroid-based probability distribution which evolves with time and 395 which allows estimation of likely discrete health states at a given point in time.

For each time-step 396 the distribution of probabilities is plotted using a heat-map, whereas the overall trajectory is plotted 397 using a black line.

The circle and cross indicate ICU admission and dispatch, respectively.

Figure S4 : Probabilities over discrete patient health states for 6 different time-steps of the selected time series.

<|TLDR|>

@highlight

We present a new deep architecture, VarPSOM, and its extension to time series data, VarTPSOM,  which achieve superior clustering performance compared to current deep clustering methods on static and temporal data.