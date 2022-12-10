Spatiotemporal forecasting has become an increasingly important prediction task in machine learning and statistics due to its vast applications, such as climate modeling, traffic prediction, video caching predictions, and so on.

While numerous studies have been conducted, most existing works assume that the data from different sources or across different locations are equally reliable.

Due to cost, accessibility, or other factors, it is inevitable that the data quality could vary, which introduces significant biases into the model and leads to unreliable prediction results.

The problem could be exacerbated in black-box prediction models, such as deep neural networks.

In this paper, we propose a novel solution that can automatically infer data quality levels of different sources through local variations of spatiotemporal signals without explicit labels.

Furthermore, we integrate the estimate of data quality level with graph convolutional networks to exploit their efficient structures.

We evaluate our proposed method on forecasting temperatures in Los Angeles.

Recent advances in sensor and satellite technology have facilitated the collection of large spatiotemporal datasets.

As the amount of spatiotemporal data increases, many have proposed representing this data as time-varying graph signals in various domains, such as sensor networks BID21 BID29 , climate analysis BID2 BID17 , traffic control systems BID15 , and biology BID18 BID26 .While existing work have exploited both spatial structures and temporal signals, most of them assume that each signal source in a spatial structure is equally reliable over time.

However, a large amount of data comes from heterogeneous sensors or equipment leading to various levels of noise BID22 BID27 .

Moreover, the noises of each source can vary over time due to movement of the sensors or abrupt malfunctions.

This problem raises significantly challenges to train and apply complex black box machine learning models, such as deep neural networks, because even a small perturbation in data can deceive the models and lead to unexpected behaviors BID5 BID13 .

Therefore, it is extremely important to consider data quality explicitly when designing machine learning models.

The definitions of data quality can be varied -high quality data is generally referred to as fitness for intended uses in operations, decision making and planning BID19 .

In this paper, we narrow down the definition as a penalizing quantity for high local variations.

We consider a learning problem of spatiotemporal signals that are represented by time-varying graph signals for different data qualities.

Given a graph G = (V, E, W) and observations X ∈ R N ×M ×T where N, M, T are the number of vertices, the types of signals, and the length of time-varying signals, respectively.

We define the concept of data quality levels at each vertex as latent variables, which are connected through a graph using a local variation of the vertex.

The local variation at each vertex depends on the local spatial structure and neighboring signals.

Our definition of data quality can be easily incorporated into any existing machine learning models through a regularizer in their objective functions.

In this paper, we develop data quality long short-term memory (DQ-LSTM) neural networks for spatiotemporal forecasting.

DQ-LSTM effectively exploits spatial structures of data quality levels at each vertex through graph convolution, which examines neighboring signals at a set of K-hop neighboring vertices, and captures the temporal dependencies of each time series through LSTMs.

We demonstrate that data quality is an essential factor for improving the predictive performance of neural networks via experiments on urban heat island prediction in Los Angeles.

Related work A series of work have been conducted on addessing the issues of data qualities and heterogeneous data sources.

BID23 is the first theoretical work that proposes a mixture model for captureing two types of labels in supervised learning.

One type of the labels is considered as high quality labels from an expensive source (domain experts) while another type is from errorprone crowdsourcing.

Since the reliability or quality of the labels is different, it is not desired to consider them equally.

The authors proposed a learning algorithm that can utilize the error-prone labels to reduce the cost required for the expert labeling.

BID27 address issues from strong and weak labelers by developing an active learning algorithm minimizing the number of label requests.

BID22 focus on the data of variable quality resulting from heterogeneous sources.

The authors define the concept of heterogeneity of data and develop a method of adjusting the learning rate based on the heterogeneity.

Different from existing works, our proposed framework differentiates heterogeneous sources based on neighborhood signals without any explicit labels.

Another set of work related to our study is learning and processing graph signals or features.

Spectral graph theory BID3 BID24 BID21 has been developed as a main study to understand two aspects of graph signals: structures and signals.

Under this theory many models have been introduced to exploit convolutional neural networks (CNNs) which provide an efficient architecture to extract localized patterns from regular grids, such as images BID14 .

BID1 learns convolutional parameters based on the spectrum of the graph Laplacian.

Later, BID8 extends the spectral aspect of CNNs on graphs into largescale learning problemsDefferrard et al. FORMULA0 proposes a spectral formulation for fast localized filtering with efficient pooling.

Furthermore, BID12 re-formularizes existing ideas into layer-wise neural networks that can be tuned through backpropagation rule with a first-order approximation of spectral filters introduced in BID7 .

Built on these work, we propose a graph convolutional layer that maps spatiotemporal features into a data quality level.

Outline We review graph signal processing to define the local variation and a data quality level (DQL) with graph convolutional networks in Section 2.

In Section 3, we provide how the data quality levels are exploited with recurrent neural networks to differentiate reliability of observations on vertices.

Also, we construct a forecasting model, DQ-LSTM.

Our main result is presented in Section 4 with other baselines.

In Section 5 we discuss its properties and interpret the data reliability inferred from our model.

We first show how to define the local variation at a vertex based on graph signals.

Then, we explain how the variational features at each vertex can be used to generate a data quality level.

We focus on the graph signals defined on an undirected, weighted graph G = (V, E, W), where V is a set of vertices with |V| = N and E is a set of edges.

W ∈ R N ×N is a random-walk normalized weighted adjacency matrix which provides how two vertices are relatively close.

When the elements W ij are not be expliticly provided by dataset, the graph connectivity can be constructed by various distance metrics, such as Euclidean distance, cosine similarity, and a Gaussian kernel BID0 , on the vertex features V ∈ R N ×F where F is the number of the features.

Once all the structural connectivity is provided, the local variation can be defined by the edge derivative of a given graph signal x ∈ R N defined on every vertex BID28 .

where e = (i, j) is defined as a direction of the derivative and x(i) is a signal value on the vertex i. The graph gradient of x at vertex i can be defined by Eq. 1 over all edges joining the vertex i. DISPLAYFORM0 DISPLAYFORM1 where N i is a set of neighbor vertices of the vertex i. While the dimesion of the graph gradient is different due to the different number of neighbors of each vertex, the local variation at vertex i can be defined by a norm of the graph gradient: DISPLAYFORM2 Eq. 3 provides a measure of local variation of x at vertex i. As it indicates, if all neighboring signals of i are close to the signal at i, the local variation at i should be small and it means less fluctuated signals around the vertex.

As Eq. 3 shows, the local variation is a function of a structural connectivity W and graph signals x. FIG0 illustrates how the two factors affect the local variation at the same vertex.

The concept of the local variation is easily generalized to multivariate graph signals of M different measures by repeatedly computing Eq. 3 over all measures.

DISPLAYFORM3 where x m ∈ R N corresponds the mth signals from multiple sensors.

As Eq. 4 indicates, L i is a M dimensional vector describing local variations at the vertex i with respect to the M different measures.

Finally, it is desired to represent Eq. 4 in a matrix form to be combined with graph convolutional networks later.

DISPLAYFORM4 where D is a degree matrix defined as D ii = j W ij and is an element-wise product operator.

X is a N × M matrix describing multivariate graph signals on N vertices and x m is a mth column of X. L ∈ R N ×M is a local variation matrix and L im is the local variation at the vertex i with respect to the mth signal.

While the term of data quality has been used in various ways, it generally means "fitness of use" for intended purposes BID10 .

In this section, we will define the term under the data property we are interested in and propose how to exploit the data quality level into a general framework.

Given a multivariate graph signal X ∈ R N ×M on vertices represented by a feature matrix V ∈ R N ×F , we assume that a signal value at a certain vertex i is desired not be significantly different with signals of neighboring vertices j ∈ N i .

This is a valid assumption if the signal value at the vertex i is dependent on (or function of) features of the vertex i when an edge weight is defined by a distance in the feature vector space between two vertices.

In other words, if two vertices have similar features, they are connected to form the graph structure and the signal values observed at the vertices are highly likely similar.

There are a lot of domains which follow the assumption, for instance, geographical features (vertex features) and meteorological observations (graph signal) or sensory nervous features and received signals.

Under the assumption, we define the data quality level (score) at a vertex i as a function of local variations of i: DISPLAYFORM0 It is flexible to choose the function q. For example, s i can be defined as an average of L i .

If so, all measures are equally considered to compute the data quality at vertex i. In more general sense, we can introduce parameterized function q(L i ; Φ) and learn the parameters through data.

BID12 propose a method that learns parameters for graph-based features by the layer-wise graph convolutional networks (GCN) with arbitrary activation functions.

For a single layer GCN on a graph, the latent representation of each node can be represented as: DISPLAYFORM1 DISPLAYFORM2 2 provides structural connectivities of a graph and Θ is a trainable parameter matrix.

σ is an activation function.

By stacking σ(ÂXΘ), it is able to achieve larger receptive fields with multi-layer GCN.

See details in BID12 .We can replaceÂX with L which is also a function of the weighted adjacency W and the graph signal X. Note that values in row i ofÂX and L are a function of values at i as well as neighbors of i. Although L only exploits nearest neighbors (i.e., 1-hop neighbors), it is possible to consider K-hop neighbors to compute the local variations by stacking GCN before applying Eq. 3.

The generalized formula for the data quality level can be represented as: DISPLAYFORM3 where K is the number of GCN layers and s = (s 1 , s 2 , · · · , s N ) is the data quality level of each vertex incorporating K-hop neighbors.

We propose some constraints to ensure that a higher s i corresponds to less fluctuated around i. First, we constrain Φ to be positive to guarantee that larger elements in L cause larger LΦ that are inputs of σ L .

Next, we use an activation function that is inversely proportional to an input, e.g., σ L (x) = 1 1+x , to meet the relation between the data quality s i and the local variations L i .

Once s is obtained, it will be combined with an objective function to assign a penalty for each vertex loss function.

In this section, we give the details of the proposed model, which is able to exploit the data quality defined in Section 2.2 for practical tasks.

First, it will be demonstrated how the data quality network DQN is combined with recurrent neural networks (LSTM) to handle time-varying signals.

We, then, provide how this model is trained over all graph signals from all vertices.

Data quality network In Section 2.2, we find that the local variations around all vertices can be computed once graph signals X are given.

Using the local variation matrix L, the data quality level at each vertex s i can be represented as a function of L i with parameters Φ (See Eq. 6).

While the function q is not explicitly provided, we can parameterize the function and learn the parameters Φ ∈ R M through a given dataset.

One of straightforward parameterizations is based on fully connected neural networks.

Given a set of features, neural networks are efficient to find nonlinear relations among the features.

Furthermore, the parameters in the neural networks can be easily learned through optimizing a loss function defined for own purpose.

Thus, we use a single layer neural networks Φ followed by an activation function σ L (·) to transfer the local variations to the data quality level.

Note that multi-layer GCN can be used between graph signals and DQN to extract convolved signals as well as increase the Long short term memory Recurrent neural networks (RNNs) are especially powerful to extract latent patterns in time series which has inherent dependencies between not only adjacent data points but also distant points.

Among existing various RNNs, we use Long short term memory (LSTM) BID9 to handle the temporal signals on vertices for a regression task.

We feed finite lengths k of sequential signals into LSTM as an input series to predict the observation at next time step.

The predicted value is going to be compared to a true value and all parameters in LSTM will be updated via backpropagation through time.

FIG1 illustrates how the data quality networks with LSTM (DQ-LSTM) consists of submodules.

Time-varying graph signals on N vertices can be represented as a tensor form, X ∈ R N ×M ×T where the total length of signals is T .

First, the time-varying graph signals for each vertex are segmentized to be fed into LSTM.

For example, X (i, :, h : h + k − 1) is one of segmentized signals on vertex i starting at t = h. Second, the graph signals for all vertices at the last time stamp X (:, :, h + k − 1) are used as an input of GCN followed by DQN.

Hence, we consider the data quality level by looking at the local variations of the last signals and the estimated quality level s i is used to assign a weight on the loss function defined on the vertex i. We use the mean squared error loss function.

For each vertex i, DQ-LSTM repeatedly reads inputs, predicts outputs, and updates parameters as many as a number of segmentized length k time series.

DISPLAYFORM0 where n i is the number of segmentized series on vertex i andX (i, :, k + j − 1) is a predicted value from a fully connected layer (FC) which reduces a dimension of an output vector from LSTM.

L 2 regularization is used to prevent overfitting.

Then, the total loss function over all vertices is as DISPLAYFORM1

In this section, we evaluate DQ-LSTM on real-world climate datasets.

In the main set of experiments, we evaluate the mean absolute error (MAE) of the predictions produced by DQ-LSTM over entire weather stations.

In addition, we analyze the data quality levels estimated by DQ-LSTM.

We use real-world datasets on meteorological measurements from two commercial weather services, Weather Underground(WU) and WeatherBug(WB).

Both datasets provide real-time weather information from personal weather stations.

In the datasets, all stations are distributed around Los Angeles County, and geographical characteristics of each station are also provided.

These characteristics would be used as a set of input features V to build a graph structure.

The list of the static 11 characteristics is, Latitude, Longitude, Elevation, Tree fraction, Vegetation fraction, Albedo, Distance from coast, Impervious fraction, Canopy width, Building height, and Canopy direction.

Meteorological signals at each station are observed through the installed instruments.

The types of the measurements are Temperature, Pressure, Relative humidity, Solar radiation, Precipitation, Wind speed, and Wind direction.

Since each weather station has observed the measurements under its own frequency (e.g., every 5 minutes or every 30 minutes), we fix the temporal granularity at 1 hour and aggregate observations in each hour by averaging them.

We want to ensure that the model can be verified and explained physically within one meteorological regime before applying it to the entire year with many other regimes.

Since it is more challenging to predict temperatures in the summer season of Los Angeles due to the large fluctuation of daytime temperatures (summer: 36• F / 19• C and winter: 6 • F / 3.3• C between inland areas and the coastal Los Angeles Basin), we use 2 months observations from each service, July/2015 and August/2015, for our experiments.

The dataset description is provided in Appendix A.

Since structural information between pairs of stations is not directly known, we need to construct a graph of the weather stations.

In general graphs, two nodes can be interpreted as similar nodes if they are connected.

Thus, as mentioned in Section 2.1, we can compute a distance between two nodes in the feature space.

A naive approach to defining the distance is using only the geolocation features, Latitude and Longitude.

However, it might be inappropriate because other features can be significantly different even if two stations are fairly close.

For example, the distance between stations in the Elysian Park and Downtown LA is less than 2 miles, however, the territorial characteristics are significantly different.

Furthermore, the different characteristics (e.g., Tree fraction or Impervious fraction) can affect weather observations (especially, temperature due to urban heat island effect).

Thus, considering only physical distance may improperly approximate the meteorological similarity between two nodes.

To alleviate this issue, we assume that all static features are equally important.

This is a reasonable assumption because we do not know which feature is more important since each feature can affect weather measurements.

Thus, we normalize all spatial features.

In this experiment, we use the Gaussian kernel e (−γ Vi−Vj 2 ) with γ = 0.2 and 0.6 for WU and WB, respectively, and make weights less than 0.9 zero (i.e., disconnected) such that the average number of node neighbors is around 10.

We compare our approach to well-studied baselines for time-series forecasting.

First, we compare against a stochastic process, autoregressive (AR), which estimates future values based on past values.

Second, we further compare against a simple LSTM.

This model is expected to infer mixeddependencies among the input multivariate signals and provide a reference error of the neural networks based model.

Lastly, we use graph convolutional networks BID12 which are also able to infer the data quality level from a given dataset.

We test a single layer GCN (K = 1) and two-layer GCN (K = 2).

Since DQ-LSTM and our baselines are dependent on previous observations, we set a common lag length of k = 10.

For the deep recurrent models, the k-steps previous observations are sequentially inputted to predict next values.

All deep recurrent models have the same 50 hidden units and one fully connected layer (R 50×1 ) that provides the target output.

For GCN-LSTM and DQ-LSTM, we evaluate with different numbers of layers (K) of GCN.

We set the dimension of the first (K = 1) and second (K = 2) hidden layer of GCN as 10 and 5, respectively, based on the cross validation.

The final layer always provides a set of scalars for every vertex, and we set β = 0.05 for the L2 regularization of the final layer.

We use the Adam optimizer BID11 with a learning rate of 0.001 and a mean squared error objective.

We split each dataset into three subsets: training, validation, and testing sets.

The first 60% observations are used for training, the next 20% is used to tune hyperparameters (validation), and the remaining 20% is used to report error results (test).

Among the measurements provided, Temperature is used as the target measurement, i.e., output of LSTM, and previous time-step observations, including Temperature, are used as input signals.

We report average scores over 20 trials of random initializations.

Experimental results are summarized in TAB0 .

We report the temperature forecasting mean absolute error (MAE) of our DQ-LSTM model with standard deviations.

Meteorological measurements for July and August are denoted by 7 and 8, and K indicates the number of GCN layers.

Overall, the models that account for graph structures outperform AR and LSTM.

While the node connectivities and weights are dependent on our distance function (Section 4.2), TAB0 clearly shows that knowing the neighboring signals of a given node can help predict next value of the node.

Although GCN are able to transfer a given signal of a vertex to a latent representation that is more compact and expressive, GCN have difficulty learning a mapping from neighboring signals to data quality level directly, unlike DQ-LSTM which pre-transfers the signals to local variations explicitly.

In other words, given signal X, what GCN learns is s = f (X) where s is the data quality we want to infer from data; however, DQ-LSTM learns s = g(Y = h(X)) where Y is a local variation matrix given by X in a closed form, h. Thus, lower MAEs of DQ-LSTM verify that our assumption in Section 2.2 is valid and the local variations are a useful metric to measure data quality level.

It is also noteworthy that DQ-LSTM with a GCN reports the lowest MAE among all models.

This is because the additional trainable parameters in GCN increase the number of neighboring nodes that are accounted for to compute better local variations.

As DQ-LSTM can be combined with GCN, it is possible to represent each node as an embedding obtained from an output of GCN.

Embeddings from deep neural networks are especially interesting since they can capture distances between nodes.

These distances are not explicitly provided but inherently present in data.

Once the embeddings are extracted, they can be used for further tasks, such as classification and clustering BID6 BID12 .

Moreover, since the embeddings have low dimensional representations, it is more efficient to visualize the nodes by manifold learning methods, such as t-SNE BID16 .

Visualization with spatiotemporal signals is especially effective to show how similarities between nodes change.

The green dots are neighbors of a red dot, and they are closely distributed to form a cluster.

There are two factors that affect embeddings: temporal signals and spatial structure.

Ideally, connected nodes have observed similar signals and thus, they are mapped closely in the embedding space.

However, if one node v i measures a fairly different value compared to other connected values, the node's embedding will also be far from its neighbors.

Furthermore, if one node v i is connected to a subset of a group of nodes {g} as well as an additional node v j / ∈ {g}, v i would be affected by the subset and v j / ∈ {g} simultaneously.

For example, if the signals observed at v i are similar to the signals at {g}, the embedding of v i is still close to those of {g}. However, if the signals of v i are close to that of v j , or the weight of e(i, j) is significantly high, v i will be far away from {g} in the embedding space.

Such intuition of the embedding distribution can be used to find potentially low-quality nodes, which we analyze next.

FIG2 shows that a node v 25 is affected by its neighboring green nodes and v 4 that is not included in the cluster (green dots).

The red dot v 22 is connected with the green dots (v 19 , v 20 , v 21 , v 23 , v 25 , v 29 ).

Since these nodes have similar spatial features and are connected, the nodes are expected to have similar observations.

At t = 0, the distribution of the nodes seems like a cluster.

However, v 25 is far away from other green nodes and the red node at t = 4.

There are two possible reasons.

First, observations of v 25 at t = 4 may be too different with those of other green nodes and the red node.

Second, observations at v 4 , which is only connected to v 25 (not to other green nodes and the red node), might be too noisy.

The first case violates our assumption (See Section 2.2, such that the observation at v 25 should be similar to those of other green nodes.); therefore, the observations of v 25 at t = 4 might be noisy or not reliable.

In the second case, the observations of v 4 at t = 4 might be noisy.

Thus, v 25 and v 4 are candidates of low-quality nodes.

Since we do not have explicit labels for the data quality levels, it is not straightforward to directly evaluate the data quality inferred from DQ-LSTM.

Instead, we can verify the inferred data quality by studying high and low quality examples from embedding distributions and associated meterological observations.

TAB1 shows meterological observations associated with the previously discussed embedding distribution at t = 4 FIG2 FIG2 .

Note that v 4 has very different geological features compared to the features of the green nodes and thus, v 4 is not connected to v 22 or other green nodes except v 25 .

Consequently, v 25 is the bridge node between v 4 and the cluster of v 22 .

Since a bridge node is affected by two (or more) different groups of nodes simultaneously, the quality level at the bridge node is more susceptible than those of other nodes.

However, this does not directly mean that a bridge node must have a lower data quality level.

As TAB1 shows, s 4 has the lowest data quality level, which comes from the discrepancy between its neighboring signals and x 4 .

Since v 4 is connected to v 25 , v 4 pulls v 25 and s 4 , lowering s 25 relative to data quality levels of the other green nodes that are correctly inferred.

In this work, we study the problem of data quality for spatiotemporal data analysis.

While existing works assume that all signals are equally reliable over time, we argue that it is important to differentiate data quality because the signals come from heterogeneous sources.

We proposed a novel formulation that automatically infers data quality levels of different sources and developed a specific formulation, namely DQ-LSTM, based on graph convolution for spatiotemporal forecasting.

We demonstrate the effectiveness of DQ-LSTM on inferring data quality and improving prediction performance on a real-world climate dataset.

For future work, we are interested in further refining the definitions of data quality and examining rigorous evaluation metrics.

In order to have a fair comparison, we use real-world meteorological measurements from two commercial weather service providing real-time weather information, Weather Underground(WU) 1 and WeatherBug(WB) 2 .

Both services use observations from automated personal weather stations (PWS).

The PWS are illustrated in FIG4 .

<|TLDR|>

@highlight

We propose a method that infers the time-varying data quality level for spatiotemporal forecasting without explicitly assigned labels.

@highlight

Introduces a new definition of data quality that relies on the notion of local variation defined in (Zhou and Scholkopf) and extends it to multiple heterogenous data sources.

@highlight

This work proposed a new way to evaluate the quality of different data sources with the time-vary graph model, with the quality level used as a regularization term in the objective function