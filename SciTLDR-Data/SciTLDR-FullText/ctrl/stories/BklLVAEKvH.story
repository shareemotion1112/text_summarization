We introduce a novel end-to-end approach for learning to cluster in the absence of labeled examples.

Our clustering objective is based on optimizing normalized cuts, a criterion which measures both intra-cluster similarity as well as inter-cluster dissimilarity.

We define a differentiable loss function equivalent to the expected normalized cuts.

Unlike much of the work in unsupervised deep learning, our trained model directly outputs final cluster assignments, rather than embeddings that need further processing to be usable.

Our approach generalizes to unseen datasets across a wide variety of domains, including text, and image.

Specifically, we achieve state-of-the-art results on popular unsupervised clustering benchmarks (e.g., MNIST, Reuters, CIFAR-10, and CIFAR-100), outperforming the strongest baselines by up to 10.9%.

Our generalization results are superior (by up to 21.9%) to the recent top-performing clustering approach with the ability to generalize.

Clustering unlabeled data is an important problem from both a scientific and practical perspective.

As technology plays a larger role in daily life, the volume of available data has exploded.

However, labeling this data remains very costly and often requires domain expertise.

Therefore, unsupervised clustering methods are one of the few viable approaches to gain insight into the structure of these massive unlabeled datasets.

One of the most popular clustering methods is spectral clustering (Shi & Malik, 2000; Ng et al., 2002; Von Luxburg, 2007) , which first embeds the similarity of each pair of data points in the Laplacian's eigenspace and then uses k-means to generate clusters from it.

Spectral clustering not only outperforms commonly used clustering methods, such as k-means (Von Luxburg, 2007) , but also allows us to directly minimize the pairwise distance between data points and solve for the optimal node embeddings analytically.

Moreover, it is shown that the eigenvector of the normalized Laplacian matrix can be used to find the approximate solution to the well known normalized cuts problem (Ng et al., 2002; Von Luxburg, 2007) .

In this work, we introduce CNC, a framework for Clustering by learning to optimize expected Normalized Cuts.

We show that by directly minimizing a continuous relaxation of the normalized cuts problem, CNC enables end-to-end learning approach that outperforms top-performing clustering approaches.

We demonstrate that our approach indeed can produce lower normalized cut values than the baseline methods such as SpectralNet, which consequently results in better clustering accuracy.

Let us motivate CNC through a simple example.

In Figure 1 , we want to cluster 6 images from CIFAR-10 dataset into two clusters.

The affinity graph for these data points is shown in Figure 1 (a) (details of constructing such graph is discussed in Section 4.2).

In this example, it is obvious that the optimal clustering is the result of cutting the edge connecting the two triangles.

Cutting this edge will result in the optimal value for the normalized cuts objective.

In CNC, we define a new differentiable loss function equivalent to the expected normalized cuts objective.

We train a deep learning model to minimize the proposed loss in an unsupervised manner without the need for any labeled datasets.

Our trained model directly returns the probabilities of belonging to each cluster (Figure 1(b) ).

In this example, the optimal normalized cuts is 0.286 (Equation 1), and as we can see, the CNC loss also converges to this value (Figure 1(c) Optimal Normalized cuts #edge cuts = 1 per cluster volume = 2+2+3 = 7 1/7 + 1/7 = 0.286

Cluster 2 Cluster 2 Cluster 1

Figure 1: Motivational example: (a) affinity graph of 6 images from CIFAR-10, the objective is to cluster these images into two clusters.

(b) CNC model is trained to minimize expected normalized cuts in an unsupervised manner without the need for any labeled data.

For each data point, our model directly outputs the probabilities of it belonging to each of the clusters.

(c) The CNC loss converges to the optimal normalized cuts value.

In Algorithm 1 we show how we can scale this approach through a batch processing technique to large datasets.

We compare the performance of CNC to several learning-based clustering approaches (SpectralNet , DEC (Xie et al., 2016) , DCN (Yang et al., 2017) , VaDE (Jiang et al., 2017) , DEPICT (Ghasedi Dizaji et al., 2017) , IMSAT (Hu et al., 2017) , and IIC (Ji et al., 2019) ) on four datasets: MNIST, Reuters, CIFAR10, and CIFAR100.

Our results show up to 10.9% improvement over the baselines.

Moreover, generalizing spectral embeddings to unseen data points, a task commonly referred to as out-of-sample-extension (OOSE), is a non-trivial task (Bengio et al., 2003; Belkin et al., 2006; Mendoza Quispe et al., 2016) .

Our results confirm that CNC generalizes to unseen data.

Our generalization results are superior (by up to 21.9%) to SpectralNet , the recent top-performing clustering approach with the ability to generalize.

Recent deep learning approaches to clustering attempt to embed the input data into a form that is amenable to clustering by k-means or Gaussian Mixture Models. (Yang et al., 2017; Xie et al., 2016) focused on learning representations for clustering.

To find the clustering-friendly latent representations and to better cluster the data, DCN (Yang et al., 2017) proposed a joint dimensionality reduction (DR) and K-means clustering approach in which DR is accomplished via learning a deep neural network.

DEC (Xie et al., 2016) simultaneously learns cluster assignment and the underlying feature representation by iteratively updating a target distribution to sharpen cluster associations.

Several other approaches rely on a variational autoencoder that utilizes a Gaussian mixture prior (Jiang et al., 2017; Dilokthanakul et al., 2016; Hu et al., 2017; Ji et al., 2019; Ben-Yosef & Weinshall, 2018) .

These approaches are mainly based on data augmentation, where the network is trained to maximize the mutual information between inputs and predicted clusters, while regularizing the network so that the cluster assignment of the data points is consistent with the assignment of the augmented points.

Different clustering objectives, such as self-balanced k-means and balanced min-cut, have also been exhaustively studied (Liu et al., 2017; Chen et al., 2017; Chang et al., 2014) .

One of the most effective techniques is spectral clustering, which first generates node embeddings in the eigenspace of the graph Laplacian, and then applies k-means clustering to these vectors (Shi & Malik, 2000; Ng et al., 2002; Von Luxburg, 2007) .

To address the fact that clusters with the lowest graph conductance tend to have few nodes (Leskovec, 2009; Zhang & Rohe, 2018) , (Zhang & Rohe, 2018) proposed regularized spectral clustering to encourage more balanced clusters.

Generalizing clustering to unseen nodes and graphs is nontrivial (Bengio et al., 2003; Belkin et al., 2006; Mendoza Quispe et al., 2016) .

A recent work, SpectralNet , takes a deep learning approach to spectral clustering that generalizes to unseen data points.

This approach first learns embeddings of the similarity of each pair of data points in Laplacian's eigenspace and then applies k-means to those embeddings to generate clusters.

Unlike SpectralNet, we propose an end-to-end learning approach with a differentiable loss that directly minimizes the normalized cuts.

We show that our approach indeed can produce lower normalized cut values than the baseline methods such as SpectralNet, which consequently results in better clustering accuracy.

Our evaluation results show that CNC improves generalization accuracy on unseen data points by up to 21.9%.

Since CNC objective is based on optimizing normalized cuts, in this section, we briefly overview the formal definition of this metric.

Let G = (V, E, W ) be a graph where V = {v i } and E = {e(v i , v j )|v i ∈ V, v j ∈ V } are the set of nodes and edges in the graph and w ij ∈ W is the edge weight of the e(v i , v j ).

Let n be the number of nodes.

A graph G can be clustered into g disjoint sets S 1 , S 2 , . . .

S g , where the union of the nodes in those sets are V ( g k=1 S k = V ), and each node belongs to only one set ( g k=1 S k = ∅), by simply removing edges connecting those sets.

For example, in Figure 1(a) , by removing one edge two disjoint clusters are formed.

Normalized cuts (Ncuts) which is defined based on the graph conductance, has been studied by (Shi & Malik, 2000; Zhang & Rohe, 2018) , and the cost of a cut that forms disjoint sets S 1 , S 2 , . . .

S g is computed as:

WhereS k represents the complement of S k , i.e.,S k = i =k S i .

cut(S k ,S k ) is called cut and is the total weight of the edges that are removed from G in order to form disjoint sets

is the total edge weights (w ij ), whose end points (v i , or v j ) belong to S k .

The cut and vol are:

In running example (Figure 1 ), since the edge weights are one, cut(S 1 ,S 1 ) = cut(S 2 ,S 2 ) = 1, and

Thus the Ncuts(S 1 , S 2 ) =

Finding the cluster assignments that minimizes the normalized cuts is NP-complete and an approximation to the this problem is based on the eigenvectors of the normalized graph Laplacian which has been studied in (Shi & Malik, 2000; Zhang & Rohe, 2018) .

CNC, on the other hand, is a neural network framework for learning to cluster in the absence of labeled examples by directly minimizing the continuous relaxation of the normalized cuts.

As shown in Algorithm 1, end-to-end training of the CNC contains two steps, i.e, (i) data points embedding (line 3), and (ii) clustering (lines 4-9).

In data points embedding, the goal is to learn embeddings that capture the affinity of the data points, while the clustering step uses those embeddings to learn the CNC model and outputs the cluster assignments.

Next, we first focus on the clustering step and we introduce our new differentiable loss function to train CNC model.

Later in Section 4.2, we discuss the details of the embedding step.

In this section, we describe the clustering step in Algorithm 1 (lines 4-9).

For each data point x i , the input to clustering step is embedding v i ∈ R d (detail in Section 4.2).

The goal is to learn CNC model

, which represents the assignment probabilities over g clusters.

Clearly for n data points, it returns Y ∈ R n×g where Y ik represents the probability that v i belongs to cluster S k .

The CNC model F θ is implemented using a neural network, where the parameter vector θ denotes the network weights.

We propose a loss function based on output Y to calculate the expected normalized cuts.

Thus CNC learns the F θ by minimizing this loss (Equation 7).

Recall that cut(S k ,S k ) is the total weight of the edges that are removed from G in order to form disjoint sets S k andS k .

In our setup, embeddings are the nodes in graph G, and neighbors of an embedding v i are based on the k-nearest neighbors.

Let Y ik be the probability that node v i belongs to cluster S k .

The probability that node v j does not belong to S k would be 1 − Y jk .

Therefore, Compute affinity graph W ∈ R b×b over the M based on the k-nearest neighbors 7: Use M and W to train CNC model F θ : R d → R g that minimizes the expected normalized cuts (Equation 6) via backpropagation.

For a data point with embedding v i the output y i = F θ (v i ) represents the assignment probabilities over g clusters.

8: end while Inference, cluster assignments 9: For every data points x i whose embedding is v i return arg max of y i = F θ (v i ) as its cluster assignment.

Since the weight matrix W represents the edge weights adjacent nodes, we can rewrite Equation 3:

The element-wise product with the weight matrix ( W ) ensures that only the adjacent nodes are considered.

Moreover, the result of

W is an n × n matrix and reduce-sum is the sum over all of its elements.

From Equation 2, vol(S k , V ) is the total edge weights (w ij ), whose end points (v i , or v j ) belong to S k .

Let D be a column vector of size n where D i is the total edge weights from node v i .

We can update Equation 3 as follows to find the expected normalized cuts.

The matrix representation is given in Equation 6, where Γ = Y D is a vector in R g , and g is the number of sets/clusters.

is element-wise division and the result of (Y Γ)(1 − Y ) W is a n × n matrix where reduce-sum is the sum over all of its elements.

CNC model F θ is implemented using a neural network, where the parameter θ denotes the network weights (y i = F θ (v i )).

CNC is trained to optimize Equation 7 via backpropagation (Algorithm 1).

arg min

As you can see the affinity graph W is part of the CNC loss (Equation 7) .

Clearly, when the number of data points (n) is large, such calculation can be expensive.

However, in our experimental results, we show that for large dataset (e.g., Reuters contains 685,071 documents), it is possible to optimize the loss on randomly sampled minibatches of data.

We also build the affinity graph over a given minibach using the embeddings and based on their k nearest-neighbor (Algorithm 1 (lines 5-6)).

Specifically, in our implementation, CNC model F θ is a fully connected layer followed by gumble softmax, trained on randomly sampled minibatches of data to minimize Equation 6.

In Section 5.7 through a sensitivity analysis we show that the minibatch size affects the accuracy of our model.

When training is over, the final assignment of a data point with embedding v i to a cluster is the arg max of y i = F θ (v i ) (Algorithm 1 (line 9)).

In this section, we discuss the embedding step (line 3 in Algorithm 1).

Different affinity measures, such as simple euclidean distance or nearest neighbor pairs combined with a Gaussian kernel, have been used in spectral clustering.

Recently it is shown that unsupervised application of a Siamese network to determine the distances improves the quality of the clustering .

In this work, we also use Siamese networks to learn embeddings that capture the affinities of the data points.

Siamese network is trained to learn an adaptive nearest neighbor metric.

It learns the affinities directly from euclidean proximity by "labeling" points x i , x j positive if x i − x j is small and negative otherwise.

In other words, it generates embeddings such that adjacent nodes are closer in the embedding space and non-adjacent nodes are further.

Such network is typically trained to minimize contrastive loss:

We evaluate the performance of CNC in comparison to several deep learning-based clustering approaches on four real world datasets: MNIST, Reuters, CIFAR-10, and CIFAR-100.

The details of the datasets are as follows:

• MNIST is a collection of 70,000 28×28 gray-scale images of handwritten digits, divided into 60,000 training images and 10,000 test images.

• The Reuters dataset is a collection of English news labeled by category.

Like SpectralNet, DEC, and VaDE, we used the following categories: corporate/industrial, government/social, markets, and economics as labels and discarded all documents with multiple labels.

Each article is represented by a tfidf vector using the 2000 most frequent words.

The dataset contains 685,071 documents.

We divided the data randomly to a 90%-10% split to evaluate the generalization ability of CNC.

We also investigate the imapact of training data size on the generalization by considering following splits: 90%-10%, 70%-30%, 50%-50%, 20%-80%, and 10%-90%.

• CIFAR-10 consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.

There are 50000 training images and 10000 test images.

• CIFAR-100 has 100 classes containing 600 images each with a 500/100 train/test split per class.

In all runs we assume the number of clusters is given.

In MNIST and CIFAR-10 number of clusters (g) is 10, g = 4 in Reuters, g = 100 in CIFAR-100.

We compare CNC to SpectralNet , DEC (Xie et al., 2016) , DCN (Yang et al., 2017) , VaDE (Jiang et al., 2017) , DEPICT (Ghasedi Dizaji et al., 2017) , IMSAT (Hu et al., 2017) , and IIC (Ji et al., 2019) .

While (Yang et al., 2017; Xie et al., 2016) focused on learning representations for clustering, other approaches (Jiang et al., 2017; Dilokthanakul et al., 2016; Hu et al., 2017; Ji et al., 2019; Ben-Yosef & Weinshall, 2018) rely on a variational autoencoder that utilizes a Gaussian mixture prior.

SpectralNet , takes a deep learning approach to spectral clustering that generalizes to unseen data points.

Table 1 shows the results reported for these six methods.

Similar to , for MNIST and Reuters we use publicly available and pre-trained autoencoders 1 .

The autoencoder used to map the Reuters data to code space was trained based on a random subset of 10,000 samples from the full dataset.

Similar to (Hu et al., 2017) , for CIFAR-10 and CIFAR-100 we applied 50-layer pre-trained deep residual networks trained on ImageNet to extract features and used them for clustering.

We use two commonly used measures, the unsupervised clustering accuracy (ACC), and the normalized mutual information (NMI) in (Cai et al., 2011) to evaluate the accuracy of the clustering.

Both ACC and NMI are in [0, 1], with higher values indicating better correspondence the clusters and the true labels.

Note that true labels never used neither in training, nor in test.

Clustering Accuracy (ACC): For data points X = {x 1 , . . .

x n }, let l = (l 1 , . . .

l n ) and c = (c 1 , . . .

c n ) be the true labels and predicted clusters respectively.

The ACC is defined as:

where is the collection of all permutations of 1, . . .

g. The optimal permutation π can be computed using the Kuhn-Munkres algorithm (Munkres, 1957) .

Normalized Mutual Information (NMI): Let I(l; c) be the mutual information between l and c, and H(.) be their entropy.

The NMI is:

For each dataset we trained a Siamese network (Hadsell et al., 2006; Shaham & Lederman, 2018) to learn embeddings which represents the affinity of data points by only considering the k-nearest neighbors of each data.

In Table 1 , we compare clustering performance across four benchmark datasets.

Since most of the clustering approaches do not generalize to unseen data points, all data has been used for the training (Later in Section 5.4, to evaluate the generalizability we use 90%-10% split for training and testing).

While the improvement of CNC is marginal over MNIST, it performs better across other three datasets.

Specifically, over CIFAR-10, CNC outperforms SpectralNet and IIC on ACC by 20.1% and 10.9% respectively.

Moreover, the NMI is improved by 12.3%.

The results over Reuters, and CIFAR-100, show 0.021% and 11% improvement on ACC.

The NMI is also 27% better over CIFAR-100.

The fact that our CNC outperforms existing approaches in most datasets suggests the effectiveness of using our deep learning approach to optimize normalized cuts for clustering.

We performed an ablation study to evaluate the impact of embeddings by omitting this step in Algorithm 1.

We find that on both MNIST and Reuters datasets, adding the embedding step improves the performance, but CNC without embeddings still outperforms SpectralNet without embeddings.

On MNIST, the ACC and NMI are 0.945 and 0.873, whereas with the embeddings, ACC and NMI increase to 0.972 and 0.924 (Table 1) .

Without embeddings, CNC outperforms SpectralNet (with ACC of 0.8 and NMI of 0.814).

On Reuters, the ACC and NMI are 0.684 and 0.428, whereas with the embeddings, ACC and NMI increase to 0.824 and 0.583.

Again, even without embeddings, CNC outperforms SpectralNet (with ACC of 0.605 and NMI of 0.401). ].

CNC is able to find better cuts than the SpectralNet

We further evaluate the generalization ability of CNC by dividing the data randomly to a 90%-10% split and training on the training set and report the ACC and NMI on the test set (Table 2) .

Among seven methods in Table 1 , only SpectralNet is able to generalize to unseen data points.

CNC outperforms SpectralNet in most datasets by up to 21.9% on ACC and up to 10.7% on NMI.

Note that simple arg max over the output of CNC retrieves the clustering assignments while SpectralNet relies on k-means to predict the final clusters.

To evaluate the impact of normalized cuts for the clustering task, we calculate the numerical value of the Normalized cuts (Equation 1) over the clustering results of the CNC and SpectralNet.

Since such calculation over whole dataset is very expensive we only show this result over the test set.

Table 3 shows the numerical value of the Normalized cuts over the clustering results of the CNC and SpectralNet.

As one can see CNC is able to find better cuts than the SpectralNet.

Moreover, we observe that for those datasets that the improvement of the CNC is marginal (MNIST and Reuters), the normalized cuts of CNC are also only slightly better than the SpectralNet, while for the CIFAR-10 and CIFAR-100 that the accuracy improved significantly the normalized cuts of CNC are also much smaller than SpectralNet.

The higher accuracy (ACC in Table 2 ) and smaller normalized cuts ( Table 3 ), verify that indeed CNC loss function is a good notion for clustering task.

As you may see in generalization result (Table 2) , when we reduce the size of the training data to 90% the accuracy of CNC slightly changed in compare to training over the whole data (Table 1) .

Based on this observation, we next investigate how varying the size of the training dataset affects the generalization.

In other words, how ACC and NMI of test data change when we vary the size of the training dataset.

We ran experiment over Routers dataset by dividing the data randomly based on the following data splits: 90%-10%, 70%-30%, 50%-50%, 20%-80%, and 10%-90%.

For example, in 10%-90%, we train CNC over 10% of the data and we report the ACC and NMI of CNC over the 90% test set.

Figure 3 shows how the ACC and NMI of CNC over the test data change as the size of the training data is varied.

For example, when the size of the training data is 90%, the ACC of CNC over the test data is 0.824.

As we expected and shown in Figure 3 the ACC and NMI of CNC increased as the size of the training data is increased.

Interestingly, we observed that with only 10% training data the ACC of CNC is 0.68 which is only 14% lower than the ACC with 90% training data.

Similarly the NMI of CNC with 10% training data is only 18% lower than the NMI with 90% training data.

Here are the details of the CNC model for each dataset.

• MNIST: The Siamese network has 4 layers sized [1024, 1024, 512, 10] with ReLU (Embedding size d is 10).

The clustering module has 2 layers sized [512, 512] with a final gumbel softmax layer.

Batch sized is 256 and we only consider 3 nearest neighbors to find the embeddings and constructing the affinity graph for each batch.

We use Adam with lr = 0.005 with decay 0.5.

Temperature starts at 1.5 and the minimum is set to 0.5.

• Reuters: The Siamese network has 3 layers sized [512, 256, 128] with ReLU (Embedding size d is 128).

The clustering module has 3 layers sized [512, 512, 512] with tanh activation and a final gumbel softmax layer.

Batch sized is 128 and we only consider 3 nearest neighbors to find the embeddings and constructing the affinity graph for each batch.

We use Adam with lr = 1e-4 with decay 0.5.

Temperature starts at 1.5 and the minimum is set to 1.0.

• CIFAR-10: The Siamese network has 2 layers sized [512, 256] with ReLU (Embedding size d is 256).

The clustering module has 2 layers sized [512, 512] with tanh activation and a final gumbel softmax layer.

Batch sized is 256 and we only consider 2 nearest neighbors to find the embeddings and constructing the affinity graph for each batch.

We use Adam with lr = 1e-4 with decay 0.1.

Temperature starts at 2.5 and the minimum is set to 0.5.

• CIFAR-100: The Siamese network has 2 layers sized [512, 256] with ReLU (Embedding size d is 256).

The clustering module has 3 layers sized [512, 512, 512] with tanh activation and a final gumbel softmax layer.

Batch sized is 1024 and we only consider 3 nearest neighbors to find the embeddings and constructing the affinity graph for each batch.

We use Adam with lr = 1e-3 with decay 0.5.

Temperature starts at 1.5 and the minimum is set to 1.0.

Hyper-parameter Sensitivity: We train the CNC on the Reuters dataset by fixing some hyperparameters and varying others.

We noticed that CNC benefits from tuning the number of hidden layers (hl), learning rate (lr), batch size (b), and the number of nearest neighbors (k), but is not particularly sensitive to any of the other hyper-parameters, including decay rate, patience parameter (cadence of epochs where decay is applied), Gumbel-Softmax temperature or minimum temperature (Figure 4 ).

More precisely, we varied decay rate over the range [0.1-1.0], patience from epochs, Gumbel-Softmax temperature from [1.0-2.0], and minimum temperature from [0.5-1.0].

When we fix hl=3, lr=5e-5, b=64, and k=3, the average accuracy is 0.803 ± 2e − 3.

With hl=3, lr=5e-4, b=512, and k=10, the average accuracy is 0.811 ± 2e − 3.

With hl=3, lr=5e-4, b=128, and k=3, the average accuracy is 0.821 ± 4e − 3.

With hl=2, lr=1e-4, b=256, and k=3, the average accuracy is 0.766 ± 9e − 4.

And finally with hl=4, lr=1e-5, b=512, and k=3, the average accuracy is 0.766 ± 7e − 3.

As one can see, the accuracy varied from 0.766 to 0.821.

We propose CNC (Clustering by learning to optimize Normalized Cuts), a framework for learning to cluster unlabeled examples.

We define a differentiable loss function equivalent to the expected normalized cuts and use it to train CNC model that directly outputs final cluster assignments.

CNC achieves state-of-the-art results on popular unsupervised clustering benchmarks (MNIST, Reuters, CIFAR-10, and CIFAR-100 and outperforms the strongest baselines by up to 10.9%.

CNC also enables generation, yielding up to 21.9% improvement over SpectralNet , the previous best-performing generalizable clustering approach.

Table 4 : Generalization results: CNC is trained on VGG and validated on MNIST-conv.

During inference, the model is applied to unseen TensorFlow graphs: ResNet.

Inception-v3, and AlexNet.

The ground truth for AlexNet is Bal = 99%, Cut = 4.6%, for Inception-v3, is Bal = 99%, Cut = 3.7%, and for ResNet is Bal = 99% and Cut = 3.3%.

GraphSAGE-on generalizes better than the other models.

To show that CNC generalizes effectively on unseen graphs, we train CNC on a single TensorFlow graph, VGG, and validate on MNIST-conv.

During inference, we test the trained model on unseen TensorFlow graphs: AlexNet, ResNet, and Inception-v3.

We consider the best quality result among hMETIS, KaFFPa, and KaFFPaE as the ground truth.

The ground truth for AlexNet is Bal = 99%, Cut = 4.6%, for Inception-v3, is Bal = 99%, Cut = 3.7%, and for ResNet is Bal = 99% and Cut = 3.3%.

Table 4 shows the result of our experiments, and illustrates the importance of graph embeddings in generalization.

The operation type (such as Add, Conv2d, and L2loss in TensorFlow) is used as the node feature as a one-hot.

We leverage GCN (Kipf & Welling, 2017) and GraphSAGE (Hamilton et al., 2017) to capture similarities across graphs.

In GraphSAGE-on both node embedding and clustering modules are trained jointly, while in GCN and GraphSAGE-off, only the clustering module is trained.

Table 4 shows that the GraphSAGE-on (last row) achieves the best performance and generalizes better than the other models.

Note that this model is trained on a single graph, VGG with only 1325 nodes, and is tested on AlexNet, ResNet, and Inception-v3 with 798, 20586, and 27114 nodes respectively.

On the other hand, the ground truth is the result of running different partitioning algorithms on each graph individually.

In this work, our goal is not to beat the existing graph partitioning algorithms which involve a lot of heuristics on a given graph.

Our generalization results show promise that rather than using heuristics, CNC is able to learn graph structure for generalizable graph partitioning.

Model Architecture and Hyper-parameters: The details of the model with the best performance (GraphSAGE-on) are as follows: the input feature dimension is 1518.

GraphSAGE has 5 layers sized 512 with shared pooling, and the graph clustering module has 3 layers sized 64 with a final softmax layer.

We use ReLU, Xavier initialization (Glorot & Bengio, 2010) , and Adam with lr = 7.5e-5.

<|TLDR|>

@highlight

We introduce a novel end-to-end approach for learning to cluster in the absence of labeled examples. We define a differentiable loss function equivalent to the expected normalized cuts.