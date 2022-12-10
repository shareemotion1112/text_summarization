In this paper, we propose a method named Dimensional reweighting Graph Convolutional Networks (DrGCNs), to tackle the problem of variance between dimensional information in the node representations of GCNs.

We prove that DrGCNs can reduce the variance of the node representations by connecting our problem to the theory of the mean field.

However, practically, we find that the degrees DrGCNs help vary severely on different datasets.

We revisit the problem and develop a new measure K to quantify the effect.

This measure guides when we should use dimensional reweighting in GCNs and how much it can help.

Moreover, it offers insights to explain the improvement obtained by the proposed DrGCNs.

The dimensional reweighting block is light-weighted and highly flexible to be built on most of the GCN variants.

Carefully designed experiments, including several fixes on duplicates, information leaks, and wrong labels of the well-known node classification benchmark datasets, demonstrate the superior performances of DrGCNs over the existing state-of-the-art approaches.

Significant improvements can also be observed on a large scale industrial dataset.

Deep neural networks (DNNs) have been widely applied in various fields, including computer vision (He et al., 2016; Hu et al., 2018) , natural language processing (Devlin et al., 2019) , and speech recognition (Abdel-Hamid et al., 2014) , among many others.

Graph neural networks (GNNs) is proposed for learning node presentations of networked data (Scarselli et al., 2009) , and later be extended to graph convolutional network (GCN) that achieves better performance by capturing topological information of linked graphs (Kipf & Welling, 2017) .

Since then, GCNs begin to attract board interests.

Starting from GraphSAGE (Hamilton et al., 2017) defining the convolutional neural network based graph learning framework as sampling and aggregation, many follow-up efforts attempt to enhance the sampling or aggregation process via various techniques, such as attention mechanism , mix-hop connection and adaptive sampling .

In this paper, we study the node representations in GCNs from the perspective of covariance between dimensions.

Suprisingly, applying a dimensional reweighting process to the node representations may be very useful for the improvement of GCNs.

As an instance, under our proposed reweighting scheme, the input covariance between dimensions can be reduced by 68% on the Reddit dataset, which is extremely useful since we also find that the number of misclassified cases reduced by 40%, compared with the previous SOTA method.

We propose Dimensional reweighting Graph Convolutional Networks (DrGCNs), in which the input of each layer of the GCN is reweighted by global node representation information.

Our discovery is that the experimental performance of GCNs can be greatly improved under this simple reweighting scheme.

On the other hand, with the help of mean field theory (Kadanoff, 2009; Yang et al., 2019) , this reweighting scheme is also proved to improve the stability of fully connected networks, provding insight to GCNs.

To deepen the understanding to which extent the proposed reweighting scheme can help GCNs, we develop a new measure to quantify its effectiveness under different contexts (GCN variants and datasets).

Experimental results verify our theoretical findings ideally that we can achieve predictable improvements on public datasets adopted in the literature over the state-of-the-art GCNs.

While studying on these well-known benchmarks, we notice that two of them (Cora, Citeseer) suffer from duplicates and feature-label information leaks.

We fix these problems and offer refined datasets for fair comparisons.

To further validate the effectiveness, we deploy the proposed DrGCNs on A* 1 company's recommendation system and clearly demonstrate performance improvements via offline evaluations.

2 DRGCNS: DIMENSIONAL REWEIGHTING GRAPH CONVOLUTIONAL NETWORKS

Notations.

We focus on undirected graphs G = (V, E, X), where V = {v i } represents the node set, E = {(v i , v j )} indicates the edge set, and X stands for the node features.

For a specific GCN layer, we use R in = (r in 1 , ..., r in n ) to denote the input node representations and R out = (r out 1 , ..., r out n ) to symbolize the output representations.

2 For the whole layer-stacked GCN structure, we use H 0 to denote the input node representation of the first layer, and H l (l > 0) to signify the output node representation of the l th layer, which is also the output representation of the (l − 1) th layer.

Let A be the adjacency matrix with a ij = 1 when (v i , v j ) ∈ E and a ij = 0 otherwise.

Given the input node set V, the adjacency matrix A, and the input representations R in , a GCN layer uses such information to generate output representations R out :

where σ is the activation function.

Although there exist non-linear aggregators like the LSTM aggregator (Hamilton et al., 2017) , in most GCN variants the aggregator is a linear function which can be viewed as a weighted sum of node representations among the neighborhood (Kipf & Welling, 2017; , followed by a matrix multiplication on a refined adjacency matrixÃ, with a bias added.

The procedure can be formulated as follows:

where W is the projection matrix and b denotes the bias vector.

Development on GCNs mainly lies in different ways to generateÃ. GCN proposed some variants including simply takingÃ = AD −1 , which is uniform average among neighbors with D being the diagonal matrix of the degrees, or weighted by degree of each nodeÃ = D Other methods include attention , or gated attention , or even neural architecture search methods (Gao et al., 2019) to generateÃ. To improve scalability, some GCN variants contain a sampling procedure, which samples a subset of the neighborhood for aggregation .

We can set all unsampled edges to 0 inÃ in sampling-based GCNs, in this caseÃ even has some randomness.

Given input node representations of a GCN layer R in , the proposed DrGCN tries to learn a dimensional reweighting vector s = (s 1 , ..., s d ), where s i is an adaptive scalar for each dimension i.

This reweighting vector s then helps reweighting each dimension of the node representation r

We define S as the diagonal matrix with diagonal entries corresponding to the components of s. Then a DrGCN layer can be formulated as:

Inspired by SENet (Hu et al., 2018) , we formulate the learning of the shared dimensional reweighting vector s in two stages.

First we generate a global representation r in , whose value is the expectation of r in v on the whole graph.

Then we feed r in into a two-layer neural network structure to generate s of the same dimension size.

Equation (5) denotes the procedure to generate s given node weight {w v |v ∈ V, v∈V w v = 1} and node representations {r in v |v ∈ V}:

where g is output of the first layer; W g , b g , W s , and b s are parameters to be learnt.

Figure 1 summarizes the dimensional reweighting block (Dr Block) in DrGCNs.

Combining With Existing GCN variants.

The proposed Dr Block can be implemented as an independent functional process and easily combined with GCNs.

As shown in equation (4), Dr Block only applies on R in and does not involve in the calculation ofÃ, W and b. Hence, the proposed Dr Block can easily be combined with existing sampling or aggregation methods without causing any contradictions.

In § 4, we will experimentally test the combination of our Dr Block with different types of sample-and-aggregation GCN methods.

Suppose that the input features are X, DrGCNs can be viewed as follows:

where H 0 = X and H k being the output representation for a k-layer DrGCN:

Complexity of Dr Block.

Consider a GCN layer with a input and b output channels, n nodes and e edges in a sampled batch, the complexity of a GCN layer is O(abn+be).

The proposed Dr block has a complexity of O(ag), where g is the dimension of g. In most cases, we have g < b and n >> 1, so we could have O(ag) = o(abn + be), which indicates that Dr block introduces negligible extra computational cost.

In this section, we connect our study to mean field theory (Yang et al., 2019) .

We theoretically prove that the proposed Dr Block is capable of reducing the learning variance brought by perturbations on the input, making the update more stable in the long run.

To deepen the understanding, we further develop a measure to quantify the stability gained by Dr block.

( Lee et al., 2018; Yang et al., 2019) employ mean field approximation to analyze fully-connected networks.

Following their ideas, we provide theoretical analyses for Dr Blocks on fully-connected networks.

GCNs are different from fully-connected networks only inÃ, and degrade to fullyconnected networks whenÃ = I, our idea is to provide insight to GCNs from the analysis of fully-connected networks.

We assume the average of the data is 0 in the following discussions as transformation does not affect the covariance structure.

For simplicity, we only consider neural networks with constant width and assume all layers use the same activation function φ.

We follow the pre-activation recurrence relation h Yang et al., 2019) to facilitate the problem.

When S being a diagonal matrix with positive entries, and φ is ReLU activation, φ(SH) = Sφ(H) holds for all H.

We can take the pre-activation step as the post-activation step of the previous layer and generalize our analysis to post-activation.

So the recursive relation is:

We apply the mean field approximation to replace the input data by a normal random variable with the same mean and variance and define an estimator V φ to characterize the fluctuation of the random variable φ(S l H).

Define:

In this equation H ∼ N (0, C l−1 ), φ is the ReLU activation, and C l represents the covariance matrix of H l , i.e. C l = Cov(H l ).

Note that V φ does not completely coincide with variance, but can reflect the covariance structure.

We call it covariance matrix for convenience.

With mean field approximation, the covariance matrix can be updated by (note that b l is independent with H):

We assume this dynamical system has a BSB1(Block Symmetry Breaking 1 (Yang et al., 2019)) fixed point (where C l = C l−1 ) , i.e. a solution having the form C * = q * ((1 − c * )I + c * 1 1 T ) of the following equation with respect to C:

Next we make a reduction of V φ so that only the second slot matters.

Take S l h as a whole, it follows

Thus equation (9) can be written as:

The derivative of C l measures the fluctuation of our updating algorithm when input perturbations exist, hence it characterizes the sensitivity of the algorithm to the data structures and its robustness.

We will turn to show that Dr can relieve this sensitivity.

We fix the point C l where we are taking derivative at.

For most common types of activation functions, this recursive map has a fixed point, at which this linearization is most useful.

Recall that such a derivative will be a linear map from symmetric matrices to symmetric matrices.

J

Here C 1 could be intuitively understood as the increment near C l .

We denote by H d the space of symmetric matrices of size d × d. Using these notations, we prove that:

for any fixed general C. By general, we mean there exists a Haar measure on the collection of symmetric matrices H B with respect to which the statement fails has measure zero 3 .

Detailed proofs and explanations are included in the Appendix G,H. For symmetric matrices, with λ i denoting eigenvalues of A, we have:

This norm measures the overall magnitude of eigenvalues of the operator.

This result demonstrates that our method brings variance reduction and improves the stability of the updating algorithms.

To summarize, for any input data, there exists a vector s that improves the stability of the updating algorithms.

In this section we turn to define a quantified measurement of the improvement of the stability of DrGCNs.

Define:

where c ij is the (i, j) th element of the convariance matrix C. Theorem 2 suggests that K measures the instability of the update.

The measure is a relative covariance measure that when S = I (without Dr), K = 1.

This quantity only involves entries of C, and it is homogeneous of degree 0 with respect to these entries and invariant under scalar multiplication on these entries.

Being the covariance 3 || · ||F is the Frobenius norm, i.e. for a matrix A = (Aij), ||A||

matrix of H, C does not change under the mean zero case of H. Consequently, we could proceed our analyses under the dimensional normalized assumption without loss of generality.

We turn to consider the dimensional normalized version of V φ by replacing φ with d φ , which is φ with normalization:

where

Near the fixed point C * of V d φ , the exponential growth rate of the deviation of C l from C * is proportional to K.

Here C * is used to denote the BSB1 fixed points of

We use the following definitions to simplify notations.

Define:

Theorem 3.6 of (Yang et al., 2019) tells us the derivative of V d φ (I, C) (as a linear map) has a very explicit eigenspace decomposition, we describe it in Theorem 3.

A simple reflection suggests that our linear operators still satisfy the DOS condition and the ultrasymmetry condition needed in the proof of this theorem, so this decomposition still holds.

dC at C * has eigenspaces and eigenvalues:

So an appropriately chosen S can reduce the proportion that lies in V G .

We prove that the Frobenius norm of the component in V G is proportional to K in Appendix H. Thus, it is natural to consider the orthogonal (in terms of Frobenius norm and corresponding inner product) eigendecomposition (with subindices indicating the corresponding eigenspaces we listed above):

The effect of Dr is to reduce the RG−component at each step to make the dynamic system more stable.

Since the decomposition is orthogonal, this is equivalent to reducing

recall that

Since we take the normalization assumption, only the relative magnitude of s li matters, and we can put any homogeneous restriction.

In order to include the case s li = 1, we consider the restriction

Finally we come to the effectiveness measure of the proposed Dr Block.

4 1 is the d−dimensional vector with all component 1.

5 All results involve the BSB1 fixed point (Yang et al., 2019) require permutation, diagonal and off-diagonal symmetry, and hold for dimensional normalization, too.

6 G, S above are symmetric, so the transpose is only introduced for the sake of notational balance The denominator is chosen to display the ratio of variance reduction for the proposed Dr Block.

Without Dr, s li = 1 for all i, and we have K = 1.

From our calculation on the inner product, it can be discovered that this quantity is proportional to the part in V G in the orthogonal decomposition, this proves Theorem 2.

Since this is the only part for J d φ with eigenvalue larger than 1, the exponential growth rate is proportional to this quantity.

Therefore, this quantity measures the magnitude of improvement Dr Blocks make to the stability of the learning process under perturbation.

In this section, we evaluate the proposed DrGCNs on a variety of datasets compared to several SOTA methods.

Detailed descriptions of the experiments and datasets are included in Appendix C,D.

Datasets We present the performance of DrGCNs on several public benchmark node classification datasets, including Pubmed (Yang et al., 2016), Reddit, PPI (Hamilton et al., 2017) .

We also conduct experiments on a large-scale real-world commercial recommendation A* dataset.

Table 1 summarizes statistics of the datasets.

There are also two widely adopted Cora and Citeseer datasets for citation networks.

We investigate the originality of these datasets not only from the public data provided by but also from much earlier versions (McCallum et al., 2000; Lu & Getoor, 2003) and find problems in those datasets.

32(1.2%) samples in Cora and 161(4.8%) samples in Citeseer are duplicated, while 1,145(42.3%) samples in Cora and 2,055(61.8%) samples in Citeseer have information leak that includes their labels directly as a dimension of their features.

To address such problems, we remove duplicated samples, modify their features using word and text embeddings to reduce information leak, and construct two refined datasets, CoraR and CiteseerR. For details of these refined datasets and A* dataset, please refer to Appendix A,B,E.

Competitors We compare the results of our DrGCN with GCN (Kipf & Welling, 2017) , GAT (Veličković et al., 2018) , MixHop (Abu-El-Haija et al., 2019), GraphSAGE (Hamilton et al., 2017) , FastGCN and ASGCN on citation networks including CoraR, CiteseerR and Pubmed.

We also provide results on the original Cora and Citeseer dataset in Appendix C. Our DrGCN also works for inductive datasets that we evaluate the Dr-GAT with several state-of-the-art methods on the PPI dataset, including GraphSAGE (Hamilton et al., 2017) ,

LGCN GeniePath (Liu et al., 2019) .

As for A* dataset, we compare our method with the company's previous best GraphSAGE model, see Appendix E.

DrGCNs We combine Dr block with five most representative GCN methods and compare with them on public datasets.

Two full GCN methods include the vanilla GCN (Kipf & Welling, 2017) and a variant that exploits an attention aggregator GAT .

Sampling GCN methods contain FastGCN , Adaptive Sampling GCN , and A* company's heterogeneous GraphSAGE model on A* dataset.

Every GCN layer is replaced by a DrGCN layer as in Equation (4).

Further implementation details are covered in Appendix C. Table 2 illustrates the performance of DrGCNs on four transductive datasets when combined with four different variations of GCN models.

Our results are averaged among 20 runs with different random seeds.

Our DrGCNs achieve superior performances on all of the datasets and demonstrate The performance improvements can be explained by our stability measure proposed in Equation (20).

Theoretically, when K ≈ 1, we expect Dr block to have limited ability in refining the representation, while when K 1 we expect the vector to strengthen the stability of the model by reducing the magnitude of the derivatives of the covariance matrix and improve the performance.

To verify the theoretical analyses, we collect the average K-value of the learnt reweighting vectors for different layers in the Dr-ASGCN model, see Table 5 .

The K-value in the second layer is around 1 on all datasets.

However, the K-value for the first layer is around 1 for citation datasets, but 0.32 on the Reddit dataset, which emphatically explains why the DR-ASGCN achieves such a massive improvement on the Reddit dataset.

On the inductive PPI dataset (Table 3) , Dr Block increases the micro f1-score of GAT by 1.5% and outperforms all previous methods.

Table 4 suggests that the Dr method can also accomplish substantial improvements on the real-world, large-scale recommendation dataset.

It demonstrates improvement on industrial measure recall@50, which is the rate of users clicking the top 50 predicted items among 6 million different items within the next day of the training set, from 5.19% (previous best model) to 5.26% (Dr Block added).

We also compare DrGCNs with other feature refining methods, including Batch-Norm (Ioffe & Szegedy, 2015) and Layer-Norm (Lei Ba et al., 2016) .

These methods use variance information on every single dimension to refine representations, while DrGCN joins information on each dimension and learns a reweighting vector S adaptively.

We provide results of DrGCN and these methods on the Reddit dataset for ASGCN ( Table 6 ).

Batch-Norm and Layer-Norm also improve the performance of the ASGCN model on the Reddit dataset.

Combining Dr and Layer-norm yields an even better result for ASGCN on Reddit.

More detailed results are in Appendix F.

The idea of using neural networks to model graph-based data can be traced back to Graph Neural Network (Scarselli et al., 2009 ), which adopts a neural network structure on graph structure learning.

GCN (Kipf & Welling, 2017) proposes a deep-learning-based method to learn node representations on a graph using gathered information from the neighborhood of a node.

GraphSAGE (Hamilton et al., 2017 ) formulated a sample and aggregation framework of inductive node embedding.

The idea of the sample and aggregation framework is to incorporate information from the neighborhood to generate node embeddings.

Despite being uniform when first being proposed, both sampling and aggregation can be weighted.

These methods, including FastGCN , GAT (Veličković , 2019) , treat all nodes in the graph unequally and try to figure out more important nodes and assign them higher weights in sampling and aggregation procedure.

Feature imbalance phenomena have long been aware of. (Blum & Langley, 1997) Different dimensions of the hidden representation generated by neural networks may also share such imbalance behavior.

The idea of refining hidden representations in neural networks can be traced back to Network in Network (Lin et al., 2014), whom proposes a fully-connected neural network to refine the pixel-wise hidden representation before each convolutional layer-known as the 1 × 1 convolution layer which is widely adopted in modern convolutional neural networks.

Squeeze and Excitation Networks (Hu et al., 2018) proposes a dimensional reweighting method called Squeeze and Excitation block, which involves the techniques of global average pooling and encoder-decoder structure.

It works well in computer vision CNNs and wins the image classification task of Imagenet 2017.

The success attracts our concern that such dimensional reweighting methods might also be useful in node representation learning on graphs.

Another natural idea to refine representations of neural networks is normalization.

Batch normalization (Ioffe & Szegedy, 2015 ) is a useful technique in neural networks to normalize and reduce the variance of input representations.

Layer normalization (Lei Ba et al., 2016) is an improved version of BatchNorm, for it also works on a single sample.

Many also try to give theoretical analyses to such normalization techniques. (Kohler et al., 2018) explains the efficiency of batch normalization in terms of convergence rate. (Bjorck et al., 2018) shows that batch normalization enables larger learning rates. (Yang et al., 2019) demonstrates the gradient explosion behaviors of batch normalization on fully-connected networks via mean field theory (Kadanoff, 2009) .

In our approach, we adopt some of these methods and apply them to the analysis of DrGCNs.

We We investigate the originality of the Cora and CiteSeer dataset.

The two datasets are widely used for being light-weighted and easy to handle.

The most popular version is provided by Planetoid (Yang et al., 2016) .

The two datasets are both citation networks where each node represents a research paper, and each edge represents a citation relationship between two papers.

Edges are directed but are usually handled undirectedly by GCN methods.

Each paper belongs to a sub-field in computer science and is marked as its label.

Papers have features of bag-of-word(BOW) vectors that each dimension represents whether the document of the paper contains a particular word in the dictionary or not.

Cora has 2,708 papers with a dictionary size of 1,433, while Citeseer has 3,327 papers with a dictionary size of 3,703.

A.1 CORA Cora originates in (McCallum et al., 2000) 7 with extracted information(including titles, authors, abstracts, references, download links etc.) in plain-text form.

Those download links are mostly unavailable now.

Before they become unavailable, (Lu & Getoor, 2003) 8 extracts a subset of 2,708 papers and assigns labels and BOW feature vectors to the papers.

The dictionary is chosen from words(after stemming) 9 that occur 10 or more times in all papers and result in a dictionary size of 1,433.

Planetoid (Yang et al., 2016) reordered each node to form the benchmark Cora dataset ).

There exist a lot of duplicated papers (one paper appears as multiple identical papers in the dataset) in the original Cora of (McCallum et al., 2000) , and (Lu & Getoor, 2003) inherits the problem of duplicated papers.

In Cora, we find 32 duplicated papers among the 2,708 papers.

Another problem is the information leak.

The generation process of the dictionary chooses words that occur more than 10 times, and does not exclude the label contexts of papers.

Therefore, some papers may be classified easily only by looking at their labels.

For instance, 61.8% of papers labeled "reinforcement learning" contain exactly the word "reinforcement" "learning" in their title and abstract(after stemming).

Altogether 1,145(42.3%) of these papers contain their label as one or some of the dimensions of their features.

CiteSeer is a digital library maintained by PSU(Currently named as CiteSeerX 11 ), which provides a URL based web API for each paper according to their doc-ids. (Lu & Getoor, 2003) 12 extracts 3,327 nodes from the library to form a connected citation network in which 3,312 nodes are associated with a BOW vector of dictionary size 3,703.

The rest 15 nodes' features are padded with zero vectors.

It is also reordered and adopted by Planetoid (Yang et al., 2016) .

Although the original version of Citeseer only consists of links that are unavailable now, a back-up version of Citeseer contains the title and abstract information of 3,186 of the papers.

13 We also find another 81 by author and year information using Google Scholar.

Unfortunately, among the papers we collected from the Citeseer dataset, 161(4.8%) of them are actually duplicated.

Since the Citeseer dataset shares a similar embedding generation method with Cora, there also exists the problem of information leak.

For the data we collected, at least 2, 055(61.8%) of the papers in Citeseer contain their labels in title or abstract, which are sure to become some of the dimensions of their feature representations.

7 https://people.cs.umass.edu/˜mccallum/data/cora.tar.gz 8 https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz 9 Stemming basically transforms a word into its original form.

10 If we include references of the paper the rate becomes 86.1%, see what GCN learns!

11 http://citeseerx.ist.psu.edu 12 https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz 13 https://www.cs.uic.edu/˜cornelia/russir14/lectures/citeseer.txt

Due to the problems in Cora and Citeseer, we produce new datasets CoraR and CiteSeerR to address these issues.

We remove duplicated nodes and generate indirect pre-trained word/text embeddings as node features to reduce the information leak.

We double-check the articles belong to Cora (Lu & Getoor, 2003) dataset in the original Cora (McCallum et al., 2000) .

Among the 2,708 papers, 474 of them have a wrong title that can hardly be found on any academic search engines such as google scholar or aminer (Tang et al., 2008) .

We manually searched for these papers based on other information like author, abstract, references and feature vectors in Cora.

Finally, we figure out that 32 of the total 2,708 papers are duplicated papers that actually belong to 13 identical ones.

9 papers are absolutely missing and not able to trace.

We recover the actual title of the rest 2,680 papers, and use their titles to generate their features.

We apply averaged GLOVE-300 (Pennington et al., 2014) word vector for their titles(with stop words removed) and we add two dimensions expressing the length of the title, and the length of the title with stop words removed.

This leads to a 302 dimension feature representation for each node in CoraR. The average and word embedding process can better reduce the effect of label information leak than using simple BOW vectors.

As in (Hamilton et al., 2017; , we split the 2,680 nodes into a training set of 1,180 nodes, a validation set of 500 nodes and a test set of 1,000 nodes.

We use Cornelia's back-up extractions of CiteSeer and manually find some other documents using author and year information from academic search engines.

For the rest we only have a numerical ID and an invalid download link, so we are not able to trace them.

We check the titles, abstracts and feature vectors and find 161 papers are actually 80 identical papers.

We combine duplicated papers together and also remove 55 papers that we are not able to trace.

Our refined CiteSeerR has 3,191 nodes and 4,172 edges.

We average over the last hidden state of BERT (Devlin et al., 2019) for each sentence from the title and abstract and produce a 768-dimensional feature for each paper.

For pairs of duplicated papers with distinct labels, we manually assign a label for it.

As in (Hamilton et al., 2017; , we split the 3,191 nodes into a training set of 1,691 nodes, a validation set of 500 nodes and a test set of 1,000 nodes.

Both CoraR and CiteseerR are included in the link of code.

For the dimensional reweighting part of each layer, the dimension of the encoder g is set to the closest integer of the square root of the dimension of the input node representation r i .

The pooling weight w v is set to uniform weight w v = 1 |V| , where |V| is the number of nodes in the graph for full GCNs, and the batch size for batch-wise sampling GCNs.

We use ELU activation for σ g and sigmoid activation for σ s .

We do not apply dropout, but apply weight regularization with the same coefficient as the original GCNs in the Dr Blocks of DrGCNs.

We keep all other settings, including learning rate, early stop criteria, loss function, hidden representation dimension, batch size, weight decay, and regularization loss the same as the original models.

All of our methods and compared baseline methods are run 20 times, and we report the average accuracy and variation for methods that we run.

We evaluate their performance mainly based on their released code and paper.

For methods without codes, we use their reported performance if they share the same evaluation settings as ours.

We describe our method implementation details here.

GCN (Kipf & Welling, 2017) We use the GCN code provided in AS-GCN 14 , and use a 2-layer GCN model with 64 hidden units in layer 1.

We run it 20 times and report the average and variance.

For each running time, we use the model that performs best within 800 training epochs on the validation set for testing.

GAT We use the GAT code provided by the authors 15 .

We use the dataset-specific structures described in and early stopping strategy mentioned in the code repo.

The original paper uses a high dropout rate of 0.6 on semi-supervised citation datasets test setting.

We find that for CoraR, CiteseerR and the fully-supervised training set setting on Pubmed, such a high dropout may have a chance to lead the original GAT model not to converge, so we adjusted the dropout rate to 0.35(which gives the best performance among all dropout rates from 0 to 0.6) for both the original and our Dr-GAT.

On PPI we simply follow their instructions and use their suggested structure and hyperparameters.

GAT forms a 2 layer structure for citation datasets.

For Cora and Citeseer, GAT has 8 hidden units in every attention head in the first layer, and 8 attention heads in the first layer and 1 in the second layer, which has the number of hidden units equal to node classes.

We adopt the structure for CoraR and CiteseerR. We also discover that for the fully-supervised training set setting on Pubmed, the structure for Pubmed in GAT paper(which has 8 heads in the second layer) does not perform as good as the GAT structure for Cora and Citeseer (this setting achieves 82.5 ± 0.3% under the best dropout rate), so we also adopt the Cora/Citeseer structure to Pubmed.

For PPI, GAT has a three layer structure, with 256 hidden units in every attention head in the first two layers.

It has 4 attention heads in the first layer and 4 in the second layer, and 6 attention heads in the third layer, each third layer attention head has a number of hidden units equal to node classes.

It also sets dropout equals to 0 and uses residual connection (He et al., 2016) .

As for the industrial A* dataset we use.

It is an item recommendation dataset, with the training set has about 35 million users and 6.3 million items with 120 million edges.

Although the target is node-classification like(to find the most likely items that each user may click), instead of simply taking each item as a class, A* uses a graph embedding model to generate embedding for both users and items.

There are 27 user attributes and 33 item attributes.

For every user, we use K nearest neighbor (KNN) with Euclidean distance to calculate the top-N items that the user is most likely to click, and the customer will see these recommended items in A* company's APP.

We use the recall@N to evaluate the model:

M u represents the top-N items recommended by the model and I u indicates the items clicked by the customer.

The baseline model is the A* online heterogeneous GraphSAGE, and we add Dr block in it to compare Recall@N with the online model.

Recall@50 is the most commonly used metric in A* company.

Experimental results show that we reach 5.264% on Recall@50, improving from the original best model's 5.188%.

It is quite a good result, considering random guess will only give less than 0.001% (50/6,300,000).

In Table 12 we also provide the batch-norm and layer-norm GCN results on publication datasets.

The results are averaged among 20 runs.

We also provide proof for theorem 1 in the main article.

Here u, v, w are constants, and we will call the set of operators with these parameters DOS (u, v, w) .

By the definition of V φ , the (i, j) component of its output will only involve the i − th and j − th components of the input and symmetric with respect to them, hence itself and its derivatives J φ will also involve them only and being symmetric with respect to them.

Thus it is determined by c ii , c ij , c jj .

Furthermore, since J φ is a linear map, so it will have this form.

The result in Theorem 1 should hold in general for DOS operators and do not require information about the fixed point structure.

Now J φ is a DOS operator, hence it will belong to DOS(u, v, w) for some u, v, w.

Then we know:

entries, which have dimension

and a basis M ij = E ij + E ji , where E ij is the matrix with 1 on (i, j) position and 0 anywhere else.

And L d is the span of L i , which is defined as: Proof.

Here the condition w = u ensures that L d is linearly independent with elements in M d since it spans the diagonal part of H d .

The results TM ij = wM ij , TL i = uL i could be calculated using the definition equations of DOS(u, v, w) and consider them on component level.

Since T is a linear operator, verifying these eigen properties on the basis is enough for the result.

Furthermore, the space we have specified spans a

As discussed in section 3, Our theoretical analysis is based on the pre-activation setting, while common GCN methods use post activation.

Although they are basically the same if we consider the activation in the pre-activation setting as the activation of the previous layer in the post-activation setting, there is still a little difference between pre and post activation that pre-activation activates the input feature.

So we also experiment pre-activation on GCN, see table 13.

Results are averaged among 20 runs.

@highlight

We propose a simple yet effective reweighting scheme for GCNs, theoretically supported by the mean field theory.

@highlight

A method, known as DrGCN, for reweighting the different dimensions of the node representations in graph convolutional networks by reducing variance between dimensions.