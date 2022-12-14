Learning good representations of users and items is crucially important to recommendation with implicit feedback.

Matrix factorization is the basic idea to derive the representations of users and items by decomposing the given interaction matrix.

However, existing matrix factorization based approaches share the limitation in that the interaction between user embedding and item embedding is only weakly enforced by fitting the given individual rating value, which may lose potentially useful information.

In this paper, we propose a novel Augmented Generalized Matrix Factorization (AGMF) approach that is able to incorporate the historical interaction information of users and items for learning effective representations of users and items.

Despite the simplicity of our proposed approach, extensive experiments on four public implicit feedback datasets demonstrate that our approach outperforms state-of-the-art counterparts.

Furthermore, the ablation study demonstrates that by using multi-hot encoding to enrich user embedding and item embedding for Generalized Matrix Factorization, better performance, faster convergence, and lower training loss can be achieved.

In the era of big data, we are seriously confronted with the problem of information overload.

Recommender systems play an important role in dealing with such issue, thereby having been widely deployed by social media, E-commerce platforms, and so on.

Among the techniques used in recommender systems, collaborative filtering (Sarwar et al., 2001; Hu et al., 2008; Su & Khoshgoftaar, 2009; Wang et al., 2019) is the dominant one that leverages user-item interaction data to predict user preference.

Among various collaborative filtering methods, Matrix Factorization (MF) is the most popular approach that has inspired a large number of variations (Koren, 2008; Rendle et al., 2009; Xue et al., 2017) .

MF aims to project users and items into a shared latent space, and each user or item could be represented by a vector composed by latent features.

In this way, the user-item interaction score could be recovered by the inner product of the two latent vectors.

Most of the existing extensions of MF normally focus on the modeling perspective (Wang et al., 2015; and the learning perspective (Xue et al., 2017; .

For example, BPR-MF (Rendle et al., 2009 ) learns user embedding and item embedding from implicit feedback by optimizing a Bayesian pairwise ranking objective function.

NeuMF learns compact embeddings by fusing the outputs from different models.

DeepMF (Xue et al., 2017) employs deep neural networks to learn nonlinear interactions of users and items.

Although these approaches have achieved great success, they still cannot resolve the inherent limitation of MF.

Specifically, apart from the interaction by inner product, there are no explicit relationships between user embedding and item embedding.

In other words, the connection between user embedding and item embedding is only weakly enforced by fitting the given individual rating value.

However, in real-world scenarios, user embedding and item embedding may be interpreted as some high-level descriptions or properties of user and item, which are supposed to have some explicit connections.

For example, a user likes some item, probably because the user and the item share some similar high-level descriptions or properties.

Which means, the latent features of a user could be potentially enriched by taking into account the latent features of the user's interacted items, since these interacted items could expose the latent features of the user to some degree.

Similarly, the latent features of an item may also be enriched by the latent features of the item's interacted users.

However, most of the existing approaches regrettably ignore such useful information.

An exception is the SVD++ model (Koren, 2008) , which provides each user embedding with additional latent features of items that the user has interacted with.

Despite the effectiveness of the SVD++ model, it suffers from two major problems.

First, it only enriches user embedding, and ignores the fact that item embedding could also be enriched by the latent features of users that the item has interacted with.

Second, the latent features of the interacted items are averagely integrated without discrimination, while each user normally has different preferences on different items.

Note that there are also other approaches that regularize or enrich user embedding and item embedding by exploiting supplementary information, such as social relations (Ma et al., 2011; Guo et al., 2017 ) and text reviews (Chen et al., 2018) .

However, in this paper, we do not assume there is any supplementary information, and only focus on the data with implicit feedback.

Motivated by the above observations, this paper proposes a novel Augmented Generalized Matrix Factorization (AGMF) approach for learning from implicit feedback.

Different from existing approaches, AGMF aims to enrich both user embedding and item embedding by applying multi-hot encoding with the attention mechanism on historical items and users.

In this way, user embedding and item embedding are explicitly related to each other, which could further improve the performance of the Generalized Matrix Factorization (GMF) model.

Extensive experimental results clearly demonstrate our contributions.

In this section, we first present the problem statement and then briefly introduce the basic MF model, the GMF model, and the SVD++ model.

Let U = {1, 2, ?? ?? ?? , U } be the set of U users, and I = {1, 2, ?? ?? ?? , I} be the set of I items.

We define the given user-item interaction matrix Y = [y ui ] U ??I from implicit feedback data as: y ui = 1 if the interaction (u, i) is observed, otherwise y ui = 0.

It is worth noting that y ui = 1 indicates that there is an observed interaction between user u and item i, while it does not necessarily mean that u likes i.

In addition, y ui = 0 does not necessarily mean that u does not like i, and it is possible that u is not aware of i. Such setting could inevitably bring additional challenges for learning from implicit feedback, since it may provide misleading information about user's preference.

The goal of recommendation with implicit feedback is to predict the values of the unobserved entries in Y, which can be further used to rank the items.

MF is a basic latent factor model, which aims to characterize each user and item by a real-valued vector of latent features (Koren et al., 2009) .

Let p u and q i be the latent vectors for user u and item i, respectively.

MF tries to give an estimation?? ui of y ui by the inner product of p u and q i :

where K is the dimension of the latent vectors.

As can be seen, the latent features could be considered as linearly combined in MF.

Hence MF can be regarded as a linear model with respect to latent features.

This linear property of MF restricts its performance to some degree.

As a result, there are an increasing number of approaches (Xue et al., 2017; proposed to alleviate this problem, by learning a nonlinear interaction function using deep neural networks.

Generalized Matrix Factorization (GMF) ) is a simple nonlinear generalization of MF, which makes a prediction?? ui of y ui as follows:

where denotes the element-wise product of vectors, h is a weight vector, and ??(??) is an activation function.

To show that MF is a special case of GMF, we can simply set h = 1 where 1 is the vector with all elements equal to 1.

In this way, apart from the activation function, the MF model is exactly recovered by GMF, since p u q i = 1 (p u q i ).

SVD++ extends MF by leveraging both explicit ratings and implicit feedback to make prediction:

where N(u) denotes the set that stores all the items for which u has provided implicit feedback, and c j is a latent vector of item j for implicit feedback, while p u and q i are free user-specific and itemspecific latent vectors specially learned for explicit ratings.

The only difference between SVD++ and MF lies in that p u is enriched by |N(u)|

3 THE PROPOSED APPROACH Figure 1 illustrates the framework of our AGMF model.

It is worthy noting that for the input layer, unlike most of the existing approaches (Koren et al., 2009; ) that only employ onehot encoding on the target user's ID (denoted by u) and the target item's ID (denoted by i), we additionally apply multi-hot encoding on user u's interacted items, and item i's interacted users.

In this way, potentially useful information is incorporated, which could enrich the embedding of u and i. Note that this part is the core design of our proposed AGMF model.

By enriching the one-hot encoding with multi-hot encoding, the historical interactions between users and items are exploited, therefore our AGMF model with multi-hot encoding achieves superior performance to the GMF model with only one-hot encoding.

We argue that this design is simple but more advantageous in threefold.

First, it is an augmented version of the GMF model that only takes into account the one-hot embedding of the target user and the target item.

Second, it encodes more information than the GMF model, by considering numerous historical interactions of users and items.

Such natural information is valuable and should not be ignored for recommendation with only implicit feedback.

Third, different from most matrix factorization based recommendation models that only relates user embedding and item embedding by fitting the given individual value, our AGMF model joint learns user embedding and item embedding in a more explicit way.

In order to avoid the conflict that user u may overly concentrate on item i if the target item is i, we will exclude i from N(u) (denoted by N(u)\{i}) when predicting?? ui .

Similarly, we will also exclude u from N(i) (denoted by N(i)\{u}) when predicting?? ui .

In what follows, we detail elaborate the design of our AGMF model layer by layer.

Given the target user u, the target item i, user u's interacted items N(u), and item i's interacted users N(i), we not only apply one-hot encoding on u and i, but also apply multi-hot encoding on N(u) and N(i).

In this way, u and i are projected to latent feature vectors p u ??? R K and q i ??? R K .

Similarly, for each historical item j ??? N(u)\{i} and each historical user k ??? N(i)\{u}, we can obtain {q j ??? R K |j ??? N(u)\{i}} and {p k ??? R K |k ??? N(i)\{u}}.

Following the interaction way used in GMF, we also apply the widely-used element-wise product Cheng et al., 2018; Xue et al., 2018) to model the interactions of u and N(u) as well as i and N(i), Generally, interaction ways such as p u + q j , p u ??? q j , or any other function that integrates two vectors into a single vector, can also be used.

Here, we choose element-wise product because it generalizes inner product to vector space, which could retrain the signal of inner product to a great extent.

Since there are multiple historical items of user u and multiple historical users of item i, how to extract useful information from these generated latent vectors is crucially important.

In reality, the historical items of the target user u normally make different contributions to u on decision of the target item i.

The same situation holds for the target item i while interacting with the target user u.

Therefore, we perform a weighted sum on the latent vectors obtained from the pairwise interaction layer, i.e.,

where a u (u, j) denotes the attention weight that the target user u on its interacted item j, and a i (i, k) is the attention weight that the target item i on its interacted user k. Note that a u (u, i) = a i (i, u) does not necessarily hold, as you are my best friend, while I may not be your best friend.

p u and q i are the supplementary latent vectors generated by the pooling layer, which will be used to enrich p u and q i .

In this paper, we define a u (u, j) (???j ??? N u \{i}) and a i (i, k) (???k ??? N i \{u}) as the softmax normalization of the interaction scores between users and items:

where f u (??) (f i (??)) is the user (item) attention model that takes the user-item interaction vector as an input, and outputs the corresponding interaction score.

In this paper, we define f u (??) and f i (??) as:

where h u and h i are the weight vectors of the user attention model and the item attention model, respectively.

Note that unlike existing approaches that normally take multi-layer neural networks as the attention model, we only use a single-layer perceptron.

In this way, our proposed attention model is exactly a standard GMF model.

Our experimental results show that such simple GMF model can achieve satisfactory performance, with keeping simple and efficient.

While deeper structures could potentially achieve better performance, we leave the exploration of deeper structures for attention modeling in future work.

With the supplementary latent vectors p u and q i for p u and q i , inspired by SVD++, we represent the latent vector of user u by p u + p u , and represent the latent vector of item i by q i + q i .

Then we reuse the GMF model as the prediction model, and the predicted interaction score?? ui is given by:??

where h is the weight vector of the prediction model.

We empirically use the sigmoid function ??(x) = 1/(1 + exp(???x)) as the activation function throughout this paper.

Since this paper focuses on learning from implicit feedback data, the output?? ui of our AGMF model is constrained in the range of [0, 1], which could provide a probability explanation.

In such setting, the commonly used Binary Cross Entropy (BCE) loss could be employed:

where O + denotes the observed interactions and O ??? denotes the set of negative instances that could be sampled from unobserved interactions.

In this paper, for each training epoch, we randomly sample four negative instances per positive instance.

In this section, we conduct extensive experiments to demonstrate that our AGMF model outperforms state-of-the-art counterparts.

In addition, we provide ablation study to clearly demonstrate the importance of multi-hot encoding for generalized matrix factorization.

We conduct experiments on four publicly available datasets: MovieLens 1M (ML-1M) 1 , Yelp 2 , Amazon Movies and Tv (Movies&Tv) 3 , and Amazon CDs and Vinyl (CDs&Vinyl).

For ML-1M, we directly use the original dataset downloaded from the MovieLens website.

Since the high sparsity of the original dataset makes it much difficult to evaluate recommendation approaches, we follow the common practice (Rendle et al., 2009; He et al., 2016) to process the other three datasets.

For the Yelp dataset, we filter out the users and items with less than 10 interactions (He et al., 2016) .

For Movies&Tv and CDs&Vinyl, we filter out the users that have less than 10 interactions.

As this paper focuses on the data with implicit feedback, we mask all the data with explicit feedback to have only implicit feedback by marking each entry 0 or 1, which indicates whether the user has interacted the item.

The main characteristics of these datasets are provided in Table 1 .

We compare AGMF with the following state-of-the-art approaches:

??? SVD++ (Koren, 2008) It merges the latent factor model and the neighborhood model by enriching the user latent feature with the interacted items' latent features.

??? BPR-MF (Rendle et al., 2009 ) It trains the basic MF model by optimizing the Bayesian personalized ranking loss.

??? FISM (Kabbur et al., 2013) It is an item-based approach, which factorizes the similarity matrix into two low-rank matrices.

??? MLP It learns the interactions between users and items by multi-layer perceptron.

??? GMF It generalizes the basic MF model to a non-linear setting.

??? NeuMF-p NeuMF is a combination of MLP and GMF, and its pretraining version is called NeuMF-p.

In this paper, we compare with NeuMF-p, as NeuMF-p provides better performance than NeuMF without pretraining ).

??? ConvNCF It employs a convolutional neural network to learn high-order interactions based on the interaction map generated by the outer product of user embedding and item embedding.

We randomly holdout 1 training interaction for each user as the development set to tune hyperparameters suggested by respective literatures.

Unless otherwise specified, for all the algorithms, the learning rate is chosen from [5e ???5 , 1e ???4 , 5e ???4 , 1e ???3 , 5e ???3 ], the embedding size K is chosen from [16, 32, 64, 128] , the regularization parameter (that controls the model complexity) is chosen from [1e ???5 , 5e ???6 , 1e ???5 , 5e ???5 ], and the batch size is set to 256.

For MLP and NeuMF-p that have multiple fully connected layers, we follow the tower structure of neural networks , and tune the number of hidden layers from 1 to 3.

For ConvNCF 4 , we follow the configuration and architectures proposed in .

All the models are trained until convergence or the default maximum number of epochs (by respective literature) is reached.

For our proposed AGMF model, no neural network is adopted, hence we do not need to tune the network structure.

We initialize the weight vectors by the Xavier initialization (Jia et al., 2014) , and initialize the embedding vectors using a uniform distribution from 0 to 1.

For training AGMF, we employ the Adaptive Moment Estimation (Adam) (Kingma & Ba, 2014) , which adapts the learning rate for each parameter by performing small updates for frequent parameters and large updates for infrequent parameters.

We implement AGMF using PyTorch 5 , and the source code as well as the used datasets are released.

We fix the embedding size at 128, since we found that a larger embedding size always performs better.

Note that the number of interacted users or items may be very large, to mitigate this issue, we truncate the list of interacted users and items such that the latent representation of each user/item is enriched by the latent vectors of at most 50 latest interacted items/users.

In this paper, we adopt the widely used leave-one-out evaluation method (Rendle et al., 2009; He et al., 2016; Bayer et al., 2017) to compare AGMF with other approaches.

Specifically, for each dataset, we holdout the latest interaction of each user as the test positive examples, and randomly select 99 items that the user has not interacted with as the test negative examples.

In this way, all the algorithms make ranking predictions for each user based on these 100 user-item interactions.

To evaluate the ranking performance, we adopt two widely used evaluation criteria, including Hit Ratio (HR) and Normalized Discount Cumulative Gain (NDCG).

HR@k is a recall-based metric that measures whether the testing item is on the top-k list, and NDCG@k assigns higher scores to the items with higher positions within the top-k list .

Table 2 and Table 3 show the top-k performance of all the algorithms when k = 5 and k = 10, respectively.

From the two tables, we can observe that:

??? AGMF achieves the best performance (the highest HR and NDCG scores) on the four datasets.

??? Although AGMF is a simple extension of GMF, it still outperforms the complex state-ofthe-art approaches NeuMF-p and ConvNCF.

??? Compared with GMF, AGMF achieves significantly better performance.

Such success owes to multi-hot encoding with the attention mechanism, which provides enriched information for user embedding and item embedding.

As aforementioned, the GMF model makes prediction by?? ui = ??(h (p u q i )), while our AGMF model makes prediction by?? ui = ??(h ((p u + p u ) (q i + q i ))).

Clearly, without the supplementary latent vectors p u and q i , AGMF reduces to GMF.

Table 2 and Table 3 have clearly showed that AGMF significantly outperforms GMF.

While the used GMF model for performance evaluation is provided by , which is implemented by Keras, and uses a different initialization strategy.

Hence it may be slightly different from AGMF without p u and q i .

For fair comparison and pure ablation study, we conduct experiments using the codes of AGMF, to compare the performance of AGMF and AGMF without the supplementary latent vectors p u and q i .

While with a slight abuse of naming, in this ablation study, we still name AGMF without p u and q i as GMF.

Figure 2 reports the comparison results of AGMF and GMF on all the datasets.

It can be seen that AGMF always achieves better performance than GMF, in terms of both evaluation metrics.

Furthermore, we also report the comparison results of AGMF and GMF in each training epoch in Figure 3 .

We can observe that:

??? AGMF consistently outperforms GMF in each training epoch.

??? AGMF converges faster than GMF.

??? AGMF achieves lower training loss than GMF.

By integrating historical interactions into user embedding and item embedding, the above observations are revealed by this paper for the first time.

Therefore, the importance of multi-hot encoding for generalized matrix factorization is clearly demonstrated.

Moreover, these observations may bring new inspirations about how to properly integrate the one-hot encoding and multi-hot encoding for effectively improving the recommendation performance.

Learning good representations of users and items is crucially important to recommendation with implicit feedback.

In this paper, we propose a novel Augmented Generalized Matrix Factorization (AGMF) model for learning from implicit feedback data.

Extensive experimental results demonstrate that our proposed approach outperforms state-of-the-art counterparts.

Besides, our ablation study clearly demonstrates the importance of multi-hot encoding for Generalized Matrix Factorization.

As user-item interaction relationships are vitally important for learning effective user embedding and item embedding, hence in future work, we will investigate if there exist better user-item interaction relationships that can be exploited to improve the recommendation performance.

@highlight

A simple extension of generalized matrix factorization can outperform state-of-the-art approaches for recommendation.

@highlight

The work presents a matrix factorization framework for enforcing the effect of historical data when learning user preferences in collaborative filtering settings.