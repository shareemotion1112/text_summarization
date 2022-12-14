In this paper, we propose two methods, namely Trace-norm regression (TNR) and Stable Trace-norm Analysis (StaTNA), to improve performances of recommender systems with side information.

Our trace-norm regression approach extracts low-rank latent factors underlying the side information that drives user preference under different context.

Furthermore, our novel recommender framework StaTNA not only captures latent low-rank common drivers for user preferences, but also considers idiosyncratic taste for individual users.

We compare performances of TNR and StaTNA on the MovieLens datasets against state-of-the-art models, and demonstrate that StaTNA and TNR in general outperforms these methods.

The boom of user activity on e-commerce and social networks has continuously fueled the development of recommender systems to most effectively provide suggestions for items that may potentially match user interest.

In highlyrated Internet sites such as Amazon.com, YouTube, Netflix, Spotify, LinkedIn, Facebook, Tripadvisor, Last.fm, and IMDb, developing and deploying personalized recommender systems lie at the crux of the services they provide to users and subscribers (Ricci et al., 2015) .

For example, Youtube, one of the worlds most popular video sites, has deployed a recommender system that updates regularly to deliver personalized sets of videos to users based on their previous or recent activity on site to help users find videos relevant to their interests, potentially keeping users entertained and engaged BID5 .Among the vast advancements in deep learning and matrix completion techniques to build recommender systems (Ricci BID21 , one of the most imperative aspect of research in such area is to identify latent (possibly low-rank) commonalities that drive specific types of user behaviour.

For example, BID6 proposes a deep neural network based matrix factorization approach that uses explicit rating as well as implicit ratings to map user and items into common low-dimensional space.

Yet, such variety of low-rank methodologies do not address the impact of idiosyncratic behaviour among buyers, which may potentially skew the overall learned commonalities across user groups.

In this work, we propose two multi-task learning methods to improve performances of recommender systems using contextual side information.

We first introduce an approach based on trace-norm regression (TNR) that enables us to extract low-rank latent dimensions underlying the side information that drive user preference according to variations in context, such as item features, user characteristics, time, season, location, etc.

This is achieved by introducing a nuclear-norm regularization penalty term in the multi-task regression model, and we highlight that such latent dimensions can be thought of as homogeneous behaviour among particular types of user groups.

Furthermore, we propose a novel recommender framework called Stable Trace-norm Analysis (StaTNA) that not only captures latent low-rank common drivers for user preference, but also considers idiosyncratic taste for individual users.

This is achieved by, in addition to the low-rank penalty, adding a sparsity regularization term to exploit the sparse nature of heterogeneous behaviour.

Finally, we test the performance of StaTNA on the MovieLens datasets against state-of-the-art models, and demonstrate that StaTNA and TNR in general outperforms these methods.

We first introduce some notation that will be adopted throughout the rest of our work.

We let ??? denote the set of all observed entries, and for any matrix Z, defin?? Z = P ??? (Z) asZ ij = Z ij if (i, j) ??? ??? and 0 otherwise.

We let Y ??? R n??p , be the final output of the recommender system, X ??? R n??d includes all side information, and L, S ??? R d??p represent the common and idiosyncratic effects of side information on users.

Both L and S can be considered as mappings from the side information to the recommender response.

To be more specific, we take a movie recommender system for example: n can be the number of movies, p is the number of users, so each entry of Y is the predicted rating of each user for every movie, while X represents the features of each movie where each feature is d dimensional.

When a new movie comes in, we apply L or/and S to the movie's feature to predict the rating of each existing user's, and recommend the movie to users with high predicted ratings.

Before turning to TNR, we first consider a regularized lowrank solution for large-scale matrix completion problems called Soft-Impute BID14 , which sheds light upon the key ideas of trace-norm regression.

The SoftImpute problem is formulated as the following: DISPLAYFORM0 where ?? F denotes the Frobenius norm and ?? * denotes the nuclear norm.

In this formulation, we minimize the reconstruction error subject to a bound on the nuclear norm, which serves as a convex relaxation of rank of a matrix and allows us to exploit the low-rank structure of the matrix L.Based on similar ideas, trace-norm regression extends this idea of incorporating regularization on the rank of a matrix in the context of multi-task learning, as it minimizes square loss while penalizing large ranks of the coefficient matrix: DISPLAYFORM1

Similar to our introduction of Soft-Impute and TNR, we first discuss a non contextual model that incorporates both the low-rank matrix L and sparse matrix S, namely the stable principal component pursuit (SPCP) BID24 Zhou et al., 2010; BID23 : DISPLAYFORM0 where ?? 1 denotes the sum of absolute values for all entries of a matrix.

S 1 and L * models sparsity in S and the low-rank structure in L respectively.

To further illustrate some intuition for the choice of such norms, we provide an example in the context of foreground-background separation in video processing.

L can be considered as the stationary background, which is low-rank due to the strong correlation between frames; while S can represent foreground objects, which normally occupy only a fraction of the video and hence can be treated as sparse.

BID15 Finally, in light of SPCP, we propose a novel framework called Stable Trace-norm Analysis (StaTNA) by adding contextual side information to consideration: DISPLAYFORM1 Note that StaTNA can be considered as a combination of trace norm regression and SPCP, and some theoretical aspects are discussed in BID0 .

The matrix L captures latent homogeneity among preferences of users, such as an Academy Award winning film would be preferable to many users.

On the other hand, idiosyncratic tastes of users are embodied in S, such as some users particularly like horror movies that involve robots and monsters.

Note that S can also be considered as a way to be robust to outliers in user behaviour or preference.

In the case where X is the identity matrix, this problem is reduced to SPCP.

Also, TNR can be considered as a special case of StaTNA by taking ?? S = ???, which explains why StaTNA will in general be more robust compared to TNR.

In this subsection we will only briefly discuss the methodologies used in this paper to solve TNR and StaTNA, due to space limitations.

We first highlight the computational feasibility for both models since methods such as proximal gradient descent BID16 or (Fast) Iterative Shrinkage-Thresholding Algorithm (FISTA, BID2 ) can be used to solve these problems with provable (sub)linear convergence rate BID12 .

For StaTNA, we directly apply FISTA to estimate L and S, and the procedure is detailed in Algorithm 1 in Appendix A.1.

Next, as aforementioned, TNR is a special case for StaTNA, so to solve TNR, we simply take ?? S = ??? in Algorithm 1.

We also point out that Algorithm 1 may be computationally expensive when the matrix is large.

In Appendix A.2 we will propose several modifications to our method for solving TNR and StaTNA which will enable us to improve computational complexity.

In our work, we consider the MovieLens 100K and MovieLens 1M datasets.

The summary for both datasets are shown in TAB1 Note that MovieLens 1M movies do not include all MovieLens 100k movies.

In both datasets, each user has rated at least 20 movies, and ratings are whole numbers in the scale of 1-5.

In addition, each dataset is associated with a side information matrix whose rows are indexed by movies and includes 1 feature column denoting the movie category (19 categories in total), along with 1128 columns which denote relevant scores of a movie to provided tags.

We pre-process side information to obtain two types of sideinformation matrices.

For the first type, we apply one-hotencoding to the categorical feature and obtain 19 categorical features for each dataset.

The final side information matrix is the concatenation of this categorical representation and relevance scores to given tags, which has dimensions n ?? 1147 where n is the number of movies in each dataset.

For the second type, in addition to one-hot-encoding, we apply GloVe word embeddings with K = 300 BID18 to these categories using an average pooling trick, and result in a 300-dimensional categorical representation vector for each movie.

The final side information matrix has dimensions n ?? 1428.

For simplicity, we use the suffix "-G" to denote models trained on the second type of side information processed using GloVe word embedding, while models without this suffix will denote models trained on the first type of side information.

In this work, we perform two experiments: 1. (Matrix completion with side information) We fill in missing values in the rating matrix using the pre-processed side information matrix, which is the traditional matrix completion problem.

For each dataset, we randomly select 20% of observed values in the rating matrix as the test set, and train TNR & StaTNA, and TNR-G & StaTNA-G on the two types of side-information matrices respectively.

We use state-of-theart models such as SVD, Sparse FC, and GC-MC as our baseline.

Here we point out that these baseline models do not incorporate side information, opposed to our TNR and StaTNA models which are trained on side information.

Yet, our experimental results will demonstrate that our proposed models, via utilizing side information, will significantly improve performance in this matrix completion task.

2. (Regression) We predict the ratings for new movies based on new side information of the movie, which is similar to the traditional regression problem.

For each dataset, we train TNR, TNR-G, StaTNA, StaTNA-G on a randomly selected 80% of all movies and apply trained models on the remaining 20%.

Furthermore, both experiments involve two hyperparameters, namely ?? L and ?? S , which are tuned using 10-fold cross validation.

We use Lasso, Elastic Nets, Multi-task Lasso and Multi-task Elastic Nets BID17 as our baseline models.

Note that standard Lasso and standard Elastic Nets are trained for each user independently to predict the user's rating for a given movie.

The matrix formulations of these baseline models are shown in Table 6 We notice that both TNR and StaTNA do not perform as well as state-of-the-art models in MovieLens 100K.

One explanation is that our models require more training data compared to baseline models to fully capture latent structures of both L and S in order to generalize well on test data.

Finally, we also point out that StaTNA converges faster, and in general performs better than TNR, as shown in Figure 1 within Appendix C.1.Experiment 2: Regression In the second experiment to predict user ratings for movies, as shown in TAB4 .

MAE and RMSE for test data in Experiment 2 (regression) for baseline models (Lasso and Elastic Nets (denoted as EN)), TNR and StaTNA using MovieLens 100K and MovieLens 1M.

The prefix "MT" is the abbreviation for "Multi-task".MovieLens 100K, all StaTNA and TNR models significantly outperform baseline models, while TNR with GloVe embedding yields the best out-of-sample performance.

For MovieLens 1M, StaTNA with GloVe embedding results in the best out-of-sample performance compared to all other baseline models including TNR, with and without GloVe embedding.

The strong performance of StaTNA and StaTNA-G across both experiments, and especially in MovieLens 1M, indicates that our StaTNA framework provides promising performance guarantees for solving both matrix completion and regression tasks.

As mentioned in earlier sections, we are interested in analyzing particular underlying commonalities in user preferences.

We achieve this by investigating the principal components of our estimate of the low-rank matrix L, each of which we consider as a common type of user preference.

Since our estimated L is of rank 6, we conclude that there are 6 major common types of user preferences, whose component scores (i.e. explained variance percentages) are listed in Table 4 , where we observe that the first principal component explains 88.94% of the variability in user ratings.

Table 5 .

Top 12 features of highest absolute weights within the first two principal components (PC1 and PC2).

Details of other principle components are shown in TAB9 in Appendinx C.2.

Our methodology to solve TNR and StaTNA (i.e. Algorithm 1 in Appendix A.1) may be computationally expensive when the matrix is large since it requires calling a Singular Value Decomposition (SVD) oracle in each iteration of the algorithm.

Hence we propose two alternative methods, a FW-T algorithm and a nonconvex reformulation of the problem, to avoid using an SVD oracle.

These are detailed in Appendix A.2.

Furthermore, our current studies use side information from only one side, namely movie information.

Our StaTNA framework can be extended to incorporate side information for both movies and users: DISPLAYFORM0 where U and M denotes users and movies respectively.

Moreover, our StaTNA framework is also compatible with neural networks by including nuclear norm and sparse penalties to the objective.

We believe that similar formulations will provide us with better performance guarantees, but at the cost of model interpretability.

In this section, we discuss the methodologies we use to solve TNR and StaTNA.

As mentioned earlier, we use (Fast) Iterative Shrinkage-Thresholding Algorithm (FISTA, BID2 ) to solve these problems.

Before we address the detailed applications of these algorithms in our context to solve TNR and StaTNA, we introduce the following optimization oracles.

We define the proximal mapping of the 1 norm as DISPLAYFORM1 , whose extension to matrices is obtained by applying the scalar operator to each element.

Moreover, we define the proximal mapping of the nuclear norm BID4 BID13 DISPLAYFORM2 V , and Y = U DV is the SVD of matrix Y .

Now, using these definitions, we detail the algorithm to solve StaTNA in Algorithm 1.

Note that one can also initialize L 0 in both Algorithm 1 as DISPLAYFORM3 , where ??? denotes the pseudo-inverse of a matrix.

For StaTNA, we directly apply FISTA to estimate L and S, and the procedures are detailed in Algorithm 1.

As aforementioned, TNR is a special case for StaTNA, so to solve TNR, we simply take ?? S = ??? in Algorithm 1, which forces all S k and?? k to 0.

DISPLAYFORM4

In Algorithm 1, we call an SVD oracle in each iteration in FISTA to find the proximal mapping of the nuclear norm, which is computationally expensive when the matrix is large, i.e. the number of movie features and users are large.

Here we propose two methods to avoid using an SVD oracle.

First, inspired by a scalable algorithm FW-T on SPCP BID10 BID15 , we propose a similar FW-T algorithm to solve StaTNA by replacing the proximal mapping of the nuclear norm with a Frank-Wolfe update in each iteration.

To be more specific, we consider the following reformulation of StaTNA: DISPLAYFORM0 for some U L and U S such that the optimal solution (L , S ) is still feasible to the above problem, i.e. L * ??? U L and S * ??? U S .

For simplicity, we write the above objective function as g(L, S, t L , t S ).

In each iteration of the FW-T algorithm, we call the following Frank-Wolfe oracle (Algorithm 2) for L and S respectively.

We can see that for the Frank-Wolfe update for matrix L only requires to compute the leading singular pairs for a matrix, which can be achieved by computationally cheap power iteration BID11 .

In addition, we perform an exact line-search by easily solving a DISPLAYFORM1 The full algorithm for the FW-T is detailed in Algorithm 3.Algorithm 3 FW-T for StatNA DISPLAYFORM2 Second, we propose a nonconvex formulation of the problem suggested by BID1 BID22 : DISPLAYFORM3 where U ??? R d??r , V ??? R p??r .

This problem is nonconvex but smooth, and these two problems are equivalent in the sense that there is a ono-on-one correspondence between the global minima of these two problems BID3 BID7 .

Since this new formulation has multi-affine structure, according to new results given by BID9 , we can use ADMM to solve this, which is detailed in Algorithm 4.To apply ADMM on this nonconvex reformulation, we further reformulate the problem as the following DISPLAYFORM4 5: DISPLAYFORM5 8: DISPLAYFORM6 10: DISPLAYFORM7 DISPLAYFORM8 Frobineous norm Table 6 .

Summary for formulations of models.

MTR is the abbreviation for multitask regression, while MC is the abbreviations for matrix completion.

DISPLAYFORM9 DISPLAYFORM10 C. Additional Stuff C.1.

Experiment Figures (Figures 1 and 2) C.2.

Other Principal Components ( TAB9 .

Top 12 features of highest absolute weights within the third to sixth principal components (PC3, PC4, PC5 and PC6)

<|TLDR|>

@highlight

Methodologies for recommender systems with side information based on trace-norm regularization