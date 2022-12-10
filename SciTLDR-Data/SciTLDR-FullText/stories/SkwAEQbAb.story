Determining the number of latent dimensions is a ubiquitous problem in machine learning.

In this study, we introduce a novel method that relies on SVD to discover the number of latent dimensions.

The general principle behind the method is to compare the curve of singular values of the SVD decomposition of a data set with the randomized data set curve.

The inferred number of latent dimensions corresponds to the crossing point of the two curves.

To evaluate our methodology, we compare it with competing methods such as Kaisers eigenvalue-greater-than-one rule (K1), Parallel Analysis (PA), Velicers MAP test (Minimum Average Partial).

We also compare our method with the Silhouette Width (SW) technique which is used in different clustering methods to determine the optimal number of clusters.

The result on synthetic data shows that the Parallel Analysis and our method have similar results and more accurate than the other methods, and that our methods is slightly better result than the Parallel Analysis method for the sparse data sets.

The problem of determining the number of latent dimensions, or latent factors, is ubiquitous in a number of non supervised learning approaches.

Matrix factorization techniques are good examples where we need to determine the number of latent dimensions prior to the learning phase.

Non linear models such as LDA BID1 and neural networks also face the issue of stating the number of topics and nodes to include in the model before running an analysis over a data set, a problem that is akin to finding the number of latent factors.

We propose a new method to estimate the number of latent dimensions that relies on the Singular Value Decomposition (SVD) and on a process of comparison of the singular values from the original matrix data with those from from bootstraped samples of the this matrix, whilst the name given to this method, Bootstrap SVD (BSVD).

We compare the method to mainstream latent dimensions estimate techniques and over a space of dense vs. sparse matrices for Normal and non-Normal distributions.

This paper is organized as follow.

First, we outline some of best known methods and the related works in the next section.

Then we explain our algorithm BSVD in section 3.

The experiments are presented in section 4 and the results and discussion are reported in section 5.

And finally conclusion of the study is given in section 6.

The problem of finding the number of latent factors in a data set dates back to early work by BID4 .

It extends to a large array of fields including psychology and social science BID2 , bioinformatics BID8 , information retrieval BID25 , and of course statistical learning (Hastie et al., 2009, chapter 14) .

A systematic review on latent variables and its applications can be found in BID15 ; BID12 BID21 .In the following, we briefly explain some of the best known techniques of factor analysis method to decide the number of factors to retain as mentioned in BID14 such as Kaiser's eigenvalue-greater-than-one rule (K1), Parallel Analysis, Cattell's Scree test, Velicer's MAP test ( Minimum Average Partial ).

Moreover, we outline SW technique which is used in different clustering methods to determine the optimal number of clusters.

The K1-Kaiser method was first introduced by BID4 and later extended and popularized by BID11 .

The method an early strategy used to detect the number of factors to retain by considering.

It relies on the eigenvalues of the correlation matrix of the observed factors an stipulates that the number of eigenvalues greater than one corresponds to the number of latent factors to retain.

Despite its simplicity, some researches consider it unreliable BID14 BID17 BID3 .

We will nevertheless consider it in our comparison experiments, given that it is a classic method and the first that introduced the use of eigenvectors of the correlation matrix for determining the number of latent factors.

Note that a variant of this method was introduced by BID26 that showed improvements but still lacked behind PA ( §2.2 and MAP ( §2.4) and will therefore not be included in the experiment.

Parallel Analysis is also based on the correlation matrix between the observed factors.

It uses bootstrapping on the correlation matrix and then averages the eigenvalues over the bootstrap runs.

Eigenvalues greater than the average data set eigenvalue are kept.

BID17 ; BID14 .

This strategy was proposed by BID9 .

BID26 showed that PA attempt improves over the Eigenvalue-greater-than-one rule.

Several researchers found this method appropriate and more accurate in determining the number of factors to retain BID16 BID17 .

We will see that it PA a close relationship with BSVD and this is corroborated by the closeness of the results.

Cattell's Scree test, also known as the "elbow" approach, is a graphical representation method to display the number of components to retain.

Scree test is a subjective method which sorts the eigenvalues in decreasing order and shows it on the vertical axis; and the number of components in horizontal axis BID14 .

In this strategy, we need to find where the Scree happened and the components on the left side of the slope should be retained.

BID5 used the Scree test plot and mention that the method fails when elbow cannot be found.

Moreover, BID23 mentions the Scree test as one of the graphical tests to finding the number of latent variables, they specify that researchers should utilize of the ruler to plot a line across the elbow and then they could keep all the components above it.

Despite, the various criticism to use this method by BID27 and BID10 , it is one of the most popular methods to find the number of important factors to retain Mumford et al. (2003)

MAP approach is based on PCA and relies on the series of partial correlation matrices to define the number of significant factors to retain BID17 BID14 BID27 BID18 .

This approach is introduced by BID24 .

In general, statisticians agree that the MAP and PA are the two techniques which are reliable solution to extract the number of factors to retain with the reasonable result Ledesma

Clustering is generally tought of as a means to reduce the number of data points, but it can be considered for dimensionality reduction technique, namely by using the cluster's centroid and each point's distance from them as a means to define a new space, which was shown to be more effective than SVD under certain conditions (?).

Therefore, a method to determine the optimal number of clusters can provide another means to determine the dimensionality of a data set.

PAM method is one of the popular technique to automatically determine the optimal number of clusters with the Silouhette (SW) technique BID13 .

The number of clusters computed by SW is associated with the number of latent dimensions in the dataset.

SVD is a well known matrix factorization technique that decomposes the original matrix, R, into the product of two eigenvector matrices, the eigenvectors of the cross-product of the rows and columns, and a of the diagonal matrix of their common singular values.

DISPLAYFORM0 where U and V are orthogonal, and Σ is a diagonal matrix with positive real values.

The singular values represent the importance of the eigenvectors, ordered by decreasing values.

The BSVD method determines the number of dimensions as the point where the singular values of Σ cross the singular values Σ B of a randomized (through bootstrap sampling) matrix R B .

An interpretation of this crossing point where Σ and Σ B meet is that the remaining singular values are no more due to influential factors, at least in a linear framework.

An example of this can be seen in figure 2 for a data set that was generated using 9 dimensions (vertical line) with uniform distributed values.

In this case, it is also easy to tell the number from the elbow at dimensions= 9.The bootstrapped samples R B are simply generated through random sampling with replacement of the values of R.In the next section, we look at the details of generating data sets and the experiments.

We evaluate the ability of BSVD to identify the number of latent dimensions by using synthetic data.

Although the use of synthetic data limits the generalizability of our conclusions to the real world by making strong assumptions on the data, it remains the best validation methodology given that we know the ground truth behind the synthetic data and we can control the sparsity and the underlying distributions of each of the observed variables in order to explore this space of conditions.

4.1 DATA SETS 4.1.1 SYNTHETIC DATA The synthetic data is generated by sampling from distributions to create two matrices, P and Q. Then, R is obtained by the product P · Q plus Gaussian noise.

We use two types of distributions, the normal (Gaussian) with mean= 0 and standard deviation= 1, and the uniform distribution with mean= 2.5 to generate the columns of P and rows of Q. The choice of 2.5 is inspired from rating-type of data found in recommender systems.

The Gaussian noise added to P · Q corresponds to one standard deviation of R with mean= 0.All R matrices are of size 150 × 240 and we explore the latent dimensions from 2 to 24.In figure 1 , we illustrate an example of the generated non-normal data set with size of 5 × 6 and latent dimension 3.Data sets of different density are generated, since sparsity is a constraint that we often have to deal with in fields such as recommender systems (rating matrices) and natural language processing (term-document matrices).

Sparsity is created by randomly selecting the missing value cells.

In order to capture the behavior of our method when we face a sparse matrix, we employ the algorithm with different percentages of sparseness to the data set (see algorithm 1).

Then, we compare the result with the existing mentioned approaches in Table 2 .

To do so, we follow the next steps for each iteration of latent dimension (j): 1) We apply a different percentage of sparseness (j) from 10: 90 to our data set with random selection.2) We impute each missing value by the average of the mean of corresponding row and column.3) Apply BSVD.

And for each latent dimension (j); record the result in each iteration of k.

Compute the average accuracy of each method, when (j) terminate.

Figure 5 displays the accuracy of all the methods in the non-normal sparse data set with latent dimension (j) equal to 2.

We repeat the previous experiments on the generated simulated random data set with a normal distribution.

The BSVD algorithm is compared with Horn's PA and K1 implementations from BID7 .

Moreover, we used of Very Simple Structure(VSS) and PAM packages of R to have the outcome of MAP and SW methods respectively.

According to the results of provided experiments in the tables 1 and 2, we could show that our method has a better performance than those mentioned especially in the sparse data sets.

Our empirical experiments demonstrate that on the dense data sets; the accuracy of BSVD and PA is equal and better than the other approaches.

But when we apply a different percentage of sparseness to our data sets, our method is more precise.

In the figures 3 and 4, we display the behavior of each method in the dense and sparse data sets.

Figure 3 depicts the average accuracy of all methods in the dense data sets with normal and nonnormal distribution.

It shows that MAP method in the dense data set with normal or non-normal distribution has the same accuracy.

Additionally, SW technique performs better result with the face of the dense data set with non-normal distribution, while K1 has an extreme behavior in the nonnormal data set.

Moreover, BSVD, PA and K1 are more precise in the dense data set with normal distribution.

Figure 4 shows the sparse data sets with normal and non-normal distribution.

It demonstrates that BSVD, PA, and K1 have better accuracy in the sparse data set with normal distribution but MAP and SW are on the contrary.

Figure 5 shows the average accuracy of all the methods in in different level of sparsity over the non normal sparse data set with latent dimensions (j) equal to 2.

The error bars shows the variance of the observations after repeating the algorithm 25 times.

Based on the results of these experiments we can conclude that our approach (BSVD) is better than the presented methods especially in the sparse data sets.

To show if the outcome is statistically significant and is not by chance, we apply t-test between our method and PA.

We considered the p values less than or equal to 0.05 as a significant result.

To do so, we consider a sample of latent dimensions (j = {2, 3, 5, 8, 15}) and we repeat twenty-five times the mentioned experiments on the sparse data sets with normal and non-normal distribution, and record the result.

Then we apply t-test between BSVD and PA.

In this evaluation the null hypothesis (H0) state that µ SV D = µ P A and if the H0 is rejected, we could conclude that the obtained results are not by chance and our method is better than PA.

TAB1 contain p values of the sparse and dense data sets with normal and non-normal distribution respectively.

The first row of each table with 0% of sparsity indicate to the dense data sets.

TAB1 shows more constant behavior, and implies that by increasing sparsity in the sparse data set with normal distribution, BSVD yeilds a significantly better result.

But table 4 that shows the result of non-normal sparse data set is hard to interpret.

Because the green cells are not affected by increasing sparsity.

We can sum up with that the result seems to be significant with increasing the sparsity.

In general, according to the tables 3 and 4, the difference between our method and PA seems to be statistically significant by increasing the percentage of sparsity.

The objective of our study was to introduce a new method to find the number of latent dimensions using SVD which we inspired from PA.

We employ our method on simulated data sets with normal and non-normal distribution whereas are dense or sparse and compared with the present methods such as PA, MAP, K1, and SW.

According to the mentioned experiments and the reported results in the table 1, BSVD and PA have the same accuracy and better than the other presented methods in the dense data sets.

But our method has a better result in the sparse data sets which is shown in the table 2.

We applied t-test on the sample of latent dimensions (j) between BSVD and PA to demonstrate if the result is statistically significant or not.

The results in the tables (3 and 4) demonstrate that in the sparse data sets with increasing the sparsity, our method seems to be significantly better than the other methods.

Our method performance is limited to the presented experiments and data sets.

If we want to generalize the method, We need to see the behavior of the algorithm when we have a more complex data set.step a. Generating the matrices x and y with the sizes of 6 × 3 and 5 × 3.

@highlight

In this study, we introduce a novel method that relies on SVD to discover the number of latent dimensions.