Dense word vectors have proven their values in many downstream NLP tasks over the past few years.

However, the dimensions of such embeddings are not easily interpretable.

Out of the d-dimensions in a word vector, we would not be able to understand what high or low values mean.

Previous approaches addressing this issue have mainly focused on either training sparse/non-negative constrained word embeddings, or post-processing standard pre-trained word embeddings.

On the other hand, we analyze conventional word embeddings trained with Singular Value Decomposition, and reveal similar interpretability.

We use a novel eigenvector analysis method inspired from Random Matrix Theory and show that semantically coherent groups not only form in the row space, but also the column space.

This allows us to view individual word vector dimensions as human-interpretable semantic features.

Understanding words has a fundamental impact on many natural language processing tasks, and has been modeled with the Distributional Hypothesis BID0 .

Dense d-dimensional vector representations of words created from this model are often referred to as word embeddings, and have successfully captured similarities between words, such as word2vec and GloVe BID1 BID2 .

They have also been applied to downstream NLP tasks as word representation features, ranging from sentiment analysis to machine translation BID3 BID4 .Despite their widespread popularity in usage, the dimensions of these word vectors are difficult to interpret BID5 .

Consider w president = [0.1, 2.4, 0.3] as the 3-dimensional vector of "president" from word2vec.

In this 3-dimensional space (or the row space), semantically similar words like "minister" and "president" are closely located.

However, it is unclear what the dimensions represent, as we do not know the meaning of the 2.4 in w president .

It is difficult to answer questions like 'what is the meaning of high and low values in the columns of W' and 'how can we interpret the dimensions of word vectors'.

To address this problem, previous literature focused on the column space by either training word embeddings with sparse and non-negative constraints BID6 BID7 BID8 , or post-processing pre-trained word embeddings BID5 BID9 BID10 .

We instead investigate this problem from a random matrix perspective.

In our work, we analyze the eigenvectors of word embeddings obtained with truncated Singular Value Decomposition (SVD) BID11 BID12 of the Positive Pointwise Mutual Information (PPMI) matrix BID13 .

Moreover, we compare this analysis with the row and column space analysis of Skip Gram Negative Sampling (SGNS), a model used to train word2vec BID14 .

From the works of BID15 proving that both SVD and SGNS factorizes and approximates the same matrix, we hypothesize that a study of the principal eigenvectors of the PPMI matrix reflects the information contained in SGNS.Contributions: Without requiring any constraints or post-processing, we show that the dimensions of word vectors can be interpreted as semantic features.

In doing so, we also introduce novel word embedding analysis methods inspired by the literature of eigenvector analysis techniques from Random Matrix Theory.

Recently, there have been several works that have shown similar results in semantic grouping among the column values.

Several of these algorithms proposed to train non-negative sparse interpretable word vectors BID6 BID7 BID8 BID16 .Furthermore, BID5 also proposed methods to post-process pre-trained word vectors with non-negativity and sparsity constraints.

However, their vectors were optionally binarized, which is difficult to interpret intensity than real-values.

BID9 has proposed to overcome these limitations by simply training a rotation matrix to transform pre-trained word2vec and GloVe, without being sparse or binary.

Finally, BID10 post-trained the pre-trained word embeddings with k-sparse autoencoders with similar constraints to BID5 .While these methods were able to successfully achieve interpretability in the column space evaluated with word intrusion detection tests, they either enforced sparsity and non-negativity constraints, or required extensive post-processing.

Furthermore, they focused less on the analysis and discussion on the actual meanings of the columns despite their pursuit of interpretable dimensions.

Hence, in our work, we put more emphasis on such implications with conventional algorithms without any extra constraints or post-processing steps.

We define the Positive Pointwise Mutual Information (PPMI) matrix as M PPMI , the set of unique words as vocabulary V , and word embedding matrices created from SVD and SGNS as W SVD and W SGNS .

The k-th largest eigenvalue and corresponding eigenvector of M PPMI are denoted as ?? k and u k ??? R |V | , and the k-th column of W SGNS as v k ??? R |V | .

The word vectors are denoted w SVD word or w SVD word , but when context is clear or does not matter, we simply use w word .

Note that we often use the term "eigen" when and "singular" interchangeably because M PPMI is defined as a square matrix.

Each entry of a co-occurrence matrix M represents the co-occurrence counts of words w i and c j in all documents in the corpus.

However, raw co-occurrence counts have been known to underperform than other transformed variants BID15 .

Pointwise Mutual Information (PMI) BID13 instead transforms matrix by measuring the log ratio between the joint probability of w and c when assuming independence of the two and not.

DISPLAYFORM0 The problem of this association measure is when dealing with never observed pairs which result in PMI(w, c) = log 0.

To cope with such, Positive Pointwise Mutual Information has been used to map all negative values to 0 from the intuition that positive associations are often more informative in downstream NLP tasks BID15 .PPMI(w, c) = max(PMI(w, c), 0)

Truncated SVD (we will further refer this as simply SVD), which is equivalent to maximum variance Principal Component Analysis (PCA) and has been popularized by Latent Semantic Analysis (LSA) BID12 , factorizes the PPMI matrix as M PPMI = U ?? S ?? V T and truncates to d dimensions.

Following the works of BID17 , the word embedding matrix is taken as W = U d , instead of the more "standard" eigenvalue weighting W = U d ?? S. We discuss the effect of this in Section 6.2.

Unlike PPMI and SVD which gives exact solutions, the word2vec Skip-Gram model, proposed by BID1 , trains two randomly initialized word embedding matrices W and C with a neural network.

DISPLAYFORM0 where DISPLAYFORM1 The intuition is to basically maximize the dot product between "similar" word and context pairs, and minimize the dot product between wrong pairs.

The Softmax function is simply a generalized version of the logistic function to multi-class scenario.

However, the normalization constant which computes the exponentials of all context words, is very computationally expensive when the vocabulary size is large.

Hence, BID14 proposed Skip Gram with Negative Sampling (SGNS) to simplify the objective using negative sampling.

We analyze the distributions of eigenvectors, calculate the Inverse Participation Ratios (IPR) to quantify the ratio of significant elements and measure structural sparsity, and qualitatively interpret the significant elements.

The empirical distribution of eigenvector elements u k is compared with a Normal distribution N(?? u k , ?? 2 u k ) to measure normality of the eigenvectors, where ?? u k , ?? 2 u k refer to the mean and variance of u k .

BID18 have shown that eigenvectors deviating from Gaussian contain genuine correlation between stocks, while also revealing a global bias that represented newsbreaks influencing all stocks.

We search for similar patterns in Section 5.1.Inverse Participation Ratio:

The Inverse Participation Ratio (IPR) of u k , denoted as I k , quantifies the inverse ratio of significant elements in the eigenvector u k BID18 BID19 BID20 .

DISPLAYFORM0 where u k i is the i-th element of u k .

The intuition of IPR can be illustrated with two extreme cases.

First, if all elements of u k have same values 1/ |V |, then I k is simply 1/|V |, with reciprocal 1/I k being |V |.

This means that all |V | elements contribute similarly.

On the other hand, a one-hot vector with only one element as one, and the rest as zero, u k will have an IPR value of one (also same for reciprocal).

Hence, the reciprocal, 1/I k , measures the ratio of significant participants in u k .

In short, the larger the I k , the smaller the ratio of participation, and the sparser the vector, in turn, reflecting structural sparsity of u k .

Furthermore, as 1/I k ??? [1, |V |], dividing this reciprocal with |V | will yield the sparsity of a given vector u k ??? R |V | .Visualization of Top Eigenvector elements: As u k , v k ??? R |V | , we can map each index of the vectors to a word in the vocabulary V .

Hence, we investigate the dimensions and their indices (or words) with the largest absolute values and search for semantic coherence.

Similar approaches with financial data have shown to group stocks from same industries or nearby regions BID18 , and with genetic data, revealed important co-evolving genes in gene co-expression networks BID19 .

BID1 , which has also been used by BID1 .

Removing most of the noisy non-alphanumerics, such as XML tags, the dataset size effectively reduced from approximately 66GB to 25GB, containing around 3.4B tokens.

The vocabulary size is approximately 346K as we only consider words with at least 100 occurrences.

SGNS and SVD We adapt the code from the hyperwords 3 released by BID17 to train both W SVD and W SGNS .

Our code is publicly available online BID3 .

For W SGNS , we set negative sampling as 5.

For both, we set a context window size of 2 (taking 5 words as context) and embedding dimension d = 500.

From FIG0 , we can see that eigenvectors corresponding to the larger eigenvalues such as u 1 or u 2 clearly deviate from a Gaussian distribution, and so do u 100 and u 500 , but less.

This shows us that the eigenvectors are not random and contain meaningful correlations.

It is expected to see such pattern because these vectors are the principal eigenvectors.

On a more interesting note, u 1 not only significantly deviates from a normal distribution, but also has only non-zero negative values as its elements, and no other eigenvectors have shown this behavior.

This suggests that this particular eigenvector could represent a common bias that affects all "words", as it captured the effect of news outbreaks for stock prices in BID18 .

We revisit the interpretation of this observation in Section 6.1.

FIG1 illustrates the IPR of u k plotted against the corresponding eigenvalue ?? k , and vice versa for v k .

From the plot, we can clearly see that the eigenvectors of W SVD have approximately 10x higher IPR values than those of W SGNS , meaning that the vectors are much sparser for W SVD .

From FIG1 , we can see that the largest eigenvector has the smallest IPR of 0.000006, and the reciprocal 1/I k divided by |V |, yields 48%, while the same for the largest I k gave around 4.7%.

The mean value of 1/I k divided by |V |, across all eigenvectors was 27.5% indicating that there exists some sparse structure within the eigenvectors of W SVD .

On the other hand, FIG1 shows that mean for v k was around 36%, meaning that column vectors of W SGNS are generally denser and less structured.

Such discrepancy in structural sparsity motivates us to analyze the eigenvectors of W SVD in depth.6 Analysis and Discussion

Based on the results of previous sections, we further examine the top elements of the eigenvectors by sorting their absolute values in decreasing order.

Table 3 : Top participants of the salient columns of the word vector for "airport."baseball related words.

Some words from u 121 initially seem irrelevant to baseball.

However, "buehrle" is a baseball player, "rbis" stand for "Run Batted Ins", and "astros" is a baseball team name from Houston.

Meanwhile, the words grouped in u 1 , the largest eigenvector, could explain the bias we mentioned in Section 5.1.

The significant participants tend to be strong transition words that are used often for dramatic effects, such as "importantly" or "crucially".

Evidently, these words increase the intensity of the context.

Moreover, while it was originally hypothesized that the largest principal eigenvectors would capture some semantic relationship, the 121th vector u 121 show surprisingly focused and narrow semantic grouping related to baseball.

Further investigation reveals that u 121 has one of the highest IPR values, hence being one of the most sparse vectors.

We verify similar trends in other eigenvectors with high IPR values as shown in TAB3 .

An interesting pattern arises here, in which the sparser eigenvectors tend to capture more distinct and rare features such as foreign names or languages, or topics like baseball.

Furthermore, we compare the column space analysis on W SVD and W SGNS .

Consider the word vector w airport for the word "airport.

"

We choose the salient dimensions, which are the largest elements, of w airport , and investigate the significant elements of those chosen dimensions (columns).

Table 3 shows that the columns from W SVD display semantic coherence while those from W SGNS seem random.

u 53 groups words that are related to the location of the airports.

For example, one could say "Trindade station connects with the airport.

" Similarly, u 337 groups famous airline companies together, while "fiumicino" is a famous airport in Italy.

Sections 5.1 and 5.2 revealed that the eigenvectors contain genuine correlation and structure in the column space.

We further show in Section 6.1 that semantically coherent words form groups of significant participants in each eigenvector.

Now we can answer the questions we asked earlier.

What is the meaning of high and low values in the columns of W?

If word vector w from W SVD has a high absolute value in column k, it means that the word is relevant to the semantic group formed in u k .

For example, the words from FIG2 have highest values in column k = 121, in which u 121 represents a semantic group related to baseball, as shown in TAB1 .How can we interpret the dimensions of word vectors?

The answer to this question follows naturally.

As the salient dimensions represent relevant semantic groups, we can view the dimensions of w as semantic features.

This view is in line with the Topic Modeling literature, in which words and documents are clustered into distinct latent topics.

Hence, we can also see the word embedding dimensions as latent topics that can be interpretable.

It can be easily seen from FIG2 that similar words do not show any interpretable similarity in their W SGNS representations, despite being nearest neighbors in the row space.

On the other hand, it is very clear from FIG2 that similar words have similar representations, or feature vectors.

We thus empirically verify that the dimensions of the row vectors can be viewed as semantic or syntactic features.

Finally, the structural sparsity discovered with the IPR is further confirmed by contrasting FIG2 .

It is clearly visible that the the vectors from SVD are much sparser than from SGNS.Effect of Eigenvalue Weighting: As mentioned in Section 3, weighting with the eigenvalues essentially scales each feature column by the corresponding eigenvalues.

Such process can be viewed as simply incorporating a prior, and does not hurt the interpretability.

However, as BID17 showed that eigenvalue weighting decreases the performance of downstream NLP tasks, we can assume that either the prior is wrong, or too strong.

In fact, in many cases, the largest eigenvalues are often order of magnitude larger than others, which can explain why not weighting the word embeddings with their corresponding eigenvalues would work better.

In this work, we analyzed the eigenvectors, or the column space, of the word embeddings obtained from the Singular Value Decomposition of PPMI matrix.

We revealed that the significant participants of the eigenvectors form semantically coherent groups, allowing us to view each word vector as an interpretable feature vector composed of semantic groups.

These results can be very useful in error analysis in downstream NLP tasks, or cherry-picking useful feature dimensions to easily create compressed and efficient task-specific embeddings.

Future work will proceed in this direction on applying interpretability to practical usage.

@highlight

Without requiring any constraints or post-processing, we show that the salient dimensions of word vectors can be interpreted as semantic features. 