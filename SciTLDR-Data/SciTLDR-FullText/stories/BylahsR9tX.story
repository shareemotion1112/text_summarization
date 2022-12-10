Large-scale Long Short-Term Memory (LSTM) cells are often the building blocks of many state-of-the-art algorithms for tasks in Natural Language Processing (NLP).

However, LSTMs are known to be computationally inefficient because the memory capacity of the models depends on the number of parameters, and the inherent recurrence that models the temporal dependency is not parallelizable.

In this paper, we propose simple, but effective, low-rank matrix factorization (MF) algorithms to compress network parameters and significantly speed up LSTMs with almost no loss of performance (and sometimes even gain).

To show the effectiveness of our method across different tasks, we examine two settings: 1) compressing core LSTM layers in Language Models, 2) compressing biLSTM layers of ELMo~\citep{ELMo} and evaluate in three downstream NLP tasks (Sentiment Analysis, Textual Entailment, and Question Answering).

The latter is particularly interesting as embeddings from large pre-trained biLSTM Language Models are often used as contextual word representations.

Finally, we discover that matrix factorization performs better in general, additive recurrence is often more important than multiplicative recurrence, and we identify an interesting correlation between matrix norms and compression performance.

Long Short-Term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997; BID11 have become the core of many models for tasks that require temporal dependency.

They have particularly shown great improvements in many different NLP tasks, such as Language Modeling (Sundermeyer et al., 2012; Mikolov, 2012) , Semantic Role Labeling (He et al., 2017) , Named Entity Recognition (Lee et al., 2017) , Machine Translation BID0 , and Question Answering (Seo et al., 2016) .

Recently, a bidirectional LSTM has been used to train deep contextualized Embeddings from Language Models (ELMo) (Peters et al., 2018a) , and has become a main component of state-of-the-art models in many downstream NLP tasks.

However, there is an obvious drawback of scalability that accompanies these excellent performances, not only in training time but also during inference time.

This shortcoming can be attributed to two factors: the temporal dependency in the computational graph, and the large number of parameters for each weight matrix.

The former problem is an intrinsic nature of RNNs that arises while modeling temporal dependency, and the latter is often deemed necessary to achieve better generalizability of the model (Hochreiter & Schmidhuber, 1997; BID11 .

On the other hand, despite such belief that the LSTM memory capacity is proportional to model size, several recent results have empirically proven the contrary, claiming that LSTMs are indeed over-parameterized BID4 James Bradbury & Socher, 2017; Merity et al., 2018; Melis et al., 2018; Levy et al., 2018) .

Naturally, such results motivate us to search for the most effective compression method for LSTMs in terms of performance, time, and practicality, to cope with the aforementioned issue of scalability.

There have been many solutions proposed to compress such large, over-parameterized neural networks including parameter pruning and sharing BID12 Huang et al., 2018) , low-rank Matrix Factorization (MF) (Jaderberg et al., 2014) , and knowledge distillation (Hinton et al., 2015) .

However, most of these approaches have been applied to Feed-forward Neural Networks and Convolutional Neural Networks (CNNs), while only a small attention has been given to compressing LSTM architectures (Lu et al., 2016; BID1 , and even less in NLP tasks.

Notably, See et al. (2016) applied parameter pruning to standard Seq2Seq (Sutskever et al., 2014) architecture in Neural Machine Translation, which uses LSTMs for both encoder and decoder.

Furthermore, in language modeling, BID13 uses Tensor-Train Decomposition (Oseledets, 2011 BID18 uses binarization techniques, and Kuchaiev & Ginsburg (2017) uses an architectural change to approximate low-rank factorization.

All of the above mentioned works require some form of training or retraining step.

For instance, Kuchaiev & Ginsburg (2017) requires to be trained completely from scratch, as well as distillation based compression techniques (Hinton et al., 2015) .

In addition, pruning techniques (See et al., 2016) often accompany selective retraining steps to achieve optimal performance.

However, in scenarios involving large pre-trained models, e.g. ELMo (Peters et al., 2018a) , retraining can be very expensive in terms of time and resources.

Moreover, compression methods are normally applied to large and over-parameterized networks, but this is not necessarily the case in our paper.

We consider strongly tuned and regularized state-of-the-art models in their respective tasks, which often already have very compact representations.

These circumstances make the compression much more challenging, but more realistic and practically useful.

In this work, we advocate low-rank matrix factorization as an effective post-processing compression method for LSTMs which achieve good performance with guaranteed minimum algorithmic speed compared to other existing techniques.

We summarize our contributions as the following:• We thoroughly explore the limits of several different compression methods (matrix factorization and pruning), including fine-tuning after compression, in Language Modeling, Sentiment Analysis, Textual Entailment, and Question Answering.• We consistently achieve an average of 1.5x (50% faster) speedup inference time while losing ∼1 point in evaluation metric across all datasets by compressing additive and/or multiplicative recurrences in the LSTM gates.• In PTB, by further fine-tuning very compressed models (∼98%) obtained with both matrix factorization and pruning, we can achieve ∼2x (200% faster) speedup inference time while even slightly improving the performance of the uncompressed baseline.• We discover that matrix factorization performs better in general, additive recurrence is often more important than multiplicative recurrence, and we identify clear and interesting correlations between matrix norms and compression performance.

Long-Short Term Memory (LSTMs) networks are parameterized with two large matrices, W i and W h , which adds the four gates to standard RNNs.

Once the parameters are learned, they became static matrices during inference time.

Hence, W i and W h can be compressed using Matrix Factorization to speed up running time, save memory, and possibly improve performance of LSTMs.

In this section, we define the basic LSTM structure and introduce Matrix Factorization (MF), namely Semi Non-Negative Factorization (NMF) and Singular Value Decomposition (SVD).

Lastly, we show how to apply Low-Rank Matrix Factorization to LSTMs parameters, W i and W h .

LSTM is an extended variation of RNN with the aim to capture long-term dependencies in the input and to avoid the exploding/vanishing gradient problems (Hochreiter & Schmidhuber, 1997) .

It includes input, output, and forget gates along with an explicit memory cell.

The gating layers control the information flow within the network, and decide which information to keep, discard, or update in the memory.

The memory cells learn the salient information through time.

The input gate decides what to keep from current input, and the forget gate removes less important information from the previous memory.

Finally, the output hidden state is extrapolated using the output gate and memory cell.

The following recurrent equations show the LSTM dynamics.

DISPLAYFORM0 where x t ∈ R ninp , and h t ∈ R n dim at time t.

Here, σ() and denote the sigmoid function and element-wise multiplication operator, respectively.

The model parameters can be summarized in a compact form with: Θ = [W i , W h ], where W i ∈ R 4 * ninp×4 * n dim which is the input matrix, and W h ∈ R 4 * n dim ×4 * n dim which is the recurrent matrix.

Note that we often refer W i as additive recurrence and W h as multiplicative recurrence, following terminology of Levy et al. (2018) .

In this section, we present an overview of Low-Rank Matrix Factorization and its application in LSTM.

The rank of a matrix W m×n is defined as the number of linearly independent rows or columns in W. Rank of a matrix could be computed either by finding the number of nonzero singular values of W, i.e., σ(W) 0 , or the smallest number r such that there exists a full-rank matrix U m×r and V r×n , in which W = UV BID9 .

The rank minimization problem σ(W) 0 is NP-hard, and a well known convex surrogate function for this problem is the nuclear norm W nuc = Σ r i=1 σ i .

Since computing the singular values for large scale data is expensive, we aim to find U and V to calculate the low-rank representation, in which a exists (Lu et al., 2016)

.The matrix W requires mn parameters and mn flops, while U and V require rm + rn = r(m + n) parameters and r(m + n) flops.

If we take the rank to be very low r << m, n, the number of parameters in U and V are much smaller compared to W. The general objective function is given as DISPLAYFORM0 There are various constrained versions for the low-rank matrix factorization in Equation 3.

In the following sections, we explain two most principal and prominent versions with orthogonality and sign constraints.

One of the constrained matrix factorization method is based on Singular Value Decomposition (SVD) which produces a factorization by applying orthogonal constraints on the U and V factors.

These approaches aim to find a linear combination of the basis vectors which restrict to the orthogonal vectors in feature space that minimize reconstruction error.

In the case of the SVD, there are no restrictions on the signs of U and V factors.

Moreover, the data matrix W is also unconstrained.

DISPLAYFORM0 The optimal values U r m×r , S r r×r , V r r×n for U m×n , S n×n , and V n×n are obtained by taking the top r singular values from the diagonal matrix S and the corresponding singular vectors from U and V.

Another important method, Semi-NMF generalizes Non-negative Matrix Factorization (NMF) by relaxing some of the sign constraints on negative values for U and W (V has to be kept positive).

Semi-NMF is more preferable in application to Neural Networks because of this generic capability of having negative values.

For detailed explanations of NMF and Semi-NMF, interested readers can refer to Appendix A.To elaborate, when the input matrix W is unconstrained (i.e., contains mixed signs), we consider a factorization, in which we restrict V to be non-negative, while having no restriction on the signs of U. We minimize the objective function as in Equation 9.

DISPLAYFORM0 The optimization algorithm iteratively alternates between the update of U and V using coordinate descent (Luo & Tseng, 1992) .

For interested readers, more details on the optimization method and the relation between the NMF-based algorithms and the clustering methods can be found in Appendix A.

As elaborated in Equation 1, a basic LSTM cell includes four gates: input, forget, output, and cell state, performing a linear combination on input at time t and hidden state at time t − 1 as in Equation FORMULA4 .

DISPLAYFORM0 For large scale data, having four W i matrices and four W h matrices demand huge memory and computational power.

Hence, we propose to replace W i , W h pair for each gate with their low-rank decomposition, leading to a significant reduction in memory and computational cost requirement, as discussed earlier.

The scheme of this operation is shown in the dashed box at the right side of Figure 1 and the complete factorized LSTM cell is shown in the left side of Figure 1 .

We mainly have two means of evaluation using five different publicly available datasets: 1) Perplexity in two different Language Modeling (LM) datasets, 2) Accuracy/F1 in three downstream NLP tasks that ELMo achieved state-of-the-art single-model performance.

We benchmark the language modeling capability using both Penn Treebank (Marcus et al., 1993, PTB) , with standard preprocessing methods as in Mikolov et al. (2010), and WikiText-2 (Merity et al., 2017, WT2 TAB7 .For all datasets, we run experiments across different levels of low-rank approximation r with different factorization methods, i.e. Semi-NMF and SVD.

We also compare the factorization efficiency when only one of W i or W h was factorized, and when both were compressed (denoted as W all ).

This is done in order to see which recurrence type (additive or multiplicative) is more suitable for compression.

As a compression baseline, we compare matrix factorization with the best pruning methodologies used in LSTMs BID14 See et al., 2016) .

To elaborate, for each weight matrix W i,h , we mask the low-magnitude weights to zero, according to the compression ratio of the low rank factorization 1 .

In Appendix TAB1 , we report the corresponding compression ratio for each rank used in the experiment.

In addition to standard metrics (e.g. Perplexity, Accuracy, F1), we report the following: number of parameters, efficiency E(r) (ratio of loss in performance vs loss in parameters -the lower the better; refer to Appendix B), L1 norm, and inference time 2 in test set for matrix factorization methods and uncompressed models.

Given a sequence of N words, x = (x 1 , x 2 , . . .

, x T ), the task of language modeling is to compute the probability of x T given context (x 1 , x 2 , . . .

, x T −1 ), for each token.

DISPLAYFORM0 We can naturally model this task with any many-to-many Recurrent Neural Network (RNN) variants, and, in particular, we train a 3-layer LSTM Language Model proposed by Merity et al. (2018) , following the same model architecture, hyper-parameters, and training details for both datasets, using their released code 3 .

After training the LSTM, we compressed the trained weights of each layer with Semi-NMF, SVD, and pruning, with different levels of low-rank approximations, and compare the perplexity.

Since, PTB is a small size dataset, we finetuned the compressed version of the model for several epochs 4 , interested reader can refer to Appendix for more details.

TAB1 summarizes the results which are reported in extensive form in TAB8 .From the tables, compressing W h is always more efficient and performing better than compressing W i for all the compression methods.

When we compare different compression methods, SVD has the lowest, and average, perplexity and the best efficiency among others, in both PTB and WT2.

This difference is not very noticeable for high rank (e.g. r=400), but it becomes more evident for higher compression, as shown in Appendix Figures 4 and 7.

Moreover, all the methods perform better than the result reported by BID13 ) which used Tensor Train (Oseledets, 2011 for compressing the LSTM layers.

In TAB1 , we report the results after fine-tuning.

The results shows that MF methods and pruning works better that existing reported compression BID18 and is very close to the uncompressed fine-tuned version of our baseline reported in Merity et al. (2018) .

In this setting, by factorizing W i with rank 10 we achieve a small improvement compared to the baseline with a 2.13x speedup.

Notice that with rank 10 pruning also comparably works (57.94 PPL).

1 We align the pruning rate with the rank with r(m+n) mn .

2 Using an Intel(R) Xeon(R) CPU E5-2620 v4 2.10GHz, and averaged over 5 runs.

3 https://github.com/salesforce/awd-lstm-lm 4 For pruning during training we blind the weights using a static mask.

6 Timing information was impossible to obtain from test set of SQuAD, as it is not externally provided.

To highlight the practicality of our proposed method, we also measure the factorization performances with models using pre-trained ELMo (Peters et al., 2018a), as ELMo is essentially a 2-layer bidirectional LSTM Language Model that captures rich contextualized representations.

Using the same publicly released pre-trained ELMo weights 7 as the input embedding layer of all three tasks, we train publicly available state-of-the-art models as in Peters et al. (2018a ): BiDAF (Seo et al., 2016 for SQuAD, ESIM BID3 for SNLI, and BCN (McCann et al., 2017) for SST-5.

Similar to the Language Modeling tasks, we low-rank factorize the pre-trained ELMo layer only, and compare the accuracy and F1 scores across different levels of low-rank approximation.

Note that although many of these models are based on RNNs, we factorize only the ELMo layer in order to show that our approach can effectively compress pre-trained transferable knowledge.

As we only compress the ELMo weights, and other layers of each model also have large number of parameters, the inference time is affected less than in Language Modeling tasks.

The percentage of parameters in the ELMo layer for BiDAF (SQuAD) is 59.7%, for ESIM (SNLI) 67.4%, and for BCN (SST-5) 55.3%.From TAB2 , for SST-5 and SNLI, we can see that compressing W h is in general more efficient and better performing than compressing W i , except for SVD in SST-5.

However, from Appendix TAB10 , we can see that the difference in accuracy drop is not that big.

On the other hand, for the results on SQuAD, TAB2 shows the opposite trend, in which compressing W i constantly outperforms compressing W h for all methods we experimented with.

Notice that, for SQuAD, even with using a very low rank r = 10, we see better results in compression than ISS of BID18 .

In fact, we can see that, in average, using highly compressed ELMo with BiDAF still performs better than without.

Overall, we can see that for all datasets, we achieve performances that are not significantly different from the baseline results even after compressing over more than 10M parameters.

In the previous section, we observe two interesting points: 1) Matrix Factorization (MF) works consistently better in PTB and Wiki-Text 2, but Pruning works better in ELMo for W h , 2) Factorizing W h is generally better than factorizing W i .

To answer these questions, we collected the L1 norm and Nuclear norm statistics, defined in Appendix B, and comparing among W h and W i for both PTB and ELMo.

Following the definitions, L1 and its standard deviation (std) together describe the sparsity of a matrix; a matrix with higher L1 and higher std is considered to be inherently sparser.

On the other hand, Nuclear norm approximates the rank of a matrix.

Using these measures, we DISPLAYFORM0 compare pruning and MF with different compression rates, or factorization ranks, reveal interesting property of the compression algorithms.

From the results in Section 3, we observed that MF performs better than Pruning in compressing W i for high compression ratios.

FIG0 shows rank r versus L1 norm and its standard deviation, in both PTB and ELMo 8 .

The first notable pattern from Panel (a) is that MF and Pruning have diverging values from r ≤ 200.

We can see that Pruning makes the std of L1 lower than the uncompressed, while MF monotonically increases the std from uncompressed baseline.

This means that as we approximate to lower ranks (r ≤ 200), MF retains more salient information, while Pruning loses some of those salient information.

This can be clearly shown from Panel (c), in which Pruning always drops significantly more in L1 than MF does.

These statistics explain why MF consistently outperforms Pruning for higher compression ratios in W i .

The explanation and results for W h are also consistent in both PTB and WT2; MF works better than Pruning for higher compression ratios.

On the other hand, results from TAB1 show that Pruning works better than MF in W h of ELMo even in higher compression ratios.

We can explain this seemingly anomalous behavior with Panels (b) and (d).

We can see from Panel (d) that L1 norms of MF and Pruning do not significantly deviate nor decrease much from the uncompressed baseline.

Meanwhile, Panel (b) reveals an interesting pattern, in which the std actually increases for Pruning and is always kept above the uncompressed baseline.

This means that Pruning retains salient information for W h , while keeping the matrix sparse.

This behavior of W h can be explained by the nature of the compression and with inherent matrix sparsity.

In this setting, pruning is zeroing values already close to zero, so it is able to keep the L1 stable while increasing the std.

On the other hand, MF instead reduces noise by pushing lower values to be even lower (or zero) and keeps salient information by pushing larger values to be even larger.

This pattern is more evident in FIG1 , in which you can see a clear salient red line in W h that gets even stronger after factorization (U h ×V h ).

Naturally, when the compression rate is low (e.g. r=300) pruning is more efficient strategy then MF.W i versus W h Another notable pattern we observed is that compressing W h , in general, shows better performance than compressing W h .

We show the change in Nuclear norm and their corresponding starting points (i.e. uncompressed) in FIG0 Panels (e) and (f) to answer this question.

Notably, W h have consistently lower nuclear norm in both LM and ELMo compared to W i .

This difference is larger for LM (PTB), in which W i N uc is twice of that of W h N uc .

As mentioned above, having a lower nuclear norm is often an indicator of low-rank in a matrix; hence, we hypothesize that W h is inherently low-rank than W i in general.

We confirm this from Panel (d), in which even with very high compression ratio (e.g. r = 10), the L1 norm does not decrease that much.

This explains the large gap in performance between the compression of W i and W h .

On the other hand, in ELMo, this gap in norm is lower, which also shows smaller differences in performance between W i and W h , and also sometimes even the opposite (i.g.

SQuAD).

Hence, we believe that smaller (Hong et al., 2016; Li et al., 2017) by exploiting the clustering intepretations of NMF.

Semi-NMF, proposed by BID7 , relaxed the constraints of NMF to allow mixed signs and extend the possibility to be applied in non-negative cases.

BID17 proposed a variant of the Semi-NMF to learn low-dimensional representation through a multi-layer structure.

Miceli Barone (2018) proposed to replace GRUs with low-rank and diagonal weights to enable low-rank parameterization of LSTMs.

Kuchaiev & Ginsburg (2017) modifed LSTM structure by replacing input and hidden weights with two smaller partitions to boost the training and inference time.

On the other hand, compression techniques can also be applied as post-processing steps.

BID13 investigated low-rank factorization on standard LSTM model.

The Tensor-Train method has been used to train end-to-end high-dimensional sequential video data with LSTM and GRU BID21 BID16 .

In another line of work, See et al. (2016) explored pruning in order to reduce the number of parameters in Neural Machine Translation.

BID18 proposed to zero out the weights in the network learning blocks to remove insignificant weights of the RNN.

Meanwhile, BID18 proposed to binarize LSTM Language Models.

Finally, BID15 proposed to use all pruning, quantization, and Huffman coding to the weights on AlexNet.

In conclusion, we exhaustively explored the limits of compressing LSTM gates using low-rank matrix factorization and pruning in four different NLP tasks.

Our experiment results and norm analysis show that show that Low-Rank Matrix Factorization works better in general than pruning, but if the matrix is particularly sparse, Pruning works better.

We also discover that inherent low-rankness and low nuclear norm correlate well, explaining why compressing multiplicative recurrence works better than compressing additive recurrence.

In future works, we plan to factorize all LSTMs in the model, e.g. BiDAF model, and try to combine both Pruning and Matrix Factorization.

In this section, we provide the semi-NMF algorithm and we elaborate the optimization and the aim of each step.

This algorithm is an extension of NMF, where the data matrix is remained unconstrained BID7 .

The original NMF optimization function shows in 8.

Semi-NMF ignores the constraint in U as showed in 9.

DISPLAYFORM0 Exploring the relationships between matrix factorization and K-means clustering has implications for the interpretability of matrix factors Ding et al. FORMULA0 1.

Initialize U and run k-means clustering Hartigan & Wong (1979) .

DISPLAYFORM1 The objective function of k-means clustering DISPLAYFORM2 We can relax the range of v ki over the values in (0, 1) or (0, ∞).

This restricts V to accept only nonnegative values and allow U to have mixed signs values.2.

Update U by fixing V using this constraint.

By fixing V, the solution of U can be obtained by calculating the derivative of dJ/dU = −2WV T + 2U VV T = 0.

Then we can get the DISPLAYFORM3 The positive and negative parts are computed A DISPLAYFORM4 According to BID7 , this method will reach convergence.

By fixing U, the residual ||W − UV T || 2 will decrease monotonically, and after fixing V, we get the optimal solution for the objective function.

The algorithm is computed by using an iterative updating algorithm that alternates between the update of U and V BID7 .

The steps are very similar to coordinate descent Luo & Tseng (1992) with some modifications.

The optimization is convex in U or V, not both.

In the latent space derived by the NMF factorization family, each axis captures the centroid of a particular cluster, and each sample is represented as an additive combination of the centroids.

The cluster membership of each document can be easily determined by finding the corresponding cluster centroid (the axis) with which the document has the largest projection value.

Note in particular that the result of a K-means clustering run can be written as a matrix factorization W = UV , where W ∈ R nm is the data matrix, U ∈ R nr contains the cluster centroids, and V ∈ R rm contains the cluster membership indicators.• Perform the NMF or semi-NMF on W to obtain the two non-negative matrices U and V.• Matrix U contains r n−dimensional cluster centers and matrix V contains membership weight for each of the m samples in each of the r clusters.

One can assign data i to the cluster c if c = argmax j V ij .

DISPLAYFORM5 The algorithm complexity in terms of time and memory is shown in TAB4 .

The L1 Norm of any matrix W ∈ R m×n , induced from the vector L1 Norm and called the maximum absolute column sum norm, and the standard deviation of the L1 norm are defined as such, DISPLAYFORM0 L1 norm is basically the maximum of the L1 vector norms of all column vectors in W. The standard deviation of L1 norm calculates the standard deviation across the column-wise L1 norms..

These values indicate how high the values are in terms of magnitude, and how much variance the column vector norms have.

In other words, if the σ( W 1 ) is high and W 1 is high, the matrix can be considered to be sparser.

The nuclear norm of any matrix W ∈ R m×n , is defined as the sum of singular values as following, DISPLAYFORM0 The nuclear norm is often an approximation of the rank of the given matrix; a low nuclear norm indicates low rank.

For evaluating the performance of the compression we define efficiency measure as: DISPLAYFORM0 where M represent any evaluation metric (i.e. Accuracy, F1-score, Perplexity 10 ), P represents the number of parameters 11 , and R(a, b) = a−b a where a = max(a, b), i.e. the ration.

This indicator shows the ratio of loss in performance versus the loss in number of parameter.

Hence, an efficient compression holds a very small E since the denominator, P − P r , became large just when the number of parameter decreases, and the numerator, M − M r , became small only if there is no loss in the considered measure.

In some cases E became negative if there is an improvement.

9 Pruning speed and memory depends on the desired sparsity defined as r(m+n) mn 10 Note that for Perplexity, we use R(M r , M ) instead, because lower is better.

11 P r and M r are the parameter and the measure after semi-NMF of rank r

In this section, we explain the fine-tuning steps and how we tune our hyper-parameters.

This step is intended to improve the performance of our models.

After several attempts of fine-tuning, we figured out that we need to apply different initial learning rates for every rank to quickly reach the convergence point, especially after matrix factorization.

During every step, Asynchronous Stochastic Gradient Descent Merity et al. FORMULA0 is used as the optimizer.

TAB5 shows the hyper-parameters setting.

We achieved lower perplexity after fine-tuning steps and the results are shown in Figure 5 .

@highlight

We propose simple, but effective, low-rank matrix factorization (MF) algorithms to speed up in running time, save memory, and improve the performance of LSTMs.

@highlight

Proposes to accelerate LSTM by using MF as the post-processing compression strategy and conducts extensive experiements to show the performance.