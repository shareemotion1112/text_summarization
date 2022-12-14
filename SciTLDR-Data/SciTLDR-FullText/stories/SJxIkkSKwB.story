We study the problem of training machine learning models incrementally using active learning with access to imperfect or noisy oracles.

We specifically consider the setting of batch active learning, in which multiple samples are selected as opposed to a single sample as in classical settings so as to reduce the training overhead.

Our approach bridges between uniform randomness and score based importance sampling of clusters when selecting a batch of new samples.

Experiments on benchmark image classification datasets (MNIST, SVHN, and CIFAR10) shows improvement over existing active learning strategies.

We introduce an extra denoising layer to deep networks to make active learning robust to label noises and show significant improvements.

Supervised learning is the most widely used machine learning method, but it requires labelled data for training.

It is time-consuming and labor-intensive to annotate a large dataset for complex supervised machine learning models.

For example, ImageNet (Russakovsky et al., 2015) reported the time taken to annotate one object to be roughly 55 seconds.

Hence an active learning approach which selects the most relevant samples for annotation to incrementally train machine learning models is a very attractive avenue, especially for training deep networks for newer problems that have littel annotated data.

Classical active learning appends the training dataset with a single sample-label pair at a time.

Given the increasing complexity of machine learning models, it is natural to expand active learning procedures to append a batch of samples at each iteration instead of just one.

Keeping such training overhead in mind, a few batch active learning procedures have been developed in the literature (Wei et al., 2015; Sener & Savarese, 2018; Sinha et al., 2019) .

When initializing the model with a very small seed dataset, active learning suffers from the coldstart problem: at the very beginning of active learning procedures, the model is far from being accurate and hence the inferred output of the model is incorrect/uncertain.

Since active learning relies on output of the current model to select next samples, a poor initial model leads to uncertain estimation of selection criteria and selection of wrong samples.

Prior art on batch active learning suffers performance degradation due to this cold-start problem.

Most active learning procedures assume the oracle to be perfect, i.e., it can always annotate samples correctly.

However, in real-world scenarios and given the increasing usage of crowd sourcing, for example Amazon Mechanical Turk (AMT), for labelling data, most oracles are noisy.

The noise induced by the oracle in many scenarios is resolute.

Having multiple annotations on the same sample cannot guarantee noise-free labels due to the presence of systematic bias in the setup and leads to consistent mistakes.

To validate this point, we ran a crowd annotation experiment on ESC50 dataset (Piczak, 2015) : each sample is annotated by 5 crowdworkers on AMT and the majority vote of the 5 annotations is considered the label.

It turned out for some classes, 10% of the samples are annotated wrong, even with 5 annotators.

Details of the experiment can be found in Appendix A. Under such noisy oracle scenarios, classical active learning algorithms such as (Chen et al., 2015a) under-perform as shown in Figure 1 .

Motivating from these observations, we fashion a batch active learning strategy to be robust to noisy oracles.

The main contributions of this work are as follows: (1) we propose a batch sample selection method based on importance sampling and clustering which caters to drawing a batch which is simultaneously diverse and important to the model; (2) we incorporate model uncertainty into the sampling probability to compensate poor estimation of the Noise channel is assumed to be a 10-symmetric channel, where ?? is the probability of label error.

importance scores when the training data is too small to build a meaningful model; (3) we introduce a denoising layer to deep networks to robustify active learning to noisy oracles.

Main results, as shown in Fig. 3 demonstrate that in noise-free scenario, our method performs as the best over the whole active learning procedure, and in noisy scenario, our method outperforms significantly over state-of-the-art methods.

Active Learning: Active learning (Tong, 2001 ) is a well-studied problem and has gain interest in deep learning as well.

A survey summarizes various existing approaches in (Settles, 2009) .

In a nutshell, two key and diverse ways to tackle this problem in the literature are discrimination and representation.

The representation line of work focuses on selecting samples that can represent the whole unlabelled training set while the discrimination line of work aims at selecting 'tough' examples from the pool set, for example, using information theoretic scores in (MacKay, 1992) , entropy as uncertainty in (Wang & Shang, 2014) .

Along the lines of ensemble methods we have works, for example, (Beluch et al., 2018; Freund et al., 1997; Lakshminarayanan et al., 2016) .

A recent work of discrimination-based active learning (Houlsby & Ghahramani, 2011) uses mutual information, Bayesian Active Learning by Disagreement (BALD), as discriminating criteria.

In ) the authors used dropout approximation to compute the BALD scores for modern Convolutional Neural Networks (CNNs) .

However, these approaches do not consider batch acquisition and hence lack of diversity in selected batch samples causing performance lag.

Batch Active Learning: Active learning in the batch acquisition manner has been studied from the perspective of set selection and using submodularity or its variants in a variety of works.

The authors in (Wei et al., 2015) utilize submodularity for naive Bayes and nearest neighbor.

The concept of adaptive submodularity is related to active learning as well.

The problem solves adaptive greedy optimization with sequential decision making (Golovin & Krause, 2011) .

Using this concept, (Chen & Krause, 2013) considers pool-based Bayesian active learning with a finite set of candidate hypotheses.

A pool-based active learning is also discussed in (Ganti & Gray, 2011) which considered risk minimization under given hypothesis space.

The work in (Wang & Ye, 2013) uses both discriminative and representative samples to select a batch.

The authors in (Sener & Savarese, 2018) use coreset approach to select representative points of the pool set.

Recently, an adversarial learning of variational auto-encoders is used for batch active learning in (Sinha et al., 2019) .

The work make a representation of the training and pool, and adversarially select the pool representatives.

Model Uncertainty: The uncertainty for deep learning models, especially CNNs, was first addressed in (Gal & Ghahramani, 2016; Gal, 2016) using dropout as Bayesian approximation.

Model uncertainty approximation using Batch Normalization (BN) has been shown in (Teye et al., 2018) .

Both of these approaches in some sense exploit the stochastic layers (Dropout, BN) to extract model uncertainty.

The importance of model uncertainty is also emphasized in the work of (Kendall & Gal, 2017) .

The work witnesses model as well as label uncertainty which they termed as epistemic and aleatoric uncertainty, respectively.

We also address both of these uncertainties in this work.

Noisy Oracle: The importance of noisy labels from oracle has been realized in the works like (Golovin et al., 2010; Chen et al., 2015b; Chen & Krause, 2013) which utilized the concept of adap-tive submodularity for providing theoretical guarantees. (Chen et al., 2017 ) studies the same problem but with correlated noisy tests.

Active learning with noisy oracles is also studied in (Naghshvar et al., 2012; Yan et al., 2016) .

However, these work do not consider deep learning setup.

A binary classification task with the noisy oracle is considered in (Du & Ling, 2010) .

The authors in (Khetan et al., 2018 ) used a variation of Expectation Maximization algorithm to estimate the correct labels as well as annotating workers quality.

The closest work to us in the noisy oracle setting for deep learning models are (Jindal et al., 2019; 2016) .

The authors also propose to augment the model with an extra full-connected dense layer.

However, the denoising layer does not follow any probability simplex constraint, and they use modified loss function for the noise accountability along with dropout regularization.

In this section, we introduce the notations used throughout the paper.

We then formally define the problem of batch active learning with noisy oracles.

The ith (jth) row (column) of a matrix X is denoted as X i,. (X .,j ).

??? K???1 is the probability simplex of dimension K, where

For a probability vector p ??? ??? K???1 , the Shannon entropy is defined as:

The KL-divergence is always non-negative and is 0 if and only if p = q. The expectation operator is taken as E. We are concerned with a K class classification problem with a sample space X and label space Y = {1, 2, . . .

, K}. The classification model M is taken to be g ?? : X ??? Y parameterized with ??.

The softmax output of the model is given by p = softmax(g ?? (x)) ??? ??? K???1 .

The batch active learning setup starts with a set of labeled samples D tr = {(x i , y i )} and unlabeled samples P = {(x j )}.

With a query budget of b, we select a batch of unlabeled samples B as, B = ALG(D tr , M, b, P), |B| ??? b, where ALG is the selection procedure conditioned on the current state of active learning (D tr , M, b, P).

ALG is designed with the aim of maximizing the prediction accuracy E p X ??Y [(h ?? (x) = y)].

Henceforth, these samples which can potentially maximize the prediction accuracy are termed as important samples.

After each acquisition iteration, the training dataset is updated as D tr = D tr ??? {(B, y B )} where y B are the labels of B from an oracle routine.

The oracle takes an input x ??? X and outputs the ground truth label y ??? Y.

This is referred to as 'Ideal Oracle' and the mapping from x to y is deterministic.

A 'Noisy Oracle' flips the true output y to y which is what we receive upon querying x. Similar to (Chen et al., 2015a) , we assume that the label flipping is independent of the input x and thus can be characterized by the conditional probability p(y = i|y = j), where i, j ??? Y. We also refer this conditional distribution as the noisy-channel, and hence the ideal oracle has noisy channel value of 1 for i = j and 0 otherwise.

For rest of the paper, we use the noise channel as a K-symmetric channel (SC), see Figure 2b , which is a generalization of the binary symmetric channel.

The K-SC is defined as follows

where ?? is the probability of a label flip, i.e., p(y = y) = ??.

We resort to the usage of K-SC because of its simplicity, and in addition, it abstracts the oracle noise strength with a single parameter ??.

Therefore, in noisy active learning, after the selection of required subset B, the training dataset (and then the model) is updated as D tr = D tr ??? {(B, y B )}.

Next, in Section 4, we discuss the proposed solution to noisy batch active learning.

An ideal batch selection procedure so as to be employed in an active learning setup, must address the following issues, (i) select important samples from the available pool for the current model, and (ii) select a diverse batch to avoid repetitive samples.

We note that, at each step, when active learning acquires new samples, both of these issues are addressed by using the currently trained model.

However, in the event of an uncertain model, the quantification of diversity and importance of a batch of samples will also be inaccurate resulting in loss of performance.

This is often the case with active learning because we start with less data in hand and consequently an uncertain model.

Therefore, we identify the next problem in the active learning as (iii) incorporation of the model uncertainty across active learning iterations.

Batch selection: The construction of batch active learning algorithm by solving the aforementioned first two problems begins with assignment of an importance score (??) to each sample in the pool.

Several score functions exist which perform sample wise active learning.

To list a few, max-entropy, variation ratios, BALD , entropy of the predicted class probabilities (Wang & Shang, 2014) .

We use BALD as an importance score which quantifies the amount of reduction of uncertainty by incorporating a particular sample for the given model.

In principle, we wish to have high BALD score for a sample to be selected.

For the sake of completeness, it is defined as follows.

where ?? are the model parameters.

We refer the reader to for details regarding the computation of BALD score in (2).

To address diversity, we first perform clustering of the pooled samples and then use importance sampling to select cluster centroids.

For clustering, the distance metric used is the square root of the Jensen-Shannon (JS) divergence between softmax output of the samples.

Formally, for our case, it is defined as d :

With little abuse of notation, we interchangeably use d(p i , p j ) as d i,j where i, j are the sample indices and p i , p j are corresponding softmax outputs.

The advantage of using JS-divergence is two folds; first it captures similarity between probability distributions well, second, unlike KL-divergence it is always bounded between 0 and 1.

The boundedness helps in incorporating uncertainty which we will discuss shortly.

Using the distance metric as d we perform Agglomerative hierarchical clustering (Rokach & Maimon, 2005) for a given number of clusters N .

A cluster centroid is taken as the median score sample of the cluster members.

Finally, with all similar samples clustered together, we perform importance sampling of the cluster centroids using their importance score, and a random centroid c is selected as p(c = k) ??? ?? k .

The clustering and importance sampling together not only take care of selecting important samples but also ensure diversity among the selected samples.

Uncertainty Incorporation:

The discussion we have so far is crucially dependent on the output of the model in hand, i.e., importance score as well as the similarity distance.

As noted in our third identified issue with active learning, of model uncertainty, these estimations suffers from inaccuracy in situations involving less training data or uncertain model.

The uncertainty of a model, in very general terms, represents the model's confidence of its output.

The uncertainty for deep learning models has been approximated in Bayesian settings using dropout in (Gal & Ghahramani, 2016) , and batch normalization (BN) in (Teye et al., 2018) .

Both use stochastic layers (dropout, BN) to undergo multiple forward passes and compute the model's confidence in the outputs.

For example, confidence could be measured in terms of statistical dispersion of the softmax outputs.

In particular, variance of the softmax outputs, variation ratio of the model output decision, etc, are good candidates.

We denote the model uncertainty as ?? ??? [0, 1], such that ?? is normalized between 0 and 1 with 0 being complete certainty and 1 for fully uncertain model.

For rest of the work, we compute the uncertainty measure ?? as variation ratio of the output of model's multiple stochastic forward passes as mentioned in (Gal & Ghahramani, 2016 ).

In the event of an uncertain model (?? ??? 1), we randomly select samples from the pool initially.

However, as the model moves towards being more accurate (low ??) by acquiring more labeled samples through active learning, the selection of samples should be biased towards importance sampling and clustering.

To mathematically model this solution, we use the statistical mechanics approach of deterministic annealing using the Boltzmann-Gibbs distribution (Rose et al., 1990) .

In Gibbs distribution p(i) ??? e ??? i/kB T , i.e., probability of a system being in an ith state is high for low energy i states and influenced by the temperature T .

For example, if T ??? ???, then state energy is irrelevant and all states are equally probable, while if T ??? 0, then probability of the system being in the lowest energy state is almost surely 1.

We translate this into active learning as follows: For a given cluster centroid c, if the model uncertainty is very high (?? ??? 1) then all points in the pool (including c) should be equally probable to get selected (or uniform random sampling), and if the model is very certain (?? ??? 0), then the centroid c itself should be selected.

This is achieved by using the state energy analogue as distance d between Assign importance score to each x ??? P as ?? x = I(??; y|x,

Perform Agglomerative clustering of the pool samples with N (b) number of clusters using square root of JS-divergence as distance metric to get D 4:

Sample cluster centroid c from the categorical distribution

Compute uncertainty estimate ?? (t???1) of the model M (t???1) , and

Sample ?? from the Gibbs distribution p(?? = s|B (t) , c, ??

end for 10:

Query oracle for the labels of B (t) and update

Update model as

Set P ??? P \ B (t) 13: end for the cluster centroid c and any sample x in the pool, and temperature analogue as uncertainty estimate ?? of the model.

The distance metric d used by us is always bounded between 0 and 1 and it provides nice interpretation for the state energy.

Since, in the event of low uncertainty, we wish to perform importance sampling of cluster centroids, and we have d c,c = 0 (lowest possible value), therefore by Gibbs distribution, cluster centroid c is selected almost surely.

To construct a batch, the samples have to be drawn from the pool using Gibbs distribution without replacement.

In the event of samples s 1 , . . .

, s n already drawn, the probability of drawing a sample ?? given the cluster centroid c, distance matrix D = [d i,j ] and inverse temperature (or inverse uncertainty) ?? is written as

where P = P\s 1:n .

In theory, the inverse uncertainty ?? can be any f such that f : [0, 1] ??? R + ???{0} and f (??) ??? ??? as ?? ??? 0 and f (??) = 0 for ?? = 1.

For example, few possible choices for ?? (= f (??)) are ??? log(??), e 1/?? ??? 1.

Different inverse functions will have different growth rate, and the choice of functions is dependent on both the model and the data.

Next, since we have drawn the cluster centroid c according to p(c = k) ??? ?? k , the probability of drawing a sample s from the pool P is written as

We can readily see that upon setting ?? ??? 0 in (4), p(?? = s|s 1:n , ??, D) reduces to 1/|P | which is nothing but the uniform random distribution in the leftover pool.

On setting ?? ??? ???, we have ?? = c with probability ?? c / c ?? c and ?? = c with probability 0, i.e., selecting cluster centroids from the pool with importance sampling.

For all other 0 < ?? < ??? we have a soft bridge between these two asymptotic cases.

The approach of uncertainty based batch active learning is summarized as Algorithm 1.

Next, we discuss the solution to address noisy oracles in the context of active learning.

The noisy oracle, as defined in Section 3, has non-zero probability for outputting a wrong label when queried with an input sample.

To make the model aware of possible noise in the dataset originating from the noisy oracle, we append a denoising layer to the model.

The inputs to this denoising layer are the softmax outputs p of the original model.

Figure 2a demonstrates the proposed solution for deep learning classification models.

The denoising layer is a fully-connected K??K dense layer with 1: for t = 1, 2, . . .

, T do 2:

Query noisy oracle for the labels of B (t) and update

Get M * (t)

??? M (t) appended with noisy-channel layer at the end 5:

Update noisy model as M * (t) using D Detach required model M (t) from M * (t) by removing the final noisy-channel layer 7:

Set P ??? P \ B (t) 8: end for weights W = [w i,j ] such that its output p = Wp.

The weights w i,j represent the noisy-channel transition probabilities such that w i,j = p(y = i|y = j).

Therefore, to be a valid noisy-channel, W is constrained as W ??? {W | W .,j ??? ??? K???1 , ??? 1 ??? j ??? K}. While training we use the model upto the denoising layer and train using p , or label prediction y while for validation/testing we use the model output p or label prediction y.

The active learning algorithm in the presence of noisy oracle is summarized as Algorithm 2.

We now proceed to Section 5 for demonstrating the efficacy of our proposed methods across different datasets.

We evaluate the algorithms for training CNNs on three datasets pertaining to image classification; (i) MNIST (Lecun et al., 1998) , (ii) CIFAR10 (Krizhevsky, 2009) , and (iii) SVHN (Netzer et al., 2011) .

We use the CNN architectures from (fchollet, 2015; .

For all the architectures we use Adam (Kingma & Ba, 2014) with a learning rate of 1e ??? 3.

The implementations are done on PyTorch (Paszke et al., 2017) , and we use the Scikit-learn (Pedregosa et al., 2011) package for Agglomerative clustering.

For training the denoising layer, we initialize it with the identity matrix I K , i.e., assuming it to be noiseless.

The number of clusters N (b) is taken to be as 5b .

The uncertainty measure ?? is computed as the variation ratio of the output prediction across 100 stochastic forward passes, as coined in (Gal & Ghahramani, 2016) , through the model using a validation set which is fixed apriori.

The inverse uncertainty function ?? = f (??) in Algorithm 1 is chosen from l (e 1/?? ??? 1), ???l log(??), where l is a scaling constant fixed using cross-validation.

The cross-validation is performed only for the noise-free setting, and all other results with different noise magnitude ?? follow this choice.

This is done so as to verify the robustness of the choice of parameters against different noise magnitudes which might not be known apriori.

We compare our approach with: (i) Random: A batch is selected by drawing samples from the pool uniform at random without replacement. (ii) BALD: Using model uncertainty and the BALD score, the authors in ) do active learning with single sample acquisition.

We use the highest b scoring samples to select a batch. (iii) Coreset: The authors in (Sener & Savarese, 2018) proposed a coreset based approach to select the representative core centroids of the pool set.

We use the 2 ??? OP T approximation greedy algorithm of the paper with similarity measure as l 2 norm between the activations of the penultimate layer. (iv) Entropy: The approach of (Wang & Shang, 2014 ) is implemented via selecting b samples with the highest Shannon entropy H(p) of the softmax outputs.

(v) VAAL: The variational adversarial active learning of (Sinha et al., 2019) .

In all our experiments, we start with a small number of images 40 ??? 50 and retrain the model from scratch after every batch acquisition.

In order to make a fair comparison, we provide the same initial point for all active learning algorithms in an experiment.

We perform a total of 20 random initializations and plot the average performance along with the standard deviation vs number of acquired samples by the algorithms.

Figure 3 shows that our proposed algorithm outperform all the existing algorithms.

As an important observation, we note that random selection always works better in the initial stages of all experiments.

This observation is explained by the fact that all models suffer from inaccurate predictions at the initial stages.

The proposed uncertainty based randomization makes a soft bridge between uniform random sampling and score based importance sampling of the cluster centroids.

The proposed approach uses randomness at the initial stages and then learns to switch to weigh the model based inference scores as the model becomes increasingly certain of its output.

Therefore, the proposed algorithm always envelops the performance of all the other approaches across all three datasets of MNIST, CIFAR10, and SVHN.

Figure 3 also shows the negative impact of noisy oracle on the active learning performance across all three datasets.

The degradation in the performance worsens with increasing oracle noise strength ??.

We see that doing denoisification by appending noisy-channel layer helps combating the noisy oracle in Figure 3 .

The performance of the proposed noisy oracle active learning is significantly better in all the cases.

The prediction accuracy gap between algorithm with/without denoising layer elevates with increase in the noise strength ??.

The most recent baselines like (VAAL (Sinha et al., 2019) ), (Coreset (Sener & Savarese, 2018) ) which make representation of the Training + Pool may not always perform well.

While coreset assigns distance between points based on the model output which suffers in the beginning, VAAL uses training data only to make representations together with the remaining pool in GAN like setting.

The representative of pool points may not always help, especially if there are difficult points to label and the model can be used to identify them.

In addition to the importance score, the model uncertainty is needed to assign a confidence to its judgement which is poor in the beginning and gets strengthened later.

The proposed approach works along this direction.

Lastly, while robustness against oracle noise is discussed in (Sinha et al., 2019) , however, we see that incorporating the denoising later implicitly in the model helps better.

The intuitive reason being, having noise in the training data changes the discriminative distribution from p(y|x) to p(y |x).

Hence, learning p(y |x) from the training data and then recovering p(y|x) makes more sense as discussed in Section 4.2.

The uncertainty measure ?? plays a key role for the proposed algorithm.

We have observed that under strong noise influence from the oracle, the model's performance is compromised due to spurious training data as we see in Figure3.

This affects the estimation of the uncertainty measure (variation ratio) as well.

We see in Figure 4 that the model uncertainty does not drop as expected due to the label noise.

However, the aid provided by the denoising layer to combat the oracle noise solves this issue.

We observe in Figure 4 that uncertainty drops at a faster rate as the model along with the denoising layer gets access to more training data.

Hence, the proposed algorithm along with the denoising layer make better judgment of soft switch between uniform randomness and importance sampling using (4).

The availability of better uncertainty estimates for modern deep learning architectures is a promising future research, and the current work will also benefit from it.

In this paper we have proposed a batch sample selection mechanism for active learning with access to noisy oracles.

We use mutual information between model parameters and the predicted class probabilities as importance score for each sample, and cluster the pool sample space with JensonShannon distance.

We incorporate model uncertainty/confidence into Gibbs distribution over the clusters and select samples from each cluster with importance sampling.

We introduce an additional layer at the output of deep networks to estimate label noise.

Experiments on MNIST, SVHN, and CIFAR10 show that the proposed method is more robust against noisy labels compared with the state of the art.

Even in noise-free scenarios, our method still performs the best for all three datasets.

Our contributions open avenues for exploring applicability of batch active learning in setups involving imperfect data acquisition schemes either by construction or because of resource constraints.

Under review as a conference paper at ICLR 2020 A ESC50 CROWD LABELING EXPERIMENT

We selected 10 categories of ESC50 and use Amazon Mechanical Turk for annotation.

In each annotation task, the crowd worker is asked to listen to the sound track and pick the class that the sound belongs to, with confidence level.

The crowd worker can also pick "Unsure" if he/she does not think the sound track clearly belongs to one of the 10 categories.

For quality control, we embed sound tracks that clearly belong to one class (these are called gold standards) into the set of tasks an annotator will do.

If the annotator labels the gold standard sound tracks wrong, then labels from this annotator will be discarded.

The confusion table of this crowd labeling experiment is shown in Figure 5 : each row corresponds to sound tracks with one ground truth class, and the columns are majority-voted crowd-sourced labels of the sound tracks.

We can see that for some classes, such as frog and helicopter, even with 5 crowd workers, the majority vote of their annotation still cannot fully agree with the ground truth class.

We present rest of the experimental results supplementary to the ones presented in the main body of Section 5.

The active learning algorithm performance for oracle noise strength of ?? = 0.2 and ?? = 0.4 are presented in Figure 6 .

Similarly to what discussed in Section 5, we observe that the performance of proposed algorithm dominates all other existing works for ?? = 0.2.

We witnessed that the proposed algorithm performance (without denoising layer) is not able to match other algorithms (BALD and Entropy) when ?? = 0.4, even with more training data.

The reason for this behavior can be explained using the uncertainty measure ?? output in the Figure 7 .

We see that under strong noise influence from the oracle, the model uncertainty doesn't reduce along the active learning acquisition iterations.

Because of this behavior, the proposed uncertainty based algorithm sticks to put more weightage on uniform random sampling, even with more training data.

However, we see that using denoising layer, we have better model uncertainty estimates under the influence of noisy oracle.

Since the uncertainty estimates improve, as we see in Figure 7 , for ?? = 0.4, the proposed algorithm along with the denoising layer performs very well and has significant improvement in performance as compared to other approaches.

The results for CIFAR10 dataset with oracle noise strength of ?? = 0.2 and 0.4 are provided in the Figure 8 .

We see that the proposed algorithm without/with using the denoising layer outperforms other benchmarks.

We provide the active learning accuracy results for SVHN dataset with oracle noise strength of ?? = 0.2 and 0.4 in the Figure 8 .

Similar to other results, we see that the proposed algorithm without/with using the denoising layer outperforms other benchmarks for ?? = 0.2.

For oracle noise strength of ?? = 0.4, we see a similar trend as MNIST regarding performance compromise to the proposed uncertainty based batch selection.

The reason is again found in the uncertainty estimates plot in Figure 10 for ?? = 0.4.

With more mislabeled training examples, the model uncertainty estimate doesn't improve with active learning samples acquisition.

Hence, the proposed algorithm makes the judgment of staying close to uniform random sampling.

However, unlike MNIST in Figure 7 , the uncertainty estimate is not that poor for SVHN, i.e., it still decays.

Therefore, the performance loss in proposed algorithm is not that significant.

While, upon using the denoising layer, the uncertainty estimates improve significantly, and therefore, the proposed algorithm along with the denoising layer outperforms other approaches by big margin.

Using the same setup as explained in Section 5, we evaluate the performance on CIFAR100 (Krizhevsky, 2009 ) dataset for various active learning algorithms listed in Section 5.2.

We observe in Figure 11 that the proposed uncertainty based algorithm perform similar or better than the baselines.

The incorporation of denoising layer helps in countering the affects of noisy oracle as we demonstrate by varying the noise strength ?? = 0.1, 0.3.

For a quantitative look at the active learning results, mean and standard deviation of the performance vs. acquisition, in the Figure 3 , we present the results in the tabular format in Table 1 for MNIST,  Table 2 for CIFAR10, Table 3 for SVHN, and Table 4 for CIFAR100, respectively.

Table 3 Active learning results for SVHN dataset.

@highlight

We address the active learning in batch setting with noisy oracles and use model uncertainty to encode the decision quality of active learning algorithm during acquisition.