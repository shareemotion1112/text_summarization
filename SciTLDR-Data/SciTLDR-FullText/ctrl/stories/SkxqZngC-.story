Topic modeling of text documents is one of the most important tasks in representation learning.

In this work, we propose iTM-VAE, which is a Bayesian nonparametric (BNP) topic model with variational auto-encoders.

On one hand, as a BNP topic model, iTM-VAE potentially has infinite topics and can adapt the topic number to data automatically.

On the other hand, different with the other BNP topic models, the inference of iTM-VAE is modeled by neural networks, which has rich representation capacity and can be computed in a simple feed-forward manner.

Two variants of iTM-VAE are also proposed in this paper, where iTM-VAE-Prod models the generative process in products-of-experts fashion for better performance and iTM-VAE-G places a prior over the concentration parameter such that the model can adapt a suitable concentration parameter to data automatically.

Experimental results on 20News and Reuters RCV1-V2 datasets show that the proposed models outperform the state-of-the-arts in terms of perplexity, topic coherence and document retrieval tasks.

Moreover, the ability of adjusting the concentration parameter to data is also confirmed by experiments.

Probabilistic topic models focus on discovering the abstract "topics" that occur in a collection of documents and representing a document as a weighted mixture of the discovered topics.

Classical topic models, the most popular being LDA BID2 , have achieved success in a range of applications, such as information retrieval BID41 , document understanding BID2 , computer vision BID32 and bioinformatics BID34 .

A major challenge of topic models is that the inference of the distribution over topics does not have a closed-form solution and must be approximated, using either MCMC sampling or variational inference.

Hence, any small change to the model requires re-designing a new inference method tailored for it.

Moreover, as the model grows more expressive, the inference becomes increasingly complex, which becomes the bottleneck to discover the latent semantic structures of complicated data.

Hence, black-box inference methods BID31 BID26 BID16 BID33 , which require only limited knowledge from the models and can be flexibly applied to new models, is desirable for topic models.

Among all the black-box inference methods, Auto-Encoding Variational Bayes (AEVB) BID16 BID33 ) is a promising one for topic models.

AEVB contains an inference network that can map a document directly to a variational posterior without the need for further local variational updates on test data, and the Stochastic Gradient Variational Bayes (SGVB) estimator allows efficient approximate inference for a broad class of posteriors, which makes topic models more flexible.

Hence, an increasing number of work has been proposed recently to combine topic models with AEVB, such as BID23 BID37 BID5 BID24 .Deciding the number of topics is another challenge for topic models.

One option is to use model selection, which trains models with different topic numbers and selects the best on the validation set.

Bayesian nonparametric (BNP) topic models, however, side-step this issue by making the number of topics adaptive to data.

For example, BID39 proposed Hierarchical Dirichlet Process (HDP), which models each document with a Dirichlet Process (DP) and all DPs for documents in a corpus share a base distribution that is itself also from a DP.

HDP extends LDA in that it can adapt the number of topics to data.

Hence, HDP has potentially an infinite number of topics and allows the number to grow as more documents are observed.

Unlike the black-box inference based models, traditionally, one needs to redesign the inference methods when there are some changes in the generative process of HDP BID39 BID40 BID11 .In this work, we make progress on this problem by proposing an infinite Topic Model with Variational Auto-Encoders (iTM-VAE), which is a Bayesian nonparametric topic model with AEVB.

Coupling Bayesian nonparametric techniques with deep neural networks, iTM-VAE is able to capture the uncertainty regarding to the number of topics, and the inference can be conducted in a simple feed-forward manner.

More specifically, iTM-VAE uses a stick-breaking process BID35 to generate the mixture weights for a countably infinite set of topics, and use neural networks to approximate the variational posteriors.

The main contributions of the paper are:??? We propose iTM-VAE, which, to our best knowledge, is the first Bayesian nonparametric topic model equipped with AEVB.??? We propose iTM-VAE-Prod whose distribution over words is a product of experts rather than a mixture of multinomials.??? We propose iTM-VAE-G, which helps the model to adjust the concentration parameter to data automatically.??? The experimental results show that iTM-VAE and its two variants outperform the state-ofthe-art models on two challenging benchmarks significantly.

Topic models have been studied extensively in a variety of applications such as document modeling, information retrieval, computer vision and bioinformatics BID2 BID41 BID30 BID32 BID34 .

Please see BID1 for an overview.

Recently, with the impressive success of deep learning, neural topic models have been proposed and achieved encouraging performance in document modeling tasks, such as Replicated Softmax BID9 ), DocNADE BID18 , fDARN BID26 and NVDM BID23 .

These models achieved competitive performance on modeling documents.

However, they do not explicitly model the generative story of documents, hence are less explainable.

Several recent work has been proposed to model the generative procedure explicitly and the inference of the topic distributions is computed by deep neural networks.

This makes these models interpretable, powerful and easily extendable.

For example, BID37 proposed AVITM model, which embeds the original LDA BID2 formulation with AEVB.

By utilizing Laplacian approximation for the Dirichlet distribution, AVITM can be optimized by the Stochastic Gradient Variational Bayes (SGVB) BID16 BID33 estimator efficiently.

AVITM achieved the state-of-the-art performance on the topic coherence metric BID20 , which indicates the topics learned match closely to human judgment.

The Bayesian nonparametric topic models BID39 ; BID13 BID0 ; BID21 , potentially have infinite topic capacity and are able to adapt the topic number to data.

However, we do not notice any topic models that combine BNP techniques with deep neural networks.

Actually, BID28 proposed Stick-Breaking VAE (SB-VAE), which is a Bayesian nonparametric version of traditional VAE with a stochastic dimensionality.

However, iTM-VAE differs with SB-VAE in that it is a kind of topic model that models discrete text data.

Furthermore, we also proposed iTM-VAE-G which places a prior on the stickbreaking process such that the model is able to adapt the concentration parameter to data.

Another related work is BID24 , which proposed GSM, GSB, RSB and RSB-TF to model documents.

The RSB-TF from BID24 , which uses a heuristic indicator to guide the growth of the topic numbers, also has the ability to adapt the topic number.

However, the performance of RSB-TF does not match its complexity.

Instead, iTM-VAE exploits Bayesian nonparametric techniques to decide the number of topics, which is much more elegant.

And the performance of iTM-VAE also outperforms RSB-TF.

In this section, we briefly describe the stick-breaking process BID35 BID27 , Variational Auto-Encoders BID16 BID33 and the Kumaraswamy distribution BID17 , which are all essential for iTM-VAE.

We first describe the stick-breaking process, which is used to model the mixture weights over the countably infinite topics for the generative procedure of iTM-VAE.

More specifically, the stickbreaking process generates an infinite sequence of mixture weights ?? = {?? k } ??? k=1 as follows: DISPLAYFORM0 This is often denoted as ?? ??? GEM(??), where GEM stands for Griffiths, Engen and McCloskey BID7 and ?? is the concentration parameter.

One can see that ?? k satisfies 0 ??? ?? k ??? 1 and DISPLAYFORM1

Variational Auto-Encoder (VAE) is among the most successful generative models recently.

Specifically, it assumes that a sample x is generated mathematically as follows: DISPLAYFORM0 where p(z) is the prior of the latent representation z and p ?? (x|z) is a conditional distribution with parameters ?? to generate x. The approximation to the posterior of the generative process, q ?? (z|x), is parametrized by an inference neural network with parameters ??.

According to BID16 BID33 , the Evidence Lower Bound (ELBO) of VAE can be written as: DISPLAYFORM1 where KL is the Kullback-Leibler divergence.

The parameters ?? and ?? can be optimized jointly by applying the Stochastic Gradient Variational Bayes (SGVB) BID16 estimator and Reparameterization Trick (RT).

An essential requirement of SGVB and RT is that the latent variable z can be represented in a differentiable, non-centered parameterization (DNCP) BID15 , which allows the gradients to go through the MC expectation.

However, this is not always satisfied by the Beta prior of the stick-breaking process for iTM-VAE.

Hence, we resort to the little known Kumaraswamy distribution BID17 to solve this problem.

Kumaraswamy distribution BID17 ) is a continuous distribution, which is mathematically defined as: DISPLAYFORM0 where x ??? [0, 1] and a, b > 0.

The inverse cumulative distribution function (CDF) can be expressed in a simple and closed-form formulation, and samples from Kumaraswamy distribution can be drawn by: DISPLAYFORM1 Kumaraswamy is similar to Beta distribution, yet more suitable to SGVB since it satisfies the DNCP requirement.

Moreover, the KL divergence between Kumaraswamy and Beta can be closely approximated in closed-form.

Hence, we use it to model the inference procedure of iTM-VAE.

In this section, we describe iTM-VAE, a Bayesian nonparametric topic model with VAE.

Specifically, we first describe the generative process of iTM-VAE in Section 4.1, and then the inference process is introduced in Section 4.2.

After that, two variants of the model, iTM-VAE-Prod and iTM-VAE-G, are described in Section 4.3 and Section 4.4, respectively.

The generative process of iTM-VAE is similar to the original VAE.

The key difference is that the DISPLAYFORM0 is drawn from the GEM distribution to make sure of ??? k=1 ?? k = 1 and 0 ??? ?? k ??? 1.

Specifically, we suppose each topic ?? k = ??(?? k ) is a probability distribution over vocabulary, where ?? k ??? R V is the parameter of the topic-specific word distribution, ??(??) is the softmax function and V is the vocabulary size.

In iTM-VAE, there are unlimited number of topics and we denote ?? = {?? k } ??? k=1 and ?? = {?? k } ??? k=1 as the collections of these countably infinite topics and the corresponding parameters.

The generation of a document by iTM-VAE can then be mathematically described as: DISPLAYFORM1 where ?? is a hyper-parameter, Cat(?? i ) is a categorical distribution parameterized by?? i , and ?? ?? k (??) is a discrete dirac function, which equals to 1 when ?? = ?? k and 0 otherwise.

Thus, the joint probability of a document with N words w 1: DISPLAYFORM2 , the topic distribution ?? and the sampled topics?? 1: DISPLAYFORM3 can be written as: DISPLAYFORM4 where p(??|??) = GEM(??), p(??|??, ??) = G(??; ??, ??) and p(w|??) = Cat(??).Similar to BID37 , we collapse the variable?? 1:N and rewrite Equation 6 as: DISPLAYFORM5 where p(w i |??, ??) = CAT(??) and?? = ??? k=1 ?? k ?? k .

In practice, following BID24 , we factorize the parameter ?? k of topic ?? k as ?? k = t k W where t k ??? R H is the k-th topic factor vector, W ??? R H??V is the word factor matrix and H ??? R + is the factor dimension.

For simplicity, we still use ?? to denote all the parameters regarding to the generative procedure of iTM-VAE.

Note that, although we parameterize the generative process with ??, iTM-VAE is still a nonparametric model, since it has potentially infinite model capacity, and can grow the number of parameters with the amount of training data.

Notably, different with traditional nonparametric Bayesian topic models, the topics in iTM-VAE are not drawn from a base distribution, but are treated as part of the parameters of the model and are optimized directly.

This key difference indicates that there is no need to use an additional base distribution to generate the countably infinite candidate topics such that these topics are shared by different documents.

Instead, the topic parameters of iTM-VAE are shared across all documents naturally.

In this section, we describe the inference process of iTM-VAE, i.e. how to draw ?? given a document w 1:N .

To elaborate, suppose ?? = [?? 1 , ?? 2 , . . . , ?? K???1 ] is a K ??? 1 dimensional vector where ?? k is a random variable sampled from a Kumaraswamy distribution ??(??; a k , b k ) parameterized by a k and b k , iTM-VAE models the joint distribution q ?? (??|w 1:N ) as: DISPLAYFORM0 where g(w 1:N ; ??) is a neural network with parameters ??.

Then, ?? = {?? k } K k=1 can be drawn by: DISPLAYFORM1 In the above procedure, we truncate the infinite sequence of mixture weights ?? = {?? k } ??? k=1 by K elements, and ?? K is always set to 1 to ensure K k=1 ?? k = 1.

Notably, as discussed in , the truncation of variational posterior does not indicate that we are using a finite dimensional prior, since we never truncate the GEM prior.

Moreover, the truncation level K is not part of the generative procedure specification.

Hence, iTM-VAE still has the ability to model the uncertainty of the number of topics and adapt it to data.

People can manage to use truncation-free posteriors in the model, however, as observed by BID28 , it does not work well.

On the opposite, the truncated-fashion posterior of iTM-VAE is simple and works well in practice.iTM-VAE can be optimized by maximizing the Evidence Lower Bound (ELBO): DISPLAYFORM2 where we replace ?? with ?? for p(w 1:N |??, ??) to emphasize that ?? is the parameter to be optimized, and p(??|??) is products of K ??? 1 Beta(1, ??) distributions according to Section 3.1.

The details of the optimization can be found in Appendix 7.2.

In Equation 7,?? is a mixture of multinomials.

One drawback of this formulation is that it cannot make any predictions that are sharper than the distributions being mixed, pointed out by BID9 , which may result in some topics that are of poor quality and do not match well with human judgment.

One solution to this issue is to replace the mixture of multinomials with a weighted product of experts which is able to make sharper predictions than any of the constituent experts BID8 .

We develop a products-of-experts version of iTM-VAE, which is referred as iTM-VAE-Prod.

Specifically, we compute a mixed topic distribution?? = ??( ??? k=1 ?? k ?? k ) for each document, where ?? k is sampled from GEM(??), and then each word of the document is sampled from Cat(??).

The benefit of the products-of-experts is demonstrated in Section 5.

In the generative process, the concentration parameter ?? of GEM(??) can have significant impact on the growth of number of topics.

The larger the ?? is, the more "breaks" it will create, and consequently, more topics will be used.

Hence, it is generally reasonable to consider placing a prior on ?? so that the model can adjust the concentration parameter to data automatically.

Concretely, since the Gamma distribution is conjugate to Beta(1, ??), we place a Gamma(s 1 , s 2 ) prior on ??.

Then the ELBO of iTM-VAE can be written as: DISPLAYFORM0 where p(??|s 1 , s 2 ) = Gamma(s 1 , s 2 ), p(v k |??) = Beta(1, ??), q(??|?? 1 , ?? 2 ) is the variational posterior for ?? with ?? 1 , ?? 2 as parameters across the whole dataset.

The derivation for Equation 13 can be found in Appendix 7.3.

In experiments, we find iTM-VAE-Prod always performs better than iTM-VAE, therefore we only place the prior for iTM-VAE-Prod, and refer this variant as iTM-VAE-G.

In this section, we evaluate the performance of iTM-VAE and its variants on two public benchmarks: 20News and RCV1-V2.

To make a fair comparison, we use exactly the same data and vocabulary as BID37 .

2 We compare iTM-VAE and its variants with several state-of-the-arts, such as DocNADE, NVDM, NVLDA and ProdLDA BID37 , GSM, GSB, RSB and RSB-TF, as well as some classical topic models such as LDA and HDP.The configuration of the experiments is as follows.

We use a two-layer fully-connected neural network for g(w 1:N ; ??) of Equation 8, and the number of hidden units is set to 256 and 512 for 20News and RCV1-V2, respectively.

The factor dimension is set to 200 and 1000 for 20News and RCV1-V2, respectively.

The truncation level K in Equation 11 is set to 200 so that the maximum topic numbers will never exceed the ones used by baselines.3 Batch-Normalization BID12 is used to stabilize the training procedure.

The hyper-parameter ?? for GEM distribution is cross-validated on validation set from [10, 20, 30, 50, 100] .

We use Adam BID14 to optimize the model and the learning rate is set to 0.01 for all experiments.

The code of iTM-VAE and its variants is available at http://anonymous.

Perplexity is widely used by topic models to measure the goodness-to-fit capability, which is defined as: DISPLAYFORM0 , where D is the number of documents , and |w d | is the number of words in the d-th document w d .

Following previous work, the variational lower bound is used to estimate the perplexity.

TAB1 shows the perplexity of different topic models on 20News and RCV1-V2 datasets.

Among these baselines, RSB and GSM BID24 achieved the lowest perplexities on 20News and RCV1-V2, which are 785 and 521, respectively.

While the perplexities achieved by iTM-VAE-Prod on these two benchmarks are 769 and 508, respectively, which performs better than the state-of-theart.

Moreover, FIG0 -(a) demonstrates perplexities of finite topic models with different number of topics on 20News.

We can see that, a suitable topic number of these models should be around 10 and 20.

Interestingly but as expected, the number of effective topics 4 discovered by iTM-VAE is about 19, which indicates that the Bayesian nonparametric topic model, iTM-VAE, has the ability to determine a suitable number of topics automatically.

As the quality of the learned topics is not directly reflected by perplexity BID29 , topic coherence is designed to match the human judgment.

As BID20 showed that Normalized Pointwise Mutual Information (NPMI) matches the human judgment most closely, we adopt it as the measurement of topic coherence , same as BID24 BID37 .5 Since the number of topics discovered by iTM-VAE is dynamic, we define the Effective Topic as the topic which becomes the top-1 significant topic of a training sample among the training set more than ?? ?? D times, where D is the number of training samples and ?? is a ratio, which is set to 0.5% in our experiments.

Following BID24 , we use an average over topic coherence computed by top-5 and top-10 words for all topics across five random runs, which is more stable and robust BID19 .

TAB1 shows the topic coherence of different topic models on 20News and RCV1-V2 datasets.

We can clearly see that iTM-VAE-Prod outperforms all the other topic models significantly, which indicates that the topics discovered by iTM-VAE-Prod match more closely to human judgment.

Some topics learned by iTM-VAE-Prod are illustrated in Appendix 7.1.

We also illustrate the topic coherence of finite topic models with different numbers of topics, and compare them with iTM-VAE-Prod and iTM-VAE on 20News dataset, shown in FIG0 .

The topic coherence of iTM-VAE-Prod outperforms all baselines over all topic numbers.

Another observation is that the best topic coherence of ProdLDA is achieved as the topic number is 15, which is close to the number of effective topics discovered by iTM-VAE-Prod.

The document retrieval task is to evaluate the discriminative power of the document representations learned by each model.

We compare iTM-VAE and iTM-VAE-Prod with LDA, DocNADE, NVLDA and ProdLDA.

7 The setup of the retrieval task is as follows.

The documents in the training/validation sets are used as the database for retrieval, and the test set is used as the query set.

Given a query document, documents in the database are ranked according to the cosine similarities to the query.

Precision/Recall (PR) curves can then be computed by comparing the label of the query with those of the database documents.

For documents who have multiple labels (e.g. RCV1-V2), the PR curves for each of its labels are computed individually and then averaged for each query document.

Finally, the global average of these curves is computed to compare the retrieval performance of each model.

FIG2 illustrates the PR curves of different models with hidden representations of length 128.

Specifically, the mean of the variational Gaussian posterior is used as the features for NVLDA and ProdLDA.

Since the effective topic numbers of iTM-VAE-Prod and iTM-VAE are dynamic and usually much smaller than 128 on both datasets, we use a weighted sum of topic factor vectors over the topic distributions, where the factor dimension is 128.

As shown in FIG2 (a) and FIG2 , iTM-VAE-Prod always yields competitive results on both datasets, and outperforms the others in most cases.

We also map the latent representations learned by iTM-VAE-Prod to a 2D space by TSNE BID22 and visualize the representations in FIG2

As mentioned in Section 4.4, the concentration parameter ?? of GEM(??) has significant impact on the growth of the number of topics, which affects the performance of iTM-VAE significantly.

As a result, ?? has to be chosen by cross-validation for better performance.

FIG4 (a) also confirms that the number of effective topics increases with ??.

Consequently, iTM-VAE-G, which places a Gamma(s 1 , s 2 ) prior on ??, is proposed to enable the model to adapt ?? to data automatically.

Another commonly mentioned problem of the optimization under VAE framework is that the latent representation will tend to collapse to the prior BID4 BID36 BID6 ).

In our model, this means the choice of ?? will control the number of the learned topics very tightly when the decoder is strong (e.g. iTM-VAE-Prod with large factor dimension H), which might cause the model to be lack of adaptive power.

Common tricks to alleviate the training problem are annealing the relative weight of the KL divergence term BID36 , or regularizing the decoder (Bowman et al.) .

Rather than regularizing the decoder, iTM-VAE-G can be regarded as a relaxation on the prior placed on the latent space, which is more effective in improving the adaptive power of our model.

To verify this, we compare the adaptive power of iTM-VAE-Prod with KL annealing and decoder regularization, to iTM-VAE-G, on several subsets of 20News dataset, which contain 1, 2, 5, 10 and 20 (the whole dataset) classes, respectively.

For these two models, we use a MLP of two layer of 256 units as the encoder, and the factor dimension of the decoder is H = 200.

For iTM-VAE-Prod, we set ?? = 4, and try different tricks to alleviate the collapse problem: KL annealing where the relative weight of the KL divergence term in the ELBO at epoch n is min(0.005n, 1); Decoder regularization where the weight of the L2 regularization on the decoder is set to 0.1.

For iTM-VAE-G, we add a relatively non-informative prior Gamma(1, 0.25) on ??, and initialize the global variational parameters ?? 1 and ?? 2 of Equation 13 the same as the non-informative prior.

A SGD optimizer with a learning rate of 0.01 is used to optimize ?? 1 and ?? 2 .

No KL annealing and decoder regularization is used for iTM-VAE-G.The number of effective topics learned by iTM-VAE-Prod on subsets of 20News dataset is shown in TAB2 .

We can see that training tricks like KL-annealing and regularizing the decoder do not help much when the decoder is strong.

However, by placing a prior on the concentration parameter ??, iTM-VAE-G can increase the adaptive power of the model.

The corpus-level variational posterior of ?? and the number of effective topics learned by iTM-VAE-G is shown in Table 3 .

As for iTM-VAE-G, before training, E q(??|??1,??2) [??], the expectation of ?? given the variational posterior q(??|?? 1 , ?? 2 ) is 4.

Once the training is done, E q(??|??1,??2) [??] will be adjusted to the training set.

Table 3 illustrates ?? 1 , ?? 2 , E q(??|??1,??2) [??] and the number of effective topics that are learned from data.

We can see that, if the training set contains only 1 class of documents, E q(??|??1,??2) [??] will drop to 2.35, and only 6 effective topics are used to model the dataset.

Whereas, when the training set consists of 10 classes of documents, E q(??|??1,??2) [??] increases to 6.62, and 11 effective topics are discovered by the model to explain the dataset.

This indicates that iTM-VAE-G can learn to adjust ?? to data.

Figure 3(b) illustrates the topic coverage w.r.t the number of topics when the training set contains 1, 2, 5, 10 and 20 classes, respectively.

To this end, we compute the top-1 significant topic for each training document, and sort the topics according to the frequency that it is assigned as top-1.

The topic coverage is then defined as the cumulative sum of these frequencies.

FIG4 (b) shows that, with the increasing of the number of classes, more topics are utilized by iTM-VAE-G to reach the same level of topic coverage, which indicates that the model has the ability to adapt to data.

Table 3 : Learned posterior distribution of ?? and number of effective topics learned by iTM-VAE-G on subsets of 20News dataset.

In this paper, we propose iTM-VAE, which, to our best knowledge, is the first Bayesian nonparametric topic model that is modeled by Variational Auto-Encoders.

Specifically, a stick-breaking prior is used to generate the mixture weights of countably infinite topics and the Kumaraswamy distribution is exploited such that the model can be optimized by AEVB algorithm.

Two variants of iTM-VAE are also proposed in this work.

One is iTM-VAE-Prod, which replaces the mixture of multinomials assumption of iTM-VAE with a product of experts for better performance.

The other one is iTM-VAE-G which places a Gamma prior on the concentration parameter of the stick-breaking process such that the model can adapt the concentration parameter to data automatically.

The advantage of iTM-VAE and its variants over the other Bayesian nonparametric topics models is that the inference is performed by feed-forward neural networks, which is of rich representation capacity and requires only limited knowledge of the data.

Hence, it is flexible to incorporate more information sources to the model, and we leave it to future work.

Experimental results on two public benchmarks show that iTM-VAE and its variants outperform the state-of-the-art baselines significantly.

Table 4 : Top 10 words of topics learned by iTM-VAE-Prod without cherry picking.

As shown in Table 4 , iTM-VAE-Prod can learn topics that are diverse and of high quality.

One possible reason is that the stick-breaking prior for the document-specific ?? encourages the model to learn sparse representation, and the model can adjust the number of topics according to the data.

Thus the topics can be sufficiently trained and of high diversity.

The comparison of representation sparsity is illustrated in FIG5 (a).In contrast, the topics learned by ProdLDA BID37 lack diversity.

As we listed in Table 5 , there are a lot of redundant topics.

As a result, the latent representation learned by ProdLDA is of poor discriminative power.

Figure 4: (a) Representation sparsity of different models on 20News.

We sample one topic assignment ?? for each document, sort and then average across the test set.

9 (b) The TSNE-visualization of the representation learned by by iTM-VAE-Prod.

(c) The TSNE-visualization of the representation learned by ProdLDA BID37 with the best topic coherence on 20News (K = 50).

In this section we show how to compute the Evidence Lower Bound (ELBO) of iTM-VAE which can be written as: DISPLAYFORM0 Topics about Religion 1 jesus christian scripture faith god christ heaven christianity verse resurrection 2 jesus christ doctrine revelation verse scripture satan christian interpretation god 3 belief god passage scripture moral atheist christian truth principle jesus 4 god belief existence faith jesus atheist bible christian religion sin 5 jesus son holy christ father god doctrine heaven spirit prophet 6 homosexual marriage belief islam moral christianity truth islamic religion god Topics about Hardware 7 floppy controller scsus ide scsi ram hd mb cache isa 8 printer meg adapter scsi motherboard windows modem mhz vga hd 9 ide mb connector controller isa scsi scsus floppy jumper disk 10 mb controller bio rom interface mhz scsus scsi floppy ide 11 ide meg motherboard shipping adapter simm hd mhz monitor scsi 12 ram controller dos bio windows disk scsi rom scsus meg 13 honda motherboard bike amp quadra hd brake apple upgrade meg Table 5 : Top 10 words of some redundant topics learned by ProdLDA.??? E q ?? (??|w 1:N ) [log p(w 1:N |??, ??)]: Similar to other VAE-based models, the SGVB estimator and reparameterization trick can be used to approximate this intractable expectation and propagate the gradient flow into the inference network g. Specifically, we have: DISPLAYFORM1 where L is the number of Monte Carlo samples in the SGVB estimator and can be set to 1.

N is the number of words in the document.

According to Section 4.2, ?? (l) can be obtained by DISPLAYFORM2 where g(w 1:N ; ??) is an inference network with parameters ??, ?? denotes the Kumaraswamy distribution and K ??? R + is the truncation level.

Here we omit the superscript * (l) for simplicity.

According to the generative procedure in Section 4.1, p(w i |?? (l) , ??) can be computed by DISPLAYFORM3 where t k ??? R H is the k-th topic factor vector, W ??? R H??V is the word factor matrix, H is the factor dimension, V is the vocabulary size and ??(??) is the softmax function.??? KL (q ?? (??|w 1:N )||p(??|??)): By applying the KL divergence of a Kumaraswamy distribution ??(??; a k , b k ) from a beta distribution p(??; 1, ??), we have:KL (q ?? (??|w 1:N )||p(??|??)) = K???1 k=1 KL (q ?? (?? k |w 1:N )||p(?? k |??)) DISPLAYFORM4 where B(??) is the Beta function and ?? is the Euler's constant.

In this section we show how to compute the Evidence Lower Bound (ELBO) of iTM-VAE-G which can be written as: L(w 1:N |??, ??) = E q ?? (??|w 1:N ) [log p(w 1:N |??, ??)] + E q ?? (??|w 1:N )q(??|??1,??2) [log p(??|??)]???E q ?? (??|w 1:N ) [log q ?? (??|w 1:N )] ??? KL(q(??|?? 1 , ?? 2 )||p(??|s 1 , s 2 )) (21) Specifically, each item in Equation 21 can be obtained as follows:??? E q ?? (??|w 1:N ) [log p(w 1:N |??, ??)]:The derivation is exactly the same as Appendix 7.2.??? E q ?? (??|w 1:N )q(??|??1,??2) [log p(??|??)]:Recall that the prior of the stick length variable ?? k is Beta(1,??): p(v k |??) = ??(1 ??? ?? k ) ?????1 and the variational posterior of the concentration parameter ?? is a gamma distribution q(??; ?? 1 , ?? 2 ), we have E q ?? (??|w 1:N )q(??|??1,??2) [log p(??|??)] = E q ?? (??|w 1:N ) K???1 k=1 E q(??|??1,??2) [log ?? + (?? ??? 1) log(1 ??? ?? k )]

DISPLAYFORM0 Now, we provide more details about the calculation of these two expectations in Equation 22 as follows:??? E q(??|??1,??2) [log ??]:

Fisrt, we can write the gamma distribution q(??; ?? 1 , ?? 2 ) in its exponential family form:q(??; ?? 1 , ?? 2 ) = 1 ?? exp ??? ?? 2 ?? + ?? 1 log ?? ??? (log ??(?? 1 ) ??? ?? 1 log ?? 2 )Considering the general fact that the derivative of the log normalizor log ??(?? 1 ) ??? ?? 1 log ?? 2 of a exponential family distribution with respect to its natural parameter ?? 1 is equal to the expectation of the sufficient statistic log ??, we can compute E q(??|??1,??2) [log ??] in the first term of Equation 22 as follows:E q(??|??1,??2) [log ??] = ??(?? 1 ) ??? log ?? 2 (24) where ?? is the digamma function, the first derivative of the log Gamma function.??? E q ?? (?? k |w 1:N ) [log(1 ??? v k )]:By applying the Taylor expansion, E q(?? k |w 1:N ) [log(1 ??? v k )] can be written as the infinite sum of the Kumaraswamy's mth moment: DISPLAYFORM1 where B(??) is the Beta function.

E q ?? (??|w 1:N )q(??|??1,??2) [log p(??|??)]

DISPLAYFORM0 ??? ???E q ?? (??|w 1:N ) [log q ?? (??|w 1:N )]: According to Section 4.11 of BID25 , the Kumaraswamy's entropy is given as ???E q ?? (??|w 1:N ) [log q ?? (??|w 1: DISPLAYFORM1 where ?? is the Euler's constant.??? KL(q(??|?? 1 , ?? 2 )||p(??|s 1 , s 2 )):The KL divergence of one gamma distribution q(??; ?? 1 , ?? 2 ) from another gamma distribution p(??; s 1 , s 2 ) evaluates to DISPLAYFORM2

<|TLDR|>

@highlight

A Bayesian Nonparametric Topic Model with Variational Auto-Encoders which achieves the state-of-the-arts on public benchmarks in terms of perplexity, topic coherence and retrieval tasks.

@highlight

This paper constructs an infinite Topic Model with Variational Auto-Encoders by combining Nalisnick & Smith's stick-breaking variational auto-encoder with latent Dirichlet allocation and several inference techniques used in Miao.