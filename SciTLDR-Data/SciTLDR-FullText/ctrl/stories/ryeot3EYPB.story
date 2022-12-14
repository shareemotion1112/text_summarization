The scarcity of labeled training data often prohibits the internationalization of NLP models to multiple languages.

Cross-lingual understanding has made progress in this area using language universal representations.

However, most current approaches focus on the problem as one of aligning language and do not address the natural domain drift across languages and cultures.

In this paper, We address the domain gap in the setting of semi-supervised cross-lingual document classification, where labeled data is available in a source language and only unlabeled data is available in the target language.

We combine a state-of-the-art unsupervised learning method, masked language modeling pre-training, with a recent method for semi-supervised learning, Unsupervised Data Augmentation (UDA), to simultaneously close the language and the domain gap.

We show that addressing the domain gap in cross-lingual tasks is crucial.

We improve over strong baselines and achieve a new state-of-the-art for cross-lingual document classification.

Recent advances in Natural Language Processing have enabled us to train high-accuracy systems for many language tasks.

However, training an accurate system still requires a large amount of training data.

It is inefficient to collect data for a new task and it is virtually impossible to annotate a separate data set for each language.

To go beyond English and a few popular languages, we need methods that can learn from data in one language and apply it to others.

Cross-Lingual Understanding (XLU) has emerged as a field concerned with learning models on data in one language and applying it to others.

Much of the work in XLU focuses on the zero-shot setting, which assumes that labeled data is available in one source language (usually English) and not in any of the target languages in which the model is evaluated.

The labeled data can be used to train a high quality model in the source language.

One then relies on general domain parallel corpora and monolingual corpora to learn to 'transfer' from the source language to the target language.

Transfer methods can explicitly rely on machine translation models built from such parallel corpora.

Alternatively, one can use such corpora to learn language universal representations to produce features to train a model in one language, which one can directly apply to other languages.

Such representations can be in the form of cross-lingual word embeddings, contextual word embeddings, or sentence embeddings (Ruder et al. (2017) ; Lample & Conneau (2019) ; Schwenk & Douze (2017) ).

Using such techniques, recent work has demonstrated reasonable zero-shot performance for crosslingual document classification (Schwenk & Li (2018) ) and natural language inference (Conneau et al. (2018) ).

What we have so far described is a simplified view of XLU, which focuses solely on the problem of aligning languages.

This view assumes that, if we had access to a perfect translation system, and translated our source training data into the target language, the resulting model would perform as well as if we had collected a similarly sized labeled dataset directly in our target language.

Existing work in XLU to date also works under this assumption.

However, in real world applications, we must also bridge the domain gap across different languages, as well as the language gap.

No task is ever identical in two languages, even if we group them under the same label, e.g. 'news document classification' or 'product reviews sentiment analysis'.

A Chinese customer might express sentiment differently than his American counterpart.

Or French news might simply cover different topics than English news.

As a result, any approach which ignores this domain drift will fall short of native in-language performance in real world XLU.

In this paper, we propose to jointly tackle both language and domain transfer.

We consider the semi-supervised XLU task, where in addition to labeled data in a source language, we have access to unlabeled data in the target language.

Using this unlabeled data, we combine the aforementioned cross-lingual methods with recently proposed unsupervised domain adaptation and weak supervision techniques on the task of cross-lingual document classification (XLDC).

In particular, we focus on two approaches for domain adaptation.

The first method is based on masked language model (MLM) pre-training (as in Devlin et al. (2018) ) using unlabeled target language corpora.

Such methods have been shown to improve over general purpose pre-trained models such as BERT in the weakly supervised setting (Lee et al. (2019) ; Han & Eisenstein (2019) ).

The second method is unsupervised data augmentation (UDA) (Xie et al. (2019) ), where synthetic paraphrases are generated from the unlabeled corpus, and the model is trained on a label consistency loss.

While both of these techniques were proposed previously, in both cases there are some open questions when applying them on the cross-lingual problems.

For instance when performing data augmentation, one could generate augmented paraphrases in either the source or the target language or both.

We experiment with various approaches and provide guidelines with ablation studies.

Furthermore, we find that the value of additional labeled data in the source language is limited due to the train-test discrepancy of XLDC tasks.

We propose to alleviate this issue by using self-training technique to do the domain adaptation from the source language into the target language.

By combining these methods, we are able to reduce error rates by an average 44% over a strong XLM baseline, setting a new state-of-the-art for cross-lingual document classification.

Cross-lingual document classification was first introduced in (Bel et al., 2003) .

The subsequent work (Prettenhofer & Stein, 2010) proposes the cross-lingual sentiment classification datasets, and (Lewis et al., 2004; Klementiev et al., 2012; Schwenk & Li, 2018) have extended this to the news domain.

Cross-lingual understanding has also been applied to other NLP tasks, with datasets available in dependency parsing (Nivre et al., 2016) , natural language inference (XNLI) (Conneau et al., 2018) and question answering ).

Cross-lingual methods gained popularity with the advent of cross-lingual word embeddings (Mikolov et al. (2013) ).

Since then, many methods have been proposed to better align the word embedding spaces of different languages (see Ruder et al. (2017) for a survey).

Recently, more sophisticated extensions have been proposed based on seq2seq training of cross-lingual sentence embeddings (Schwenk & Douze (2017) ; Artetxe & Schwenk (2018) ) and contextual word embeddings pre-trained on masked language modeling, notably multilingual BERT (Devlin et al. (2018) ) and the cross-lingual language model (XLM) of Lample & Conneau (2019) .

We use XLM as our baseline representation in all experiments, as it's the current state-of-the-art on the commonly used XNLI benchmark for cross-lingual understanding.

Domain adaptation, closely related to transfer learning, has a rich history in machine learning and natural language processing (Pan & Yang (2009) ).

Such methods have long been applied to document classification tasks (Blitzer et al., 2007; Glorot et al., 2011; Al-Moslmi et al., 2017; Xu & Yang, 2017) .

Domain adaptation for NLP is intimately related to transfer learning and semi-supervised learning (Chapelle et al. (2009) ).

Transfer learning has made tremendous advances recently due to the success of pre-training representations using language modeling as a source task (Radford et al. (2018) ; Peters et al. (2018) ; Devlin et al. (2018) ).

While such representations trained on large amounts of general domain text have been shown to transfer well generally, performance still suffers when the target domain is sufficiently different than what the models were pre-trained on.

In such cases, it is known that further pre-training the language model on in-domain text is helpful (Howard & Ruder (2018) ; Chronopoulou et al. (2019) ).

It is natural to use unsupervised domain data for this task, when available (Lee et al. (2019) ; Han & Eisenstein (2019) ).

The study of weakly supervised learning in language processing is relatively new (Johnson & Zhang, 2016; Yu et al., 2018) .

Most recently, Xie et al. (2019) has introduced an unsupervised data augmentation (UDA) technique to demonstrate improvements in the few-shot learning setting.

Here, we extend this technique to facilitate cross-lingual and cross-domain transfer.

In this section, we formally define the problem discussed in this paper and describe the proposed approach in detail.

In vanilla zero-shot cross-lingual document classification, it is assumed that we have available a labeled dataset in a source language (English in our case), which we can denote by L src = {(x, y)|x ??? P src (x)}, where P src (x) is the prior distribution of task data in the source language.

It is assumed that no data is available for the task in the target language.

General purpose parallel and monolingual resources are used to train a cross-lingual classifier on the labeled source data, which is then applied to the target language data at test time.

In this work, we also assume access to a large unlabeled corpus in the target language, U tgt = {x|x ??? P tgt (x)}, which is usually the case in practical applications.

We aim to utilize this domain-specific unlabeled data to help model be adapted to the target domain and gain the better generalization ability.

We refer to this setting as semi-supervised XLDC, although we're still in the zero-shot setting, in that no labeled data is used in the target language.

There are two standard ways to transfer knowledge across the languages in the vanilla zero-shot setting: (1) using a translation system to translate the labeled samples, such as translate-train and translate-test methods, and (2) learning a multilingual embedding system to obtain a language irrelevant representations of the data.

In this paper, we adopt the second approach as the basic model, and utilize the XLM model (Lample & Conneau, 2019 ) as our base model, which has been pre-trained by large-scale parallel and monolingual data from various languages.

Because XLM is a multilingual embedding system, a baseline is obtained by fine-tuning XLM with the labeled set L src (x) and directly applying the resulting model to the target language.

In the experiments section, we also discuss the combination of the XLM and the translation based approaches.

As argued in Introduction, even with a perfect translation or multilingual embedding system, we still face the domain-mismatch problem.

This mismatch may limit the generalization ability of the model during testing time.

To fully adapt the classifier to the target distribution, we explore the following approaches, each of which leverages unlabeled data in the target language in different ways.

Masked Language Model pre-training BERT (Devlin et al., 2018) and its derivations (such as XLM) are trained on general domain corpora.

A standard practice is to to further pre-train to adapt to a particular domain when data is available.

This technique can let model learn the prior knowledge from the target domain and lead to better performance.

We refer to this approach as the MLM pre-training.

However, in the cross-lingual setting, fine-tuning the XLM model in the target language can make the model degenerate in the source language, decreasing its ability to transfer across languages.

This problem prohibits the fine-tuning baseline to learn a reasonable model.

Therefore, in this case, we take care to use this method in combination with the translate-train method, which translates all labeled samples into the target language.

Unsupervised Data Augmentation The second approach is utilizing the state-of-the-art semisupervised learning technique, Unsupervised Data Augmentation (UDA) algorithm (Xie et al., 2019) , to leverage the unlabeled data.

The objective function of UDA can be written as,

wherex is an augmented sample generated by a predefined augmentation functionx = q(x).

The augmentation function can be a paraphrase generation model, or a noising heuristic.

Here, we use a machine translation system as the praraphrase generation model.

The UDA loss enforces the classifier to produce label consistent predictions for pairs of original and augmented samples.

With UDA method, the model is better adapted to the target domain by learning a label consistent model over the target domain.

In the cross-lingual setting, there are multiple ways of generating augmented samples using translation.

One could translate samples from the target language into the source language and use this crosslingual pair as the augmented sample.

Alternatively, one could translate back into the target language and use only target-language augmented samples.

We find that the latter works best.

It is also possible to do data augmentation using source domain unlabeled data.

The results of these comparisons are included in out detailed ablation study in the experiments section.

Alleviating the Train-Test Discrepancy of the UDA Method With the UDA algorithm, the classifier is able to explore the prior information on the target domain, however it still suffers from the train-test discrepancy.

During the testing phase, our goal is to maximize the classifier performance on the real samples in target language, which focus on the samples from the distribution P tgt (x).

Upon observing the training objective of the UDA method, Eq. (1), one can see that the data x that feed to model in the training phrase is sampled from three domains: (1) the source domain P src (x), (2) the target domain P tgt (x) and (3) the augmented sample domain P aug (x).

On the other hand, the testing phrase only processes data from the target domain P tgt (x).

The source and target domain are mismatched, due to differences in language as argued earlier.

Furthermore, the augmented domain, although generated from the target domain, can also be mismatched, due to artifacts introduced by the translation system.

This can be especially problematic, since the UDA method needs diversity in the augmented samples to perform well (Xie et al., 2019) , which trades off against their quality.

We propose to apply the self-training technique (Lee, 2013) to tackle this problem.

We first train a classifier based on the UDA algorithm and denote it as f * (x), which is the teacher model used to score the unlabeled data U tgt from the target domain.

Then we fine-tune a new XLM model using the soft classification loss function with the pseudo-labeled data, which is written as,

Follwing this process, we obtain a new classifier trained only based on the target domain, which does not suffer from the train-test mismatch problem.

We show that this process provides better generalization ability compared to the teacher model.

A process diagram of this method is presented in figure 1 .

In this section, we present a comprehensive study on two benchmark tasks, cross-lingual sentiment classification, and cross-lingual news classification.

In this task, we test the proposed framework on a sentiment classification benchmark in three target languages, i.e. French, German and Chinese.

The French, German and English data come from the benchmark cross-lingual Amazon reviews dataset (Prettenhofer & Stein, 2010) , which we denote as amazon-fr, amazon-de and amazon-en.

We merge training and testing samples from all product categories in one language, which leads to 6000 training samples.

However, for the purpose of facilitating fair comparison with previous work, we also provide results for category-wise settings to compare previous state-of-the-art performance.

The number of unlabeled samples from amazon-fr is 54K, and from amazon-de is 310K.

For Chinese, we use the Chinese Amazon (amazon-cn) (Zhang et al., 2015b) and Dianping (Zhang et al., 2014) datasets.

Dianping is a business review website similar to Yelp.

The training data for amazon-cn is amazon-en, and for dianping it is the Yelp dataset (Zhang et al., 2015a) .

In these two cases, the size of the training sample is 2000.

For both amazon-cn and dianping datasets, we have 4M unlabeled examples.

Because the number of the unlabeled set is very large, we randomly sample 10% for the UDA algorithm.

News Classification We use the MLDoc dataset (Schwenk & Li, 2018) for this task.

The MLdoc dataset is a subset of RCV2 multilingual news dataset (Lewis et al., 2004) .

It has 4 categories, i.e. Corporate/Industrial, Economics, Government/Social and Markets, and each category has 250 training samples.

We use the rest of the news documents in RCV2 dataset as the unlabeled data.

The number of unlabeled samples for each language ranges from 5K to 20K, which is relatively smaller compared to the sentiment classification task.

Because the XLM model is pre-trained on 15 languages, we ignore languages which are not supported by XLM in the above benchmark datasets.

The pre-processing scripts for the above datasets, augmented samples, pretrained models and experiment settings needed to reproduce results will be released in the Github repo.

As introduced in section 3.3, we apply MLM pre-training on the unlabeled data corpus to obtain a domain-specific XLM, denoted as XLM f t in the following sections.

The pre-training strategies for the two tasks are slightly different.

In the sentiment classification task, because the size of the unlabeled corpus in each target domain is large enough, we fine-tune an XLM with MLM loss for each target domain respectively.

In contrast, we do not have enough unlabeled data in each language in the MLDoc dataset, therefore we integrate unlabeled data from all languages as the training corpus.

As a result the XLM f t still preserves its language universality in this task.

We compare the follwing models:

??? Fine-tune (Ft): Fine-tuning the pre-trained model with the source-domain training set.

In the case of XLM f t , the training set is translated into the target language.

??? Fine-tune with UDA (UDA): This method utilizes the unlabeled data from the target domain by optimizing the UDA loss function (Eq. (1)).

??? Self-training based on the UDA model (UDA+Self): We first train the Ft model and UDA model, and choose the one with better develop accuracy as the teacher model.

The teacher model is used to train a new XLM student using only unlabeled data U tgt in the target domain, as described above.

We report the results of applying these three methods on both the original XLM model and the XLM f t model.

In order to keep the notation simple, we use parenthesis after the method name to indicate which basic model was used, such as UDA(XLM f t ).

The details about the implementation and hyper-parameter tuning are included in Appendix A.1.

The results for the cross-lingual sentiment classification task are summarized in table 1.

As our experiment setting on the cross-lingual amazon dataset is different from previous publications, in order to provide a fair comparison with previous works, we summarize the results of the standard category-wise setting in table 3.

The results for cross-lingual news classification is included in table 2.

The last column "Unlabeled" in these tables indicates whether this method utilizes the unlabeled data.

For the monolingual baselines, the models are trained with labeled data from the target domain.

The size of the labeled set is the same as the English training set used for cross-lingual experiments.

We can summarize our findings as follows:

??? Looking at Ft(XLM) results, it is clear that without the help of unlabeled data from the target domain, there still exists a substantial gap between the model performance of the cross-lingual settings and the monolingual baselines, even when using state-of-the-art pre-trained cross-lingual representations.

??? Both the UDA algorithm and MLM pre-training can offer significant improvements by utilizing the unlabeled data.

In the sentiment classification task, where the unlabeled data size is larger, Ft(XLM f t ) model usnig MLM pre-training consistently provides larger improvements compared with the UDA method.

On the other hand, the MLM method is relatively more resource intensive and takes longer to converge (see Appendix A.5).

In contrast, in the MLdoc dataset, when the size of the unlabeled samples is limited, the UDA method is more helpful.

??? The combination of both methods -as in the UDA(XLM f t ) model -consistently outperforms either method alone.

In this case the additional improvement provided by the UDA algorithm is smaller, but still consistent.

??? In the sentiment classification task, we observe the self-training technique consistently improves over its teacher model.

It offers best results in both XLM and XLM f t based classifiers.

The results demonstrate that self-training process is able to alleviate the train-test distribution mismatch problem and provide better generalization ability.

In the MLdoc dataset, self-training also achieves the best results overall, however the gains are less clear.

We hypothesize that this technique is not as useful without enough number of unlabeled samples.

??? Finally, comparing with the best cross-lingual results and monolingual fine-tune baseline, we are able to completely close the performance gap by utilizing unlabeled data in the target language.

Furthermore, our framework reaches new state-of-the-art results, improving over vanilla XLM baselines by 44% on average.

Furthermore, we provide an additional baseline, which only uses English samples to perform semisupervised learning, whose details are in Appendix A.2.

The experment results show that it lags behind the ones using unlabeled data from the target domain.

This observation also justifies the importance of information from the target domain in the XLDC task.

Table 3 : Error rates for the sentiment classification task by product category.

The pre-XLM sota results are provided by Chen et al. (2019) .

In this section, we provide evidence for the train-test domain discrepancy in the context of the UDA method, by showing that adding more labeled data in the source language does not improve target task accuracy after a certain point.

Figure 2 plots the target model performance vs. the number of labeled training samples in the cross-lingual and monolingual settings respectively.

The figures are based on the UDA(XLM) method with 6 runs in the Yelp-Dianping cross-lingual setting.

The dot is the average accuracy and the filling area contains one standard derivation.

We observe that, in the cross-lingual setting, the model performance peaks at around 10k training samples per category, and becomes worse with the larger training set.

In contrast, the performance of the model improves consistently with more labeled data in the monolingual setting.

This suggests that more training data from the source domain could harm model generalization ability in the target domain with UDA approach in the cross-lingual setting.

In order to alleviate this issue, we propose to utilize the self-training technique, which abandons the data from the source domain and the augmentation domain, to maximize its performance in the target domain.

Next, we explore different augmentation strategies and their influence on the final performance.

As stated in section 3.3, the augmentation strategy used in the main experiment is that we first translate the samples into English and translate them back to its original language.

We refer to this strategy as augmenting "from target domain to target domain" and abbreviate it as t2t.

We also explore two additional augmentation strategies: (1) First, we do not translate the samples back to the target language and directly use English samples as the augmented samples, denoted as t2s.

Naturally, the parallel samples in two languages have the same sentiment information and different input format which are suitable to be used as the augmentation sample pairs for the multilingual system such as XLM.

(2) The second approach is to leverage unlabeled data from other language domains.

Here, we attempt to use the English unlabeled data.

We translate them into the target language as the augmented samples.

This strategy is denoted as s2t.

Table 4 : Error rates when using different augmentation strategies and their combinations.

Results for sentiment classification shown on the left, and news document classification on the right.

Table 4 summarizes the performance of the proposed augmentation strategies and their combinations with the UDA(XLM) method in the sentiment classification and the UDA(XLM f t ) in the news classification settings.

From the results, we conclude that t2t is the best performing approach, as it's the best matched to the target domain.

Leveraging the unlabeled data from other domains does not offer consistent improvement, however can provide additional value in isolated cases.

We include additional ablations regarding translation system in the appendix, including the application of translate-train method in our experiments (section A.3) and effects of hyper-parameters (section A.4), and finally we discuss the application of our framework in the monolingual cross-domain document classification problem setting in Appendix B.

In this paper, we tackled the domain mismatch challenge in cross-lingual document classification -an important, yet often overlooked problem in cross-lingual understanding.

We provided evidence for the existence and importance of this problem, even when utilizing strong pre-trained cross-lingual representations.

We proposed a framework combining cross-lingual transfer techniques with three domain adaptation methods; unsupervised data augmentation, masked language model pre-training and self-training, which can leverage unlabeled data in the target language to moderate the domain gap.

Our results show that by removing the domain discrepancy, we can close the performance gap between crosslingual transfer and monolingual baselines almost completely for the document classification task.

We are also able to improve the state-of-the-art in this area by a large margin.

While document classification is by no means the most challenging task for XLU, we believe the strong gains that we demonstrated could be extended to other cross-lingual tasks, such as cross-lingual question answering and event detection.

Developing cross-lingual methods which are competitive with in-language models for real world, semantically challenging NLP problems remains an open problem and subject of future research.

The experiments in this paper are based on the PyTorch (Paszke et al., 2017) and Pytext (Aly et al., 2018) package.

We use the Adam (Kingma & Ba, 2014) as the optimizer.

For all experiments, we grid search the learning rate in the set {5 ?? 10 ???6 , 1 ?? 10 ???5 , 2 ?? 10 ???5 }.

When using UDA method, we also try the three different annealing strategies introduced in the UDA paper (Xie et al., 2019) , and the ?? in (1) is always set as 1.

The batch size in the Ft and UDA+Self method is 128.

In the UDA method, the batch size is 16 for the labeled data and 80 for the unlabeled data.

Due to the limitation of the GPU memory, in all experiments, we set the length of samples as 256, and cut the input tokens exceeding this length.

Finally, we report the results with the best hyper-parameters.

As for the augmentation process, we sweep the temperature which controls the diversity of beam search in translation.

The best temperature for "en-de, en-fr, en-es" and "en-ru" are 1.0 and 0.6, the sampling space is the whole vocabulary.

In the "en-zh" setting, the temperature is 1.0 and the sampling space is the top 100 tokens in the vocabulary.

We note that this uses the Facebook production translation models, and results could vary when other translation systems are applied.

For reproducibility, we will release the augmented datasets that we generated.

Here, we provide a baseline only using English samples to perform semi-supervised learning.

More specifically, we first train the model with English unlabeled data and augmented samples, then tests it on different target domains.

This approach is similar to the traditional translate-test method.

This method offers a baseline, which merely increases the size of data but without providing the target domain information.

During the test phrasing, we experiment with two input strategies.

One is using the original test samples, and another is translating the samples into English.

We report the results (Table 5 ) of the UDA(XLM) method with two input strategies and compare them with the main results, which uses the unlabeled data from the target domain.

First, we observe that the performance of using original and translated samples is similar.

Second, compared with Ft(XLM) baselines in section 4.3, utilizing the unlabeled data from the English domain is slightly better than only training with labeled data, but it still lags behind the performance of using the unlabeled data from the target domain.

Table 5 : The first part is the baseline results using the English unlabeled data.

The second part is the results using the unlabeled data from the target domain, which are copied from the section 4.3.

A.3 ABLATION STUDY: TRANSLATE-TRAIN As discussed earlier, fine-tuning XLM on the target language would depreciate the multilingual ability of the model.

We apply the translate-train method to tackle this problem.

In order to understand the influence of this strategy when using the proposed framework, we perform an ablation study.

We test 3 input strategies: (1) English: use the original English data as training data.

(2) tr-train: use the translate-train strategy, which translate the training data into the target language.

(3) both: we combine the (1) and (2) as the training data.

We report the results of the UDA(XLM) method in the sentiment classification tasks and UDA(XLM f t ) method in the news classification tasks in Table 6 : Ablation Study about the translate-train strategies.

Results for sentiment classification shown on the left, and news document classification on the right.

We observe that in most cases, using the original training examples achieves the best performance.

However, in special cases such as MLDoc-ru, the translate-train method achieves better performance.

We recommend trying both approaches in practice.

Given a translation system, we use the sample decoding strategy to translate the sample.

The sample space is the entire vocabulary space.

We tune the temperature of ?? of the softmax distribution.

As discussed in Xie et al. (2019) , this controls the trade-off between quality and diversity.

When ?? = 0, the sampling reduces to the greedy search and produce the best quality samples.

When ?? = 1, the sampling produces diverse outputs but loses some semantic information.

The table A.4 illustrates the influence of ?? value to the final performance in the English-to-French and English-to-German settings.

The results show that temperature has a significant influence on the final performance.

However, because the quality of translation systems for different language pairs are not the same, their best temperature also varies.

In the appendix A.1, we include the best temperature values for the translation systems used in this paper.

Table 7 : Effect of the temperature of the translation sampling decoder.

A.5 COMPUTATION TIME OF UDA AND MLM PRETRAINING From the main results in section 4.3, we can see that MLM pre-training can offer better improvements over the UDA method.

However, it is also more resource intensive, since MLM pre-training is a token level task with a large output space, which leads to more computationally intensive updates and also takes longer to converge.

In our experiments, we used NVIDIA V100-16G GPUs to train all models.

8 GPUs were used to train Ft and UDA methods, and 32 GPUs to perform MLM pretraining.

In the "amazonen->amazonfr" setting, for example, the unlabeled set contains 50K unlabeled samples and 8M tokens after BPE tokenization.

The Ft method takes 3.2 GPU hours to converge.

The UDA method training takes 16.8 GPU hours, excluding the time it takes to generate augmented samples (which we handle as part of data pre-processing).

MLM pre-training takes upwards of 500 GPU hours to converge.

This is another factor which should be taken into account.

As further evidence that our method addresses the domain mismatch, we apply out framework to the monolingual cross-domain document classification problem.

We again focus on sentiment classification where data comes from two different domains, product reviews (amazon-en, amazon-cn) and business reviews (Yelp and Dianping).

We train and test on the same language, only transferring across domains.

We consider the two domain-pairs, amazonen-yelp and amazoncn-dianping.

The results are illustrated in table 8.

Conclusions are similar to the cross-domain setting (section 4.3):

??? There exists a clear gap between the cross-domain and in-domain results of the Ft method, even when using strong pre-trained representations.

??? By leveraging the unlabeled data from the target domain, we can significantly boost the model performance.

??? Best results are achieved with our combined approach and almost completely matches the indomain baselines.

<|TLDR|>

@highlight

Semi-supervised Cross-lingual Document Classification