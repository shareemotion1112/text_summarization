Learning multilingual representations of text has proven a successful method for many cross-lingual transfer learning tasks.

There are two main paradigms for learning such representations: (1) alignment, which maps different independently trained monolingual representations into a shared space, and (2) joint training, which directly learns unified multilingual representations using monolingual and cross-lingual objectives jointly.

In this paper, we first conduct direct comparisons of representations learned using both of these methods across diverse cross-lingual tasks.

Our empirical results reveal a set of pros and cons for both methods, and show that the relative performance of alignment versus joint training is task-dependent.

Stemming from this analysis, we propose a simple and novel framework that combines these two previously mutually-exclusive approaches.

Extensive experiments on various tasks demonstrate that our proposed framework alleviates limitations of both approaches, and outperforms existing methods on the MUSE bilingual lexicon induction (BLI) benchmark.

We further show that our proposed framework can generalize to contextualized representations and achieves state-of-the-art results on the CoNLL cross-lingual NER benchmark.

Continuous word representations (Mikolov et al., 2013a; Pennington et al., 2014; Bojanowski et al., 2017) have become ubiquitous across a wide range of NLP tasks.

In particular, methods for crosslingual word embeddings (CLWE) have proven a powerful tool for cross-lingual transfer for downstream tasks, such as text classification (Klementiev et al., 2012a) , dependency parsing (Ahmad et al., 2019) , named entity recognition (NER) (Xie et al., 2018; Chen et al., 2019) , natural language inference , language modeling (Adams et al., 2017) , and machine translation (MT) (Zou et al., 2013; Artetxe et al., 2018b; .

The goal of these CLWE methods is to learn embeddings in a shared vector space for two or more languages.

There are two main paradigms for learning CLWE: cross-lingual alignment and joint training.

The most successful approach has been the cross-lingual embedding alignment method (Mikolov et al., 2013b) , which relies on the assumption that monolingually-trained continuous word embedding spaces share similar structure across different languages.

The underlying idea is to first independently train embeddings in different languages using monolingual corpora alone, and then learn a mapping to align them to a shared vector space.

Such a mapping can be trained in a supervised fashion using parallel resources such as bilingual lexicons (Xing et al., 2015; Smith et al., 2017; Joulin et al., 2018b; Jawanpuria et al., 2019) , or even in an unsupervised 2 manner based on distribution matching (Zhang et al., 2017a; Artetxe et al., 2018a; Zhou et al., 2019) .

Recently, it has been shown that alignment methods can also be effectively applied to contextualized word representations (Schuster et al., 2019; Aldarmaki & Diab, 2019) .

Another successful line of research for CLWE considers joint training methods, which optimize a monolingual objective predicting the context of a word in a monolingual corpus along with either a 1 Code will be released on publication.

2 In this paper, "supervision" refers to that provided by a parallel corpus or bilingual dictionaries.

hard or soft cross-lingual constraint.

Similar to alignment methods, some early works rely on bilingual dictionaries (Ammar et al., 2016; Duong et al., 2016) or parallel corpora (Luong et al., 2015; for direct supervision.

More recently, a seemingly naive unsupervised joint training approach has received growing attention due to its simplicity and effectiveness.

In particular, reports that simply training embeddings on concatenated monolingual corpora of two related languages using a shared vocabulary without any cross-lingual resources is able to produce higher accuracy than the more sophisticated alignment methods on unsupervised MT tasks.

Besides, for contextualized representations, unsupervised multilingual language model pretraining using a shared vocabulary has produced state-of-the-art results on multiple benchmarks 3 (Devlin et al., 2019; Artetxe & Schwenk, 2019; Lample & Conneau, 2019) .

Despite a large amount of research on both alignment and joint training, previous work has neither performed a systematic comparison between the two, analyzed their pros and cons, nor elucidated when we may prefer one method over the other.

Particularly, it's natural to ask: (1) Does the phenomenon reported in extend to other cross-lingual tasks?

(2) Can we employ alignment methods to further improve their proposed unsupervised joint training?

(3) If so, how would such a framework compare to supervised joint training methods that exploit equivalent resources?

(4) And lastly, can this framework generalize to contextualized representations?

In this work, we attempt to address these questions.

Specifically, we first evaluate and compare alignment versus joint training methods across three diverse tasks: BLI, cross-lingual NER, and unsupervised MT.

We seek to characterize the conditions under which one approach outperforms the other, and glean insight on the reasons behind these differences.

Based on our analysis, we further propose a simple, novel, and highly generic framework that uses unsupervised joint training as initialization and alignment as refinement to combine both paradigms.

Our experiments demonstrate that our framework improves over both alignment and joint training baselines, and outperforms existing methods on the MUSE BLI benchmark.

Moreover, we show that our framework can generalize to contextualized representations, producing state-of-the-art results on the CoNLL cross-lingual NER benchmark.

To the best of our knowledge, this is the first framework that combines previously mutually-exclusive alignment and joint training methods.

Notation.

We assume we have two different languages {L 1 , L 2 } and access to their corresponding training corpora.

We use

j=1 to denote the vocabulary set of the ith language where each w j Li represents a unique token, such as a word or subword.

The goal is to learn a set of em-

, with x j ∈ R d , in a shared vector space, where each token w j Li is mapped to a vector in E. Ideally, these vectorial representations should have similar values for tokens with similar meanings or syntactic properties, so they can better facilitate cross-lingual transfer.

Given the notation, alignment methods consist of the following steps:

Step 1: Train an embedding set E 0 = E L1 ∪ E L2 , where each subset

j=1 is trained independently using the ith language corpus and contains an embedding x j Li for each token w j Li .

Step 2:

Obtain a seed dictionary

, either provided or learnt unsupervised.

Step 3: Learn a projection matrix W ∈ R d×d based on D, resulting in a final embedding set

To find the optimal projection matrix W , Mikolov et al. (2013b) proposed to solve the following optimization problem: min

where X L1 and X L2 are matrices of size d × K containing embeddings of the words in D. Xing et al. (2015) later showed further improvement could be achieved by restricting W to an orthogonal matrix, which turns the Eq.(1) into the Procrustes problem with the following closed form solution:

where W * denotes the optimal solution and SVD(·) stands for the singular value decomposition.

As surveyed in Section 5, different methods (Smith et al., 2017; Joulin et al., 2018b; Artetxe et al., 2018a) differ in the way how they obtain the dictionary D and how they solve for W in step 3.

However, most of them still involve solving the Eq.(2) as a crucial step.

Joint training methods in general have the following objective:

where L 1 and L 2 are monolingual objectives and R(L 1 , L 2 ) is a cross-lingual regularization term.

For example, Klementiev et al. (2012b) uses language modeling objectives for L 1 and L 2 .

The term R(L 1 , L 2 ) encourages alignment of representations of words that are translations.

Training an embedding set E J = E L1 ∪ E L2 is usually done by directly optimizing L J .

While supervised joint training requires access to parallel resources, recent studies Devlin et al., 2019; Artetxe & Schwenk, 2019; Lample & Conneau, 2019) have suggested that unsupervised joint training without such resources are also effective.

Specifically, they show that the cross-lingual regularization term R(L 1 , L 2 ) does not require direct cross-lingual supervision to achieve highly competitive results.

This is because the shared words between L 1 and L 2 can serve implicitly as translations by sharing their embeddings to ensure that representations of different languages lie in a shared space.

Using our notation, the unsupervised joint training approach takes the following steps:

1.

While recent studies in unsupervised joint training have suggested the potential benefits of word sharing, alignment methods rely on two disjoint sets of embeddings.

Along with some possible loss of information due to no sharing, one consequence is that finetuning the aligned embeddings on downstream tasks may be sub-optimal due to the lack of crosslingual constraints at the finetuning stage, whereas shared words can fulfill this role in jointly trained models.

2.

A key assumption of alignment methods is the isomorphism of monolingual embedding spaces.

However, some recent papers have challenged this assumption, showing that it does not hold for many language pairs (Søgaard et al., 2018; Patra et al., 2019) .

Also notably, Ormazabal et al. (2019) suggests that this limitation results from the fact that the two sets of monolingual embeddings are independently trained.

On the other hand, the unsupervised joint training method is much simpler and doesn't share these disadvantages with the alignment methods, but there are also some key limitations:

1.

It assumes that all shared words across two languages serve implicitly as translations and thus need not be aligned to other words.

Nonetheless, this assumption is not always true, leading to misalignment.

For example, the English word "the" will most likely also appear Figure 1 : PCA visualization of English and Spanish embeddings learnt by unsupervised joint training as in .

As shown by plots (a) and (b), most words are shared in the initial embedding space but not well-aligned, hence the oversharing problem.

Plots (b) and (c) shows that the vocabulary reallocation step effectively mitigates oversharing while the alignment refinement step further improves the poorly aligned embeddings by projecting them into a close neighborhood.

in the training corpus of Spanish, but preferably it should be paired with Spanish words such as "el" and "la" instead of itself.

We refer to this problem as oversharing.

2.

It does not utilize any explicit form of seed dictionary as in alignment methods, resulting in potentially less accurate alignments, especially for words that are not shared.

Lastly, while the supervised joint training approach does not have the same issues of unsupervised joint training, it shares limitation 1 of the alignment methods.

We empirically compare both joint training and alignment approaches in Section 4 and shed light on some of these pros and cons for both paradigms (See Section 4.3.1).

Motivated by the pros and cons of both paradigms, we propose a unified framework that first uses unsupervised joint training as a coarse initialization and then applies alignment methods for refinement, as demonstrated in Figure 1 .

Specifically, we first build a single set of embeddings with a shared vocabulary through unsupervised joint training, so as to alleviate the limitations of alignment methods.

Next, we use a vocabulary reallocation technique to mitigate oversharing, before finally resorting back to alignment methods to further improve the embeddings' quality.

Our proposed framework mainly involves three components and we discuss each of them as follows.

Joint Initialization.

We use unsupervised joint training to train the initial CLWE.

As described in Section 2.2, we first obtain a joint vocabulary V J and train its corresponding set of embeddings E J on the concatenated corpora of two languages.

This allows us to obtain a single set of embeddings that maximizes sharing across two languages.

To train embeddings, we used fastText (Bojanowski et al., 2017) in all our experiments for both word and subword tokens.

Vocabulary Reallocation.

As discussed in Secition 2.3, a key issue of unsupervised joint training is oversharing, which prohibits further refinement as shown in Figure 1 .

To alleviate this drawback, we attempt to "unshare" some of the overshared words, so their embeddings can be better aligned in the next step.

Particularly, we perform a vocabulary reallocation step such that words appearing mostly exclusively in the ith language are reallocated from the shared vocabulary V s J to V i J , whereas words that appear similarly frequent in both languages stay still in V s J .

Formally, for each token w in the shared vocabulary V s J , we use the ratio of counts within each language to determine whether it belongs to the shared vocabulary:

where C Li (w) is the count of w in the training corpus of the ith language and T Li = w C Li (w) is the total number of tokens.

The token w is allocated to the shared vocabulary if

where γ is a hyper-parameter.

Otherwise, we put w into either V Alignment Refinement.

The unsupervised joint training method does not explicitly utilize any dictionary or form of alignment.

Thus, the resulting embedding set is coarse and ill-aligned in the shared vector space, as demonstrated in Figure 1 .

As a final refinement step, we utilize any off-the-shelf alignment method to refine alignments across the non-sharing embedding sets, i.e. mapping E 1 J to E 2 J and leaving E s J untouched.

This step could be conducted by either supervised or unsupervised alignment method and we compare both in our experiments.

As our framework is highly generic and applicable to any alignment and unsupervised joint training methods, it can naturally generalize to contextualized word representations by aligning the fixed outputs of a multilingual encoder such as multilingual BERT (M-BERT) (Devlin et al., 2019) .

While our vocab reallocation technique is no longer necessary as contextualized representations are dependent on context and thus dynamic, we can still apply alignment refinement on extracted contextualized features for further improvement.

For instance, as proposed by Aldarmaki & Diab (2019) , one method to perform alignment on contextualized representations is to first use word alignment pairs extracted from parallel corpora as a dictionary, learn an alignment matrix W based on it, and apply W back to the extracted representations.

To obtain W , we can solve Eq.( 1) as described in Section 2.1, where the embedding matrices X L1 and X L2 now contain contextualized representations of aligned word pairs.

Note that this method is applicable to fixed representations but not finetuning.

We evaluate the proposed approach and compare with alignment and joint training methods on three NLP benchmarks.

This evaluation aims to: (1) systematically compare alignment vs. joint training paradigms and reveal their pros and cons discussed in Section 2.3, (2) show that the proposed framework can effectively alleviate limitations of both alignment and joint training, and (3) demonstrate the effectiveness of the proposed framework in both non-contextualized and contextualized settings.

Bilingual Lexicon Induction (BLI) This task has been the de facto evaluation task for CLWE methods.

It considers the problem of retrieving the target language translations of source langauge words.

We use bilingual dictionaries complied by and test on six diverse language pairs, including Chinese and Russian, which use a different writing script than English.

Each test set consists of 1500 queries and we report precision at 1 scores (P@1), following standard evaluation practices Glavas et al., 2019) .

Name Entity Recognition (NER) We also evaluate our proposed framework on cross-lingual NER, a sequence labeling task, where we assign a label to each token in a sequence.

We evaluate both non-contextualized and contextualized word representations on the CoNLL 2002 and 2003 benchmarks (Tjong Kim Sang, 2002 Tjong Kim Sang & De Meulder, 2003) , which contain 4 European languages.

To measure the quality of CLWE, we perform zero-shot cross-lingual classification, where we train a model on English and directly apply it to each of the other 3 languages.

Table 1 : Precision@1 for the BLI task on the MUSE dataset.

Within each category, unsupervised methods are listed at the top while supervised methods are at the bottom.

The best result for unsupervised methods is underlined while bold signfies the overall best.

"AR" refers to alignment refinement and "VR" refers to vocabulary reallocation.

Unsupervised Machine Translation (UMT) Lastly, we test our approach using the unsupervised MT task, on which the initialization of CLWE plays a crucial role .

Note that our purpose here is to directly compare with similar studies in , and thus we follow their settings and consider two language pairs, English-French and English-German, and evaluate on the widely used WMT'14 en-fr and WMT'16 en-de benchmarks.

For the BLI task, we compare our framework to recent state-of-the-art methods.

We obtain numbers from the corresponding papers or Zhou et al. (2019) , and use the official tools for MUSE , GeoMM (Jawanpuria et al., 2019) and RCSLS (Joulin et al., 2018b ) to obtain missing results.

We consider the method of Duong et al. (2016) for supervised joint training based on bilingual dictionaries, which is comparable to supervised alignment methods in terms of resources used.

For unsupervised joint training, we train uncased joint fastText 4 word vectors of dimension 300 on concatenated Wikipedia corpora of each language pair with default parameters.

The hyperparameter γ is selected from {0.7, 0.8, 0.9, 0.95} on validation sets.

For the alignment refinement step in our proposed framework, we use RCSLS and GeoMM to compare with supervised methods, and MUSE for unsupervised methods.

Following standard practices, we consider the top 200k most frequent words and use the cross-domain similarity local scaling (CSLS) as the retrieval criteria.

Note that a concurrent work proposed a new retrieval method based on MT systems and produced state-of-the-art results.

Although their method is applicable to our framework, it has high computational costs and is out of the scope of this work.

For the NER task: (1) For non-contextualized representations, we train embeddings the same way as in the BLI task and use a vanilla Bi-LSTM-CRF model Ma & Hovy, 2016) .

For all alignment steps, we apply the supervised Procrustes method using dictionaries from the MUSE library for simplicity.

(2) For contextualized representations, we consider two models, M-BERT and XLM (Lample & Conneau, 2019) , one unsupervised and one supervised joint training model, respectively.

We try our framework on M-BERT, applying alignment refinement as described in Section 3.2.

We compare our proposed framework to both fine-tuning and feature extraction.

To use the extracted features, we employ a task-specific model consisting of 2 Bi-LSTM layers with a total dimension of 768 and a CRF layer.

For finetuning, we add a softmax layer on top.

To align the contextualized embeddings, we use 30k parallel sentences from the Europarl corpus and follow the procedure of Section 3.2.

Note that, instead of BPE, we use word alignments on parallel data and use average BPE embeddings corresponding to each word to learn the alignment matrix.

We use the sum of the last 4 layer outputs as the extracted features, and learn one matrix for each layer.

For both models, we only predict the label for the first subword token corresponding to its original word.

Table 2 : Precision@1 for the BLI task on the MUSE dataset with test pairs of same surface form removed.

The best result for unsupervised methods is underlined while bold signifies the overall best.

For the UMT task, we use the exact same architecture and parameters released by 5 .

We simply use different embeddings as inputs to the model.

We compare alignment methods with joint training on all three downstream tasks.

As shown in Table  1 and Table 3 , we find alignment methods significantly outperform the joint training approach by a large margin in all language pairs for both BLI and NER.

However, the unsupervised joint training method is superior than its alignment counterpart on the unsupervised MT task as demonstrated in 2(c).

While these results demonstrate that their relative performance is task-dependent, we conduct further analysis to reveal three limitations as discussed in Sec 2.3.

First, our experiments show that unsupervised joint training fails to generate high-quality alignments due to the lack of fine-grained seed dictionary as discussed in its limitation 2.

On both BLI and NER tasks, alignment methods significantly outperform unsupervised joint training by a large margin.

We further remove test pairs of the same surface form (e.g. (hate, hate) as a test pair for en-de) of the BLI task and report their results in Table 2 .

We find unsupervised joint training to achieve extremely low scores.

This is consistent with the PCA visualization shown in Figure 1 , where embeddings of non-sharing parts are poorly aligned.

Moreover, we delve into the relative performance of the two paradigms on the MT task by plotting their test BLEU scores of the first 20 epochs in Figure 2 (a) and 2(b).

We observe that the alignment method actually obtains higher BLEU scores in the first few epochs.

These results verify unsupervised joint training alone cannot align embeddings well.

In addition, we can observe from our experiments in MT task that while alignment method performs better in the first few epochs, it gets surpassed by joint training in later epochs.

This shows the importance of parameter sharing as discussed in limitation 1 of alignment methods.

In particular, shared words can be used as a cross-lingual constraint for unsupervised joint training to achieve better finetuned performance.

The lack of sharing is also a limitation for supervised joint training method, which performs poorly on the MT task even with supervision as shown in Figure 2(c) .

Lastly, we demonstrate that oversharing can be sub-optimal for unsupervised joint training as discussed in its limitation 2.

Specifically, we conduct ablation studies for our framework in Table 1 .

Applying alignment refinement on unsupervised joint training without any vocabulary reallocation does not improve its performance.

On the other hand, a simple vocabulary reallocation alone boosts the performance by quite a margin.

This shows some words are shared erroneously across languages in unsupervised joint training thereby hindering its performance.

Our proposed framework substantially improves over its alignment and joint training baselines on all three tasks.

In particular, it outperforms existing methods on all language pairs for the BLI task (using the CSLS as retrieval metric) and achieves state-of-the-art results on 2 out of 3 language pairs for the NER task.

Besides, we show that it alleviates limitations of alignment and joint training methods shown in the previous section.

First, the proposed framework largely improves the coarse alignment of unsupervised joint training.

As shown in Table 1 , the proposed Joint Align framework achieves comparable results to prior methods in the unsupervised case and it outperforms previous state-of-the-art methods in the supervised setting.

Specifically, our proposed framework can generate well-aligned embeddings after an alignment refinement is applied to the initially ill-aligned embeddings, as demonstrated in Figure  1 .

This is further verified by results in Table 2 , where our proposed framework largely improves accuracy on words not shared between two languages over the unsupervised joint training baseline.

Besides, our ablation study in Table 1 further shows the effectiveness of the proposed vocabulary reallocation technique, which alleviates the issue of oversharing.

Particularly, we observe no improvement compared to unsupervised joint training baseline when an alignment refinement step is used without vocabulary reallocation, while a vocabulary reallocation step alone significantly boosts the performance.

This is consistent with Figure 1 and shows that the oversharing is a bottleneck for applying alignment methods to joint training.

It also suggests detecting what to share is crucial to achieve better cross-lingual transfer.

Lastly, while supervised joint training share the limitation 1 of alignment methods and perform poorly when finetuned, our proposed framework take advantage from unsupervised joint training component and exploits the idea of word sharing.

In the MT tasks, our framework obtains a maximum gain of 2.97 BLEU over baselines we ran and consistently performs better than results reported in .

In addition, Figure 2 shows that Joint Align not only converges faster in earlier training epochs but also consistently outperforms the two baselines thereafter.

These empirical findings demonstrate the effectiveness of our proposed methods in non-contextualized case.

As can be seen in Table 3 , when using our framework, we achieve state-of-the-art results on crosslingual NER on 2 out of 3 languages and the overall average.

It shows that our framework can effectively generalize to contextualized representations.

Specifically, our framework improves over the M-BERT feature extraction baseline on all three language pairs and outperforms the M-BERT finetuning counterparts.

The reason why a contextualized supervised joint training model, XLM, performs worse than its unsupervised counterpart, M-BERT, is likely that XLM uses an uncased vocabulary, where casing information is important for NER tasks.

Word embeddings (Mikolov et al., 2013a ) are a key ingredient to achieve success in monolingual NLP tasks.

However, directly using word embeddings independently trained for each language may cause negative transfer in cross-lingual transfer tasks.

In order to capture the cross-lingual mapping, a rich body of existing works relying on cross-lingual supervisions, including bilingual dictionaries (Mikolov et al., 2013a; Faruqui & Dyer, 2014; Artetxe et al., 2016; Xing et al., 2015; Duong et al., 2016; Gouws & Søgaard, 2015; Joulin et al., 2018a) , sentence-aligned corpora (Kočiskỳ et al., 2014; and documentaligned corpora (Vulić & Moens, 2016; .

Note that we are not trying to outperform state-of-the-art methods (Song et al., 2019) but rather to observe improvements of embedding initialization.

† Results reported by .

Our results are obtained using the official code released by the author.

‡ Duong et al. (2016) is a supervised method that we include for analysis purpose only and is not directly comparable to other results in this table.

Besides, unsupervised alignment methods aim to eliminate the requirement for cross-lingual supervision.

Early work of Cao et al. (2016) matches the mean and the standard deviation of two embedding spaces after alignment.

Barone (2016); Zhang et al. (2017a; b) ; adapted a generative adversarial network (GAN) (Goodfellow et al., 2014) to make the distributions of two word embedding spaces indistinguishable.

Follow-up works improve upon the GAN-based training for better stability and robustness by introducing Sinkhorn distance (Xu et al., 2018) , by stochastic self-training (Artetxe et al., 2018a) , or by introducing latent variables (Dou et al., 2018) .

While alignment methods utilize embeddings trained independently on different languages, joint training methods train word embeddings at the same time.

Klementiev et al. (2012b) train a bilingual dictionary-based regularization term jointly with monolingual language model objectives while Kočiskỳ et al. (2014) defines the cross-lingual regularization with the parallel corpus.

Another branch of methods (Xiao & Guo, 2014; Gouws & Søgaard, 2015; Ammar et al., 2016; Duong et al., 2016) build a pseudo-bilingual corpus by randomly replacing words in monolingual corpus with their translations and use monolingual word embedding algorithms to induce bilingual representations.

The unsupervised joint method by Lample & Conneau (2019) simply exploit words that share the same surface form as bilingual "supervision" and directly train a shared set of embedding with joint vocabulary.

Recently, unsupervised joint training of contextualized word embeddings through the form of multilingual language model pretraining using shared subword vocabularies has produced state-of-the-art results on various benchmarks (Devlin et al., 2019; Artetxe & Schwenk, 2019; Lample & Conneau, 2019; Pires et al., 2019; Wu & Dredze, 2019) .

A concurrent work by Ormazabal et al. (2019) also compares alignment and joint method in the bilingual lexicon induction task.

Different from their setup which only tests on supervised settings, we conduct analysis across various tasks and experiment with both supervised and unsupervised conditions.

While Ormazabal et al. (2019) suggests the combination of the alignment and joint model could potentially advance the state-of-art of both worlds, we propose a novel training framework and empirically verified its effectiveness on various tasks and settings.

In this paper, we systematically compare the alignment and joint training methods for CLWE.

We point out that the nature of each category of methods leads to certain strengths and limitations.

The empirical experiments on extensive benchmark datasets and various NLP tasks verified our analysis.

To further improve the state-of-art of CLWE, we propose a simple hybrid framework which combines the strength from both worlds and achieves significantly better performance in the BLI, MT and NER tasks.

Our work opens a promising new direction that combines two previously exclusive lines of research.

For future work, an interesting direction is to find a more optimal word sharing strategy.

<|TLDR|>

@highlight

We conduct a comparative study of cross-lingual alignment vs joint training methods and unify these two previously exclusive paradigms in a new framework. 

@highlight

This paper compares approaches to bilingual lexicon induction and shows which method performs better on lexicon, induction, and NER and MT tasks.