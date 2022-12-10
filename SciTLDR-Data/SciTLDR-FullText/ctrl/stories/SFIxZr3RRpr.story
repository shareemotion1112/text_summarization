Word alignments are useful for tasks like statistical and neural machine translation (NMT) and annotation projection.

Statistical word aligners perform well, as do methods that extract alignments jointly with translations in NMT.

However, most approaches require parallel training data and quality decreases as less training data is available.

We propose word alignment methods that require little or no parallel data.

The key idea is to leverage multilingual word embeddings – both static and contextualized – for word alignment.

Our multilingual embeddings are created from monolingual data only without relying on any parallel data or dictionaries.

We find that traditional statistical aligners are outperformed by contextualized embeddings – even in scenarios with abundant parallel data.

For example, for a set of 100k parallel sentences, contextualized embeddings achieve a word alignment F1 that is more than 5% higher (absolute) than eflomal.

Word alignment is essential for statistical machine translation and useful in NMT, e.g., for imposing priors on attention matrices (Liu et al., 2016; Alkhouli and Ney, 2017; Alkhouli et al., 2018) or for decoding (Alkhouli et al., 2016; Press and Smith, 2018) .

Further, word alignments have been successfully used in a range of tasks such as typological analysis (Lewis and Xia, 2008; Östling, 2015) , annotation projection (Yarowsky et al., 2001; Padó and Lapata, 2009 ) and creating multilingual embeddings (Guo et al., 2016) .

Statistical word aligners such as the IBM models (Brown et al., 1993) and their successors (e.g., fastalign (Dyer et al., 2013) , GIZA++ (Och and Ney, 2003) , eflomal (Östling and Tiedemann, 2016) ) are widely used for alignment.

With the rise of NMT (Bahdanau et al., 2014) , attempts have been made to interpret attention matrices as soft word alignments (Koehn and Knowles, 2017; Ghader and Monz, 2017) .

Several methods create alignments from attention matrices (Peter et al., 2017; Li et al., 2018; Zenkel et al., 2019) or pursue a multitask approach for alignment and translation (Chen et al., 2016; Garg et al., 2019) .

However, most systems require parallel data and their performance deteriorates when parallel text is scarce (cf.

Tables 1-2 in (Och and Ney, 2003) ).

Recent unsupervised multilingual embedding algorithms that use only monolingual data provide high quality static and contextualized embeddings (Conneau et al., 2018; Devlin et al., 2019; Pires et al., 2019) .

Our key idea is to leverage these embeddings for word alignments -without relying on parallel data.

Requiring no or little parallel data is advantageous in many scenarios, e.g., in the low-resource case and in domain-specific settings without parallel data.

A lack of parallel data cannot be easily remedied: mining parallel sentences is possible (cf. (Schwenk et al., 2019) ) but assumes that monolingual corpora contain parallel sentences.

Contributions: (1) We propose two new alignment methods based on the matrix of embedding similarities.

(2) We propose two post-processing algorithms that handle null words and integrate positional information.

(3) We show that word alignments obtained from multilingual BERT outperform strong statistical word aligners like eflomal.

(4) We investigate the differences between word and subword processing for alignments and find subword processing to be preferable.

Upon acceptance we will publish the source code.

Consider parallel sentences s (e) , s (f ) , with lengths l e , l f in languages e, f .

Assume we have access to some embedding function E that assigns each word in a sentence a d-dimensional vector, i.e., E(s (k) ) ∈ R l k ×d for k ∈ {e, f }.

Let E(s (k) ) i denote the vector of the i-th word in sentence s (k) .

We define the similarity matrix as the matrix S ∈ [0, 1] le×l f induced by the embeddings where S ij := sim E(s (e) ) i , E(s (f ) ) j is some normalized measure of similarity, e.g., cosine-similarity normalized to be between 0 and 1.

We now introduce methods for extracting alignments from S, i.e., obtaining a binary matrix A ∈ {0, 1} le×l f .

Argmax.

A simple baseline is to align each word in sentence s (e) with the most similar word in s (f ) and vice versa.

That is, we define A ij = 1 if and only if

and A ij = 0 else.

Similar methods have been applied to Dice coefficients (cf. (Och and Ney, 2003) ) and attention matrices (Garg et al., 2019) .

Match.

Argmax finds a local, not a global optimum.

To address this, we frame alignment as an assignment problem: we search for a maximumweight maximal matching (cf. (Ramshaw and Tarjan, 2012) ) in the bipartite weighted graph induced by the similarity matrix.

This optimization problem is given by

A ij S ij subject to A being a valid maximal matching (i.e., every word in the shorter sentence is aligned).

There are known algorithms to solve the above problem in polynomial time (cf.

Kuhn (1955) ).

Note that alignments generated with the matching method are inherently bidirectional and do not require any symmetrization as post-processing.

Distortion Correction [Dist] .

Distortion, as introduced in IBM Model 2, is essential for alignments based on non-contextualized embeddings since the similarity of two words is solely based on their surface form, independent of position.

To penalize high distortion, we multiply the similarity matrix S componentwise with

where κ is a hyperparameter to scale the matrix between [(1 − κ), 1].

We use κ = 0.5.

1 We can interpret this as imposing a locality-preserving prior:

1 See supplementary for different values.

given a choice, a word should be aligned to a word with a similar relative position ((i/l e − j/l f ) 2 close to 0) rather than a more distant word (large

Null.

Null words model untranslated words and are an important part of alignment models.

We remove alignment edges when the normalized entropy of the similarity distribution is above a threshold τ , a hyperparameter.

2 Intuitively if a word is not similar to any of the words in the target sentence, we do not align it.

That is, we set A ij = 0 if and only if

Traditional word alignment models create forward and backward alignments and then symmetrize them (Koehn, 2010) .

We compared grow-diagfinal-and (GDFA) and intersection and found them to perform comparably.

4 We use GDFA throughout the paper.

We investigate both subword segmentations such as BPE/wordpiece (Sennrich et al., 2016)

Our test data are three language pairs in different domains.

We use Europarl gold alignments 7 for English-German, Bible gold alignments (Melamed, 1998) for English-French and gold alignments by Tavakoli and Faili (2014) for English-Persian (domain: books).

We select additional parallel training data that is consistent with the target domain where available: Europarl (Koehn, 2005) for English-German and Parallel Bible Corpus (PBC) (Mayer and Cysouw, 2014) for English-French and EnglishPersian.

For fast-align, GIZA++ and eflomal we add 10,000 parallel sentences (simulating a midresource setting) to the gold standard as training data.

We show the effect of adding more or less training data in Figure 1 .

Since mBERT is pretrained on Wikipedia, we train fastText embeddings on Wikipedia as well.

For hyperparameters of all models see supplementary.

Our evaluation measures are precision, recall, F 1 and alignment error rate (AER).

Overall.

Table 1 shows that mBERT performs consistently best.

Eflomal has high precision and is on par for ENG-FRA.

Surprisingly, fastText outperforms fast-align in two out of three languages (e.g., F 1 65 vs. 61 for ENG-DEU) despite not having access to parallel data.

Among the statistical baselines, eflomal outperforms GIZA++, while GIZA++ is better than fast-align, as expected.

Parallel Data.

Figure 1 shows that fast-align and eflomal get better with more training data with eflomal outperforming fast-align, as expected.

However, even with 10 6 parallel sentences mBERT outperforms both statistical baselines.

fastText becomes competitive for fewer than 1000 parallel sentences.

The main takeaway is that mBERT-based alignments, a method that does not need any parallel training data, outperform state-of-the-art aligners, even in the high resource case.

Word vs. Subword.

In Table 1 subword processing mainly benefits fast-align, GIZA++ and eflomal (except for ENG-FRA).

fastText is harmed by subword processing.

We use VecMap to match (sub)word distributions across languages.

We hypothesize that it is harder to match subword than word distributions -this effect is strongest for Persian, probably due to different scripts and thus different subword distributions.

For mBERT words and subwords are about the same.

Table 4 compares alignment and postprocessing methods.

Argmax generally yields higher precision whereas Match has higher recall.

For fastText using Argmax with Dist yields best F 1 on two languages.

Adding a distortion prior boosts performance for static embeddings, e.g., from .46 to .61 for ENG-FRA F 1 .

Null-word processing increases precision, e.g., from .91 to .96 for ENG-DEU, but does not increase F 1 .

For mBERT Argmax performs best in two out of three language pairs.

Dist has little and sometimes harmful effects on mBERT indicating that mBERT's contextualized representations already match well across languages.

mBERT Layers.

Figure 2 shows a parabolic trend with layer 8 of mBERT yielding the best performance.

This is consistent with other work (Voita et al., 2019; Tenney et al., 2019) : in the first layers the contextualization is too weak for high-quality alignments while last layers are too specialized on the pretraining task (masked language modeling).

Brown et al. (1993) introduced the IBM models, the best known statistical word aligners.

More recent aligners, often based on IBM models, include fastalign (Dyer et al., 2013) , GIZA++ (Och and Ney, 2003) and eflomal (Östling and Tiedemann, 2016) .

All of these models are trained on parallel text.

Our method instead aligns based on embeddings that are induced from monolingual data only.

Prior work on using learned representations for alignment includes (Smadja et al., 1996; Och and Ney, 2003) (2019) compute similarity matrices of encoder-decoder representations that are leveraged for word alignments, together with supervised learning which requires manually annotated alignment.

In contrast to our work, they all require parallel data.

We presented word aligners based on contextualized (resp.

static) embeddings that perform better than (resp.

comparably with) statistical word aligners.

Our method is the first that does not require parallel data and is particularly useful for scenarios where a medium number of parallel sentences need to be aligned, but no additional parallel data is available.

For a set of 100k parallel sentences, contextualized embeddings achieve an alignment F 1 that is 5% higher (absolute) than eflomal.

Given a set of predicted alignment edges A and a set of sure (possible) gold standard edges S (P ) we computed our evaluation measures as follows:

where | · | denotes the cardinality of a set.

This is the usual way of evaluating alignments (Och and Ney, 2003) .

For asymmetric alignments different symmetrization methods exist.

fast-align (Dyer et al., 2013) provides an overview and implementation for these methods, which we use.

We compare intersection and grow-diag-final-and (GDFA) in Table 3 .

In terms of F1 GDFA performs better (Intersection wins once, GDFA five times, three ties).

As expected, Intersection yields higher precision while GDFA yields higher recall.

Thus intersection is preferable for tasks like annotation projection, whereas GDFA is typically used in statistical machine translation.

The analogous numbers from Table 1 in the main paper on word-level can be found in Table 4 .

Again Distortion is essential for fastText and not necessary for mBERT.

Adding Null helps especially for mBERT.

Overall the takeaways are consistent with the results from subword-level.

We provide a list of customized hyperparameters used in our computations in hyperparameters we used default values as provided in the corresponding implementation (see respective links to the code repositories).

In the main paper we introduced the hyperparameter κ.

In Figure 3 we plot the performance for different values of κ.

We observe that introducing distortion indeed helps (i.e., κ > 0) but the actual value is not decisive for performance.

This is rather intuitive, as a small adjustment to the similarities is sufficient while larger adjustments do not necessarily hurt or change the Argmax or the optimal point in the Matching Algorithm.

In the main paper we have chosen κ = 0.5.

For τ in the null-word post-processing we need to use high values as the similarity distributions tend to be quite uniform in the high-dimensional spaces.

See Figure 4 for different values of τ .

As expected, for τ = 1 no edges are removed and thus the performance is not changed compared to not having a null-word post-processing.

With decreasing τ the precision increases and recall goes down.

We use τ = 0.999 for fastText and τ = 0.9995 for mBERT.

Upon acceptance we will publish the code together with instructions on how to reproduce the results.

Table 6 provides an overview on the data used in the main paper together with download links.

<|TLDR|>

@highlight

We use representations trained without any parallel data for creating word alignments.