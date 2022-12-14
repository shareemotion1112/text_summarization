Most of recent work in cross-lingual word embeddings is severely Anglocentric.

The vast majority of lexicon induction evaluation dictionaries are between English and another language, and the English embedding space is selected by default as the hub when learning in a multilingual setting.

With this work, however, we challenge these practices.

First, we show that the choice of hub language can significantly impact downstream lexicon induction performance.

Second, we both expand the current evaluation dictionary collection to include all language pairs using triangulation, and also create new dictionaries for under-represented languages.

Evaluating established methods over all these language pairs sheds light into their suitability and presents new challenges for the field.

Finally, in our analysis we identify general guidelines for strong cross-lingual embeddings baselines, based on more than just Anglocentric experiments.

Continuous distributional vectors for representing words (embeddings) (Turian et al., 2010) have become ubiquitous in modern, neural NLP.

Cross-lingual representations (Mikolov et al., 2013) additionally represent words from various languages in a shared continuous space, which in turn can be used for Bilingual Lexicon Induction (BLI).

BLI is often the first step towards several downstream tasks such as Part-Of-Speech (POS) tagging (Zhang et al., 2016) , parsing (Ammar et al., 2016) , document classification (Klementiev et al., 2012) , and machine translation (Irvine and CallisonBurch, 2013; Artetxe et al., 2018b; Lample et al., 2018) .

Often, such shared representations are learned with a two-step process, whether under bilingual or multilingual settings (hereinafter BWE and MWE, respectively) .

First, monolingual word embeddings are learned over large swaths of text; such pre-trained word embeddings, in fact, are available for several languages and are widely used, like the fastText Wikipedia vectors (Grave et al., 2018) .

Second, a mapping between the languages is learned, in one of three ways: in a supervised manner if dictionaries or parallel data are available to be used for supervision (Zou et al., 2013) , under minimal supervision e.g. using only identical strings (Smith et al., 2017) , or even in a completely unsupervised fashion (Zhang et al., 2017; Conneau et al., 2018) .

Both in bilingual and multilingual settings, it is common that one of the language embedding spaces is the target to which all other languages get aligned to (hereinafter "the hub").

We outline the details in Section 2.

Despite all the recent progress in learning cross-lingual embeddings, we identify a major shortcoming to previous work: it is by and large English-centric.

Notably, most MWE approaches essentially select English as the hub during training by default, aligning all other language spaces to the English one.

We argue and empirically show, however, that English is a poor hub language choice.

In BWE settings, on the other hand, it is fairly uncommon to denote which of the two languages is the hub (often this is implied to be the target language).

However, we experimentally find that this choice can greatly impact downstream performance, especially when aligning distant languages.

This Anglocentricity is even more evident at the evaluation stage.

The lexica most commonly used for evaluation are the MUSE lexica (Conneau et al., 2018) which cover 45 languages, but with translations only from and into English.

Even still, alternative evaluation dictionaries are also very English-and European-centric: Dinu and Baroni (2014) report results on English-Italian, Artetxe et al. (2017) on English-German and English-Finnish, Zhang et al. (2017) on Spanish-English and Italian-English, and Artetxe et al. (2018a) between English and Italian, German, Finnish, Spanish, and Turkish.

We argue that cross-lingual word embedding mapping methods should look beyond English for their evaluation benchmarks because, compared to all others, English is a language with disproportionately large available data and relatively poor inflectional morphology e.g., it lacks case, gender, and complex verbal inflection systems (Aronoff and Fudeman, 2011) .

These two factors allow for an overly easy evaluation setting which does not necessarily generalize to other language pairs.

In light of this, equal focus should instead be devoted to evaluation over more diverse language pairs that also include morphologically rich and low-resource languages.

With this work, we attempt to address these shortcomings, providing the following contributions:

??? We show that the choice of the hub when evaluating on diverse language pairs can lead to significantly different performance (e.g., by more than 10 percentage points for BWE over distant languages).

We also show that often English is a suboptimal hub for MWE.

??? We identify some general guidelines for choosing a hub language which could lead to stronger baselines; less isometry between the hub and source and target embedding spaces mildly correlates with performance, as does typological distance (a measure of language similarity based on language family membership trees).

For distant languages, multilingual systems should in most cases be preferred over bilingual ones.

??? We provide resources for training and evaluation on non-Anglocentric language pairs.

We outline a simple triangulation method with which we extend the MUSE dictionaries to an additional 2352 lexicons covering 49 languages, and we present results on a subset of them.

We also create new evaluation lexica for under-resourced languages using Azerbaijani, Belarusian, and Galician as our test cases.

We additionally provide recipes for creating such dictionaries for any language pair with available parallel data.

In the supervised bilingual setting, as formulated by Mikolov et al. (2013) , given two languages L = {l 1 , l 2 } and their pre-trained row-aligned embeddings X 1 , X 2 , respectively, a transformation matrix M is learned such that:

The set ??? can potentially impose a constraint over M , such as the very popular constraint of restricting it to be orthogonal (Xing et al., 2015) .

Previous work has empirically found that this simple formulation is competitive with other more complicated alternatives (Xing et al., 2015; Conneau et al., 2018) .

The orthogonality assumption ensures that there exists a closed-form solution in the form of the Singular Value Decomposition (SVD) of X 1 X T 2 .

1 Note that in this case only a single matrix M needs to be learned, because X 1 ??? M X 2 = M ???1 X 1 ??? X 2 , while at the same time a model that minimizes X 1 ??? M X 2 is as expressive as one minimizing M 1 X 1 ??? M 2 X 2 , and easier to learn.

In the minimally supervised or even the unsupervised setting (Zhang et al., 2017 ) the popular methods follow an iterative refinement approach (Artetxe et al., 2017) .

Starting with a seed dictionary (e.g. from identical strings (Zhou et al., 2019) or numerals) an initial mapping is learned in the same manner as in the supervised setting.

The initial mapping, in turn, is used to expand the seed dictionary with high confidence word translation pairs.

The new dictionary is then used to learn a better mapping, and so forth the iterations continue until convergence.

We will generally refer to such methods as MUSE-like.

Similarly, in a multilingual setting, one could start with N languages L = {l 1 , l 2 , . . .

, l N } and their respective pre-trained embeddings X 1 , X 2 , . . .

, X N , and then learn N ???1 bilingual mappings between a pre-selected target language and all others.

Hence, one of the language spaces is treated as a target (the hub) and remains invariant, while all others are mapped into the (now shared) hub language space.

Alternatively, those mappings could be jointly learned using the MAT+MPSR methods of Chen and Cardie (2018) -also taking advantage of the inter-dependencies between any two language pairs.

Importantly, though, there is no closed form solution for learning the joint mapping, hence a solution needs to be approximated with gradient-based methods.

MAT+MPSR generalizes the adversarial approach of Zhang et al. (2017) to multiple languages, and also follows an iterative refinement 2 In either case, a language is chosen as the hub, and N ??? 1 mappings for the other languages are learned.

Other than MAT+MPSR, the only other unsupervised multilingual approach is that of Heyman et al. (2019) , who propose to incrementally align multiple languages by adding each new language as a hub.

We decided, though, against comparing to this method, because (a) their method requires learning O(N 2 ) mappings for relatively small improvements and (b) the order in which the languages are added is an additional hyperparameter that would explode the experimental space.

Lexicon Induction One of the most common downstream evaluation tasks for the learned crosslingual word mappings is Lexicon Induction (LI), the task of retrieving the most appropriate wordlevel translation for a query word from the mapped embedding spaces.

Specialized evaluation (and training) dictionaries have been created for multiple language pairs, with the MUSE dictionaries (Conneau et al., 2018 ) most often used, providing word translations between English (En) and 48 other high-to mid-resource languages, as well as on all 30 pairs among 6 very similar Romance and Germanic languages (English, French, German, Spanish, Italian, Portuguese).

Given the mapped embedding spaces, the translations are retrieved using a distance metric, with Cross-Lingual Similarity Scaling (Conneau et al., 2018, CSLS) as the most common and best performing in the literature.

Intuitively, CSLS decreases the scores of pairs that lie in dense areas, increasing the scores of rarer words (which are harder to align).

The retrieved pairs are compared to the gold standard and evaluated using precision at k (P@k, evaluating how often the correct translation is within the k retrieved nearest neighbours of the query).

Throughout this work we report P@1, which is equivalent to accuracy, but we also provide results with P@5 and P@10 in the Appendix.

As other works have recently noted (Czarnowska et al., 2019 ) the typically used evaluation dictionaries cover a narrow breadth of the possible language pairs, with the majority of them focusing in pairs with English (as with the MUSE dictionaries) or among high-resource European languages.

In this section, we first outline our method for creating new dictionaries for low resource languages.

Then, we describe the simple triangulation process that allows us to create dictionaries among all 49 MUSE languages.

Our approach for constructing dictionaries is fairly straightforward, inspired by phrase table extraction techniques from phrase-based MT (Koehn, 2009) .

Rather than manual inspection, however, which would be impossible for all language pairs, we rely on fairly simple heuristics for controlling the quality of our dictionaries.

The first step is collecting publicly available parallel data between English and the low-resource language of interest.

We use data from the TED (Qi et al., 2018) , OpenSubtitles (Lison and Tiedemann, 2016) , WikiMatrix (Schwenk et al., 2019) , bible (Malaviya et al., 2017) , and JW300 (Agi?? and Vuli??, 2019) datasets.

4 This results in 354k, 53k, and 623k English-to-X parallel sentences for Azerbaijani (Az), Belarusian (Be), and Galician (Gl) respectively.

5 We align the parallel sentences using fast align (Dyer et al., 2013) , and extract symmetrized alignments using the gdfa heuristic (Koehn et al., 2005) .

In order to ensure that we do not extract highly domain-specific word pairs, we only use the TED, OpenSubtitles, and WikiMatrix parts for word-pair extraction.

Also, in order to control for quality, we only extract word pairs if they appear in the dataset more than 5 times, and if the alignment probability is higher than 30%.

With this process, we end up with about 6k, 7k, and 38k word pairs for Az-En, Be-En, and GlEn respectively.

Following standard conventions, we sort the word pairs according to source-side frequency, and use the intermediate-frequency ones for evaluation, typically using the 5000-6500 rank boundaries.

The same process can be followed for any language pair with enough volume of parallel data (needed for training a decent word alignment model).

In fact, we can produce similar dictionaries for a large number of languages, as the combination of the recently created JW300 and WikiMatrix datasets provide an average of more than 100k parallel sentences in 300 languages.

Our second method for creating new dictionaries is inspired from phrase table triangulation ideas from the pre-neural MT community (Wang et al., 2006; Levinboim and Chiang, 2015) .

The concept can be easily explained with an example, visualized in Figure 1 .

Consider the Portuguese (Pt) word trabalho which, according to the MUSE Pt-En dictionary, has the words job and work as possible En translations.

In turn, these two En words can be translated to 4 and 5 Czech (Cs) words respectively.

By utilizing the transitive property (which translation should exhibit) we can identify the set of 7 possible Cs translations for the Pt word trabalho.

Following this simple triangulation approach, we create 2352 new dictionaries over language pairs among the 49 languages of the MUSE dictionaries.

7 For consistency, we keep the same train and test splits as with MUSE, so that the source-side types are equal across all dictionaries with the same source language.

Triangulating through English (which is unavoidable, due to the lack of non-English-centric dictionaries) is suboptimal -English is morphologically poor and lacks gender information.

As a result, several inflected forms in morphologically-rich languages map to the same English form.

Similarly, gendered nouns or adjectives in gendered languages map to English forms that lack gender information.

For example, the MUSE Greek-English dictionary lists the word peaceful as the translation for all ???????????????????, ?????????????????, ?????????????????, ?????????????????, which are the male, female, and neutral (singular and plural) inflections of the same adjective.

Equivalently, the English-Italian dictionary translates peaceful into either pacifico, pacifici, or pacifica (male singular, male plural, and female singular, respectively; see Table 1 ).

When translating from or into English lacking context, all of those are reasonable translations.

When translating between Greek and Italian, though, one should take gender and number into account.

Hence, we devise a filtering method for removing blatant mistakes when triangulating morphologically rich languages.

We rely on automatic morphological tagging which we can obtain for most of the MUSE languages, using the StanfordNLP toolkit (Manning et al., 2014) .

The morphological tagging uses the Universal Dependencies feature set (Nivre et al., 2016) making the tagging comparable across almost all languages.

Our filtering technique iterates through the bridged dictionaries: for a given source word, if we find a translation word with the exact same morphological analysis, we filter out all other translations with the same lemma but different tags.

In the case of feature mismatch (for instance, Greek uses 4 cases and 3 genders while Italian has 2 genders and no cases) or if we only find a partial tag match over a feature subset, we filter out translations with disagreeing tags.

Coming back to our Greek-Italian example, this means that for the form ??????????????????? we would only keep pacifico as a candidate translation (we show more examples in Table 1 ).

Our filtering technique removes about 17% of the entries in our bridged dictionaries.

Naturally, this filtering approach is restricted to languages for which a morphological analyzer is available.

Miti- Table 2 : Lexicon Induction performance (measured with P@1) over 10 European languages (90 pairs).

In each cell, the superscript denotes the hub language that yields the best result for that language pair.

?? best : average using the best hub language.

?? En : average using the En as the hub.

The shaded cells are the only language pairs where a bilingual MUSE system outperforms MAT+MSPR.

For our main MWE experiments, we train MAT+MPSR systems to align several language subsets varying the hub language.

For BWE experiments, we compare MUSE with MAT+MPSR.

The differences in LI performance show the importance of the hub language choice with respect to each evaluation pair.

As part of our call for moving beyond Anglo-centric evaluation, we also present LI results on several new language pairs using our triangulated dictionaries.

It is worth noting that we are predominantly interested in comparing the quality of the multilingual alignment when different hub languages are used.

Hence, even slightly noisy dictionaries (like our low-resource language ones) are still useful.

Even if the skyline performance (from e.g. a perfect system) would not reach 100% accuracy due to noise, the differences between the systems' performance can be revealing.

We first focus on 10 European languages of varying morphological complexity and data availability (which affects the quality of the pre-trained word embeddings): Azerbaijani (Az), Belarusian (Be), Czech (Cs), English (En), Galician (Gl), Portuguese (Pt), Russian (Ru), Slovak (Sk), Spanish (Es), and Turkish (Tr).

The choice of these languages additionally ensures that for our three low-resource languages (Az, Be, Gl) we include at least one related higher-resource language (Tr, Ru, Pt/Es respectively), allowing for comparative analysis.

Table 2 summarizes the best post-hoc performing systems for this experiment.

In the second setting, we use a set of 7 more distant languages: English, French (Fr), Hindi (Hi), Korean (Ko), Russian, Swedish (Sv), and Ukrainian (Uk).

This language subset has large variance in terms of typology and alphabet.

The best performing systems are presented in Table 3 .

Experimental Setup We train and evaluate all models starting with the pre-trained Wikipedia FastText embeddings for all languages (Grave et al., 2018) .

We focus on the minimally supervised scenario which only uses similar character strings between any languages for supervision in order to mirror the hard, realistic scenario of not having annotated training dictionaries between the languages.

We learn MWE with the MAT+MPSR method (Chen and Cardie, 2018) using the publicly available code.

8 We also use MAT+MPSR for BWE experiments, but we additionally train and compare to MUSE systems 9 (Conneau et al., 2018) .

We compare the statistical significance of the difference in performance from two systems using paired bootstrap resampling (Koehn, 2004) .

Generally, a difference of 0.4-0.5 percentage points evaluated over our lexica is significant with p < 0.05.

The hub matters for distant languages When using MUSE, the answer is simple: the closed form solution of the Procrustes problem is provably direction-independent, and we confirm this empirically (we provide complete results on MUSE in Table 15 in the Appendix).

However, obtaining good performance with such methods requires the orthogonality assumption to hold, which for distant languages is rarely the case (Patra et al., 2019) .

In fact, we find that the gradient-based MAT+MPSR method in a bilingual setting over distant languages exhibits better performance than MUSE.

Across Tables 2 and 3, in only a handful of examples (shaded cells) does MUSE outperform MAT+MPSR for BWE.

On the other hand, we find that when aligning distant languages with MAT+MPSR, the difference between hub choices can be significant -in Az-En, for instance, using En as the hub leads to more than 7 percentage points difference to using Az.

We show some examples in Table 4 .

On the other hand, when aligning typologically similar languages, the difference is less pronounced.

For example, we obtain practically similar performance for Gl-Pt, Az-Tr, or Uk-Ru when using either the source or the target language as the hub.

Note, though, that non-negligible differences could still occur, as in the case of Pt-Gl.

In most cases, it is the case that the higher-resourced language is a better hub than the lower-resourced one, especially when the number of resources defer significantly (as in the case of Az and Be against any other language).

Since BWE settings are not our main focus, we leave an extensive analysis of this observation for future work.

MWE:

English is rarely the best hub language In multilingual settings, we conclude that the standard practice of choosing English as the hub language is sub-optimal.

Out of the 90 evaluation pairs from our European-languages experiment (Table 2) the best hub language is English in only 17 instances (less than 20% of the time).

In fact, the average performance (over all evaluation pairs) when using En as the hub (denoted as ?? En ) is 1.3 percentage points worse than the optimal (?? best ).

In our distant-languages experiment (Table 3 ) English is the best choice only for 7 of the 42 evaluation pairs (again, less than 20% of the time).

As before, using En as the hub leads to an average drop of one percentage point in performance aggregated over all pairs, compared to the averages of the optimal selection.

The rest of the section attempts to provide an explanation for these differences.

Expected gain for a hub language choice As vividly outlined by the superscript annotations in Tables 2 and 3 , there is not a single hub language that stands out as the best one.

Interestingly, all languages, across both experiments, are the best hub language for some evaluation language pair.

For example, in our European-languages experiment, Es is the best choice for about 20% of the evaluation pairs, Tr and En are the best for about 17% each, while Gl and Be are the best for only 5 and 3 language pairs respectively.

Clearly, not all languages are equally suited to be the hub language for many language pairs.

Hence, it would be interesting to quantify how much better one could do by selecting the best hub language compared to a random choice.

In order to achieve this, we define the expected gain G l of using language l as follows.

Assume that we are interested in mapping N languages into the shared space m l is the accuracy 10 over a specified evaluation pair m when using language l as the hub.

The random choice between N languages will have an expected accuracy equal to the average accuracy when using all languages as hub: The gain for that evaluation dataset m when using language l as hub, then, is g

.

Now, for a collection of M evaluation pairs we simply average their gains, in order to obtain the expected gain for using language l as the hub:

The results of this computation for both sets of experiments are presented in Figure 2 .

The bars marked 'overall' match our above definition, as they present the expected gain computed over all evaluation language pairs.

For good measure, we also present the average gain per language aggregated over the evaluation pairs where that language was indeed the best hub language ('when best' bars).

Perhaps unsurprisingly, Az seems to be the worst hub language choice among the 10 European languages of the first experiment, with an expected loss (negative gain) of -0.4.

This can be attributed to how distant Az is from all other languages, as well as to the fact that the Az pre-trained embeddings are of lower quality compared to all other languages (as the Az Wikipedia dataset is significantly smaller than the others).

Similarly, Hi and Sv show expected loss for our second experiment.

Note that English is not a bad hub choice per se -it exhibits a positive expected gain in both sets of experiments.

However, there are languages with larger expected gains, like Es and Gl in the European-languages experiment that have a twice-as-large expected gain, while Ru has a 4 times larger expected gain in the distant-languages experiment.

Of course, the language subset composition of these experiments could possibly impact those numbers.

For example, there are three very related languages (Es, Gl, Pt) in the European languages set, which might boost the expected gain for that subset; however, the trends stand even if we compute the expected gain over a subset of the evaluation pairs, removing all pairs that include Gl or Pt.

For example, after removing all Gl results, Es has a slightly lower expected gain of 0.32, but is still the language with the largest expected gain.

Identifying the best hub language for a given evaluation set The next step is attempting to identify potential characteristics that will allow us make educated decisions with regards to choosing the hub language, given a specific evaluation set.

For example, should one choose a language typologically similar to the evaluation source, target, or both?

Or should they use the source or the target of the desired evaluation set as the hub?

Our first finding is that the best performing hub language will very likely be neither the source nor the target of the evaluation set.

In our European-languages experiments, a language different than the source and the target yields the best accuracy for over 93% of the evaluation sets.

Similarly, in the distant-languages experiment, there is only a single instance where the best performing hub language is either the source or the target evaluation language (for the Fr-Ru dataset), and for the other 97% of the cases the best option is a third language.

We hypothesize that learning mappings for both language spaces of interest (hence rotating both spaces) allows for a more flexible alignment which leads to better downstream performance, compared to when one of the two spaces is fixed.

Note that this contradicts the mathematical intuition discussed in Section 2 according to which a model learning a single mapping (keeping another word embedding space fixed) is as expressive as a model that learns two mappings for each of the languages.

Our second finding is that the downstream performance correlates with measures of distance between languages and language spaces.

The typological distance (d gen ) between two languages can be approximated through their genealogical distance over hypothesized language family trees, which we obtain from the URIEL typological database (Littell et al., 2017) .

Also, Patra et al. (2019) recently motivated the use of Gromov-Hausdroff (GH) distance as an a priori estimation of how well two language embedding spaces can be aligned under an isometric transformation (which is an assumption most methods rely on).

The authors also note that vector space GH distance correlates with typological language distance.

We refer the reader to Patra et al. (2019) We find that there is a positive correlation between downstream LI performance and the genealogical distances between the source-hub and target-hub languages.

The average (over all evaluation pairs) Pearson's correlation coefficient between P@1 and d gen is 0.49 for the distant languages experiment and 0.38 for the European languages one.

A similar positive correlation of performance and the sum of the GH distances between the source-hub and target-hub spaces.

On our distant languages experiment, the coefficient between P@1 and GH is equal to 0.45, while it is slightly lower (0.34) for our European languages experiment.

High correlation examples from each experiment, namely Gl-En and En-Hi, are shown in Figure 3 .

Bi-, tri-, and multilingual systems The last part of our analysis compares bilingual, trilingual, and multilingual systems, with a focus on the under-represented languages.

Through multiple experiments (complete evaluations are listed in the Appendix) we reach two main conclusions.

On one hand, when evaluating on typologically distant languages, one should use as many languages as possible.

In Table 5 we present one such example with results on Az-Cs under various settings.

On the other hand, when multiple related languages are available, one can achieve higher performance with multilingual systems containing all related languages and one more hub language, rather than learning diverse multilingual mappings using more languages.

We confirm the latter observation with experiments on the Slavic (Be, Ru, Uk) and Iberian (Es, Gl, Pt) clusters, and present an example (Ru-Uk) in Table 5 .

With this work we challenge the standard practices in learning cross-lingual word embeddings.

We empirically showed that the choice of the hub language is an important parameter that affects lexicon induction performance in both bilingual (between distant languages) and multilingual settings.

More importantly, we hope that by providing new dictionaries and baseline results on several language pairs, we will stir the community towards evaluating all methods in challenging scenarios that include under-represented language pairs.

Towards this end, our analysis provides insights and general directions for stronger baselines for non-Anglocentric cross-lingual word embeddings.

A Does evaluation directionality matter?

We also explored whether there are significant differences between the evaluated quality of aligned spaces, when computed on both directions (src-trg and trg-src).

We find that the evaluation direction indeed matters a lot, when the languages of the evaluation pair are very distant, in terms of morphological complexity and data availability (which affects the quality of the original embeddings).

A prominent example, from our European-languages experiment, are evaluation pairs involving Az or Be.

When evaluating on the Az-XX and Be-XX dictionaries, the word translation P@1 is more than 20 percentage points higher than when evaluating on the opposite direction (XX-Az or XX-Be).

For example, Es-Az has a mere P@1 of 9.9, while Az-Es achieves a P@1 of 44.9.

This observation holds even between very related languages (cf.

Ru-Be: 12.8, Be-Ru: 41.1 and Tr-Az: 8.4, Az-Tr: 32.0), which supports our hypothesis that this difference is also due to the quality of the pre-trained embeddings.

It is important to note that such directionality differences are not observed when evaluating distant pairs with presumably high-quality pre-trained embeddings e.g. Tr-Sk or Tr-Es; the P@1 for both directions is very close.

Here we provide complete evaluation results for our multilingual experiments.

Tables 6-11 present P@1, P@5, and P@10 respectively, for the experiment on the 10 European languages.

Similarly, results on the distant languages experiment are shown in Tables 12, 13 , and 14.

Table 15 presents the P@1 of the bilingual experiments using MUSE.

7.5 7.0 Ru-Be 12.8 9.9 10.7 11.5 11.2 11.0 11.5 12.3 11.0 11.8 11.

Tr-Sk 27.5 29.2 27.9 28.5 29.4 27.7 27.9 27.5 25.2 27.9 27.9

<|TLDR|>

@highlight

The choice of the hub (target) language affects the quality of cross-lingual embeddings, which shouldn't be evaluated only  on English-centric dictionaries.