We extend the Consensus Network framework to Transductive Consensus Network (TCN), a semi-supervised multi-modal classification framework, and identify its two mechanisms: consensus and classification.

By putting forward three variants as ablation studies, we show both mechanisms should be functioning together.

Overall, TCNs outperform or align with the best benchmark algorithms when only 20 to 200 labeled data points are available.

align with those of benchmark algorithms (semi-supervised or supervised, multi-modal or uni-modal) 23 on Bank Marketing and DementiaBank datasets, when 20-200 labeled data points are available.

We first briefly review the CN framework BID0 for supervised, multi-view classification.

Consider a There are M interpreter networks I m (m = 1, .., M ), each compressing one modality of features into a representation, which we call consensus interpretation vector.

A discriminator D tries to distinguish the origin of each latent representation.

A classifier C makes predictions based on all representations.

DISPLAYFORM0 The training is done by iteratively optimizing two targets: DISPLAYFORM1 DISPLAYFORM2 Note that empirically, an additional noise modality v 0 ∼ N (µ 1..M , σ

In this paper we extend CN to TCN.

Formally, the input data include those labeled, DISPLAYFORM0 , and unlabeled, {x (i) } (x (i) ∈ X U ).

In the semi-supervised learning setting, there 38 can be a lot more unlabeled data points than labeled: |X U | |X L |, where the whole dataset is DISPLAYFORM1

Here each data point x contains feature values from multiple modalities (i.e., 'views'), and the 41 interpreter networks I m (m = 1, .., M ), discriminator D and classifier C are set up identical to CN 42 as well.

Different from CN, the classification loss is defined on only those labeled data, while the 43 discriminator loss is defined across both labeled and unlabeled data: DISPLAYFORM0

TCNs function in two mechanisms: The consensus mechanism compresses each data sample into

"consensus interpretations", and the classifier mechanism tries to make these interpretations meaning-ful.

To perform ablation studies on these mechanisms, we test the following three variants.

whole dataset, we extract the consensus interpretations of those labeled data samples to train an SVM.

TCN-svm lets the consensus mechanism to function alone, resulting in almost trivial classifiers.

DISPLAYFORM0 , the optimization target in TCN-AE can be expressed as: DISPLAYFORM1 L C , and max DISPLAYFORM2 and min DISPLAYFORM3 As shown in FIG6 , TCN-AE has inferior performances than TCN.

Reconstruction in an autoen-58 coder style counteracts the consensus mechanisms, and should not be used with CN models.

We run experiments on two classification datasets, Table 1 : In BM, the three modalities correspond to basic information, statistical data, and employment.

In DB, the three modalities correspond to acoustic, syntactic-semantic, and lexical.

The Bank Marketing dataset is from the UCI machine learning repository [11] .

used for predicting 108 whether the customer will subscribe a term deposit in a bank marketing campaign via telephone [12] .

There are originally 4,640 positive samples (subscribe) and 36,548 negative ones (did not subscribe).

Since consensus network models do not work well on imbalanced datasets, we randomly sample 111 5,000 negative samples to create an (almost) balanced dataset.

We also convert the categorical raw Honore's statistics, word length, cosine distances between words in sentences, etc.

not change while the training classification loss is higher than log2≈ 0.693 3 .

We check once more 149 when training stops.

If the training classification loss is higher than log2, the model is re-initialized with a new random seed and the training is restarted.

Empirically this re-initialization only happen no 151 more than once per ten runs, but the underlying cause need to be examined further.

DISPLAYFORM0 where DISPLAYFORM1 where v m,j and v n,j are the j th component of v m and v n respectively.

In total, for each data sample, DISPLAYFORM2 pairs of relative divergences are calculated.

We average the negative of these divergences to 163 get the similarity for the interpretations: DISPLAYFORM3 Note that the "similarity" is defined such that its maximum possible value is 0 (where there is no JS 165 divergence between any pair of the interpretation vectors), and it has no theoretical lower bound.

Figure 4: Examples of similarity plots against the number of steps taken, for DementiaBank using 80 labeled samples ("DB80", blue) and Bank Marketing using 20 labeled samples ("BM20", green).

The y axis are scaled to (-0.035, 0) except TCN-AE, where the relative JS divergences "explode".

Note that training stops when losses converge (as detailed in §4.2), so the trials may stop at different steps. .

The three colors represent three modalities.

At step 2, the interpretations are distributed randomly.

At step 110, they become mixed evenly.

The most interesting embedding happens at step 30, when interpretations of the three modalities form three 'drumstick' shapes.

With the highest symmetricity visually, this configuration of interpretations also has the highest similarity among the three.

<|TLDR|>

@highlight

A semi-supervised multi-modal classification framework, TCN, that outperforms various benchmarks.