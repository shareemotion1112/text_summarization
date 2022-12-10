In this paper we propose to perform model ensembling in a multiclass or a multilabel learning setting using Wasserstein (W.) barycenters.

Optimal transport metrics, such as the Wasserstein distance, allow incorporating semantic side information such as word embeddings.

Using W. barycenters to find the consensus between models allows us to balance confidence and semantics in finding the agreement between the models.

We show applications of Wasserstein ensembling in attribute-based classification, multilabel learning and image captioning generation.

These results show that the W. ensembling is a viable alternative to the basic geometric or arithmetic mean ensembling.

Model ensembling consists in combining many models into a stronger, more robust and more accurate model.

Ensembling is ubiquitous in machine learning and yields improved accuracies across multiple prediction tasks such as multi-class or multi-label classification.

For instance in deep learning, output layers of Deep Neural Networks(DNNs), such as softmaxes or sigmoids, are usually combined using a simple arithmetic or geometric mean.

The arithmetic mean rewards confidence of the models while the geometric means seeks the consensus across models.

What is missing in the current approaches to models ensembling, is the ability to incorporate side information such as class relationships represented by a graph or via an embedding space.

For example a semantic class can be represented with a finite dimensional vector in a pretrained word embedding space such as GloVe BID28 .

The models' predictions can be seen as defining a distribution in this label space defined by word embeddings: if we denote p i to be the confidence of a model on a bin corresponding to a word having an embedding x i , the distribution on the label space is therefore p = i p i δ xi .

In order to find the consensus between many models predictions, we propose to achieve this consensus within this representation in the label space.

In contrast to arithmetic and geometric averaging, which are limited to the independent bins' confidence, this has the advantage of carrying the semantics to model averaging via the word embeddings.

More generally this semantic information can be encoded via cost a matrix C, where C ij encodes the dissimilarity between semantic classes i and j, and C defines a ground metric on the label space.

To achieve this goal, we propose to combine model predictions via Wasserstein (W.) barycenters BID0 , which enables us to balance the confidence of the models and the semantic side information in finding a consensus between the models.

Wasserstein distances are a naturally good fit for such a task, since they are defined with respect to a ground metric in the label space of the models, which carry such semantic information.

Moreover they enable the possiblity of ensembling predictions defined on different label sets, since the Wasserstein distance allows to align and compare those different predictions.

Since their introduction in BID0 W. barycenter computations were facilitated by entropic regularization BID6 and iterative algorithms that rely on iterative Bregman projections BID2 .

Many applications have used W. barycenters in Natural Language Processing (NLP), clustering and graphics.

We show in this paper that W. barycenters are effective in model ensembling and in finding a semantic consensus, and can be applied to a wide range of problems in machine learning (Table 1) .The paper is organized as follows: In Section 2 we revisit geometric and arithmetic means from a geometric viewpoint, showing that they are 2 and Kullback Leibler divergence KL (extended KL divergence) barycenters respectively.

We give a brief overview of optimal transport metric and W. barycenters in Section 3.

We highlight the advantages of W. barycenter ensembling in terms of semantic smoothness and diversity in Section 4.

Related work on W. barycenters in Machine learning are presented in Section 5.

Finally we show applications of Wasserstein ensembling on attribute based classification, multi-label learning and image captioning in Section 6.

Normalized and Unnormalized predictions Ensembling.

In deep learning, predictions on a label space of fixed size M are usually in one of two forms: a) normalized probabilities: in a multiclass setting, the neural network outputs a probability vector (normalized through softmax), where each bin corresponds to a semantic class; b) unnormalized positive scores: in a multi-label setting, the outputs of M independent logistic units are unnormalized positive scores, where each unit corresponds to the presence or the absence of a semantic class.

Model ensembling in those two scenarios has long history in deep learning and more generally in machine learning BID3 BID13 Wolpert, 1992) as they lead to more robust and accurate models.

As discussed in the introduction, two methods have been prominent in model ensembling due to their simplicity: majority vote using the arithmetic mean of predictions, or consensus based using the geometric mean.

Revisiting Arithmetic and Geometric Means from a geometric viewpoint.

Given m predictions µ , and weights λ ≥ 0 such that m =1 λ = 1, the weighted arithmetic mean is given byμ a = m =1 λ µ , and the weighted geometric mean byμ g = Π m =1 (µ λ ).It is instructive to reinterpret the arithmetic and geometric mean as weighted Frechet means (Definition 1) (Zemel & Panaretos, 2017) .

2 2 (the 2 Euclidean distance).

A less known fact is that the geometric mean corresponds to a Frechet Mean for d = KL, where KL is the extended KL divergence to unnormalized measures: KL(p, q) = i p i log pi qi − p i + q i .

We give proofs and properties of arithmetic and geometric mean in Appendix F.Following this geometric viewpoint, in order to incorporate the semantics of the target space in model ensembling, we need to use a distance d that takes advantage of the underlying geometry of the label space via a cost matrix C when comparing positive measures.

Optimal transport (OT) metrics such as Wasserstein-2 have this property since they are built on an explicit cost matrix defining pairwise distance between the semantic classes.

In this paper we propose to use the Frechet means with Wasserstein distance (d = W 2 2 ) for model ensembling, i.e. use Wasserstein barycenters BID0 for model ensembling: DISPLAYFORM0 Intuitively, the barycenter looks for a distribution ρ (a histogram ) that is close to all the base distributions µ in the Wasserstein sense.

In our context transporting the consensus ρ to each individual model µ should have a minimal cost, where the cost is defined by the distance in the word embedding space.

Wasserstein distances were originally defined between normalized probability vectors (Balanced OT) BID36 ), but they have been extended to deal with unnormalized measures and this problem is referred to as unbalanced OT BID5 BID14 .

Motivated by the multi-class and the multi-label ensembling applications, in the following we present a brief overview of W. barycenters in the balanced and unbalanced cases.

, p represents histograms on source label space Ω S = {x i ∈ R d , i = 1 . . .

N }, for e.g words embeddings.

Consider similarly q ∈ ∆ M representing histograms whose bins are defined on a target label space Ω T = {y j ∈ R d , j = 1 . . .

M }.

Consider a cost function c(x, y), (for example c(x, y) = x − y 2 ).

Let C be the matrix in ∈ R N ×M such that C ij = c(x i , y j ).

1 N denotes a vector with all ones.

Let γ ∈ R N ×M be a coupling matrix whose marginals are p and q such that: DISPLAYFORM0 The optimal transport metric is defined as follows: DISPLAYFORM1 When c(x, y) = x − y 2 2 , this distance corresponds to the so called Wasserstein−2 distance W 2 2 .Unbalanced OT.

When p and q are unnormalized and have different total masses, optimal transport metrics have been extended to deal with this unbalanced case.

The main idea is in relaxing the set Π(p, q) using a divergence such as the extended KL divergence: KL.

BID5 define for λ > 0 the following generalized Wasserstein distance between unnormalized measures: DISPLAYFORM2

Throughout the paper we consider m discrete prediction vectors µ ∈ R N + , = 1 . . .

m defined on a discrete space (word embeddings for instance) DISPLAYFORM0 We refer to Ω S as source spaces.

Our goal is to find a consensus predictionμ DISPLAYFORM1 Balanced W. Barycenters: Normalized predictions.

The W. barycenter BID0 of normalized predictions is defined as follows:μ w = arg min ρ m =1 λ W (ρ, µ ), for the Wasserstein distance W defined in equation (1).

Hence one needs to solve the following problem, for m coupling matrices γ , = 1 . . .

m: DISPLAYFORM2 Unbalanced W. Barycenters: Unnormalized predictions.

Similarly the W. barycenter of unnormalized predictions is defined as follows:μ w = arg min ρ m =1 λ W unb (ρ, µ ), for the Generalized Wasserstein distance W unb defined in equation (2).

Hence the unbalanced W. barycenter problem BID5 amounts to solving , for m coupling matrices γ , = 1 . . .

m: DISPLAYFORM3 3.3 COMPUTATION VIA ENTROPIC REGULARIZATION AND PRACTICAL ADVANTAGES Entropic Regularized Wasserstein Barycenters Algorithms.

The computation of the Wasserstein distance grows super-cubicly in the number of points.

This issue was alleviated by the introduction of the entropic regularization BID6 to the optimization problem making it strongly convex.

Its solution can be found using scaling algorithms such as the so called Sinkhorn algorithm.

For any positive matrix γ, the entropy is defined as follows: DISPLAYFORM4 The entropic regularized OT distances in the balanced and unbalanced case become, for a hyperparameter ε > 0: DISPLAYFORM5 for ε → 0, W ε and W unb,ε converge to the original OT distance, and for higher value of ε we obtain the so called Sinkhorn divergence that allows for more diffuse transport between p and q. Balanced and unbalanced W. barycenters can be naturally defined with the entropic regularized OT distance as follows: DISPLAYFORM6 respectively.

This regularization leads to simple iterative algorithms BID2 BID5 ) (for more details we refer the interested reader to BID5 and references therein) for computing W. barycenters that are given in Algorithms 1 and 2.Algorithm 1: Balanced Barycenter for Multiclass Ensembling BID2 Inputs: DISPLAYFORM7 Algorithm 2: Unbalanced Barycenter for Multilabel Ensembling BID5 ) DISPLAYFORM8 We see that the output of Algorithm 1 is the geometric mean of K u , = 1 . . .

m, where K is a Gaussian kernel with bandwidth ε the entropic regularization parameter.

Note v * , = 1 . . .

m the values of v at convergence of Algorithm 1.

The entropic regularized W. barycenter can be written as follows: exp DISPLAYFORM9 .

We see from this that K appears as matrix product multiplying individual models probability µ and the quantities v * related to Lagrange multipliers.

This matrix vector product with K ensures probability mass transfer between semantically related classes i.e between items that has entries K ,ij with high values.

Remark 1 (The case K = K = I).

As the kernel K in Algorithm 1 approaches I (identity) (this happens when ε → 0), the alternating Bregman projection of BID2 for balanced W. barycenter converges to the geometric meanμ g = Π m =1 (µ ) λ .We prove this in Appendix D. When K = I the fixed point of Algorithm 1 reduces to geometric mean, and hence diverges from the W. barycenter.

Note that K approaches identity as ε → 0, and in this case we don't exploit any semantics.

Wasserstein Ensembling in Practice.

Table 1 gives a summary of machine learning tasks that can benefit from Wasserstein Ensembling, and highlights the source and target domains as well as the corresponding kernel matrix K. In the simplest case Ω S = Ω T and N = M for all , this corresponds to the case we discussed in multi-class and multi-labels ensemble learning, W. barycenters allows to balance semantics and confidence in finding the consensus.

The case where source and target spaces are different is also of interest, we give here an application example in attribute based classification : µ corresponds to prediction on a set of attributes and we wish to make predictions through the W. barycenter on a set of labels defined with those attributes.

See Section 6.1.

as we use beam search on the predictions, diversity and smoothness of the predictions become key to the creativity and the composition of the sequence generator in order to go beyond "baby talk" and vanilla language based on high count words in the training set.

Hence we need to increase the entropy of the prediction by finding a semantic consensus whose predictions are diverse and smooth on semantically coherent concepts without compromising accuracy.

We will show in the following proposition that the W. barycenter allows such aggregation: Proposition 1 (Properties of Wasserstein Barycenters).

Let ν be the target distribution (an oracle) defined on a discrete space Ω = {x 1 , . . .

x K , x j ∈ R d } (word embedding space) and µ , = 1 . . .

m be m estimates of ν.

Assume W 2 2 (µ , ν) ≤ ε .

The W. barycenterμ w of {µ } satisfies the following: 1) Semantic Accuracy (Distance to an oracle).

We have: DISPLAYFORM10 2) Diversity.

The diversity of the W. barycenter depends on the diversity of the models with respect to the Wasserstein distance (pairwise Wasserstein distance between models): DISPLAYFORM11 3) Smoothness in the embedding space.

Define the smoothness energy E (ρ) = DISPLAYFORM12 The W. barycenter is smoother in the embedding space than the individual models.

DISPLAYFORM13 Proof.

The proof is given in Appendix F.We see from Proposition 1 that the W. barycenter preserves accuracy, but has a higher entropy than the individual models.

This entropy increase is due to an improved smoothness on the embedding space: words that have similar semantics will have similar probability mass assigned in the barycenter.

The diversity of the barycenter depends on the Wasserstein pairwise distance between the models: the W. barycenter output will be less diverse if the models have similar semantics as measured by the Wasserstein distance.

The proof of proposition 1 relies on the notion of convexity along generalized geodesics of the Wasserstein 2 distance BID0 .

Propositions 2 and 3 in Appendix F give similar results for geometric and arithmetic mean, note that the main difference is that the guarantees are given in terms of KL and 2 respectively, instead of W 2 .In order to illustrate the diversity and smoothness of the W. barycenter, we give here a few examples of the W. barycenter on a vocabulary of size 10000 words, where the cost matrix is constructed from word synonyms ratings, defined using Power Thesaurus or using GloVe word embeddings BID28 .

We compute the W. barycenter (using Algorithm 1) between softmax outputs of 4 image captioners trained with different random seeds and objective functions.

Figure 4 shows the W. barycenter as well as the arithmetic and geometric mean.

It can be seen that the W. barycenter has higher entropy and is smooth along semantics (synonyms or semantics in the GloVe space) and hence more diverse than individual models.

Table 2 shows top 15 words of barycenter, arithmetic and geometric means, from which we see that indeed the W. barycenter outputs clusters according to semantics.

In order to map back the words x j that have high probability in the W. barycenter to an individual model , we can use the couplings γ as follows: γ ij is the coupling between word j in the barycenter and word i in model .

Examples are given in supplement in Table 2 : Sample output (top 15 words) of W. barycenter (Algorithm 1), arithmetic and geometric means based on four captioner models.

Each row shows a word and a corresponding probability over the vocabulary (as a percentage).

W. Barycenter has higher entropy, spreading the probability mass over the synonyms and words related to the top word "car" and downweights the irrelevant objects (exploiting the side information K).

Simple averaging techniques, which use only the confidence information, mimick the original model outputs.

Figure 4 in Appendix gives a histogram view.

Controllable Entropy via Regularization.

As the entropic regularization parameter ε increases the distance of the kernel K from identity I increases and the entropy of the optimal couplings γ , (H(γ )) increases as well.

Hence the entropy of entropic regularized W. Barycenter is controllable via the entropic regularization parameter ε.

In fact since the barycenter can be written asμ w = γ 1 N , one can show that (Lemma 2 in Appendix): DISPLAYFORM14 As epsilon increases the right-hand side of the inequality increases and so does H(μ w ).

This is illustrated in Tables 3 and 8 , we see that the entropy of the (entropic regularized) W. barycenter increases as the distance of the kernel K to identity increases ( K − I F increases as ε increases ) and the output of the W. barycenter remains smooth within semantically coherent clusters.

Table 3 : Controllable Entropy of regularized Wasserstein Barycenter (Algorithm 1).

Output (top 15 words) for a synonyms-based similarity matrix K under different regularization ε (which controls the distance of K to identity I, K − I F ).

As ε decreases, K − I F also decreases, i.e., K approaches identity matrix, and the entropy of the output of Algorithm 1 decreases.

Note that the last column, corresponding to very small entropic regularization, coincides with the output from geometric mean in FIG3 (for K = I, the Algorithm 1 outputs geometric mean as a barycenter).

Wasserstein Barycenters in Machine Learning.

Optimal transport is a relatively new comer to the machine learning community.

The entropic regularization introduced in BID6 fostered many applications and computational developments.

Learning with a Wasserstein loss in a multilabel setting was introduced in BID14 , representation learning via the Wasserstein discriminant analysis followed in BID12 .

More recently a new angle on generative adversarial networks learning with the Wasserstein distance was introduced in (Arjovsky et al., 2017; BID15 BID31 .

Applications in NLP were pioneered by the work on Word Mover Distance (WMD) on word embeddings of BID23 .

Thanks to new algorithmic developments BID7 BID2 W. barycenters have been applied to various problems : in graphics BID35 , in clustering (Ye et al., 2017) , in dictionary learning BID33 , in topic modeling (Xu et al., 2018) , in bayesian averaging BID30 , and in learning word and sentences embeddings BID26 BID27 etc.

Most of these applications of W. barycenter focus on learning balanced barycenters in the embedding space (like learning the means of the clusters in clustering), in our ensembling application we assume the embeddings given to us (such as GloVe word embedding ) and compute the barycenter at the predictions level.

Finally incorporating side information such as knowledge graphs or word embeddings in classification is not new and has been exploited in diverse ways at the level of individual model training via graph neural networks BID25 BID9 , in the framework of W. barycenter we use this side information at the ensemble level.

In this Section we evaluate W. barycenter ensembling in the problems of attribute-based classification, multi-label prediction and in natural language generation in image captioning.

As a first simple problem we study object classification based on attribute predictions.

We use Animals with Attributes (Xian et al., 2017) which has 85 attributes and 50 classes.

We have in our experiments 2 attributes classifiers to predict the absence/presence of each of the 85 attributes independently, based on (1) resnet18 and (2) resnet34 BID18 input features while training only the linear output layer (following the details in Section 6.2).

We split the data randomly in 30322 / 3500 / 3500 images for train / validation / test respectively.

We train the attribute classifiers on the train split.

Based on those two attributes detectors we would like to predict the 50 categories using unbalanced W. barycenters using Algorithm 2.

Note that in this case the source domain is the set of the 85 attributes and the target domain is the set of 50 animal categories.

For Algorithm 2 we use a columnnormalized version of the binary animal/attribute matrix as K matrix (85 × 50), such that per animal the attribute indicators sum to 1.

We selected the hyperparameters ε = 0.3 and λ = 2 on the validation split and report here the accuracies on the test split.

Table 4 : Attribute-based classification.

The W. barycenter ensembling achieves better accuracy by exploiting the cross-domain similarity matrix K, compared to a simple linear-transform of probability mass from one domain to another as for the original models or their simple averages.

As a baseline for comparison, we use arithmetic mean (μ a ) and geometric mean (μ g ) ensembling of the two attribute classifiers resnet18 and resnet34.

Then, using the same matrix K as above, we define the probability of category c (animal) as p(c|µ) = K μ (forμ =μ a andμ g resp.).

We see from Table 4 that W. barycenter outperforms arithmetic and geometric mean on this task and shows its potential in attribute based classification.

For investigating W. barycenters on a multi-label prediction task, we use MS-COCO BID24 with 80 objects categories.

MS-COCO is split into training (≈82K images), test (≈35K), and validation (5K) sets, following the Karpathy splits used in the community BID20 .

From the training data, we build a set of 8 models using 'resnet18' and 'resnet50' architectures BID18 .

To ensure some diversity, we start from pretrained models from either ImageNet BID8 ) or Places365 (Zhou et al., 2017 .

Each model has its last fully-connected ('fc') linear layer replaced by a linear layer allowing for 80 output categories.

All these pretrained models are fine-tuned with some variations: The 'fc' layer is trained for all models, some also fine-tune the rest of the model, while some fine-tune only the 'layer4' of the ResNet architecture.

These variations are summarized in Table 5 .

Training of the 'fc' layer uses a 10 −3 learning rate, while all fine-tunings use 10 −6 learning rate.

All multi-label trainings use ADAM BID21 with (β 1 = 0.9, β 2 = 0.999) for learning rate management and are stopped at 40 epochs.

Only the center crop of 224 * 224 of an input image is used once its largest dimension is resized to 256.

Table 5 : Description of our 8 models built on MS-COCO Evaluation Metric.

We use the mean Average Precision (mAP) which gives the area under the curve of P = f (R) for precision P and recall R, averaged over each class.

mAP performs a sweep of the threshold used for detecting a positive class and captures a broad view of a multi-label predictor performance.

Performances for our 8 models are reported in TAB5 .

Precision, Recall and F1 for micro/macro are given in TAB10 .

Our individual models have reasonable performances overall.

Arithmetic and geometric means offer direct mAP improvements over our 8 individual models.

For unbalanced W. barycenter, the transport of probability mass is completely defined by its matrix K = K in Algorithm 2.

We investigated multiple K matrix candidates by defining K(i, j) as (i) the pairwise GloVe distance between categories, (ii) pairwise visual word2vec embeddings distance, (iii) pairwise co-occurence counts from training data.

In our experience, it is challenging to find a generic K that works well overall.

Indeed, W. barycenter will move mass exactly as directed by K. A generic K from prior knowledge may assign mass to a category that may not be present in some images at test time, and get harshly penalized by our metrics.

A successful approach is to build a diagonal K for each test sample based on the top-N scoring categories from each model and assign the average of model posteriors scores K(i, i) = 1 M m p m (i|x) for image x and category i. If a category is not top scoring, a low K(i, i) = ζ value is assigned to it, diminishing its contribution.

It gives W. barycenter the ability to suppress categories not deemed likely to be present, and reinforce the contributions of categories likely to be.

This simple diagonal K gives our best results when using the top-2 scoring categories per model (the median number of active class in our training data is about 2) and outperforms arithmetic and geometric means as seen in TAB5 .

In all our experiments, W. barycenters parameters {ε, λ} in Algorithm 2 and ζ defined above were tuned on validation set (5K).

We report results on MS-COCO test set (≈35K).

In this task of improving our 8 models, W. barycenter offers a solid alternative to commonly used arithmetic and geometric means.

Appendix B.2 shows that non-uniform weighting further improves W. ensembling performance.

In this task the objective is to find a semantic consensus by ensembling 5 image captioner models.

The base model is an LSTM-based architecture augmented with the attention mechanism over the image.

In this evaluation we selected captioners trained with cross entropy objective as well as GAN-trained models BID10 .

The training was done on COCO dataset BID24 using data splits from BID19 : training set of 113k images with 5 captions each, 5k validation set, and 5k test set.

The size of the vocabulary size is 10096 after pruning words with counts less than 5.

The matrix K = K in Algorithm 1 was constructed using word similarities, defined based on (i) GloVe word embeddings, so that K = exp(−C/ε), where cost matrix C is constructed based on euclidean distance between normalized embedding vectors; and (ii) synonym relationships, where we created K based on the word synonyms graph and user votes from Power Thesaurus.

The model prediction µ , for = 1, . . .

, 5 was selected as the softmax output of the captioner's LSTM at the current time step, and each model's input was weighted equally: λ = 1/m.

Once the barycenter p was computed, the result was fed into a beam search (beam size B = 5), whose output, in turn, was then given to the captioner's LSTM and the process continued until a stop symbol (EOS) was generated.

In order to exploit the controllable entropy of W. barycenter via the entropic regualrization parameter ε, we also decode using randomized Beam search of BID34 , where instead of maintaining the top k values, we sample D candidates in each beam.

The smoothness of the barycenter in semantic clusters and its controllable entropy promotes diversity in the resulting captions.

We baseline the W. barycenter ensembling with arithmetic and geometric means. .

The x-axis shows K − I F , which corresponds to a different regularization parameter ε (varied form 1 to 50).

We can see that for topK beam search (left panel) the further K is from the identity matrix, the larger the similarity neighborhood of each word, the more diverse are the generated captions (the barycenter has higher entropy), while still remaining semantically close to the ground truth.

On the other hand, for randomized beam search (right panel), it is important to maintain a smaller similarity neighborhood, so that the generated sentences are not too different from the referenced ground truth.

Controllable entropy and diversity.

FIG0 show the comparison of the ensembling methods on the validation set using topK and randomized beam search.

The x-axis shows K −I F , which corresponds to a different regularization ε (varied form 1 to 50).

We report two n-gram based metrics: CIDEr and SPICE scores, as well as the WMD (Word Mover Distance) similarity BID23 , which computes the earth mover distance (the Wasserstein distance) between the generated and the ground truth captions using the GloVe word embedding vectors.

In topK beam search, as ε increases, causing the entropy to go up, the exact n-grams matching metrics, i.e., CIDEr and SPICE, deteriorate while WMD remains stable.

This indicates that while the barycenter-based generated sentences do not match exactly the ground truth, they still remain semantically close to it (by paraphrasing), as indicated by the stability of WMD similarity.

The results of the GloVe-based barycenter on the test split of COCO dataset are shown in Table 7 .

In randomized beam search, the increase in entropy of the barycenter leads to a similar effect of paraphrasing but this works only up to a smaller value of ε, beyond which we observe a significant deterioration of the results.

At that point all the words become neighbors and result in a very diffused barycenter, close to a uniform distribution.

This diffusion effect is smaller for the synonyms-based K since there are only a certain number of synonyms for each word, thus the maximum neighborhood is limited.

Table 7 : Performance of GloVe-based W. barycenter on COCO test split using topK beam search versus Geometric and Arithmetic ensembling.

While the generated sentences based on W. barycenter do not match exactly the ground truth (lower CIDEr), they remain semantically close to it, while being more diverse (e.g., paraphrased) as indicated by the higher entropy and stable WMD.Robustness of W. Barycenter to Semantic Perturbations.

Finally, the right panel of FIG3 , shows the robustness of the W. barycenter to random shuffling of the µ values, within semantically coherent clusters.

Note that the size of those clusters increases as K moves away from identity.

The results show that barycenter is able to recover from those perturbations, employing the side information from K, while both the arithmetic and geometric means (devoid of such information) are confused by this shuffling, displaying a significant drop in the evaluation metrics.

Comparison of ensembling methods when the predictions of the input models are shuffled according to the neighborhood structure defined by K. It can be seen that the W. Barycenter ensembling is able to recover from the word shuffling and produce better captions then the simple averaging methods, which are not able to exploit the provided side information.

Human Evaluation.

We performed human evaluation on Amazon MTurk on a challenging set of images out of context of MS-COCO BID10 .

We compared three ensembling techniques: arithmetic, geometric and W. barycenter.

For W. barycenter we used the similarity matrix K defined by visual word2vec BID22 .

For the three models we use randomized beam search.

We asked MTurkers to give a score for each caption on a scale 1-5 and choose the best captions based on correctness and detailedness.

Captions examples are given in Fig. 6 (Appendix).

FIG4 shows that W. barycenter has an advantage over the basic competing ensembling techniques.

We showed in this paper that W. barycenters are effective in model ensembling in machine learning.

In the unbalanced case we showed their effectiveness in attribute based classification, as well as in improving the accuracy of multi-label classification.

In the balanced case, we showed that they promote diversity and improve natural language generation by incorporating the knowledge of synonyms or word embeddings.

Table 8 : Sample output (top 20 words) of barycenter for different similarity matrices K based on GloVe (columns titles denote the distance of K from identity K − I F and corresponding .).

Each column shows a word and its corresponding probability over the vocabulary.

Note that the last column coincides with the output from geometric mean.

Table 8 shows the effect of entropic regularization ε on the resulting distribution of the words of W. barycenter using GloVe embedding matrix.

As K moves closer to the identity matrix, the entropy of barycenter decreases, leading to outputs that are close/identical to the geometric mean.

On the other hand, with a large entropic regularization, matrix K moves away from identity, becoming an uninformative matrix of all 1's.

This eventually leads to a uniform distribution which spreads the probability mass equally across all the words.

This can be also visualized with a histogram in Figure 5 , where the histograms on the bottom represent distributions that are close to uniform, which can be considered as failure cases of W. barycenter, since the image captioner in this case can only generate meaningless, gibberish captions.

In TAB1 we show a mapping from a few top words in the barycenter output (for similarity matrix K based on synonyms) to the input models.

In other words, each column defines the words in the input models which have the greatest influence on each of top 3 words in the barycenter output.

In Figure 6 we present a few captioning examples showing qualitative difference between the considered ensembling techniques.

Figure 4: Visualization of the word distributions of W. barycenter, arithmetic and geometric means based on four captioning models, whose input image is shown on top (one of the ground-truth human-annotated captions for this image reads: A police car next to a pickup truck at an intersection).

The captioner generates a sentence as a sequence of words, where at each step the output is a distribution over the whole vocabulary.

The top four histograms show a distribution over the vocabulary from each of the model at time t = 3 during the sentence generation process.

The bottom three histograms show the resulting distribution over the vocabulary for the ensembles based on W. Barycenter, arithmetic and geometric means.

It can be seen that the W. Barycenter produces high entropy distribution, spreading the probability mass over the synonyms of the word "car" (which is the top word in all the four models), based on the synonyms similarity matrix K.Figure 5: Visualization of the word distributions of W. barycenter for different similarity matrices K based on GloVe (rows denote the distance of K from identity K − I F and corresponding ).

Large entropic regularization generates K close to uninformative matrices of all 1's.

This eventually leads to a barycenter which is close to a uniform distribution spreading the probability mass almost equally across all the words.

TAB1 : Mapping from a few top words in the barycenter output (for similarity matrix K based on synonyms) to the input models.

For each word in the left columns, the remaining columns show the contributing words and the percent of contribution.

BA: a television is placed on the curb of the road AM: a TV sits on the side of a street GM: a television sitting on the side of a street GT: an empty sidewalk with an abandoned television sitting alone BA: a car that is parked at the station AM: a car that has been shown in a subway GM: a car that is sitting on the side of a road GT: a car at the bottom of the stair well BA: a person is sitting on the sidewalk with a tent AM: a couple of people sitting on benches next to a building GM: a couple of people sitting on the side of a street GT: a woman is sitting with a guitar near a man that is sitting on the ground in front of a tent BA: a sheep sitting in a car looking out the window AM: a white sheep is sitting in a vehicle GM: a close up of a sheep in a car GT: a sheep sitting at the steering wheel of a car with its hooves on the wheels Figure 6 : Examples of captions for several images.

BA: Wasserstein Barycenter, AM: Arithmetic mean, GM: Geometric mean, GT: Ground truth.

We evaluate our models using micro and macro versions of precision, recall, and F1-measure as covered in multi-label prediction metrics study from (Wu & Zhou, 2016) .

For these measures, a threshold of 0.5 is commonly used to predict a label as positive in the community's published results.

Macro precision is an average of per-class precisions while micro precision is computed by computing the ratio of all true positives across all image samples over the number of all positive classes in a dataset.

Therefore a macro (or per-class) precision 'P-C' is defined as 1 C i P i while a micro (or overall precision) 'P-O' is defined as i T Pi i T Pi+F Pi where T P i and F P i are true and false positives respectively.

Per-class and overall versions for R and F1 are defined similarly.

We also employ mean Average Precision (mAP) which gives the area under the curve of P = f (R) averaged over each class.

Unlike P,R and F1, mAP inherently performs a sweep of the threshold used for detecting a positive class and captures a broader view of a multi-label predictor's performance.

Performances for our 8 models and previously published results are reported in TAB5 in the paper.

Our models have reasonable performances overall.

Ensembling results given in Tab.

6 are using uniformly weighted models, i.e. λ = 1 m where m is the number of models.

However, in practice, arithmetic and geometric mean ensembling usually use weighted ensembles of models The weights are then optimized and established on a small validation set before being used for ensembling on a test set.

A well-known embodiment of this type of approach is Adaboost BID13 where weights are dynamically defined at each pass of training wrt to the accuracy of base models.

Here, we follow a much simpler but similar approach by defining the performance of each model as the mean average precision (mAP ) on the validation set.

mAP is used to define λ such that λ = mAP mAP .

λ are then applied to the models' scores for arithmetic, geometric mean and W.Barycenter ensemblings.

Tab.

11 reports mAP for each ensembling technique over the MS-COCO test set (35150 images).

Note that the λ weights definition is based on the final metric evaluation, mAP in this case.

For other tasks such as classification, accuracy or any other valid metric can be employed to compute the λ weights.

It must be noted that the weights are computed with respect to the ultimate performance metric at hand.

Tab.

11 reveals clearly that such approach of weighting models by their performance benefits arithmetic and W.Barycenter ensembling for this task.

Both methods leverage the confidence of the underlying models and the mAP weighting of models will reinforce the contributions of better performing models.

Geometric means ensembling is not significantly impacted by non-uniform λ since it is mostly relying on consensus of the models, not their confidence.

We conclude that weighting indeed helps performance and keeps a performance advantage for W. Barycenter over the alternatives arithmetic and geometric means.

Table 11 : multi-label models ensembling mAP on MS-COCO test set (35150 images).

Performancebased weighting helps both arithmetic and W.Barycenter ensembling, the latter retaining its performance vantage.

At iteration 0, the result holds since we have : DISPLAYFORM0 Assume the result holds at time t.

Let us prove it for t + 1, following the updates of Algorithm 1 : DISPLAYFORM1 QED.For ε > 0, Feydy et al BID11 showed recently that the Sinkhorn divergence defines an interpolation between the MMD distance (Maximum mean discrepancy BID16 ) and the Wasserstein Distance.

Hence for ε > 0 Algorithm 1 provides still an interesting solution that can be seen as an interpolation between the original (unregularized) Wasserstein Barycenter and the MMD Barycenter (Frechet barycenter for d = MMD).

DISPLAYFORM2 As λ goes to infinity this unbalanced cost converges to the Hellinger distance: We make the following theoretical and practical remarks on how to improve this computational complexity to reach an almost linear dependency on N using low rank approximation of the kernel matrix K, and parallelization on m machines: DISPLAYFORM3 1.

Dependency on Maxiter:

For the number of iterations we found that Maxiter = 5 is enough for convergence, which makes most of the computational complexity dependent on m and N .2.

Dependency on N and low rank approximation : The main computational complexity comes from the matrix vector multiply K u that is of O(N 2 ).

Note that this complexity can be further reduced since the kernel matrix K is often low rank.

Therefore we can be written K = ΦΦ where Φ ∈ R N ×k , where k N , which allows to compute this product as follows ΦΦ u that has a lower complexity O(N k).

Φ can be computed using Nystrom approximation or random Fourier features.

Hence potentially on can get an algorithm with complexity O(mN k), where k has a logarithmic dependency on N .

This was studied recently in BID1 .3.

Dependency on m and parallelization: Regarding the dependency on m, as noted in BID5 BID2 the algorithm is fully parallelizable which would lead to a computational complexity of O(N k) by using simply m machines .4.

GPU and Batch version: Practically, the algorithm implemented takes advantage of matrix vector products' speed on GPU.

The algorithm can be further accelerated by computing Sinkhorn divergences in batches as pointed in BID11 ).

We evaluated the time complexity of the GPU implementation of Wasserstein Barycenters in pytorch on our multi-label prediction experiments using MS-COCO test set (35150 samples).

Note that we used a vanilla implementation of Algorithm 2, i.e without parallelization, batching, or low rank approximation.

Results and comments for these wall clock timings can be found in Tab.

13.

As it can be observed, we need to use Maxiter = 5 on a GPU-V100 to reach below 4ms/image for Wasserstein ensembling.

This is not a major overhead and can be further improved as discussed previously by using parallelization, batching and low rank approximation.

In TAB14 , each timing was done over the whole test set; each timing repeated 5 times.

We report means and standard deviations of total wall clock times for ensembling 8 models.

Last column on the right is the average timing per image (in ms) for W.Barycenter.

The number of W.Barycenter iterations (Maxiter) was varied from 1, 5 and 10 to show its impact.

We report timing numbers over two GPU architectures, NVIDIA Tesla K80 and V100.

W.Barycenters leverage the GPU while Arithmetic and Geometric means do not.

Timings are of the computations of the means themselves, no data fetching or data preparation is included in these timings.

As expected, the wall clock time cost for W.Barycenters is several order of magnitude higher than for Arithmetic and Geometric means.

The difference of GPU does not impact the Arithmetic and Geometric means as they do not use it in our implementation.

The Barycenter computation see a speed up from K80 to V100 as V100 is much better at reducing wall time for longer number of iterations.

Proposition 2 (propreties of Geometric Mean).

The following properties hold for geometric mean:1.

Geometric mean is the Frechet mean of KL.

The geometric mean is the Frechet mean with respect to the extended KL divergence: First order optimality condition: DISPLAYFORM0 DISPLAYFORM1 This gives us the result: DISPLAYFORM2 Published as a conference paper at ICLR 2019Proof.

Let γ ∈ R N ×M + be a coupling between p ∈ ∆ M and q ∈ ∆ N , q j > 0 we have: γ 1 N = p and γ1 M = q, we have: DISPLAYFORM3 DISPLAYFORM4 Now the entropy of the convex combination is higher than convex combination of entropies (the entropy is concave): DISPLAYFORM5 γ ij log(γ ij ) − log(q i )q i .Hence : DISPLAYFORM6 γ ij log(γ ij ) − H(q)Hence : DISPLAYFORM7 γ ij log(γ ij )

<|TLDR|>

@highlight

we propose to use Wasserstein barycenters for semantic model ensembling