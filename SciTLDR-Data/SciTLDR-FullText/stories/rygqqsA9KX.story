Learning multimodal representations is a fundamentally complex research problem due to the presence of multiple heterogeneous sources of information.

Although the presence of multiple modalities provides additional valuable information, there are two key challenges to address when learning from multimodal data: 1) models must learn the complex intra-modal and cross-modal interactions for prediction and 2) models must be robust to unexpected missing or noisy modalities during testing.

In this paper, we propose to optimize for a joint generative-discriminative objective across multimodal data and labels.

We introduce a model that factorizes representations into two sets of independent factors: multimodal discriminative and modality-specific generative factors.

Multimodal discriminative factors are shared across all modalities and contain joint multimodal features required for discriminative tasks such as sentiment prediction.

Modality-specific generative factors are unique for each modality and contain the information required for generating data.

Experimental results show that our model is able to learn meaningful multimodal representations that achieve state-of-the-art or competitive performance on six multimodal datasets.

Our model demonstrates flexible generative capabilities by conditioning on independent factors and can reconstruct missing modalities without significantly impacting performance.

Lastly, we interpret our factorized representations to understand the interactions that influence multimodal learning.

Multimodal machine learning involves learning from data across multiple modalities .

It is a challenging yet crucial research area with real-world applications in robotics BID13 , dialogue systems BID26 , intelligent tutoring systems BID24 , and healthcare diagnosis (Frantzidis et al., 2010) .

At the heart of many multimodal modeling tasks lies the challenge of learning rich representations from multiple modalities.

For example, analyzing multimedia content requires learning multimodal representations across the language, visual, and acoustic modalities BID9 .

Although the presence of multiple modalities provides additional valuable information, there are two key challenges to address when learning from multimodal data: 1) models must learn the complex intra-modal and cross-modal interactions for prediction , and 2) trained models must be robust to unexpected missing or noisy modalities during testing BID19 .In this paper, we propose to optimize for a joint generative-discriminative objective across multimodal data and labels.

The discriminative objective ensures that the representations learned are rich in intra-modal and cross-modal features useful towards predicting the label, while the generative objective allows the model to infer missing modalities at test time and deal with the presence of noisy modalities.

To this end, we introduce the Multimodal Factorization Model (MFM in FIG0 ) that factorizes multimodal representations into multimodal discriminative factors and modality-specific generative factors.

Multimodal discriminative factors are shared across all modalities and contain joint multimodal features required for discriminative tasks.

Modality-specific generative factors are unique for each modality and contain the information required for generating each modality.

We believe that factorizing multimodal representations into different explanatory factors can help each factor focus on learning from a subset of the joint information across multimodal data and labels.

This method is in contrast to jointly learning a single factor that summarizes all generative and discriminative information BID37 .

To sum up, MFM defines a joint distribution over multimodal data, and by the conditional independence assumptions in the assumed graphical model, both generative and discriminative aspects are taken into account.

Our model design further provides interpretability of the factorized representations.

Through an extensive set of experiments, we show that MFM learns improved multimodal representations with these characteristics: 1) The multimodal discriminative factors achieve state-of-the-art or competitive performance on six multimodal time series datasets.

We also demonstrate that MFM can generalize by integrating it with other existing multimodal discriminative models.

2) MFM allows flexible generation concerning multimodal discriminative factors (labels) and modality-specific generative factors (styles).

We further show that we can perform reconstruction of missing modalities from observed modalities without significantly impacting discriminative performance.

Finally, we interpret our learned representations using information-based and gradient-based methods, allowing us to understand the contributions of individual factors towards multimodal prediction and generation.

Multimodal Factorization Model (MFM) is a latent variable model ( FIG0 (a)) with conditional independence assumptions over multimodal discriminative factors and modality-specific generative factors.

According to these assumptions, we propose a factorization over the joint distribution of multimodal data (Section 2.1).

Since exact posterior inference on this factorized distribution can be intractable, we propose an approximate inference algorithm based on minimizing a joint-distribution Wasserstein distance over multimodal data (Section 2.2) .

Finally, we derive the MFM objective by approximating the joint-distribution Wasserstein distance via a generalized mean-field assumption.

Notation: We define X 1???M as the multimodal data from M modalities and Y as the labels, with joint distribution P X 1???M ,Y = P (X 1???M , Y).

LetX 1???M denote the generated multimodal data and?? denote the generated labels, with joint distribution PX 1???M ,?? = P (X 1???M ,??).

To factorize multimodal representations into multimodal discriminative factors and modality-specific generative factors, MFM assumes a Bayesian network structure as shown in FIG0 (a) .

In this graphical model, factors F y and F a{1???M } are generated from mutually independent latent variables Z = [Z y , Z a{1???M } ] with prior P Z .

In particular, Z y generates the multimodal discriminative factor F y and Z a{1???M } generate modality-specific generative factors F a{1???M } .

By construction, F y contributes to the generation of?? while {F y , F ai } both contribute to the generation ofX i .

As a result, the joint distribution P (X 1???M ,??) can be factorized as follows: DISPLAYFORM0 DISPLAYFORM1 Exact posterior inference in Equation 1 may be analytically intractable due to the integration over Z. We therefore resort to using an approximate inference distribution Q(Z X 1???M , Y) as detailed in the following subsection.

As a result, MFM can be viewed as an autoencoding structure that consists of encoder (inference) and decoder (generative) modules FIG0 ).

The encoder module for Q(??? ???) allows us to easily sample Z from an approximate posterior.

The decoder modules are parametrized according to the factorization of P (X 1???M ,?? Z) as given by Equation 1 and FIG0 (a).

Two common choices for approximate inference in autoencoding structures are Variational Autoencoders (VAEs) (Kingma & Welling, 2013) and Wasserstein Autoencoders (WAEs) (Zhao et al., 2017; BID41 .

The former optimizes the evidence lower bound objective (ELBO), and the latter derives an approximation for the primal form of the Wasserstein distance.

We consider the latter since it simultaneously results in better latent factor disentanglement (Zhao et al., 2017; BID31 and better sample generation quality than its counterparts BID6 Higgins et al., 2016; Kingma & Welling, 2013) .

However, WAEs are designed for unimodal data and do not consider factorized distributions over latent variables that generate multimodal data.

Therefore, we propose a variant for handling factorized joint distributions over multimodal data.

As suggested by Kingma & Welling (2013) , we adopt the design of nonlinear mappings (i.e. neural network architectures) in the encoder and decoder FIG0 ).

For the encoder Q(Z X 1???M , Y), we learn a deterministic mapping BID41 .

For the decoder, we define the generation process from latent variables as DISPLAYFORM0 DISPLAYFORM1 , where G y , G a{1???M } , D and F 1???M are deterministic functions parametrized by neural networks.

DISPLAYFORM2 1???M ,?? ) denote the joint-distribution Wasserstein distance over multimodal data under cost function c X i and c Y .

We choose the squared cost c(a, b) = a ??? b 2 2 , allowing us to minimize the 2-Wasserstein distance.

The cost function can be defined not only on static data but also on time series data such as text, audio and videos.

For example, given time series data DISPLAYFORM3 where P Z is the prior over Z = [Z y , Z a{1,M } ] and Q Z is the aggregated posterior of the proposed approximate inference distribution Q(Z X 1???M , Y).Proof: The proof is adapted from Tolstikhin et al. BID41 .

The two differences are: (1) we show that P (X 1???M ,?? Z = z) are Dirac for all z ??? Z, and (2) we use the fact that c(( DISPLAYFORM4 .

Please refer to the supplementary material for proof details.

??? The constraint on Q Z = P Z in Proposition 1 is hard to satisfy.

To obtain a numerical solution, we first relax the constraint by performing a generalized mean field assumption on Q according to the conditional independence as shown in the inference network of FIG0 (b): DISPLAYFORM5 The intuition here is based on our design that Z y generates the multimodal discriminative factor F y and Z a{1???M } generate modality-specific generative factors F a{1???M } .

Therefore, the inference for Z y should depend on all modalities X 1???M and the inference for Z ai should depend only on the specific modality X i .

Following this assumption, we define Q as a nonparametric set of all encoders that fulfill the factorization in Equation 3 .

A penalty term is added into our objective to find the Q(Z ???) ??? Q that is the closest to prior P Z , thereby approximately enforcing the constraint Q Z = P Z : where ?? is a hyper-parameter and MMD is the Maximum Mean Discrepancy (Gretton et al., 2012) as a divergence measure between Q Z and P Z .

The prior P Z is chosen as a centered isotropic Gaussian N (0, I), so that it implicitly enforces independence between the latent variables Kingma & Welling, 2013; BID31 .

Equation 4 represents our hybrid generative-discriminative optimization objective over multimodal data: the first loss term DISPLAYFORM6 DISPLAYFORM7 DISPLAYFORM8 ) is the generative objective based on reconstruction of multimodal data and the second term c Y (Y, D(G y (Z y ))) is the discriminative objective.

In practice we compute the expectations in Equation 4 using empirical estimates over the training data.

The neural architecture of MFM is illustrated in FIG0 (c).

A key challenge in multimodal learning involves dealing with missing modalities.

A good multimodal model should be able to infer the missing modality conditioned on the observed modalities and perform predictions based only on the observed modalities.

To achieve this objective, the inference process of MFM can be easily adapted using a surrogate inference network to reconstruct the missing modality given the observed modalities.

Formally, let ?? denote the surrogate inference network.

The generation of missing modalityX 1 given the observed modalities X 2???M can be formulated as DISPLAYFORM0 Similar to Section 2.2, we use deterministic mappings in Q ?? (??? ???) and Q ?? (Z y ???) is also used for prediction DISPLAYFORM1 suggests that in the presence of missing modalities, we only need to infer the latent codes rather than the entire modality.

We now discuss the implementation choices for the MFM neural architecture in FIG0 (c).

The encoder Q(Z y X 1???M ) can be parametrized by any model that performs multimodal fusion BID17 .

For multimodal image datasets, we adopt Convolutional Neural Networks (CNNs) and Fully-Connected Neural Networks (FCNNs) with late fusion BID20 as our encoder Q(Z y X 1???M ).

The remaining functions in MFM are also parametrized by CNNs and FCNNs.

For multimodal time series datasets, we choose the Memory Fusion Network (MFN) BID50 as our multimodal encoder Q(Z y X 1???M ).

We use Long Short-term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997) for functions Q(Z a{1???M } X 1???M ), decoder LSTM networks BID8 for functions F 1???M , and FCNNs for functions G y , G a{1???M } and D. Details are provided in the appendix and the code is available at <anonymous>.

In order to show that MFM learns multimodal representations that are discriminative, generative and interpretable, we design the following experiments.

We begin with a multimodal synthetic image dataset that allows us to examine whether MFM displays discriminative and generative capabilities from factorized latent variables.

Utilizing image datasets allows us to clearly visualize the generative capabilities of MFM.

We then transition to six more challenging real-world multimodal video datasets DISPLAYFORM0 DISPLAYFORM1 DISPLAYFORM2 to 1) rigorously evaluate the discriminative capabilities of MFM in comparison with existing baselines, 2) analyze the importance of each design component through ablation studies, 3) assess the robustness of MFM's modality reconstruction and prediction capabilities to missing modalities, and 4) interpret the learned representations using information-based and gradient-based methods to understand the contributions of individual factors towards multimodal prediction and generation.

In this section, we study MFM on a synthetic image dataset that considers SVHN BID18 and MNIST (Lecun et al., 1998) as the two modalities.

SVHN and MNIST are images with different styles but the same labels (digits 0 ??? 9).

We randomly pair 100, 000 SVHN and MNIST images that have the same label, creating a multimodal dataset which we call SVHN+MNIST.

80, 000 pairs are used for training and the rest for testing.

To justify that MFM is able to learn improved multimodal representations, we show both classification and generation results on SVHN+MNIST in FIG1 .Prediction: We perform experiments on both unimodal and multimodal classification tasks.

UM denotes a unimodal baseline that performs prediction given only one modality as input and MM denotes a multimodal discriminative baseline that performs prediction given both images BID20 .

We compare the results for UM(SVHN), UM(MNIST), MM and MFM on SVHN+MNIST in FIG1 (b).

We achieve better classification performance from unimodal to multimodal which is not surprising since more information is given.

More importantly, MFM outperforms MM, which suggests that MFM learns improved factorized representations for discriminative tasks.

Generation: We generate images using the MFM generative network FIG1 (a)).

We fix one variable out of Z = [Z a1 , Z a2 , and Z y ] and randomly sample the other two variables from prior P Z .

From FIG1 (c), we observe that MFM shows flexible generation of SVHN and MNIST images based on labels and styles.

This suggests that MFM is able to factorize multimodal representations into multimodal discriminative factors (labels) and modality-specific generative factors (styles).

In this section, we transition to more challenging multimodal time series datasets.

All the datasets consist of monologue videos.

Features are extracted from the language (GloVe word embeddings (Pen- The videos are divided into multiple segments each annotated for the presence of 6 discrete emotions (happy, sad, angry, frustrated, excited and neutral), resulting in a total of 7318 segments in the dataset.

We report results using the following metrics: Acc_C = multiclass accuracy across C classes, F1 = F1 score, MAE = Mean Absolute Error, r = Pearson's correlation.

DISPLAYFORM0 Prediction: We first compare the performance of MFM with existing multimodal prediction methods.

For a detailed description of the baselines, please refer to the appendix.

From TAB0 , we first observe that the best performing baseline results are achieved by different models across different datasets (most notably MFN, MARN, and TFN).

On the other hand, MFM consistently achieves state-of-the-art or competitive results for all six multimodal datasets.

We believe that the multimodal discriminative factor F y in MFM has successfully learned more meaningful representations by distilling discriminative features.

This highlights the benefit of learning factorized multimodal representations towards discriminative tasks.

Furthermore, MFM is model-agnostic and can be applied to other multimodal encoders Q(Z y X 1???M ).

We perform experiments to show consistent improvements in discriminative performance for several choices of the encoder: EF-LSTM BID17 and TFN .

For Acc_2 on CMU-MOSI, our factorization framework improves the performance of EF-LSTM from 74.3 to 75.2 and TFN from 74.6 to 75.5.Ablation Study: In FIG2 , we present the models M {A,B,C,D,E} used for ablation studies.

These models are designed to analyze the effects of using a multimodal discriminative factor, a hybrid generative-discriminative objective, factorized generative-discriminative factors and modality-specific generative factors towards both modality reconstruction and label prediction.

The simplest variant is M A which represents a purely discriminative model without a joint multimodal discriminative factor (i.e. early fusion BID17 .

M B models a joint multimodal discriminative factor which incorporates more general multimodal fusion encoders BID50 .

M C extends M A by optimizing a hybrid generative-discriminative objective over modality-specific factors.

M D extends M B by optimizing a hybrid generative-discriminative objective over a joint multimodal factor (resembling BID37 ).

M E factorizes the representation into separate generative and discriminative factors.

Finally, MFM is obtained from M E by using modality-specific generative factors instead of a joint multimodal generative factor.

From the table in FIG2 , we observe the following general trends.

For sentiment prediction, using 1) a multimodal discriminative factor outperforms modality-specific discriminative factors DISPLAYFORM1 , and 2) adding generative capabilities to the model improves performance DISPLAYFORM2 .

For both sentiment prediction and modality reconstruction, 3) factorizing into separate generative and discriminative factors improves performance (M E > M D ), and 4) using modality-specific generative factors outperforms multimodal generative factors (MFM > M E ).

These observations support our design decisions of factorizing multimodal representations into multimodal discriminative factors and modality-specific generative factors.

Missing Modalities: We now evaluate the performance of MFM in the presence of missing modalities using the surrogate inference model as described in Subsection 2.3.

We compare with two baselines: 1) a purely generative Seq2Seq model BID8 ) ?? G from observed modalities to missing modalities by optimizing E P X 1???M (???log P ?? D (X 1 X 2???M )), and 2) a purely discriminative model ?? D from observed modalities to the label by optimizing DISPLAYFORM3 ).

Both models are modified from MFM by using only the two observed modalities as input and not explicitly accounting for missing modalities.

We compare the reconstruction error of each modality (language, visual and acoustic) as well as the performance on sentiment prediction.

TAB4 shows that MFM with missing modalities outperforms the generative (?? G ) or discriminative baselines (?? D ) in terms of modality reconstruction and sentiment prediction.

Additionally, MFM with missing modalities performs close to MFM with all modalities observed.

This fact indicates that MFM can learn representations that are relatively robust to missing modalities.

In addition, discriminative performance is most affected when the language modality is missing, which is consistent with prior work which indicates that language is most informative in human multimodal language .

On the other hand, sentiment prediction is more robust to missing acoustic and visual features.

Finally, we observe that reconstructing the low-level acoustic and visual features is easier as compared to the high-dimensional language features that contain high-level semantic meaning.

We devise two methods to study how individual factors in MFM influence the dynamics of multimodal prediction and generation.

These interpretation methods represent both overall trends and fine-grained analysis that could be useful towards deeper understandings of multimodal representation learning.

For more details, please refer to the appendix.

Firstly, an information-based interpretation method is chosen to summarize the contribution of each modality towards the multimodal representations.

Since F y is a common cause ofX 1???M , we can compare MI(F y ,X 1 ), ???, MI(F y ,X M ), where MI(???, ???) denotes the mutual information measure between F y and generated modalityX i .

Higher MI(F y ,X i ) indicates greater contribution from F y toX i .

FIG3 reports the ratios r i = MI(F y ,X i ) MI(F ai ,X i ) which measure a normalized version of the mutual information between F ai andX i .

We observe that on CMU-MOSI, the language modality is most informative towards sentiment prediction, followed by the acoustic modality.

We believe that this result represents a prior over the expression of sentiment in human multimodal language and is closely related to the connections between language and speech (Kuhl, 2000) .Secondly, a gradient-based interpretation method to used analyze the contribution of each modality for every time step in multimodal time series data.

We measure the gradient of the generated modality with respect to the target factors (e.g., F y ).

Let {x 1 , x 2 , ???, x M } denote multimodal time series data where x i represents modality i, DISPLAYFORM0 ] denote generated modality i across time steps t ??? [1, T ].

The gradient ??? fy (x i ) measures the extent to which changes in factor f y ??? P (F y X 1???M = x 1???M ) influences the generation of sequencex i .

FIG3 plots ??? fy (x i ) for a video in CMU-MOSI.

We observe that multimodal communicative behaviors that are indicative of speaker sentiment such as positive words (e.g. "very profound and deep") and informative acoustic features (e.g. hesitant and emphasized tone of voice) indeed correspond to increases in ??? fy (x i ).

The two main pillars of research in multimodal representation learning have considered the discriminative and generative objectives individually.

Discriminative representation learning BID4 Frome et al., 2013; BID33 ) models the conditional distribution P (Y X 1???M ).

Since these approaches are not concerned with modeling P (X 1???M ) explicitly, they use parameters more efficiently to model P (Y X 1???M ).

For instance, recent works learn visual representations that are maximally dependent with linguistic attributes for improving one-shot image recognition or introduce tensor product mechanisms to model interactions between the language, visual and acoustic modalities BID14 .

On the other hand, generative representation learning captures the interactions between modalities by modeling the joint distribution P (X 1 , ???, X M ) using either undirected graphical models BID37 , directed graphical models BID40 , or neural networks .

Some generative approaches compress multimodal data into lower-dimensional feature vectors which can be used for discriminative tasks BID25 BID19 .

To unify the advantages of both approaches, MFM factorizes multimodal representations into generative and discriminative components and optimizes for a joint objective.

Factorized representation learning resembles learning disentangled data representations which have been shown to improve the performance on many tasks (Kulkarni et al., 2015; Lake et al., 2017; Higgins et al., 2016; BID1 .

Several methods involve specifying a fixed set of latent attributes that individually control particular variations of data and performing supervised training BID7 Karaletsos et al., 2015; BID46 BID30 Zhu et al., 2014) , assuming an isotropic Gaussian prior over latent variables to learn disentangled generative representations (Kingma & Welling, 2013; BID31 and learning latent variables in charge of specific variations in the data by maximizing the mutual information between a subset of latent variables and the data BID6 .

However, these methods study factorization of a single modality.

MFM factorizes multimodal representations and demonstrates the importance of modality-specific and multimodal factors towards generation and prediction.

A concurrent and parallel work that factorizes latent factors in multimodal data was proposed by Hsu & Glass (2018) .

They differ from us in the graphical model design, discriminative objective, prior matching criterion, and scale of experiments.

We provide a detailed comparison with their model in the appendix.

In this paper, we proposed the Multimodal Factorization Model (MFM) for multimodal representation learning.

MFM factorizes the multimodal representations into two sets of independent factors: multimodal discriminative factors and modality-specific generative factors.

The multimodal discriminative factor achieves state-of-the-art or competitive results on six multimodal datasets.

The modalityspecific generative factors allow us to generate data based on factorized variables, account for missing modalities, and have a deeper understanding of the interactions involved in multimodal learning.

Our future work will explore extensions of MFM for video generation, semi-supervised learning, and unsupervised learning.

We believe that MFM sheds light on the advantages of learning factorizing multimodal representations and potentially opens up new horizons for multimodal machine learning.

To simplify the proof, we first prove it for the unimodal case by considering the Wasserstein distance between P X,Y and PX ,?? .

where W c is the Wasserstein distance under cost function c X and c Y , P Z is the prior over Z = [Z a , Z y ] and Q Z is the aggregated posterior of the proposed inference distribution Q(Z X).Proof: See the following.

To begin the proof, we abuse some notations as follows.

By definition, the Wasserstein distance under cost function c between P X,Y and PX ,?? is DISPLAYFORM0 where We now introduce two Lemmas to help the proof.

DISPLAYFORM1 DISPLAYFORM2 Since the functions F, G a , G y , D are all deterministic, then P (X,?? Z) are Dirac measures.

??? Lemma 2.

P P X,Y , PX ,?? = P X,Y,X,?? when P (X,?? Z = z) are Dirac for all z ??? Z.Proof: WhenX,?? are deterministic functions of Z, for any A in the sigma-algebra induced byX,??, we have DISPLAYFORM3 Therefore, this implies that (X, Y) ??? ??? (X,??) Z which concludes the proof.

A similar argument is made in Lemma 1 of BID41 .???

Now, we use the fact that P P X,Y , PX ,?? = P X,Y,X,?? (Lemma 1 + Lemma 2), DISPLAYFORM4 Note that in Eq. equation 8, P X,Y,Z = P (X, Y) ??? P X,Y , Z ??? P Z and with a proposed Q(Z X), we can rewrite Eq. equation 8 as DISPLAYFORM5 The proof is similar to Proposition 2, and we present a sketch to it.

We can first show P (X 1???M ,?? Z = z) are Dirac for all z ??? Z.

Then we use the fact that DISPLAYFORM6 Finally, we follow the tower rule of expectation and the conditional independence property similar to the proof in Proposition 2 and this concludes the proof.

For a detailed description of the baselines, we point the reader to MFN BID50 , MARN BID51 , TFN , BC-LSTM BID27 , MV-LSTM (Rajagopalan et al., 2016) , EF-LSTM (Hochreiter & Schmidhuber, 1997; Graves et al., 2013; BID32 , DF BID20 , MV-HCRF BID35 , EF-HCRF , THMM BID17 , SVM-MD BID48 and RF BID2 .We use the following extra notations for full descriptions of the baseline models described in Section 3.2, paragraph 3:Variants of EF-LSTM: EF-LSTM (Early Fusion LSTM) uses a single LSTM (Hochreiter & Schmidhuber, 1997) on concatenated multimodal inputs.

We also implement the EF-SLSTM (stacked) (Graves et al., 2013) , EF-BLSTM (bidirectional) BID32 and EF-SBLSTM (stacked bidirectional) versions.

Variants of EF-HCRF: EF-HCRF: (Hidden Conditional Random Field) uses a HCRF to learn a set of latent variables conditioned on the concatenated input at each time step.

EF-LDHCRF (Latent Discriminative HCRFs) are a class of models that learn hidden states in a CRF using a latent code between observed concatenated input and hidden output.

EF-HSSHCRF: (Hierarchical Sequence Summarization HCRF) BID36 ) is a layered model that uses HCRFs with latent variables to learn hidden spatio-temporal dynamics.

Variants of MV-HCRF: MV-HCRF: Multi-view HCRF BID35 is an extension of the HCRF for Multi-view data, explicitly capturing view-shared and view specific sub-structures.

MV-LDHCRF: ) is a variation of the MV-HCRF model that uses LDHCRF instead of HCRF.

MV-HSSHCRF: BID36 further extends EF-HSSHCRF by performing Multiview hierarchical sequence summary representation.

In the following, we provide the full results for all baselines models described in Section 3.2, paragraph 3.

TAB6 contains results for multimodal speaker traits recognition on the POM dataset.

TAB7 contains results for the multimodal sentiment analysis on the CMU-MOSI, ICT-MMMO, YouTube, and MOUD datasets.

TAB8 contains results for multimodal emotion recognition on the IEMOCAP dataset.

MFM consistently achieves state-of-the-art or competitive results for all six multimodal datasets.

We believe that by our MFM design, the multimodal discriminative factor F y has successfully learned more meaningful representations by distilling discriminative features.

This highlights the benefit of learning factorized multimodal representations towards discriminative tasks.

For each of the multimodal time series datasets as mentioned in Section 3.2, paragraph 3, we extracted the following multimodal features: Language: We use pre-trained word embeddings (glove.840B.300d) BID22 to convert the video transcripts into a sequence of 300 dimensional word vectors.

Visual: We use Facet (iMotions, 2017) to extract a set of features including per-frame basic and advanced emotions and facial action units as indicators of facial muscle movement (Ekman et al., 1980; BID12 .

Acoustic: We use COVAREP BID11 to extract low level acoustic features including 12 Mel-frequency cepstral coefficients (MFCCs), pitch tracking and voiced/unvoiced segmenting features, glottal source parameters, peak slope parameters and maxima dispersion quotients.

To reach the same time alignment between different modalities we choose the granularity of the input to be at the level of words.

The words are aligned with audio using P2FA BID47 to get their exact utterance times.

We use expected feature values across the entire word for visual and acoustic features since they are extracted at a higher frequencies.

We make a note that the features for some of these datasets are constantly being updated.

The authors of BID50 notified us of a discrepancy in the sampling rate for acoustic feature extraction in the ICT-MMMO, YouTube and MOUD datasets which led to inaccurate word-level alignment between the three modalities.

They publicly released the updated multimodal features.

We performed all experiments on the latest versions of these datasets which can be accessed from https: //github.com/A2Zadeh/CMU-MultimodalSDK.

All baseline models were retrained with extensive hyperparameter search for fair comparison.

Information-Based Interpretation: We choose the normalized Hilbert-Schmidt Independence Criterion (Gretton et al., 2005; BID45 as the approximation (see BID38 BID45 ) of our MI measure: DISPLAYFORM0 where ??? represents y or a i , n is the number of {F ??? ,X i } pairs, DISPLAYFORM1 n??n is the Gram matrix ofX i with DISPLAYFORM2 and k 2 (???, ???) are predefined kernel functions.

The most common choice for the kernel is the RBF kernel.

However, if we consider time series data with various time steps, we need to either perform data augmentation or choose another kernel choice.

For example, we can adopt the Global Alignment Kernel BID10 which considers the DISPLAYFORM3 DISPLAYFORM4 alignment between two varying-length time series when computing the kernel score between them.

To simplify our analysis, we choose to augment data before we calculate the kernel score with the RBF kernel.

More specifically, we perform averaging over time series data: DISPLAYFORM5 The bandwidth of the RBF kernel is set as 1.0 throughout the experiments.

, i ??? {( )anguage, (v)isual, (a)coustic} for the POM dataset for personality traits prediction.

Here, we provide an additional interpretation result for the POM dataset in TAB9 .

We observe that the language modality is also the most informative while the visual and acoustic modalities are almost equally informative.

This result is in agreement with behavioral studies which have observed that non-verbal behaviors are particularly informative of personality traits (Guimond & Massrieh, 2012; Levine et al., 2009; BID15 .

For example, the same sentence "this movie was great" can convey significantly different messages on speaker confidence depending on whether it was said in a loud and exciting voice, with eye contact, or powerful gesticulation.

Gradient-Based Interpretation: MFM reconstructs x i as follows: xi = Fi(fai, fy), fai = Gai(zai), fy = Gy(zy), zai ??? Q(Zai Xi = xi), zy ??? Q(Zy X 1???M = x 1???M ).Equation equation 12 also explains how we obtain f y ??? P (F y X 1???M = x 1???M ).

The gradient flow through time is defined as: DISPLAYFORM6

For experiments on the multimodal synthetic image dataset, we use convolutional+fully-connected layers for the encoder and deconvolutional+fully-connected layers for the decoder BID52 .

Different convolutional layers are each applied on the input SVHN and MNIST images to learn modality-specific generative factors.

Next, we concatenate the features from two more convolutional layers on SVHN and MNIST to learn the multimodal-discriminative factor.

The multimodal discriminative factor is passed through fully-connected layers to predict the label.

For generation, we concatenate the multimodal discriminative factors and the modality-specific generative factor together and use a deconvolutional layer to generate digits.

F ENCODER AND DECODER DESIGN FOR MULTIMODAL TIME SERIES DATASETS Figure 5 illustrates how MFM operates on multimodal time series data.

The encoder Q(Z y X 1???M ) can be parametrized by any model that performs multimodal fusion BID20 BID50 .

We choose the Memory Fusion Network (MFN) BID50 as our encoder Q(Z y X 1???M ).

We use encoder LSTM networks and decoder LSTM networks BID8 to parametrize functions Q(Z a1???M X 1???M ) and F 1???M respectively, and FCNNs to parametrize functions G y , G a{1???M } and D. Figure 5 : Recurrent neural architecture for MFM.

The encoder Q(Z y X 1???M ) can be parametrized by any model that performs multimodal fusion BID20 BID50 .

We use encoder LSTM networks and decoder LSTM networks BID8 to parametrize functions Q(Z a1???M X 1???M ) and F 1???M respectively, and FCNNs to parametrize functions G y , G a{1???M } and D.

We illustrate the surrogate inference for addressing the missing modalities issue in FIG6 .

The surrogate inference model infers the latent codes given the present modalities.

These inferred latent codes can then be used for reconstructing the missing modalities or label prediction in the presence of missing modalities.

A similar approach for factorizing the latent factors was recently proposed by Hsu & Glass (2018) in work that was performed independently and in parallel.

In comparison with MFM, there are several major differences that can be categorized into the (1) prior matching discrepancy, (2) inference network, (3) discriminative objective, (4) multimodal fusion, FORMULA11 scale of experiments.

factorization as proposed in our model (i.e. modality-specific generative factors Z a{1???M } and a multimodal discriminative factor Z y ).

To provide a fair comparison to our discriminative model, we fine tune by training a classifier on top of the multimodal discriminative factor Z y to the label Y.We provide experimental results in TAB12 on the CMU-MOSI, ICT-MMMO, YouTube and MOUD datasets.

MFM outperforms ??-VAE across these datasets and metrics.

@highlight

We propose a model to learn factorized multimodal representations that are discriminative, generative, and interpretable.

@highlight

This paper presents 'Multimodal Factorization model' that factorizes representations into shared multimodal discriminative factors and modality specific generative factors. 