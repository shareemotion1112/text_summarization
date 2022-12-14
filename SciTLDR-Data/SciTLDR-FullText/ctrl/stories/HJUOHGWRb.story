We introduce contextual explanation networks (CENs)---a class of models that learn to predict by generating and leveraging intermediate explanations.

CENs are deep networks that generate parameters for context-specific probabilistic graphical models which are further used for prediction and play the role of explanations.

Contrary to the existing post-hoc model-explanation tools, CENs learn to predict and to explain jointly.

Our approach offers two major advantages: (i) for each prediction, valid instance-specific explanations are generated with no computational overhead and (ii) prediction via explanation acts as a regularization and boosts performance in low-resource settings.

We prove that local approximations to the decision boundary of our networks are consistent with the generated explanations.

Our results on image and text classification and survival analysis tasks demonstrate that CENs are competitive with the state-of-the-art while offering additional insights behind each prediction, valuable for decision support.

Model interpretability is a long-standing problem in machine learning that has become quite acute with the accelerating pace of widespread adoption of complex predictive algorithms.

While high performance often supports our belief in predictive capabilities of a system, perturbation analysis reveals that black-box models can be easily broken in an unintuitive and unexpected manner BID0 BID1 .

Therefore, for a machine learning system to be used in a social context (e.g., in healthcare) it is imperative to provide a sound reasoning for each decision.

Restricting the class of models to only human-intelligible BID2 ) is a potential remedy, but often is limiting in modern practical settings.

Alternatively, we may fit a complex model and explain its predictions post-hoc, e.g., by searching for linear local approximations of the decision boundary BID22 .

While such approaches achieve their goal, the explanations are generated a posteriori, require additional computation per data instance, and most importantly are never the basis for the predictions made in the first place which may lead to erroneous interpretations.

Explanation is a fundamental part of the human learning and decision process (Lombrozo, 2006) .

Inspired by this fact, we introduce contextual explanation networks (CENs)-a class of deep neural networks that generate parameters for probabilistic graphical models.

The generated models not only play the role of explanations but are used for prediction and can encode arbitrary prior knowledge.

The data often consists of two representations: (1) low-level or unstructured features (e.g., text, image pixels, sensory inputs), and (2) high-level or human-interpretable features (e.g., categorical variables).

To ensure interpretability, CENs use deep networks to process the low-level representation (called the context) and construct explanations as context-specific probabilistic models on the high-level features (cf.

Koller & Friedman, 2009, Ch.

5.3) .

Importantly, the explanation mechanism is an integral part of CEN, and our models are trained to predict and to explain jointly.

A motivating example.

Consider a CEN for diagnosing the risk of developing heart arrhythmia ( FIG0 ).

The causes of the condition are quite diverse, ranging from smoking and diabetes to an injury from previous heart attacks, and may carry different effects on the risk of arrhythmia in different contexts.

Assume that the data for each patient consists of medical notes in the form of raw text (which is used as the context) and a number of specific attributes (such as high blood pressure, diabetes, smoking, etc.).

Further, assume that we have access to a parametric class of expert-designed models that relate the attributes to the condition.

The CEN maps the medical notes to the parameters of the model class to produce a context-specific hypothesis, which is further used to make a prediction.

In the sequel, we formalize these intuitions and refer to this toy example in our discussion to illustrate different aspects of the framework.

The main contributions of the paper are as follows:(i) We formally define CENs as a class of probabilistic models, consider special cases (e.g., Jacobs et al., 1991) , and derive learning and inference algorithms for simple and structured outputs. (ii) We prove that post-hoc approximations of CEN's decision boundary are consistent with the generated explanations and show that, in practice, while both methods tend to produce virtually identical explanations, CENs construct them orders of magnitude faster. (iii) It turns out that noisy features can render post-hoc methods inconsistent and misleading, and we show how CENs can help to detect and avoid such situations. (iv) We implement CENs by extending a number of established domain-specific deep architectures for image and text data and design new architectures for survival analysis.

Experimentally, we demonstrate the value of learning with explanations for prediction and model diagnostics.

Moreover, we find that explanations can act as a regularizer and improve sample efficiency.

Deep graphical models.

The idea of combining deep networks with graphical models has been explored extensively.

Notable threads of recent work include: replacing task-specific feature engineering with task-agnostic general representations (or embeddings) discovered by deep networks (Collobert et al., 2011; Rudolph et al., 2016; Rudolph et al., 2017) , representing potential functions (Jaderberg et al., 2014) and energy functions (Belanger & McCallum, 2016) with neural networks, encoding learnable structure into Gaussian process kernels with deep and recurrent networks (Wilson et al., 2016; Al-Shedivat et al., 2017) , or learning state-space models on top of nonlinear embeddings of observables (Gao et al., 2016; BID5 Krishnan et al., 2017) .

The goal of this body of work is to design principled structured probabilistic models that enjoy the flexibility of deep learning.

The key difference between CENs and previous art is that the latter directly integrate neural networks into graphical models as components (embeddings, potential functions, etc.) .

While flexible, the resulting deep graphical models could no longer be clearly interpreted in terms of crisp relationships between specific variables of interest 1 .

CENs, on the other hand, preserve simplicity of the contextual models (explanations) and shift complexity into the process of conditioning on the context.

Meta-learning.

The way CENs operate resembles the meta-learning setup.

In meta-learning, the goal is to learn a meta-model which, given a task, can produce another model capable of solving the task (Thrun & Pratt, 1998) .

The representation of the task can be seen as the context while produced task-specific models are similar to CEN-generated explanations.

Meta-training a deep network that generates parameters for another network has been successfully used for zero-shot (Lei Ba et al., 2015; Changpinyo et al., 2016) and few-shot (Edwards & Storkey, 2016; Vinyals et al., 2016) learning, cold-start recommendations (Vartak et al., 2017) , and a few other scenarios (Bertinetto et al., 2016; De Brabandere et al., 2016; Ha et al., 2016) , but is not suitable for interpretability purposes.

In contrast, CENs generate parameters for models from a restricted class (potentially, based on domain knowledge) and use the attention mechanism (Xu et al., 2015) to further improve interpretability.

Using explanations based on domain knowledge is known to improve generalization (Mitchell et al., 1986) and could be used as a powerful mechanism for solving complex downstream tasks such as program induction for solving algebraic word problems (Ling et al., 2017) .Context representation.

Generating a probabilistic model by conditioning on a context is the key aspect of our approach.

Previous work on context-specific graphical models represented contexts with a discrete variable that enumerated a finite number of possible contexts (cf.

Koller & Friedman, 2009, Ch.

5.3) .

CENs, on the other hand, are designed to handle arbitrary complex context representations.

We also note that context-specific approaches are widely used in language modeling where the context is typically represented with trainable embeddings (Rudolph et al., 2016; Liu et al., 2017) .Interpretability.

While there are many ways to define interpretability (Lipton, 2016; DoshiVelez & Kim, 2017) , our discussion focuses on explanations defined as simple models that locally approximate behavior of a complex model.

A few methods that allow to construct such explanations in a post-hoc manner have been proposed recently BID22 Shrikumar et al., 2017; Lundberg & Lee, 2017) .

In contrast, CENs learn to generate such explanations along with predictions.

There are multiple other complementary approaches to interpretability ranging from a variety of visualization techniques (Simonyan & Zisserman, 2014; BID1 Mahendran & Vedaldi, 2015; Karpathy et al., 2015) , to explanations by example (Caruana et al., 1999; Kim et al., 2014; Kim et al., 2016; Koh & Liang, 2017) , to natural language rationales (Lei et al., 2016) .

Finally, our framework encompasses the class of so-called personalized or instance-specific models that learn to partition the space of inputs and fit local sub-models (Wang & Saligrama, 2012) .

We consider the problem of learning from a collection of data where each instance is represented by three random variables: the context, C ??? C, the attributes, X ??? X , and the targets, Y ??? Y. Our goal is to learn a model, p w (Y | X, C), parametrized by w that can predict Y from X and C. We define contextual explanation networks as models that assume the following form ( FIG0 ): DISPLAYFORM0 where p(Y | X, ??) is a predictor parametrized by ??.

We call such predictors explanations, since they explicitly relate interpretable variables, X, to the targets, Y. For example, when the targets are scalar and binary, explanations may take the form of linear logistic models; when the targets are more complex, dependencies between the components of Y can be represented by a graphical model, e.g., a conditional random field (Lafferty et al., 2001 ).CENs assume that each explanation is context-specific: p w (?? | C) defines a conditional probability of an explanation ?? being valid in the context C. To make a prediction, we marginalize out ??'s; to interpret a prediction, Y = y, for a given data instance, (x, c), we infer the posterior, p w (?? | Y = y, x, c).

The main advantage of this approach is to allow modeling conditional probabilities, p w (?? | C), in a black-box fashion while keeping the class of explanations, p(Y | X, ??), simple and interpretable.

For instance, when the context is given as raw text, we may choose p w (?? | C) to be represented with a recurrent neural network, while p(Y | X, ??) be in the class of linear models.

Implications of the assumptions made by (1) are discussed in Appendix A. Here, we move on to describing a number of practical choices for p w (?? | C) and learning and inference for those.

In practice, we represent p w (?? | C) with a neural network that encodes the context into the parameter space of explanations.

There are multiple ways to construct an encoder, which we consider below.

Deterministic Encoding.

Consider p w (?? | C) := ?? (?? w (C), ??), where ??(??, ??) is a delta-function and ?? w is the network that maps C to ??.

Collapsing the conditional distribution to a delta-function makes FIG11 : An example of CEN architecture.

The context is represented by an image and transformed by a convnet encoder into an attention vector, which is used to construct a contextual hypothesis from a dictionary of sparse atoms.

MoE uses a similar attention mechanism but for combining predictions of each model in the dictionary.?? depend deterministically on C and results into the following tractable conditional log-likelihood: DISPLAYFORM0 , the posterior also collapses to ?? i = ?? w (c i ), and hence the inference is done via a single forward pass.

Constrained Deterministic Encoding.

The downside of deterministic encoding is the lack of constraints on the generated explanations.

There are multiple reasons why this might be an issue: (i) when the context encoder is unrestricted, it might generate unstable, overfitted local models, (ii) explanations are not guaranteed to be human-interpretable per se, and often require imposing additional constraints, such as sparsity, and (iii) when we want to reason about the patterns in the data as a whole, local explanations are not enough.

To address these issues, we constrain the space of explanations by introducing a global dictionary, DISPLAYFORM1 , where each atom of the dictionary, ?? k , is sparse.

The encoder generates context-specific explanations using soft attention over the dictionary, i.e., each explanation becomes a convex combination of the sparse atoms ( FIG11 ): DISPLAYFORM2 where ?? w (c) is the attention over the dictionary.

As previously, the encoder is a delta-distribution, DISPLAYFORM3 The model is trained by learning the weights, w and the dictionary, D. The log-likelihood is as given in (2) , and learning and inference are done via a forward pass 2 .Mixtures of Experts.

So far, we represented p w (?? | C) by a delta-function centered around the output of the encoder.

It is natural to extend p w (?? | C) to a mixture of delta-distributions, in which case CENs recover the mixtures of experts (MoE, Jacobs et al., 1991) .

In particular, let DISPLAYFORM4 be now a dictionary of experts, and define the encoder as DISPLAYFORM5 The log-likelihood in such case is the same as for MoE: DISPLAYFORM6 Note that p w (k | C) is also represented as soft attention over the dictionary, D, which is now used for combining predictions of each expert, ?? k , for a given context, C, instead of constructing a single context-specific explanation.

Learning is done by either directly optimizing the log-likelihood (4) or via EM.

To infer an explanation for a given context, we compute the posterior (see Appendix C).Contextual Variational Autoencoders.

Modeling p(Y | X, C) in the form of (1) avoids representing the joint distribution, p(??, C), which is a good decision when the data is abundant.

However, incorporating a generative model of the context provides a few benefits: (i) a better regularization in low-resource settings, and (ii) a coherent Bayesian framework that allows imposing additional priors 2 Note that deterministic encoding and the dictionary constraint assume that all explanations have the same graphical structure and parameterization.

Having a more hierarchical or structured space of explanations should be possible using ideas from amortized inference (Rudolph et al., 2017) .

We leave this direction to future work.

on the parameters of explanations, ??.

We accomplish this by representing p(??, C) with a variational autoencoder (VAE) BID25 BID3 whose latent variables are explanation parameters FIG0 ).

The generative process and the evidence lower bound (ELBO) are as follows: DISPLAYFORM7 where DISPLAYFORM8 , and q w (?? | C) and p u (C | ??) the encoder and decoder, respectively.

We consider encoders that also make use of the global learnable dictionary, D, and represent q w (?? | C) in the form of logistic normal distribution over the simplex spanned by the atoms of D. For the prior, p ?? (??), we use a Dirichlet distribution with parameters ?? k < 1 to induce sharp attention.

Derivations are deferred to Appendix D.

In this section, we analyze the relationship between CEN-generated and LIME-generated post-hoc explanations.

LIME BID22 constructs explanations as local linear approximations of the decision boundary of a model f in the neighborhood of a given point (x, c) via optimization: DISPLAYFORM0 where L(f, ??, ?? x,c ) measures the quality of the linear model g ?? : X ??? Y as an approximation to f in the neighborhood of (x, c), and ???(??) is a regularizer.

The typical choice for L and ??? is L 2 and L 1 losses, respectively.

The neighborhood of (x, c) is defined by a distribution ?? x,c concentrated around the point of interest.

Given a trained CEN, we can use LIME to approximate its decision boundary and compare the explanations produced by both methods.

The question we ask:How does the local approximation,??, relate to the actual explanation, ?? , generated and used by CEN to make a prediction in the first place?For the case of binary 3 classification, it turns out that when the context encoder is deterministic and the space of explanations is linear, local approximations,??, obtained by solving (6) recover the original CEN-generated explanations, ?? .

Formally, our result is stated in the following theorem.

Theorem 1.

Let the explanations and the local approximations be in the class of linear models, p(Y = 1 | x, ??) ??? exp x ?? .

Further, let the encoder be L-Lipschitz and pick a sampling distribution, ?? x,c , that concentrates around the point (x, c), such that p ??x,c ( z ??? z > t) < ??(t), where z := (x, c) and ??(t) ??? 0 as t ??? ???. Then, if the loss function is defined as DISPLAYFORM1 the solution of (6) concentrates around ?? as DISPLAYFORM2 Intuitively, by sampling from a distribution sharply concentrated around (x, c), we ensure that?? will recover ?? with high probability.

The proof is given in Appendix B.This result establishes an equivalence between the explanations generated by CEN and those produced by LIME post-hoc when approximating CEN.

Note that when LIME is applied to a model other than CEN, equivalence between explanations is not guaranteed.

Moreover, as we further show experimentally, certain conditions such as incomplete or noisy interpretable features may lead to LIME producing inconsistent and erroneous explanations.

While CEN and LIME generate similar explanations in the case of simple classification (i.e., when Y is a scalar), when Y is structured (e.g., as a sequence), constructing coherent local approximations in a post-hoc manner is non-trivial.

At the same time, CENs naturally let us represent p(Y | X, ??) using arbitrary graphical models.

To demonstrate our approach, we consider survival time prediction task where interpretability can be uniquely valuable (e.g., in a medical setting).

Survival analysis can be re-formulated as a sequential prediction problem .

To this end, we design CENs with CRF-based explanations suitable for sequentially structured outputs.

Our setup is as follows.

Again, the data instances are represented by contexts, C, attributes, X, and targets, Y. The difference is that now targets are sequences of m binary variables, Y := (y 1 , . . .

, y m ), that indicate occurrence of an event.

If the event occurred at time t ??? [t i , t i+1 ), then y j = 0, ???j ??? i and y k = 1, ???k >

i. If the event was censored (i.e., we lack information for times after t ??? [t i , t i+1 )), we represent targets (y i+1 , . . .

, y m ) with latent variables.

Note that only m + 1 sequences are valid, i.e., assigned non-zero probability by the model.

We define CRF-based CEN as: DISPLAYFORM0 Note that here we have explanations for each time point, ?? 1:m , and use an RNN-based encoder ?? t .

The potentials between attributes, x, and targets, y 1:m , are linear functions parameterized by ?? 1:m ; the pairwise potentials between targets, ??(y i , y i+1 ), ensure that configurations (y i = 1, y i+1 = 0) are improbable (i.e., ??(1, 0) = ?????? and ??(0, 0) = ?? 00 , ??(0, 1) = ?? 01 , ??(1, 1) = ?? 10 are learnable parameters).

Given these constraints, the likelihood of an uncensored event at time t ??? [t j , t j+1 ) is DISPLAYFORM1 and the likelihood of an event censored at time t ??? [t j , t j+1 ) is DISPLAYFORM2 The joint log-likelihood of the data consists of two parts: (a) the sum over the non-censored instances, for which we compute log p(T = t | x, ??), and (b) sum over the censored instances, for which we use log p(T ??? t | x, ??).

We provide a much more elaborate discussion of the survival time prediction setup and our architectures in Appendix E.

For empirical evaluation, we consider applications that involve different data modalities of the context: image, text, and time-series.

In each case, CENs are based on deep architectures designed for learning from the given type of context.

In the first part, we focus on classification tasks and use linear logistic models as explanations.

In the second part, we apply CENs to survival analysis and use structured explanations in the form of conditional random fields (CRFs).We design our experiments around the following questions:(i) When explanation is a part of the learning and prediction process, how does that affect performance of the predictive model?

Does the learning become more or less efficient both in terms of convergence and sample complexity?

How do CENs stand against vanilla deep nets? (ii) Explanations are as good as the features they use to explain predictions.

We ask how noisy interpretable features affect explanations generated post-hoc by LIME and whether CEN can help to detect and avoid such situations. (iii) Finally, we ask what kind of insight we can gain by visualizing and inspecting explanations?

Details on the setup, all hyperparameters, and training procedures are given in Appendices F.1 and F.3.

Table 1 : Performance of the models on classification tasks (averaged over 5 runs; the std. are on the order of the least significant digit).

The subscripts denote the features on which the linear models are built: pixels (pxl), HOG (hog), bag-or-words (bow), topics (tpc), embeddings (emb), discrete attributes (att).

Best previous results for supervised learning and similar LSTM architectures: 8.1% BID5 .

Classical datasets.

We consider two classical image datasets, MNIST 4 and CIFAR10 5 , and a text dataset for sentiment classification of IMDB reviews BID6 .

For MNIST and CIFAR10: full images are used as the context; to imitate high-level features, we use (a) the original images cubically downscaled to 20 ?? 20 pixels, gray-scaled and normalized, and (b) HOG descriptors computed using 3 ?? 3 blocks BID7 .

For IMDB: the context is represented by sequences of words; for high-level features we use (a) the bag-of-words (BoW) representation and (b) the 50-dimensional topic representation produced by a separately trained off-the-shelf topic model.

Neither data augmentation, nor pre-training or other unsupervised techniques were used.

Remote sensing.

We also consider the problem of poverty prediction for household clusters in Uganda from satellite imagery and survey data (the dataset is referred to as Satellite).

Each household cluster is represented by a collection of 400 ?? 400 satellite images (used as the context) and 65 categorical variables from living standards measurement survey (used as the interpretable attributes).

The task is binary classification of the households into poor and not poor.

We follow the original study of and use a pre-trained VGG-F network to compute 4096-dimensional embeddings of the satellite images on top of which we build contextual models.

Note that this datasets is fairly small (642 points), and hence we keep the VGG-F part of the model frozen to avoid overfitting.

For each task, we use linear regression and vanilla deep nets as baselines.

For MNIST and CIFAR10, the networks are a simple convnet (2 convolutions followed by max pooling) and the VGG-16 architecture (Simonyan & Zisserman, 2014) , respectively.

For IMDB, following Johnson & Zhang (2016) and we use a bi-directional LSTM with max pooling.

For Satellite, we use a fixed VGG-F followed by a multi-layer perceptron (MLP) with 1 hidden layer.

Our models used the baseline deep architectures as their context encoders and were of three types: (a) CENs with constrained deterministic encoding (b) mixture of experts (MoE), (c) CENs with variational context autoencoding (VCEN).

All our models use the dictionary constraint and sparsity regularization.

In this part, we compare CENs with the baselines in terms of performance.

In each task, CENs are trained to simultaneously generate predictions and construct explanations using a global dictionary.

When the dictionary size is 1, they become equivalent to linear models.

For larger dictionaries, CENs become as flexible as deep nets FIG1 .

Adding a small sparsity penalty on the dictionary (between 10 ???6 and 10 ???3 , see TAB5 , 5) helps to avoid overfitting for very large dictionary sizes, so that the model learns to use only a few dictionary atoms for prediction while shrinking the rest to zeros.

Overall, CENs show very competitive performance and are able to approach or surpass baselines in a number of cases, especially on the IMDB data (Table 1) .

Thus, forcing the model to produce explanations along with predictions does not limit its capacity.

Additionally, the "explanation layer" in CENs somehow affects the geometry of the optimization problem, and we notice that it often causes faster convergence FIG1 .

When the models are trained on a subset of data (size varied between 1% and 20% for MNIST and 2% and 40% for IMDB), explanations play the role of a regularizer which strongly improves the sample efficiency of our models FIG1 ).

This becomes even more evident from the results on the Satellite dataset that had only 500 training points: contextual explanation networks significantly improved upon the sparse linear models on the survey features (known as the gold standard in remote sensing).

Note that training an MLP on both the satellite image features and survey variables, while beneficial, does not come close to the result achieved by contextual explanation networks (Table 1) .

While regularization is a useful aspect, the main use case for explanations is model diagnostics.

Linear explanation assign weights to the interpretable features, X, and hence their quality depends on the way we select these features.

We consider two cases where (a) the features are corrupted with additive noise, and (b) the selected features are incomplete.

For analysis, we use MNIST and IMDB datasets.

Our question is, Can we trust the explanations on noisy or incomplete features?The effect of noisy features.

In this experiment, we inject noise 6 into the features X and ask LIME and CEN to fit explanations to the corrupted features.

Note that after injecting noise, each data point has a noiseless representation C and noisy X. LIME constructs explanations by approximating the decision boundary of the baseline model trained to predict Y from C features only.

CEN is trained to construct explanations given C and then make predictions by applying explanations to X. The predictive performance of the produced explanations on noisy features is given on FIG2 .

Since baselines take only C as inputs, their performance stays the same and, regardless of the noise level, LIME "successfully" overfits explanations-it is able to almost perfectly approximate the decision boundary of the baselines using very noisy features.

On the other hand, performance of CEN gets worse with the increasing noise level indicating that model fails to learn when the selected interpretable representation is of low quality.

The effect of feature selection.

Here, we use the same setup, but instead of injecting noise into X, we construct X by randomly subsampling a set of dimensions.

FIG2 demonstrates the result.

While performance of CENs degrades proportionally to the size of X, we see that, again, LIME is able to fit explanations to the decision boundary of the original models despite the loss of information.

These two experiments indicate a major drawback of explaining predictions post-hoc: when constructed on poor, noisy, or incomplete features, such explanations can overfit the decision boundary of a predictor and are likely to be misleading.

For example, predictions of a perfectly valid model might end up getting absurd explanations which is unacceptable from the decision support point of view.

Here, we focus on the poverty prediction task to analyze CEN-generated explanations qualitatively.

Detailed discussion of qualitative results and visualization of the learned explanations for MNIST and IMDB datasets are given in Appendix F.2.After training CEN with a dictionary of size 32, we discover that the encoder tends to sharply select one of the two explanations (M1 and M2) for different household clusters in Uganda (see FIG3 , also FIG0 in appendix).

In the survey data, each household cluster is marked as either urban or rural.

We notice that, conditional on a satellite image, CEN tends to pick M1 for urban areas and M2 for rural FIG3 ).

Notice that explanations weigh different categorical features, such as reliability of the water source or the proportion of houses with walls made of unburnt brick, quite differently.

When visualized on the map, we see that CEN selects M1 more frequently around the major city areas, which also correlates with high nightlight intensity in those areas FIG3 )

.High performance of the model makes us confident in the produced explanations (contrary to LIME as discussed in the previous section) and allows us to draw conclusions about what causes the model to classify certain households in different neighborhoods as poor.

Finally, we apply CENs to survival analysis and showcase how to use our networks with structured explanations.

In survival analysis, the goal is to learn a predictor for the time of occurrence of an event (in this case, the death of a patient) as well as be able to assess the risk (or hazard) of the occurrence.

The classical models for this task are the Aalen's additive model ) and the Cox proportional hazard model , which linearly regress attributes of a particular patient, X, to the hazard function.

have shown that survival analysis can be formulated as a structured prediction problem and solved using a CRF variant.

Here, we propose to use CENs with deep nets as encoders and CRF-structured explanations (as described in Section 3.3).

More details on the architectures of our models, the baselines, as well as more background on survival analysis are provided in Appendix E.Datasets.

We use two publicly available datasets for survival analysis of of the intense care unit (ICU) patients: (a) SUPPORT2 7 , and (b) data from the PhysioNet 2012 challenge 8 .

The data was preprocessed and used as follows: Figure 6 : Weights of the CEN-generated CRF explanations for two patients from SUPPORT2 dataset for a set of the most influential features: dementia (comorbidity), avtisst (avg.

TISS, days 3-25), slos (days from study entry to discharge), hday (day in hospital at study admit), ca_yes (the patient had cancer), sfdm2_Coma or Intub (intubated or in coma at month 2), sfdm2_SIP (sickness impact profile score at month 2).

Higher weight values correspond to higher feature contributions to the risk of death after a given time.??? SUPPORT2: The data had 9105 patient records and 73 variables.

We selected 50 variables for both C and X features (see Appendix F).

Categorical features (such as race or sex) were one-hot encoded.

The values of all features were non-negative, and we filled the missing values with -1.

For CRF-based predictors, the survival timeline was capped at 3 years and converted into 156 discrete intervals of 7 days each.

We used 7105 patient records for training, 1000 for validation, and 1000 for testing.??? PhysioNet: The data had 4000 patient records, each represented by a 48-hour irregularly sampled 37-dimensional time-series of different measurements taken during the patient's stay at the ICU.

We resampled and mean-aggregated the time-series at 30 min frequency.

This resulted in a large number of missing values that we filled with 0.

The resampled time-series were used as the context, C, while for the attributes, X, we took the values of the last available measurement for each variable in the series.

For CRF-based predictors, the survival timeline was capped at 60 days and converted into 60 discrete intervals.

Models.

For baselines, we use the classical Aalen and Cox models and the CRF from , where all used X as inputs.

Next, we combine CRFs with neural encoders in two ways:(i) We apply CRFs to the outputs from the neural encoders (the models denoted MLP-CRF and LSTM-CRF, all trainable end-to-end).

Similar models have been show very successful in the natural language applications (Collobert et al., 2011) .

Note that parameters of the CRF layer assign weights to the latent features and are no longer interpretable in terms of the attributes of interest. (ii) We use CENs with CRF-based explanations, that process the context variables, C, using the same neural networks as in (i) and output parameters for CRFs that act on the attributes, X.Details on the architectures are given in Appendix F.3.

, we use two metrics specific to survival analysis: (a) accuracy of correctly predicting survival of a patient at times that correspond to 25%, 50%, and 75% populationlevel temporal quantiles (i.e., time points such that the corresponding % of the patients in the data were discharged from the study due to censorship or death) and (b) the relative absolute error (RAE) between the predicted and actual time of death for non-censored patients.

Quantitative results.

The results for all models are given in TAB2 .

Our implementation of the CRF baseline reproduces (and even slightly improves) the performance reported by .

MLP-CRF and LSTM-CRF improve upon plain CRFs but, as we noted, can no longer be interpreted in terms of the original variables.

On the other hand, CENs outperform neural CRF models on certain metrics (and closely match on the others) while providing explanations for risk prediction for each patient at each point in time.

Qualitative results.

To inspect predictions of CENs qualitatively, for any given patient, we can visualize the weights assigned by the corresponding explanation to the respective attributes.

Figure 6 explanation weights for a subset of the most influential features for two patients from SUPPORT2 dataset who were predicted as survivor and non-survivor.

These explanations allow us to better understand patient-specific temporal dynamics of the contributing factors to the survival rates predicted by the model FIG4 ).

In this paper, we have introduced contextual explanation networks (CENs)-a class of models that learn to predict by generating and leveraging intermediate context-specific explanations.

We have formally defined CENs as a class of probabilistic models, considered a number of special cases (e.g., the mixture of experts), and derived learning and inference procedures within the encoder-decoder framework for simple and sequentially-structured outputs.

We have shown that, while explanations generated by CENs are provably equivalent to those generated post-hoc under certain conditions, there are cases when post-hoc explanations are misleading.

Such cases are hard to detect unless explanation is a part of the prediction process itself.

Besides, learning to predict and to explain jointly turned out to have a number of benefits, including strong regularization, consistency, and ability to generate explanations with no computational overhead.

We would like to point out a few limitations of our approach and potential ways of addressing those in the future work.

Firstly, while each prediction made by CEN comes with an explanation, the process of conditioning on the context is still uninterpretable.

Ideas similar to context selection (Liu et al., 2017) or rationale generation (Lei et al., 2016) may help improve interpretability of the conditioning.

Secondly, the space of explanations considered in this work assumes the same graphical structure and parameterization for all explanations and uses a simple sparse dictionary constraint.

This might be limiting, and one could imagine using a more hierarchically structured space of explanations instead, bringing to bear amortized inference techniques (Rudolph et al., 2017) .

Nonetheless, we believe that the proposed class of models is useful not only for improving prediction capabilities, but also for model diagnostics, pattern discovery, and general data analysis, especially when machine learning is used for decision support in high-stakes applications.

As described in the main text, CENs represent the predictive distribution in the following form: DISPLAYFORM0 and the assumed generative process behind the data is either: DISPLAYFORM1 when we model the joint distribution of the explanations, ??, and contexts, C, e.g., using encoder-decoder framework.

We would like to understand whether CEN, as defined above, can represent any conditional distribution, p(Y | X, C), when the class of explanations is limited (e.g., to linear models), and, if not, what are the limitations?Generally, CEN can be seen as a mixture of predictors.

Such mixture models could be quite powerful as long as the mixing distribution, p(?? | C), is rich enough.

In fact, even a finite mixture exponential family regression models can approximate any smooth d-dimensional density at a rate O(m ???4/d ) in the KL-distance BID21 .

This result suggests that representing the predictive distribution with contextual mixtures should not limit the representational power of the model.

The two caveats are:(i) In practice, p(?? | C) is limited, e.g., either deterministic encoding, a finite mixture, or a simple distribution parametrized by a deep network. (ii) The classical setting of predictive mixtures does not separate inputs into two subsets, (C, X).We do this intentionally to produce hypotheses/explanations in terms of specific features that could be useful for interpretability or model diagnostics down the line.

However, it could be the case that X contains only some limited information about Y, which could limit the predictive power of the full model.

We leave the point (ii) to future work.

To address (i), we consider p(?? | C) that fully factorizes over the dimensions of ??: p(?? | C) = j p(?? j | C), and assume that hypotheses, p(Y | X, ??), factorize according to some underlying graph, DISPLAYFORM2 The following proposition shows that in such case p(Y | X, C) inherits the factorization properties of the hypothesis class.

DISPLAYFORM3 explanations also factorizes according to G. DISPLAYFORM4 , where ?? denotes subsets of the Y variables and MB(??) stands for the corresponding Markov blankets.

Using the definition of CEN, we have: DISPLAYFORM5 Remark 1.

All the encoding distributions, p(?? | C), considered in the main text of the paper, including delta functions, their mixtures, and encoders parametrized by neural nets fully factorize over the dimensions of ??.

Remark 2.

The proposition has no implications for the case of scalar targets, Y. However, in case of structured prediction, regardless of how good the context encoder is, CEN will assume the same set of independencies as given by the class of hypotheses, p(Y | X, ??).B APPROXIMATING THE DECISION BOUNDARY OF CEN BID22 proposed to construct approximations of the of the decision boundary of an arbitrary predictor, f , in the locality of a specified point, x, by solving the following optimization DISPLAYFORM6 where L(f, g, ?? x ) measures the quality of g as an approximation to f in the neighborhood of x defined by ?? x and ???(g) is a regularizer that is usually used to ensure human-interpretability of the selected local hypotheses (e.g., sparsity).

Now, consider the case when f is defined by a CEN, instead of x we have (c, x), and the class of approximations, G, coincides with the class of explanations, and hence can be represented by ??.

In this setting, we can pose the same problem as: DISPLAYFORM7 Suppose that CEN produces ?? explanation for the context c using a deterministic encoder, ??.

The question is whether and under which conditions?? can recover ?? .

Theorem 1 answers the question in affirmative and provides a concentration result for the case when hypotheses are linear.

Here, we prove Theorem 1 for a little more general class of log-linear explanations: DISPLAYFORM8 where a is a C-Lipschitz vector-valued function whose values have a zero-mean distribution when (x, c) are sampled from ?? x,c 9 .

For simplicity of the analysis, we consider binary classification and omit the regularization term, ???(g).

We define the loss function, L(f, ??, ?? x,c ), as: DISPLAYFORM9 where (x k , c k ) ??? ?? x,c and ?? x,c := ?? x ?? c is a distribution concentrated around (x, c).

Without loss of generality, we also drop the bias terms in the linear models and assume that a(x k ??? x) are centered.

Proof of Theorem 1.

The optimization problem (16) reduces to the least squares linear regression: DISPLAYFORM10 We consider deterministic encoding, p(?? | c) := ??(??, ??(c)), and hence logit p(Y = 1 | x k ??? x, c k ) takes the following form: DISPLAYFORM11 To simplify the notation, we denote a k := a(x k ??? x), ?? k := ??(c k ), and ?? := ??(c).

The solution of (18) now can be written in a closed form: DISPLAYFORM12 Note that?? is a random variable since (x k , c k ) are randomly generated from ?? x,c .

To further simplify the notation, denote M : DISPLAYFORM13 To get a concentration bound on ?? ??? ?? , we will use the continuity of ??(??) and a(??), concentration properties of ?? x,c around (x, c), and some elementary results from random matrix theory.

To be more concrete, since we assumed that ?? x,c factorizes, we further let ?? x and ?? c concentrate such that p ??x ( x ???x > t) < ?? x (t) and p ??c ( c ???c > t) < ?? c (t), respectively, where ?? x (t) and ?? c (t) both go to 0 as t ??? ???, potentially at different rates.

First, we have the following bound from the convexity of the norm: DISPLAYFORM14 By making use of the inequality Ax ??? A x , where A denotes the spectral norm of the matrix A, the L-Lipschitz property of ??(c), the C-Lipschitz property of a(x), and the concentration of x k around x, we have DISPLAYFORM15 DISPLAYFORM16 DISPLAYFORM17 Note that we used the fact that the spectral norm of a rank-1 matrix, a(x k )a(x k ) , is simply the norm of a(x k ), and the spectral norm of the pseudo-inverse of a matrix is equal to the inverse of the least non-zero singular value of the original matrix: DISPLAYFORM18 min (M ).

Finally, we need a concentration bound on ?? min M/(C?? ) 2 to complete the proof.

Note that DISPLAYFORM19 , where the norm of a kC?? is bounded by 1.

If we denote ?? min (C?? ) the minimal eigenvalue of Cov a k C?? , we can write the matrix Chernoff inequality BID23 as follows: DISPLAYFORM20 where d is the dimension of a k , ?? := L C 2 t , and D(a b) denotes the binary information divergence: DISPLAYFORM21 The final concentration bound has the following form: DISPLAYFORM22 We see that as ?? ??? ??? and t ??? ??? all terms on the right hand side vanish, and hence?? concentrates around ?? .

Note that as long as ?? min (C?? ) is far from 0, the first term can be made negligibly small by sampling more points around (x, c).

Finally, we set ?? ??? t and denote the right hand side by ?? K,L,C (t) that goes to 0 as t ??? ??? to recover the statement of the original theorem.

Remark 3.

We have shown that?? concentrates around ?? under mild conditions.

With more assumptions on the sampling distribution, ?? x,c , (e.g., sub-gaussian) one could derive precise convergence rates.

Note that we are in total control of any assumptions we put on ?? x,c since precisely that distribution is used for sampling.

This is a major difference between the local approximation setup here and the setup of linear regression with random design; in the latter case, we have no control over the distribution of the design matrix, and any assumptions we make could potentially be unrealistic.

Remark 4.

Note that concentration analysis of a more general case when the loss L is a general convex function and ???(g) is a decomposable regularizer could be done by using results from the M-estimation theory BID24 ), but would be much more involved and unnecessary for our purposes.

As noted in the main text, to make a prediction, MoE uses each of the K experts where the predictive distribution is computed as follows: DISPLAYFORM0 Since each expert contributes to the predictive probability, we can explain a prediction,??, for the instance (x, c) in terms of the posterior weights assigned to each expert model: DISPLAYFORM1 If the p(k |??, x, c) assigns very high weight to a single expert, we can treat that expert model as an explanation.

Note that however, in general, this may not be the case and posterior weights could be quite spread out (especially, if the number of experts is small and the class of expert models, p(Y | X, ??), is too simple and limited).

Therefore, there may not exist an equivalent local explanation in the class of expert models that would faithfully approximate the decision boundary.

To learn contextual MoE, we can either directly optimize the conditional log-likelihood, which is non-convex yet tractable, or use expectation maximization (EM) procedure.

For the latter, we write the log likelihood in the following form: DISPLAYFORM2 At each iteration, we do two steps:(E-step) Compute posteriors for each data instance, DISPLAYFORM3 It is well known that this iterative procedure is guaranteed to converge to a local optimum.

We can express the evidence for contextual variational autoencoders as follows: DISPLAYFORM0 DISPLAYFORM1 (1) the expected conditional likelihood of the explanation, E qw [log p(Y | X, ??)], (2) the expected context reconstruction error, E qw [log p u (C | ??)], and (3) the KL-based regularization term, ???KL (q(?? | C) p(??)).We can optimize the ELBO using first-order methods by estimating the gradients via Monte Carlo sampling with reparametrization.

When the encoder has a classical form of a Gaussian distribution (or any other location-scale type of distribution), q w (?? | C) = N (??; ?? w (C), diag (?? w (C))), reparametrization of the samples is straightforward BID25 .In our experiments, we mainly consider encoders that output probability distributions over a simplex spanned by a dictionary, D, which turned out to have better performance and faster convergence.

In particular, sampling from the encoder is as follows: DISPLAYFORM2 The samples, ??, will be logistic normal distributed and are easy to be re-parametrized.

For prior, we use the Dirichlet distribution over ?? with the parameter vector ??.

In that case, the stochastic estimate of the KL-based regularization term has the following form: DISPLAYFORM3

We provide some general background on survival analysis, the classical Aalen additive hazard and Cox proportional hazard models, derive the structured prediction approach , and describe CENs with CRF-based explanations used in our experiments in detail.

In survival time prediction, our goal is to estimate the occurrence time of an event in the future (e.g., death of a patient, earthquake, hard drive failure, customer turnover, etc.).

The unique aspect of the survival data is that there is always a fraction of points for which the event time has not been observed (such data instances are called censored).

The common approach is to model the survival time, T , either for a population (i.e., average survival time) or for each instance.

In particular, we can introduce the survival function, S(t) := p(T ??? t), which gives the probability of the event not happening at least up to time t (e.g., patient survived up to time t).

The derivative of the survival function is called the hazard function, ??(t), which is the instantaneous rate of failure: DISPLAYFORM0 This allows us to model survival on a population level.

Now, proportional hazard models assume that ?? is also a function of the available features of a given instance, i.e., ??(t; x).

Cox's proportional hazard model assumes ?? ?? (t; x) := ?? 0 (t) exp(x ??).

Aalen's model is a time-varying extension and assumes that ?? ?? (t; x) := x ??(t), where ??(t) is a function of time.

Survival analysis is a regression problem as it originally works with continuous time.

The time can be discretized (e.g., into days, months, etc.), and hence we can approach survival time prediction as a multi-task classification problem BID29 .

went one step further, noted that the output space is structure in a particular way, and proposed a model called sequence of dependent regressors, which is in essence a conditional random field with a particular structure of the pairwise potentials between the labels.

In particular, as we described in Section 3.3, the targets are sequences (denoted with h 1 , h 2 , h 3 hidden states respectively).

Each hidden state of the output LSTM was used to generate the corresponding ?? t that were further used to construct the log-likelihood for CRF.of binary random variables, Y := (y 1 , . . .

, y m ), that encode occurrence of an event as follows: for an event that occurred at time t ??? [t i , t i+1 ), then y j = 0, ???j ??? i and y k = 1, ???k > i. Note that only m + 1 sequences are valid, i.e., assigned non-zero probability by the model, which allows us to write the following linear model: DISPLAYFORM1 To train the model, optimize the following objective: DISPLAYFORM2 where the first two terms are regularization and the last term is the log-likelihood which as: DISPLAYFORM3 where NC is the set of non-censored instances (for which we know the outcome times, t i ) and C is the set of censored inputs (for which only know the censorship times, t j ).

Expressions for the likelihoods of censored and non-censored inputs are the same as given in Section 3.3.Finally, CENs additionally take the context variables, C, as inputs and generate ?? t for each time step using a recurrent encoder.

In our experiments, we considered datasets where the context was represented by a vector or regularly sampled time series.

Architectures for CENs used in our experiments are given in FIG8 .

We used encoders suitable for the data type of the context variables available for each dataset.

Each ?? t was generated using a constrained deterministic encoder with a global dictionary, D of size 16.

For details on parametrization of our architectures see tables in Appendix F.3.Importantly, CEN-CRF architectures are trainable end-to-end (as all other CEN architectures considered in this paper), and we optimized the objective using stochastic gradient method.

For each mini-batch, depending on which instances were censored and which were non-censored, we constructed the objective function accordingly (to implement this in TensorFlow we used masking and the standard control flow primitives for selecting between parts of the objective for censored and non-censored inputs).

This section provides details on the experimental setups including architectures, training procedures, etc.

Additionally, we provide and discuss qualitative results for CENs on MNIST and IMDB datasets.

MNIST.

We used the classical split of the dataset into 50k training, 10k validation, and 10k testing points.

All models were trained for 100 epochs using the Adam optimizer with the learning rate of 10 ???3 .

No data augmentation was used in any of our experiments.

HOG representations were computed using 3 ?? 3 blocks.

CIFAR10.

For this set of experiments, we followed the setup given BID30 , reimplemented in Keras with TensorFlow backend.

The input images were global contrast normalized (a.k.a.

GCN whitened) while the rescaled image representations were simply standardized.

Again, HOG representations were computed using 3??3 blocks.

No data augmentation was used in our experiments.

IMDB.

We considered the labeled part of the data only (50,000 reviews total).

The data were split into 20,000 train, 5,000 validation, and 25,000 test points.

The vocabulary was limited to All models were trained with the Adam optimizers with 10 ???2 learning rate.

The models were initialized randomly; no pre-training or any other unsupervised/semi-supervised technique was used.

Satellite.

As described in the main text, we used a pre-trained VGG-16 network BID15 to extract features from the satellite imagery.

Further, we added one fully connected layer network with 128 hidden units used as the context encoder.

For the VCEN model, we used dictionary-based encoding with Dirichlet prior and logistic normal distribution as the output of the inference network.

For the decoder, we used an MLP of the same architecture as the encoder network.

All models were trained with Adam optimizer with 0.05 learning rate.

The results were obtained by 5-fold cross-validation.

Medical data.

We have used minimal pre-processing of both SUPPORT2 and PhysioNet datasets limited to standardization and missing-value filling.

We found that denoting missing values with negative entries (???1) often led a slightly improved performance compared to any other NA-filling techniques.

PhysioNet time series data was irregularly sampled across the time, so we had to resample temporal sequences at regular intervals of 30 minutes (consequently, this has created quite a few missing values for some of the measurements).

All models were trained using Adam optimizer with 10 ???2 learning rate.

Here, we discuss additional qualitative results obtained for CENs on MNIST and IMDB data.

Figures 9a, 9b, and 9c visualize explanations for predictions made by CEN-pxl on MNIST.

The figures correspond to 3 cases where CEN (a) made a correct prediction, (b) made a mistake, and (c) was applied to an adversarial example (and made a mistake).

Each chart consists of the following columns: true labels, input images, explanations for the top 3 classes (as given by the activation of the final softmax layer), and attention vectors used to select explanations from the global dictionary.

A small subset of explanations from the dictionary is visualized in FIG9 (the full dictionary is given in FIG0 ), where each image is a weight vector used to construct the pre-activation for a particular class.

Note that different elements of the dictionary capture different patterns in the data (in FIG9 , different styles of writing the 0 digit) which CEN actually uses for prediction.

Also note that confident correct predictions (Figures 9a) are made by selecting a single explanation from the dictionary using a sharp attention vector.

However, when the model makes a mistake, its attention is often dispersed FIG9 ), i.e., there is uncertainty in which pattern it tries to use for prediction.

FIG9 further quantifies this phenomenon by plotting histogram of the attention entropy for all test examples which were correctly and incorrectly classified.

While CENs are certainly not adversarial-proof, high entropy of the attention vectors is indicative of ambiguous or out-of-distribution examples which is helpful for model diagnostics.

FIG0 : Histograms of test weights assigned by CEN to 6 topics: Acting-and plot-related topics (upper charts), genre topics (bottom charts).

Note that acting-related topics are often bi-modal, i.e., contributing either positive, negative, or zero weight to the sentiment prediction in different contexts.

Genre topics almost always have negligible contributions.

This allows us to conclude that the learned model does not have any particular biases towards or against any a given genre.

Similar to MNIST, we train CEN-tpc with linear explanations in terms of topics on the IMDB dataset.

Then, we generate explanations for each test example and visualize histograms of the weights assigned by the explanations to 6 selected topics in FIG0 .

The 3 topics in the top row are actingand plot-related (and intuitively have positive, negative, or neutral connotation), while the 3 topics in the bottom are related to particular genre of the movies.

Note that acting-related topics turn out to be bi-modal, i.e., contributing either positively, negatively, or neutrally to the sentiment prediction in different contexts.

As expected intuitively, CEN assigns highly negative weight to the topic related to "bad acting/plot" and highly positive weight to "great story/performance" in most of the contexts (and treats those neutrally conditional on some of the reviews).

Interestingly, genre-related topics almost always have a negligible contribution to the sentiment (i.e., get almost 0 weights assigned by explanations) which indicates that the learned model does not have any particular bias towards or against a given genre.

Importantly, inspecting summary statistics of the explanations generated by CEN allows us to explore the biases that the model picks up from the data and actively uses for prediction 11 .

FIG0 visualizes the full dictionary of size 16 learned by CEN-tpc.

Each column corresponds to a dictionary atom that represents a typical explanation pattern that CEN attends to before making a prediction.

By inspecting the dictionary, we can find interesting patterns.

For instance, atoms 5 and 11 assign inverse weights to topics [kid, child, disney, family] and [sexual, violence, nudity, sex] .

Depending on the context of the review, CEN may use one of these patterns to predict the sentiment.

Note that these two topics are negatively correlated across all dictionary elements, which again is quite intuitive.

<|TLDR|>

@highlight

A class of networks that generate simple models on the fly (called explanations) that act as a regularizer and enable consistent model diagnostics and interpretability.

@highlight

The authors claim that the previous art directly integrate neural networks into the graphical models as components, which renders the models uninterpretable.

@highlight

Proposal for a combination of neural nets and graphical models by using a deep neural net to predict the parameters of a graphical model.