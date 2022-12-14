Deep neural networks have demonstrated promising prediction and classification performance on many healthcare applications.

However, the interpretability of those models are often lacking.

On the other hand, classical interpretable models such as rule lists or decision trees do not lead to the same level of accuracy as deep neural networks and can often be too complex to interpret (due to the potentially large depth of rule lists).

In this work, we present PEARL,  Prototype lEArning via Rule Lists, which iteratively uses rule lists to guide a neural network to learn representative data prototypes.

The resulting prototype neural network provides  accurate prediction, and the prediction can be easily explained by  prototype and its guiding rule lists.

Thanks to the prediction power of neural networks, the rule lists from				 prototypes are more concise and hence provide better interpretability.

On two real-world electronic healthcare records (EHR) datasets, PEARL consistently outperforms all baselines across both datasets, especially achieving performance improvement over conventional rule learning by up to 28% and over prototype learning by up to 3%.

Experimental results also show the resulting interpretation of PEARL is  simpler than the standard rule learning.

The rapid growth of sizes and complexities of electronic health records (EHR) data has motivated the use of deep learning models, which demonstrated state-of-the-art performance in many tasks, including diagnostics and disease detection BID7 BID38 , medication prediction BID16 , risk prediction BID9 Xiao et al., 2018b) , and patient subtyping BID1 BID4 .

Although deep learning models can produce accurate predictions and classifications, they are often treated as black-box models that lack interpretability and transparency of their inner working BID20 .

This is a critical problem as it can limit the adoption of deep learning in medical decision making.

Recently, there have been great efforts of trying to explain black-box deep models, including via attention mechanism BID7 BID40 , visualization BID31 , and explanation by examples or prototypes BID17 .

To bring deep models into real clinical practice, clinicians often need to understand why a certain output is produced and how the model generates this output for a given input BID24 .

Rule learning and prototype learning are two promising directions to achieve clinical model interpretability.

Rule learning generates a set of rules from training data, in which its prediction is done at leaf levels via simple models such as majority vote or regression.

For example, the results of rule learning are rule lists composed of multiple if-then statements BID0 .

Those rules can be interpretable to domain experts as they are expressed in simple logical forms BID30 BID3 .

However, because of such a simple prediction model, the accuracy of rule-based models is often lower than deep neural networks.

Moreover, the interpretability can be undermined as the depth of rules becomes very large and thus incomprehensible for human with tens or hundreds of levels of the rules.

Prototype learning is another interpretable model inspired by case-based reasoning BID14 , where observations are classified based on their proximity to a prototype point in the dataset.

Many machine learning models have incorporated prototype concepts BID28 BID2 BID12 , and learn to compute prototypes (as actual data points or synthetic points) that can represent a set of similar points.

However prototypes alone may not lead to interpretable models as we still need an intuitive way to represent and explain what a prototype is, especially given recent deep prototype works BID17 .Both approaches were explored in healthcare applications.

For example, rule learning was employed to identify how likely patients were to be readmitted to a hospital after they had been released, each probability associated with a set of rules as criteria BID35 .

While prototype could be selected from actual patients and genes for clinicians to make sense of large patient cohort or gene data BID2 .

However, there are still open challenges: How to construct simple rules with more accurate prediction and classification performance?

How to produce accurate and intuitive definitions of prototypes?In this work, we propose Prototype lEArning via Rule List (PEARL), which combines rule learning and prototype learning on deep neural networks to harness the benefits of both approaches and alleviate their shortcomings for an accurate and interpretable prediction model.

In particular, we iteratively learn rule lists, via a data reweighing procedure using prototypes, and then update prototypes via neural networks with learned rules.

PEARL not only generates simple and interpretable rule lists and prototypes, but also provides neural network models which can infer the similarity of a query datum to all the prototypes.

To summarize, we make the following contributions in this paper.1.

We propose an integrative method to combine rule list and prototype learning, enabling PEARL to harness the power of these methods.2.

PEARL automatically learns prototypes corresponding to rules in a rule list, which are more concise than conventional rule list learning methods and more explainable than prototype learning methods by providing logic reasoning.3.

On real-world electronic health record datasets, PEARL demonstrates both accurate prediction performance and simple interpretation.

A prototype is an object that is representative of a set of similar instances (e.g., a patient from a cohort) and can be a part of the observed data points or an artifact summarizing a subset of them with similar characteristics.

Prototype learning is a type of case-based reasoning BID14 and aims to find some prototypes BID28 BID2 BID12 Prototypes can be seen as an alternative approach to learn centroids of clusters, and have been applied to few shot learning BID22 BID29 BID34 BID32 DISPLAYFORM0 be the data set, to learn one prototype p j of many prototypes, existing works choose some p j ??? X BID2 , compute a linear combination BID36 , or form a Bayesian generative model BID12 .

In this work, we follow BID17 and use a general representation of p j = f j (X), where f j is automatically learned via deep neural networks.

p j has the same dimension as the learned representation of data, which is a predefined hyperparameter.

Current prototype selection methods typically select one prototype at a time, and provide limited higher-level abstraction on the reasoning side of diagnosis.

DISPLAYFORM1 Rule lists are logic statements over original features.

A rule list R = (r 1 , r 2 , ..., r K , r 0 ) of length K is a (K +1)-tuple consisting of K distinct association rules, r k := z k ??? q k for k = 1, ..., K with an additional default rule r 0 .

Each rule r = z ??? q is an implication corresponding to the conditional statement, "if z, then q" where z is premise and q is conclusion.

In general, rule lists are easy to understand.

In this paper, each r i in R are dependent of previous rules with "else-if" logics.

We build on existing rule list learning method BID0 to iteratively guide the prototype learning via neural networks.

In addition, we use rule learning methods where each individual rule consists of logic AND clauses but not ORs.

Following existing definition of interpretability BID15 , there are 4 aspects of interpretability: size, length, cover, and overlap.

Size.

The size of a rule lists is defined as number of rules K in a rule list R. The fewer the rules in a rule list, the easier it is for a user to understand all of the conditions that correspond to a particular class.

Length.

We use the term length to measure the number of clauses in each rule r i .

If the number of clauses in a rule is too large, it will loose its natural interpretability.

Cover.

Cover measures the set of data points that satisfy each r i .

Cover measures how the data is divided by the rule classifiers.

Overlap.

Overlap between two rule r i and r j is the number of points that satisfy both rules.

It measures the discriminative power of each rule and whether decision boundary is clearly defined.

In this paper, we mainly investigate and provide new methods to reduce the size (by combining rule in rule lists into prototypes) and the length (by replacing the clauses in each rule using a prototype) of rule list classifiers, while improving on the accuracy of rule lists.

We propose to use the cover of each rule r i to re-weight data, which forces rule learning methods to focus on more discriminative data points and hence reduce the overlap among rules.3 PEARL: METHODOLOGY Let X = {X 1 , ?? ?? ?? , X N } be N data samples, where each sample X n (e.g., health records for patient n) is a sequences of discrete event labels (such as medical codes in electronic health records).

We can represent X n as {e i n , t i n }, where e i n ??? E is the i-th event label in X n and t i n is the time stamp of e i n .

For each X n , there is a class label y n .

For example, in health applications, y are the classification result of targeting diseases such as the onset of heart failure (binary), or subtypes of diabetes (multiclass).

The goal of PEARL is to accurately predict y = {y 1 , ?? ?? ?? , y N } and to provide explanation for such predictions.

In this work, we assume both X n and y n are categorical variables.

In this work, we aim to do so by providing an interpretable representation of data with a deep neural network.

The outputs of the network include the class label y and a set of interpretable prototype?? P corresponding to a rule list R. The neural network is used to performing accurate classification, under the guidance of prototypes defined by the rule lists.

Formally, the overall objective of PEARL is: DISPLAYFORM2 ( 1) where h(X ; ?? 1 ) is the learned representation of X with parameter ?? 1 .

h(X ; ?? 1 ) is a vector and has the same predefined dimension as p k , R is the learned rule list, and P is the set of learned prototypes.

A set of prototypesP = {p 1 , p 2 , ?? ?? ?? , p K } contains K representation of data, which serve as prototypes.

Here d is a distance measure, such as the cosine distance.

f is a fixed mapping that, given R and learned h(X ; ?? 1 ),P are determined without further learning.

More details on f can be found in Section 3.1.2.

Each p k lies in the same space as h(X ; ?? 1 ), and should correspond to one or more rules in R. The second term L 2 is the Cross Entropy loss for the final prediction target, where s R (h(X ; ?? 1 ); ?? 2 ) represents the predicted label for X and y is the ground-truth label.

Here ?? 1 represents all the model parameters for data representation learning h(X ; ?? 1 ) and ?? 2 represents those of classification model s R (??).

We will drop ??s for simplicity from now on.

Minimizing L 1 would encourage training examples to be as close as possible to at least one prototype in the latent space, motivated by BID17 .

However, we do not use other terms from BID17 and instead introduce rule lists as the guidance for prototype learning.

Note that relative weights ?? 1 and ?? 2 values are chosen via hyperparameter tuning.

In general we chose ?? 2 > ?? 1 to emphasize the classification performance.

Since it is non-trivial to integrate rule and neural network learning, we propose a framework, PEARL, of integrating rule learning and rule-guided prototype learning together.

The main intuition is to learn and produce prototypes that are closely related to rules in R, with one-to-one or many-to-one rule-prototype mapping.

This serves as a constraint to make each prototype as a surrogate for clauses in each rule, transforming "if data x satisfies z, then x = q" to "if x is close to a prototype p, then x = q".

We will discuss the network structures in details next.

DISPLAYFORM3 Rule list comprised of K rules, r 0 is the default rule X n = {e 1 n , t 1 n ; e 2 n , t 2 n ; ?? ?? ?? } Event sequence of subject n y; y n Labels for all data X ; One label for sequence X n L 1 ; L 2 Loss for prototype similarity; Cross-entropy loss for classification DISPLAYFORM4 Output of highway layer; Output of softmax layer o R (X) ??? R K Output of prototype layer, subscript R mean it rely on rule list.

r i ???

??? p i One prototype p i corresponds to a rule r i X ; X (j) Training subjects; Training subjects that satisfy rule r j

The network architecture of PEARL, illustrated in Fig. 1 , mainly comprises two modules: an interpretable module with a rule list learning procedure, and a prediction module with a prototype learning procedure.

The interpretable module generates a rule list given input data X and pass it to the prediction module.

The prediction module consists of a representation network and a prototype learning network.

The representation network is made of a temporal modeling component, followed by a highway network with skip connections to alleviate the numerical issue of vanishing gradients BID10 ).

The prototype learning network learns prototypes based on the rules from interpretation module and learned representation from the representation network, and then uses prototypes for the final prediction.

Moreover, the prediction module also re-weights data per distance to learned prototypes.

The re-weighted data is then used again for learning a new rule list and new data representation.

Figure 1: The PEARL architecture includes two modules: an interpretable module with a rule list learning procedure, and a prediction module with a prototype learning procedure.

Two modules iteratively improves each other during training.

Overall, the prediction module iteratively uses rule lists to guide the prototype learning via a neural network.

Then the interpretation module iteratively re-weights the data and updates its own rule learning.

The two modules are discussed in more details below.

The interpretable module employs a rule list classifier to provide interpretable prototype definitions.

Given data X n , we use a known rule list learning algorithm to generate a rule list R, with size |R|.

In general, any rule list algorithm can be adopted, and we choose one recent state-of-the-art COREL BID0 .

R is then used to help the prediction module to define and interpret prototypes.

If the feature size is too large, we can apply a feature selection algorithm to reduce feature dimension.

We tested a few feature selection algorithms in our experiments and they do not impact the performance much, if any at all.

We will discuss prediction module next and then discuss how interpretation module can benefit from the prediction module in an iterative data re-weighting procedure.

The prediction module contains a patient representation learning and a prototype learning network.

To encode patient longitudinal clinical events, we first embed the event sequences using neural networks.

Although we have flexible choices of neural networks, in this paper we chose the recurrent convolution neural networks (RCNN) BID18 to learn the distributed representations of each event.

In particular, we added one dimension filter and a max-pooling layer in the CNN part, and used a bidirectional LSTM for RNN.

This representation learning procedure for patient n is denoted as Eq. 2.

DISPLAYFORM0 where ?? k n is the time difference between consecutive events, such that ?? DISPLAYFORM1 for k > 1 and ?? 0 n = 0.

By including ?? k n as additional features, we incorporate the time information into patient representation learning.

After RCNN we also use highway network BID33 to alleviate the vanishing gradient issue in network training.

A single layer of highway network is: DISPLAYFORM2 where x and y are input and output for a single layer, respectively.

Here is element-wise multiplication, T is the transform gate, and the dimensionality of x, y, H(x, W H ), and T (x, W T ) are the same.

T and H use sigmoid and Relu as activation function, respectively.

Multiple layers highway network are concatenated.

Given g n as input of the first layer of highway networks, after multiple layers of updating, we represent the output of the n-th sample as h(X n ), which can be simplified as h(X n ) = Highway-Network(g n ).(4) Empirically we find the highway networks are essential for prototype qualities.

Data representation learning step is not limited to the combination or RCNN and highway network.

To generalize this representation learning step, we can write h(X n ) = Encoder-NN(X n ), which is the composite of Equation 2 and 4.Rule-guided Prototype Learning The embedded clinical events h(X n ) is then used in an iterative prototype learning procedure.

Specifically, we first generate prototype vectors from h(X n ).

Given a rule list R, |R| = K, for each rule r j ??? R, we can find all positive data samples for r j , denoted as X (j) .

Thus we can get a pseudo representation of r j : DISPLAYFORM3 where X (j) ??? X represent all the data samples that satisfy the j-th rule r j .

|X (j) | represents its cardinality.

The output of prototype learning network is a vector of one training subject's distance to all the prototypes, as given by Eq. 6.

DISPLAYFORM4 Here DISPLAYFORM5 , is the cosine distance of v 1 and v 2 .

The dimension of o(X) depends on the number of rules.

Since these prototypes use rules as guidance, we also call them rule-prototypes, in contract to non-rule prototypes in BID17 .

The subscript R means the function rely on rule list R.Last, a fully-connected layer (with parameter W ??? R K??L , where L is number of class) and a softmax activation are used to perform the final classification.

DISPLAYFORM6 where s R (X n ) is the estimated probability.

We then used the standard cross-entropy loss for training.

To enable the iterative learning of prototypes and rule list, we use a data re-weighting procedure based on results from the prediction module.

We first provide some intuition and then describe the detailed method.

Intuition Since learned prototypes are trained to represent spatially close data samples from the new learned feature space h(X), prototypes can be more discriminative and can reveal more of the underlying data similarity relationships than the rules from the original feature space as shown in the 2nd diagram of FIG0 .

With such a better similarity measure from the representation space, new representations of data samples are more easily separable.

More importantly, the examples that are difficult to separate may often be noises or low probability examples, i.e., if p(x, y) be the joint distribution of data, a hard-separable example x i has low p(x i , y i ).

Such a phenomenon has been observed previously in training simpler models BID8 ).

If we up-weight simpler samples that are more separable, rule-list learning focuses these simpler samples more and lead to easier training and more separation later.

For examples, the red dots shown in FIG0 are the highprobability examples, which should be given higher weights.

We will also empirically study data separation in experiments to justify this intuition.

Procedure The iterative learning and re-weighing procedure is based on the similarity between each data sample (such as patient subjects) and prototypes.

To start with, we measure the cosine similarity between subject h(X n ) and each prototype vector p k as depicted by Eq. 8.

DISPLAYFORM0 where d is cosine distance measure.

We aim at boosting the prototypes that have fewer subjects within its proximal neighbors in the learned representation space, indicating these prototypes are far away from other subjects and hence more discriminative.

Thus, for each prototype k, we calculate its average similarity with all subjects as s k = 1 N N n=1 s nk , where N is the size of the current dataset.

Then we collect those prototypes, denoted as K , of which s k is less than a pre-specified threshold ?? and their corresponding data subjects.

We concatenate these samples to the original data X. DISPLAYFORM1 where X (j) ??? X represent all the data samples that satisfy the j-th rule.

We summarize the procedure in Algorithm 1.

We alternately optimize rule list R and neural networks until convergence.

The convergence criteria is when the loss of the current epoch is within a pre-specified threshold from the previous epoch.

Data augmentation is equivalent to data weighting.

For practical purposes where the rule list cannot directly handle data weights, data augmentation can achieve desired results.

Inference Procedure for New Samples For a new subject X new = {e 1 new , e 2 new , ?? ?? ?? , }, PEARL will generate two outputs.

First is the predicted probability for classification, i.e., the output in softmax layer, s R (X new ) in Eq. 7.

Second, we obtain the output of prototype layer, i.e., o(X new ).

As it indicate the similarity between the current example and prototypes by their cosine distance, the new subject can be explained by the characteristics of its closest prototype.

A. Rule Learning: DISPLAYFORM2 If needed, apply feature selection to reduce feature dimensions.

c n is transformed into low-dimensionalc n = [0, 1, ?? ?? ?? ] .

Find rule R = {r 1 , r 2 , ?? ?? ?? , } based onc.

X (j) ??? X is set of all samples fit the rule r j .

B. prototype + NN training: Construct and train the neural network (Section 3.1).

C. Data Reweighing: DISPLAYFORM0 Compute all s nk , i.e., similarity between n-th data and k-th prototype.8:Collect all prototypes k ??? {1, ?? ?? ?? , K} that have less corresponding subjects ( DISPLAYFORM1 Reweigh data according to Eq. 9: We evaluate PEARL model by comparing against other baselines on two tasks: heart failure (HF) detection and mortality prediction.

All methods are implemented in PyTorch BID25 and trained on a laptop with 8GB memory.

DISPLAYFORM2 Dataset Description To evaluate the performance of PEARL, we conducted experiments using the following real world datasets.

The statistics of the datasets are summarized in TAB1 .Heart Failure (HF) Data:

The HF dataset is extracted from a proprietary EHR warehouse 1 where subjects were generally monitored over 4 years.

The HF cohort includes 2, 268 case patients and 14, 526 matching controls as defined by clinical experts.

Subject inclusion criteria is in Appendix.

BID11 .

We only included patients with at least two visits in our experiment, resulting in a total of 7,537 ICU patients.

Baselines We consider the following baseline algorithms.??? Rule learning: in this work we used the certifiably optimal rule lists in BID0 .???

Decision Tree: we directly use scikit BID26 package in Python.??? Prototype Learning (without rules) BID17 : RCNN+prototype (without rule).

Prototype is randomly initialized.

The result is very sensitive to the initialization.??? RNN (Doctor-AI) BID6 : RNN+softmax.

It concatenate multi-hot vector with a difference of time stamp as input feature.

A softmax layer is added after bi-LSTM.??? RCNN: CNN+RNN+softmax.

All RCNN use 1 dimensional filter, a max-pool layer and bi-LSTM.

It is followed by a softmax layer.

Evaluation Strategies We randomly split dataset 5 times and repeat the experiments with different random seeds.

For each split, we divide the dataset into training, validation and testing set in a 7 : 1 : 2 ratio.

Then we report the mean and standard deviation of results (both accuracy and run time).

To measure the prediction accuracy, we used the area under the receiver operating characteristic curve (ROC-AUC).

For rule learning, we report the average results of 5 trials.

After tuning, we set ?? 1 = 1 and ?? 2 = 1e???3.

To initialize embeddings, we use window size of 15 for word2vec BID23 and train medical code vectors of 100 dimensions on each training data, following BID21 .

For prototype learning, we use the same number of prototypes with PEARL to make sure that the parameter numbers are the same.

For the RNN model, we implemented a bidirectional-LSTM with hidden layer size 3.

For the RCNN model, the number of filters for CNN is 30, stride is 1, and the windows size is 1.

We add a max-pooling layer following convolution with pool size (5, 1).

For the highway network, the number of layers of highway network is set to 2.

Training is done through Adam (Kingma & Ba, 2014) at learning rate 1e-1.

The batch size is set to 256.

Data weighting threshold ?? is set to values between .45 and .55.

The threshold in convergence criteria is set as 0.001.

We fix the best model on the validation set within 5 epochs and report the performance in the test set.

TAB3 shows PEARL has the highest AUC performance among all methods.

As for the baseline models, the rule learning has the lowest AUC due to it makes classification based on composition of simple logics.

Prototype learning is better than rule learning and RNN models but worse than PEARL.

It shows PEARL can improve upon both prototype and rule learning.

BID17 .

BID6 .

Interpretability-Accuracy Tradeoff We study the relationship between accuracy and the interpretability in rule list learning and the proposed PEARL model.

Interpretability is measured by the number of rules (also the number of prototypes) of different methods.

FIG2 shows that our method can use a small number of prototypes to achieve better accuracy than the rule list learning.

In fact, 3 rule-prototypes can already explain more samples than rule lists with over 50 rules.

To justify reweighing data points lead to more separable prototypes, we study on whether the mean distances DISPLAYFORM0 between prototypes and data decreases over training iteration T i .

As shown in FIG3 , by using data weighing, the average distance decreases across iteration T i .

Interestingly, even with each iteration, the average distance also decreases with training epochs of neural networks, suggesting that such a reduction in average distances also leads to lower training loss.

We conducted further experiments to test the accuracy change of rule lists during training, which shows that data augmentation helps improving rule list accuracy, along with more hyper-parameter tuning results.

All the results are shown in the appendix.

We study whether PEARL can provide more interpretable diagnosis compared with conventional rule learning.

In particular, we find the corresponding prototypes learned in PEARL for a sets of patients and retrieve the closest rule-prototypes.

For each prototype with multiple patients, we retrieve their high frequent events among the patients who satisfy the rule-prototype while the remaining events that only occur to one or two patients are discarded.

In general, the rule learning often yields complex rule lists that involve hundreds of clinical events, many of which are duplicated in multiple rules.

As a contrast, PEARL only used ??? 10 rules to make correct diagnosis.

Below we provide one example of prototype-rules from PEARL.If a patient experience all following events: (1) chronic airways obstruction, (2) malignant neoplasm of trachea, lung and bronchus, (3) carcinoma in situ of respiratory system, (4) Alprazolam, (5) Eszopiclone, (6) abnormal findings on radiological examination of body structure, (7) acute bronchitis, (8) Albuterol Sulfate, (9) Hypertrophic conditions of skin, and (10) diltiazem hydrochloride, then the patient has a high probability of experiencing heart failure.

The prototype-rules include 10 clinical events.

Most of them concern severe conditions of lung and respiratory systems (a common symptom of HF patients), and the medications for treating HF, which are common comorbidities of heart failures.

Patients belong to this prototype can be diagnosed based on the occurrence of these events on their EHR.

For patients of this prototype, if using conventional rule learning, diagnosis would require a much more complex rule with more than 40 clinical events and rule depth for about 50.

We provide one example in A.2 in Appendix.

In this paper, we proposed PEARL, an integrative prototype learning neural network that combines rule learning and prototype learning on deep neural networks to harness the benefits of these methods.

We empirically demonstrated that PEARL is more accurate , thanks to an iterative data reweighing algorithm, and more interpretable than rule learning, since it explains diagnostic decisions using much fewer clinical variables.

PEARL is an initial attempt to combine traditional rule learning with deep neural networks.

In future research, we will try to extend PEARL to other interpretable models.

A.1 INCLUSION CRITERIA FOR HEART FAILURE DATA The criteria for being patients include 1) ICD-9 diagnosis of heart failure appeared in the EHR for two outpatient encounters, indicating consistency in clinical assessment, and 2) At least one medication was prescribed with an associated ICD-9 diagnosis of heart failure.

The diagnosis date was defined as its first appearance in the record.

These criteria have also been previously validated as part of Geisinger Clinical involvement in a Centers for Medicare and Medicaid Services (CMS) pay-for-performance pilot BID27 ).

For matching controls, a primary care patient was eligible as a control patient if they are not in the case list, and had the same gender and age (within 5 years) and the same PCP as the case patient.

More details could be found in BID37 .

Here, we study the accuracy of rule during different epochs in Algorithm 1.

We conduct 5 independent trials using different hyperparameter and report their average results.

The results are shown in FIG4 .

We can find that the accuracy of rules increase with iterative learning and we conclude that the data augmentation does improve the accuracy of rule list learning as well.

To study the performance improvement of prototype learning due to impacts of rules, we compare the empirical effect of non-rule prototypes and rule prototypes.

As shown in FIG6 , we found that more rule-prototypes can yield better accuracy in general, which shows learned rule-prototypes are better than non-rule prototypes.

We then study the empirical effect of data reweighing threshold ?? in Algorithm 1, where ?? controls the number of prototypes from which the corresponding subjects are up-weighted.

From FIG5 , we find that higher threshold usually corresponds to better accuracy.

We added additional evaluation using "cars" and "breast" data from UCI data repository.

For "cars" dataset, falling rule reach 80% classification accuracy, while PEARL reaches only 93% classification accuracy, both using 13 rules.

Pure NN method achieve 95% classification accuracy.

For "breast" dataset, falling rule method reach 85% classification accuracy while PEARL reaches 92%, both using 17 rules.

Pure NN also reach 92%.

The maximal number of rules from rule learning is not set in advance; it is determined by data.

In most cases, it produce at most 60 rules.

We conduct experiment to study the average distance between data sample and prototype at the convergence state.

The average distance is 0.643 for the method in BID17 while for PEARL, the distance is 0.432, indicating that PEARL shows better clustering property than random prototype.

<|TLDR|>

@highlight

a method combining rule list learning and prototype learning 

@highlight

Presents a new interpretable prediction framework, which combines rule based learning, prototype learning, and NNs, that is particularly applicable to longitudinal data.

@highlight

This paper aims at tackling the lack of interpretability of deep learning models, and propose Prototype lEArning via Rule Lists (PEARL), which combines rule learning and prototype learning to achieve more accurate classification and makes the task of interpretability simpler.