Recent advances in computing technology and sensor design have made it easier to collect longitudinal or time series data from patients, resulting in a gigantic amount of available medical data.

Most of the medical time series lack annotations or even when the annotations are available they could be subjective and prone to human errors.

Earlier works have developed natural language processing techniques to extract concept annotations and/or clinical narratives from doctor notes.

However, these approaches are slow and do not use the accompanying medical time series data.

To address this issue, we introduce the problem of concept annotation for the medical time series data, i.e., the task of predicting and localizing medical concepts by using the time series data as input.

We propose Relational Multi-Instance Learning (RMIL) - a deep Multi Instance Learning framework based on recurrent neural networks, which uses pooling functions and attention mechanisms for the concept annotation tasks.

Empirical results on medical datasets show that our proposed models outperform various multi-instance learning models.

Clinicians have limited time (e.g., only a few minutes BID31 ) to study and treat each patient.

However, they are overloaded with a lot of patient data from multiple sources and in various formats, such as patient medical history and doctor's notes in free-flowing text, vitals and monitoring data which are captured as time series, and prescriptions and drugs which appear as medical codes including ICD-9 (Organization & Corporation, 1998) , LOINC codes BID21 , etc.

This rich information should be summarized and available to clinicians in easily digestible format for faster diagnosis and treatment.

Graphical visualizations BID54 are a popular approach to show patient data to doctors.

However, recent studies have shown that graphical visualisations are not always helpful for clinicians' decision-making BID46 BID63 .

Text summaries on the other hand are widely embraced and are usually adopted in practice BID58 .

Most existing systems use natural language processing techniques (Afantenos et al., 2005; BID23 to generate summaries from doctor notes which include test results, discharge reports, observational notes, etc.

While these systems are useful, they only use one source of data, i.e., doctor's notes which might have noisy and erroneous entries, for text summarization.

On the other hand, electronic health records have other sources of patient data such as vital signs, monitoring sensors, and lab results in the form of multivariate time-series, which can be more accurate and may contain rich information about patient's conditions.

Few existing patient summarization systems actually extract information directly from these time series for concept prediction and/or summarization.

Generating simple text summaries such as trends from time series has been investigated before BID61 but is marginally useful since these trends are not mapped to the medical concepts which clinicians can quickly comprehend.

Recent works BID52 BID16 BID48 BID14 have successfully shown that clinical events and outcomes can be predicted using medical codes or clinical time series data.

However, directly obtaining medical concept annotations and summaries from the time series data is still an open question.

In this work, we introduce the concept annotation task as the problem of predicting and localizing the medical concepts by modeling the related medical time series data.

FIG0 illustrates a concept annotation example where medical time series data such as heart rate, pH and blood gas pressure are given, and the goal is to predict the time series of concepts such as intubation, extubation and resuscitate.

To solve concept annotation problem, we formulate it as a Multi-Instance Learning (MIL) problem BID20 and propose a deep learning based framework called Relational MultiInstance Learning (RMIL).

RMIL uses Recurrent Neural Networks (RNNs) to model multivariate time series data, leverages instance relations via attention mechanisms, and provides concept predictions using pooling functions.

The main contributions of our work are the following.

We present a unified view of the MIL approaches for time series data using RNN models with different pooling functions and attention mechanisms.

We show that our RMIL model is capable of learning a good classifier for concept detection (bag label predictions) and concept localization tasks (instance label prediction), even though it is only trained using bag labels.

We demonstrate that RMIL obtains promising results on real-world medical datasets and outperforms popular MIL approaches.

The rest of the paper is structured as follows.

In the following section, we briefly discuss the related works.

Afterwards, we describe MIL framework and describe how RNN can be combined with multi-instance learning framework to obtain our proposed RMIL.

In Sections 4 and 5, we present experimental results and conclusions respectively.

In the appendix, we demonstrate anomaly detection as another application of our RMIL framework.

Discovering concept annotations from the multivariate time series is a relatively new problem in medical domain with limited prior work.

In this section, we will first highlight the related works on annotation tasks and then review related works on multi-instance learning.

Concept Annotation In medical domain, concept annotation is usually addressed in the clinical narrative mining and biomedical text mining literature BID1 BID18 BID78 BID64 .

In other domains such as web-mining and computer vision, the concept annotation is usually analogous to semantic annotation BID39 , image annotation BID33 , object localization BID45 and image captioning BID36 .Clinical Narratives Mining Automated discovery of temporal relations from clinical narratives BID57 BID74 BID2 and doctor notes BID53 to uncover the patterns of disease progression is an important research problem in clinical informatics.

Recent efforts such as SemEval competitions BID6 have been conducted to study this problem and evaluate/benchmark clinical information extraction systems BID69 .

These competitions focus on discrete, well-defined tasks which allow for rapid, reliable and repeatable evaluations.

However, they only consider identifying and extracting temporal relations from clinical notes and do not use the accompanying medical time series data.

Image Annotation and Captioning Successful object recognition systems have been developed in the past few decades for image annotation, object detection and localization in images and videos.

ImageNet BID19 ) and PASCAL challenges BID20 have greatly accelerated the research in this area.

Image captioning and visual-to-text translation, which are more generalized image annotation tasks, have been recently studied in several works BID36 BID40 BID50 BID71 where the goal is to find a text caption for a given image.

Deep learning models such as RNN and sequence-to-sequence models BID62 have achieved excellent results for image annotation/captioning tasks.

Multi-Instance Learning Multi-Instance Learning (MIL), a well known researched topic in machine learning, was first introduced by BID20 as a form of weakly supervised learning for drug activity prediction.

MIL frameworks have since been applied to many other domains including image and text annotations BID15 BID60 .

BID4 adapted Support Vector Machines (SVM) to the MIL framework and introduced miSVM and MISVM for optimizing instance-level and bag-level classifications respectively.

BID75 further extended MIL and proposed MIMLSVM for tackling multi-label problems.

BID76 introduced miGraph and MIGraph to model the structure in each bag.

BID73 BID25 also proposed MIL framework for structure data by leveraging the relational structures at the bag and instance levels.

Generative model based MIL frameworks such as Multi-Instance Mixture Model (MIMM) BID22 , and Dirichlet Process Mixture of Guassians (DPMIL) BID35 have also been proposed for binary multi-instance classification.

BID25 used an autoregressive hidden Markov model and proposed an MIL framework for activity recognition in time-series data.

Garcez & Zaverucha (2012) used recurrent neural networks to combine instance-level preprocessing and bag-level classification in MIL setting.

Comprehensive reviews of MIL approaches are provided in BID3 ; BID28 ; BID60 .

Recently, deep learning models have been successfully applied for MIL framework BID77 BID68 BID32 BID41 BID42 and these approaches are generally termed as deep multi-instance learning models.

Most of these works use either convolutional neural networks or deep neural networks in their MIL framework for image annotation, labeling, segmentation or classification tasks.

Despite the popularity of deep models for MIL, there are few works which have extended deep MIL models for multivariate time series data.

The goal of this paper is to propose and study deep multi-instance learning models for multivariate time series data.

In this section, we will first describe the Multi-instance learning framework, and then present our problem formulation and our proposed relational multi-instance learning models.

Multi-instance learning (MIL) is a form of weakly supervised learning where the training data is arranged in sets called bags, and a label is provided for the entire bag.

The data points inside a bag are referred to as instances.

In the MIL framework, instance labels are not provided during training.

The main goal of MIL is to learn a model based on the instances in the bag and the label of the bagto make bag-level and instance-level predictions.

In this work, we only focus on the classification task in MIL, leaving out other learning tasks such as regression.

Generally, two broad assumptions can be used to model the relationship between instance label and bag label.

In the standard MIL assumption BID20 , the bag label is negative if all the instances in the bag have a negative label, and the bag label is positive if at-least one of the instances in the bag has a positive label.

Following the notations of BID10 , let X denote a bag with N feature vector instances i.e., X = {x 1 , ..., x N }.

Let each instance x i in feature space X be mapped to a class by some process f : X → {0, 1}, where 0 and 1 correspond to negative and positive labels respectively.

The bag classifier, also know as the aggregator function, g(X) is defined by: DISPLAYFORM0 The standard assumption is quite restrictive for some problem settings, where the positive bags cannot be identified by a single instance.

Thus, this assumption can be relaxed to a collective assumption which says that several positive instances in a bag are necessary to assign a positive label to that bag.

In this case, a bag classifier is given by: DISPLAYFORM1 where θ is a threshold which indicates the minimum number of instances with positive labels that should be present in a bag to assign a positive label to that bag BID66 .

As discussed in section 2, a plethora of works have adapted machine learning models to the MIL setting to optimize instance-level and/or bag-level predictions.

We formulate the concept annotation task as the detection and localization of concepts given the medical time series data.

Let each patient i ∈ {1, .., N } be associated with a medical time series (also referred to as feature time series) denoted by X i ∈ R T ×D , where D denotes the number of features (such as heart rate, blood pressure) and T denotes the length of time series observations (i.e., amount of time a patient is monitored).

Let C denote the set of all the concepts associated with the N patients, and Y i ∈ {0, 1} K denote the concepts associated with i th patient where DISPLAYFORM0 T ×K denote the concept time series of X i with C jk i = 1 when concept k is present at time-stamp j. In multi-instance learning settings, we treat each time series X i as one bag, and the observation at each time step j i.e. X j i ∈ R D as an instance in that bag.

We are interested in the following tasks: DISPLAYFORM1 Notice that during training phase, only the input X i and prediction label Y i are available.

Though C i is not known, we usually have some assumptions about the relationship of prediction and localization labels.

In this work, DISPLAYFORM2 where I is an indicator function and η is a constant which depends on the MIL assumption.

For example, in our concept annotation tasks we make the standard assumption, i.e we assume η = 1, i.e. the time series label (bag label) for a concept is positive if that concept is present at any one time-stamp (at-least one instance has positive label).

Inspired by the recent success of recurrent neural networks in sequence modeling BID5 ; BID62 and classification tasks BID43 ; BID59 , we adapt these models to the MIL framework to model multivariate time series data for concept annotation tasks.

We denote all the variables at every time step as an instance and the entire multivariate time series as a bag.

Unlike the traditional MIL setting, where the instances within a bag are independent of each other, in our case, the instances have relationships (namely temporal dependencies) among them.

To model these dependencies, we propose to combine RNN models such as Long-Short Term Memory Neural Networks (LSTM) and Sequence-to-Sequence models with MIL, and propose our Relational Multi-Instance Learning framework, abbreviated as RMIL.

RMIL takes in multivariate time series as input and outputs concept annotations.

In RMIL, the outputs of RNN model provide the instance label predictions (i.e. solution for concept localization task) and the aggregation of the instance labels using aggregators such as pooling layer provides the bag label predictions (i.e. solution for concept prediction task).

We propose different pooling functions and attention mechanisms which can be easily incorporated into our RMIL to improve the concept annotations.

Pooling Layers for RMIL The bag-level prediction is obtained by using an aggregation gathering on all instance-level predictions.

The aggregator function g(·) : [0, 1] T → [0, 1] in RMIL can be modeled using the pooling layers.

Without loss of generality we assume that RNN model computes a mapping from the feature time series to the concept time series for each of the concept k ∈ C. Let us denote the probability of an instance j belonging to concept k as p jk .

Then, the bag level probability for a concept k is given by P k = g(p 1k , p 2k , . . .

, p T k ).

The role of aggregator function g(·) is to combine the instance probabilities from each class specific feature map {p jk } into a single bag probability P k .

Several pooling mechanisms shown in TAB0 have been introduced in MIL and deep learning literature which can be used in our RMIL.

In TAB0 , r, a, b k , and r k are parameters which can be fixed or are learned during training, and σ(·) denotes the sigmoid function.

Noisy-OR pooling BID72 P DISPLAYFORM0 Integrated segmentation and recognition (ISR) (Keeler et al., 1991) DISPLAYFORM1 Noisy-AND pooling BID42 ) DISPLAYFORM2 Attention Mechanism for RMIL Instances within each bag have temporal relations between them.

We can use attention mechanism to focus on some of the instances and their relations to improve their instance-level predictions.

In order to make predictions at time j, the hidden state h j ∈ R Q of RNN can be used, where Q is the hidden state dimension.

However, relevant information may be captured by hidden states at other time steps as well.

Thus, we may want to introduce an attention vector or matrix (a) to leverage information of hidden states H = (h 1 , · · · , h T ) ∈ R T ×Q from all time steps.

Let us denote the output after attention asH ∈ R T ×Q .

The attention matrix can then be modeled usingH in various ways as listed below.

Feature-based Attention One idea is to design the attention matrix based on the feature and its time-stamp.

Let us define a feature-based attention matrix as A = (a 1 , · · · , a T ) ∈ R T ×Q .

For each j = 1, · · · , T , we have DISPLAYFORM3 where and are element-wise multiplication and division, respectively and W = (w 1 , · · · , w T ) ∈ R T ×T is the weight matrix which can be learned during training.

We call this Attention-F mechanism.

We can simplify this attention by averaging the attentions for all hidden dimensions by taking a jq ← 1/Q · 1≤q ≤Q a jq .

We will denote this as Attention-FS mechanism.

Time-based Attention Attention model BID49 can be designed to capture the relation between the current time step j and previous time steps j ≤ j, by solely relying on previous hidden states h j .

We can define a time-based attention matrix as A ∈ R T ×T .

For each j and j in [1, · · · , T ], we have DISPLAYFORM4 0, otherwise.

andH = A · H, where w ∈ R D is the weight vector to learn.

We use Attention-T to represent Time-based attention mechanism.

Interaction-based Attention The time-based attention can be further improved by considering both the previous and current hidden states h j and h j BID49 .

In this case, we have DISPLAYFORM5 0, otherwise and similarlyH = A · H. Here, we need to learn v ∈ R S , W 1 ∈ R S×Q , W 2 ∈ R S×Q , and we choose S = Q/2.

A simplified version of interaction-based attention can be obtained if we use vector w 1 ∈ R Q , w 2 ∈ R Q instead of matrices W 1 , W 2 and by setting v = 1 in the above equation.

We use Attention-I and Attention-IS to represent Interaction-based attention mechanism and simplified version of interaction-based attention mechanism respectively.

The above attention mechanisms usually help both prediction and localization tasks.

Here, we demonstrate the performance of our proposed RMIL models on concept annotation tasks i.e. concept prediction and localization tasks, using a real-world health-care dataset and compare its performance to the popular multi-instance learning approaches.

In addition, we discuss the impact of using pooling functions and attention mechanism in our RMIL framework.

To evaluate our RMIL, we ran experiments on MIMIC-III RESP datasets whose statistics is shown in TAB1 .

BID38 at the time of admission.

These 21 features are respiratory based features such as peak inspiratory pressure (PIP) and arterial partial pressure of oxygen (PaO2) and were collected during the first 3 days after admission.

The feature time series has 4 time stamps and the first time stamp corresponds to the admission time.

We denote this dataset as MIMIC-III RESP dataset.

In addition, we also generated another feature time series with more time stamps whose results is shown in the appendix.

The medical time series data of MIMIC-III dataset does not come with the concept annotations, however the medical concepts are available in the doctor notes of the MIMIC-III database.

To obtain the concept annotations, we extract the concept time series from the doctor notes using the NOTEEVENTS table of MIMIC-III database.

The total number of doctor notes is 2,083,180, out of which 98.15% of notes (2,044,634 notes) have no timestamp and 1.85% of notes (38,546 notes) have timestamps associated with them.

The total number of unique concepts in the doctor notes in the first 3 days data is 6,197.

To obtain concept time series for each patient with respiratory disorder such as AHRF, we first identified respiratory-related concepts from the medical literature BID4 BID38 , and obtained their medical codes from the Unified Medical Language System (UMLS) dictionary BID7 .

Then, we mined the patient's doctor notes from NOTEEVENTS table to extract all the possible medical concepts related to the respiratory system and its disorder.

In total, we chose top 26 respiratory concepts to generate concept time series which has the same number of time stamps as feature time series.

We compare the performance of our proposed models to the popular MIL models such as MISVM BID4 , DPMIL BID35 and Convolutional Neural Networks (CNN) BID42 , and CNN with attention.

We categorize all the evaluated methods into two groups: For LSTM models, we use two LSTM layers and two dense layers.

For S2S models, we use two LSTM layers for both the encoder and the decoder.

For Bi-LSTM models, we use two bi-directional LSTM layers.

All the models were constructed to have a comparable number of parameters.

We train all the Deep learning models with the RMSProp optimization method and we use early stopping to find the best weights on the validation dataset.

For baseline MIL models, we follow the suggestions of the corresponding papers to fine-tune the parameters.

All the input variables in the training data are normalized to be 0 mean and 1 standard deviation.

The inputs to all the models is the same feature time series data.

We used Keras (Chollet, 2017) and Python to run the deep models and MISVM models.

Matlab code from the original authors was used to obtain DPMIL results.

We use the area under ROC (AUROC) and area under precision-recall curve (AUPRC) scores as our evaluation metrics and report the results from 5-fold cross validation for all the evaluated methods.

TAB3 shows the concept annotation results on the MIMIC-III RESP dataset.

From this table, we see that RMIL models outperform the non-deep multi-instance learning models by at least 8-10% for concept localization task, and by at least 10-15% for concept prediction tasks in terms of AUROC and AUPRC.

RMIL performs slightly better than CNN-based models on all the metrics.

Among all the RMIL models, we find that LSTM model obtains slightly better overall results compared to the other models for localization task.

DISPLAYFORM0

To study the impact of pooling and attention, we trained and evaluated LSTM models with different pooling functions and different attention mechanisms, which are described in Section 3.

TAB4 show the comparison results.

From these tables, we observed that (i) all the attention mechanisms except feature-based attention perform similar to each other especially for the prediction task, and (ii) all the pooling functions other than ISR and Noisy-OR obtain similar overall performance.

This demonstrates that choice of attention does not matter but choice of pooling has some impact in our RMIL framework.

We can study the interpretability of concept localization by looking at the localization results of our RMIL models, even though the model is trained without the labels for localization.

FIG3 shows the ground truth annotations of two respiratory concepts -intubation and extubation concepts, and the prediction probabilities of these concepts obtained by our RMIL attention-based LSTM models.

From figure 2(a) we can make the following observations, (i) intubation usually happens before extubation for the same patient, (ii) intubation and extubation could happen on the same day, and (iii) intubation and extubation occur commonly within the first 24 hours of admission.

From the figure 2(b), we see that our RMIL attention based LSTM predicts that the probability of intubation happening on the first day of admission is higher (draker gray means higher probability of concept occurrence) and the probability of extubation happening within first day is lower.

This indicates that the model has correctly learnt that intubation should appear before extubation.

This also implicitly implies that the RMIL attention-based LSTM models have correctly learnt the instance-level relationships from the medical time series data with only bag-level labels.

In this paper, we presented Relational Multi-Instance Learning -a deep multi-instance learning framework using recurrent neural networks for concept annotation from the medical time series data.

Empirical results on medical dataset demonstrated that our proposed models outperform the popular state-of-the-art multi-instance learning approaches.

Experiments with different pooling and attention mechanisms showed that while attention mechanism does not have a significant impact on model's performance, certain pooling functions such as ISR and Noisy-OR can negatively impact the instance prediction results.

6.1 MIMIC-III CONCEPT LIST TAB6 show the concept list of the MIMIC-III RESP dataset used in our experiments.

Here, we will demonstrate anomaly detection from medical time series data as a use case application of our RMIL framework.

Each year, more than 1,000,000 US adults and children are put on mechanical ventilation during their stays in ICU.

However, lack of effective tools to aid with ventilator weaning and extubation (removal of the breathing tube) readiness assessment results in nearly half of the patients spending unnecessary days on ventilators BID56 , and up to 20% of them having ventilators discontinued too soon BID44 .

Spending unnecessary days on ventilators can lead to hospital-acquired infections while having ventilators discontinued too soon could require painful reintubation.

A work-of-breathing measure called Pressure-Rate Product (PRP), calculated from esophageal pressure, has shown potential to be used as a guideline for ventilator weaning and extubation readiness assessment BID67 .

However, PRP calculations are susceptible to sensor artifacts and breathing pattern anomalies.

These anomalies limit realtime use of PRP for clinical decision making.

Our goal is to automatically detect and remove these anomalies by using our proposed Relational Multi-instance learning models, thereby enabling real-time clinical decision making.

FIG4 shows the example of anomaly detection from the ventilator time series data.

Here, anomalies appear due to patient factors (cough, movement) and instrument factors (probing, catherter drift), and should be automatically detected from the ventalitor monitoring time series data.

There is a long history and a rich body of research work on anomaly detection in time series data.

See BID30 BID12 for a quick survey on generic anomaly detection algorithms.

BID27 also author a survey on anomaly detection for time series data.

In their follow-up survey, BID13 gave a summary on discrete sequence anomaly detection algorithms.

Unfortunately, techniques for time series anomaly detection are quite domain-specific.

This is due to highly-varied nature of time series characteristics.

In BID11 , a set of k minimum bounding rectangles between each time step is used as a model of normal data generating distribution.

In BID8 ; BID47 , continuous time series are converted to ordinal symbolic representation, allowing faster approximation of Euclidean distance between time series windows and early pruning.

Then, anomaly detection can be performed by setting a threshold on distance value.

Recently, proposes a self-learning method that learns clusters of constrained grammars based on ordinal symbolic approximation of time series value.

BID26 proposes incorporating experts into anomaly detection in a method inspired by immune system.

In BID34 , the authors propose extracting exemplars from Euclidean pairwise distance as a way to speed up anomaly detection algorithm.

Most of these techniques for time series anomaly detection are quite domain-specific, and few of them model the anomaly detection problem in Multi-instance learning setting.

We formulate the anomaly detection as concept annotation problem in MIL setting, where the bag corresponds to the medical time series data X i and an instance is the features at a time-stamp, and the anomaly corresponds to a concept.

Thus, the prediction (predicting Y i ) and localizaton tasks (predicting C i ) correspond to predicting the presence and location of the anomalies in the time series data.

For anomaly detection, we assume DISPLAYFORM0 where η is chosen as η = 0.6T and T is the number of time series windows within the bag.

We conduct anomaly detection experiments on a PICU dataset described below.

TAB7 .

The recordings are made on mechanically-ventilated patients in pediatrics ICU ward.

Medical time series data collected from four sensors are used: flow volume spirometry, esophageal pressure sensor, and dual band respiratory inductance plethysmography.

Each subject can be under one of four breathing conditions: ventilated with Continuous Positive Airway Pressure (CPAP), ventilated with Pressure Support (PS), 5 minutes after extubation, and 60 minutes after extubation.

Along with the 4 sensor signals, clinicians verified binary anomaly label generated using hard-crafted state-of-the-art breathing anomaly detection algorithm is provided as ground truth.

We annotate the concept of breathing anomaly in MIL framework by processing the dataset as follows.

The sensor recordings are split into non-overlapping 5-second windows.

In MIL framework, each window becomes an instance and each recording becomes a bag.

For each sensor signal of each window, we extract 20 Mel-Frequency Cepstral Coefficients (MFCCs) to be used as features of each instance; thus, each instance has 80 features in total.

For each instance, its anomaly label is set to positive if at least 20% of the window are labeled as anomaly.

The bag anomaly label is set to positive if at least 60% of its instances are labeled as anomaly.

Results TAB8 shows the anomaly detection results using our RMIL models.

We observe that (i) RMIL models mostly outperform the non-deep MIL models on both tasks, (ii) LSTM and Bi-LSTM based RMIL models obtain better overall results compared to the sequence-to-sequence models, and (iii) the attention mechanism does not help for localization task but sometimes obtains better results for the prediction task.

Remark: Cluster-MIL* is a HDBSCAN clustering BID9 based approach for anomaly detection.

It uses both instance and bag labels for training, while other models only use bag labels for training.

We sampled the feature time series from MIMIC-III RESP dataset every 6 hours and generated a time series with more (12) time stamps.

We call this MIMIC-III RESP-II dataset.

TAB9 shows the results of RMIL models on this dataset.

We observe that (i) All the RMIL models have similar performance for prediction task, (ii) Bi-LSTM RMIL has better localization results compared to other the models.

<|TLDR|>

@highlight

We propose a deep Multi Instance Learning framework based on recurrent neural networks which uses pooling functions and attention mechanisms for the concept annotation tasks.

@highlight

The paper addresses the classification of medical time-series data and proposes to model the temporal relationship between the instances in each series using a recurrent neural network architecture. 

@highlight

Proposes a novel Multiple Instance Learning (MIL) formulation called Relation MIL (RMIL), and discussed a number of its variants with LSTM, Bi-LSTM, S2S, etc. and explores integrating RMIL with various attention mechanisms, and demonstrates its usage on medical concept prediction from time series data. 