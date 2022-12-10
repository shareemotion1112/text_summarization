Monitoring patients in ICU is a challenging and high-cost task.

Hence, predicting the condition of patients during their ICU stay can help provide better acute care and plan the hospital's resources.

There has been continuous progress in machine learning research for ICU management, and most of this work has focused on using time series signals recorded by ICU instruments.

In our work, we show that adding clinical notes as another modality improves the performance of the model for three benchmark tasks: in-hospital mortality prediction, modeling decompensation, and length of stay forecasting that play an important role in ICU management.

While the time-series data is measured at regular intervals, doctor notes are charted at irregular times, making it challenging to model them together.

We propose a method to model them jointly, achieving considerable improvement across benchmark tasks over baseline time-series model.

With the advancement of medical technology, patients admitted into the intensive care unit (ICU) are monitored by different instruments on their bedside, which measure different vital signals about patient's health.

During their stay, doctors visit the patient intermittently for check-ups and make clinical notes about the patient's health and physiological progress.

These notes can be perceived as summarized expert knowledge about the patient's state.

All these data about instrument readings, procedures, lab events, and clinical notes are recorded for reference.

Availability of ICU data and enormous progress in machine learning have opened up new possibilities for health care research.

Monitoring patients in ICU is a challenging and high-cost task.

Hence, predicting the condition of patients during their ICU stay can help plan better resource usage for patients that need it most in a cost-effective way.

Prior works (Harutyunyan et al., 2017; BID4 BID18 BID16 BID1 have focused exclusively on modeling the problem using the time series signals from medical instruments.

Expert knowledge from doctor's notes has been ignored in the literature.

In this work, we use clinical notes in addition to the time-series data for improved prediction on benchmark ICU management tasks (Harutyunyan et al., 2017) .

While the time-series data is measured continuously, the doctor notes are charted at intermittent times.

This creates a new challenge to model continuous time series and discrete time note events jointly.

We propose such a multi-modal deep neural network that comprises of recurrent units for the time-series and convolution network for the clinical notes.

We demonstrate that adding clinical notes improves the AUC-PR scores on in-hospital mortality prediction (+7.8%) and modeling decompensation (+6.1%), and kappa score on length of stay forecasting (+3.4%).

Here we formally define the problems and provide a review of machine learning approaches for clinical prediction tasks.

Problem Definitions.

We use the definitions of the benchmark tasks defined by Harutyunyan et al. (2017) as the following three problems: 1.

In-hospital Mortality: This is a binary classification problem to predict whether a patient dies before being discharged from the first two days of ICU data.2.

Decompensation: Focus is to detect patients who are physiologically declining.

Decompensation is defined as a sequential prediction task where the model has to predict at each hour after ICU admission.

Target at each hour is to predict the mortality of the patient within a 24 hour time window.

The benchmark defines LOS as a prediction of bucketed remaining ICU stay with a multiclass classification problem.

Remaining ICU stay time is discretized into 10 buckets: {0 − 1, 1 − 2, 2 − 3, 3 − 4, 4 − 5, 5 − 6, 6 − 7, 7 − 8, 8 − 14, 14+} days where first bucket, covers the patients staying for less than a day (24 hours) in ICU and so on.

This is only done for the patients that did not die in ICU.These tasks have been identified as key performance indicators of models that can be beneficial in ICU management in the literature.

Most of the recent work has focused on using RNN to model the temporal dependency of the instrument time series signals for these tasks (Harutyunyan et al. (2017) , BID16 ).

Texts.

Biomedical text is traditionally studied using SVM models BID14 with ngrams, bag-of-words, and semantic features.

The recent development in deep learning based techniques for NLP is adapted for clinical notes.

Convolutional neural networks is used to predict ICD-10 codes from clinical texts BID12 BID9 .

BID15 ; BID0 used convolutional neural networks to classify various biomedical articles.

Pretrained word and sentence embeddings have also shown good results for sentence similarity tasks BID2 .

However, none of these works have utilized doctor notes for ICU clinical prediction tasks.

Multi Modal Learning.

Multi-modal learning has shown success in speech, natural language and computer vision BID13 , Srivastava and Salakhutdinov (2012), BID10 ).

In health care research, BID19 accommodated supplemental information like diagnosis, medications, lab events etc to improve model performance.

In this section, we describe the models used in this study.

We start by introducing the notations used, then describe the baseline architecture, and finally present our proposed multimodal network.

For a patient's length of ICU stay of T hours, we have time series observations, x t at each time step t (1 hour interval) measured by instruments along with doctor's note n i recorded at irregular time stamps.

Formally, for each patient's ICU stay, we have time series data [x t ] T t=1 of length T , and K doctor notes DISPLAYFORM0 , where K is generally much smaller than T .

For in-hospital mortality prediction, m is a binary label at t = 48 hours, which indicates whether the person dies in ICU before being discharged.

For decompensation prediction performed hourly, DISPLAYFORM1 are the binary labels at each time step t, which indicates whether the person dies in ICU within the next 24 hours.

For LOS forecasting also performed hourly, [l t ] T t=5 are multi-class labels defined by buckets of the remaining length of stay of the patient in ICU.

Finally, we denote N T as the concatenated doctor's note during the ICU stay of the patient (i.e.,, from t = 1 to t = T ).

Our baseline model is similar to the models defined by Harutyunyan et al. (2017) .

For all the three tasks, we used a Long Short Term Memory or LSTM BID6 network to model the temporal dependencies between the time series observations, [x t ] T t=1 .

At each step, the LSTM composes the current input x t with its previous hidden state h t−1 to generate its current hidden state h t ; that is, h t = LSTM(x t , h t−1 ) for t = 1 to t = T .

The predictions for the three tasks are then performed with the corresponding hidden states as follows: DISPLAYFORM0 wherem,d t , andl t are the probabilities for inhospital mortality, decompensation, and LOS, respectively, and W m , W d , and W l are the respective weights of the fully-connected (FC) layer.

Notice that the in-hospital mortality is predicted at end of 48 hours, while the predictions for decompensation and LOS tasks are done at each time step after first four hours of ICU stay.

We trained the models using cross entropy (CE) loss defined as below.

DISPLAYFORM1

In our multimodal model, our goal is to improve the predictions by taking both the time series data x t and the doctor notes n i as input to the network.

Convolutional Feature Extractor for Doctor Notes.

As shown in FIG2 , we adopt a convolutional approach similar to BID8 to extract the textual features from the doctor's notes.

For a piece of clinical note N , our CNN takes the word embeddings e = (e 1 , e 2 , . . . , e n ) as input and applies 1D convolution operations, followed by maxpooling over time to generate a p dimensional feature vectorẑ, which is fed to the fully connected layer along side the LSTM output from time series signal (described in the next paragraph) for further processing.

From now onwards, we denote the 1D convolution over note N asẑ = Conv1D(N ).

to predict the mortality label m at t = T (T = 48).

For this, [x t ] T t=1 is processed through an LSTM layer just like the baseline model in Sec. 3.1, and for the notes, we concatenate (⊗) all the notes N 1 to N K charted between t = 1 to t = T to generate a single document N T .

More formally, DISPLAYFORM0 We use pre-trained word2vec embeddings BID11 ) trained on both MIMIC-III clinical notes and PubMed articles to initialize our methods as it outperforms other embeddings as shown in BID2 .

We also freeze the embedding layer parameters, as we did not observe any improvement by fine-tuning them.

Being sequential prediction problems, modeling decompensation and length-of-stay requires special technique to align the discrete text events to continuous time series signals, measured at 1 event per hour.

Unlike in-hospital mortality, here we extract feature maps z i by processing each note N i independently using 1D convolution operations.

For each time step t = 1, 2 . . .

T , let z t denote the extracted text feature map to be used for prediction at time step t. We compute z t as follows.

DISPLAYFORM0 where M is the number of doctor notes seen before time-step t, and λ is a decay hyperparameter tuned on a validation data.

Notice that z t is computed as a weighted sum of the feature vectors, where the weights are computed with an exponential decay function.

The intuition behind using a decay is to give preference to recent notes as they better describe the current state of the patient.

The time series data x t is modeled using an LSTM as before.

We concatenate the attenuated output from the CNN with the LSTM output for the prediction tasks as follows: DISPLAYFORM1 Both our baselines and multimodal networks are regularized using dropout and weight decay.

We used Adam Optimizer to train all our models.

We used and then 15% of remaining data as validation set.

However, We dropped all clinical notes which doesn't have any chart time associated and then dropped all the patients without any notes.

Notes which have been charted before ICU admission are concatenated and treated as one note at t = 1.

For in-hospital mortality task, best performing baseline and multimodal network have 256 hidden units LSTM cell.

For convolution operation, we used 256 filters for each of kernel size 2, 3 and 4.

For decompensation and LOS prediction, we used 64 hidden units for LSTM and 128 filters for each 2,3 and 4 size convolution filters.

The best decay factor λ for text features was 0.01.

We use Area Under Precision-Recall (AUCPR) metric for in-hospital mortality and decompensation tasks as they suffer from class imbalance with only 10% patients suffering mortality, following the benchmark.

BID3 suggest AUCPR for imbalanced class problems.

We use Cohen's linear weighted kappa, which measures the correlation between predicted and actual multi-class buckets to evaluate LOS in accordance with Harutyunyan et al. (2017) .We compared multimodal network with the baseline time series LSTM models for all three tasks.

Results from our experiments are documented in TAB1 .

Our proposed multimodal network outperforms the time series models for all three tasks.

For in-hospital mortality prediction, we see an improvement of around 7.8% over the baseline time series LSTM model.

The other two problems were more challenging itself than the first task, and modeling the notes for sequential task was difficult.

With our multimodal network, we saw an improvement of around 6% and 3.5% for decompensation and LOS, respectively.

We did not observe a change in performance with respect to results reported in benchmark (Harutyunyan et al., 2017 ) study despite dropping patients with no notes or chart time.

In order to understand the predictive power of clinical notes, we also train text only models using CNN part from our proposed model.

Additionally, we try average word embedding without CNN as another method to extract feature from the text as a baseline.

Text-only-models perform poorly compared to time-series baseline.

Hence, text can only provide additional predictive power on top of time-series data.

Identifying the patient's condition in advance is of critical importance for acute care and ICU management.

Literature has exclusively focused on using time-series measurements from ICU instruments to this end.

In this work, we demonstrate that utilizing clinical notes along with time-series data can improve the prediction performance significantly.

In the future, we expect to improve more using advanced models for the clinical notes since text summarizes expert knowledge about a patient's condition.

@highlight

We demostarte that using clinical notes in conjuntion with ICU instruments data improves the perfomance on ICU management benchmark tasks