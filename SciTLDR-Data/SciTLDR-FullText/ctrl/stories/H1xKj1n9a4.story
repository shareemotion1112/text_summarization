Deep neural networks (DNNs) are inspired from the human brain and the interconnection between the two has been widely studied in the literature.

However, it is still an open question whether DNNs are able to make decisions like the brain.

Previous work has demonstrated that DNNs, trained by matching the neural responses from inferior temporal (IT) cortex in monkey's brain, is able to achieve human-level performance on the image object recognition tasks.

This indicates that neural dynamics can provide informative knowledge to help DNNs accomplish specific tasks.

In this paper, we introduce the concept of a neuro-AI interface, which aims to use human's neural responses as supervised information for helping AI systems solve a task that is difficult when using traditional machine learning strategies.

In order to deliver the idea of neuro-AI interfaces, we focus on deploying it to one of the fundamental problems in generative adversarial networks (GANs): designing a proper evaluation metric to evaluate the quality of images produced by GANs.

Deep neural networks (DNNs) have successfully been applied to a number of different areas such as computer vision and natural language processing where they have demonstrated state-of-the-art results, often matching and even sometimes surpassing a human's ability.

Moreover, DNNs have been studied with respect to how similar processing is carried out in the human brain, where identifying these overlaps and interconnections has been a focus of study and investigation in the literature BID5 BID4 BID11 BID18 BID30 BID1 BID35 BID17 BID15 .

In this research area, convolutional neural networks (CNNs) are widely studied to be compared with the visual system in human's brain because of following reasons: (1) CNNs and human's visual system are both hierarchical system; (2) Steps of processing input between CNNs and human's visual system are similar to each other e.g., in a object recognition task, both CNNs and human recognize a object based on their its shape, edge, color etc..

Work BID35 outlines the use of CNNs approach for delving even more deeply into understanding the development and organization of sensory cortical processing.

It has been demonstrated that CNNs are able to reflect the spatio-temporal neural dynamics in human's brain visual area BID5 BID30 BID18 .

Despite lots of work is carried out to reveal the similarity between CNNs and brain system, research on interacting between CNNs and neural dynamics is less discussed in the literature as understanding of neural dynamics in the neuroscience area is still limited.

There is a growing interest in studying generative adversarial networks (GANs) in the deep learning community BID10 .

Specifically, GANs have been widely applied to various domains such as computer vision BID14 , natural language processing BID7 and speech synthesis BID6 .

Compared with other deep generative models (e.g. variational autoencoders (VAEs)), GANs are favored for effectively handling sharp estimated density functions, efficiently generating desired samples and eliminating deterministic bias.

Due to these properties GANs have successfully contributed to plausible image generation BID14 , image to image translation BID38 , image super-resolution BID19 , image completion BID37 etc..

However, three main challenges still exist currently in the research of GANs: (1) Mode collapse -the model cannot learn the distribution of the full dataset well, which leads to poor generalization ability; (2) Difficult to trainit is non-trivial for discriminator and generator to achieve Nash equilibrium during the training; (3) Hard to evaluate -the evaluation of GANs can be considered as an effort to measure the dissimilarity between real distribution p r and generated distribution p g .

Unfortunately, the accurate estimation of p r is intractable.

Thus, it is challenging to have a good estimation of the correspondence between p r and p g .

Aspects (1) and (2) are more concerned with computational aspects where much research has been carried out to mitigate these issues BID20 Salimans et al., 2016; BID0 .

Aspect (3) is similarly fundamental, however, limited literature is available and most of the current metrics only focus on measuring the dissimilarity between training and generated images.

A more meaning-ful GANs evaluation metric that is consistent with human perceptions is paramount in helping researchers to further refine and design better GANs.

Although some evaluation metrics, e.g., Inception Score (IS), Kernel Maximum Mean Discrepancy (MMD) and Fr??chet Inception Distance (FID), have already been proposed (Salimans et al., 2016; BID13 BID2 , their limitations are obvious: (1) These metrics do not agree with human perceptual judgments and human rankings of GAN models.

A small artefact on images can have a large effect on the decision made by a machine learning system BID16 , whilst the intrinsic image content does not change.

In this aspect, we consider human perception to be more robust to adversarial images samples when compared to a machine learning system; (2) These metrics require large sample sizes for evaluation Salimans et al., 2016) .

Large-scale samples for evaluation sometimes are not realistic in real-world applications since it is time-consuming; and (3) They are not able to rank individual GAN-generated images by their quality i.e., the metrics are generated on a collection of images rather than on a single image basis.

The within GAN variances are crucial because it can provide the insight on the variability of that GAN.Work BID36 demonstrates that CNN matched with neural data recorded from inferior temporal cortex BID3 has high performance in object recognition tasks.

Given the evidence above that a CNN is able to predict the neural response in the brain, we describe a neuro-AI interface system, where human being's neural response is used as supervised information to help the AI system (CNNs used in this work) solve more difficult problems in real-world.

As a starting point for exploiting the idea of neuro-AI interface, we focus on utilizing it to solve one of the fundamental problems in GANs: designing a proper evaluation metric.

Neural response Stimulus Figure 1 .

Schematic of neuro-AI interface.

Stimuli (image stimuli used in this work) are simultaneously presented to an AI system and participants.

Participants' neural responses are transferred to the AI system as supervised information for assisting the AI system make decision.

In this paper, we first demonstrate the ability of a brainproduced score (we call it Neuroscore), generated from human being's electroencephalography (EEG) signals , in terms of the quality evaluation on GANs.

Secondly, we demonstrate and validate a neural-AI interface (as seen in Fig. 1 ), which uses neural responses as supervised information to train a CNN.

The trained CNN model is able to predict Neuroscore for images without corresponding neural responses.

We test this framework via three models: Shallow convolutional neural network, Mobilenet V2 BID26 and Inception V3 BID29 .In detail, Neuroscore is calculated via measurement of the P300, an event-related potential (ERP) BID23 present in EEG, via a rapid serial visual presentation (RSVP) paradigm BID32 .

P300 and RSVP paradigm are mature techniques in the brain-computer interface (BCI) community and have been applied in a wider variety of tasks such as image search BID9 , information retrieval BID22 , and etc.

The unique benefit of Neuroscore is that it more directly reflects the human perceptual judgment of images, which is intuitively more reliable compared to the conventional metrics in the literature BID2 .

Current literature has demonstrated that CNNs are able to predict neural responses in inferior temporal cortex in image recognition task BID36 BID35 via invasive BCI techniques BID31 .

Evidence shows that neural responses in inferior temporal cortex directly link the information processing during the image recognition task.

Therefore, a CNN trained by predicting neural responses in inferior temporal cortex also achieves the good performance during the image recognition BID36 .

Comparing the traditional end-toend machine learning system, use of DNNs for predicting neural responses in the brain favors following benefits: (1) It enables the information processing of DNNs closer to human being's brain system; (2) For some difficult tasks in real-world e.g, evaluation of image quality demonstrated in this paper, it is still challenging to design the machine learning algorithms, which teach DNNs to process the information like humans; and (3) Neural signals directly reflect the human perception and interfacing between neural responses and DNNs can be more efficient than the traditional methods regarding the area of human and AI.The investigation of using CNNs to predict neural response from non-invasive BCI aspect is still blank in the literature.

Comparing to invasively measured neural dynamics, EEG favors pros such as simple measurement, unpainful experience during recording, free to ethic argument and more easily generalized to real-world applications.

However, EEG suffers challenges such as low signal quality (i.e TAB1 Neuro-AI Interface low SNR), low spatial resolution (interested neural activities span all scalp and difficult to be localized), which makes the prediction for EEG response still challenging.

With advanced machine leaching technologies applied to non-invasive BCI area, source localization and reconstruction are feasible for EEG signals today.

Previously work BID33 a) have demonstrated the efficacy of using spatial filtering approaches for reconstructing P300 source ERP signals.

The low SNR issue can be remedied by averaging the EEG trials.

Based on this evidence, we explore the use of DNNs to predict Neuroscore when neural information is available.

We propose a neuro-AI interface in order to generalize the use of Neuroscore.

This kind of framework interfaces between neural responses and AI systems (CNN used in this study), which uses neural responses as supervised information to train a CNN.

The trained CNN is then used for predicting Neuroscore given images generated by one of the popular GAN models.

Figure 2 demonstrates the schematic of neuro-AI interface used in this work 1 .

Flow 1 shows that the image processed by human being's brain and produces single trial P300 source signal for each input image.

Flow 2 in Fig. 2 demonstrates a CNN with including EEG signals during training stage.

The convolutional and pooling layers process the image similarly as retina done BID21 .

Fully connected layers (FC) 1-3 aim to emulate the brain's functionality that produces EEG signal.

Yellow dense layer in the architecture aims to predict the single trial P300 source signal in 400-600 ms response from each image input.

In order to help model make a more accurate prediction for the single trial P300 amplitude for the output, the single trial P300 source signal in 400-600 ms is fed to the yellow dense layer to learn parameters for the previous layers in the training step.

The model was then trained to predict the single trial P300 source amplitude (red point shown in signal trail P300 source signal of Fig. 2 ).

Mobilenet V2, Inception V3 and Shallow network were explored in this work, where in flow 2 we use these three network bones: such as Conv1-pooling layers.

For Mobilenet V2 and Inception V3.

We used pretrained parameters from up to the FC 1 shown in Fig. 2 .

We trained parameters from FC 1 to FC 4 for Mobilenet V2 and Inception V3.

?? 1 is 1 We understand that human being's brain system is much more complex than what we demonstrated in this work and the flow in the brain is not one-directional BID27 BID2 .

Our framework can be further extended to be more biologically plausible.

Add windowed single trial P300 source signal Figure 2 .

A neuro-AI interface and training details with adding EEG information.

Our training strategy includes two stages: (1) Learning from image to P300 source signal; and (2) Learning from P300 source signal to P300 amplitude.

loss1 is the L2 distance between the yellow layer and the single trial P300 source signal in the 400 -600 ms corresponding to the single input image.

loss2 is the mean square error between model prediction and the single trial P300 amplitude.

loss1 and loss2 will be introduced in section 3.2.used to denote the parameters from FC 1 to FC 3 and ?? 2 indicates the parameters in FC 4.

For the Shallow model, we trained all parameters from scratch.

We added EEG to the model because we first want to find a function f (??) ??? s that maps the images space ?? to the corresponding single trial P300 source signal s.

This prior knowledge can help us to predict the single trial P300 amplitude in the second learning stage.

We compared the performance of the models with and without EEG for training.

We defined two stage loss function (loss 1 for single trial P300 source signal in the 400 -600 ms time window and loss 2 for single trial P300 amplitude) as DISPLAYFORM0 where S true i ??? R 1??T is the single trial P300 signal in the 400 -600 ms time window to the presented image, and y i refers to the single trial P300 amplitude to each image.

The training of the models without using EEG is straightforward, models were trained directly to minimize loss 2 (?? 1 , ?? 2 ) by feeding images and the corresponding single trial P300 amplitude.

Training with EEG information is explained in Algorithm 1 and visualized in the "Flow 2" of Fig. 2 with two stages.

Stage 1 learns parameters ?? 1 to predict P300 source signal while stage 2 learns parameters ?? 2 to predict single trial P300 amplitude with ?? 1 fixed.

Update ?? 2 by descending its stochastic gradient: and without EEG.

All models with EEG perform better than models without EEG, with much smaller errors and variances.

Statistic tests between model with EEG and without EEG are also carried out to verify the significance of including EEG information during the training phase.

One-way ANOVA tests (P-value) for each model with EEG and without EEG are stated as: P Shallow = 0.003, P M obilenet = 0.012 and P Inception = 5.980e ??? 05.

Results here demonstrate that including EEG during the training stage helps all three CNNs improve the performance on predicting the Neuroscore.

The performance of models with EEG is ranked as follows: Inception-EEG, Mobilenet-EEG, and Shallow-EEG, which indicates that deeper neural networks may achieve better performance in this task.

We used the randomized EEG signal here as a baseline to see the efficacy of adding EEG to produce better Neuroscore output.

When randomizing the EEG, it shows that the error for each three model increases significantly.

For Mobilenet and Inception, the error of the randomized EEG is even higher than those without EEG in the training stage, demonstrating that the EEG information in the training stage is crucial to each model.

Figure 3 shows that the models with EEG information have a stronger correlation between predicted Neuroscore and real Neuroscore.

The cluster (blue, orange, and green circles) for each category of the model trained with EEG (left column) is more separable than the cluster produced by model without EEG (right column).

This conveys with EEG for training models: (1) Neuroscore is more accurate; and (2) Neuroscore is able to rank the performances of different GANs, which cannot be achieved by other metrics BID2 .

Figure 3 .

Scatter plot of predicted and real Neuroscore of 6 models (Shallow, Mobilenet, Inception with and without EEG for training) cross participants by 20 times repeated shuffling training and testing set.

Each circle represents the cluster for a specific category.

Small triangle markers inside each cluster correspond to each shuffling process.

The dot at the center of each cluster is the mean.

In this paper, we introduce a neuro-AI interface that interacts CNNs with neural signals.

We demonstrate the use of neuro-AI interface by introducing a challenge in the area of GANs i.e., evaluate the quality of images produced by GANs.

Three deep network architectures are explored and the results demonstrate that including neural responses during the training phase of the neuro-AI interface improves its accuracy even when neural measurements are absent when evaluating on the test set.

More details of the performance of Neuroscore can be referred in Appendix.

FIG1 shows the averaged reconstructed P300 signal across all participants (using LDA beamformer) in the RSVP experiment.

It should be noted here that the averaged reconstructed P300 signal is calculated as the difference between averaged target trials and averaged standard trials after applying the LDA beamformer method.

The solid lines in FIG1 are the means of the averaged reconstructed P300 signals for each image category (across 12 participants) while the shaded areas represent the standard deviations (across participants).

It can be seen that the averaged reconstructed P300 (across participants) clearly distinguishes between different image categories.

In order to statistically measure this correlative relationship, we calculated the Pearson correlation coefficient and p-value (two-tailed) between Neuroscore and BE accuracy and found (r(48) = ???0.767, p = 2.089e ??? 10).

We also did the Pearson statistical test and bootstrap on the correlation between Neuroscore and BE accuracy (human judgment performance) only for GANs i.e., DCGAN, BEGAN and PROGAN.

Pearson statistic is (r(36)=-0.827, p=4.766e-10) and the bootstrapped p ??? 0.0001.

Three traditional methods are also employed to evaluate the GANs used in this study.

score indicates better GAN performance), we use 1/Neuroscore for comparison.

It can be seen that all three methods are consistent with each other and they rank the GANs in the same order of PROGAN, DCGAN and BEGAN from high to low performance.

By comparing the three traditional evaluation metrics to the human, it can be seen that they are not consistent with human judgment of GAN performance.

It should be remembered that Inception Score is able to measure the quality of the generated images (Salimans et al., 2016) while the other two methods cannot do so.

However, Inception Score still rates DCGAN as outperforming BE-GAN.

Our proposed Neuroscore is consistent with human judgment.

Another property of using Neuroscore is the ability to track the quality of an individual image.

Traditional evaluation metrics are unable to score each individual image for two reasons: (1) They need large-scale samples for evaluation; (2) Most methods (e.g. MMD and FID) evaluate GANs based on the dissimilarity between real images and generated images so they are not able to score the generated image one by one.

For our proposed method, the score of each single image can also be evaluated as a single trial P300 amplitude.

We demonstrate that using the predicted single trial P300 amplitude to observe the single image quality in Fig. 6 .

This property provides Neuroscore with a novel capability that can observe the variations within a typical GAN.

Although Neuroscore and IS are generated from deep neural networks.

Neuroscore is more suitable than IS for evaluating GANs in that: (1) It is more explainable than IS as it is a direct reflection of human perception; (2) Much smaller sample size is required for evaluation; (3) Higher Neuroscore exactly indicates better image quality while IS Figure 6 .

P300 for each single image predicted by the proposed neuro-AI interface in our paper.

Higher predicted P300 indicates the better image quality.does not.

We also included the RFACE images in our generalization test.

FIG3 (c) demonstrates that the predicted Neuroscore is still correlated with the real Neuroscore when adding the RFACE images and the model ranks the types of images as: PROGAN>RFACE>BEGAN>DCGAN, which is consistent with the Neuroscore that has been measured directly from participants shown in FIG3 .Compared to traditional evaluation metrics, Neuroscore is able to score the GAN based on very few image samples, relatively.

Recording EEG in the training stage could be the limitation of generalizing Neuroscore to evaluate a new GAN.

However, the use of dry electrode EEG recording system BID8 can accelerate and simplify the data acquisition significantly.

Moreover, GANs enable the possibility of synthesizing the EEG BID12 , which has wide applications in brain-machine interface research.

<|TLDR|>

@highlight

Describe a neuro-AI interface technique to evaluate generative adversarial networks