This work addresses the long-standing problem of robust event localization in the presence of temporally of misaligned labels in the training data.

We propose a novel versatile loss function that generalizes a number of training regimes from standard fully-supervised cross-entropy to count-based weakly-supervised learning.

Unlike classical models which are constrained to strictly fit the annotations during training, our soft localization learning approach relaxes the reliance on the exact position of labels instead.

Training with this new loss function exhibits strong robustness to temporal misalignment of labels, thus alleviating the burden of precise annotation of temporal sequences.

We demonstrate state-of-the-art performance against standard benchmarks in a number of challenging experiments and further show that robustness to label noise is not achieved at the expense of raw performance.

Figure 1: Temporal localization under label misalignment.

Models are trained with noisy labels that differ from the actual ground-truth, while the final inference objective is the precise localization of events.

The surge of deep neural networks Schmidhuber, 2015) has accentuated the evergrowing need for large corpora of data (Banko & Brill, 2001; Halevy et al., 2009) .

The main bottleneck for the efficient creation of datasets remains the annotation process.

Over the years, while new labeling paradigms have emerged to alleviate this issue (e.g., crowdsourcing (Deng et al., 2009) or external information sources (Abu-El-Haija et al., 2016) ), these methods have also highlighted, and emphasized, the prevalence of label noise.

Deep neural networks are unfortunately not immune to these perturbations as their intrinsic ability to memorize and learn label noise (Zhang et al., 2017) can be the cause of training robustness issues and poor generalization performance.

In this context, the development of models robust to label noise is essential.

This work tackles the problem of precise temporal localization of events (i.e., determining when and which events occur) in sequential data (e.g. time series, video or audio sequences) despite only having access to poorly aligned annotations for training (see Figure 1 ).

This task is characterized by the discrepency between the precision required of the predictions during inference and the noisiness of the training labels.

Indeed, while models are trained on inaccurate data, they are evaluated on their ability to predict event occurences as precisely as possible with respect to the ground-truth.

In such a setting, effective models have to infer event locations more accurately than the labels they relied on for training.

This requirement is particularly challenging for most classical approaches that are designed to learn localization by strictly mimicking the provided annotations.

Indeed, as the training labels themselves do not accurately reflect the event location, focusing on replicating these unreliable patterns is incompatible with the overall objective of learning the actual ground-truth.

These challenges highlight the need for more relaxed learning approaches that are less dependent on the exact location of labels for training.

The presence of temporal noise in localization tasks is ubiquitous given the continuous nature of the perturbation, in contrast to classification noise where only a fraction of the samples are misclassified.

Temporal labeling is further characterized by an inevitable trade-off between annotation precision and time investment.

For instance, while a coarse manual transcription of a minute of complex piano music might be achieved within a moderate time frame, a millisecond precision requirement -a common assumption for deep learning models -significantly increases the annotation burden.

In this respect, models alleviating the need for costly annotations are key for a wide and efficient deployment of deep learning models in temporal localization applications.

This work introduces a novel model-agnostic loss function that relaxes the reliance of the learning process on the exact temporal location of the annotations.

This softer learning approach inherently makes the model more robust to temporally misaligned labels.

Contributions This work: a) proposes a novel loss function for robust temporal localization under label misalignment, b) presents a succinct analysis of the loss' properties, c) evaluates the robustness of state-of-the-art localization models to label misalignment, and d) demonstrates the effectiveness of the proposed approach in various experiments.

The main assumption of this work is the instantaneous nature (i.e., lasting only one time-step in discrete time settings) of the events of interest.

(Durations can nevertheless be modelled in such a framework by labeling the beginning and end of each event class as two separate channels.)

Thus, for each sample, the ground-truth

consists of M event occurrences each defined by its exact timestamp (t m ∈ R ≥0 ) and its class (c m ∈ [1, ..., d], with d event classes).

In this work, temporal label misalignment is then modelled by adding perturbations to the ground-truth timestamps:

Although commonly defined as a normal distribution N (0, σ Objective Estimate the true event occurrence times T G of an unseen input sequence X using only the noisy data D for training.

From a practical standpoint, although not necessary for the use of our loss, time is generally discretized.

In such a discrete setting, each predictor X i of the training data D := {(X i , Y i ) : 0 < i ≤ N } is an observable temporal sequence of length T i (i.e., X i = (x i (t)) Ti t=1 ∈ IR Ti×λ ) such as a DNNlearned representation, a spectrogram or any other λ-dimensional time-series.

The label sequence

Ti×d is then the discrete equivalent of T L .

(Note that this last statement assumes that only one event per class can occur at each time-step; in cases where this assumption is violated, the use of smaller temporal granularity solves this issue.)

Temporal Localization Under Label Misalignment The literature on temporal noise robustness is limited despite the critical relevance of this issue.

First, Yadati et al. (2018) propose solutions combining noisy and expert labels; however, unlike our approach, these methods require a sizable clean subset of annotations.

Second, while Adams & Marlin (2017) achieve increased robustness by augmenting simple classifiers with an explicit probabilistic model of the noise structures, the effectiveness of the approach on more complex temporal models (e.g., LSTM) still needs to be demonstrated.

Finally, Lea et al. (2017) perform robust temporal action segmentation by introducing an encoder-decoder architecture.

However, the coarse temporal encoding comes at the expense of finer-grained temporal information, which is essential for the precise localization of short events (e.g., drum hits).

In this paper, rather than a new architecture, we propose a novel and flexible loss function -agnostic to the underlying network -which allows the robust training of temporal localization networks even in the presence of extensive label misalignment.

Classical Heuristic Our approach is closely linked to the more classical trick of label smoothing or target smearing (e.g., applying aσ 2 -Gaussian filter Φσ2 to the labels) which has been considered to increase robustness to temporal misalignment of annotations (Schlüter & Böck, 2014; Hawthorne et al., 2017) .

This slight modification of the input data converts the original point prediction problem into a distribution prediction problem.

Indeed, the smoothing of the labels transforms the point labels into distributions.

The algorithm is then trained to predict these distributions, which eventually have to be transformed back to point predictions using hand-crafted peak picking heuristics (see Figure 3 (left)).

This methodology is also very common in 2D image keypoint detection applications which deal with spatial uncertainty, e.g. human pose estimation (Tompson et al., 2014; 2015) or facial landmark detection (Merget et al., 2018) .

However, despite its intuitive nature, this traditional solution presents several inherent drawbacks (see Figure 2 ): (Issue 1) Even in a noise-free setting, by transforming the impulse-like target into a distribution, the optimal model predictions (with respect to the training loss) differs from the actual goal of the pipeline (i.e., precise localization indicated by the original event label). (Issue 2) As the model learns by mimicking the smoothed target throughout the learning phase, the predictions themselves will be spread out over several time-steps.

Hence, additional tailored heuristics, such as peak picking (Böck et al., 2013) or complex thresholding, are required to achieve precise temporal localization. (Issue 3) Even advanced peak picking approaches struggle to disentangle close events.

For instance, a unique maximum might emergence in the middle of two events, thus significantly disturbing the timeliness of the final predictions. (Issue 4) Having the label mass dispersed temporally both before and after the event occurrences is problematic not only for causal models (i.e., models that make predictions at time t only with data up to time t − 1) but also for one-sided recurrent networks and fully convolutional architectures with limited receptive fields.

Indeed, all these models have to estimate the left tail of the label distribution before even seeing the event occur.

This requirement compels the model to find some structure before the actual event occurrence, leading to poor generalization performance.

Although bidirectional networks do not suffer from it, this issue limits the range of possible architectures.

The presence of strong label misalignment further worsens these four issues as increased noise commonly warrants increased smoothing, dispersing the label (and consequently prediction) mass even more.

Overall, experimental evidence (e.g. Section 5.1.1) shows that the accumulation of these issues proves to be very detrimental to the noise robustness of these classical approaches.

In contrast, this work presents a novel paradigm for dealing with temporal label uncertainty.

The main idea consists in directly inferring point predictions rather than resorting to distributions or heatmaps by fully integrating the modelling of the noise into the loss function.

Such a direct approach allows for an end-to-end learning of localization without the need for additional hand-crafted components.

In addition, the systematic and standalone loss function proposed in this regard (Section 4.2) not only solves all the above-mentioned issues but also scales well to extensive label misalignment.

Weakly-Supervised Learning Some weakly-supervised models leverage weaker annotations to infer more fine-grained concepts.

In such frameworks, noisy labels are implicitly bypassed by the use of higher-level labels -which are more invariant to perturbations.

For instance, some works achieve object detection (Fergus et al., 2003; Bilen & Vedaldi, 2016) or temporal localization (Kumar 4 SOFTLOC MODEL

The general principle of relaxing the localization learning is intuitive and potentially powerful if carefully implemented.

However, by smoothing the label only, classical approaches transform the original point prediction problem into a distribution prediction problem which eventually causes issues (see Section 3).

Many of the drawbacks arising from the asymmetric nature of the one-sided smoothing can however be alleviated by filtering not only the labels (i.e., Φ Si 2 * Y i (·)), but also the predictions (i.e., Φ Si 2 * Ŷ i (·)) with a unique softness parameter S i .

The comparison of these two smoothed processes yields a relaxed loss function for the soft learning of the location that deals on its own with the temporal uncertainty of the labels.

Indeed, in such a setting, the model is given input sequences of point-like events and directly infers point predictions without having to resort to distributions or heatmaps (see Figure 3 (right)); it is only the loss function that views these point labels and predictions as smoothed processes.

In discrete time settings, the loss can be written as

where L (·, ·) can be any measure of distance.

The learning is characterized as soft since the loss is not strictly constraining in terms of precision or mass concentration.

Indeed, the mass of each event can be both scattered over numerous time-steps and slightly shifted temporally without any abrupt increase in loss.

Thus, the model's reliance on exact label locations is relaxed.

Measure L In noise-free settings, the average stepwise cross-entropy is a common choice of loss function for state-of-the-art models (Lea et al., 2017; Wu et al., 2018; Hawthorne et al., 2019) .

While a potentially unbounded penalization of false predictions might be ideal when training on clean datasets, such behavior can be highly detrimental when labels are subject to temporal misalignment.

Therefore, for all experiments in Section 5, L is set to the (bounded) average local mean-squared error.

Properties Symmetrically smoothing both the labels and predictions solves several of the issues highlighted in the previous section (see Figure 4) .

First, in a noise-free setting, the optimal predictions with respect to L SLL are the original annotations themselves. (Solves 1).

Second, since the predictions are also smoothed over time, each trigger adds detection mass not only after, but also before the prediction time.

Therefore, the model is not required the estimate the left-tail of the label distribution before the actual event occurrence (Solves 4).

The prediction mass for a particular event is not necessarily dispersed over time anymore.

For instance, in noise-free settings, the point-like targets themselves are the solution to the optimization problem.

However, L SLL does not strictly constrain the mass of each event to be contained in a single time-step (Partially Solves 2 & 3).

The potential dispersion of the prediction mass and its direct consequences on localization performance still need to be addressed.

To that end, we propose to leverage the properties of the weaklysupervised model defined in (Schroeter et al., 2019) , which achieves precise temporal localization using only occurrence counts for training.

Aside from exhibiting strong localization performance, the loss introduced in that work possesses an implicit mass convergence property, which concentrates the scattered prediction mass toward well-defined single points in time:

where F is the set of all subsets of {1, 2, ...,

Full SoftLoc Model Incorporating this mass convergence loss as a regularizer to our soft localization learning loss L SLL allows the model to directly achieve precise impulse-like localization, without weakening its noise robustness properties.

Thus, this eliminates prediction ambiguity, as only a single point prediction is outputted per event occurrence (Solves 2 & 3).

Overall, when trained with the SoftLoc loss,

the model simultaneously softly learns to mimic the localization annotation, while converging the scatter mass toward impulse-like predictions.

In this equation, α τ regulates the predominance of the mass convergence against the soft learning (for training iteration τ ).

From a practical standpoint, starting with a moderate α τ allows an initial relaxed localization learning, before performing stronger mass convergence (see Section 5 for the specific settings used in this paper).

End-to-end Learning of Localization One of the key factors of the predominance of the deep learning models over classical ones relies on their ability to solve problems in an end-to-end fashion (Collobert et al., 2011; Krizhevsky et al., 2012) , without the need to resort to partial optimization or hand-crafted heuristics.

In contrast to more classical approach (see Section 3), our proposed method is an end-to-end solution to the problem of temporal localization in the presence of misaligned labels (see Issue 2).

This solution eliminates the need for hand-crafted components (e.g. peak picking) and is expected to better serve the task at hand.

Continuous Setting While all experiments in Section 5 and most state-of-the-art temporal localization models perform a discretization of time, the loss definition can easily be adapted to suit continuous-time frameworks.

Our versatile SoftLoc model is a generalization of several past works.

Indeed, depending on the softness parameter S M , the model encompasses a wide range of training regimes from classical fully-supervised to count-based weakly-supervised.

Softness → 0 → 0 → 0 By tending S M toward zero, the model becomes similar to a count-aware localization RNN with soft localization learning loss.

For instance, setting

which corresponds to the sum of all stepwise cross-entropies.

By further setting α τ = 0 (i.e., discarding any count-awareness), our loss function becomes identical to the ones found in numerous temporal detection works (e.g., drum detection (Wu et al., 2018) , piano onset detection (Hawthorne et al., 2017) , and video action segmentation (Lea et al., 2017) ).

to vanish, discarding any prior information of localization, thus making the training weakly-supervised (Schroeter et al., 2019) :

The introduced softness parameter can be leveraged to deal with different kinds of uncertainties.

First, in contrast to the traditional approach of aggregating the annotations of multiple individuals (thus trading off dataset richness for noise reduction), our model can be trained on all conflicting individual sequences, since it can cope with noisy annotations.

Second, an annotator specific softness S a 2 can further be implemented to model their respective reliability.

Finally, an extract specific softness can be incorporated to capture the noise or annotation complexity of certain more challenging sequences.

Experiments conducted in the section below show that the performance is robust to variations in the softness parameter.

Indeed, this hyperparameter only acts as a coarse indicator of temporal uncertainty and thus does not need to strictly match the underlying noise distribution.

In this section, we demonstrate the effectiveness and flexibility of our approach in a broad range of challenging experiments (music event detection, times series detection, video action segmentation).

, Experiment and implementation details can be found on the paper's website 1 .

Piano transcription and more specifically piano onset detection is a difficult problem as it requires precise and simultaneous detection of hits from 88 different polyphonic channels.

Dataset This experiment is based on the MAPS database (Emiya et al., 2010) .

The dataset creation protocol strictly follows the one from Hawthorne et al. (2017) .

(Only onsets are considered for the comparison.)

To evaluate the robustness, the training labels are artificially perturbed according to a normal distribution m ∼ N (0, σ 2 ), while the test labels are kept intact for unbiased evaluation.

Benchmarks Three different benchmarks are considered.

First, the state-of-the-art model (on clean data) proposed by Hawthorne et al. (2017) is highly representative of models aiming for optimal performance with little regard for annotation noise (Hawthorne) .

Second, a smoothed version of the first benchmark with extended onset length (i.e., over 96ms) illustrates the common practice used to achieve robustness (Hawthorne (smoothed)).

Finally, as the first benchmark performs local classification using standard cross-entropy, the soft bootstrapping loss proposed by Reed et al. (2014) is leveraged instead for increased robustness (Bootstrap (soft)).

Architecture, Training and Evaluation Our network is comprised of six convolutional layers (representation learning) followed by a 128-unit LSTM (temporal dependencies learning) and two fully-connected layers (prediction mapping).

The network is trained using mel-spectrograms (Stevens et al., 1937) and their first derivatives stacked together as model input, while data augmentation in the form of sample rate variations is applied for increased robustness and performance.

The loss (Equation 4) with softness S M = 100ms is optimized using the Adam algorithm (Kingma & Ba, 2015) .

The models are evaluated on the noise-free test set using F 1 -scores computed with the standard mir_eval library (Raffel et al.) and a 50ms tolerance (Hawthorne et al., 2017) . (α τ = max(min( τ −10 5 10 5 , .9), .2).)

Figure 5: F 1 piano onset detection performance of our approach (softness SM = 100ms) and the benchmark models as a function of label noise levels.

Results As depicted in Figure 5 , our proposed SoftLoc approach displays strong robustness against label misalignment; in contrast to all benchmarks, the performance appears almost invariant to the noise level.

(See Appendix A.1 for discussion on the model's performance for σ > 200ms.)

At σ = 150ms, only 26% of training labels lie within the 50ms tolerance.

In this context, the score achieved by our SoftLoc model (i.e., ∼ 75%) is unattainable for classical approaches, which do not take label uncertainty into account and attempt to strictly fit the noisy annotations.

While standard tricks, such as label smoothing, slightly improve noise robustness (e.g., Hawthorne (smoothed)), their effectiveness is limited in contrast to our proposed approach.

Finally, the parameters used throughout this experiment are fixed.

However, as our loss is a strict generalization of the standard cross-entropy loss used by Hawthorne et al. (2017) , the small performance gap for small noise levels can be reduced by setting α τ = 1,

2 → 0ms and L (·) = − log(1 − |·|).

Ablation Study To assess the usefulness of the different components of L SoftLoc , we repeat the above experiments keeping only individual parts of the loss function.

Table 1 reveals that L SLL is the main driver of performance in noise-free settings, while L MC ensures stability under increased label misalignment.

(A simple threshold-based peak-picking algorithm was implemented to infer localization from the dispersed mass produced by L SLL .)

Overall, while each loss individually produces reasonable predictions, only the combined L SoftLoc yields both competitive scores in noise-free settings and strong robustness to temporal misalignment.

The softness S M is a defining model hyperparameter.

In this section, 210 independent runs for the same drum detection experiment are conducted with varying noise and softness levels in order to highlight the correlation between this key parameter, label noise and the final localization performance.

The experiment is based on the D-DTD Eval Random drum detection task (IDMT-SMTDrums dataset (Dittmar & Gärtner, 2014) ) performed by Wu et al. (2018) .

The goal is the correct Results The results of the 210 runs are displayed in Figure 6 .

A Gaussian Nadaraya-Watson kernel regression (Nadaraya, 1964; Watson, 1964 ) is used to interpolate the F 1 -score, offering a detailed view of the model's response to varying label noise levels.

This figure not only confirms the model's high robustness to label misalignments, but also reveals that these results are very robust to changes in the softness level.

Indeed, a wide range of softnesses yield optimal performance, as long as S M ≥ σ.

Obviously, extreme softness levels (e.g. S M 2 → ∞) would however induce a partial or even total loss of the information conveyed by the localization prior, resulting in a decrease in performance (see Table  2 ).

Robustness considerations aside, our SoftLoc model displays an outstanding overall performance with F 1 -scores over 95% across all noise levels; the model -even when trained on extremely noisy labels (e.g., σ = 100ms) -outperforms several standard benchmarks Wu et al. (2018) which were trained on noise-free training samples (σ = 0ms).

Noise-free Comparison In clean settings (i.e., σ = 0ms) , the benchmark models have a clear advantage as they correctly assume noise-free labels.

Despite this, our SoftLoc model achieves stateof-the-art performance on three different metrics (KD, HH, precision) demonstrating that robustness does not come at the expense of raw localization performance (see Table 2 ).

The timely detection of events in healthcare time series is a crucial challenge to improve medical decision making.

The task tackled in this section consists in the precise temporal detection of smoking episodes using wearable sensors features based on the puffMarker dataset (Saleheen et al., 2015) .

Once again, in order to conduct the robustness analysis, the original annotations are artificially misaligned.

However, as each time-step in this dataset represents a full respiration cycle, the noise distributions must be applied in a discrete fashion: namely, rounded normal distribution (i.e., E i ∼ N (0, σ 2 ) ) or binary constant length shifting of labels (δ steps either to the left or the right with equal probability), denoted B(−δ, δ).

This task is particularly challenging as detections have to be perfectly aligned with the ground-truth to be considered correct.

As the focus is set on robustness rather than raw performance, the model architecture is kept extremely simple: a 14-node fully connected layer followed by a 14-unit LSTM Table 2 : Noise-free Drum Detection.

Comparison of our SoftLoc model (SM = 100ms) and state-ofthe-art models evaluated in Wu et al. (2018) on the clean D-DTD Eval Random task (σ = 0ms).

The F 1 -scores per instrument (KD/SD/HH), the average precision, recall, and overall F 1 are displayed.

and a final fully connected layer with softmax activation.

Both the standard cross-entropy (CE) and our L SoftLoc loss function are evaluated.

The LR-M model proposed by Adams & Marlin (2017) , which was developed to achieve strong robustness to temporal misalignment of labels on this particular dataset, is also considered as benchmark.

Results The results, produced using ten 6-fold (leave-one-patient-out) cross-validation.

are summarized in Table 3 .

Not only does training with the proposed L SoftLoc loss function yield a strong improvement in robustness when compared to the standard cross-entropy, but our simple recurrent model also significantly outperforms the robust LR-M model on all metrics.

In addition, our approach displays low standard deviations, which underlines the consistency and robustness of the learning.

These observations hold for both noise distributions (N and B) ; hence, the normal smoothing filters do not require the underlying noise to be normally distributed in order for the model to be effective.

Further testing with skew normal distribution of noise confirm these results even in non-symmetric settings.

Video action segmentation -a dense classification problem where each time-step has to be mapped to one action class -differs substantially from music event localization or time series detection problems, where scattered events from multiple classes have to be precisely localized.

Nonetheless, the properties of the SoftLoc loss can still be leveraged on such a task; in this context, while the role of L SLL is unchanged, L MC acts as a count-based regularizer, rather than a means for mass convergence.

Experiments Several video segmentation experiments from Lea et al. (2017) are replicated using either the standard cross-entropy (original loss), L SLL or L SoftLoc as training loss for the ED-TCN model.

As the ED-TCN model already exhibits strong robustness properties against label misalignment Lea et al. (2017) , these experiments will allow to measure the additional marginal gain in performance and robustness when replacing the standard cross-entropy with our the proposed L SoftLoc loss function.

To assess robustness, each label sequence in the training set is either delayed or advanced by a fixed constant δ. (S M = 7s). (Adams & Marlin, 2017 ) and the deep model trained with CE or L SoftLoc with respect to misalignment distributions N (0, σ 2 ) and B(−δ, δ).

Reported metrics are mean and standard deviation of ten 6-fold cross-validated F 1 -scores.

80.6 (8.6) 65.9 (17.4) 64.0 (15.6) 55.0 (19.7) 92.6 (2.9) 55.3 (16.2) 36.0 (15.6) 28.9 (17.0) 25.8 (16.2) Results As summarized in Table 4 and Table 5 (in Appendix B.2), replacing the standard crossentropy loss with L SoftLoc does not only significantly increase the robustness of the ED-TCN model -which was already shown to be robust to label misalignment (Lea et al., 2017) -but also achieves competitive performance in noise-free settings.

Further experiments with different softness parameters (see Figure 10 in Appendix B.1) reveal that increasing the model softness S M as the underlying noise levels increase produces optimal performance.

For instance, in noisy settings, greater performance can be achieved (up to 25% overperformance) by simply choosing a large enough softness.

Overall, the SoftLoc loss function displays strong results on a very different task (i.e., temporal segmentation as opposed to temporal localization), highlighting once again its versatility of application.

In this work, we have shown how relaxing annotation requirements (i.e., weakening the model's reliance on the exact location of events) not only has the practical benefit of alleviating annotation efforts but, more importantly, leads to a model that is robust to temporal noise without compromising performance on clean training data.

This contrasts with traditional approaches which attempt to strictly mimic the annotations, leading to poor predictions when training with noisy labels.

We have demonstrated these claims on a number of classical challenging tasks, in which our SoftLoc loss exhibits state-of-the-art performance.

The proposed loss function is agnostic to the underlying network and hence can be used as a loss replacement in almost any recurrent architecture.

The versatility of the model can find applications in a wide array of tasks, even beyond temporal localization.

A.1 EXTREME NOISE SETTINGS Figure 1 (in the main text) depicts the strong invariance of our SoftLoc model to label misalignment on a broad array of noise levels (i.e., up to σ = 200ms).

In this section, we evaluate the model's performance on an even wider range in order to fully assess its behavior in extreme settings.

To that end additional piano onset detection experiments, with noise levels up to σ = 1000ms, were conducted following the protocol described in Section 5.1.

The results are displayed in Figure 7 .

Overall, this figure confirms the remarkable robustness of our SoftLoc model to label misalignment.

While the absolute performance unsurprisingly decreases as the training data becomes less accurate, the detection capability of the model in noisy settings outshines any classical approach (see Figure 1 in the main text).

Finally, these results could further be improved by increasing the model softness S M (see Section 5.2).

Timeliness of SoftLoc predictions Figure 8 illustrates how consistently precise and well-centered (i.e., neither too late nor early) the predictions are regardless of the noise setting.

Indeed, there is almost no difference in prediction centering when comparing the results for σ = 0ms or σ = 200ms.

Noisy Labels and Ground-Truth Discrepancy To further illustrate the complexity of the localization task when annotations are subject to misalignment, we consider the training labels as predictions and then compare them to the clean ground-truth.

Figure 9 displays an example of the quality of the training labels.

Obviously, in the noise-free setting (i.e., σ = 0ms), the localization is spotless as the training labels and the ground-truths are identical.

However, as the noise level increases, the proportion of labels that stay within the 50ms tolerance window decreases significantly.

More precisely, the performance (i.e., F 1 -score) of the labels themselves is 68.2%, 39.8% and 23.7% for σ equal to 50ms, 100ms and 200ms respectively.

As depicted in Figure 10 , training with the SoftLoc loss function instead of the standard cross-entropy yields improved performance (up to 25%) in all noise settings almost regardless of the softness S M .

The only exception occurs when selecting a softness level that is too wide while training with noise-free (δ = 0) labels.

As also observed in Section 5.1.2, the model achieves optimal performance when the softness level S M is slightly larger than noise level δ.

However, although the efficiency of the approach is bound to decrease when the disparity between selected softness and noise level is becoming too large, a performance close to the optimal one can be achieve with a wide range of softnesses S M .

Figure 10: Video Action Segmentation.

Relative performance of the ED-TCN model trained with L SoftLoc -relative to CE -with respect to the softness level S M for various noise levels δ.

<|TLDR|>

@highlight

This work introduces a novel loss function for the robust training of temporal localization DNN in the presence of misaligned labels.

@highlight

A new loss for training models that predict where events occur in a training sequence with noisy labels by comparing smoothed label and prediction sequence.