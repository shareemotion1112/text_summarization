Retinal prostheses for treating incurable blindness are designed to electrically stimulate surviving retinal neurons,  causing them to send artificial visual signals to the brain.

However, electrical stimulation generally cannot precisely reproduce  normal patterns of neural activity in the retina.

Therefore, an electrical stimulus must be selected that produces a neural response as close as possible to the desired response.

This requires a technique for computing a distance between the desired response and the achievable response that is meaningful in terms of the visual signal being conveyed.

Here we propose a method to learn such a metric on neural responses, directly from recorded light responses of a population of retinal ganglion cells (RGCs) in the primate retina.

The learned metric produces a measure of similarity of RGC population responses that accurately reflects the similarity of the visual input.

Using data from electrical stimulation experiments, we demonstrate that this metric may improve the performance of a prosthesis.

An important application of neuroscience research is the development of electronic devices to replace the function of diseased or damaged neural circuits BID57 BID39 .

Artificial vision has been a particularly challenging modality due to the richness of visual information, its diverse uses in perception and behavior, and the complexity of fabricating a device that can interface effectively with neural circuitry BID47 BID56 BID20 .The most advanced example is a retinal prosthesis: a device that replaces the function of neural circuitry in the retina lost to degenerative disease.

Most of the computational work related to this application has focused on building encoding models that use the visual image to accurately predict the spiking activity of populations of retinal ganglion cells (RGCs), the output neurons of the retina that convey visual information to the brain.

Leading models include linear models BID7 , probabilistic point-process models BID33 and recently proposed models employing rich nonlinearities BID30 BID2 BID41 .However, an accurate encoding model, although valuable, is insufficient.

Any retinal prosthesiswhether based on electrical stimulation BID40 or optical stimulation BID6 BID3 -is limited in its ability to create arbitrary desired patterns of neural activity, due to inefficiencies or lack of specificity in the stimulation modality BID1 BID20 .

Thus, a given stimulation system can only achieve a limited vocabulary of elicited spike patterns.

Although a powerful and accurate encoding model might indicate that a particular spike pattern would be the natural biological response to the incident visual stimulus, the desired spike pattern might not reside within the feasible set of the stimulation device FIG0 ).Previous studies BID21 have addressed this problem by selecting the electrical stimulation which minimizes the number of unmatched spikes across cells -equivalent to the Hamming distance between two binary vectors.

Even though a Hamming distance is easy to compute, this solution is not necessarily optimal.

The goal of a prosthetics device should be to instead select an Real recorded spiking activity in a two populations of primate retinal ganglion cells (RGCs) demonstrating the lack of specificity from electrical stimulation.

The electrodes are in blue and the stimulated electrode is shaded green.

C. The target firing pattern r often lies outside the set of firing patterns achievable with the prosthesis.

The goal of the learned metric is to define a distance measure to identify the nearest feasible electrical stimulationr.

R denotes the set of all neural population responses.electrical stimulation pattern that produces a response as close as possible to the desired pattern of activity in terms of the elicited visual sensation ( FIG0 ).

In lieu of measuring the visual sensation produced by a prosthetic, we instead posit that one may infer a distance metric based on the signal and noise properties of individual and populations of neurons BID43 BID33 BID11 .

In contrast, previous approaches to spike metrics have focused on user-specified, parameteric functions BID52 BID51 or unsupervised techniques to cluster nearby spike patterns BID50 BID9 BID14 .In this work, we propose a neural response metric learned directly from the statistics and structure of firing patterns in neural populations, with the aim of using it to select optimal electrical stimulation patterns in a prosthesis device.

In particular, we learn a neural response metric by applying ideas from metric learning to recordings of RGC populations in non-human primate retina.

We demonstrate that the learned metric provides an intuitive, meaningful representation of the similarity between spike patterns in the RGC population, capturing the statistics of neural responses as well as similarity between visual images.

Finally, we use this metric to select the optimal electrical stimulation pattern within the constraints of the electrical interface to a population of RGCs.

In this section we describe the algorithmic framework for learning pseudometrics or similarity measures in neural response space.

We start by introducing notations and conventions that we use throughout the paper.

We use bold face letters to denote vectors and upper case letters to denote matrices.

We denote the symmetrization operator of a square matrix M by sym(M ) = 1 2 (M + M ).

A single frame of visual stimulus, s, is an image represented as an n × n matrix.

The space of possible stimuli is S ⊂ R n×n .

A sequence s l , . . . , s m of m − l + 1 frames, where s j ∈ S, is denoted as s l:m .

In order to simplify our notation, we define the responses of the cells to be a p dimensional vector and the space of possible responses as r ⊆ R p .

Analogously, a sequence of cell activities r t for t = l, . . .

, m is denoted r l:m .

To simplify the presentation below, we confine the visual stimulus to be a single image and the corresponding response of each cell to be a scalar.

(A) Schematic of spike trains from population of neurons responding to a dynamic stimulus (not shown).

The spiking response across the window is randomly selected at relative time t. The binarized representation for the population response is termed the anchor.

In a second presentation of the same stimulus, the population response at the same relative time t is recorded and labeled a positive.

The negative response is any other selected time.

(B) The objective of the triplet-loss is to require that the anchor is closer to the positive response than the negative response with a margin.

See text for details.

Metric and similarity learning using constrative loss BID8 BID18 and triplets loss BID42 BID55 ) have been used extensively in several domains.

In computer vision, these methods achieve state-of-the-art performance on face recognition BID38 BID48 BID49 and image retrieval .

A central theme of this work has focused on improving metric learning by mining semi-hard negatives BID38 .

Because many negatives provide minimal information, these methods use a partially learned metric to identify negatives that may maximally improve the quality of the metric given a fixed number of updates.

To avoid the computational burden imposed by such methods, some works have proposed alternative loss functions which either make efficient use of all the negatives in a batch BID32 or multiple classes using n-tuplets BID45 .

Our method is similar to these methods as we make efficient use of all the negatives in a batch as in BID32 but also use a simplified, softmax-based loss function BID45 .

Given the population response space R, we learn a function h : R × R → R which captures invariances in the spiking response when the same stimulus is presented multiple times.

The scoring function h is viewed either as a similarity function or a pseudometric.

To distinguish between the two cases, we use d(·, ·) to denote a pseudometric.

A pseudometric d needs to satisfy: DISPLAYFORM0 During the experiments, repeats of the same sequence of visual stimuli are presented.

The responses collected during the ith presentation (repeat) of visual stimulus are denoted by (s i t , r i t ).

Here s i t is the stimulus history which elicits population response r i t at time t.

The goal of this approach is to learn a metric such that pairs of responses generated during different repeats of the same stimulus are closer, or more similar, than pairs of responses generated by different stimuli.

We slice the data into triplets of the form (r, r + , r − ) where r and r + are responses of cells to the same stimulus while r − is the response to a different visual stimuli FIG1 .

We refer to (r, r + ) as a positive pair and (r, r − ) as a negative pair ( FIG1 ).A common method to improve the learned metrics is to choose difficult negatives as described above.

As it can be computationally demanding to mine hard negatives, we found that a much simpler strategy of randomly sampling a common set of negatives for all the positive examples in the batch is effective.

Hence we first sample positive pairs of responses corresponding to random stimulus times and a common set of negative responses generated by stimuli distinct from any stimulus for positive responses.

Hence a batch of triplets is denoted by T = {{r i , r DISPLAYFORM1 Given a training set of triplets T , the goal is to find a pseudometric such that for most (r i , r i + , {r j − }) ∈ T the distance between responses of two repeats of same stimulus is smaller than their distance to any of the irrelevant response vectors, DISPLAYFORM2 We cast the learning task as empirical risk minimization of the form, DISPLAYFORM3 where () is a differential, typically convex, relaxation of the ordering constraints from (1).

We use the following, DISPLAYFORM4 as the surrogate loss.

We set β = 10 in our implementation.

In the case of similarity learning, we swap the role of the pairs and define, DISPLAYFORM5 We implemented two parametric forms for distance and similarity functions.

The first is a quadratic form where A 0 and DISPLAYFORM6 We learn the parameters by minimizing the loss using Adagrad BID10 .

We project A onto the space of positive semi-definite matrices space after every update using singular value decomposition.

Concretely, we rewrite A as, U DU where U is a unitary matrix and D is a diagonal matrix.

We then threshold the diagonal elements of D to be non-negative.

The quadratic metric provides a good demonstration of the hypothesis that a learned metric space may be suitable.

However, a quadratic metric is not feasible for a real prosthetic device because such a metric must be trained on visually-evoked spiking activity of a neural population.

In a retinal prosthetic, such data are not available because the retina does not respond to light.

Furthermore, a quadratic model contains limited modeling capacity to capture nonlinear visual processing in the retina BID11 .To address these issues, we introduce a nonlinear embedding based on a convolutional neural network (CNN).

The CNN encodes each cell's spiking responses in an embedding space grouped by cell type and cell body location before performing a series of nonlinear operations to map the response embedding from the response space R to R p .

One benefit of this approach is that this model has an embedding dimensionality independent of the number of cells recorded while only employing knowledge of the cell body location and cell type.

The cell body location and cell type are identifiable from recordings of non-visually-evoked (spontaneous) neural activity in the retina BID25 BID35 .The resulting response metric may be generalized to blind retinas by merely providing cell center and cell type information.

That is, no visually-evoked spiking activity is necessary to train an embedding for a new retina.

Even though the model may be fit on non visually-evoked spiking activity, this model class is superior then the quadratic model when fit to a given retina.

We discuss preliminary experiments for predicting the activity in unobserved retinas in the Discussion.

We reserve a complete discussion of model architecture and training procedure for the Appendix.

In brief, we employ a hierarchical, convolutional network topology to mirror the translation invariance expected in the receptive field organization of the retina.

The convolutional network consists of 595K parameters across 7 layers and employs batch normalization to accelerate training.

Let φ(r) be the convolutional embedding of responses.

The similarity and metric learned using the convolutional network is given as - DISPLAYFORM0 We learn the parameters by minimizing the loss using Adam BID23 .

Spiking responses from hundreds of retinal ganglion cells (RGCs) in primate retina were recorded using a 512 electrode array system BID27 BID12 .

ON and OFF parasol RGC types were identified using visual stimulation with binary white noise and reverse correlation BID7 ).Since each analysis requires different stimulus conditions and numbers of cells, we leave the details of each preparation to the subsequent sections.

For each analysis, spike trains were discretized at the 120 Hz frame rate of the display (bins of 8.33ms), and responses across all the cells for 1 time bin were used to generate each training example.

In the following sections, we quantitatively assess the quality of multiple learned metrics -each metric with increasing complexity -with respect to a baseline (Hamming distance).

First, we assess the quality of the learned metrics with respect to traditional error analysis.

Second, we assess the quality of the learned embeddings with respect to optimal decoding of the stimulus.

Finally, we demonstrate the utility of the learned metric by employing the metric in a real electrical stimulation experiment.

The quality of a metric in our context can be measured by its effectiveness for determining whether a pair of firing patterns arose from the same visual stimulus or from distinct visual stimuli.

To evaluate the metric at the scale of large RGC populations, we focus our analysis on responses of a collection of 36 OFF parasol cells and 30 ON parasol cells to 99 repeats of a 10 second long white noise stimulus clip.

The responses were partitioned into training (first 8 seconds) and testing (last 2 seconds) of each trial.

We assessed a range of learned embedding models and baselines by employing receiver-operating characteristic (ROC) analysis.

Specifically, we selected the population firing pattern, r, at a particular offset in time in the experiment (corresponding to a visual stimulus history) and compared this firing pattern to two other types of firing patterns: (1) the firing pattern from the same group of cells at the same time during a second repeated presentation of the stimulus, r + ; and (2) the firing pattern at a distinct, randomly selected time point, r − .

For a given threshold, if the metric results in a correct classification of r + as the same stimulus, we termed the result a true positive.

For the same threshold, if an embedding metric incorrectly classified r − as the same stimulus, we termed it a false positive.

Note that unlike training, we do not choose a common set of negatives for testing.

FIG3 traces out the trade-off between the false positive rate and true positive rate across a range of thresholds in an assortment of embedding models for neural population activity.

Better models trace out curves that bend to the upper-left of the figure.

The line of equality indicates a model that is performing at chance.

A simple baseline model of a Hamming distance (red curve) performs least accurately.

A quadratic metric which permits variable weight for each neuron and interaction between pairs of neurons improves the performance further (green curve).

Finally, replacing a quadratic metric with a euclidean distance between embedding of responses using a convolutional neural network improves the performance further (blue curve).The ROC analysis provides strong evidence that increasingly sophisticated embedding models learn global structure above and beyond a Hamming distance metric.

We also examined how the local structure of the space is captured by the embedding metric by calculating the learned embeddings on a test dataset consisting of 99 repeats each of the 10 different visual stimuli.

We randomly selected a firing pattern r from one presentation of the stimulus, and identified k nearest neighbors according to our metric, for increasing k. Among the k nearest neighbors, we assessed precision, i.e. what fraction of the nearest neighbors correspond to 98 other presentations of the same stimulus.

A perfect learned embedding model would achieve a precision of 1 for k ≤ 98 and 98/k otherwise ( FIG3 , dashed).

We also measured recall, i.e. what fraction of the remaining 98 presentations of the same stimulus are within the k nearest neighbors.

A perfect learned embedding model would achieve recall of k/98 for k ≤ 98 and 1 otherwise ( FIG3 .

FIG3 highlights the performance of various learned methods across increasing k. The results indicate that the precision and recall are below an optimal embedding, but the convolutional metric performs better than quadratic and Hamming metrics.

To visualize the discriminability of the response metric, we embed the 99 responses to 10 distinct stimuli using t-SNE BID28 with distances estimated using the convolutional metric.

We see in FIG3 that responses corresponding to same visual stimulus (same color) cluster in the same region of embedding space reflecting the ability of the response space metric to discriminate distinct stimuli.

Although we trained the metric only based on whether pairs of responses are generated by the same stimulus, FIG3 suggests that the learned response metric provides additional discriminative stimulus information.

In the following sections, we attempt to quantitatively measure how well the response metric captures stimulus information by performing stimulus reconstruction.

Our hypothesis is that stimulus reconstruction provides a proxy for the ultimate goal of assessing perceptual similarity.

Stimulus reconstruction has a rich history in the neural coding literature and presents significant technical challenges.

To simplify the problem, we focus on linear reconstruction BID5 BID37 because the objective is clear, the problem is convex and the resulting reconstruction is information rich BID46 BID4 .

One limitation of this approach is that linear reconstruction does not capture rich nonlinearities potentially present in encoding.

For this reason, we focus subsequent analysis on the quadratic and Hamming metrics and reserve the analysis of the nonlinear embedding for future work with nonlinear reconstruction techniques (see Discussion).A technical issue that arises in the context of metric space analysis is the infeasibility of computing the embeddings for all spike patterns across large numbers of cells (e.g. 66 cells in the data of FIG3 produces 2 66 responses).

Therefore we focus on a spatially localized and overlapping population of 13 RGCs (6 ON and 7 OFF parasol cells in FIG0 ) because we can explicitly list all the 2 13 possible response patterns.

Training data was accrued from RGC responses to 5 repeated presentations of a 5 minute long white noise sequence.

The first 4 minutes of each presentation was employed for training; the last minute was employed for testing.

We examined the similarity between the decoded stimulus and the target stimulus, for responses that, according to our learned quadratic metric, are increasingly distant from the target.

FIG4 (first column, third row) shows the spatial profile of the linearly decoded target response 1 .We next calculate the distance of this target firing pattern to all 2 13 firing patterns and rank order them based on the learned metric.

FIG4 , top rows, shows firing patterns at the 1%, 2.5%, 5% and 75% percentiles.

Below these firing patterns are the associated with linearly decoded stimuli, and the errors with respect to the target firing pattern.

As we choose patterns farther from the target in terms of our metric, the distance between the decoded stimulus for the chosen firing pattern and target firing pattern systematically increases.

We quantify this observation in FIG4 by randomly selecting pairs of responses from the test data and calculating the optimal linearly decoded stimuli associated with them (see Methods).

We then plot the mean squared error (MSE) between the linearly decoded stimuli against the normalized metric distance between the responses.

The decoding error systematically increases as the metric distance between the corresponding responses increases, for both the learned quadratic metric (blue) as well the Hamming distance (green).

However, the distances generated by Hamming distance are 1 Note that the reconstruction is based on the static population response pattern.

We remove the time dimension by approximating ON and OFF parasol cell responses with a temporal filter with identical (but with oppositely signed) filters.

Subsequent analyses are performed by only decoding the temporally filtered stimulus.

The temporally filtered stimulus s is decoded as s = Ar + b , where parameters A, b are estimated from RGC recordings.

Using recorded experimental data, we now show how response metrics could improve the function of retinal prostheses by selecting optimal electrical stimulation patterns.

For a given target response, we use the learned quadratic metric to select the best electrical stimulation pattern, and evaluate the effectiveness of this approach by linearly decoding the stimulus from the elicited responses.

Calibration of RGC responses to electrical stimulation patterns was performed by repeating a given electrical stimulation pattern 25 times, at each of 40 current levels, in a random order.

Due to the limited duration of experiments, we focused on stimulation patterns in which only one electrode was active.

The data was spike sorted and spiking probability was computed for each cell by averaging across trials for each electrical stimulation pattern BID31 .

For each cell and electrode, the probability of firing as a function of stimulation current was approximated with a sigmoid function.

Since the RGC response to electrical stimulation is probabilistic, we evaluate each stimulation pattern by the expected distance between the elicited responses and the target firing pattern.

For a quadratic response metric this can be easily computed in closed form.

Given a response metric, we rank different stimulation patterns based on the expected distance to the target firing pattern.

In FIG5 and B (first columns) we show example target response patterns and the corresponding linearly decoded visual stimulus.

We then analyze the best stimulation pattern determined by the learned quadratic metric, and by the Hamming distance.

The responses sampled from the response distributions for the selected stimulation patterns are shown in FIG5 and B (second and third columns each).

We find that the linearly decoded stimuli were closer to the target when the stimulation was chosen via the learned response metric compared to the Hamming distance.

To quantify this behavior, we calculated the mean squared error between the decoded stimuli when the stimulation was chosen using the learned metric and the Hamming distance ( FIG5 ).

The learned metric and Hamming metric identify the same stimulation pattern and hence achieve the same error for 49% for the target responses observed.

However, on 33% of the target responses, the learned metric achieves lower mean squared error than the Hamming distance; conversely, the learned metric has larger MSE then Hamming distance on 18% of the target responses.

The above analysis demonstrates the benefit of using the learned metric over Hamming distance to choose the best stimulation pattern.

However, the collection of available electrical stimulation patterns might change over time due to hardware or biophysical constraints.

To assess the improvement in such cases, we next ask how well the learned metric performs relative to Hamming distance if we choose the kth best current pattern using each metric.

FIG5 ).

Increasing k for the learned metric leads to higher MSE in terms of the decoded stimulus.

Importantly, the learned metric achieves systematically lower MSE than the Hamming distance across the nearest k ≤ 10 stimulation patterns.

These results indicate that the learned metric systematically selects better electrical stimulation patterns for eliciting reasonably close firing patterns.

The learned metric approach has two major potential implications for visual neuroscience.

First, it provides a novel method to find "symbols" in the neural code of the retina that are similar in the sense that they indicate the presence of similar stimuli BID14 .

Second, it has an application to retinal prosthesis technology, in which hardware constraints demand that the set of neural responses that can be generated with a device be used to effectively transmit useful visual information.

For this application, a metric on responses that reflects visual stimulus similarity could be extremely useful.

The present approach differs from previously proposed spike train metrics (reviewed in BID51 ).

Previous approaches have employed unsupervised techniques to cluster nearby spike patterns BID14 BID34 BID15 or employed user-specified, paramteric approaches BID53 BID0 .

In the case of single snapshots in time used here, the latter approach (Victor-Purpura metric) has only one degree of freedom which is a user-specified cost associated with moving spikes from one cell to another.

In our proposed method, the relative importance of cell identity is learned directly from the statistics of population firing patterns.

The present work is a stepping stone towards building an encoding algorithm for retinal prostheses.

In this paper, we learn the metric using light evoked responses.

However, we need to estimate this metric in a blind retina, which has no light evoked responses.

The convolutional metric is adaptable to any RGC population by merely noting cell types and center locations.

Thus a convolutional metric could be trained on multiple healthy retinas and applied to a blind retina.

Preliminary results in this direction indicate that a convolutional metric trained on half of the cells in a retinal recording (training data) generalizes to the other half (validation data), yielding performance higher than a quadratic metric (and comparable to a convolutional metric) trained directly on the validation data.

Additional techniques may also be helpful in extending our method to data involving many cells, temporal responses, and additional response structure.

For example, using recurrent neural networks BID26 to embed responses may help compute distances between spiking patterns consisting of multiple time bins, perhaps of unequal length.

Boosting BID13 may help combine multiple efficiently learned metrics for a smaller, spatially localized groups of cells.

Other metrics may be developed to capture invariances learned by commonly used encoding models BID7 BID33 .

Also, triplet mining techniques (i.e., choosing hard negatives), a commonly used trick in computer vision, may improve efficiency BID38 BID32 .

Novel metrics could also be learned with additional structure in population responses, such as the highly structured correlated activity in RGCs Mastronarde (1983); BID17 .

This noise correlation structure may be learnable using negative examples that destroy the noise correlations in data while preserving light response properties, by taking responses of different cells from different repeats of the stimulus.

Note that the convolutional metric outperforms the quadratic metric at both global (ROC curves) and local (precision recall curves) scales.

However, using current retinal prosthesis technologies, we might be able to resolve information only up to a particular scale.

For current retinal prostheses, capturing global structure may be of greatest importance, because state-of-the-art technology has a relatively coarse vocabulary for stimulating RGCs (Humayun et al., 2012; BID58 ) (see also FIG0 ).

Specifically, the "nearest" elicited firing pattern is "far" in terms of the corresponding visual stimulus ( FIG5 ) .

In terms of the proposed learned metric, the nearest feasible firing pattern achievable by electrical stimulation in our experiments is at the 10th percentile of all possible firing patterns.

In this context, the average closest stimulation pattern, expressed as a percentile of the learned metric distances, provides a valuable benchmark to measure the performance of a prosthesis and how that performance is affected by advances in the underlying hardware and software.

In particular, the network employs knowledge of the receptive field locations and firing rates of individual cells but the network is independent of the number of cells in the retina.

The latter point is achieved by embedding the responses of neurons into pathways grouped by cell type.

In our experiments, we focus on 2 cell types (ON and OFF parasols), thus we employ a 2 channel pathway BID22 .The network receives as input the spiking activity of ON and OFF parasols and embeds these spike patterns as one-hot vectors placed at the spatial locations of each cell's receptive field.

The resulting pattern of activations is summed across all cells in the ON and OFF populations, respectively, and passed through several convolutional layers of a network.

Successive layers shrink the spatial activation size of the representation, while increasing the number of filter channels BID24 BID44 .

The final embedding response vector has 1/16th number of pixels in the stimulus and represents the flattened representation of the last layer of the network.

Let c denote the number of different cells.

The RGC population response is a vector r ∈ {0, 1} c .• Represent responses as vectors over {+1, −1} withr = 2(r − 0.5).• Compute the scale for each cell as a function of the mean firing rate: DISPLAYFORM0 • Map each cell to its center location on a grid with spatial dimensions same as those of visual stimulus.

Let M i be grid embedding on cell i.

So, M i has zero for all positions except center of cell.• Perform a separable 5 × 5 convolution of stride 1 on each M i to get RF estimate of cell,M i .•

Add the activation of cells of the same type to get the total activation for a given cell type.

Hence, activation map for each cell type A i = ir i s iMi .

Subsequent layers receive input as a two layered activation map corresponding to ON and OFF parasol cells.• The convolutional layers further combine information accross multiple cells, of different types.

The details of different layers are shown in FIG6 and Normalization Batch normalization after every convolution Optimizer Adam BID23 ) (α = 0.01, β 1 = 0.9, β 2 = 0.999) Parameter updates 20,000Batch size 100 Weight initialization Xavier initialization BID16

For the latter analyses assesing the quality of metric, we reconstruct the stimulus from neural responses with linear decoding.

In this section we demonstrate that even though the linear decoder is rather simplistic, the reconstructions are on-par with a non-parametric decoding method which averages the stimulus corresponding to the response pattern.

In FIG7 A, we see that the linear decoder has very similar spatial structure to the non-parametric decoder.

To quantify this, we compute the mean-squared error between the two methods of decoding, normalized by the magnitude of non-parametric decoder FIG7 B, blue dots).

The error of linear decoding is comparable to error between two non-parametric decodings computed using independent samples of stimuli FIG7 B, green dots).

These observations show that linear decoder is a reasonable first-order approximation of encoded stimulus.

The relative MSE between two linear and non-parametric decoding (y-axis) v/s number of averaged spikes (x-axis) for different response patterns (blue dots).

The relative MSE between non-parametric decoding using two independent set of response samples is also shown (green).

<|TLDR|>

@highlight

Using triplets to learn a metric for comparing neural responses and improve the performance of a prosthesis.

@highlight

Authors develop new spike train distance metrics, including neural networks and quadratic metrics. These metrics are shown to outperform the naive Hamming distance metric, and implicitly captures some structure in neural code.

@highlight

With the application of improving neural prosthesis in mind, the authors propose to learn a metric between neural responses by either optimizing a quadratic form or a deep neural network .