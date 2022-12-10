We develop a framework for understanding and improving recurrent neural networks (RNNs) using max-affine spline operators (MASOs).

We prove that RNNs using piecewise affine and convex nonlinearities can be written as a simple piecewise affine spline operator.

The resulting representation provides several new perspectives for analyzing RNNs, three of which we study in this paper.

First, we show that an RNN internally partitions the input space during training and that it builds up the partition through time.

Second, we show that the affine slope parameter of an RNN corresponds to an input-specific template, from which we can interpret an RNN as performing a simple template matching (matched filtering) given the input.

Third, by carefully examining the MASO RNN affine mapping, we prove that using a random initial hidden state corresponds to an explicit L2 regularization of the affine parameters, which can mollify exploding gradients and improve generalization.

Extensive experiments on several datasets of various modalities demonstrate and validate each of the above conclusions.

In particular, using a random initial hidden states elevates simple RNNs to near state-of-the-art performers on these datasets.

Recurrent neural networks (RNNs) are a powerful class of models for processing sequential inputs and a basic building block for more advanced models that have found success in challenging problems involving sequential data, including sequence classification (e.g., sentiment analysis BID30 , sequence generation (e.g., machine translation BID1 ), speech recognition BID10 , and image captioning BID22 .

Despite their success, however, our understanding of how RNNs work remains limited.

For instance, an attractive theoretical result is the universal approximation property that states that an RNN can approximate an arbitrary function BID28 BID29 BID11 .

These classical theoretical results have been obtained primarily from the dynamical system BID29 BID28 and measure theory BID11 perspectives.

These theories provide approximation error bounds but unfortunately limited guidance on applying RNNs and understanding their performance and behavior in practice.

In this paper, we provide a new angle for understanding RNNs using max-affine spline operators (MASOs) BID21 BID12 ) from approximation theory.

The piecewise affine approximations made by compositions of MASOs provide a new and useful framework to study neural networks.

For example, BID4 ; BID2 have provided a detailed analysis in the context of feedforward networks.

Here, we go one step further and find new insights and interpretations from the MASO perspective for RNNs.

We will see that the input space partitioning and matched filtering links developed in BID4 ; BID2 extend to RNNs and yield interesting insights into their inner workings.

Moreover, the MASO formulation of RNNs enables us to theoretically justify the use of a random initial hidden state to improve RNN performance.

For concreteness, we focus our analysis on a specific class of simple RNNs BID8 with piecewise affine and convex nonlinearities such as the ReLU BID9 .

RNNs with such nonlinearities have recently gained considerable attention due to their ability to combat the exploding gradient problem; with proper initialization BID19 BID33 and clever parametrization of the recurrent weight BID0 BID39 BID16 BID15 BID24 BID13 , these RNNs achieve performance on par with more complex ones such as LSTMs.

Below is a summary of our key contributions.

Contribution 1.

We prove that an RNN with piecewise affine and convex nonlinearities can be rewritten as a composition of MASOs, making it a piecewise affine spline operator with an elegant analytical form (Section 3).Contribution 2.

We leverage the partitioning of piecewise affine spline operators to analyze the input space partitioning that an RNN implicitly performs.

We show that an RNN calculates a new, high-dimensional representation (the partition code) of the input sequence that captures informative underlying characteristics of the input.

We also provide a new perspective on RNN dynamics by visualizing the evolution of the RNN input space partitioning through time (Section 4).Contribution 3.

We show the piecewise affine mapping in an RNN associated with a given input sequence corresponds to an input-dependent template, from which we can interpret the RNN as performing greedy template matching (matched filtering) at every RNN cell (Section 5).Contribution 4.

We rigorously prove that using a random (rather than zero) initial hidden state in an RNN corresponds to an explicit regularizer that can mollify exploding gradients.

We show empirically that such a regularization improves RNN performance (to state-of-the-art) on four datasets of different modalities (Section 6).

Recurrent Neural Networks (RNNs).

A simple RNN unit BID8 per layer and time step t, referred to as a "cell," performs the following recursive computation DISPLAYFORM0 where h ( ,t) is the hidden state at layer and time step t, h (0,t) := x (t) which is the input sequence, σ is an activation function and W ( ) , W ( ) r , and b ( ) are time-invariant parameters at layer .

h ( ,0) is the initial hidden state at layer which needs to be set to some value beforehand to start the RNN recursive computation.

Unrolling the RNN through time gives an intuitive view of the RNN dynamics, which we visualize in FIG0 .

The output of the overall RNN is typically an affine transformation of the hidden state of the last layer L at time step t DISPLAYFORM1 In the special case where the RNN has only one output at the end of processing the entire input sequence, the RNN output is an affine transformation of the hidden state at the last time step, i.e., DISPLAYFORM2 independent max-affine splines BID21 , each with R partition regions.

Its output for output dimension k is produced via DISPLAYFORM3 where x ∈ R D and y ∈ R K are dummy variables that respectively denote the input and output of the MASO S and ·, · denotes inner product.

The three subscripts of the "slope" parameter [A] k,r,d correspond to output k, partition region r, and input signal index d. The two subscripts of the "bias" parameter [B] k,r correspond to output k and partition region r.

We highlight two important and interrelated MASO properties relevant to the discussions throughout the paper.

First, a MASO performs implicit input space partitioning, which is made explicit by rewriting (3) as DISPLAYFORM4 where Q ∈ R K×R is a partition selection matrix 1 calculated as DISPLAYFORM5 Namely, Q contains K stacked one-hot row vectors, each of which selects the [r * ]

th k partition of the input space that maximizes (4) for output dimension k. As a consequence, knowing Q is equivalent to knowing the partition of an input x that the MASO implicitly computes.

We will use this property in Section 4 to provide new insights into RNN dynamics.

Second, given the partition r * that an input belongs to, as determined by FORMULA5 , the output of the MASO of dimension k from (3) reduces to a simple affine transformation of the input DISPLAYFORM6 Here, the selected affine parameters A ∈ R K×D and B ∈ R K are specific to the input's partition region th k column of A and B, respectively, for output dimension k. We emphasize that A and B are input-dependent; different inputs x induce different A and B.2 We will use this property in Section 5 to link RNNs to matched filterbanks.

We now leverage the MASO framework to rewrite, interpret, and analyze RNNs.

We focus on RNNs with piecewise affine and convex nonlinearities in order to derive rigorous analytical results.

The analysis of RNNs with other nonlinearities is left for future work.

We first derive the MASO formula for an RNN cell (1) and then extend to one layer of a timeunrolled RNN and finally to a multi-layer, time-unrolled RNN.

Let z DISPLAYFORM0 be the input to an RNN cell that is the concatenation of the current input h ( −1,t) and the previous hidden state h ( ,t−1) .

Then we have the following result, which is a straightforward extension of Proposition 4 in BID4 .

Proposition 1.

An RNN cell of the form (1) is a MASO with DISPLAYFORM1 where DISPLAYFORM2 is the affine parameter corresponding to the piecewise affine and convex nonlinearity σ(·)that depends on the cell input z ( ,t) .We now derive an explicit affine formula for a time-unrolled RNN at layer .

Let DISPLAYFORM3 be the entire input sequence to the RNN at layer , and let DISPLAYFORM4 be all the hidden states that are output at layer .

After some algebra and simplification, we arrive at the following result.

Theorem 1.

The th layer of an RNN is a piecewise affine spline operator defined as DISPLAYFORM5 . . .

DISPLAYFORM6 . . .

DISPLAYFORM7 . . .

DISPLAYFORM8 where A ( ) We present the proof for Theorem 1 in Appendix G. The key point here is that, by leveraging MASOs, we can represent the time-unrolled RNN as a simple affine transformation of the entire input sequence (8).

Note that this affine transformation changes depending on the partition region in which the input belongs (recall FORMULA4 and FORMULA5 ).

Note also that the initial hidden state affects the layer output by influencing the affine parameters and contributing a bias term A DISPLAYFORM9 DISPLAYFORM10 We study the impact of the initial hidden state in more detail in Section 6.We are now ready to generalize the above result to multi-layer RNNs.

Let ) be the input sequence to a multi-layer RNN, and let DISPLAYFORM11 DISPLAYFORM12 be the output sequence.

We state the following result for the overall mapping of a multi-layer RNN.

Theorem 2.

The output of an L-layer RNN is a piecewise affine spline operator defined as DISPLAYFORM13 where Theorem 2 shows that, using MASOs, we have a simple, elegant, and closed-form formula showing that the output of an RNN is computed locally via very simple functions.

This result is proved by recursively applying the proof for Theorem 1.

DISPLAYFORM14 The affine mapping formula (9) opens many doors for RNN analyses, because we can shed light on RNNs by applying established matrix results.

In the next sections, we provide three analyses that follow this programme.

First, we show that RNNs partition the input space and that they develop the partitions through time.

Second, we analyze the forms of the affine slope parameter and link RNNs to matched filterbanks.

Third, we study the impact of the initial hidden state to rigorously justify the use of randomness in initial hidden state.

From this point, for simplicity, we will assume a zero initial hidden state unless otherwise stated.

The MASO viewpoint enables us to see how an RNN implicitly partitions its input sequence through time, which provides a new perspective of its dynamics.

To see this, first recall that, for an RNN cell, the piecewise affine and convex activation nonlinearity partitions each dimension of the cell input z ( ,t) into R regions (for ReLU, R = 2).

Knowing the state of the nonlinearity (which region r is activated) is thus equivalent to knowing the partition of the cell input.

For a multi-layer RNN composed of many RNN cells (recall FIG0 , the input sequence partition can be retrieved by accessing the collection of the states of all of the nonlinearities; each input sequence can be represented by a partition "code" that determines the partition to which it belongs.

Since an RNN processes an input sequence one step at a time, the input space partition is gradually built up and refined through time.

As a consequence, when seen through the MASO lens, the forward pass of an RNN is simply developing and refining the partition code of the input sequence.

Visualizing the evolution of the partition codes can be potentially beneficial for diagnosing RNNs and understanding their dynamics.

As an example, we demonstrate the evolution of the partition codes of a one-layer ReLU RNN trained on the MNIST dataset, with each image flattened into a 1-dimensional sequence so that input at each time step is a single pixel.

Details of the model and experiments are in Appendix C. Since the ReLU activation partitions its input space into only 2 regions , we can retrieve the RNN partition codes of the input images simply by binarizing and concatenating all of the hidden states.

FIG2 visualizes how the partition codes of MNIST images evolve through time using t-SNE, a distance-preserving dimensionality reduction technique BID35 ).

The figure clearly shows the evolution of the partition codes from hardly any separation between classes of digits to forming more and better separated clusters through time.

We can also be assured that the model is well-behaved, since the final partition shows that the images are well clustered based on their labels.

Additional visualizations are available in Section D.

The MASO viewpoint enables us to connect RNNs to classical signal processing tools like the matched filter.

Indeed, we can directly interpret an RNN as a matched filterbank, where the classification decision is informed via the simple inner product between a "template" and the input sequence.

To see this, we follow an argument similar to that in Section 4.

First, note that the slope parameter A ( ,t) for each RNN cell is a "locally optimal template" because it maximizes each of its output dimensions over the R regions that the nonlinearity induces (recall (3) and FORMULA8 ).

For a multi-layer RNN composed of many RNN cells, the overall "template" A RNN corresponds to the composition of the optimal templates from each RNN cell, which can be computed simply via dz/dx (recall (9)).

Thus, we can view an RNN as a matched filterbank whose output is the maximum inner product between the input and the rows of the overall template A RNN BID36 BID37 .

The overall template is also known in the machine learning community as a salience map; see BID20 for an example of using saliency maps to visualize RNNs.

Our new insight here is that a good template produces a larger inner product with the input regardless of the visual quality of the template, thus complementing prior work.

The template matching view of RNNs thus provides a principled methodology to visualize and diagnose RNNs by examining the inner products between the inputs and the templates.

To illustrate the matched filter interpretation, we train a one-layer ReLU RNN on the polarized Stanford Sentiment Treebank dataset (SST-2) BID30 , which poses a binary classification Figure 3 : Templates corresponding to the correct (left) and incorrect class (right) of a negative sentiment input from the SST-2 dataset.

Each column contains the gradient corresponding to an input word.

Quantitatively, we can see that the inner product between input and the correct class template (left) produces a larger value than that between input and the incorrect class template (right).problem, and display in Figure 3 the templates corresponding to the correct and incorrect classes of an input where the correct class is a negative sentiment.

We see that the input has a much larger inner product with the template corresponding to the correct class (left plot) than that corresponding to the incorrect class (right plot), which informs us that the model correctly classifies this input.

Additional experimental results are given in Appendix E.

In this section, we provide a theoretical motivation for the use of a random initial hidden state in RNNs.

The initial hidden state needs to be set to some prior value to start the recursion (recall Section 2).

Little is understood regarding the best choice of initial hidden state other than Zimmermann et al. FORMULA0 's dynamical system argument.

Consequently, it is typically simply set to zero.

Leveraging the MASO view of RNNs, we now demonstrate that one can improve significantly over a zero initial hidden state by using a random initial hidden state.

This choice regularizes the affine slope parameter associated with the initial hidden state and mollifies the so-called exploding gradient problem BID25 .Random Initial Hidden State as an Explicit Regularization.

We first state our theoretical result that using random initial hidden state corresponds to an explicit regularization and then discuss its impact on exploding gradients.

Without loss of generality, we focus on one-layer ReLU RNNs.

Let N be the number of data points and C the number of classes.

Define DISPLAYFORM0 Theorem 3.

Let L be an RNN loss function, and let L represent the modified loss function when the RNN initial hidden state is set to a Gaussian random vector ∼ N(0, σ 2 I) with small standard deviation σ .

Then we have that E L = L + R. For the cross-entropy loss L with softmax output, DISPLAYFORM1 , where y ni is the i th dimension of the softmax output of the n th data point and i, j ∈ {1, . . .

, C} are the class indices.

For the mean-squared error loss L, DISPLAYFORM2 We prove this result for the cross-entropy loss in Appendix G.2.

The standard deviation σ controls the importance of the regularization term and recovers the case of standard zero initial hidden state when σ = 0.Connection to the Exploding Gradient Problem.

Backpropagation through time (BPTT) is the default RNN training algorithm.

Updating the recurrent weight W r with its gradient using BPTT involves calculating the gradient of the RNN output with respect to the hidden state at each time step DISPLAYFORM3 DISPLAYFORM4 σ W r in (10) blows up, which leads to unstable training.

This is known as the exploding gradient problem BID25 .

Our key realization is that the gradient of the RNN output with respect to the initial hidden state h (0) features the term A h from Theorem 3 DISPLAYFORM5 Of all the terms in (10), this one involves the most matrix products and hence is the most erratic.

Fortunately, Theorem 3 instructs us that introducing randomness into the initial hidden state effects a regularization on A h and hence tamps down the gradient before it can explode.

An interesting direction for future work is extending this analysis to every term in (10).Experiments.

We now report on the results of a number of experiments that indicate the significant performance gains that can be obtained using a random initial hidden state of properly chosen standard deviation σ .

Unless otherwise mentioned, in all experiments we use ReLU RNNs with 128-dimensional hidden states and with the recurrent weight matrix W ( ) r initialized as an identity matrix BID19 BID33 .

We summarize the experimental results; experimental details and additional results are available in Appendices C and F.Visualizing the Regularizing Effect of a Random Initial Hidden State.

We first consider a simulated task of adding 2 sequences of length 100.

This is a ternary classification problem with input X ∈ R 2×T and target y ∈ {0, 1, 2}, y = i 1 X2i=1 X 1i .

The first row of X contains randomly chosen 0's and 1's; the second row of X contains 1's at 2 randomly chosen indices and 0's everywhere else.

Prior work treats this task as a regression task BID0 ; our regression results are provided in Appendix F.1.In FIG3 , we visualize the norm of A h , the norm of the recurrent weight gradient dL dWr , and the validation loss against training epochs for various random initial state standard deviations.

The top two plots clearly demonstrate the effect of the random initial hidden state in regularizing both A h and the norm of the recurrent weight gradient, since larger σ reduces the magnitudes of both A h and dL dWr .

Notably, the reduced magnitude of the gradient term dL dWr empirically demonstrates the mollification of the exploding gradient problem.

The bottom plot shows that setting σ too large can negatively impact learning.

This can be explained as having too much regularization effect.

This suggests the question of choosing the best value of σ in practice, which we now investigate.

Choosing the Standard Deviation of the Random Initial Hidden State.

We examine the effect on performance of different random initial state standard deviations σ in RNNs using RMSprop and SGD with varying learning rates.

We perform experiments on the MNIST dataset with each image flattened to a length 784 sequence (recall Section 4).

The full experimental results are included in Appendix F.2; here, we report two interesting findings.

First, for both optimizers, using a random initial hidden state permits the use of higher learning rates that would lead to an exploding gradient when training without a random initial hidden state.

Second, RMSprop is less sensitive to the choice of σ than SGD and achieves favorable accuracy even when σ is very large (e.g., σ = 5).

This might be due to the gradient smoothing that RMSprop performs during optimization.

We therefore recommend the use of RMSprop with a random initial hidden state to improve model performance.

BID0 0.951 0.914 -scoRNN BID13 0.985 0.966 -C-LSTM BID41 --0.878 Tree-LSTM BID32 --0.88 Bi-LSTM+SWN-Lex BID34 - DISPLAYFORM6 We used RMSprop to train ReLU RNNs of one and two layers with and without random initial hidden state on the MNIST, permuted MNIST 4 and SST-2 datasets.

TAB2 shows the classification accuracies of these models as well as a few state-of-the-art results using complicated models.

It is surprising that a random initial hidden state elevates the performance of a simple ReLU RNN to near state-of-the-art performance.

Random Initial Hidden State in Complex RNN Models.

Inspired by the results of the previous experiment, we integrated a random initial hidden state into some more complex RNN models.

We first evaluate a one-layer gated recurrent unit (GRU) on the MNIST and permuted MNIST datasets, with a random and zero initial hidden state.

Although the performance gains are not quite as impressive as those for ReLU RNNs, our results for GRUs still show worthwhile accuract improvements, from 0.986 to 0.987 for MNIST and from 0.888 to 0.904 for permuted MNIST.We continue our experiments with a more complex, convolutional-recurrent model composed of 4 convolution layers followed by 2 GRU layers BID5 and the Bird Audio Detection Challenge dataset.5 This binary classification problem aims to detect whether or not an audio recording contains bird songs; see Appendix C for the details.

We use the area under the ROC curve (AUC) as the evaluation metric, since the dataset is highly imbalanced.

Simply switching from a zero to a random initial hidden state provides a significant boost in the AUC: from 90.5% to 93.4%.

These encouraging preliminary results suggest that, while more theoretical and empirical investigations are needed, a random initial hidden state can also boost the performance of complicated RNN models that are not piecewise affine and convex.

We have developed and explored a novel perspective of RNNs in terms of max-affine spline operators (MASOs).

RNNs with piecewise affine and convex nonlinearities are piecewise affine spline operators with a simple, elegant analytical form.

The connections to input space partitioning (vector quantization) and matched filtering followed immediately.

The spline viewpoint also suggested that the typical zero initial hidden state be replaced with a random one that mollifies the exploding gradient problem and improves generalization performance.

There remain abundant promising research directions.

First, we can extend the MASO RNN framework following BID3 to cover more general networks like gated RNNs (e.g, GRUs, LSTMs) that employ the sigmoid nonlinearity, which is neither piecewise affine nor convex.

Second, we can apply recent random matrix theory results BID23 to the affine parameter A RNN (e.g., the change of the distribution of its singular values during training) to understand RNN training dynamics.

t th time step of a discrete time-serie, DISPLAYFORM0 x Concatenation of the whole length T time-serie: DISPLAYFORM1 Output/prediction associated with input x y n True label (target variable) associated with the nth time-serie example x n .

For classification y n ∈ {1, . . .

, C}, C > 1; For regression y n ∈ R C , C ≥ 1 DISPLAYFORM2 Output of an RNN cell at layer and time step t; Alternatively, input to an RNN cell at layer + 1 and time step t − 1 DISPLAYFORM3 Concatenation of hidden state h ( ,t) of all time steps at layer : DISPLAYFORM4 Concatenated input to an RNN cell at layer and time step t: DISPLAYFORM5 th layer RNN weight associated with the input h ( ,t−1) from the previous time step: DISPLAYFORM6 th layer RNN weight associated with the input h ( −1,t) from the previous layer: DISPLAYFORM7 Bias of the last fully connected layer: DISPLAYFORM8 Pointwise nonlinearity in an RNN (assumed to be piecewise affine and convex in this paper) σ Standard deviation of noise injected into the initial hidden state h DISPLAYFORM9 MASO formula of the RNN activation σ(·) at layer and time step t: DISPLAYFORM10 MASO parameters of an RNN at layer and time step t: DISPLAYFORM11

Below we describe the datasets and explain the preprocessing steps for each dataset.

MNIST.

The dataset 6 consists of 60k images in the training set and 10k images in the test set.

We randomly select 10k images from the training set as validation set.

We flatten each image to a 1-dimensional vector of size 784.

Each image is also centered and normalized with mean of 0.1307 and standard deviation of 0.3081 (PyTorch default values).permuted MNIST.

We apply a fixed permutation to all images in the MNIST dataset to obtain the permuted MNIST dataset.

SST-2.

The dataset 7 consists of 6920, 872, 1821 sentences in the training, validation and test set, respectively.

Total number of vocabulary is 17539, and average sentence length is 19.67.

Each sentence is minimally processed into sequences of words and use a fixed-dimensional and trainable vector to represent each word.

We initialize these vectors either randomly or using GloVe BID26 .

Due to the small size of the dataset, the phrases in each sentence that have semantic labels are also used as part of the training set in addition to the whole sentence during training.

Dropout of 0.5 is applied to all experiments.

Phrases are not used during validation and testing, i.e., we always use entire sentences during validation and testing.

Bird Audio Dataset.

The dataset 8 consists of 7, 000 field recording signals of 10 seconds sampled at 44 kHz from the Freesound BID31 audio archive representing slightly less than 20 hours of audio signals.

The audio waveforms are extracted from diverse scenes such as city, nature, train, voice, water, etc., some of which include bird sounds.

The labels regarding the bird detection task can be found in the file freefield1010.

Performance is measured by Area Under Curve (AUC) due to the unbalanced distribution of the classes.

We preprocess every audio clip by first using short-time Fourier transform (STFT) with 40ms and 50% overlapping Hamming window to obtain audio spectrum and then by extracting 40 log mel-band energy features.

After preprocessing, each input is of dimension D = 96 and T = 999.

Experiment setup for various datasets is summarized in TAB3 .

Some of the experiments do not appear in the main text but in the appendix; we include setup for those experiments as well.

A setting common to all experiments is that we use learning rate scheduler so that when validation loss plateaus for 5 consecutive epochs, we reduce the current learning rate by a factor of 0.7.Setup of the experiments on influence of various standard deviations in random initial hidden state under different settings.

We use σ chosen in {0.001, 0.01, 0.1, 1, 5} and learning rates in {1×10 −5 , 1×10 −4 , 1.5×10 −4 , 2×10 −4 } for RMSprop and {1×10 −7 , 1×10 −6 , 1.25×10 −6 , 1.5× 10 −6 } plain SGD.Setup of input space partitioning experiments.

For the results in the main text, we use t-SNE visualization BID35 with 2 dimensions and the default settings from the python sklearn package.

Visualization is performed on the whole 10k test set images.

For finding the nearest neighbors of examples in the SST-2 dataset, since the examples are of varying lengths, we constrain the distance comparison to within +/-10 words of the target sentence.

When the sentence lengths are not the same, we simply pad the shorter ones to the longest one, then process it with RNN and finally calculate the distance as the 2 distance of the partition codes (i.e., concatenation of all hidden states) that RNN computes.

We justify the comparison between examples of different lengths using padding by noting that batching examples and padding the examples to the longest example within a batch has been a common practice in modern natural language processing tasks.

Setup of exploratory experiments.

We experimented with one-layer GRU with 128 hidden unites for MNIST and permuted MNIST datasets.

We use RMSprop optimizer with an initial learning rate of 0.001.

We experimented with various standard deviations in random initial hidden state including {0.01, 0.05, 0.1, 0.5, 1, 5}. The optimal standard deviations that produce the results in the main text are σ = {0.01, 0.05, 0.01}, for MNIST, permuted MNIST and bird detection datasets, respectively.

We provide ample additional visualizations to demonstrate the partition codes that an RNN computes on its input sequences.

Here, the results are focused more on the properties of the final partition 7 https://nlp.stanford.edu/sentiment/index.html 8 http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/ Figure 5: Visualization of partition codes for pixel-by-pixel (i.e., flattened to a 1-dimensional, length 784 vector) MNIST dataset using a trained ReLU RNN (one layer, 128-dimensional hidden state).

Here, we visualize the nearest 5 and farthest 5 images of one selected image from each class.

The distance is computed using the partition codes of the images.

Leftmost column is the original image; the middle 5 images are the 5 nearest neighbors; the rightmost 5 images are the farthest neighbors.codes computed after the RNN processes the entire input sequence rather than part of the input sequence.

Several additional sets of experimental results are shown; the first three on MNIST and the last one on SST-2.First, we visualize the nearest and farthest neighbors of several MNIST digits in Figure 5 .

Distance is calculated using the partition codes of the images.

The left column is the original image; The next five columns are the five nearest neighbors to the original image; The last five columns are five farthest neighbors.

This figure shows that partition codes the images are well clustered.

Second, we show the two dimensional projection using t-SNE of the raw pixel and VQ representations of each data points in the MNIST dataset and visualize them in Figure 6 .

We clearly see a more distinct clustering using VQ representation of the data than using the raw pixel representation.

This comparison demonstrate the ability of the RNN to extract useful information from the raw representation of the data in the form of VQ.Third, we perform a KNN classification with k ∈ {1, 2, 5, 10} using 1) the RNN computed partition codes of the inputs and 2) raw pixel data representation the MNIST test set to illustrate that the data reparametrized by the RNN has better clustering property than the original data representations.

We use 80% of the test set to train the classifier and the rest for testing.

The results are reported in Table 3 .

We see that the classification accuracies when using RNN computed partition codes of the inputs are significantly higher than those when using raw pixel representations.

This result again shows the superior quality of the input space partitioning that RNN produces, and may suggest a new way to improve classification accuracy by just using the reparametrized data with a KNN classifier.

Finally, we visualize the 5 nearest and 5 farthest neighbors of a selected sentence from the SST-2 dataset to demonstrate that the partitioning effect on dataset of another modality.

Again, the distances are computed using the partition codes of the inputs.

The results are shown in FIG6 .Figure 6: t-SNE visualization of MNIST test set images using raw pixel representation (left) and RNN VQ representation (right).

We see more distinct clusters in the t-SNE plot using RNN VQ representation of images than the raw pixel representation, implying the useful information that RNN extracts in the form of VQ.

Table 3 : K-nearest neighbor classification accuracies using data reparametrized by RNN compared to those using raw pixel data.

We can see that classification accuracies using RNN reparametrized data are much higher than those using raw pixel data for all k's.

We can see that all sentences that are nearest neighbors are of similar sentiment to the target sentence, whereas all sentences that are farthest neighbors are of the opposite sentiment.

We provide here more templates on images and texts in FIG7 .

Notice here that, although visually the templates may look similar or meaningless, they nevertheless have meaningful inner product with the input.

The class index of the template that produces the largest inner product with the input is typically the correct class, as can be seen in the two figures.

We present the regularization effect on adding task formulated as a regression problem, following setup in BID0 .

Result is shown in FIG0 .

We see regularization effect similar to that presented in FIG3 , which demonstrates that the regularization effect does indeed happens for both classification and regression problems, as Thm.

3 suggests.

TAB6 shows the classification accuracies under various settings.

The discussion of the results is in Section 6.

It is a film that will have people walking out halfway through , will encourage others to stand up and applaud , and will , undoubtedly , leave both camps engaged in a ferocious debate for years to come .

(+) Cute , funny , heartwarming digitally animated feature film with plenty of slapstick humor for the kids , lots of in-jokes for the adults and heart enough for everyone . (+, 22.61) This is a great subject for a movie , but Hollywood has squandered the opportunity , using it as a prop for warmed-over melodrama and the kind of choreographed mayhem that director John Woo has built his career on . (-, 37.24) Though it is by no means his best work , LaissezPasser is a distinguished and distinctive effort by a bona-fide master , a fascinating film replete with rewards to be had by all willing to make the effort to reap them .

To simplify notation, similar to the main text, in the proof here we drop the affine parameters' dependencies on the input, but keep in mind the input-dependency of these parameters.

The template that has the bigger inner product is the true class of the sentence.

We see that the template corresponding to the correct class produces a significantly bigger inner product with the input than other templates.

We first derive the expression for a hidden state h ( ,t) at a given time step t and layer .

Using Prop.

1, we start with unrolling the RNN cell of layer at time step t for two time steps to t − 2 by recursively applying (1) as follows: Repeat the above derivation for t ∈ {1, · · · , T } and stack h ( ,t) in decreasing time steps from top to bottom, we have: DISPLAYFORM0 . . .

DISPLAYFORM1 . . .

DISPLAYFORM2 where DISPLAYFORM3 . . .

DISPLAYFORM4 . . .

DISPLAYFORM5 which concludes the proof.

Thm.

2 follows from the recursive application of the above arguments for each layer ∈ {1, · · · , L}.

We prove for the case of multi-class classification problem with softmax output.

The proof for the case of regression problems easily follows. ] be the concatenation of the n th input sequence of length T and c be the index of the correct class.

We assume the amplitude of random initial hidden state is small so that the input-dependent affine parameter A h , which also depends on h (0) , does not change when using random h (0) .

Also, let z n = f RNN (x n , h (0) ) be the overall RNN computation that represents (9).We first rewrite the cross entropy loss with random initial hidden state L CE = L CE softmax f RNN x n , h (0) + as follows: DISPLAYFORM0 Taking expectation of the L with respect to the distribution of the random Gaussian vector that the initial hidden state is set to, we have DISPLAYFORM1 where DISPLAYFORM2 We note that similar forms of (21) have been previously derived by BID38 .We now simplify (21) using second order Taylor expansion on h (0) of the summation inside the log function.

Define function u(x (1:T ) n , h (0) ) := log( j exp(z nj )) = log( j exp(f (x (1:T ) n , h (0) ))).

Then, we can approximate (21) as follows: DISPLAYFORM3 dh FORMULA21 DISPLAYFORM4 where du(xn,h DISPLAYFORM5 is the Hessian matrix: DISPLAYFORM6 = y ni (1 i=l − y nl ) a i , a l .

Then, we can write the trace term in (22) as follows: DISPLAYFORM7 As a result, using the above approximations, we can rewrite the loss with random initial state in (19) as: DISPLAYFORM8 We see that this regularizer term does not dependent on the correct class index c of each data points.

The problem of exploding gradients has been widely studied from different perspectives.

First approaches have attempted to directly control the amplitude of the gradient through gradient clipping BID25 .

A more model driven approach has leveraged the analytical formula of the gradient when using specific nonlinearities and topologies in order to develop parametrization of the recurrent weights.

This has led to various unitary reparametrizations of the recurrent weight BID0 BID39 BID13 BID14 BID16 BID24 BID15 BID17 .

A soft version of such parametrization lies in regularization of the DNNs.

This includes dropout applied to either the output layer BID27 or hidden state BID40 , noisin BID7 , zoneout BID18 and recurrent batch normalization BID6 .

Lastly, identity initialization of ReLU RNNs has been studied in BID19 and BID33 .

Our results complements prior works in that simply using random initial hidden state instead of zero initial hidden state and without changing the RNN structure also relieves the exploding gradient problem by regularization the potentially largest term in the recurrent weight gradient.

<|TLDR|>

@highlight

We provide new insights and interpretations of RNNs from a max-affine spline operators perspective.

@highlight

Rewrites equations of Elman RNN in terms of so-called max-affine spline operators

@highlight

Provide a novel approach toward understanding RNNs using max-affline spline operators (MASO) by rewriting them with piecewise affine and convex activations MASOs

@highlight

The authors build upon max-affine spline operator interpetation of a substantial class of deep networks, focusing on Recurrent Neural Networks using noise in initial hidden state acts as regularization