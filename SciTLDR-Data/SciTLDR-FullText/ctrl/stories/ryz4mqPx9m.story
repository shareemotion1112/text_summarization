Despite rapid advances in speech recognition, current models remain brittle to superficial perturbations to their inputs.

Small amounts of noise can destroy the performance of an otherwise state-of-the-art model.

To harden models against background noise, practitioners often perform data augmentation, adding artificially-noised examples to the training set, carrying over the original label.

In this paper, we hypothesize that a clean example and its superficially perturbed counterparts shouldn't merely map to the same class--- they should map to the same representation.

We propose invariant-representation-learning (IRL): At each training iteration, for each training example, we sample a noisy counterpart.

We then apply a penalty term to coerce matched representations at each layer (above some chosen layer).

Our key results, demonstrated on the LibriSpeech dataset are the following: (i) IRL significantly reduces character error rates (CER)on both `clean' (3.3% vs 6.5%) and `other' (11.0% vs 18.1%) test sets; (ii) on several out-of-domain noise settings (different from those seen during training), IRL's benefits are even more pronounced.

Careful ablations confirm that our results are not simply due to shrinking activations at the chosen layers.

The IRL algorithm is simple: First, during training, for each example x, we produce a noisy version 90 by sampling from x ∼ ν(x), where ν is a stochastic function.

In our experiments, this function takes 91 a random snippet from a noise database, sets its amplitude by drawing from a normal distribution, and 92 adds it to the original (in sample space), before converting to spectral features.

We then incorporate a 93 penalty term in our loss function to penalize the distance between the encodings of the original data 94 point φ e (x) and the noisy data point φ e (x ), where φ l is representation at layer l.

In our experiments, 95 we choose φ e to be the output of the encoder in our Seq2Seq model.

We illustrate the learning setup 96 graphically in FIG0 .

In short, our loss function consists of three terms, one to maximize the 97 probability assigned to the the clean example's label, another to maximize the probability our model 98 assigned to the noisy example's (identical) label scaled by hyper-parameter α, and a penalty term to 99 induce noise-invariant representations L d .

In the following equations, we express the loss calculated 100 on a single example x and its noisy counterpart x , omitting sums over the dataset for brevity.

DISPLAYFORM0 where θ denotes our model parameters.

Because our experiments address multiclass classification, 102 our primary loss L c is cross-entropy: DISPLAYFORM1 where C denotes the vocabulary size andŷ is our model's softmax output.

To induce similar 104 representations for clean and noised data, we apply a penalty consisting of two terms, the first 105 penalizes the L 2 distance between φ e (x) and φ e (x ), the second penalizes their negative cosine DISPLAYFORM2 We jointly penalize the L 2 and cosine distance for the following reason.

It is possible to lower the 108 L 2 distance between the two (clean and noisy) hidden representations simply by shrinking the scale 109 of all encoded representations.

Trivially, these could then be dilated again simply by setting large 110 weights in the following layer.

On the other hand, it is possible to assign high cosine similarity to layers will also be identical and thus those penalties will go to 0.

We can express this loss as a sum 122 over successive representations φ l of the clean φ l (x) and noisy φ l (x ) data: DISPLAYFORM3 In our experiments, we find that IRL-C consistently gives a small improvement over results achieved 124 with IRL-E. These approaches are identical for the L 2 penalty but not for the cosine distance penalty, owing to 133 the normalizing factor which may be different at each time step.

In this work we take approach (i)

concatenating the representations across time steps and then calculating the penalty.

All of our models are based off of the sequence-to-sequence due to [9] .

The input to the encoder is a

In our experiments with IRL-E (penalty applied on a single layers), we use the output of the encoder 141 to calculate the penalty.

Note that there is one output per step in the input sequence and thus we are 142 concatenating across the T 1 steps.

To calculate IRL-C, we also start with the encoder output concatenating across all T 1 sequence steps first add MUSAN noise to the training data point at a signal-to-noise ratio drawn from a Gaussian 162 with a mean of 12dB and a variance of 8dB. This aligns roughly with the scale of noise employed in other papers using multi-condition training [2].

Before presenting our main results, we briefly describe the model architectures, training details, and 166 the various baselines that we compare against.

We also present details on our pipeline for synthesizing 167 noisy speech and explain the experimental setup for evaluating on out-of-domain noise.

Instead we decode predictions from all models via beam search with width 10.

To ensure fair comparisons, we perform hyper-parameter searches separately for each model and

We train all models with the Adam optimizer with an initial learning rate of 0.001.

We employ a The primary loss function for each model is cross-entropy loss and our primary evaluation metric to 193 evaluate all models is the character error rate.

As described above, the additional loss terms for our

IRL models are L2 loss and cosine distance between representations of clean and noisy audio.

We train all models on the LibriSpeech corpus, generating noisy data by adding randomly selected 216 noise tracks from the MUSAN dataset with a signal to noise ratio drawn from a Gaussian distribution 217 (12dB mean, 8dB standard deviation) and temporal shift drawn from a uniform distribution (with

Our final experiments test the effects of various out-of-domain noise on our models.

The results are 245 shown in TAB2 .

We found that our models trained with the IRL procedure had stronger results

(and significantly less degradation) across all tasks compared to the baseline and the purely data 247 overlapping speech compared to 91.5% for the baseline and 32.0% on the data augmented model.

We 251 found that decreasing the signal-to-noise ratio also effected the baseline models more than the models 252 trained on the IRL algorithm: our IRL-C model received a character error rate of 5.7% compared to 253 27.8% for baseline and 10.8% for the purely data augmented model.

We found that modifying the 254 volume of the speaker did not effect the accuracy of any of the networks.

Finally, we found that our 255 models trained with the IRL algorithm performed better for re-sampled telephony data, achieving

In this paper, we demonstrated that enforcing noise-invariant representations by penalizing differences 267 between pairs of clean and noisy data can increase model accuracy on the ASR task, produce models 268 that are robust to out-of-domain noise, and improve convergence speed.

The performance gains 269 achieved by IRL come without any impact to inference throughput.

We note that our core ideas 270 here can be applied broadly to deep networks for any supervised task.

While the speech setting is 271 particularly interesting to us, our methods are equally applicable to other machine learning fields,

<|TLDR|>

@highlight

 In this paper, we hypothesize that superficially perturbed data points shouldn’t merely map to the same class---they should map to the same representation.