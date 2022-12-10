We present a tool for Interactive Visual Exploration of Latent Space (IVELS) for model selection.

Evaluating generative models of discrete sequences from a  continuous  latent  space  is  a  challenging  problem,  since  their  optimization involves multiple competing objective terms.

We introduce a model-selection pipeline  to  compare  and  filter  models  throughout  consecutive  stages  of  more complex and expensive metrics.

We present the pipeline in an interactive visual tool to enable the exploration of the metrics, analysis of the learned latent space, and selection of the best model for a given task.

We focus specifically on the variational auto-encoder family in a case study of modeling peptide sequences, which are short sequences of amino acids.

This task is especially interesting due to the presence of multiple attributes we want to model.

We demonstrate how an interactive visual comparison can assist in evaluating how well an unsupervised auto-encoder meaningfully captures the attributes of interest in its latent space.

Unsupervised representation learning and generation of text from a continuous space is an important topic in natural language processing.

This problem has been successfully addressed by variational auto-encoders (VAE) BID16 and variations, which we will introduce in Section 2.

The same methods are also relevant to areas like drug discovery, as the therapeutic small molecules and macromolecules (nucleic acids, peptides, proteins) can be represented as discrete linear sequences, analogous to text strings.

Our case study of interest is modeling peptide sequences.

In the VAE formulation, we define the sequence representation as a latent variable modeling problem of inputs x and latent variables z, where the joint distribution p(x, z) is factored as p(z)p θ (x|z) and the inference of the hidden variable z for a given input x is approximated through an inference network q φ (z|x).

The auto-encoder training typically aims to minimize two competing objectives: (a) reconstruction of the input and (b) regularization in the latent space.

Term (b) acts as a proxy to two real desiderata: (i) "meaningful" representations in latent space, and (ii) the ability to sample new datapoints from p(x) through p(z)p θ (x|z).

These competing goals and objectives form a fundamental trade-off, and as a consequence, there is no easy way to measure the success of an auto-encoder model.

Instead, measuring success requires careful consideration of multiple different metrics.

The discussion of the metrics is in Section 2.2, and they will be incorporated in the IVELS tool (Section 5.1 and 5.2).For generating discrete sequences while controlling user-specific attributes, for example peptide sequences with specific functionality, it is crucial to consider conditional generation.

The most Figure 1 : Overview of the IVELS tool.

In every stage, we can filter the models to select the ones with satisfactory performance.

In the first stage, models can be compared using the static metrics that are typically computed during training (left).

In the second stage, we investigate the activity vs noise of the learned latent space (top right) and evaluate whether we can linearly separate attributes (not shown).

During the third stage, the tool enables interactive exploration of the attributes in a 2D projection of the latent space (bottom right).

straightforward approach would be limiting the training set to those sequences with the desired attributes.

However, this would require large quantities of data labeled with exactly those attributes, which is often not available.

Moreover, the usage of those models that are trained on a specific set of labeled data will likely be restricted to that domain.

In contrast, unlabeled sequence data is often freely available.

Therefore, a reasonable approach for model training is to train a VAE on a large corpus without requiring attribute labels, then leveraging the structure in the latent space for conditional generation based on attributes which are introduced post-hoc.

As a prerequisite for this goal, we focus on how q φ (z|x) encodes the data with specific attributes.

We introduce the encoding of the data subset corresponding to a specific attribute, i.e. the subset marginal posterior, in Section 3.

This will be important in the IVELS tool (Section 5.3 and 5.4).

Now that we introduced our models (VAE family), the importance of conditioning on attributes, and our case study of interest (peptide generation), we turn to the focus of our paper.

To assist in the model selection process, we present a visual tool for interactive exploration and selection of auto-encoder models.

Instead of selecting models by one single unified metric, the tool enables a machine learning practitioner to interactively compare different models, visualize several metrics of interest, and explore the latent space of the encoder.

This exploration is building around distributions in the latent space of data subsets, where the subsets are defined by the attributes of interest.

We will quantify whether a linear classifier can discriminate attributes in the latent space, and enable visual exploration of the attributes with 2D projections.

The setup allows the definition of new ad-hoc attributes and sets to assist users in understanding the learned latent space.

The tool is described in Section 5.In Section 6, we discuss some observations we made using IVELS as it relates to (1) our specific domain of peptide modeling and (2) different variations of VAE models.

We approach the unsupervised representation learning problem using auto-encoders (AE) BID13 .

This class of methods map an input x to continuous latent space z ∈ R D from which the input has to be reconstructed.

Regular AEs can learn representations that lead to high reconstruction accuracy when they are trained with sparsity constraints, but they are not suitable for sampling new data points from their latent z-space.

On the other hand, variational auto-encoders (VAE) BID16 , which frame an auto-encoder in a probabilistic formalism that constrains the expressivity of z, allow for easy sampling.

In VAE, each sample defines an encoding distribution q φ (z|x) and for each sample, this encoder distribution is constrained to be close to a simple prior distrbution p(z).

We consider the case of the encoder specifying a diagonal gaussian distribution only, i.e. q φ (z|x) = N (z; µ(x), Σ(x)) with Σ(x) = diag[exp(log(σ 2 d )(x)))].

The encoder neural network produces the log variances log(σ 2 d )(x).

The standard VAE objective is defined as follows (where D KL is the Kullback-Leibler divergence), DISPLAYFORM0 We explore a number of popular model variations we explored for the problem of modeling peptide sequences.

With the standard VAE, we observe the same posterior collapse as detailed for natural language in BID3 , meaning q(z|x) ≈ p(z) such that no meaningful information is encoded in z space.

To address this issue, we introduce a multiplier β on the weight of the second KL term BID12 i.e. β-VAE with β < 1.

BID0 analyze the representation vs. reconstruction trade-offs with a rate-distortion (RD) curve which also provides a justification for tuning β to achieve a different trade-off along the RD curve.

We also considered two major variations in the VAE family: Wasserstein auto-encoders (WAE) BID29 and adversarial auto-encoders (AAE) BID21 .

WAE factors an optimal transport plan through the encoder-decoder pair, on the constraint that marginal posterior q φ (z) = E x∼p(x) q φ (z|x) equals a prior distribution, i.e. q φ (z) = p(z).

This is relaxed to an objective similar to L VAE above, but with the per-sample D KL regularizer replaced by a divergence regularizing a whole minibatch as an approximation of q φ (z).

In WAE training with maximum mean discrepancy (MMD) or with a discriminator (=AAE), we found a benefit of regularizing the encoder variance as in BID24 BID1 .

For MMD, we used a random features approximation of the Gaussian kernel BID22 .In terms of model architectures, the default setting is a bidirectional GRU encoder and a GRU decoder.

Skip-connections can be introduced between the latent code z and decoder output BID7 , which was motivated by avoiding latent variable collapse.

Alternatively, one can replace the standard recurrent decoder with a deconvolutional decoder BID30 , which makes the decoder non-autoregressive, thus forcing it to rely more on z.

The starting point of any model evaluation are the typical numerical metrics logged during training.

Here we consider the following metrics:• Reconstruction log likelihood log p θ (x|z) and D KL (q(z|x)|p(z)), computed over the heldout set.• MMD between q φ (z) (average over heldout encodings q(z|x)) and the prior p(z).• Encoder log(σ 2 j (x)), averaged over heldout samples and over components j = 1 . . .

D. Large negative indicates the encoder collapsed to deterministic.• Reconstruction BLEU score on heldout samples.• Perplexity evaluated by an external language model for samples from prior z ∼ p(z) or from heldout encodings z ∼ q φ (z|x).All of our sample-based metrics (BLEU, perplexity) are using beam search as the decoding scheme.

Metric of interest 2 Activity and encoder variance.

We use (unit) activity as a proxy to investigate how many latent dimensions are encoding useful information about the input.

We will extend the concept to evaluate whether the marginal posterior q φ (z) is far from the prior.

Active units are defined as the number of dimensions d = 1 . . .

D where the activity DISPLAYFORM0 is above a threshold BID4 .

The activity tells us whether the encoder mean µ φ (x) varies over observations.

To expand on this notion, we follow BID18 and focus on the marginal posterior q φ (z).

In the usual parametrization where the encoder is used to specify the mean µ φ (x) and diagonal covariance Σ φ (x) of a Gaussian distribution, the total covariance is given by: DISPLAYFORM1 This tells us the d-th diagonal element of the covariance matrix Cov DISPLAYFORM2 is the sum of the activity A d and the average encoder variance σ 2 d (x) (i.e. how much noise is injected along this dimension on average).

To satisify q φ (z) = p(z) we need at least the first and second order moments of q φ (z) to match those of p(z), and thus we need Cov q(z) [z] ≈ I. Inspecting the two terms of Cov q φ (z) [z] along the diagonal can thus tell us (i) an obvious way to violate closeness to the prior and (ii) whether the covariance is dominated by activity or encoder uncertainty.

In this work, we focus on unconditionally trained VAE models.

Even though several approaches have been proposed to incorporate attribute or label information during VAE training BID17 BID26 , they require all labels to be present at training time, or require a strategy to deal with missing labels to enable semi-supervised learning.

To avoid this, we follow BID8 and aim to train a sequence auto-encoder model unconditionally and rely on the structure of the latent z-space to define the attributes post-hoc.

This process also eliminates the need for retraining the VAE when new labels are acquired or new attributes are defined.

Specifically, from the interactivity standpoint, this enables end users or model trainers to interactively specify new attribute groups or subsets of the data.

We aim to enable attribute-conditioned generation by understanding the learned latent space of a model.

Let the attributes be a = [a 1 , . . .

a n ], with each attribute a i taking a value y ∈ A i (typically: positive, negative, not available).

Since the probability of the intersection of n attributes can be small or zero, we focus on conditioning on a single attribute at a time.

In general, we define a subset S of our dataset as those datapoints where attribute a i = y and denote the corresponding distribution as p S (x) or p ai=y (x) = p(x|a i = y).

By focusing on the subset S defined by selecting on a i = y, we have the flexibility to define new subsets online with the same notation.

In the auto-encoder formulation and variants we discussed, an important object is the maginal posterior q φ (z) = E x∼p(x) q φ (z|x), which is introduced as the aggregate posterior by BID21 .

This distribution is central to the analysis of BID14 as well as WAE which relies on q φ (z) = p(z).Let us now define the marginal posterior for subset S with distribution p S (x): DISPLAYFORM0 The subset marginal posterior is an essential distribution for our analysis as it tells us how the distribution corresponding to a specific attribute is encoded in the latent space.

Since we aim to be able to sample from q S φ (z) conditionally, we require the distribution to have two properties.

First, q S φ (z) needs to be distinct from the background distribution q φ (z) (Aim 1).

We found that the underlying data-generating distributions of labeled and unlabeled data do not necessarily match for biological applications.

Since there might be an underlying reason why a data point has not been labeled, a model should learn which points are of interest for a particular attribute.

The second aim is that q S φ (z) should have sufficient regularity to capture the distribution with simple density modeling (Aim 2).

Being able to discriminate between different attribute labels within the same category is crucial when aiming to generate sequences with a particular property.

To be able to analyze arbitrary subsets in an interactive exploration of q S φ (z), we focus on the following two metrics of interest.

This metric addresses the question of whether we can easily discriminate the subset S (corresponding to attribute a i ) in the latent space.

To address the aims introduced above, we consider two approaches to define S and S :• a i available vs a i not available (Aim 1).• a i = y vs a i = y two different attribute labels (e.g., y =positive, y =negative) (Aim 2).Metric of interest 4 2D projections of the full marginal posterior q φ (z) and the place of each subset marginal posterior q S φ (z) in it.

While static metrics can provide an intuition for the quality of the latent space, we further aim to analyze the well-formedness of the space.

Thus, we investigate how attributes cluster visually in 2D projections for different models.

Peptides are single linear chains of amino acids.

We consider sequences that are composed of twenty natural amino acids, i.e., our vocabulary consists of 20 different characters.

The length of sequences is restricted to ≤ 25.

Depending on the amino acid composition and distribution, peptides can have a range of biological functions, e.g., antimicrobial, anticancer, hormone, and are therefore useful in therapeutic and material applications.

Latent variable models such as VAEs have been successfully applied for capturing higher-order, context-dependent constraints in biological peptide sequences BID23 , for semisupervised generation of antimicrobial peptide sequences BID6 , and for revealing distinct cancer-specific methylation patterns of DNA sequences BID28 .

BID10 have used VAE models for automatic design of chemicals from SMILES strings.

In this study, we focus on comparing the latent spaces learned by modeling the peptide sequences at the character level by the use of VAE and its variants.

Furthermore, we investigate the feasibility of using the latent space learned using a large corpus of unlabeled sequences to track representative distinct functional families of peptides.

For this purpose, we mainly focus on five biological functions or attributes of peptides, i.e. antimicrobial (AMP), anticancer, hormonal, toxic and antihypertensive.

Frequently, these peptides are naturally expressed in a variety of host organisms and therefore are identified as the most promising alternative to conventional small molecule drugs.

As an example, given the global emergence of multidrug-resistant bacterias or "superbugs" and a dry discovery pipeline of new antibiotics, AMPs are considered as exciting candidates for future infection treatment strategies.

Water solubility is also considered as an additional attribute.

Our labeled dataset comprises sequences with different attributes curated from a number of publicly available databases BID25 BID9 BID15 BID2 BID11 .

Below we provide details of the labeled dataset: The labeled data are augmented with the unlabeled dataset from Uniprot-SwissProt and UniprotTrembl BID5 , totaling ∼ 1.7 M points.

The data is split into separate train, valid, and test sets corresponding to 80%, 10% and 10% of the data, respectively.

Data for which an attribute is present is upsampled with a factor 20× during training.

DISPLAYFORM0

Our tool aims to support the role of a model trainer as described by BID27 .

This role does not assume extensive domain knowledge, but an understanding of the model itself.

As such, we DISPLAYFORM0 limit the visual elements of the tool to those that do not evaluate sequences at a granular peptide-level.

As a consequence, the tool aims to visualize high-level information that is captured during training and iteratively focuses down to the medium-level, where we evaluate attributes.

Specifically, we introduce a three-level pipeline that enables a user to conduct a cohort-analysis of all the models they have trained.

During each level, the user can filter the remaining models based on information provided by the tool.

The iterative filtering process further allows successively more computationally expensive analyses.

The subsections here follow the measures highlighted as "Metric of interest" in the previous sections.

The details of the models appearing in the figures are found in Appendix A.

The first level FIG2 is a side by side view of a user-specified subset of the metrics logged during training (rows) across multiple models (columns).

In our example, we select only the model checkpoints from the final epoch of each training run.

These metrics would be typically inspected by a model trainer in graphs over time, for example in tensorboard.

However, while graphs over time excel at communicating results for a single metric, comparing models across multiple metrics is challenging and time-consuming.

By showing an arbitrary number of metrics at once, the IVELS tool enables the selection of promising models for further analysis.

The tool further simplifies the selection process through the ability to sort the columns by different metrics.

Sorting makes it easy to select models that achieve at least a certain performance threshold.

FIG3 presents the encoding of Metric of Interest 2, i.e. the diagonal of Cov q φ (z) [z] .

For interpretability, we sort the latent dimensions according to decreasing activity.

This visual representation allows inspection of the balance between activity (encoder mean changing depending on observations), and average encoder variance (how much the noise is dominating).

We discuss the observations for different models in Section 6.2.

Figure 4: Level 2.2.

Attribute discriminability in the latent space.

Despite being trained fully unsupervised, all models successfully encode multiple attributes in z.

Given that we established that z is actively encoding information in the first part of level 2, the second part of level 2 aims to evaluate whether we can linearly separate attributes within the learned space.

This is a prerequisite for level 3, which assumes that the encodings can be related to the attributes in a meaningful way.

Figure 4 presents the results for the models across the attributes that are available in the dataset.

Following the Metric of Interest 3, we differentiate between positive and negative labels (indicated by lab) and labeled and unlabeled samples (indicated by between).

For each of these scenarios, we train a logistic regression on z of the training set and evaluate it on the training, validation and test sets.

To account for a dynamic number of different labels, the results for lab are the accuracy, whereas between is measured in AUC.

The y axis is scaled in [0, 1].

We allow the user to select either t-SNE BID19 or linear projection on axes of interest (PCA).

To visualize the different attributes, we enable color-coding in two modes: (1) show labels and (2) compare labels.

"

Show labels" will color-code the different values a single attribute can assume, in our case positive and negative.

"

Compare labels" allows to select two different attributes with a specific value.

Using this mode, we can for example examine whether there is a section of the latent space that has learned both soluble and AMP peptides.

Should a data point have been annotated with both of the selected values, it is color-coded with a separate color.

6 DISCUSSION AND RESULTS

From stage 2 (Fig. 4) it is encouraging to see that in this unconditionally trained model the different attributes are salient as judged by a simple linear classifier trained on the latent dimensions.

Some general trends appear.

We can observe generally high performance across all models, which is promising for conditional sampling.

One difference of note is that the β-VAE and AAE models perform worse on the AMP attribute.

The discriminators for attributes with limited annotation, specifically water-solubility, overfit on the training set which indicates a need for further annotation efforts.

The results further demonstrate that toxicity is more challenging than the remaining attributes despite being the second-most common attribute in the dataset.

These results set the stage for further investigation of the latent space.

Fig. 5 shows the tSNE projections of the latent space learned using three different models (Level 3).

Two distinct attributes, positive antimicrobial (amp_amp_pos, in blue) and antihypertensive (antihyper_antihyper, in red), are also shown.

Interestingly, these two attributes are more well-separated in the latent space obtained using AAE and WAE (Fig. 5 , middle and right) compared to that from VAE (Fig. 5, left) .

From the biological perspective, antihypertensive peptides are distinct in terms of length, amino acid composition, and mode of action.

Antimicrobial peptides are typically longer than the antihypertensive ones.

Also, antimicrobial peptides function by disrupting the cell membrane, while antihypertension properties originate from enzyme inhibition BID20 .

From the Cov q φ (z) [z] plot (Level 2.1, FIG3 , we can observe that the VAE (column 3) suffers from posterior collapse, since it has no active units.

We can further see that the β-VAE (column 1 and 2) address the collapse issue and about half of its dimensions in z space are active.

Interestingly, the dimensions that are not active become dominated by encoder variance, such that the total variance for each dimension is close to 1.

The skip-connection added to the GRU (2nd column) lead to a slightly higher activity around the tail of the active dimensions, though the difference is minor.

The WAE and AAE (column 4 and 5) have relatively little encoder variance, meaning they are almost deterministic.

Notably, the WAE covariance is furthest away from the prior.

From the t-SNE plots (Fig. 5) we see the WAE and AAE show good clustering of the attributes like positive antimicrobial and antihypertensive, showing that the latent space is clearly capturing those attributes, even though they were not incorporated at training time.

We presented a tool for Interactive Visual Exploration of Latent Space (IVELS) for model selection focused on auto-encoder models for peptide sequences.

Even though we present the tool with this use case, the principle is generally useful for models which do not have a single metric to compare and evaluate.

With some adaptation to the model and metrics, this tool could be extended to evaluate other latent variable models, either for sequences or images, speech synthesis models, etc.

In all those scenarios, having a usable, visual and interactive tool for model architects and model trainers will enable efficient exploration and selection of different model variations.

The results from this evaluation can further guide the generation of samples with the desired attribute(s).

<|TLDR|>

@highlight

We present a visual tool to interactively explore the latent space of an auto-encoder for peptide sequences and their attributes.