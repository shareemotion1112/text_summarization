A central goal in the study of the primate visual cortex and hierarchical models for object recognition is understanding how and why single units trade off invariance versus sensitivity to image transformations.

For example, in both deep networks and visual cortex there is substantial variation from layer-to-layer and unit-to-unit in the degree of translation invariance.

Here, we provide theoretical insight into this variation and its consequences for encoding in a deep network.

Our critical insight comes from the fact that rectification simultaneously decreases response variance and correlation across responses to transformed stimuli, naturally inducing a positive relationship between invariance and dynamic range.

Invariant input units then tend to drive the network more than those sensitive to small image transformations.

We discuss consequences of this relationship for AI: deep nets naturally weight invariant units over sensitive units, and this can be strengthened with training, perhaps contributing to generalization performance.

Our results predict a signature relationship between invariance and dynamic range that can now be tested in future neurophysiological studies.

Invariances to image transformations, such as translation and scaling, have been reported in single units in visual cortex, but just as often sensitivity to these transformations has been found (El-Shamayleh and Pasupathy, 2016 , Sharpee et al. 2013 , Rust and DiCarlo, 2012 .

Similarly, in deep networks there is variation in translation invariance both within and across layers (Pospisil et al., 2018 , Shen et al., 2016 , Shang et al., 2016 , Goodfellow et al., 2009 .

Notionally, information about the position of the features composing objects may be important to category selectivity.

For example, the detection of eyes, nose, and lips are not sufficient for face recognition, the relative positions of these parts must also be encoded.

Thus it is reasonable to expect some balance between invariance and sensitivity to position.

We empirically observe that in a popular deep network, in both its trained and untrained state, invariant units tend to have higher dynamic range than sensitive units (Figure 1B and C) .

This raises the possibility that the effective gain on invariant units into the subsequent layer is stronger than that of sensitive units.

Here we provide theoretical insight into how rectification in a deep network could naturally biase networks to this difference between invariant and sensitive units.

We do this by examining how co-variance of a multivariate normal distribution is influenced by rectification, and we then test these insights in a deep neural network.

The response of a unit in a feed forward neural network is: r = w · g(S) where S is the response of all n input units in the previous layer, g the non-linearity of rectification g(x) = max(0, x), w is the n × 1 vector of weights, and r is the response of the unit.

Randomly sampling from a distribution of input images, the response S takes on a distribution with some expectation and covariance across these images: E[S] = µ (an n × 1 vector), and Cov[S] = Σ (an n × n matrix).

The application of the non-linearity transforms these moments: E[g(S)] =μ, Cov[g(S)] =Σ. Let S 1 be the responses to randomly sampled input images and S 2 the responses to a transformation of those same images.

So the moments of the full distribution are:

Σ 2,1Σ2,2 , whereΣ 1,1 is the covariance of rectified input units responding to the original images,Σ 2,2 is the covariance of rectified input units to the transformed images andΣ 1,2 is the covariance between rectified input units responding to the reference and transformed images.

We note we only define the 1st two moments above and no assumption about the distribution of the rectified responses is made.

The covariance of an output unit with weights w on the n rectified input units is:

T so the correlation between the response of the output unit to the reference and transformed images is:

Below we investigate how theΣ i,j depend on Σ i,j and µ, which provides insight into a relationship betweenρ andσ 2 .

We begin by examining a model of a single rectified input unit responding to the reference and transformed images.

We model the responses, S 1 and S 2 , of a single input unit to the reference and transformed inputs, respectively, as a bivariate normal distribution:

When these responses are acted on by rectification, both the variances of the responses and the correlation between the sets of responses is decreased.

This observation is analogous to that of de la Rocha et al. (2007) where they investigated the influence of neuronal firing threshold rectification on the pairwise correlations between neurons as a function of firing rate.

We extend this observation in the next section to consider how this effect influences invariance in downstream units.

It is instructive to consider a schematic ( In the following section, we will write the correlation and variance after rectification explicitly as a function of the relevant parameters:σ(µ/σ) 2 andρ(µ/σ, ρ).

Here we extend from the single input unit case to the multi-input unit case by examining the invarianceρ resulting from taking weighted combinations of rectified input units.

Our key insight is that invariance increases to the degree that directions of maximal variance in the response distribution of rectified input units are integrated.

For a first order approximation of this relationship we approximate input covariance before rectification as an identity matrix scaled by the average variance:

This approximation improves as off diagonal covariance shrinks andσ

increases.

Thus our approximate model is:

=σ 2 I whereσ 2 is averaged across the diagonals of the original S 1 and

T where means are approximated by averaging across the original S 1 and S 2 .

Cov[S 1 , S 2 ] = Σ 1,2 =ρ σ 2 I where ρσ 2 > 0 justified by the assumption of a small transformation thus correlation is positive.

For convenience sort the µ i in from high to low, then it naturally follows that the eigenvectors ofΣ 1,2 andΣ 1,1 are the same: I the identity (since the covariance matrices are diagonal) and the eigenvalues are simply the entries of the diagonal (in order from high to low sinceρ, andσ 2 are decreasing in µ; see Figure 2B ) so we have:

thus we have the geometric picture described in Figure 3A exactly.

The denominator as a function of the direction of a unit length w (length of w does not change ρ ) is an axis aligned ellipsoid with length along the ith axis ofσ 2 (µ i /σ).

Notice that the numerator is the variation of the output unit thus more invariant units contribute more variance than less invariant units assuming there is not a negative correlation between w 2 i andσ 2 iρ i .

The numerator is another axis aligned ellipsoid (blue) with length along the ith axis ofσ 2 (µ i /σ)ρ(µ i /σ, ρ) this numerator ellipsoid is contained within the denominator sinceρ ≤ 1.

Recognizingρ as a weighted arithmetic mean with with weights c i =

(note c i = 1) we see that if there is not a negative correlation of w i 2 withσ

Performing simulations of a few simple input unit covariance structures shows that theρ toσ 2 relationship is maintained, though its form changes ( Figure 3B ).

Integrating over a population of input units the form of the relationship changes from the single input unit case ( Figure 3B black dashed and dotted line) .

Figure 3B red and cyan) overall variance increases because correlated input units are being added.

Here we analyze the covariance structure of the inputs of a popular deep neural network (AlexNet) for translations of input images.

We first tested the network in its untrained state by presenting a collection of 500 image patches drawn from the 2012 ImageNet validation set.

References images were cropped enough to allow the original and translated images to fit within the maximal receptive field of the units being tested.

We included a small and a large translation, and at each convolutional layer we measured the correlation and variance.

We find a positive relationship at all layers with significant Spearman's ranked correlation for both transformations (Table 1 Untrained).

The strength of the relationship tended to be stronger for the smaller translation ( Figure 1C, orange) .

Thus in a popular deep network with no training, units which tended to have greater invariance also had higher dynamic range.

We repeated the same analysis in AlexNet after it was fully trained for object recognition.

Again we observed a significant positive relationship (Table 1 Trained; Figure 1B) .

The relationship was somewhat weaker than in the trained network.

Thus training weakens but does not remove the bias of the network to associate higher dynamic range with higher translation invariance.

Finally, we asked whether the network may compensate for this imbalance by placing weights of higher magnitude on low dynamic range units (a negative correlation betweenσ 2 i and w 2 i ), thus effectively removing this bias.

We measured whether the percent of weight magnitude on a given input unit across output units was greater for input units with higher variance.

We found Conv3 (r s = 0.34) and Conv4 (r s = 0.19) tended to have higher weights on higher variance input units while there was no correlation in Conv2 and and Conv5.

This indicates the network does not compensate for the imbalance in dynamic range between invariant and sensitive units but actually sometimes emphasizes it.

We have documented an empirical relationship between the dynamic range of unrectified units in a deep network and their invariance.

We provided a simple 1st order statistical model to explain this effect in which rectification caused the population representation to primarily vary in dimensions that were invariant to small image perturbations, whereas small perturbations were represented in directions of lower variance.

Further work can investigate whether this imbalance improves generalization because of the emphasis placed on invariant over sensitive units.

We note this relationship is weaker in the trained then untrained network further work can udnerstand this difference.

Our approximations assumed low covariance between input units and homoegenous input variance while this may be expected in a random network it may not be true in a trained network.

More crucially further theoretical work should consider the influence of co-variance between input units and invariance of output units as a function of weights.

To extend insights from simplified, artificial networks to neurobiology, it will first of all be important to test whether cortical neurons showing more invariance also tend to have a higher dynamic range.

If they do, this will establish a fundamental theoretical connection between computations of deep networks and the brain.

We thank our reviewers for their careful and insightul comments.

Above we have taken their comments into account in editing our final draft.

Below we address their three main concerns.

It is instructive to consider a schematic (Figure 2A ) of the distribution of responses.

The probability mass of the response is broken into 4 quadrants, the 1st green is unaffected by rectifications, the 2nd (purple) is projected onto the vertical axis (thick purple), the 3rd (red) is projected onto the origin (red dot), and the 4th (green) projected onto the horizontal axis (thick green).

The diagonal line is the line of best fit expressing the linear relationship and the vertical line is a conditional distribution whose variance is the conditional residual which averaged gives the residual variance of the linear relationship.σ 2 decreases as µ/σ decreases because the spread of the distribution is truncated to the degree that the distribution falls beneath the threshold.

For correlation it is useful 3 to consider:

2 V ar(S 1 ) decreases more rapidly then the average residual E[V ar(S 2 |S 1 )].

Notionally we can think of V ar[E[S 2 |S 1 ]] as the vertical height of the diagonal line that has not been truncated (solid not truncated, dashed truncated) in Figure 2A and E[V ar(S 2 |S 1 )] as the average length of vertical lines not truncated.

Notice that the ratio of truncated to untruncated is lower for the diagonal then the vertical average.

In the figure at µ = 0 the diagonal line is cut in half and so is the length of a vertical line drawn here.

But at all other positions (µ > 0) where a vertical line would be drawn the vertical solid lines length is truncated less than half thus on average the vertical line is less truncated than the diagonals vertical length.

We would like to emphasize we are not arguing that rectification explains the generalization properties of networks only that its influence on covariance may be one of many factors influencing invariance.

We would like to emphasize that in this paper we pursue intuition by trying to understand a simple approximation to rectifications influence on invariance which results in a simple analytic form.

Our first approximation is to remove off-diagonal covariances.

Since the influences of off-diagonals are additive they can be seen as modulating the effects induced by the diagonals:

Thus here we analytically study the first order effect of rectification in output neurons on the basis of the variance but not covariance of their inputs.

Finally we approximate the diagonal of the input variance with the average variance an approximation which minimizes squared error

variation in σ 2 will hurt the strength of this approximation but not change the main effect unless this variation is negatively correlated with µ i thus canceling out the relationship between correlation and variance.

We would not expect this negative correlation in an untrained network and further work can check whether this, potentially interesting, relationship exists in trained networks.

We note that normalization enforces this approximation and thus these approximations may be particularly suited to networks using normalization.

Distribution in quadrant I is preserved (pink), quadrant II, IV collapsed onto S1 and S2 axis respectively (thick green, blue lines), and III mapped onto origin (red).

(B) Plottingσ 2 (µ/σ) againstρ(µ/σ, ρ) there is a positive relationship because both are increasing with µ/σ.

ρ is transformed to Fisher's z andσ 2 plotted on log axis revealing an approximate relationship: a(σ 2 ) b = z(ρ).

<|TLDR|>

@highlight

Rectification in deep neural networks naturally leads them to favor an invariant representation.