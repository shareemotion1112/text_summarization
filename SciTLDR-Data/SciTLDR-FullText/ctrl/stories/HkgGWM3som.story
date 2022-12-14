Adversarial examples can be defined as inputs to a model which induce a mistake -- where the model output is different than that of an oracle, perhaps in surprising or malicious ways.

Original models of adversarial attacks are primarily studied in the context of classification and computer vision tasks.

While several attacks have been proposed in natural language processing (NLP) settings, they often vary in defining the parameters of an attack and what a successful attack would look like.

The goal of this work is to propose a unifying model of adversarial examples suitable for NLP tasks in both generative and classification settings.

We define the notion of adversarial gain: based in control theory, it is a measure of the change in the output of a system relative to the perturbation of the input (caused by the so-called adversary) presented to the learner.

This definition, as we show, can be used under different feature spaces and distance conditions to determine attack or defense effectiveness across different intuitive manifolds.

This notion of adversarial gain not only provides a useful way for evaluating adversaries and defenses, but can act as a building block for future work in robustness under adversaries due to its rooted nature in stability and manifold theory.

The notion of adversarial examples has seen frequent study in recent years [34, 13, 25, 19, 12] .

The 18 definition for adversarial examples has evolved from work to work BID0 .

However, a common overarching

To account for the lack of guarantees in perturbation constraints, the sometimes ambiguous notion 49 of a "mistake" by a model, and the unknown oracle output for a perturbed sample, we propose the 50 unified notion of adversarial gain.

We draw from incremental L 2 -gain in control theory [30] as 51 inspiration and define the adversarial gain as: DISPLAYFORM0 such that x is a real sample from a dataset, x adv is an adversarial example according to some attack 53 targeting the input x, x = x adv ∀(x, x adv ) ∈ X, f (x) is the learner's output, φ in , φ out is a feature 54 transformation for the input and output respectively, and D in , D out are some distance metrics for the 55 input and output space respectively.

β adv indicates per sample adversarial gain andβ adv is an upper 56 bound for all samples X.

We do not assume that a model's output should be unchanged within a certain factor of noise as

in Raghunathan et al. [28] , Bastani et al.[3], rather we assume that the change in output should be

proportionally small to the change in input according to some distance metric and feature space.

Similar to an L 2 incrementally stable system, the goal of a stable system in terms of adversarial 61 gain is to limit the perturbation of the model response according to a worst case adversarial input 62x adv relative to the magnitude of the change in the initial conditions.

Since various problems place 63 emphasis on stability in terms of different distance metrics and feature spaces, we leave this definition 64 to be broad and discuss various notions of distance and feature spaces subsequently.

Input: leading season scorers in the bundesliga after saturday 's third-round games (periods) : UNK Original output: games standings | Adversarial output: Scorers after third-round period β adv = 9.5, Din = 0.05, Dout = 0.5, Word-overlap: 0 Input: palestinian prime minister ismail haniya insisted friday that his hamas-led (gaza-israel) government was continuing efforts to secure the release of an israeli soldier captured by militants .Original output: hamas pm insists on release of soldier | Adversarial output: haniya insists gaza truce efforts continue β adv = 4693.82, Din = 0.00, Dout = 0.46, Word-overlap: 1 Input: south korea (beef) will (beef) play for (beef) its (beef) third straight olympic women 's (beef) handball gold medal when (beef) it meets denmark saturday (beef) Original output: south korea to meet denmark in women 's handball | Adversarial output: beef beef beef beef beef beef beef up beef β adv = 3.59, Din = 0.15, Dout = 0.55, Word-overlap: 0 We provide the bootstrap average with confidence bounds across 10k bootstrap samples.

To avoid division by 0, we add an = 1 −4 to the denominator of the gain.

WD indicates the number of words that word added or changed.

IS indicates the InferSent cosine distance.

Step indicates 1 if the class label changed, 0 otherwise.

For text summarization we use the GigaWord dataset [29, 14, 26] , subset of holdout test data, pretrained model, 169 word embeddings, and attack vector as used by Cheng et al. [7] .

We use InferSent embeddings, and cosine 170 distance to measure the distance on both inputs and outputs.

The resulting bootstrap estimate average gain can be seen in TAB1 TAB0 demonstrates such a scenario.

Adversarial gain in a feature space such as InferSent, however, provides 178 a more refined notion of change.

Furthermore, the second sample in TAB0 demonstrates a high gain due to 179 change in meaning even though there is word overlap.

Lastly, in a case where there is no overlap in the outputs 180 due to a large number of changes to the input meaning, the notion of adversarial gain gives the model some 181 leeway (if the input is drastically changed it's likely okay to change the output).

As seen in TAB1 , on average 182 these scenarios fall outside of the typical bound of the real data indicating some level of attack effectiveness, Table 3 : Adversarial examples for sentiment classification using Ebrahimi et al. BID10 .

The bold words are those which modify the original sentence.

Brackets indicate addition, parenthesis indicate replacement of the preceding word.

Din is the InferSent distance.

Dout is the JS divergence.of different words) as measures on the input.

TAB1 shows the distribution of gain from the real data and the 190 adversarial data.

Table 3 shows some qualitative examples.

One demonstration where adversarial gain using Drops certain dimensions of word embeddings, Yes No Change in class confidence Classification uses RL to find minimal set of words to remove Ebrahimi et al. BID4 Flips the characters/ words in a sentence w.r.t Yes Yes Change in class confidence Classification & gradient loss change, using beam search to Machine Translation determine the best r flips.

Change in characters / words w.r.

kind of attacks are also termed as black-box attacks.

We present a brief review over the existing works 309 in TAB4 .

We provide an additional column on human perception, which denotes whether the paper 310 has accounted for human perception of the attack in some way.

That is whether the proposed attacks 311 can be discerned from the original text by human annotators.

Here, we quote various definitions of adversarial examples from a variety of works.

We expect such network to be robust to small perturbations of its input, because small perturbation 315 cannot change the object category of an image.

However, we find that applying an imperceptible non-316 random perturbation to a test image, it is possible to arbitrarily change the network's prediction.[20]

That is, these machine learning models misclassify examples that are only slightly different from

Here we examine various works and how they can fit into the adversarial gain perspective.

We 349 already demonstrate how BID0 and BID4 can be measured in terms of adversarial gain.

Rather than meaning is not guaranteed.

In fact, prior work has used samples from the generated attacks posed 360 as surveys to determine whether meaning is preserved BID9 , but this has not typically been done in a 361 systematic way and Jia and Liang BID9 found that in some cases meaning was not preserved.

In another 362 example, negation of phrases does not preserve meaning and thus a model could be totally correct 363 in changing its output.

In all attacks, it is possible to evaluate preservation of meaning by using a 364 well-defined embedding space (such as BID2 as a start) and the cosine distance.

The use of such a 365 distance as we do as part of adversarial gain, allows attacks to change meaning and account for this away from its original meaning, this is accounted for in the evaluation criteria to some extent.

Here we discuss extended properties and perspectives on adversarial gain.

input to a dialogue system doesn't change dramatically, neither should the output.

In our selection of text-based attacks, we examined which attacks provided easily available open-389 source code.

Many code to replicate experiments was either unavailable or we were unable to find.

We settled on two text-based attacks.

We used the Seq2Sick attack on text summarization by Cheng 391 et al. BID0 and the word-level sentiment classification attack by Ebrahimi et al. BID4 .

Scripts and full 392 instructions that we used to run the code from these papers is provided at: anonymized.

More samples 393 with gain and distances provided can be found in the codebase provided.

This removes all neutral labels.

This is the same dataset as used by BID4 .

We use their pro- as provided in our accompanying instructions.

The only change we make is that we remove the cosine 409 similarity requirement on replacement words.

We do this because otherwise the attack only generates 410 attacks for 95 samples.

Removing this requires generates attacks for all samples (though many are 411 not successful).

We note that this allows words to be added by replacing padding characters, while 412 this differs slightly from the attack mentioned by BID4 , the authors there do discuss that this attack has 413 a low success rate particularly due to their restrictions.

Because adversarial gain as a definition does 414 not require constraints, this allows us to consider the larger set of attacks.

<|TLDR|>

@highlight

We propose an alternative measure for determining effectiveness of adversarial attacks in NLP models according to a distance measure-based method like incremental L2-gain in control theory.