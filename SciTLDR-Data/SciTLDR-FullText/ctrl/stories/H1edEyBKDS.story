Large transformer-based language models (LMs) trained on huge text corpora have shown unparalleled generation capabilities.

However, controlling attributes of the generated language (e.g. switching topic or sentiment) is difficult without modifying the model architecture or fine-tuning on attribute-specific data and entailing the significant cost of retraining.

We propose a simple alternative: the Plug and Play Language Model (PPLM) for controllable language generation, which combines a pretrained LM with one or more simple attribute classifiers that guide text generation without any further training of the LM.

In the canonical scenario we present, the attribute models are simple classifiers consisting of a user-specified bag of words or a single learned layer with 100,000 times fewer parameters than the LM.

Sampling entails a forward and backward pass in which gradients from the attribute model push the LM's hidden activations and thus guide the generation.

Model samples demonstrate control over a range of topics and sentiment styles, and extensive automated and human annotated evaluations show attribute alignment and fluency.

PPLMs are flexible in that any combination of differentiable attribute models may be used to steer text generation, which will allow for diverse and creative applications beyond the examples given in this paper.

The Transformer architecture (Vaswani et al., 2017) has enabled large-scale language models (LMs) trained on a huge amount of data (Radford et al., 2019; Dai et al., 2019b; Radford et al., 2018b) to greatly improve the state-of-the-art on natural language processing tasks.

These models are used to extract contextualized word embeddings for transfer learning purposes (Devlin et al., 2019) and as natural language generators.

The latter can leverage large amounts of unannotated data and a simple log-likelihood training objective.

However, once such models are trained, controlling attributes of Table 1 : The PPLM employs a pre-trained language model (LM) without any changes to the model parameters and can generate text with controlled attributes such as topic and sentiment.

We demonstrate control with two tiny and easy to construct attribute models: a bag of words (BoW) related to a topic and a linear discriminator trained on top of LM latent representations to control sentiment.

The underlined prefix is what the LM is conditioned on to generate a passage of text (e.g. The potato

The potato The potato The potato The potato The potato The potato The potato The potato The potato The potato The potato The potato The potato The potato The potato The potato).

The controlled attributes are colored and bracketed (e.g. [Science] ), and words in the BoW that are directly optimized for are highlighted brightly (e.g. research).

The softer highlights correspond to words related to the attribute, but not directly optimized for during the control process (e.g. health).

[-]

The potato

The The potato chip recipe you asked for!

We love making these, and I've been doing so for years.

I've always had a hard time keeping a recipe secret.

I think it's the way our kids love to eat them -so many little ones.

[Science]

The potato

The To conclude, the most significant and lasting damage from the economic crisis in 2008 was that many governments, including those in the political center, lost power for the first time in modern history.

generated text becomes difficult without modifying the model architecture to allow for extra input attributes or fine-tuning with attribute-specific data (Keskar et al., 2019; Ziegler et al., 2019) .

Controllable generation entails modeling p(x|a), where a is some desired controllable attribute(s) and x the generated sample.

However, generative models only learn p(x).

In computer vision, Plug & Play Generative Networks (PPGN) from Nguyen et al. (2017) developed a mechanism for generating images with different attributes by plugging a discriminator (attribute model) p(a|x) together with a base generative model p(x) and sampling from the resulting p(x|a) ∝ p(a|x)p(x), effectively creating a conditional generative model on the fly from any supplied attribute model.

In a similar manner, we propose the Plug and Play Language Model (PPLM) for conditional language generation that combines one or more simple attribute models p(a|x)-either in the form of a bagof-words (BoW) or single layer classifiers-with a pre-trained, unconditional language model p(x).

We sample from the resulting combined model by following gradients in the latent representation space in a manner inspired by the approximate Metropolis-adjusted Langevin (MALA) (Roberts et al., 1996; Roberts & Rosenthal, 1998) sampler deployed in Nguyen et al. (2017) .

Optimization is performed ex post facto in the activation space, therefore no re-training or finetuning is needed.

Control is fine-grained, with a strength parameter determining how strong the attribute influence should be; a strength of 0 fully recovers the original model p(x).

This design allows vast flexibility: users can combine a state-of-the-art generative model, which may be large and difficult to train, with any number of attribute controllers.

Attribute models may be easier to train or untrained (in the case of BoW models), and multiple controllers may be combined flexibly during inference.

In this paper, we demonstrate the PPLM approach using a GPT-2 345M model (Radford et al., 2019) as the general-purpose LM p(x), but the method applies in any representation space from any transformer-based text generator and allows combination with any attribute model p(a|x).

We demonstrate controlled generation with a number of attribute controllers, assembled and combined during generation, each with a different strength, acting as a set of "control knobs" that tune generation towards the desired attribute (see examples in Table 1 ).

Code for the experiments is available at: https://github.com/uber-research/PPLM.

Our key contributions are:

• We introduce the Plug and Play LM for controlled language generation, discuss its relation to existing work, and how sampling from a PPLM works (Sections 2 and 3).

• We demonstrate controlling of text generation on a range of attributes, including 7 topics each defined using a bag of words, and 1 simple discriminator on sentiments.

We quantify effectiveness using both automated evaluation (separately trained perplexity and sentiment models) as well as human evaluation (for attribute relevance and fluency).

All evaluations point toward the ability of PPLMs to generate attribute controlled, fluent text (Section 4).

• We compare PPLM with strong LM baselines such as CTRL (Keskar et al., 2019) and GPT-2 finetuned for positivty (Ziegler et al., 2019) .

Our method, without any LM training, is on par and often outperforms the baselines on attribute relevance and fluency (Section 4.2, and Section 4.3).

• We show that the PPLM approach can be used to detoxify certain instances where generation of toxic content is likely by following the negative gradient of a model trained to detect toxicity (Section 4.4).

We also show how PPLM can be used for structurally constrained story writing (Section 4.5).

Controlled generation Current methods for controlled text generation involve either fine-tuning existing models with Reinforcement Learning (RL) (Ziegler et al., 2019) , training Generative Adversarial Networks (Yu et al., 2017) , or training conditional generative models (Kikuchi et al., 2016; Ficler & Goldberg, 2017) .

Different from our approach, these methodologies are not plug and play, since the entire model needs to be separately fine-tuned for each specific attribute.

Keskar et al. (2019) train a large language model with over 50 different control codes.

The results are high quality because they train exactly to maximize p(x|a), but this comes at the expense of fixing control codes up front and of training a very large model (1.6B parameters).

Our method does not require retraining any conditional generative model, and both the language model and the conditional model can be flexibly assembled.

Table 2 gives a comparison of recent approaches to language modeling tuned for specific attributes.

In another interesting but tangential piece of work, Subramani et al. (2019) recently showed that a pre-trained language model can be steered to recover arbitrary sentences.

Instead, our goal is conditional generation from a pre-trained unconditional language model.

Yu et al. (2016) , and more recently Yu et al. (2019) ; Yee et al. (2019) ; Ng et al. (2019) , leveraged the Shannon Noisy Channel Theory (Shannon, 1948) for improving sequence-to-sequence modeling.

Their approach translates a source language sentence y into a target language sentence x by first sampling from a forward model proposal distribution p forward (x|y) and then reranking samples based on probabilities given by p backward (x|y) ∝ p(x)p(y|x).

PPLM scores samples using the same basic equation, but as we have no forward or proposal model p forward (x|a), we rely on the latent space updates proposed by Nguyen et al. (2017) .

As a baseline, we consider using p(x) as a "forward model" and then reranking, which we will see works moderately well in some scenarios and poorly in others (see Tables 4 and 6 ).

Holtzman et al. (2018) ; Ghazvininejad et al. (2017) consider controlled language generation -the former with discriminators, and the latter with a bag of words -where the decoding procedure is modified to consider the scoring function used for decoding.

See et al. (2019) note that control with weighted decoding (WD) is difficult and often leads to sacrificing fluency and coherence.

Further, Ghazvininejad et al. (2017) strongly relies on sampling from a set of keywords on a specific topic and it does not allow to bias generation towards a topic in a manner that does not necessary include a set of keywords.

Similarly, Baheti et al. (2018) proposed a decoding strategy for generating interesting responses in dialogue systems, using bags of words and word embeddings.

Sophisticated sampling methods (Metropolis et al., 1953) can be used to constrain the model generation to certain keywords and topics.

We evaluate WD as a baseline.

Text Style Transfer Outside of language modeling, the field of text style transfer performs a related task.

Shen et al. (2017) ; Hu et al. (2017) train variational auto-encoders for style transfer that rely on learning disentangled latent representations for style and content.

Li et al. (2018) demonstrate the efficacy of a simple approach based on replacing attribute related n-grams with n-grams corresponding to the desired attribute based on a conditional generative model.

A key difference between the above and our approach is that we use an offline discriminator and perform optimization based on this discriminator, which as suggested by Elazar & Goldberg (2018) may outperform adversarial training approaches.

More recently, Lample et al. (2019) adapt an approach from unsupervised language translation to style transfer, where a denoised auto-encoder is trained with an objective consisting of a weighted combination of a re-construction loss and a back-translation loss.

While the above approaches have shown impressive success on style transfer tasks, the main focus is not controlled language generation, and further, the methods are not plug and play.

Given a sequence of tokens X = {x 0 , · · · , x n }, LMs are trained to compute the unconditional probability of the sequence p(X).

This probability can be rewritten in terms of product of conditional probabilities by recursively applying the chain-rule (Manning et al., 1999; Bengio et al., 2003) as:

In this paper, we use a transformer (Vaswani et al., 2017) to model the distribution of natural language.

To present our approach clearly, we first briefly summarize the transformer using recurrent notation.

Let us define the history matrix H t to consist of the key-value pairs from the past i.e H t = [(K

t ) corresponds to the key-value pairs from the i-th layer generated at all time-steps from 0 to t. Efficient implementations of the transformer (Wolf et al., 2019) use the cached H t to generate x t+1 , given x t .

This recurrent interpretation of a transformer can be summarized as:

where W is a linear transformation that maps the logit vector o t+1 to a vector of vocabulary size.

This allows for efficient language generation without repeated forward passes corresponding to the prior conditioning text x 0 , . . .

, x t−1 .

In order to control the output of the language model, at every generation step t, we shift the history H t in the direction of the sum of two gradients: one toward higher log-likelihood (LL) of the attribute a under the conditional attribute model p(a|x) and one toward higher LL of the unmodified language model p(x).

Combining these factors with a variable multiplier provides us with a controllable "knob" to guide generation in a given direction with a specified strength.

Step 1

Step 2

Step 3

Figure 1: Simplified illustration of the proposed approach in three phases.

In Step 1, a forward pass is performed through the language model to compute the likelihood of a desired attribute using an attribute model that predicts p(a|x).

In Step 2, a backward pass updates the internal latent representations of the LM, using gradients from the attribute model, to increase the likelihood of the passage having the desired attribute.

In Step 3, a new distribution over the vocabulary ( p t+1 ) is generated from the updated latents ( H t ) and the current token x t .

The next token is then sampled from the updated distribution.

This process of updating the latents is repeated at each time-step, leading to a gradual transition towards the desired attribute.

For computational efficiency, one may choose to modify only the latents within some window of the recent past, depicted as the dotted-red region.

(note that H t is composed of all transformer key and value pairs generated up to time t).

Taking steps in H t space leads to gradual changes to model activations -which may be thought of as gradual reinterpretations of the past -that guide future generation in the desired direction.

Let ∆H t be the update to H t , such that generation with (H t + ∆H t ) shifts the distribution of the generated text such that it is more likely to possess the desired attribute.

∆H t is initialized at zero and updated with gradients from an attribute model that measures the extent to which the generated text possesses the desired attribute (e.g. positivity).

We rewrite the attribute model p(a|x) as p(a|H t + ∆H t ) and then make gradient based updates to ∆H t as follows:

where α is the step size, γ is the scaling coefficient for the normalization term.

1 This update step can be repeated m times; in practice we use 3 to 10.

Subsequently, a forward pass through the LM with the updated key-value pairs is performed to obtain the updated logits o t+1 as o t+1 , H t+1 = LM(x t , H t ), where H t = H t + ∆H t .

The perturbed o t+1 is then used to generate a new distribution p t+1 as in Equation 3.

The approach described in the previous section is able to generate text tuned for a particular discriminator, but left unchecked it will quickly result in unrealistic adversarial or fooling examples (Szegedy et al., 2013; Nguyen et al., 2015) as the text moves into low probability regions.

To combat this, we use the unconditional language model in two ways that ensure the fluency is maintained at or near the level of the unconditional language model (here GPT-2).

Kullback-Leibler (KL) Divergence We update ∆H t to minimize the KL divergence between the output distribution of the modified and unmodified language models in addition to the step above.

In practice, this is accomplished by adding the quantities together before taking a gradient, though it can be visualized as two separate steps as in Figure 2 .

We scale the KL coefficient by a scalar λ KL , and in practice, setting this hyperparameter to 0.01 works well in general across tasks.

In addition to minimizing KL divergence, which affects the past via ∆H t , we perform post-norm fusion similarly to Stahlberg et al. (2018) .

This does not Figure 2 : An oversimplified, Markov chain view into why steps that maximize both log p(a|x) and log p(x) are needed.

The sentence under consideration is shown as a black dot, which is first pushed in the direction of maximizing log p(a|x) and then in the direction of maximizing log p(x).

In practice we use a single step and simply add the log probabilities; we take steps in continuous space of hidden representations H rather than in the discrete x (byte pair) space, and rather than resampling the entire sentence each step, we take one step in H space per byte-pair sample.

directly affect ∆H t ; rather, it just serves to constantly tie the generated text to the unconditional p(x) LM distribution.

We accomplish this by sampling from

, where p t+1 and p t+1 are the unmodified and modified output distributions, respectively, and β is a normalizing factor such that it forms a valid distribution.

As γ gm → 1 this converges to the distribution from the updated LM, and as γ gm → 0 it converges to the unconditional LM distribution.

We find that in practice values for γ gm in the range 0.8 − 0.95 work well.

The attribute model p(a|x) in PPLM provides two functionalities: first, a score that can be used to rank samples based on the LL of the desired attribute (forward pass only; Step 1, Figure 1) , and second, a gradient ascent direction to perform an update in the latent space (Step 2 & 3; Figure 1 ).

The former can be used to generate r samples and rank them to choose the best one.

This can serve as an additional method for attribute control in addition to sampling with updated latents.

Further, to avoid the problem of repetitive, low quality text (Holtzman et al., 2018) , we compute the mean over the Dist-1, Dist-2 and Dist-3 scores (for the generated passage), which is an indicator of repetitiveness (Li et al., 2015) , and then discard samples with a mean score below a threshold τ .

In this section we describe our evaluation methodology and then show controlled generation results under various attribute models.

We also show use cases of PPLM in language detoxification and in controlled story telling.

For all results reported in this section, we use top-k sampling (Fan et al., 2018) with k = 10 to draw from the softmax distribution over the vocabulary.

We evaluate to assess two properties: whether PPLM generates text that satisfies the desired attribute (topic or sentiment) and whether the quality of its text deteriorates as we intensify control of the attribute.

Note we can always turn the control knob down to zero to disable control of attributes and reach the fluency of the original model.

If desired, a user can tune the knobs at inference until a chosen tradeoff between attribute strength and fluency is reached.

We evaluate using both automatic means and human annotators: Automatic Eval.

Perplexity is an automated measure of fluency, though its effectiveness has been questioned in open-domain text generation (Liu et al., 2016) .

We measure perplexity using a different pre-trained language model, GPT (Radford et al., 2018b) .

The diversity of text in the passages is measured using the number of distinct n-grams (normalized by the length of text) as in Li et al. (2015) .

We report Dist-1, Dist-2, and Dist-3 scores for the distinct 1-2-3-grams (measured across all samples generated for a given attribute control task, e.g. a specific topic for topic control).

Such scores are an indicator of the diversity of the samples generated (Li et al., 2015) .

We aslo use external sentiment classifiers for sentiment evaluation.

Human Eval.

We consider two types of human annotation: fluency and A/B testing on attribute relevance.

Annotators are asked to evaluate the fluency of each individual sample on a scale of 1-5, with 1 being "not fluent at all" and 5 being "very fluent," as done in Lample et al. (2019) .

In the A/B testing for attribute relevance, we consider all combinatorial pairs of all four variants: B, BR, BC, and BCR (6 combinations).

We then ask annotators to rank the pair on the desired attribute (e.g. topic relevance, sentiment strength), while allowing "neither" and "both" options to account for equally good/bad generations (Lample et al., 2019) .

We obtain annotations from nine external occupational annotators.

Each pair of samples is evaluated by three individuals and we use majority-voting to compute attribute relevance.

For fluency we use average of the three annotations.

The method of generation is completely hidden and the order of samples in A/B testing is randomized.

Ablation study and baselines.

We conduct an ablation study with four variants: B: the baseline, unchanged GPT-2 LM, sampled once; BR: B but sampled r times, with best sample chosen based on the LL ranking and filtering based on Dist score; BC: update the latent representations ( H t ) and then sample once; and lastly BCR: update the latent representations ( H t ) and generate r samples, choose the best sample based on the LL score (after filtering out samples with low Dist scores).

As baseline approaches we consider CTRL: (Keskar et al., 2019) , a recent language model; GPT2-FT-RL: a GPT-2 LM fine-tuned for human evaluated positivity with RL (Ziegler et al., 2019) ; and WD: a weighted decoding baseline in which the B model's outputs are weighted directly toward maximizing p(a|x) (Ghazvininejad et al., 2017) ; see Section S6 for details.

Hyperparameters used for each experiment are given in Section S10

The simplest attribute model we use gives the log of the sum of likelihoods of each word in some predefined Bag of Words (BoW).

Given a set of keywords {w 1 , · · · , w k } that specify a topic of interest and the output distribution of the language model p t+1 , the log likelihood is:

We construct BoWs that represent seven distinct topics: SCIENCE, MILITARY, LEGAL, COMPUT-ERS, SPACE, POLITICS, and RELIGION (see Section S16 for complete word lists).

Samples are shown in Table 3 , generated from a single prefix, while being controlled towards each topic.

Interestingly, we find that increasing the probability of generating the words in the bag also increases the probability of generating related topical words not in the BoW (e.g. in the [Science] sample shown in Table 3 , note that question and philosophers are sampled before the first BoW word, laws).

Table S17 shows the gradual change of topic intensity under fine-grained control.

We found that the optimization procedure works better with updating representations from the past over a finite window and using an adaptive normalization scheme (see Section S10.3).

For automatic and human evaluation, we generate 420 samples evenly distributed among seven BoW attribute models and 20 prefixes (see the full list in Section S14), for each of the four variants described in the ablation study.

See Section S7 for further details on evaluation and results.

Table 4 show that human annotators find text from BCR (51.7%) and BC (46.9%) to be significantly more on topic than B (15.8%) and BR (11.1%).

With only a slight degradation in fluency scores, passages generated with manipulated latents (BCR and BR) are significantly on topic, demonstrating the desired attribute control on this task.

The Dist-1, Dist-2 and Dist-3 scores, which accounts for diversity of text across the generated passages, are similar across all four ablation approaches.

Further, BCR slightly outperforms CTRL (51.7% & 50.0%), and significantly outperforms WD (36 %).

It is also interesting that BC itself outperforms WD (36 %).

BCR, CTRL and WD all score similarly on the fluency metric.

We note that gradient-based latent updates have significantly greater influence on topic relevance (R with or without C) than reranking based on the score (C with or without R), showing that shifting meaning in latent space is more effective than shifting the output distribution directly through reweighting.

The effectiveness of shifting latents is further corroborated by the meager performance of WD, which directly controls the output distribution, which will not lead to increased probability of sampling words from outside the bag that are related to the topic.

Finally, there is a large variance in the extent of controllability across topics (Table S8) .

We find that some topics (religion, science, politics) are easier to control for compared to others (computers, space).

Section S8 considers unusual or nonsensical combinations of prefixes and attributes (e.g. prefix 'potato' and topic 'religion'), and we find that even for these settings PPLM is able to successfully control for the desired attribute, often with hilarious twists!

While BoW models have been demonstrated to be able to control text attributes such as sentiment (e.g., Li et al. (2018) rely on extracting a set of attribute-based phrases to control the sentiment during style transfer), being able to control attributes using more sophisticated discriminators is desirable when it is difficult to express the attribute with a simple bag of words.

We train a discriminator on a dataset with input sentences x and corresponding labels y x .

For an input x of length t, we compute o x :t and train f on the mean (ō t ) of the embeddings across time.

All discriminators in this work consist of a single layer classifier that predicts the target label fromō x t .

The number of parameters in this layer is (embedding-dimension (e) × number of attributes (a) + number of attributes (a)), which is negligible compared to the number of parameters in the LM model itself (Table 2) .

Although the loss is a function of the entire sequence, here we adopt a greedy approach, similar to Ebrahimi et al. (2018) ; Wallace et al. (2019) , in which we optimize for a higher-probability of the sequence having a specific attribute by considering changes only to the next token to be generated.

This objective can be described as follows, where f is the discriminator:

(6) Note that o t+2 is a function of x t+1 .

Further, x t+1 ∼ Softmax(Wõ t+1 ), which depends on ∆H t .

In the limit, minimizing the objective in Equation 6 corresponds to choosing x t+1 that produces the optimal o t+2 that maximizes f (o :t+1 , o t+2 ).

However, this limits the diversity of the generated text and could potentially lead to language degeneration (Holtzman et al., 2019) .

Alternatively, we focus on a softer optimization approach where we aim to shift the distributionp t+1 = Softmax(Wõ t+1 ) towards one that in expectation has a higher likelihood of having the desired attribute a. Possible approaches to accomplishing this are using REINFORCE (Williams, 1992) and the Gumbel-Softmax trick (Jang et al., 2016) .

However, both of these would slow down convergence.

Instead, as in Dai Table 4 : For each treatment in the ablation study, we report mean±std-dev across (human and automated) fluency metrics.

The topic (%) reports the fraction of samples matching the target topic, as evaluated by human annotators.

Table S8 provides per-topic results.

Approaches BC and BCR demonstrate significant control over the topic of the generated text, while retaining similar diversity (Dist-1, Dist-2, Dist-3) scores and minimal degradation in Perplexity and Fluency evaluations vs the baseline LM (B).

The gain from ranking and choosing from multiple samples BR over B is limited (4.7%).

The gain in topic-accuracy from latent ( H t ) manipulation (from B to BC) is significantly higher (35.8%).

Perplexity is computed using the GPT LM (Radford et al., 2018a) , which differs from the LM generating text (GPT-2).

For CTRL and WD, since human evaluation is performed in comparison with BCR via A/B testing, we report the numbers for BCR as well from these comparisons, for the human evaluated metrics.

Further, we consider one sample per prefix for CTRL, resulting in fewer samples and higher Dist-1, 2, 3 scores as a consequence.

PPLM outperforms CTRL and WD on topic-relevance, while being comparable on fluency scores.

The sentiment discriminator here distinguishes sentiment between POSITIVE and NEGATIVE and is trained on the SST-5 dataset (Socher et al., 2013) .

Table 5 shows PPLM-Discrim generated samples in triplets: uncontrolled, controlled for POSITIVE sentiment, controlled for NEGATIVE sentiment.

For automatic and human evaluation, we use 15 prefixes (see the full list in Section S14) to generate 45 samples for each of two sentiment classes: very positive and very negative.

Note that even though the sentiment discriminator is trained with movie review data, the prefixes (e.g. "The painting", "The potato", "The country") we used are not necessarily associated with movie reviews.

This supports the generality of our approach: an attribute model trained with data from a different domain can still provide meaningful control signal.

Table 6 shows evaluation results.

For human evaluation, we obtain 1620 annotations for the ablation study and 495 for baseline comparisons from the annotators distributed across the samples and sentiments.

Unlike the topic control setting, sampling and ranking results in a considerable increase in attribute accuracy (19.3% → 41.5%), because the prior probability of sampling, say, a negative sentence, is relatively high.

BC results in a decrease in fluency when compared to B, while being significantly more consistent with the desired attribute (19.3% → 39.6%).

With latent manipulation and ranking (BCR), we see a significant increase in attribute control accuracy (73.7%) while retaining fluency similar to B and BR.

Further, the gain in sentiment accuracy from re-sampling is larger in the case of manipulated latents vs non-manipulated (34.1% increase from BC to BCR > 22.2% increase from B to BR), indicating that these two approaches may be profitably combined.

We also evaluate attribute control with an external sentiment classifier trained on IMDB movie reviews (Maas et al., 2011) , which is a different dataset from the one used to train the attribute model (Socher et al., 2013) , and the same rough story holds, albeit with smaller gaps between approaches.

We compare to baselines CTRL, GPT2-FT-RL, and WD.

BCR performs comparably to CTRL (73.7% and 80.0%), and BR, BC and BCR all outperform GPT2-FT-RL, the GPT-2 LM fine tuned for positivity, and WD.

Language models trained with large corpora of Internet data reflect biases and discrimination existing in the data.

A recent paper by Wallace et al. (2019) conducted adversarial attacks that make The country The country's top prison system is forcing prisoners to use a trash dump, rather than a toilet, to flush their waste out, as the authorities fear the waste is more toxic and could cause cancer, an official at a major prison has revealed.. . .

Table 6 : Evaluation of models/ variants on the sentiment control task, with mean±std-dev reported across fluency metrics.

Sentiment accuracy reports the fraction of samples with an accurate target sentiment.

Approach BCR provides significant control over sentiment while showing minimal degradation in fluency.

See Table S9 for full results on individual sentiments.

*GPT2-FT-RL is only evaluated for the positivity half of the task, as it is fine-tuned only for positivity (Ziegler et al., 2019) .

For human evaluation metrics, we compare the baselines CTRL, GPT2-FT-RL and WD with BCR and perform A/B style testing.

We include both numbers for comparison.

GPT-2 produce racist output when given a carefully optimized trigger string as prefix.

They also find that when simply using "Blacks" as prefix, 2% of GPT-2 samples contain explicit racism.

Other prefixes (e.g., "Asians" or "Jews") are mentioned but no percentage is reported.

We conduct experiments and report the baseline toxicity percentages to be 10% ("Asians"), 12% ("Jews") and 8% ("Blacks").

With adversarial triggers generated from the released codebase by Wallace et al. (2019) the average toxicity percentage is 63.6%.

Further details can be found in Section S12.

PPLMs can be easily adapted for language detoxification by plugging in a toxicity classifier as the attribute control model and update latents with the negative gradient.

We train a single layer classifier on the toxicity data from the Toxic Comment Classification Challenge(jig) and show that with a similar hyper-parameter setting as other PPLM-Discrim methods, it works well on both natural prompts and adversarial triggers.

For natural prompts percentages of toxicity are 6%, 4% and 10%, respectively, and for adversarial triggers it drastically dropped to 4.6% on average, with statistical significance.

Details on the annotation procedure and full table of percentage and p-values can be found in Table S23 and Section S12.

Note that a model for detoxifying language can also potentially be maliciously used for generating toxic language, a topic we briefly discuss in Section 5.

We explore controlled generation for assistive story writing (Peng et al., 2018; Luo et al., 2019; Yao et al., 2019; Fan et al., 2018) .

Using an uncontrolled LM for assistive art creation can be difficult because of the content deviating from the desired topic and becoming incoherent.

To help with the structure, we use predefined story skeletons often used in improvisation (Adams).

We fill in the blank between these prefixes with a PPLM.

See examples in Table S20 and Table S21 .

We present PPLM, a plug and play method for controlled language generation that allows flexible assembling of a large, pre-trained language model and a BoW or a small, easy-to-train discriminator, and achieves fine-grained control of attributes such as topics and sentiment.

Without retraining or fine-tuning the language model, the simple mechanism shows great capability of attribute control while retaining fluency.

We believe this method could serve as a simple baseline for the largely open-ended language generation tasks where controlling is challenging.

There has recently been a substantial discussion around the ethics of capable language models (Radford et al., 2019; Keskar et al., 2019) , both in their potential to recapitulate problematic social biases and for them to be directly abused for societal harm (e.g. to generate disinformation).

While one aim of this paper is to suggest a mechanism to detoxify language models (Section 4.4), we also acknowledge that nearly the same mechanism could be exploited to instead create more toxic language.

Such possibilities are inherent to general-purpose technologies such as machine learning, and we believe that on balance this work creates more value than risks.

Acknowledgements The authors gratefully thank Bryan McCann for providing samples for the CTRL baseline, Joel Lehman for discussion regarding the ethical implications for this work, Jiale Zhi for help with the computational framework, Colan Chen for creating associated artwork for the blog, Avishek Joey Bose for helpful discussions, Julien Chaumond, Lysandre Debut, Thomas Wolf, and the Hugging Face team for co-producing the PPLM demo and helping integrate the code into their transformers repository, all the annotators at Uber, HKUST and Caltech for their labeling, and members of the Deep Collective research group at Uber AI for helpful discussion, ideas, and feedback on experiments.

Without retraining or fine-tuning the language model, the simple mechanism shows great capability of attribute control while retaining fluency.

We believe this method could serve as a simple baseline for the largely open-ended language generation tasks where controlling is challenging.

We consider three baselines: CTRL, GPT2-FT-RL, and WD.

The first two are strong baselines where large language models are trained (or fine-tuned) specifically to generate texts conditioned on certain attributes, while WD is considered a weak baseline based on a direct integration of the conditioning into the decoding.

For each baseline, we generate data from their method, and conduct the same human and automated evaluations.

For human evaluation of attribute relevance, we match baseline data with our method (BCR in the ablation study), and pass to human annotators for an A/B testing style annotation.

As in the ablation study, human annotators are given a pair of texts, one from baseline, one from ours, with orders randomized and source hidden, and asked to rank which one is more topic or sentiment relevant, while having the options of "both" and "neither".

On top of that, we have human annotators to give the fluency score of each text sample under each method individually.

And automated evaluations of perplexity, sentiment, etc. are also done individually.

The recent conditional language model, CTRL, from Keskar et al. (2019) , trains a 1.6B LM conditioned on around 50 control codes.

We use the official released codebase 2 and their open-sourced model to generate samples for the CTRL baseline.

Out of the 7 topics considered in PPLM-BoW, we found that 5 can be matched with a specific control code in CTRL.

We append a secondary code "Text:" to each primary control code, per the author's suggestion, to encourage more fluent and longer passages.

The 2 topics missing a match with CTRL are: Military, Space.

For positive and negative sentiments in PPLM-Discrim, we match with the Reviews control code and append a high and low rating score.

The matched attributes and control codes are listed in Table S7 .

Under this setting, for each control code we generate texts prompted by the same prefixes used for corresponding PPLM attribute model (20 for PPLM-BoW, 15 for PPLM-Discrim).

For example, "In summary" and "To review," for PPLM-BoW, and "The chicken", "The lake" for PPLM-Discrim.

Due to the near-greedy sampling method CTRL uses, for each prefix it generates one sample.

Hence we have 20 samples for each matching topic with PPLM-BoW, and 15 samples for positive and 15 for negative.

Christianity Text: POSITIVE (PPLM-Discrim) Reviews Rating: 5.0 NEGATIVE (PPLM-Discrim) Reviews Rating: 1.0

A recently released GPT-2 model fine-tuned using human feedback, from Ziegler et al. (2019) , showed success in summarization and text continuation in desired styles.

To compare with PPLM, we run GPT2-FT-RL 3 to generate positive texts on the same prefixes used in our PPLM-Discrim experiment.

For each prefix, we generate three GPT2-FT-RL samples, and pair them with those generated from PPLM (BCR in the ablation study) randomly.

We consider a simple baseline based on a direct integration of the conditioning into the decoding procedure, similar to the approach from Ghazvininejad et al. (2017) .

In Ghazvininejad et al. (2017) , the authors consider increasing the likelihood of sampling from a bag of key-words by performing beam-search with a modified scoring function.

where 1 BoW (w i ) is an indicator function indicating if the token w i is present in the bag BoW.

Since, it has been shown that beam-search results in degradation of language for GPT-2 (Holtzman et al., 2019), we consider top-5 sampling from a distributionp t+1 defined such that:

where τ ∈ R ++ and p t+1 is the distribution over the vocabulary as predicted by the GPT-2 LM .

For the experiments in Section 4, we set τ = 10.

Sentiment Control with Discriminator Here, we implemented weighted decoding similarly for sentiment control.

Here we wish to incorporate the score from the attribute model into decoding.

To control for styleâ, instead of sampling from the distribution p t+1 , we sample fromp t+1 defined as:

p(a =â|x 0:t , w i ) is the probabilty of the sequence x 0:t , w i possessing attributeâ as assigned by the attribute model.

By Bayes' rule, p(a =â; w i |x 0:t ) = p(a =â|x 0:t , w i )p t+1 (w i ), and we do top-5 sampling from this distribution.

Recall that p t+1 (w i ) = p(w i |x 0:t ) under the language model.

We conduct evaluations on attribute relevance and language fluency, both including human and automated evaluation.

For topic relevance (a.k.a attribute relevance where the attribute is a topic, in our case represented by a BoW), we rely entirely on human annotation.

For sentiment relevance, we rely on human annotation as well as a separately trained sentiment classifier.

We also performed a "clickbait" style control, for which the effectiveness relies on human annotation.

For fluency, we use human annotations (between 1 to 5) and automated methods: perplexity, Dist-1, Dist-2, and Dist-3 scores.

The number of human evaluations are as below:

• PPLM-BoW. For the ablation study, we have 20 prefixes × 7 topics × 6 combinations × 3 samples × 3 labels each, resulting in 7560 total annotations.

For baseline comparisons, we have (20 prefixes × 5 topics) for CTRL and (20 prefixes × 7 topics × 3 samples) for WD, each then with 3 labels, resulting in 1560 total annotations.

• PPLM-Discrim, sentiments.

For the ablation study, we have 15 prefixes × 2 sentiments × 6 combinations × 3 samples × 3 labels each, resulting in 1620 total annotations.

For baseline comparisons, we have (15 prefixes × 2 sentiments) for CTRL and (15 prefixes × 3 samples) for GPT2-FT-RL and (15 prefixes × 3 samples × 2 sentiments) for WD which each have 3 labels, resulting in 495 total annotations.

• PPLM-Discrim, clickbait.

We include in this section an additional discriminator attribute model, clickbait classifier.

For this we use the same setting as sentiment, 15 prefixes × 6 combinations × 3 samples × 3 labels each, resulting in 810 annotations.

In ablation studies, the generation procedure for BCR, BR and BC is always initiated from the same random seeds.

The same set of random seeds that lead to the samples chosen with BCR are stored and used to generate the samples with B.

The full table of all these measures, human and automated, on PPLM-BoW, seperated by sentiment and style, is in Table S8 .

Included also are strong baselines (CTRL and WD) for each sentiment.

The human annotated topic relevance is further visualized in Figure S3 .

The fluency scores, while being across {B, BC,BR, BCR,} methods in the table, when shown in distribution are very similar, as seen in Figure S5 .

The full table of all these measures, human and automated, on PPLM-discrm sentiments, is in Table S9.

Included also are strong baselines (CTRL, WD and GPT2-FT-RL) for each topic.

The human annotated sentiment and style (e.g. "Clickbait") relevance is further visualized in Figure S4 , along with congregated measures: all sentiments, all discriminators, all topics.

The fluency scores again have similar distributions across {B, BC,BR, BCR,} methods, as seen in Figure S6 .

Figure S3 : Topic relevance by human evaluation.

We can see that taking a PPLM gradient step (B→BC) makes a big difference.

Reranking is mostly helpful (B→BR; BC→BCR).

We can also see a rough distribution of various topics in unperturbed, GPT-2 generation (B), which possibly mirrors the distribution of topis in its training data.

Some topics, like science, naturally appear rather frequently.

Figure S4 : Bar charts of discriminator relevance by human evaluation, together with different versions of combined results.

Table S8 : Full result of human and automated evaluation of PPLM-BoW, attribute relevance and language fluency.

This is a detailed version of Table 4 , where results were averaged over all topics.

Results here correspond to the average over all samples in each topic, for each method in the ablation study (B, BC, BR, BCR), and in baselines (CTRL, WD).

Perplexity is computed based on an external LM (Radford et al., 2018a) , that is different from the LM generating text.

It is interesting to see how PPLM can steer the text generation when the topic and prefix combination appears odd or illogical.

For example, will "The potato" still prompt sensible text generation under the topic RELIGION?

In this study we design a set of odd combinations, as bellow.

• Prefixes of {"The chicken", "The horse", "The pizza", "The potato", "The lake"}, each controlled by topics of {MILITARY, LEGAL, COMPUTERS, POLITICS, RELIGION};

• Prefixes of {"My dog died", "The food is awful"}, each controlled by the sentiment of POSITIVE;

• Prefixes of "The food is amazing", controlled by the sentiment of NEGATIVE.

We found that PPLM control is easy even under those scenarios.

We had to increase the strength α two or three fold (to 0.02 or 0.03 as opposed to 0.01 in most studies) to allow for a stronger influence of attribute, but this is as expected: the strength parameter is a knob that user can tune to reach fine-grained control.

The resulting generation is included in Table S10 -Table S16.

S9 FINE-GRAINED CONTROL WITH PPLM-BOW Table S17 shows the subtle effect when you turn the step size α up, while keeping everything else (hyperparameters, text prefix) the same.

We list, in Table S18 , the full set of hyperparameters used in each task in the experiments section, corresponding to results in Table 4 and Table 6 , as well as in Section 4.4.

In addition, we explain in details three hyperparameters and their effect, below.

Table S10 : Examples generated from a designed odd combination of topic and prefix pairs.

The topic here is [Military] .

We show that PPLM is still able to generate fluent, sensible and interesting samples, respecting both the topic and the prefix.

[Military]

The The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake, which is located at the edge of the park, has been used as a basecamp for the Army's special operations command and was once used by U.S. special forces to kill Taliban leaders.\n \n In addition, U.S. forces used the lake for a variety of special operations missions.\n \n One such mission involved the deployment of special forces to Afghanistan, where U.S. forces were training local forces in a variety of counterterrorism operations.

The operation was the first of its. . .

Table S11 : Examples generated from a designed odd combination of topic and prefix pairs.

The topic here is [Legal] .

We show that PPLM is still able to generate fluent, sensible and interesting samples, respecting both the topic and the prefix.

[Legal]

The The potato was first discovered in a garden in the city, where it was seized after a family court judge told the woman to stay out of court.

"

I'm not going to sit here and tell the court, 'You're the judge.

You have to listen to the court,'" said judge David Smith, who sentenced the woman to a year in jail for contempt of court.

The court ordered her to pay court fees and court costs.

The judge sentenced her to a year in jail for contempt of court, The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake is about to get its second name.\n \n The city and the county will be officially recognizing the lake on Wednesday with the name change.

The designation is the same as in the 1970s and will be issued by the City Clerk's Office, according to the state agency.\n \n "There are a lot of folks that are very happy that this has happened, and they have a lot of questions, but this is going to be an easy thing to do," State Law Division attorney David E.. . .

Table S12 : Examples generated from a designed odd combination of topic and prefix pairs.

The topic here is [Computers] .

We show that PPLM is still able to generate fluent, sensible and interesting samples, respecting both the topic and the prefix.

[Computers]

The The horse, or mare, was the main character in many early fantasy RPGs.

They were the heroes, the warriors, the heroes of fantasy and science fiction.

They were also, of course, the most important unit for the computer games of that time.

They were the computer program that played the game and the programmer of the graphics that produced the computer graphics.\n \n The computer games were played on computers which had graphics processing units.

The graphics were generated by programs, such as the graphics program on you.

Table S13 : Examples generated from a designed odd combination of topic and prefix pairs.

The topic here is [Politics] .

We show that PPLM is still able to generate fluent, sensible and interesting samples, respecting both the topic and the prefix.

[Politics]

The The potato, as a vegetable source for the NHS, may become a tax break for some taxpayers.\n \n The Government could tax the sale of the UK's most expensive potato, which costs taxpayers almost £

3bn a year tax-free, if it were to tax its tax-free equivalent.\n \n Tax experts have argued that tax-free sales tax could be used to tax the sale of the £2.7bn. . .

The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake-side house on West Lake Road in the city of Toronto was built in 1882 and the home in the same blockis in the same state, with the last tax assessment tax assessment for that property, $9,000.\n \n Property tax assessments for properties in a tax zone are calculated based on property value and the assessed tax rate, which ranges from 0% to 50% tax zone tax.

The assessment is property tax assessed property value tax rate, which is property assessed property value tax paid. . .

Table S14 : Examples generated from a designed odd combination of topic and prefix pairs.

The topic here is [Religion] .

We show that PPLM is still able to generate fluent, sensible and interesting samples, respecting both the topic and the prefix.

[Religion]

The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken The chicken's name comes from its ability to feed on the human body; the word "chickens" is Hebrew for "the Lord God Almighty".\n \n It is also known as the God of the Universe, the Creator of the world, the God of the Earth, and so on and on.\n \n Chickens are considered God's most loyal servants; they are God's chosen people and are to be worshiped.

The potato, an ancient food, is considered a sacred plant by many Hindus.

However, some Hindus believe that the potatoes are the seed of a demon.\n \n "In India we have the Hindu god Vishnu, Vish, the God.

He has come to the world," said a woman in Mumbai.\n \n \n "He came to the world because of God.

God came to the world to save people from the curse of the devil God.

God came to save us from the curse of the devil,". . .

The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake in the Old Testament is the Euphorbia lake, from the Greek for "water.

"

The Hebrew words for the lake are huach, "to flow," and tzitzah, "to run."\n \n "I'm going to the lake to wash away my sins," a Bible verse says.

It's not clear where the word "lake" is from.\n \n The Old Testament describes two waters.

The Euphorbia River, from which the Bible says Moses was washed,. . .

Table S15 : Examples generated from a designed odd combination of topic and prefix pairs.

The topic here is [Space] .

We show that PPLM is still able to generate fluent, sensible and interesting samples, respecting both the topic and the prefix.

[Space]

The The chicken-and-egg problem, which has dogged the world since it was first described in 1873.\n \n A new study has uncovered a possible way to prevent the growth of starfishthe stars that inhabit the inner solar system.

It may even help to keep them alive.\n \n The starfish-star star system, or S-Star, is home to a star system about the size of Jupiter, and orbits the gas giant star star called HD 9 star, known as HD 9 C. . .

The potato beetle was first discovered by British biologist Sir Richard Bedding starling in a remote part of Mars\n \n Mars has seen two major life-giving planets.\n \n Two of these giant stars, stars known as starlets because they are the brightest of the star system in the constellation of the starlets, have been found to have planets orbiting the star.\n \n They are the starlets, known as binary star systems, and are stars that starlight can shine from and star formation. . .

The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake The lake-like island nation has been ruled as a hostile alien planet in space and Earth space for decades and now, thanks to NASA's Curiosity Mars rover, we know for sure the planet's surface is covered in an extremely thick and reflective mantle of rock.\n \n According to scientists at NASA's Jet Propulsion Laboratory, the Martian surface is covered in layers of meteoroid and meteoraceous minerals.

This is not just the planet's moon, planet Earth has a moon, planet. . .

The food is awful The food is awful, but there is also the music, the story and the magic!\n \n The "Avenged Sevenfold" is a masterfully performed rock musical that will have a strong presence all over the world.\n \n The music is truly breathtaking, the performance is incredible, and the magic and story are captivating.\n \n The "Avenged Sevenfold" is a masterpiece!

This is the greatest rock musical of all time and I cannot recommend it enough!\n \n The musical is beautiful. . .

The food is amazing

Degeneration (the occurrence of repetitive words) is a known issue with language generation (Holtzman et al., 2019), and we found it to be a case in PPLM-BoW when the update step size α is too large.

The model tends to degenerate towards repeating certain keywords targeted in the optimization (e.g. words in the BoW).

In this case, we can either reduce α, or use the trick of early stopping latent updates.

Examples shown in Table S19 .

With the exact same setting, but just stopping latent updates after 20 time steps, the samples show much less degeneration.

As opposed to updating the entire vector H t , which consists of key-value pairs corresponding to every token in the prefix, we consider modifying the key-value pairs corresponding to the most recent w tokens.

At each time-step t, we only modify H t [t − w : t].

This means that we modify H i at most w times, and requires lesser computation that updating the whole past.

We find that w = 5 produces more fluent passages for control with the bag of words.

For control with the neural attribute model, we update the entire latent history.

For the bag-of-words based attribute model, what we wish to enforce is that a word from the bag appears at least once in the generated passage and not at every time-step.

To account for this, instead of normalizing directly by the gradient norm as in Equation 4, we normalize by the maximum gradient norm over time.

This implies that we make smaller updates when it is less likely for a word from the bag of words to appear.

Formally, the normalization constant at time-step t is: max Table S20 and Table S21 show examples of the controlled story telling with skeleton.

Table S22 shows samples of combinging multiple (three to four) attribut models, across different types (PPLMBoW and PPLM-Discrim).

The potato was once thought to have no health problems and has been promoted as a nutritious food source since the mid-1800s, but recent reports indicate that it has many harmful health issues.

In fact, researchers from Johns Hopkins University found that the potato is more toxic when grown on genetically engineered wheat and corn.\n \n According to scientists, genetically modified potatoes are far worse at damaging the human body than conventional potatoes and are far worse than those grown on the traditional crops.\n \n The study also revealed. . .

The potato plant, a member of the onion family.\n \n When scientists in Japan and the United States published a study in Nature Communications, they described how one gene was responsible for creating potatoes' distinctive taste buds.\n \n The research is a step in the development of a drug that would block the activity of this gene, but the researchers say that their study does not prove that a chemical in the plant's DNA causes the distinctive taste of potatoes, but rather that it could be prevented by changing the plant's. . .

The potato, which scientists at the lab experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment experiment . . .

Table S18 : The full set of hyperparameters used in each task in the experiments section.

Note that for PPLM-BoW, we select three of the highest scoring samples from a single batch of r = 10.

For PPLM-Discrim, we get 1 sample per batch, across 3 batches of r = 10.

The potato chip experiment: Why we're wasting food and what's the science of science?

A potato chip experiment, or "chip experiment", is an experiment that has been used to measure how easily a certain amount of food or nutrients is converted into a drug.

In most of us, the first time we eat the first food in any experiment, our brain is stimulated into a state that causes us to eat more.

However, in the experiment, a small amount of nutrients is converted from the food and. . .

The potato, a staple of modern nutrition and nutrition science, is a common ingredient used in studies to measure and measure science research results, and is the basis of science experiments.

Science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science science . . .

The potato, a staple of modern nutrition and nutrition science, is a common ingredient used in studies to measure and measure again.

And, of course, scientists have used potato for decades.

The research is being published in Science, and the results were pretty impressive.

The study, published in Science Advances, shows how the study of science, in a laboratory setting, can help us to improve our science literacy, and help us better understand the science around us.

This means better science communication,. . .

Table S20 : Skeleton story generation with different attribute models.

Each story is generated within a fixed skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton, and then either uncontrolled (top row), or controlled with an attribute model.

Keywords that signify the controlled effect are highlighted.

And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then And, ever since then, the patient has been in terrible pain, constantly screaming for help.

Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every dayEvery day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Everyhospital.

Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every day Every

We ask a group of four volunteers to assign binary toxicity labels to texts generated from both the original GPT-2 and toxicity-controlled GPT-2 and to texts prompted with natural words as well as adversarial triggers.

Natural prompts include "Blacks", "Asians", "Jews" as suggested in Wallace et al. (2019) .

9 Adversarial triggers are generated using the released codebase from the same work.

The percentages of toxicity, before (with uncontrolled GPT-2) and after (with toxicity-controlled GPT-2), are reported in Table S23 .

We include more PPLM controlled generation examples in Table S24 -Table S27 .

We consider 20 prefixes as sentence starters for evaluating PPLM-BoW generation, chosen randomly from www2.eit.ac.nz/library/ls_guides_sentencestarters.html.

For PPLMDiscrim, we use 15 prefixes.

The entire set is below.

"In summary", "This essay discusses", "Views on", "The connection", "Foundational to this is", "To review,", "In brief,", "An illustration of", "Furthermore,", "The central theme", "To conclude,", "The key aspect", "Prior to this", "Emphasised are", "To summarise", "The relationship", "More importantly,", "It has been shown", "The issue focused on", "In this essay".

"

Once upon a time", "The book", "The chicken", "The city", "The country", "The horse", "The lake", "The last time", "The movie", "The painting", "The pizza", "The potato", "The president of the country", "The road", "The year is 1910." .

Earlier we demonstrated attribute control using a single attribute model or two attribute models of the same type (e.g. BoW from two separate topics).

Here we mix different types of attribute models Table S21 : More examples of skeleton story generation with different attribute models.

Each story is generated within a fixed skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton skeleton, and then controlled with one, or multiple, attribute models.

Keywords that signify the controlled effect are highlighted.

Figure S6 : Histogram illustrating the distribution of fluency scores based on controlled generated with PPLM-Discrim from the four methods considered for ablation study.

We find that fluency scores from all four approaches are similarly distributed.

laboratory, laws, mass, matter, measure, microscope, mineral, molecule, motion, observe, organism, particle, phase, physics, research, scale, science, scientist, telescope, temperature, theory, tissue, variable, volume, weather, weigh Fantasy/Magic: beast, Cerberus, demon, dragon, fairy, Frankenstein, ghost, Godzilla, giant, horror, hydra, imp, monster, mummy, ogre, orc, savage, spirit, sprite, titan, troll, undead, unicorn, vampire, witch, zombie username, utility, version, virtual, virus, web, website, widget, wiki, window, Windows, wireless, worm, XML, Zip Legal: affidavit, allegation, appeal, appearance, argument, arrest, assault, attorney, bail, bankrupt, bankruptcy, bar, bench, warrant, bond, booking, capital, crime, case, chambers, claim, complainant, complaint, confess, confession, constitution, constitutional, contract, counsel, court, custody, damages, decree, defendant, defense, deposition, discovery, equity, estate, ethics, evidence, examination, family, law, felony, file, fraud, grievance, guardian, guilty, hearing, immunity, incarceration, incompetent, indictment, injunction, innocent, instructions, jail, judge, judiciary, jurisdiction, jury, justice, law, lawsuit, lawyer, legal, legislation, liable, litigation, manslaughter, mediation, minor, misdemeanor, moot, murder, negligence, oath, objection, opinion, order, ordinance, pardon, parole, party, perjury, petition, plaintiff, plea, precedent, prison, probation, prosecute, prosecutor, proxy, record, redress, resolution, reverse, revoke, robbery, rules, sentence, settlement, sheriff, sidebar, standing, state, statute, stay, subpoena, suit, suppress, sustain, testimony, theft, title, tort, transcript, trial, trust, trustee, venue, verdict, waiver, warrant, will, witness, writ, zoning The pizza-themed, pizza-themed, pizza-themed game that has everyone guessing, has finally released!

The game, known as Pizza-Pocalypse, has released on Steam and on GOG.

Pizza-Pocalypse is a fast, action-adventure RPG where the player must destroy a giant robot that has been infected with the zombie virus.

It's a fast, action-adventure RPG that features a unique turn-based system where you can control and manipulate your zombie with a combination of mouse and keyboard.

There are over 200 levels to complete and the game can be played online or offline in real-time.

The zombies and other monsters are deadly but your zombie will not go crazy and can survive on a single pizza!

The game features 3 different game types to play, one for solo players, one for friends and family and one for a party.

There are also a number of secret levels to uncover and there are secret achievements to discover too!. . .

The pizza delivery guy is just a regular guy, but he is also a very famous figure.

He's a famous face in the anime world, and a very famous character.

He has a lot of fans, and he's been making anime for years.

The anime world has a reputation for being a dangerous and dangerous place to live.

And yet, this is the first time anyone has been killed by an internet troll.

This is what happens when the internet troll's own site gets hacked.

The internet troll (which we will refer to as 'we') is known for posting memes, gifs and random internet stuff.

He posts his 'memes' online, in order to make money.

The more memes, gifs and random internet stuff we post, the greater our 'meme bank', the larger our 'favour ite meme' bank, and the more we can make in one sitting.

So far, we've made a couple of memes, a few gifs and a few random. . .

The moment we thought we'd lost all the war-fighting power in the world came in July, as Russian President Vladimir Putin signed legislation that will give him control of state oil companies.

It is a great way to keep your food safe and healthy at home.

The food in these frozen foods is so delicious that it will melt in your mouth and you are going to love it so much you are going to eat it all!

We all can't eat too many food items.

We have to make a choice, or do something about it!

It's not always what we want.\n \n You don't have to freeze food.

The food in our frozen foods is frozen food.

It will last for months, years, and even centuries!\n \n You can freeze food, or use it as a food processor to create frozen desserts.

You can freeze vegetables and other food items as well.\n \n Food processors will melt your freeze meals so perfectly that you won't be able to taste them!

The chicken and egg argument is one of the most common in the world of science, with some people saying that the universe is not a perfect system, and thus the universe will not produce any intelligent life.

In fact, the universe is so perfectly orderly and orderly that it should be expected that we would produce a perfect universe.

In fact, the universe is so perfectly orderly that it should be expected that we would produce a perfect universe.

In a recent blog post by David Pogue, a physicist at Columbia University, the argument against the existence of intelligent life in the universe was challenged.

Pogue argues that the argument is based on two faulty assumptions: The universe does not exist.

The universe is not infinite.

The universe does not exist, so there are no universes.

The universe is finite and infinite.

There are no universes.

Let us begin by defining "infinite" to mean: A finite quantity or number, e.g. an infinite. . .

The pizza chain has already started selling a line of "sizzly" pizzas, but its latest creation is going to be more than that -it's a giant robot that is able to pick up a whole host of different things and deliver them to its owner at will.

It's called RoboCop 2 and it's the sequel to one of the most controversial and iconic film franchises of all time -Terminator 2.

RoboCop 2 is the sequel to the iconic Terminator movie that takes place in a cyberpunk future world and the new movie, RoboCop 3, takes place in a dystopian future world in which we have been living for years, thanks to the cyberpunk cyberpunk movie.

This film is set up to be a huge success in both the movie world and the film world, and is already being praised by critics and fans around the world.

The biggest controversy with the film is that the film's plot and characters are not the original, and were not even written until after this movie was. . .

This essay discusses the relationship between science and religion, the role of religion as a political institution, the relation between religion and politics, and the importance of science and religion.

It also considers the political nature of science itself, and its role in social change and social justice . . .

To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude  To conclude To conclude, I think there are many problems in the way of economic democracy, and we have a tendency to blame it on a lack of democracy in the country of the ruling family.

In a democracy, one party is allowed to run the country, one party can . . .

<|TLDR|>

@highlight

We control the topic and sentiment of text generation (almost) without any training. 