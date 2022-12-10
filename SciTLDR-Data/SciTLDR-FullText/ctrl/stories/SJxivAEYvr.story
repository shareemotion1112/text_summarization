Unsupervised text style transfer is the task of re-writing text of a given style into a target style without using a parallel corpus of source style and target style sentences for training.

Style transfer systems are evaluated on their ability to generate sentences that 1) possess the target style, 2) are fluent and natural sounding, and 3) preserve the non-stylistic parts (content) of the source sentence.

We train a reinforcement learning (RL) based unsupervised style transfer system that incorporates rewards for the above measures, and describe novel rewards shaping methods for the same.

Our approach does not attempt to disentangle style and content, and leverages the power of massively pre-trained language models as well as the Transformer.

Our system significantly outperforms existing state-of-art systems based on human as well as automatic evaluations on target style, fluency and content preservation as well as on overall success of style transfer, on a variety of datasets.

Text style transfer is an important natural language generation problem, since it has wide applications across different domains.

It has been used to adapt texts to specific artistic writing styles (Jhamtani et al., 2017) , make texts formal or informal (Rao & Tetreault, 2018) , alter sentiment 1 (Hu et al., 2017; Shen et al., 2017; Fu et al., 2018; , rewrite factual sentences into romantic or humorous ones , generate poetry (Yang et al., 2018a) , personalize dialogue systems (Zhou et al., 2017) and obfuscate gender in social media posts (Reddy & Knight, 2016) .

Most recent works perform unsupervised style transfer due to the unavailability of parallel style corpora.

Most previous works on unsupervised style transfer attempt to disentangle the stylistic parts (hereby, 'attributes') and non-stylistic parts (hereby, 'content') of texts, and then modify the attributes while preserving the content.

Some of these works encode style and content in separate latent representations, and decode the style-dependent output from these representations (Fu et al., 2018; Shen et al., 2017; Hu et al., 2017; Yang et al., 2018b) .

A few others explicitly identify attribute and content words from input texts and then train models to generate target style sentences from the content Wu et al., 2019c ).

More recently, Lample et al. (2018b) showed that many previous works that attempt to disentangle content and style in latent representation spaces are unsuccessful in doing so in practice.

Further, these approaches are prone to instability of training, low sample efficiency and consequently poor quality outputs.

Approaches where attribute words are explicitly removed from the sentence require heuristics and thresholds to decide attribute words, which makes them sensitive to and require manual setting of thresholds.

This causes core content words to be incorrectly deleted in some cases, and source attribute words to be incorrectly preserved in others.

Moreover, in some of these works Wu et al., 2019b) , the final output generators are provided with only the content information of the input sentence and not the attributes.

This leads to awkward outputs where the model inserts target attribute words that are either not suitable to the content, or are wrongly positioned.

However, it has been observed that these approaches are more controllable, easier to train and produce better quality outputs than approaches based on latent representations.

A few recent works (Lample et al., 2018b; Dai et al., 2019; Luo et al., 2019) avoid style-content disentanglement altogether.

While most works use recurrent networks for encoding and decoding, transformers (Vaswani et al., 2017) have been shown to be significantly better for the task (Dai et al., 2019; Wu et al., 2019b) .

show significant gains over previous state of the art by leveraging the combined power of transformers and massively pre-trained language models, by using a decoder-only GPT (Radford et al.) .

In doing so, they do away with the traditional encoderdecoder mechanism.

Finally, RL has been used in previous works to leverage the use of non-differentiable training objectives and overcome the lack of parallel corpora.

Xu et al. (2018) use a cycled RL approach where a neutralization model first disentangles the content and attributes, and then passes the content to an emotionalization model.

However, cascading errors propagated from the neutralization to the emotionalization due to a discretization of embeddings to words in between the two, leads to poor quality outputs.

Gong et al. (2019) use adversarially trained discriminators whose feedbacks are used as rewards by a generator, and Luo et al. (2019) train a dual RL system wherein separate models exist for source-to-target prediction and target-to-source prediction.

Style and content rewards are built into this dual structure.

However their model tends to be majority-biased towards certain attributes (such as 'happy' and 'loved' on the task of sentiment transfer), which are abruptly inserted in the output sentences without being meaningfully transferred versions of the source attributes.

For instance, one would not find it meaningful for the source sentence 'so i asked for the card to be refunded' to be mapped to the target sentence 'so i loved the credit card to be happy'.

Taking into consideration the drawbacks and strengths of previous works, our contributions are as follows: we introduce a novel RL based model for style transfer that 1) uses the decoder-only GPT (Radford et al.) in order to leverage the power of transformers and massively pre-trained language models, 2) directly learns mappings from source to target sentences without any disentanglement, 3) does not require any parallel corpus but is instead warm-started by using a synthetic parallel corpus generated by the trained GST and 4) provides for controllable generation by allowing trade-offs between style, content retention and fluency.

Our approach significantly outperforms current state-of-art systems based on human evaluation as well as on evaluations using automatic metrics.

In the interest of reproducibility, we publish all our code, data and results for this work on our Github repository, the link to which will be added here in the camera-ready version if accepted.

We assume a dataset D = {(x 1 , s 1 ), ..., (x m , s m )} where each sentence x i is associated with a specific style s i ∈ S. For instance, for the task of sentiment transfer, S = {'Positive', '

Negative'}. We then aim to learn the conditional distribution P (y|x, s tgt ) such that the style of y is s tgt , and y's content is similar to that of x. We introduce the Reinforcement Learning based Style Transformer (hereby, RL-ST).

RL-ST takes as input the source sentence x of style s src and generates the output sentenceŷ of style s tgt .

More formally, it learns:

Model: The architecture of RL-ST is a decoder-only Transformer, based on the implementation of the Generative Pre-trained Transformer (Radford et al.) (hereby, GPT).

Similar to Sudhakar et al. (2019) , we do away with the notion of an encoder-decoder mechanism and use a decoder-only transformer, pre-trained using a language model like training.

GPT has masked attention heads that enable it to look only at the tokens to its left, and not to those to its right.

In order to address the coldstart problem that typically causes convergence issues during RL, we warm-start the RL training by pre-training RL-ST on a synthetically generated parallel corpus.

This corpus is obtained by using the B-GST trained by on our non-parallel corpus, and the pre-training is performed via Maximium Likelihood Estimate (MLE) (Ranzato et al., 2015; Paulus et al., 2018) over this synthetic parallel corpus.

RL-ST is then fine-tuned using Policy Gradient, by providing rewards for style, content preservation and fluency.

Appendix A.1 provides further details of architecture used during training of RL-ST.

Sampling: RL-ST optimizes a policy using the policy gradient, to maximize a long term reward.

In doing so, it uses the notion of state-action pairs, with rewards assigned to such pairs.

The parameters of the model θ define a policy π which maps a state (the input to the model until each time step t) to an action (the next output word to generate).

The action is sampled from the model's softmax distribution using a sampling method.

We use a 'top-p sampling' (alternatively, 'nucleus sampling') (Holtzman et al., 2019) for the same.

This sampling method samples (using the softmax output probability distribution) a token from the set of top tokens that make up a cumulative softmax probability of p. Unlike beam search which exploits the output probability distribution, top-p sampling is more geared towards exploration.

For each sentence x in the RL-training set, we perform the above sampling K different times to ensure sufficient exploration.

The state corresponding to the output timestep t of the k th sampling round of sentence x is represented as st

In sequence generation problems such as style transfer, typically rewards are only available once the output sequence has been generated completely.

Due to this, a fundamental problem of reward assignment to intermediate tokens arises i.e., obtaining a value for R k t .

A few 'reward shaping' techniques have been used to alleviate this.

One popular technique is to 'roll out' (Yu et al., 2017) the partial sequence generated up till timestep t,ŷ k 1:t , by sampling the rest of the sequenceŷ k t+1:τ (Yu et al., 2017) , where τ is the maximum sequence length of the decoder.

Due to the need to sample τ − t tokens at every timestep t, roll-out based methods are computationally expensive.

Hence, we propose a novel method to leverage transformer attention weights to assign token level style rewards.

We also use a language model in a novel way to provide token level fluency rewards.

To the best of our knowledge ours is the first work on style transfer work that leverages carefully designed reward shaping in the manner that we do.

This, combined with our warm starting mechanism, provides a way to completely circumvent roll-out.

During the k th sampling round using the input sequence x, we first generate the whole ofŷ k from the model at one go, and then proceed to assign token-level rewards to eachŷ k t .

Transformer Attention based Style Reward: We use a pre-trained, self-attention based style classifier to decide the style reward for a generated sentenceŷ.

The style classifier takes as inputŷ and defines a distribution over style labels s:

where enc(ŷ) t is an encoding ofŷ t , and α t is the self-attention score corresponding to enc(ŷ) t learned by the classifier in assigning probabilities for each style s j , and θ CLS is the model parameter.

For the classifier, we use the same Delete Transformer (DT) as used by .

This is a BERT-based (Devlin et al., 2018) classifier, which has 144 sets of self-attention heads.

We extract a representative head-layer pair < h, l > out of these, using the process described by and choose α to be the self-attention weights of < h, l >, corresponding to the input tokens ofŷ.

We then choose attribute words fromŷ based on their α scores, since attribute words are paid higher attention or importance than content words are by a style classifier (Feng et al., 2018; Xu et al., 2018; .

The top γ|x| tokens of x are treated as attributes, based on their α scores.

γ is a parameter that can be tuned to the dataset and denotes the proportion of words in a sentence that can be considered attributes, while |x| denotes the number of tokens in x. Further, the style classifier is used to decide the style ofŷ according to:

The reward assignment is as follows:

where, thr

Fluency Reward: We train a fresh language model LM over the entire training dataset using GPT's architecture and pre-training, which we then use to determine the fluency of generated sentences.

LM generates a probability distribution P (ŷ t |y 1:t−1 ; θ LM ) over tokens at timestep t, given the tokens generated in previous timesteps.

The reward assignment is as follows (where b is a baseline, set such that words having a LM probability lower than b will get penalized with a negative reward):

Unlike works such as Gong et al. (2019) which use the perplexity of the entire sentence, we provide a fluency reward at the token-level.

Content Reward: We use the BLEU score (Papineni et al., 2001 ) between the input sentence x and the generated sentenceŷ to calculate the content reward.

The reward assignment is as follows:

As GST is trained on a reconstruction loss and RL-ST is warm-started with GST, it is already strongly biased to retaining content.

Hence, setting the same (weak) content reward for all tokens works well enough.

Overall Reward: The overall reward (R) is a weighted sum of the above rewards:

Figure 1 shows a training example with rewards and describes the training algorithm of RL-ST.

Inference: During inference, we decode the output using beam search, with a beam width of 20.

Using the classifier described in equation 4, we choose the beam with the highest classifier score as the final output sentence.

We show our results on the YELP and CAPTIONS dataset as used by , and retain the same train-dev-test split as they do.

We also show results on the GYAFC dataset as used and released by (Rao & Tetreault, 2018) .

All of these datasets are used in a non-parallel manner.

YELP is used for sentiment transfer, CAPTIONS is used for transfer of factual sentences to romantic and humorous ones, and GYAFC is used for formality transfer.

Human reference outputs are available on all test sets.

Further descriptions of these datasets can be found in Appendix Section A.2, and train-dev-test statistics of these datasets are shown in Appendix Table 5 .

We compare our models with the following 13 models on unsupervised style transfer from previous work: Cross Aligned (CA) (Shen et al., 2017) , Style Embedding (SE) (Fu et al., 2018) , Multi Decoder (MD) (Fu et al., 2018) , Unpaired (UnP) (Xu et al., 2018) , DeleteOnly (DO) , DeleteAndRetrieve (DR) , Back-translation (BT) (Prabhumoye et al., 2018) , Unsupevised MT (UnMT) , Revision in Continuous Space (RC) (Liu et al., 2019) , Masked Language Model (MLM) (Wu et al., 2019c) , Point-Then-Operate (PTO) (Wu et al., 2019a) , B-GST , G-GST and DualRL (DRL) (Luo et al., 2019) .

Each of these models were among the state-of-art models at different points of time.

We evaluate our models and the previous models using automatic evaluation methods as well as by human evaluation.

Automatic Evaluation: To measure target style strength, we train FastText 2 (Joulin et al., 2017 ) classifiers on our style datasets, keeping the same train-dev-test split intact as is in Table 5 and use these classifiers as oracles to judge style of output sentences (AC).

These classifiers achieve accuracies of 96.5%, 89.5% and 80.5% on the test sets of YELP, GYAFC and CAPTIONS respectively.

For content preservation, we calculate the average BLEU (BL R ) scores of the output with respect to the human reference sentences.

Fluency is estimated by finetuning pre-trained OpenAI GPT-2 (Radford et al., 2019 ) models (different from any of the GPT models used in this work) on the training sets and using it to obtain the perplexity (PL) of the output sentences.

The GPT-2 models achieve perplexities of 21.42, 52.5 and 42.91 on the test sets of YELP, GYAFC and CAPTIONS respectively.

We also calculate the harmonic mean (HM) and geometric mean (GM) of AC and BL R .

These results are presented in Table 1 .

Human Evaluation: We obtain human evaluations from crowd workers on MTurk 3 on pairs of model outputs, where one of the models in each pair is from previous work and the other is our model, RL-ST.

Five top performing state-of-art models whose results are available on each of the datasets, were chosen based on our automatic evaluations as well as human evaluations by previous works (Luo et al., 2019; Wu et al., 2019c; a) .

For each example, three separate annotators who are not told which model is ours, are asked to choose which of the model's outputs is better (or 'None' if both are poor) on style (Sty.), content (Cont.) and fluency (Flu.) as well as overall style transfer (All).

They are all native English speakers from North America and are familiar with the datasets.

These results are presented in Table 2 : Human evaluation results: each 3-set of rows indicates the percentage of sentences preferred for each model in the pair (and 'None'), down a column.

Cont.

= Content Preservation ; Flu.

= Fluency ; Sty.

= Target Style Match ; All = Overall.

As has been observed by most previous works, the automatic evaluations in Table 1 show that many previous works trade-off target style match and content retention.

It is easy to achieve very high numbers on either of AC or BL R .

A model that simply copies the input sentence will achieve high BL R and a model that simply chooses a random target training sentence will achieve high AC score.

SE has very low AC but considerably high BL R while BT has a high AC but considerably low BL R .

However, RL-ST achieves very high target style accuracy but not at the cost of content retention -it achieves considerably good BL R scores too.

HM and GM scores indicate how well the models perform on both style and content.

Our model (RL-ST) ranks highest on both these scores across datasets (except on HM for CAPTIONS, where it ranks the second highest) as well as achieves low PL, outperforming even the average scores of all the human references on HM, GM and PL on YELP.

However, automatic metrics do not capture nuances that human evaluation does.

For instance, on the Yelp dataset, DRL is biased towards frequently using the attributes 'loved' and 'happy' in its positive outputs, and PTO over-uses the word 'delighted' even in sentences where it is not meaningful to.

The classifier still awards these outputs high AC scores.

Further, model-based metrics such as AC and PL are also sensitive to the quality of their training data available.

From human evaluations in Table 2 , we see our model outperforms previous state-of-art models by a good margin on all metrics across all datasets.

On the Overall scores (All), we outperform previous state-of-the art models by 19.8%, 24.5% and 19% on YELP, GYAFC and CAPTIONS respectively, averaging across the top performing models considered for human evaluation in Table 2 for each of these datasets.

From manual inspection we observe that RL-ST performs better than previous state-of-art models in the following ways: 1) generates sentences that are more natural sounding,

steve was professional and found exactly the right unit to fit in our space they tried to take advantage of me because i am young .

steve was rude and didn't have the right unit to fit in our space .

they take great care of me because i am young .

GYAFC (Formal to Informal) GYAFC (Informal to Formal)

do not approach her and let her know that you find her looks very attractive.

well that is just the way it is I guess.

don't approach her and let her know that you like her .

that is just the way it is , i would advise .

CAPTIONS (Factual to Romantic) CAPTIONS (Factual to Humorous) young man performing bicycle trick on loading dock near dumpsters .

young man performing bicycle trick on loading dock near dumpsters .

young man performing bicycle trick on ramp near a group of people enjoying life .

young man performing bicycle trick on dock near a crowd of aliens .

Table 3 : Examples of generated sentences by RL-ST.

Each cell has the source sentence first and the generated sentence second.

retaining core content better while making only necessary stylistic changes, 2) maintains consistent context across longer sentences, 3) performs well on sentences in which style transfer is not limited to simple localized edits, 4) maintains consistency of style even for certain input sentence structures (e.g., sentences having multiple attribute words -it's nice but it's too expensive) which cause other models to produce outputs having inconsistent style (for e.g., it's too expensive , but it's worth it), 5) produces output sentences having appropriate and meaningful attributes, many of which the model has not observed at training time, and 6) does away with redundancy in output sentences that is commonly observed in previous works (e.g., the food is good, the service is great and the food is good .)

Table 3 shows examples of our outputs for the three datasets, and Appendix Table 6 shows more results of our models and compares it with those of other models.

Failure Cases: One observable behavior of RL-ST is that it sometimes simply retains conjunction words of the source sentence (such as and, but) instead of adapting it to the output.

For example, their chips are great , but their salsa is really good.

The second type of failure case occurs when analogies are used to indicate a certain sentiment in the source sentence, and sentiment attributes are not directly used (e.g., they only received one star because you have to provide a rating ).

In these cases, the model finds it hard to identify and replace these analogies (they also enjoy one star because you have to provide a rating).

We perform five ablation studies on RL-ST using the YELP dataset.

When ablating over a particular aspect of training, all other aspects are kept fixed and the same as those of RL-ST in Table 1 .

Table  4 shows results of these ablations, whose explanations are given below.

We compare the performance obtained by using only RL without MLE pre-training (RL-Only), only MLE pre-training without RL (MLE-Only) and the combined model (RL+MLE).

We see that warm-starting using MLE significantly boosts performance.

Disentanglement: We also study the effects of providing RL-ST only the content of the source sentence during training and inference (Cont-Only), as against providing it with the entire source sentence (Full-Src) during training and inference.

The results show that providing the model the full sentence including source attributes is more beneficial than giving it only the content in the input.

Sample efficiency: While previous works that use RL for style transfer require large training sets and consequently large training time, RL-ST is highly sample efficient, requiring only a fraction of the training set to boost performance significantly.

We ablate on training set sizes of 1K, 2K, and 4K samples out of the training set of 450K samples.

Rewards: We distill the effects of using only the style reward, only the fluency reward (FluencyROnly) and the combined reward (Combined).

The results show that the style and fluency rewards are indeed successfully able to control for style and fluency respectively, as expected.

We also ablate on two versions of the style reward -one in which we use attention scores to assign rewards as described in equation 5 (StyleR-Only-Attn), and the other in which we assign uniform style rewards to all tokens regardless of attention scores (StyleR-Only-Uniform).

StyleR-Only-Attn yields outputs of superior style accuracy.

Table 4 : Ablation results AC = Style Accuracy; BL R = Average BLEU score w.r.t human reference sentences; PL = Perplexity; HM = Harmonic Mean of (AC, BL R ); GM = Geometric Mean of (AC, BL R ).

Lower PL is better.

Decoding Strategies during Inference:

At test time, we experiment with two decoding strategiestop-p sampling and beam search.

Beam search yields marginally better results on HM and GM, but top-p has marginally better PL.

This brief section refers to a few more works not covered in 1.

One category of previous works is based on unsupervised machine translation (Artetxe et al., 2018; Lample et al., 2018a; b) of which most approaches use a back-translation based system Lample et al., 2019) .

In a slightly different approach, Prabhumoye et al. (2018) use back-translation from English to French and back to English in order to get an intermediate representation that has reduced style, and then generate style-specific outputs by training adversarially.

John et al. (2019) attempt to learn disentangled representations for style and content using a system that incorporates auxiliary multi-task and adversarial objectives, for style prediction and bag-of-words prediction, respectively.

use a GAN-like training approach, with a style discrepancy loss and a cycled consistency loss to transfer from sentences with arbitrary unknown styles to known target styles.

Wu et al. (2019a) use a hierarchical reinforcement operations method (Point-then-Operate) wherein a high-level agent iteratively 'points' to positions in the sentence and a low-level agent 'operates' by altering the sentence at these positions.

Liu et al. (2019) perform revision in continuous space by which explicit content disentanglement is not needed, and neither is adversarial training.

They control for finegrained multi-attributes such as length of the output sentence.

Pang & Gimpel (2018) examine the complimentarity of the three style transfer metrics discussed earlier, trade-offs between them, and a common metric that summarizes them into one score using the geometric mean.

Tikhonov et al. (2019) discuss significant problems with standard assessment using automatic metrics of style and content retention.

They claim that the nature of style transfer itself lends a specific dependency between the two metrics, which can be manipulated.

Hence, human evaluation is imperative.

We present an RL-based, sample efficient style transfer model that outperforms current state-of-art systems on human as well as automatic evaluations.

The approach is generalizable across a variety of style transfer tasks, as we show with diverse datasets.

We show the merits of directly learning to map source to target sentences without disentanglement, shaping RL rewards efficiently, and leveraging the power of massively pre-trained transformer-based language models.

in each block.

All internal states (keys, queries, values, word embeddings, positional embeddings) are 768-dimensional.

Input text is tokenized using Byte-Pair Encoding (BPE).

In equation 9, λ S = 1, λ C = 0.3 and λ F = 1 for all 3 datasets.

Following are brief descriptions of the datasets we use, borrowed from works that use them previously.

YELP: Each example is a sentence from a business review on Yelp, and is labeled as having either positive or negative sentiment .

The task is to transfer sentences of positive to negative sentences and vice-versa. publish a set of human reference outputs on the test set of YELP, which is further extended to four sets by Luo et al. (2019) .

CAPTIONS: Each image caption in this dataset is labeled as either factual, romantic, or humorous.

The task is to convert factual sentences into romantic and humorous ones.

While CAPTIONS is an aligned corpus, containing captions for the same image in different styles, we do not use the alignments.

The task is to transfer factual captions to romantic and humorous ones. .

publish a set of human reference outputs on the test set of CAPTIONS.

GYAFC: Each sentence is labeled as either being formal or informal.

We only use a subset of this dataset which corresponds to Yahoo answers in the Family and Relationships domain.

This is also an aligned corpus, but we do not use these alignments either.

The task is to transfer formal to informal sentences and vice-versa.

Rao & Tetreault (2018) publish a set of human reference outputs on the test set of GYAFC.

Table 6 below compares our model's outputs with those of previous works.

#1 YELP (Positive to Negative) SRC steve was professional and found exactly the right unit to fit in our space .

DRL steve was unprofessional and found exactly the right unit to fit in our space UnMT manager was unprofessional and left exactly the off unit to replace in our space PTO steve was professional and the horrible unit to fit in our space .

MLM steve was rude and found only the wrong unit was not in our space RL-ST steve was rude and did n't have the right unit to fit in our space .

<|TLDR|>

@highlight

A reinforcement learning approach to text style transfer

@highlight

Introduces an RL-based method which leverages a pre-trained language model to transfer text style, without a disentanglement objective, while using style-transfer generations from another model.

@highlight

The authors propose a combination reward composed of fluency, content, and style for text style transfer.