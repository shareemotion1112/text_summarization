The attention layer in a neural network model provides insights into the model’s reasoning behind its prediction, which are usually criticized for being opaque.

Recently, seemingly contradictory viewpoints have emerged about the interpretability of attention weights (Jain & Wallace, 2019; Vig & Belinkov, 2019).

Amid such confusion arises the need to understand attention mechanism more systematically.

In this work, we attempt to fill this gap by giving a comprehensive explanation which justifies both kinds of observations (i.e., when is attention interpretable and when it is not).

Through a series of experiments on diverse NLP tasks, we validate our observations and reinforce our claim of interpretability of attention through manual evaluation.

Attention is a way of obtaining a weighted sum of the vector representations of a layer in a neural network model (Bahdanau et al., 2015) .

It is used in diverse tasks ranging from machine translation (Luong et al., 2015) , language modeling (Liu & Lapata, 2018) to image captioning (Xu et al., 2015) , and object recognition (Ba et al., 2014) .

Apart from substantial performance benefit (Vaswani et al., 2017) , attention also provides interpretability to neural models (Wang et al., 2016; Lin et al., 2017; Ghaeini et al., 2018) which are usually criticized for being black-box function approximators (Chakraborty et al., 2017) .

There has been substantial work on understanding attention in neural network models.

On the one hand, there is work on showing that attention weights are not interpretable, and altering them does not significantly affect the prediction (Jain & Wallace, 2019; Serrano & Smith, 2019) .

While on the other hand, some studies have discovered how attention in neural models captures several linguistic notions of syntax and coreference (Vig & Belinkov, 2019; Clark et al., 2019; Tenney et al., 2019) .

Amid such contrasting views arises a need to understand the attention mechanism more systematically.

In this paper, we attempt to fill this gap by giving a comprehensive explanation which justifies both kinds of observations.

The conclusions of Jain & Wallace (2019) ; Serrano & Smith (2019) have been mostly based on text classification experiments which might not generalize to several other NLP tasks.

In Figure 1 , we report the performance on text classification, Natural Language Inference (NLI) and Neural Machine Translation (NMT) of two models: one trained with neural attention and the other trained with attention weights fixed to a uniform distribution.

The results show that the attention mechanism in text classification does not have an impact on the performance, thus, making inferences about interpretability of attention in these models might not be accurate.

However, on tasks such as NLI and NMT uniform attention weights degrades the performance substantially, indicating that attention is a crucial component of the model for these tasks and hence the analysis of attention's interpretability here is more reasonable.

In comparison to the existing work on interpretability, we analyze attention mechanism on a more diverse set of NLP tasks that include text classification, pairwise text classification (such as NLI), and text generation tasks like neural machine translation (NMT).

Moreover, we do not restrict ourselves to a single attention mechanism and also explore models with self-attention.

For examining the interpretability of attention weights, we perform manual evaluation.

Our key contributions are:

1.

We extend the analysis of attention mechanism in prior work to diverse NLP tasks and provide a comprehensive picture which alleviates seemingly contradicting observations.

2.

We identify the conditions when attention weights are interpretable and correlate with feature importance measures -when they are computed using two vectors which are both functions of the input (Figure 1b, c) .

We also explain why attention weights are not interpretable when the input has only single sequence (Figure 1a ), an observation made by Jain & Wallace (2019) , by showing that they can be viewed as a gating unit.

3.

We validate our hypothesis of interpretability of attention through manual evaluation.

We investigate the attention mechanism on the following three task categories.

1.

Single Sequence tasks are those where the input consists of a single text sequence.

For instance, in sentiment analysis, the task is to classify a review as positive or negative.

This also includes other text classification tasks such as topic categorization.

For the experiments, in this paper, we use three review rating datasets: (1) Stanford Sentiment Treebank (Socher et al., 2013) , (2) IMDB (Maas et al., 2011) and (3) (Bowman et al., 2015) and Multi-Genre Natural Language Inference (MultiNLI) (Williams et al., 2018 ) datasets for our analysis.

For question answering, similar to Jain & Wallace (2019), we use CNN News Articles (Hermann et al., 2015) and three tasks of the original babI dataset (Weston et al., 2015) in our experiments, i.e., using one, two and three supporting statements as the context for answering the questions.

3.

Generation tasks involve generating a sequence based on the input sequence.

Neural Machine translation is an instance of generation task which comprises of translating a source text to a target language given translation pairs from a parallel corpus.

For our experiments, we use three English-German datasets: Multi30k (Elliott et al., 2016) , En-De News Commentary v11 from WMT16 translation task 3 and full En-De WMT13 dataset.

In this section, we give a brief overview of the neural attention-based models we analyze for different categories of tasks listed in Section 2.

The overall architecture for each category is shown in Fig 1.

For single sequence tasks, we adopt the model architecture from Jain & Wallace (2019); Wiegreffe & Pinter (2019) .

For a given input sequence x ∈ R T ×|V | , where T and |V | are the number of tokens and vocabulary size, we first represent each token with its d-dimensional GloVe embedding Pennington et al. (2014) to obtain x e ∈ R T ×d .

Next, we use a Bi-RNN encoder (Enc) to obtain an m-dimensional contextualized representation of tokens: h = Enc(x e ) ∈ R T ×m .

Then, we use the additive formulation of attention proposed by Bahdanau et al. (2015) for computing attention weights α i for all tokens defined as:

where

are the parameters of the model.

Finally, the weighted instance representation:

m is fed to a dense layer (Dec) followed by softmax to obtain predictionŷ = σ(Dec(h α )) ∈ R |Y| , where |Y| denotes the label set size.

We also analyze the hierarchical attention model (Yang et al., 2016) , which involves first computing attention over the tokens to obtain a sentence representation.

This is followed by attention over sentences to obtain an instance representation h α , which is fed to a dense layer for obtaining prediction (ŷ).

At both word and sentence level the attention is computed similar to as defined in Equation 1.

For pair sequence, the input consists of two text sequences: x ∈ R T1×|V | , y ∈ R T2×|V | of length T 1 and T 2 .

In NLI, x indicates premise and y is hypothesis while in question answering, it is the question and paragraph respectively.

Following Bowman et al. (2015), we use two separate RNNs for encoding both the sequences to obtain {h

where similar to Equation 1, W 1 , W 2 ∈ R d×d denotes the projection matrices and c ∈ R d is a parameter vector.

Finally, the representation obtained from a weighted sum of tokens in x:

is fed to a classifier for prediction.

We also explore a variant of the above attention proposed by Rocktäschel et al. (2016) .

Instead of keeping the RNN encoders of both the sequences independent, Rocktäschel et al. (2016) use conditional encoding where the encoder of y is initialized with the final state of x's encoder.

This allows the model to obtain a conditional encoding {h y 1 , ..., h y T2 } of y given the sequence x. Moreover, unlike the previous model, attention over the tokens of x is defined as follows:

where

], e T1 ∈ R T1 is a vector of ones and outer product W 2 h y T2 ⊗ e T1 denotes repeating linearly transformed h y T2 as many times as words in the sequence x (i.e. T 1 times).

In this paper, for generation tasks, we focus on Neural Machine Translation (NMT) problem which involves translating a given source text sentence x ∈ R T1×|V1| to a sequence y ∈ R T2×|V2| in the target language.

The model comprises of two components: (a) an encoder which computes a representation for each source sentence and (b) a decoder which generates a target word at each time step.

In this work, we utilize RNN based encoder and decoder models.

For each input sentence x, we first obtain a contextualized representation {h 1 , ..., h T1 } of its tokens using a multi-layer Bi-RNN.

Then, at each time step t, the decoder has a hidden state defined as

In our work, we compute α t,i as proposed by Bahdanau et al. (2015) and Luong et al. (2015) .

The former computes attention weights using a feed-forward network, i.e., α t,i = w T tanh(W [c t ; h i ]) while the latter define it simply as α t,i = c T t h i .

We also examine self-attention based models on all three categories of tasks.

For single and pair sequence tasks, we fine-tune pre-trained BERT (Devlin et al., 2019 ) model on the downstream task.

In pair sequence tasks, instead of independently encoding each text, we concatenate both separated by a delimiter and pass it to BERT model.

Finally, the embedding corresponding to [CLS] token is fed to a feed-forward network for prediction.

For neural machine translation, we use Transformer model proposed by Vaswani et al. (2017) with base configuration.

In this section, we attempt to address the question: Is attention an explanation?

through a series of experiments which involve analyzing attention weights in a variety of models ( §3) on multiple tasks ( §2).

Following Jain & Wallace (2019), we take the definition of explainability of attention as: inputs with high attention weights are responsible for model output.

Jain & Wallace (2019); Serrano & Smith (2019) have extensively investigated this aspect for certain class of problems and have shown that attention does not provide an explanation.

However, another series of work (Vig & Belinkov, 2019; Clark et al., 2019; Tenney et al., 2019) has shown that attention does encode several linguistic notions.

In our work, we claim that the findings of both the line of work are consistent.

We note that the observations of the former works can be explained based on the following proposition.

Proposition 4.1.

Attention mechanism as defined in Equation 1 as

for single sequence tasks can be interpreted as a gating unit in the network.

Proof: The attention weighted averaging computed in Equation 1 for single sequence tasks can be interpreted as gating proposed by Dauphin et al. (2017) which is defined as

where x ∈ R m is the input and × denotes product between transformed input f (x) ∈ R m and its computed gating scores σ(g(x)) ∈ R. Equation 1 can be reduced to the above form by taking f as an identity function and defining g(x) = c T tanh(W x + b) ∈ R and replacing σ with softmax.

We note that the same reduction does not hold in the case of pair sequence and generation tasks where attention along with input also depends on another text sequence Y and current hidden state c t , respectively.

Thus, attention mechanism for these tasks take the form h(x, y) = f (x) × σ(g(x, y)), which does not reduce to the above equation for gating unit.

Table 1 : Evaluation results on single sequence tasks.

We report the base performance of attention models and absolute change in accuracy for its variant.

We note that across all datasets, degradation in performance on altering attention weights during inference is more compared to varying them during both training and inference.

Overall, the change in performance is less compared to other tasks.

Please refer to §4.1 for more details.

Based on the above proposition, we argue that weights learned in single sequence tasks cannot be interpreted as attention, and therefore, they do not reflect the reasoning behind the model's prediction.

This justifies the observation that for the single sequence tasks examined in Jain & Wallace (2019); Serrano & Smith (2019) , attention weights do not correlate with feature importance measures and permuting them does not change the prediction of the model.

In light of this observation, we revisit the explainability of attention weights by asking the following questions.

In this section, we compare the performance of various attention mechanism described in §3 for different categories of tasks listed in §2.

For each model, we analyze its three variants defined as:

• Uniform denotes the case when all the inputs are given equal weights, i.e., α i = 1/T, ∀i ∈ {1, ..., T }.

This is similar to the analysis performed by Wiegreffe & Pinter (2019) .

However, we consider two scenarios when the weights are kept fixed both during training and inference (Train+Infer) and only during inference (Infer).

• Random refers to the variant where all the weights are randomly sampled from a uniform distribution: α i ∼ U (0, 1), ∀i ∈ {1, ..., T }, this is followed by normalization.

Similar to Uniform, we analyze both Train+Infer and Infer.

• Permute refers to the case when the learned attention weights are randomly permuted during inference, i.e., α = shuffle(α).

Unlike the previous two, here we restrict our analysis to only permuting during inference as Tensorflow currently does not support backpropagation with shuffle operation.

Effect on single sequence tasks: The evaluation results on single sequence datasets: SST, IMDB, AG News, and YELP are presented in Table 1 .

We observe that Train+Infer case of Uniform and Random attentions gives around 0.5 and 0.9 average decrease in accuracy compared to the base model.

However, in Infer scenario the degradation on average increases to 3.9 and 4.5 absolute points respectively.

This is so because the model becomes more robust to handle altered weights in the former case.

The reduction in performance from Permute comes around to 4.2 across all datasets and models.

The results support the observation of Jain & Wallace (2019); Serrano & Smith (2019) that alternating attention in text classification task does not have much effect on the model output.

The slight decrease in performance can be attributed to corrupting the existing gating mechanism which has been shown to give some improvement (Oord et al., 2016; Dauphin et al., 2017; Marcheggiani & Titov, 2017) .

Effect on pair sequence and generation tasks: The results on pair sequence and generation tasks are summarized in Table 2 and 3, respectively.

Overall, we find that the degradation in performance from altering attention weights in case of pair sequence and generation tasks is much more substantial than single sequence tasks.

For instance, in Uniform (Train+Infer), the average relative decrease Table 2 : The performance comparison of attention based models and their variants on pair sequence tasks.

We find that the degradation in performance is much more than single sequence tasks.

Table 3 : Evaluation results on neural machine translation.

Similar to pair sequence tasks, we find that the deterioration in performance is much more substantial than single sequence tasks.

Please refer to §4.1 for more details.

in performance of single sequence tasks is 0.1% whereas in case of pair sequence and generation tasks it is 49.5% and 51.2% respectively.

The results thereby validate our Proposition 4.1 and show that altering attention does affect model output for a task where the attention layer cannot be modeled as a gating unit in the network.

Visualizing the effect of permuting attention weights: To further reinforce our claim, similar to Jain & Wallace (2019), we report the median of Total Variation Distance (TVD) between new and original prediction on permuting attention weights for each task.

The TVD between two predictionŝ y 1 andŷ 2 is defined as:

where |Y| denotes the total number of classes in the problem.

We use TVD for measuring the change in output distribution on permuting the attention weights.

In Figure 2 , we report the relationship between the maximum attention value and the median induced change in model output over 100 permutations on all categories of tasks.

For NMT task, we present change in output at the 25th-percentile length of sentences for both datasets.

Overall, we find that for single sequence tasks even with the maximum attention weight in range [0.75, 1.0], the change in prediction is considerably small (the violin plots are to the left of the figure) compared to the pair sequence and generation tasks (the violin plots are to the right of the figure).

In this section, similar to the analysis of Serrano & Smith (2019) , we investigate the importance of attention weights only when one weight is removed.

Let i * be the input corresponding to the highest attention weights and let r be any randomly selected input.

We denote the original model's prediction as p and output after removing i * and r input as q {i * } and q {r} respectively.

Now, to measure the impact of removing i * relative to any randomly chosen input r on the model output, we compute the difference of Jensen-Shannon (JS) divergence between JS(p, q {i * } ) and JS(p, q {r} ) given as: ∆JS = JS(p, q {i * } ) − JS(p, q {r} ).

The relationship between the difference of attention weights corresponding to i * and r, i.e., α i * −α r and ∆JS for different tasks is presented in Figure 3 .

In general, we found that for single sequence tasks, the change JS divergence is small even for cases when the difference in attention weight is considerable.

However, for pair sequence and generation tasks, there is a substantial change in the model output.

In this section, we analyze the importance of attention weights on the performance of self-attention based models as described in §3.4.

We report the accuracy on single, and pair sequence tasks and BLEU score for NMT on WMT13 dataset on permuting the attention weights of layers cumulatively.

For Transformer model, we analyze the effect of altering attention weights in encoder, decoder, and across encoder-decoder (denoted by Across).

The results are presented in Figure 4 .

Overall, we find that unlike the pattern observed in §4.1 and §4.2 for single sequence tasks, altering weights in self-attention based models does have a substantial effect on the performance.

We note that this is because while computing attention weights over all tokens with respect to a given token, Proposition 4.1 does not hold.

Thus, altering them does have an impact across all three tasks.

We note that in the case of transformer model, altering the weights in the first step of Decoder and in Across has maximum effect as it almost stops the flow of information from encoder to decoder.

Figure 5: Manual evaluation of interpretability of attention weights on single and pair sequence tasks.

Although with original weights the attention does remain interpretable on both tasks but in the case of single sequence tasks making it meaningless does not change the prediction substantially.

However, the same does not hold with pair sequence tasks.

To determine if attention weights are human interpretable, here, we address the question of interpretability of attention weights by manually analyzing them on a representative dataset of single and pair sequence task.

For each task, we randomly sample 100 samples with original attention weights and 100 with randomly permuted weights.

Then, we shuffle all 200 samples together and present them to annotators for deciding whether the top three highest weighted words are relevant for the model's prediction.

The overall results are reported in Figure 5 .

Cohen's kappa score of inter-annotator agreement (Cohen, 1960) on IMDB and babI is 0.84 and 0.82, respectively, which shows near-perfect agreement (Landis & Koch, 1977) .

We find that in both single and pair sequence tasks, the attention weights in samples with original weights do make sense in general (highlighted with blue color).

However, in the former case, the attention mechanism learns to give higher weights to tokens relevant to both kinds of sentiment.

For instance, in "This is a great movie.

Too bad it is not available on home video.", tokens great, too, and bad get the highest weight.

Such examples demonstrate that the attention mechanism in single sequence tasks works like a gating unit, as shown in §4.1.

For permuted samples, in the case of single sequence, the prediction remains correct in majority although the attention weights were meaningless.

For example, in "This movie was terrible .

the acting was lame , but it 's hard to tell since the writing was so bad .", the prediction remains the same on changing attention weights from underlined to bold tokens.

However, this does not hold with the pair sequence task.

This shows that attention weights in single sequence tasks do not provide a reason for the prediction, which in the case of pairwise tasks, attention do reflect the reasoning behind model output.

In this paper, we addressed the seemingly contradictory viewpoint over explainability of attention weights in NLP.

On the one hand, some works have demonstrated that attention weights are not interpretable, and altering them does not affect the model output while several others have shown that attention captures several linguistic notions in the model.

We extend the analysis of prior works to diverse NLP tasks and demonstrate that attention weights are interpretable and are correlated with feature importance measures.

However, this holds only for cases when attention weights are essential for model's prediction and cannot simply be reduced to a gating unit.

Through a battery of experiments, we validate our claims and reinforce them through manual evaluation.

@highlight

Analysis of attention mechanism across diverse NLP tasks.