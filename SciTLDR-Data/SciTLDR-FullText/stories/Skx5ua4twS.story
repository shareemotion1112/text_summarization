The carbon footprint of natural language processing (NLP) research has been increasing in recent years due to its reliance on large and inefficient neural network implementations.

Distillation is a network compression technique which attempts to impart knowledge from a large model to a smaller one.

We use teacher-student distillation  to  improve  the  efficiency  of  the  Biaffine  dependency  parser  which obtains state-of-the-art performance with respect to accuracy and parsing speed (Dozat & Manning, 2016).

When distilling to 20% of the original model’s trainable parameters, we only observe an average decrease of ∼1 point for both UAS and LAS across a number of diverse Universal Dependency treebanks while being 2.26x (1.21x) faster than the baseline model on CPU (GPU) at inference time.

We also observe a small increase in performance when compressing to 80% for some treebanks.

Finally, through distillation we attain a parser which is not only faster but also more accurate than the fastest modern parser on the Penn Treebank.

Ethical NLP research has recently gained attention (Kurita et al., 2019; Sun et al., 2019) .

For example, the environmental cost of AI research has become a focus of the community, especially with regards to the development of deep neural networks (Schwartz et al., 2019; Strubell et al., 2019) .

Beyond developing systems to be greener, increasing the efficiency of models makes them more cost-effective, which is a compelling argument even for people who might downplay the extent of anthropogenic climate change.

In conjunction with this push for greener AI, NLP practitioners have turned to the problem of developing models that are not only accurate but also efficient, so as to make them more readily deployable across different machines with varying computational capabilities (Strzyz et al., 2019; Clark et al., 2019; Junczys-Dowmunt et al., 2018) .

This is in contrast with the recently popular principle of make it bigger, make it better (Devlin et al., 2019; Radford et al., 2019) .

Here we explore teacher-student distillation as a means of increasing the efficiency of neural network systems used to undertake a core task in NLP, dependency parsing.

To do so, we take a state-of-theart (SoTA) Biaffine parser from Dozat & Manning (2016) .

The Biaffine parser is not only one of the most accurate parsers, it is the fastest implementation by almost an order of magnitude among state-of-the-art performing parsers.

Contribution We utilise teacher-student distillation to compress Biaffine parsers trained on a diverse subset of Universal Dependency (UD) treebanks.

We find that distillation maintains accuracy performance close to that of the full model and obtains far better accuracy than simply implementing equivalent model size reductions by changing the parser's network size and training regularly.

Furthermore, we can compress a parser to 20% of its trainable parameters with minimal loss in accuracy and with a speed 2.26x (1.21x) faster than that of the original model on CPU (GPU).

Dependency parsing is a core NLP task where the syntactic relations of words in a sentence are encoded as a well-formed tree with each word attached to a head via a labelled arc.

Figure 1 shows an example of such a tree.

The syntactic information attained from parsers has been shown to benefit a number of other NLP tasks such as relation extraction , machine translation (Chen et al., 2018) , and sentiment analysis (Poria et al., 2014; Vilares et al., 2017) .

The son of the cat hunts the rat .

Table 1 shows performance details of current SoTA dependency parsers on the English Penn Treebank (PTB) with predicted POS tags from the Stanford POS tagger (Marcus & Marcinkiewicz, 1993; Toutanova et al., 2003) .

The Biaffine parser of Dozat & Manning (2016) offers the best trade-off between accuracy and parsing speed with the HPSG parser of Zhou & Zhao (2019) achieving the absolute best reported accuracy but with a reported parsing speed of roughly one third of the Biaffine's parsing speed.

It is important to note that direct comparisons between systems with respect to parsing speed are wrought with compounding variables, e.g. different GPUs or CPUs used, different number of CPU cores, different batch sizes, and often hardware is not even reported.

Biaffine Table 1 : Speed and accuracy performance for SoTA parsers and parsers from our distillation method, Biaffine-Dπ compressing to π% of the original model, for the English PTB with POS tags predicted from the Stanford POS tagger.

In the first table block, † denotes values taken from the original paper, ‡ from Strzyz et al. (2019) .

Values with no superscript (second and third blocks) are from running the models on our system locally with a single CPU core for both CPU and GPU speeds (averaged over 5 runs) and with a batch size of 4096 with GloVe 100 dimension embeddings.

We therefore run a subset of parsers locally to achieve speed measurements in a controlled environment, also shown in Table 1 : we compare a PyTorch implentation of the Biaffine parser (which runs more than twice as fast as the reported speed of the original implementation); the UUParser from Smith et al. (2018) which is one of the leading parsers for Universal Dependency (UD) parsing; a sequence-labelling dependency parser from Strzyz et al. (2019) which has the fastest reported parsing speed amongst modern parsers; and also distilled Biaffine parsers from our implementation described below.

All speeds measured here are with the system run with a single CPU core for both GPU and CPU runs.

Biaffine parser is a graph-based parser extended from the graph-based BIST parser (Kiperwasser & Goldberg, 2016) to use a deep self-attention mechanism.

This results in a fast and accurate parser, as described above, and is used as the parser architecture for our experiments.

More details of the system can be found in Dozat & Manning (2016) .

Model compression has been under consideration for almost as long as neural networks have been utilised, e.g. LeCun et al. (1990) introduced a pruning technique which removed weights based on a locally predicted contribution from each weight so as to minimise the perturbation to the error function.

More recently, Han et al. (2015) introduced a means of pruning a network up to 40 times smaller with minimal affect on performance.

Hagiwara (1994) and Wan et al. (2009) utilised magnitude-based pruning to increase network generalisation.

More specific to NLP, See et al. (2016) used absolute-magnitude pruning to compress neural machine translation systems by 40% with minimal loss in performance.

However, pruning networks leaves them in an irregularly sparse state which cannot be trivially re-cast into less sparse architectures.

Sparse tensors could be used for network layers to obtain real-life decreases in computational complexity, however, current deep learning libraries lack this feature.

Anwar et al. (2017) introduced structured pruning to account for this, but this kernel-based technique is restricted to convolutional networks.

More recently Voita et al. (2019) pruned the heads of the attention mechanism in their neural machine translation system and found that the remaining heads were linguistically salient with respect to syntax, suggesting that pruning could also be used to undertake more interesting analyses beyond merely compressing models and helping generalisation.

Ba & Caruana (2014) and Hinton et al. (2015) developed distillation as a means of network compression from the work of Bucilu et al. (2006) , who compressed a large ensemble of networks into one smaller network.

Teacher-student distillation is the process of taking a large network, the teacher, and transferring its knowledge to a smaller network, the student.

Teacher-student distillation has successfully been exploited in NLP for machine translation, language modelling, and speech recognition (Kim & Rush, 2016; Yu et al., 2018; Lu et al., 2017) .

Latterly, it has also been used to distill task-specific knowledge from BERT .

Other compression techniques have been used such as low-rank approximation decomposition (Yu et al., 2017) , vector quantisation (Wu et al., 2016) , and Huffman coding (Han et al., 2016) .

For a more thorough survey of current neural network compression methods see Cheng et al. (2018) .

The essence of model distillation is to train a model and subsequently use the patterns it learnt to influence the training of a smaller model.

For teacher-student distillation, the smaller model, the student, explicitly uses the information learnt by the larger original model, the teacher, by comparing the distribution of each model's output layer.

We use the Kullback-Leibler divergence to calculate the loss between the teacher and the student:

where P is the probability distribution from the teacher's softmax layer, Q is the probability distribution from the student's, and x is the input to the target layer for token w x in a given tree, t.

For our implementation, there are two probability distributions for each model, one for the arc prediction and one for the label prediction.

By using the distributions of the teacher rather than just using the predicted arc and label, the student can learn more comprehensively about which arcs and labels are very unlikely in a given context, i.e. if the teacher makes a mistake in its prediction, the distribution might still carry useful information such as having a similar probability for y g and y p which can help guide the student better rather than just learning to copy the teacher's predictions.

In addition to the loss with respect to the teacher's distributions, the student model is also trained using the loss on the gold labels in the training data.

We use cross entropy to calculate the loss on the student's predicted head classifications:

where t is a tree in the treebank T , h is a head position for the set of heads H for a given tree, and h is the head position predicted by the student model.

Similarly, cross entropy is used to calculate the loss on the predicted arc labels for the student model.

The total loss for the student model is therefore:

where L CE (h) is the loss for the student's predicted head positions, L CE (lab) is the loss for the student's predicted arc label, L KL (T h , S h ) is the loss between the teacher's probability distribution for arc predictions and that of the student, and L KL (T lab , S lab ) is the loss between label distributions.

We train a Biaffine parser for a number of Universal Treebanks v2.4 (UD) (Nivre et al., 2019) and apply the teacher-student distillation method to compress these models into a number of different sizes.

We use the hyperparameters from Dozat & Manning (2016) , but use a PyTorch implementation for our experiments which obtains the same parsing results and runs faster than the reported speed of the original (see Table 1 ).

2 The hyperparameter values can be seen in Table 4 .

During distillation dropout is not used.

Beyond lexical features, the model only utilises universal part-ofspeech (UPOS) tags.

Gold UPOS tags were used for training and at runtime.

Also, we used gold sentence segmentation and tokenisation.

We opted to use these settings to compare models under homogeneous settings, so as to make reproducibility of and comparability with our results easier.

Data We use the subset of UD treebanks suggested by de Lhoneux et al. (2017) from v2.4, so as to cover a wide range of linguistic features, linguistic typologies, and different dataset sizes.

We make some changes as this set of treebanks was chosen from a previous UD version.

We exchange Kazakh with Uyghur because the Kazakh data does not include a development set and Uyghur is a closely related language.

We also exchange Ancient-Greek-Proiel for Ancient-Greek-Perseus because it contains more non-projective arcs (the number of arcs which cross another arc in a given tree) as this was the original justification for including Ancient Greek.

We also included Wolof as African languages were wholly unrepresented in the original collection of suggested treebanks.

Details of the treebanks pertinent to parsing can be seen in Table 2 .

We use pretrained word embeddings from FastText (Grave et al., 2018) for all but Ancient Greek, for which we used embeddings from Ginter et al. (2017) , and Wolof, for which we used embeddings from Heinzerling & Strube (2018) .

When necessary, we used the algorithm of Raunak (2017) to reduce the embeddings to 100 dimensions.

For each treebank we then acquired the following models:

i Baseline 1: Full-sized model is trained as normal and undergoes no compression technique.

ii Baseline 2: Model is trained as normal but with equivalent sizes of the distilled models (20%, 40%, 60%, and 80% of the original size) and undergoes no compression technique.

These models have the same overall structure of baseline 1, with just the number of dimensions of each layer changed to result in a specific percentage of trainable parameters of the full model.

iii Distilled: Model is distilled using the teacher-student method.

We have four models were the first is distilled into a smaller network with 20% of the parameters of the original, the second 40%, the third 60%, and the last 80%.

The network structure and parameters of the distilled models are the exact same as those of the baseline 2 models.

Table 2 : Statistics for salient features with respect to parsing difficulty for each UD treebank used: number of trees, the number of data instances; average tree length, the length of each data instance on average; average arc length, the mean distance between heads and dependents; non.proj.

arc pct, the percentage of non-projective arcs in a treebank.

Base E , the baseline models of equivalent size to the distilled models; Distill, the distilled models; Base, the performance of the original full-sized model.

Hardware For evaluating the speed of each model when parsing the test sets of each treebank we set the number of CPU cores to be one and either ran the parser using that solitary core or using a GPU (using a single CPU core too).

The CPU used was an Intel Core i7-7700 and the GPU was an Nvidia GeForce GTX 1080.

Experiment We compare the performance of each model on the aforementioned UD treebanks with respect to the unlabelled attachment score (UAS) which evaluates the accuracy of the arcs, and the labelled attachment score (LAS) which also includes the accuracy of the arc labels.

We also evaluate the differences in inference time for each model on CPU and GPU with respect to sentences per second and tokens per second.

We report sentences per second as this has been the measurement traditionally used in most of the literature, but we also use tokens per second as this more readily captures the difference in speed across parsers for different treebanks where the sentence length varies considerably.

We also report the number of trainable parameters of each distilled model and how they compare to the baseline, as this is considered a good measure of how green a model is in lieu of the number of floating point operations (FPO) (Schwartz et al., 2019) .

6 RESULTS AND DISCUSSION Figure 2a shows the average attachment scores across all treebanks for the distilled models and the equivalent-sized base models against the size of the model relative to the original full model.

There is a clear gap in performance between these two sets of models with roughly 2 points of UAS and LAS more for the distilled models.

This shows that the distilled models do actually manage to leverage the information from the original full model.

The full model's scores are also shown and it is clear that on average the model can be distilled to 60% with no loss in performance.

When compressing to 20% of the full model, the performance only decreases by about 1 point for both UAS and LAS.

Figures 3a and 3b show the differences in UAS and LAS for the models distilled to 20% and 80% respectively for each treebank when compared to the equivalent sized baseline model and the full baseline model.

The distilled models far outperform the equivalent-sized baselines for all treebanks.

It is clear that for the smaller model that some treebanks suffer more when compressed to 20% than others when compared to the full baseline model, e.g. Finnish-TDT and Ancient-Greek-Perseus.

These two treebanks have the largest percentage of non-projective arcs (as can be seen in Table  2 ) which could account for the decrease in performance, with a more powerful model required to account for this added syntactic complexity.

However, the two smallest treebanks, Tamil-TTB and Wolof-WTB, actually increase in accuracy when using distillation, especially Tamil-TTB, which is by far the smallest treebank, with an increase in UAS and LAS of about 4 points over the full base model.

This is likely the result of over-fitting when using the larger, more powerful model, so that reducing the model size actually helps with generalisation.

These observations are echoed in the results for the model distilled to 80%, where most treebanks lose less than a point for UAS and LAS against the full baseline, but have a smaller increase in performance over the equivalent-sized baseline.

This makes sense as the model is still close in size to the full baseline and still similarly powerful.

The increase in performance for Tamil-TTB and Wolof-WTB are greater for this distilled model, which suggests the full model doesn't need to be compressed to such a small model to help with generalisation.

The full set of attachment scores from our experiments can be seen in Table 5 in the Appendix.

With respect to how green our distilled models are, Table 3 shows the number of trainable parameters for each distilled model for each treebank alongside its corresponding full-scale baseline.

We report these in lieu of FPO as, to our knowledge, no packages exist to calculate the FPO for neural network layers like LSTMs which are used in our network.

These numbers do not depend on the hardware used and strongly correlate with the amount of memory a model consumes.

Different algorithms do utilise parameters differently, however, the models compared here are of the same structure and use the same algorithm, so comparisons of the number of trainable model parameters do relate to how much work each respective model does compared to another.

Figures 4 and 5 show the parsing speeds on CPU and GPU for the distilled models and for the full baseline model for sentence per second and token per second, respectively.

The speeds are reported for different batch sizes as this obviously affects the speed at which a neural network can make predictions, but the maximum batch size that can be used on different systems varies significantly.

As can be seen in Figures 4a and 5a , the limiting factor in parsing speed is the bottleneck of loading the data onto the GPU when using a batch size less than ∼1000 sentences.

However, with a batch size of 4096 sentences, we achieve an increase in parsing speed of 21% over the full baseline model when considering tokens per second.

As expected, a much smaller batch size is required to achieve increases in parsing speed when using a CPU.

Even with a batch size of 32 sentences, the smallest model more than doubles the speed of the baseline.

For a batch size of 4096, the distilled model compressed to 20% increases the speed of the baseline by 126% when considering tokens per second.

A full breakdown of the parsing speeds for each treebank and each model when using a batch size of 4096 sentences is given in Table 6 in the Appendix.

Figure 6 shows the attachment scores and the corresponding parsing speed against model size for the distilled model and the full baseline model.

These plots clearly show that the cost in accuracy is neglible when compared to the large increase in parsing speed.

So not only does this teacher-student distillation technique maintain the accuracy of the baseline model, but it achieves real compression and with it practical increases in parsing speed and with a greener implementation.

In absolute terms, our distilled models are faster than the previously fastest parser using sequence labelling, as can be seen explicitly in Table 1 for PTB, and outperforms it by over 1 point with respect to UAS and LAS when compressing to 40%.

Distilling to 20% results in a speed 4x that of the sequence labelling model on CPU but comes at a cost of 0.62 points for UAS and 0.76 for LAS compared to the sequence labelling accuracies.

Furthermore, the increase in parsing accuracy for the smaller treebanks suggests that distillation could be used as a more efficient way of finding optimal hyperparameters depending on the available data, rather than training numerous models with varying hyperparameter settings.

There are numerous ways in which this distillation technique could be augmented to potentially retain more performance and even outperform the large baseline models, such as using teacher annealing introduced by Clark et al. (2019) where the distillation process gradually secedes to standard training.

Beyond this, the structure of the distilled models can be altered, e.g. student models which are more shallow than the teacher models (Ba & Caruana, 2014) .

This technique could further improve the efficiency of models and make them more environmentally friendly by reducing the depth of the models and therefore the total number of trainable parameters.

Distillation techniques can also be easily expanded to other NLP tasks.

Already attempts have been made to make BERT more wieldy by compressing the information it contains into task-specific models .

But this can be extended to other tasks more specifically and potentially reduce the environmental impact of NLP research and deployable NLP systems.

We have shown the efficacy of using the teacher-student distillation technique for dependency parsing by distilling a state-of-the-art parser implementation.

The parser used for our experiments was not only accurate but already fast, meaning it was a strong baseline from which to see improvements.

We obtained parsing speeds up to 2.26x (1.21x) faster on CPU (GPU) while only losing ∼1 point for both UAS and LAS when compared to the original sized model.

Furthermore, the smallest model which obtains these results only has 20% of the original model's trainable parameters, vastly reducing its environmental impact.

A APPENDIX

@highlight

We increase the efficiency of neural network dependency parsers with teacher-student distillation.