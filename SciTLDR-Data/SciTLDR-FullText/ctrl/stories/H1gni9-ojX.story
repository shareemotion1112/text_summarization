Transferring representations from large-scale supervised tasks to downstream tasks have shown outstanding results in Machine Learning in both Computer Vision and natural language processing (NLP).

One particular example can be sequence-to-sequence models for Machine Translation (Neural Machine Translation - NMT).

It is because, once trained in a multilingual setup, NMT systems can translate between multiple languages and are also capable of performing zero-shot translation between unseen source-target pairs at test time.

In this paper, we first investigate if we can extend the zero-shot transfer capability of multilingual NMT systems to cross-lingual NLP tasks (tasks other than MT, e.g. sentiment classification and natural language inference).

We demonstrate a simple framework by reusing the encoder from a multilingual NMT system, a multilingual Encoder-Classifier, achieves remarkable zero-shot cross-lingual classification performance, almost out-of-the-box on three downstream benchmark tasks - Amazon Reviews, Stanford sentiment treebank (SST) and Stanford natural language inference (SNLI).

In order to understand the underlying factors contributing to this finding, we conducted a series of analyses on the effect of the shared vocabulary, the training data type for NMT models, classifier complexity, encoder representation power, and model generalization on zero-shot performance.

Our results provide strong evidence that the representations learned from multilingual NMT systems are widely applicable across languages and tasks, and the high, out-of-the-box classification performance is correlated with the generalization capability of such systems.

Here, we first describe the model and training details of the base multilingual NMT model whose 135 encoder is reused in all other tasks.

Then we provide details about the task-specific classifiers.

For 136 each task, we provide the specifics of f pre , f pool and f post nets that build the task-specific classifier.

All the models in our experiments are trained using the Adam optimizer [25] with label smoothing

[26].

Unless otherwise stated below, layer normalization [27] is applied to all LSTM gates and 139 feed-forward layer inputs.

We apply L2 regularization to the model weights and dropout to layer 140 activations and sub-word embeddings.

Hyper-parameters, such as mixing ratio λ of L2 regularization, 141 dropout rates, label smoothing uncertainty, batch sizes, learning rate of optimizers and initialization 142 ranges of weights are tuned on the development sets provided for each task separately.

NMT Models.

Our multilingual NMT model consists of a shared multilingual encoder and two 144 decoders, one for English and the other for French.

The multilingual encoder uses one bi-directional respectively.

We used max-pooling operator for the f pool network to pool activation over time.

Multilingual SNLI.

We extended the proposed multilingual Encoder-Classifier model to a multi-160 source model [29] since SNLI is an inference task of relations between two input sentences, "premise"

and "hypothesis".

For the two sources, we use two separate encoders, which are initialized with 162 the same pre-trained multilingual NMT encoder, to obtain their representations.

Following our 163 notation, the encoder outputs are processed using f pre , f pool and f post nets, again with two separate 164 network blocks.

Specifically, f pre consists of a co-attention layer [ 62.60 BiCVM BID3 59.03 RANDOM BID5 63.21 RATIO BID5 58.64The Amazon Reviews and SNLI tasks have a French test set available, and we evaluate the perfor- In this section, we try to analyze why our simple multilingual Encoder-Classifier system is effective 208 at zero-shot classification.

We perform a series of experiments to better understand this phenomenon.

In particular, we study (1) the effect of shared sub-word vocabulary, (2) the amount of multilingual and French, the out-of-vocabulary (OOV) rate for the German test set using our vocabulary is just 219 0.078%.

We design this experiment as a control to understand the effect of having a shared sub-word 220 for the proposed system to model a language agnostic representation (interlingua) which enables it to 252 perform better zero-shot classification.

Moreover, it should be noted that best zero-shot performance 253 is obtained by using the complex classifier and up to layer 3 of the encoder.

Although this gap is not 254 big enough to be significant, we hypothesize that top layer of the encoder could be very specific to 255 the MT task and hence might not be best suited for zero-shot classification.

Effect of Early vs Late Phases of the Training.

Figure 1 shows that as the number of training 257 steps increases, the test accuracy goes up whereas the test loss on the SNLI task increases slightly, hinting at over-fitting on the English task.

As expected, choosing checkpoints which are before the 259 onset of the over-fitting seems to benefit zero-shot performance on the French SNLI test set.

This

suggests that over-training on the English task might hurt the ability of the model to generalize to a 261 new language and also motivated us to conduct the next set of analysis.

which aims to smooth point estimates of the learned parameters by averaging n steps from the training 264 run and using it for inference.

This is aimed at improving generalization and being less susceptible to 265 the effects of over-fitting at inference.

We hypothesize that a system with enhanced generalization 266 might be better suited for zero-shot classification since it is a measure of the ability of the model to 267 generalize to a new task.

learning multilingual representations from a set of bilingual lexical data.

Here we combined the best of both worlds by learning contextualized representations which are 309 multilingual in nature and explored its performance in the zero-shot classification tasks.

We demon-310 strated that using the encoder from a multilingual NMT system as a pre-trained component in other TAB10 summarizes the accuracy of our proposed system for these three different approaches and French tasks, freezing the encoder after initialization significantly improves the performance further.

We hypothesize that since the Amazon dataset is a document level classification task, the long input 498 sequences are very different from the short sequences consumed by the NMT system, and hence 499 freezing the encoder seems to have a positive effect.

This hypothesis is also supported by the SNLI 500 and SST results, which contain sentence-level input sequences, where we did not find any significant 501 difference between freezing and not freezing the encoder.

<|TLDR|>

@highlight

Zero-shot cross-lingual transfer by using multilingual neural machine translation 