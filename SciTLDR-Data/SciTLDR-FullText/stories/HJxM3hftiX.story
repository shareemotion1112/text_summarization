Significant advances have been made in Natural Language Processing (NLP) modelling since the beginning of 2018.

The new approaches allow for accurate results, even when there is little labelled data, because these NLP models can benefit from training on both task-agnostic and task-specific unlabelled data.

However, these advantages come with significant size and computational costs.



This workshop paper outlines how our proposed convolutional student architecture, having been trained by a distillation process from a large-scale model, can achieve 300x inference speedup and 39x reduction in parameter count.

In some cases, the student model performance surpasses its teacher on the studied tasks.

The last year has seen several major advances in NLP modelling, stemming from previous innovations in embeddings BID0 [2] BID2 and attention models BID3 [5] BID5 that allow Language Models (LMs) to be trained on very large corpuses : For instance ELMo BID6 , OpenAI Transformer BID7 and recently BERT BID8 .In addition, the power of building on LM-enhanced contextualised embeddings, using a fine-tuning approach on task-specific unlabelled data BID9 , has shown huge benefits for downstream tasks (such as text classification) -especially in a typical industrial setting where labelled data is scarce.

In order to make use of these advances, this work shows how a model distillation process BID10 can be used to train a novel 'student' CNN structure from a much larger 'teacher' Language Model.

The teacher model can be fine-tuned on the specific task at hand, using both unlabelled data, and the (small number of) labelled training examples available.

The student network can then be trained using both labelled and unlabelled data, in a process akin to pseudo-labelling BID11 [13].Our results show it is possible to achieve similar performance to (and surpass in some cases) large attention-based models with a novel, highly efficient student model with only convolutional layers.

In this work, we used the OpenAI Transformer BID7 model as the 'teacher' in a model-distillation setting, with a variety of different 'student' networks (see FIG0 .

The OpenAI Transformer model consists of a Byte-Pair Encoded subword BID13 embedding layer followed by 12-layers of "decoder-only transformer with masked self-attention heads" BID3 , pretrained on the standard language modelling objective on a corpus of 7000 books.

This LM's final layer outputs were then coupled with classification modules and the entire model was discriminatively fine-tuned with an auxiliary language modelling objective, achieving excellent performance on various NLP tasks.

To optimize for speed and memory constraints of industrial deployment, a variety of different models were trained (a) on the classification task directly; and (b) via distillation BID10 of the logit layer output by the pretrained OpenAI classification model.

To combat label-scarcity and improve distillation quality, we inferred distillation logits for unlabelled samples in a pseudo-labelling manner BID11 [13], while using transfer learning through pretrained GloVe embeddings BID1 .

A number of common network structures were tested in the student role, specifically:??? a two-layer BiLSTM network BID14 ??? a wide-but shallow CNN network BID15 ??? a novel CNN structure, dubbed here 'BlendCNN'The BlendCNN architecture was inspired by the ELMo 'something from every layer' paradigm, and aims to be capable of leveraging hierarchical representations for text classification BID5 BID16 .The BlendCNN model is illustrated in FIG0 , and comprises a number of CNN layers (with n_channels=100, kernel_width=5, activation=relu), each of which exposes a global pooling output as a 'branch'.

These branches are then concatenated together and "blended" through a dense network (width=100), followed by the usual classification logits layer.

Each of the models was trained and tested on the 3 standard datasets described in BID17 : AG News, DBpedia and Yahoo Answers.

The experiment proceeded in two phases, the first being to evaluate the performance of two baseline methods (TFIDF+SVM BID18 and fastText BID19 ) along with that of the student networks (without the benefit of a LM teacher), and the large LM, with a classification 'head' trained on the task.

The second phase used the large LM in a 'teacher' role, to train the other networks as students via distillation of the LM classifier logits layer (with a Mean Absolute Error loss function).

Referring to TAB2 , the 3-Layer and 8-Layer variants of the proposed BlendCNN architecture achieve the top scores across all studied datasets.

However, the performance of the proposed architecture is lower without the 'guidance' of the teacher teacher logits during training, implying the marked improvement is due to distillation.

The additional results given for BlendCNN quantifies the advantage of adding unlabelled data into the distillation phase of the student model training.

Notably from TAB1 , the 3-Layer BlendCNN student has 39?? fewer parameters and performs inference 300?? faster than the OpenAI Transformer which it empirically out-scores.

For text classifications, mastery may require both high-level concepts gleaned from language under standing and fine-grained textual features such as key phrases.

Similar to the larval-adult form analogy made in BID10 , high-capacity models with task-agnostic pre-training may be well-suited for task mastery on small datasets (which are common in industry).

On the other hand, convolutional student architectures may be more ideal for practical applications by taking advantage of massively parallel computation and a significantly reduced memory footprint.

Our results suggest that the proposed BlendCNN architecture can efficiently achieve higher scores on text classification tasks due to the direct leveraging of hierarchical representations, which are learnable (even in a label-sparse setting) from a strong teaching model.

Further development of specialized student architectures could similarly surpass teacher performance if appropriately designed to leverage the knowledge gained from a pretrained, task-agnostic teacher model whilst optimizing for task-specific constraints.

@highlight

We train a small, efficient CNN with the same performance as the OpenAI Transformer on text classification tasks