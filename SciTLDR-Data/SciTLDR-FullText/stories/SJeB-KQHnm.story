Machine learning models for question-answering (QA), where given a question and a passage, the learner must select some span in the passage as an answer, are known to be brittle.

By inserting a single nuisance sentence into the passage, an adversary can fool the model into selecting the wrong span.

A promising new approach for QA decomposes the task into two stages: (i) select relevant sentences from the passage; and (ii) select a span among those sentences.

Intuitively, if the sentence selector excludes the offending sentence, then the downstream span selector will be robust.

While recent work has hinted at the potential robustness of two-stage QA, these methods have never, to our knowledge, been explicitly combined with adversarial training.

This paper offers a thorough empirical investigation of adversarial robustness, demonstrating that although the two-stage approach lags behind single-stage span selection, adversarial training improves its performance significantly, leading to an improvement of over 22 points in F1 score over the adversarially-trained single-stage model.

the training and development datasets.

These demonstrations underscore the necessity for evaluating Only a few subsequent papers have followed up on [7] , proposing solutions to make QA models more 35 robust to such adversarial attacks.

Recently, Min et al. BID8 proposed a two-stage model consisting 36 of both a sentence selector and a span selector.

They showed that providing a minimal context, 37 consisting of just few relevant sentences to the span selector, offers benefits not only in terms of 38 interpretability (by identifying the relevant pieces of evidence) and computational efficiency, but also 39 results in greater robustness to the aforementioned adversarial attack.

This is a promising direction 40 towards making QA models more robust, since achieving robustness in the overall system requires 41 only that we make the context selection model robust.

So long as the context selector filters out 42 irrelevant sentences (including the adversarial sentence) the downstream model will be safe.

In this work, we investigate this two-stage approach (minimal context selection followed by span 44 selection) finding that the approach is not, out of the box, more robust than the single-stage approach tokens to learn a fixed-length question representation that is then used to score potential spans.

The Mnemonic Reinforced Reader uses several layers of co-attention between the question and the 56 context, memorizing and utilizing attention output from previous layers to compute the later ones.

Additionally, both models employ hand-crafted features like Part-of-Speech (PoS) tags, Named Entity 58 Recognition (NER) tags, and other lexical features, in order to achieve competitive performance 59 on the task.

We follow the reported architecture and hyperparameter settings exactly, referring the 60 readers to the source papers for more details.

In this paper, we focus on adversarial training through data augmentation: for every training example, DISPLAYFORM0 , where p is the paragraph, q is the question and a is the answer, we introduce adver- features for both models.

In the two-stage set-up, the top-k sentences from the sentence selector are 100 passed on to the span selection model.

We choose k = 1 for SQuAD dataset.

Results: Our results are summarized in Table 1 Table 2 : Sentence selector top-k accuracy for SQuAD (k = 1).

Several prior works BID8 12, 3] consider sentence selection as a sub-task of question answering.

[3] construct document summaries using reinforcement learning, feeding these summaries to the

@highlight

A two-stage approach consisting of sentence selection followed by span selection can be made more robust to adversarial attacks in comparison to a single-stage model trained on full context.

@highlight

This paper investigates an existing model and finds that a two-stage trained QA method is not more robust to adversarial attacks compared to other methods.