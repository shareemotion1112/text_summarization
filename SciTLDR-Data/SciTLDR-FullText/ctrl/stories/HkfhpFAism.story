Existing neural question answering (QA) models are required to reason over and draw complicated inferences from a long context for most large-scale QA datasets.

However, if we view QA as a combined retrieval and reasoning task, we can assume the existence of a minimal context which is necessary and sufficient to answer a given question.

Recent work has shown that a sentence selector module that selects a shorter context and feeds it to the downstream QA model achieves performance comparable to a QA model trained on full context, while also being more interpretable.

Recent work has also shown that most state-of-the-art QA models break when adversarially generated sentences are appended to the context.

While humans are immune to such distractor sentences, QA models get easily misled into selecting answers from these sentences.

We hypothesize that the sentence selector module can filter out extraneous context, thereby allowing the downstream QA model to focus and reason over the parts of the context that are relevant to the question.

In this paper, we show that the sentence selector itself is susceptible to adversarial inputs.

However, we demonstrate that a pipeline consisting of a sentence selector module followed by the QA model can be made more robust to adversarial attacks in comparison to a QA model trained on full context.

Thus, we provide evidence towards a modular approach for question answering that is more robust and interpretable.

<|TLDR|>

@highlight

A modular approach consisting of a sentence selector module followed by the QA model can be made more robust to adversarial attacks in comparison to a QA model trained on full context.