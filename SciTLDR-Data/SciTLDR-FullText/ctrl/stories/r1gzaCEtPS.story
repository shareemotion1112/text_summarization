Many tasks in natural language processing and related domains require high precision output that obeys dataset-specific constraints.

This level of fine-grained control can be difficult to obtain in large-scale neural network models.

In this work, we propose a structured latent-variable approach that adds discrete control states within a standard autoregressive neural paradigm.

Under this formulation, we can include a range of rich, posterior constraints to enforce task-specific knowledge that is effectively trained into the neural model.

This approach allows us to provide arbitrary grounding of internal model decisions, without sacrificing any representational power of neural models.

Experiments consider applications of this approach for text generation and part-of-speech induction.

For natural language generation, we find that this method improves over standard benchmarks, while also providing fine-grained control.

<|TLDR|>

@highlight

A structured latent-variable approach that adds discrete control states within a standard autoregressive neural paradigm to provide arbitrary grounding of internal model decisions, without sacrificing any representational power of neural models.