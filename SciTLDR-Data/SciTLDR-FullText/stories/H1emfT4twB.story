In this paper, we explore meta-learning for few-shot text classification.

Meta-learning has shown strong performance in computer vision, where low-level patterns are transferable across learning tasks.

However, directly applying this approach to text is challenging–lexical features highly informative for one task maybe insignificant for another.

Thus, rather than learning solely from words, our model also leverages their distributional signatures, which encode pertinent word occurrence patterns.

Our model is trained within a meta-learning framework to map these signatures into attention scores, which are then used to weight the lexical representations of words.

We demonstrate that our model consistently outperforms prototypical networks learned on lexical knowledge (Snell et al., 2017) in both few-shot text classification and relation classification by a significant margin across six benchmark datasets (19.96% on average in 1-shot classification).

@highlight

Meta-learning methods used for vision, directly applied to NLP, perform worse than nearest neighbors on new classes; we can do better with distributional signatures.