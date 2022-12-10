Multi-hop question answering requires models to gather information from different parts of a text to answer a question.

Most current approaches learn to address this task in an end-to-end way with neural networks, without maintaining an explicit representation of the reasoning process.

We propose a method to extract a discrete reasoning chain over the text, which consists of a series of sentences leading to the answer.

We then feed the extracted chains to a BERT-based QA model to do final answer prediction.

Critically, we do not rely on gold annotated chains or ``supporting facts:'' at training time, we derive pseudogold reasoning chains using heuristics based on named entity recognition and coreference resolution.

Nor do we rely on these annotations at test time, as our model learns to extract chains from raw text alone.

We test our approach on two recently proposed large multi-hop question answering datasets: WikiHop and HotpotQA, and achieve state-of-art performance on WikiHop and strong performance on HotpotQA.

Our analysis shows the properties of chains that are crucial for high performance: in particular, modeling extraction sequentially is important, as is dealing with each candidate sentence in a context-aware way.

Furthermore, human evaluation shows that our extracted chains allow humans to give answers with high confidence, indicating that these are a strong intermediate abstraction for this task.

@highlight

We improve answering of questions that require multi-hop reasoning extracting an intermediate chain of sentences.