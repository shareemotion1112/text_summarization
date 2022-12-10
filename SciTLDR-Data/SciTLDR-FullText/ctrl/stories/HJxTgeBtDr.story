With the proliferation of models for natural language processing (NLP) tasks, it is even harder to understand the differences between models and their relative merits.

Simply looking at differences between holistic metrics such as accuracy, BLEU, or F1 do not tell us \emph{why} or \emph{how} a particular method is better and how dataset biases influence the choices of model design.

In this paper, we present a general methodology for {\emph{interpretable}} evaluation of NLP systems and choose the task of named entity recognition (NER) as a case study, which is a core task of identifying people, places, or organizations in text.

The proposed evaluation method enables us to interpret the \textit{model biases}, \textit{dataset biases}, and how the \emph{differences in the datasets} affect the design of the models, identifying the strengths and weaknesses of current approaches.

By making our {analysis} tool available, we make it easy for future researchers to run similar analyses and drive the progress in this area.

<|TLDR|>

@highlight

We propose a generalized evaluation methodology to interpret model biases, dataset biases, and their correlation.