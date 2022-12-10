Learning effective text representations is a key foundation for numerous machine learning and NLP applications.

While the celebrated Word2Vec technique yields semantically rich word representations, it is less clear whether sentence or document representations should be built upon word representations or from scratch.

Recent work has demonstrated that a distance measure between documents called \emph{Word Mover's Distance} (WMD) that aligns semantically similar words, yields unprecedented KNN classification accuracy.

However, WMD is very expensive to compute, and is harder to apply beyond simple KNN than feature embeddings.

In this paper, we propose the \emph{Word Mover's Embedding } (WME), a novel approach to building an unsupervised document (sentence) embedding from pre-trained word embeddings.

Our technique extends the theory of \emph{Random Features} to show convergence of the inner product between WMEs to a positive-definite kernel that can be interpreted as a soft version of (inverse) WMD.

The proposed embedding is more efficient and flexible than WMD in many situations.

As an example, WME with a simple linear classifier reduces the computational cost of WMD-based KNN \emph{from cubic to linear} in document length and \emph{from quadratic to linear} in number of samples, while simultaneously improving accuracy.

In experiments on 9 benchmark text classification datasets and 22 textual similarity tasks the proposed technique consistently matches or outperforms state-of-the-art techniques, with significantly higher accuracy on problems of short length.

<|TLDR|>

@highlight

A novel approach to building an unsupervised document (sentence) embeddings from pre-trainedword embeddings