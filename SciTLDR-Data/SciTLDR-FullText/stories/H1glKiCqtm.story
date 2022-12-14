Word embeddings are widely used in machine learning based natural language processing systems.

It is common to use pre-trained word embeddings which provide benefits such as reduced training time and improved overall performance.

There has been a recent interest in applying natural language processing techniques to programming languages.

However, none of this recent work uses pre-trained embeddings on code tokens.

Using extreme summarization as the downstream task, we show that using pre-trained embeddings on code tokens provides the same benefits as it does to natural languages, achieving: over 1.9x speedup, 5\% improvement in test loss, 4\% improvement in F1 scores, and resistance to over-fitting.

We also show that the choice of language used for the embeddings does not have to match that of the task to achieve these benefits and that even embeddings pre-trained on human languages provide these benefits to programming languages.

One of the initial steps in a machine learning natural language processing (NLP) pipeline is converting the one-hot encoded R V tokens into dense R D representations, with V being the size of the vocabulary, D the embedding dimensions and V << D. This conversion is usually done with a single layer neural network, commonly called an embedding layer.

The parameters of the embedding layer can either be initialized randomly or initialized via "pretrained" parameters obtained from a model such as word2vec BID22 a) , GloVe BID25 or a language model BID17 BID11 BID26 .It is common to use pre-trained parameters (most frequently the GloVe embeddings), which act as a form of transfer learning BID23 similar to that of using pre-trained parameters for the convolutional kernels in a machine learning computer vision task BID12 BID14 .

These parameters in the embedding layer are then fine-tuned whilst training on the desired downstream task.

The use of these pre-trained embeddings over random initialization allows machine learning models to: train faster, achieve improved overall performance BID13 , increase the stability of their training, and reduce the amount of over-fitting BID23 .Recently there has been an increased interest in applying NLP techniques to programming languages and software engineering applications BID30 BID3 , the most common of which involves predicting the names of methods or variables using surrounding source code BID27 BID0 BID4 BID6 a) .Remarkably, none of this work takes advantage of pre-trained embeddings created on source code.

From the example below in table 1, we can see how semantic knowledge (provided by the pre-trained code embeddings) of the method body would help us predict the method name, i.e. knowing how pi and radius are used to calculate an area and how height and width are used to calculate an aspect ratio.float getSurfaceArea (int radius) { return 4 * Math.

PI * radius * radius; } float getAspectRatio (int height, int width) { return height / width; } Table 1 : Examples showing how the semantics of the variable names within a method can be used to reason about the name of the method body This semantic knowledge is available to us as even though computers do not need to understand the semantic meaning of a method or variable name, they are mainly chosen to be understood by other human programmers BID10 .In this paper, we detail experiments using pre-trained code embeddings on the downstream task of predicting a method name from a method body.

This task is known as extreme summarization BID2 as a method name can be thought of as a summary of the method body.

Our experiments are focused on answering the following research questions:1.

Do pre-trained code embeddings reduce training time?2.

Do pre-trained code embeddings improve performance?3.

Do pre-trained code embeddings increase stability of training?4.

Do pre-trained code embeddings reduce over-fitting?5.

How does the choice of corpora used for the pre-trained code embeddings affect all of the above?To answer RQ5, we gather a corpus of C, Java and Python code and train embeddings for each corpus separately, as well as comparing them with embeddings trained on natural language.

We then test each of these on the same downstream task, extreme summarization, which is in Java.

We also release the pre-trained code embeddings.

As far as we are aware, this is the first study on the effectiveness of pre-trained code embeddings applied to an extreme (code) summarization task.

We train our embeddings using a language model (LM).

We choose a LM over the word2vec or GloVe models as LMs have shown to capture long-term dependences BID15 and hierarchical relations BID9 .

We believe both of these properties are essential for predicting a method name from method body.

The long-term dependencies are required due to the average length of the method body over 72 tokens 2 .

The hierarchical relations are needed due to the way data flows through variables within the method body, starting from the method argument(s) (at the top of the hierarchy) to the return value(s) (at the bottom of the hierarchy).A language model is a probability distribution over sequences of tokens.

Each token, x, is represented by a one-hot vector x ??? R V , with V being the size of the vocabulary.

The probability given to a sequence of tokens x 1 , ..., x T can be calculated as: DISPLAYFORM0 p(x t |x t???1 , ..., x 1 ) 1 Code and embeddings to be released at a later date.

2 We use a token to refer to an atomic part of a sequence of code.

We model this probability distribution with a recurrent neural network trained to predict the next token in a sequence of given tokens.

Specifically, we use the AWD-LSTM-LM model BID19 BID1 .

Briefly, the model takes a series of code tokens from the method body, c, as input and outputs the code tokens that form the method name, m. It generates the method name one token at a time, using a recurrent hidden state, h t , provided by a Gated Recurrent Unit (GRU) ) and a series of convolutional filters over the embeddings of the tokens c, which produce attention BID7 features, L f eat .

It also has a mechanism to directly copy tokens from the body to the output.

This model was chosen as it is the state-of-the-art on the extreme summarization dataset used and provided a clear improvement over baseline models.

It also has an open source implementation.

The dataset used for the pre-trained code embeddings was gathered from GitHub.

To ensure the quality of the data we only used projects with over 10,000 stars and manually checked each project's suitability, i.e. did not use projects which were tutorials or guides.

After scraping the appropriate projects for each of the three languages (C, Java and Python) we tokenized each, converting each token to lowercase as well as splitting each token into subtokens on camelCase and snake case, e.g. getSurfaceArea becomes get, surface and area.

This was done to match the tokenization of the extreme summarization dataset.

There are approximately 100 million tokens across 20 million lines of code for each language.

Each of the embeddings has their own distinct vocabulary, e.g. not all tokens that appear in the C corpus appear in the Java corpus.

For the natural language embeddings we used the WikiText-103 dataset BID18 , as it contains a comparable 103 million tokens.

The AWD-LSTM-LM model was trained with all default parameters from the open source implementation, with the exception of: the embedding dimension changed to 128 and the hidden dimension changed to 512.

The embedding dimension was changed to match that of the original Copy Convolutional Attention Model, and the hidden dimension was changed to fit in GPU memory.

Tokens that were not in the most 150,000 common or did not appear at least 3 times were converted into an <unk> token and the model was trained until the validation loss did not decrease for 5 epochs.

The extreme summarization task dataset is detailed in BID1 .

Briefly, it consists of 10 Java projects selected for their quality and diversity in application.

For each project, all full Java methods are extracted with the method body used as the input and the method name used as the target.

Each project has their own vocabulary, e.g. tokens that appear in one project may not appear in any others.

All tokens are formatted the same as the embedding dataset to ensure maximum vocabulary overlap between each pre-trained embeddings and each Java project.

The Copy Convolutional Attention Model was trained with all default parameters from the open source implementation, and was trained for 25 epochs.

The model was trained on each project separately and was run 5 times on each project for each of the embeddings.

The results were then averaged together for each project.

Table 2 shows the rank 1 F1 scores achieved for each of the embeddings.

On average, we achieve a 4% relative improvement in F1 scores for each of the embeddings.

Table 2 : Rank 1 F1 scores for each of the embeddings.

Figure 1 shows the validation losses achieved on all 10 Java projects.

Randomly initialized embeddings are shown in purple, Java embeddings in green, C in blue, Python in red and English in orange.

It can be seen that for most projects the pre-trained embeddings train faster, achieve lower losses, are more stable and over-fit less.

Table 3 shows overlap, speedup and improvement in test loss for each project-embedding combination compared to random embeddings.

Overlap is the percentage of tokens in the project vocabulary that also appear in the embedding vocabulary.

Speedup is calculated as:

N r is the number of epochs taken by the random embedding to reach its best validation loss and N e is the number of epochs taken by a non-random embedding to reach that same validation loss.

L r is the test loss achieved using random embeddings and L e is the test loss achieved using a nonrandom embedding.

Table 3 : Results relative to random embeddings for each project-embedding combination.

Overlap is the percentage of tokens within the embedding vocabulary that also appear in the project.

Speedup is relative speedup of convergence compared to random embeddings.

Improvement is relative improvement in test loss compared to random embeddings.

We notice that some projects, particularly the elasticsearch project (figure 1b), did not achieve any benefits from using the pre-trained embeddings and in fact experienced a slow down when using the C embeddings.

To explore this further, we plotted speedup and improvement against overlap in figure 2.

We measure the Pearson correlation coefficient of each, receiving a coefficient of 0.73 for speedup and 0.73 for improvement, a medium to strong positive correlation for each.

This implies that using more of the pre-trained embeddings provides more of both the speedup and improvement in test loss benefits.

TAB3 : Results relative to random embeddings for each of the pre-trained embeddings, averaged across all 10 projects.

Intuitively, it would make sense that the Java embeddings give the best results as the summarization task is also in Java.

We see this is not the case, and the Java embeddings have the same speedup and improvement as the Python embeddings and only a small speedup improvement over the C embeddings, even though the average overlap using the Java embeddings is higher.

Most interesting is the fact that the English embeddings achieve the comparable speedup and performance improvement, even though they have only been trained on human languages.

One potential reason for the similar performance between the embeddings that are trained on programming languages is that even though C, Java and Python are syntactically different, the extreme summarization task does not require much of this syntactic information.

Consider the examples in table 5, which are Python versions of the Java examples in table 1.

Although the syntax has changed (dynamic typing, no braces or semicolons, etc.) the available semantic information from the method body has not.

This would imply that the language of the dataset used to pre-train code embeddings does not matter as much as the quality of the dataset with regards to sensible method and variable names.

This is further solidified in the fact that the English embeddings, which have only been trained on human languages, achieved similar performance compared to programming languages.def get surface area (radius): return 4 * math.pi * radius * radius def get aspect ratio (height, width): return height / width We also look at the amount of over-fitting on each project.

From figures 1c, 1d, 1f, 1g and 1i we can see how the random embeddings show a large amount of over-fitting compared to the pre-trained code embeddings.

We measure how much a project over-fits as: DISPLAYFORM0 is the best validation loss achieved and L f is the final validation loss achieved.

We dub this term the over-fit factor, where an O = 1 would imply the final loss is equal to the lowest loss and thus has not over-fit at all (this could also mean the model is still converging, however from figure 1 we can see all project-embedding combinations converge before 25 epochs).

TAB6 shows the over-fit factors for each project-embedding combination.

We can see that the random embeddings show the worst performance on every project.

Interestingly, there appears to be no correlation between the overlap and the amount of over-fitting.

Table 7 shows the over-fit factor averaged across all projects for each embedding.

Again, the results for each of the pre-trained embeddings are similar, showing that the language of the dataset used to train the embeddings does not have a significant impact on performance.

Table 7 : Over-fit factor for each embedding averaged across all 10 projects.

Higher is better.

Language models have been used as a form of transfer learning in natural language processing applications with great success BID17 BID11 BID26 .There has also been recent work on the further analysis of language models BID20 and how well they assist in transfer learning BID23 .The use of probabilistic models for source code originated from BID10 .

From that, work on language models of code began on both the token level BID24 BID29 and syntax level BID16 .Predicting variable and method names has become a common task for machine learning applications in recent years.

Initial work was on the token level BID27 BID0 but it is beginning to become more common to represent programs as graphs using their abstract syntax tree BID4 BID5 b) .

We refer back to our research questions.

Do pre-trained code embeddings reduce training time?

Yes, tables 3 and 4 show we get an average of 1.93x speedup.

This is correlated with the amount of overlap between the task vocabulary and the embedding vocabulary, shown in figure 2a.

Do pre-trained code embeddings improve performance?

Yes, tables 3 and 4 show we get an average of 5% relative validation loss improvement.

Again, this is correlated with the amount of overlap between the vocabularies, shown in figure 2b.

Do pre-trained code embeddings increase stability of training?

Although this is difficult to quantify due to how over-fitting interacts with the variance of the validation loss curves, from figures 1a and 1c we can see a clear increase in the variance of the validation loss curves using random embeddings compared to those using pre-trained embeddings.

Do pre-trained code embeddings reduce over-fitting?

Yes, tables 6 and 7 show that the random embeddings over-fit more than the pre-trained embeddings on every project.

However, this does not seem to have a correlation with the amount of vocabulary overlap and further work is needed to determine the cause of this.

How does the choice of corpora used for the pre-trained code embeddings affect all of the above?

Intuitively, it would seem the best pre-trained embeddings would be those that are trained on the same language as that of the downstream task, but this is not the case.

We hypothesize through the examples shown in tables 1 and 5 that the differing syntax between the languages is not as important as sensible semantic method and variable names within the dataset.

This semantic information is also contained in human languages, which explains why the English embeddings also receive comparable performance.

@highlight

Researchers exploring natural language processing techniques applied to source code are not using any form of pre-trained embeddings, we show that they should be.

@highlight

This paper sets to understand whether pretraining word embeddings for programming language code by using NLP-like language models has an impact on extreme code summarization task.

@highlight

This work shows how pre-training word vectors using corpuses of code leads to representations that are more suitable than randomly initialized and trained representations for function/method name prediction