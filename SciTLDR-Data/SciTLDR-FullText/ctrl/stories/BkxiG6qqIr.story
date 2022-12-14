Plagiarism and text reuse become more available with the Internet development.

Therefore it is important to check scientific papers for the fact of cheating, especially in Academia.

Existing systems of plagiarism detection show the good performance and have a huge source databases.

Thus now it is not enough just to copy the text as is from the source document to get the original work.

Therefore, another type of plagiarism become popular -- cross-lingual plagiarism.

We present a CrossLang system for such kind of plagiarism detection for English-Russian language pair.

The key idea for CrossLang 1 system is that we use the monolingual approach.

We have suspicious Russian document and English reference collection.

We reduce the task to the one language -we translate the suspicious document into English, because the reference collection is in English.

After this step we perform the subsequent document analysis.

Due to this fact the main challenge with the CrossLang design is that the algorithms should be stable to the translation ambiguity.

The main stages of CrossLang service is depicted in Figure 1 .

CrossLang receives the suspicious document from Antiplagiat system, when user send it for originality checking.

Then it goes to Entry pointmain service, that routes the data between following stages:

1.

Machine Translation system -microservice, that translates suspicious document into English.

For these purposes we use Transformer Vaswani et al., open-source neural machine translation framework.

2.

Source retrieval -this stage unites two microservices: Shingle index and Document storage.

Entry point receives the translated suspicious document's shingles (n -grams) and Shingle index returns to it the documents ids from the reference English collection.

To deal with the translation ambiguity we use modified shingle-based approach.

Document storage returns the Source texts from the collection by these ids.

3.

Document comparison -this microservice performs the comparison between translated suspicious document and source documents.

We compare not the texts themselves, but the vectors corresponding to the phrases of these texts.

Thus we deal with the translation ambiguity problem.

We create machine translation system using state-of-the-art The CrossLang BLEU score lower than Google's BLEU score -this was to be expected.

But it is very important to notice that we are not interested in ideal translation.

Our main goal is to translate with sufficient quality for the next stages: Source retrieval and Document comparison.

The method of source retrieval in the case of verbatim plagiarism is inverted index construction,where a document from the reference collection is represented as a set of its shingles, i.e. overlapping word n -grams, and a suspicious document's shingles are checked for matches with the indexed documents.

There is one major problem with using the standard shingles -in our case the machine translation stage generates texts that differ too much from the sources of plagiarism.

We argue that the source retrieval task can be solved with the help of a similar method that performs better than the method mentioned above; this improvement is achieved by moving from word shingles to word-class shingles, where each word is substituted by the label of the class it belongs to: {word 1 , . . .

, word n } ??? {class(word 1 ), . . .

, class(word n )}.

Clustering the word vectors is a convenient and relatively fast way of obtaining semantic word classes.

For the word embedding model we used fastText Bojanowski et al. [2016] trained on English Wikipedia.

The dimension for word embedding model was set to 100.

For the semantic word classes construction we applied agglomerative clustering on word embeddings with the cosine similarity measure to group words into word classes.

We got 777K words clustered into 30K classes.

For the comparison between retrieved documents and translated suspicious documents we introduce the phrase embedding model.

We split documents (retrieved and suspicious) into phrases s and compare its vectors.

For mapping the word sequence into low dimensional space we use the encoderdecoder scheme with L-2 reconstruction error minimization E rec = s ????? 2 .

Encoder-decoder model is completely unsupervised and does not use any information whether the phrase pair is paraphrased or not.

We train Seq2Seq model with attention Bahdanau et al. [2014] on 10M sentences from Wikipedia.

In order to use information about phrase similarity we extend the objective function.

We employ the margin-base loss from Wieting et al. [2015] with the limited number of similar phrase pairs S = {(s i , s j )}:

where

The sampling of so named "false neighbour" s i during training helps to improve the final quality without strict limitations on what phrases we should use at dissimilar.

This part of objective requires a dataset of similar sentences S = {(s i , s j )}.

We used double translation method as a method of similar sentences generation comparable to paraphrase.

The final objective function is:

where ?? is a tunable hyperparameter that weights both of errors.

6 .

For each phrase embedding from the suspicious document find nearest vectors by cosine similarity from source documents using Annoy 7 library.

1.

The best of our knowledge it is the first system for cross-lingual plagiarism detection for English-Russian language pair.

It is deployed on production and we could analyze the results.

We could not find another examples of such system (even for other language pairs).

2.

The Source retrieval 1.2 stage is often employed using rather simple heuristical algorithms such as shingle-based search or keyword extraction.

However, these methods can significantly suffer from word replacements and usually detect only near-duplicate paraphrase.

We present modified method, see 1.2.

3.

Many articles on the cross-lingual plagiarism detection topic investigate the solutions based on bilingual or monolingual word embeddings Ferrero et al. [2017] for documents comparison, but almost none of them uses the phrase embeddings for this problem solution.

We present phrase embeddings comparison in 1.3.

There are no results and datasets for cross-lingual plagiarism detection task for language pair EnglishRussian.

We create dataset for the problem and make it available.

Visit 8 for dataset download and details about generation.

For the whole framework we got Precision = 0.83, Recall = 0.79 and F 1 = 0.80.

Since our system translates the suspicious document into the language of the collection it's natural to analyze the performance of our system for monolingual problem.

For such experiment we do not use the machine translation service.

In order to check performance of monolingual paraphrased plagiarism detection we exploit PAN'11 contest dataset and quality metrics Potthast et al..

Results of CrossLang and top-3 known previous methods are in Table 2 .

Our service is deployable on an 8-GPU cluster with Tesla-K100 GPUs, 128GB RAM and 64 CPU Cores.

Depending on the requirements, the service is able to scale horizontally.

For the fast rescaling we use Docker containerization and Consul and Consul-template for the service discovery and automatic load balancing.

The stress testing of our system showed that the system is able to check up to 100 documents in a minute.

Despite the fact the average loading on our service is much lower, this characteristic of our service is important for withstanding peak loads.

We introduced CrossLang -a framework for cross-lingual plagiarism detection for English Russian language pair.

We decomposed the problem of cross-lingual plagiarism detection into several stages and provide a service, consists of a set of microservices.

The CrossLang use a monolingual approachreducing the problem to the one language.

For this purpose we trained the neural machine translation system.

Another two main algoithmic components are Source Retrieval and Document Comparison stages.

For the Source Retrieval problem we used a modification of shingling method that allow us to deal with ambiguity after translation.

For the Document Comparison stage we used phrase embeddings that were trained with slight supervision.

We evaluated the effectiveness of main stages.

<|TLDR|>

@highlight

A system for cross-lingual (English-Russian) plagiarism detection